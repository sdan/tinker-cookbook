from dataclasses import dataclass
from typing import Literal, Sequence

import chz
import numpy as np
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.memory_rl.task_utils import (
    RewardType,
    build_single_step_prompt,
    compute_single_step_reward,
    parse_single_step_guess,
    validate_secret,
)
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree


class SingleStepEnv(Env):
    def __init__(
        self,
        fixed_secret: int,
        N: int,
        renderer: renderers.Renderer,
        reward_type: RewardType,
        reward_bins: int,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        validate_secret(fixed_secret, N)
        self.fixed_secret = fixed_secret
        self.N = N
        self.renderer = renderer
        self.reward_type = reward_type
        self.reward_bins = reward_bins
        self.convo_prefix = convo_prefix or []

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _messages(self) -> list[renderers.Message]:
        prompt = build_single_step_prompt(self.N)
        return self.convo_prefix + [{"role": "user", "content": prompt}]

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return self.renderer.build_generation_prompt(self._messages()), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        guess = parse_single_step_guess(message["content"], self.N)
        reward, correct_bool, distance = compute_single_step_reward(
            secret=self.fixed_secret,
            guess=guess,
            reward_type=self.reward_type,
            N=self.N,
            reward_bins=self.reward_bins,
        )
        correct = float(correct_bool)
        guess_display = "?" if guess is None else str(guess)

        logtree.log_text(f"[SingleStepEnv] Prompt: {build_single_step_prompt(self.N)}")
        logtree.log_text(f"[SingleStepEnv] Response: {message['content']}")
        logtree.log_text(
            f"[SingleStepEnv] Secret: {self.fixed_secret}, Guess: {guess_display}, Reward: {reward:.2f}"
        )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "correct": correct,
                "distance": distance,
                "reward_signal": reward,
            },
        )


@dataclass(frozen=True)
class SingleStepEnvGroupBuilder(EnvGroupBuilder):
    """Builder for creating groups of identical single-step envs."""

    fixed_secret: int
    N: int
    renderer: renderers.Renderer
    reward_type: RewardType
    reward_bins: int
    num_envs: int
    convo_prefix: list[renderers.Message] | None = None

    async def make_envs(self) -> Sequence[Env]:
        return [
            SingleStepEnv(
                fixed_secret=self.fixed_secret,
                N=self.N,
                renderer=self.renderer,
                reward_type=self.reward_type,
                reward_bins=self.reward_bins,
                convo_prefix=self.convo_prefix,
            )
            for _ in range(self.num_envs)
        ]

    def logging_tags(self) -> list[str]:
        suffix = f"_B{self.reward_bins}" if self.reward_type.startswith("binned") else ""
        return [f"single_step_N{self.N}_{self.reward_type}{suffix}"]


class SingleStepDataset(RLDataset):
    """Dataset of single-step memorization problems."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        N: int,
        reward_type: RewardType,
        reward_bins: int,
        convo_prefix: list[renderers.Message] | None,
        n_batches: int,
        fixed_secret: int | None,
        seed: int,
        split: Literal["train", "test"] = "train",
    ):
        self._rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.N = N
        self.reward_type = reward_type
        self.reward_bins = reward_bins
        self.convo_prefix = convo_prefix
        self.n_batches = n_batches
        self.split = split

        if fixed_secret is not None:
            validate_secret(fixed_secret, N)
            self.fixed_secret = int(fixed_secret)
        elif split == "train":
            self.fixed_secret = int(self._rng.randint(0, N))
        else:
            self.fixed_secret = None

        if split == "test":
            if self.fixed_secret is not None:
                total = n_batches * batch_size
                self.test_secrets = [self.fixed_secret] * total
            else:
                test_rng = np.random.RandomState(seed + 1)
                self.test_secrets = test_rng.randint(0, N, size=n_batches * batch_size).tolist()

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if self.split == "train":
            if self.fixed_secret is None:
                raise RuntimeError("Training split requires a fixed secret")
            return [
                SingleStepEnvGroupBuilder(
                    fixed_secret=self.fixed_secret,
                    N=self.N,
                    renderer=self.renderer,
                    reward_type=self.reward_type,
                    reward_bins=self.reward_bins,
                    num_envs=self.group_size,
                    convo_prefix=self.convo_prefix,
                )
                for _ in range(self.batch_size)
            ]

        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.test_secrets))
        return [
            SingleStepEnvGroupBuilder(
                fixed_secret=secret,
                N=self.N,
                renderer=self.renderer,
                reward_type=self.reward_type,
                reward_bins=self.reward_bins,
                num_envs=self.group_size,
                convo_prefix=self.convo_prefix,
            )
            for secret in self.test_secrets[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class SingleStepDatasetBuilder(RLDatasetBuilder):
    """Builder for creating train/test datasets."""

    batch_size: int
    group_size: int = 1
    model_name_for_tokenizer: str
    renderer_name: str
    N: int = 16
    reward_type: RewardType = "binary"
    reward_bins: int = 8
    n_batches: int = 100
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = None
    fixed_secret: int | None = None
    seed: int = 1337

    async def __call__(self) -> tuple[SingleStepDataset, SingleStepDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        if self.reward_type in {"binned_distance", "binned_distance_signed"} and self.reward_bins < 2:
            raise ValueError("reward_bins must be >= 2 when using a binned reward type.")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1.")

        if self.convo_prefix == "standard":
            prefix = [
                {"role": "user", "content": build_single_step_prompt(self.N)},
                {"role": "assistant", "content": "0"},
            ]
        else:
            prefix = self.convo_prefix

        train_dataset = SingleStepDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            N=self.N,
            reward_type=self.reward_type,
            reward_bins=self.reward_bins,
            convo_prefix=prefix,
            n_batches=self.n_batches,
            fixed_secret=self.fixed_secret,
            seed=self.seed,
            split="train",
        )

        test_dataset = SingleStepDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            N=self.N,
            reward_type=self.reward_type,
            reward_bins=self.reward_bins,
            convo_prefix=prefix,
            n_batches=max(10, self.n_batches // 10),
            fixed_secret=train_dataset.fixed_secret,
            seed=self.seed,
            split="test",
        )

        return train_dataset, test_dataset
