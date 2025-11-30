from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import chz
import numpy as np
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.memory_rl.task_utils import (
    format_secret_bits,
    num_bits_for_space,
    parse_bit_guess,
    validate_secret,
)
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree


class MultiStepEnv(Env):
    """
    Environment where the agent must memorize a latent integer by outputting
    its bitstring one bit at a time.

    - A secret z is sampled in [0, N).
    - Let k = ceil(log2 N) and B be the k-bit binary representation of z.
    - At timestep i in [0, k-1], the agent outputs a single bit (0/1).
    - Reward r_i = 1 if the bit matches B[i], else 0.
    - Episode ends after k steps.
    """

    def __init__(
        self,
        *,
        fixed_secret: int,
        N: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        validate_secret(fixed_secret, N)
        self.fixed_secret = fixed_secret
        self.N = N
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []

        self.secret_bits = format_secret_bits(fixed_secret, N)
        self.num_bits = num_bits_for_space(N)

        # Position of the next bit to guess (0-based)
        self.position = 0
        # Conversation history: alternating user/assistant messages
        self.conversation: list[renderers.Message] = []

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _build_system_prompt(self) -> str:
        """Build the system prompt explaining the task."""
        return (
            f"You are playing a bit-by-bit memorization game. "
            f"The secret number is between 0 and {self.N - 1} and uses {self.num_bits} bits. "
            f"Output exactly one bit (0 or 1) per response with no punctuation."
        )

    def _build_bit_request(self, position: int) -> str:
        """Build a simple request for the next bit."""
        remaining = self.num_bits - position - 1
        return f"Provide bit #{position + 1}. Remaining bits after this: {remaining}."

    def _messages(self) -> list[renderers.Message]:
        """Return the full conversation history."""
        system_msg = {"role": "system", "content": self._build_system_prompt()}
        return [system_msg] + self.convo_prefix + self.conversation

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Add the first user message asking for bit #1
        first_request = {"role": "user", "content": self._build_bit_request(0)}
        self.conversation.append(first_request)
        return self.renderer.build_generation_prompt(self._messages()), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """
        Consume one bit from the policy and move to the next position.
        """
        # Parse the agent's response
        message, parse_success = self.renderer.parse_response(action)
        content = message["content"]
        guessed_bit = parse_bit_guess(content)

        # Add the agent's response to conversation history
        self.conversation.append({"role": "assistant", "content": content})

        # Compute reward
        correct_bit = self.secret_bits[self.position]
        correct = guessed_bit == correct_bit
        reward = 1.0 if correct else 0.0

        # Move to next position
        self.position += 1
        episode_done = self.position >= self.num_bits

        # Add next user request if not done
        if not episode_done:
            next_request = {"role": "user", "content": self._build_bit_request(self.position)}
            self.conversation.append(next_request)

        # Logging for qualitative inspection
        logtree.log_text(
            "[MultiStepEnv] "
            f"pos={self.position - 1}/{self.num_bits - 1}, "
            f"secret_bits={self.secret_bits}, "
            f"guess_raw={content!r}, "
            f"guess_bit={guessed_bit}, "
            f"correct_bit={correct_bit}, "
            f"reward={reward:.2f}"
        )

        if episode_done:
            next_observation = tinker.ModelInput.empty()
        else:
            next_observation = self.renderer.build_generation_prompt(self._messages())

        return StepResult(
            reward=reward,
            episode_done=episode_done,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics={
                "correct_bit": float(correct),
                "reward_signal": reward,
                "bit_index": float(self.position - 1),
                "episode_done": float(episode_done),
            },
        )


@dataclass(frozen=True)
class MultiStepEnvGroupBuilder(EnvGroupBuilder):
    """Builder for creating groups of identical multi-step envs."""

    fixed_secret: int
    N: int
    renderer: renderers.Renderer
    num_envs: int
    convo_prefix: list[renderers.Message] | None = None

    async def make_envs(self) -> Sequence[Env]:
        return [
            MultiStepEnv(
                fixed_secret=self.fixed_secret,
                N=self.N,
                renderer=self.renderer,
                convo_prefix=self.convo_prefix,
            )
            for _ in range(self.num_envs)
        ]

    def logging_tags(self) -> list[str]:
        k_bits = num_bits_for_space(self.N)
        return [f"multi_step_N{self.N}_k{k_bits}"]


class MultiStepDataset(RLDataset):
    """
    Dataset of multi-step memorization problems.

    Training split:
        - Uses a single fixed secret (provided or sampled once).
        - All episodes share the same secret so the policy can memorize it.

    Test split:
        - By default uses the same secret as the training split, repeated.
        - Alternatively can use a list of random secrets (if desired later).
    """

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        N: int,
        n_batches: int,
        convo_prefix: list[renderers.Message] | None,
        fixed_secret: int | None,
        seed: int,
        split: Literal["train", "test"] = "train",
    ):
        self._rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.N = N
        self.convo_prefix = convo_prefix
        self.n_batches = n_batches
        self.split = split

        if fixed_secret is not None:
            validate_secret(fixed_secret, N)
            self.fixed_secret = int(fixed_secret)
        elif split == "train":
            # Sample a single secret for this run and reuse it across episodes.
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
                MultiStepEnvGroupBuilder(
                    fixed_secret=self.fixed_secret,
                    N=self.N,
                    renderer=self.renderer,
                    num_envs=self.group_size,
                    convo_prefix=self.convo_prefix,
                )
            ]

        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.test_secrets))
        return [
            MultiStepEnvGroupBuilder(
                fixed_secret=secret,
                N=self.N,
                renderer=self.renderer,
                num_envs=self.group_size,
                convo_prefix=self.convo_prefix,
            )
            for secret in self.test_secrets[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class MultiStepDatasetBuilder(RLDatasetBuilder):
    """Builder for creating train/test datasets for the multi-step bitstring env."""

    batch_size: int
    group_size: int = 1
    model_name_for_tokenizer: str
    renderer_name: str
    N: int = 16
    n_batches: int = 100
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = None
    fixed_secret: int | None = None
    seed: int = 1337

    async def __call__(self) -> tuple[MultiStepDataset, MultiStepDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        if self.group_size < 1:
            raise ValueError("group_size must be >= 1.")

        if self.convo_prefix == "standard":
            # Simple few-shot prefix demonstrating the bit-by-bit behavior.
            num_bits = num_bits_for_space(self.N)
            prefix: list[renderers.Message] | None = [
                {"role": "user", "content": f"Provide bit #1. Remaining bits after this: {num_bits - 1}."},
                {"role": "assistant", "content": "0"},
            ]
        else:
            prefix = self.convo_prefix

        train_dataset = MultiStepDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            N=self.N,
            n_batches=self.n_batches,
            convo_prefix=prefix,
            fixed_secret=self.fixed_secret,
            seed=self.seed,
            split="train",
        )

        test_dataset = MultiStepDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            N=self.N,
            n_batches=max(10, self.n_batches // 10),
            convo_prefix=prefix,
            fixed_secret=train_dataset.fixed_secret,
            seed=self.seed,
            split="test",
        )

        return train_dataset, test_dataset
