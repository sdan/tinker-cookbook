import asyncio
from datetime import datetime
from typing import Literal

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train
from tinker_cookbook.recipes.memory_rl.multi_step_env import MultiStepDatasetBuilder
from tinker_cookbook.recipes.memory_rl.single_step_env import SingleStepDatasetBuilder


@chz.chz
class CLIConfig:
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: str | None = None
    load_checkpoint_path: str | None = "tinker://ca0553de-bbad-4175-b203-a1b6d1011fa3/weights/final"


    # Task knobs
    N: int = 16  # secret space size (log2 N bits)
    reward_type: str = "binary"  # "binary" | "binned_distance" | "binned_distance_signed"
    reward_bins: int = 8  # Used for binned reward types (>=2)
    fixed_secret: int | None = None  # set an int in [0, N-1] to choose a specific secret
    # Use a few-shot prefix (set False to keep observation minimally informative)
    use_standard_prefix: bool = False
    env_type: Literal["single_step", "multi_step"] = "single_step"

    # Training / sampling
    batch_size: int = 1
    group_size: int = 1  # envs per group
    n_batches: int = 1000
    lora_rank: int = 1
    learning_rate: float = 4e-5
    max_tokens: int = 8
    eval_every: int = 10
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    dataset_seed: int = 0

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None


def build_config(cli: CLIConfig) -> train.Config:
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)
    
    if cli.env_type == "single_step":
        if cli.reward_type not in {"binary", "binned_distance", "binned_distance_signed"}:
            raise ValueError(
                f"Unsupported reward_type '{cli.reward_type}'. Use 'binary', 'binned_distance', or 'binned_distance_signed'."
            )
        if cli.reward_type != "binary" and cli.reward_bins < 2:
            raise ValueError("reward_bins must be >= 2 when using a binned reward.")

    prefix = "standard" if cli.use_standard_prefix else None

    if cli.env_type == "single_step":
        builder = SingleStepDatasetBuilder(
            batch_size=cli.batch_size,
            group_size=cli.group_size,
            renderer_name=renderer_name,
            model_name_for_tokenizer=cli.model_name,
            N=cli.N,
            reward_type=cli.reward_type,
            reward_bins=cli.reward_bins,
            n_batches=cli.n_batches,
            fixed_secret=cli.fixed_secret,
            convo_prefix=prefix,
            seed=cli.dataset_seed,
        )
    elif cli.env_type == "multi_step":
        builder = MultiStepDatasetBuilder(
            batch_size=cli.batch_size,
            group_size=cli.group_size,
            renderer_name=renderer_name,
            model_name_for_tokenizer=cli.model_name,
            N=cli.N,
            n_batches=cli.n_batches,
            fixed_secret=cli.fixed_secret,
            convo_prefix=prefix,
            seed=cli.dataset_seed,
        )
    else:
        raise ValueError(f"Unknown env_type {cli.env_type}")

    date_and_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    reward_suffix = f"_B{cli.reward_bins}" if cli.reward_type.startswith("binned") else ""
    default_run_name = (
        f"{cli.env_type}_{cli.reward_type}{reward_suffix}_N{cli.N}_bs{cli.batch_size}_gs{cli.group_size}_lr{cli.learning_rate}_{date_and_time}"
    )
    log_path = cli.log_path or f"/tmp/tinker-examples/memory_rl/{default_run_name}"
    wandb_name = cli.wandb_name or default_run_name

    return train.Config(
        model_name=cli.model_name,
        load_checkpoint_path=cli.load_checkpoint_path,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli.learning_rate,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        eval_every=cli.eval_every,
        loss_fn=cli.loss_fn,
        wandb_project=cli.wandb_project,
        wandb_name=wandb_name,
    )


async def main():
    cli = chz.entrypoint(CLIConfig)
    config = build_config(cli)

    import math
    entropy_bits = math.log2(cli.N) if cli.N > 0 else 0
    print(f"\n{'='*80}")
    print(f"{config.wandb_name} summary:")
    print(f"{'='*80}")
    print(f"Environment: {cli.env_type}")
    if cli.env_type == "single_step":
        print(f"Reward type: {cli.reward_type}")
        if cli.reward_type.startswith("binned"):
            print(f"Reward bins (B): {cli.reward_bins}")
    print(f"N (secret space): {cli.N} ({entropy_bits:.2f} bits entropy)")
    print(f"Fixed secret: {cli.fixed_secret if cli.fixed_secret is not None else 'Random'}")
    print(f"Dataset seed: {cli.dataset_seed}")
    print(f"Group size (envs per group): {cli.group_size}")
    total_eps = cli.n_batches * cli.batch_size * cli.group_size
    print(
        f"Batches: {cli.n_batches} × {cli.batch_size} groups × {cli.group_size} envs = {total_eps} total envs"
    )
    print(f"Log path: {config.log_path}")
    print(f"{'='*80}\n")

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    await train.main(config)


if __name__ == "__main__":
    asyncio.run(main())
