"""
GRPO training for visual geolocation.

2x2 Ablation Design:
- env_type: baseline | thinking | tool | agent
- Renderer auto-selected based on env_type

Usage:
    python -m tinker_cookbook.recipes.geospot.train env_type=baseline
    python -m tinker_cookbook.recipes.geospot.train env_type=thinking
    python -m tinker_cookbook.recipes.geospot.train env_type=tool
    python -m tinker_cookbook.recipes.geospot.train env_type=agent

Reference: tinker_cookbook/recipes/math_rl/train.py
"""

import asyncio
import logging
from datetime import datetime
from typing import Literal

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import Config, main
from tinker_cookbook.recipes.geospot.geo_env import (
    GeoDatasetBuilder,
    ENV_RENDERER_MAP,
    ENV_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """CLI configuration for geospot RL training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None  # Auto-select based on env_type
    load_checkpoint_path: str | None = None

    # Environment (2x2 ablation)
    env_type: str = "baseline"  # baseline, thinking, tool, agent

    # Data
    hf_repo: str = "osv5m/osv5m"
    data_dir: str | None = None  # If set, use WebDataset instead of HF
    seed: int = 0

    # Training
    batch_size: int = 64
    group_size: int = 8
    learning_rate: float = 4e-5
    max_tokens: int = 0  # 0 = auto-select based on env_type
    temperature: float = 1.0

    # Environment config
    max_image_size: int = 512
    format_penalty: float = 0.1
    reward_type: Literal["exp", "geoguessr", "geo_r", "geo_r_hierarchical"] = "geo_r_hierarchical"
    coord_tau: float = 2000.0
    max_zooms: int = 3
    zoom_cost: float = 0.0

    # Logging
    log_path: str | None = None
    wandb_project: str | None = "geospot-rl"
    wandb_name: str | None = None

    # Checkpointing & Evaluation
    save_every: int = 25
    eval_every: int = 0  # 0 = no eval during training

    # Service
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to training config and run."""

    # Auto-select renderer based on env_type
    renderer_name = cli_config.renderer_name or ENV_RENDERER_MAP.get(cli_config.env_type, "qwen3_vl_instruct")

    # Auto-select max_tokens based on env_type
    max_tokens = cli_config.max_tokens if cli_config.max_tokens > 0 else ENV_MAX_TOKENS.get(cli_config.env_type, 128)

    # Build run name
    model_short = cli_config.model_name.replace("/", "-")
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"geospot-{cli_config.env_type}-{model_short}-"
        f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
        f"{cli_config.group_size}group-{cli_config.batch_size}batch-"
        f"seed{cli_config.seed}-{date_str}"
    )

    if cli_config.log_path:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/geospot_rl/{run_name}"

    wandb_name = cli_config.wandb_name or run_name

    # Build dataset builder
    dataset_builder = GeoDatasetBuilder(
        env_type=cli_config.env_type,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        hf_repo=cli_config.hf_repo,
        data_dir=cli_config.data_dir,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        seed=cli_config.seed,
        max_image_size=cli_config.max_image_size,
        format_penalty=cli_config.format_penalty,
        reward_type=cli_config.reward_type,
        coord_tau=cli_config.coord_tau,
        max_zooms=cli_config.max_zooms,
        zoom_cost=cli_config.zoom_cost,
    )

    # Build training config
    config = Config(
        model_name=cli_config.model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        lora_rank=cli_config.lora_rank,
        max_tokens=max_tokens,
        temperature=cli_config.temperature,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    logger.info(f"Starting GRPO training: env_type={cli_config.env_type}, renderer={renderer_name}")
    logger.info(f"Model: {cli_config.model_name}, batch={cli_config.batch_size}, group={cli_config.group_size}")
    logger.info(f"Reward: {cli_config.reward_type}, max_tokens={max_tokens}")

    await main(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
