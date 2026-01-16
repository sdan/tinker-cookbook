"""
Supervised fine-tuning (SFT) for visual geolocation.

Warm-start training before RL. Creates training examples from
(image, coordinates) pairs with the target response format.

Usage:
    python -m tinker_cookbook.recipes.geospot.train_sft env_type=baseline
    python -m tinker_cookbook.recipes.geospot.train_sft env_type=thinking

Reference: tinker_cookbook/recipes/vlm_classifier/train.py
"""

import asyncio
import logging
from datetime import datetime
from typing import Literal

import chz
import torch
from tinker_cookbook import cli_utils
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.renderers import TrainOnWhat, get_renderer, Message
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.image_processing_utils import get_image_processor

from tinker_cookbook.recipes.geospot.data import GeoSample, iterate_samples, iterate_samples_webdataset
from tinker_cookbook.recipes.geospot.geo_env import (
    GEO_BASELINE_PROMPT,
    GEO_THINKING_PROMPT,
    ENV_RENDERER_MAP,
)

logger = logging.getLogger(__name__)


# Target format for SFT
def format_baseline_response(lat: float, lon: float, country: str | None = None) -> str:
    """Format response for baseline env (no thinking)."""
    country_line = f"Country: {country}\n" if country else "Country: Unknown\n"
    return f"{country_line}Coordinates: ({lat:.6f}, {lon:.6f})"


def format_thinking_response(lat: float, lon: float, country: str | None = None) -> str:
    """Format response for thinking env (with reasoning)."""
    country_line = f"Country: {country}\n" if country else "Country: Unknown\n"
    # Minimal thinking to demonstrate format
    return f"</think>\n{country_line}Coordinates: ({lat:.6f}, {lon:.6f})"


class GeoSFTDataset(SupervisedDataset):
    """SFT dataset for geolocation."""

    def __init__(
        self,
        samples: list[GeoSample],
        renderer,
        env_type: str,
        max_length: int,
        train_on_what: TrainOnWhat,
    ):
        self.samples = samples
        self.renderer = renderer
        self.env_type = env_type
        self.max_length = max_length
        self.train_on_what = train_on_what

        # Select prompt and response formatter based on env_type
        if env_type in ("baseline", "tool"):
            self.prompt = GEO_BASELINE_PROMPT
            self.format_response = format_baseline_response
        else:  # thinking, agent
            self.prompt = GEO_THINKING_PROMPT
            self.format_response = format_thinking_response

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(self, index: int):
        """Get a single datum for training."""
        sample = self.samples[index]
        return self._sample_to_datum(sample)

    def _sample_to_datum(self, sample: GeoSample):
        """Convert GeoSample to training Datum."""
        try:
            user_content = [
                {"type": "image", "image": sample.image},
                {"type": "text", "text": self.prompt},
            ]
            assistant_content = self.format_response(sample.lat, sample.lon, sample.country)

            messages = [
                Message(role="user", content=user_content),
                Message(role="assistant", content=assistant_content),
            ]

            model_input, weights = self.renderer.build_supervised_example(
                messages, train_on_what=self.train_on_what
            )

            return datum_from_model_input_weights(model_input, weights, max_length=self.max_length)
        except Exception as e:
            logger.debug(f"Failed to create datum: {e}")
            return None


@chz.chz
class GeoSFTDatasetBuilder(SupervisedDatasetBuilder):
    """Builds SFT dataset from streaming data."""

    env_type: str = "baseline"
    model_name_for_tokenizer: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    renderer_name: str | None = None

    # Data source
    hf_repo: str = "osv5m/osv5m"
    data_dir: str | None = None
    num_samples: int = 10000
    seed: int = 0

    # Training params
    batch_size: int = 32
    max_length: int = 8192
    max_image_size: int = 512
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE

    def __call__(self):
        # Auto-select renderer
        renderer_name = self.renderer_name or ENV_RENDERER_MAP.get(self.env_type, "qwen3_vl_instruct")

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        image_processor = get_image_processor(self.model_name_for_tokenizer)
        renderer = get_renderer(renderer_name, tokenizer=tokenizer, image_processor=image_processor)

        # Load samples
        if self.data_dir:
            sample_iter = iterate_samples_webdataset(
                data_dir=self.data_dir, seed=self.seed, max_image_size=self.max_image_size
            )
        else:
            sample_iter = iterate_samples(
                hf_repo=self.hf_repo, seed=self.seed, max_image_size=self.max_image_size
            )

        samples = []
        for i, sample in enumerate(sample_iter):
            if i >= self.num_samples:
                break
            samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples for SFT")

        # Split 90/10 for train/test
        split_idx = int(len(samples) * 0.9)
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]

        train_dataset = GeoSFTDataset(
            samples=train_samples,
            renderer=renderer,
            env_type=self.env_type,
            max_length=self.max_length,
            train_on_what=self.train_on_what,
        )

        test_dataset = GeoSFTDataset(
            samples=test_samples,
            renderer=renderer,
            env_type=self.env_type,
            max_length=self.max_length,
            train_on_what=self.train_on_what,
        ) if test_samples else None

        return train_dataset, test_dataset


@chz.chz
class CLIConfig:
    """CLI configuration for geospot SFT training."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment type (determines prompt/response format)
    env_type: str = "baseline"

    # Data
    hf_repo: str = "osv5m/osv5m"
    data_dir: str | None = None
    num_samples: int = 10000
    seed: int = 0

    # Training
    batch_size: int = 32
    learning_rate: float = 5e-4
    num_epochs: int = 3
    max_length: int = 8192
    max_image_size: int = 512

    # Logging
    log_path: str | None = None
    wandb_project: str | None = "geospot-sft"
    wandb_name: str | None = None

    # Checkpointing
    save_every: int = 100
    eval_every: int = 50

    # Service
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def run_sft(cli_config: CLIConfig):
    """Run SFT training."""

    # Build run name
    model_short = cli_config.model_name.replace("/", "-")
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"geospot-sft-{cli_config.env_type}-{model_short}-"
        f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
        f"{cli_config.batch_size}batch-{date_str}"
    )

    if cli_config.log_path:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/geospot_sft/{run_name}"

    wandb_name = cli_config.wandb_name or run_name

    # Build dataset builder
    dataset_builder = GeoSFTDatasetBuilder(
        env_type=cli_config.env_type,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=cli_config.renderer_name,
        hf_repo=cli_config.hf_repo,
        data_dir=cli_config.data_dir,
        num_samples=cli_config.num_samples,
        seed=cli_config.seed,
        batch_size=cli_config.batch_size,
        max_length=cli_config.max_length,
        max_image_size=cli_config.max_image_size,
    )

    # Build training config
    config = train.Config(
        model_name=cli_config.model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        num_epochs=cli_config.num_epochs,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    logger.info(f"Starting SFT training: env_type={cli_config.env_type}")
    logger.info(f"Model: {cli_config.model_name}, samples={cli_config.num_samples}")

    asyncio.run(train.main(config))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    chz.nested_entrypoint(run_sft)
