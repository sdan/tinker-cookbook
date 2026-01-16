"""
Evaluation utilities for geospot training.

Runs inference on held-out test set and computes distance/score metrics.

Usage:
    python -m tinker_cookbook.recipes.geospot.eval checkpoint_path=tinker://... env_type=baseline

Reference: tinker_cookbook/recipes/vlm_classifier/eval.py
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

import chz
import tinker

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.image_processing_utils import get_image_processor

from tinker_cookbook.recipes.geospot.data import GeoSample, iterate_samples, iterate_samples_webdataset
from tinker_cookbook.recipes.geospot.geo_env import (
    haversine_km,
    geoguessr_score,
    parse_geo_response,
    create_geo_env,
    GeoGroundTruth,
    GEO_BASELINE_PROMPT,
    GEO_THINKING_PROMPT,
    ENV_RENDERER_MAP,
    ENV_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Results from running evaluation."""
    mean_distance_km: float
    mean_score: float
    median_distance_km: float
    acc_1km: float
    acc_25km: float
    acc_200km: float
    acc_750km: float
    num_samples: int
    num_failed: int


def eval_result_to_dict(result: EvalResult, prefix: str = "eval") -> dict:
    """Convert EvalResult to dict for wandb logging."""
    return {
        f"{prefix}/distance_km": result.mean_distance_km,
        f"{prefix}/score": result.mean_score,
        f"{prefix}/median_distance_km": result.median_distance_km,
        f"{prefix}/acc_1km": result.acc_1km,
        f"{prefix}/acc_25km": result.acc_25km,
        f"{prefix}/acc_200km": result.acc_200km,
        f"{prefix}/acc_750km": result.acc_750km,
        f"{prefix}/num_samples": result.num_samples,
        f"{prefix}/num_failed": result.num_failed,
    }


async def run_eval(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    env_type: str,
    *,
    hf_repo: str = "osv5m/osv5m",
    data_dir: str | None = None,
    split: str = "test",
    num_samples: int = 100,
    max_tokens: int = 128,
    temperature: float = 0.0,
    seed: int = 42,
    max_image_size: int = 512,
    max_parallel: int = 32,
) -> EvalResult:
    """
    Run evaluation on test set.

    Args:
        sampling_client: Tinker sampling client
        renderer: Message renderer
        env_type: Environment type (baseline, thinking, tool, agent)
        num_samples: Number of samples to evaluate
        max_parallel: Max concurrent sampling requests

    Returns:
        EvalResult with distance/score metrics
    """
    # Load samples
    if data_dir:
        sample_iter = iterate_samples_webdataset(
            data_dir=data_dir, seed=seed, max_image_size=max_image_size
        )
    else:
        sample_iter = iterate_samples(
            hf_repo=hf_repo, split=split, seed=seed, max_image_size=max_image_size
        )

    samples = []
    for i, sample in enumerate(sample_iter):
        if i >= num_samples:
            break
        samples.append(sample)

    if not samples:
        return EvalResult(
            mean_distance_km=float("inf"), mean_score=0.0, median_distance_km=float("inf"),
            acc_1km=0.0, acc_25km=0.0, acc_200km=0.0, acc_750km=0.0,
            num_samples=0, num_failed=0,
        )

    # Select prompt based on env_type
    prompt = GEO_THINKING_PROMPT if env_type in ("thinking", "agent") else GEO_BASELINE_PROMPT

    stop_sequences = renderer.get_stop_sequences()
    sampling_params = tinker.SamplingParams(
        stop=stop_sequences, max_tokens=max_tokens, temperature=temperature
    )

    semaphore = asyncio.Semaphore(max_parallel)

    async def sample_one(sample: GeoSample) -> tuple[GeoSample, str | None]:
        async with semaphore:
            try:
                messages = [renderers.Message(role="user", content=[
                    {"type": "image", "image": sample.image},
                    {"type": "text", "text": prompt},
                ])]
                model_input = renderer.build_generation_prompt(messages)

                result = await sampling_client.sample_async(
                    prompt=model_input, num_samples=1, sampling_params=sampling_params
                )
                text = renderer.tokenizer.decode(result.sequences[0].tokens)
                return sample, text
            except Exception as e:
                logger.debug(f"Eval sample failed: {e}")
                return sample, None

    results = await asyncio.gather(*[asyncio.create_task(sample_one(s)) for s in samples])

    distances = []
    scores = []
    num_failed = 0

    for sample, response_text in results:
        if response_text is None:
            num_failed += 1
            continue

        parsed = parse_geo_response(response_text)
        if not parsed.has_coords:
            num_failed += 1
            continue

        dist = haversine_km(parsed.lat, parsed.lon, sample.lat, sample.lon)
        distances.append(dist)
        scores.append(geoguessr_score(dist))

    if not distances:
        return EvalResult(
            mean_distance_km=float("inf"), mean_score=0.0, median_distance_km=float("inf"),
            acc_1km=0.0, acc_25km=0.0, acc_200km=0.0, acc_750km=0.0,
            num_samples=0, num_failed=num_failed,
        )

    distances_sorted = sorted(distances)
    n = len(distances)

    return EvalResult(
        mean_distance_km=sum(distances) / n,
        mean_score=sum(scores) / n,
        median_distance_km=distances_sorted[n // 2],
        acc_1km=sum(1 for d in distances if d < 1) / n,
        acc_25km=sum(1 for d in distances if d < 25) / n,
        acc_200km=sum(1 for d in distances if d < 200) / n,
        acc_750km=sum(1 for d in distances if d < 750) / n,
        num_samples=n,
        num_failed=num_failed,
    )


async def run_env_eval(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    env_type: str,
    *,
    hf_repo: str = "osv5m/osv5m",
    data_dir: str | None = None,
    split: str = "test",
    num_samples: int = 100,
    max_tokens: int = 128,
    temperature: float = 0.0,
    seed: int = 42,
    max_image_size: int = 512,
    format_penalty: float = 0.1,
    reward_type: str = "exp",
    coord_tau: float = 2000.0,
    max_zooms: int = 3,
    zoom_cost: float = 0.0,
    max_parallel: int = 16,
    max_steps_per_episode: int = 16,
) -> EvalResult:
    """Evaluate by rolling out the actual environment (supports tools, multi-turn)."""
    if data_dir:
        sample_iter = iterate_samples_webdataset(
            data_dir=data_dir, seed=seed, max_image_size=max_image_size
        )
    else:
        sample_iter = iterate_samples(
            hf_repo=hf_repo, split=split, seed=seed, max_image_size=max_image_size
        )

    samples = []
    for i, sample in enumerate(sample_iter):
        if i >= num_samples:
            break
        samples.append(sample)

    if not samples:
        return EvalResult(
            mean_distance_km=float("inf"), mean_score=0.0, median_distance_km=float("inf"),
            acc_1km=0.0, acc_25km=0.0, acc_200km=0.0, acc_750km=0.0,
            num_samples=0, num_failed=0,
        )

    semaphore = asyncio.Semaphore(max_parallel)

    async def eval_one(sample: GeoSample) -> tuple[float | None, float | None]:
        async with semaphore:
            try:
                gt = GeoGroundTruth(
                    lat=sample.lat, lon=sample.lon,
                    country=sample.country, region=sample.region, city=sample.city,
                )
                env_kwargs = {
                    "max_image_size": max_image_size,
                    "format_penalty": format_penalty,
                    "reward_type": reward_type,
                    "coord_tau": coord_tau,
                    "original_image": sample.original_image,
                }
                if env_type in ("tool", "agent"):
                    env_kwargs.update({"max_zooms": max_zooms, "zoom_cost": zoom_cost})

                env = create_geo_env(
                    env_type=env_type,
                    image=sample.image,
                    ground_truth=gt,
                    renderer=renderer,
                    **env_kwargs,
                )

                ob, stop = await env.initial_observation()
                for _ in range(max_steps_per_episode):
                    sampling_params = tinker.SamplingParams(
                        stop=stop, max_tokens=max_tokens, temperature=temperature
                    )
                    result = await sampling_client.sample_async(
                        prompt=ob, num_samples=1, sampling_params=sampling_params
                    )
                    step = await env.step(result.sequences[0].tokens)
                    if step.episode_done:
                        dist = step.metrics.get("distance_km")
                        if isinstance(dist, (int, float)):
                            return float(dist), float(geoguessr_score(float(dist)))
                        return None, None
                    ob, stop = step.next_observation, step.next_stop_condition
                return None, None
            except Exception as e:
                logger.debug(f"Env eval failed: {e}")
                return None, None

    results = await asyncio.gather(*[asyncio.create_task(eval_one(s)) for s in samples])

    distances = []
    scores = []
    num_failed = 0
    for dist, score in results:
        if dist is None or score is None:
            num_failed += 1
            continue
        distances.append(dist)
        scores.append(score)

    if not distances:
        return EvalResult(
            mean_distance_km=float("inf"), mean_score=0.0, median_distance_km=float("inf"),
            acc_1km=0.0, acc_25km=0.0, acc_200km=0.0, acc_750km=0.0,
            num_samples=0, num_failed=num_failed,
        )

    distances_sorted = sorted(distances)
    n = len(distances)
    return EvalResult(
        mean_distance_km=sum(distances) / n,
        mean_score=sum(scores) / n,
        median_distance_km=distances_sorted[n // 2],
        acc_1km=sum(1 for d in distances if d < 1) / n,
        acc_25km=sum(1 for d in distances if d < 25) / n,
        acc_200km=sum(1 for d in distances if d < 200) / n,
        acc_750km=sum(1 for d in distances if d < 750) / n,
        num_samples=n,
        num_failed=num_failed,
    )


# =============================================================================
# CLI
# =============================================================================


@chz.chz
class CLIConfig:
    """CLI configuration for standalone evaluation."""

    checkpoint_path: str  # tinker:// path to model checkpoint
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    env_type: str = "baseline"
    renderer_name: str | None = None

    hf_repo: str = "osv5m/osv5m"
    data_dir: str | None = None
    split: str = "test"
    num_samples: int = 500
    seed: int = 42

    max_tokens: int = 0  # 0 = auto
    temperature: float = 0.0
    max_image_size: int = 512
    max_parallel: int = 32

    base_url: str | None = None


async def cli_main(cli_config: CLIConfig):
    """Run standalone evaluation."""
    renderer_name = cli_config.renderer_name or ENV_RENDERER_MAP.get(cli_config.env_type, "qwen3_vl_instruct")
    max_tokens = cli_config.max_tokens if cli_config.max_tokens > 0 else ENV_MAX_TOKENS.get(cli_config.env_type, 128)

    tokenizer = get_tokenizer(cli_config.model_name)
    image_processor = get_image_processor(cli_config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer, image_processor=image_processor)

    service_client = tinker.ServiceClient(base_url=cli_config.base_url)
    sampling_client = await service_client.create_sampling_client_from_weights_async(cli_config.checkpoint_path)

    logger.info(f"Evaluating {cli_config.checkpoint_path} on {cli_config.num_samples} samples...")

    if cli_config.env_type in ("tool", "agent"):
        result = await run_env_eval(
            sampling_client=sampling_client,
            renderer=renderer,
            env_type=cli_config.env_type,
            hf_repo=cli_config.hf_repo,
            data_dir=cli_config.data_dir,
            split=cli_config.split,
            num_samples=cli_config.num_samples,
            max_tokens=max_tokens,
            temperature=cli_config.temperature,
            seed=cli_config.seed,
            max_image_size=cli_config.max_image_size,
            max_parallel=cli_config.max_parallel,
        )
    else:
        result = await run_eval(
            sampling_client=sampling_client,
            renderer=renderer,
            env_type=cli_config.env_type,
            hf_repo=cli_config.hf_repo,
            data_dir=cli_config.data_dir,
            split=cli_config.split,
            num_samples=cli_config.num_samples,
            max_tokens=max_tokens,
            temperature=cli_config.temperature,
            seed=cli_config.seed,
            max_image_size=cli_config.max_image_size,
            max_parallel=cli_config.max_parallel,
        )

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {cli_config.env_type}")
    print(f"{'='*60}")
    print(f"Samples:        {result.num_samples} (failed: {result.num_failed})")
    print(f"Mean Distance:  {result.mean_distance_km:.1f} km")
    print(f"Median Distance: {result.median_distance_km:.1f} km")
    print(f"Mean Score:     {result.mean_score:.0f} / 5000")
    print(f"Acc@1km:        {result.acc_1km:.1%}")
    print(f"Acc@25km:       {result.acc_25km:.1%}")
    print(f"Acc@200km:      {result.acc_200km:.1%}")
    print(f"Acc@750km:      {result.acc_750km:.1%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
