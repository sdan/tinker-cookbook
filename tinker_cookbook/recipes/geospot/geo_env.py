"""
Geolocation environments for visual geolocation RL training.

Four environments for 2x2 ablation (thinking x tools):
1. GeoBaselineEnv     - Single-turn, no thinking, no tools (Geo-R style)
2. GeoThinkingEnv     - Single-turn with <think> blocks, no tools
3. GeoToolEnv         - Multi-turn with image_zoom_in tool, no forced thinking
4. GeoAgentEnv        - Multi-turn with tools + native thinking

Each env implements continuous distance-based rewards (not binary).

Usage:
    python -m tinker_cookbook.recipes.geospot.train env_type=baseline
    python -m tinker_cookbook.recipes.geospot.train env_type=thinking
    python -m tinker_cookbook.recipes.geospot.train env_type=tool
    python -m tinker_cookbook.recipes.geospot.train env_type=agent
"""

import json
import logging
import math
import re
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Sequence, TypedDict

import chz
import tinker
from PIL import Image

from tinker_cookbook import renderers
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.image_processing_utils import get_image_processor

from tinker_cookbook.recipes.geospot.data import GeoSample, iterate_samples, iterate_samples_webdataset

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


# =============================================================================
# Reward Functions
# =============================================================================


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def geoguessr_score(distance_km: float) -> int:
    """GeoGuessr-style score (0-5000)."""
    if distance_km < 0.05:
        return 5000
    return max(0, int(5000 * math.exp(-distance_km / 2000)))


def geo_r_reward(distance_km: float) -> float:
    """Geo-R paper piecewise linear reward (Equation 5).

    Provides continuous gradients within each distance band.
    """
    d = distance_km
    if d <= 750:
        return 1.0 - 0.5 * (d / 750)
    elif d <= 2500:
        return 0.5 - 0.3 * (d - 750) / 1750
    else:
        return max(0.0, 0.2 - 0.2 * (d - 2500) / 17500)


def normalize_country(country: str | None) -> str | None:
    """Normalize country to ISO 2-letter code."""
    if not country:
        return None
    country = country.strip()
    if len(country) == 2:
        return country.upper()
    try:
        import pycountry
        result = pycountry.countries.get(name=country)
        if result:
            return result.alpha_2
        results = pycountry.countries.search_fuzzy(country)
        if results:
            return results[0].alpha_2
    except (ImportError, LookupError):
        pass
    return country.upper()[:2] if len(country) >= 2 else country.upper()


def geo_r_hierarchical_reward(
    distance_km: float,
    pred_country: str | None,
    gt_country: str | None,
    pred_region: str | None = None,
    gt_region: str | None = None,
    country_weight: float = 0.3,
    region_weight: float = 0.1,
) -> tuple[float, dict]:
    """Hierarchical reward with partial credit for country/region."""
    base_weight = 1.0 - country_weight - region_weight
    distance_reward = geo_r_reward(distance_km) * base_weight

    pred_norm = normalize_country(pred_country)
    gt_norm = normalize_country(gt_country)
    country_match = pred_norm is not None and gt_norm is not None and pred_norm == gt_norm
    country_bonus = country_weight if country_match else 0.0

    region_bonus = 0.0
    region_match = False
    if country_match and pred_region and gt_region:
        pred_r = pred_region.strip().lower()
        gt_r = gt_region.strip().lower()
        region_match = pred_r == gt_r or pred_r in gt_r or gt_r in pred_r
        region_bonus = region_weight if region_match else 0.0

    return distance_reward + country_bonus + region_bonus, {
        "distance_reward": distance_reward,
        "country_match": country_match,
        "country_bonus": country_bonus,
        "region_match": region_match,
        "region_bonus": region_bonus,
    }


def distance_bucket(distance_km: float) -> str:
    """For stratified metrics."""
    if distance_km < 1:
        return "<1km"
    elif distance_km < 25:
        return "1-25km"
    elif distance_km < 200:
        return "25-200km"
    elif distance_km < 750:
        return "200-750km"
    elif distance_km < 2500:
        return "750-2500km"
    return ">2500km"


# =============================================================================
# Response Parsing
# =============================================================================


@dataclass
class GeoGroundTruth:
    """Ground truth for a geolocation sample."""
    lat: float
    lon: float
    country: str | None = None
    region: str | None = None
    city: str | None = None


@dataclass
class ParsedGeoResponse:
    """Parsed model output."""
    lat: float | None = None
    lon: float | None = None
    country: str | None = None
    region: str | None = None
    raw_text: str = ""

    @property
    def has_coords(self) -> bool:
        return self.lat is not None and self.lon is not None


def parse_geo_response(text: str) -> ParsedGeoResponse:
    """Parse model response for coordinates.

    Accepts:
        Latitude: <degrees>, Longitude: <degrees>
        Coordinates: (<lat>, <lon>)
        <lat>, <lon>
    """
    # Strip thinking blocks
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in clean_text:
        clean_text = clean_text.split("</think>", 1)[-1].strip()

    lat, lon = None, None
    country, region = None, None

    # Try Latitude/Longitude format
    lat_match = re.search(r"Latitude:\s*(-?\d+\.?\d*)", clean_text, re.IGNORECASE)
    lon_match = re.search(r"Longitude:\s*(-?\d+\.?\d*)", clean_text, re.IGNORECASE)
    if lat_match and lon_match:
        try:
            lat, lon = float(lat_match.group(1)), float(lon_match.group(1))
        except ValueError:
            pass

    # Try Coordinates: (lat, lon)
    if lat is None:
        coord_match = re.search(
            r"Coordinates:\s*\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?",
            clean_text, re.IGNORECASE
        )
        if coord_match:
            try:
                lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
            except ValueError:
                pass

    # Try bare lat, lon
    if lat is None:
        bare_match = re.search(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", clean_text)
        if bare_match:
            try:
                lat, lon = float(bare_match.group(1)), float(bare_match.group(2))
            except ValueError:
                pass

    # Validate coordinates
    if lat is not None and lon is not None:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            lat, lon = None, None

    # Parse labels
    country_match = re.search(r"Country:\s*(.+?)(?:\n|$)", clean_text, re.IGNORECASE)
    region_match = re.search(r"(?:Region|State):\s*(.+?)(?:\n|$)", clean_text, re.IGNORECASE)
    if country_match:
        country = country_match.group(1).strip()
    if region_match:
        region = region_match.group(1).strip()

    return ParsedGeoResponse(lat=lat, lon=lon, country=country, region=region, raw_text=clean_text)


# =============================================================================
# Prompts
# =============================================================================

GEO_BASELINE_PROMPT = """Where is this image located?

Output format:
Country: <country name>
Coordinates: (<latitude>, <longitude>)"""

GEO_THINKING_PROMPT = """Where is this image located?

Think step by step about visual clues (landscape, architecture, vegetation, signs, vehicles) before answering.

Output format:
Country: <country name>
Coordinates: (<latitude>, <longitude>)"""

GEO_TOOL_PROMPT = """Where is this image located?

You can use image_zoom_in_tool to examine details like signs, license plates, or text.

Output format:
Country: <country name>
Coordinates: (<latitude>, <longitude>)"""

GEO_AGENT_PROMPT = """Where is this image located?

Think step by step about visual clues. You can use image_zoom_in_tool to examine details.

Output format:
Country: <country name>
Coordinates: (<latitude>, <longitude>)"""

THINKING_PREFILL = "<think>\nLet me analyze the visual clues in this image to determine the location.\n"


# =============================================================================
# Image Zoom Tool
# =============================================================================


class ToolSpec(TypedDict):
    name: str
    description: str
    parameters: dict


def get_zoom_tool_schema() -> ToolSpec:
    """Qwen3-VL compatible zoom tool schema."""
    return {
        "name": "image_zoom_in_tool",
        "description": "Zoom in on a region of the image to see details like signs, license plates, or text. Coordinates are in [0, 1000] relative space.",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Bounding box [x1, y1, x2, y2] in [0, 1000] coordinates"
                },
                "label": {
                    "type": "string",
                    "description": "What you're zooming into (e.g., 'license plate', 'street sign')"
                },
            },
            "required": ["bbox_2d"]
        }
    }


def crop_image_by_bbox(image: Image.Image, bbox: list[int]) -> Image.Image:
    """Crop image using bbox in [0, 1000] relative coordinates."""
    w, h = image.size
    x1 = int(bbox[0] * w / 1000)
    y1 = int(bbox[1] * h / 1000)
    x2 = int(bbox[2] * w / 1000)
    y2 = int(bbox[3] * h / 1000)

    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))

    if x2 - x1 < 10:
        x2 = min(w, x1 + 50)
    if y2 - y1 < 10:
        y2 = min(h, y1 + 50)

    return image.crop((x1, y1, x2, y2))


# =============================================================================
# Base Geo Environment
# =============================================================================


RewardType = Literal["exp", "geoguessr", "geo_r", "geo_r_hierarchical"]


class GeoEnv(Env):
    """Base class for geolocation environments.

    Subclasses implement:
    - get_prompt() -> the task prompt
    - can override step() for tool handling
    """

    def __init__(
        self,
        image: Image.Image,
        ground_truth: GeoGroundTruth,
        renderer: renderers.Renderer,
        max_image_size: int = 512,
        format_penalty: float = 0.1,
        reward_type: RewardType = "exp",
        coord_tau: float = 2000.0,
        original_image: Image.Image | None = None,
    ):
        self.original_image = original_image if original_image is not None else image
        self.ground_truth = ground_truth
        self.renderer = renderer
        self.format_penalty = format_penalty
        self.reward_type = reward_type
        self.coord_tau = coord_tau

        # Resize image for model input (preserve aspect ratio)
        w, h = image.size
        if max_image_size and min(w, h) > max_image_size:
            scale = max_image_size / min(w, h)
            self.image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        else:
            self.image = image

        self.messages: list[renderers.Message] = []

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    @abstractmethod
    def get_prompt(self) -> str:
        """Return the task prompt."""
        pass

    def compute_reward(self, parsed: ParsedGeoResponse) -> tuple[float, Metrics]:
        """Compute reward from parsed response."""
        if not parsed.has_coords:
            return -self.format_penalty, {"format_error": 1}

        dist = haversine_km(parsed.lat, parsed.lon, self.ground_truth.lat, self.ground_truth.lon)

        if self.reward_type == "geoguessr":
            reward = geoguessr_score(dist) / 5000.0
        elif self.reward_type == "geo_r":
            reward = geo_r_reward(dist)
        elif self.reward_type == "geo_r_hierarchical":
            reward, breakdown = geo_r_hierarchical_reward(
                distance_km=dist,
                pred_country=parsed.country,
                gt_country=self.ground_truth.country,
                pred_region=parsed.region,
                gt_region=self.ground_truth.region,
            )
        elif self.reward_type == "exp":
            reward = math.exp(-dist / self.coord_tau)
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        metrics: Metrics = {
            "distance_km": dist,
            "reward": reward,
            "pred_lat": parsed.lat,
            "pred_lon": parsed.lon,
            "gt_lat": self.ground_truth.lat,
            "gt_lon": self.ground_truth.lon,
        }

        if self.reward_type == "geo_r_hierarchical":
            metrics.update({
                "country_match": breakdown["country_match"],
                "region_match": breakdown["region_match"],
            })

        return reward, metrics

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        self.messages = [renderers.Message(role="user", content=[
            {"type": "image", "image": self.image},
            {"type": "text", "text": self.get_prompt()},
        ])]
        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """Default single-turn step: parse response, compute reward, done."""
        msg, _ = self.renderer.parse_response(action)
        text = renderers.get_text_content(msg)
        parsed = parse_geo_response(text)
        reward, metrics = self.compute_reward(parsed)

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


# =============================================================================
# 1. Baseline Environment (Geo-R style)
# =============================================================================


class GeoBaselineEnv(GeoEnv):
    """Single-turn geolocation without thinking or tools."""

    def get_prompt(self) -> str:
        return GEO_BASELINE_PROMPT


# =============================================================================
# 2. Thinking Environment
# =============================================================================


class GeoThinkingEnv(GeoEnv):
    """Single-turn geolocation with native thinking."""

    def get_prompt(self) -> str:
        return GEO_THINKING_PROMPT

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        self.messages = [renderers.Message(role="user", content=[
            {"type": "image", "image": self.image},
            {"type": "text", "text": self.get_prompt()},
        ])]
        return self.renderer.build_generation_prompt(self.messages, prefill=THINKING_PREFILL), self.stop_condition


# =============================================================================
# 3. Tool Environment
# =============================================================================


class GeoToolEnv(GeoEnv):
    """Multi-turn geolocation with image_zoom_in tool."""

    def __init__(
        self,
        image: Image.Image,
        ground_truth: GeoGroundTruth,
        renderer: renderers.Renderer,
        max_image_size: int = 512,
        format_penalty: float = 0.1,
        reward_type: RewardType = "exp",
        coord_tau: float = 2000.0,
        max_zooms: int = 3,
        zoom_cost: float = 0.0,
        original_image: Image.Image | None = None,
    ):
        super().__init__(image, ground_truth, renderer, max_image_size, format_penalty,
                         reward_type, coord_tau, original_image)
        self.max_zooms = max_zooms
        self.zoom_cost = zoom_cost
        self.zoom_count = 0

    def get_prompt(self) -> str:
        return GEO_TOOL_PROMPT

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        tool_schemas = [get_zoom_tool_schema()]
        self.messages = self.renderer.create_conversation_prefix_with_tools(tools=tool_schemas, system_prompt="")
        self.messages.append(renderers.Message(role="user", content=[
            {"type": "image", "image": self.image},
            {"type": "text", "text": self.get_prompt()},
        ]))
        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        msg, _ = self.renderer.parse_response(action)
        self.messages.append(msg)

        if "tool_calls" in msg and msg["tool_calls"]:
            tool_call = msg["tool_calls"][0]

            if tool_call.function.name == "image_zoom_in_tool":
                self.zoom_count += 1

                if self.zoom_count > self.max_zooms:
                    return StepResult(
                        reward=-self.format_penalty, episode_done=True,
                        next_observation=tinker.ModelInput.empty(),
                        next_stop_condition=self.stop_condition,
                        metrics={"error": "max_zooms_exceeded", "zoom_count": self.zoom_count},
                    )

                try:
                    args = json.loads(tool_call.function.arguments)
                    bbox = args["bbox_2d"]
                    cropped = crop_image_by_bbox(self.original_image, bbox)
                except Exception as e:
                    logger.warning(f"Failed to parse zoom args: {e}")
                    return StepResult(
                        reward=-self.format_penalty, episode_done=True,
                        next_observation=tinker.ModelInput.empty(),
                        next_stop_condition=self.stop_condition,
                        metrics={"error": "zoom_parse_failed"},
                    )

                tool_response: renderers.Message = {
                    "role": "tool",
                    "content": [{"type": "image", "image": cropped}, {"type": "text", "text": "Zoomed region:"}],
                }
                if tool_call.id is not None:
                    tool_response["tool_call_id"] = tool_call.id
                self.messages.append(tool_response)

                return StepResult(
                    reward=-self.zoom_cost, episode_done=False,
                    next_observation=self.renderer.build_generation_prompt(self.messages),
                    next_stop_condition=self.stop_condition,
                    metrics={"zoom_count": self.zoom_count},
                )

            return StepResult(
                reward=-self.format_penalty, episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"error": f"unknown_tool_{tool_call.function.name}"},
            )

        text = renderers.get_text_content(msg)
        parsed = parse_geo_response(text)
        reward, metrics = self.compute_reward(parsed)
        metrics["zoom_count"] = self.zoom_count
        return StepResult(reward=reward, episode_done=True, next_observation=tinker.ModelInput.empty(),
                          next_stop_condition=self.stop_condition, metrics=metrics)


# =============================================================================
# 4. Agent Environment (tools + native thinking)
# =============================================================================


class GeoAgentEnv(GeoToolEnv):
    """Multi-turn geolocation with tools AND native thinking."""

    def get_prompt(self) -> str:
        return GEO_AGENT_PROMPT

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        tool_schemas = [get_zoom_tool_schema()]
        self.messages = self.renderer.create_conversation_prefix_with_tools(tools=tool_schemas, system_prompt="")
        self.messages.append(renderers.Message(role="user", content=[
            {"type": "image", "image": self.image},
            {"type": "text", "text": self.get_prompt()},
        ]))
        return self.renderer.build_generation_prompt(self.messages, prefill=THINKING_PREFILL), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        msg, _ = self.renderer.parse_response(action)
        self.messages.append(msg)

        if "tool_calls" in msg and msg["tool_calls"]:
            tool_call = msg["tool_calls"][0]

            if tool_call.function.name == "image_zoom_in_tool":
                self.zoom_count += 1

                if self.zoom_count > self.max_zooms:
                    return StepResult(
                        reward=-self.format_penalty, episode_done=True,
                        next_observation=tinker.ModelInput.empty(),
                        next_stop_condition=self.stop_condition,
                        metrics={"error": "max_zooms_exceeded", "zoom_count": self.zoom_count},
                    )

                try:
                    args = json.loads(tool_call.function.arguments)
                    bbox = args["bbox_2d"]
                    cropped = crop_image_by_bbox(self.original_image, bbox)
                except Exception as e:
                    logger.warning(f"Failed to parse zoom args: {e}")
                    return StepResult(
                        reward=-self.format_penalty, episode_done=True,
                        next_observation=tinker.ModelInput.empty(),
                        next_stop_condition=self.stop_condition,
                        metrics={"error": "zoom_parse_failed"},
                    )

                tool_response: renderers.Message = {
                    "role": "tool",
                    "content": [{"type": "image", "image": cropped}, {"type": "text", "text": "Zoomed region:"}],
                }
                if tool_call.id is not None:
                    tool_response["tool_call_id"] = tool_call.id
                self.messages.append(tool_response)

                # Continue with thinking prefill
                return StepResult(
                    reward=-self.zoom_cost, episode_done=False,
                    next_observation=self.renderer.build_generation_prompt(self.messages, prefill=THINKING_PREFILL),
                    next_stop_condition=self.stop_condition,
                    metrics={"zoom_count": self.zoom_count},
                )

            return StepResult(
                reward=-self.format_penalty, episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"error": f"unknown_tool_{tool_call.function.name}"},
            )

        text = renderers.get_text_content(msg)
        parsed = parse_geo_response(text)
        reward, metrics = self.compute_reward(parsed)
        metrics["zoom_count"] = self.zoom_count
        return StepResult(reward=reward, episode_done=True, next_observation=tinker.ModelInput.empty(),
                          next_stop_condition=self.stop_condition, metrics=metrics)


# =============================================================================
# Environment Factory & Dataset Builder
# =============================================================================

ENV_TYPES = {
    "baseline": GeoBaselineEnv,
    "thinking": GeoThinkingEnv,
    "tool": GeoToolEnv,
    "agent": GeoAgentEnv,
}

# Renderer mapping: thinking-enabled envs need qwen3_vl, others use qwen3_vl_instruct
ENV_RENDERER_MAP = {
    "baseline": "qwen3_vl_instruct",
    "thinking": "qwen3_vl",
    "tool": "qwen3_vl_instruct",
    "agent": "qwen3_vl",
}

ENV_MAX_TOKENS = {
    "baseline": 128,
    "thinking": 512,
    "tool": 256,
    "agent": 512,
}


def create_geo_env(
    env_type: str,
    image: Image.Image,
    ground_truth: GeoGroundTruth,
    renderer: renderers.Renderer,
    **kwargs,
) -> GeoEnv:
    """Factory function to create geo environments."""
    if env_type not in ENV_TYPES:
        raise ValueError(f"Unknown env_type: {env_type}. Available: {list(ENV_TYPES.keys())}")
    return ENV_TYPES[env_type](image=image, ground_truth=ground_truth, renderer=renderer, **kwargs)


@dataclass(frozen=True)
class GeoGroupBuilder(EnvGroupBuilder):
    """Builds N copies of a GeoEnv for GRPO training."""
    env_thunk: Callable[[], GeoEnv]
    num_envs: int
    dataset_name: str = "geospot"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectories: list[Trajectory], envs: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectories]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


class GeoRLDataset(RLDataset):
    """RL dataset that yields GeoGroupBuilders from streaming data."""

    def __init__(
        self,
        samples: list[GeoSample],
        renderer: renderers.Renderer,
        env_type: str,
        group_size: int,
        max_image_size: int = 512,
        format_penalty: float = 0.1,
        reward_type: RewardType = "exp",
        coord_tau: float = 2000.0,
        max_zooms: int = 3,
        zoom_cost: float = 0.0,
    ):
        self.samples = samples
        self.renderer = renderer
        self.env_type = env_type
        self.group_size = group_size
        self.max_image_size = max_image_size
        self.format_penalty = format_penalty
        self.reward_type = reward_type
        self.coord_tau = coord_tau
        self.max_zooms = max_zooms
        self.zoom_cost = zoom_cost

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        sample = self.samples[index]
        gt = GeoGroundTruth(
            lat=sample.lat, lon=sample.lon,
            country=sample.country, region=sample.region, city=sample.city,
        )
        env_class = ENV_TYPES[self.env_type]

        if self.env_type in ("tool", "agent"):
            def make_env():
                return env_class(
                    image=sample.image, ground_truth=gt, renderer=self.renderer,
                    max_image_size=self.max_image_size, format_penalty=self.format_penalty,
                    reward_type=self.reward_type, coord_tau=self.coord_tau,
                    max_zooms=self.max_zooms, zoom_cost=self.zoom_cost,
                    original_image=sample.original_image,
                )
        else:
            def make_env():
                return env_class(
                    image=sample.image, ground_truth=gt, renderer=self.renderer,
                    max_image_size=self.max_image_size, format_penalty=self.format_penalty,
                    reward_type=self.reward_type, coord_tau=self.coord_tau,
                )

        return [GeoGroupBuilder(env_thunk=make_env, num_envs=self.group_size)]


@chz.chz
class GeoDatasetBuilder(RLDatasetBuilder):
    """Builds GeoRLDataset from streaming data sources."""

    env_type: str = "baseline"
    model_name_for_tokenizer: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    renderer_name: str | None = None  # Auto-select based on env_type

    # Data source
    hf_repo: str = "osv5m/osv5m"
    data_dir: str | None = None
    batch_size: int = 64
    group_size: int = 8
    seed: int = 0

    # Environment config
    max_image_size: int = 512
    format_penalty: float = 0.1
    reward_type: RewardType = "exp"
    coord_tau: float = 2000.0
    max_zooms: int = 3
    zoom_cost: float = 0.0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        # Auto-select renderer
        renderer_name = self.renderer_name or ENV_RENDERER_MAP.get(self.env_type, "qwen3_vl_instruct")

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        image_processor = get_image_processor(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer, image_processor=image_processor)

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
            if i >= self.batch_size:
                break
            samples.append(sample)

        train_dataset = GeoRLDataset(
            samples=samples, renderer=renderer, env_type=self.env_type,
            group_size=self.group_size, max_image_size=self.max_image_size,
            format_penalty=self.format_penalty, reward_type=self.reward_type,
            coord_tau=self.coord_tau, max_zooms=self.max_zooms, zoom_cost=self.zoom_cost,
        )

        return train_dataset, None
