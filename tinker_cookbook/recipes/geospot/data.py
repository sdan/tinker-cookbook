"""
Dataset loading for geolocation training.

Supports two data sources:
1. HuggingFace streaming (osv5m/osv5m) - default
2. WebDataset (local .tar shards) - for Modal volumes

Reference: tinker_cookbook/recipes/vlm_classifier/data.py
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GeoSample:
    """A geolocation sample with image and coordinates."""
    image: Image.Image  # Resized for model input
    lat: float
    lon: float
    # Hierarchical labels
    country: str | None = None
    region: str | None = None
    city: str | None = None
    source: str | None = None
    # Full-res image for tool zoom (optional)
    original_image: Image.Image | None = None


def _resize_shortest_side(image: Image.Image, target_size: int) -> Image.Image:
    """Resize so the shortest side is target_size, preserving aspect ratio."""
    if target_size <= 0:
        return image
    w, h = image.size
    short = min(w, h)
    if short <= 0 or short == target_size:
        return image
    scale = target_size / short
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def iterate_samples(
    hf_repo: str = "osv5m/osv5m",
    split: str = "train",
    seed: int = 0,
    shuffle_buffer: int = 1000,
    max_image_size: int = 512,
) -> Iterator[GeoSample]:
    """
    Stream GeoSamples from HuggingFace dataset.

    Args:
        hf_repo: HuggingFace dataset repo (e.g., "osv5m/osv5m")
        split: Dataset split ("train", "test", etc.)
        seed: Random seed for shuffling
        shuffle_buffer: Buffer size for streaming shuffle
        max_image_size: Resize images so the shortest side is this size

    Yields:
        GeoSample objects with image, lat, lon, and metadata
    """
    logger.info(f"Loading {hf_repo} ({split}) with streaming=True...")

    ds = load_dataset(hf_repo, split=split, streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    for idx, sample in enumerate(ds):
        try:
            image = sample.get("image")
            if image is None or not isinstance(image, Image.Image):
                continue

            # Convert to RGB
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

            original_image = image

            # Resize if needed
            if max_image_size:
                image = _resize_shortest_side(original_image, max_image_size)

            lat_raw = sample.get("latitude")
            lon_raw = sample.get("longitude")
            if lat_raw is None or lon_raw is None:
                continue
            lat, lon = float(lat_raw), float(lon_raw)

            # Validate coordinates
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            if idx == 0:
                logger.info(f"First sample received from {hf_repo}")

            yield GeoSample(
                image=image,
                lat=lat,
                lon=lon,
                country=sample.get("country"),
                region=sample.get("region"),
                city=sample.get("city"),
                source=hf_repo,
                original_image=original_image,
            )
        except Exception as e:
            logger.debug(f"Skipping sample: {e}")
            continue


def iterate_samples_webdataset(
    data_dir: str = "/data/train",
    seed: int = 0,
    max_image_size: int = 512,
    shuffle_buffer: int = 10000,
) -> Iterator[GeoSample]:
    """
    Stream GeoSamples from WebDataset (local .tar shards).

    Use this when data is stored locally (e.g., Modal volume).
    Falls back to HuggingFace streaming if webdataset not available.

    Args:
        data_dir: Path to directory containing .tar shards
        seed: Random seed for shuffling
        max_image_size: Resize images so the shortest side is this size
        shuffle_buffer: Buffer size for sample-level shuffling

    Yields:
        GeoSample objects with image, lat, lon, and metadata
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.warning("webdataset not installed, falling back to HuggingFace")
        yield from iterate_samples(seed=seed, max_image_size=max_image_size)
        return

    shards = list(Path(data_dir).glob("**/*.tar"))
    if not shards:
        logger.warning(f"No .tar shards found in {data_dir}, falling back to HuggingFace")
        yield from iterate_samples(seed=seed, max_image_size=max_image_size)
        return

    shard_urls = [str(s) for s in shards]
    logger.info(f"Found {len(shards)} shards in {data_dir}")

    ds = (
        wds.WebDataset(shard_urls, shardshuffle=True, seed=seed)
        .shuffle(shuffle_buffer, seed=seed)
        .decode("pil")
    )

    first_sample_logged = False

    for sample in ds:
        try:
            img = sample.get("jpg") or sample.get("png") or sample.get("image")
            if img is None or not isinstance(img, Image.Image):
                continue

            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")

            original_img = img
            if max_image_size:
                img = _resize_shortest_side(original_img, max_image_size)

            # Extract coordinates
            lat = sample.get("latitude") or sample.get("lat")
            lon = sample.get("longitude") or sample.get("lon") or sample.get("lng")
            country = sample.get("country") or sample.get("country_code")
            region = sample.get("region") or sample.get("state") or sample.get("admin1")
            city = sample.get("city")

            # Check JSON metadata
            if "json" in sample:
                meta = sample["json"] if isinstance(sample["json"], dict) else json.loads(sample["json"])
                lat = lat or meta.get("latitude") or meta.get("lat")
                lon = lon or meta.get("longitude") or meta.get("lon")
                country = country or meta.get("country") or meta.get("country_code")
                region = region or meta.get("region")
                city = city or meta.get("city")

            if lat is None or lon is None:
                continue

            lat, lon = float(lat), float(lon)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            if not first_sample_logged:
                logger.info(f"First sample received from {data_dir}")
                first_sample_logged = True

            yield GeoSample(
                image=img,
                lat=lat,
                lon=lon,
                country=country,
                region=region,
                city=city,
                source=data_dir,
                original_image=original_img,
            )
        except Exception as e:
            logger.debug(f"Skipping sample: {e}")
            continue
