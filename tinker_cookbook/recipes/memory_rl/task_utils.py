from __future__ import annotations

import math
from typing import Literal


RewardType = Literal["binary", "binned_distance", "binned_distance_signed"]


def validate_secret(secret: int, N: int) -> None:
    if N <= 0:
        raise ValueError("N must be positive")
    if not (0 <= secret < N):
        raise ValueError(f"Secret {secret} must be in [0, {N - 1}] when N={N}")


def build_single_step_prompt(N: int) -> str:
    """Standard prompt shared between RL envs and SFT data."""
    return (
        "You are playing a memorization game. "
        f"There is a secret integer between 0 and {N - 1}. "
        "Output ONLY the integer, on the first line, with NO extra text.\n"
        "What is the secret number?"
    )


def build_multi_step_prompt(
    N: int,
    position: int,
    generated_bits: list[str],
) -> str:
    """
    Standard bit-by-bit prompt used by the multi-step RL environment.

    Kept here so that prompt wording stays centralized alongside the single-step
    prompt used for SFT.
    """
    num_bits = num_bits_for_space(N)
    remaining = num_bits - position
    so_far = "".join(generated_bits) or "(none yet)"
    return (
        "You are playing a bit-by-bit memorization game.\n"
        f"The secret number is between 0 and {N - 1} and uses {num_bits} bits.\n"
        "Output exactly one bit (0 or 1) per response with no punctuation.\n"
        f"Bits you have already provided: {so_far}.\n"
        f"Provide bit #{position + 1} now. Remaining bits after this: {max(0, remaining - 1)}."
    )


def parse_single_step_guess(sample_str: str, N: int) -> int | None:
    """Parse the first integer from the completion, clamping to valid range."""
    stripped = sample_str.strip()
    if not stripped:
        return None
    token = stripped.split()[0]
    try:
        guess = int(token)
    except ValueError:
        return None
    return guess if 0 <= guess < N else None


def parse_bit_guess(sample_str: str) -> str | None:
    """Parse a single bit ('0' or '1') from the completion."""
    stripped = sample_str.strip()
    if not stripped:
        return None
    ch = stripped[0]
    if ch in ("0", "1"):
        return ch
    return None


def compute_single_step_reward(
    secret: int,
    guess: int | None,
    reward_type: RewardType,
    N: int,
    reward_bins: int,
) -> tuple[float, bool, float]:
    """
    Compute the reward contribution (before format penalties) plus metrics.

    Returns:
        base_reward: float
        correct: bool
        distance: float
    """
    max_distance = max(N - 1, 1)
    if guess is None:
        distance = max_distance
        correct = False
        unsigned_score = 0.0
        signed_score = -1.0
    else:
        distance = abs(guess - secret)
        correct = guess == secret
        unsigned_score = 1.0 - (distance / max_distance)
        signed_score = (guess - secret) / max_distance

    if reward_bins < 2 and reward_type in {"binned_distance", "binned_distance_signed"}:
        raise ValueError(f"reward_bins must be >= 2 when using {reward_type}")

    if reward_type == "binary":
        base_reward = float(correct)
    elif reward_type == "binned_distance":
        score = max(0.0, min(1.0, unsigned_score))
        bin_idx = min(int(score * reward_bins), reward_bins - 1)
        base_reward = (2 * bin_idx + 1) / (2 * reward_bins)
    elif reward_type == "binned_distance_signed":
        score_01 = max(0.0, min(1.0, (signed_score + 1.0) / 2.0))
        bin_idx = min(int(score_01 * reward_bins), reward_bins - 1)
        base_reward = -1.0 + (2 * bin_idx + 1) / reward_bins
    else:  # pragma: no cover - guarded by RewardType Literal
        raise ValueError(f"Unknown reward_type {reward_type}")

    return base_reward, correct, float(distance)


def num_bits_for_space(N: int) -> int:
    """Return number of bits needed to represent integers in [0, N)."""
    if N <= 1:
        return 1
    return max(1, math.ceil(math.log2(N)))


def format_secret_bits(secret: int, N: int) -> str:
    """Return zero-padded bitstring for the secret."""
    width = num_bits_for_space(N)
    return format(secret, f"0{width}b")


def compute_bit_reward(secret_bits: str, position: int, guess_bit: str | None) -> tuple[float, bool]:
    """
    Compute per-step reward for the bitwise RL environment.

    Args:
        secret_bits: Bitstring representation of the secret (e.g. '0101').
        position: Index of the bit currently being guessed (0-based).
        guess_bit: Parsed guess ('0' or '1'), or None if parsing failed.

    Returns:
        reward: 1.0 if the guessed bit matches, else 0.0
        correct: bool indicating whether the guess was correct
    """
    if not (0 <= position < len(secret_bits)):
        raise ValueError(f"position {position} out of range for secret_bits of length {len(secret_bits)}")
    correct_bit = secret_bits[position]
    correct = guess_bit == correct_bit
    reward = 1.0 if correct else 0.0
    return reward, correct
