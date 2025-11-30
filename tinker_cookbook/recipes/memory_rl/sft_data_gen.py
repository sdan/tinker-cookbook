"""
SFT data generation for Q2: Supervised baseline.

Generates JSONL files in `conversations.jsonl`-style format that mirror the
prompts used by the single-step RL environment, enabling apples-to-apples
comparisons between SFT and RL on the same memorization task.
"""

from pathlib import Path
from typing import Literal
import json

import chz
import numpy as np

from tinker_cookbook.recipes.memory_rl.task_utils import build_single_step_prompt, validate_secret


def get_prompt(N: int) -> str:
    """Get the standard prompt for the guessing task."""
    return build_single_step_prompt(N)


def generate_sft_data(
    N: int,
    num_examples: int,
    output_path: str,
    fixed_secret: int | None = None,
    seed: int = 0,
    split: Literal["train", "test"] = "train",
) -> None:
    """
    Generate JSONL dataset for SFT training.

    Each example mirrors the single-step RL environment:
        - User message: standard memorization prompt for a given N.
        - Assistant message: the secret integer to be memorized.

    Args:
        N: Secret space size [0, N-1].
        num_examples: Number of examples to generate.
        output_path: Path to output JSONL file.
        fixed_secret: If set, use same secret for all examples (memorization test).
        seed: Random seed.
        split: "train" or "test".
    """
    rng = np.random.RandomState(seed)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    prompt = get_prompt(N)
    metadata = {"N": int(N)}

    with output_file.open("w", encoding="utf-8") as f:
        for _ in range(num_examples):
            if fixed_secret is not None:
                secret = int(fixed_secret)
            else:
                secret = int(rng.randint(0, N))

            validate_secret(secret, N)

            example = {
                "split": split,
                "metadata": metadata,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": str(secret)},
                ],
            }
            f.write(json.dumps(example) + "\n")


@chz.chz
class SFTDataGenConfig:
    """CLI config for SFT data generation."""

    N: int = 16
    num_train: int = 1000
    num_test: int = 100
    output_dir: str = "/tmp/tinker-examples/memory_rl/sft_data"
    fixed_secret: int | None = None
    seed: int = 0


def main() -> None:
    """
    CLI entrypoint that writes train/test JSONL files to output_dir.
    """
    cli = chz.entrypoint(SFTDataGenConfig)
    output_dir = Path(cli.output_dir)
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    generate_sft_data(
        N=cli.N,
        num_examples=cli.num_train,
        output_path=str(train_path),
        fixed_secret=cli.fixed_secret,
        seed=cli.seed,
        split="train",
    )
    generate_sft_data(
        N=cli.N,
        num_examples=cli.num_test,
        output_path=str(test_path),
        fixed_secret=cli.fixed_secret,
        seed=cli.seed + 1,
        split="test",
    )

    print(f"Wrote train data to {train_path}")
    print(f"Wrote test data to {test_path}")


if __name__ == "__main__":
    main()
