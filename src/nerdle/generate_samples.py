"""Generate classic Nerdle gameplay samples using the legacy solver."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import GameConfig, load_config
from .solver_adapter import (
    filter_candidates,
    load_classic_answers,
    mask_guess,
    random_solution,
)


SOLVED_MASK_CACHE: Dict[int, str] = {}


def solved_mask(length: int) -> str:
    if length not in SOLVED_MASK_CACHE:
        SOLVED_MASK_CACHE[length] = "2" * length
    return SOLVED_MASK_CACHE[length]


def simulate_episode(
    solution: str,
    answers: Sequence[str],
    config: GameConfig,
    rng: random.Random,
) -> Optional[dict]:
    """Simulate a Nerdle game returning the first four turns of history."""

    possible = answers
    guessed: List[str] = []
    history: List[dict] = []
    guess = config.initial_guess
    solved = solved_mask(config.equation_length)

    for attempt in range(config.max_attempts):
        mask = mask_guess(guess, solution)
        history.append({"guess": guess, "mask": mask})
        guessed.append(guess)
        possible = filter_candidates(guess, mask, possible)

        if attempt == 3:  # We only keep four turns of history
            break

        if mask == solved:
            return None  # Solved too early for this dataset shape

        next_candidates = [p for p in possible if p not in guessed]
        if not next_candidates:
            return None
        guess = rng.choice(next_candidates)
    else:
        return None

    if len(history) != 4:
        return None
    if history[-1]["mask"] == solved:
        return None

    next_candidates = [p for p in possible if p not in guessed]
    if not next_candidates:
        return None

    next_guess = rng.choice(next_candidates)

    return {
        "solution": solution,
        "history": history,
        "next_guess": next_guess,
        "remaining_candidates": possible,
    }


def generate_samples(
    count: int,
    config: GameConfig,
    out_path: Path,
    seed: Optional[int] = None,
) -> List[dict]:
    rng = random.Random(seed)
    episodes: List[dict] = []
    attempts = 0
    max_attempts = count * 200  # safety net to avoid infinite loops
    answers = tuple(load_classic_answers())

    while len(episodes) < count and attempts < max_attempts:
        attempts += 1
        solution = random_solution(rng)
        episode = simulate_episode(solution, answers, config, rng)
        if not episode:
            continue
        episodes.append(episode)

    if len(episodes) < count:
        raise RuntimeError(
            f"Only generated {len(episodes)} samples after {attempts} attempts. "
            "Consider relaxing constraints or increasing max_attempts."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/nerdle_samples.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config().game
    episodes = generate_samples(args.count, config, args.out, args.seed)
    print(f"Wrote {len(episodes)} samples to {args.out}")


if __name__ == "__main__":
    main()

