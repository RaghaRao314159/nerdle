#!/usr/bin/env python3
"""
Generate a conversational Nerdle dataset that mirrors the format of dataset.jsonl.

Each conversation contains a user message with three consecutive Nerdle attempts
and coloured feedback, followed by an assistant message that supplies the solver's
next equation guess. The history traces are produced by replicating the solving
strategy used in Nerdle-Master.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

WORD_SIZE = 8
DEFAULT_HISTORY_LENGTH = 3
DEFAULT_START_GUESS = "48-32=16"
DEFAULT_SAMPLE_SIZE = 120
DEFAULT_OUTPUT_NAME = "dataset_nerdle.jsonl"
VALID_CHARACTERS = tuple("0123456789+-*/=")


def load_equations(repo_root: Path) -> List[str]:
    """Load the library of valid Nerdle equations shipped with Nerdle-Master."""
    source = repo_root / "all_starting_guesses.txt"
    if not source.exists():
        raise FileNotFoundError(f"Could not locate {source}")

    with source.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def evaluate_guess(guess: str, target: str) -> str:
    """Return Nerdle colour feedback string (g, p, b) for guess against target."""
    guess_chars = list(guess)
    target_chars = list(target)
    feedback = ["b"] * WORD_SIZE

    for idx in range(WORD_SIZE):
        if guess_chars[idx] == target_chars[idx]:
            feedback[idx] = "g"
            guess_chars[idx] = None
            target_chars[idx] = None

    for idx in range(WORD_SIZE):
        if feedback[idx] == "g":
            continue
        char = guess_chars[idx]
        if char is not None and char in target_chars:
            feedback[idx] = "p"
            target_chars[target_chars.index(char)] = None

    return "".join(feedback)


def score_guess(candidate: str, possible: Sequence[str]) -> float:
    """Information gain score for a candidate guess over remaining possibilities."""
    partitions = {}
    total = len(possible)
    for solution in possible:
        pattern = evaluate_guess(candidate, solution)
        partitions[pattern] = partitions.get(pattern, 0) + 1

    score = 0.0
    for count in partitions.values():
        probability = count / total
        score -= probability * math.log2(probability)
    return score


def choose_next_guess(possible: Sequence[str], sample_size: int) -> str:
    """Pick the highest scoring guess from a sampled subset of remaining equations."""
    if not possible:
        raise ValueError("No candidate equations remain after filtering.")

    if len(possible) <= sample_size:
        candidates = list(possible)
    else:
        candidates = random.sample(possible, sample_size)
        # Always include the first remaining candidate for determinism.
        candidates.append(possible[0])

    best_candidate = max(candidates, key=lambda item: score_guess(item, possible))
    return best_candidate


def filter_candidates(
    candidates: Iterable[str], guess: str, feedback: str
) -> List[str]:
    """Filter candidate equations to those consistent with feedback for a guess."""
    return [item for item in candidates if evaluate_guess(guess, item) == feedback]


def simulate_history(
    solution: str,
    universe: Sequence[str],
    *,
    history_length: int,
    starting_guess: str,
    sample_size: int,
) -> Optional[Tuple[List[Tuple[int, str, str]], str]]:
    """Return (attempt history, next guess) if the solver needs the full history."""
    possible = list(universe)
    guess = starting_guess
    history: List[Tuple[int, str, str]] = []

    for attempt_idx in range(1, history_length + 1):
        if guess == solution:
            return None

        feedback = evaluate_guess(guess, solution)
        history.append((attempt_idx, guess, feedback))
        possible = filter_candidates(possible, guess, feedback)

        if not possible:
            return None

        next_guess = choose_next_guess(possible, sample_size)
        guess = next_guess

    return history, guess


def format_history(history: Sequence[Tuple[int, str, str]]) -> str:
    """Render the attempt history into a simple text prompt."""
    lines: List[str] = []
    lines.append("This is a nerdle game.")
    lines.append("Legend: g = green (correct spot), p = purple (wrong spot), b = black (not in equation).")
    allowed_chars = list(VALID_CHARACTERS)
    char_status = {char: "unknown" for char in allowed_chars}

    for attempt_idx, guess, feedback in history:
        tokens = " ".join(f"{char}{colour}" for char, colour in zip(guess, feedback))
        lines.append(f"Attempt {attempt_idx}: {guess} | Feedback: {tokens}")

        for char, colour in zip(guess, feedback):
            if colour in ("g", "p"):
                char_status[char] = "present"

        for char in set(guess):
            if char_status.get(char) == "present":
                continue
            char_feedbacks = [
                colour for candidate, colour in zip(guess, feedback) if candidate == char
            ]
            if char_feedbacks and all(colour == "b" for colour in char_feedbacks):
                char_status[char] = "absent"

        allowed_chars = [char for char in allowed_chars if char_status[char] != "absent"]

    allowed_chars_str = " ".join(allowed_chars)
    lines.append(f"Valid characters: {allowed_chars_str}")
    lines.append("Find the correct Nerdle equation.")
    return "\n".join(lines)


def build_dataset(
    equations: Sequence[str],
    *,
    num_examples: int,
    history_length: int,
    starting_guess: str,
    sample_size: int,
    progress_interval: int,
) -> List[dict]:
    """Generate the dataset entries leveraging the Nerdle solving strategy."""
    dataset: List[dict] = []
    pool = list(equations)
    random.shuffle(pool)
    pool_iter = iter(pool)
    max_attempts = max(len(pool) * 10, num_examples * 20)
    attempts = 0

    while len(dataset) < num_examples and attempts < max_attempts:
        attempts += 1
        try:
            solution = next(pool_iter)
        except StopIteration:
            random.shuffle(pool)
            pool_iter = iter(pool)
            solution = next(pool_iter)

        simulation = simulate_history(
            solution,
            equations,
            history_length=history_length,
            starting_guess=starting_guess,
            sample_size=sample_size,
        )

        if simulation is None:
            continue

        history, next_guess = simulation
        prompt = format_history(history)
        dataset.append(
            {
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": next_guess},
                ]
            }
        )

        if (
            progress_interval > 0
            and len(dataset) % progress_interval == 0
        ):
            print(
                f"Generated {len(dataset)} / {num_examples} conversations...",
                flush=True,
            )

    if len(dataset) < num_examples:
        raise RuntimeError(
            f"Could not assemble {num_examples} examples; produced {len(dataset)}."
        )

    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Nerdle history traces for training a conversational model."
    )
    # user needs to easily adjust example count
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of dataset rows to generate (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_NAME),
        help=f"Output .jsonl path (default: {DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=DEFAULT_HISTORY_LENGTH,
        help="Number of Nerdle attempts to include in the prompt history.",
    )
    parser.add_argument(
        "--starting-guess",
        type=str,
        default=DEFAULT_START_GUESS,
        help="Initial equation guess used by the solver.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Candidate sample size for scoring next guesses.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=20,
        help="Report progress every N generated conversations (0 disables updates).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for reproducible dataset generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    repo_root = Path(__file__).resolve().parent
    equations = load_equations(repo_root)

    dataset_rows = build_dataset(
        equations,
        num_examples=args.examples,
        history_length=args.history_length,
        starting_guess=args.starting_guess,
        sample_size=args.sample_size,
        progress_interval=args.progress_interval,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset_rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    print(f"Wrote {len(dataset_rows)} conversations to {output_path}")


if __name__ == "__main__":
    main()
