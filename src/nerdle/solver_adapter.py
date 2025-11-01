"""Wrapper utilities around the original Nerdle solver code.

This module keeps the historical solver logic in ``solver/`` as the single
source of truth and exposes a small API for the RL tooling.
"""

from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

from .config import GameConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOLVER_ROOT = PROJECT_ROOT / "solver"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from solver.wordle import filtermaskW, genmaskW  # type: ignore  # noqa: E402
from solver.NerdleGenerator import checkworks  # type: ignore  # noqa: E402
from solver.MaxiGenerator import parser as maxi_parser  # type: ignore  # noqa: E402


CLASSIC_RESTRICTED_PATH = SOLVER_ROOT / "NerdleClassicRestricted.txt"


def load_classic_answers() -> List[str]:
    """Return the list of classic Nerdle answers.

    The restricted dictionary mirrors the official game constraints (no leading
    zeroes, integer RHS, etc.). The file ships with the upstream solver repo.
    """

    return _load_lines(CLASSIC_RESTRICTED_PATH)


def load_classic_guesses() -> List[str]:
    """Return the list of valid guesses.

    For now we reuse the restricted list. A future improvement could expose the
    raw list for exploratory guesses.
    """

    return load_classic_answers()


@lru_cache(maxsize=1)
def _load_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def mask_guess(guess: str, solution: str) -> str:
    """Return the Nerdle mask for ``guess`` compared to ``solution``."""

    return genmaskW(guess, solution)


def filter_candidates(guess: str, mask: str, candidates: Sequence[str]) -> List[str]:
    """Filter ``candidates`` by applying Nerdle mask semantics."""

    return filtermaskW(mask, guess, list(candidates))


def is_valid_equation(equation: str) -> bool:
    """Return ``True`` if ``equation`` is a syntactically valid classic Nerdle."""

    if len(equation) != GameConfig().equation_length:
        return False
    return checkworks(equation)


def evaluate_with_maxi_parser(equation: str) -> bool:
    """Use the Maxi parser to validate more complex expressions.

    Useful as a second line of defence when checking LLM outputs that may use
    characters outside the classic ruleset. Returns ``True`` when the parsed
    value matches the RHS integer and is non-negative.
    """

    if "=" not in equation:
        return False
    lhs, rhs = equation.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if not lhs or not rhs:
        return False

    try:
        rhs_value = int(rhs)
    except ValueError:
        return False

    try:
        value = maxi_parser.parse_expression(lhs)
    except Exception:  # pragma: no cover - defensive fallback
        return False

    # Convert Fractions to integers when possible.
    try:
        numerator = value.numerator  # type: ignore[attr-defined]
        denominator = value.denominator  # type: ignore[attr-defined]
    except AttributeError:
        evaluated = value
    else:
        if denominator != 1:
            return False
        evaluated = numerator

    if not isinstance(evaluated, (int, float)):
        return False

    return int(evaluated) == rhs_value and rhs_value >= 0


def random_solution(rng: random.Random | None = None) -> str:
    """Return a random solution from the classic dictionary."""

    choices = load_classic_answers()
    rng = rng or random
    return rng.choice(choices)


def random_guess(exclude: Iterable[str] = (), rng: random.Random | None = None) -> str:
    """Return a random valid guess excluding ``exclude``."""

    available = [
        guess
        for guess in load_classic_guesses()
        if guess not in set(exclude)
    ]
    if not available:
        raise ValueError("No available guesses remaining.")
    rng = rng or random
    return rng.choice(available)


__all__ = [
    "CLASSIC_RESTRICTED_PATH",
    "evaluate_with_maxi_parser",
    "filter_candidates",
    "is_valid_equation",
    "load_classic_answers",
    "load_classic_guesses",
    "mask_guess",
    "random_guess",
    "random_solution",
]

