"""Configuration primitives for Nerdle RL tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GameConfig:
    equation_length: int = 8
    max_attempts: int = 6
    initial_guess: str = "6+4*-1=2"


@dataclass(frozen=True)
class RewardConfig:
    green: float = 3.0
    purple: float = 1.0
    black: float = 0.0
    invalid_penalty: float = -30.0
    win_bonus: float = 30.0
    decay_base: float = 1.1


@dataclass(frozen=True)
class Config:
    game: GameConfig = GameConfig()
    reward: RewardConfig = RewardConfig()


def load_config(
    game: Optional[GameConfig] = None,
    reward: Optional[RewardConfig] = None,
) -> Config:
    """Return a composed :class:`Config` with optional overrides."""

    return Config(game=game or GameConfig(), reward=reward or RewardConfig())

