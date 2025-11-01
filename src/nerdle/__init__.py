"""Nerdle RL helpers.

This package reuses the original solver utilities under `solver/` and exposes
lightweight wrappers used by the GRPO training workflow.
"""

from .config import GameConfig, RewardConfig, load_config

__all__ = [
    "GameConfig",
    "RewardConfig",
    "load_config",
]

