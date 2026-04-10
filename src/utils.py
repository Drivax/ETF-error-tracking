"""Utility functions used across the project."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def set_random_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory path if it does not exist."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Compute MAPE while protecting from division by near-zero values."""
    denominator = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denominator))


def time_split(data: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/test subsets while preserving temporal order."""
    split_idx = int(len(data) * (1 - test_size))
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    return train_df, test_df


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a dictionary payload to JSON on disk."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a dictionary payload from JSON on disk."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
