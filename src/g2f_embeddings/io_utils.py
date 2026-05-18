"""
Utility functions for file system and pandas IO operations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_fields(path: str | Path) -> pd.DataFrame:
    """Loads the field locations and coordinates CSV."""
    return pd.read_csv(path)


def load_environment(path: str | Path) -> pd.DataFrame:
    """Loads the environmental metadata CSV."""
    return pd.read_csv(path)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
