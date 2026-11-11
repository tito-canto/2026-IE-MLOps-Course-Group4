"""Shared file helpers for pipeline outputs."""
from pathlib import Path

import joblib
import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame without its index, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_model(path: Path):
    """Load a trusted model artifact."""
    return joblib.load(path)
