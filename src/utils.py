"""Shared file helpers for pipeline outputs."""
from pathlib import Path

import joblib
import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV without its index, creating missing parent folders.

    Pipeline outputs (processed data, predictions) go to folders that may not
    exist on a fresh clone, so the folder is created first.

    Args:
        frame: Data to write.
        path: Destination CSV file; an existing file is overwritten.

    Returns:
        None. The file is written to ``path``.

    Raises:
        OSError: If the folder cannot be created or the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_model(path: Path):
    """Load a saved model bundle written by :func:`src.train.save_model`.

    joblib files can run code when loaded, so only load files this team produced.

    Args:
        path: Location of the trusted ``.joblib`` file.

    Returns:
        The saved object, normally the bundle dictionary with ``model``,
        ``preprocessor``, ``predictors``, ``categorical`` and ``feature_columns``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    return joblib.load(path)
