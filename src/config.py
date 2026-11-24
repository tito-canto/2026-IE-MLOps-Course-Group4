"""Load the YAML settings file that is the single source of truth for the pipeline."""
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    """Read the pipeline settings from a YAML file.

    Every stage reads its paths, column lists, split seed and quality gate from
    this one file, so settings change without code changes.

    Args:
        path: Location of the YAML settings file, e.g. ``config.yaml``.

    Returns:
        The parsed settings as a dictionary; an empty file gives ``{}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return cfg
