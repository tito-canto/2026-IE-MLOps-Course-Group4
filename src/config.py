import yaml
from pathlib import Path


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return cfg
