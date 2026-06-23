from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from src.configs.default import DEFAULT_CONFIG


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return cfg

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    return deep_update(cfg, user_cfg)
