#!/usr/bin/env python3
"""
Utility helpers for loading YAML/JSON config files with graceful fallbacks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

DEFAULT_CONFIG = Path("config/default.yaml")


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load a YAML/JSON config file. Falls back to the default path when none provided.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    text = cfg_path.read_text()
    if cfg_path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to parse YAML configs. "
            "Install via `pip install pyyaml` or provide a JSON config."
        )
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must define a mapping: {cfg_path}")
    return data


def get_from_config(cfg: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """
    Retrieve nested config values using a list of keys.
    """
    ref: Any = cfg
    for key in keys:
        if not isinstance(ref, dict):
            return default
        ref = ref.get(key)
        if ref is None:
            return default
    return ref
