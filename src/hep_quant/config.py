"""YAML configuration loading with dotted-path CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, resolving an optional `defaults: <path>` base config."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if "defaults" in cfg:
        base_path = path.parent / cfg.pop("defaults")
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `key.subkey=value` CLI overrides onto a config dict (YAML-typed values)."""
    cfg = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}', expected key.path=value")
        key_path, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        keys = key_path.split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return cfg
