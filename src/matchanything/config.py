from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .runtime import get_repo_root


def get_default_config_dir() -> Path:
    return get_repo_root() / "config"


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def load_runtime_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    resolved = Path(config_path).expanduser().resolve() if config_path else get_default_config_dir()
    if resolved.is_file():
        return _read_yaml(resolved)

    app_cfg = _read_yaml(resolved / "app.yaml")
    defaults_cfg = _read_yaml(resolved / "defaults.yaml")
    models_cfg = _read_yaml(resolved / "models.yaml")
    return {
        "server": app_cfg.get("server", {}),
        "api": app_cfg.get("api", {}),
        "defaults": defaults_cfg.get("defaults", defaults_cfg),
        "matcher_zoo": models_cfg.get("matcher_zoo", models_cfg),
    }
