from __future__ import annotations

import os
from pathlib import Path

MODEL_ARCHIVE_FILE_ID = "12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d"
EXPECTED_MODEL_FILES = (
    "matchanything_eloftr.ckpt",
    "matchanything_roma.ckpt",
)


def get_package_root() -> Path:
    return Path(__file__).resolve().parent


def get_repo_root() -> Path:
    return get_package_root().parents[1]


def get_runtime_home() -> Path:
    root = os.environ.get("MATCHANYTHING_HOME")
    if root:
        return Path(root).expanduser().resolve()
    return (Path.home() / ".cache" / "matchanything").resolve()


def get_logs_dir() -> Path:
    return get_runtime_home() / "logs"


def get_models_dir(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get("MATCHANYTHING_MODELS_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (get_runtime_home() / "models").resolve()


def get_vendor_root() -> Path:
    return get_package_root() / "third_party" / "MatchAnything"


def ensure_runtime_dirs(models_dir: str | Path | None = None) -> None:
    get_runtime_home().mkdir(parents=True, exist_ok=True)
    get_logs_dir().mkdir(parents=True, exist_ok=True)
    get_models_dir(models_dir).mkdir(parents=True, exist_ok=True)


def get_weights_dir(models_dir: str | Path | None = None) -> Path:
    resolved = get_models_dir(models_dir)
    nested = resolved / "weights"
    return nested if nested.exists() else resolved


def get_expected_weight_paths(models_dir: str | Path | None = None) -> dict[str, Path]:
    weights_dir = get_weights_dir(models_dir)
    return {name: weights_dir / name for name in EXPECTED_MODEL_FILES}
