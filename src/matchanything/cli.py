from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from . import __version__
from .config import get_default_config_dir, load_runtime_config
from .runtime import (
    EXPECTED_MODEL_FILES,
    MODEL_ARCHIVE_FILE_ID,
    ensure_runtime_dirs,
    get_expected_weight_paths,
    get_models_dir,
    get_repo_root,
    get_runtime_home,
)


def _set_models_dir(models_dir: str | Path | None) -> Path:
    resolved = get_models_dir(models_dir)
    os.environ["MATCHANYTHING_MODELS_DIR"] = str(resolved)
    ensure_runtime_dirs(resolved)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="matchanything")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ui_parser = subparsers.add_parser("ui", help="Launch the Gradio UI")
    ui_parser.add_argument("--config", default=str(get_default_config_dir()))
    ui_parser.add_argument("--server-name", default="0.0.0.0")
    ui_parser.add_argument("--server-port", type=int, default=7860)
    ui_parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    ui_parser.add_argument("--models-dir")
    ui_parser.set_defaults(func=run_ui)

    api_parser = subparsers.add_parser("api", help="Launch the FastAPI server")
    api_parser.add_argument("--config", default=str(get_default_config_dir()))
    api_parser.add_argument("--host", default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=8001)
    api_parser.add_argument("--default-matcher", default="matchanything_eloftr")
    api_parser.add_argument("--models-dir")
    api_parser.set_defaults(func=run_api)

    setup_parser = subparsers.add_parser("setup", help="Download or validate model assets")
    setup_parser.add_argument("--models-dir")
    setup_parser.add_argument("--check", action="store_true")
    setup_parser.add_argument("--force", action="store_true")
    setup_parser.set_defaults(func=run_setup)

    doctor_parser = subparsers.add_parser("doctor", help="Inspect local runtime readiness")
    doctor_parser.add_argument("--models-dir")
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.set_defaults(func=run_doctor)

    match_parser = subparsers.add_parser("match", help="Run a local image-pair match and write JSON")
    match_parser.add_argument("image0")
    match_parser.add_argument("image1")
    match_parser.add_argument("--config", default=str(get_default_config_dir()))
    match_parser.add_argument("--matcher", default="matchanything_eloftr")
    match_parser.add_argument("--models-dir")
    match_parser.add_argument("--output")
    match_parser.set_defaults(func=run_match)

    return parser


def collect_doctor_report(models_dir: str | Path | None = None) -> dict[str, Any]:
    resolved_models = get_models_dir(models_dir)
    expected_paths = get_expected_weight_paths(resolved_models)
    dependency_names = ["cv2", "torch", "gradio", "fastapi", "uvicorn", "yaml", "gdown", "numpy"]
    dependencies = {}
    for name in dependency_names:
        try:
            __import__(name)
            dependencies[name] = "ok"
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            dependencies[name] = f"missing: {exc.__class__.__name__}"

    torch_info: dict[str, Any] = {"available": False}
    try:
        import torch

        torch_info = {
            "available": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        torch_info["error"] = str(exc)

    return {
        "repo_root": str(get_repo_root()),
        "runtime_home": str(get_runtime_home()),
        "models_dir": str(resolved_models),
        "weights": {name: path.exists() for name, path in expected_paths.items()},
        "dependencies": dependencies,
        "torch": torch_info,
    }


def run_doctor(args: argparse.Namespace) -> int:
    report = collect_doctor_report(args.models_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"repo_root: {report['repo_root']}")
    print(f"runtime_home: {report['runtime_home']}")
    print(f"models_dir: {report['models_dir']}")
    print("weights:")
    for name, present in report["weights"].items():
        status = "ok" if present else "missing"
        print(f"  - {name}: {status}")
    print("dependencies:")
    for name, status in report["dependencies"].items():
        print(f"  - {name}: {status}")
    torch_info = report["torch"]
    if torch_info.get("available"):
        print(
            "torch: ok"
            f" | cuda_available={torch_info.get('cuda_available')}"
            f" | device_count={torch_info.get('device_count')}"
        )
    else:
        print(f"torch: {torch_info.get('error', 'missing')}")
    return 0


def run_setup(args: argparse.Namespace) -> int:
    models_dir = _set_models_dir(args.models_dir)
    if args.check:
        expected_paths = get_expected_weight_paths(models_dir)
        missing = [name for name, path in expected_paths.items() if not path.exists()]
        if missing:
            print("missing model assets:")
            for name in missing:
                print(f"  - {name}")
            return 1
        print(f"all expected model assets are present in {models_dir}")
        return 0

    archive_path = models_dir / "weights.zip"
    if archive_path.exists() and not args.force:
        print(f"reusing existing archive: {archive_path}")
    else:
        try:
            import gdown
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("gdown is required for `matchanything setup`") from exc
        if archive_path.exists():
            archive_path.unlink()
        gdown.download(
            id=MODEL_ARCHIVE_FILE_ID,
            output=str(archive_path),
            quiet=False,
            fuzzy=True,
        )

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(models_dir)

    expected_paths = get_expected_weight_paths(models_dir)
    missing = [name for name, path in expected_paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Downloaded archive did not produce expected weights: "
            + ", ".join(missing)
        )
    print(f"model assets are ready in {models_dir}")
    return 0


def run_ui(args: argparse.Namespace) -> int:
    _set_models_dir(args.models_dir)
    from .ui.app_class import ImageMatchingApp

    app = ImageMatchingApp(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        config=args.config,
    )
    app.run()
    return 0


def run_api(args: argparse.Namespace) -> int:
    _set_models_dir(args.models_dir)
    import uvicorn

    from .api.server import create_app

    app = create_app(
        config=args.config,
        default_matcher=args.default_matcher,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def run_match(args: argparse.Namespace) -> int:
    _set_models_dir(args.models_dir)
    from copy import deepcopy

    import numpy as np
    from PIL import Image

    from .api.core import ImageMatchingAPI
    from .hloc import DEVICE
    from .ui.utils import get_matcher_zoo

    config = load_runtime_config(args.config)
    matcher_zoo = get_matcher_zoo(config["matcher_zoo"])
    if args.matcher not in matcher_zoo:
        raise SystemExit(f"unknown matcher `{args.matcher}`")

    image0 = np.array(Image.open(args.image0).convert("RGB"))
    image1 = np.array(Image.open(args.image1).convert("RGB"))
    api = ImageMatchingAPI(conf=deepcopy(matcher_zoo[args.matcher]), device=str(DEVICE))
    prediction = api(image0, image1)
    serializable = {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in prediction.items()
        if key not in {"image0_orig", "image1_orig"}
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(output_path)
    else:
        print(json.dumps(serializable, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def main_ui() -> int:
    return main(["ui", *sys.argv[1:]])


def main_api() -> int:
    return main(["api", *sys.argv[1:]])


def main_setup() -> int:
    return main(["setup", *sys.argv[1:]])


def main_doctor() -> int:
    return main(["doctor", *sys.argv[1:]])
