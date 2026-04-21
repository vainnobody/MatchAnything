import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_module(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MATCHANYTHING_HOME"] = str(ROOT / ".tmp-test-home")
    return subprocess.run(
        [sys.executable, "-m", "matchanything", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_matchanything_has_no_repo_side_effects():
    result = run_module("--help")
    assert result.returncode == 0
    assert not (ROOT / "log.txt").exists()


@pytest.mark.parametrize(
    "args",
    [
        ("--help",),
        ("ui", "--help"),
        ("api", "--help"),
        ("setup", "--help"),
        ("doctor", "--help"),
        ("match", "--help"),
    ],
)
def test_cli_help(args):
    result = run_module(*args)
    assert result.returncode == 0, result.stderr


def test_doctor_json():
    result = run_module("doctor", "--json")
    assert result.returncode == 0, result.stderr
    assert '"models_dir"' in result.stdout


def test_import_imcui_is_compatible():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", "import imcui; print(imcui.__version__)"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
