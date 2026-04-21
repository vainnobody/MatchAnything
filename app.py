import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from matchanything.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["ui", *sys.argv[1:]]))
