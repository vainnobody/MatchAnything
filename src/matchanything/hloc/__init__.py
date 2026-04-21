import logging
from pathlib import Path

import torch

from ..runtime import ensure_runtime_dirs, get_logs_dir

__version__ = "1.5"

ensure_runtime_dirs()
LOG_PATH = get_logs_dir() / "matchanything.log"

formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)

file_handler = logging.FileHandler(filename=LOG_PATH)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler()
stdout_handler.setFormatter(formatter)
stdout_handler.setLevel(logging.INFO)
logger = logging.getLogger("matchanything")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
logger.propagate = False

try:
    import pycolmap
except ImportError:
    logger.warning("pycolmap is not installed, some geometry features may not work.")
else:
    min_version = "0.6.0"
    found_version = pycolmap.__version__
    if found_version != "dev" and found_version < min_version:
        logger.warning("pycolmap>=%s is recommended, found pycolmap==%s", min_version, found_version)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
