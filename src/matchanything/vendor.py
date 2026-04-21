from __future__ import annotations

import sys
from pathlib import Path

from .runtime import get_vendor_root


def ensure_vendor_imports() -> Path:
    vendor_root = get_vendor_root()
    candidates = [vendor_root.parent, vendor_root]
    for path in candidates:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return vendor_root
