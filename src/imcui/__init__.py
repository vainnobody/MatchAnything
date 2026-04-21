"""Compatibility shim for the historical `imcui` package name."""

import matchanything as _matchanything
from matchanything import __version__

__path__ = list(_matchanything.__path__)

__all__ = ["__version__"]
