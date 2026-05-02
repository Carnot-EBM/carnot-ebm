"""Compatibility wrapper for the Space-local Slitherlink cartridge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_WOPR_DIR = Path(__file__).resolve().parents[2] / "wopr-games"
if str(_WOPR_DIR) not in sys.path:
    sys.path.insert(0, str(_WOPR_DIR))

_REAL_FILE = _WOPR_DIR / "games" / "slitherlink.py"
_SPEC = importlib.util.spec_from_file_location("_wopr_games_slitherlink_real", _REAL_FILE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib guard
    raise ImportError(f"Could not load Slitherlink cartridge from {_REAL_FILE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

for _name in getattr(_MODULE, "__all__", ()):
    globals()[_name] = getattr(_MODULE, _name)

__all__ = list(getattr(_MODULE, "__all__", ()))
