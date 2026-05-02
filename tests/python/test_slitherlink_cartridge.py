"""Pytest collector for the Space-local Slitherlink cartridge tests.

Spec traces: REQ-SLITHERLINK-001 and REQ-SAMPLE-003.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SPACE_TEST = (
    Path(__file__).resolve().parents[2] / "spaces" / "wopr-games" / "tests" / "test_slitherlink.py"
)
_SPEC = importlib.util.spec_from_file_location("wopr_slitherlink_space_tests", _SPACE_TEST)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load Slitherlink tests from {_SPACE_TEST}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if _name.startswith("test_"):
        globals()[_name] = getattr(_MODULE, _name)
