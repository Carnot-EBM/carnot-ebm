"""Pytest collector for the Space-local Masyu cartridge tests.

Spec traces: REQ-MASYU-001, SCENARIO-MASYU-001, and SCENARIO-MASYU-002.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SPACE_TEST = (
    Path(__file__).resolve().parents[2] / "spaces" / "wopr-games" / "tests" / "test_masyu.py"
)
_SPEC = importlib.util.spec_from_file_location("wopr_masyu_space_tests", _SPACE_TEST)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load Masyu tests from {_SPACE_TEST}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if _name.startswith("test_"):
        globals()[_name] = getattr(_MODULE, _name)
