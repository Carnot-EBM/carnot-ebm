"""Tests for scripts/check_spec_coverage.py.

Spec: REQ-REPORT-037, SCENARIO-REPORT-037.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_checker():
    checker_path = Path(__file__).resolve().parents[2] / "scripts" / "check_spec_coverage.py"
    spec = importlib.util.spec_from_file_location("check_spec_coverage", checker_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_req_report_037_spec_pattern_accepts_multisegment_ids() -> None:
    """REQ-REPORT-037: checker recognizes canonical multi-part OpenSpec ids."""
    checker = _load_checker()

    accepted = [
        "REQ-INFER-SOTA-004",
        "SCENARIO-INFER-SOTA-004-001",
        "REQ-VER-MATH-001",
        "SCENARIO-VER-MATH-001",
        "REQ-KAN-VERIFY-001",
        "SCENARIO-KAN-VERIFY-001",
        "REQ-PUBLISH-010",
        "REQ-INFRA-046b",
        "SCENARIO-INFRA-055b",
    ]

    for token in accepted:
        assert checker.SPEC_PATTERN.fullmatch(token)

    assert checker.SPEC_PATTERN.fullmatch("REQ-INFER-SOTA") is None
    assert checker.SPEC_PATTERN.fullmatch("SCENARIO--001") is None


def test_req_report_037_python_file_check_accepts_multisegment_file_metadata(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-037: file-level multi-part metadata covers tests."""
    checker = _load_checker()
    test_file = tmp_path / "test_generated_traceability.py"
    test_file.write_text(
        '"""Spec: REQ-INFER-SOTA-004, SCENARIO-INFER-SOTA-004-001."""\n\n'
        "def test_generated_case():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    assert checker.check_python_files([test_file]) == []
