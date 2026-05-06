"""Tests for ``scripts/figure_integrity_audit.py``.

Spec anchor: REQ-PUBLISH-010 (the paper-v5 recompile gate must execute the
figure integrity audit and record untraced constants). The audit script is the
mechanical enforcement of that requirement; these tests pin its behaviour
against fabricated, traced, and edge-case constants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import figure_integrity_audit as fia  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_extract_numeric_constants_skips_trivials():
    """Constants below MIN_AUDIT_VALUE and known-trivial ints must be
    excluded. Otherwise the audit would drown in false positives from
    plot-styling values like alpha=0.5, dpi=300, linewidth=2.
    """
    src = "x = 0.5\ny = 100\nz = 24.83\nw = 1.0\n"
    constants = fia._extract_numeric_constants(src)
    values = {v for v, _ in constants}
    assert 24.83 in values
    assert 0.5 not in values
    assert 100 not in values  # in TRIVIAL_INTS
    assert 1.0 not in values


def test_value_appears_handles_int_and_float_forms():
    """JSON renders floats and ints differently; the audit must accept
    both ``"117"`` and ``"117.0"`` for a value of 117.0 in a figure script.
    """
    haystack = '{"latency_us": 117, "auroc": 0.9545}'
    assert fia._value_appears(117.0, haystack)
    assert fia._value_appears(117, haystack)
    assert not fia._value_appears(118.0, haystack)


def test_audit_flags_untraced_constant(tmp_path: Path):
    """An untraced constant in a figure script must surface in the audit
    report and drive a non-zero exit code path.
    """
    figures = tmp_path / "figures"
    results = tmp_path / "results"
    _write(figures / "fig_test.py", "import matplotlib\nvalue = 999.99\n")
    _write(results / "experiment_x.json", json.dumps({"unrelated": 1}))
    report = fia.audit(figures_dir=figures, results_dir=results)
    assert "fig_test.py" in report
    assert any(item["value"] == 999.99 for item in report["fig_test.py"])


def test_audit_passes_when_traced(tmp_path: Path):
    """A constant present in any results JSON must not be flagged."""
    figures = tmp_path / "figures"
    results = tmp_path / "results"
    _write(figures / "fig_ok.py", "x = 24.83\n")
    _write(results / "experiment_y.json", json.dumps({"latency_us": 24.83}))
    report = fia.audit(figures_dir=figures, results_dir=results)
    assert report == {}


def test_repository_audit_currently_passes():
    """End-to-end smoke test against the real repo. exp1205 fixed the
    paper-v5 critical issues; the audit must remain clean as a regression
    guard for future figure additions.
    """
    report = fia.audit()
    assert report == {}, fia._format_report(report)


def test_format_report_summarises_untraced():
    """The human-readable report must list each untraced constant with
    its line number so a reviewer can locate the offending code quickly.
    """
    report = {"fig_x.py": [{"value": 42.0, "line": 7}]}
    text = fia._format_report(report)
    assert "fig_x.py" in text
    assert "42.0" in text
    assert "line 7" in text
    assert "TOTAL UNTRACED: 1" in text


def test_format_report_clean_message():
    """When nothing is flagged, the message must be unambiguously OK."""
    assert fia._format_report({}).startswith("OK")


def test_main_returns_zero_when_clean():
    """The CLI entry point must exit 0 when the repo is clean, so it can
    be wired into pre-commit / CI without false positives.
    """
    assert fia.main() == 0
