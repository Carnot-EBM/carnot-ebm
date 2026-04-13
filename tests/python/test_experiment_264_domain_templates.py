"""Tests for experiment_264_domain_templates — domain constraint template mining.

Validates: template schema shape, precision threshold enforcement (>= 0.50),
and deterministic generation from the same corpus.

Spec: REQ-CONSTRAINT-264-A, REQ-CONSTRAINT-264-B, REQ-CONSTRAINT-264-C
"""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Import the module-under-test via path so it runs standalone without install
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "experiment_264_domain_templates.py"


def _load_module():
    """Load experiment_264_domain_templates as a module from its path."""
    spec = importlib.util.spec_from_file_location("exp264", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, "Cannot find script"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# Helpers: build synthetic corpus rows for unit testing
# ---------------------------------------------------------------------------

def _row(
    partial_response: str,
    violation_label: bool,
    model: str = "Qwen3.5-0.8B",
    domain: str = "reasoning",
    case_id: str = "c0",
) -> dict[str, Any]:
    """Minimal corpus row for testing pattern mining."""
    return {
        "case_id": case_id,
        "domain": domain,
        "model": model,
        "partial_response": partial_response,
        "violation_label": violation_label,
        "token_pattern_features": {"digit_density": 0.1, "equals_count": 0, "operator_count": 0, "sentence_count": 1},
        "experiment": 262,
        "run_date": "20260413",
    }


# ---------------------------------------------------------------------------
# Schema shape tests (REQ-CONSTRAINT-264-A)
# ---------------------------------------------------------------------------


class TestTemplateSchemaShape:
    """Every template record must carry all required fields with correct types."""

    REQUIRED_STR_FIELDS = {
        "domain",
        "token_pattern_regex",
        "associated_claim_route",
        "model_specificity",
        "run_date",
    }
    REQUIRED_FLOAT_FIELDS = {"corpus_precision", "corpus_recall"}
    REQUIRED_INT_FIELDS = {"n_positive_cases", "n_negative_cases", "experiment"}

    def _validate_template(self, rec: dict[str, Any]) -> None:
        for f in self.REQUIRED_STR_FIELDS:
            assert f in rec, f"Missing field: {f}"
            assert isinstance(rec[f], str), f"Field {f} must be str, got {type(rec[f])}"
        for f in self.REQUIRED_FLOAT_FIELDS:
            assert f in rec, f"Missing field: {f}"
            assert isinstance(rec[f], float), f"Field {f} must be float, got {type(rec[f])}"
        for f in self.REQUIRED_INT_FIELDS:
            assert f in rec, f"Missing field: {f}"
            assert isinstance(rec[f], int), f"Field {f} must be int, got {type(rec[f])}"

    def test_make_template_record_arithmetic(self, mod):
        """_make_template_record returns all required fields for arithmetic domain."""
        rec = mod._make_template_record(
            domain="arithmetic",
            regex=r"\b\d+\s*\+\s*\d+",
            route="arithmetic",
            precision=0.65,
            recall=0.40,
            model_spec="both",
            n_pos=20,
            n_neg=11,
        )
        self._validate_template(rec)
        assert rec["domain"] == "arithmetic"
        assert rec["associated_claim_route"] == "arithmetic"
        assert rec["corpus_precision"] == 0.65
        assert rec["model_specificity"] == "both"
        assert rec["experiment"] == 264
        assert rec["run_date"] == "20260413"

    def test_make_template_record_cardinality(self, mod):
        """_make_template_record returns all required fields for cardinality domain."""
        rec = mod._make_template_record(
            domain="cardinality",
            regex=r"\b(total|each|sum)\b",
            route="cardinality",
            precision=0.58,
            recall=0.30,
            model_spec="Gemma4-E4B-it",
            n_pos=15,
            n_neg=11,
        )
        self._validate_template(rec)
        assert rec["domain"] == "cardinality"
        assert rec["model_specificity"] == "Gemma4-E4B-it"

    def test_make_template_record_set_membership(self, mod):
        """_make_template_record returns all required fields for set_membership domain."""
        rec = mod._make_template_record(
            domain="set_membership",
            regex=r"\b(including|such as)\b",
            route="set_membership",
            precision=0.55,
            recall=0.20,
            model_spec="Qwen3.5-0.8B",
            n_pos=8,
            n_neg=7,
        )
        self._validate_template(rec)
        assert rec["domain"] == "set_membership"


# ---------------------------------------------------------------------------
# Precision threshold enforcement (REQ-CONSTRAINT-264-B)
# ---------------------------------------------------------------------------


class TestPrecisionThreshold:
    """corpus_precision < 0.50 must be excluded from output."""

    def test_pattern_stats_precision_correct(self, mod):
        """_pattern_stats computes correct precision and recall."""
        rows = [
            _row("The total is 4 + 3 = 7", violation_label=True, case_id="c1"),
            _row("The total is 4 + 3 = 7", violation_label=True, case_id="c2"),
            _row("The total is 4 + 3 = 7", violation_label=False, case_id="c3"),
            _row("Nothing here", violation_label=True, case_id="c4"),
        ]
        # Pattern matches c1,c2,c3 (3 of 4 rows). Positive matches: 2. Negative matches: 1.
        prec, recall, n_pos, n_neg = mod._pattern_stats(r"\b\d+\s*\+\s*\d+", rows)
        assert abs(prec - (2 / 3)) < 1e-9, f"Expected precision=2/3, got {prec}"
        assert abs(recall - (2 / 3)) < 1e-9, f"Expected recall=2/3 (2 of 3 positives matched), got {recall}"
        assert n_pos == 2
        assert n_neg == 1

    def test_pattern_stats_no_matches_returns_zeros(self, mod):
        """_pattern_stats returns 0.0 precision/recall when pattern matches nothing."""
        rows = [_row("plain text no numbers", violation_label=True, case_id="c1")]
        prec, recall, n_pos, n_neg = mod._pattern_stats(r"\b\d{4}\b", rows)
        assert prec == 0.0
        assert recall == 0.0
        assert n_pos == 0
        assert n_neg == 0

    def test_mine_templates_enforces_precision_floor(self, mod):
        """mine_templates excludes templates with corpus_precision < 0.50."""
        # Pattern r"\bx\b" matches only violation=False rows → precision 0.0
        rows = [
            _row("x marks the spot", violation_label=False, case_id="c1"),
            _row("x marks the spot", violation_label=False, case_id="c2"),
            _row("nothing", violation_label=True, case_id="c3"),
        ]
        patterns = [
            ("arithmetic", r"\bx\b", "arithmetic"),
        ]
        templates = mod.mine_templates(rows, patterns, min_precision=0.50, min_matches=1)
        assert len(templates) == 0, f"Expected 0 templates above threshold, got {len(templates)}"

    def test_mine_templates_keeps_templates_above_threshold(self, mod):
        """mine_templates keeps templates with corpus_precision >= 0.50."""
        rows = [
            _row("4 + 3 = 7", violation_label=True, case_id="c1"),
            _row("4 + 3 = 7", violation_label=True, case_id="c2"),
            _row("4 + 3 = 7", violation_label=False, case_id="c3"),
        ]
        patterns = [
            ("arithmetic", r"\b\d+\s*\+\s*\d+", "arithmetic"),
        ]
        templates = mod.mine_templates(rows, patterns, min_precision=0.50, min_matches=1)
        assert len(templates) == 1
        assert templates[0]["corpus_precision"] >= 0.50
        assert templates[0]["corpus_precision"] == pytest.approx(2 / 3)

    def test_mine_templates_enforces_min_matches(self, mod):
        """mine_templates excludes templates that match fewer than min_matches rows."""
        rows = [_row("4 + 3", violation_label=True, case_id="c1")]
        patterns = [("arithmetic", r"\b\d+\s*\+\s*\d+", "arithmetic")]
        # Require at least 5 matches — only 1 match exists
        templates = mod.mine_templates(rows, patterns, min_precision=0.50, min_matches=5)
        assert len(templates) == 0


# ---------------------------------------------------------------------------
# Deterministic generation (REQ-CONSTRAINT-264-C)
# ---------------------------------------------------------------------------


class TestDeterministicGeneration:
    """Running the miner twice on the same input must produce identical output."""

    def test_mine_templates_deterministic(self, mod):
        """mine_templates produces identical output on two successive calls."""
        rows = [
            _row("4 + 3 = 7 and 10 / 2 = 5", violation_label=True, case_id="c1"),
            _row("total = 4 + 3", violation_label=True, case_id="c2"),
            _row("simple answer", violation_label=False, case_id="c3"),
        ]
        patterns = [
            ("arithmetic", r"\b\d+\s*\+\s*\d+", "arithmetic"),
            ("arithmetic", r"\b\d+\s*/\s*\d+", "arithmetic"),
        ]
        first = mod.mine_templates(rows, patterns, min_precision=0.0, min_matches=1)
        second = mod.mine_templates(rows, patterns, min_precision=0.0, min_matches=1)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_domain_patterns_constant(self, mod):
        """DOMAIN_PATTERNS is a fixed-length mapping — not stochastic."""
        dp = mod.DOMAIN_PATTERNS
        assert isinstance(dp, list), "DOMAIN_PATTERNS must be a list"
        assert len(dp) > 0
        # Each entry is (domain, regex, route) triple
        for entry in dp:
            assert len(entry) == 3, f"Each pattern entry must be 3-tuple, got {entry!r}"
            domain, regex, route = entry
            assert domain in {"arithmetic", "cardinality", "set_membership"}
            assert isinstance(regex, str)
            assert route in {"arithmetic", "cardinality", "set_membership"}

    def test_model_specificity_returns_valid_value(self, mod):
        """_model_specificity returns 'both', or a known model name."""
        rows = [
            _row("4 + 3", violation_label=True, model="Qwen3.5-0.8B", case_id="c1"),
            _row("4 + 3", violation_label=True, model="Gemma4-E4B-it", case_id="c2"),
            _row("4 + 3", violation_label=False, model="Qwen3.5-0.8B", case_id="c3"),
        ]
        result = mod._model_specificity(r"\b\d+\s*\+\s*\d+", rows)
        assert result in {"both", "Qwen3.5-0.8B", "Gemma4-E4B-it"}, f"Unexpected: {result!r}"


# ---------------------------------------------------------------------------
# Output artifact shape (integration-level smoke test)
# ---------------------------------------------------------------------------


class TestOutputArtifactShape:
    """Smoke-test that run_mining returns a summary dict with required keys."""

    def test_run_mining_returns_summary_structure(self, mod):
        """run_mining returns dict with template_counts, precision_stats, model_stats."""
        rows = [
            _row("4 + 3 = 7", violation_label=True, case_id="c1"),
            _row("8 - 2 = 6", violation_label=True, case_id="c2"),
            _row("plain text", violation_label=False, case_id="c3"),
        ]
        templates, summary = mod.run_mining(rows, min_precision=0.0, min_matches=1)
        assert "template_counts_by_domain" in summary
        assert "precision_stats" in summary
        assert "model_stats" in summary
        assert "experiment" in summary
        assert summary["experiment"] == 264
        assert isinstance(templates, list)
        for rec in templates:
            assert "domain" in rec
            assert "corpus_precision" in rec
            assert rec["corpus_precision"] >= 0.0
