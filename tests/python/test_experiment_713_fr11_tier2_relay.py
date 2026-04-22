"""Tests for scripts/experiment_713_fr11_tier2_relay.py.

Covers:
- load_cascade_gate_status reads cascade_gate_open correctly (REQ-LEARN-022-1, REQ-LEARN-023-1)
- Source selection: gate open → JEPA path; gate closed → Exp 694 fallback (SCENARIO-LEARN-022, SCENARIO-LEARN-023)
- extract_jepa_violations returns entries from cascade_violations list (REQ-LEARN-022-3)
- extract_jepa_violations returns sentinel when cascade_violations is absent (REQ-LEARN-022-3)
- extract_exp694_fallback_violations builds one pattern per hard question (REQ-LEARN-023-3)
- extract_exp694_fallback_violations returns empty when signed_improvement <= 0
- run_experiment wires expected patterns into library (REQ-LEARN-022, REQ-LEARN-023)
- run_experiment honest_verdict is correct for both gate states (REQ-LEARN-022-4, REQ-LEARN-023-4)
- run_experiment fr11_tier_advancement == 2 always (REQ-LEARN-022, REQ-LEARN-023)
- ViolationEntry dataclass construction

Spec: REQ-LEARN-022, REQ-LEARN-023, SCENARIO-LEARN-022, SCENARIO-LEARN-023
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_713_fr11_tier2_relay import (
    ViolationEntry,
    build_benchmark_responses,
    extract_exp694_fallback_violations,
    extract_jepa_violations,
    load_cascade_gate_status,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def exp705_gate_open(tmp_path):
    """Exp 705 artifact with cascade_gate_open=True and one cascade violation."""
    artifact = {
        "experiment": 705,
        "cascade_gate_open": True,
        "cascade_violations": [
            {
                "constraint_type": "jepa_ood",
                "step_text_fragment": "JEPA_STEP: mismatch at position 3",
            }
        ],
        "status": "success",
    }
    p = tmp_path / "exp705_open.json"
    p.write_text(json.dumps(artifact))
    return p


@pytest.fixture()
def exp705_gate_closed(tmp_path):
    """Exp 705 artifact with cascade_gate_open=False (no cascade_violations key)."""
    artifact = {
        "experiment": 705,
        "cascade_gate_open": False,
        "status": "success",
    }
    p = tmp_path / "exp705_closed.json"
    p.write_text(json.dumps(artifact))
    return p


@pytest.fixture()
def exp694_standard(tmp_path):
    """Exp 694 artifact with qwen_signed_improvement=1.0, n_hard_questions=10."""
    artifact = {
        "experiment": 694,
        "qwen_signed_improvement": 1.0,
        "n_hard_questions": 10,
        "status": "success",
    }
    p = tmp_path / "exp694.json"
    p.write_text(json.dumps(artifact))
    return p


@pytest.fixture()
def exp694_no_improvement(tmp_path):
    """Exp 694 artifact with qwen_signed_improvement=0.0 (no improvement)."""
    artifact = {
        "experiment": 694,
        "qwen_signed_improvement": 0.0,
        "n_hard_questions": 10,
        "status": "success",
    }
    p = tmp_path / "exp694_bad.json"
    p.write_text(json.dumps(artifact))
    return p


# ---------------------------------------------------------------------------
# ViolationEntry dataclass
# ---------------------------------------------------------------------------


def test_violation_entry_fields():
    """ViolationEntry holds all three fields with correct types."""
    entry = ViolationEntry(
        constraint_type="arithmetic",
        pattern="COMPUTE: 5 + 7 = 12",
        source_label="jepa_v17_cascade",
    )
    assert entry.constraint_type == "arithmetic"
    assert entry.pattern == "COMPUTE: 5 + 7 = 12"
    assert entry.source_label == "jepa_v17_cascade"


# ---------------------------------------------------------------------------
# load_cascade_gate_status
# ---------------------------------------------------------------------------


def test_load_gate_status_open(exp705_gate_open):
    """Returns True when cascade_gate_open=True in artifact. (REQ-LEARN-022-1)"""
    assert load_cascade_gate_status(exp705_gate_open) is True


def test_load_gate_status_closed(exp705_gate_closed):
    """Returns False when cascade_gate_open=False in artifact. (REQ-LEARN-023-1)"""
    assert load_cascade_gate_status(exp705_gate_closed) is False


def test_load_gate_status_missing_key(tmp_path):
    """Returns False when cascade_gate_open key is absent (safe default)."""
    p = tmp_path / "exp705_minimal.json"
    p.write_text(json.dumps({"experiment": 705}))
    assert load_cascade_gate_status(p) is False


# ---------------------------------------------------------------------------
# extract_jepa_violations
# ---------------------------------------------------------------------------


def test_extract_jepa_violations_from_artifact(exp705_gate_open):
    """Reads cascade_violations list and returns one entry per violation. (REQ-LEARN-022-3)"""
    entries = extract_jepa_violations(exp705_gate_open)
    assert len(entries) == 1
    assert entries[0].constraint_type == "jepa_ood"
    assert "mismatch at position 3" in entries[0].pattern


def test_extract_jepa_violations_multiple(tmp_path):
    """Multiple cascade violations produce multiple ViolationEntry objects."""
    artifact = {
        "cascade_gate_open": True,
        "cascade_violations": [
            {"constraint_type": "jepa_ood", "step_text_fragment": "step A error"},
            {"constraint_type": "arithmetic", "step_text_fragment": "carry error"},
        ],
    }
    p = tmp_path / "exp705_multi.json"
    p.write_text(json.dumps(artifact))
    entries = extract_jepa_violations(p)
    assert len(entries) == 2


def test_extract_jepa_violations_sentinel_when_empty(exp705_gate_closed):
    """Returns a sentinel entry when cascade_violations key is absent. (REQ-LEARN-022-3)"""
    entries = extract_jepa_violations(exp705_gate_closed)
    assert len(entries) == 1
    assert "jepa_v17_cascade" in entries[0].source_label


def test_extract_jepa_violations_skips_empty_pattern(tmp_path):
    """Violations with blank step_text_fragment are skipped; sentinel fills the gap."""
    artifact = {
        "cascade_violations": [
            {"constraint_type": "jepa_ood", "step_text_fragment": "   "},
        ]
    }
    p = tmp_path / "exp705_blank.json"
    p.write_text(json.dumps(artifact))
    entries = extract_jepa_violations(p)
    # blank pattern skipped → falls through to sentinel
    assert len(entries) == 1
    assert "sentinel" in entries[0].source_label


def test_extract_jepa_violations_source_label(exp705_gate_open):
    """Entry source_label references 'jepa_v17_cascade'."""
    entries = extract_jepa_violations(exp705_gate_open)
    assert "jepa_v17_cascade" in entries[0].source_label


# ---------------------------------------------------------------------------
# extract_exp694_fallback_violations
# ---------------------------------------------------------------------------


def test_extract_exp694_n_entries(exp694_standard):
    """Returns one entry per hard question when signed_improvement > 0. (REQ-LEARN-023-3)"""
    entries = extract_exp694_fallback_violations(exp694_standard)
    assert len(entries) == 10  # n_hard_questions=10


def test_extract_exp694_constraint_type(exp694_standard):
    """All fallback entries have constraint_type='arithmetic'. (REQ-LEARN-023-2)"""
    entries = extract_exp694_fallback_violations(exp694_standard)
    assert all(e.constraint_type == "arithmetic" for e in entries)


def test_extract_exp694_source_label(exp694_standard):
    """All fallback entries have source_label='exp694_qwen_fallback'."""
    entries = extract_exp694_fallback_violations(exp694_standard)
    assert all(e.source_label == "exp694_qwen_fallback" for e in entries)


def test_extract_exp694_pattern_contains_synthetic_marker(exp694_standard):
    """Each pattern contains 'synthetic_agg_pattern' marker for audit trail."""
    entries = extract_exp694_fallback_violations(exp694_standard)
    assert all("synthetic_agg_pattern" in e.pattern for e in entries)


def test_extract_exp694_no_improvement_returns_empty(exp694_no_improvement):
    """Returns empty list when signed_improvement <= 0 (no confirmed repairs)."""
    entries = extract_exp694_fallback_violations(exp694_no_improvement)
    assert entries == []


def test_extract_exp694_indexed_patterns(exp694_standard):
    """Pattern strings are indexed (q000, q001, ...) for uniqueness."""
    entries = extract_exp694_fallback_violations(exp694_standard)
    patterns = [e.pattern for e in entries]
    assert "q000" in patterns[0]
    assert "q001" in patterns[1]


# ---------------------------------------------------------------------------
# build_benchmark_responses
# ---------------------------------------------------------------------------


def test_benchmark_responses_count():
    """Returns exactly 10 benchmark responses."""
    responses = build_benchmark_responses()
    assert len(responses) == 10


def test_benchmark_responses_are_strings():
    """All benchmark responses are non-empty strings."""
    responses = build_benchmark_responses()
    assert all(isinstance(r, str) and len(r) > 0 for r in responses)


def test_benchmark_responses_no_synthetic_markers():
    """Benchmark responses do not contain 'synthetic_agg_pattern' (avoids false positives)."""
    responses = build_benchmark_responses()
    assert not any("synthetic_agg_pattern" in r for r in responses)


# ---------------------------------------------------------------------------
# run_experiment — gate open path (SCENARIO-LEARN-022)
# ---------------------------------------------------------------------------


def test_run_experiment_gate_open_source(exp705_gate_open, exp694_standard, tmp_path):
    """source == 'jepa_v17_cascade_violations' when gate is open. (SCENARIO-LEARN-022)"""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_open,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    assert result["source"] == "jepa_v17_cascade_violations"


def test_run_experiment_gate_open_verdict(exp705_gate_open, exp694_standard, tmp_path):
    """honest_verdict == 'fr11_tier2_real_violations' when gate is open. (REQ-LEARN-022-4)"""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_open,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    assert result["honest_verdict"] == "fr11_tier2_real_violations"


def test_run_experiment_gate_open_tier_advancement(exp705_gate_open, exp694_standard, tmp_path):
    """fr11_tier_advancement == 2 when gate is open. (SCENARIO-LEARN-022)"""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_open,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    assert result["fr11_tier_advancement"] == 2


def test_run_experiment_gate_open_patterns_added(exp705_gate_open, exp694_standard, tmp_path):
    """n_patterns_added > 0 when gate is open and violations exist. (REQ-LEARN-022-3)"""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_open,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    assert result["n_patterns_added"] > 0


# ---------------------------------------------------------------------------
# run_experiment — gate closed path (SCENARIO-LEARN-023)
# ---------------------------------------------------------------------------


def test_run_experiment_gate_closed_source(exp705_gate_closed, exp694_standard, tmp_path):
    """source == 'exp694_qwen_fallback' when gate is closed. (SCENARIO-LEARN-023)"""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_closed,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    assert result["source"] == "exp694_qwen_fallback"


def test_run_experiment_gate_closed_verdict(exp705_gate_closed, exp694_standard, tmp_path):
    """honest_verdict == 'fr11_tier2_fallback_relay' when gate is closed. (REQ-LEARN-023-4)"""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_closed,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    assert result["honest_verdict"] == "fr11_tier2_fallback_relay"


def test_run_experiment_gate_closed_tier_advancement(exp705_gate_closed, exp694_standard, tmp_path):
    """fr11_tier_advancement == 2 when gate is closed. (SCENARIO-LEARN-023)"""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_closed,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    assert result["fr11_tier_advancement"] == 2


def test_run_experiment_gate_closed_n_violations(exp705_gate_closed, exp694_standard, tmp_path):
    """n_violations > 0 when fallback Exp 694 has signed_improvement > 0. (SCENARIO-LEARN-023)"""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_closed,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    assert result["n_violations"] > 0


# ---------------------------------------------------------------------------
# run_experiment — artifact completeness
# ---------------------------------------------------------------------------


def test_run_experiment_required_fields(exp705_gate_closed, exp694_standard, tmp_path):
    """Artifact contains all required output fields."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_closed,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    required = [
        "source",
        "n_violations",
        "n_patterns_added",
        "n_patterns_total",
        "fp_rate_before",
        "fp_rate_after",
        "fp_rate_delta",
        "fr11_tier_advancement",
        "honest_verdict",
        "n_benchmark_queries",
        "cascade_gate_open",
    ]
    for field in required:
        assert field in result, f"Missing required field: {field}"


def test_run_experiment_fp_delta_equals_after_minus_before(exp705_gate_closed, exp694_standard, tmp_path):
    """fp_rate_delta == fp_rate_after - fp_rate_before (arithmetic check)."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_closed,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    expected_delta = round(result["fp_rate_after"] - result["fp_rate_before"], 6)
    assert abs(result["fp_rate_delta"] - expected_delta) < 1e-9


def test_run_experiment_n_patterns_total_consistent(exp705_gate_closed, exp694_standard, tmp_path):
    """n_patterns_total >= n_patterns_added (total is at least what was added)."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_closed,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    assert result["n_patterns_total"] >= result["n_patterns_added"]


def test_run_experiment_cascade_gate_open_recorded(exp705_gate_closed, exp694_standard, tmp_path):
    """cascade_gate_open field in result matches what was in Exp 705 artifact."""
    lib_path = str(tmp_path / "lib.json")
    result = run_experiment(
        exp705_path=exp705_gate_closed,
        exp694_path=exp694_standard,
        library_path=lib_path,
    )
    assert result["cascade_gate_open"] is False
