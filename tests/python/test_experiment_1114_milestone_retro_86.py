"""Tests for ``scripts/experiment_1114_milestone_retro_86.py``.

Spec: REQ-OPS-001 — Milestone retrospective produces a valid artifact with
      all required schema fields and accurate criterion evaluation.

WHY THESE TESTS:
    - ``evaluate_criteria`` is the load-bearing function; a silent regression
      there means the conductor sees the wrong criterion count and makes
      incorrect archiving/planning decisions.  We pin it against synthetic
      fixture dicts so any future refactor that changes the evaluation logic
      is immediately caught.
    - ``build_slowest_5`` is tested for structure only (5 entries, required
      keys) since the slowest-5 ranking is derived from a hardcoded analysis
      of conductor-log timestamps rather than from parsed data.
    - The artifact JSON is tested for required schema fields and value
      constraints so the conductor's post-run JSON-schema validation passes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1114_milestone_retro_86.py"
ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_1114_milestone_retro_86.json"


def _load_module():
    """Load the retro script as a module without requiring it to be installed."""
    spec = importlib.util.spec_from_file_location("exp1114", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1114"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp1114():
    return _load_module()


@pytest.fixture(scope="module")
def artifact():
    """Load the on-disk artifact produced by running the script."""
    assert ARTIFACT_PATH.exists(), f"Artifact not found: {ARTIFACT_PATH}"
    with ARTIFACT_PATH.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# evaluate_criteria unit tests
# ---------------------------------------------------------------------------


def _make_fixtures():
    """Return minimal passing fixture dicts for all 10 source experiments."""
    e1104 = {
        "failure_ledger_id_fix_deployed": True,
        "manifest_dispatch_enforcement_deployed": True,
    }
    e1105 = {
        "failure_ledger_cap_reset_deployed": True,
        "stable_deliverable_mtime_fix_deployed": True,
    }
    e1106 = {"phase1a_false_pass_below_5pct": True}
    e1107 = {"new_diverse_verifiers_deployed_3_verifiers": True}
    e1108_pass = {"and_composition_viable_at_k6": True}
    e1108_fail = {"and_composition_viable_at_k6": False}
    e1109 = {"kv260_v3_kl_measured_below_threshold": True}
    e1110 = {"rlvr_ssd_v2_non_degenerate_honest_result": True}
    e1111 = {"thinkprm_v2_auroc_above_099": True}
    e1112 = {
        "llm_failure_exemplar_corpus_30_exemplars": True,
        "goodfire_cascade_tp_rate_measured": True,
    }
    e1113 = {"arxiv_bundle_complete": True}
    return e1104, e1105, e1106, e1107, e1108_pass, e1108_fail, e1109, e1110, e1111, e1112, e1113


def test_evaluate_criteria_all_pass(exp1114):
    """All criteria pass when every field is True (including k6 AND-composition)."""
    e1104, e1105, e1106, e1107, e1108_pass, _, e1109, e1110, e1111, e1112, e1113 = _make_fixtures()
    result = exp1114.evaluate_criteria(
        e1104,
        e1105,
        e1106,
        e1107,
        e1108_pass,
        e1109,
        e1110,
        e1111,
        e1112,
        e1113,
    )
    assert all(result.values()), f"Expected all True, got: {result}"
    assert len(result) == 12


def test_evaluate_criteria_and_composition_false(exp1114):
    """Criterion 5 is False when and_composition_viable_at_k6 is False."""
    e1104, e1105, e1106, e1107, _, e1108_fail, e1109, e1110, e1111, e1112, e1113 = _make_fixtures()
    result = exp1114.evaluate_criteria(
        e1104,
        e1105,
        e1106,
        e1107,
        e1108_fail,
        e1109,
        e1110,
        e1111,
        e1112,
        e1113,
    )
    assert result["and_composition_viable_r_corr_below_05"] is False
    # All other criteria should still pass.
    others = {k: v for k, v in result.items() if k != "and_composition_viable_r_corr_below_05"}
    assert all(others.values())


def test_evaluate_criteria_retro_always_true(exp1114):
    """Criterion 12 (retro_complete) is True regardless of other inputs."""
    empties = [{} for _ in range(10)]
    result = exp1114.evaluate_criteria(*empties)
    assert result["retro_complete"] is True


def test_evaluate_criteria_returns_12_keys(exp1114):
    empties = [{} for _ in range(10)]
    result = exp1114.evaluate_criteria(*empties)
    assert len(result) == 12


# ---------------------------------------------------------------------------
# build_slowest_5 structure tests
# ---------------------------------------------------------------------------


def test_build_slowest_5_returns_5_entries(exp1114):
    result = exp1114.build_slowest_5([])
    assert len(result) == 5


def test_build_slowest_5_has_required_keys(exp1114):
    required_keys = {"rank", "id", "title", "duration_min", "diagnosis"}
    for entry in exp1114.build_slowest_5([]):
        missing = required_keys - entry.keys()
        assert not missing, f"Entry {entry.get('id')} missing keys: {missing}"


def test_build_slowest_5_ranks_are_1_through_5(exp1114):
    ranks = [e["rank"] for e in exp1114.build_slowest_5([])]
    assert ranks == [1, 2, 3, 4, 5]


def test_build_slowest_5_durations_positive(exp1114):
    for entry in exp1114.build_slowest_5([]):
        assert entry["duration_min"] > 0, f"Non-positive duration for {entry['id']}"


# ---------------------------------------------------------------------------
# Artifact JSON schema tests
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACT_FIELDS = {
    "experiment",
    "title",
    "milestone",
    "run_date",
    "started_at",
    "finished_at",
    "schema",
    "criteria_results",
    "criteria_met",
    "criteria_total",
    "criteria_pct",
    "slowest_experiments",
    "bottlenecks_identified",
    "improvements_suggested",
    "dualgpu_consecutive_idle_count",
    "retro_complete",
    "honest_verdict",
}


def test_artifact_has_all_required_fields(artifact):
    missing = REQUIRED_ARTIFACT_FIELDS - artifact.keys()
    assert not missing, f"Missing required artifact fields: {missing}"


def test_artifact_criteria_total_is_12(artifact):
    assert artifact["criteria_total"] == 12


def test_artifact_criteria_met_consistent_with_results(artifact):
    computed = sum(1 for v in artifact["criteria_results"].values() if v)
    assert artifact["criteria_met"] == computed


def test_artifact_criteria_pct_matches_met_over_total(artifact):
    expected = round(100.0 * artifact["criteria_met"] / artifact["criteria_total"], 1)
    assert artifact["criteria_pct"] == expected


def test_artifact_retro_complete_true(artifact):
    assert artifact["retro_complete"] is True


def test_artifact_honest_verdict_known_value(artifact):
    assert artifact["honest_verdict"] in {
        "all_criteria_met",
        "strong_milestone_one_criterion_missed",
        "partial_milestone_majority_met",
        "weak_milestone_majority_missed",
    }


def test_artifact_slowest_experiments_has_5(artifact):
    assert len(artifact["slowest_experiments"]) == 5


def test_artifact_bottlenecks_nonempty(artifact):
    assert len(artifact["bottlenecks_identified"]) >= 1


def test_artifact_improvements_nonempty(artifact):
    assert len(artifact["improvements_suggested"]) >= 1


def test_artifact_dualgpu_count_decreased_from_18(artifact):
    # .85 ended with 18 consecutive idle; .86 must show improvement.
    assert artifact["dualgpu_consecutive_idle_count"] < 18
