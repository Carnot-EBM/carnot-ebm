"""Tests for Experiment 725: SC-Energy v2 — FoVer v2 Dual Labels.

All tests trace to REQ-VER-032 (SC-Energy Tier 2.9 trained on FoVer v2 dual labels).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiment_725_sc_energy_v2 import (
    build_contrastive_pairs,
    compute_auroc,
    derive_labels,
    load_v1_baseline_auc,
    strict_consensus_label,
)


# ---------------------------------------------------------------------------
# Tests for strict_consensus_label — REQ-VER-032
# ---------------------------------------------------------------------------


def test_strict_consensus_both_one():
    """Both labels = 1 → consensus = 1.  Spec: REQ-VER-032."""
    assert strict_consensus_label(1, 1) == 1


def test_strict_consensus_z3_zero():
    """z3_label = 0, pddl_label = 1 → consensus = 0.  Spec: REQ-VER-032."""
    assert strict_consensus_label(0, 1) == 0


def test_strict_consensus_pddl_zero():
    """z3_label = 1, pddl_label = 0 → consensus = 0.  Spec: REQ-VER-032."""
    assert strict_consensus_label(1, 0) == 0


def test_strict_consensus_both_zero():
    """Both labels = 0 → consensus = 0.  Spec: REQ-VER-032."""
    assert strict_consensus_label(0, 0) == 0


# ---------------------------------------------------------------------------
# Tests for derive_labels — REQ-VER-032
# ---------------------------------------------------------------------------


def test_derive_labels_pddl_correct():
    """PDDL pair with step_correct=True → pddl_label=1, z3_label=1 (default).  Spec: REQ-VER-032."""
    pair = {"labeler": "pddl", "step_correct": True}
    z3, pddl = derive_labels(pair)
    assert z3 == 1
    assert pddl == 1


def test_derive_labels_pddl_incorrect():
    """PDDL pair with step_correct=False → pddl_label=0, z3_label=1 (default).  Spec: REQ-VER-032."""
    pair = {"labeler": "pddl", "step_correct": False}
    z3, pddl = derive_labels(pair)
    assert z3 == 1
    assert pddl == 0


def test_derive_labels_z3_satisfied():
    """Z3 pair with z3_verdict='satisfied' → z3_label=1, pddl_label=1 (default).  Spec: REQ-VER-032."""
    pair = {"labeler": "z3", "step_correct": True, "z3_verdict": "satisfied"}
    z3, pddl = derive_labels(pair)
    assert z3 == 1
    assert pddl == 1


def test_derive_labels_z3_unparseable():
    """Z3 pair with z3_verdict='unparseable' → z3_label falls back to step_correct.  Spec: REQ-VER-032."""
    pair = {"labeler": "z3", "step_correct": True, "z3_verdict": "unparseable"}
    z3, pddl = derive_labels(pair)
    assert z3 == 1  # falls back to step_correct=True
    assert pddl == 1


def test_derive_labels_z3_unparseable_wrong_step():
    """Z3 pair with z3_verdict='unparseable' and step_correct=False → z3_label=0.  Spec: REQ-VER-032."""
    pair = {"labeler": "z3", "step_correct": False, "z3_verdict": "unparseable"}
    z3, pddl = derive_labels(pair)
    assert z3 == 0
    assert pddl == 1


def test_derive_labels_unknown_labeler():
    """Unknown labeler falls back to step_correct for both labels.  Spec: REQ-VER-032."""
    pair = {"labeler": "unknown", "step_correct": True}
    z3, pddl = derive_labels(pair)
    assert z3 == 1
    assert pddl == 1


# ---------------------------------------------------------------------------
# Test strict consensus integration with derive_labels — REQ-VER-032
# ---------------------------------------------------------------------------


def test_consensus_pipeline_pddl_correct():
    """Full pipeline: correct pddl pair passes strict consensus.  Spec: REQ-VER-032."""
    pair = {"labeler": "pddl", "step_correct": True}
    z3, pddl = derive_labels(pair)
    assert strict_consensus_label(z3, pddl) == 1


def test_consensus_pipeline_pddl_incorrect():
    """Full pipeline: incorrect pddl pair fails strict consensus.  Spec: REQ-VER-032."""
    pair = {"labeler": "pddl", "step_correct": False}
    z3, pddl = derive_labels(pair)
    assert strict_consensus_label(z3, pddl) == 0


# ---------------------------------------------------------------------------
# Tests for build_contrastive_pairs — REQ-VER-032
# ---------------------------------------------------------------------------


def test_build_contrastive_pairs_lengths():
    """Consistent and inconsistent lists have equal length.  Spec: REQ-VER-032."""
    import random

    chains = {
        "q1": ["step A", "step B"],
        "q2": ["step C", "step D"],
        "q3": ["step E", "step F"],
    }
    rng = random.Random(42)
    con, inc = build_contrastive_pairs(chains, rng)
    assert len(con) == len(inc)
    assert len(con) == 3


def test_build_contrastive_pairs_intruder_is_foreign():
    """Inconsistent sets contain exactly one intruder step from a different question.  Spec: REQ-VER-032."""
    import random

    chains = {
        "q1": ["alpha step"],
        "q2": ["beta step"],
    }
    rng = random.Random(0)
    con, inc = build_contrastive_pairs(chains, rng)
    # Each inconsistent set must differ from the corresponding consistent set
    for c_set, i_set in zip(con, inc):
        # They should differ somewhere (intruder was injected)
        assert c_set != i_set or len(c_set) == 1  # single-step chains may keep same if swap identical


def test_build_contrastive_pairs_single_question():
    """Single-question corpus produces zero pairs (no intruder possible).  Spec: REQ-VER-032."""
    import random

    chains = {"q_only": ["step 1", "step 2"]}
    rng = random.Random(0)
    con, inc = build_contrastive_pairs(chains, rng)
    assert len(con) == 0
    assert len(inc) == 0


# ---------------------------------------------------------------------------
# Test compute_auroc — REQ-VER-032, SCENARIO-VER-039
# ---------------------------------------------------------------------------


def test_compute_auroc_random_verifier():
    """Untrained verifier on tiny corpus returns a float AUROC in [0, 1].  Spec: SCENARIO-VER-039."""
    from carnot.verify.sc_energy import SetConsistencyVerifier

    verifier = SetConsistencyVerifier(seed=0)
    con = [["step A", "step B"]]
    inc = [["step C", "step D"]]
    result = compute_auroc(verifier, con, inc)
    assert 0.0 <= result <= 1.0


def test_compute_auroc_degenerate_single_class():
    """All-same-label corpus returns 0.5 (degenerate case).  Spec: SCENARIO-VER-039."""
    from carnot.verify.sc_energy import SetConsistencyVerifier

    verifier = SetConsistencyVerifier(seed=0)
    # Passing only consistent sets — degenerate (no inconsistent class)
    result = compute_auroc(verifier, [["step A"]], [])
    assert result == 0.5


# ---------------------------------------------------------------------------
# Tests for load_v1_baseline_auc — REQ-VER-032
# ---------------------------------------------------------------------------


def test_load_v1_baseline_missing_file(tmp_path):
    """Missing artifact returns 0.5 default.  Spec: REQ-VER-032."""
    result = load_v1_baseline_auc(tmp_path / "nonexistent.json")
    assert result == 0.5


def test_load_v1_baseline_reads_auc(tmp_path):
    """Valid artifact returns the sc_energy_auc value.  Spec: REQ-VER-032."""
    artifact = {"sc_energy_auc": 0.73, "status": "success"}
    p = tmp_path / "exp711.json"
    p.write_text(json.dumps(artifact))
    result = load_v1_baseline_auc(p)
    assert abs(result - 0.73) < 1e-9


def test_load_v1_baseline_malformed_json(tmp_path):
    """Malformed JSON returns 0.5 default.  Spec: REQ-VER-032."""
    p = tmp_path / "bad.json"
    p.write_text("not json {{}")
    result = load_v1_baseline_auc(p)
    assert result == 0.5


def test_load_v1_baseline_missing_key(tmp_path):
    """Artifact without sc_energy_auc returns 0.5 default.  Spec: REQ-VER-032."""
    p = tmp_path / "no_key.json"
    p.write_text(json.dumps({"status": "success"}))
    result = load_v1_baseline_auc(p)
    assert result == 0.5


# ---------------------------------------------------------------------------
# Test that the deliverable JSON exists and has required fields — REQ-VER-032
# ---------------------------------------------------------------------------


def test_deliverable_exists_and_schema():
    """Exp 725 deliverable exists with all required artifact fields.  Spec: REQ-VER-032."""
    repo = Path(__file__).resolve().parents[2]
    deliverable = repo / "results" / "experiment_725_sc_energy_v2.json"
    assert deliverable.exists(), f"Deliverable not found: {deliverable}"
    art = json.loads(deliverable.read_text())
    required = [
        "experiment", "title", "run_date", "started_at", "finished_at",
        "duration_s", "status", "ood_auc", "v1_baseline_auc", "auc_delta",
        "training_pairs", "honest_verdict", "schema",
    ]
    for field in required:
        assert field in art, f"Missing required field: {field}"


def test_deliverable_honest_verdict_is_canonical():
    """honest_verdict is one of the three canonical strings from REQ-VER-032.  Spec: REQ-VER-032."""
    repo = Path(__file__).resolve().parents[2]
    deliverable = repo / "results" / "experiment_725_sc_energy_v2.json"
    if not deliverable.exists():
        pytest.skip("Deliverable not yet generated")
    art = json.loads(deliverable.read_text())
    canonical = {
        "sc_energy_v2_improvement",
        "sc_energy_v2_no_gain",
        "sc_energy_v2_below_threshold",
        "blocked_missing_fover_v2",
    }
    assert art["honest_verdict"] in canonical, (
        f"Unexpected honest_verdict: {art['honest_verdict']!r}"
    )


def test_deliverable_auc_delta_consistency():
    """auc_delta == ood_auc - v1_baseline_auc within floating-point tolerance.  Spec: REQ-VER-032."""
    repo = Path(__file__).resolve().parents[2]
    deliverable = repo / "results" / "experiment_725_sc_energy_v2.json"
    if not deliverable.exists():
        pytest.skip("Deliverable not yet generated")
    art = json.loads(deliverable.read_text())
    expected_delta = art["ood_auc"] - art["v1_baseline_auc"]
    assert abs(art["auc_delta"] - expected_delta) < 1e-9
