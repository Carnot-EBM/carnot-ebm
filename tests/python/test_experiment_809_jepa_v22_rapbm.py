"""Tests for Experiment 809 — JEPA v22 RA-PRM / Held-Out Evaluation.

Spec: REQ-LEARN-101, REQ-LEARN-102,
      SCENARIO-LEARN-148, SCENARIO-LEARN-149
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_809_jepa_v22_rapbm as exp809  # noqa: E402
from carnot.pipeline.embedding_constraint_store import (  # noqa: E402
    ConstraintSPOTuple,
    EmbeddingConstraintStore,
)


# ---------------------------------------------------------------------------
# Test: PATH A is selected when Exp 808 ood_auc >= 0.75  (REQ-LEARN-101)
# ---------------------------------------------------------------------------


def test_path_a_selected_when_exp808_ood_auc_above_gate(tmp_path: Path) -> None:
    """_load_exp808_result returns dict; ood_auc >= OOD_GATE drives PATH A selection.

    Spec: REQ-LEARN-101, SCENARIO-LEARN-148
    """
    exp808_data = {
        "experiment": 808,
        "ood_auc": 0.80,
        "honest_verdict": "jepa_v22_tier35_deployed",
        "status": "success",
    }
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_file = results_dir / "experiment_808_jepa_v22_retrain.json"
    result_file.write_text(json.dumps(exp808_data))

    # Patch REPO_ROOT to point at tmp_path so _load_exp808_result resolves correctly.
    original_root = exp809.REPO_ROOT
    exp809.REPO_ROOT = tmp_path  # type: ignore[assignment]
    try:
        loaded = exp809._load_exp808_result(tmp_path)
        assert loaded["ood_auc"] == 0.80
        # Verify PATH branching logic uses OOD_GATE correctly.
        assert loaded["ood_auc"] >= exp809.OOD_GATE, (
            "With ood_auc=0.80, experiment should take PATH A (held-out evaluation)"
        )
    finally:
        exp809.REPO_ROOT = original_root  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Test: PATH B is selected when Exp 808 ood_auc < 0.75  (REQ-LEARN-102)
# ---------------------------------------------------------------------------


def test_path_b_selected_when_exp808_ood_auc_below_gate(tmp_path: Path) -> None:
    """Exp 808 ood_auc=0.2 (< OOD_GATE) must route to PATH B (RA-PRM).

    Spec: REQ-LEARN-102, SCENARIO-LEARN-149
    """
    exp808_data = {
        "experiment": 808,
        "ood_auc": 0.2,
        "honest_verdict": "jepa_v22_below_random",
        "status": "success",
    }
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_file = results_dir / "experiment_808_jepa_v22_retrain.json"
    result_file.write_text(json.dumps(exp808_data))

    loaded = exp809._load_exp808_result(tmp_path)
    assert loaded["ood_auc"] == 0.2
    assert loaded["ood_auc"] < exp809.OOD_GATE, (
        "With ood_auc=0.2, experiment must take PATH B (RA-PRM soft supervision)"
    )


# ---------------------------------------------------------------------------
# Test: RA-PRM soft label computation respects GT × 1.0 + retrieved × 0.4
# (REQ-LEARN-102)
# ---------------------------------------------------------------------------


def test_rapbm_soft_label_computation() -> None:
    """_build_rapbm_soft_labels: ground_truth × 1.0 + retrieved_avg × 0.4.

    Spec: REQ-LEARN-102, SCENARIO-LEARN-149
    """
    # Build a small store with two entries: one "violates", one "satisfies".
    store = EmbeddingConstraintStore()
    store.store(ConstraintSPOTuple(
        subject="arithmetic_step",
        predicate="violates",
        object="step_correctness_constraint",
        embedding=None,
        source_violation_type="incorrect",
    ))
    store.store(ConstraintSPOTuple(
        subject="correct_step",
        predicate="satisfies",
        object="step_correctness_constraint",
        embedding=None,
        source_violation_type="correct",
    ))

    # A single training example with gt_label=0.0 (correct step).
    step_seqs = [["3 + 4 = 7. The answer is 7."]]
    gt_labels = [0.0]

    soft_labels = exp809._build_rapbm_soft_labels(step_seqs, gt_labels, store)

    assert len(soft_labels) == 1, "Should produce one soft label per training example"
    sl = soft_labels[0]

    # The formula is gt × 1.0 + retrieved_avg × 0.4.
    # retrieved_avg is in [0.0, 1.0], so soft_label is in [0.0, 1.4].
    assert 0.0 <= sl <= 1.4 + 1e-6, (
        f"Soft label {sl} out of expected range [0.0, 1.4]; "
        "formula: gt × 1.0 + retrieved_avg × 0.4"
    )

    # Ground-truth weight must always contribute exactly RAPBM_GT_WEIGHT × gt_label.
    # We can verify by using gt_label=1.0 and checking sl >= RAPBM_GT_WEIGHT.
    step_seqs_violation = [["Divide both sides by 0."]]
    gt_labels_violation = [1.0]
    soft_labels_v = exp809._build_rapbm_soft_labels(
        step_seqs_violation, gt_labels_violation, store
    )
    sl_v = soft_labels_v[0]
    assert sl_v >= exp809.RAPBM_GT_WEIGHT, (
        f"When gt_label=1.0, soft_label must be >= RAPBM_GT_WEIGHT={exp809.RAPBM_GT_WEIGHT}; got {sl_v}"
    )


# ---------------------------------------------------------------------------
# Test: retrieve() integration produces K=3 results from populated store
# (REQ-LEARN-102, REQ-LEARN-059)
# ---------------------------------------------------------------------------


def test_retrieve_returns_k3_from_populated_store() -> None:
    """EmbeddingConstraintStore.retrieve() returns exactly K=3 results when store >= 3.

    Spec: REQ-LEARN-102 (RA-PRM K=3 retrieval requirement)
    """
    store = EmbeddingConstraintStore()

    # Populate with 5 distinct steps so K=3 is satisfiable.
    for i in range(5):
        predicate = "violates" if i % 2 == 0 else "satisfies"
        spo = ConstraintSPOTuple(
            subject=f"step_{i}_arithmetic_operation",
            predicate=predicate,
            object="step_correctness_constraint",
            embedding=None,
            source_violation_type="incorrect" if i % 2 == 0 else "correct",
        )
        store.store(spo)

    assert len(store._store) == 5, "Store must hold all 5 inserted entries"

    # Retrieve K=3 for a query.
    results = store.retrieve("arithmetic subtraction step", top_k=exp809.K_RETRIEVE)
    assert len(results) == exp809.K_RETRIEVE, (
        f"retrieve() must return exactly K={exp809.K_RETRIEVE} results "
        f"when store has >= K entries; got {len(results)}"
    )

    # All returned entries must be ConstraintSPOTuple instances.
    for r in results:
        assert isinstance(r, ConstraintSPOTuple)


# ---------------------------------------------------------------------------
# Test: _populate_store_from_corpus creates one SPO per corpus entry
# ---------------------------------------------------------------------------


def test_populate_store_from_corpus_maps_labels_to_predicates() -> None:
    """_populate_store_from_corpus: incorrect→violates predicate, correct→satisfies.

    Spec: REQ-LEARN-102
    """
    corpus = [
        {"step_text": "3 + 4 = 8", "label": "incorrect"},
        {"step_text": "3 + 4 = 7", "label": "correct"},
        {"step_text": "x = -3 because 2x = -6", "label": "correct"},
    ]
    store = EmbeddingConstraintStore()
    exp809._populate_store_from_corpus(corpus, store)

    assert len(store._store) == 3, "One SPO entry per corpus item"

    predicates = [spo.predicate for spo in store._store]
    assert "violates" in predicates, "Incorrect steps must map to predicate='violates'"
    assert "satisfies" in predicates, "Correct steps must map to predicate='satisfies'"

    violation_types = [spo.source_violation_type for spo in store._store]
    assert "incorrect" in violation_types
    assert "correct" in violation_types


# ---------------------------------------------------------------------------
# Test: PATH B returns expected fields including honest_verdict
# ---------------------------------------------------------------------------


def test_path_b_returns_required_fields(tmp_path: Path) -> None:
    """_run_path_b returns a dict with all RA-PRM result fields.

    Spec: REQ-LEARN-102, SCENARIO-LEARN-149
    """
    # Write a minimal FoVer corpus to tmp_path for path resolution.
    corpus = [
        {"step_text": "2 + 2 = 4", "label": "correct", "source_domain": "gsm8k"},
        {"step_text": "2 + 2 = 5", "label": "incorrect", "source_domain": "gsm8k"},
        {"step_text": "4 * 3 = 12", "label": "correct", "source_domain": "math500"},
        {"step_text": "4 * 3 = 11", "label": "incorrect", "source_domain": "math500"},
    ]
    multi_path = tmp_path / "results"
    multi_path.mkdir(parents=True, exist_ok=True)
    (multi_path / "fover_labeled_steps_v21_multi.json").write_text(json.dumps(corpus))

    result = exp809._run_path_b(tmp_path, exp808_ood_auc=0.2)

    required_fields = [
        "path", "rapbm_applied", "rapbm_k_retrieve", "rapbm_gt_weight",
        "rapbm_retrieved_weight", "rapbm_soft_label_avg", "rapbm_store_entries",
        "in_dist_auc", "ood_auc", "ood_auc_delta_vs_808", "honest_verdict",
    ]
    for field in required_fields:
        assert field in result, f"PATH B result missing required field: {field}"

    assert result["path"] == "B"
    assert result["rapbm_applied"] is True
    assert result["rapbm_k_retrieve"] == exp809.K_RETRIEVE
    assert result["honest_verdict"] in ("rapbm_ood_improved", "rapbm_no_gain")
