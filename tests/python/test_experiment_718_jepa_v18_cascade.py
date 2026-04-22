"""Tests for Experiment 718 — JEPA v18 Cascade Deploy.

Covers:
- Gate-blocked path: when Exp 717 gate="fail", artifact has gated_blocked status.
- v18 checkpoint loads without error (save → load cycle).
- Cascade AUC field is present and in [0, 1] in the artifact.
- Version-blocked model raises ValueError (REQ-INFRA-043).

Spec: REQ-INFRA-043, REQ-INFRA-044, REQ-INFRA-045,
      SCENARIO-INFRA-052, SCENARIO-INFRA-053, SCENARIO-INFRA-054
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module imports — all three modules under test
# ---------------------------------------------------------------------------

from scripts.experiment_718_jepa_v18_cascade import (
    EXPERIMENT_ID,
    DELIVERABLE,
    write_gated_blocked_artifact,
    run_experiment,
    make_smoke_eval_groups,
)
from carnot.cascade.tier2_jepa import (
    load_v18_from_manifest,
    save_checkpoint,
    _load_weights_from_npz,
)
from carnot.samplers.jepa_v18_lambdarank import JEPALambdaRankV18


# ---------------------------------------------------------------------------
# Helper: minimal repo root with required structure
# ---------------------------------------------------------------------------


def _make_repo_root(tmp_path: Path) -> Path:
    """Create a temporary directory tree that looks like the repo root.

    Produces:
        <root>/results/                 — output dir
        <root>/results/jepa_v18_gate.json  — gate file (pass or fail, caller sets it)
        <root>/scripts/conductor_exclusion_manifest.json — minimal manifest

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    Path
        The fake repo root.
    """
    root = tmp_path / "carnot"
    (root / "results").mkdir(parents=True)
    (root / "results" / "checkpoints").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "openspec" / "capabilities" / "pipeline").mkdir(parents=True)

    # Minimal exclusion manifest
    manifest = {
        "jepa_v18_active": True,
        "excluded": [
            {
                "experiment_id": "jepa_v17_cascade",
                "reason": "ood_auc_below_random",
            }
        ],
    }
    (root / "scripts" / "conductor_exclusion_manifest.json").write_text(
        json.dumps(manifest)
    )
    return root


# ---------------------------------------------------------------------------
# Test 1: Gate-blocked path writes correct status
# Spec: REQ-INFRA-043, SCENARIO-INFRA-052
# ---------------------------------------------------------------------------


def test_gate_blocked_path_writes_correct_status(tmp_path: Path) -> None:
    """When Exp 717 gate='fail', write_gated_blocked_artifact writes the required schema.

    REQ-INFRA-043: the blocked artifact must carry the exact fields that the conductor
    uses to identify a blocked result and skip Exp 718 logic entirely.
    """
    root = _make_repo_root(tmp_path)

    artifact = write_gated_blocked_artifact(repo_root=root)

    # Deliverable file must exist
    out_path = root / DELIVERABLE
    assert out_path.exists(), "Deliverable must be written even for gated_blocked"

    # Parse persisted artifact — must match return value
    persisted = json.loads(out_path.read_text())
    assert persisted == artifact

    # Required schema fields
    assert artifact["experiment"] == EXPERIMENT_ID
    assert artifact["status"] == "gated_blocked"
    assert artifact["gate_source"] == "exp717"
    assert artifact["honest_verdict"] == "gated_blocked_jepa_v18_below_threshold"
    assert artifact["schema"] == "carnot.result.v1"

    # Required ExperimentTemplate-compatible fields
    for field in ("run_date", "started_at", "finished_at", "duration_s"):
        assert field in artifact, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# Test 2: v18 checkpoint save → load cycle works without error
# Spec: REQ-INFRA-043, SCENARIO-INFRA-053
# ---------------------------------------------------------------------------


def test_v18_checkpoint_loads_without_error(tmp_path: Path) -> None:
    """A JEPALambdaRankV18 model can be saved and reloaded without raising any exception.

    REQ-INFRA-043: the checkpoint load must restore identical weights so that inference
    results are reproducible across process restarts (no silent weight corruption).

    SCENARIO-INFRA-053: predict_score() on the reloaded model returns a float.
    """
    ckpt_path = str(tmp_path / "weights.npz")

    # Create and save
    model_orig = JEPALambdaRankV18(feature_dim=1024, hidden_dim=64)
    save_checkpoint(model_orig, ckpt_path)

    # Reload via load_v18_from_manifest
    model_loaded = load_v18_from_manifest(version="v18", checkpoint_path=ckpt_path)

    # Weights must be numerically identical
    np.testing.assert_array_equal(model_orig.W1, model_loaded.W1)
    np.testing.assert_array_equal(model_orig.W2, model_loaded.W2)
    np.testing.assert_array_equal(model_orig.W3, model_loaded.W3)

    # predict_score must return a float on any string
    score = model_loaded.predict_score("Step 1: 3 + 5 = 8.")
    assert isinstance(score, float), "predict_score must return float"


# ---------------------------------------------------------------------------
# Test 3: Cascade AUC field present in artifact
# Spec: REQ-INFRA-044, SCENARIO-INFRA-054
# ---------------------------------------------------------------------------


def test_cascade_auc_field_present_in_artifact(tmp_path: Path) -> None:
    """run_experiment() writes an artifact that contains cascade_auc in [0, 1].

    REQ-INFRA-044: the smoke test must evaluate AUC and record it in the artifact.
    SCENARIO-INFRA-054: cascade_auc is in [0, 1].
    """
    root = _make_repo_root(tmp_path)

    artifact = run_experiment(repo_root=root)

    # Deliverable must exist
    out_path = root / DELIVERABLE
    assert out_path.exists(), "Deliverable must be written by run_experiment"

    # cascade_auc field must be present and in [0, 1]
    assert "cascade_auc" in artifact, "artifact must contain cascade_auc"
    auc = artifact["cascade_auc"]
    assert isinstance(auc, float), "cascade_auc must be a float"
    assert 0.0 <= auc <= 1.0, f"cascade_auc must be in [0, 1], got {auc}"

    # latency_delta_ms must be present and non-negative
    assert "latency_delta_ms" in artifact, "artifact must contain latency_delta_ms"
    assert artifact["latency_delta_ms"] >= 0.0

    # honest_verdict must be one of the three defined verdicts
    valid_verdicts = {
        "cascade_deploy_success",
        "cascade_deploy_latency_fail",
        "cascade_deploy_auc_fail",
    }
    assert artifact["honest_verdict"] in valid_verdicts, (
        f"honest_verdict must be one of {valid_verdicts}, got '{artifact['honest_verdict']}'"
    )

    # schema must be present — build_result() sets it to a sorted list of all keys
    assert "schema" in artifact
    schema = artifact["schema"]
    assert isinstance(schema, list), "build_result sets schema to a list of field names"
    assert "cascade_auc" in schema

    # Gate file must have been written
    gate_path = root / "results" / "jepa_v18_cascade_gate.json"
    assert gate_path.exists(), "Gate file for Exp 719 must be written"
    gate = json.loads(gate_path.read_text())
    assert "gate" in gate
    assert gate["gate"] in ("pass", "fail")


# ---------------------------------------------------------------------------
# Test 4: Blocked JEPA versions raise ValueError
# Spec: REQ-INFRA-043, SCENARIO-INFRA-052
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocked_version", ["v15", "v16", "v17"])
def test_blocked_version_raises_value_error(blocked_version: str) -> None:
    """load_v18_from_manifest raises ValueError for any blocked JEPA version.

    REQ-INFRA-043: the manifest enforces version-pinning — loading a below-threshold
    version must raise immediately so stale code is caught before it reaches inference.

    SCENARIO-INFRA-052: the error message must contain "blocked" so callers can
    identify the root cause without reading the manifest file.
    """
    with pytest.raises(ValueError, match="blocked"):
        load_v18_from_manifest(version=blocked_version)


# ---------------------------------------------------------------------------
# Test 5: make_smoke_eval_groups produces valid group structure
# Spec: REQ-INFRA-044
# ---------------------------------------------------------------------------


def test_make_smoke_eval_groups_structure() -> None:
    """make_smoke_eval_groups returns well-formed groups for AUC evaluation.

    Each group must have at least one correct (label=1) and one incorrect (label=0)
    step so that evaluate_auc can compute pairwise comparisons.
    """
    groups = make_smoke_eval_groups(n=10)
    assert len(groups) == 10

    for i, group in enumerate(groups):
        steps = group["steps"]
        assert len(steps) >= 2, f"Group {i} must have >= 2 steps"
        labels = [s["label"] for s in steps]
        assert 1 in labels, f"Group {i} must have at least one correct step"
        assert 0 in labels, f"Group {i} must have at least one incorrect step"

        for step in steps:
            assert "text" in step
            assert "label" in step
            assert step["label"] in (0, 1)
