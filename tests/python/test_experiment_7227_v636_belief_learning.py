"""Verify the prospective lossless belief-learning replay.

Spec refs: REQ-CL-7227 and SCENARIO-CL-7227-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7227_v636_belief_learning as exp7227


def _one_seed_views() -> exp7227.StreamViews:
    """Load one sealed stream so focused tests use real producer bytes."""

    views = exp7227.load_stream_views(exp7227.REPO_ROOT, exp7227.DEFAULT_UPSTREAM_ARTIFACT)
    seed = exp7227.STREAM_SEEDS[0]
    return exp7227.StreamViews(
        [row for row in views.public if row["seed"] == seed],
        [row for row in views.authority if row["seed"] == seed],
        [row for row in views.releases if row["seed"] == seed],
        {**views.manifest, "seeds": [seed], "total_events": exp7227.EVENTS_PER_SEED},
    )


def test_req_cl_7227_preconditions_fail_closed_and_unwrap(tmp_path: Path) -> None:
    """REQ-CL-7227: an absent authenticated contract produces a row-free block."""

    assert exp7227.unwrap_principled({"principle": "why", "value": 1}) == 1
    assert exp7227.ExperimentPaths.defaults() == exp7227.ExperimentPaths(
        exp7227.DEFAULT_STATE_PATH,
        exp7227.DEFAULT_DECISION_ROWS_PATH,
        exp7227.DEFAULT_ARTIFACT_PATH,
    )
    arbitrary = {"principle": "why", "value": 1, "extra": True}
    assert exp7227.unwrap_principled(arbitrary) is arbitrary
    paths = exp7227.ExperimentPaths.under(tmp_path)
    checks, upstream, hashes = exp7227.collect_preconditions(
        exp7227.REPO_ROOT,
        paths,
        upstream_artifact=tmp_path / "missing.json",
    )
    assert upstream == {}
    assert hashes[str(tmp_path / "missing.json")] is None
    artifact = exp7227.build_blocked_artifact(checks, hashes, paths, duration_s=0.01)
    assert artifact["status"] == "blocked"
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["passed"] is False
    assert exp7227.validate_artifact(artifact, check_files=False) == []
    blocked_run = exp7227.build_and_seal(
        exp7227.REPO_ROOT,
        paths,
        upstream_artifact=tmp_path / "missing.json",
        seeds=exp7227.STREAM_SEEDS[:1],
        progress=True,
    )
    assert blocked_run["status"] == "blocked"


def test_scenario_cl_7227_chronology_matched_and_cost(tmp_path: Path) -> None:
    """SCENARIO-CL-7227-CHRONOLOGY: real rows decide before due feedback."""

    views = _one_seed_views()
    panel = exp7227.run_learning_panel(
        views,
        state_path=tmp_path / "state.json",
        seeds=exp7227.STREAM_SEEDS[:1],
        progress=False,
    )
    assert len(panel.rows) == len(exp7227.ARMS)
    assert len(panel.decisions) == len(exp7227.ARMS) * exp7227.EVENTS_PER_SEED
    assert exp7227.panel_conformance_errors(panel, seeds=exp7227.STREAM_SEEDS[:1]) == []
    packed = [row for row in panel.decisions if row["arm"] == "packed_online_memory"]
    reference = [row for row in panel.decisions if row["arm"] == "reference_online_version_space"]
    assert all(row["prediction_before_release"] for row in packed)
    assert [row["query_selected"] for row in packed] == [row["query_selected"] for row in reference]
    assert panel.parity_counts == {
        "prediction_mismatch_count": 0,
        "query_mismatch_count": 0,
        "energy_mismatch_count": 0,
    }
    assert max(row["pending_capacity_use"] for row in packed) <= exp7227.PENDING_CAPACITY
    assert all(row["query_count"] <= exp7227.QUERY_CEILING for row in panel.rows)
    summary = exp7227.latency_summary(panel)
    assert set(summary["operations"]) == set(exp7227.COST_NAMES)
    assert all(value["count"] > 0 for value in summary["operations"].values())
    assert summary["allocated_bytes"]["maximum_state_bytes"] > 0
    empty = exp7227.exp7226.PackedBeliefController.from_survivors({"lower_bound": set()})
    assert exp7227._packed_energy(empty, "abstain", views.public[0]) is None
    assert exp7227._percentile([], 0.5) == 0.0
    assert exp7227._bootstrap_interval([], draws=4, salt="empty")["estimate"] == 0.0
    panel.parity_counts["prediction_mismatch_count"] = 1
    assert "packed_reference_parity" in exp7227.panel_conformance_errors(
        panel, seeds=exp7227.STREAM_SEEDS[:1]
    )


def test_scenario_cl_7227_controls_causality_and_gates(tmp_path: Path) -> None:
    """SCENARIO-CL-7227-CAUSAL: withholding can only affect later decisions."""

    panel = exp7227.run_learning_panel(
        _one_seed_views(),
        state_path=tmp_path / "state.json",
        seeds=exp7227.STREAM_SEEDS[:1],
        progress=False,
    )
    comparisons = exp7227.build_comparison_rows(panel.rows, draws=128)
    deletion = exp7227.build_memory_deletion_rows(panel.decisions)
    gate = exp7227.score_acceptance_gate(comparisons, panel.parity_counts, deletion)
    assert len(comparisons) == 3
    assert len(comparisons[0]["seed_differences"]) == 1
    assert len(deletion) == 1
    assert deletion[0]["pre_release_difference_count"] == 0
    assert gate["parity"]["passed"] is True
    assert gate["cost"]["required_for_learning_value"] is False
    assert gate["learning_value_passed"] in {True, False}

    bad_parity = deepcopy(panel.parity_counts)
    bad_parity["prediction_mismatch_count"] = 1
    failed = exp7227.score_acceptance_gate(comparisons, bad_parity, deletion)
    assert failed["parity"]["passed"] is False
    assert failed["learning_value_passed"] is False


def test_scenario_cl_7227_terminal_build_and_cold_validation(tmp_path: Path) -> None:
    """SCENARIO-CL-7227-TERMINAL: a complete one-seed fixture remains valid evidence."""

    paths = exp7227.ExperimentPaths.under(tmp_path)
    artifact = exp7227.build_and_seal(
        exp7227.REPO_ROOT,
        paths,
        seeds=exp7227.STREAM_SEEDS[:1],
        progress=True,
    )
    assert artifact["status"] == "complete"
    assert artifact["belief_run_complete_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["verdict_class"] in {"circular_positive", "null"}
    assert Path(artifact["decision_rows_path"]["path"]).is_file()
    assert (
        exp7227.validate_artifact(
            artifact,
            check_files=True,
            repo_root=exp7227.REPO_ROOT,
            expected_seeds=exp7227.STREAM_SEEDS[:1],
        )
        == []
    )
    exp7227.write_artifact(paths.artifact, artifact)
    assert json.loads(paths.artifact.read_text()) == artifact

    broken = deepcopy(artifact)
    broken["belief_run_complete_score"] = 0
    assert "completion_score" in exp7227.validate_artifact(
        broken, check_files=False, expected_seeds=exp7227.STREAM_SEEDS[:1]
    )
    with pytest.raises(ValueError, match="invalid_exp7227_artifact:test_error"):
        exp7227._require_valid(["test_error"])


def test_req_cl_7227_main_and_thin_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7227: the fixed command delegates through one small entrypoint."""

    failed_check = exp7227.gate_check("test_fixture", "test", "ready", True, False)
    expected = exp7227._base_artifact([failed_check], {}, exp7227.ExperimentPaths.under(tmp_path))
    expected.update(
        {
            "status": "blocked",
            "inference_substrate": "blocked_no_run",
            "inference_substrate_class": "blocked_no_run",
            "verdict_class": "blocked",
            "honest_verdict": "blocked_test_fixture",
        }
    )
    expected["reproducibility_checksum"] = exp7227.reproducibility_checksum(expected)
    monkeypatch.setattr(exp7227, "build_and_seal", lambda *args, **kwargs: expected)
    output = tmp_path / "output"
    assert exp7227.main(["--date", "20260911", "--output-root", str(output)]) == 0
    assert (output / exp7227.DEFAULT_ARTIFACT_PATH.name).is_file()
    with pytest.raises(ValueError, match="run_date"):
        exp7227.main(["--date", "20260910", "--output-root", str(tmp_path / "bad")])
