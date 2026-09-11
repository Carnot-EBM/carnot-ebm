"""Verify witnessed predicate learning against the frozen Exp7212 fixture."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7213_v635_refinement_learning as exp


def test_fixed_contract_and_principle_wrapper() -> None:
    """REQ-CL-7213: Freeze the five arms, budgets, seed, and safe wrapper."""

    assert exp.ARMS == (
        "warmup_frozen",
        "passive_query_committed",
        "random_query_committed",
        "witness_query_committed",
        "witness_query_version_space",
    )
    assert exp.BOOTSTRAP_SEED == 7_213_001
    assert exp.BOOTSTRAP_DRAWS == 10_000
    assert exp.QUERY_BUDGET == 64
    assert exp.FITTING_BUDGET == 48
    assert exp.VALIDATION_BUDGET == 16
    assert exp.PENDING_CAPACITY == 4
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False

    wrapped = {"principle": "why", "value": {"passed": True}}
    arbitrary = {"value": 1, "other": 2}
    extra = {"principle": "why", "value": 1, "evidence": "x"}
    assert exp.unwrap_principled(wrapped) == {"passed": True}
    assert exp.unwrap_principled(arbitrary) is arbitrary
    assert exp.unwrap_principled(extra) is extra


def test_preconditions_authenticate_fixture_and_independent_quarantine(tmp_path: Path) -> None:
    """SCENARIO-CL-7213-PRECONDITIONS: Authenticate bytes and both quarantine channels."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, fixture, hashes = exp.collect_preconditions(exp.REPO_ROOT, paths)
    assert fixture["refinement_fixture_ready_score"] == 1
    assert all(row["passed"] for row in checks)
    assert hashes[str(exp.DEFAULT_UPSTREAM_ARTIFACT)] is not None

    clean = exp.quarantine_state({}, "", "artifact.json", "exp-clean")
    flagged = exp.quarantine_state({"flagged_adversarial": True}, "", "artifact.json", "exp")
    listed = exp.quarantine_state({}, "- artifact.json\n", "artifact.json", "exp")
    assert clean["quarantined"] is False
    assert flagged["quarantined"] is True
    assert listed["quarantined"] is True

    assert exp.ExperimentPaths.defaults().artifact == exp.DEFAULT_ARTIFACT_PATH
    assert exp._sha256_path(tmp_path / "missing") is None
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    malformed.write_text("[]", encoding="utf-8")
    assert exp._load_object(malformed) == {}


def test_one_stream_panel_preserves_chronology_budgets_and_matched_schedule(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7213-CHRONOLOGY/ACQUISITION: Seal actions before authority."""

    views = exp.load_fixture_views(exp.REPO_ROOT, exp.DEFAULT_UPSTREAM_ARTIFACT)
    seed = exp.STREAM_SEEDS[0]
    panel = exp.run_learning_panel(
        views,
        state_root=tmp_path / "state",
        seeds=(seed,),
        progress=True,
    )

    assert len(panel.rows) == len(exp.ARMS)
    assert len(panel.decision_rows) == exp.EVENTS_PER_SEED * len(exp.ARMS)
    assert exp.panel_conformance_errors(panel, seeds=(seed,)) == []
    assert all(
        row["prediction_operation_index"]
        < row["query_decision_operation_index"]
        < row["authority_score_operation_index"]
        for row in panel.decision_rows
    )
    assert all(row["future_label_visible_to_decision"] is False for row in panel.decision_rows)
    assert all(row["hidden_parameter_visible_to_decision"] is False for row in panel.decision_rows)
    assert all(row["hidden_audit_visible_to_decision"] is False for row in panel.decision_rows)

    witness = [
        (row["event_id"], row["role"], row["release_index"])
        for row in panel.query_rows
        if row["arm"] == "witness_query_committed" and row["query_charged"]
    ]
    version = [
        (row["event_id"], row["role"], row["release_index"])
        for row in panel.query_rows
        if row["arm"] == "witness_query_version_space" and row["query_charged"]
    ]
    assert witness == version
    assert all(row["max_pending"] <= exp.PENDING_CAPACITY for row in panel.rows)
    assert all(row["query_count"] <= exp.QUERY_BUDGET for row in panel.rows)
    assert all(row["fitting_query_count"] <= exp.FITTING_BUDGET for row in panel.rows)
    assert all(row["validation_query_count"] <= exp.VALIDATION_BUDGET for row in panel.rows)
    assert all(row["validation_used_for_elimination"] is False for row in panel.query_rows)
    assert all(row["predicted_before_request"] is True for row in panel.query_rows)


def test_committed_template_deletion_preserves_acquisition_state(tmp_path: Path) -> None:
    """SCENARIO-CL-7213-COMMIT/DELETION: Remove real template records on a shadow."""

    views = exp.load_fixture_views(exp.REPO_ROOT, exp.DEFAULT_UPSTREAM_ARTIFACT)
    panel = exp.run_learning_panel(
        views,
        state_root=tmp_path / "state",
        seeds=(exp.STREAM_SEEDS[0],),
    )
    assert panel.commit_deletion_rows
    assert all(
        row["acquisition_state_hash_before"] == row["acquisition_state_hash_after"]
        for row in panel.commit_deletion_rows
    )
    assert all(row["template_count_after"] == 0 for row in panel.commit_deletion_rows)
    assert all(row["full_reset_is_separate"] is True for row in panel.commit_deletion_rows)
    assert any(row["deletion_changed_prediction"] for row in panel.commit_deletion_rows)
    assert any(row["commit_event"] is not None for row in panel.query_rows)


def test_conformance_detects_chronology_schedule_and_capacity_attacks(tmp_path: Path) -> None:
    """SCENARIO-CL-7213-ACQUISITION: Named attacks close the panel gate."""

    views = exp.load_fixture_views(exp.REPO_ROOT, exp.DEFAULT_UPSTREAM_ARTIFACT)
    seed = exp.STREAM_SEEDS[0]
    panel = exp.run_learning_panel(
        views,
        state_root=tmp_path / "state",
        seeds=(seed,),
    )
    changed = deepcopy(panel)
    changed.decision_rows[0]["authority_score_operation_index"] = 0
    assert "prediction_authority_chronology" in exp.panel_conformance_errors(changed, seeds=(seed,))
    changed = deepcopy(panel)
    changed.rows[0]["max_pending"] = exp.PENDING_CAPACITY + 1
    assert "pending_capacity" in exp.panel_conformance_errors(changed, seeds=(seed,))
    changed = deepcopy(panel)
    next(row for row in changed.query_rows if row["arm"] == "witness_query_version_space")[
        "event_id"
    ] = "mismatch"
    assert "witness_schedule_parity" in exp.panel_conformance_errors(changed, seeds=(seed,))


def test_bootstrap_and_primary_gate_keep_secondary_evidence_separate() -> None:
    """SCENARIO-CL-7213-METRICS/DEPLOYMENT: Apply fixed paired gates without rescue."""

    rows = []
    for seed in range(20):
        for arm, error, false_accept, recurrence, cost in (
            ("warmup_frozen", 200, 30, 60, 1_000),
            ("passive_query_committed", 170, 25, 50, 900),
            ("random_query_committed", 180, 25, 55, 900),
            ("witness_query_committed", 100, 20, 40, 100),
            ("witness_query_version_space", 101, 20, 41, 400),
        ):
            rows.append(
                {
                    "unit_id": f"{seed}:{arm}",
                    "seed": seed,
                    "arm": arm,
                    "metric": "prospective_full_denominator_error",
                    "error": error,
                    "abstention": 0,
                    "event_count": 992,
                    "error_rate": error / 992,
                    "false_accept": false_accept,
                    "false_accept_rate": false_accept / 992,
                    "recurrence_error": recurrence,
                    "recurrence_error_rate": recurrence / 256,
                    "total_cost_ns": cost,
                }
            )
    deletion = [
        {"seed": seed, "prospective": True, "error": 0, "deleted_error": 1} for seed in range(20)
    ]
    comparisons = exp.build_comparison_rows(rows, deletion, draws=200)
    gate = exp.score_acceptance_gate(comparisons, violation_count=0)
    assert gate["primary_learning_gate_passed"] is True
    assert gate["secondary_compiled_deployment_gate_passed"] is True
    assert gate["secondary_can_rescue_primary"] is False

    failed = exp.score_acceptance_gate(comparisons, violation_count=1)
    assert failed["primary_learning_gate_passed"] is False
    assert failed["secondary_compiled_deployment_gate_passed"] is True
    assert exp._percentile([], 0.5) == 0.0


def test_blocked_artifact_is_row_free_and_names_failed_gate(tmp_path: Path) -> None:
    """SCENARIO-CL-7213-PRECONDITIONS: External absence gets a terminal diagnosis."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks = [exp.gate_check("fixture_missing", "exp7212", "path", True, False)]
    artifact = exp.build_blocked_artifact(
        checks,
        source_hashes={},
        paths=paths,
        duration_s=0.1,
    )
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["decision_rows"] == []
    assert artifact["query_rows"] == []
    assert artifact["commit_deletion_rows"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "fixture_missing"
    assert exp.validate_artifact(artifact) == []

    from_builder = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        upstream_artifact=tmp_path / "missing-upstream.json",
        duration_s=0.1,
        progress=True,
    )
    assert from_builder["verdict_class"] == "blocked"


def test_complete_build_has_raw_rows_checkpoint_and_cold_validation(tmp_path: Path) -> None:
    """SCENARIO-CL-7213-TERMINAL: A full run is complete even when value is null."""

    paths = exp.ExperimentPaths.under(tmp_path)
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        duration_s=1.25,
        progress=True,
    )
    assert artifact["status"] == "complete"
    assert artifact["refinement_run_complete_score"] == 1
    assert artifact["refinement_value_score"] in {0, 1}
    assert artifact["verdict_class"] in {"null", "circular_positive"}
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert len(artifact["rows"]) == len(exp.STREAM_SEEDS) * len(exp.ARMS)
    assert len(artifact["decision_rows"]) == 20 * 5 * 1024
    assert artifact["sample_size_budget"]["bootstrap_resamples_completed"] == 10_000
    assert Path(artifact["checkpoint_path"]).is_file()
    assert exp.validate_artifact(artifact, check_files=True) == []

    changed = dict(artifact)
    changed["decision_rows"] = list(artifact["decision_rows"])
    changed["decision_rows"][0] = dict(changed["decision_rows"][0])
    changed["decision_rows"][0]["error"] ^= 1
    assert "reproducibility_checksum" in exp.validate_artifact(changed)


def test_cli_writes_only_after_validation_and_rejects_wrong_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7213: The command validates before its final atomic publication."""

    paths = exp.ExperimentPaths.under(tmp_path)
    expected = exp.build_blocked_artifact(
        [exp.gate_check("forced", "test", "value", True, False)],
        source_hashes={},
        paths=paths,
        duration_s=0.1,
    )
    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: expected)
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)]) == 0
    written = json.loads(paths.artifact.read_text(encoding="utf-8"))
    assert written == expected

    with pytest.raises(ValueError, match="run_date_must_equal_20260911"):
        exp.main(["--date", "20260910", "--output-root", str(tmp_path)])

    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)])


def test_validator_rejects_missing_fields() -> None:
    """SCENARIO-CL-7213-TERMINAL: Incomplete owned work never validates."""

    assert exp.validate_artifact({})[0].startswith("missing_fields:")


def test_defensive_release_and_query_routing_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-7213-ACQUISITION: Invalid runtime routing cannot read feedback."""

    fallback = exp.exp7212.FrozenWarmupFallback.from_releases([])
    runtime = exp.ArmRuntime(arm="invalid", seed=1, fallback=fallback)
    event = {
        "event_id": "event",
        "family_id": "lower_bound",
        "numeric_value": 1,
        "chronology_index": exp.WARMUP_COUNT,
    }
    assert exp._query_decision(runtime, event, set()) is None
    runtime.arm = "witness_query_version_space"
    assert exp._query_decision(runtime, event, set()) is None
    pending = exp.PendingRecord(
        event=event,
        role="fitting",
        request_index=32,
        release_index=32,
        observed_label="accept",
        exact_label="accept",
        poisoned=False,
        query_row={},
    )
    with pytest.raises(RuntimeError, match="committed_release_requires_controller"):
        exp._apply_committed_release(runtime, pending)
    with pytest.raises(RuntimeError, match="version_release_requires_controller"):
        exp._apply_version_release(runtime, pending)
    with pytest.raises(RuntimeError, match="released_feedback_requires_learning_arm"):
        exp._apply_release(runtime, pending)
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp._require_valid(["forced"])


def test_repository_wrapper_invokes_module_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CL-7213: The executable wrapper delegates to the tested module."""

    monkeypatch.setattr(exp, "main", lambda: 7)
    wrapper = exp.REPO_ROOT / "scripts/experiments/experiment_7213_v635_refinement_learning.py"
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert stopped.value.code == 7
