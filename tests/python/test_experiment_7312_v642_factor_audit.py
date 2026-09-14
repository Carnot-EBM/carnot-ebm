"""Behavior tests for REQ-CL-7312 and SCENARIO-CL-7312-*."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7312_v642_factor_audit as audit


@pytest.fixture(scope="module")
def measured_audit(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[audit.ExperimentPaths, dict[str, object]]:
    """Build one cold stream from each stratum for shared audit assertions."""

    paths = audit.ExperimentPaths.under(tmp_path_factory.mktemp("exp7312-audit"))
    paths.upstream_artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.upstream_artifact.write_bytes((audit.REPO_ROOT / audit.UPSTREAM_ARTIFACT).read_bytes())
    artifact = audit.build_and_seal(
        audit.REPO_ROOT,
        paths,
        stream_ids=("evaluation-01", "evaluation-09", "evaluation-17"),
        bootstrap_draws=200,
        progress=True,
    )
    return paths, artifact


def test_req_cl_7312_freezes_aggregation_only_contract() -> None:
    """REQ-CL-7312 fixes the audit identity, seeds, arms, and no-model state."""

    spec = (audit.REPO_ROOT / audit.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-CL-7312" in spec
    assert len(set(audit.SCENARIO_PATTERN.findall(spec))) == 7
    assert audit.ARMS == audit.learning.ARMS
    assert audit.MODEL_SPECS == []
    assert audit.MODEL_INVOKED is False
    assert set(audit.INVOCATION_COUNTS.values()) == {0}
    assert audit.INFERENCE_SUBSTRATE == "aggregation_from_upstream_artifacts"
    assert audit.INFERENCE_SUBSTRATE_CLASS == "aggregation"
    assert set(audit.REQUIRED_ARTIFACT_FIELDS) <= set(audit.FIELD_PRINCIPLES)
    assert audit.ExperimentPaths.defaults().artifact == audit.REPO_ROOT / audit.DEFAULT_ARTIFACT


def test_req_cl_7312_helpers_fail_closed_on_malformed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7312 rejects malformed task, manifest, receipt, hash, and JSONL evidence."""

    assert audit._task_identity("[") == {}
    assert audit._task_identity("milestone: wrong") == {}
    assert audit._task_identity("[]") == {}
    assert audit._manifest_excludes(
        {"nested": [{"experiment_ids": ["exp7312-factor-audit"]}]}, 7312
    )
    assert audit._manifest_excludes({"nested": {"experiment_id": "7312"}}, 7312)
    assert audit._receipt_matches(audit.REPO_ROOT, None) is False

    upstream = json.loads((audit.REPO_ROOT / audit.UPSTREAM_ARTIFACT).read_text())
    malformed = deepcopy(upstream)
    malformed["source_artifact_hashes"] = []
    assert (
        audit._relevant_upstream_hash_mismatches(audit.REPO_ROOT, malformed)[0]["path"]
        == "source_artifact_hashes"
    )
    mismatched = deepcopy(upstream)
    source_path = next(
        path
        for path in mismatched["source_artifact_hashes"]
        if "/python/carnot/experiment_7311_" in path
    )
    mismatched["source_artifact_hashes"][source_path] = "sha256:" + "0" * 64
    assert audit._relevant_upstream_hash_mismatches(audit.REPO_ROOT, mismatched) == [
        {
            "path": source_path,
            "expected": "sha256:" + "0" * 64,
            "observed": audit._sha256_path(Path(source_path)),
        }
    ]

    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('\n{"stream_id":"evaluation-01"}\n1\n')
    iterator = audit._iter_jsonl(rows_path)
    assert next(iterator) == {"stream_id": "evaluation-01"}
    with pytest.raises(ValueError, match="invalid_jsonl_row"):
        next(iterator)

    bad_fixture = deepcopy(upstream)
    bad_fixture["source_artifact_states"]["exp7310"]["sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="exp7310_artifact_hash"):
        audit._fixture_artifact(audit.REPO_ROOT, bad_fixture)
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text("{}")
    not_ready = deepcopy(upstream)
    not_ready["source_artifact_states"]["exp7310"] = {
        "path": str(fixture_path),
        "sha256": audit._sha256_path(fixture_path),
    }
    with pytest.raises(ValueError, match="exp7310_fixture_not_ready"):
        audit._fixture_artifact(audit.REPO_ROOT, not_ready)

    paths = audit.ExperimentPaths.under(tmp_path / "yaml-error")
    paths.upstream_artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.upstream_artifact.write_bytes((audit.REPO_ROOT / audit.UPSTREAM_ARTIFACT).read_bytes())
    monkeypatch.setattr(
        audit.yaml, "safe_load", lambda _text: (_ for _ in ()).throw(audit.yaml.YAMLError())
    )
    checks, _, _ = audit.collect_preconditions(audit.REPO_ROOT, paths)
    assert next(row for row in checks if row["check"] == "v642_task_identity")["passed"] is False


def test_scenario_cl_7312_preconditions_preserve_exact_failure(tmp_path: Path) -> None:
    """SCENARIO-CL-7312-PRECONDITIONS keeps the exact failed upstream field."""

    paths = audit.ExperimentPaths.under(tmp_path)
    upstream = json.loads((audit.REPO_ROOT / audit.UPSTREAM_ARTIFACT).read_text())
    upstream["factor_capture_complete_score"] = 0
    paths.upstream_artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.upstream_artifact.write_text(json.dumps(upstream))

    checks, hashes, _ = audit.collect_preconditions(audit.REPO_ROOT, paths)
    capture = next(row for row in checks if row["check"] == "factor_capture_complete")
    assert capture == {
        "check": "factor_capture_complete",
        "upstream": str(paths.upstream_artifact),
        "field": "factor_capture_complete_score",
        "expected_value": 1,
        "observed_value": 0,
        "passed": False,
        "principle": "Only complete prospective capture can authorize the audit.",
    }
    blocked = audit.build_blocked_artifact(checks, hashes)
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == blocked["causal_intervention_rows"] == []
    assert blocked["gate_check_summary"]["first_failure"] == capture
    assert "factor_capture_complete_score" in blocked["honest_verdict"]
    assert audit.validate_artifact(blocked) == []

    upstream["factor_capture_complete_score"] = 1
    upstream["flagged_adversarial"] = True
    upstream["verdict_class"] = "disqualified"
    paths.upstream_artifact.write_text(json.dumps(upstream))
    checks, _, _ = audit.collect_preconditions(audit.REPO_ROOT, paths)
    assert (
        next(row for row in checks if row["check"] == "upstream_not_quarantined")["passed"] is False
    )
    assert (
        next(row for row in checks if row["check"] == "upstream_not_disqualified")["passed"]
        is False
    )

    with pytest.raises(ValueError, match="blocked_artifact_without_failure"):
        audit.build_blocked_artifact([], {})


def test_scenario_cl_7312_cold_replay_reconstructs_every_selected_unit(
    measured_audit: tuple[audit.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7312-COLD-REPLAY checks predictions and state transitions."""

    _, artifact = measured_audit
    parity = artifact["cold_replay_parity"]
    assert parity["fresh_process"] is True
    assert parity["producer_aggregate_accessed"] is False
    assert parity["private_regime_used_for_prediction"] is False
    assert parity["matched_prediction_rows"] == 3 * 5 * 896
    assert parity["matched_transition_rows"] == 3 * 5 * 128
    assert parity["prediction_mismatch_count"] == 0
    assert parity["state_hash_mismatch_count"] == 0
    assert parity["transition_mismatch_count"] == 0
    assert parity["mismatch_rows"] == []
    assert len(artifact["rows"]) == 3 * 5


def test_scenario_cl_7312_cold_worker_and_mismatch_paths_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-CL-7312-COLD-REPLAY retains exact direct-worker mismatch classes."""

    paths = audit.ExperimentPaths.under(tmp_path)
    paths.upstream_artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.upstream_artifact.write_bytes((audit.REPO_ROOT / audit.UPSTREAM_ARTIFACT).read_bytes())
    cold = audit.cold_replay_impl(audit.REPO_ROOT, paths, ("evaluation-01",))
    assert cold["cold_replay_parity"]["matched_prediction_rows"] == 5 * 896
    assert cold["cold_replay_parity"]["matched_transition_rows"] == 5 * 128

    step = {
        "unit_id": "s:a:1",
        "prediction": "accept",
        "state_hash_before_prediction": "before",
        "state_hash_at_seal": "after",
        "prediction_time_ns": 1,
        "nested": [{"prediction_cost_ns": 1, "kept": True}],
    }
    volatile_only = {
        **step,
        "prediction_time_ns": 2,
        "nested": [{"prediction_cost_ns": 2, "kept": True}],
    }
    feedback = {"unit_id": "s:release:1", "arm_updates": {"update_latency_ns": 1}}
    stable = audit._compare_stream([step], [feedback], [volatile_only], [feedback])
    assert stable["matched_prediction_rows"] == 1
    assert stable["matched_transition_rows"] == len(audit.ARMS)

    changed = {
        **volatile_only,
        "prediction": "reject",
        "state_hash_before_prediction": "wrong",
    }
    mismatch = audit._compare_stream([step], [feedback], [changed], [])
    assert mismatch["matched_prediction_rows"] == 0
    assert mismatch["prediction_mismatch_count"] == 1
    assert mismatch["state_hash_mismatch_count"] == 1
    assert mismatch["transition_mismatch_count"] == len(audit.ARMS)
    assert {row["kind"] for row in mismatch["mismatch_rows"]} == {
        "prediction_or_state",
        "state_transition",
    }


def test_scenario_cl_7312_interventions_have_independent_outcomes(
    measured_audit: tuple[audit.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7312-INTERVENTIONS checks all attacks and corrupt rollback."""

    _, artifact = measured_audit
    rows = artifact["causal_intervention_rows"]
    assert {row["control"] for row in rows} == set(audit.CONTROL_NAMES)
    assert all(row["passed"] is True for row in rows)
    assert all(row["evaluation_rows_immutable"] is True for row in rows)
    by_name = {row["control"]: row for row in rows}
    assert by_name["future_label_permutation"]["observed"]["earlier_prediction_invariant"] is True
    assert by_name["consistent_factor_name_permutation"]["observed"]["decision_invariant"] is True
    assert by_name["retained_witness_erasure"]["observed"]["prediction_difference_count"] > 0
    assert (
        by_name["delay_all_feedback_beyond_evaluation"]["observed"]["evaluation_update_count"] == 0
    )
    rollback = by_name["corrupt_rollback_hash"]["observed"]
    assert rollback["error"] == "invalid_rollback"
    assert rollback["state_bytes_unchanged_after_rejection"] is True


def test_scenario_cl_7312_intervals_keep_denominators_labels_and_memory_equal(
    measured_audit: tuple[audit.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7312-INTERVALS keeps streams, strata, labels, and caps exact."""

    _, artifact = measured_audit
    intervals = artifact["independent_stream_intervals"]
    assert intervals
    assert {row["stratum"] for row in intervals} == {
        "overall",
        "isolated_factor_changes",
        "overlapping_factor_changes",
        "stationary_controls",
    }
    assert {row["independent_unit"] for row in intervals} == {"stream"}
    assert {row["metric"] for row in intervals} >= {
        "future_error_rate",
        "non_feedback_error_rate",
        "false_accept_rate",
        "coverage_rate",
    }
    parity = artifact["arm_parity"]
    assert parity["same_label_schedule"] is True
    assert parity["same_evaluator_labels"] is True
    assert parity["same_memory_cap"] is True
    assert parity["arms"] == list(audit.ARMS)
    assert all(row["censored"] is False for row in artifact["rows"])


def test_scenario_cl_7312_causality_requires_outcomes_not_state_activity(
    measured_audit: tuple[audit.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7312-CAUSALITY separates witness value from byte changes."""

    _, artifact = measured_audit
    causal = artifact["causal_summary"]
    assert causal["legitimate_later_changed_prediction_count"] > 0
    assert causal["state_change_alone_credited"] is False
    assert causal["retained_vs_erased_prediction_difference_count"] > 0
    assert causal["future_error_delta_vs_witness_erased"] < 0
    assert artifact["factor_promotion_score"] == 0


def test_scenario_cl_7312_terminal_keeps_audit_and_promotion_separate() -> None:
    """SCENARIO-CL-7312-TERMINAL keeps a complete efficacy failure null."""

    assert audit.derive_terminal_scores(True, False, True) == (1, 0, "null")
    assert audit.derive_terminal_scores(True, True, True) == (1, 1, "circular_positive")
    assert audit.derive_terminal_scores(True, True, False) == (1, 1, "positive")
    assert audit.derive_terminal_scores(False, True, True) == (0, 0, "partial")
    assert audit._terminal_verdict("circular_positive", []).startswith(
        "complete_circular_positive:"
    )
    assert audit.RETIREMENT_SCOPE in audit._terminal_verdict("null", ["safety"])
    assert audit._terminal_verdict("partial", []).startswith("partial:")
    assert audit.build_independent_stream_intervals([]) == []
    with pytest.raises(ValueError, match="paired_streams_unavailable"):
        audit._interval([], 10, "empty")


def test_scenario_cl_7312_e2e_validates_and_writes_atomic_private_shard(
    measured_audit: tuple[audit.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7312-E2E binds replay, controls, reduction, and terminal bytes."""

    paths, artifact = measured_audit
    assert [row["stage"] for row in artifact["e2e_rows"]] == list(audit.E2E_STAGES)
    assert artifact["status"] == "partial"
    assert artifact["factor_audit_complete_score"] == 0
    assert artifact["factor_promotion_score"] == 0
    assert artifact["sample_size_budget"]["completed_prediction_rows"] == 3 * 5 * 896
    assert (
        audit.validate_artifact(
            artifact,
            expected_stream_ids=("evaluation-01", "evaluation-09", "evaluation-17"),
            check_files=True,
        )
        == []
    )

    changed = deepcopy(artifact)
    changed["cold_replay_parity"]["prediction_mismatch_count"] = 1
    assert "checksum" in audit.validate_artifact(changed)
    assert "cold_replay_parity" in audit.validate_artifact(changed)

    receipt = {
        "command": "focused-test",
        "exit_code": 0,
        "duration_s": 0.1,
        "log_sha256": "sha256:" + "0" * 64,
        "scope": "REQ-CL-7312 private validation",
    }
    sealed = audit.attach_validation_receipts(artifact, [receipt])
    output = paths.artifact
    audit.write_artifact(
        output,
        sealed,
        expected_stream_ids=("evaluation-01", "evaluation-09", "evaluation-17"),
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "partial"
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        audit.attach_validation_receipts(artifact, [{"command": "missing fields"}])
    invalid = deepcopy(artifact)
    invalid["schema"] = "wrong"
    invalid["reproducibility_checksum"] = audit.reproducibility_checksum(invalid)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        audit.write_artifact(
            paths.artifact,
            invalid,
            expected_stream_ids=("evaluation-01", "evaluation-09", "evaluation-17"),
        )


def test_scenario_cl_7312_precondition_block_stops_before_replay(tmp_path: Path) -> None:
    """SCENARIO-CL-7312-PRECONDITIONS makes an external failure terminal blocked."""

    paths = audit.ExperimentPaths.under(tmp_path)
    upstream = json.loads((audit.REPO_ROOT / audit.UPSTREAM_ARTIFACT).read_text())
    upstream["factor_capture_complete_score"] = 0
    paths.upstream_artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.upstream_artifact.write_text(json.dumps(upstream))
    artifact = audit.build_and_seal(audit.REPO_ROOT, paths)
    assert artifact["status"] == "blocked"
    assert artifact["rows"] == []


def test_req_cl_7312_cold_subprocess_failures_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7312 does not turn worker failure or malformed output into success."""

    paths = audit.ExperimentPaths.under(tmp_path)
    failed = {
        "command": "worker",
        "scope": "cold",
        "exit_code": 1,
        "duration_s": 0.1,
        "log_sha256": "sha256:" + "0" * 64,
    }
    monkeypatch.setattr(audit.learning, "_command_receipt", lambda *_args, **_kwargs: (failed, ""))
    with pytest.raises(RuntimeError, match="cold_replay_failed"):
        audit.audit_raw_evidence(audit.REPO_ROOT, paths, ("evaluation-01",))

    passed = {**failed, "exit_code": 0}
    monkeypatch.setattr(audit.learning, "_command_receipt", lambda *_args, **_kwargs: (passed, ""))
    with pytest.raises(ValueError, match="cold_replay_schema"):
        audit.audit_raw_evidence(audit.REPO_ROOT, paths, ("evaluation-01",))


def test_req_cl_7312_thin_entrypoint_and_validation_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7312 keeps orchestration reusable and validation task-scoped."""

    commands = audit._validation_commands(tmp_path / "candidate.json")
    flat = [part for command in commands for part in command]
    assert "scripts/check_spec_coverage.py" in flat
    assert "scripts/adversarial_verify.py" in flat
    assert "scripts/verdict_row_consistency_lint.py" in flat
    assert "--strict" in flat
    assert "mypy" in flat
    assert "ruff" in flat

    wrapper = audit.REPO_ROOT / audit.WRAPPER_PATH
    source = wrapper.read_text(encoding="utf-8")
    assert len(source.splitlines()) <= 24
    monkeypatch.setattr(audit, "main", lambda: 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert stopped.value.code == 0
