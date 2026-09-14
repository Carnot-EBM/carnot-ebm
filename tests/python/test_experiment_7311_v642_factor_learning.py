"""Behavior tests for REQ-CL-7311 prospective factor learning."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7311_v642_factor_learning as exp


def _real_views() -> exp.prototype.StreamViews:
    paths = exp.ExperimentPaths.defaults()
    checks, _, upstream = exp.collect_preconditions(exp.REPO_ROOT, paths)
    assert exp.gate_check_summary(checks)["passed"] is True
    return exp.load_authenticated_views(exp.REPO_ROOT, upstream)


def test_scenario_cl_7311_preconditions_preserve_exact_upstream_failure(tmp_path: Path) -> None:
    """SCENARIO-CL-7311-PRECONDITIONS: readiness and quarantine fail exactly."""

    paths = exp.ExperimentPaths.under(tmp_path)
    upstream = json.loads((exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT).read_text())
    upstream["factor_fixture_ready_score"] = 0
    paths.upstream_artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.upstream_artifact.write_text(json.dumps(upstream))

    checks, hashes, _ = exp.collect_preconditions(exp.REPO_ROOT, paths)
    readiness = next(row for row in checks if row["check"] == "factor_fixture_ready")
    assert readiness == {
        "check": "factor_fixture_ready",
        "upstream": str(paths.upstream_artifact),
        "field": "factor_fixture_ready_score",
        "expected_value": 1,
        "observed_value": 0,
        "passed": False,
        "principle": "Only the complete bounded factor fixture can authorize evaluation.",
    }
    blocked = exp.build_blocked_artifact(checks, hashes)
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == blocked["feedback_update_rows"] == []
    assert "factor_fixture_ready_score" in blocked["honest_verdict"]
    assert exp.validate_artifact(blocked) == []
    receipt = exp.write_artifact(tmp_path / "blocked.json", blocked)
    assert receipt["sha256"] == exp._sha256_path(tmp_path / "blocked.json")

    upstream["factor_fixture_ready_score"] = 1
    upstream["flagged_adversarial"] = True
    paths.upstream_artifact.write_text(json.dumps(upstream))
    checks, _, _ = exp.collect_preconditions(exp.REPO_ROOT, paths)
    quarantine = next(row for row in checks if row["check"] == "upstream_not_quarantined")
    assert quarantine["passed"] is False

    terminal_block = exp.build_and_seal(exp.REPO_ROOT, paths, progress=True)
    assert terminal_block["status"] == "blocked"
    with pytest.raises(ValueError, match="blocked_artifact_without_failure"):
        exp.build_blocked_artifact([], {})


def test_scenario_cl_7311_preconditions_rejects_malformed_helpers(tmp_path: Path) -> None:
    """SCENARIO-CL-7311-PRECONDITIONS: malformed identities and rows fail closed."""

    missing = tmp_path / "missing"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    scalar_rows = tmp_path / "scalar.jsonl"
    scalar_rows.write_text("\n1\n")

    assert exp._sha256_path(missing) is None
    assert exp._load_object(missing) == {}
    assert exp._load_object(malformed) == {}
    assert exp._task_identity("[") == {}
    assert exp._task_identity("milestone: wrong\ntasks: []\n") == {}
    assert exp._task_identity(f"milestone: {exp.MILESTONE}\ntasks: []\n") == {}
    assert exp._receipt_authenticates(exp.REPO_ROOT, None) is False
    with pytest.raises(ValueError, match="invalid_jsonl_row"):
        exp._read_jsonl(scalar_rows)


def test_scenario_cl_7311_preconditions_rejects_each_view_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7311-PRECONDITIONS: stream content and authority seals fail closed."""

    _, _, upstream = exp.collect_preconditions(exp.REPO_ROOT, exp.ExperimentPaths.defaults())
    malformed = deepcopy(upstream)
    malformed["stream_manifest"]["receipts"] = []
    with pytest.raises(ValueError, match="evaluation_receipts"):
        exp.load_authenticated_views(exp.REPO_ROOT, malformed)

    bad_hash = deepcopy(upstream)
    bad_hash["stream_manifest"]["receipts"]["evaluation_public"]["sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="view_hash:evaluation_public"):
        exp.load_authenticated_views(exp.REPO_ROOT, bad_hash)

    real_builder = exp.prototype.build_stream_views
    altered = real_builder("evaluation")
    altered.public[0] = {**altered.public[0], "numeric_value": -1}
    monkeypatch.setattr(exp.prototype, "build_stream_views", lambda _kind: altered)
    with pytest.raises(ValueError, match="sealed_view_content"):
        exp.load_authenticated_views(exp.REPO_ROOT, upstream)
    monkeypatch.setattr(exp.prototype, "build_stream_views", real_builder)

    monkeypatch.setattr(exp.prototype, "stream_conformance_errors", lambda *_args: ["bad"])
    with pytest.raises(ValueError, match="stream_conformance"):
        exp.load_authenticated_views(exp.REPO_ROOT, upstream)
    monkeypatch.setattr(exp.prototype, "stream_conformance_errors", lambda *_args: [])
    bad_shuffle = deepcopy(upstream)
    bad_shuffle["stream_manifest"]["shuffled_label_hashes"]["evaluation"] = "sha256:bad"
    with pytest.raises(ValueError, match="shuffled_label_hash"):
        exp.load_authenticated_views(exp.REPO_ROOT, bad_shuffle)


def test_scenario_cl_7311_chronology_uses_pipeline_before_due_feedback(tmp_path: Path) -> None:
    """SCENARIO-CL-7311-CHRONOLOGY/PIPELINE: one real stream uses the hook in order."""

    views = _real_views()
    steps, updates, controls = exp.replay_stream(
        views,
        "evaluation-01",
        tmp_path / "state",
    )

    assert len(steps) == len(exp.ARMS) * (exp.EVENTS_PER_STREAM - exp.WARMUP_COUNT)
    assert len(updates) == exp.FUTURE_LABEL_COUNT
    assert exp.step_row_errors(steps, ("evaluation-01",)) == []
    assert exp.feedback_row_errors(updates, ("evaluation-01",)) == []
    assert all(row["actual_pipeline_hook"] for row in steps)
    assert all(
        row["prediction_order"] < row["evaluator_order"] < row["release_order"]
        for row in steps
        if row["release_order"] is not None
    )
    assert controls["cold_restart"]["passed"] is True
    assert controls["rollback"]["passed"] is True

    disabled = exp.prototype.FactorPipelineHook(tmp_path / "disabled")
    with pytest.raises(exp.prototype.FactorRevisionRejected, match="factor_hook_disabled"):
        exp._controller(disabled)
    with pytest.raises(ValueError, match="incomplete_stream"):
        exp.replay_stream(views, "missing-stream", tmp_path / "missing")


def test_scenario_cl_7311_accounting_catches_same_label_and_missing_rows(tmp_path: Path) -> None:
    """SCENARIO-CL-7311-ACCOUNTING: state activity alone is not future progress."""

    steps, updates, _ = exp.replay_stream(
        _real_views(),
        "evaluation-02",
        tmp_path / "state",
    )
    reduced = exp.reduce_step_rows(steps, updates)
    factor = next(row for row in reduced if row["arm"] == exp.FACTOR_ARM)

    assert factor["future_prediction_count"] == 896
    assert factor["future_label_count"] == 128
    assert factor["state_change_update_count"] >= factor["factor_change_update_count"]
    assert factor["later_changed_prediction_count"] == sum(
        row["later_prediction_changed"] for row in updates
    )
    assert all(
        row["first_later_changed_prediction_event_id"] is None
        or row["first_later_changed_prediction_index"] > row["release_index"]
        for row in updates
    )

    assert "step_row_count" in exp.step_row_errors(steps[:-1], ("evaluation-02",))
    tampered = deepcopy(updates)
    tampered[0]["prediction_persisted_before_release"] = False
    assert "feedback_chronology" in exp.feedback_row_errors(tampered, ("evaluation-02",))


def test_scenario_cl_7311_reduction_bootstraps_streams_not_events() -> None:
    """SCENARIO-CL-7311-REDUCTION: paired intervals use independent stream rows."""

    rows = []
    for number, stratum in enumerate(exp.STRATA, start=1):
        for arm, rate in (
            (exp.FACTOR_ARM, 0.10),
            ("global_reset_on_contradiction", 0.20),
            ("local_reset_without_retained_witnesses", 0.18),
            ("frozen_warmup", 0.12),
            ("label_shuffled_factor_local_revision", 0.22),
        ):
            rows.append(
                {
                    "stream_id": f"evaluation-{number:02d}",
                    "stratum": stratum,
                    "arm": arm,
                    "future_error_rate": rate,
                    "non_feedback_error_rate": rate,
                    "recurrence_error_rate": rate,
                    "false_accept_rate": rate / 10,
                    "coverage_rate": 0.95,
                }
            )

    comparisons = exp.build_comparison_rows(rows)
    assert comparisons
    assert all(row["bootstrap_draws"] == 10_000 for row in comparisons)
    assert all(row["independent_unit"] == "stream" for row in comparisons)
    future = next(
        row
        for row in comparisons
        if row["comparison_id"] == "future_error_vs_global_reset" and row["stratum"] == "overall"
    )
    assert future["paired_stream_count"] == 3
    assert future["ci95_upper"] < 0
    assert exp.build_comparison_rows([]) == []
    with pytest.raises(ValueError, match="paired_streams_unavailable"):
        exp._bootstrap_interval([], "empty")
    with pytest.raises(StopIteration):
        exp._comparison(comparisons, "missing")


def test_scenario_cl_7311_e2e_credits_only_later_prediction(tmp_path: Path) -> None:
    """SCENARIO-CL-7311-E2E: the control credits a future pre-label prediction."""

    result = exp.run_e2e_controls(_real_views(), tmp_path)
    repeated = exp.run_e2e_controls(_real_views(), tmp_path)

    assert result["prediction_before_release"] is not None
    assert result["release_state_hash_changed"] is True
    assert result["later_prediction_index"] > result["release_index"]
    assert result["later_prediction_was_pre_label"] is True
    assert result["cold_restart_parity"] is True
    assert result["rollback_byte_identical"] is True
    assert repeated == result


def test_scenario_cl_7311_terminal_keeps_capture_separate_from_value() -> None:
    """SCENARIO-CL-7311-TERMINAL: complete failed efficacy is an honest null."""

    capture = {
        name: exp.gate(True, True, True, "complete capture") for name in exp.CAPTURE_GATE_NAMES
    }
    value = {
        name: exp.gate("frozen", "failed", False, "frozen scientific check")
        for name in exp.VALUE_GATE_NAMES
    }
    scores = exp.derive_terminal_scores({**capture, **value})

    assert scores["factor_capture_complete_score"] == 1
    assert scores["factor_value_score"] == 0
    assert scores["verdict_class"] == "null"
    assert scores["honest_verdict"].startswith("complete_null:")

    passed = {
        name: exp.gate(True, True, True, "complete")
        for name in (*exp.CAPTURE_GATE_NAMES, *exp.VALUE_GATE_NAMES)
    }
    promoted = exp.derive_terminal_scores(passed)
    assert promoted["factor_value_score"] == 1
    assert promoted["verdict_class"] == "circular_positive"


def test_scenario_cl_7311_terminal_private_build_runs_cold_reducer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7311-TERMINAL: a bounded private shard exercises the full builder."""

    paths = exp.ExperimentPaths.from_results_root(
        tmp_path,
        exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT,
    )
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        stream_ids=("evaluation-03", "evaluation-10", "evaluation-18"),
        progress=True,
    )

    assert artifact["status"] == "complete"
    assert artifact["factor_capture_complete_score"] == 0
    assert artifact["verdict_class"] == "partial"
    assert artifact["sample_size_budget"]["completed_post_warmup_predictions"] == 13_440
    assert (
        exp.validate_artifact(
            artifact,
            expected_stream_ids=("evaluation-03", "evaluation-10", "evaluation-18"),
            check_files=True,
        )
        == []
    )

    resumed = exp.run_learning_panel(
        _real_views(),
        paths,
        stream_ids=("evaluation-03", "evaluation-10", "evaluation-18"),
        progress=True,
    )
    assert resumed.completed_stream_ids == ["evaluation-03", "evaluation-10", "evaluation-18"]

    bad_steps = tmp_path / "bad_steps.jsonl"
    bad_feedback = tmp_path / "bad_feedback.jsonl"
    bad_steps.write_text('{"stream_id":"bad"}\n')
    bad_feedback.write_text("")
    with pytest.raises(ValueError, match="raw_row_conformance"):
        exp.independent_reduce(bad_steps, bad_feedback)

    valid_receipt = {
        "command": "private validation",
        "scope": "REQ-CL-7311",
        "exit_code": 0,
        "duration_s": 0.1,
        "log_sha256": "sha256:" + "0" * 64,
    }
    attached = exp.attach_validation_receipts(artifact, [valid_receipt])
    assert attached["validation_receipts"][-1] == valid_receipt
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        exp.attach_validation_receipts(artifact, [{}])

    incomplete = deepcopy(artifact)
    incomplete["status"] = "working"
    incomplete["reproducibility_checksum"] = exp.reproducibility_checksum(incomplete)
    assert "status" in exp.validate_artifact(incomplete)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(tmp_path / "invalid.json", incomplete)

    missing_raw = deepcopy(artifact)
    missing_raw["raw_evidence_receipts"]["step_rows"]["path"] = str(tmp_path / "absent")
    missing_raw["reproducibility_checksum"] = exp.reproducibility_checksum(missing_raw)
    assert "cold_reducer" in exp.validate_artifact(missing_raw, check_files=True)

    panel = exp.EvaluationPanel([], [], [], [], [], ["evaluation-01"])
    monkeypatch.setattr(exp, "run_learning_panel", lambda *_args, **_kwargs: panel)
    with pytest.raises(RuntimeError, match="measurement_limit_censored"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.from_results_root(
                tmp_path / "censored",
                exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT,
            ),
        )


def test_scenario_cl_7311_accounting_checkpoint_errors_timeout_and_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7311-ACCOUNTING: checkpoints fail closed and timeout stays explicit."""

    views = _real_views()
    paths = exp.ExperimentPaths.from_results_root(
        tmp_path,
        exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT,
    )
    checkpoint = paths.checkpoint_dir / "evaluation-01.json"
    exp._atomic_write(
        checkpoint,
        {
            "schema": exp.SCHEMA,
            "status": "complete",
            "stream_id": "evaluation-01",
            "step_rows": [],
            "feedback_update_rows": [],
            "controls": {},
        },
    )
    with pytest.raises(ValueError, match="stream_conformance"):
        exp.run_learning_panel(views, paths, stream_ids=("evaluation-01",))

    timeout_paths = exp.ExperimentPaths.from_results_root(
        tmp_path / "timeout",
        exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT,
    )
    monotonic_values = iter((0.0, 1.0))
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(monotonic_values))
    timed = exp.run_learning_panel(
        views,
        timeout_paths,
        stream_ids=("evaluation-01",),
        measurement_limit_s=0.5,
    )
    assert timed.censored_stream_ids == ["evaluation-01"]

    valid_steps, valid_feedback, valid_controls = exp.replay_stream(
        views,
        "evaluation-01",
        tmp_path / "heartbeat-state",
    )
    heartbeat_paths = exp.ExperimentPaths.from_results_root(
        tmp_path / "heartbeat",
        exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT,
    )
    exp._atomic_write(
        heartbeat_paths.checkpoint_dir / "evaluation-01.json",
        {
            "schema": exp.SCHEMA,
            "status": "complete",
            "stream_id": "evaluation-01",
            "step_rows": valid_steps,
            "feedback_update_rows": valid_feedback,
            "controls": valid_controls,
        },
    )
    monotonic_values = iter((0.0, 0.0, 61.0))
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(monotonic_values))
    heartbeat = exp.run_learning_panel(
        views,
        heartbeat_paths,
        stream_ids=("evaluation-01",),
        progress=True,
    )
    assert heartbeat.completed_stream_ids == ["evaluation-01"]


def test_scenario_cl_7311_terminal_rejects_reducer_and_validator_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7311-TERMINAL: both independent terminal guards reject mismatch."""

    passed_check = exp.precondition_gate("synthetic", "test", "field", True, True, "pass")
    monkeypatch.setattr(
        exp,
        "collect_preconditions",
        lambda *_args: ([passed_check], {}, {"schema": exp.prototype.SCHEMA}),
    )
    monkeypatch.setattr(exp, "load_authenticated_views", lambda *_args: object())
    mismatch_panel = exp.EvaluationPanel([], [], [{"different": True}], [], [], [])
    monkeypatch.setattr(exp, "run_learning_panel", lambda *_args, **_kwargs: mismatch_panel)
    monkeypatch.setattr(exp, "independent_reduce", lambda *_args: [])
    paths = exp.ExperimentPaths.from_results_root(tmp_path, tmp_path / "upstream.json")
    with pytest.raises(ValueError, match="independent_reducer_mismatch"):
        exp.build_and_seal(exp.REPO_ROOT, paths, stream_ids=("evaluation-01",))

    reduced = [
        {
            "arm": exp.FACTOR_ARM,
            "later_changed_prediction_count": 0,
            "time_limit_violations": 0,
            "byte_limit_violations": 0,
            "rejected_update_count": 0,
            "abstention_count": 0,
        }
    ]
    final_panel = exp.EvaluationPanel(
        [{"memory_bytes": 1, "memory_categories": {}}],
        [],
        reduced,
        [
            {
                "cold_restart": {"passed": True},
                "rollback": {"passed": True},
            }
        ],
        ["evaluation-01"],
        [],
    )
    monkeypatch.setattr(exp, "run_learning_panel", lambda *_args, **_kwargs: final_panel)
    monkeypatch.setattr(exp, "independent_reduce", lambda *_args: reduced)
    monkeypatch.setattr(exp, "build_comparison_rows", lambda *_args: [])
    monkeypatch.setattr(
        exp,
        "score_value_gates",
        lambda *_args: {
            name: exp.gate(True, False, False, "synthetic") for name in exp.VALUE_GATE_NAMES
        },
    )
    monkeypatch.setattr(
        exp,
        "run_e2e_controls",
        lambda *_args: {
            "cold_restart_parity": True,
            "rollback_byte_identical": True,
            "actual_pipeline_hook": True,
        },
    )
    monkeypatch.setattr(exp, "step_row_errors", lambda *_args: [])
    monkeypatch.setattr(exp, "feedback_row_errors", lambda *_args: [])
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forced"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp.build_and_seal(exp.REPO_ROOT, paths, stream_ids=("evaluation-01",))


@pytest.mark.parametrize("field", ["model_invoked", "no_model_weight_mutation"])
def test_scenario_cl_7311_terminal_validation_rejects_boundary_tampering(field: str) -> None:
    """SCENARIO-CL-7311-TERMINAL: validator rejects model or weight boundary changes."""

    check = exp.precondition_gate("synthetic", "test", "field", True, False, "failure")
    artifact = exp.build_blocked_artifact([check], {"test": None})
    artifact[field] = not artifact[field]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)

    errors = exp.validate_artifact(artifact)
    assert "learning_boundary" in errors
