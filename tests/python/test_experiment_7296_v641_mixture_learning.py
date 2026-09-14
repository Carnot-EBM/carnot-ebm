"""Verify prospective fixed-share efficacy under delayed labels.

Spec refs: REQ-CL-7296 and SCENARIO-CL-7296-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7296_v641_mixture_learning as exp


@pytest.fixture(scope="module")
def measured_stream(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[exp.ExperimentPaths, exp.EvaluationPanel]:
    """Run one authentic sealed stream once for the focused behavior tests."""

    paths = exp.ExperimentPaths.under(tmp_path_factory.mktemp("exp7296-panel"))
    upstream = exp._load_object(exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT)
    views = exp.load_authenticated_views(exp.REPO_ROOT, upstream)
    panel = exp.run_learning_panel(
        views,
        paths,
        stream_ids=("evaluation-01",),
        progress=True,
    )
    return paths, panel


def test_req_cl_7296_freezes_prospective_contract() -> None:
    """REQ-CL-7296 freezes streams, arms, labels, bytes, draws, and no LLM work."""

    spec = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-CL-7296" in spec
    assert len(set(exp.SCENARIO_PATTERN.findall(spec))) == 6
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert set(exp.INVOCATION_COUNTS.values()) == {0}
    assert exp.EVALUATION_STREAM_SEEDS == tuple(range(7_295_101, 7_295_125))
    assert exp.ARMS == exp.fixture.ARMS
    assert (exp.EVENTS_PER_STREAM, exp.WARMUP_COUNT, exp.FUTURE_LABEL_COUNT) == (1024, 128, 128)
    assert (exp.FEEDBACK_DELAY, exp.ETA, exp.FIXED_SHARE) == (4, 0.5, 0.02)
    assert (exp.MEMORY_CAP_BYTES, exp.BOOTSTRAP_RESAMPLES) == (69_632, 10_000)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(exp.FIELD_PRINCIPLES)


def test_scenario_cl_7296_preconditions_authenticate_fixture_and_manifest(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7296-PRECONDITIONS binds readiness and exact stream bytes."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, hashes, upstream = exp.collect_preconditions(exp.REPO_ROOT, paths)
    assert all(row["passed"] is True for row in checks)
    assert upstream["mixture_fixture_ready_score"] == 1
    assert hashes[str(exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT)] is not None
    views = exp.load_authenticated_views(exp.REPO_ROOT, upstream)
    assert len(views.public) == 24 * 1_024
    assert len(views.releases) == 24 * 256
    assert len(views.authority) == 24 * 1_024
    assert views.manifest["scorer_only_evaluation_labels"] is True


def test_scenario_cl_7296_chronology_records_causal_updates(
    measured_stream: tuple[exp.ExperimentPaths, exp.EvaluationPanel],
) -> None:
    """SCENARIO-CL-7296-CHRONOLOGY predicts before each delayed update."""

    _, panel = measured_stream
    assert len(panel.step_rows) == len(exp.ARMS) * (exp.EVENTS_PER_STREAM - exp.WARMUP_COUNT)
    assert len(panel.feedback_update_rows) == exp.FUTURE_LABEL_COUNT
    assert exp.step_row_errors(panel.step_rows, ("evaluation-01",)) == []
    assert exp.feedback_row_errors(panel.feedback_update_rows, ("evaluation-01",)) == []
    assert all(row["prediction_order"] < row["evaluator_order"] for row in panel.step_rows)
    assert all(
        row["release_index"] == row["source_index"] + 4 for row in panel.feedback_update_rows
    )
    assert all(
        row["prediction_persisted_before_release"] is True for row in panel.feedback_update_rows
    )
    assert any(
        row["arm_updates"]["fixed_share_mixture"]["weights_before"]
        != row["arm_updates"]["fixed_share_mixture"]["weights_after"]
        for row in panel.feedback_update_rows
    )
    assert panel.later_changed_prediction_count >= 1
    bounded = next(row for row in panel.per_stream_results if row["arm"] == "fixed_share_mixture")
    reference = next(
        row for row in panel.per_stream_results if row["arm"] == "unbounded_memory_reference"
    )
    assert reference["maximum_memory_bytes"] > bounded["maximum_memory_bytes"]
    assert reference["larger_memory_reference"] is True


def test_scenario_cl_7296_rows_cold_reduce_all_and_nonfeedback_subsets(
    measured_stream: tuple[exp.ExperimentPaths, exp.EvaluationPanel],
) -> None:
    """SCENARIO-CL-7296-ROWS reconstructs outcomes and charges from disk."""

    paths, panel = measured_stream
    reduced = exp.independent_reduce(paths.step_rows, paths.feedback_rows)
    assert reduced == panel.per_stream_results
    assert len(reduced) == len(exp.ARMS)
    assert all(row["future_prediction_count"] == 896 for row in reduced)
    assert all(row["non_feedback_future_prediction_count"] == 768 for row in reduced)
    assert all(row["warmup_label_count"] == 128 for row in reduced)
    assert all(row["future_label_count"] == 128 for row in reduced)
    assert all(row["prediction_cost_ns"] > 0 for row in reduced)
    assert all(row["censored"] is False for row in reduced)


def _favorable_rows() -> list[dict[str, object]]:
    """Create 24 independent stream summaries with every frozen gate favorable."""

    rows: list[dict[str, object]] = []
    for offset in range(exp.EVALUATION_STREAM_COUNT):
        stream_id = f"evaluation-{offset + 1:02d}"
        stratum = "separated_recurrence" if offset < 12 else "overlapping_recurrence"
        for arm in exp.ARMS:
            error = 0.10 if arm == "fixed_share_mixture" else 0.20
            if arm == "label_shuffled_fixed_share":
                error = 0.18
            rows.append(
                {
                    "stream_id": stream_id,
                    "seed": exp.EVALUATION_STREAM_SEEDS[offset],
                    "stratum": stratum,
                    "arm": arm,
                    "future_error_rate": error,
                    "non_feedback_future_error_rate": error,
                    "recurrence_error_rate": error,
                    "false_accept_rate": 0.01 if arm == "fixed_share_mixture" else 0.03,
                    "coverage": 0.99 if arm == "fixed_share_mixture" else 0.98,
                }
            )
    return rows


def test_scenario_cl_7296_gates_use_paired_stream_bootstrap() -> None:
    """SCENARIO-CL-7296-GATES keeps whole streams and both strata separate."""

    comparisons = exp.build_comparison_rows(_favorable_rows())
    assert comparisons
    assert {row["stratum"] for row in comparisons} == {
        "overall",
        "separated_recurrence",
        "overlapping_recurrence",
    }
    assert all(row["bootstrap_resamples"] == 10_000 for row in comparisons)
    assert all(row["independent_unit"] == "stream" for row in comparisons)
    causal = {
        "later_changed_prediction_count": 24,
        "chronology_violation_count": 0,
        "bounded_memory_violation_count": 0,
    }
    gates = exp.score_acceptance_gates(comparisons, causal)
    assert all(row["passed"] is True for row in gates.values())
    assert exp.classify_result(gates) == (
        1,
        "circular_positive",
        "complete_circular_positive: prospective fixed-share learning passed every frozen gate under exact evaluator authority",
    )
    failed = deepcopy(gates)
    failed["true_feedback_future_error_vs_shuffled"]["passed"] = False
    failed["true_feedback_future_error_vs_shuffled"]["pass"] = False
    assert exp.classify_result(failed)[0:2] == (0, "null")


def test_scenario_cl_7296_e2e_restart_preserves_later_prediction(tmp_path: Path) -> None:
    """SCENARIO-CL-7296-E2E exercises prediction, update, restart, and reuse."""

    rows = exp.run_e2e_controls(tmp_path)
    assert {row["stage"] for row in rows} == {
        "observation",
        "pre_label_prediction",
        "delayed_feedback",
        "bounded_online_update",
        "later_unrevealed_prediction",
        "checkpoint_restart_parity",
    }
    assert all(row["passed"] is True for row in rows)


def test_scenario_cl_7296_terminal_block_and_measured_validation(
    measured_stream: tuple[exp.ExperimentPaths, exp.EvaluationPanel],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7296-TERMINAL separates external blocks from measured nulls."""

    paths, _ = measured_stream
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        stream_ids=("evaluation-01", "evaluation-13"),
        progress=True,
    )
    assert artifact["status"] == "complete"
    assert artifact["mixture_capture_complete_score"] == 1
    assert artifact["verdict_class"] in {"null", "circular_positive"}
    assert artifact["rows"] == artifact["per_stream_results"]
    assert (
        exp.validate_artifact(
            artifact,
            expected_stream_ids=("evaluation-01", "evaluation-13"),
            check_files=True,
        )
        == []
    )
    output = tmp_path / "terminal.json"
    exp.write_artifact(
        output,
        artifact,
        expected_stream_ids=("evaluation-01", "evaluation-13"),
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    failed = exp.fixture.gate_check("missing", "upstream", "status", "complete", None)
    blocked_paths = exp.ExperimentPaths.under(tmp_path / "blocked")
    monkeypatch.setattr(exp, "collect_preconditions", lambda *args: ([failed], {}, {}))
    blocked = exp.build_and_seal(exp.REPO_ROOT, blocked_paths, progress=True)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["gate_check_summary"]["first_failure"]["field"] == "status"
    assert exp.validate_artifact(blocked) == []


def test_req_cl_7296_thin_entrypoint_and_cli_helpers(
    measured_stream: tuple[exp.ExperimentPaths, exp.EvaluationPanel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7296 keeps orchestration reusable and the script entrypoint thin."""

    paths, _ = measured_stream
    args = exp._parse_args(["--date", "20260914"])
    assert args.date == "20260914"
    commands = exp._validation_commands(paths.terminal_candidate)
    flattened = [part for command in commands for part in command]
    assert "scripts/check_spec_coverage.py" in flattened
    assert "scripts/adversarial_verify.py" in flattened
    assert "scripts/verdict_row_consistency_lint.py" in flattened
    wrapper = exp.REPO_ROOT / exp.WRAPPER_PATH
    monkeypatch.setattr(exp, "main", lambda: 0)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert result.value.code == 0


def test_req_cl_7296_input_and_replay_defenses(
    measured_stream: tuple[exp.ExperimentPaths, exp.EvaluationPanel],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7296 fails closed on malformed authority, rows, and checkpoints."""

    assert exp.ExperimentPaths.defaults().artifact == exp.REPO_ROOT / exp.DEFAULT_ARTIFACT
    assert exp._task_identity("[") == {}
    assert exp._task_identity("{}") == {}
    assert exp._task_identity("tasks: []") == {}
    assert exp._receipt(tmp_path / "absent")["sha256"] is None
    bad_repo = tmp_path / "bad-repo"
    (bad_repo / "ops").mkdir(parents=True)
    (bad_repo / "ops/exclusion_manifest.yaml").write_text("[", encoding="utf-8")
    bad_checks, _, _ = exp.collect_preconditions(
        bad_repo,
        exp.ExperimentPaths.under(tmp_path / "bad-repo-output"),
    )
    assert bad_checks

    upstream = exp._load_object(exp.REPO_ROOT / exp.UPSTREAM_ARTIFACT)
    manifest = exp._load_object(Path(upstream["stream_manifest_path"]))
    receipt = upstream["raw_evidence_receipts"]["stream_manifest"]

    with monkeypatch.context() as patch:
        patch.setattr(exp, "_sha256_path", lambda path: "bad")
        with pytest.raises(ValueError, match="stream_manifest_hash"):
            exp.load_authenticated_views(exp.REPO_ROOT, upstream)

    mutations = (
        ({}, "evaluation_manifest"),
        ({"evaluation": {}}, "evaluation_receipts"),
        ({"evaluation": {"receipts": {}}}, "missing_view:public"),
    )
    for changed_manifest, expected in mutations:
        with monkeypatch.context() as patch:
            patch.setattr(exp, "_load_object", lambda path, value=changed_manifest: value)
            patch.setattr(exp, "_sha256_path", lambda path: receipt["sha256"])
            with pytest.raises(ValueError, match=expected):
                exp.load_authenticated_views(exp.REPO_ROOT, upstream)

    with monkeypatch.context() as patch:
        patch.setattr(exp, "_load_object", lambda path: manifest)
        patch.setattr(
            exp,
            "_sha256_path",
            lambda path: (
                receipt["sha256"] if Path(path) == Path(upstream["stream_manifest_path"]) else "bad"
            ),
        )
        with pytest.raises(ValueError, match="view_hash:public"):
            exp.load_authenticated_views(exp.REPO_ROOT, upstream)

    with monkeypatch.context() as patch:
        patch.setattr(exp, "_load_object", lambda path: manifest)
        patch.setattr(
            exp,
            "_sha256_path",
            lambda path: (
                receipt["sha256"]
                if Path(path) == Path(upstream["stream_manifest_path"])
                else manifest["evaluation"]["receipts"]["public"]["sha256"]
            ),
        )
        patch.setattr(exp, "_read_jsonl", lambda path: [{}])
        with pytest.raises(ValueError, match="view_row_count:public"):
            exp.load_authenticated_views(exp.REPO_ROOT, upstream)

    valid_views = exp.load_authenticated_views(exp.REPO_ROOT, upstream)
    with monkeypatch.context() as patch:
        patch.setattr(exp.fixture, "stream_conformance_errors", lambda *args: ["forced"])
        with pytest.raises(ValueError, match="stream_conformance:forced"):
            exp.load_authenticated_views(exp.REPO_ROOT, upstream)
    with pytest.raises(ValueError, match="incomplete_stream"):
        exp._stream_rows(valid_views, "missing")

    assert exp.step_row_errors([], ("evaluation-01",))
    assert exp.feedback_row_errors([], ("evaluation-01",))
    bad_step = tmp_path / "bad-step.jsonl"
    bad_feedback = tmp_path / "bad-feedback.jsonl"
    bad_step.write_text("{}\n", encoding="utf-8")
    bad_feedback.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_row_conformance"):
        exp.independent_reduce(bad_step, bad_feedback)
    with pytest.raises(ValueError, match="paired_streams_unavailable"):
        exp._bootstrap_interval([], 10, "empty")
    assert exp.build_comparison_rows([]) == []

    paths, panel = measured_stream
    censored = exp.run_learning_panel(
        valid_views,
        exp.ExperimentPaths.under(tmp_path / "censored"),
        stream_ids=("evaluation-01",),
        measurement_limit_s=-1.0,
    )
    assert censored.censored_stream_ids == ["evaluation-01"]
    invalid_paths = exp.ExperimentPaths.under(tmp_path / "invalid-checkpoint")
    exp._atomic_write(
        invalid_paths.checkpoint_dir / "evaluation-01.json",
        exp._canonical_bytes(
            {
                "schema": exp.SCHEMA,
                "status": "complete",
                "step_rows": [],
                "feedback_update_rows": [],
            }
        ),
    )
    with pytest.raises(ValueError, match="stream_conformance"):
        exp.run_learning_panel(valid_views, invalid_paths, stream_ids=("evaluation-01",))

    clock = iter((0.0, 0.0, 61.0))
    with monkeypatch.context() as patch:
        patch.setattr(exp.time, "monotonic", lambda: next(clock))
        replayed = exp.run_learning_panel(
            valid_views,
            paths,
            stream_ids=("evaluation-01",),
        )
    assert replayed.per_stream_results == panel.per_stream_results


def test_req_cl_7296_artifact_and_cli_defenses(
    measured_stream: tuple[exp.ExperimentPaths, exp.EvaluationPanel],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7296 rejects invalid publication and records CLI subprocess outcomes."""

    paths, panel = measured_stream
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        stream_ids=("evaluation-01", "evaluation-13"),
        progress=False,
    )
    exp._atomic_write(paths.terminal_candidate, exp._canonical_bytes(artifact))
    receipt = {
        "command": "extra-check",
        "exit_code": 0,
        "duration_s": 0.1,
        "log_sha256": "sha256:" + "1" * 64,
        "classification": "passed",
    }
    attached = exp.attach_validation_receipts(artifact, [receipt])
    assert attached["validation_receipts"][-1] == receipt
    assert (
        exp.validate_artifact(
            attached,
            expected_stream_ids=("evaluation-01", "evaluation-13"),
            check_files=True,
        )
        == []
    )

    invalid = deepcopy(artifact)
    invalid["schema"] = "bad"
    invalid["reproducibility_checksum"] = exp.reproducibility_checksum(invalid)
    assert "identity" in exp.validate_artifact(invalid)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(
            tmp_path / "bad.json",
            invalid,
            expected_stream_ids=("evaluation-01", "evaluation-13"),
        )
    assert "status" in exp.validate_artifact({"status": "other"})

    missing_raw = deepcopy(artifact)
    missing_raw["raw_evidence_receipts"]["step_rows"]["path"] = str(tmp_path / "missing")
    missing_raw["reproducibility_checksum"] = exp.reproducibility_checksum(missing_raw)
    assert "cold_reducer" in exp.validate_artifact(
        missing_raw,
        expected_stream_ids=("evaluation-01", "evaluation-13"),
        check_files=True,
    )

    censored_panel = exp.EvaluationPanel([], [], [], [], ["evaluation-01"], 0, 0, 0)
    with monkeypatch.context() as patch:
        patch.setattr(exp, "run_learning_panel", lambda *args, **kwargs: censored_panel)
        with pytest.raises(RuntimeError, match="measurement_limit_censored"):
            exp.build_and_seal(
                exp.REPO_ROOT,
                exp.ExperimentPaths.under(tmp_path / "build-censored"),
                stream_ids=("evaluation-01",),
                progress=False,
            )

    with monkeypatch.context() as patch:
        patch.setattr(exp, "run_learning_panel", lambda *args, **kwargs: panel)
        patch.setattr(exp, "independent_reduce", lambda *args: [])
        with pytest.raises(ValueError, match="independent_reducer_mismatch"):
            exp.build_and_seal(
                exp.REPO_ROOT,
                exp.ExperimentPaths.under(tmp_path / "build-reducer"),
                stream_ids=("evaluation-01",),
                progress=False,
            )

    with monkeypatch.context() as patch:
        patch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
        with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
            exp.build_and_seal(
                exp.REPO_ROOT,
                paths,
                stream_ids=("evaluation-01", "evaluation-13"),
                progress=False,
            )

    assert exp.main(["--date", "20260914", "--validate-raw", str(paths.terminal_candidate)]) == 1

    failed_check = exp.fixture.gate_check("missing", "upstream", "status", "complete", None)
    blocked = exp.build_blocked_artifact(
        [failed_check],
        {},
        ("evaluation-01",),
        started_at="2026-09-14T00:00:00+00:00",
        duration_s=1.0,
    )
    cli_paths = exp.ExperimentPaths.under(tmp_path / "cli")
    with monkeypatch.context() as patch:
        patch.setattr(exp.ExperimentPaths, "defaults", classmethod(lambda cls: cli_paths))
        patch.setattr(exp, "build_and_seal", lambda *args, **kwargs: deepcopy(blocked))
        assert exp.main(["--date", "20260914"]) == 0
    assert cli_paths.artifact.exists()

    def fake_receipt(command: list[str]) -> tuple[dict[str, object], str]:
        return ({**receipt, "command": " ".join(command)}, "ok\n")

    with monkeypatch.context() as patch:
        patch.setattr(exp.ExperimentPaths, "defaults", classmethod(lambda cls: cli_paths))
        patch.setattr(exp, "build_and_seal", lambda *args, **kwargs: deepcopy(artifact))
        patch.setattr(exp, "write_artifact", lambda *args, **kwargs: {})
        patch.setattr(exp, "_validation_commands", lambda candidate: [[sys.executable, "-V"]])
        patch.setattr(exp.fixture, "_command_receipt", fake_receipt)
        assert exp.main(["--date", "20260914"]) == 0

    def failed_receipt(command: list[str]) -> tuple[dict[str, object], str]:
        return ({**receipt, "command": " ".join(command), "exit_code": 1}, "failed\n")

    with monkeypatch.context() as patch:
        patch.setattr(exp.ExperimentPaths, "defaults", classmethod(lambda cls: cli_paths))
        patch.setattr(exp, "build_and_seal", lambda *args, **kwargs: deepcopy(artifact))
        patch.setattr(exp, "write_artifact", lambda *args, **kwargs: {})
        patch.setattr(exp, "_validation_commands", lambda candidate: [[sys.executable, "-V"]])
        patch.setattr(exp.fixture, "_command_receipt", failed_receipt)
        assert exp.main(["--date", "20260914"]) == 1

    with monkeypatch.context() as patch:
        patch.setattr(exp.ExperimentPaths, "defaults", classmethod(lambda cls: cli_paths))
        patch.setattr(exp, "build_and_seal", lambda *args, **kwargs: deepcopy(artifact))
        patch.setattr(exp, "write_artifact", lambda *args, **kwargs: {})
        patch.setattr(
            exp, "_validation_commands", lambda candidate: [[sys.executable, "tests/python"]]
        )
        patch.setattr(exp.fixture, "_command_receipt", failed_receipt)
        assert exp.main(["--date", "20260914"]) == 0
