"""Verify the recognition audit required by REQ-CL-7269."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot import experiment_7268_v639_recognition_learning as learning
from carnot import experiment_7269_v639_recognition_audit as audit


def test_preconditions_admit_complete_null_upstream(tmp_path: Path) -> None:
    """SCENARIO-CL-7269-PRECONDITIONS: completion, not value, opens the audit."""

    paths = audit.ExperimentPaths.under(tmp_path)
    checks, hashes, upstream = audit.collect_preconditions(audit.REPO_ROOT, paths)

    assert audit.gate_summary(checks)["passed"] is True
    assert upstream["recognition_run_complete_score"] == 1
    assert upstream["recognition_value_score"] == 0
    assert hashes[str(audit.REPO_ROOT / audit.DEFAULT_UPSTREAM_ARTIFACT)] == (
        audit.EXPECTED_UPSTREAM_SHA256
    )
    assert all("recognition_value_score" not in row["check"] for row in checks)


def test_absent_external_input_is_terminal_blocked(tmp_path: Path) -> None:
    """SCENARIO-CL-7269-PRECONDITIONS: absence is blocked and row-free."""

    paths = audit.ExperimentPaths.under(tmp_path / "out")
    checks, hashes, upstream = audit.collect_preconditions(
        audit.REPO_ROOT,
        paths,
        upstream_path=tmp_path / "missing.json",
    )
    artifact = audit.build_blocked_artifact(
        checks,
        hashes,
        upstream,
        ("prospective-01",),
        started_at="2026-09-13T00:00:00+00:00",
        duration_s=0.01,
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["recognition_audit_complete_score"] == 0
    assert artifact["recognition_promotion_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"]["observed_value"] is None
    assert audit.validate_artifact(artifact, expected_stream_ids=("prospective-01",)) == []


def test_cold_reducer_reconstructs_one_full_eight_arm_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7269-REDUCTION: raw rows reconstruct metrics and gates."""

    monkeypatch.setenv("CARNOT_EXP7269_COLD_WORKER", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    result = audit.audit_raw_evidence(
        audit.REPO_ROOT,
        stream_ids=("prospective-01",),
        bootstrap_draws=50,
        progress=True,
    )
    upstream = json.loads(
        (audit.REPO_ROOT / audit.DEFAULT_UPSTREAM_ARTIFACT).read_text(encoding="utf-8")
    )
    expected = [row for row in upstream["rows"] if row["stream_id"] == "prospective-01"]

    assert result["rows"] == expected
    assert len(result["rows"]) == len(learning.ARMS)
    assert len(result["comparison_rows"]) == len(learning.COMPARISON_SPECS) * 3
    assert result["causal_summary"]["pre_release_difference_count"] == 0
    assert all(row["passed"] is True for row in result["raw_check_rows"])
    assert all(row["passed"] is True for row in result["replay_rows"])
    assert result["process_receipt"]["fresh_process"] is True
    assert result["overlap_rows"] == []


def test_missing_raw_field_fails_instead_of_becoming_zero(tmp_path: Path) -> None:
    """SCENARIO-CL-7269-REDUCTION: a missing source field is not a zero effect."""

    source = audit._selected_raw_rows(
        audit.REPO_ROOT / audit.DEFAULT_RAW_ROWS,
        ("prospective-01",),
    )[:1]
    source[0].pop("false_accept")
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(json.dumps(source[0]) + "\n", encoding="utf-8")

    with pytest.raises(audit.AuditEvidenceError, match="missing_raw_fields:false_accept"):
        audit._selected_raw_rows(malformed, ("prospective-01",))


def test_intervention_join_requires_release_before_later_change() -> None:
    """SCENARIO-CL-7269-REPLAY: changed queries join only to later predictions."""

    rows = []
    for arm, query_index, prediction in (
        ("active_recognition", 1, "accept"),
        ("random_query_recognition", 2, "reject"),
        ("shuffled_archive_association", 1, "reject"),
    ):
        for index in range(4):
            rows.append(
                {
                    "stream_id": "prospective-01",
                    "stratum": "overlapping_recurrence",
                    "arm": arm,
                    "chronology_index": index,
                    "event_id": f"e{index}",
                    "query_selected": index == query_index,
                    "query_release_index": 2 if index == query_index else None,
                    "prediction": prediction if index == 3 else "accept",
                    "selection_change_count": int(index == 3 and arm != "random_query_recognition"),
                }
            )

    interventions = audit.build_intervention_rows(rows, block_size=4)

    active_random = next(row for row in interventions if row["comparison"] == "active_vs_random")
    active_shuffle = next(row for row in interventions if row["comparison"] == "active_vs_shuffled")
    assert active_random["changed_nomination_identity"] is True
    assert active_random["later_changed_prediction_count"] == 1
    assert active_random["release_precedes_effect"] is True
    assert active_shuffle["changed_association_count"] == 1


def test_bounds_keep_overlap_recall_false_accepts_and_zero_effects() -> None:
    """SCENARIO-CL-7269-BOUNDS: overlap and zero effects remain explicit."""

    rows = [
        {
            "unit_id": "prospective-13:active_recognition",
            "stream_id": "prospective-13",
            "arm": "active_recognition",
            "stratum": "overlapping_recurrence",
            "query_count": 0,
            "maximum_memory_bytes": 0,
            "retained_constraint_count": 0,
            "false_accept": 0,
            "false_accept_rate": 0.0,
            "recurrence_event_count": 4,
            "recurrence_error": 1,
            "recurrence_error_rate": 0.25,
        }
    ]

    bounds, overlap = audit.build_bound_rows(rows)

    assert bounds[0]["label_budget_used"] == 0
    assert bounds[0]["memory_bytes"] == 0
    assert bounds[0]["retained_constraint_count"] == 0
    assert bounds[0]["passed"] is True
    assert overlap[0]["false_accept_count"] == 0
    assert overlap[0]["recurrence_recall"] == 0.75


def test_five_mutations_reject_and_preserve_bytes(tmp_path: Path) -> None:
    """SCENARIO-CL-7269-MUTATIONS: each invalid update preserves parent bytes."""

    rows = audit.run_mutation_controls(tmp_path)

    assert {row["mutation"] for row in rows} == {
        "future_label",
        "regime_id",
        "duplicate_release",
        "stale_parent",
        "corrupted_archive",
    }
    assert all(row["rejected"] is True for row in rows)
    assert all(row["prior_bytes_preserved"] is True for row in rows)
    assert all(row["passed"] is True for row in rows)


def test_e2e_accept_reject_reload_and_rollback(tmp_path: Path) -> None:
    """SCENARIO-CL-7269-E2E: changed accepted state reloads and rolls back."""

    rows, rollback_rows = audit.run_e2e_controls(tmp_path)

    assert {row["control"] for row in rows} == {
        "prediction_before_release",
        "accepted_commit_changes_state",
        "invalid_commit_preserves_state",
        "cold_reload_parity",
        "later_prediction_observes_commit",
        "rollback_restores_parent",
    }
    assert all(row["passed"] is True for row in rows)
    assert rollback_rows == [
        {
            "control": "rollback_restores_parent",
            "parent_bytes_preserved": True,
            "cold_restore_exercised": True,
            "passed": True,
        }
    ]


def test_complete_null_audit_does_not_promote() -> None:
    """SCENARIO-CL-7269-TERMINAL: failed science does not erase audit completion."""

    upstream = json.loads(
        (audit.REPO_ROOT / audit.DEFAULT_UPSTREAM_ARTIFACT).read_text(encoding="utf-8")
    )
    gates = learning.score_acceptance_gates(
        learning.build_comparison_rows(upstream["rows"], draws=50),
        upstream["causal_summary"],
    )

    assert audit.derive_terminal_scores(True, True, gates) == (
        1,
        0,
        "null",
        "complete_null: recognition audit completed but promotion gates did not all pass",
    )
    passing = {name: {"passed": True} for name in learning.SCIENTIFIC_GATE_NAMES}
    assert audit.derive_terminal_scores(True, True, passing)[1:3] == (1, "circular_positive")


def test_one_stream_build_is_valid_and_atomic(tmp_path: Path) -> None:
    """REQ-CL-7269: cold review seals a valid terminal result after measurement."""

    paths = audit.ExperimentPaths.under(tmp_path)
    artifact = audit.build_and_seal(
        audit.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        bootstrap_draws=50,
        progress=True,
    )

    assert artifact["status"] == "complete"
    assert artifact["recognition_audit_complete_score"] == 1
    assert artifact["recognition_promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert len(artifact["rows"]) == len(learning.ARMS)
    assert len(artifact["mutation_rows"]) == 5
    assert paths.checkpoint.is_file()
    assert paths.raw_summary.is_file()
    assert paths.mutation_sidecar.is_file()
    assert paths.e2e_sidecar.is_file()
    assert not paths.artifact.exists()
    assert (
        audit.validate_artifact(
            artifact,
            repo_root=audit.REPO_ROOT,
            expected_stream_ids=("prospective-01",),
            check_files=True,
        )
        == []
    )

    receipt = audit.write_artifact(
        paths.artifact,
        artifact,
        repo_root=audit.REPO_ROOT,
        expected_stream_ids=("prospective-01",),
    )
    assert receipt["sha256"] == audit._sha256_path(paths.artifact)
    assert json.loads(paths.artifact.read_text(encoding="utf-8"))["experiment_id"] == 7269


def test_validator_and_receipts_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-7269-TERMINAL: false promotion and false receipts are rejected."""

    paths = audit.ExperimentPaths.under(tmp_path)
    checks, hashes, upstream = audit.collect_preconditions(audit.REPO_ROOT, paths)
    blocked = audit.build_blocked_artifact(
        checks,
        hashes,
        upstream,
        ("prospective-01",),
        duration_s=0.01,
    )
    changed = deepcopy(blocked)
    changed["verdict_class"] = "positive"
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    assert "blocked_contract" in audit.validate_artifact(
        changed, expected_stream_ids=("prospective-01",)
    )
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        audit.attach_validation_receipts(blocked, [{"command": "pytest"}])


def test_thin_wrapper_and_scoped_validation_commands() -> None:
    """REQ-CL-7269: the entrypoint stays thin and validation stays scoped."""

    source = (audit.REPO_ROOT / audit.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "from carnot.experiment_7269_v639_recognition_audit import main" in source
    assert len(source.splitlines()) < 20
    commands = [" ".join(command) for command in audit._validation_commands(Path("candidate"))]
    assert any("test_experiment_7269_v639_recognition_audit.py" in command for command in commands)
    assert any(
        "test_experiment_7267_v639_recognition_prototype.py" in command for command in commands
    )
    assert any(
        "test_experiment_7268_v639_recognition_learning.py" in command for command in commands
    )
    assert not any(command.rstrip().endswith("tests/python -q") for command in commands)


def test_cli_dispatch_and_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CL-7269: worker, validation, blocked, and failure paths stay explicit."""

    assert audit.ExperimentPaths.defaults().artifact == audit.REPO_ROOT / audit.DEFAULT_ARTIFACT
    with pytest.raises(SystemExit, match="run_date_must_be"):
        audit.main(["--date", "19000101"])

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{\n", encoding="utf-8")
    with pytest.raises(audit.AuditEvidenceError, match="invalid_jsonl"):
        audit._selected_raw_rows(invalid, ("prospective-01",))
    non_object = tmp_path / "non-object.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(audit.AuditEvidenceError, match="non_object_jsonl"):
        audit._selected_raw_rows(non_object, ("prospective-01",))

    paths = audit.ExperimentPaths.under(tmp_path / "blocked")
    blocked = audit.build_and_seal(
        tmp_path / "absent-repository",
        paths,
        stream_ids=("prospective-01",),
        bootstrap_draws=10,
        progress=True,
    )
    assert blocked["status"] == "blocked"

    receipt = {
        "command": "focused check",
        "exit_code": 0,
        "classification": "passed",
        "log_sha256": "sha256:" + "1" * 64,
    }
    assert audit.attach_validation_receipts(blocked, [receipt])["validation_receipts"] == [receipt]

    monkeypatch.setattr(audit, "audit_raw_evidence", lambda *args, **kwargs: {"worker": True})
    assert (
        audit.main(
            [
                "--date",
                audit.RUN_DATE,
                "--audit-worker",
                "--stream-ids",
                "prospective-01",
                "--bootstrap-draws",
                "10",
            ]
        )
        == 0
    )
    assert audit.RESULT_PREFIX in capsys.readouterr().out

    monkeypatch.setattr(audit.ExperimentPaths, "under", classmethod(lambda cls, root: paths))
    paths.artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.artifact.write_text(json.dumps(blocked), encoding="utf-8")
    assert audit.main(["--date", audit.RUN_DATE, "--output-root", str(tmp_path), "--validate"]) == 0

    monkeypatch.setattr(audit, "build_and_seal", lambda *args, **kwargs: blocked)
    monkeypatch.setattr(
        audit,
        "write_artifact",
        lambda *args, **kwargs: {"sha256": "sha256:" + "2" * 64, "bytes": 10},
    )
    assert audit.main(["--date", audit.RUN_DATE, "--output-root", str(tmp_path)]) == 0


def test_raw_reader_and_precondition_defenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7269-PRECONDITIONS: malformed evidence cannot become zero."""

    with pytest.raises(audit.AuditEvidenceError, match="raw_rows_unavailable"):
        audit._selected_raw_rows(tmp_path / "missing.jsonl", ("prospective-01",))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(audit.AuditEvidenceError, match="raw_rows_unavailable"):
        audit._selected_raw_rows(empty, ("prospective-01",))

    monkeypatch.setattr(
        learning,
        "reproducibility_checksum",
        lambda artifact: (_ for _ in ()).throw(ValueError("bad producer")),
    )
    checks, _, _ = audit.collect_preconditions(
        audit.REPO_ROOT, audit.ExperimentPaths.under(tmp_path / "outputs")
    )
    checksum = next(row for row in checks if row["check"] == "exp7268_checksum")
    assert checksum["observed_value"] is False

    monkeypatch.setattr(
        prototype := audit.prototype, "stream_conformance_errors", lambda *args: ["bad"]
    )
    with pytest.raises(audit.AuditEvidenceError, match="sealed_stream_conformance:bad"):
        audit._load_views(audit.REPO_ROOT)
    assert prototype is audit.prototype


def test_cold_reducer_defensive_mismatch_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7269-REDUCTION: incomplete and mismatched replay fails closed."""

    monkeypatch.setenv("CARNOT_EXP7269_COLD_WORKER", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        audit,
        "_selected_raw_rows",
        lambda *args: [{"stream_id": "prospective-02"}],
    )
    with pytest.raises(audit.AuditEvidenceError, match="selected_streams_incomplete"):
        audit._audit_raw_evidence_impl(
            audit.REPO_ROOT,
            stream_ids=("prospective-01",),
            bootstrap_draws=5,
            progress=False,
        )

    monkeypatch.undo()
    rows = audit._selected_raw_rows(audit.REPO_ROOT / audit.DEFAULT_RAW_ROWS, ("prospective-01",))
    monkeypatch.setenv("CARNOT_EXP7269_COLD_WORKER", "1")
    monkeypatch.setattr(audit, "_selected_raw_rows", lambda *args: rows)
    monkeypatch.setattr(
        audit,
        "_raw_check_rows",
        lambda *args: [{"check": "forced", "passed": False}],
    )
    with pytest.raises(audit.AuditEvidenceError, match="raw_check_failed"):
        audit._audit_raw_evidence_impl(
            audit.REPO_ROOT,
            stream_ids=("prospective-01",),
            bootstrap_draws=5,
            progress=False,
        )

    monkeypatch.setattr(
        audit,
        "_raw_check_rows",
        lambda *args: [{"check": "forced", "passed": True}],
    )
    monkeypatch.setattr(audit, "_reduce_rows", lambda *args: [])
    with pytest.raises(audit.AuditEvidenceError, match="stream_arm_matrix"):
        audit._audit_raw_evidence_impl(
            audit.REPO_ROOT,
            stream_ids=("prospective-01",),
            bootstrap_draws=5,
            progress=False,
        )


def test_cold_reducer_rejects_replay_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-CL-7269-REPLAY: a replay mismatch is terminal audit evidence."""

    rows = audit._selected_raw_rows(audit.REPO_ROOT / audit.DEFAULT_RAW_ROWS, ("prospective-01",))
    monkeypatch.setenv("CARNOT_EXP7269_COLD_WORKER", "1")
    monkeypatch.setattr(audit, "_selected_raw_rows", lambda *args: rows)
    monkeypatch.setattr(
        audit,
        "_replay_stream",
        lambda *args: {"stream_id": "prospective-01", "passed": False},
    )
    with pytest.raises(audit.AuditEvidenceError, match="cold_reconstruction_mismatch"):
        audit._audit_raw_evidence_impl(
            audit.REPO_ROOT,
            stream_ids=("prospective-01",),
            bootstrap_draws=5,
            progress=False,
        )


def test_public_cold_api_and_partial_score_defenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7269-TERMINAL: only a real cold result can complete."""

    monkeypatch.delenv("CARNOT_EXP7269_COLD_WORKER", raising=False)
    with pytest.raises(audit.AuditEvidenceError, match="cold_worker_requires_repository_root"):
        audit.audit_raw_evidence(tmp_path, stream_ids=("prospective-01",))
    monkeypatch.setattr(audit, "_spawn_audit_worker", lambda *args: {"cold": True})
    assert audit.audit_raw_evidence(
        audit.REPO_ROOT, stream_ids=("prospective-01",), bootstrap_draws=5
    ) == {"cold": True}
    assert audit.derive_terminal_scores(False, True, {}) == (
        0,
        0,
        "partial",
        "partial: task-owned audit work is incomplete",
    )


def test_build_and_write_reject_incomplete_or_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7269-TERMINAL: own incomplete work stays checkpointed."""

    incomplete_worker = {
        "rows": [],
        "science_gates": {},
        "intervention_rows": [],
        "overlap_rows": [],
        "bound_rows": [],
        "raw_check_rows": [],
        "replay_rows": [],
        "comparison_rows": [],
        "causal_summary": {},
        "process_receipt": {
            "fresh_process": True,
            "gpu_disabled": True,
            "network_cache_offline": True,
            "no_model_load": True,
        },
    }
    monkeypatch.setattr(audit, "_spawn_audit_worker", lambda *args: incomplete_worker)
    with pytest.raises(audit.AuditEvidenceError, match="task_owned_audit_incomplete"):
        audit.build_and_seal(
            audit.REPO_ROOT,
            audit.ExperimentPaths.under(tmp_path / "incomplete"),
            stream_ids=("prospective-01",),
        )
    with pytest.raises(audit.AuditEvidenceError, match="artifact_validation_failed"):
        audit.write_artifact(tmp_path / "bad.json", {})


def test_build_rejects_failed_final_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7269-TERMINAL: final cold validation controls publication."""

    upstream = json.loads(
        (audit.REPO_ROOT / audit.DEFAULT_UPSTREAM_ARTIFACT).read_text(encoding="utf-8")
    )
    rows = [row for row in upstream["rows"] if row["stream_id"] == "prospective-01"]
    causal = {
        "prospective_selection_change_count": 1,
        "later_changed_prediction_count": 1,
        "pre_release_difference_count": 0,
        "cap_violation_count": 0,
        "constraint_addition_count": 1,
        "constraint_deactivation_count": 1,
        "reactivation_count": 1,
    }
    comparisons = learning.build_comparison_rows(rows, draws=5)
    bounds, overlap = audit.build_bound_rows(rows)
    worker = {
        "rows": rows,
        "science_gates": learning.score_acceptance_gates(comparisons, causal),
        "intervention_rows": [
            {
                "changed_nomination_identity": True,
                "later_changed_prediction_count": 1,
                "release_precedes_effect": True,
            }
        ],
        "overlap_rows": overlap,
        "bound_rows": bounds,
        "raw_check_rows": [{"passed": True}],
        "replay_rows": [{"passed": True}],
        "comparison_rows": comparisons,
        "causal_summary": causal,
        "process_receipt": {
            "fresh_process": True,
            "gpu_disabled": True,
            "network_cache_offline": True,
            "no_model_load": True,
        },
    }
    monkeypatch.setattr(audit, "_spawn_audit_worker", lambda *args: worker)
    monkeypatch.setattr(audit, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(audit.AuditEvidenceError, match="artifact_validation_failed:forced"):
        audit.build_and_seal(
            audit.REPO_ROOT,
            audit.ExperimentPaths.under(tmp_path),
            stream_ids=("prospective-01",),
            bootstrap_draws=5,
        )


def test_subprocess_receipt_and_main_complete_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7269: validation logs and complete publication paths are exercised."""

    receipt = audit._command_receipt([sys.executable, "-c", "print('focused-ok')"])
    assert receipt["exit_code"] == 0
    assert receipt["classification"] == "passed"
    assert receipt["log_sha256"].startswith("sha256:")

    paths = audit.ExperimentPaths.under(tmp_path / "complete")
    minimal = {"status": "complete", "validation_receipts": []}
    monkeypatch.setattr(audit.ExperimentPaths, "under", classmethod(lambda cls, root: paths))
    monkeypatch.setattr(audit, "build_and_seal", lambda *args, **kwargs: minimal)
    monkeypatch.setattr(audit, "_validation_commands", lambda candidate: [["fake"]])
    monkeypatch.setattr(
        audit,
        "_command_receipt",
        lambda command: {
            "command": " ".join(command),
            "exit_code": 1,
            "classification": "failed",
            "log_sha256": "sha256:" + "3" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="focused_validation_failed"):
        audit.main(["--date", audit.RUN_DATE, "--output-root", str(tmp_path)])

    monkeypatch.setattr(audit, "_validation_commands", lambda candidate: [])
    monkeypatch.setattr(
        audit,
        "write_artifact",
        lambda *args, **kwargs: {"sha256": "sha256:" + "4" * 64, "bytes": 1},
    )
    assert audit.main(["--date", audit.RUN_DATE, "--output-root", str(tmp_path)]) == 0

    e2e_paths = audit.ExperimentPaths.under(tmp_path / "e2e")
    monkeypatch.setattr(audit.ExperimentPaths, "under", classmethod(lambda cls, root: e2e_paths))
    assert (
        audit.main(
            [
                "--date",
                audit.RUN_DATE,
                "--output-root",
                str(tmp_path),
                "--e2e-worker",
            ]
        )
        == 0
    )

    invalid = tmp_path / "invalid-artifact.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(audit.AuditEvidenceError, match="artifact_validation_failed"):
        audit.main(
            [
                "--date",
                audit.RUN_DATE,
                "--validate",
                "--artifact-path",
                str(invalid),
            ]
        )


def test_streaming_subprocess_defensive_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CL-7269: timeout, heartbeat, drain, and malformed results stay visible."""

    class FakeStdout:
        def __init__(self, line: str = "", trailing: tuple[str, ...] = ()) -> None:
            self.line = line
            self.trailing = trailing

        def readline(self) -> str:
            return self.line

        def __iter__(self):
            return iter(self.trailing)

    class FakeProcess:
        def __init__(
            self,
            polls: list[int | None],
            stdout: FakeStdout,
            exit_code: int = 0,
        ) -> None:
            self.polls = polls
            self.stdout = stdout
            self.exit_code = exit_code
            self.killed = False

        def poll(self) -> int | None:
            return self.polls.pop(0) if self.polls else self.exit_code

        def wait(self) -> int:
            return self.exit_code

        def kill(self) -> None:
            self.killed = True

    class FakeSelector:
        def __init__(self, ready: list[list[tuple[object, object]]]) -> None:
            self.ready = ready

        def register(self, *args: object) -> None:
            assert args

        def select(self, timeout: float) -> list[tuple[object, object]]:
            assert timeout == 30.0
            return self.ready.pop(0) if self.ready else []

    heartbeat = FakeProcess(
        [None, 0],
        FakeStdout(trailing=("tail\n", audit.RESULT_PREFIX + json.dumps({"ok": True}) + "\n")),
    )
    monkeypatch.setattr(audit.subprocess, "Popen", lambda *args, **kwargs: heartbeat)
    monkeypatch.setattr(audit.selectors, "DefaultSelector", lambda: FakeSelector([[]]))
    assert audit._spawn_worker(["fake"], "heartbeat") == {"ok": True}
    assert "subprocess heartbeat" in capsys.readouterr().out

    non_object = FakeProcess([None], FakeStdout(audit.RESULT_PREFIX + json.dumps([]) + "\n"))
    monkeypatch.setattr(audit.subprocess, "Popen", lambda *args, **kwargs: non_object)
    monkeypatch.setattr(
        audit.selectors, "DefaultSelector", lambda: FakeSelector([[(object(), object())]])
    )
    with pytest.raises(RuntimeError, match="subprocess_non_object"):
        audit._spawn_worker(["fake"], "non-object")

    failed = FakeProcess([1], FakeStdout(), exit_code=1)
    monkeypatch.setattr(audit.subprocess, "Popen", lambda *args, **kwargs: failed)
    monkeypatch.setattr(audit.selectors, "DefaultSelector", lambda: FakeSelector([]))
    with pytest.raises(RuntimeError, match="subprocess_failed"):
        audit._spawn_worker(["fake"], "failed")

    timed = FakeProcess([None], FakeStdout())
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(audit.subprocess, "Popen", lambda *args, **kwargs: timed)
    monkeypatch.setattr(audit.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="subprocess_timeout"):
        audit._spawn_worker(["fake"], "timeout", timeout_s=1.0)
    assert timed.killed is True


def test_validation_subprocess_heartbeat_and_trailing_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CL-7269: validation emits a heartbeat and hashes trailing output."""

    class Output:
        def readline(self) -> str:
            return ""

        def __iter__(self):
            return iter(("trailing validation output\n",))

    class Process:
        stdout = Output()

        def __init__(self) -> None:
            self.polls: list[int | None] = [None, 0]

        def poll(self) -> int | None:
            return self.polls.pop(0)

        def wait(self) -> int:
            return 0

    class Selector:
        def register(self, *args: object) -> None:
            assert args

        def select(self, timeout: float) -> list[tuple[object, object]]:
            assert timeout == 30.0
            return []

    monkeypatch.setattr(audit.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(audit.selectors, "DefaultSelector", Selector)
    receipt = audit._command_receipt(["fake-validation"])
    assert receipt["exit_code"] == 0
    output = capsys.readouterr().out
    assert "validation heartbeat" in output
    assert "trailing validation output" in output
