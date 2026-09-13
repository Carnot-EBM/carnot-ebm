"""Verify the independent bounded-coverage audit for REQ-CL-7255."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7253_v638_coverage_memory as exp7253
from carnot import experiment_7255_v638_coverage_audit as exp7255


def test_preconditions_authenticate_null_learner(tmp_path: Path) -> None:
    """SCENARIO-CL-7255-PRECONDITIONS: a complete null remains auditable."""

    assert exp7255.RUN_DATE == "20260913"
    paths = exp7255.ExperimentPaths.under(tmp_path)
    checks, upstreams, hashes = exp7255.collect_preconditions(exp7255.REPO_ROOT, paths)

    assert exp7255.gate_summary(checks)["passed"] is True
    assert upstreams["exp7253"]["coverage_fixture_ready_score"] == 1
    assert upstreams["exp7254"]["coverage_learning_value_score"] == 0
    assert upstreams["exp7254"]["verdict_class"] == "null"
    assert hashes[str(exp7255.DEFAULT_LEARNER_ARTIFACT)] == exp7255.EXPECTED_LEARNER_SHA256
    assert hashes[str(exp7255.DEFAULT_OLD_AUDIT)] == exp7255.EXPECTED_OLD_AUDIT_SHA256


def test_missing_upstream_builds_exact_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-CL-7255-PRECONDITIONS: external absence is blocked, not partial."""

    paths = exp7255.ExperimentPaths.under(tmp_path / "out")
    checks, upstreams, hashes = exp7255.collect_preconditions(
        exp7255.REPO_ROOT,
        paths,
        learner_artifact=tmp_path / "missing.json",
    )
    artifact = exp7255.build_blocked_artifact(
        checks,
        upstreams,
        hashes,
        paths,
        stream_ids=("prospective-01",),
        duration_s=0.01,
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["coverage_audit_complete_score"] == 0
    assert artifact["coverage_promotion_score"] == 0
    assert artifact["gate_check_summary"]["field"]
    assert artifact["gate_check_summary"]["observed_value"] is None
    assert exp7255.validate_artifact(artifact, expected_stream_ids=("prospective-01",)) == []


def test_private_authority_and_public_leakage_mutations_fail() -> None:
    """SCENARIO-CL-7255-CAUSALITY: hidden labels and regimes fail closed."""

    views = exp7253.build_stream_views("prospective")
    public = deepcopy(views.public[:2])
    releases = deepcopy(views.releases[:2])
    authority = deepcopy(views.authority[:2])

    assert exp7255.stream_authority_errors(public, releases, authority) == []
    authority[0]["regime_id"] = "private-mutation"
    assert "private_regime_mismatch" in exp7255.stream_authority_errors(public, releases, authority)

    row = {
        "oracle_control": False,
        "controller_input_fields": ["event_id", "family_id", "numeric_value"],
        "held_out_label_visible_to_controller": False,
        "prediction_frozen_before_release": True,
    }
    assert exp7255.decision_visibility_errors(row) == []
    row["controller_input_fields"].append("later_released_label")
    assert "future_label_leakage" in exp7255.decision_visibility_errors(row)


def test_independent_reducer_rebuilds_one_authenticated_stream() -> None:
    """SCENARIO-CL-7255-REDUCTION: raw rows, not producer aggregates, define metrics."""

    result = exp7255.audit_raw_evidence(
        exp7255.REPO_ROOT,
        stream_ids=("prospective-01",),
        bootstrap_draws=100,
        progress=True,
    )
    learner = json.loads(
        (exp7255.REPO_ROOT / exp7255.DEFAULT_LEARNER_ARTIFACT).read_text(encoding="utf-8")
    )
    expected = [row for row in learner["rows"] if row["stream_id"] == "prospective-01"]

    assert result["rows"] == expected
    assert len(result["rows"]) == len(exp7253.ARMS)
    assert len(result["comparison_rows"]) == len(exp7255.COMPARISON_SPECS)
    assert all(row["bootstrap_draws"] == 100 for row in result["comparison_rows"])
    assert all(row["passed"] is True for row in result["raw_check_rows"])
    assert all(row["passed"] is True for row in result["replay_rows"])
    assert result["causal_summary"]["pre_release_difference_count"] == 0


def test_shuffle_effect_rows_keep_zero_headroom() -> None:
    """SCENARIO-CL-7255-CONTROLS: canceled selection and zero headroom stay visible."""

    observations = [
        {
            "stream_id": "prospective-01",
            "admission_mode": "coverage",
            "mapping_changed": True,
            "selection_changed": False,
            "eligible_candidate_count": 1,
        },
        {
            "stream_id": "prospective-01",
            "admission_mode": "coverage",
            "mapping_changed": True,
            "selection_changed": True,
            "eligible_candidate_count": 2,
        },
    ]
    decisions = [
        {
            "stream_id": "prospective-01",
            "admission_mode": "coverage",
            "aligned_prediction": "accept",
            "shuffled_prediction": "reject",
            "released_query_count_before": 1,
        }
    ]

    rows = exp7255.build_shuffle_effect_rows(observations, decisions)

    assert rows == [
        {
            "unit_id": "prospective-01:coverage",
            "stream_id": "prospective-01",
            "admission_mode": "coverage",
            "nomination_count": 2,
            "candidate_mapping_difference_count": 2,
            "selected_archive_difference_count": 1,
            "later_decision_difference_count": 1,
            "pre_release_decision_difference_count": 0,
            "zero_headroom_count": 1,
            "censored": False,
        }
    ]


def test_all_five_mutations_are_rejected(tmp_path: Path) -> None:
    """SCENARIO-CL-7255-MUTATIONS: every isolated attack fails its named check."""

    rows = exp7255.run_mutation_controls(tmp_path)

    assert {row["mutation"] for row in rows} == {
        "fifo_coverage_alias",
        "shuffle_canceled",
        "future_label_leak",
        "private_regime_change",
        "durable_write_omission",
    }
    assert all(row["expected_rejection"] == row["observed_rejection"] for row in rows)
    assert all(row["passed"] is True for row in rows)


def test_e2e_cold_commit_reject_reload_and_rollback(tmp_path: Path) -> None:
    """SCENARIO-CL-7255-E2E: accepted state has fresh-process decision parity."""

    rows = exp7255.run_e2e_controls(tmp_path, progress=True)

    assert {row["control"] for row in rows} == {
        "cold_load_unseen_query_before_feedback",
        "accepted_delayed_feedback_commit",
        "wrong_parent_rejected_without_write",
        "fresh_process_reload_decision_parity",
        "rollback_restores_parent_bytes",
    }
    assert all(row["passed"] is True for row in rows)


def test_recomputed_gates_preserve_recurrence_limit_and_null() -> None:
    """SCENARIO-CL-7255-TERMINAL: recurrence failure prevents promotion."""

    learner = json.loads(
        (exp7255.REPO_ROOT / exp7255.DEFAULT_LEARNER_ARTIFACT).read_text(encoding="utf-8")
    )
    comparisons = exp7255.build_comparison_rows(learner["rows"], draws=100)
    gates = exp7255.score_exp7254_gates(comparisons, learner["causal_summary"])
    scores = exp7255.derive_terminal_scores(
        audit_complete=True,
        gates=gates,
        safety_passed=True,
    )

    assert gates["recurrence_error_increase_vs_frozen_lte_0_02"]["expected"] == "<=0.02"
    assert gates["recurrence_error_increase_vs_frozen_lte_0_02"]["passed"] is False
    assert scores == (
        1,
        0,
        "null",
        "complete_null: coverage audit completed but promotion criteria did not all pass",
    )


def test_one_stream_build_seals_sidecar_and_terminal(tmp_path: Path) -> None:
    """REQ-CL-7255: a fresh worker builds a valid audit artifact atomically."""

    paths = exp7255.ExperimentPaths.under(tmp_path)
    artifact = exp7255.build_and_seal(
        exp7255.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        bootstrap_draws=100,
        progress=True,
    )

    assert artifact["status"] == "complete"
    assert artifact["coverage_audit_complete_score"] == 1
    assert artifact["coverage_promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert len(artifact["rows"]) == len(exp7253.ARMS)
    assert len(artifact["mutation_rows"]) == 5
    assert paths.mutation_sidecar.is_file()
    assert paths.checkpoint.is_file()
    assert (
        exp7255.validate_artifact(
            artifact,
            repo_root=exp7255.REPO_ROOT,
            expected_stream_ids=("prospective-01",),
            check_files=True,
        )
        == []
    )

    receipt = exp7255.write_artifact(
        paths.artifact,
        artifact,
        repo_root=exp7255.REPO_ROOT,
        expected_stream_ids=("prospective-01",),
    )
    assert receipt["sha256"] == exp7255._sha256_path(paths.artifact)
    assert receipt["bytes"] == paths.artifact.stat().st_size
    assert json.loads(paths.artifact.read_text(encoding="utf-8"))["experiment_id"] == 7255


def test_validator_rejects_false_promotion_and_bad_receipt(tmp_path: Path) -> None:
    """SCENARIO-CL-7255-TERMINAL: oracle and receipt rules fail closed."""

    paths = exp7255.ExperimentPaths.under(tmp_path)
    checks, upstreams, hashes = exp7255.collect_preconditions(exp7255.REPO_ROOT, paths)
    blocked = exp7255.build_blocked_artifact(
        checks,
        upstreams,
        hashes,
        paths,
        stream_ids=("prospective-01",),
        duration_s=0.01,
    )
    changed = deepcopy(blocked)
    changed["verdict_class"] = "positive"
    changed["reproducibility_checksum"] = exp7255.reproducibility_checksum(changed)

    assert "blocked_contract" in exp7255.validate_artifact(
        changed, expected_stream_ids=("prospective-01",)
    )
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        exp7255.attach_validation_receipts(blocked, [{"command": "pytest"}])


def test_cli_validation_and_wrapper_are_thin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7255: command dispatch validates and writes only through the module."""

    paths = exp7255.ExperimentPaths.under(tmp_path)
    checks, upstreams, hashes = exp7255.collect_preconditions(exp7255.REPO_ROOT, paths)
    blocked = exp7255.build_blocked_artifact(
        checks,
        upstreams,
        hashes,
        paths,
        stream_ids=("prospective-01",),
        duration_s=0.01,
    )
    paths.artifact.parent.mkdir(parents=True, exist_ok=True)
    paths.artifact.write_text(json.dumps(blocked), encoding="utf-8")

    monkeypatch.setattr(exp7255.ExperimentPaths, "under", classmethod(lambda cls, root: paths))
    assert (
        exp7255.main(["--date", exp7255.RUN_DATE, "--output-root", str(tmp_path), "--validate"])
        == 0
    )
    with pytest.raises(ValueError, match="run_date_must_equal"):
        exp7255.main(["--date", "19000101", "--output-root", str(tmp_path)])

    wrapper = exp7255.REPO_ROOT / "scripts/experiments/experiment_7255_v638_coverage_audit.py"
    source = wrapper.read_text(encoding="utf-8")
    assert "from carnot.experiment_7255_v638_coverage_audit import main" in source
    assert len(source.splitlines()) < 20


def test_fail_closed_helpers_and_all_cli_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-CL-7255-TERMINAL: defensive branches remain measured and fail closed."""

    assert (
        exp7255.ExperimentPaths.defaults().artifact == exp7255.REPO_ROOT / exp7255.DEFAULT_ARTIFACT
    )

    class BrokenChecksum:
        @staticmethod
        def reproducibility_checksum(value: object) -> str:
            del value
            raise ValueError("malformed")

    assert exp7255._artifact_checksum_valid(BrokenChecksum, {}) is False
    assert exp7255._load_text(tmp_path / "missing") == ""
    assert exp7255.decision_visibility_errors({"oracle_control": True}) == []
    assert set(
        exp7255.decision_visibility_errors(
            {
                "controller_input_fields": [],
                "held_out_label_visible_to_controller": True,
                "prediction_frozen_before_release": False,
            }
        )
    ) == {"future_label_leakage", "feedback_chronology"}

    views = exp7253.build_stream_views("prospective")
    malformed_authority = deepcopy(views.authority[:1])
    malformed_authority[0].pop("stream_seed")
    assert "authority_shape" in exp7255.stream_authority_errors(
        views.public[:1], views.releases[:1], malformed_authority
    )

    invalid_jsonl = tmp_path / "invalid.jsonl"
    invalid_jsonl.write_text("{\n", encoding="utf-8")
    with pytest.raises(exp7255.AuditEvidenceError, match="invalid_jsonl"):
        exp7255._selected_stream_rows(invalid_jsonl, ("prospective-01",))
    non_object_jsonl = tmp_path / "non-object.jsonl"
    non_object_jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(exp7255.AuditEvidenceError, match="non_object_jsonl"):
        exp7255._selected_stream_rows(non_object_jsonl, ("prospective-01",))

    first = exp7255._selected_stream_rows(
        exp7255.REPO_ROOT / exp7255.DEFAULT_PREQUENTIAL_ROWS,
        ("prospective-01",),
    )[0]
    malformed_row = deepcopy(first)
    malformed_row["stream_id"] = "wrong-stream"
    checks = exp7255._raw_evidence_checks(
        [malformed_row],
        exp7253.StreamViews(views.public[:1], views.authority[:1], views.releases[:1], {}),
        ("prospective-01",),
    )
    assert any(row["passed"] is False for row in checks)
    assert exp7255._bootstrap_interval([], 10, "empty") == {
        "estimate": 0.0,
        "ci95": [0.0, 0.0],
    }
    assert exp7255.derive_terminal_scores(
        audit_complete=True,
        gates={"all": {"passed": True}},
        safety_passed=True,
    )[1:3] == (1, "circular_positive")
    with pytest.raises(exp7255.AuditEvidenceError, match="deliberate"):
        exp7255._require_valid(["deliberate"])

    paths = exp7255.ExperimentPaths.under(tmp_path / "blocked")
    blocked = exp7255.build_and_seal(
        tmp_path / "absent-repository",
        paths,
        stream_ids=("prospective-01",),
        bootstrap_draws=10,
        progress=True,
    )
    assert blocked["status"] == "blocked"
    receipt = {
        "command": "pytest focused",
        "exit_code": 0,
        "classification": "required_scoped_coverage",
        "log_sha256": "sha256:" + "1" * 64,
    }
    assert exp7255.attach_validation_receipts(blocked, [receipt])["validation_receipts"] == [
        receipt
    ]

    state_path = tmp_path / "reload.json"
    exp7253.CoverageArchiveController().save(state_path)
    event = {"event_id": "q", "family_id": "lower_bound", "numeric_value": 313}
    assert exp7255._reload_worker(state_path, event)["prediction"][0] == "accept"
    with pytest.raises(ValueError, match="reload_worker_arguments"):
        exp7255.main(["--date", exp7255.RUN_DATE, "--reload-worker"])
    assert (
        exp7255.main(
            [
                "--date",
                exp7255.RUN_DATE,
                "--reload-worker",
                "--state-path",
                str(state_path),
                "--event-json",
                json.dumps(event),
            ]
        )
        == 0
    )

    monkeypatch.setattr(exp7255, "audit_raw_evidence", lambda *args, **kwargs: {"mock": True})
    assert exp7255.main(["--date", exp7255.RUN_DATE, "--audit-worker"]) == 0

    monkeypatch.setattr(exp7255, "build_and_seal", lambda *args, **kwargs: blocked)
    monkeypatch.setattr(exp7255, "validate_artifact", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        exp7255,
        "write_artifact",
        lambda *args, **kwargs: {"sha256": "sha256:" + "2" * 64, "bytes": 10},
    )
    assert exp7255.main(["--date", exp7255.RUN_DATE]) == 0
    assert "atomic terminal write" in capsys.readouterr().out


def test_corrupt_nomination_and_missing_durable_state_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-7255-CAUSALITY: independent safety evidence rejects corruption."""

    witnesses = [
        {
            "event_id": f"witness-{index}",
            "family_id": "lower_bound",
            "numeric_value": 0,
            "observed_label": "accept",
        }
        for index in range(8)
    ]
    masks = dict.fromkeys(exp7253.FAMILIES, 1)
    archive = exp7253.diagnostic_archive(
        "archive-0",
        0,
        exp7253.snapshot_signature(masks, witnesses),
        masks=masks,
        witness_ids=[row["event_id"] for row in witnesses],
    )
    receipt, selected = exp7253.nominate_archives([archive], witnesses, shuffled=False)
    receipt["before_evaluations"][0]["contradiction_count"] = 1
    unsafe = deepcopy(selected)
    assert unsafe is not None
    unsafe["survivor_masks"] = dict.fromkeys(exp7253.FAMILIES, 0)
    assert set(
        exp7255._nomination_error_names(receipt, {"archive-0": archive}, witnesses, unsafe)
    ) == {"reactivation_witness_gate", "unsafe_reactivation"}

    replay = {
        ("prospective-01", "coverage_archive_aligned"): {
            "final_state_sha256": "sha256:" + "0" * 64,
            "final_state_bytes": 1,
        }
    }
    assert exp7255._durable_state_error_count(tmp_path, replay, replay) == 1
