"""Receipt-boundary tests for REQ-ARC-WMTE-7424."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7424_v651_arc_receipt_boundary as exp7424
from carnot.reporting import current_work_receipt


REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _validation_rows(tmp_path: Path) -> list[dict[str, object]]:
    """Return complete scoped receipts for a small artifact fixture."""

    names = (
        *exp7424.validation_scope.REQUIRED_CHECK_NAMES,
        *exp7424.REQUIRED_E2E,
        exp7424.ORIGINAL_REPLAY_NAME,
        exp7424.CONTRADICTION_CONTROL_NAME,
        *exp7424.REQUIRED_TERMINAL,
    )
    rows: list[dict[str, object]] = []
    for name in names:
        expected_exit = (
            1 if name in {exp7424.ORIGINAL_REPLAY_NAME, exp7424.CONTRADICTION_CONTROL_NAME} else 0
        )
        rows.append(
            {
                "name": name,
                "command_argv": ["fixture", name],
                "environment": {"COVERAGE_FILE": str(tmp_path / ".coverage")},
                "exit_code": expected_exit,
                "expected_exit_code": expected_exit,
                "duration_s": 0.01,
                "log_path": str(tmp_path / f"{name}.log"),
                "log_sha256": "sha256:" + "a" * 64,
                "passed": True,
                "timed_out": False,
                "output_tail": "fixture",
            }
        )
    return rows


def _panel() -> dict[str, object]:
    """Return all terminal budget dispositions without model-shaped evidence."""

    rows = [
        {
            "control_id": "limit",
            "episode_id": "limit",
            "request_id": "primary",
            "branch": "primary",
            "permit_granted": True,
            "attempted": True,
            "dispatched": True,
            "cancelled": False,
            "remaining_capacity": 1,
            "terminal_result": "completed",
            "recovered_after_restart": False,
        },
        {
            "control_id": "exception",
            "episode_id": "exception",
            "request_id": "repair",
            "branch": "repair",
            "permit_granted": True,
            "attempted": True,
            "dispatched": True,
            "cancelled": False,
            "remaining_capacity": 1,
            "terminal_result": "failed",
            "recovered_after_restart": False,
        },
        {
            "control_id": "cancel",
            "episode_id": "cancel",
            "request_id": "refinement",
            "branch": "refinement",
            "permit_granted": True,
            "attempted": True,
            "dispatched": True,
            "cancelled": True,
            "remaining_capacity": 1,
            "terminal_result": "cancelled",
            "recovered_after_restart": False,
        },
        {
            "control_id": "limit",
            "episode_id": "limit",
            "request_id": "repair-refused",
            "branch": "repair",
            "permit_granted": False,
            "attempted": False,
            "dispatched": False,
            "cancelled": False,
            "remaining_capacity": 0,
            "terminal_result": "rejected_before_dispatch",
            "recovered_after_restart": False,
        },
    ]
    return {
        "factory": "make_carnot_agent",
        "policy_class": "E3AgentPolicy",
        "request_budget_rows": rows,
        "scripted_events": [{"event_type": "scripted_http_exchange"}],
        "third_rejected_before_dispatch": True,
        "restart_ownership_passed": True,
        "interrupted_child_released": True,
        "all_controls_passed": True,
    }


def _artifact(tmp_path: Path) -> dict[str, object]:
    """Build one valid no-model artifact with immutable fixture sidecars."""

    historical = {
        "original_flags": {
            "status": "complete_disqualified_required_evidence",
            "verdict_class": "disqualified",
            "flagged_adversarial": True,
        },
        "typed_counter_paths": [
            {
                "path": "$.overflow_diagnosis.generation_calls_attempted",
                "field": "generation_calls_attempted",
                "value": 35,
            }
        ],
        "original_findings": [
            "INFERENCE_PROVENANCE_CONTRADICTION",
            "SUBSTRATE_CLASS_MISMATCH",
        ],
    }
    contradiction = exp7424.contradiction_fixture()
    manifest, references = exp7424.write_evidence_sidecars(
        root=tmp_path,
        sidecar_dir=tmp_path / "sidecars",
        historical_payload=historical,
        scripted_payload={
            "scripted_events": _panel()["scripted_events"],
            "contradiction_control": contradiction,
        },
    )
    now = 5_000_000_000
    return exp7424.build_terminal_artifact(
        root=tmp_path,
        started_at_utc="2026-09-19T00:00:00+00:00",
        ended_at_utc="2026-09-19T00:00:05+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=now,
        phase_spans=[
            {"phase": "measurement", "start_s": 0.0, "end_s": 3.0},
            {"phase": "validation", "start_s": 3.0, "end_s": 5.0},
            {"phase": "contradiction_control", "start_s": 1.0, "end_s": 2.0},
            {"phase": "startup", "start_s": 0.0, "end_s": 0.5},
        ],
        preconditions_checked=[
            exp7424.gate_row(
                "fixture",
                "precondition",
                True,
                True,
                upstream="test",
                artifact_field="fixture",
                principle="The fixture is available before dependent work.",
            )
        ],
        source_artifact_hashes={
            "fixture": {
                "path": "fixture",
                "sha256": "sha256:" + "b" * 64,
                "original_flags": historical["original_flags"],
            }
        },
        callback_panel=_panel(),
        scripted_evidence_manifest=manifest,
        sidecar_references=references,
        validation_receipts=_validation_rows(tmp_path),
        original_summary={
            "artifact_path": exp7424.EXP7411_PATH.as_posix(),
            "artifact_sha256": "sha256:" + "c" * 64,
            "status": "complete_disqualified_required_evidence",
            "verdict_class": "disqualified",
            "flagged_adversarial": True,
            "finding_kinds": [
                "INFERENCE_PROVENANCE_CONTRADICTION",
                "SUBSTRATE_CLASS_MISMATCH",
            ],
            "typed_counter_locations": ["$.overflow_diagnosis.generation_calls_attempted"],
        },
    )


@pytest.fixture(scope="module")
def ready_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    """Build one reusable artifact for independent mutation checks."""

    root = tmp_path_factory.mktemp("exp7424-artifact")
    return root, _artifact(root)


def test_original_disqualification_and_exact_nested_counter_are_preserved() -> None:
    """SCENARIO-ARC-WMTE-7424-ORIGINAL-DISQUALIFICATION keeps the old failure."""

    value = exp7424.inspect_original_disqualification(REPO)
    assert value["original_flags"] == {
        "status": "complete_disqualified_required_evidence",
        "verdict_class": "disqualified",
        "flagged_adversarial": True,
    }
    assert value["original_findings"][:2] == [
        "INFERENCE_PROVENANCE_CONTRADICTION",
        "SUBSTRATE_CLASS_MISMATCH",
    ]
    assert value["typed_counter_paths"] == [
        {
            "path": "$.overflow_diagnosis.generation_calls_attempted",
            "field": "generation_calls_attempted",
            "value": 35,
        }
    ]
    assert value["original_disqualification_preserved"] is True
    summary = exp7424._original_summary(value)
    assert summary["typed_counter_locations"] == ["$.overflow_diagnosis.generation_calls_attempted"]


def test_typed_counter_reducer_finds_only_nonzero_model_invocation_values() -> None:
    """REQ-ARC-WMTE-7424 independently finds typed invocation leakage."""

    value = {
        "invocation_counts": {"generation_calls_attempted": 0},
        "nested": {
            "generation_calls_attempted": 2,
            "model_loads_failed": 1,
            "attempted": 99,
        },
    }
    assert exp7424.find_typed_invocation_paths(value) == [
        {
            "path": "$.nested.generation_calls_attempted",
            "field": "generation_calls_attempted",
            "value": 2,
        },
        {
            "path": "$.nested.model_loads_failed",
            "field": "model_loads_failed",
            "value": 1,
        },
    ]


def test_actual_e3_callbacks_cover_limit_exception_cancellation_restart_and_child(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7424-CALLBACK-LIFECYCLE drives the real callback seam."""

    panel = exp7424.run_callback_panel(tmp_path)
    assert panel["factory"] == "make_carnot_agent"
    assert panel["policy_class"] == "E3AgentPolicy"
    assert panel["third_rejected_before_dispatch"] is True
    assert panel["restart_ownership_passed"] is True
    assert panel["interrupted_child_released"] is True
    assert panel["all_controls_passed"] is True
    rows = panel["request_budget_rows"]
    assert {row["branch"] for row in rows} >= {"primary", "refinement", "repair"}
    assert {row["terminal_result"] for row in rows} >= {
        "completed",
        "failed",
        "cancelled",
        "rejected_before_dispatch",
    }
    assert all(row["remaining_capacity"] >= 0 for row in rows)
    assert not any(row["terminal_result"] == "in_flight" for row in rows)
    reduction = exp7424.reduce_request_budget_rows(rows)
    assert reduction["accounting_valid"] is True
    assert reduction["budget_violations"] == 0
    assert reduction["all_attempts_terminal"] is True


def test_current_receipt_keeps_fixture_bytes_external_and_detects_tampering(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7424-CURRENT-WORK-BOUNDARY hash-binds sidecars."""

    artifact = _artifact(tmp_path)
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == current_work_receipt.ZERO_INVOCATION_COUNTS
    assert artifact["current_invocation_events"] == []
    assert exp7424.find_typed_invocation_paths(artifact) == []
    assert current_work_receipt.validate_current_work_receipt(artifact, root=tmp_path) == []
    sidecar = tmp_path / artifact["receipt_sidecars"][0]["path"]
    sidecar.write_text("{}\n", encoding="utf-8")
    errors = current_work_receipt.validate_current_work_receipt(artifact, root=tmp_path)
    assert any(error.startswith("sidecar_hash_mismatch:") for error in errors)


def test_contradiction_fixture_is_genuinely_current_and_cannot_be_ready() -> None:
    """SCENARIO-ARC-WMTE-7424-CONTRADICTION-CONTROL retains a failing mutation."""

    control = exp7424.contradiction_fixture()
    assert control["MODEL_SPECS"] == []
    assert control["model_invoked"] is False
    assert control["inference_substrate_class"] == "no_model_load"
    assert control["invocation_counts"]["generation_calls_attempted"] == 1
    assert control["arc_receipt_boundary_ready_score"] == 0


def test_terminal_artifact_is_clean_null_with_independent_reduction(
    ready_artifact: tuple[Path, dict[str, object]],
) -> None:
    """SCENARIO-ARC-WMTE-7424-TERMINAL accepts only the isolated no-model receipt."""

    root, artifact = ready_artifact
    assert exp7424.validate_artifact(artifact, root=root) == []
    assert artifact["schema"] == exp7424.SCHEMA
    assert artifact["experiment_id"] == exp7424.EXPERIMENT_ID
    assert artifact["milestone"] == exp7424.MILESTONE
    assert artifact["run_date"] == exp7424.RUN_DATE
    assert artifact["arc_receipt_boundary_ready_score"] == 1
    assert artifact["live_efficacy_score"] == artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["flagged_adversarial"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["small_ebm_training"]["performed"] is False
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["reproducibility_checksum"] == exp7424.artifact_checksum(artifact)
    path = root / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    reduced = exp7424.independent_reduce_file(path)
    assert reduced["matches_declared"] is True
    assert reduced["declared_ready"] == 1


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "wrong", "identity_invalid"),
        ("milestone", "wrong", "identity_invalid"),
        ("MODEL_SPECS", [{"model": "forbidden"}], "current_receipt_invalid"),
        ("model_invoked", True, "current_receipt_invalid"),
        ("inference_substrate_class", "gpu", "substrate_invalid"),
        ("execution_venue", "remote", "current_receipt_invalid"),
        ("arc_receipt_boundary_ready_score", 0, "readiness_mismatch"),
        ("promotion_score", 1, "score_invalid"),
        ("live_efficacy_score", 1, "score_invalid"),
        ("verdict_class", "positive", "verdict_invalid"),
    ],
)
def test_validator_rejects_claim_drift(
    ready_artifact: tuple[Path, dict[str, object]], field: str, value: object, error: str
) -> None:
    """REQ-ARC-WMTE-7424 rejects current-provenance and readiness drift."""

    root, source = ready_artifact
    changed = deepcopy(source)
    changed[field] = value
    changed["reproducibility_checksum"] = exp7424.artifact_checksum(changed)
    assert error in exp7424.validate_artifact(changed, root=root)


def test_validator_rejects_rows_manifest_principles_and_checksum(
    ready_artifact: tuple[Path, dict[str, object]],
) -> None:
    """SCENARIO-ARC-WMTE-7424-TERMINAL fails closed on evidence drift."""

    root, source = ready_artifact
    changed = deepcopy(source)
    changed["request_budget_rows"][0]["terminal_result"] = "in_flight"
    assert "raw_reduction_mismatch" in exp7424.validate_artifact(changed, root=root)

    changed = deepcopy(source)
    changed["invocation_counts"]["generation_calls_attempted"] = 1
    changed["reproducibility_checksum"] = exp7424.artifact_checksum(changed)
    assert "typed_invocation_leak" in exp7424.validate_artifact(changed, root=root)

    changed = deepcopy(source)
    changed["scripted_evidence_manifest"] = {}
    changed["reproducibility_checksum"] = exp7424.artifact_checksum(changed)
    assert "scripted_manifest_invalid" in exp7424.validate_artifact(changed, root=root)

    changed = deepcopy(source)
    changed["field_principles"].pop("schema")
    changed["reproducibility_checksum"] = exp7424.artifact_checksum(changed)
    assert "field_principles_incomplete" in exp7424.validate_artifact(changed, root=root)

    changed = deepcopy(source)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "checksum_mismatch" in exp7424.validate_artifact(changed, root=root)


def test_preconditions_plans_entrypoint_and_terminal_scope(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7424 freezes exact affected and E2E validation commands."""

    checks, hashes, summary = exp7424.collect_preconditions(REPO)
    assert all(row["passed"] for row in checks)
    assert exp7424.EXP7411_PATH.as_posix() in hashes
    assert summary["original_disqualification_preserved"] is True

    manifest = exp7424.affected_manifest()
    assert manifest.test_paths == (
        exp7424.TEST_PATH.as_posix(),
        exp7424.CURRENT_RECEIPT_TEST_PATH.as_posix(),
        exp7424.BUDGET_TEST_PATH.as_posix(),
    )
    commands = exp7424.build_validation_plan(REPO, tmp_path / "validation")
    assert exp7424.validation_contract.validate_command_plan(REPO, manifest, commands) == []
    assert {row.name for row in commands} == set(exp7424.validation_scope.REQUIRED_CHECK_NAMES)
    e2e = exp7424.e2e_command_specs(REPO, tmp_path / "e2e")
    assert {row.name for row in e2e} == set(exp7424.REQUIRED_E2E)
    smoke = next(row for row in e2e if row.name == "e2e_offline_smoke")
    assert "12" in smoke.argv
    assert str(tmp_path) in " ".join(smoke.argv)

    terminal = exp7424.terminal_command_specs(REPO, tmp_path / "candidate.json")
    assert [row.name for row in terminal] == list(exp7424.REQUIRED_TERMINAL)
    assert all(str(tmp_path / "candidate.json") in row.argv for row in terminal)
    control = exp7424._control_command(REPO, "control", tmp_path / "control.json")
    assert control.name == "control"
    assert str(tmp_path / "control.json") in control.argv

    wrapper = REPO / exp7424.WRAPPER_PATH
    text = wrapper.read_text(encoding="utf-8")
    assert text.count("from carnot.") == 1
    assert "main()" in text
    assert "research_conductor" not in text

    args = exp7424.parse_args(["--date", exp7424.RUN_DATE])
    assert args.date == exp7424.RUN_DATE
    with pytest.raises(SystemExit):
        exp7424.parse_args(["--date", "20260920"])


def test_helpers_report_failures_without_weakening_expected_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7424 keeps real control exits and progress boundaries."""

    source = tmp_path / "value.json"
    exp7424.atomic_json(source, {"value": 7})
    assert exp7424.load_object(source) == {"value": 7}
    source.write_text("not-json", encoding="utf-8")
    assert exp7424.load_object(source) == {}
    assert exp7424.load_object(tmp_path / "missing.json") == {}

    receipt = {
        "name": exp7424.ORIGINAL_REPLAY_NAME,
        "exit_code": 1,
        "passed": False,
        "timed_out": False,
        "output_tail": ("INFERENCE_PROVENANCE_CONTRADICTION SUBSTRATE_CLASS_MISMATCH"),
    }
    normalized = exp7424.classify_expected_failure(
        receipt,
        required_tokens=(
            "INFERENCE_PROVENANCE_CONTRADICTION",
            "SUBSTRATE_CLASS_MISMATCH",
        ),
    )
    assert normalized["exit_code"] == 1
    assert normalized["expected_exit_code"] == 1
    assert normalized["passed"] is True
    assert exp7424.validation_receipts_pass([normalized], [exp7424.ORIGINAL_REPLAY_NAME])

    failed = exp7424.classify_expected_failure(
        {**receipt, "output_tail": "missing second token"},
        required_tokens=("one", "two"),
    )
    assert failed["passed"] is False

    exp7424.progress(0.0, "test", "boundary", completed_units=1)
    assert "completed_units=1" in capsys.readouterr().out
    assert "+00:00" in exp7424.utc_now()
    phases: list[dict[str, object]] = []
    now = exp7424.time.monotonic()
    exp7424._phase(phases, "fixture", now, now, 1)
    assert phases[0]["completed_units"] == 1

    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(exp7424, "run_experiment", lambda root, date: calls.append((root, date)))
    assert exp7424.main(["--date", exp7424.RUN_DATE]) == 0
    assert calls == [(REPO, exp7424.RUN_DATE)]


def test_run_specs_preserves_command_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7424-TERMINAL retains exact command-local environment."""

    spec = exp7424.validation_contract.EnvironmentCommandSpec(
        "fixture",
        ("python", "-c", "pass"),
        "fixture",
        command_environment=(("COVERAGE_FILE", str(tmp_path / ".coverage")),),
    )
    smoke = exp7424.validation_scope.CommandSpec(
        "e2e_offline_smoke", ("python", "-c", "pass"), "fixture"
    )

    def fake_run_commands(
        root: Path,
        specs: list[object],
        *,
        log_dir: Path,
        extra_env: dict[str, str],
    ) -> list[dict[str, object]]:
        assert root == REPO
        current = specs[0]
        if current is spec:
            assert log_dir.name == "00_fixture"
            assert extra_env == {"COVERAGE_FILE": str(tmp_path / ".coverage")}
        else:
            assert current is smoke
            assert log_dir.name == "01_e2e_offline_smoke"
            assert extra_env == {"CARNOT_ARC_DISABLE_INDUCTION": "1"}
        return [{"name": current.name, "exit_code": 0, "passed": True, "timed_out": False}]

    monkeypatch.setattr(exp7424.validation_scope, "run_commands", fake_run_commands)
    rows = exp7424.run_specs(REPO, [spec, smoke], tmp_path / "logs")
    assert rows[0]["environment"] == {"COVERAGE_FILE": str(tmp_path / ".coverage")}
    assert rows[0]["started_at_utc"] <= rows[0]["ended_at_utc"]
    assert rows[1]["environment"] == {"CARNOT_ARC_DISABLE_INDUCTION": "1"}


def test_missing_precondition_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7424 records unavailable branch-local input without promotion."""

    spec = tmp_path / "spec.md"
    spec.write_text("REQ-ARC-WMTE-7424", encoding="utf-8")
    ops = tmp_path / "ops"
    ops.mkdir()
    (ops / "exclusion_manifest.yaml").write_text("retired: []\n", encoding="utf-8")
    monkeypatch.setattr(exp7424, "INPUT_PATHS", (Path("missing.json"),))
    monkeypatch.setattr(exp7424, "SPEC_PATH", Path("spec.md"))
    monkeypatch.setattr(exp7424, "EXP7411_PATH", Path("missing.json"))
    checks, hashes, summary = exp7424.collect_preconditions(tmp_path)
    assert checks[0]["passed"] is False
    assert hashes == {}
    assert summary["original_disqualification_preserved"] is False
