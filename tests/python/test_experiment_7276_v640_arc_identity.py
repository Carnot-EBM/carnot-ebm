"""REQ-ARC-WMTE-7276 tests for the runtime identity handoff repair."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_7276_v640_arc_identity as experiment
from carnot.agentic import arc_eval_provenance as provenance


HF_ID = "fixture-owner/fixture-model-GGUF"
REVISION = "7276fixture"
FILENAME = "fixture-model.gguf"
PAYLOAD = b"isolated GGUF-shaped fixture bytes\n"


def _fixture(directory: Path) -> tuple[dict[str, str], Path, Path, dict[str, str]]:
    """Create a content-addressed snapshot with a pre-existing stable hard link."""

    digest = hashlib.sha256(PAYLOAD).hexdigest()
    root = directory / "models--fixture-owner--fixture-model-GGUF"
    blob = root / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(PAYLOAD)
    external = directory / "task-owned-model-copy.gguf"
    os.link(blob, external)
    requested = root / "snapshots" / REVISION / FILENAME
    requested.parent.mkdir(parents=True)
    requested.symlink_to(Path("../../blobs") / digest)
    spec = {
        "model_path": str(requested),
        "model_filename": FILENAME,
        "hf_id": HF_ID,
        "revision": REVISION,
        "model_file_hash": "sha256:" + digest,
    }
    return spec, requested, blob, {"model_path": str(requested), "model_alias": FILENAME}


def test_stable_preexisting_hard_link_is_supported_without_weakening_capture(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7276-STABLE-HARD-LINK."""

    spec, requested, blob, props = _fixture(tmp_path)
    source = provenance.capture_arc_model_identity_source_provenance(
        raw_server_props=props,
        requested_model_path=requested,
        source_kind="exp7276_scripted_props",
    )
    receipt = provenance.build_typed_arc_model_identity_receipt(
        selected_model_spec=spec,
        launch_model_argument=str(requested),
        raw_server_props=props,
        source_provenance=source,
    )
    obligations = {row["obligation"]: row for row in receipt["identity_obligation_rows"]}

    assert blob.stat().st_nlink == 2
    assert obligations["unique_file_identity"]["status"] == "supported"
    assert provenance.validate_typed_arc_model_identity_receipt(receipt).valid


def test_runtime_process_start_tick_is_bound_into_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7276-POLICY-HANDOFF binds process identity."""

    spec, requested, _blob, props = _fixture(tmp_path)
    start_tick = provenance.process_start_tick(os.getpid())
    assert isinstance(start_tick, int) and start_tick > 0
    source = provenance.capture_arc_model_identity_source_provenance(
        raw_server_props=props,
        requested_model_path=requested,
        source_kind="exp7276_scripted_props",
        launch_model_argument=str(requested),
        server_pid=os.getpid(),
        server_pid_start_tick=start_tick,
    )
    receipt = provenance.build_typed_arc_model_identity_receipt(
        selected_model_spec=spec,
        launch_model_argument=str(requested),
        raw_server_props=props,
        source_provenance=source,
    )
    process = receipt["source_provenance"]["runtime_process"]

    assert process["pid"] == os.getpid()
    assert process["launch_start_tick"] == process["observed_start_tick"] == start_tick
    assert provenance.validate_typed_arc_model_identity_receipt(receipt).valid

    stale_source = provenance.capture_arc_model_identity_source_provenance(
        raw_server_props=props,
        requested_model_path=requested,
        source_kind="exp7276_scripted_props",
        launch_model_argument=str(requested),
        server_pid=os.getpid(),
        server_pid_start_tick=start_tick + 1,
    )
    stale = provenance.build_typed_arc_model_identity_receipt(
        selected_model_spec=spec,
        launch_model_argument=str(requested),
        raw_server_props=props,
        source_provenance=stale_source,
    )
    stale_rows = {row["obligation"]: row for row in stale["identity_obligation_rows"]}
    assert stale_rows["source_provenance"]["status"] == "contradicted"
    assert not provenance.validate_typed_arc_model_identity_receipt(stale).valid


def test_live_server_source_without_process_evidence_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7276-UNSUPPORTED-CONTROLS rejects missing PID evidence."""

    spec, requested, _blob, props = _fixture(tmp_path)
    source = provenance.capture_arc_model_identity_source_provenance(
        raw_server_props=props,
        requested_model_path=requested,
        source_kind="live_server_props",
        launch_model_argument=str(requested),
    )
    receipt = provenance.build_typed_arc_model_identity_receipt(
        selected_model_spec=spec,
        launch_model_argument=str(requested),
        raw_server_props=props,
        source_provenance=source,
    )
    rows = {row["obligation"]: row for row in receipt["identity_obligation_rows"]}

    assert rows["source_provenance"]["status"] == "contradicted"
    assert not provenance.validate_typed_arc_model_identity_receipt(receipt).valid


def test_cpu_panel_reaches_real_policy_and_rejects_every_control(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7276-UNSUPPORTED-CONTROLS uses the shared path."""

    panel = experiment.run_cpu_identity_panel(tmp_path)
    reduction = experiment.reduce_identity_rows(panel["rows"])
    by_unit = {row["unit"]: row for row in panel["rows"]}

    assert set(by_unit) == set(experiment.PANEL_CASES)
    assert reduction["arc_identity_ready_score"] == 1
    assert reduction == panel["independent_reduction"]
    positive = by_unit["stable_preexisting_hardlink"]
    assert positive["accepted"] is True
    assert positive["legacy_unique_file_identity_status"] == "contradicted"
    assert positive["policy_entrypoint_receipt"]["factory"] == "make_carnot_agent"
    assert positive["policy_entrypoint_receipt"]["policy_class"] == "E3AgentPolicy"
    assert positive["policy_entrypoint_receipt"]["choose_action_called"] is True
    assert positive["policy_entrypoint_receipt"]["provenance_valid"] is True
    assert positive["policy_entrypoint_receipt"]["provenance_hash"].startswith("sha256:")
    for case in experiment.UNSUPPORTED_CASES:
        row = by_unit[case]
        assert row["accepted"] is False
        assert row["passed"] is True
        assert row["unsupported_obligations"]
        assert row["policy_entrypoint_receipt"]["choose_action_called"] is False


def test_historical_diagnosis_names_obligation_without_blaming_model() -> None:
    """REQ-ARC-WMTE-7276 reconstructs the Exp7263 lifecycle boundary."""

    rows = experiment.diagnose_historical_episodes(experiment.REPO_ROOT)

    assert len(rows) == 4
    assert {row["episode_id"] for row in rows} == {
        "re86:current_feedback",
        "re86:typed_witness_feedback",
        "r11l:current_feedback",
        "r11l:typed_witness_feedback",
    }
    assert all(row["unsupported_obligation"] == "unique_file_identity" for row in rows)
    assert all(row["failure_boundary"] == experiment.PROVENANCE_FAILURE_BOUNDARY for row in rows)
    assert all(row["model_or_gguf_wrong_inferred"] is False for row in rows)
    assert all(row["raw_server_props_retained"] is False for row in rows)
    assert all(row["source_receipt_retained"] is False for row in rows)
    assert all(row["target_link_count_at_run"] == 2 for row in rows)
    assert all(row["link_count_change_predates_run"] is True for row in rows)


def _complete_artifact(tmp_path: Path) -> dict[str, object]:
    panel = experiment.run_cpu_identity_panel(tmp_path / "panel")
    history = experiment.diagnose_historical_episodes(experiment.REPO_ROOT)
    raw_path = tmp_path / "rows.json"
    raw_path.write_text(json.dumps({"rows": panel["rows"]}), encoding="utf-8")
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps({"fixture": "hashed"}), encoding="utf-8")
    return experiment.build_complete_artifact(
        started_at_utc="2026-09-13T12:00:00+00:00",
        ended_at_utc="2026-09-13T12:00:01+00:00",
        duration_s=1.0,
        preconditions=[experiment.gate_check("fixture", "test", "available", True, True)],
        source_hashes={"fixture": "sha256:" + "1" * 64},
        panel=panel,
        historical_rows=history,
        raw_rows_path=raw_path,
        sidecar_path=sidecar_path,
        phase_spans=[{"phase": "cpu_panel", "duration_s": 0.5}],
        validation_receipts=[
            {
                "name": "focused",
                "command": "pytest focused",
                "exit_code": 0,
                "expected_exit_code": 0,
                "duration_s": 0.1,
                "log_path": "raw/focused.log",
                "log_sha256": "sha256:" + "2" * 64,
                "passed": True,
                "timed_out": False,
            }
        ],
    )


def test_complete_artifact_is_cold_validated_and_independently_reduced(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7276-TERMINAL-EVIDENCE validates ordinary fields."""

    artifact = _complete_artifact(tmp_path)

    assert experiment.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert str(artifact["honest_verdict"]).startswith("complete_circular_positive")
    assert artifact["arc_identity_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == experiment.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["execution_venue"] == "host"
    assert artifact["verifier_is_oracle"] is True
    assert set(artifact) == set(artifact["field_principles"])


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update({"MODEL_SPECS": [{}]}), "MODEL_SPECS"),
        (lambda value: value.update({"model_invoked": True}), "model_invoked"),
        (
            lambda value: value.update(
                {"invocation_counts": {**experiment.ZERO_INVOCATION_COUNTS, "usable_answers": 1}}
            ),
            "invocation_counts",
        ),
        (lambda value: value.update({"status": "partial"}), "status"),
        (lambda value: value.update({"verdict_class": "positive"}), "verdict_class"),
        (lambda value: value.update({"arc_identity_ready_score": 0}), "ready score"),
        (lambda value: value["rows"].pop(), "rows do not reduce"),
        (lambda value: value["field_principles"].pop("rows"), "field principles"),
        (lambda value: value.update({"random_seed": 0}), "checksum"),
    ],
)
def test_terminal_validator_rejects_forged_fields(tmp_path: Path, mutation, expected: str) -> None:
    """REQ-ARC-WMTE-7276 rejects success-shaped or internally stale artifacts."""

    artifact = deepcopy(_complete_artifact(tmp_path))
    mutation(artifact)
    if expected != "checksum":
        artifact["reproducibility_checksum"] = experiment.artifact_checksum(artifact)
    assert any(expected in error for error in experiment.validate_artifact(artifact))


def test_blocked_prerequisite_is_terminal_and_names_exact_gate(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7276-TERMINAL-EVIDENCE keeps external blocks terminal."""

    failed = experiment.gate_check("required_input", "upstream.json", "exists", True, False)
    artifact = experiment.build_blocked_artifact(
        started_at_utc="2026-09-13T12:00:00+00:00",
        ended_at_utc="2026-09-13T12:00:01+00:00",
        duration_s=1.0,
        preconditions=[failed],
        source_hashes={},
    )

    assert experiment.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["gate_check_summary"] == [
        {
            "blocked_by": "upstream.json",
            "failed_check": "required_input",
            "field": "exists",
            "observed_value": False,
            "expected_value": True,
        }
    ]


def test_raw_reducer_atomic_writer_and_cli_fail_closed(tmp_path: Path, capsys) -> None:
    """REQ-ARC-WMTE-7276 keeps raw evidence separate from terminal publication."""

    panel = experiment.run_cpu_identity_panel(tmp_path / "panel")
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps({"rows": panel["rows"]}), encoding="utf-8")
    assert experiment.independent_reduce(raw)["arc_identity_ready_score"] == 1
    assert experiment.main(["--reduce-raw", str(raw)]) == 0
    assert '"arc_identity_ready_score": 1' in capsys.readouterr().out
    raw.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="raw identity rows"):
        experiment.independent_reduce(raw)

    output = tmp_path / "nested" / "artifact.json"
    experiment.atomic_write(output, {"stable": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"stable": True}
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_validation_plan_is_scoped_and_contains_required_e2e(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7276 uses focused coverage and the E2E-009/010 paths."""

    commands = experiment.build_validation_commands(
        raw_rows=tmp_path / "rows.json",
        terminal_candidate=tmp_path / "candidate.json",
        coverage_json=tmp_path / "coverage.json",
    )
    rendered = [" ".join(row["command"]) for row in commands]

    assert any("test_experiment_7276_v640_arc_identity.py" in row for row in rendered)
    assert any("test_arc_induction_state_persistence.py" in row for row in rendered)
    assert any("test_arc_tool_grammar_transport.py" in row for row in rendered)
    assert any("scripts/arc_loop_solve.py" in row and "r11l" in row for row in rendered)
    assert any("ruff check" in row for row in rendered)
    assert any("ruff format --check" in row for row in rendered)
    assert any("mypy" in row for row in rendered)
    assert any("check_spec_coverage.py" in row for row in rendered)
    assert all("tests/python -q" not in row for row in rendered)
    assert all(row.get("timeout_s", 0) <= 900 for row in commands)


def test_thin_entrypoint_delegates_to_package_module() -> None:
    """REQ-ARC-WMTE-7276 keeps the experiment entrypoint thin."""

    wrapper = experiment.REPO_ROOT / experiment.WRAPPER_PATH
    source = wrapper.read_text(encoding="utf-8")
    assert "experiment_7276_v640_arc_identity import main" in source
    assert "raise SystemExit(main())" in source
    assert len(source.splitlines()) <= 20


def test_validator_defensive_matrix_and_transport_order(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7276 rejects malformed terminals and out-of-order props."""

    assert experiment.validate_artifact([]) == ["artifact must be an object"]
    transport = experiment._ScriptedIdentityTransport({"model_path": "/fixture"})
    with pytest.raises(RuntimeError, match="has not started"):
        transport.props()

    malformed = deepcopy(_complete_artifact(tmp_path))
    malformed.update(
        {
            "unexpected": True,
            "schema": "wrong",
            "experiment_id": "wrong",
            "run_date": "wrong",
            "inference_substrate": "wrong",
            "inference_substrate_class": "wrong",
            "execution_venue": "wrong",
            "verifier_is_oracle": False,
            "verdict_class": "unknown",
            "honest_verdict": "wrong",
        }
    )
    malformed["reproducibility_checksum"] = experiment.artifact_checksum(malformed)
    errors = experiment.validate_artifact(malformed)
    assert "field principles and terminal keys differ" in errors
    assert "schema is invalid" in errors
    assert "experiment identity is invalid" in errors
    assert "run_date is invalid" in errors
    assert "inference_substrate is invalid" in errors
    assert "inference_substrate_class is invalid" in errors
    assert "execution_venue must be host" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "verdict_class is outside the closed vocabulary" in errors
    assert "complete honest_verdict must start complete_" in errors


def test_blocked_validator_rejects_success_shape(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7276 keeps a blocked artifact fail-closed."""

    failed = experiment.gate_check("input", "missing", "exists", True, False)
    artifact = experiment.build_blocked_artifact(
        started_at_utc="2026-09-13T12:00:00+00:00",
        ended_at_utc="2026-09-13T12:00:01+00:00",
        duration_s=1.0,
        preconditions=[failed],
        source_hashes={},
    )
    artifact["verdict_class"] = "null"
    artifact["honest_verdict"] = "complete_null"
    artifact["gate_check_summary"] = []
    artifact["reproducibility_checksum"] = experiment.artifact_checksum(artifact)
    errors = experiment.validate_artifact(artifact)
    assert "blocked status requires blocked verdict_class" in errors
    assert "blocked honest_verdict must start blocked_" in errors
    assert "blocked result requires gate_check_summary" in errors


def test_cli_default_progress_and_non_list_raw_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """REQ-ARC-WMTE-7276 covers bounded CLI dispatch and progress output."""

    raw = tmp_path / "not-a-list.json"
    raw.write_text(json.dumps({"rows": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="raw identity rows"):
        experiment.independent_reduce(raw)
    monkeypatch.setattr(experiment, "run_experiment", lambda date: int(date != "20260913"))
    assert experiment.main(["--date", "20260913"]) == 0
    experiment._progress(0.0, "test", "event", "1/1")
    assert "phase=test event=event" in capsys.readouterr().out
