"""Receipt-only ARC lifecycle evidence for REQ-ARC-WMTE-7304."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import pytest

import scripts.adversarial_verify as adversarial
from carnot.experiment_7289_v641_arc_boundary import exercise_live_caller_seams
from carnot.experiment_7304_v642_arc_receipt import (
    EXPERIMENT_ID,
    MILESTONE,
    MODEL_SPECS,
    REQUIRED_VALIDATION_NAMES,
    RUN_DATE,
    ZERO_INVOCATION_COUNTS,
    _load_json_object,
    _progress,
    _relative_or_absolute,
    _repository_health,
    _terminal_validation_commands,
    _utc_now,
    artifact_checksum,
    atomic_write,
    authenticate_inputs,
    build_blocked_artifact,
    build_caller_handoff,
    build_terminal_artifact,
    build_validation_commands,
    check_dependency,
    parse_args,
    project_lifecycle_panel,
    run_cpu_receipt_panel,
    run_sidecar_negative_controls,
    validate_artifact,
    verify_sidecar,
)


REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("artifact", "observed"),
    [
        (None, "missing_artifact"),
        (
            {"status": "complete", "validation_scope_ready_score": 1, "quarantined": True},
            "quarantined",
        ),
        (
            {
                "status": "complete",
                "validation_scope_ready_score": 1,
                "verdict_class": "disqualified",
            },
            "disqualified",
        ),
        ({"status": "running", "validation_scope_ready_score": 1}, "running"),
        ({"status": "complete", "validation_scope_ready_score": 0}, 0),
    ],
)
def test_scenario_7304_dependency_rejects_unsafe_numeric_success(
    artifact: dict[str, object] | None, observed: object
) -> None:
    """SCENARIO-ARC-WMTE-7304-DEPENDENCY-BLOCK preserves the failed comparison."""

    row = check_dependency(
        artifact,
        upstream="results/experiment_7303_v642_validation_scope.json",
        field="validation_scope_ready_score",
        expected=1,
    )

    assert row["passed"] is False
    assert row["expected_value"] in {1, "complete"}
    assert row["observed_value"] == observed


def test_scenario_7304_dependency_failure_builds_terminal_blocked_receipt() -> None:
    """SCENARIO-ARC-WMTE-7304-DEPENDENCY-BLOCK is terminal, not partial."""

    failed = check_dependency(
        None,
        upstream="results/experiment_7303_v642_validation_scope.json",
        field="validation_scope_ready_score",
        expected=1,
    )
    artifact = build_blocked_artifact(
        started_at_utc="2026-09-14T00:00:00+00:00",
        completed_at_utc="2026-09-14T00:00:01+00:00",
        duration_s=1.0,
        preconditions_checked=[failed],
        source_artifact_hashes={},
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["arc_receipt_ready_score"] == 0
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["first_failure"] == failed
    assert validate_artifact(artifact) == []


def test_scenario_7304_lifecycle_controls_use_real_boundary_and_clean_children(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7304-LIFECYCLE-CONTROLS drives all shipped CPU cases."""

    panel = run_cpu_receipt_panel(tmp_path / "raw")
    rows, manifest = project_lifecycle_panel(panel, root=tmp_path)
    by_case = {row["case"]: row for row in rows}

    assert set(by_case) == {
        "pre_load_failure",
        "load_no_generation",
        "generation_timeout",
        "generation_unusable",
        "duplicate_events",
        "orphan_cleanup",
    }
    assert all(row["passed"] for row in rows)
    assert by_case["generation_timeout"]["receipt_state_counts"]["in_flight"] == 1
    assert by_case["generation_timeout"]["receipt_state_counts"]["completed"] == 1
    assert by_case["generation_unusable"]["receipt_state_counts"]["completed"] == 2
    assert by_case["duplicate_events"]["duplicate_event_count"] == 2
    assert all(row["owned_child_alive_after_cleanup"] is False for row in rows)
    assert all(entry["sha256"].startswith("sha256:") for entry in manifest)

    terminal_projection = json.dumps({"rows": rows, "fixture_sidecar_manifest": manifest})
    for forbidden in (
        "model_identity",
        "call_rows",
        "generation_calls_attempted",
        "model_loads_attempted",
    ):
        assert forbidden not in terminal_projection


def test_scenario_7304_sidecar_hash_and_event_count_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7304-SIDECAR-INTEGRITY rejects tampering and event loss."""

    panel = run_cpu_receipt_panel(tmp_path / "raw")
    _, manifest = project_lifecycle_panel(panel, root=tmp_path)
    first = manifest[1]
    authentic = verify_sidecar(
        tmp_path / first["path"],
        expected_sha256=first["sha256"],
        expected_event_count=first["event_count"],
    )
    controls = run_sidecar_negative_controls(
        tmp_path / first["path"],
        output_dir=tmp_path / "negative",
    )

    assert authentic == {"passed": True, "reason": "hash_and_event_count_match"}
    assert [row["control"] for row in controls] == ["hash_tampering", "missing_event"]
    assert all(row["rejected"] is True and row["passed"] is True for row in controls)
    assert controls[0]["reason"] == "sha256_mismatch"
    assert controls[1]["reason"] == "event_count_mismatch"


def test_scenario_7304_live_callers_and_next_handoff_are_hash_bound(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7304-CALLER-HANDOFF binds real seams and Exp7305 arguments."""

    seams = exercise_live_caller_seams(tmp_path / "seams")
    handoff = build_caller_handoff(REPO, seams)

    assert seams["load_boundary_reachable"] is True
    assert seams["selfparse_generation_boundary_reachable"] is True
    assert seams["identity_rejection_preserved"] is True
    assert handoff["entrypoint"] == "scripts/experiments/experiment_7305_v642_arc_selfparse.py"
    assert handoff["arguments"] == ["--date", RUN_DATE]
    assert handoff["selfparse_call_arguments"]["selfparse"] is True
    assert handoff["selfparse_call_arguments"]["tools_payload"] is None
    assert handoff["handoff_sha256"].startswith("sha256:")
    assert all(value.startswith("sha256:") for value in handoff["caller_code_hashes"].values())
    assert handoff["production_default_changed"] is False


def test_req_7304_authenticates_original_flags_dependency_and_validation_scope(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-7304 hashes real inputs before consuming Exp7303 readiness."""

    checks, hashes, history = authenticate_inputs(REPO, output_path=tmp_path / "result.json")

    assert all(row["passed"] for row in checks)
    assert (
        history["original_artifact"]["sha256"]
        == hashes["results/experiment_7289_v641_arc_boundary.json"]["sha256"]
    )
    assert history["original_adversarial_log"]["flag_kinds"] == [
        "INFERENCE_PROVENANCE_CONTRADICTION",
        "SUBSTRATE_CLASS_MISMATCH",
    ]
    dependency = next(row for row in checks if row["check"] == "dependency_gate")
    assert dependency["field"] == "validation_scope_ready_score"
    assert dependency["observed_value"] == dependency["expected_value"] == 1


def test_req_7304_validation_plan_is_scoped_and_contains_both_e2e_checks(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7304 runs exact affected tests, coverage, static checks, and smoke."""

    commands = build_validation_commands(REPO, private_root=tmp_path)
    by_name = {row.name: row for row in commands}

    assert "focused_exp7304" in by_name
    assert "affected_arc_eval_provenance" in by_name
    assert "e2e_009_cross_call_persistence" in by_name
    assert "e2e_010_tool_transport" in by_name
    assert "e2e_009_llm_off_environment_smoke" in by_name
    assert "changed_module_coverage_report" in by_name
    assert "changed_module_mypy" in by_name
    assert "scoped_spec_coverage" in by_name
    assert all("tests/python" not in row.argv for row in commands)
    assert "resolved_imports" in by_name["worktree_imports"].argv[3]


def _terminal_inputs(tmp_path: Path) -> dict[str, object]:
    panel = run_cpu_receipt_panel(tmp_path / "raw")
    rows, manifest = project_lifecycle_panel(panel, root=tmp_path)
    controls = run_sidecar_negative_controls(
        tmp_path / manifest[1]["path"], output_dir=tmp_path / "negative"
    )
    seams = exercise_live_caller_seams(tmp_path / "seams")
    handoff = build_caller_handoff(REPO, seams)
    return {
        "started_at_utc": "2026-09-14T00:00:00+00:00",
        "completed_at_utc": "2026-09-14T00:00:03+00:00",
        "duration_s": 3.0,
        "phase_durations_s": {"cpu_controls": 2.0, "validation": 1.0},
        "preconditions_checked": [
            {
                "upstream": "results/experiment_7303_v642_validation_scope.json",
                "check": "dependency_gate",
                "field": "validation_scope_ready_score",
                "observed_value": 1,
                "expected_value": 1,
                "passed": True,
            }
        ],
        "source_artifact_hashes": {
            "results/experiment_7289_v641_arc_boundary.json": {
                "sha256": "sha256:" + "a" * 64,
                "terminal_class": "disqualified",
                "quarantined": True,
            }
        },
        "lifecycle_rows": rows,
        "fixture_sidecar_manifest": manifest,
        "sidecar_control_rows": controls,
        "caller_handoff": handoff,
        "historical_evidence_sidecars": {
            "original_artifact": {
                "path": "results/experiment_7289_v641_arc_boundary.json",
                "sha256": "sha256:" + "a" * 64,
            },
            "original_adversarial_log": {
                "path": "results/raw/experiment_7289_v641_arc_boundary/validation/"
                "00_terminal_candidate_adversarial_verify.log",
                "sha256": "sha256:" + "b" * 64,
                "flag_kinds": [
                    "INFERENCE_PROVENANCE_CONTRADICTION",
                    "SUBSTRATE_CLASS_MISMATCH",
                ],
            },
        },
        "repository_health": {
            "affects_required_checks": False,
            "historical_failures": [
                {
                    "log_path": "results/raw/experiment_7289_v641_arc_boundary/validation/"
                    "04_full_python_suite.log",
                    "log_sha256": "sha256:" + "c" * 64,
                }
            ],
        },
        "validation_receipts": [
            {"name": name, "exit_code": 0, "passed": True} for name in REQUIRED_VALIDATION_NAMES
        ],
    }


def test_scenario_7304_terminal_artifact_has_clean_current_provenance(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7304-CURRENT-PROVENANCE emits only sidecar references."""

    artifact = build_terminal_artifact(**_terminal_inputs(tmp_path))

    assert validate_artifact(artifact) == []
    assert artifact["schema"] == "carnot.experiment_7304.v642.arc_receipt.v1"
    assert artifact["experiment_id"] == EXPERIMENT_ID
    assert artifact["milestone"] == MILESTONE
    assert artifact["MODEL_SPECS"] == MODEL_SPECS == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == ZERO_INVOCATION_COUNTS
    assert artifact["current_model_load_count"] == 0
    assert artifact["current_generation_count"] == 0
    assert artifact["arc_receipt_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["reproducibility_checksum"] == artifact_checksum(artifact)
    live, _, _ = adversarial._typed_invocation_evidence(artifact)
    assert live == []


def test_scenario_7304_unchanged_verifier_rejects_contradictory_control(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7304-CURRENT-PROVENANCE keeps the classifier strict."""

    artifact = build_terminal_artifact(**_terminal_inputs(tmp_path))
    clean_path = tmp_path / "clean.json"
    clean_path.write_text(json.dumps(artifact), encoding="utf-8")
    clean = adversarial._verify_artifact_impl(clean_path)

    contradictory = deepcopy(artifact)
    contradictory["model_invoked"] = True
    contradictory["reproducibility_checksum"] = artifact_checksum(contradictory)
    control_path = tmp_path / "contradictory.json"
    control_path.write_text(json.dumps(contradictory), encoding="utf-8")
    rejected = adversarial._verify_artifact_impl(control_path)

    assert not [row for row in clean["flags"] if row["severity"] == "critical"]
    assert {row["kind"] for row in rejected["flags"] if row["severity"] == "critical"} >= {
        "INFERENCE_PROVENANCE_CONTRADICTION",
        "SUBSTRATE_CLASS_MISMATCH",
    }


def test_req_7304_validator_rejects_current_claim_and_broken_sidecar_reference(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-7304 rejects current activity and unauthenticated evidence."""

    artifact = build_terminal_artifact(**_terminal_inputs(tmp_path))
    artifact["model_invoked"] = True
    artifact["fixture_sidecar_manifest"][0]["sha256"] = "sha256:bad"

    errors = validate_artifact(artifact)

    assert "current_model_invoked_must_be_false" in errors
    assert "fixture_sidecar_hash_invalid" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_req_7304_thin_entrypoint_and_arguments() -> None:
    """REQ-ARC-WMTE-7304 keeps the repository wrapper thin and date-bound."""

    wrapper = REPO / "scripts/experiments/experiment_7304_v642_arc_receipt.py"
    assert parse_args(["--date", RUN_DATE]).date == RUN_DATE
    assert wrapper.read_text(encoding="utf-8").count("from carnot.") == 1
    assert "main()" in wrapper.read_text(encoding="utf-8")
    assert "experiment_7304" not in (REPO / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    )


def test_req_7304_utility_failures_and_atomic_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7304 covers bounded utility failures without publishing placeholders."""

    output = tmp_path / "nested" / "receipt.json"
    atomic_write(output, {"ready": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ready": True}
    assert datetime.fromisoformat(_utc_now()).tzinfo is not None
    _progress(0.0, "test", "boundary", "unit=1")
    assert "phase=test event=boundary" in capsys.readouterr().out

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert _load_json_object(malformed) is None
    assert _load_json_object(tmp_path / "missing.json") is None
    assert _relative_or_absolute(output, tmp_path) == "nested/receipt.json"
    assert _relative_or_absolute(output, tmp_path / "elsewhere") == str(output.resolve())
    assert verify_sidecar(
        tmp_path / "missing.jsonl",
        expected_sha256="sha256:" + "0" * 64,
        expected_event_count=1,
    ) == {"passed": False, "reason": "missing_sidecar"}

    rows, manifest = project_lifecycle_panel({"rows": ["not-a-row"]}, root=tmp_path)
    assert rows == manifest == []

    commands = _terminal_validation_commands(REPO, tmp_path / "candidate.json")
    assert [row.name for row in commands] == [
        "terminal_candidate_adversarial_verify",
        "terminal_candidate_row_consistency_strict",
    ]
    health = _repository_health(REPO)
    assert health["affects_required_checks"] is False
    assert health["historical_failures"][0]["log_sha256"].startswith("sha256:")
