"""REQ-REPORT-7111: forward ARC provenance writer and consumer canaries."""

from __future__ import annotations

import copy
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest

from carnot.agentic.arc_eval_provenance import (
    NO_LLM_INFERENCE_SUBSTRATE,
    NOT_APPLICABLE,
    ArcEvalProvenanceInput,
    build_arc_eval_provenance,
    build_arc_level_claim_receipts,
    read_complete_legacy_arc_eval_provenance,
    serialize_arc_evaluation_payload,
    validate_arc_evaluation_row,
)
from carnot.experiment_7111_v624_arc_provenance_canary import (
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    main,
    validate_artifact,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import outer_loop_dashboard as dashboard_module  # noqa: E402


def _load_script(name: str, relative_path: str):
    if relative_path == "scripts/outer_loop_dashboard.py":
        return dashboard_module
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, REPO / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _record(solve_provenance: str) -> dict[str, object]:
    na = NOT_APPLICABLE
    return build_arc_eval_provenance(
        ArcEvalProvenanceInput(
            inference_substrate=NO_LLM_INFERENCE_SUBSTRATE,
            gpu_uuid=na,
            gpu_model=na,
            cuda_device=na,
            model_repository=na,
            model_filename=na,
            model_hash=na,
            n_ctx=na,
            server_binary=na,
            server_binary_hash=na,
            server_command_hash=na,
            endpoint=na,
            port=na,
            lease_id=na,
            lease_hash=na,
            lease_issued_at=na,
            lease_expires_at=na,
            lease_checked_at=na,
            request_count=0,
            completion_count=0,
            error_count=0,
            policy_hash="sha256:" + "1" * 64,
            factory_hash="sha256:" + "2" * 64,
            git_commit="3" * 40,
            solve_provenance=solve_provenance,
        )
    )


def _row(
    solve_provenance: str,
    *,
    game: str = "canary-live",
    levels: int = 2,
) -> dict[str, object]:
    frames = [
        {"action_index": 1, "level_before": 0, "level_after": 0},
        {"action_index": 2, "level_before": 0, "level_after": 1},
        {"action_index": 3, "level_before": 1, "level_after": 2},
    ]
    row: dict[str, object] = {
        "game": game,
        "solve_provenance": solve_provenance,
        "started_at": "2026-09-07T00:00:00+00:00",
        "finished_at": "2026-09-07T00:00:03+00:00",
        "actions": 3,
        "levels": levels,
        "frame_sequence": frames,
        "arc_eval_provenance": _record(solve_provenance),
    }
    row.update(
        build_arc_level_claim_receipts(
            game=game,
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            actions=3,
            level_up_actions=[2, 3][:levels],
            frame_sequence=frames,
            induction_attempts=[],
            level_induction_events=[],
        )
    )
    return row


@pytest.mark.parametrize(
    ("solve_provenance", "headline_eligible"),
    [
        ("live_agent_self_discovery", True),
        ("development_proxy", False),
        ("outer_loop_re", False),
    ],
)
def test_allowed_provenance_values_preserve_evidence_but_bound_credit(
    solve_provenance: str, headline_eligible: bool
) -> None:
    """SCENARIO-REPORT-7111-ENUM: all classes serialize, only live earns credit."""

    payload = {"policy": "e3", "per_game": [_row(solve_provenance)]}
    observed = json.loads(serialize_arc_evaluation_payload(payload))

    assert observed["per_game"][0]["solve_provenance"] == solve_provenance
    assert observed["per_game"][0]["arc_headline_eligible"] is headline_eligible
    assert (
        validate_arc_evaluation_row(observed["per_game"][0]).headline_eligible is headline_eligible
    )


@pytest.mark.parametrize("defect", ["missing", "invalid", "contradictory"])
def test_writer_rejects_missing_invalid_or_contradictory_provenance(defect: str) -> None:
    """SCENARIO-REPORT-7111-REJECT: defective new provenance never serializes."""

    row = _row("live_agent_self_discovery")
    if defect == "missing":
        del row["solve_provenance"]
    elif defect == "invalid":
        row["solve_provenance"] = "historical_guess"
    else:
        row["solve_provenance"] = "outer_loop_re"

    with pytest.raises(ValueError, match="solve_provenance"):
        serialize_arc_evaluation_payload({"policy": "e3", "per_game": [row]})


@pytest.mark.parametrize(
    "defect",
    [
        "missing_attempt",
        "attempt_cross_game",
        "bad_start",
        "bad_finish",
        "bad_actions",
        "short_levelups",
        "missing_runtime",
        "runtime_cross_game",
        "missing_frames",
        "bad_frame_count",
        "unjsonable_frames",
        "bad_hash",
        "bad_attempt_list",
        "bad_event_list",
    ],
)
def test_live_level_claim_requires_own_attempt_and_runtime_re_receipts(defect: str) -> None:
    """SCENARIO-REPORT-7111-RECEIPTS: missing or foreign receipts are ineligible."""

    row = _row("live_agent_self_discovery")
    if defect == "missing_attempt":
        del row["attempt_receipt"]
    elif defect == "attempt_cross_game":
        row["attempt_receipt"]["game"] = "different-game"
    elif defect == "bad_start":
        row["attempt_receipt"]["started_at"] = "not-a-timestamp"
    elif defect == "bad_finish":
        row["attempt_receipt"]["finished_at"] = "not-a-timestamp"
    elif defect == "bad_actions":
        row["actions"] = 0
    elif defect == "short_levelups":
        row["attempt_receipt"]["level_up_actions"] = []
    elif defect == "missing_runtime":
        del row["runtime_re_receipt"]
    elif defect == "runtime_cross_game":
        row["runtime_re_receipt"]["game"] = "different-game"
    elif defect == "missing_frames":
        del row["frame_sequence"]
    elif defect == "bad_frame_count":
        row["runtime_re_receipt"]["frame_count"] = 99
    elif defect == "unjsonable_frames":
        row["frame_sequence"] = [object()]
    elif defect == "bad_hash":
        row["runtime_re_receipt"]["frame_sequence_hash"] = "sha256:" + "0" * 64
    elif defect == "bad_attempt_list":
        row["runtime_re_receipt"]["induction_attempts"] = None
    else:
        row["runtime_re_receipt"]["level_induction_events"] = None

    decision = validate_arc_evaluation_row(row)
    assert decision.valid
    assert not decision.headline_eligible
    assert decision.headline_ineligibility


def test_non_object_and_zero_level_rows_are_validated_without_credit() -> None:
    """REQ-REPORT-7111 keeps malformed input and zero-level evidence out of headlines."""

    assert not validate_arc_evaluation_row([]).valid
    zero = _row("live_agent_self_discovery", levels=0)
    decision = validate_arc_evaluation_row(zero)
    assert decision.valid and not decision.headline_eligible
    record_only = {
        "solve_provenance": "development_proxy",
        "arc_eval_provenance": _record("development_proxy"),
    }
    assert validate_arc_evaluation_row(record_only).headline_eligible


@pytest.mark.parametrize("payload", [None, {}, {"per_game": "not-a-list"}])
def test_serializer_rejects_a_malformed_payload_container(payload: object) -> None:
    """REQ-REPORT-7111 applies before per-game rows can be silently skipped."""

    with pytest.raises(ValueError, match="payload|per_game"):
        serialize_arc_evaluation_payload(payload)


def test_dashboard_preserves_provenance_groups_and_uses_only_eligible_live_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7111-DASHBOARD: uncredited classes stay out of the headline."""

    dashboard = _load_script("_arc_dashboard_7111", "scripts/outer_loop_dashboard.py")
    monkeypatch.setattr(dashboard, "REPO", tmp_path)
    run_dir = tmp_path / "results" / "arc_leaderboard_eval_runs"
    run_dir.mkdir(parents=True)

    accepted_payload = {
        "policy": "e3",
        "per_game": [
            _row("live_agent_self_discovery", game="live", levels=2),
            _row("development_proxy", game="proxy", levels=3),
            _row("outer_loop_re", game="outer", levels=4),
        ],
    }
    (run_dir / "accepted.json").write_text(
        serialize_arc_evaluation_payload(accepted_payload), encoding="utf-8"
    )
    historical = run_dir / "legacy-missing.json"
    historical.write_text(
        json.dumps({"policy": "e3", "per_game": [{"game": "legacy", "levels": 5}]}),
        encoding="utf-8",
    )
    before = historical.read_bytes()

    observed = dashboard.generalization_levels()

    assert observed["levels"] == 14
    assert (observed["headline_levels"], observed["headline_games"]) == (2, 1)
    assert observed["provenance_level_counts"] == {
        "live_agent_self_discovery": 2,
        "development_proxy": 3,
        "outer_loop_re": 4,
        "missing_or_invalid": 5,
    }
    assert observed["provenance_rejected_games"] == 3
    assert historical.read_bytes() == before


def test_dashboard_legacy_read_is_compatible_but_never_credited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7111-LEGACY: read compatibility is not a backfill."""

    dashboard = _load_script("_arc_dashboard_7111_legacy", "scripts/outer_loop_dashboard.py")
    monkeypatch.setattr(dashboard, "REPO", tmp_path)
    run_dir = tmp_path / "results" / "arc_leaderboard_eval_runs"
    run_dir.mkdir(parents=True)
    legacy = run_dir / "historical.json"
    legacy.write_text(
        json.dumps({"policy": "e3", "per_game": [{"game": "old", "levels": 5}]}),
        encoding="utf-8",
    )
    before = legacy.read_bytes()

    observed = dashboard.generalization_levels()

    assert observed["measured"] is True
    assert observed["headline_levels"] == 0
    assert observed["headline_eligible"] is False
    assert observed["provenance_level_counts"]["missing_or_invalid"] == 5
    assert legacy.read_bytes() == before

    legacy_record = _record("development_proxy")
    assert read_complete_legacy_arc_eval_provenance(legacy_record) == legacy_record


def test_experiment_artifact_recomputes_score_verdict_and_checksum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7111-ARTIFACT: deterministic rows close the V624 gate."""

    scripts_path = str(REPO / "scripts")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != scripts_path])
    artifact = build_artifact(REPO, execution_date="20260907")

    assert validate_artifact(artifact) == []
    assert set(REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert artifact["arc_forward_provenance_ready_score"] == 1
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["offline_reproduced"] is False
    assert artifact["historical_rows_backfilled"] is False
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_registry_hash_before"] == artifact["arc_registry_hash_after"]
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete:")

    forged = copy.deepcopy(artifact)
    forged["arc_forward_provenance_ready_score"] = 0
    assert validate_artifact(forged)


def test_missing_preconditions_emit_a_complete_blocked_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-7111 preconditions fail closed without running a canary."""

    artifact = build_artifact(
        tmp_path,
        execution_date="20260907",
        output_path=tmp_path / "missing" / "artifact.json",
    )

    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["failed_check"] == "arc_evaluator_readable"
    assert validate_artifact(artifact) == []


def test_canary_exception_is_terminal_disqualification(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7111 converts a writer fault into a closed failed gate."""

    from carnot import experiment_7111_v624_arc_provenance_canary as experiment

    def fail_serialization(_payload: object) -> str:
        raise RuntimeError("forced canary fault")

    monkeypatch.setattr(experiment, "serialize_arc_evaluation_payload", fail_serialization)
    artifact = experiment.build_artifact(REPO, execution_date="20260907")

    assert artifact["verdict_class"] == "disqualified"
    assert artifact["arc_forward_provenance_ready_score"] == 0
    assert "forced canary fault" in artifact["gate_check_summary"]["observed_value"]
    assert validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda value: value.pop("rows"), "artifact fields"),
        (lambda value: value.__setitem__("field_principles", None), "field_principles"),
        (
            lambda value: value["field_principles"].__setitem__("rows", ""),
            "field principle",
        ),
        (lambda value: value.__setitem__("execution_venue", "unknown"), "execution_venue"),
        (lambda value: value.__setitem__("solve_provenance", "outer_loop_re"), "development_proxy"),
        (lambda value: value.__setitem__("offline_reproduced", True), "offline reproduction"),
        (lambda value: value.__setitem__("historical_rows_backfilled", True), "backfilled"),
        (lambda value: value.__setitem__("verifier_is_oracle", True), "not an ARC oracle"),
        (
            lambda value: value.__setitem__("arc_forward_provenance_ready_score", 0),
            "ready_score",
        ),
        (
            lambda value: value.__setitem__("inference_substrate_class", "aggregation"),
            "positive verdict",
        ),
        (lambda value: value.__setitem__("honest_verdict", "positive"), "terminal complete"),
        (lambda value: value.__setitem__("verdict_class", "null"), "verdict_class"),
        (lambda value: value.__setitem__("reproducibility_checksum", "forged"), "checksum"),
    ],
)
def test_artifact_validator_rejects_each_forged_contract_field(
    mutation, expected_error: str
) -> None:
    """SCENARIO-REPORT-7111-ARTIFACT: independent checks reject forged fields."""

    artifact = build_artifact(REPO, execution_date="20260907")
    mutation(artifact)
    assert any(expected_error in error for error in validate_artifact(artifact))


def test_blocked_validator_rejects_bad_class_and_diagnostic(tmp_path: Path) -> None:
    """REQ-REPORT-7111 blocked artifacts retain an exact no-run diagnosis."""

    artifact = build_artifact(tmp_path, execution_date="20260907")
    artifact["inference_substrate_class"] = "no_model_load"
    artifact["gate_check_summary"] = {}
    assert {
        "blocked verdict requires blocked_no_run",
        "blocked verdict requires an exact failed precondition summary",
    } <= set(validate_artifact(artifact))


def test_main_writes_success_validation_error_and_disqualified_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7111 CLI writes one terminal artifact for every closed return."""

    from carnot import experiment_7111_v624_arc_provenance_canary as experiment

    positive = build_artifact(REPO, execution_date="20260907")
    monkeypatch.setattr(
        experiment, "build_artifact", lambda *args, **kwargs: copy.deepcopy(positive)
    )
    output = tmp_path / "positive.json"
    assert main(["--date", "20260907", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "positive"

    monkeypatch.setattr(experiment, "validate_artifact", lambda _artifact: ["forced invalid"])
    assert main(["--date", "20260907", "--output", str(tmp_path / "invalid.json")]) == 2

    disqualified = copy.deepcopy(positive)
    disqualified["verdict_class"] = "disqualified"
    monkeypatch.setattr(experiment, "build_artifact", lambda *args, **kwargs: disqualified)
    monkeypatch.setattr(experiment, "validate_artifact", lambda _artifact: [])
    assert main(["--date", "20260907", "--output", str(tmp_path / "failed.json")]) == 1


def test_module_entrypoint_executes_the_same_terminal_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7111 keeps `python -m` on the tested main path."""

    output = tmp_path / "module-entry.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_7111", "--date", "20260907", "--output", str(output)],
    )
    with pytest.warns(RuntimeWarning), pytest.raises(SystemExit) as stopped:
        runpy.run_module("carnot.experiment_7111_v624_arc_provenance_canary", run_name="__main__")
    assert stopped.value.code == 0
    assert json.loads(output.read_text())["arc_forward_provenance_ready_score"] == 1
