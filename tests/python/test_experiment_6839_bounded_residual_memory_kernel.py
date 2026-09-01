"""Tests for the bounded residual-memory kernel canary.

Spec refs: REQ-CL-6839, SCENARIO-CL-6839-PRECONDITIONS,
SCENARIO-CL-6839-CHRONOLOGY, SCENARIO-CL-6839-PROPOSAL-IDENTITY,
SCENARIO-CL-6839-CREDIT, SCENARIO-CL-6839-BOUNDS,
SCENARIO-CL-6839-STALE, SCENARIO-CL-6839-RECOVERY,
SCENARIO-CL-6839-DUPLICATES, and SCENARIO-CL-6839-VERDICT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_6839_bounded_residual_memory_kernel as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = exp.source_paths_for_root(REPO_ROOT)


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """SCENARIO-CL-6839-PRECONDITIONS reuses the frozen terminal artifacts."""

    return exp.load_sources(SOURCE_PATHS)


@pytest.fixture(scope="module")
def artifact(sources: dict[str, dict], tmp_path_factory: pytest.TempPathFactory) -> dict:
    """REQ-CL-6839 builds one deterministic canary artifact for row checks."""

    return exp.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        state_root=tmp_path_factory.mktemp("exp6839-state"),
        run_date="20260901",
        duration_s=0.25,
    )


def test_req_cl_6839_spec_precedes_implementation() -> None:
    """REQ-CL-6839 declares scenarios, paths, and all artifact fields."""

    spec = (REPO_ROOT / exp.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = spec.split("## REQ-CL-6839", 1)[1]
    for requirement_id in exp.OPEN_SPEC_IDS[1:]:
        assert requirement_id in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for path in (exp.MODULE_RELATIVE_PATH, exp.SCRIPT_RELATIVE_PATH, exp.RESULT_RELATIVE_PATH):
        assert path.as_posix() in section


def test_scenario_cl_6839_preconditions_accept_frozen_sources(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6839-PRECONDITIONS accepts the complete frozen inputs."""

    summary = exp.check_preconditions(sources, SOURCE_PATHS)
    assert summary["passed"] is True
    assert summary["failed_checks"] == []
    assert all(row["passed"] for row in summary["checks"])


@pytest.mark.parametrize(
    ("fault", "failed_check"),
    [
        ("v598", "v598_evidence_root_ready_score"),
        ("row_count", "complete_exp6827_stream"),
        ("order_hash", "stable_order_hashes"),
        ("split_hash", "stable_split_hash"),
        ("outcome", "exact_later_outcomes"),
        ("headroom", "nonzero_decision_headroom"),
        ("source_hash", "source_artifact_hashes"),
    ],
)
def test_scenario_cl_6839_preconditions_fail_closed(
    sources: dict[str, dict],
    tmp_path: Path,
    fault: str,
    failed_check: str,
) -> None:
    """SCENARIO-CL-6839-PRECONDITIONS writes blocked diagnostics."""

    changed = deepcopy(sources)
    paths = dict(SOURCE_PATHS)
    if fault == "v598":
        changed["exp6835"]["v598_evidence_root_ready_score"] = 0
    elif fault == "row_count":
        changed["exp6827"]["rows"].pop()
    elif fault == "order_hash":
        changed["exp6827"]["order_hashes"]["order_1"] = "sha256:" + "0" * 64
    elif fault == "split_hash":
        changed["exp6827"]["split_manifest"]["held_future_count_per_family"] = 0
    elif fault == "outcome":
        first = exp.select_canary_events(changed["exp6827"])[0]["row_id"]
        for row in changed["exp6827"]["rows"]:
            if row["row_id"] == first:
                row["outcome_identity"] = None
                break
    elif fault == "headroom":
        changed["exp6827"]["headroom_metrics"]["later_read_opportunity_count"] = 0
    else:
        paths["exp6827"] = tmp_path / "missing.json"

    summary = exp.check_preconditions(changed, paths)
    failed = next(row for row in summary["checks"] if row["check"] == failed_check)
    assert summary["passed"] is False
    assert failed["passed"] is False
    assert "observed" in failed

    blocked = exp.build_artifact(
        changed,
        source_paths=paths,
        state_root=tmp_path,
        run_date="20260901",
        duration_s=0.25,
    )
    assert blocked["status"] == exp.BLOCKED_STATUS
    assert blocked["rows"] == []
    assert blocked["residual_memory_kernel_ready_score"] == 0.0
    assert blocked["csl_kernel_execution_complete_score"] == 0.0
    assert blocked["honest_verdict"].startswith(exp.BLOCKED_STATUS)
    assert failed_check in blocked["gate_check_summary"]["failed_checks"]


def test_scenario_cl_6839_chronology_and_proposal_identity(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6839-CHRONOLOGY freezes proposals before outcome reveal."""

    event = exp.select_canary_events(sources["exp6827"])[0]
    state = exp.KernelState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    proposal = exp.propose_action(event, state, random_seed=exp.RANDOM_SEED)
    repeated = exp.propose_action(event, state, random_seed=exp.RANDOM_SEED)
    changed_arm = exp.propose_action(
        event,
        exp.KernelState.for_arm(exp.READ_ONLY_ARM),
        random_seed=exp.RANDOM_SEED,
    )

    assert proposal == repeated
    assert proposal["proposal_hash"] != changed_arm["proposal_hash"]
    assert proposal["decision_frozen_before_outcome_reveal"] is True
    assert set(proposal["decision_input_keys"]).isdisjoint(exp.OUTCOME_FIELD_DENYLIST)
    assert "outcome_identity" not in proposal["decision_material"]
    assert -1.0 <= proposal["action_dose"] <= 1.0


def test_scenario_cl_6839_credit_uses_exact_direction_and_bounded_dose(
    sources: dict[str, dict],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6839-CREDIT computes signed action-level credit."""

    event = exp.select_canary_events(sources["exp6827"])[3]
    proposal = exp.propose_action(
        event,
        exp.KernelState.for_arm(exp.READ_ONLY_ARM),
        random_seed=exp.RANDOM_SEED,
    )
    outcome = exp.exact_outcome_for_event(event)
    credit = exp.credit_for_transition(event, exp.READ_ONLY_ARM, proposal, outcome)

    assert outcome["signed_direction"] in {-1, 0, 1}
    assert credit["signed_credit"] == pytest.approx(
        outcome["signed_direction"] * proposal["action_dose"]
    )
    assert exp.bounded_dose(20.0) == 1.0
    assert exp.bounded_dose(-20.0) == -1.0

    zero_credit = exp.credit_for_transition(
        event,
        exp.NO_MEMORY_ARM,
        {**proposal, "action_dose": 0.0, "proposed_action": "route_neutral"},
        {**outcome, "signed_direction": 1},
    )
    assert zero_credit["signed_credit"] == 0.0
    assert zero_credit["credited"] is False

    state = exp.KernelState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    bad_proposal = {
        **exp.propose_action(event, state, random_seed=exp.RANDOM_SEED),
        "action_dose": 2.0,
    }
    rejected, _, _ = state.admit(event, bad_proposal, outcome, 1)
    assert rejected["reason"] == "dose_out_of_bounds"

    list_path = tmp_path / "list.json"
    bad_path = tmp_path / "bad.json"
    missing_path = tmp_path / "missing.json"
    list_path.write_text("[]", encoding="utf-8")
    bad_path.write_text("{", encoding="utf-8")
    loaded = exp.load_sources({"list": list_path, "bad": bad_path, "missing": missing_path})
    assert loaded["list"] == {"_load_error": "not_object"}
    assert loaded["bad"] == {"_load_error": "JSONDecodeError"}
    assert loaded["missing"] == {"_load_error": "FileNotFoundError"}
    assert exp.select_canary_events({"rows": "not-a-list"}) == []


def test_scenario_cl_6839_admission_capacity_eviction_and_stale_expiry(
    sources: dict[str, dict],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6839-BOUNDS and STALE keep memory finite."""

    events = exp.select_canary_events(sources["exp6827"])[:40]
    run = exp.run_state_machine(events, tmp_path, exercise_recovery=False)
    assert run.capacity_and_eviction_receipts
    assert run.stale_decay_receipts
    assert all(
        receipt["active_count_after"] <= exp.MEMORY_CAPACITY for receipt in run.admission_rows
    )
    assert all(
        receipt["active_count_after"] <= exp.MEMORY_CAPACITY for receipt in run.stale_decay_receipts
    )
    assert any(row["admitted"] for row in run.admission_rows)
    assert any(row["reason"] == "exact_nonpositive_direction" for row in run.admission_rows)


def test_scenario_cl_6839_persistence_restart_rollback_duplicates_and_crash_recovery(
    sources: dict[str, dict],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6839-RECOVERY and DUPLICATES preserve canonical bytes."""

    events = exp.select_canary_events(sources["exp6827"])
    run = exp.run_state_machine(events, tmp_path, exercise_recovery=True)
    assert run.restart_receipt["bytes_identity"] is True
    assert run.restart_receipt["crash_recovery_ignored_partial_checkpoint"] is True
    assert run.rollback_receipt["restored_parent_bytes"] is True
    assert run.rollback_receipt["rolled_back"] is True
    assert run.duplicate_receipt["reason"] == "duplicate_event"
    assert run.invalid_update_receipt["reason"] == "invalid_parent_hash"

    persisted = tmp_path / "checkpoint.json"
    partial = tmp_path / "checkpoint.json.tmp"
    partial.write_text("{partial", encoding="utf-8")
    recovered = exp.load_persisted_states(persisted)
    assert exp.combined_state_hash(recovered) == run.final_state_hash


def test_scenario_cl_6839_artifact_fields_and_verdict_are_row_derived(
    artifact: dict,
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6839-VERDICT is a null canary despite readiness."""

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(artifact)
    assert len(artifact["rows"]) == exp.CANARY_EVENT_COUNT * len(exp.ARM_NAMES)
    assert len(artifact["exact_outcome_credit_rows"]) == len(artifact["rows"])
    assert {row["arm"] for row in artifact["rows"]} == set(exp.ARM_NAMES)
    assert all(row["decision_frozen_before_outcome_reveal"] for row in artifact["rows"])
    assert any(row["admitted"] for row in artifact["admission_rows"])
    assert (
        artifact["restart_receipt"]["clean_replay_state_hash"]
        == artifact["clean_replay_state_hash"]
    )
    assert artifact["restart_receipt"]["final_state_hash"] == artifact["clean_replay_state_hash"]
    assert artifact["residual_memory_kernel_ready_score"] == 1.0
    assert artifact["csl_kernel_execution_complete_score"] == 1.0
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert "held_future_benefit" not in artifact["honest_verdict"]
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact) == []

    temp_artifact = exp.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        state_root=None,
        run_date="20260901",
        duration_s=0.25,
    )
    assert temp_artifact["clean_replay_state_hash"] == artifact["clean_replay_state_hash"]


def test_req_cl_6839_validation_writer_and_entry_points(
    artifact: dict,
    sources: dict[str, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6839 validates artifacts before publishing bytes."""

    changed = deepcopy(artifact)
    changed.pop("rows")
    assert "required field set mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["field_principles"].pop("rows")
    assert "field_principles coverage mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["inference_substrate"] = "LLM"
    assert "inference_substrate mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["continuous_self_learning_task"] = False
    assert "continuous_self_learning_task must be true" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["verifier_is_oracle"] = True
    assert "verifier_is_oracle must be false" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["verdict_class"] = "invented"
    assert "verdict_class outside closed enum" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "blocked: no complete prefix"
    assert "honest_verdict lacks complete_ prefix" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"].pop()
    assert "complete row count mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["exact_outcome_credit_rows"].pop()
    assert "credit row count mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["residual_memory_kernel_ready_score"] = 0.0
    assert "complete artifact missing ready score" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["csl_kernel_execution_complete_score"] = 0.0
    assert "complete artifact missing execution score" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["status"] = exp.BLOCKED_STATUS
    assert "blocked artifact must not expose transition rows" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["verdict_class"] = "positive"
    changed["honest_verdict"] = "complete_positive: unsupported canary benefit"
    assert "canary must not claim held-future benefit" in exp.validate_artifact(changed)

    result_path = tmp_path / "artifact.json"
    exp.write_artifact(result_path, artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert exp.main(["--validate", "--result-path", str(result_path)]) == 0

    with pytest.raises(ValueError, match="required field set mismatch"):
        exp.write_artifact(tmp_path / "bad.json", {"field_principles": {}})

    bad_validate_path = tmp_path / "bad-validate.json"
    bad_validate_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="canary must not claim held-future benefit"):
        exp.main(["--validate", "--result-path", str(bad_validate_path)])

    original_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced build error"])
    with pytest.raises(ValueError, match="forced build error"):
        exp.build_artifact(
            sources,
            source_paths=SOURCE_PATHS,
            state_root=tmp_path / "forced-build",
            run_date="20260901",
            duration_s=0.25,
        )
    monkeypatch.setattr(exp, "validate_artifact", original_validate)

    generated_path = tmp_path / "generated.json"
    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: artifact)
    assert exp.main(["--date", "20260901", "--result-path", str(generated_path)]) == 0
    assert json.loads(generated_path.read_text(encoding="utf-8")) == artifact

    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: changed)
    with pytest.raises(ValueError, match="canary must not claim held-future benefit"):
        exp.main(["--date", "20260901", "--result-path", str(tmp_path / "bad-main.json")])

    script = REPO_ROOT / exp.SCRIPT_RELATIVE_PATH
    monkeypatch.setattr(
        "sys.argv",
        [script.name, "--validate", "--result-path", str(result_path)],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
