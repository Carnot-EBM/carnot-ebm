"""Tests for the first residual-memory chronological comparison shard.

Spec refs: REQ-CL-6840, SCENARIO-CL-6840-PRECONDITIONS,
SCENARIO-CL-6840-ISOLATION, SCENARIO-CL-6840-PARITY,
SCENARIO-CL-6840-CREDIT, SCENARIO-CL-6840-RESTART, and
SCENARIO-CL-6840-METRICS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_6840_residual_memory_chronological_shard_a as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = exp.source_paths_for_root(REPO_ROOT)


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """SCENARIO-CL-6840-PRECONDITIONS reuses frozen upstream artifacts."""

    return exp.load_sources(SOURCE_PATHS)


@pytest.fixture(scope="module")
def artifact(sources: dict[str, dict], tmp_path_factory: pytest.TempPathFactory) -> dict:
    """REQ-CL-6840 builds one deterministic shard artifact."""

    return exp.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        state_root=tmp_path_factory.mktemp("exp6840-state"),
        run_date="20260901",
        duration_s=0.25,
    )


def test_req_cl_6840_spec_precedes_implementation() -> None:
    """REQ-CL-6840 declares its paths, fields, and scenarios."""

    spec = (REPO_ROOT / exp.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = spec.split("## REQ-CL-6840", 1)[1]
    for requirement_id in exp.OPEN_SPEC_IDS[1:]:
        assert requirement_id in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for path in (exp.MODULE_RELATIVE_PATH, exp.SCRIPT_RELATIVE_PATH, exp.RESULT_RELATIVE_PATH):
        assert path.as_posix() in section


def test_scenario_cl_6840_preconditions_accept_sources(sources: dict[str, dict]) -> None:
    """SCENARIO-CL-6840-PRECONDITIONS accepts the frozen complete inputs."""

    summary = exp.check_preconditions(sources, SOURCE_PATHS)
    assert summary["passed"] is True
    assert summary["failed_checks"] == []
    assert all(row["passed"] for row in summary["checks"])


@pytest.mark.parametrize(
    ("fault", "failed_check"),
    [
        ("kernel", "residual_memory_kernel_ready_score"),
        ("assigned_order", "complete_assigned_orders"),
        ("source_hash", "source_artifact_hashes"),
        ("outcome", "exact_later_outcomes"),
        ("headroom", "nonzero_headroom"),
        ("held_disjoint", "disjoint_held_future_identities"),
    ],
)
def test_scenario_cl_6840_preconditions_fail_closed(
    sources: dict[str, dict],
    tmp_path: Path,
    fault: str,
    failed_check: str,
) -> None:
    """SCENARIO-CL-6840-PRECONDITIONS records blocked gate details."""

    changed = deepcopy(sources)
    paths = dict(SOURCE_PATHS)
    if fault == "kernel":
        changed["exp6839"]["residual_memory_kernel_ready_score"] = 0.0
    elif fault == "assigned_order":
        changed["exp6827"]["rows"] = [
            row for row in changed["exp6827"]["rows"] if row.get("order_id") != "order_3"
        ]
    elif fault == "source_hash":
        paths["exp6839"] = tmp_path / "missing.json"
    elif fault == "outcome":
        event_id = exp.select_assigned_events(changed["exp6827"])[0]["row_id"]
        for row in changed["exp6827"]["rows"]:
            if row["row_id"] == event_id:
                row["outcome_identity"] = None
                break
    elif fault == "headroom":
        changed["exp6827"]["headroom_metrics"]["later_read_opportunity_count"] = 0
    else:
        held_id = next(
            row["event_id"]
            for row in changed["exp6827"]["rows"]
            if row.get("split") == "held_future"
        )
        changed["exp6827"]["split_manifest"]["by_family"]["unsloth/Qwen3.6-35B-A3B-GGUF"][
            "development"
        ].append(held_id)

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
    assert blocked["csl_shard_a_complete_score"] == 0.0
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith(exp.BLOCKED_STATUS)
    assert failed_check in blocked["gate_check_summary"]["failed_checks"]


def test_scenario_cl_6840_isolation_freezes_before_outcomes(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6840-ISOLATION keeps future outcomes out of decisions."""

    event = exp.select_assigned_events(sources["exp6827"])[0]
    state = exp.ShardState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    proposal = exp.freeze_decision(event, state, seed=exp.RANDOM_SEEDS[0], event_sequence_index=1)
    repeated = exp.freeze_decision(event, state, seed=exp.RANDOM_SEEDS[0], event_sequence_index=1)
    changed_seed = exp.freeze_decision(
        event, state, seed=exp.RANDOM_SEEDS[1], event_sequence_index=1
    )
    changed_arm = exp.freeze_decision(
        event,
        exp.ShardState.for_arm(exp.READ_ONLY_ARM),
        seed=exp.RANDOM_SEEDS[0],
        event_sequence_index=1,
    )

    assert proposal == repeated
    assert proposal["decision_hash"] != changed_seed["decision_hash"]
    assert proposal["decision_hash"] != changed_arm["decision_hash"]
    assert proposal["decision_frozen_before_outcome_reveal"] is True
    assert set(proposal["pre_reveal_input_keys"]).isdisjoint(exp.OUTCOME_FIELD_DENYLIST)
    assert "outcome_identity" not in proposal["decision_material"]
    assert proposal["memory_read_receipt"]["max_record_event_sequence_index"] is None


def test_scenario_cl_6840_credit_rejects_unauthenticated_updates(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6840-CREDIT requires exact source and outcome authority."""

    event = exp.select_assigned_events(sources["exp6827"])[4]
    state = exp.ShardState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    proposal = exp.freeze_decision(event, state, seed=exp.RANDOM_SEEDS[0], event_sequence_index=5)
    outcome = exp.reveal_exact_outcome(event)
    credit = exp.credit_for_decision(event, exp.VERIFIED_RESIDUAL_ARM, proposal, outcome)

    assert credit["signed_credit"] == pytest.approx(
        outcome["signed_direction"] * proposal["memory_dose"]
    )
    assert exp.bounded_dose(10.0) == 1.0
    assert exp.bounded_dose(-10.0) == -1.0

    parent_hash = state.state_hash()
    bad_event = {**event, "action_identity": None}
    rejected = state.apply_update(
        bad_event,
        proposal,
        {**outcome, "outcome_identity": None},
        credit,
        event_sequence_index=5,
        seed=exp.RANDOM_SEEDS[0],
    )
    assert rejected["reason"] == "missing_update_authority"
    assert rejected["memory_records_changed"] is False
    assert state.state_hash() == parent_hash

    accepted_or_rejected = state.apply_update(
        event,
        proposal,
        outcome,
        credit,
        event_sequence_index=5,
        seed=exp.RANDOM_SEEDS[0],
    )
    duplicate = state.apply_update(
        event,
        proposal,
        outcome,
        credit,
        event_sequence_index=5,
        seed=exp.RANDOM_SEEDS[0],
    )
    assert accepted_or_rejected["reason"] != "duplicate_update"
    assert duplicate["reason"] == "duplicate_update"

    saved_bytes = state.state_bytes()
    state.residual_pressure = 1.0
    state._restore_bytes(saved_bytes)
    assert state.state_bytes() == saved_bytes

    tmp_bad_json = Path("/tmp/carnot-exp6840-bad.json")
    tmp_bad_json.write_text("{", encoding="utf-8")
    try:
        loaded = exp.load_sources(
            {"list": tmp_bad_json.parent, "bad": tmp_bad_json, "missing": Path("/tmp/nope.json")}
        )
        assert loaded["list"] == {"_load_error": "IsADirectoryError"}
        assert loaded["bad"] == {"_load_error": "JSONDecodeError"}
        assert loaded["missing"] == {"_load_error": "FileNotFoundError"}
    finally:
        tmp_bad_json.unlink(missing_ok=True)

    list_json = Path("/tmp/carnot-exp6840-list.json")
    list_json.write_text("[]", encoding="utf-8")
    try:
        assert exp.load_sources({"list": list_json})["list"] == {"_load_error": "not_object"}
    finally:
        list_json.unlink(missing_ok=True)

    assert exp.select_assigned_events({"rows": "not-a-list"}) == []
    malformed_split = deepcopy(sources)
    malformed_split["exp6827"]["split_manifest"]["by_family"]["bad"] = []
    malformed = exp.check_preconditions(malformed_split, SOURCE_PATHS)
    assert malformed["passed"] is True


def test_scenario_cl_6840_arm_seed_and_capacity_parity(artifact: dict) -> None:
    """SCENARIO-CL-6840-PARITY gives every arm and seed identical work."""

    rows = artifact["rows"]
    expected_rows = exp.EXPECTED_ASSIGNED_EVENT_COUNT * len(exp.ARM_NAMES) * len(exp.RANDOM_SEEDS)
    assert len(rows) == expected_rows
    assert artifact["split_manifest"]["assigned_order_indices"] == [0, 1, 2]
    assert artifact["split_manifest"]["no_pooling_with_second_shard"] is True
    assert {row["order_index"] for row in rows} == {0, 1, 2}
    assert {row["arm"] for row in rows} == set(exp.ARM_NAMES)
    assert {row["seed"] for row in rows} == set(exp.RANDOM_SEEDS)
    assert {
        contract["compute_budget_units"] for contract in artifact["arm_contracts"].values()
    } == {1}
    assert {contract["capacity_budget"] for contract in artifact["arm_contracts"].values()} == {
        exp.CAPACITY_BUDGET
    }
    assert {tuple(contract["random_seeds"]) for contract in artifact["arm_contracts"].values()} == {
        exp.RANDOM_SEEDS
    }
    assert all(row["active_count_after"] <= exp.CAPACITY_BUDGET for row in rows)

    by_event_seed: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        key = (row["source_event_row_id"], row["seed"])
        by_event_seed.setdefault(key, set()).add(row["arm"])
    assert all(arms == set(exp.ARM_NAMES) for arms in by_event_seed.values())


def test_scenario_cl_6840_restart_replay_is_exact(
    sources: dict[str, dict],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6840-RESTART reloads canonical checkpoint bytes."""

    events = exp.select_assigned_events(sources["exp6827"])[:40]
    run = exp.run_comparison(events, state_root=tmp_path, exercise_restart=True)
    clean = exp.run_comparison(events, state_root=tmp_path / "clean", exercise_restart=False)
    manifest = exp.checkpoint_manifest_for_run(run, clean.final_state_hash)

    assert manifest["bytes_identity"] is True
    assert manifest["matches_clean_replay"] is True
    assert manifest["live_final_state_hash"] == clean.final_state_hash
    assert run.checkpoint_receipt["loaded_state_hash"] == run.checkpoint_receipt["saved_state_hash"]


def test_scenario_cl_6840_metrics_and_writer_are_row_complete(
    artifact: dict,
    sources: dict[str, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-6840-METRICS validates complete row-derived fields."""

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["csl_shard_a_complete_score"] == 1.0
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] in {"null", "positive", "partial"}
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact) == []

    for row in artifact["rows"]:
        assert exp.REQUIRED_ROW_FIELDS <= set(row)
        assert row["decision_frozen_before_outcome_reveal"] is True
        assert row["outcome_revealed_after_decision"] is True
        assert row["capacity_budget"] == exp.CAPACITY_BUDGET
        assert row["held_future_metric"]["is_held_future"] == (row["split"] == "held_future")

    held_rows = [row for row in artifact["rows"] if row["held_future_metric"]["is_held_future"]]
    no_headroom = [row for row in artifact["rows"] if row["no_headroom"]]
    assert sum(
        result["held_future_rows"] for result in artifact["held_future_results"].values()
    ) == len(held_rows)
    assert artifact["headroom_summary"]["no_headroom_rows"] == len(no_headroom)
    for arm, result in artifact["negative_transfer_results"].items():
        assert result["wins"] + result["ties"] + result["losses"] == result["held_future_rows"]
        if arm == exp.NO_MEMORY_ARM:
            assert result["negative_transfer_count"] == 0

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
    changed["honest_verdict"] = "blocked: no prefix"
    assert "honest_verdict lacks complete_ prefix" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"].pop()
    assert "complete row count mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"][0].pop("row_id")
    assert "row field coverage mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["csl_shard_a_complete_score"] = 0.0
    assert "complete artifact missing shard score" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["status"] = exp.BLOCKED_STATUS
    assert "blocked artifact must not expose rows" in exp.validate_artifact(changed)

    result_path = tmp_path / "artifact.json"
    exp.write_artifact(result_path, artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert exp.main(["--validate", "--result-path", str(result_path)]) == 0

    with pytest.raises(ValueError, match="required field set mismatch"):
        exp.write_artifact(tmp_path / "bad.json", {"field_principles": {}})

    bad_validate_path = tmp_path / "bad-validate.json"
    bad_validate_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="blocked artifact must not expose rows"):
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
    temp_artifact = exp.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        state_root=None,
        run_date="20260901",
        duration_s=0.25,
    )
    assert temp_artifact["checkpoint_manifest"]["matches_clean_replay"] is True

    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: artifact)
    assert exp.main(["--date", "20260901", "--result-path", str(generated_path)]) == 0
    assert json.loads(generated_path.read_text(encoding="utf-8")) == artifact

    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: changed)
    with pytest.raises(ValueError, match="blocked artifact must not expose rows"):
        exp.main(["--date", "20260901", "--result-path", str(tmp_path / "bad-main.json")])

    script = REPO_ROOT / exp.SCRIPT_RELATIVE_PATH
    monkeypatch.setattr(
        "sys.argv",
        [script.name, "--validate", "--result-path", str(result_path)],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
