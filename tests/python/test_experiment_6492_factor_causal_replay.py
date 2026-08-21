"""Tests for Exp6492 executed factor causal replay.

Spec refs: REQ-VERIFY-6492, SCENARIO-VERIFY-6492-GATES,
SCENARIO-VERIFY-6492-FROZEN-MANIFEST, SCENARIO-VERIFY-6492-ADD-DROP,
SCENARIO-VERIFY-6492-CONTROLS-DOSE, SCENARIO-VERIFY-6492-NO-JUDGE,
SCENARIO-VERIFY-6492-ROWS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_6492_factor_causal_replay as mod
from carnot import task_runtime_receipts as receipts


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def _accepted_exp6491_fixture(tmp_path: Path) -> Path:
    """Create one accepted proposal from the real stream without rerunning a model."""

    payload = json.loads((REPO / mod.EXP6491_RELATIVE_PATH).read_text(encoding="utf-8"))
    event = payload["frozen_event_manifest"]["events"][0]
    proposal = payload["proposal_rows"][0]
    compile_row = payload["exact_compile_rows"][0]
    variable, value = next(iter(event["visible_context"]["partial_assignment"].items()))
    factor = {
        "factor_id": "accepted_prefix_pin",
        "kind": "partial_assignment_eq",
        "scope": [variable],
        "weight": 1,
        "variable": variable,
        "value": value,
    }
    semantic_payload = {
        "event_id": event["event_id"],
        "kind": "partial_assignment_eq",
        "scope": [variable],
        "weight": 1,
        "variable": variable,
        "value": value,
    }
    semantic_hash = receipts.sha256_json(semantic_payload)
    proposal.update(
        {
            "proposal": factor,
            "parse_receipt": {
                **proposal["parse_receipt"],
                "parse_status": "parsed",
                "forbidden_keys": [],
                "boundary_violation": False,
            },
            "answer_field_present": False,
            "label_field_present": False,
            "verifier_field_present": False,
            "release_authority_claimed": False,
            "final_outcome_field_present": False,
        }
    )
    compile_row.update(
        {
            "compile_outcome": "accept",
            "reason": "syntactic_and_visible_semantic_checks_passed",
            "factor_id": factor["factor_id"],
            "semantic_payload": semantic_payload,
            "semantic_hash": semantic_hash,
        }
    )
    payload["aggregate_row_recomputation"]["compile_outcome_counts"]["accept"] = 1
    payload["aggregate_row_recomputation"]["compile_outcome_counts"]["no_proposal"] -= 1
    path = tmp_path / "accepted_exp6491.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _artifact(tmp_path: Path, *, exp6491_path: Path | None = None) -> dict[str, Any]:
    return mod.build_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        exp6491_path=exp6491_path or mod.EXP6491_RELATIVE_PATH,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )


def test_req_verify_6492_spec_declares_causal_replay_contract() -> None:
    """REQ-VERIFY-6492: OpenSpec owns the replay audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6492") : text.index("REQ-VERIFY-6486")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-6492-GATES",
        "SCENARIO-VERIFY-6492-FROZEN-MANIFEST",
        "SCENARIO-VERIFY-6492-ADD-DROP",
        "SCENARIO-VERIFY-6492-CONTROLS-DOSE",
        "SCENARIO-VERIFY-6492-NO-JUDGE",
        "SCENARIO-VERIFY-6492-ROWS",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6492_real_stream_completes_as_null_with_zero_admissions(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6492-GATES/CONTROLS-DOSE: no accepted proposals stay rows."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())
    aggregate = artifact["aggregate_row_recomputation"]

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["factor_causal_audit_complete_score"] == 1.0
    assert artifact["causal_factor_signal_ready_score"] == 0.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    gates = {row["artifact_id"]: row for row in artifact["upstream_gate_receipts"]}
    assert gates["exp6489"]["field"] == "trajectory_contract_ready_score"
    assert gates["exp6489"]["observed"] == 1.0
    assert gates["exp6491"]["field"] == "factor_proposal_stream_ready_score"
    assert gates["exp6491"]["observed"] == 1.0
    assert gates["exp6478_requested"]["path"].endswith(
        "experiment_6478_held_exact_constraint_energy_selection.json"
    )
    assert gates["exp6478_requested"]["gate_passed"] is False
    assert gates["exp6478_canonical"]["observed"] == 1.0

    assert len(artifact["factor_eligibility_rows"]) == 4
    assert {row["eligibility"] for row in artifact["factor_eligibility_rows"]} == {
        "not_eligible"
    }
    assert artifact["replay_rows"] == []
    assert artifact["paired_effect_rows"] == []
    assert artifact["harmful_flip_rows"] == []
    assert aggregate == mod.recompute_aggregates_from_rows(artifact["per_unit_rows"])
    assert aggregate["accepted_model_factor_count"] == 0
    assert aggregate["factor_causal_audit_complete_score_from_rows"] == 1.0
    assert aggregate["causal_factor_signal_ready_score_from_rows"] == 0.0

    for row in artifact["dose_matching_rows"]:
        assert row["proposal_opportunity_count"] == 4
        assert row["admitted_event_count"] == 0
        assert row["exposure_dose"] == 0
        assert row["dose_matched_to_model"] is True


def test_scenario_verify_6492_accepted_factor_runs_add_drop_and_controls(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6492-ADD-DROP/CONTROLS-DOSE: accepted factors replay."""

    exp6491_path = _accepted_exp6491_fixture(tmp_path)
    artifact = _artifact(tmp_path / "accepted", exp6491_path=exp6491_path)
    rows = artifact["replay_rows"]
    paired = artifact["paired_effect_rows"]
    controls = artifact["control_matching_rows"]
    aggregate = artifact["aggregate_row_recomputation"]
    replay_event_count = artifact["frozen_replay_manifest"]["replay_event_count_per_factor"]

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_null"
    assert artifact["factor_causal_audit_complete_score"] == 1.0
    assert artifact["causal_factor_signal_ready_score"] == 0.0
    assert len([row for row in artifact["factor_eligibility_rows"] if row["eligibility"] == "eligible"]) == 1
    assert {row["control_type"] for row in controls} == set(mod.CONTROL_TYPES)
    assert replay_event_count == 2
    assert len(rows) == 2 * (1 + len(mod.CONTROL_TYPES)) * replay_event_count
    assert len(paired) == (1 + len(mod.CONTROL_TYPES)) * replay_event_count
    assert aggregate["accepted_model_factor_count"] == 1
    assert aggregate["all_expected_replays_present"] is True

    by_factor = {
        (row["factor_instance_id"], row["source_raw_row_hash"], row["arm"]): row
        for row in rows
    }
    model_pair = next(
        row
        for row in paired
        if row["factor_source"] == "model" and row["replay_split"] == "development"
    )
    assert (model_pair["factor_instance_id"], model_pair["source_raw_row_hash"], "absent") in by_factor
    assert (model_pair["factor_instance_id"], model_pair["source_raw_row_hash"], "present") in by_factor
    assert model_pair["delta_expansions"] == (
        by_factor[(model_pair["factor_instance_id"], model_pair["source_raw_row_hash"], "present")][
            "expansions"
        ]
        - by_factor[(model_pair["factor_instance_id"], model_pair["source_raw_row_hash"], "absent")][
            "expansions"
        ]
    )
    assert model_pair["delta_expansions"] == 0
    assert model_pair["delta_exact_check_calls"] >= 0
    assert model_pair["exact_authority"] == "exact_counterfactual_solver_replay"

    for row in rows:
        assert row["solver_configuration"]["backend"] == "exhaustive_prefix_replay"
        assert row["state_hash"].startswith("sha256:")
        assert row["solver_outcome"] in {"satisfiable", "infeasible"}
        assert row["termination"] in {"solution_found", "state_space_exhausted"}
        assert row["verifier_is_oracle"] is True
        assert row["model_score_used_as_label"] is False
        assert row["human_judgment_used_as_label"] is False

    for row in artifact["dose_matching_rows"]:
        assert row["proposal_opportunity_count"] == 4
        assert row["admitted_event_count"] == 1
        assert row["exposure_dose"] == replay_event_count * 2
        assert row["dose_matched_to_model"] is True


def test_scenario_verify_6492_rows_no_judge_and_validation_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6492-NO-JUDGE/ROWS: validation guards stay live."""

    clean = _artifact(tmp_path / "clean", exp6491_path=_accepted_exp6491_fixture(tmp_path))

    missing = deepcopy(clean)
    del missing["status"]
    assert mod.validate_artifact(missing) == ["missing required fields: status"]

    bad_checksum = deepcopy(clean)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    for field, message, value in (
        ("field_principles", "field_principles must cover exactly required fields", {}),
        ("field_provenance", "field_provenance must cover exactly required fields", {}),
        ("inference_substrate", "inference_substrate mismatch", "bad"),
        ("verifier_is_oracle", "verifier_is_oracle must be true for exact solver outcomes", False),
    ):
        mutated = deepcopy(clean)
        mutated[field] = value
        _with_checksum(mutated)
        assert message in mod.validate_artifact(mutated)

    judge = deepcopy(clean)
    judge["replay_rows"][0]["model_score_used_as_label"] = True
    _with_checksum(judge)
    assert "model scores or human judgment used as labels" in mod.validate_artifact(judge)

    bad_aggregate = deepcopy(clean)
    bad_aggregate["aggregate_row_recomputation"]["accepted_model_factor_count"] = -1
    _with_checksum(bad_aggregate)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad_aggregate)

    bad_complete = deepcopy(clean)
    bad_complete["factor_causal_audit_complete_score"] = 0.0
    _with_checksum(bad_complete)
    assert "factor_causal_audit_complete_score mismatch" in mod.validate_artifact(bad_complete)

    bad_signal = deepcopy(clean)
    bad_signal["causal_factor_signal_ready_score"] = 1.0
    _with_checksum(bad_signal)
    assert "causal_factor_signal_ready_score mismatch" in mod.validate_artifact(bad_signal)

    bad_protected = deepcopy(clean)
    bad_protected["protected_files_unchanged"]["active_roadmap_and_conductor_unchanged"] = False
    _with_checksum(bad_protected)
    assert "protected files changed" in mod.validate_artifact(bad_protected)


def test_scenario_verify_6492_replay_handles_infeasible_and_noneligible_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6492-FROZEN-MANIFEST: terminal row states are explicit."""

    exp6491_path = _accepted_exp6491_fixture(tmp_path)
    artifact = _artifact(tmp_path / "accepted", exp6491_path=exp6491_path)
    accepted = next(row for row in artifact["factor_eligibility_rows"] if row["eligibility"] == "eligible")
    raw_by_hash = {
        row["raw_row_hash"]: row
        for row in json.loads((REPO / mod.EXP6489_RELATIVE_PATH).read_text())["raw_trajectory_rows"]
    }
    raw = raw_by_hash[accepted["source_raw_row_hash"]]
    variable, value = next(iter(raw["partial_assignment"].items()))
    infeasible_factor = {
        "factor_instance_id": "infeasible-test",
        "factor_source": "random_control",
        "factor_kind": "partial_assignment_eq",
        "semantic_payload": {
            "kind": "partial_assignment_eq",
            "scope": [variable],
            "weight": 1,
            "variable": variable,
            "value": 1 - int(value),
        },
    }

    row = mod.execute_replay(
        raw_row=raw,
        factor=infeasible_factor,
        arm="present",
        seed=mod.REPLAY_SEED,
    )
    assert row["solver_outcome"] == "infeasible"
    assert row["final_validity"] is False
    assert row["termination"] == "state_space_exhausted"

    dispositions = {row["compile_outcome"] for row in artifact["factor_eligibility_rows"]}
    assert {"accept", "reject", "no_proposal"} <= dispositions
    for row in artifact["factor_eligibility_rows"]:
        assert row["terminal_row_state"] in mod.TERMINAL_ROW_STATES


def test_scenario_verify_6492_helper_branches_and_run_path(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """SCENARIO-VERIFY-6492-ROWS: helper branches remain covered."""

    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert mod._read_json(tmp_path / "missing.json") is None
    assert mod._status_and_verdict(0.0, 0.0, {"all_gates_passed": False})[0].startswith(
        "blocked"
    )
    assert mod._status_and_verdict(1.0, 1.0, {"all_gates_passed": True})[0] == (
        "complete_positive"
    )
    assert mod._status_and_verdict(0.0, 0.0, {"all_gates_passed": True})[0] == (
        "disqualified"
    )

    run_artifact = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "run.json",
        root=REPO,
        tests_run=TESTS_RUN,
    )
    assert mod.validate_artifact(run_artifact) == []

    clean = _artifact(tmp_path / "clean", exp6491_path=_accepted_exp6491_fixture(tmp_path))
    raw_rows = json.loads((REPO / mod.EXP6489_RELATIVE_PATH).read_text())["raw_trajectory_rows"]
    raw_by_hash = {row["raw_row_hash"]: row for row in raw_rows}
    accepted = next(row for row in clean["factor_eligibility_rows"] if row["eligibility"] == "eligible")
    events = mod._replay_events_for_factor(
        accepted["factor_instance_id"],
        clean["frozen_replay_manifest"],
        raw_by_hash,
    )
    assert len(events) == 2
    assert mod._held_match_for(events[0], []) is None
    missing_manifest = mod.build_frozen_replay_manifest(
        {"raw_trajectory_rows": []},
        [{**accepted, "source_raw_row_hash": "sha256:missing"}],
    )
    assert missing_manifest["replay_event_count_per_factor"] == 0
    exp6489_payload = json.loads((REPO / mod.EXP6489_RELATIVE_PATH).read_text())
    fallback_manifest = deepcopy(clean["frozen_replay_manifest"])
    fallback_manifest["replay_event_groups"].insert(
        0,
        {
            "factor_instance_id": "not-the-control",
            "source_raw_row_hash": "sha256:not-the-source",
            "replay_events": [],
        },
    )
    fallback_rows = mod.build_replay_rows(
        exp6489_payload=exp6489_payload,
        eligibility_rows=[],
        control_rows=[clean["control_matching_rows"][0]],
        manifest=fallback_manifest,
    )
    assert fallback_rows

    raw = events[0]
    assignment = dict(raw["partial_assignment"])
    assert mod._factor_arity(None) == 0
    assert mod._factor_footprint(None) == []
    assert mod._factor_predicate(
        {"kind": "candidate_count_at_least", "threshold": 0},
        raw,
        assignment,
    )
    assert mod._factor_predicate(
        {"kind": "residual_weight_at_most", "threshold": 999},
        raw,
        assignment,
    )
    assert not mod._factor_predicate({"kind": "unknown"}, raw, assignment)
    assert mod.paired_effect_rows([clean["replay_rows"][0]]) == []

    absent = deepcopy(clean["replay_rows"][0])
    present = deepcopy(absent)
    absent["arm"] = "absent"
    absent["replay_row_hash"] = "sha256:absent"
    present["arm"] = "present"
    present["final_validity"] = False
    present["solver_outcome"] = "infeasible"
    present["exact_check_calls"] = absent["exact_check_calls"] + 1
    present["replay_row_hash"] = "sha256:present"
    harmful_pair = mod.paired_effect_rows([absent, present])
    assert harmful_pair[0]["harmful_flip"] is True
    assert mod.harmful_flip_rows(harmful_pair)[0]["row_type"] == "harmful_flip"

    monkeypatch.setattr(mod, "_git_output", lambda *_: " M scripts/research_conductor.py")
    protected = mod._protected_files_unchanged(REPO)
    assert protected["active_roadmap_and_conductor_unchanged"] is False

    for mutate, message in (
        (
            lambda a: a["paired_effect_rows"].append(deepcopy(a["paired_effect_rows"][0])),
            "paired_effect_rows mismatch",
        ),
        (
            lambda a: a["harmful_flip_rows"].append({"row_type": "harmful_flip"}),
            "harmful_flip_rows mismatch",
        ),
        (
            lambda a: a["factor_eligibility_rows"][0].update({"terminal_row_state": "bad"}),
            "factor eligibility terminal states must be enumerated",
        ),
        (
            lambda a: a.update({"honest_verdict": "bad"}),
            "honest_verdict lacks required terminal prefix",
        ),
    ):
        mutated = deepcopy(clean)
        mutate(mutated)
        _with_checksum(mutated)
        assert message in mod.validate_artifact(mutated)
