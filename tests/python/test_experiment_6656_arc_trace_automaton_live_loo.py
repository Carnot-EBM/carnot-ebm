"""Tests for REQ-ARC-WMTE-6656 and its held-family trace-FSM scenarios."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_trajectory_supervisor import TraceAutomatonSupervisor
from carnot import experiment_6656_arc_trace_automaton_live_loo as exp


class _Frame:
    """Provide the frame fields used by the canonical E3 action path."""

    def __init__(self, value: int = 0) -> None:
        grid = np.full((8, 8), value, dtype=int)
        self.frame = [grid.tolist()]
        self.levels_completed = 0
        self.state = "NOT_FINISHED"
        self.score = 0
        self.available_actions = [1, 2, 3, 4, 5, 6]


def _fsm(*, repeat: int = 2, stagnant: int = 1) -> dict:
    return {
        "schema": exp.FSM_SCHEMA,
        "states": list(exp.FSM_STATES),
        "initial_state": "bootstrap",
        "features": list(exp.POLICY_VISIBLE_FEATURES),
        "thresholds": {
            "same_action_run": repeat,
            "actions_since_observed_change": stagnant,
        },
        "transitions": list(exp.FSM_TRANSITIONS),
        "redirect_arms": ["reset_after_stagnant_repeat"],
        "tie_rules": ["single_eligible_arm", "reset_has_no_game_specific_payload"],
        "training_support_actions": 12,
        "training_family_count": 2,
        "frozen_before_held_evaluation": True,
    }


def test_scenario_6656_live_influence_runs_in_e3_next_move(monkeypatch):
    """SCENARIO-ARC-WMTE-6656-LIVE-INFLUENCE-AND-OUTCOME."""

    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    policy = E3AgentPolicy("opaque-held-family", proposer=None, explore_budget=4)
    supervisor = TraceAutomatonSupervisor(_fsm())
    policy.install_trace_automaton_supervisor(supervisor)
    policy.plan = [{"action": 1, "data": None}, {"action": 1, "data": None}]
    policy.phase = "execute"
    policy.induced = True
    frame = _Frame()

    first = policy.next_move([frame], frame)
    second = policy.next_move([frame], frame)
    policy.finalize_trace_automaton_supervisor()

    assert first == (1, None)
    assert second == ("RESET", None)
    receipt = policy.trace_automaton_supervisor_diagnostics()
    assert receipt["firings"] == 1
    assert receipt["action_influences"] == 1
    assert receipt["rows"][1]["blocked_valid_action"] is True
    assert receipt["rows"][0]["next_outcome"]["frame_changed"] is False


def test_scenario_6656_default_path_stays_inert(monkeypatch):
    """REQ-ARC-WMTE-6656 keeps the canonical path unchanged without opt-in."""

    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    policy = E3AgentPolicy("opaque-held-family", proposer=None, explore_budget=4)
    policy.plan = [{"action": 2, "data": None}, {"action": 2, "data": None}]
    policy.phase = "execute"
    policy.induced = True
    frame = _Frame()

    assert policy.next_move([frame], frame) == (2, None)
    assert policy.next_move([frame], frame) == (2, None)
    assert policy.trace_automaton_supervisor_diagnostics()["enabled"] is False


def test_scenario_6656_receipt_admission_rejects_attacks(tmp_path):
    """SCENARIO-ARC-WMTE-6656-RECEIPT-ADMISSION."""

    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps(
            {
                "live_path_entrypoint": "make_carnot_agent -> E3AgentPolicy.next_move",
                "episodes": [],
            }
        ),
        encoding="utf-8",
    )
    base = {
        "game": "train-a",
        "seed": 11,
        "budget": 4,
        "arm_label": "run-0",
        "provenance_armed": True,
        "provenance": {
            "schema": "carnot.arc.action_provenance.v1",
            "rows": [
                {
                    "i": 0,
                    "game": "train-a",
                    "action": 1,
                    "data": None,
                    "frame_changed_since_last_action": None,
                    "level_before": 0,
                    "level_after": 0,
                },
                {
                    "i": 1,
                    "game": "train-a",
                    "action": 1,
                    "data": None,
                    "frame_changed_since_last_action": False,
                    "level_before": 0,
                    "level_after": 0,
                },
            ],
        },
    }
    good = tmp_path / "good.json"
    good.write_text(json.dumps(base), encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(base), encoding="utf-8")
    shadow = tmp_path / "shadow.json"
    shadow_payload = copy.deepcopy(base)
    shadow_payload["shadow_only"] = True
    shadow.write_text(json.dumps(shadow_payload), encoding="utf-8")
    forbidden = tmp_path / "forbidden.json"
    forbidden_payload = copy.deepcopy(base)
    forbidden_payload["read_game_source"] = True
    forbidden.write_text(json.dumps(forbidden_payload), encoding="utf-8")
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical_payload = copy.deepcopy(base)
    noncanonical_payload["provenance_armed"] = False
    noncanonical.write_text(json.dumps(noncanonical_payload), encoding="utf-8")
    wrong_schema = tmp_path / "wrong-schema.json"
    wrong_schema_payload = copy.deepcopy(base)
    wrong_schema_payload["provenance"]["schema"] = "wrong"
    wrong_schema.write_text(json.dumps(wrong_schema_payload), encoding="utf-8")

    rows, audit = exp.collect_policy_visible_trace_rows(
        [good, duplicate, shadow, forbidden, noncanonical, wrong_schema], parent_path=parent
    )

    assert len(rows) == 1
    assert rows[0]["next_outcome"]["frame_changed"] is False
    assert audit["duplicate_action_count"] == 1
    assert {row["reason"] for row in audit["rejected_receipts"]} == {
        "shadow_only_evidence",
        "forbidden_evidence_marker",
        "not_canonical_live_e3_receipt",
        "wrong_receipt_schema",
    }
    assert audit["missing_outcome_action_count"] == 2


def test_scenario_6656_freeze_is_game_blind_and_stable():
    """SCENARIO-ARC-WMTE-6656-FREEZE-AND-ISOLATION."""

    rows = [
        {
            "family": family,
            "pre_action_features": {
                "previous_frame_changed": False,
                "same_action_run": run,
                "actions_since_observed_change": run,
                "level_progress_since_previous_action": False,
            },
        }
        for family in ("train-a", "train-b")
        for run in range(1, 5)
    ]

    first = exp.learn_frozen_fsm(rows)
    second = exp.learn_frozen_fsm(list(reversed(rows)))

    assert first["fsm_hash"] == second["fsm_hash"]
    assert first["training_family_count"] == 2
    assert "game" not in json.dumps(first["features"]).lower()
    assert first["frozen_before_held_evaluation"] is True


def test_scenario_6656_artifact_rows_attacks_and_no_solve():
    """SCENARIO-ARC-WMTE-6656-SAFETY-AND-PROGRESS-ROWS and ATTACKS."""

    artifact = exp.build_artifact(write=False, duration_s=0.001)

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["held_family_manifest"]["held_family_count"] >= 3
    assert artifact["held_family_manifest"]["seed_count"] >= 3
    assert artifact["held_family_manifest"]["train_held_disjoint"] is True
    assert artifact["frozen_fsm"]["frozen_before_held_evaluation"] is True
    assert any(row["arm"] == "on" and row["fired"] for row in artifact["paired_live_rows"])
    assert artifact["aggregate_row_recomputation"] == exp.recompute_aggregates(
        artifact["paired_live_rows"], artifact["attack_rows"]
    )
    assert artifact["no_solve_and_no_source_receipt"]["claimed_game_or_level_solve"] is False
    assert artifact["verifier_is_oracle"] is False
    assert exp.validate_artifact(artifact) == []


def test_scenario_6656_validator_catches_mutation_and_cli_writes(tmp_path):
    """SCENARIO-ARC-WMTE-6656-ARTIFACT-AND-CLI."""

    output = tmp_path / "result.json"
    assert exp.main(["--date", "20260827", "--result-path", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exp.validate_artifact(payload) == []
    assert exp.main(["--validate", "--result-path", str(output)]) == 0

    payload["arc_generalization_slot_complete_score"] = 1.0
    issues = exp.validate_artifact(payload)
    assert "reproducibility checksum mismatch" in issues
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert exp.main(["--validate", "--result-path", str(output)]) == 1


def test_scenario_6656_validator_rejects_each_structural_gate():
    """REQ-ARC-WMTE-6656 validates every row, freeze, and no-solve gate."""

    artifact = exp.build_artifact(write=False, duration_s=0.001)
    mutations = [
        (lambda row: row.pop("status"), "required fields mismatch"),
        (lambda row: row.__setitem__("status", "running"), "status lacks terminal prefix"),
        (lambda row: row.__setitem__("verdict_class", "maybe"), "verdict_class invalid"),
        (
            lambda row: row.__setitem__("inference_substrate", "wrong"),
            "inference substrate mismatch",
        ),
        (
            lambda row: row.__setitem__("verifier_is_oracle", True),
            "verifier_is_oracle must be false",
        ),
        (
            lambda row: row["frozen_fsm"].__setitem__("fsm_hash", "wrong"),
            "frozen fsm hash mismatch",
        ),
        (
            lambda row: row["held_family_manifest"].__setitem__("held_family_count", 1),
            "held manifest too small",
        ),
        (
            lambda row: row["held_family_manifest"].__setitem__("train_held_disjoint", False),
            "train and held families overlap",
        ),
        (lambda row: row["attack_rows"].pop(), "attack rows mismatch"),
        (
            lambda row: row["aggregate_row_recomputation"].__setitem__("on_firing_count", -1),
            "aggregate recomputation mismatch",
        ),
        (
            lambda row: row["no_solve_and_no_source_receipt"].__setitem__(
                "claimed_game_or_level_solve", True
            ),
            "solve claim present",
        ),
        (
            lambda row: row["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
            "protected file changed",
        ),
    ]
    for mutate, expected in mutations:
        changed = copy.deepcopy(artifact)
        mutate(changed)
        assert expected in exp.validate_artifact(changed)


def test_scenario_6656_missing_helpers_and_atomic_path_branches(tmp_path, monkeypatch):
    """SCENARIO-ARC-WMTE-6656-ARTIFACT-AND-CLI covers blocked input paths."""

    missing = tmp_path / "missing.json"
    assert exp.sha256_file(missing) == "missing"
    assert exp._load_json(missing) == {}

    relative = exp._write_artifact_json(Path("relative.json"), {"ok": True}, tmp_path)
    inside = exp._write_artifact_json(tmp_path / "results" / "inside.json", {"ok": True}, tmp_path)
    assert relative.is_file()
    assert inside.is_file()

    artifact = exp.build_artifact(write=False, duration_s=0.001)
    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp, "_load_json", lambda path: artifact)
    assert exp.main(["--validate", "--result-path", "relative.json"]) == 0

    written = []
    monkeypatch.setattr(
        exp, "_write_artifact_json", lambda path, payload, root: written.append(path)
    )
    exp.build_artifact(result_path="results/not-written.json", write=True, duration_s=0.001)
    assert written and written[0].is_absolute()
