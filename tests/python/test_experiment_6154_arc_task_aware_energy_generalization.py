"""Tests for Exp6154 ARC task-aware energy generalization.

Spec refs: REQ-ARC-WMTE-6154,
SCENARIO-ARC-WMTE-6154-LIVE-ENTRYPOINT-AND-PROVENANCE,
SCENARIO-ARC-WMTE-6154-TRAINING-HELD-ISOLATION,
SCENARIO-ARC-WMTE-6154-METRICS-CONTROLS-AND-NO-SOLVE.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6154_arc_task_aware_energy_generalization as mod
from carnot.agentic import arc_task_aware_energy as energy


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
ARC_SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game, changes in {
        "r11l": [0, 3, 0, 2],
        "ls20": [1, 0, 4, 0],
        "lp85": [0, 2, 0, 5],
    }.items():
        for index, changed_cells in enumerate(changes):
            rows.append(
                {
                    "row_id": f"{game}:{index}",
                    "game": game,
                    "held_game": game,
                    "seed": 6154,
                    "action_index": index,
                    "action_id": 6,
                    "changed_cell_count": changed_cells,
                    "frame_changed": changed_cells > 0,
                    "safety_event": "none",
                    "latency_ms": 0.1 + index,
                    "level_before": 0,
                    "level_after": 0,
                    "reward_delta": 0.0,
                    "source": "live_agent_runtime_action",
                }
            )
    return rows


def test_req_6154_spec_declares_live_adapter_disabled_contract() -> None:
    """REQ-ARC-WMTE-6154: OpenSpec names the artifact fields and principles."""

    text = ARC_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-ARC-WMTE-6154") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-ARC-WMTE-6154",
        "SCENARIO-ARC-WMTE-6154-LIVE-ENTRYPOINT-AND-PROVENANCE",
        "SCENARIO-ARC-WMTE-6154-TRAINING-HELD-ISOLATION",
        "SCENARIO-ARC-WMTE-6154-METRICS-CONTROLS-AND-NO-SOLVE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6154_task_aware_freeze_excludes_held_game() -> None:
    """SCENARIO-ARC-WMTE-6154-TRAINING-HELD-ISOLATION: folds use train games only."""

    rows = _synthetic_rows()
    manifest = energy.fit_task_aware_calibration(rows, held_game="lp85")

    assert manifest["held_game"] == "lp85"
    assert manifest["training_games"] == ["ls20", "r11l"]
    assert manifest["held_row_count_used_for_fit"] == 0
    assert manifest["hand_calibrated_per_game"] is False
    assert manifest["min_changed_cells"] == 1
    assert manifest["manifest_hash"] == energy.manifest_hash(manifest)

    no_op = next(row for row in rows if row["game"] == "lp85" and row["changed_cell_count"] == 0)
    changed = next(row for row in rows if row["game"] == "lp85" and row["changed_cell_count"] > 0)
    global_noop = energy.score_transition(no_op, energy.global_freeze_manifest(), arm="global")
    aware_noop = energy.score_transition(no_op, manifest, arm="task_aware")
    aware_changed = energy.score_transition(changed, manifest, arm="task_aware")

    assert global_noop["triggered"] is True
    assert global_noop["admitted"] is True
    assert aware_noop["triggered"] is True
    assert aware_noop["abstained"] is True
    assert aware_changed["admitted"] is True


def test_req_6154_metrics_controls_and_validation(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6154-METRICS-CONTROLS-AND-NO-SOLVE: gates fail closed."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        games=("lp85", "su15", "tu93"),
        held_games=("lp85", "su15", "tu93"),
        action_budget=8,
        seeds=(6154,),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["arc_task_aware_generalization_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["llm_invocation_count"] == 0
    assert artifact["used_game_source"] is False
    assert artifact["offline_ground_truth_bfs"] is False
    assert artifact["hand_calibrated_per_game"] is False
    assert artifact["solve_claimed"] is False
    assert artifact["offline_reproduced"] is False
    assert artifact["level_credit_delta"] == 0
    assert artifact["registry_level_fields_unchanged"]["unchanged"] is True
    assert artifact["per_arm_triggered_decision_counts"]["global"] > 0
    assert artifact["per_arm_triggered_decision_counts"]["task_aware"] > 0
    assert (
        artifact[
            "shuffled_label_alias_identity_noop_light_inventor_raise_denominator_and_no_trigger_controls"
        ]["all_controls_passed"]
        is True
    )
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    bad_no_trigger = deepcopy(artifact)
    bad_no_trigger["per_arm_triggered_decision_counts"]["task_aware"] = 0
    bad_no_trigger["arc_task_aware_generalization_ready_score"] = mod.ready_score(bad_no_trigger)
    bad_no_trigger["status"] = mod.status(bad_no_trigger)
    bad_no_trigger["honest_verdict"] = mod.honest_verdict(bad_no_trigger)
    bad_no_trigger["reproducibility_checksum"] = mod.reproducibility_checksum(bad_no_trigger)
    assert bad_no_trigger["arc_task_aware_generalization_ready_score"] == 0.0
    with pytest.raises(ValueError, match="triggered"):
        mod.validate_artifact(bad_no_trigger)

    bad_solve = deepcopy(artifact)
    bad_solve["solve_claimed"] = True
    bad_solve["reproducibility_checksum"] = mod.reproducibility_checksum(bad_solve)
    with pytest.raises(ValueError, match="solve_claimed"):
        mod.validate_artifact(bad_solve)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_json({"wrong": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_scenario_6154_real_live_path_rows_and_import_reachability(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6154-LIVE-ENTRYPOINT-AND-PROVENANCE: real E3 rows trigger."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        games=("lp85", "su15", "tu93"),
        held_games=("lp85", "su15", "tu93"),
        action_budget=8,
        seeds=(6154,),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    assert artifact["status"] == "complete_positive"
    assert artifact["live_entrypoint_and_import_reachability"]["make_carnot_agent_constructed"] is True
    assert artifact["live_entrypoint_and_import_reachability"]["e3_policy_seen"] is True
    assert artifact["live_entrypoint_and_import_reachability"]["calibration_module_in_live_import_closure"] is True
    assert artifact["own_attempt_transition_provenance"]["all_rows_live_agent_owned"] is True
    assert artifact["own_attempt_transition_provenance"]["scored_row_count"] >= 24
    assert set(artifact["per_game_transition_change_safety_action_and_latency_metrics"]) == {
        "lp85",
        "su15",
        "tu93",
    }
    assert mod.validate_artifact(artifact) is True


def test_req_6154_adversarial_verify_classifies_exact_substrate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6154: the exact no-LLM substrate is allowlisted."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        games=("lp85", "su15", "tu93"),
        held_games=("lp85", "su15", "tu93"),
        action_budget=8,
        seeds=(6154,),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )
    report = adversarial_verify.verify_artifact(tmp_path / mod.RESULT_RELATIVE_PATH.name)
    kinds = {flag["kind"] for flag in report["flags"]}

    assert adversarial_verify._classify_inference_substrate(artifact)["kind"] == "no_llm"
    assert "DURATION_TOO_SHORT" not in kinds
    assert "IMPLAUSIBLE_PERFECT" not in kinds
    assert "METHODOLOGY_MISSING" not in kinds


def test_req_6154_helper_boundaries_and_blocked_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6154: defensive helpers and null gates are explicit."""

    rows = _synthetic_rows()
    manifest = energy.fit_task_aware_calibration(rows, held_game="lp85")
    with pytest.raises(ValueError, match="unknown arm"):
        energy.score_transition(rows[0], manifest, arm="bad")

    proposer = mod._NoLLMProposer()
    with pytest.raises(RuntimeError, match="disables LLM"):
        proposer.generate("x")
    assert proposer.calls == 1
    base = mod._BaseAgent("lp85")
    base.cleanup()
    assert base._cleanup is True

    assert mod._grid_of_frame(np.asarray([[1, 2]])).shape == (1, 2)
    assert mod._action_id(SimpleNamespace(name="RESET")) is None
    assert mod._action_data_dict(None) == {}
    assert mod._action_data_dict({"x": 1}) == {"x": 1}
    assert mod._action_data_dict(SimpleNamespace(model_dump=lambda: {"y": 2})) == {"y": 2}
    assert mod._action_data_dict(SimpleNamespace(game_id="g", x=3)) == {"game_id": "g", "x": 3}
    assert mod._available_action_ids(
        SimpleNamespace(available_actions=[1, SimpleNamespace(name="ACTION6")])
    ) == {1, 6}
    assert mod._changed_cell_count(np.zeros((1, 1)), np.zeros((2, 1))) == 3
    monkeypatch.setenv("CARNOT_ARC_ACTIVE_PROBE", "existing")
    with mod._adapter_disabled_live_context() as receipt:
        assert receipt["adapter_disabled"] is True
    assert os.environ["CARNOT_ARC_ACTIVE_PROBE"] == "existing"

    fixture_artifact = mod.run(
        result_path=tmp_path / "fixture.json",
        live_rows=rows,
        games=("r11l", "ls20", "lp85"),
        held_games=("lp85",),
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.1,
        write=False,
    )
    assert fixture_artifact["status"] == "complete_null"
    assert "own_attempt_transition_provenance" in mod._blocked_reasons(fixture_artifact)

    bad = deepcopy(fixture_artifact)
    bad["preconditions_checked"]["root_clutter"]["ok"] = False
    bad["registry_precheck_and_no_duplicate_receipt"]["ok"] = False
    bad["live_entrypoint_and_import_reachability"][
        "calibration_module_in_live_import_closure"
    ] = False
    bad["per_arm_triggered_decision_counts"]["global"] = 0
    bad["grouped_paired_intervals"]["support"]["positive_game_grouped_support"] = False
    bad["false_confident_admission_and_abstention_matrices"][
        "task_aware_reduces_or_preserves_false_confident"
    ] = False
    bad[
        "shuffled_label_alias_identity_noop_light_inventor_raise_denominator_and_no_trigger_controls"
    ]["all_controls_passed"] = False
    bad["llm_invocation_count"] = 1
    bad["used_game_source"] = True
    bad["offline_ground_truth_bfs"] = True
    bad["hand_calibrated_per_game"] = True
    bad["solve_claimed"] = True
    bad["offline_reproduced"] = True
    bad["level_credit_delta"] = 1
    bad["registry_level_fields_unchanged"]["unchanged"] = False
    bad["protected_files_unchanged"]["unchanged"] = False
    bad["inference_substrate"] = "wrong"
    bad["verifier_is_oracle"] = True
    reasons = set(mod._blocked_reasons(bad))
    assert {
        "root_clutter",
        "registry_precheck",
        "live_import_reachability",
        "own_attempt_transition_provenance",
        "triggered_decision_counts",
        "nonpositive_task_aware_lift",
        "false_confident_regression",
        "control_failure",
        "llm_invocation_count",
        "used_game_source",
        "offline_ground_truth_bfs",
        "hand_calibrated_per_game",
        "solve_claimed",
        "offline_reproduced",
        "level_credit_delta",
        "registry_level_fields_unchanged",
        "protected_files_unchanged",
        "inference_substrate",
        "verifier_is_oracle",
    } <= reasons
    bad["retirement_triggered"] = True
    assert mod.status(bad) == "retired"
    assert mod.honest_verdict(bad).startswith("retired:")
