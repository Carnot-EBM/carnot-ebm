"""Tests for Exp 4567 `.421` archive / `.422` activation.

Spec refs: REQ-CAPSTONE-4567, SCENARIO-CAPSTONE-4567,
SCENARIO-CAPSTONE-4567-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4567_archive_421_activate_422 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _green_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=0,
        stdout="green",
        stderr="",
    )


def _capstone() -> JsonDict:
    return {
        "honest_verdict": "complete: verifier_router_null_reinduction_retired_or_refined",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "efficiency_moved": False,
        "generic_transfer_moved": {
            "baseline": 0.04,
            "coheadline_rate": 0.04,
            "generic_transfer_ci": [0.0, 0.1],
            "moved": False,
            "reason": "no_clean_verifier_router_value_added_above_0.04",
        },
        "reinduction_retired": True,
        "reproducible_total_levels": 52,
        "reproducible_total_levels_delta": {
            "a3_new_levels_banked": 0,
            "a4_new_levels_banked": 0,
            "a6_new_levels_banked": 0,
            "capability_grew": False,
            "current_total": 52,
            "delta": 0,
            "prior_total": 52,
        },
        "generic_transfer_rate_over_variants": 0.04,
        "generic_transfer_ci": [0.0, 0.1],
        "verifier_router_value_added": {
            "generic_transfer_ci": [None, None],
            "generic_transfer_delta": None,
            "generic_transfer_rate_baseline": 0.04,
            "generic_transfer_rate_with_verifier": None,
            "headline_numbers_aggregated": False,
            "random_router_control_passed": False,
            "solve_rate_preserved": False,
            "status": "false_negative_risk_open",
            "value_added": False,
            "verifier_is_oracle": None,
        },
        "executable_proposer_positive_control": {
            "barrier_refinement": "positive_control_failed: executable proposer gate failed.",
            "core_efficiency_baseline": 2.0074,
            "core_efficiency_best": None,
            "efficiency_claim_valid": False,
            "false_negative_risk_checked": False,
            "false_negative_risk_open": True,
            "positive_control_passed": False,
            "status": "false_negative_risk_open",
        },
        "flagged_artifacts_handled": {
            "excluded": [
                {
                    "artifact_key": "A1_verifier_router",
                    "path": "results/experiment_4556_verifier_router_generic_transfer.json",
                    "reason": "false_negative_risk_open",
                    "stamped_flagged_adversarial": True,
                },
                {
                    "artifact_key": "A5_integration",
                    "path": "results/experiment_4560_integration_8game_gate.json",
                    "reason": "false_negative_risk_open",
                    "stamped_flagged_adversarial": True,
                },
            ],
            "positive_control_failed_or_false_negative_risk_open": [
                {
                    "artifact_key": "A2_executable_proposer",
                    "positive_control_passed": False,
                    "false_negative_risk_checked": False,
                    "reason": "false_negative_risk_open",
                }
            ],
        },
        "scorecard": {
            "a1_verifier_router": {
                "generic_transfer_delta": None,
                "generic_transfer_rate_baseline": 0.04,
                "random_router_control_passed": False,
                "status": "false_negative_risk_open",
                "value_added": False,
            },
            "a2_executable_proposer": {
                "core_efficiency_baseline": 2.0074,
                "core_efficiency_best": None,
                "false_negative_risk_checked": False,
                "false_negative_risk_open": True,
                "positive_control_passed": False,
                "status": "false_negative_risk_open",
            },
            "a3_levelup_attempt": {
                "banked_levels": 0,
                "offline_reproduced": True,
                "status": "no_new_level_banked",
                "target_game": "m0r0",
                "target_level": 2,
            },
            "a4_hidden_state_probe": {
                "banked_levels": 0,
                "offline_reproduced": False,
                "status": "no_new_level_banked",
                "target_game": "",
                "target_level": None,
            },
            "a5_integration": {
                "core_efficiency_integrated": None,
                "core_solves_preserved": False,
                "generic_transfer_rate_integrated": None,
                "integrated_metric_improved": False,
                "status": "false_negative_risk_open",
            },
            "a6_transfer": {
                "any_transfer_value_added": False,
                "new_levels_banked": 0,
                "offline_reproduced_new_level": False,
                "status": "transfer_null",
                "transfer_games": ["tu93", "tr87", "sc25"],
                "transfer_value_per_game": {
                    "tu93": {"ordering_gain": 0, "value_added": False},
                    "tr87": {"ordering_gain": 0, "value_added": False},
                    "sc25": {"ordering_gain": 0, "value_added": False},
                },
            },
            "b1_generic_transfer_coheadline": {
                "generic_transfer_ci": [0.0, 0.1],
                "generic_transfer_rate_over_variants": 0.04,
                "reproducible_total_levels": 52,
                "status": "clean_generic_transfer_coheadline",
            },
            "baseline_core_efficiency": 2.0074,
            "baseline_generic_transfer": 0.04,
        },
    }


def _a1_verifier_router() -> JsonDict:
    return {
        "honest_verdict": "complete: verifier_router_no_value_added_honest_null_gap_sharpened",
        "flagged_adversarial": True,
        "generic_transfer_delta": 0.0,
        "generic_transfer_ci": [0.0, 0.0],
        "random_router_control_passed": False,
        "false_negative_risk_checked": False,
        "offline_reproduced": True,
    }


def _a5_integration() -> JsonDict:
    return {
        "honest_verdict": "complete: no_lever_raises_a_metric_honest_null",
        "flagged_adversarial": True,
        "false_negative_risk_checked": True,
        "core_efficiency_integrated": 2.0074,
        "adversarial_flags": [
            {
                "kind": "DURATION_TOO_SHORT",
                "detail": "duration_s=36.66 but artifact references compute-bound markers.",
            }
        ],
    }


def _a6_transfer() -> JsonDict:
    transfer_results = []
    for game in ("tu93", "tr87", "sc25"):
        transfer_results.append(
            {
                "game": game,
                "offline_reproduced_new_level": False,
                "value_added": False,
                "dead_end": "no cached candidate reached the offline reproduction gate",
                "ranking": {
                    "ordering_gain": 0,
                    "incoming_candidates": [
                        {"candidate_id": "baseline", "reaches_goal": False, "target": False},
                        {"candidate_id": "verifier", "reaches_goal": False, "target": False},
                        {"candidate_id": "random", "reaches_goal": False, "target": False},
                    ],
                },
                "transfer_value": {
                    "candidate_count": 3,
                    "ordering_gain": 0,
                    "target_rank_after": None,
                    "target_rank_before": None,
                    "value_added": False,
                },
            }
        )
    return {
        "honest_verdict": "complete: primitive_persisted_transfer_null_characterized",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "primitive_persisted": True,
        "registry_updated": True,
        "transfer_results": transfer_results,
        "transfer_dead_ends": {
            "tu93": "no cached candidate reached the offline reproduction gate",
            "tr87": "no cached candidate reached the offline reproduction gate",
            "sc25": "no cached candidate reached the offline reproduction gate",
        },
    }


def _write_repo_fixture(root: Path) -> None:
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.422\n"
        "tasks:\n"
        "  - id: exp4567-phase0\n"
        "    deliverable: results/experiment_4567_archive_421_activate_422.json\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.421\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-21'\n"
        "reproducible_total_levels: 52\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4566_capstone_v421.json", _capstone())
    _write_json(
        root / "results" / "experiment_4556_verifier_router_generic_transfer.json",
        _a1_verifier_router(),
    )
    _write_json(root / "results" / "experiment_4560_integration_8game_gate.json", _a5_integration())
    _write_json(
        root / "results" / "experiment_4561_primitive_persist_transfer.json",
        _a6_transfer(),
    )


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4567_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4567: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4567" in spec
    assert "SCENARIO-CAPSTONE-4567" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "reproducible_total_levels=52" in spec
    assert "winner-not-in-pool" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4567_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4567: already-activated `.422` still writes `.421` close-state."""

    _write_repo_fixture(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    written = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["honest_verdict"] == "complete: archive_421_activate_422_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.421",
        "activated_milestone": "2026.06.422",
        "active_milestone_confirmed": True,
        "activation_state": "already_active_roadmap_next_consumed",
        "archive_state": "research_complete_contains_2026.06.421",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.422"
    assert artifact["preconditions_checked"]["offline_arcade"]["available"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_421"]
    assert close["reproducible_total_levels"] == 52
    assert close["efficiency_moved"] is False
    assert close["generic_transfer_moved"] is False
    assert close["core_efficiency_plateau"] == {
        "core_efficiency": 2.0074,
        "milestones": ["2026.06.418", "2026.06.419", "2026.06.420", "2026.06.421"],
        "milestone_count": 4,
    }
    assert close["a1_verifier_router"]["generic_transfer_delta"] == 0.0
    assert close["a1_verifier_router"]["random_router_control_passed"] is False
    assert close["a1_verifier_router"]["flagged_and_excluded"] is True
    assert close["a2_reinduction"]["positive_control_passed"] is False
    assert close["a2_reinduction"]["reinduction_retired"] is True
    assert close["a3_levelup_attempt"]["target_game"] == "m0r0"
    assert close["a3_levelup_attempt"]["target_level"] == 2
    assert close["a3_levelup_attempt"]["new_levels_banked"] == 0
    assert close["a3_levelup_attempt"]["already_banked"] is True
    assert close["a4_hidden_state_probe"]["new_levels_banked"] == 0
    assert close["a5_integration"]["duration_too_short_flagged"] is True
    assert close["a5_integration"]["flagged_and_excluded"] is True
    assert close["a6_transfer"]["root_cause"] == "winning_candidate_never_in_pool"
    assert close["a6_transfer"]["ordering_gain"] == 0
    assert close["a6_transfer"]["candidate_pool_contains_winner"] is False
    assert close["a6_transfer"]["new_levels_banked"] == 0
    assert close["b1_generic_transfer_coheadline"]["generic_transfer_rate_over_variants"] == 0.04
    assert close["b1_generic_transfer_coheadline"]["generic_transfer_ci"] == [0.0, 0.1]
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4567_blocks_without_fabricating_missing_capstone(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4567: missing required close-state input blocks honestly."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "results" / "experiment_4566_capstone_v421.json").unlink()

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_missing_experiment_4566_capstone_v421"
    assert artifact["preconditions_checked"]["capstone_4566"]["available"] is False
    assert artifact["close_state_421"] == {}
    assert artifact["transition"]["active_milestone_confirmed"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4567_records_next_roadmap_activation_state(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4567: an extant next roadmap is recorded as activation input."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.422\ntasks: []\n",
        encoding="utf-8",
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["parses"] is True


def test_scenario_capstone_4567_precondition_blockers_are_classified(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4567: each required precondition has an honest blocked reason."""

    preconditions = _artifact(tmp_path)["preconditions_checked"]

    active_bad = copy.deepcopy(preconditions)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.421"
    active_bad["research_roadmap_next_yaml"]["available"] = False
    active_bad["research_roadmap_next_yaml"]["parses"] = False
    assert mod._first_blocker(active_bad) == "research_roadmap_422_unavailable"

    next_ok = copy.deepcopy(active_bad)
    next_ok["research_roadmap_next_yaml"]["parses"] = True
    next_ok["research_roadmap_next_yaml"]["milestone"] = "2026.06.422"
    assert mod._first_blocker(next_ok) is None

    offline_bad = copy.deepcopy(preconditions)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    smart_bad = copy.deepcopy(preconditions)
    smart_bad["smart_subset_pretest_gate"]["passed"] = False
    assert mod._first_blocker(smart_bad) == "smart_subset_pretest_gate"

    registry_bad = copy.deepcopy(preconditions)
    registry_bad["registry"]["available"] = False
    assert mod._first_blocker(registry_bad) == "arc_solve_registry"

    capstone_bad = copy.deepcopy(preconditions)
    capstone_bad["capstone_4566"]["available"] = False
    assert mod._first_blocker(capstone_bad) == "missing_experiment_4566_capstone_v421"

    a1_bad = copy.deepcopy(preconditions)
    a1_bad["a1_verifier_router"]["available"] = False
    assert mod._first_blocker(a1_bad) == "missing_experiment_4556_verifier_router_generic_transfer"

    a5_bad = copy.deepcopy(preconditions)
    a5_bad["a5_integration"]["available"] = False
    assert mod._first_blocker(a5_bad) == "missing_experiment_4560_integration_8game_gate"

    a6_bad = copy.deepcopy(preconditions)
    a6_bad["a6_transfer"]["available"] = False
    assert mod._first_blocker(a6_bad) == "missing_experiment_4561_primitive_persist_transfer"


def test_scenario_capstone_4567_parse_helpers_are_defensive(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4567: malformed inputs are detected instead of fabricated."""

    assert mod._list(None) == []
    assert mod._float(True, 7.0) == 7.0
    assert mod._float("bad", 9.0) == 9.0
    assert mod._int(False, 2) == 2
    assert mod._int("bad", 3) == 3
    assert mod._registry_total_levels(tmp_path / "missing.yaml") is None
    assert mod._contains_flag_kind({"adversarial_flags": [{"kind": "DURATION_TOO_SHORT"}]}, "DURATION_TOO_SHORT")
    assert mod._contains_flag_kind({"corrigendum_pending": [{"kind": "DURATION_TOO_SHORT"}]}, "DURATION_TOO_SHORT")
    assert not mod._contains_flag_kind({"adversarial_flags": [{"kind": "OTHER"}]}, "DURATION_TOO_SHORT")
    assert mod._moved(True) is True
    assert mod._moved(False) is False

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("milestone: [\n", encoding="utf-8")
    assert mod._yaml_info(bad_yaml)["parses"] is False

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not-a-map\n", encoding="utf-8")
    assert mod._registry_total_levels(list_yaml) is None

    list_json = tmp_path / "list.json"
    list_json.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(list_json)

    assert mod._candidate_pool_contains_winner([]) is False
    assert mod._candidate_pool_contains_winner([{"offline_reproduced_new_level": True}])
    assert mod._candidate_pool_contains_winner([{"ranking": {"incoming_candidates": [{"reaches_goal": True}]}}])


def test_scenario_capstone_4567_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4567-FIELD-PRINCIPLES: schema drift fails loudly."""

    valid = _artifact(tmp_path)

    missing = copy.deepcopy(valid)
    del missing["honest_verdict"]
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_verdict = copy.deepcopy(valid)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = copy.deepcopy(valid)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_provenance = copy.deepcopy(valid)
    bad_provenance["field_provenance"] = {}
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)

    blocked = mod._blocked_artifact(
        reason="unit_test",
        preconditions_checked=valid["preconditions_checked"],
        duration_s=0.1,
        cited_upstream_artifacts=valid["cited_upstream_artifacts"],
    )
    blocked["close_state_421"] = {"fabricated": True}
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .422"):
        mod.validate_artifact(inactive)

    wrong_total = copy.deepcopy(valid)
    wrong_total["close_state_421"]["reproducible_total_levels"] = 51
    with pytest.raises(ValueError, match="true .421 close-state"):
        mod.validate_artifact(wrong_total)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_421"]["a1_verifier_router"]["generic_transfer_delta"] = None
    with pytest.raises(ValueError, match="A1 verifier-router null"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_421"]["a2_reinduction"]["reinduction_retired"] = False
    with pytest.raises(ValueError, match="A2 re-induction retirement"):
        mod.validate_artifact(wrong_a2)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_421"]["a3_levelup_attempt"]["target_game"] = "sp80"
    with pytest.raises(ValueError, match="A3 m0r0 L2"):
        mod.validate_artifact(wrong_a3)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_421"]["a4_hidden_state_probe"]["new_levels_banked"] = 1
    with pytest.raises(ValueError, match="A4 zero bank"):
        mod.validate_artifact(wrong_a4)

    wrong_a5 = copy.deepcopy(valid)
    wrong_a5["close_state_421"]["a5_integration"]["duration_too_short_flagged"] = False
    with pytest.raises(ValueError, match="A5 DURATION_TOO_SHORT"):
        mod.validate_artifact(wrong_a5)

    wrong_a6 = copy.deepcopy(valid)
    wrong_a6["close_state_421"]["a6_transfer"]["root_cause"] = "unknown"
    with pytest.raises(ValueError, match="A6 winner-not-in-pool"):
        mod.validate_artifact(wrong_a6)

    wrong_b1 = copy.deepcopy(valid)
    wrong_b1["close_state_421"]["b1_generic_transfer_coheadline"][
        "generic_transfer_rate_over_variants"
    ] = 0.05
    with pytest.raises(ValueError, match="B1 generic transfer"):
        mod.validate_artifact(wrong_b1)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum_value = copy.deepcopy(valid)
    bad_checksum_value["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum_value)
