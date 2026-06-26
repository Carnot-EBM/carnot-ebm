"""Tests for Exp 4800 `.441` archive / `.442` activation record.

Spec refs: REQ-CAPSTONE-4800, SCENARIO-CAPSTONE-4800,
SCENARIO-CAPSTONE-4800-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4800-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4800_archive_441_activate_442 as mod


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
        stdout="101 passed, 1 warning in 6.0s",
        stderr="",
    )


def _red_poison_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 100 passed, 1 warning in 6.2s",
        stderr="test_expected_old_441_s2_bounded_headline still expects bounded S2",
    )


def _s2_4791() -> JsonDict:
    return {
        "experiment": "experiment_4791_structural_energy_s2_offpath_trust_gate",
        "experiment_id": 4791,
        "honest_verdict": "complete_structural_energy_s2_no_live_trust_value",
        "flagged_adversarial": True,
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "energy_minus_accuracy_delta": 0.0,
        "energy_minus_accuracy_delta_ci95": [0.0, 0.0],
        "energy_selected_offpath_cell_recall": 0.32018531293855285,
        "accuracy_gate_selected_offpath_cell_recall": 0.32018531293855285,
        "n_heldout_games": 5,
        "min_heldout_games": 5,
        "s3_authorized": False,
        "game_results": [
            {
                "game": "ar25",
                "candidate_rows": [
                    {
                        "candidate_name": "results/arc_e3/ar25/world_model.py",
                        "heldout_cell_recall": 1.0,
                        "offpath_structural_energy": 189.14479776397968,
                    },
                    {
                        "candidate_name": "results/arc_logo_snapshot/ar25/world_model.py",
                        "heldout_cell_recall": 1.0,
                        "offpath_structural_energy": 189.14479776397968,
                    },
                ],
            },
            {
                "game": "cd82",
                "candidate_rows": [
                    {
                        "candidate_name": "results/arc_e3/cd82/world_model.best.py",
                        "heldout_cell_recall": 0.0,
                        "offpath_structural_energy": 1000000.0,
                    },
                    {
                        "candidate_name": "results/arc_e3/cd82/world_model.py",
                        "heldout_cell_recall": 0.0,
                        "offpath_structural_energy": 497.7490630125705,
                    },
                ],
            },
            {
                "game": "cn04",
                "candidate_rows": [
                    {
                        "candidate_name": "results/arc_e3/cn04/world_model.py",
                        "heldout_cell_recall": 0.04477611940298507,
                        "offpath_structural_energy": 131.26142683026688,
                    },
                    {
                        "candidate_name": "results/arc_logo_snapshot/cn04/world_model.py",
                        "heldout_cell_recall": 0.04477611940298507,
                        "offpath_structural_energy": 131.26142683026688,
                    },
                    {
                        "candidate_name": "results/arc_e3_seedproto/cn04/world_model.py",
                        "heldout_cell_recall": 0.05099502487562189,
                        "offpath_structural_energy": 134.8143870447679,
                    },
                ],
            },
            {
                "game": "ka59",
                "candidate_rows": [
                    {
                        "candidate_name": "results/arc_e3/ka59/world_model.py",
                        "heldout_cell_recall": 0.864406779661017,
                        "offpath_structural_energy": 310.6825626327457,
                    },
                    {
                        "candidate_name": "results/arc_logo_snapshot/ka59/world_model.py",
                        "heldout_cell_recall": 0.864406779661017,
                        "offpath_structural_energy": 310.6825626327457,
                    },
                ],
            },
            {
                "game": "sc25",
                "candidate_rows": [
                    {
                        "candidate_name": "results/arc_e3/sc25/world_model.py",
                        "heldout_cell_recall": 0.12162162162162163,
                        "offpath_structural_energy": 125.34472928121748,
                    },
                    {
                        "candidate_name": "results/arc_logo_snapshot/sc25/world_model.py",
                        "heldout_cell_recall": 0.0,
                        "offpath_structural_energy": 1000000.0,
                    },
                ],
            },
        ],
    }


def _capstone_4799() -> JsonDict:
    return {
        "experiment": "experiment_4799_capstone_v441",
        "experiment_id": 4799,
        "capstone_ready": True,
        "honest_verdict": "complete_s2_structural_energy_bounded_v441_capstone_ready",
        "reproducible_total_levels": 65,
        "s2_structural_energy_verdict": {
            "verdict": "bounded",
            "artifact_skipped": True,
            "metrics_imported": False,
            "reason": "s2_artifact_skipped_live_or_genuine_flag",
            "reported_energy_minus_accuracy_delta": 0.0,
            "reported_energy_minus_accuracy_delta_ci95": [0.0, 0.0],
            "s3_authorized": False,
            "upstream_honest_verdict": "complete_structural_energy_s2_no_live_trust_value",
            "verifier_is_oracle": False,
        },
        "readiness": {"ready_for_operator_submit": False, "s2_verdict": "bounded"},
        "levelup_bank": {"new_levels_banked": 0, "reproducible_total_levels_after": 65},
        "flagged_artifacts_skipped": [
            {
                "experiment_id": 4791,
                "source": "S2",
                "reason": "live_critical_recheck",
            }
        ],
        "sota_handoff": {
            "flagged_for_v442": [{"candidate": "cold_cfg_value_tree_generator_for_s3"}]
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.442",
    next_present: bool = False,
    registry_total: int = 65,
    capstone_present: bool = True,
    s2_present: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4800-phase0\n"
        "    deliverable: results/experiment_4800_archive_441_activate_442.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.442\n"
            "tasks:\n"
            "  - id: exp4800-phase0\n"
            "    deliverable: results/experiment_4800_archive_441_activate_442.json\n"
            "  - id: exp4801-a1\n"
            "    deliverable: results/experiment_4801_structural_energy_s2v2_diverse_trust_gate.json\n",
            encoding="utf-8",
        )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-26'\n"
        f"reproducible_total_levels: {registry_total}\n",
        encoding="utf-8",
    )
    spec = root / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("REQ-CAPSTONE-4800\n", encoding="utf-8")
    if capstone_present:
        _write_json(root / "results" / "experiment_4799_capstone_v441.json", _capstone_4799())
    if s2_present:
        _write_json(root / "results" / "experiment_4791_structural_energy_s2_offpath_trust_gate.json", _s2_4791())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4800_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4800: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4800" in spec
    assert "SCENARIO-CAPSTONE-4800" in spec
    assert "SCENARIO-CAPSTONE-4800-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4800-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4800_records_s2_inconclusive_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4800: active `.442` records S2 as inconclusive."""

    _write_repo_fixture(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=2.0,
        now_s=2.3,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert artifact["honest_verdict"] == (
        "complete_441_archived_442_activated_already_active_s2_inconclusive_recorded"
    )
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.441",
        "activated_milestone": "2026.06.442",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "archive_noop_or_already_recorded",
    }
    assert artifact["s2_recorded_as_inconclusive"] is True
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"] == {
        "path": "research-roadmap-next.yaml",
        "available": False,
        "parses": False,
        "milestone": None,
        "literal_precondition_command": (
            ".venv/bin/python -c \"import yaml; yaml.safe_load(open("
            "'research-roadmap-next.yaml')); print('ok')\""
        ),
        "literal_precondition_passed": False,
        "activation_attempted": False,
        "activation_error": "",
        "accepted_missing_because_already_active": True,
    }
    assert artifact["poison_test_resolved"] == {
        "resolved": True,
        "current_gate_passed": True,
        "poison_tests": [],
        "action": "no_poison_observed_current_gate_green",
    }
    close = artifact["close_state_441"]
    assert close["capstone_honest_verdict"] == "complete_s2_structural_energy_bounded_v441_capstone_ready"
    assert close["capstone_reported_s2_verdict"] == "bounded"
    assert close["capstone_misrecorded_bounded_null"] is True
    assert close["reproducible_total_levels"] == 65
    s2 = close["s2_corrected_record"]
    assert s2["corrected_verdict"] == "inconclusive_degenerate_pool"
    assert s2["reported_honest_verdict"] == "complete_structural_energy_s2_no_live_trust_value"
    assert s2["reported_energy_minus_accuracy_delta"] == 0.0
    assert s2["reported_energy_minus_accuracy_delta_ci95"] == [0.0, 0.0]
    assert s2["n_total_games"] == 5
    assert s2["n_effective_games"] == 2
    assert s2["effective_games"] == ["cn04", "sc25"]
    assert s2["behaviorally_identical_games"] == ["ar25", "ka59"]
    assert s2["equal_recall_non_effective_games"] == ["ar25", "cd82", "ka59"]
    assert s2["min_effective_games_required"] == 5
    assert s2["effective_game_gate_passed"] is False
    assert s2["energy_direction_state"] == "inconclusive_not_bounded_not_passed"
    assert s2["s2v2_required"] is True
    assert artifact["v442_pivot"]["task_id"] == "exp4801-a1"
    assert artifact["v442_pivot"]["enforces_behaviorally_diverse_candidate_pool"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4800_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4800: present next roadmap is activated onto active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.441", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=3.0,
        now_s=3.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.442"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["honest_verdict"] == (
        "complete_441_archived_442_activated_from_next_s2_inconclusive_recorded"
    )


def test_scenario_capstone_4800_blockers_and_poison_signature_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4800-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.441", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_442_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["s2_recorded_as_inconclusive"] is False
    assert artifact["close_state_441"] == {}
    assert artifact["v442_pivot"] == {}

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4800",
        "registry": "arc_solve_registry",
        "capstone_4799": "missing_experiment_4799_capstone_v441",
        "s2_4791": "missing_experiment_4791_structural_energy_s2_offpath_trust_gate",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4800"] = False
        assert mod._first_blocker(bad) == expected

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["reproducible_total_levels"] = 64
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_65"

    activation_bad = copy.deepcopy(checks)
    activation_bad["research_roadmap_next_yaml"]["activation_error"] = "permission denied"
    assert mod._first_blocker(activation_bad) == "research_roadmap_activation_error"

    bad_smart = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=5.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_red_poison_smart_subset,
    )
    assert bad_smart["honest_verdict"] == "blocked_smart_subset_pretest_gate"
    assert bad_smart["poison_test_resolved"]["poison_tests"] == [
        {
            "id": "test_expected_old_441_s2_bounded_headline",
            "reason": "single-failure smart-subset signature matches a stale honest-verdict expectation",
            "action": "blocked_for_fix_or_quarantine_before_tail_continues",
        }
    ]

    other_smart_failure = copy.deepcopy(checks)
    other_smart_failure["smart_subset_pretest_gate"]["passed"] = False
    other_smart_failure["smart_subset_pretest_gate"]["stdout_tail"] = "2 failed, 99 passed"
    other_smart_failure["smart_subset_pretest_gate"]["stderr_tail"] = ""
    assert mod._poison_test_resolution(other_smart_failure)["poison_tests"] == []


def test_scenario_capstone_4800_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4800-FIELD-PRINCIPLES: schema drift fails loudly."""

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

    bad_principles = copy.deepcopy(valid)
    bad_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_poison = copy.deepcopy(valid)
    bad_poison["poison_test_resolved"]["resolved"] = False
    with pytest.raises(ValueError, match="poison"):
        mod.validate_artifact(bad_poison)

    blocked = mod._blocked_artifact(
        reason="unit_test",
        preconditions_checked=valid["preconditions_checked"],
        poison_test_resolved=valid["poison_test_resolved"],
        duration_s=0.1,
        cited_upstream_artifacts=valid["cited_upstream_artifacts"],
    )
    blocked["close_state_441"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    wrong_total = copy.deepcopy(valid)
    wrong_total["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_total)

    wrong_s2_record = copy.deepcopy(valid)
    wrong_s2_record["s2_recorded_as_inconclusive"] = False
    with pytest.raises(ValueError, match="S2"):
        mod.validate_artifact(wrong_s2_record)

    wrong_corrected = copy.deepcopy(valid)
    wrong_corrected["close_state_441"]["s2_corrected_record"]["corrected_verdict"] = "bounded"
    with pytest.raises(ValueError, match="S2"):
        mod.validate_artifact(wrong_corrected)

    wrong_effective = copy.deepcopy(valid)
    wrong_effective["close_state_441"]["s2_corrected_record"]["n_effective_games"] = 5
    with pytest.raises(ValueError, match="S2"):
        mod.validate_artifact(wrong_effective)

    wrong_gate = copy.deepcopy(valid)
    wrong_gate["close_state_441"]["s2_corrected_record"]["effective_game_gate_passed"] = True
    with pytest.raises(ValueError, match="S2"):
        mod.validate_artifact(wrong_gate)

    wrong_direction = copy.deepcopy(valid)
    wrong_direction["close_state_441"]["s2_corrected_record"]["energy_direction_state"] = "bounded"
    with pytest.raises(ValueError, match="S2"):
        mod.validate_artifact(wrong_direction)

    wrong_capstone = copy.deepcopy(valid)
    wrong_capstone["close_state_441"]["capstone_misrecorded_bounded_null"] = False
    with pytest.raises(ValueError, match="S2"):
        mod.validate_artifact(wrong_capstone)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v442_pivot"]["enforces_behaviorally_diverse_candidate_pool"] = False
    with pytest.raises(ValueError, match="v442 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

    assert mod._candidate_floats([{"heldout_cell_recall": True}, {"heldout_cell_recall": "bad"}]) == []
    assert mod._candidate_floats([object(), {"heldout_cell_recall": 1.0}]) == [1.0]
    assert mod._candidate_floats([{"offpath_structural_energy": 1.2}], key="offpath_structural_energy") == [
        1.2
    ]
    assert mod._s2_inconclusive_state({"game_results": [{"game": "empty"}]})["n_effective_games"] == 0
    assert mod._activate_next_roadmap(tmp_path, next_info={"available": False}) == (False, "")

    activation_root = tmp_path / "activation_error"
    activation_root.mkdir()
    (activation_root / "research-roadmap.yaml").mkdir()
    (activation_root / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.442\n",
        encoding="utf-8",
    )
    activated, activation_error = mod._activate_next_roadmap(
        activation_root,
        next_info={"available": True, "parses": True, "milestone": "2026.06.442"},
    )
    assert activated is False
    assert activation_error

    def _offline_raises() -> bool:
        raise RuntimeError("offline arcade unavailable")

    offline_artifact = mod.build_artifact(
        tmp_path,
        started_s=6.0,
        now_s=6.1,
        offline_arcade_checker=_offline_raises,
        smart_subset_checker=_green_smart_subset,
    )
    assert offline_artifact["honest_verdict"] == "blocked_offline_arcade"
    assert offline_artifact["preconditions_checked"]["offline_arcade"]["error"] == "offline arcade unavailable"

    missing_s2_root = tmp_path / "missing_s2"
    _write_repo_fixture(missing_s2_root, s2_present=False)
    missing_s2_artifact = mod.build_artifact(
        missing_s2_root,
        started_s=7.0,
        now_s=7.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )
    assert missing_s2_artifact["honest_verdict"] == (
        "blocked_missing_experiment_4791_structural_energy_s2_offpath_trust_gate"
    )
