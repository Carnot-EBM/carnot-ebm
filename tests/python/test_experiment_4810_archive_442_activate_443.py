"""Tests for Exp 4810 `.442` archive / `.443` activation record.

Spec refs: REQ-CAPSTONE-4810, SCENARIO-CAPSTONE-4810,
SCENARIO-CAPSTONE-4810-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4810-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4810_archive_442_activate_443 as mod


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
        stdout="127 passed in 7.0s",
        stderr="",
    )


def _red_poison_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 126 passed in 7.1s",
        stderr="test_expected_old_442_s2v2_degenerate_capstone still expects inconclusive",
    )


def _s2v2_4801() -> JsonDict:
    effective_games = ["ar25", "cd82", "cn04", "ka59", "sc25"]
    diversity_rows = [
        {
            "game": game,
            "effective": True,
            "n_candidates": 3,
            "distinct_heldout_cell_recall_count": 2,
            "heldout_cell_recall_spread": 0.01,
            "candidates_genuinely_induced": True,
        }
        for game in effective_games
    ]
    diversity_rows.extend(
        [
            {
                "game": "dc22",
                "effective": False,
                "n_candidates": 1,
                "distinct_heldout_cell_recall_count": 1,
                "heldout_cell_recall_spread": 0.0,
                "candidates_genuinely_induced": True,
            },
            {
                "game": "m0r0",
                "effective": False,
                "n_candidates": 1,
                "distinct_heldout_cell_recall_count": 1,
                "heldout_cell_recall_spread": 0.0,
                "candidates_genuinely_induced": True,
            },
        ]
    )
    return {
        "experiment": "experiment_4801_structural_energy_s2v2_diverse_trust_gate",
        "experiment_id": 4801,
        "honest_verdict": "complete_structural_energy_s2v2_bounded_diverse_pool",
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "energy_selected_offpath_cell_recall": 0.21112602926401988,
        "accuracy_gate_selected_offpath_cell_recall": 0.36878379278939066,
        "energy_minus_accuracy_delta": -0.15765776352537078,
        "energy_minus_accuracy_delta_ci95": [-0.47765014592872246, 0.004044943820224719],
        "n_effective_games": 5,
        "min_heldout_games": 5,
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
        "s3_authorized": False,
        "candidate_pool_diversity": diversity_rows,
        "game_results": [{"game": game, "energy_minus_accuracy_delta": -0.1} for game in effective_games],
    }


def _capstone_4809() -> JsonDict:
    return {
        "experiment": "experiment_4809_capstone_v442",
        "experiment_id": 4809,
        "capstone_ready": True,
        "honest_verdict": "complete_s2v2_inconclusive_degenerate_pool_capstone_ready",
        "reproducible_total_levels": 65,
        "s2v2_structural_energy_verdict": {
            "verdict": "inconclusive",
            "reason": "degenerate_candidate_pool_live_check",
            "degenerate_candidate_pool_flagged": True,
            "reported_energy_minus_accuracy_delta": -0.157657763525,
            "reported_energy_minus_accuracy_delta_ci95": [
                -0.47765014592872246,
                0.004044943820224719,
            ],
            "n_effective_games": 5,
            "min_heldout_games": 5,
            "positive_control_passed": True,
            "verifier_is_oracle": False,
            "live_path_reachable": True,
            "s3_authorized": False,
            "upstream_honest_verdict": "complete_structural_energy_s2v2_bounded_diverse_pool",
        },
        "readiness": {"ready_for_operator_submit": False, "s2v2_verdict": "inconclusive"},
        "sota_handoff": {
            "flagged_for_v443_candidates": [
                "bolt_cold_cfg_value_tree_generator_for_s3",
                "bes_energy_fitness_pool_inserter",
            ]
        },
    }


def _write_environment_files(root: Path, count: int = 25) -> None:
    names = [
        "ft09",
        "sc25",
        "cd82",
        "tu93",
        "sk48",
        "dc22",
        "s5i5",
        "sb26",
        "m0r0",
        "re86",
        "cn04",
        "lf52",
        "tn36",
        "vc33",
        "wa30",
        "ka59",
        "tr87",
        "sp80",
        "lp85",
        "su15",
        "bp35",
        "ar25",
        "ls20",
        "r11l",
        "g50t",
    ][:count]
    for name in names:
        (root / "environment_files" / name).mkdir(parents=True, exist_ok=True)


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.443",
    next_present: bool = False,
    registry_total: int = 65,
    capstone_present: bool = True,
    s2v2_present: bool = True,
    environment_count: int = 25,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4810-phase0\n"
        "    deliverable: results/experiment_4810_archive_442_activate_443.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.443\n"
            "tasks:\n"
            "  - id: exp4810-phase0\n"
            "    deliverable: results/experiment_4810_archive_442_activate_443.json\n",
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
    spec.write_text("REQ-CAPSTONE-4810\n", encoding="utf-8")
    _write_environment_files(root, environment_count)
    if capstone_present:
        _write_json(root / "results" / "experiment_4809_capstone_v442.json", _capstone_4809())
    if s2v2_present:
        _write_json(
            root / "results" / "experiment_4801_structural_energy_s2v2_diverse_trust_gate.json",
            _s2v2_4801(),
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


def test_req_capstone_4810_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4810: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4810_records_under_covered_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4810: active `.443` records S2-v2 as under-covered."""

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
        "complete_442_archived_443_activated_already_active_"
        "bounded_but_under_covered_5_of_25"
    )
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.442",
        "activated_milestone": "2026.06.443",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "archive_noop_or_already_recorded",
    }
    assert artifact["s2v2_recorded_as_under_covered"] is True
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "literal_precondition_passed"
    ] is False
    assert artifact["poison_test_resolved"] == {
        "resolved": True,
        "current_gate_passed": True,
        "poison_tests": [],
        "action": "no_poison_observed_current_gate_green",
    }

    close = artifact["close_state_442"]
    assert close["capstone_honest_verdict"] == "complete_s2v2_inconclusive_degenerate_pool_capstone_ready"
    assert close["capstone_reported_s2v2_verdict"] == "inconclusive"
    assert close["capstone_recorded_degenerate_candidate_pool"] is True
    assert close["reproducible_total_levels"] == 65

    s2 = close["s2v2_corrected_record"]
    assert s2["corrected_verdict"] == "bounded_but_under_covered_5_of_25"
    assert s2["reported_honest_verdict"] == "complete_structural_energy_s2v2_bounded_diverse_pool"
    assert s2["reported_energy_minus_accuracy_delta"] == pytest.approx(-0.15765776352537078)
    assert s2["reported_energy_minus_accuracy_delta_ci95"] == [
        -0.47765014592872246,
        0.004044943820224719,
    ]
    assert s2["n_effective_games"] == 5
    assert s2["n_available_games"] == 25
    assert s2["n_games_attempted"] == 5
    assert s2["effective_coverage_fraction"] == pytest.approx(0.2)
    assert s2["min_effective_games_required_under_tightened_gate"] == 15
    assert s2["tightened_effective_game_gate_passed"] is False
    assert s2["upstream_genuine_bounded_result"] is True
    assert s2["tested_effective_games"] == ["ar25", "cd82", "cn04", "ka59", "sc25"]
    assert artifact["v443_pivot"]["task_id"] == "exp4811-a1"
    assert artifact["v443_pivot"]["retests_corpus_wide"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4810_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4810: present next roadmap is activated onto active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.442", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=3.0,
        now_s=3.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.443"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["honest_verdict"] == (
        "complete_442_archived_443_activated_from_next_bounded_but_under_covered_5_of_25"
    )


def test_scenario_capstone_4810_blockers_and_poison_signature_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4810-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.442", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_443_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["s2v2_recorded_as_under_covered"] is False
    assert artifact["close_state_442"] == {}
    assert artifact["v443_pivot"] == {}

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4810",
        "registry": "arc_solve_registry",
        "capstone_4809": "missing_experiment_4809_capstone_v442",
        "s2v2_4801": "missing_experiment_4801_structural_energy_s2v2_diverse_trust_gate",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4810"] = False
        assert mod._first_blocker(bad) == expected

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["reproducible_total_levels"] = 64
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_65"

    environment_bad = copy.deepcopy(checks)
    environment_bad["environment_files"]["available"] = False
    assert mod._first_blocker(environment_bad) == "environment_files"

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
            "id": "test_expected_old_442_s2v2_degenerate_capstone",
            "reason": "single-failure smart-subset signature matches a stale honest-verdict expectation",
            "action": "blocked_for_fix_or_quarantine_before_tail_continues",
        }
    ]

    other_smart_failure = copy.deepcopy(checks)
    other_smart_failure["smart_subset_pretest_gate"]["passed"] = False
    other_smart_failure["smart_subset_pretest_gate"]["stdout_tail"] = "2 failed, 125 passed"
    other_smart_failure["smart_subset_pretest_gate"]["stderr_tail"] = ""
    assert mod._poison_test_resolution(other_smart_failure)["poison_tests"] == []


def test_scenario_capstone_4810_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4810-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_442"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    wrong_total = copy.deepcopy(valid)
    wrong_total["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_total)

    wrong_s2_record = copy.deepcopy(valid)
    wrong_s2_record["s2v2_recorded_as_under_covered"] = False
    with pytest.raises(ValueError, match="S2-v2"):
        mod.validate_artifact(wrong_s2_record)

    for field, value in {
        "corrected_verdict": "bounded",
        "n_effective_games": 6,
        "n_available_games": 24,
        "min_effective_games_required_under_tightened_gate": 10,
        "tightened_effective_game_gate_passed": True,
        "upstream_genuine_bounded_result": False,
    }.items():
        wrong = copy.deepcopy(valid)
        wrong["close_state_442"]["s2v2_corrected_record"][field] = value
        with pytest.raises(ValueError, match="S2-v2"):
            mod.validate_artifact(wrong)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v443_pivot"]["retests_corpus_wide"] = False
    with pytest.raises(ValueError, match="v443 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

    assert mod._available_game_count(tmp_path / "missing") == 0
    assert mod._float(True, 3.0) == 3.0
    assert mod._effective_game_names({"candidate_pool_diversity": "bad"}) == []
    assert mod._s2v2_under_covered_state({}, 0)["n_available_games"] == 0
    assert mod._activate_next_roadmap(tmp_path, next_info={"available": False}) == (False, "")

    activation_root = tmp_path / "activation_error"
    activation_root.mkdir()
    (activation_root / "research-roadmap.yaml").mkdir()
    (activation_root / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.443\n",
        encoding="utf-8",
    )
    activated, activation_error = mod._activate_next_roadmap(
        activation_root,
        next_info={"available": True, "parses": True, "milestone": "2026.06.443"},
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
    assert offline_artifact["preconditions_checked"]["offline_arcade"]["error"] == (
        "offline arcade unavailable"
    )

    missing_s2_root = tmp_path / "missing_s2"
    _write_repo_fixture(missing_s2_root, s2v2_present=False)
    missing_s2_artifact = mod.build_artifact(
        missing_s2_root,
        started_s=7.0,
        now_s=7.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )
    assert missing_s2_artifact["honest_verdict"] == (
        "blocked_missing_experiment_4801_structural_energy_s2v2_diverse_trust_gate"
    )


def test_arc_solver_kit_bare_import_shim_exposes_offline_arcade() -> None:
    """REQ-CAPSTONE-4810: the bare precondition import resolves in-package."""

    import arc_solver_kit

    assert callable(arc_solver_kit.offline_arcade)
