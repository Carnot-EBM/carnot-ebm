"""Tests for Exp 4790 `.440` archive / `.441` activation record.

Spec refs: REQ-CAPSTONE-4790, SCENARIO-CAPSTONE-4790,
SCENARIO-CAPSTONE-4790-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4790-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4790_archive_440_activate_441 as mod


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
        stdout="97 passed, 1 warning in 5.8s",
        stderr="",
    )


def _red_poison_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 97 passed, 1 warning in 6.1s",
        stderr="test_expected_old_440_capstone_headline still expects bounded S1",
    )


def _s1_4781() -> JsonDict:
    return {
        "experiment": "experiment_4781_structural_energy_s1_contrastive_landscape",
        "experiment_id": 4781,
        "honest_verdict": "success_structural_energy_s1_landscape_authorizes_s2",
        "s1_gate_passed": True,
        "s2_authorized": True,
        "energy_ranking_loo_auroc_mean": 0.7134961314270525,
        "energy_ranking_loo_auroc_ci95": [0.7133175599984811, 0.7137104171413382],
        "energy_ranking_loo_auroc_per_seed": [
            0.7132104171413381,
            0.7139247028556239,
            0.7139247028556239,
            0.7132104171413381,
            0.713567559998481,
            0.7132104171413381,
            0.713567559998481,
            0.7132104171413381,
            0.7139247028556239,
            0.7132104171413381,
        ],
        "n_seeds": 10,
        "random_seeds_used": [4781, 4782, 4783, 4784, 4785, 4786, 4787, 4788, 4789, 4790],
        "denoising_direction_agreement": 0.6223390275952694,
        "origin_probe_auroc": 0.5,
        "shuffled_label_control_auroc": 0.49335645814441664,
        "controls": {"v2_frame_marginal_energy_ranking_loo_auroc_mean": 0.48397091893626004},
        "per_family_loo": {"frame_delta": 0.7262429748613959, "object_relational": 0.6602487486471862},
        "n_candidate_rows": 463,
        "n_pos": 186,
        "n_neg": 277,
        "n_held_out_games": 16,
        "verifier_is_oracle": False,
        "retire_if_same_verdict": True,
        "retire_energy_guided_direction": False,
    }


def _capstone_4789() -> JsonDict:
    return {
        "experiment": "experiment_4789_capstone_v440",
        "experiment_id": 4789,
        "capstone_ready": True,
        "honest_verdict": "success_s1_structural_energy_usable_landscape_s2_authorized",
        "reproducible_total_levels": 65,
        "s1_structural_energy_verdict": {
            "verdict": "usable_landscape",
            "s1_gate_passed": True,
            "s2_authorized": True,
            "energy_ranking_loo_auroc_mean": 0.713496131427,
            "n_seeds": 10,
            "denoising_direction_agreement": 0.622339027595,
            "leak_controls_hold": True,
            "leak_controls": {
                "origin_probe_auroc": 0.5,
                "origin_probe_passed": True,
                "shuffled_label_control_auroc": 0.493356458144,
                "shuffled_label_control_passed": True,
                "marginal_control_loo_auroc": 0.483970918936,
                "marginal_control_passed": True,
            },
            "verifier_is_oracle": False,
            "upstream_honest_verdict": "success_structural_energy_s1_landscape_authorizes_s2",
        },
        "levelup_bank": {"new_levels_banked": 0, "reproducible_total_levels_after": 65},
        "readiness": {"ready_for_operator_submit": False, "s1_verdict": "usable_landscape"},
        "silent_bug_audit": {"s1_audit_note": "audit_note_recorded_does_not_override_live_clean_s1_pass"},
        "sota_handoff": {
            "flagged_for_v441_candidates": [
                "energy_value_guided_mcts_frontier_controller",
                "ebm_poe_planner_for_s3_generation",
            ]
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.441",
    next_present: bool = False,
    registry_total: int = 65,
    capstone_present: bool = True,
    s1_present: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4790-phase0\n"
        "    deliverable: results/experiment_4790_archive_440_activate_441.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.441\n"
            "tasks:\n"
            "  - id: exp4790-phase0\n"
            "    deliverable: results/experiment_4790_archive_440_activate_441.json\n"
            "  - id: exp4791-a1\n"
            "    deliverable: results/experiment_4791_structural_energy_s2_offpath_trust_gate.json\n",
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
    spec.write_text("REQ-CAPSTONE-4790\n", encoding="utf-8")
    if capstone_present:
        _write_json(root / "results" / "experiment_4789_capstone_v440.json", _capstone_4789())
    if s1_present:
        _write_json(root / "results" / "experiment_4781_structural_energy_s1_contrastive_landscape.json", _s1_4781())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4790_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4790: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4790" in spec
    assert "SCENARIO-CAPSTONE-4790" in spec
    assert "SCENARIO-CAPSTONE-4790-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4790-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4790_records_s1_pass_when_already_activated(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4790: active `.441` records S1 pass without next YAML."""

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
    assert artifact["honest_verdict"] == "complete_440_archived_441_activated_already_active_s1_pass_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.440",
        "activated_milestone": "2026.06.441",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "archive_noop_or_already_recorded",
    }
    assert artifact["s1_pass_recorded"] is True
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
    close = artifact["close_state_440"]
    assert close["capstone_honest_verdict"] == "success_s1_structural_energy_usable_landscape_s2_authorized"
    assert close["capstone_ready"] is True
    assert close["reproducible_total_levels"] == 65
    s1 = close["s1_pass"]
    assert s1["headline"] == "S1 PASSED -- usable LANDSCAPE; S2 authorized"
    assert s1["honest_verdict"] == "success_structural_energy_s1_landscape_authorizes_s2"
    assert s1["verdict"] == "usable_landscape"
    assert s1["s1_gate_passed"] is True
    assert s1["s2_authorized"] is True
    assert s1["energy_ranking_loo_auroc_mean_rounded"] == 0.713
    assert s1["n_seeds"] == 10
    assert s1["seed_floor_met"] is True
    assert s1["denoising_direction_agreement_rounded"] == 0.622
    assert s1["denoising_direction_passed"] is True
    assert s1["leak_controls_clean"] is True
    assert s1["verifier_is_oracle"] is False
    assert artifact["v441_pivot"]["headline"] == "S2 off-path trust gate"
    assert artifact["v441_pivot"]["s2_authorized_by_s1"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4790_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4790: present next roadmap is activated onto active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.440", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=3.0,
        now_s=3.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.441"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["honest_verdict"] == "complete_440_archived_441_activated_from_next_s1_pass_recorded"


def test_scenario_capstone_4790_blockers_and_poison_signature_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4790-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.440", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_441_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["s1_pass_recorded"] is False
    assert artifact["close_state_440"] == {}
    assert artifact["v441_pivot"] == {}

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4790",
        "registry": "arc_solve_registry",
        "capstone_4789": "missing_experiment_4789_capstone_v440",
        "s1_4781": "missing_experiment_4781_structural_energy_s1_contrastive_landscape",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4790"] = False
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
            "id": "test_expected_old_440_capstone_headline",
            "reason": "single-failure smart-subset signature matches a stale honest-verdict expectation",
            "action": "blocked_for_fix_or_quarantine_before_tail_continues",
        }
    ]

    other_smart_failure = copy.deepcopy(checks)
    other_smart_failure["smart_subset_pretest_gate"]["passed"] = False
    other_smart_failure["smart_subset_pretest_gate"]["stdout_tail"] = "2 failed, 96 passed"
    other_smart_failure["smart_subset_pretest_gate"]["stderr_tail"] = ""
    assert mod._poison_test_resolution(other_smart_failure)["poison_tests"] == []


def test_scenario_capstone_4790_field_principle_validation_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4790-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_440"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    wrong_total = copy.deepcopy(valid)
    wrong_total["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_total)

    wrong_s1_record = copy.deepcopy(valid)
    wrong_s1_record["s1_pass_recorded"] = False
    with pytest.raises(ValueError, match="S1"):
        mod.validate_artifact(wrong_s1_record)

    wrong_verdict = copy.deepcopy(valid)
    wrong_verdict["close_state_440"]["s1_pass"]["verdict"] = "bounded"
    with pytest.raises(ValueError, match="S1"):
        mod.validate_artifact(wrong_verdict)

    wrong_s2 = copy.deepcopy(valid)
    wrong_s2["close_state_440"]["s1_pass"]["s2_authorized"] = False
    with pytest.raises(ValueError, match="S1"):
        mod.validate_artifact(wrong_s2)

    wrong_leak = copy.deepcopy(valid)
    wrong_leak["close_state_440"]["s1_pass"]["leak_controls_clean"] = False
    with pytest.raises(ValueError, match="S1"):
        mod.validate_artifact(wrong_leak)

    wrong_oracle = copy.deepcopy(valid)
    wrong_oracle["close_state_440"]["s1_pass"]["verifier_is_oracle"] = True
    with pytest.raises(ValueError, match="S1"):
        mod.validate_artifact(wrong_oracle)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v441_pivot"]["s2_authorized_by_s1"] = False
    with pytest.raises(ValueError, match="v441 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

    assert mod._float(True, 7.0) == 7.0
    assert mod._float("bad", 9.0) == 9.0
    assert mod._int(False, 2) == 2
    assert mod._int("bad", 3) == 3
    assert mod._registry_total_levels(tmp_path / "missing.yaml") is None
    assert mod._activate_next_roadmap(tmp_path, next_info={"available": False}) == (False, "")
    assert mod._json_object(tmp_path / "missing.json") == {}

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("milestone: [\n", encoding="utf-8")
    assert mod._yaml_info(bad_yaml)["parses"] is False
    assert mod._registry_total_levels(bad_yaml) is None

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not-a-map\n", encoding="utf-8")
    assert mod._yaml_info(list_yaml)["milestone"] is None
    assert mod._registry_total_levels(list_yaml) is None

    bool_registry = tmp_path / "bool.yaml"
    bool_registry.write_text("reproducible_total_levels: true\n", encoding="utf-8")
    assert mod._registry_total_levels(bool_registry) is None

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[\n", encoding="utf-8")
    assert mod._json_object(bad_json) == {}

    list_json = tmp_path / "list.json"
    list_json.write_text("[1]\n", encoding="utf-8")
    assert mod._json_object(list_json) == {}

    activation_root = tmp_path / "activation_error"
    activation_root.mkdir()
    (activation_root / "research-roadmap.yaml").mkdir()
    (activation_root / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.441\n",
        encoding="utf-8",
    )
    activated, activation_error = mod._activate_next_roadmap(
        activation_root,
        next_info={"available": True, "parses": True, "milestone": "2026.06.441"},
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

    missing_s1_root = tmp_path / "missing_s1"
    _write_repo_fixture(missing_s1_root, s1_present=False)
    missing_s1_artifact = mod.build_artifact(
        missing_s1_root,
        started_s=7.0,
        now_s=7.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )
    assert missing_s1_artifact["honest_verdict"] == (
        "blocked_missing_experiment_4781_structural_energy_s1_contrastive_landscape"
    )
