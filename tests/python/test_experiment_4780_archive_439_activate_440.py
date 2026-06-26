"""Tests for Exp 4780 `.439` archive / `.440` activation record.

Spec refs: REQ-CAPSTONE-4780, SCENARIO-CAPSTONE-4780,
SCENARIO-CAPSTONE-4780-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4780-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4780_archive_439_activate_440 as mod


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
        stdout="94 passed, 1 warning in 6.12s",
        stderr="",
    )


def _red_poison_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 94 passed, 1 warning in 6.55s",
        stderr="test_expected_stale_honest_verdict failed against now-correct honest verdict",
    )


def _s0prime_4771() -> JsonDict:
    return {
        "experiment": "experiment_4771_structural_energy_s0prime_origin_matched",
        "experiment_id": 4771,
        "honest_verdict": "success_structural_energy_s0prime_reopens_s1",
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "TAUTOLOGY",
                "detail": (
                    "chance-floor false-positive: loo_auroc_majority_control=0.5 "
                    "and origin_probe_auroc=0.5"
                ),
            }
        ],
        "verifier_is_oracle": False,
        "s0prime_gate_passed": True,
        "retire_energy_guided_direction": False,
        "retire_if_same_verdict": True,
        "loo_auroc_structural": 0.7386642861889572,
        "loo_auroc_ci95": [0.636412794237013, 0.8332008450933205],
        "loo_auroc_marginal_control": 0.510766138123749,
        "loo_auroc_majority_control": 0.5,
        "origin_probe_auroc": 0.5,
        "shuffled_label_control_auroc": 0.5033091959271814,
        "structural_minus_marginal_delta_ci95": [0.1271826145701076, 0.33248035484558097],
        "in_sample_auroc": 0.8479387446139514,
        "n_candidate_rows": 463,
        "n_pos": 186,
        "n_neg": 277,
        "n_held_out_games": 16,
        "per_family_loo": {"frame_delta": 0.7392638081947291, "object_relational": 0.659731635551948},
        "dataset_diagnostics": {"origin_matched": True},
        "origin_probe": {
            "loo_auroc": 0.5,
            "origin_counts": {"induced": 463},
            "status": "origin_matched_single_origin_all_induced",
        },
    }


def _capstone_4779() -> JsonDict:
    return {
        "experiment": "experiment_4779_capstone_v439",
        "experiment_id": 4779,
        "honest_verdict": "complete_s0prime_structural_energy_retires_v439_capstone_ready",
        "reproducible_total_levels": 65,
        "s0prime_structural_energy_verdict": {
            "artifact_skipped": True,
            "control_numbers_source": "BUG_AUDIT",
            "direction": "RETIRES",
            "experiment_id": 4771,
            "origin_probe_auroc": 0.5,
            "reason": "s0prime_flagged_adversarial_skipped_controls_unfired",
            "s1_queued": False,
            "shuffled_label_control_auroc": 0.503309195927,
            "source": "S0PRIME",
            "verifier_is_oracle": False,
        },
        "flagged_artifacts_skipped": [
            {
                "experiment_id": 4771,
                "path": "results/experiment_4771_structural_energy_s0prime_origin_matched.json",
                "reason": "flagged_adversarial",
                "source": "S0PRIME",
            },
            {"experiment_id": 4773, "reason": "flagged_adversarial", "source": "SELF_PLAY"},
            {"experiment_id": 4776, "reason": "flagged_adversarial", "source": "PACKAGE"},
        ],
        "levelup_bank": {
            "new_levels_banked": 0,
            "reproducible_total_levels_before": 65,
            "reproducible_total_levels_after": 65,
            "reproducible_total_levels_delta": 0,
            "target_game": "ka59",
            "verifier_is_oracle": True,
        },
        "readiness": {"ready_for_operator_submit": False},
        "silent_bug_audit": {
            "reopened_null_ids": ["experiment_4771_structural_energy_s0prime_origin_matched"],
            "s0prime_reopened_for_control_bug": True,
        },
        "sota_handoff": {
            "flagged_for_v440_candidates": [
                "slot_relational_contrastive_energy_s0prime_guarded",
                "poe_code_world_model_trust_gate_after_s0prime",
            ],
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.440",
    next_present: bool = False,
    registry_total: int = 65,
    capstone_present: bool = True,
    s0prime_present: bool = True,
    note_present: bool = True,
) -> None:
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4780-phase0\n"
        "    deliverable: results/experiment_4780_archive_439_activate_440.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.440\n"
            "tasks:\n"
            "  - id: exp4780-phase0\n"
            "    deliverable: results/experiment_4780_archive_439_activate_440.json\n"
            "  - id: exp4781-a1\n"
            "    deliverable: results/experiment_4781_structural_energy_s1_contrastive_landscape.json\n",
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
    spec.write_text("REQ-CAPSTONE-4780\n", encoding="utf-8")
    if capstone_present:
        _write_json(root / "results" / "experiment_4779_capstone_v439.json", _capstone_4779())
    if s0prime_present:
        _write_json(
            root / "results" / "experiment_4771_structural_energy_s0prime_origin_matched.json",
            _s0prime_4771(),
        )
    if note_present:
        note = root / "docs" / "research-notes" / (
            "oracle-distinct-structural-energy-program-2026-06-26.md"
        )
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(
            "# Oracle-Distinct Structural Energy on ARC\n\n"
            "S0' (exp4771) REOPENS to S1: origin_probe 0.733 -> 0.500, "
            "structural LOO 0.739 CI excludes chance.\n",
            encoding="utf-8",
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


def test_req_capstone_4780_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4780: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4780" in spec
    assert "SCENARIO-CAPSTONE-4780" in spec
    assert "SCENARIO-CAPSTONE-4780-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4780-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4780_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4780: active `.440` allows a complete record without next YAML."""

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
    assert artifact["honest_verdict"] == "complete_439_archived_440_activated_already_active_s0prime_reopen_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.439",
        "activated_milestone": "2026.06.440",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "archive_noop_or_already_recorded",
    }
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["s0prime_reopen_recorded"] is True
    next_check = artifact["preconditions_checked"]["research_roadmap_next_yaml"]
    assert next_check["available"] is False
    assert next_check["literal_precondition_passed"] is False
    assert next_check["accepted_missing_because_already_active"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True
    assert artifact["poison_test_resolved"] == {
        "resolved": True,
        "current_gate_passed": True,
        "poison_tests": [],
        "action": "no_poison_observed_current_gate_green",
    }

    close = artifact["close_state_439"]
    assert close["capstone_honest_verdict"] == "complete_s0prime_structural_energy_retires_v439_capstone_ready"
    assert close["capstone_skipped_s0prime"] is True
    assert close["reproducible_total_levels"] == 65
    s0prime = close["s0prime_true_close_state"]
    assert s0prime["direction"] == "REOPENS_TO_S1"
    assert s0prime["headline"] == "S0' REOPENS to S1 despite stale-conductor TAUTOLOGY skip"
    assert s0prime["honest_verdict"] == "success_structural_energy_s0prime_reopens_s1"
    assert s0prime["artifact_flagged_adversarial"] is True
    assert s0prime["flag_is_known_false_positive"] is True
    assert s0prime["origin_probe_auroc_before"] == 0.733
    assert s0prime["origin_probe_auroc_after"] == 0.5
    assert s0prime["loo_auroc_structural_rounded"] == 0.739
    assert s0prime["loo_auroc_ci95"][0] > 0.5
    assert s0prime["verifier_is_oracle"] is False
    assert close["note_citation"]["path"] == str(mod.RESEARCH_NOTE_REL_PATH)
    assert artifact["v440_pivot"]["headline"] == "S1 contrastive energy landscape"
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4780_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4780: present next roadmap is activated onto the active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.439", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=3.0,
        now_s=3.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.440"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True


def test_scenario_capstone_4780_blockers_and_poison_signature_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4780-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.439", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_440_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["s0prime_reopen_recorded"] is False
    assert artifact["close_state_439"] == {}
    assert artifact["v440_pivot"] == {}

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4780",
        "registry": "arc_solve_registry",
        "capstone_4779": "missing_experiment_4779_capstone_v439",
        "s0prime_4771": "missing_experiment_4771_structural_energy_s0prime_origin_matched",
        "research_note": "missing_oracle_distinct_structural_energy_program_note",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4780"] = False
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
            "id": "test_expected_stale_honest_verdict",
            "reason": "single-failure smart-subset signature matches a stale honest-verdict expectation",
            "action": "blocked_for_fix_or_quarantine_before_tail_continues",
        }
    ]


def test_scenario_capstone_4780_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4780-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_439"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    wrong_total = copy.deepcopy(valid)
    wrong_total["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_total)

    wrong_reopen = copy.deepcopy(valid)
    wrong_reopen["s0prime_reopen_recorded"] = False
    with pytest.raises(ValueError, match="S0"):
        mod.validate_artifact(wrong_reopen)

    wrong_direction = copy.deepcopy(valid)
    wrong_direction["close_state_439"]["s0prime_true_close_state"]["direction"] = "RETIRES"
    with pytest.raises(ValueError, match="S0"):
        mod.validate_artifact(wrong_direction)

    wrong_flag = copy.deepcopy(valid)
    wrong_flag["close_state_439"]["s0prime_true_close_state"]["flag_is_known_false_positive"] = False
    with pytest.raises(ValueError, match="S0"):
        mod.validate_artifact(wrong_flag)

    wrong_capstone_skip = copy.deepcopy(valid)
    wrong_capstone_skip["close_state_439"]["capstone_skipped_s0prime"] = False
    with pytest.raises(ValueError, match="capstone"):
        mod.validate_artifact(wrong_capstone_skip)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v440_pivot"]["s1_authorized_by_s0prime"] = False
    with pytest.raises(ValueError, match="v440 pivot"):
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
    assert mod._capstone_skipped_s0prime({}) is False

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("milestone: [\n", encoding="utf-8")
    assert mod._yaml_info(bad_yaml)["parses"] is False
    assert mod._registry_total_levels(bad_yaml) is None

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not-a-map\n", encoding="utf-8")
    assert mod._yaml_info(list_yaml)["milestone"] is None
    assert mod._registry_total_levels(list_yaml) is None

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
        "milestone: 2026.06.440\n",
        encoding="utf-8",
    )
    activated, activation_error = mod._activate_next_roadmap(
        activation_root,
        next_info={"available": True, "parses": True, "milestone": "2026.06.440"},
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
