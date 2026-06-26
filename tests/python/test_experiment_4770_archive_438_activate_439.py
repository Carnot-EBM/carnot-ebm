"""Tests for Exp 4770 `.438` archive / `.439` activation record.

Spec refs: REQ-CAPSTONE-4770, SCENARIO-CAPSTONE-4770,
SCENARIO-CAPSTONE-4770-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4770-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4770_archive_438_activate_439 as mod


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
        stdout="91 passed, 1 warning in 6.12s",
        stderr="",
    )


def _red_poison_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 91 passed, 1 warning in 6.55s",
        stderr="test_expected_stale_honest_verdict failed against now-correct honest verdict",
    )


def _s0_4761() -> JsonDict:
    return {
        "experiment_id": 4761,
        "honest_verdict": "complete: structural_energy_s0_retired_loo_0.746_null_or_leaky",
        "s0_gate_passed": False,
        "retire_energy_guided_direction": True,
        "retire_if_same_verdict": True,
        "loo_auroc_structural": 0.745588188062,
        "loo_auroc_majority_control": 0.5,
        "loo_auroc_marginal_control": 0.464578396462,
        "origin_probe_auroc": 0.732792721071,
        "structural_minus_marginal_delta_ci95": [
            0.17481816435901296,
            0.39015500191631836,
        ],
        "n_held_out_games": 16,
        "n_candidate_rows": 463,
        "verifier_is_oracle": False,
    }


def _capstone_4769() -> JsonDict:
    return {
        "experiment_id": 4769,
        "honest_verdict": "complete: s0_structural_energy_retired_v438_capstone_ready",
        "reproducible_total_levels": 65,
        "s0_structural_energy_verdict": {
            "direction": "RETIRED",
            "s1_queued": False,
            "s0_gate_passed": False,
            "loo_auroc_structural": 0.745588188062,
            "origin_probe_auroc": 0.732792721071,
            "retire_energy_guided_direction": True,
            "retire_if_same_verdict": True,
            "reason": "s0_gate_failed_or_null_or_leaky",
        },
        "levelup_bank": {
            "target_game": "re86",
            "new_levels_banked": 0,
            "reproducible_total_levels_after": 65,
            "reproducible_total_levels_delta": 0,
        },
        "flagged_artifacts_skipped": [
            {"experiment_id": 4763, "source": "SELF_PLAY", "reason": "flagged_adversarial"},
            {"experiment_id": 4766, "source": "PACKAGE", "reason": "flagged_adversarial"},
        ],
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.439",
    next_present: bool = False,
    registry_total: int = 65,
    capstone_present: bool = True,
    s0_present: bool = True,
) -> None:
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4770-phase0\n"
        "    deliverable: results/experiment_4770_archive_438_activate_439.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.439\n"
            "tasks:\n"
            "  - id: exp4770-phase0\n"
            "    deliverable: results/experiment_4770_archive_438_activate_439.json\n"
            "  - id: exp4771-a1\n"
            "    deliverable: results/experiment_4771_structural_energy_s0prime_origin_matched.json\n",
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
    spec.write_text("REQ-CAPSTONE-4770\n", encoding="utf-8")
    if capstone_present:
        _write_json(root / "results" / "experiment_4769_capstone_v438.json", _capstone_4769())
    if s0_present:
        _write_json(root / "results" / "experiment_4761_structural_energy_s0_core_bet_probe.json", _s0_4761())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4770_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4770: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4770" in spec
    assert "SCENARIO-CAPSTONE-4770" in spec
    assert "SCENARIO-CAPSTONE-4770-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4770-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4770_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4770: active `.439` allows a complete record without next YAML."""

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
    assert artifact["honest_verdict"] == "complete_438_archived_439_activated_already_active_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.438",
        "activated_milestone": "2026.06.439",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "archive_noop_or_already_recorded",
    }
    assert artifact["reproducible_total_levels"] == 65
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

    close = artifact["close_state_438"]
    assert close["source_capstone_honest_verdict"] == "complete: s0_structural_energy_retired_v438_capstone_ready"
    assert close["reproducible_total_levels"] == 65
    s0 = close["s0_structural_energy"]
    assert s0["direction"] == "RETIRED"
    assert s0["headline"] == "structural LOO 0.746 but leak-audit-failed -> retired"
    assert s0["loo_auroc_structural"] == 0.745588188062
    assert s0["loo_auroc_structural_rounded"] == 0.746
    assert s0["origin_probe_auroc"] == 0.732792721071
    assert s0["leak_audit_failed"] is True
    assert s0["retired_on_leak"] is True
    assert close["levelup_bank"]["new_levels_banked"] == 0
    assert close["flagged_artifacts_skipped"] == [
        {"experiment_id": 4763, "source": "SELF_PLAY", "reason": "flagged_adversarial"},
        {"experiment_id": 4766, "source": "PACKAGE", "reason": "flagged_adversarial"},
    ]
    assert artifact["v439_pivot"] == {
        "headline": "S0' origin-matched structural-energy re-test",
        "task_id": "exp4771-a1",
        "origin_matched_retest": True,
        "positive_class": "induced_correct_prediction",
        "negative_class": "induced_wrong_prediction",
        "purpose": "disentangle correctness signal from S0 real-vs-induced origin leak",
    }
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4770_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4770: present next roadmap is activated onto the active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.438", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=3.0,
        now_s=3.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.439"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True


def test_scenario_capstone_4770_blockers_and_poison_signature_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4770-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.438", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_439_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["poison_test_resolved"]["resolved"] is False
    assert artifact["close_state_438"] == {}
    assert artifact["v439_pivot"] == {}

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4770",
        "registry": "arc_solve_registry",
        "capstone_4769": "missing_experiment_4769_capstone_v438",
        "s0_4761": "missing_experiment_4761_structural_energy_s0_core_bet_probe",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4770"] = False
        assert mod._first_blocker(bad) == expected

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["reproducible_total_levels"] = 64
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_65"

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


def test_scenario_capstone_4770_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4770-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_438"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    wrong_total = copy.deepcopy(valid)
    wrong_total["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_total)

    wrong_s0_direction = copy.deepcopy(valid)
    wrong_s0_direction["close_state_438"]["s0_structural_energy"]["direction"] = "ALIVE"
    with pytest.raises(ValueError, match="S0"):
        mod.validate_artifact(wrong_s0_direction)

    wrong_leak = copy.deepcopy(valid)
    wrong_leak["close_state_438"]["s0_structural_energy"]["leak_audit_failed"] = False
    with pytest.raises(ValueError, match="S0"):
        mod.validate_artifact(wrong_leak)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v439_pivot"]["origin_matched_retest"] = False
    with pytest.raises(ValueError, match="v439 pivot"):
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
        "milestone: 2026.06.439\n",
        encoding="utf-8",
    )
    activated, activation_error = mod._activate_next_roadmap(
        activation_root,
        next_info={"available": True, "parses": True, "milestone": "2026.06.439"},
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
