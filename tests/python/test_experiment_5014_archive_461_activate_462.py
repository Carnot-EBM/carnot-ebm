"""Tests for REQ-CAPSTONE-5014 / SCENARIO-CAPSTONE-5014."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5014_archive_461_activate_462 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _capstone_artifact() -> JsonDict:
    best_arm = {
        "arm": "EBRM",
        "arm_id": "D3",
        "corpus": "MuSR",
        "delta_vs_tuned_sc": 0.0,
        "headroom_present": True,
        "mcnemar_p": 1.0,
        "n_questions": 200,
        "oracle_at_k": 0.93,
        "paired_ci95": [-0.03, 0.025],
        "selection_accuracy": 0.585,
        "source_experiment_id": 5005,
        "tuned_sc_accuracy": 0.585,
        "verifier_is_oracle": False,
        "win_vs_tuned_sc": False,
    }
    return {
        "arc_deliverable_locked": True,
        "best_arm_and_delta": {
            "arm": "EBRM",
            "arm_id": "D3",
            "corpus": "MuSR",
            "delta_vs_tuned_sc": 0.0,
            "headroom_present": True,
            "paired_ci95": [-0.03, 0.025],
            "source_experiment_id": 5005,
            "verifier_is_oracle": False,
            "win_vs_tuned_sc": False,
        },
        "capstone_ready": True,
        "diffusiongemma_gate_status": {
            "activation": "not_activated",
            "autonomously_flipped_to_met": False,
            "conditions_satisfied_off_arc": False,
            "operator_gated": True,
            "status": "STILL-PENDING",
        },
        "flagged_artifacts_skipped": [
            {
                "experiment_id": 5003,
                "honest_verdict": "running_lora_ebm_scorer_musr_pretrain_skeleton",
                "path": "results/experiment_5003_lora_ebm_scorer_musr.json",
                "reason": "flagged_adversarial",
                "source": "D1_LORA_EBM",
            },
            {
                "experiment_id": 5004,
                "honest_verdict": "blocked_uprm_logprob_candidate_cache",
                "path": "results/experiment_5004_uprm_replication.json",
                "reason": "flagged_adversarial",
                "source": "D2_UPRM",
            },
            {
                "experiment_id": 5006,
                "honest_verdict": "running_moat_second_corpus_mmlu_pro_hard_skeleton",
                "path": "results/experiment_5006_moat_second_corpus.json",
                "reason": "flagged_adversarial",
                "source": "D4_SECOND_CORPUS",
            },
        ],
        "honest_verdict": "complete_capstone_v461_moat_musr_scoped_ebrm_musr_delta_0p000",
        "moat_verdict": {
            "decision": "MIXED-SCOPED",
            "moat_realized": False,
            "moat_retired_bounded": False,
            "state": "mixed_musr_scoped",
            "summary": "D5 is scoped rather than realized or retired.",
        },
        "next_milestone_pointer": {
            "best_arm": "EBRM",
            "direction": "tighten_strongest_arm",
            "milestone": "2026.06.462",
        },
        "per_arm_table": [best_arm],
        "reproducible_total_levels": 69,
    }


def _d1_artifact() -> JsonDict:
    return {
        "flagged_adversarial": True,
        "honest_verdict": "running_lora_ebm_scorer_musr_pretrain_skeleton",
        "n_pairs": 0,
        "preconditions_checked": [
            {"available": True, "resource": "trainable_qwen_base"},
            {"available": True, "resource": "cuda"},
            {"available": True, "resource": "cached_musr_candidates"},
            {"available": True, "resource": "fover_pairs"},
        ],
        "train_loss": None,
    }


def _d2_artifact() -> JsonDict:
    return {
        "flagged_adversarial": True,
        "honest_verdict": "blocked_uprm_logprob_candidate_cache",
        "preconditions_checked": [
            {"available": True, "resource": "gemma_gguf_cache"},
            {"available": True, "resource": "llama_server_logprobs"},
            {"available": True, "resource": "target_corpus"},
            {
                "available": False,
                "detail": "0 cached uPRM logprob row(s), required >= 200; "
                "CARNOT_UPRM_ENABLE_FRESH_GENERATION=''",
                "resource": "uprm_logprob_candidate_cache",
            },
        ],
    }


def _d3_artifact() -> JsonDict:
    return {
        "delta_vs_tuned_sc": 0.0,
        "ebrm_selection_accuracy": 0.585,
        "evaluation": {
            "mcnemar_p": 1.0,
            "paired_ci95": [-0.03, 0.025],
            "point_estimate_accuracy": 0.515,
            "tuned_self_consistency": {
                "accuracy": 0.585,
                "config": {"k": 1, "temperature": "cached"},
            },
        },
        "headroom_present": True,
        "honest_verdict": "complete_ebrm_no_win_musr_plus_0p000_ci_incl_0",
        "mcnemar_p": 1.0,
        "model_specs": {
            "base_scorer": "registry_quality_ensemble",
            "tuned_self_consistency_config": {"k": 1, "temperature": "cached"},
        },
        "paired_ci95": [-0.03, 0.025],
        "tuned_sc_accuracy": 0.585,
        "uncertainty_calibration": {
            "calibration_curve": [
                {"abstain_rate": 0.975, "accuracy": 0.625, "coverage": 0.025}
            ],
            "claim": "post_hoc_reward_distribution_spread_abstains_to_tuned_sc",
        },
        "verifier_is_oracle": False,
    }


def _d4_artifact() -> JsonDict:
    return {
        "flagged_adversarial": True,
        "honest_verdict": "running_moat_second_corpus_mmlu_pro_hard_skeleton",
        "preconditions_checked": [
            {
                "available": False,
                "detail": "D1 artifact is blocked/skeleton or lacks numeric verifier metrics",
                "resource": "d1_verifier",
            },
            {
                "available": False,
                "detail": "D2 artifact is blocked/skeleton or lacks numeric verifier metrics",
                "resource": "d2_verifier",
            },
            {"available": True, "resource": "d3_verifier"},
            {"available": True, "resource": "second_corpus_mmlu_pro_hard"},
        ],
    }


def _make_root(
    root: Path,
    *,
    include_active: bool = True,
    include_next: bool = False,
    include_capstone: bool = True,
    active_milestone: str = "2026.06.462",
) -> None:
    roadmap_text = (
        f"milestone: {active_milestone}\n"
        "note: 'PHASE D remains the MAJORITY lever for .462; ARC LOCKED at levels 69; "
        "DiffusionGemma STILL-PENDING; energy-as-ARC S0 CONCLUDED; verifier-as-reward retired.'\n"
        "context: 'D3 EBRM degenerate = abstained 97.5% to a k=1 strawman tuned-SC.'\n"
    )
    if include_active:
        _write_text(root / "research-roadmap.yaml", roadmap_text)
    if include_next:
        _write_text(root / "research-roadmap-next.yaml", "milestone: 2026.06.462\n")
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 69\n")
    if include_capstone:
        _write_json(root / mod.CAPSTONE_REL_PATH, _capstone_artifact())
    _write_json(root / mod.D1_REL_PATH, _d1_artifact())
    _write_json(root / mod.D2_REL_PATH, _d2_artifact())
    _write_json(root / mod.D3_REL_PATH, _d3_artifact())
    _write_json(root / mod.D4_REL_PATH, _d4_artifact())


def _runner(calls: list[list[str]], *, roadmap_ok: bool, offline_ok: bool, pretest_ok: bool):
    def run(command: list[str], _root: Path) -> mod.CommandResult:
        calls.append(command)
        command_text = " ".join(command)
        if "research-roadmap.yaml" in command_text:
            return mod.CommandResult(command, 0 if roadmap_ok else 1, "ok active\n", "roadmap")
        if "offline_arcade" in command_text:
            return mod.CommandResult(command, 0 if offline_ok else 1, "", "arcade")
        return mod.CommandResult(command, 0 if pretest_ok else 1, "passed", "failed")

    return run


def test_req_capstone_5014_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5014: OpenSpec declares the .461/.462 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5014") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert str(mod.OUTPUT_REL_PATH) in section
    for field in mod.REQUIRED_FIELDS:
        assert field in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    assert "D1/D2/D4 as flagged-adversarial execution failures" in section
    assert mod.PRETEST_COMMAND == [
        ".venv/bin/pytest",
        "tests/python/test_experiment_5014_archive_461_activate_462.py",
        "-q",
        "--no-cov",
    ]


def test_scenario_capstone_5014_complete_transition_records_461_close_state(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5014: active .462 roadmap records the true .461 close-state."""

    _make_root(tmp_path, include_active=True, include_next=False)
    calls: list[list[str]] = []
    exit_code = mod.main(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
    )
    artifact = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(calls) == 3
    assert "p if os.path.exists(p) else q" in " ".join(calls[0])
    assert calls[2] == mod.PRETEST_COMMAND
    assert artifact["honest_verdict"] == "complete_461_archived_462_activated_phase_d_continues"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["phase_d_first_real_test"] is True
    assert artifact["prior_milestone_moat_verdict"] == "MIXED-SCOPED"
    assert [row["arm_id"] for row in artifact["execution_defects_to_fix"]] == [
        "D1",
        "D2",
        "D3",
    ]
    assert artifact["pretest_gate"]["ran"] is True
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["transition"]["active_milestone_confirmed"] == "2026.06.462"
    assert artifact["transition"]["active_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["transition"]["activation_state"] == "already_active_or_activated_462"
    assert artifact["transition_performed"] is True
    assert artifact["phase_d_majority_lever"] is True
    assert artifact["arc_locked"] is True
    assert artifact["leaderboard_submission"] is False
    assert artifact["diffusiongemma_gate_status"]["status"] == "STILL-PENDING"
    assert artifact["diffusiongemma_gate_status"]["autonomously_flipped_to_met"] is False
    assert artifact["reproducible_total_levels"] == 69

    close = artifact["close_state_461"]
    assert close["capstone"]["honest_verdict"] == (
        "complete_capstone_v461_moat_musr_scoped_ebrm_musr_delta_0p000"
    )
    assert close["moat_verdict"]["decision"] == "MIXED-SCOPED"
    assert close["moat_verdict"]["moat_realized"] is False
    assert close["moat_verdict"]["moat_retired_bounded"] is False
    assert close["clean_arm"]["arm_id"] == "D3"
    assert close["clean_arm"]["delta_vs_tuned_sc"] == 0.0
    assert close["clean_arm"]["paired_ci95"] == [-0.03, 0.025]
    assert close["clean_arm"]["mcnemar_p"] == 1.0
    assert close["d3_degeneracy"]["abstention_rate"] == 0.975
    assert close["d3_degeneracy"]["tuned_sc_k"] == 1
    assert close["d3_degeneracy"]["weak_base_scorer"] == "registry_quality_ensemble"
    assert close["d1_skeleton_bail"]["n_pairs"] == 0
    assert close["d1_skeleton_bail"]["train_loss"] is None
    assert close["d2_logprob_cache_block"]["fresh_generation_disabled"] is True
    assert close["d4_cross_corpus_skeleton_bail"]["flagged_adversarial"] is True
    assert [row["experiment_id"] for row in close["flagged_adversarial_skipped"]] == [
        5003,
        5004,
        5006,
    ]
    assert close["phase_d"]["majority_lever_for_462"] is True
    assert close["phase_d"]["first_real_test_in_462"] is True
    assert close["do_not_queue"] == ["energy-as-ARC", "verifier-as-reward"]
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5014_blocked_missing_roadmaps_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5014-BLOCKED-PRECONDITION: neither roadmap parses."""

    _make_root(tmp_path, include_active=False, include_next=False)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=False, offline_ok=True, pretest_ok=True),
        started_s=10.0,
        now_s=10.25,
    )

    assert artifact["honest_verdict"] == "blocked_roadmap_yaml_missing"
    assert len(calls) == 2
    assert artifact["pretest_gate"] == {
        "green": False,
        "ran": False,
        "reason": "skipped_after_precondition_failure",
    }
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["transition"]["active_milestone_confirmed"] == "unknown"
    assert artifact["transition_performed"] is False
    assert artifact["close_state_461"]["reproducible_total_levels"] == 69
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5014_pretest_failure_blocks_after_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5014: red pre-test gate is recorded, not hidden."""

    _make_root(tmp_path)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=False),
        started_s=4.0,
        now_s=4.5,
    )

    assert artifact["honest_verdict"] == "blocked_pretest_gate_failed"
    assert len(calls) == 3
    assert artifact["pretest_gate"]["ran"] is True
    assert artifact["pretest_gate"]["green"] is False
    assert artifact["transition_performed"] is False
    assert artifact["poison_test_resolved"] == {"quarantined": False, "test": "", "reason": ""}
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5014_missing_capstone_blocks_before_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5014-BLOCKED-PRECONDITION: missing close-state blocks."""

    _make_root(tmp_path, include_capstone=False)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=2.0,
        now_s=2.25,
    )

    assert artifact["honest_verdict"] == "blocked_capstone_v461_missing"
    assert len(calls) == 2
    assert artifact["pretest_gate"]["ran"] is False
    assert artifact["transition_performed"] is False
    assert artifact["close_state_461"]["capstone"]["honest_verdict"] == ""
    assert artifact["phase_d_first_real_test"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5014_resource_blockers_and_optional_next(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-5014: malformed resources fail closed without fabrication."""

    bad_json = tmp_path / "bad.json"
    bad_yaml = tmp_path / "bad.yaml"
    bad_json.write_text("{", encoding="utf-8")
    bad_yaml.write_text("bad: [", encoding="utf-8")

    assert mod._read_json_object_safe(bad_json) == {}
    assert mod._read_yaml_object_safe(bad_yaml) == {}

    ok = {
        "active_roadmap_yaml": {"passed": True, "active_exists": True, "next_exists": False},
        "offline_arcade": {"passed": True},
        "registry": {"exists": True, "loadable": True},
        "capstone_v461": {"exists": True, "loadable": True},
    }
    roadmap_bad = {**ok, "active_roadmap_yaml": {"passed": False, "active_exists": True}}
    offline_bad = {**ok, "offline_arcade": {"passed": False}}
    registry_missing = {**ok, "registry": {"exists": False, "loadable": False}}
    registry_bad = {**ok, "registry": {"exists": True, "loadable": False}}
    capstone_bad = {**ok, "capstone_v461": {"exists": True, "loadable": False}}

    assert mod.precondition_blocker(roadmap_bad) == "blocked_roadmap_yaml_unparseable"
    assert mod.precondition_blocker(offline_bad) == "blocked_offline_arcade_unavailable"
    assert mod.precondition_blocker(registry_missing) == "blocked_arc_solve_registry_missing"
    assert mod.precondition_blocker(registry_bad) == "blocked_arc_solve_registry_unloadable"
    assert mod.precondition_blocker(capstone_bad) == "blocked_capstone_v461_unloadable"

    _make_root(tmp_path, include_next=True)
    cited_paths = [row["path"] for row in mod.cited_upstream_artifacts(tmp_path)]
    assert "research-roadmap-next.yaml" in cited_paths
