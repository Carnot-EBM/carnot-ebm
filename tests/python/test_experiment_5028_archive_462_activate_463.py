"""Tests for REQ-CAPSTONE-5028 / SCENARIO-CAPSTONE-5028."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5028_archive_462_activate_463 as mod


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
    return {
        "arc_deliverable_locked": {
            "arc_work_mode": "opportunistic",
            "deliverable": "levels_69_plus_publishable_fover_paper",
            "locked": True,
        },
        "best_arm_and_delta": {
            "arm": "EBRM",
            "arm_id": "D3",
            "corpus": "MuSR",
            "delta_vs_tuned_sc": None,
            "execution_status": "blocked",
            "headroom_present": False,
            "source_experiment_id": 5019,
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
        "efficiency_win": False,
        "flagged_artifacts_skipped": [
            {
                "experiment_id": 5017,
                "honest_verdict": "blocked_trainable_qwen_base",
                "path": "results/experiment_5017_lora_ebm_scorer_musr_v2.json",
                "reason": "flagged_adversarial",
                "source": "D1_LORA_EBM",
            },
            {
                "experiment_id": 5018,
                "honest_verdict": "blocked_b2_logprob_cache",
                "path": "results/experiment_5018_uprm_replication_v2.json",
                "reason": "flagged_adversarial",
                "source": "D2_UPRM",
            },
            {
                "experiment_id": 5020,
                "honest_verdict": "blocked_judge_server",
                "path": "results/experiment_5020_uncertainty_routed_cascade.json",
                "reason": "flagged_adversarial",
                "source": "D6_CASCADE",
            },
            {
                "experiment_id": 5021,
                "honest_verdict": "blocked_no_best_verifier",
                "path": "results/experiment_5021_moat_second_corpus_v2.json",
                "reason": "flagged_adversarial",
                "source": "D4_SECOND_CORPUS",
            },
        ],
        "honest_verdict": "complete_capstone_v462_moat_execution_incomplete_ebrm",
        "infra_rollup": {
            "b1_genuine_sc_baseline": {
                "built": True,
                "degeneracy_guard_fires": True,
                "genuine_headroom_present": True,
                "genuine_tuned_sc_accuracy": 0.585,
                "honest_verdict": "success_genuine_sc_baseline_fixed_degeneracy_guard_shipped",
                "oracle_at_k": 0.865,
                "status": "complete",
            },
            "b2_logprob_cache": {
                "cache_built": False,
                "has_per_token_logprobs": False,
                "honest_verdict": "blocked_generation_or_cache_error",
                "n_cached_rows": 0,
                "status": "present_not_complete",
            },
        },
        "moat_verdict": {
            "decision": "EXECUTION-INCOMPLETE",
            "execution_incomplete_arms": [
                {
                    "arm": "EBRM",
                    "arm_id": "D3",
                    "corpus": "MuSR",
                    "execution_status": "blocked",
                    "honest_verdict": "blocked_gate_check_failed",
                    "source_experiment_id": 5019,
                }
            ],
            "moat_realized": False,
            "moat_retired_bounded": False,
            "source": "D5_MOAT_GATE",
            "state": "execution_incomplete",
            "summary": "D5 is execution-incomplete; this is not a clean null.",
        },
        "next_milestone_pointer": {
            "arm_id": "D3",
            "best_arm": "EBRM",
            "direction": "rerun_unexecuted_arm",
            "milestone": "2026.06.463",
            "plan": "Re-run the unexecuted or pre-gated arm; route off Codex if the same arm bails twice.",
        },
        "per_arm_table": [
            {
                "arm": "EBRM",
                "arm_id": "D3",
                "corpus": "MuSR",
                "execution_status": "blocked",
                "headroom_present": False,
                "honest_verdict": "blocked_gate_check_failed",
                "n_questions": 0,
                "source_experiment_id": 5019,
                "verifier_is_oracle": False,
                "win_vs_tuned_sc": False,
            }
        ],
        "reproducible_total_levels": 69,
    }


def _b1_artifact() -> JsonDict:
    return {
        "degeneracy_guard_fires": True,
        "genuine_headroom_present": True,
        "genuine_tuned_sc_accuracy": 0.585,
        "harness_module_path": "python/carnot/moat_benchmark_harness.py",
        "honest_verdict": "success_genuine_sc_baseline_fixed_degeneracy_guard_shipped",
        "no_new_llm_generation": True,
        "oracle_at_k": 0.865,
        "result_path": "results/experiment_5015_genuine_sc_baseline_fix.json",
    }


def _d1_artifact() -> JsonDict:
    return {
        "flagged_adversarial": True,
        "honest_verdict": "blocked_trainable_qwen_base",
        "model_specs": {"base_model": "Qwen/Qwen3.5-1.7B"},
        "preconditions_checked": [
            {
                "available": False,
                "detail": "Qwen/Qwen3.5-1.7B not cached and download failed: RepositoryNotFoundError: 404 Client Error.",
                "resource": "trainable_qwen_base",
            },
            {"available": True, "detail": "torch.cuda.is_available=true", "resource": "cuda"},
        ],
        "scorer_trained": False,
        "train_loss": None,
    }


def _b2_artifact() -> JsonDict:
    return {
        "candidate_cache_built": False,
        "duration_s": 379.158895,
        "honest_verdict": "blocked_generation_or_cache_error",
        "n_cached_rows": 0,
        "preconditions_checked": [
            {"available": True, "resource": "gemma_gguf_cache"},
            {"available": True, "resource": "llama_server_logprobs"},
            {"available": True, "resource": "musr_corpus"},
            {
                "available": False,
                "detail": "UprmScoringError: marker completion lacked top_logprobs",
                "resource": "generation_or_cache_error",
            },
        ],
    }


def _d2_artifact() -> JsonDict:
    return {
        "flagged_adversarial": True,
        "honest_verdict": "blocked_b2_logprob_cache",
        "preconditions_checked": [
            {
                "available": False,
                "detail": "RuntimeError: only 0 uPRM-ready B2 cache rows available; need 200",
                "resource": "b2_logprob_cache",
            }
        ],
    }


def _d3_artifact() -> JsonDict:
    return {
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": "1 of 1 gate(s) failed; first failure: exp5017-d1.scorer_trained (actual=False == expected=True)",
        "gates_evaluated": [
            {
                "actual": False,
                "artifact_field": "scorer_trained",
                "expected": True,
                "passed": False,
                "upstream": "exp5017-d1",
            }
        ],
        "honest_verdict": "blocked_gate_check_failed",
        "status": "blocked",
    }


def _d6_artifact() -> JsonDict:
    return {"flagged_adversarial": True, "honest_verdict": "blocked_judge_server"}


def _d4_artifact() -> JsonDict:
    return {
        "flagged_adversarial": True,
        "honest_verdict": "blocked_no_best_verifier",
        "preconditions_checked": [
            {"available": False, "resource": "d1_verifier"},
            {"available": False, "resource": "d2_verifier"},
            {"available": False, "resource": "d3_verifier"},
        ],
    }


def _make_root(
    root: Path,
    *,
    include_active: bool = True,
    include_next: bool = False,
    include_capstone: bool = True,
    active_milestone: str = "2026.06.463",
) -> None:
    roadmap_text = (
        f"milestone: {active_milestone}\n"
        "note: 'PHASE D remains the MAJORITY lever for .463; ARC LOCKED at levels 69; "
        "third PHASE D execution attempt; energy-as-ARC S0 CONCLUDED; verifier-as-reward retired.'\n"
        "pointer: 'rerun_unexecuted_arm; route off Codex if the same arm bails twice.'\n"
    )
    if include_active:
        _write_text(root / "research-roadmap.yaml", roadmap_text)
    if include_next:
        _write_text(root / "research-roadmap-next.yaml", "milestone: 2026.06.463\n")
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 69\n")
    _write_text(root / mod.HARNESS_REL_PATH, "# moat harness\n")
    if include_capstone:
        _write_json(root / mod.CAPSTONE_REL_PATH, _capstone_artifact())
    _write_json(root / mod.B1_REL_PATH, _b1_artifact())
    _write_json(root / mod.D1_REL_PATH, _d1_artifact())
    _write_json(root / mod.B2_REL_PATH, _b2_artifact())
    _write_json(root / mod.D2_REL_PATH, _d2_artifact())
    _write_json(root / mod.D3_REL_PATH, _d3_artifact())
    _write_json(root / mod.D6_REL_PATH, _d6_artifact())
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


def test_req_capstone_5028_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5028: OpenSpec declares the .462/.463 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5028") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert str(mod.OUTPUT_REL_PATH) in section
    for field in mod.REQUIRED_FIELDS:
        assert field in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    assert "Qwen/Qwen3.5-1.7B" in section
    assert "379s zero-row logprob cache" in section
    assert mod.PRETEST_COMMAND == [
        ".venv/bin/pytest",
        "tests/python/test_experiment_5028_archive_462_activate_463.py",
        "-q",
        "--no-cov",
    ]


def test_scenario_capstone_5028_complete_transition_records_462_close_state(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5028: active .463 roadmap records the true .462 close-state."""

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
    assert artifact["honest_verdict"] == (
        "complete_462_archived_463_activated_phase_d_third_attempt"
    )
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["phase_d_third_execution_attempt"] is True
    assert artifact["transition"]["active_milestone_confirmed"] == "2026.06.463"
    assert artifact["transition"]["active_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["transition"]["activation_state"] == "already_active_or_activated_463"
    assert artifact["transition_performed"] is True
    assert artifact["pretest_gate"]["ran"] is True
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["phase_d_majority_lever"] is True
    assert artifact["arc_locked"] is True
    assert artifact["leaderboard_submission"] is False
    assert artifact["reproducible_total_levels"] == 69

    causes = artifact["prior_milestone_root_causes"]
    assert [row["source"] for row in causes] == ["D1_LORA_EBM", "B2_LOGPROB_CACHE"]
    assert causes[0]["base_model"] == "Qwen/Qwen3.5-1.7B"
    assert "RepositoryNotFoundError" in causes[0]["evidence"]
    assert causes[1]["n_cached_rows"] == 0
    assert causes[1]["duration_s"] == 379.158895
    assert causes[1]["cascaded_to"] == ["D2", "D3"]

    b1 = artifact["reusable_b1_baseline"]
    assert b1["source_experiment_id"] == 5015
    assert b1["genuine_tuned_sc_accuracy"] == 0.585
    assert b1["oracle_at_k"] == 0.865
    assert b1["headroom_delta"] == 0.28
    assert b1["headroom_present"] is True
    assert b1["degeneracy_guard_fires"] is True
    assert b1["harness_module_path"] == "python/carnot/moat_benchmark_harness.py"
    assert b1["reuse_action"] == "reuse_not_rebuild"

    close = artifact["close_state_462"]
    assert close["capstone"]["honest_verdict"] == "complete_capstone_v462_moat_execution_incomplete_ebrm"
    assert close["moat_verdict"]["decision"] == "EXECUTION-INCOMPLETE"
    assert close["moat_verdict"]["moat_realized"] is False
    assert close["moat_verdict"]["moat_retired_bounded"] is False
    assert close["next_milestone_pointer"]["direction"] == "rerun_unexecuted_arm"
    assert close["next_milestone_pointer"]["route_off_codex_if_same_arm_bails_twice"] is True
    assert close["d1_404_base"]["scientific_null"] is False
    assert close["b2_zero_row_logprob_cache"]["scientific_null"] is False
    assert close["d2_cascade_block"]["blocked_on"] == "b2_logprob_cache"
    assert close["d3_cascade_block"]["gated_on"] == "D1.scorer_trained"
    assert [row["experiment_id"] for row in close["flagged_adversarial_skipped"]] == [
        5017,
        5018,
        5020,
        5021,
    ]
    assert close["phase_d"]["third_execution_attempt"] is True
    assert close["do_not_queue"] == ["energy-as-ARC", "verifier-as-reward"]
    assert artifact["diffusiongemma_gate_status"]["status"] == "STILL-PENDING"
    assert artifact["diffusiongemma_gate_status"]["autonomously_flipped_to_met"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5028_blocked_missing_roadmaps_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5028-BLOCKED-PRECONDITION: neither roadmap parses."""

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
    assert artifact["phase_d_third_execution_attempt"] is False
    assert artifact["close_state_462"]["reproducible_total_levels"] == 69
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5028_pretest_failure_blocks_after_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5028: red pre-test gate is recorded, not hidden."""

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


def test_scenario_capstone_5028_missing_capstone_blocks_before_pretest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5028-BLOCKED-PRECONDITION: missing close-state blocks."""

    _make_root(tmp_path, include_capstone=False)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=2.0,
        now_s=2.25,
    )

    assert artifact["honest_verdict"] == "blocked_capstone_v462_missing"
    assert len(calls) == 2
    assert artifact["pretest_gate"]["ran"] is False
    assert artifact["transition_performed"] is False
    assert artifact["close_state_462"]["capstone"]["honest_verdict"] == ""
    assert artifact["phase_d_third_execution_attempt"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5028_resource_blockers_and_optional_next(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-5028: malformed resources fail closed without fabrication."""

    bad_json = tmp_path / "bad.json"
    bad_yaml = tmp_path / "bad.yaml"
    bad_json.write_text("{", encoding="utf-8")
    bad_yaml.write_text("bad: [", encoding="utf-8")

    assert mod._read_json_object_safe(bad_json) == {}
    assert mod._read_yaml_object_safe(bad_yaml) == {}
    assert mod._resource_detail({"preconditions_checked": []}, "missing") == ""
    assert (
        mod._first_failed_resource_detail(
            {"preconditions_checked": [{"available": True, "detail": "ok"}]}
        )
        == ""
    )
    assert mod._d3_gated_on({"gates_evaluated": [{"upstream": "other"}]}) == ""

    ok = {
        "active_roadmap_yaml": {"passed": True, "active_exists": True, "next_exists": False},
        "offline_arcade": {"passed": True},
        "registry": {"exists": True, "loadable": True},
        "capstone_v462": {"exists": True, "loadable": True},
    }
    roadmap_bad = {**ok, "active_roadmap_yaml": {"passed": False, "active_exists": True}}
    offline_bad = {**ok, "offline_arcade": {"passed": False}}
    registry_missing = {**ok, "registry": {"exists": False, "loadable": False}}
    registry_bad = {**ok, "registry": {"exists": True, "loadable": False}}
    capstone_bad = {**ok, "capstone_v462": {"exists": True, "loadable": False}}

    assert mod.precondition_blocker(roadmap_bad) == "blocked_roadmap_yaml_unparseable"
    assert mod.precondition_blocker(offline_bad) == "blocked_offline_arcade_unavailable"
    assert mod.precondition_blocker(registry_missing) == "blocked_arc_solve_registry_missing"
    assert mod.precondition_blocker(registry_bad) == "blocked_arc_solve_registry_unloadable"
    assert mod.precondition_blocker(capstone_bad) == "blocked_capstone_v462_unloadable"

    _make_root(tmp_path, include_next=True)
    cited_paths = [row["path"] for row in mod.cited_upstream_artifacts(tmp_path)]
    assert "research-roadmap-next.yaml" in cited_paths
    assert "python/carnot/moat_benchmark_harness.py" in cited_paths
