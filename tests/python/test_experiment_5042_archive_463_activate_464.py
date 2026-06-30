"""Tests for REQ-CAPSTONE-5042 / SCENARIO-CAPSTONE-5042."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5042_archive_463_activate_464 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _d1() -> JsonDict:
    return {
        "experiment": "experiment_5031_lora_ebm_scorer_musr_v3",
        "experiment_id": 5031,
        "honest_verdict": "complete_lora_ebm_no_win_musr_plus_0p080_ci_incl_0",
        "delta_vs_tuned_sc": 0.08,
        "paired_ci95": [0.0, 0.165],
        "mcnemar_p": 0.076369,
        "n_questions": 200,
        "genuine_tuned_sc_accuracy": 0.585,
        "trained_scorer_accuracy": 0.665,
        "scorer_trained": True,
        "headroom_present": True,
        "oracle_at_k": 0.865,
        "verifier_is_oracle": False,
    }


def _d2() -> JsonDict:
    return {
        "experiment": "experiment_5032_uprm_replication_v3",
        "experiment_id": 5032,
        "honest_verdict": "complete_uprm_no_win_musr_minus_0p110_mcnemar_or_headroom_gate",
        "delta_vs_tuned_sc": -0.11,
        "paired_ci95": [-0.195, -0.03],
        "mcnemar_p": 0.016853,
        "n_questions": 200,
        "genuine_tuned_sc_accuracy": 0.585,
        "uprm_selection_accuracy": 0.475,
        "headroom_present": True,
        "oracle_at_k": 0.865,
        "verifier_is_oracle": False,
    }


def _d3() -> JsonDict:
    return {
        "experiment": "experiment_5033_ebrm_uncertainty_verifier_v3",
        "experiment_id": 5033,
        "honest_verdict": "complete_ebrm_no_win_musr_plus_0p080_ci_incl_0",
        "delta_vs_tuned_sc": 0.08,
        "paired_ci95": [0.0, 0.165],
        "mcnemar_p": 0.076369,
        "n_questions": 200,
        "genuine_tuned_sc_accuracy": 0.585,
        "ebrm_selection_accuracy": 0.665,
        "headroom_present": True,
        "oracle_at_k": 0.865,
        "abstention_rate": 0.0,
        "verifier_is_oracle": False,
    }


def _d6() -> JsonDict:
    return {
        "experiment": "experiment_5034_uncertainty_routed_cascade_v2",
        "experiment_id": 5034,
        "honest_verdict": "blocked_judge_server",
        "blocked_error": "URLError: <urlopen error [Errno 111] Connection refused>",
        "preconditions_checked": [
            {"available": False, "resource": "judge_server", "detail": "connection refused"}
        ],
        "cascade_accuracy": None,
        "judge_call_fraction": None,
        "verifier_is_oracle": False,
    }


def _d4() -> JsonDict:
    return {
        "experiment": "experiment_5035_moat_second_corpus_v3",
        "experiment_id": 5035,
        "honest_verdict": "blocked_second_corpus_unavailable",
        "blocked_error": "no priority second corpus had enough cached headroom candidate rows",
        "preconditions_checked": [
            {"available": False, "resource": "candidate_cache_mmlu_pro_hard"}
        ],
        "delta_vs_tuned_sc_second": None,
        "paired_ci95_second": None,
        "second_corpus": None,
        "verifier_is_oracle": False,
    }


def _capstone() -> JsonDict:
    return {
        "experiment": "experiment_5041_capstone_v463",
        "experiment_id": 5041,
        "honest_verdict": "complete_capstone_v463_moat_execution_incomplete_lora_ebm",
        "moat_verdict": {
            "decision": "EXECUTION-INCOMPLETE",
            "state": "execution_incomplete",
            "moat_realized": False,
            "moat_retired_bounded": False,
            "execution_incomplete_arms": [
                {"arm_id": "D6", "honest_verdict": "blocked_judge_server"},
                {"arm_id": "D4", "honest_verdict": "blocked_second_corpus_unavailable"},
            ],
        },
        "best_arm_and_delta": {
            "arm": "LoRA-EBM",
            "arm_id": "D1",
            "corpus": "MuSR",
            "delta_vs_tuned_sc": 0.08,
            "paired_ci95": [0.0, 0.165],
            "scorer_trained": True,
        },
        "efficiency_win": False,
        "hardware_rollup": {
            "honest_verdict": "success_kv260_reachable_overlay_loaded_energy_ok",
            "kv260_reachable": True,
            "loaded_overlay": "carnot_ising_v2_n64",
            "energy_smoke": {"success": True, "energy": -7, "expected_energy": -7},
        },
        "arc_opportunistic_rollup": {
            "honest_verdict": "complete_lp85_no_new_level_residual_no_grounded_l6_delta",
            "new_levels_banked": 0,
            "reproducible_total_levels_after": 69,
        },
        "arc_deliverable_locked": {"locked": True, "arc_work_mode": "opportunistic"},
        "reproducible_total_levels": 69,
    }


def _kv260() -> JsonDict:
    return {
        "experiment": 5037,
        "honest_verdict": "success_kv260_reachable_overlay_loaded_energy_ok",
        "kv260_ssh_reachable": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "energy_smoke": {"success": True, "energy": -7, "expected_energy": -7},
        "uio_devices": ["/dev/uio0"],
    }


def _arc() -> JsonDict:
    return {
        "experiment": "experiment_5040_levelup_attempt",
        "experiment_id": 5040,
        "honest_verdict": "complete_lp85_no_new_level_residual_no_grounded_l6_delta",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 5,
        "reproducible_total_levels_after": 69,
        "target_game": "lp85",
        "target_level": 6,
    }


def _make_root(root: Path, *, active_yaml: str = "milestone: 2026.06.464\n") -> None:
    _write_text(root / "research-roadmap.yaml", active_yaml)
    _write_json(root / mod.CAPSTONE_REL_PATH, _capstone())
    _write_json(root / mod.D1_REL_PATH, _d1())
    _write_json(root / mod.D2_REL_PATH, _d2())
    _write_json(root / mod.D3_REL_PATH, _d3())
    _write_json(root / mod.D6_REL_PATH, _d6())
    _write_json(root / mod.D4_REL_PATH, _d4())
    _write_json(root / mod.KV260_REL_PATH, _kv260())
    _write_json(root / mod.ARC_LEVEL_REL_PATH, _arc())


def _runner(calls: list[list[str]], *, pretest_ok: bool = True):
    def run(command: list[str], _root: Path) -> mod.CommandResult:
        calls.append(command)
        return mod.CommandResult(command, 0 if pretest_ok else 1, "passed", "failed")

    return run


def test_req_capstone_5042_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5042: OpenSpec declares the .463/.464 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5042") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert str(mod.OUTPUT_REL_PATH) in section
    for field in mod.REQUIRED_FIELDS:
        assert field in section
    assert mod.PRETEST_COMMAND == [
        ".venv/bin/pytest",
        "tests/python/test_experiment_5042_archive_463_activate_464.py",
        "-q",
        "--no-cov",
    ]


def test_scenario_capstone_5042_complete_transition_records_real_463_close_state(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5042: active .464 roadmap records true .463 close-state."""

    _make_root(tmp_path)
    calls: list[list[str]] = []
    exit_code = mod.main(root=tmp_path, command_runner=_runner(calls))
    artifact = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert exit_code == 0
    assert calls == [mod.PRETEST_COMMAND]
    assert artifact["honest_verdict"] == (
        "complete_463_archived_464_activated_phase_d_power_confirmation"
    )
    assert artifact["prior_milestone"] == "2026.06.463"
    assert artifact["next_milestone"] == "2026.06.464"
    assert artifact["transition"]["active_milestone_confirmed"] == "2026.06.464"
    assert artifact["transition"]["active_conductor_changed"] is False
    assert artifact["transition"]["pre_staged_roadmap_status"] == "absent_already_promoted"
    assert artifact["transition_performed"] is True
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["d1_delta_vs_tuned_sc"] == 0.08
    assert artifact["d1_ci_touches_zero"] is True
    assert artifact["d6_blocked_reason"] == "blocked_judge_server"
    assert artifact["d4_blocked_reason"] == "blocked_second_corpus_unavailable"
    assert artifact["lora_ebm_signal"] == "real_but_underpowered"
    assert artifact["scalar_uprm_result"] == "negative"
    assert artifact["ebrm_result"] == "tie_with_d1"
    assert artifact["blocked_confirmation_axes"] == [
        "D6_judge_cascade",
        "D4_second_corpus",
    ]
    assert artifact["kv260_continuity_live"] is True
    assert artifact["no_arc_level_bank"] is True
    assert artifact["leaderboard_submission"] is False

    close = artifact["close_state_463"]
    assert close["moat_verdict"]["decision"] == "EXECUTION-INCOMPLETE"
    assert close["moat_verdict"]["moat_realized"] is False
    assert close["moat_verdict"]["moat_retired_bounded"] is False
    assert close["d1_lora_ebm_signal"]["real_signal"] is True
    assert close["d1_lora_ebm_signal"]["underpowered"] is True
    assert close["d2_scalar_uprm_result"]["negative_result"] is True
    assert close["d3_ebrm_result"]["tied_d1"] is True
    assert close["kv260_continuity"]["live"] is True
    assert close["arc_level_bank"]["new_levels_banked"] == 0
    assert close["arc_level_bank"]["no_arc_level_bank"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5042_bad_yaml_writes_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5042-BLOCKED-YAML: YAML parse failure is terminal."""

    _make_root(tmp_path, active_yaml="milestone: [\n")
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls),
        started_s=10.0,
        now_s=10.25,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "blocked_yaml_parse"
    assert artifact["pretest_gate"]["ran"] is False
    assert artifact["transition_performed"] is False
    assert artifact["close_state_463"] == {}
    assert artifact["preconditions_checked"]["roadmaps"]["active"]["parse_ok"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_5042_missing_required_field_blocks(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5042: missing source fields are reported, not inferred."""

    _make_root(tmp_path)
    d1 = _d1()
    d1.pop("delta_vs_tuned_sc")
    _write_json(tmp_path / mod.D1_REL_PATH, d1)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls),
        started_s=20.0,
        now_s=20.5,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "blocked_missing_required_field"
    assert artifact["pretest_gate"]["ran"] is False
    assert {
        "path": str(mod.D1_REL_PATH),
        "field": "delta_vs_tuned_sc",
    } in artifact["missing_required_fields"]
    assert mod.validate_artifact(artifact) == []
