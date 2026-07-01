"""Tests for Exp 5056 .464-to-.465 transition recording.

Spec refs: REQ-CAPSTONE-5056, SCENARIO-CAPSTONE-5056,
SCENARIO-CAPSTONE-5056-BLOCKED-YAML.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from carnot import experiment_5056_archive_464_activate_465 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_active_roadmap(root: Path, milestone: str = "2026.06.465") -> None:
    (root / mod.ROADMAP_ACTIVE_REL_PATH).write_text(
        f"milestone: {milestone}\ntasks:\n- id: exp5056-phase0\n",
        encoding="utf-8",
    )


def _capstone() -> JsonDict:
    return {
        "experiment": "experiment_5055_capstone_v464",
        "experiment_id": 5055,
        "honest_verdict": "complete_capstone_v464_execution_incomplete_fr11_no_credible_positive_evidence",
        "capstone_ready": True,
        "milestone": "2026.06.464",
        "moat_state": "execution_incomplete",
        "best_arm_and_delta": {
            "arm_id": "D1",
            "delta": 0.08,
            "ci95": [0.0, 0.165],
            "evidence_status": "blocked",
            "headline_countable": False,
        },
        "best_verifier_evidence": {
            "arm_id": "D1",
            "honest_verdict": "blocked_sota_candidate_refresh_unavailable",
            "delta_vs_tuned_sc": 0.08,
            "paired_ci95": [0.0, 0.165],
            "execution_status": "blocked",
            "proper_musr_win": False,
            "headline_countable": False,
        },
        "second_corpus_state": {
            "state": "flagged_not_counted",
            "execution_status": "flagged",
            "honest_verdict": "success_second_corpus_confirms_musr_margin_constraintbench_exact_v1_plus_0p370",
            "reported_confirmed": True,
            "headline_counted": False,
            "delta_vs_tuned_sc_second": 0.37,
            "paired_ci95_second": [0.28, 0.47],
        },
        "cascade_state": {
            "state": "blocked",
            "execution_status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "efficiency_win": False,
            "judge_call_fraction": None,
            "paired_ci95": None,
        },
        "fr11_state": "guarded_negative",
        "fr11_self_learning_result": {
            "state": "guarded_negative",
            "honest_verdict": "complete_verifier_trace_self_learning_replay_memory_minus_0p050",
            "self_learning_loop_executed": True,
            "credible_evidence": False,
            "heldout_delta": -0.05,
            "pre_update_accuracy": 0.7,
            "post_update_accuracy": 0.65,
        },
        "hardware_state": "packet_built",
        "hardware_result": {
            "state": "packet_built",
            "honest_verdict": "success_kv260_pbit_timing_ratio_packet_built",
            "kv260_ssh_reachable": True,
            "overlay_loaded": True,
            "timing_ratio_packet_built": True,
            "cpu_reference_ok": True,
            "kv260_result_ok": True,
            "claim_scope": "local_ssh_attached_kv260_python_parity_workload_only_no_general_fpga_speedup_claim",
        },
        "arc_state": "no_bank",
        "arc_result": {
            "state": "no_bank",
            "honest_verdict": "complete_tu93_no_new_level_residual_duplicate_depth",
            "new_levels_banked": 0,
            "reproducible_total_levels_after": 69,
        },
    }


def _moat_gate() -> JsonDict:
    return {
        "experiment": "experiment_5050_moat_gate_resolution_v464",
        "experiment_id": 5050,
        "honest_verdict": "complete_moat_execution_incomplete_v464_blocked_or_missing_phase_d",
        "moat_state": "execution_incomplete",
        "best_arm": "D1",
        "best_arm_delta": 0.08,
        "best_arm_ci": [0.0, 0.165],
        "second_corpus_confirmed": False,
        "cascade_efficiency_win": False,
        "blocked_upstream_artifacts": [
            {
                "arm": "powered_lora_ebm_eorm",
                "arm_id": "D1",
                "honest_verdict": "blocked_sota_candidate_refresh_unavailable",
            },
            {
                "arm": "cross_model_cascade",
                "arm_id": "D6",
                "honest_verdict": "blocked_gate_check_failed",
                "status": "blocked",
            },
        ],
        "flagged_upstream_artifacts": [
            {
                "arm": "second_corpus_confirmation",
                "arm_id": "D4",
                "honest_verdict": "success_second_corpus_confirms_musr_margin_constraintbench_exact_v1_plus_0p370",
            }
        ],
        "missing_upstream_artifacts": [
            {
                "arm": "cross_model_cascade",
                "arm_id": "D6",
                "path": "/repo/results/experiment_5048_cross_model_cascade_repair.json",
            }
        ],
        "execution_incomplete_reasons": [
            "D1 blocked: blocked_sota_candidate_refresh_unavailable",
            "D6 blocked: blocked_gate_check_failed",
        ],
        "per_arm_table": [
            {
                "arm_id": "D1",
                "execution_status": "blocked",
                "honest_verdict": "blocked_sota_candidate_refresh_unavailable",
                "delta_vs_tuned_sc": 0.08,
                "paired_ci95": [0.0, 0.165],
            }
        ],
        "cascade_artifact": {
            "arm_id": "D6",
            "execution_status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "efficiency_win": False,
        },
        "second_corpus_artifact": {
            "arm_id": "D4",
            "best_arm": "D1",
            "execution_status": "flagged",
            "honest_verdict": "success_second_corpus_confirms_musr_margin_constraintbench_exact_v1_plus_0p370",
            "second_corpus_confirmed": True,
            "delta_vs_tuned_sc_second": 0.37,
            "paired_ci95_second": [0.28, 0.47],
        },
    }


def _write_sources(root: Path, *, capstone: JsonDict | None = None, gate: JsonDict | None = None) -> None:
    _write_json(root / mod.CAPSTONE_REL_PATH, capstone or _capstone())
    _write_json(root / mod.MOAT_GATE_REL_PATH, gate or _moat_gate())


def _ok_runner(command: list[str], cwd: Path) -> mod.CommandResult:
    assert command == mod.PRETEST_COMMAND
    assert cwd.exists()
    return mod.CommandResult(command=command, exit_code=0, stdout="1 passed", stderr="", duration_s=0.01)


def _fail_runner(command: list[str], cwd: Path) -> mod.CommandResult:
    assert command == mod.PRETEST_COMMAND
    assert cwd.exists()
    return mod.CommandResult(command=command, exit_code=1, stdout="", stderr="failed", duration_s=0.01)


def _unexpected_runner(command: list[str], cwd: Path) -> mod.CommandResult:
    raise AssertionError(f"unexpected command: {command} in {cwd}")


def test_req_capstone_5056_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5056: OpenSpec anchors the transition record."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-5056",
        "SCENARIO-CAPSTONE-5056",
        "SCENARIO-CAPSTONE-5056-BLOCKED-YAML",
        "experiment_5056_archive_464_activate_465.py",
        "results/experiment_5056_archive_464_activate_465.json",
        "blocked_sota_candidate_refresh_unavailable",
        "guarded-negative",
    ):
        assert marker in spec


def test_scenario_capstone_5056_records_honest_execution_incomplete_transition(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5056: the .464 close-state is not a moat claim."""

    _write_active_roadmap(tmp_path)
    _write_sources(tmp_path)

    artifact = mod.run(root=tmp_path, command_runner=_ok_runner, started_s=1.0, now_s=2.0)

    assert artifact["honest_verdict"] == (
        "complete_464_archived_465_activated_execution_incomplete_not_moat_claim"
    )
    assert artifact["prior_milestone"] == "2026.06.464"
    assert artifact["next_milestone"] == "2026.06.465"
    assert artifact["prior_capstone_verdict"] == _capstone()["honest_verdict"]
    assert artifact["moat_state"] == "execution_incomplete"
    assert artifact["moat_claim"] is False
    assert artifact["activation_ready"] is True
    assert artifact["transition_performed"] is True
    assert artifact["transition"]["active_milestone_confirmed"] == "2026.06.465"
    assert artifact["transition"]["pre_staged_roadmap_status"] == "absent_already_promoted"
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["d1_state"]["state"] == "blocked_candidate_refresh"
    assert artifact["d1_state"]["delta_vs_tuned_sc"] == 0.08
    assert artifact["d1_state"]["moat_claim"] is False
    assert artifact["d4_state"]["state"] == "flagged_not_counted"
    assert artifact["d4_state"]["reported_confirmed"] is True
    assert artifact["d6_state"]["state"] == "blocked"
    assert artifact["fr11_state"]["state"] == "guarded_negative"
    assert artifact["fr11_state"]["heldout_delta"] == -0.05
    assert artifact["kv260_state"]["state"] == "packet_built"
    assert artifact["arc_state"]["new_levels_banked"] == 0
    assert artifact["arc_state"]["state"] == "no_bank"
    assert mod.validate_artifact(artifact) == []
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact


def test_scenario_capstone_5056_blocks_bad_present_roadmap_yaml(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5056-BLOCKED-YAML: bad YAML blocks before sources."""

    _write_active_roadmap(tmp_path)
    (tmp_path / mod.ROADMAP_NEXT_REL_PATH).write_text("milestone: [", encoding="utf-8")

    artifact = mod.run(root=tmp_path, command_runner=_unexpected_runner, started_s=1.0, now_s=1.5)

    assert artifact["honest_verdict"] == "blocked_yaml_parse"
    assert artifact["activation_ready"] is False
    assert artifact["moat_claim"] is False
    assert artifact["pretest_gate"]["ran"] is False
    assert artifact["transition"]["transition_performed"] is False
    assert artifact["preconditions_checked"]["roadmaps"]["pre_staged"]["parse_ok"] is False
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_5056_blocks_missing_fields_and_failed_pretest(
    tmp_path: Path,
    capsys: Any,
) -> None:
    """REQ-CAPSTONE-5056: missing fields and red own-tests block explicitly."""

    _write_active_roadmap(tmp_path)
    broken_capstone = _capstone()
    del broken_capstone["best_verifier_evidence"]["delta_vs_tuned_sc"]
    _write_sources(tmp_path, capstone=broken_capstone)

    missing = mod.run(root=tmp_path, command_runner=_unexpected_runner, started_s=1.0, now_s=1.1)

    assert missing["honest_verdict"] == "blocked_missing_required_field"
    assert missing["activation_ready"] is False
    assert {
        "path": str(mod.CAPSTONE_REL_PATH),
        "field": "best_verifier_evidence.delta_vs_tuned_sc",
    } in missing["missing_required_fields"]

    _write_sources(tmp_path)
    pretest_failed = mod.run(root=tmp_path, command_runner=_fail_runner, started_s=1.0, now_s=1.2)

    assert pretest_failed["honest_verdict"] == "blocked_pretest_gate_failed"
    assert pretest_failed["pretest_gate"]["green"] is False
    assert pretest_failed["activation_ready"] is False

    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact({"honest_verdict": "bad"})
    assert "invalid_moat_claim" in mod.validate_artifact({"honest_verdict": "blocked_test", "moat_claim": True})

    exit_code = mod.main(root=tmp_path, command_runner=_ok_runner)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip().endswith(str(mod.OUTPUT_REL_PATH))


def test_req_capstone_5056_resource_edge_helpers(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5056: parser and blocker edge states are explicit."""

    command = [sys.executable, "-c", "print('ok')"]
    result = mod.run_command(command, tmp_path)
    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"

    payload, missing_status = mod._read_json_mapping(tmp_path / "missing.json")
    assert payload == {}
    assert missing_status["error"] == "missing"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._read_json_mapping(bad_json)[1]["loadable"] is False

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._read_json_mapping(list_json)[1]["error"] == "json_not_object"

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    empty_payload, empty_status = mod._parse_yaml_status(
        tmp_path,
        Path("empty.yaml"),
        absent_status="missing",
    )
    assert empty_payload == {}
    assert empty_status["parse_ok"] is True

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    assert mod._parse_yaml_status(tmp_path, Path("list.yaml"), absent_status="missing")[1][
        "error"
    ] == "yaml_not_mapping"

    assert mod.roadmap_blocker({"active": {"parse_ok": None}, "pre_staged": {}}) == (
        "blocked_missing_active_roadmap"
    )
    assert mod.source_blocker({"source": {"exists": False}}) == (
        "blocked_missing_required_artifact"
    )
    assert mod.source_blocker({"source": {"exists": True, "loadable": False}}) == (
        "blocked_unloadable_required_artifact"
    )

    top_missing_capstone = _capstone()
    del top_missing_capstone["moat_state"]
    top_missing = mod.missing_required_fields(
        {"CAPSTONE": top_missing_capstone, "MOAT_GATE": _moat_gate()}
    )
    assert {"path": str(mod.CAPSTONE_REL_PATH), "field": "moat_state"} in top_missing

    bad_nested_capstone = _capstone()
    bad_nested_capstone["best_arm_and_delta"] = []
    nested_missing = mod.missing_required_fields(
        {"CAPSTONE": bad_nested_capstone, "MOAT_GATE": _moat_gate()}
    )
    assert {"path": str(mod.CAPSTONE_REL_PATH), "field": "best_arm_and_delta"} in nested_missing
    assert mod._find_arm_row([], "D9") == {}
