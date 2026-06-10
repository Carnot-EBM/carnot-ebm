"""Tests for Exp 3985 .368 GAP-4 execution-verifier capstone.

Spec refs: REQ-CAPSTONE-3985, SCENARIO-CAPSTONE-3985.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v368_3985 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_payloads() -> dict[int, JsonDict]:
    return {
        3975: {
            "honest_verdict": "complete: gap4_positive_control_failed_auroc0.00",
            "positive_control_passed": False,
            "program_synthesis_coverage": 0.0,
            "duration_s": 0.075,
            "inference_substrate": "dsl-only",
        },
        3976: {
            "honest_verdict": "blocked_gate_check_failed",
            "duration_s": 0.0,
        },
        3978: {
            "honest_verdict": "success: verifier_earns_place_efficiency_parity_8789.7x_cheaper",
            "accuracy_parity": True,
            "cost_ratio_judge_over_verifier": 8789.706,
            "verifier_actually_invoked": True,
            "duration_s": 31.8,
            "inference_substrate": "offline_arc_agi3_plus_local_gemma4_gguf_judge",
        },
        3979: {
            "honest_verdict": "complete: exec_guided_trustworthy_0of6",
            "n_trustworthy_at_0.15": 0,
            "positive_control_passed": True,
            "duration_s": 61.8,
            "inference_substrate": "offline_arc_agi3_execution_guided_program_synthesis_exact_replay_consistency_verified",
        },
        3980: {
            "honest_verdict": "complete: l2_wall_holds_r11l_l2_re_induction",
            "new_levels_solved_this_task": 0,
            "ACCURACY_levels_solved": 1,
            "duration_s": 0.678,
            "inference_substrate": "offline_arc_agi3_per_level_execution_guided_reinduction",
        },
        3981: {
            "honest_verdict": "complete: fourth_game_no_solve_budget_exceeded",
            "game_solved": "none",
            "ACCURACY_levels_solved": 0,
            "real_env_confirmed": True,
            "duration_s": 0.0,
            "inference_substrate": "offline_arc_agi3_perception_planner_real_env_confirmed",
        },
        3982: {
            "honest_verdict": "success: arcmemo_solve_transfer_2668to17_actions",
            "solve_transfer_win": True,
            "duration_s": 1.9,
            "inference_substrate": "offline_arc_agi3_real_env_steps_plus_gamegraph_arcmemo_concept_memory",
        },
        3983: {
            "honest_verdict": "complete: hardware_continuity_3983",
            "duration_s": 7.555,
            "inference_substrate": "hardware_smoke",
        },
        3984: {
            "honest_verdict": "complete: retro_commit_detector_fixed_backfill_counts_restored",
            "duration_s": 0.343,
            "inference_substrate": "git_history_added_terminal_artifact_scan",
        },
    }


def _write_artifacts(root: Path, payloads: dict[int, JsonDict]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for experiment_id, payload in payloads.items():
        path = root / "results" / f"experiment_{experiment_id}_fixture.json"
        _write_json(path, payload)
        paths[experiment_id] = path
    return paths


def _summary_statuses(ids: list[int] | tuple[int, ...]) -> dict[int, JsonDict]:
    return {
        experiment_id: {
            "returncode": 0,
            "stdout": f"summarized {experiment_id}",
            "stderr": "",
        }
        for experiment_id in ids
    }


def test_req_capstone_3985_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-3985: OpenSpec declares the .368 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-3985" in spec
    assert "SCENARIO-CAPSTONE-3985" in spec
    assert "verifier_earns_accuracy" in spec
    assert "cost_ratio_judge_over_verifier >= 10" in spec


def test_scenario_capstone_3985_actual_shape_records_missing_accuracy_and_efficiency_win(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-3985: missing exp3977 is recorded while efficiency can still earn."""

    paths = _write_artifacts(tmp_path, _artifact_payloads())

    artifact = mod.build_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(tuple(paths)),
        started_s=10.0,
        now_s=12.5,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["verifier_earns_accuracy"] is False
    assert artifact["verifier_earns_efficiency"] is True
    assert artifact["gap4_program_synthesis_coverage"] == 0.0
    assert artifact["n_trustworthy_world_models"] == 0
    assert artifact["total_games_solved"] == 3
    assert artifact["total_new_levels_this_milestone"] == 0
    assert artifact["arcmemo_solve_transfer_win"] is True
    assert artifact["missing_upstream_artifacts"] == [{"experiment_id": 3977}]
    assert artifact["flagged_artifacts_skipped"] == []
    assert artifact["duration_s"] == 2.5
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"]["verifier_earns_efficiency"].startswith("BARE BOOL")

    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert set(cited) == set(paths)
    assert cited[3978]["fields_imported"] == [
        "accuracy_parity",
        "cost_ratio_judge_over_verifier",
        "verifier_actually_invoked",
    ]
    assert cited[3975]["sha256"] == hashlib.sha256(paths[3975].read_bytes()).hexdigest()
    assert artifact["upstream_artifact_state"]["3977"]["exists"] is False
    mod.validate_artifact(artifact)


def test_req_capstone_3985_accuracy_gate_needs_both_gap4_artifacts(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3985: the accuracy axis requires exp3976 and exp3977 confirmation."""

    assert mod.trustworthy_world_models({3979: {"positive_control_passed": False, "n_trustworthy_at_0.15": 6}}) == 0

    payloads = _artifact_payloads()
    payloads[3976] = {
        "honest_verdict": "success: gap4_beats_vote",
        "gap4_beats_vote": True,
        "executed_consistency_pass2": True,
        "headroom_capture_fraction": 0.58,
        "program_synthesis_coverage": 0.71,
    }
    payloads[3977] = {
        "honest_verdict": "success: gap4_positive_confirmed",
        "gap4_positive_confirmed": True,
    }
    payloads[3980]["new_levels_solved_this_task"] = 2
    payloads[3981].update(
        {
            "honest_verdict": "success: fourth_game_solved",
            "game_solved": "su15-1944f8ab",
            "ACCURACY_levels_solved": 1,
            "real_env_confirmed": True,
        }
    )
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["verifier_earns_accuracy"] is True
    assert artifact["gap4_program_synthesis_coverage"] == 0.71
    assert artifact["gap4_headroom_capture_fraction"] == 0.58
    assert artifact["total_games_solved"] == 4
    assert artifact["total_new_levels_this_milestone"] == 3
    assert artifact["fourth_game_solved"] is True


def test_req_capstone_3985_flagged_or_not_invoked_efficiency_counts_false(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3985: flagged or not-invoked results cannot satisfy owed-axis gates."""

    payloads = _artifact_payloads()
    payloads[3978] = {
        **payloads[3978],
        "flagged_adversarial": True,
        "accuracy_parity": True,
        "cost_ratio_judge_over_verifier": 99999.0,
        "verifier_actually_invoked": True,
    }
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["verifier_earns_efficiency"] is False
    assert artifact["efficiency_accuracy_parity"] is False
    assert artifact["efficiency_cost_ratio_judge_over_verifier"] == 0.0
    assert artifact["flagged_artifacts_skipped"] == [
        {
            "experiment_id": 3978,
            "path": "results/experiment_3978_fixture.json",
            "reason": "flagged_adversarial:true",
        }
    ]
    assert 3978 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}

    payloads = _artifact_payloads()
    payloads[3978]["verifier_actually_invoked"] = False
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["verifier_earns_efficiency"] is False
    assert artifact["efficiency_accuracy_parity"] is True
    assert artifact["efficiency_verifier_actually_invoked"] is False


def test_req_capstone_3985_write_artifact_and_validate_rejects_regressions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3985: artifact writing validates required bare fields."""

    paths = _write_artifacts(tmp_path, _artifact_payloads())
    output = mod.write_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(tuple(paths)),
        started_s=1.0,
        now_s=1.125,
    )

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["schema"] == "carnot.capstone_v368_3985.v1"
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    bad = dict(written)
    bad["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["total_games_solved"] = True
    with pytest.raises(ValueError, match="bare int"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["verifier_earns_accuracy"] = "false"
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["gap4_program_synthesis_coverage"] = True
    with pytest.raises(ValueError, match="bare float"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["cited_upstream_artifacts"] = [{"experiment_id": 3975, "fields_imported": []}]
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)


def test_req_capstone_3985_summary_runner_uses_mandated_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-3985: upstream reading shells through summarize_artifact.py."""

    path = tmp_path / "results" / "experiment_3975_fixture.json"
    _write_json(path, _artifact_payloads()[3975])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["text"] is True
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(command, 0, stdout="summary", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    status = mod.run_summarize_artifact(tmp_path, path)

    assert status == {"returncode": 0, "stdout": "summary", "stderr": ""}
    assert calls == [[str(mod.PYTHON_BIN), "scripts/summarize_artifact.py", str(path)]]

    monkeypatch.setattr(
        mod,
        "run_summarize_artifact",
        lambda root, artifact_path: {
            "returncode": 0,
            "stdout": f"summary for {artifact_path.name}",
            "stderr": "",
        },
    )
    statuses = mod.summarize_existing_artifacts(tmp_path, {3975: path, 3976: None}, supplied=None)
    assert statuses == {
        3975: {
            "returncode": 0,
            "stdout": "summary for experiment_3975_fixture.json",
            "stderr": "",
        }
    }


def test_scenario_capstone_3985_script_wrapper_exists() -> None:
    """SCENARIO-CAPSTONE-3985: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/experiment_3985_capstone_v368.py")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "capstone_v368_3985" in text
    assert "write_artifact" in text
