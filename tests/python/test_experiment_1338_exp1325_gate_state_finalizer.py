"""Tests for Exp 1338 stale Exp 1325 gate-state finalization.

Spec: REQ-VERIFY-1338,
      SCENARIO-VERIFY-1338
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import exp1325_skeleton_and_gate_state_finalizer as mod


def _exp1324() -> dict[str, Any]:
    return {
        "status": "complete",
        "minimum_parseable_attempts_to_recover": 6,
        "artifact_metadata": {"parse_gate": 0.75},
        "parse_recovery_recommendation": (
            "Recover at least 6 parseable attempts before reopening exp1325."
        ),
    }


def _exp1325_skeleton() -> dict[str, Any]:
    return {
        "status": "in_progress",
        "honest_verdict": "in_progress",
        "certificate_parse_rate": None,
        "certificate_truthfulness_rate": None,
        "parse_rate_delta_over_exp1312": None,
    }


def _exp1327_blocked() -> dict[str, Any]:
    return {
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "gate_check_summary": "upstream artifact not found for exp1326",
    }


def _retro() -> dict[str, Any]:
    return {
        "status": "success",
        "improvements_suggested": [
            "Add dependency pruning so stale exp1325 closes validator, safe-prefix, DVI-tail, and GRPO/VPRM dependent tasks.",
            "Add a disk-quota and inode preflight before launching Codex.",
        ],
    }


CONDUCTOR_LOG = """
| 2026-05-05 07:25 UTC | Triggered Certificate Extraction v5 | FAIL | Codex CLI error: [Errno 122] Disk quota exceeded |
| 2026-05-05 07:31 UTC | SatIR/NSVIF Constraint Index + Semantic Validator | GATE_BLOCK | Pre-emptive skip: upstream retired exp1325 |
| 2026-05-05 07:35 UTC | BEAVER-lite/Cactus Safe-Prefix Acceptance v5 | GATE_BLOCK | exp1326 validator gate failed |
| 2026-05-05 07:44 UTC | DVI Certificate-Tail Online Update v2 | GATE_BLOCK | Pre-emptive skip: upstream retired exp1325 |
| 2026-05-05 07:47 UTC | GRPO/VPRM v12 Micro-Audit | GATE_BLOCK | upstream retired exp1328 |
"""


def test_exp1338_builds_terminal_carry_forward_for_stale_exp1325() -> None:
    """REQ-VERIFY-1338-3/4/5/6/7: stale skeleton closes downstream gates."""
    artifact = mod.build_gate_state_artifact(
        exp1324_artifact=_exp1324(),
        exp1325_artifact=_exp1325_skeleton(),
        exp1327_artifact=_exp1327_blocked(),
        retro_artifact=_retro(),
        conductor_log=CONDUCTOR_LOG,
        run_date="20260505",
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["exp1325_terminal_classification"] == "stale_skeleton_environment_failure"
    assert artifact["minimum_parseable_attempts_to_recover"] == 6
    assert artifact["parse_gate_threshold"] == 0.75
    assert artifact["certificate_recovery_ready"] is True
    assert artifact["rerun_is_materially_different"] is True
    assert artifact["stale_artifacts_not_modified"] is True
    assert artifact["evidence_summary"]["disk_quota_failures"] == 1
    assert artifact["evidence_summary"]["gate_block_rows"] == 4
    assert artifact["honest_verdict"] == (
        "exp1325_closed_as_stale_environment_skeleton_downstream_gates_remain_closed"
    )

    closed_categories = {task["category"] for task in artifact["downstream_tasks_to_keep_closed"]}
    assert closed_categories == {
        "semantic_validator",
        "safe_prefix",
        "dvi_certificate_tail",
        "grpo_vprm",
    }
    assert "trigger-before-constrain generation" in artifact["required_method_changes"]
    assert "dynamic grammar dispatch" in artifact["required_method_changes"]
    assert "semantic validation branch" in artifact["required_method_changes"]


def test_exp1338_rerun_materiality_requires_all_method_changes() -> None:
    """REQ-VERIFY-1338-6: a partial method tweak is still a blind rerun."""
    artifact = mod.build_gate_state_artifact(
        exp1324_artifact={"status": "complete", "minimum_parseable_attempts_to_recover": 6},
        exp1325_artifact=_exp1325_skeleton(),
        exp1327_artifact=_exp1327_blocked(),
        retro_artifact=_retro(),
        conductor_log=CONDUCTOR_LOG,
        proposed_method_changes=["trigger-before-constrain generation"],
    )

    assert artifact["rerun_is_materially_different"] is False
    assert artifact["certificate_recovery_ready"] is False
    assert artifact["honest_verdict"] == (
        "exp1325_closed_as_stale_environment_skeleton_waiting_on_materially_different_recovery_plan"
    )
    assert artifact["parse_gate_threshold"] is None
    assert mod.classify_exp1325_terminal_state({"certificate_parse_rate": False}) == (
        "stale_skeleton_environment_failure"
    )


def test_exp1338_completed_metrics_are_not_reclassified_as_stale() -> None:
    """REQ-VERIFY-1338-3: substantive certificate metrics prevent skeleton classification."""
    completed = _exp1325_skeleton() | {
        "status": "complete",
        "honest_verdict": "certificate_parse_gate_still_closed_runtime_fixed_v5",
        "certificate_parse_rate": 0.71,
    }
    artifact = mod.build_gate_state_artifact(
        exp1324_artifact=_exp1324(),
        exp1325_artifact=completed,
        exp1327_artifact=_exp1327_blocked(),
        retro_artifact=_retro(),
        conductor_log=CONDUCTOR_LOG,
    )

    assert artifact["exp1325_terminal_classification"] == "substantive_certificate_artifact_present"
    assert artifact["certificate_recovery_ready"] is False
    assert artifact["rerun_is_materially_different"] is False


def test_exp1338_run_experiment_writes_in_progress_then_final_without_source_edits(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1338-1/2 and SCENARIO-VERIFY-1338: source artifacts are read-only."""
    results = tmp_path / "results"
    ops = tmp_path / "ops"
    results.mkdir()
    ops.mkdir()
    paths = {
        "exp1324_path": results / "experiment_1324.json",
        "exp1325_path": results / "experiment_1325.json",
        "exp1327_path": results / "experiment_1327.json",
        "retro_path": results / "operational_retro_2026_04_103.json",
    }
    payloads = {
        paths["exp1324_path"]: _exp1324(),
        paths["exp1325_path"]: _exp1325_skeleton(),
        paths["exp1327_path"]: _exp1327_blocked(),
        paths["retro_path"]: _retro(),
    }
    for path, payload in payloads.items():
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    conductor_log_path = ops / "conductor-log.md"
    conductor_log_path.write_text(CONDUCTOR_LOG, encoding="utf-8")
    before = {path: path.read_text(encoding="utf-8") for path in payloads}

    writes: list[dict[str, Any]] = []
    output_path = results / "experiment_1338.json"
    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        output_path=output_path,
        conductor_log_path=conductor_log_path,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
        **paths,
    )

    assert writes[0]["status"] == "in_progress"
    assert writes[-1]["status"] == "complete"
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert {path: path.read_text(encoding="utf-8") for path in payloads} == before
    assert artifact["stale_artifacts_not_modified"] is True
