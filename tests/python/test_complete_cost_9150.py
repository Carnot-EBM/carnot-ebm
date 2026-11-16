"""REQ-REPORT-9150-V779; SCENARIO-REPORT-9150-V779: inventory is not service timing."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting import complete_cost_9150 as d
from carnot.reporting.current_work_receipt import atomic_json


def test_missing_and_zero_operands(tmp_path: Path) -> None:
    """REQ-REPORT-9150-V779: canonical absence cannot inherit another producer's work."""
    path = tmp_path / d.INPUTS[9142][0]
    atomic_json(
        path,
        dict(
            experiment_id=9142,
            required_checks_passed=True,
            logit_runtime_ready_score=0,
            shard_feasible_score=True,
        ),
    )
    work = d.collect(tmp_path, tmp_path / "raw")
    value = d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")
    assert value["experiment_id"] == 9150 and value["task_id"] == "exp9150-complete-cost"
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["requested_inference_substrate_class"] == "model_load_no_generation"
    assert value["model_invocation_counts"]["model_loads_attempted"] == 0
    assert value["generated_tokens"] == value["current_llm_invocations"] == 0
    assert value["MODEL_SPECS"][0]["invocation_status"] == "not_attempted"
    checks = value["gate_check_summary"]
    assert {type(c["observed"]) for c in checks} >= {str, int, bool}
    assert {"missing_file", "missing_field"} <= {str(c["observed"]) for c in checks}
    assert any(
        c["artifact_field"] == "logit_runtime_ready_score"
        and c["observed"] == 0
        and not c["passed"]
        for c in checks
    )
    assert any(
        c["artifact_field"] == "shard_feasible_score" and c["observed"] is True and not c["passed"]
        for c in checks
    )
    assert not any(value["branch_eligibility"].values())
    assert value["cost_inventory_ready_score"] == 1
    assert value["service_cost_ready_score"] == value["adaptation_cost_ready_score"] == 0
    assert value["energy_joules_or_null"] is value["amdahl_ceiling_or_null"] is None
    assert value["amdahl_target_requirements"][0]["minimum_accelerated_fraction"] == 0.99
    assert value["amdahl_target_requirements"][1]["minimum_accelerated_fraction"] == 0.9
    assert all(
        r["measured_fraction"] is None and r["unavailable_reason"]
        for r in value["hardware_operation_rows"]
    )
    assert all(r["task"] == 9150 for r in value["rows"])
    assert d.eligibility({9138: True, 9140: True, 9142: True, 9145: True, 9146: True}) == dict(
        static=True, online=False
    )
    work["eligibility"]["static"] = True
    with pytest.raises(ValueError, match="requires_measurement"):
        d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")


def test_original_panel_all_arms_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9150-V779: natural original cohort slots survive missing services."""
    work = d.collect(d.ROOT, tmp_path / "raw")
    value = d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")
    assert len(value["selected_units"]) == 60
    assert len({u["qid"] for u in value["selected_units"]}) == 20
    assert len(value["cost_rows"]) == 60 * 11 * 3 * 2 + 3
    assert len({r["unit_id"] for r in value["rows"]}) == len(value["rows"])
    assert {r["arm"] for r in value["cost_rows"]} >= set(d.PROTOCOL["online_arms"])
    assert all(r["complete_clock_s"] is None and r["censor_reason"] for r in value["cost_rows"])
    assert value["sample_size_budget"]["intended"] == len(value["cost_rows"])
    assert value["sample_size_budget"]["attempted"] == 0
    assert value["sample_size_budget"]["effective_independent_groups"] == 0
    assert value["sample_size_budget"]["unstarted"] == len(value["cost_rows"])
    assert value["protocol"]["arm_order_seed"] == 776108
    assert value["protocol"]["parity_atol"] == 1e-5
    assert value["protocol"]["parity_rtol"] == 1e-4
    assert value["claim_scope"]["independent_generalization_score"] == 0
    assert value["claim_scope"]["exposure_scope"]
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert d.replay(path)["passed"]
    changed = deepcopy(value)
    changed["service_cost_ready_score"] = 1
    changed["reproducibility_checksum"] = d.checksum(changed)
    atomic_json(path, changed)
    assert not d.replay(path)["passed"]
    value["reproducibility_checksum"] = "bad"
    atomic_json(path, value)
    assert d.replay(path)["reason"] == "checksum_drift"
    blocked = d.build(work, dict(passed=False, receipts=[], coverage={}), "20261116")
    assert blocked["verdict_class"] == "disqualified"


def test_real_cli_private_replay_refusal_and_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9150-V779 / E2E-018/021: actual CLI needs no PYTHONPATH."""
    from carnot.reporting.complete_cost_runner_9136 import plan

    assert plan(tmp_path / "checks", producer=d)
    output = tmp_path / (d.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [str(d.ROOT / ".venv/bin/python"), str(d.ROOT / d.CLI)]

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv + list(args), cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90
        )

    result = run("--date", "20261116", "--sources", str(tmp_path), "--private-output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["fixture_only"] and value["cost_inventory_ready_score"] == 0
    assert run("--cold-replay", str(output)).returncode == 0
    assert run("--date", "bad").returncode == 2
    assert run("--date", "20261116", "--private-output", str(d.ROOT / "bad.json")).returncode == 2
    value["cost_inventory_ready_score"] = 1
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(output, value)
    assert run("--cold-replay", str(output)).returncode == 1
