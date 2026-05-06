"""Spec: REQ-ISING-025, SCENARIO-ISING-035."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "openspec/capabilities/ising-backend/spec.md"
RTL_PATH = ROOT / "hardware/kv260/discrete_sb_256.v"
TB_PATH = ROOT / "hardware/kv260/discrete_sb_256_tb.v"
ARTIFACT_PATH = ROOT / "results/experiment_1441_discrete_sb_rtl_source_implementation.json"


def test_req_ising_025_spec_anchor_exists() -> None:
    """REQ-ISING-025, SCENARIO-ISING-035: source work is spec-anchored."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ISING-025" in spec
    assert "SCENARIO-ISING-035" in spec
    assert "hardware/kv260/discrete_sb_256.v" in spec


def test_discrete_sb_256_rtl_contract() -> None:
    """REQ-ISING-025: RTL exposes the minimum deterministic dSB source contract."""
    source = RTL_PATH.read_text(encoding="utf-8")

    assert "REQ-ISING-025" in source
    assert "SCENARIO-ISING-035" in source
    assert re.search(r"\bmodule\s+discrete_sb_256\b", source)
    assert "parameter integer N_VARIABLES = 256" in source
    assert "parameter integer COUPLING_BITS = 8" in source

    for port_name in (
        "clk",
        "rst",
        "start",
        "load_init",
        "init_word_index",
        "init_word_data",
        "load_coupling",
        "coupling_addr",
        "coupling_data",
        "max_steps",
        "eta_q1_15",
        "pressure_start_q1_15",
        "pressure_delta_q1_15",
        "busy",
        "done",
        "spin_out",
    ):
        assert re.search(rf"\b{port_name}\b", source), port_name

    assert "j_matrix" in source
    assert "spin_snapshot" in source
    assert "field_acc" in source
    assert "candidate_q1_15" in source


def test_discrete_sb_testbench_drives_one_update_step() -> None:
    """SCENARIO-ISING-035: testbench drives reset, inputs, and a full update."""
    testbench = TB_PATH.read_text(encoding="utf-8")

    assert "REQ-ISING-025" in testbench
    assert "SCENARIO-ISING-035" in testbench
    assert "discrete_sb_256" in testbench
    assert "load_coupling" in testbench
    assert "init_word_data" in testbench
    assert "start" in testbench
    assert "done" in testbench
    assert "SIMULATION RESULT: PASS" in testbench


def test_exp1441_artifact_records_source_and_next_command() -> None:
    """SCENARIO-ISING-035: terminal artifact has required source/probe fields."""
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    required_fields = {
        "status",
        "rtl_source_created",
        "rtl_source_path",
        "testbench_created",
        "testbench_path",
        "spec_requirements_covered",
        "syntax_probe_command",
        "commands_run",
        "honest_verdict",
    }
    assert required_fields <= set(payload)
    assert payload["status"] == "complete"
    assert payload["rtl_source_created"] is True
    assert payload["rtl_source_path"] == "hardware/kv260/discrete_sb_256.v"
    assert payload["testbench_created"] is True
    assert payload["testbench_path"] == "hardware/kv260/discrete_sb_256_tb.v"
    assert "REQ-ISING-025" in payload["spec_requirements_covered"]
    assert "SCENARIO-ISING-035" in payload["spec_requirements_covered"]
    assert "hardware/kv260/discrete_sb_256.v" in payload["syntax_probe_command"]
    assert payload["commands_run"]
    assert "board" not in payload["honest_verdict"].lower()
