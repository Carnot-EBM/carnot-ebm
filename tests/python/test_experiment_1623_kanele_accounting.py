"""Tests for Exp 1623 KANELÉ vs Ising v3 hardware accounting.

Spec refs: REQ-KAN-1623, SCENARIO-KAN-1623.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.hardware.kanele_accounting import (  # noqa: E402
    KV260_LUT_BUDGET,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_has_required_fields,
    build_artifact,
    build_ising_v3_node_accounting,
    build_kanele_node_accounting,
    count_lut6_primitives,
    estimate_clock_mhz,
    load_json,
    parse_ising_v3_documented_utilization,
    run_experiment,
)


def _kan_lut_block_text(lut_count: int = 8) -> str:
    instances = []
    for index in range(lut_count):
        instances.append(
            f"""
    LUT6 #(
        .INIT(64'h{index + 1:016X})
    ) lut_{index} (
        .O(y[{index}]),
        .I0(x[0]),
        .I1(x[1]),
        .I2(x[2]),
        .I3(x[3]),
        .I4(x[4]),
        .I5(x[5])
    );
"""
        )
    return "module kan_lut_block(input wire [5:0] x, output wire [7:0] y);\n" + "".join(
        instances
    ) + "\nendmodule\n"


def _ising_v3_text() -> str:
    return """
module ising_sampler_v3;
// N=64: fits XCK26 at 48.5% LUTs post-synthesis with v3 EMA stage.
endmodule
"""


def test_kanele_per_node_lut_consumption_uses_lut6_block() -> None:
    """REQ-KAN-1623: KANELÉ per-node LUT count derives from LUT6 edge blocks."""
    accounting = build_kanele_node_accounting(lut6_primitives_per_edge=8, fan_in=3)

    assert accounting["edge_lut6_primitives_per_edge"] == 8
    assert accounting["fan_in_edges_per_node"] == 3
    assert accounting["edge_luts_per_node"] == 24
    assert accounting["accumulator_luts_per_node"] == 16
    assert accounting["control_luts_per_node"] == 4
    assert accounting["total_luts_per_node"] == 44


def test_ising_v3_per_node_lut_consumption_uses_documented_utilization() -> None:
    """REQ-KAN-1623: Ising v3 per-node LUT count derives from the KV260 RTL comment."""
    n_nodes, utilization_pct = parse_ising_v3_documented_utilization(_ising_v3_text())
    accounting = build_ising_v3_node_accounting(n_nodes=n_nodes, utilization_pct=utilization_pct)

    assert n_nodes == 64
    assert utilization_pct == pytest.approx(48.5)
    assert accounting["kv260_lut_budget"] == KV260_LUT_BUDGET
    assert accounting["documented_total_luts"] == 56803
    assert accounting["total_luts_per_node"] == 888
    assert accounting["node_definition"] == "one Ising v3 spin update lane"


def test_logic_depth_clock_estimate_is_timing_model_only() -> None:
    """REQ-KAN-1623: clock estimates are depth based and make no timing-closure claim."""
    estimates = estimate_clock_mhz(
        {"kanele": 4, "ising_v3": 18},
        lut_delay_ns=0.35,
        register_overhead_ns=0.8,
        practical_cap_mhz=300.0,
    )

    assert estimates["kanele"]["critical_path_ns"] == pytest.approx(2.2)
    assert estimates["kanele"]["raw_max_clock_mhz"] == pytest.approx(454.545, abs=0.001)
    assert estimates["kanele"]["capped_max_clock_mhz"] == pytest.approx(300.0)
    assert estimates["ising_v3"]["critical_path_ns"] == pytest.approx(7.1)
    assert estimates["ising_v3"]["raw_max_clock_mhz"] == pytest.approx(140.845, abs=0.001)
    assert estimates["ising_v3"]["capped_max_clock_mhz"] == pytest.approx(140.845, abs=0.001)


def test_run_experiment_writes_required_accounting_artifact(tmp_path: Path) -> None:
    """SCENARIO-KAN-1623: runner writes deterministic KANELÉ vs Ising accounting JSON."""
    kan_lut_block_path = tmp_path / "kan_lut_block.v"
    ising_v3_path = tmp_path / "ising_sampler_v3.v"
    exp1621_path = tmp_path / "experiment_1621_kanele_mapping.json"
    deliverable_path = tmp_path / "experiment_1623_kanele_accounting.json"
    kan_lut_block_path.write_text(_kan_lut_block_text())
    ising_v3_path.write_text(_ising_v3_text())
    exp1621_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "spec": "REQ-KAN-1621",
                "kan_lut_verilog_ready": True,
                "lut_config_bits_generated": True,
                "kan_lut_block_written": True,
            }
        )
    )

    artifact = run_experiment(
        kan_lut_block_path=kan_lut_block_path,
        ising_v3_path=ising_v3_path,
        exp1621_path=exp1621_path,
        deliverable_path=deliverable_path,
        run_date="2026-05-09T00:00:00Z",
    )

    payload = json.loads(deliverable_path.read_text())
    assert payload == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert artifact_has_required_fields(payload)
    assert payload["per_node_lut_consumption"]["kanele"]["total_luts_per_node"] == 44
    assert payload["per_node_lut_consumption"]["ising_v3"]["total_luts_per_node"] == 888
    assert payload["per_node_lut_consumption"]["ising_to_kanele_lut_ratio"] == pytest.approx(
        20.1818,
        abs=0.0001,
    )
    assert payload["max_clock_frequency_estimate_mhz"]["kanele"] == pytest.approx(300.0)
    assert payload["max_clock_frequency_estimate_mhz"]["ising_v3"] == pytest.approx(140.845, abs=0.001)
    assert payload["hardware_claim_allowed"] is False
    assert payload["synthesis_performed"] is False
    assert payload["board_execution_performed"] is False
    assert payload["honest_verdict"].startswith("complete:")


def test_invalid_inputs_and_claim_drift_fail_clearly(tmp_path: Path) -> None:
    """REQ-KAN-1623: malformed inputs and hardware-claim drift are rejected."""
    kan_lut_block_path = tmp_path / "empty.v"
    kan_lut_block_path.write_text("module empty; endmodule\n")
    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text("[]")

    assert count_lut6_primitives(_kan_lut_block_text()) == 8
    assert not artifact_has_required_fields({})
    with pytest.raises(ValueError, match="expected at least one LUT6"):
        count_lut6_primitives(kan_lut_block_path.read_text())
    with pytest.raises(ValueError, match="could not find documented Ising v3"):
        parse_ising_v3_documented_utilization("module ising_sampler_v3; endmodule\n")
    with pytest.raises(ValueError, match="expected JSON object"):
        load_json(bad_json_path)
    with pytest.raises(ValueError, match="positive"):
        build_kanele_node_accounting(lut6_primitives_per_edge=0)

    artifact = build_artifact(
        exp1621={"status": "complete", "kan_lut_verilog_ready": True},
        kanele_node=build_kanele_node_accounting(lut6_primitives_per_edge=8),
        ising_node=build_ising_v3_node_accounting(n_nodes=64, utilization_pct=48.5),
        logic_depth={"kanele": 4, "ising_v3": 18},
        clock_estimates=estimate_clock_mhz({"kanele": 4, "ising_v3": 18}),
        run_date="2026-05-09T00:00:00Z",
        duration_s=0.0,
    )
    artifact["hardware_claim_allowed"] = True
    assert not artifact_has_required_fields(artifact)
