"""Exp 1623 KANELE vs Ising v3 theoretical KV260 accounting.

This module compares the direct LUT6 KANELE edge block from Exp 1621 with the
existing KV260 Ising v3 RTL formulation. The result is an accounting artifact:
it does not run Vivado, it does not close timing, and it does not claim board
execution.

Spec refs: REQ-KAN-1623, SCENARIO-KAN-1623.
"""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path
import re
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KAN_LUT_BLOCK_PATH = PROJECT_ROOT / "hardware" / "kv260" / "kan_lut_block.v"
ISING_V3_PATH = PROJECT_ROOT / "hardware" / "kv260" / "ising_sampler_v3.v"
EXP1621_PATH = PROJECT_ROOT / "results" / "experiment_1621_kanele_mapping.json"
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1623_kanele_accounting.json"

EXPERIMENT_ID = "1623"
RUN_DATE = "2026-05-09"
SCHEMA = "kanele_vs_ising_v3_accounting_v1"
KV260_LUT_BUDGET = 117_120
DEFAULT_KANELE_FAN_IN = 3
DEFAULT_ACCUMULATOR_BITS = 8
DEFAULT_CONTROL_LUTS_PER_NODE = 4
DEFAULT_LOGIC_DEPTH = {"kanele": 4, "ising_v3": 18}
DEFAULT_LUT_DELAY_NS = 0.35
DEFAULT_REGISTER_OVERHEAD_NS = 0.8
DEFAULT_KV260_PRACTICAL_CAP_MHZ = 300.0
TERMINAL_VERDICT_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_ARTIFACT_FIELDS = {
    "schema",
    "status",
    "experiment_id",
    "spec",
    "per_node_lut_consumption",
    "logic_depth_estimate",
    "max_clock_frequency_estimate_mhz",
    "kv260_budget",
    "hardware_claim_allowed",
    "honest_verdict",
}


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object used as local accounting evidence."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def count_lut6_primitives(verilog_text: str) -> int:
    """Count instantiated Xilinx LUT6 primitives in a Verilog module."""
    count = len(re.findall(r"^\s*LUT6\s*#\s*\(", verilog_text, flags=re.MULTILINE))
    if count <= 0:
        raise ValueError("expected at least one LUT6 primitive in KANELE LUT block")
    return count


def parse_ising_v3_documented_utilization(verilog_text: str) -> tuple[int, float]:
    """Extract the documented N and LUT-utilization percentage from Ising v3 RTL."""
    match = re.search(
        r"^\s*//\s*N\s*=\s*(?P<n_nodes>\d+)\s*:\s*.*?"
        r"(?P<utilization>[0-9]+(?:\.[0-9]+)?)%\s+LUTs",
        verilog_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        raise ValueError("could not find documented Ising v3 N and LUT utilization")
    return int(match.group("n_nodes")), float(match.group("utilization"))


def build_kanele_node_accounting(
    *,
    lut6_primitives_per_edge: int,
    fan_in: int = DEFAULT_KANELE_FAN_IN,
    accumulator_bits: int = DEFAULT_ACCUMULATOR_BITS,
    control_luts_per_node: int = DEFAULT_CONTROL_LUTS_PER_NODE,
) -> dict[str, Any]:
    """Return per-node LUT accounting for one KANELE additive node."""
    if lut6_primitives_per_edge <= 0 or fan_in <= 0:
        raise ValueError("KANELE LUT primitives and fan-in must be positive")
    edge_luts = lut6_primitives_per_edge * fan_in
    accumulator_luts = max(fan_in - 1, 0) * accumulator_bits
    total_luts = edge_luts + accumulator_luts + control_luts_per_node
    return {
        "architecture": "kanele_lut6_edge_mapping",
        "node_definition": "one KANELE additive node with fan-in edge LUT blocks",
        "edge_lut6_primitives_per_edge": int(lut6_primitives_per_edge),
        "fan_in_edges_per_node": int(fan_in),
        "edge_luts_per_node": int(edge_luts),
        "accumulator_luts_per_node": int(accumulator_luts),
        "control_luts_per_node": int(control_luts_per_node),
        "total_luts_per_node": int(total_luts),
        "derivation": (
            "fan_in * LUT6_per_edge + (fan_in - 1) * accumulator_bits "
            "+ control_luts"
        ),
    }


def build_ising_v3_node_accounting(
    *,
    n_nodes: int,
    utilization_pct: float,
    kv260_lut_budget: int = KV260_LUT_BUDGET,
) -> dict[str, Any]:
    """Return per-node LUT accounting for the documented Ising v3 spin lanes."""
    documented_total_luts = int(round(kv260_lut_budget * utilization_pct / 100.0))
    return {
        "architecture": "ising_sampler_v3",
        "node_definition": "one Ising v3 spin update lane",
        "n_nodes": int(n_nodes),
        "source_utilization_pct": float(utilization_pct),
        "kv260_lut_budget": int(kv260_lut_budget),
        "documented_total_luts": documented_total_luts,
        "total_luts_per_node": int(ceil(documented_total_luts / n_nodes)),
        "derivation": "ceil(round(kv260_lut_budget * utilization_pct / 100) / n_nodes)",
    }


def estimate_clock_mhz(
    logic_depth: dict[str, int],
    *,
    lut_delay_ns: float = DEFAULT_LUT_DELAY_NS,
    register_overhead_ns: float = DEFAULT_REGISTER_OVERHEAD_NS,
    practical_cap_mhz: float = DEFAULT_KV260_PRACTICAL_CAP_MHZ,
) -> dict[str, dict[str, float]]:
    """Estimate max clock from LUT-depth timing assumptions without synthesis."""
    estimates: dict[str, dict[str, float]] = {}
    for name, depth in logic_depth.items():
        critical_path_ns = depth * lut_delay_ns + register_overhead_ns
        raw_max_clock_mhz = 1000.0 / critical_path_ns
        estimates[name] = {
            "logic_depth_lut_levels": int(depth),
            "critical_path_ns": round(critical_path_ns, 3),
            "raw_max_clock_mhz": round(raw_max_clock_mhz, 3),
            "capped_max_clock_mhz": round(min(raw_max_clock_mhz, practical_cap_mhz), 3),
        }
    return estimates


def artifact_has_required_fields(artifact: dict[str, Any]) -> bool:
    """Return whether the Exp 1623 artifact satisfies the safe accounting schema."""
    verdict = str(artifact.get("honest_verdict", ""))
    return (
        REQUIRED_ARTIFACT_FIELDS <= set(artifact)
        and artifact.get("status") == "complete"
        and artifact.get("kanele_accounting_ready") is True
        and artifact.get("exp1621_mapping_ready") is True
        and artifact.get("hardware_claim_allowed") is False
        and artifact.get("synthesis_performed") is False
        and artifact.get("board_execution_performed") is False
        and verdict.startswith(TERMINAL_VERDICT_PREFIXES)
    )


def build_artifact(
    *,
    exp1621: dict[str, Any],
    kanele_node: dict[str, Any],
    ising_node: dict[str, Any],
    logic_depth: dict[str, int],
    clock_estimates: dict[str, dict[str, float]],
    run_date: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 1623 no-synthesis accounting artifact."""
    node_count = int(ising_node["n_nodes"])
    kanele_total_luts_at_ising_node_count = int(kanele_node["total_luts_per_node"] * node_count)
    ising_total_luts = int(ising_node["documented_total_luts"])
    exp1621_ready = (
        exp1621.get("status") == "complete"
        and exp1621.get("kan_lut_verilog_ready") is True
        and exp1621.get("kan_lut_block_written") is True
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "status": "complete",
        "spec": ["REQ-KAN-1623", "SCENARIO-KAN-1623"],
        "kanele_accounting_ready": True,
        "exp1621_mapping_ready": exp1621_ready,
        "per_node_lut_consumption": {
            "kanele": kanele_node,
            "ising_v3": ising_node,
            "ising_to_kanele_lut_ratio": round(
                ising_node["total_luts_per_node"] / kanele_node["total_luts_per_node"],
                4,
            ),
        },
        "total_lut_projection_at_64_nodes": {
            "node_count": node_count,
            "kanele_luts": kanele_total_luts_at_ising_node_count,
            "ising_v3_luts": ising_total_luts,
            "kanele_utilization_pct": round(
                100.0 * kanele_total_luts_at_ising_node_count / KV260_LUT_BUDGET,
                4,
            ),
            "ising_v3_utilization_pct": ising_node["source_utilization_pct"],
        },
        "logic_depth_estimate": logic_depth,
        "clock_timing_model": {
            "lut_delay_ns": DEFAULT_LUT_DELAY_NS,
            "register_overhead_ns": DEFAULT_REGISTER_OVERHEAD_NS,
            "kv260_practical_cap_mhz": DEFAULT_KV260_PRACTICAL_CAP_MHZ,
            "timing_closure_claimed": False,
        },
        "clock_estimate_detail": clock_estimates,
        "max_clock_frequency_estimate_mhz": {
            "kanele": clock_estimates["kanele"]["capped_max_clock_mhz"],
            "ising_v3": clock_estimates["ising_v3"]["capped_max_clock_mhz"],
        },
        "kv260_budget": {
            "part": "xck26-sfvc784-2LV-c",
            "lut_budget": KV260_LUT_BUDGET,
            "source": "openspec/capabilities/fpga/spec.md",
        },
        "source_files": {
            "kanele_lut_block": "hardware/kv260/kan_lut_block.v",
            "ising_v3": "hardware/kv260/ising_sampler_v3.v",
            "exp1621": "results/experiment_1621_kanele_mapping.json",
        },
        "accounting_assumptions": [
            "KANELE node uses one 8-bit LUT6 edge block per fan-in edge.",
            "KANELE accumulation uses one LUT per accumulator bit per adder stage.",
            "Ising v3 LUT count uses the documented 48.5% KV260 utilization comment.",
            "Clock estimates are theoretical LUT-depth estimates, not Vivado timing results.",
        ],
        "synthesis_performed": False,
        "board_execution_performed": False,
        "hardware_claim_allowed": False,
        "honest_verdict": (
            "complete: kanele vs ising v3 accounting ready; no synthesis or hardware claim"
        ),
    }
    artifact["kanele_accounting_ready"] = artifact_has_required_fields(artifact)
    return artifact


def run_experiment(
    *,
    kan_lut_block_path: Path = KAN_LUT_BLOCK_PATH,
    ising_v3_path: Path = ISING_V3_PATH,
    exp1621_path: Path = EXP1621_PATH,
    deliverable_path: Path = DELIVERABLE_PATH,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run Exp 1623 and write the KANELE vs Ising v3 accounting artifact."""
    started = time.perf_counter()
    exp1621 = load_json(exp1621_path)
    lut6_primitives = count_lut6_primitives(kan_lut_block_path.read_text())
    n_nodes, utilization_pct = parse_ising_v3_documented_utilization(ising_v3_path.read_text())
    logic_depth = dict(DEFAULT_LOGIC_DEPTH)
    clock_estimates = estimate_clock_mhz(logic_depth)
    artifact = build_artifact(
        exp1621=exp1621,
        kanele_node=build_kanele_node_accounting(
            lut6_primitives_per_edge=lut6_primitives,
        ),
        ising_node=build_ising_v3_node_accounting(
            n_nodes=n_nodes,
            utilization_pct=utilization_pct,
        ),
        logic_depth=logic_depth,
        clock_estimates=clock_estimates,
        run_date=run_date,
        duration_s=time.perf_counter() - started,
    )
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    deliverable_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
