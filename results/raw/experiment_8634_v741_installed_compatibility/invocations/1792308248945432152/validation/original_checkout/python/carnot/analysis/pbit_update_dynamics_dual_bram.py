"""Build the Exp 1348 p-bit update dynamics and dual-BRAM handoff packet.

The packet is deliberately a CPU-only planning artifact.  It turns the tiny
Ising evidence from Exp 1320 plus a tiny KAN spline/LUT shape into concrete
hardware assumptions for a future RTL milestone, while refusing KV260 or
hardware claims until synthesis or board execution actually happens.

Spec refs: REQ-HW-047, SCENARIO-HW-047.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIOR_PACKET_PATH = PROJECT_ROOT / "results" / "experiment_1320_pbit_sampler_portability_packet.json"
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json"
)

EXPERIMENT_ID = 1348
SCHEMA = "pbit_update_dynamics_dual_bram_packet_v2"
DEFAULT_RUN_DATE = "20260505"
CPU_ONLY_HONEST_VERDICT = "cpu_only_update_dynamics_dual_bram_packet_ready_hardware_not_run"
HARDWARE_HONEST_VERDICT = "hardware_claim_allowed_after_local_synthesis_or_board_run"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "sync_async_regime",
    "reuse_factor_grid",
    "bram_layout",
    "dac_precision_assumption",
    "finite_delay_assumption",
    "kv260_claim_allowed",
    "hardware_claim_allowed",
    "next_rtl_requirements",
    "honest_verdict",
}

REFERENCE_BASIS = [
    "Scientific Reports 2026 p-bit update-dynamics performance-cost landscape",
    "arXiv 2602.16143 dual-BRAM fully-connected p-bit annealer",
]


def load_prior_packet(path: str | Path = PRIOR_PACKET_PATH) -> dict[str, Any]:
    """Load the Exp 1320 CPU-only p-bit evidence packet."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tiny_kan_case() -> dict[str, Any]:
    """Describe a tiny KAN case as a hardware-facing spline/LUT workload.

    Carnot's KAN implementation stores learnable spline control points per edge
    and per bias.  For hardware planning, the relevant fact is not JAX execution;
    it is the number of spline tables that would need BRAM/LUT storage alongside
    the Ising coupling rows.
    """
    edges = [(0, 1), (1, 2), (2, 3)]
    num_knots = 4
    degree = 1
    entries_per_spline = num_knots + degree
    return {
        "name": "n4_sparse_linear_spline_kan",
        "logical_inputs": 4,
        "edges": [list(edge) for edge in edges],
        "bias_splines": 4,
        "edge_splines": len(edges),
        "num_knots": num_knots,
        "degree": degree,
        "entries_per_spline": entries_per_spline,
        "total_spline_entries": (len(edges) + 4) * entries_per_spline,
        "hardware_role": "KAN edge/bias splines map to coefficient LUT segments, not analog p-bits",
    }


def build_tiny_workloads(prior_packet: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize the tiny Ising and KAN cases used by the handoff packet."""
    tiny_ising = dict(prior_packet["tiny_ising_case"])
    return {
        "ising": {
            "name": tiny_ising["name"],
            "logical_spins": tiny_ising["n_spins"],
            "beta": tiny_ising["beta"],
            "field_clip": tiny_ising["field_clip"],
            "source": "results/experiment_1320_pbit_sampler_portability_packet.json",
        },
        "kan": tiny_kan_case(),
    }


def build_sync_async_regime() -> list[dict[str, Any]]:
    """Define the p-bit update-dynamics regimes to carry into RTL review."""
    return [
        {
            "name": "synchronous_snapshot_parallel",
            "update_order": "all selected p-bits read the same BRAM_A snapshot and commit together",
            "parallelism": "maximal within the selected phase",
            "detailed_balance_risk": "high for dense coupling unless delayed snapshot semantics are validated",
            "hardware_verified": False,
            "reference_basis": REFERENCE_BASIS,
            "rtl_implication": "requires registered snapshot fields and no within-cycle neighbor visibility",
        },
        {
            "name": "asynchronous_single_site_gibbs_like",
            "update_order": "one logical spin updates per phase, matching the Exp 1320 CPU Gibbs-like row",
            "parallelism": "one p-bit update per phase",
            "detailed_balance_risk": "lowest CPU-reference risk but lowest parallelism",
            "hardware_verified": False,
            "reference_basis": REFERENCE_BASIS,
            "rtl_implication": "requires round-robin update_index and per-spin local-field recompute",
        },
        {
            "name": "phase_serialized_delayed_snapshot",
            "update_order": "reuse_factor phases update disjoint spin groups with one-cycle delayed bank swap",
            "parallelism": "bounded by physical p-bits after reuse",
            "detailed_balance_risk": "medium; finite-delay KL drift must be measured after RTL timing",
            "hardware_verified": False,
            "reference_basis": REFERENCE_BASIS,
            "rtl_implication": "requires BRAM_A/BRAM_B ping-pong and explicit phase-boundary commit",
        },
    ]


def _regime_for_reuse_factor(reuse_factor: int, logical_spins: int) -> str:
    if reuse_factor <= 1:
        return "synchronous_snapshot_parallel"
    if reuse_factor >= logical_spins:
        return "asynchronous_single_site_gibbs_like"
    return "phase_serialized_delayed_snapshot"


def build_reuse_factor_grid(prior_packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Translate Exp 1320 reuse rows into hardware memory-reuse assumptions."""
    rows: list[dict[str, Any]] = []
    for prior in prior_packet["reuse_factor_sweep"]:
        logical_spins = int(prior["logical_spins"])
        reuse_factor = int(prior["reuse_factor"])
        rows.append(
            {
                "reuse_factor": reuse_factor,
                "logical_spins": logical_spins,
                "physical_pbits": int(prior["physical_pbits"]),
                "logical_spins_per_pbit": int(prior["logical_spins_per_pbit"]),
                "parallel_update_width": int(prior["max_parallel_updates"]),
                "regime_name": _regime_for_reuse_factor(reuse_factor, logical_spins),
                "bram_phase_semantics": (
                    "phase = tick mod reuse_factor; reads use BRAM_A snapshot; writes land in BRAM_B"
                ),
                "cpu_kl_to_gibbs": float(prior["kl_to_cpu_gibbs"]),
                "cpu_l1_to_gibbs": float(prior["l1_to_cpu_gibbs"]),
                "source_update_policy": prior["update_policy"],
            }
        )
    return rows


def build_bram_layout(tiny_workloads: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Draft the dual-BRAM layout needed by a future p-bit RTL milestone."""
    workloads = dict(tiny_workloads or {})
    kan = dict(workloads.get("kan", tiny_kan_case()))
    return {
        "bank_count": 2,
        "bank_a": {
            "name": "BRAM_A",
            "role": "snapshot_read",
            "contents": [
                "spin_state_snapshot",
                "ising_coupling_rows",
                "ising_bias_terms",
                "kan_spline_lut_segments",
            ],
            "read_contract": "stable for a full update phase",
        },
        "bank_b": {
            "name": "BRAM_B",
            "role": "delayed_write_next_snapshot",
            "contents": [
                "next_spin_state",
                "quantized_local_field_cache",
                "rng_threshold_fifo",
                "phase_completion_flags",
            ],
            "write_contract": "writes become visible only after phase_done",
        },
        "kan_lut_shape": {
            "case_name": kan["name"],
            "segments": kan["edge_splines"] + kan["bias_splines"],
            "entries_per_segment": kan["entries_per_spline"],
            "total_entries": kan["total_spline_entries"],
        },
        "bank_swap_rule": "swap BRAM_A and BRAM_B only at phase boundary",
        "portability_note": "layout is an RTL handoff sketch, not a Vivado BRAM utilization report",
    }


def build_dac_precision_assumption() -> dict[str, Any]:
    """Record the local-field DAC precision assumptions for future hardware."""
    return {
        "selected_bits": 6,
        "bit_widths_to_sweep": [4, 6, 8],
        "field_clip": 2.0,
        "quantization_rule": "signed clipped uniform local-field ladder",
        "rounding": "nearest representable level after clipping",
        "analog_dac_validated": False,
        "assumption_source": "Exp 1320 showed 6-bit CPU p-bit KL below the 4-bit row",
        "acceptance_gate": "RTL and analog mapping must rerun KL, L1, and energy-rank drift per bit width",
    }


def build_finite_delay_assumption() -> dict[str, Any]:
    """Record finite-delay semantics without claiming measured hardware timing."""
    return {
        "delay_cycles_grid": [0, 1, 2, 4],
        "selected_delay_cycles": 1,
        "delay_model": "registered local-field and spin writes become visible after the next phase boundary",
        "local_delay_measurement_available": False,
        "metastability_claim_allowed": False,
        "acceptance_gate": "KL and energy-rank drift must be remeasured after RTL timing",
    }


def build_next_rtl_requirements() -> list[dict[str, str]]:
    """Name concrete future files and interfaces required before a hardware claim."""
    return [
        {
            "path": "hardware/kv260/pbit_dual_bram_pkg.sv",
            "interface": "typed constants for spin width, DAC bits, reuse_factor, and BRAM_A/BRAM_B records",
            "requirement": "define stable packet structs shared by RTL, testbench, and Python driver",
            "claim_gate": "required_before_hardware_claim",
        },
        {
            "path": "hardware/kv260/ising_sampler_v7_pbit_dual_bram.v",
            "interface": "AXI-Lite control registers plus BRAM_A/BRAM_B ping-pong memory ports",
            "requirement": "implement phase-serialized p-bit updates with delayed snapshot commit",
            "claim_gate": "required_before_hardware_claim",
        },
        {
            "path": "hardware/kv260/synth_pbit_dual_bram.tcl",
            "interface": "Vivado batch synthesis entrypoint for xck26-sfvc784-2LV-c",
            "requirement": "emit timing, LUT, FF, BRAM, and seed metadata into results/",
            "claim_gate": "required_before_hardware_claim",
        },
        {
            "path": "python/carnot/hardware/pbit_dual_bram_driver.py",
            "interface": "Python AXI-Lite register map and BRAM preload/readback API",
            "requirement": "load Ising rows, KAN LUT segments, DAC settings, and reuse-factor schedule",
            "claim_gate": "required_before_hardware_claim",
        },
        {
            "path": "tests/python/test_pbit_dual_bram_rtl_contract.py",
            "interface": "cocotb or simulator-facing BRAM_A/BRAM_B contract tests",
            "requirement": "prove no updated neighbor is visible before the phase boundary",
            "claim_gate": "required_before_hardware_claim",
        },
    ]


def kv260_claim_allowed(synthesis_performed: bool = False, board_executed: bool = False) -> bool:
    """Allow KV260 claims only after local synthesis or board execution."""
    return bool(synthesis_performed or board_executed)


def hardware_claim_allowed(synthesis_performed: bool = False, board_executed: bool = False) -> bool:
    """Allow generic hardware claims only after local synthesis or board execution."""
    return bool(synthesis_performed or board_executed)


def build_artifact(
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    prior_packet: Mapping[str, Any] | None = None,
    synthesis_performed: bool = False,
    board_executed: bool = False,
) -> dict[str, Any]:
    """Build the complete Exp 1348 handoff packet."""
    prior = dict(prior_packet or load_prior_packet())
    tiny_workloads = build_tiny_workloads(prior)
    synthesis = bool(synthesis_performed)
    board = bool(board_executed)
    kv260_claim = kv260_claim_allowed(synthesis, board)
    hardware_claim = hardware_claim_allowed(synthesis, board)
    artifact = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": run_date,
            "project_root": str(project_root),
            "prior_packet": "results/experiment_1320_pbit_sampler_portability_packet.json",
            "synthesis_performed": synthesis,
            "board_executed": board,
            "local_hardware_synthesis_or_board_run": bool(synthesis or board),
        },
        "status": "complete",
        "reference_basis": REFERENCE_BASIS,
        "tiny_workloads": tiny_workloads,
        "sync_async_regime": build_sync_async_regime(),
        "reuse_factor_grid": build_reuse_factor_grid(prior),
        "bram_layout": build_bram_layout(tiny_workloads),
        "dac_precision_assumption": build_dac_precision_assumption(),
        "finite_delay_assumption": build_finite_delay_assumption(),
        "kv260_claim_allowed": kv260_claim,
        "hardware_claim_allowed": hardware_claim,
        "next_rtl_requirements": build_next_rtl_requirements(),
        "honest_verdict": HARDWARE_HONEST_VERDICT if hardware_claim else CPU_ONLY_HONEST_VERDICT,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the public schema and the hardware-claim honesty gates."""
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    if len(artifact["sync_async_regime"]) < 3:
        raise ValueError("sync_async_regime must include at least three regimes")
    if len(artifact["reuse_factor_grid"]) < 3:
        raise ValueError("reuse_factor_grid must include at least three entries")
    layout = artifact["bram_layout"]
    if layout.get("bank_count") != 2:
        raise ValueError("bram_layout must define exactly two banks")
    if layout.get("bank_a", {}).get("role") != "snapshot_read":
        raise ValueError("bram_layout bank_a must be snapshot_read")
    if layout.get("bank_b", {}).get("role") != "delayed_write_next_snapshot":
        raise ValueError("bram_layout bank_b must be delayed_write_next_snapshot")
    if "kan_spline_lut_segments" not in layout.get("bank_a", {}).get("contents", []):
        raise ValueError("bram_layout must include kan_spline_lut_segments")
    metadata = artifact.get("metadata", {})
    actual_hardware_run = bool(
        metadata.get("synthesis_performed") or metadata.get("board_executed")
    )
    if artifact["kv260_claim_allowed"] and not actual_hardware_run:
        raise ValueError("kv260_claim_allowed requires synthesis_performed or board_executed")
    if artifact["hardware_claim_allowed"] and not actual_hardware_run:
        raise ValueError("hardware_claim_allowed requires synthesis_performed or board_executed")
    if artifact["honest_verdict"] not in {CPU_ONLY_HONEST_VERDICT, HARDWARE_HONEST_VERDICT}:
        raise ValueError(f"unknown honest_verdict: {artifact['honest_verdict']}")


def write_artifact(
    path: str | Path = DELIVERABLE_PATH, artifact: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Write the validated Exp 1348 packet and return the payload."""
    payload = dict(artifact or build_artifact())
    validate_artifact(payload)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
