"""CPU-only p-bit sampler portability packet for future FPGA work.

The experiment is intentionally not a hardware run.  It builds a small Ising
case, compares a p-bit-style update schedule to a CPU sequential-Gibbs baseline,
and records the dual-BRAM sketch and tool-gating facts needed before anyone can
make a Vivado or KV260 claim.

Spec refs: REQ-HW-046, SCENARIO-HW-046.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1320_pbit_sampler_portability_packet.json"

EXPERIMENT_ID = 1320
SCHEMA = "pbit_sampler_portability_packet_v1"
DEFAULT_RUN_DATE = "20260505"
DEFAULT_GIBBS_SWEEPS = 180
DEFAULT_PBIT_SWEEPS = 140
DEFAULT_REUSE_FACTORS = (1, 2, 4)
DEFAULT_DAC_BITS = (2, 3, 4, 6)
DEFAULT_FIELD_CLIP = 2.0

HONEST_VERDICTS = {
    "cpu_portability_packet_ready_hardware_not_run",
    "hardware_claim_allowed_after_execution",
}

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "dual_bram_mapping_ready",
    "reuse_factor_sweep",
    "dac_bits_sweep",
    "kl_to_cpu_gibbs",
    "vivado_required_for_next_step",
    "hardware_claim_allowed",
    "honest_verdict",
}


@dataclass(frozen=True)
class IsingCase:
    """A tiny Ising model small enough to enumerate exactly."""

    name: str
    j_matrix: np.ndarray
    bias: np.ndarray
    beta: float
    field_clip: float = DEFAULT_FIELD_CLIP

    @property
    def n_spins(self) -> int:
        """Return the number of logical p-bits/spins in the case."""
        return int(self.bias.shape[0])


def tiny_ising_case() -> IsingCase:
    """Return the deterministic four-spin case used for the packet."""
    j_matrix = np.array(
        [
            [0.0, 0.75, -0.35, 0.20],
            [0.75, 0.0, 0.45, -0.25],
            [-0.35, 0.45, 0.0, 0.55],
            [0.20, -0.25, 0.55, 0.0],
        ],
        dtype=np.float64,
    )
    bias = np.array([0.10, -0.05, 0.0, 0.08], dtype=np.float64)
    return IsingCase(
        name="n4_signed_ring_chord",
        j_matrix=j_matrix,
        bias=bias,
        beta=1.25,
    )


def enumerate_spin_states(n_spins: int) -> np.ndarray:
    """Enumerate all {-1, +1} states in lexicographic order."""
    if n_spins <= 0:
        raise ValueError("n_spins must be positive")
    return np.array(list(itertools.product((-1, 1), repeat=n_spins)), dtype=np.int8)


def ising_energy(case: IsingCase, spins: np.ndarray) -> float:
    """Compute E = -0.5 * s.T J s - b.T s for one spin state."""
    spin_vec = np.asarray(spins, dtype=np.float64)
    pair_energy = -0.5 * float(spin_vec @ case.j_matrix @ spin_vec)
    bias_energy = -float(case.bias @ spin_vec)
    return pair_energy + bias_energy


def exact_boltzmann_distribution(case: IsingCase, states: np.ndarray) -> np.ndarray:
    """Compute the exact Boltzmann distribution for the enumerated tiny case."""
    energies = np.array([ising_energy(case, state) for state in states], dtype=np.float64)
    shifted = -case.beta * energies
    shifted -= float(np.max(shifted))
    weights = np.exp(shifted)
    return weights / weights.sum()


def _state_index(states: np.ndarray) -> dict[tuple[int, ...], int]:
    return {tuple(int(v) for v in state): idx for idx, state in enumerate(states)}


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def quantize_field(field: float, dac_bits: int, clip: float = DEFAULT_FIELD_CLIP) -> float:
    """Quantize a local field to a signed, clipped DAC ladder."""
    if dac_bits < 1:
        raise ValueError("dac_bits must be >= 1")
    levels = (2**dac_bits) - 1
    clipped = min(max(float(field), -clip), clip)
    normalized = (clipped + clip) / (2.0 * clip)
    quantized_index = round(normalized * levels)
    return (quantized_index / levels) * (2.0 * clip) - clip


def _prob_plus(case: IsingCase, state: np.ndarray, spin_index: int, dac_bits: int | None) -> float:
    field = float(case.j_matrix[spin_index] @ state + case.bias[spin_index])
    if dac_bits is not None:
        field = quantize_field(field, dac_bits=dac_bits, clip=case.field_clip)
    return _sigmoid(2.0 * case.beta * field)


def _apply_update_group(
    dist: np.ndarray,
    states: np.ndarray,
    index_by_state: Mapping[tuple[int, ...], int],
    case: IsingCase,
    selected_spins: Sequence[int],
    dac_bits: int | None,
) -> np.ndarray:
    next_dist = np.zeros_like(dist, dtype=np.float64)
    assignments = list(itertools.product((-1, 1), repeat=len(selected_spins)))
    for state_idx, state in enumerate(states):
        base_prob = float(dist[state_idx])
        spin_probs = [
            _prob_plus(case, state.astype(np.float64), spin_index, dac_bits)
            for spin_index in selected_spins
        ]
        for assignment in assignments:
            new_state = state.copy()
            assignment_prob = base_prob
            for spin_index, new_spin, plus_prob in zip(selected_spins, assignment, spin_probs):
                assignment_prob *= plus_prob if new_spin == 1 else 1.0 - plus_prob
                new_state[spin_index] = new_spin
            next_dist[index_by_state[tuple(int(v) for v in new_state)]] += assignment_prob
    return next_dist / next_dist.sum()


def cpu_gibbs_distribution(
    case: IsingCase, states: np.ndarray, sweeps: int = DEFAULT_GIBBS_SWEEPS
) -> np.ndarray:
    """Propagate the exact distribution under round-robin CPU Gibbs updates."""
    dist = np.full(len(states), 1.0 / len(states), dtype=np.float64)
    index_by_state = _state_index(states)
    for step in range(sweeps * case.n_spins):
        spin_index = step % case.n_spins
        dist = _apply_update_group(
            dist,
            states,
            index_by_state,
            case,
            selected_spins=(spin_index,),
            dac_bits=None,
        )
    return dist


def pbit_distribution(
    case: IsingCase,
    states: np.ndarray,
    reuse_factor: int,
    dac_bits: int,
    sweeps: int = DEFAULT_PBIT_SWEEPS,
) -> np.ndarray:
    """Propagate a p-bit update schedule with reuse and DAC quantization."""
    phases = max(1, min(int(reuse_factor), case.n_spins))
    dist = np.full(len(states), 1.0 / len(states), dtype=np.float64)
    index_by_state = _state_index(states)
    for _sweep in range(sweeps):
        for phase in range(phases):
            selected = tuple(idx for idx in range(case.n_spins) if idx % phases == phase)
            dist = _apply_update_group(
                dist,
                states,
                index_by_state,
                case,
                selected_spins=selected,
                dac_bits=dac_bits,
            )
    return dist


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    """Compute KL(p || q), with a tiny floor for numerical safety."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.shape != q_arr.shape:
        raise ValueError("distributions must have the same shape")
    p_norm = p_arr / p_arr.sum()
    q_norm = q_arr / q_arr.sum()
    mask = p_norm > 0.0
    return float(np.sum(p_norm[mask] * np.log(p_norm[mask] / np.maximum(q_norm[mask], epsilon))))


def distribution_l1(p: np.ndarray, q: np.ndarray) -> float:
    """Return the L1 distance between two probability distributions."""
    p_norm = np.asarray(p, dtype=np.float64)
    p_norm = p_norm / p_norm.sum()
    q_norm = np.asarray(q, dtype=np.float64)
    q_norm = q_norm / q_norm.sum()
    return float(np.sum(np.abs(p_norm - q_norm)))


def _round_metric(value: float) -> float:
    return round(float(value), 12)


def _update_policy(reuse_factor: int, n_spins: int) -> str:
    phases = max(1, min(int(reuse_factor), n_spins))
    if phases == 1:
        return "fully_parallel_snapshot"
    if phases >= n_spins:
        return "single_site_gibbs_like"
    return "time_multiplexed_snapshot"


def reuse_factor_sweep(
    case: IsingCase,
    states: np.ndarray,
    baseline: np.ndarray,
    reuse_factors: Sequence[int] = DEFAULT_REUSE_FACTORS,
    dac_bits: int = 6,
    sweeps: int = DEFAULT_PBIT_SWEEPS,
) -> list[dict[str, Any]]:
    """Run the CPU p-bit simulator across reuse factors."""
    rows: list[dict[str, Any]] = []
    for reuse_factor in reuse_factors:
        phases = max(1, min(int(reuse_factor), case.n_spins))
        distribution = pbit_distribution(case, states, phases, dac_bits=dac_bits, sweeps=sweeps)
        rows.append(
            {
                "reuse_factor": phases,
                "logical_spins": case.n_spins,
                "physical_pbits": int(math.ceil(case.n_spins / phases)),
                "logical_spins_per_pbit": phases,
                "max_parallel_updates": int(math.ceil(case.n_spins / phases)),
                "dac_bits": int(dac_bits),
                "update_policy": _update_policy(phases, case.n_spins),
                "kl_to_cpu_gibbs": _round_metric(kl_divergence(distribution, baseline)),
                "l1_to_cpu_gibbs": _round_metric(distribution_l1(distribution, baseline)),
            }
        )
    return rows


def dac_bits_sweep(
    case: IsingCase,
    states: np.ndarray,
    baseline: np.ndarray,
    dac_bits_values: Sequence[int] = DEFAULT_DAC_BITS,
    reuse_factor: int = 4,
    sweeps: int = DEFAULT_PBIT_SWEEPS,
) -> list[dict[str, Any]]:
    """Run the CPU p-bit simulator across DAC precision assumptions."""
    rows: list[dict[str, Any]] = []
    phases = max(1, min(int(reuse_factor), case.n_spins))
    for dac_bits in dac_bits_values:
        distribution = pbit_distribution(
            case, states, phases, dac_bits=int(dac_bits), sweeps=sweeps
        )
        rows.append(
            {
                "dac_bits": int(dac_bits),
                "dac_levels": int(2 ** int(dac_bits)),
                "field_clip": case.field_clip,
                "reuse_factor": phases,
                "update_policy": _update_policy(phases, case.n_spins),
                "kl_to_cpu_gibbs": _round_metric(kl_divergence(distribution, baseline)),
                "l1_to_cpu_gibbs": _round_metric(distribution_l1(distribution, baseline)),
            }
        )
    return rows


def dual_bram_mapping_sketch(
    case: IsingCase, reuse_factors: Sequence[int] = DEFAULT_REUSE_FACTORS
) -> dict[str, Any]:
    """Sketch the BRAM banking pattern for a later RTL implementation."""
    return {
        "bank_count": 2,
        "bank_a": {
            "name": "BRAM_A",
            "role": "read_snapshot",
            "contents": ["spin_state_snapshot", "coupling_rows", "bias_terms"],
        },
        "bank_b": {
            "name": "BRAM_B",
            "role": "delayed_write_update",
            "contents": ["next_spin_state", "quantized_local_fields", "rng_thresholds"],
        },
        "read_snapshot_path": True,
        "write_update_path": True,
        "spin_serial_schedule": {
            "logical_spins": case.n_spins,
            "reuse_factors": [int(value) for value in reuse_factors],
            "phase_rule": "phase = tick mod reuse_factor; update spins where i mod reuse_factor == phase",
        },
        "portability_notes": [
            "Bank A is read-only within an update phase so all selected p-bits see the same snapshot.",
            "Bank B receives delayed writes and becomes the next snapshot after the phase boundary.",
            "The sketch is CPU-derived only; it is not a synthesized BRAM utilization report.",
        ],
    }


def is_dual_bram_mapping_ready(sketch: Mapping[str, Any]) -> bool:
    """Return whether the mapping sketch contains the minimum RTL handoff facts."""
    return bool(
        sketch.get("bank_count") == 2
        and sketch.get("bank_a", {}).get("role") == "read_snapshot"
        and sketch.get("bank_b", {}).get("role") == "delayed_write_update"
        and sketch.get("read_snapshot_path") is True
        and sketch.get("write_update_path") is True
        and sketch.get("spin_serial_schedule")
    )


def detect_fpga_environment(
    which: Callable[[str], str | None] = shutil.which,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Detect local FPGA tool availability without executing synthesis."""
    env_map = os.environ if env is None else env
    tool_commands = {
        "vivado": "vivado",
        "vitis": "vitis",
        "yosys": "yosys",
        "nextpnr_xilinx": "nextpnr-xilinx",
        "nextpnr_ice40": "nextpnr-ice40",
        "icepack": "icepack",
    }
    detected = {name: which(command) for name, command in tool_commands.items()}
    kv260_bitfile = str(env_map.get("CARNOT_KV260_BITFILE", ""))
    return {
        **{f"{name}_path": path for name, path in detected.items()},
        **{f"{name}_available": path is not None for name, path in detected.items()},
        "equivalent_fpga_tool_available": any(path is not None for path in detected.values()),
        "kv260_bitfile_env": kv260_bitfile,
        "kv260_bitfile_configured": bool(kv260_bitfile),
        "hardware_execution_confirmed": False,
    }


def hardware_claim_allowed(
    fpga_environment: Mapping[str, Any],
    synthesis_performed: bool = False,
    board_executed: bool = False,
) -> bool:
    """Allow hardware claims only after actual synthesis or board execution."""
    return bool(fpga_environment and (synthesis_performed or board_executed))


def vivado_required_for_next_step(synthesis_performed: bool = False) -> bool:
    """Return the next-step gate required by REQ-HW-046."""
    return not bool(synthesis_performed)


def _case_payload(case: IsingCase) -> dict[str, Any]:
    return {
        "name": case.name,
        "n_spins": case.n_spins,
        "beta": case.beta,
        "field_clip": case.field_clip,
        "j_matrix": case.j_matrix.tolist(),
        "bias": case.bias.tolist(),
    }


def build_artifact(
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    fpga_environment: Mapping[str, Any] | None = None,
    synthesis_performed: bool = False,
    board_executed: bool = False,
) -> dict[str, Any]:
    """Build the complete CPU-only portability packet."""
    case = tiny_ising_case()
    states = enumerate_spin_states(case.n_spins)
    baseline = cpu_gibbs_distribution(case, states, sweeps=DEFAULT_GIBBS_SWEEPS)
    exact = exact_boltzmann_distribution(case, states)
    reuse_rows = reuse_factor_sweep(case, states, baseline)
    dac_rows = dac_bits_sweep(case, states, baseline)
    sketch = dual_bram_mapping_sketch(case)
    mapping_ready = is_dual_bram_mapping_ready(sketch)
    environment = dict(fpga_environment or detect_fpga_environment())
    claim_allowed = hardware_claim_allowed(
        environment,
        synthesis_performed=synthesis_performed,
        board_executed=board_executed,
    )
    selected = dac_rows[-1]
    verdict = (
        "hardware_claim_allowed_after_execution"
        if claim_allowed
        else "cpu_portability_packet_ready_hardware_not_run"
    )
    artifact = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": run_date,
            "project_root": str(project_root),
            "synthesis_performed": bool(synthesis_performed),
            "board_executed": bool(board_executed),
        },
        "status": "complete",
        "tiny_ising_case": _case_payload(case),
        "cpu_gibbs_baseline": {
            "algorithm": "exact_distribution_propagation_round_robin_gibbs",
            "sweeps": DEFAULT_GIBBS_SWEEPS,
            "state_order": states.tolist(),
            "distribution": [_round_metric(value) for value in baseline.tolist()],
            "kl_to_exact_boltzmann": _round_metric(kl_divergence(baseline, exact)),
        },
        "pbit_update_model": {
            "probability_rule": "P(s_i=+1)=sigmoid(2*beta*quantized_h_i)",
            "reuse_factor_semantics": "reuse_factor phases serialize logical spins onto fewer physical p-bits",
            "dac_quantization": "signed clipped uniform field ladder",
        },
        "dual_bram_mapping_ready": mapping_ready,
        "dual_bram_mapping_sketch": sketch,
        "reuse_factor_sweep": reuse_rows,
        "dac_bits_sweep": dac_rows,
        "selected_cpu_equivalence": {
            "baseline": "cpu_sequential_gibbs",
            "reuse_factor": selected["reuse_factor"],
            "dac_bits": selected["dac_bits"],
            "kl_to_cpu_gibbs": selected["kl_to_cpu_gibbs"],
            "l1_to_cpu_gibbs": selected["l1_to_cpu_gibbs"],
        },
        "kl_to_cpu_gibbs": selected["kl_to_cpu_gibbs"],
        "fpga_environment": environment,
        "vivado_required_for_next_step": vivado_required_for_next_step(synthesis_performed),
        "hardware_claim_allowed": claim_allowed,
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the public schema and hardware-claim gate."""
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if len(artifact["reuse_factor_sweep"]) < 3:
        raise ValueError("reuse_factor_sweep must include at least three entries")
    if len(artifact["dac_bits_sweep"]) < 3:
        raise ValueError("dac_bits_sweep must include at least three entries")
    if not artifact["dual_bram_mapping_ready"]:
        raise ValueError("dual_bram_mapping_ready must be true for the final packet")
    metadata = artifact.get("metadata", {})
    actual_hardware_run = bool(
        metadata.get("synthesis_performed") or metadata.get("board_executed")
    )
    if artifact["hardware_claim_allowed"] and not actual_hardware_run:
        raise ValueError("hardware_claim_allowed requires synthesis_performed or board_executed")
    if artifact["vivado_required_for_next_step"] is False and not actual_hardware_run:
        raise ValueError(
            "vivado_required_for_next_step can be false only after synthesis or board execution"
        )
    if artifact["honest_verdict"] not in HONEST_VERDICTS:
        raise ValueError(f"unknown honest_verdict: {artifact['honest_verdict']}")


def write_artifact(
    path: str | Path = DELIVERABLE_PATH, artifact: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Write a validated artifact JSON and return the payload."""
    payload = dict(artifact or build_artifact())
    validate_artifact(payload)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
