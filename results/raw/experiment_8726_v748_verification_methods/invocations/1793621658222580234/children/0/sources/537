"""Projection-only SSQA dual-BRAM register-map plan for Exp 2985.

Spec refs: REQ-HW-081, SCENARIO-HW-081.

The GateMate n=16 tile now has a built and flashed bitstream, but the prior
artifacts also show the important limitation: there is no host-visible sampler
IO, register window, UART, GPIO, or GateMate SRAM readback path yet. This module
therefore produces an architecture plan only. It turns the current evidence into
explicit registers, memory banks, smoke vectors, and readback checks that a later
hardware milestone can implement and execute.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_FILENAME = "experiment_2985_ssqa_dual_bram_register_map_plan_v1.json"
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
EXP2955_REL_PATH = Path("results/experiment_2955_gatemate_constraints_materialization_v4.json")
EXP2956_REL_PATH = Path("results/experiment_2956_gatemate_n16_bitstream_build_v4.json")
EXP2972_REL_PATH = Path("results/experiment_2972_gatemate_post_flash_output_hash_v3.json")
EXP2984_REL_PATH = Path("results/experiment_2984_gatemate_readback_smoke_vector_v4.json")

SOURCE_REL_PATHS = [
    EXP2972_REL_PATH,
    EXP2984_REL_PATH,
    EXP2955_REL_PATH,
    EXP2956_REL_PATH,
    Path("research-references.md"),
    Path("research-hardware-wishlist.md"),
    Path("hardware/gatemate/ising_n16_gatemate.v"),
    Path("hardware/gatemate/ising_n16_gatemate.ccf"),
    Path("hardware/gatemate/ising_n16_gatemate_test_vector.json"),
]

RUN_DATE = "20260524"
INFERENCE_SUBSTRATE = "architecture_projection_only"
TARGET_BOARDS = ["GateMate A1-EVB-2M", "AMD/Xilinx KV260"]
BRAM36_BITS = 36_864
KV260_BRAM36_BUDGET = 144

N_SPINS = 16
CURRENT_GATEMATE_COUPLING_BITS = 8
PROJECTED_COUPLING_BITS = 16
BIAS_BITS = 16
FIELD_CACHE_BITS = 24
RNG_THRESHOLD_BITS = 16

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "register_map_plan_ready",
    "projection_only",
    "target_boards",
    "register_map",
    "memory_layout",
    "smoke_vectors",
    "readback_checks",
    "resource_accounting",
    "risks",
    "sampler_claim_allowed",
    "speedup_claim_allowed",
    "inference_substrate",
    "duration_s",
}


@dataclass(frozen=True)
class SourcePayloads:
    """Prior artifacts used as evidence, never as permission to make new claims."""

    exp2955: Mapping[str, Any]
    exp2956: Mapping[str, Any]
    exp2972: Mapping[str, Any]
    exp2984: Mapping[str, Any]


def bram36_blocks(bits: int) -> int:
    """Round a bit payload to KV260 BRAM36 blocks for projection accounting."""

    return int(ceil(max(0, int(bits)) / BRAM36_BITS))


def _bytes_for_bits(bits: int) -> int:
    return int(ceil(int(bits) / 8))


def _duration(started_s: float, now_s: float | None) -> float:
    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - float(started_s)), 6)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():  # pragma: no cover - exercised only by missing-evidence runs.
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _register(
    name: str,
    offset: int,
    access: str,
    fields: Mapping[str, str],
    reset: str | int = 0,
) -> dict[str, Any]:
    return {
        "name": name,
        "offset": int(offset),
        "width_bits": 32,
        "access": access,
        "reset": reset,
        "fields": dict(fields),
    }


def build_register_map() -> dict[str, Any]:
    """Build the planned 32-bit host-visible register map.

    The offsets deliberately separate control/status from memory windows so a
    later AXI-Lite, UIO, UART-bridge, or JTAG-register implementation can expose
    the same logical contract on both GateMate and KV260.
    """

    return {
        "address_unit": "byte",
        "word_width_bits": 32,
        "endianness": "little",
        "register_groups": {
            "input_control": {
                "purpose": "Host writes problem shape, schedule, and start/reset controls.",
                "registers": [
                    _register(
                        "CONTROL",
                        0x0000,
                        "rw",
                        {
                            "bit0": "start",
                            "bit1": "reset",
                            "bit2": "load_commit",
                            "bit3": "clear_error",
                            "bit4": "bank_swap_enable",
                        },
                    ),
                    _register("N_SPINS", 0x0010, "rw", {"bits15_0": "n_spins"}, N_SPINS),
                    _register("MAX_STEPS", 0x0014, "rw", {"bits15_0": "max_steps"}, 8),
                    _register("ETA_Q1_15", 0x0018, "rw", {"bits15_0": "eta_q1_15"}, 2949),
                    _register(
                        "PRESSURE_START_Q1_15",
                        0x001C,
                        "rw",
                        {"bits15_0": "pressure_start_q1_15"},
                    ),
                    _register(
                        "PRESSURE_DELTA_Q1_15",
                        0x0020,
                        "rw",
                        {"bits15_0": "pressure_delta_q1_15"},
                    ),
                    _register("FANOUT_K", 0x0024, "rw", {"bits7_0": "sparse fanout or 0=dense"}),
                ],
            },
            "seed_state": {
                "purpose": "Host writes deterministic seeds and reads state snapshots.",
                "registers": [
                    _register("RNG_SEED_LO", 0x0030, "rw", {"bits31_0": "rng_seed_lo"}),
                    _register("RNG_SEED_HI", 0x0034, "rw", {"bits31_0": "rng_seed_hi"}),
                    _register("INIT_SPINS_LO", 0x0038, "rw", {"bits15_0": "initial n=16 spins"}),
                    _register("CURRENT_SPINS_LO", 0x003C, "ro", {"bits15_0": "current spins"}),
                    _register("NEXT_SPINS_LO", 0x0040, "ro", {"bits15_0": "delayed-write spins"}),
                    _register("PHASE", 0x0044, "ro", {"bit0": "0=bank_a_read, 1=bank_b_commit"}),
                ],
            },
            "energy_verifier": {
                "purpose": "Read projected energy/checksum evidence, not sampler quality claims.",
                "registers": [
                    _register("ENERGY_ACCUM_LO", 0x0050, "ro", {"bits31_0": "energy low word"}),
                    _register("ENERGY_ACCUM_HI", 0x0054, "ro", {"bits31_0": "energy high word"}),
                    _register("FIELD_ACCUM_Q16_8", 0x0058, "ro", {"bits31_0": "last field sum"}),
                    _register(
                        "VERIFIER_HASH_LO",
                        0x005C,
                        "rw",
                        {"bits31_0": "expected smoke-vector hash low word"},
                    ),
                    _register(
                        "VERIFIER_HASH_HI",
                        0x0060,
                        "rw",
                        {"bits31_0": "expected smoke-vector hash high word"},
                    ),
                ],
            },
            "output": {
                "purpose": "Host-visible sample and timing outputs for later smoke runs.",
                "registers": [
                    _register("SAMPLE_OUT_LO", 0x0070, "ro", {"bits15_0": "n=16 spin_out"}),
                    _register("SAMPLE_OUT_HI", 0x0074, "ro", {"bits31_0": "reserved for n>32"}),
                    _register("STEP_COUNT", 0x0078, "ro", {"bits31_0": "completed steps"}),
                    _register("OUTPUT_HASH_LO", 0x007C, "ro", {"bits31_0": "observed output hash low"}),
                    _register("OUTPUT_HASH_HI", 0x0080, "ro", {"bits31_0": "observed output hash high"}),
                    _register("CYCLE_COUNT_LO", 0x0084, "ro", {"bits31_0": "cycle counter low"}),
                    _register("CYCLE_COUNT_HI", 0x0088, "ro", {"bits31_0": "cycle counter high"}),
                ],
            },
            "status_error": {
                "purpose": "Readiness, error, and readback-integrity checks.",
                "registers": [
                    _register(
                        "STATUS",
                        0x0090,
                        "ro",
                        {
                            "bit0": "ready",
                            "bit1": "busy",
                            "bit2": "done",
                            "bit3": "error",
                            "bit4": "readback_supported",
                        },
                    ),
                    _register("ERROR_CODE", 0x0094, "ro", {"bits31_0": "first failing condition"}),
                    _register(
                        "CLAIM_FLAGS",
                        0x0098,
                        "ro",
                        {"bit0": "sampler_claim_allowed", "bit1": "speedup_claim_allowed"},
                    ),
                    _register("SCRATCH", 0x009C, "rw", {"bits31_0": "write/readback sanity word"}),
                    _register("BANK_A_CRC32", 0x00A0, "ro", {"bits31_0": "bank A CRC32"}),
                    _register("BANK_B_CRC32", 0x00A4, "ro", {"bits31_0": "bank B CRC32"}),
                ],
            },
        },
        "memory_windows": {
            "BANK_A_READ_SNAPSHOT": {
                "base": 0x1000,
                "size_bytes": 0x1000,
                "access": "rw_before_start_ro_while_busy",
            },
            "BANK_B_DELAYED_WRITE": {
                "base": 0x2000,
                "size_bytes": 0x1000,
                "access": "ro_for_host_rw_for_core",
            },
            "TRACE_LOG": {
                "base": 0x3000,
                "size_bytes": 0x1000,
                "access": "ro",
            },
        },
    }


def build_memory_layout() -> dict[str, Any]:
    """Build the dual-bank memory projection for n=16 and later SSQA state."""

    current_coupling_bits = N_SPINS * N_SPINS * CURRENT_GATEMATE_COUPLING_BITS
    bank_a_bits = {
        "dense_coupling_matrix_q8_8_bits": N_SPINS * N_SPINS * PROJECTED_COUPLING_BITS,
        "bias_q8_8_bits": N_SPINS * BIAS_BITS,
        "current_spin_bits": N_SPINS,
    }
    bank_b_bits = {
        "next_spin_bits": N_SPINS,
        "field_cache_q16_8_bits": N_SPINS * FIELD_CACHE_BITS,
        "rng_threshold_bits": N_SPINS * RNG_THRESHOLD_BITS,
        "phase_flag_bits": N_SPINS,
    }
    bank_a_total = sum(bank_a_bits.values())
    bank_b_total = sum(bank_b_bits.values())
    total_bits = bank_a_total + bank_b_total
    return {
        "assumptions": {
            "n_spins": N_SPINS,
            "coefficient_width_bits": PROJECTED_COUPLING_BITS,
            "bias_width_bits": BIAS_BITS,
            "field_cache_width_bits": FIELD_CACHE_BITS,
            "rng_threshold_width_bits": RNG_THRESHOLD_BITS,
            "layout_scope": "projection for register-map planning, not synthesized RTL",
        },
        "current_gatemate_rtl_floor": {
            "top_module": "ising_n16_gatemate",
            "dense_coupling_bits_q7": current_coupling_bits,
            "spin_register_floor_bits": N_SPINS * 3,
            "note": "Current RTL stores j_matrix as registers; the plan moves state into banks.",
        },
        "banks": {
            "bank_a_read_snapshot": {
                "role": "stable row-read snapshot during one sweep",
                "fields": bank_a_bits,
                "formula": "n*n*16 + n*16 + n",
                "total_bits": bank_a_total,
                "total_bytes": _bytes_for_bits(bank_a_total),
            },
            "bank_b_delayed_write": {
                "role": "delayed writes, field cache, stochastic thresholds, phase metadata",
                "fields": bank_b_bits,
                "formula": "n*(1 + 24 + 16 + 1)",
                "total_bits": bank_b_total,
                "total_bytes": _bytes_for_bits(bank_b_total),
            },
        },
        "total_projected_bits": total_bits,
        "total_projected_bytes": _bytes_for_bits(total_bits),
        "kv260_bram36_blocks_min_by_bank": {
            "bank_a_read_snapshot": bram36_blocks(bank_a_total),
            "bank_b_delayed_write": bram36_blocks(bank_b_total),
            "total_if_banks_are_separate": bram36_blocks(bank_a_total)
            + bram36_blocks(bank_b_total),
        },
        "kv260_bram36_budget": KV260_BRAM36_BUDGET,
        "gatemate_block_ram_budget": "unknown_until_nextpnr_resource_report_is_parsed",
        "bank_swap_semantics": (
            "Bank A is stable during row reads; Bank B accumulates delayed writes; "
            "CONTROL.bank_swap_enable permits commit only after STATUS.busy clears."
        ),
    }


def build_smoke_vectors() -> list[dict[str, Any]]:
    """Return later-executable IO smoke vectors with projection-only expected values."""

    return [
        {
            "name": "reset_status_scratch",
            "source": "register_map_plan",
            "sequence": [
                "write CONTROL.reset=1",
                "write SCRATCH=0xa5a55a5a",
                "read SCRATCH",
                "read STATUS",
            ],
            "pass_condition": "SCRATCH == 0xa5a55a5a and STATUS.ready=1 and STATUS.error=0",
            "requires_host_visible_io": True,
            "claim_unlocked_by_vector": "none",
        },
        {
            "name": "n16_ring_chord_from_exp2955",
            "source": "hardware/gatemate/ising_n16_gatemate_test_vector.json",
            "init_spins_hex": "0xace1",
            "max_steps": 8,
            "software_reference_spin_out_hex": "0xe7ac",
            "sequence": [
                "load INIT_SPINS_LO=0xace1",
                "load exp2955 couplings into BANK_A_READ_SNAPSHOT",
                "write MAX_STEPS=8",
                "write CONTROL.start=1",
                "poll STATUS.done",
                "read SAMPLE_OUT_LO and STEP_COUNT",
            ],
            "pass_condition": "after STATUS.done=1, SAMPLE_OUT_LO[15:0] == 0xe7ac and STEP_COUNT == 8",
            "requires_host_visible_io": True,
            "claim_unlocked_by_vector": "none",
        },
        {
            "name": "bank_swap_crc",
            "source": "ssqa_dual_bram_projection",
            "sequence": [
                "write dense q8.8 couplings to Bank A",
                "write delayed state pattern to Bank B",
                "read BANK_A_CRC32 and BANK_B_CRC32",
                "toggle CONTROL.bank_swap_enable after done",
                "read PHASE",
            ],
            "pass_condition": "CRC32 values match host-computed payloads and PHASE toggles once",
            "requires_host_visible_io": True,
            "claim_unlocked_by_vector": "none",
        },
    ]


def build_readback_checks() -> list[dict[str, Any]]:
    """Return readback checks that distinguish contact from sampler IO."""

    return [
        {
            "name": "bitstream_hash_recheck",
            "registers_or_files": ["bitstream_path", "bitstream_sha256"],
            "pass_condition": "local bitstream SHA256 still matches Exp 2972/2956",
            "current_status": "already_available_as_file_hash_not_sampler_output",
            "claim_unlocked_by_check": "none",
        },
        {
            "name": "status_register_round_trip",
            "registers_or_files": ["STATUS", "ERROR_CODE", "SCRATCH"],
            "pass_condition": "read STATUS/ERROR_CODE and write-read SCRATCH through a real host IO path",
            "current_status": "blocked_until_register_transport_exists",
            "claim_unlocked_by_check": "none",
        },
        {
            "name": "bank_crc32_round_trip",
            "registers_or_files": ["BANK_A_CRC32", "BANK_B_CRC32"],
            "pass_condition": "hardware CRC32 equals host CRC32 for Bank A and Bank B payloads",
            "current_status": "blocked_until_bank_readback_or_crc_registers_exist",
            "claim_unlocked_by_check": "none",
        },
        {
            "name": "cycle_counter_timing_readback",
            "registers_or_files": ["CYCLE_COUNT_LO", "CYCLE_COUNT_HI", "STEP_COUNT"],
            "pass_condition": "cycle counter increases while busy and STEP_COUNT reaches requested steps",
            "current_status": "planned_timing_evidence_not_physical_performance_claim",
            "claim_unlocked_by_check": "none",
        },
        {
            "name": "sample_output_hash_readback",
            "registers_or_files": ["SAMPLE_OUT_LO", "OUTPUT_HASH_LO", "OUTPUT_HASH_HI"],
            "pass_condition": "observed sample output and hash match the selected smoke vector",
            "current_status": "blocked_by_exp2984_no_host_visible_smoke_io",
            "claim_unlocked_by_check": "none",
        },
    ]


def build_resource_accounting(
    source_payloads: SourcePayloads,
    memory_layout: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize source evidence and unknowns required before hardware claims."""

    exp2955 = source_payloads.exp2955
    exp2956 = source_payloads.exp2956
    exp2972 = source_payloads.exp2972
    exp2984 = source_payloads.exp2984
    utilization = exp2956.get("utilization_summary", {})
    utilization = utilization if isinstance(utilization, Mapping) else {}
    timing = exp2956.get("timing_summary", {})
    timing = timing if isinstance(timing, Mapping) else {}
    return {
        "formulas": {
            "current_gatemate_dense_coupling_bits": "16*16*8",
            "projected_bank_a_bits": "16*16*16 + 16*16 + 16",
            "projected_bank_b_bits": "16*(1 + 24 + 16 + 1)",
            "kv260_bram36_blocks": "ceil(bits/36864) rounded per independent bank",
        },
        "current_gatemate_evidence": {
            "device": exp2956.get("device", exp2955.get("device", "unknown")),
            "top_module": exp2956.get("top_module", "ising_n16_gatemate"),
            "bitstream_built": bool(exp2956.get("gatemate_bitstream_built")),
            "bitstream_path": exp2972.get("bitstream_path", exp2956.get("bitstream_path", "")),
            "bitstream_sha256": exp2972.get("bitstream_sha256", exp2956.get("bitstream_sha256", "")),
            "board_detected": bool(exp2972.get("board_detected") or exp2984.get("board_detected")),
            "flash_succeeded": bool(exp2972.get("flash_succeeded") or exp2984.get("flash_succeeded")),
            "readback_supported": bool(exp2984.get("readback_supported")),
            "host_visible_smoke_io_available": bool(
                exp2984.get("smoke_vector_attempted") and exp2984.get("smoke_vector_passed")
            ),
            "timing_summary": dict(timing),
            "utilization_summary": dict(utilization),
            "utilization_counts_available": bool(
                utilization.get("yosys_cells_total") is not None
                or utilization.get("yosys_cell_counts")
                or utilization.get("nextpnr_resource_lines")
            ),
            "io_gap": exp2984.get(
                "expected_smoke_output",
                exp2972.get("timing_observation", {}).get("readback_reason", "unknown"),
            ),
        },
        "constraints_evidence": {
            "constraints_ready": bool(exp2955.get("gatemate_constraints_ready")),
            "constraints_sha256": exp2955.get("constraints_sha256", ""),
            "clock_assumption": exp2955.get("clock_assumption", ""),
            "pin_assumption": exp2955.get("pin_assumption", ""),
            "nextpnr_options_required": list(exp2955.get("nextpnr_options_required", [])),
        },
        "memory_projection": {
            "total_projected_bits": memory_layout["total_projected_bits"],
            "total_projected_bytes": memory_layout["total_projected_bytes"],
            "kv260_bram36_blocks_min_by_bank": dict(
                memory_layout["kv260_bram36_blocks_min_by_bank"]
            ),
            "gatemate_block_ram_budget": memory_layout["gatemate_block_ram_budget"],
        },
        "unknowns_before_implementation": [
            "GateMate placed LUT/FF/BRAM counts are unavailable in the current Exp 2956 artifact.",
            "GateMate host register transport is not present in the current n=16 bitstream.",
            "KV260 UIO register compatibility for this projected map has not been implemented.",
            "Dual-BRAM arbitration LUT/FF cost requires RTL before synthesis accounting is real.",
        ],
    }


def build_artifact(
    *,
    source_payloads: SourcePayloads,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 2985 projection artifact from already-loaded evidence."""

    register_map = build_register_map()
    memory_layout = build_memory_layout()
    artifact = {
        "honest_verdict": "complete: ssqa_dual_bram_register_map_plan_ready_projection_only",
        "register_map_plan_ready": True,
        "projection_only": True,
        "target_boards": list(TARGET_BOARDS),
        "target_board_details": {
            "GateMate A1-EVB-2M": {
                "current_role": "n=16 tile bring-up and future IO smoke-vector target",
                "current_interface_gap": "no host-visible sampler IO path per Exp 2984",
            },
            "AMD/Xilinx KV260": {
                "current_role": "future AXI-Lite/UIO implementation target for the same map",
                "precondition": "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'",
                "planned_register_transport": "/dev/uio* register window",
            },
        },
        "register_map": register_map,
        "memory_layout": memory_layout,
        "smoke_vectors": build_smoke_vectors(),
        "readback_checks": build_readback_checks(),
        "resource_accounting": build_resource_accounting(source_payloads, memory_layout),
        "risks": [
            "The current GateMate bitstream proves flash/contact only; it exposes no sampler output.",
            "The current GateMate CCF is build-only and intentionally leaves non-clock IO unconstrained.",
            "Resource counts beyond memory bit formulas remain unknown until RTL and PnR reports exist.",
            "Cycle-count readback is required before timing evidence can be sample-level evidence.",
            "This plan does not permit sampler, thermodynamic, or physical performance claims.",
        ],
        "sampler_claim_allowed": False,
        "speedup_claim_allowed": False,
        "thermodynamic_claim_allowed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "run_date": RUN_DATE,
        "source_artifacts": [path.as_posix() for path in SOURCE_REL_PATHS],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema and claim-boundary invariants before writing."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["projection_only"] is not True:
        raise ValueError("projection_only must be true")
    if artifact["register_map_plan_ready"] is not True:
        raise ValueError("register_map_plan_ready must be true")
    if artifact["sampler_claim_allowed"] is not False:
        raise ValueError("sampler_claim_allowed must be false")
    if artifact["speedup_claim_allowed"] is not False:
        raise ValueError("speedup_claim_allowed must be false")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def _load_source_payloads(root_path: Path) -> SourcePayloads:
    return SourcePayloads(
        exp2955=_read_json(root_path / EXP2955_REL_PATH),
        exp2956=_read_json(root_path / EXP2956_REL_PATH),
        exp2972=_read_json(root_path / EXP2972_REL_PATH),
        exp2984=_read_json(root_path / EXP2984_REL_PATH),
    )


def run_experiment(
    root_path: Path | str = REPO_ROOT,
    *,
    write: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Run Exp 2985 as a read-only architecture projection and write JSON."""

    root = Path(root_path)
    started = time.perf_counter() if started_s is None else float(started_s)
    artifact = build_artifact(
        source_payloads=_load_source_payloads(root),
        duration_s=_duration(started, now_s),
    )
    if write:
        _write_json(root / OUTPUT_REL_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    artifact = run_experiment(args.root)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "result": str(args.root / OUTPUT_REL_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
