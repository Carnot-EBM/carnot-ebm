"""Exp 2930 KV260 p-bit/SSQA resource and memory scaling projection.

Spec refs: REQ-HW-070, SCENARIO-HW-070.

This module is an accounting pass over existing evidence. It does not contact
the KV260, run Vivado/Yosys, flash a bitstream, rerun a sampler, or create a new
hardware speedup claim. The projection formulas are intentionally simple so the
artifact can be audited like a spreadsheet.
"""

from __future__ import annotations

import argparse
from math import ceil, log2
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP2913_REL_PATH = Path("results/experiment_2913_kv260_hardware_cpu_claim_boundary_v1.json")
EXP2898_REL_PATH = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
EXP2912_REL_PATH = Path("results/experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1.json")
OUTPUT_REL_PATH = Path("results/experiment_2930_kv260_pbit_ssqa_scaling_projection_v1.json")

RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "aggregation_plus_simulation"
BRAM36_BITS = 36_864
KV260_BRAM36_BUDGET = 144
KV260_LUT_BUDGET = 117_120
DEFAULT_FANOUT = 16
COUPLING_BITS = 16
BIAS_BITS = 16
FIELD_CACHE_BITS = 18
RNG_THRESHOLD_BITS = 16

SOURCE_ARTIFACTS = [
    EXP2913_REL_PATH.as_posix(),
    EXP2898_REL_PATH.as_posix(),
    EXP2912_REL_PATH.as_posix(),
    "results/experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json",
    "results/experiment_1320_pbit_sampler_portability_packet.json",
    "research-references.md",
    "research-hardware-wishlist.md",
    "hardware/kv260/ising_sampler_v4_spec.md",
    "hardware/kv260/ising_sampler_v3.v",
]

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "kv260_scaling_projection_ready",
    "projection_only",
    "no_new_hardware_run",
    "source_artifacts",
    "n64_real_evidence_summary",
    "projection_models",
    "n128_projection",
    "n256_projection",
    "assumptions",
    "not_a_speedup_claim",
    "inference_substrate",
    "duration_s",
    "run_date",
}


def bram36_blocks(bits: int) -> int:
    """Return the minimum number of KV260 BRAM36 blocks for a bit payload."""

    return int(ceil(max(0, int(bits)) / BRAM36_BITS))


def _bytes_for_bits(bits: int) -> int:
    return int(ceil(int(bits) / 8))


def _duration(started_s: float, now_s: float | None) -> float:
    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - float(started_s)), 6)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dense_memory_projection(n_spins: int, *, coupling_bits: int = COUPLING_BITS) -> dict[str, Any]:
    """Project dense q8.8 storage for an n-by-n Ising coupling matrix.

    Spec refs: REQ-HW-070. The matrix is intentionally counted as a full n*n
    table, matching the hardware accounting question rather than exploiting
    symmetry, because the KV260 upload path stores row-oriented coefficients.
    """

    coupling_bits_total = int(n_spins) * int(n_spins) * int(coupling_bits)
    bias_bits_total = int(n_spins) * BIAS_BITS
    spin_bits_total = int(n_spins)
    total_bits = coupling_bits_total + bias_bits_total + spin_bits_total
    return {
        "layout": "dense_q8_8_full_row_table",
        "formula": "n*n*coupling_bits + n*bias_bits + n*spin_bits",
        "n_spins": int(n_spins),
        "coupling_bits_per_entry": int(coupling_bits),
        "coupling_bits": coupling_bits_total,
        "bias_bits": bias_bits_total,
        "spin_state_bits": spin_bits_total,
        "total_bits": total_bits,
        "total_bytes": _bytes_for_bits(total_bits),
        "bram36_blocks_min": bram36_blocks(total_bits),
        "bram36_budget_pct": round(100.0 * bram36_blocks(total_bits) / KV260_BRAM36_BUDGET, 6),
    }


def sparse_memory_projection(
    n_spins: int,
    *,
    fanout: int = DEFAULT_FANOUT,
    coupling_bits: int = COUPLING_BITS,
) -> dict[str, Any]:
    """Project sparse q8.8 row storage using the real n=64 uploaded fan-out."""

    index_bits = max(1, int(ceil(log2(max(2, int(n_spins))))))
    coupling_bits_total = int(n_spins) * int(fanout) * int(coupling_bits)
    index_bits_total = int(n_spins) * int(fanout) * index_bits
    bias_bits_total = int(n_spins) * BIAS_BITS
    spin_bits_total = int(n_spins)
    total_bits = coupling_bits_total + index_bits_total + bias_bits_total + spin_bits_total
    return {
        "layout": "sparse_q8_8_row_table",
        "formula": "n*k*(coupling_bits + ceil(log2(n))) + n*bias_bits + n*spin_bits",
        "n_spins": int(n_spins),
        "fanout_k": int(fanout),
        "coupling_bits_per_entry": int(coupling_bits),
        "neighbor_index_bits_per_entry": index_bits,
        "coupling_bits": coupling_bits_total,
        "neighbor_index_bits": index_bits_total,
        "bias_bits": bias_bits_total,
        "spin_state_bits": spin_bits_total,
        "total_bits": total_bits,
        "total_bytes": _bytes_for_bits(total_bits),
        "bram36_blocks_min": bram36_blocks(total_bits),
        "bram36_budget_pct": round(100.0 * bram36_blocks(total_bits) / KV260_BRAM36_BUDGET, 6),
    }


def dual_bram_projection(n_spins: int, *, fanout: int = DEFAULT_FANOUT) -> dict[str, Any]:
    """Project the p-bit/SSQA-inspired two-bank delay layout for sparse rows.

    Bank A is the read snapshot: sparse coupling rows, sparse neighbor indices,
    bias terms, and current spins. Bank B holds delayed writes: next spins,
    quantized local-field cache, RNG thresholds, and phase flags. Couplings are
    not double-counted because the local Exp 1320/1348 sketches keep them in the
    read bank and swap phase-visible state at the boundary.
    """

    bank_a = sparse_memory_projection(n_spins, fanout=fanout)
    bank_b_bits = int(n_spins) * (1 + FIELD_CACHE_BITS + RNG_THRESHOLD_BITS + 1)
    bank_a_blocks = int(bank_a["bram36_blocks_min"])
    bank_b_blocks = bram36_blocks(bank_b_bits)
    total_bits = int(bank_a["total_bits"]) + bank_b_bits
    return {
        "layout": "dual_bram_ssqa_delay_sparse_snapshot",
        "formula": (
            "bank_a=sparse_q8_8_rows+bias+current_spins; "
            "bank_b=n*(next_spin + field_cache_bits + rng_threshold_bits + phase_flag)"
        ),
        "n_spins": int(n_spins),
        "fanout_k": int(fanout),
        "bank_a_bits": int(bank_a["total_bits"]),
        "bank_b_bits": bank_b_bits,
        "bank_a_bram36_blocks_min": bank_a_blocks,
        "bank_b_bram36_blocks_min": bank_b_blocks,
        "total_bits": total_bits,
        "total_bytes": _bytes_for_bits(total_bits),
        "bram36_blocks_min": bank_a_blocks + bank_b_blocks,
        "bram36_budget_pct": round(
            100.0 * (bank_a_blocks + bank_b_blocks) / KV260_BRAM36_BUDGET,
            6,
        ),
        "bank_b_fields": {
            "next_spin_bits_per_spin": 1,
            "field_cache_bits_per_spin": FIELD_CACHE_BITS,
            "rng_threshold_bits_per_spin": RNG_THRESHOLD_BITS,
            "phase_flag_bits_per_spin": 1,
        },
    }


def _lut_pressure(n_spins: int, model: str, *, fanout: int) -> dict[str, Any]:
    if model == "dense_q8_8":
        estimated = 290_000 if int(n_spins) == 128 else 1_160_000
        formula = "n=128 dense LUT basis from v4 spec; n=256 scales coupling-dominated logic by (256/128)^2"
        evidence = "hardware/kv260/ising_sampler_v4_spec.md dense estimate"
    else:
        estimated = int(n_spins) * int(fanout) * 14 + int(n_spins) * 25 + 4_000
        formula = "n*k*14 LUT mult-add estimate + n*25 EMA estimate + 4000 control/AXI LUTs"
        evidence = "hardware/kv260/ising_sampler_v4_spec.md sparse K=16 breakdown"
    return {
        "estimated_lut": estimated,
        "formula": formula,
        "evidence": evidence,
        "kv260_lut_budget": KV260_LUT_BUDGET,
        "lut_budget_pct": round(100.0 * estimated / KV260_LUT_BUDGET, 6),
        "fits_kv260_lut_budget": estimated <= KV260_LUT_BUDGET,
        "projection_not_synthesis": True,
    }


def _unknown_ff_pressure(n_spins: int) -> dict[str, Any]:
    return {
        "total_ff_estimate": "unknown",
        "known_state_floor_bits": int(n_spins) * (1 + FIELD_CACHE_BITS),
        "reason": "no FF utilization report exists for the projected n=128/n=256 designs",
    }


def projection_models(fanout: int) -> dict[str, Any]:
    """Return the formulas used for the three projection families."""

    return {
        "dense_q8_8": {
            "memory_formula": "n*n*16 + n*16 + n bits",
            "lut_formula": "290000 LUTs at n=128 from local v4 spec; n=256 uses 4x dense coupling scale",
            "assumption": "full row-oriented dense coupling table with q8.8 coefficients",
        },
        "sparse_q8_8_k16": {
            "memory_formula": "n*k*(16 + ceil(log2(n))) + n*16 + n bits",
            "lut_formula": "n*k*14 + n*25 + 4000",
            "assumption": f"k={int(fanout)} sparse fan-out inherited from the real n=64 upload",
        },
        "dual_bram_ssqa_delay_k16": {
            "memory_formula": (
                "bank_a sparse_q8_8 memory plus bank_b n*(1+18+16+1) delayed-update bits; "
                "BRAM blocks are rounded per bank"
            ),
            "lut_formula": "same sparse compute estimate; extra arbitration LUTs are unknown without RTL",
            "assumption": "dual-BRAM p-bit/SSQA delay layout from Exp 1320/1348 sketches, not synthesis",
        },
    }


def _latency_by_count(rows: Sequence[Any], field: str) -> dict[str, float]:
    by_count: dict[int, list[float]] = {}
    for row in rows:
        if (
            isinstance(row, dict)
            and isinstance(row.get("n_samples"), int)
            and isinstance(row.get(field), (int, float))
        ):
            by_count.setdefault(int(row["n_samples"]), []).append(float(row[field]))
    return {
        str(count): round(sorted(values)[len(values) // 2], 6)
        for count, values in sorted(by_count.items())
        if values
    }


def _extract_n64_evidence(
    exp2913: Mapping[str, Any],
    exp2898: Mapping[str, Any],
    exp2912: Mapping[str, Any],
) -> dict[str, Any]:
    problem_payload = exp2898.get("problem_payload", {})
    if not isinstance(problem_payload, dict):
        problem_payload = {}
    rows = exp2898.get("sample_count_sweep_results", [])
    rows = rows if isinstance(rows, list) else []
    return {
        "source_exp2913_honest_verdict": exp2913.get("honest_verdict", ""),
        "n_spins": problem_payload.get("n_spins", exp2912.get("n_spins")),
        "sparse_fanout_k": problem_payload.get("max_degree_uploaded", DEFAULT_FANOUT),
        "random_seeds_used": problem_payload.get("random_seeds_used", exp2912.get("random_seeds_used", [])),
        "sample_count_sweep": problem_payload.get("n_sample_counts", exp2912.get("sample_count_sweep", [])),
        "kv260_latency_us_median_by_sample_count": _latency_by_count(
            rows, "per_sample_wall_clock_us_median"
        ),
        "kv260_latency_us_p95_by_sample_count": _latency_by_count(
            rows, "per_sample_wall_clock_us_p95"
        ),
        "cpu_latency_us_median_by_sample_count": exp2912.get(
            "cpu_latency_us_median_by_sample_count",
            {},
        ),
        "speedup_ratio_median_by_sample_count": exp2913.get(
            "speedup_ratio_median_by_sample_count",
            {},
        ),
        "topology_summary": {
            "layout": "uploaded_sparse_q8_8_rows",
            "max_degree_uploaded": problem_payload.get("max_degree_uploaded", DEFAULT_FANOUT),
            "problem_count": len(problem_payload.get("problems", []))
            if isinstance(problem_payload.get("problems"), list)
            else 0,
        },
        "resource_fields_present_in_exp2913_upstreams": False,
        "resource_unknowns": [
            "Exp 2913/2898/2912 do not carry placed LUT, FF, BRAM, or DSP utilization fields.",
            "n=64 RTL comments and v4 spec estimates are treated as projection inputs, not fresh synthesis.",
        ],
    }


def _projection_for_n(n_spins: int, *, fanout: int) -> dict[str, Any]:
    dense_memory = dense_memory_projection(n_spins)
    sparse_memory = sparse_memory_projection(n_spins, fanout=fanout)
    dual_memory = dual_bram_projection(n_spins, fanout=fanout)
    return {
        "n_spins": int(n_spins),
        "dense_q8_8": {
            "memory": dense_memory,
            "lut_pressure": _lut_pressure(n_spins, "dense_q8_8", fanout=fanout),
            "ff_pressure": _unknown_ff_pressure(n_spins),
            "bram_pressure": {
                "fits_kv260_bram36_budget": dense_memory["bram36_blocks_min"]
                <= KV260_BRAM36_BUDGET
            },
        },
        "sparse_q8_8_k16": {
            "memory": sparse_memory,
            "lut_pressure": _lut_pressure(n_spins, "sparse_q8_8_k16", fanout=fanout),
            "ff_pressure": _unknown_ff_pressure(n_spins),
            "bram_pressure": {
                "fits_kv260_bram36_budget": sparse_memory["bram36_blocks_min"]
                <= KV260_BRAM36_BUDGET
            },
        },
        "dual_bram_ssqa_delay_k16": {
            "memory": dual_memory,
            "lut_pressure": {
                **_lut_pressure(n_spins, "sparse_q8_8_k16", fanout=fanout),
                "extra_dual_bram_control_lut": "unknown",
            },
            "ff_pressure": _unknown_ff_pressure(n_spins),
            "bram_pressure": {
                "fits_kv260_bram36_budget": dual_memory["bram36_blocks_min"]
                <= KV260_BRAM36_BUDGET
            },
        },
    }


def blocked_artifact(*, duration_s: float, assumptions: Sequence[str]) -> dict[str, Any]:
    """Build the fail-closed artifact required when clean Exp 2913 is absent."""

    return {
        "honest_verdict": "blocked_clean_kv260_basis_missing",
        "kv260_scaling_projection_ready": False,
        "projection_only": True,
        "no_new_hardware_run": True,
        "source_artifacts": list(SOURCE_ARTIFACTS),
        "n64_real_evidence_summary": {},
        "projection_models": {},
        "n128_projection": {},
        "n256_projection": {},
        "assumptions": list(assumptions),
        "not_a_speedup_claim": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "run_date": RUN_DATE,
    }


def build_artifact(
    exp2913: Mapping[str, Any],
    exp2898: Mapping[str, Any],
    exp2912: Mapping[str, Any],
    *,
    duration_s: float,
) -> dict[str, Any]:
    """Build the ready projection artifact from already-loaded JSON objects."""

    if exp2913.get("hardware_speedup_claim_eligible") is not True:
        return blocked_artifact(
            duration_s=duration_s,
            assumptions=["Exp 2913 hardware_speedup_claim_eligible was not true."],
        )

    n64 = _extract_n64_evidence(exp2913, exp2898, exp2912)
    fanout = int(n64.get("sparse_fanout_k") or DEFAULT_FANOUT)
    return {
        "honest_verdict": "complete: kv260_pbit_ssqa_scaling_projection_ready_no_new_hardware_claim",
        "kv260_scaling_projection_ready": True,
        "projection_only": True,
        "no_new_hardware_run": True,
        "source_artifacts": list(SOURCE_ARTIFACTS),
        "n64_real_evidence_summary": n64,
        "projection_models": projection_models(fanout),
        "n128_projection": _projection_for_n(128, fanout=fanout),
        "n256_projection": _projection_for_n(256, fanout=fanout),
        "assumptions": [
            "The real evidence basis is Exp 2913 plus its Exp 2898 and Exp 2912 upstream artifacts.",
            "The prompt's hardware_vs_cpu Exp 2913 filename is absent; the repo's actual Exp 2913 artifact is used.",
            "No new KV260 command, synthesis run, bitstream build, or sampler benchmark was performed.",
            "Dense memory uses full row-oriented q8.8 n*n coupling storage rather than symmetry compression.",
            f"Sparse and dual-BRAM projections inherit k={fanout} from the real n=64 upload.",
            "Dual-BRAM accounting follows Exp 1320/1348 p-bit sketches as a memory-layout projection only.",
            "Total FF and dual-BRAM arbitration LUTs are unknown without projected RTL synthesis.",
        ],
        "not_a_speedup_claim": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "run_date": RUN_DATE,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema and the no-new-claim invariants before writing."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:  # pragma: no cover - defensive schema guard
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["projection_only"] is not True:
        raise ValueError("projection_only must remain true")
    if artifact["no_new_hardware_run"] is not True:  # pragma: no cover
        raise ValueError("no_new_hardware_run must remain true")
    if artifact["not_a_speedup_claim"] is not True:  # pragma: no cover
        raise ValueError("not_a_speedup_claim must remain true")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:  # pragma: no cover
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    if artifact["run_date"] != RUN_DATE:  # pragma: no cover
        raise ValueError("run_date must be 20260523")


def run_experiment(
    root_path: Path | str = REPO_ROOT,
    *,
    write: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Run Exp 2930 as a read-only aggregation/projection and write JSON."""

    root = Path(root_path)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp2913 = _read_json(root / EXP2913_REL_PATH)
    exp2898 = _read_json(root / EXP2898_REL_PATH)
    exp2912 = _read_json(root / EXP2912_REL_PATH)
    if not exp2913:
        artifact = blocked_artifact(
            duration_s=_duration(started, now_s),
            assumptions=["Exp 2913 artifact is absent or malformed."],
        )
    else:
        artifact = build_artifact(
            exp2913,
            exp2898,
            exp2912,
            duration_s=_duration(started, now_s),
        )

    validate_artifact(artifact)
    if write:
        _write_json(root / OUTPUT_REL_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
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
