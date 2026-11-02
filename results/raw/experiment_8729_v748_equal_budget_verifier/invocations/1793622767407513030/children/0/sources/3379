"""Exp 2913 KV260 hardware/CPU claim-boundary aggregation.

Spec: REQ-HW-065, SCENARIO-HW-065.

This module only compares upstream artifacts. It does not touch the KV260 board
or rerun the CPU sampler, because a speedup statement is only honest when the
already-recorded hardware and CPU measurements share the same problem basis,
seeds, sample counts, and per-sample microsecond timing units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
KV260_ARTIFACT_REL_PATH = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
CPU_ARTIFACT_REL_PATH = Path(
    "results/experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1.json"
)
OUTPUT_REL_PATH = Path(
    "results/experiment_2913_kv260_hardware_cpu_claim_boundary_v1.json"
)

RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
CLAIM_SCOPE = "matched n=64 sparse Ising KV260 hardware-smoke versus CPU Gibbs baseline"
RATIO_DEFINITION = (
    "CPU per-sample microsecond latency divided by KV260 per-sample wall-clock "
    "microsecond latency; values greater than 1 mean lower KV260 latency."
)

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "kv260_claim_boundary_ready",
    "same_basis_verified",
    "hardware_speedup_claim_eligible",
    "speedup_ratio_median_by_sample_count",
    "speedup_ratio_p95_by_sample_count",
    "comparison_notes",
    "matrix_row_candidate",
    "paper_claim_boundary",
    "speedup_claim_made",
    "inference_substrate",
    "duration_s",
    "run_date",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_canonical(value: Any) -> str:
    """Return a stable SHA-256 for JSON-like tensors imported from artifacts."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute median of an empty sequence")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute p95 of an empty sequence")
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _positive_float(row: Mapping[str, Any], field: str) -> bool:
    value = row.get(field)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0


def _duration(started_s: float, now_s: float | None) -> float:
    now = time.perf_counter() if now_s is None else now_s
    return round(max(0.0, now - started_s), 6)


def _seed_list(payload: Mapping[str, Any]) -> list[int]:
    seeds = payload.get("random_seeds_used")
    if isinstance(seeds, list) and all(isinstance(seed, int) for seed in seeds):
        return list(seeds)
    nested = payload.get("problem_payload", {})
    if isinstance(nested, dict):
        nested_seeds = nested.get("random_seeds_used")
        if isinstance(nested_seeds, list) and all(isinstance(seed, int) for seed in nested_seeds):
            return list(nested_seeds)
    return []


def _n_spins(payload: Mapping[str, Any]) -> int | None:
    n_spins = payload.get("n_spins")
    if isinstance(n_spins, int):
        return n_spins
    problem_payload = payload.get("problem_payload", {})
    if isinstance(problem_payload, dict) and isinstance(problem_payload.get("n_spins"), int):
        return int(problem_payload["n_spins"])
    spec = payload.get("ising_problem_spec", {})
    if isinstance(spec, dict) and isinstance(spec.get("n_spins"), int):
        return int(spec["n_spins"])
    return None


def _kv260_basis_checksums(payload: Mapping[str, Any]) -> dict[int, dict[str, str]]:
    problem_payload = payload.get("problem_payload", {})
    if not isinstance(problem_payload, dict):
        return {}
    problems = problem_payload.get("problems", [])
    if not isinstance(problems, list):
        return {}

    checksums: dict[int, dict[str, str]] = {}
    for problem in problems:
        if not isinstance(problem, dict) or not isinstance(problem.get("random_seed"), int):
            continue
        upload = problem.get("upload")
        if not isinstance(upload, dict):
            continue
        if not all(field in upload for field in ("adjacency", "couplings_q88", "h_q88")):
            continue
        checksums[int(problem["random_seed"])] = {
            "topology": sha256_canonical(upload["adjacency"]),
            "couplings": sha256_canonical(upload["couplings_q88"]),
            "fields": sha256_canonical(upload["h_q88"]),
        }
    return checksums


def _kv260_latency_rows(payload: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    rows = payload.get("sample_count_sweep_results", [])
    if not isinstance(rows, list):
        return {}
    indexed: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("seed"), int) and isinstance(
            row.get("n_samples"), int
        ):
            indexed[(int(row["seed"]), int(row["n_samples"]))] = row
    return indexed


def _cpu_latency_rows(payload: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    rows = payload.get("cpu_per_seed_results", [])
    if not isinstance(rows, list):
        return {}
    indexed: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("seed"), int) and isinstance(
            row.get("sample_count"), int
        ):
            indexed[(int(row["seed"]), int(row["sample_count"]))] = row
    return indexed


def _counts_by_seed(indexed_rows: Mapping[tuple[int, int], Mapping[str, Any]]) -> dict[int, list[int]]:
    by_seed: dict[int, list[int]] = {}
    for seed, count in indexed_rows:
        by_seed.setdefault(seed, []).append(count)
    return {seed: sorted(counts) for seed, counts in sorted(by_seed.items())}


def _basis_match_notes(
    kv260_basis: Mapping[int, Mapping[str, str]],
    cpu_rows: Mapping[tuple[int, int], Mapping[str, Any]],
    cpu_payload: Mapping[str, Any],
) -> list[str]:
    notes: list[str] = []
    if cpu_payload.get("matched_sparse_topology") is not True:
        notes.append("sparse_topology_not_marked_matched_by_exp2912")
    if cpu_payload.get("matched_coupling_tensor") is not True:
        notes.append("coupling_tensor_not_marked_matched_by_exp2912")
    if cpu_payload.get("matched_field_tensor") is not True:
        notes.append("field_tensor_not_marked_matched_by_exp2912")

    for (seed, _count), row in sorted(cpu_rows.items()):
        expected = kv260_basis.get(seed)
        if expected is None:
            notes.append("sparse_topology_mismatch")
            continue
        if row.get("matched_sparse_topology_checksum") != expected["topology"]:
            notes.append("sparse_topology_mismatch")
        if row.get("matched_coupling_tensor_checksum") != expected["couplings"]:
            notes.append("coupling_tensor_mismatch")
        if row.get("matched_field_tensor_checksum") != expected["fields"]:
            notes.append("field_tensor_mismatch")
    return sorted(set(notes))


def _timing_fields_match(
    kv260_rows: Mapping[tuple[int, int], Mapping[str, Any]],
    cpu_rows: Mapping[tuple[int, int], Mapping[str, Any]],
) -> bool:
    for key in sorted(kv260_rows):
        kv260_row = kv260_rows[key]
        cpu_row = cpu_rows.get(key, {})
        if not (
            _positive_float(kv260_row, "per_sample_wall_clock_us_median")
            and _positive_float(kv260_row, "per_sample_wall_clock_us_p95")
            and _positive_float(cpu_row, "cpu_latency_us_median")
            and _positive_float(cpu_row, "cpu_latency_us_p95")
        ):
            return False
    return bool(kv260_rows)


def _compute_ratios(
    kv260_rows: Mapping[tuple[int, int], Mapping[str, Any]],
    cpu_rows: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    per_seed: list[dict[str, Any]] = []
    median_ratios_by_count: dict[int, list[float]] = {}
    p95_ratios_by_count: dict[int, list[float]] = {}

    for seed, count in sorted(kv260_rows, key=lambda item: (item[1], item[0])):
        kv260_row = kv260_rows[(seed, count)]
        cpu_row = cpu_rows[(seed, count)]
        kv260_median = float(kv260_row["per_sample_wall_clock_us_median"])
        kv260_p95 = float(kv260_row["per_sample_wall_clock_us_p95"])
        cpu_median = float(cpu_row["cpu_latency_us_median"])
        cpu_p95 = float(cpu_row["cpu_latency_us_p95"])
        median_ratio = round(cpu_median / kv260_median, 6)
        p95_ratio = round(cpu_p95 / kv260_p95, 6)
        median_ratios_by_count.setdefault(count, []).append(median_ratio)
        p95_ratios_by_count.setdefault(count, []).append(p95_ratio)
        per_seed.append(
            {
                "seed": seed,
                "sample_count": count,
                "kv260_median_us": kv260_median,
                "cpu_median_us": cpu_median,
                "speedup_ratio_median": median_ratio,
                "kv260_p95_us": kv260_p95,
                "cpu_p95_us": cpu_p95,
                "speedup_ratio_p95": p95_ratio,
            }
        )

    medians = {
        str(count): round(_median(values), 6) for count, values in sorted(median_ratios_by_count.items())
    }
    p95s = {str(count): round(_p95(values), 6) for count, values in sorted(p95_ratios_by_count.items())}
    return per_seed, medians, p95s


def _matrix_row_candidate(
    *,
    eligible: bool,
    n_spins: int | None,
    seeds: Sequence[int],
    sample_counts: Sequence[int],
    medians: Mapping[str, float],
    p95s: Mapping[str, float],
    notes: Sequence[str],
) -> dict[str, Any]:
    return {
        "experiment_id": "exp2913",
        "eligible_for_matrix_v9": bool(eligible),
        "eligible_for_paper_v6": bool(eligible),
        "claim_scope": CLAIM_SCOPE if eligible else "no numeric KV260/CPU speedup claim",
        "n_spins": n_spins,
        "random_seeds_used": list(seeds),
        "sample_count_sweep": list(sample_counts),
        "speedup_ratio_definition": RATIO_DEFINITION,
        "speedup_ratio_median_by_sample_count": dict(medians),
        "speedup_ratio_p95_by_sample_count": dict(p95s),
        "blocked_conditions": [] if eligible else list(notes),
        "upstream_artifacts": [
            KV260_ARTIFACT_REL_PATH.as_posix(),
            CPU_ARTIFACT_REL_PATH.as_posix(),
        ],
    }


def _paper_claim_boundary(
    *,
    eligible: bool,
    medians: Mapping[str, float],
    p95s: Mapping[str, float],
    notes: Sequence[str],
) -> str:
    if not eligible:
        reason = "; ".join(notes) if notes else "matched evidence was not established"
        return (
            "No numeric KV260/CPU speedup claim is eligible for matrix v9 or paper-v6 "
            f"because {reason}. KV260 latency may still be described as a board-level "
            "hardware-smoke measurement, but not as a CPU speedup."
        )

    ratio_text = ", ".join(
        f"n_samples={count}: median {medians[count]:.2f}x, p95 {p95s[count]:.2f}x"
        for count in sorted(medians, key=lambda item: int(item))
    )
    return (
        "A bounded numeric KV260/CPU latency claim is eligible for the matched n=64 "
        f"sparse Ising workload: CPU/KV260 per-sample latency ratios are {ratio_text}. "
        "The claim is restricted to Exp 2898 hardware-smoke timing versus Exp 2912 "
        "same-basis CPU Gibbs timing; it is not a broad FPGA acceleration claim."
    )


def _blocked_artifact(*, verdict: str, duration_s: float, note: str) -> dict[str, Any]:
    return {
        "honest_verdict": verdict,
        "kv260_claim_boundary_ready": False,
        "same_basis_verified": False,
        "hardware_speedup_claim_eligible": False,
        "speedup_ratio_median_by_sample_count": {},
        "speedup_ratio_p95_by_sample_count": {},
        "comparison_notes": [note],
        "matrix_row_candidate": {},
        "paper_claim_boundary": _paper_claim_boundary(eligible=False, medians={}, p95s={}, notes=[note]),
        "speedup_claim_made": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
        "per_seed_speedup_ratios": [],
    }


def build_artifact(
    kv260_payload: Mapping[str, Any],
    cpu_payload: Mapping[str, Any],
    *,
    duration_s: float,
) -> dict[str, Any]:
    """Build the claim-boundary artifact from already-loaded upstream JSON."""

    kv260_rows = _kv260_latency_rows(kv260_payload)
    cpu_rows = _cpu_latency_rows(cpu_payload)
    kv260_n = _n_spins(kv260_payload)
    cpu_n = _n_spins(cpu_payload)
    kv260_seeds = _seed_list(kv260_payload)
    cpu_seeds = _seed_list(cpu_payload)
    kv260_counts_by_seed = _counts_by_seed(kv260_rows)
    cpu_counts_by_seed = _counts_by_seed(cpu_rows)
    kv260_basis = _kv260_basis_checksums(kv260_payload)

    notes: list[str] = []
    n_spins_match = kv260_n is not None and kv260_n == cpu_n
    if n_spins_match:
        notes.append(f"matched_n_spins={kv260_n}")
    else:
        notes.append("n_spins_mismatch")

    seeds_match = kv260_seeds == cpu_seeds and bool(kv260_seeds)
    if seeds_match:
        notes.append("matched_random_seeds")
    else:
        notes.append("seeds_mismatch")

    sample_counts_match = kv260_counts_by_seed == cpu_counts_by_seed and bool(kv260_counts_by_seed)
    if sample_counts_match:
        notes.append("matched_sample_counts")
    else:
        notes.append("sample_counts_mismatch")

    basis_notes = _basis_match_notes(kv260_basis, cpu_rows, cpu_payload)
    if basis_notes:
        notes.extend(basis_notes)
    else:
        notes.append("matched_sparse_topology_and_tensors")

    timing_units_match = _timing_fields_match(kv260_rows, cpu_rows)
    if timing_units_match:
        notes.append("matched_timing_units=per_sample_microseconds")
    else:
        notes.append("timing_units_or_latency_fields_mismatch")

    same_basis_verified = (
        n_spins_match
        and seeds_match
        and sample_counts_match
        and not basis_notes
    )
    eligible = same_basis_verified and timing_units_match

    per_seed_ratios: list[dict[str, Any]] = []
    medians: dict[str, float] = {}
    p95s: dict[str, float] = {}
    if eligible:
        per_seed_ratios, medians, p95s = _compute_ratios(kv260_rows, cpu_rows)

    sample_counts = sorted({count for _seed, count in kv260_rows})
    candidate = _matrix_row_candidate(
        eligible=eligible,
        n_spins=kv260_n if n_spins_match else None,
        seeds=kv260_seeds if seeds_match else [],
        sample_counts=sample_counts if sample_counts_match else [],
        medians=medians,
        p95s=p95s,
        notes=notes,
    )

    verdict = (
        "complete: kv260_same_basis_hardware_cpu_speedup_claim_eligible"
        if eligible
        else "complete: kv260_claim_boundary_ready_no_speedup_claim"
    )
    return {
        "honest_verdict": verdict,
        "kv260_claim_boundary_ready": True,
        "same_basis_verified": bool(same_basis_verified),
        "hardware_speedup_claim_eligible": bool(eligible),
        "speedup_ratio_median_by_sample_count": medians,
        "speedup_ratio_p95_by_sample_count": p95s,
        "comparison_notes": sorted(set(notes)),
        "matrix_row_candidate": candidate,
        "paper_claim_boundary": _paper_claim_boundary(
            eligible=eligible, medians=medians, p95s=p95s, notes=sorted(set(notes))
        ),
        "speedup_claim_made": bool(eligible),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
        "per_seed_speedup_ratios": per_seed_ratios,
        "upstream_artifacts": [
            KV260_ARTIFACT_REL_PATH.as_posix(),
            CPU_ARTIFACT_REL_PATH.as_posix(),
        ],
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 2913 schema and claim gate before writing."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    if artifact["run_date"] != RUN_DATE:
        raise ValueError("run_date must be 20260523")
    if artifact["speedup_claim_made"] is not artifact["hardware_speedup_claim_eligible"]:
        raise ValueError("speedup_claim_made must match hardware_speedup_claim_eligible")
    if artifact["hardware_speedup_claim_eligible"]:
        if artifact["same_basis_verified"] is not True:
            raise ValueError("eligible speedup requires same_basis_verified")
        if not artifact["speedup_ratio_median_by_sample_count"]:
            raise ValueError("eligible speedup requires median ratios")
        if not artifact["speedup_ratio_p95_by_sample_count"]:
            raise ValueError("eligible speedup requires p95 ratios")


def run_experiment(
    root_path: Path | str = REPO_ROOT,
    *,
    write: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Run the Exp 2913 aggregation and optionally write the deliverable JSON."""

    root = Path(root_path)
    started = time.perf_counter() if started_s is None else started_s
    cpu_payload = _read_json(root / CPU_ARTIFACT_REL_PATH)
    if cpu_payload.get("same_basis_cpu_baseline_ready") is not True:
        artifact = _blocked_artifact(
            verdict="blocked_cpu_baseline_not_ready",
            duration_s=_duration(started, now_s),
            note="Exp 2912 same_basis_cpu_baseline_ready is not true.",
        )
        validate_artifact(artifact)
        if write:
            _write_json(root / OUTPUT_REL_PATH, artifact)
        return artifact

    kv260_payload = _read_json(root / KV260_ARTIFACT_REL_PATH)
    if not kv260_payload:
        artifact = _blocked_artifact(
            verdict="blocked_kv260_artifact_not_ready",
            duration_s=_duration(started, now_s),
            note="Exp 2898 KV260 artifact is missing, malformed, or not loadable.",
        )
        validate_artifact(artifact)
        if write:
            _write_json(root / OUTPUT_REL_PATH, artifact)
        return artifact

    artifact = build_artifact(
        kv260_payload,
        cpu_payload,
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
