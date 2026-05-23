"""Exp 2912 same-basis CPU Gibbs baseline for the Exp 2898 KV260 run.

Spec: REQ-HW-064, SCENARIO-HW-064.

The important constraint is provenance, not performance theater.  Exp 2898
uploaded a sparse q8.8 Ising problem to the KV260; this module refuses to build
a replacement problem if that exact upload is absent.  When the basis is
recoverable, it runs a deterministic CPU Gibbs reference on those same sparse
rows and records timing/energy checksums without making a hardware speedup
claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_KV260_ARTIFACT = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
RESULT_ARTIFACT = Path("results/experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1.json")

EXPERIMENT_ID = 2912
RUN_DATE = "20260523"
N_SPINS = 64
DEFAULT_SAMPLE_COUNTS = (100, 1000, 10000)
INFERENCE_SUBSTRATE = "cpu_sampler"
CPU_UPDATE_SCHEDULE = "cpu_sequential_round_robin_uploaded_sparse_rows_one_sweep_per_sample"

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "same_basis_cpu_baseline_ready",
    "upstream_kv260_artifact",
    "n_spins",
    "matched_sparse_topology",
    "matched_coupling_tensor",
    "matched_field_tensor",
    "random_seeds_used",
    "sample_count_sweep",
    "cpu_per_seed_results",
    "cpu_latency_us_median_by_sample_count",
    "cpu_latency_us_p95_by_sample_count",
    "reproducibility_checksum",
    "speedup_claim_made",
    "inference_substrate",
    "duration_s",
    "run_date",
}


@dataclass(frozen=True)
class SparseProblemBasis:
    """One recovered Exp 2898 sparse upload.

    The CPU baseline intentionally uses the q8.8 upload tensors rather than a
    regenerated dense matrix, because those are the values the board actually
    saw through AXI-Lite.
    """

    seed: int
    n_spins: int
    beta_final_q88: int
    adjacency: np.ndarray
    couplings_q88: np.ndarray
    h_q88: np.ndarray
    topology_checksum: str
    coupling_tensor_checksum: str
    field_tensor_checksum: str
    dense_j_matrix_checksum: str
    dense_h_vector_checksum: str


@dataclass(frozen=True)
class RecoveredProblemBasis:
    """Recovered same-basis inputs shared by every CPU run."""

    n_spins: int
    random_seeds_used: list[int]
    sample_count_sweep: list[int]
    problems: list[SparseProblemBasis]
    upstream_reproducibility_checksum: str


class ProblemBasisUnrecoverableError(ValueError):
    """Raised when Exp 2898 exists but lacks exact tensors needed by Exp 2912."""

    def __init__(self, missing_fields: Sequence[str]) -> None:
        self.missing_fields = list(missing_fields)
        super().__init__(", ".join(self.missing_fields))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_canonical(value: Any) -> str:
    """Hash JSON-compatible values in the same stable form across runs."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _int_array(value: Any, shape: tuple[int, ...], field: str, missing: list[str]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError):
        missing.append(field)
        return np.empty(shape, dtype=np.int64)
    if array.shape != shape:
        missing.append(field)
    return array


def _sample_counts_from_sweep(exp2898: dict[str, Any]) -> list[int]:
    rows = exp2898.get("sample_count_sweep_results")
    if not isinstance(rows, list):
        return []
    counts = {
        int(row["n_samples"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("n_samples"), int)
    }
    return sorted(counts)


def recover_problem_basis(exp2898: dict[str, Any]) -> RecoveredProblemBasis:
    """Recover the exact sparse upload basis from an Exp 2898 artifact.

    This function is deliberately strict.  Regenerating the SK matrix from the
    seed would be easy, but it would hide schema drift if the uploaded topology
    or q8.8 tensors ever changed.
    """
    missing: list[str] = []
    payload = exp2898.get("problem_payload")
    if not isinstance(payload, dict):
        raise ProblemBasisUnrecoverableError(["problem_payload"])

    n_spins = payload.get("n_spins", exp2898.get("ising_problem_spec", {}).get("n_spins"))
    if n_spins != N_SPINS:
        missing.append("problem_payload.n_spins")

    random_seeds = payload.get("random_seeds_used", exp2898.get("random_seeds_used"))
    if not isinstance(random_seeds, list) or not all(isinstance(seed, int) for seed in random_seeds):
        missing.append("problem_payload.random_seeds_used")
        random_seeds = []

    sample_counts = payload.get("n_sample_counts")
    if not isinstance(sample_counts, list) or not all(
        isinstance(count, int) for count in sample_counts
    ):
        sample_counts = _sample_counts_from_sweep(exp2898)
    if not sample_counts:
        missing.append("problem_payload.n_sample_counts")

    specs_by_seed = {
        int(spec.get("random_seed")): spec
        for spec in payload.get("ising_problem_specs", [])
        if isinstance(spec, dict) and isinstance(spec.get("random_seed"), int)
    }

    problem_rows = payload.get("problems")
    if not isinstance(problem_rows, list) or not problem_rows:
        raise ProblemBasisUnrecoverableError(["problem_payload.problems"])

    problems: list[SparseProblemBasis] = []
    for index, row in enumerate(problem_rows):
        prefix = f"problem_payload.problems[{index}]"
        if not isinstance(row, dict):
            missing.append(prefix)
            continue

        seed = row.get("random_seed")
        if not isinstance(seed, int):
            missing.append(f"{prefix}.random_seed")
            seed = -1

        upload = row.get("upload")
        if not isinstance(upload, dict):
            missing.append(f"{prefix}.upload")
            continue

        max_degree = upload.get("max_degree")
        if not isinstance(max_degree, int) or max_degree <= 0:
            missing.append(f"{prefix}.upload.max_degree")
            max_degree = 0

        adjacency = _int_array(
            upload.get("adjacency"),
            (N_SPINS, max_degree),
            f"{prefix}.upload.adjacency",
            missing,
        )
        couplings_q88 = _int_array(
            upload.get("couplings_q88"),
            (N_SPINS, max_degree),
            f"{prefix}.upload.couplings_q88",
            missing,
        )
        h_q88 = _int_array(upload.get("h_q88"), (N_SPINS,), f"{prefix}.upload.h_q88", missing)

        spec = specs_by_seed.get(seed, {})
        dense_j_checksum = str(spec.get("j_matrix_sha256", ""))
        dense_h_checksum = str(spec.get("h_vector_sha256", ""))
        if len(dense_j_checksum) != 64:
            missing.append(f"{prefix}.j_matrix_sha256")
        if len(dense_h_checksum) != 64:
            missing.append(f"{prefix}.h_vector_sha256")

        beta_final_q88 = row.get("beta_final_q88", 256)
        if not isinstance(beta_final_q88, int):
            missing.append(f"{prefix}.beta_final_q88")
            beta_final_q88 = 256

        if not missing or all(not item.startswith(prefix) for item in missing):
            problems.append(
                SparseProblemBasis(
                    seed=seed,
                    n_spins=N_SPINS,
                    beta_final_q88=beta_final_q88,
                    adjacency=adjacency,
                    couplings_q88=couplings_q88,
                    h_q88=h_q88,
                    topology_checksum=sha256_canonical(adjacency.tolist()),
                    coupling_tensor_checksum=sha256_canonical(couplings_q88.tolist()),
                    field_tensor_checksum=sha256_canonical(h_q88.tolist()),
                    dense_j_matrix_checksum=dense_j_checksum,
                    dense_h_vector_checksum=dense_h_checksum,
                )
            )

    if missing:
        raise ProblemBasisUnrecoverableError(sorted(set(missing)))
    if [problem.seed for problem in problems] != list(random_seeds):
        raise ProblemBasisUnrecoverableError(["problem_payload.problems.random_seed_order"])

    return RecoveredProblemBasis(
        n_spins=N_SPINS,
        random_seeds_used=list(random_seeds),
        sample_count_sweep=sorted(int(count) for count in sample_counts),
        problems=problems,
        upstream_reproducibility_checksum=str(exp2898.get("reproducibility_checksum", "")),
    )


def _sparse_energy(
    state: np.ndarray,
    adjacency: np.ndarray,
    couplings: np.ndarray,
    fields: np.ndarray,
) -> float:
    valid = (adjacency >= 0) & (adjacency < state.shape[0])
    safe_adjacency = np.where(valid, adjacency, 0)
    neighbor_state = state[safe_adjacency]
    pair_terms = couplings * state[:, None] * neighbor_state
    return float(-(fields @ state) - 0.5 * np.sum(pair_terms[valid]))


def run_cpu_gibbs_for_problem(problem: SparseProblemBasis, sample_count: int) -> dict[str, Any]:
    """Run one CPU Gibbs chain on the recovered sparse rows.

    One recorded CPU sample is one full round-robin sweep over spins 0..63.
    That keeps the baseline tied to the uploaded row order while avoiding a
    hidden hardware-speedup comparison.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    rng = np.random.default_rng(problem.seed)
    state = (rng.integers(0, 2, size=problem.n_spins, dtype=np.int8) * 2 - 1).astype(np.int8)
    adjacency = problem.adjacency
    couplings = problem.couplings_q88.astype(np.float64) / 256.0
    fields = problem.h_q88.astype(np.float64) / 256.0

    latencies_us: list[float] = []
    energies: list[float] = []

    for _ in range(sample_count):
        started_ns = time.perf_counter_ns()
        for spin_index in range(problem.n_spins):
            neighbors = adjacency[spin_index]
            valid = (neighbors >= 0) & (neighbors < problem.n_spins)
            local_field = fields[spin_index]
            if np.any(valid):
                local_field += float(couplings[spin_index, valid] @ state[neighbors[valid]])
            p_plus = _sigmoid(2.0 * (problem.beta_final_q88 / 256.0) * float(local_field))
            state[spin_index] = 1 if rng.random() < p_plus else -1
        latencies_us.append((time.perf_counter_ns() - started_ns) / 1000.0)
        energies.append(_sparse_energy(state, adjacency, couplings, fields))

    rounded_energies = [round(energy, 12) for energy in energies]
    final_state = [int(value) for value in state.tolist()]
    reproducibility_payload = {
        "seed": problem.seed,
        "sample_count": sample_count,
        "basis": {
            "topology": problem.topology_checksum,
            "couplings_q88": problem.coupling_tensor_checksum,
            "h_q88": problem.field_tensor_checksum,
        },
        "energy_trace_checksum": sha256_canonical(rounded_energies),
        "final_state": final_state,
        "update_schedule": CPU_UPDATE_SCHEDULE,
    }

    return {
        "seed": problem.seed,
        "sample_count": int(sample_count),
        "n_spins": problem.n_spins,
        "beta_final_q88": problem.beta_final_q88,
        "beta_final": problem.beta_final_q88 / 256.0,
        "update_schedule": CPU_UPDATE_SCHEDULE,
        "cpu_latency_us_median": _median(latencies_us),
        "cpu_latency_us_p95": _p95(latencies_us),
        "final_energy": rounded_energies[-1],
        "energy_trace_checksum": reproducibility_payload["energy_trace_checksum"],
        "final_state_checksum": sha256_canonical(final_state),
        "reproducibility_checksum": sha256_canonical(reproducibility_payload),
        "matched_sparse_topology_checksum": problem.topology_checksum,
        "matched_coupling_tensor_checksum": problem.coupling_tensor_checksum,
        "matched_field_tensor_checksum": problem.field_tensor_checksum,
        "latency_trace_checksum": sha256_canonical([round(value, 6) for value in latencies_us]),
    }


def _blocked_artifact(
    *,
    verdict: str,
    duration_s: float,
    missing_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "honest_verdict": verdict,
        "same_basis_cpu_baseline_ready": False,
        "upstream_kv260_artifact": UPSTREAM_KV260_ARTIFACT.as_posix(),
        "n_spins": N_SPINS,
        "matched_sparse_topology": False,
        "matched_coupling_tensor": False,
        "matched_field_tensor": False,
        "random_seeds_used": [],
        "sample_count_sweep": [],
        "cpu_per_seed_results": [],
        "cpu_latency_us_median_by_sample_count": {},
        "cpu_latency_us_p95_by_sample_count": {},
        "reproducibility_checksum": "",
        "speedup_claim_made": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "run_date": RUN_DATE,
        "missing_problem_basis_fields": list(missing_fields or []),
    }


def build_success_artifact(
    basis: RecoveredProblemBasis,
    cpu_results: list[dict[str, Any]],
    *,
    sample_counts: Sequence[int],
    duration_s: float,
) -> dict[str, Any]:
    """Assemble the deliverable JSON from recovered tensors and CPU runs."""
    medians_by_count: dict[str, float] = {}
    p95_by_count: dict[str, float] = {}
    for count in sample_counts:
        rows = [row for row in cpu_results if row["sample_count"] == count]
        medians_by_count[str(count)] = _median([row["cpu_latency_us_median"] for row in rows])
        p95_by_count[str(count)] = _p95([row["cpu_latency_us_p95"] for row in rows])

    basis_checksums = {
        "uploaded_sparse_topology_by_seed": {
            str(problem.seed): problem.topology_checksum for problem in basis.problems
        },
        "uploaded_coupling_tensor_by_seed": {
            str(problem.seed): problem.coupling_tensor_checksum for problem in basis.problems
        },
        "uploaded_field_tensor_by_seed": {
            str(problem.seed): problem.field_tensor_checksum for problem in basis.problems
        },
        "dense_j_matrix_by_seed": {
            str(problem.seed): problem.dense_j_matrix_checksum for problem in basis.problems
        },
        "dense_h_vector_by_seed": {
            str(problem.seed): problem.dense_h_vector_checksum for problem in basis.problems
        },
    }
    reproducibility_checksum = sha256_canonical(
        {
            "basis_checksums": basis_checksums,
            "cpu_result_checksums": [
                row["reproducibility_checksum"] for row in sorted(
                    cpu_results, key=lambda item: (item["seed"], item["sample_count"])
                )
            ],
            "sample_count_sweep": list(sample_counts),
            "upstream_reproducibility_checksum": basis.upstream_reproducibility_checksum,
        }
    )

    return {
        "honest_verdict": "complete: same_basis_cpu_gibbs_baseline_ready_no_speedup_claim",
        "same_basis_cpu_baseline_ready": True,
        "upstream_kv260_artifact": UPSTREAM_KV260_ARTIFACT.as_posix(),
        "n_spins": basis.n_spins,
        "matched_sparse_topology": True,
        "matched_coupling_tensor": True,
        "matched_field_tensor": True,
        "random_seeds_used": basis.random_seeds_used,
        "sample_count_sweep": list(sample_counts),
        "cpu_per_seed_results": sorted(
            cpu_results, key=lambda item: (item["seed"], item["sample_count"])
        ),
        "cpu_latency_us_median_by_sample_count": medians_by_count,
        "cpu_latency_us_p95_by_sample_count": p95_by_count,
        "reproducibility_checksum": reproducibility_checksum,
        "speedup_claim_made": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "run_date": RUN_DATE,
        "basis_checksums": basis_checksums,
        "cpu_update_schedule": CPU_UPDATE_SCHEDULE,
        "speedup_claim_note": "CPU baseline only; no hardware speedup is claimed.",
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the stable Exp 2912 schema before writing it to disk."""
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["speedup_claim_made"] is not False:
        raise ValueError("speedup_claim_made must remain false")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be cpu_sampler")
    if artifact["run_date"] != RUN_DATE:
        raise ValueError("run_date must be 20260523")
    if artifact["n_spins"] != N_SPINS:
        raise ValueError("n_spins must be 64")

    if artifact["same_basis_cpu_baseline_ready"]:
        for field in ("matched_sparse_topology", "matched_coupling_tensor", "matched_field_tensor"):
            if artifact[field] is not True:
                raise ValueError(f"{field} must be true for a ready baseline")
        if not artifact["cpu_per_seed_results"]:
            raise ValueError("ready baseline requires cpu_per_seed_results")
        checksum = artifact["reproducibility_checksum"]
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError("ready baseline requires a sha256 reproducibility_checksum")


def run_experiment(
    *,
    root_path: Path = REPO_ROOT,
    sample_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run Exp 2912 and write the deliverable artifact under ``root_path``."""
    started = time.perf_counter()
    upstream_path = root_path / UPSTREAM_KV260_ARTIFACT
    result_path = root_path / RESULT_ARTIFACT

    if not upstream_path.exists():
        artifact = _blocked_artifact(
            verdict="blocked_kv260_latency_artifact_missing",
            duration_s=time.perf_counter() - started,
        )
        validate_artifact(artifact)
        _write_json(result_path, artifact)
        return artifact

    exp2898 = json.loads(upstream_path.read_text(encoding="utf-8"))
    try:
        basis = recover_problem_basis(exp2898)
    except ProblemBasisUnrecoverableError as exc:
        artifact = _blocked_artifact(
            verdict="blocked_kv260_problem_basis_unrecoverable",
            duration_s=time.perf_counter() - started,
            missing_fields=exc.missing_fields,
        )
        validate_artifact(artifact)
        _write_json(result_path, artifact)
        return artifact

    active_counts = list(sample_counts if sample_counts is not None else basis.sample_count_sweep)
    cpu_results = [
        run_cpu_gibbs_for_problem(problem, sample_count=count)
        for problem in basis.problems
        for count in active_counts
    ]
    artifact = build_success_artifact(
        basis,
        cpu_results,
        sample_counts=active_counts,
        duration_s=time.perf_counter() - started,
    )
    validate_artifact(artifact)
    _write_json(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--print-result-path", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(root_path=args.root)
    if args.print_result_path:
        print(args.root / RESULT_ARTIFACT)
    else:
        print(json.dumps({"honest_verdict": artifact["honest_verdict"], "result": str(args.root / RESULT_ARTIFACT)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
