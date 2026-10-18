"""Exp 2941 PolarFire 500-clause SAT dispatch continuation.

Spec refs: REQ-HW-073, SCENARIO-HW-073.

This module keeps the Exp 2900 hardware claim deliberately narrow: it dispatches
a deterministic SAT scorer to the Linux CPU cores on the PolarFire SoC over SSH,
then accepts the run only when the scorer output hash pulled back from the board
matches the scorer output recomputed locally from the same 500-clause instance.
The timing comparison is also narrow. It compares per-clause wall-clock time
against Exp 2900's 50-clause median so a larger SAT payload cannot quietly turn
into a fabricated or unbounded "hardware acceleration" claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from carnot.hardware import polarfire_dispatch_smoke as dispatch


EXPERIMENT_ID = 2941
SCHEMA = "carnot.polarfire_continuation.v1"
TRANSCRIPT_SCHEMA = "carnot.polarfire_sat_scorer_transcript.v2"
SPEC_REFS = ["REQ-HW-073", "SCENARIO-HW-073"]
REPO_ROOT = Path(__file__).resolve().parents[3]
EXP2900_REL_PATH = Path("results/experiment_2900_polarfire_carnot_dispatch_smoke_v1.json")
OUTPUT_REL_PATH = Path("results/experiment_2941_polarfire_continuation_v1.json")
TRANSCRIPT_REL_PATH = Path("results/experiment_2941_polarfire_transcript_v1.json")
DEFAULT_HOST = "polarfire"
DEFAULT_REMOTE_RUNTIME_S = 15.25
MIN_ACCEPTED_DURATION_S = 15.0
RANDOM_SEED = 2941
N_CLAUSES = 500
NUM_VARIABLES = 32
INFERENCE_SUBSTRATE = "hardware_smoke"

CommandResult = dispatch.CommandResult
HarnessFiles = dispatch.HarnessFiles


class Runner(Protocol):
    def __call__(
        self, args: Sequence[str], timeout: float | None = None
    ) -> CommandResult: ...


@dataclass(frozen=True)
class ContinuationError(RuntimeError):
    verdict: str
    detail: str

    def __str__(self) -> str:
        return self.detail


REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "polarfire_ssh_uptime_at_run",
    "n_clauses",
    "per_clause_wall_clock_us_median",
    "per_clause_wall_clock_us_p95",
    "scaling_ratio_vs_exp2900",
    "scorer_output_sha256",
    "scorer_output_hash_verified",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
}


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def generate_sat_instance(
    *, n_clauses: int = N_CLAUSES, random_seed: int = RANDOM_SEED
) -> dict[str, Any]:
    """Build the deterministic 500-clause SAT payload for the PolarFire board.

    The generator uses a fixed seed and stores that seed in the payload. That
    makes the instance reproducible from the JSON alone, while still avoiding a
    hand-written clause table that could accidentally be tuned to one scorer
    output. Literals use the DIMACS convention: positive integers are variables
    and negative integers are logical negations.
    """

    rng = random.Random(random_seed)
    assignment = [rng.randrange(2) == 1 for _ in range(NUM_VARIABLES)]
    clauses: list[list[int]] = []
    for _ in range(n_clauses):
        variables = rng.sample(range(1, NUM_VARIABLES + 1), 3)
        clauses.append([variable if rng.randrange(2) else -variable for variable in variables])
    return {
        "schema": "carnot.sat_instance.v2",
        "spec_refs": SPEC_REFS,
        "random_seed": random_seed,
        "num_variables": NUM_VARIABLES,
        "clauses": clauses,
        "assignment": assignment,
    }


def _literal_satisfied(literal: int, assignment: Sequence[bool]) -> bool:
    value = bool(assignment[abs(literal) - 1])
    return not value if literal < 0 else value


def score_sat_instance(instance: dict[str, Any]) -> dict[str, Any]:
    clauses = instance["clauses"]
    assignment = instance["assignment"]
    per_clause_satisfied = [
        any(_literal_satisfied(int(literal), assignment) for literal in clause)
        for clause in clauses
    ]
    violated_clause_indices = [
        idx for idx, satisfied in enumerate(per_clause_satisfied) if not satisfied
    ]
    return {
        "schema": "carnot.sat_scorer_output.v2",
        "spec_refs": SPEC_REFS,
        "instance_sha256": sha256_json(instance),
        "num_clauses": len(clauses),
        "random_seed": int(instance["random_seed"]),
        "assignment_bits": "".join("1" if value else "0" for value in assignment),
        "per_clause_satisfied": per_clause_satisfied,
        "satisfied_clause_count": sum(1 for value in per_clause_satisfied if value),
        "violated_clause_indices": violated_clause_indices,
        "unsatisfied_clause_count": len(violated_clause_indices),
    }


def percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def build_remote_scorer_source() -> str:
    """Return the self-contained scorer copied to the riscv64 board."""

    return r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import platform
import sys
import time

SPEC_REFS = ["REQ-HW-073", "SCENARIO-HW-073"]
TRANSCRIPT_SCHEMA = "carnot.polarfire_sat_scorer_transcript.v2"


def canonical_json_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload):
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def literal_satisfied(literal, assignment):
    literal = int(literal)
    value = bool(assignment[abs(literal) - 1])
    return (not value) if literal < 0 else value


def evaluate_clause(clause, assignment):
    return any(literal_satisfied(literal, assignment) for literal in clause)


def score_sat_instance(instance):
    clauses = instance["clauses"]
    assignment = instance["assignment"]
    per_clause_satisfied = [evaluate_clause(clause, assignment) for clause in clauses]
    violated_clause_indices = [
        idx for idx, satisfied in enumerate(per_clause_satisfied) if not satisfied
    ]
    return {
        "schema": "carnot.sat_scorer_output.v2",
        "spec_refs": SPEC_REFS,
        "instance_sha256": sha256_json(instance),
        "num_clauses": len(clauses),
        "random_seed": int(instance["random_seed"]),
        "assignment_bits": "".join("1" if value else "0" for value in assignment),
        "per_clause_satisfied": per_clause_satisfied,
        "satisfied_clause_count": sum(1 for value in per_clause_satisfied if value),
        "violated_clause_indices": violated_clause_indices,
        "unsatisfied_clause_count": len(violated_clause_indices),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_path")
    parser.add_argument("transcript_path")
    parser.add_argument("--min-runtime-s", type=float, default=15.25)
    parser.add_argument("--min-cycles", type=int, default=1)
    args = parser.parse_args()

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)

    clauses = instance["clauses"]
    assignment = instance["assignment"]
    scorer_output = score_sat_instance(instance)
    clause_elapsed_ns = [0] * len(clauses)
    cycles = 0

    started = time.perf_counter()
    while True:
        for idx, clause in enumerate(clauses):
            clause_started = time.perf_counter_ns()
            evaluate_clause(clause, assignment)
            clause_elapsed_ns[idx] += time.perf_counter_ns() - clause_started
        cycles += 1
        total_wall_clock_s = time.perf_counter() - started
        if total_wall_clock_s >= args.min_runtime_s and cycles >= args.min_cycles:
            break

    transcript = {
        "schema": TRANSCRIPT_SCHEMA,
        "spec_refs": SPEC_REFS,
        "remote_arch": platform.machine(),
        "remote_python": sys.version.split()[0],
        "sat_instance_sha256": sha256_json(instance),
        "scorer_output": scorer_output,
        "scorer_output_sha256": sha256_json(scorer_output),
        "per_clause_wall_clock_us": [
            elapsed_ns / cycles / 1000.0 for elapsed_ns in clause_elapsed_ns
        ],
        "total_wall_clock_s": total_wall_clock_s,
        "evaluation_cycles_per_clause": cycles,
    }

    with open(args.transcript_path, "w", encoding="utf-8") as handle:
        json.dump(transcript, handle, sort_keys=True, indent=2)


if __name__ == "__main__":
    main()
'''


def write_harness_files(local_dir: Path, instance: dict[str, Any]) -> HarnessFiles:
    local_dir.mkdir(parents=True, exist_ok=True)
    instance_path = local_dir / "sat_instance.json"
    scorer_path = local_dir / "scorer.py"
    instance_path.write_text(json.dumps(instance, sort_keys=True, indent=2), encoding="utf-8")
    scorer_path.write_text(build_remote_scorer_source(), encoding="utf-8")
    return HarnessFiles(instance_path=instance_path, scorer_path=scorer_path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContinuationError("blocked_exp2900_artifact_missing", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise ContinuationError("blocked_exp2900_artifact_invalid_json", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ContinuationError("blocked_exp2900_artifact_invalid_json", "expected object")
    return payload


def load_exp2900_median(root_path: Path = REPO_ROOT) -> float:
    """Load the 50-clause median timing used as the scaling baseline."""

    payload = _read_json(root_path / EXP2900_REL_PATH)
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ContinuationError("blocked_exp2900_not_hardware_smoke", "wrong substrate")
    if payload.get("scorer_output_hash_verified") is not True:
        raise ContinuationError("blocked_exp2900_hash_not_verified", "hash gate failed")
    timings = payload.get("per_clause_wall_clock_us")
    if not isinstance(timings, list) or not timings:
        raise ContinuationError("blocked_exp2900_timing_missing", "missing per-clause timings")
    return float(statistics.median(float(value) for value in timings))


def dispatch_to_board(
    runner: Runner,
    files: HarnessFiles,
    local_transcript_path: Path,
    *,
    host: str,
    remote_dir: str,
    min_remote_runtime_s: float,
) -> dict[str, Any]:
    return dispatch.dispatch_to_board(
        runner,
        files,
        local_transcript_path,
        host=host,
        remote_dir=remote_dir,
        min_remote_runtime_s=min_remote_runtime_s,
    )


def _base_artifact(
    verdict: str,
    checks: Sequence[dict[str, Any]],
    metadata: dict[str, str],
    instance: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": list(checks),
        "polarfire_ssh_uptime_at_run": metadata.get("uptime", ""),
        "polarfire_kernel": metadata.get("polarfire_kernel", ""),
        "polarfire_arch": metadata.get("polarfire_arch", ""),
        "n_clauses": len(instance.get("clauses", [])),
        "sat_instance_sha256": sha256_json(instance),
        "per_clause_wall_clock_us_median": 0.0,
        "per_clause_wall_clock_us_p95": 0.0,
        "exp2900_per_clause_wall_clock_us_median": 0.0,
        "scaling_ratio_vs_exp2900": 0.0,
        "scorer_output_sha256": "",
        "expected_scorer_output_sha256": "",
        "scorer_output_hash_verified": False,
        "random_seed": int(instance.get("random_seed", RANDOM_SEED)),
        "reproducibility_checksum": "",
        "total_wall_clock_s": 0.0,
        "duration_s": round(float(duration_s), 6),
        "no_fpga_fabric_claim": True,
    }


def _reproducibility_checksum(
    *,
    instance: dict[str, Any],
    expected_hash: str,
    remote_hash: str,
    exp2900_median_us: float,
    median_us: float,
    p95_us: float,
) -> str:
    return sha256_json(
        {
            "sat_instance_sha256": sha256_json(instance),
            "expected_scorer_output_sha256": expected_hash,
            "scorer_output_sha256": remote_hash,
            "random_seed": int(instance["random_seed"]),
            "n_clauses": len(instance["clauses"]),
            "exp2900_per_clause_wall_clock_us_median": float(exp2900_median_us),
            "per_clause_wall_clock_us_median": float(median_us),
            "per_clause_wall_clock_us_p95": float(p95_us),
        }
    )


def blocked_artifact(
    verdict: str,
    checks: Sequence[dict[str, Any]],
    metadata: dict[str, str],
    instance: dict[str, Any],
    duration_s: float,
    detail: str = "",
) -> dict[str, Any]:
    artifact = _base_artifact(verdict, checks, metadata, instance, duration_s)
    artifact["failure_detail"] = detail
    artifact["reproducibility_checksum"] = sha256_json(
        {
            "blocked_verdict": verdict,
            "sat_instance_sha256": artifact["sat_instance_sha256"],
            "random_seed": artifact["random_seed"],
            "n_clauses": artifact["n_clauses"],
        }
    )
    validate_artifact(artifact, allow_blocked=True)
    return artifact


def compose_terminal_artifact(
    *,
    checks: Sequence[dict[str, Any]],
    metadata: dict[str, str],
    instance: dict[str, Any],
    transcript: dict[str, Any],
    exp2900_median_us: float,
    duration_s: float,
) -> dict[str, Any]:
    expected_output = score_sat_instance(instance)
    expected_hash = sha256_json(expected_output)
    remote_hash = str(transcript.get("scorer_output_sha256", ""))
    hash_verified = remote_hash == expected_hash
    per_clause_us = [
        float(value) for value in transcript.get("per_clause_wall_clock_us", [])
    ]
    median_us = float(statistics.median(per_clause_us)) if per_clause_us else 0.0
    p95_us = percentile_nearest_rank(per_clause_us, 95.0)
    scaling_ratio = median_us / exp2900_median_us if exp2900_median_us > 0 else 0.0

    if not hash_verified:
        verdict = "failed_polarfire_scorer_output_hash_mismatch"
    elif duration_s < MIN_ACCEPTED_DURATION_S:
        verdict = "failed_polarfire_duration_gate"
    else:
        verdict = "complete: polarfire_500_clause_constraint_scorer_hash_verified"

    artifact = _base_artifact(verdict, checks, metadata, instance, duration_s)
    artifact.update(
        {
            "per_clause_wall_clock_us_median": median_us,
            "per_clause_wall_clock_us_p95": p95_us,
            "exp2900_per_clause_wall_clock_us_median": float(exp2900_median_us),
            "scaling_ratio_vs_exp2900": scaling_ratio,
            "scorer_output_sha256": remote_hash,
            "expected_scorer_output_sha256": expected_hash,
            "scorer_output_hash_verified": hash_verified,
            "reproducibility_checksum": _reproducibility_checksum(
                instance=instance,
                expected_hash=expected_hash,
                remote_hash=remote_hash,
                exp2900_median_us=exp2900_median_us,
                median_us=median_us,
                p95_us=p95_us,
            ),
            "total_wall_clock_s": float(transcript.get("total_wall_clock_s", 0.0)),
            "remote_transcript_schema": transcript.get("schema", ""),
            "remote_evaluation_cycles_per_clause": transcript.get(
                "evaluation_cycles_per_clause", 0
            ),
            "remote_arch": transcript.get("remote_arch", ""),
            "remote_python": transcript.get("remote_python", ""),
        }
    )
    validate_artifact(artifact, allow_blocked=False)
    return artifact


def validate_artifact(artifact: dict[str, Any], *, allow_blocked: bool = False) -> None:
    missing = REQUIRED_ARTIFACT_FIELDS - artifact.keys()
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be hardware_smoke")
    if artifact["n_clauses"] != N_CLAUSES:
        raise ValueError("n_clauses must be 500")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed must match the deterministic instance")
    if not isinstance(artifact["reproducibility_checksum"], str):
        raise ValueError("reproducibility_checksum must be a string")
    if not allow_blocked and artifact["honest_verdict"].startswith("complete:"):
        if artifact["scorer_output_hash_verified"] is not True:
            raise ValueError("complete artifacts require hash verification")
        if float(artifact["duration_s"]) < MIN_ACCEPTED_DURATION_S:
            raise ValueError("complete artifacts require duration_s >= 15")
        if float(artifact["per_clause_wall_clock_us_median"]) <= 0:
            raise ValueError("complete artifacts require positive median timing")
        if float(artifact["per_clause_wall_clock_us_p95"]) <= 0:
            raise ValueError("complete artifacts require positive p95 timing")
        if float(artifact["scaling_ratio_vs_exp2900"]) <= 0:
            raise ValueError("complete artifacts require a positive scaling ratio")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def run_experiment(
    *,
    root_path: Path = REPO_ROOT,
    runner: Runner,
    host: str = DEFAULT_HOST,
    clock: Any = time.monotonic,
    remote_dir: str | None = None,
    min_remote_runtime_s: float = DEFAULT_REMOTE_RUNTIME_S,
) -> dict[str, Any]:
    started = clock()
    instance = generate_sat_instance()
    checks, metadata, blocker = dispatch.check_preconditions(runner, host)
    output_path = root_path / OUTPUT_REL_PATH
    transcript_output_path = root_path / TRANSCRIPT_REL_PATH

    if blocker is not None:
        artifact = blocked_artifact(blocker, checks, metadata, instance, clock() - started)
        _write_json(output_path, artifact)
        return artifact

    try:
        exp2900_median_us = load_exp2900_median(root_path)
    except ContinuationError as exc:
        artifact = blocked_artifact(
            exc.verdict,
            checks,
            metadata,
            instance,
            clock() - started,
            detail=exc.detail,
        )
        _write_json(output_path, artifact)
        return artifact

    active_remote_dir = remote_dir or f"/tmp/carnot_exp2941_{os.getpid()}_{int(time.time())}"
    try:
        with tempfile.TemporaryDirectory(prefix="carnot_exp2941_") as tmpdir:
            tmp_path = Path(tmpdir)
            files = write_harness_files(tmp_path, instance)
            local_transcript_path = tmp_path / "transcript.json"
            transcript = dispatch_to_board(
                runner,
                files,
                local_transcript_path,
                host=host,
                remote_dir=active_remote_dir,
                min_remote_runtime_s=min_remote_runtime_s,
            )
            _write_json(transcript_output_path, transcript)
        artifact = compose_terminal_artifact(
            checks=checks,
            metadata=metadata,
            instance=instance,
            transcript=transcript,
            exp2900_median_us=exp2900_median_us,
            duration_s=clock() - started,
        )
    except dispatch.DispatchError as exc:
        artifact = blocked_artifact(
            exc.verdict,
            checks,
            metadata,
            instance,
            clock() - started,
            detail=exc.detail,
        )

    _write_json(output_path, artifact)
    return artifact


def _run_local_command(args: Sequence[str], timeout: float | None = None) -> CommandResult:
    try:  # pragma: no cover - exercised by the live experiment path.
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return CommandResult(tuple(args), proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as exc:  # pragma: no cover
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
        return CommandResult(tuple(args), 124, stdout, stderr)
    except FileNotFoundError as exc:  # pragma: no cover
        return CommandResult(tuple(args), 127, "", str(exc))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--remote-dir", default=None)
    parser.add_argument("--min-remote-runtime-s", type=float, default=DEFAULT_REMOTE_RUNTIME_S)
    args = parser.parse_args(argv)

    artifact = run_experiment(
        root_path=args.root,
        runner=_run_local_command,
        host=args.host,
        remote_dir=args.remote_dir,
        min_remote_runtime_s=args.min_remote_runtime_s,
    )
    print(json.dumps(artifact, sort_keys=True, indent=2))
    return 0 if artifact["honest_verdict"].startswith("complete:") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
