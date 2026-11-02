"""PolarFire SoC CPU dispatch smoke for a deterministic Carnot constraint scorer.

Spec refs: REQ-HW-062, SCENARIO-HW-062.

This module intentionally targets the Linux CPU harts on the PolarFire SoC
Discovery Kit, not the FPGA fabric. The point of the smoke is to prove that a
Carnot-style constraint workload was actually dispatched to the riscv64 board:
we ship a deterministic SAT instance and scorer, pull back the transcript, and
only accept the run when the remote deterministic scorer-output hash matches
the output hash computed locally from the same instance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


EXPERIMENT_ID = 2900
SCHEMA = "carnot.polarfire_carnot_dispatch_smoke.v1"
SPEC_REFS = ["REQ-HW-062", "SCENARIO-HW-062"]
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = REPO_ROOT / "results" / "experiment_2900_polarfire_carnot_dispatch_smoke_v1.json"
TRANSCRIPT_PATH = REPO_ROOT / "results" / "experiment_2900_polarfire_transcript_v1.json"
DEFAULT_HOST = "polarfire"
DEFAULT_REMOTE_RUNTIME_S = 10.25
MIN_ACCEPTED_DURATION_S = 10.0


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class HarnessFiles:
    instance_path: Path
    scorer_path: Path


class Runner(Protocol):
    def __call__(
        self, args: Sequence[str], timeout: float | None = None
    ) -> CommandResult: ...


class DispatchError(RuntimeError):
    def __init__(self, verdict: str, detail: str) -> None:
        super().__init__(detail)
        self.verdict = verdict
        self.detail = detail


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def generate_sat_instance() -> dict[str, Any]:
    """Build the deterministic 50-clause SAT payload dispatched to PolarFire.

    The clause generator is deliberately simple instead of random-at-runtime:
    every run ships the same compact constraint instance, so the expected scorer
    hash is reproducible on the host and the board. A clause is represented as
    three signed one-indexed literals, using the common DIMACS convention where
    negative values mean logical negation.
    """
    num_variables = 16
    assignment = [idx % 5 in {0, 2, 3} for idx in range(num_variables)]
    clauses: list[list[int]] = []
    for idx in range(50):
        clause: list[int] = []
        for offset in range(3):
            variable = ((idx * 5) + (offset * 7)) % num_variables + 1
            literal = -variable if ((idx + offset * 3) % 4 == 0) else variable
            clause.append(literal)
        clauses.append(clause)
    return {
        "schema": "carnot.sat_instance.v1",
        "spec_refs": SPEC_REFS,
        "num_variables": num_variables,
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
        "schema": "carnot.sat_scorer_output.v1",
        "spec_refs": SPEC_REFS,
        "instance_sha256": sha256_json(instance),
        "num_clauses": len(clauses),
        "assignment_bits": "".join("1" if value else "0" for value in assignment),
        "per_clause_satisfied": per_clause_satisfied,
        "satisfied_clause_count": sum(1 for value in per_clause_satisfied if value),
        "violated_clause_indices": violated_clause_indices,
        "unsatisfied_clause_count": len(violated_clause_indices),
    }


def parse_python_version(version_text: str) -> tuple[int, int, int] | None:
    match = re.search(r"Python\s+(\d+)\.(\d+)(?:\.(\d+))?", version_text)
    if not match:
        return None
    patch = int(match.group(3) or "0")
    return int(match.group(1)), int(match.group(2)), patch


def _command_string(args: Sequence[str]) -> str:
    return shlex.join([str(arg) for arg in args])


def _observed(result: CommandResult) -> str:
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        return stdout
    if stderr:
        return stderr
    return f"returncode={result.returncode}"


def _check_row(
    resource: str, args: Sequence[str], result: CommandResult, passed: bool, observed: str | None = None
) -> dict[str, Any]:
    return {
        "resource": resource,
        "command": _command_string(args),
        "passed": passed,
        "observed": observed if observed is not None else _observed(result),
    }


def check_preconditions(
    runner: Runner, host: str = DEFAULT_HOST
) -> tuple[list[dict[str, Any]], dict[str, str], str | None]:
    checks: list[dict[str, Any]] = []
    metadata: dict[str, str] = {}

    ssh_cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, "true"]
    ssh_result = runner(ssh_cmd, timeout=10)
    ssh_ok = ssh_result.returncode == 0
    checks.append(_check_row("polarfire_ssh", ssh_cmd, ssh_result, ssh_ok))
    if not ssh_ok:
        return checks, metadata, "blocked_polarfire_ssh_unreachable"

    arch_cmd = ["ssh", host, "uname -m"]
    arch_result = runner(arch_cmd, timeout=10)
    arch = arch_result.stdout.strip()
    metadata["polarfire_arch"] = arch
    arch_ok = arch_result.returncode == 0 and arch == "riscv64"
    checks.append(_check_row("polarfire_arch", arch_cmd, arch_result, arch_ok, arch))
    if not arch_ok:
        return checks, metadata, "blocked_polarfire_wrong_architecture"

    python_cmd = ["ssh", host, "python3 --version"]
    python_result = runner(python_cmd, timeout=10)
    python_text = (python_result.stdout or python_result.stderr).strip()
    metadata["polarfire_python"] = python_text
    python_version = parse_python_version(python_text)
    python_ok = (
        python_result.returncode == 0
        and python_version is not None
        and python_version >= (3, 10, 0)
    )
    checks.append(_check_row("polarfire_python", python_cmd, python_result, python_ok, python_text))
    if not python_ok:
        return checks, metadata, "blocked_polarfire_python_missing"

    uptime_cmd = ["ssh", host, "uptime"]
    uptime_result = runner(uptime_cmd, timeout=10)
    metadata["uptime"] = uptime_result.stdout.strip()
    checks.append(_check_row("polarfire_uptime", uptime_cmd, uptime_result, uptime_result.returncode == 0))

    kernel_cmd = ["ssh", host, "uname -r"]
    kernel_result = runner(kernel_cmd, timeout=10)
    metadata["polarfire_kernel"] = kernel_result.stdout.strip()
    checks.append(_check_row("polarfire_kernel", kernel_cmd, kernel_result, kernel_result.returncode == 0))

    return checks, metadata, None


def build_remote_scorer_source() -> str:
    """Return the self-contained scorer source copied to the riscv64 board."""
    return r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import platform
import sys
import time

SPEC_REFS = ["REQ-HW-062", "SCENARIO-HW-062"]


def canonical_json_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload):
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def literal_satisfied(literal, assignment):
    value = bool(assignment[abs(int(literal)) - 1])
    return (not value) if int(literal) < 0 else value


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
        "schema": "carnot.sat_scorer_output.v1",
        "spec_refs": SPEC_REFS,
        "instance_sha256": sha256_json(instance),
        "num_clauses": len(clauses),
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
    parser.add_argument("--min-runtime-s", type=float, default=10.25)
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
        "schema": "carnot.polarfire_sat_scorer_transcript.v1",
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
    instance_path.write_text(json.dumps(instance, sort_keys=True, indent=2))
    scorer_path.write_text(build_remote_scorer_source())
    return HarnessFiles(instance_path=instance_path, scorer_path=scorer_path)


def _require_success(result: CommandResult, stage: str) -> None:
    if result.returncode != 0:
        detail = f"{stage}: rc={result.returncode}; stderr={result.stderr[:300]}"
        raise DispatchError("blocked_polarfire_dispatch_failed", detail)


def dispatch_to_board(
    runner: Runner,
    files: HarnessFiles,
    local_transcript_path: Path,
    *,
    host: str,
    remote_dir: str,
    min_remote_runtime_s: float,
) -> dict[str, Any]:
    mkdir_cmd = ["ssh", host, f"mkdir -p {shlex.quote(remote_dir)}"]
    cleanup_cmd = ["ssh", host, f"rm -rf {shlex.quote(remote_dir)}"]
    try:
        mkdir_result = runner(mkdir_cmd, timeout=20)
        _require_success(mkdir_result, "remote_mkdir")

        scp_push_cmd = [
            "scp",
            "-q",
            str(files.instance_path),
            str(files.scorer_path),
            f"{host}:{remote_dir}/",
        ]
        scp_push_result = runner(scp_push_cmd, timeout=60)
        _require_success(scp_push_result, "scp_push")

        run_cmd = [
            "ssh",
            host,
            (
                f"cd {shlex.quote(remote_dir)} && "
                "python3 scorer.py sat_instance.json transcript.json "
                f"--min-runtime-s {min_remote_runtime_s:.3f}"
            ),
        ]
        run_result = runner(run_cmd, timeout=max(90, int(min_remote_runtime_s) + 30))
        _require_success(run_result, "remote_scorer")

        scp_pull_cmd = [
            "scp",
            "-q",
            f"{host}:{remote_dir}/transcript.json",
            str(local_transcript_path),
        ]
        scp_pull_result = runner(scp_pull_cmd, timeout=60)
        _require_success(scp_pull_result, "scp_pull")
        return json.loads(local_transcript_path.read_text())
    finally:
        runner(cleanup_cmd, timeout=20)


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
        "inference_substrate": "hardware_smoke",
        "preconditions_checked": list(checks),
        "polarfire_ssh_uptime_at_run": metadata.get("uptime", ""),
        "polarfire_kernel": metadata.get("polarfire_kernel", ""),
        "polarfire_arch": metadata.get("polarfire_arch", ""),
        "sat_instance_sha256": sha256_json(instance),
        "scorer_output_sha256": "",
        "scorer_output_hash_verified": False,
        "per_clause_wall_clock_us": [],
        "total_wall_clock_s": 0.0,
        "duration_s": round(float(duration_s), 6),
        "no_fpga_fabric_claim": True,
    }


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
    return artifact


def compose_terminal_artifact(
    *,
    checks: Sequence[dict[str, Any]],
    metadata: dict[str, str],
    instance: dict[str, Any],
    transcript: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    expected_output = score_sat_instance(instance)
    expected_hash = sha256_json(expected_output)
    remote_hash = str(transcript.get("scorer_output_sha256", ""))
    hash_verified = remote_hash == expected_hash

    if not hash_verified:
        verdict = "failed_polarfire_scorer_output_hash_mismatch"
    elif duration_s < MIN_ACCEPTED_DURATION_S:
        verdict = "failed_polarfire_duration_gate"
    else:
        verdict = "complete: polarfire_riscv64_constraint_scorer_hash_verified"

    artifact = _base_artifact(verdict, checks, metadata, instance, duration_s)
    artifact.update(
        {
            "scorer_output_sha256": remote_hash,
            "expected_scorer_output_sha256": expected_hash,
            "scorer_output_hash_verified": hash_verified,
            "per_clause_wall_clock_us": [
                float(value) for value in transcript.get("per_clause_wall_clock_us", [])
            ],
            "total_wall_clock_s": float(transcript.get("total_wall_clock_s", 0.0)),
            "remote_transcript_schema": transcript.get("schema", ""),
            "remote_evaluation_cycles_per_clause": transcript.get(
                "evaluation_cycles_per_clause", 0
            ),
            "remote_arch": transcript.get("remote_arch", ""),
            "remote_python": transcript.get("remote_python", ""),
        }
    )
    return artifact


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def run_experiment(
    *,
    output_path: Path = RESULT_PATH,
    transcript_output_path: Path = TRANSCRIPT_PATH,
    runner: Runner,
    host: str = DEFAULT_HOST,
    clock: Any = time.monotonic,
    remote_dir: str | None = None,
    min_remote_runtime_s: float = DEFAULT_REMOTE_RUNTIME_S,
) -> dict[str, Any]:
    started = clock()
    instance = generate_sat_instance()
    checks, metadata, blocker = check_preconditions(runner, host)
    if blocker is not None:
        artifact = blocked_artifact(blocker, checks, metadata, instance, clock() - started)
        _write_json(output_path, artifact)
        return artifact

    active_remote_dir = remote_dir or f"/tmp/carnot_exp2900_{os.getpid()}_{int(time.time())}"
    try:
        with tempfile.TemporaryDirectory(prefix="carnot_exp2900_") as tmpdir:
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
            duration_s=clock() - started,
        )
    except DispatchError as exc:
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
    try:  # pragma: no cover - exercised by the live experiment, not unit tests.
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
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--transcript-output", type=Path, default=TRANSCRIPT_PATH)
    parser.add_argument("--remote-dir", default=None)
    parser.add_argument("--min-remote-runtime-s", type=float, default=DEFAULT_REMOTE_RUNTIME_S)
    args = parser.parse_args(argv)

    artifact = run_experiment(
        output_path=args.output,
        transcript_output_path=args.transcript_output,
        runner=_run_local_command,
        host=args.host,
        remote_dir=args.remote_dir,
        min_remote_runtime_s=args.min_remote_runtime_s,
    )
    print(json.dumps(artifact, sort_keys=True, indent=2))
    return 0 if artifact["honest_verdict"].startswith("complete:") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
