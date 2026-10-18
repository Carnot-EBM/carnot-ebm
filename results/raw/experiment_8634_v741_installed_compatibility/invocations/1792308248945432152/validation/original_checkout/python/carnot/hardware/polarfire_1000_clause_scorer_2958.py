"""Exp 2958 PolarFire 1000-clause SAT scorer continuation.

Spec refs: REQ-HW-076, SCENARIO-HW-076.

The purpose of this module is deliberately narrow: it extends the already
hash-verified Exp 2941 PolarFire scorer path from 500 clauses to 1000 clauses.
It does not report speedup, acceleration, or FPGA-fabric execution. The only
success condition is that a deterministic 1000-clause input sent to the
PolarFire Linux CPU substrate returns a transcript whose scorer-output hash
matches the locally recomputed hash for the same input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

from carnot.hardware import polarfire_continuation_2941 as baseline
from carnot.hardware import polarfire_dispatch_smoke as dispatch


EXPERIMENT_ID = 2958
SCHEMA = "carnot.polarfire_1000_clause_scorer.v2"
TRANSCRIPT_SCHEMA = baseline.TRANSCRIPT_SCHEMA
SPEC_REFS = ["REQ-HW-076", "SCENARIO-HW-076"]
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_500_CLAUSE_REL_PATH = baseline.OUTPUT_REL_PATH
BASELINE_500_TRANSCRIPT_REL_PATH = baseline.TRANSCRIPT_REL_PATH
OUTPUT_REL_PATH = Path("results/experiment_2958_polarfire_1000_clause_scorer_v2.json")
TRANSCRIPT_REL_PATH = Path("results/experiment_2958_polarfire_1000_clause_transcript_v2.json")
DEFAULT_HOST = baseline.DEFAULT_HOST
DEFAULT_REMOTE_RUNTIME_S = 1.0
RANDOM_SEED = 2958
CLAUSE_COUNT = 1000
NUM_VARIABLES = baseline.NUM_VARIABLES
INFERENCE_SUBSTRATE = "hardware_smoke"

CommandResult = dispatch.CommandResult
HarnessFiles = dispatch.HarnessFiles

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "preconditions_checked",
    "polarfire_1000_clause_hash_verified",
    "baseline_500_clause_artifact",
    "clause_count",
    "input_sha256",
    "scorer_output_sha256",
    "transcript_paths",
    "elapsed_ms",
    "board_reachable",
    "no_speedup_claim",
    "no_general_acceleration_claim",
    "inference_substrate",
    "duration_s",
}


class Runner(Protocol):
    def __call__(
        self, args: Sequence[str], timeout: float | None = None
    ) -> CommandResult: ...


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def generate_sat_instance() -> dict[str, Any]:
    """Build the deterministic 1000-clause payload for the PolarFire scorer.

    The instance generator intentionally reuses the Exp 2941 clause generator
    and changes only the experiment seed, clause count, and spec refs. That
    keeps the continuation tied to the proven 500-clause path while preventing
    the 1000-clause hash from being a duplicated baseline artifact.
    """

    instance = baseline.generate_sat_instance(n_clauses=CLAUSE_COUNT, random_seed=RANDOM_SEED)
    instance["spec_refs"] = SPEC_REFS
    return instance


def score_sat_instance(instance: dict[str, Any]) -> dict[str, Any]:
    scorer_output = baseline.score_sat_instance(instance)
    scorer_output["spec_refs"] = SPEC_REFS
    return scorer_output


def build_remote_scorer_source() -> str:
    """Return the self-contained scorer copied to the PolarFire riscv64 board."""

    return baseline.build_remote_scorer_source().replace(
        'SPEC_REFS = ["REQ-HW-073", "SCENARIO-HW-073"]',
        'SPEC_REFS = ["REQ-HW-076", "SCENARIO-HW-076"]',
    )


def write_harness_files(local_dir: Path, instance: dict[str, Any]) -> HarnessFiles:
    local_dir.mkdir(parents=True, exist_ok=True)
    instance_path = local_dir / "sat_instance.json"
    scorer_path = local_dir / "scorer.py"
    instance_path.write_text(json.dumps(instance, sort_keys=True, indent=2), encoding="utf-8")
    scorer_path.write_text(build_remote_scorer_source(), encoding="utf-8")
    return HarnessFiles(instance_path=instance_path, scorer_path=scorer_path)


def _check_row(resource: str, path: Path, passed: bool, observed: str) -> dict[str, Any]:
    return {
        "resource": resource,
        "path": path.as_posix(),
        "passed": passed,
        "observed": observed,
    }


def _load_json_object(path: Path, missing_verdict: str, invalid_verdict: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise baseline.ContinuationError(missing_verdict, path.as_posix()) from exc
    except json.JSONDecodeError as exc:
        raise baseline.ContinuationError(invalid_verdict, str(exc)) from exc
    if not isinstance(payload, dict):
        raise baseline.ContinuationError(invalid_verdict, "expected object")
    return payload


def check_baseline_context(root_path: Path = REPO_ROOT) -> tuple[list[dict[str, Any]], str | None]:
    """Verify the Exp 2941 baseline artifact exists before extending it.

    A missing or unverified baseline is not a hardware failure; it means this
    task lacks the prior hash context needed to make the 1000-clause result
    interpretable. The caller records that as a blocked artifact instead of
    attempting a board run without provenance.
    """

    artifact_path = root_path / BASELINE_500_CLAUSE_REL_PATH
    transcript_path = root_path / BASELINE_500_TRANSCRIPT_REL_PATH
    checks: list[dict[str, Any]] = []

    try:
        artifact = _load_json_object(
            artifact_path,
            "blocked_baseline_500_clause_artifact_missing",
            "blocked_baseline_500_clause_artifact_invalid_json",
        )
    except baseline.ContinuationError as exc:
        checks.append(_check_row("baseline_500_clause_artifact", artifact_path, False, exc.detail))
        return checks, exc.verdict

    baseline_ok = (
        artifact.get("scorer_output_hash_verified") is True
        and artifact.get("n_clauses") == 500
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
    )
    checks.append(
        _check_row(
            "baseline_500_clause_artifact",
            artifact_path,
            baseline_ok,
            str(artifact.get("scorer_output_sha256", "")),
        )
    )
    if not baseline_ok:
        return checks, "blocked_baseline_500_clause_hash_unverified"

    try:
        _load_json_object(
            transcript_path,
            "blocked_baseline_500_clause_transcript_missing",
            "blocked_baseline_500_clause_transcript_invalid_json",
        )
    except baseline.ContinuationError as exc:
        checks.append(
            _check_row("baseline_500_clause_transcript", transcript_path, False, exc.detail)
        )
        return checks, exc.verdict

    checks.append(_check_row("baseline_500_clause_transcript", transcript_path, True, "present"))
    return checks, None


def _base_artifact(
    verdict: str,
    checks: Sequence[dict[str, Any]],
    instance: dict[str, Any],
    duration_s: float,
    *,
    board_reachable: bool,
    verified: bool,
    scorer_output_sha256: str,
    transcript_paths: Sequence[str],
    elapsed_ms: float,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "honest_verdict": verdict,
        "preconditions_checked": list(checks),
        "polarfire_1000_clause_hash_verified": bool(verified),
        "baseline_500_clause_artifact": BASELINE_500_CLAUSE_REL_PATH.as_posix(),
        "clause_count": len(instance.get("clauses", [])),
        "input_sha256": sha256_json(instance),
        "scorer_output_sha256": scorer_output_sha256,
        "transcript_paths": list(transcript_paths),
        "elapsed_ms": float(elapsed_ms),
        "board_reachable": bool(board_reachable),
        "no_speedup_claim": True,
        "no_general_acceleration_claim": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
    }


def blocked_artifact(
    verdict: str,
    checks: Sequence[dict[str, Any]],
    instance: dict[str, Any],
    duration_s: float,
    *,
    board_reachable: bool = False,
    detail: str = "",
) -> dict[str, Any]:
    transcript_paths = (
        [BASELINE_500_TRANSCRIPT_REL_PATH.as_posix()]
        if any(
            check.get("resource") == "baseline_500_clause_transcript" and check.get("passed")
            for check in checks
        )
        else []
    )
    artifact = _base_artifact(
        verdict,
        checks,
        instance,
        duration_s,
        board_reachable=board_reachable,
        verified=False,
        scorer_output_sha256="",
        transcript_paths=transcript_paths,
        elapsed_ms=0.0,
    )
    artifact["failure_detail"] = detail
    validate_artifact(artifact)
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
    verdict = (
        "complete: polarfire_1000_clause_constraint_scorer_hash_verified"
        if hash_verified
        else "failed_polarfire_1000_clause_hash_mismatch"
    )
    artifact = _base_artifact(
        verdict,
        checks,
        instance,
        duration_s,
        board_reachable=True,
        verified=hash_verified,
        scorer_output_sha256=remote_hash,
        transcript_paths=[
            BASELINE_500_TRANSCRIPT_REL_PATH.as_posix(),
            TRANSCRIPT_REL_PATH.as_posix(),
        ],
        elapsed_ms=float(transcript.get("total_wall_clock_s", 0.0)) * 1000.0,
    )
    artifact.update(
        {
            "expected_scorer_output_sha256": expected_hash,
            "polarfire_arch": metadata.get("polarfire_arch", ""),
            "remote_arch": transcript.get("remote_arch", ""),
            "remote_python": transcript.get("remote_python", ""),
        }
    )
    validate_artifact(artifact)
    return artifact


def _looks_like_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = REQUIRED_ARTIFACT_FIELDS - artifact.keys()
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["clause_count"] != CLAUSE_COUNT:
        raise ValueError("clause_count must be 1000")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be hardware_smoke")
    if artifact["no_speedup_claim"] is not True:
        raise ValueError("no_speedup_claim must be true")
    if artifact["no_general_acceleration_claim"] is not True:
        raise ValueError("no_general_acceleration_claim must be true")
    if not _looks_like_sha256(artifact["input_sha256"]):
        raise ValueError("input_sha256 must be a sha256 hex digest")
    if artifact["polarfire_1000_clause_hash_verified"] and not _looks_like_sha256(
        artifact["scorer_output_sha256"]
    ):
        raise ValueError("scorer_output_sha256 must be a sha256 hex digest when verified")
    if not artifact["transcript_paths"] and artifact["board_reachable"]:
        raise ValueError("transcript_paths must be present for reachable-board artifacts")


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
    checks, baseline_blocker = check_baseline_context(root_path)
    output_path = root_path / OUTPUT_REL_PATH
    transcript_output_path = root_path / TRANSCRIPT_REL_PATH

    if baseline_blocker is not None:
        artifact = blocked_artifact(
            baseline_blocker,
            checks,
            instance,
            clock() - started,
            detail="Exp 2941 baseline context is unavailable or unverified.",
        )
        _write_json(output_path, artifact)
        return artifact

    board_checks, metadata, board_blocker = dispatch.check_preconditions(runner, host)
    checks = [*checks, *board_checks]
    if board_blocker is not None:
        artifact = blocked_artifact(board_blocker, checks, instance, clock() - started)
        _write_json(output_path, artifact)
        return artifact

    active_remote_dir = remote_dir or f"/tmp/carnot_exp2958_{os.getpid()}_{int(time.time())}"
    try:
        with tempfile.TemporaryDirectory(prefix="carnot_exp2958_") as tmpdir:
            tmp_path = Path(tmpdir)
            files = write_harness_files(tmp_path, instance)
            local_transcript_path = tmp_path / "transcript.json"
            transcript = dispatch.dispatch_to_board(
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
    except dispatch.DispatchError as exc:
        artifact = blocked_artifact(
            exc.verdict,
            checks,
            instance,
            clock() - started,
            board_reachable=True,
            detail=exc.detail,
        )
    _write_json(output_path, artifact)
    return artifact


def _run_local_command(args: Sequence[str], timeout: float | None = None) -> CommandResult:
    try:  # pragma: no cover - exercised only by live SSH hardware execution.
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
    return 0 if artifact["honest_verdict"].startswith(("complete:", "blocked_")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
