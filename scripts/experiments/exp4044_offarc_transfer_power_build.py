"""Exp 4044 build gate for OFF-ARC full-power transfer.

Spec refs: REQ-VERIFY-4044, SCENARIO-VERIFY-4044.

This script checks live resources, smokes the Exp 4045 runner on two code
tasks, launches the full HumanEval plus MBPP run in the background, and writes
the collector-facing build artifact. It does not wait for the full measurement
to finish.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import offarc_transfer_power_run as runner

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "scripts" / "experiments" / "offarc_transfer_power_run.py"
OUTPUT = REPO_ROOT / "results" / "experiment_4044_offarc_transfer_power_build.json"
SMOKE_OUTPUT = REPO_ROOT / "results" / "experiment_4044_offarc_transfer_power_smoke_raw.json"
SMOKE_CHECKPOINT = (
    REPO_ROOT / "results" / "experiment_4044_offarc_transfer_power_smoke.checkpoint.json"
)
FULL_OUTPUT = REPO_ROOT / "results" / "experiment_4045_offarc_transfer_power_raw.json"
LOG_PATH = REPO_ROOT / "logs" / "offarc_power_run.log"

INFERENCE_SUBSTRATE = "live_llm_inference"
SUCCESS_VERDICT = "success: offarc_power_runner_built_smoked_launched_humaneval_mbpp"

REQUIRED_BUILD_FIELDS = [
    "honest_verdict",
    "runner_ready",
    "smoke_passed",
    "launched_pid",
    "arms_implemented",
    "preconditions_checked",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix build verdict; blocked resources stop before inference.",
    "runner_ready": "Bare bool; the collector gates on this value.",
    "smoke_passed": "Bare bool; a failed two-task smoke must not launch the full run.",
    "launched_pid": "Bare int; Exp 4045 polls the process owning the full run.",
    "arms_implemented": "List of the four arms required for the stronger-discriminator comparison.",
    "preconditions_checked": "List of resources verified before any inference.",
    "inference_substrate": "live_llm_inference because smoke and full launch use the local GGUF.",
}


def build_artifact(
    *,
    honest_verdict: str,
    runner_ready: bool,
    smoke_passed: bool,
    launched_pid: int,
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
    smoke_output_path: Path = SMOKE_OUTPUT,
    full_output_path: Path = FULL_OUTPUT,
    log_path: Path = LOG_PATH,
    error: str | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": "experiment_4044_offarc_transfer_power_build",
        "schema": "carnot.experiment_4044_offarc_transfer_power_build.v1",
        "honest_verdict": honest_verdict,
        "runner_ready": runner_ready,
        "smoke_passed": smoke_passed,
        "launched_pid": launched_pid,
        "arms_implemented": list(runner.ARMS_IMPLEMENTED),
        "preconditions_checked": preconditions_checked,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 2),
        "runner_path": str(RUNNER_PATH),
        "smoke_raw_path": str(smoke_output_path),
        "full_raw_path": str(full_output_path),
        "log_path": str(log_path),
        "field_principles": FIELD_PRINCIPLES,
    }
    if error:
        artifact["error"] = error
    validate_build_artifact(artifact)
    return artifact


def validate_build_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_BUILD_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required build field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("success:")
        or verdict.startswith("complete:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in ("runner_ready", "smoke_passed"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    if not isinstance(artifact["launched_pid"], int) or isinstance(artifact["launched_pid"], bool):
        raise ValueError("launched_pid must be a bare int")
    if verdict.startswith("success:") and artifact["launched_pid"] <= 0:
        raise ValueError("launched_pid must be positive after a successful launch")
    if not isinstance(artifact["arms_implemented"], list):
        raise ValueError("arms_implemented must be a list")
    if artifact["arms_implemented"] != runner.ARMS_IMPLEMENTED:
        raise ValueError("arms_implemented must list the four required arms")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")


def default_smoke_runner(**kwargs: Any) -> dict[str, Any]:  # pragma: no cover - live GGUF smoke.
    return runner.run(
        output_path=kwargs["output_path"],
        checkpoint_path=SMOKE_CHECKPOINT,
        n_tasks=kwargs.get("n_tasks", 2),
        k=kwargs.get("k", 1),
        mode="smoke",
        preconditions_checked=kwargs["preconditions_checked"],
    )


def launch_full_run(  # pragma: no cover - process launch exercised by required command.
    *,
    runner_path: Path,
    output_path: Path,
    log_path: Path,
    n_tasks: int,
    k: int,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nohup",
        sys.executable,
        str(runner_path),
        "--output",
        str(output_path),
        "--n-tasks",
        str(n_tasks),
        "--k",
        str(k),
        "--mode",
        "full",
    ]
    log_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_handle.close()
    return int(process.pid)


def run_build(
    *,
    output_path: Path = OUTPUT,
    smoke_output_path: Path = SMOKE_OUTPUT,
    full_output_path: Path = FULL_OUTPUT,
    log_path: Path = LOG_PATH,
    precondition_checker: Callable[[], list[dict[str, Any]]] = runner.check_preconditions,
    smoke_runner: Callable[..., dict[str, Any]] = default_smoke_runner,
    launcher: Callable[..., int] = launch_full_run,
    full_n_tasks: int = runner.DEFAULT_N_TASKS,
    full_k: int = runner.DEFAULT_K,
) -> dict[str, Any]:
    started = time.time()
    preconditions = precondition_checker()
    blocker = runner.blocker_from_preconditions(preconditions)
    if blocker:
        artifact = build_artifact(
            honest_verdict=blocker,
            runner_ready=False,
            smoke_passed=False,
            launched_pid=0,
            preconditions_checked=preconditions,
            duration_s=time.time() - started,
            smoke_output_path=smoke_output_path,
            full_output_path=full_output_path,
            log_path=log_path,
        )
        _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    try:
        smoke_artifact = smoke_runner(
            output_path=smoke_output_path,
            n_tasks=2,
            k=1,
            mode="smoke",
            preconditions_checked=preconditions,
        )
        runner.validate_raw_artifact(smoke_artifact, require_full=False)
    except Exception as exc:
        artifact = build_artifact(
            honest_verdict="blocked_smoke_failed",
            runner_ready=True,
            smoke_passed=False,
            launched_pid=0,
            preconditions_checked=preconditions,
            duration_s=time.time() - started,
            smoke_output_path=smoke_output_path,
            full_output_path=full_output_path,
            log_path=log_path,
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    try:
        pid = launcher(
            runner_path=RUNNER_PATH,
            output_path=full_output_path,
            log_path=log_path,
            n_tasks=full_n_tasks,
            k=full_k,
        )
    except Exception as exc:
        artifact = build_artifact(
            honest_verdict="blocked_launch_failed",
            runner_ready=True,
            smoke_passed=True,
            launched_pid=0,
            preconditions_checked=preconditions,
            duration_s=time.time() - started,
            smoke_output_path=smoke_output_path,
            full_output_path=full_output_path,
            log_path=log_path,
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    artifact = build_artifact(
        honest_verdict=SUCCESS_VERDICT,
        runner_ready=True,
        smoke_passed=True,
        launched_pid=pid,
        preconditions_checked=preconditions,
        duration_s=time.time() - started,
        smoke_output_path=smoke_output_path,
        full_output_path=full_output_path,
        log_path=log_path,
    )
    _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']} pid={pid}", flush=True)
    return artifact


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover
    run_build()


if __name__ == "__main__":  # pragma: no cover
    main()
