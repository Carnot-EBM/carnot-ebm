"""Exp 4056 build gate for EvalPlus OFF-ARC power measurement.

Spec refs: REQ-VERIFY-4056, SCENARIO-VERIFY-4056.

The build step checks live resources, smokes the EvalPlus hidden-test runner on
two tasks, records whether the smoke oracle still has headroom, and launches
the long Exp 4057 raw run detached. It exits after launch.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import offarc_power_evalplus_run as runner

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "scripts" / "experiments" / "offarc_power_evalplus_run.py"
OUTPUT = REPO_ROOT / "results" / "experiment_4056_offarc_power_evalplus_build.json"
SMOKE_OUTPUT = REPO_ROOT / "results" / "experiment_4056_offarc_power_evalplus_smoke_raw.json"
SMOKE_CHECKPOINT = REPO_ROOT / "results" / "offarc_power_evalplus_gemma12b_k8.smoke.checkpoint.json"
RAW_OUTPUT = REPO_ROOT / "results" / "experiment_4057_offarc_power_evalplus_raw.json"
LOG_PATH = REPO_ROOT / "logs" / "offarc_power_evalplus.log"

INFERENCE_SUBSTRATE = "live_llm_inference"
SUCCESS_VERDICT = "success: offarc_power_evalplus_runner_resumed_launched"

REQUIRED_BUILD_FIELDS = [
    "honest_verdict",
    "runner_ready",
    "smoke_passed",
    "smoke_oracle_headroom_present",
    "evaluation_corpus",
    "stable_checkpoint_path",
    "resumed_from_n",
    "launched_pid",
    "arms_implemented",
    "preconditions_checked",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix build verdict; blocked resources stop before inference.",
    "runner_ready": "Bare bool; the COLLECT task gates on this value.",
    "smoke_passed": "Bare bool; a failed two-task smoke must not launch the full run.",
    "smoke_oracle_headroom_present": "Bare bool; oracle<1.0 catches saturation before claims.",
    "evaluation_corpus": "EvalPlus hidden tests are the unsaturated corpus fix.",
    "stable_checkpoint_path": "Corpus+model+k-keyed checkpoint accumulates across windows.",
    "resumed_from_n": "Bare int count from the Exp 4045 candidate-generation checkpoint.",
    "launched_pid": "Bare int; Exp 4057 polls the process owning this run.",
}


def build_artifact(
    *,
    honest_verdict: str,
    runner_ready: bool,
    smoke_passed: bool,
    smoke_oracle_headroom_present: bool,
    preconditions_checked: list[dict[str, Any]],
    resumed_from_n: int,
    launched_pid: int,
    duration_s: float,
    stable_checkpoint_path: Path = runner.STABLE_CHECKPOINT,
    smoke_output_path: Path = SMOKE_OUTPUT,
    raw_output_path: Path = RAW_OUTPUT,
    log_path: Path = LOG_PATH,
    error: str | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": "experiment_4056_offarc_power_evalplus_build",
        "schema": "carnot.experiment_4056_offarc_power_evalplus_build.v1",
        "honest_verdict": honest_verdict,
        "runner_ready": runner_ready,
        "smoke_passed": smoke_passed,
        "smoke_oracle_headroom_present": smoke_oracle_headroom_present,
        "needs_harder_corpus_livecodebench": not smoke_oracle_headroom_present,
        "evaluation_corpus": runner.EVALUATION_CORPUS,
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "resumed_from_n": resumed_from_n,
        "launched_pid": launched_pid,
        "arms_implemented": list(runner.ARMS_IMPLEMENTED),
        "preconditions_checked": preconditions_checked,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 2),
        "runner_path": str(RUNNER_PATH),
        "smoke_raw_path": str(smoke_output_path),
        "raw_output_path": str(raw_output_path),
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
    for field in ("runner_ready", "smoke_passed", "smoke_oracle_headroom_present"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    if not isinstance(artifact.get("needs_harder_corpus_livecodebench"), bool):
        raise ValueError("needs_harder_corpus_livecodebench must be a bare bool")
    for field in ("resumed_from_n", "launched_pid"):
        if not isinstance(artifact[field], int) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare int")
    if verdict.startswith("success:") and artifact["launched_pid"] <= 0:
        raise ValueError("launched_pid must be positive after a successful launch")
    if artifact["evaluation_corpus"] != runner.EVALUATION_CORPUS:
        raise ValueError("evaluation_corpus must name EvalPlus hidden tests")
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
        legacy_checkpoint_path=runner.LEGACY_CHECKPOINT,
        n_tasks=kwargs.get("n_tasks", 2),
        k=kwargs.get("k", runner.DEFAULT_K),
        mode="smoke",
        preconditions_checked=kwargs["preconditions_checked"],
    )


def launch_full_run(  # pragma: no cover - process launch exercised by required command.
    *,
    runner_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    log_path: Path,
    n_tasks: int,
    k: int,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "setsid",
        "nohup",
        sys.executable,
        str(runner_path),
        "--output",
        str(output_path),
        "--checkpoint",
        str(checkpoint_path),
        "--legacy-checkpoint",
        str(runner.LEGACY_CHECKPOINT),
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
            close_fds=True,
        )
    finally:
        log_handle.close()
    return int(process.pid)


def run_build(
    *,
    output_path: Path = OUTPUT,
    smoke_output_path: Path = SMOKE_OUTPUT,
    raw_output_path: Path = RAW_OUTPUT,
    log_path: Path = LOG_PATH,
    stable_checkpoint_path: Path = runner.STABLE_CHECKPOINT,
    precondition_checker: Callable[[], list[dict[str, Any]]] = runner.check_preconditions,
    smoke_runner: Callable[..., dict[str, Any]] = default_smoke_runner,
    launcher: Callable[..., int] = launch_full_run,
    resumed_counter: Callable[[Path], int] = runner.count_checkpoint_completed,
    full_n_tasks: int = runner.DEFAULT_N_TASKS,
    full_k: int = runner.DEFAULT_K,
) -> dict[str, Any]:
    started = time.time()
    preconditions = precondition_checker()
    resumed_from_n = resumed_counter(runner.LEGACY_CHECKPOINT)
    blocker = runner.blocker_from_preconditions(preconditions)
    if blocker:
        artifact = build_artifact(
            honest_verdict=blocker,
            runner_ready=False,
            smoke_passed=False,
            smoke_oracle_headroom_present=False,
            launched_pid=0,
            preconditions_checked=preconditions,
            resumed_from_n=resumed_from_n,
            duration_s=time.time() - started,
            stable_checkpoint_path=stable_checkpoint_path,
            smoke_output_path=smoke_output_path,
            raw_output_path=raw_output_path,
            log_path=log_path,
        )
        _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    try:
        smoke_artifact = smoke_runner(
            output_path=smoke_output_path,
            n_tasks=2,
            k=runner.DEFAULT_K,
            mode="smoke",
            preconditions_checked=preconditions,
        )
        runner.validate_raw_artifact(smoke_artifact, require_full=False)
        smoke_headroom = runner.smoke_oracle_headroom_present(smoke_artifact)
    except Exception as exc:
        artifact = build_artifact(
            honest_verdict="blocked_smoke_failed",
            runner_ready=True,
            smoke_passed=False,
            smoke_oracle_headroom_present=False,
            launched_pid=0,
            preconditions_checked=preconditions,
            resumed_from_n=resumed_from_n,
            duration_s=time.time() - started,
            stable_checkpoint_path=stable_checkpoint_path,
            smoke_output_path=smoke_output_path,
            raw_output_path=raw_output_path,
            log_path=log_path,
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    try:
        pid = launcher(
            runner_path=RUNNER_PATH,
            output_path=raw_output_path,
            checkpoint_path=stable_checkpoint_path,
            log_path=log_path,
            n_tasks=full_n_tasks,
            k=full_k,
        )
    except Exception as exc:
        artifact = build_artifact(
            honest_verdict="blocked_launch_failed",
            runner_ready=True,
            smoke_passed=True,
            smoke_oracle_headroom_present=smoke_headroom,
            launched_pid=0,
            preconditions_checked=preconditions,
            resumed_from_n=resumed_from_n,
            duration_s=time.time() - started,
            stable_checkpoint_path=stable_checkpoint_path,
            smoke_output_path=smoke_output_path,
            raw_output_path=raw_output_path,
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
        smoke_oracle_headroom_present=smoke_headroom,
        launched_pid=pid,
        preconditions_checked=preconditions,
        resumed_from_n=resumed_from_n,
        duration_s=time.time() - started,
        stable_checkpoint_path=stable_checkpoint_path,
        smoke_output_path=smoke_output_path,
        raw_output_path=raw_output_path,
        log_path=log_path,
    )
    _write_json(output_path, artifact)
    print(
        f"-> {artifact['honest_verdict']} pid={pid} smoke_oracle_headroom={smoke_headroom}",
        flush=True,
    )
    return artifact


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI adapter.
    run_build()


if __name__ == "__main__":  # pragma: no cover.
    main()
