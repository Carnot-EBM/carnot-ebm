"""Exp 4047 build gate for the MoE-base decentralization run.

Spec refs: REQ-VERIFY-4047, SCENARIO-VERIFY-4047.

The build half checks the Qwen MoE resources, smokes Exp 4048 on two tasks,
and launches the full best-of-N run in the background. It does not wait for the
long run; Exp 4048 checkpoints after each task.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import experiment_4048_decentralization_moe_base_best_of_n as run4048
from experiment_4002_gap4_local_generator_arm import POOL
from scripts.experiment_template import _compute_repro_checksum

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "results" / "experiment_4047_decentralization_moe_base_build.json"
SMOKE_OUTPUT = REPO_ROOT / "results" / "experiment_4048_decentralization_moe_base_smoke.json"
RAW_OUTPUT = REPO_ROOT / "results" / "experiment_4048_decentralization_moe_base_raw.json"
SMOKE_CHECKPOINT = (
    REPO_ROOT / "results" / "experiment_4048_decentralization_moe_base_smoke.checkpoint.json"
)
FULL_CHECKPOINT = (
    REPO_ROOT / "results" / "experiment_4048_decentralization_moe_base_raw.checkpoint.json"
)
LOG_PATH = REPO_ROOT / "logs" / "decentralization_moe_run.log"

INFERENCE_SUBSTRATE = "live_llm_inference"
DEFAULT_FULL_K = 8
DEFAULT_SMOKE_K = 8
DEFAULT_SMOKE_LIMIT = 2
DEFAULT_FULL_TIME_BUDGET_S = 4500.0

REQUIRED_BUILD_FIELDS = [
    "honest_verdict",
    "runner_ready",
    "moe_base_model",
    "smoke_per_task_seconds",
    "smoke_passed",
    "launched_pid",
    "preconditions_checked",
    "inference_substrate",
]


@dataclass(frozen=True)
class LaunchSpec:
    argv: list[str]
    log_path: Path
    k: int
    max_wall_s: float

    @property
    def command_text(self) -> str:
        return " ".join(self.argv) + f" > {self.log_path} 2>&1 &"


def _is_bare_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bare_float(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def validate_build_artifact(artifact: dict[str, Any]) -> None:
    """Validate the fields that the conductor gates on."""
    for field in REQUIRED_BUILD_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
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
    if not _is_bare_float(artifact["smoke_per_task_seconds"]):
        raise ValueError("smoke_per_task_seconds must be a bare float")
    if not _is_bare_int(artifact["launched_pid"]):
        raise ValueError("launched_pid must be a bare int")
    for field in ("moe_base_model", "inference_substrate"):
        if not isinstance(artifact[field], str):
            raise ValueError(f"{field} must be a string")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")
    if "duration_s" in artifact and not _is_bare_float(artifact["duration_s"]):
        raise ValueError("duration_s must be a bare float")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repro_checksum(pool_path: Path | str) -> str:
    return _compute_repro_checksum(
        run4048.SEED,
        [
            str(Path(__file__).resolve()),
            str(
                (
                    REPO_ROOT
                    / "scripts"
                    / "experiments"
                    / "experiment_4048_decentralization_moe_base_best_of_n.py"
                )
            ),
            str(
                (REPO_ROOT / "scripts" / "experiments" / "experiment_4012_gap4_local_best_of_n.py")
            ),
        ],
        str(pool_path),
    )


def blocked_build_artifact(
    verdict: str,
    *,
    chosen_model: dict[str, str] | None,
    preconditions: list[dict[str, Any]],
    duration_s: float,
    output_path: Path,
    smoke_error: str | None = None,
    pool_path: Path | str = POOL,
) -> dict[str, Any]:
    """Build a blocked Exp 4047 artifact without launching inference."""
    artifact = {
        "experiment": "experiment_4047_decentralization_moe_base_build",
        "schema": "carnot.experiment_4047_decentralization_moe_base_build.v1",
        "title": "Decentralization MoE-base runner build and launch gate",
        "honest_verdict": verdict,
        "runner_ready": False,
        "moe_base_model": chosen_model["name"] if chosen_model else "none",
        "smoke_per_task_seconds": 0.0,
        "smoke_passed": False,
        "launched_pid": 0,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 2),
        "build_artifact_path": str(output_path),
        "smoke_error": smoke_error,
        "model_specs": {
            "generator_model": chosen_model["name"] if chosen_model else "none",
            "generator_hf_id": chosen_model["hf_id"] if chosen_model else "none",
            "generator_gguf_path": chosen_model["model_path"] if chosen_model else "none",
        },
        "random_seed": run4048.SEED,
        "reproducibility_checksum": _repro_checksum(pool_path),
    }
    validate_build_artifact(artifact)
    return artifact


def _python_executable() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def make_launch_spec(
    *,
    raw_output_path: Path,
    checkpoint_path: Path,
    log_path: Path,
    model_key: str,
    k: int,
    max_wall_s: float,
    n_ctx: int,
    batch_size: int,
) -> LaunchSpec:
    script = (
        REPO_ROOT
        / "scripts"
        / "experiments"
        / "experiment_4048_decentralization_moe_base_best_of_n.py"
    )
    argv = [
        "nohup",
        _python_executable(),
        str(script),
        "--model",
        model_key,
        "--k",
        str(k),
        "--max-wall-s",
        str(max_wall_s),
        "--n-ctx",
        str(n_ctx),
        "--batch-size",
        str(batch_size),
        "--output",
        str(raw_output_path),
        "--checkpoint",
        str(checkpoint_path),
    ]
    return LaunchSpec(argv=argv, log_path=log_path, k=k, max_wall_s=max_wall_s)


def launch_background(spec: LaunchSpec) -> int:  # pragma: no cover - exercised by operator run.
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = spec.log_path.open("wb")
    proc = subprocess.Popen(
        spec.argv,
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    return int(proc.pid)


def _smoke_seconds_per_task(smoke_artifact: dict[str, Any]) -> float:
    per_task = smoke_artifact.get("per_task")
    if isinstance(per_task, list) and per_task:
        unique_tasks = {row.get("task") for row in per_task if row.get("task") is not None}
        denom = max(1, len(unique_tasks))
    else:
        denom = max(1, int(smoke_artifact.get("n_unique_tasks", 1) or 1))
    return round(float(smoke_artifact.get("local_seconds", 0.0)) / denom, 2)


def success_build_artifact(
    *,
    chosen_model: dict[str, str],
    preconditions: list[dict[str, Any]],
    smoke_artifact: dict[str, Any],
    launch_spec: LaunchSpec,
    launched_pid: int,
    output_path: Path,
    smoke_output_path: Path,
    raw_output_path: Path,
    duration_s: float,
    pool_path: Path | str,
) -> dict[str, Any]:
    """Build the successful Exp 4047 launch receipt."""
    artifact = {
        "experiment": "experiment_4047_decentralization_moe_base_build",
        "schema": "carnot.experiment_4047_decentralization_moe_base_build.v1",
        "title": "Decentralization MoE-base runner build and launch gate",
        "honest_verdict": "success: decentralization_moe_base_runner_launched_qwen35moe",
        "runner_ready": True,
        "moe_base_model": chosen_model["name"],
        "smoke_per_task_seconds": _smoke_seconds_per_task(smoke_artifact),
        "smoke_passed": True,
        "launched_pid": int(launched_pid),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 2),
        "build_artifact_path": str(output_path),
        "smoke_artifact_path": str(smoke_output_path),
        "raw_artifact_path": str(raw_output_path),
        "launch_command": launch_spec.command_text,
        "full_time_budget_s": float(launch_spec.max_wall_s),
        "full_k_samples_per_task": int(launch_spec.k),
        "smoke_best_of_n_coverage": float(smoke_artifact["best_of_n_coverage"]),
        "model_specs": smoke_artifact["model_specs"],
        "random_seed": run4048.SEED,
        "reproducibility_checksum": _repro_checksum(pool_path),
    }
    validate_build_artifact(artifact)
    return artifact


def run_build(
    *,
    pool_path: Path | str = POOL,
    output_path: Path = OUTPUT,
    smoke_output_path: Path = SMOKE_OUTPUT,
    raw_output_path: Path = RAW_OUTPUT,
    smoke_checkpoint_path: Path = SMOKE_CHECKPOINT,
    full_checkpoint_path: Path = FULL_CHECKPOINT,
    log_path: Path = LOG_PATH,
    model_key: str = "auto",
    resolver: Callable[[str], str | None] = run4048.exp4012.resolve_local_gguf,
    cache_dir: Path | str = run4048.MOE_CACHE_DIR,
    llama_available_override: bool | None = None,
    smoke_sampler: Any | None = None,
    launcher: Callable[[LaunchSpec], int] = launch_background,
    smoke_k: int = DEFAULT_SMOKE_K,
    smoke_limit: int = DEFAULT_SMOKE_LIMIT,
    full_k: int = DEFAULT_FULL_K,
    full_time_budget_s: float = DEFAULT_FULL_TIME_BUDGET_S,
    n_ctx: int = 16384,
    batch_size: int = run4048.DEFAULT_DRAW_BATCH_SIZE,
    write: bool = True,
) -> dict[str, Any]:
    """Check resources, smoke Exp 4048, write the build receipt, and launch full Exp 4048."""
    started = time.time()
    preconditions, chosen_model = run4048.check_preconditions(
        model_key=model_key,
        pool_path=pool_path,
        resolver=resolver,
        cache_dir=cache_dir,
        llama_available_override=llama_available_override,
    )
    blocker = run4048.blocker_from_preconditions(preconditions)
    if blocker:
        artifact = blocked_build_artifact(
            blocker,
            chosen_model=chosen_model,
            preconditions=preconditions,
            duration_s=time.time() - started,
            output_path=output_path,
            pool_path=pool_path,
        )
        if write:
            _write_json(output_path, artifact)
        return artifact

    if chosen_model is None:  # pragma: no cover - defensive; blocker should have caught this.
        raise RuntimeError("MoE model unavailable after precondition pass")

    try:
        smoke_artifact = run4048.run(
            model_key=chosen_model["model_key"],
            pool_path=pool_path,
            output_path=smoke_output_path,
            checkpoint_path=smoke_checkpoint_path,
            k=smoke_k,
            limit=smoke_limit,
            max_wall_s=full_time_budget_s,
            n_ctx=n_ctx,
            batch_size=batch_size,
            sampler=smoke_sampler,
            resolver=resolver,
            cache_dir=cache_dir,
            llama_available_override=llama_available_override,
            write=True,
        )
        run4048.validate_raw_artifact(smoke_artifact)
        smoke_ok = smoke_artifact.get("runner_ready") is True
    except Exception as exc:  # pragma: no cover - live smoke failure path.
        smoke_artifact = {}
        smoke_ok = False
        smoke_error = f"{type(exc).__name__}: {exc}"
    else:
        smoke_error = None

    if not smoke_ok:
        artifact = blocked_build_artifact(
            "blocked_smoke_failed",
            chosen_model=chosen_model,
            preconditions=preconditions,
            duration_s=time.time() - started,
            output_path=output_path,
            smoke_error=smoke_error,
            pool_path=pool_path,
        )
        if write:
            _write_json(output_path, artifact)
        return artifact

    launch_spec = make_launch_spec(
        raw_output_path=raw_output_path,
        checkpoint_path=full_checkpoint_path,
        log_path=log_path,
        model_key=chosen_model["model_key"],
        k=full_k,
        max_wall_s=full_time_budget_s,
        n_ctx=n_ctx,
        batch_size=batch_size,
    )
    launched_pid = launcher(launch_spec)
    artifact = success_build_artifact(
        chosen_model=chosen_model,
        preconditions=preconditions,
        smoke_artifact=smoke_artifact,
        launch_spec=launch_spec,
        launched_pid=launched_pid,
        output_path=output_path,
        smoke_output_path=smoke_output_path,
        raw_output_path=raw_output_path,
        duration_s=time.time() - started,
        pool_path=pool_path,
    )
    if write:
        _write_json(output_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised by required operator command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-time-budget-s", type=float, default=DEFAULT_FULL_TIME_BUDGET_S)
    parser.add_argument("--smoke-k", type=int, default=DEFAULT_SMOKE_K)
    parser.add_argument("--batch-size", type=int, default=run4048.DEFAULT_DRAW_BATCH_SIZE)
    args = parser.parse_args()
    artifact = run_build(
        full_time_budget_s=args.full_time_budget_s,
        smoke_k=args.smoke_k,
        batch_size=args.batch_size,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
