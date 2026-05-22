"""Exp 2839 TruthfulQA-generation dual-condition v7b evaluator.

This module is the third TruthfulQA dual-condition attempt. It reuses the
Exp 2831 v7b artifact builder so the schema stays compatible with the earlier
blocked run, but writes a distinct Exp 2839 artifact and adds the explicit
``.venv/bin/python3`` CUDA precondition requested for the post-torch-fix rerun.
Missing live resources still produce a terminal ``blocked_*`` artifact with
null metrics rather than invented AUROC, BLEURT threshold, or per-verifier
scores.

Spec traces: REQ-VERIFY-2839-TQA,
SCENARIO-VERIFY-2839-TQA-BLOCKED,
SCENARIO-VERIFY-2839-TQA-LIVE.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from carnot.eval import truthfulqa_ensemble_v7b as base


OUTPUT_FILENAME = "experiment_2839_truthfulqa_ensemble_eval.json"
ARTIFACT = "experiment_2839_truthfulqa_ensemble_eval"
SCHEMA = "carnot.truthfulqa_ensemble_v7b_exp2839"
REPO_ROOT = base.REPO_ROOT

ExperimentConfig = base.ExperimentConfig
PreconditionCheck = base.PreconditionCheck
SeedMeasurement = base.SeedMeasurement
LiveMeasurementUnavailable = base.LiveMeasurementUnavailable
discover_fr11_state_files = base.discover_fr11_state_files

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _venv_python_path(repo_root: Path) -> Path:
    return repo_root / ".venv" / "bin" / "python3"


def venv_cuda_check(
    config: ExperimentConfig,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> PreconditionCheck:
    """Check CUDA through the repository venv before live TruthfulQA work."""

    python_path = _venv_python_path(config.repo_root)
    if not python_path.is_file():
        return PreconditionCheck("cuda", False, f"{python_path} missing")

    script = """
import json
try:
    import torch
    available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    detail = (
        f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}; "
        f"device_count={torch.cuda.device_count()}"
    )
except Exception as exc:
    available = False
    detail = f"{type(exc).__name__}: {exc}"
print(json.dumps({"available": available, "detail": detail}))
"""
    try:
        proc = command_runner(
            [str(python_path), "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreconditionCheck("cuda", False, f"{type(exc).__name__}: {exc}")

    if proc.returncode != 0:
        return PreconditionCheck(
            "cuda",
            False,
            (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip(),
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return PreconditionCheck(
            "cuda",
            False,
            f"invalid JSON probe output: {proc.stdout[:200]}",
        )
    return PreconditionCheck(
        "cuda",
        bool(payload.get("available")),
        f"cmd=.venv/bin/python3; {payload.get('detail', '')}",
    )


def probe_preconditions(
    config: ExperimentConfig,
    state_files: Sequence[dict[str, object]],
    *,
    cuda_check: Callable[[ExperimentConfig], PreconditionCheck] = venv_cuda_check,
    hf_truthfulqa_check: Callable[[], PreconditionCheck] = base._hf_truthfulqa_check,
    qwen_cache_check: Callable[[Path], PreconditionCheck] = base._qwen_cache_check,
    bleurt_check: Callable[[Path], PreconditionCheck] = base._bleurt_check,
) -> list[PreconditionCheck]:
    """Check Exp 2839 resources before generation, BLEURT, or verifier scoring."""

    return [
        cuda_check(config),
        hf_truthfulqa_check(),
        qwen_cache_check(config.repo_root),
        PreconditionCheck(
            "fr11_state_files",
            bool(state_files),
            f"{len(state_files)} FR-11 state files discovered",
        ),
        bleurt_check(config.repo_root),
    ]


def _tag_exp2839(artifact: dict[str, object]) -> dict[str, object]:
    artifact["artifact"] = ARTIFACT
    artifact["schema"] = SCHEMA
    return artifact


def write_artifact(results_dir: Path, artifact: dict[str, object]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / OUTPUT_FILENAME).write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    precondition_probe: Callable[
        [ExperimentConfig, Sequence[dict[str, object]]], list[PreconditionCheck]
    ] = probe_preconditions,
    measurement_runner: Callable[
        [ExperimentConfig, Sequence[dict[str, object]]], Sequence[SeedMeasurement]
    ] = base.default_live_measurement_runner,
    write: bool = True,
) -> dict[str, object]:
    """Run Exp 2839 or write an honest blocked artifact before inference."""

    config = config or ExperimentConfig()
    start = config.start_time()
    results_dir = config.output_dir()
    state_files = discover_fr11_state_files(config.repo_root)
    checks = precondition_probe(config, state_files)
    specs = {
        **base.model_specs(config.repo_root),
        "cuda_precondition_python": str(_venv_python_path(config.repo_root)),
    }

    if any(not check.available for check in checks):
        artifact = base._blocked_artifact(
            config=config,
            duration_s=config.clock() - start,
            state_files=state_files,
            checks=checks,
            specs=specs,
        )
    else:
        try:
            measurements = list(measurement_runner(config, state_files))
        except LiveMeasurementUnavailable:
            checks = [*checks, PreconditionCheck("live_backend", False, "backend unavailable")]
            artifact = base._blocked_artifact(
                config=config,
                duration_s=config.clock() - start,
                state_files=state_files,
                checks=checks,
                specs=specs,
            )
        else:
            artifact = base._success_artifact(
                config=config,
                duration_s=config.clock() - start,
                state_files=state_files,
                checks=checks,
                specs=specs,
                measurements=measurements,
            )

    artifact = _tag_exp2839(artifact)
    if write:
        write_artifact(results_dir, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--n-questions", type=int, default=base.DEFAULT_N_QUESTIONS)
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root)
    run_experiment(
        ExperimentConfig(
            repo_root=repo_root,
            results_dir=Path(args.results_dir) if args.results_dir else repo_root / "results",
            n_questions=args.n_questions,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
