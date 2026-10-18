"""Exp 2839 HumanEval full dual-condition evaluator.

This runner is the HumanEval retry gated by Exp 2836's SOTA runtime preflight.
It records a mandated local GGUF path from the preflight artifact, checks the
full HumanEval dataset and sandboxed execution before candidate work, and writes
a terminal blocked artifact rather than inferring benchmark metrics when a live
generation/scoring backend is unavailable.

Spec: REQ-VERIFY-2839,
      SCENARIO-VERIFY-2839-BLOCKED,
      SCENARIO-VERIFY-2839-LIVE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import (
    discover_fr11_state_files,
    state_files_restored_sha_match,
)


OUTPUT_FILENAME = "experiment_2839_humaneval_dual_condition_v3.json"
EXP2836_FILENAME = "experiment_2836_sota_runtime_preflight.json"
CORPUS = "HumanEval-full"
DEFAULT_RANDOM_SEEDS = (42, 137, 271, 314, 1729)
DEFAULT_N_TASKS = 164
PRIMARY_SOTA_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LEGACY_CPU_SMOKE_ONLY = ("Qwen3.5-0.8B", "gemma-4-E4B-it")
REPO_ROOT = Path(__file__).resolve().parents[3]

FIELD_PRINCIPLES = {
    "honest_verdict": 'MUST start with "complete:" / "success:" or "blocked_".',
    "n_tasks": "HumanEval full benchmark should be 164 tasks.",
    "condition_a_production_auroc_mean": "Production AUROC measured from test labels.",
    "condition_b_architecture_only_auroc_mean": (
        "Architecture-only AUROC measured from test labels."
    ),
    "learning_contribution": "A - B transfer contribution.",
    "per_verifier_condition_b_auroc": "Cross-corpus matrix input.",
    "model_specs": "Mandated SOTA GGUF recorded.",
    "preconditions_checked": "Explains blocks honestly.",
    "duration_s": "Real compute wall-time; no sleep padding.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before any HumanEval candidate work."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SeedEvaluation:
    """Measured HumanEval outcomes for one seed across both memory conditions."""

    seed: int
    n_tasks: int
    n_candidates: int
    condition_a_ensemble_auroc: float
    condition_b_ensemble_auroc: float
    condition_a_per_verifier_auroc: dict[str, float]
    condition_b_per_verifier_auroc: dict[str, float]
    vanilla_pass_at_1: float
    condition_a_ranked_pass_at_1: float
    condition_b_ranked_pass_at_1: float
    scorer_or_generator_model_path: str
    candidate_label_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "n_tasks": self.n_tasks,
            "n_candidates": self.n_candidates,
            "condition_a_production_auroc": self.condition_a_ensemble_auroc,
            "condition_b_architecture_only_auroc": self.condition_b_ensemble_auroc,
            "condition_a_per_verifier_auroc": dict(self.condition_a_per_verifier_auroc),
            "condition_b_per_verifier_auroc": dict(self.condition_b_per_verifier_auroc),
            "vanilla_pass_at_1": self.vanilla_pass_at_1,
            "condition_a_ranked_pass_at_1": self.condition_a_ranked_pass_at_1,
            "condition_b_ranked_pass_at_1": self.condition_b_ranked_pass_at_1,
            "condition_a_ranking_lift": (
                self.condition_a_ranked_pass_at_1 - self.vanilla_pass_at_1
            ),
            "condition_b_ranking_lift": (
                self.condition_b_ranked_pass_at_1 - self.vanilla_pass_at_1
            ),
            "scorer_or_generator_model_path": self.scorer_or_generator_model_path,
            "candidate_label_sha256": self.candidate_label_sha256,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 2839."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    exp2836_path: Path | None = None
    random_seeds: tuple[int, ...] = DEFAULT_RANDOM_SEEDS
    n_tasks: int = DEFAULT_N_TASKS
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    probe_timeout_s: int = 30

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def preflight_path(self) -> Path:
        if self.exp2836_path is not None:
            return self.exp2836_path
        return self.output_dir() / EXP2836_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


class LiveHumanEvalMeasurementUnavailable(RuntimeError):
    """Raised when no real HumanEval generation/scoring backend is attached."""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def load_exp2836_preflight(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_model_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"model_path", "path", "resolved_gguf", "resolved_path"} and item:
                paths.append(str(item))
            else:
                paths.extend(_extract_model_paths(item))
    elif isinstance(value, list | tuple):
        for item in value:
            paths.extend(_extract_model_paths(item))
    return paths


def model_specs_from_exp2836(preflight: dict[str, Any]) -> dict[str, object]:
    """Normalize Exp 2836 runtime evidence for the HumanEval artifact."""

    cached_pair = dict(preflight.get("cached_sota_pair_result") or {})
    cached_pair_paths = _extract_model_paths(cached_pair.get("result"))
    smoke_results = [
        dict(row)
        for row in preflight.get("smoke_load_results", [])
        if row.get("load_success") and row.get("headline_usable") and row.get("model_path")
    ]
    cached_models = [
        dict(row)
        for row in preflight.get("sota_models_cached", [])
        if row.get("hf_id") in PRIMARY_SOTA_MODEL_IDS and row.get("path")
    ]
    smoke_paths = [str(row["model_path"]) for row in smoke_results]
    cached_model_paths = [str(row["path"]) for row in cached_models]
    selected_path = (cached_pair_paths or smoke_paths or cached_model_paths or [None])[0]
    selected_hf_id = None
    for row in [*smoke_results, *cached_models]:
        if row.get("model_path") == selected_path or row.get("path") == selected_path:
            selected_hf_id = str(row.get("hf_id"))
            break
    raw_specs = dict(preflight.get("model_specs") or {})
    return {
        "headline_required_any_of": list(raw_specs.get("primary") or PRIMARY_SOTA_MODEL_IDS),
        "legacy_cpu_smoke_only": list(
            raw_specs.get("legacy_cpu_smoke_only") or LEGACY_CPU_SMOKE_ONLY
        ),
        "sota_runtime_ready": bool(preflight.get("sota_runtime_ready")),
        "selected_python": preflight.get("selected_python"),
        "cached_sota_pair_result": cached_pair,
        "cached_sota_pair_model_paths": cached_pair_paths,
        "smoke_model_paths": smoke_paths,
        "sota_models_cached": cached_models,
        "selected_model_path": selected_path,
        "selected_model_hf_id": selected_hf_id,
        "scorer_or_generator_model_paths_used": [],
    }


def _run_json_probe(
    *,
    selected_python: str,
    repo_root: Path,
    script: str,
    resource: str,
    command_runner: CommandRunner = subprocess.run,
    timeout_s: int = 30,
) -> PreconditionCheck:
    if not selected_python:
        return PreconditionCheck(resource, False, "selected_python missing")
    env = os.environ.copy()
    python_dir = str(repo_root / "python")
    env["PYTHONPATH"] = python_dir + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    try:
        proc = command_runner(
            [selected_python, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - host dependent
        return PreconditionCheck(resource, False, f"{type(exc).__name__}: {exc}")
    if proc.returncode != 0:
        return PreconditionCheck(
            resource,
            False,
            (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip(),
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return PreconditionCheck(resource, False, f"invalid JSON probe output: {proc.stdout[:200]}")
    return PreconditionCheck(
        resource,
        bool(payload.get("available")),
        str(payload.get("detail", "")),
    )


def _selected_python_cuda_check(
    selected_python: str,
    repo_root: Path,
    *,
    command_runner: CommandRunner = subprocess.run,
    timeout_s: int = 30,
) -> PreconditionCheck:
    script = """
import json
try:
    import torch
    available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    detail = f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}; device_count={torch.cuda.device_count()}"
except Exception as exc:
    available = False
    detail = f"{type(exc).__name__}: {exc}"
print(json.dumps({"available": available, "detail": detail}))
"""
    return _run_json_probe(
        selected_python=selected_python,
        repo_root=repo_root,
        script=script,
        resource="selected_python_cuda",
        command_runner=command_runner,
        timeout_s=timeout_s,
    )


def _humaneval_dataset_check(
    selected_python: str,
    repo_root: Path,
    *,
    command_runner: CommandRunner = subprocess.run,
    timeout_s: int = 30,
) -> PreconditionCheck:
    script = """
import importlib.util
import json
if importlib.util.find_spec("datasets") is None:
    print(json.dumps({"available": False, "detail": "datasets package is not installed"}))
else:
    try:
        from datasets import load_dataset
        rows = load_dataset("openai_humaneval", split="test")
        required = {"prompt", "test", "entry_point"}
        first = rows[0] if len(rows) else {}
        has_required_fields = required.issubset(set(first))
        available = bool(len(rows) == 164 and has_required_fields)
        detail = f"loaded openai_humaneval test, n={len(rows)}, has_required_fields={has_required_fields}"
    except Exception as exc:
        available = False
        detail = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"available": available, "detail": detail}))
"""
    return _run_json_probe(
        selected_python=selected_python,
        repo_root=repo_root,
        script=script,
        resource="humaneval_dataset",
        command_runner=command_runner,
        timeout_s=timeout_s,
    )


def _sandbox_check(
    selected_python: str,
    repo_root: Path,
    *,
    command_runner: CommandRunner = subprocess.run,
    timeout_s: int = 30,
) -> PreconditionCheck:
    script = """
import json
import os
os.environ["CARNOT_REQUIRE_SANDBOX"] = "1"
try:
    from carnot.verify.sandbox import get_sandbox_status
    from carnot.verify.python_types import safe_exec_function
    status = get_sandbox_status()
    result, error = safe_exec_function("def f():\\n    return 1\\n", "f", (), timeout=1.0)
    available = bool(status.get("available") and error is None and result == 1)
    detail = f"runsc_available={status.get('available')}; smoke_error={error}"
except Exception as exc:
    available = False
    detail = f"{type(exc).__name__}: {exc}"
print(json.dumps({"available": available, "detail": detail}))
"""
    return _run_json_probe(
        selected_python=selected_python,
        repo_root=repo_root,
        script=script,
        resource="sandboxed_unit_test_execution",
        command_runner=command_runner,
        timeout_s=timeout_s,
    )


def probe_preconditions(
    config: ExperimentConfig,
    state_files: Sequence[dict[str, object]],
    model_specs: dict[str, object],
    *,
    command_runner: CommandRunner = subprocess.run,
) -> list[PreconditionCheck]:
    """Check all Exp 2839 live resources before candidate generation."""

    selected_python = str(model_specs.get("selected_python") or "")
    selected_model_path = str(model_specs.get("selected_model_path") or "")
    model_path_ok = bool(selected_model_path and Path(selected_model_path).is_file())
    checks = [
        PreconditionCheck(
            "exp2836_artifact",
            config.preflight_path().is_file(),
            str(config.preflight_path()) if config.preflight_path().is_file() else "missing",
        ),
        PreconditionCheck(
            "exp2836_sota_runtime_ready",
            bool(model_specs.get("sota_runtime_ready")),
            f"sota_runtime_ready={model_specs.get('sota_runtime_ready')}",
        ),
        PreconditionCheck(
            "exp2836_selected_python",
            bool(selected_python),
            selected_python if selected_python else "missing",
        ),
        PreconditionCheck(
            "mandated_sota_model_path",
            model_path_ok,
            selected_model_path if selected_model_path else "missing",
        ),
    ]
    checks.extend(
        [
            _selected_python_cuda_check(
                selected_python,
                config.repo_root,
                command_runner=command_runner,
                timeout_s=config.probe_timeout_s,
            ),
            _humaneval_dataset_check(
                selected_python,
                config.repo_root,
                command_runner=command_runner,
                timeout_s=config.probe_timeout_s,
            ),
            _sandbox_check(
                selected_python,
                config.repo_root,
                command_runner=command_runner,
                timeout_s=config.probe_timeout_s,
            ),
            PreconditionCheck(
                "fr11_state_files",
                bool(state_files),
                f"count={len(state_files)}",
            ),
        ]
    )
    return checks


def _blocked_verdict(checks: Sequence[PreconditionCheck]) -> str | None:
    verdict_by_resource = {
        "exp2836_artifact": "blocked_exp2836_missing",
        "exp2836_sota_runtime_ready": "blocked_sota_runtime_not_ready",
        "exp2836_selected_python": "blocked_selected_python_missing",
        "mandated_sota_model_path": "blocked_model_path",
        "selected_python_cuda": "blocked_cuda_runtime",
        "humaneval_dataset": "blocked_humaneval_dataset",
        "sandboxed_unit_test_execution": "blocked_sandboxed_unit_test_execution",
        "fr11_state_files": "blocked_fr11_state_files",
        "live_backend": "blocked_live_humaneval_backend_unavailable",
    }
    for check in checks:
        if not check.available:
            return verdict_by_resource.get(check.resource, f"blocked_{check.resource}")
    return None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _population_std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def summarize_evaluations(
    evaluations: Sequence[SeedEvaluation],
    *,
    n_tasks: int,
) -> dict[str, object]:
    """Aggregate measured HumanEval seed rows without inventing missing values."""

    if not evaluations:
        raise ValueError("at least one seed evaluation is required")
    if any(item.n_tasks != n_tasks for item in evaluations):
        raise ValueError("all seed evaluations must use the configured n_tasks")

    a_values = [item.condition_a_ensemble_auroc for item in evaluations]
    b_values = [item.condition_b_ensemble_auroc for item in evaluations]
    a_mean = _mean(a_values)
    b_mean = _mean(b_values)
    per_a: dict[str, list[float]] = {}
    per_b: dict[str, list[float]] = {}
    for item in evaluations:
        for name, value in item.condition_a_per_verifier_auroc.items():
            per_a.setdefault(name, []).append(float(value))
        for name, value in item.condition_b_per_verifier_auroc.items():
            per_b.setdefault(name, []).append(float(value))

    vanilla = [item.vanilla_pass_at_1 for item in evaluations]
    ranked_a = [item.condition_a_ranked_pass_at_1 for item in evaluations]
    ranked_b = [item.condition_b_ranked_pass_at_1 for item in evaluations]
    return {
        "condition_a_production_auroc_mean": a_mean,
        "condition_a_production_auroc_std": _population_std(a_values),
        "condition_b_architecture_only_auroc_mean": b_mean,
        "condition_b_architecture_only_auroc_std": _population_std(b_values),
        "learning_contribution": a_mean - b_mean,
        "per_verifier_condition_a_auroc": dict(sorted(per_a.items())),
        "per_verifier_condition_b_auroc": dict(sorted(per_b.items())),
        "pass_at_1": {
            "vanilla_mean": _mean(vanilla),
            "condition_a_ranked_mean": _mean(ranked_a),
            "condition_b_ranked_mean": _mean(ranked_b),
        },
        "ranking_lift": {
            "condition_a_vs_vanilla_mean": _mean(ranked_a) - _mean(vanilla),
            "condition_b_vs_vanilla_mean": _mean(ranked_b) - _mean(vanilla),
        },
        "candidate_execution_summary": {
            "n_labeled_candidates": sum(item.n_candidates for item in evaluations),
            "n_tasks": n_tasks,
            "n_seeds": len(evaluations),
        },
        "per_seed_results": [item.as_dict() for item in evaluations],
        "scorer_or_generator_model_paths_used": sorted(
            {item.scorer_or_generator_model_path for item in evaluations}
        ),
    }


def _reproducibility_checksum(
    *,
    state_files: Sequence[dict[str, object]],
    model_specs: dict[str, object],
    seeds: Sequence[int],
    n_tasks: int,
) -> str:
    payload = {
        "corpus": CORPUS,
        "state_files": list(state_files),
        "model_specs": model_specs,
        "random_seeds_used": list(seeds),
        "n_tasks": n_tasks,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
) -> dict[str, object]:
    return {
        "artifact": "experiment_2839_humaneval_dual_condition_v3",
        "schema": "carnot.humaneval_dual_condition_v3",
        "corpus": CORPUS,
        "n_tasks": config.n_tasks,
        "n_problems": config.n_tasks,
        "n_seeds": len(config.random_seeds),
        "random_seeds_used": list(config.random_seeds),
        "fr11_state_files": list(state_files),
        "state_files_restored_sha_match": state_files_restored_sha_match(
            config.repo_root, state_files
        ),
        "model_specs": model_specs,
        "preconditions_checked": [check.as_dict() for check in checks],
        "duration_s": duration_s,
        "reproducibility_checksum": _reproducibility_checksum(
            state_files=state_files,
            model_specs=model_specs,
            seeds=config.random_seeds,
            n_tasks=config.n_tasks,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "ground_truth_source": "official HumanEval check() execution",
    }


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
) -> dict[str, object]:
    failed = [check for check in checks if not check.available]
    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        state_files=state_files,
        checks=checks,
        model_specs=model_specs,
    )
    artifact.update(
        {
            "honest_verdict": _blocked_verdict(checks) or "blocked_unknown_resource",
            "blocked_resources": [check.resource for check in failed],
            "condition_a_production_auroc_mean": None,
            "condition_a_production_auroc_std": None,
            "condition_b_architecture_only_auroc_mean": None,
            "condition_b_architecture_only_auroc_std": None,
            "learning_contribution": None,
            "per_verifier_condition_a_auroc": {},
            "per_verifier_condition_b_auroc": {},
            "pass_at_1": None,
            "ranking_lift": None,
            "candidate_execution_summary": {
                "n_labeled_candidates": 0,
                "n_tasks": config.n_tasks,
                "n_seeds": len(config.random_seeds),
            },
            "per_seed_results": [],
            "methodology_note": (
                "Blocked before HumanEval candidate generation, execution, or scoring "
                "because required resources were missing: "
                + ", ".join(check.resource for check in failed)
                + ". No AUROC, pass@1, ranking, candidate, or verifier metrics were inferred."
            ),
        }
    )
    return artifact


def _success_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    evaluations: Sequence[SeedEvaluation],
) -> dict[str, object]:
    summary = summarize_evaluations(evaluations, n_tasks=config.n_tasks)
    model_specs = {
        **model_specs,
        "scorer_or_generator_model_paths_used": summary["scorer_or_generator_model_paths_used"],
    }
    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        state_files=state_files,
        checks=checks,
        model_specs=model_specs,
    )
    artifact.update(
        {
            "honest_verdict": (
                "complete: HumanEval full dual-condition v3 measured from official "
                "candidate labels with mandated SOTA GGUF in the scorer/generator path"
            ),
            **{
                key: value
                for key, value in summary.items()
                if key != "scorer_or_generator_model_paths_used"
            },
            "methodology_note": (
                "Condition A scores use production FR-11 state. Condition B scores "
                "must use the same HumanEval candidates after a non-destructive "
                "architecture-only state reset and Python restart. Candidate labels "
                "come from official HumanEval check() tests executed in the safe harness."
            ),
        }
    )
    return artifact


def default_live_measurement_runner(
    _config: ExperimentConfig,
    _state_files: Sequence[dict[str, object]],
    _model_specs: dict[str, object],
) -> Sequence[SeedEvaluation]:
    raise LiveHumanEvalMeasurementUnavailable(
        "live HumanEval candidate generation/scoring backend is not configured in this process"
    )


def write_artifact(results_dir: Path, artifact: dict[str, object]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / OUTPUT_FILENAME).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    precondition_probe: Callable[
        [ExperimentConfig, Sequence[dict[str, object]], dict[str, object]],
        list[PreconditionCheck],
    ] = probe_preconditions,
    measurement_runner: Callable[
        [ExperimentConfig, Sequence[dict[str, object]], dict[str, object]],
        Sequence[SeedEvaluation],
    ] = default_live_measurement_runner,
    write: bool = True,
) -> dict[str, object]:
    """Run Exp 2839 or write a blocked artifact before candidate work."""

    config = config or ExperimentConfig()
    start = config.start_time()
    state_files = discover_fr11_state_files(config.repo_root)
    preflight = load_exp2836_preflight(config.preflight_path())
    model_specs = model_specs_from_exp2836(preflight)
    checks = precondition_probe(config, state_files, model_specs)

    if _blocked_verdict(checks) is not None:
        artifact = _blocked_artifact(
            config=config,
            duration_s=config.clock() - start,
            state_files=state_files,
            checks=checks,
            model_specs=model_specs,
        )
    else:
        try:
            evaluations = list(measurement_runner(config, state_files, model_specs))
        except LiveHumanEvalMeasurementUnavailable:
            checks = [*checks, PreconditionCheck("live_backend", False, "backend unavailable")]
            artifact = _blocked_artifact(
                config=config,
                duration_s=config.clock() - start,
                state_files=state_files,
                checks=checks,
                model_specs=model_specs,
            )
        else:
            artifact = _success_artifact(
                config=config,
                duration_s=config.clock() - start,
                state_files=state_files,
                checks=checks,
                model_specs=model_specs,
                evaluations=evaluations,
            )

    if write:
        write_artifact(config.output_dir(), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--n-tasks", type=int, default=DEFAULT_N_TASKS)
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root)
    run_experiment(
        ExperimentConfig(
            repo_root=repo_root,
            results_dir=Path(args.results_dir) if args.results_dir else repo_root / "results",
            n_tasks=args.n_tasks,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
