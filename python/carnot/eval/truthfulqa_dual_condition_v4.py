"""Exp 2840 TruthfulQA dual-condition v4 evaluator.

This runner is the post-2836 TruthfulQA retry. It is intentionally conservative:
it checks every local resource before candidate generation or scoring and writes
a terminal blocked artifact when the host cannot produce real local evidence.
Successful runs must use a reproducible local scorer, never a closed-weight LLM
judge, and must not reuse the retired fabricated Exp 2823 metrics.

Spec: REQ-VERIFY-2840,
      SCENARIO-VERIFY-2840-BLOCKED,
      SCENARIO-VERIFY-2840-LIVE.
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


OUTPUT_FILENAME = "experiment_2840_truthfulqa_dual_condition_v4.json"
EXP2836_FILENAME = "experiment_2836_sota_runtime_preflight.json"
CORPUS = "TruthfulQA-generation"
DEFAULT_RANDOM_SEEDS = (42, 137, 271, 314, 1729)
DEFAULT_N_QUESTIONS = 200
DEFAULT_CALIBRATION_SIZE = 50
RUN_DATE = "20260522"
PRIMARY_SOTA_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LEGACY_CPU_SMOKE_ONLY = ("Qwen3.5-0.8B", "gemma-4-E4B-it")
REPO_ROOT = Path(__file__).resolve().parents[3]

FIELD_PRINCIPLES = {
    "honest_verdict": 'MUST start with "complete:" / "success:" or "blocked_".',
    "n_questions": "TruthfulQA planned sample size is 200.",
    "local_scorer": "No closed-weight judge; scorer must be reproducible.",
    "condition_a_production_auroc_mean": "Production AUROC measured from local labels.",
    "condition_b_architecture_only_auroc_mean": (
        "Architecture-only AUROC measured from local labels."
    ),
    "learning_contribution": "A - B transfer contribution.",
    "retired_exp2823_not_used": "Fabricated prior artifact must not contaminate results.",
    "model_specs": "Mandated SOTA GGUF recorded.",
    "preconditions_checked": "Explains blocks honestly.",
    "duration_s": "Real compute wall-time; no sleep padding.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before TruthfulQA candidate work."""

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
    """Measured TruthfulQA outcomes for one seed across both memory conditions."""

    seed: int
    n_questions: int
    n_candidates: int
    condition_a_ensemble_auroc: float
    condition_b_ensemble_auroc: float
    condition_a_per_verifier_auroc: dict[str, float]
    condition_b_per_verifier_auroc: dict[str, float]
    scorer_name: str
    scorer_version: str
    scorer_threshold: float
    calibration_size: int
    scorer_or_generator_model_path: str
    candidate_label_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "n_questions": self.n_questions,
            "n_candidates": self.n_candidates,
            "condition_a_production_auroc": self.condition_a_ensemble_auroc,
            "condition_b_architecture_only_auroc": self.condition_b_ensemble_auroc,
            "condition_a_per_verifier_auroc": dict(self.condition_a_per_verifier_auroc),
            "condition_b_per_verifier_auroc": dict(self.condition_b_per_verifier_auroc),
            "scorer_name": self.scorer_name,
            "scorer_version": self.scorer_version,
            "scorer_threshold": self.scorer_threshold,
            "calibration_size": self.calibration_size,
            "scorer_or_generator_model_path": self.scorer_or_generator_model_path,
            "candidate_label_sha256": self.candidate_label_sha256,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 2840."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    exp2836_path: Path | None = None
    random_seeds: tuple[int, ...] = DEFAULT_RANDOM_SEEDS
    n_questions: int = DEFAULT_N_QUESTIONS
    calibration_size: int = DEFAULT_CALIBRATION_SIZE
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


class LiveTruthfulQAMeasurementUnavailable(RuntimeError):
    """Raised when no real TruthfulQA generation/scoring backend is attached."""


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
    """Normalize Exp 2836 runtime evidence for the TruthfulQA artifact."""

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


def _truthfulqa_dataset_check(
    selected_python: str,
    repo_root: Path,
    *,
    n_questions: int,
    calibration_size: int,
    command_runner: CommandRunner = subprocess.run,
    timeout_s: int = 30,
) -> PreconditionCheck:
    script = f"""
import importlib.util
import json
if importlib.util.find_spec("datasets") is None:
    print(json.dumps({{"available": False, "detail": "datasets package is not installed"}}))
else:
    try:
        from datasets import load_dataset
        rows = load_dataset("truthful_qa", "generation", split="validation")
        required = {{"question", "best_answer"}}
        first = rows[0] if len(rows) else {{}}
        has_required_fields = required.issubset(set(first))
        needed = {n_questions + calibration_size}
        available = bool(len(rows) >= needed and has_required_fields)
        detail = f"loaded truthful_qa generation validation, n={{len(rows)}}, needed={{needed}}, has_required_fields={{has_required_fields}}"
    except Exception as exc:
        available = False
        detail = f"{{type(exc).__name__}}: {{exc}}"
    print(json.dumps({{"available": available, "detail": detail}}))
"""
    return _run_json_probe(
        selected_python=selected_python,
        repo_root=repo_root,
        script=script,
        resource="truthfulqa_generation_split",
        command_runner=command_runner,
        timeout_s=timeout_s,
    )


def _local_scorer_check(
    selected_python: str,
    repo_root: Path,
    *,
    command_runner: CommandRunner = subprocess.run,
    timeout_s: int = 30,
) -> PreconditionCheck:
    script = """
import importlib.util
import json

metadata = None
if importlib.util.find_spec("bleurt") is not None:
    try:
        import bleurt
        metadata = {
            "name": "BLEURT-base-128",
            "version": getattr(bleurt, "__version__", "unknown"),
            "reference": "BLEURT-base-128 local package",
            "closed_weight_judge": False,
        }
    except Exception:
        metadata = None

if metadata is None and importlib.util.find_spec("sklearn") is not None:
    try:
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer().fit(["reference answer", "candidate answer"])
        score = float(cosine_similarity(vectorizer.transform(["reference answer"]), vectorizer.transform(["candidate answer"]))[0, 0])
        metadata = {
            "name": "sklearn_tfidf_cosine",
            "version": sklearn.__version__,
            "reference": "openspec/capabilities/pipeline/spec.md",
            "closed_weight_judge": False,
            "smoke_score": score,
        }
    except Exception:
        metadata = None

if metadata is None:
    local_cache = "models/bleurt-base-128"
    print(json.dumps({"available": False, "detail": "no bleurt package, BLEURT cache, or sklearn TF-IDF scorer available"}))
else:
    print(json.dumps({"available": True, "detail": json.dumps(metadata, sort_keys=True)}))
"""
    return _run_json_probe(
        selected_python=selected_python,
        repo_root=repo_root,
        script=script,
        resource="local_scorer",
        command_runner=command_runner,
        timeout_s=timeout_s,
    )


def _retired_exp2823_check(repo_root: Path) -> PreconditionCheck:
    manifest_path = repo_root / "ops" / "exclusion_manifest.yaml"
    fabricated_path = (
        repo_root / "legacy" / "fabricated" / "experiment_2823_truthfulqa_ensemble_eval.json"
    )
    if not manifest_path.is_file():
        return PreconditionCheck("retired_exp2823", False, "ops/exclusion_manifest.yaml missing")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    has_2823 = "2823" in manifest_text
    has_retirement_reason = "fabricated" in manifest_text.lower() or "retired" in manifest_text.lower()
    if not (has_2823 and has_retirement_reason):
        return PreconditionCheck(
            "retired_exp2823",
            False,
            "exp2823 retirement/fabrication evidence missing from exclusion manifest",
        )
    if not fabricated_path.is_file():
        return PreconditionCheck(
            "retired_exp2823",
            False,
            "legacy/fabricated/experiment_2823_truthfulqa_ensemble_eval.json missing",
        )
    return PreconditionCheck(
        "retired_exp2823",
        True,
        "exclusion manifest retires exp2823; legacy artifact present but not parsed",
    )


def probe_preconditions(
    config: ExperimentConfig,
    state_files: Sequence[dict[str, object]],
    model_specs: dict[str, object],
    *,
    command_runner: CommandRunner = subprocess.run,
) -> list[PreconditionCheck]:
    """Check all Exp 2840 live resources before TruthfulQA scoring."""

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
            _truthfulqa_dataset_check(
                selected_python,
                config.repo_root,
                n_questions=config.n_questions,
                calibration_size=config.calibration_size,
                command_runner=command_runner,
                timeout_s=config.probe_timeout_s,
            ),
            _local_scorer_check(
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
            _retired_exp2823_check(config.repo_root),
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
        "truthfulqa_generation_split": "blocked_truthfulqa_generation_split",
        "local_scorer": "blocked_local_scorer",
        "fr11_state_files": "blocked_fr11_state_files",
        "retired_exp2823": "blocked_exp2823_retirement_evidence",
        "live_backend": "blocked_live_truthfulqa_backend_unavailable",
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


def _local_scorer_from_checks(checks: Sequence[PreconditionCheck]) -> dict[str, object]:
    scorer_check = next((check for check in checks if check.resource == "local_scorer"), None)
    if scorer_check is None:
        return {
            "available": False,
            "name": None,
            "version": None,
            "closed_weight_judge": None,
            "detail": "local_scorer precondition was not checked",
        }
    try:
        payload = json.loads(scorer_check.detail)
    except json.JSONDecodeError:
        return {
            "available": scorer_check.available,
            "name": None,
            "version": None,
            "closed_weight_judge": None,
            "detail": scorer_check.detail,
        }
    if not isinstance(payload, dict):
        return {
            "available": scorer_check.available,
            "name": None,
            "version": None,
            "closed_weight_judge": None,
            "detail": scorer_check.detail,
        }
    payload = dict(payload)
    payload.pop("smoke_score", None)
    return {
        "available": scorer_check.available,
        "closed_weight_judge": bool(payload.get("closed_weight_judge", False)),
        "name": payload.get("name"),
        "reference": payload.get("reference"),
        "version": payload.get("version"),
    }


def summarize_evaluations(
    evaluations: Sequence[SeedEvaluation],
    *,
    n_questions: int,
) -> dict[str, object]:
    """Aggregate measured TruthfulQA seed rows without inventing missing values."""

    if not evaluations:
        raise ValueError("at least one seed evaluation is required")
    if any(item.n_questions != n_questions for item in evaluations):
        raise ValueError("all seed evaluations must use the configured n_questions")

    a_values = [item.condition_a_ensemble_auroc for item in evaluations]
    b_values = [item.condition_b_ensemble_auroc for item in evaluations]
    thresholds = [item.scorer_threshold for item in evaluations]
    a_mean = _mean(a_values)
    b_mean = _mean(b_values)
    per_a: dict[str, list[float]] = {}
    per_b: dict[str, list[float]] = {}
    for item in evaluations:
        for name, value in item.condition_a_per_verifier_auroc.items():
            per_a.setdefault(name, []).append(float(value))
        for name, value in item.condition_b_per_verifier_auroc.items():
            per_b.setdefault(name, []).append(float(value))

    calibration_sizes = {item.calibration_size for item in evaluations}
    calibration_size: int | str
    calibration_size = calibration_sizes.pop() if len(calibration_sizes) == 1 else "mixed"
    return {
        "condition_a_production_auroc_mean": a_mean,
        "condition_a_production_auroc_std": _population_std(a_values),
        "condition_b_architecture_only_auroc_mean": b_mean,
        "condition_b_architecture_only_auroc_std": _population_std(b_values),
        "learning_contribution": a_mean - b_mean,
        "per_verifier_condition_a_auroc": dict(sorted(per_a.items())),
        "per_verifier_condition_b_auroc": dict(sorted(per_b.items())),
        "calibration": {
            "threshold_mean": _mean(thresholds),
            "threshold_std": _population_std(thresholds),
            "calibration_size": calibration_size,
            "label_source": "local_scorer_against_best_answer",
        },
        "candidate_summary": {
            "n_labeled_candidates": sum(item.n_candidates for item in evaluations),
            "n_questions": n_questions,
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
    n_questions: int,
    calibration_size: int,
    local_scorer: dict[str, object],
) -> str:
    payload = {
        "corpus": CORPUS,
        "state_files": list(state_files),
        "model_specs": model_specs,
        "random_seeds_used": list(seeds),
        "n_questions": n_questions,
        "calibration_size": calibration_size,
        "local_scorer": local_scorer,
        "retired_exp2823_not_used": True,
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
    local_scorer = _local_scorer_from_checks(checks)
    return {
        "artifact": "experiment_2840_truthfulqa_dual_condition_v4",
        "schema": "carnot.truthfulqa_dual_condition_v4",
        "run_date": RUN_DATE,
        "corpus": CORPUS,
        "n_questions": config.n_questions,
        "n_seeds": len(config.random_seeds),
        "random_seeds_used": list(config.random_seeds),
        "calibration_size": config.calibration_size,
        "local_scorer": local_scorer,
        "retired_exp2823_not_used": True,
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
            n_questions=config.n_questions,
            calibration_size=config.calibration_size,
            local_scorer=local_scorer,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "ground_truth_source": (
            "local semantic scorer against TruthfulQA best_answer/reference; "
            "no closed-weight judge"
        ),
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
            "calibration": None,
            "candidate_summary": {
                "n_labeled_candidates": 0,
                "n_questions": config.n_questions,
                "n_seeds": len(config.random_seeds),
            },
            "per_seed_results": [],
            "methodology_note": (
                "Blocked before TruthfulQA candidate generation, local scoring, or verifier "
                "measurement because required resources were missing: "
                + ", ".join(check.resource for check in failed)
                + ". No AUROC, calibration threshold, candidate label, or per-verifier "
                "metric was inferred. The retired Exp 2823 artifact was not used as data."
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
    summary = summarize_evaluations(evaluations, n_questions=config.n_questions)
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
                "complete: TruthfulQA dual-condition v4 measured with local scorer "
                "and mandated SOTA GGUF evidence"
            ),
            **{
                key: value
                for key, value in summary.items()
                if key != "scorer_or_generator_model_paths_used"
            },
            "methodology_note": (
                "Condition A scores use production FR-11 state. Condition B scores "
                "must use the same TruthfulQA question/candidate/label rows after a "
                "non-destructive architecture-only state reset and Python restart. "
                "Labels come from a reproducible local scorer against best_answer or "
                "reference text; closed-weight judges and retired Exp 2823 metrics are excluded."
            ),
        }
    )
    return artifact


def default_live_measurement_runner(
    _config: ExperimentConfig,
    _state_files: Sequence[dict[str, object]],
    _model_specs: dict[str, object],
) -> Sequence[SeedEvaluation]:
    raise LiveTruthfulQAMeasurementUnavailable(
        "live TruthfulQA candidate generation/scoring backend is not configured in this process"
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
    """Run Exp 2840 or write a blocked artifact before TruthfulQA work."""

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
        except LiveTruthfulQAMeasurementUnavailable:
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
    parser.add_argument("--n-questions", type=int, default=DEFAULT_N_QUESTIONS)
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
