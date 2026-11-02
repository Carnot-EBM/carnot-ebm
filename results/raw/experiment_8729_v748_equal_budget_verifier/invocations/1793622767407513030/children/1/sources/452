"""Exp 2829 MBPP dual-memory evaluator.

This module is deliberately strict about live-resource provenance. It can
summarize a completed five-seed MBPP verifier run, but its default entrypoint
first checks CUDA, MBPP dataset access, the Qwen3.6 GGUF cache, and FR-11 state
files. If any required resource is missing, it writes a blocked artifact with
null metrics instead of filling in plausible-looking numbers.

Spec traces: REQ-VERIFY-2829, SCENARIO-VERIFY-2829,
SCENARIO-VERIFY-2829-LIVE.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_FILENAME = "experiment_2829_mbpp_ensemble_eval.json"
CORPUS = "MBPP-sanitized-test"
DEFAULT_RANDOM_SEEDS = (42, 137, 271, 314, 1729)
DEFAULT_N_PROBLEMS = 100
MODEL_NAME = "Qwen3.6-35B-A3B-GGUF"
MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
MODEL_QUANT = "Q4_K_M"
REPO_ROOT = Path(__file__).resolve().parents[3]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix.",
    "corpus": "Identifies corpus.",
    "n_problems": "Sample size.",
    "n_seeds": "Adversarial replication.",
    "condition_a_production_auroc_mean": "Production headline.",
    "condition_a_production_auroc_std": "Replication noise.",
    "condition_b_architecture_only_auroc_mean": "Architecture-only baseline.",
    "condition_b_architecture_only_auroc_std": "Replication noise on architecture-only.",
    "learning_contribution": "= A - B.",
    "per_verifier_condition_a_auroc": "Per-verifier production AUROC.",
    "per_verifier_condition_b_auroc": "Per-verifier architecture-only AUROC.",
    "vanilla_qwen36_pass_at_1": "Baseline for model without Carnot.",
    "random_seeds_used": "Determinism.",
    "reproducibility_checksum": "Catches drift.",
    "model_specs": "Names compute target.",
    "duration_s": "Real wall-clock measurement; never padded.",
    "preconditions_checked": "Anti-fabrication.",
    "fr11_state_files": "Names state reset for condition B.",
    "state_files_restored_sha_match": "Non-destructive proof.",
    "methodology_note": "Honest interpretation.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check performed before any live MBPP inference."""

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
class SeedMeasurement:
    """One completed seed of the exp2829 dual-memory measurement."""

    seed: int
    condition_a_ensemble_auroc: float
    condition_b_ensemble_auroc: float
    condition_a_per_verifier: dict[str, float]
    condition_b_per_verifier: dict[str, float]
    vanilla_pass_at_1: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for exp2829.

    Tests inject a deterministic clock and measurement runner. Production uses
    the defaults, which record real wall time and block before measurement when
    live resources are absent.
    """

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    random_seeds: tuple[int, ...] = DEFAULT_RANDOM_SEEDS
    n_problems: int = DEFAULT_N_PROBLEMS
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


class LiveMeasurementUnavailable(RuntimeError):
    """Raised when no real live-GPU MBPP measurement backend is attached."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_fr11_state_files(repo_root: Path) -> list[dict[str, object]]:
    """Return FR-11 state file metadata without moving or mutating files."""

    patterns = (
        "results/fr11_*.json",
        "results/fr11_*.jsonl",
        "results/nexus_constraint_memory*.json",
        "results/session_memory_*/**/session_state.json",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in repo_root.glob(pattern) if path.is_file())

    state_files: list[dict[str, object]] = []
    for path in sorted(paths):
        stat = path.stat()
        state_files.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(path),
                "n_bytes": stat.st_size,
            }
        )
    return state_files


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC with average ranks for tied scores.

    Labels use the verifier convention: ``1`` is the positive/error class and
    larger scores mean "more likely incorrect." This small implementation keeps
    exp2829 independent of sklearn for artifact construction.
    """

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    n_pos = sum(1 for label in labels if int(label) == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative labels")

    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]))
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ranked):
        end = cursor + 1
        while end < len(ranked) and float(ranked[end][1]) == float(ranked[cursor][1]):
            end += 1
        avg_rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[ranked[offset][0]] = avg_rank
        cursor = end

    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if int(label) == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def summarize_measurements(measurements: Sequence[SeedMeasurement]) -> dict[str, object]:
    """Summarize completed dual-memory measurements across seeds."""

    if not measurements:
        raise ValueError("at least one seed measurement is required")

    condition_a = [item.condition_a_ensemble_auroc for item in measurements]
    condition_b = [item.condition_b_ensemble_auroc for item in measurements]
    per_a: dict[str, list[float]] = {}
    per_b: dict[str, list[float]] = {}
    for item in measurements:
        for name, value in item.condition_a_per_verifier.items():
            per_a.setdefault(name, []).append(value)
        for name, value in item.condition_b_per_verifier.items():
            per_b.setdefault(name, []).append(value)
    mean_a = _mean(condition_a)
    mean_b = _mean(condition_b)
    return {
        "condition_a_production_auroc_mean": mean_a,
        "condition_a_production_auroc_std": _std(condition_a),
        "condition_b_architecture_only_auroc_mean": mean_b,
        "condition_b_architecture_only_auroc_std": _std(condition_b),
        "learning_contribution": mean_a - mean_b,
        "per_verifier_condition_a_auroc": per_a,
        "per_verifier_condition_b_auroc": per_b,
        "vanilla_qwen36_pass_at_1": _mean([item.vanilla_pass_at_1 for item in measurements]),
        "results_by_seed": [
            {
                "seed": item.seed,
                "condition_a_production_auroc": item.condition_a_ensemble_auroc,
                "condition_b_architecture_only_auroc": item.condition_b_ensemble_auroc,
                "vanilla_qwen36_pass_at_1": item.vanilla_pass_at_1,
            }
            for item in measurements
        ],
    }


def model_specs(repo_root: Path) -> dict[str, object]:
    cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + MODEL_HF_ID.replace("/", "--"))
    )
    revision_ref = cache_dir / "refs" / "main"
    revision_sha = (
        revision_ref.read_text(encoding="utf-8").strip() if revision_ref.exists() else None
    )
    snapshots_dir = cache_dir / "snapshots"
    gguf_files = sorted(snapshots_dir.glob("*/**/*.gguf")) if snapshots_dir.exists() else []
    project_files = sorted((repo_root / "models").glob("**/*Qwen3.6*35B*.gguf"))
    resolved_files = [path for path in (*gguf_files, *project_files) if path.is_file()]
    return {
        "name": MODEL_NAME,
        "hf_id": MODEL_HF_ID,
        "quant": MODEL_QUANT,
        "revision_sha": revision_sha,
        "cache_complete": bool(resolved_files),
        "resolved_gguf": str(resolved_files[0]) if resolved_files else None,
    }


def build_reproducibility_checksum(
    *,
    seeds: Sequence[int],
    n_problems: int,
    state_files: Sequence[dict[str, object]],
    model_specs: dict[str, object],
) -> str:
    payload = {
        "corpus": CORPUS,
        "n_problems": n_problems,
        "random_seeds_used": list(seeds),
        "fr11_state_files": list(state_files),
        "model_specs": model_specs,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_files_restored_sha_match(
    repo_root: Path, state_files: Sequence[dict[str, object]]
) -> bool:
    for item in state_files:
        path = repo_root / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            return False
    return True


def probe_preconditions(
    config: ExperimentConfig,
    state_files: Sequence[dict[str, object]],
) -> list[PreconditionCheck]:
    """Check all exp2829 live resources before any inference attempt."""

    checks = [_cuda_check(), _hf_mbpp_check(), _qwen_cache_check(config.repo_root)]
    checks.append(
        PreconditionCheck(
            "fr11_state_files",
            bool(state_files),
            f"{len(state_files)} FR-11 state files discovered",
        )
    )
    return checks


def _cuda_check() -> PreconditionCheck:
    try:
        import torch  # type: ignore[import-not-found]

        available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        detail = (
            f"torch import ok; cuda_available={available}; device_count={torch.cuda.device_count()}"
        )
    except Exception as exc:  # pragma: no cover - depends on local torch install
        available = False
        detail = f"torch import failed: {exc!r}"
    return PreconditionCheck("cuda", available, detail)


def _hf_mbpp_check() -> PreconditionCheck:
    if importlib.util.find_spec("datasets") is None:
        return PreconditionCheck("hf_mbpp", False, "datasets package is not installed")
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        rows = load_dataset("google-research-datasets/mbpp", "sanitized", split="test[:1]")
        available = len(rows) > 0
        detail = f"loaded google-research-datasets/mbpp sanitized test[:1], n={len(rows)}"
    except Exception as exc:  # pragma: no cover - depends on network/HF state
        available = False
        detail = f"MBPP load failed: {exc!r}"
    return PreconditionCheck("hf_mbpp", available, detail)


def _qwen_cache_check(repo_root: Path) -> PreconditionCheck:
    specs = model_specs(repo_root)
    if specs["cache_complete"]:
        return PreconditionCheck("qwen36_gguf_cache", True, f"resolved {specs['resolved_gguf']}")
    return PreconditionCheck(
        "qwen36_gguf_cache",
        False,
        "no real .gguf found in HF snapshots or project models directory",
    )


def _blocked_verdict(checks: Iterable[PreconditionCheck]) -> str:
    verdict_by_resource = {
        "cuda": "blocked_cuda_unavailable",
        "hf_mbpp": "blocked_hf_mbpp_unavailable",
        "qwen36_gguf_cache": "blocked_model_not_cached_qwen36_35b_a3b_gguf",
        "fr11_state_files": "blocked_fr11_state_files_missing",
        "live_backend": "blocked_live_qwen36_backend_unavailable",
    }
    for check in checks:
        if not check.available:
            return verdict_by_resource.get(check.resource, f"blocked_{check.resource}")
    return "blocked_unknown_resource"


def _base_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    specs: dict[str, object],
) -> dict[str, object]:
    return {
        "corpus": CORPUS,
        "n_problems": config.n_problems,
        "n_seeds": len(config.random_seeds),
        "random_seeds_used": list(config.random_seeds),
        "reproducibility_checksum": build_reproducibility_checksum(
            seeds=config.random_seeds,
            n_problems=config.n_problems,
            state_files=state_files,
            model_specs=specs,
        ),
        "model_specs": specs,
        "duration_s": duration_s,
        "preconditions_checked": [check.as_dict() for check in checks],
        "fr11_state_files": list(state_files),
        "state_files_restored_sha_match": state_files_restored_sha_match(
            config.repo_root, state_files
        ),
        "field_principles": FIELD_PRINCIPLES,
    }


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    specs: dict[str, object],
) -> dict[str, object]:
    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        state_files=state_files,
        checks=checks,
        specs=specs,
    )
    failed = [check for check in checks if not check.available]
    artifact.update(
        {
            "honest_verdict": _blocked_verdict(checks),
            "blocked_resources": [check.resource for check in failed],
            "condition_a_production_auroc_mean": None,
            "condition_a_production_auroc_std": None,
            "condition_b_architecture_only_auroc_mean": None,
            "condition_b_architecture_only_auroc_std": None,
            "learning_contribution": None,
            "per_verifier_condition_a_auroc": {},
            "per_verifier_condition_b_auroc": {},
            "vanilla_qwen36_pass_at_1": None,
            "methodology_note": (
                "Blocked before live MBPP inference because required resources were missing: "
                + ", ".join(check.resource for check in failed)
                + ". No AUROC, pass@1, candidate, or ensemble metrics were inferred."
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
    specs: dict[str, object],
    measurements: Sequence[SeedMeasurement],
) -> dict[str, object]:
    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        state_files=state_files,
        checks=checks,
        specs=specs,
    )
    summary = summarize_measurements(measurements)
    artifact.update(
        {
            "honest_verdict": (
                "complete: MBPP dual-memory verifier ensemble measured "
                f"over {config.n_problems} problems and {len(measurements)} seeds"
            ),
            **summary,
            "methodology_note": (
                "Condition A used full FR-11 state; Condition B is expected to come from "
                "the same candidate pairs after a non-destructive FR-11 state reset and "
                "Python restart. This artifact reports only measured runner output."
            ),
        }
    )
    return artifact


def default_live_measurement_runner(
    _config: ExperimentConfig,
    _state_files: Sequence[dict[str, object]],
) -> list[SeedMeasurement]:
    raise LiveMeasurementUnavailable(
        "live Qwen3.6 MBPP generation/scoring backend is not configured in this process"
    )


def write_artifact(results_dir: Path, artifact: dict[str, object]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / OUTPUT_FILENAME).write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    precondition_probe: Callable[
        [ExperimentConfig, Sequence[dict[str, object]]], list[PreconditionCheck]
    ] = probe_preconditions,
    measurement_runner: Callable[
        [ExperimentConfig, Sequence[dict[str, object]]], Sequence[SeedMeasurement]
    ] = default_live_measurement_runner,
    write: bool = True,
) -> dict[str, object]:
    """Run exp2829 or write an honest blocked artifact before inference."""

    config = config or ExperimentConfig()
    start = config.start_time()
    results_dir = config.output_dir()
    state_files = discover_fr11_state_files(config.repo_root)
    checks = precondition_probe(config, state_files)
    specs = model_specs(config.repo_root)

    if any(not check.available for check in checks):
        artifact = _blocked_artifact(
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
            artifact = _blocked_artifact(
                config=config,
                duration_s=config.clock() - start,
                state_files=state_files,
                checks=checks,
                specs=specs,
            )
        else:
            artifact = _success_artifact(
                config=config,
                duration_s=config.clock() - start,
                state_files=state_files,
                checks=checks,
                specs=specs,
                measurements=measurements,
            )

    if write:
        write_artifact(results_dir, artifact)
    return artifact
