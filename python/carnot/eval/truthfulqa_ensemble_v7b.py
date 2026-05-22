"""Exp 2831 TruthfulQA-generation dual-memory evaluator.

This module is a provenance-first harness for the honest rerun of the retired
Exp 2823 TruthfulQA artifact. It can summarize a completed five-seed
dual-condition measurement, but the default entrypoint first checks the live
resources required to make that measurement real: CUDA, HuggingFace
TruthfulQA generation split access, Qwen3.6 GGUF cache, FR-11 state files, and
BLEURT-base-128 availability. Missing resources produce a terminal
``blocked_*`` artifact with null metrics rather than inferred AUROC values.

Spec traces: REQ-VERIFY-2831, SCENARIO-VERIFY-2831,
SCENARIO-VERIFY-2831-LIVE.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


OUTPUT_FILENAME = "experiment_2831_truthfulqa_ensemble_eval.json"
CORPUS = "TruthfulQA-generation"
SCORING_METHOD = "BLEURT-base-128, threshold tuned on 50-Q held-out"
DEFAULT_RANDOM_SEEDS = (42, 137, 271, 314, 1729)
DEFAULT_N_QUESTIONS = 200
DEFAULT_SAMPLE_SEED = 42
DEFAULT_CALIBRATION_SIZE = 50
GPT3_MC1_APPROX = 0.28
MODEL_NAME = "Qwen3.6-35B-A3B-GGUF"
MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
MODEL_QUANT = "Q4_K_M"
BLEURT_MODEL_NAME = "BLEURT-base-128"
BLEURT_HF_ID = "Elron/bleurt-base-128"
REPO_ROOT = Path(__file__).resolve().parents[3]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix separates complete, blocked, and failed runs.",
    "corpus": "Identifies the benchmark split so results are not cross-corpus mixed.",
    "n_questions": "Sample size must be 200 to match the planned TruthfulQA subset.",
    "n_seeds": "Five adversarial replication seeds expose seed-sensitive lifts.",
    "condition_a_production_auroc_mean": "Production FR-11 verifier scoring quality.",
    "condition_a_production_auroc_std": "Replication noise for production scoring.",
    "condition_b_architecture_only_auroc_mean": "Architecture-only verifier baseline.",
    "condition_b_architecture_only_auroc_std": "Replication noise for architecture-only.",
    "learning_contribution": (
        "= A - B. On TruthfulQA the contribution is expected to be near zero "
        "(FR-11 state is FoVer-derived); a positive contribution means "
        "FoVer-derived rules transferred."
    ),
    "per_verifier_condition_a_auroc": "Shows which production verifiers transfer.",
    "per_verifier_condition_b_auroc": (
        "KEY finding is which verifiers transfer from FoVer-math to TruthfulQA-factual."
    ),
    "scoring_method": "Records BLEURT rather than a closed-weight judge.",
    "bleurt_threshold": "The binary label boundary must come from held-out calibration.",
    "random_seeds_used": "Determinism is required for replication.",
    "reproducibility_checksum": "Content hash catches corpus, model, or state drift.",
    "model_specs": "Names the local open-weight model and scoring target.",
    "duration_s": "Real compute takes wall-time; expected >= 1800s. Sleep-padding forbidden.",
    "preconditions_checked": "Records exactly which resources were verified before inference.",
    "fr11_state_files": "Names every FR-11 state file that Condition B must reset.",
    "state_files_restored_sha_match": "Non-destructive reset proof.",
    "methodology_note": "Honest interpretation of measured or blocked state.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check performed before live TruthfulQA inference."""

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
    """One completed seed of the Exp 2831 dual-memory measurement."""

    seed: int
    condition_a_ensemble_auroc: float
    condition_b_ensemble_auroc: float
    condition_a_per_verifier: dict[str, float]
    condition_b_per_verifier: dict[str, float]
    bleurt_threshold: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2831.

    Tests inject deterministic clocks and measured seed rows. Production keeps
    the default clock and blocks unless a real live-GPU backend is supplied.
    """

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    random_seeds: tuple[int, ...] = DEFAULT_RANDOM_SEEDS
    n_questions: int = DEFAULT_N_QUESTIONS
    sample_seed: int = DEFAULT_SAMPLE_SEED
    calibration_size: int = DEFAULT_CALIBRATION_SIZE
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    published_bleurt_verifier_comparators: tuple[Mapping[str, object], ...] = ()

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


class LiveMeasurementUnavailable(RuntimeError):
    """Raised when no real live-GPU TruthfulQA backend is attached."""


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
    larger scores mean "more likely incorrect."
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


def select_truthfulqa_indices(
    total_rows: int,
    n_questions: int = DEFAULT_N_QUESTIONS,
    calibration_size: int = DEFAULT_CALIBRATION_SIZE,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> dict[str, list[int]]:
    """Select disjoint TruthfulQA test and BLEURT-calibration row indices."""

    needed = n_questions + calibration_size
    if total_rows < needed:
        raise ValueError(
            f"not enough rows for {n_questions} test and {calibration_size} calibration examples"
        )
    selected = random.Random(seed).sample(range(total_rows), needed)
    return {
        "test_indices": selected[:n_questions],
        "calibration_indices": selected[n_questions:],
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def summarize_measurements(measurements: Sequence[SeedMeasurement]) -> dict[str, object]:
    """Summarize completed dual-memory measurements across seeds."""

    if not measurements:
        raise ValueError("at least one seed measurement is required")

    condition_a_auroc = [item.condition_a_ensemble_auroc for item in measurements]
    condition_b_auroc = [item.condition_b_ensemble_auroc for item in measurements]
    thresholds = [item.bleurt_threshold for item in measurements]

    per_a: dict[str, list[float]] = {}
    per_b: dict[str, list[float]] = {}
    for item in measurements:
        for name, value in item.condition_a_per_verifier.items():
            per_a.setdefault(name, []).append(value)
        for name, value in item.condition_b_per_verifier.items():
            per_b.setdefault(name, []).append(value)

    mean_a = _mean(condition_a_auroc)
    mean_b = _mean(condition_b_auroc)
    return {
        "condition_a_production_auroc_mean": mean_a,
        "condition_a_production_auroc_std": _std(condition_a_auroc),
        "condition_b_architecture_only_auroc_mean": mean_b,
        "condition_b_architecture_only_auroc_std": _std(condition_b_auroc),
        "learning_contribution": mean_a - mean_b,
        "per_verifier_condition_a_auroc": per_a,
        "per_verifier_condition_b_auroc": per_b,
        "bleurt_threshold": _mean(thresholds),
        "results_by_seed": [
            {
                "seed": item.seed,
                "condition_a_production_auroc": item.condition_a_ensemble_auroc,
                "condition_b_architecture_only_auroc": item.condition_b_ensemble_auroc,
                "bleurt_threshold": item.bleurt_threshold,
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
        "verifier": "ensemble_v7b",
        "scoring_model": BLEURT_MODEL_NAME,
    }


def build_reproducibility_checksum(
    *,
    seeds: Sequence[int],
    n_questions: int,
    sample_seed: int,
    calibration_size: int,
    state_files: Sequence[dict[str, object]],
    model_specs: dict[str, object],
    scoring_method: str,
) -> str:
    payload = {
        "corpus": CORPUS,
        "n_questions": n_questions,
        "test_sample_seed": sample_seed,
        "calibration_size": calibration_size,
        "random_seeds_used": list(seeds),
        "fr11_state_files": list(state_files),
        "model_specs": model_specs,
        "scoring_method": scoring_method,
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
    """Check all Exp 2831 live resources before any inference attempt."""

    checks = [
        _cuda_check(),
        _hf_truthfulqa_check(),
        _qwen_cache_check(config.repo_root),
        PreconditionCheck(
            "fr11_state_files",
            bool(state_files),
            f"{len(state_files)} FR-11 state files discovered",
        ),
        _bleurt_check(config.repo_root),
    ]
    return checks


def _cuda_check() -> PreconditionCheck:
    if "torch" not in sys.modules:
        script = """
import json
try:
    import torch
    available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    detail = f"torch import ok; cuda_available={available}; device_count={torch.cuda.device_count()}"
except Exception as exc:
    available = False
    detail = f"torch import failed: {exc!r}"
print(json.dumps({"available": available, "detail": detail}))
"""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
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
            return PreconditionCheck("cuda", False, f"invalid JSON probe output: {proc.stdout[:200]}")
        return PreconditionCheck(
            "cuda",
            bool(payload.get("available")),
            str(payload.get("detail", "")),
        )
    try:
        import torch  # type: ignore[import-not-found]

        available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        detail = (
            f"torch import ok; cuda_available={available}; device_count={torch.cuda.device_count()}"
        )
    except Exception as exc:
        available = False
        detail = f"torch import failed: {exc!r}"
    return PreconditionCheck("cuda", available, detail)


def _hf_truthfulqa_check() -> PreconditionCheck:
    if importlib.util.find_spec("datasets") is None:
        return PreconditionCheck(
            "hf_truthfulqa_generation", False, "datasets package is not installed"
        )
    if "datasets" not in sys.modules:
        script = """
import json
try:
    from datasets import load_dataset
    rows = load_dataset("truthful_qa", "generation", split="validation[:1]")
    available = len(rows) > 0
    detail = f"loaded truthful_qa generation validation[:1], n={len(rows)}"
except Exception as exc:
    available = False
    detail = f"TruthfulQA load failed: {exc!r}"
print(json.dumps({"available": available, "detail": detail}))
"""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return PreconditionCheck(
                "hf_truthfulqa_generation",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        if proc.returncode != 0:
            return PreconditionCheck(
                "hf_truthfulqa_generation",
                False,
                (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip(),
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return PreconditionCheck(
                "hf_truthfulqa_generation",
                False,
                f"invalid JSON probe output: {proc.stdout[:200]}",
            )
        return PreconditionCheck(
            "hf_truthfulqa_generation",
            bool(payload.get("available")),
            str(payload.get("detail", "")),
        )
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        rows = load_dataset("truthful_qa", "generation", split="validation[:1]")
        available = len(rows) > 0
        detail = f"loaded truthful_qa generation validation[:1], n={len(rows)}"
    except Exception as exc:
        available = False
        detail = f"TruthfulQA load failed: {exc!r}"
    return PreconditionCheck("hf_truthfulqa_generation", available, detail)


def _qwen_cache_check(repo_root: Path) -> PreconditionCheck:
    specs = model_specs(repo_root)
    if specs["cache_complete"]:
        return PreconditionCheck("qwen36_gguf_cache", True, f"resolved {specs['resolved_gguf']}")
    return PreconditionCheck(
        "qwen36_gguf_cache",
        False,
        "no real .gguf found in HF snapshots or project models directory",
    )


def _bleurt_check(repo_root: Path) -> PreconditionCheck:
    if importlib.util.find_spec("bleurt") is not None:
        return PreconditionCheck("bleurt_base_128", True, "bleurt package is importable")
    local_cache = repo_root / "models" / "bleurt-base-128"
    if local_cache.exists():
        return PreconditionCheck("bleurt_base_128", True, f"resolved {local_cache}")
    if importlib.util.find_spec("huggingface_hub") is not None:
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            info = HfApi().model_info(BLEURT_HF_ID, files_metadata=False)
        except Exception as exc:
            return PreconditionCheck(
                "bleurt_base_128",
                False,
                f"BLEURT-base-128 HF cacheability check failed: {exc!r}",
            )
        return PreconditionCheck(
            "bleurt_base_128",
            True,
            f"HF model hub reachable for {BLEURT_HF_ID}; sha={getattr(info, 'sha', 'unknown')}",
        )
    return PreconditionCheck(
        "bleurt_base_128",
        False,
        "bleurt package missing, models/bleurt-base-128 was not found, and huggingface_hub is unavailable",
    )


def _blocked_verdict(checks: Iterable[PreconditionCheck]) -> str:
    verdict_by_resource = {
        "cuda": "blocked_cuda_unavailable",
        "hf_truthfulqa_generation": "blocked_hf_truthfulqa_generation_unavailable",
        "qwen36_gguf_cache": "blocked_model_not_cached_qwen36_35b_a3b_gguf",
        "fr11_state_files": "blocked_fr11_state_files_missing",
        "bleurt_base_128": "blocked_bleurt_base_128_unavailable",
        "live_backend": "blocked_live_qwen36_backend_unavailable",
    }
    for check in checks:
        if not check.available:
            return verdict_by_resource.get(check.resource, f"blocked_{check.resource}")
    return "blocked_unknown_resource"


def _published_bleurt_verifier_comparators(config: ExperimentConfig) -> list[dict[str, object]]:
    return [dict(item) for item in config.published_bleurt_verifier_comparators]


def _baseline_comparison(
    summary: Mapping[str, object],
    comparators: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    production = float(summary["condition_a_production_auroc_mean"])
    comparator_values = [
        float(item["auroc"])
        for item in comparators
        if isinstance(item.get("auroc"), (int, float))
    ]
    comparator_best = max(comparator_values) if comparator_values else None
    return {
        "gpt3_mc1_approx": GPT3_MC1_APPROX,
        "production_minus_gpt3_mc1_approx": production - GPT3_MC1_APPROX,
        "published_bleurt_verifier_best_auroc": comparator_best,
        "production_minus_bleurt_comparator_best": (
            None if comparator_best is None else production - comparator_best
        ),
        "comparison_note": (
            "GPT-3 MC1 is an external accuracy reference, not an AUROC-matched verifier "
            "baseline; BLEURT verifier comparators are AUROC-matched when supplied."
        ),
    }


def _null_baseline_comparison() -> dict[str, object]:
    return {
        "gpt3_mc1_approx": GPT3_MC1_APPROX,
        "production_minus_gpt3_mc1_approx": None,
        "published_bleurt_verifier_best_auroc": None,
        "production_minus_bleurt_comparator_best": None,
        "comparison_note": (
            "No production AUROC was measured, so baseline deltas are intentionally null."
        ),
    }


def _field_provenance(status: str) -> dict[str, dict[str, str]]:
    return {
        field: {
            "principle": principle,
            "satisfied_by": status,
        }
        for field, principle in FIELD_PRINCIPLES.items()
    }


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
        "n_questions": config.n_questions,
        "n_seeds": len(config.random_seeds),
        "test_sample_seed": config.sample_seed,
        "calibration_size": config.calibration_size,
        "scoring_method": SCORING_METHOD,
        "random_seeds_used": list(config.random_seeds),
        "reproducibility_checksum": build_reproducibility_checksum(
            seeds=config.random_seeds,
            n_questions=config.n_questions,
            sample_seed=config.sample_seed,
            calibration_size=config.calibration_size,
            state_files=state_files,
            model_specs=specs,
            scoring_method=SCORING_METHOD,
        ),
        "model_specs": specs,
        "duration_s": duration_s,
        "preconditions_checked": [check.as_dict() for check in checks],
        "fr11_state_files": list(state_files),
        "state_files_restored_sha_match": state_files_restored_sha_match(
            config.repo_root, state_files
        ),
        "ground_truth_source": (
            "BLEURT-base-128 semantic similarity against TruthfulQA best_answer; "
            "50-question held-out threshold calibration; no closed-weight judge"
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
            "bleurt_threshold": None,
            "published_bleurt_verifier_comparators": [],
            "baseline_comparison": _null_baseline_comparison(),
            "methodology_note": (
                "Blocked before live TruthfulQA inference, BLEURT scoring, or ensemble "
                "measurement because required resources were missing: "
                + ", ".join(check.resource for check in failed)
                + ". No AUROC, threshold, candidate, label, or per-verifier metrics "
                "were inferred."
            ),
            "field_provenance": _field_provenance("blocked before measurement"),
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
    comparators = _published_bleurt_verifier_comparators(config)
    artifact.update(
        {
            "honest_verdict": (
                "complete: TruthfulQA-generation dual-memory verifier ensemble measured "
                f"over {config.n_questions} questions and {len(measurements)} seeds"
            ),
            **summary,
            "published_bleurt_verifier_comparators": comparators,
            "baseline_comparison": _baseline_comparison(summary, comparators),
            "methodology_note": (
                "Condition A used full FR-11 state. Condition B is expected to use the "
                "same TruthfulQA question/candidate pairs after a non-destructive FR-11 "
                "state reset and Python restart. Labels come from BLEURT-base-128 "
                "similarity to best_answer with a held-out calibration threshold. "
                "This artifact reports only measured runner output."
            ),
            "field_provenance": _field_provenance("measured live run"),
        }
    )
    return artifact


def default_live_measurement_runner(
    _config: ExperimentConfig,
    _state_files: Sequence[dict[str, object]],
) -> list[SeedMeasurement]:
    raise LiveMeasurementUnavailable(
        "live Qwen3.6 TruthfulQA generation/BLEURT/scoring backend is not configured"
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
    """Run Exp 2831 or write an honest blocked artifact before inference."""

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
