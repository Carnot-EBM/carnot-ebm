"""Exp 2830 HumanEval-full dual-memory evaluator.

The module is intentionally conservative: it can summarize a completed
five-seed HumanEval verifier run, but the default entrypoint first checks the
live resources required to make that run real. Missing CUDA, missing
`openai_humaneval`, missing Qwen3.6 GGUF cache, or missing FR-11 state produces
a `blocked_*` artifact with null metrics instead of guessed pass@1 numbers.

Spec traces: REQ-VERIFY-2830, SCENARIO-VERIFY-2830,
SCENARIO-VERIFY-2830-LIVE.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


OUTPUT_FILENAME = "experiment_2830_humaneval_full_ensemble_eval.json"
CORPUS = "HumanEval-full"
DEFAULT_RANDOM_SEEDS = (42, 137, 271, 314, 1729)
DEFAULT_N_PROBLEMS = 164
MODEL_NAME = "Qwen3.6-35B-A3B-GGUF"
MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
MODEL_QUANT = "Q4_K_M"
REPO_ROOT = Path(__file__).resolve().parents[3]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix separates complete, blocked, and failed runs.",
    "corpus": "Identifies the benchmark split so results are not cross-corpus mixed.",
    "n_problems": "HumanEval-full sample size must be 164 to support benchmark claims.",
    "n_seeds": "Five adversarial replication seeds expose seed-sensitive lifts.",
    "condition_a_production_auroc_mean": "Production verifier scoring quality.",
    "condition_a_production_auroc_std": "Replication noise for production scoring.",
    "condition_b_architecture_only_auroc_mean": "Architecture-only verifier scoring baseline.",
    "condition_b_architecture_only_auroc_std": "Replication noise for architecture-only scoring.",
    "condition_a_production_pass_at_1_mean": "Production repair-pipeline pass@1.",
    "condition_a_production_pass_at_1_std": "Production pass@1 replication noise.",
    "condition_b_architecture_only_pass_at_1_mean": "Architecture-only repair-pipeline pass@1.",
    "condition_b_architecture_only_pass_at_1_std": "Architecture-only pass@1 replication noise.",
    "pass_at_1_vanilla": "Qwen3.6 baseline before Carnot correction.",
    "pass_at_1_after_carnot_correct_production": (
        "Repair-pipeline lift under both memory conditions."
    ),
    "pass_at_1_after_carnot_correct_architecture_only": (
        "Repair-pipeline lift under both memory conditions."
    ),
    "learning_contribution": "Production minus architecture-only pass@1.",
    "auroc_learning_contribution": "Production minus architecture-only AUROC.",
    "per_verifier_condition_a_auroc": "Per-verifier production AUROC.",
    "per_verifier_condition_b_auroc": "Per-verifier architecture-only AUROC.",
    "random_seeds_used": "Determinism for replication.",
    "reproducibility_checksum": "Content hash catches corpus, model, or state drift.",
    "model_specs": "Names the compute target actually required by the task.",
    "duration_s": "Real wall-clock measurement; never sleep-padded.",
    "preconditions_checked": "Anti-fabrication evidence recorded before inference.",
    "fr11_state_files": "Names every FR-11 state file that Condition B must reset.",
    "state_files_restored_sha_match": "Non-destructive reset proof.",
    "ground_truth_source": "HumanEval correctness comes from code execution.",
    "peer_humaneval_verifier_baselines": "External comparison context.",
    "baseline_comparison": "Shows lift versus vanilla and peer baselines.",
    "methodology_note": "Honest interpretation of measured or blocked state.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check performed before live HumanEval inference."""

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
    """One completed seed of the Exp 2830 dual-memory measurement."""

    seed: int
    condition_a_ensemble_auroc: float
    condition_b_ensemble_auroc: float
    condition_a_per_verifier: dict[str, float]
    condition_b_per_verifier: dict[str, float]
    vanilla_pass_at_1: float
    production_pass_at_1: float
    architecture_only_pass_at_1: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2830.

    Tests inject deterministic clocks and measured seed rows. Production keeps
    the default clock and blocks unless a real live-GPU backend is supplied.
    """

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    random_seeds: tuple[int, ...] = DEFAULT_RANDOM_SEEDS
    n_problems: int = DEFAULT_N_PROBLEMS
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    peer_baselines: tuple[Mapping[str, object], ...] = ()

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


class LiveMeasurementUnavailable(RuntimeError):
    """Raised when no real live-GPU HumanEval backend is attached."""


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

    Labels use the verifier convention: `1` is the positive/error class and
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
    vanilla_pass = [item.vanilla_pass_at_1 for item in measurements]
    production_pass = [item.production_pass_at_1 for item in measurements]
    architecture_pass = [item.architecture_only_pass_at_1 for item in measurements]

    per_a: dict[str, list[float]] = {}
    per_b: dict[str, list[float]] = {}
    for item in measurements:
        for name, value in item.condition_a_per_verifier.items():
            per_a.setdefault(name, []).append(value)
        for name, value in item.condition_b_per_verifier.items():
            per_b.setdefault(name, []).append(value)

    mean_a_auroc = _mean(condition_a_auroc)
    mean_b_auroc = _mean(condition_b_auroc)
    mean_vanilla = _mean(vanilla_pass)
    mean_production = _mean(production_pass)
    mean_architecture = _mean(architecture_pass)
    return {
        "condition_a_production_auroc_mean": mean_a_auroc,
        "condition_a_production_auroc_std": _std(condition_a_auroc),
        "condition_b_architecture_only_auroc_mean": mean_b_auroc,
        "condition_b_architecture_only_auroc_std": _std(condition_b_auroc),
        "condition_a_production_pass_at_1_mean": mean_production,
        "condition_a_production_pass_at_1_std": _std(production_pass),
        "condition_b_architecture_only_pass_at_1_mean": mean_architecture,
        "condition_b_architecture_only_pass_at_1_std": _std(architecture_pass),
        "pass_at_1_vanilla": mean_vanilla,
        "pass_at_1_after_carnot_correct_production": mean_production,
        "pass_at_1_after_carnot_correct_architecture_only": mean_architecture,
        "learning_contribution": mean_production - mean_architecture,
        "auroc_learning_contribution": mean_a_auroc - mean_b_auroc,
        "repair_lift_production_vs_vanilla": mean_production - mean_vanilla,
        "repair_lift_architecture_only_vs_vanilla": mean_architecture - mean_vanilla,
        "per_verifier_condition_a_auroc": per_a,
        "per_verifier_condition_b_auroc": per_b,
        "results_by_seed": [
            {
                "seed": item.seed,
                "condition_a_production_auroc": item.condition_a_ensemble_auroc,
                "condition_b_architecture_only_auroc": item.condition_b_ensemble_auroc,
                "vanilla_pass_at_1": item.vanilla_pass_at_1,
                "production_pass_at_1": item.production_pass_at_1,
                "architecture_only_pass_at_1": item.architecture_only_pass_at_1,
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
    """Check all Exp 2830 live resources before any inference attempt."""

    checks = [_cuda_check(), _hf_humaneval_check(), _qwen_cache_check(config.repo_root)]
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
    except Exception as exc:
        available = False
        detail = f"torch import failed: {exc!r}"
    return PreconditionCheck("cuda", available, detail)


def _hf_humaneval_check() -> PreconditionCheck:
    if importlib.util.find_spec("datasets") is None:
        return PreconditionCheck("hf_openai_humaneval", False, "datasets package is not installed")
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        rows = load_dataset("openai_humaneval", split="test[:1]")
        available = len(rows) > 0
        detail = f"loaded openai_humaneval test[:1], n={len(rows)}"
    except Exception as exc:
        available = False
        detail = f"HumanEval load failed: {exc!r}"
    return PreconditionCheck("hf_openai_humaneval", available, detail)


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
        "hf_openai_humaneval": "blocked_hf_openai_humaneval_unavailable",
        "qwen36_gguf_cache": "blocked_model_not_cached_qwen36_35b_a3b_gguf",
        "fr11_state_files": "blocked_fr11_state_files_missing",
        "live_backend": "blocked_live_qwen36_backend_unavailable",
    }
    for check in checks:
        if not check.available:
            return verdict_by_resource.get(check.resource, f"blocked_{check.resource}")
    return "blocked_unknown_resource"


def _peer_baselines(config: ExperimentConfig) -> list[dict[str, object]]:
    return [dict(item) for item in config.peer_baselines]


def _baseline_comparison(
    summary: Mapping[str, object],
    peer_baselines: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    production = float(summary["pass_at_1_after_carnot_correct_production"])
    architecture = float(summary["pass_at_1_after_carnot_correct_architecture_only"])
    vanilla = float(summary["pass_at_1_vanilla"])
    peer_values = [
        float(item["pass_at_1"])
        for item in peer_baselines
        if isinstance(item.get("pass_at_1"), int | float)
    ]
    peer_best = max(peer_values) if peer_values else None
    return {
        "production_minus_vanilla": production - vanilla,
        "architecture_only_minus_vanilla": architecture - vanilla,
        "production_minus_peer_best": None if peer_best is None else production - peer_best,
        "peer_best_pass_at_1": peer_best,
    }


def _null_baseline_comparison() -> dict[str, object]:
    return {
        "production_minus_vanilla": None,
        "architecture_only_minus_vanilla": None,
        "production_minus_peer_best": None,
        "peer_best_pass_at_1": None,
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
        "ground_truth_source": "official HumanEval check() execution",
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
            "condition_a_production_pass_at_1_mean": None,
            "condition_a_production_pass_at_1_std": None,
            "condition_b_architecture_only_pass_at_1_mean": None,
            "condition_b_architecture_only_pass_at_1_std": None,
            "pass_at_1_vanilla": None,
            "pass_at_1_after_carnot_correct_production": None,
            "pass_at_1_after_carnot_correct_architecture_only": None,
            "learning_contribution": None,
            "auroc_learning_contribution": None,
            "repair_lift_production_vs_vanilla": None,
            "repair_lift_architecture_only_vs_vanilla": None,
            "per_verifier_condition_a_auroc": {},
            "per_verifier_condition_b_auroc": {},
            "peer_humaneval_verifier_baselines": [],
            "baseline_comparison": _null_baseline_comparison(),
            "methodology_note": (
                "Blocked before live HumanEval inference, code execution, or repair because "
                "required resources were missing: "
                + ", ".join(check.resource for check in failed)
                + ". No pass@1, AUROC, candidate, execution, or ensemble metrics were inferred."
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
    peer_baselines = _peer_baselines(config)
    artifact.update(
        {
            "honest_verdict": (
                "complete: HumanEval-full dual-memory verifier ensemble measured "
                f"over {config.n_problems} problems and {len(measurements)} seeds"
            ),
            **summary,
            "peer_humaneval_verifier_baselines": peer_baselines,
            "baseline_comparison": _baseline_comparison(summary, peer_baselines),
            "methodology_note": (
                "Condition A used full FR-11 state. Condition B is expected to use the "
                "same HumanEval problem/candidate pairs after a non-destructive FR-11 "
                "state reset and Python restart. Correctness labels come from code "
                "execution, and this artifact reports only measured runner output."
            ),
        }
    )
    return artifact


def default_live_measurement_runner(
    _config: ExperimentConfig,
    _state_files: Sequence[dict[str, object]],
) -> list[SeedMeasurement]:
    raise LiveMeasurementUnavailable(
        "live Qwen3.6 HumanEval generation/execution/scoring backend is not configured"
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
    """Run Exp 2830 or write an honest blocked artifact before inference."""

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
