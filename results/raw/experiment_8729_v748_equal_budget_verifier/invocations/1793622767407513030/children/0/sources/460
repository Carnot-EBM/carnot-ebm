"""Exp 3928 independent-corpus moat scissor replication.

This evaluator repeats the Exp 3916 residual-catch property measurement on a
second held-out corpus. It tries ProcessBench first because that benchmark has
human first-incorrect-step labels; if that cannot be loaded in the local
environment, it falls back to the repository's held-out FoVer test file.

Spec refs: REQ-VERIFY-3928, SCENARIO-VERIFY-3928,
SCENARIO-VERIFY-3928-BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import (
    _fr11_memory_score,
    _label_to_int,
    _load_fr11_memory_index,
    _score_text_verifiers,
)
from carnot.eval.moat_scissor_accuracy_3916 import (
    EXP3915_ARTIFACT_REL_PATH,
    GGUF_HARNESS_MODULE_PATH,
    load_exp3915_gguf_harness_source,
    load_robust_generator,
    score_reasoner_arm_with_robust_generator,
)
from carnot.eval.moat_scissor_in_distribution import _score_digest, _sha256_file
from carnot.eval.moat_scissor_regated import (
    SelfVerifyArmScoring,
    compute_arm_scissor_metrics,
)
from carnot.eval.verifier_error_independence_scissor_at_scale import (
    ScissorMetrics,
    _checksum,
)


OUTPUT_REL_PATH = Path("results/experiment_3928_moat_scissor_replication.json")
TITLE = "moat_scissor_replication"
EXPERIMENT_ID = 3928
DEFAULT_RANDOM_SEED = 3928
DEFAULT_BOOTSTRAP_SEED = 3928
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_PROCESSBENCH_MIN_ITEMS = 120
PROCESSBENCH_DATASET = "Qwen/ProcessBench"
FOVER_FALLBACK_REL_PATH = Path("data/fover_test_v4.json")
LIVE_FLOOR_S = 60.0
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "live_llm_inference:robust_gguf_generator_self_verify_plus_independent_corpus_energy_ensemble"
)

REQUIRED_PRINCIPLE_FIELDS = (
    "corpus_used",
    "residual_catch_rate",
    "residual_catch_ci95",
    "error_overlap_jaccard",
    "n_residual_errors",
    "reasoner_auroc_strong",
    "carnot_ensemble_auroc",
    "moat_replicates",
    "n_items",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "random_seeds_used",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
)

METHODOLOGY_PRINCIPLE = (
    "Pre-Launch + Adversarial-Verify - a live self-verify run over an independent "
    "corpus takes wall-clock (live floor 60s); <60s = fabrication."
)

FIELD_PRINCIPLES: dict[str, str] = {
    "corpus_used": (
        "Which independent corpus loaded (processbench_slice / fover_test_v4_fallback) - "
        "neither verifier was tuned on it."
    ),
    "residual_catch_rate": (
        "THE moat metric on the independent corpus - of the reasoner's MISSED errors, "
        "the fraction energy catches."
    ),
    "residual_catch_ci95": "Bootstrap CI95; the gate is on the lower bound (>0.5).",
    "error_overlap_jaccard": (
        "Independence - low overlap => different null spaces (the moat replicates); high => redundant."
    ),
    "n_residual_errors": "Size of set B; the residual CI is meaningful only if >=30.",
    "reasoner_auroc_strong": (
        "FINDING - the strong self-verify AUROC on the independent corpus."
    ),
    "carnot_ensemble_auroc": (
        "The energy ensemble AUROC on the independent corpus (records cross-corpus discrimination)."
    ),
    "moat_replicates": (
        "BARE BOOL - residual_catch_ci95.low>0.5 AND overlap<0.6 AND n_res>=30 on the SECOND corpus."
    ),
    "n_items": METHODOLOGY_PRINCIPLE,
    "preconditions_checked": METHODOLOGY_PRINCIPLE,
    "model_specs": METHODOLOGY_PRINCIPLE,
    "random_seed": METHODOLOGY_PRINCIPLE,
    "random_seeds_used": METHODOLOGY_PRINCIPLE,
    "reproducibility_checksum": METHODOLOGY_PRINCIPLE,
    "duration_s": METHODOLOGY_PRINCIPLE,
    "inference_substrate": METHODOLOGY_PRINCIPLE,
}

WRAPPED_VALUE_FORBIDDEN_FIELDS = (
    "corpus_used",
    "residual_catch_rate",
    "residual_catch_ci95",
    "error_overlap_jaccard",
    "n_residual_errors",
    "reasoner_auroc_strong",
    "carnot_ensemble_auroc",
    "moat_replicates",
    "n_items",
    "duration_s",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource checked before the Exp 3928 live replication run."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"resource": self.resource, "available": self.available, "detail": self.detail}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 3928."""

    repo_root: Path
    output_path: Path | None = None
    random_seed: int = DEFAULT_RANDOM_SEED
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    min_processbench_items: int = DEFAULT_PROCESSBENCH_MIN_ITEMS
    processbench_dataset: str = PROCESSBENCH_DATASET
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    max_tokens_weak: int = 96
    max_tokens_strong: int = 160
    n_ctx: int = 2048
    cuda_probe_timeout_s: int = 60

    def resolved_output_path(self) -> Path:
        return self.output_path if self.output_path is not None else self.repo_root / OUTPUT_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def venv_python(self) -> Path:
        return self.repo_root / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class IndependentPanel:
    """Independent held-out rows plus stable labels and corpus provenance."""

    rows: tuple[dict[str, Any], ...]
    labels: tuple[int, ...]
    texts: tuple[str, ...]
    corpus_used: str
    panel_sha256: str
    corpus_source: dict[str, object]


@dataclass(frozen=True)
class EnergyScoring:
    """Production energy ensemble scores and thresholded catches."""

    scores: Sequence[float]
    error_preds: Sequence[int]
    threshold: float


CorpusLoader = Callable[[ExperimentConfig], IndependentPanel]
GeneratorLoader = Callable[[dict[str, object], ExperimentConfig], tuple[object, dict[str, object]]]
ReasonerScorer = Callable[[IndependentPanel, object, dict[str, object], ExperimentConfig], SelfVerifyArmScoring]
EnergyScorer = Callable[[IndependentPanel, Path], EnergyScoring]
CudaProbe = Callable[[ExperimentConfig], PreconditionCheck]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "blocked_"))


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = payload.get("items") or payload.get("examples") or payload.get("data") or payload.get("rows") or []
    else:
        candidates = []
    return [dict(row) for row in candidates if isinstance(row, dict)]


def _relative_to_repo(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _panel_sha(rows: Sequence[Mapping[str, object]], corpus_used: str) -> str:
    payload = [
        {
            "corpus_used": corpus_used,
            "corpus_item_id": row.get("corpus_item_id"),
            "question_id": row.get("question_id"),
            "label": row.get("label"),
            "step_text_sha256": hashlib.sha256(
                str(row.get("step_text", "")).encode("utf-8")
            ).hexdigest(),
        }
        for row in rows
    ]
    return _checksum(payload)


def _labels_have_both_classes(labels: Sequence[int]) -> bool:
    return len({int(label) for label in labels}) == 2


def _first_error_index(row: Mapping[str, object]) -> int:
    value = row.get("label", row.get("first_error_step", row.get("first_incorrect_step", -1)))
    if isinstance(value, bool):
        raise ValueError("first-error label must be an integer index, not bool")
    return int(value)


def panel_from_processbench_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    source_detail: str,
    min_problem_items: int,
) -> IndependentPanel:
    """Normalize ProcessBench first-error problem rows into per-step labels."""

    normalized: list[dict[str, Any]] = []
    problem_items = 0
    for problem_index, row in enumerate(rows):
        raw_steps = row.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            continue
        steps = [str(step).strip() for step in raw_steps if str(step).strip()]
        if not steps:
            continue
        try:
            first_error = _first_error_index(row)
        except (TypeError, ValueError):
            continue
        if first_error >= len(steps):
            continue
        if first_error < -1:
            continue
        problem_id = str(row.get("id") or row.get("problem_id") or f"processbench-{problem_index}")
        problem_items += 1
        for step_index, step_text in enumerate(steps):
            incorrect = step_index == first_error
            normalized.append(
                {
                    "corpus_item_id": f"{problem_id}:step{step_index}",
                    "question_id": problem_id,
                    "problem": row.get("problem"),
                    "step_index": step_index,
                    "first_error_step": first_error,
                    "step_text": step_text,
                    "label": "incorrect" if incorrect else "correct",
                    "source": "Qwen/ProcessBench",
                    "generator": row.get("generator"),
                    "synthetic": False,
                }
            )
    if problem_items < min_problem_items:
        raise ValueError(
            f"ProcessBench yielded {problem_items} labeled problem items; required>={min_problem_items}"
        )
    labels = tuple(_label_to_int(row["label"]) for row in normalized)
    if not _labels_have_both_classes(labels):
        raise ValueError("ProcessBench slice must contain both correct and first-error step labels")
    texts = tuple(str(row["step_text"]) for row in normalized)
    corpus_source = {
        "dataset": PROCESSBENCH_DATASET,
        "source_detail": source_detail,
        "processbench_problem_items": problem_items,
        "n_items": len(normalized),
        "n_incorrect": sum(labels),
        "label_semantics": "step is incorrect only at the human first-incorrect-step index",
    }
    return IndependentPanel(
        rows=tuple(normalized),
        labels=labels,
        texts=texts,
        corpus_used="processbench_slice",
        panel_sha256=_panel_sha(normalized, "processbench_slice"),
        corpus_source=corpus_source,
    )


def _take_rows(dataset: Iterable[Mapping[str, object]], limit: int) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for row in dataset:
        rows.append(dict(row))
        if len(rows) >= limit:
            break
    return rows


def load_processbench_panel(config: ExperimentConfig) -> IndependentPanel:
    """Load a ProcessBench GSM8K/math slice with first-error labels."""

    from datasets import load_dataset  # pragma: no cover - import is environment-dependent.

    attempts: list[str] = []
    candidates = (
        ("gsm8k", "test"),
        ("math", "test"),
        ("default", "gsm8k"),
        ("default", "math"),
    )
    for config_name, split_name in candidates:
        try:
            dataset = load_dataset(
                config.processbench_dataset,
                config_name,
                split=split_name,
                streaming=True,
            )
            rows = _take_rows(dataset, config.min_processbench_items)
            return panel_from_processbench_rows(
                rows,
                source_detail=f"config={config_name},split={split_name},streaming=true",
                min_problem_items=config.min_processbench_items,
            )
        except Exception as exc:  # pragma: no cover - exercised through fallback tests by monkeypatch.
            attempts.append(f"{config_name}/{split_name}: {exc!r}")
    raise RuntimeError("; ".join(attempts))


def load_fover_fallback_panel(
    config: ExperimentConfig,
    *,
    processbench_failure: Exception,
) -> IndependentPanel:
    """Load the local held-out FoVer fallback corpus."""

    path = config.repo_root / FOVER_FALLBACK_REL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"{FOVER_FALLBACK_REL_PATH.as_posix()} is missing")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_json_rows(_read_json(path))):
        step_text = str(row.get("step_text", "")).strip()
        if not step_text:
            continue
        try:
            label = _label_to_int(row.get("label"))
        except ValueError:
            continue
        rows.append(
            {
                **row,
                "corpus_item_id": str(row.get("corpus_item_id") or f"fover_test_v4:{index}"),
                "question_id": str(row.get("question_id") or f"fover_test_v4:{index}"),
                "step_text": step_text,
                "label": "incorrect" if label == 1 else "correct",
                "source": row.get("source", "fover_test_v4"),
                "synthetic": bool(row.get("synthetic", False)),
            }
        )
    labels = tuple(_label_to_int(row["label"]) for row in rows)
    if not rows or not _labels_have_both_classes(labels):
        raise ValueError("fover_test_v4 fallback must contain both labels")
    corpus_source = {
        "corpus_path": FOVER_FALLBACK_REL_PATH.as_posix(),
        "corpus_sha256": _sha256_file(path),
        "n_items": len(rows),
        "n_incorrect": sum(labels),
        "processbench_failure": repr(processbench_failure),
        "flagged_adversarial": False,
    }
    return IndependentPanel(
        rows=tuple(rows),
        labels=labels,
        texts=tuple(str(row["step_text"]) for row in rows),
        corpus_used="fover_test_v4_fallback",
        panel_sha256=_panel_sha(rows, "fover_test_v4_fallback"),
        corpus_source=corpus_source,
    )


def load_independent_corpus(config: ExperimentConfig) -> IndependentPanel:
    """Try ProcessBench first and fall back to the local held-out FoVer corpus."""

    try:
        return load_processbench_panel(config)
    except Exception as exc:
        return load_fover_fallback_panel(config, processbench_failure=exc)


def score_energy_ensemble(
    panel: IndependentPanel,
    repo_root: Path,
    *,
    verifier_scorer: Callable[[Sequence[str]], dict[str, list[float]]] = _score_text_verifiers,
    memory_loader: Callable[[Path], dict[str, object]] = _load_fr11_memory_index,
) -> EnergyScoring:
    """Score rows with the Exp 2837 production energy ensemble aggregation."""

    verifier_scores = verifier_scorer(tuple(panel.texts))
    tier0r = verifier_scores["tier0r_curry_howard"]
    tier0u = verifier_scores["tier0u_logical_consistency"]
    if len(tier0r) != len(panel.texts) or len(tier0u) != len(panel.texts):
        raise ValueError("energy verifier score lengths do not match panel")
    memory_index = memory_loader(repo_root)
    scores = tuple(
        float((0.9 * r_score) + (0.1 * u_score) + _fr11_memory_score(row, memory_index))
        for row, r_score, u_score in zip(panel.rows, tier0r, tier0u, strict=True)
    )
    threshold = float(statistics.median(scores))
    preds = tuple(1 if score > threshold else 0 for score in scores)
    return EnergyScoring(scores=scores, error_preds=preds, threshold=threshold)


def score_strong_reasoner(
    panel: IndependentPanel,
    generator: object,
    model_specs: dict[str, object],
    config: ExperimentConfig,
) -> SelfVerifyArmScoring:
    """Run the boosted strong self-verification arm through the robust generator."""

    return score_reasoner_arm_with_robust_generator(
        panel,
        generator,
        model_specs,
        arm="strong",
        max_tokens=config.max_tokens_strong,
        random_seed=config.random_seed,
    )


def _moat_replicates(metrics: ScissorMetrics) -> bool:
    return (
        float(metrics.residual_catch_ci95["low"]) > 0.5
        and metrics.error_overlap_jaccard < 0.6
        and metrics.n_residual_errors >= 30
    )


def classify_verdict(metrics: ScissorMetrics, *, corpus_used: str) -> str:
    """Apply the Exp 3928 independent-corpus falsification gate."""

    if metrics.n_residual_errors < 30:
        return f"complete: moat_scissor_INCONCLUSIVE_nres{metrics.n_residual_errors}_on_{corpus_used}"
    ci_low = float(metrics.residual_catch_ci95["low"])
    ci_high = float(metrics.residual_catch_ci95["high"])
    overlap = metrics.error_overlap_jaccard
    if ci_low > 0.5 and overlap < 0.6:
        return (
            "complete: "
            f"moat_scissor_REPLICATES_on_{corpus_used}_"
            f"residcatch{metrics.residual_catch_rate:.4f}_"
            f"ci{ci_low:.4f}-{ci_high:.4f}_overlap{overlap:.4f}_"
            f"nres{metrics.n_residual_errors}_not_a_corpus_artifact"
        )
    if ci_high < 0.3 or overlap > 0.7:
        return (
            "complete: "
            f"moat_scissor_NOT_REPLICATED_on_{corpus_used}_"
            f"residcatch{metrics.residual_catch_rate:.4f}_"
            f"overlap{overlap:.4f}_corpus_sensitivity_finding"
        )
    return f"complete: moat_scissor_INCONCLUSIVE_boundary_on_{corpus_used}"


def _build_per_step_results(
    panel: IndependentPanel,
    reasoner: SelfVerifyArmScoring,
    energy: EnergyScoring,
) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "corpus_item_id": row.get("corpus_item_id"),
            "question_id": row.get("question_id"),
            "step_index": row.get("step_index"),
            "label": row.get("label"),
            "corpus_used": panel.corpus_used,
            "reasoner_strong_raw_response": str(reasoner.raw_responses[index]),
            "reasoner_strong_score": float(reasoner.error_scores[index]),
            "reasoner_strong_rejects": int(reasoner.error_preds[index]) == 1,
            "carnot_score": float(energy.scores[index]),
            "carnot_rejects": int(energy.error_preds[index]) == 1,
            "energy_threshold": float(energy.threshold),
        }
        for index, row in enumerate(panel.rows)
    ]


def build_artifact_from_metrics(
    *,
    metrics: ScissorMetrics,
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    corpus_used: str,
    corpus_source: dict[str, object],
    panel_sha256: str,
    reasoner_error_scores: Sequence[float],
    carnot_error_scores: Sequence[float],
    per_step_results: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build the terminal Exp 3928 artifact from computed scissor metrics."""

    started_at = config.start_time()
    finished_at = config.clock()
    duration_s = finished_at - started_at
    verdict = (
        "blocked_live_duration_floor"
        if duration_s < LIVE_FLOOR_S
        else classify_verdict(metrics, corpus_used=corpus_used)
    )
    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "corpus_used": corpus_used,
        "corpus_source": corpus_source,
        "panel_sha256": panel_sha256,
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "bootstrap_seed": config.bootstrap_seed,
        "reasoner_error_scores_digest": _score_digest(reasoner_error_scores),
        "carnot_error_scores_digest": _score_digest(carnot_error_scores),
        "metrics": {
            "residual_catch_rate": metrics.residual_catch_rate,
            "residual_catch_ci95": metrics.residual_catch_ci95,
            "error_overlap_jaccard": metrics.error_overlap_jaccard,
            "reasoner_auroc_strong": metrics.reasoner_self_verify_auroc,
            "carnot_ensemble_auroc": metrics.carnot_ensemble_auroc,
            "n_residual_errors": metrics.n_residual_errors,
        },
    }
    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "honest_verdict": verdict,
        "status": verdict,
        "corpus_used": corpus_used,
        "residual_catch_rate": metrics.residual_catch_rate,
        "residual_catch_ci95": metrics.residual_catch_ci95,
        "error_overlap_jaccard": metrics.error_overlap_jaccard,
        "n_residual_errors": metrics.n_residual_errors,
        "reasoner_auroc_strong": metrics.reasoner_self_verify_auroc,
        "carnot_ensemble_auroc": metrics.carnot_ensemble_auroc,
        "moat_replicates": _moat_replicates(metrics) and verdict.startswith("complete:"),
        "corpus_source": corpus_source,
        "n_items": metrics.n_items,
        "n_gold_incorrect": metrics.n_gold_incorrect,
        "n_reasoner_caught_errors_strong": len(metrics.reasoner_caught_error_indices),
        "n_carnot_caught_errors": len(metrics.carnot_caught_error_indices),
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "random_seeds_used": {
            "corpus_selection": config.random_seed,
            "reasoner_self_verify_strong": config.random_seed,
            "bootstrap": config.bootstrap_seed,
        },
        "panel_sha256": panel_sha256,
        "reasoner_error_scores_sha256": _score_digest(reasoner_error_scores),
        "carnot_error_scores_sha256": _score_digest(carnot_error_scores),
        "frozen_fover_auroc_unchanged": FROZEN_FOVER_AUROC,
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methodology_note": (
            "This is an error-set independence measurement on a second corpus, not a "
            "generation test. Flagged adversarial artifacts are not aggregated."
        ),
        "per_step_results": list(per_step_results or []),
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    model_specs: dict[str, object] | None = None,
    corpus_source: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a terminal blocked artifact without metric claims."""

    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "honest_verdict": reason,
        "status": reason,
        "corpus_used": None,
        "residual_catch_rate": None,
        "residual_catch_ci95": None,
        "error_overlap_jaccard": None,
        "n_residual_errors": 0,
        "reasoner_auroc_strong": None,
        "carnot_ensemble_auroc": None,
        "moat_replicates": False,
        "corpus_source": corpus_source,
        "n_items": 0,
        "n_gold_incorrect": 0,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs or {},
        "random_seed": None,
        "random_seeds_used": {},
        "reproducibility_checksum": _checksum(
            {
                "experiment": EXPERIMENT_ID,
                "reason": reason,
                "preconditions_checked": [check.as_dict() for check in preconditions_checked],
                "model_specs": model_specs or {},
                "corpus_source": corpus_source,
            }
        ),
        "duration_s": duration_s,
        "inference_substrate": "none_blocked_preflight",
        "per_step_results": [],
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(output_path: Path, artifact: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate Exp 3928 schema, bare-field discipline, and terminal verdicts."""

    required = {"honest_verdict", "status", "field_principles", *REQUIRED_PRINCIPLE_FIELDS}
    missing = sorted(required - artifact.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not _terminal_verdict(verdict):
        raise ValueError(f"honest_verdict lacks terminal prefix: {verdict}")
    principles = artifact["field_principles"]
    if not isinstance(principles, dict):
        raise ValueError("field_principles must be a dict")
    for field in REQUIRED_PRINCIPLE_FIELDS:
        note = principles.get(field)
        if not isinstance(note, str) or not note:
            raise ValueError(f"missing string principle note for {field}")
    for field in WRAPPED_VALUE_FORBIDDEN_FIELDS:
        value = artifact.get(field)
        if isinstance(value, dict) and {"value", "principle"} <= set(value):
            raise ValueError(f"{field} must be a bare value, not a wrapper")
    if not isinstance(artifact.get("moat_replicates"), bool):
        raise ValueError("moat_replicates must be a bare bool")
    for field in ("n_items", "n_residual_errors"):
        value = artifact.get(field)
        if not isinstance(value, int):
            raise ValueError(f"{field} must be a bare int")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float):
        raise ValueError("duration_s must be a bare number")
    checksum = str(artifact.get("reproducibility_checksum") or "")
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum.lower()):
        raise ValueError("reproducibility_checksum must be a sha256 hex string")


def _probe_cuda_with_venv(config: ExperimentConfig) -> PreconditionCheck:  # pragma: no cover
    try:
        proc = subprocess.run(
            [str(config.venv_python()), "-c", "import torch; assert torch.cuda.is_available()"],
            capture_output=True,
            cwd=config.repo_root,
            text=True,
            timeout=config.cuda_probe_timeout_s,
            check=False,
        )
    except Exception as exc:
        return PreconditionCheck("cuda_available", False, repr(exc))
    detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
    return PreconditionCheck("cuda_available", proc.returncode == 0, detail)


def probe_preconditions(
    config: ExperimentConfig,
    *,
    cuda_probe: CudaProbe = _probe_cuda_with_venv,
) -> tuple[tuple[PreconditionCheck, ...], str | None, dict[str, object] | None]:
    """Check hard Exp 3928 resources before any live model scoring."""

    checks: list[PreconditionCheck] = [cuda_probe(config)]
    gguf_source: dict[str, object] | None = None
    try:
        gguf_source = load_exp3915_gguf_harness_source(config.repo_root)
        checks.append(
            PreconditionCheck(
                "exp3915_gguf_harness_ready",
                True,
                (
                    f"model_used={gguf_source.get('model_used')} "
                    f"smoke_tokens={gguf_source.get('smoke_tokens')}"
                ),
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3915_gguf_harness_ready", False, repr(exc)))

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not available.get("exp3915_gguf_harness_ready", False):
        blocked_reason = "blocked_upstream_gguf_harness_not_ready"
    return tuple(checks), blocked_reason, gguf_source


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    corpus_loader: CorpusLoader = load_independent_corpus,
    generator_loader: GeneratorLoader = load_robust_generator,
    reasoner_scorer: ReasonerScorer = score_strong_reasoner,
    energy_scorer: EnergyScorer = score_energy_ensemble,
    cuda_probe: CudaProbe = _probe_cuda_with_venv,
    write: bool = True,
) -> dict[str, object]:
    """Run Exp 3928 end to end, or write a blocked artifact on failed gates."""

    if config is None:  # pragma: no cover
        config = ExperimentConfig(repo_root=Path(__file__).resolve().parents[3])
    started = config.start_time()
    active_config = replace(config, started_at=started)
    checks, blocked_reason, gguf_source = probe_preconditions(active_config, cuda_probe=cuda_probe)
    if blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=blocked_reason,
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            model_specs=gguf_source or {},
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact
    if gguf_source is None:
        artifact = build_blocked_artifact(
            reason="blocked_upstream_gguf_harness_not_ready",
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    try:
        panel = corpus_loader(active_config)
    except Exception as exc:
        fail_checks = (*checks, PreconditionCheck("independent_corpus", False, repr(exc)))
        artifact = build_blocked_artifact(
            reason="blocked_independent_corpus_unavailable",
            preconditions_checked=fail_checks,
            duration_s=active_config.clock() - started,
            model_specs=gguf_source,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    try:
        energy = energy_scorer(panel, active_config.repo_root)
    except Exception as exc:
        fail_checks = (*checks, PreconditionCheck("energy_ensemble_scoring", False, repr(exc)))
        artifact = build_blocked_artifact(
            reason="blocked_energy_ensemble_scoring_failed",
            preconditions_checked=fail_checks,
            duration_s=active_config.clock() - started,
            model_specs=gguf_source,
            corpus_source=panel.corpus_source,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    try:
        generator, generator_meta = generator_loader(gguf_source, active_config)
    except Exception as exc:
        fail_checks = (*checks, PreconditionCheck("robust_gguf_generator_load", False, repr(exc)))
        artifact = build_blocked_artifact(
            reason="blocked_all_gguf_inference_failed",
            preconditions_checked=fail_checks,
            duration_s=active_config.clock() - started,
            model_specs=gguf_source,
            corpus_source=panel.corpus_source,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    model_specs = {**gguf_source, **generator_meta}
    try:
        reasoner = reasoner_scorer(panel, generator, model_specs, active_config)
    except Exception as exc:
        fail_checks = (*checks, PreconditionCheck("reasoner_self_verify_inference", False, repr(exc)))
        artifact = build_blocked_artifact(
            reason="blocked_reasoner_self_verify_inference_failed",
            preconditions_checked=fail_checks,
            duration_s=active_config.clock() - started,
            model_specs=model_specs,
            corpus_source=panel.corpus_source,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    metrics = compute_arm_scissor_metrics(
        labels=panel.labels,
        reasoner_error_scores=reasoner.error_scores,
        reasoner_error_preds=reasoner.error_preds,
        carnot_error_scores=energy.scores,
        carnot_error_preds=energy.error_preds,
        bootstrap_seed=active_config.bootstrap_seed,
        bootstrap_resamples=active_config.bootstrap_resamples,
    )
    artifact = build_artifact_from_metrics(
        metrics=metrics,
        config=active_config,
        preconditions_checked=checks,
        model_specs=model_specs,
        corpus_used=panel.corpus_used,
        corpus_source=panel.corpus_source,
        panel_sha256=panel.panel_sha256,
        reasoner_error_scores=reasoner.error_scores,
        carnot_error_scores=energy.scores,
        per_step_results=_build_per_step_results(panel, reasoner, energy),
    )
    if write:
        write_artifact(active_config.resolved_output_path(), artifact)
    return artifact


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(repo_root=args.repo_root, output_path=args.output_path),
        write=True,
    )
    output_path = args.output_path if args.output_path is not None else args.repo_root / OUTPUT_REL_PATH
    print(f"{output_path.name} wrote {artifact['honest_verdict']}")
    return 0 if str(artifact["honest_verdict"]).startswith("complete:") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
