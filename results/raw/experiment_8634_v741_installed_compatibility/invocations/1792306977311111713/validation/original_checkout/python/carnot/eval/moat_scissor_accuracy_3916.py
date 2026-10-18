"""Exp 3916 robust-GGUF moat scissor accuracy measurement.

This evaluator measures whether Carnot catches gold-incorrect FoVer steps that
a reasoner's own self-verification misses.  Exp 3916 uses the Exp 3915 robust
GGUF generator path for live judgment, and it treats the Exp 3894 fixture AUROC
as the harness-validity control.  The in-distribution reasoner AUROC is only a
finding about self-correction behavior on this corpus.

Spec refs: REQ-VERIFY-3916, SCENARIO-VERIFY-3916,
SCENARIO-VERIFY-3916-BLOCKED.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.eval.moat_scissor_in_distribution import (
    DEFAULT_MIN_INCORRECT_STEPS,
    Exp3884Panel,
    PreconditionCheck,
    _score_digest,
    _sha256_file,
    load_exp3884_panel,
)
from carnot.eval.moat_scissor_regated import (
    CARNOT_AUROC_MIN,
    EXP3894_ARTIFACT_REL_PATH,
    HARNESS_FIXTURE_AUROC_MIN,
    HARNESS_MODULE_PATH,
    SelfVerifyArmScoring,
    build_boosted_judge_prompt,
    classify_verdict,
    compute_arm_scissor_metrics,
    load_exp3894_harness_source,
)
from carnot.eval.moat_scissor_tested_harness import _checked_sequence, _probe_cuda_with_venv
from carnot.eval.verifier_error_independence_scissor_at_scale import ScissorMetrics, _checksum
from carnot.verify.gguf_inference import (
    DEFAULT_PREFER_ORDER,
    generate as gguf_generate,
    load_gguf_generator,
)
from carnot.verify.reasoner_self_verification import reasoner_self_verify


OUTPUT_REL_PATH = Path("results/experiment_3916_moat_scissor_accuracy.json")
EXP3915_ARTIFACT_REL_PATH = Path("results/experiment_3915_robust_gguf_inference_harness.json")
GGUF_HARNESS_MODULE_PATH = "python/carnot/verify/gguf_inference.py"
TITLE = "moat_scissor_accuracy"
EXPERIMENT_ID = 3916
DEFAULT_RANDOM_SEED = 3916
DEFAULT_BOOTSTRAP_SEED = 3916
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
UPSTREAM_CARNOT_AUROC_MIN = 0.65
INFERENCE_SUBSTRATE = "live_llm_inference:robust_gguf_generator_self_verify_plus_exp3884_disk_carnot_scores"

REQUIRED_PRINCIPLE_FIELDS = (
    "gguf_harness_model_used",
    "harness_fixture_auroc",
    "residual_catch_rate_weak",
    "residual_catch_ci95_weak",
    "residual_catch_rate_strong",
    "residual_catch_ci95_strong",
    "reasoner_auroc_weak",
    "reasoner_auroc_strong",
    "error_overlap_jaccard_weak",
    "error_overlap_jaccard_strong",
    "n_residual_errors_weak",
    "n_residual_errors_strong",
    "carnot_ensemble_auroc",
    "harness_source",
    "corpus_source",
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
    "Pre-Launch + Adversarial-Verify - a live GGUF run over 2 arms takes real "
    "wall-clock (live floor 60s); implausibly short = fabrication."
)

FIELD_PRINCIPLES: dict[str, str] = {
    "gguf_harness_model_used": (
        "Which GGUF the robust harness loaded for the judge; fallback may skip the 35B."
    ),
    "harness_fixture_auroc": (
        "The harness-validity control from exp3894; >0.6 means the judge harness is sound."
    ),
    "residual_catch_rate_weak": (
        "THE moat metric on the WEAK arm - of the reasoner's MISSED errors, "
        "the fraction Carnot independently catches."
    ),
    "residual_catch_ci95_weak": "Bootstrap CI95 on the weak arm - the gate is on the lower bound.",
    "residual_catch_rate_strong": (
        "Moat metric on the STRONG boosted self-verify arm - the moat must survive a competent self-verifier."
    ),
    "residual_catch_ci95_strong": "Bootstrap CI95 on the strong arm - the conservative gate.",
    "reasoner_auroc_weak": (
        "FINDING (not a control): weak self-verify AUROC in-distribution."
    ),
    "reasoner_auroc_strong": (
        "FINDING: boosted self-verify AUROC; the moat is durable iff it survives this."
    ),
    "error_overlap_jaccard_weak": (
        "Independence - low overlap => different null spaces; high => redundant."
    ),
    "error_overlap_jaccard_strong": (
        "Independence - low overlap => different null spaces; high => redundant."
    ),
    "n_residual_errors_weak": "Size of set B for the weak arm; the residual CI is meaningful only if >=30.",
    "n_residual_errors_strong": "Size of set B for the strong arm; the residual CI is meaningful only if >=30.",
    "carnot_ensemble_auroc": "Positive control - in-band by the exp3884 corpus; must be >=0.65.",
    "harness_source": METHODOLOGY_PRINCIPLE,
    "corpus_source": METHODOLOGY_PRINCIPLE,
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
    "gguf_harness_model_used",
    "harness_fixture_auroc",
    "residual_catch_rate_weak",
    "residual_catch_ci95_weak",
    "residual_catch_rate_strong",
    "residual_catch_ci95_strong",
    "reasoner_auroc_weak",
    "reasoner_auroc_strong",
    "error_overlap_jaccard_weak",
    "error_overlap_jaccard_strong",
    "n_residual_errors_weak",
    "n_residual_errors_strong",
    "carnot_ensemble_auroc",
    "n_items",
    "duration_s",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 3916."""

    repo_root: Path
    output_path: Path | None = None
    random_seed: int = DEFAULT_RANDOM_SEED
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    min_incorrect_steps: int = DEFAULT_MIN_INCORRECT_STEPS
    upstream_auroc_min: float = UPSTREAM_CARNOT_AUROC_MIN
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
class PreflightResult:
    """Preconditions plus fixed inputs for a real Exp 3916 run."""

    checks: tuple[PreconditionCheck, ...]
    blocked_reason: str | None
    model_specs: dict[str, object]
    panel: Exp3884Panel | None
    harness_source: dict[str, object] | None
    gguf_harness_source: dict[str, object] | None


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ReasonerScorer = Callable[[Exp3884Panel, object, dict[str, object]], SelfVerifyArmScoring]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "blocked_"))


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_exp3915_gguf_harness_source(repo_root: Path) -> dict[str, object]:
    """Load and validate the Exp 3915 robust-GGUF readiness artifact."""

    artifact_path = repo_root / EXP3915_ARTIFACT_REL_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{EXP3915_ARTIFACT_REL_PATH.as_posix()} is missing")
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("exp3915 artifact is not a JSON object")
    if artifact.get("unit_test_passed") is not True:
        raise ValueError("exp3915 unit_test_passed is not true")
    smoke_tokens = int(artifact.get("smoke_tokens") or 0)
    if smoke_tokens <= 0:
        raise ValueError("exp3915 smoke_tokens must be >0")
    harness_module_path = str(artifact.get("harness_module_path") or "")
    if harness_module_path != GGUF_HARNESS_MODULE_PATH:
        raise FileNotFoundError(f"harness module path mismatch: {harness_module_path}")
    module_path = _resolve_repo_path(repo_root, harness_module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"harness module is missing: {harness_module_path}")

    return {
        "artifact_path": EXP3915_ARTIFACT_REL_PATH.as_posix(),
        "artifact_sha256": _sha256_file(artifact_path),
        "harness_module_path": harness_module_path,
        "harness_module_sha256": _sha256_file(module_path),
        "unit_test_passed": True,
        "smoke_tokens": smoke_tokens,
        "model_used": artifact.get("model_used"),
        "n_gpu_layers_used": artifact.get("n_gpu_layers_used"),
        "honest_verdict": artifact.get("honest_verdict"),
        "model_specs": artifact.get("model_specs", {}),
    }


def _prefer_order_from_source(gguf_harness_source: dict[str, object]) -> list[str]:
    model_used = gguf_harness_source.get("model_used")
    prefer_order: list[str] = []
    if isinstance(model_used, str) and model_used:
        prefer_order.append(model_used)
    for model_name in DEFAULT_PREFER_ORDER:
        if model_name not in prefer_order:
            prefer_order.append(model_name)
    return prefer_order


def load_robust_generator(
    gguf_harness_source: dict[str, object],
    config: ExperimentConfig,
) -> tuple[object, dict[str, object]]:
    """Load the live judge generator through the robust Exp 3915 harness."""

    n_gpu_layers = int(gguf_harness_source.get("n_gpu_layers_used") or -1)
    generator, meta = load_gguf_generator(
        prefer_order=_prefer_order_from_source(gguf_harness_source),
        n_ctx=config.n_ctx,
        max_n_gpu_layers=n_gpu_layers,
    )
    return generator, {
        **dict(meta),
        "loader": "carnot.verify.gguf_inference.load_gguf_generator",
        "source_exp3915_model_used": gguf_harness_source.get("model_used"),
        "source_exp3915_n_gpu_layers_used": gguf_harness_source.get("n_gpu_layers_used"),
        "n_ctx": config.n_ctx,
        "max_tokens_weak": config.max_tokens_weak,
        "max_tokens_strong": config.max_tokens_strong,
    }


def _make_robust_generator_factory(generator: object) -> Callable[..., object]:
    class RobustGeneratorAdapter:
        def __call__(self, prompt: str, **kwargs: object) -> str:
            return gguf_generate(generator, prompt, max_tokens=int(kwargs.get("max_tokens", 96)))

    adapter = RobustGeneratorAdapter()
    return lambda **_kwargs: adapter


def score_reasoner_arm_with_robust_generator(
    panel: Exp3884Panel,
    generator: object,
    model_specs: dict[str, object],
    *,
    arm: str,
    max_tokens: int = 96,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SelfVerifyArmScoring:
    """Call the tested Exp 3894 harness through the preloaded robust generator."""

    if arm not in {"weak", "strong"}:
        raise ValueError(f"arm must be weak or strong, got {arm!r}")
    kwargs: dict[str, object] = {
        "gold_labels": panel.labels,
        "llama_factory": _make_robust_generator_factory(generator),
        "max_tokens": max_tokens,
        "n_gpu_layers": int(model_specs.get("n_gpu_layers_used", -1)),
        "n_ctx": int(model_specs.get("n_ctx", 2048)),
        "n_batch": int(model_specs.get("n_batch", 64)),
        "offload_kqv": int(model_specs.get("n_gpu_layers_used", -1)) != 0,
        "random_seed": random_seed,
    }
    if arm == "strong":
        kwargs["prompt_builder"] = build_boosted_judge_prompt

    result = reasoner_self_verify(
        list(panel.texts),
        str(model_specs.get("gguf_path") or model_specs.get("model_path") or "robust_gguf_generator"),
        **kwargs,
    )
    raw_responses = tuple(
        str(item)
        for item in _checked_sequence(result.get("raw_responses"), field="raw_responses", n_expected=len(panel.texts))
    )
    scores = tuple(
        float(item)
        for item in _checked_sequence(result.get("per_step_score"), field="per_step_score", n_expected=len(panel.texts))
    )
    raw_preds = tuple(
        int(item)
        for item in _checked_sequence(result.get("per_step_pred"), field="per_step_pred", n_expected=len(panel.texts))
    )
    return SelfVerifyArmScoring(
        raw_responses=raw_responses,
        error_scores=scores,
        error_preds=tuple(1 if pred == 1 else 0 for pred in raw_preds),
        parsed_count=int(result.get("parsed_count") or 0),
        unparsed_count=int(result.get("unparsed_count") or 0),
        parser_constant_prediction=bool(result.get("parser_constant_prediction")),
    )


def _build_per_step_results(
    panel: Exp3884Panel,
    weak: SelfVerifyArmScoring,
    strong: SelfVerifyArmScoring,
) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "corpus_item_id": row.get("corpus_item_id"),
            "question_id": row.get("question_id"),
            "label": row.get("label"),
            "source": row.get("source"),
            "synthetic": bool(row.get("synthetic")),
            "reasoner_weak_raw_response": str(weak.raw_responses[index]),
            "reasoner_weak_rejects": int(weak.error_preds[index]) == 1,
            "reasoner_strong_raw_response": str(strong.raw_responses[index]),
            "reasoner_strong_rejects": int(strong.error_preds[index]) == 1,
            "carnot_score": float(panel.carnot_error_scores[index]),
            "carnot_rejects": int(panel.carnot_error_preds[index]) == 1,
        }
        for index, row in enumerate(panel.rows)
    ]


def build_artifact_from_metrics(
    *,
    weak_metrics: ScissorMetrics,
    strong_metrics: ScissorMetrics,
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    gguf_harness_source: dict[str, object] | None,
    harness_source: dict[str, object] | None,
    corpus_source: dict[str, object] | None,
    panel_sha256: str | None,
    weak_reasoner_error_scores: Sequence[float],
    strong_reasoner_error_scores: Sequence[float],
    carnot_error_scores: Sequence[float],
    per_step_results: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build the terminal Exp 3916 artifact from already-computed metrics."""

    started_at = config.start_time()
    finished_at = config.clock()
    harness_fixture_auroc = float((harness_source or {}).get("fixture_auroc") or 0.0)
    carnot_ensemble_auroc = strong_metrics.carnot_ensemble_auroc
    gguf_model_used = model_specs.get("model_used") or (gguf_harness_source or {}).get("model_used")
    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "panel_sha256": panel_sha256,
        "gguf_harness_source": gguf_harness_source,
        "harness_source": harness_source,
        "corpus_source": corpus_source,
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "bootstrap_seed": config.bootstrap_seed,
        "weak_reasoner_error_scores_digest": _score_digest(weak_reasoner_error_scores),
        "strong_reasoner_error_scores_digest": _score_digest(strong_reasoner_error_scores),
        "carnot_error_scores_digest": _score_digest(carnot_error_scores),
        "weak_metrics": {
            "residual_catch_rate": weak_metrics.residual_catch_rate,
            "residual_catch_ci95": weak_metrics.residual_catch_ci95,
            "error_overlap_jaccard": weak_metrics.error_overlap_jaccard,
            "reasoner_self_verify_auroc": weak_metrics.reasoner_self_verify_auroc,
        },
        "strong_metrics": {
            "residual_catch_rate": strong_metrics.residual_catch_rate,
            "residual_catch_ci95": strong_metrics.residual_catch_ci95,
            "error_overlap_jaccard": strong_metrics.error_overlap_jaccard,
            "reasoner_self_verify_auroc": strong_metrics.reasoner_self_verify_auroc,
            "carnot_ensemble_auroc": strong_metrics.carnot_ensemble_auroc,
        },
    }
    verdict = classify_verdict(
        harness_fixture_auroc=harness_fixture_auroc,
        carnot_ensemble_auroc=carnot_ensemble_auroc,
        strong_metrics=strong_metrics,
    )
    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "honest_verdict": verdict,
        "status": verdict,
        "gguf_harness_model_used": gguf_model_used,
        "harness_fixture_auroc": harness_fixture_auroc,
        "residual_catch_rate_weak": weak_metrics.residual_catch_rate,
        "residual_catch_ci95_weak": weak_metrics.residual_catch_ci95,
        "residual_catch_rate_strong": strong_metrics.residual_catch_rate,
        "residual_catch_ci95_strong": strong_metrics.residual_catch_ci95,
        "reasoner_auroc_weak": weak_metrics.reasoner_self_verify_auroc,
        "reasoner_auroc_strong": strong_metrics.reasoner_self_verify_auroc,
        "error_overlap_jaccard_weak": weak_metrics.error_overlap_jaccard,
        "error_overlap_jaccard_strong": strong_metrics.error_overlap_jaccard,
        "n_residual_errors_weak": weak_metrics.n_residual_errors,
        "n_residual_errors_strong": strong_metrics.n_residual_errors,
        "carnot_ensemble_auroc": carnot_ensemble_auroc,
        "positive_controls": {
            "harness_fixture_auroc_min": HARNESS_FIXTURE_AUROC_MIN,
            "harness_fixture_auroc_passed": harness_fixture_auroc > HARNESS_FIXTURE_AUROC_MIN,
            "carnot_ensemble_auroc_min": CARNOT_AUROC_MIN,
            "carnot_ensemble_auroc_passed": carnot_ensemble_auroc >= CARNOT_AUROC_MIN,
            "reasoner_in_distribution_auroc_is_finding_not_control": True,
        },
        "gguf_harness_source": gguf_harness_source,
        "harness_source": harness_source,
        "corpus_source": corpus_source,
        "n_items": strong_metrics.n_items,
        "n_gold_incorrect": strong_metrics.n_gold_incorrect,
        "n_reasoner_caught_errors_weak": len(weak_metrics.reasoner_caught_error_indices),
        "n_reasoner_caught_errors_strong": len(strong_metrics.reasoner_caught_error_indices),
        "n_carnot_caught_errors": len(strong_metrics.carnot_caught_error_indices),
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "random_seeds_used": {
            "corpus_order": "exp3884_as_recorded_no_sampling",
            "reasoner_self_verify_weak": config.random_seed,
            "reasoner_self_verify_strong": config.random_seed + 1,
            "bootstrap_weak": config.bootstrap_seed,
            "bootstrap_strong": config.bootstrap_seed + 1,
        },
        "panel_sha256": panel_sha256,
        "weak_reasoner_error_scores_sha256": _score_digest(weak_reasoner_error_scores),
        "strong_reasoner_error_scores_sha256": _score_digest(strong_reasoner_error_scores),
        "carnot_error_scores_sha256": _score_digest(carnot_error_scores),
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": finished_at - started_at,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methodology_note": (
            "No generation headroom gate is applied. Weak and strong reasoner judgments "
            "both come through python/carnot/verify/reasoner_self_verification.py "
            "backed by python/carnot/verify/gguf_inference.py."
        ),
        "per_step_results": list(per_step_results or []),
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def build_artifact_for_panel(
    panel: Exp3884Panel,
    *,
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    gguf_harness_source: dict[str, object],
    harness_source: dict[str, object],
    generator: object,
    weak_reasoner_scorer: ReasonerScorer | None = None,
    strong_reasoner_scorer: ReasonerScorer | None = None,
    write: bool = False,
) -> dict[str, object]:
    """Score the fixed panel with weak and boosted robust-generator arms."""

    weak_scorer = weak_reasoner_scorer or (
        lambda selected_panel, selected_generator, selected_model_specs: score_reasoner_arm_with_robust_generator(
            selected_panel,
            selected_generator,
            selected_model_specs,
            arm="weak",
            max_tokens=config.max_tokens_weak,
            random_seed=config.random_seed,
        )
    )
    strong_scorer = strong_reasoner_scorer or (
        lambda selected_panel, selected_generator, selected_model_specs: score_reasoner_arm_with_robust_generator(
            selected_panel,
            selected_generator,
            selected_model_specs,
            arm="strong",
            max_tokens=config.max_tokens_strong,
            random_seed=config.random_seed + 1,
        )
    )
    weak = weak_scorer(panel, generator, model_specs)
    strong = strong_scorer(panel, generator, model_specs)
    weak_metrics = compute_arm_scissor_metrics(
        labels=panel.labels,
        reasoner_error_scores=weak.error_scores,
        reasoner_error_preds=weak.error_preds,
        carnot_error_scores=panel.carnot_error_scores,
        carnot_error_preds=panel.carnot_error_preds,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    strong_metrics = compute_arm_scissor_metrics(
        labels=panel.labels,
        reasoner_error_scores=strong.error_scores,
        reasoner_error_preds=strong.error_preds,
        carnot_error_scores=panel.carnot_error_scores,
        carnot_error_preds=panel.carnot_error_preds,
        bootstrap_seed=config.bootstrap_seed + 1,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    artifact = build_artifact_from_metrics(
        weak_metrics=weak_metrics,
        strong_metrics=strong_metrics,
        config=config,
        preconditions_checked=preconditions_checked,
        model_specs=model_specs,
        gguf_harness_source=gguf_harness_source,
        harness_source=harness_source,
        corpus_source=panel.corpus_source,
        panel_sha256=panel.panel_sha256,
        weak_reasoner_error_scores=weak.error_scores,
        strong_reasoner_error_scores=strong.error_scores,
        carnot_error_scores=panel.carnot_error_scores,
        per_step_results=_build_per_step_results(panel, weak, strong),
    )
    artifact["tested_harness_runtime"] = {
        "weak": {
            "parsed_count": weak.parsed_count,
            "unparsed_count": weak.unparsed_count,
            "parser_constant_prediction": weak.parser_constant_prediction,
        },
        "strong": {
            "parsed_count": strong.parsed_count,
            "unparsed_count": strong.unparsed_count,
            "parser_constant_prediction": strong.parser_constant_prediction,
        },
    }
    validate_artifact(artifact)
    if write:
        write_artifact(config.resolved_output_path(), artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    model_specs: dict[str, object] | None = None,
    gguf_harness_source: dict[str, object] | None = None,
    harness_source: dict[str, object] | None = None,
    corpus_source: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a non-fabricated blocked artifact with metric fields empty."""

    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "honest_verdict": reason,
        "status": reason,
        "gguf_harness_model_used": None,
        "harness_fixture_auroc": None,
        "residual_catch_rate_weak": None,
        "residual_catch_ci95_weak": None,
        "residual_catch_rate_strong": None,
        "residual_catch_ci95_strong": None,
        "reasoner_auroc_weak": None,
        "reasoner_auroc_strong": None,
        "error_overlap_jaccard_weak": None,
        "error_overlap_jaccard_strong": None,
        "n_residual_errors_weak": 0,
        "n_residual_errors_strong": 0,
        "carnot_ensemble_auroc": None,
        "gguf_harness_source": gguf_harness_source,
        "harness_source": harness_source,
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
                "gguf_harness_source": gguf_harness_source,
                "harness_source": harness_source,
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


def write_blocked_artifact(
    output_path: Path,
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    model_specs: dict[str, object] | None = None,
    gguf_harness_source: dict[str, object] | None = None,
    harness_source: dict[str, object] | None = None,
    corpus_source: dict[str, object] | None = None,
) -> dict[str, object]:
    artifact = build_blocked_artifact(
        reason=reason,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        model_specs=model_specs,
        gguf_harness_source=gguf_harness_source,
        harness_source=harness_source,
        corpus_source=corpus_source,
    )
    write_artifact(output_path, artifact)
    return artifact


def probe_preconditions(
    config: ExperimentConfig,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> PreflightResult:
    """Probe live resources and upstream disk gates before scoring."""

    checks: list[PreconditionCheck] = [_probe_cuda_with_venv(config, command_runner=command_runner)]

    gguf_harness_source: dict[str, object] | None = None
    try:
        gguf_harness_source = load_exp3915_gguf_harness_source(config.repo_root)
        checks.append(
            PreconditionCheck(
                "exp3915_gguf_harness_ready",
                True,
                (
                    f"model_used={gguf_harness_source.get('model_used')} "
                    f"smoke_tokens={gguf_harness_source.get('smoke_tokens')}"
                ),
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3915_gguf_harness_ready", False, repr(exc)))

    for module_name, resource in (
        ("carnot.verify", "carnot_verify_import"),
        ("carnot.verify.gguf_inference", "gguf_inference_import"),
        ("carnot.verify.reasoner_self_verification", "reasoner_self_verification_import"),
    ):
        try:
            importlib.import_module(module_name)
            checks.append(PreconditionCheck(resource, True, f"import {module_name} OK"))
        except Exception as exc:
            checks.append(PreconditionCheck(resource, False, repr(exc)))

    harness_source: dict[str, object] | None = None
    try:
        harness_source = load_exp3894_harness_source(config.repo_root)
        checks.append(
            PreconditionCheck(
                "exp3894_judge_ready",
                True,
                f"fixture_auroc={harness_source['fixture_auroc']}",
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3894_judge_ready", False, repr(exc)))

    panel: Exp3884Panel | None = None
    try:
        panel = load_exp3884_panel(
            config.repo_root,
            min_incorrect=config.min_incorrect_steps,
            min_auroc=config.upstream_auroc_min,
        )
        checks.append(
            PreconditionCheck(
                "exp3884_corpus_in_band",
                True,
                (
                    f"items={len(panel.rows)} incorrect={sum(panel.labels)} "
                    f"recorded_auroc={panel.corpus_source['exp3884_recorded_carnot_ensemble_auroc_on_corpus']}"
                ),
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3884_corpus_in_band", False, repr(exc)))

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not available.get("exp3915_gguf_harness_ready", False):
        blocked_reason = "blocked_upstream_gguf_harness_not_ready"
    elif not (
        available.get("carnot_verify_import", False)
        and available.get("gguf_inference_import", False)
        and available.get("reasoner_self_verification_import", False)
    ):
        blocked_reason = "blocked_carnot_verify_import"
    elif not available.get("exp3894_judge_ready", False):
        blocked_reason = "blocked_upstream_judge_not_ready"
    elif not available.get("exp3884_corpus_in_band", False):
        blocked_reason = "blocked_upstream_corpus_not_in_band"

    model_specs = {
        "prefer_order": _prefer_order_from_source(gguf_harness_source or {}),
        "source_exp3915_model_used": (gguf_harness_source or {}).get("model_used"),
        "source_exp3915_n_gpu_layers_used": (gguf_harness_source or {}).get("n_gpu_layers_used"),
        "n_ctx": config.n_ctx,
        "max_tokens_weak": config.max_tokens_weak,
        "max_tokens_strong": config.max_tokens_strong,
    }
    return PreflightResult(
        checks=tuple(checks),
        blocked_reason=blocked_reason,
        model_specs=model_specs,
        panel=panel,
        harness_source=harness_source,
        gguf_harness_source=gguf_harness_source,
    )


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, object]:
    """Run Exp 3916 end to end, or write a blocked artifact on failed gates."""

    if config is None:  # pragma: no cover
        config = ExperimentConfig(repo_root=Path(__file__).resolve().parents[3])
    started = config.start_time()
    active_config = replace(config, started_at=started)
    preflight = probe_preconditions(active_config)
    if preflight.blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=preflight.blocked_reason,
            preconditions_checked=preflight.checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight.model_specs,
            gguf_harness_source=preflight.gguf_harness_source,
            harness_source=preflight.harness_source,
            corpus_source=preflight.panel.corpus_source if preflight.panel is not None else None,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    if preflight.gguf_harness_source is None:
        artifact = build_blocked_artifact(
            reason="blocked_upstream_gguf_harness_not_ready",
            preconditions_checked=preflight.checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight.model_specs,
            harness_source=preflight.harness_source,
            corpus_source=preflight.panel.corpus_source if preflight.panel is not None else None,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    if preflight.harness_source is None:
        artifact = build_blocked_artifact(
            reason="blocked_upstream_judge_not_ready",
            preconditions_checked=preflight.checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight.model_specs,
            gguf_harness_source=preflight.gguf_harness_source,
            corpus_source=preflight.panel.corpus_source if preflight.panel is not None else None,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    if preflight.panel is None:
        artifact = build_blocked_artifact(
            reason="blocked_upstream_corpus_not_in_band",
            preconditions_checked=preflight.checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight.model_specs,
            gguf_harness_source=preflight.gguf_harness_source,
            harness_source=preflight.harness_source,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    try:
        generator, generator_meta = load_robust_generator(preflight.gguf_harness_source, active_config)
    except Exception as exc:
        checks = (*preflight.checks, PreconditionCheck("robust_gguf_generator_load", False, repr(exc)))
        artifact = build_blocked_artifact(
            reason="blocked_all_gguf_inference_failed",
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight.model_specs,
            gguf_harness_source=preflight.gguf_harness_source,
            harness_source=preflight.harness_source,
            corpus_source=preflight.panel.corpus_source,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    model_specs = {**preflight.model_specs, **generator_meta}
    try:
        return build_artifact_for_panel(
            preflight.panel,
            config=active_config,
            preconditions_checked=preflight.checks,
            model_specs=model_specs,
            gguf_harness_source=preflight.gguf_harness_source,
            harness_source=preflight.harness_source,
            generator=generator,
            write=write,
        )
    except Exception as exc:
        checks = (*preflight.checks, PreconditionCheck("reasoner_self_verify_inference", False, repr(exc)))
        artifact = build_blocked_artifact(
            reason="blocked_reasoner_self_verify_inference_failed",
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            model_specs=model_specs,
            gguf_harness_source=preflight.gguf_harness_source,
            harness_source=preflight.harness_source,
            corpus_source=preflight.panel.corpus_source,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate the required Exp 3916 schema and terminal verdict discipline."""

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
    if "generation_headroom" in artifact:
        raise ValueError("Exp3916 must not apply or record a generation headroom gate")
    substrate = str(artifact.get("inference_substrate") or "")
    if "llama_cpp" in substrate:
        raise ValueError("Exp3916 inference_substrate must cite robust gguf_inference, not direct llama_cpp")


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=args.repo_root,
            output_path=args.output_path,
        ),
        write=True,
    )
    output_path = args.output_path if args.output_path is not None else args.repo_root / OUTPUT_REL_PATH
    print(f"{output_path.name} wrote {artifact['honest_verdict']}")
    return 0 if str(artifact["honest_verdict"]).startswith("complete:") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
