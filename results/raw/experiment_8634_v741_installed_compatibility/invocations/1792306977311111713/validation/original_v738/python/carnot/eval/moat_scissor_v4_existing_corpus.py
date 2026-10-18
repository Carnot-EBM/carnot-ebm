"""Exp 3869 existing-corpus moat scissor v4.

This evaluator measures one product-risk property: whether Carnot catches
gold-incorrect reasoning steps that a strong local reasoner's own
self-verification misses. The corpus is already checked in and balanced, so
the runner deliberately does not generate, rebuild, or subsample data.

Spec: REQ-VERIFY-3869, SCENARIO-VERIFY-3869,
      SCENARIO-VERIFY-3869-BLOCKED.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
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
from carnot.eval.verifier_error_independence_scissor_at_scale import (
    CarnotScoring,
    PreconditionCheck,
    ReasonerScoring,
    ScissorMetrics,
    _checksum,
    compute_scissor_metrics,
    parse_reasoner_error_score,
    reasoner_self_verify_prompt,
)
from carnot.inference.sota_models import resolve_cached_gguf


OUTPUT_REL_PATH = Path("results/experiment_3869_moat_scissor_v4_existing_corpus.json")
TITLE = "moat_scissor_v4_existing_corpus"
DEFAULT_RANDOM_SEED = 3869
DEFAULT_BOOTSTRAP_SEED = 3869
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_MIN_INCORRECT_STEPS = 100
REASONER_AUROC_MIN = 0.55
REASONER_AUROC_MAX = 0.97
CARNOT_AUROC_MIN = 0.65
PRIMARY_REASONER_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
FALLBACK_REASONER_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
INFERENCE_SUBSTRATE = "live_llama_cpp_self_verification_plus_carnot_k15_ensemble_existing_prmbench_corpus"

REQUIRED_PRINCIPLE_FIELDS = (
    "residual_catch_rate",
    "residual_catch_ci95",
    "error_overlap_jaccard",
    "n_residual_errors",
    "reasoner_self_verify_auroc",
    "carnot_ensemble_auroc",
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
    "Pre-Launch + Adversarial-Verify + Inference-Substrate methodology - "
    "a live 35B GGUF run over 1000 steps takes real wall-clock; "
    "implausibly short = fabrication."
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "residual_catch_rate": {
        "principle": (
            "THE moat metric - of the reasoner's MISSED errors, the fraction "
            "Carnot independently catches. High => durable error-independent "
            "value; low => o1-subsumption risk (DT-P2)."
        )
    },
    "residual_catch_ci95": {
        "principle": (
            "Bootstrap CI95 - the lower bound is the gate; turns exp3827's "
            "~15-error point estimate into a defensible interval "
            "(Adversarial-Confirmation)."
        )
    },
    "error_overlap_jaccard": {
        "principle": (
            "Independence check - low overlap between reasoner-caught and "
            "Carnot-caught errors = different null spaces (the moat); high "
            "overlap = Carnot redundant with frontier self-verification."
        )
    },
    "n_residual_errors": {
        "principle": (
            "Size of set B (reasoner-missed gold errors); the residual CI is "
            "meaningful only if this is >=30. With 500 gold errors this "
            "should be comfortably met."
        )
    },
    "reasoner_self_verify_auroc": {
        "principle": (
            "Positive control - a degenerate reasoner would trivially inflate "
            "residual_catch; confirms genuine self-verification."
        )
    },
    "carnot_ensemble_auroc": {
        "principle": (
            "Positive control - the ensemble must discriminate on THIS corpus "
            "(>=0.65). PRMBench != FoVer, so the frozen 0.9131 does NOT apply here."
        )
    },
    "corpus_source": {
        "principle": (
            "Records that this ran against data/step_error_balanced_v2.json "
            "(PRMBench, 500 incorrect) - the provenance that makes the "
            "residual CI meaningful, unlike exp3844's 22-error FoVer."
        )
    },
    "n_items": {"principle": METHODOLOGY_PRINCIPLE},
    "preconditions_checked": {"principle": METHODOLOGY_PRINCIPLE},
    "model_specs": {"principle": METHODOLOGY_PRINCIPLE},
    "random_seed": {"principle": METHODOLOGY_PRINCIPLE},
    "random_seeds_used": {"principle": METHODOLOGY_PRINCIPLE},
    "reproducibility_checksum": {"principle": METHODOLOGY_PRINCIPLE},
    "duration_s": {"principle": METHODOLOGY_PRINCIPLE},
    "inference_substrate": {"principle": METHODOLOGY_PRINCIPLE},
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for the Exp 3869 runner."""

    repo_root: Path
    output_path: Path | None = None
    random_seed: int = DEFAULT_RANDOM_SEED
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    min_incorrect_steps: int = DEFAULT_MIN_INCORRECT_STEPS
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    max_tokens: int = 10
    cuda_probe_timeout_s: int = 60

    def resolved_output_path(self) -> Path:
        return self.output_path if self.output_path is not None else self.repo_root / OUTPUT_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def venv_python(self) -> Path:
        return self.repo_root / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class ExistingCorpusPanel:
    """Full step-error corpus plus stable identity fields."""

    rows: tuple[dict[str, Any], ...]
    labels: tuple[int, ...]
    texts: tuple[str, ...]
    panel_sha256: str
    corpus_source: dict[str, object]


@dataclass(frozen=True)
class PreflightResult:
    """Preconditions plus resolved inputs for a real Exp 3869 run."""

    checks: tuple[PreconditionCheck, ...]
    blocked_reason: str | None
    model_specs: dict[str, object]
    corpus_path: Path | None
    rows: tuple[dict[str, Any], ...]


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ReasonerScorer = Callable[[ExistingCorpusPanel, dict[str, object]], ReasonerScoring]
CarnotScorer = Callable[[ExistingCorpusPanel], CarnotScoring]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = payload.get("items") or []
    else:
        candidates = []
    return [dict(row) for row in candidates if isinstance(row, dict)]


def _relative_data_path(path: Path) -> str:
    if path.parent.name == "data":
        return (Path("data") / path.name).as_posix()
    return path.as_posix()


def _validate_existing_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    missing = [
        key
        for key in ("error_axis", "label", "question", "question_id", "source", "step_text")
        if key not in row
    ]
    if missing:
        raise ValueError(f"corpus row {index} missing required keys: {missing}")
    _label_to_int(row["label"])
    if str(row.get("step_text") or "").strip() == "":
        raise ValueError(f"corpus row {index} has empty step_text")
    return dict(row)


def load_existing_corpus_rows(
    repo_root: Path,
    *,
    min_incorrect: int = DEFAULT_MIN_INCORRECT_STEPS,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load the checked-in step-error corpus exactly as stored on disk."""

    path = repo_root / "data" / "step_error_balanced_v2.json"
    if not path.is_file():
        raise FileNotFoundError("data/step_error_balanced_v2.json is missing")
    rows = [_validate_existing_row(row, index) for index, row in enumerate(_json_rows(_read_json(path)))]
    n_incorrect = sum(1 for row in rows if _label_to_int(row["label"]) == 1)
    if n_incorrect < min_incorrect:
        raise ValueError(f"corpus has {n_incorrect} incorrect rows; required>={min_incorrect}")
    return path, rows


def _panel_sha(rows: Sequence[dict[str, Any]]) -> str:
    payload = [
        {
            "question_id": row.get("question_id"),
            "label": row.get("label"),
            "error_axis": row.get("error_axis"),
            "source": row.get("source"),
            "step_text_sha256": hashlib.sha256(
                str(row.get("step_text", "")).encode("utf-8")
            ).hexdigest(),
        }
        for row in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_existing_corpus_panel(
    rows: Sequence[dict[str, Any]],
    source_path: Path,
) -> ExistingCorpusPanel:
    """Return the full existing corpus panel without sampling or rebalancing."""

    normalized = tuple(dict(row) for row in rows)
    labels = tuple(_label_to_int(row["label"]) for row in normalized)
    texts = tuple(str(row.get("step_text", "")) for row in normalized)
    error_axes = sorted({str(row.get("error_axis")) for row in normalized if row.get("error_axis")})
    source_values = sorted({str(row.get("source")) for row in normalized if row.get("source")})
    corpus_source = {
        "path": _relative_data_path(source_path),
        "primary_source": source_values[0] if len(source_values) == 1 else "mixed",
        "source_values": source_values,
        "n_items": len(normalized),
        "n_incorrect_steps": sum(labels),
        "error_axis_coverage": error_axes,
    }
    return ExistingCorpusPanel(
        rows=normalized,
        labels=labels,
        texts=texts,
        panel_sha256=_panel_sha(normalized),
        corpus_source=corpus_source,
    )


def score_reasoner_with_llama_cpp(
    panel: ExistingCorpusPanel,
    model_specs: dict[str, object],
    *,
    max_tokens: int = 10,
    llama_factory: Callable[..., Any] | None = None,
) -> ReasonerScoring:
    """Run live llama.cpp self-verification for every existing-corpus step."""

    if llama_factory is None:
        from llama_cpp import Llama  # pragma: no cover

        llama_factory = Llama  # pragma: no cover
    llm = llama_factory(
        model_path=str(model_specs["model_path"]),
        n_gpu_layers=-1,
        n_ctx=2048,
        n_batch=256,
        verbose=False,
    )
    responses: list[str] = []
    for text in panel.texts:
        result = llm(
            reasoner_self_verify_prompt(text),
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["\n"],
        )
        responses.append(str(result["choices"][0]["text"]).strip())
    return ReasonerScoring(
        raw_responses=tuple(responses),
        error_scores=tuple(parse_reasoner_error_score(response) for response in responses),
    )


def score_carnot_ensemble(panel: ExistingCorpusPanel, repo_root: Path) -> CarnotScoring:
    """Score rows with the Exp 2837 production Tier0r/Tier0u/FR-11 aggregation."""

    verifier_scores = _score_text_verifiers(panel.texts)
    formal_scores = [
        0.9 * r_score + 0.1 * u_score
        for r_score, u_score in zip(
            verifier_scores["tier0r_curry_howard"],
            verifier_scores["tier0u_logical_consistency"],
            strict=True,
        )
    ]
    memory_index = _load_fr11_memory_index(repo_root)
    memory_scores = [_fr11_memory_score(row, memory_index) for row in panel.rows]
    full_scores = tuple(
        float(formal_score + memory_score)
        for formal_score, memory_score in zip(formal_scores, memory_scores, strict=True)
    )
    threshold = float(statistics.median(full_scores))
    preds = tuple(1 if score > threshold else 0 for score in full_scores)
    return CarnotScoring(scores=full_scores, error_preds=preds, threshold=threshold)


def _reasoner_control_passed(metrics: ScissorMetrics) -> bool:
    return REASONER_AUROC_MIN <= metrics.reasoner_self_verify_auroc <= REASONER_AUROC_MAX


def _carnot_control_passed(metrics: ScissorMetrics) -> bool:
    return metrics.carnot_ensemble_auroc >= CARNOT_AUROC_MIN


def classify_verdict(metrics: ScissorMetrics) -> str:
    """Apply the Exp 3869 terminal falsification gate."""

    failed_controls: list[str] = []
    if not _reasoner_control_passed(metrics):
        failed_controls.append("reasoner_self_verify_auroc")
    if not _carnot_control_passed(metrics):
        failed_controls.append("carnot_ensemble_auroc")
    if failed_controls:
        return f"complete: moat_scissor_v4_INCONCLUSIVE_{'_and_'.join(failed_controls)}"
    if metrics.n_residual_errors < 30:
        return "complete: moat_scissor_v4_INCONCLUSIVE_n_residual_errors_lt30"

    ci_low = float(metrics.residual_catch_ci95["low"])
    ci_high = float(metrics.residual_catch_ci95["high"])
    overlap = metrics.error_overlap_jaccard
    if ci_low > 0.5 and overlap < 0.6:
        return (
            "complete: "
            f"moat_scissor_v4_MOAT_SURVIVES_residcatch{metrics.residual_catch_rate:.4f}_"
            f"ci{ci_low:.4f}-{ci_high:.4f}_overlap{overlap:.4f}_"
            f"nres{metrics.n_residual_errors}"
        )
    if ci_high < 0.3 or overlap > 0.7:
        return (
            "complete: "
            f"moat_scissor_v4_MOAT_SUBSUMED_residcatch{metrics.residual_catch_rate:.4f}_"
            f"overlap{overlap:.4f}_o1_subsumption_risk_nres{metrics.n_residual_errors}"
        )
    return "complete: moat_scissor_v4_INCONCLUSIVE_boundary_gate"


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "complete_", "success:", "passed:", "shipped:", "blocked_"))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _score_digest(scores: Sequence[float]) -> str:
    rounded = [round(float(score), 12) for score in scores]
    return _checksum({"scores": rounded})


def _build_per_step_results(
    panel: ExistingCorpusPanel,
    reasoner: ReasonerScoring,
    carnot: CarnotScoring,
) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "question_id": row.get("question_id"),
            "label": row.get("label"),
            "error_axis": row.get("error_axis"),
            "source": row.get("source"),
            "reasoner_raw_response": str(reasoner.raw_responses[index]),
            "reasoner_rejects": int(reasoner.error_preds[index]) == 1,
            "carnot_score": float(carnot.scores[index]),
            "carnot_rejects": int(carnot.error_preds[index]) == 1,
        }
        for index, row in enumerate(panel.rows)
    ]


def build_artifact_from_metrics(
    *,
    metrics: ScissorMetrics,
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    corpus_source: dict[str, object] | None,
    panel_sha256: str | None,
    reasoner_error_scores: Sequence[float],
    carnot_error_scores: Sequence[float],
    per_step_results: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build the terminal Exp 3869 artifact from already-computed metrics."""

    started_at = config.start_time()
    finished_at = config.clock()
    checksum_payload = {
        "experiment": 3869,
        "panel_sha256": panel_sha256,
        "corpus_source": corpus_source,
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "bootstrap_seed": config.bootstrap_seed,
        "reasoner_error_scores_digest": _score_digest(reasoner_error_scores),
        "carnot_error_scores_digest": _score_digest(carnot_error_scores),
        "metrics": {
            "residual_catch_rate": metrics.residual_catch_rate,
            "residual_catch_ci95": metrics.residual_catch_ci95,
            "error_overlap_jaccard": metrics.error_overlap_jaccard,
            "reasoner_self_verify_auroc": metrics.reasoner_self_verify_auroc,
            "carnot_ensemble_auroc": metrics.carnot_ensemble_auroc,
        },
    }
    verdict = classify_verdict(metrics)
    artifact: dict[str, object] = {
        "experiment": 3869,
        "title": TITLE,
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "honest_verdict": verdict,
        "status": verdict,
        "residual_catch_rate": metrics.residual_catch_rate,
        "residual_catch_ci95": metrics.residual_catch_ci95,
        "error_overlap_jaccard": metrics.error_overlap_jaccard,
        "n_residual_errors": metrics.n_residual_errors,
        "reasoner_self_verify_auroc": metrics.reasoner_self_verify_auroc,
        "carnot_ensemble_auroc": metrics.carnot_ensemble_auroc,
        "positive_controls": {
            "reasoner_self_verify_auroc_range": [REASONER_AUROC_MIN, REASONER_AUROC_MAX],
            "reasoner_self_verify_auroc_passed": _reasoner_control_passed(metrics),
            "carnot_ensemble_auroc_min": CARNOT_AUROC_MIN,
            "carnot_ensemble_auroc_passed": _carnot_control_passed(metrics),
        },
        "corpus_source": corpus_source,
        "n_items": metrics.n_items,
        "n_gold_incorrect": metrics.n_gold_incorrect,
        "n_reasoner_caught_errors": len(metrics.reasoner_caught_error_indices),
        "n_carnot_caught_errors": len(metrics.carnot_caught_error_indices),
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "random_seeds_used": {
            "corpus_order": "as_is_no_sampling",
            "bootstrap": config.bootstrap_seed,
        },
        "panel_sha256": panel_sha256,
        "reasoner_error_scores_sha256": _score_digest(reasoner_error_scores),
        "carnot_error_scores_sha256": _score_digest(carnot_error_scores),
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": finished_at - started_at,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methodology_note": (
            "No generation headroom gate is applied. This artifact measures "
            "error-set independence on data/step_error_balanced_v2.json as-is."
        ),
        "per_step_results": list(per_step_results or []),
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def build_artifact_for_panel(
    panel: ExistingCorpusPanel,
    *,
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    reasoner_scorer: ReasonerScorer | None = None,
    carnot_scorer: CarnotScorer | None = None,
    write: bool = False,
) -> dict[str, object]:
    """Score the full corpus with injected or live scorers and build the artifact."""

    scorer = reasoner_scorer or (
        lambda selected_panel, selected_model_specs: score_reasoner_with_llama_cpp(
            selected_panel,
            selected_model_specs,
            max_tokens=config.max_tokens,
        )
    )
    reasoner = scorer(panel, model_specs)
    carnot = (carnot_scorer or (lambda selected_panel: score_carnot_ensemble(selected_panel, config.repo_root)))(panel)
    metrics = compute_scissor_metrics(
        labels=panel.labels,
        reasoner_error_scores=reasoner.error_scores,
        carnot_error_scores=carnot.scores,
        carnot_error_preds=carnot.error_preds,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    artifact = build_artifact_from_metrics(
        metrics=metrics,
        config=config,
        preconditions_checked=preconditions_checked,
        model_specs=model_specs,
        corpus_source=panel.corpus_source,
        panel_sha256=panel.panel_sha256,
        reasoner_error_scores=reasoner.error_scores,
        carnot_error_scores=carnot.scores,
        per_step_results=_build_per_step_results(panel, reasoner, carnot),
    )
    if write:
        write_artifact(config.resolved_output_path(), artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    model_specs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a non-fabricated blocked artifact with metric fields left empty."""

    artifact: dict[str, object] = {
        "experiment": 3869,
        "title": TITLE,
        "honest_verdict": reason,
        "status": reason,
        "residual_catch_rate": None,
        "residual_catch_ci95": None,
        "error_overlap_jaccard": None,
        "n_residual_errors": 0,
        "reasoner_self_verify_auroc": None,
        "carnot_ensemble_auroc": None,
        "corpus_source": None,
        "n_items": 0,
        "n_gold_incorrect": 0,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs or {},
        "random_seed": None,
        "random_seeds_used": {},
        "reproducibility_checksum": _checksum(
            {
                "experiment": 3869,
                "reason": reason,
                "preconditions_checked": [check.as_dict() for check in preconditions_checked],
                "model_specs": model_specs or {},
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
) -> dict[str, object]:
    artifact = build_blocked_artifact(
        reason=reason,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        model_specs=model_specs,
    )
    write_artifact(output_path, artifact)
    return artifact


def _resolve_reasoner_model() -> tuple[dict[str, object], list[PreconditionCheck]]:
    checks: list[PreconditionCheck] = []
    qwen_path = resolve_cached_gguf(PRIMARY_REASONER_HF_ID)
    qwen_available = qwen_path is not None and Path(qwen_path).is_file() and Path(qwen_path).stat().st_size > 0
    checks.append(
        PreconditionCheck(
            "qwen3.6_35b_gguf_cached",
            qwen_available,
            str(qwen_path) if qwen_path else "missing; checking fallback",
        )
    )
    selected_hf_id = PRIMARY_REASONER_HF_ID
    selected_path = qwen_path
    fallback_used = False
    if not qwen_available:
        fallback_path = resolve_cached_gguf(FALLBACK_REASONER_HF_ID)
        fallback_available = (
            fallback_path is not None and Path(fallback_path).is_file() and Path(fallback_path).stat().st_size > 0
        )
        checks.append(
            PreconditionCheck(
                "fallback_gemma_26b_gguf_cached",
                fallback_available,
                str(fallback_path) if fallback_path else "missing",
            )
        )
        selected_hf_id = FALLBACK_REASONER_HF_ID
        selected_path = fallback_path
        fallback_used = True

    model_specs = {
        "hf_id": selected_hf_id,
        "model_path": selected_path,
        "fallback_used": fallback_used,
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "max_tokens": 10,
    }
    return model_specs, checks


def _probe_cuda_with_venv(
    config: ExperimentConfig,
    *,
    command_runner: CommandRunner,
) -> PreconditionCheck:
    command = [
        str(config.venv_python()),
        "-c",
        "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0",
    ]
    try:
        proc = command_runner(
            command,
            capture_output=True,
            text=True,
            timeout=config.cuda_probe_timeout_s,
            check=False,
        )
        detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
        return PreconditionCheck("cuda_available", proc.returncode == 0, detail)
    except Exception as exc:
        return PreconditionCheck("cuda_available", False, repr(exc))


def probe_preconditions(
    config: ExperimentConfig,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> PreflightResult:
    """Probe required live resources before scoring the existing corpus."""

    checks: list[PreconditionCheck] = [
        _probe_cuda_with_venv(config, command_runner=command_runner)
    ]

    try:
        importlib.import_module("carnot.verify")
        checks.append(PreconditionCheck("carnot_verify_import", True, "import carnot.verify OK"))
    except Exception as exc:
        checks.append(PreconditionCheck("carnot_verify_import", False, repr(exc)))

    model_specs, model_checks = _resolve_reasoner_model()
    checks.extend(model_checks)

    try:
        importlib.import_module("llama_cpp")
        checks.append(PreconditionCheck("llama_cpp_import", True, "import llama_cpp OK"))
    except Exception as exc:
        checks.append(PreconditionCheck("llama_cpp_import", False, repr(exc)))

    corpus_path: Path | None = None
    rows: tuple[dict[str, Any], ...] = ()
    try:
        corpus_path, loaded_rows = load_existing_corpus_rows(
            config.repo_root,
            min_incorrect=config.min_incorrect_steps,
        )
        rows = tuple(loaded_rows)
        n_incorrect = sum(1 for row in rows if _label_to_int(row["label"]) == 1)
        checks.append(
            PreconditionCheck(
                "step_error_balanced_v2_corpus",
                n_incorrect >= config.min_incorrect_steps,
                f"{_relative_data_path(corpus_path)} rows={len(rows)} incorrect={n_incorrect}",
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("step_error_balanced_v2_corpus", False, repr(exc)))

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not available.get("carnot_verify_import", False):
        blocked_reason = "blocked_carnot_verify_import"
    elif not model_specs.get("model_path"):
        blocked_reason = "blocked_model_not_cached_qwen3.6_35b"
    elif not available.get("llama_cpp_import", False):
        blocked_reason = "blocked_llama_cpp_not_installed"
    elif not available.get("step_error_balanced_v2_corpus", False):
        blocked_reason = "blocked_corpus_missing"

    return PreflightResult(
        checks=tuple(checks),
        blocked_reason=blocked_reason,
        model_specs=model_specs,
        corpus_path=corpus_path,
        rows=rows,
    )


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, object]:
    """Run Exp 3869 end to end, or write a blocked artifact on failed preconditions."""

    config = config or ExperimentConfig(repo_root=Path(__file__).resolve().parents[3])
    started = config.start_time()
    active_config = replace(config, started_at=started)
    preflight = probe_preconditions(active_config)
    if preflight.blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=preflight.blocked_reason,
            preconditions_checked=preflight.checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight.model_specs,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    if preflight.corpus_path is None:
        artifact = build_blocked_artifact(
            reason="blocked_corpus_missing",
            preconditions_checked=preflight.checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight.model_specs,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    panel = build_existing_corpus_panel(preflight.rows, preflight.corpus_path)
    return build_artifact_for_panel(
        panel,
        config=active_config,
        preconditions_checked=preflight.checks,
        model_specs=preflight.model_specs,
        write=write,
    )


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate the required Exp 3869 schema and terminal verdict discipline."""

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
        if not isinstance(note, dict) or not note.get("principle"):
            raise ValueError(f"missing principle note for {field}")
    if "generation_headroom" in artifact:
        raise ValueError("Exp3869 must not apply or record a generation headroom gate")


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
