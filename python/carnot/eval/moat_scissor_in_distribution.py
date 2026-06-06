"""Exp 3885 in-distribution moat scissor.

This evaluator measures the DT-P2 durability property on the Exp 3884
in-distribution corpus: whether Carnot catches gold-incorrect reasoning steps
that a strong local reasoner's own self-verification misses. Carnot scores are
read from Exp 3884's persisted per-item ensemble artifact, so this runner is a
property measurement over fixed data rather than a generation or corpus-build
test.

Spec refs: REQ-VERIFY-3885, SCENARIO-VERIFY-3885,
SCENARIO-VERIFY-3885-BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import _label_to_int
from carnot.eval.verifier_error_independence_scissor_at_scale import (
    PreconditionCheck,
    ReasonerScoring,
    ScissorMetrics,
    _checksum,
    compute_scissor_metrics,
    parse_reasoner_error_score,
    reasoner_self_verify_prompt,
)
from carnot.inference.sota_models import resolve_cached_gguf


OUTPUT_REL_PATH = Path("results/experiment_3885_moat_scissor_in_distribution.json")
EXP3884_ARTIFACT_REL_PATH = Path("results/experiment_3884_in_distribution_error_rich_corpus.json")
TITLE = "moat_scissor_in_distribution"
DEFAULT_RANDOM_SEED = 3885
DEFAULT_BOOTSTRAP_SEED = 3885
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_MIN_INCORRECT_STEPS = 100
UPSTREAM_CARNOT_AUROC_MIN = 0.65
REASONER_AUROC_MIN = 0.55
REASONER_AUROC_MAX = 0.97
CARNOT_AUROC_MIN = 0.65
PRIMARY_REASONER_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
FALLBACK_REASONER_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
INFERENCE_SUBSTRATE = (
    "live_llama_cpp_qwen3.6_35b_self_verification_plus_exp3884_disk_carnot_ensemble_scores"
)

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
    "Pre-Launch + Adversarial-Verify + Inference-Substrate - a live 35B GGUF run "
    "takes real wall-clock; implausibly short = fabrication."
)

FIELD_PRINCIPLES: dict[str, str] = {
    "residual_catch_rate": (
        "THE moat metric - of the reasoner's MISSED errors, the fraction Carnot "
        "independently catches. High => durable; low => o1-subsumption risk."
    ),
    "residual_catch_ci95": (
        "Bootstrap CI95 - the gate is on the lower bound (Adversarial-Confirmation)."
    ),
    "error_overlap_jaccard": (
        "Independence - low overlap => different null spaces (the moat); high => redundant."
    ),
    "n_residual_errors": "Size of set B; the residual CI is meaningful only if >=30.",
    "reasoner_self_verify_auroc": (
        "Positive control - a degenerate reasoner trivially inflates residual_catch "
        "(the .357 failure)."
    ),
    "carnot_ensemble_auroc": (
        "Positive control - in-band by the exp3884 corpus; confirms the ensemble discriminates."
    ),
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


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 3885."""

    repo_root: Path
    output_path: Path | None = None
    random_seed: int = DEFAULT_RANDOM_SEED
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    min_incorrect_steps: int = DEFAULT_MIN_INCORRECT_STEPS
    upstream_auroc_min: float = UPSTREAM_CARNOT_AUROC_MIN
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
class Exp3884Panel:
    """Exp 3884 corpus rows and disk-backed Carnot ensemble decisions."""

    rows: tuple[dict[str, Any], ...]
    labels: tuple[int, ...]
    texts: tuple[str, ...]
    carnot_error_scores: tuple[float, ...]
    carnot_error_preds: tuple[int, ...]
    panel_sha256: str
    scores_sha256: str
    corpus_source: dict[str, object]


@dataclass(frozen=True)
class PreflightResult:
    """Preconditions plus fixed inputs for a real Exp 3885 run."""

    checks: tuple[PreconditionCheck, ...]
    blocked_reason: str | None
    model_specs: dict[str, object]
    panel: Exp3884Panel | None


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ReasonerScorer = Callable[[Exp3884Panel, dict[str, object]], ReasonerScoring]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = payload.get("items") or payload.get("rows") or payload.get("scores") or []
    else:
        candidates = []
    return [dict(row) for row in candidates if isinstance(row, dict)]


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _relative_to_repo(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _row_identity(row: dict[str, Any]) -> dict[str, object]:
    return {
        "corpus_item_id": row.get("corpus_item_id"),
        "question_id": row.get("question_id"),
        "label": row.get("label"),
        "step_text_sha256": hashlib.sha256(str(row.get("step_text", "")).encode("utf-8")).hexdigest(),
    }


def _panel_sha(rows: Sequence[dict[str, Any]]) -> str:
    encoded = json.dumps(
        [_row_identity(row) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_digest(scores: Sequence[float]) -> str:
    return _checksum({"scores": [round(float(score), 12) for score in scores]})


def _validate_alignment(row: dict[str, Any], score: dict[str, Any], index: int) -> None:
    for key in ("label", "carnot_ensemble_score", "carnot_rejects"):
        if key not in score:
            raise ValueError(f"score row {index} missing {key}")
    if "label" not in row or "step_text" not in row:
        raise ValueError(f"corpus row {index} missing label or step_text")
    if str(row.get("step_text") or "").strip() == "":
        raise ValueError(f"corpus row {index} has empty step_text")
    _label_to_int(row["label"])
    if str(row.get("label")) != str(score.get("label")):
        raise ValueError(f"label mismatch at row {index}")
    row_item_id = row.get("corpus_item_id")
    score_item_id = score.get("corpus_item_id")
    if row_item_id is not None and score_item_id is not None and row_item_id != score_item_id:
        raise ValueError(f"corpus_item_id mismatch at row {index}")
    expected_sha = hashlib.sha256(str(row.get("step_text", "")).encode("utf-8")).hexdigest()
    score_sha = score.get("step_text_sha256")
    if score_sha is not None and score_sha != expected_sha:
        raise ValueError(f"step_text_sha256 mismatch at row {index}")


def load_exp3884_panel(
    repo_root: Path,
    *,
    min_incorrect: int = DEFAULT_MIN_INCORRECT_STEPS,
    min_auroc: float = UPSTREAM_CARNOT_AUROC_MIN,
) -> Exp3884Panel:
    """Load the Exp 3884 corpus and its persisted per-item ensemble scores."""

    artifact_path = repo_root / EXP3884_ARTIFACT_REL_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{EXP3884_ARTIFACT_REL_PATH.as_posix()} is missing")
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("exp3884 artifact is not a JSON object")
    if artifact.get("flagged_adversarial") is True:
        raise ValueError("exp3884 artifact has flagged_adversarial=true")
    recorded_auroc = float(artifact.get("carnot_ensemble_auroc_on_corpus") or 0.0)
    if recorded_auroc < min_auroc:
        raise ValueError(f"carnot_ensemble_auroc_on_corpus {recorded_auroc:.4f} < {min_auroc:.4f}")

    corpus_rel = str(artifact.get("corpus_path") or "")
    scores_rel = str(artifact.get("per_item_ensemble_scores_path") or "")
    corpus_path = _resolve_repo_path(repo_root, corpus_rel)
    scores_path = _resolve_repo_path(repo_root, scores_rel)
    if not corpus_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError("exp3884 corpus or per-item score path is missing")

    rows = _json_rows(_read_json(corpus_path))
    score_rows = _json_rows(_read_json(scores_path))
    if len(rows) != len(score_rows):
        raise ValueError(f"corpus/scores length mismatch: {len(rows)} != {len(score_rows)}")

    labels: list[int] = []
    texts: list[str] = []
    carnot_scores: list[float] = []
    carnot_preds: list[int] = []
    for index, (row, score) in enumerate(zip(rows, score_rows, strict=True)):
        _validate_alignment(row, score, index)
        labels.append(_label_to_int(row["label"]))
        texts.append(str(row["step_text"]))
        carnot_scores.append(float(score["carnot_ensemble_score"]))
        carnot_preds.append(1 if bool(score["carnot_rejects"]) else 0)

    n_incorrect = sum(labels)
    if n_incorrect < min_incorrect:
        raise ValueError(f"exp3884 corpus has {n_incorrect} incorrect rows; required>={min_incorrect}")

    corpus_source = {
        "artifact_path": EXP3884_ARTIFACT_REL_PATH.as_posix(),
        "corpus_path": _relative_to_repo(repo_root, corpus_path),
        "per_item_ensemble_scores_path": _relative_to_repo(repo_root, scores_path),
        "exp3884_recorded_carnot_ensemble_auroc_on_corpus": recorded_auroc,
        "n_items": len(rows),
        "n_incorrect_steps": n_incorrect,
        "corpus_sha256": _sha256_file(corpus_path),
        "scores_sha256": _sha256_file(scores_path),
        "artifact_sha256": _sha256_file(artifact_path),
    }
    return Exp3884Panel(
        rows=tuple(rows),
        labels=tuple(labels),
        texts=tuple(texts),
        carnot_error_scores=tuple(carnot_scores),
        carnot_error_preds=tuple(carnot_preds),
        panel_sha256=_panel_sha(rows),
        scores_sha256=_score_digest(carnot_scores),
        corpus_source=corpus_source,
    )


def score_reasoner_with_llama_cpp(
    panel: Exp3884Panel,
    model_specs: dict[str, object],
    *,
    max_tokens: int = 10,
    llama_factory: Callable[..., Any] | None = None,
) -> ReasonerScoring:
    """Run live llama.cpp self-verification for every Exp 3884 corpus step."""

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


def _reasoner_control_passed(metrics: ScissorMetrics) -> bool:
    return REASONER_AUROC_MIN <= metrics.reasoner_self_verify_auroc <= REASONER_AUROC_MAX


def _carnot_control_passed(metrics: ScissorMetrics) -> bool:
    return metrics.carnot_ensemble_auroc >= CARNOT_AUROC_MIN


def classify_verdict(metrics: ScissorMetrics) -> str:
    """Apply the Exp 3885 terminal falsification gate."""

    failed_controls: list[str] = []
    if not _reasoner_control_passed(metrics):
        failed_controls.append("reasoner_self_verify_auroc")
    if not _carnot_control_passed(metrics):
        failed_controls.append("carnot_ensemble_auroc")
    if failed_controls:
        return f"complete: moat_scissor_indist_INCONCLUSIVE_{'_and_'.join(failed_controls)}"
    if metrics.n_residual_errors < 30:
        return "complete: moat_scissor_indist_INCONCLUSIVE_n_residual_errors_lt30"

    ci_low = float(metrics.residual_catch_ci95["low"])
    ci_high = float(metrics.residual_catch_ci95["high"])
    overlap = metrics.error_overlap_jaccard
    if ci_low > 0.5 and overlap < 0.6:
        return (
            "complete: "
            f"moat_scissor_indist_MOAT_SURVIVES_residcatch{metrics.residual_catch_rate:.4f}_"
            f"ci{ci_low:.4f}-{ci_high:.4f}_overlap{overlap:.4f}_"
            f"nres{metrics.n_residual_errors}"
        )
    if ci_high < 0.3 or overlap > 0.7:
        return (
            "complete: "
            f"moat_scissor_indist_MOAT_SUBSUMED_residcatch{metrics.residual_catch_rate:.4f}_"
            f"overlap{overlap:.4f}_o1_subsumption_risk_nres{metrics.n_residual_errors}"
        )
    return "complete: moat_scissor_indist_INCONCLUSIVE_boundary_gate"


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "complete_", "success:", "passed:", "shipped:", "blocked_"))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _build_per_step_results(
    panel: Exp3884Panel,
    reasoner: ReasonerScoring,
) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "corpus_item_id": row.get("corpus_item_id"),
            "question_id": row.get("question_id"),
            "label": row.get("label"),
            "source": row.get("source"),
            "synthetic": bool(row.get("synthetic")),
            "reasoner_raw_response": str(reasoner.raw_responses[index]),
            "reasoner_rejects": int(reasoner.error_preds[index]) == 1,
            "carnot_score": float(panel.carnot_error_scores[index]),
            "carnot_rejects": int(panel.carnot_error_preds[index]) == 1,
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
    """Build the terminal Exp 3885 artifact from already-computed metrics."""

    started_at = config.start_time()
    finished_at = config.clock()
    checksum_payload = {
        "experiment": 3885,
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
        "experiment": 3885,
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
            "corpus_order": "exp3884_as_recorded_no_sampling",
            "bootstrap": config.bootstrap_seed,
        },
        "panel_sha256": panel_sha256,
        "reasoner_error_scores_sha256": _score_digest(reasoner_error_scores),
        "carnot_error_scores_sha256": _score_digest(carnot_error_scores),
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": finished_at - started_at,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methodology_note": (
            "No generation headroom gate is applied. This artifact measures error-set "
            "independence on the fixed Exp 3884 in-distribution corpus."
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
    reasoner_scorer: ReasonerScorer | None = None,
    write: bool = False,
) -> dict[str, object]:
    """Score the fixed panel with injected or live reasoner self-verification."""

    scorer = reasoner_scorer or (
        lambda selected_panel, selected_model_specs: score_reasoner_with_llama_cpp(
            selected_panel,
            selected_model_specs,
            max_tokens=config.max_tokens,
        )
    )
    reasoner = scorer(panel, model_specs)
    metrics = compute_scissor_metrics(
        labels=panel.labels,
        reasoner_error_scores=reasoner.error_scores,
        carnot_error_scores=panel.carnot_error_scores,
        carnot_error_preds=panel.carnot_error_preds,
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
        carnot_error_scores=panel.carnot_error_scores,
        per_step_results=_build_per_step_results(panel, reasoner),
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
    """Build a non-fabricated blocked artifact with metric fields empty."""

    artifact: dict[str, object] = {
        "experiment": 3885,
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
                "experiment": 3885,
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
        "n_ctx": 2048,
        "n_batch": 256,
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
        "import torch; assert torch.cuda.is_available()",
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
    """Probe live resources and the Exp 3884 in-band corpus gate."""

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
    elif not available.get("carnot_verify_import", False):
        blocked_reason = "blocked_carnot_verify_import"
    elif not model_specs.get("model_path"):
        blocked_reason = "blocked_model_not_cached"
    elif not available.get("llama_cpp_import", False):
        blocked_reason = "blocked_llama_cpp_not_installed"
    elif not available.get("exp3884_corpus_in_band", False):
        blocked_reason = "blocked_upstream_corpus_not_in_band"

    return PreflightResult(
        checks=tuple(checks),
        blocked_reason=blocked_reason,
        model_specs=model_specs,
        panel=panel,
    )


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, object]:
    """Run Exp 3885 end to end, or write a blocked artifact on failed gates."""

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

    if preflight.panel is None:
        artifact = build_blocked_artifact(
            reason="blocked_upstream_corpus_not_in_band",
            preconditions_checked=preflight.checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight.model_specs,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    try:
        return build_artifact_for_panel(
            preflight.panel,
            config=active_config,
            preconditions_checked=preflight.checks,
            model_specs=preflight.model_specs,
            write=write,
        )
    except Exception as exc:
        checks = (
            *preflight.checks,
            PreconditionCheck("llama_cpp_inference", False, repr(exc)),
        )
        artifact = build_blocked_artifact(
            reason="blocked_llama_cpp_inference_failed",
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight.model_specs,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate the required Exp 3885 schema and terminal verdict discipline."""

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
    if "generation_headroom" in artifact:
        raise ValueError("Exp3885 must not apply or record a generation headroom gate")


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
