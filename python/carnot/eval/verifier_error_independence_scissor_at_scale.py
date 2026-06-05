"""Exp 3844 verifier error-independence scissor measurement.

This module measures a narrow product-risk property: whether Carnot catches
gold-incorrect FoVer steps that a strong local reasoner's own self-verification
misses.  AUROC remains a positive control, but the headline moat metric is the
residual catch rate over the reasoner's missed errors.

Spec: REQ-VERIFY-3844, SCENARIO-VERIFY-3844.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import (
    _fr11_memory_score,
    _label_to_int,
    _load_fr11_memory_index,
    _score_text_verifiers,
    _select_balanced_subset,
    compute_auroc,
)
from carnot.inference.sota_models import resolve_cached_gguf


OUTPUT_REL_PATH = Path("results/experiment_3844_verifier_error_independence_scissor_at_scale.json")
TITLE = "verifier_error_independence_scissor_at_scale"
DEFAULT_RANDOM_SEED = 42
DEFAULT_BOOTSTRAP_SEED = 3844
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_N_ITEMS = 1000
FROZEN_FOVER_AUROC = 0.9131
CARNOT_AUROC_TOLERANCE = 0.02
REASONER_AUROC_MIN = 0.55
REASONER_AUROC_MAX = 0.95
PRIMARY_REASONER_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
FALLBACK_REASONER_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
INFERENCE_SUBSTRATE = "live_llama_cpp_self_verification_plus_carnot_fover_ensemble"

REQUIRED_PRINCIPLE_FIELDS = (
    "residual_catch_rate",
    "residual_catch_ci95",
    "error_overlap_jaccard",
    "reasoner_self_verify_auroc",
    "carnot_ensemble_auroc",
    "n_items",
    "n_residual_errors",
    "preconditions_checked",
    "cited_upstream_artifacts",
    "model_specs",
    "random_seed",
    "random_seeds_used",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "residual_catch_rate": {
        "principle": (
            "THE moat metric -- of the reasoner's MISSED errors, what fraction Carnot "
            "independently catches. High => error-independent durable value; low => "
            "o1-subsumption risk (DT-P2)."
        )
    },
    "residual_catch_ci95": {
        "principle": (
            "Bootstrap CI95 -- turns exp3827's round-number point estimate into a "
            "defensible interval; the lower bound is the gate field "
            "(Adversarial-Confirmation Discipline)."
        )
    },
    "error_overlap_jaccard": {
        "principle": (
            "Independence check -- low overlap between reasoner-caught and Carnot-caught "
            "errors means the two verifiers occupy different null spaces (the moat); "
            "high overlap means Carnot is redundant with frontier self-verification."
        )
    },
    "reasoner_self_verify_auroc": {
        "principle": (
            "Positive control -- a degenerate (all-accept/all-reject) reasoner would "
            "inflate residual_catch trivially; this confirms the reasoner genuinely "
            "self-verifies."
        )
    },
    "carnot_ensemble_auroc": {
        "principle": (
            "Positive control -- MUST reproduce the frozen 0.9131 +/-0.02 on this "
            "corpus or the ensemble path is unfaithful and ALL residual numbers are "
            "void (exp3820 mode)."
        )
    },
    "n_items": {
        "principle": (
            "N>=1000 -- the scale-up requirement that makes exp3827's 100-item "
            "preliminary result falsifiable."
        )
    },
    "n_residual_errors": {
        "principle": (
            "Residual-set size -- the residual_catch CI is only meaningful if the "
            "residual error set is itself large enough; if <30, widen N."
        )
    },
    "preconditions_checked": {
        "principle": (
            "Pre-Launch methodology -- records CUDA, GGUF, carnot.verify, llama.cpp, "
            "FoVer corpus, and upstream artifact checks before live scoring."
        )
    },
    "cited_upstream_artifacts": {
        "principle": (
            "Adversarial-Verify provenance -- cites exp3827 and exp2837 artifacts and "
            "their scripts with SHA256 so method reuse is auditable."
        )
    },
    "model_specs": {
        "principle": (
            "Inference-substrate methodology -- records the actual GGUF path loaded by "
            "llama.cpp rather than a tokenizer-less HuggingFace repo id."
        )
    },
    "random_seed": {
        "principle": "Balanced FoVer panel seed for exact reproduction.",
    },
    "random_seeds_used": {
        "principle": "All deterministic seeds used by sampling and bootstrap stages.",
    },
    "reproducibility_checksum": {
        "principle": (
            "Adversarial-verify checksum over panel identity, scores, upstream hashes, "
            "model specs, and seeds."
        )
    },
    "duration_s": {
        "principle": (
            "Real measured wall-clock duration; live 35B GGUF scoring over N>=1000 "
            "should not look implausibly short."
        )
    },
    "inference_substrate": {
        "principle": "Declares live llama.cpp self-verification plus Carnot verifier scoring.",
    },
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource gate checked before any live Exp 3844 measurement."""

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
class ExperimentConfig:
    """Runtime configuration for the Exp 3844 runner."""

    repo_root: Path
    output_path: Path | None = None
    n_items: int = DEFAULT_N_ITEMS
    random_seed: int = DEFAULT_RANDOM_SEED
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    max_tokens: int = 10

    def resolved_output_path(self) -> Path:
        return self.output_path if self.output_path is not None else self.repo_root / OUTPUT_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


@dataclass(frozen=True)
class FoVerPanel:
    """Selected FoVer rows plus stable identity fields for reproducibility."""

    rows: tuple[dict[str, Any], ...]
    labels: tuple[int, ...]
    texts: tuple[str, ...]
    panel_sha256: str


@dataclass(frozen=True)
class ReasonerScoring:
    """Strong reasoner self-verification outputs."""

    raw_responses: Sequence[str]
    error_scores: Sequence[float]

    @property
    def error_preds(self) -> tuple[int, ...]:
        return tuple(1 if float(score) > 0.0 else 0 for score in self.error_scores)


@dataclass(frozen=True)
class CarnotScoring:
    """Carnot production FoVer ensemble scores and thresholded catches."""

    scores: Sequence[float]
    error_preds: Sequence[int]
    threshold: float


@dataclass(frozen=True)
class ScissorMetrics:
    """The Exp3827 error-set metrics and Exp3844 positive controls."""

    residual_catch_rate: float
    residual_catch_ci95: dict[str, float | int]
    error_overlap_jaccard: float
    reasoner_self_verify_auroc: float
    carnot_ensemble_auroc: float
    n_items: int
    n_residual_errors: int
    n_gold_incorrect: int
    reasoner_caught_error_indices: tuple[int, ...]
    carnot_caught_error_indices: tuple[int, ...]


@dataclass(frozen=True)
class PreflightResult:
    """Preconditions plus resolved inputs for a real Exp3844 run."""

    checks: tuple[PreconditionCheck, ...]
    blocked_reason: str | None
    model_specs: dict[str, object]
    corpus_path: Path | None
    rows: tuple[dict[str, Any], ...]
    cited_upstream_artifacts: dict[str, object]


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
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
        candidates = payload.get("items") or payload.get("examples") or payload.get("data") or []
    else:
        candidates = []
    return [dict(row) for row in candidates if isinstance(row, dict)]


def load_fover_test_rows(repo_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Load the preferred cached FoVer test corpus and validate required columns."""

    for rel_path in (
        Path("data/fover_test_v4.json"),
        Path("data/fover_test_v3.json"),
        Path("data/fover_test.json"),
    ):
        path = repo_root / rel_path
        if not path.is_file():
            continue
        valid_rows: list[dict[str, Any]] = []
        for row in _json_rows(_read_json(path)):
            has_question = "question_id" in row or "question" in row
            if not has_question or "step_text" not in row or "label" not in row:
                continue
            try:
                _label_to_int(row["label"])
            except ValueError:
                continue
            valid_rows.append(row)
        if valid_rows:
            return path, valid_rows
    raise FileNotFoundError("no FoVer test corpus with {question_id|question, step_text, label}")


def _panel_sha(rows: Sequence[dict[str, Any]]) -> str:
    payload = [
        {
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "label": row.get("label"),
            "step_text_sha256": hashlib.sha256(
                str(row.get("step_text", "")).encode("utf-8")
            ).hexdigest(),
        }
        for row in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_balanced_panel(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    n_items: int,
) -> FoVerPanel:
    """Draw the balanced FoVer panel used by both verifiers."""

    subset = _select_balanced_subset(rows, seed=seed, n_examples=n_items)
    labels = tuple(_label_to_int(row["label"]) for row in subset)
    texts = tuple(str(row.get("step_text", "")) for row in subset)
    return FoVerPanel(
        rows=tuple(dict(row) for row in subset),
        labels=labels,
        texts=texts,
        panel_sha256=_panel_sha(subset),
    )


def reasoner_self_verify_prompt(step_text: str) -> str:
    """Build the Exp3827 self-verification prompt verbatim."""

    return (
        "A user has provided a step in a reasoning problem. Is this step correct? "
        "Answer strictly YES if it is correct, and NO if it contains an error.\n\n"
        f"Step: {step_text}\nAnswer:"
    )


def parse_reasoner_error_score(response: str) -> int:
    """Return 1 when the Exp3827 parser treats the reasoner response as an error."""

    return 1 if "no" in str(response).strip().lower() else 0


def score_reasoner_with_llama_cpp(
    panel: FoVerPanel,
    model_specs: dict[str, object],
    *,
    max_tokens: int = 10,
    llama_factory: Callable[..., Any] | None = None,
) -> ReasonerScoring:
    """Run live llama.cpp self-verification for every selected FoVer step."""

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
        response = str(result["choices"][0]["text"]).strip()
        responses.append(response)
    return ReasonerScoring(
        raw_responses=tuple(responses),
        error_scores=tuple(parse_reasoner_error_score(response) for response in responses),
    )


def score_carnot_ensemble(panel: FoVerPanel, repo_root: Path) -> CarnotScoring:
    """Score rows with the Exp2837 production Tier0r/Tier0u/FR-11 aggregation."""

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
    full_scores = [
        float(formal_score + memory_score)
        for formal_score, memory_score in zip(formal_scores, memory_scores, strict=True)
    ]
    threshold = float(statistics.median(full_scores))
    preds = tuple(1 if score > threshold else 0 for score in full_scores)
    return CarnotScoring(scores=tuple(full_scores), error_preds=preds, threshold=threshold)


def bootstrap_binary_ci(
    values: Sequence[int | float],
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, float | int]:
    """Bootstrap a CI95 for a binary catch-rate sample."""

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "mean": 0.0,
            "low": 0.0,
            "high": 0.0,
            "n_resamples": n_resamples,
            "bootstrap_seed": seed,
        }
    rng = random.Random(seed)
    n = len(numeric)
    means = [
        sum(numeric[rng.randrange(n)] for _ in range(n)) / n
        for _resample in range(n_resamples)
    ]
    means.sort()
    low = means[int(0.025 * (n_resamples - 1))]
    high = means[int(0.975 * (n_resamples - 1))]
    return {
        "mean": sum(numeric) / n,
        "low": low,
        "high": high,
        "n_resamples": n_resamples,
        "bootstrap_seed": seed,
    }


def _assert_aligned(*sequences: Sequence[object]) -> None:
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError(f"all score/label sequences must align, got lengths={sorted(lengths)}")


def compute_scissor_metrics(
    *,
    labels: Sequence[int],
    reasoner_error_scores: Sequence[float],
    carnot_error_scores: Sequence[float],
    carnot_error_preds: Sequence[int],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> ScissorMetrics:
    """Compute Exp3827 residual-catch and overlap definitions at Exp3844 scale."""

    _assert_aligned(labels, reasoner_error_scores, carnot_error_scores, carnot_error_preds)
    reasoner_error_preds = tuple(1 if float(score) > 0.0 else 0 for score in reasoner_error_scores)
    label_ints = tuple(int(label) for label in labels)
    reasoner_caught = tuple(
        idx for idx, (label, pred) in enumerate(zip(label_ints, reasoner_error_preds, strict=True))
        if label == 1 and pred == 1
    )
    reasoner_missed = tuple(
        idx for idx, (label, pred) in enumerate(zip(label_ints, reasoner_error_preds, strict=True))
        if label == 1 and pred == 0
    )
    carnot_caught = tuple(
        idx for idx, (label, pred) in enumerate(zip(label_ints, carnot_error_preds, strict=True))
        if label == 1 and int(pred) == 1
    )
    carnot_caught_set = set(carnot_caught)
    residual_catches = [1 if idx in carnot_caught_set else 0 for idx in reasoner_missed]
    residual_ci = bootstrap_binary_ci(
        residual_catches,
        seed=bootstrap_seed,
        n_resamples=bootstrap_resamples,
    )
    reasoner_set = set(reasoner_caught)
    carnot_set = set(carnot_caught)
    union = reasoner_set | carnot_set
    overlap = len(reasoner_set & carnot_set) / len(union) if union else 0.0
    return ScissorMetrics(
        residual_catch_rate=float(residual_ci["mean"]),
        residual_catch_ci95=residual_ci,
        error_overlap_jaccard=overlap,
        reasoner_self_verify_auroc=compute_auroc(label_ints, reasoner_error_scores),
        carnot_ensemble_auroc=compute_auroc(label_ints, carnot_error_scores),
        n_items=len(label_ints),
        n_residual_errors=len(reasoner_missed),
        n_gold_incorrect=sum(1 for label in label_ints if label == 1),
        reasoner_caught_error_indices=reasoner_caught,
        carnot_caught_error_indices=carnot_caught,
    )


def _reasoner_control_passed(metrics: ScissorMetrics) -> bool:
    return REASONER_AUROC_MIN <= metrics.reasoner_self_verify_auroc <= REASONER_AUROC_MAX


def _carnot_control_passed(metrics: ScissorMetrics) -> bool:
    return abs(metrics.carnot_ensemble_auroc - FROZEN_FOVER_AUROC) <= CARNOT_AUROC_TOLERANCE


def classify_verdict(metrics: ScissorMetrics) -> str:
    """Apply the Exp3844 terminal falsification gate."""

    if not _reasoner_control_passed(metrics):
        return "complete: scissor_at_scale_INCONCLUSIVE_reasoner_self_verify_auroc"
    if not _carnot_control_passed(metrics):
        return "complete: scissor_at_scale_INCONCLUSIVE_carnot_ensemble_auroc"
    if metrics.n_items < DEFAULT_N_ITEMS:
        return "complete: scissor_at_scale_INCONCLUSIVE_n_items_lt1000"
    if metrics.n_residual_errors < 30:
        return "complete: scissor_at_scale_INCONCLUSIVE_n_residual_errors_lt30"

    ci_low = float(metrics.residual_catch_ci95["low"])
    ci_high = float(metrics.residual_catch_ci95["high"])
    overlap = metrics.error_overlap_jaccard
    if ci_low > 0.5 and overlap < 0.6:
        return (
            "complete: "
            f"scissor_at_scale_MOAT_SURVIVES_residcatch{metrics.residual_catch_rate:.4f}_"
            f"ci{ci_low:.4f}-{ci_high:.4f}_overlap{overlap:.4f}_n{metrics.n_items}"
        )
    if ci_high < 0.3 or overlap > 0.7:
        return (
            "complete: "
            f"scissor_at_scale_MOAT_SUBSUMED_residcatch{metrics.residual_catch_rate:.4f}_"
            f"overlap{overlap:.4f}_o1_subsumption_risk_n{metrics.n_items}"
        )
    return "complete: scissor_at_scale_INCONCLUSIVE_boundary_gate"


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "complete_", "success:", "passed:", "shipped:", "blocked_"))


def collect_upstream_artifacts(repo_root: Path) -> dict[str, object]:
    """Collect SHA256 provenance for the Exp3827 method and Exp2837 scorer."""

    upstream: dict[str, object] = {}
    for key, rel_path in {
        "exp3827_result": Path("results/experiment_3827_verifier_error_independence_scissor.json"),
        "exp3827_script": Path("scripts/experiments/experiment_3827_verifier_error_independence_scissor.py"),
        "exp2837_result": Path("results/experiment_2837_fover_memory_leakage_v3.json"),
        "exp2837_script": Path("scripts/experiment_2837_fover_memory_leakage_v3.py"),
    }.items():
        path = repo_root / rel_path
        record: dict[str, object] = {
            "path": rel_path.as_posix(),
            "sha256": _sha256_file(path),
            "exists": path.is_file(),
        }
        if path.suffix == ".json" and path.is_file():
            try:
                payload = _read_json(path)
            except json.JSONDecodeError:
                payload = {}
            record["flagged_adversarial"] = bool(
                isinstance(payload, dict) and payload.get("flagged_adversarial") is True
            )
        upstream[key] = record
    return upstream


def _upstream_flagged(cited_upstream_artifacts: dict[str, object]) -> bool:
    return any(
        isinstance(record, dict) and record.get("flagged_adversarial") is True
        for record in cited_upstream_artifacts.values()
    )


def _resolve_reasoner_model() -> tuple[dict[str, object], list[PreconditionCheck]]:
    checks: list[PreconditionCheck] = []
    qwen_path = resolve_cached_gguf(PRIMARY_REASONER_HF_ID)
    checks.append(
        PreconditionCheck(
            "qwen3.6_35b_gguf_cached",
            qwen_path is not None and Path(qwen_path).is_file() and Path(qwen_path).stat().st_size > 0,
            str(qwen_path) if qwen_path else "missing; checking fallback",
        )
    )
    selected_hf_id = PRIMARY_REASONER_HF_ID
    selected_path = qwen_path
    fallback_used = False
    if selected_path is None or not Path(selected_path).is_file() or Path(selected_path).stat().st_size == 0:
        fallback_path = resolve_cached_gguf(FALLBACK_REASONER_HF_ID)
        checks.append(
            PreconditionCheck(
                "fallback_gemma_26b_gguf_cached",
                fallback_path is not None
                and Path(fallback_path).is_file()
                and Path(fallback_path).stat().st_size > 0,
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
    }
    return model_specs, checks


def probe_preconditions(config: ExperimentConfig) -> PreflightResult:
    """Probe required live resources before selecting or scoring rows."""

    checks: list[PreconditionCheck] = []
    try:
        import torch

        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        checks.append(
            PreconditionCheck(
                "cuda_available",
                bool(torch.cuda.is_available() and cuda_count > 0),
                f"device_count={cuda_count}",
            )
        )
    except Exception as exc:  # pragma: no cover
        checks.append(PreconditionCheck("cuda_available", False, repr(exc)))

    try:
        importlib.import_module("carnot.verify")
        checks.append(PreconditionCheck("carnot_verify_import", True, "import carnot.verify OK"))
    except Exception as exc:  # pragma: no cover
        checks.append(PreconditionCheck("carnot_verify_import", False, repr(exc)))

    model_specs, model_checks = _resolve_reasoner_model()
    checks.extend(model_checks)

    try:
        importlib.import_module("llama_cpp")
        checks.append(PreconditionCheck("llama_cpp_import", True, "import llama_cpp OK"))
    except Exception as exc:  # pragma: no cover
        checks.append(PreconditionCheck("llama_cpp_import", False, repr(exc)))

    corpus_path: Path | None = None
    rows: tuple[dict[str, Any], ...] = ()
    try:
        corpus_path, loaded_rows = load_fover_test_rows(config.repo_root)
        rows = tuple(loaded_rows)
        n_positive = sum(1 for row in rows if _label_to_int(row["label"]) == 1)
        n_negative = len(rows) - n_positive
        required_positive = config.n_items // 2
        required_negative = config.n_items - required_positive
        checks.append(
            PreconditionCheck(
                "fover_test_corpus",
                len(rows) >= config.n_items,
                f"{corpus_path.as_posix()} rows={len(rows)} required>={config.n_items}",
            )
        )
        checks.append(
            PreconditionCheck(
                "fover_balanced_panel_capacity",
                n_positive >= required_positive and n_negative >= required_negative,
                (
                    f"{corpus_path.as_posix()} positives={n_positive} negatives={n_negative} "
                    f"required_positive>={required_positive} required_negative>={required_negative}"
                ),
            )
        )
    except Exception as exc:  # pragma: no cover
        checks.append(PreconditionCheck("fover_test_corpus", False, repr(exc)))

    cited = collect_upstream_artifacts(config.repo_root)
    upstream_available = all(
        isinstance(record, dict) and record.get("exists") and record.get("sha256")
        for record in cited.values()
    )
    checks.append(
        PreconditionCheck(
            "upstream_artifacts_available",
            upstream_available,
            "exp3827 and exp2837 artifacts/scripts have sha256"
            if upstream_available
            else "one or more upstream artifacts missing",
        )
    )
    checks.append(
        PreconditionCheck(
            "upstream_adversarial_flags_absent",
            not _upstream_flagged(cited),
            "no cited upstream artifact has flagged_adversarial=true",
        )
    )

    blocked_reason = None
    available_by_resource = {check.resource: check.available for check in checks}
    if not available_by_resource.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not available_by_resource.get("carnot_verify_import", False):
        blocked_reason = "blocked_carnot_verify_import"
    elif not model_specs.get("model_path"):
        blocked_reason = "blocked_model_not_cached_qwen3.6_35b"
    elif not available_by_resource.get("llama_cpp_import", False):
        blocked_reason = "blocked_llama_cpp_not_installed"
    elif not available_by_resource.get("fover_test_corpus", False):
        blocked_reason = "blocked_fover_corpus_not_available"
    elif not available_by_resource.get("fover_balanced_panel_capacity", False):
        blocked_reason = "blocked_fover_balanced_corpus_not_available"
    elif not available_by_resource.get("upstream_artifacts_available", False):
        blocked_reason = "blocked_upstream_artifacts_unavailable"
    elif not available_by_resource.get("upstream_adversarial_flags_absent", False):
        blocked_reason = "blocked_upstream_adversarial_flag"

    return PreflightResult(
        checks=tuple(checks),
        blocked_reason=blocked_reason,
        model_specs=model_specs,
        corpus_path=corpus_path,
        rows=rows,
        cited_upstream_artifacts=cited,
    )


def _checksum(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def build_artifact_from_metrics(
    *,
    metrics: ScissorMetrics,
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    cited_upstream_artifacts: dict[str, object],
    panel_sha256: str,
    reasoner_error_scores: Sequence[float],
    carnot_error_scores: Sequence[float],
) -> dict[str, object]:
    """Build the terminal Exp3844 artifact from already-computed metrics."""

    started_at = config.start_time()
    finished_at = config.clock()
    checksum_payload = {
        "experiment": 3844,
        "panel_sha256": panel_sha256,
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "bootstrap_seed": config.bootstrap_seed,
        "reasoner_error_scores": [float(score) for score in reasoner_error_scores],
        "carnot_error_scores": [round(float(score), 12) for score in carnot_error_scores],
        "cited_upstream_artifacts": cited_upstream_artifacts,
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
        "experiment": 3844,
        "title": TITLE,
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "honest_verdict": verdict,
        "status": verdict,
        "residual_catch_rate": metrics.residual_catch_rate,
        "residual_catch_ci95": metrics.residual_catch_ci95,
        "error_overlap_jaccard": metrics.error_overlap_jaccard,
        "reasoner_self_verify_auroc": metrics.reasoner_self_verify_auroc,
        "carnot_ensemble_auroc": metrics.carnot_ensemble_auroc,
        "positive_controls": {
            "reasoner_self_verify_auroc_range": [REASONER_AUROC_MIN, REASONER_AUROC_MAX],
            "reasoner_self_verify_auroc_passed": _reasoner_control_passed(metrics),
            "carnot_ensemble_auroc_expected": FROZEN_FOVER_AUROC,
            "carnot_ensemble_auroc_tolerance": CARNOT_AUROC_TOLERANCE,
            "carnot_ensemble_auroc_passed": _carnot_control_passed(metrics),
        },
        "n_items": metrics.n_items,
        "n_residual_errors": metrics.n_residual_errors,
        "n_gold_incorrect": metrics.n_gold_incorrect,
        "n_reasoner_caught_errors": len(metrics.reasoner_caught_error_indices),
        "n_carnot_caught_errors": len(metrics.carnot_caught_error_indices),
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "cited_upstream_artifacts": cited_upstream_artifacts,
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "random_seeds_used": {
            "balanced_panel": config.random_seed,
            "bootstrap": config.bootstrap_seed,
        },
        "panel_sha256": panel_sha256,
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": finished_at - started_at,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "methodology_note": (
            "No generation headroom gate is applied. This artifact measures "
            "error-set independence: Carnot catches among the strong reasoner's "
            "self-verification misses."
        ),
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


ReasonerScorer = Callable[[FoVerPanel, dict[str, object]], ReasonerScoring]
CarnotScorer = Callable[[FoVerPanel], CarnotScoring]


def build_artifact_for_panel(
    panel: FoVerPanel,
    *,
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    cited_upstream_artifacts: dict[str, object],
    reasoner_scorer: ReasonerScorer | None = None,
    carnot_scorer: CarnotScorer | None = None,
    write: bool = False,
) -> dict[str, object]:
    """Score a selected panel with injected or live scorers and build the artifact."""

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
        cited_upstream_artifacts=cited_upstream_artifacts,
        panel_sha256=panel.panel_sha256,
        reasoner_error_scores=reasoner.error_scores,
        carnot_error_scores=carnot.scores,
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
    cited_upstream_artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a non-fabricated blocked artifact with metric fields left empty."""

    artifact: dict[str, object] = {
        "experiment": 3844,
        "title": TITLE,
        "honest_verdict": reason,
        "status": reason,
        "residual_catch_rate": None,
        "residual_catch_ci95": None,
        "error_overlap_jaccard": None,
        "reasoner_self_verify_auroc": None,
        "carnot_ensemble_auroc": None,
        "n_items": 0,
        "n_residual_errors": 0,
        "n_gold_incorrect": 0,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "cited_upstream_artifacts": cited_upstream_artifacts or {},
        "model_specs": model_specs or {},
        "random_seed": None,
        "random_seeds_used": {},
        "reproducibility_checksum": _checksum(
            {
                "experiment": 3844,
                "reason": reason,
                "preconditions_checked": [check.as_dict() for check in preconditions_checked],
                "model_specs": model_specs or {},
            }
        ),
        "duration_s": duration_s,
        "inference_substrate": "none_blocked_preflight",
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
    cited_upstream_artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    artifact = build_blocked_artifact(
        reason=reason,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        model_specs=model_specs,
        cited_upstream_artifacts=cited_upstream_artifacts,
    )
    write_artifact(output_path, artifact)
    return artifact


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, object]:
    """Run Exp3844 end to end, or write a blocked artifact on failed preconditions."""

    config = config or ExperimentConfig(repo_root=Path(__file__).resolve().parents[3])
    started = config.start_time()
    preflight = probe_preconditions(config)
    if preflight.blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=preflight.blocked_reason,
            preconditions_checked=preflight.checks,
            duration_s=config.clock() - started,
            model_specs=preflight.model_specs,
            cited_upstream_artifacts=preflight.cited_upstream_artifacts,
        )
        if write:
            write_artifact(config.resolved_output_path(), artifact)
        return artifact

    panel = select_balanced_panel(
        preflight.rows,
        seed=config.random_seed,
        n_items=config.n_items,
    )
    return build_artifact_for_panel(
        panel,
        config=config,
        preconditions_checked=preflight.checks,
        model_specs=preflight.model_specs,
        cited_upstream_artifacts=preflight.cited_upstream_artifacts,
        write=write,
    )


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate the required Exp3844 schema and terminal verdict discipline."""

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
        raise ValueError("Exp3844 must not apply or record a generation headroom gate")
    if verdict.startswith("complete:") and not str(verdict).startswith(
        "complete: scissor_at_scale_INCONCLUSIVE"
    ):
        if int(artifact["n_items"]) < DEFAULT_N_ITEMS:
            raise ValueError("complete non-inconclusive Exp3844 artifacts require n_items>=1000")
