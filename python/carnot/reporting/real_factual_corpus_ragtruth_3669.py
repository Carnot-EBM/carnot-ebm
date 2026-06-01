"""Build the Exp 3669 real factual hallucination corpus.

Spec refs: REQ-REPORT-3669, SCENARIO-REPORT-3669,
SCENARIO-REPORT-3669-DEGENERATE, SCENARIO-REPORT-3669-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3669_build_real_factual_corpus.json")
CORPUS_REL_PATH = Path("data/real_factual_corpus_ragtruth.jsonl")
RAGTRUTH_CACHE_DIR = Path("data/ragtruth")
RAGTRUTH_RESPONSE_CACHE = RAGTRUTH_CACHE_DIR / "response.jsonl"
RAGTRUTH_SOURCE_CACHE = RAGTRUTH_CACHE_DIR / "source_info.jsonl"
LOCAL_HALUEVAL_V3_PATH = Path("data/realistic_factual_corpus_v3.jsonl")

HF_MODELS_API_URL = "https://huggingface.co/api/models"
RAGTRUTH_RESPONSE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"
)
RAGTRUTH_SOURCE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"
)
RANDOM_SEED = 3669
MIN_EXAMPLES = 200
MIN_CLASS_FRACTION = 0.20
CONFIDENCE_DEGENERATE_AUROC = 0.95

VERDICT_RAGTRUTH_NON_DEGENERATE = (
    "complete: real_factual_corpus_built_ragtruth_non_degenerate"
)
VERDICT_FALLBACK_NON_DEGENERATE = (
    "complete: real_factual_corpus_built_fallback_felm_or_halueval_non_degenerate"
)
VERDICT_DEGENERATE_CONFIDENCE = (
    "complete: blocked_real_corpus_degenerate_confidence_perfect"
)
VERDICT_BLOCKED_NO_CORPUS = "complete: blocked_no_network_and_no_cached_real_corpus"

REQUIRED_CORPUS_FIELDS = (
    "question",
    "answer",
    "is_hallucination",
    "evidence_passage",
    "model_confidence",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "corpus_path",
    "source_benchmark",
    "n_examples",
    "class_balance",
    "confidence_baseline_auroc",
    "corpus_non_degenerate",
    "real_factual_corpus_built",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: builds + baselines a cached corpus; no FoVer LLM load)."
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: builds + "
        "baselines a cached corpus; no FoVer LLM load)."
    ),
    "corpus_path": (
        "Where the real corpus lives -- the provenance for exp3670."
    ),
    "source_benchmark": (
        "RAGTruth / FELM / HaluEval + version -- external-benchmark provenance "
        "(the whole point: real, not synthetic)."
    ),
    "n_examples": (
        "Sample-size rigor (>=200 for a percentage-point AUROC claim downstream)."
    ),
    "class_balance": (
        "Both classes >= 20% -- guards against the imbalance that made the .335 "
        "code corpus fragile."
    ),
    "confidence_baseline_auroc": (
        "The bar the grounding verifier must beat downstream; MUST be < 0.95 "
        "(a perfect confidence baseline = degenerate corpus, exp3574)."
    ),
    "corpus_non_degenerate": (
        "True iff confidence AUROC < 0.95 AND class balance ok -- a degenerate "
        "corpus cannot test the facts hypothesis."
    ),
    "real_factual_corpus_built": (
        "BARE bool. True iff a real, non-degenerate, schema-valid corpus was "
        "persisted. STORE AS BARE true/false -- gates exp3670."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
PLACEHOLDER_RE = re.compile(r"^[RHQ]\d+$")
QUESTION_LINE_RE = re.compile(r"^\s*(?:question|q)\s*:\s*(.+)$", re.IGNORECASE)


JsonDict = dict[str, Any]
ConfidenceFn = Callable[[JsonDict], float]
NetworkChecker = Callable[[], bool]
SourceLoader = Callable[[], "BenchmarkPayload | None"]
TextFetcher = Callable[[str], str]


@dataclass(frozen=True)
class BenchmarkPayload:
    """Raw external benchmark rows before projection to the v3 schema."""

    benchmark: str
    version: str
    responses: list[JsonDict]
    sources: list[JsonDict]
    source_urls: tuple[str, ...] = ()
    from_cache: bool = False


@dataclass(frozen=True)
class CorpusValidation:
    """Validation metrics for the projected real factual corpus."""

    n_examples: int
    n_correct: int
    n_hallucinated: int
    class_balance: dict[str, float]
    confidence_baseline_auroc: float
    class_balance_ok: bool
    confidence_non_degenerate: bool
    placeholder_tokens_rejected: bool
    schema_valid: bool
    corpus_non_degenerate: bool
    degeneracy_reasons: list[str]


def check_hf_models_api() -> bool:  # pragma: no cover - network dependent.
    """Return whether the required Hugging Face API precondition is reachable."""
    try:
        proc = subprocess.run(  # noqa: S603
            ["curl", "-sf", "-o", "/dev/null", HF_MODELS_API_URL],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except Exception:
        return False
    return proc.returncode == 0


def fetch_url_text(url: str) -> str:  # pragma: no cover - network dependent.
    """Fetch UTF-8 text from a public benchmark URL."""
    with urllib.request.urlopen(url, timeout=180) as handle:
        return handle.read().decode("utf-8")


def read_jsonl_text(text: str) -> list[JsonDict]:
    """Parse JSONL text into object rows, ignoring blank lines."""
    rows: list[JsonDict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _read_jsonl_path(path: Path) -> list[JsonDict]:
    return read_jsonl_text(path.read_text(encoding="utf-8"))


def _write_cache(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def default_source_loader(
    repo_root: str | Path = REPO_ROOT,
    *,
    network_ok: bool,
    fetcher: TextFetcher = fetch_url_text,
) -> BenchmarkPayload | None:
    """Load RAGTruth from the network, cached raw files, or a local HaluEval fallback."""
    root = Path(repo_root)
    if network_ok:
        try:
            response_text = fetcher(RAGTRUTH_RESPONSE_URL)
            source_text = fetcher(RAGTRUTH_SOURCE_URL)
        except Exception:
            response_text = ""
            source_text = ""
        if response_text and source_text:
            _write_cache(root / RAGTRUTH_RESPONSE_CACHE, response_text)
            _write_cache(root / RAGTRUTH_SOURCE_CACHE, source_text)
            return BenchmarkPayload(
                benchmark="RAGTruth",
                version="ParticleMedia/RAGTruth main dataset, 2024-02 data update",
                responses=read_jsonl_text(response_text),
                sources=read_jsonl_text(source_text),
                source_urls=(RAGTRUTH_RESPONSE_URL, RAGTRUTH_SOURCE_URL),
                from_cache=False,
            )

    response_cache = root / RAGTRUTH_RESPONSE_CACHE
    source_cache = root / RAGTRUTH_SOURCE_CACHE
    if response_cache.is_file() and source_cache.is_file():
        return BenchmarkPayload(
            benchmark="RAGTruth",
            version="ParticleMedia/RAGTruth cached raw dataset, 2024-02 data update",
            responses=_read_jsonl_path(response_cache),
            sources=_read_jsonl_path(source_cache),
            source_urls=(str(RAGTRUTH_RESPONSE_CACHE), str(RAGTRUTH_SOURCE_CACHE)),
            from_cache=True,
        )

    fallback = root / LOCAL_HALUEVAL_V3_PATH
    if fallback.is_file():
        return BenchmarkPayload(
            benchmark="HaluEval",
            version="local v3 evidence corpus from exp3640",
            responses=_read_jsonl_path(fallback),
            sources=[],
            source_urls=(str(LOCAL_HALUEVAL_V3_PATH),),
            from_cache=True,
        )
    return None


def has_placeholder_token(*values: object) -> bool:
    """Return true for the toy R/H/Q placeholders used by prior poisoned corpora."""
    return any(PLACEHOLDER_RE.match(str(value).strip()) for value in values)


def _content_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def _source_index(sources: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("source_id")): row for row in sources if row.get("source_id") is not None}


def _question_from_prompt(prompt: object) -> str:
    text = str(prompt or "")
    for line in text.splitlines():
        match = QUESTION_LINE_RE.match(line)
        if match:
            return match.group(1).strip()
    return text.strip()


def _question_and_evidence(source_row: Mapping[str, Any]) -> tuple[str, str]:
    source_info = source_row.get("source_info")
    prompt = str(source_row.get("prompt") or "")
    if isinstance(source_info, Mapping):
        question = str(source_info.get("question") or "").strip() or _question_from_prompt(prompt)
        evidence_obj = (
            source_info.get("passages")
            or source_info.get("context")
            or source_info.get("document")
            or source_info
        )
        evidence = (
            evidence_obj
            if isinstance(evidence_obj, str)
            else json.dumps(evidence_obj, sort_keys=True, ensure_ascii=True)
        )
        return question, str(evidence).strip()
    return _question_from_prompt(prompt), str(source_info or "").strip()


def _labels_mark_hallucination(labels: object) -> int:
    if not isinstance(labels, list):
        return 0
    for label in labels:
        if not isinstance(label, Mapping):
            return 1
        if not bool(label.get("implicit_true", False)):
            return 1
    return 0


def default_model_confidence(row: JsonDict) -> float:
    """Return a non-label confidence proxy from generation metadata and answer text."""
    model = str(row.get("model") or "").lower()
    temperature = _coerce_float(row.get("temperature"), 0.7)
    answer = str(row.get("answer") or "")
    if "gpt-4" in model:
        model_prior = 0.78
    elif "gpt-3.5" in model:
        model_prior = 0.68
    elif "70b" in model:
        model_prior = 0.64
    elif "13b" in model:
        model_prior = 0.57
    elif "mistral" in model:
        model_prior = 0.55
    elif "7b" in model:
        model_prior = 0.50
    else:
        model_prior = 0.58
    answer_tokens = _content_tokens(answer)
    length_penalty = min(len(answer_tokens) / 180.0, 1.0)
    hedge_terms = (
        "unclear",
        "likely",
        "may",
        "might",
        "according",
        "suggests",
        "appears",
        "approximately",
        "around",
    )
    hedge_bonus = min(sum(term in answer.lower() for term in hedge_terms) * 0.025, 0.10)
    confidence = 0.15 + 0.70 * model_prior + 0.10 * (1.0 - temperature)
    confidence += hedge_bonus - 0.12 * length_penalty
    return max(0.0, min(1.0, float(confidence)))


def project_ragtruth_payload(
    payload: BenchmarkPayload,
    *,
    confidence_fn: ConfidenceFn = default_model_confidence,
) -> list[JsonDict]:
    """Project RAGTruth-style benchmark rows into the required v3 JSONL schema."""
    source_by_id = _source_index(payload.sources)
    records: list[JsonDict] = []
    for response in payload.responses:
        if set(REQUIRED_CORPUS_FIELDS).issubset(response):
            record = _project_existing_v3_row(response, confidence_fn=confidence_fn)
        else:
            record = _project_response_row(response, source_by_id, confidence_fn=confidence_fn)
        if record is not None:
            records.append(record)
    return records


def _project_existing_v3_row(row: Mapping[str, Any], *, confidence_fn: ConfidenceFn) -> JsonDict | None:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    evidence = str(row.get("evidence_passage") or "").strip()
    if not question or not answer or not evidence:
        return None
    if has_placeholder_token(question, answer, evidence):
        return None
    draft: JsonDict = {
        "question": question,
        "answer": answer,
        "is_hallucination": int(bool(row.get("is_hallucination"))),
        "evidence_passage": evidence,
        "model": row.get("model", "unknown"),
        "temperature": row.get("temperature", 0.7),
    }
    draft["model_confidence"] = _bounded_confidence(confidence_fn(draft))
    return {field: draft[field] for field in REQUIRED_CORPUS_FIELDS}


def _project_response_row(
    response: Mapping[str, Any],
    source_by_id: Mapping[str, Mapping[str, Any]],
    *,
    confidence_fn: ConfidenceFn,
) -> JsonDict | None:
    if response.get("quality") and str(response.get("quality")) != "good":
        return None
    source = source_by_id.get(str(response.get("source_id")))
    if source is None:
        return None
    question, evidence = _question_and_evidence(source)
    answer = str(response.get("response") or "").strip()
    if not question or not answer or not evidence:
        return None
    if has_placeholder_token(question, answer, evidence):
        return None
    draft: JsonDict = {
        "question": question,
        "answer": answer,
        "is_hallucination": _labels_mark_hallucination(response.get("labels")),
        "evidence_passage": evidence,
        "model": response.get("model", "unknown"),
        "temperature": response.get("temperature", 0.7),
        "source_id": response.get("source_id"),
        "split": response.get("split"),
    }
    draft["model_confidence"] = _bounded_confidence(confidence_fn(draft))
    return {field: draft[field] for field in REQUIRED_CORPUS_FIELDS}


def _bounded_confidence(value: object) -> float:
    score = _coerce_float(value, 0.5)
    return max(0.0, min(1.0, float(score)))


def _coerce_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute tie-aware AUROC for label 1 as the positive class."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(1 for label in labels if int(label) == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranked = sorted((float(score), int(label)) for label, score in zip(labels, scores, strict=True))
    rank_sum_positive = 0.0
    rank = 1
    idx = 0
    while idx < len(ranked):
        end = idx + 1
        while end < len(ranked) and ranked[end][0] == ranked[idx][0]:
            end += 1
        avg_rank = (rank + end) / 2.0
        rank_sum_positive += sum(avg_rank for _, label in ranked[idx:end] if label == 1)
        rank += end - idx
        idx = end
    auc = (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)
    return round(float(auc), 6)


def validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    min_examples: int = MIN_EXAMPLES,
) -> CorpusValidation:
    """Validate schema, class balance, placeholders, and confidence non-degeneracy."""
    n_examples = len(records)
    n_hallucinated = sum(1 for row in records if int(row.get("is_hallucination", 0)) == 1)
    n_correct = sum(1 for row in records if int(row.get("is_hallucination", 0)) == 0)
    class_balance = {
        "correct": round(n_correct / n_examples, 6) if n_examples else 0.0,
        "hallucinated": round(n_hallucinated / n_examples, 6) if n_examples else 0.0,
    }
    schema_valid = bool(
        n_examples >= min_examples
        and all(set(row.keys()) == set(REQUIRED_CORPUS_FIELDS) for row in records)
        and all(str(row.get("question") or "").strip() for row in records)
        and all(str(row.get("answer") or "").strip() for row in records)
        and all(str(row.get("evidence_passage") or "").strip() for row in records)
    )
    placeholder_tokens_rejected = not any(
        has_placeholder_token(
            row.get("question", ""),
            row.get("answer", ""),
            row.get("evidence_passage", ""),
        )
        for row in records
    )
    class_balance_ok = (
        class_balance["correct"] >= MIN_CLASS_FRACTION
        and class_balance["hallucinated"] >= MIN_CLASS_FRACTION
    )
    correctness_labels = [1 - int(row.get("is_hallucination", 0)) for row in records]
    confidence_scores = [float(row.get("model_confidence", 0.5)) for row in records]
    confidence_auroc = binary_auroc(correctness_labels, confidence_scores)
    confidence_non_degenerate = confidence_auroc < CONFIDENCE_DEGENERATE_AUROC
    reasons = []
    if n_examples < min_examples:
        reasons.append("n_examples_below_200")
    if not class_balance_ok:
        reasons.append("class_balance_below_20_percent")
    if not confidence_non_degenerate:
        reasons.append("confidence_auroc_at_or_above_0.95")
    if not schema_valid:
        reasons.append("schema_invalid_or_missing_text")
    if not placeholder_tokens_rejected:
        reasons.append("placeholder_tokens_present")
    corpus_non_degenerate = bool(
        schema_valid
        and placeholder_tokens_rejected
        and class_balance_ok
        and confidence_non_degenerate
    )
    return CorpusValidation(
        n_examples=n_examples,
        n_correct=n_correct,
        n_hallucinated=n_hallucinated,
        class_balance=class_balance,
        confidence_baseline_auroc=confidence_auroc,
        class_balance_ok=class_balance_ok,
        confidence_non_degenerate=confidence_non_degenerate,
        placeholder_tokens_rejected=placeholder_tokens_rejected,
        schema_valid=schema_valid,
        corpus_non_degenerate=corpus_non_degenerate,
        degeneracy_reasons=reasons,
    )


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Persist projected corpus rows with stable JSONL encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in records),
        encoding="utf-8",
    )


def sample_examples(records: Sequence[Mapping[str, Any]], *, limit: int = 3) -> list[JsonDict]:
    """Return short real-text examples for operator sanity checks."""
    examples = []
    for row in records[:limit]:
        examples.append(
            {
                "question": _truncate(str(row.get("question") or ""), 180),
                "answer": _truncate(str(row.get("answer") or ""), 220),
                "is_hallucination": int(row.get("is_hallucination", 0)),
                "evidence_passage": _truncate(str(row.get("evidence_passage") or ""), 260),
                "model_confidence": float(row.get("model_confidence", 0.0)),
            }
        )
    return examples


def _truncate(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: max(0, limit - 3)].rstrip() + "..."


def _source_benchmark(payload: BenchmarkPayload | None) -> str:
    if payload is None:
        return "none_available"
    cache_note = "cached" if payload.from_cache else "fetched"
    return f"{payload.benchmark} {payload.version} ({cache_note})"


def _terminal_verdict(payload: BenchmarkPayload | None, validation: CorpusValidation) -> str:
    if payload is None or validation.n_examples == 0:
        return VERDICT_BLOCKED_NO_CORPUS
    if not validation.corpus_non_degenerate:
        return VERDICT_DEGENERATE_CONFIDENCE
    if payload.benchmark.lower() == "ragtruth":
        return VERDICT_RAGTRUTH_NON_DEGENERATE
    return VERDICT_FALLBACK_NON_DEGENERATE


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Return a stable short checksum for drift detection."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _artifact(
    *,
    payload: BenchmarkPayload | None,
    validation: CorpusValidation,
    duration_s: float,
    checksum: str,
    records: Sequence[Mapping[str, Any]],
) -> JsonDict:
    real_built = bool(validation.corpus_non_degenerate and validation.n_examples >= MIN_EXAMPLES)
    verdict = _terminal_verdict(payload, validation)
    artifact: JsonDict = {
        "honest_verdict": verdict,
        "honest_outcome": (
            "blocked"
            if verdict == VERDICT_BLOCKED_NO_CORPUS
            else (
                "degenerate_confidence_perfect"
                if verdict == VERDICT_DEGENERATE_CONFIDENCE
                else "corpus_built"
            )
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "corpus_path": str(CORPUS_REL_PATH),
        "source_benchmark": _source_benchmark(payload),
        "source_urls": list(payload.source_urls if payload is not None else []),
        "n_examples": validation.n_examples,
        "n_correct": validation.n_correct,
        "n_hallucinated": validation.n_hallucinated,
        "class_balance": validation.class_balance,
        "class_balance_ok": validation.class_balance_ok,
        "confidence_baseline_auroc": validation.confidence_baseline_auroc,
        "corpus_non_degenerate": validation.corpus_non_degenerate,
        "real_factual_corpus_built": real_built,
        "schema_valid": validation.schema_valid,
        "placeholder_tokens_rejected": validation.placeholder_tokens_rejected,
        "degeneracy_reasons": list(validation.degeneracy_reasons),
        "sample_examples": sample_examples(records),
        "acceptance_gate": {
            "condition": (
                "real_factual_corpus_built == true AND corpus_non_degenerate == true "
                "AND n_examples >= 200"
            ),
            "passed": bool(real_built and validation.corpus_non_degenerate),
            "principle": (
                "A facts re-measurement is only meaningful on a real, non-degenerate, "
                "adequately-sized corpus -- a degenerate or tiny corpus repeats the "
                ".335/.330 limitation."
            ),
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": checksum,
        "duration_s": round(max(0.0, float(duration_s)), 6),
    }
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise ValueError when the Exp 3669 artifact contract is violated."""
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    for key in ("real_factual_corpus_built", "corpus_non_degenerate"):
        if type(artifact.get(key)) is not bool:
            raise ValueError(f"{key} must be a bare boolean")
    if not str(artifact.get("honest_verdict", "")).startswith("complete: "):
        raise ValueError("honest_verdict must start with 'complete: '")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(principles)
    if missing_principles:
        raise ValueError(f"field_principles missing fields: {sorted(missing_principles)}")
    if int(artifact.get("n_examples", 0)) < 0:
        raise ValueError("n_examples must be non-negative")
    confidence = _coerce_float(artifact.get("confidence_baseline_auroc"), -1.0)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence_baseline_auroc must be in [0, 1]")
    if _coerce_float(artifact.get("duration_s"), -1.0) < 0.0:
        raise ValueError("duration_s must be non-negative")


def _blocked_validation() -> CorpusValidation:
    return CorpusValidation(
        n_examples=0,
        n_correct=0,
        n_hallucinated=0,
        class_balance={"correct": 0.0, "hallucinated": 0.0},
        confidence_baseline_auroc=0.0,
        class_balance_ok=False,
        confidence_non_degenerate=False,
        placeholder_tokens_rejected=True,
        schema_valid=False,
        corpus_non_degenerate=False,
        degeneracy_reasons=["no_network_and_no_cached_real_corpus"],
    )


def build_artifact(
    repo_root: str | Path = REPO_ROOT,
    *,
    source_loader: SourceLoader | None = None,
    network_checker: NetworkChecker = check_hf_models_api,
    confidence_fn: ConfidenceFn = default_model_confidence,
    duration_s: float | None = None,
) -> JsonDict:
    """Build and persist the Exp 3669 artifact and projected corpus."""
    started = time.perf_counter()
    root = Path(repo_root)
    network_ok = bool(network_checker())
    payload = (
        source_loader()
        if source_loader is not None
        else default_source_loader(root, network_ok=network_ok)
    )
    if payload is None:
        records: list[JsonDict] = []
        validation = _blocked_validation()
    else:
        records = project_ragtruth_payload(payload, confidence_fn=confidence_fn)
        validation = validate_records(records)
        if records:
            write_jsonl(root / CORPUS_REL_PATH, records)
    elapsed = duration_s if duration_s is not None else time.perf_counter() - started
    checksum = reproducibility_checksum(
        {
            "source_benchmark": _source_benchmark(payload),
            "n_examples": validation.n_examples,
            "class_balance": validation.class_balance,
            "confidence_baseline_auroc": validation.confidence_baseline_auroc,
            "records_sha256": hashlib.sha256(
                "".join(
                    json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in records
                ).encode("utf-8")
            ).hexdigest(),
            "random_seed": RANDOM_SEED,
        }
    )
    artifact = _artifact(
        payload=payload,
        validation=validation,
        duration_s=float(elapsed),
        checksum=checksum,
        records=records,
    )
    validate_artifact(artifact)
    output = root / OUTPUT_REL_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run_experiment(
    repo_root: str | Path = REPO_ROOT,
    *,
    source_loader: SourceLoader | None = None,
    network_checker: NetworkChecker = check_hf_models_api,
    confidence_fn: ConfidenceFn = default_model_confidence,
    duration_s: float | None = None,
) -> Path:
    """Run Exp 3669 and return the artifact path."""
    root = Path(repo_root)
    build_artifact(
        root,
        source_loader=source_loader,
        network_checker=network_checker,
        confidence_fn=confidence_fn,
        duration_s=duration_s,
    )
    return root / OUTPUT_REL_PATH


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = build_artifact(REPO_ROOT)
    for idx, example in enumerate(artifact.get("sample_examples", []), start=1):
        print(
            f"example {idx}: hallucination={example['is_hallucination']} "
            f"confidence={example['model_confidence']:.3f} "
            f"answer={example['answer']}"
        )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0
