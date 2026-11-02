#!/usr/bin/env python3
"""Build Exp 3640 factual corpus v3 from real evidence-bearing labels.

Spec: REQ-REPORT-3640,
      SCENARIO-REPORT-3640,
      SCENARIO-REPORT-3640-DEGENERATE,
      SCENARIO-REPORT-3640-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - import-environment dependent.
    sys.path.insert(0, str(ROOT))
if str(ROOT / "python") not in sys.path:  # pragma: no cover - import-environment dependent.
    sys.path.insert(0, str(ROOT / "python"))
if str(ROOT / "scripts") not in sys.path:  # pragma: no cover - import-environment dependent.
    sys.path.insert(0, str(ROOT / "scripts"))


CORPUS_PATH = Path("data/realistic_factual_corpus_v3.jsonl")
RESULT_PATH = Path("results/experiment_3640_build_factual_corpus_v3.json")
LOCAL_HALUEVAL_MANIFEST = Path("data/eval_manifests/halueval_20260522.jsonl")
HALUEVAL_QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
HF_DATASETS_API_URL = "https://huggingface.co/api/datasets"
HALUEVAL_HF_URL = "https://huggingface.co/datasets/pminervini/HaluEval"
DEFAULT_RANDOM_SEED = 3640
DEFAULT_MAX_SOURCE_PAIRS = 250
DEFAULT_MIN_RECORDS = 200

VERDICT_VALIDATED = (
    "complete: factual_corpus_v3_built_real_evidence_dataset_confidence_headroom_"
    "confirmed_bare_fields_emitted"
)
VERDICT_DEGENERATE = (
    "complete: factual_corpus_v3_built_but_confidence_out_of_band_facts_row_degenerate"
)
VERDICT_BLOCKED = "complete: blocked_no_evidence_dataset"

CORPUS_RECORD_FIELDS = (
    "question",
    "answer",
    "is_hallucination",
    "evidence_passage",
    "model_confidence",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "corpus_path_used",
    "corpus_source",
    "n_examples",
    "n_hallucinated",
    "n_correct",
    "confidence_baseline_auroc_on_corpus",
    "evidence_independent_of_label",
    "placeholder_tokens_rejected",
    "facts_corpus_has_evidence",
    "facts_corpus_validated",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "Fetches + validates a labeled dataset; no live LLM generation in the headline path."
    ),
    "corpus_path_used": "Which corpus downstream rows must score (v3).",
    "corpus_source": "Dataset, split, evidence field, and URL provenance.",
    "n_examples": "Sample-size rigor; >=200 for a percentage-point AUROC claim downstream.",
    "n_hallucinated": "Class balance.",
    "n_correct": "Class balance.",
    "confidence_baseline_auroc_on_corpus": (
        "Must be in (0.5,0.95) so confidence has signal but verifiers have headroom."
    ),
    "evidence_independent_of_label": "True only when evidence is paired across both labels.",
    "placeholder_tokens_rejected": "Confirms toy R/H/Q placeholder rows are absent.",
    "facts_corpus_has_evidence": "BARE bool gate for grounding-verifier path.",
    "facts_corpus_validated": "BARE bool final corpus gate.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection across replications.",
    "duration_s": "Plausibility floor.",
}

PLACEHOLDER_RE = re.compile(r"^[RHQ]\d+$")
LABEL_LEAK_RE = re.compile(
    r"\b(?:is_hallucination|hallucinated_answer|right_answer|gold_answer)\b|\blabel\s*[:=]",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CorpusBuildConfig:
    """Runtime settings for the Exp 3640 corpus builder."""

    repo_root: Path = ROOT
    random_seed: int = DEFAULT_RANDOM_SEED
    max_source_pairs: int = DEFAULT_MAX_SOURCE_PAIRS
    min_records: int = DEFAULT_MIN_RECORDS
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


@dataclass(frozen=True)
class CorpusValidation:
    """Validation summary for the built factual corpus."""

    n_examples: int
    n_hallucinated: int
    n_correct: int
    confidence_baseline_auroc_on_corpus: float
    evidence_independent_of_label: bool
    placeholder_tokens_rejected: bool
    facts_corpus_has_evidence: bool
    facts_corpus_validated: bool


SourceLoader = Callable[[], list[dict[str, str]]]
NetworkChecker = Callable[[], bool]
ConfidenceFn = Callable[[str, str], float]


def check_hf_datasets_api() -> bool:  # pragma: no cover - network dependent.
    """Return whether the Hugging Face datasets API precondition is reachable."""

    proc = subprocess.run(
        ["curl", "-sf", "-o", "/dev/null", HF_DATASETS_API_URL],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    return proc.returncode == 0


def fetch_halueval_qa_rows(max_source_pairs: int) -> list[dict[str, str]]:  # pragma: no cover
    """Fetch HaluEval QA rows from the public source mirror without pyarrow."""

    rows: list[dict[str, str]] = []
    with urllib.request.urlopen(HALUEVAL_QA_URL, timeout=60) as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line.decode("utf-8"))
            rows.append(
                {
                    "knowledge": str(row.get("knowledge") or ""),
                    "question": str(row.get("question") or ""),
                    "right_answer": str(row.get("right_answer") or ""),
                    "hallucinated_answer": str(row.get("hallucinated_answer") or ""),
                }
            )
            if len(rows) >= max_source_pairs:
                break
    return rows


def _split_prompt(prompt: str) -> tuple[str, str]:
    context, marker, question = prompt.partition("Question:")
    if not marker:
        return prompt.strip(), ""
    context = context.strip()
    if context.startswith("Context:"):
        context = context[len("Context:") :].strip()
    return context, question.strip()


def load_local_manifest_rows(repo_root: Path) -> list[dict[str, str]]:
    """Load paired HaluEval rows from the checked-in local manifest when present."""

    path = repo_root / LOCAL_HALUEVAL_MANIFEST
    if not path.is_file():
        return []
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or "")
            evidence, question = _split_prompt(prompt)
            if not evidence or not question:
                continue
            key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            grouped[key]["knowledge"] = evidence
            grouped[key]["question"] = question
            label = int(row.get("label"))
            candidate = str(row.get("candidate") or "")
            if label == 0:
                grouped[key]["right_answer"] = candidate
            elif label == 1:
                grouped[key]["hallucinated_answer"] = candidate
    return [
        row
        for row in grouped.values()
        if row.get("knowledge")
        and row.get("question")
        and row.get("right_answer")
        and row.get("hallucinated_answer")
    ]


def default_source_loader(config: CorpusBuildConfig, network_ok: bool) -> list[dict[str, str]]:
    """Prefer freshly fetched HaluEval QA rows, then fall back to cached manifests."""

    if network_ok:
        try:
            fetched = fetch_halueval_qa_rows(config.max_source_pairs)
        except Exception:
            fetched = []
        if fetched:
            return fetched
    return load_local_manifest_rows(config.repo_root)[: config.max_source_pairs]


def has_placeholder_token(*values: object) -> bool:
    """Return true for the toy R/H/Q placeholders that polluted prior corpora."""

    return any(PLACEHOLDER_RE.match(str(value).strip()) for value in values)


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def default_model_confidence(question: str, answer: str) -> float:
    """Compute a non-grounding answer-surface confidence proxy.

    The score intentionally uses only the question and candidate answer, never
    the evidence passage or label. It is a weak confidence baseline, not a
    grounding verifier.
    """

    answer_tokens = _tokens(answer)
    question_tokens = set(_tokens(question))
    if not answer_tokens:
        return 0.0
    concise = 1.0 / (1.0 + len(answer_tokens))
    question_overlap = sum(1 for token in answer_tokens if token in question_tokens) / len(
        answer_tokens
    )
    confidence = 0.15 + 0.65 * concise + 0.20 * question_overlap
    return max(0.0, min(1.0, float(confidence)))


def build_corpus_records(
    source_rows: Sequence[dict[str, str]],
    *,
    max_source_pairs: int,
    confidence_fn: ConfidenceFn,
) -> list[dict[str, Any]]:
    """Convert HaluEval-style paired source rows into v3 JSONL records."""

    records: list[dict[str, Any]] = []
    for row in source_rows[:max_source_pairs]:
        question = str(row.get("question") or "").strip()
        evidence = str(row.get("knowledge") or "").strip()
        right = str(row.get("right_answer") or "").strip()
        hallucinated = str(row.get("hallucinated_answer") or "").strip()
        if not question or not evidence or not right or not hallucinated:
            continue
        if has_placeholder_token(question, evidence, right, hallucinated):
            continue
        for is_hallucination, answer in ((0, right), (1, hallucinated)):
            records.append(
                {
                    "question": question,
                    "answer": answer,
                    "is_hallucination": is_hallucination,
                    "evidence_passage": evidence,
                    "model_confidence": float(confidence_fn(question, answer)),
                }
            )
    return records


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC for label 1 as the positive class."""

    positives = [float(score) for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [float(score) for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        wins += sum(1.0 for negative in negatives if positive > negative)
        wins += 0.5 * sum(1.0 for negative in negatives if positive == negative)
    return wins / (len(positives) * len(negatives))


def _evidence_is_independent(records: Sequence[dict[str, Any]], has_evidence: bool) -> bool:
    if not has_evidence:
        return False
    label_by_question_evidence: dict[tuple[str, str], set[int]] = defaultdict(set)
    for record in records:
        evidence = str(record.get("evidence_passage") or "")
        if LABEL_LEAK_RE.search(evidence):
            return False
        key = (str(record.get("question") or ""), evidence)
        label_by_question_evidence[key].add(int(record.get("is_hallucination")))
    return bool(label_by_question_evidence) and all(
        labels == {0, 1} for labels in label_by_question_evidence.values()
    )


def validate_corpus_records(records: Sequence[dict[str, Any]], *, min_records: int) -> CorpusValidation:
    """Validate sample size, class balance, evidence pairing, placeholders, and AUROC."""

    n_examples = len(records)
    n_hallucinated = sum(1 for record in records if int(record.get("is_hallucination")) == 1)
    n_correct = sum(1 for record in records if int(record.get("is_hallucination")) == 0)
    placeholder_tokens_rejected = not any(
        has_placeholder_token(
            record.get("question", ""),
            record.get("answer", ""),
            record.get("evidence_passage", ""),
        )
        for record in records
    )
    evidence_by_label = {0: 0, 1: 0}
    for record in records:
        evidence = str(record.get("evidence_passage") or "").strip()
        if evidence:
            evidence_by_label[int(record.get("is_hallucination"))] += 1
    facts_corpus_has_evidence = (
        n_examples > 0
        and all(str(record.get("evidence_passage") or "").strip() for record in records)
        and all(count > 0 for count in evidence_by_label.values())
    )
    evidence_independent_of_label = _evidence_is_independent(records, facts_corpus_has_evidence)
    correctness_labels = [1 - int(record.get("is_hallucination")) for record in records]
    confidence_scores = [float(record.get("model_confidence") or 0.0) for record in records]
    auroc = binary_auroc(correctness_labels, confidence_scores)
    facts_corpus_validated = bool(
        n_examples >= min_records
        and n_hallucinated > 0
        and n_correct > 0
        and 0.50 < auroc < 0.95
        and evidence_independent_of_label
        and placeholder_tokens_rejected
    )
    return CorpusValidation(
        n_examples=n_examples,
        n_hallucinated=n_hallucinated,
        n_correct=n_correct,
        confidence_baseline_auroc_on_corpus=float(auroc),
        evidence_independent_of_label=bool(evidence_independent_of_label),
        placeholder_tokens_rejected=bool(placeholder_tokens_rejected),
        facts_corpus_has_evidence=bool(facts_corpus_has_evidence),
        facts_corpus_validated=bool(facts_corpus_validated),
    )


def write_corpus(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Persist the v3 corpus with a stable JSONL encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True, ensure_ascii=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _blocked_validation() -> CorpusValidation:
    return CorpusValidation(
        n_examples=0,
        n_hallucinated=0,
        n_correct=0,
        confidence_baseline_auroc_on_corpus=0.0,
        evidence_independent_of_label=False,
        placeholder_tokens_rejected=True,
        facts_corpus_has_evidence=False,
        facts_corpus_validated=False,
    )


def _artifact(
    *,
    verdict: str,
    validation: CorpusValidation,
    corpus_source: str,
    corpus_path: Path,
    random_seed: int,
    duration_s: float,
    checksum: str,
) -> dict[str, Any]:
    payload = {
        "honest_verdict": verdict,
        "inference_substrate": (
            "aggregation_from_upstream_artifacts (principle: fetches + validates a "
            "labeled dataset; no live LLM generation in the headline path)."
        ),
        "corpus_path_used": str(corpus_path),
        "corpus_source": corpus_source,
        "n_examples": validation.n_examples,
        "n_hallucinated": validation.n_hallucinated,
        "n_correct": validation.n_correct,
        "confidence_baseline_auroc_on_corpus": validation.confidence_baseline_auroc_on_corpus,
        "evidence_independent_of_label": validation.evidence_independent_of_label,
        "placeholder_tokens_rejected": validation.placeholder_tokens_rejected,
        "facts_corpus_has_evidence": validation.facts_corpus_has_evidence,
        "facts_corpus_validated": validation.facts_corpus_validated,
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "duration_s": max(0.0, float(duration_s)) if math.isfinite(float(duration_s)) else 0.0,
        "confidence_signal": (
            "answer_surface_confidence_from_candidate_and_question_only_no_evidence_or_label"
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(payload)
    return payload


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compute_repro_checksum(seed: int, code_files: Sequence[str], data_path: str | None) -> str:
    digest = hashlib.sha256()
    digest.update(seed.to_bytes(8, "big"))
    for code_file in code_files:
        path = Path(code_file)
        if path.is_file():
            digest.update(path.read_bytes())
    if data_path:
        path = Path(data_path)
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _checksum(config: CorpusBuildConfig, corpus_abs_path: Path) -> str:
    data_path = str(corpus_abs_path) if corpus_abs_path.is_file() else None
    return _compute_repro_checksum(config.random_seed, [__file__], data_path)


def run_experiment(
    *,
    config: CorpusBuildConfig | None = None,
    source_loader: SourceLoader | None = None,
    network_checker: NetworkChecker | None = None,
    confidence_fn: ConfidenceFn = default_model_confidence,
) -> dict[str, Any]:
    """Build the corpus and write the Exp 3640 terminal artifact."""

    config = config or CorpusBuildConfig()
    started = config.start_time()
    network_ok = (network_checker or check_hf_datasets_api)()
    loader = source_loader or (lambda: default_source_loader(config, network_ok))
    source_rows = loader()
    corpus_abs_path = config.repo_root / CORPUS_PATH
    result_abs_path = config.repo_root / RESULT_PATH

    if not source_rows:
        validation = _blocked_validation()
        artifact = _artifact(
            verdict=VERDICT_BLOCKED,
            validation=validation,
            corpus_source="none_available: checked Hugging Face API and local HaluEval manifest",
            corpus_path=CORPUS_PATH,
            random_seed=config.random_seed,
            duration_s=config.clock() - started,
            checksum=_checksum(config, corpus_abs_path),
        )
        write_artifact(result_abs_path, artifact)
        return artifact

    records = build_corpus_records(
        source_rows,
        max_source_pairs=config.max_source_pairs,
        confidence_fn=confidence_fn,
    )
    if records:
        write_corpus(corpus_abs_path, records)
    validation = validate_corpus_records(records, min_records=config.min_records)
    verdict = VERDICT_VALIDATED if validation.facts_corpus_validated else VERDICT_DEGENERATE
    artifact = _artifact(
        verdict=verdict,
        validation=validation,
        corpus_source=(
            "HaluEval QA split=data; evidence_field=knowledge; "
            "answer_fields=right_answer/hallucinated_answer; "
            f"dataset_url={HALUEVAL_HF_URL}; source_url={HALUEVAL_QA_URL}"
        ),
        corpus_path=CORPUS_PATH,
        random_seed=config.random_seed,
        duration_s=config.clock() - started,
        checksum=_checksum(config, corpus_abs_path),
    )
    write_artifact(result_abs_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment()
    print(json.dumps({"honest_verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
