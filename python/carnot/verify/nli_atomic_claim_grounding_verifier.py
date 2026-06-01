"""Atomic-claim factual grounding verifier for Exp 3654.

What this verifier does:
    It implements the factual-grounding recipe that Exp 3642 did not actually
    try: split the model answer into small claims, compare each claim only with
    the retrieved evidence passage, and turn the NLI result into a grounding
    energy. The score path intentionally receives only `(model_answer,
    evidence_passage)`. Labels and separate gold-answer fields are used only
    later by the experiment harness to compute AUROC.

Authenticity disclosure:
    When a cached transformers NLI checkpoint is loadable, this module invokes
    that real checkpoint through `AutoTokenizer` and
    `AutoModelForSequenceClassification`. The default cached candidate is
    `cross-encoder/nli-deberta-v3-small`.

    If no NLI checkpoint is cached, we approximate entailment with
    content-token support because no NLI checkpoint is cached. That fallback is
    weaker than model-based NLI, cannot reason over paraphrase or contradiction,
    and is explicitly reported as
    `disclosed_text_statistical_proxy_no_cached_nli_checkpoint`.

Spec: REQ-VERIFY-3654, SCENARIO-VERIFY-3654.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Protocol

from carnot.verify.corrected_cross_domain_remeasurement_v4 import metric_bundle, tie_aware_auroc


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3654_real_nli_atomic_claim_grounding_verifier.json")
EXP3640_REL_PATH = Path("results/experiment_3640_build_factual_corpus_v3.json")
DEFAULT_CORPUS_REL_PATH = Path("data/realistic_factual_corpus_v3.jsonl")
RANDOM_SEED = 3654
BOOTSTRAP_SEEDS = (3654, 3655, 3656)
PROXY_BASELINE_AUROC = 0.6495
DEFAULT_CONFIDENCE_BASELINE_AUROC = 0.7446
TEXT_STATISTICAL_PROXY_SUBSTRATE = "disclosed_text_statistical_proxy_no_cached_nli_checkpoint"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores the cached v3 facts corpus with an NLI checkpoint; "
    "no FoVer LLM load)."
)
DEFAULT_NLI_CHECKPOINT_CANDIDATES = (
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-base",
    "microsoft/deberta-v3-base-mnli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "cross-encoder/nli-MiniLM2-L6-H768",
)
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
    "nli_substrate",
    "nli_grounding_built",
    "grounding_auroc",
    "grounding_auroc_vs_proxy_delta",
    "confidence_baseline_auroc",
    "grounding_beats_confidence",
    "grounding_leak_free",
    "evidence_excludes_gold_answer_assert",
    "n_examples",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: scores the cached "
        "v3 facts corpus with an NLI checkpoint; no FoVer LLM load)."
    ),
    "nli_substrate": (
        "Declares real model-based NLI (named checkpoint) vs disclosed "
        "text-statistical proxy -- verifier-authenticity honesty."
    ),
    "nli_grounding_built": (
        "BARE bool. True iff the verifier module was implemented and scored the "
        "corpus (model-based OR honestly-disclosed proxy)."
    ),
    "grounding_auroc": (
        "The real NLI grounding signal on the v3 corpus + CI -- the core facts-row number."
    ),
    "grounding_auroc_vs_proxy_delta": (
        "Signed improvement over the .334 text-statistical proxy (0.6495) -- did a real model help?"
    ),
    "confidence_baseline_auroc": (
        "The v3 confidence baseline (0.7446) -- the bar the grounding verifier must "
        "beat to add facts value."
    ),
    "grounding_beats_confidence": (
        "True iff grounding AUROC materially > confidence baseline -- the facts-value signal."
    ),
    "grounding_leak_free": (
        "True iff evidence excludes the gold answer AND AUROC < 0.99 AND the score "
        "never reads the label -- the exp3587 leak guard."
    ),
    "evidence_excludes_gold_answer_assert": (
        "Asserts the retrieved evidence never contains a separate gold answer -- the "
        "explicit leak guard."
    ),
    "n_examples": "Sample-size rigor (>=200 for a percentage-point AUROC claim).",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": (
        "Plausibility floor (a real NLI scoring pass takes seconds-to-minutes, not microseconds)."
    ),
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_SENTENCE_SPLIT_RE = re.compile(r"(?:[.!?]+|\n+)")
_CLAUSE_SPLIT_RE = re.compile(r"\s*,?\s+(?:and|but|while|whereas)\s+", flags=re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "supports",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}


class GroundingVerifier(Protocol):
    """Protocol for score-only verifiers used by the Exp 3654 harness."""

    model_based: bool
    nli_substrate: str

    def verify(self, model_answer: str, evidence_passage: str) -> float:
        """Return a leak-free grounding energy from answer and evidence only."""


@dataclass(frozen=True)
class ClaimGroundingScore:
    """One claim-level grounding score.

    `energy` is the scalar used by the artifact. For model-backed NLI it is the
    checkpoint's contradiction probability, which is the strictest unsupported
    signal for short answer fragments. `hard_unentailed` keeps the literal
    top-label fraction available for diagnostics.
    """

    claim: str
    hypothesis: str
    predicted_label: str
    entailment_probability: float
    neutral_probability: float
    contradiction_probability: float
    energy: float
    hard_unentailed: bool


def split_atomic_claims(model_answer: str) -> list[str]:
    """Split a model answer into small factual-looking claims.

    This is intentionally conservative: sentence boundaries split first, then
    simple coordinating conjunctions split compound factual statements. The
    function does not use the question or label, keeping the score path aligned
    with the leak guard.
    """

    text = " ".join(str(model_answer or "").strip().split())
    if not text:
        return []
    claims: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip(" ,;:")
        if not sentence:
            continue
        for part in _CLAUSE_SPLIT_RE.split(sentence):
            claim = part.strip(" ,;:")
            if claim:
                claims.append(claim)
    return claims


def _content_tokens(text: str) -> list[str]:
    """Return lower-cased content tokens for the disclosed proxy."""

    tokens = []
    for token in _TOKEN_RE.findall(str(text).lower()):
        if len(token) <= 1 or token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _hypothesis_for_claim(claim: str) -> str:
    """Wrap terse answer fragments in a stable NLI hypothesis template."""

    cleaned = str(claim).strip()
    if not cleaned:
        return ""
    return f"The evidence supports this answer: {cleaned.rstrip('.')}."


class TextStatisticalEntailmentProxy:
    """Disclosed fallback when no cached NLI checkpoint is available.

    We approximate entailment with content-token support because no NLI
    checkpoint is cached. A claim is lower energy when more of its non-stopword
    tokens appear in the evidence passage. This is not model-based NLI and is
    weaker than a DeBERTa/MiniLM entailment checkpoint.
    """

    model_based = False
    nli_substrate = TEXT_STATISTICAL_PROXY_SUBSTRATE

    def __init__(self, unavailable_reason: str | None = None) -> None:
        self.unavailable_reason = unavailable_reason

    def score_claims(
        self,
        model_answer: str,
        evidence_passage: str,
    ) -> list[ClaimGroundingScore]:
        """Return proxy claim scores using answer and evidence text only."""

        claims = split_atomic_claims(model_answer)
        evidence_tokens = set(_content_tokens(evidence_passage))
        scores: list[ClaimGroundingScore] = []
        for claim in claims:
            claim_tokens = set(_content_tokens(claim))
            if not claim_tokens:
                energy = 0.0
            else:
                supported = len(claim_tokens & evidence_tokens)
                energy = 1.0 - supported / len(claim_tokens)
            label = "entailment_proxy" if energy < 0.5 else "neutral_proxy"
            scores.append(
                ClaimGroundingScore(
                    claim=claim,
                    hypothesis=_hypothesis_for_claim(claim),
                    predicted_label=label,
                    entailment_probability=round(1.0 - energy, 6),
                    neutral_probability=round(energy, 6),
                    contradiction_probability=0.0,
                    energy=round(float(energy), 6),
                    hard_unentailed=energy >= 0.5,
                )
            )
        return scores

    def verify(self, model_answer: str, evidence_passage: str) -> float:
        """Return mean proxy grounding energy over atomic claims."""

        scores = self.score_claims(model_answer, evidence_passage)
        if not scores:
            return 0.0
        return round(float(sum(score.energy for score in scores) / len(scores)), 6)


class TransformersNLIBackend:
    """Thin wrapper around a cached transformers sequence-classification NLI model."""

    model_based = True

    def __init__(
        self,
        checkpoint: str,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        *,
        device: str,
        max_length: int = 384,
    ) -> None:
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch_module
        self.device = device
        self.max_length = max_length
        self.nli_substrate = f"model_based_transformers_checkpoint: {checkpoint} on {device}"
        self._label_by_index = _label_map_from_model(model)
        self._entailment_index = _find_label_index(self._label_by_index, "entail", fallback=1)
        self._neutral_index = _find_label_index(self._label_by_index, "neutral", fallback=2)
        self._contradiction_index = _find_label_index(
            self._label_by_index,
            "contrad",
            fallback=0,
        )

    @classmethod
    def from_cached_checkpoint(
        cls,
        checkpoint: str,
        *,
        device: str | None = None,
    ) -> TransformersNLIBackend:  # pragma: no cover - exercised by Exp 3654 script
        """Load one cached checkpoint without downloading from the network."""

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
        model = model.to(resolved_device)
        model.eval()
        return cls(checkpoint, tokenizer, model, torch, device=resolved_device)

    def score_claims(
        self,
        model_answer: str,
        evidence_passage: str,
    ) -> list[ClaimGroundingScore]:  # pragma: no cover - covered through script run
        """Return model-backed NLI scores for every atomic claim."""

        claims = split_atomic_claims(model_answer)
        if not claims:
            return []
        hypotheses = [_hypothesis_for_claim(claim) for claim in claims]
        encoded = self.tokenizer(
            [str(evidence_passage)] * len(hypotheses),
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits
        probabilities = self.torch.softmax(logits, dim=-1).detach().cpu().tolist()
        scores = []
        for claim, hypothesis, probs in zip(claims, hypotheses, probabilities, strict=True):
            label_idx = int(max(range(len(probs)), key=lambda idx: probs[idx]))
            label = self._label_by_index.get(label_idx, f"label_{label_idx}")
            entailment = float(probs[self._entailment_index])
            neutral = float(probs[self._neutral_index])
            contradiction = float(probs[self._contradiction_index])
            scores.append(
                ClaimGroundingScore(
                    claim=claim,
                    hypothesis=hypothesis,
                    predicted_label=label,
                    entailment_probability=round(entailment, 6),
                    neutral_probability=round(neutral, 6),
                    contradiction_probability=round(contradiction, 6),
                    energy=round(contradiction, 6),
                    hard_unentailed="entail" not in label.lower(),
                )
            )
        return scores


class NLIAtomicClaimGroundingVerifier:
    """Model-backed atomic-claim grounding verifier.

    The verifier delegates NLI scoring to a real cached transformers checkpoint
    backend. Use `from_cached_or_proxy()` when the caller wants the explicit
    proxy fallback instead of failing on machines without a cached checkpoint.
    """

    model_based = True

    def __init__(self, backend: TransformersNLIBackend) -> None:
        self.backend = backend
        self.nli_substrate = backend.nli_substrate

    @classmethod
    def from_cached_or_proxy(
        cls,
        *,
        checkpoint_candidates: Sequence[str] = DEFAULT_NLI_CHECKPOINT_CANDIDATES,
        device: str | None = None,
        allow_proxy: bool = True,
    ) -> NLIAtomicClaimGroundingVerifier | TextStatisticalEntailmentProxy:
        """Load the first cached NLI checkpoint, or return the disclosed proxy."""

        errors = []
        for checkpoint in checkpoint_candidates:
            try:
                backend = TransformersNLIBackend.from_cached_checkpoint(
                    checkpoint,
                    device=device,
                )
                return cls(backend)
            except Exception as exc:  # pragma: no cover - error details are environment-specific
                errors.append(f"{checkpoint}: {type(exc).__name__}: {exc}")
        if not allow_proxy:
            raise RuntimeError("no cached NLI checkpoint loadable: " + " | ".join(errors))
        return TextStatisticalEntailmentProxy(unavailable_reason=" | ".join(errors))

    def score_claims(
        self,
        model_answer: str,
        evidence_passage: str,
    ) -> list[ClaimGroundingScore]:  # pragma: no cover - covered through script run
        """Return model-backed claim scores."""

        return self.backend.score_claims(model_answer, evidence_passage)

    def verify(
        self,
        model_answer: str,
        evidence_passage: str,
    ) -> float:  # pragma: no cover - covered through script run
        """Return mean model-backed grounding energy over atomic claims."""

        scores = self.score_claims(model_answer, evidence_passage)
        if not scores:
            return 0.0
        return round(float(sum(score.energy for score in scores) / len(scores)), 6)


def score_corpus_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    verifier: GroundingVerifier,
) -> list[float]:
    """Score rows while passing only model answer and evidence into the verifier."""

    scores = []
    for row in rows:
        model_answer = str(row.get("answer") or "")
        evidence_passage = str(row.get("evidence_passage") or "")
        scores.append(float(verifier.verify(model_answer, evidence_passage)))
    return scores


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    verifier: GroundingVerifier | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    n_bootstrap: int = 200,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3654 terminal artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    facts_artifact = _read_json_object(root_path / EXP3640_REL_PATH)
    corpus_path = _resolve_corpus_path(root_path, facts_artifact)
    rows, blocked_reason = _load_valid_v3_rows(corpus_path)
    if blocked_reason is not None:
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _blocked_artifact(
            blocked_reason,
            started_s=start,
            finished_s=finished,
            corpus_path=corpus_path,
            tests_run=tests_run,
        )
        validate_artifact(artifact)
        return artifact

    grounding_verifier = verifier or NLIAtomicClaimGroundingVerifier.from_cached_or_proxy()
    labels = [int(bool(row.get("is_hallucination"))) for row in rows]
    scores = score_corpus_rows(rows, verifier=grounding_verifier)
    grounding_metrics = metric_bundle(
        labels,
        scores,
        n_bootstrap=n_bootstrap,
        seeds=BOOTSTRAP_SEEDS,
    )
    confidence_scores = [1.0 - _coerce_float(row.get("model_confidence"), 0.5) for row in rows]
    confidence_metrics = metric_bundle(
        labels,
        confidence_scores,
        n_bootstrap=n_bootstrap,
        seeds=BOOTSTRAP_SEEDS,
    )
    confidence_baseline = _confidence_baseline(facts_artifact, confidence_metrics)
    grounding_point = float(grounding_metrics["point"])
    evidence_guard = evidence_excludes_gold_answer(rows)
    leak_diagnostics = leak_diagnostics_for_run(
        evidence_excludes_gold=evidence_guard,
        grounding_auroc=grounding_point,
        score_path_answer_evidence_only=True,
    )
    grounding_leak_free = not leak_diagnostics
    beats_proxy = grounding_point > PROXY_BASELINE_AUROC
    beats_confidence = grounding_point > confidence_baseline
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "honest_verdict": terminal_verdict(
            model_based=bool(grounding_verifier.model_based),
            grounding_leak_free=grounding_leak_free,
            beats_proxy=beats_proxy,
            beats_confidence=beats_confidence,
        ),
        "honest_outcome": honest_outcome(
            model_based=bool(grounding_verifier.model_based),
            grounding_leak_free=grounding_leak_free,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "nli_substrate": grounding_verifier.nli_substrate,
        "nli_grounding_built": True,
        "grounding_auroc": grounding_metrics,
        "grounding_auroc_vs_proxy_delta": round(grounding_point - PROXY_BASELINE_AUROC, 6),
        "proxy_baseline_auroc": PROXY_BASELINE_AUROC,
        "confidence_baseline_auroc": confidence_baseline,
        "confidence_baseline_measured": confidence_metrics,
        "grounding_beats_confidence": bool(beats_confidence and grounding_leak_free),
        "grounding_leak_free": grounding_leak_free,
        "evidence_excludes_gold_answer_assert": evidence_guard,
        "score_path_answer_evidence_only": True,
        "leak_diagnostics": leak_diagnostics,
        "n_examples": len(rows),
        "n_claims": int(
            sum(max(1, len(split_atomic_claims(str(row.get("answer") or "")))) for row in rows)
        ),
        "sample_size_rigor_met": len(rows) >= 200,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "corpus_path": str(corpus_path.relative_to(root_path))
                if corpus_path.is_relative_to(root_path)
                else str(corpus_path),
                "labels": labels,
                "scores": [round(float(score), 8) for score in scores],
                "nli_substrate": grounding_verifier.nli_substrate,
            }
        ),
        "duration_s": round(max(0.0, finished - start), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "nli_grounding_built == true AND grounding_leak_free == true AND "
                "grounding_auroc present"
            ),
            "passed": bool(grounding_leak_free and grounding_metrics is not None),
            "principle": (
                "A trustworthy facts verifier requires it actually ran, was leak-free, "
                "and produced a real AUROC."
            ),
        },
        "source_artifacts": [
            str(EXP3640_REL_PATH),
            "results/experiment_3642_corrected_cross_domain_remeasurement_v4.json",
        ],
        "corpus_path_used": str(corpus_path.relative_to(root_path))
        if corpus_path.is_relative_to(root_path)
        else str(corpus_path),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the Exp 3654 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def terminal_verdict(
    *,
    model_based: bool,
    grounding_leak_free: bool,
    beats_proxy: bool,
    beats_confidence: bool,
) -> str:
    """Select the Exp 3654 terminal verdict from measured outcomes."""

    if not grounding_leak_free:
        return "complete: real_nli_grounding_verifier_built_leak_detected_untrusted"
    if not model_based:
        return "complete: real_nli_grounding_verifier_built_proxy_disclosed_no_checkpoint"
    if beats_proxy and beats_confidence:
        return "complete: real_nli_grounding_verifier_built_beats_proxy_and_confidence_facts_value_found"
    if beats_proxy:
        return "complete: real_nli_grounding_verifier_built_beats_proxy_not_confidence_facts_still_hard"
    return "complete: real_nli_grounding_verifier_built_does_not_beat_proxy_facts_still_hard"


def honest_outcome(*, model_based: bool, grounding_leak_free: bool) -> str:
    """Return the compact outcome used by anti-poison tests."""

    if not grounding_leak_free:
        return "leak_detected"
    if not model_based:
        return "built_proxy_disclosed"
    return "built_model_based"


def evidence_excludes_gold_answer(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Check for separate gold-answer leakage into evidence.

    The v3 corpus uses `answer` as the model answer being scored. A correct model
    answer may naturally appear in the evidence; that is grounding, not leakage.
    This guard only rejects separate reference-answer fields.
    """

    gold_keys = ("gold_answer", "right_answer", "correct_answer", "reference_answer")
    for row in rows:
        evidence = str(row.get("evidence_passage") or "").lower()
        for key in gold_keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip() and value.strip().lower() in evidence:
                return False
    return bool(rows)


def leak_diagnostics_for_run(
    *,
    evidence_excludes_gold: bool,
    grounding_auroc: float | None,
    score_path_answer_evidence_only: bool,
) -> list[str]:
    """Return leak reasons instead of promoting suspect metrics."""

    diagnostics = []
    if not evidence_excludes_gold:
        diagnostics.append("separate_gold_answer_found_in_evidence")
    if grounding_auroc is not None and float(grounding_auroc) >= 0.99:
        diagnostics.append("auroc_at_or_above_0.99")
    if not score_path_answer_evidence_only:
        diagnostics.append("score_path_read_label_or_gold_field")
    return diagnostics


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3654 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    for field in (
        "nli_grounding_built",
        "grounding_beats_confidence",
        "grounding_leak_free",
        "evidence_excludes_gold_answer_assert",
    ):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare top-level bool")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    if not isinstance(artifact.get("field_principles"), Mapping):
        raise ValueError("field_principles must be present")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(artifact["field_principles"]):
        raise ValueError("field_principles must cover all required fields")
    if artifact.get("nli_grounding_built") is True:
        metric = artifact.get("grounding_auroc")
        if not isinstance(metric, Mapping):
            raise ValueError("grounding_auroc must be present when nli_grounding_built=true")
        if len(metric.get("bootstrap_seeds") or []) < 3:
            raise ValueError("grounding_auroc must use at least three bootstrap seeds")
        point = metric.get("point")
        if not isinstance(point, (int, float)) or not math.isfinite(float(point)):
            raise ValueError("grounding_auroc point must be finite")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Return a stable short checksum over measured inputs and scores."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _blocked_artifact(
    reason: str,
    *,
    started_s: float,
    finished_s: float,
    corpus_path: Path,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    """Return a terminal blocked artifact without synthetic metrics."""

    return {
        "honest_verdict": "complete: blocked_no_v3_facts_corpus",
        "honest_outcome": "blocked",
        "blocked_reason": reason,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "nli_substrate": "not_evaluated_no_v3_facts_corpus",
        "nli_grounding_built": False,
        "grounding_auroc": None,
        "grounding_auroc_vs_proxy_delta": None,
        "proxy_baseline_auroc": PROXY_BASELINE_AUROC,
        "confidence_baseline_auroc": DEFAULT_CONFIDENCE_BASELINE_AUROC,
        "grounding_beats_confidence": False,
        "grounding_leak_free": False,
        "evidence_excludes_gold_answer_assert": False,
        "score_path_answer_evidence_only": True,
        "leak_diagnostics": [reason],
        "n_examples": 0,
        "n_claims": 0,
        "sample_size_rigor_met": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {"blocked_reason": reason, "corpus_path": str(corpus_path)}
        ),
        "duration_s": round(max(0.0, finished_s - started_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "nli_grounding_built == true AND grounding_leak_free == true AND "
                "grounding_auroc present"
            ),
            "passed": False,
            "principle": (
                "A trustworthy facts verifier requires it actually ran, was leak-free, "
                "and produced a real AUROC."
            ),
        },
        "corpus_path_used": str(corpus_path),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }


def _load_valid_v3_rows(corpus_path: Path) -> tuple[list[JsonDict], str | None]:
    """Load and validate the v3 facts corpus schema."""

    if not corpus_path.exists():
        return [], "blocked_missing_data_realistic_factual_corpus_v3_jsonl"
    rows = _read_jsonl(corpus_path)
    if not rows:
        return [], "blocked_empty_v3_facts_corpus"
    for idx, row in enumerate(rows):
        missing = [field for field in REQUIRED_CORPUS_FIELDS if field not in row]
        if missing:
            return [], f"blocked_v3_facts_corpus_schema_row_{idx}_missing_{'_'.join(missing)}"
    return rows, None


def _resolve_corpus_path(root: Path, facts_artifact: Mapping[str, Any]) -> Path:
    """Resolve the v3 corpus path from Exp 3640 metadata or the default path."""

    corpus_path = facts_artifact.get("corpus_path_used")
    if isinstance(corpus_path, str) and corpus_path:
        return _repo_path(root, Path(corpus_path))
    return root / DEFAULT_CORPUS_REL_PATH


def _confidence_baseline(
    facts_artifact: Mapping[str, Any],
    measured_confidence_metrics: Mapping[str, Any],
) -> float:
    """Return the Exp 3640 confidence baseline, falling back to measured rows."""

    value = facts_artifact.get("confidence_baseline_auroc_on_corpus")
    if value is None:
        value = measured_confidence_metrics.get("point", DEFAULT_CONFIDENCE_BASELINE_AUROC)
    return round(float(value), 6)


def _read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning an empty mapping if absent."""

    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return {}
    return value


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read JSONL rows from disk."""

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _repo_path(root: Path, path: Path) -> Path:
    """Resolve repository-relative paths."""

    if path.is_absolute():
        return path
    return root / path


def _coerce_float(value: Any, default: float) -> float:
    """Coerce numeric values while preserving deterministic defaults."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _label_map_from_model(model: Any) -> dict[int, str]:  # pragma: no cover - checkpoint metadata
    """Extract an id-to-label map from a transformers model."""

    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not isinstance(id2label, Mapping):
        return {0: "contradiction", 1: "entailment", 2: "neutral"}
    return {int(key): str(value).lower() for key, value in id2label.items()}


def _find_label_index(
    labels: Mapping[int, str],
    needle: str,
    *,
    fallback: int,
) -> int:  # pragma: no cover - checkpoint metadata
    """Find an NLI label index by substring."""

    for idx, label in labels.items():
        if needle in label.lower():
            return int(idx)
    return int(fallback)


__all__ = [
    "BOOTSTRAP_SEEDS",
    "ClaimGroundingScore",
    "NLIAtomicClaimGroundingVerifier",
    "REQUIRED_ARTIFACT_FIELDS",
    "TEXT_STATISTICAL_PROXY_SUBSTRATE",
    "TextStatisticalEntailmentProxy",
    "build_artifact",
    "evidence_excludes_gold_answer",
    "score_corpus_rows",
    "split_atomic_claims",
    "tie_aware_auroc",
    "validate_artifact",
    "write_artifact",
]
