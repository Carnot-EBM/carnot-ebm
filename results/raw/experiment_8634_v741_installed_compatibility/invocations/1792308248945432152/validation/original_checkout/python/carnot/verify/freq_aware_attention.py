"""Frequency-Aware Attention cached top-k proxy.

Frequency-Aware Attention needs raw attention matrices to measure attention
energy on high-frequency token fragments.  The live SOTA telemetry cached for
Exp 2397 does not persist attention tensors, so this Tier 0f prototype uses the
available top-k token distributions as a deterministic proxy: it measures how
much renormalized top-k probability mass lands on high-frequency stopwords or
punctuation fragments.

Spec: REQ-TIER0-011, SCENARIO-TIER0-011, Exp 2397.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from carnot.verify.semantic_energy import binary_auroc

DEFAULT_MANIFEST_PATH = Path("results/live_sota_balanced_telemetry_manifest_1480.jsonl")
DEFAULT_OUTPUT_PATH = Path("results/experiment_2397_freq_aware_attn.json")
DEFAULT_RANDOM_SEED = 42
SEMANTIC_ENERGY_BASELINE_AUROC = 0.685

JsonDict = dict[str, Any]

HIGH_FREQUENCY_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

HIGH_FREQUENCY_FRAGMENTS = frozenset(
    {
        "'",
        '"',
        ".",
        ",",
        ":",
        ";",
        "!",
        "?",
        "-",
        "--",
        "*",
        "**",
        "/",
        "\\",
        "|",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
    }
)

_OUTER_FRAGMENT_RE = re.compile(r"^[\s\"'`*_()\[\]{}]+|[\s\"'`*_()\[\]{}]+$")
_PUNCT_ONLY_RE = re.compile(r"^[^\w\s]+$")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_topk_token(token: Any) -> str:
    """Normalize a cached token string before high-frequency matching.

    Tokenizers often record leading spaces or word-boundary markers in
    alternatives.  This helper keeps the matching deterministic without relying
    on tokenizer-specific libraries.

    Spec: REQ-TIER0-011-2
    """

    text = str(token).replace("Ġ", " ").replace("▁", " ").strip().lower()
    if text in HIGH_FREQUENCY_FRAGMENTS:
        return text
    return _OUTER_FRAGMENT_RE.sub("", text).strip()


def is_high_frequency_token(token: Any) -> bool:
    """Return whether a top-k token is a stopword or punctuation fragment.

    Spec: REQ-TIER0-011-2
    """

    normalized = normalize_topk_token(token)
    return (
        normalized in HIGH_FREQUENCY_STOPWORDS
        or normalized in HIGH_FREQUENCY_FRAGMENTS
        or bool(_PUNCT_ONLY_RE.fullmatch(normalized))
    )


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return float(maximum + math.log(sum(math.exp(value - maximum) for value in values)))


def _score_top_logprobs(
    top_logprobs: Sequence[Any], *, probability_weighted: bool = True
) -> tuple[float, int, int]:
    high_frequency_mass = 0.0
    total_mass = 0.0
    high_frequency_count = 0
    total_count = 0

    for alternatives in top_logprobs:
        if not isinstance(alternatives, dict):
            continue
        token_logprobs = [
            (token, logprob)
            for token, raw_value in alternatives.items()
            if (logprob := _finite_float(raw_value)) is not None
        ]
        if not token_logprobs:
            continue

        local_logsum = _logsumexp([logprob for _token, logprob in token_logprobs])
        for token, logprob in token_logprobs:
            weight = math.exp(logprob - local_logsum) if probability_weighted else 1.0
            total_mass += weight
            total_count += 1
            if is_high_frequency_token(token):
                high_frequency_mass += weight
                high_frequency_count += 1

    if total_mass <= 0.0:
        raise ValueError("entry top_logprobs must contain at least one finite token")

    return float(high_frequency_mass / total_mass), high_frequency_count, total_count


def _score_token_texts(token_texts: Sequence[Any]) -> tuple[float, int, int]:
    total_count = 0
    high_frequency_count = 0
    for token in token_texts:
        total_count += 1
        if is_high_frequency_token(token):
            high_frequency_count += 1

    if total_count == 0:
        raise ValueError("entry token_texts must contain at least one token")
    return float(high_frequency_count / total_count), high_frequency_count, total_count


def freq_attn_score_from_entry(entry: JsonDict, *, probability_weighted: bool = True) -> float:
    """Compute the Frequency-Aware Attention top-k proxy score.

    Spec: REQ-TIER0-011-1, REQ-TIER0-011-2
    """

    score, _high_frequency_count, _total_count = _score_entry(
        entry, probability_weighted=probability_weighted
    )
    return score


def _score_entry(entry: JsonDict, *, probability_weighted: bool = True) -> tuple[float, int, int]:
    top_logprobs = entry.get("top_logprobs") or []
    if top_logprobs:
        return _score_top_logprobs(top_logprobs, probability_weighted=probability_weighted)

    token_texts = entry.get("token_texts") or []
    if token_texts:
        return _score_token_texts(token_texts)

    raise ValueError("entry must contain top_logprobs or token_texts")


def label_from_entry(entry: JsonDict) -> int:
    """Return 1 for hallucination/incorrect rows and 0 for factual/correct rows."""

    correctness = str(entry.get("correctness_label", "")).strip().lower()
    if correctness == "incorrect":
        return 1
    if correctness == "correct":
        return 0
    if entry.get("correct") is False:
        return 1
    if entry.get("correct") is True:
        return 0
    raise ValueError("entry does not contain a binary correctness label")


class FreqAwareAttentionDetector:
    """Tier 0f top-k token-frequency proxy for Frequency-Aware Attention.

    Args:
        threshold: Decision threshold over `freq_attn_score`.
        probability_weighted: When true, score the renormalized probability mass
            assigned to high-frequency tokens inside each top-k distribution.
            When false, score the unweighted fraction of top-k alternatives.
    """

    def __init__(self, threshold: float = 0.25, probability_weighted: bool = True) -> None:
        self.threshold = float(threshold)
        self.probability_weighted = bool(probability_weighted)

    def compute_freq_attn_score(self, entry: JsonDict) -> float:
        """Return a finite high-frequency token-distribution score.

        Spec: REQ-TIER0-011-1, REQ-TIER0-011-2
        """

        score = freq_attn_score_from_entry(
            entry, probability_weighted=self.probability_weighted
        )
        if not math.isfinite(score):
            raise ValueError("freq_attn_score must be finite")
        return min(max(float(score), 0.0), 1.0)

    def verify(self, entry: JsonDict) -> JsonDict:
        """Return the Tier 0f proxy score and thresholded risk flag.

        Spec: REQ-TIER0-011-1
        """

        score, high_frequency_count, total_count = _score_entry(
            entry, probability_weighted=self.probability_weighted
        )
        score = min(max(float(score), 0.0), 1.0)
        return {
            "freq_attn_score": score,
            "is_high_freq_pattern": bool(score >= self.threshold),
            "tier": "0f",
            "proxy_strategy": "stopword_fraction",
            "score_weighting": (
                "topk_probability_mass" if self.probability_weighted else "topk_token_count"
            ),
            "high_frequency_token_count": int(high_frequency_count),
            "topk_token_count": int(total_count),
        }


def verify(entry: JsonDict) -> JsonDict:
    """Convenience wrapper around the default detector."""

    return FreqAwareAttentionDetector().verify(entry)


def _read_jsonl(path: Path, limit: int | None = None) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _preconditions(manifest_path: Path) -> JsonDict:
    checked: JsonDict = {
        "telemetry_manifest_present": manifest_path.is_file(),
        "telemetry_manifest_path": str(manifest_path),
        "telemetry_fields": [],
        "top_logprobs_field_present": False,
        "token_texts_field_present": False,
    }
    if manifest_path.is_file():
        rows = _read_jsonl(manifest_path, limit=1)
        fields = list(rows[0].keys()) if rows else []
        checked["telemetry_fields"] = fields
        checked["top_logprobs_field_present"] = "top_logprobs" in fields
        checked["token_texts_field_present"] = "token_texts" in fields
    return checked


def _blocked_artifact(
    *,
    honest_verdict: str,
    checked: JsonDict,
    start: float,
    random_seed: int,
    proxy_strategy: str = "stopword_fraction",
) -> JsonDict:
    return {
        "status": "blocked",
        "experiment": 2397,
        "honest_verdict": honest_verdict,
        "freq_attn_validated": False,
        "freq_attn_auroc": None,
        "freq_attn_mean_score": None,
        "freq_attn_vs_semantic_energy_delta": None,
        "proxy_strategy": proxy_strategy,
        "n_eval_examples": 0,
        "random_seed": int(random_seed),
        "duration_s": round(time.perf_counter() - start, 6),
        "preconditions_checked": checked,
        "acceptance_gates": {"freq_attn_validated": False},
    }


def build_experiment_artifact(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    n_eval_examples: int = 36,
    random_seed: int = DEFAULT_RANDOM_SEED,
    semantic_energy_baseline: float = SEMANTIC_ENERGY_BASELINE_AUROC,
) -> JsonDict:
    """Evaluate the Tier 0f proxy on cached live SOTA telemetry rows.

    Spec: REQ-TIER0-011-3, REQ-TIER0-011-4
    """

    start = time.perf_counter()
    manifest = Path(manifest_path)
    checked = _preconditions(manifest)
    if not checked["telemetry_manifest_present"]:
        return _blocked_artifact(
            honest_verdict="blocked_telemetry_manifest_missing",
            checked=checked,
            start=start,
            random_seed=random_seed,
        )

    if not (checked["top_logprobs_field_present"] or checked["token_texts_field_present"]):
        return _blocked_artifact(
            honest_verdict="blocked_telemetry_token_fields_missing",
            checked=checked,
            start=start,
            random_seed=random_seed,
        )

    entries = _read_jsonl(manifest, limit=n_eval_examples)
    detector = FreqAwareAttentionDetector()
    labels = [label_from_entry(entry) for entry in entries]
    row_results: list[JsonDict] = []
    scores: list[float] = []

    for entry, label in zip(entries, labels, strict=True):
        result = detector.verify(entry)
        score = float(result["freq_attn_score"])
        scores.append(score)
        row_results.append(
            {
                "case_id": entry.get("case_id"),
                "correctness_label": entry.get("correctness_label"),
                "binary_label": label,
                "freq_attn_score": score,
                "is_high_freq_pattern": result["is_high_freq_pattern"],
                "high_frequency_token_count": result["high_frequency_token_count"],
                "topk_token_count": result["topk_token_count"],
            }
        )

    auroc = binary_auroc(labels, scores)
    score_array = np.asarray(scores, dtype=np.float64)
    nontrivial = len({round(score, 12) for score in scores}) > 1
    expected_count = int(n_eval_examples)
    validated = bool(
        len(entries) == expected_count
        and expected_count > 0
        and nontrivial
        and math.isfinite(float(auroc))
    )
    duration_s = round(time.perf_counter() - start, 6)

    return {
        "status": "complete",
        "experiment": 2397,
        "title": "Frequency-Aware Attention Tier 0f cached top-k proxy validation",
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "module_path": "python/carnot/verify/freq_aware_attention.py",
        "spec_refs": ["REQ-TIER0-011", "SCENARIO-TIER0-011"],
        "field_principles": {
            "honest_verdict": "Terminal-prefix required.",
            "freq_attn_validated": (
                "True if FreqAwareAttentionDetector ran on real data."
            ),
            "freq_attn_auroc": (
                "Primary metric. Honest result vs baseline 0.685."
            ),
            "freq_attn_vs_semantic_energy_delta": (
                "Delta vs baseline. Key signal."
            ),
            "proxy_strategy": (
                "Records which proxy was used (stopword_fraction or direct_attention)."
            ),
            "n_eval_examples": "Must be 36.",
            "random_seed": "Must be 42.",
            "duration_s": "Guards against fabrication.",
            "preconditions_checked": (
                "Records telemetry manifest + field inspection."
            ),
        },
        "honest_verdict": (
            "complete: FreqAwareAttentionDetector Tier 0f stopword_fraction proxy "
            f"ran on {len(entries)} cached telemetry entries; AUROC={float(auroc):.6f}."
        ),
        "freq_attn_validated": validated,
        "freq_attn_auroc": float(auroc),
        "freq_attn_mean_score": float(np.mean(score_array)) if scores else 0.0,
        "freq_attn_vs_semantic_energy_delta": float(auroc - semantic_energy_baseline),
        "semantic_energy_baseline_auroc": float(semantic_energy_baseline),
        "proxy_strategy": "stopword_fraction",
        "proxy_detail": (
            "renormalized_topk_probability_mass_on_stopwords_and_punctuation_fragments"
        ),
        "n_eval_examples": len(entries),
        "n_factual_examples": int(labels.count(0)),
        "n_hallucination_examples": int(labels.count(1)),
        "random_seed": int(random_seed),
        "duration_s": duration_s,
        "preconditions_checked": checked,
        "score_direction": "higher_score_means_more_hallucination_like",
        "score_field": "freq_attn_score",
        "score_weighting": "topk_probability_mass",
        "score_summary": {
            "min": float(np.min(score_array)) if scores else 0.0,
            "max": float(np.max(score_array)) if scores else 0.0,
            "mean": float(np.mean(score_array)) if scores else 0.0,
            "std": float(np.std(score_array)) if scores else 0.0,
            "mean_factual": float(np.mean(score_array[np.asarray(labels) == 0])),
            "mean_hallucination": float(np.mean(score_array[np.asarray(labels) == 1])),
        },
        "source_artifact": str(manifest),
        "evaluation_design": (
            "Load the first 36 live SOTA balanced telemetry rows used by Exp 2351 "
            "and score each row with a cached top-k stopword-frequency proxy."
        ),
        "per_entry_results": row_results,
        "acceptance_gates": {
            "freq_attn_validated": validated,
            "n_eval_examples_is_36": len(entries) == 36,
            "nontrivial_scores": nontrivial,
        },
    }


def write_experiment_artifact(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> JsonDict:
    """Write the Exp 2397 Frequency-Aware Attention deliverable JSON."""

    artifact = build_experiment_artifact(manifest_path=manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    print(json.dumps(write_experiment_artifact(), indent=2, sort_keys=True))
