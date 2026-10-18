"""Label-free sentence alignment with an exactly normalized latent state.

REQ-VERIFY-7714. The features are evidence candidates, not truth certificates.
Natural response labels belong to a later experiment and never enter this view.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

HASH_SEED = b"carnot-v672-alignment-7714"
TOKEN_PATTERN = r"\w+|[^\w\s]"
HASH_BINS = 64
FEATURE_DIM = 132
MAX_WINDOWS = 128
MAX_ANSWER_UNITS = 16
NEGATIONS = frozenset({"not", "no", "never", "neither", "without", "cannot"})


def sentence_spans(data: bytes) -> list[bytes]:
    """Split at complete punctuation and retain every whitespace byte."""
    text = data.decode("utf-8", "strict")
    spans: list[bytes] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?:\s+|$)", text):
        end = match.end()
        spans.append(text[start:end].encode("utf-8"))
        start = end
    if start < len(text):
        spans.append(text[start:].encode("utf-8"))
    return spans


def _tokens(data: bytes) -> list[str]:
    return re.findall(TOKEN_PATTERN, data.decode("utf-8", "strict").casefold())


def _hash_vector(tokens: list[str], bigram: bool) -> list[float]:
    items = list(zip(tokens, tokens[1:], strict=False)) if bigram else tokens
    vector = [0.0] * HASH_BINS
    for item in items:
        raw = "\x1f".join(item) if isinstance(item, tuple) else item
        digest = hashlib.blake2b(raw.encode(), digest_size=8, key=HASH_SEED).digest()
        number = int.from_bytes(digest, "big")
        vector[number % HASH_BINS] += 1.0 if number & 64 else -1.0
    scale = max(len(items), 1)
    return [value / scale for value in vector]


def pair_features(window: bytes, answer_units: list[bytes]) -> list[float]:
    """Compare one visible window with every complete answer sentence."""
    source_tokens = _tokens(window)
    source_set = set(source_tokens)
    source_numbers = {token for token in source_tokens if token.isdecimal()}
    source_negated = bool(source_set & NEGATIONS)
    features: list[list[float]] = []
    for unit in answer_units:
        answer_tokens = _tokens(unit)
        answer_set = set(answer_tokens)
        token_source = _hash_vector(source_tokens, False)
        token_answer = _hash_vector(answer_tokens, False)
        bigram_source = _hash_vector(source_tokens, True)
        bigram_answer = _hash_vector(answer_tokens, True)
        overlap = len(source_set & answer_set) / max(len(answer_set), 1)
        mismatch_negation = float(source_negated != bool(answer_set & NEGATIONS))
        answer_numbers = {token for token in answer_tokens if token.isdecimal()}
        mismatch_numeric = float(bool(answer_numbers) and answer_numbers != source_numbers)
        uncovered = 1.0 - overlap
        features.append(
            [a * b for a, b in zip(token_source, token_answer, strict=True)]
            + [a * b for a, b in zip(bigram_source, bigram_answer, strict=True)]
            + [overlap, mismatch_negation, mismatch_numeric, uncovered]
        )
    return [sum(unit[index] for unit in features) / len(features) for index in range(FEATURE_DIM)]


def prepare(source: bytes, answer: bytes) -> dict[str, Any]:
    """Make all views without discarding original input or an oversized family."""
    source_sentences = sentence_spans(source)
    answer_units = sentence_spans(answer)
    windows = list(source_sentences)
    windows.extend(
        b"".join(source_sentences[index : index + 3])
        for index in range(max(0, len(source_sentences) - 2))
    )
    abstention = None
    if len(answer_units) > MAX_ANSWER_UNITS:
        abstention = "answer_units_over_budget"
    elif len(windows) > MAX_WINDOWS:
        abstention = "source_windows_over_budget"
    elif not answer_units:
        abstention = "empty_answer"
    pairs = [pair_features(window, answer_units) for window in windows] if not abstention else []
    group_ids: list[int] = []
    groups: dict[bytes, int] = {}
    for window in windows:
        key = window.strip()
        if key not in groups:
            groups[key] = len(groups)
        group_ids.append(groups[key])
    return {
        "source_bytes": source,
        "answer_bytes": answer,
        "source_sentences": source_sentences,
        "answer_units": answer_units,
        "windows": windows,
        "group_ids": group_ids,
        "pair_features": pairs,
        "abstention": abstention,
    }


def zero_parameters() -> dict[str, list[Any]]:
    """Return a shape-stable initialization for a later supervised fit."""
    return {"weights": [[0.0] * FEATURE_DIM for _ in range(2)], "bias": [0.0, 0.0]}


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _states(view: dict[str, Any]) -> tuple[list[list[float]], list[float]]:
    if view["abstention"]:
        raise ValueError(f"abstained family: {view['abstention']}")
    groups = view["group_ids"]
    group_count = len(set(groups))
    state_features = [*view["pair_features"], pair_features(b"", view["answer_units"])]
    counts = [groups.count(group) for group in groups]
    prior = [1.0 / ((group_count + 1) * count) for count in counts]
    prior.append(1.0 / (group_count + 1))
    return state_features, prior


def distribution(view: dict[str, Any], params: dict[str, list[Any]]) -> dict[str, Any]:
    """Normalize over labels and visible or null evidence in log space."""
    features, priors = _states(view)
    scores = [
        [
            math.log(prior)
            - float(params["bias"][label])
            - sum(
                float(weight) * value
                for weight, value in zip(params["weights"][label], row, strict=True)
            )
            for label in range(2)
        ]
        for row, prior in zip(features, priors, strict=True)
    ]
    normalizer = _logsumexp([score for pair in scores for score in pair])
    joint = [[math.exp(score - normalizer) for score in pair] for pair in scores]
    marginal = [sum(pair[label] for pair in joint) for label in range(2)]
    return {
        "joint": joint,
        "marginal": marginal,
        "null_mass": sum(joint[-1]),
        "log_normalizer": normalizer,
        "state_prior": priors,
    }


def enumerate_marginal(view: dict[str, Any], params: dict[str, list[Any]]) -> list[float]:
    """Use direct enumeration as an independent small-fixture math check."""
    features, priors = _states(view)
    values = [
        [
            prior
            * math.exp(
                -float(params["bias"][label])
                - sum(
                    float(weight) * value
                    for weight, value in zip(params["weights"][label], row, strict=True)
                )
            )
            for label in range(2)
        ]
        for row, prior in zip(features, priors, strict=True)
    ]
    total = sum(sum(pair) for pair in values)
    return [sum(pair[label] for pair in values) / total for label in range(2)]


def pooled_features(view: dict[str, Any]) -> list[float]:
    """Give pooled controls the same label-free source-answer pair input."""
    features, priors = _states(view)
    return [
        sum(row[index] * prior for row, prior in zip(features, priors, strict=True))
        for index in range(FEATURE_DIM)
    ]


def control_scores(
    view: dict[str, Any], params: dict[str, list[Any]], *, kind: str = "logistic"
) -> list[float]:
    """Score registered pooled logistic or bounded pooled MLP control."""
    pooled = pooled_features(view)
    if kind == "mlp":
        pooled = [math.tanh(value) for value in pooled]
    elif kind != "logistic":
        raise ValueError(f"unknown control: {kind}")
    scores = [
        -float(params["bias"][label])
        - sum(
            float(weight) * value
            for weight, value in zip(params["weights"][label], pooled, strict=True)
        )
        for label in range(2)
    ]
    normalizer = _logsumexp(scores)
    return [math.exp(score - normalizer) for score in scores]


def fixture_groups() -> list[dict[str, Any]]:
    """Declare independent families; source views are repeated measures."""
    groups = []
    for index in range(96):
        number = index + 11
        source = (
            f"Record {index} says {number} units. "
            f"It is not final on Sunday. The amount may change next season."
        ).encode()
        answer = (
            f"Record {index} says {number if index % 2 else number + 1} units. "
            f"The amount may change next season."
        ).encode()
        groups.append(
            {
                "family_id": f"fixture-{index:03d}",
                "role": "held" if index >= 64 else "qualification",
                "source": source,
                "answer": answer,
            }
        )
    return groups
