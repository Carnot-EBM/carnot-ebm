"""Two complete-byte source views for REQ-VERIFY-7756.

Windows change the evidence layout while the original facts stay fixed.
Labels never enter this builder, so a label edit cannot change its output.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from carnot.verify import source_alignment as alignment
from carnot.verify import training_runtime as runtime

ARMS = {
    "response_set": {
        "head": "energy_local",
        "mode": "canonical",
        "paired": False,
        "parameters": 266,
    },
    "local_set": {"head": "energy_local", "mode": "canonical", "paired": False, "parameters": 266},
    "augmented_set": {
        "head": "energy_local",
        "mode": "ordinary",
        "paired": True,
        "parameters": 266,
    },
    "constrained_set": {
        "head": "energy_local",
        "mode": "constrained",
        "paired": True,
        "parameters": 268,
    },
    "augmented_mlp": {"head": "mlp_local", "mode": "ordinary", "paired": True, "parameters": 2162},
    "constrained_mlp": {
        "head": "mlp_local",
        "mode": "constrained",
        "paired": True,
        "parameters": 2164,
    },
    "local_logistic": {
        "head": "logistic_local",
        "mode": "canonical",
        "paired": False,
        "parameters": 266,
    },
    "source_erased_constrained_set": {
        "head": "energy_local",
        "mode": "constrained",
        "paired": True,
        "parameters": 268,
    },
    "complete_static_constrained_set": {
        "head": "energy_local",
        "mode": "constrained",
        "paired": True,
        "parameters": 268,
    },
}


def digest(data: bytes) -> str:
    """Bind the exact input bytes, including spaces and Unicode encoding."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _offsets(parts: list[bytes]) -> list[tuple[int, int]]:
    start = 0
    result = []
    for part in parts:
        end = start + len(part)
        result.append((start, end))
        start = end
    return result


def _view(
    source: bytes, answer: bytes, sentences: list[bytes], units: list[bytes], width: int
) -> dict[str, Any]:
    source_offsets = _offsets(sentences)
    windows = list(sentences)
    ranges = list(source_offsets)
    for index in range(max(0, len(sentences) - width + 1)):
        windows.append(b"".join(sentences[index : index + width]))
        ranges.append((source_offsets[index][0], source_offsets[index + width - 1][1]))
    groups: dict[bytes, int] = {}
    group_ids = []
    for window in windows:
        group_ids.append(groups.setdefault(window.strip(), len(groups)))
    return {
        "source_bytes": source,
        "answer_bytes": answer,
        "source_sha256": digest(source),
        "answer_sha256": digest(answer),
        "source_sentences": sentences,
        "answer_units": units,
        "source_offsets": source_offsets,
        "answer_offsets": _offsets(units),
        "window_offsets": ranges,
        "windows": windows,
        "group_ids": group_ids,
        "pair_features": [],
        "abstention": None,
    }


def prepare_views(source: bytes, answer: bytes) -> dict[str, dict[str, Any]]:
    """Construct both layouts before applying one shared limit decision."""
    if not isinstance(source, bytes) or not isinstance(answer, bytes):
        raise TypeError("source and answer must be bytes")
    try:
        sentences = alignment.sentence_spans(source)
        units = alignment.sentence_spans(answer)
        invalid = False
    except UnicodeDecodeError:
        sentences, units, invalid = [], [], True
    a = _view(source, answer, sentences, units, 3)
    b = _view(source, answer, sentences, units, 2)
    reason = None
    if invalid:
        reason = "invalid_utf8"
    elif len(units) > alignment.MAX_ANSWER_UNITS:
        reason = "answer_units_over_budget"
    elif max(len(a["windows"]), len(b["windows"])) > alignment.MAX_WINDOWS:
        reason = "source_windows_over_budget"
    elif not units:
        reason = "empty_answer"
    for view in (a, b):
        view["abstention"] = reason
        if reason is None:
            view["pair_features"] = [
                alignment.pair_features(window, units) for window in view["windows"]
            ]
    return {"a": a, "b": b}


def location_prior(view: dict[str, Any]) -> list[float]:
    """Give each distinct evidence group and the null location equal mass."""
    groups = view["group_ids"]
    count = len(set(groups))
    return [1 / ((count + 1) * groups.count(group)) for group in groups] + [1 / (count + 1)]


def pooled_sentence_features(view: dict[str, Any]) -> list[list[float]]:
    """Pool source locations per answer sentence before a shared control head."""
    if view["abstention"]:
        raise ValueError("abstained view")
    prior = location_prior(view)
    pooled = []
    for unit in view["answer_units"]:
        rows = [alignment.pair_features(window, [unit]) for window in [*view["windows"], b""]]
        pooled.append(
            [
                sum(weight * row[index] for weight, row in zip(prior, rows, strict=True))
                for index in range(alignment.FEATURE_DIM)
            ]
        )
    return pooled


def logistic_averaged_feature_risk(
    params: dict[str, Any], pair: dict[str, dict[str, Any]]
) -> float:
    """Average pooled A/B features before one logistic head and support product."""
    a = pooled_sentence_features(pair["a"])
    b = pooled_sentence_features(pair["b"])
    support = []
    for left, right in zip(a, b, strict=True):
        features = [(x + y) / 2 for x, y in zip(left, right, strict=True)]
        logits = [
            float(params["b"][label])
            + sum(value * float(params["w"][index][label]) for index, value in enumerate(features))
            for label in range(2)
        ]
        support.append(1 / (1 + math.exp(max(-700, min(700, logits[0] - logits[1])))))
    return 1 - math.prod(support)


def decision(
    arm: str,
    risk_a: float | None,
    risk_b: float | None,
    temperature: float,
    label: int | None = None,
) -> dict[str, Any]:
    """Apply one temperature and one typed policy after raw view aggregation."""
    if arm not in ARMS:
        raise ValueError("unregistered arm")
    if risk_a is None or (ARMS[arm]["paired"] and risk_b is None):
        return {
            "raw_risk": 0.5,
            "probability_unsupported": 0.5,
            "action": "escalate",
            "brier": 0.25,
            "realized_cost": 0.25,
            "eligible": False,
        }
    raw = (risk_a + float(risk_b)) / 2 if ARMS[arm]["paired"] else risk_a
    probability = runtime.temperature_risk(raw, temperature)
    action = runtime.action(probability)
    realized = None
    if label is not None:
        realized = {"accept": 5 * label, "reject": 1 - label, "escalate": 0.25}[action]
    return {
        "raw_risk": raw,
        "probability_unsupported": probability,
        "action": action,
        "brier": None if label is None else (probability - label) ** 2,
        "realized_cost": realized,
        "eligible": True,
    }


def serialize_pair(pair: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Make exact bytes portable through JSON without decoding or rewriting."""
    keys = {"source_bytes", "answer_bytes"}
    lists = {"source_sentences", "answer_units", "windows"}
    return {
        name: {
            key: value.hex()
            if key in keys
            else [part.hex() for part in value]
            if key in lists
            else value
            for key, value in view.items()
        }
        for name, view in pair.items()
    }


def deserialize_pair(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Restore byte fields and tuple offsets for a cold exact-view replay."""
    keys = {"source_bytes", "answer_bytes"}
    lists = {"source_sentences", "answer_units", "windows"}
    offsets = {"source_offsets", "answer_offsets", "window_offsets"}
    return {
        name: {
            key: bytes.fromhex(item)
            if key in keys
            else [bytes.fromhex(part) for part in item]
            if key in lists
            else [tuple(span) for span in item]
            if key in offsets
            else item
            for key, item in view.items()
        }
        for name, view in value.items()
    }
