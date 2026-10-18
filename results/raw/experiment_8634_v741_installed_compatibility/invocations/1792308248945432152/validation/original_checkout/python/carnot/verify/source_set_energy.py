"""Finite sentence evidence sets for REQ-VERIFY-7728.

One latent window is chosen per complete answer sentence. The product of the
sentence marginals is a modeling assumption, not a truth certificate.
"""

from __future__ import annotations

from itertools import product
import math
from typing import Any

from carnot.verify import source_alignment as alignment

FEATURE_DIM = alignment.FEATURE_DIM
MAX_PARAMETERS = 4096
MLP_WIDTH = 16
MLP_PARAMETERS = FEATURE_DIM * MLP_WIDTH + MLP_WIDTH + MLP_WIDTH * 2 + 2


def prepare(source: bytes, answer: bytes) -> dict[str, Any]:
    """Keep the previous byte-exact segmentation, features and hard bounds."""
    return alignment.prepare(source, answer)


def zero_parameters() -> dict[str, Any]:
    """Return a finite two-state linear energy head with 266 parameters."""
    return alignment.zero_parameters()


def _logsumexp(values: list[float]) -> float:
    largest = max(values)
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def _prior(view: dict[str, Any]) -> list[float]:
    groups = view["group_ids"]
    count = len(set(groups))
    return [1 / ((count + 1) * groups.count(group)) for group in groups] + [1 / (count + 1)]


def _scores(view: dict[str, Any], params: dict[str, Any]) -> list[list[list[float]]]:
    if view["abstention"]:
        raise ValueError(f"abstained family: {view['abstention']}")
    if len(params["weights"]) != 2 or any(len(row) != FEATURE_DIM for row in params["weights"]):
        raise ValueError("parameter shape")
    if len(params["bias"]) != 2:
        raise ValueError("parameter shape")
    priors = _prior(view)
    result = []
    for unit in view["answer_units"]:
        features = [alignment.pair_features(window, [unit]) for window in view["windows"]]
        features.append(alignment.pair_features(b"", [unit]))
        state_scores = []
        for row, prior in zip(features, priors, strict=True):
            state_scores.append(
                [
                    math.log(prior)
                    - float(params["bias"][label])
                    - sum(float(w) * x for w, x in zip(params["weights"][label], row, strict=True))
                    for label in range(2)
                ]
            )
        if not all(math.isfinite(score) for state in state_scores for score in state):
            raise ValueError("nonfinite energy")
        result.append(state_scores)
    return result


def distribution(view: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Marginalize each location, then multiply support in log space."""
    joint = []
    support = []
    null_mass = []
    for scores in _scores(view, params):
        normalizer = _logsumexp([score for state in scores for score in state])
        probabilities = [[math.exp(score - normalizer) for score in state] for state in scores]
        joint.append(probabilities)
        support.append(sum(state[1] for state in probabilities))
        null_mass.append(sum(probabilities[-1]))
    log_support = sum(math.log(value) for value in support) if all(support) else -math.inf
    response_support = math.exp(log_support)
    return {
        "joint": joint,
        "sentence_support": support,
        "null_mass": null_mass,
        "response_support": response_support,
        "response_unsupported": -math.expm1(log_support),
        "assumption": "conditional_independence",
    }


def enumerate_response(view: dict[str, Any], params: dict[str, Any]) -> float:
    """Enumerate tiny joint state spaces independently of the factorized path."""
    scores = _scores(view, params)
    if len(scores) > 3 or any(len(states) > 5 for states in scores):
        raise ValueError("enumeration fixture too large")
    sentence_states = [list(product(range(len(states)), range(2))) for states in scores]
    numerator = 0.0
    denominator = 0.0
    for assignment in product(*sentence_states):
        log_weight = sum(
            scores[index][location][label] for index, (location, label) in enumerate(assignment)
        )
        weight = math.exp(log_weight)
        denominator += weight
        if all(label == 1 for _, label in assignment):
            numerator += weight
    return numerator / denominator


def shared_control(view: dict[str, Any], params: dict[str, Any]) -> list[float]:
    """Keep the original one-location whole-answer energy unchanged."""
    return alignment.distribution(view, params)["marginal"]


def pooled_control(view: dict[str, Any], params: dict[str, Any], kind: str) -> list[float]:
    """Score same-input pooled logistic or a bounded two-layer MLP."""
    if kind == "logistic":
        return alignment.control_scores(view, params)
    if kind != "mlp":
        raise ValueError("unknown control")
    x = alignment.pooled_features(view)
    if params.get("parameter_count") != MLP_PARAMETERS or MLP_PARAMETERS > MAX_PARAMETERS:
        raise ValueError("MLP budget or shape")
    hidden = [
        math.tanh(sum(x[i] * params["w1"][i][j] for i in range(FEATURE_DIM)) + params["b1"][j])
        for j in range(MLP_WIDTH)
    ]
    scores = [
        sum(hidden[j] * params["w2"][j][label] for j in range(MLP_WIDTH)) + params["b2"][label]
        for label in range(2)
    ]
    normalizer = _logsumexp(scores)
    return [math.exp(score - normalizer) for score in scores]


def zero_mlp_parameters() -> dict[str, Any]:
    """Freeze the future matched MLP shape before any labels are seen."""
    return {
        "w1": [[0.0] * MLP_WIDTH for _ in range(FEATURE_DIM)],
        "b1": [0.0] * MLP_WIDTH,
        "w2": [[0.0, 0.0] for _ in range(MLP_WIDTH)],
        "b2": [0.0, 0.0],
        "parameter_count": MLP_PARAMETERS,
    }
