"""Seal a source-aware feature and decision protocol for later measurement.

This module reads only Exp7410 predictor records. It defines transparent text
overlap proxies and a small Gibbs decision head, but it does not treat lexical
agreement as entailment. Real corpus labels and real model fitting remain for a
later experiment.

Spec refs: REQ-AUTO-7412 and SCENARIO-AUTO-7412-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import unicodedata

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7382_v648_decision_protocol import (
    _clopper_pearson_upper,
    typed_decision as _typed_decision,
)
from carnot.experiment_7410_v650_source_corpus import CorpusReaders, reload_corpus
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)
from carnot.verify.pcib_probe import PCIBProbe


JsonDict = dict[str, Any]
RUN_DATE = "20260919"
MILESTONE = "2026.09.650"
EXPERIMENT_ID = "exp7412-source-features"
SCHEMA = "carnot.exp7412.v650.source_features.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7412_v650_source_features.json")
RAW_DIR = Path("results/raw/experiment_7412_v650_source_features")
MODULE_PATH = Path("python/carnot/experiment_7412_v650_source_features.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7412_v650_source_features.py")
TEST_PATH = Path("tests/python/test_experiment_7412_v650_source_features.py")
SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")
UPSTREAM_PATH = Path("results/experiment_7410_v650_source_corpus.json")
UPSTREAM_MANIFEST_PATH = Path("results/raw/experiment_7410_v650_source_corpus/corpus_manifest.json")

EXPECTED_UPSTREAM_SHA256 = "sha256:f56c1acc842959972ceb425777cda16ba979f6565aa512dbfd3bb0f6db247ffe"
EXPECTED_UPSTREAM_MANIFEST_BYTE_SHA256 = (
    "sha256:be0f0b29d6216eaf9a2c1a8ddefd98a5761256c2c9a091038baaf901ab4c277a"
)
EXPECTED_UPSTREAM_MANIFEST_HASH = (
    "sha256:2843bdf316cb6e75f0fcd878704ca3ac937eaa43b2812ef2b73bea3106f7b5d1"
)
EXPECTED_SOURCE_REVISION = "06638fd6fa5c599f3249e27d1cb489b9bd584411"

INFERENCE_SUBSTRATE = "no_model_load"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
TRAINING_SEEDS = (65_001, 65_002, 65_003, 65_004, 65_005)
MAX_STEPS = 500
LEARNING_RATE = 0.01
L2 = 0.001
HIDDEN_DIM = 4
SOURCE_INPUT_DIM = 6
RESPONSE_INPUT_DIM = 2
ACCEPT_THRESHOLDS = (0.01, 0.025, 0.05)
REJECT_THRESHOLDS = (0.90, 0.95, 0.99)
THRESHOLD_PAIRS = tuple(
    (accept, reject) for accept in ACCEPT_THRESHOLDS for reject in REJECT_THRESHOLDS
)
ARMS = (
    "training_prevalence",
    "l2_logistic_six_input",
    "response_only_2_4_1_gibbs",
    "source_aware_6_4_1_gibbs",
)
FAMILYWISE_ALPHA = 0.05
SIMULTANEOUS_TEST_COUNT = len(ARMS) * len(TRAINING_SEEDS) * len(THRESHOLD_PAIRS) * 2
ALPHA_PER_TEST = FAMILYWISE_ALPHA / SIMULTANEOUS_TEST_COUNT
INCORRECT_ACCEPT_BUDGET = 0.05
CORRECT_REJECT_BUDGET = 0.10

QUESTION_TOKEN_LIMIT = 1_024
ANSWER_TOKEN_LIMIT = 1_024
SENTENCE_TOKEN_LIMIT = 1_024
CONTEXT_TOKEN_LIMIT = 4_096
SOURCE_FEATURE_NAMES = (
    "numeric_novelty_with_context",
    "falsifiability_score",
    "normalized_number_token_overlap",
    "normalized_content_token_overlap",
    "max_answer_source_sentence_overlap",
    "missing_or_empty_source",
)
RESPONSE_FEATURE_NAMES = ("entity_uptake", "falsifiability_score")
PREDICTOR_PARTITIONS = (
    "train",
    "probability_calibration",
    "policy_calibration",
    "online_stream",
    "final_test",
    "excluded_test_overlap",
)

_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[+-]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])"
)
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_details",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "promotion_score",
    "source_feature_protocol_ready_score",
    "feature_definitions",
    "protocol_manifest_path",
    "challenge_manifest_path",
    "annotation_risk_scope",
)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7410_v650_source_corpus.py"),
    Path("python/carnot/experiment_7382_v648_decision_protocol.py"),
    Path("python/carnot/experiment_7385_v648_decision_training.py"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
    Path("python/carnot/models/gibbs/__init__.py"),
    Path("python/carnot/verify/pcib_probe.py"),
    SPEC_PATH,
    UPSTREAM_PATH,
    UPSTREAM_MANIFEST_PATH,
)

V650_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def utc_now() -> str:
    """Return an aware UTC boundary for receipts and checkpoints."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush one truthful phase boundary so long validation stays observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7412] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _normalized_text(value: Any) -> str:
    """Normalize representation without adding information to missing text."""

    return unicodedata.normalize("NFKC", value if isinstance(value, str) else "").lower()


def bounded_word_tokens(value: Any, *, token_limit: int) -> list[str]:
    """Return at most ``token_limit`` normalized word tokens.

    A hard token count bounds both runtime and the amount of source text that can
    influence one row. The regex is lexical only and makes no semantic claim.
    """

    if token_limit <= 0:
        raise ValueError("token_limit must be positive")
    return _WORD_PATTERN.findall(_normalized_text(value))[:token_limit]


def _canonical_decimal(token: str) -> str | None:
    """Convert one finite decimal token to a stable equality representation."""

    try:
        value = Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == 0:
        return "0"
    normalized = value.normalize()
    return format(normalized, "f")


def canonical_number_tokens(value: Any, *, token_limit: int = ANSWER_TOKEN_LIMIT) -> set[str]:
    """Return unique finite decimal values from a bounded text prefix."""

    if token_limit <= 0:
        raise ValueError("token_limit must be positive")
    matches = _NUMBER_PATTERN.findall(_normalized_text(value))[:token_limit]
    converted = (_canonical_decimal(token) for token in matches)
    return {token for token in converted if token is not None}


def _content_tokens(value: Any, *, token_limit: int) -> set[str]:
    """Keep unique non-stop words that can be inspected by a human."""

    return {
        token
        for token in bounded_word_tokens(value, token_limit=token_limit)
        if len(token) >= 2 and token not in _STOP_WORDS
    }


def _bounded_text(value: Any, *, token_limit: int) -> str:
    """Rejoin a bounded word stream for the existing PCIB text helper."""

    return " ".join(bounded_word_tokens(value, token_limit=token_limit))


def _response_text(row: Mapping[str, Any]) -> str:
    """Use the selected sentence when present, or the answer for unscored accounting."""

    sentence = row.get("sentence")
    value = sentence if isinstance(sentence, str) and sentence.strip() else row.get("answer")
    return value if isinstance(value, str) else ""


def _overlap_fraction(answer_tokens: set[str], source_tokens: set[str]) -> float:
    """Normalize unique lexical matches by the answer-side token count."""

    if not answer_tokens:
        return 0.0
    return len(answer_tokens & source_tokens) / len(answer_tokens)


def _source_sentences(context: str) -> list[str]:
    """Split only bounded context text so sentence comparisons have a fixed cost."""

    bounded = _bounded_text(context, token_limit=CONTEXT_TOKEN_LIMIT)
    return [part.strip() for part in _SENTENCE_SPLIT.split(bounded) if part.strip()]


def feature_definitions() -> JsonDict:
    """Describe each frozen proxy and the unchanged response-only ablation."""

    common = {
        "bounds": [0.0, 1.0],
        "normalization": "Unicode NFKC and lowercase lexical matching",
        "token_limits": {
            "question": QUESTION_TOKEN_LIMIT,
            "answer": ANSWER_TOKEN_LIMIT,
            "selected_sentence": SENTENCE_TOKEN_LIMIT,
            "context": CONTEXT_TOKEN_LIMIT,
        },
        "semantic_scope": "interpretable_proxy_not_entailment",
    }
    definitions = {
        "numeric_novelty_with_context": {
            "rule": "unique response numbers absent from context divided by unique response numbers",
            "empty_denominator": 0.0,
            "empty_source": "all response numbers are novel",
        },
        "falsifiability_score": {
            "rule": "existing PCIBProbe falsifiability score with actual bounded context",
            "empty_denominator": 0.0,
        },
        "normalized_number_token_overlap": {
            "rule": "unique response numbers in context divided by unique response numbers",
            "empty_denominator": 0.0,
        },
        "normalized_content_token_overlap": {
            "rule": "unique response content words in context divided by response content words",
            "empty_denominator": 0.0,
        },
        "max_answer_source_sentence_overlap": {
            "rule": "maximum response-content overlap against one bounded source sentence",
            "empty_denominator": 0.0,
        },
        "missing_or_empty_source": {
            "rule": "one for missing or whitespace-only context, else zero",
            "empty_denominator": 1.0,
        },
    }
    return {
        "ordered_source_features": list(SOURCE_FEATURE_NAMES),
        "source_features": definitions,
        "response_only_ablation": {
            "ordered_features": list(RESPONSE_FEATURE_NAMES),
            "context": "empty string to preserve the shipped two-feature PCIB representation",
        },
        **common,
    }


def extract_feature_row(row: Mapping[str, Any]) -> JsonDict:
    """Compute six bounded features from an Exp7410 predictor record only."""

    response = _response_text(row)
    context = row.get("context") if isinstance(row.get("context"), str) else ""
    response_bounded = _bounded_text(response, token_limit=SENTENCE_TOKEN_LIMIT)
    context_bounded = _bounded_text(context, token_limit=CONTEXT_TOKEN_LIMIT)
    missing_source = not context.strip()
    response_numbers = canonical_number_tokens(response, token_limit=SENTENCE_TOKEN_LIMIT)
    context_numbers = canonical_number_tokens(context, token_limit=CONTEXT_TOKEN_LIMIT)
    response_content = _content_tokens(response, token_limit=SENTENCE_TOKEN_LIMIT)
    context_content = _content_tokens(context, token_limit=CONTEXT_TOKEN_LIMIT)
    number_overlap = (
        _overlap_fraction(response_numbers, context_numbers) if not missing_source else 0.0
    )
    numeric_novelty = (1.0 - number_overlap) if response_numbers else 0.0
    content_overlap = (
        _overlap_fraction(response_content, context_content) if not missing_source else 0.0
    )
    sentence_overlaps = [
        _overlap_fraction(
            response_content,
            _content_tokens(sentence, token_limit=CONTEXT_TOKEN_LIMIT),
        )
        for sentence in _source_sentences(context)
    ]
    maximum_sentence_overlap = max(sentence_overlaps, default=0.0) if not missing_source else 0.0
    probe = PCIBProbe()
    source_values = {
        "numeric_novelty_with_context": numeric_novelty,
        "falsifiability_score": probe.compute_falsifiability_score(
            response_bounded, context_bounded
        ),
        "normalized_number_token_overlap": number_overlap,
        "normalized_content_token_overlap": content_overlap,
        "max_answer_source_sentence_overlap": maximum_sentence_overlap,
        "missing_or_empty_source": float(missing_source),
    }
    if list(source_values) != list(SOURCE_FEATURE_NAMES) or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in source_values.values()
    ):
        raise ValueError("source features must be finite ordered values in [0, 1]")
    ablation = {
        "entity_uptake": probe.compute_entity_uptake(response_bounded, ""),
        "falsifiability_score": probe.compute_falsifiability_score(response_bounded, ""),
    }
    return {
        "row_key": str(row.get("row_key") or ""),
        "group_id": str(row.get("group_id") or ""),
        "partition": str(row.get("partition") or ""),
        "source_features": source_values,
        "response_only_ablation": ablation,
    }


def _validate_training_data(
    features: Any,
    labels: Any,
    *,
    allowed_dims: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reject malformed arrays before any optimizer update can occur."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] not in allowed_dims or y.shape != (x.shape[0],):
        raise ValueError("training feature or label shape is invalid")
    if x.shape[0] == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("training values must be non-empty and finite")
    if set(y.tolist()) != {0.0, 1.0}:
        raise ValueError("training data must contain both labels")
    return x, y


def _numeric_tree(value: Any) -> None:
    """Reject executable, Boolean, and non-finite checkpoint leaves."""

    if isinstance(value, bool):
        raise ValueError("checkpoint numeric values cannot be booleans")
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError("checkpoint values must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _numeric_tree(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("checkpoint keys must be strings")
            _numeric_tree(item)
        return
    raise ValueError("checkpoint values must be plain numeric JSON")


def initial_gibbs_checkpoint(*, seed: int, input_dim: int = SOURCE_INPUT_DIM) -> JsonDict:
    """Initialize the fixed one-hidden-layer scalar-energy head."""

    if input_dim not in {RESPONSE_INPUT_DIM, SOURCE_INPUT_DIM}:
        raise ValueError("input_dim must be 2 or 6")
    rng = np.random.default_rng(seed)
    limit = math.sqrt(6.0 / (input_dim + HIDDEN_DIM))
    return {
        "w1": rng.uniform(-limit, limit, size=(HIDDEN_DIM, input_dim)).tolist(),
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float64).tolist(),
        "w_out": np.zeros(HIDDEN_DIM, dtype=np.float64).tolist(),
        "b_out": 0.0,
    }


def validate_gibbs_checkpoint(
    value: Mapping[str, Any], *, input_dim: int = SOURCE_INPUT_DIM
) -> JsonDict:
    """Return a plain finite checkpoint only when exact 6-4-1 shapes match."""

    if not isinstance(value, Mapping) or set(value) != {"w1", "b1", "w_out", "b_out"}:
        raise ValueError("checkpoint keys do not match the fixed Gibbs state")
    _numeric_tree(value)
    checkpoint = deepcopy(dict(value))
    w1 = np.asarray(checkpoint["w1"], dtype=np.float64)
    b1 = np.asarray(checkpoint["b1"], dtype=np.float64)
    w_out = np.asarray(checkpoint["w_out"], dtype=np.float64)
    b_out = np.asarray(checkpoint["b_out"], dtype=np.float64)
    if (
        w1.shape != (HIDDEN_DIM, input_dim)
        or b1.shape != (HIDDEN_DIM,)
        or w_out.shape != (HIDDEN_DIM,)
        or b_out.shape != ()
    ):
        raise ValueError("checkpoint shape does not match the fixed Gibbs architecture")
    return checkpoint


def write_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    """Atomically persist a validated numeric checkpoint."""

    input_dim = len(checkpoint.get("w1", [[]])[0]) if checkpoint.get("w1") else 0
    atomic_json(path, validate_gibbs_checkpoint(checkpoint, input_dim=input_dim))


def load_checkpoint(path: Path, *, input_dim: int = SOURCE_INPUT_DIM) -> JsonDict:
    """Reload one checkpoint without accepting code or implicit conversions."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint unreadable: {exc}") from exc
    return validate_gibbs_checkpoint(value, input_dim=input_dim)


def _sigmoid_array(value: np.ndarray) -> np.ndarray:
    """Evaluate sigmoid without overflow for optimizer and scoring paths."""

    positive = value >= 0.0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def probability_from_energy(energy: float) -> float:
    """Convert finite scalar energy to ``P(incorrect)`` without overflow."""

    if isinstance(energy, bool) or not math.isfinite(float(energy)):
        raise ValueError("energy must be finite")
    value = float(energy)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def gibbs_energy(checkpoint: Mapping[str, Any], features: Sequence[float]) -> float:
    """Evaluate scalar energy from a validated plain numeric checkpoint."""

    vector = np.asarray(features, dtype=np.float64)
    if vector.shape not in {(RESPONSE_INPUT_DIM,), (SOURCE_INPUT_DIM,)} or not np.all(
        np.isfinite(vector)
    ):
        raise ValueError("features must be a finite 2-vector or 6-vector")
    state = validate_gibbs_checkpoint(checkpoint, input_dim=int(vector.shape[0]))
    w1 = np.asarray(state["w1"], dtype=np.float64)
    hidden_linear = w1 @ vector + np.asarray(state["b1"], dtype=np.float64)
    hidden = hidden_linear * _sigmoid_array(hidden_linear)
    return float(np.asarray(state["w_out"], dtype=np.float64) @ hidden + state["b_out"])


def _adam_step(
    params: dict[str, np.ndarray],
    gradients: Mapping[str, np.ndarray],
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    step_number: int,
) -> None:
    """Apply the one frozen Adam update in place."""

    for key in params:
        first[key] = 0.9 * first[key] + 0.1 * gradients[key]
        second[key] = 0.999 * second[key] + 0.001 * gradients[key] ** 2
        corrected_first = first[key] / (1.0 - 0.9**step_number)
        corrected_second = second[key] / (1.0 - 0.999**step_number)
        params[key] -= LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)


def _checkpoint_from_arrays(params: Mapping[str, np.ndarray]) -> JsonDict:
    """Convert optimizer arrays to plain JSON numbers before persistence."""

    return {
        "w1": params["w1"].tolist(),
        "b1": params["b1"].tolist(),
        "w_out": params["w_out"].tolist(),
        "b_out": float(params["b_out"]),
    }


def fit_gibbs_head(
    features: Any,
    labels: Any,
    *,
    seed: int,
    steps: int = MAX_STEPS,
) -> JsonDict:
    """Fit the fixed Gibbs head with natural-prevalence Bernoulli loss."""

    if not 0 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be between zero and {MAX_STEPS}")
    x, y = _validate_training_data(
        features,
        labels,
        allowed_dims={RESPONSE_INPUT_DIM, SOURCE_INPUT_DIM},
    )
    input_dim = int(x.shape[1])
    initial = initial_gibbs_checkpoint(seed=seed, input_dim=input_dim)
    params = {
        "w1": np.asarray(initial["w1"], dtype=np.float64),
        "b1": np.asarray(initial["b1"], dtype=np.float64),
        "w_out": np.asarray(initial["w_out"], dtype=np.float64),
        "b_out": np.asarray(initial["b_out"], dtype=np.float64),
    }
    first = {key: np.zeros_like(value) for key, value in params.items()}
    second = {key: np.zeros_like(value) for key, value in params.items()}
    curve: list[JsonDict] = []
    for update in range(steps + 1):
        linear = x @ params["w1"].T + params["b1"]
        sigmoid = _sigmoid_array(linear)
        hidden = linear * sigmoid
        logits = hidden @ params["w_out"] + params["b_out"]
        proper = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
        penalty = L2 * (float(np.sum(params["w1"] ** 2)) + float(np.sum(params["w_out"] ** 2)))
        curve.append({"step": update, "loss": proper + penalty})
        if update == steps:
            break
        residual = (_sigmoid_array(logits) - y) / len(y)
        grad_w_out = hidden.T @ residual + 2.0 * L2 * params["w_out"]
        grad_b_out = np.asarray(np.sum(residual))
        grad_hidden = residual[:, None] * params["w_out"][None, :]
        silu_gradient = sigmoid + linear * sigmoid * (1.0 - sigmoid)
        grad_linear = grad_hidden * silu_gradient
        gradients = {
            "w1": grad_linear.T @ x + 2.0 * L2 * params["w1"],
            "b1": np.sum(grad_linear, axis=0),
            "w_out": grad_w_out,
            "b_out": grad_b_out,
        }
        _adam_step(params, gradients, first, second, update + 1)
    checkpoint = _checkpoint_from_arrays(params)
    validate_gibbs_checkpoint(checkpoint, input_dim=input_dim)
    return {
        "seed": seed,
        "input_dim": input_dim,
        "hidden_dim": HIDDEN_DIM,
        "objective": "natural_prevalence_bernoulli_loss",
        "learning_rate": LEARNING_RATE,
        "l2": L2,
        "update_count": steps,
        "loss_curve": curve,
        "checkpoint": checkpoint,
        "checkpoint_sha256": canonical_hash(checkpoint),
    }


def fit_logistic_control(
    features: Any,
    labels: Any,
    *,
    seed: int,
    steps: int = MAX_STEPS,
) -> JsonDict:
    """Fit the frozen L2 logistic control on the same six inputs."""

    if not 0 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be between zero and {MAX_STEPS}")
    x, y = _validate_training_data(features, labels, allowed_dims={SOURCE_INPUT_DIM})
    rng = np.random.default_rng(seed)
    params = {
        "coef": rng.normal(0.0, 0.01, size=SOURCE_INPUT_DIM),
        "bias": np.asarray(0.0),
    }
    first = {key: np.zeros_like(value) for key, value in params.items()}
    second = {key: np.zeros_like(value) for key, value in params.items()}
    curve: list[JsonDict] = []
    for update in range(steps + 1):
        logits = x @ params["coef"] + params["bias"]
        proper = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
        loss = proper + L2 * float(np.sum(params["coef"] ** 2))
        curve.append({"step": update, "loss": loss})
        if update == steps:
            break
        residual = (_sigmoid_array(logits) - y) / len(y)
        gradients = {
            "coef": x.T @ residual + 2.0 * L2 * params["coef"],
            "bias": np.asarray(np.sum(residual)),
        }
        _adam_step(params, gradients, first, second, update + 1)
    state = {"coef": params["coef"].tolist(), "bias": float(params["bias"])}
    _numeric_tree(state)
    return {
        "seed": seed,
        "input_dim": SOURCE_INPUT_DIM,
        "objective": "natural_prevalence_bernoulli_loss_l2_logistic",
        "learning_rate": LEARNING_RATE,
        "l2": L2,
        "update_count": steps,
        "loss_curve": curve,
        "checkpoint": state,
        "checkpoint_sha256": canonical_hash(state),
    }


def typed_decision(
    p_incorrect: float | None,
    accept_threshold: float,
    reject_threshold: float,
    model_version: str,
) -> JsonDict:
    """Reuse the shipped typed-action contract with this protocol's thresholds."""

    return _typed_decision(
        p_incorrect,
        accept_threshold=accept_threshold,
        reject_threshold=reject_threshold,
        model_version=model_version,
    )


def label_blind_group_representatives(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Choose the lowest row key per group without inspecting any label."""

    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        group_id = str(row.get("group_id") or "")
        row_key = str(row.get("row_key") or "")
        if not group_id or not row_key:
            raise ValueError("group_id and row_key are required")
        current = selected.get(group_id)
        if current is None or row_key < str(current.get("row_key")):
            selected[group_id] = row
    return [deepcopy(dict(selected[group])) for group in sorted(selected)]


def exact_risk_certificate(harmful_outcomes: Sequence[int], *, risk_budget: float) -> JsonDict:
    """Apply the frozen Bonferroni one-sided exact binomial certificate."""

    if any(value not in {0, 1} for value in harmful_outcomes):
        raise ValueError("harmful outcomes must be binary")
    selected = len(harmful_outcomes)
    harmful = sum(harmful_outcomes)
    if selected == 0:
        return {
            "selected_groups": 0,
            "harmful_outcomes": 0,
            "upper_risk_bound": None,
            "risk_budget": risk_budget,
            "alpha_familywise": FAMILYWISE_ALPHA,
            "alpha_per_test": ALPHA_PER_TEST,
            "simultaneous_test_count": SIMULTANEOUS_TEST_COUNT,
            "certified": False,
            "action_enabled": False,
            "reason": "no_selected_groups_no_certificate",
        }
    upper = _clopper_pearson_upper(harmful, selected, ALPHA_PER_TEST)
    certified = upper <= risk_budget
    return {
        "selected_groups": selected,
        "harmful_outcomes": harmful,
        "upper_risk_bound": upper,
        "risk_budget": risk_budget,
        "alpha_familywise": FAMILYWISE_ALPHA,
        "alpha_per_test": ALPHA_PER_TEST,
        "simultaneous_test_count": SIMULTANEOUS_TEST_COUNT,
        "certified": certified,
        "action_enabled": certified,
        "reason": "exact_bound_within_budget" if certified else "exact_bound_exceeds_budget",
    }


_CHALLENGE_DEFINITIONS: tuple[JsonDict, ...] = (
    {
        "family": "negation",
        "source": "Mira approved the permit.",
        "base": "Mira approved the permit.",
        "contrast": "Mira did not approve the permit.",
        "control": "Permit status: approved by Mira.",
        "relation": ["Mira", "approved", "permit"],
        "contrast_scope": "negated_relation",
    },
    {
        "family": "subject_object_reversal",
        "source": "Lena coached Omar.",
        "base": "Lena coached Omar.",
        "contrast": "Omar coached Lena.",
        "control": "Omar was coached by Lena.",
        "relation": ["Lena", "coached", "Omar"],
        "contrast_scope": "reversed_arguments",
    },
    {
        "family": "comparator_reversal",
        "source": "River A is longer than River B.",
        "base": "River A is longer than River B.",
        "contrast": "River A is shorter than River B.",
        "control": "Compared with River B, River A is longer.",
        "relation": ["River A", "longer_than", "River B"],
        "contrast_scope": "reversed_comparator",
    },
    {
        "family": "time_qualifier",
        "source": "The clinic opened in 2020.",
        "base": "The clinic opened in 2020.",
        "contrast": "The clinic opened in 2021.",
        "control": "Opening year for the clinic: 2020.",
        "relation": ["clinic", "opened_in", "2020"],
        "contrast_scope": "changed_time",
    },
    {
        "family": "unit_mismatch",
        "source": "The parcel weighs 5 kilograms.",
        "base": "The parcel weighs 5 kilograms.",
        "contrast": "The parcel weighs 5 pounds.",
        "control": "Parcel weight: 5 kg.",
        "relation": ["parcel", "weight", "5 kilograms"],
        "contrast_scope": "changed_unit",
    },
    {
        "family": "count_mismatch",
        "source": "Ivo won 3 medals.",
        "base": "Ivo won 3 medals.",
        "contrast": "Ivo won 4 medals.",
        "control": "Medals won by Ivo: three.",
        "relation": ["Ivo", "medal_count", "3"],
        "contrast_scope": "changed_count",
    },
    {
        "family": "omitted_condition",
        "source": "Rae is eligible if Rae completes training.",
        "base": "Rae is eligible if Rae completes training.",
        "contrast": "Rae is eligible.",
        "control": "Completing training makes Rae eligible.",
        "relation": ["training_complete", "implies", "Rae eligible"],
        "contrast_scope": "omitted_condition",
    },
    {
        "family": "coreference_ambiguity",
        "source": "Nia called Ava while Ava drove home.",
        "base": "Ava drove home.",
        "contrast": "Nia called Ava while she drove home.",
        "control": "The person driving home was Ava.",
        "relation": ["Ava", "drove", "home"],
        "contrast_scope": "ambiguous_coreference",
    },
)


def _challenge_hash(value: Mapping[str, Any]) -> str:
    """Hash challenge content while excluding its self-referential hash field."""

    payload = deepcopy(dict(value))
    payload["manifest_hash"] = ""
    return canonical_hash(payload)


def build_challenge_manifest() -> JsonDict:
    """Build 16 minimal-pair cases and eight equivalent controls."""

    predictors: list[JsonDict] = []
    evaluators: list[JsonDict] = []
    for index, definition in enumerate(_CHALLENGE_DEFINITIONS, start=1):
        pair_id = f"pair-{index:02d}-{definition['family']}"
        cases = (
            ("base", definition["base"], "supported", "declared_relation"),
            (
                "contrast",
                definition["contrast"],
                "unsupported_or_ambiguous",
                definition["contrast_scope"],
            ),
            ("control", definition["control"], "supported", "equivalent_paraphrase_or_format"),
        )
        for case_name, answer, expected, scope in cases:
            case_id = f"{pair_id}-{case_name}"
            predictors.append(
                {
                    "case_id": case_id,
                    "pair_id": pair_id,
                    "family": definition["family"],
                    "case_kind": "equivalent_control"
                    if case_name == "control"
                    else "minimal_pair_member",
                    "source": definition["source"],
                    "question": "Extract the relation and preserve its qualifiers.",
                    "answer": answer,
                }
            )
            evaluators.append(
                {
                    "case_id": case_id,
                    "expected_verdict": expected,
                    "expected_scope": scope,
                    "source_relation": definition["relation"],
                    "answer_span": [0, len(answer)],
                    "answer_span_text": answer,
                    "authority": "constructed_source_defined_fixture",
                }
            )
    manifest: JsonDict = {
        "schema": "carnot.exp7412.challenge_manifest.v1",
        "training_use": "prohibited",
        "external_accuracy_evidence": False,
        "expected_fields_exposed_to_prediction_reader": False,
        "pair_count": len(_CHALLENGE_DEFINITIONS),
        "case_count": len(predictors),
        "predictor_records": predictors,
        "evaluator_records": evaluators,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = _challenge_hash(manifest)
    return manifest


def load_challenge_manifest_value(value: Mapping[str, Any]) -> JsonDict:
    """Validate challenge identity, masks, spans, and fixed family coverage."""

    manifest = deepcopy(dict(value))
    if manifest.get("manifest_hash") != _challenge_hash(manifest):
        raise ValueError("challenge_manifest_hash_mismatch")
    predictors = manifest.get("predictor_records")
    evaluators = manifest.get("evaluator_records")
    if not isinstance(predictors, list) or not isinstance(evaluators, list):
        raise ValueError("challenge_records_missing")
    if len(predictors) != 24 or len(evaluators) != 24:
        raise ValueError("challenge_case_count_mismatch")
    denied = {"expected_verdict", "expected_scope", "source_relation", "answer_span"}
    if any(denied.intersection(row) for row in predictors):
        raise ValueError("challenge_predictor_leak")
    predictor_by_id = {str(row.get("case_id")): row for row in predictors}
    if len(predictor_by_id) != 24:
        raise ValueError("challenge_case_identity_mismatch")
    for row in evaluators:
        predictor = predictor_by_id.get(str(row.get("case_id")))
        span = row.get("answer_span")
        if predictor is None or not isinstance(span, list) or len(span) != 2:
            raise ValueError("challenge_evaluator_identity_mismatch")
        start, end = span
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start <= end <= len(str(predictor.get("answer") or ""))
            or str(predictor.get("answer") or "")[start:end] != row.get("answer_span_text")
        ):
            raise ValueError("challenge_answer_span_mismatch")
    return manifest


def write_challenge_manifest(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically seal a validated challenge manifest."""

    atomic_json(path, load_challenge_manifest_value(value))


def load_challenge_manifest(path: Path) -> JsonDict:
    """Reload one sealed challenge manifest from fresh bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"challenge_manifest_unreadable:{exc}") from exc
    return load_challenge_manifest_value(value)


def _planned_conditions() -> list[JsonDict]:
    """Enumerate every deferred real arm, seed, and threshold pair."""

    return [
        {
            "comparative_unit": f"{arm}:seed-{seed}:threshold-{index}",
            "arm": arm,
            "seed": seed,
            "threshold_index": index,
            "accept_threshold": accept,
            "reject_threshold": reject,
            "condition": "real_corpus_fit",
            "status": "unstarted",
            "reason": "sealed_for_exp7413_no_official_labels_opened",
        }
        for arm in ARMS
        for seed in TRAINING_SEEDS
        for index, (accept, reject) in enumerate(THRESHOLD_PAIRS)
    ]


def _protocol_hash(value: Mapping[str, Any]) -> str:
    """Hash protocol content while excluding its self-referential hash field."""

    payload = deepcopy(dict(value))
    payload["manifest_hash"] = ""
    return canonical_hash(payload)


def _path_label(path: Path, root: Path = REPO_ROOT) -> str:
    """Use repository-relative paths when files belong to this checkout."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def write_protocol_manifests(
    raw_dir: Path,
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    source_manifest_hash: str,
) -> JsonDict:
    """Seal predictor features, future fitting rules, and challenge fixtures."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_path = raw_dir / "source_feature_rows.json"
    feature_payload = {
        "schema": "carnot.exp7412.source_feature_rows.v1",
        "label_fields_present": False,
        "records": [deepcopy(dict(row)) for row in feature_rows],
    }
    atomic_json(feature_path, feature_payload)
    challenge_path = raw_dir / "challenge_manifest.json"
    write_challenge_manifest(challenge_path, build_challenge_manifest())
    challenge = load_challenge_manifest(challenge_path)
    protocol: JsonDict = {
        "schema": "carnot.exp7412.protocol_manifest.v1",
        "source_manifest_hash": source_manifest_hash,
        "feature_rows_path": _path_label(feature_path),
        "feature_rows_sha256": sha256_file(feature_path),
        "feature_row_count": len(feature_rows),
        "feature_definitions": feature_definitions(),
        "architecture": {
            "response_only": [RESPONSE_INPUT_DIM, HIDDEN_DIM, 1],
            "source_aware": [SOURCE_INPUT_DIM, HIDDEN_DIM, 1],
            "activation": "silu",
            "energy_output": "scalar",
            "probability": "sigmoid_energy_is_p_incorrect",
        },
        "optimizer": {
            "name": "adam",
            "steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "l2": L2,
            "loss": "natural_prevalence_bernoulli",
            "architecture_or_learning_rate_sweep": False,
        },
        "training_seeds": list(TRAINING_SEEDS),
        "arms": list(ARMS),
        "partition_roles": {
            "fit_weights": "train",
            "fit_probabilities": "probability_calibration",
            "select_policy": "policy_calibration",
            "final_evaluation": "final_test_trusted_evaluator_only",
        },
        "thresholds": {
            "accept": list(ACCEPT_THRESHOLDS),
            "reject": list(REJECT_THRESHOLDS),
            "pairs": [list(pair) for pair in THRESHOLD_PAIRS],
        },
        "intervals": {
            "method": "one_sided_exact_binomial_clopper_pearson",
            "familywise_confidence": 0.95,
            "correction": "bonferroni",
            "simultaneous_test_count": SIMULTANEOUS_TEST_COUNT,
            "alpha_per_test": ALPHA_PER_TEST,
            "representative_rule": "lowest_row_key_per_independent_group_label_blind",
            "empty_action": "disabled_without_certificate",
        },
        "annotation_budgets": {
            "scope": "machine_annotation_only_not_truth",
            "incorrect_accept": INCORRECT_ACCEPT_BUDGET,
            "correct_reject": CORRECT_REJECT_BUDGET,
        },
        "task_acceptance_rules": {
            "protocol_ready": "fixture_reader_checkpoint_and_validation_checks_all_pass",
            "scientific_benefit": "not_measured_in_exp7412",
            "lexical_match_proves_entailment": False,
            "valid_span_proves_entailment": False,
            "constructed_challenge_is_external_accuracy": False,
        },
        "challenge_manifest_path": _path_label(challenge_path),
        "challenge_manifest_sha256": sha256_file(challenge_path),
        "challenge_manifest_hash": challenge["manifest_hash"],
        "official_test_labels_opened": False,
        "real_fitting_performed": False,
        "planned_conditions": _planned_conditions(),
        "manifest_hash": "",
    }
    protocol["manifest_hash"] = _protocol_hash(protocol)
    protocol_path = raw_dir / "protocol_manifest.json"
    atomic_json(protocol_path, protocol)
    return {
        "protocol_path": protocol_path,
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_hash": protocol["manifest_hash"],
        "challenge_path": challenge_path,
        "challenge_sha256": sha256_file(challenge_path),
        "challenge_hash": challenge["manifest_hash"],
        "feature_rows_path": feature_path,
        "feature_rows_sha256": sha256_file(feature_path),
        "feature_row_count": len(feature_rows),
    }


def load_protocol_manifest(path: Path) -> JsonDict:
    """Reload and validate one frozen future-fitting protocol."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"protocol_manifest_unreadable:{exc}") from exc
    if not isinstance(value, dict) or value.get("manifest_hash") != _protocol_hash(value):
        raise ValueError("protocol_manifest_hash_mismatch")
    if value.get("planned_conditions") != _planned_conditions():
        raise ValueError("protocol_planned_conditions_mismatch")
    if value.get("official_test_labels_opened") is not False:
        raise ValueError("protocol_official_labels_opened")
    return value


def load_predictor_feature_rows(
    root: Path, *, emit_progress: bool = False
) -> tuple[list[JsonDict], JsonDict]:
    """Reload Exp7410 and compute features without calling its label reader."""

    raw_dir = root / UPSTREAM_MANIFEST_PATH.parent
    reloaded = reload_corpus(raw_dir)
    readers = CorpusReaders(reloaded)
    predictors: list[JsonDict] = []
    for partition in PREDICTOR_PARTITIONS:
        predictors.extend(readers.read_predictors(partition))
    rows: list[JsonDict] = []
    started = time.monotonic()
    for index, row in enumerate(predictors, start=1):
        rows.append(extract_feature_row(row))
        if emit_progress and (index % 500 == 0 or index == len(predictors)):
            progress(
                started, "feature_extraction", "checkpoint", completed=index, total=len(predictors)
            )
    return rows, {
        "source_manifest_hash": reloaded["manifest_hash"],
        "predictor_rows_read": len(predictors),
        "feature_rows_completed": len(rows),
        "partition_counts": dict(Counter(row["partition"] for row in rows)),
        "official_labels_opened": False,
        "reader_method": "CorpusReaders.read_predictors",
    }


def run_fixture_training(*, steps: int = MAX_STEPS) -> JsonDict:
    """Exercise every Gibbs seed on synthetic fixtures without corpus labels."""

    features = np.full((41, SOURCE_INPUT_DIM), 0.25, dtype=np.float64)
    features[-1] = np.linspace(0.0, 1.0, SOURCE_INPUT_DIM)
    labels = np.asarray([0.0] * 40 + [1.0], dtype=np.float64)
    receipts: list[JsonDict] = []
    with tempfile.TemporaryDirectory(prefix="exp7412-checkpoint-", dir="/tmp") as temporary:
        checkpoint_dir = Path(temporary)
        for seed in TRAINING_SEEDS:
            source_fit = fit_gibbs_head(features, labels, seed=seed, steps=steps)
            response_fit = fit_gibbs_head(features[:, :2], labels, seed=seed, steps=steps)
            path = checkpoint_dir / f"source-{seed}.json"
            write_checkpoint(path, source_fit["checkpoint"])
            reloaded = load_checkpoint(path, input_dim=SOURCE_INPUT_DIM)
            receipts.append(
                {
                    "seed": seed,
                    "source_checkpoint_sha256": source_fit["checkpoint_sha256"],
                    "response_checkpoint_sha256": response_fit["checkpoint_sha256"],
                    "checkpoint_reload_sha256": canonical_hash(reloaded),
                    "source_final_loss": source_fit["loss_curve"][-1]["loss"],
                    "response_final_loss": response_fit["loss_curve"][-1]["loss"],
                    "updates_per_head": steps,
                    "passed": canonical_hash(reloaded) == source_fit["checkpoint_sha256"],
                }
            )
        logistic = fit_logistic_control(features, labels, seed=TRAINING_SEEDS[0], steps=steps)
    finite = all(
        math.isfinite(float(row["source_final_loss"]))
        and math.isfinite(float(row["response_final_loss"]))
        for row in receipts
    ) and math.isfinite(float(logistic["loss_curve"][-1]["loss"]))
    return {
        "performed": True,
        "scope": "synthetic_test_fixtures_only",
        "fixture_rows": len(features),
        "class_counts": {"correct": 40, "incorrect": 1},
        "constant_feature_columns": True,
        "steps": steps,
        "receipts": receipts,
        "logistic_checkpoint_sha256": logistic["checkpoint_sha256"],
        "checkpoint_reload_passed": all(row["passed"] for row in receipts),
        "finite_losses": finite,
        "official_labels_opened": False,
        "passed": all(row["passed"] for row in receipts) and finite,
    }


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Record one machine-readable gate without wrapping scalar values."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _upstream_row(
    check: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    operator: str = "==",
    passed: bool | None = None,
) -> JsonDict:
    """Name every upstream field so a blocked result is actionable."""

    return {
        "check": check,
        "upstream": "exp7410-source-corpus",
        "path": UPSTREAM_PATH.as_posix(),
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def upstream_gate_rows(
    upstream: Mapping[str, Any], manifest: Mapping[str, Any], root: Path
) -> list[JsonDict]:
    """Authenticate exact Exp7410 bytes, eligibility, and manifest identity."""

    upstream_path = root / UPSTREAM_PATH
    manifest_path = root / UPSTREAM_MANIFEST_PATH
    verdict = upstream.get("verdict_class")
    allowed = ["positive", "circular_positive", "null"]
    rows = [
        _upstream_row(
            "upstream_artifact_sha256",
            "artifact_sha256",
            EXPECTED_UPSTREAM_SHA256,
            sha256_file(upstream_path) if upstream_path.is_file() else None,
        ),
        _upstream_row(
            "upstream_source_ready",
            "source_corpus_ready_score",
            1,
            upstream.get("source_corpus_ready_score"),
        ),
        _upstream_row(
            "upstream_verdict_allowed",
            "verdict_class",
            allowed,
            verdict,
            operator="in",
            passed=verdict in allowed,
        ),
        _upstream_row(
            "upstream_unflagged",
            "flagged_adversarial",
            False,
            upstream.get("flagged_adversarial"),
        ),
        _upstream_row(
            "upstream_manifest_field",
            "corpus_manifest_hash",
            EXPECTED_UPSTREAM_MANIFEST_HASH,
            upstream.get("corpus_manifest_hash"),
        ),
        _upstream_row(
            "upstream_manifest_path",
            "corpus_manifest_path",
            UPSTREAM_MANIFEST_PATH.as_posix(),
            upstream.get("corpus_manifest_path"),
        ),
        _upstream_row(
            "upstream_manifest_byte_sha256",
            "manifest_byte_sha256",
            EXPECTED_UPSTREAM_MANIFEST_BYTE_SHA256,
            sha256_file(manifest_path) if manifest_path.is_file() else None,
        ),
        _upstream_row(
            "upstream_manifest_hash",
            "manifest_hash",
            EXPECTED_UPSTREAM_MANIFEST_HASH,
            manifest.get("manifest_hash"),
        ),
        _upstream_row(
            "upstream_source_revision",
            "source_revision",
            EXPECTED_SOURCE_REVISION,
            manifest.get("source_revision"),
        ),
        _upstream_row(
            "upstream_label_authority",
            "label_authority",
            "machine_annotation",
            manifest.get("label_authority"),
        ),
    ]
    return rows


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:
    """Hash local authorities and authenticate Exp7410 before feature work."""

    checks: list[JsonDict] = []
    source_hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "bytes",
                "operator": "==",
                "expected": "readable_nonempty_bytes",
                "observed": observed,
                "passed": observed == "readable_nonempty_bytes",
            }
        )
        if observed:
            source_hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": False if relative == UPSTREAM_PATH else None,
            }
    try:
        upstream_value = json.loads((root / UPSTREAM_PATH).read_text(encoding="utf-8"))
        manifest_value = json.loads((root / UPSTREAM_MANIFEST_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        upstream_value, manifest_value = {}, {}
    if not isinstance(upstream_value, dict):
        upstream_value = {}
    if not isinstance(manifest_value, dict):
        manifest_value = {}
    checks.extend(upstream_gate_rows(upstream_value, manifest_value, root))
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _upstream_row(
            "driving_requirement",
            "REQ-*",
            "REQ-AUTO-7412",
            "REQ-AUTO-7412" if "REQ-AUTO-7412" in spec_text else None,
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    quarantined = "experiment_id: 7412" in exclusion
    checks.append(
        _upstream_row(
            "current_task_not_quarantined",
            EXPERIMENT_ID,
            False,
            quarantined,
        )
    )
    return checks, source_hashes, upstream_value, manifest_value


def _field_principles() -> dict[str, str]:
    """Explain ordinary fields without turning their values into wrappers."""

    principles = {
        key: "Keep this ordinary field machine-readable and independently replayable."
        for key in REQUIRED_ARTIFACT_FIELDS
    }
    principles.update(
        {
            "schema": "Version ordinary top-level fields with experiment identity and terminal status.",
            "run_date": "Use 20260919 and retain actual UTC start and end boundaries.",
            "preconditions_checked": "Record exact paths, hashes, and resource checks before dependent work.",
            "MODEL_SPECS": "Use an empty list because this experiment makes no current LLM call.",
            "model_invoked": "Remain false because no current model load or generation is attempted.",
            "invocation_counts": "Reduce current owned events; every LLM count is zero here.",
            "inference_substrate": "Use a truthful string and keep software detail separate.",
            "inference_substrate_class": "Declare no_model_load without padding runtime.",
            "execution_venue": "Use the closed venue string host.",
            "duration_s": "Measure current monotonic work apart from historical evidence.",
            "phase_spans": "Retain real boundaries, completed units, and checkpoints.",
            "random_seed": "Freeze the five fitting seeds and no inapplicable random work.",
            "reproducibility_checksum": "Bind code, protocol, source bytes, and raw rows.",
            "source_artifact_hashes": "Hash exact sources and preserve upstream adversarial flags.",
            "rows": "Retain every deferred arm, seed, and threshold condition.",
            "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted work.",
            "acceptance_gate_results": "Separate validity checks from unmeasured scientific benefit.",
            "gate_check_summary": "Name exact blocked path, check, field, expected, and observed values.",
            "verifier_is_oracle": "No deployed verifier or annotation supplies exact truth in this protocol.",
            "honest_verdict": "Use complete_ for finished protocol findings and blocked_ for absent prerequisites.",
            "verdict_class": "Use the closed terminal class; readiness without efficacy is null.",
            "flagged_adversarial": "Preserve every critical validation finding and deny readiness when flagged.",
            "validation_receipts": "Retain exact argv, environment, exits, durations, and hashed logs.",
            "field_principles": "Explain fields separately while gate scalars remain ordinary values.",
            "promotion_score": "Remain zero because this protocol changes no production behavior.",
            "source_feature_protocol_ready_score": "One means the feature and decision protocol passed, not that it has value.",
            "feature_definitions": "Freeze six predictor-only functions and the response-only ablation.",
            "protocol_manifest_path": "Point to numeric architecture, seeds, splits, costs, thresholds, and intervals.",
            "challenge_manifest_path": "Point to masked semantic pairs and controls for later experiments.",
            "annotation_risk_scope": "Limit every future certificate to the dataset annotation rule.",
        }
    )
    return principles


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce required gates while keeping scientific support independent."""

    failures = [
        deepcopy(dict(row))
        for row in gates
        if row.get("category") != "scientific_support" and row.get("passed") is not True
    ]
    return {
        "required_checks_passed": not failures,
        "failed_required_checks": failures,
        "support_limited": any(
            row.get("category") == "scientific_support" and row.get("passed") is not True
            for row in gates
        ),
        "blocked_upstream": None,
        "blocked_path": None,
        "blocked_check": None,
        "blocked_field": None,
        "blocked_expected": None,
        "blocked_observed": None,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash all stable artifact evidence except the checksum itself."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def relative_monotonic_bounds(started_ns: int, ended_ns: int) -> tuple[int, int]:
    """Keep measured duration while avoiding wall-clock-like absolute metrics.

    The receipt validator needs ordered monotonic bounds, not the process-wide
    clock origin. A zero origin keeps the same measured duration and prevents a
    generic metric checker from mistaking two nearby timestamps for duplicated
    scientific results.
    """

    if ended_ns < started_ns:
        raise ValueError("monotonic boundary order is invalid")
    return 0, ended_ns - started_ns


def _source_hash_rows(extra_paths: Sequence[Path] = ()) -> JsonDict:
    """Hash implementation, tests, sources, and newly sealed raw files."""

    rows: JsonDict = {}
    for relative in (*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH):
        path = REPO_ROOT / relative
        if path.is_file():
            rows[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": False if relative == UPSTREAM_PATH else None,
            }
    for path in extra_paths:
        if path.is_file():
            label = _path_label(path)
            rows[label] = {
                "path": label,
                "sha256": sha256_file(path),
                "original_flagged_adversarial": None,
            }
    return rows


def build_artifact(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    protocol_outputs: Mapping[str, Any],
    fixture_training: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    reader_receipt: Mapping[str, Any],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    started_ns: int,
    ended_ns: int,
    flagged_adversarial: bool = False,
    fixture_artifact: bool = False,
) -> JsonDict:
    """Build one schema-complete result from sealed protocol evidence."""

    protocol = load_protocol_manifest(Path(protocol_outputs["protocol_path"]))
    challenge = load_challenge_manifest(Path(protocol_outputs["challenge_path"]))
    required_names = set(validation_scope.REQUIRED_CHECK_NAMES)
    received = {
        row.get("name")
        for row in validation_receipts
        if row.get("passed") is True and row.get("exit_code") == 0
    }
    validation_passed = required_names.issubset(received)
    gates = [
        _gate(
            "preconditions",
            "validity",
            "all",
            True,
            all(row.get("passed") is True for row in preconditions),
            all(row.get("passed") is True for row in preconditions),
            "All exact source identities must pass before protocol work.",
        ),
        _gate(
            "predictor_only_reader",
            "safety",
            "==",
            False,
            reader_receipt.get("official_labels_opened"),
            reader_receipt.get("official_labels_opened") is False,
            "Feature work cannot open official labels.",
        ),
        _gate(
            "feature_row_completion",
            "completion",
            "==",
            reader_receipt.get("predictor_rows_read"),
            len(feature_rows),
            reader_receipt.get("predictor_rows_read") == len(feature_rows),
            "Every predictor row must have one bounded feature row.",
        ),
        _gate(
            "fixture_training",
            "safety",
            "==",
            True,
            fixture_training.get("passed"),
            fixture_training.get("passed") is True,
            "Fixture training must keep finite state and reload every seed.",
        ),
        _gate(
            "challenge_reader",
            "safety",
            "==",
            {"pairs": 8, "cases": 24, "expected_exposed": False},
            {
                "pairs": challenge.get("pair_count"),
                "cases": challenge.get("case_count"),
                "expected_exposed": challenge.get("expected_fields_exposed_to_prediction_reader"),
            },
            challenge.get("pair_count") == 8
            and challenge.get("case_count") == 24
            and challenge.get("expected_fields_exposed_to_prediction_reader") is False,
            "Constructed expectations stay outside prediction records.",
        ),
        _gate(
            "protocol_manifest",
            "completion",
            "==",
            {"conditions": 180, "labels_opened": False, "real_fit": False},
            {
                "conditions": len(protocol.get("planned_conditions") or []),
                "labels_opened": protocol.get("official_test_labels_opened"),
                "real_fit": protocol.get("real_fitting_performed"),
            },
            len(protocol.get("planned_conditions") or []) == 180
            and protocol.get("official_test_labels_opened") is False
            and protocol.get("real_fitting_performed") is False,
            "Real fitting remains sealed and explicitly unstarted.",
        ),
        _gate(
            "affected_validation",
            "validation",
            "contains",
            sorted(required_names),
            sorted(received),
            validation_passed,
            "All frozen affected checks must pass.",
        ),
        _gate(
            "real_source_feature_efficacy",
            "scientific_support",
            "==",
            True,
            False,
            False,
            "Exp7412 seals a protocol and does not measure real efficacy.",
        ),
    ]
    summary = _gate_summary(gates)
    ready = int(summary["required_checks_passed"] and not flagged_adversarial)
    relative_started_ns, relative_ended_ns = relative_monotonic_bounds(started_ns, ended_ns)
    invocation = build_current_work_receipt(
        run_id="exp7412-current",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "work": "host CPU text features, NumPy fixture optimization, hashing, and JSON serialization",
            "neural_text_model_loaded": False,
            "jax_platform": os.environ.get("JAX_PLATFORMS", "not_used"),
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=relative_started_ns,
        ended_monotonic_ns=relative_ended_ns,
        phase_spans=phase_spans,
        small_ebm_training={
            "performed": True,
            "scope": "synthetic_test_fixtures_only",
            "receipts": deepcopy(list(fixture_training.get("receipts") or [])),
            "official_labels_opened": False,
        },
    )
    rows = deepcopy(list(protocol["planned_conditions"]))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete" if ready else "disqualified",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **invocation,
        "random_seed": {"training_seeds": list(TRAINING_SEEDS), "resampling_seed": None},
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": len(rows),
            "independent_groups": len({str(row.get("group_id")) for row in feature_rows}),
            "predictor_feature_rows_completed": len(feature_rows),
            "stop_rule": "seal all arm-seed-threshold conditions; perform no real label fit in Exp7412",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "honest_verdict": "complete_null_source_feature_protocol_ready_real_efficacy_unmeasured"
        if ready
        else "complete_disqualified_source_feature_protocol",
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "source_feature_protocol_ready_score": ready,
        "feature_definitions": feature_definitions(),
        "protocol_manifest_path": _path_label(Path(protocol_outputs["protocol_path"])),
        "challenge_manifest_path": _path_label(Path(protocol_outputs["challenge_path"])),
        "annotation_risk_scope": "machine_annotation_only_not_truth",
        "feature_rows_path": _path_label(Path(protocol_outputs["feature_rows_path"])),
        "feature_rows_sha256": protocol_outputs["feature_rows_sha256"],
        "protocol_manifest_hash": protocol_outputs["protocol_hash"],
        "challenge_manifest_hash": protocol_outputs["challenge_hash"],
        "reader_receipt": deepcopy(dict(reader_receipt)),
        "fixture_training_receipt": deepcopy(dict(fixture_training)),
        "real_efficacy_measured": False,
        "lexical_match_proves_entailment": False,
        "valid_span_proves_entailment": False,
        "fixture_artifact": bool(fixture_artifact),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    protocol_outputs: Mapping[str, Any],
    fixture_training: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a ready synthetic artifact without launching child commands."""

    source_hashes = _source_hash_rows(
        (
            Path(protocol_outputs["protocol_path"]),
            Path(protocol_outputs["challenge_path"]),
            Path(protocol_outputs["feature_rows_path"]),
        )
    )
    return build_artifact(
        feature_rows=feature_rows,
        protocol_outputs=protocol_outputs,
        fixture_training=fixture_training,
        validation_receipts=validation_receipts,
        preconditions=[{"check": "fixture", "passed": True}],
        source_hashes=source_hashes,
        reader_receipt={
            "predictor_rows_read": len(feature_rows),
            "feature_rows_completed": len(feature_rows),
            "official_labels_opened": False,
        },
        phase_spans=[],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:00+00:00",
        started_ns=0,
        ended_ns=0,
        fixture_artifact=True,
    )


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish an unchanged external prerequisite failure as a complete block."""

    started = time.monotonic_ns()
    invocation = build_current_work_receipt(
        run_id="exp7412-blocked",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={"work": "precondition checks only"},
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started,
        ended_monotonic_ns=started,
        small_ebm_training={"performed": False, "receipts": []},
    )
    blocked = {
        "upstream": failed.get("upstream"),
        "path": failed.get("path"),
        "check": failed.get("check"),
        "field": failed.get("field"),
        "expected": failed.get("expected"),
        "observed": failed.get("observed"),
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": utc_now(),
        "completed_at_utc": utc_now(),
        "preconditions_checked": [{**blocked, "passed": False}],
        **invocation,
        "random_seed": {"training_seeds": list(TRAINING_SEEDS), "resampling_seed": None},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": len(_planned_conditions()),
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": len(_planned_conditions()),
            "independent_groups": 0,
            "predictor_feature_rows_completed": 0,
            "stop_rule": "blocked before predictor feature work",
        },
        "acceptance_gate_results": [
            _gate(
                str(failed.get("check")),
                "precondition",
                str(failed.get("operator") or "=="),
                failed.get("expected"),
                failed.get("observed"),
                False,
                "An ineligible upstream cannot supply a ready protocol.",
            )
        ],
        "gate_check_summary": {
            "required_checks_passed": False,
            "failed_required_checks": [{**blocked, "passed": False}],
            "support_limited": False,
            "blocked_upstream": blocked["upstream"],
            "blocked_path": blocked["path"],
            "blocked_check": blocked["check"],
            "blocked_field": blocked["field"],
            "blocked_expected": blocked["expected"],
            "blocked_observed": blocked["observed"],
        },
        "verifier_is_oracle": False,
        "honest_verdict": f"blocked_{failed.get('check')}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "source_feature_protocol_ready_score": 0,
        "feature_definitions": feature_definitions(),
        "protocol_manifest_path": None,
        "challenge_manifest_path": None,
        "annotation_risk_scope": "machine_annotation_only_not_truth",
        "feature_rows_path": None,
        "feature_rows_sha256": None,
        "protocol_manifest_hash": None,
        "challenge_manifest_hash": None,
        "reader_receipt": {"official_labels_opened": False},
        "fixture_training_receipt": {"performed": False, "passed": False},
        "real_efficacy_measured": False,
        "lexical_match_proves_entailment": False,
        "valid_span_proves_entailment": False,
        "fixture_artifact": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check declarations, manifests, deferred rows, receipt, and checksum."""

    errors = [f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "promotion_score": 0,
        "annotation_risk_scope": "machine_annotation_only_not_truth",
        "real_efficacy_measured": False,
        "lexical_match_proves_entailment": False,
        "valid_span_proves_entailment": False,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(f"declaration_mismatch:{field}")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if set(value.get("field_principles") or {}) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if value.get("feature_definitions") != feature_definitions():
        errors.append("feature_definitions_mismatch")
    if value.get("verdict_class") == "blocked":
        if value.get("source_feature_protocol_ready_score") != 0 or not str(
            value.get("honest_verdict") or ""
        ).startswith("blocked_"):
            errors.append("blocked_disposition_invalid")
    else:
        rows = value.get("rows") or []
        if len(rows) != len(_planned_conditions()) or rows != _planned_conditions():
            errors.append("planned_row_count_mismatch")
        budget = value.get("sample_size_budget") or {}
        if (
            budget.get("planned") != len(rows)
            or budget.get("attempted") != 0
            or budget.get("completed") != 0
            or budget.get("unstarted") != len(rows)
        ):
            errors.append("sample_size_budget_mismatch")
        expected_ready = int(
            value.get("gate_check_summary", {}).get("required_checks_passed") is True
            and value.get("flagged_adversarial") is False
        )
        if value.get("source_feature_protocol_ready_score") != expected_ready:
            errors.append("source_feature_protocol_ready_score_mismatch")
        protocol_path = Path(str(value.get("protocol_manifest_path") or ""))
        challenge_path = Path(str(value.get("challenge_manifest_path") or ""))
        if not protocol_path.is_absolute():
            protocol_path = REPO_ROOT / protocol_path
        if not challenge_path.is_absolute():
            challenge_path = REPO_ROOT / challenge_path
        try:
            protocol = load_protocol_manifest(protocol_path)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if protocol.get("manifest_hash") != value.get("protocol_manifest_hash"):
                errors.append("protocol_manifest_identity_mismatch")
        try:
            challenge = load_challenge_manifest(challenge_path)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if challenge.get("manifest_hash") != value.get("challenge_manifest_hash"):
                errors.append("challenge_manifest_identity_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    errors.extend(validate_current_work_receipt(value, root=REPO_ROOT))
    return list(dict.fromkeys(errors))


def cold_replay(path: Path) -> list[str]:
    """Freshly reload manifests and independently recompute source feature rows."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"artifact_unreadable:{type(exc).__name__}:{exc}"]
    if not isinstance(value, dict):
        return ["artifact_not_object"]
    errors = validate_artifact(value)
    feature_path = Path(str(value.get("feature_rows_path") or ""))
    if not feature_path.is_absolute():
        feature_path = REPO_ROOT / feature_path
    try:
        feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"feature_rows_unreadable:{type(exc).__name__}:{exc}")
        return list(dict.fromkeys(errors))
    if sha256_file(feature_path) != value.get("feature_rows_sha256"):
        errors.append("feature_rows_sha256_mismatch")
    if not value.get("fixture_artifact"):
        recomputed, receipt = load_predictor_feature_rows(REPO_ROOT)
        if feature_payload.get("records") != recomputed:
            errors.append("independent_feature_row_recompute_mismatch")
        if receipt.get("official_labels_opened") is not False:
            errors.append("independent_reader_opened_labels")
    return list(dict.fromkeys(errors))


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    """Record a disjoint monotonic phase with one completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build cold replay, adversarial verification, and strict row checks."""

    python = ".venv/bin/python"
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--cold-replay",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
    ]


def run_experiment(  # pragma: no cover - exercised by the declared entrypoint.
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Authenticate, seal, validate, replay, and atomically publish the protocol."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(run_started, "preconditions", "start")
    preconditions, source_hashes, _upstream, _manifest = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        artifact = build_blocked_artifact(failed)
        progress(run_started, "write", "before_atomic_terminal", status="blocked")
        atomic_json(root / output_path, artifact)
        progress(run_started, "write", "after_atomic_terminal", status="blocked")
        return artifact

    phase_started = time.monotonic()
    progress(run_started, "feature_extraction", "before_predictor_benchmark")
    feature_rows, reader_receipt = load_predictor_feature_rows(root, emit_progress=True)
    spans.append(_span("feature_extraction", phase_started, run_started, len(feature_rows)))
    progress(
        run_started,
        "feature_extraction",
        "after_predictor_benchmark",
        completed=len(feature_rows),
    )

    phase_started = time.monotonic()
    raw_dir = root / RAW_DIR
    progress(run_started, "protocol_seal", "start")
    outputs = write_protocol_manifests(
        raw_dir,
        feature_rows,
        source_manifest_hash=EXPECTED_UPSTREAM_MANIFEST_HASH,
    )
    spans.append(_span("protocol_seal", phase_started, run_started, 3))
    progress(run_started, "protocol_seal", "end", completed=3)

    phase_started = time.monotonic()
    progress(run_started, "fixture_training", "before_small_ebm_training")
    fixture_training = run_fixture_training(steps=MAX_STEPS)
    spans.append(_span("fixture_training", phase_started, run_started, len(TRAINING_SEEDS)))
    progress(
        run_started,
        "fixture_training",
        "after_small_ebm_training",
        completed=len(TRAINING_SEEDS),
        passed=fixture_training["passed"],
    )

    phase_started = time.monotonic()
    private_root = Path(tempfile.mkdtemp(prefix="exp7412-validation-", dir="/tmp"))
    commands = build_command_plan(root, V650_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, V650_MANIFEST, commands)
    progress(
        run_started,
        "validation",
        "before_affected_subprocesses",
        completed=0,
        planned=len(commands),
        plan_errors=len(plan_errors),
    )
    validation_receipts: list[JsonDict] = []
    if not plan_errors:
        validation_receipts = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    reduction = reduce_affected_receipts(root, V650_MANIFEST, validation_receipts)
    spans.append(_span("affected_validation", phase_started, run_started, len(validation_receipts)))
    progress(
        run_started,
        "validation",
        "after_affected_subprocesses",
        completed=len(validation_receipts),
        passed=reduction["passed"],
    )

    extra_paths = (
        Path(outputs["protocol_path"]),
        Path(outputs["challenge_path"]),
        Path(outputs["feature_rows_path"]),
    )
    source_hashes = {**source_hashes, **_source_hash_rows(extra_paths)}
    candidate = build_artifact(
        feature_rows=feature_rows,
        protocol_outputs=outputs,
        fixture_training=fixture_training,
        validation_receipts=validation_receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        reader_receipt=reader_receipt,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        flagged_adversarial=not reduction["passed"],
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    terminal_commands = _terminal_commands(candidate_path)
    progress(
        run_started,
        "terminal_validation",
        "before_subprocesses",
        completed=0,
        planned=len(terminal_commands),
    )
    terminal = run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )

    final = build_artifact(
        feature_rows=feature_rows,
        protocol_outputs=outputs,
        fixture_training=fixture_training,
        validation_receipts=[*validation_receipts, *terminal],
        preconditions=preconditions,
        source_hashes=source_hashes,
        reader_receipt=reader_receipt,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        flagged_adversarial=not reduction["passed"] or not terminal_passed or critical,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed execution date and fresh-process replay mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the source feature protocol or replay one measured candidate."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
