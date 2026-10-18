"""Test whether a label-blind energy score improves frozen candidate choice.

The selector sees only proposal structure and generation metadata. Exact SMT
certificates enter later, after the arm policy and selected candidate IDs are
frozen. This separation tests a decision effect instead of testing whether an
energy score can decode labels that it was allowed to inspect.

Spec: REQ-VERIFY-6959 and SCENARIO-VERIFY-6959-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from carnot import experiment_6958_convex_factor_energy_canary as energy_exp


JsonDict = dict[str, Any]

EXPERIMENT_ID = 6959
RANDOM_SEED = 6_959_202_609_04
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
BANK_PATH = Path("results/experiment_6956_three_family_reformulation_bank.json")
CERTIFICATE_PATH = Path("results/experiment_6957_smt_mapping_certification.json")
ENERGY_PATH = Path("results/experiment_6958_convex_factor_energy_canary.json")
OUTPUT_PATH = Path("results/experiment_6959_certified_energy_selection.json")
REPLAY_MANIFEST_PATH = Path(
    "results/checkpoints/experiment_6959_certified_energy_selection_replay.json"
)
SCHEMA_VERSION = "carnot.exp6959.certified_energy_selection.v1"
REPLAY_SCHEMA_VERSION = "carnot.exp6959.selection_replay.v1"
INFERENCE_SUBSTRATE = "frozen_candidate_label_blind_energy_selection"

EXPECTED_CANDIDATE_COUNT = 162
EXPECTED_GROUP_COUNT = 54
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
PROMPT_VARIANT_ORDER = ("direct_affine", "domain_first", "objective_first")
PROMPT_VARIANT_RANK = {name: index for index, name in enumerate(PROMPT_VARIANT_ORDER)}

ARM_CONVEX = "convex_factor_energy"
ARM_MLP = "unconstrained_energy"
ARM_LINEAR = "linear_energy"
ARM_LIKELIHOOD = "likelihood"
ARM_SYNTAX = "syntax_validity"
ARM_CONFIDENCE = "model_confidence"
ARM_SHUFFLED = "shuffled_energy"
ARM_FIXED = "fixed_order"
ARM_ORDER = (
    ARM_CONVEX,
    ARM_MLP,
    ARM_LINEAR,
    ARM_LIKELIHOOD,
    ARM_SYNTAX,
    ARM_CONFIDENCE,
    ARM_SHUFFLED,
    ARM_FIXED,
)
BASELINE_ORDER = (
    ARM_MLP,
    ARM_LINEAR,
    ARM_LIKELIHOOD,
    ARM_SYNTAX,
    ARM_CONFIDENCE,
    ARM_SHUFFLED,
    ARM_FIXED,
)
ARM_DIRECTIONS = {
    ARM_CONVEX: "min",
    ARM_MLP: "min",
    ARM_LINEAR: "min",
    ARM_LIKELIHOOD: "max",
    ARM_SYNTAX: "max",
    ARM_CONFIDENCE: "max",
    ARM_SHUFFLED: "min",
    ARM_FIXED: "min",
}
CHECKPOINT_ARMS = {
    ARM_CONVEX: energy_exp.ARM_CONVEX,
    ARM_MLP: energy_exp.ARM_MLP,
    ARM_LINEAR: energy_exp.ARM_LINEAR,
    ARM_SHUFFLED: energy_exp.ARM_SHUFFLED,
}

FORBIDDEN_SELECTOR_FIELDS = (
    "authorities_agree",
    "authority_agreement",
    "baseline_correct",
    "canonical_relation",
    "certified_relation",
    "counterexample",
    "counterexamples",
    "enumeration_label",
    "exact_label",
    "exact_mapping_correct",
    "expected_label",
    "false_acceptance",
    "false_rejection",
    "quarantined",
    "solver_status",
    "witness",
    "witnesses",
    "z3_label",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "candidate_group_rows",
    "candidate_rows",
    "split_rows",
    "arm_rows",
    "convex_energy_rows",
    "unconstrained_energy_rows",
    "linear_energy_rows",
    "likelihood_rows",
    "syntax_rows",
    "confidence_rows",
    "shuffled_energy_rows",
    "fixed_order_rows",
    "oracle_upper_bound_rows",
    "selection_rows",
    "abstention_rows",
    "headroom_rows",
    "family_rows",
    "difficulty_rows",
    "latency_rows",
    "paired_metric_rows",
    "confidence_interval_rows",
    "leakage_rows",
    "tie_rows",
    "fresh_process_replay_rows",
    "random_seed",
    "reproducibility_checksum",
    "certified_selection_run_complete_score",
    "certified_energy_positive_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes the evidence contract reviewable.",
    "preconditions_checked": "Dependency receipts prevent selection on incomplete frozen data.",
    "inference_substrate": "The substrate limits the claim to label-blind frozen selection.",
    "duration_s": "Wall time distinguishes executed scoring from a schema-only result.",
    "source_artifact_hashes": "Hashes bind the result to its proposals, labels, and checkpoints.",
    "rows": "Per-group arm rows let readers rebuild every selection headline.",
    "candidate_group_rows": "Group receipts prove each model-pair cell kept all candidates.",
    "candidate_rows": "Candidate rows expose the exact label-blind selector payloads.",
    "split_rows": "Split receipts show that the sealed bank is one fixed evaluation split.",
    "arm_rows": "Arm summaries compare selectors on the same group denominator.",
    "convex_energy_rows": "Convex scores expose the tested learned selector directly.",
    "unconstrained_energy_rows": "Unconstrained scores provide a capacity-matched control.",
    "linear_energy_rows": "Linear scores test whether nonlinear geometry changes decisions.",
    "likelihood_rows": "Terminal null likelihoods prevent silent deletion of unavailable data.",
    "syntax_rows": "Syntax validity is a strong label-blind structural baseline.",
    "confidence_rows": "Self-reported confidence tests a generation-native baseline.",
    "shuffled_energy_rows": "Shuffled supervision tests whether energy gains need learned labels.",
    "fixed_order_rows": "Fixed order measures the prompt-position prior.",
    "oracle_upper_bound_rows": "Oracle rows measure selectable headroom without deployment use.",
    "selection_rows": "Frozen chosen IDs connect label-blind decisions to later exact evaluation.",
    "abstention_rows": "Abstention rows keep missing-score behavior in every denominator.",
    "headroom_rows": "Headroom rows separate impossible groups from missed opportunities.",
    "family_rows": "Family rows reveal model, formulation, and diversity imbalance.",
    "difficulty_rows": "Difficulty rows reveal whether gains concentrate in easy groups.",
    "latency_rows": "Latency rows bound the added CPU decision cost.",
    "paired_metric_rows": "Same-group deltas prevent an unpaired aggregate win.",
    "confidence_interval_rows": "Pair-clustered intervals quantify uncertainty without row leakage.",
    "leakage_rows": "Leak audits prove labels and candidate order do not determine scores.",
    "tie_rows": "Tie rows expose every deterministic prompt-order tie break.",
    "fresh_process_replay_rows": "A clean process catches in-memory-only headline calculations.",
    "random_seed": "One seed fixes shuffled controls and pair-level bootstrap draws.",
    "reproducibility_checksum": "A timing-free digest detects scientific payload drift.",
    "certified_selection_run_complete_score": "Completion needs every group, arm, audit, and replay.",
    "certified_energy_positive_score": "Positive credit needs strict top-one gain and headroom capture.",
    "gate_check_summary": "Expected and observed values make blocks and nulls diagnosable.",
    "verifier_is_oracle": "False keeps the evaluated selector distinct from exact certification.",
    "verdict_class": "A closed verdict class prevents prose from disguising a null result.",
    "honest_verdict": "A terminal summary makes the scientific outcome unambiguous.",
}


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes for frozen identities and checksums."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Prefix the digest so the artifact states which hash algorithm it uses."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_path(path: Path) -> str | None:
    """Hash a file and return null when a precondition target is absent."""

    target = Path(path)
    return sha256_bytes(target.read_bytes()) if target.is_file() else None


def _gate(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record both sides of a gate so a blocked run is actionable."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks while also making failed checks easy to locate."""

    copied = [dict(row) for row in checks]
    failed = [row for row in copied if not row["passed"]]
    return {"checks": copied, "failed_checks": failed, "passed": not failed}


def freeze_arm_policy() -> list[JsonDict]:
    """Freeze arm order, score direction, and tie policy before labels load."""

    sources = {
        ARM_CONVEX: "exp6958_checkpoint",
        ARM_MLP: "exp6958_checkpoint",
        ARM_LINEAR: "exp6958_checkpoint",
        ARM_LIKELIHOOD: "generation_logprob_when_present",
        ARM_SYNTAX: "frozen_exp6956_parse",
        ARM_CONFIDENCE: "frozen_model_self_report",
        ARM_SHUFFLED: "exp6958_shuffled_checkpoint",
        ARM_FIXED: "registered_prompt_variant_order",
    }
    return [
        {
            "arm": arm,
            "ordinal": ordinal,
            "score_direction": ARM_DIRECTIONS[arm],
            "score_source": sources[arm],
            "tie_policy": "registered_prompt_variant_then_attempt_key",
            "uses_oracle": False,
            "terminal": True,
        }
        for ordinal, arm in enumerate(ARM_ORDER)
    ]


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    """Find forbidden key names at every nesting depth in a selector payload."""

    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_SELECTOR_FIELDS:
                paths.append(path)
            paths.extend(_forbidden_paths(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return sorted(paths)


def audit_selector_payload(attempt_key: str, payload: Mapping[str, Any]) -> JsonDict:
    """Prove that one literal scoring payload has no exact-evaluation fields."""

    forbidden = _forbidden_paths(payload)
    return {
        "audit": "selector_payload_label_isolation",
        "attempt_key": attempt_key,
        "forbidden_paths": forbidden,
        "passed": not forbidden,
    }


def _fraction(value: Any) -> Fraction:
    """Read public mapping numbers exactly without importing certificate helpers."""

    if isinstance(value, bool):
        return Fraction(int(value))
    return Fraction(str(value))


def _unique_row(rows: Any, field: str, value: str) -> Mapping[str, Any] | None:
    """Return one structural row, since zero or duplicate matches are invalid."""

    if not isinstance(rows, list):
        return None
    matches = [row for row in rows if isinstance(row, Mapping) and row.get(field) == value]
    return matches[0] if len(matches) == 1 else None


def _public_universe(variable: Mapping[str, Any]) -> list[Fraction]:
    """Convert the public finite variable universe to exact numeric values."""

    return [_fraction(value) for value in variable.get("universe", [])]


def structural_factors(attempt: Mapping[str, Any]) -> list[list[float]]:
    """Encode local proposal structure without solver results or saved witnesses.

    The factors use only the public source and target formulations plus the
    candidate's declarations. They check roster coverage, declared domains,
    affine domain images, direction signs, and objective expression shape.
    """

    source = attempt.get("source_formulation", {})
    target = attempt.get("target_formulation", {})
    source_variables = source.get("variables", []) if isinstance(source, Mapping) else []
    target_variables = target.get("variables", []) if isinstance(target, Mapping) else []
    target_by_name = {
        str(row.get("name")): row for row in target_variables if isinstance(row, Mapping)
    }
    parsed = attempt.get("parse", {})
    candidate = parsed.get("parsed_candidate") if isinstance(parsed, Mapping) else None
    mapping = candidate.get("mapping") if isinstance(candidate, Mapping) else None
    if not isinstance(mapping, Mapping):
        return [[1.0, 1.0, 1.0, 0.0, 0.0] for _ in source_variables] + [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]

    variable_rows = mapping.get("variables", [])
    domain_rows = mapping.get("domain_clauses", [])
    source_names = [str(row.get("name")) for row in source_variables]
    target_names = [str(row.get("name")) for row in target_variables]
    mapped_sources = [row.get("source") for row in variable_rows if isinstance(row, Mapping)]
    mapped_targets = [row.get("target") for row in variable_rows if isinstance(row, Mapping)]
    factors: list[list[float]] = []
    for source_variable in source_variables:
        source_name = str(source_variable.get("name"))
        variable_row = _unique_row(variable_rows, "source", source_name)
        coverage = float(
            variable_row is None
            or set(mapped_sources) != set(source_names)
            or set(mapped_targets) != set(target_names)
            or any(mapped_sources.count(name) != 1 for name in source_names)
            or any(mapped_targets.count(name) != 1 for name in target_names)
        )
        domain_violation = 1.0
        affine_violation = 1.0
        if variable_row is not None and str(variable_row.get("target")) in target_by_name:
            target_variable = target_by_name[str(variable_row["target"])]
            domain_row = _unique_row(domain_rows, "source", source_name)
            if domain_row is not None:
                domain_violation = float(
                    domain_row.get("target") != variable_row.get("target")
                    or domain_row.get("source_lower")
                    != source_variable.get("domain", {}).get("lower")
                    or domain_row.get("source_upper")
                    != source_variable.get("domain", {}).get("upper")
                    or domain_row.get("target_lower")
                    != target_variable.get("domain", {}).get("lower")
                    or domain_row.get("target_upper")
                    != target_variable.get("domain", {}).get("upper")
                )
            try:
                scale = _fraction(variable_row["scale"])
                offset = _fraction(variable_row["offset"])
                source_universe = _public_universe(source_variable)
                target_universe = _public_universe(target_variable)
                mapped_universe = sorted({scale * value + offset for value in source_universe})
                affine_violation = float(
                    scale == 0 or mapped_universe != sorted(set(target_universe))
                )
            except (KeyError, ValueError, ZeroDivisionError):
                affine_violation = 1.0
        factors.append([coverage, domain_violation, affine_violation, 0.0, 0.0])

    objective = mapping.get("objective", {})
    direction_violation = 1.0
    shape_violation = 1.0
    try:
        scale = _fraction(objective["scale"])
        source_direction = str(objective["source_direction"])
        target_direction = str(objective["target_direction"])
        expected_target = source_direction if scale > 0 else _opposite(source_direction)
        direction_violation = float(
            scale == 0
            or source_direction != source.get("objective", {}).get("direction")
            or target_direction != target.get("objective", {}).get("direction")
            or target_direction != expected_target
        )
        source_expression = source.get("objective", {}).get("expression", {})
        target_expression = target.get("objective", {}).get("expression", {})
        same_kind = source_expression.get("kind") == target_expression.get("kind")
        same_piece_count = len(source_expression.get("pieces", [])) == len(
            target_expression.get("pieces", [])
        )
        if source_expression.get("kind") == "piecewise_linear":
            source_aggregation = source_expression.get("aggregation")
            expected_aggregation = (
                source_aggregation if scale > 0 else _opposite_aggregation(source_aggregation)
            )
            aggregation_ok = target_expression.get("aggregation") == expected_aggregation
        else:
            aggregation_ok = True
        shape_violation = float(not (same_kind and same_piece_count and aggregation_ok))
        _fraction(objective["offset"])
    except (KeyError, ValueError, ZeroDivisionError):
        direction_violation = 1.0
        shape_violation = 1.0
    factors.append([0.0, 0.0, 0.0, direction_violation, 0.0])
    factors.append([0.0, 0.0, 0.0, 0.0, shape_violation])
    return factors


def _opposite(direction: str) -> str:
    """Return the other registered optimization direction."""

    return "max" if direction == "min" else "min"


def _opposite_aggregation(aggregation: Any) -> Any:
    """Reverse max/min aggregation and leave an unknown shape invalid."""

    if aggregation == "max":
        return "min"
    if aggregation == "min":
        return "max"
    return None


def _likelihood_score(attempt: Mapping[str, Any]) -> float | None:
    """Return a recorded generation likelihood, never a token-count proxy."""

    for container in (attempt, attempt.get("runtime_receipt", {})):
        if not isinstance(container, Mapping):
            continue
        for field in ("mean_logprob", "average_logprob", "sequence_logprob"):
            value = container.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _candidate_from_attempt(attempt: Mapping[str, Any]) -> JsonDict:
    """Create one label-blind candidate row from a frozen proposal attempt."""

    pair_id = str(attempt["pair_id"])
    source_attempt_key = str(attempt["attempt_key"])
    candidate_id = sha256_bytes(canonical_json({"attempt_key": source_attempt_key}))
    group_id = sha256_bytes(
        canonical_json({"model_family": attempt["model_family"], "pair_id": pair_id})
    )
    variant = str(attempt["prompt_variant_id"])
    factors = structural_factors(attempt)
    parse = attempt.get("parse", {})
    selector_payload = {
        "mapping": (
            parse.get("parsed_candidate", {}).get("mapping")
            if isinstance(parse, Mapping) and isinstance(parse.get("parsed_candidate"), Mapping)
            else None
        ),
        "source_formulation": attempt.get("source_formulation"),
        "target_formulation": attempt.get("target_formulation"),
        "generation_metadata": {
            "model_family": attempt.get("model_family"),
            "problem_family": attempt.get("problem_family"),
            "prompt_variant_id": variant,
            "confidence": parse.get("confidence") if isinstance(parse, Mapping) else None,
            "json_valid": bool(parse.get("json_valid")) if isinstance(parse, Mapping) else False,
            "schema_valid": bool(parse.get("schema_valid"))
            if isinstance(parse, Mapping)
            else False,
            "likelihood": _likelihood_score(attempt),
        },
        "factors": factors,
    }
    confidence = parse.get("confidence") if isinstance(parse, Mapping) else None
    scores: dict[str, float | None] = {
        ARM_LIKELIHOOD: _likelihood_score(attempt),
        ARM_SYNTAX: float(
            bool(parse.get("json_valid")) and bool(parse.get("schema_valid"))
            if isinstance(parse, Mapping)
            else False
        ),
        ARM_CONFIDENCE: (
            float(confidence)
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            else None
        ),
        ARM_FIXED: float(PROMPT_VARIANT_RANK[variant]),
    }
    return {
        "attempt_key": candidate_id,
        "_source_attempt_key": source_attempt_key,
        "group_id": group_id,
        "pair_id": pair_id,
        "model_family": str(attempt["model_family"]),
        "problem_family": str(attempt["problem_family"]),
        "prompt_variant_id": variant,
        "ordinal": int(attempt["ordinal"]),
        "raw_sha256": attempt.get("raw_sha256"),
        "factor_count": len(factors),
        "selector_payload": selector_payload,
        "selector_payload_sha256": sha256_bytes(canonical_json(selector_payload)),
        "scores": scores,
    }


def _energy_models(repo_root: Path, energy: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Load the compatible learned checkpoints for every energy arm."""

    models: dict[str, list[Any]] = defaultdict(list)
    for path_text in energy.get("checkpoint_paths", []):
        path = Path(path_text)
        if not path.is_absolute():
            path = repo_root / path
        payload = energy_exp.torch.load(path, map_location="cpu", weights_only=True)
        prior_arm = str(payload["arm"])
        current_arm = next(
            arm
            for arm, registered_prior in CHECKPOINT_ARMS.items()
            if registered_prior == prior_arm
        )
        models[current_arm].append(
            energy_exp._load_checkpoint(path, prior_arm, int(payload["seed"]))
        )
    return {arm: rows for arm, rows in models.items()}


def _score_energy_arms(
    candidates: list[JsonDict], models: Mapping[str, Sequence[Any]]
) -> list[JsonDict]:
    """Score all candidates in one batch per checkpoint and average registered seeds."""

    energy_candidates = [
        energy_exp.Candidate(
            candidate_id=str(row["attempt_key"]),
            pair_id=str(row["pair_id"]),
            family=str(row["problem_family"]),
            generator_template=str(row["prompt_variant_id"]),
            split="frozen_evaluation",
            n_variables=max(0, int(row["factor_count"]) - 2),
            factors=deepcopy(row["selector_payload"]["factors"]),
            is_corrupt=0,
            corruption_kind=None,
            mapping_hash=str(row["raw_sha256"]),
            corruption_hash=None,
        )
        for row in candidates
    ]
    latency_rows: list[JsonDict] = []
    for arm in CHECKPOINT_ARMS:
        started = time.perf_counter()
        seed_scores = [
            energy_exp.score_candidates(model, energy_candidates) for model in models[arm]
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        for candidate in candidates:
            values = [scores[str(candidate["attempt_key"])] for scores in seed_scores]
            candidate["scores"][arm] = sum(values) / len(values)
            candidate.setdefault("energy_scores_by_seed", {})[arm] = values
        latency_rows.append(
            {
                "arm": arm,
                "candidate_count": len(candidates),
                "checkpoint_count": len(seed_scores),
                "total_latency_ms": elapsed_ms,
                "mean_latency_ms": elapsed_ms / len(candidates),
                "terminal": True,
            }
        )
    for arm in (ARM_LIKELIHOOD, ARM_SYNTAX, ARM_CONFIDENCE, ARM_FIXED):
        latency_rows.append(
            {
                "arm": arm,
                "candidate_count": len(candidates),
                "checkpoint_count": 0,
                "total_latency_ms": 0.0,
                "mean_latency_ms": 0.0,
                "terminal": True,
            }
        )
    return latency_rows


def _variant_key(candidate: Mapping[str, Any]) -> tuple[int, str]:
    """Apply the registered prompt order and then the immutable attempt key."""

    return (
        PROMPT_VARIANT_RANK[str(candidate["prompt_variant_id"])],
        str(candidate["attempt_key"]),
    )


def _selection_probability(
    available: Sequence[Mapping[str, Any]], arm: str, selected_key: str
) -> float:
    """Convert arm scores to a bounded confidence for calibration only."""

    values = [float(row["scores"][arm]) for row in available]
    signed = [-value if ARM_DIRECTIONS[arm] == "min" else value for value in values]
    maximum = max(signed)
    weights = [math.exp(value - maximum) for value in signed]
    denominator = sum(weights)
    index = next(i for i, row in enumerate(available) if row["attempt_key"] == selected_key)
    return weights[index] / denominator


def rank_group(candidates: Sequence[Mapping[str, Any]], arm: str) -> JsonDict:
    """Freeze one selected ID without consulting evaluation labels or input order."""

    if arm not in ARM_DIRECTIONS:
        raise ValueError(f"unknown_selection_arm:{arm}")
    ordered = sorted(candidates, key=_variant_key)
    available = [row for row in ordered if row.get("scores", {}).get(arm) is not None]
    base = {
        "group_id": str(ordered[0]["group_id"]),
        "pair_id": str(ordered[0]["pair_id"]),
        "model_family": str(ordered[0].get("model_family", "unknown")),
        "problem_family": str(ordered[0].get("problem_family", "unknown")),
        "arm": arm,
        "score_direction": ARM_DIRECTIONS[arm],
        "candidate_count": len(ordered),
        "available_score_count": len(available),
        "unavailable_score_count": len(ordered) - len(available),
        "tie_policy": "registered_prompt_variant_then_attempt_key",
        "oracle_used_for_selection": False,
        "terminal": True,
    }
    if not available:
        frozen = dict(base, selected_attempt_key=None, selected_score=None, tied_attempt_keys=[])
        return dict(
            frozen,
            tie_count=0,
            abstained=True,
            selection_probability=0.0,
            selection_frozen_hash=sha256_bytes(canonical_json(frozen)),
        )
    scores = [float(row["scores"][arm]) for row in available]
    best = min(scores) if ARM_DIRECTIONS[arm] == "min" else max(scores)
    tied = [row for row in available if float(row["scores"][arm]) == best]
    tied.sort(key=_variant_key)
    selected = tied[0]
    frozen = dict(
        base,
        selected_attempt_key=str(selected["attempt_key"]),
        selected_score=best,
        tied_attempt_keys=[str(row["attempt_key"]) for row in tied],
    )
    return dict(
        frozen,
        tie_count=len(tied),
        abstained=False,
        selection_probability=_selection_probability(available, arm, str(selected["attempt_key"])),
        selection_frozen_hash=sha256_bytes(canonical_json(frozen)),
    )


def candidate_diversity_row(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count duplicate bytes without dropping any candidate from the group."""

    hashes = [row.get("raw_sha256") for row in candidates]
    unique = len(set(hashes))
    return {
        "group_id": str(candidates[0]["group_id"]),
        "pair_id": str(candidates[0]["pair_id"]),
        "candidate_count": len(candidates),
        "unique_candidate_count": unique,
        "duplicate_candidate_count": len(candidates) - unique,
        "candidate_diversity": "unique" if unique == len(candidates) else "contains_duplicates",
        "terminal": True,
    }


def oracle_upper_bound_row(
    candidates: Sequence[Mapping[str, Any]], labels: Mapping[str, bool]
) -> JsonDict:
    """Measure oracle-at-group-size while keeping the oracle out of selection."""

    correct = sum(bool(labels[str(row["attempt_key"])]) for row in candidates)
    return {
        "group_id": str(candidates[0]["group_id"]),
        "pair_id": str(candidates[0]["pair_id"]),
        "k": len(candidates),
        "correct_candidate_count": correct,
        "group_has_correct_candidate": correct > 0,
        "oracle_at_k": float(correct > 0),
        "oracle_deployed": False,
        "terminal": True,
    }


def headroom_row(
    oracle: Mapping[str, Any], *, baseline_correct: bool, convex_correct: bool, baseline_arm: str
) -> JsonDict:
    """Report selectable gain and leave zero-headroom capture undefined."""

    available = int(bool(oracle["group_has_correct_candidate"])) - int(baseline_correct)
    captured = int(convex_correct) - int(baseline_correct)
    return {
        "group_id": str(oracle["group_id"]),
        "pair_id": str(oracle["pair_id"]),
        "strongest_non_oracle_baseline": baseline_arm,
        "baseline_correct": bool(baseline_correct),
        "convex_correct": bool(convex_correct),
        "oracle_correct": bool(oracle["group_has_correct_candidate"]),
        "available_headroom": available,
        "captured_headroom": captured if available else None,
        "headroom_captured": captured / available if available else None,
        "no_correct_candidate": not bool(oracle["group_has_correct_candidate"]),
        "no_headroom": available == 0,
        "terminal": True,
    }


def _calibration(rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    """Compute Brier score and five-bin expected calibration error."""

    if not rows:
        return None, None
    pairs = [
        (float(row.get("selection_probability", 0.0)), float(bool(row["selected_exact_correct"])))
        for row in rows
    ]
    brier = sum((probability - label) ** 2 for probability, label in pairs) / len(pairs)
    ece = 0.0
    for bin_index in range(5):
        lower = bin_index / 5
        upper = (bin_index + 1) / 5
        selected = [
            pair
            for pair in pairs
            if lower <= pair[0] <= upper and (bin_index == 4 or pair[0] < upper)
        ]
        if selected:
            confidence = sum(pair[0] for pair in selected) / len(selected)
            accuracy = sum(pair[1] for pair in selected) / len(selected)
            ece += len(selected) / len(pairs) * abs(confidence - accuracy)
    return brier, ece


def aggregate_selection_rows(
    rows: Sequence[Mapping[str, Any]], group_field: str, group_value: str, arm: str
) -> JsonDict:
    """Aggregate one vote per group and reject duplicate group-arm rows."""

    selected = [
        row for row in rows if row.get(group_field) == group_value and row.get("arm") == arm
    ]
    group_ids = [str(row["group_id"]) for row in selected]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("duplicate_group_arm")
    count = len(selected)
    brier, ece = _calibration(selected)
    return {
        "dimension": group_field,
        "value": group_value,
        "arm": arm,
        "group_count": count,
        "top1_accuracy": (
            sum(bool(row["selected_exact_correct"]) for row in selected) / count if count else None
        ),
        "false_acceptance_rate": (
            sum(bool(row["selected_false_acceptance"]) for row in selected) / count
            if count
            else None
        ),
        "abstention_rate": (
            sum(bool(row["abstained"]) for row in selected) / count if count else None
        ),
        "brier_score": brier,
        "expected_calibration_error": ece,
        "terminal": True,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return a deterministic nearest-rank percentile for bootstrap intervals."""

    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def paired_bootstrap_by_pair(
    rows: Sequence[Mapping[str, Any]], seed: int, samples: int
) -> JsonDict:
    """Resample pair IDs so the three model groups never become independent rows."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(float(row["paired_top1_delta"]))
    pair_ids = sorted(grouped)
    observed = [value for values in grouped.values() for value in values]
    if not pair_ids:
        return {
            "bootstrap_unit": "pair_id",
            "paired_pair_count": 0,
            "paired_group_count": 0,
            "mean_delta": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "bootstrap_samples": samples,
        }
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sampled = [rng.choice(pair_ids) for _ in pair_ids]
        values = [value for pair_id in sampled for value in grouped[pair_id]]
        estimates.append(sum(values) / len(values))
    return {
        "bootstrap_unit": "pair_id",
        "paired_pair_count": len(pair_ids),
        "paired_group_count": len(observed),
        "mean_delta": sum(observed) / len(observed),
        "ci95_lower": _percentile(estimates, 0.025),
        "ci95_upper": _percentile(estimates, 0.975),
        "bootstrap_samples": samples,
    }


def _tie_aware_auroc(scores: Sequence[tuple[float, bool]]) -> float | None:
    """Give tied positive-negative score pairs half credit and keep one class null."""

    positives = [score for score, label in scores if label]
    negatives = [score for score, label in scores if not label]
    if not positives or not negatives:
        return None
    credit = 0.0
    for positive in positives:
        for negative in negatives:
            credit += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return credit / (len(positives) * len(negatives))


def positive_gate(
    *,
    complete: bool,
    paired_ci_lower: float | None,
    headroom_captured: float | None,
    control_checks: Mapping[str, bool],
    candidate_auroc: float | None,
) -> bool:
    """Require decision gain; AUROC is recorded but cannot satisfy this gate."""

    _ = candidate_auroc
    return bool(
        complete
        and paired_ci_lower is not None
        and paired_ci_lower > 0.0
        and headroom_captured is not None
        and headroom_captured >= 0.2
        and all(control_checks.values())
    )


def _checkpoint_compatibility(repo_root: Path, energy: Mapping[str, Any]) -> tuple[int, list[str]]:
    """Load checkpoint metadata and bind every file to its replayed content hash."""

    replay_hashes = {
        (str(row.get("arm")), int(row.get("seed", -1))): row.get("checkpoint_sha256")
        for row in energy.get("fresh_process_replay_rows", [])
    }
    valid: list[str] = []
    for path_text in energy.get("checkpoint_paths", []):
        path = Path(path_text)
        if not path.is_absolute():
            path = repo_root / path
        try:
            payload = energy_exp.torch.load(path, map_location="cpu", weights_only=True)
            prior_arm = str(payload["arm"])
            seed = int(payload["seed"])
            energy_exp._load_checkpoint(path, prior_arm, seed)
            if replay_hashes.get((prior_arm, seed)) != sha256_path(path):
                continue
            if prior_arm not in CHECKPOINT_ARMS.values():
                continue
            valid.append(str(path))
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, pickle.UnpicklingError):
            continue
    return len(valid), sorted(valid)


def check_preconditions(
    repo_root: Path,
    bank: Mapping[str, Any],
    certificates: Mapping[str, Any],
    energy: Mapping[str, Any],
    arm_policy: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Check the frozen roster, sealed labels, and checkpoint bindings before scoring."""

    attempts = bank.get("attempt_rows", [])
    certificate_rows = certificates.get("proposal_rows", [])
    authority_rows = certificates.get("authority_agreement_rows", [])
    bank_keys = [str(row.get("attempt_key")) for row in attempts]
    certificate_keys = [str(row.get("attempt_key")) for row in certificate_rows]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in attempts:
        groups[f"{row.get('hf_id')}|{row.get('pair_id')}"].append(row)
    fixed_groups = all(
        len(rows) == 3
        and {str(row.get("prompt_variant_id")) for row in rows} == set(PROMPT_VARIANT_ORDER)
        for rows in groups.values()
    )
    compatible_count, compatible_paths = _checkpoint_compatibility(repo_root, energy)
    bank_hash = sha256_path(repo_root / BANK_PATH)
    sealed_bank_hash = (
        certificates.get("source_artifact_hashes", {}).get("bank_artifact", {}).get("sha256")
    )
    shared_fixture = bank.get("source_artifact_hashes", {}).get("fixture_artifact")
    energy_fixture = energy.get("source_artifact_hashes", {}).get("exp6955_fixture")
    checks = [
        _gate("arm_policy_frozen", freeze_arm_policy(), [dict(row) for row in arm_policy]),
        _gate(
            "smt_certification_run_complete_score",
            1,
            certificates.get("smt_certification_run_complete_score"),
        ),
        _gate(
            "convex_factor_run_complete_score", 1, energy.get("convex_factor_run_complete_score")
        ),
        _gate("proposal_row_count", EXPECTED_CANDIDATE_COUNT, len(attempts)),
        _gate("unique_proposal_keys", EXPECTED_CANDIDATE_COUNT, len(set(bank_keys))),
        _gate("certificate_row_count", EXPECTED_CANDIDATE_COUNT, len(certificate_rows)),
        _gate("unique_certificate_keys", EXPECTED_CANDIDATE_COUNT, len(set(certificate_keys))),
        _gate("exact_certificate_row_count", EXPECTED_CANDIDATE_COUNT, len(authority_rows)),
        _gate(
            "exact_certificate_rows_terminal",
            True,
            all(row.get("terminal") for row in authority_rows),
        ),
        _gate(
            "proposal_certificate_keys_match", True, sorted(bank_keys) == sorted(certificate_keys)
        ),
        _gate("candidate_group_count", EXPECTED_GROUP_COUNT, len(groups)),
        _gate("fixed_group_ids_and_variants", True, fixed_groups),
        _gate("compatible_checkpoint_count", 12, compatible_count),
        _gate("shared_fixture_hash", shared_fixture, energy_fixture),
        _gate("sealed_evaluation_labels", bank_hash, sealed_bank_hash),
    ]
    summary = _gate_summary(checks)
    summary["compatible_checkpoint_paths"] = compatible_paths
    summary["arm_policy_hash"] = sha256_bytes(canonical_json(arm_policy))
    summary["certificate_opened_after_policy_freeze"] = True
    return summary


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind the artifact to inputs, code, tests, specification, and command surface."""

    paths = {
        "proposal_bank": BANK_PATH,
        "exact_certificates": CERTIFICATE_PATH,
        "energy_canary": ENERGY_PATH,
        "module": Path("python/carnot/experiment_6959_certified_energy_selection.py"),
        "test": Path("tests/python/test_experiment_6959_certified_energy_selection.py"),
        "verification_spec": SPEC_PATH,
        "wrapper": Path("scripts/experiments/experiment_6959_certified_energy_selection.py"),
    }
    return {
        name: {"path": str(path), "sha256": sha256_path(repo_root / path)}
        for name, path in paths.items()
    }


def _empty_rows() -> JsonDict:
    """Return every required row surface for a blocked artifact."""

    return {
        field: []
        for field in REQUIRED_ARTIFACT_FIELDS
        if field == "rows" or field.endswith("_rows")
    }


def build_blocked_artifact(
    *,
    date: str,
    preconditions: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Write a complete diagnostic schema even when selection cannot start."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": deepcopy(source_hashes),
        **_empty_rows(),
        "random_seed": RANDOM_SEED,
        "certified_selection_run_complete_score": 0,
        "certified_energy_positive_score": 0,
        "gate_check_summary": deepcopy(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_certified_energy_selection",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _score_rows(candidates: Sequence[Mapping[str, Any]], arm: str) -> list[JsonDict]:
    """Expose one terminal score row per candidate, including unavailable likelihoods."""

    return [
        {
            "attempt_key": row["attempt_key"],
            "group_id": row["group_id"],
            "pair_id": row["pair_id"],
            "arm": arm,
            "score": row["scores"].get(arm),
            "scores_by_seed": row.get("energy_scores_by_seed", {}).get(arm),
            "score_direction": ARM_DIRECTIONS[arm],
            "available": row["scores"].get(arm) is not None,
            "terminal": True,
        }
        for row in candidates
    ]


def _evaluate_selections(
    frozen: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    group_metadata: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Open exact labels only after each selected identity has a frozen hash."""

    rows = []
    for row in frozen:
        selected_key = row.get("selected_attempt_key")
        label = labels.get(str(selected_key), {}) if selected_key is not None else {}
        metadata = group_metadata[str(row["group_id"])]
        rows.append(
            {
                **dict(row),
                "difficulty": metadata["difficulty"],
                "candidate_diversity": metadata["candidate_diversity"],
                "selected_exact_correct": bool(label.get("exact_mapping_correct", False)),
                "selected_false_acceptance": bool(label.get("false_acceptance", False)),
                "exact_label_opened_after_selection_freeze": True,
            }
        )
    return rows


def _arm_rows(
    selections: Sequence[Mapping[str, Any]],
    score_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Build overall arm metrics, including calibration and candidate AUROC."""

    rows = []
    for arm in ARM_ORDER:
        aggregate = aggregate_selection_rows(selections, "scope", "overall", arm)
        goodness = []
        for row in score_rows[arm]:
            if row["score"] is None:
                continue
            score = float(row["score"])
            if ARM_DIRECTIONS[arm] == "min":
                score = -score
            goodness.append((score, bool(labels[str(row["attempt_key"])]["exact_mapping_correct"])))
        rows.append(
            {
                **aggregate,
                "scientific_role": (
                    "learned_energy_headline" if arm == ARM_CONVEX else "non_oracle_baseline"
                ),
                "candidate_auroc": _tie_aware_auroc(goodness),
            }
        )
    return rows


def _grouped_rows(selections: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Report group-weighted outcomes for all required analysis dimensions."""

    family_rows = []
    for field in ("model_family", "problem_family", "candidate_diversity"):
        for value in sorted({str(row[field]) for row in selections}):
            for arm in ARM_ORDER:
                family_rows.append(aggregate_selection_rows(selections, field, value, arm))
    difficulty_rows = [
        aggregate_selection_rows(selections, "difficulty", value, arm)
        for value in sorted({str(row["difficulty"]) for row in selections})
        for arm in ARM_ORDER
    ]
    return family_rows, difficulty_rows


def replay_headline_metrics(selection_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rebuild headline arm metrics from terminal per-group selection rows only."""

    rows = [dict(row, scope="overall") for row in selection_rows]
    return {
        arm: {
            key: value
            for key, value in aggregate_selection_rows(rows, "scope", "overall", arm).items()
            if key
            in {
                "group_count",
                "top1_accuracy",
                "false_acceptance_rate",
                "abstention_rate",
                "brier_score",
                "expected_calibration_error",
            }
        }
        for arm in ARM_ORDER
    }


def replay_manifest_payload(artifact: Mapping[str, Any]) -> JsonDict:
    """Serialize only the row evidence needed for independent headline replay."""

    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "selection_rows": artifact["selection_rows"],
    }


def replay_manifest(path: Path) -> JsonDict:
    """Load serialized rows and recompute the complete headline metric digest."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError("replay_manifest_schema_mismatch")
    metrics = replay_headline_metrics(manifest.get("selection_rows", []))
    return {"metrics": metrics, "headline_checksum": sha256_bytes(canonical_json(metrics))}


def _write_json(path: Path, value: Any) -> None:
    """Replace JSON atomically so interruption cannot leave a plausible partial file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}-", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def _fresh_process_replay(repo_root: Path, manifest_path: Path) -> JsonDict:
    """Use a clean interpreter so replay cannot reuse parent-process aggregates."""

    output_path = Path(manifest_path).with_suffix(".child.json")
    env = dict(os.environ)
    python_root = str(repo_root / "python")
    env["PYTHONPATH"] = python_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "carnot.experiment_6959_certified_energy_selection",
            "--replay-manifest",
            str(manifest_path),
            "--replay-output",
            str(output_path),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fresh_process_replay_failed:{completed.stderr[-300:]}")
    child = json.loads(output_path.read_text(encoding="utf-8"))
    output_path.unlink(missing_ok=True)
    return child


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding measured timing fields."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "latency_rows", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(payload))


def _expected_verdict(artifact: Mapping[str, Any]) -> str:
    """Map score state to the only permitted verdict class."""

    if artifact["honest_verdict"] == "blocked_certified_energy_selection":
        return "blocked"
    if artifact["certified_energy_positive_score"]:
        return "positive"
    if artifact["certified_selection_run_complete_score"]:
        return "null"
    return "partial"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject schema, oracle, checksum, verdict, or row-derived aggregate drift."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing_artifact_fields:{','.join(missing)}")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate_mismatch")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle_mismatch")
    if (
        artifact["certified_energy_positive_score"]
        and not artifact["certified_selection_run_complete_score"]
    ):
        raise ValueError("positive_without_completion")
    if artifact["verdict_class"] != _expected_verdict(artifact):
        raise ValueError("verdict_class_score_mismatch")
    if artifact["certified_selection_run_complete_score"]:
        replayed = replay_headline_metrics(artifact["selection_rows"])
        stored = {
            row["arm"]: {key: row[key] for key in replayed[row["arm"]]}
            for row in artifact["arm_rows"]
        }
        if stored != replayed:
            raise ValueError("aggregate_row_mismatch")
    if payload_checksum(artifact) != artifact["reproducibility_checksum"]:
        raise ValueError("reproducibility_checksum_mismatch")


def build_artifact(
    *,
    date: str,
    repo_root: Path,
    bank_path: Path,
    certificate_path: Path,
    energy_path: Path,
    replay_manifest_path: Path,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> JsonDict:
    """Build the full causal selection test from the three frozen predecessors."""

    started = time.perf_counter()
    arm_policy = freeze_arm_policy()
    bank = json.loads(Path(bank_path).read_text(encoding="utf-8"))
    energy = json.loads(Path(energy_path).read_text(encoding="utf-8"))
    certificates = json.loads(Path(certificate_path).read_text(encoding="utf-8"))
    preconditions = check_preconditions(repo_root, bank, certificates, energy, arm_policy)
    hashes = source_artifact_hashes(repo_root)
    if not preconditions["passed"]:
        return build_blocked_artifact(
            date=date,
            preconditions=preconditions,
            source_hashes=hashes,
            duration_s=time.perf_counter() - started,
        )

    candidates = [_candidate_from_attempt(row) for row in bank["attempt_rows"]]
    candidates.sort(key=lambda row: str(row["attempt_key"]))
    leakage_rows = [
        audit_selector_payload(str(row["attempt_key"]), row["selector_payload"])
        for row in candidates
    ]
    models = _energy_models(repo_root, energy)
    latency_rows = _score_energy_arms(candidates, models)
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["group_id"])].append(candidate)
    diversity_rows = {
        group_id: candidate_diversity_row(rows) for group_id, rows in sorted(grouped.items())
    }

    frozen_selections = [
        rank_group(grouped[group_id], arm) for group_id in sorted(grouped) for arm in ARM_ORDER
    ]
    tie_rows = [
        {
            "group_id": row["group_id"],
            "pair_id": row["pair_id"],
            "arm": row["arm"],
            "tie_count": row["tie_count"],
            "tied_attempt_keys": row["tied_attempt_keys"],
            "tie_policy": row["tie_policy"],
            "terminal": True,
        }
        for row in frozen_selections
    ]
    for group_id, rows in sorted(grouped.items()):
        for arm in ARM_ORDER:
            forward = rank_group(rows, arm)
            reverse = rank_group(list(reversed(rows)), arm)
            leakage_rows.append(
                {
                    "audit": "candidate_order_invariance",
                    "group_id": group_id,
                    "arm": arm,
                    "forward_selected": forward["selected_attempt_key"],
                    "reverse_selected": reverse["selected_attempt_key"],
                    "passed": forward["selection_frozen_hash"] == reverse["selection_frozen_hash"],
                }
            )
    leakage_rows.append(
        {
            "audit": "label_access_order",
            "arm_policy_frozen_order": 1,
            "certificate_preflight_order": 2,
            "selection_frozen_order": 3,
            "exact_evaluation_order": 4,
            "passed": True,
        }
    )

    labels = {
        sha256_bytes(canonical_json({"attempt_key": str(row["attempt_key"])})): row
        for row in certificates["proposal_rows"]
    }
    difficulty_by_group: dict[str, str] = {}
    for group_id, rows in grouped.items():
        values = {str(labels[str(row["attempt_key"])]["difficulty"]) for row in rows}
        difficulty_by_group[group_id] = sorted(values)[0]
    group_metadata = {
        group_id: {
            "difficulty": difficulty_by_group[group_id],
            "candidate_diversity": diversity_rows[group_id]["candidate_diversity"],
        }
        for group_id in grouped
    }
    selections = _evaluate_selections(frozen_selections, labels, group_metadata)
    selections = [dict(row, scope="overall") for row in selections]
    score_rows = {arm: _score_rows(candidates, arm) for arm in ARM_ORDER}
    arm_rows = _arm_rows(selections, score_rows, labels)
    strongest = max(
        BASELINE_ORDER,
        key=lambda arm: (
            next(row["top1_accuracy"] for row in arm_rows if row["arm"] == arm),
            -BASELINE_ORDER.index(arm),
        ),
    )
    selection_by_group_arm = {(str(row["group_id"]), str(row["arm"])): row for row in selections}
    oracle_rows = []
    headroom_rows = []
    paired_rows = []
    candidate_group_rows = []
    boolean_labels = {key: bool(row["exact_mapping_correct"]) for key, row in labels.items()}
    for group_id, group_candidates in sorted(grouped.items()):
        oracle = oracle_upper_bound_row(group_candidates, boolean_labels)
        oracle_rows.append(oracle)
        convex = selection_by_group_arm[(group_id, ARM_CONVEX)]
        baseline = selection_by_group_arm[(group_id, strongest)]
        headroom_rows.append(
            headroom_row(
                oracle,
                baseline_correct=bool(baseline["selected_exact_correct"]),
                convex_correct=bool(convex["selected_exact_correct"]),
                baseline_arm=strongest,
            )
        )
        paired_rows.append(
            {
                "group_id": group_id,
                "pair_id": oracle["pair_id"],
                "strongest_non_oracle_baseline": strongest,
                "convex_correct": bool(convex["selected_exact_correct"]),
                "baseline_correct": bool(baseline["selected_exact_correct"]),
                "paired_top1_delta": int(bool(convex["selected_exact_correct"]))
                - int(bool(baseline["selected_exact_correct"])),
                "exact_tie": bool(convex["selected_exact_correct"])
                == bool(baseline["selected_exact_correct"]),
                "terminal": True,
            }
        )
        candidate_group_rows.append(
            {
                **diversity_rows[group_id],
                "model_family": group_candidates[0]["model_family"],
                "problem_family": group_candidates[0]["problem_family"],
                "difficulty": difficulty_by_group[group_id],
                "group_has_correct_candidate": oracle["group_has_correct_candidate"],
            }
        )
    interval = paired_bootstrap_by_pair(paired_rows, RANDOM_SEED, bootstrap_samples)
    interval.update(
        {
            "comparison": f"{ARM_CONVEX}_minus_{strongest}",
            "strictly_above_zero": interval["ci95_lower"] is not None
            and interval["ci95_lower"] > 0,
        }
    )
    available_headroom = sum(int(row["available_headroom"]) for row in headroom_rows)
    captured_headroom = sum(
        int(row["captured_headroom"])
        for row in headroom_rows
        if row["captured_headroom"] is not None
    )
    headroom_capture = captured_headroom / available_headroom if available_headroom else None
    family_rows, difficulty_rows = _grouped_rows(selections)

    _write_json(
        replay_manifest_path,
        {"schema_version": REPLAY_SCHEMA_VERSION, "selection_rows": selections},
    )
    child_replay = _fresh_process_replay(repo_root, replay_manifest_path)
    expected_replay = replay_headline_metrics(selections)
    replay_match = child_replay.get("metrics") == expected_replay
    fresh_rows = [
        {
            "process": "fresh_python",
            "headline_checksum": child_replay.get("headline_checksum"),
            "expected_headline_checksum": sha256_bytes(canonical_json(expected_replay)),
            "replay_matches": replay_match,
            "terminal": True,
        }
    ]
    control_checks = {
        "leakage": all(row["passed"] for row in leakage_rows),
        "ties": len(tie_rows) == EXPECTED_GROUP_COUNT * len(ARM_ORDER)
        and all(row["terminal"] for row in tie_rows),
        "shuffled": all(row["terminal"] for row in selections if row["arm"] == ARM_SHUFFLED),
        "fixed": all(row["terminal"] for row in selections if row["arm"] == ARM_FIXED),
    }
    completion_checks = [
        _gate("candidate_count", EXPECTED_CANDIDATE_COUNT, len(candidates)),
        _gate("candidate_group_count", EXPECTED_GROUP_COUNT, len(grouped)),
        _gate("selection_row_count", EXPECTED_GROUP_COUNT * len(ARM_ORDER), len(selections)),
        _gate("all_selection_rows_terminal", True, all(row["terminal"] for row in selections)),
        _gate(
            "all_score_surfaces_complete",
            True,
            all(len(score_rows[arm]) == EXPECTED_CANDIDATE_COUNT for arm in ARM_ORDER),
        ),
        _gate("leakage_checks", True, control_checks["leakage"]),
        _gate("tie_checks", True, control_checks["ties"]),
        _gate("fresh_process_replay", True, replay_match),
    ]
    complete = int(all(row["passed"] for row in completion_checks))
    convex_auroc = next(row["candidate_auroc"] for row in arm_rows if row["arm"] == ARM_CONVEX)
    positive = int(
        positive_gate(
            complete=bool(complete),
            paired_ci_lower=interval["ci95_lower"],
            headroom_captured=headroom_capture,
            control_checks=control_checks,
            candidate_auroc=convex_auroc,
        )
    )
    positive_checks = [
        _gate(
            "paired_ci95_lower_strictly_positive",
            True,
            interval["ci95_lower"] is not None and interval["ci95_lower"] > 0,
        ),
        _gate(
            "headroom_capture_at_least_20_percent",
            True,
            headroom_capture is not None and headroom_capture >= 0.2,
        ),
        _gate("all_control_checks", True, all(control_checks.values())),
        _gate("oracle_not_deployed", True, all(not row["oracle_deployed"] for row in oracle_rows)),
    ]
    gate_summary = _gate_summary([*completion_checks, *positive_checks])
    gate_summary["strongest_non_oracle_baseline"] = strongest
    gate_summary["available_oracle_headroom"] = available_headroom
    gate_summary["captured_oracle_headroom"] = captured_headroom
    gate_summary["headroom_capture_rate"] = headroom_capture
    gate_summary["control_checks"] = control_checks

    verdict_class = "positive" if positive else "null" if complete else "partial"
    honest_verdict = (
        "complete_positive_certified_energy_selection"
        if positive
        else "complete_null_certified_energy_selection"
        if complete
        else "partial_certified_energy_selection"
    )
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": time.perf_counter() - started,
        "source_artifact_hashes": hashes,
        "rows": selections,
        "candidate_group_rows": candidate_group_rows,
        "candidate_rows": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in candidates
        ],
        "split_rows": [
            {
                "split": "frozen_evaluation",
                "candidate_count": len(candidates),
                "group_count": len(grouped),
                "evaluation_labels_sealed": True,
                "terminal": True,
            }
        ],
        "arm_rows": arm_rows,
        "convex_energy_rows": score_rows[ARM_CONVEX],
        "unconstrained_energy_rows": score_rows[ARM_MLP],
        "linear_energy_rows": score_rows[ARM_LINEAR],
        "likelihood_rows": score_rows[ARM_LIKELIHOOD],
        "syntax_rows": score_rows[ARM_SYNTAX],
        "confidence_rows": score_rows[ARM_CONFIDENCE],
        "shuffled_energy_rows": score_rows[ARM_SHUFFLED],
        "fixed_order_rows": score_rows[ARM_FIXED],
        "oracle_upper_bound_rows": oracle_rows,
        "selection_rows": selections,
        "abstention_rows": [
            {
                "group_id": row["group_id"],
                "pair_id": row["pair_id"],
                "arm": row["arm"],
                "model_family": row["model_family"],
                "problem_family": row["problem_family"],
                "difficulty": row["difficulty"],
                "candidate_diversity": row["candidate_diversity"],
                "abstained": row["abstained"],
                "terminal": True,
            }
            for row in selections
        ],
        "headroom_rows": headroom_rows,
        "family_rows": family_rows,
        "difficulty_rows": difficulty_rows,
        "latency_rows": latency_rows,
        "paired_metric_rows": paired_rows,
        "confidence_interval_rows": [interval],
        "leakage_rows": leakage_rows,
        "tie_rows": tie_rows,
        "fresh_process_replay_rows": fresh_rows,
        "random_seed": RANDOM_SEED,
        "certified_selection_run_complete_score": complete,
        "certified_energy_positive_score": positive,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(date: str, repo_root: Path | None = None, output_path: Path | None = None) -> JsonDict:
    """Build, validate, and atomically write the canonical selection artifact."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    output = Path(output_path) if output_path is not None else root / OUTPUT_PATH
    artifact = build_artifact(
        date=date,
        repo_root=root,
        bank_path=root / BANK_PATH,
        certificate_path=root / CERTIFICATE_PATH,
        energy_path=root / ENERGY_PATH,
        replay_manifest_path=root / REPLAY_MANIFEST_PATH,
    )
    validate_artifact(artifact)
    _write_json(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - required CLI surface.
    """Expose normal execution and the private fresh-process replay boundary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260904")
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--replay-output", type=Path)
    args = parser.parse_args(argv)
    if args.replay_manifest is not None:
        if args.replay_output is None:
            parser.error("--replay-output is required with --replay-manifest")
        _write_json(args.replay_output, replay_manifest(args.replay_manifest))
        return 0
    artifact = run(args.date)
    print(
        json.dumps(
            {
                field: artifact[field]
                for field in (
                    "certified_selection_run_complete_score",
                    "certified_energy_positive_score",
                    "verdict_class",
                    "honest_verdict",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution is the replay boundary.
    raise SystemExit(main())
