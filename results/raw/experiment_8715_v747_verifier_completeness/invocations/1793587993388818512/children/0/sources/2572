"""Freeze the Exp6869 semantic rule from calibration rows only.

The reducer reads plaintext labels only from the calibration manifest. It
checks the held score file by hash without parsing that file. This boundary
keeps rule selection separate from the later held evaluation in Exp6870.

Spec refs: REQ-VERIFY-6869 and SCENARIO-VERIFY-6869-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
import tempfile
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_PATH = Path("results/experiment_6869_calibration_only_paired_semantic_rule.json")
EXP6867_PATH = Path("results/experiment_6867_tokenizer_aware_semantic_preregistration_v2.json")
EXP6868_PATH = Path("results/experiment_6868_three_family_semantic_scoring_stream_v2.json")
CALIBRATION_SIDECAR_PATH = Path(
    "results/sidecars/experiment_6868_semantic_scoring_calibration.json"
)
HELD_SIDECAR_PATH = Path("results/sidecars/experiment_6868_semantic_scoring_held.json")
WRAPPER_PATH = Path("scripts/experiments/experiment_6869_calibration_only_paired_semantic_rule.py")
TEST_PATH = Path("tests/python/test_experiment_6869_calibration_only_paired_semantic_rule.py")
MODULE_PATH = Path("python/carnot/experiment_6869_calibration_only_paired_semantic_rule.py")

SCHEMA = "carnot.experiment_6869.calibration_only_paired_semantic_rule.v1"
INFERENCE_SUBSTRATE = "deterministic CPU calibration-only paired reduction replay"
RUN_DATE = "20260902"
RANDOM_SEED = 6867
EXPECTED_EXP6867_SHA256 = "sha256:31f445cf96627221d286db6859392c5a19eef211b703dc91dc663b3d3e0cc480"
EXPECTED_EXP6868_SHA256 = "sha256:f870c42bebf52f4beb1ff9c2a4b96f51a8f707880d80458f63da27a75576a006"
EXPECTED_CALIBRATION_SIDECAR_SHA256 = (
    "sha256:af3027bc81d9f0050799b709252490bc211c2c5f7f251f7a521c30f2f9e87a17"
)
EXPECTED_HELD_SIDECAR_SHA256 = (
    "sha256:5fd8636376b19a2345a550e1bb6f3a56b5df919dcb457dda700f441e5dfafa3b"
)
BLOCKED_VERDICT = "complete_blocked_calibration_only_paired_semantic_rule"
READY_VERDICT = "complete_positive_calibration_only_paired_semantic_rule_ready_no_held_access"
NULL_VERDICT = "complete_null_calibration_does_not_support_paired_semantic_rule"
MINIMUM_EFFECT = 0.0
MISSINGNESS_CEILING = 0.0
MINIMUM_HELD_GROUPS = 20

MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_FAMILIES = {
    MODEL_SPECS[0]: "qwen_moe",
    MODEL_SPECS[1]: "gemma_dense",
    MODEL_SPECS[2]: "gemma_moe",
}
SEMANTIC_FAMILIES = (
    "arc_guard",
    "diagnostic_obligation",
    "exact_energy",
    "memory_guard",
    "satisfaction_predicate",
)
NUISANCE_CONTROLS = (
    "identifier",
    "order",
    "normalization",
    "label_position",
    "token_count",
    "character_length",
    "surface_form",
)
FROZEN_BOOTSTRAP_CONTRACT = {
    "method": "BCa_cluster_bootstrap",
    "confidence_level": 0.95,
    "resamples": 10000,
    "cluster_unit": "semantic_group_identity",
    "random_seed": RANDOM_SEED,
    "family_weighting": "equal_semantic_family_weight",
}
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "calibration_access_log",
    "held_label_access_count",
    "rows",
    "per_model_calibration_effects",
    "per_family_calibration_effects",
    "pooled_calibration_effect",
    "nuisance_control_effects",
    "missing_cell_rows",
    "bootstrap_rows",
    "family_replication_result",
    "frozen_held_reducer_hash",
    "held_acceptance_contract",
    "random_seed",
    "reproducibility_checksum",
    "semantic_contrast_rule_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each top-level field states why it exists.",
    "preconditions_checked": "Exact hashes and seals stop drift before reduction.",
    "inference_substrate": "The value states that no model or held evaluator ran.",
    "duration_s": "Measured CPU wall time exposes skipped reduction work.",
    "source_artifact_hashes": "Exact source bytes bind the frozen rule.",
    "calibration_access_log": "The log makes every allowed file and field access auditable.",
    "held_label_access_count": "Zero proves that calibration did not open a held label.",
    "rows": "Each model-group-control row preserves disaggregated evidence.",
    "per_model_calibration_effects": "Model results prevent pooled masking.",
    "per_family_calibration_effects": "Family results enforce replication.",
    "pooled_calibration_effect": "Equal weighting reports the pooled estimate separately.",
    "nuisance_control_effects": "Control bounds test whether nuisance explains the effect.",
    "missing_cell_rows": "Missing evidence stays missing instead of becoming zero.",
    "bootstrap_rows": "Interval receipts bind every reported confidence bound.",
    "family_replication_result": "The result prevents one family from carrying the claim.",
    "frozen_held_reducer_hash": "The hash prevents held-time reducer tuning.",
    "held_acceptance_contract": "Exact thresholds freeze held acceptance before labels open.",
    "random_seed": "One seed makes every bootstrap repeatable.",
    "reproducibility_checksum": "One digest binds all stable artifact content.",
    "semantic_contrast_rule_ready_score": "Readiness freezes a rule, not a held result.",
    "gate_check_summary": "The first failure keeps exact expected and observed values.",
    "verifier_is_oracle": "False keeps token likelihood separate from exact authority.",
    "verdict_class": "A closed class makes the terminal state machine-readable.",
    "honest_verdict": "A complete prefix records one terminal outcome.",
    "schema": "The version prevents silent consumer drift.",
    "experiment_id": "The fixed number prevents artifact confusion.",
    "run_date": "The supplied date binds this execution.",
    "status": "The status distinguishes complete and blocked execution.",
    "result_path": "The stable path gives Exp6870 one exact source.",
    "spec_refs": "Requirement anchors connect spec, tests, and evidence.",
}


class HeldLabelAccessError(RuntimeError):
    """Raised before a held label value enters calibration memory."""


class CalibrationLabelStore:
    """Expose calibration labels and deny every held identity before lookup."""

    def __init__(
        self,
        calibration_labels: Mapping[str, Mapping[str, bool]],
        held_group_ids: Iterable[str],
        *,
        source_path: str,
        access_log: list[JsonDict] | None = None,
    ) -> None:
        self._calibration_labels = {
            str(group): {str(candidate): bool(label) for candidate, label in labels.items()}
            for group, labels in calibration_labels.items()
        }
        self._held_group_ids = {str(value) for value in held_group_ids}
        self.source_path = str(source_path)
        self.access_log = access_log if access_log is not None else []
        self.held_label_access_count = 0
        self.held_label_access_attempt_count = 0

    def load(self, semantic_group_identity: str) -> dict[str, bool]:
        """Return calibration labels or deny a held identity before value lookup."""

        group_id = str(semantic_group_identity)
        if group_id in self._held_group_ids:
            self.held_label_access_attempt_count += 1
            self.access_log.append(
                {
                    "path": self.source_path,
                    "access_scope": "held_label_guard",
                    "field": "sealed_held_group_manifest[*].label",
                    "semantic_group_identity": group_id,
                    "outcome": "denied_before_value_load",
                }
            )
            raise HeldLabelAccessError(f"held label access denied: {group_id}")
        labels = dict(self._calibration_labels[group_id])
        self.access_log.append(
            {
                "path": self.source_path,
                "access_scope": "calibration_label",
                "field": "calibration_group_manifest[*].candidate_labels_by_slot",
                "semantic_group_identity": group_id,
                "outcome": "loaded",
            }
        )
        return labels


def canonical_json(value: Any) -> str:
    """Serialize content consistently for hashes and checksums."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return one prefixed digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON value."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file without parsing its content."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Build one exact comparison record."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve all checks and expose the first exact failure."""

    rows = [dict(row) for row in checks]
    failed = [row for row in rows if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "passed": first is None,
        "checks": rows,
        "failed_check": first.get("check") if first else None,
        "failed_checks": [str(row.get("check")) for row in failed],
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
    }


def split_collisions(calibration_ids: Iterable[str], held_ids: Iterable[str]) -> list[str]:
    """Return sorted group identities assigned to both splits."""

    return sorted({str(value) for value in calibration_ids} & {str(value) for value in held_ids})


def _contains_plaintext_label(value: Any) -> bool:
    """Detect label keys that a sealed held manifest must not contain."""

    if isinstance(value, Mapping):
        if {"expected_label", "candidate_labels_by_slot", "label"} & set(value):
            return True
        return any(_contains_plaintext_label(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_plaintext_label(item) for item in value)
    return False


def evaluate_preconditions(
    *,
    exp6867: Mapping[str, Any],
    exp6868: Mapping[str, Any],
    exp6867_sha256: str,
    exp6868_sha256: str,
    calibration_sidecar_sha256: str,
    calibration_payload_sha256: str,
    held_sidecar_sha256: str,
) -> JsonDict:
    """Check exact upstream hashes, split authority, and held seals."""

    calibration_manifest = exp6867.get("calibration_group_manifest")
    held_manifest = exp6867.get("sealed_held_group_manifest")
    calibration_rows = calibration_manifest if isinstance(calibration_manifest, list) else []
    held_rows = held_manifest if isinstance(held_manifest, list) else []
    calibration_ids = [
        str(row.get("semantic_group_identity"))
        for row in calibration_rows
        if isinstance(row, Mapping)
    ]
    held_ids = [
        str(row.get("semantic_group_identity")) for row in held_rows if isinstance(row, Mapping)
    ]
    calibration_labels_valid = bool(calibration_rows) and all(
        isinstance(row, Mapping)
        and row.get("candidate_labels_by_slot") == [True, False]
        and len(row.get("candidate_ids") or []) == 2
        for row in calibration_rows
    )
    held_sealed = (
        bool(held_rows)
        and not _contains_plaintext_label(held_rows)
        and all(
            isinstance(row, Mapping)
            and str(row.get("label_commitment") or "").startswith("sha256:")
            for row in held_rows
        )
    )
    calibration_score_manifest = exp6868.get("calibration_score_manifest")
    calibration_score_manifest = (
        calibration_score_manifest if isinstance(calibration_score_manifest, Mapping) else {}
    )
    held_score_manifest = exp6868.get("sealed_held_score_manifest")
    held_score_manifest = held_score_manifest if isinstance(held_score_manifest, Mapping) else {}
    checks = [
        gate_check(
            "semantic_contrast_stream_v2_complete_score",
            1,
            exp6868.get("semantic_contrast_stream_v2_complete_score"),
        ),
        gate_check("exact_exp6867_hash", EXPECTED_EXP6867_SHA256, exp6867_sha256),
        gate_check("exact_exp6868_hash", EXPECTED_EXP6868_SHA256, exp6868_sha256),
        gate_check("readable_calibration_manifest", True, calibration_labels_valid),
        gate_check("sealed_held_manifest", True, held_sealed),
        gate_check("split_group_disjointness", [], split_collisions(calibration_ids, held_ids)),
        gate_check(
            "calibration_sidecar_file_hash",
            calibration_score_manifest.get("sha256"),
            calibration_sidecar_sha256,
        ),
        gate_check(
            "calibration_sidecar_payload_hash",
            calibration_score_manifest.get("payload_sha256"),
            calibration_payload_sha256,
        ),
        gate_check(
            "held_sidecar_file_hash_bytes_only",
            held_score_manifest.get("sha256"),
            held_sidecar_sha256,
        ),
        gate_check("held_score_labels_present", False, held_score_manifest.get("labels_present")),
        gate_check("held_score_effect_reduced", False, held_score_manifest.get("effect_reduced")),
    ]
    return gate_summary(checks)


def score_row_hash(row: Mapping[str, Any]) -> str:
    """Recompute the Exp6868 row digest without trusting its receipt."""

    unsigned = dict(row)
    unsigned["row_hash"] = ""
    return sha256_json(unsigned)


def _missing_cell(control_cell: Mapping[str, Any], reason: str, detail: Any) -> JsonDict:
    """Create one typed no-imputation receipt for a failed group cell."""

    return {
        "model_hf_id": control_cell.get("model_hf_id"),
        "model_family": control_cell.get("model_family"),
        "semantic_family": control_cell.get("semantic_family"),
        "semantic_group_identity": control_cell.get("semantic_group_identity"),
        "reason": reason,
        "detail": detail,
        "imputed": False,
    }


def _control_receipt_errors(cell: Mapping[str, Any]) -> list[str]:
    """Validate the structural nuisance matches frozen by Exp6867."""

    errors: list[str] = []
    candidate_ids = cell.get("candidate_ids")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) != 2
        or len(set(candidate_ids)) != 2
    ):
        errors.append("identifier")
    token_counts = cell.get("candidate_token_counts")
    if not isinstance(token_counts, Mapping) or token_counts.get("slot_0") != token_counts.get(
        "slot_1"
    ):
        errors.append("token_count")
    character_counts = cell.get("candidate_character_counts")
    if not isinstance(character_counts, Mapping) or character_counts.get(
        "slot_0"
    ) != character_counts.get("slot_1"):
        errors.append("character_length")
    prompt_ids = cell.get("paired_prompt_token_ids")
    if not isinstance(prompt_ids, Mapping) or prompt_ids.get("base") != prompt_ids.get(
        "label_swap"
    ):
        errors.extend(["order", "label_position"])
    if cell.get("presentation_order") != {"base": [0, 1], "label_swap": [1, 0]}:
        errors.extend(["order", "label_position"])
    if cell.get("normalization") != "NFC":
        errors.append("normalization")
    surface = cell.get("surface_control")
    if not isinstance(surface, Mapping) or not all(
        str(surface.get(key) or "").startswith("sha256:")
        for key in ("raw_prompt_template_sha256", "candidate_sequence_template_sha256")
    ):
        errors.append("surface_form")
    if cell.get("accepted") is not True or cell.get("rejection_reasons") not in ([], None):
        errors.append("accepted_cell")
    return sorted(set(errors))


def _token_mean(row: Mapping[str, Any]) -> float:
    """Recompute one candidate score from raw token values."""

    if row.get("row_hash") != score_row_hash(row):
        raise ValueError("score_row_hash_invalid")
    values = row.get("token_logprobs")
    if not isinstance(values, list) or not values:
        raise ValueError("token_logprobs_missing")
    scores = [float(value) for value in values]
    if row.get("finite") is not True or not all(math.isfinite(value) for value in scores):
        raise ValueError("nonfinite_token_logprob")
    return sum(scores) / len(scores)


def _effect_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one reduced row without its self-referential field."""

    unsigned = dict(row)
    unsigned["row_hash"] = ""
    return sha256_json(unsigned)


def reduce_group_rows(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    labels: Mapping[str, bool],
    control_cell: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Reduce one model-group cell into seven explicit nuisance rows."""

    receipt_errors = _control_receipt_errors(control_cell)
    if receipt_errors:
        return [], [_missing_cell(control_cell, "nuisance_match_failed", receipt_errors)]
    if len(labels) != 2 or sorted(bool(value) for value in labels.values()) != [False, True]:
        return [], [_missing_cell(control_cell, "calibration_label_pair_invalid", sorted(labels))]

    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in score_rows:
        key = (str(row.get("nuisance_transform_identity")), str(row.get("candidate_id")))
        if key in indexed:
            return [], [_missing_cell(control_cell, "duplicate_score_cell", list(key))]
        indexed[key] = row
    expected = {
        (transform, candidate) for transform in ("base", "label_swap") for candidate in labels
    }
    if set(indexed) != expected:
        return [], [
            _missing_cell(
                control_cell,
                "required_score_cell_missing",
                {
                    "expected": [list(value) for value in sorted(expected)],
                    "observed": [list(value) for value in sorted(indexed)],
                },
            )
        ]
    try:
        scores = {key: _token_mean(row) for key, row in indexed.items()}
    except ValueError as exc:
        return [], [_missing_cell(control_cell, str(exc), "raw token row rejected")]

    valid_id = next(candidate for candidate, label in labels.items() if label)
    invalid_id = next(candidate for candidate, label in labels.items() if not label)
    base_valid = scores[("base", valid_id)]
    base_invalid = scores[("base", invalid_id)]
    swap_valid = scores[("label_swap", valid_id)]
    swap_invalid = scores[("label_swap", invalid_id)]
    base_contrast = base_valid - base_invalid
    swap_contrast = swap_valid - swap_invalid
    semantic_effect = (base_contrast + swap_contrast) / 2.0
    matched_effects = {
        "identifier": 0.0,
        "order": ((base_valid + base_invalid) - (swap_valid + swap_invalid)) / 2.0,
        "normalization": 0.0,
        "label_position": (base_contrast - swap_contrast) / 2.0,
        "token_count": 0.0,
        "character_length": 0.0,
        "surface_form": 0.0,
    }
    sign_reversal = base_contrast * swap_contrast < 0.0
    zero_headroom = base_contrast == 0.0 and swap_contrast == 0.0
    reduced: list[JsonDict] = []
    for control in NUISANCE_CONTROLS:
        nuisance_effect = matched_effects[control]
        row: JsonDict = {
            "model_hf_id": control_cell.get("model_hf_id"),
            "model_family": control_cell.get("model_family"),
            "semantic_family": control_cell.get("semantic_family"),
            "semantic_group_identity": control_cell.get("semantic_group_identity"),
            "nuisance_control": control,
            "orientation": "valid_minus_invalid",
            "normalization": "mean_token_log_probability",
            "valid_mean_log_likelihood_per_token": (base_valid + swap_valid) / 2.0,
            "invalid_mean_log_likelihood_per_token": (base_invalid + swap_invalid) / 2.0,
            "base_paired_contrast": base_contrast,
            "label_swap_paired_contrast": swap_contrast,
            "semantic_effect": semantic_effect,
            "matched_nuisance_effect": nuisance_effect,
            "nuisance_difference_in_differences": semantic_effect - nuisance_effect,
            "nuisance_wins": abs(nuisance_effect) >= semantic_effect,
            "sign_reversal": sign_reversal,
            "zero_headroom": zero_headroom,
            "status": "complete",
            "row_hash": "",
        }
        row["row_hash"] = _effect_row_hash(row)
        reduced.append(row)
    return reduced, []


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated quantile from sorted values."""

    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = min(1.0, max(0.0, probability)) * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _bca_bounds(
    estimate: float,
    bootstrap_values: Sequence[float],
    jackknife_values: Sequence[float],
    *,
    confidence_level: float,
) -> tuple[float, float]:
    """Compute bias-corrected and accelerated bootstrap bounds."""

    boot = sorted(float(value) for value in bootstrap_values)
    if not boot or all(value == boot[0] for value in boot):
        return estimate, estimate
    normal = NormalDist()
    less = sum(value < estimate for value in boot)
    equal = sum(value == estimate for value in boot)
    probability = (less + 0.5 * equal) / len(boot)
    epsilon = 0.5 / len(boot)
    z0 = normal.inv_cdf(min(1.0 - epsilon, max(epsilon, probability)))
    jack = [float(value) for value in jackknife_values]
    if len(jack) < 2:
        acceleration = 0.0
    else:
        jack_mean = sum(jack) / len(jack)
        deviations = [jack_mean - value for value in jack]
        denominator = 6.0 * sum(value * value for value in deviations) ** 1.5
        acceleration = sum(value**3 for value in deviations) / denominator if denominator else 0.0
    alpha = (1.0 - confidence_level) / 2.0

    def adjusted(probability_value: float) -> float:
        z = normal.inv_cdf(probability_value)
        denominator = 1.0 - acceleration * (z0 + z)
        adjusted_z = z0 + (z0 + z) / denominator if denominator else z0 + z
        return normal.cdf(adjusted_z)

    return _percentile(boot, adjusted(alpha)), _percentile(boot, adjusted(1.0 - alpha))


def bca_mean_interval(
    values: Iterable[float],
    *,
    random_seed: int = RANDOM_SEED,
    resamples: int = 10000,
    confidence_level: float = 0.95,
) -> JsonDict:
    """Return the preregistered BCa interval for scalar group effects."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("bootstrap_values_empty")
    estimate = sum(ordered) / len(ordered)
    rng = random.Random(int(random_seed))
    bootstrap_values = [
        sum(ordered[rng.randrange(len(ordered))] for _ in ordered) / len(ordered)
        for _ in range(int(resamples))
    ]
    jackknife_values = (
        [
            sum(ordered[:index] + ordered[index + 1 :]) / (len(ordered) - 1)
            for index in range(len(ordered))
        ]
        if len(ordered) > 1
        else [estimate]
    )
    lower, upper = _bca_bounds(
        estimate,
        bootstrap_values,
        jackknife_values,
        confidence_level=confidence_level,
    )
    return {
        "method": "BCa_cluster_bootstrap",
        "confidence_level": confidence_level,
        "resamples": int(resamples),
        "cluster_unit": "semantic_group_identity",
        "random_seed": int(random_seed),
        "estimate": estimate,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_clusters": len(ordered),
    }


def _equal_weight_estimate(rows: Sequence[Mapping[str, Any]], value_field: str) -> float:
    """Give equal weight to models, families, and then groups."""

    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_cell[(str(row.get("model_hf_id")), str(row.get("semantic_family")))].append(
            float(row[value_field])
        )
    if not by_cell:
        raise ValueError("aggregate_rows_empty")
    by_model: dict[str, list[float]] = defaultdict(list)
    for (model, _family), values in sorted(by_cell.items()):
        by_model[model].append(sum(values) / len(values))
    model_effects = [sum(values) / len(values) for _, values in sorted(by_model.items())]
    return sum(model_effects) / len(model_effects)


def stratified_bca_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    random_seed: int = RANDOM_SEED,
    resamples: int = 10000,
) -> JsonDict:
    """Bootstrap semantic groups while keeping semantic-family weights fixed."""

    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("semantic_family")),
            str(row.get("semantic_group_identity")),
            str(row.get("model_hf_id")),
        ),
    )
    if not ordered:
        raise ValueError("bootstrap_rows_empty")
    estimate = _equal_weight_estimate(ordered, value_field)
    # Each model-family cell receives equal weight. Vectorized sampling keeps
    # 10,000 preregistered resamples practical while each draw remains a whole
    # semantic-group observation.
    by_cell: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for row in ordered:
        by_cell[(str(row.get("model_hf_id")), str(row.get("semantic_family")))].append(
            (str(row.get("semantic_group_identity")), float(row[value_field]))
        )
    rng = np.random.Generator(np.random.PCG64(int(random_seed)))
    cell_bootstrap: list[np.ndarray] = []
    for cell in sorted(by_cell):
        values = np.asarray(
            [value for _group, value in sorted(by_cell[cell])],
            dtype=np.float64,
        )
        indexes = rng.integers(0, len(values), size=(int(resamples), len(values)))
        cell_bootstrap.append(values[indexes].mean(axis=1))
    bootstrap_values = np.stack(cell_bootstrap, axis=1).mean(axis=1).tolist()
    cluster_ids = sorted({str(row.get("semantic_group_identity")) for row in ordered})
    jackknife_values = (
        [
            _equal_weight_estimate(
                [row for row in ordered if str(row.get("semantic_group_identity")) != group_id],
                value_field,
            )
            for group_id in cluster_ids
        ]
        if len(cluster_ids) > 1
        else [estimate]
    )
    lower, upper = _bca_bounds(
        estimate,
        bootstrap_values,
        jackknife_values,
        confidence_level=float(FROZEN_BOOTSTRAP_CONTRACT["confidence_level"]),
    )
    return {
        "method": FROZEN_BOOTSTRAP_CONTRACT["method"],
        "confidence_level": FROZEN_BOOTSTRAP_CONTRACT["confidence_level"],
        "resamples": int(resamples),
        "cluster_unit": FROZEN_BOOTSTRAP_CONTRACT["cluster_unit"],
        "random_seed": int(random_seed),
        "family_weighting": FROZEN_BOOTSTRAP_CONTRACT["family_weighting"],
        "estimate": estimate,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_clusters": len(cluster_ids),
    }


def bootstrap_contract_errors(manifest: Mapping[str, Any]) -> list[str]:
    """Name every interval setting that drifted from Exp6867."""

    return [
        f"bootstrap.{key}"
        for key, expected in FROZEN_BOOTSTRAP_CONTRACT.items()
        if manifest.get(key) != expected
    ]


def family_replication_result(
    family_rows: Sequence[Mapping[str, Any]],
    *,
    required_models: Sequence[str] = MODEL_SPECS,
    required_families: Sequence[str] = SEMANTIC_FAMILIES,
) -> JsonDict:
    """Require every model-family estimate to have the same positive sign."""

    indexed = {
        (str(row.get("model_hf_id")), str(row.get("semantic_family"))): row for row in family_rows
    }
    missing = [
        {"model_hf_id": model, "semantic_family": family}
        for model in required_models
        for family in required_families
        if (str(model), str(family)) not in indexed
    ]
    failed = [
        {"model_hf_id": str(model), "semantic_family": str(family)}
        for model in required_models
        for family in required_families
        if (str(model), str(family)) in indexed
        and float(indexed[(str(model), str(family))].get("estimate", 0.0)) <= 0.0
    ]
    signs = {
        model: [
            1 if float(indexed[(str(model), str(family))]["estimate"]) > 0.0 else -1
            for family in required_families
            if (str(model), str(family)) in indexed
        ]
        for model in required_models
    }
    passed = (
        not missing
        and not failed
        and all(
            len(model_signs) == len(required_families) and set(model_signs) == {1}
            for model_signs in signs.values()
        )
    )
    return {
        "passed": passed,
        "rule": "all_five_semantic_families_positive_in_all_three_models",
        "missing_cells": missing,
        "failed_cells": failed,
        "model_signs": signs,
    }


def rule_readiness(
    *,
    per_model_effects: Sequence[Mapping[str, Any]],
    per_family_effects: Sequence[Mapping[str, Any]],
    nuisance_effects: Sequence[Mapping[str, Any]],
    missing_cell_rows: Sequence[Mapping[str, Any]],
    sign_reversal_count: int,
    preconditions_passed: bool,
    dry_run_without_held_access: bool,
    required_models: Sequence[str] = MODEL_SPECS,
    required_families: Sequence[str] = SEMANTIC_FAMILIES,
) -> JsonDict:
    """Apply the frozen conjunction without letting pooled results override cells."""

    models = {str(row.get("model_hf_id")): row for row in per_model_effects}
    coverage_passed = set(models) == {str(model) for model in required_models}
    lower_bounds_passed = coverage_passed and all(
        float(models[str(model)].get("lower_bound", -math.inf)) > MINIMUM_EFFECT
        for model in required_models
    )
    nuisance_by_model: dict[str, list[float]] = defaultdict(list)
    for row in nuisance_effects:
        nuisance_by_model[str(row.get("model_hf_id"))].append(
            float(row.get("absolute_upper_bound", math.inf))
        )
    nuisance_passed = coverage_passed and all(
        nuisance_by_model[str(model)]
        and float(models[str(model)].get("estimate", -math.inf))
        > max(nuisance_by_model[str(model)])
        for model in required_models
    )
    replication = family_replication_result(
        per_family_effects,
        required_models=required_models,
        required_families=required_families,
    )
    checks = [
        gate_check("preconditions_passed", True, bool(preconditions_passed)),
        gate_check("required_model_coverage", True, coverage_passed),
        gate_check("model_lower_bound_exceeds_minimum_effect", True, lower_bounds_passed),
        gate_check("semantic_effect_exceeds_nuisance", True, nuisance_passed),
        gate_check("family_replication", True, replication["passed"]),
        gate_check("missingness_ceiling", 0, len(missing_cell_rows)),
        gate_check("transform_sign_consistency", 0, int(sign_reversal_count)),
        gate_check("dry_run_without_held_access", True, bool(dry_run_without_held_access)),
    ]
    summary = gate_summary(checks)
    summary.update(
        {
            "semantic_contrast_rule_ready_score": int(summary["passed"]),
            "family_replication_result": replication,
            "pooled_or_model_win_cannot_override_family_failure": bool(
                lower_bounds_passed and replication["passed"] is not True
            ),
        }
    )
    return summary


def freeze_acceptance_contract(
    nuisance_effects: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Freeze exact held thresholds from calibration control bounds."""

    nuisance_ceiling = max(
        (float(row.get("absolute_upper_bound", 0.0)) for row in nuisance_effects),
        default=0.0,
    )
    return {
        "orientation": "valid_minus_invalid",
        "normalization": "mean_token_log_probability_recomputed_from_token_logprobs",
        "semantic_effect": "mean_of_base_and_label_swap_valid_minus_invalid_contrasts",
        "nuisance_difference_in_differences": (
            "semantic_effect_minus_signed_matched_nuisance_effect"
        ),
        "bootstrap_interval": deepcopy(FROZEN_BOOTSTRAP_CONTRACT),
        "minimum_effect": MINIMUM_EFFECT,
        "minimum_effect_rule": "every_model_lower_bound_strictly_above_minimum_effect",
        "maximum_nuisance_effect": nuisance_ceiling,
        "nuisance_rule": "every_model_effect_strictly_exceeds_each_absolute_nuisance_bound",
        "family_replication_rule": (
            "all_five_semantic_families_positive_in_each_model_and_all_models_positive"
        ),
        "pooled_aggregation": {
            "model_weighting": "equal_model_weight",
            "family_weighting": "equal_semantic_family_weight",
            "group_weighting": "equal_semantic_group_weight_within_model_family",
            "pooled_win_overrides_cell_failure": False,
        },
        "missing_cell_rule": {
            "imputation": "none",
            "analysis_set": "complete_nuisance_eligible_cells_only",
            "maximum_missing_fraction": MISSINGNESS_CEILING,
            "minimum_held_groups_per_model": MINIMUM_HELD_GROUPS,
        },
        "disqualification_rules": [
            "source_hash_drift",
            "split_collision",
            "score_sidecar_hash_drift",
            "row_hash_or_token_alignment_failure",
            "bootstrap_contract_drift",
            "missing_required_cell",
            "transform_sign_reversal",
            "model_or_family_replication_failure",
            "held_label_access_during_calibration",
        ],
    }


def frozen_held_reducer_hash(contract: Mapping[str, Any]) -> str:
    """Bind held reduction source and the exact acceptance contract."""

    functions = (
        reduce_group_rows,
        stratified_bca_interval,
        family_replication_result,
        rule_readiness,
        freeze_acceptance_contract,
    )
    source = "\n".join(inspect.getsource(function) for function in functions)
    return sha256_json({"source": source, "contract": dict(contract)})


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable content while excluding wall time and the digest itself."""

    excluded = {"duration_s", "field_principles", "reproducibility_checksum"}
    return sha256_json({key: value for key, value in artifact.items() if key not in excluded})


def _finish_artifact(artifact: JsonDict) -> JsonDict:
    """Attach complete principles and one stable checksum."""

    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, "This field preserves calibration-only rule evidence.")
        for key in artifact
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _base_artifact(*, run_date: str, duration_s: float) -> JsonDict:
    """Create the complete schema before any precondition can fail."""

    return {
        "schema": SCHEMA,
        "experiment_id": 6869,
        "run_date": str(run_date),
        "status": "blocked",
        "result_path": RESULT_PATH.as_posix(),
        "spec_refs": ["REQ-VERIFY-6869", "SCENARIO-VERIFY-6869-*"],
        "field_principles": {},
        "preconditions_checked": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": {},
        "calibration_access_log": [],
        "held_label_access_count": 0,
        "rows": [],
        "per_model_calibration_effects": [],
        "per_family_calibration_effects": [],
        "pooled_calibration_effect": {},
        "nuisance_control_effects": [],
        "missing_cell_rows": [],
        "bootstrap_rows": [],
        "family_replication_result": {},
        "frozen_held_reducer_hash": "",
        "held_acceptance_contract": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "semantic_contrast_rule_ready_score": 0,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Mapping[str, Any],
    access_log: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Return complete blocked evidence for an input or seal failure."""

    artifact = _base_artifact(run_date=run_date, duration_s=duration_s)
    contract = freeze_acceptance_contract([])
    reducer_hash = frozen_held_reducer_hash(contract)
    artifact.update(
        {
            "preconditions_checked": deepcopy(dict(preconditions)),
            "source_artifact_hashes": deepcopy(dict(source_artifact_hashes or {})),
            "calibration_access_log": [dict(row) for row in access_log],
            "gate_check_summary": deepcopy(dict(preconditions)),
            "held_acceptance_contract": {**contract, "frozen_held_reducer_hash": reducer_hash},
            "frozen_held_reducer_hash": reducer_hash,
        }
    )
    return _finish_artifact(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return every schema, access, row, and verdict error."""

    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("held_label_access_count") != 0:
        errors.append("held_label_access_count")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    rows = artifact.get("rows")
    rows = rows if isinstance(rows, list) else []
    if any(
        not isinstance(row, Mapping) or row.get("row_hash") != _effect_row_hash(row) for row in rows
    ):
        errors.append("rows")
    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary")
        if not isinstance(summary, Mapping) or not summary.get("failed_check"):
            errors.append("blocked_gate_check_summary")
    return errors


def _read_json_object(path: Path) -> tuple[JsonDict, bytes, str | None]:
    """Read one JSON object and keep a typed error for blocked evidence."""

    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("top-level JSON object required")
        return dict(value), raw, None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, b"", f"{type(exc).__name__}:{exc}"


def _source_hash_rows(root: Path) -> JsonDict:
    """Hash code and evidence files that exist at execution time."""

    paths = {
        "exp6867": EXP6867_PATH,
        "exp6868": EXP6868_PATH,
        "calibration_sidecar": CALIBRATION_SIDECAR_PATH,
        "held_sidecar": HELD_SIDECAR_PATH,
        "module": MODULE_PATH,
        "wrapper": WRAPPER_PATH,
        "test": TEST_PATH,
        "spec": SPEC_PATH,
    }
    expected = {
        "exp6867": EXPECTED_EXP6867_SHA256,
        "exp6868": EXPECTED_EXP6868_SHA256,
        "calibration_sidecar": EXPECTED_CALIBRATION_SIDECAR_SHA256,
        "held_sidecar": EXPECTED_HELD_SIDECAR_SHA256,
    }
    output: JsonDict = {}
    for key, relative in paths.items():
        path = root / relative
        current = sha256_file(path) if path.is_file() else None
        output[key] = {
            "path": relative.as_posix(),
            "sha256": current,
            "expected_sha256": expected.get(key),
            "hash_match": current == expected[key] if key in expected else current is not None,
        }
    return output


def _calibration_labels(manifest: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    """Build candidate-label maps from the calibration manifest only."""

    output: dict[str, dict[str, bool]] = {}
    for row in manifest:
        candidates = list(row.get("candidate_ids") or [])
        labels = list(row.get("candidate_labels_by_slot") or [])
        if len(candidates) == len(labels) == 2:
            output[str(row.get("semantic_group_identity"))] = {
                str(candidate): bool(label)
                for candidate, label in zip(candidates, labels, strict=True)
            }
    return output


def _summaries(
    effect_cells: Sequence[Mapping[str, Any]],
    reduced_rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict, list[JsonDict], list[JsonDict]]:
    """Build model, family, pooled, nuisance, and bootstrap receipts."""

    per_model: list[JsonDict] = []
    per_family: list[JsonDict] = []
    nuisance: list[JsonDict] = []
    bootstrap_rows: list[JsonDict] = []
    for model in MODEL_SPECS:
        selected = [row for row in effect_cells if row.get("model_hf_id") == model]
        interval = stratified_bca_interval(
            selected,
            value_field="semantic_effect",
            resamples=resamples,
        )
        summary = {
            "model_hf_id": model,
            "model_family": MODEL_FAMILIES[model],
            "n_groups": len(selected),
            **interval,
        }
        per_model.append(summary)
        bootstrap_rows.append({"scope": "model", **summary})
        for family in SEMANTIC_FAMILIES:
            family_selected = [row for row in selected if row.get("semantic_family") == family]
            family_interval = stratified_bca_interval(
                family_selected,
                value_field="semantic_effect",
                resamples=resamples,
            )
            family_summary = {
                "model_hf_id": model,
                "model_family": MODEL_FAMILIES[model],
                "semantic_family": family,
                "n_groups": len(family_selected),
                **family_interval,
            }
            per_family.append(family_summary)
            bootstrap_rows.append({"scope": "model_semantic_family", **family_summary})
        for control in NUISANCE_CONTROLS:
            control_rows = [
                row
                for row in reduced_rows
                if row.get("model_hf_id") == model and row.get("nuisance_control") == control
            ]
            interval = stratified_bca_interval(
                control_rows,
                value_field="matched_nuisance_effect",
                resamples=resamples,
            )
            nuisance_summary = {
                "model_hf_id": model,
                "model_family": MODEL_FAMILIES[model],
                "nuisance_control": control,
                "absolute_upper_bound": max(
                    abs(float(interval["lower_bound"])),
                    abs(float(interval["upper_bound"])),
                    abs(float(interval["estimate"])),
                ),
                **interval,
            }
            nuisance.append(nuisance_summary)
            bootstrap_rows.append({"scope": "model_nuisance_control", **nuisance_summary})
    pooled = {
        "weighting": "equal_model_then_family_then_group",
        **stratified_bca_interval(
            effect_cells,
            value_field="semantic_effect",
            resamples=resamples,
        ),
    }
    bootstrap_rows.append({"scope": "pooled", **pooled})
    return per_model, per_family, pooled, nuisance, bootstrap_rows


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    resamples: int = 10000,
) -> JsonDict:
    """Read allowed inputs, reduce calibration rows, and freeze the held rule."""

    started = time.monotonic()
    access_log: list[JsonDict] = []
    exp6867, exp6867_raw, exp6867_error = _read_json_object(root / EXP6867_PATH)
    access_log.append(
        {
            "path": EXP6867_PATH.as_posix(),
            "access_scope": "preregistration_metadata_and_calibration_labels",
            "fields": [
                "calibration_group_manifest",
                "sealed_held_group_manifest.commitments_only",
                "split_manifest.calibration_group_ids",
                "split_manifest.held_group_ids",
                "rows[*].split_for_routing",
                "rows[split=calibration].nuisance_match_fields",
            ],
            "outcome": exp6867_error or "loaded",
        }
    )
    exp6868, exp6868_raw, exp6868_error = _read_json_object(root / EXP6868_PATH)
    access_log.append(
        {
            "path": EXP6868_PATH.as_posix(),
            "access_scope": "stream_metadata_only",
            "fields": [
                "semantic_contrast_stream_v2_complete_score",
                "calibration_score_manifest",
                "sealed_held_score_manifest",
                "failed_cell_manifest[split=calibration]",
            ],
            "outcome": exp6868_error or "loaded",
        }
    )
    calibration, calibration_raw, calibration_error = _read_json_object(
        root / CALIBRATION_SIDECAR_PATH
    )
    access_log.append(
        {
            "path": CALIBRATION_SIDECAR_PATH.as_posix(),
            "access_scope": "calibration_raw_token_rows",
            "fields": ["schema", "split", "row_count", "rows", "sha256", "effect_reduced"],
            "outcome": calibration_error or "loaded",
        }
    )
    held_path = root / HELD_SIDECAR_PATH
    held_sha = sha256_file(held_path) if held_path.is_file() else ""
    access_log.append(
        {
            "path": HELD_SIDECAR_PATH.as_posix(),
            "access_scope": "file_bytes_sha256_only",
            "fields": [],
            "json_parsed": False,
            "outcome": "hashed_without_parse" if held_sha else "unreadable",
        }
    )
    source_hashes = _source_hash_rows(root)
    calibration_rows_value = calibration.get("rows")
    calibration_rows = calibration_rows_value if isinstance(calibration_rows_value, list) else []
    calibration_payload_sha = sha256_json(calibration_rows) if calibration_rows else ""
    preconditions = evaluate_preconditions(
        exp6867=exp6867,
        exp6868=exp6868,
        exp6867_sha256=sha256_bytes(exp6867_raw) if exp6867_raw else "",
        exp6868_sha256=sha256_bytes(exp6868_raw) if exp6868_raw else "",
        calibration_sidecar_sha256=sha256_bytes(calibration_raw) if calibration_raw else "",
        calibration_payload_sha256=calibration_payload_sha,
        held_sidecar_sha256=held_sha,
    )
    read_errors = [value for value in (exp6867_error, exp6868_error, calibration_error) if value]
    if read_errors:
        preconditions = gate_summary(
            [gate_check("required_files_readable", [], read_errors), *preconditions["checks"]]
        )
    if preconditions["passed"] is not True:
        return build_blocked_artifact(
            run_date=run_date,
            duration_s=time.monotonic() - started,
            preconditions=preconditions,
            access_log=access_log,
            source_artifact_hashes=source_hashes,
        )

    calibration_manifest = [
        row for row in exp6867["calibration_group_manifest"] if isinstance(row, Mapping)
    ]
    held_ids = {
        str(row.get("semantic_group_identity"))
        for row in exp6867["sealed_held_group_manifest"]
        if isinstance(row, Mapping)
    }
    store = CalibrationLabelStore(
        _calibration_labels(calibration_manifest),
        held_ids,
        source_path=EXP6867_PATH.as_posix(),
        access_log=access_log,
    )
    control_cells = {
        (str(row.get("model_hf_id")), str(row.get("semantic_group_identity"))): row
        for row in exp6867.get("rows", [])
        if isinstance(row, Mapping)
        and row.get("split") == "calibration"
        and row.get("accepted") is True
    }
    score_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        if isinstance(row, Mapping):
            score_groups[
                (str(row.get("model_hf_id")), str(row.get("semantic_group_identity")))
            ].append(row)

    reduced_rows: list[JsonDict] = []
    missing_rows: list[JsonDict] = []
    for key, control_cell in sorted(control_cells.items()):
        labels = store.load(key[1])
        rows, missing = reduce_group_rows(
            score_groups.get(key, []),
            labels=labels,
            control_cell=control_cell,
        )
        reduced_rows.extend(rows)
        missing_rows.extend(missing)
    unexpected = sorted(set(score_groups) - set(control_cells))
    missing_rows.extend(
        {
            "model_hf_id": model,
            "semantic_group_identity": group,
            "reason": "unexpected_calibration_score_cell",
            "detail": "score has no accepted Exp6867 calibration cell",
            "imputed": False,
        }
        for model, group in unexpected
    )
    failed_cells = [
        row
        for row in exp6868.get("failed_cell_manifest", [])
        if isinstance(row, Mapping) and row.get("split") == "calibration"
    ]
    missing_rows.extend(
        {
            **dict(row),
            "reason": str(row.get("reason") or "explicit_calibration_score_failure"),
            "imputed": False,
        }
        for row in failed_cells
    )

    effect_cells = [row for row in reduced_rows if row["nuisance_control"] == "identifier"]
    per_model, per_family, pooled, nuisance, bootstrap_rows = _summaries(
        effect_cells,
        reduced_rows,
        resamples=int(resamples),
    )
    replication = family_replication_result(per_family)
    sign_reversal_count = sum(bool(row.get("sign_reversal")) for row in effect_cells)
    decision = rule_readiness(
        per_model_effects=per_model,
        per_family_effects=per_family,
        nuisance_effects=nuisance,
        missing_cell_rows=missing_rows,
        sign_reversal_count=sign_reversal_count,
        preconditions_passed=True,
        dry_run_without_held_access=store.held_label_access_count == 0,
    )
    contract = freeze_acceptance_contract(nuisance)
    reducer_hash = frozen_held_reducer_hash(contract)
    contract["frozen_held_reducer_hash"] = reducer_hash
    readiness = int(decision["semantic_contrast_rule_ready_score"])
    gate_checks = [*preconditions["checks"], *decision["checks"]]
    artifact = _base_artifact(run_date=run_date, duration_s=time.monotonic() - started)
    artifact.update(
        {
            "status": "complete",
            "source_artifact_hashes": source_hashes,
            "preconditions_checked": preconditions,
            "calibration_access_log": access_log,
            "held_label_access_count": store.held_label_access_count,
            "rows": reduced_rows,
            "per_model_calibration_effects": per_model,
            "per_family_calibration_effects": per_family,
            "pooled_calibration_effect": pooled,
            "nuisance_control_effects": nuisance,
            "missing_cell_rows": missing_rows,
            "bootstrap_rows": bootstrap_rows,
            "family_replication_result": replication,
            "frozen_held_reducer_hash": reducer_hash,
            "held_acceptance_contract": contract,
            "semantic_contrast_rule_ready_score": readiness,
            "gate_check_summary": gate_summary(gate_checks),
            "verdict_class": "positive" if readiness else "null",
            "honest_verdict": READY_VERDICT if readiness else NULL_VERDICT,
            "duration_s": round(time.monotonic() - started, 6),
        }
    )
    return _finish_artifact(artifact)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace only the requested artifact after a complete JSON write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by E2E command.
    """Run Exp6869 and write its terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    args = parser.parse_args(argv)
    artifact = build_artifact(run_date=str(args.date))
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"invalid Exp6869 artifact: {errors}")
    write_json_atomic(args.output, artifact)
    print(
        canonical_json(
            {
                "output": str(args.output),
                "ready": artifact["semantic_contrast_rule_ready_score"],
                "held_label_access_count": artifact["held_label_access_count"],
                "honest_verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
