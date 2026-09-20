"""Seal a raw-logit option interface and bounded source-support protocol.

The task qualifies data flow and score extraction with scripted numbers. It
does not load a language model, fit a selector, or measure predictive value.

Spec refs: REQ-VERIFY-7462 and SCENARIO-VERIFY-7462-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7423_v651_annotated_protocol import (
    DEFAULT_CACHE_ROOT as RAGTRUTH_CACHE_ROOT,
    authenticate_assets as authenticate_ragtruth_assets,
    load_release as load_ragtruth_release,
)
from carnot.experiment_7449_v653_source_protocol import (
    RAGTRUTH_CAPS,
    select_ragtruth_panel,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
Tokenizer = Callable[..., Sequence[int]]

RUN_DATE = "20260920"
MILESTONE = "2026.09.654"
EXPERIMENT_ID = "exp7462-v654-option-protocol"
SCHEMA = "carnot.exp7462.v654.option_protocol.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7462_v654_option_protocol.json")
RAW_DIR = Path("results/raw/experiment_7462_v654_option_protocol")
MODULE_PATH = Path("python/carnot/experiment_7462_v654_option_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7462_v654_option_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7462_v654_option_protocol.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verification/spec.md"
V653_RESULT_PATH = Path("results/experiment_7449_v653_source_protocol.json")
V653_RAW_DIR = Path("results/raw/experiment_7449_v653_source_protocol")
V653_EMBEDDING_PATH = Path("results/experiment_7452_v653_source_embeddings.json")
JEVBENCH_RESULT_PATH = Path("results/jevbench_readout_eval_2026_09_20.json")

MODEL_SPECS: list[JsonDict] = []
MODEL_SPECS_LOWER: list[JsonDict] = []
INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
INFERENCE_SUBSTRATE = "scripted_raw_logit_interface_and_protocol_sealing"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
OPTION_IDS = ("supported", "contains_unsupported")
OPTION_TEXT = {
    "supported": "The full response is supported by the supplied source.",
    "contains_unsupported": "The response contains unsupported content.",
}
DISPLAY_LABELS = (" A", " B")
FIT_SEEDS = (65_401, 65_402, 65_403, 65_404, 65_405)
ORDERING_SEED = 65_400
AUDIT_SEED = 65_409
BOOTSTRAP_SEED = 6_540_010
ONLINE_CAP = 160
EXTERNAL_CAP = 100
TOKEN_CEILING = 2_048
MINIMUM_GROUPS = {
    "training": 150,
    "calibration_tuning": 40,
    "internal_test": 40,
    "online": 120,
    "external": 60,
}
ROLE_CAPS = {
    "training": 180,
    "calibration_tuning": 60,
    "internal_test": 60,
    "online": ONLINE_CAP,
    "external": EXTERNAL_CAP,
}
ROLE_ORDER = {
    "training": 0,
    "calibration_tuning": 1,
    "internal_test": 2,
    "online": 3,
    "external": 4,
}
RELEASES = {
    "ragtruth": {
        "license": "MIT",
        "revision": "c103204b9ce28d6bbad859304bf30de72b8ed8fe",
        "annotation_provenance": "RAGTruth human unsupported-span annotations",
    },
    "faithbench": {
        "license": "CC BY-NC-SA 4.0",
        "revision": "cf89797d82812c23b5d5e5c121f1d9b8983bbbce",
        "annotation_provenance": "FaithBench public human annotation release",
    },
}
SMALL_EBM_TRAINING = {
    "attempted": False,
    "fit_attempts": 0,
    "fit_completions": 0,
    "fit_failures": 0,
    "duration_s": 0.0,
    "receipt_class": "small_ebm_training",
    "deferred_to": "post_capture_numeric_fit",
}

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("openspec/capabilities/verification/spec.md"),
    Path("scripts/jevbench_readout_eval.py"),
    Path("python/carnot/experiment_7449_v653_source_protocol.py"),
    Path("python/carnot/experiment_7423_v651_annotated_protocol.py"),
    Path("python/carnot/autoresearch/calibrated_decision_benchmark.py"),
    Path("python/carnot/models/compositional_energy.py"),
    Path("docs/research-notes/jevbench-semif-decision-readout-2026-09-20.md"),
    V653_RESULT_PATH,
    V653_RAW_DIR / "corpus_manifest.json",
    V653_RAW_DIR / "predictors.jsonl",
    V653_RAW_DIR / "evaluators.jsonl",
    JEVBENCH_RESULT_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "ended_at_utc",
    "started_monotonic_ns",
    "ended_monotonic_ns",
    "clock_identity",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "device_identity",
    "duration_s",
    "duration_components_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "honest_verdict",
    "verdict_class",
    "verifier_is_oracle",
    "flagged_adversarial",
    "validation_receipts",
    "repository_health",
    "field_principles",
    "option_protocol_ready_score",
    "cohort_manifest",
    "comparison_plan",
    "option_readout_contract",
    "small_ebm_training",
    "current_invocation_events",
    "historical_inference_sidecars",
    "confirmatory_minima_met",
    "independent_reduction",
    "promotion_score",
    "numbered_e2e_applicable",
    "capability_e2e",
    "production_defaults_changed",
    "external_publication_authorized",
)

AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
FULL_SUITE_OBSERVATION_NAME = "all_python_tests_required_once"
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


class OptionProtocolError(ValueError):
    """Reject an option interface or cohort that can silently change meaning."""


def _stable_hash(value: str) -> str:
    """Return a stable hash for outcome-blind ordering and identity checks."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_source_hash(value: str) -> str:
    """Hash normalized source text so cosmetic spacing cannot cross roles."""

    return _stable_hash(" ".join(value.casefold().split()))


def build_option_prompt(source: str, response: str, option_order: Sequence[str]) -> str:
    """Build the complete-input prompt with stable option IDs and no truncation."""

    order = tuple(option_order)
    if len(order) != len(OPTION_IDS) or set(order) != set(OPTION_IDS):
        raise OptionProtocolError("option_ids_invalid")
    lines = [
        f"{label.strip()}: {option_id} — {OPTION_TEXT[option_id]}"
        for label, option_id in zip(DISPLAY_LABELS, order, strict=True)
    ]
    return (
        "Classify support for the complete response using only the supplied source.\n\n"
        f"SOURCE\n{source}\n\nRESPONSE\n{response}\n\n"
        "OPTIONS\n" + "\n".join(lines) + "\n\nReturn only the option label.\nAnswer:"
    )


def read_option_logits(
    source: str,
    response: str,
    option_order: Sequence[str],
    *,
    tokenize: Tokenizer,
    score_rows: Sequence[Sequence[float]],
) -> JsonDict:
    """Read raw option logits at the final evaluated prompt token.

    The score buffer can be larger than the evaluated prompt. Using its final
    row would read unwritten memory, so the contract records the exact row.
    """

    order = tuple(option_order)
    prompt = build_option_prompt(source, response, order)
    prompt_tokens = list(tokenize(prompt, add_bos=True))
    if not prompt_tokens:  # pragma: no cover - defensive tokenizer contract.
        raise OptionProtocolError("prompt_tokens_missing")
    label_tokens: list[int] = []
    for label in DISPLAY_LABELS:
        token_ids = list(tokenize(label, add_bos=False))
        if len(token_ids) != 1:
            raise OptionProtocolError(f"label_not_single_token:{label.strip()}")
        label_tokens.append(int(token_ids[0]))
        combined = list(tokenize(prompt + label, add_bos=True))
        if combined != [*prompt_tokens, int(token_ids[0])]:
            raise OptionProtocolError(f"prompt_label_boundary_invalid:{label.strip()}")
    if len(set(label_tokens)) != len(label_tokens):
        raise OptionProtocolError("duplicate_label_token")
    position = len(prompt_tokens) - 1
    if len(score_rows) <= position:
        raise OptionProtocolError("score_buffer_short")
    selected_row = score_rows[position]
    try:
        selected = [float(selected_row[token_id]) for token_id in label_tokens]
    except (IndexError, TypeError) as exc:  # pragma: no cover - defensive backend shape.
        raise OptionProtocolError("label_logit_missing") from exc
    if any(not math.isfinite(value) for value in selected):
        raise OptionProtocolError("nonfinite_logit")
    if max(selected) - min(selected) <= 1e-12:
        raise OptionProtocolError("uniform_logits")
    maximum = max(selected)
    weights = [math.exp(value - maximum) for value in selected]
    total = sum(weights)
    raw = dict(zip(order, selected, strict=True))
    probabilities = {
        option_id: weight / total for option_id, weight in zip(order, weights, strict=True)
    }
    return {
        "option_ids_in_prompt": list(order),
        "display_labels": list(DISPLAY_LABELS),
        "label_token_ids": label_tokens,
        "label_to_option_id": dict(zip(DISPLAY_LABELS, order, strict=True)),
        "prompt_sha256": _stable_hash(prompt),
        "prompt_token_count": len(prompt_tokens),
        "last_evaluated_prompt_position": position,
        "score_buffer_rows": len(score_rows),
        "raw_logits_by_option_id": raw,
        "probabilities_by_option_id": probabilities,
        "normalization": "stable_softmax_after_raw_logit_capture",
    }


def _validate_probabilities(value: object) -> dict[str, float]:
    """Return a stable-ID distribution or reject altered option membership."""

    if not isinstance(value, Mapping) or set(value) != set(OPTION_IDS):
        raise OptionProtocolError("probability_ids_invalid")
    probabilities = {option_id: float(value[option_id]) for option_id in OPTION_IDS}
    if any(not math.isfinite(item) or item < 0.0 for item in probabilities.values()):
        raise OptionProtocolError("probability_value_invalid")  # pragma: no cover
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise OptionProtocolError("probability_sum_invalid")  # pragma: no cover
    return probabilities


def average_order_readouts(
    original: Mapping[str, Any], reversed_order: Mapping[str, Any]
) -> JsonDict:
    """Average both distributions only after stable option-ID remapping."""

    if original.get("option_ids_in_prompt") != list(OPTION_IDS) or reversed_order.get(
        "option_ids_in_prompt"
    ) != list(reversed(OPTION_IDS)):
        raise OptionProtocolError("option_order_pair_invalid")
    left = _validate_probabilities(original.get("probabilities_by_option_id"))
    right = _validate_probabilities(reversed_order.get("probabilities_by_option_id"))
    return {option_id: (left[option_id] + right[option_id]) / 2.0 for option_id in OPTION_IDS}


def annotation_disposition(evaluator: Mapping[str, Any]) -> str:
    """Map human evidence to the prespecified whole-response support target."""

    labels = evaluator.get("annotation_labels") or []
    if evaluator.get("ambiguous") is True or "Questionable" in labels:
        return "uncertain_or_contested"
    label = evaluator.get("label")
    if label == 1:
        return "supported"
    if label == 0:
        return "contains_unsupported"
    raise OptionProtocolError("support_label_invalid")


def _joined_rows(
    predictors: Sequence[Mapping[str, Any]], evaluators: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Join access-separated rows by identity and reject missing or duplicate keys."""

    predictor_by_key = {str(row.get("row_key")): row for row in predictors}
    evaluator_by_key = {str(row.get("row_key")): row for row in evaluators}
    if (
        len(predictor_by_key) != len(predictors)
        or len(evaluator_by_key) != len(evaluators)
        or set(predictor_by_key) != set(evaluator_by_key)
    ):
        raise OptionProtocolError("cohort_row_identity_invalid")  # pragma: no cover
    return [(row, evaluator_by_key[str(row["row_key"])]) for row in predictors]


def freeze_cohort(
    v653_predictors: Sequence[Mapping[str, Any]],
    v653_evaluators: Sequence[Mapping[str, Any]],
    online_predictors: Sequence[Mapping[str, Any]],
    online_evaluators: Sequence[Mapping[str, Any]],
    *,
    online_cap: int = ONLINE_CAP,
) -> JsonDict:
    """Preserve V653 roles and add disjoint online groups by stable hash."""

    selected: list[tuple[JsonDict, JsonDict]] = []
    groups: list[JsonDict] = []
    exclusions: list[JsonDict] = []
    used_groups: set[str] = set()
    used_sources: set[str] = set()

    def add_pair(
        predictor_value: Mapping[str, Any], evaluator_value: Mapping[str, Any], role: str
    ) -> None:
        predictor = deepcopy(dict(predictor_value))
        evaluator = deepcopy(dict(evaluator_value))
        group_id = str(predictor.get("group_id"))
        corpus = str(predictor.get("corpus"))
        source = str(predictor.get("source_text"))
        response = str(predictor.get("response_text"))
        if evaluator.get("group_id") != group_id or evaluator.get("corpus") != corpus:
            raise OptionProtocolError("cohort_join_metadata_invalid")  # pragma: no cover
        if role not in ROLE_CAPS or corpus not in RELEASES:
            raise OptionProtocolError("cohort_role_or_release_invalid")  # pragma: no cover
        source_hash = _normalized_source_hash(source)
        response_hash = _stable_hash(response)
        if group_id in used_groups or source_hash in used_sources:
            raise OptionProtocolError("cohort_cross_role_duplicate")
        used_groups.add(group_id)
        used_sources.add(source_hash)
        release = RELEASES[corpus]
        projected_predictor = {
            "row_key": predictor["row_key"],
            "group_id": group_id,
            "corpus": corpus,
            "role": role,
            "source_text": source,
            "response_text": response,
            "source_hash": source_hash,
            "response_hash": response_hash,
            "license": release["license"],
            "release_revision": release["revision"],
        }
        projected_evaluator = {
            "row_key": evaluator["row_key"],
            "group_id": group_id,
            "corpus": corpus,
            "role": role,
            "source_id": evaluator.get("source_id"),
            "response_id": evaluator.get("response_id"),
            "official_split": evaluator.get("official_split"),
            "label": evaluator.get("label"),
            "annotation_labels": deepcopy(evaluator.get("annotation_labels") or []),
            "ambiguous": bool(evaluator.get("ambiguous")),
            "label_policy": evaluator.get("label_policy"),
            "annotation_disposition": annotation_disposition(evaluator),
            "annotation_provenance": release["annotation_provenance"],
        }
        group_hash = canonical_hash(
            {
                "corpus": corpus,
                "group_id": group_id,
                "role": role,
                "source_hash": source_hash,
                "response_hash": response_hash,
            }
        )
        selected.append((projected_predictor, projected_evaluator))
        groups.append(
            {
                "group_id": group_id,
                "group_hash": group_hash,
                "source_hash": source_hash,
                "response_hash": response_hash,
                "role": role,
                "corpus": corpus,
                "license": release["license"],
                "release_revision": release["revision"],
                "annotation_provenance": release["annotation_provenance"],
                "annotation_disposition": projected_evaluator["annotation_disposition"],
            }
        )

    old_pairs = _joined_rows(v653_predictors, v653_evaluators)
    for predictor, evaluator in old_pairs:
        role = str(predictor.get("role"))
        if evaluator.get("role") != role:
            raise OptionProtocolError("cohort_role_mismatch")  # pragma: no cover
        add_pair(predictor, evaluator, role)

    candidates: list[tuple[Mapping[str, Any], Mapping[str, Any], str]] = []
    for predictor, evaluator in _joined_rows(online_predictors, online_evaluators):
        group_id = str(predictor.get("group_id"))
        source_hash = _normalized_source_hash(str(predictor.get("source_text")))
        if group_id in used_groups or source_hash in used_sources:
            exclusions.append(
                {
                    "group_id": group_id,
                    "source_hash": source_hash,
                    "reason": "excluded_previous_role_source_hash",
                }
            )
            continue
        order_key = _stable_hash(f"v654-online:{group_id}:{source_hash}")
        candidates.append((predictor, evaluator, order_key))
    for predictor, evaluator, _order_key in sorted(candidates, key=lambda row: row[2])[
        : max(0, online_cap)
    ]:
        add_pair(predictor, evaluator, "online")

    selected.sort(key=lambda pair: (ROLE_ORDER[str(pair[0]["role"])], str(pair[0]["group_id"])))
    groups.sort(key=lambda row: (ROLE_ORDER[str(row["role"])], str(row["group_id"])))
    counts = Counter(str(row["role"]) for row in groups)
    return {
        "predictors": [row[0] for row in selected],
        "evaluators": [row[1] for row in selected],
        "groups": groups,
        "exclusions": sorted(exclusions, key=lambda row: str(row["group_id"])),
        "counts": {role: counts.get(role, 0) for role in ROLE_ORDER},
        "selection_without_outcomes": True,
        "group_and_source_role_disjoint": len(groups) == len(used_groups) == len(used_sources),
    }


def comparison_plan() -> JsonDict:
    """Freeze controls, metrics, costs, stopping rules, and multiplicity."""

    return {
        "support_unit": "complete_response",
        "target": "supported_vs_contains_any_human_annotated_unsupported_content",
        "complete_claim_extraction_claimed": False,
        "option_orders": [list(OPTION_IDS), list(reversed(OPTION_IDS))],
        "primary_readout": "mean_of_two_distributions_after_stable_id_remap",
        "forward_charge_per_full_source_group": 2,
        "complete_prompt_token_ceiling": TOKEN_CEILING,
        "truncate_to_fit": False,
        "overlength_disposition": "explicit_exclusion_without_replacement_after_scoring",
        "fit_seeds": list(FIT_SEEDS),
        "bootstrap": {"draws": 10_000, "unit": "source_group"},
        "familywise_alpha": 0.05,
        "multiplicity": "familywise_prespecified",
        "primary_outcome": "paired_multiclass_brier",
        "secondary_outcomes": [
            "log_loss",
            "ece",
            "coverage",
            "false_acceptance",
            "service_time",
        ],
        "calibration_controls": {
            "reserved_groups": 40,
            "arms": ["response_only", "shuffled_source"],
            "selection": "first stable-hash calibration groups before outcomes",
        },
        "decision_costs": {
            "false_accept": 10.0,
            "false_reject": 1.0,
            "escalate": 0.2,
            "false_accept_sensitivity": [5.0, 20.0],
            "operator_approved_deployment_policy": False,
        },
        "threshold_fit_role": "calibration_tuning_only",
        "minimum_usable_groups": deepcopy(MINIMUM_GROUPS),
        "role_shortfall_policy": "blocks_confirmatory_benefit_without_role_mixing",
        "stopping_rule": "freeze denominators before capture; do not replace failures or exclusions",
    }


def protocol_gates(
    counts: Mapping[str, int],
    *,
    interface_passed: bool,
    cohort_disjoint: bool,
    predictor_isolated: bool,
    validation_passed: bool,
) -> list[JsonDict]:
    """Keep protocol validity separate from later confirmatory benefit."""

    def gate(
        check: str,
        category: str,
        expected: Any,
        observed: Any,
        operator: str,
        passed: bool,
        principle: str,
    ) -> JsonDict:
        return {
            "check": check,
            "category": category,
            "expected": expected,
            "observed": observed,
            "op": operator,
            "passed": passed,
            "principle": principle,
        }

    rows = [
        gate(
            "raw_logit_interface",
            "protocol_validity",
            True,
            interface_passed,
            "==",
            interface_passed,
            "Raw logits must survive token, position, order, and finite-value controls.",
        ),
        gate(
            "cohort_role_disjointness",
            "protocol_validity",
            True,
            cohort_disjoint,
            "==",
            cohort_disjoint,
            "One source group in two roles would leak fitting evidence into evaluation.",
        ),
        gate(
            "predictor_information_isolation",
            "protocol_validity",
            True,
            predictor_isolated,
            "==",
            predictor_isolated,
            "Predictors must not receive human labels, notes, detector outputs, or model identity.",
        ),
        gate(
            "affected_and_terminal_validation",
            "required_validation",
            True,
            validation_passed,
            "==",
            validation_passed,
            "Every declared current check must pass before protocol readiness is published.",
        ),
    ]
    rows.extend(
        gate(
            f"minimum_groups:{role}",
            "confirmatory_benefit",
            minimum,
            int(counts.get(role, 0)),
            ">=",
            int(counts.get(role, 0)) >= minimum,
            "A role shortfall blocks a confirmatory claim and never permits role mixing.",
        )
        for role, minimum in MINIMUM_GROUPS.items()
    )
    return rows


def reduce_protocol(gates: Sequence[Mapping[str, Any]], *, flagged_adversarial: bool) -> JsonDict:
    """Reduce interface readiness without turning protocol work into efficacy."""

    required = [row for row in gates if row.get("category") != "confirmatory_benefit"]
    minima = [row for row in gates if row.get("category") == "confirmatory_benefit"]
    ready = bool(required) and all(row.get("passed") is True for row in required)
    confirmatory = bool(minima) and all(row.get("passed") is True for row in minima)
    if not ready or flagged_adversarial:
        return {
            "option_protocol_ready_score": 0,
            "confirmatory_minima_met": confirmatory,
            "status": "disqualified",
            "honest_verdict": "complete_disqualified_option_protocol_validation",
            "verdict_class": "disqualified",
            "promotion_score": 0,
        }
    return {
        "option_protocol_ready_score": 1,
        "confirmatory_minima_met": confirmatory,
        "status": "complete",
        "honest_verdict": "complete_null_option_protocol_ready_no_efficacy_measurement",
        "verdict_class": "null",
        "promotion_score": 0,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind all terminal evidence except the checksum's own value."""

    stable = {
        key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"
    }
    return canonical_hash(stable)


def validate_artifact_shape(value: Mapping[str, Any]) -> list[str]:
    """Check required identity and no-model fields without reading raw shards."""

    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if errors:
        return [f"missing_field:{field}" for field in errors]
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
        or value.get("run_date") != RUN_DATE
    ):
        errors.append("identity_invalid")  # pragma: no cover - terminal reader branch.
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != INVOCATION_COUNTS
        or value.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
        or value.get("execution_venue") != EXECUTION_VENUE
    ):
        errors.append("current_model_contract_invalid")
    if value.get("option_protocol_ready_score") not in {0, 1}:
        errors.append("option_protocol_ready_score_invalid")  # pragma: no cover
    if value.get("promotion_score") != 0:
        errors.append("promotion_nonzero")  # pragma: no cover
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_invalid")  # pragma: no cover
    return list(dict.fromkeys(errors))


def utc_now() -> str:  # pragma: no cover - real process boundary.
    """Return an aware UTC timestamp for one measured boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print one flushed phase or slow-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7462] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:  # pragma: no cover - bounded file input.
    """Read one JSON object or return an empty object for invalid bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:  # pragma: no cover - bounded file input.
    """Read one authenticated JSONL shard as object rows."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise OptionProtocolError(f"jsonl_object_required:{path}")
    return rows


def _precondition(
    check: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:  # pragma: no cover - artifact construction.
    """Record one exact prerequisite comparison before protocol work."""

    return {
        "check": check,
        "upstream": path,
        "path": path,
        "field": field,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": observed == expected,
    }


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], dict[str, JsonDict]]:  # pragma: no cover
    """Authenticate source bytes, ownership, device IDs, and historical flags."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
            )
        )
        if available:
            stat = path.stat()
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "owner_uid": stat.st_uid,
                "owner_gid": stat.st_gid,
                "device_id": stat.st_dev,
                "original_flagged_adversarial": None,
            }

    spec_text = SPEC_PATH.read_text(encoding="utf-8") if SPEC_PATH.is_file() else ""
    checks.append(
        _precondition(
            "driving_requirement",
            "openspec/capabilities/verification/spec.md",
            "REQ-*",
            "REQ-VERIFY-7462",
            "REQ-VERIFY-7462" if "REQ-VERIFY-7462" in spec_text else None,
        )
    )
    v653 = _load_object(root / V653_RESULT_PATH)
    for field, expected in {
        "source_protocol_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }.items():
        checks.append(
            _precondition(
                f"v653_{field}", V653_RESULT_PATH.as_posix(), field, expected, v653.get(field)
            )
        )
    hashes[V653_RESULT_PATH.as_posix()]["original_flagged_adversarial"] = v653.get(
        "flagged_adversarial"
    )

    embedding = _load_object(root / V653_EMBEDDING_PATH)
    if embedding:
        forward_counts = embedding.get("invocation_counts", {}).get("forward_calls", {})
        observed_embedding = {
            "embedding_capture_ready_score": embedding.get("embedding_capture_ready_score"),
            "forward_calls_attempted": forward_counts.get("attempted"),
            "honest_verdict": embedding.get("honest_verdict"),
            "flagged_adversarial": embedding.get("flagged_adversarial"),
        }
        expected_embedding = {
            "embedding_capture_ready_score": 0,
            "forward_calls_attempted": 0,
            "honest_verdict": "complete_null_blocked_embedding_surface_unavailable",
            "flagged_adversarial": False,
        }
        hashes[V653_EMBEDDING_PATH.as_posix()] = {
            "path": V653_EMBEDDING_PATH.as_posix(),
            "sha256": sha256_file(root / V653_EMBEDDING_PATH),
            "original_flagged_adversarial": embedding.get("flagged_adversarial"),
        }
    else:
        observed_embedding = "artifact_absent_no_forward_capture"
        expected_embedding = "artifact_absent_no_forward_capture"
    checks.append(
        _precondition(
            "v653_source_embeddings_preserved",
            V653_EMBEDDING_PATH.as_posix(),
            "historical_capture_state",
            expected_embedding,
            observed_embedding,
        )
    )

    jevbench = _load_object(root / JEVBENCH_RESULT_PATH)
    checks.append(
        _precondition(
            "jevbench_aggregate_has_no_per_question_training_rows",
            JEVBENCH_RESULT_PATH.as_posix(),
            "rows",
            "absent",
            "absent" if "rows" not in jevbench else "present",
        )
    )
    hashes[JEVBENCH_RESULT_PATH.as_posix()]["original_flagged_adversarial"] = jevbench.get(
        "flagged_adversarial"
    )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            EXPERIMENT_ID in exclusion_text,
        )
    )
    return checks, hashes


def collect_device_identity() -> JsonDict:  # pragma: no cover - host observation.
    """Separate current host devices from historical board evidence."""

    cpu = platform.processor() or platform.machine()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        cuda = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        cuda = []
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu": cpu,
        "cuda_devices_observed": cuda,
        "cuda_used": False,
        "historical_board_evidence": {
            "path": "ops/e2e-test-plan.md",
            "used_for_current_result": False,
            "reason": "pure option-protocol work has no hardware E2E",
        },
    }


def previous_repository_health(root: Path) -> JsonDict:  # pragma: no cover - diagnostic replay.
    """Preserve the one broad-suite observation outside required validation."""

    previous = _load_object(root / RESULT_PATH)
    stored = previous.get("repository_health")
    if isinstance(stored, Mapping) and stored.get("observations"):
        return deepcopy(dict(stored))
    receipts = previous.get("validation_receipts")
    observations = [
        deepcopy(dict(row))
        for row in receipts or []
        if isinstance(row, Mapping) and row.get("name") == FULL_SUITE_OBSERVATION_NAME
    ]
    for row in observations:
        row["required_for_exp7462"] = False
        row["classification"] = "unrelated_repository_health"
        row["interrupted_after_failures"] = row.get("exit_code") == 2
    return {
        "status": "degraded_unrelated_baseline" if observations else "not_observed",
        "observations": observations,
        "controls_option_protocol_readiness": False,
        "principle": "A broad repository failure cannot replace or invalidate passing affected checks.",
    }


def _scripted_tokenize(text: str, *, add_bos: bool) -> list[int]:  # pragma: no cover
    """Provide a deterministic tokenizer only for interface qualification."""

    labels = {" A": 11, " B": 17}
    prefix = [1] if add_bos else []
    if text in labels:
        return prefix + [labels[text]]
    for label, token_id in labels.items():
        if text.endswith(label):
            return _scripted_tokenize(text[: -len(label)], add_bos=add_bos) + [token_id]
    return prefix + [2 + (ord(character) % 7) for character in text]


def _scripted_scores(
    prompt_count: int, left: float, right: float
) -> list[list[float]]:  # pragma: no cover
    """Put signal at the evaluated row and leave the final buffer row uniform."""

    rows = [[0.0] * 24 for _ in range(prompt_count + 3)]
    rows[prompt_count - 1][11] = left
    rows[prompt_count - 1][17] = right
    return rows


def run_interface_controls() -> JsonDict:  # pragma: no cover - exercised by entrypoint E2E.
    """Run each known readout mutation against the reusable helper."""

    source, response = "Source evidence.", "Supplied response."

    def score(order: Sequence[str], left: float, right: float) -> JsonDict:
        prompt = build_option_prompt(source, response, order)
        count = len(_scripted_tokenize(prompt, add_bos=True))
        return read_option_logits(
            source,
            response,
            order,
            tokenize=_scripted_tokenize,
            score_rows=_scripted_scores(count, left, right),
        )

    original = score(OPTION_IDS, 4.0, 1.0)
    reversed_result = score(tuple(reversed(OPTION_IDS)), 1.0, 4.0)
    averaged = average_order_readouts(original, reversed_result)

    attacks: dict[str, bool] = {}

    def rejected(name: str, operation: Callable[[], object]) -> None:
        try:
            operation()
        except OptionProtocolError:
            attacks[name] = True
        else:
            attacks[name] = False

    prompt = build_option_prompt(source, response, OPTION_IDS)
    count = len(_scripted_tokenize(prompt, add_bos=True))
    rejected(
        "uniform_stub",
        lambda: read_option_logits(
            source,
            response,
            OPTION_IDS,
            tokenize=_scripted_tokenize,
            score_rows=_scripted_scores(count, 2.0, 2.0),
        ),
    )
    bad_nonfinite = _scripted_scores(count, 4.0, float("nan"))
    rejected(
        "nonfinite",
        lambda: read_option_logits(
            source,
            response,
            OPTION_IDS,
            tokenize=_scripted_tokenize,
            score_rows=bad_nonfinite,
        ),
    )

    def missing(text: str, *, add_bos: bool) -> list[int]:
        return [] if text == " B" else _scripted_tokenize(text, add_bos=add_bos)

    rejected(
        "missing_label",
        lambda: read_option_logits(source, response, OPTION_IDS, tokenize=missing, score_rows=[]),
    )

    def duplicate(text: str, *, add_bos: bool) -> list[int]:
        if text in DISPLAY_LABELS:
            return ([1] if add_bos else []) + [11]
        if text.endswith(" B"):
            return duplicate(text[:-2], add_bos=add_bos) + [11]
        return _scripted_tokenize(text, add_bos=add_bos)

    rejected(
        "duplicate_tokens",
        lambda: read_option_logits(source, response, OPTION_IDS, tokenize=duplicate, score_rows=[]),
    )
    swapped = deepcopy(reversed_result)
    swapped["option_ids_in_prompt"] = list(OPTION_IDS)
    rejected("option_id_swap", lambda: average_order_readouts(original, swapped))
    attacks["unused_buffer_scores_minus_one"] = (
        original["last_evaluated_prompt_position"] == original["prompt_token_count"] - 1
        and original["score_buffer_rows"] > original["prompt_token_count"]
        and len(set(original["raw_logits_by_option_id"].values())) == 2
    )
    return {
        "schema": "carnot.exp7462.option_readout_contract.v1",
        "stable_option_ids": list(OPTION_IDS),
        "display_labels": list(DISPLAY_LABELS),
        "prompt_boundary_checked": True,
        "last_evaluated_position_rule": "len(prompt_tokens)-1",
        "raw_logits_saved_before_normalization": True,
        "original_order": original,
        "reversed_order": reversed_result,
        "averaged_distribution_by_stable_id": averaged,
        "mutation_controls": attacks,
        "passed": all(attacks.values()),
        "scripted_logits_only": True,
        "current_model_forward": False,
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:  # pragma: no cover
    """Serialize stable JSONL bytes for hash-bound raw evidence."""

    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:  # pragma: no cover
    """Write one complete shard before replacing its task-owned path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _planned_rows(groups: Sequence[Mapping[str, Any]]) -> list[JsonDict]:  # pragma: no cover
    """Enumerate every future forward cell, including registered controls."""

    rows: list[JsonDict] = []
    for group in groups:
        for option_order in ("original", "reversed"):
            rows.append(
                {
                    "unit_id": f"{group['group_hash']}:{option_order}:full_source_response",
                    "group_id": group["group_id"],
                    "group_hash": group["group_hash"],
                    "source_hash": group["source_hash"],
                    "role": group["role"],
                    "corpus": group["corpus"],
                    "arm": "full_source_response",
                    "option_order": option_order,
                    "fit_seed": None,
                    "token_count": None,
                    "eligible": None,
                    "attempted": False,
                    "completed": False,
                    "failed": False,
                    "censored": False,
                    "unstarted": True,
                    "status": "unstarted",
                    "failure": None,
                    "raw_logits": None,
                    "service_time_s": None,
                }
            )
    calibration = sorted(
        (row for row in groups if row["role"] == "calibration_tuning"),
        key=lambda row: _stable_hash(f"v654-control:{row['group_hash']}"),
    )[:40]
    for group in calibration:
        for arm in ("response_only", "shuffled_source"):
            for option_order in ("original", "reversed"):
                rows.append(
                    {
                        "unit_id": f"{group['group_hash']}:{option_order}:{arm}",
                        "group_id": group["group_id"],
                        "group_hash": group["group_hash"],
                        "source_hash": group["source_hash"],
                        "role": group["role"],
                        "corpus": group["corpus"],
                        "arm": arm,
                        "option_order": option_order,
                        "fit_seed": None,
                        "token_count": None,
                        "eligible": None,
                        "attempted": False,
                        "completed": False,
                        "failed": False,
                        "censored": False,
                        "unstarted": True,
                        "status": "unstarted",
                        "failure": None,
                        "raw_logits": None,
                        "service_time_s": None,
                    }
                )
    return rows


def seal_raw_protocol(
    raw_dir: Path, cohort: Mapping[str, Any], interface: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover
    """Seal source, evaluator, group, plan, and interface evidence separately."""

    planned = _planned_rows(cohort["groups"])
    payloads: list[tuple[str, str, bytes, int]] = []
    for name, kind, rows in (
        ("cohort_predictors.jsonl", "predictor", cohort["predictors"]),
        ("cohort_evaluators.jsonl", "evaluator", cohort["evaluators"]),
        ("cohort_groups.jsonl", "group", cohort["groups"]),
        ("planned_rows.jsonl", "planned_cell", planned),
    ):
        payloads.append((name, kind, _jsonl_bytes(rows), len(rows)))
    interface_payload = (
        json.dumps(interface, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    payloads.append(("option_readout_contract.json", "interface", interface_payload, 1))
    shards: list[JsonDict] = []
    for name, kind, payload, row_count in payloads:
        path = raw_dir / name
        _atomic_bytes(path, payload)
        shards.append(
            {
                "path": name,
                "kind": kind,
                "rows": row_count,
                "size_bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest: JsonDict = {
        "schema": "carnot.exp7462.raw_manifest.v1",
        "raw_dir": RAW_DIR.as_posix(),
        "shards": shards,
        "counts": deepcopy(cohort["counts"]),
        "total_groups": len(cohort["groups"]),
        "planned_cells": len(planned),
        "control_group_ids": sorted(
            {
                row["group_id"]
                for row in planned
                if row["arm"] in {"response_only", "shuffled_source"}
            }
        ),
        "licenses": {
            corpus: {"license": value["license"], "revision": value["revision"]}
            for corpus, value in RELEASES.items()
        },
        "selection_without_outcomes": True,
        "group_and_source_role_disjoint": cohort["group_and_source_role_disjoint"],
        "excluded_online_candidates": deepcopy(cohort["exclusions"]),
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    atomic_json(raw_dir / "manifest.json", manifest)
    return {"manifest": manifest, "rows": planned}


def reload_raw_protocol(raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Rehash raw shards and independently reduce counts and interface controls."""

    manifest = _load_object(raw_dir / "manifest.json")
    expected_hash = manifest.get("manifest_hash")
    stable = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if expected_hash != canonical_hash(stable):
        raise OptionProtocolError("raw_manifest_hash_mismatch")
    by_kind: dict[str, Any] = {}
    for shard in manifest.get("shards") or []:
        path = raw_dir / str(shard["path"])
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise OptionProtocolError(f"raw_shard_hash_mismatch:{shard['path']}")
        if shard["kind"] == "interface":
            by_kind["interface"] = _load_object(path)
        else:
            rows = _load_jsonl(path)
            if len(rows) != shard["rows"]:
                raise OptionProtocolError(f"raw_shard_count_mismatch:{shard['path']}")
            by_kind[str(shard["kind"])] = rows
    group_rows = by_kind.get("group", [])
    counts = Counter(str(row["role"]) for row in group_rows)
    source_roles: dict[str, set[str]] = {}
    group_roles: dict[str, set[str]] = {}
    for row in group_rows:
        source_roles.setdefault(str(row["source_hash"]), set()).add(str(row["role"]))
        group_roles.setdefault(str(row["group_id"]), set()).add(str(row["role"]))
    return {
        "manifest_hash": expected_hash,
        "counts": {role: counts.get(role, 0) for role in ROLE_ORDER},
        "total_groups": len(group_rows),
        "planned_cells": len(by_kind.get("planned_cell", [])),
        "all_cells_unstarted": all(
            row.get("status") == "unstarted"
            and row.get("attempted") is False
            and row.get("raw_logits") is None
            for row in by_kind.get("planned_cell", [])
        ),
        "interface_passed": by_kind.get("interface", {}).get("passed") is True,
        "cohort_disjoint": all(len(roles) == 1 for roles in source_roles.values())
        and all(len(roles) == 1 for roles in group_roles.values()),
        "predictor_isolated": all(
            not (
                set(row)
                & {
                    "label",
                    "annotation_labels",
                    "annotations",
                    "notes",
                    "detector_outputs",
                    "response_generator_identity",
                    "source_features",
                }
            )
            for row in by_kind.get("predictor", [])
        ),
        "option_contract_hash": canonical_hash(by_kind.get("interface", {})),
    }


def _validation_passed(receipts: object, names: Sequence[str]) -> bool:  # pragma: no cover
    """Require exactly one passing current receipt for each named command."""

    return isinstance(receipts, list) and all(
        sum(row.get("name") == name and row.get("passed") is True for row in receipts) == 1
        for name in names
    )


def independent_reduce(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> JsonDict:  # pragma: no cover
    """Rebuild readiness from raw shards and exact validation receipts."""

    raw = reload_raw_protocol(root / RAW_DIR)
    names = [*AFFECTED_CHECK_NAMES]
    if require_terminal:
        names.extend(TERMINAL_CHECK_NAMES)
    validation_ok = _validation_passed(artifact.get("validation_receipts"), names)
    gates = protocol_gates(
        raw["counts"],
        interface_passed=bool(raw["interface_passed"]),
        cohort_disjoint=bool(raw["cohort_disjoint"]),
        predictor_isolated=bool(raw["predictor_isolated"]),
        validation_passed=validation_ok,
    )
    reduced = reduce_protocol(gates, flagged_adversarial=bool(artifact.get("flagged_adversarial")))
    return {
        **reduced,
        "raw_manifest_hash": raw["manifest_hash"],
        "group_counts": raw["counts"],
        "total_groups": raw["total_groups"],
        "planned_cells": raw["planned_cells"],
        "all_cells_unstarted": raw["all_cells_unstarted"],
        "option_contract_hash": raw["option_contract_hash"],
        "required_validation_passed": validation_ok,
        "acceptance_gate_results": gates,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Name the first required failure without hiding later failures."""

    failed = [row for row in gates if row.get("passed") is not True]
    required = [row for row in failed if row.get("category") != "confirmatory_benefit"]
    first = required[0] if required else None
    return {
        "passed": not required,
        "failed_check": first.get("check") if first else None,
        "upstream": "current_exp7462_protocol" if first else None,
        "path": RAW_DIR.as_posix() if first else None,
        "field": "passed" if first else None,
        "expected": first.get("expected") if first else None,
        "observed": first.get("observed") if first else None,
        "op": first.get("op") if first else None,
        "failed_required_checks": [row["check"] for row in required],
        "failed_confirmatory_checks": [
            row["check"] for row in failed if row.get("category") == "confirmatory_benefit"
        ],
    }


def _field_principles() -> dict[str, str]:  # pragma: no cover
    """Explain why every required terminal field exists."""

    specific = {
        "schema": "A versioned schema fixes the exact experiment, milestone, and terminal format.",
        "run_date": "The fixed date and measured clocks prevent a replay from posing as this run.",
        "preconditions_checked": "Exact paths, ownership, devices, and prior flags prevent unauthenticated work.",
        "MODEL_SPECS": "An empty list states that this protocol made no current LLM call.",
        "model_specs": "The lowercase alias keeps readers consistent without inventing a model.",
        "model_invoked": "The flag separates current attempted calls from archived and scripted evidence.",
        "invocation_counts": "Balanced load, forward, and generation counters expose unfinished calls.",
        "inference_substrate": "The substrate names scripted numeric protocol work instead of live inference.",
        "inference_substrate_class": "The no-model class prevents short protocol work from claiming model compute.",
        "execution_venue": "The host venue stays separate from actual device and historical board evidence.",
        "duration_s": "Measured duration exposes missing work without padding the run.",
        "phase_spans": "Monotonic phase spans bind progress events to completed checkpoints.",
        "random_seed": "Frozen fit, order, audit, and bootstrap seeds prevent outcome-driven choices.",
        "reproducibility_checksum": "One hash binds code, protocol, cohorts, raw shards, and validation scope.",
        "source_artifact_hashes": "Exact upstream bytes and old flags preserve history without rehabilitation.",
        "rows": "Every planned group, order, and control stays visible as unstarted work.",
        "sample_size_budget": "Planned, attempted, complete, failed, censored, and unstarted units stay distinct.",
        "acceptance_gate_results": "Typed gates keep protocol validity separate from confirmatory benefit.",
        "gate_check_summary": "A failed result names its exact check, upstream, path, field, and value.",
        "honest_verdict": "A terminal null says the protocol is ready without claiming scientific efficacy.",
        "verdict_class": "The closed class lets readers distinguish null, blocked, and disqualified results.",
        "verifier_is_oracle": "False records that interface checks are not the future evaluation oracle.",
        "flagged_adversarial": "Reader flags remain truthful and cannot be cleared to open a gate.",
        "validation_receipts": "Exact command, exit, duration, and log hashes prove each current check ran.",
        "repository_health": "Broad-suite failures stay visible without replacing scoped required checks.",
        "field_principles": "Field explanations let auditors inspect why evidence exists.",
        "option_protocol_ready_score": "A bare score qualifies only the protocol and analytic interface.",
        "cohort_manifest": "Group and role hashes prevent outcome leakage and cross-role reuse.",
        "comparison_plan": "Controls, costs, minima, stopping, and multiplicity are frozen before capture.",
        "option_readout_contract": "Token position, boundaries, stable IDs, and raw logits expose readout defects.",
    }
    return {
        field: specific.get(
            field, "This field preserves one required part of the terminal audit record."
        )
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _source_hashes_valid(root: Path, values: object) -> bool:  # pragma: no cover
    """Rehash repository and authenticated cache inputs by their declared paths."""

    if not isinstance(values, Mapping):
        return False
    for row in values.values():
        if not isinstance(row, Mapping):
            return False
        path = Path(str(row.get("path") or ""))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
            return False
    return True


def _phase_span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    """Record one measured monotonic span and its completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint": checkpoint,
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh-process capability replay and strict candidate readers."""

    python = ".venv/bin/python"
    common = ("--date", RUN_DATE, "--root", ".")
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "declared_entrypoint_cold_replay",
                (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
                "candidate_artifact",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_raw_reduction",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    *common,
                    "--independent-reduce",
                    str(candidate),
                ),
                "candidate_raw_rows",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate_artifact",
            ),
            "required_validation",
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
                "candidate_artifact",
            ),
            "required_validation",
            True,
        ),
    ]


def _build_artifact(  # pragma: no cover - runtime assembly.
    *,
    root: Path,
    preconditions: list[JsonDict],
    source_hashes: dict[str, JsonDict],
    raw: Mapping[str, Any],
    interface: Mapping[str, Any],
    device_identity: Mapping[str, Any],
    repository_health: Mapping[str, Any],
    validation_receipts: list[JsonDict],
    phase_spans: list[JsonDict],
    started_at: str,
    started_ns: int,
    ended_ns: int,
    protocol_s: float,
    flagged_adversarial: bool,
    require_terminal: bool,
) -> JsonDict:
    """Assemble one schema-complete candidate from reloadable evidence."""

    counts = raw["manifest"]["counts"]
    gates = protocol_gates(
        counts,
        interface_passed=bool(interface.get("passed")),
        cohort_disjoint=bool(raw["manifest"]["group_and_source_role_disjoint"]),
        predictor_isolated=True,
        validation_passed=_validation_passed(
            validation_receipts,
            [
                *AFFECTED_CHECK_NAMES,
                *(TERMINAL_CHECK_NAMES if require_terminal else ()),
            ],
        ),
    )
    reduction = reduce_protocol(gates, flagged_adversarial=flagged_adversarial)
    duration = (ended_ns - started_ns) / 1_000_000_000
    rows = raw["rows"]
    source_hashes = deepcopy(source_hashes)
    for relative in (
        RAW_DIR / "manifest.json",
        *(RAW_DIR / str(row["path"]) for row in raw["manifest"]["shards"]),
    ):
        path = root / relative
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "original_flagged_adversarial": None,
        }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": reduction["status"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "clock_identity": {
            "wall": "datetime.now(datetime.UTC)",
            "monotonic": "time.monotonic_ns",
        },
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "device_identity": deepcopy(dict(device_identity)),
        "duration_s": duration,
        "duration_components_s": {
            "model_load_s": 0.0,
            "forward_s": 0.0,
            "generation_s": 0.0,
            "numeric_fitting_s": 0.0,
            "protocol_s": protocol_s,
            "validation_s": sum(float(row.get("duration_s") or 0.0) for row in validation_receipts),
        },
        "phase_spans": phase_spans,
        "random_seed": {
            "fit_seeds": list(FIT_SEEDS),
            "ordering_seed": ORDERING_SEED,
            "audit_seed": AUDIT_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "deterministic_audits": "no_rng",
        },
        "reproducibility_checksum": None,
        "source_artifact_hashes": source_hashes,
        "rows": rows,
        "sample_size_budget": {
            "independent_groups": deepcopy(counts),
            "planned": len(rows),
            "attempted": 0,
            "complete": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": len(rows),
            "token_ceiling": TOKEN_CEILING,
            "overlength_groups": 0,
            "overlength_status": "deferred_until_embedded_tokenizer_capture",
            "replacement_after_scoring": False,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": reduction["honest_verdict"],
        "verdict_class": reduction["verdict_class"],
        "verifier_is_oracle": False,
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": validation_receipts,
        "repository_health": deepcopy(dict(repository_health)),
        "field_principles": _field_principles(),
        "option_protocol_ready_score": reduction["option_protocol_ready_score"],
        "cohort_manifest": deepcopy(raw["manifest"]),
        "comparison_plan": comparison_plan(),
        "option_readout_contract": deepcopy(dict(interface)),
        "small_ebm_training": deepcopy(SMALL_EBM_TRAINING),
        "current_invocation_events": [],
        "historical_inference_sidecars": [
            {
                "path": JEVBENCH_RESULT_PATH.as_posix(),
                "sha256": source_hashes[JEVBENCH_RESULT_PATH.as_posix()]["sha256"],
                "scope": "historical_model_receipts",
                "model_invoked_in_historical_run": True,
            },
            {
                "path": V653_EMBEDDING_PATH.as_posix(),
                "sha256": source_hashes[V653_EMBEDDING_PATH.as_posix()]["sha256"],
                "scope": "historical_model_receipts",
                "forward_calls_attempted_in_historical_run": 0,
            },
        ],
        "confirmatory_minima_met": reduction["confirmatory_minima_met"],
        "independent_reduction": None,
        "promotion_score": 0,
        "numbered_e2e_applicable": [],
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "cold_replay": "fresh_process",
            "independent_raw_reduction": "fresh_process",
            "numbered_runtime_e2e": "not_applicable_pure_protocol_work",
        },
        "production_defaults_changed": False,
        "external_publication_authorized": False,
    }
    artifact["independent_reduction"] = independent_reduce(
        artifact, root=root, require_terminal=require_terminal
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:  # pragma: no cover - fresh-process terminal reader.
    """Cold-check identity, raw reduction, hashes, principles, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    errors = validate_artifact_shape(artifact)
    try:
        reduction = independent_reduce(artifact, root=root, require_terminal=require_terminal)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        reduction = {}
        errors.append("independent_reduction_failed")
    if artifact.get("independent_reduction") != reduction:
        errors.append("independent_reduction_mismatch")
    for key in (
        "option_protocol_ready_score",
        "confirmatory_minima_met",
        "status",
        "honest_verdict",
        "verdict_class",
        "promotion_score",
    ):
        if artifact.get(key) != reduction.get(key):
            errors.append(f"{key}_mismatch")
    if not _source_hashes_valid(root, artifact.get("source_artifact_hashes")):
        errors.append("source_artifact_hashes_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def run_experiment(
    root: Path, run_date: str, *, output: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared capability E2E.
    """Authenticate, seal, validate, replay, and atomically publish the protocol."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    root = root.resolve()
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes = collect_preconditions(root)
    repository_health = previous_repository_health(root)
    for observation in repository_health["observations"]:
        log_label = str(observation.get("log_path") or "")
        log_path = root / log_label
        if log_label and log_path.is_file():
            source_hashes[f"repository_health:{log_label}"] = {
                "path": log_label,
                "sha256": sha256_file(log_path),
                "original_flagged_adversarial": None,
            }
    progress(run_started, "device_identity", "before_subprocess")
    device_identity = collect_device_identity()
    progress(run_started, "device_identity", "after_subprocess")
    failed = [row for row in preconditions if row["passed"] is not True]
    spans.append(
        _phase_span("preconditions", phase_started, run_started, len(preconditions), "inputs")
    )
    progress(run_started, "preconditions", "complete", failed=len(failed))
    if failed:
        row = failed[0]
        raise OptionProtocolError(
            f"precondition_failed:{row['path']}:{row['field']}:{row['observed']}"
        )

    progress(run_started, "model_load", "before", planned=0)
    phase_started = time.monotonic()
    spans.append(_phase_span("model_load", phase_started, run_started, 0, "no_model_load"))
    progress(run_started, "model_load", "after", completed=0)
    progress(run_started, "generation", "before", planned=0)
    phase_started = time.monotonic()
    spans.append(_phase_span("generation", phase_started, run_started, 0, "no_generation"))
    progress(run_started, "generation", "after", completed=0)

    protocol_started = time.monotonic()
    progress(run_started, "raw_logit_interface", "before_benchmark", planned=7)
    phase_started = time.monotonic()
    interface = run_interface_controls()
    spans.append(_phase_span("raw_logit_interface", phase_started, run_started, 7, "interface"))
    progress(
        run_started,
        "raw_logit_interface",
        "after_benchmark",
        passed=interface["passed"],
    )

    progress(run_started, "cohort", "before_benchmark", planned=560)
    phase_started = time.monotonic()
    old_predictors = _load_jsonl(root / V653_RAW_DIR / "predictors.jsonl")
    old_evaluators = _load_jsonl(root / V653_RAW_DIR / "evaluators.jsonl")
    ragtruth_receipt = authenticate_ragtruth_assets(RAGTRUTH_CACHE_ROOT)
    source_rows, response_rows = load_ragtruth_release(ragtruth_receipt, started=run_started)
    expanded = select_ragtruth_panel(
        source_rows,
        response_rows,
        caps={**RAGTRUTH_CAPS, "internal_test": RAGTRUTH_CAPS["internal_test"] + ONLINE_CAP},
    )
    online_pairs = [
        (predictor, evaluator)
        for predictor, evaluator in _joined_rows(expanded["predictors"], expanded["evaluators"])
        if predictor.get("role") == "internal_test"
    ]
    cohort = freeze_cohort(
        old_predictors,
        old_evaluators,
        [pair[0] for pair in online_pairs],
        [pair[1] for pair in online_pairs],
    )
    spans.append(_phase_span("cohort", phase_started, run_started, len(cohort["groups"]), "cohort"))
    progress(run_started, "cohort", "after_benchmark", completed=len(cohort["groups"]))

    progress(run_started, "seal", "start")
    phase_started = time.monotonic()
    raw = seal_raw_protocol(root / RAW_DIR, cohort, interface)
    replay = reload_raw_protocol(root / RAW_DIR)
    if replay["counts"] != cohort["counts"] or not replay["all_cells_unstarted"]:
        raise OptionProtocolError("raw_protocol_replay_mismatch")
    spans.append(_phase_span("seal", phase_started, run_started, len(raw["rows"]), "raw_shards"))
    progress(run_started, "seal", "complete", rows=len(raw["rows"]))
    protocol_s = time.monotonic() - protocol_started

    private_root = Path(tempfile.mkdtemp(prefix="exp7462-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "affected_validation", phase_started, run_started, len(affected), "affected_checks"
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )

    flagged = bool(plan_errors) or not affected_reduction["passed"]
    candidate = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        raw=raw,
        interface=interface,
        device_identity=device_identity,
        repository_health=repository_health,
        validation_receipts=affected,
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        protocol_s=protocol_s,
        flagged_adversarial=flagged,
        require_terminal=False,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_plan = _terminal_commands(candidate_path)
    progress(run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_plan))
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root, terminal_plan, log_dir=root / RAW_DIR / "validation/terminal"
    )
    spans.append(
        _phase_span(
            "terminal_validation", phase_started, run_started, len(terminal), "terminal_readers"
        )
    )
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
    final = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        raw=raw,
        interface=interface,
        device_identity=device_identity,
        repository_health=repository_health,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        protocol_s=protocol_s,
        flagged_adversarial=flagged or not terminal_passed or critical,
        require_terminal=True,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "publish", "before_atomic_terminal", path=output)
    atomic_json(root / output, final)
    progress(run_started, "publish", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed run date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the protocol or one fresh-process terminal reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = (
            validate_artifact(value, root=root, require_terminal=False)
            if value
            else ["artifact_unreadable"]
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        reduced = independent_reduce(value, root=root, require_terminal=False) if value else {}
        passed = bool(value) and reduced == value.get("independent_reduction")
        print(json.dumps({"passed": passed, "reduced": reduced}, sort_keys=True), flush=True)
        return int(not passed)
    run_experiment(root, args.date, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
