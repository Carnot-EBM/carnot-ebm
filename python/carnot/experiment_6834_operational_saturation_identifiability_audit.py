"""Cold-audit operational saturation without trusting producer aggregates.

Spec refs: REQ-CONSTRAINT-6834 and SCENARIO-CONSTRAINT-6834-*.

The source corpus contains authentic model bytes. This module independently
parses those bytes and reimplements the public fixture rules. Parse failure is
an observed transport failure, but it cannot reveal which latent operational
fields the model would preserve after transport is corrected.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
FIXTURE_PATH = Path("results/experiment_6832_operational_obligation_saturation_fixture.json")
CORPUS_PATH = Path("results/experiment_6833_sota_operational_obligation_saturation_corpus.json")
MODULE_PATH = Path("python/carnot/experiment_6834_operational_saturation_identifiability_audit.py")
WRAPPER_PATH = Path(
    "scripts/experiments/experiment_6834_operational_saturation_identifiability_audit.py"
)
OUTPUT_PATH = Path("results/experiment_6834_operational_saturation_identifiability_audit.json")

ARTIFACT_SCHEMA = "carnot.experiment_6834.operational_saturation_identifiability_audit.v1"
FIXTURE_SCHEMA = "carnot.experiment_6832.operational_obligation_saturation_fixture.v1"
CORPUS_SCHEMA = "carnot.experiment_6833.sota_operational_obligation_saturation_corpus.v1"
EXPECTED_FIXTURE_SHA256 = "sha256:05c88b1cc075fc789763e155df1ab67fa5ea5d3845851aea4d4d83f24027993f"
EXPECTED_CORPUS_SHA256 = "sha256:05cdd74188f48163a9e9e875ad4fda2389a07e1b4aa6c060590b25ef772ee6c5"
EXPECTED_MODEL_HASHES = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": (
        "sha256:ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
    ),
    "unsloth/gemma-4-31B-it-GGUF": (
        "sha256:9fdf3dc8b0384830b4402d151388c140bd8eb2abf8d60588d8224231198254a1"
    ),
    "unsloth/gemma-4-26B-A4B-it-GGUF": (
        "sha256:34c746b1d50ab813e29cd46c4796e3f43c741901a582f93a67b55b9fc9687b35"
    ),
}
MODEL_FAMILY_BY_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "qwen36",
    "unsloth/gemma-4-31B-it-GGUF": "gemma31_dense",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "gemma26_moe",
}
OBLIGATION_FIELDS = (
    "prerequisite",
    "authority",
    "fallback",
    "execution_consequence",
    "priority",
)
PRIORITY_ORDER = ("hard", "binding", "soft")
OBLIGATION_COUNTS = (1, 2, 4, 6, 8)
ARMS = ("typed", "compressed")
RANDOM_SEED = 6834
BOOTSTRAP_REPLICATES = 2_000
INFERENCE_SUBSTRATE = (
    "deterministic_verifier_plus_replay (fresh-process deterministic CPU audit, no LLM)"
)
INDEPENDENT_PARSER_ID = "exp6834.strict_canonical_json_raw_bytes.v1"
INDEPENDENT_REDUCER_ID = "exp6834.public_contract_field_reducer.v1"
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ATTACK_IDS = (
    "row_deletion",
    "row_duplicate",
    "row_order",
    "action_label_permutation",
    "model_label_permutation",
    "model_label_masking",
    "prompt_label_permutation",
    "checker_mutation",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "independent_parser_id",
    "independent_reducer_id",
    "per_unit_rows",
    "row_coverage",
    "per_obligation_metrics",
    "joint_success_metrics",
    "parse_failure_metrics",
    "interaction_penalties",
    "saturation_curves",
    "paired_arm_effects",
    "identifiability_result",
    "minimum_identifying_support",
    "collision_witnesses",
    "attack_results",
    "operational_saturation_audit_complete",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "The versioned name fixes the cold-audit contract.",
    "experiment_id": "The stable identifier binds the artifact to Exp6834.",
    "run_date": "The supplied date distinguishes this execution from later audits.",
    "status": "The status separates a complete audit from a precondition stop.",
    "field_principles": "Every top-level field explains why it exists.",
    "preconditions_checked": "Source and receipt checks stop incomplete evidence before reduction.",
    "inference_substrate": "The substrate records deterministic CPU replay with no model call.",
    "duration_s": "Measured wall time makes an incomplete execution visible.",
    "random_seed": "The fixed seed makes scenario bootstrap and attacks repeatable.",
    "reproducibility_checksum": "The checksum binds inputs, code, rows, commands, and output.",
    "source_artifact_hashes": "Fixture, corpus, and model hashes bind the evidence identities.",
    "implementation_hashes": "Independent module and wrapper hashes bind the executed code.",
    "methodology": "The method states estimands, units, commands, and the no-import boundary.",
    "independent_parser_id": "A distinct parser identity prevents producer parse reuse.",
    "independent_reducer_id": "A distinct reducer identity prevents producer aggregate reuse.",
    "per_unit_rows": "All source and attack rows support cold aggregate recomputation.",
    "row_coverage": "Exact identities expose deletion, duplication, and label collapse.",
    "per_obligation_metrics": "Typed field cells keep operational meanings separate.",
    "joint_success_metrics": "Joint cells measure compositional success over all source rows.",
    "parse_failure_metrics": "Transport failure stays separate from semantic saturation.",
    "interaction_penalties": "Fixed-count matched contrasts isolate interaction from count.",
    "saturation_curves": "Unpooled family curves prevent one model from hiding another.",
    "paired_arm_effects": "Scenario matches and intervals preserve the arm comparison unit.",
    "identifiability_result": "The disposition says whether support separates target field policies.",
    "minimum_identifying_support": "The finite certificate lists every required and missing cell.",
    "collision_witnesses": "Constructive policy pairs make non-identification replayable.",
    "attack_results": "Named mutations show which audit invariants were exercised.",
    "operational_saturation_audit_complete": "Completion records procedure, not effect direction.",
    "gate_check_summary": "The summary names the exact blocking check or complete gate.",
    "verifier_is_oracle": "False limits this audit to the frozen public contract.",
    "verdict_class": "A closed class keeps scientific disposition machine-readable.",
    "honest_verdict": "The terminal prefix states the row-supported audit boundary.",
}


class AuditError(ValueError):
    """Expose one stable error when a finite audit invariant is malformed."""


def canonical_bytes(value: Any) -> bytes:
    """Encode JSON once so equality and hash checks use the same bytes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed digest so every identity uses one representation."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value through the canonical byte representation."""

    return sha256_bytes(canonical_bytes(value))


def _json_object(raw: bytes) -> JsonDict:
    """Return a JSON object or an empty object for a failed source read."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record expected and observed values without hiding a failed gate."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def parse_raw_output(raw: bytes) -> JsonDict:
    """Parse exact canonical JSON without extraction, fences, or repair."""

    def failure(code: str) -> JsonDict:
        return {"error": code, "parsed": False, "selected_action_ids": []}

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return failure("invalid_utf8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return failure("invalid_json")
    if canonical_bytes(value) != raw:
        return failure("non_canonical_json")
    if not isinstance(value, dict) or set(value) != {"selected_action_ids"}:
        return failure("invalid_response_fields")
    action_ids = value["selected_action_ids"]
    if not isinstance(action_ids, list):
        return failure("invalid_action_list")
    if any(not isinstance(item, str) or not item for item in action_ids):
        return failure("invalid_action_id")
    if len(action_ids) != len(set(action_ids)):
        return failure("duplicate_action_id")
    return {"error": None, "parsed": True, "selected_action_ids": sorted(action_ids)}


def _priority_key(obligation: Mapping[str, Any]) -> tuple[int, int, int, str]:
    """Order conflicts through the public priority and authority fields."""

    contract = obligation["contract"]
    priority = contract["priority"]
    return (
        PRIORITY_ORDER.index(priority["class"]),
        int(contract["authority"]["order"]),
        -int(priority["weight"]),
        str(obligation["obligation_id"]),
    )


def resolve_scenario(scenario: Mapping[str, Any]) -> JsonDict:
    """Resolve the public five-field contract without importing its producer."""

    candidates = {row["action_id"]: row for row in scenario["candidates"]}
    facts = set(scenario["observed_facts"])
    active: list[Mapping[str, Any]] = []
    inactive: list[Mapping[str, Any]] = []
    invalid = False
    for obligation in scenario["obligations"]:
        contract = obligation["contract"]
        prerequisite = contract["prerequisite"]
        is_active = set(prerequisite["all_of"]).issubset(facts) and not (
            set(prerequisite["none_of"]) & facts
        )
        (active if is_active else inactive).append(obligation)
        if is_active:
            action = candidates.get(obligation["action"]["action_id"])
            invalid = invalid or action is None
            if action is not None:
                invalid = invalid or action.get("authority") != contract["authority"]["issuer"]
                invalid = invalid or action.get("consequence") != contract["execution_consequence"]

    if invalid:
        fail_closed = scenario["fail_closed_action_id"]
        return {
            "legal_action_ids": [fail_closed],
            "obligations": {
                row["obligation_id"]: {
                    "action_id": fail_closed,
                    "active": row in active,
                    "disposition": "fail_closed",
                }
                for row in scenario["obligations"]
            },
        }

    resolutions: dict[str, JsonDict] = {}
    for obligation in inactive:
        resolutions[obligation["obligation_id"]] = {
            "action_id": obligation["contract"]["fallback"]["action_id"],
            "active": False,
            "disposition": "fallback",
        }
    by_resource: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for obligation in active:
        action = candidates[obligation["action"]["action_id"]]
        by_resource[action["resource"]].append(obligation)
    for group in by_resource.values():
        ordered = sorted(group, key=_priority_key)
        winner = ordered[0]
        resolutions[winner["obligation_id"]] = {
            "action_id": winner["action"]["action_id"],
            "active": True,
            "disposition": "constructive",
        }
        for obligation in ordered[1:]:
            resolutions[obligation["obligation_id"]] = {
                "action_id": obligation["contract"]["fallback"]["action_id"],
                "active": True,
                "disposition": "preempted",
            }
    return {
        "legal_action_ids": sorted({row["action_id"] for row in resolutions.values()}),
        "obligations": resolutions,
    }


def _action_index(scenario: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index candidate actions for the independent field checks."""

    return {row["action_id"]: row for row in scenario["candidates"]}


def _expected_obligation(
    scenario: Mapping[str, Any], obligation: Mapping[str, Any]
) -> tuple[JsonDict, Mapping[str, Any] | None]:
    """Return the resolved obligation and its selected candidate action."""

    resolution = resolve_scenario(scenario)["obligations"][obligation["obligation_id"]]
    return resolution, _action_index(scenario).get(resolution["action_id"])


def _field_decisions(
    scenario: Mapping[str, Any], selected: set[str], obligation: Mapping[str, Any]
) -> dict[str, bool]:
    """Reimplement all five public checks in one auditable decision block."""

    resolution, action = _expected_obligation(scenario, obligation)
    fail_closed = resolution["disposition"] == "fail_closed"
    expected_id = resolution["action_id"]
    prerequisite = expected_id in selected

    if fail_closed:
        boundary = selected == {scenario["fail_closed_action_id"]}
        return {field: boundary for field in OBLIGATION_FIELDS}

    action_present = action is not None and expected_id in selected
    if not action_present:
        authority = False
        execution_consequence = False
        priority = False
    else:
        assert action is not None
        if resolution["disposition"] == "constructive":
            issuer = obligation["contract"]["authority"]["issuer"]
            resource = action["resource"]
            relevant = [
                row
                for item, row in _action_index(scenario).items()
                if item in selected and row["resource"] == resource
            ]
            authority = action["authority"] == issuer and all(
                row["authority"] == issuer for row in relevant
            )
            execution_consequence = (
                action["consequence"] == obligation["contract"]["execution_consequence"]
            )
        else:
            authority = action["authority"] == "fixture_safety_boundary"
            execution_consequence = action["kind"] == "no_op" and action["consequence"] == {
                "add": [],
                "remove": [],
            }
        target = _action_index(scenario)[obligation["action"]["action_id"]]
        selected_on_resource = {
            row["action_id"]
            for row in scenario["candidates"]
            if row["action_id"] in selected and row["resource"] == target["resource"]
        }
        expected_on_resource = {
            row["action_id"]
            for row in resolve_scenario(scenario)["obligations"].values()
            if row["disposition"] == "constructive"
            and _action_index(scenario)[row["action_id"]]["resource"] == target["resource"]
        }
        priority = selected_on_resource == expected_on_resource

    target_id = obligation["action"]["action_id"]
    fallback_id = obligation["contract"]["fallback"]["action_id"]
    if resolution["disposition"] == "constructive":
        fallback = target_id in selected and fallback_id not in selected
    else:
        fallback = fallback_id in selected and target_id not in selected
    return {
        "prerequisite": prerequisite,
        "authority": authority,
        "fallback": fallback,
        "execution_consequence": execution_consequence,
        "priority": priority,
    }


def check_obligation(
    scenario: Mapping[str, Any], selected_action_ids: Sequence[str], obligation_id: str
) -> JsonDict:
    """Check every typed field for one public obligation identifier."""

    obligations = {row["obligation_id"]: row for row in scenario["obligations"]}
    if obligation_id not in obligations:
        return {"fields": {field: False for field in OBLIGATION_FIELDS}, "passed": False}
    fields = _field_decisions(scenario, set(selected_action_ids), obligations[obligation_id])
    return {"fields": fields, "passed": all(fields.values())}


def check_joint(scenario: Mapping[str, Any], raw: bytes) -> JsonDict:
    """Check exact action-set equality after the independent strict parse."""

    parsed = parse_raw_output(raw)
    if not parsed["parsed"]:
        return {
            "obligation_checks": {},
            "parse_error": parsed["error"],
            "parsed": False,
            "passed": False,
            "selected_action_ids": [],
        }
    selected = parsed["selected_action_ids"]
    checks = {
        row["obligation_id"]: check_obligation(scenario, selected, row["obligation_id"])
        for row in scenario["obligations"]
    }
    expected = sorted(scenario["legal_action_ids"])
    return {
        "obligation_checks": checks,
        "parse_error": None,
        "parsed": True,
        "passed": selected == expected and all(row["passed"] for row in checks.values()),
        "selected_action_ids": selected,
    }


def expected_row_ids(
    fixture: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Build the complete model, scenario, and arm identity roster."""

    return sorted(
        f"{spec['hub_id']}|{scenario['scenario_id']}|{arm}"
        for spec in model_specs
        for scenario in fixture.get("scenarios", [])
        for arm in ARMS
    )


def _row_identity(row: Mapping[str, Any]) -> str:
    """Rebuild an identity from semantic labels instead of trusting row_id."""

    return f"{row.get('model_id')}|{row.get('scenario_id')}|{row.get('arm')}"


def coverage_report(rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]) -> JsonDict:
    """Detect missing, duplicate, unexpected, and masked model identities."""

    observed = [_row_identity(row) for row in rows if row.get("row_type", "source") == "source"]
    counts = Counter(observed)
    expected = set(expected_ids)
    observed_set = set(observed)
    missing = sorted(expected - observed_set)
    unexpected = sorted(observed_set - expected)
    duplicates = sorted(identity for identity, count in counts.items() if count > 1)
    model_count = len(
        {str(row.get("model_id")) for row in rows if row.get("row_type", "source") == "source"}
    )
    valid = (
        len(observed) == len(expected_ids)
        and len(observed_set) == len(expected_ids)
        and not missing
        and not unexpected
        and not duplicates
        and model_count == len(EXPECTED_MODEL_HASHES)
    )
    return {
        "expected": len(expected_ids),
        "observed": len(observed),
        "observed_unique": len(observed_set),
        "model_identity_count": model_count,
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "valid": valid,
    }


def _decode_receipt(row: Mapping[str, Any]) -> tuple[bytes, str | None]:
    """Decode one raw receipt and preserve a stable failure code."""

    encoded = row.get("raw_output_bytes_b64")
    if not isinstance(encoded, str):
        return b"", "missing_raw_output_bytes"
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error):
        return b"", "invalid_raw_output_base64"
    if len(raw) != row.get("raw_output_byte_length"):
        return raw, "raw_output_length_mismatch"
    if sha256_bytes(raw) != row.get("raw_output_sha256"):
        return raw, "raw_output_hash_mismatch"
    return raw, None


def recompute_source_rows(
    fixture: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Cold-reparse and recheck every source row in canonical identity order."""

    scenarios = {row["scenario_id"]: row for row in fixture.get("scenarios", [])}
    reduced: list[JsonDict] = []
    for source in source_rows:
        scenario_id = str(source.get("scenario_id"))
        if scenario_id not in scenarios:
            raise AuditError(f"unknown_scenario:{scenario_id}")
        scenario = scenarios[scenario_id]
        raw, receipt_error = _decode_receipt(source)
        checked = (
            check_joint(scenario, raw)
            if receipt_error is None
            else {
                "obligation_checks": {},
                "parse_error": receipt_error,
                "parsed": False,
                "passed": False,
                "selected_action_ids": [],
            }
        )
        if checked["parsed"]:
            obligation_results = checked["obligation_checks"]
        else:
            obligation_results = {
                obligation["obligation_id"]: {
                    "fields": {field: None for field in OBLIGATION_FIELDS},
                    "passed": None,
                }
                for obligation in scenario["obligations"]
            }
        producer_parse = source.get("parse_status", {})
        producer_joint = source.get("joint_result", {})
        producer_match = (
            producer_parse.get("parsed") == checked["parsed"]
            and producer_parse.get("error") == checked["parse_error"]
            and producer_joint.get("passed") == checked["passed"]
            and producer_joint.get("selected_action_ids") == checked["selected_action_ids"]
            and source.get("obligation_results") == checked["obligation_checks"]
        )
        reduced.append(
            {
                "row_type": "source",
                "row_id": str(source.get("row_id")),
                "model_id": str(source.get("model_id")),
                "model_family": MODEL_FAMILY_BY_ID.get(str(source.get("model_id")), "unknown"),
                "scenario_id": scenario_id,
                "template_id": scenario["template_id"],
                "permutation_id": scenario["permutation_id"],
                "arm": str(source.get("arm")),
                "obligation_count": int(scenario["obligation_count"]),
                "dependency_mode": scenario["dependency_mode"],
                "semantic_class": scenario["semantic_class"],
                "raw_output_bytes_b64": source.get("raw_output_bytes_b64"),
                "raw_output_byte_length": source.get("raw_output_byte_length"),
                "raw_output_sha256": source.get("raw_output_sha256"),
                "raw_receipt_error": receipt_error,
                "parse_status": {
                    "error": checked["parse_error"],
                    "parsed": checked["parsed"],
                },
                "selected_action_ids": checked["selected_action_ids"],
                "obligation_results": obligation_results,
                "joint_passed": checked["passed"],
                "producer_score_match": producer_match,
                "source_row_sha256": source.get("row_sha256"),
            }
        )
    return sorted(reduced, key=lambda row: row["row_id"])


def join_prompt_arms(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Join typed and compressed rows on the same model and scenario."""

    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("row_type", "source") != "source":
            continue
        key = (str(row["model_id"]), str(row["scenario_id"]))
        arm = str(row["arm"])
        if arm in grouped[key]:
            raise AuditError(f"duplicate_matched_arm:{key[0]}:{key[1]}:{arm}")
        grouped[key][arm] = row
    pairs: list[JsonDict] = []
    for (model_id, scenario_id), arms in sorted(grouped.items()):
        if set(arms) != set(ARMS):
            raise AuditError(f"incomplete_matched_pair:{model_id}:{scenario_id}")
        typed = dict(arms["typed"])
        compressed = dict(arms["compressed"])
        pairs.append(
            {
                "pair_id": f"{model_id}|{scenario_id}",
                "model_id": model_id,
                "scenario_id": scenario_id,
                "obligation_count": typed["obligation_count"],
                "dependency_mode": typed["dependency_mode"],
                "typed": typed,
                "compressed": compressed,
            }
        )
    return pairs


def bootstrap_interval(values: Sequence[float], *, seed: int, replicates: int) -> JsonDict:
    """Bootstrap complete scenario units with replacement under a frozen seed."""

    if not values:
        return {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "unit_count": 0,
            "bootstrap_replicates": replicates,
        }
    if replicates <= 0:
        raise AuditError("bootstrap_replicates_must_be_positive")
    numeric = [float(value) for value in values]
    estimate = sum(numeric) / len(numeric)
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(numeric) for _ in numeric) / len(numeric) for _ in range(replicates)
    )
    low = draws[int(0.025 * (replicates - 1))]
    high = draws[int(0.975 * (replicates - 1))]
    return {
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "unit_count": len(numeric),
        "bootstrap_replicates": replicates,
    }


def _group_seed(*parts: Any) -> int:
    """Derive stable bootstrap streams without Python's randomized hash."""

    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return RANDOM_SEED + int.from_bytes(digest[:4], "big")


def build_paired_arm_effects(pairs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Estimate typed-minus-compressed effects from scenario-matched rows."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(str(pair["model_id"]), int(pair["obligation_count"]))].append(pair)
    results: list[JsonDict] = []
    for (model_id, count), units in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        deltas = [
            float(bool(unit["typed"]["joint_passed"]))
            - float(bool(unit["compressed"]["joint_passed"]))
            for unit in units
        ]
        interval = bootstrap_interval(
            deltas,
            seed=_group_seed("arm", model_id, count),
            replicates=BOOTSTRAP_REPLICATES,
        )
        results.append(
            {
                "model_id": model_id,
                "model_family": MODEL_FAMILY_BY_ID.get(model_id, "unknown"),
                "obligation_count": count,
                "effect_definition": "typed_joint_minus_compressed_joint",
                "bootstrap_unit": "scenario_id",
                **interval,
            }
        )
    return results


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a rate only when its denominator is observed."""

    return numerator / denominator if denominator else None


def _interaction_breakdown(rows: Sequence[Mapping[str, Any]], outcome: str) -> JsonDict:
    """Keep independent and interacting counts visible inside one metric cell."""

    result: JsonDict = {}
    for mode in ("independent", "interacting"):
        selected = [row for row in rows if row["dependency_mode"] == mode]
        passed = sum(bool(row[outcome]) for row in selected)
        result[mode] = {
            "rows": len(selected),
            "passed": passed,
            "success_rate": _rate(passed, len(selected)),
        }
    return result


def _build_per_obligation_metrics(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce field outcomes by model, arm, count, type, and interaction."""

    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["arm"]), int(row["obligation_count"]))].append(row)
    metrics: list[JsonDict] = []
    for (model_id, arm, count), cell_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        for field in OBLIGATION_FIELDS:
            values = [
                result["fields"][field]
                for row in cell_rows
                for result in row["obligation_results"].values()
            ]
            observed = [value for value in values if isinstance(value, bool)]
            passed = sum(value is True for value in observed)
            by_interaction: JsonDict = {}
            for mode in ("independent", "interacting"):
                mode_values = [
                    result["fields"][field]
                    for row in cell_rows
                    if row["dependency_mode"] == mode
                    for result in row["obligation_results"].values()
                ]
                mode_observed = [value for value in mode_values if isinstance(value, bool)]
                mode_passed = sum(value is True for value in mode_observed)
                by_interaction[mode] = {
                    "expected_obligations": len(mode_values),
                    "observed_obligations": len(mode_observed),
                    "passed": mode_passed,
                    "success_rate": _rate(mode_passed, len(mode_values)),
                    "conditional_success_rate": _rate(mode_passed, len(mode_observed)),
                }
            metrics.append(
                {
                    "model_id": model_id,
                    "model_family": MODEL_FAMILY_BY_ID.get(model_id, "unknown"),
                    "arm": arm,
                    "obligation_count": count,
                    "obligation_type": field,
                    "expected_obligations": len(values),
                    "observed_obligations": len(observed),
                    "passed": passed,
                    "success_rate": _rate(passed, len(values)),
                    "conditional_success_rate": _rate(passed, len(observed)),
                    "by_interaction_class": by_interaction,
                }
            )
    return metrics


def _build_joint_metrics(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce joint success over all rows, including transport failures."""

    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["arm"]), int(row["obligation_count"]))].append(row)
    metrics: list[JsonDict] = []
    for (model_id, arm, count), cell_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        passed = sum(bool(row["joint_passed"]) for row in cell_rows)
        interval = bootstrap_interval(
            [float(bool(row["joint_passed"])) for row in cell_rows],
            seed=_group_seed("joint", model_id, arm, count),
            replicates=BOOTSTRAP_REPLICATES,
        )
        metrics.append(
            {
                "model_id": model_id,
                "model_family": MODEL_FAMILY_BY_ID.get(model_id, "unknown"),
                "arm": arm,
                "obligation_count": count,
                "rows": len(cell_rows),
                "joint_passed": passed,
                "joint_success_rate": _rate(passed, len(cell_rows)),
                "bootstrap_unit": "scenario_id",
                "ci95_low": interval["ci95_low"],
                "ci95_high": interval["ci95_high"],
                "by_interaction_class": _interaction_breakdown(cell_rows, "joint_passed"),
            }
        )
    return metrics


def _build_parse_metrics(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce strict transport failures without calling them semantic decay."""

    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["arm"]), int(row["obligation_count"]))].append(row)
    metrics: list[JsonDict] = []
    for (model_id, arm, count), cell_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        failures = [row for row in cell_rows if not row["parse_status"]["parsed"]]
        by_error = Counter(str(row["parse_status"]["error"]) for row in failures)
        by_interaction = {}
        for mode in ("independent", "interacting"):
            selected = [row for row in cell_rows if row["dependency_mode"] == mode]
            failed = sum(not row["parse_status"]["parsed"] for row in selected)
            by_interaction[mode] = {
                "rows": len(selected),
                "parse_failed": failed,
                "parse_failure_rate": _rate(failed, len(selected)),
            }
        metrics.append(
            {
                "model_id": model_id,
                "model_family": MODEL_FAMILY_BY_ID.get(model_id, "unknown"),
                "arm": arm,
                "obligation_count": count,
                "rows": len(cell_rows),
                "parse_failed": len(failures),
                "parse_failure_rate": _rate(len(failures), len(cell_rows)),
                "error_counts": dict(sorted(by_error.items())),
                "by_interaction_class": by_interaction,
            }
        )
    return metrics


def _interaction_pair_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    """Match independent and interacting templates by case and permutation."""

    match = re.fullmatch(r"template-oc\d+-t(\d\d)", str(row["template_id"]))
    if match is None:
        raise AuditError(f"invalid_template_id:{row['template_id']}")
    case_index = int(match.group(1)) % 5
    scenario_match = re.search(r"-p([012])$", str(row["scenario_id"]))
    if scenario_match is None:
        raise AuditError(f"invalid_scenario_id:{row['scenario_id']}")
    return (
        str(row["model_id"]),
        str(row["arm"]),
        int(row["obligation_count"]),
        case_index,
        scenario_match.group(1),
    )


def _build_interaction_penalties(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Contrast interaction classes at fixed model, arm, count, case, and order."""

    pairs: dict[tuple[str, str, int, int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = _interaction_pair_key(row)
        mode = str(row["dependency_mode"])
        if mode in pairs[key]:
            raise AuditError(f"duplicate_interaction_pair:{key}:{mode}")
        pairs[key][mode] = row
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for key, modes in pairs.items():
        if set(modes) != {"independent", "interacting"}:
            raise AuditError(f"incomplete_interaction_pair:{key}")
        delta = float(bool(modes["interacting"]["joint_passed"])) - float(
            bool(modes["independent"]["joint_passed"])
        )
        grouped[key[:3]].append(delta)
    results: list[JsonDict] = []
    for (model_id, arm, count), deltas in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        interval = bootstrap_interval(
            deltas,
            seed=_group_seed("interaction", model_id, arm, count),
            replicates=BOOTSTRAP_REPLICATES,
        )
        results.append(
            {
                "model_id": model_id,
                "model_family": MODEL_FAMILY_BY_ID.get(model_id, "unknown"),
                "arm": arm,
                "obligation_count": count,
                "effect_definition": "interacting_joint_minus_independent_joint",
                "bootstrap_unit": "matched_case_permutation",
                **interval,
            }
        )
    return results


def _build_saturation_curves(joint_metrics: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build count-ordered curves without pooling model families."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in joint_metrics:
        grouped[(str(row["model_id"]), str(row["arm"]))].append(row)
    curves: list[JsonDict] = []
    for (model_id, arm), metrics in sorted(grouped.items()):
        ordered = sorted(metrics, key=lambda row: int(row["obligation_count"]))
        counts = [int(row["obligation_count"]) for row in ordered]
        if counts != list(OBLIGATION_COUNTS):
            raise AuditError(f"invalid_count_order:{model_id}:{arm}:{counts}")
        rates = [float(row["joint_success_rate"]) for row in ordered]
        points = [
            {
                "obligation_count": row["obligation_count"],
                "rows": row["rows"],
                "joint_passed": row["joint_passed"],
                "joint_success_rate": row["joint_success_rate"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
            }
            for row in ordered
        ]
        curves.append(
            {
                "model_id": model_id,
                "model_family": MODEL_FAMILY_BY_ID.get(model_id, "unknown"),
                "arm": arm,
                "headline_pooling": "none",
                "saturation_definition": "joint success declines as obligation count increases",
                "points": points,
                "first_to_last_change": rates[-1] - rates[0],
                "declining_joint_success": rates[-1] < rates[0],
                "monotone_nonincreasing": all(
                    right <= left for left, right in zip(rates, rates[1:])
                ),
            }
        )
    return curves


def reduce_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build every required metric only from independently recomputed rows."""

    source_rows = [row for row in rows if row.get("row_type", "source") == "source"]
    joint = _build_joint_metrics(source_rows)
    pairs = join_prompt_arms(source_rows)
    return {
        "per_obligation_metrics": _build_per_obligation_metrics(source_rows),
        "joint_success_metrics": joint,
        "parse_failure_metrics": _build_parse_metrics(source_rows),
        "interaction_penalties": _build_interaction_penalties(source_rows),
        "saturation_curves": _build_saturation_curves(joint),
        "paired_arm_effects": build_paired_arm_effects(pairs),
    }


def metric_signature(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Return order-free sufficient statistics for permutation attacks."""

    counts: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        if row.get("row_type", "source") != "source":
            continue
        passed_fields = sum(
            value is True
            for result in row["obligation_results"].values()
            for value in result["fields"].values()
        )
        observed_fields = sum(
            isinstance(value, bool)
            for result in row["obligation_results"].values()
            for value in result["fields"].values()
        )
        key = (
            row["model_id"],
            row["arm"],
            row["obligation_count"],
            row["dependency_mode"],
            bool(row["parse_status"]["parsed"]),
            bool(row["joint_passed"]),
            passed_fields,
            observed_fields,
        )
        counts[key] += 1
    return [list(key) + [count] for key, count in sorted(counts.items())]


def permute_group_labels(
    rows: Sequence[Mapping[str, Any]], field: str
) -> tuple[list[JsonDict], dict[str, str]]:
    """Apply a bijective semantic label permutation for an invariance attack."""

    labels = sorted({str(row[field]) for row in rows})
    if len(labels) < 2:
        raise AuditError(f"not_enough_labels:{field}")
    rotated = labels[1:] + labels[:1]
    mapping = dict(zip(labels, rotated, strict=True))
    inverse = {new: old for old, new in mapping.items()}
    changed = deepcopy(list(rows))
    for row in changed:
        row[field] = mapping[str(row[field])]
    return changed, inverse


def canonicalized_metric_signature(
    rows: Sequence[Mapping[str, Any]], field: str, inverse: Mapping[str, str]
) -> list[list[Any]]:
    """Undo a label permutation before comparing sufficient statistics."""

    restored = deepcopy(list(rows))
    for row in restored:
        row[field] = inverse[str(row[field])]
    return metric_signature(restored)


def mask_model_labels(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Collapse model identity to prove that unpooled analysis needs labels."""

    changed = deepcopy(list(rows))
    for row in changed:
        row["model_id"] = "masked_model"
        row["model_family"] = "masked_family"
    return changed


def permute_action_labels(scenario: Mapping[str, Any], raw: bytes) -> tuple[JsonDict, bytes]:
    """Rename every action consistently so checker meaning stays unchanged."""

    parsed = parse_raw_output(raw)
    if not parsed["parsed"]:
        raise AuditError("action_label_attack_requires_parseable_raw")
    changed = deepcopy(dict(scenario))
    action_ids = sorted(row["action_id"] for row in changed["candidates"])
    mapping = {
        action_id: f"permuted-action-{index:03d}" for index, action_id in enumerate(action_ids)
    }
    for candidate in changed["candidates"]:
        candidate["action_id"] = mapping[candidate["action_id"]]
    changed["fail_closed_action_id"] = mapping[changed["fail_closed_action_id"]]
    changed["legal_action_ids"] = sorted(mapping[item] for item in changed["legal_action_ids"])
    changed["candidate_prompt_order"] = [
        mapping[item] for item in changed["candidate_prompt_order"]
    ]
    for obligation in changed["obligations"]:
        obligation["action"]["action_id"] = mapping[obligation["action"]["action_id"]]
        fallback = obligation["contract"]["fallback"]
        fallback["action_id"] = mapping[fallback["action_id"]]
    selected = [mapping[item] for item in parsed["selected_action_ids"]]
    return changed, canonical_bytes({"selected_action_ids": selected})


def mutate_checker_contract(scenario: Mapping[str, Any]) -> JsonDict:
    """Change authority truth so a checker mutation cannot pass silently."""

    changed = deepcopy(dict(scenario))
    changed["obligations"][0]["contract"]["authority"]["issuer"] = "mutated_authority"
    return changed


def run_attack_suite(
    fixture: Mapping[str, Any], corpus: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Run the eight required coverage, order, label, and checker attacks."""

    expected = expected_row_ids(fixture, corpus.get("MODEL_SPECS", []))
    base_signature = metric_signature(rows)
    deleted = coverage_report(list(rows)[1:], expected)
    duplicated = coverage_report([*rows, rows[0]], expected)
    model_rows, model_inverse = permute_group_labels(rows, "model_id")
    prompt_rows, prompt_inverse = permute_group_labels(rows, "arm")
    masked = coverage_report(mask_model_labels(rows), expected)

    scenario = fixture["scenarios"][0]
    legal_raw = canonical_bytes({"selected_action_ids": scenario["legal_action_ids"]})
    permuted_scenario, permuted_raw = permute_action_labels(scenario, legal_raw)
    action_labels_pass = (
        check_joint(scenario, legal_raw)["passed"]
        and check_joint(permuted_scenario, permuted_raw)["passed"]
    )

    passed_row = next((row for row in rows if row["joint_passed"]), None)
    checker_changed = False
    if passed_row is not None:
        scenario_index = {item["scenario_id"]: item for item in fixture["scenarios"]}
        source_scenario = scenario_index[passed_row["scenario_id"]]
        raw = base64.b64decode(passed_row["raw_output_bytes_b64"], validate=True)
        checker_changed = check_joint(source_scenario, raw) != check_joint(
            mutate_checker_contract(source_scenario), raw
        )

    rows_out = [
        {
            "row_type": "attack",
            "attack_id": "row_deletion",
            "passed": not deleted["valid"] and len(deleted["missing"]) == 1,
            "observed": {"missing_count": len(deleted["missing"])},
        },
        {
            "row_type": "attack",
            "attack_id": "row_duplicate",
            "passed": not duplicated["valid"] and len(duplicated["duplicates"]) == 1,
            "observed": {"duplicate_count": len(duplicated["duplicates"])},
        },
        {
            "row_type": "attack",
            "attack_id": "row_order",
            "passed": metric_signature(list(reversed(rows))) == base_signature,
            "observed": "order_free_metric_signature",
        },
        {
            "row_type": "attack",
            "attack_id": "action_label_permutation",
            "passed": bool(action_labels_pass),
            "observed": "joint_truth_preserved_after_bijection",
        },
        {
            "row_type": "attack",
            "attack_id": "model_label_permutation",
            "passed": canonicalized_metric_signature(model_rows, "model_id", model_inverse)
            == base_signature,
            "observed": "unpooled_cells_preserved_after_inverse_mapping",
        },
        {
            "row_type": "attack",
            "attack_id": "model_label_masking",
            "passed": not masked["valid"] and masked["model_identity_count"] == 1,
            "observed": {"model_identity_count": masked["model_identity_count"]},
        },
        {
            "row_type": "attack",
            "attack_id": "prompt_label_permutation",
            "passed": canonicalized_metric_signature(prompt_rows, "arm", prompt_inverse)
            == base_signature,
            "observed": "arm_cells_preserved_after_inverse_mapping",
        },
        {
            "row_type": "attack",
            "attack_id": "checker_mutation",
            "passed": checker_changed,
            "observed": "public_contract_mutation_changed_recomputed_truth",
        },
    ]
    return rows_out


def _target_cell_id(
    model_id: str, scenario_id: str, arm: str, obligation_id: str, field: str
) -> str:
    """Name one finite latent field-policy coordinate."""

    return "|".join((model_id, scenario_id, arm, obligation_id, field))


def _target_cells(
    fixture: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Enumerate the full finite policy target in deterministic order."""

    return sorted(
        _target_cell_id(
            str(spec["hub_id"]),
            str(scenario["scenario_id"]),
            arm,
            str(obligation["obligation_id"]),
            field,
        )
        for spec in model_specs
        for scenario in fixture.get("scenarios", [])
        for arm in ARMS
        for obligation in scenario["obligations"]
        for field in OBLIGATION_FIELDS
    )


def _observed_field_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    """Extract direct field observations only from strictly parsed rows."""

    observed: dict[str, bool] = {}
    for row in rows:
        if not row["parse_status"]["parsed"]:
            continue
        for obligation_id, result in row["obligation_results"].items():
            for field, value in result["fields"].items():
                if isinstance(value, bool):
                    observed[
                        _target_cell_id(
                            str(row["model_id"]),
                            str(row["scenario_id"]),
                            str(row["arm"]),
                            str(obligation_id),
                            str(field),
                        )
                    ] = value
    return observed


def _cell_class(cell: str, scenarios: Mapping[str, Mapping[str, Any]]) -> tuple[Any, ...]:
    """Group missing coordinates for compact but representative witnesses."""

    model_id, scenario_id, arm, _, field = cell.split("|", 4)
    scenario = scenarios[scenario_id]
    return (
        model_id,
        arm,
        scenario["obligation_count"],
        scenario["dependency_mode"],
        field,
    )


def validate_collision_witness(witness: Mapping[str, Any], observed_cells: set[str]) -> bool:
    """Check that one policy pair is indistinguishable until its named cell is added."""

    cell = witness.get("differing_target_cell")
    return (
        isinstance(cell, str)
        and cell not in observed_cells
        and witness.get("observed_signature_left") == witness.get("observed_signature_right")
        and witness.get("target_value_left") != witness.get("target_value_right")
        and witness.get("smallest_added_cells") == [cell]
    )


def audit_identifiability(
    fixture: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, JsonDict, list[JsonDict]]:
    """Enumerate binary completions and construct missing-cell collisions."""

    targets = _target_cells(fixture, model_specs)
    observed_values = _observed_field_values(rows)
    observed_cells = sorted(set(targets) & set(observed_values))
    missing = sorted(set(targets) - set(observed_cells))
    signature = sha256_json([[cell, observed_values[cell]] for cell in observed_cells])
    scenarios = {row["scenario_id"]: row for row in fixture.get("scenarios", [])}

    representatives: dict[tuple[Any, ...], str] = {}
    for cell in missing:
        representatives.setdefault(_cell_class(cell, scenarios), cell)
    witnesses = [
        {
            "witness_id": f"missing-cell-class-{index:03d}",
            "policy_id_left": "compatible_completion_cell_zero",
            "policy_id_right": "compatible_completion_cell_one",
            "observed_signature_left": signature,
            "observed_signature_right": signature,
            "differing_target_cell": cell,
            "target_value_left": 0,
            "target_value_right": 1,
            "target_rate_delta": 1 / len(targets),
            "smallest_added_cells": [cell],
            "missing_cell_class": {
                "model_id": key[0],
                "arm": key[1],
                "obligation_count": key[2],
                "interaction_class": key[3],
                "obligation_type": key[4],
            },
        }
        for index, (key, cell) in enumerate(sorted(representatives.items()))
    ]
    disposition = "separated" if not missing else "not_separated"
    result = {
        "disposition": disposition,
        "target_policy": "binary operational-field preservation on every finite fixture cell",
        "target_cell_count": len(targets),
        "observed_cell_count": len(observed_cells),
        "missing_cell_count": len(missing),
        "compatible_policy_class": "all binary completions of unobserved target cells",
        "compatible_policy_count": "1" if not missing else f"2^{len(missing)}",
        "enumeration_method": "symbolic Cartesian product over each missing binary coordinate",
        "observed_signature_sha256": signature,
    }
    minimum = {
        "policy_class": "unrestricted binary finite-field policies",
        "minimum_support_size": len(targets),
        "current_support_size": len(observed_cells),
        "smallest_added_cell_count": len(missing),
        "observed_cells": observed_cells,
        "smallest_added_cells": missing,
        "target_support_sha256": sha256_json(targets),
        "minimality_certificate": (
            "Every omitted coordinate admits two compatible policies that agree on all observed "
            "cells and differ only on that coordinate. Therefore each target cell is necessary."
        ),
    }
    return result, minimum, witnesses


def _raw_receipt_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check complete prompt and raw bytes before any output is reduced."""

    errors: list[str] = []
    for row in rows:
        row_id = str(row.get("row_id"))
        _, raw_error = _decode_receipt(row)
        if raw_error is not None:
            errors.append(f"{row_id}:{raw_error}")
        prompt_b64 = row.get("prompt_bytes_b64")
        try:
            prompt = base64.b64decode(prompt_b64, validate=True)
        except (TypeError, ValueError, base64.binascii.Error):
            errors.append(f"{row_id}:invalid_prompt_base64")
            continue
        if len(prompt) != row.get("prompt_byte_length"):
            errors.append(f"{row_id}:prompt_length_mismatch")
        if sha256_bytes(prompt) != row.get("prompt_sha256"):
            errors.append(f"{row_id}:prompt_hash_mismatch")
    return errors


def _budget_receipt(corpus: Mapping[str, Any]) -> JsonDict:
    """Confirm every source row used the same frozen decode budget."""

    rows = corpus.get("per_unit_rows", [])
    settings = {canonical_bytes(row.get("decode_settings")).decode() for row in rows}
    allowed = {row.get("token_counts", {}).get("allowed_generated") for row in rows}
    receipt = corpus.get("budget_parity", {})
    passed = (
        receipt.get("passed") is True
        and receipt.get("tokens_allowed_equal") is True
        and len(settings) == 1
        and len(allowed) == 1
        and None not in allowed
    )
    return {
        "passed": passed,
        "decode_setting_count": len(settings),
        "allowed_token_count": len(allowed),
        "producer_budget_parity": receipt.get("passed"),
    }


def _process_receipt_errors(corpus: Mapping[str, Any]) -> list[str]:
    """Bind every row session to one authentic process and clean teardown."""

    receipts = corpus.get("process_receipts", [])
    corpus_receipts = {
        row.get("session_id"): row for row in receipts if row.get("purpose") == "corpus"
    }
    errors: list[str] = []
    used_sessions = set()
    for row in corpus.get("per_unit_rows", []):
        identity = row.get("process_identity", {})
        session = identity.get("session_id")
        used_sessions.add(session)
        receipt = corpus_receipts.get(session)
        if receipt is None:
            errors.append(f"missing_process_receipt:{session}")
            continue
        if receipt.get("model_id") != row.get("model_id"):
            errors.append(f"process_model_mismatch:{session}")
    for session in sorted(used_sessions, key=str):
        receipt = corpus_receipts.get(session, {})
        complete = (
            receipt.get("authentic") is True
            and receipt.get("cuda_offload") is True
            and receipt.get("lease_owned") is True
            and receipt.get("lease_released") is True
            and receipt.get("teardown_complete") is True
            and receipt.get("process_absent_after_exit") is True
            and receipt.get("error") is None
            and bool(receipt.get("first_token_b64"))
            and bool(receipt.get("final_token_b64"))
        )
        if not complete:
            errors.append(f"incomplete_process_or_teardown:{session}")
    canary_models = {
        row.get("model_id")
        for row in receipts
        if row.get("purpose") == "canary"
        and row.get("authentic") is True
        and row.get("teardown_complete") is True
    }
    if canary_models != set(EXPECTED_MODEL_HASHES):
        errors.append("incomplete_model_canaries")
    return errors


def evaluate_preconditions(fixture_bytes: bytes, corpus_bytes: bytes) -> list[JsonDict]:
    """Evaluate every source, coverage, byte, budget, process, and teardown gate."""

    fixture = _json_object(fixture_bytes)
    corpus = _json_object(corpus_bytes)
    model_hashes = {
        row.get("hub_id"): row.get("model_sha256") for row in corpus.get("MODEL_SPECS", [])
    }
    expected = expected_row_ids(fixture, corpus.get("MODEL_SPECS", []))
    coverage = coverage_report(corpus.get("per_unit_rows", []), expected)
    raw_errors = _raw_receipt_errors(corpus.get("per_unit_rows", []))
    process_errors = _process_receipt_errors(corpus)
    fixture_receipt = corpus.get("fixture_receipt", {})
    checkpoint = corpus.get("checkpoint_manifest", {})
    return [
        _check("fixture_file_sha256", EXPECTED_FIXTURE_SHA256, sha256_bytes(fixture_bytes)),
        _check("corpus_file_sha256", EXPECTED_CORPUS_SHA256, sha256_bytes(corpus_bytes)),
        _check("fixture_schema", FIXTURE_SCHEMA, fixture.get("schema")),
        _check("corpus_schema", CORPUS_SCHEMA, corpus.get("schema")),
        _check(
            "operational_saturation_fixture_ready",
            True,
            fixture.get("operational_saturation_fixture_ready"),
        ),
        _check(
            "operational_saturation_corpus_ready",
            True,
            corpus.get("operational_saturation_corpus_ready"),
        ),
        _check("fixture_scenario_count", 150, len(fixture.get("scenarios", []))),
        _check(
            "fixture_receipt_hash", sha256_bytes(fixture_bytes), fixture_receipt.get("file_sha256")
        ),
        _check("model_hashes", EXPECTED_MODEL_HASHES, model_hashes),
        _check("unique_source_rows", True, coverage["valid"]),
        _check("complete_raw_bytes", [], raw_errors),
        _check("equal_budgets", True, _budget_receipt(corpus)["passed"]),
        _check("process_receipts_and_teardown", [], process_errors),
        _check("checkpoint_valid", True, checkpoint.get("valid")),
        _check("checkpoint_row_count", 900, checkpoint.get("completed_row_count")),
    ]


def _source_artifact_hashes(fixture_bytes: bytes, corpus_bytes: bytes) -> JsonDict:
    """Record exact source and model identities used by this audit."""

    fixture = _json_object(fixture_bytes)
    corpus = _json_object(corpus_bytes)
    return {
        "fixture": {
            "path": str(FIXTURE_PATH),
            "file_sha256": sha256_bytes(fixture_bytes),
            "schema": fixture.get("schema"),
            "reproducibility_checksum": fixture.get("reproducibility_checksum"),
        },
        "corpus": {
            "path": str(CORPUS_PATH),
            "file_sha256": sha256_bytes(corpus_bytes),
            "schema": corpus.get("schema"),
            "reproducibility_checksum": corpus.get("reproducibility_checksum"),
        },
        "models": {
            row.get("hub_id"): {
                "model_sha256": row.get("model_sha256"),
                "revision": row.get("revision"),
                "filename": row.get("filename"),
            }
            for row in corpus.get("MODEL_SPECS", [])
        },
    }


def _file_hash(path: Path) -> str | None:
    """Hash owned code when it exists and expose a missing file."""

    full = path if path.is_absolute() else REPO_ROOT / path
    return sha256_bytes(full.read_bytes()) if full.is_file() else None


def _implementation_hashes() -> JsonDict:
    """Bind the independent module and command wrapper identities."""

    return {
        "module": {"path": str(MODULE_PATH), "sha256": _file_hash(MODULE_PATH)},
        "wrapper": {"path": str(WRAPPER_PATH), "sha256": _file_hash(WRAPPER_PATH)},
    }


def _methodology(run_date: str) -> JsonDict:
    """State the frozen estimands, units, and command in the artifact."""

    return {
        "audit_authority": "fresh raw-byte parse and Exp6832 public-contract reimplementation",
        "producer_aggregate_authority": False,
        "parser_repair": False,
        "joint_saturation_definition": (
            "joint success declines as obligation count increases; parse failure is separate"
        ),
        "paired_arm_effect": "typed joint success minus compressed joint success",
        "bootstrap_unit": "scenario_id",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "headline_pooling": "none across model families",
        "identifiability_policy_class": "unrestricted binary finite-field policies",
        "audit_command": (
            ".venv/bin/python scripts/experiments/"
            "experiment_6834_operational_saturation_identifiability_audit.py "
            f"--date {run_date}"
        ),
        "producer_imports_allowed": False,
        "llm_calls": 0,
    }


def _empty_payload(
    *, run_date: str, duration_s: float, fixture_bytes: bytes, corpus_bytes: bytes
) -> JsonDict:
    """Create all terminal fields before a blocked or complete branch."""

    return {
        "schema": ARTIFACT_SCHEMA,
        "experiment_id": "exp6834-operational-saturation-identifiability-audit",
        "run_date": run_date,
        "status": "building",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": _source_artifact_hashes(fixture_bytes, corpus_bytes),
        "implementation_hashes": _implementation_hashes(),
        "methodology": _methodology(run_date),
        "independent_parser_id": INDEPENDENT_PARSER_ID,
        "independent_reducer_id": INDEPENDENT_REDUCER_ID,
        "per_unit_rows": [],
        "row_coverage": {
            "expected": 900,
            "observed": 0,
            "observed_unique": 0,
            "model_identity_count": 0,
            "missing": [],
            "unexpected": [],
            "duplicates": [],
            "valid": False,
        },
        "per_obligation_metrics": [],
        "joint_success_metrics": [],
        "parse_failure_metrics": [],
        "interaction_penalties": [],
        "saturation_curves": [],
        "paired_arm_effects": [],
        "identifiability_result": {"disposition": "not_run"},
        "minimum_identifying_support": {},
        "collision_witnesses": [],
        "attack_results": [],
        "operational_saturation_audit_complete": False,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_operational_saturation_identifiability_audit",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind deterministic content while excluding measured wall time and itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def build_artifact(
    *, run_date: str, duration_s: float, fixture_bytes: bytes, corpus_bytes: bytes
) -> JsonDict:
    """Build a blocked artifact or the complete independent cold audit."""

    artifact = _empty_payload(
        run_date=run_date,
        duration_s=duration_s,
        fixture_bytes=fixture_bytes,
        corpus_bytes=corpus_bytes,
    )
    checks = evaluate_preconditions(fixture_bytes, corpus_bytes)
    artifact["preconditions_checked"] = checks
    failed = next((row for row in checks if not row["passed"]), None)
    if failed is not None:
        artifact["status"] = "complete_blocked_operational_saturation_identifiability_audit"
        artifact["gate_check_summary"] = {
            "passed": False,
            "failed_check": failed["check"],
            "expected": failed["expected"],
            "observed": failed["observed"],
        }
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    fixture = _json_object(fixture_bytes)
    corpus = _json_object(corpus_bytes)
    rows = recompute_source_rows(fixture, corpus["per_unit_rows"])
    expected = expected_row_ids(fixture, corpus["MODEL_SPECS"])
    coverage = coverage_report(rows, expected)
    metrics = reduce_metrics(rows)
    identifiability, minimum, witnesses = audit_identifiability(
        fixture, corpus["MODEL_SPECS"], rows
    )
    attacks = run_attack_suite(fixture, corpus, rows)
    complete = (
        coverage["valid"]
        and len(rows) == 900
        and all(row["raw_receipt_error"] is None for row in rows)
        and all(row["passed"] for row in attacks)
        and identifiability["disposition"] in {"separated", "not_separated"}
    )
    any_decline = any(row["declining_joint_success"] for row in metrics["saturation_curves"])
    artifact.update(metrics)
    artifact.update(
        {
            "status": "complete",
            "per_unit_rows": [*rows, *attacks],
            "row_coverage": coverage,
            "identifiability_result": identifiability,
            "minimum_identifying_support": minimum,
            "collision_witnesses": witnesses,
            "attack_results": attacks,
            "operational_saturation_audit_complete": complete,
            "gate_check_summary": {
                "passed": complete,
                "failed_check": None if complete else "cold_audit_completeness",
                "expected": {
                    "source_rows": 900,
                    "coverage_valid": True,
                    "all_attacks_passed": True,
                    "identifiability_disposition": ["separated", "not_separated"],
                },
                "observed": {
                    "source_rows": len(rows),
                    "coverage_valid": coverage["valid"],
                    "all_attacks_passed": all(row["passed"] for row in attacks),
                    "identifiability_disposition": identifiability["disposition"],
                },
            },
            "verdict_class": (
                "null"
                if identifiability["disposition"] == "not_separated" or not any_decline
                else "positive"
            ),
            "honest_verdict": (
                "complete_null_operational_field_preservation_not_identified"
                if identifiability["disposition"] == "not_separated"
                else "complete_positive_operational_saturation_identified"
                if any_decline
                else "complete_null_no_operational_saturation_decay"
            ),
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute terminal invariants so the writer cannot bless itself silently."""

    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        errors.append("required artifact fields are missing")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field principles do not cover every top-level field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    if artifact.get("independent_parser_id") != INDEPENDENT_PARSER_ID:
        errors.append("independent parser identity mismatch")
    if artifact.get("independent_reducer_id") != INDEPENDENT_REDUCER_ID:
        errors.append("independent reducer identity mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict class is outside the closed set")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest verdict lacks a terminal prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")

    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        if (
            artifact.get("status")
            != "complete_blocked_operational_saturation_identifiability_audit"
        ):
            errors.append("blocked status mismatch")
        if artifact.get("per_unit_rows"):
            errors.append("blocked artifact reduced source rows")
        if artifact.get("gate_check_summary", {}).get("failed_check") is None:
            errors.append("blocked artifact lacks failed check")
        return errors

    source_rows = [
        row for row in artifact.get("per_unit_rows", []) if row.get("row_type") == "source"
    ]
    attack_rows = [
        row for row in artifact.get("per_unit_rows", []) if row.get("row_type") == "attack"
    ]
    if not artifact.get("operational_saturation_audit_complete"):
        errors.append("complete artifact is not marked complete")
    if len(source_rows) != 900:
        errors.append("complete artifact source-row count mismatch")
    if artifact.get("row_coverage", {}).get("valid") is not True:
        errors.append("complete artifact row coverage is invalid")
    if len(attack_rows) != len(ATTACK_IDS) or not all(row.get("passed") for row in attack_rows):
        errors.append("complete artifact attack rows are invalid")
    if artifact.get("attack_results") != attack_rows:
        errors.append("attack results do not match per-unit attack rows")
    if artifact.get("identifiability_result", {}).get("disposition") not in {
        "separated",
        "not_separated",
    }:
        errors.append("identifiability disposition is not terminal")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write through a same-directory temporary file and replace atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(artifact, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)


def _read_source(path: Path) -> bytes:
    """Turn an unreadable source into bytes that fail the normal hash gate."""

    full = path if path.is_absolute() else REPO_ROOT / path
    try:
        return full.read_bytes()
    except OSError as exc:
        return canonical_bytes({"read_error": type(exc).__name__, "path": str(full)})


def main(argv: list[str] | None = None) -> int:
    """Run the fresh-process deterministic audit and write its terminal artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    fixture_bytes = _read_source(FIXTURE_PATH)
    corpus_bytes = _read_source(CORPUS_PATH)
    artifact = build_artifact(
        run_date=args.date,
        duration_s=0.0,
        fixture_bytes=fixture_bytes,
        corpus_bytes=corpus_bytes,
    )
    artifact["duration_s"] = round(max(time.perf_counter() - started, 0.000001), 6)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"validation_errors": errors}, indent=2), file=sys.stderr)
        return 1
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    write_artifact(output, artifact)
    print(
        json.dumps(
            {
                "artifact": str(output),
                "complete": artifact["operational_saturation_audit_complete"],
                "identifiability": artifact["identifiability_result"]["disposition"],
                "verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper owns the command surface.
    raise SystemExit(main())
