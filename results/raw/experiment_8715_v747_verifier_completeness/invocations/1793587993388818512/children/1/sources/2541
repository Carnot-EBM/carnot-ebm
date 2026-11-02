"""Freeze V598 evidence with an atom-level failure taxonomy.

Spec refs: REQ-CONSTRAINT-6835 and SCENARIO-CONSTRAINT-6835-*.

The source artifacts already contain live model bytes and a terminal null
identifiability audit. This module does not repair those bytes. It only
replays them with local deterministic code and preserves the null boundary.
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
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6835_v598_terminal_evidence_freeze.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6835_v598_terminal_evidence_freeze.py")
OUTPUT_PATH = Path("results/experiment_6835_v598_terminal_evidence_freeze.json")

SOURCE_ARTIFACT_PATHS: dict[str, Path] = {
    "exp6831": Path("results/experiment_6831_v597_evidence_admissibility_contract.json"),
    "exp6832": Path("results/experiment_6832_operational_obligation_saturation_fixture.json"),
    "exp6833": Path("results/experiment_6833_sota_operational_obligation_saturation_corpus.json"),
    "exp6834": Path("results/experiment_6834_operational_saturation_identifiability_audit.json"),
}
EXPECTED_SOURCE_HASHES = {
    "exp6831": "sha256:52783c6bc08656c80c658c9027b9f47574024d24d0e716b464822107c153b4f9",
    "exp6832": "sha256:05c88b1cc075fc789763e155df1ab67fa5ea5d3845851aea4d4d83f24027993f",
    "exp6833": "sha256:05cdd74188f48163a9e9e875ad4fda2389a07e1b4aa6c060590b25ef772ee6c5",
    "exp6834": "sha256:23f03b747c9b72a5b2caf2342d64faa2795a5e03d1199353345964a956cb42c0",
}
EXPECTED_SCHEMAS = {
    "exp6831": "carnot.experiment_6831.v597_evidence_admissibility_contract.v1",
    "exp6832": "carnot.experiment_6832.operational_obligation_saturation_fixture.v1",
    "exp6833": "carnot.experiment_6833.sota_operational_obligation_saturation_corpus.v1",
    "exp6834": "carnot.experiment_6834.operational_saturation_identifiability_audit.v1",
}

ARTIFACT_SCHEMA = "carnot.experiment_6835.v598_terminal_evidence_freeze.v1"
INFERENCE_SUBSTRATE = "deterministic CPU replay; no LLM calls"
RANDOM_SEED = 6835
ARMS = ("typed", "compressed")
OBLIGATION_FIELDS = (
    "prerequisite",
    "authority",
    "fallback",
    "execution_consequence",
    "priority",
)
PRIORITY_ORDER = ("hard", "binding", "soft")
MODEL_FAMILY_BY_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "qwen36",
    "unsloth/gemma-4-31B-it-GGUF": "gemma31_dense",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "gemma26_moe",
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
    "reproducibility_checksum",
    "rows",
    "failure_taxonomy",
    "observed_target_cells",
    "missing_target_cells",
    "collision_witnesses",
    "compatible_policy_lower_bound",
    "source_null_preserved",
    "obligation_failure_taxonomy_complete_score",
    "v598_evidence_root_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "The schema fixes the artifact contract for downstream readers.",
    "experiment_id": "The identifier binds the record to Exp6835.",
    "run_date": "The date separates this freeze from later rebuilds.",
    "status": "The status separates a complete freeze from a gate stop.",
    "field_principles": "Each top-level field states why it exists.",
    "preconditions_checked": "The checks stop source drift before row replay.",
    "inference_substrate": "The substrate states deterministic CPU replay and no LLM.",
    "duration_s": "Wall time exposes incomplete or interrupted execution.",
    "random_seed": "The seed records that no stochastic inference was used here.",
    "source_artifact_hashes": "Source hashes bind the terminal V597 evidence bytes.",
    "implementation_hashes": "Code hashes bind this parser and command wrapper.",
    "methodology": "The method records the no-import and no-repair boundary.",
    "reproducibility_checksum": "The checksum binds deterministic content.",
    "rows": "One row per source unit and obligation cell supports the taxonomy.",
    "failure_taxonomy": "The taxonomy separates transport and semantic failures.",
    "observed_target_cells": "Observed cells quantify what model bytes reveal.",
    "missing_target_cells": "Missing cells quantify unidentified field support.",
    "collision_witnesses": "Witnesses preserve the Exp6834 non-identification proof.",
    "compatible_policy_lower_bound": "The bound states how many policies remain possible.",
    "source_null_preserved": "This field keeps the Exp6834 null terminal.",
    "obligation_failure_taxonomy_complete_score": "This is integrity readiness, not science.",
    "v598_evidence_root_ready_score": "This is evidence-root readiness, not a positive result.",
    "gate_check_summary": "The summary names the exact failed or completed gate.",
    "verifier_is_oracle": "False states that this replay is not an oracle.",
    "verdict_class": "A closed class keeps the terminal disposition machine-readable.",
    "honest_verdict": "The terminal prefix states the row-supported boundary.",
}


class FreezeError(ValueError):
    """Expose one stable error for malformed finite evidence."""


def canonical_bytes(value: Any) -> bytes:
    """Encode JSON once so byte checks use a single canonical form."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed SHA-256 digest for stable artifact identities."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value through canonical bytes."""

    return sha256_bytes(canonical_bytes(value))


def _json_object(raw: bytes) -> JsonDict:
    """Return an object, or an empty object when a source cannot be decoded."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record exact expected and observed values for gate replay."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def check_by_name(checks: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    """Find one named precondition check in tests and validators."""

    for check in checks:
        if check.get("check") == name:
            return check
    raise FreezeError(f"missing_check:{name}")


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
    """Order conflicts from public priority and authority fields."""

    contract = obligation["contract"]
    priority = contract["priority"]
    return (
        PRIORITY_ORDER.index(priority["class"]),
        int(contract["authority"]["order"]),
        -int(priority["weight"]),
        str(obligation["obligation_id"]),
    )


def resolve_scenario(scenario: Mapping[str, Any]) -> JsonDict:
    """Resolve the five-field public contract with local deterministic code."""

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
    """Index candidate actions by identifier for local field checks."""

    return {row["action_id"]: row for row in scenario["candidates"]}


def _expected_obligation(
    scenario: Mapping[str, Any], obligation: Mapping[str, Any]
) -> tuple[JsonDict, Mapping[str, Any] | None]:
    """Return one resolved obligation and its expected selected action."""

    resolution = resolve_scenario(scenario)["obligations"][obligation["obligation_id"]]
    return resolution, _action_index(scenario).get(resolution["action_id"])


def _field_decisions(
    scenario: Mapping[str, Any], selected: set[str], obligation: Mapping[str, Any]
) -> dict[str, bool]:
    """Evaluate all five public fields for one obligation."""

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
    """Check every field for one obligation identifier."""

    obligations = {row["obligation_id"]: row for row in scenario["obligations"]}
    if obligation_id not in obligations:
        return {"fields": {field: False for field in OBLIGATION_FIELDS}, "passed": False}
    fields = _field_decisions(scenario, set(selected_action_ids), obligations[obligation_id])
    return {"fields": fields, "passed": all(fields.values())}


def check_joint(scenario: Mapping[str, Any], raw: bytes) -> JsonDict:
    """Check the whole selected action set after strict local parsing."""

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


def _target_cell_id(
    model_id: str, scenario_id: str, arm: str, obligation_id: str, field: str
) -> str:
    """Name one finite target field cell."""

    return "|".join((model_id, scenario_id, arm, obligation_id, field))


def _target_cells(
    fixture: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Enumerate the full V597 binary field-policy target."""

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


def _expected_row_ids(
    fixture: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Build the complete model, scenario, and prompt-arm roster."""

    return sorted(
        f"{spec['hub_id']}|{scenario['scenario_id']}|{arm}"
        for spec in model_specs
        for scenario in fixture.get("scenarios", [])
        for arm in ARMS
    )


def _row_identity(row: Mapping[str, Any]) -> str:
    """Rebuild an identity from labels instead of trusting row order."""

    return f"{row.get('model_id')}|{row.get('scenario_id')}|{row.get('arm')}"


def _identity_report(observed: Sequence[str], expected: Sequence[str]) -> JsonDict:
    """Report missing, duplicate, and unexpected finite identities."""

    counts = Counter(observed)
    expected_set = set(expected)
    observed_set = set(observed)
    duplicates = sorted(identity for identity, count in counts.items() if count > 1)
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)
    return {
        "row_count": len(observed),
        "unique_count": len(observed_set),
        "expected_count": len(expected),
        "missing_count": len(missing),
        "duplicate_count": len(duplicates),
        "unexpected_count": len(unexpected),
        "missing": missing[:10],
        "duplicates": duplicates[:10],
        "unexpected": unexpected[:10],
    }


def _decode_receipt(row: Mapping[str, Any]) -> tuple[bytes, str | None]:
    """Decode one raw-output receipt and retain a stable failure code."""

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


def _raw_receipt_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check byte receipts without treating invalid JSON as source drift."""

    errors: list[str] = []
    for row in rows:
        _, error = _decode_receipt(row)
        if error is not None:
            errors.append(f"{row.get('row_id')}:{error}")
    return errors


def _recorded_source_receipts(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Check that terminal artifacts carry their own source hash receipts."""

    missing: list[str] = []
    for name in ("exp6831", "exp6832", "exp6834"):
        receipt = payloads.get(name, {}).get("implementation_hashes")
        if not _receipt_has_paths_and_hashes(receipt):
            missing.append(name)
    if not _receipt_has_paths_and_hashes(payloads.get("exp6833", {}).get("code_receipts")):
        missing.append("exp6833")
    return {"missing_receipts": missing, "artifact_count": len(payloads)}


def _receipt_has_paths_and_hashes(receipt: Any) -> bool:
    """Return true when each recorded source row has a path and digest."""

    return isinstance(receipt, dict) and all(
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and isinstance(value.get("sha256"), str)
        and value["sha256"].startswith("sha256:")
        for value in receipt.values()
    )


def _payloads(source_bytes: Mapping[str, bytes]) -> dict[str, JsonDict]:
    """Decode all known source payloads with absent keys as empty objects."""

    return {name: _json_object(source_bytes.get(name, b"")) for name in SOURCE_ARTIFACT_PATHS}


def _source_hashes(source_bytes: Mapping[str, bytes]) -> dict[str, str]:
    """Hash all supplied source bytes with missing entries made explicit."""

    return {name: sha256_bytes(source_bytes.get(name, b"")) for name in SOURCE_ARTIFACT_PATHS}


def evaluate_preconditions(source_bytes: Mapping[str, bytes]) -> list[JsonDict]:
    """Evaluate all source, identity, receipt, and terminal-null gates."""

    payloads = _payloads(source_bytes)
    hashes = _source_hashes(source_bytes)
    fixture = payloads["exp6832"]
    corpus = payloads["exp6833"]
    audit = payloads["exp6834"]
    scenarios = fixture.get("scenarios", [])
    model_specs = corpus.get("MODEL_SPECS", [])
    expected_rows = _expected_row_ids(fixture, model_specs)
    observed_rows = [_row_identity(row) for row in corpus.get("per_unit_rows", [])]
    row_report = _identity_report(observed_rows, expected_rows)
    scenario_ids = [str(row.get("scenario_id")) for row in scenarios]
    scenario_report = _identity_report(scenario_ids, sorted(set(scenario_ids)))
    receipts = _recorded_source_receipts(payloads)
    identifiability = audit.get("identifiability_result", {})
    minimum = audit.get("minimum_identifying_support", {})

    checks: list[JsonDict] = []
    for name in SOURCE_ARTIFACT_PATHS:
        checks.append(_check(f"{name}_file_sha256", EXPECTED_SOURCE_HASHES[name], hashes[name]))
        checks.append(
            _check(f"{name}_schema", EXPECTED_SCHEMAS[name], payloads[name].get("schema"))
        )
    checks.extend(
        [
            _check(
                "exp6831_v597_contract_ready", True, payloads["exp6831"].get("v597_contract_ready")
            ),
            _check(
                "exp6832_fixture_ready",
                True,
                fixture.get("operational_saturation_fixture_ready"),
            ),
            _check(
                "exp6833_corpus_ready",
                True,
                corpus.get("operational_saturation_corpus_ready"),
            ),
            _check(
                "exp6834_audit_complete",
                True,
                audit.get("operational_saturation_audit_complete"),
            ),
            _check(
                "exp6832_unique_scenario_identities",
                {
                    "row_count": 150,
                    "unique_count": 150,
                    "expected_count": 150,
                    "missing_count": 0,
                    "duplicate_count": 0,
                    "unexpected_count": 0,
                    "missing": [],
                    "duplicates": [],
                    "unexpected": [],
                },
                scenario_report,
            ),
            _check(
                "exp6833_unique_row_identities",
                {
                    "row_count": 900,
                    "unique_count": 900,
                    "expected_count": 900,
                    "missing_count": 0,
                    "duplicate_count": 0,
                    "unexpected_count": 0,
                    "missing": [],
                    "duplicates": [],
                    "unexpected": [],
                },
                row_report,
            ),
            _check(
                "exp6833_raw_receipts_stable",
                [],
                _raw_receipt_errors(corpus.get("per_unit_rows", [])),
            ),
            _check(
                "owned_source_hash_receipts_present",
                {"missing_receipts": [], "artifact_count": 4},
                receipts,
            ),
            _check(
                "exp6834_terminal_null_preserved",
                {"verdict_class": "null", "disposition": "not_separated"},
                {
                    "verdict_class": audit.get("verdict_class"),
                    "disposition": identifiability.get("disposition"),
                },
            ),
            _check(
                "exp6834_target_support_accounting",
                {
                    "target_cell_count": 18900,
                    "observed_cell_count": 395,
                    "missing_cell_count": 18505,
                    "current_support_size": 395,
                    "smallest_added_cell_count": 18505,
                },
                {
                    "target_cell_count": identifiability.get("target_cell_count"),
                    "observed_cell_count": identifiability.get("observed_cell_count"),
                    "missing_cell_count": identifiability.get("missing_cell_count"),
                    "current_support_size": minimum.get("current_support_size"),
                    "smallest_added_cell_count": minimum.get("smallest_added_cell_count"),
                },
            ),
        ]
    )
    return checks


def _contradicting_actions(
    scenario: Mapping[str, Any],
    obligation: Mapping[str, Any],
    selected: set[str],
    expected_action_id: str,
) -> list[str]:
    """Find selected actions that conflict with this obligation's resource."""

    actions = _action_index(scenario)
    target = actions[obligation["action"]["action_id"]]
    selected_candidates = [
        (item, actions.get(item)) for item in selected if item != expected_action_id
    ]
    conflicting: list[str] = []
    for action_id, action in selected_candidates:
        if action is None:
            conflicting.append(action_id)
            continue
        if action.get("resource") == target["resource"]:
            conflicting.append(action_id)
    return sorted(conflicting)


def _failure_class(row: Mapping[str, Any]) -> str:
    """Collapse booleans into one primary class without losing the booleans."""

    if row["protocol_failure"]:
        return "protocol_failure"
    if row["omission"] and row["contradiction"]:
        return "omission_and_contradiction"
    if row["contradiction"]:
        return "contradiction"
    if row["omission"]:
        return "omission"
    if row["atom_pass"]:
        return "atom_pass"
    return "joint_semantic_failure"


def classify_source_row(scenario: Mapping[str, Any], source: Mapping[str, Any]) -> list[JsonDict]:
    """Emit one obligation-cell row for one Exp6833 source unit."""

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
    selected = set(checked["selected_action_ids"])
    resolutions = resolve_scenario(scenario)["obligations"]
    rows: list[JsonDict] = []
    model_id = str(source.get("model_id"))
    source_row_id = str(source.get("row_id"))
    for index, obligation in enumerate(scenario["obligations"]):
        obligation_id = str(obligation["obligation_id"])
        resolution = resolutions[obligation_id]
        expected_action_id = str(resolution["action_id"])
        target_cell_ids = [
            _target_cell_id(
                model_id,
                str(scenario["scenario_id"]),
                str(source.get("arm")),
                obligation_id,
                field,
            )
            for field in OBLIGATION_FIELDS
        ]
        if checked["parsed"]:
            result = checked["obligation_checks"][obligation_id]
            fields = dict(result["fields"])
            protocol_failure = False
            omitted = expected_action_id not in selected
            contradictory = _contradicting_actions(
                scenario, obligation, selected, expected_action_id
            )
            observed_cells = target_cell_ids
            missing_cells: list[str] = []
            atom_pass = bool(result["passed"])
        else:
            fields = {field: None for field in OBLIGATION_FIELDS}
            protocol_failure = True
            omitted = False
            contradictory = []
            observed_cells = []
            missing_cells = target_cell_ids
            atom_pass = False
        row: JsonDict = {
            "row_type": "source_obligation_cell",
            "row_id": f"{source_row_id}|{obligation_id}",
            "source_row_id": source_row_id,
            "model_id": model_id,
            "model_family": MODEL_FAMILY_BY_ID.get(model_id, "unknown"),
            "arm": str(source.get("arm")),
            "scenario_id": str(scenario["scenario_id"]),
            "scenario_hash": scenario.get("scenario_hash"),
            "template_id": scenario.get("template_id"),
            "permutation_id": scenario.get("permutation_id"),
            "dependency_mode": scenario.get("dependency_mode"),
            "semantic_class": scenario.get("semantic_class"),
            "obligation_count": int(scenario["obligation_count"]),
            "obligation_index": index,
            "obligation_id": obligation_id,
            "expected_action_id": expected_action_id,
            "expected_disposition": resolution["disposition"],
            "target_action_id": obligation["action"]["action_id"],
            "raw_output_sha256": source.get("raw_output_sha256"),
            "raw_receipt_error": receipt_error,
            "parse_status": {
                "parsed": checked["parsed"],
                "error": checked["parse_error"],
            },
            "parse_error": checked["parse_error"],
            "selected_action_ids": checked["selected_action_ids"],
            "field_results": fields,
            "observed_target_cell_ids": observed_cells,
            "missing_target_cell_ids": missing_cells,
            "protocol_failure": protocol_failure,
            "omission": omitted,
            "contradiction": bool(contradictory),
            "contradicting_action_ids": contradictory,
            "atom_pass": atom_pass,
            "joint_pass": bool(checked["passed"]),
            "joint_semantic_failure": checked["parsed"] and not checked["passed"],
        }
        row["failure_class"] = _failure_class(row)
        rows.append(row)
    return rows


def build_rows(fixture: Mapping[str, Any], corpus: Mapping[str, Any]) -> list[JsonDict]:
    """Classify every Exp6833 source unit and obligation cell."""

    scenarios = {row["scenario_id"]: row for row in fixture.get("scenarios", [])}
    rows: list[JsonDict] = []
    for source in corpus.get("per_unit_rows", []):
        scenario_id = source.get("scenario_id")
        if scenario_id not in scenarios:
            raise FreezeError(f"unknown_scenario:{scenario_id}")
        rows.extend(classify_source_row(scenarios[scenario_id], source))
    return sorted(rows, key=lambda row: row["row_id"])


def _failure_taxonomy(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count protocol, omission, contradiction, atom, and joint outcomes."""

    source_units: dict[str, JsonDict] = {}
    field_pass = 0
    field_fail = 0
    for row in rows:
        source = source_units.setdefault(
            str(row["source_row_id"]),
            {
                "parsed": bool(row["parse_status"]["parsed"]),
                "protocol_failure": bool(row["protocol_failure"]),
                "joint_pass": bool(row["joint_pass"]),
            },
        )
        source["parsed"] = source["parsed"] or bool(row["parse_status"]["parsed"])
        source["protocol_failure"] = source["protocol_failure"] and bool(row["protocol_failure"])
        source["joint_pass"] = source["joint_pass"] or bool(row["joint_pass"])
        for value in row["field_results"].values():
            field_pass += value is True
            field_fail += value is False

    by_class = Counter(str(row["failure_class"]) for row in rows)
    return {
        "source_units": {
            "total": len(source_units),
            "parsed": sum(bool(row["parsed"]) for row in source_units.values()),
            "protocol_failure": sum(bool(row["protocol_failure"]) for row in source_units.values()),
            "joint_pass": sum(bool(row["joint_pass"]) for row in source_units.values()),
        },
        "obligation_cells": {
            "total": len(rows),
            "protocol_failure": sum(bool(row["protocol_failure"]) for row in rows),
            "omission": sum(bool(row["omission"]) for row in rows),
            "contradiction": sum(bool(row["contradiction"]) for row in rows),
            "atom_pass": sum(bool(row["atom_pass"]) for row in rows),
            "joint_pass": sum(bool(row["joint_pass"]) for row in rows),
            "joint_semantic_failure": sum(bool(row["joint_semantic_failure"]) for row in rows),
            "by_failure_class": dict(sorted(by_class.items())),
        },
        "field_cells": {
            "observed": field_pass + field_fail,
            "passed": field_pass,
            "failed": field_fail,
            "missing_from_protocol_failure": sum(
                len(row["missing_target_cell_ids"]) for row in rows
            ),
        },
    }


def _support_cells(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """Extract observed and missing target-cell identifiers from row evidence."""

    observed = sorted({cell for row in rows for cell in row["observed_target_cell_ids"]})
    missing = sorted({cell for row in rows for cell in row["missing_target_cell_ids"]})
    return observed, missing


def _source_artifact_hashes(source_bytes: Mapping[str, bytes]) -> JsonDict:
    """Record exact source artifact identities used by this freeze."""

    payloads = _payloads(source_bytes)
    hashes = _source_hashes(source_bytes)
    return {
        name: {
            "path": str(SOURCE_ARTIFACT_PATHS[name]),
            "file_sha256": hashes[name],
            "schema": payloads[name].get("schema"),
            "reproducibility_checksum": payloads[name].get("reproducibility_checksum"),
            "honest_verdict": payloads[name].get("honest_verdict"),
            "verdict_class": payloads[name].get("verdict_class"),
        }
        for name in SOURCE_ARTIFACT_PATHS
    }


def _file_hash(path: Path) -> str | None:
    """Hash a file when present so the command surface is reproducible."""

    full = path if path.is_absolute() else REPO_ROOT / path
    return sha256_bytes(full.read_bytes()) if full.is_file() else None


def _implementation_hashes() -> JsonDict:
    """Bind this module and the thin command wrapper."""

    return {
        "module": {"path": str(MODULE_PATH), "sha256": _file_hash(MODULE_PATH)},
        "wrapper": {"path": str(WRAPPER_PATH), "sha256": _file_hash(WRAPPER_PATH)},
    }


def _methodology(run_date: str) -> JsonDict:
    """State the replay boundary consumed by later experiments."""

    return {
        "source_reduction_imports_allowed": False,
        "llm_calls": 0,
        "parser_repair": False,
        "row_unit": "model_id, prompt arm, scenario_id, and obligation_id",
        "field_cell_unit": "model_id, prompt arm, scenario_id, obligation_id, and field",
        "readiness_is_scientific_result": False,
        "command": (
            ".venv/bin/python scripts/experiments/"
            f"experiment_6835_v598_terminal_evidence_freeze.py --date {run_date}"
        ),
    }


def _empty_artifact(
    *, run_date: str, duration_s: float, source_bytes: Mapping[str, bytes]
) -> JsonDict:
    """Create the full terminal shape before blocked or complete branches."""

    return {
        "schema": ARTIFACT_SCHEMA,
        "experiment_id": "exp6835-v598-terminal-evidence-freeze",
        "run_date": run_date,
        "status": "building",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "source_artifact_hashes": _source_artifact_hashes(source_bytes),
        "implementation_hashes": _implementation_hashes(),
        "methodology": _methodology(run_date),
        "reproducibility_checksum": "",
        "rows": [],
        "failure_taxonomy": {},
        "observed_target_cells": {"count": 0, "cells": [], "sha256": sha256_json([])},
        "missing_target_cells": {"count": 0, "cells": [], "sha256": sha256_json([])},
        "collision_witnesses": [],
        "compatible_policy_lower_bound": {
            "policy_class": "unrestricted binary finite-field policies",
            "exponent": None,
            "lower_bound": None,
        },
        "source_null_preserved": {"preserved": False},
        "obligation_failure_taxonomy_complete_score": 0,
        "v598_evidence_root_ready_score": 0,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_v598_terminal_evidence_freeze",
    }


def _observed_cells_payload(cells: Sequence[str]) -> JsonDict:
    """Pack observed cells with count and digest."""

    ordered = sorted(cells)
    return {
        "count": len(ordered),
        "cells": ordered,
        "sha256": sha256_json(ordered),
    }


def _missing_cells_payload(cells: Sequence[str]) -> JsonDict:
    """Pack missing cells with count and digest."""

    ordered = sorted(cells)
    return {
        "count": len(ordered),
        "cells": ordered,
        "sha256": sha256_json(ordered),
    }


def _source_null_payload(audit: Mapping[str, Any]) -> JsonDict:
    """Preserve the prior terminal null rather than reinterpret it."""

    identifiability = audit.get("identifiability_result", {})
    preserved = (
        audit.get("verdict_class") == "null"
        and identifiability.get("disposition") == "not_separated"
        and str(audit.get("honest_verdict", "")).startswith("complete_null_")
    )
    return {
        "preserved": preserved,
        "source_experiment_id": audit.get("experiment_id"),
        "source_verdict_class": audit.get("verdict_class"),
        "source_honest_verdict": audit.get("honest_verdict"),
        "source_disposition": identifiability.get("disposition"),
        "semantic_preservation_identified": not preserved,
        "terminal_finding": "generated-answer semantic preservation remains unidentified",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind deterministic content while excluding duration and itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def build_artifact(
    *, run_date: str, duration_s: float, source_bytes: Mapping[str, bytes]
) -> JsonDict:
    """Build the blocked artifact or complete V598 evidence root."""

    artifact = _empty_artifact(
        run_date=run_date,
        duration_s=duration_s,
        source_bytes=source_bytes,
    )
    checks = evaluate_preconditions(source_bytes)
    artifact["preconditions_checked"] = checks
    failed = next((row for row in checks if not row["passed"]), None)
    if failed is not None:
        artifact["status"] = "complete_blocked_v598_terminal_evidence_freeze"
        artifact["gate_check_summary"] = {
            "passed": False,
            "failed_check": failed["check"],
            "expected": failed["expected"],
            "observed": failed["observed"],
        }
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    payloads = _payloads(source_bytes)
    fixture = payloads["exp6832"]
    corpus = payloads["exp6833"]
    audit = payloads["exp6834"]
    rows = build_rows(fixture, corpus)
    observed_cells, missing_cells = _support_cells(rows)
    target_cells = _target_cells(fixture, corpus.get("MODEL_SPECS", []))
    taxonomy = _failure_taxonomy(rows)
    source_null = _source_null_payload(audit)
    compatible = {
        "policy_class": "unrestricted binary finite-field policies",
        "exponent": len(missing_cells),
        "lower_bound": "1" if not missing_cells else f"2^{len(missing_cells)}",
        "source_exp6834_bound": audit["identifiability_result"]["compatible_policy_count"],
        "readiness_is_positive_result": False,
    }
    taxonomy_complete = int(
        len(rows) == len(target_cells) // len(OBLIGATION_FIELDS)
        and len(observed_cells) == audit["identifiability_result"]["observed_cell_count"]
        and len(missing_cells) == audit["identifiability_result"]["missing_cell_count"]
        and taxonomy["field_cells"]["observed"] == len(observed_cells)
    )
    ready = int(
        taxonomy_complete == 1
        and source_null["preserved"] is True
        and compatible["lower_bound"] == audit["identifiability_result"]["compatible_policy_count"]
        and len(audit.get("collision_witnesses", [])) > 0
    )
    artifact.update(
        {
            "status": "complete",
            "rows": rows,
            "failure_taxonomy": taxonomy,
            "observed_target_cells": _observed_cells_payload(observed_cells),
            "missing_target_cells": _missing_cells_payload(missing_cells),
            "collision_witnesses": deepcopy(audit.get("collision_witnesses", [])),
            "compatible_policy_lower_bound": compatible,
            "source_null_preserved": source_null,
            "obligation_failure_taxonomy_complete_score": taxonomy_complete,
            "v598_evidence_root_ready_score": ready,
            "gate_check_summary": {
                "passed": ready == 1,
                "failed_check": None if ready == 1 else "v598_evidence_root_completeness",
                "expected": {
                    "obligation_rows": 3780,
                    "observed_target_cells": 395,
                    "missing_target_cells": 18505,
                    "source_null_preserved": True,
                    "ready_score": 1,
                },
                "observed": {
                    "obligation_rows": len(rows),
                    "observed_target_cells": len(observed_cells),
                    "missing_target_cells": len(missing_cells),
                    "source_null_preserved": source_null["preserved"],
                    "ready_score": ready,
                },
            },
            "verdict_class": "null",
            "honest_verdict": "complete_null_v598_terminal_evidence_freeze_source_null_preserved",
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recheck terminal invariants before the artifact is written."""

    errors: list[str] = []
    if set(REQUIRED_ARTIFACT_FIELDS) - set(artifact):
        errors.append("required artifact fields are missing")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field principles do not cover every top-level field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict class is outside the closed set")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest verdict lacks a terminal prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")

    if artifact.get("verdict_class") == "blocked":
        if artifact.get("status") != "complete_blocked_v598_terminal_evidence_freeze":
            errors.append("blocked status mismatch")
        if artifact.get("rows"):
            errors.append("blocked artifact emitted rows")
        if artifact.get("v598_evidence_root_ready_score") != 0:
            errors.append("blocked artifact is marked ready")
        if artifact.get("gate_check_summary", {}).get("failed_check") is None:
            errors.append("blocked artifact lacks failed check")
        return errors

    if artifact.get("status") != "complete":
        errors.append("complete artifact status mismatch")
    if len(artifact.get("rows", [])) != 3780:
        errors.append("complete artifact obligation-row count mismatch")
    if artifact.get("observed_target_cells", {}).get("count") != 395:
        errors.append("observed target-cell count mismatch")
    if artifact.get("missing_target_cells", {}).get("count") != 18505:
        errors.append("missing target-cell count mismatch")
    if artifact.get("compatible_policy_lower_bound", {}).get("lower_bound") != "2^18505":
        errors.append("compatible-policy lower bound mismatch")
    if artifact.get("source_null_preserved", {}).get("preserved") is not True:
        errors.append("source null was not preserved")
    if artifact.get("obligation_failure_taxonomy_complete_score") != 1:
        errors.append("taxonomy complete score mismatch")
    if artifact.get("v598_evidence_root_ready_score") != 1:
        errors.append("evidence root ready score mismatch")
    if artifact.get("verdict_class") == "positive":
        errors.append("readiness was converted into a positive result")
    if artifact.get("gate_check_summary", {}).get("passed") is not True:
        errors.append("complete artifact gate did not pass")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write atomically through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(artifact, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)


def _read_source(path: Path) -> bytes:
    """Turn read failures into normal bytes that fail hash and schema gates."""

    full = path if path.is_absolute() else REPO_ROOT / path
    try:
        return full.read_bytes()
    except OSError as exc:
        return canonical_bytes({"read_error": type(exc).__name__, "path": str(full)})


def _read_all_sources() -> dict[str, bytes]:
    """Read every configured terminal source artifact."""

    return {name: _read_source(path) for name, path in SOURCE_ARTIFACT_PATHS.items()}


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic CPU replay and write the evidence root."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    artifact = build_artifact(
        run_date=args.date,
        duration_s=0.0,
        source_bytes=_read_all_sources(),
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
                "ready": artifact["v598_evidence_root_ready_score"],
                "source_null_preserved": artifact["source_null_preserved"]["preserved"],
                "verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper owns CLI execution.
    raise SystemExit(main())
