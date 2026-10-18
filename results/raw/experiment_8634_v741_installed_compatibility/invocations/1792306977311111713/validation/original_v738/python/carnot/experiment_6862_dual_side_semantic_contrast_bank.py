"""Build the Exp6862 dual-side exact semantic contrast bank.

Spec refs: REQ-CONSTRAINT-6862 and SCENARIO-CONSTRAINT-6862-*.

The reducer reads raw JSON evidence. It does not import any upstream reducer.
The structure and solution checkers also do not call each other. This design
lets each checker expose a different implementation error in the other one.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6862_dual_side_semantic_contrast_bank.json")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6862_dual_side_semantic_contrast_bank.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6862_dual_side_semantic_contrast_bank.py")
TEST_PATH = Path("tests/python/test_experiment_6862_dual_side_semantic_contrast_bank.py")
SOURCE_PATHS = {
    "exp6849": Path("results/experiment_6849_typed_program_isomorphic_authority_audit.json"),
    "exp6852": Path("results/experiment_6852_compatibility_shortcut_authority_audit.json"),
    "exp6861": Path("results/experiment_6861_v600_branch_retirement_evidence_contract.json"),
}

ARTIFACT_SCHEMA = "carnot.experiment_6862.dual_side_semantic_contrast_bank.v1"
INFERENCE_SUBSTRATE = "deterministic CPU dual-side exact checking"
RANDOM_SEED = 6862
RUN_DATE = "20260902"
ATOM_FIELDS = (
    "prerequisite",
    "authority",
    "fallback",
    "execution_consequence",
    "priority",
)
TYPED_FAMILIES = (
    "exact_energy",
    "satisfaction_predicate",
    "memory_guard",
    "arc_guard",
    "diagnostic_obligation",
)
NUISANCE_TRANSFORMS = (
    "identifier_rename",
    "label_swap",
    "row_reorder",
    "normalization_variant",
    "surface_paraphrase",
    "atom_order",
)
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
    "random_seed",
    "reproducibility_checksum",
    "rows",
    "fresh_reducer_manifest",
    "typed_family_manifest",
    "semantic_contrast_group_manifest",
    "structure_side_check_rows",
    "solution_side_check_rows",
    "authority_disagreement_witnesses",
    "identity_collision_witnesses",
    "rejected_group_manifest",
    "nuisance_transform_manifest",
    "semantic_mutation_rows",
    "accepted_contrast_group_count",
    "dual_side_semantic_contrast_bank_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

SPEC_REFS = [
    "REQ-CONSTRAINT-6862",
    "SCENARIO-CONSTRAINT-6862-PRECONDITIONS",
    "SCENARIO-CONSTRAINT-6862-IDENTITY-COLLISION",
    "SCENARIO-CONSTRAINT-6862-SEMANTIC-ALIAS",
    "SCENARIO-CONSTRAINT-6862-OMITTED-ATOM",
    "SCENARIO-CONSTRAINT-6862-VACUOUS-CONSTRAINT",
    "SCENARIO-CONSTRAINT-6862-CHECKER-DISAGREEMENT",
    "SCENARIO-CONSTRAINT-6862-SPLIT-MUTATION",
    "SCENARIO-CONSTRAINT-6862-NUISANCE-INVARIANCE",
    "SCENARIO-CONSTRAINT-6862-SEMANTIC-MUTATION",
    "SCENARIO-CONSTRAINT-6862-FROZEN-TEMPLATES",
]

FIELD_PRINCIPLES = {
    "schema": "The schema fixes the exact bank contract.",
    "experiment_id": "The identifier prevents artifact confusion.",
    "run_date": "The supplied date identifies this execution.",
    "status": "Status separates a ready bank from a blocked precondition.",
    "result_path": "The path gives Exp6863 one stable source.",
    "spec_refs": "Requirement anchors connect evidence to tests.",
    "field_principles": "Each top-level field states why it exists.",
    "preconditions_checked": "Source gates stop unsupported reduction.",
    "inference_substrate": "The declaration records exact CPU checking and no LLM.",
    "duration_s": "Wall time proves that the reducer executed.",
    "source_artifact_hashes": "Hashes bind all immutable authority inputs.",
    "random_seed": "The fixed seed records deterministic construction.",
    "reproducibility_checksum": "The checksum detects evidence drift.",
    "rows": "Each row binds one candidate, authority side, and transform.",
    "fresh_reducer_manifest": "The manifest states the independent reduction design.",
    "typed_family_manifest": "The manifest proves balanced family coverage.",
    "semantic_contrast_group_manifest": "The manifest freezes accepted exact pairs.",
    "structure_side_check_rows": "Rows prove variables, atoms, and mutations.",
    "solution_side_check_rows": "Rows independently prove exact candidate labels.",
    "authority_disagreement_witnesses": "Witnesses preserve rejected checker conflicts.",
    "identity_collision_witnesses": "Witnesses preserve rejected identity conflicts.",
    "rejected_group_manifest": "The manifest preserves every negative control.",
    "nuisance_transform_manifest": "The manifest proves surface-label invariance.",
    "semantic_mutation_rows": "Rows prove one changed atom changes each label.",
    "accepted_contrast_group_count": "The count enforces the 96-group floor.",
    "dual_side_semantic_contrast_bank_ready_score": "The exact gate is consumed by Exp6863.",
    "gate_check_summary": "The summary names every failed readiness check.",
    "frozen_sequence_template_manifest": "The manifest freezes score-free Exp6863 inputs.",
    "verifier_is_oracle": "False records that two external exact authorities supply truth.",
    "verdict_class": "The closed class prevents readiness from becoming an effect claim.",
    "honest_verdict": "The complete prefix gives one terminal outcome.",
}

RAW_PROMPT_TEMPLATE = (
    "Exact semantic obligation program.\n"
    "PROGRAM_JSON_BEGIN\n{program_json}\nPROGRAM_JSON_END\n"
    "Return only the candidate sequence."
)
CANDIDATE_SEQUENCE_TEMPLATE = "CANDIDATE_JSON_BEGIN\n{candidate_json}\nCANDIDATE_JSON_END"


def canonical_json(value: Any) -> bytes:
    """Encode JSON once so content identities do not depend on key order."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return one prefixed SHA-256 representation for all identities."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON content."""

    return sha256_bytes(canonical_json(value))


def content_id(namespace: str, value: Any) -> str:
    """Bind an identity to its namespace and full canonical content."""

    return sha256_json({"namespace": namespace, "content": value})


def spec_anchors(text: str) -> list[str]:
    """Return requirement and scenario anchors from OpenSpec text."""

    return re.findall(r"(?:REQ|SCENARIO)-[A-Z]+-\d+(?:-[A-Z0-9-]+)?", text)


def _json_mapping(raw: bytes) -> JsonDict | None:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_source_bytes(repo_root: Path = REPO_ROOT) -> dict[str, bytes]:
    """Read upstream bytes without importing an upstream implementation."""

    result: dict[str, bytes] = {}
    for source_id, relative_path in SOURCE_PATHS.items():
        try:
            result[source_id] = (repo_root / relative_path).read_bytes()
        except OSError:
            result[source_id] = b""
    return result


def _json_line(prompt_text: str, prefix: str) -> Any:
    for line in prompt_text.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise ValueError(f"missing prompt section: {prefix}")


def parse_typed_program(prompt_text: str) -> JsonDict:
    """Parse either raw Exp6849 prompt form into one typed program mapping."""

    if "HANDOFF_JSON_BEGIN\n" in prompt_text:
        body = prompt_text.split("HANDOFF_JSON_BEGIN\n", 1)[1].split("\nHANDOFF_JSON_END", 1)[0]
        value = json.loads(body)
    else:
        value = {
            "observed_facts": _json_line(prompt_text, "OBSERVED_FACTS_JSON="),
            "obligations": _json_line(prompt_text, "OBLIGATIONS_JSON="),
            "candidate_actions": _json_line(prompt_text, "CANDIDATE_ACTIONS_JSON="),
            "output_schema": _json_line(prompt_text, "OUTPUT_JSON_SCHEMA="),
        }
    if not isinstance(value, dict):
        raise ValueError("typed program is not a mapping")
    return value


def source_program_records(exp6849: Mapping[str, Any]) -> list[JsonDict]:
    """Extract fresh typed sources and exact source candidates from Exp6849."""

    records: list[JsonDict] = []
    pairs = exp6849.get("sanitized_candidate_pair_manifest", [])
    if not isinstance(pairs, list):
        return records
    for pair in pairs:
        if not isinstance(pair, Mapping):
            continue
        raw_inputs = pair.get("raw_sequence_inputs")
        candidates = pair.get("candidates")
        if not isinstance(raw_inputs, Mapping) or not isinstance(candidates, list):
            continue
        prompt_text = raw_inputs.get("prompt_text")
        if not isinstance(prompt_text, str):
            continue
        try:
            program = parse_typed_program(prompt_text)
        except (ValueError, json.JSONDecodeError):
            continue
        valid = [
            row for row in candidates if isinstance(row, Mapping) and row.get("exact_label") is True
        ]
        invalid = [
            row
            for row in candidates
            if isinstance(row, Mapping) and row.get("exact_label") is False
        ]
        if len(valid) != 1 or len(invalid) != 1:
            continue
        selected = valid[0].get("selected_action_ids")
        if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
            continue
        records.append(
            {
                "source_pair_id": pair.get("pair_id"),
                "source_semantic_identity": pair.get("semantic_identity"),
                "case_kind": pair.get("case_kind"),
                "program": program,
                "program_id": content_id("program", program),
                "prompt_text_sha256": sha256_bytes(prompt_text.encode()),
                "valid_selected_action_ids": sorted(selected),
            }
        )
    return records


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"check": name, "expected": expected, "observed": observed, "passed": passed}


def evaluate_preconditions(source_bytes: Mapping[str, bytes]) -> list[JsonDict]:
    """Check the V600 gate and both semantic authority sources before work."""

    exp6861 = _json_mapping(source_bytes.get("exp6861", b""))
    exp6849 = _json_mapping(source_bytes.get("exp6849", b""))
    exp6852 = _json_mapping(source_bytes.get("exp6852", b""))

    v600_observed: Any = (
        "unreadable" if exp6861 is None else exp6861.get("v600_evidence_contract_ready_score")
    )
    checks = [
        _check(
            "v600_evidence_contract_ready_score",
            1,
            v600_observed,
            v600_observed == 1,
        )
    ]

    if exp6849 is None:
        program_observed: Any = "unreadable"
        program_passed = False
    else:
        records = source_program_records(exp6849)
        program_observed = {
            "typed_program_authority_ready_score": exp6849.get(
                "typed_program_authority_ready_score"
            ),
            "typed_source_count": len(records),
        }
        program_passed = (
            exp6849.get("typed_program_authority_ready_score") == 1 and len(records) >= 4
        )
    checks.append(
        _check(
            "exp6849_typed_sources_readable",
            {"typed_program_authority_ready_score": 1, "typed_source_count": ">=4"},
            program_observed,
            program_passed,
        )
    )

    if exp6852 is None:
        failure_observed: Any = "unreadable"
        failure_passed = False
    else:
        summary = exp6852.get("gate_check_summary", {})
        failed_checks = (
            {
                row.get("check")
                for row in summary.get("failed_checks", [])
                if isinstance(row, Mapping)
            }
            if isinstance(summary, Mapping)
            else set()
        )
        failed_shortcuts = sum(
            row.get("passed") is False
            for row in exp6852.get("shortcut_attack_results", [])
            if isinstance(row, Mapping)
        )
        failure_observed = {
            "failed_checks": sorted(item for item in failed_checks if isinstance(item, str)),
            "failed_shortcut_count": failed_shortcuts,
        }
        failure_passed = {
            "shortcut_controls_clear",
            "isomorphic_direction_survives",
        } <= failed_checks and failed_shortcuts > 0
    checks.append(
        _check(
            "exp6852_failure_witnesses_present",
            {
                "failed_checks": [
                    "isomorphic_direction_survives",
                    "shortcut_controls_clear",
                ],
                "failed_shortcut_count": ">0",
            },
            failure_observed,
            failure_passed,
        )
    )
    return checks


def program_atom_rows(program: Mapping[str, Any]) -> list[JsonDict]:
    """Create the complete typed atom ledger from all obligations."""

    rows: list[JsonDict] = []
    for obligation in program.get("obligations", []):
        if not isinstance(obligation, Mapping):
            continue
        obligation_id = obligation.get("obligation_id")
        contract = obligation.get("contract")
        if not isinstance(obligation_id, str) or not isinstance(contract, Mapping):
            continue
        for field in ATOM_FIELDS:
            rows.append(
                {
                    "atom_key": f"{obligation_id}|{field}",
                    "obligation_id": obligation_id,
                    "field": field,
                    "value": deepcopy(contract.get(field)),
                }
            )
    return rows


def _mutated_atom_value(field: str, value: Any) -> Any:
    changed = deepcopy(value)
    if field == "prerequisite":
        changed["none_of"] = sorted([*changed.get("none_of", []), "mutation:blocked"])
    elif field == "authority":
        changed["issuer"] = "mutation:unauthorized"
    elif field == "fallback":
        changed["action_id"] = "mutation:missing-fallback"
    elif field == "execution_consequence":
        changed["add"] = sorted([*changed.get("add", []), "mutation:wrong-effect"])
    else:
        changed["weight"] = int(changed.get("weight", 0)) + 1
    return changed


def _semantic_candidate_content(content: Mapping[str, Any]) -> JsonDict:
    assertions = sorted(
        deepcopy(content.get("atom_assertions", [])), key=lambda row: str(row.get("atom_key"))
    )
    return {
        "family": content.get("family"),
        "group_semantic_identity": content.get("group_semantic_identity"),
        "selected_action_ids": sorted(content.get("selected_action_ids", [])),
        "atom_assertions": assertions,
    }


def refresh_candidate_record(candidate: JsonDict, group_semantic_identity: str) -> None:
    """Refresh content identities after a test or nuisance transform changes content."""

    content = candidate["content"]
    content["group_semantic_identity"] = group_semantic_identity
    candidate["candidate_id"] = content_id("candidate", content)
    candidate["semantic_identity"] = content_id(
        "candidate_semantics", _semantic_candidate_content(content)
    )


def build_contrast_group(source: Mapping[str, Any], family: str, atom_slot: int) -> JsonDict:
    """Build one exact pair without asking either checker for its labels."""

    if family not in TYPED_FAMILIES:
        raise ValueError(f"unknown typed family: {family}")
    program = deepcopy(source["program"])
    atoms = program_atom_rows(program)
    first_obligation_atoms = atoms[: len(ATOM_FIELDS)]
    target = first_obligation_atoms[atom_slot]
    semantic_basis = {
        "source_semantic_identity": source.get("source_semantic_identity"),
        "case_kind": source.get("case_kind"),
        "family": family,
        "target_obligation_index": 0,
        "target_field": target["field"],
    }
    semantic_identity = content_id("contrast_group_semantics", semantic_basis)
    valid_content = {
        "family": family,
        "group_semantic_identity": semantic_identity,
        "selected_action_ids": sorted(source["valid_selected_action_ids"]),
        "atom_assertions": deepcopy(atoms),
    }
    invalid_content = deepcopy(valid_content)
    invalid_target = next(
        row for row in invalid_content["atom_assertions"] if row["atom_key"] == target["atom_key"]
    )
    after_value = _mutated_atom_value(target["field"], target["value"])
    invalid_target["value"] = after_value
    candidates = [
        {"expected_label": True, "content": valid_content},
        {"expected_label": False, "content": invalid_content},
    ]
    for candidate in candidates:
        refresh_candidate_record(candidate, semantic_identity)
    mutation_map = {
        "atom_key": target["atom_key"],
        "obligation_id": target["obligation_id"],
        "field": target["field"],
        "before": deepcopy(target["value"]),
        "after": after_value,
        "before_identity": content_id("atom_value", target["value"]),
        "after_identity": content_id("atom_value", after_value),
        "changed_atom_count": 1,
    }
    group_basis = {
        "semantic_identity": semantic_identity,
        "program_id": source["program_id"],
        "candidate_semantic_identities": sorted(row["semantic_identity"] for row in candidates),
        "mutation_map": mutation_map,
    }
    return {
        "group_id": content_id("contrast_group", group_basis),
        "semantic_identity": semantic_identity,
        "family": family,
        "source_pair_id": source.get("source_pair_id"),
        "program_id": source["program_id"],
        "program": program,
        "mutation_map": mutation_map,
        "candidates": candidates,
    }


def validate_program_structure(
    program: Mapping[str, Any], mutation_map: Mapping[str, Any]
) -> list[str]:
    """Reject ambiguous or empty structure before a contrast enters the bank."""

    reasons: set[str] = set()
    obligations = program.get("obligations", [])
    actions = program.get("candidate_actions", [])
    if not isinstance(obligations, list) or not obligations or not isinstance(actions, list):
        return ["ambiguous_program"]
    obligation_ids = [row.get("obligation_id") for row in obligations if isinstance(row, Mapping)]
    action_ids = [row.get("action_id") for row in actions if isinstance(row, Mapping)]
    if (
        len(obligation_ids) != len(obligations)
        or len(action_ids) != len(actions)
        or any(not isinstance(item, str) or not item for item in obligation_ids + action_ids)
        or len(set(obligation_ids)) != len(obligation_ids)
        or len(set(action_ids)) != len(action_ids)
    ):
        reasons.add("ambiguous_program")
    action_id_set = set(action_ids)
    for obligation in obligations:
        if not isinstance(obligation, Mapping):
            reasons.add("ambiguous_program")
            continue
        contract = obligation.get("contract")
        target = obligation.get("action")
        if not isinstance(contract, Mapping) or not isinstance(target, Mapping):
            reasons.add("ambiguous_program")
            continue
        prerequisite = contract.get("prerequisite", {})
        authority = contract.get("authority", {})
        fallback = contract.get("fallback", {})
        consequence = contract.get("execution_consequence", {})
        priority = contract.get("priority", {})
        effective = [
            isinstance(prerequisite, Mapping)
            and bool([*prerequisite.get("all_of", []), *prerequisite.get("none_of", [])]),
            isinstance(authority, Mapping) and bool(authority.get("issuer")),
            isinstance(fallback, Mapping)
            and bool(fallback.get("action_id"))
            and bool(fallback.get("reason")),
            isinstance(consequence, Mapping)
            and bool([*consequence.get("add", []), *consequence.get("remove", [])]),
            isinstance(priority, Mapping) and bool(priority.get("class")),
        ]
        if not all(effective):
            reasons.add("vacuous_constraint")
        if (
            target.get("action_id") not in action_id_set
            or fallback.get("action_id") not in action_id_set
        ):
            reasons.add("ambiguous_program")
    atoms = {row["atom_key"]: row for row in program_atom_rows(program)}
    target_atom = atoms.get(mutation_map.get("atom_key"))
    if target_atom is None or canonical_json(target_atom["value"]) != canonical_json(
        mutation_map.get("before")
    ):
        reasons.add("ambiguous_program")
    if mutation_map.get("changed_atom_count") != 1:
        reasons.add("split_mutation")
    if canonical_json(mutation_map.get("before")) == canonical_json(mutation_map.get("after")):
        reasons.add("vacuous_constraint")
    output_schema = program.get("output_schema")
    if not isinstance(output_schema, Mapping) or "selected_action_ids" not in output_schema:
        reasons.add("ambiguous_program")
    return sorted(reasons)


def structure_side_check(
    program: Mapping[str, Any],
    candidate: Mapping[str, Any],
    mutation_map: Mapping[str, Any],
) -> JsonDict:
    """Prove the variable, atom ledger, and declared mutation structure."""

    program_reasons = validate_program_structure(program, mutation_map)
    expected_atoms = {row["atom_key"]: row for row in program_atom_rows(program)}
    raw_assertions = candidate.get("atom_assertions", [])
    provided: dict[str, Mapping[str, Any]] = {}
    duplicate_keys: list[str] = []
    if isinstance(raw_assertions, list):
        for assertion in raw_assertions:
            if not isinstance(assertion, Mapping) or not isinstance(assertion.get("atom_key"), str):
                duplicate_keys.append("<malformed>")
                continue
            atom_key = assertion["atom_key"]
            if atom_key in provided:
                duplicate_keys.append(atom_key)
            provided[atom_key] = assertion
    missing = sorted(set(expected_atoms) - set(provided))
    extra = sorted(set(provided) - set(expected_atoms))
    mismatched = sorted(
        atom_key
        for atom_key in set(expected_atoms) & set(provided)
        if canonical_json(expected_atoms[atom_key]["value"])
        != canonical_json(provided[atom_key].get("value"))
    )
    selected = candidate.get("selected_action_ids")
    action_ids = {
        row.get("action_id")
        for row in program.get("candidate_actions", [])
        if isinstance(row, Mapping)
    }
    variable_valid = (
        isinstance(selected, list)
        and all(isinstance(item, str) and item in action_ids for item in selected)
        and len(selected) == len(set(selected))
    )
    mutation_shape_valid = not mismatched or mismatched == [mutation_map.get("atom_key")]
    authority_passed = not (
        program_reasons
        or missing
        or extra
        or duplicate_keys
        or not variable_valid
        or not mutation_shape_valid
    )
    return {
        "authority_side": "structure",
        "authority_passed": authority_passed,
        "observed_label": not mismatched and not missing and not extra and not duplicate_keys,
        "variable": "selected_action_ids",
        "variable_valid": variable_valid,
        "obligation_ids": sorted(
            row.get("obligation_id")
            for row in program.get("obligations", [])
            if isinstance(row, Mapping) and isinstance(row.get("obligation_id"), str)
        ),
        "mutation_atom_key": mutation_map.get("atom_key"),
        "mismatched_atom_keys": mismatched,
        "missing_atom_keys": missing,
        "extra_atom_keys": extra,
        "duplicate_atom_keys": sorted(duplicate_keys),
        "program_reasons": program_reasons,
    }


def solution_side_check(
    program: Mapping[str, Any], candidate: Mapping[str, Any], family: str
) -> JsonDict:
    """Independently derive the legal action set and exact candidate label."""

    diagnostics: list[str] = []
    actions: dict[str, Mapping[str, Any]] = {}
    for action in program.get("candidate_actions", []):
        if not isinstance(action, Mapping) or not isinstance(action.get("action_id"), str):
            diagnostics.append("malformed_action")
            continue
        action_id = action["action_id"]
        if action_id in actions:
            diagnostics.append("duplicate_action")
        actions[action_id] = action
    observed_facts = set(program.get("observed_facts", []))
    fail_closed = sorted(
        action_id for action_id, action in actions.items() if action.get("kind") == "fail_closed"
    )
    expected_actions: list[str] = []
    occupied_resources: set[Any] = set()
    obligations = [row for row in program.get("obligations", []) if isinstance(row, Mapping)]
    obligations.sort(
        key=lambda row: (
            0 if row.get("contract", {}).get("priority", {}).get("class") == "hard" else 1,
            -int(row.get("contract", {}).get("priority", {}).get("weight", 0)),
            int(row.get("contract", {}).get("authority", {}).get("order", 0)),
        )
    )
    impossible = False
    for obligation in obligations:
        contract = obligation.get("contract", {})
        prerequisite = contract.get("prerequisite", {})
        target_id = obligation.get("action", {}).get("action_id")
        fallback_id = contract.get("fallback", {}).get("action_id")
        applicable = set(prerequisite.get("all_of", [])) <= observed_facts and not (
            set(prerequisite.get("none_of", [])) & observed_facts
        )
        if not applicable:
            if fallback_id not in actions:
                impossible = True
                break
            expected_actions.append(fallback_id)
            continue
        target = actions.get(target_id)
        if (
            target is None
            or target.get("authority") != contract.get("authority", {}).get("issuer")
            or canonical_json(target.get("consequence"))
            != canonical_json(contract.get("execution_consequence"))
        ):
            impossible = True
            break
        resource = target.get("resource")
        if resource in occupied_resources:
            if fallback_id not in actions:
                impossible = True
                break
            expected_actions.append(fallback_id)
        else:
            expected_actions.append(target_id)
            occupied_resources.add(resource)
    if impossible:
        expected_actions = fail_closed

    selected = candidate.get("selected_action_ids")
    action_set_valid = (
        isinstance(selected, list)
        and len(selected) == len(set(selected))
        and set(selected) == set(expected_actions)
    )
    if not action_set_valid:
        diagnostics.append("wrong_action_set")

    assertions = candidate.get("atom_assertions", [])
    assertion_by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    if isinstance(assertions, list):
        for assertion in assertions:
            if isinstance(assertion, Mapping) and isinstance(assertion.get("atom_key"), str):
                assertion_by_key[assertion["atom_key"]].append(assertion)
            else:
                diagnostics.append("malformed_atom_assertion")
    else:
        diagnostics.append("malformed_atom_ledger")
    expected_keys: set[str] = set()
    mismatched: list[str] = []
    for obligation in program.get("obligations", []):
        if not isinstance(obligation, Mapping):
            diagnostics.append("malformed_obligation")
            continue
        obligation_id = obligation.get("obligation_id")
        contract = obligation.get("contract", {})
        for field in ATOM_FIELDS:
            atom_key = f"{obligation_id}|{field}"
            expected_keys.add(atom_key)
            matching = assertion_by_key.get(atom_key, [])
            if len(matching) != 1:
                diagnostics.append(f"atom_cardinality:{atom_key}")
                continue
            if canonical_json(matching[0].get("value")) != canonical_json(contract.get(field)):
                mismatched.append(atom_key)
                diagnostics.append(f"atom_mismatch:{atom_key}")
    for extra_key in sorted(set(assertion_by_key) - expected_keys):
        diagnostics.append(f"extra_atom:{extra_key}")

    energy = int(not action_set_valid) + len(diagnostics)
    satisfaction = (
        action_set_valid
        and not mismatched
        and not any(
            item.startswith(("atom_cardinality:", "extra_atom:", "malformed_"))
            for item in diagnostics
        )
    )
    selected_rows = (
        [actions.get(item, {}) for item in selected] if isinstance(selected, list) else []
    )
    memory_guard = satisfaction and all(
        row.get("authority") != "untrusted_candidate" for row in selected_rows
    )
    selected_resources = [
        row.get("resource")
        for row in selected_rows
        if row.get("kind") not in {"no_op", "fail_closed"}
    ]
    arc_guard = satisfaction and len(selected_resources) == len(set(selected_resources))
    labels = {
        "exact_energy": energy == 0,
        "satisfaction_predicate": satisfaction,
        "memory_guard": memory_guard,
        "arc_guard": arc_guard,
        "diagnostic_obligation": not diagnostics,
    }
    return {
        "authority_side": "solution",
        "authority_passed": family in labels,
        "observed_label": labels.get(family, False),
        "expected_action_ids": sorted(expected_actions),
        "observed_action_ids": sorted(selected) if isinstance(selected, list) else [],
        "energy": energy,
        "satisfaction_predicate": satisfaction,
        "memory_guard": memory_guard,
        "arc_guard": arc_guard,
        "mismatched_atom_keys": sorted(mismatched),
        "diagnostics": sorted(diagnostics),
    }


def audit_group(group: Mapping[str, Any]) -> JsonDict:
    """Compare both independent authorities and reject any unsafe group."""

    program = group["program"]
    mutation_map = group["mutation_map"]
    rejection_reasons = set(validate_program_structure(program, mutation_map))
    structure_rows: list[JsonDict] = []
    solution_rows: list[JsonDict] = []
    disagreements: list[JsonDict] = []
    omitted_keys: set[str] = set()
    label_map: dict[bool, bool] = {}
    seen_candidate_ids: set[str] = set()
    seen_semantic_ids: set[str] = set()
    expected_labels: list[bool] = []
    for candidate_record in group.get("candidates", []):
        content = candidate_record["content"]
        expected_label = candidate_record.get("expected_label")
        expected_labels.append(expected_label)
        expected_id = content_id("candidate", content)
        expected_semantic_id = content_id(
            "candidate_semantics", _semantic_candidate_content(content)
        )
        if candidate_record.get("candidate_id") != expected_id:
            rejection_reasons.add("identity_collision")
        if candidate_record.get("semantic_identity") != expected_semantic_id:
            rejection_reasons.add("semantic_alias")
        if expected_id in seen_candidate_ids or expected_semantic_id in seen_semantic_ids:
            rejection_reasons.add("identity_collision")
        seen_candidate_ids.add(expected_id)
        seen_semantic_ids.add(expected_semantic_id)

        structure = structure_side_check(program, content, mutation_map)
        solution = solution_side_check(program, content, group["family"])
        structure.update({"candidate_id": expected_id, "expected_label": expected_label})
        solution.update({"candidate_id": expected_id, "expected_label": expected_label})
        structure_rows.append(structure)
        solution_rows.append(solution)
        omitted_keys.update(structure["missing_atom_keys"])
        if structure["missing_atom_keys"]:
            rejection_reasons.add("omitted_atom")
        if len(structure["mismatched_atom_keys"]) > 1:
            rejection_reasons.add("split_mutation")
        if structure["observed_label"] != solution["observed_label"]:
            rejection_reasons.add("checker_disagreement")
            disagreements.append(
                {
                    "candidate_id": expected_id,
                    "structure_label": structure["observed_label"],
                    "solution_label": solution["observed_label"],
                    "rejected": True,
                }
            )
        if (
            structure["observed_label"] is not expected_label
            or solution["observed_label"] is not expected_label
            or not structure["authority_passed"]
            or not solution["authority_passed"]
        ):
            rejection_reasons.add("authority_label_mismatch")
        if isinstance(expected_label, bool):
            label_map[expected_label] = solution["observed_label"]
    if sorted(expected_labels) != [False, True]:
        rejection_reasons.add("ambiguous_program")
    return {
        "accepted": not rejection_reasons,
        "rejection_reasons": sorted(rejection_reasons),
        "omitted_atom_keys": sorted(omitted_keys),
        "authority_disagreement_witnesses": disagreements,
        "structure_rows": structure_rows,
        "solution_rows": solution_rows,
        "labels": label_map,
    }


def identity_collision_witnesses(
    rows: Sequence[Mapping[str, Any]],
    identifier_field: str,
    semantic_field: str,
    namespace: str,
) -> list[JsonDict]:
    """Preserve identifiers that point at more than one semantic content."""

    grouped: dict[Any, set[Any]] = defaultdict(set)
    for row in rows:
        grouped[row.get(identifier_field)].add(row.get(semantic_field))
    return [
        {
            "collision_kind": "content_identity_collision",
            "identity": identity,
            "namespace": namespace,
            "semantic_identities": sorted(semantics),
            "rejected": True,
        }
        for identity, semantics in sorted(grouped.items(), key=lambda item: str(item[0]))
        if len(semantics) > 1
    ]


def semantic_alias_witnesses(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Preserve different group identifiers that claim the same semantics."""

    grouped: dict[Any, set[Any]] = defaultdict(set)
    for row in rows:
        grouped[row.get("semantic_identity")].add(row.get("group_id"))
    return [
        {
            "collision_kind": "semantic_alias",
            "group_ids": sorted(group_ids),
            "semantic_identity": semantic_identity,
            "rejected": True,
        }
        for semantic_identity, group_ids in sorted(grouped.items(), key=lambda item: str(item[0]))
        if len(group_ids) > 1
    ]


def _recursive_replace(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_recursive_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _recursive_replace(item, replacements) for key, item in value.items()}
    return value


def _identifier_replacements(program: Mapping[str, Any]) -> dict[str, str]:
    categories: dict[str, set[str]] = defaultdict(set)
    for obligation in program.get("obligations", []):
        categories["obligation"].add(obligation["obligation_id"])
        contract = obligation["contract"]
        categories["action"].update(
            [obligation["action"]["action_id"], contract["fallback"]["action_id"]]
        )
        categories["authority"].add(contract["authority"]["issuer"])
        for field in ("all_of", "none_of"):
            categories["fact"].update(contract["prerequisite"].get(field, []))
        for field in ("add", "remove"):
            categories["fact"].update(contract["execution_consequence"].get(field, []))
    for action in program.get("candidate_actions", []):
        categories["action"].add(action["action_id"])
        categories["authority"].add(action.get("authority", ""))
        categories["resource"].add(action.get("resource", ""))
        for field in ("add", "remove"):
            categories["fact"].update(action.get("consequence", {}).get(field, []))
    categories["fact"].update(program.get("observed_facts", []))
    return {
        original: f"renamed-{category}-{index:02d}"
        for category, values in sorted(categories.items())
        for index, original in enumerate(sorted(value for value in values if value))
    }


def _refresh_transformed_group(group: JsonDict, transform_kind: str) -> None:
    for candidate in group["candidates"]:
        refresh_candidate_record(candidate, group["semantic_identity"])
    basis = {
        "base_semantic_identity": group["semantic_identity"],
        "transform_kind": transform_kind,
        "program": group["program"],
        "candidate_contents": [row["content"] for row in group["candidates"]],
        "surface": group.get("surface"),
    }
    group["program_id"] = content_id("program", group["program"])
    group["group_id"] = content_id("transformed_contrast_group", basis)


def nuisance_variants(group: Mapping[str, Any]) -> list[JsonDict]:
    """Create six surface changes that must leave exact labels unchanged."""

    variants: list[JsonDict] = []
    for transform_kind in NUISANCE_TRANSFORMS:
        changed = deepcopy(group)
        if transform_kind == "identifier_rename":
            replacements = _identifier_replacements(changed["program"])
            changed["program"] = _recursive_replace(changed["program"], replacements)
            changed["mutation_map"] = _recursive_replace(changed["mutation_map"], replacements)
            for candidate in changed["candidates"]:
                candidate["content"] = _recursive_replace(candidate["content"], replacements)
                for assertion in candidate["content"]["atom_assertions"]:
                    assertion["atom_key"] = f"{assertion['obligation_id']}|{assertion['field']}"
            changed["mutation_map"]["atom_key"] = (
                f"{changed['mutation_map']['obligation_id']}|{changed['mutation_map']['field']}"
            )
        elif transform_kind == "label_swap":
            changed["candidates"].reverse()
            changed["surface"] = {"display_labels": ["B", "A"]}
        elif transform_kind == "row_reorder":
            changed["program"]["obligations"].reverse()
            changed["program"]["candidate_actions"].reverse()
            changed["candidates"].reverse()
        elif transform_kind == "normalization_variant":
            changed["surface"] = {"json_whitespace": "indented", "unicode_normalization": "NFC"}
        elif transform_kind == "surface_paraphrase":
            changed["surface"] = {
                "instruction": "Choose the actions that satisfy every typed duty."
            }
        else:
            for candidate in changed["candidates"]:
                candidate["content"]["atom_assertions"].reverse()
        _refresh_transformed_group(changed, transform_kind)
        transform_basis = {
            "transform_kind": transform_kind,
            "group_id": changed["group_id"],
            "program_id": changed["program_id"],
            "candidate_ids": [row["candidate_id"] for row in changed["candidates"]],
            "surface": changed.get("surface"),
        }
        variants.append(
            {
                "transform_kind": transform_kind,
                "transform_id": content_id("nuisance_transform", transform_basis),
                "group": changed,
            }
        )
    return variants


def _negative_control_evidence(
    base_group: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    rejected: list[JsonDict] = []

    def record(kind: str, evidence: Any) -> None:
        basis = {"rejection_kind": kind, "evidence": evidence}
        rejected.append(
            {
                "rejection_id": content_id("rejected_group", basis),
                "rejection_kind": kind,
                "evidence": evidence,
                "rejected": True,
            }
        )

    ambiguous = deepcopy(base_group)
    ambiguous["program"]["obligations"].append(deepcopy(ambiguous["program"]["obligations"][0]))
    record("ambiguous_program", audit_group(ambiguous)["rejection_reasons"])

    omitted = deepcopy(base_group)
    omitted_valid = next(row for row in omitted["candidates"] if row["expected_label"] is True)
    omitted_atom = omitted_valid["content"]["atom_assertions"].pop()
    refresh_candidate_record(omitted_valid, omitted["semantic_identity"])
    omitted_result = audit_group(omitted)
    record(
        "omitted_atom",
        {"atom_key": omitted_atom["atom_key"], "reasons": omitted_result["rejection_reasons"]},
    )

    vacuous = deepcopy(base_group)
    vacuous_contract = vacuous["program"]["obligations"][0]["contract"]
    vacuous_contract["prerequisite"] = {"all_of": [], "none_of": []}
    record("vacuous_constraint", audit_group(vacuous)["rejection_reasons"])

    disagreement = deepcopy(base_group)
    disagreement_valid = next(
        row for row in disagreement["candidates"] if row["expected_label"] is True
    )
    disagreement_valid["content"]["selected_action_ids"] = []
    refresh_candidate_record(disagreement_valid, disagreement["semantic_identity"])
    disagreement_result = audit_group(disagreement)
    disagreement_witnesses = disagreement_result["authority_disagreement_witnesses"]
    record("checker_disagreement", disagreement_witnesses)

    split = deepcopy(base_group)
    split_invalid = next(row for row in split["candidates"] if row["expected_label"] is False)
    split_invalid["content"]["atom_assertions"][1]["value"] = {"split_mutation": True}
    refresh_candidate_record(split_invalid, split["semantic_identity"])
    record("split_mutation", audit_group(split)["rejection_reasons"])

    collision_rows = [
        {"candidate_id": "negative-control-collision", "semantic_identity": "semantic-a"},
        {"candidate_id": "negative-control-collision", "semantic_identity": "semantic-b"},
    ]
    collision_witnesses = identity_collision_witnesses(
        collision_rows, "candidate_id", "semantic_identity", "negative_control_candidate"
    )
    record("identity_collision", collision_witnesses)

    alias_witnesses = semantic_alias_witnesses(
        [
            {"group_id": "negative-control-a", "semantic_identity": "semantic-alias"},
            {"group_id": "negative-control-b", "semantic_identity": "semantic-alias"},
        ]
    )
    record("semantic_alias", alias_witnesses)
    return rejected, disagreement_witnesses, collision_witnesses


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "checks": [dict(row) for row in checks],
        "failed_checks": failed,
        "failed_check": failed[0]["check"] if failed else None,
        "observed": failed[0]["observed"] if failed else "all checks pass",
        "passed": not failed,
    }


def _source_hashes(source_bytes: Mapping[str, bytes]) -> JsonDict:
    return {
        source_id: sha256_bytes(source_bytes.get(source_id, b""))
        for source_id in sorted(SOURCE_PATHS)
    }


def _template_manifest() -> JsonDict:
    basis = {
        "raw_prompt_template": RAW_PROMPT_TEMPLATE,
        "candidate_sequence_template": CANDIDATE_SEQUENCE_TEMPLATE,
    }
    return {
        **basis,
        "template_identity": content_id("exp6863_sequence_templates", basis),
        "calibration_group_assignment": None,
        "held_group_assignment": None,
        "tokenizer_inspected": False,
        "model_scores_present": False,
    }


def _base_artifact(
    run_date: str,
    duration_s: float,
    source_bytes: Mapping[str, bytes],
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": ARTIFACT_SCHEMA,
        "experiment_id": "experiment_6862_dual_side_semantic_contrast_bank",
        "run_date": run_date,
        "status": "complete_blocked_dual_side_semantic_contrast_bank",
        "result_path": RESULT_PATH.as_posix(),
        "spec_refs": SPEC_REFS,
        "field_principles": {},
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": _source_hashes(source_bytes),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "rows": [],
        "fresh_reducer_manifest": {},
        "typed_family_manifest": [],
        "semantic_contrast_group_manifest": [],
        "structure_side_check_rows": [],
        "solution_side_check_rows": [],
        "authority_disagreement_witnesses": [],
        "identity_collision_witnesses": [],
        "rejected_group_manifest": [],
        "nuisance_transform_manifest": [],
        "semantic_mutation_rows": [],
        "accepted_contrast_group_count": 0,
        "dual_side_semantic_contrast_bank_ready_score": 0,
        "gate_check_summary": _gate_summary(preconditions),
        "frozen_sequence_template_manifest": _template_manifest(),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_dual_side_semantic_contrast_bank",
    }


def _finish_artifact(artifact: JsonDict) -> JsonDict:
    artifact["field_principles"] = {
        field: FIELD_PRINCIPLES.get(field, "This field preserves exact experiment evidence.")
        for field in artifact
    }
    checksum_payload = deepcopy(artifact)
    checksum_payload["duration_s"] = None
    checksum_payload["reproducibility_checksum"] = None
    artifact["reproducibility_checksum"] = sha256_json(checksum_payload)
    return artifact


def build_artifact(
    repo_root: Path = REPO_ROOT,
    *,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    source_bytes: Mapping[str, bytes] | None = None,
) -> JsonDict:
    """Build a blocked artifact or the complete 100-group exact bank."""

    del repo_root
    sources = dict(source_bytes) if source_bytes is not None else read_source_bytes(REPO_ROOT)
    preconditions = evaluate_preconditions(sources)
    artifact = _base_artifact(run_date, duration_s, sources, preconditions)
    if any(row["passed"] is not True for row in preconditions):
        return _finish_artifact(artifact)

    exp6849 = _json_mapping(sources["exp6849"])
    assert exp6849 is not None
    source_programs = source_program_records(exp6849)
    groups = [
        build_contrast_group(source, family, atom_slot)
        for source in source_programs
        for family in TYPED_FAMILIES
        for atom_slot in range(len(ATOM_FIELDS))
    ]
    alias_witnesses = semantic_alias_witnesses(groups)
    group_collision_witnesses = identity_collision_witnesses(
        groups, "group_id", "semantic_identity", "accepted_group"
    )

    structure_rows: list[JsonDict] = []
    solution_rows: list[JsonDict] = []
    nuisance_manifest: list[JsonDict] = []
    mutation_rows: list[JsonDict] = []
    accepted_groups: list[JsonDict] = []
    build_rejections: list[JsonDict] = []
    accepted_disagreements: list[JsonDict] = []
    for group in groups:
        variants = [
            {
                "transform_kind": "base",
                "transform_id": content_id(
                    "base_transform",
                    {
                        "group_id": group["group_id"],
                        "candidate_ids": [row["candidate_id"] for row in group["candidates"]],
                    },
                ),
                "group": group,
            },
            *nuisance_variants(group),
        ]
        group_passed = True
        base_audit: JsonDict | None = None
        for variant in variants:
            audit = audit_group(variant["group"])
            if variant["transform_kind"] == "base":
                base_audit = audit
            if not audit["accepted"]:
                group_passed = False
                build_rejections.append(
                    {
                        "rejection_id": content_id(
                            "build_rejection",
                            {
                                "group_id": group["group_id"],
                                "transform_id": variant["transform_id"],
                                "reasons": audit["rejection_reasons"],
                            },
                        ),
                        "rejection_kind": "authority_or_transform_failure",
                        "group_id": group["group_id"],
                        "transform_id": variant["transform_id"],
                        "evidence": audit["rejection_reasons"],
                        "rejected": True,
                    }
                )
            accepted_disagreements.extend(audit["authority_disagreement_witnesses"])
            for authority_row in audit["structure_rows"]:
                row_basis = {
                    "group_id": group["group_id"],
                    "candidate_id": authority_row["candidate_id"],
                    "authority_side": "structure",
                    "transform_id": variant["transform_id"],
                }
                structure_rows.append(
                    {
                        "row_id": content_id("authority_row", row_basis),
                        **row_basis,
                        "transform_kind": variant["transform_kind"],
                        "expected_label": authority_row["expected_label"],
                        "observed_label": authority_row["observed_label"],
                        "authority_passed": authority_row["authority_passed"],
                        "mismatched_atom_keys": authority_row["mismatched_atom_keys"],
                    }
                )
            for authority_row in audit["solution_rows"]:
                row_basis = {
                    "group_id": group["group_id"],
                    "candidate_id": authority_row["candidate_id"],
                    "authority_side": "solution",
                    "transform_id": variant["transform_id"],
                }
                solution_rows.append(
                    {
                        "row_id": content_id("authority_row", row_basis),
                        **row_basis,
                        "transform_kind": variant["transform_kind"],
                        "expected_label": authority_row["expected_label"],
                        "observed_label": authority_row["observed_label"],
                        "authority_passed": authority_row["authority_passed"],
                        "energy": authority_row["energy"],
                    }
                )
            if variant["transform_kind"] != "base":
                nuisance_manifest.append(
                    {
                        "transform_id": variant["transform_id"],
                        "transform_kind": variant["transform_kind"],
                        "group_id": group["group_id"],
                        "transformed_group_id": variant["group"]["group_id"],
                        "labels_preserved": audit["labels"] == {False: False, True: True},
                        "both_authorities_passed": audit["accepted"],
                    }
                )
        assert base_audit is not None
        valid_structure = next(
            row for row in base_audit["structure_rows"] if row["expected_label"] is True
        )
        invalid_structure = next(
            row for row in base_audit["structure_rows"] if row["expected_label"] is False
        )
        valid_solution = next(
            row for row in base_audit["solution_rows"] if row["expected_label"] is True
        )
        invalid_solution = next(
            row for row in base_audit["solution_rows"] if row["expected_label"] is False
        )
        mutation_basis = {
            "group_id": group["group_id"],
            "mutation_map": group["mutation_map"],
        }
        mutation_rows.append(
            {
                "mutation_id": content_id("semantic_mutation", mutation_basis),
                "group_id": group["group_id"],
                "atom_key": group["mutation_map"]["atom_key"],
                "changed_atom_count": group["mutation_map"]["changed_atom_count"],
                "structure_before_label": valid_structure["observed_label"],
                "structure_after_label": invalid_structure["observed_label"],
                "structure_label_changed": valid_structure["observed_label"]
                is not invalid_structure["observed_label"],
                "solution_before_label": valid_solution["observed_label"],
                "solution_after_label": invalid_solution["observed_label"],
                "solution_label_changed": valid_solution["observed_label"]
                is not invalid_solution["observed_label"],
            }
        )
        if group_passed:
            accepted_groups.append(group)

    negative_rejections, negative_disagreements, negative_collisions = _negative_control_evidence(
        groups[0]
    )
    all_rejections = [*build_rejections, *negative_rejections]
    family_manifest = [
        {
            "family": family,
            "family_id": content_id("typed_family", family),
            "accepted_group_count": sum(group["family"] == family for group in accepted_groups),
            "both_authorities_passed": all(
                group["family"] != family or group in accepted_groups for group in groups
            ),
        }
        for family in TYPED_FAMILIES
    ]
    group_manifest = [
        {
            "group_id": group["group_id"],
            "semantic_identity": group["semantic_identity"],
            "family": group["family"],
            "source_pair_id": group["source_pair_id"],
            "program_id": group["program_id"],
            "program": group["program"],
            "mutation_map": group["mutation_map"],
            "candidates": group["candidates"],
            "contrast_id": content_id(
                "contrast",
                {
                    "group_id": group["group_id"],
                    "candidate_ids": [row["candidate_id"] for row in group["candidates"]],
                },
            ),
        }
        for group in accepted_groups
    ]
    readiness_checks = [
        _check("accepted_group_floor", ">=96", len(accepted_groups), len(accepted_groups) >= 96),
        _check("semantic_identities_unique", [], alias_witnesses, not alias_witnesses),
        _check(
            "content_identities_unique",
            [],
            group_collision_witnesses,
            not group_collision_witnesses,
        ),
        _check(
            "accepted_authorities_agree",
            [],
            accepted_disagreements,
            not accepted_disagreements,
        ),
        _check(
            "nuisance_labels_preserved",
            True,
            all(
                row["labels_preserved"] and row["both_authorities_passed"]
                for row in nuisance_manifest
            ),
            all(
                row["labels_preserved"] and row["both_authorities_passed"]
                for row in nuisance_manifest
            ),
        ),
        _check(
            "semantic_mutations_flip",
            True,
            all(
                row["changed_atom_count"] == 1
                and row["structure_label_changed"]
                and row["solution_label_changed"]
                for row in mutation_rows
            ),
            all(
                row["changed_atom_count"] == 1
                and row["structure_label_changed"]
                and row["solution_label_changed"]
                for row in mutation_rows
            ),
        ),
        _check(
            "negative_controls_rejected",
            7,
            len(negative_rejections),
            len(negative_rejections) == 7 and all(row["rejected"] for row in negative_rejections),
        ),
        _check("build_rejection_count", 0, len(build_rejections), not build_rejections),
    ]
    gate_summary = _gate_summary([*preconditions, *readiness_checks])
    ready = int(gate_summary["passed"])
    artifact.update(
        {
            "status": "complete",
            "rows": [*structure_rows, *solution_rows],
            "fresh_reducer_manifest": {
                "reducer": "fresh raw Exp6849 JSON parser and typed atom ledger",
                "imports_upstream_reducers": False,
                "structure_checker": "structure_side_check",
                "solution_checker": "solution_side_check",
                "checker_call_graph_shared": False,
                "source_program_manifest": source_programs,
                "group_construction": "4 source programs x 5 families x 5 semantic atoms",
                "candidate_labels_derived_from_names_or_order": False,
            },
            "typed_family_manifest": family_manifest,
            "semantic_contrast_group_manifest": group_manifest,
            "structure_side_check_rows": structure_rows,
            "solution_side_check_rows": solution_rows,
            "authority_disagreement_witnesses": [
                *accepted_disagreements,
                *negative_disagreements,
            ],
            "identity_collision_witnesses": [
                *group_collision_witnesses,
                *negative_collisions,
            ],
            "rejected_group_manifest": all_rejections,
            "nuisance_transform_manifest": nuisance_manifest,
            "semantic_mutation_rows": mutation_rows,
            "accepted_contrast_group_count": len(accepted_groups),
            "dual_side_semantic_contrast_bank_ready_score": ready,
            "gate_check_summary": gate_summary,
            "verdict_class": "null" if ready else "disqualified",
            "honest_verdict": (
                "complete_null_dual_side_semantic_contrast_bank_ready_no_model_scores"
                if ready
                else "complete_disqualified_dual_side_semantic_contrast_bank"
            ),
        }
    )
    return _finish_artifact(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate terminal schema, gates, identities, and stable checksum."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("wrong_inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_oracle_must_be_false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("nonterminal_honest_verdict")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_do_not_cover_top_level")
    ready = artifact.get("dual_side_semantic_contrast_bank_ready_score")
    count = artifact.get("accepted_contrast_group_count")
    if ready == 1 and (not isinstance(count, int) or count < 96):
        errors.append("ready_score_without_96_groups")
    if ready == 1 and artifact.get("gate_check_summary", {}).get("passed") is not True:
        errors.append("ready_score_with_failed_gate")
    if ready == 1:
        groups = artifact.get("semantic_contrast_group_manifest", [])
        group_ids = [row.get("group_id") for row in groups if isinstance(row, Mapping)]
        semantic_ids = [row.get("semantic_identity") for row in groups if isinstance(row, Mapping)]
        if len(group_ids) != len(set(group_ids)):
            errors.append("duplicate_group_id")
        if len(semantic_ids) != len(set(semantic_ids)):
            errors.append("duplicate_group_semantic_identity")
    checksum_payload = deepcopy(dict(artifact))
    observed_checksum = checksum_payload.get("reproducibility_checksum")
    checksum_payload["duration_s"] = None
    checksum_payload["reproducibility_checksum"] = None
    if observed_checksum != sha256_json(checksum_payload):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> Path:
    """Write the validated terminal artifact to its required stable path."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp6862 artifact: " + "; ".join(errors))
    output_path = repo_root / RESULT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic bank builder and write its terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    started = time.monotonic()
    sources = read_source_bytes(REPO_ROOT)
    artifact = build_artifact(
        REPO_ROOT,
        run_date=args.date,
        duration_s=0.0,
        source_bytes=sources,
    )
    artifact["duration_s"] = time.monotonic() - started
    artifact = _finish_artifact(artifact)
    output_path = write_artifact(artifact, REPO_ROOT)
    print(
        json.dumps(
            {
                "accepted_contrast_group_count": artifact["accepted_contrast_group_count"],
                "dual_side_semantic_contrast_bank_ready_score": artifact[
                    "dual_side_semantic_contrast_bank_ready_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
                "result_path": output_path.relative_to(REPO_ROOT).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper is the CLI surface.
    raise SystemExit(main())
