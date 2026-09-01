"""Build the fresh typed-program isomorphic authority audit.

Spec refs: REQ-CONSTRAINT-6849 and SCENARIO-CONSTRAINT-6849-*.

This module reads raw Exp6836 JSON and prompt text. It does not import the
Exp6836 reducer. The fresh reducer derives exact action sets from the typed
contracts, then compiles all views from one atom ledger. This separation lets
Exp6849 test the producer instead of trusting the producer's own reductions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6849_typed_program_isomorphic_authority_audit.json")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6849_typed_program_isomorphic_authority_audit.py")
WRAPPER_PATH = Path(
    "scripts/experiments/experiment_6849_typed_program_isomorphic_authority_audit.py"
)
TEST_PATH = Path("tests/python/test_experiment_6849_typed_program_isomorphic_authority_audit.py")
SOURCE_PATHS = {
    "exp6836": Path("results/experiment_6836_typed_obligation_program_fixture.json"),
    "exp6847": Path("results/experiment_6847_v598_independent_capstone.json"),
    "exp6848": Path("results/experiment_6848_v599_method_change_evidence_contract.json"),
}

ARTIFACT_SCHEMA = "carnot.experiment_6849.typed_program_isomorphic_authority_audit.v1"
INFERENCE_SUBSTRATE = "deterministic CPU exact compilation"
RANDOM_SEED = 6849
RUN_DATE = "20260901"
ATOM_FIELDS = (
    "prerequisite",
    "authority",
    "fallback",
    "execution_consequence",
    "priority",
)
COMPILED_VIEW_NAMES = (
    "scalar_energy",
    "satisfaction_predicate",
    "memory_admission_guard",
    "arc_shadow_action_guard",
    "per_atom_diagnostic",
)
ISOMORPHIC_TRANSFORMS = (
    "identifier_permutation",
    "atom_rename",
    "label_swap",
    "row_reordering",
    "surface_paraphrase",
    "duplicate_removal",
)
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
DISCREPANCY_EXPECTATIONS = {
    "exp6836.compile_parity": False,
    "exp6836.obligation_pair_fixture_ready_score": 0,
    "exp6836.typed_obligation_program_ready_score": 0,
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
    "candidate_identity_manifest",
    "collision_witnesses",
    "compiled_view_parity_rows",
    "isomorphic_transform_manifest",
    "semantic_mutation_rows",
    "duplicate_removal_results",
    "sanitized_candidate_pair_manifest",
    "authority_audit_complete_score",
    "typed_program_authority_ready_score",
    "isomorphic_fixture_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

SPEC_REFS = [
    "REQ-CONSTRAINT-6849",
    "SCENARIO-CONSTRAINT-6849-PRECONDITIONS",
    "SCENARIO-CONSTRAINT-6849-DUPLICATE-IDS",
    "SCENARIO-CONSTRAINT-6849-SEMANTIC-ALIAS",
    "SCENARIO-CONSTRAINT-6849-ATOM-OMISSION",
    "SCENARIO-CONSTRAINT-6849-IMPOSSIBLE",
    "SCENARIO-CONSTRAINT-6849-ROW-COLLISION",
    "SCENARIO-CONSTRAINT-6849-ISOMORPHIC",
    "SCENARIO-CONSTRAINT-6849-REDUCER-MUTATION",
    "SCENARIO-CONSTRAINT-6849-SANITIZED-FIXTURE",
]

FIELD_PRINCIPLES = {
    "schema": "The schema fixes the authority audit contract.",
    "experiment_id": "The experiment identifier prevents artifact confusion.",
    "run_date": "The supplied date identifies this deterministic execution.",
    "status": "Status separates a complete audit from a blocked precondition.",
    "result_path": "The path gives Exp6851 one stable authority source.",
    "spec_refs": "Requirement anchors connect the artifact to executable tests.",
    "field_principles": "Each top-level field explains why it exists.",
    "preconditions_checked": "Exact source gates stop reduction when authority is missing.",
    "inference_substrate": "The audit uses deterministic CPU compilation and no LLM.",
    "duration_s": "Wall time shows that the reducer executed.",
    "source_artifact_hashes": "Hashes bind the raw fixture and discrepancy evidence.",
    "random_seed": "The fixed seed records deterministic transform ordering.",
    "reproducibility_checksum": "The checksum detects evidence drift except wall time.",
    "rows": "Each row binds one typed view, candidate, and transform.",
    "fresh_reducer_manifest": "The manifest states the independent exact reducer design.",
    "candidate_identity_manifest": "The manifest defines every candidate exactly once.",
    "collision_witnesses": "Witnesses expose source aliases and sanitized collisions.",
    "compiled_view_parity_rows": "Rows prove every compiled view has exact parity.",
    "isomorphic_transform_manifest": "The manifest records each label-preserving transform.",
    "semantic_mutation_rows": "Exact negative controls must change compatible labels.",
    "duplicate_removal_results": "The result proves source aliases were removed safely.",
    "sanitized_candidate_pair_manifest": "The score-free manifest is the Exp6851 input.",
    "authority_audit_complete_score": "This gate requires every exact authority check.",
    "typed_program_authority_ready_score": "This gate requires parity and unique identities.",
    "isomorphic_fixture_ready_score": "This gate requires invariance and mutation sensitivity.",
    "gate_check_summary": "The summary names each failed check and observed value.",
    "verifier_is_oracle": "False records that external exact checkers supply truth.",
    "verdict_class": "The closed class prevents readiness from becoming a benefit claim.",
    "honest_verdict": "The complete_ prefix gives one terminal audit outcome.",
}


class AuthorityAuditError(ValueError):
    """Report one stable error when raw typed material cannot be parsed safely."""


def canonical_json(value: Any) -> bytes:
    """Encode JSON in one stable form for identities and checksums."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return one prefixed SHA-256 form for all audit identities."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON after canonical encoding removes order-only differences."""

    return sha256_bytes(canonical_json(value))


def sha256_file(path: Path) -> str | None:
    """Hash a readable file, or return null so a missing source cannot pass."""

    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return None


def spec_anchors(text: str) -> list[str]:
    """Return requirement and scenario anchors from one OpenSpec section."""

    return re.findall(r"(?:REQ|SCENARIO)-[A-Z]+-\d+(?:-[A-Z0-9-]+)?", text)


def _short_identity(prefix: str, value: Any, length: int = 24) -> str:
    digest = sha256_json(value).split(":", 1)[1]
    return f"{prefix}-{digest[:length]}"


def _json_mapping(raw: bytes) -> JsonDict | None:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_source_bytes(repo_root: Path = REPO_ROOT) -> dict[str, bytes]:
    """Read raw source bytes without importing any producer implementation."""

    result: dict[str, bytes] = {}
    for source_id, relative in SOURCE_PATHS.items():
        try:
            result[source_id] = (repo_root / relative).read_bytes()
        except OSError:
            result[source_id] = b""
    return result


def _check(name: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    return {
        "check": name,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed if passed is None else bool(passed),
    }


def _raw_fixture_observation(payload: Mapping[str, Any] | None) -> Any:
    if payload is None:
        return "unreadable"
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return {"row_count": 0, "raw_candidate_count": 0, "raw_prompt_count": 0}
    raw_prompt_count = sum(isinstance(row.get("prompt_text"), str) for row in rows)
    raw_candidate_count = sum(
        isinstance(candidate.get("raw_text"), str)
        for row in rows
        if isinstance(row, Mapping)
        for candidate in row.get("candidates", [])
        if isinstance(candidate, Mapping)
    )
    return {
        "row_count": len(rows),
        "raw_candidate_count": raw_candidate_count,
        "raw_prompt_count": raw_prompt_count,
    }


def _discrepancy_observation(payload: Mapping[str, Any] | None) -> Any:
    if payload is None:
        return "unreadable"
    found = {
        str(row.get("criterion_id")): row.get("observed_value")
        for row in payload.get("rows", [])
        if isinstance(row, Mapping) and row.get("criterion_id") in DISCREPANCY_EXPECTATIONS
    }
    return found


def evaluate_preconditions(source_bytes: Mapping[str, bytes]) -> list[JsonDict]:
    """Check the V599 gate and both raw authority inputs before reduction."""

    payloads = {
        source_id: _json_mapping(source_bytes.get(source_id, b"")) for source_id in SOURCE_PATHS
    }
    contract = payloads["exp6848"]
    contract_value = (
        contract.get("v599_evidence_contract_ready_score") if contract is not None else "unreadable"
    )
    raw_observed = _raw_fixture_observation(payloads["exp6836"])
    raw_ready = isinstance(raw_observed, Mapping) and raw_observed == {
        "row_count": 8,
        "raw_candidate_count": 16,
        "raw_prompt_count": 8,
    }
    discrepancy = _discrepancy_observation(payloads["exp6847"])
    return [
        _check("v599_evidence_contract_ready_score", 1, contract_value),
        _check(
            "exp6836_raw_fixture_material_readable",
            {"row_count": 8, "raw_candidate_count": 16, "raw_prompt_count": 8},
            raw_observed,
            raw_ready,
        ),
        _check(
            "exp6847_discrepancy_rows_present",
            DISCREPANCY_EXPECTATIONS,
            discrepancy,
            discrepancy == DISCREPANCY_EXPECTATIONS,
        ),
    ]


def parse_prompt_program(prompt_text: str) -> JsonDict:
    """Parse either raw Exp6836 prompt form without calling its prompt reducer."""

    marker = re.search(
        r"HANDOFF_JSON_BEGIN\n(?P<body>\{.*\})\nHANDOFF_JSON_END",
        prompt_text,
        flags=re.DOTALL,
    )
    if marker is not None:
        try:
            value = json.loads(marker.group("body"))
        except json.JSONDecodeError as exc:
            raise AuthorityAuditError("raw_prompt_program_missing") from exc
    else:
        keys = {
            "observed_facts": "OBSERVED_FACTS_JSON",
            "obligations": "OBLIGATIONS_JSON",
            "candidate_actions": "CANDIDATE_ACTIONS_JSON",
            "output_schema": "OUTPUT_JSON_SCHEMA",
        }
        value = {}
        try:
            for target, label in keys.items():
                match = re.search(rf"^{label}=(.*)$", prompt_text, flags=re.MULTILINE)
                if match is None:
                    raise AuthorityAuditError("raw_prompt_program_missing")
                value[target] = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise AuthorityAuditError("raw_prompt_program_missing") from exc
    if not isinstance(value, Mapping) or not all(
        isinstance(value.get(key), list)
        for key in ("candidate_actions", "obligations", "observed_facts")
    ):
        raise AuthorityAuditError("raw_prompt_program_missing")
    result = deepcopy(dict(value))
    fail_closed = next(
        (
            str(row.get("action_id"))
            for row in result["candidate_actions"]
            if row.get("kind") == "fail_closed"
        ),
        "",
    )
    match = re.fullmatch(r"act-(.+)-fail-closed", fail_closed)
    if "scenario_id" not in result and match is not None:
        result["scenario_id"] = f"scenario-{match.group(1)}-p0"
    return result


def parse_candidate_text(raw_text: str) -> JsonDict:
    """Parse one canonical candidate and reject missing typed fields."""

    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AuthorityAuditError("candidate_json_invalid") from exc
    required = {
        "atom_values",
        "candidate_id",
        "padding_control",
        "scenario_id",
        "selected_action_ids",
        "surface_form",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise AuthorityAuditError("candidate_fields_invalid")
    return value


def _canonical_program_source(source: Mapping[str, Any]) -> JsonDict:
    return {
        "candidate_actions": sorted(
            deepcopy(list(source.get("candidate_actions", []))),
            key=lambda row: str(row.get("action_id")),
        ),
        "obligations": sorted(
            deepcopy(list(source.get("obligations", []))),
            key=lambda row: str(row.get("obligation_id")),
        ),
        "observed_facts": sorted(str(value) for value in source.get("observed_facts", [])),
        "output_schema": deepcopy(dict(source.get("output_schema", {}))),
        "scenario_id": str(source.get("scenario_id", "")),
    }


def _semantic_atom_key(obligation_id: str, field: str) -> str:
    return f"{obligation_id}|{field}"


class FreshTypedProgram:
    """Compile exact typed views from one independently parsed source ledger."""

    def __init__(self, source: Mapping[str, Any], *, atom_namespace: str) -> None:
        self.source = _canonical_program_source(source)
        self.scenario_id = self.source["scenario_id"]
        self.atom_namespace = atom_namespace
        self.semantic_source_identity = sha256_json(self.source)
        self.program_id = _short_identity(
            "program-6849", {"namespace": atom_namespace, "source": self.source}
        )
        self.legal_action_ids, self.impossible = self._resolve_action_set()
        self.atoms = self._compile_atoms()
        self.atom_ids = tuple(row["atom_id"] for row in self.atoms)
        self.atom_by_semantic = {row["semantic_key"]: row["atom_id"] for row in self.atoms}
        self.view_ids = {
            name: _short_identity("view-6849", {"program_id": self.program_id, "view_name": name})
            for name in COMPILED_VIEW_NAMES
        }

    def _resolve_action_set(self) -> tuple[tuple[str, ...], bool]:
        actions = self.source["candidate_actions"]
        obligations = self.source["obligations"]
        action_ids = [str(row.get("action_id")) for row in actions]
        obligation_ids = [str(row.get("obligation_id")) for row in obligations]
        if len(action_ids) != len(set(action_ids)) or len(obligation_ids) != len(
            set(obligation_ids)
        ):
            return (), True
        action_index = {str(row.get("action_id")): row for row in actions}
        fail_closed = [
            str(row.get("action_id")) for row in actions if row.get("kind") == "fail_closed"
        ]
        facts = set(self.source["observed_facts"])
        active: list[Mapping[str, Any]] = []
        inactive: list[Mapping[str, Any]] = []
        invalid = False
        for obligation in obligations:
            contract = obligation.get("contract")
            if not isinstance(contract, Mapping):
                invalid = True
                continue
            prerequisite = contract.get("prerequisite")
            if not isinstance(prerequisite, Mapping):
                invalid = True
                continue
            all_of = set(prerequisite.get("all_of", []))
            none_of = set(prerequisite.get("none_of", []))
            is_active = all_of.issubset(facts) and not none_of.intersection(facts)
            (active if is_active else inactive).append(obligation)
            if is_active:
                target = obligation.get("action")
                target_id = target.get("action_id") if isinstance(target, Mapping) else None
                action = action_index.get(str(target_id))
                authority = contract.get("authority")
                expected_issuer = (
                    authority.get("issuer") if isinstance(authority, Mapping) else None
                )
                if (
                    action is None
                    or action.get("authority") != expected_issuer
                    or action.get("consequence") != contract.get("execution_consequence")
                ):
                    invalid = True
        if invalid:
            return ((fail_closed[0],), False) if len(fail_closed) == 1 else ((), True)

        resolutions: list[str] = []
        for obligation in inactive:
            fallback = obligation["contract"].get("fallback")
            fallback_id = fallback.get("action_id") if isinstance(fallback, Mapping) else None
            if fallback_id not in action_index:
                return ((fail_closed[0],), False) if len(fail_closed) == 1 else ((), True)
            resolutions.append(str(fallback_id))
        by_resource: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for obligation in active:
            target_id = str(obligation["action"]["action_id"])
            by_resource[str(action_index[target_id].get("resource"))].append(obligation)
        priority_rank = {"hard": 0, "binding": 1, "soft": 2}
        for group in by_resource.values():
            ordered = sorted(
                group,
                key=lambda obligation: (
                    priority_rank.get(str(obligation["contract"]["priority"].get("class")), 99),
                    int(obligation["contract"]["authority"].get("order", 0)),
                    -int(obligation["contract"]["priority"].get("weight", 0)),
                    str(obligation["obligation_id"]),
                ),
            )
            resolutions.append(str(ordered[0]["action"]["action_id"]))
            for obligation in ordered[1:]:
                fallback_id = obligation["contract"]["fallback"].get("action_id")
                if fallback_id not in action_index:
                    return ((fail_closed[0],), False) if len(fail_closed) == 1 else ((), True)
                resolutions.append(str(fallback_id))
        return tuple(sorted(set(resolutions))), False

    def _compile_atoms(self) -> list[JsonDict]:
        atoms: list[JsonDict] = []
        for obligation in self.source["obligations"]:
            obligation_id = str(obligation.get("obligation_id"))
            contract = obligation.get("contract")
            for field in ATOM_FIELDS:
                semantic_key = _semantic_atom_key(obligation_id, field)
                atoms.append(
                    {
                        "atom_id": _short_identity(
                            "atom-6849",
                            {
                                "namespace": self.atom_namespace,
                                "semantic_key": semantic_key,
                                "source": contract.get(field)
                                if isinstance(contract, Mapping)
                                else None,
                            },
                        ),
                        "field": field,
                        "kind": "field",
                        "obligation_id": obligation_id,
                        "semantic_key": semantic_key,
                    }
                )
        joint_key = _semantic_atom_key("__joint__", "joint_action_set")
        atoms.append(
            {
                "atom_id": _short_identity(
                    "atom-6849",
                    {
                        "namespace": self.atom_namespace,
                        "semantic_key": joint_key,
                        "source": list(self.legal_action_ids),
                    },
                ),
                "field": "joint_action_set",
                "kind": "joint",
                "obligation_id": "__joint__",
                "semantic_key": joint_key,
            }
        )
        return atoms

    def evaluate(self, candidate: Mapping[str, Any] | str) -> JsonDict:
        """Evaluate energy, predicates, guards, and diagnostics from one ledger."""

        value = parse_candidate_text(candidate) if isinstance(candidate, str) else dict(candidate)
        atom_values = value.get("atom_values")
        selected_values = value.get("selected_action_ids")
        well_formed = (
            isinstance(atom_values, Mapping)
            and isinstance(selected_values, list)
            and all(isinstance(item, str) for item in selected_values)
            and len(selected_values) == len(set(selected_values))
            and value.get("scenario_id") == self.scenario_id
        )
        selected = tuple(sorted(selected_values)) if isinstance(selected_values, list) else ()
        exact_action_set = not self.impossible and well_formed and selected == self.legal_action_ids
        extra_atoms = (
            sorted(set(atom_values) - set(self.atom_ids))
            if isinstance(atom_values, Mapping)
            else []
        )
        diagnostics: list[JsonDict] = []
        for atom in self.atoms:
            atom_id = atom["atom_id"]
            assertion = atom_values.get(atom_id) if isinstance(atom_values, Mapping) else None
            actual_pass = bool(exact_action_set and not extra_atoms)
            cause = None
            if self.impossible:
                cause = "impossible_program"
            elif not well_formed:
                cause = "candidate_shape_invalid"
            elif atom_id not in atom_values:
                cause = "atom_omission"
            elif assertion not in {"allow", "block"}:
                cause = "invalid_atom_value"
            elif extra_atoms:
                cause = "atom_identity_drift"
            elif not actual_pass:
                cause = (
                    "joint_action_set_mismatch" if atom["kind"] == "joint" else "field_violation"
                )
            elif assertion == "block":
                cause = "atom_contradiction"
            passed = cause is None
            diagnostics.append(
                {
                    "actual_pass": actual_pass,
                    "assertion_value": assertion,
                    "atom_id": atom_id,
                    "cause": cause,
                    "field": atom["field"],
                    "kind": atom["kind"],
                    "obligation_id": atom["obligation_id"],
                    "passed": passed,
                }
            )
        energy = sum(row["passed"] is not True for row in diagnostics)
        satisfied = energy == 0
        views: JsonDict = {
            "scalar_energy": energy,
            "satisfaction_predicate": satisfied,
            "memory_admission_guard": satisfied,
            "arc_shadow_action_guard": satisfied,
            "per_atom_diagnostic": deepcopy(diagnostics),
        }
        return {
            "arc_shadow_action_guard": satisfied,
            "atom_ids": list(self.atom_ids),
            "diagnostics": diagnostics,
            "energy": energy,
            "memory_admission_guard": satisfied,
            "satisfaction_predicate": satisfied,
            "views": views,
            "view_atom_identities": {name: list(self.atom_ids) for name in COMPILED_VIEW_NAMES},
        }


def identity_collision_witnesses(
    records: Sequence[Mapping[str, Any]],
    *,
    namespace: str,
    identifier_field: str,
    semantic_field: str,
) -> list[JsonDict]:
    """Return exact duplicate-ID and semantic-alias witnesses."""

    witnesses: list[JsonDict] = []
    by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_semantic: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_id[str(row.get(identifier_field))].append(row)
        by_semantic[str(row.get(semantic_field))].append(row)
    for identity, group in sorted(by_id.items()):
        if len(group) > 1:
            witnesses.append(
                {
                    "collision_kind": "duplicate_identifier",
                    "identity": identity,
                    "namespace": namespace,
                    "occurrences": len(group),
                    "semantic_identities": sorted({str(row.get(semantic_field)) for row in group}),
                }
            )
    for semantic, group in sorted(by_semantic.items()):
        if len(group) > 1:
            witnesses.append(
                {
                    "collision_kind": "semantic_alias",
                    "identifiers": sorted(str(row.get(identifier_field)) for row in group),
                    "namespace": namespace,
                    "occurrences": len(group),
                    "semantic_identity": semantic,
                }
            )
    return witnesses


def row_collision_witnesses(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Check both serialized row IDs and semantic row identities."""

    return identity_collision_witnesses(
        rows,
        namespace="row",
        identifier_field="row_id",
        semantic_field="semantic_row_identity",
    )


def _source_atom_claims(candidate: Mapping[str, Any], raw: Mapping[str, Any]) -> JsonDict:
    diagnostics = candidate.get("exact_check", {}).get("diagnostics", [])
    old_to_semantic = {
        str(row.get("atom_id")): _semantic_atom_key(
            str(row.get("obligation_id")), str(row.get("field"))
        )
        for row in diagnostics
        if isinstance(row, Mapping)
    }
    return {
        old_to_semantic.get(str(atom_id), f"unknown|{atom_id}"): value
        for atom_id, value in raw.get("atom_values", {}).items()
    }


def _source_occurrences(payload: Mapping[str, Any]) -> list[JsonDict]:
    occurrences: list[JsonDict] = []
    for row in payload.get("rows", []):
        prompt_program = parse_prompt_program(str(row["prompt_text"]))
        prompt_program["scenario_id"] = str(row["scenario_id"])
        source_identity = sha256_json(_canonical_program_source(prompt_program))
        for position, candidate in enumerate(row.get("candidates", [])):
            raw = parse_candidate_text(str(candidate["raw_text"]))
            atom_claims = _source_atom_claims(candidate, raw)
            semantic_payload = {
                "atom_claims": atom_claims,
                "program_source_identity": source_identity,
                "selected_action_ids": sorted(raw["selected_action_ids"]),
            }
            occurrence = {
                "atom_claims": atom_claims,
                "case_kind": str(row["case_kind"]),
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_position": position,
                "label_swap": str(row["label_swap"]),
                "pair_id": str(row["pair_id"]),
                "prompt_program": prompt_program,
                "prompt_text": str(row["prompt_text"]),
                "raw_candidate": raw,
                "row_id": str(row["row_id"]),
                "row_order": int(row["row_order"]),
                "semantic_identity": sha256_json(semantic_payload),
                "source_identity": source_identity,
            }
            source_program = FreshTypedProgram(
                prompt_program,
                atom_namespace=_short_identity("source-atomspace-6849", source_identity),
            )
            source_payload = _candidate_payload_for_program(
                occurrence,
                source_program,
                candidate_id=str(candidate["candidate_id"]),
            )
            occurrence["exact_label"] = source_program.evaluate(source_payload)[
                "satisfaction_predicate"
            ]
            occurrences.append(occurrence)
    return occurrences


def _candidate_payload_for_program(
    record: Mapping[str, Any],
    program: FreshTypedProgram,
    *,
    candidate_id: str,
    action_mapping: Mapping[str, str] | None = None,
    obligation_mapping: Mapping[str, str] | None = None,
    reverse_actions: bool = False,
    surface_form: str = "sanitized_json",
) -> JsonDict:
    action_mapping = action_mapping or {}
    obligation_mapping = obligation_mapping or {}
    atom_values: JsonDict = {}
    for semantic_key, value in record["atom_claims"].items():
        obligation_id, field = str(semantic_key).split("|", 1)
        mapped_obligation = obligation_mapping.get(obligation_id, obligation_id)
        mapped_key = _semantic_atom_key(mapped_obligation, field)
        atom_id = program.atom_by_semantic.get(mapped_key, f"unknown-atom:{mapped_key}")
        atom_values[atom_id] = value
    selected = [
        action_mapping.get(value, value) for value in record["raw_candidate"]["selected_action_ids"]
    ]
    if reverse_actions:
        selected = list(reversed(selected))
    return {
        "atom_values": atom_values,
        "candidate_id": candidate_id,
        "padding_control": "",
        "scenario_id": program.scenario_id,
        "selected_action_ids": selected,
        "surface_form": surface_form,
    }


def _candidate_text(payload: Mapping[str, Any]) -> str:
    return canonical_json(payload).decode("ascii")


def _fixed_sequence(prompt_text: str, candidate_text: str) -> str:
    return f"{prompt_text}\nCANDIDATE_TEXT_BEGIN\n{candidate_text}\nCANDIDATE_TEXT_END"


def _deduplicate_occurrences(
    occurrences: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], JsonDict]:
    by_semantic: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in occurrences:
        by_semantic[str(row["semantic_identity"])].append(row)
    unique: list[JsonDict] = []
    removed: list[JsonDict] = []
    for semantic_identity, group in sorted(by_semantic.items()):
        ordered = sorted(
            group,
            key=lambda row: (
                int(row["row_order"]),
                int(row["candidate_position"]),
                str(row["candidate_id"]),
            ),
        )
        unique.append(deepcopy(dict(ordered[0])))
        for alias in ordered[1:]:
            removed.append(
                {
                    "removed_candidate_id": alias["candidate_id"],
                    "removed_row_id": alias["row_id"],
                    "semantic_identity": semantic_identity,
                }
            )
    labels_preserved = all(
        len({bool(row["exact_label"]) for row in group}) == 1 for group in by_semantic.values()
    )
    return unique, {
        "source_candidate_occurrence_count": len(occurrences),
        "source_unique_semantic_candidate_count": len(unique),
        "removed_alias_count": len(removed),
        "removed_aliases": removed,
        "sanitized_candidate_count": len(unique),
        "labels_preserved": labels_preserved,
    }


def _sanitize_pairs(
    occurrences: Sequence[Mapping[str, Any]],
    unique: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    unique_by_semantic = {str(row["semantic_identity"]): row for row in unique}
    pair_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in occurrences:
        pair_groups[str(row["pair_id"])].append(row)
    pairs: list[JsonDict] = []
    candidate_manifest: list[JsonDict] = []
    for source_pair_id, group in sorted(pair_groups.items()):
        semantic_ids = sorted({str(row["semantic_identity"]) for row in group})
        canonical = [unique_by_semantic[identity] for identity in semantic_ids]
        prompt_program = deepcopy(dict(canonical[0]["prompt_program"]))
        pair_semantic_identity = sha256_json(
            {
                "candidates": semantic_ids,
                "program": _canonical_program_source(prompt_program),
            }
        )
        pair_id = _short_identity("pair-6849", pair_semantic_identity)
        atom_namespace = _short_identity("atomspace-6849", pair_semantic_identity)
        program = FreshTypedProgram(prompt_program, atom_namespace=atom_namespace)
        candidates: list[JsonDict] = []
        for record in canonical:
            candidate_id = _short_identity("candidate-6849", record["semantic_identity"])
            payload = _candidate_payload_for_program(
                record,
                program,
                candidate_id=candidate_id,
            )
            evaluation = program.evaluate(payload)
            text = _candidate_text(payload)
            candidate_row = {
                "candidate_id": candidate_id,
                "exact_label": evaluation["satisfaction_predicate"],
                "raw_sequence_inputs": {
                    "candidate_text": text,
                    "fixed_sequence_text": _fixed_sequence(record["prompt_text"], text),
                },
                "selected_action_ids": list(payload["selected_action_ids"]),
                "semantic_identity": record["semantic_identity"],
                "source_candidate_id": record["candidate_id"],
            }
            candidates.append(candidate_row)
            candidate_manifest.append(
                {
                    "candidate_id": candidate_id,
                    "exact_label": evaluation["satisfaction_predicate"],
                    "pair_id": pair_id,
                    "semantic_identity": record["semantic_identity"],
                    "source_candidate_id": record["candidate_id"],
                }
            )
        candidates.sort(key=lambda row: (not row["exact_label"], row["candidate_id"]))
        pairs.append(
            {
                "atom_namespace": atom_namespace,
                "candidate_order": [row["candidate_id"] for row in candidates],
                "candidates": candidates,
                "case_kind": canonical[0]["case_kind"],
                "pair_id": pair_id,
                "raw_sequence_inputs": {
                    "prompt_text": canonical[0]["prompt_text"],
                    "prompt_text_sha256": sha256_bytes(canonical[0]["prompt_text"].encode()),
                },
                "semantic_identity": pair_semantic_identity,
                "source_pair_id": source_pair_id,
                "token_equality_claimed": False,
                "tokenizer_receipts": [],
            }
        )
    return pairs, sorted(candidate_manifest, key=lambda row: row["candidate_id"])


def _pair_source(pair: Mapping[str, Any]) -> JsonDict:
    source = parse_prompt_program(str(pair["raw_sequence_inputs"]["prompt_text"]))
    candidate = parse_candidate_text(
        str(pair["candidates"][0]["raw_sequence_inputs"]["candidate_text"])
    )
    source["scenario_id"] = candidate["scenario_id"]
    return source


def _candidate_record_for_pair(pair: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    payload = parse_candidate_text(str(candidate["raw_sequence_inputs"]["candidate_text"]))
    program = FreshTypedProgram(_pair_source(pair), atom_namespace=str(pair["atom_namespace"]))
    claims = {
        atom["semantic_key"]: payload["atom_values"][atom["atom_id"]]
        for atom in program.atoms
        if atom["atom_id"] in payload["atom_values"]
    }
    return {
        "atom_claims": claims,
        "raw_candidate": payload,
        "semantic_identity": candidate["semantic_identity"],
    }


def _rename_program(source: Mapping[str, Any], transform_id: str) -> tuple[JsonDict, JsonDict]:
    action_ids = sorted(str(row["action_id"]) for row in source["candidate_actions"])
    obligation_ids = sorted(str(row["obligation_id"]) for row in source["obligations"])
    facts = sorted(
        {
            str(value)
            for obligation in source["obligations"]
            for value in (
                *obligation["contract"]["prerequisite"].get("all_of", []),
                *obligation["contract"]["prerequisite"].get("none_of", []),
                *obligation["contract"]["execution_consequence"].get("add", []),
                *obligation["contract"]["execution_consequence"].get("remove", []),
            )
        }
        | {str(value) for value in source["observed_facts"]}
    )
    authorities = sorted(
        {
            str(obligation["contract"]["authority"].get("issuer"))
            for obligation in source["obligations"]
        }
    )
    resources = sorted(str(row.get("resource")) for row in source["candidate_actions"])
    mapping = {
        **{
            value: f"perm-action-{index:02d}-{transform_id[-6:]}"
            for index, value in enumerate(action_ids)
        },
        **{
            value: f"perm-obligation-{index:02d}-{transform_id[-6:]}"
            for index, value in enumerate(obligation_ids)
        },
        **{
            value: f"perm-fact-{index:02d}-{transform_id[-6:]}" for index, value in enumerate(facts)
        },
        **{
            value: f"perm-authority-{index:02d}-{transform_id[-6:]}"
            for index, value in enumerate(authorities)
        },
        **{
            value: f"perm-resource-{index:02d}-{transform_id[-6:]}"
            for index, value in enumerate(resources)
        },
    }

    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: visit(child) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child) for child in value]
        return mapping.get(value, value) if isinstance(value, str) else value

    renamed = visit(source)
    renamed["scenario_id"] = f"permuted-scenario-{transform_id[-8:]}"
    return renamed, {
        "actions": {value: mapping[value] for value in action_ids},
        "obligations": {value: mapping[value] for value in obligation_ids},
    }


def _transformed_pair(
    pair: Mapping[str, Any], transform_kind: str, transform_id: str
) -> tuple[FreshTypedProgram, list[tuple[Mapping[str, Any], JsonDict]], str]:
    source = _pair_source(pair)
    action_mapping: Mapping[str, str] = {}
    obligation_mapping: Mapping[str, str] = {}
    reverse_actions = False
    surface_form = "sanitized_json"
    if transform_kind == "identifier_permutation":
        source, mappings = _rename_program(source, transform_id)
        action_mapping = mappings["actions"]
        obligation_mapping = mappings["obligations"]
    elif transform_kind == "row_reordering":
        source["candidate_actions"] = list(reversed(source["candidate_actions"]))
        source["obligations"] = list(reversed(source["obligations"]))
        source["observed_facts"] = list(reversed(source["observed_facts"]))
        reverse_actions = True
    elif transform_kind == "surface_paraphrase":
        surface_form = "paraphrased_typed_sequence"
    namespace = _short_identity(
        "atomspace-transform", {"pair": pair["pair_id"], "transform": transform_id}
    )
    program = FreshTypedProgram(source, atom_namespace=namespace)
    candidates = list(pair["candidates"])
    if transform_kind == "label_swap":
        candidates.reverse()
    transformed: list[tuple[Mapping[str, Any], JsonDict]] = []
    for candidate in candidates:
        record = _candidate_record_for_pair(pair, candidate)
        payload = _candidate_payload_for_program(
            record,
            program,
            candidate_id=str(candidate["candidate_id"]),
            action_mapping=action_mapping,
            obligation_mapping=obligation_mapping,
            reverse_actions=reverse_actions,
            surface_form=surface_form,
        )
        transformed.append((candidate, payload))
    prompt_text = (
        "Typed handoff. Apply the same contract and return the exact action identifiers."
        if transform_kind == "surface_paraphrase"
        else str(pair["raw_sequence_inputs"]["prompt_text"])
    )
    return program, transformed, prompt_text


def _view_label(view_name: str, value: Any) -> bool:
    if view_name == "scalar_energy":
        return value == 0
    if view_name == "per_atom_diagnostic":
        return all(row.get("passed") is True for row in value)
    return value is True


def _compile_view_rows(
    pair: Mapping[str, Any],
    *,
    transform_id: str,
    transform_kind: str,
    program: FreshTypedProgram,
    candidates: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for candidate, payload in candidates:
        evaluation = program.evaluate(payload)
        expected_label = bool(candidate["exact_label"])
        for view_name in COMPILED_VIEW_NAMES:
            observed_label = _view_label(view_name, evaluation["views"][view_name])
            semantic_row_identity = sha256_json(
                {
                    "candidate": candidate["semantic_identity"],
                    "pair": pair["semantic_identity"],
                    "transform_kind": transform_kind,
                    "view_name": view_name,
                }
            )
            rows.append(
                {
                    "atom_identities_match": evaluation["view_atom_identities"][view_name]
                    == evaluation["atom_ids"],
                    "candidate_id": candidate["candidate_id"],
                    "expected_label": expected_label,
                    "observed_label": observed_label,
                    "pair_id": pair["pair_id"],
                    "parity_passed": observed_label == expected_label,
                    "row_id": _short_identity("row-6849", semantic_row_identity),
                    "semantic_row_identity": semantic_row_identity,
                    "transform_id": transform_id,
                    "transform_kind": transform_kind,
                    "view_id": program.view_ids[view_name],
                    "view_name": view_name,
                }
            )
    return rows


def _compile_all_transforms(
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    parity_rows: list[JsonDict] = []
    transforms: list[JsonDict] = []
    atom_manifest: list[JsonDict] = []
    view_manifest: list[JsonDict] = []
    for pair in pairs:
        base_transform_id = _short_identity("transform-6849", [pair["pair_id"], "base"])
        program, candidates, _ = _transformed_pair(pair, "duplicate_removal", base_transform_id)
        parity_rows.extend(
            _compile_view_rows(
                pair,
                transform_id=base_transform_id,
                transform_kind="base",
                program=program,
                candidates=candidates,
            )
        )
        atom_manifest.extend(
            {**deepcopy(atom), "program_id": program.program_id} for atom in program.atoms
        )
        view_manifest.extend(
            {
                "program_id": program.program_id,
                "view_id": program.view_ids[name],
                "view_name": name,
            }
            for name in COMPILED_VIEW_NAMES
        )
        for transform_kind in ISOMORPHIC_TRANSFORMS:
            transform_id = _short_identity("transform-6849", [pair["pair_id"], transform_kind])
            transformed_program, transformed_candidates, prompt_text = _transformed_pair(
                pair, transform_kind, transform_id
            )
            transform_rows = _compile_view_rows(
                pair,
                transform_id=transform_id,
                transform_kind=transform_kind,
                program=transformed_program,
                candidates=transformed_candidates,
            )
            parity_rows.extend(transform_rows)
            atom_manifest.extend(
                {**deepcopy(atom), "program_id": transformed_program.program_id}
                for atom in transformed_program.atoms
            )
            view_manifest.extend(
                {
                    "program_id": transformed_program.program_id,
                    "view_id": transformed_program.view_ids[name],
                    "view_name": name,
                }
                for name in COMPILED_VIEW_NAMES
            )
            transforms.append(
                {
                    "all_views_recompiled": len(transform_rows)
                    == len(pair["candidates"]) * len(COMPILED_VIEW_NAMES),
                    "exact_negative_control": False,
                    "labels_preserved": all(row["parity_passed"] for row in transform_rows),
                    "pair_id": pair["pair_id"],
                    "prompt_text_sha256": sha256_bytes(prompt_text.encode()),
                    "transform_id": transform_id,
                    "transform_kind": transform_kind,
                }
            )
    return parity_rows, transforms, atom_manifest, view_manifest


def _semantic_mutations(pairs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for pair in pairs:
        compatible = next(row for row in pair["candidates"] if row["exact_label"] is True)
        source = _pair_source(pair)
        program = FreshTypedProgram(source, atom_namespace=str(pair["atom_namespace"]))
        payload = parse_candidate_text(str(compatible["raw_sequence_inputs"]["candidate_text"]))
        before = FreshTypedProgram(source, atom_namespace=str(pair["atom_namespace"])).evaluate(
            payload
        )
        mutated_atom = next(
            atom_id for atom_id, value in payload["atom_values"].items() if value == "allow"
        )
        mutated = deepcopy(payload)
        mutated["atom_values"][mutated_atom] = "block"
        after = FreshTypedProgram(source, atom_namespace=str(pair["atom_namespace"])).evaluate(
            mutated
        )
        rows.append(
            {
                "after_failed_atom_ids": [
                    row["atom_id"] for row in after["diagnostics"] if row["passed"] is not True
                ],
                "after_label": after["satisfaction_predicate"],
                "all_views_recompiled": all(
                    _view_label(name, after["views"][name]) == after["satisfaction_predicate"]
                    for name in COMPILED_VIEW_NAMES
                ),
                "before_label": before["satisfaction_predicate"],
                "candidate_id": compatible["candidate_id"],
                "label_changed": before["satisfaction_predicate"]
                is not after["satisfaction_predicate"],
                "mutated_atom_id": mutated_atom,
                "mutation_id": _short_identity(
                    "mutation-6849", [pair["pair_id"], compatible["candidate_id"], mutated_atom]
                ),
                "pair_id": pair["pair_id"],
            }
        )
    return rows


def _source_hash_manifest(repo_root: Path, source_bytes: Mapping[str, bytes]) -> JsonDict:
    manifest = {
        source_id: {
            "file_sha256": sha256_bytes(source_bytes.get(source_id, b"")),
            "path": relative.as_posix(),
        }
        for source_id, relative in SOURCE_PATHS.items()
    }
    manifest["implementation"] = {
        name: {"file_sha256": sha256_file(repo_root / path), "path": path.as_posix()}
        for name, path in (
            ("module", MODULE_PATH),
            ("wrapper", WRAPPER_PATH),
            ("test", TEST_PATH),
            ("spec", SPEC_PATH),
        )
    }
    return manifest


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "checks": [deepcopy(dict(row)) for row in checks],
        "expected": first.get("expected") if first else True,
        "failed_check": first.get("check") if first else None,
        "failed_checks": failed,
        "observed": first.get("observed") if first else True,
        "passed": not failed,
    }


def _base_artifact(
    repo_root: Path,
    *,
    run_date: str,
    duration_s: float,
    source_bytes: Mapping[str, bytes],
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": ARTIFACT_SCHEMA,
        "experiment_id": "6849",
        "run_date": run_date,
        "status": "complete_blocked_typed_program_isomorphic_authority_audit",
        "result_path": RESULT_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "field_principles": {},
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": _source_hash_manifest(repo_root, source_bytes),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "rows": [],
        "fresh_reducer_manifest": {},
        "candidate_identity_manifest": [],
        "collision_witnesses": {"sanitized": [], "source": []},
        "compiled_view_parity_rows": [],
        "isomorphic_transform_manifest": [],
        "semantic_mutation_rows": [],
        "duplicate_removal_results": {},
        "sanitized_candidate_pair_manifest": [],
        "authority_audit_complete_score": 0,
        "typed_program_authority_ready_score": 0,
        "isomorphic_fixture_ready_score": 0,
        "gate_check_summary": _gate_summary(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_typed_program_isomorphic_authority_audit",
    }


def _attach_principles(artifact: JsonDict) -> None:
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} is required by REQ-CONSTRAINT-6849.")
        for key in artifact
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind deterministic audit content while excluding measured duration."""

    payload = deepcopy(dict(artifact))
    payload["duration_s"] = 0.0
    payload["reproducibility_checksum"] = ""
    return sha256_json(payload)


def build_artifact(
    repo_root: Path = REPO_ROOT,
    *,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    source_bytes: Mapping[str, bytes] | None = None,
) -> JsonDict:
    """Build a ready authority audit, or stop at the first failed source gate."""

    raw_sources = dict(source_bytes) if source_bytes is not None else read_source_bytes(repo_root)
    preconditions = evaluate_preconditions(raw_sources)
    artifact = _base_artifact(
        repo_root,
        run_date=run_date,
        duration_s=duration_s,
        source_bytes=raw_sources,
        preconditions=preconditions,
    )
    if not all(row["passed"] is True for row in preconditions):
        _attach_principles(artifact)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    producer = _json_mapping(raw_sources["exp6836"])
    assert producer is not None
    occurrences = _source_occurrences(producer)
    unique, duplicate_results = _deduplicate_occurrences(occurrences)
    pairs, candidate_manifest = _sanitize_pairs(occurrences, unique)
    parity_rows, transforms, atom_manifest, view_manifest = _compile_all_transforms(pairs)
    mutation_rows = _semantic_mutations(pairs)

    source_witnesses = identity_collision_witnesses(
        occurrences,
        namespace="source_candidate",
        identifier_field="candidate_id",
        semantic_field="semantic_identity",
    )
    sanitized_witnesses = identity_collision_witnesses(
        candidate_manifest,
        namespace="candidate",
        identifier_field="candidate_id",
        semantic_field="semantic_identity",
    )
    sanitized_witnesses.extend(
        identity_collision_witnesses(
            pairs,
            namespace="pair",
            identifier_field="pair_id",
            semantic_field="semantic_identity",
        )
    )
    sanitized_witnesses.extend(row_collision_witnesses(parity_rows))
    sanitized_witnesses.extend(
        identity_collision_witnesses(
            atom_manifest,
            namespace="atom",
            identifier_field="atom_id",
            semantic_field="atom_id",
        )
    )
    sanitized_witnesses.extend(
        identity_collision_witnesses(
            view_manifest,
            namespace="view",
            identifier_field="view_id",
            semantic_field="view_id",
        )
    )
    sanitized_witnesses.extend(
        identity_collision_witnesses(
            transforms,
            namespace="transform",
            identifier_field="transform_id",
            semantic_field="transform_id",
        )
    )

    parity_ready = bool(parity_rows) and all(
        row["parity_passed"] and row["atom_identities_match"] for row in parity_rows
    )
    transform_ready = bool(transforms) and all(
        row["labels_preserved"] and row["all_views_recompiled"] for row in transforms
    )
    mutation_ready = bool(mutation_rows) and all(
        row["label_changed"] and row["all_views_recompiled"] for row in mutation_rows
    )
    duplicate_ready = (
        duplicate_results["source_candidate_occurrence_count"] == 16
        and duplicate_results["source_unique_semantic_candidate_count"] == 8
        and duplicate_results["removed_alias_count"] == 8
        and duplicate_results["labels_preserved"] is True
    )
    manifest_ready = (
        len(pairs) == 4
        and len(candidate_manifest) == 8
        and all(pair["token_equality_claimed"] is False for pair in pairs)
        and "model_score" not in json.dumps(pairs, sort_keys=True)
    )
    uniqueness_ready = not sanitized_witnesses
    audit_checks = [
        _check("compiled_view_exact_parity", True, parity_ready),
        _check("sanitized_identity_uniqueness", True, uniqueness_ready),
        _check("isomorphic_label_invariance", True, transform_ready),
        _check("one_atom_semantic_mutation_sensitivity", True, mutation_ready),
        _check("duplicate_removal_exactness", True, duplicate_ready),
        _check("sanitized_score_free_fixture", True, manifest_ready),
    ]
    all_checks = [*preconditions, *audit_checks]
    program_ready = parity_ready and uniqueness_ready and duplicate_ready
    isomorphic_ready = transform_ready and mutation_ready and manifest_ready
    audit_ready = program_ready and isomorphic_ready
    artifact.update(
        {
            "status": "complete" if audit_ready else "complete_partial_authority_audit",
            "rows": parity_rows,
            "fresh_reducer_manifest": {
                "atom_identity_manifest": atom_manifest,
                "compiled_view_names": list(COMPILED_VIEW_NAMES),
                "exact_action_resolution": "typed prerequisites, authority, fallback, consequences, and priority",
                "fresh_reducer": "FreshTypedProgram",
                "producer_reducer_imported": False,
                "program_count": len(pairs),
                "raw_program_parser": "parse_prompt_program",
                "single_typed_source_per_compile": True,
                "view_identity_manifest": view_manifest,
            },
            "candidate_identity_manifest": candidate_manifest,
            "collision_witnesses": {
                "sanitized": sanitized_witnesses,
                "source": source_witnesses,
            },
            "compiled_view_parity_rows": parity_rows,
            "isomorphic_transform_manifest": transforms,
            "semantic_mutation_rows": mutation_rows,
            "duplicate_removal_results": duplicate_results,
            "sanitized_candidate_pair_manifest": pairs,
            "authority_audit_complete_score": int(audit_ready),
            "typed_program_authority_ready_score": int(program_ready),
            "isomorphic_fixture_ready_score": int(isomorphic_ready),
            "gate_check_summary": _gate_summary(all_checks),
            "verdict_class": "null" if audit_ready else "partial",
            "honest_verdict": (
                "complete_null_typed_program_isomorphic_authority_ready_no_model_scores"
                if audit_ready
                else "complete_partial_typed_program_isomorphic_authority_audit"
            ),
        }
    )
    _attach_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return all structural and authority errors in a terminal artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles must cover every top-level field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict_class must use the closed enum")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict must start with complete_")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s must be a nonnegative number")
    scores = [
        artifact.get("authority_audit_complete_score"),
        artifact.get("typed_program_authority_ready_score"),
        artifact.get("isomorphic_fixture_ready_score"),
    ]
    if any(score not in {0, 1} for score in scores):
        errors.append("readiness scores must be zero or one")

    blocked = artifact.get("status") == (
        "complete_blocked_typed_program_isomorphic_authority_audit"
    )
    if blocked:
        if artifact.get("rows") != [] or scores != [0, 0, 0]:
            errors.append("blocked artifact must have no rows or readiness")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked artifact must use blocked verdict class")
        if not artifact.get("gate_check_summary", {}).get("failed_check"):
            errors.append("blocked artifact must name the failed check")
    elif artifact.get("authority_audit_complete_score") == 1:
        if artifact.get("verdict_class") != "null":
            errors.append("ready artifact must have a null verdict")
        rows = artifact.get("rows", [])
        if row_collision_witnesses(rows):
            errors.append("ready artifact has identity collisions")
        parity = artifact.get("compiled_view_parity_rows", [])
        transforms = artifact.get("isomorphic_transform_manifest", [])
        mutations = artifact.get("semantic_mutation_rows", [])
        authority_passed = (
            bool(parity)
            and all(row.get("parity_passed") is True for row in parity)
            and bool(transforms)
            and all(row.get("labels_preserved") is True for row in transforms)
            and bool(mutations)
            and all(row.get("label_changed") is True for row in mutations)
            and not artifact.get("collision_witnesses", {}).get("sanitized")
            and artifact.get("gate_check_summary", {}).get("passed") is True
        )
        if not authority_passed:
            errors.append("ready artifact has a failed authority gate")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one complete artifact so readers never observe partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _valid_date(value: str) -> bool:
    if re.fullmatch(r"\d{8}", value) is None:
        return False
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Build or validate Exp6849 without invoking an LLM or producer reducer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output if args.output is not None else args.repo_root / RESULT_PATH
    if args.validate:
        try:
            artifact = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"failed to read artifact: {exc}")
            return 1
        errors = validate_artifact(artifact)
        for error in errors:
            print(error)
        return int(bool(errors))
    if not _valid_date(str(args.date)):
        print("invalid execution date; expected YYYYMMDD")
        return 2
    started = time.monotonic()
    artifact = build_artifact(
        args.repo_root,
        run_date=str(args.date),
        duration_s=0.0,
    )
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        for error in errors:
            print(error)
        return 1
    write_json_atomic(output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository uses the CLI wrapper.
    raise SystemExit(main())
