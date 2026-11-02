"""Build a balanced, source-grouped exact mapping contrast fixture.

The fixture uses executable mappings from Exp6955. It converts every mapping
through the frozen Exp6975 ConstraintIR syntax before two exact engines certify
it. Exact solvers provide the labels, so this is controlled oracle evidence.

Spec refs: REQ-VERIFY-6984 and SCENARIO-VERIFY-6984-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_6955_reformulation_fixture as fixture_exp
from carnot import experiment_6957_smt_mapping_certification as certificate_exp
from carnot import experiment_6975_delayed_constraint_candidate_bank as bank_exp


JsonDict = dict[str, Any]

EXPERIMENT_ID = 6984
SCHEMA_VERSION = "carnot.exp6984.exact_contrast_fixture.v1"
RUN_DATE = "20260904"
RANDOM_SEED = 6_984_202_609_04
EXPECTED_PAIR_COUNT = 36
INFERENCE_SUBSTRATE = "deterministic_z3_contrast_fixture_no_llm"
RESULT_PATH = Path("results/experiment_6984_exact_contrast_fixture.json")
ROW_ROOT = Path("results/raw/experiment_6984_exact_contrast_fixture")

V609_RESULT_PATH = Path("results/experiment_6955_reformulation_fixture.json")
V611_BANK_PATH = Path("results/experiment_6975_delayed_constraint_candidate_bank.json")
V611_CERTIFICATION_PATH = Path("results/experiment_6976_exact_candidate_certification.json")
EXPECTED_V609_HASH = "sha256:5fb31e3b393db0c986274ef076311baeef1447fb07f7cfafd96423c350cd8263"
EXPECTED_V611_BANK_HASH = "sha256:4a9faf7223d174729091248f8cb763cc6b76e481abe69bdadc19adfe7dac5455"
EXPECTED_V611_CERTIFICATION_HASH = (
    "sha256:7d174415abe6b9c3777bc56bdbf0eaa38a37aba390c64685c667f625aa6e05b9"
)
V611_MAPPING_SCHEMA_VERSION = "carnot.constraint_ir.mapping.v1"

FORMULATION_FAMILIES = tuple(fixture_exp.FAMILIES)
SPLITS = ("train", "calibration", "held_out")
SPLIT_GROUP_COUNTS = {"train": 18, "calibration": 6, "held_out": 12}
FAULT_FAMILIES = (
    "bound_change",
    "coefficient_swap",
    "objective_direction_reversal",
    "constraint_omission",
)
DECISION_STATUSES = {"proved", "counterexample"}
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

FEATURE_ROW_FIELDS = (
    "candidate_id",
    "contrast_group_id",
    "split",
    "formulation_family",
    "pair_position",
    "serialized_candidate",
    "serialization_hash",
)
FORBIDDEN_FEATURE_FIELDS = {
    "source_group_id",
    "source_pair_id",
    "fault_family",
    "fault_count",
    "mutation",
    "exact_label",
    "expected_label",
    "certified_relation",
    "authorities_agree",
    "authority_outcome",
    "unblinded_identifiers",
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "source_manifest_rows",
    "source_manifest_hash",
    "rows",
    "per_pair_results",
    "per_candidate_rows",
    "mutation_attempt_rows",
    "fault_family_rows",
    "z3_authority_rows",
    "enumeration_authority_rows",
    "authority_agreement_rows",
    "alpha_rename_rows",
    "serialization_blinding_rows",
    "split_rows",
    "split_hashes",
    "label_balance_rows",
    "source_overlap_rows",
    "expected_pair_count",
    "observed_pair_count",
    "contrast_fixture_complete_score",
    "label_balance_ready_score",
    "controlled_fixture_only",
    "live_extraction_claimed",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema lets replay reject incompatible evidence.",
    "experiment_id": "A stable identity prevents another run from supplying these rows.",
    "run_date": "A fixed date makes the execution boundary explicit.",
    "field_principles": "A reason for each field makes the evidence contract reviewable.",
    "preconditions_checked": "Exact gates stop changed inputs from entering the fixture.",
    "inference_substrate": "The substrate states that deterministic solvers supplied labels.",
    "duration_s": "Measured wall time proves that fixture construction executed.",
    "source_artifact_hashes": "Source hashes bind the fixture to frozen mapping evidence.",
    "source_manifest_rows": "The manifest fixes source groups before candidate labels open.",
    "source_manifest_hash": "A manifest hash detects any later group or split change.",
    "rows": "Raw candidate rows preserve the complete denominator before aggregation.",
    "per_pair_results": "Pair rows keep positive and one-fault outcomes together.",
    "per_candidate_rows": "A blinded feature table prevents oracle provenance shortcuts.",
    "mutation_attempt_rows": "Mutation receipts prove that each negative has one fault.",
    "fault_family_rows": "Fault summaries prove required held-out mutation coverage.",
    "z3_authority_rows": "Z3 receipts provide symbolic independent decisions.",
    "enumeration_authority_rows": "Enumeration receipts exhaust every bounded assignment.",
    "authority_agreement_rows": "Parity rows stop either exact engine from certifying itself.",
    "alpha_rename_rows": "Renaming receipts show that identifiers do not determine labels.",
    "serialization_blinding_rows": "Blinding receipts exclude source and oracle metadata.",
    "split_rows": "Split rows expose every source assignment for leakage checks.",
    "split_hashes": "Per-split hashes freeze label-blind partitions.",
    "label_balance_rows": "Balance rows prevent constant-label calibration shortcuts.",
    "source_overlap_rows": "Overlap rows prove source groups do not cross partitions.",
    "expected_pair_count": "The preregistered denominator prevents silent pair replacement.",
    "observed_pair_count": "The observed count exposes omissions from the frozen manifest.",
    "contrast_fixture_complete_score": "One requires 36 accepted pairs and full hash replay.",
    "label_balance_ready_score": "One requires exact balance, nonconstant labels, and no leakage.",
    "controlled_fixture_only": "True limits this result to constructed fixture evidence.",
    "live_extraction_claimed": "False prevents a constructed fixture from becoming an extraction claim.",
    "random_seed": "One seed fixes pair order and alpha renaming.",
    "reproducibility_checksum": "A timing-free digest detects scientific-content drift.",
    "gate_check_summary": "Failed checks retain expected and observed values for diagnosis.",
    "verifier_is_oracle": "True states that exact authorities supplied the candidate labels.",
    "verdict_class": "A closed class keeps oracle evidence separate from a learned win.",
    "honest_verdict": "A terminal prefix gives automation a stable outcome.",
}


class ImmutableRowError(RuntimeError):
    """Report an attempted change to a path that already holds frozen bytes."""


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for equality and content hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return one SHA-256 digest with the repository prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON content without formatting or dictionary-order effects."""

    return sha256_bytes(canonical_json(value))


def sha256_path(path: Path) -> str | None:
    """Hash one file while keeping absence visible as a null value."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record one exact precondition comparison with both values."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep only failed checks while preserving diagnostic values."""

    return [
        {
            "check": row.get("check"),
            "expected_value": deepcopy(row.get("expected_value")),
            "observed_value": deepcopy(row.get("observed_value")),
        }
        for row in checks
        if row.get("passed") is not True
    ]


def write_immutable_json(path: Path, value: Any) -> str:
    """Create one row once and allow later runs only when bytes are identical."""

    payload = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ImmutableRowError(f"immutable_row_mismatch:{path}") from None
    return sha256_bytes(payload)


def write_json_atomic(path: Path, value: Any) -> None:
    """Replace the aggregate only after complete JSON bytes exist beside it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> JsonDict:
    """Read an object artifact and return an empty object for a failed boundary."""

    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _row_storage_writable(row_root: Path) -> bool:
    """Probe exclusive row creation without changing any requested row path."""

    try:
        row_root.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".write-probe-", dir=row_root)
        os.close(descriptor)
        Path(name).unlink()
    except OSError:
        return False
    return True


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind the new fixture to the three frozen evidence artifacts it consumes."""

    return {
        "experiment_6955_v609_fixture": sha256_path(repo_root / V609_RESULT_PATH),
        "experiment_6975_v611_bank": sha256_path(repo_root / V611_BANK_PATH),
        "experiment_6976_v611_certification": sha256_path(repo_root / V611_CERTIFICATION_PATH),
    }


def collect_preconditions(repo_root: Path, row_root: Path) -> list[JsonDict]:
    """Check frozen schemas, exact authorities, family support, and row storage."""

    v609 = _read_json(repo_root / V609_RESULT_PATH)
    v611_bank = _read_json(repo_root / V611_BANK_PATH)
    v611_certification = _read_json(repo_root / V611_CERTIFICATION_PATH)
    v609_observed = {
        "artifact_hash": sha256_path(repo_root / V609_RESULT_PATH),
        "artifact_schema": v609.get("schema"),
        "mapping_schema": fixture_exp.MAPPING_SCHEMA_VERSION,
        "ready_score": v609.get("reformulation_fixture_ready_score"),
    }
    v611_observed = {
        "bank_hash": sha256_path(repo_root / V611_BANK_PATH),
        "bank_schema": v611_bank.get("schema"),
        "bank_ready_score": v611_bank.get("candidate_bank_complete_score"),
        "certification_hash": sha256_path(repo_root / V611_CERTIFICATION_PATH),
        "certification_schema": v611_certification.get("schema"),
        "certification_ready_score": v611_certification.get(
            "candidate_certification_complete_score"
        ),
        "mapping_schema": V611_MAPPING_SCHEMA_VERSION,
    }
    family_values = sorted(
        {str(row.get("family")) for row in v609.get("family_rows", []) if row.get("family")}
    )
    return [
        gate_check(
            "v609_mapping_schema",
            {
                "artifact_hash": EXPECTED_V609_HASH,
                "artifact_schema": "carnot.exp6955.reformulation_fixture.v1",
                "mapping_schema": "carnot.reformulation_mapping.v1",
                "ready_score": 1,
            },
            v609_observed,
        ),
        gate_check(
            "v611_mapping_schema",
            {
                "bank_hash": EXPECTED_V611_BANK_HASH,
                "bank_schema": "carnot.experiment_6975.delayed_constraint_candidate_bank.v1",
                "bank_ready_score": 1,
                "certification_hash": EXPECTED_V611_CERTIFICATION_HASH,
                "certification_schema": "carnot.exp6976.exact_candidate_certification.v1",
                "certification_ready_score": 1,
                "mapping_schema": V611_MAPPING_SCHEMA_VERSION,
            },
            v611_observed,
        ),
        gate_check("z3_available", True, fixture_exp.z3 is not None),
        gate_check(
            "bounded_enumeration_available",
            True,
            callable(certificate_exp.certify_with_enumerator),
        ),
        gate_check(
            "supported_formulation_families",
            sorted(FORMULATION_FAMILIES),
            family_values,
        ),
        gate_check("immutable_row_paths_writable", True, _row_storage_writable(row_root)),
    ]


def frozen_source_pairs() -> dict[str, JsonDict]:
    """Regenerate the hash-bound V609 corpus through its executable specification."""

    return {
        str(row["pair_id"]): deepcopy(row)
        for row in fixture_exp.generate_pairs(fixture_exp.RANDOM_SEED)
    }


def _source_slots() -> list[tuple[str, int, int, int]]:
    """Return fixed family, template, ordinal, and split choices for 36 groups."""

    slots: list[tuple[str, int, int, int]] = []
    for family_index, family in enumerate(FORMULATION_FAMILIES):
        del family
        coordinates = [
            *(("train", template, ordinal) for template in (0, 1) for ordinal in range(3)),
            *(("calibration", 2, ordinal) for ordinal in range(2)),
            ("held_out", 2, 2),
            *(("held_out", 3, ordinal) for ordinal in range(3)),
        ]
        slots.extend(
            (split, family_index, template, ordinal) for split, template, ordinal in coordinates
        )
    return slots


def build_source_manifest(source_pairs: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze source identity and split assignment without reading candidate labels."""

    rows: list[JsonDict] = []
    family_split_indices: dict[tuple[str, str], int] = {}
    for position, (split, family_index, template, ordinal) in enumerate(_source_slots()):
        pair_id = f"{family_index}-{template}-{ordinal}"
        pair = source_pairs[pair_id]
        family = str(pair["family"])
        index_key = (family, split)
        family_split_index = family_split_indices.get(index_key, 0)
        family_split_indices[index_key] = family_split_index + 1
        source_hash = fixture_exp.formulation_hash(pair["source"])
        target_hash = fixture_exp.formulation_hash(pair["target"])
        mapping_hash = fixture_exp.mapping_hash(pair["mapping"])
        source_group_id = sha256_json(
            {
                "source_pair_id": pair_id,
                "source_hash": source_hash,
                "target_hash": target_hash,
                "mapping_hash": mapping_hash,
            }
        )
        rows.append(
            {
                "manifest_position": position,
                "source_group_id": source_group_id,
                "contrast_group_id": sha256_json(
                    {"seed": RANDOM_SEED, "source_group_id": source_group_id}
                ),
                "source_pair_id": pair_id,
                "formulation_family": family,
                "split": split,
                "family_split_index": family_split_index,
                "source_formulation_hash": source_hash,
                "target_formulation_hash": target_hash,
                "base_mapping_hash": mapping_hash,
                "manifest_frozen_before_candidate_outcomes": True,
            }
        )
    return rows


def _objective_term_maps(expression: Mapping[str, Any]) -> list[JsonDict]:
    """Return mutable objective coefficient maps for linear and piecewise forms."""

    if expression["kind"] == "linear":
        return [expression["terms"]]
    return [piece["terms"] for piece in expression["pieces"]]


def apply_fault(base_pair: Mapping[str, Any], fault_family: str) -> tuple[JsonDict, JsonDict]:
    """Apply one declared semantic fault while keeping the mapping claim blinded."""

    if fault_family not in FAULT_FAMILIES:
        raise ValueError(f"unknown_fault_family:{fault_family}")
    pair = deepcopy(base_pair)
    pair["mapping"]["claimed_relation"] = "equivalent"
    before_hash = fixture_exp.pair_hash(pair)
    detail: JsonDict
    if fault_family == "bound_change":
        constraint = pair["target"]["constraints"][0]
        old = constraint["rhs"]
        constraint["rhs"] = "999" if constraint["op"] in {">=", "=="} else "-999"
        detail = {"surface": "target.constraints[0].rhs", "before": old, "after": constraint["rhs"]}
    elif fault_family == "coefficient_swap":
        selected: tuple[JsonDict, str, str] | None = None
        for terms in _objective_term_maps(pair["target"]["objective"]["expression"]):
            names = sorted(terms)
            for left_index, left in enumerate(names):
                for right in names[left_index + 1 :]:
                    if terms[left] != terms[right]:
                        selected = (terms, left, right)
                        break
                if selected is not None:
                    break
            if selected is not None:
                break
        if selected is None:
            raise ValueError("coefficient_swap_requires_distinct_coefficients")
        terms, left, right = selected
        before = {left: terms[left], right: terms[right]}
        terms[left], terms[right] = terms[right], terms[left]
        detail = {
            "surface": "target.objective.coefficients",
            "coefficient_names": [left, right],
            "before": before,
            "after": {left: terms[left], right: terms[right]},
        }
    elif fault_family == "objective_direction_reversal":
        old = pair["target"]["objective"]["direction"]
        pair["target"]["objective"]["direction"] = "max" if old == "min" else "min"
        detail = {
            "surface": "target.objective.direction",
            "before": old,
            "after": pair["target"]["objective"]["direction"],
        }
    else:
        removed = pair["target"]["constraints"].pop(0)
        detail = {"surface": "target.constraints[0]", "before": removed, "after": None}
    after_hash = fixture_exp.pair_hash(pair)
    return pair, {
        "fault_family": fault_family,
        "fault_count": 1,
        "changed": before_hash != after_hash,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "mutation_detail": detail,
        "terminal": True,
    }


def _rename_formulation(formulation: Mapping[str, Any], rename: Mapping[str, str]) -> JsonDict:
    """Rename every variable occurrence so the executable meaning stays unchanged."""

    result = deepcopy(formulation)
    for variable in result["variables"]:
        variable["name"] = rename[str(variable["name"])]
    for constraint in result["constraints"]:
        constraint["terms"] = {
            rename[str(name)]: value for name, value in constraint["terms"].items()
        }
    expression = result["objective"]["expression"]
    for terms in _objective_term_maps(expression):
        renamed = {rename[str(name)]: value for name, value in terms.items()}
        terms.clear()
        terms.update(renamed)
    return fixture_exp.validate_formulation(result)


def _rename_map(names: Sequence[str], prefix: str, blind_key: str) -> dict[str, str]:
    """Assign canonical aliases in a seed-derived order that does not use labels."""

    ordered = sorted(
        names,
        key=lambda name: sha256_json(
            {"seed": RANDOM_SEED, "blind_key": blind_key, "identifier": name}
        ),
    )
    return {name: f"{prefix}{index}" for index, name in enumerate(ordered)}


def alpha_rename_pair(pair: Mapping[str, Any], blind_key: str) -> tuple[JsonDict, JsonDict]:
    """Rename both formulations and mapping rows without accepting a label input."""

    source_names = [str(row["name"]) for row in pair["source"]["variables"]]
    target_names = [str(row["name"]) for row in pair["target"]["variables"]]
    source_rename = _rename_map(source_names, "s", blind_key)
    target_rename = _rename_map(target_names, "t", blind_key)
    result = deepcopy(pair)
    result["source"] = _rename_formulation(pair["source"], source_rename)
    result["target"] = _rename_formulation(pair["target"], target_rename)
    for row in result["mapping"]["variables"]:
        row["source"] = source_rename[str(row["source"])]
        row["target"] = target_rename[str(row["target"])]
    for row in result["mapping"]["domain_clauses"]:
        row["source"] = source_rename[str(row["source"])]
        row["target"] = target_rename[str(row["target"])]
    result["mapping"] = fixture_exp.canonical_mapping(
        result["mapping"], result["source"], result["target"]
    )
    new_names = {
        *(str(row["name"]) for row in result["source"]["variables"]),
        *(str(row["name"]) for row in result["target"]["variables"]),
    }
    original_names = set(source_names) | set(target_names)
    receipt = {
        "blind_key": blind_key,
        "source_rename_map": source_rename,
        "target_rename_map": target_rename,
        "injective": len(set(source_rename.values())) == len(source_rename)
        and len(set(target_rename.values())) == len(target_rename),
        "label_blind": True,
        "original_identifiers_present": bool(original_names & new_names),
        "before_hash": fixture_exp.pair_hash(pair),
        "after_hash": fixture_exp.pair_hash(result),
    }
    return result, receipt


def _v611_mapping(mapping: Mapping[str, Any]) -> JsonDict:
    """Serialize the exact V609 mapping through the frozen V611 surface schema."""

    objective = mapping["objective"]
    return {
        "schema_version": V611_MAPPING_SCHEMA_VERSION,
        "variable_map": deepcopy(mapping["variables"]),
        "objective_map": {
            "direction": (
                "same"
                if objective["source_direction"] == objective["target_direction"]
                else "reversed"
            ),
            "scale": objective["scale"],
            "offset": objective["offset"],
        },
    }


def _adapt_v611_mapping(candidate: Mapping[str, Any], pair: Mapping[str, Any]) -> JsonDict:
    """Restore exact domain clauses after the V611 surface parser accepts syntax."""

    variables = deepcopy(candidate["variable_map"])
    source_domains = {str(row["name"]): row["domain"] for row in pair["source"]["variables"]}
    target_domains = {str(row["name"]): row["domain"] for row in pair["target"]["variables"]}
    source_direction = str(pair["source"]["objective"]["direction"])
    target_direction = (
        source_direction
        if candidate["objective_map"]["direction"] == "same"
        else ("max" if source_direction == "min" else "min")
    )
    mapping = {
        "schema_version": fixture_exp.MAPPING_SCHEMA_VERSION,
        "variables": variables,
        "domain_clauses": [
            {
                "source": row["source"],
                "target": row["target"],
                "source_lower": source_domains[str(row["source"])]["lower"],
                "source_upper": source_domains[str(row["source"])]["upper"],
                "target_lower": target_domains[str(row["target"])]["lower"],
                "target_upper": target_domains[str(row["target"])]["upper"],
            }
            for row in variables
        ],
        "objective": {
            "source_direction": source_direction,
            "target_direction": target_direction,
            "scale": candidate["objective_map"]["scale"],
            "offset": candidate["objective_map"]["offset"],
        },
        "claimed_relation": "equivalent",
    }
    return fixture_exp.canonical_mapping(mapping, pair["source"], pair["target"])


def _nested_keys(value: Any) -> set[str]:
    """Collect dictionary keys so feature blinding checks nested content too."""

    if isinstance(value, Mapping):
        return set(map(str, value)) | set().union(*(_nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value)) if value else set()
    return set()


def blinded_feature_row(
    pair: Mapping[str, Any],
    *,
    candidate_id: str,
    contrast_group_id: str,
    split: str,
    family: str,
    pair_position: int,
) -> tuple[JsonDict, JsonDict]:
    """Create the future feature row without source, mutation, or oracle metadata."""

    payload = {
        "source_formulation": deepcopy(pair["source"]),
        "target_formulation": deepcopy(pair["target"]),
        "mapping_candidate": _v611_mapping(pair["mapping"]),
    }
    serialized = canonical_json(payload).decode()
    row = {
        "candidate_id": candidate_id,
        "contrast_group_id": contrast_group_id,
        "split": split,
        "formulation_family": family,
        "pair_position": pair_position,
        "serialized_candidate": serialized,
        "serialization_hash": sha256_bytes(serialized.encode()),
    }
    forbidden = sorted(_nested_keys(row) & FORBIDDEN_FEATURE_FIELDS)
    receipt = {
        "candidate_id": candidate_id,
        "forbidden_fields_present": forbidden,
        "canonical_serialization": serialized == canonical_json(json.loads(serialized)).decode(),
        "serialization_hash": row["serialization_hash"],
        "serialization_hash_replays": row["serialization_hash"]
        == sha256_bytes(serialized.encode()),
        "terminal": True,
    }
    return row, receipt


def build_authority_agreement(
    candidate_id: str,
    enumeration: Mapping[str, Any],
    z3_row: Mapping[str, Any],
) -> JsonDict:
    """Compare every required exact obligation and reject all nondecisions."""

    status_agreement = enumeration.get("status") == z3_row.get("status")
    relation_agreement = enumeration.get("label") == z3_row.get("label")
    domain_fields = ("forward_feasible", "reverse_feasible", "variable_coverage_complete")
    domain_agreement = all(enumeration.get(field) == z3_row.get(field) for field in domain_fields)
    satisfiability_agreement = all(
        enumeration.get(field) == z3_row.get(field)
        for field in ("forward_feasible", "reverse_feasible")
    )
    optimum_agreement = all(
        enumeration.get(field) == z3_row.get(field)
        for field in ("objective_direction_valid", "objective_affine_preserved")
    )
    objective_order_agreement = enumeration.get("objective_order_preserved") == z3_row.get(
        "objective_order_preserved"
    )
    solution_space_relation_agreement = domain_agreement and relation_agreement
    decided = (
        enumeration.get("status") in DECISION_STATUSES and z3_row.get("status") in DECISION_STATUSES
    )
    all_required = all(
        (
            decided,
            status_agreement,
            relation_agreement,
            domain_agreement,
            satisfiability_agreement,
            optimum_agreement,
            objective_order_agreement,
            solution_space_relation_agreement,
        )
    )
    return {
        "candidate_id": candidate_id,
        "enumeration_status": enumeration.get("status"),
        "z3_status": z3_row.get("status"),
        "status_agreement": status_agreement,
        "domain_agreement": domain_agreement,
        "satisfiability_agreement": satisfiability_agreement,
        "optimum_agreement": optimum_agreement,
        "objective_order_agreement": objective_order_agreement,
        "solution_space_relation_agreement": solution_space_relation_agreement,
        "relation_agreement": relation_agreement,
        "certified_relation": enumeration.get("label") if all_required else None,
        "all_required_agreement": all_required,
        "terminal": decided,
    }


def certify_pair(pair: Mapping[str, Any], candidate_id: str) -> JsonDict:
    """Parse one V611 candidate and certify its V609 exact meaning twice."""

    source = fixture_exp.validate_formulation(pair["source"])
    target = fixture_exp.validate_formulation(pair["target"])
    v611_mapping = _v611_mapping(pair["mapping"])
    raw_mapping = canonical_json(v611_mapping).decode()
    syntax = bank_exp.parse_syntax(raw_mapping)
    if syntax.get("constraintir_shape_valid") is not True:
        raise ValueError(f"v611_mapping_schema_rejected:{syntax.get('syntax_reason')}")
    mapping = _adapt_v611_mapping(json.loads(raw_mapping), {"source": source, "target": target})
    engine_pair = {
        "pair_id": candidate_id,
        "source": source,
        "target": target,
        "mapping": mapping,
    }
    enumeration = certificate_exp.certify_with_enumerator(engine_pair)
    z3_row = certificate_exp.certify_with_z3(engine_pair)
    agreement = build_authority_agreement(candidate_id, enumeration, z3_row)
    return {
        "enumeration": enumeration,
        "z3": z3_row,
        "agreement": agreement,
        "v611_schema_valid": True,
    }


def pair_is_accepted(
    agreements: Sequence[Mapping[str, Any]],
    fault_counts: Sequence[int],
    certified_relations: Sequence[str | None],
) -> bool:
    """Admit only a decided positive and one-fault negative with full parity."""

    return (
        len(agreements) == 2
        and all(row.get("all_required_agreement") is True for row in agreements)
        and sorted(fault_counts) == [0, 1]
        and sorted(certified_relations, key=str) == ["equivalent", "non_equivalent"]
    )


def _candidate_order(source_group_id: str) -> list[str]:
    """Randomize pair positions from the seed without exposing the later labels."""

    return sorted(
        ("positive", "negative"),
        key=lambda role: sha256_json(
            {"seed": RANDOM_SEED, "source_group_id": source_group_id, "position_key": role}
        ),
    )


def _persist_row(row_root: Path, category: str, identity: str, row: JsonDict) -> JsonDict:
    """Write one immutable evidence row and attach a replay receipt to the aggregate."""

    relative = Path(category) / f"{identity}.json"
    digest = write_immutable_json(row_root / relative, row)
    return {
        **deepcopy(row),
        "row_path": str(relative),
        "row_sha256": digest,
        "row_hash_replays": sha256_json(row) == digest,
    }


def _split_rows(manifest: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project only label-blind source assignments into the split ledger."""

    return [
        {
            "source_group_id": row["source_group_id"],
            "contrast_group_id": row["contrast_group_id"],
            "source_pair_id": row["source_pair_id"],
            "formulation_family": row["formulation_family"],
            "split": row["split"],
        }
        for row in manifest
    ]


def _split_hashes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Hash each frozen source partition before candidate labels are available."""

    return {
        split: sha256_json([dict(row) for row in rows if row["split"] == split]) for split in SPLITS
    }


def _label_balance_rows(pair_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce certified labels by split without trusting expected role names."""

    rows: list[JsonDict] = []
    for split in SPLITS:
        labels = [
            label
            for pair in pair_rows
            if pair["split"] == split
            for label in pair["certified_relations"]
        ]
        positive = labels.count("equivalent")
        negative = labels.count("non_equivalent")
        group_count = sum(pair["split"] == split for pair in pair_rows)
        rows.append(
            {
                "split": split,
                "group_count": group_count,
                "candidate_count": len(labels),
                "positive_count": positive,
                "negative_count": negative,
                "exactly_balanced": positive == negative == group_count,
                "nonconstant": positive > 0 and negative > 0,
                "terminal": True,
            }
        )
    return rows


def _source_overlap_rows(split_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Compare every split pair so source leakage cannot hide in an aggregate."""

    groups = {
        split: {str(row["source_group_id"]) for row in split_rows if row["split"] == split}
        for split in SPLITS
    }
    return [
        {
            "left_split": left,
            "right_split": right,
            "overlap_count": len(groups[left] & groups[right]),
            "overlap_source_group_ids": sorted(groups[left] & groups[right]),
            "terminal": True,
        }
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
    ]


def _fault_family_rows(mutations: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose every split and fault count, including zero-count cells."""

    return [
        {
            "split": split,
            "fault_family": fault,
            "pair_count": sum(
                row["split"] == split and row["fault_family"] == fault for row in mutations
            ),
            "terminal": True,
        }
        for split in SPLITS
        for fault in FAULT_FAMILIES
    ]


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding wall time and the digest itself."""

    payload = {
        key: deepcopy(value)
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(payload)


def _empty_rows() -> JsonDict:
    """Return all required row families for a schema-complete blocked result."""

    return {
        field: []
        for field in (
            "source_manifest_rows",
            "rows",
            "per_pair_results",
            "per_candidate_rows",
            "mutation_attempt_rows",
            "fault_family_rows",
            "z3_authority_rows",
            "enumeration_authority_rows",
            "authority_agreement_rows",
            "alpha_rename_rows",
            "serialization_blinding_rows",
            "split_rows",
            "label_balance_rows",
            "source_overlap_rows",
        )
    }


def build_blocked_artifact(
    *, repo_root: Path, preconditions: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Build the full fail-closed schema when any prerequisite is absent."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        **_empty_rows(),
        "source_manifest_hash": None,
        "split_hashes": {},
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "observed_pair_count": 0,
        "contrast_fixture_complete_score": 0,
        "label_balance_ready_score": 0,
        "controlled_fixture_only": True,
        "live_extraction_claimed": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(preconditions),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_exact_contrast_fixture",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(repo_root: Path, row_root: Path, *, duration_s: float) -> JsonDict:
    """Freeze, construct, certify, persist, and reduce all 36 contrast pairs."""

    preconditions = collect_preconditions(repo_root, row_root)
    if gate_summary(preconditions):
        return build_blocked_artifact(
            repo_root=repo_root, preconditions=preconditions, duration_s=duration_s
        )

    source_pairs = frozen_source_pairs()
    manifest = build_source_manifest(source_pairs)
    source_manifest_hash = sha256_json(manifest)
    manifest_file_hash = write_immutable_json(row_root / "source_manifest.json", manifest)
    split_rows = _split_rows(manifest)
    split_hashes = _split_hashes(split_rows)

    raw_rows: list[JsonDict] = []
    feature_rows: list[JsonDict] = []
    pair_results: list[JsonDict] = []
    mutation_rows: list[JsonDict] = []
    z3_rows: list[JsonDict] = []
    enumeration_rows: list[JsonDict] = []
    agreement_rows: list[JsonDict] = []
    alpha_rows: list[JsonDict] = []
    blinding_rows: list[JsonDict] = []

    for manifest_row in manifest:
        base = deepcopy(source_pairs[str(manifest_row["source_pair_id"])])
        fault = FAULT_FAMILIES[int(manifest_row["family_split_index"]) % len(FAULT_FAMILIES)]
        negative, mutation = apply_fault(base, fault)
        candidates = {"positive": base, "negative": negative}
        order = _candidate_order(str(manifest_row["source_group_id"]))
        candidate_ids: dict[str, str] = {}
        certified_relations: list[str | None] = []
        pair_agreements: list[JsonDict] = []
        pair_alpha_rows: list[JsonDict] = []
        pair_hash_replays: list[bool] = []

        for position, role in enumerate(order):
            candidate_id = sha256_json(
                {
                    "seed": RANDOM_SEED,
                    "contrast_group_id": manifest_row["contrast_group_id"],
                    "pair_position": position,
                }
            )
            candidate_ids[role] = candidate_id
            candidate = candidates[role]
            renamed, alpha_receipt = alpha_rename_pair(
                candidate, str(manifest_row["contrast_group_id"])
            )
            feature, blind_receipt = blinded_feature_row(
                renamed,
                candidate_id=candidate_id,
                contrast_group_id=str(manifest_row["contrast_group_id"]),
                split=str(manifest_row["split"]),
                family=str(manifest_row["formulation_family"]),
                pair_position=position,
            )
            raw_row = _persist_row(row_root, "candidates", candidate_id[7:], feature)
            raw_rows.append(raw_row)
            feature_rows.append(feature)
            blinding_rows.append(blind_receipt)

            before_certificate = certify_pair(candidate, f"{candidate_id}:before_alpha")
            certificate = certify_pair(renamed, candidate_id)
            alpha_receipt.update(
                {
                    "candidate_id": candidate_id,
                    "relation_before": before_certificate["agreement"]["certified_relation"],
                    "relation_after": certificate["agreement"]["certified_relation"],
                    "certification_invariant": before_certificate["agreement"]["certified_relation"]
                    == certificate["agreement"]["certified_relation"],
                    "terminal": True,
                }
            )
            alpha_rows.append(alpha_receipt)
            pair_alpha_rows.append(alpha_receipt)

            enum_row = {"candidate_id": candidate_id, **certificate["enumeration"]}
            z3_row = {"candidate_id": candidate_id, **certificate["z3"]}
            stored_enum = _persist_row(
                row_root, "enumeration_authority", candidate_id[7:], enum_row
            )
            stored_z3 = _persist_row(row_root, "z3_authority", candidate_id[7:], z3_row)
            enumeration_rows.append(stored_enum)
            z3_rows.append(stored_z3)
            agreement = certificate["agreement"]
            agreement_rows.append(agreement)
            pair_agreements.append(agreement)
            certified_relations.append(agreement["certified_relation"])
            pair_hash_replays.extend(
                [
                    raw_row["row_hash_replays"],
                    stored_enum["row_hash_replays"],
                    stored_z3["row_hash_replays"],
                ]
            )

        mutation.update(
            {
                "source_group_id": manifest_row["source_group_id"],
                "source_pair_id": manifest_row["source_pair_id"],
                "contrast_group_id": manifest_row["contrast_group_id"],
                "candidate_id": candidate_ids["negative"],
                "split": manifest_row["split"],
                "formulation_family": manifest_row["formulation_family"],
            }
        )
        mutation_rows.append(mutation)
        accepted = pair_is_accepted(
            pair_agreements, [0, int(mutation["fault_count"])], certified_relations
        )
        pair_results.append(
            {
                "source_group_id": manifest_row["source_group_id"],
                "source_pair_id": manifest_row["source_pair_id"],
                "contrast_group_id": manifest_row["contrast_group_id"],
                "split": manifest_row["split"],
                "formulation_family": manifest_row["formulation_family"],
                "candidate_ids": [candidate_ids[role] for role in order],
                "positive_candidate_id": candidate_ids["positive"],
                "negative_candidate_id": candidate_ids["negative"],
                "fault_family": fault,
                "fault_count": mutation["fault_count"],
                "certified_relations": certified_relations,
                "authorities_agree": all(row["all_required_agreement"] for row in pair_agreements),
                "alpha_rename_invariant": all(
                    row["certification_invariant"] for row in pair_alpha_rows
                ),
                "hashes_replay": all(pair_hash_replays),
                "pair_accepted": accepted,
                "replacement_used": False,
                "terminal": all(row["terminal"] for row in pair_agreements),
            }
        )

    label_balance_rows = _label_balance_rows(pair_results)
    source_overlap_rows = _source_overlap_rows(split_rows)
    fault_rows = _fault_family_rows(mutation_rows)
    label_balance_ready = int(
        all(row["exactly_balanced"] and row["nonconstant"] for row in label_balance_rows)
        and all(row["overlap_count"] == 0 for row in source_overlap_rows)
        and all(
            sum(row["split"] == split for row in manifest) == SPLIT_GROUP_COUNTS[split]
            for split in SPLITS
        )
    )
    held_faults = {str(row["fault_family"]) for row in mutation_rows if row["split"] == "held_out"}
    held_families = {
        str(row["formulation_family"]) for row in pair_results if row["split"] == "held_out"
    }
    hashes_replay = (
        manifest_file_hash == source_manifest_hash
        and split_hashes == _split_hashes(split_rows)
        and all(row["row_hash_replays"] for row in raw_rows + z3_rows + enumeration_rows)
    )
    complete = int(
        len(pair_results) == EXPECTED_PAIR_COUNT
        and all(row["terminal"] and row["pair_accepted"] for row in pair_results)
        and all(row["all_required_agreement"] for row in agreement_rows)
        and all(row["fault_count"] == 1 and row["changed"] for row in mutation_rows)
        and all(row["certification_invariant"] for row in alpha_rows)
        and all(not row["forbidden_fields_present"] for row in blinding_rows)
        and held_faults == set(FAULT_FAMILIES)
        and held_families == set(FORMULATION_FAMILIES)
        and hashes_replay
    )
    scientific_checks = [
        gate_check("observed_pair_count", EXPECTED_PAIR_COUNT, len(pair_results)),
        gate_check("all_pairs_accepted", True, all(row["pair_accepted"] for row in pair_results)),
        gate_check("every_hash_replays", True, hashes_replay),
        gate_check("label_balance_ready", 1, label_balance_ready),
        gate_check("held_out_fault_families", sorted(FAULT_FAMILIES), sorted(held_faults)),
        gate_check(
            "held_out_formulation_families",
            sorted(FORMULATION_FAMILIES),
            sorted(held_families),
        ),
    ]
    ready = complete == 1 and label_balance_ready == 1
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        "source_manifest_rows": manifest,
        "source_manifest_hash": source_manifest_hash,
        "rows": raw_rows,
        "per_pair_results": pair_results,
        "per_candidate_rows": feature_rows,
        "mutation_attempt_rows": mutation_rows,
        "fault_family_rows": fault_rows,
        "z3_authority_rows": z3_rows,
        "enumeration_authority_rows": enumeration_rows,
        "authority_agreement_rows": agreement_rows,
        "alpha_rename_rows": alpha_rows,
        "serialization_blinding_rows": blinding_rows,
        "split_rows": split_rows,
        "split_hashes": split_hashes,
        "label_balance_rows": label_balance_rows,
        "source_overlap_rows": source_overlap_rows,
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "observed_pair_count": len(pair_results),
        "contrast_fixture_complete_score": complete,
        "label_balance_ready_score": label_balance_ready,
        "controlled_fixture_only": True,
        "live_extraction_claimed": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary([*preconditions, *scientific_checks]),
        "verifier_is_oracle": True,
        "verdict_class": "circular_positive" if ready else "disqualified",
        "honest_verdict": (
            "complete_circular_exact_contrast_fixture"
            if ready
            else "complete_disqualified_exact_contrast_fixture"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _stored_row_hash_replays(row: Mapping[str, Any]) -> bool:
    """Recompute one immutable row hash from its embedded scientific payload."""

    payload = {
        key: deepcopy(value)
        for key, value in row.items()
        if key not in {"row_path", "row_sha256", "row_hash_replays"}
    }
    return row.get("row_sha256") == sha256_json(payload) and row.get("row_hash_replays") is True


def validate_artifact(artifact: Mapping[str, Any], *, repo_root: Path) -> list[str]:
    """Recompute schema, claims, rows, scores, verdict, and stable digest."""

    del repo_root  # Source byte identity is already captured by precondition rows.
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing_required_fields:{missing}")
        return errors
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles")
    for field in ("contrast_fixture_complete_score", "label_balance_ready_score"):
        if type(artifact[field]) is not int or artifact[field] not in {0, 1}:
            errors.append(f"bare_score:{field}")
    if (
        artifact["controlled_fixture_only"] is not True
        or artifact["live_extraction_claimed"] is not False
    ):
        errors.append("claim_boundary")
    if artifact["verifier_is_oracle"] is not True:
        errors.append("oracle_declaration")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact["verdict_class"] == "blocked":
        if (
            artifact["contrast_fixture_complete_score"] != 0
            or artifact["label_balance_ready_score"] != 0
        ):
            errors.append("blocked_score")
        if not artifact["gate_check_summary"]:
            errors.append("blocked_gate_summary")
        if not str(artifact["honest_verdict"]).startswith("blocked_exact_contrast_fixture"):
            errors.append("blocked_verdict")
    else:
        if artifact["expected_pair_count"] != EXPECTED_PAIR_COUNT:
            errors.append("expected_pair_count")
        if artifact["observed_pair_count"] != len(artifact["per_pair_results"]):
            errors.append("pair_count")
        if len(artifact["per_pair_results"]) != EXPECTED_PAIR_COUNT:
            errors.append("pair_count")
        if len(artifact["rows"]) != 2 * EXPECTED_PAIR_COUNT:
            errors.append("candidate_count")
        if any(set(row) != set(FEATURE_ROW_FIELDS) for row in artifact["per_candidate_rows"]):
            errors.append("feature_blinding")
        if any(
            _nested_keys(row) & FORBIDDEN_FEATURE_FIELDS for row in artifact["per_candidate_rows"]
        ):
            errors.append("feature_blinding")
        if artifact["source_manifest_hash"] != sha256_json(artifact["source_manifest_rows"]):
            errors.append("source_manifest_hash")
        if artifact["split_hashes"] != _split_hashes(artifact["split_rows"]):
            errors.append("split_hashes")
        recomputed_overlap = _source_overlap_rows(artifact["split_rows"])
        if artifact["source_overlap_rows"] != recomputed_overlap or any(
            row["overlap_count"] for row in recomputed_overlap
        ):
            errors.append("source_overlap")
        recomputed_balance = _label_balance_rows(artifact["per_pair_results"])
        if artifact["label_balance_rows"] != recomputed_balance:
            errors.append("label_balance_rows")
        balance_score = int(
            all(row["exactly_balanced"] and row["nonconstant"] for row in recomputed_balance)
            and all(row["overlap_count"] == 0 for row in recomputed_overlap)
        )
        if artifact["label_balance_ready_score"] != balance_score:
            errors.append("label_balance_score")
        row_hashes_replay = all(
            _stored_row_hash_replays(row)
            for row in (
                list(artifact["rows"])
                + list(artifact["z3_authority_rows"])
                + list(artifact["enumeration_authority_rows"])
            )
        )
        complete = int(
            len(artifact["per_pair_results"]) == EXPECTED_PAIR_COUNT
            and all(
                row.get("terminal") is True and row.get("pair_accepted") is True
                for row in artifact["per_pair_results"]
            )
            and len(artifact["authority_agreement_rows"]) == 2 * EXPECTED_PAIR_COUNT
            and all(
                row.get("all_required_agreement") is True
                for row in artifact["authority_agreement_rows"]
            )
            and len(artifact["mutation_attempt_rows"]) == EXPECTED_PAIR_COUNT
            and all(
                row.get("fault_count") == 1 and row.get("changed") is True
                for row in artifact["mutation_attempt_rows"]
            )
            and all(
                row.get("certification_invariant") is True for row in artifact["alpha_rename_rows"]
            )
            and all(
                not row.get("forbidden_fields_present")
                and row.get("serialization_hash_replays") is True
                for row in artifact["serialization_blinding_rows"]
            )
            and row_hashes_replay
        )
        if artifact["contrast_fixture_complete_score"] != complete:
            errors.append("contrast_complete_score")
        ready = complete == 1 and balance_score == 1
        expected_class = "circular_positive" if ready else "disqualified"
        if artifact["verdict_class"] != expected_class:
            errors.append("verdict_class_consistency")
        expected_prefix = "complete_circular_" if ready else "complete_disqualified_"
        if not str(artifact["honest_verdict"]).startswith(expected_prefix):
            errors.append("honest_verdict_consistency")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        errors.append("checksum")
    return errors


def run(
    *,
    repo_root: Path,
    result_path: Path,
    row_root: Path,
    run_date: str,
) -> JsonDict:
    """Build, validate, and write one terminal result artifact."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:expected={RUN_DATE}:observed={run_date}")
    started = time.perf_counter()
    artifact = build_artifact(repo_root, row_root, duration_s=0.0)
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact, repo_root=repo_root)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command wrapper.
    """Expose the fixed-date command used by the research conductor."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    artifact = run(
        repo_root=repo_root,
        result_path=repo_root / RESULT_PATH,
        row_root=repo_root / ROW_ROOT,
        run_date=args.date,
    )
    print(json.dumps({"verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module command surface.
    raise SystemExit(main())
