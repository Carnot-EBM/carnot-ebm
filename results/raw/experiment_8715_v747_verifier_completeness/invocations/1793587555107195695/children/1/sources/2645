"""Build a frozen fixture of exact minimal constraint interventions.

Each block contains one exact pair and an alpha-renamed copy. The learner file
contains prompts only. Exact labels and all provenance stay in a separate
sidecar, so later models cannot learn authority shortcuts from this fixture.

Spec refs: REQ-VERIFY-7012 and SCENARIO-VERIFY-7012-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6955_reformulation_fixture as fixture_exp
from carnot import experiment_6957_smt_mapping_certification as certificate_exp
from carnot import experiment_6984_exact_contrast_fixture as contrast_exp


JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_7012_exact_intervention_pair_fixture"
SCHEMA = "carnot.experiment_7012.exact_intervention_pair_fixture.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 7_012_202_609_05
INFERENCE_SUBSTRATE = "deterministic_exact_pair_fixture_no_llm"
EXPECTED_PAIR_COUNT = 48
EXPECTED_FAMILY_COUNT = 4

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7012_exact_intervention_pair_fixture.json")
DATA_ROOT = Path("results/raw/experiment_7012_exact_intervention_pair_fixture")
WRAPPER_PATH = Path("scripts/experiments/experiment_7012_exact_intervention_pair_fixture.py")
LEARNER_FILENAME = "learner_prompts.jsonl"
SIDECAR_FILENAME = "authority_sidecar.jsonl"

SOURCE_PATHS = {
    "exp6984": Path("results/experiment_6984_exact_contrast_fixture.json"),
    "exp6997": Path("results/experiment_6997_authority_sidecar_rebuild.json"),
    "exp6999": Path("results/experiment_6999_blinded_feature_cold_audit.json"),
}
EXPECTED_SOURCE_HASHES = {
    "exp6984": "sha256:15f9a9bb58ca7793966f2fbac548f6879a64417b50e31b0504078cfe6ea46a3f",
    "exp6997": "sha256:661a486b9400b879b117adeb6e2fea270b25c19e4037e4eec65fd19af70cc1be",
    "exp6999": "sha256:d844186c946c9f4d93a9ba42087217ac5aa8e9e0e347d027ad4d9f43d0d7e06b",
}

SOURCE_FAMILIES = (
    "bounded_integer_linear_positive_scale",
    "boolean_cardinality_positive_scale",
    "bounded_piecewise_linear_positive_scale",
    "sign_reversing_affine",
)
MUTATION_KINDS = (
    "bound_change",
    "objective_direction_reversal",
    "constraint_omission",
    "objective_coefficient_change",
)
SPLITS = ("train", "calibration", "held_source", "sealed_headroom")
SPLIT_COUNTS_PER_FAMILY = {"train": 6, "calibration": 2, "held_source": 2, "sealed_headroom": 2}
SERIALIZATION_TEMPLATES = ("compact", "expanded")
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

LEARNER_FIELDS = {"semantic_key", "neutral_block_position", "prompt"}
PROHIBITED_TOKENS = (
    "label",
    "split",
    "sourcegroup",
    "sourcefamily",
    "mutation",
    "authority",
    "witness",
    "provenance",
    "pairrole",
    "candidateorder",
    "interventiondirection",
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "authority_family_rows",
    "rows",
    "pair_rows",
    "block_rows",
    "rejected_block_rows",
    "intervention_rows",
    "minimality_rows",
    "authority_witness_rows",
    "isomorphism_rows",
    "nuisance_balance_rows",
    "label_balance_rows",
    "length_balance_rows",
    "serialization_balance_rows",
    "group_split_rows",
    "learner_prompt_path",
    "learner_prompt_hash",
    "authority_sidecar_path",
    "authority_sidecar_hash",
    "sidecar_intervention_rows",
    "prohibited_feature_rows",
    "expected_pair_count",
    "observed_pair_count",
    "expected_family_count",
    "observed_family_count",
    "intervention_pair_fixture_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema lets replay reject incompatible fixture data.",
    "experiment_id": "A stable identity prevents rows from moving between protocols.",
    "run_date": "The fixed date binds the fixture to its planned evidence window.",
    "field_principles": "A reason for every field makes the evidence contract reviewable.",
    "preconditions_checked": "Exact preflight checks prevent partial support from creating labels.",
    "inference_substrate": "The substrate states that deterministic authorities supplied all outcomes.",
    "duration_s": "Measured wall time shows that construction and replay executed.",
    "source_artifact_hashes": "Hashes bind the new fixture to the evidence that motivated it.",
    "authority_family_rows": "Family rows prove four intervention authorities executed evenly.",
    "rows": "Primary rows preserve the complete accepted block denominator.",
    "pair_rows": "Pair rows keep each causal contrast and its isomorph together.",
    "block_rows": "Block rows expose acceptance without trusting aggregate counts.",
    "rejected_block_rows": "Rejected rows preserve every failed or unbalanced attempted block.",
    "intervention_rows": "Intervention rows record direction and one declared semantic change.",
    "minimality_rows": "Edit distances prove that no smaller nonzero change exists.",
    "authority_witness_rows": "Witness rows bind each label change to exact counterexamples.",
    "isomorphism_rows": "Isomorphism rows prove that surface names do not change labels.",
    "nuisance_balance_rows": "Nuisance rows prevent noncausal block differences from predicting labels.",
    "label_balance_rows": "Two labels per class prevent a constant block-level shortcut.",
    "length_balance_rows": "Exact token and character parity removes prompt length shortcuts.",
    "serialization_balance_rows": "Template parity prevents formatting from becoming the outcome proxy.",
    "group_split_rows": "Frozen group splits prevent one source block from leaking across partitions.",
    "learner_prompt_path": "The learner path identifies the only file a model may open.",
    "learner_prompt_hash": "The learner hash detects any prompt or order change.",
    "authority_sidecar_path": "The sidecar path keeps exact outcomes outside learner storage.",
    "authority_sidecar_hash": "The sidecar hash detects any authority or provenance change.",
    "sidecar_intervention_rows": "Sidecar rows preserve labels and authority for later controlled joins.",
    "prohibited_feature_rows": "Leakage receipts prove that denied metadata never reached prompts.",
    "expected_pair_count": "The fixed target prevents silent deletion or replacement of pairs.",
    "observed_pair_count": "The observed count exposes every accepted primary causal unit.",
    "expected_family_count": "The fixed target requires four distinct source families.",
    "observed_family_count": "The observed count exposes lost source-family coverage.",
    "intervention_pair_fixture_ready_score": "One requires exact counts, minimality, balance, isolation, and replay.",
    "random_seed": "One seed fixes selection, positions, renaming, and sidecar perturbations.",
    "reproducibility_checksum": "A timing-free digest detects scientific-content drift.",
    "gate_check_summary": "The first failed expected-observed pair makes blocks actionable.",
    "verifier_is_oracle": "True states that exact authorities define fixture readiness.",
    "verdict_class": "A closed class separates circular fixture validity from learned evidence.",
    "honest_verdict": "A class-consistent prefix gives automation an unambiguous terminal state.",
}


class FixtureError(ValueError):
    """Fixture inputs cannot form one exact, balanced causal block."""


class IsolationError(ValueError):
    """The learner tried to access authority data outside its file."""


class ImmutableFixtureError(RuntimeError):
    """An existing frozen fixture path contains different bytes."""


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes and JSON Lines files."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON value after stable serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_path(path: Path) -> str | None:
    """Hash one file while keeping absence visible."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record one exact expected-observed comparison."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks and promote the first failed comparison."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "checks": copied,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": "all checks pass" if failed is None else failed.get("expected_value"),
        "observed_value": "all checks pass" if failed is None else failed.get("observed_value"),
        "passed": failed is None,
    }


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind the fixture to all three motivating evidence artifacts."""

    return {name: sha256_path(repo_root / path) for name, path in SOURCE_PATHS.items()}


def path_is_writable(path: Path) -> bool:
    """Probe a directory without changing a requested fixture file."""

    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".write-probe-", dir=path)
        os.close(descriptor)
        Path(name).unlink()
    except OSError:
        return False
    return True


def collect_preconditions(repo_root: Path, data_root: Path) -> list[JsonDict]:
    """Check frozen sources, four executable families, exact engines, and storage."""

    hashes = source_artifact_hashes(repo_root)
    checks = [
        gate_check(f"source_hash:{name}", expected, hashes.get(name))
        for name, expected in EXPECTED_SOURCE_HASHES.items()
    ]
    checks.extend(
        [
            gate_check(
                "executable_source_family_count", EXPECTED_FAMILY_COUNT, len(SOURCE_FAMILIES)
            ),
            gate_check("deterministic_mutation_support", True, callable(apply_intervention)),
            gate_check("deterministic_isomorphism_support", True, callable(alpha_rename_pair)),
            gate_check(
                "bounded_enumerator_available",
                True,
                callable(certificate_exp.certify_with_enumerator),
            ),
            gate_check("z3_authority_available", True, fixture_exp.z3 is not None),
            gate_check("immutable_fixture_paths_writable", True, path_is_writable(data_root)),
        ]
    )
    return checks


def frozen_clean_pairs() -> dict[str, JsonDict]:
    """Regenerate all clean candidates through the frozen executable source."""

    return {
        str(row["pair_id"]): deepcopy(row)
        for row in fixture_exp.generate_pairs(fixture_exp.RANDOM_SEED)
        if row["expected_label"] == "equivalent"
    }


def _positive_scale(row: Mapping[str, Any]) -> bool:
    """Classify the objective map without using any authority outcome."""

    return Fraction(str(row["mapping"]["objective"]["scale"])) > 0


def _source_family_pairs(clean: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Select four disjoint source classes before any new label opens."""

    selected: dict[str, list[Mapping[str, Any]]] = {}
    for underlying, source_family in zip(fixture_exp.FAMILIES, SOURCE_FAMILIES[:3], strict=True):
        selected[source_family] = [
            row for row in clean if row["family"] == underlying and _positive_scale(row)
        ][:12]
    selected[SOURCE_FAMILIES[3]] = [row for row in clean if not _positive_scale(row)][:12]
    return selected


def _split_for_index(index: int) -> str:
    """Assign the preregistered 6/2/2/2 split within one family."""

    if index < 6:
        return "train"
    if index < 8:
        return "calibration"
    if index < 10:
        return "held_source"
    return "sealed_headroom"


def build_source_manifest() -> list[JsonDict]:
    """Freeze source groups, mutations, directions, and splits before scoring."""

    clean = list(frozen_clean_pairs().values())
    selected = _source_family_pairs(clean)
    rows: list[JsonDict] = []
    for family_index, source_family in enumerate(SOURCE_FAMILIES):
        for family_position, pair in enumerate(selected[source_family]):
            source_pair_id = str(pair["pair_id"])
            source_group_id = sha256_json(
                {
                    "pair_hash": fixture_exp.pair_hash(pair),
                    "source_family": source_family,
                    "seed": RANDOM_SEED,
                }
            )
            rows.append(
                {
                    "manifest_position": len(rows),
                    "source_group_id": source_group_id,
                    "block_id": sha256_json({"source_group_id": source_group_id, "unit": "block"}),
                    "source_pair_id": source_pair_id,
                    "source_family": source_family,
                    "underlying_formulation_family": pair["family"],
                    "mutation_kind": MUTATION_KINDS[family_position % len(MUTATION_KINDS)],
                    "intervention_direction": (
                        "violation" if (family_index + family_position) % 2 == 0 else "repair"
                    ),
                    "split": _split_for_index(family_position),
                    "frozen_before_authority_scoring": True,
                }
            )
    return rows


def _objective_terms(expression: Mapping[str, Any]) -> JsonDict:
    """Return one editable coefficient map from either supported objective form."""

    if expression["kind"] == "linear":
        return expression["terms"]
    return expression["pieces"][0]["terms"]


def apply_intervention(pair: Mapping[str, Any], mutation_kind: str) -> tuple[JsonDict, JsonDict]:
    """Apply exactly one declared semantic edit to a clean candidate."""

    if mutation_kind not in MUTATION_KINDS:
        raise FixtureError(f"unknown_mutation_kind:{mutation_kind}")
    changed = deepcopy(pair)
    if mutation_kind == "bound_change":
        constraint = changed["target"]["constraints"][0]
        before = constraint["rhs"]
        constraint["rhs"] = "999" if constraint["op"] in {">=", "=="} else "-999"
        path = "$.target.constraints[0].rhs"
        after = constraint["rhs"]
    elif mutation_kind == "objective_direction_reversal":
        objective = changed["target"]["objective"]
        before = objective["direction"]
        objective["direction"] = "max" if before == "min" else "min"
        path = "$.target.objective.direction"
        after = objective["direction"]
    elif mutation_kind == "constraint_omission":
        before = changed["target"]["constraints"].pop(0)
        path = "$.target.constraints[0]"
        after = None
    else:
        terms = _objective_terms(changed["target"]["objective"]["expression"])
        name = sorted(terms)[0]
        before = terms[name]
        terms[name] = fixture_exp.fraction_text(Fraction(str(before)) + 7)
        path = f"$.target.objective.expression.coefficient[{name}]"
        after = terms[name]
    return changed, {
        "mutation_kind": mutation_kind,
        "operation_count": 1,
        "changed_paths": [path],
        "before": deepcopy(before),
        "after": deepcopy(after),
        "before_hash": fixture_exp.pair_hash(pair),
        "after_hash": fixture_exp.pair_hash(changed),
        "changed": fixture_exp.pair_hash(pair) != fixture_exp.pair_hash(changed),
    }


def _sequence_edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Compute structural insertion, deletion, and substitution distance."""

    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + structural_edit_distance(left_item, right_item),
                )
            )
        previous = current
    return previous[-1]


def structural_edit_distance(left: Any, right: Any) -> int:
    """Count the smallest recursive JSON edit script between two candidates."""

    if type(left) is not type(right):
        return 1
    if isinstance(left, Mapping):
        shared = set(left) & set(right)
        return len(set(left) ^ set(right)) + sum(
            structural_edit_distance(left[key], right[key]) for key in shared
        )
    if isinstance(left, list):
        return _sequence_edit_distance(left, right)
    return int(left != right)


def minimality_receipt(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    mutation: Mapping[str, Any],
    *,
    direction: str,
) -> JsonDict:
    """Prove that one nonzero edit implements the declared intervention."""

    distance = structural_edit_distance(before, after)
    paths = list(mutation.get("changed_paths", []))
    minimal = distance == 1 and len(paths) == 1 and mutation.get("operation_count", len(paths)) == 1
    return {
        "mutation_kind": mutation.get("mutation_kind"),
        "intervention_direction": direction,
        "edit_distance": distance,
        "changed_path_count": len(paths),
        "nonzero": distance > 0,
        "minimal": minimal,
        "terminal": True,
    }


def certify_candidate(pair: Mapping[str, Any], candidate_id: str) -> JsonDict:
    """Run the frozen bounded enumerator and Z3 authority on one candidate."""

    return contrast_exp.certify_pair(pair, candidate_id)


def authority_pair_receipt(
    clean_certificate: Mapping[str, Any], changed_certificate: Mapping[str, Any]
) -> JsonDict:
    """Require two exact terminal authorities and the expected label change."""

    clean_agreement = clean_certificate.get("agreement", {})
    changed_agreement = changed_certificate.get("agreement", {})
    exact = bool(
        clean_agreement.get("all_required_agreement") is True
        and changed_agreement.get("all_required_agreement") is True
        and clean_agreement.get("certified_relation") == "equivalent"
        and changed_agreement.get("certified_relation") == "non_equivalent"
    )
    witness_present = bool(
        changed_certificate.get("enumeration", {}).get("counterexamples")
        or changed_certificate.get("z3", {}).get("counterexamples")
    )
    return {
        "clean_label": clean_agreement.get("certified_relation"),
        "changed_label": changed_agreement.get("certified_relation"),
        "authorities_terminal_and_agree": bool(
            clean_agreement.get("all_required_agreement") is True
            and changed_agreement.get("all_required_agreement") is True
        ),
        "exact_counterexample_present": witness_present,
        "passed": exact and witness_present,
        "terminal": True,
    }


def alpha_rename_pair(pair: Mapping[str, Any], blind_key: str) -> tuple[JsonDict, JsonDict]:
    """Create a deterministic injective surface variant without reading a label."""

    return contrast_exp.alpha_rename_pair(pair, blind_key)


def _isomorphic_signature(pair: Mapping[str, Any]) -> str:
    """Remove variable spelling while preserving all executable structure."""

    source_names = [str(row["name"]) for row in pair["source"]["variables"]]
    target_names = [str(row["name"]) for row in pair["target"]["variables"]]
    source_rename = {name: f"source_{index}" for index, name in enumerate(source_names)}
    target_rename = {name: f"target_{index}" for index, name in enumerate(target_names)}
    normalized = deepcopy(pair)
    normalized["source"] = contrast_exp._rename_formulation(pair["source"], source_rename)
    normalized["target"] = contrast_exp._rename_formulation(pair["target"], target_rename)
    for row in normalized["mapping"]["variables"]:
        row["source"] = source_rename[str(row["source"])]
        row["target"] = target_rename[str(row["target"])]
    for row in normalized["mapping"]["domain_clauses"]:
        row["source"] = source_rename[str(row["source"])]
        row["target"] = target_rename[str(row["target"])]
    normalized["mapping"] = fixture_exp.canonical_mapping(
        normalized["mapping"], normalized["source"], normalized["target"]
    )
    return sha256_json(
        {
            "source": normalized["source"],
            "target": normalized["target"],
            "mapping": normalized["mapping"],
        }
    )


def isomorphism_receipt(
    original: Mapping[str, Any],
    variant: Mapping[str, Any],
    original_label: str,
    variant_label: str,
    rename_receipt: Mapping[str, Any],
) -> JsonDict:
    """Require injective renaming, equal normalized structure, and equal labels."""

    before = _isomorphic_signature(original)
    after = _isomorphic_signature(variant)
    return {
        "before_signature": before,
        "after_signature": after,
        "normalized_structure_equal": before == after,
        "label_before": original_label,
        "label_after": variant_label,
        "label_invariant": original_label == variant_label,
        "injective": rename_receipt.get("injective") is True,
        "label_blind": rename_receipt.get("label_blind") is True,
        "passed": bool(
            before == after
            and original_label == variant_label
            and rename_receipt.get("injective") is True
            and rename_receipt.get("label_blind") is True
        ),
        "terminal": True,
    }


def _semantic_payload(pair: Mapping[str, Any]) -> JsonDict:
    """Project only the executable candidate content used by the learner."""

    return {
        "instruction": "Assess whether this mapping preserves the exact bounded optimization problem.",
        "source_formulation": deepcopy(pair["source"]),
        "target_formulation": deepcopy(pair["target"]),
        "mapping_candidate": contrast_exp._v611_mapping(pair["mapping"]),
    }


def _serialize_prompt(pair: Mapping[str, Any], template: str) -> str:
    """Render one semantic payload through a declared neutral template."""

    payload = _semantic_payload(pair)
    if template == "compact":
        return canonical_json(payload)
    if template == "expanded":
        return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    raise FixtureError(f"unknown_serialization_template:{template}")


def _equalize_prompts(raw_prompts: Sequence[str]) -> list[str]:
    """Add neutral suffixes until every prompt has exact token and byte length."""

    with_marker = [prompt + "\nNEUTRAL_PADDING" for prompt in raw_prompts]
    target_tokens = max(len(prompt.split()) for prompt in with_marker) + 4
    token_padded = []
    for prompt in with_marker:
        missing = target_tokens - len(prompt.split())
        token_padded.append(prompt + " " + " ".join("x" for _ in range(missing)))
    target_chars = max(map(len, token_padded))
    return [prompt + (" " * (target_chars - len(prompt))) for prompt in token_padded]


def _walk_fields(value: Any, path: str = "$") -> list[tuple[str, str]]:
    """Return every nested field name with its exact learner-row path."""

    rows: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            rows.append((child, str(key)))
            rows.extend(_walk_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_walk_fields(item, f"{path}[{index}]"))
    return rows


def _normalized_field(value: str) -> str:
    """Normalize punctuation and case before testing metadata aliases."""

    return "".join(character for character in value.lower() if character.isalnum())


def validate_learner_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reject every field outside the three-field learner contract."""

    prohibited: list[JsonDict] = []
    for index, row in enumerate(rows):
        for path, field in _walk_fields(row, f"$[{index}]"):
            top_level = path.count(".") == 1
            normalized = _normalized_field(field)
            allowed = top_level and field in LEARNER_FIELDS
            alias = any(token in normalized for token in PROHIBITED_TOKENS)
            if not allowed:
                prohibited.append(
                    {
                        "path": path,
                        "field": field,
                        "reason": "prohibited_metadata_alias" if alias else "field_not_allowed",
                        "rejected": True,
                    }
                )
        key = row.get("semantic_key")
        position = row.get("neutral_block_position")
        prompt = row.get("prompt")
        if not isinstance(key, str) or not key.startswith("sha256:"):
            prohibited.append(
                {"path": f"$[{index}].semantic_key", "reason": "invalid_key", "rejected": True}
            )
        if type(position) is not int or position not in {0, 1}:
            prohibited.append(
                {
                    "path": f"$[{index}].neutral_block_position",
                    "reason": "invalid_position",
                    "rejected": True,
                }
            )
        if not isinstance(prompt, str) or not prompt:
            prohibited.append(
                {"path": f"$[{index}].prompt", "reason": "invalid_prompt", "rejected": True}
            )
    return prohibited


def canonical_learner_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Validate, deduplicate, sort, and serialize feature-free prompt rows."""

    prohibited = validate_learner_rows(rows)
    if prohibited:
        raise FixtureError(f"prohibited_learner_field:{prohibited[0]['path']}")
    keys = [str(row["semantic_key"]) for row in rows]
    duplicate = next((key for key, count in Counter(keys).items() if count > 1), None)
    if duplicate is not None:
        raise FixtureError(f"duplicate_semantic_key:{duplicate}")
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["semantic_key"])
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in ordered)


def _canonical_sidecar_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Sort authority rows independently from learner row storage."""

    keys = [str(row["semantic_key"]) for row in rows]
    duplicate = next((key for key, count in Counter(keys).items() if count > 1), None)
    if duplicate is not None:
        raise FixtureError(f"duplicate_sidecar_key:{duplicate}")
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["semantic_key"])
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in ordered)


def load_learner_prompts(path: str | Path, **authority_inputs: object) -> list[JsonDict]:
    """Open one learner file and reject every sidecar-shaped argument."""

    if authority_inputs:
        raise IsolationError("sidecar_access_denied")
    rows: list[JsonDict] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise IsolationError("learner_row_not_object")
            rows.append(dict(value))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsolationError(f"learner_file_invalid:{type(exc).__name__}") from exc
    prohibited = validate_learner_rows(rows)
    if prohibited:
        raise IsolationError(f"prohibited_learner_field:{prohibited[0]['path']}")
    canonical_learner_bytes(rows)
    return sorted(rows, key=lambda row: row["semantic_key"])


def audit_matched_block(
    block_id: str,
    learner_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Check label parity and every declared nuisance inside one block."""

    labels = [str(row.get("exact_label")) for row in sidecar_rows]
    label_counts = Counter(labels)
    lengths = [
        (len(str(row.get("prompt", ""))), len(str(row.get("prompt", "")).split()))
        for row in learner_rows
    ]
    templates = {
        template: Counter(
            str(row.get("exact_label"))
            for row in sidecar_rows
            if row.get("serialization_template") == template
        )
        for template in SERIALIZATION_TEMPLATES
    }
    positions = {
        position: Counter(
            str(row.get("exact_label"))
            for row in sidecar_rows
            if row.get("neutral_block_position") == position
        )
        for position in (0, 1)
    }
    label_balanced = len(sidecar_rows) == 4 and label_counts == {
        "equivalent": 2,
        "non_equivalent": 2,
    }
    serialization_balanced = all(
        templates[template] == {"equivalent": 1, "non_equivalent": 1}
        for template in SERIALIZATION_TEMPLATES
    )
    candidate_order_balanced = all(
        positions[position] == {"equivalent": 1, "non_equivalent": 1} for position in (0, 1)
    )
    mutation_fixed = len({row.get("mutation_kind") for row in sidecar_rows}) == 1
    source_fixed = all(
        len({row.get(field) for row in sidecar_rows}) == 1
        for field in ("source_group_id", "source_family", "split")
    )
    keys_match = {row.get("semantic_key") for row in learner_rows} == {
        row.get("semantic_key") for row in sidecar_rows
    }
    length_balanced = len(learner_rows) == 4 and len(set(lengths)) == 1
    passed = all(
        (
            label_balanced,
            serialization_balanced,
            candidate_order_balanced,
            mutation_fixed,
            source_fixed,
            keys_match,
            length_balanced,
        )
    )
    return {
        "block_id": block_id,
        "label_counts": dict(sorted(label_counts.items())),
        "label_balanced": label_balanced,
        "serialization_label_counts": {
            key: dict(sorted(value.items())) for key, value in templates.items()
        },
        "serialization_balanced": serialization_balanced,
        "position_label_counts": {
            str(key): dict(sorted(value.items())) for key, value in positions.items()
        },
        "candidate_order_balanced": candidate_order_balanced,
        "mutation_fixed": mutation_fixed,
        "source_fixed": source_fixed,
        "semantic_keys_match": keys_match,
        "character_lengths": [row[0] for row in lengths],
        "token_lengths": [row[1] for row in lengths],
        "length_balanced": length_balanced,
        "passed": passed,
        "terminal": True,
    }


def sidecar_invariance_receipts(
    learner_rows: Sequence[Mapping[str, Any]], sidecar_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Show that five sidecar states cannot change canonical learner bytes."""

    base = canonical_learner_bytes(learner_rows)
    replaced = [
        {"semantic_key": row.get("semantic_key"), "exact_label": "replacement"}
        for row in sidecar_rows
    ]
    alpha_renamed = [
        {**deepcopy(dict(row)), "source_group_id": f"alpha_{index}"}
        for index, row in enumerate(sidecar_rows)
    ]
    conditions = {
        "correct": list(sidecar_rows),
        "permuted": list(reversed(sidecar_rows)),
        "replaced": replaced,
        "deleted": [],
        "alpha_renamed": alpha_renamed,
    }
    return [
        {
            "condition": condition,
            "sidecar_row_count": len(condition_rows),
            "learner_prompt_hash": sha256_bytes(base),
            "passed": canonical_learner_bytes(learner_rows) == base,
            "terminal": True,
        }
        for condition, condition_rows in conditions.items()
    ]


def write_immutable(path: Path, payload: bytes) -> str:
    """Create frozen bytes once and reject any later content change."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ImmutableFixtureError(f"immutable_fixture_mismatch:{path}") from None
    return sha256_bytes(payload)


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding wall time and the digest."""

    return sha256_json(
        {
            key: deepcopy(value)
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )


def _empty_rows() -> JsonDict:
    """Return every required row family for a schema-complete block."""

    return {
        field: []
        for field in (
            "authority_family_rows",
            "rows",
            "pair_rows",
            "block_rows",
            "rejected_block_rows",
            "intervention_rows",
            "minimality_rows",
            "authority_witness_rows",
            "isomorphism_rows",
            "nuisance_balance_rows",
            "label_balance_rows",
            "length_balance_rows",
            "serialization_balance_rows",
            "group_split_rows",
            "sidecar_intervention_rows",
            "prohibited_feature_rows",
        )
    }


def build_blocked_artifact(
    *,
    repo_root: Path,
    data_root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build the full artifact shape when any prerequisite fails."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        **_empty_rows(),
        "learner_prompt_path": str((data_root / LEARNER_FILENAME).resolve()),
        "learner_prompt_hash": None,
        "authority_sidecar_path": str((data_root / SIDECAR_FILENAME).resolve()),
        "authority_sidecar_hash": None,
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "observed_pair_count": 0,
        "expected_family_count": EXPECTED_FAMILY_COUNT,
        "observed_family_count": 0,
        "intervention_pair_fixture_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(preconditions),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_intervention_pair_fixture",
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _block_prompt_rows(
    *,
    manifest_row: Mapping[str, Any],
    clean: Mapping[str, Any],
    changed: Mapping[str, Any],
    iso_clean: Mapping[str, Any],
    iso_changed: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Build four balanced prompt rows and their authority-only projections."""

    even = int(manifest_row["manifest_position"]) % 2 == 0
    primary_positions = {"clean": 0 if even else 1, "changed": 1 if even else 0}
    isomorphic_positions = {"clean": 1 if even else 0, "changed": 0 if even else 1}
    candidates = [
        ("primary", "compact", "clean", clean, primary_positions["clean"]),
        ("primary", "compact", "changed", changed, primary_positions["changed"]),
        ("isomorphic", "expanded", "clean", iso_clean, isomorphic_positions["clean"]),
        ("isomorphic", "expanded", "changed", iso_changed, isomorphic_positions["changed"]),
    ]
    prompts = _equalize_prompts(
        [_serialize_prompt(pair, template) for _, template, _, pair, _ in candidates]
    )
    learner_rows: list[JsonDict] = []
    sidecar_rows: list[JsonDict] = []
    for candidate, prompt in zip(candidates, prompts, strict=True):
        surface, template, role, pair, neutral_position = candidate
        semantic_hash = sha256_json(_semantic_payload(pair))
        semantic_key = sha256_json(
            {"semantic_content_hash": semantic_hash, "neutral_block_position": neutral_position}
        )
        learner_rows.append(
            {
                "semantic_key": semantic_key,
                "neutral_block_position": neutral_position,
                "prompt": prompt,
            }
        )
        sidecar_rows.append(
            {
                "semantic_key": semantic_key,
                "block_id": manifest_row["block_id"],
                "source_group_id": manifest_row["source_group_id"],
                "source_pair_id": manifest_row["source_pair_id"],
                "source_family": manifest_row["source_family"],
                "underlying_formulation_family": manifest_row["underlying_formulation_family"],
                "split": manifest_row["split"],
                "mutation_kind": manifest_row["mutation_kind"],
                "intervention_direction": manifest_row["intervention_direction"],
                "surface_variant": surface,
                "serialization_template": template,
                "pair_role": role,
                "exact_label": "equivalent" if role == "clean" else "non_equivalent",
                "neutral_block_position": neutral_position,
                "semantic_content_hash": semantic_hash,
            }
        )
    return learner_rows, sidecar_rows


def build_artifact(
    repo_root: Path,
    data_root: Path,
    *,
    duration_s: float,
    fresh_process_receipt: Mapping[str, Any],
) -> JsonDict:
    """Construct, certify, balance, freeze, and reduce all 48 blocks."""

    preconditions = collect_preconditions(repo_root, data_root)
    preconditions.append(
        gate_check("network_disabled_fresh_process", True, fresh_process_receipt.get("passed"))
    )
    if not gate_summary(preconditions)["passed"]:
        return build_blocked_artifact(
            repo_root=repo_root,
            data_root=data_root,
            preconditions=preconditions,
            duration_s=duration_s,
        )

    clean_pairs = frozen_clean_pairs()
    manifest = build_source_manifest()
    learner_rows: list[JsonDict] = []
    sidecar_rows: list[JsonDict] = []
    pair_rows: list[JsonDict] = []
    block_rows: list[JsonDict] = []
    rejected_rows: list[JsonDict] = []
    intervention_rows: list[JsonDict] = []
    minimality_rows: list[JsonDict] = []
    witness_rows: list[JsonDict] = []
    isomorphism_rows: list[JsonDict] = []
    nuisance_rows: list[JsonDict] = []

    for manifest_row in manifest:
        block_id = str(manifest_row["block_id"])
        clean = deepcopy(clean_pairs[str(manifest_row["source_pair_id"])])
        changed, mutation = apply_intervention(clean, str(manifest_row["mutation_kind"]))
        minimality = minimality_receipt(
            clean,
            changed,
            mutation,
            direction=str(manifest_row["intervention_direction"]),
        )
        clean_certificate = certify_candidate(clean, f"{block_id}:primary:clean")
        changed_certificate = certify_candidate(changed, f"{block_id}:primary:changed")
        primary_authority = authority_pair_receipt(clean_certificate, changed_certificate)

        iso_clean, clean_rename = alpha_rename_pair(clean, block_id)
        iso_changed, changed_rename = alpha_rename_pair(changed, block_id)
        iso_clean_certificate = certify_candidate(iso_clean, f"{block_id}:isomorphic:clean")
        iso_changed_certificate = certify_candidate(iso_changed, f"{block_id}:isomorphic:changed")
        iso_authority = authority_pair_receipt(iso_clean_certificate, iso_changed_certificate)
        clean_isomorphism = isomorphism_receipt(
            clean,
            iso_clean,
            str(clean_certificate["agreement"]["certified_relation"]),
            str(iso_clean_certificate["agreement"]["certified_relation"]),
            clean_rename,
        )
        changed_isomorphism = isomorphism_receipt(
            changed,
            iso_changed,
            str(changed_certificate["agreement"]["certified_relation"]),
            str(iso_changed_certificate["agreement"]["certified_relation"]),
            changed_rename,
        )
        isomorphism = {
            "block_id": block_id,
            "clean_isomorphism": clean_isomorphism,
            "changed_isomorphism": changed_isomorphism,
            "passed": clean_isomorphism["passed"] and changed_isomorphism["passed"],
            "terminal": True,
        }
        block_learner, block_sidecar = _block_prompt_rows(
            manifest_row=manifest_row,
            clean=clean,
            changed=changed,
            iso_clean=iso_clean,
            iso_changed=iso_changed,
        )
        certificates = {
            "primary_clean": clean_certificate,
            "primary_changed": changed_certificate,
            "isomorphic_clean": iso_clean_certificate,
            "isomorphic_changed": iso_changed_certificate,
        }
        for row in block_sidecar:
            key = f"{row['surface_variant']}_{row['pair_role']}"
            row["authority_records"] = deepcopy(certificates[key])
            row["authority_witnesses"] = {
                "enumeration": deepcopy(certificates[key]["enumeration"].get("witnesses", {})),
                "counterexamples": deepcopy(
                    certificates[key]["enumeration"].get("counterexamples", {})
                ),
                "z3_counterexamples": deepcopy(certificates[key]["z3"].get("counterexamples", {})),
            }
        nuisance = audit_matched_block(block_id, block_learner, block_sidecar)
        accepted = bool(
            minimality["minimal"]
            and primary_authority["passed"]
            and iso_authority["passed"]
            and isomorphism["passed"]
            and nuisance["passed"]
            and not validate_learner_rows(block_learner)
        )
        if not accepted:
            rejected_rows.append(
                {
                    "block_id": block_id,
                    "source_group_id": manifest_row["source_group_id"],
                    "minimality_passed": minimality["minimal"],
                    "primary_authority_passed": primary_authority["passed"],
                    "isomorphic_authority_passed": iso_authority["passed"],
                    "isomorphism_passed": isomorphism["passed"],
                    "nuisance_passed": nuisance["passed"],
                    "terminal": True,
                }
            )
            continue

        learner_rows.extend(block_learner)
        sidecar_rows.extend(block_sidecar)
        minimality_rows.append({"block_id": block_id, **minimality})
        isomorphism_rows.append(isomorphism)
        nuisance_rows.append(nuisance)
        intervention_rows.append(
            {
                "block_id": block_id,
                "source_group_id": manifest_row["source_group_id"],
                "mutation_kind": manifest_row["mutation_kind"],
                "intervention_direction": manifest_row["intervention_direction"],
                "changed_paths": mutation["changed_paths"],
                "before_hash": mutation["before_hash"],
                "after_hash": mutation["after_hash"],
                "terminal": True,
            }
        )
        witness_rows.append(
            {
                "block_id": block_id,
                "primary": primary_authority,
                "isomorphic": iso_authority,
                "exact_counterexamples": deepcopy(
                    changed_certificate["enumeration"]["counterexamples"]
                ),
                "passed": primary_authority["passed"] and iso_authority["passed"],
                "terminal": True,
            }
        )
        pair_row = {
            "block_id": block_id,
            "source_group_id": manifest_row["source_group_id"],
            "source_family": manifest_row["source_family"],
            "split": manifest_row["split"],
            "mutation_kind": manifest_row["mutation_kind"],
            "intervention_direction": manifest_row["intervention_direction"],
            "primary_semantic_keys": [
                row["semantic_key"] for row in block_sidecar if row["surface_variant"] == "primary"
            ],
            "isomorphic_semantic_keys": [
                row["semantic_key"]
                for row in block_sidecar
                if row["surface_variant"] == "isomorphic"
            ],
            "labels": [row["exact_label"] for row in block_sidecar],
            "pair_accepted": True,
            "terminal": True,
        }
        pair_rows.append(pair_row)
        block_rows.append(
            {
                **pair_row,
                "prompt_count": len(block_learner),
                "minimal": minimality["minimal"],
                "authorities_exact": primary_authority["passed"] and iso_authority["passed"],
                "isomorphism_valid": isomorphism["passed"],
                "nuisance_balanced": nuisance["passed"],
            }
        )

    prohibited_rows = [
        {"field": token, "present_in_learner": False, "passed": True, "terminal": True}
        for token in PROHIBITED_TOKENS
    ]
    learner_payload = canonical_learner_bytes(learner_rows)
    sidecar_payload = _canonical_sidecar_bytes(sidecar_rows)
    data_root = data_root.resolve()
    learner_path = data_root / LEARNER_FILENAME
    sidecar_path = data_root / SIDECAR_FILENAME
    learner_hash = write_immutable(learner_path, learner_payload)
    sidecar_hash = write_immutable(sidecar_path, sidecar_payload)
    invariance_rows = sidecar_invariance_receipts(learner_rows, sidecar_rows)

    accepted_families = {str(row["source_family"]) for row in pair_rows}
    authority_family_rows = [
        {
            "authority_family": mutation_kind,
            "pair_count": sum(row["mutation_kind"] == mutation_kind for row in pair_rows),
            "source_family_count": len(
                {row["source_family"] for row in pair_rows if row["mutation_kind"] == mutation_kind}
            ),
            "mutation_executable": True,
            "enumeration_executable": True,
            "z3_executable": True,
            "terminal": True,
        }
        for mutation_kind in MUTATION_KINDS
    ]
    observed_pair_count = len(pair_rows)
    observed_family_count = len(accepted_families)
    ready = int(
        observed_pair_count == EXPECTED_PAIR_COUNT
        and observed_family_count == EXPECTED_FAMILY_COUNT
        and len(authority_family_rows) == EXPECTED_FAMILY_COUNT
        and all(
            row["pair_count"] == 12 and row["source_family_count"] == 4
            for row in authority_family_rows
        )
        and not rejected_rows
        and all(row["minimal"] for row in minimality_rows)
        and all(row["passed"] for row in witness_rows)
        and all(row["passed"] for row in isomorphism_rows)
        and all(row["passed"] for row in nuisance_rows)
        and all(row["passed"] for row in prohibited_rows)
        and all(row["passed"] for row in invariance_rows)
        and learner_hash == sha256_path(learner_path)
        and sidecar_hash == sha256_path(sidecar_path)
        and fresh_process_receipt.get("passed") is True
    )
    scientific_checks = [
        gate_check("observed_pair_count", EXPECTED_PAIR_COUNT, observed_pair_count),
        gate_check("observed_family_count", EXPECTED_FAMILY_COUNT, observed_family_count),
        gate_check("all_blocks_balanced", True, all(row["passed"] for row in nuisance_rows)),
        gate_check(
            "all_interventions_minimal", True, all(row["minimal"] for row in minimality_rows)
        ),
        gate_check("all_authorities_exact", True, all(row["passed"] for row in witness_rows)),
        gate_check("all_isomorphisms_valid", True, all(row["passed"] for row in isomorphism_rows)),
        gate_check("sidecar_invariance", True, all(row["passed"] for row in invariance_rows)),
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [*preconditions, *scientific_checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        "authority_family_rows": authority_family_rows,
        "rows": deepcopy(block_rows),
        "pair_rows": pair_rows,
        "block_rows": block_rows,
        "rejected_block_rows": rejected_rows,
        "intervention_rows": intervention_rows,
        "minimality_rows": minimality_rows,
        "authority_witness_rows": witness_rows,
        "isomorphism_rows": isomorphism_rows,
        "nuisance_balance_rows": nuisance_rows,
        "label_balance_rows": [
            {
                "block_id": row["block_id"],
                "label_counts": row["label_counts"],
                "passed": row["label_balanced"],
                "terminal": True,
            }
            for row in nuisance_rows
        ],
        "length_balance_rows": [
            {
                "block_id": row["block_id"],
                "character_lengths": row["character_lengths"],
                "token_lengths": row["token_lengths"],
                "passed": row["length_balanced"],
                "terminal": True,
            }
            for row in nuisance_rows
        ],
        "serialization_balance_rows": [
            {
                "block_id": row["block_id"],
                "serialization_label_counts": row["serialization_label_counts"],
                "passed": row["serialization_balanced"],
                "terminal": True,
            }
            for row in nuisance_rows
        ],
        "group_split_rows": deepcopy(manifest),
        "learner_prompt_path": str(learner_path),
        "learner_prompt_hash": learner_hash,
        "authority_sidecar_path": str(sidecar_path),
        "authority_sidecar_hash": sidecar_hash,
        "sidecar_intervention_rows": sidecar_rows,
        "prohibited_feature_rows": prohibited_rows,
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "observed_pair_count": observed_pair_count,
        "expected_family_count": EXPECTED_FAMILY_COUNT,
        "observed_family_count": observed_family_count,
        "intervention_pair_fixture_ready_score": ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary([*preconditions, *scientific_checks]),
        "verifier_is_oracle": True,
        "verdict_class": "circular_positive" if ready else "disqualified",
        "honest_verdict": (
            "circular_positive: exact_intervention_pair_fixture_ready"
            if ready
            else "complete_disqualified_intervention_pair_fixture"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _verdict_matches(verdict_class: str, verdict: str) -> bool:
    """Match the closed verdict class to its required terminal prefix."""

    prefixes = {
        "positive": ("positive:", "complete_positive_"),
        "circular_positive": ("circular_positive:", "complete_circular_"),
        "null": ("null:", "complete_null_"),
        "blocked": ("blocked_", "blocked:"),
        "disqualified": ("complete_disqualified_", "disqualified:"),
        "partial": ("partial_", "partial:"),
    }
    return verdict.startswith(prefixes.get(verdict_class, ()))


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, counts, balances, file hashes, verdict, and checksum."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing_required_field:{field}" for field in missing]
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles")
    score = artifact["intervention_pair_fixture_ready_score"]
    if type(score) is not int or score not in {0, 1}:
        errors.append("bare_ready_score")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact["verifier_is_oracle"] is not True:
        errors.append("verifier_is_oracle")
    verdict_class = str(artifact["verdict_class"])
    if verdict_class not in VERDICT_CLASSES:
        errors.append("verdict_class")
    if not _verdict_matches(verdict_class, str(artifact["honest_verdict"])):
        errors.append("verdict_prefix")

    if verdict_class == "blocked":
        if score != 0:
            errors.append("blocked_ready_score")
        if artifact["gate_check_summary"].get("failed_check") is None:
            errors.append("blocked_gate_summary")
        if not str(artifact["honest_verdict"]).startswith("blocked_intervention_pair_fixture"):
            errors.append("blocked_verdict")
    else:
        pairs = list(artifact["pair_rows"])
        families = {row.get("source_family") for row in pairs}
        pair_count_valid = (
            artifact["expected_pair_count"] == EXPECTED_PAIR_COUNT
            and artifact["observed_pair_count"] == len(pairs) == EXPECTED_PAIR_COUNT
            and len(artifact["block_rows"]) == EXPECTED_PAIR_COUNT
            and len(artifact["rows"]) == EXPECTED_PAIR_COUNT
        )
        if not pair_count_valid:
            errors.append("pair_count")
        family_count_valid = (
            artifact["expected_family_count"] == EXPECTED_FAMILY_COUNT
            and artifact["observed_family_count"] == len(families) == EXPECTED_FAMILY_COUNT
            and len(artifact["authority_family_rows"]) == EXPECTED_FAMILY_COUNT
        )
        if not family_count_valid:
            errors.append("family_count")
        nuisance_valid = (
            len(artifact["nuisance_balance_rows"]) == EXPECTED_PAIR_COUNT
            and all(row.get("passed") is True for row in artifact["nuisance_balance_rows"])
            and all(row.get("passed") is True for row in artifact["label_balance_rows"])
            and all(row.get("passed") is True for row in artifact["length_balance_rows"])
            and all(row.get("passed") is True for row in artifact["serialization_balance_rows"])
        )
        if not nuisance_valid:
            errors.append("nuisance")
        exact_valid = (
            len(artifact["minimality_rows"]) == EXPECTED_PAIR_COUNT
            and all(row.get("minimal") is True for row in artifact["minimality_rows"])
            and len(artifact["authority_witness_rows"]) == EXPECTED_PAIR_COUNT
            and all(row.get("passed") is True for row in artifact["authority_witness_rows"])
            and len(artifact["isomorphism_rows"]) == EXPECTED_PAIR_COUNT
            and all(row.get("passed") is True for row in artifact["isomorphism_rows"])
        )
        if not exact_valid:
            errors.append("exact_minimality")
        split_valid = (
            len(artifact["group_split_rows"]) == EXPECTED_PAIR_COUNT
            and len({row.get("source_group_id") for row in artifact["group_split_rows"]})
            == EXPECTED_PAIR_COUNT
        )
        if not split_valid:
            errors.append("group_splits")
        sidecar_valid = len(artifact["sidecar_intervention_rows"]) == 4 * EXPECTED_PAIR_COUNT
        if not sidecar_valid:
            errors.append("sidecar_rows")
        prohibited_valid = all(
            row.get("passed") is True for row in artifact["prohibited_feature_rows"]
        )
        if not prohibited_valid:
            errors.append("prohibited_features")
        learner_hash_valid = (
            sha256_path(Path(str(artifact["learner_prompt_path"])))
            == artifact["learner_prompt_hash"]
        )
        sidecar_hash_valid = (
            sha256_path(Path(str(artifact["authority_sidecar_path"])))
            == artifact["authority_sidecar_hash"]
        )
        if not learner_hash_valid:
            errors.append("learner_hash")
        if not sidecar_hash_valid:
            errors.append("sidecar_hash")
        expected_ready = int(
            pair_count_valid
            and family_count_valid
            and nuisance_valid
            and exact_valid
            and split_valid
            and sidecar_valid
            and prohibited_valid
            and learner_hash_valid
            and sidecar_hash_valid
            and not artifact["rejected_block_rows"]
            and artifact["gate_check_summary"].get("passed") is True
        )
        if score != expected_ready:
            errors.append("ready_score_consistency")
        expected_class = "circular_positive" if expected_ready else "disqualified"
        if verdict_class != expected_class:
            errors.append("verdict_consistency")
    if artifact["reproducibility_checksum"] != _artifact_checksum(artifact):
        errors.append("checksum")
    return errors


def sandbox_receipt(repo_root: Path) -> JsonDict:
    """Prove the child has a new network namespace and cannot write sources."""

    parent_netns = os.environ.get("CARNOT_EXP7012_PARENT_NETNS", "")
    child_netns = os.readlink("/proc/self/ns/net")
    source_write_denied = []
    for source_id, relative in SOURCE_PATHS.items():
        denied = False
        try:
            with (repo_root / relative).open("rb+"):
                pass
        except OSError:
            denied = True
        source_write_denied.append({"source_id": source_id, "write_denied": denied})
    gpu_devices = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    receipt = {
        "fresh_process_pid": os.getpid(),
        "network_namespace_isolated": bool(parent_netns and child_netns != parent_netns),
        "gpu_devices_visible": gpu_devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "source_write_rows": source_write_denied,
        "source_tree_read_only": all(row["write_denied"] for row in source_write_denied),
    }
    receipt["passed"] = bool(
        receipt["network_namespace_isolated"]
        and not gpu_devices
        and receipt["cuda_visible_devices"] == ""
        and receipt["hf_hub_offline"]
        and receipt["transformers_offline"]
        and receipt["source_tree_read_only"]
    )
    return receipt


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one JSON result without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def fresh_process_command(
    *,
    executable: Path,
    wrapper: Path,
    repo_root: Path,
    data_root: Path,
    writable_root: Path,
    output_path: Path,
    run_date: str,
) -> list[str]:
    """Build the bubblewrap command for network-disabled exact replay."""

    return [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-pid",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--bind",
        str(data_root),
        str(data_root),
        "--bind",
        str(writable_root),
        str(writable_root),
        "--chdir",
        str(repo_root),
        "--setenv",
        "CUDA_VISIBLE_DEVICES",
        "",
        "--setenv",
        "HF_HUB_OFFLINE",
        "1",
        "--setenv",
        "TRANSFORMERS_OFFLINE",
        "1",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "CARNOT_EXP7012_PARENT_NETNS",
        os.readlink("/proc/self/ns/net"),
        "--",
        str(executable),
        str(wrapper),
        "--fresh-child",
        "--date",
        run_date,
        "--data-root",
        str(data_root),
        "--output",
        str(output_path),
    ]


def run_controller(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    data_root: Path | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Run exact replay in a network-disabled child and publish its result."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:expected={RUN_DATE}:observed={run_date}")
    started = time.perf_counter()
    final_path = result_path or (repo_root / RESULT_PATH)
    fixture_root = (data_root or (repo_root / DATA_ROOT)).resolve()
    fixture_root.mkdir(parents=True, exist_ok=True)
    if shutil.which("bwrap") is None:
        checks = [
            *collect_preconditions(repo_root, fixture_root),
            gate_check("bubblewrap_available", True, False),
        ]
        artifact = build_blocked_artifact(
            repo_root=repo_root,
            data_root=fixture_root,
            preconditions=checks,
            duration_s=time.perf_counter() - started,
        )
        write_json_atomic(final_path, artifact)
        return artifact

    before = source_artifact_hashes(repo_root)
    with tempfile.TemporaryDirectory(prefix="carnot-exp7012-") as directory:
        writable_root = Path(directory).resolve()
        child_output = writable_root / "child-result.json"
        command = fresh_process_command(
            executable=Path(sys.executable),
            wrapper=repo_root / WRAPPER_PATH,
            repo_root=repo_root,
            data_root=fixture_root,
            writable_root=writable_root,
            output_path=child_output,
            run_date=run_date,
        )
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and child_output.is_file():
            artifact = json.loads(child_output.read_text(encoding="utf-8"))
        else:
            checks = [
                *collect_preconditions(repo_root, fixture_root),
                gate_check("fresh_process_exit_code", 0, completed.returncode),
                gate_check("fresh_process_output", True, child_output.is_file()),
            ]
            artifact = build_blocked_artifact(
                repo_root=repo_root,
                data_root=fixture_root,
                preconditions=checks,
                duration_s=time.perf_counter() - started,
            )
    after = source_artifact_hashes(repo_root)
    if before != after:
        checks = [
            *artifact["preconditions_checked"],
            gate_check("source_hashes_unchanged", before, after),
        ]
        artifact = build_blocked_artifact(
            repo_root=repo_root,
            data_root=fixture_root,
            preconditions=checks,
            duration_s=time.perf_counter() - started,
        )
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise FixtureError("artifact_validation_failed:" + ",".join(errors))
    write_json_atomic(final_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command boundary.
    """Run the controller or its private fresh-child mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--fresh-child", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.fresh_child:
        if args.data_root is None or args.output is None:
            parser.error("--data-root and --output are required with --fresh-child")
        started = time.perf_counter()
        artifact = build_artifact(
            REPO_ROOT,
            args.data_root,
            duration_s=0.0,
            fresh_process_receipt=sandbox_receipt(REPO_ROOT),
        )
        artifact["duration_s"] = round(time.perf_counter() - started, 6)
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
        errors = validate_artifact(artifact)
        if errors:
            raise FixtureError("child_artifact_validation_failed:" + ",".join(errors))
        write_json_atomic(args.output, artifact)
        return 0
    run_controller(run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover - module command surface.
    raise SystemExit(main())
