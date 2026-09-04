"""Replay certified candidate selection without importing its producers.

The audit binds source bytes before it parses any source JSON. It then rebuilds
the exact certificate and every selector from raw rows and saved tensor state.
This order prevents a stored headline from becoming evidence for itself.

Spec refs: REQ-VERIFY-6960 and SCENARIO-VERIFY-6960-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
import torch.nn.functional as torch_functional
import z3


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260904"
EXPERIMENT_ID = 6960
SCHEMA = "carnot.exp6960.certified_selection_cold_audit.v1"
INFERENCE_SUBSTRATE = "fresh_process_exact_candidate_selection_replay"
RANDOM_SEED = 6_960_202_609_04
BOOTSTRAP_SAMPLES = 10_000
EXPECTED_CANDIDATE_COUNT = 162
EXPECTED_GROUP_COUNT = 54

MODULE_PATH = Path("python/carnot/experiment_6960_certified_selection_cold_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6960_certified_selection_cold_audit.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULT_PATH = Path("results/experiment_6960_certified_selection_cold_audit.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_6958_convex_factor_energy_canary")

SOURCE_PATHS = {
    "proposal_bank": Path("results/experiment_6956_three_family_reformulation_bank.json"),
    "certificates": Path("results/experiment_6957_smt_mapping_certification.json"),
    "certificate_checkpoint": Path(
        "results/checkpoints/experiment_6957_smt_mapping_certification_inputs.json"
    ),
    "energy_canary": Path("results/experiment_6958_convex_factor_energy_canary.json"),
    "upstream_selection": Path("results/experiment_6959_certified_energy_selection.json"),
}
EXPECTED_SOURCE_HASHES = {
    "proposal_bank": "sha256:caf07266aa26d1a9a324339554efc1d07ab23328aa405af848c220c7f2f98ede",
    "certificates": "sha256:23d50dba2e830f97af3f5bed1e8bc53609af5ff74c89e9b2f32fe838c1ef8322",
    "certificate_checkpoint": "sha256:922036219d4258436bf709a895e3d0b41b2b12f91c18766f1a702e75f2cc0e77",
    "energy_canary": "sha256:8e595307423c690e3b2acbde69d957f9f60168aa9db13743304e10c0ee2414c9",
    "upstream_selection": "sha256:d89a7d94ab508ac3c562fe2a5ff41496ef09a3ed9f4501e414a29ab828c96aa3",
}

EXPECTED_FACTOR_HASHES = {
    (
        "input_convex_factor_sum",
        69580,
    ): "sha256:bf7891db11c87ab64215a6e187ef65f4a09609838d943bb6321099783a9e2a33",
    (
        "input_convex_factor_sum",
        69581,
    ): "sha256:886277df05d283cc91b967d738638f9022bf274ddf4795b9221953570f3bc567",
    (
        "input_convex_factor_sum",
        69582,
    ): "sha256:80389356ea58837cacbb1f90f818820e520110794583387fcef78dd9281547ba",
    (
        "unconstrained_mlp_factor_sum",
        69580,
    ): "sha256:fd08c007d90549bd850679f35f70671d84ad5d933afecd875ca2d29ed15bfafb",
    (
        "unconstrained_mlp_factor_sum",
        69581,
    ): "sha256:ffb9184eb92af80623b8b92e374ae27264ba1506aabae8efabcb1efae0bcfc6e",
    (
        "unconstrained_mlp_factor_sum",
        69582,
    ): "sha256:8aa4814076381ecf33937eafef8948e2999613e1373d198c32fe2c7b28bf1604",
    (
        "linear_factor_score",
        69580,
    ): "sha256:94708669b987e5eef903a2851d6061405b137f2a68b6e70d53405705af0f9885",
    (
        "linear_factor_score",
        69581,
    ): "sha256:4d4c871fdc67d20949d391b249a4ec7ba8d57d6396c18f2a758ba777ff3d295e",
    (
        "linear_factor_score",
        69582,
    ): "sha256:fd3566a087ada7c2527e328bb627ab4818f04071723257ee4492744ff1ddaea5",
    (
        "shuffled_label_convex_energy",
        69580,
    ): "sha256:0c2540b139eb94901de0f7934b221836bb76a583e9b131ff625fef994d238044",
    (
        "shuffled_label_convex_energy",
        69581,
    ): "sha256:376336e643943596c2279041766fdb70563c67ff66df231fb10d7b721368e8e9",
    (
        "shuffled_label_convex_energy",
        69582,
    ): "sha256:b3c8a5a1fd810c070b0ce8c958c3305131812032f496bd7135f62a7f9dde9fff",
}

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
CHECKPOINT_ARM_NAMES = {
    ARM_CONVEX: "input_convex_factor_sum",
    ARM_MLP: "unconstrained_mlp_factor_sum",
    ARM_LINEAR: "linear_factor_score",
    ARM_SHUFFLED: "shuffled_label_convex_energy",
}

FORBIDDEN_SELECTOR_FIELDS = {
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
}

REQUIRED_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "audit_rows",
    "hash_rows",
    "certificate_replay_rows",
    "checkpoint_reload_rows",
    "arm_recompute_rows",
    "candidate_order_rows",
    "proposal_budget_rows",
    "label_isolation_rows",
    "tie_policy_rows",
    "selection_rows",
    "headroom_rows",
    "family_rows",
    "paired_metric_rows",
    "confidence_interval_rows",
    "aggregate_consistency_rows",
    "contradiction_report_rows",
    "random_seed",
    "reproducibility_checksum",
    "certified_selection_audit_complete_score",
    "audited_certified_energy_positive_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes the audit contract reviewable.",
    "preconditions_checked": "Pinned prerequisites stop replay on incomplete frozen evidence.",
    "inference_substrate": "The substrate states that a clean process performed exact replay.",
    "duration_s": "Measured wall time proves the audit executed instead of copying a schema.",
    "source_artifact_hashes": "Byte hashes bind the audit to the frozen evidence chain.",
    "rows": "Primary per-group rows let readers rebuild the selection result.",
    "audit_rows": "Terminal checks separate completed audit work from a favorable result.",
    "hash_rows": "Per-file receipts expose source, raw-text, and checkpoint drift.",
    "certificate_replay_rows": "Dual-engine rows make exact labels independently testable.",
    "checkpoint_reload_rows": "Reload receipts prove scores came from durable tensor state.",
    "arm_recompute_rows": "Candidate-arm rows expose every independently recomputed score.",
    "candidate_order_rows": "Forward and reverse trials detect input-order dependence.",
    "proposal_budget_rows": "Budget rows keep all candidates and groups in the denominator.",
    "label_isolation_rows": "Nested path audits keep exact outcomes outside selector inputs.",
    "tie_policy_rows": "Tie rows expose deterministic prompt-order decisions.",
    "selection_rows": "Frozen selected IDs connect scores to later exact evaluation.",
    "headroom_rows": "Per-group headroom prevents impossible gains from becoming credit.",
    "family_rows": "Group-weighted strata show where selection effects occur.",
    "paired_metric_rows": "Same-group differences prevent unpaired accuracy claims.",
    "confidence_interval_rows": "Pair-level bootstrap bounds the top-one effect uncertainty.",
    "aggregate_consistency_rows": "Stored headlines must equal values rebuilt from rows.",
    "contradiction_report_rows": "Each disagreement remains visible and disqualifies the claim.",
    "random_seed": "A fixed seed makes pair bootstrap draws reproducible.",
    "reproducibility_checksum": "A timing-free digest detects stable evidence drift.",
    "certified_selection_audit_complete_score": "Completion requires a terminal row for each audit.",
    "audited_certified_energy_positive_score": "Positive credit requires raw gates and upstream positive status.",
    "gate_check_summary": "Expected and observed values make every failure actionable.",
    "verifier_is_oracle": "False keeps the audit from posing as a deployed selector.",
    "verdict_class": "A closed class prevents a contradiction from reading as success.",
    "honest_verdict": "A stable terminal phrase states the audited outcome.",
}

_TOP_LEVEL_KEYS = {"mapping", "confidence", "rationale"}
_MAPPING_KEYS = {"schema_version", "variables", "domain_clauses", "objective", "claimed_relation"}
_VARIABLE_KEYS = {"source", "target", "scale", "offset"}
_DOMAIN_KEYS = {
    "source",
    "target",
    "source_lower",
    "source_upper",
    "target_lower",
    "target_upper",
}
_OBJECTIVE_KEYS = {"source_direction", "target_direction", "scale", "offset"}


def canonical_json(value: Any) -> bytes:
    """Return stable bytes so identities do not depend on dictionary insertion order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Label each digest so another algorithm cannot be substituted silently."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_path(path: Path) -> str | None:
    """Return an exact file digest, or null when the file does not exist."""

    target = Path(path)
    return sha256_bytes(target.read_bytes()) if target.is_file() else None


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Record both gate sides so a failure identifies the required correction."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected if passed is None else bool(passed),
        "terminal": True,
    }


def _summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all terminal checks and provide a direct failed-check view."""

    rows = [dict(row) for row in checks]
    failed = [row for row in rows if not row.get("passed")]
    return {"checks": rows, "failed_checks": failed, "passed": not failed}


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    """Compare stored numeric results while preserving null and Boolean types."""

    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def load_frozen_sources(
    repo_root: Path,
    *,
    source_paths: Mapping[str, Path] = SOURCE_PATHS,
    expected_hashes: Mapping[str, str] = EXPECTED_SOURCE_HASHES,
) -> JsonDict:
    """Verify every source byte hash before parsing any upstream JSON field."""

    root = Path(repo_root)
    hash_rows = []
    raw_bytes: dict[str, bytes] = {}
    for name, relative in source_paths.items():
        path = relative if relative.is_absolute() else root / relative
        data = path.read_bytes() if path.is_file() else None
        observed = sha256_bytes(data) if data is not None else None
        expected = expected_hashes.get(name)
        hash_rows.append(
            {
                **_gate(f"source_hash:{name}", expected, observed),
                "path": str(relative),
                "hash_kind": "source_artifact",
            }
        )
        if data is not None:
            raw_bytes[name] = data
    if not all(row["passed"] for row in hash_rows):
        return {"passed": False, "sources": {}, "hash_rows": hash_rows}
    try:
        sources = {name: json.loads(raw_bytes[name].decode("utf-8")) for name in source_paths}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        row = _gate("source_json_parse", "valid_utf8_json", f"{type(exc).__name__}:{exc}")
        return {"passed": False, "sources": {}, "hash_rows": [*hash_rows, row]}
    return {"passed": True, "sources": sources, "hash_rows": hash_rows}


def _mapping_schema_error(mapping: Any, attempt: Mapping[str, Any]) -> str | None:
    """Reapply the frozen bank schema without completing or repairing a field."""

    if not isinstance(mapping, Mapping) or set(mapping) != _MAPPING_KEYS:
        return "mapping_keys"
    if mapping.get("schema_version") != "carnot.reformulation_mapping.v1":
        return "mapping_schema_version"
    variables = mapping.get("variables")
    domains = mapping.get("domain_clauses")
    objective = mapping.get("objective")
    if not isinstance(variables, list) or any(
        not isinstance(row, Mapping) or set(row) != _VARIABLE_KEYS for row in variables
    ):
        return "variable_rows"
    if not isinstance(domains, list) or any(
        not isinstance(row, Mapping) or set(row) != _DOMAIN_KEYS for row in domains
    ):
        return "domain_clause_rows"
    if not isinstance(objective, Mapping) or set(objective) != _OBJECTIVE_KEYS:
        return "objective_keys"
    source_names = [str(row.get("name")) for row in attempt["source_formulation"]["variables"]]
    target_names = [str(row.get("name")) for row in attempt["target_formulation"]["variables"]]
    rosters = (
        ("source_variable_roster", [row.get("source") for row in variables], source_names),
        ("target_variable_roster", [row.get("target") for row in variables], target_names),
        ("source_domain_roster", [row.get("source") for row in domains], source_names),
        ("target_domain_roster", [row.get("target") for row in domains], target_names),
    )
    for reason, observed, expected in rosters:
        if sorted(observed) != sorted(expected) or len(set(observed)) != len(observed):
            return reason
    rationals = [
        *(row.get(field) for row in variables for field in ("scale", "offset")),
        *(
            row.get(field)
            for row in domains
            for field in ("source_lower", "source_upper", "target_lower", "target_upper")
        ),
        objective.get("scale"),
        objective.get("offset"),
    ]
    if any(not isinstance(value, str) or not value for value in rationals):
        return "rational_string_required"
    if objective.get("source_direction") not in {"min", "max"} or objective.get(
        "target_direction"
    ) not in {"min", "max"}:
        return "objective_direction"
    if mapping.get("claimed_relation") not in {"equivalent", "non_equivalent"}:
        return "claimed_relation"
    return None


def strict_parse(raw_text: str, attempt: Mapping[str, Any]) -> JsonDict:
    """Parse one complete raw document and never recover JSON from surrounding text."""

    if not raw_text:
        return {
            "json_valid": False,
            "schema_valid": False,
            "failure_reason": "empty_output",
            "parsed_candidate": None,
            "confidence": None,
            "rationale": None,
        }
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return {
            "json_valid": False,
            "schema_valid": False,
            "failure_reason": "malformed_json",
            "parsed_candidate": None,
            "confidence": None,
            "rationale": None,
        }
    reason = None
    if not isinstance(parsed, Mapping):
        reason = "response_object_required"
    elif "mapping" not in parsed or not set(parsed) <= _TOP_LEVEL_KEYS:
        reason = "response_keys"
    else:
        confidence = parsed.get("confidence")
        rationale = parsed.get("rationale")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            reason = "confidence_range"
        elif rationale is not None and (not isinstance(rationale, str) or len(rationale) > 240):
            reason = "rationale_length"
        else:
            reason = _mapping_schema_error(parsed["mapping"], attempt)
    return {
        "json_valid": True,
        "schema_valid": reason is None,
        "failure_reason": reason,
        "parsed_candidate": deepcopy(parsed),
        "confidence": parsed.get("confidence") if isinstance(parsed, Mapping) else None,
        "rationale": parsed.get("rationale") if isinstance(parsed, Mapping) else None,
    }


def audit_proposal_bank(bank: Mapping[str, Any]) -> JsonDict:
    """Rebuild raw parses, fixed groups, candidate order, and the proposal budget."""

    attempts = bank.get("attempt_rows", [])
    raw_rows = bank.get("raw_output_rows", [])
    raw_by_key = {str(row.get("attempt_key")): row for row in raw_rows if isinstance(row, Mapping)}
    keys = [str(row.get("attempt_key")) for row in attempts]
    raw_keys = [str(row.get("attempt_key")) for row in raw_rows]
    checks = [
        _gate("proposal_row_count", EXPECTED_CANDIDATE_COUNT, len(attempts)),
        _gate("raw_output_row_count", EXPECTED_CANDIDATE_COUNT, len(raw_rows)),
        _gate("unique_proposal_keys", EXPECTED_CANDIDATE_COUNT, len(set(keys))),
        _gate("unique_raw_keys", EXPECTED_CANDIDATE_COUNT, len(set(raw_keys))),
        _gate("proposal_raw_keys_match", sorted(keys), sorted(raw_keys)),
        _gate(
            "candidate_ordinal_order",
            list(range(len(attempts))),
            [row.get("ordinal") for row in attempts],
        ),
    ]
    candidate_rows = []
    reparsed_attempts = []
    raw_hash_rows = []
    for attempt in attempts:
        key = str(attempt.get("attempt_key"))
        raw = raw_by_key.get(key, {})
        raw_text = attempt.get("raw_text")
        observed_hash = (
            sha256_bytes(str(raw_text).encode("utf-8")) if isinstance(raw_text, str) else None
        )
        hash_match = (
            observed_hash == attempt.get("raw_sha256")
            and raw.get("raw_sha256") == observed_hash
            and raw.get("raw_text") == raw_text
        )
        parsed = strict_parse(str(raw_text or ""), attempt)
        stored_parse = attempt.get("parse", {})
        parse_match = all(
            stored_parse.get(field) == parsed.get(field)
            for field in (
                "json_valid",
                "schema_valid",
                "failure_reason",
                "parsed_candidate",
                "confidence",
                "rationale",
            )
        )
        candidate_rows.append(
            {
                "attempt_key": key,
                "ordinal": attempt.get("ordinal"),
                "raw_hash_matches": hash_match,
                "parse_matches": parse_match,
                "json_valid": parsed["json_valid"],
                "schema_valid": parsed["schema_valid"],
                "terminal": True,
            }
        )
        raw_hash_rows.append(
            {
                **_gate(
                    f"raw_text_hash:{key}", attempt.get("raw_sha256"), observed_hash, hash_match
                ),
                "path": key,
                "hash_kind": "raw_proposal_text",
            }
        )
        replay_attempt = deepcopy(dict(attempt))
        replay_attempt["parse"] = parsed
        reparsed_attempts.append(replay_attempt)
    checks.extend(
        [
            _gate(
                "all_raw_hashes_match", True, all(row["raw_hash_matches"] for row in candidate_rows)
            ),
            _gate(
                "all_raw_parses_match", True, all(row["parse_matches"] for row in candidate_rows)
            ),
        ]
    )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in attempts:
        groups[(str(row.get("model_family")), str(row.get("pair_id")))].append(row)
    budget_rows = []
    for (model_family, pair_id), rows in sorted(groups.items()):
        variants = [str(row.get("prompt_variant_id")) for row in rows]
        budget_rows.append(
            {
                "scope": "model_pair_group",
                "model_family": model_family,
                "pair_id": pair_id,
                "candidate_count": len(rows),
                "expected_candidate_count": len(PROMPT_VARIANT_ORDER),
                "observed_prompt_variants": variants,
                "expected_prompt_variants": list(PROMPT_VARIANT_ORDER),
                "passed": len(rows) == 3 and variants == list(PROMPT_VARIANT_ORDER),
                "terminal": True,
            }
        )
    budget_rows.append(
        {
            "scope": "overall",
            "candidate_count": len(attempts),
            "expected_candidate_count": EXPECTED_CANDIDATE_COUNT,
            "group_count": len(groups),
            "expected_group_count": EXPECTED_GROUP_COUNT,
            "passed": len(attempts) == EXPECTED_CANDIDATE_COUNT
            and len(groups) == EXPECTED_GROUP_COUNT
            and all(row["passed"] for row in budget_rows),
            "terminal": True,
        }
    )
    checks.extend(
        [
            _gate("proposal_group_count", EXPECTED_GROUP_COUNT, len(groups)),
            _gate("proposal_group_budget", True, all(row["passed"] for row in budget_rows)),
        ]
    )
    failed = [str(row["check"]) for row in checks if not row["passed"]]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "candidate_rows": candidate_rows,
        "proposal_budget_rows": budget_rows,
        "raw_hash_rows": raw_hash_rows,
        "reparsed_attempts": reparsed_attempts,
    }


def _fraction(value: Any) -> Fraction:
    """Read an exact rational and reject Boolean or binary floating-point inputs."""

    if isinstance(value, (bool, float)):
        raise ValueError(f"non_exact_rational:{value!r}")
    return Fraction(value)


def _canonical_mapping(
    mapping: Any, source: Mapping[str, Any], target: Mapping[str, Any]
) -> JsonDict:
    """Validate the stricter exact-engine mapping schema without repairing values."""

    if not isinstance(mapping, Mapping) or set(mapping) != _MAPPING_KEYS:
        raise ValueError("mapping_keys")
    variables = mapping.get("variables")
    domains = mapping.get("domain_clauses")
    objective = mapping.get("objective")
    if (
        not isinstance(variables, list)
        or not isinstance(domains, list)
        or not isinstance(objective, Mapping)
    ):
        raise ValueError("mapping_container_type")
    source_names = {str(row["name"]) for row in source["variables"]}
    target_names = {str(row["name"]) for row in target["variables"]}
    mapped_sources = [str(row["source"]) for row in variables]
    mapped_targets = [str(row["target"]) for row in variables]
    if set(mapped_sources) != source_names or len(mapped_sources) != len(set(mapped_sources)):
        raise ValueError("source_variable_coverage")
    if set(mapped_targets) != target_names or len(mapped_targets) != len(set(mapped_targets)):
        raise ValueError("target_variable_coverage")
    for row in variables:
        if set(row) != _VARIABLE_KEYS or _fraction(row["scale"]) == 0:
            raise ValueError("variable_mapping")
        _fraction(row["offset"])
    if {(str(row["source"]), str(row["target"])) for row in domains} != {
        (str(row["source"]), str(row["target"])) for row in variables
    }:
        raise ValueError("domain_clause_coverage")
    for row in domains:
        if set(row) != _DOMAIN_KEYS:
            raise ValueError("domain_clause_keys")
        for field in ("source_lower", "source_upper", "target_lower", "target_upper"):
            if row[field] is not None:
                _fraction(row[field])
    if set(objective) != _OBJECTIVE_KEYS or _fraction(objective["scale"]) == 0:
        raise ValueError("objective_mapping")
    _fraction(objective["offset"])
    if objective["source_direction"] not in {"min", "max"} or objective["target_direction"] not in {
        "min",
        "max",
    }:
        raise ValueError("objective_direction")
    return deepcopy(dict(mapping))


def _all_assignments(formulation: Mapping[str, Any]) -> list[JsonDict]:
    """Enumerate the finite public universe in declaration order."""

    names = [str(row["name"]) for row in formulation["variables"]]
    universes = [row["universe"] for row in formulation["variables"]]
    return [dict(zip(names, values, strict=True)) for values in product(*universes)]


def _linear_value(expression: Mapping[str, Any], assignment: Mapping[str, Any]) -> Fraction:
    """Evaluate one affine expression with exact rational arithmetic."""

    return _fraction(expression.get("constant", "0")) + sum(
        _fraction(coefficient) * int(assignment[name])
        for name, coefficient in expression["terms"].items()
    )


def _objective_value(formulation: Mapping[str, Any], assignment: Mapping[str, Any]) -> Fraction:
    """Evaluate a linear or piecewise-linear objective exactly."""

    expression = formulation["objective"]["expression"]
    if expression["kind"] == "linear":
        return _linear_value(expression, assignment)
    values = [_linear_value(piece, assignment) for piece in expression["pieces"]]
    return max(values) if expression["aggregation"] == "max" else min(values)


def _compare(left: Fraction, operator: str, right: Fraction) -> bool:
    """Evaluate one registered exact relation without floating-point conversion."""

    return {
        "<=": left <= right,
        ">=": left >= right,
        "==": left == right,
        "<": left < right,
        ">": left > right,
    }[operator]


def _is_feasible(formulation: Mapping[str, Any], assignment: Mapping[str, Any]) -> bool:
    """Apply semantic domains and constraints to one public-universe assignment."""

    for variable in formulation["variables"]:
        value = _fraction(int(assignment[str(variable["name"])]))
        lower = variable["domain"]["lower"]
        upper = variable["domain"]["upper"]
        if lower is not None and value < _fraction(lower):
            return False
        if upper is not None and value > _fraction(upper):
            return False
    for constraint in formulation["constraints"]:
        left = sum(
            _fraction(coefficient) * int(assignment[name])
            for name, coefficient in constraint["terms"].items()
        )
        if not _compare(left, str(constraint["op"]), _fraction(constraint["rhs"])):
            return False
    return True


def _feasible_assignments(formulation: Mapping[str, Any]) -> list[JsonDict]:
    """Return every feasible assignment from the finite declared universe."""

    return [row for row in _all_assignments(formulation) if _is_feasible(formulation, row)]


def _forward(assignment: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, Fraction]:
    """Apply the exact affine source-to-target map."""

    return {
        str(row["target"]): _fraction(row["scale"]) * int(assignment[str(row["source"])])
        + _fraction(row["offset"])
        for row in mapping["variables"]
    }


def _reverse(assignment: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, Fraction]:
    """Apply the exact inverse of each nonzero affine variable map."""

    return {
        str(row["source"]): (int(assignment[str(row["target"])]) - _fraction(row["offset"]))
        / _fraction(row["scale"])
        for row in mapping["variables"]
    }


def _typed_assignment(
    formulation: Mapping[str, Any], values: Mapping[str, Fraction]
) -> JsonDict | None:
    """Convert exact mapped values only when every target type admits them."""

    result: JsonDict = {}
    for variable in formulation["variables"]:
        name = str(variable["name"])
        value = values.get(name)
        if value is None or value.denominator != 1:
            return None
        integer = value.numerator
        if variable["kind"] == "boolean":
            if integer not in {0, 1}:
                return None
            result[name] = bool(integer)
        else:
            result[name] = integer
        if result[name] not in variable["universe"]:
            return None
    return result


def _direction_valid(
    mapping: Mapping[str, Any], source: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    """Check declared directions and the sign-induced preserve-or-reverse rule."""

    objective = mapping["objective"]
    scale = _fraction(objective["scale"])
    expected = (
        objective["source_direction"]
        if scale > 0
        else ("max" if objective["source_direction"] == "min" else "min")
    )
    return bool(
        objective["source_direction"] == source["objective"]["direction"]
        and objective["target_direction"] == target["objective"]["direction"]
        and objective["target_direction"] == expected
    )


def _better(direction: str, left: Fraction, right: Fraction) -> bool:
    """Use weak preference so exact objective ties remain ties."""

    return left <= right if direction == "min" else left >= right


def certify_with_enumerator(
    source: Mapping[str, Any], target: Mapping[str, Any], mapping: Mapping[str, Any]
) -> JsonDict:
    """Check all bounded assignments and ordered feasible pairs independently."""

    mapping = _canonical_mapping(mapping, source, target)
    source_feasible = _feasible_assignments(source)
    target_feasible = _feasible_assignments(target)
    mapped_rows: list[tuple[JsonDict, JsonDict, Fraction, Fraction]] = []
    forward_ok = True
    reverse_ok = True
    for source_row in source_feasible:
        target_row = _typed_assignment(target, _forward(source_row, mapping))
        if target_row is None or not _is_feasible(target, target_row):
            forward_ok = False
        else:
            mapped_rows.append(
                (
                    source_row,
                    target_row,
                    _objective_value(source, source_row),
                    _objective_value(target, target_row),
                )
            )
    for target_row in target_feasible:
        source_row = _typed_assignment(source, _reverse(target_row, mapping))
        if source_row is None or not _is_feasible(source, source_row):
            reverse_ok = False
    scale = _fraction(mapping["objective"]["scale"])
    offset = _fraction(mapping["objective"]["offset"])
    affine_ok = all(
        target_value == scale * source_value + offset
        for _, _, source_value, target_value in mapped_rows
    )
    order_ok = all(
        _better(source["objective"]["direction"], left[2], right[2])
        == _better(target["objective"]["direction"], left[3], right[3])
        for left, right in product(mapped_rows, repeat=2)
    )
    direction_ok = _direction_valid(mapping, source, target)
    equivalent = forward_ok and reverse_ok and direction_ok and affine_ok and order_ok
    return {
        "status": "proved" if equivalent else "counterexample",
        "label": "equivalent" if equivalent else "non_equivalent",
        "forward_feasible": forward_ok,
        "reverse_feasible": reverse_ok,
        "objective_direction_valid": direction_ok,
        "objective_affine_preserved": affine_ok,
        "objective_order_preserved": order_ok,
        "source_feasible_count": len(source_feasible),
        "target_feasible_count": len(target_feasible),
    }


def _z3_number(value: Any) -> Any:
    """Create one exact Z3 rational without a binary float."""

    exact = _fraction(value)
    return z3.Q(exact.numerator, exact.denominator)


def _z3_variables(formulation: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Create typed symbols for one independent formulation copy."""

    return {
        str(variable["name"]): (
            z3.Bool(f"{prefix}_{variable['name']}")
            if variable["kind"] == "boolean"
            else z3.Int(f"{prefix}_{variable['name']}")
        )
        for variable in formulation["variables"]
    }


def _z3_numeric(symbol: Any) -> Any:
    """Map Boolean cardinality symbols to exact zero-or-one arithmetic."""

    return z3.If(symbol, z3.IntVal(1), z3.IntVal(0)) if z3.is_bool(symbol) else symbol


def _z3_linear(expression: Mapping[str, Any], symbols: Mapping[str, Any]) -> Any:
    """Encode one affine expression over typed Z3 symbols."""

    return _z3_number(expression.get("constant", "0")) + sum(
        _z3_number(coefficient) * _z3_numeric(symbols[name])
        for name, coefficient in expression["terms"].items()
    )


def _z3_relation(left: Any, operator: str, right: Any) -> Any:
    """Translate the closed constraint operator set to Z3."""

    return {
        "<=": left <= right,
        ">=": left >= right,
        "==": left == right,
        "<": left < right,
        ">": left > right,
    }[operator]


def _z3_feasible(formulation: Mapping[str, Any], symbols: Mapping[str, Any]) -> Any:
    """Encode finite universes, semantic domains, and formulation constraints."""

    clauses = []
    for variable in formulation["variables"]:
        symbol = symbols[str(variable["name"])]
        if variable["kind"] == "integer":
            clauses.append(z3.Or(*[symbol == value for value in variable["universe"]]))
        numeric = _z3_numeric(symbol)
        lower = variable["domain"]["lower"]
        upper = variable["domain"]["upper"]
        if lower is not None:
            clauses.append(numeric >= _z3_number(lower))
        if upper is not None:
            clauses.append(numeric <= _z3_number(upper))
    for constraint in formulation["constraints"]:
        clauses.append(
            _z3_relation(
                _z3_linear({"terms": constraint["terms"], "constant": "0"}, symbols),
                str(constraint["op"]),
                _z3_number(constraint["rhs"]),
            )
        )
    return z3.And(*clauses) if clauses else z3.BoolVal(True)


def _z3_objective(formulation: Mapping[str, Any], symbols: Mapping[str, Any]) -> Any:
    """Encode a linear or exact piecewise-linear Z3 objective."""

    expression = formulation["objective"]["expression"]
    if expression["kind"] == "linear":
        return _z3_linear(expression, symbols)
    values = [_z3_linear(piece, symbols) for piece in expression["pieces"]]
    current = values[0]
    for value in values[1:]:
        if expression["aggregation"] == "max":
            current = z3.If(current >= value, current, value)
        else:
            current = z3.If(current <= value, current, value)
    return current


def _z3_mapping(
    mapping: Mapping[str, Any], source_symbols: Mapping[str, Any], target_symbols: Mapping[str, Any]
) -> Any:
    """Encode every exact affine variable equation."""

    return z3.And(
        *[
            _z3_numeric(target_symbols[str(row["target"])])
            == _z3_number(row["scale"]) * _z3_numeric(source_symbols[str(row["source"])])
            + _z3_number(row["offset"])
            for row in mapping["variables"]
        ]
    )


def _z3_status(*clauses: Any) -> str:
    """Run one bounded symbolic obligation and retain timeout as unknown."""

    solver = z3.Solver()
    solver.set(timeout=2_000)
    solver.add(*clauses)
    status = solver.check()
    if status == z3.sat:
        return "sat"
    if status == z3.unsat:
        return "unsat"
    return "unknown"


def certify_with_z3(
    source: Mapping[str, Any], target: Mapping[str, Any], mapping: Mapping[str, Any], key: str
) -> JsonDict:
    """Search four symbolic counterexamples without using enumerator outcomes."""

    mapping = _canonical_mapping(mapping, source, target)
    stem = hashlib.sha256(key.encode()).hexdigest()[:12]
    source_one = _z3_variables(source, f"s1_{stem}")
    target_one = _z3_variables(target, f"t1_{stem}")
    source_two = _z3_variables(source, f"s2_{stem}")
    target_two = _z3_variables(target, f"t2_{stem}")
    source_feasible = _z3_feasible(source, source_one)
    target_feasible = _z3_feasible(target, target_one)
    mapping_one = _z3_mapping(mapping, source_one, target_one)
    forward = _z3_status(
        source_feasible,
        z3.Not(z3.Exists(list(target_one.values()), z3.And(mapping_one, target_feasible))),
    )
    reverse = _z3_status(
        target_feasible,
        z3.Not(z3.Exists(list(source_one.values()), z3.And(mapping_one, source_feasible))),
    )
    affine = _z3_status(
        source_feasible,
        target_feasible,
        mapping_one,
        _z3_objective(target, target_one)
        != _z3_number(mapping["objective"]["scale"]) * _z3_objective(source, source_one)
        + _z3_number(mapping["objective"]["offset"]),
    )
    source_better = (
        _z3_objective(source, source_one) <= _z3_objective(source, source_two)
        if source["objective"]["direction"] == "min"
        else _z3_objective(source, source_one) >= _z3_objective(source, source_two)
    )
    target_better = (
        _z3_objective(target, target_one) <= _z3_objective(target, target_two)
        if target["objective"]["direction"] == "min"
        else _z3_objective(target, target_one) >= _z3_objective(target, target_two)
    )
    order = _z3_status(
        source_feasible,
        target_feasible,
        mapping_one,
        _z3_feasible(source, source_two),
        _z3_feasible(target, target_two),
        _z3_mapping(mapping, source_two, target_two),
        source_better != target_better,
    )
    statuses = (forward, reverse, affine, order)
    direction = _direction_valid(mapping, source, target)
    if "unknown" in statuses:
        return {"status": "unknown", "label": None, "query_statuses": list(statuses)}
    equivalent = all(status == "unsat" for status in statuses) and direction
    return {
        "status": "proved" if equivalent else "counterexample",
        "label": "equivalent" if equivalent else "non_equivalent",
        "query_statuses": list(statuses),
    }


def replay_certificates(
    attempts: Sequence[Mapping[str, Any]],
    certificates: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> JsonDict:
    """Recompute both exact labels and compare all serialized authority states."""

    stored_z3 = {str(row["attempt_key"]): row for row in certificates["z3_rows"]}
    stored_enum = {str(row["attempt_key"]): row for row in certificates["enumeration_rows"]}
    stored_proposals = {str(row["attempt_key"]): row for row in certificates["proposal_rows"]}
    checkpoint_rows = {str(row["attempt_key"]): row for row in checkpoint["inputs"]}
    rows = []
    for attempt in attempts:
        key = str(attempt["attempt_key"])
        parsed = attempt["parse"]
        if not parsed["json_valid"]:
            enum = {"status": "parse_rejected", "label": None}
            symbolic = {"status": "parse_rejected", "label": None}
        elif not parsed["schema_valid"]:
            enum = {"status": "schema_rejected", "label": None}
            symbolic = {"status": "schema_rejected", "label": None}
        else:
            mapping = parsed["parsed_candidate"]["mapping"]
            try:
                _canonical_mapping(
                    mapping, attempt["source_formulation"], attempt["target_formulation"]
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                enum = {"status": "schema_rejected", "label": None}
                symbolic = {"status": "schema_rejected", "label": None}
            else:
                enum = certify_with_enumerator(
                    attempt["source_formulation"], attempt["target_formulation"], mapping
                )
                symbolic = certify_with_z3(
                    attempt["source_formulation"], attempt["target_formulation"], mapping, key
                )
        stored_enum_row = stored_enum.get(key, {})
        stored_z3_row = stored_z3.get(key, {})
        proposal = stored_proposals.get(key, {})
        serialized = checkpoint_rows.get(key, {})
        status_match = enum["status"] == stored_enum_row.get("status") and symbolic[
            "status"
        ] == stored_z3_row.get("status")
        label_match = enum["label"] == stored_enum_row.get("label") and symbolic[
            "label"
        ] == stored_z3_row.get("label")
        authorities_agree = (
            enum["status"] == symbolic["status"] and enum["label"] == symbolic["label"]
        )
        certified_relation = (
            enum["label"]
            if authorities_agree and enum["status"] in {"proved", "counterexample"}
            else None
        )
        canonical_relation = serialized.get("canonical_relation")
        exact_correct = certified_relation is not None and certified_relation == canonical_relation
        checkpoint_matches = all(
            serialized.get(field) == attempt.get(field)
            for field in (
                "attempt_key",
                "pair_id",
                "model_family",
                "problem_family",
                "prompt_variant_id",
                "raw_sha256",
                "source_formulation",
                "target_formulation",
                "parse",
            )
        )
        proposal_matches = (
            proposal.get("canonical_relation") == canonical_relation
            and proposal.get("certified_relation") == certified_relation
            and proposal.get("exact_mapping_correct") is exact_correct
        )
        rows.append(
            {
                "attempt_key": key,
                "pair_id": attempt["pair_id"],
                "difficulty": proposal.get("difficulty", serialized.get("difficulty", "unknown")),
                "enumeration_status": enum["status"],
                "z3_status": symbolic["status"],
                "enumeration_label": enum["label"],
                "z3_label": symbolic["label"],
                "certified_relation": certified_relation,
                "canonical_relation": canonical_relation,
                "exact_mapping_correct": exact_correct,
                "false_acceptance": certified_relation == "equivalent"
                and canonical_relation == "non_equivalent",
                "status_matches": status_match,
                "label_matches": label_match,
                "authorities_agree": authorities_agree,
                "checkpoint_matches": checkpoint_matches,
                "proposal_matches": proposal_matches,
                "enumeration_executed": enum["status"] in {"proved", "counterexample"},
                "z3_executed": symbolic["status"] in {"proved", "counterexample"},
                "passed": status_match
                and label_match
                and authorities_agree
                and checkpoint_matches
                and proposal_matches,
                "terminal": True,
            }
        )
    return {
        "rows": rows,
        "passed": len(rows) == EXPECTED_CANDIDATE_COUNT and all(row["passed"] for row in rows),
    }


def _unique_row(rows: Any, field: str, value: str) -> Mapping[str, Any] | None:
    """Return one structural row because zero or duplicate matches are invalid."""

    if not isinstance(rows, list):
        return None
    matches = [row for row in rows if isinstance(row, Mapping) and row.get(field) == value]
    return matches[0] if len(matches) == 1 else None


def _opposite(value: Any) -> Any:
    """Reverse the two registered direction or aggregation values."""

    return "max" if value == "min" else "min" if value == "max" else None


def _feature_fraction(value: Any) -> Fraction:
    """Treat public Boolean universe values as zero or one for structural factors."""

    return Fraction(int(value)) if isinstance(value, bool) else Fraction(str(value))


def structural_factors(attempt: Mapping[str, Any]) -> list[list[float]]:
    """Encode only public proposal structure into the frozen five channels."""

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
    factors = []
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
                scale = _feature_fraction(variable_row["scale"])
                offset = _feature_fraction(variable_row["offset"])
                source_universe = [
                    _feature_fraction(value) for value in source_variable.get("universe", [])
                ]
                target_universe = [
                    _feature_fraction(value) for value in target_variable.get("universe", [])
                ]
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
        scale = _feature_fraction(objective["scale"])
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
            expected_aggregation = (
                source_expression.get("aggregation")
                if scale > 0
                else _opposite(source_expression.get("aggregation"))
            )
            aggregation_ok = target_expression.get("aggregation") == expected_aggregation
        else:
            aggregation_ok = True
        shape_violation = float(not (same_kind and same_piece_count and aggregation_ok))
        _feature_fraction(objective["offset"])
    except (KeyError, ValueError, ZeroDivisionError):
        direction_violation = 1.0
        shape_violation = 1.0
    factors.append([0.0, 0.0, 0.0, direction_violation, 0.0])
    factors.append([0.0, 0.0, 0.0, 0.0, shape_violation])
    return factors


def _likelihood_score(attempt: Mapping[str, Any]) -> float | None:
    """Use a recorded generation likelihood and never substitute token count."""

    for container in (attempt, attempt.get("runtime_receipt", {})):
        if not isinstance(container, Mapping):
            continue
        for field in ("mean_logprob", "average_logprob", "sequence_logprob"):
            value = container.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def candidate_from_attempt(attempt: Mapping[str, Any]) -> JsonDict:
    """Create one opaque label-free candidate from the independently parsed row."""

    source_key = str(attempt["attempt_key"])
    candidate_id = sha256_bytes(canonical_json({"attempt_key": source_key}))
    group_id = sha256_bytes(
        canonical_json({"model_family": attempt["model_family"], "pair_id": attempt["pair_id"]})
    )
    parsed = attempt["parse"]
    candidate = parsed.get("parsed_candidate")
    factors = structural_factors(attempt)
    variant = str(attempt["prompt_variant_id"])
    payload = {
        "mapping": candidate.get("mapping") if isinstance(candidate, Mapping) else None,
        "source_formulation": attempt["source_formulation"],
        "target_formulation": attempt["target_formulation"],
        "generation_metadata": {
            "model_family": attempt["model_family"],
            "problem_family": attempt["problem_family"],
            "prompt_variant_id": variant,
            "confidence": parsed.get("confidence"),
            "json_valid": bool(parsed.get("json_valid")),
            "schema_valid": bool(parsed.get("schema_valid")),
            "likelihood": _likelihood_score(attempt),
        },
        "factors": factors,
    }
    confidence = parsed.get("confidence")
    return {
        "attempt_key": candidate_id,
        "source_attempt_key": source_key,
        "group_id": group_id,
        "pair_id": str(attempt["pair_id"]),
        "model_family": str(attempt["model_family"]),
        "problem_family": str(attempt["problem_family"]),
        "prompt_variant_id": variant,
        "ordinal": int(attempt["ordinal"]),
        "raw_sha256": attempt.get("raw_sha256"),
        "factor_count": len(factors),
        "selector_payload": payload,
        "selector_payload_sha256": sha256_bytes(canonical_json(payload)),
        "scores": {
            ARM_LIKELIHOOD: _likelihood_score(attempt),
            ARM_SYNTAX: float(bool(parsed.get("json_valid")) and bool(parsed.get("schema_valid"))),
            ARM_CONFIDENCE: (
                float(confidence)
                if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                else None
            ),
            ARM_FIXED: float(PROMPT_VARIANT_RANK[variant]),
        },
        "scores_by_seed": {},
    }


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    """Find forbidden exact-authority keys at every selector payload depth."""

    paths = []
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


def audit_label_payload(candidate_id: str, payload: Mapping[str, Any]) -> JsonDict:
    """Prove that one literal selector payload contains no exact authority field."""

    forbidden = _forbidden_paths(payload)
    return {
        "attempt_key": candidate_id,
        "forbidden_paths": forbidden,
        "passed": not forbidden,
        "terminal": True,
    }


def _resolve_checkpoint_path(repo_root: Path, path_text: str) -> Path:
    """Resolve legacy absolute receipts by their frozen checkpoint filename."""

    path = Path(path_text)
    if path.is_absolute() and path.is_file():
        return path
    relative = path if not path.is_absolute() else CHECKPOINT_DIR / path.name
    return Path(repo_root) / relative


def audit_checkpoint_hashes(
    repo_root: Path, entries: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Verify every checkpoint byte hash before torch deserializes tensor state."""

    rows = []
    for entry in entries:
        path = _resolve_checkpoint_path(repo_root, str(entry["path"]))
        expected = entry.get("expected_sha256")
        observed = sha256_path(path)
        rows.append(
            {
                **_gate(f"checkpoint_hash:{path.name}", expected, observed),
                "path": str(path),
                "hash_kind": "factor_checkpoint",
            }
        )
    return rows


def _checkpoint_entries(energy: Mapping[str, Any]) -> list[JsonDict]:
    """Join the frozen path roster to the independently pinned arm and seed hashes."""

    entries = []
    for path_text in energy.get("checkpoint_paths", []):
        stem = Path(path_text).stem
        arm, seed_text = stem.rsplit("_seed_", 1)
        entries.append(
            {
                "path": str(path_text),
                "arm": arm,
                "seed": int(seed_text),
                "expected_sha256": EXPECTED_FACTOR_HASHES.get((arm, int(seed_text))),
            }
        )
    return entries


def load_factor_checkpoints(
    repo_root: Path, energy: Mapping[str, Any], hash_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Load minimal tensor dictionaries only after all factor hashes pass."""

    if not hash_rows or not all(row.get("passed") for row in hash_rows):
        return {"passed": False, "models": {}, "rows": []}
    models: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rows = []
    for entry in _checkpoint_entries(energy):
        path = _resolve_checkpoint_path(repo_root, str(entry["path"]))
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            shapes = {name: list(tensor.shape) for name, tensor in payload["state_dict"].items()}
            valid = (
                payload.get("schema_version") == "carnot.exp6958.convex_factor_replay.v1"
                and payload.get("arm") == entry["arm"]
                and payload.get("seed") == entry["seed"]
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            payload = {}
            shapes = {}
            valid = False
            error = f"{type(exc).__name__}:{exc}"
        else:
            error = None
        if valid:
            models[str(entry["arm"])].append(payload)
        rows.append(
            {
                "path": str(path),
                "arm": entry["arm"],
                "seed": entry["seed"],
                "schema_version": payload.get("schema_version"),
                "state_shapes": shapes,
                "error": error,
                "passed": valid,
                "terminal": True,
            }
        )
    passed = len(rows) == len(EXPECTED_FACTOR_HASHES) and all(row["passed"] for row in rows)
    return {"passed": passed, "models": dict(models), "rows": rows}


def _score_tensors(
    state: Mapping[str, Any], arm: str, candidates: Sequence[Mapping[str, Any]]
) -> list[float]:
    """Evaluate one saved tensor state with the frozen full-candidate batch shape."""

    factor_rows = [row["selector_payload"]["factors"] for row in candidates]
    maximum = max(len(row) for row in factor_rows)
    values = torch.zeros((len(factor_rows), maximum, 5), dtype=torch.float64)
    mask = torch.zeros((len(factor_rows), maximum), dtype=torch.float64)
    for index, factors in enumerate(factor_rows):
        values[index, : len(factors)] = torch.tensor(factors, dtype=torch.float64)
        mask[index, : len(factors)] = 1.0
    if arm == "linear_factor_score":
        factor_energy = torch_functional.linear(
            values, state["linear.weight"], state["linear.bias"]
        ).squeeze(-1)
    else:
        first = torch_functional.softplus(
            torch_functional.linear(values, state["input_weight"], state["input_bias"])
        )
        second = torch_functional.softplus(
            torch_functional.linear(first, state["hidden_weight"], state["hidden_bias"])
            + torch_functional.linear(values, state["skip_weight"])
        )
        factor_energy = (
            torch_functional.linear(second, state["output_weight"], state["output_bias"])
            + torch_functional.linear(values, state["linear_weight"])
        ).squeeze(-1)
    return [float(value) for value in (factor_energy * mask).sum(dim=-1)]


def recompute_arm_scores(
    candidates: list[JsonDict],
    models: Mapping[str, Sequence[Mapping[str, Any]]],
    upstream: Mapping[str, Any],
) -> list[JsonDict]:
    """Recompute every candidate-arm score and compare the stored value."""

    stored = {str(row["attempt_key"]): row for row in upstream["candidate_rows"]}
    latency: dict[str, float] = {}
    for current_arm, checkpoint_arm in CHECKPOINT_ARM_NAMES.items():
        started = time.perf_counter()
        payloads = models.get(checkpoint_arm, [])
        score_columns = [
            _score_tensors(payload["state_dict"], checkpoint_arm, candidates)
            for payload in payloads
        ]
        for index, candidate in enumerate(candidates):
            values = [column[index] for column in score_columns]
            candidate["scores_by_seed"][current_arm] = values
            candidate["scores"][current_arm] = sum(values) / len(values) if values else None
        latency[current_arm] = (time.perf_counter() - started) * 1000.0
    for arm in (ARM_LIKELIHOOD, ARM_SYNTAX, ARM_CONFIDENCE, ARM_FIXED):
        latency[arm] = 0.0
    rows = []
    for candidate in candidates:
        expected_candidate = stored.get(str(candidate["attempt_key"]), {})
        identity_match = all(
            expected_candidate.get(field) == candidate.get(field)
            for field in (
                "attempt_key",
                "group_id",
                "pair_id",
                "model_family",
                "problem_family",
                "prompt_variant_id",
                "ordinal",
                "raw_sha256",
                "factor_count",
                "selector_payload",
                "selector_payload_sha256",
            )
        )
        for arm in ARM_ORDER:
            score = candidate["scores"].get(arm)
            expected_score = expected_candidate.get("scores", {}).get(arm)
            seed_scores = candidate["scores_by_seed"].get(arm)
            expected_seed_scores = expected_candidate.get("energy_scores_by_seed", {}).get(arm)
            seed_match = expected_seed_scores == seed_scores if seed_scores is not None else True
            rows.append(
                {
                    "attempt_key": candidate["attempt_key"],
                    "group_id": candidate["group_id"],
                    "pair_id": candidate["pair_id"],
                    "arm": arm,
                    "score_direction": ARM_DIRECTIONS[arm],
                    "score": score,
                    "upstream_score": expected_score,
                    "scores_by_seed": seed_scores,
                    "identity_matches": identity_match,
                    "score_matches": _same_number(score, expected_score),
                    "seed_scores_match": seed_match,
                    "arm_latency_ms": latency[arm],
                    "passed": identity_match and _same_number(score, expected_score) and seed_match,
                    "terminal": True,
                }
            )
    return rows


def synthetic_candidate(key: str, variant: str, score: float) -> JsonDict:
    """Create the smallest candidate needed to test direction and tie policy."""

    return {
        "attempt_key": key,
        "group_id": "synthetic_group",
        "pair_id": "synthetic_pair",
        "model_family": "synthetic_model",
        "problem_family": "synthetic_problem",
        "prompt_variant_id": variant,
        "scores": {ARM_CONVEX: score},
    }


def _variant_key(candidate: Mapping[str, Any]) -> tuple[int, str]:
    """Use registered prompt order and then the immutable candidate ID."""

    return (
        PROMPT_VARIANT_RANK[str(candidate["prompt_variant_id"])],
        str(candidate["attempt_key"]),
    )


def _selection_probability(
    available: Sequence[Mapping[str, Any]], arm: str, selected_key: str, direction: str
) -> float:
    """Convert scores to a bounded calibration value without changing selection."""

    values = [float(row["scores"][arm]) for row in available]
    signed = [-value if direction == "min" else value for value in values]
    maximum = max(signed)
    weights = [math.exp(value - maximum) for value in signed]
    index = next(i for i, row in enumerate(available) if row["attempt_key"] == selected_key)
    return weights[index] / sum(weights)


def rank_group(
    candidates: Sequence[Mapping[str, Any]], arm: str, *, direction: str | None = None
) -> JsonDict:
    """Select one candidate without consulting exact labels or input order."""

    if not candidates:
        raise ValueError("empty_candidate_group")
    if arm not in ARM_DIRECTIONS:
        raise ValueError(f"unknown_arm:{arm}")
    selected_direction = direction or ARM_DIRECTIONS[arm]
    if selected_direction not in {"min", "max"}:
        raise ValueError(f"unknown_direction:{selected_direction}")
    ordered = sorted(candidates, key=_variant_key)
    available = [row for row in ordered if row.get("scores", {}).get(arm) is not None]
    base = {
        "group_id": str(ordered[0]["group_id"]),
        "pair_id": str(ordered[0]["pair_id"]),
        "model_family": str(ordered[0].get("model_family", "unknown")),
        "problem_family": str(ordered[0].get("problem_family", "unknown")),
        "arm": arm,
        "score_direction": selected_direction,
        "candidate_count": len(ordered),
        "available_score_count": len(available),
        "unavailable_score_count": len(ordered) - len(available),
        "tie_policy": "registered_prompt_variant_then_attempt_key",
        "oracle_used_for_selection": False,
        "terminal": True,
    }
    if not available:
        return {
            **base,
            "selected_attempt_key": None,
            "selected_score": None,
            "tied_attempt_keys": [],
            "tie_count": 0,
            "abstained": True,
            "selection_probability": 0.0,
        }
    scores = [float(row["scores"][arm]) for row in available]
    best = min(scores) if selected_direction == "min" else max(scores)
    tied = sorted([row for row in available if float(row["scores"][arm]) == best], key=_variant_key)
    selected_key = str(tied[0]["attempt_key"])
    return {
        **base,
        "selected_attempt_key": selected_key,
        "selected_score": best,
        "tied_attempt_keys": [str(row["attempt_key"]) for row in tied],
        "tie_count": len(tied),
        "abstained": False,
        "selection_probability": _selection_probability(
            available, arm, selected_key, selected_direction
        ),
    }


def compare_selection_policy(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> JsonDict:
    """Compare the fields that define score direction, ties, and selected identity."""

    fields = (
        "score_direction",
        "selected_attempt_key",
        "selected_score",
        "tied_attempt_keys",
        "tie_count",
        "tie_policy",
        "abstained",
    )
    mismatches = [field for field in fields if expected.get(field) != observed.get(field)]
    return {
        "check": "selection_policy",
        "mismatches": mismatches,
        "passed": not mismatches,
        "terminal": True,
    }


def _calibration(rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    """Compute Brier score and five-bin expected calibration error."""

    if not rows:
        return None, None
    pairs = [
        (float(row["selection_probability"]), float(bool(row["selected_exact_correct"])))
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
            confidence = sum(row[0] for row in selected) / len(selected)
            accuracy = sum(row[1] for row in selected) / len(selected)
            ece += len(selected) / len(pairs) * abs(confidence - accuracy)
    return brier, ece


def _tie_aware_auroc(scores: Sequence[tuple[float, bool]]) -> float | None:
    """Give tied positive-negative pairs half credit and keep one class null."""

    positives = [score for score, label in scores if label]
    negatives = [score for score, label in scores if not label]
    if not positives or not negatives:
        return None
    credit = 0.0
    for positive in positives:
        for negative in negatives:
            credit += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return credit / (len(positives) * len(negatives))


def aggregate_selection_rows(
    rows: Sequence[Mapping[str, Any]], group_field: str, group_value: str, arm: str
) -> JsonDict:
    """Give each selected group one vote and reject duplicate group-arm rows."""

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
    """Use the frozen nearest-rank percentile for deterministic intervals."""

    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def paired_bootstrap_by_pair(
    rows: Sequence[Mapping[str, Any]], seed: int, samples: int
) -> JsonDict:
    """Resample pair IDs so model groups for one pair stay clustered."""

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
            "terminal": True,
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
        "terminal": True,
    }


def _arm_metrics(
    selections: Sequence[Mapping[str, Any]],
    arm_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Recompute top-one, calibration, and candidate AUROC for every arm."""

    metrics = []
    scoped = [dict(row, scope="overall") for row in selections]
    for arm in ARM_ORDER:
        aggregate = aggregate_selection_rows(scoped, "scope", "overall", arm)
        goodness = []
        for row in arm_rows:
            if row["arm"] != arm or row["score"] is None:
                continue
            score = float(row["score"])
            if ARM_DIRECTIONS[arm] == "min":
                score = -score
            goodness.append((score, bool(labels[str(row["attempt_key"])]["exact_mapping_correct"])))
        metrics.append({**aggregate, "candidate_auroc": _tie_aware_auroc(goodness)})
    return metrics


def recompute_selections(
    candidates: Sequence[Mapping[str, Any]],
    certificate_rows: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
) -> JsonDict:
    """Freeze every arm choice, then open independently replayed exact labels."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["group_id"])].append(candidate)
    labels_by_source = {str(row["attempt_key"]): row for row in certificate_rows}
    labels = {
        sha256_bytes(canonical_json({"attempt_key": source_key})): row
        for source_key, row in labels_by_source.items()
    }
    stored_selection = {
        (str(row["group_id"]), str(row["arm"])): row for row in upstream["selection_rows"]
    }
    stored_ties = {(str(row["group_id"]), str(row["arm"])): row for row in upstream["tie_rows"]}
    selection_rows = []
    tie_rows = []
    order_rows = []
    for group_id, group in sorted(grouped.items()):
        difficulty = str(labels[str(group[0]["attempt_key"])].get("difficulty", "unknown"))
        raw_hashes = [row.get("raw_sha256") for row in group]
        diversity = "unique" if len(set(raw_hashes)) == len(raw_hashes) else "contains_duplicates"
        for arm in ARM_ORDER:
            replayed = rank_group(group, arm)
            label = labels.get(str(replayed["selected_attempt_key"]), {})
            selection = {
                **replayed,
                "difficulty": difficulty,
                "candidate_diversity": diversity,
                "selected_exact_correct": bool(label.get("exact_mapping_correct", False)),
                "selected_false_acceptance": bool(label.get("false_acceptance", False)),
                "exact_label_opened_after_selection_freeze": True,
                "scope": "overall",
            }
            expected = stored_selection.get((group_id, arm), {})
            comparable_fields = (
                "score_direction",
                "selected_attempt_key",
                "selected_score",
                "tied_attempt_keys",
                "tie_count",
                "tie_policy",
                "abstained",
                "selected_exact_correct",
                "selected_false_acceptance",
            )
            selection["upstream_matches"] = all(
                _same_number(selection.get(field), expected.get(field))
                for field in comparable_fields
            )
            selection_rows.append(selection)
            stored_tie = stored_ties.get((group_id, arm), {})
            tie_rows.append(
                {
                    "group_id": group_id,
                    "pair_id": selection["pair_id"],
                    "arm": arm,
                    "tie_count": selection["tie_count"],
                    "tied_attempt_keys": selection["tied_attempt_keys"],
                    "tie_policy": selection["tie_policy"],
                    "upstream_matches": all(
                        selection.get(field) == stored_tie.get(field)
                        for field in ("tie_count", "tied_attempt_keys", "tie_policy")
                    ),
                    "passed": selection["tie_policy"]
                    == "registered_prompt_variant_then_attempt_key"
                    and all(
                        selection.get(field) == stored_tie.get(field)
                        for field in ("tie_count", "tied_attempt_keys", "tie_policy")
                    ),
                    "terminal": True,
                }
            )
            reversed_selection = rank_group(list(reversed(group)), arm)
            order_rows.append(
                {
                    "group_id": group_id,
                    "pair_id": selection["pair_id"],
                    "arm": arm,
                    "forward_selected_attempt_key": replayed["selected_attempt_key"],
                    "reverse_selected_attempt_key": reversed_selection["selected_attempt_key"],
                    "forward_tied_attempt_keys": replayed["tied_attempt_keys"],
                    "reverse_tied_attempt_keys": reversed_selection["tied_attempt_keys"],
                    "passed": replayed == reversed_selection,
                    "terminal": True,
                }
            )
    return {
        "selection_rows": selection_rows,
        "tie_policy_rows": tie_rows,
        "candidate_order_rows": order_rows,
        "labels": labels,
        "groups": dict(grouped),
    }


def reduce_group_metrics(
    selections: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, Mapping[str, Any]],
    arm_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
) -> JsonDict:
    """Derive the baseline, headroom, paired rows, intervals, and family effects."""

    metrics = _arm_metrics(selections, arm_rows, labels)
    strongest = max(
        BASELINE_ORDER,
        key=lambda arm: (
            next(float(row["top1_accuracy"]) for row in metrics if row["arm"] == arm),
            -BASELINE_ORDER.index(arm),
        ),
    )
    by_group_arm = {(str(row["group_id"]), str(row["arm"])): row for row in selections}
    headroom_rows = []
    paired_rows = []
    for group_id, candidates in sorted(groups.items()):
        oracle = any(
            bool(labels[str(row["attempt_key"])]["exact_mapping_correct"]) for row in candidates
        )
        convex = by_group_arm[(group_id, ARM_CONVEX)]
        baseline = by_group_arm[(group_id, strongest)]
        baseline_correct = bool(baseline["selected_exact_correct"])
        convex_correct = bool(convex["selected_exact_correct"])
        available = int(oracle) - int(baseline_correct)
        captured = int(convex_correct) - int(baseline_correct)
        headroom_rows.append(
            {
                "group_id": group_id,
                "pair_id": convex["pair_id"],
                "strongest_non_oracle_baseline": strongest,
                "baseline_correct": baseline_correct,
                "convex_correct": convex_correct,
                "oracle_correct": oracle,
                "available_headroom": available,
                "captured_headroom": captured if available else None,
                "headroom_captured": captured / available if available else None,
                "no_correct_candidate": not oracle,
                "no_headroom": available == 0,
                "terminal": True,
            }
        )
        paired_rows.append(
            {
                "group_id": group_id,
                "pair_id": convex["pair_id"],
                "strongest_non_oracle_baseline": strongest,
                "convex_correct": convex_correct,
                "baseline_correct": baseline_correct,
                "paired_top1_delta": int(convex_correct) - int(baseline_correct),
                "exact_tie": convex_correct == baseline_correct,
                "terminal": True,
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
    capture_rate = captured_headroom / available_headroom if available_headroom else None
    family_rows = []
    for field in ("model_family", "problem_family", "candidate_diversity", "difficulty"):
        for value in sorted({str(row[field]) for row in selections}):
            for arm in ARM_ORDER:
                family_rows.append(aggregate_selection_rows(selections, field, value, arm))
    return {
        "arm_metrics": metrics,
        "strongest_non_oracle_baseline": strongest,
        "headroom_rows": headroom_rows,
        "paired_metric_rows": paired_rows,
        "confidence_interval_rows": [interval],
        "available_oracle_headroom": available_headroom,
        "captured_oracle_headroom": captured_headroom,
        "headroom_capture_rate": capture_rate,
        "family_rows": family_rows,
    }


def replay_summary_from_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    """Recover the independently computed headlines from serialized audit rows."""

    selection_rows = artifact["selection_rows"]
    arm_rows = artifact["arm_recompute_rows"]
    labels: dict[str, JsonDict] = {}
    for row in arm_rows:
        labels[str(row["attempt_key"])] = {
            "exact_mapping_correct": bool(row.get("exact_mapping_correct", False))
        }
    return {
        "arm_metrics": _arm_metrics(selection_rows, arm_rows, labels),
        "strongest_non_oracle_baseline": artifact["paired_metric_rows"][0][
            "strongest_non_oracle_baseline"
        ],
        "confidence_interval": artifact["confidence_interval_rows"][0],
        "available_oracle_headroom": artifact["gate_check_summary"]["available_oracle_headroom"],
        "captured_oracle_headroom": artifact["gate_check_summary"]["captured_oracle_headroom"],
        "headroom_capture_rate": artifact["gate_check_summary"]["headroom_capture_rate"],
        "complete_score": artifact["certified_selection_audit_complete_score"],
        "replay_positive": artifact["gate_check_summary"]["raw_positive_gate_passed"],
    }


def _comparison_row(check: str, expected: Any, observed: Any) -> JsonDict:
    """Create one aggregate comparison with stable expected and observed fields."""

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        passed = _same_number(expected, observed)
    else:
        passed = expected == observed
    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "terminal": True,
    }


def compare_upstream_aggregates(upstream: Mapping[str, Any], replay: Mapping[str, Any]) -> JsonDict:
    """Compare every upstream selection headline with its cold row reduction."""

    rows = []
    stored_metrics = {str(row["arm"]): row for row in upstream.get("arm_rows", [])}
    metric_fields = (
        "group_count",
        "top1_accuracy",
        "false_acceptance_rate",
        "abstention_rate",
        "brier_score",
        "expected_calibration_error",
        "candidate_auroc",
    )
    for metric in replay["arm_metrics"]:
        arm = str(metric["arm"])
        stored = stored_metrics.get(arm, {})
        for field in metric_fields:
            rows.append(_comparison_row(f"arm:{arm}:{field}", metric.get(field), stored.get(field)))
    rows.extend(
        [
            _comparison_row(
                "strongest_non_oracle_baseline",
                replay["strongest_non_oracle_baseline"],
                upstream.get("gate_check_summary", {}).get("strongest_non_oracle_baseline"),
            ),
            _comparison_row(
                "available_oracle_headroom",
                replay["available_oracle_headroom"],
                upstream.get("gate_check_summary", {}).get("available_oracle_headroom"),
            ),
            _comparison_row(
                "captured_oracle_headroom",
                replay["captured_oracle_headroom"],
                upstream.get("gate_check_summary", {}).get("captured_oracle_headroom"),
            ),
            _comparison_row(
                "headroom_capture_rate",
                replay["headroom_capture_rate"],
                upstream.get("gate_check_summary", {}).get("headroom_capture_rate"),
            ),
        ]
    )
    stored_interval = (upstream.get("confidence_interval_rows") or [{}])[0]
    for field in (
        "comparison",
        "bootstrap_unit",
        "paired_pair_count",
        "paired_group_count",
        "mean_delta",
        "ci95_lower",
        "ci95_upper",
        "bootstrap_samples",
        "strictly_above_zero",
    ):
        rows.append(
            _comparison_row(
                f"paired_interval:{field}",
                replay["confidence_interval"].get(field),
                stored_interval.get(field),
            )
        )
    upstream_positive = upstream.get("certified_energy_positive_score")
    rows.append(
        _comparison_row(
            "certified_energy_positive_score",
            int(bool(replay["replay_positive"])),
            upstream_positive,
        )
    )
    expected_class = "positive" if replay["replay_positive"] else "null"
    rows.append(
        _comparison_row("upstream_verdict_class", expected_class, upstream.get("verdict_class"))
    )
    contradictions = [
        {
            "check": row["check"],
            "expected_value": row["expected_value"],
            "observed_value": row["observed_value"],
            "terminal": True,
        }
        for row in rows
        if not row["passed"]
    ]
    return {"rows": rows, "contradictions": contradictions}


def reduce_verdict(
    *,
    audit_complete: bool,
    replay_positive: bool,
    upstream: Mapping[str, Any],
    contradictions: Sequence[Mapping[str, Any]],
) -> tuple[int, str, str]:
    """Apply the upstream claim ceiling and disqualify any replay disagreement."""

    upstream_class = str(upstream.get("verdict_class", "partial"))
    if contradictions:
        return 0, "disqualified", "complete_disqualified_certified_selection_cold_audit"
    if not audit_complete:
        return 0, "partial", "partial_certified_selection_cold_audit"
    if upstream_class == "disqualified":
        return 0, "disqualified", "complete_disqualified_certified_selection_cold_audit"
    upstream_positive = (
        upstream_class == "positive" and upstream.get("certified_energy_positive_score") == 1
    )
    if replay_positive and upstream_positive:
        return 1, "positive", "complete_positive_certified_selection_cold_audit"
    return 0, "null", "complete_null_certified_selection_cold_audit"


def _source_hash_map(hash_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Convert source hash receipts to a stable name-keyed artifact field."""

    result = {}
    for row in hash_rows:
        if row.get("hash_kind") != "source_artifact":
            continue
        name = str(row["check"]).split(":", 1)[1]
        result[name] = {
            "path": row.get("path"),
            "expected_sha256": row.get("expected_value"),
            "observed_sha256": row.get("observed_value"),
            "passed": row.get("passed"),
        }
    return result


def _upstream_source_hash(upstream: Mapping[str, Any], name: str) -> Any:
    """Read one Exp6959 source hash through its explicit named receipt."""

    value = upstream.get("source_artifact_hashes", {}).get(name)
    return value.get("sha256") if isinstance(value, Mapping) else None


def preflight_loaded_sources(
    repo_root: Path,
    sources: Mapping[str, Mapping[str, Any]],
    proposal_audit: Mapping[str, Any],
) -> JsonDict:
    """Check complete frozen rows and source chains after outer hashes pass."""

    bank = sources["proposal_bank"]
    certificates = sources["certificates"]
    certificate_checkpoint = sources["certificate_checkpoint"]
    energy = sources["energy_canary"]
    upstream = sources["upstream_selection"]
    certificate_keys = [
        str(row.get("attempt_key")) for row in certificates.get("proposal_rows", [])
    ]
    bank_keys = [str(row.get("attempt_key")) for row in bank.get("attempt_rows", [])]
    authority_surfaces = {
        name: certificates.get(name, [])
        for name in ("authority_agreement_rows", "z3_rows", "enumeration_rows")
    }
    declared_factor_hashes = {
        (str(row.get("arm")), int(row.get("seed", -1))): row.get("checkpoint_sha256")
        for row in energy.get("fresh_process_replay_rows", [])
    }
    checkpoint_inputs = certificate_checkpoint.get("inputs", [])
    checks = [
        _gate(
            "certified_selection_run_complete_score",
            1,
            upstream.get("certified_selection_run_complete_score"),
        ),
        _gate("proposal_bank_complete_score", 1, bank.get("reformulation_bank_complete_score")),
        _gate(
            "smt_certification_run_complete_score",
            1,
            certificates.get("smt_certification_run_complete_score"),
        ),
        _gate(
            "convex_factor_run_complete_score", 1, energy.get("convex_factor_run_complete_score")
        ),
        _gate("proposal_raw_audit", True, proposal_audit.get("passed")),
        _gate("certificate_row_count", EXPECTED_CANDIDATE_COUNT, len(certificate_keys)),
        _gate("unique_certificate_keys", EXPECTED_CANDIDATE_COUNT, len(set(certificate_keys))),
        _gate("proposal_certificate_keys_match", sorted(bank_keys), sorted(certificate_keys)),
        _gate(
            "certificate_checkpoint_input_count", EXPECTED_CANDIDATE_COUNT, len(checkpoint_inputs)
        ),
        _gate(
            "certificate_checkpoint_keys_match",
            sorted(bank_keys),
            sorted(str(row.get("attempt_key")) for row in checkpoint_inputs),
        ),
        _gate(
            "factor_checkpoint_count",
            len(EXPECTED_FACTOR_HASHES),
            len(energy.get("checkpoint_paths", [])),
        ),
        _gate(
            "factor_checkpoint_hash_roster",
            [
                {"arm": arm, "seed": seed, "sha256": digest}
                for (arm, seed), digest in sorted(EXPECTED_FACTOR_HASHES.items())
            ],
            [
                {"arm": arm, "seed": seed, "sha256": digest}
                for (arm, seed), digest in sorted(declared_factor_hashes.items())
            ],
        ),
        _gate(
            "certificate_binds_proposal_bank",
            EXPECTED_SOURCE_HASHES["proposal_bank"],
            certificates.get("source_artifact_hashes", {}).get("bank_artifact", {}).get("sha256"),
        ),
        _gate(
            "selection_binds_proposal_bank",
            EXPECTED_SOURCE_HASHES["proposal_bank"],
            _upstream_source_hash(upstream, "proposal_bank"),
        ),
        _gate(
            "selection_binds_certificates",
            EXPECTED_SOURCE_HASHES["certificates"],
            _upstream_source_hash(upstream, "exact_certificates"),
        ),
        _gate(
            "selection_binds_energy_canary",
            EXPECTED_SOURCE_HASHES["energy_canary"],
            _upstream_source_hash(upstream, "energy_canary"),
        ),
        _gate(
            "shared_fixture_hash",
            bank.get("source_artifact_hashes", {}).get("fixture_artifact"),
            energy.get("source_artifact_hashes", {}).get("exp6955_fixture"),
        ),
        _gate("fresh_process_module", True, (Path(repo_root) / MODULE_PATH).is_file()),
        _gate("fresh_process_wrapper", True, (Path(repo_root) / WRAPPER_PATH).is_file()),
    ]
    for name, rows in authority_surfaces.items():
        checks.extend(
            [
                _gate(f"{name}_count", EXPECTED_CANDIDATE_COUNT, len(rows)),
                _gate(f"{name}_terminal", True, all(row.get("terminal") for row in rows)),
            ]
        )
    return _summary(checks)


def _empty_row_surfaces() -> JsonDict:
    """Return every required row list so blocked artifacts keep one schema."""

    return {field: [] for field in REQUIRED_FIELDS if field == "rows" or field.endswith("_rows")}


def _stable_value(value: Any) -> Any:
    """Remove measured timing recursively while preserving scientific content."""

    if isinstance(value, Mapping):
        return {
            key: _stable_value(child)
            for key, child in value.items()
            if key not in {"duration_s", "arm_latency_ms", "reproducibility_checksum"}
        }
    if isinstance(value, list):
        return [_stable_value(child) for child in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable audit evidence and exclude only measured timing fields."""

    return sha256_bytes(canonical_json(_stable_value(artifact)))


def _blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    hash_rows: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write the full fail-closed schema without inventing replay evidence."""

    summary = _summary(checks)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "blocked",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": summary,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": _source_hash_map(hash_rows),
        **_empty_row_surfaces(),
        "audit_rows": list(checks),
        "hash_rows": list(hash_rows),
        "random_seed": RANDOM_SEED,
        "certified_selection_audit_complete_score": 0,
        "audited_certified_energy_positive_score": 0,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_certified_selection_cold_audit",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _row_contradictions(
    rows: Sequence[Mapping[str, Any]], prefix: str, *, expected_field: str = "passed"
) -> list[JsonDict]:
    """Convert failed terminal comparisons into explicit contradiction rows."""

    contradictions = []
    for index, row in enumerate(rows):
        if row.get(expected_field):
            continue
        contradictions.append(
            {
                "check": f"{prefix}:{index}",
                "expected_value": True,
                "observed_value": False,
                "terminal": True,
            }
        )
    return contradictions


def build_from_repo(
    repo_root: Path = REPO_ROOT,
    *,
    run_date: str = RUN_DATE,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    fresh_process: bool = True,
    parent_pid: int | None = None,
) -> JsonDict:
    """Build the audit from pinned files after a byte-first fail-closed preflight."""

    started = time.perf_counter()
    root = Path(repo_root)
    loaded = load_frozen_sources(root)
    if not loaded["passed"]:
        return _blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            hash_rows=loaded["hash_rows"],
            checks=loaded["hash_rows"],
        )
    sources = loaded["sources"]
    proposal = audit_proposal_bank(sources["proposal_bank"])
    preconditions = preflight_loaded_sources(root, sources, proposal)
    checkpoint_entries = _checkpoint_entries(sources["energy_canary"])
    checkpoint_hash_rows = audit_checkpoint_hashes(root, checkpoint_entries)
    checkpoint_hash_gate = _gate(
        "all_factor_checkpoint_hashes",
        True,
        len(checkpoint_hash_rows) == len(EXPECTED_FACTOR_HASHES)
        and all(row["passed"] for row in checkpoint_hash_rows),
    )
    preflight_checks = [*preconditions["checks"], checkpoint_hash_gate]
    all_hash_rows = [*loaded["hash_rows"], *proposal["raw_hash_rows"], *checkpoint_hash_rows]
    if not all(row["passed"] for row in preflight_checks):
        return _blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            hash_rows=all_hash_rows,
            checks=preflight_checks,
        )
    checkpoint_result = load_factor_checkpoints(
        root, sources["energy_canary"], checkpoint_hash_rows
    )
    if not checkpoint_result["passed"]:
        load_gate = _gate("factor_checkpoint_reload", True, checkpoint_result["passed"])
        return _blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            hash_rows=all_hash_rows,
            checks=[*preflight_checks, load_gate],
        )
    certificate_result = replay_certificates(
        proposal["reparsed_attempts"],
        sources["certificates"],
        sources["certificate_checkpoint"],
    )
    candidates = [candidate_from_attempt(row) for row in proposal["reparsed_attempts"]]
    label_rows = [
        audit_label_payload(str(row["attempt_key"]), row["selector_payload"]) for row in candidates
    ]
    arm_rows = recompute_arm_scores(
        candidates, checkpoint_result["models"], sources["upstream_selection"]
    )
    exact_by_source = {str(row["attempt_key"]): row for row in certificate_result["rows"]}
    for row in arm_rows:
        candidate = next(item for item in candidates if item["attempt_key"] == row["attempt_key"])
        row["exact_mapping_correct"] = bool(
            exact_by_source[candidate["source_attempt_key"]]["exact_mapping_correct"]
        )
    selection_result = recompute_selections(
        candidates, certificate_result["rows"], sources["upstream_selection"]
    )
    grouped_metrics = reduce_group_metrics(
        selection_result["selection_rows"],
        selection_result["groups"],
        selection_result["labels"],
        arm_rows,
        bootstrap_samples=bootstrap_samples,
    )
    interval = grouped_metrics["confidence_interval_rows"][0]
    controls = {
        "label_isolation": all(row["passed"] for row in label_rows),
        "candidate_order": all(row["passed"] for row in selection_result["candidate_order_rows"]),
        "tie_policy": all(row["passed"] for row in selection_result["tie_policy_rows"]),
        "arm_parity": len(arm_rows) == EXPECTED_CANDIDATE_COUNT * len(ARM_ORDER),
        "oracle_not_deployed": all(
            not row["oracle_used_for_selection"] for row in selection_result["selection_rows"]
        ),
    }
    replay_positive = bool(
        interval["ci95_lower"] is not None
        and interval["ci95_lower"] > 0
        and grouped_metrics["headroom_capture_rate"] is not None
        and grouped_metrics["headroom_capture_rate"] >= 0.2
        and all(controls.values())
    )
    replay_summary = {
        "arm_metrics": grouped_metrics["arm_metrics"],
        "strongest_non_oracle_baseline": grouped_metrics["strongest_non_oracle_baseline"],
        "confidence_interval": interval,
        "available_oracle_headroom": grouped_metrics["available_oracle_headroom"],
        "captured_oracle_headroom": grouped_metrics["captured_oracle_headroom"],
        "headroom_capture_rate": grouped_metrics["headroom_capture_rate"],
        "complete_score": 1,
        "replay_positive": replay_positive,
    }
    compared = compare_upstream_aggregates(sources["upstream_selection"], replay_summary)
    contradictions = list(compared["contradictions"])
    contradictions.extend(_row_contradictions(certificate_result["rows"], "certificate_replay"))
    contradictions.extend(_row_contradictions(arm_rows, "arm_recompute"))
    contradictions.extend(
        _row_contradictions(
            selection_result["selection_rows"],
            "selection_replay",
            expected_field="upstream_matches",
        )
    )
    contradictions.extend(_row_contradictions(selection_result["tie_policy_rows"], "tie_policy"))
    contradictions.extend(
        _row_contradictions(selection_result["candidate_order_rows"], "candidate_order")
    )
    contradictions.extend(_row_contradictions(label_rows, "label_isolation"))
    audit_rows = [
        *preflight_checks,
        _gate("factor_checkpoint_reload", True, checkpoint_result["passed"]),
        _gate("certificate_replay", True, certificate_result["passed"]),
        _gate("arm_score_replay", True, all(row["passed"] for row in arm_rows)),
        _gate("label_isolation", True, controls["label_isolation"]),
        _gate("arm_group_parity", True, controls["arm_parity"]),
        _gate("candidate_order_invariance", True, controls["candidate_order"]),
        _gate("tie_policy", True, controls["tie_policy"]),
        _gate("aggregate_consistency", True, not contradictions),
        {
            **_gate("fresh_process_entrypoint", True, fresh_process),
            "process_id": os.getpid(),
            "parent_process_id": parent_pid,
        },
    ]
    audit_complete = bool(
        all(row.get("terminal") for row in audit_rows)
        and len(certificate_result["rows"]) == EXPECTED_CANDIDATE_COUNT
        and len(selection_result["selection_rows"]) == EXPECTED_GROUP_COUNT * len(ARM_ORDER)
    )
    audited_positive, verdict_class, honest_verdict = reduce_verdict(
        audit_complete=audit_complete,
        replay_positive=replay_positive,
        upstream=sources["upstream_selection"],
        contradictions=contradictions,
    )
    gate_checks = [
        _gate("certified_selection_audit_complete_score", 1, int(audit_complete)),
        _gate(
            "paired_ci95_lower_strictly_positive",
            True,
            interval["ci95_lower"] is not None and interval["ci95_lower"] > 0,
        ),
        _gate(
            "headroom_capture_at_least_20_percent",
            True,
            grouped_metrics["headroom_capture_rate"] is not None
            and grouped_metrics["headroom_capture_rate"] >= 0.2,
        ),
        _gate("all_control_checks", True, all(controls.values())),
        _gate("aggregate_consistency", True, not contradictions),
        _gate(
            "upstream_positive_authority",
            True,
            sources["upstream_selection"].get("verdict_class") == "positive"
            and sources["upstream_selection"].get("certified_energy_positive_score") == 1,
        ),
    ]
    gate_summary = _summary(gate_checks)
    gate_summary.update(
        {
            "strongest_non_oracle_baseline": grouped_metrics["strongest_non_oracle_baseline"],
            "available_oracle_headroom": grouped_metrics["available_oracle_headroom"],
            "captured_oracle_headroom": grouped_metrics["captured_oracle_headroom"],
            "headroom_capture_rate": grouped_metrics["headroom_capture_rate"],
            "raw_positive_gate_passed": replay_positive,
            "control_checks": controls,
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete" if audit_complete else "partial",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": _summary(preflight_checks),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": time.perf_counter() - started,
        "source_artifact_hashes": _source_hash_map(loaded["hash_rows"]),
        "rows": selection_result["selection_rows"],
        "audit_rows": audit_rows,
        "hash_rows": all_hash_rows,
        "certificate_replay_rows": certificate_result["rows"],
        "checkpoint_reload_rows": checkpoint_result["rows"],
        "arm_recompute_rows": arm_rows,
        "candidate_order_rows": selection_result["candidate_order_rows"],
        "proposal_budget_rows": proposal["proposal_budget_rows"],
        "label_isolation_rows": label_rows,
        "tie_policy_rows": selection_result["tie_policy_rows"],
        "selection_rows": selection_result["selection_rows"],
        "headroom_rows": grouped_metrics["headroom_rows"],
        "family_rows": grouped_metrics["family_rows"],
        "paired_metric_rows": grouped_metrics["paired_metric_rows"],
        "confidence_interval_rows": grouped_metrics["confidence_interval_rows"],
        "aggregate_consistency_rows": compared["rows"],
        "contradiction_report_rows": contradictions,
        "random_seed": RANDOM_SEED,
        "certified_selection_audit_complete_score": int(audit_complete),
        "audited_certified_energy_positive_score": audited_positive,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject schema, checksum, score, verdict, and contradiction drift."""

    errors = []
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
        return errors
    if set(artifact["field_principles"]) != set(REQUIRED_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle_mismatch")
    if (
        artifact["audited_certified_energy_positive_score"]
        and not artifact["certified_selection_audit_complete_score"]
    ):
        errors.append("positive_without_complete_audit")
    if artifact["contradiction_report_rows"] and artifact["verdict_class"] != "disqualified":
        errors.append("contradiction_requires_disqualified_verdict")
    expected_classes = {
        "blocked_certified_selection_cold_audit": "blocked",
        "partial_certified_selection_cold_audit": "partial",
        "complete_null_certified_selection_cold_audit": "null",
        "complete_positive_certified_selection_cold_audit": "positive",
        "complete_disqualified_certified_selection_cold_audit": "disqualified",
    }
    if expected_classes.get(str(artifact["honest_verdict"])) != artifact["verdict_class"]:
        errors.append("verdict_class_prefix_mismatch")
    if reproducibility_checksum(artifact) != artifact["reproducibility_checksum"]:
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_output(artifact: Mapping[str, Any], path: Path) -> None:
    """Publish validated JSON atomically so interruption cannot leave a partial result."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}-", delete=False
    ) as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def _child_command(*, date: str, repo_root: Path, output_path: Path, parent_pid: int) -> list[str]:
    """Build the isolated child command with explicit paths and process identity."""

    return [
        sys.executable,
        str(Path(repo_root) / MODULE_PATH),
        "--fresh-child",
        "--date",
        date,
        "--repo-root",
        str(repo_root),
        "--output",
        str(output_path),
        "--parent-pid",
        str(parent_pid),
    ]


def launch_fresh_process(date: str, repo_root: Path, output_path: Path) -> int:
    """Run the full audit in a new interpreter with no inherited Python objects."""

    completed = subprocess.run(
        _child_command(
            date=date, repo_root=repo_root, output_path=output_path, parent_pid=os.getpid()
        ),
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0 and completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by E2E command.
    """Expose the mandatory fresh-process command and its private child boundary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fresh-child", action="store_true")
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    output = args.output or root / RESULT_PATH
    if not args.fresh_child:
        return launch_fresh_process(args.date, root, output)
    artifact = build_from_repo(
        root,
        run_date=args.date,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        fresh_process=True,
        parent_pid=args.parent_pid,
    )
    write_output(artifact, output)
    print(
        json.dumps(
            {
                field: artifact[field]
                for field in (
                    "certified_selection_audit_complete_score",
                    "audited_certified_energy_positive_score",
                    "verdict_class",
                    "honest_verdict",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - required isolated process boundary.
    raise SystemExit(main())
