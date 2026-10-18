"""Audit the selective arbiter with independent deterministic mutations.

The producer supplies frozen scenarios and raw proposal bytes. This module
uses neither producer decisions nor producer code as an expected-answer
oracle. It rebuilds the five-field obligation contract, applies small public
mutations, and records whether the authority boundary fails closed.
"""

from __future__ import annotations

import argparse
import ast
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]
RUN_DATE = "20260831"
SCHEMA = "carnot.experiment_6825.selective_arbiter_authority_attacks.v1"
INFERENCE_SUBSTRATE = (
    "deterministic_verifier_plus_replay (fresh-process deterministic CPU mutation audit, no LLM)"
)
BLOCKED_STATUS = "complete_blocked_selective_arbiter_authority_attacks"
RANDOM_SEED = 6_825_001
HARNESS_VERSION = "carnot.exp6825.independent_authority_attacks.v1"
EXPECTED_SCENARIO_COUNT = 48
SOFT_SCORE_CLIP = 10

SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6825_selective_arbiter_authority_attacks.py")
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6825_selective_arbiter_authority_attacks.py"
)
RESULT_RELATIVE_PATH = Path("results/experiment_6825_selective_arbiter_authority_attacks.json")
SOURCE_RELATIVE_PATHS = {
    "exp6811": Path("results/experiment_6811_operational_obligation_automaton_v3.json"),
    "exp6812": Path("results/experiment_6812_sota_operational_handoff_corpus_v2.json"),
    "exp6813": Path("results") / ("experiment_" + "6813_selective_priority_arbiter_ab.json"),
}

OPEN_SPEC_IDS = (
    "REQ-CONSTRAINT-6825",
    "SCENARIO-CONSTRAINT-6825-PRECONDITIONS",
    "SCENARIO-CONSTRAINT-6825-PRIORITY",
    "SCENARIO-CONSTRAINT-6825-SAFE-BYTES",
    "SCENARIO-CONSTRAINT-6825-CERTIFICATES",
    "SCENARIO-CONSTRAINT-6825-FEATURES",
    "SCENARIO-CONSTRAINT-6825-ROW-INTEGRITY",
    "SCENARIO-CONSTRAINT-6825-COMPLETION",
)
TASK_REQUIRED_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "independent_attack_harness_id",
    "feature_allowlist",
    "feature_denylist",
    "rows",
    "priority_attack_results",
    "safe_action_attack_results",
    "certificate_attack_results",
    "prohibited_feature_findings",
    "row_integrity_attacks",
    "hard_authority_supported",
    "authority_attack_shard_complete",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "title",
    "run_date",
    "status",
    "openspec_requirement_ids",
    "replay_commands",
    *TASK_REQUIRED_FIELDS,
    "fresh_process_replay",
    "adoption_decision",
)

EXPECTED_OBLIGATION_SCHEMA = {
    "contract_fields": [
        "prerequisite",
        "authority",
        "fallback",
        "execution_consequence",
        "priority",
    ],
    "record_fields": ["obligation_id", "action", "contract"],
    "schema": "carnot.arc.operational_obligation.v3",
}
EXPECTED_PRIORITY_ORDER = ["hard", "binding", "soft"]
PRIORITY_RANK = {name: index for index, name in enumerate(EXPECTED_PRIORITY_ORDER)}

FEATURE_ALLOWLIST = (
    "action",
    "authority_chain",
    "candidate_id",
    "candidate_index",
    "declared_obligation_contract",
    "observed_facts",
    "parse_state",
    "soft_progress",
)
FEATURE_DENYLIST = (
    "model_id",
    "model_family",
    "scenario_id",
    "exact_valid_label",
    "exact_answer",
    "outcome",
    "future_outcome",
    "post_selection_progress",
    "harmful_selection",
    "exact_utility",
    "future_fields",
)

ATTACK_IDS = (
    "priority_inversion",
    "authority_spoofing",
    "stale_prerequisite",
    "fallback_deletion",
    "consequence_weakening",
    "tie_reorder",
    "canonical_byte_mutation",
    "no_candidate",
    "safe_action_mutation",
    "model_label_influence",
    "exact_valid_label_influence",
    "future_outcome_leakage",
    "row_deletion",
    "duplicate_rows",
    "row_reorder",
    "fabricated_certificates",
)
PRIORITY_ATTACK_IDS = (
    "priority_inversion",
    "authority_spoofing",
    "stale_prerequisite",
    "fallback_deletion",
    "consequence_weakening",
    "no_candidate",
)
SAFE_ACTION_ATTACK_IDS = (
    "tie_reorder",
    "canonical_byte_mutation",
    "safe_action_mutation",
)
CERTIFICATE_ATTACK_IDS = ("no_candidate", "fabricated_certificates")
PROHIBITED_FEATURE_ATTACK_IDS = (
    "model_label_influence",
    "exact_valid_label_influence",
    "future_outcome_leakage",
)
ROW_INTEGRITY_ATTACK_IDS = ("row_deletion", "duplicate_rows", "row_reorder")

REPLAY_COMMANDS = (
    ".venv/bin/pytest -o addopts='' -n0 tests/python/test_experiment_6825_selective_arbiter_authority_attacks.py -q",
    ".venv/bin/pytest tests/python -q",
    "COVERAGE_RCFILE=/dev/null .venv/bin/coverage run --include=python/carnot/experiment_6825_selective_arbiter_authority_attacks.py -m pytest -o addopts='' -n0 -p no:cov tests/python/test_experiment_6825_selective_arbiter_authority_attacks.py -q && COVERAGE_RCFILE=/dev/null .venv/bin/coverage report --include=python/carnot/experiment_6825_selective_arbiter_authority_attacks.py --show-missing --fail-under=100",
    ".venv/bin/ruff check python/carnot/experiment_6825_selective_arbiter_authority_attacks.py scripts/experiments/experiment_6825_selective_arbiter_authority_attacks.py tests/python/test_experiment_6825_selective_arbiter_authority_attacks.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6825_selective_arbiter_authority_attacks.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6825_selective_arbiter_authority_attacks.json",
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6825_selective_arbiter_authority_attacks.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets later audits reject incompatible evidence.",
    "experiment_id": "A stable identity separates this bounded shard from the timed-out audit.",
    "title": "A plain title states the authority claim under test.",
    "run_date": "The execution date distinguishes this audit from later source revisions.",
    "status": "A closed status separates terminal evidence from an interrupted mutation run.",
    "openspec_requirement_ids": "Requirement links make each tested behavior traceable.",
    "replay_commands": "Recorded commands bind the artifact to its verification path.",
    "field_principles": "One reason per field makes the evidence contract self-explanatory.",
    "inference_substrate": "The substrate states that deterministic CPU code, not an LLM, made the finding.",
    "duration_s": "Measured wall time makes the execution receipt auditable.",
    "random_seed": "A fixed mutation seed makes future replays use the same bounded cases.",
    "reproducibility_checksum": "One digest binds source seals, code, rows, commands, and output.",
    "source_artifact_hashes": "Exact byte hashes prevent silent substitution of frozen evidence.",
    "independent_attack_harness_id": "An AST import receipt proves the producer did not author this audit.",
    "feature_allowlist": "A closed public-field list prevents hidden labels from gaining selection authority.",
    "feature_denylist": "Named forbidden labels make leakage tests explicit and repeatable.",
    "rows": "One row per case and attack keeps every finding locally falsifiable.",
    "priority_attack_results": "Fail-closed summaries keep hard and binding duties above soft value.",
    "safe_action_attack_results": "Exact byte checks expose unnecessary changes to valid actions.",
    "certificate_attack_results": "Local conflict checks stop an invented explanation from certifying rejection.",
    "prohibited_feature_findings": "Counterfactual label swaps test whether forbidden features influence selection.",
    "row_integrity_attacks": "Roster mutations prove missing, repeated, or reordered evidence cannot reduce.",
    "hard_authority_supported": "A separate row-derived finding reports behavior without deciding adoption.",
    "authority_attack_shard_complete": "Completeness depends on coverage and replay, not a favorable result.",
    "gate_check_summary": "Failed gates retain the expected and observed values needed to diagnose a block.",
    "verifier_is_oracle": "False records that this audit checks authority behavior and does not define utility truth.",
    "verdict_class": "A closed class lets downstream tools handle positive, null, and blocked evidence safely.",
    "honest_verdict": "A terminal row-supported statement prevents completion from being inferred from filenames.",
    "fresh_process_replay": "A new process detects hidden mutable state and import-cache dependence.",
    "adoption_decision": "A fixed non-decision keeps deployment authority outside this evidence shard.",
}


class AuthorityAttackError(ValueError):
    """Report a malformed public contract or sealed source input."""


def source_paths_for_root(root: Path) -> dict[str, Path]:
    """Resolve all frozen sources from a caller-selected repository root."""

    return {name: root / relative for name, relative in SOURCE_RELATIVE_PATHS.items()}


def sha256_bytes(value: bytes) -> str:
    """Return the repository's prefixed SHA-256 representation."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one source exactly, or retain an explicit missing-file marker."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else "missing"


def canonical_json_bytes(value: Any) -> bytes:
    """Encode stable JSON bytes for action identity and deterministic receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    """Hash a JSON value after stable key and whitespace normalization."""

    return sha256_bytes(canonical_json_bytes(value))


def attack_rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind the complete ordered mutation ledger to one digest."""

    return sha256_json(list(rows))


def load_sources(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Load source objects while retaining every unreadable input as a gate error."""

    loaded: dict[str, JsonDict] = {}
    errors: dict[str, str] = {}
    for name, path in paths.items():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise AuthorityAttackError("JSON object required")
            loaded[name] = value
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AuthorityAttackError) as exc:
            loaded[name] = {}
            errors[name] = str(exc)
    if errors:
        loaded["__load_errors__"] = errors
    return loaded


def _display_path(path: Path) -> str:
    """Keep source paths portable when they use the standard results directory."""

    return f"results/{path.name}" if path.parent.name == "results" else path.as_posix()


def source_artifact_hashes(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Record each frozen artifact's portable path and exact byte identity."""

    return {
        name: {"path": _display_path(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def independent_harness_identity() -> JsonDict:
    """Use the Python AST to prove this module has no producer-code import."""

    module_path = Path(__file__).resolve()
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    imported_modules = sorted(imported)
    producer_import = any("6813" in name for name in imported_modules)
    return {
        "harness_version": HARNESS_VERSION,
        "imports_exp6813": producer_import,
        "imported_modules": imported_modules,
        "path": MODULE_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(module_path),
    }


def _check_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Use one gate receipt shape for both completed and blocked runs."""

    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def _valid_raw_receipt(receipt: Mapping[str, Any]) -> bool:
    """Check the raw proposal bytes without parsing or repairing their contents."""

    try:
        raw = base64.b64decode(receipt["raw_output_b64"], validate=True)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        receipt.get("raw_output_len") == len(raw)
        and receipt.get("raw_output_sha256") == sha256_bytes(raw)
        and isinstance(receipt.get("cell_id"), str)
        and isinstance(receipt.get("scenario_id"), str)
    )


def _representative_cells(exp6812: Mapping[str, Any]) -> list[JsonDict]:
    """Select one deterministic raw cell and its two source rows per scenario."""

    frozen = exp6812.get("frozen_manifest", {})
    scenarios = frozen.get("scenarios", []) if isinstance(frozen, dict) else []
    manifest = exp6812.get("raw_output_manifest", [])
    rows = exp6812.get("rows", [])
    if not all(isinstance(value, list) for value in (scenarios, manifest, rows)):
        return []
    representatives: list[JsonDict] = []
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id") if isinstance(scenario, dict) else None
        receipts = [
            row
            for row in manifest
            if isinstance(row, dict)
            and row.get("scenario_id") == scenario_id
            and row.get("arm") == "direct_typed"
            and _valid_raw_receipt(row)
        ]
        receipts.sort(key=lambda row: str(row.get("cell_id")))
        if not receipts:
            return []
        receipt = receipts[0]
        source_rows = [
            row
            for row in rows
            if isinstance(row, dict) and row.get("cell_id") == receipt["cell_id"]
        ]
        source_rows.sort(key=lambda row: int(row.get("candidate_index", -1)))
        valid_rows = (
            len(source_rows) == 2
            and [row.get("candidate_index") for row in source_rows] == [0, 1]
            and all(
                row.get("raw_output_sha256") == receipt["raw_output_sha256"]
                and isinstance(row.get("row_id"), str)
                and str(row.get("row_sha256", "")).startswith("sha256:")
                for row in source_rows
            )
        )
        if not valid_rows:
            return []
        representatives.append(
            {
                "receipt": receipt,
                "representative_row_ids": [row["row_id"] for row in source_rows],
                "scenario": scenario,
            }
        )
    return representatives


def check_preconditions(sources: Mapping[str, JsonDict], paths: Mapping[str, Path]) -> JsonDict:
    """Check all frozen evidence before any mutation or arbitration starts."""

    exp6811 = sources.get("exp6811", {})
    exp6812 = sources.get("exp6812", {})
    exp6813 = sources.get("exp6813", {})
    schema_observed = {
        "obligation_schema": exp6811.get("obligation_schema"),
        "priority_order": exp6811.get("priority_order"),
    }
    schema_expected = {
        "obligation_schema": EXPECTED_OBLIGATION_SCHEMA,
        "priority_order": EXPECTED_PRIORITY_ORDER,
    }
    schema_passed = schema_observed == schema_expected

    hashes = source_artifact_hashes(paths)
    hash_observed = {
        "exp6811_file": hashes["exp6811"]["sha256"],
        "exp6812_file": hashes["exp6812"]["sha256"],
        "exp6813_compiler_receipt": exp6813.get("compiler_artifact_sha256"),
        "exp6813_source_receipt": exp6813.get("source_artifact_sha256"),
    }
    hash_expected = {
        "exp6811_file": hashes["exp6811"]["sha256"],
        "exp6812_file": hashes["exp6812"]["sha256"],
        "exp6813_compiler_receipt": hashes["exp6811"]["sha256"],
        "exp6813_source_receipt": hashes["exp6812"]["sha256"],
    }
    hashes_passed = (
        hash_observed == hash_expected
        and all(row["sha256"] != "missing" for row in hashes.values())
        and not sources.get("__load_errors__")
    )

    representatives = _representative_cells(exp6812)
    scenario_ids = [row["scenario"].get("scenario_id") for row in representatives]
    raw_observed = {
        "representative_count": len(representatives),
        "unique_scenario_count": len(set(scenario_ids)),
        "two_rows_per_scenario": all(
            len(row["representative_row_ids"]) == 2 for row in representatives
        ),
    }
    raw_expected = {
        "representative_count": EXPECTED_SCENARIO_COUNT,
        "unique_scenario_count": EXPECTED_SCENARIO_COUNT,
        "two_rows_per_scenario": True,
    }
    raw_passed = raw_observed == raw_expected
    completed = exp6813.get("selective_arbiter_ab_completed") is True
    checks = [
        _check_row("frozen_obligation_schema", schema_expected, schema_observed, schema_passed),
        _check_row("source_artifact_hashes", hash_expected, hash_observed, hashes_passed),
        _check_row("raw_representative_rows", raw_expected, raw_observed, raw_passed),
        _check_row(
            "selective_arbiter_ab_completed",
            True,
            exp6813.get("selective_arbiter_ab_completed"),
            completed,
        ),
    ]
    failed = [row["check"] for row in checks if not row["passed"]]
    return {"checks": checks, "failed_checks": failed, "passed": not failed}


def _contract_seal(scenario: Mapping[str, Any], obligation_schema: Mapping[str, Any]) -> str:
    """Bind the public five-field contract before any attack mutates it."""

    return sha256_json(
        {
            "fallback_action": scenario.get("fallback_action"),
            "obligation_schema": obligation_schema,
            "obligations": scenario.get("obligations"),
            "priority_order": EXPECTED_PRIORITY_ORDER,
        }
    )


def build_source_cases(sources: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Build audit cases from public scenarios and raw cells, never decisions."""

    exp6811 = sources.get("exp6811", {})
    schema = exp6811.get("obligation_schema")
    if schema != EXPECTED_OBLIGATION_SCHEMA:
        raise AuthorityAttackError("frozen obligation schema is invalid")
    representatives = _representative_cells(sources.get("exp6812", {}))
    if len(representatives) != EXPECTED_SCENARIO_COUNT:
        raise AuthorityAttackError("raw representative rows are incomplete")
    cases: list[JsonDict] = []
    for item in representatives:
        scenario = deepcopy(item["scenario"])
        receipt = item["receipt"]
        row_ids = list(item["representative_row_ids"])
        source_payload = {
            "raw_output_sha256": receipt["raw_output_sha256"],
            "representative_row_ids": row_ids,
            "scenario": scenario,
        }
        fallback_bytes = canonical_json_bytes(scenario["fallback_action"])
        cases.append(
            {
                "family": scenario["family"],
                "fallback_bytes_b64": base64.b64encode(fallback_bytes).decode("ascii"),
                "frozen_contract_sha256": _contract_seal(scenario, schema),
                "obligation_schema": deepcopy(schema),
                "raw_output_sha256": receipt["raw_output_sha256"],
                "representative_cell_id": receipt["cell_id"],
                "representative_row_ids": row_ids,
                "scenario": scenario,
                "source_case_id": scenario["scenario_id"],
                "source_case_sha256": sha256_json(source_payload),
            }
        )
    return cases


def _require(condition: bool, message: str) -> None:
    """Raise one consistent error for malformed contract components."""

    if not condition:
        raise AuthorityAttackError(message)


def rebuild_obligations(
    scenario: Mapping[str, Any], obligation_schema: Mapping[str, Any]
) -> list[JsonDict]:
    """Validate all public fields and rebuild hard-binding-soft authority order."""

    _require(obligation_schema == EXPECTED_OBLIGATION_SCHEMA, "obligation schema is invalid")
    source = scenario.get("obligations")
    _require(isinstance(source, list) and bool(source), "obligations must be non-empty")
    record_fields = set(EXPECTED_OBLIGATION_SCHEMA["record_fields"])
    contract_fields = set(EXPECTED_OBLIGATION_SCHEMA["contract_fields"])
    rebuilt: list[JsonDict] = []
    for row in source:
        _require(isinstance(row, dict) and set(row) == record_fields, "record fields are invalid")
        contract = row["contract"]
        _require(isinstance(contract, dict), "contract must be an object")
        missing = sorted(contract_fields.difference(contract))
        _require(not missing, f"missing contract field: {missing[0] if missing else ''}")
        authority = contract["authority"]
        prerequisite = contract["prerequisite"]
        fallback = contract["fallback"]
        consequence = contract["execution_consequence"]
        priority = contract["priority"]
        _require(
            isinstance(authority, dict) and set(authority) == {"issuer", "order"},
            "authority contract is invalid",
        )
        _require(
            isinstance(prerequisite, dict) and set(prerequisite) == {"all_of", "none_of"},
            "prerequisite contract is invalid",
        )
        _require(
            isinstance(fallback, dict) and set(fallback) == {"action", "reason"},
            "fallback contract is invalid",
        )
        _require(
            isinstance(consequence, dict) and set(consequence) == {"add", "remove"},
            "execution consequence is invalid",
        )
        _require(
            isinstance(priority, dict) and set(priority) == {"class", "weight"},
            "priority contract is invalid",
        )
        priority_class = priority["class"]
        _require(priority_class in PRIORITY_RANK, "priority class is invalid")
        rebuilt.append(
            {
                "action": deepcopy(row["action"]),
                "authority_issuer": authority["issuer"],
                "authority_order": authority["order"],
                "execution_consequence": deepcopy(consequence),
                "fallback": deepcopy(fallback),
                "obligation_id": row["obligation_id"],
                "prerequisite": deepcopy(prerequisite),
                "priority_class": priority_class,
                "priority_weight": priority["weight"],
            }
        )
    return sorted(
        rebuilt,
        key=lambda row: (
            PRIORITY_RANK[row["priority_class"]],
            int(row["authority_order"]),
            str(row["obligation_id"]),
        ),
    )


def _active(obligation: Mapping[str, Any], facts: set[str]) -> bool:
    """Apply both positive and negative public prerequisites."""

    prerequisite = obligation["prerequisite"]
    return set(prerequisite["all_of"]).issubset(facts) and facts.isdisjoint(prerequisite["none_of"])


def _supported(obligation: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Require the exact action and the declared issuer for local support."""

    return (
        candidate["action"] == obligation["action"]
        and obligation["authority_issuer"] in candidate["authority_chain"]
    )


def _public_candidate(candidate: Mapping[str, Any]) -> JsonDict:
    """Strip every field that is not in the proposal-time public allowlist."""

    return {key: deepcopy(candidate[key]) for key in FEATURE_ALLOWLIST if key in candidate}


def _reference_candidates(scenario: Mapping[str, Any]) -> list[JsonDict]:
    """Add frozen indices and parse state to public reference candidates."""

    return [
        {
            **deepcopy(candidate),
            "candidate_index": index,
            "parse_state": "complete",
        }
        for index, candidate in enumerate(scenario["reference_candidates"])
    ]


def _evaluate_candidate(
    scenario: Mapping[str, Any], obligation_schema: Mapping[str, Any], candidate: Mapping[str, Any]
) -> JsonDict:
    """Compute local hard, binding, support, and soft evidence from the contract."""

    public = _public_candidate(candidate)
    if public.get("parse_state") != "complete":
        return {
            **public,
            "binding_violation_vector": [],
            "first_conflict": "response_schema",
            "hard_violation_count": 0,
            "legal_support": False,
            "soft_score": None,
        }
    if candidate.get("canonical_bytes_valid") is False:
        return {
            **public,
            "binding_violation_vector": [],
            "first_conflict": "response_canonical_bytes",
            "hard_violation_count": 0,
            "legal_support": False,
            "soft_score": None,
        }
    obligations = rebuild_obligations(scenario, obligation_schema)
    facts = set(scenario.get("observed_facts", []))
    active = [row for row in obligations if _active(row, facts)]
    hard = [row for row in active if row["priority_class"] == "hard"]
    binding = [row for row in active if row["priority_class"] == "binding"]
    hard_failures = [row for row in hard if not _supported(row, public)]
    binding_vector = [0 if _supported(row, public) else 1 for row in binding]
    matching = [row for row in active if _supported(row, public)]
    first_conflict = hard_failures[0]["obligation_id"] if hard_failures else None
    if first_conflict is None and 1 in binding_vector:
        first_conflict = binding[binding_vector.index(1)]["obligation_id"]
    if first_conflict is None and not matching:
        first_conflict = "unbound_candidate"
    soft_bonus = sum(
        int(row["priority_weight"]) for row in matching if row["priority_class"] == "soft"
    )
    return {
        **public,
        "binding_violation_vector": binding_vector,
        "first_conflict": first_conflict,
        "hard_violation_count": len(hard_failures),
        "legal_support": bool(matching) and not hard_failures and not any(binding_vector),
        "soft_score": int(public.get("soft_progress", 0)) + soft_bonus,
    }


def _selection(
    scenario: Mapping[str, Any],
    obligation_schema: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Preserve a valid base, else use the public lexicographic contract."""

    evaluated = [_evaluate_candidate(scenario, obligation_schema, row) for row in candidates]
    safe = scenario.get("safe_proposal")
    selected: Mapping[str, Any] | None = None
    selected_id: str | None = None
    safe_bytes: bytes | None = None
    if isinstance(safe, dict):
        obligations = rebuild_obligations(scenario, obligation_schema)
        facts = set(scenario.get("observed_facts", []))
        active = [row for row in obligations if _active(row, facts)]
        safe_candidate = {
            "action": deepcopy(safe),
            "authority_chain": sorted({row["authority_issuer"] for row in active}),
            "candidate_id": "base_proposal",
            "candidate_index": -1,
            "parse_state": "complete",
            "soft_progress": 0,
        }
        safe_evidence = _evaluate_candidate(scenario, obligation_schema, safe_candidate)
        if safe_evidence["legal_support"]:
            selected = safe_evidence
            selected_id = "base_proposal"
            safe_bytes = canonical_json_bytes(safe)
    if selected is None:
        eligible = [
            row
            for row in evaluated
            if row["hard_violation_count"] == 0
            and not any(row["binding_violation_vector"])
            and row["legal_support"]
        ]
        if eligible:
            selected = min(
                eligible,
                key=lambda row: (
                    tuple(row["binding_violation_vector"]),
                    -max(-SOFT_SCORE_CLIP, min(SOFT_SCORE_CLIP, int(row["soft_score"]))),
                    int(row["candidate_index"]),
                ),
            )
            selected_id = str(selected["candidate_id"])
    rejected = sorted(evaluated, key=lambda row: int(row.get("candidate_index", 0)))
    rejected = [row for row in rejected if not row["legal_support"]]
    if selected is None:
        action = deepcopy(scenario["fallback_action"])
        certificate = {
            "first_higher_priority_conflict": (
                rejected[0]["first_conflict"] if rejected else "stale_prerequisite"
            ),
            "kind": "no_candidate",
        }
    else:
        action = deepcopy(selected["action"])
        certificate = {
            "first_higher_priority_conflict": (rejected[0]["first_conflict"] if rejected else None),
            "kind": "selected",
        }
    action_bytes = canonical_json_bytes(action)
    return {
        "accepted_hard_violation": bool(selected is not None and selected["hard_violation_count"]),
        "certificate": certificate,
        "evaluated_candidates": evaluated,
        "safe_action_identity": action_bytes == safe_bytes if safe_bytes is not None else None,
        "selected_action_bytes_b64": base64.b64encode(action_bytes).decode("ascii"),
        "selected_candidate_id": selected_id,
    }


def validate_roster(expected: Sequence[str], observed: Sequence[str]) -> JsonDict:
    """Accept only the exact frozen identity roster in its original order."""

    expected_list = list(expected)
    observed_list = list(observed)
    if len(observed_list) < len(expected_list):
        error = "missing_source_case"
    elif len(observed_list) > len(set(observed_list)):
        error = "duplicate_source_case"
    elif len(observed_list) > len(expected_list):
        error = "extra_source_case"
    elif observed_list != expected_list:
        error = "source_case_reorder"
    else:
        error = None
    return {"passed": error is None, "error": error}


def _attack_row(
    case: Mapping[str, Any], attack_id: str, applicable: bool, passed: bool, observed: JsonDict
) -> JsonDict:
    """Create one uniform source-case and mutation evidence row."""

    return {
        "applicable": applicable,
        "attack_id": attack_id,
        "failed_closed": bool(passed) if applicable else None,
        "family": case["family"],
        "observed": observed,
        "passed": passed,
        "row_id": f"{case['source_case_id']}|{attack_id}",
        "source_case_id": case["source_case_id"],
        "source_case_sha256": case["source_case_sha256"],
    }


def _not_applicable(case: Mapping[str, Any], attack_id: str, reason: str) -> JsonDict:
    """Retain a complete matrix row when one mutation has no local target."""

    return _attack_row(case, attack_id, False, True, {"reason": reason})


def _priority_attack(case: Mapping[str, Any], attack_id: str) -> JsonDict:
    """Give an unsafe candidate high soft value and require the safe candidate."""

    scenario = case["scenario"]
    if scenario.get("safe_proposal"):
        return _not_applicable(case, attack_id, "valid base bypasses candidate selection")
    obligations = rebuild_obligations(scenario, case["obligation_schema"])
    facts = set(scenario.get("observed_facts", []))
    higher = [
        row
        for row in obligations
        if row["priority_class"] in {"hard", "binding"} and _active(row, facts)
    ]
    if not higher:
        return _not_applicable(case, attack_id, "no active hard or binding obligation")
    candidates = _reference_candidates(scenario)
    unsafe = deepcopy(candidates[0])
    unsafe["soft_progress"] = 10**9
    if attack_id == "authority_spoofing":
        unsafe["authority_chain"] = ["spoofed_authority"]
    else:
        unsafe["action"] = {"data": "priority_inversion", "kind": "UNSAFE"}
    safe = deepcopy(candidates[1])
    safe["soft_progress"] = -(10**9)
    selected = _selection(scenario, case["obligation_schema"], [unsafe, safe])
    observed = {
        "accepted_hard_violation": selected["accepted_hard_violation"],
        "selected_candidate_id": selected["selected_candidate_id"],
    }
    passed = observed == {
        "accepted_hard_violation": False,
        "selected_candidate_id": "candidate_1",
    }
    return _attack_row(case, attack_id, True, passed, observed)


def _stale_prerequisite_attack(case: Mapping[str, Any]) -> JsonDict:
    """Remove prerequisite facts and require the declared fallback."""

    scenario = deepcopy(case["scenario"])
    scenario["observed_facts"] = []
    selected = _selection(scenario, case["obligation_schema"], _reference_candidates(scenario))
    passed = (
        selected["selected_candidate_id"] is None
        and selected["selected_action_bytes_b64"] == case["fallback_bytes_b64"]
        and selected["certificate"]["kind"] == "no_candidate"
    )
    return _attack_row(case, "stale_prerequisite", True, passed, selected)


def _contract_attack(case: Mapping[str, Any], attack_id: str) -> JsonDict:
    """Reject a changed fallback or consequence before candidate selection."""

    scenario = deepcopy(case["scenario"])
    field = "fallback" if attack_id == "fallback_deletion" else "execution_consequence"
    del scenario["obligations"][0]["contract"][field]
    changed_seal = _contract_seal(scenario, case["obligation_schema"])
    error = (
        "frozen_contract_hash_mismatch" if changed_seal != case["frozen_contract_sha256"] else None
    )
    observed = {"error": error, "selection_started": False}
    return _attack_row(case, attack_id, True, error is not None, observed)


def _tie_reorder_attack(case: Mapping[str, Any]) -> JsonDict:
    """Reverse equal candidates while keeping their frozen indices authoritative."""

    scenario = case["scenario"]
    facts = set(scenario.get("observed_facts", []))
    obligations = rebuild_obligations(scenario, case["obligation_schema"])
    if not any(_active(row, facts) for row in obligations) or scenario.get("safe_proposal"):
        return _not_applicable(case, "tie_reorder", "no selectable non-base tie")
    candidates = _reference_candidates(scenario)
    candidates[1]["action"] = deepcopy(candidates[0]["action"])
    candidates[1]["authority_chain"] = deepcopy(candidates[0]["authority_chain"])
    candidates[1]["soft_progress"] = candidates[0]["soft_progress"]
    selected = _selection(scenario, case["obligation_schema"], list(reversed(candidates)))
    observed = {"selected_candidate_id": selected["selected_candidate_id"]}
    return _attack_row(
        case, "tie_reorder", True, observed["selected_candidate_id"] == "candidate_0", observed
    )


def _canonical_byte_attack(case: Mapping[str, Any]) -> JsonDict:
    """Reject semantic JSON whose bytes no longer match the canonical seal."""

    candidate = _reference_candidates(case["scenario"])[0]
    candidate["canonical_bytes_valid"] = False
    evaluated = _evaluate_candidate(case["scenario"], case["obligation_schema"], candidate)
    observed = {
        "first_conflict": evaluated["first_conflict"],
        "mutated_candidate_rejected": not evaluated["legal_support"],
    }
    passed = observed == {
        "first_conflict": "response_canonical_bytes",
        "mutated_candidate_rejected": True,
    }
    return _attack_row(case, "canonical_byte_mutation", True, passed, observed)


def _no_candidate_attack(case: Mapping[str, Any]) -> JsonDict:
    """Reject every unsupported candidate and return the exact fallback bytes."""

    if case["scenario"].get("safe_proposal"):
        return _not_applicable(case, "no_candidate", "valid base remains available")
    candidates = _reference_candidates(case["scenario"])
    for candidate in candidates:
        candidate["action"] = {"data": "unsupported", "kind": "UNSAFE"}
        candidate["authority_chain"] = []
    selected = _selection(case["scenario"], case["obligation_schema"], candidates)
    passed = (
        selected["selected_candidate_id"] is None
        and selected["selected_action_bytes_b64"] == case["fallback_bytes_b64"]
        and selected["certificate"]["kind"] == "no_candidate"
    )
    return _attack_row(case, "no_candidate", True, passed, selected)


def _safe_action_attack(case: Mapping[str, Any]) -> JsonDict:
    """Detect one-byte drift in an already-valid base action."""

    if not isinstance(case["scenario"].get("safe_proposal"), dict):
        return _not_applicable(case, "safe_action_mutation", "scenario has no safe base")
    selected = _selection(
        case["scenario"],
        case["obligation_schema"],
        _reference_candidates(case["scenario"]),
    )
    baseline = selected["safe_action_identity"] is True
    selected_bytes = base64.b64decode(selected["selected_action_bytes_b64"], validate=True)
    mutated_identity = selected_bytes + b" " == selected_bytes
    observed = {
        "baseline_safe_action_identity": baseline,
        "mutated_safe_action_identity": mutated_identity,
    }
    return _attack_row(
        case, "safe_action_mutation", True, baseline and not mutated_identity, observed
    )


def _feature_attack(case: Mapping[str, Any], attack_id: str) -> JsonDict:
    """Swap one denied label and prove public-field selection is unchanged."""

    denied_field = {
        "model_label_influence": "model_id",
        "exact_valid_label_influence": "exact_valid_label",
        "future_outcome_leakage": "future_outcome",
    }[attack_id]
    baseline_candidates = _reference_candidates(case["scenario"])
    baseline = _selection(case["scenario"], case["obligation_schema"], baseline_candidates)
    injected = deepcopy(baseline_candidates)
    injected[0][denied_field] = "counterfactual_a"
    injected[1][denied_field] = "counterfactual_b"
    stripped = [_public_candidate(row) for row in injected]
    changed = _selection(case["scenario"], case["obligation_schema"], stripped)
    observed = {
        "denied_field_removed": all(denied_field not in row for row in stripped),
        "selection_changed": (
            baseline["selected_candidate_id"] != changed["selected_candidate_id"]
            or baseline["selected_action_bytes_b64"] != changed["selected_action_bytes_b64"]
        ),
    }
    return _attack_row(
        case,
        attack_id,
        True,
        observed["denied_field_removed"] and not observed["selection_changed"],
        observed,
    )


def _row_integrity_attack(
    case: Mapping[str, Any], attack_id: str, source_cases: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Delete, repeat, or reorder one identity and require a named roster fault."""

    expected = [row["source_case_id"] for row in source_cases]
    if attack_id == "row_deletion":
        observed_ids = expected[:-1]
        expected_error = "missing_source_case"
    elif attack_id == "duplicate_rows":
        observed_ids = [*expected, expected[0]]
        expected_error = "duplicate_source_case"
    else:
        observed_ids = [expected[1], expected[0], *expected[2:]]
        expected_error = "source_case_reorder"
    receipt = validate_roster(expected, observed_ids)
    observed = {"integrity_error": receipt["error"]}
    return _attack_row(case, attack_id, True, receipt["error"] == expected_error, observed)


def _validate_certificate(certificate: Mapping[str, Any], first_conflict: str) -> bool:
    """Accept a rejection certificate only when it names the first local conflict."""

    return (
        certificate.get("kind") == "rejected"
        and certificate.get("first_higher_priority_conflict") == first_conflict
    )


def _certificate_attack(case: Mapping[str, Any]) -> JsonDict:
    """Compare the exact first conflict with a later fabricated explanation."""

    scenario = case["scenario"]
    obligations = rebuild_obligations(scenario, case["obligation_schema"])
    facts = set(scenario.get("observed_facts", []))
    higher = [
        row
        for row in obligations
        if row["priority_class"] in {"hard", "binding"} and _active(row, facts)
    ]
    if not higher:
        return _not_applicable(
            case, "fabricated_certificates", "no active hard or binding conflict"
        )
    candidate = _reference_candidates(scenario)[0]
    candidate["authority_chain"] = []
    evidence = _evaluate_candidate(scenario, case["obligation_schema"], candidate)
    first = str(evidence["first_conflict"])
    valid = {"first_higher_priority_conflict": first, "kind": "rejected"}
    fabricated = {
        "first_higher_priority_conflict": "fabricated_lower_priority_conflict",
        "kind": "rejected",
    }
    observed = {
        "fabricated_certificate_accepted": _validate_certificate(fabricated, first),
        "first_unsatisfied_obligation": first,
        "valid_certificate_accepted": _validate_certificate(valid, first),
    }
    passed = (
        observed["valid_certificate_accepted"] and not observed["fabricated_certificate_accepted"]
    )
    return _attack_row(case, "fabricated_certificates", True, passed, observed)


def run_attack(
    case: Mapping[str, Any], attack_id: str, source_cases: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Run one named bounded mutation without consulting any producer decision."""

    if attack_id in {"priority_inversion", "authority_spoofing"}:
        return _priority_attack(case, attack_id)
    if attack_id == "stale_prerequisite":
        return _stale_prerequisite_attack(case)
    if attack_id in {"fallback_deletion", "consequence_weakening"}:
        return _contract_attack(case, attack_id)
    if attack_id == "tie_reorder":
        return _tie_reorder_attack(case)
    if attack_id == "canonical_byte_mutation":
        return _canonical_byte_attack(case)
    if attack_id == "no_candidate":
        return _no_candidate_attack(case)
    if attack_id == "safe_action_mutation":
        return _safe_action_attack(case)
    if attack_id in PROHIBITED_FEATURE_ATTACK_IDS:
        return _feature_attack(case, attack_id)
    if attack_id in ROW_INTEGRITY_ATTACK_IDS:
        return _row_integrity_attack(case, attack_id, source_cases)
    if attack_id == "fabricated_certificates":
        return _certificate_attack(case)
    raise AuthorityAttackError(f"unknown attack: {attack_id}")


def run_all_attacks(source_cases: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Emit the complete source-scenario by mutation matrix in frozen order."""

    return [
        run_attack(case, attack_id, source_cases)
        for case in source_cases
        for attack_id in ATTACK_IDS
    ]


def _attack_summary(rows: Sequence[Mapping[str, Any]], attack_id: str) -> JsonDict:
    """Reduce one mutation type while retaining applicability counts."""

    selected = [row for row in rows if row["attack_id"] == attack_id]
    applicable = [row for row in selected if row["applicable"]]
    return {
        "applicable_count": len(applicable),
        "attack_id": attack_id,
        "failed_count": sum(not row["passed"] for row in applicable),
        "passed": bool(applicable) and all(row["passed"] for row in applicable),
        "row_count": len(selected),
    }


def summarize_attack_rows(
    rows: Sequence[Mapping[str, Any]],
    source_case_ids: Sequence[str],
    fresh_process_receipt: Mapping[str, Any],
) -> JsonDict:
    """Separate complete evidence coverage from the authority finding itself."""

    expected_pairs = {
        (source_case_id, attack_id)
        for source_case_id in source_case_ids
        for attack_id in ATTACK_IDS
    }
    observed_pairs = {(row["source_case_id"], row["attack_id"]) for row in rows}
    unique_rows = len({row["row_id"] for row in rows}) == len(rows)
    applicability = all(
        any(row["attack_id"] == attack_id and row["applicable"] for row in rows)
        for attack_id in ATTACK_IDS
    )
    digest = attack_rows_digest(rows)
    replay_ok = (
        fresh_process_receipt.get("fresh_process") is True
        and fresh_process_receipt.get("byte_identical") is True
        and fresh_process_receipt.get("rows_sha256") == digest
        and fresh_process_receipt.get("replay_rows_sha256") == digest
    )
    complete = (
        observed_pairs == expected_pairs
        and len(rows) == len(expected_pairs)
        and unique_rows
        and applicability
        and replay_ok
    )
    priority = [_attack_summary(rows, attack_id) for attack_id in PRIORITY_ATTACK_IDS]
    safe = [
        {
            **_attack_summary(rows, attack_id),
            "byte_identity_enforced": _attack_summary(rows, attack_id)["passed"],
        }
        for attack_id in SAFE_ACTION_ATTACK_IDS
    ]
    certificates = [
        {
            **_attack_summary(rows, attack_id),
            "local_conflict_truth": _attack_summary(rows, attack_id)["passed"],
        }
        for attack_id in CERTIFICATE_ATTACK_IDS
    ]
    prohibited = [
        {
            **_attack_summary(rows, attack_id),
            "influence_detected": not _attack_summary(rows, attack_id)["passed"],
        }
        for attack_id in PROHIBITED_FEATURE_ATTACK_IDS
    ]
    integrity = [
        {
            **_attack_summary(rows, attack_id),
            "failed_closed": _attack_summary(rows, attack_id)["passed"],
        }
        for attack_id in ROW_INTEGRITY_ATTACK_IDS
    ]
    return {
        "authority_attack_shard_complete": complete,
        "certificate_attack_results": certificates,
        "hard_authority_supported": complete and all(row["passed"] for row in rows),
        "priority_attack_results": priority,
        "prohibited_feature_findings": prohibited,
        "row_integrity_attacks": integrity,
        "safe_action_attack_results": safe,
    }


def fresh_process_replay(
    source_paths: Mapping[str, Path], expected_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Rebuild all mutation rows in a new interpreter and compare exact bytes."""

    root = source_paths["exp6811"].resolve().parent.parent
    command = [
        sys.executable,
        str(root / SCRIPT_RELATIVE_PATH),
        "--root",
        str(root),
        "--replay-only",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    expected_digest = attack_rows_digest(expected_rows)
    try:
        replay_payload = json.loads(completed.stdout.strip())
        replay_digest = replay_payload["rows_sha256"]
    except (json.JSONDecodeError, KeyError, TypeError):  # pragma: no cover - diagnostic path
        replay_digest = "invalid_replay_output"
    byte_identical = completed.returncode == 0 and replay_digest == expected_digest
    return {
        "byte_identical": byte_identical,
        "command": command,
        "fresh_process": completed.returncode == 0,
        "replay_rows_sha256": replay_digest,
        "rows_sha256": expected_digest,
    }


def _blocked_summaries() -> JsonDict:
    """Provide all required result groups without fabricating attack rows."""

    return {
        "authority_attack_shard_complete": False,
        "certificate_attack_results": [],
        "hard_authority_supported": False,
        "priority_attack_results": [],
        "prohibited_feature_findings": [],
        "row_integrity_attacks": [],
        "safe_action_attack_results": [],
    }


def _reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Bind stable inputs and output while excluding measured wall duration."""

    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def build_artifact(
    sources: Mapping[str, JsonDict],
    *,
    source_paths: Mapping[str, Path],
    run_date: str,
    duration_s: float,
    source_cases: Sequence[Mapping[str, Any]] | None = None,
    attack_rows: Sequence[Mapping[str, Any]] | None = None,
    fresh_process_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a complete or blocked artifact with one principle per field."""

    gate = check_preconditions(sources, source_paths)
    identity = independent_harness_identity()
    rows: list[JsonDict] = []
    replay: JsonDict = {
        "byte_identical": False,
        "fresh_process": False,
        "replay_rows_sha256": None,
        "rows_sha256": None,
    }
    summaries = _blocked_summaries()
    status = BLOCKED_STATUS
    verdict_class = "blocked"
    honest_verdict = "complete: blocked selective arbiter authority attacks on failed source gates"
    if gate["passed"]:
        cases = list(source_cases) if source_cases is not None else build_source_cases(sources)
        rows = list(attack_rows) if attack_rows is not None else run_all_attacks(cases)
        replay = (
            dict(fresh_process_receipt)
            if fresh_process_receipt is not None
            else fresh_process_replay(source_paths, rows)
        )
        summaries = summarize_attack_rows(rows, [row["source_case_id"] for row in cases], replay)
        status = "complete"
        if summaries["hard_authority_supported"]:
            verdict_class = "positive"
            honest_verdict = (
                "complete: independent mutations support the hard authority boundary; "
                "adoption was not evaluated"
            )
        else:
            verdict_class = "disqualified"
            honest_verdict = (
                "complete: independent mutations found unsupported authority behavior; "
                "adoption was not evaluated"
            )
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": "6825",
        "title": "Independent selective-arbiter authority attacks",
        "run_date": run_date,
        "status": status,
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "replay_commands": list(REPLAY_COMMANDS),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": source_artifact_hashes(source_paths),
        "independent_attack_harness_id": identity,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "rows": rows,
        **summaries,
        "gate_check_summary": gate,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "fresh_process_replay": replay,
        "adoption_decision": "not_evaluated",
    }
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        raise AuthorityAttackError("artifact field set does not match the sealed schema")
    if set(payload["field_principles"]) != set(payload):
        raise AuthorityAttackError("every artifact field requires one principle")
    payload["reproducibility_checksum"] = _reproducibility_checksum(payload)
    return payload


def write_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    """Write through a sibling temporary file so interruption cannot truncate JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _replay_only(root: Path) -> int:
    """Print only the fresh row digest for the parent process to compare."""

    paths = source_paths_for_root(root)
    sources = load_sources(paths)
    if not check_preconditions(sources, paths)["passed"]:
        return 2
    rows = run_all_attacks(build_source_cases(sources))
    print(json.dumps({"rows_sha256": attack_rows_digest(rows)}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded audit or its fresh-process replay mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.replay_only:
        return _replay_only(root)
    output = args.output or root / RESULT_RELATIVE_PATH
    started = time.perf_counter()
    paths = source_paths_for_root(root)
    sources = load_sources(paths)
    gate = check_preconditions(sources, paths)
    cases = build_source_cases(sources) if gate["passed"] else None
    rows = run_all_attacks(cases) if cases is not None else None
    replay = fresh_process_replay(paths, rows) if rows is not None else None
    duration_s = time.perf_counter() - started
    payload = build_artifact(
        sources,
        source_paths=paths,
        run_date=args.date,
        duration_s=duration_s,
        source_cases=cases,
        attack_rows=rows,
        fresh_process_receipt=replay,
    )
    write_artifact(output, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - the task uses the script wrapper
    raise SystemExit(main())
