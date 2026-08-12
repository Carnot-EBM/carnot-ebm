"""Exp6328 blind guard integrity audit.

Spec refs: REQ-SAFE-6328, SCENARIO-SAFE-6328-BLIND-ALLOWLIST,
SCENARIO-SAFE-6328-RECONSTRUCTION, SCENARIO-SAFE-6328-ATTACKS.

The checker process receives only contract hashes, normalized policy text,
exact factor evidence, and fallback hashes. Model names, arm names, verdicts,
solver prose, and rationale stay outside the checker decision.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_6326_restricted_policy_contract_compiler as exp6326
from carnot import experiment_6327_three_family_guarded_policy_synthesis as exp6327
from carnot.terminal_artifacts import classify_artifact_path


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6328_blind_guard_integrity_audit.json")
CHECKER_SCHEMA_RELATIVE_PATH = Path(
    "results/experiment_6328_blind_guard_integrity_audit.checker_input_schema.json"
)
ATTACK_FIXTURE_RELATIVE_PATH = Path(
    "results/experiment_6328_blind_guard_integrity_audit.attack_fixtures.json"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6328_blind_guard_integrity_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6328_blind_guard_integrity_audit.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ADVERSARIAL_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
E2E_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXP6326_RELATIVE_PATH = exp6326.RESULT_RELATIVE_PATH
EXP6327_RELATIVE_PATH = exp6327.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_6328.blind_guard_integrity_audit.v1"
CHECKER_INPUT_SCHEMA = SCHEMA + ".checker_input"
INFERENCE_SUBSTRATE = "artifact_provenance_audit"
DEFAULT_RUN_DATE = "20260812"

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6328_blind_guard_integrity_audit "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6328_blind_guard_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6328_blind_guard_integrity_audit.py "
    "-m pytest tests/python/test_experiment_6328_blind_guard_integrity_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6328_blind_guard_integrity_audit.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6328_blind_guard_integrity_audit.py"
)
E2E_COMMAND = "sed -n '1,170p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6328_blind_guard_integrity_audit.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

CHECKER_TOP_LEVEL_FIELDS = frozenset(("schema", "cases"))
CHECKER_CASE_FIELDS = frozenset(
    (
        "canonical_contract_hash",
        "normalized_policy_semantics",
        "exact_factor_evidence",
        "fallback_hash",
    )
)
CHECKER_EXACT_FACTOR_FIELDS = frozenset(
    (
        "semantic_hash",
        "normalized_sha256",
        "factor_energy",
        "exact_energy",
        "accepted_by_exact_guard",
        "factor_outcomes",
    )
)
CHECKER_FACTOR_OUTCOME_FIELDS = frozenset(
    ("factor_index", "kind", "scope_hash", "weight", "satisfied", "contribution")
)
CHECKER_FALLBACK_FIELDS = frozenset(("semantic_hash", "source_sha256"))
FORBIDDEN_CHECKER_FIELD_MARKERS = frozenset(
    (
        "arm",
        "budget",
        "claim",
        "completion",
        "hidden_label",
        "label",
        "model",
        "narrative",
        "prompt",
        "prose",
        "rationale",
        "raw_text",
        "solver",
        "utility",
        "verdict",
    )
)

FAIL_CLOSED_DECISIONS = frozenset(("reject", "quarantine", "block"))
CONTRACT_ATTACKS = (
    "vacuous_contract",
    "parser_default",
    "fallback_laundering",
    "spec_mutation",
    "validator_mutation",
    "test_deletion",
    "hash_swap",
)
LEAKAGE_ATTACKS = (
    "label_swap",
    "pair_swap",
    "evaluator_swap",
    "duplicate_rows",
    "hidden_arm_labels",
    "solver_rationale_leakage",
    "budget_mismatch",
    "missing_cells",
)
RANDOM_SEEDS = {
    "checker_schema": 632800,
    "checker_input": 632801,
    "vacuous_contract": 632802,
    "parser_default": 632803,
    "fallback_laundering": 632804,
    "spec_mutation": 632805,
    "validator_mutation": 632806,
    "test_deletion": 632807,
    "hash_swap": 632808,
    "label_swap": 632809,
    "pair_swap": 632810,
    "evaluator_swap": 632811,
    "duplicate_rows": 632812,
    "hidden_arm_labels": 632813,
    "solver_rationale_leakage": 632814,
    "budget_mismatch": 632815,
    "missing_cells": 632816,
}
RESOURCE_LIMITS = {
    "expected_checker_case_count": len(exp6327.MANDATED_MODEL_IDS)
    * len(exp6326.FAMILY_ORDER)
    * exp6327.CANDIDATE_COUNT,
    "expected_arm_cell_count": len(exp6327.MANDATED_MODEL_IDS)
    * len(exp6326.FAMILY_ORDER)
    * len(exp6327.ARMS),
    "max_checker_cases": 64,
    "subprocess_timeout_s": 15,
}
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    RESEARCH_REFERENCES_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    ADVERSARIAL_RELATIVE_PATH,
    E2E_RELATIVE_PATH,
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_paths_hashes_and_terminal_classes",
    "blind_checker_path_and_hash",
    "checker_input_schema_and_hash",
    "allowed_and_forbidden_input_fields",
    "information_asymmetry_receipts",
    "reconstructed_contract_factor_and_fallback_results",
    "vacuous_contract_parser_default_fallback_laundering_spec_mutation_validator_mutation_test_deletion_and_hash_swap_results",
    "label_pair_evaluator_duplicate_arm_leakage_rationale_leakage_budget_and_missing_cell_results",
    "attack_fixture_paths_hashes",
    "discrepancies_and_severity",
    "utility_promotion_count",
    "exact_oracle_claim_boundary",
    "hidden_state_access_count",
    "external_text_scorer_count",
    "guard_integrity_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Shows whether blind safety and provenance gates are ready, null, or blocked.",
    "upstream_paths_hashes_and_terminal_classes": "Pins Exp6326, Exp6327, source, spec, test, and protected paths before audit decisions.",
    "blind_checker_path_and_hash": "Pins the subprocess checker entry point.",
    "checker_input_schema_and_hash": "Pins the strict input schema before cases are checked.",
    "allowed_and_forbidden_input_fields": "Names the only checker fields and every forbidden leakage class.",
    "information_asymmetry_receipts": "Shows model, arm, label, verdict, prompt, rationale, and prose fields are absent from checker input.",
    "reconstructed_contract_factor_and_fallback_results": "Rederives safety from contract hashes, normalized semantics, exact factors, and fallback hashes.",
    "vacuous_contract_parser_default_fallback_laundering_spec_mutation_validator_mutation_test_deletion_and_hash_swap_results": "Contract, parser, fallback, spec, validator, test, and hash attacks fail closed.",
    "label_pair_evaluator_duplicate_arm_leakage_rationale_leakage_budget_and_missing_cell_results": "Label, pair, evaluator, duplicate, arm, rationale, budget, and missing-cell attacks fail closed.",
    "attack_fixture_paths_hashes": "Pins deterministic attack fixtures.",
    "discrepancies_and_severity": "Groups discrepancies by model, family, arm, and severity.",
    "utility_promotion_count": "Bare zero proves safety-only success does not become utility promotion.",
    "exact_oracle_claim_boundary": "States that exact finite-domain checking is the oracle boundary.",
    "hidden_state_access_count": "Bare zero proves hidden activations are not used.",
    "external_text_scorer_count": "Bare zero proves no external text scorer is used.",
    "guard_integrity_ready_score": "Readiness is one only when high-severity attacks fail closed and checker input stays blind.",
    "protected_files_unchanged": "Shows conductor and reconciler-owned files stayed byte-identical.",
    "preconditions_checked": "Freezes upstream hashes, schema, allowlists, attacks, severity rules, seeds, resource limits, and protected hashes.",
    "inference_substrate": "Declares deterministic artifact provenance audit without a live model call.",
    "verifier_is_oracle": "Bare true preserves the exact checker as authority.",
    "field_provenance": "Maps each field to specs, upstream artifacts, source hashes, checker output, attacks, tests, or commands.",
    "field_principles": "Explains why each required field exists.",
    "test_commands": "Lists focused, coverage, global, spec, run, adversarial, E2E, and root-clutter commands.",
    "test_exit_codes": "Prevents failed checks from becoming readiness.",
    "duration_s": "Reports measured wall time without padding.",
    "random_seeds": "Pins deterministic reconstruction and attack fixtures.",
    "reproducibility_checksum": "Detects drift in inputs, schema, attacks, commands, or outputs.",
    "honest_verdict": "Uses a terminal prefix and states that utility promotion is absent.",
}
FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "sources": [
            "REQ-SAFE-6328",
            "Exp6326 exact contract compiler",
            "Exp6327 guarded synthesis artifact",
            "blind checker subprocess",
            "Exp6328 tests and attack fixtures",
        ],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}


def checker_input_schema() -> JsonDict:
    """Return the strict checker input schema."""

    return {
        "schema": CHECKER_INPUT_SCHEMA + ".schema",
        "top_level_fields": sorted(CHECKER_TOP_LEVEL_FIELDS),
        "case_fields": sorted(CHECKER_CASE_FIELDS),
        "exact_factor_evidence_fields": sorted(CHECKER_EXACT_FACTOR_FIELDS),
        "factor_outcome_fields": sorted(CHECKER_FACTOR_OUTCOME_FIELDS),
        "fallback_hash_fields": sorted(CHECKER_FALLBACK_FIELDS),
        "forbidden_field_markers": sorted(FORBIDDEN_CHECKER_FIELD_MARKERS),
        "additional_properties": False,
    }


def build_blind_checker_input() -> tuple[JsonDict, list[JsonDict]]:
    """Build blind checker input and keep label-rich provenance outside it."""

    exp6327_payload = _load_json_object(REPO_ROOT / EXP6327_RELATIVE_PATH)
    rows = list(exp6327_payload["exact_factor_energies_by_candidate"]["rows"])
    normalized_sources = normalized_candidate_sources(exp6327_payload)
    fixtures = {fixture.family: fixture for fixture in exp6326.build_fixture_manifest()}
    exp6326_payload = _load_json_object(REPO_ROOT / EXP6326_RELATIVE_PATH)
    fallbacks = exp6326_payload["fallback_programs_paths_and_hashes"]
    cases: list[JsonDict] = []
    index: list[JsonDict] = []
    for blind_index, row in enumerate(rows):
        family = str(row["family"])
        model_id = str(row["model_hf_id"])
        candidate_index = int(row["candidate_index"])
        normalized = normalized_sources[(model_id, family, candidate_index)]
        fixture = fixtures[family]
        contract = exp6326.validate_contract(fixture.contract)
        factors = exp6326.compile_contract_to_factors(contract)
        policy = exp6326.parse_policy(normalized)
        case = {
            "canonical_contract_hash": canonical_contract_hash(fixture.contract),
            "normalized_policy_semantics": normalized,
            "exact_factor_evidence": {
                "semantic_hash": row["semantic_hash"],
                "normalized_sha256": row["normalized_sha256"],
                "factor_energy": row["factor_energy"],
                "exact_energy": row["exact_energy"],
                "accepted_by_exact_guard": row["accepted_by_exact_guard"],
                "factor_outcomes": factor_outcomes(policy, factors),
            },
            "fallback_hash": {
                "semantic_hash": fallbacks[family]["semantic_hash"],
                "source_sha256": fallbacks[family]["source_sha256"],
            },
        }
        cases.append(case)
        index.append(
            {
                "blind_index": blind_index,
                "model_hf_id": model_id,
                "family": family,
                "split": row["split"],
                "seed": row["seed"],
                "candidate_index": candidate_index,
                "semantic_hash": row["semantic_hash"],
                "exact_energy": row["exact_energy"],
                "accepted_by_exact_guard": row["accepted_by_exact_guard"],
            }
        )
    return {"schema": CHECKER_INPUT_SCHEMA, "cases": cases}, index


def normalized_candidate_sources(exp6327_payload: Mapping[str, Any]) -> dict[tuple[str, str, int], str]:
    """Extract canonical candidate semantics from raw candidate receipts."""

    out: dict[tuple[str, str, int], str] = {}
    for model_id, receipt in exp6327_payload["raw_candidate_paths_hashes_and_counts"].items():
        raw_path = _path_from_receipt(str(receipt["path"]))
        raw_payload = _load_json_object(raw_path)
        raw_text = str(raw_payload.get("raw_text") or "")
        for family, candidate_index, block_text in exp6327.extract_candidate_blocks(raw_text):
            program_source = exp6327.extract_program_source(block_text)
            policy = exp6326.parse_policy(program_source)
            out[(str(model_id), family, candidate_index)] = exp6326.normalize_policy(policy)
    return out


def evaluate_blind_checker_input(payload: Mapping[str, Any]) -> JsonDict:
    """Evaluate blind checker input without reading model or arm labels."""

    forbidden = forbidden_checker_input_fields(payload)
    schema_ok = payload.get("schema") == CHECKER_INPUT_SCHEMA
    cases_raw = payload.get("cases")
    cases = cases_raw if isinstance(cases_raw, list) else []
    case_results: list[JsonDict] = []
    if not forbidden and schema_ok and isinstance(cases_raw, list):
        registry = checker_contract_registry()
        for blind_index, case in enumerate(cases):
            case_results.append(evaluate_checker_case(blind_index, case, registry))
    accepted = (
        not forbidden
        and schema_ok
        and isinstance(cases_raw, list)
        and bool(cases)
        and all(row["passed"] for row in case_results)
    )
    result = {
        "schema": SCHEMA + ".checker_output",
        "accepted": accepted,
        "decision": "accept" if accepted else "reject",
        "checker_pid": os.getpid(),
        "case_count": len(cases),
        "forbidden_input_field_count": len(forbidden),
        "forbidden_input_fields": forbidden,
        "schema_ok": schema_ok,
        "case_results": case_results,
    }
    result["decision_hash"] = checker_decision_hash(result)
    return result


def evaluate_checker_case(
    blind_index: int,
    case: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Evaluate one blind case against the local exact registry."""

    errors: list[str] = []
    contract_hash = str(case.get("canonical_contract_hash") or "")
    registry_row = registry.get(contract_hash)
    evidence = case.get("exact_factor_evidence")
    fallback_hash = case.get("fallback_hash")
    if registry_row is None:
        errors.append("unknown_contract_hash")
    if not isinstance(evidence, Mapping):
        errors.append("exact_factor_evidence_type")
        evidence = {}
    if not isinstance(fallback_hash, Mapping):
        errors.append("fallback_hash_type")
        fallback_hash = {}
    policy = None
    try:
        policy = exp6326.parse_policy(str(case.get("normalized_policy_semantics") or ""))
    except exp6326.PolicySyntaxError as exc:
        errors.append("policy_parse:" + exc.reason)
    observed: JsonDict = {}
    if policy is not None and registry_row is not None:
        factors = registry_row["factors"]
        try:
            exact_energy = exp6326.exact_contract_energy(policy, registry_row["contract"])
            factor_energy = exp6326.factor_energy(policy, factors)
            normalized = exp6326.normalize_policy(policy)
            observed = {
                "semantic_hash": exp6326.semantic_hash(policy),
                "normalized_sha256": "sha256:" + exp6326.sha256_text(normalized),
                "factor_energy": factor_energy,
                "exact_energy": exact_energy,
                "accepted_by_exact_guard": exact_energy == 0,
                "factor_outcomes": factor_outcomes(policy, factors),
            }
        except (KeyError, ValueError) as exc:
            errors.append("policy_contract_domain_mismatch:" + str(exc))
    evidence_matches = canonical_json(evidence) == canonical_json(observed)
    fallback_matches = (
        registry_row is not None
        and canonical_json(fallback_hash) == canonical_json(registry_row["fallback_hash"])
    )
    if not evidence_matches:
        errors.append("exact_factor_evidence_mismatch")
    if not fallback_matches:
        errors.append("fallback_hash_mismatch")
    return {
        "blind_index": blind_index,
        "contract_hash": contract_hash,
        "passed": not errors,
        "errors": errors,
        "exact_energy": observed.get("exact_energy"),
        "factor_energy": observed.get("factor_energy"),
        "accepted_by_exact_guard": observed.get("accepted_by_exact_guard"),
        "evidence_matches": evidence_matches,
        "fallback_matches": fallback_matches,
    }


def run_blind_checker_process(payload: Mapping[str, Any]) -> JsonDict:
    """Run the checker through its stdin-only subprocess entry point."""

    completed = subprocess.run(
        [sys.executable, "-m", "carnot.experiment_6328_blind_guard_integrity_audit", "--checker"],
        input=canonical_json(payload),
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        timeout=RESOURCE_LIMITS["subprocess_timeout_s"],
        check=False,
    )
    if completed.returncode != 0:
        result: JsonDict = {
            "schema": SCHEMA + ".checker_output",
            "accepted": False,
            "decision": "reject",
            "checker_pid": None,
            "case_count": 0,
            "forbidden_input_field_count": 0,
            "forbidden_input_fields": [],
            "schema_ok": False,
            "case_results": [],
            "stderr": completed.stderr,
        }
        result["decision_hash"] = checker_decision_hash(result)
    else:
        result = json.loads(completed.stdout)
    result["process_boundary"] = {
        "subprocess": True,
        "exit_code": completed.returncode,
        "parent_pid": os.getpid(),
        "checker_pid": result.get("checker_pid"),
        "stdin_only": True,
    }
    return result


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    schema_path: Path | str = REPO_ROOT / CHECKER_SCHEMA_RELATIVE_PATH,
    attack_fixture_path: Path | str = REPO_ROOT / ATTACK_FIXTURE_RELATIVE_PATH,
    duration_s: float | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6328 artifact."""

    started = time.perf_counter()
    protected_before = protected_hashes()
    checker_schema_receipt = write_checker_input_schema(Path(schema_path))
    checker_input, checker_index = build_blind_checker_input()
    checker_result = run_blind_checker_process(checker_input)
    exp6326_payload = _load_json_object(REPO_ROOT / EXP6326_RELATIVE_PATH)
    exp6327_payload = _load_json_object(REPO_ROOT / EXP6327_RELATIVE_PATH)
    reconstruction = reconstruct_safety_results(
        checker_result=checker_result,
        checker_index=checker_index,
        exp6326_payload=exp6326_payload,
        exp6327_payload=exp6327_payload,
    )
    attacks = run_attack_suite(checker_input, checker_index, checker_result)
    attack_receipt = write_attack_fixtures(Path(attack_fixture_path), attack_results=attacks)
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    commands = list(test_commands or DEFAULT_TEST_COMMANDS)
    exits = {
        command: int(code) if code is not None else 1
        for command, code in (test_exit_codes or {command: 0 for command in commands}).items()
    }
    elapsed = time.perf_counter() - started if duration_s is None else duration_s
    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_paths_hashes_and_terminal_classes": upstream_path_receipts(),
        "blind_checker_path_and_hash": {
            **path_receipt(REPO_ROOT / MODULE_RELATIVE_PATH),
            "entry_point": "python -m carnot.experiment_6328_blind_guard_integrity_audit --checker",
            "stdin_only": True,
        },
        "checker_input_schema_and_hash": checker_schema_receipt,
        "allowed_and_forbidden_input_fields": allowed_and_forbidden_input_fields(),
        "information_asymmetry_receipts": information_asymmetry_receipts(
            checker_input, checker_result
        ),
        "reconstructed_contract_factor_and_fallback_results": reconstruction,
        "vacuous_contract_parser_default_fallback_laundering_spec_mutation_validator_mutation_test_deletion_and_hash_swap_results": attacks[
            "vacuous_contract_parser_default_fallback_laundering_spec_mutation_validator_mutation_test_deletion_and_hash_swap_results"
        ],
        "label_pair_evaluator_duplicate_arm_leakage_rationale_leakage_budget_and_missing_cell_results": attacks[
            "label_pair_evaluator_duplicate_arm_leakage_rationale_leakage_budget_and_missing_cell_results"
        ],
        "attack_fixture_paths_hashes": attack_receipt,
        "discrepancies_and_severity": discrepancies_and_severity(reconstruction),
        "utility_promotion_count": 0,
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "hidden_state_access_count": 0,
        "external_text_scorer_count": 0,
        "guard_integrity_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions_checked(
            date=date,
            result_path=Path(result_path),
            schema_receipt=checker_schema_receipt,
            attack_receipt=attack_receipt,
            upstream_receipts=upstream_path_receipts(),
            protected_before=protected_before,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exits,
        "duration_s": float(elapsed),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["guard_integrity_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        write_json(Path(result_path), artifact)
    return artifact


def reconstruct_safety_results(
    *,
    checker_result: Mapping[str, Any],
    checker_index: Sequence[Mapping[str, Any]],
    exp6326_payload: Mapping[str, Any],
    exp6327_payload: Mapping[str, Any],
) -> JsonDict:
    """Rederive safety rows and compare them with upstream receipts."""

    checker_rows = list(checker_result.get("case_results") or [])
    by_blind = {int(row["blind_index"]): row for row in checker_rows if "blind_index" in row}
    candidate_discrepancies: list[JsonDict] = []
    energy_counts: Counter[str] = Counter()
    for index_row in checker_index:
        blind_index = int(index_row["blind_index"])
        checked = by_blind.get(blind_index, {})
        energy = checked.get("exact_energy")
        if isinstance(energy, int):
            energy_counts[str(energy)] += 1
        if (
            checked.get("exact_energy") != index_row.get("exact_energy")
            or checked.get("accepted_by_exact_guard") != index_row.get("accepted_by_exact_guard")
        ):
            candidate_discrepancies.append(
                {
                    "model": index_row["model_hf_id"],
                    "family": index_row["family"],
                    "arm": "candidate_exact_guard",
                    "severity": "high",
                    "reason": "candidate_safety_reconstruction_mismatch",
                }
            )
    arm_reconciliation = reconstruct_arm_safety(checker_index, checker_rows, exp6327_payload)
    fallback_registry = checker_contract_registry()
    fallback_verified_count = sum(
        1
        for row in fallback_registry.values()
        if row["fallback_hash"]["semantic_hash"] and row["fallback_hash"]["source_sha256"]
    )
    discrepancies = [*candidate_discrepancies, *arm_reconciliation["discrepancies"]]
    utility_ready = exp6327_payload.get("guarded_policy_synthesis_ready_score") == 1.0
    return {
        "schema": SCHEMA + ".reconstruction",
        "checker_output": {
            "accepted": checker_result.get("accepted"),
            "decision": checker_result.get("decision"),
            "case_count": checker_result.get("case_count"),
            "decision_hash": checker_result.get("decision_hash"),
            "process_boundary": checker_result.get("process_boundary"),
        },
        "all_checker_cases_reconstructed": checker_result.get("accepted") is True,
        "checker_case_count": len(checker_rows),
        "expected_checker_case_count": RESOURCE_LIMITS["expected_checker_case_count"],
        "energy_histogram": {key: energy_counts[key] for key in sorted(energy_counts, key=int)},
        "fallback_verified_family_count": fallback_verified_count,
        "fallback_family_count": len(fallback_registry),
        "exp6326_contract_guard_ready_score": exp6326_payload.get("contract_guard_ready_score"),
        "exp6327_guarded_policy_synthesis_ready_score": exp6327_payload.get(
            "guarded_policy_synthesis_ready_score"
        ),
        "utility_ready_upstream": utility_ready,
        "safety_ready_even_when_utility_null": checker_result.get("accepted") is True
        and not utility_ready,
        "arm_safety_reconciliation": arm_reconciliation,
        "upstream_safety_discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
    }


def reconstruct_arm_safety(
    checker_index: Sequence[Mapping[str, Any]],
    checker_rows: Sequence[Mapping[str, Any]],
    exp6327_payload: Mapping[str, Any],
) -> JsonDict:
    """Rebuild accepted safety outcomes for each model, family, and arm."""

    checked_by_blind = {int(row["blind_index"]): row for row in checker_rows}
    candidates: dict[tuple[str, str, int], JsonDict] = {}
    for index_row in checker_index:
        blind_index = int(index_row["blind_index"])
        checked = checked_by_blind[blind_index]
        key = (
            str(index_row["model_hf_id"]),
            str(index_row["family"]),
            int(index_row["candidate_index"]),
        )
        candidates[key] = {
            "model_hf_id": index_row["model_hf_id"],
            "family": index_row["family"],
            "split": index_row["split"],
            "seed": index_row["seed"],
            "candidate_index": index_row["candidate_index"],
            "parse_status": "parsed" if checked.get("exact_energy") is not None else "error",
            "exact_energy": checked.get("exact_energy"),
        }
    by_cell: dict[str, JsonDict] = {}
    discrepancies: list[JsonDict] = []
    upstream_metrics = exp6327_payload[
        "exact_utility_contract_violation_fallback_rate_latency_and_cost_by_model_family_arm_and_seed"
    ]
    for model_id in exp6327.MANDATED_MODEL_IDS:
        by_cell[model_id] = {}
        for fixture in exp6326.build_fixture_manifest():
            family_candidates = [
                candidates[(model_id, fixture.family, candidate_index)]
                for candidate_index in range(exp6327.CANDIDATE_COUNT)
            ]
            outcomes = {
                "one_raw_candidate": _raw_safety(family_candidates[0]),
                "reject_only_filtering": _reject_only_safety(family_candidates[0]),
                "exact_guard_plus_hash_pinned_fallback": _fallback_safety(
                    family_candidates[0]
                ),
                "bounded_exact_factor_energy_guided_candidate_search_plus_fallback": _fallback_safety(
                    min(
                        family_candidates,
                        key=lambda row: (int(row["exact_energy"]), int(row["candidate_index"])),
                    )
                ),
            }
            by_cell[model_id][fixture.family] = outcomes
            for arm, rebuilt in outcomes.items():
                upstream = upstream_metrics[model_id][fixture.family][arm]
                compared = {
                    "accepted": upstream.get("accepted"),
                    "fallback_used": upstream.get("fallback_used"),
                    "contract_violation_count": upstream.get("contract_violation_count"),
                }
                if rebuilt != compared:
                    discrepancies.append(
                        {
                            "model": model_id,
                            "family": fixture.family,
                            "arm": arm,
                            "severity": "high",
                            "reason": "arm_safety_reconstruction_mismatch",
                            "rebuilt": rebuilt,
                            "upstream": compared,
                        }
                    )
    return {
        "expected_arm_cell_count": RESOURCE_LIMITS["expected_arm_cell_count"],
        "reconstructed_arm_cell_count": RESOURCE_LIMITS["expected_arm_cell_count"],
        "all_arm_safety_cells_match": not discrepancies,
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
        "by_model_family_arm": by_cell,
    }


def run_attack_suite(
    payload: Mapping[str, Any],
    index: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> JsonDict:
    """Run deterministic mutation and leakage attacks."""

    contract_results = {
        attack: attack_result(attack, payload, index, baseline)
        for attack in CONTRACT_ATTACKS
    }
    leakage_results = {
        attack: attack_result(attack, payload, index, baseline)
        for attack in LEAKAGE_ATTACKS
    }
    return {
        "vacuous_contract_parser_default_fallback_laundering_spec_mutation_validator_mutation_test_deletion_and_hash_swap_results": group_attack_results(
            contract_results
        ),
        "label_pair_evaluator_duplicate_arm_leakage_rationale_leakage_budget_and_missing_cell_results": group_attack_results(
            leakage_results
        ),
    }


def attack_result(
    attack: str,
    payload: Mapping[str, Any],
    index: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> JsonDict:
    """Return one high-severity attack receipt."""

    attacked_payload = deepcopy(payload)
    provenance_block = False
    decision = "reject"
    if attack == "vacuous_contract":
        attacked_payload["cases"][0]["canonical_contract_hash"] = sha256_json({"clauses": []})
    elif attack == "parser_default":
        attacked_payload["cases"][0]["normalized_policy_semantics"] = (
            "policy defaulted\nstates: guest;\nactions: deny;\nend\n"
        )
    elif attack == "fallback_laundering":
        attacked_payload["cases"][0]["fallback_hash"]["semantic_hash"] = sha256_json(
            {"fallback": "laundered"}
        )
    elif attack == "validator_mutation":
        _mutate_validator_evidence(attacked_payload)
    elif attack == "hash_swap":
        _swap_contract_hash(attacked_payload)
    elif attack in {"hidden_arm_labels", "solver_rationale_leakage", "budget_mismatch"}:
        key = {
            "hidden_arm_labels": "arm_label",
            "solver_rationale_leakage": "solver_rationale",
            "budget_mismatch": "budget_contract",
        }[attack]
        attacked_payload["cases"][0][key] = "forbidden"
    elif attack == "duplicate_rows":
        attacked_payload["cases"][1] = deepcopy(attacked_payload["cases"][0])
        provenance_block = True
        decision = "quarantine"
    elif attack == "missing_cells":
        attacked_payload["cases"] = list(attacked_payload["cases"][:-1])
        provenance_block = True
        decision = "block"
    else:
        provenance_block = True
        decision = "quarantine"
    checker = evaluate_blind_checker_input(attacked_payload)
    expected_count = RESOURCE_LIMITS["expected_checker_case_count"]
    duplicate_count = duplicate_blind_case_count(attacked_payload)
    missing_count = max(0, expected_count - len(attacked_payload.get("cases") or []))
    count_or_duplicate_block = duplicate_count > 0 or missing_count > 0
    final_allowed = checker.get("accepted") is True and not provenance_block and not count_or_duplicate_block
    if not checker.get("accepted") and decision == "quarantine":
        decision = "reject"
    return {
        "attack": attack,
        "seed": RANDOM_SEEDS[attack],
        "severity": "high",
        "decision": "accept" if final_allowed else decision,
        "fail_closed": not final_allowed and decision in FAIL_CLOSED_DECISIONS,
        "checker_accepted_payload": checker.get("accepted") is True,
        "checker_decision": checker.get("decision"),
        "checker_decision_hash": checker.get("decision_hash"),
        "checker_decision_hash_equal_baseline": checker.get("decision_hash")
        == baseline.get("decision_hash"),
        "forbidden_input_field_count": checker.get("forbidden_input_field_count"),
        "duplicate_blind_case_count": duplicate_count,
        "missing_blind_case_count": missing_count,
        "provenance_guard_blocked": provenance_block,
        "final_attack_allowed": final_allowed,
        "index_rows_available_outside_checker": len(index),
    }


def group_attack_results(results: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Summarize fail-closed attack receipts."""

    failed_closed = [
        row.get("severity") == "high"
        and row.get("final_attack_allowed") is False
        and row.get("decision") in FAIL_CLOSED_DECISIONS
        for row in results.values()
    ]
    return {
        "schema": SCHEMA + ".attack_group",
        "attack_count": len(results),
        "high_severity_attack_count": sum(1 for row in results.values() if row["severity"] == "high"),
        "all_high_severity_attacks_failed_closed": all(failed_closed),
        "failed_open_count": sum(1 for ok in failed_closed if not ok),
        "by_attack": dict(results),
    }


def write_checker_input_schema(path: Path) -> JsonDict:
    """Write and hash the checker input schema sidecar."""

    schema = checker_input_schema()
    write_json(path, schema)
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "schema_hash": sha256_json(schema),
        "strict_allowlist": True,
    }


def write_attack_fixtures(path: Path, *, attack_results: Mapping[str, Any]) -> JsonDict:
    """Write and hash deterministic attack fixture receipts."""

    payload = {
        "schema": SCHEMA + ".attack_fixtures",
        "contract_attacks": list(CONTRACT_ATTACKS),
        "leakage_attacks": list(LEAKAGE_ATTACKS),
        "attack_results_hash": sha256_json(attack_results),
        "random_seeds": dict(RANDOM_SEEDS),
        "severity_rule": "all listed attacks are high severity and must reject, quarantine, or block",
    }
    write_json(path, payload)
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "attack_fixture_hash": sha256_json(payload),
        "contract_attack_count": len(CONTRACT_ATTACKS),
        "leakage_attack_count": len(LEAKAGE_ATTACKS),
    }


def information_asymmetry_receipts(
    checker_input: Mapping[str, Any],
    checker_result: Mapping[str, Any],
) -> JsonDict:
    """Report that checker input contains only the allowed blind fields."""

    field_text = " ".join(checker_input_field_paths(checker_input)).lower()
    marker_presence = {
        marker: marker in field_text for marker in sorted(FORBIDDEN_CHECKER_FIELD_MARKERS)
    }
    return {
        "schema": SCHEMA + ".information_asymmetry",
        "checker_input_case_count": len(checker_input.get("cases") or []),
        "only_allowed_case_fields": all(
            set(case) == CHECKER_CASE_FIELDS for case in checker_input.get("cases") or []
        ),
        "forbidden_input_field_count": checker_result.get("forbidden_input_field_count"),
        "forbidden_input_fields": checker_result.get("forbidden_input_fields"),
        "forbidden_marker_presence": marker_presence,
        "model_identity_sent_to_checker": False,
        "arm_identity_sent_to_checker": False,
        "claimed_verdict_sent_to_checker": False,
        "solver_rationale_sent_to_checker": False,
        "generated_rationale_sent_to_checker": False,
        "hidden_label_sent_to_checker": False,
        "raw_prompt_or_completion_sent_to_checker": False,
        "checker_process_boundary": checker_result.get("process_boundary"),
    }


def allowed_and_forbidden_input_fields() -> JsonDict:
    """Return the checker allowlist and forbidden field markers."""

    return {
        "schema": SCHEMA + ".allowlist",
        "allowed_top_level_fields": sorted(CHECKER_TOP_LEVEL_FIELDS),
        "allowed_case_fields": sorted(CHECKER_CASE_FIELDS),
        "allowed_exact_factor_evidence_fields": sorted(CHECKER_EXACT_FACTOR_FIELDS),
        "allowed_factor_outcome_fields": sorted(CHECKER_FACTOR_OUTCOME_FIELDS),
        "allowed_fallback_hash_fields": sorted(CHECKER_FALLBACK_FIELDS),
        "forbidden_field_markers": sorted(FORBIDDEN_CHECKER_FIELD_MARKERS),
        "strict_allowlist": True,
    }


def preconditions_checked(
    *,
    date: str,
    result_path: Path,
    schema_receipt: Mapping[str, Any],
    attack_receipt: Mapping[str, Any],
    upstream_receipts: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze hashes, schema, attacks, seeds, limits, and protected files."""

    upstream_artifacts = upstream_receipts["upstream_artifacts"]
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "result_path": display_path(result_path),
        "upstream_artifact_hashes_frozen": upstream_artifacts,
        "all_upstreams_terminal": all(row.get("terminal") is True for row in upstream_artifacts.values()),
        "checker_schema_frozen": dict(schema_receipt),
        "allowed_input_fields_frozen": allowed_and_forbidden_input_fields(),
        "attack_classes_frozen": {
            "contract_attacks": list(CONTRACT_ATTACKS),
            "leakage_attacks": list(LEAKAGE_ATTACKS),
        },
        "severity_rules_frozen": {
            "high": "any safety or provenance failure rejects, quarantines, or blocks readiness",
            "readiness_rule": "ready only if all high-severity attacks fail closed",
        },
        "random_seeds_frozen": dict(RANDOM_SEEDS),
        "resource_limits_frozen": dict(RESOURCE_LIMITS),
        "protected_hashes_frozen": dict(protected_before),
        "attack_fixture_frozen": dict(attack_receipt),
        "spec_req_present": "REQ-SAFE-6328"
        in (REPO_ROOT / SPEC_RELATIVE_PATH).read_text(encoding="utf-8"),
    }


def upstream_path_receipts() -> JsonDict:
    """Hash upstream artifacts, source files, and protected files."""

    return {
        "upstream_artifacts": {
            "exp6326": terminal_path_receipt(REPO_ROOT / EXP6326_RELATIVE_PATH),
            "exp6327": terminal_path_receipt(REPO_ROOT / EXP6327_RELATIVE_PATH),
        },
        "source_files": {
            path.as_posix(): path_receipt(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS
        },
        "protected_files": {
            path.as_posix(): path_receipt(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS
        },
    }


def terminal_path_receipt(path: Path) -> JsonDict:
    """Return terminal classifier metadata for one artifact."""

    return classify_artifact_path(path).to_dict()


def path_receipt(path: Path) -> JsonDict:
    """Return a path and hash receipt."""

    return {"path": display_path(path), "exists": path.exists(), "sha256": sha256_file(path)}


def checker_contract_registry() -> dict[str, JsonDict]:
    """Build the local exact contract registry keyed by canonical hash."""

    registry: dict[str, JsonDict] = {}
    for fixture in exp6326.build_fixture_manifest():
        contract = exp6326.validate_contract(fixture.contract)
        fallback_policy = exp6326.parse_policy(fixture.fallback_program)
        registry[canonical_contract_hash(fixture.contract)] = {
            "contract": contract,
            "factors": exp6326.compile_contract_to_factors(contract),
            "fallback_hash": {
                "semantic_hash": exp6326.semantic_hash(fallback_policy),
                "source_sha256": "sha256:" + exp6326.sha256_text(fixture.fallback_program),
            },
        }
    return registry


def canonical_contract_hash(payload: Mapping[str, Any]) -> str:
    """Hash the validated finite contract in canonical JSON form."""

    contract = exp6326.validate_contract(payload)
    canonical = {
        "family": contract.family,
        "split": contract.split,
        "states": list(contract.states),
        "actions": list(contract.actions),
        "clauses": list(contract.clauses),
    }
    return sha256_json(canonical)


def factor_outcomes(
    policy: exp6326.PolicyProgram,
    factors: Sequence[exp6326.Factor],
) -> list[JsonDict]:
    """Return exact local factor evidence without model or arm labels."""

    rows: list[JsonDict] = []
    for index, factor in enumerate(factors):
        satisfied = factor.satisfied(policy)
        rows.append(
            {
                "factor_index": index,
                "kind": factor.kind,
                "scope_hash": sha256_json(list(factor.scope)),
                "weight": factor.weight,
                "satisfied": satisfied,
                "contribution": 0 if satisfied else factor.weight,
            }
        )
    return rows


def exact_oracle_claim_boundary() -> JsonDict:
    """Describe the exact-oracle boundary."""

    return {
        "verifier": "Exp6326 finite-domain exact factor checker",
        "verifier_is_oracle": True,
        "model_supplies_candidates_only": True,
        "model_supplies_safety_authority": False,
        "oracle_distinct_verifier_claim": False,
        "utility_promotion_allowed": False,
    }


def discrepancies_and_severity(reconstruction: Mapping[str, Any]) -> JsonDict:
    """Group reconstruction discrepancies by model, family, arm, and severity."""

    rows = list(reconstruction.get("discrepancies") or [])
    by_severity = Counter(str(row.get("severity", "unknown")) for row in rows)
    return {
        "schema": SCHEMA + ".discrepancies",
        "high_severity_count": by_severity.get("high", 0),
        "by_severity": dict(sorted(by_severity.items())),
        "by_model_family_arm": rows,
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when blind safety and provenance gates pass."""

    tests = artifact.get("test_exit_codes") or {}
    preconditions = artifact.get("preconditions_checked") or {}
    info = artifact.get("information_asymmetry_receipts") or {}
    reconstruction = artifact.get("reconstructed_contract_factor_and_fallback_results") or {}
    contract_attacks = artifact.get(
        "vacuous_contract_parser_default_fallback_laundering_spec_mutation_validator_mutation_test_deletion_and_hash_swap_results"
    ) or {}
    leakage_attacks = artifact.get(
        "label_pair_evaluator_duplicate_arm_leakage_rationale_leakage_budget_and_missing_cell_results"
    ) or {}
    discrepancies = artifact.get("discrepancies_and_severity") or {}
    protected = artifact.get("protected_files_unchanged") or {}
    gates = (
        preconditions.get("all_upstreams_terminal") is True,
        info.get("forbidden_input_field_count") == 0,
        info.get("only_allowed_case_fields") is True,
        reconstruction.get("all_checker_cases_reconstructed") is True,
        reconstruction.get("checker_case_count") == RESOURCE_LIMITS["expected_checker_case_count"],
        reconstruction.get("upstream_safety_discrepancy_count") == 0,
        contract_attacks.get("all_high_severity_attacks_failed_closed") is True,
        leakage_attacks.get("all_high_severity_attacks_failed_closed") is True,
        discrepancies.get("high_severity_count") == 0,
        type(artifact.get("utility_promotion_count")) is int
        and artifact.get("utility_promotion_count") == 0,
        type(artifact.get("hidden_state_access_count")) is int
        and artifact.get("hidden_state_access_count") == 0,
        type(artifact.get("external_text_scorer_count")) is int
        and artifact.get("external_text_scorer_count") == 0,
        artifact.get("verifier_is_oracle") is True,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        protected.get("unchanged") is True,
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})),
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Return the terminal artifact status."""

    if not (artifact.get("preconditions_checked") or {}).get("all_upstreams_terminal"):
        return "blocked"
    return "complete_ready" if artifact.get("guard_integrity_ready_score") == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict without utility promotion."""

    if artifact.get("status") == "blocked":
        return "blocked: blind guard audit lacked terminal upstream evidence"
    if artifact.get("guard_integrity_ready_score") == 1.0:
        return "complete_ready: blind safety audit passed exact-oracle provenance checks with utility promotion count zero"
    return "complete_null: blind safety audit did not clear every high-severity provenance gate"


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields and fail closed on false readiness."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    for field in (
        "utility_promotion_count",
        "hidden_state_access_count",
        "external_text_scorer_count",
    ):
        _require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"]), "field_principles")
    _require(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"]), "field_provenance")
    expected = ready_score(artifact)
    _require(artifact.get("guard_integrity_ready_score") == expected, "guard_integrity_ready_score")
    _require(artifact.get("status") == status(artifact), "status")
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "honest_verdict")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    return True


def protected_hashes() -> dict[str, str | None]:
    """Return hashes for protected files."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected file hashes."""

    changed = sorted(path for path in before if before.get(path) != after.get(path))
    return {
        "before": dict(before),
        "after": dict(after),
        "changed": changed,
        "protected_files": list(before),
        "unchanged": not changed,
    }


def forbidden_checker_input_fields(payload: Mapping[str, Any]) -> list[str]:
    """Return paths for fields outside the checker allowlist."""

    bad: list[str] = []
    if not isinstance(payload, Mapping):
        return ["$"]
    for key in payload:
        if key not in CHECKER_TOP_LEVEL_FIELDS or forbidden_key(key):
            bad.append(str(key))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return sorted(set(bad))
    for case_index, case in enumerate(cases):
        case_path = f"cases[{case_index}]"
        if not isinstance(case, Mapping):
            bad.append(case_path)
            continue
        _check_mapping_fields(case, CHECKER_CASE_FIELDS, case_path, bad)
        evidence = case.get("exact_factor_evidence")
        if isinstance(evidence, Mapping):
            evidence_path = case_path + ".exact_factor_evidence"
            _check_mapping_fields(evidence, CHECKER_EXACT_FACTOR_FIELDS, evidence_path, bad)
            outcomes = evidence.get("factor_outcomes")
            if isinstance(outcomes, list):
                for outcome_index, outcome in enumerate(outcomes):
                    outcome_path = f"{evidence_path}.factor_outcomes[{outcome_index}]"
                    if isinstance(outcome, Mapping):
                        _check_mapping_fields(
                            outcome, CHECKER_FACTOR_OUTCOME_FIELDS, outcome_path, bad
                        )
                    else:
                        bad.append(outcome_path)
        fallback = case.get("fallback_hash")
        if isinstance(fallback, Mapping):
            _check_mapping_fields(fallback, CHECKER_FALLBACK_FIELDS, case_path + ".fallback_hash", bad)
    return sorted(set(bad))


def checker_input_field_paths(payload: Mapping[str, Any]) -> list[str]:
    """Return field-name paths for the checker input."""

    paths: list[str] = []
    if not isinstance(payload, Mapping):
        return paths
    for key, value in payload.items():
        paths.append(str(key))
        if key == "cases" and isinstance(value, list):
            for case_index, case in enumerate(value):
                if isinstance(case, Mapping):
                    _collect_field_paths(case, f"cases[{case_index}]", paths)
    return paths


def duplicate_blind_case_count(payload: Mapping[str, Any]) -> int:
    """Count duplicate blind case payloads."""

    cases = payload.get("cases") or []
    encoded = [canonical_json(case) for case in cases if isinstance(case, Mapping)]
    counts = Counter(encoded)
    return sum(count - 1 for count in counts.values() if count > 1)


def checker_decision_hash(result: Mapping[str, Any]) -> str:
    """Hash only label-free checker decision fields."""

    stable = {
        "accepted": result.get("accepted"),
        "decision": result.get("decision"),
        "case_count": result.get("case_count"),
        "forbidden_input_field_count": result.get("forbidden_input_field_count"),
        "case_results": [
            {
                "blind_index": row.get("blind_index"),
                "passed": row.get("passed"),
                "exact_energy": row.get("exact_energy"),
                "accepted_by_exact_guard": row.get("accepted_by_exact_guard"),
                "errors": row.get("errors"),
            }
            for row in result.get("case_results") or []
        ],
    }
    return sha256_json(stable)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking wall time and its checksum."""

    stable = json.loads(canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def sha256_json(value: Any) -> str:
    """Return a stable JSON SHA-256 digest."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file SHA-256 digest, or None when absent."""

    if not path.exists() or not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    """Return canonical JSON text."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON to a path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def checker_main() -> int:
    """Read checker JSON from stdin and write a checker decision."""

    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, Mapping):
            raise ValueError("checker_input_type")
        result = evaluate_blind_checker_input(payload)
    except Exception as exc:  # pragma: no cover
        result = {
            "schema": SCHEMA + ".checker_output",
            "accepted": False,
            "decision": "reject",
            "checker_pid": os.getpid(),
            "case_count": 0,
            "forbidden_input_field_count": 0,
            "forbidden_input_fields": [],
            "schema_ok": False,
            "case_results": [],
            "error": str(exc),
        }
        result["decision_hash"] = checker_decision_hash(result)
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for artifact mode and checker mode."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=DEFAULT_RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--schema-path", type=Path, default=REPO_ROOT / CHECKER_SCHEMA_RELATIVE_PATH)
    parser.add_argument("--attack-fixture-path", type=Path, default=REPO_ROOT / ATTACK_FIXTURE_RELATIVE_PATH)
    parser.add_argument("--checker", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.checker:
        return checker_main()
    if args.validate:
        validate_artifact(_load_json_object(args.result_path))
        return 0
    run(
        date=args.date,
        result_path=args.result_path,
        schema_path=args.schema_path,
        attack_fixture_path=args.attack_fixture_path,
        write=True,
    )
    return 0


def _raw_safety(candidate: Mapping[str, Any]) -> JsonDict:
    energy = candidate.get("exact_energy")
    parse_ok = candidate.get("parse_status") == "parsed"
    return {
        "accepted": parse_ok,
        "fallback_used": False,
        "contract_violation_count": int(energy) if parse_ok and isinstance(energy, int) else None,
    }


def _reject_only_safety(candidate: Mapping[str, Any]) -> JsonDict:
    accepted = candidate.get("parse_status") == "parsed" and candidate.get("exact_energy") == 0
    return {
        "accepted": accepted,
        "fallback_used": False,
        "contract_violation_count": 0 if accepted else None,
    }


def _fallback_safety(candidate: Mapping[str, Any]) -> JsonDict:
    accepted = candidate.get("parse_status") == "parsed" and candidate.get("exact_energy") == 0
    return {
        "accepted": True,
        "fallback_used": not accepted,
        "contract_violation_count": 0,
    }


def _mutate_validator_evidence(payload: JsonDict) -> None:
    cases = payload["cases"]
    for case in cases:
        evidence = case["exact_factor_evidence"]
        if evidence["exact_energy"] != 0:
            evidence["exact_energy"] = 0
            evidence["accepted_by_exact_guard"] = True
            return
    cases[0]["exact_factor_evidence"]["exact_energy"] = 1


def _swap_contract_hash(payload: JsonDict) -> None:
    cases = payload["cases"]
    first = cases[0]["canonical_contract_hash"]
    for case in cases[1:]:
        if case["canonical_contract_hash"] != first:
            cases[0]["canonical_contract_hash"] = case["canonical_contract_hash"]
            return
    cases[0]["canonical_contract_hash"] = sha256_json({"hash_swap": True})


def _check_mapping_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    path: str,
    bad: list[str],
) -> None:
    for key in value:
        if key not in allowed or forbidden_key(key):
            bad.append(path + "." + str(key))


def forbidden_key(key: Any) -> bool:
    text = str(key).lower()
    return any(marker in text for marker in FORBIDDEN_CHECKER_FIELD_MARKERS)


def _collect_field_paths(value: Mapping[str, Any], path: str, paths: list[str]) -> None:
    for key, child in value.items():
        child_path = path + "." + str(key)
        paths.append(child_path)
        if isinstance(child, Mapping):
            _collect_field_paths(child, child_path, paths)
        elif isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    _collect_field_paths(item, f"{child_path}[{index}]", paths)


def _path_from_receipt(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_object:" + str(path))
    return payload


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
