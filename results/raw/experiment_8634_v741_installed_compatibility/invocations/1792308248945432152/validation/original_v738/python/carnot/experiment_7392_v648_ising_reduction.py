"""Separate exact Ising laws from quality measured on preserved chains.

This module never starts a sampler. It authenticates the earlier law fixture,
audit, validation logs, and compressed traces. It then recomputes exact laws
and finite-chain statistics from those preserved bytes.

Spec refs: REQ-SAMPLER-7392 and SCENARIO-SAMPLER-7392-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7377_v647_ising_law as exact_law
from carnot import experiment_7378_v647_ising_audit as historical_audit
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260918"
MILESTONE = "2026.09.648"
PHASE = 3
EXPERIMENT_ID = "exp7392-v648-ising-reduction"
SCHEMA = "carnot.exp7392.v648.ising_reduction.v1"
RESULT_PATH = Path("results/experiment_7392_v648_ising_reduction.json")
RAW_DIR = Path("results/raw/experiment_7392_v648_ising_reduction")
LAW_PATH = Path("results/experiment_7377_v647_ising_law.json")
AUDIT_PATH = Path("results/experiment_7378_v647_ising_audit.json")
TRACE_PATH = Path("results/raw/experiment_7378_v647_ising_audit/ordered_chain_traces.jsonl.gz")
MODULE_PATH = Path("python/carnot/experiment_7392_v648_ising_reduction.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7392_v648_ising_reduction.py")
TEST_PATH = Path("tests/python/test_experiment_7392_v648_ising_reduction.py")
SPEC_PATH = Path("openspec/capabilities/samplers/spec.md")

EXPECTED_LAW_SHA256 = "sha256:c0ccdce53df962a8bed7a90e159ba5a7e8d4c353483f8873d3b4c1570b013c69"
EXPECTED_AUDIT_SHA256 = "sha256:a9f6ae5b2f2d189cdb8d81542cb5ca6ef5ebc94f69208d5a933527834ac8afd8"
EXPECTED_TRACE_SHA256 = "sha256:48e5e5bc60137a7bbd696301b18139a9a3ddf1f96858423fbede36d8c3bc377b"

BETA_GRID = (0.5, 1.0, 2.0)
EXACT_CONDITIONS = (
    "source_only",
    "appended_implied_clause",
    "proof_assisted_source_only",
)
CHAIN_CONDITIONS = (
    "source_only",
    "proof_assisted_source_only",
    historical_audit.NEGATIVE_CONDITION,
)
SUPPORT_CLASSES = (
    "contradictory_clamp_empty_support",
    "satisfiable_clamp_no_zero_energy_state",
    "ordinary_nonempty_support",
)
ENERGY_RESIDUAL_LIMIT = 1e-12
EXACT_TV_LIMIT = 1e-10
OBSERVABLE_ERROR_LIMIT = 0.05
ESS_MINIMUM = 1_000.0

REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_CURRENT_INVOCATIONS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
    "raw_receipts": [],
}
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "repository_health",
    "field_principles",
    "promotion_score",
    "ising_reduction_complete_score",
    "law_preservation_confirmed_score",
    "original_gate_results",
    "support_rows",
    "sample_quality_rows",
    "historical_trace_manifest",
)
REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7377_v647_ising_law.py"),
    Path("python/carnot/experiment_7378_v647_ising_audit.py"),
    Path("openspec/capabilities/ising-backend/spec.md"),
    SPEC_PATH,
    LAW_PATH,
    AUDIT_PATH,
    TRACE_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
V648_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON data so two cold reductions can be compared exactly."""

    return validation_contract.canonical_hash(value)


def sha256_file(path: Path) -> str:
    """Hash the exact bytes of an input without loading a large file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def precondition_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
    terminal_blocking: bool = True,
) -> JsonDict:
    """Record one exact prerequisite and its observed value before reduction."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else passed,
        "terminal_blocking": terminal_blocking,
    }


def _read_object(path: Path) -> JsonDict:
    """Read one required JSON object and reject arrays or scalar documents."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _historical_log_rows(root: Path, audit: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate every validation log cited by the historical producer."""

    rows: list[JsonDict] = []
    for receipt in audit.get("validation_receipts") or []:
        relative = Path(str(receipt.get("log_path") or ""))
        path = relative if relative.is_absolute() else root / relative
        present = path.is_file() and path.stat().st_size > 0
        observed = sha256_file(path) if present else "missing"
        rows.append(
            {
                "name": receipt.get("name"),
                "path": relative.as_posix(),
                "sha256": observed,
                "expected_sha256": receipt.get("log_sha256"),
                "byte_count": path.stat().st_size if present else 0,
                "authenticated": present and observed == receipt.get("log_sha256"),
                "historical_passed": receipt.get("passed"),
                "historical_exit_code": receipt.get("exit_code"),
            }
        )
    return rows


def collect_preconditions(repo_root: Path) -> tuple[list[JsonDict], dict[str, str]]:
    """Check exact files, producer identities, historical status, and resources."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            precondition_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else "missing",
            )
        )
        if present:
            hashes[relative.as_posix()] = sha256_file(path)

    if not all((root / path).is_file() for path in (LAW_PATH, AUDIT_PATH, TRACE_PATH)):
        return checks, hashes

    try:
        law = _read_object(root / LAW_PATH)
        audit = _read_object(root / AUDIT_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        checks.append(
            precondition_row(
                "preserved_json_parse",
                f"{LAW_PATH.as_posix()}|{AUDIT_PATH.as_posix()}",
                "json",
                "valid_json_objects",
                str(error),
            )
        )
        return checks, hashes

    exact_checks = (
        ("law_artifact_sha256", LAW_PATH, EXPECTED_LAW_SHA256),
        ("audit_artifact_sha256", AUDIT_PATH, EXPECTED_AUDIT_SHA256),
        ("trace_archive_sha256", TRACE_PATH, EXPECTED_TRACE_SHA256),
    )
    for check, path, expected in exact_checks:
        checks.append(
            precondition_row(
                check,
                path.as_posix(),
                "sha256",
                expected,
                hashes.get(path.as_posix()),
            )
        )

    law_expected = {
        "experiment_id": "exp7377-v647-ising-law",
        "verdict_class": "circular_positive",
        "law_fixture_ready_score": 1,
        "flagged_adversarial": False,
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
    }
    for field, expected in law_expected.items():
        checks.append(
            precondition_row(
                f"law_producer_{field}", LAW_PATH.as_posix(), field, expected, law.get(field)
            )
        )
    law_errors = exact_law.validate_artifact(law)
    checks.append(
        precondition_row(
            "law_producer_validation",
            LAW_PATH.as_posix(),
            "producer_validator_errors",
            [],
            law_errors,
        )
    )

    audit_expected = {
        "experiment_id": "exp7378-v647-ising-audit",
        "verdict_class": "disqualified",
        "flagged_adversarial": True,
        "execution_venue": "host_cpu",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
    }
    for field, expected in audit_expected.items():
        checks.append(
            precondition_row(
                f"historical_audit_{field}",
                AUDIT_PATH.as_posix(),
                field,
                expected,
                audit.get(field),
            )
        )
    historical_errors = historical_audit.validate_artifact(audit)
    checks.append(
        precondition_row(
            "historical_closed_venue_rejection",
            "carnot.experiment_7378_v647_ising_audit.validate_artifact",
            "execution_venue",
            "execution_venue_invalid",
            "execution_venue_invalid" if "execution_venue_invalid" in historical_errors else None,
        )
    )
    upstream_hash = (audit.get("upstream_producer") or {}).get("byte_sha256")
    checks.append(
        precondition_row(
            "historical_audit_law_hash",
            AUDIT_PATH.as_posix(),
            "upstream_producer.byte_sha256",
            EXPECTED_LAW_SHA256,
            upstream_hash,
        )
    )
    original_gate = next(
        (
            row
            for row in audit.get("acceptance_gate_results") or []
            if row.get("check") == "all_source_cells_qualified"
        ),
        {},
    )
    checks.append(
        precondition_row(
            "historical_all_cell_gate_failed",
            AUDIT_PATH.as_posix(),
            "acceptance_gate_results.all_source_cells_qualified.observed",
            False,
            original_gate.get("observed"),
        )
    )
    manifest = audit.get("raw_trace_archive") or {}
    checks.extend(
        [
            precondition_row(
                "trace_manifest_hash",
                AUDIT_PATH.as_posix(),
                "raw_trace_archive.sha256",
                EXPECTED_TRACE_SHA256,
                manifest.get("sha256"),
            ),
            precondition_row(
                "trace_manifest_records",
                AUDIT_PATH.as_posix(),
                "raw_trace_archive.record_count",
                540,
                manifest.get("record_count"),
            ),
        ]
    )

    log_rows = _historical_log_rows(root, audit)
    checks.append(
        precondition_row(
            "historical_validation_log_roster",
            AUDIT_PATH.as_posix(),
            "validation_receipts.log_path",
            14,
            len(log_rows),
            passed=len(log_rows) == 14 and all(row["authenticated"] for row in log_rows),
        )
    )
    for row in log_rows:
        hashes[str(row["path"])] = str(row["sha256"])

    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        precondition_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-SAMPLER-7392",
            "REQ-SAMPLER-7392" if "REQ-SAMPLER-7392" in spec_text else None,
        )
    )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7392" in exclusion_text or EXPERIMENT_ID in exclusion_text
    checks.append(
        precondition_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
        )
    )
    cpu_count = os.cpu_count()
    checks.append(
        precondition_row(
            "host_cpu_available",
            "os.cpu_count",
            "logical_cpu_count",
            "positive_integer",
            cpu_count,
            passed=isinstance(cpu_count, int) and cpu_count > 0,
        )
    )
    return checks, hashes


def load_preserved_inputs(repo_root: Path) -> JsonDict:
    """Load only preserved evidence after every blocking precondition passes."""

    root = repo_root.resolve()
    preconditions, hashes = collect_preconditions(root)
    failures = [row for row in preconditions if row["terminal_blocking"] and not row["passed"]]
    if failures:
        raise ValueError(f"preserved_input_precondition_failed:{failures[0]['check']}")
    law = _read_object(root / LAW_PATH)
    audit = _read_object(root / AUDIT_PATH)
    records = historical_audit.load_trace_archive(root / TRACE_PATH, audit["raw_trace_archive"])
    logs = _historical_log_rows(root, audit)
    return {
        "preconditions": preconditions,
        "source_hashes": hashes,
        "law_artifact": law,
        "audit_artifact": audit,
        "trace_records": records,
        "validation_logs": logs,
        "historical_validator_errors": historical_audit.validate_artifact(audit),
        "historical_sidecars": [
            {
                "label": "exp7377_finite_source_law",
                "path": LAW_PATH.as_posix(),
                "sha256": hashes[LAW_PATH.as_posix()],
                "producer_class": law.get("inference_substrate_class"),
                "flagged_adversarial": law.get("flagged_adversarial"),
                "verdict_class": law.get("verdict_class"),
                "counted_as_current": False,
                "model_receipts": deepcopy(law.get("historical_inference_sidecars") or []),
            },
            {
                "label": "exp7378_disqualified_sampler_diagnostic",
                "path": AUDIT_PATH.as_posix(),
                "sha256": hashes[AUDIT_PATH.as_posix()],
                "producer_class": audit.get("inference_substrate_class"),
                "execution_venue": audit.get("execution_venue"),
                "flagged_adversarial": audit.get("flagged_adversarial"),
                "verdict_class": audit.get("verdict_class"),
                "counted_as_current": False,
                "eligible_for_diagnosis": True,
                "eligible_for_current_readiness": False,
                "original_model_specs": deepcopy(audit.get("MODEL_SPECS")),
                "original_invocation_counts": deepcopy(audit.get("invocation_counts")),
            },
        ],
    }


def _support_class(formula: Mapping[str, Any], source_energies: Sequence[float]) -> JsonDict:
    """Classify clamp semantics without using energy minima as support tests."""

    fixed: dict[int, int] = {}
    contradiction = False
    for raw_literal in formula.get("assumptions") or []:
        literal = int(raw_literal)
        index = abs(literal) - 1
        value = int(literal > 0)
        contradiction = contradiction or (index in fixed and fixed[index] != value)
        fixed[index] = value
    states = exact_law.enumerate_bits(int(formula["n_vars"]))
    support = [
        not contradiction and all(bits[index] == value for index, value in fixed.items())
        for bits in states
    ]
    supported = [energy for energy, keep in zip(source_energies, support) if keep]
    if contradiction:
        support_class = SUPPORT_CLASSES[0]
        reason = "same variable is clamped to both Boolean values; no conditional law exists"
        minimum: float | None = None
    elif supported and min(supported) > 0.0:
        support_class = SUPPORT_CLASSES[1]
        reason = "the clamp is consistent and support is nonempty despite positive minimum energy"
        minimum = min(supported)
    else:
        support_class = SUPPORT_CLASSES[2]
        reason = "the clamp is consistent and support includes a zero-energy source state"
        minimum = min(supported) if supported else None
    return {
        "support": support,
        "support_size": sum(support),
        "support_class": support_class,
        "support_reason": reason,
        "minimum_source_energy": minimum,
        "contradictory_clamp": contradiction,
    }


def _total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute finite total variation in the shared enumerated state order."""

    return 0.5 * math.fsum(abs(float(a) - float(b)) for a, b in zip(left, right))


def enumerate_support_rows(formulas: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Enumerate all three exact arms without confusing energy with support."""

    rows: list[JsonDict] = []
    for formula in formulas:
        n_vars = int(formula["n_vars"])
        states = exact_law.enumerate_bits(n_vars)
        source_clauses = [list(map(int, clause)) for clause in formula["original_clauses"]]
        appended_clauses = [*source_clauses, list(map(int, formula["implied_clause"]))]
        source_energies = [
            historical_audit.independent_source_energy(bits, source_clauses) for bits in states
        ]
        support_info = _support_class(formula, source_energies)
        support = support_info["support"]
        source_probabilities_by_beta = {
            beta: exact_law.normalized_probabilities(source_energies, support, beta)[0]
            for beta in BETA_GRID
        }
        for beta in BETA_GRID:
            for condition in EXACT_CONDITIONS:
                clauses = (
                    appended_clauses if condition == "appended_implied_clause" else source_clauses
                )
                direct = [
                    historical_audit.independent_source_energy(bits, clauses) for bits in states
                ]
                compiled = exact_law.compile_2cnf(n_vars, clauses)
                compiled_energies = [
                    compiled.energy(exact_law.bits_to_spins(bits)) for bits in states
                ]
                direct_probabilities, normalizer = exact_law.normalized_probabilities(
                    direct, support, beta
                )
                compiled_probabilities, _ = exact_law.normalized_probabilities(
                    compiled_energies, support, beta
                )
                supported_arm = [energy for energy, keep in zip(direct, support) if keep]
                rows.append(
                    {
                        "row_type": "support_law",
                        "formula_id": formula["formula_id"],
                        "source_hash": formula["source_hash"],
                        "seed": formula["seed"],
                        "feature": formula["feature"],
                        "n_vars": n_vars,
                        "assumptions": deepcopy(formula.get("assumptions") or []),
                        "beta": beta,
                        "condition": condition,
                        "support_class": support_info["support_class"],
                        "support_reason": support_info["support_reason"],
                        "support_size": support_info["support_size"],
                        "law_defined": normalizer > 0.0,
                        "normalizer": normalizer,
                        "minimum_source_energy": support_info["minimum_source_energy"],
                        "minimum_evaluated_energy": min(supported_arm) if supported_arm else None,
                        "zero_energy_state_count": sum(
                            keep and energy == 0.0 for keep, energy in zip(support, direct)
                        ),
                        "max_abs_energy_residual": max(
                            abs(left - right) for left, right in zip(compiled_energies, direct)
                        ),
                        "exact_total_variation": _total_variation(
                            compiled_probabilities, direct_probabilities
                        ),
                        "source_law_total_variation": _total_variation(
                            direct_probabilities, source_probabilities_by_beta[beta]
                        ),
                        "energy_residual_limit": ENERGY_RESIDUAL_LIMIT,
                        "exact_tv_limit": EXACT_TV_LIMIT,
                        "outcome": "undefined_conditional_law"
                        if normalizer == 0.0
                        else "defined_finite_law",
                        "costs": {
                            "states_enumerated": len(states),
                            "current_llm_calls": 0,
                            "new_sampler_draws": 0,
                        },
                        "failures": ["contradictory_clamp_empty_support"]
                        if normalizer == 0.0
                        else [],
                        "censored": False,
                    }
                )
    return rows


def _chain_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row.get("cell_id")), int(row.get("chain_order", -1))


def _cell_rosters(
    law: Mapping[str, Any], audit: Mapping[str, Any]
) -> tuple[set[str], set[tuple[str, int]]]:
    formula_ids = [str(row["formula_id"]) for row in law.get("frozen_formulas") or []]
    negative_ids = [str(value) for value in audit.get("negative_control_formula_ids") or []]
    cells = {
        f"{formula_id}|{beta}|{condition}"
        for formula_id in formula_ids
        for beta in BETA_GRID
        for condition in historical_audit.FAITHFUL_CONDITIONS
    }
    cells.update(
        f"{formula_id}|{beta}|{historical_audit.NEGATIVE_CONDITION}"
        for formula_id in negative_ids
        for beta in BETA_GRID
    )
    chains = {(cell_id, order) for cell_id in cells for order in range(4)}
    return cells, chains


def validate_preserved_structure(
    law: Mapping[str, Any],
    audit: Mapping[str, Any],
    trace_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Reject roster, source, protocol, cost, clamp, and trace mutations."""

    errors: list[str] = []
    formulas = list(law.get("frozen_formulas") or [])
    formula_by_id = {str(row.get("formula_id")): row for row in formulas}
    if len(formulas) != 24 or len(formula_by_id) != 24:
        errors.append("formula_roster_incomplete")
    fixture_hash = exact_law.fixture_hash(formulas) if formulas else None
    if fixture_hash != (law.get("frozen_sampling_protocol") or {}).get("fixture_sha256"):
        errors.append("fixture_hash_mismatch")
    expected_cells, expected_chains = _cell_rosters(law, audit)
    cells = list(audit.get("cell_results") or [])
    observed_cells = {str(row.get("cell_id")) for row in cells}
    if observed_cells != expected_cells or len(cells) != len(expected_cells):
        errors.append("cell_roster_incomplete")
    chains = list(audit.get("per_chain_results") or [])
    chain_by_key = {_chain_key(row): row for row in chains}
    if set(chain_by_key) != expected_chains or len(chains) != len(expected_chains):
        errors.append("chain_roster_incomplete")

    for row in chains:
        formula = formula_by_id.get(str(row.get("formula_id")))
        if formula is None:
            errors.append("stale_source_version")
            continue
        if row.get("source_hash") != formula.get("source_hash"):
            errors.append("stale_source_version")
        beta = row.get("beta")
        if beta not in BETA_GRID:
            errors.append("altered_beta")
        condition = str(row.get("condition"))
        if condition not in CHAIN_CONDITIONS:
            errors.append("unknown_chain_condition")
        expected_cell_id = f"{row.get('formula_id')}|{beta}|{condition}"
        if row.get("cell_id") != expected_cell_id:
            errors.append("altered_beta")
        if not isinstance(row.get("costs"), Mapping):
            errors.append("missing_chain_cost")

    for row in cells:
        formula = formula_by_id.get(str(row.get("formula_id")))
        if formula is None:
            errors.append("stale_source_version")
            continue
        states = exact_law.enumerate_bits(int(formula["n_vars"]))
        source = [
            historical_audit.independent_source_energy(bits, formula["original_clauses"])
            for bits in states
        ]
        support = _support_class(formula, source)
        expected_status = (
            "empty_conditioned_support"
            if support["support_size"] == 0
            else "nonempty_conditioned_support"
        )
        if row.get("support_status") != expected_status:
            errors.append("inconsistent_clamp")

    observed_traces: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in trace_records:
        key = _chain_key(record)
        if key in observed_traces:
            errors.append("duplicate_trace")
        observed_traces[key] = record
        indices = [int(value) for value in record.get("state_indices") or []]
        if canonical_hash(indices) != record.get("trace_sha256"):
            errors.append("forged_sample")
        formula = formula_by_id.get(str(record.get("formula_id")))
        state_limit = 1 << int(formula["n_vars"]) if formula else 0
        if formula is None or any(value < 0 or value >= state_limit for value in indices):
            errors.append("forged_sample")
        public = chain_by_key.get(key)
        counts = {str(index): count for index, count in sorted(Counter(indices).items())}
        if public is None:
            errors.append("trace_without_chain")
        elif (
            len(indices) != int(public.get("recorded_samples", -1))
            or counts != public.get("state_counts")
            or record.get("trace_sha256") != public.get("trace_sha256")
        ):
            errors.append("forged_sample")
        if formula is not None and record.get("source_hash") != formula.get("source_hash"):
            errors.append("stale_source_version")
        if record.get("beta") not in BETA_GRID:
            errors.append("altered_beta")
    expected_trace_keys = {
        key for key, row in chain_by_key.items() if int(row.get("recorded_samples", 0)) > 0
    }
    if set(observed_traces) != expected_trace_keys:
        errors.append("trace_roster_incomplete")

    all_cell_gate = next(
        (
            row
            for row in audit.get("acceptance_gate_results") or []
            if row.get("check") == "all_source_cells_qualified"
        ),
        None,
    )
    if (
        audit.get("verdict_class") != "disqualified"
        or audit.get("flagged_adversarial") is not True
        or not isinstance(all_cell_gate, Mapping)
        or all_cell_gate.get("observed") is not False
        or all_cell_gate.get("passed") is not False
    ):
        errors.append("original_gate_results_changed")
    return list(dict.fromkeys(errors))


def _decode_states(indices: Sequence[int], n_vars: int) -> np.ndarray:
    """Decode stable integer state IDs into the original Boolean variable order."""

    values = np.asarray(indices, dtype=np.int64)
    shifts = np.arange(n_vars - 1, -1, -1, dtype=np.int64)
    return ((values[:, None] >> shifts) & 1).astype(np.int8)


def _observable_row(row: Mapping[str, Any]) -> JsonDict:
    """Keep each measured mean and autocorrelation result without sample series."""

    autocorrelation = deepcopy(dict(row["autocorrelation"]))
    return {
        "observable": row["observable"],
        "exact_mean": row["exact_mean"],
        "observed_mean": row["observed_mean"],
        "absolute_error": row["absolute_error"],
        "autocorrelation": autocorrelation,
    }


def _close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    """Compare nullable numeric values without hiding a changed missing value."""

    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def reduce_preserved_evidence(
    law: Mapping[str, Any],
    audit: Mapping[str, Any],
    trace_records: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Recompute exact-law and empirical conclusions from the archived samples."""

    errors = validate_preserved_structure(law, audit, trace_records)
    support_rows = enumerate_support_rows(law.get("frozen_formulas") or [])
    formula_by_id = {str(row["formula_id"]): row for row in law.get("frozen_formulas") or []}
    traces = {_chain_key(row): row for row in trace_records}
    private_chains: dict[tuple[str, int], JsonDict] = {}
    quality_rows: list[JsonDict] = []

    for public in audit.get("per_chain_results") or []:
        key = _chain_key(public)
        formula = formula_by_id[str(public["formula_id"])]
        beta = float(public["beta"])
        condition = str(public["condition"])
        source_clauses = [list(map(int, clause)) for clause in formula["original_clauses"]]
        clauses = (
            [*source_clauses, list(map(int, formula["implied_clause"]))]
            if condition == historical_audit.NEGATIVE_CONDITION
            else source_clauses
        )
        target = historical_audit.enumerated_target(formula, beta, clauses=clauses)
        if int(public.get("recorded_samples", 0)) == 0:
            private = deepcopy(dict(public))
            private["observable_series"] = {}
            private_chains[key] = private
            quality_rows.append(
                {
                    "row_type": "chain_quality",
                    "cell_id": public["cell_id"],
                    "formula_id": public["formula_id"],
                    "source_hash": public["source_hash"],
                    "beta": beta,
                    "condition": condition,
                    "chain_order": public["chain_order"],
                    "seed": public["seed"],
                    "support_status": "empty_conditioned_support",
                    "sample_size": 0,
                    "trace_sha256": None,
                    "state_count_total": 0,
                    "max_abs_energy_residual": None,
                    "max_observable_error": None,
                    "minimum_effective_sample_size": None,
                    "autocorrelation_defined": False,
                    "observable_rows": [],
                    "qualified": False,
                    "outcome": "undefined_conditional_law_no_draws",
                    "costs": deepcopy(public.get("costs")),
                    "failures": ["empty_conditioned_support_has_no_probability_law"],
                    "censored": False,
                }
            )
            continue

        record = traces[key]
        indices = [int(value) for value in record["state_indices"]]
        samples = _decode_states(indices, int(formula["n_vars"]))
        series, observable_rows = historical_audit._analyze_samples(samples, target, source_clauses)
        compiled = exact_law.compile_2cnf(int(formula["n_vars"]), source_clauses)
        unique_states = {_index for _index in indices}
        energy_residual = max(
            abs(
                compiled.energy(
                    exact_law.bits_to_spins(exact_law.enumerate_bits(int(formula["n_vars"]))[index])
                )
                - historical_audit.independent_source_energy(
                    exact_law.enumerate_bits(int(formula["n_vars"]))[index], source_clauses
                )
            )
            for index in unique_states
        )
        effective = [
            row["autocorrelation"]["effective_sample_size"]
            for row in observable_rows
            if row["autocorrelation"]["effective_sample_size"] is not None
        ]
        minimum_ess = min(map(float, effective)) if effective else float(len(indices))
        maximum_error = max(float(row["absolute_error"]) for row in observable_rows)
        private = {
            **deepcopy(dict(public)),
            "observable_series": series,
            "observable_results": observable_rows,
        }
        private_chains[key] = private
        old_observables = {
            str(row["observable"]): row for row in public.get("observable_results") or []
        }
        if any(
            name not in old_observables
            or not _close(row["observed_mean"], old_observables[name].get("observed_mean"))
            or not _close(row["absolute_error"], old_observables[name].get("absolute_error"))
            for name, row in ((str(item["observable"]), item) for item in observable_rows)
        ):
            errors.append(f"chain_metric_mismatch:{key}")
        quality_rows.append(
            {
                "row_type": "chain_quality",
                "cell_id": public["cell_id"],
                "formula_id": public["formula_id"],
                "source_hash": public["source_hash"],
                "beta": beta,
                "condition": condition,
                "chain_order": public["chain_order"],
                "seed": public["seed"],
                "support_status": "nonempty_conditioned_support",
                "sample_size": len(indices),
                "trace_sha256": canonical_hash(indices),
                "state_count_total": sum(Counter(indices).values()),
                "max_abs_energy_residual": energy_residual,
                "max_observable_error": maximum_error,
                "minimum_effective_sample_size": minimum_ess,
                "autocorrelation_defined": True,
                "observable_rows": [_observable_row(row) for row in observable_rows],
                "observable_error_limit": OBSERVABLE_ERROR_LIMIT,
                "effective_sample_minimum": ESS_MINIMUM,
                "qualified": maximum_error <= OBSERVABLE_ERROR_LIMIT
                and all(row["autocorrelation"]["qualified"] for row in observable_rows),
                "outcome": "complete_archived_trace_reanalysis",
                "costs": deepcopy(public.get("costs")),
                "failures": [],
                "censored": False,
            }
        )

    for public_cell in audit.get("cell_results") or []:
        formula = formula_by_id[str(public_cell["formula_id"])]
        beta = float(public_cell["beta"])
        condition = str(public_cell["condition"])
        source_clauses = [list(map(int, clause)) for clause in formula["original_clauses"]]
        clauses = (
            [*source_clauses, list(map(int, formula["implied_clause"]))]
            if condition == historical_audit.NEGATIVE_CONDITION
            else source_clauses
        )
        target = historical_audit.enumerated_target(formula, beta, clauses=clauses)
        chains = [private_chains[(str(public_cell["cell_id"]), order)] for order in range(4)]
        measured = historical_audit.aggregate_cell(
            str(formula["formula_id"]), beta, condition, chains, target
        )
        if (
            not _close(measured["max_observable_error"], public_cell.get("max_observable_error"))
            or not _close(
                measured["minimum_effective_samples"],
                public_cell.get("minimum_effective_samples"),
            )
            or measured["qualified"] is not public_cell.get("qualified")
        ):
            errors.append(f"cell_metric_mismatch:{public_cell['cell_id']}")
        empty = measured["support_status"] == "empty_conditioned_support"
        quality_rows.append(
            {
                "row_type": "cell_quality",
                "cell_kind": public_cell["cell_kind"],
                "cell_id": public_cell["cell_id"],
                "formula_id": public_cell["formula_id"],
                "beta": beta,
                "condition": condition,
                "support_status": measured["support_status"],
                "sample_size": int(measured["recorded_samples"]),
                "chain_count": int(measured["chain_count"]),
                "max_abs_energy_residual": measured["target_energy_parity_max_error"],
                "max_observable_error": measured["max_observable_error"],
                "minimum_effective_sample_size": measured["minimum_effective_samples"],
                "autocorrelation_defined": not empty,
                "observable_rows": deepcopy(measured["observable_results"]),
                "observable_error_limit": OBSERVABLE_ERROR_LIMIT,
                "effective_sample_minimum": ESS_MINIMUM,
                "qualified": bool(measured["qualified"]),
                "outcome": "undefined_conditional_law_no_draws"
                if empty
                else "complete_archived_cell_reanalysis",
                "costs": deepcopy(public_cell.get("costs")),
                "failures": deepcopy(measured.get("failures") or []),
                "censored": False,
            }
        )

    errors = list(dict.fromkeys(errors))
    defined = [row for row in support_rows if row["law_defined"]]
    undefined = [row for row in support_rows if not row["law_defined"]]
    source_cells = [
        row
        for row in quality_rows
        if row["row_type"] == "cell_quality" and row["cell_kind"] == "source_faithful"
    ]
    nonempty_source = [
        row for row in source_cells if row["support_status"] == "nonempty_conditioned_support"
    ]
    empty_source = [
        row for row in source_cells if row["support_status"] == "empty_conditioned_support"
    ]
    original_gate = next(
        row
        for row in audit["acceptance_gate_results"]
        if row["check"] == "all_source_cells_qualified"
    )
    result: JsonDict = {
        "passed": not errors,
        "errors": errors,
        "trace_record_count": len(trace_records),
        "chain_row_count": sum(row["row_type"] == "chain_quality" for row in quality_rows),
        "cell_row_count": sum(row["row_type"] == "cell_quality" for row in quality_rows),
        "support_rows": support_rows,
        "sample_quality_rows": quality_rows,
        "exact_law_conclusions": {
            "defined_row_count": len(defined),
            "undefined_row_count": len(undefined),
            "maximum_energy_residual": max(
                float(row["max_abs_energy_residual"]) for row in support_rows
            ),
            "maximum_exact_total_variation": max(
                float(row["exact_total_variation"]) for row in support_rows
            ),
            "law_preservation_confirmed": bool(defined)
            and all(
                row["max_abs_energy_residual"] <= ENERGY_RESIDUAL_LIMIT
                and row["exact_total_variation"] <= EXACT_TV_LIMIT
                for row in defined
            )
            and all(row["support_class"] == SUPPORT_CLASSES[0] for row in undefined),
            "undefined_conditionals_are_distributions": False,
            "positive_minimum_energy_implies_empty_law": False,
        },
        "finite_chain_conclusions": {
            "source_cell_count": len(source_cells),
            "nonempty_source_cell_count": len(nonempty_source),
            "empty_source_cell_count": len(empty_source),
            "nonempty_source_cells_qualified": sum(row["qualified"] for row in nonempty_source),
            "all_nonempty_source_cells_qualified": bool(nonempty_source)
            and all(row["qualified"] for row in nonempty_source),
            "original_all_cell_gate_passed": original_gate["passed"],
            "original_all_cell_gate_observed": original_gate["observed"],
            "prospective_pass_claimed": False,
            "conclusion": "nonempty archived cells meet fixed quality gates; undefined cells keep the original all-cell gate failed",
        },
    }
    result["reduction_checksum"] = canonical_hash(
        {
            "trace_record_count": result["trace_record_count"],
            "support_rows": support_rows,
            "sample_quality_rows": quality_rows,
            "exact_law_conclusions": result["exact_law_conclusions"],
            "finite_chain_conclusions": result["finite_chain_conclusions"],
            "errors": errors,
        }
    )
    return result


def run_reducer_attack_controls(
    law: Mapping[str, Any],
    audit: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Mutate each required boundary and require its named detector to fire."""

    cases: list[
        tuple[str, str, Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ] = []

    omitted = dict(audit)
    omitted["cell_results"] = list(audit["cell_results"])[1:]
    cases.append(("omitted_cell", "cell_roster_incomplete", law, omitted, traces))

    clamp_law = dict(law)
    clamp_formulas = list(law["frozen_formulas"])
    changed_formula = dict(clamp_formulas[0])
    changed_formula["assumptions"] = [1, -1]
    clamp_formulas[0] = changed_formula
    clamp_law["frozen_formulas"] = clamp_formulas
    cases.append(("inconsistent_clamp", "fixture_hash_mismatch", clamp_law, audit, traces))

    stale = dict(audit)
    stale_chains = list(audit["per_chain_results"])
    stale_row = dict(stale_chains[0])
    stale_row["source_hash"] = "sha256:" + "0" * 64
    stale_chains[0] = stale_row
    stale["per_chain_results"] = stale_chains
    cases.append(("stale_source_version", "stale_source_version", law, stale, traces))

    beta_audit = dict(audit)
    beta_chains = list(audit["per_chain_results"])
    beta_row = dict(beta_chains[0])
    beta_row["beta"] = 9.0
    beta_chains[0] = beta_row
    beta_audit["per_chain_results"] = beta_chains
    cases.append(("altered_beta", "altered_beta", law, beta_audit, traces))

    forged_traces = list(traces)
    forged = dict(forged_traces[0])
    indices = list(forged["state_indices"])
    indices[0] = (indices[0] + 1) % 64
    forged["state_indices"] = indices
    forged_traces[0] = forged
    cases.append(("forged_sample", "forged_sample", law, audit, forged_traces))

    missing = dict(audit)
    missing_chains = list(audit["per_chain_results"])
    missing_row = dict(missing_chains[0])
    missing_row.pop("costs", None)
    missing_chains[0] = missing_row
    missing["per_chain_results"] = missing_chains
    cases.append(("missing_chain_cost", "missing_chain_cost", law, missing, traces))

    rows: list[JsonDict] = []
    for control_id, detector, candidate_law, candidate_audit, candidate_traces in cases:
        observed = validate_preserved_structure(candidate_law, candidate_audit, candidate_traces)
        rows.append(
            {
                "row_type": "reducer_attack_control",
                "control_id": control_id,
                "expected_detector": detector,
                "observed_errors": observed,
                "detected": detector in observed,
                "outcome": "complete",
                "costs": {"new_sampler_draws": 0, "current_llm_calls": 0},
                "failures": [] if detector in observed else ["expected_detector_missing"],
                "censored": False,
            }
        )
    return rows


def build_validation_plan(
    repo_root: Path, private_root: Path
) -> list[validation_scope.CommandSpec]:
    """Build the exact Exp7358 affected plan with private temporary parents."""

    return validation_contract.build_command_plan(repo_root, V648_MANIFEST, private_root)


def validate_validation_plan(
    repo_root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject any drift from the scoped Exp7358 command plan."""

    return validation_contract.validate_command_plan(repo_root, V648_MANIFEST, commands)


def _receipt_set_passes(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one successful bounded receipt for each named command."""

    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def _original_gate_results(audit: Mapping[str, Any], audit_sha256: str) -> JsonDict:
    """Copy the historical decision fields without revising their failed values."""

    return {
        "source_path": AUDIT_PATH.as_posix(),
        "source_sha256": audit_sha256,
        "status": audit.get("status"),
        "verdict_class": audit.get("verdict_class"),
        "honest_verdict": audit.get("honest_verdict"),
        "flagged_adversarial": audit.get("flagged_adversarial"),
        "ising_sample_capture_complete_score": audit.get("ising_sample_capture_complete_score"),
        "ising_law_value_score": audit.get("ising_law_value_score"),
        "promotion_score": audit.get("promotion_score"),
        "acceptance_gate_results": deepcopy(audit.get("acceptance_gate_results") or []),
        "gate_check_summary": deepcopy(audit.get("gate_check_summary") or {}),
        "preservation_disposition": "unchanged_historical_disqualified_diagnostic",
    }


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute current completion and scores without changing the old gate."""

    preconditions = [
        row
        for row in artifact.get("preconditions_checked") or []
        if row.get("terminal_blocking") is True
    ]
    prerequisites = bool(preconditions) and all(row.get("passed") is True for row in preconditions)
    support_rows = list(artifact.get("support_rows") or [])
    quality_rows = list(artifact.get("sample_quality_rows") or [])
    support_roster = (
        len(support_rows) == 216
        and len(
            {(row.get("formula_id"), row.get("beta"), row.get("condition")) for row in support_rows}
        )
        == 216
    )
    chain_rows = [row for row in quality_rows if row.get("row_type") == "chain_quality"]
    cell_rows = [row for row in quality_rows if row.get("row_type") == "cell_quality"]
    quality_roster = (
        len(chain_rows) == 612
        and len(cell_rows) == 153
        and len({_chain_key(row) for row in chain_rows}) == 612
        and len({str(row.get("cell_id")) for row in cell_rows}) == 153
    )
    cold = list(artifact.get("cold_archive_reductions") or [])
    cold_match = (
        len(cold) == 2
        and all(row.get("passed") is True for row in cold)
        and cold[0].get("reduction_checksum") == cold[1].get("reduction_checksum")
        and cold[0].get("reduction_checksum") == artifact.get("archived_reduction_checksum")
    )
    attacks = list(artifact.get("reducer_attack_controls") or [])
    attacks_passed = len(attacks) == 6 and all(row.get("detected") is True for row in attacks)
    exact = artifact.get("exact_law_conclusions") or {}
    law_preserved = exact.get("law_preservation_confirmed") is True
    original = artifact.get("original_gate_results") or {}
    original_gate = next(
        (
            row
            for row in original.get("acceptance_gate_results") or []
            if row.get("check") == "all_source_cells_qualified"
        ),
        {},
    )
    original_preserved = (
        original.get("verdict_class") == "disqualified"
        and original.get("flagged_adversarial") is True
        and original_gate.get("observed") is False
        and original_gate.get("passed") is False
    )
    receipts = list(artifact.get("validation_receipts") or [])
    affected = _receipt_set_passes(receipts, REQUIRED_CHECK_NAMES)
    terminal = _receipt_set_passes(receipts, TERMINAL_CHECK_NAMES)
    validation = affected and terminal
    flagged = artifact.get("flagged_adversarial") is True
    evidence_complete = support_roster and quality_roster and cold_match and attacks_passed
    safe = original_preserved and not flagged
    if not prerequisites:
        verdict = "blocked"
        honest = "blocked_required_preserved_ising_input_unavailable"
    elif not evidence_complete or not law_preserved or not validation or not safe:
        verdict = "disqualified"
        honest = "complete_disqualified_ising_reduction_validation_or_safety_failure"
    else:
        verdict = "null"
        honest = "complete_null_preserved_all_cell_gate_failed_with_defined_laws_confirmed"
    ready = int(verdict == "null" and evidence_complete and validation and safe)
    return {
        "preconditions_passed": prerequisites,
        "support_row_roster_valid": support_roster,
        "sample_quality_row_roster_valid": quality_roster,
        "double_cold_reduction_match": cold_match,
        "reducer_attack_controls_passed": attacks_passed,
        "law_preservation_confirmed": law_preserved,
        "original_gate_results_preserved": original_preserved,
        "affected_validation_passed": affected,
        "capability_e2e_passed": terminal,
        "required_validation_passed": validation,
        "evidence_complete": evidence_complete,
        "current_safety_passed": safe,
        "original_all_cell_gate_passed": original_gate.get("passed"),
        "verdict_class": verdict,
        "honest_verdict": honest,
        "ising_reduction_complete_score": ready,
        "law_preservation_confirmed_score": int(ready and law_preserved),
        "promotion_score": 0,
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    operator: str = "==",
    terminal_blocking: bool,
) -> JsonDict:
    """Keep the threshold operator separate from expected and observed values."""

    if operator == "==":
        passed = observed == expected
    elif operator == "<=":
        passed = observed is not None and float(observed) <= float(expected)
    elif operator == ">=":
        passed = observed is not None and float(observed) >= float(expected)
    else:
        raise ValueError(f"unsupported_gate_operator:{operator}")
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "operator": operator,
        "passed": passed,
        "terminal_blocking": terminal_blocking,
    }


def build_acceptance_gates(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Separate current completion and safety from the preserved failed gate."""

    reduced = independent_reduce(artifact)
    exact = artifact.get("exact_law_conclusions") or {}
    nonempty_source = [
        row
        for row in artifact.get("sample_quality_rows") or []
        if row.get("row_type") == "cell_quality"
        and row.get("cell_kind") == "source_faithful"
        and row.get("support_status") == "nonempty_conditioned_support"
    ]
    maximum_observable_error = (
        max(float(row["max_observable_error"]) for row in nonempty_source)
        if nonempty_source
        else None
    )
    minimum_effective_samples = (
        min(float(row["minimum_effective_sample_size"]) for row in nonempty_source)
        if nonempty_source
        else None
    )
    gates = [
        _gate(
            "preconditions_passed",
            "prerequisite",
            True,
            reduced["preconditions_passed"],
            terminal_blocking=True,
        ),
        _gate(
            "support_row_roster_valid",
            "completion",
            True,
            reduced["support_row_roster_valid"],
            terminal_blocking=True,
        ),
        _gate(
            "sample_quality_row_roster_valid",
            "completion",
            True,
            reduced["sample_quality_row_roster_valid"],
            terminal_blocking=True,
        ),
        _gate(
            "double_cold_reduction_match",
            "safety",
            True,
            reduced["double_cold_reduction_match"],
            terminal_blocking=True,
        ),
        _gate(
            "reducer_attack_controls_passed",
            "safety",
            True,
            reduced["reducer_attack_controls_passed"],
            terminal_blocking=True,
        ),
        _gate(
            "maximum_energy_residual",
            "scientific_efficacy",
            ENERGY_RESIDUAL_LIMIT,
            exact.get("maximum_energy_residual"),
            operator="<=",
            terminal_blocking=True,
        ),
        _gate(
            "maximum_exact_total_variation",
            "scientific_efficacy",
            EXACT_TV_LIMIT,
            exact.get("maximum_exact_total_variation"),
            operator="<=",
            terminal_blocking=True,
        ),
        _gate(
            "law_preservation_confirmed",
            "scientific_efficacy",
            True,
            reduced["law_preservation_confirmed"],
            terminal_blocking=True,
        ),
        _gate(
            "original_gate_results_preserved",
            "safety",
            True,
            reduced["original_gate_results_preserved"],
            terminal_blocking=True,
        ),
        _gate(
            "nonempty_support_maximum_observable_error",
            "historical_diagnostic",
            OBSERVABLE_ERROR_LIMIT,
            maximum_observable_error,
            operator="<=",
            terminal_blocking=False,
        ),
        _gate(
            "nonempty_support_minimum_effective_sample_size",
            "historical_diagnostic",
            ESS_MINIMUM,
            minimum_effective_samples,
            operator=">=",
            terminal_blocking=False,
        ),
        _gate(
            "affected_validation_passed",
            "required_validation",
            True,
            reduced["affected_validation_passed"],
            terminal_blocking=True,
        ),
        _gate(
            "capability_e2e_passed",
            "required_validation",
            True,
            reduced["capability_e2e_passed"],
            terminal_blocking=True,
        ),
        _gate(
            "original_all_cell_gate",
            "historical_scientific_efficacy",
            True,
            reduced["original_all_cell_gate_passed"],
            terminal_blocking=False,
        ),
        _gate("automatic_promotion", "promotion", 0, 0, terminal_blocking=False),
    ]
    return gates


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failure and the first exact current blocking field."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    blocking = [row for row in failures if row.get("terminal_blocking") is True]
    return {
        "passed": not blocking,
        "failed_count": len(failures),
        "blocking_failed_count": len(blocking),
        "first_failure": blocking[0] if blocking else (failures[0] if failures else None),
        "failures": failures,
    }


def field_principles(artifact: Mapping[str, Any]) -> dict[str, str]:
    """Explain every output field while preserving its ordinary JSON type."""

    specific = {
        "schema": "Use a versioned schema with ordinary experiment and milestone fields.",
        "status": "Use a terminal status only after real reduction and required validation.",
        "run_date": "Use 20260918 with actual UTC start and end timestamps.",
        "preconditions_checked": "Record exact input paths, identities, hashes, classes, and resources before reduction.",
        "MODEL_SPECS": "Use an empty list because this reduction performs no current LLM work.",
        "model_invoked": "Stay false because no current LLM load or generation is attempted.",
        "invocation_counts": "Record every current LLM counter as zero with no raw current receipts.",
        "inference_substrate": "Describe current host CPU enumeration and archived-trace analysis with device and lease details.",
        "inference_substrate_class": "Use the closed CPU exact solver or simulator class.",
        "execution_venue": "Use exactly host; retain historical host_cpu only in labeled history.",
        "duration_s": "Use measured monotonic time without padding or sleep.",
        "phase_spans": "Record measured read, build, load, generate, evaluate, validate, and write spans.",
        "random_seed": "Retain the frozen formula and historical chain seeds; no new random draw occurs.",
        "reproducibility_checksum": "Bind exact code, sources, protocol, rows, controls, and validation receipts.",
        "source_artifact_hashes": "Hash original laws, audit, raw archive, logs, and current reduction sources.",
        "rows": "Retain every exact arm, archived chain, cell, and mutation-control outcome.",
        "sample_size_budget": "Account for planned, archived, censored, and unstarted work without new samples.",
        "acceptance_gate_results": "Keep expected, observed, operator, passed, category, and blocking status separate.",
        "gate_check_summary": "Name each failed field and the first current blocking failure.",
        "verifier_is_oracle": "True because exact enumeration defines finite-law truth.",
        "honest_verdict": "Use complete_ for finished work and blocked_ for unavailable preserved input.",
        "verdict_class": "Use exactly positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Describe only a current critical finding; historical flags remain in sidecars.",
        "validation_receipts": "Retain executed argv, environment, scope, return code, duration, and log hash.",
        "repository_health": "Keep unrelated dated health findings separate from affected checks.",
        "field_principles": "Explain every ordinary field without wrapping its value.",
        "promotion_score": "Always remain zero because no rollout, weight change, or publication follows.",
        "ising_reduction_complete_score": "Equal one only for full authentic archived accounting and current checks.",
        "law_preservation_confirmed_score": "Equal one only for independently enumerated equivalence wherever defined.",
        "original_gate_results": "Copy V647 gates, failures, class, and adversarial flag without revision.",
        "support_rows": "Record support definition and reason for each formula, beta, and exact energy arm.",
        "sample_quality_rows": "Record each original chain and cell with null statistics for undefined targets.",
        "historical_trace_manifest": "Retain the immutable original trace path, hash, counts, producer class, and flag.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in artifact
        if key != "field_principles"
    } | {"field_principles": specific["field_principles"]}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable source, protocol, row, decision, and validation evidence."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "random_seed",
        "source_artifact_hashes",
        "original_gate_results",
        "historical_trace_manifest",
        "support_rows",
        "sample_quality_rows",
        "reducer_attack_controls",
        "cold_archive_reductions",
        "sample_size_budget",
        "validation_receipts",
        "verdict_class",
        "honest_verdict",
        "ising_reduction_complete_score",
        "law_preservation_confirmed_score",
        "promotion_score",
    )
    return canonical_hash({key: artifact.get(key) for key in keys})


def _base_artifact(
    preserved: Mapping[str, Any],
    reduction: Mapping[str, Any],
    second_reduction: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    *,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    repository_health: Mapping[str, Any],
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one terminal-shaped record from independently reducible evidence."""

    law = preserved["law_artifact"]
    audit = preserved["audit_artifact"]
    support_rows = deepcopy(reduction["support_rows"])
    quality_rows = deepcopy(reduction["sample_quality_rows"])
    controls = [deepcopy(dict(row)) for row in attacks]
    trace_manifest = audit["raw_trace_archive"]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "building_terminal_record",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": deepcopy(preserved["preconditions"]),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": deepcopy(ZERO_CURRENT_INVOCATIONS),
            "historical": {
                "counted_as_current": False,
                "sidecar_count": len(preserved["historical_sidecars"]),
            },
        },
        "inference_substrate": "host_cpu_exact_enumeration_and_preserved_jax_trace_reduction_no_model",
        "host_computation": {
            "description": "host CPU exact enumeration and reduction of preserved JAX chain traces",
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor() or "host_cpu",
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "resource_lease": "current process owns host CPU reduction only",
            "current_gpu_operations": 0,
            "current_llm_operations": 0,
            "new_sampler_draws": 0,
        },
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "formula_seeds": deepcopy(law.get("frozen_sampling_protocol", {}).get("formula_seeds")),
            "historical_chain_seeds": deepcopy(
                audit.get("sampler_protocol", {}).get("chain_seeds")
            ),
            "new_sampling_seed": None,
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": deepcopy(preserved["source_hashes"]),
        "historical_inference_sidecars": deepcopy(preserved["historical_sidecars"]),
        "historical_validation_logs": deepcopy(preserved["validation_logs"]),
        "historical_validator_errors": deepcopy(preserved["historical_validator_errors"]),
        "historical_trace_manifest": {
            "original_path": trace_manifest.get("path"),
            "resolved_preserved_path": TRACE_PATH.as_posix(),
            "sha256": trace_manifest.get("sha256"),
            "byte_count": trace_manifest.get("byte_count"),
            "record_count": trace_manifest.get("record_count"),
            "producer_experiment_id": audit.get("experiment_id"),
            "producer_class": audit.get("inference_substrate_class"),
            "producer_execution_venue": audit.get("execution_venue"),
            "flagged_adversarial": audit.get("flagged_adversarial"),
            "verdict_class": audit.get("verdict_class"),
            "eligible_for_diagnosis": True,
            "eligible_for_current_readiness": False,
        },
        "original_gate_results": _original_gate_results(
            audit, preserved["source_hashes"][AUDIT_PATH.as_posix()]
        ),
        "protocol": {
            "formula_count": 24,
            "beta_grid": list(BETA_GRID),
            "exact_conditions": list(EXACT_CONDITIONS),
            "chain_conditions": list(CHAIN_CONDITIONS),
            "chain_count_per_cell": 4,
            "warmup_steps_per_chain": 1_000,
            "recorded_samples_per_nonempty_chain": 4_000,
            "energy_residual_limit": ENERGY_RESIDUAL_LIMIT,
            "exact_tv_limit": EXACT_TV_LIMIT,
            "observable_error_limit": OBSERVABLE_ERROR_LIMIT,
            "effective_sample_minimum": ESS_MINIMUM,
            "new_sampling": False,
            "burn_in_changed": False,
            "beta_changed": False,
            "sampler_added": False,
            "hardware_port": False,
        },
        "small_ebm_training": {
            "invoked": False,
            "fit_count": 0,
            "note": "No Gibbs fitting or other small EBM training was needed.",
        },
        "support_rows": support_rows,
        "sample_quality_rows": quality_rows,
        "rows": [*deepcopy(support_rows), *deepcopy(quality_rows), *deepcopy(controls)],
        "exact_law_conclusions": deepcopy(reduction["exact_law_conclusions"]),
        "finite_chain_conclusions": deepcopy(reduction["finite_chain_conclusions"]),
        "archived_reduction_checksum": reduction["reduction_checksum"],
        "cold_archive_reductions": [
            {
                "reduction_order": 1,
                "passed": reduction["passed"],
                "errors": deepcopy(reduction["errors"]),
                "reduction_checksum": reduction["reduction_checksum"],
                "trace_record_count": reduction["trace_record_count"],
            },
            {
                "reduction_order": 2,
                "passed": second_reduction["passed"],
                "errors": deepcopy(second_reduction["errors"]),
                "reduction_checksum": second_reduction["reduction_checksum"],
                "trace_record_count": second_reduction["trace_record_count"],
            },
        ],
        "reducer_attack_controls": controls,
        "sample_size_budget": {
            "planned_formulas": 24,
            "attempted_formulas": 24,
            "completed_formulas": 24,
            "planned_exact_rows": 216,
            "completed_exact_rows": len(support_rows),
            "planned_historical_chains": 612,
            "completed_nonempty_historical_chains": 540,
            "undefined_empty_support_chains": 72,
            "planned_historical_cells": 153,
            "completed_historical_cells": 153,
            "archived_recorded_samples": sum(
                int(row["sample_size"])
                for row in quality_rows
                if row["row_type"] == "chain_quality"
            ),
            "attempted_new_samples": 0,
            "completed_new_samples": 0,
            "censored_new_samples": 0,
            "unstarted_new_samples": 0,
            "stopping_rule": "Reduce the frozen archive twice; never start, extend, or tune a chain.",
            "remaining_work": 0,
        },
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "repository_health": deepcopy(dict(repository_health)),
        "verifier_is_oracle": True,
        "flagged_adversarial": flagged_adversarial,
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "rust_changed": False,
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": ["declared_entrypoint", "independent_cold_replay"],
        "fresh_sampling_performed": False,
        "prospective_pass_claimed": False,
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "independent_reduction": {},
        "ising_reduction_complete_score": 0,
        "law_preservation_confirmed_score": 0,
        "promotion_score": 0,
        "honest_verdict": "complete_disqualified_building_terminal_record",
        "verdict_class": "disqualified",
        "field_principles": {},
    }
    reduced = independent_reduce(artifact)
    artifact["independent_reduction"] = reduced
    for key in (
        "ising_reduction_complete_score",
        "law_preservation_confirmed_score",
        "promotion_score",
        "honest_verdict",
        "verdict_class",
    ):
        artifact[key] = reduced[key]
    artifact["status"] = {
        "blocked": "blocked_ising_reduction_required_input",
        "disqualified": "complete_ising_reduction_disqualified",
        "null": "complete_ising_reduction_null_preserved_gate",
    }.get(str(artifact["verdict_class"]), "complete_ising_reduction_terminal")
    artifact["acceptance_gate_results"] = build_acceptance_gates(artifact)
    artifact["gate_check_summary"] = gate_summary(artifact["acceptance_gate_results"])
    artifact["field_principles"] = field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test(
    preserved: Mapping[str, Any],
    reduction: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a complete in-memory artifact without writing tracked state."""

    attacks = run_reducer_attack_controls(
        preserved["law_artifact"], preserved["audit_artifact"], preserved["trace_records"]
    )
    spans = [
        {
            "phase": name,
            "start_s": index / 100,
            "end_s": (index + 1) / 100,
            "duration_s": 0.01,
            "checkpoint_at_utc": "2026-09-18T00:00:00+00:00",
        }
        for index, name in enumerate(
            ("read", "build", "load", "generate", "evaluate", "validate", "write")
        )
    ]
    return _base_artifact(
        preserved,
        reduction,
        deepcopy(reduction),
        attacks,
        receipts,
        phase_spans=spans,
        started_at_utc="2026-09-18T00:00:00+00:00",
        completed_at_utc="2026-09-18T00:00:01+00:00",
        duration_s=1.0,
        repository_health={
            "as_of": RUN_DATE,
            "status": "not_assessed_by_unit_fixture",
            "affects_required_checks": False,
            "historical_failures": [],
        },
    )


def build_blocked_artifact_for_test(failures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build an external-absence artifact used by fail-closed unit tests."""

    return build_blocked_artifact(
        preconditions=failures,
        source_hashes={},
        started_at_utc="2026-09-18T00:00:00+00:00",
        completed_at_utc="2026-09-18T00:00:01+00:00",
        duration_s=1.0,
        phase_spans=[],
    )


def build_blocked_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Publish unchanged external absence without dependent reduction work."""

    failed = [deepcopy(dict(row)) for row in preconditions if row.get("passed") is not True]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "blocked_ising_reduction_required_input",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": deepcopy(ZERO_CURRENT_INVOCATIONS),
            "historical": {"counted_as_current": False, "sidecar_count": 0},
        },
        "inference_substrate": "host_cpu_precondition_checks_only_no_model",
        "host_computation": {
            "description": "host CPU precondition checks only",
            "resource_lease": "current process",
            "current_llm_operations": 0,
            "new_sampler_draws": 0,
        },
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": None,
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [],
        "historical_validation_logs": [],
        "historical_validator_errors": [],
        "historical_trace_manifest": {},
        "original_gate_results": {},
        "protocol": {"new_sampling": False},
        "small_ebm_training": {"invoked": False, "fit_count": 0},
        "support_rows": [],
        "sample_quality_rows": [],
        "rows": [],
        "exact_law_conclusions": {},
        "finite_chain_conclusions": {},
        "archived_reduction_checksum": None,
        "cold_archive_reductions": [],
        "reducer_attack_controls": [],
        "sample_size_budget": {
            "planned_historical_chains": 612,
            "attempted_historical_chains": 0,
            "completed_historical_chains": 0,
            "unstarted_historical_chains": 612,
            "attempted_new_samples": 0,
            "remaining_work": 612,
            "stopping_rule": "Stop before dependent work when a preserved input is unavailable.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "passed": False,
            "failed_count": len(failed),
            "blocking_failed_count": len(failed),
            "first_failure": failed[0] if failed else None,
            "failures": failed,
        },
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_required_preserved_ising_input_unavailable",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "as_of": RUN_DATE,
            "status": "not_assessed_blocked_before_validation",
            "affects_required_checks": False,
            "historical_failures": [],
        },
        "independent_reduction": {},
        "field_principles": {},
        "promotion_score": 0,
        "ising_reduction_complete_score": 0,
        "law_preservation_confirmed_score": 0,
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "rust_changed": False,
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": [],
        "fresh_sampling_performed": False,
        "prospective_pass_claimed": False,
    }
    artifact["field_principles"] = field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


@lru_cache(maxsize=4)
def _cold_repo_reduction(
    root_text: str, law_sha256: str, audit_sha256: str, trace_sha256: str
) -> tuple[JsonDict, JsonDict]:
    """Cache one process-local replay while binding every preserved byte hash."""

    del law_sha256, audit_sha256, trace_sha256
    preserved = load_preserved_inputs(Path(root_text))
    reduction = reduce_preserved_evidence(
        preserved["law_artifact"], preserved["audit_artifact"], preserved["trace_records"]
    )
    return preserved, reduction


def validate_artifact(artifact: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> list[str]:
    """Cold-check declarations, source replay, rows, gates, scores, and checksum."""

    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current_model_declaration_invalid")
    if (artifact.get("invocation_counts") or {}).get("current") != ZERO_CURRENT_INVOCATIONS:
        errors.append("current_invocation_counts_invalid")
    if artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("verdict_class") not in CLOSED_VERDICTS:
        errors.append("verdict_class_invalid")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("circularity_declaration_invalid")
    if set(artifact.get("field_principles") or {}) != set(artifact):
        errors.append("field_principles_incomplete")
    if any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_artifact_field_missing")

    if artifact.get("verdict_class") == "blocked":
        if artifact.get("rows") or artifact.get("validation_receipts"):
            errors.append("blocked_artifact_has_dependent_work")
        if (artifact.get("gate_check_summary") or {}).get("first_failure") is None:
            errors.append("blocked_gate_summary_missing")
        if any(
            artifact.get(field) != 0
            for field in (
                "ising_reduction_complete_score",
                "law_preservation_confirmed_score",
                "promotion_score",
            )
        ):
            errors.append("blocked_scores_nonzero")
    else:
        if len(artifact.get("support_rows") or []) != 216:
            errors.append("support_row_roster_invalid")
        if len(artifact.get("sample_quality_rows") or []) != 765:
            errors.append("sample_quality_row_roster_invalid")
        try:
            root = repo_root.resolve()
            current_hashes = (
                sha256_file(root / LAW_PATH),
                sha256_file(root / AUDIT_PATH),
                sha256_file(root / TRACE_PATH),
            )
            preserved, cold = _cold_repo_reduction(str(root), *current_hashes)
            if artifact.get("support_rows") != cold["support_rows"]:
                errors.append("cold_support_rows_mismatch")
            if artifact.get("sample_quality_rows") != cold["sample_quality_rows"]:
                errors.append("cold_sample_quality_rows_mismatch")
            if artifact.get("archived_reduction_checksum") != cold["reduction_checksum"]:
                errors.append("cold_reduction_checksum_mismatch")
            expected_original = _original_gate_results(
                preserved["audit_artifact"], preserved["source_hashes"][AUDIT_PATH.as_posix()]
            )
            if artifact.get("original_gate_results") != expected_original:
                errors.append("original_gate_results_changed")
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("preserved_evidence_unavailable")
        reduced = independent_reduce(artifact)
        if artifact.get("independent_reduction") != reduced:
            errors.append("stored_reduction_mismatch")
        for key in (
            "verdict_class",
            "honest_verdict",
            "ising_reduction_complete_score",
            "law_preservation_confirmed_score",
            "promotion_score",
        ):
            if artifact.get(key) != reduced[key]:
                errors.append("score_mismatch")
                break
        if artifact.get("flagged_adversarial") is True and any(
            artifact.get(field) != 0
            for field in ("ising_reduction_complete_score", "law_preservation_confirmed_score")
        ):
            errors.append("adversarial_readiness_nonzero")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def utc_now() -> str:  # pragma: no cover - runtime timestamp.
    """Return one real UTC checkpoint timestamp."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit one flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7392] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, *, operation: str = "executed"
) -> JsonDict:  # pragma: no cover - runtime evidence.
    """Measure one real phase span and attach a UTC checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "operation": operation,
        "checkpoint_at_utc": utc_now(),
    }


def _terminal_commands(
    candidate: Path,
) -> list[validation_contract.PlannedCommand]:  # pragma: no cover - runtime command plan.
    """Build cold replay, raw reduction, adversarial, and strict row checks."""

    python = ".venv/bin/python"
    reducer_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7392_v648_ising_reduction import validate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v);print({'independent_raw_reduction_errors':e},flush=True);"
        "raise SystemExit(bool(e))"
    )
    return [
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--replay-artifact",
                    str(candidate),
                ),
                "measured_candidate_and_preserved_archive",
            ),
            "completion",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "independent_raw_reduction",
                (python, "-u", "-c", reducer_code, str(candidate)),
                "measured_candidate_and_preserved_archive",
            ),
            "completion",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
            ),
            "safety",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
    ]


def run_experiment(  # pragma: no cover - exercised by the declared entrypoint.
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Cold-reduce preserved evidence twice, validate it, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, hashes = collect_preconditions(root)
    spans.append(_span("read", phase_started, started))
    prerequisites_passed = all(
        row["passed"] for row in preconditions if row["terminal_blocking"] is True
    )
    progress(started, "preconditions", "end", passed=prerequisites_passed)
    if not prerequisites_passed:
        blocked = build_blocked_artifact(
            preconditions=preconditions,
            source_hashes=hashes,
            started_at_utc=started_at,
            completed_at_utc=utc_now(),
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        progress(started, "write", "before_atomic_blocked", path=output_path)
        validation_contract.atomic_json(root / output_path, blocked)
        progress(started, "write", "after_atomic_blocked", status=blocked["status"])
        return blocked

    phase_started = time.monotonic()
    progress(started, "build", "start")
    preserved = load_preserved_inputs(root)
    spans.append(_span("build", phase_started, started))
    progress(started, "build", "end", traces=len(preserved["trace_records"]))

    phase_started = time.monotonic()
    progress(started, "load", "before", operation="no_model_load_required")
    spans.append(_span("load", phase_started, started, operation="not_applicable_no_model"))
    progress(started, "load", "after", model_invoked=False)
    phase_started = time.monotonic()
    progress(started, "generate", "before", operation="no_generation_required")
    spans.append(
        _span("generate", phase_started, started, operation="not_applicable_no_generation")
    )
    progress(started, "generate", "after", generation_calls=0)

    phase_started = time.monotonic()
    progress(started, "evaluate", "before_cold_reduction", reduction=1)
    first = reduce_preserved_evidence(
        preserved["law_artifact"], preserved["audit_artifact"], preserved["trace_records"]
    )
    progress(
        started,
        "evaluate",
        "after_cold_reduction",
        reduction=1,
        checksum=first["reduction_checksum"],
    )
    progress(started, "evaluate", "before_cold_reduction", reduction=2)
    second = reduce_preserved_evidence(
        preserved["law_artifact"], preserved["audit_artifact"], preserved["trace_records"]
    )
    progress(
        started,
        "evaluate",
        "after_cold_reduction",
        reduction=2,
        checksum=second["reduction_checksum"],
    )
    attacks = run_reducer_attack_controls(
        preserved["law_artifact"], preserved["audit_artifact"], preserved["trace_records"]
    )
    spans.append(_span("evaluate", phase_started, started))
    progress(
        started,
        "evaluate",
        "end",
        cold_match=first["reduction_checksum"] == second["reduction_checksum"],
        attacks_passed=all(row["detected"] for row in attacks),
    )

    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="exp7392-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private)
    plan_errors = validate_validation_plan(root, commands)
    phase_started = time.monotonic()
    progress(started, "validate", "before_required_subprocesses", plan_errors=len(plan_errors))
    affected: list[JsonDict] = []
    if not plan_errors:
        affected = validation_contract.run_categorized_commands(
            root,
            [
                validation_contract.PlannedCommand(command, "required_validation", True)
                for command in commands
            ],
            log_dir=raw_dir / "validation/required",
        )
    progress(
        started,
        "validate",
        "after_required_subprocesses",
        passed=_receipt_set_passes(affected, REQUIRED_CHECK_NAMES),
    )
    spans.append(_span("validate", phase_started, started, operation="affected_checks"))
    repository_health = {
        "as_of": RUN_DATE,
        "status": "not_assessed_by_scoped_experiment",
        "scope": "affected checks only; no repository-wide fallback command",
        "affects_required_checks": False,
        "historical_failures": [],
    }
    candidate = _base_artifact(
        preserved,
        first,
        second,
        attacks,
        affected,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        repository_health=repository_health,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    validation_contract.atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    progress(started, "validate", "before_terminal_subprocesses")
    terminal = validation_contract.run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("validate", phase_started, started, operation="terminal_checks"))
    terminal_passed = _receipt_set_passes(terminal, TERMINAL_CHECK_NAMES)
    critical = any("[CRITICAL]" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "validate",
        "after_terminal_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    phase_started = time.monotonic()
    spans.append(_span("write", phase_started, started, operation="terminal_payload_assembly"))
    final = _base_artifact(
        preserved,
        first,
        second,
        attacks,
        [*affected, *terminal],
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        repository_health=repository_health,
        flagged_adversarial=critical,
    )
    errors = validate_artifact(final, root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "write", "before_atomic_terminal", path=output_path)
    validation_contract.atomic_json(candidate_path, final)
    validation_contract.atomic_json(root / output_path, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the declared run and cold-replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--replay-artifact", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reduction or cold-check one existing artifact."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.replay_artifact is not None:
        artifact = _read_object(args.replay_artifact)
        errors = validate_artifact(artifact, REPO_ROOT)
        print(json.dumps({"cold_artifact_replay_errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
