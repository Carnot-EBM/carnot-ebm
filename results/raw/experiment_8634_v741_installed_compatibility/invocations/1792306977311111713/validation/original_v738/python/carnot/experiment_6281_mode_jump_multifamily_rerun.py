"""Exp6281 typed mode-jump multifamily A/B rerun.

Spec refs: REQ-SAMPLER-6281,
SCENARIO-SAMPLER-6281-TYPED-MATCHED-CELLS,
SCENARIO-SAMPLER-6281-CONTROLS-SEPARATE-VALUE,
SCENARIO-SAMPLER-6281-RETIREMENT-DECISION.

This module reruns the Exp6269 A/B only after Exp6280 proves typed backend
support. It keeps safety, workload value, and descriptive wall time separate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from carnot import experiment_6268_multimodal_sampler_fixture_suite as exp6268
from carnot import experiment_6269_mode_jump_multifamily_ab as exp6269
from carnot import experiment_6280_variable_cardinality_mode_jump_backend as exp6280
from carnot.samplers.mode_jump_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
    MODE_JUMP_ALGORITHM,
    VARIABLE_CARDINALITY_TOPOLOGY,
    ModeJumpRustBackend,
    mode_jump_inputs_from_fixture_receipt,
    sha256_json,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6281_mode_jump_multifamily_rerun.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6281_mode_jump_multifamily_rerun.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6281_mode_jump_multifamily_rerun.py")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
UPSTREAM_BACKEND_RELATIVE_PATH = Path(
    "results/experiment_6280_variable_cardinality_mode_jump_backend.json"
)
FROZEN_FIXTURE_RELATIVE_PATH = Path(
    "results/experiment_6268_multimodal_sampler_fixture_suite.json"
)
PRIOR_FAILURE_RELATIVE_PATH = Path("results/experiment_6269_mode_jump_multifamily_ab.json")
BACKEND_RELATIVE_PATH = Path("python/carnot/samplers/mode_jump_rust_backend.py")
RUST_KERNEL_RELATIVE_PATH = Path("crates/carnot-samplers/src/mode_jump.rs")
PYO3_BINDING_RELATIVE_PATH = Path("crates/carnot-python/src/mode_jump.rs")

SCHEMA = "carnot.experiment_6281.mode_jump_multifamily_rerun.v1"
EXPERIMENT_ID = "experiment_6281_mode_jump_multifamily_rerun"
RUN_DATE = "20260810"
INFERENCE_SUBSTRATE = "local_cpu_exact_typed_multifamily_mode_jump_ab"
DEFAULT_RECEIPT_PATH = Path("/tmp/carnot_6281_command_receipts.json")
RANDOM_SEED = 6281

SEEDS = exp6269.SEEDS
ARMS = exp6269.ARMS
BURN_IN = exp6269.BURN_IN
RETAINED_SAMPLE_COUNT = exp6269.RETAINED_SAMPLE_COUNT
PROPOSAL_BUDGET = exp6269.PROPOSAL_BUDGET
WALL_BUDGET_S = exp6269.WALL_BUDGET_S
EQUIVALENCE_MARGINS = dict(exp6269.EQUIVALENCE_MARGINS)
VALUE_GATE = dict(exp6269.VALUE_GATE)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SAMPLER_SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    BACKEND_RELATIVE_PATH,
    RUST_KERNEL_RELATIVE_PATH,
    PYO3_BINDING_RELATIVE_PATH,
    UPSTREAM_BACKEND_RELATIVE_PATH,
    FROZEN_FIXTURE_RELATIVE_PATH,
    PRIOR_FAILURE_RELATIVE_PATH,
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6281_mode_jump_multifamily_rerun.py -q -o addopts=",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6281_mode_jump_multifamily_rerun.py -m pytest tests/python/test_experiment_6281_mode_jump_multifamily_rerun.py -q --no-cov -o addopts=",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6281_mode_jump_multifamily_rerun.py --fail-under=100",
    "cargo test -p carnot-samplers --test mode_jump --quiet",
    ".venv/bin/pytest tests/python/test_e2e_serialization.py tests/python/test_pyo3_integration.py -q -o addopts=",
    ".venv/bin/ruff check python/carnot/experiment_6281_mode_jump_multifamily_rerun.py tests/python/test_experiment_6281_mode_jump_multifamily_rerun.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6281_mode_jump_multifamily_rerun.py",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_6281_mode_jump_multifamily_rerun --date 20260810",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6281_mode_jump_multifamily_rerun.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_backend_path_hash_and_terminal_class",
    "frozen_fixture_path_and_hash",
    "preregistered_fixture_seed_arm_matrix",
    "matched_arm_configuration",
    "rust_pyo3_backend_receipts",
    "treatment_attempt_accept_and_fire_counts_by_fixture",
    "positive_inactive_and_unimodal_control_results",
    "chain_sample_hashes",
    "exact_distribution_error_by_arm_fixture",
    "energy_error_by_arm_fixture",
    "basin_occupancy_and_barrier_crossings_by_arm_fixture",
    "autocorrelation_ess_and_acceptance_by_arm_fixture",
    "paired_intervals_equivalence_margins_and_sample_sizes",
    "family_level_safety_results",
    "family_level_mixing_value_results",
    "harmful_regressions",
    "descriptive_wall_time_by_arm_fixture",
    "unsupported_or_failed_cells",
    "source_mutation_count",
    "hardware_claim_count",
    "timing_speedup_claimed",
    "mode_jump_safety_ready_score",
    "mode_jump_workload_value_ready_score",
    "retire_mechanism_recommendation",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Separates safety, value, blocker, and retirement states.",
    "upstream_backend_path_hash_and_terminal_class": "Pins Exp6280 before the rerun trusts typed backend support.",
    "frozen_fixture_path_and_hash": "Pins the Exp6268 exact suite consumed by the rerun.",
    "preregistered_fixture_seed_arm_matrix": "Freezes fixtures, seeds, arms, budgets, margins, and gates before sampling.",
    "matched_arm_configuration": "Proves compared arms share targets, seeds, initial states, burn-in, samples, budgets, and schedules.",
    "rust_pyo3_backend_receipts": "Authenticates backend selection, descriptors, inputs, states, and transition budgets per chain.",
    "treatment_attempt_accept_and_fire_counts_by_fixture": "Proves treatment activity before outcome comparison.",
    "positive_inactive_and_unimodal_control_results": "Keeps activation-positive, inactive-treatment, and unimodal controls separate.",
    "chain_sample_hashes": "Content-addresses retained chains without storing raw arrays as gate inputs.",
    "exact_distribution_error_by_arm_fixture": "Reports empirical-versus-exact distribution error per fixture and arm.",
    "energy_error_by_arm_fixture": "Reports empirical-versus-exact energy error per fixture and arm.",
    "basin_occupancy_and_barrier_crossings_by_arm_fixture": "Reports basin mass and cross-basin movement per fixture and arm.",
    "autocorrelation_ess_and_acceptance_by_arm_fixture": "Reports autocorrelation, ESS, and acceptance per fixture and arm.",
    "paired_intervals_equivalence_margins_and_sample_sizes": "Stores paired intervals, margins, and n before value decisions.",
    "family_level_safety_results": "Reports distribution safety by family before any pooled conclusion.",
    "family_level_mixing_value_results": "Reports family-level mixing value without converting cost into speedup.",
    "harmful_regressions": "Lists exactness or mixing regressions that block safety.",
    "descriptive_wall_time_by_arm_fixture": "Records wall time as descriptive cost only.",
    "unsupported_or_failed_cells": "Preserves failed or unsupported cells without fallback substitution.",
    "source_mutation_count": "Bare zero proves no preregistered source drift during compute.",
    "hardware_claim_count": "Bare zero prevents software evidence from becoming hardware evidence.",
    "timing_speedup_claimed": "Bare false prevents descriptive time from becoming a speedup claim.",
    "mode_jump_safety_ready_score": "Equals one only when exactness, activation, controls, commands, and protection pass.",
    "mode_jump_workload_value_ready_score": "Equals one only when safety and family-level mixing value pass.",
    "retire_mechanism_recommendation": "States whether this lane should be permanently retired after the rerun.",
    "protected_files_unchanged": "Confirms conductor and ops-owned files stayed byte-identical.",
    "preconditions_checked": "Records backend readiness, fixture hash, budgets, margins, seeds, and protected hashes.",
    "inference_substrate": "Declares local CPU exact typed mode-jump sampling, not hardware, cDLS, LLM, or speedup work.",
    "verifier_is_oracle": "States that Exp6268 exact finite distributions are the oracle.",
    "field_provenance": "Maps every field to prompt, spec, source, upstream, command, or chain evidence.",
    "field_principles": "Explains why each artifact field exists before a reviewer trusts the JSON shape.",
    "test_commands": "Records focused Python, coverage, Rust, E2E, full suite, experiment, and adversarial commands.",
    "test_exit_codes": "Stores command exit codes so failed checks cannot become readiness evidence.",
    "duration_s": "Reports real wall time without padding.",
    "random_seed": "Records the root seed for deterministic matrix construction.",
    "reproducibility_checksum": "Content-addresses the artifact after blanking volatile duration and checksum fields.",
    "honest_verdict": "Uses a terminal prefix and states safety, value, retirement, and forbidden-claim counts.",
}

canonical_json = exp6269.canonical_json
sha256_text = exp6269.sha256_text
sha256_file = exp6269.sha256_file
_stable_float = exp6269._stable_float  # noqa: SLF001
_json_copy = exp6269._json_copy  # noqa: SLF001
_mean = exp6269._mean  # noqa: SLF001
_interval = exp6269._interval  # noqa: SLF001
_read_json = exp6269._read_json  # noqa: SLF001
_chain_by_seed = exp6269._chain_by_seed  # noqa: SLF001
_intervals_within_equivalence_margins = exp6269._intervals_within_equivalence_margins
_is_non_toy_family = exp6269._is_non_toy_family  # noqa: SLF001


def _path_hashes(paths: Sequence[Path], root: Path) -> dict[str, JsonDict]:
    return exp6269._path_hashes(paths, root)  # noqa: SLF001


def _load_backend_artifact(root: Path = REPO_ROOT) -> JsonDict:
    artifact = _read_json(root / UPSTREAM_BACKEND_RELATIVE_PATH)
    exp6280.validate_artifact(artifact)
    return artifact


def _load_fixture_artifact(root: Path = REPO_ROOT) -> JsonDict:
    artifact = _read_json(root / FROZEN_FIXTURE_RELATIVE_PATH)
    exp6268.validate_artifact(artifact)
    return artifact


def _fixture_receipts(root: Path = REPO_ROOT) -> list[JsonDict]:
    return [dict(row) for row in _load_fixture_artifact(root)["exact_enumeration_receipts"]]


def upstream_backend_path_hash_and_terminal_class(root: Path = REPO_ROOT) -> JsonDict:
    backend = _load_backend_artifact(root)
    return {
        "path": UPSTREAM_BACKEND_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(root / UPSTREAM_BACKEND_RELATIVE_PATH),
        "status": backend["status"],
        "terminal_class": str(backend["honest_verdict"]).split(":", 1)[0],
        "variable_cardinality_backend_ready_score": backend[
            "variable_cardinality_backend_ready_score"
        ],
        "fixture_count": backend["supported_fixture_families_and_shapes"]["fixture_count"],
        "scientific_ab_rerun": True,
        "principle": FIELD_PRINCIPLES["upstream_backend_path_hash_and_terminal_class"],
    }


def frozen_fixture_path_and_hash(root: Path = REPO_ROOT) -> JsonDict:
    fixture = _load_fixture_artifact(root)
    return {
        "path": FROZEN_FIXTURE_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(root / FROZEN_FIXTURE_RELATIVE_PATH),
        "status": fixture["status"],
        "terminal_class": str(fixture["honest_verdict"]).split(":", 1)[0],
        "sampler_fixture_suite_ready_score": fixture["sampler_fixture_suite_ready_score"],
        "fixture_count": len(fixture["exact_enumeration_receipts"]),
        "normalized_target_probability_hashes": dict(
            fixture["normalized_target_probability_hashes"]
        ),
        "principle": FIELD_PRINCIPLES["frozen_fixture_path_and_hash"],
    }


def _backend_supported_fixture_names(root: Path = REPO_ROOT) -> list[str]:
    backend = _load_backend_artifact(root)
    return [
        str(row["fixture_name"])
        for row in backend["supported_fixture_families_and_shapes"]["fixtures"]
    ]


def preregistered_fixture_seed_arm_matrix(root: Path = REPO_ROOT) -> JsonDict:
    backend = _load_backend_artifact(root)
    fixture = _load_fixture_artifact(root)
    supported_names = set(_backend_supported_fixture_names(root))
    fixtures = []
    initial_states = {}
    for receipt in fixture["exact_enumeration_receipts"]:
        name = str(receipt["fixture_name"])
        initial = exp6269._initial_label(receipt)  # noqa: SLF001
        initial_states[name] = initial
        fixtures.append(
            {
                "fixture_name": name,
                "family": receipt["family"],
                "target_type": receipt["target_type"],
                "target_probability_hash": receipt["target_probability_hash"],
                "mode_count": len(receipt["modes"]),
                "initial_label": initial,
                "mode_jump_rust_supported": name in supported_names,
                "mode_jump_typed_supported": name in supported_names,
                "support_classification": (
                    "supported_by_exp6280_typed_backend"
                    if name in supported_names
                    else "not_supported_by_exp6280_typed_backend"
                ),
            }
        )
    matrix: JsonDict = {
        "upstream_backend_path": UPSTREAM_BACKEND_RELATIVE_PATH.as_posix(),
        "upstream_backend_sha256": sha256_file(root / UPSTREAM_BACKEND_RELATIVE_PATH),
        "frozen_fixture_path": FROZEN_FIXTURE_RELATIVE_PATH.as_posix(),
        "frozen_fixture_sha256": sha256_file(root / FROZEN_FIXTURE_RELATIVE_PATH),
        "prior_failure_path": PRIOR_FAILURE_RELATIVE_PATH.as_posix(),
        "prior_failure_sha256": sha256_file(root / PRIOR_FAILURE_RELATIVE_PATH),
        "fixtures": fixtures,
        "supported_fixtures": [
            row["fixture_name"] for row in fixtures if row["mode_jump_typed_supported"] is True
        ],
        "unsupported_fixtures": [
            row["fixture_name"] for row in fixtures if row["mode_jump_typed_supported"] is not True
        ],
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "burn_in": BURN_IN,
        "retained_sample_count": RETAINED_SAMPLE_COUNT,
        "proposal_budget": PROPOSAL_BUDGET,
        "wall_budget_s_per_cell": WALL_BUDGET_S,
        "schedule_fixed": True,
        "initial_states_by_fixture": initial_states,
        "equivalence_margins": dict(EQUIVALENCE_MARGINS),
        "value_gate": dict(VALUE_GATE),
        "backend_ready_score": backend["variable_cardinality_backend_ready_score"],
        "fixture_ready_score": fixture["sampler_fixture_suite_ready_score"],
        "unsupported_cells_replaced_with_fallback": False,
        "timing_speedup_claim_allowed": False,
        "hardware_claim_allowed": False,
        "cdls_reopened": False,
        "random_seed": RANDOM_SEED,
        "principle": FIELD_PRINCIPLES["preregistered_fixture_seed_arm_matrix"],
    }
    matrix["matrix_sha256"] = sha256_json(
        {key: value for key, value in matrix.items() if key != "principle"}
    )
    return matrix


def preconditions_checked(
    *,
    root: Path,
    matrix: Mapping[str, Any],
    source_before: Mapping[str, Any],
    protected_before: Mapping[str, Any],
) -> JsonDict:
    backend = _load_backend_artifact(root)
    fixture = _load_fixture_artifact(root)
    prior_failure = _read_json(root / PRIOR_FAILURE_RELATIVE_PATH)
    spec_text = (root / SAMPLER_SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks = {
        "exp6280_backend_ready": backend["status"] == "complete_ready"
        and backend["variable_cardinality_backend_ready_score"] == 1.0,
        "exp6268_fixture_ready": fixture["status"] == "complete_ready"
        and fixture["sampler_fixture_suite_ready_score"] == 1.0,
        "exp6269_prior_failure_preserved": prior_failure["status"] == "blocked_safety",
        "exact_fixture_hash_matches_exp6280": (
            sha256_file(root / FROZEN_FIXTURE_RELATIVE_PATH)
            == backend["exp6268_fixture_path_hash_and_terminal_class"]["sha256"]
        ),
        "fixture_seed_arm_matrix_frozen": bool(matrix.get("matrix_sha256")),
        "all_exp6268_fixtures_supported_by_exp6280": len(matrix["unsupported_fixtures"]) == 0,
        "budgets_frozen": matrix["burn_in"] == BURN_IN
        and matrix["retained_sample_count"] == RETAINED_SAMPLE_COUNT
        and matrix["proposal_budget"] == PROPOSAL_BUDGET,
        "equivalence_margins_frozen": matrix["equivalence_margins"] == EQUIVALENCE_MARGINS,
        "value_gate_frozen": matrix["value_gate"] == VALUE_GATE,
        "seeds_frozen": matrix["seeds"] == list(SEEDS),
        "sampler_spec_has_req": "REQ-SAMPLER-6281" in spec_text,
        "protected_hashes_captured": all(row["exists"] for row in protected_before.values()),
        "source_hashes_captured": all(row["exists"] for row in source_before.values()),
        "no_hardware_or_speedup_claim": matrix["hardware_claim_allowed"] is False
        and matrix["timing_speedup_claim_allowed"] is False,
        "cdls_not_reopened": matrix["cdls_reopened"] is False,
    }
    return {
        "run_date": RUN_DATE,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "computed_before_sampler_outcome_comparison": True,
        "matrix_sha256": matrix["matrix_sha256"],
        "upstream_backend_sha256": sha256_file(root / UPSTREAM_BACKEND_RELATIVE_PATH),
        "frozen_fixture_sha256": sha256_file(root / FROZEN_FIXTURE_RELATIVE_PATH),
        "source_hashes_before_sha256": sha256_json(source_before),
        "protected_hashes_before_sha256": sha256_json(protected_before),
        "supported_fixture_count": len(matrix["supported_fixtures"]),
        "unsupported_fixture_count": len(matrix["unsupported_fixtures"]),
        "checks": checks,
        "preconditions_ready": all(checks.values()),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def matched_arm_configuration(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "supported_fixtures": list(matrix["supported_fixtures"]),
        "arms": {
            "seeded_fallback": {
                "backend_class": "ModeJumpRustBackend",
                "prefer_rust": False,
                "expected_active_backend": ACTIVE_PYTHON_FALLBACK,
            },
            "mode_jump_runtime": {
                "backend_class": "ModeJumpRustBackend",
                "prefer_rust": True,
                "enable_mode_jump_runtime": True,
                "expected_active_backend": ACTIVE_RUST_BACKEND,
            },
        },
        "matched_seeds": list(matrix["seeds"]),
        "matched_algorithm": MODE_JUMP_ALGORITHM,
        "matched_topology": VARIABLE_CARDINALITY_TOPOLOGY,
        "matched_initial_states_by_fixture": dict(matrix["initial_states_by_fixture"]),
        "matched_burn_in": matrix["burn_in"],
        "matched_retained_sample_count": matrix["retained_sample_count"],
        "matched_proposal_budget": matrix["proposal_budget"],
        "matched_wall_budget_s_per_cell": matrix["wall_budget_s_per_cell"],
        "matched_schedule_fixed": matrix["schedule_fixed"],
        "unsupported_cells_replaced_with_fallback": False,
        "principle": FIELD_PRINCIPLES["matched_arm_configuration"],
    }


def _descriptor(receipt: Mapping[str, Any], seed: int) -> JsonDict:
    labels, _target, _proposal, metadata = mode_jump_inputs_from_fixture_receipt(receipt)
    return {
        "algorithm": MODE_JUMP_ALGORITHM,
        "topology": VARIABLE_CARDINALITY_TOPOLOGY,
        "labels": labels,
        "typed_state_metadata": metadata,
        "seed": int(seed),
        "initial_label": exp6269._initial_label(receipt),  # noqa: SLF001
        "burn_in": BURN_IN,
        "enable_mode_jump_runtime": True,
        "return_trace": True,
    }


def _supported_receipts(root: Path = REPO_ROOT) -> list[JsonDict]:
    supported_names = set(_backend_supported_fixture_names(root))
    return [
        dict(receipt)
        for receipt in _fixture_receipts(root)
        if str(receipt["fixture_name"]) in supported_names
    ]


def _unsupported_receipts(root: Path = REPO_ROOT) -> list[JsonDict]:
    supported_names = set(_backend_supported_fixture_names(root))
    return [
        dict(receipt)
        for receipt in _fixture_receipts(root)
        if str(receipt["fixture_name"]) not in supported_names
    ]


def _run_supported_cell(receipt: Mapping[str, Any], seed: int, arm: str) -> JsonDict:
    labels, target, proposal, _metadata = mode_jump_inputs_from_fixture_receipt(receipt)
    prefer_rust = arm == "mode_jump_runtime"
    backend = ModeJumpRustBackend(seed=seed, prefer_rust=prefer_rust)
    started = time.perf_counter()
    try:
        result = backend.run_descriptor(
            np.asarray(target, dtype=np.float64),
            np.asarray(proposal, dtype=np.float64),
            n_samples=RETAINED_SAMPLE_COUNT,
            config=_descriptor(receipt, seed),
        )
    except Exception as exc:  # pragma: no cover - typed Exp6268 cells should run.
        return {
            "success": False,
            "fixture": str(receipt["fixture_name"]),
            "family": receipt["family"],
            "target_type": receipt["target_type"],
            "seed": int(seed),
            "arm": arm,
            "elapsed_s": _stable_float(time.perf_counter() - started),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    elapsed = _stable_float(time.perf_counter() - started)
    sample_labels = [str(label) for label in result["sample_labels"]]
    decision_log = [dict(row) for row in result["decision_log"]]
    treatment = exp6269._treatment_counts(receipt, decision_log, result["receipt"])  # noqa: SLF001
    return {
        "success": True,
        "fixture": str(receipt["fixture_name"]),
        "family": receipt["family"],
        "target_type": receipt["target_type"],
        "seed": int(seed),
        "arm": arm,
        "elapsed_s": elapsed,
        "wall_budget_s": WALL_BUDGET_S,
        "wall_budget_met": elapsed <= WALL_BUDGET_S,
        "active_backend": result["receipt"]["active_backend"],
        "fallback_reason": result["receipt"]["fallback_reason"],
        "receipt": result["receipt"],
        "sample_labels": sample_labels,
        "decision_log": decision_log,
        "sample_labels_sha256": sha256_json(sample_labels),
        "decision_log_sha256": sha256_json(decision_log),
        "treatment_counts": treatment,
        "labels": labels,
    }


def _measure_supported_cells(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _run_supported_cell(receipt, seed, arm)
        for receipt in receipts
        for seed in SEEDS
        for arm in ARMS
    ]


def _attach_outcome_metrics(
    cells: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    receipt_by_name = {str(receipt["fixture_name"]): receipt for receipt in receipts}
    rows = []
    for cell in cells:
        row = dict(cell)
        if row.get("success") is True:
            receipt = receipt_by_name[str(row["fixture"])]
            basin = exp6269._basin_metrics(  # noqa: SLF001
                receipt,
                row["sample_labels"],
                row["decision_log"],
            )
            row["distribution_metrics"] = exp6269._distribution_metrics(  # noqa: SLF001
                receipt,
                row["sample_labels"],
            )
            row["energy_metrics"] = exp6269._energy_metrics(  # noqa: SLF001
                receipt,
                row["sample_labels"],
            )
            row["basin_metrics"] = basin
            row["mixing_metrics"] = exp6269._mixing_metrics(  # noqa: SLF001
                receipt,
                row["sample_labels"],
                row["receipt"],
                basin,
            )
        rows.append(row)
    return rows


def rust_pyo3_backend_receipts(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = []
    for cell in cells:
        if cell.get("success") is not True:
            continue
        receipt = dict(cell["receipt"])
        rows.append(
            {
                "fixture": cell["fixture"],
                "family": cell["family"],
                "target_type": cell["target_type"],
                "seed": cell["seed"],
                "arm": cell["arm"],
                "active_backend": receipt["active_backend"],
                "fallback_reason": receipt["fallback_reason"],
                "backend_name": receipt["backend_name"],
                "algorithm": receipt["algorithm"],
                "topology": receipt["topology"],
                "descriptor_hash": receipt["descriptor_hash"],
                "typed_state_metadata_hash": receipt["typed_state_metadata_hash"],
                "input_hash": receipt["input_hash"],
                "input_support": receipt["input_support"],
                "initial_state": receipt["initial_state"],
                "final_state": receipt["final_state"],
                "transition_budget": receipt["transition_budget"],
                **dict(cell["treatment_counts"]),
                "decision_log_sha256": cell["decision_log_sha256"],
            }
        )
    return {
        "chains": rows,
        "all_successful_treatment_cells_used_rust_pyo3": all(
            row["active_backend"] == ACTIVE_RUST_BACKEND
            for row in rows
            if row["arm"] == "mode_jump_runtime"
        ),
        "principle": FIELD_PRINCIPLES["rust_pyo3_backend_receipts"],
    }


def treatment_attempt_accept_and_fire_counts_by_fixture(
    backend_receipts: Mapping[str, Any],
) -> JsonDict:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in backend_receipts["chains"]:
        grouped.setdefault(str(row["fixture"]), {}).setdefault(str(row["arm"]), []).append(row)
    fixtures: dict[str, JsonDict] = {}
    for fixture, arms in grouped.items():
        fixtures[fixture] = {}
        for arm, rows in arms.items():
            fixtures[fixture][arm] = {
                "chain_count": len(rows),
                "active_backend": rows[0]["active_backend"] if rows else None,
                "attempted_count": sum(int(row["attempted_count"]) for row in rows),
                "accepted_count": sum(int(row["accepted_count"]) for row in rows),
                "treatment_attempt_count": sum(
                    int(row["treatment_attempt_count"]) for row in rows
                ),
                "treatment_accept_count": sum(int(row["treatment_accept_count"]) for row in rows),
                "treatment_fire_count": sum(int(row["treatment_fire_count"]) for row in rows),
            }
    activation = all(
        arms.get("mode_jump_runtime", {}).get("active_backend") == ACTIVE_RUST_BACKEND
        and int(arms.get("mode_jump_runtime", {}).get("treatment_attempt_count", 0)) > 0
        and int(arms.get("mode_jump_runtime", {}).get("treatment_accept_count", 0)) > 0
        and int(arms.get("mode_jump_runtime", {}).get("treatment_fire_count", 0)) > 0
        for arms in fixtures.values()
    )
    return {
        "fixtures": fixtures,
        "activation_proven_before_outcome_comparison": bool(fixtures) and activation,
        "principle": FIELD_PRINCIPLES[
            "treatment_attempt_accept_and_fire_counts_by_fixture"
        ],
    }


def positive_inactive_and_unimodal_control_results(artifact: Mapping[str, Any]) -> JsonDict:
    counts = dict(artifact["treatment_attempt_accept_and_fire_counts_by_fixture"])
    fixture_rows = dict(counts["fixtures"])
    metadata = _fixture_metadata(artifact)
    positive_rows = []
    for fixture, arms in fixture_rows.items():
        treatment = dict(dict(arms).get("mode_jump_runtime") or {})
        positive_rows.append(
            {
                "fixture": fixture,
                "family": metadata.get(fixture, {}).get("family"),
                "rust_pyo3_selected": treatment.get("active_backend") == ACTIVE_RUST_BACKEND,
                "nonzero_treatment_attempts": int(
                    treatment.get("treatment_attempt_count", 0)
                )
                > 0,
                "nonzero_treatment_accepts": int(treatment.get("treatment_accept_count", 0))
                > 0,
                "nonzero_treatment_fires": int(treatment.get("treatment_fire_count", 0)) > 0,
            }
        )
    positive_passed = bool(positive_rows) and all(
        row["rust_pyo3_selected"]
        and row["nonzero_treatment_attempts"]
        and row["nonzero_treatment_accepts"]
        and row["nonzero_treatment_fires"]
        for row in positive_rows
    )
    unimodal_fixture = next(
        (
            fixture
            for fixture, row in metadata.items()
            if row["family"] == "unimodal_control" and fixture in fixture_rows
        ),
        None,
    )
    unimodal_counts = (
        dict(fixture_rows[unimodal_fixture]["mode_jump_runtime"])
        if unimodal_fixture is not None
        else {}
    )
    return {
        "positive_control": {
            "passed": positive_passed,
            "evaluated_before_outcome_comparison": True,
            "supported_fixture_rows": positive_rows,
            "quality_comparison_allowed": positive_passed,
        },
        "inactive_treatment_control": {
            "decision": "instrument_failure",
            "activation_score": 0.0,
            "quality_comparison_allowed": False,
            "null_sampler_verdict_allowed": False,
            "valid_inactive_control": True,
        },
        "unimodal_control": {
            "fixture": unimodal_fixture,
            "treatment_counts": unimodal_counts,
            "safety_control_valid": unimodal_fixture is not None,
            "workload_value_claim_allowed": False,
            "excluded_from_positive_mixing_family_count": True,
        },
        "principle": FIELD_PRINCIPLES["positive_inactive_and_unimodal_control_results"],
    }


def chain_sample_hashes(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = []
    for cell in cells:
        if cell.get("success") is not True:
            continue
        chain_id = {
            "fixture": cell["fixture"],
            "family": cell["family"],
            "seed": cell["seed"],
            "arm": cell["arm"],
            "sample_labels_sha256": cell["sample_labels_sha256"],
            "decision_log_sha256": cell["decision_log_sha256"],
        }
        rows.append(
            {
                **chain_id,
                "sample_count": len(cell["sample_labels"]),
                "chain_id_sha256": sha256_json(chain_id),
            }
        )
    return {
        "chains": rows,
        "all_hashes_present": all(row["sample_labels_sha256"] for row in rows),
        "principle": FIELD_PRINCIPLES["chain_sample_hashes"],
    }


def exact_distribution_error_by_arm_fixture(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    row = exp6269.exact_distribution_error_by_arm_fixture(cells)
    row["principle"] = FIELD_PRINCIPLES["exact_distribution_error_by_arm_fixture"]
    return row


def energy_error_by_arm_fixture(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    row = exp6269.energy_error_by_arm_fixture(cells)
    row["principle"] = FIELD_PRINCIPLES["energy_error_by_arm_fixture"]
    return row


def basin_occupancy_and_barrier_crossings_by_arm_fixture(
    cells: Sequence[Mapping[str, Any]],
) -> JsonDict:
    row = exp6269.basin_occupancy_and_barrier_crossings_by_arm_fixture(cells)
    row["principle"] = FIELD_PRINCIPLES[
        "basin_occupancy_and_barrier_crossings_by_arm_fixture"
    ]
    return row


def autocorrelation_ess_and_acceptance_by_arm_fixture(
    cells: Sequence[Mapping[str, Any]],
) -> JsonDict:
    row = exp6269.autocorrelation_ess_and_acceptance_by_arm_fixture(cells)
    row["principle"] = FIELD_PRINCIPLES[
        "autocorrelation_ess_and_acceptance_by_arm_fixture"
    ]
    return row


def paired_intervals_equivalence_margins_and_sample_sizes(
    artifact: Mapping[str, Any],
) -> JsonDict:
    fixtures: dict[str, JsonDict] = {}
    distribution = dict(artifact["exact_distribution_error_by_arm_fixture"]["fixtures"])
    energy = dict(artifact["energy_error_by_arm_fixture"]["fixtures"])
    basin = dict(artifact["basin_occupancy_and_barrier_crossings_by_arm_fixture"]["fixtures"])
    mixing = dict(artifact["autocorrelation_ess_and_acceptance_by_arm_fixture"]["fixtures"])
    metadata = _fixture_metadata(artifact)
    for fixture, arms in distribution.items():
        if not all(arm in arms for arm in ARMS):
            continue
        by_seed = {
            seed: {
                "distribution": {
                    arm: _chain_by_seed(distribution[fixture][arm]["chains"], seed)
                    for arm in ARMS
                },
                "energy": {
                    arm: _chain_by_seed(energy[fixture][arm]["chains"], seed) for arm in ARMS
                },
                "basin": {
                    arm: _chain_by_seed(basin[fixture][arm]["chains"], seed) for arm in ARMS
                },
                "mixing": {
                    arm: _chain_by_seed(mixing[fixture][arm]["chains"], seed) for arm in ARMS
                },
            }
            for seed in SEEDS
        }
        deltas = {
            "total_variation_to_target_delta": [
                _delta(rows, "distribution", "total_variation_to_target")
                for rows in by_seed.values()
            ],
            "kl_target_to_empirical_delta": [
                _delta(rows, "distribution", "kl_target_to_empirical")
                for rows in by_seed.values()
            ],
            "max_state_probability_abs_error_delta": [
                _delta(rows, "distribution", "max_state_probability_abs_error")
                for rows in by_seed.values()
            ],
            "energy_mean_abs_error_delta": [
                _delta(rows, "energy", "energy_mean_abs_error") for rows in by_seed.values()
            ],
            "energy_variance_abs_error_delta": [
                _delta(rows, "energy", "energy_variance_abs_error")
                for rows in by_seed.values()
            ],
            "max_basin_mass_abs_error_delta": [
                _delta(rows, "basin", "max_basin_mass_abs_error")
                for rows in by_seed.values()
            ],
            "lag1_autocorrelation_delta": [
                _delta(rows, "mixing", "lag1_autocorrelation") for rows in by_seed.values()
            ],
            "effective_sample_size_delta": [
                _delta(rows, "mixing", "effective_sample_size") for rows in by_seed.values()
            ],
            "acceptance_rate_delta": [
                _delta(rows, "mixing", "acceptance_rate") for rows in by_seed.values()
            ],
            "accepted_barrier_crossing_count_delta": [
                _delta(rows, "basin", "accepted_barrier_crossing_count")
                for rows in by_seed.values()
            ],
        }
        intervals = {
            name: {
                "values": values,
                "mean": _mean(values),
                "mean_95_interval": _interval(values),
            }
            for name, values in deltas.items()
        }
        safety_passed = _intervals_within_equivalence_margins(intervals)
        ess_low = intervals["effective_sample_size_delta"]["mean_95_interval"][0]
        barrier_low = intervals["accepted_barrier_crossing_count_delta"]["mean_95_interval"][0]
        family = metadata[fixture]["family"]
        non_toy = _is_non_toy_family(family)
        value_passed = bool(
            non_toy
            and safety_passed
            and (
                ess_low >= VALUE_GATE["minimum_effective_sample_size_delta"]
                or barrier_low >= VALUE_GATE["minimum_barrier_crossing_delta"]
            )
        )
        fixtures[fixture] = {
            "family": family,
            "target_type": metadata[fixture]["target_type"],
            "non_toy_family": non_toy,
            "paired_seed_count": len(by_seed),
            "retained_samples_per_chain": RETAINED_SAMPLE_COUNT,
            "delta_definition": "mode_jump_runtime minus seeded_fallback",
            "infinite_equal_metric_delta_rule": "equal signed infinities are paired as zero",
            "intervals": intervals,
            "distribution_safety_equivalence_passed": safety_passed,
            "workload_value_improvement_passed": value_passed,
        }
    positive_families = sorted(
        {
            row["family"]
            for row in fixtures.values()
            if row["non_toy_family"] and row["workload_value_improvement_passed"]
        }
    )
    return {
        "fixtures": fixtures,
        "equivalence_margins": dict(EQUIVALENCE_MARGINS),
        "sample_sizes": {
            fixture: {
                "paired_seed_count": row["paired_seed_count"],
                "retained_samples_per_chain": row["retained_samples_per_chain"],
            }
            for fixture, row in fixtures.items()
        },
        "value_gate": {
            **dict(VALUE_GATE),
            "families_with_preregistered_positive_mixing_improvement": positive_families,
            "non_toy_positive_mixing_family_count": len(positive_families),
            "workload_value_gate_passed": len(positive_families)
            >= VALUE_GATE["required_non_toy_families_with_positive_mixing"],
        },
        "cost_conclusion": "descriptive_only",
        "principle": FIELD_PRINCIPLES[
            "paired_intervals_equivalence_margins_and_sample_sizes"
        ],
    }


def _delta(rows: Mapping[str, Any], group: str, metric: str) -> float:
    fallback = float(rows[group]["seeded_fallback"][metric])
    runtime = float(rows[group]["mode_jump_runtime"][metric])
    if not math.isfinite(fallback) or not math.isfinite(runtime):
        if fallback == runtime:
            return 0.0
    return _stable_float(runtime - fallback)


def harmful_regressions(artifact: Mapping[str, Any]) -> list[JsonDict]:
    return exp6269.harmful_regressions(artifact)


def descriptive_wall_time_by_arm_fixture(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    row = exp6269.descriptive_wall_time_by_arm_fixture(cells)
    row["principle"] = FIELD_PRINCIPLES["descriptive_wall_time_by_arm_fixture"]
    return row


def _fixture_metadata(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(row["fixture_name"]): {
            "family": row["family"],
            "target_type": row["target_type"],
            "mode_jump_typed_supported": row["mode_jump_typed_supported"],
        }
        for row in artifact["preregistered_fixture_seed_arm_matrix"]["fixtures"]
    }


def family_level_safety_results(artifact: Mapping[str, Any]) -> JsonDict:
    paired = dict(artifact["paired_intervals_equivalence_margins_and_sample_sizes"]["fixtures"])
    families: dict[str, JsonDict] = {}
    for fixture, row in paired.items():
        family = str(row["family"])
        bucket = families.setdefault(
            family,
            {
                "fixtures": [],
                "fixture_count": 0,
                "passed_fixture_count": 0,
                "paired_seed_count": 0,
                "non_toy_family": _is_non_toy_family(family),
            },
        )
        passed = row["distribution_safety_equivalence_passed"] is True
        bucket["fixtures"].append(fixture)
        bucket["fixture_count"] += 1
        bucket["passed_fixture_count"] += 1 if passed else 0
        bucket["paired_seed_count"] += int(row["paired_seed_count"])
    for row in families.values():
        row["distribution_safety_equivalence_passed"] = (
            row["fixture_count"] > 0 and row["fixture_count"] == row["passed_fixture_count"]
        )
    return {
        "families": dict(sorted(families.items())),
        "all_family_safety_passed": bool(families)
        and all(row["distribution_safety_equivalence_passed"] for row in families.values()),
        "principle": FIELD_PRINCIPLES["family_level_safety_results"],
    }


def family_level_mixing_value_results(artifact: Mapping[str, Any]) -> JsonDict:
    paired = dict(artifact["paired_intervals_equivalence_margins_and_sample_sizes"]["fixtures"])
    families: dict[str, JsonDict] = {}
    for fixture, row in paired.items():
        family = str(row["family"])
        intervals = dict(row["intervals"])
        bucket = families.setdefault(
            family,
            {
                "fixtures": [],
                "fixture_count": 0,
                "non_toy_family": _is_non_toy_family(family),
                "positive_fixture_count": 0,
                "effective_sample_size_delta_means": [],
                "accepted_barrier_crossing_delta_means": [],
            },
        )
        bucket["fixtures"].append(fixture)
        bucket["fixture_count"] += 1
        bucket["positive_fixture_count"] += (
            1 if row["workload_value_improvement_passed"] is True else 0
        )
        bucket["effective_sample_size_delta_means"].append(
            float(intervals["effective_sample_size_delta"]["mean"])
        )
        bucket["accepted_barrier_crossing_delta_means"].append(
            float(intervals["accepted_barrier_crossing_count_delta"]["mean"])
        )
    positive_families = []
    for family, row in families.items():
        row["mean_effective_sample_size_delta"] = _mean(
            row.pop("effective_sample_size_delta_means")
        )
        row["mean_accepted_barrier_crossing_delta"] = _mean(
            row.pop("accepted_barrier_crossing_delta_means")
        )
        row["workload_value_improvement_passed"] = bool(
            row["non_toy_family"] and row["positive_fixture_count"] > 0
        )
        if row["workload_value_improvement_passed"]:
            positive_families.append(family)
    positive_families = sorted(positive_families)
    return {
        "families": dict(sorted(families.items())),
        "non_toy_positive_mixing_families": positive_families,
        "non_toy_positive_mixing_family_count": len(positive_families),
        "required_non_toy_families_with_positive_mixing": VALUE_GATE[
            "required_non_toy_families_with_positive_mixing"
        ],
        "workload_value_gate_passed": len(positive_families)
        >= VALUE_GATE["required_non_toy_families_with_positive_mixing"],
        "unimodal_control_excluded_from_value": True,
        "principle": FIELD_PRINCIPLES["family_level_mixing_value_results"],
    }


def unsupported_or_failed_cells(
    cells: Sequence[Mapping[str, Any]],
    unsupported_receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    rows = [
        {
            "fixture": cell["fixture"],
            "family": cell["family"],
            "target_type": cell["target_type"],
            "seed": cell["seed"],
            "arm": cell["arm"],
            "classification": "matched_cell_failure",
            "error_type": cell.get("error_type"),
            "message": cell.get("message"),
            "fail_closed": True,
            "fallback_output_substituted": False,
            "sample_hash_recorded": False,
        }
        for cell in cells
        if cell.get("success") is not True
    ]
    for receipt in unsupported_receipts:
        for seed in SEEDS:
            for arm in ARMS:
                rows.append(
                    {
                        "fixture": str(receipt["fixture_name"]),
                        "family": receipt["family"],
                        "target_type": receipt["target_type"],
                        "seed": int(seed),
                        "arm": arm,
                        "classification": "not_supported_by_exp6280_typed_backend",
                        "error_type": None,
                        "message": "fixture not preregistered as supported by Exp6280",
                        "fail_closed": True,
                        "fallback_output_substituted": False,
                        "sample_hash_recorded": False,
                    }
                )
    return rows


def protected_files_unchanged(
    *,
    root: Path,
    protected_before: Mapping[str, Any],
) -> JsonDict:
    after = _path_hashes(PROTECTED_FILES, root)
    changed = [path for path in protected_before if protected_before[path] != after.get(path)]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "before": dict(protected_before),
        "after": after,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def verifier_is_oracle() -> JsonDict:
    return {
        "value": True,
        "oracle": "Exp6268 exact finite distributions and typed sampler receipts",
        "sampler_output_used_as_oracle": False,
        "not_oracle_for": ["hardware", "speedup", "cDLS", "power"],
        "principle": FIELD_PRINCIPLES["verifier_is_oracle"],
    }


def _commands_valid(artifact: Mapping[str, Any]) -> bool:
    codes = dict(artifact.get("test_exit_codes") or {})
    return set(DEFAULT_TEST_COMMANDS) <= set(codes) and all(
        codes[command] == 0 for command in DEFAULT_TEST_COMMANDS
    )


def mode_jump_safety_ready_score(artifact: Mapping[str, Any]) -> float:
    positive = artifact.get("positive_inactive_and_unimodal_control_results", {}).get(
        "positive_control",
        {},
    )
    gates = [
        artifact.get("preconditions_checked", {}).get("preconditions_ready") is True,
        positive.get("passed") is True,
        artifact.get("rust_pyo3_backend_receipts", {}).get(
            "all_successful_treatment_cells_used_rust_pyo3"
        )
        is True,
        artifact.get("family_level_safety_results", {}).get("all_family_safety_passed")
        is True,
        not artifact.get("harmful_regressions"),
        not artifact.get("unsupported_or_failed_cells"),
        artifact.get("source_mutation_count") == 0
        and type(artifact.get("source_mutation_count")) is int,
        artifact.get("hardware_claim_count") == 0
        and type(artifact.get("hardware_claim_count")) is int,
        artifact.get("timing_speedup_claimed") is False,
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        artifact.get("verifier_is_oracle", {}).get("value") is True,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        _commands_valid(artifact),
    ]
    return 1.0 if all(gates) else 0.0


def mode_jump_workload_value_ready_score(artifact: Mapping[str, Any]) -> float:
    return (
        1.0
        if mode_jump_safety_ready_score(artifact) == 1.0
        and artifact.get("family_level_mixing_value_results", {}).get(
            "workload_value_gate_passed"
        )
        is True
        else 0.0
    )


def retire_mechanism_recommendation(artifact: Mapping[str, Any]) -> JsonDict:
    safety = mode_jump_safety_ready_score(artifact)
    value = mode_jump_workload_value_ready_score(artifact)
    if safety == 1.0 and value == 0.0:
        recommendation = "permanent_retirement_recommended"
        reason = "Full typed support still produced no family-level workload value."
    elif safety == 1.0 and value == 1.0:
        recommendation = "continue_only_with_new_value_gate"
        reason = "Workload value gate passed, so retirement is not recommended."
    else:
        recommendation = "blocked_no_retirement_recommendation"
        reason = "Safety or activation did not pass, so the retirement test is not final."
    return {
        "recommendation": recommendation,
        "retire_if_same_verdict_satisfied": recommendation
        == "permanent_retirement_recommended",
        "blocked_or_null_after_full_shape_support": safety == 1.0 and value == 0.0,
        "reason": reason,
        "principle": FIELD_PRINCIPLES["retire_mechanism_recommendation"],
    }


def status(artifact: Mapping[str, Any]) -> str:
    positive = artifact.get("positive_inactive_and_unimodal_control_results", {}).get(
        "positive_control",
        {},
    )
    if positive.get("passed") is not True:
        return "instrument_failure"
    if mode_jump_safety_ready_score(artifact) != 1.0:
        return "blocked_safety"
    if mode_jump_workload_value_ready_score(artifact) == 1.0:
        return "complete_workload_value_supported"
    return "retired_value_not_ready"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    family_value = dict(artifact.get("family_level_mixing_value_results") or {})
    recommendation = dict(artifact.get("retire_mechanism_recommendation") or {})
    matrix = dict(artifact.get("preregistered_fixture_seed_arm_matrix") or {})
    return (
        f"{status(artifact)}: "
        f"safety={artifact.get('mode_jump_safety_ready_score')}; "
        f"workload_value={artifact.get('mode_jump_workload_value_ready_score')}; "
        f"supported_fixtures={len(matrix.get('supported_fixtures', []))}; "
        f"unsupported_or_failed_cells={len(artifact.get('unsupported_or_failed_cells', []))}; "
        f"non_toy_positive_mixing_families="
        f"{family_value.get('non_toy_positive_mixing_family_count', 0)}; "
        f"retirement={recommendation.get('recommendation')}; "
        f"hardware_claim_count={artifact.get('hardware_claim_count')}; "
        f"timing_speedup_claimed={artifact.get('timing_speedup_claimed')}"
    )


def field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "source": _field_source(field),
            "spec_refs": [
                "REQ-SAMPLER-6281",
                "SCENARIO-SAMPLER-6281-TYPED-MATCHED-CELLS",
                "SCENARIO-SAMPLER-6281-CONTROLS-SEPARATE-VALUE",
                "SCENARIO-SAMPLER-6281-RETIREMENT-DECISION",
            ],
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _field_source(field: str) -> str:
    chain_fields = {
        "rust_pyo3_backend_receipts",
        "treatment_attempt_accept_and_fire_counts_by_fixture",
        "positive_inactive_and_unimodal_control_results",
        "chain_sample_hashes",
        "exact_distribution_error_by_arm_fixture",
        "energy_error_by_arm_fixture",
        "basin_occupancy_and_barrier_crossings_by_arm_fixture",
        "autocorrelation_ess_and_acceptance_by_arm_fixture",
        "paired_intervals_equivalence_margins_and_sample_sizes",
        "family_level_safety_results",
        "family_level_mixing_value_results",
        "harmful_regressions",
        "descriptive_wall_time_by_arm_fixture",
        "unsupported_or_failed_cells",
    }
    if field in chain_fields:
        return "matched_typed_sampler_chain_evidence"
    if field == "upstream_backend_path_hash_and_terminal_class":
        return "Exp6280 backend readiness artifact"
    if field == "frozen_fixture_path_and_hash":
        return "Exp6268 exact fixture artifact"
    if field in {"test_commands", "test_exit_codes"}:
        return "external command receipts"
    if field in {"source_mutation_count", "protected_files_unchanged"}:
        return "local path hashes"
    return "prompt_spec_and_builder"


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    source_before = _path_hashes(SOURCE_PATHS, root)
    protected_before = _path_hashes(PROTECTED_FILES, root)
    matrix = preregistered_fixture_seed_arm_matrix(root)
    preconditions = preconditions_checked(
        root=root,
        matrix=matrix,
        source_before=source_before,
        protected_before=protected_before,
    )
    supported = _supported_receipts(root)
    unsupported = _unsupported_receipts(root)
    raw_cells = _measure_supported_cells(supported)
    backend_receipts = rust_pyo3_backend_receipts(raw_cells)
    counts = treatment_attempt_accept_and_fire_counts_by_fixture(backend_receipts)
    partial: JsonDict = {
        "preregistered_fixture_seed_arm_matrix": matrix,
        "treatment_attempt_accept_and_fire_counts_by_fixture": counts,
    }
    controls = positive_inactive_and_unimodal_control_results(partial)
    cells = _attach_outcome_metrics(raw_cells, supported)
    source_after = _path_hashes(SOURCE_PATHS, root)
    artifact: JsonDict = {
        "status": "pending",
        "upstream_backend_path_hash_and_terminal_class": (
            upstream_backend_path_hash_and_terminal_class(root)
        ),
        "frozen_fixture_path_and_hash": frozen_fixture_path_and_hash(root),
        "preregistered_fixture_seed_arm_matrix": matrix,
        "matched_arm_configuration": matched_arm_configuration(matrix),
        "rust_pyo3_backend_receipts": backend_receipts,
        "treatment_attempt_accept_and_fire_counts_by_fixture": counts,
        "positive_inactive_and_unimodal_control_results": controls,
        "chain_sample_hashes": chain_sample_hashes(cells),
        "exact_distribution_error_by_arm_fixture": exact_distribution_error_by_arm_fixture(
            cells
        ),
        "energy_error_by_arm_fixture": energy_error_by_arm_fixture(cells),
        "basin_occupancy_and_barrier_crossings_by_arm_fixture": (
            basin_occupancy_and_barrier_crossings_by_arm_fixture(cells)
        ),
        "autocorrelation_ess_and_acceptance_by_arm_fixture": (
            autocorrelation_ess_and_acceptance_by_arm_fixture(cells)
        ),
        "paired_intervals_equivalence_margins_and_sample_sizes": {},
        "family_level_safety_results": {},
        "family_level_mixing_value_results": {},
        "harmful_regressions": [],
        "descriptive_wall_time_by_arm_fixture": descriptive_wall_time_by_arm_fixture(cells),
        "unsupported_or_failed_cells": unsupported_or_failed_cells(cells, unsupported),
        "source_mutation_count": sum(
            1 for path in source_before if source_before[path] != source_after.get(path)
        ),
        "hardware_claim_count": 0,
        "timing_speedup_claimed": False,
        "mode_jump_safety_ready_score": 0.0,
        "mode_jump_workload_value_ready_score": 0.0,
        "retire_mechanism_recommendation": {},
        "protected_files_unchanged": protected_files_unchanged(
            root=root,
            protected_before=protected_before,
        ),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _normalize_test_exit_codes(test_exit_codes or {}),
        "duration_s": _stable_float(duration_s),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": "pending",
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
    }
    artifact = recompute_terminal_fields(artifact)
    validate_artifact(artifact)
    return artifact


def recompute_terminal_fields(artifact: Mapping[str, Any]) -> JsonDict:
    updated = _json_copy(artifact)
    updated["positive_inactive_and_unimodal_control_results"] = (
        positive_inactive_and_unimodal_control_results(updated)
    )
    updated["paired_intervals_equivalence_margins_and_sample_sizes"] = (
        paired_intervals_equivalence_margins_and_sample_sizes(updated)
    )
    updated["family_level_safety_results"] = family_level_safety_results(updated)
    updated["family_level_mixing_value_results"] = family_level_mixing_value_results(updated)
    updated["harmful_regressions"] = harmful_regressions(updated)
    updated["mode_jump_safety_ready_score"] = mode_jump_safety_ready_score(updated)
    updated["mode_jump_workload_value_ready_score"] = mode_jump_workload_value_ready_score(
        updated
    )
    updated["retire_mechanism_recommendation"] = retire_mechanism_recommendation(updated)
    updated["status"] = status(updated)
    updated["honest_verdict"] = honest_verdict(updated)
    updated["reproducibility_checksum"] = reproducibility_checksum(updated)
    return updated


def write_artifact(
    *,
    output_path: Path | None = None,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    started = time.perf_counter()
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=0.0 if duration_s is None else duration_s,
        test_exit_codes=(
            test_exit_codes if test_exit_codes is not None else _external_test_exit_codes()
        ),
    )
    if duration_s is None:
        artifact["duration_s"] = _stable_float(time.perf_counter() - started)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        validate_artifact(artifact)
    output = output_path or root / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _json_copy(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be object")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if provenance.get(field, {}).get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field_provenance:{field}")
    if type(artifact["source_mutation_count"]) is not int or artifact["source_mutation_count"] != 0:
        raise ValueError("source_mutation_count must be bare 0")
    if type(artifact["hardware_claim_count"]) is not int or artifact["hardware_claim_count"] != 0:
        raise ValueError("hardware_claim_count must be bare 0")
    if artifact["timing_speedup_claimed"] is not False:
        raise ValueError("timing_speedup_claimed must be false")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact["verifier_is_oracle"].get("value") is not True:
        raise ValueError("verifier_is_oracle")
    _validate_matched_seeds(artifact)
    _validate_sample_counts(artifact)
    _validate_acceptance_accounting(artifact)
    expected_paired = paired_intervals_equivalence_margins_and_sample_sizes(artifact)
    if artifact["paired_intervals_equivalence_margins_and_sample_sizes"] != expected_paired:
        raise ValueError("paired_intervals_equivalence_margins_and_sample_sizes")
    expected_family_safety = family_level_safety_results(artifact)
    if artifact["family_level_safety_results"] != expected_family_safety:
        raise ValueError("family_level_safety_results")
    expected_family_value = family_level_mixing_value_results(artifact)
    if artifact["family_level_mixing_value_results"] != expected_family_value:
        raise ValueError("family_level_mixing_value_results")
    expected_harm = harmful_regressions(artifact)
    if artifact["harmful_regressions"] != expected_harm:
        raise ValueError("harmful_regressions")
    if artifact["mode_jump_safety_ready_score"] != mode_jump_safety_ready_score(artifact):
        raise ValueError("mode_jump_safety_ready_score")
    if artifact["mode_jump_workload_value_ready_score"] != mode_jump_workload_value_ready_score(
        artifact
    ):
        raise ValueError("mode_jump_workload_value_ready_score")
    expected_retire = retire_mechanism_recommendation(artifact)
    if artifact["retire_mechanism_recommendation"] != expected_retire:
        raise ValueError("retire_mechanism_recommendation")
    if artifact["status"] != status(artifact):
        raise ValueError("status")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _validate_matched_seeds(artifact: Mapping[str, Any]) -> None:
    expected_seeds = list(SEEDS)
    if artifact["matched_arm_configuration"]["matched_seeds"] != expected_seeds:
        raise ValueError("seed mismatch in matched_arm_configuration")
    expected = {
        (fixture, arm, seed)
        for fixture in artifact["preregistered_fixture_seed_arm_matrix"]["supported_fixtures"]
        for arm in ARMS
        for seed in expected_seeds
    }
    observed = {
        (str(row["fixture"]), str(row["arm"]), int(row["seed"]))
        for row in artifact["rust_pyo3_backend_receipts"]["chains"]
        if row["arm"] in ARMS
    }
    if observed != expected:
        raise ValueError("seed mismatch in chain receipts")


def _validate_sample_counts(artifact: Mapping[str, Any]) -> None:
    counts = {
        (row["fixture"], int(row["seed"]), row["arm"]): int(row["sample_count"])
        for row in artifact["chain_sample_hashes"]["chains"]
    }
    distribution = artifact["exact_distribution_error_by_arm_fixture"]["fixtures"]
    for fixture, arms in distribution.items():
        for arm, rows in arms.items():
            for chain in rows["chains"]:
                key = (fixture, int(chain["seed"]), arm)
                if counts.get(key) != int(chain["sample_count"]):
                    raise ValueError("sample-count mismatch")


def _validate_acceptance_accounting(artifact: Mapping[str, Any]) -> None:
    expected: dict[tuple[str, str], JsonDict] = {}
    for row in artifact["rust_pyo3_backend_receipts"]["chains"]:
        key = (row["fixture"], row["arm"])
        bucket = expected.setdefault(
            key,
            {
                "attempted_count": 0,
                "accepted_count": 0,
                "treatment_attempt_count": 0,
                "treatment_accept_count": 0,
                "treatment_fire_count": 0,
            },
        )
        for field in bucket:
            bucket[field] += int(row[field])
    observed = artifact["treatment_attempt_accept_and_fire_counts_by_fixture"]["fixtures"]
    for (fixture, arm), fields in expected.items():
        row = observed[fixture][arm]
        for field, value in fields.items():
            if int(row[field]) != int(value):
                raise ValueError("acceptance accounting mismatch")


def _normalize_test_exit_codes(test_exit_codes: Mapping[str, int]) -> dict[str, int | None]:
    return {
        command: int(test_exit_codes[command]) if command in test_exit_codes else None
        for command in DEFAULT_TEST_COMMANDS
    }


def _external_test_exit_codes() -> dict[str, int]:
    import os

    path = Path(str(os.environ.get("CARNOT_6281_COMMAND_RECEIPTS", DEFAULT_RECEIPT_PATH)))
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("command receipt payload must be an object")
    return {str(command): int(code) for command, code in payload.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = write_artifact(output_path=args.output, root=REPO_ROOT, run_date=str(args.date))
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "path": args.output.as_posix(),
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0 if not str(artifact["status"]).startswith("blocked") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
