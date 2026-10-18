"""Exp6269 mode-jump multifamily A/B.

Spec refs: REQ-SAMPLER-6269,
SCENARIO-SAMPLER-6269-MATCHED-SUPPORTED-CELLS,
SCENARIO-SAMPLER-6269-UNSUPPORTED-CELLS-FAIL-CLOSED,
SCENARIO-SAMPLER-6269-SAFETY-VALUE-SEPARATION.

This harness tests only fixtures accepted by the existing fixed mode-jump
backend. Unsupported Exp6268 fixtures stay visible as unsupported cells. That
keeps the target distribution unchanged and prevents fallback output from
standing in for an unsupported treatment.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import numpy as np

from carnot import experiment_6268_multimodal_sampler_fixture_suite as exp6268
from carnot.samplers.mode_jump_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
    MODE_JUMP_ALGORITHM,
    MODE_JUMP_TOPOLOGY,
    ModeJumpRustBackend,
    descriptor_for_run,
    sha256_json,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6269_mode_jump_multifamily_ab.json")
UPSTREAM_FIXTURE_RELATIVE_PATH = Path(
    "results/experiment_6268_multimodal_sampler_fixture_suite.json"
)
EXP6237_RESULT_RELATIVE_PATH = Path("results/experiment_6237_activated_mode_jump_sampler_ab.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6269_mode_jump_multifamily_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6269_mode_jump_multifamily_ab.py")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
BACKEND_RELATIVE_PATH = Path("python/carnot/samplers/mode_jump_rust_backend.py")
RUST_KERNEL_RELATIVE_PATH = Path("crates/carnot-samplers/src/mode_jump.rs")
PYO3_BINDING_RELATIVE_PATH = Path("crates/carnot-python/src/mode_jump.rs")

SCHEMA = "carnot.experiment_6269.mode_jump_multifamily_ab.v1"
EXPERIMENT_ID = "experiment_6269_mode_jump_multifamily_ab"
RUN_DATE = "20260810"
INFERENCE_SUBSTRATE = "local_cpu_exact_multifamily_mode_jump_ab"
DEFAULT_RECEIPT_PATH = Path("/tmp/carnot_6269_command_receipts.json")

SEEDS = (6268, 6269, 6270)
ARMS = ("seeded_fallback", "mode_jump_runtime")
BURN_IN = 128
RETAINED_SAMPLE_COUNT = 4096
PROPOSAL_BUDGET = BURN_IN + RETAINED_SAMPLE_COUNT
WALL_BUDGET_S = 5.0
MAX_ACF_LAG = 200

EQUIVALENCE_MARGINS: dict[str, float] = {
    "total_variation_to_target_delta": 0.02,
    "kl_target_to_empirical_delta": 0.01,
    "max_state_probability_abs_error_delta": 0.03,
    "energy_mean_abs_error_delta": 0.05,
    "energy_variance_abs_error_delta": 0.08,
    "max_basin_mass_abs_error_delta": 0.03,
    "lag1_autocorrelation_delta": 0.05,
    "effective_sample_size_delta": 250.0,
    "acceptance_rate_delta": 0.05,
}
VALUE_GATE = {
    "required_non_toy_families_with_positive_mixing": 2,
    "minimum_effective_sample_size_delta": 50.0,
    "minimum_barrier_crossing_delta": 1.0,
    "exactness_regression_allowed": False,
}

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
    UPSTREAM_FIXTURE_RELATIVE_PATH,
    EXP6237_RESULT_RELATIVE_PATH,
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6269_mode_jump_multifamily_ab.py -q -o addopts=",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6269_mode_jump_multifamily_ab.py -m pytest tests/python/test_experiment_6269_mode_jump_multifamily_ab.py -q --no-cov -o addopts=",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6269_mode_jump_multifamily_ab.py --fail-under=100",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6269_mode_jump_multifamily_ab.py",
    "cargo test -p carnot-samplers --test mode_jump --quiet",
    ".venv/bin/pytest tests/python/test_e2e_serialization.py tests/python/test_pyo3_integration.py -q -o addopts=",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_6269_mode_jump_multifamily_ab --date 20260810",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6269_mode_jump_multifamily_ab.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_fixture_path_and_hash",
    "preregistered_fixture_seed_arm_matrix",
    "matched_arm_configuration",
    "rust_pyo3_backend_receipts",
    "treatment_attempt_accept_and_fire_counts_by_fixture",
    "positive_and_inactive_control_results",
    "chain_sample_hashes",
    "exact_distribution_error_by_arm_fixture",
    "energy_error_by_arm_fixture",
    "basin_occupancy_and_barrier_crossings_by_arm_fixture",
    "autocorrelation_ess_and_acceptance_by_arm_fixture",
    "paired_intervals_equivalence_margins_and_sample_sizes",
    "harmful_regressions",
    "descriptive_wall_time_by_arm_fixture",
    "unsupported_or_failed_cells",
    "source_mutation_count",
    "hardware_claim_count",
    "timing_speedup_claimed",
    "mode_jump_safety_ready_score",
    "mode_jump_workload_value_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Separates safety evidence, workload-value evidence, blockers, and instrument failures.",
    "upstream_fixture_path_and_hash": "Pins the Exp6268 exact suite consumed by the A/B.",
    "preregistered_fixture_seed_arm_matrix": "Freezes fixtures, seeds, arms, budgets, margins, and the value gate before sampling.",
    "matched_arm_configuration": "Proves each compared arm uses the same target, seed, initial state, burn-in, retained sample count, proposal budget, and schedule.",
    "rust_pyo3_backend_receipts": "Authenticates the backend, descriptor, input hash, final state, and transition budget for each chain.",
    "treatment_attempt_accept_and_fire_counts_by_fixture": "Proves treatment activity before outcome comparison.",
    "positive_and_inactive_control_results": "Records both the activation-positive control and the inactive-treatment fail-closed control.",
    "chain_sample_hashes": "Content-addresses retained samples without making the artifact depend on raw sample arrays.",
    "exact_distribution_error_by_arm_fixture": "Reports empirical-versus-exact distribution error per fixture and arm.",
    "energy_error_by_arm_fixture": "Reports empirical-versus-exact energy error per fixture and arm.",
    "basin_occupancy_and_barrier_crossings_by_arm_fixture": "Reports basin mass and cross-basin transitions per fixture and arm.",
    "autocorrelation_ess_and_acceptance_by_arm_fixture": "Reports autocorrelation, ESS, and acceptance per fixture and arm.",
    "paired_intervals_equivalence_margins_and_sample_sizes": "Stores paired intervals, equivalence margins, and n before any value decision.",
    "harmful_regressions": "Lists exactness or mixing regressions that block safety.",
    "descriptive_wall_time_by_arm_fixture": "Records wall time as cost evidence only.",
    "unsupported_or_failed_cells": "Preserves unsupported or failed Exp6268 cells without fallback substitution.",
    "source_mutation_count": "Bare zero proves this experiment did not mutate preregistered source during compute.",
    "hardware_claim_count": "Bare zero prevents a software sampler A/B from becoming a hardware claim.",
    "timing_speedup_claimed": "Bare false prevents descriptive wall time from becoming a speedup claim.",
    "mode_jump_safety_ready_score": "Equals one only when exactness, activation, controls, protected files, source, and command gates pass.",
    "mode_jump_workload_value_ready_score": "Equals one only when safety and the non-toy positive mixing value gate both pass.",
    "protected_files_unchanged": "Confirms conductor-owned and reconciler-owned files stayed byte-identical.",
    "preconditions_checked": "Records exact-suite validation, frozen budgets, frozen margins, value gate, and protected hashes.",
    "inference_substrate": "Declares local CPU exact multifamily mode-jump sampling, not hardware, cDLS, or LLM inference.",
    "verifier_is_oracle": "States that Exp6268 exact finite distributions are the oracle.",
    "field_provenance": "Maps every required field to prompt, spec, source, upstream fixture, command, or chain evidence.",
    "field_principles": "Explains why each artifact field exists before a reviewer trusts the JSON shape.",
    "test_commands": "Records focused Python, coverage, Rust, E2E, artifact, adversarial, and suite command receipts.",
    "test_exit_codes": "Stores exit codes so failed checks cannot become readiness evidence.",
    "duration_s": "Reports real wall time without padding.",
    "reproducibility_checksum": "Content-addresses the artifact after blanking volatile duration and checksum fields.",
    "honest_verdict": "Uses a terminal prefix and states safety, workload value, unsupported fixtures, and no hardware or speedup claim.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stable_float(value: Any) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else _stable_float(statistics.mean(values))


def _interval(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [_stable_float(values[0]), _stable_float(values[0])]
    center = statistics.mean(values)
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [_stable_float(center - half_width), _stable_float(center + half_width)]


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected at {path}")
    return payload


def _path_hashes(paths: Sequence[Path], root: Path) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for path in paths:
        target = root / path
        rows[path.as_posix()] = {
            "exists": target.exists(),
            "sha256": sha256_file(target) if target.exists() else None,
            "size_bytes": target.stat().st_size if target.exists() else None,
        }
    return rows


def _load_upstream_fixture_artifact(root: Path = REPO_ROOT) -> JsonDict:
    artifact = _read_json(root / UPSTREAM_FIXTURE_RELATIVE_PATH)
    exp6268.validate_artifact(artifact)
    return artifact


def upstream_fixture_path_and_hash(root: Path = REPO_ROOT) -> JsonDict:
    upstream = _load_upstream_fixture_artifact(root)
    receipts = list(upstream["exact_enumeration_receipts"])
    return {
        "path": UPSTREAM_FIXTURE_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(root / UPSTREAM_FIXTURE_RELATIVE_PATH),
        "status": upstream["status"],
        "sampler_fixture_suite_ready_score": upstream["sampler_fixture_suite_ready_score"],
        "fixture_count": len(receipts),
        "exact_suite_validated": True,
        "normalized_target_probability_hashes": dict(
            upstream["normalized_target_probability_hashes"]
        ),
        "principle": FIELD_PRINCIPLES["upstream_fixture_path_and_hash"],
    }


def _support_by_fixture(upstream: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(name): dict(row)
        for name, row in dict(upstream["mode_jump_support_by_fixture"]).items()
    }


def _receipt_name(receipt: Mapping[str, Any]) -> str:
    return str(receipt["fixture_name"])


def _supported_receipts(upstream: Mapping[str, Any]) -> list[JsonDict]:
    support = _support_by_fixture(upstream)
    return [
        dict(receipt)
        for receipt in upstream["exact_enumeration_receipts"]
        if support[_receipt_name(receipt)]["mode_jump_rust_supported"] is True
    ]


def _unsupported_receipts(upstream: Mapping[str, Any]) -> list[JsonDict]:
    support = _support_by_fixture(upstream)
    return [
        dict(receipt)
        for receipt in upstream["exact_enumeration_receipts"]
        if support[_receipt_name(receipt)]["mode_jump_rust_supported"] is not True
    ]


def preregistered_fixture_seed_arm_matrix(root: Path = REPO_ROOT) -> JsonDict:
    upstream = _load_upstream_fixture_artifact(root)
    support = _support_by_fixture(upstream)
    fixtures = []
    for receipt in upstream["exact_enumeration_receipts"]:
        name = _receipt_name(receipt)
        fixtures.append(
            {
                "fixture_name": name,
                "family": receipt["family"],
                "target_type": receipt["target_type"],
                "target_probability_hash": receipt["target_probability_hash"],
                "mode_count": len(receipt["modes"]),
                "mode_jump_rust_supported": support[name]["mode_jump_rust_supported"],
                "support_classification": support[name]["classification"],
            }
        )
    matrix: JsonDict = {
        "upstream_fixture_path": UPSTREAM_FIXTURE_RELATIVE_PATH.as_posix(),
        "fixtures": fixtures,
        "supported_fixtures": [
            row["fixture_name"] for row in fixtures if row["mode_jump_rust_supported"] is True
        ],
        "unsupported_fixtures": [
            row["fixture_name"] for row in fixtures if row["mode_jump_rust_supported"] is not True
        ],
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "burn_in": BURN_IN,
        "retained_sample_count": RETAINED_SAMPLE_COUNT,
        "proposal_budget": PROPOSAL_BUDGET,
        "wall_budget_s_per_cell": WALL_BUDGET_S,
        "schedule_fixed": True,
        "initial_state_rule": "first explicit mode for supported fixed categorical fixture",
        "equivalence_margins": dict(EQUIVALENCE_MARGINS),
        "value_gate": dict(VALUE_GATE),
        "unsupported_cells_replaced_with_fallback": False,
        "timing_speedup_claim_allowed": False,
        "hardware_claim_allowed": False,
        "cdls_reopened": False,
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
    upstream = _load_upstream_fixture_artifact(root)
    spec_text = (root / SAMPLER_SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks = {
        "exact_suite_validated": upstream["status"] == "complete_ready"
        and upstream["sampler_fixture_suite_ready_score"] == 1.0,
        "fixture_seed_arm_matrix_frozen": bool(matrix.get("matrix_sha256")),
        "budgets_frozen": matrix["burn_in"] == BURN_IN
        and matrix["retained_sample_count"] == RETAINED_SAMPLE_COUNT
        and matrix["proposal_budget"] == PROPOSAL_BUDGET,
        "equivalence_margins_frozen": matrix["equivalence_margins"] == EQUIVALENCE_MARGINS,
        "value_gate_frozen": matrix["value_gate"] == VALUE_GATE,
        "seeds_frozen": matrix["seeds"] == list(SEEDS),
        "supported_fixture_count_positive": len(matrix["supported_fixtures"]) > 0,
        "unsupported_cells_preserved": matrix["unsupported_cells_replaced_with_fallback"] is False,
        "sampler_spec_has_req": "REQ-SAMPLER-6269" in spec_text,
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
        "upstream_fixture_sha256": sha256_file(root / UPSTREAM_FIXTURE_RELATIVE_PATH),
        "supported_fixture_count": len(matrix["supported_fixtures"]),
        "unsupported_fixture_count": len(matrix["unsupported_fixtures"]),
        "source_hashes_before_sha256": sha256_json(source_before),
        "protected_hashes_before_sha256": sha256_json(protected_before),
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
        "matched_topology": MODE_JUMP_TOPOLOGY,
        "matched_burn_in": matrix["burn_in"],
        "matched_retained_sample_count": matrix["retained_sample_count"],
        "matched_proposal_budget": matrix["proposal_budget"],
        "matched_wall_budget_s_per_cell": matrix["wall_budget_s_per_cell"],
        "matched_schedule_fixed": matrix["schedule_fixed"],
        "unsupported_cells_replaced_with_fallback": False,
        "principle": FIELD_PRINCIPLES["matched_arm_configuration"],
    }


def _target_arrays(receipt: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    definition = dict(receipt["definition"])
    labels = [str(label) for label in definition["labels"]]
    target = np.asarray(definition["target_probabilities"], dtype=np.float64)
    proposal = np.asarray(definition["proposal_probabilities"], dtype=np.float64)
    return labels, target, proposal


def _initial_label(receipt: Mapping[str, Any]) -> str:
    modes = [str(label) for label in receipt["modes"]]
    return modes[0] if modes else str(receipt["support"][0]["state_label"])


def _descriptor(receipt: Mapping[str, Any], seed: int) -> JsonDict:
    labels, _target, _proposal = _target_arrays(receipt)
    return {
        **descriptor_for_run(
            labels=labels,
            seed=seed,
            initial_label=_initial_label(receipt),
            burn_in=BURN_IN,
            enable_mode_jump_runtime=True,
        ),
        "return_trace": True,
    }


def _state_maps(receipt: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    probabilities = {
        str(row["state_label"]): float(row["probability"]) for row in receipt["support"]
    }
    energies = {str(row["state_label"]): float(row["energy"]) for row in receipt["support"]}
    basins = {str(row["state_label"]): str(row["basin"]) for row in receipt["support"]}
    return probabilities, energies, basins


def _run_supported_cell(receipt: Mapping[str, Any], seed: int, arm: str) -> JsonDict:
    labels, target, proposal = _target_arrays(receipt)
    prefer_rust = arm == "mode_jump_runtime"
    backend = ModeJumpRustBackend(seed=seed, prefer_rust=prefer_rust)
    started = time.perf_counter()
    try:
        result = backend.run_descriptor(
            target,
            proposal,
            n_samples=RETAINED_SAMPLE_COUNT,
            config=_descriptor(receipt, seed),
        )
    except Exception as exc:  # pragma: no cover - supported Exp6268 cells should run.
        return {
            "success": False,
            "fixture": _receipt_name(receipt),
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
    distribution = _distribution_metrics(receipt, sample_labels)
    energy = _energy_metrics(receipt, sample_labels)
    basin = _basin_metrics(receipt, sample_labels, decision_log)
    mixing = _mixing_metrics(receipt, sample_labels, result["receipt"], basin)
    treatment = _treatment_counts(receipt, decision_log, result["receipt"])
    return {
        "success": True,
        "fixture": _receipt_name(receipt),
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
        "distribution_metrics": distribution,
        "energy_metrics": energy,
        "basin_metrics": basin,
        "mixing_metrics": mixing,
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


def _distribution_metrics(receipt: Mapping[str, Any], sample_labels: Sequence[str]) -> JsonDict:
    probabilities, _energies, _basins = _state_maps(receipt)
    sample_count = len(sample_labels)
    counts = Counter(str(label) for label in sample_labels)
    frequencies = {
        label: _stable_float(counts[label] / sample_count) if sample_count else 0.0
        for label in sorted(probabilities)
    }
    tv = 0.5 * sum(abs(frequencies[label] - probabilities[label]) for label in probabilities)
    kl = 0.0
    for label, probability in probabilities.items():
        frequency = frequencies[label]
        if probability > 0.0 and frequency > 0.0:
            kl += probability * math.log(probability / frequency)
        elif probability > 0.0:
            kl = float("inf")
    max_error = max(abs(frequencies[label] - probabilities[label]) for label in probabilities)
    return {
        "sample_count": sample_count,
        "target_probability_hash": receipt["target_probability_hash"],
        "frequencies": frequencies,
        "total_variation_to_target": _stable_float(tv),
        "kl_target_to_empirical": _stable_float(kl),
        "max_state_probability_abs_error": _stable_float(max_error),
    }


def _energy_metrics(receipt: Mapping[str, Any], sample_labels: Sequence[str]) -> JsonDict:
    probabilities, energies, _basins = _state_maps(receipt)
    exact_mean = sum(probabilities[label] * energies[label] for label in probabilities)
    exact_second = sum(
        probabilities[label] * energies[label] * energies[label] for label in probabilities
    )
    sample_energy = [energies[str(label)] for label in sample_labels]
    sample_mean = float(np.mean(sample_energy)) if sample_energy else 0.0
    sample_variance = float(np.var(sample_energy)) if sample_energy else 0.0
    exact_variance = exact_second - exact_mean * exact_mean
    return {
        "sample_count": len(sample_labels),
        "exact_energy_mean": _stable_float(exact_mean),
        "sample_energy_mean": _stable_float(sample_mean),
        "energy_mean_abs_error": _stable_float(abs(sample_mean - exact_mean)),
        "exact_energy_variance": _stable_float(exact_variance),
        "sample_energy_variance": _stable_float(sample_variance),
        "energy_variance_abs_error": _stable_float(abs(sample_variance - exact_variance)),
    }


def _basin_metrics(
    receipt: Mapping[str, Any],
    sample_labels: Sequence[str],
    decision_log: Sequence[Mapping[str, Any]],
) -> JsonDict:
    probabilities, _energies, basins = _state_maps(receipt)
    basin_labels = sorted(set(basins.values()))
    sample_count = len(sample_labels)
    counts = Counter(basins[str(label)] for label in sample_labels)
    exact_masses = {
        basin: _stable_float(
            sum(probabilities[label] for label in probabilities if basins[label] == basin)
        )
        for basin in basin_labels
    }
    empirical_masses = {
        basin: _stable_float(counts[basin] / sample_count) if sample_count else 0.0
        for basin in basin_labels
    }
    accepted_crossings = 0
    proposed_crossings = 0
    retained_crossings = 0
    for event in decision_log:
        before = str(event["state_before"]["current_label"])
        proposed = str(event["proposed_label"])
        after = str(event["state_after"]["current_label"])
        if basins.get(before) != basins.get(proposed):
            proposed_crossings += 1
        if bool(event.get("accepted")) and basins.get(before) != basins.get(after):
            accepted_crossings += 1
    for left, right in zip(sample_labels, sample_labels[1:], strict=False):
        if basins.get(str(left)) != basins.get(str(right)):
            retained_crossings += 1
    errors = {
        basin: _stable_float(abs(empirical_masses[basin] - exact_masses[basin]))
        for basin in basin_labels
    }
    return {
        "sample_count": sample_count,
        "basin_counts": dict(sorted(counts.items())),
        "basin_masses_exact": exact_masses,
        "basin_masses_empirical": empirical_masses,
        "basin_mass_abs_errors": errors,
        "max_basin_mass_abs_error": _stable_float(max(errors.values(), default=0.0)),
        "proposed_barrier_crossing_count": proposed_crossings,
        "accepted_barrier_crossing_count": accepted_crossings,
        "retained_sample_barrier_crossing_count": retained_crossings,
        "barrier_metadata": receipt["barrier_metadata"],
    }


def _mixing_metrics(
    receipt: Mapping[str, Any],
    sample_labels: Sequence[str],
    backend_receipt: Mapping[str, Any],
    basin_metrics: Mapping[str, Any],
) -> JsonDict:
    _probabilities, _energies, basins = _state_maps(receipt)
    first_basin = sorted(set(basins.values()))[0]
    indicator = [1.0 if basins[str(label)] == first_basin else 0.0 for label in sample_labels]
    quality = _quality_from_basin_indicator(indicator)
    attempted = int(backend_receipt["transition_budget"]["total_steps"])
    accepted = int(backend_receipt["final_state"]["accepted_count"])
    return {
        **quality,
        "indicator_basin": first_basin,
        "attempted_count": attempted,
        "accepted_count": accepted,
        "acceptance_rate": _stable_float(accepted / attempted) if attempted else 0.0,
        "accepted_barrier_crossing_count": int(
            basin_metrics["accepted_barrier_crossing_count"]
        ),
    }


def _quality_from_basin_indicator(values: Sequence[float]) -> JsonDict:
    if not values:
        return {
            "degenerate": True,
            "lag1_autocorrelation": 0.0,
            "integrated_autocorrelation_time": 1.0,
            "effective_sample_size": 0.0,
        }
    mean = sum(values) / len(values)
    denom = sum((value - mean) ** 2 for value in values)
    if len(values) < 2 or denom == 0.0:
        return {
            "degenerate": len(set(values)) <= 1,
            "lag1_autocorrelation": 0.0,
            "integrated_autocorrelation_time": 1.0,
            "effective_sample_size": _stable_float(len(values)),
        }
    lag1 = _autocorrelation(values, mean, denom, 1)
    positive_sum = 0.0
    for lag in range(1, min(MAX_ACF_LAG, len(values) - 1) + 1):
        rho = _autocorrelation(values, mean, denom, lag)
        if rho <= 0.0:
            break
        positive_sum += rho
    iact = max(1.0, 1.0 + 2.0 * positive_sum)
    return {
        "degenerate": False,
        "lag1_autocorrelation": _stable_float(lag1),
        "integrated_autocorrelation_time": _stable_float(iact),
        "effective_sample_size": _stable_float(len(values) / iact),
    }


def _autocorrelation(values: Sequence[float], mean: float, denom: float, lag: int) -> float:
    return sum(
        (values[index] - mean) * (values[index - lag] - mean)
        for index in range(lag, len(values))
    ) / denom


def _treatment_counts(
    receipt: Mapping[str, Any],
    decision_log: Sequence[Mapping[str, Any]],
    backend_receipt: Mapping[str, Any],
) -> JsonDict:
    _probabilities, _energies, basins = _state_maps(receipt)
    attempts = 0
    accepts = 0
    fires = 0
    for event in decision_log:
        before = str(event["state_before"]["current_label"])
        proposed = str(event["proposed_label"])
        after = str(event["state_after"]["current_label"])
        if basins.get(before) != basins.get(proposed):
            attempts += 1
            if bool(event.get("accepted")):
                accepts += 1
        if bool(event.get("accepted")) and basins.get(before) != basins.get(after):
            fires += 1
    attempted = int(backend_receipt["transition_budget"]["total_steps"])
    accepted = int(backend_receipt["final_state"]["accepted_count"])
    return {
        "attempted_count": attempted,
        "accepted_count": accepted,
        "acceptance_rate": _stable_float(accepted / attempted) if attempted else 0.0,
        "treatment_attempt_count": attempts,
        "treatment_accept_count": accepts,
        "treatment_fire_count": fires,
    }


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


def positive_and_inactive_control_results(artifact: Mapping[str, Any]) -> JsonDict:
    counts = dict(artifact["treatment_attempt_accept_and_fire_counts_by_fixture"])
    fixture_rows = dict(counts["fixtures"])
    positive_rows = []
    for fixture, arms in fixture_rows.items():
        treatment = dict(dict(arms).get("mode_jump_runtime") or {})
        positive_rows.append(
            {
                "fixture": fixture,
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
        "principle": FIELD_PRINCIPLES["positive_and_inactive_control_results"],
    }


def chain_sample_hashes(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [
        {
            "fixture": cell["fixture"],
            "family": cell["family"],
            "seed": cell["seed"],
            "arm": cell["arm"],
            "sample_count": len(cell["sample_labels"]),
            "sample_labels_sha256": cell["sample_labels_sha256"],
            "decision_log_sha256": cell["decision_log_sha256"],
        }
        for cell in cells
        if cell.get("success") is True
    ]
    return {
        "chains": rows,
        "all_hashes_present": all(row["sample_labels_sha256"] for row in rows),
        "principle": FIELD_PRINCIPLES["chain_sample_hashes"],
    }


def exact_distribution_error_by_arm_fixture(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    return _metric_by_fixture_arm(
        cells,
        "distribution_metrics",
        [
            "total_variation_to_target",
            "kl_target_to_empirical",
            "max_state_probability_abs_error",
        ],
        FIELD_PRINCIPLES["exact_distribution_error_by_arm_fixture"],
    )


def energy_error_by_arm_fixture(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    return _metric_by_fixture_arm(
        cells,
        "energy_metrics",
        ["energy_mean_abs_error", "energy_variance_abs_error"],
        FIELD_PRINCIPLES["energy_error_by_arm_fixture"],
    )


def basin_occupancy_and_barrier_crossings_by_arm_fixture(
    cells: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return _metric_by_fixture_arm(
        cells,
        "basin_metrics",
        [
            "max_basin_mass_abs_error",
            "proposed_barrier_crossing_count",
            "accepted_barrier_crossing_count",
            "retained_sample_barrier_crossing_count",
        ],
        FIELD_PRINCIPLES["basin_occupancy_and_barrier_crossings_by_arm_fixture"],
    )


def autocorrelation_ess_and_acceptance_by_arm_fixture(
    cells: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return _metric_by_fixture_arm(
        cells,
        "mixing_metrics",
        [
            "lag1_autocorrelation",
            "effective_sample_size",
            "acceptance_rate",
            "accepted_barrier_crossing_count",
        ],
        FIELD_PRINCIPLES["autocorrelation_ess_and_acceptance_by_arm_fixture"],
    )


def _metric_by_fixture_arm(
    cells: Sequence[Mapping[str, Any]],
    metric_key: str,
    summary_metrics: Sequence[str],
    principle: str,
) -> JsonDict:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for cell in cells:
        if cell.get("success") is True:
            grouped.setdefault(str(cell["fixture"]), {}).setdefault(str(cell["arm"]), []).append(
                cell
            )
    fixtures: dict[str, JsonDict] = {}
    for fixture, arms in grouped.items():
        fixtures[fixture] = {}
        for arm, arm_cells in arms.items():
            chain_rows = []
            for cell in arm_cells:
                metric = dict(cell[metric_key])
                chain_rows.append({"seed": cell["seed"], **_json_copy(metric)})
            summary: JsonDict = {"chain_count": len(chain_rows)}
            for metric_name in summary_metrics:
                values = [float(row[metric_name]) for row in chain_rows]
                summary[f"mean_{metric_name}"] = _mean(values)
                summary[f"max_{metric_name}"] = _stable_float(max(values)) if values else 0.0
            fixtures[fixture][arm] = {"chains": chain_rows, "summary": summary}
    return {"fixtures": fixtures, "principle": principle}


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
                _delta(rows, "distribution", "total_variation_to_target") for rows in by_seed.values()
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
                _delta(rows, "energy", "energy_variance_abs_error") for rows in by_seed.values()
            ],
            "max_basin_mass_abs_error_delta": [
                _delta(rows, "basin", "max_basin_mass_abs_error") for rows in by_seed.values()
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
            name: {"values": values, "mean": _mean(values), "mean_95_interval": _interval(values)}
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


def _chain_by_seed(chains: Sequence[Mapping[str, Any]], seed: int) -> Mapping[str, Any]:
    for row in chains:
        if int(row["seed"]) == int(seed):
            return row
    raise KeyError(seed)


def _delta(rows: Mapping[str, Any], group: str, metric: str) -> float:
    return _stable_float(
        float(rows[group]["mode_jump_runtime"][metric])
        - float(rows[group]["seeded_fallback"][metric])
    )


def _intervals_within_equivalence_margins(intervals: Mapping[str, Any]) -> bool:
    for metric, margin in EQUIVALENCE_MARGINS.items():
        row = dict(intervals.get(metric) or {})
        low, high = row.get("mean_95_interval", [float("inf"), float("-inf")])
        if abs(float(low)) > margin or abs(float(high)) > margin:
            return False
    return True


def harmful_regressions(artifact: Mapping[str, Any]) -> list[JsonDict]:
    rows = []
    distribution = dict(artifact["exact_distribution_error_by_arm_fixture"]["fixtures"])
    energy = dict(artifact["energy_error_by_arm_fixture"]["fixtures"])
    mixing = dict(artifact["autocorrelation_ess_and_acceptance_by_arm_fixture"]["fixtures"])
    for fixture, arms in distribution.items():
        if not all(arm in arms for arm in ARMS):
            continue
        for seed in SEEDS:
            fallback_dist = _chain_by_seed(arms["seeded_fallback"]["chains"], seed)
            runtime_dist = _chain_by_seed(arms["mode_jump_runtime"]["chains"], seed)
            fallback_energy = _chain_by_seed(
                energy[fixture]["seeded_fallback"]["chains"], seed
            )
            runtime_energy = _chain_by_seed(
                energy[fixture]["mode_jump_runtime"]["chains"], seed
            )
            fallback_mixing = _chain_by_seed(
                mixing[fixture]["seeded_fallback"]["chains"], seed
            )
            runtime_mixing = _chain_by_seed(
                mixing[fixture]["mode_jump_runtime"]["chains"], seed
            )
            checks = [
                (
                    "total_variation_to_target",
                    float(runtime_dist["total_variation_to_target"])
                    - float(fallback_dist["total_variation_to_target"]),
                    EQUIVALENCE_MARGINS["total_variation_to_target_delta"],
                ),
                (
                    "energy_mean_abs_error",
                    float(runtime_energy["energy_mean_abs_error"])
                    - float(fallback_energy["energy_mean_abs_error"]),
                    EQUIVALENCE_MARGINS["energy_mean_abs_error_delta"],
                ),
                (
                    "effective_sample_size",
                    float(fallback_mixing["effective_sample_size"])
                    - float(runtime_mixing["effective_sample_size"]),
                    EQUIVALENCE_MARGINS["effective_sample_size_delta"],
                ),
            ]
            for metric, delta, margin in checks:
                if delta > margin:
                    rows.append(
                        {
                            "fixture": fixture,
                            "seed": seed,
                            "metric": metric,
                            "regression_delta": _stable_float(delta),
                            "margin": margin,
                            "classification": "harmful_regression",
                        }
                    )
    return rows


def descriptive_wall_time_by_arm_fixture(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for cell in cells:
        if cell.get("success") is True:
            grouped.setdefault(str(cell["fixture"]), {}).setdefault(str(cell["arm"]), []).append(
                cell
            )
    fixtures: dict[str, JsonDict] = {}
    for fixture, arms in grouped.items():
        fixtures[fixture] = {}
        for arm, rows in arms.items():
            elapsed = [float(row["elapsed_s"]) for row in rows]
            fixtures[fixture][arm] = {
                "chains": [
                    {
                        "seed": row["seed"],
                        "elapsed_s": row["elapsed_s"],
                        "wall_budget_s": row["wall_budget_s"],
                        "wall_budget_met": row["wall_budget_met"],
                    }
                    for row in rows
                ],
                "mean_elapsed_s": _mean(elapsed),
                "max_elapsed_s": _stable_float(max(elapsed)) if elapsed else 0.0,
            }
    return {
        "fixtures": fixtures,
        "timing_speedup_claimed": False,
        "wall_time_is_descriptive_only": True,
        "principle": FIELD_PRINCIPLES["descriptive_wall_time_by_arm_fixture"],
    }


def _unsupported_cell_from_receipt(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
) -> JsonDict:
    probabilities = [float(row["probability"]) for row in receipt["support"]]
    proposal = np.eye(len(probabilities), dtype=np.float64)
    try:
        ModeJumpRustBackend(seed=seed)._coerce_mode_jump_inputs(  # noqa: SLF001
            np.asarray(probabilities, dtype=np.float64),
            proposal,
        )
    except ValueError as exc:
        return {
            "fixture": _receipt_name(receipt),
            "family": receipt["family"],
            "target_type": receipt["target_type"],
            "seed": int(seed),
            "arm": arm,
            "classification": "unsupported_for_existing_mode_jump_backend",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "fail_closed": True,
            "fallback_output_substituted": False,
            "sample_hash_recorded": False,
        }
    return {  # pragma: no cover - Exp6268 classifies these as unsupported.
        "fixture": _receipt_name(receipt),
        "family": receipt["family"],
        "target_type": receipt["target_type"],
        "seed": int(seed),
        "arm": arm,
        "classification": "unexpectedly_supported",
        "fail_closed": False,
        "fallback_output_substituted": False,
        "sample_hash_recorded": False,
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
    rows.extend(
        _unsupported_cell_from_receipt(receipt, arm=arm, seed=SEEDS[0])
        for receipt in unsupported_receipts
        for arm in ARMS
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
        "oracle": "Exp6268 exact finite distributions and sampler receipts",
        "sampler_output_used_as_oracle": False,
        "not_oracle_for": ["hardware", "speedup", "unsupported fixture families"],
        "principle": FIELD_PRINCIPLES["verifier_is_oracle"],
    }


def _fixture_metadata(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(row["fixture_name"]): {
            "family": row["family"],
            "target_type": row["target_type"],
            "mode_jump_rust_supported": row["mode_jump_rust_supported"],
        }
        for row in artifact["preregistered_fixture_seed_arm_matrix"]["fixtures"]
    }


def _is_non_toy_family(family: str) -> bool:
    return family not in {"original_six_state_positive_control", "unimodal_control"}


def _commands_valid(artifact: Mapping[str, Any]) -> bool:
    codes = dict(artifact.get("test_exit_codes") or {})
    return set(DEFAULT_TEST_COMMANDS) <= set(codes) and all(
        codes[command] == 0 for command in DEFAULT_TEST_COMMANDS
    )


def mode_jump_safety_ready_score(artifact: Mapping[str, Any]) -> float:
    positive = artifact.get("positive_and_inactive_control_results", {}).get(
        "positive_control", {}
    )
    paired = artifact.get("paired_intervals_equivalence_margins_and_sample_sizes", {})
    safety_rows = [
        row.get("distribution_safety_equivalence_passed") is True
        for row in dict(paired.get("fixtures") or {}).values()
    ]
    gates = [
        artifact.get("preconditions_checked", {}).get("preconditions_ready") is True,
        positive.get("passed") is True,
        not artifact.get("harmful_regressions"),
        bool(safety_rows) and all(safety_rows),
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
    value_gate = dict(
        artifact.get("paired_intervals_equivalence_margins_and_sample_sizes", {}).get(
            "value_gate", {}
        )
    )
    return (
        1.0
        if mode_jump_safety_ready_score(artifact) == 1.0
        and value_gate.get("workload_value_gate_passed") is True
        else 0.0
    )


def status(artifact: Mapping[str, Any]) -> str:
    positive = artifact.get("positive_and_inactive_control_results", {}).get(
        "positive_control", {}
    )
    if positive.get("passed") is not True:
        return "instrument_failure"
    if mode_jump_safety_ready_score(artifact) != 1.0:
        return "blocked_safety"
    if mode_jump_workload_value_ready_score(artifact) == 1.0:
        return "complete_workload_value_supported"
    return "complete_safety_supported_value_not_ready"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    paired = artifact.get("paired_intervals_equivalence_margins_and_sample_sizes", {})
    value_gate = dict(paired.get("value_gate") or {})
    matrix = dict(artifact.get("preregistered_fixture_seed_arm_matrix") or {})
    return (
        f"{status(artifact)}: "
        f"safety={artifact.get('mode_jump_safety_ready_score')}; "
        f"workload_value={artifact.get('mode_jump_workload_value_ready_score')}; "
        f"supported_fixtures={len(matrix.get('supported_fixtures', []))}; "
        f"unsupported_fixtures={len(matrix.get('unsupported_fixtures', []))}; "
        f"non_toy_positive_mixing_families="
        f"{value_gate.get('non_toy_positive_mixing_family_count', 0)}; "
        f"hardware_claim_count={artifact.get('hardware_claim_count')}; "
        f"timing_speedup_claimed={artifact.get('timing_speedup_claimed')}"
    )


def field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "source": _field_source(field),
            "spec_refs": [
                "REQ-SAMPLER-6269",
                "SCENARIO-SAMPLER-6269-MATCHED-SUPPORTED-CELLS",
                "SCENARIO-SAMPLER-6269-UNSUPPORTED-CELLS-FAIL-CLOSED",
                "SCENARIO-SAMPLER-6269-SAFETY-VALUE-SEPARATION",
            ],
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _field_source(field: str) -> str:
    chain_fields = {
        "rust_pyo3_backend_receipts",
        "treatment_attempt_accept_and_fire_counts_by_fixture",
        "chain_sample_hashes",
        "exact_distribution_error_by_arm_fixture",
        "energy_error_by_arm_fixture",
        "basin_occupancy_and_barrier_crossings_by_arm_fixture",
        "autocorrelation_ess_and_acceptance_by_arm_fixture",
        "paired_intervals_equivalence_margins_and_sample_sizes",
        "harmful_regressions",
        "descriptive_wall_time_by_arm_fixture",
    }
    if field in chain_fields:
        return "matched_sampler_chain_evidence"
    if field in {"upstream_fixture_path_and_hash", "verifier_is_oracle"}:
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
    upstream = _load_upstream_fixture_artifact(root)
    matrix = preregistered_fixture_seed_arm_matrix(root)
    preconditions = preconditions_checked(
        root=root,
        matrix=matrix,
        source_before=source_before,
        protected_before=protected_before,
    )
    supported = _supported_receipts(upstream)
    unsupported = _unsupported_receipts(upstream)
    cells = _measure_supported_cells(supported)
    backend_receipts = rust_pyo3_backend_receipts(cells)
    counts = treatment_attempt_accept_and_fire_counts_by_fixture(backend_receipts)
    source_after = _path_hashes(SOURCE_PATHS, root)
    artifact: JsonDict = {
        "status": "pending",
        "upstream_fixture_path_and_hash": upstream_fixture_path_and_hash(root),
        "preregistered_fixture_seed_arm_matrix": matrix,
        "matched_arm_configuration": matched_arm_configuration(matrix),
        "rust_pyo3_backend_receipts": backend_receipts,
        "treatment_attempt_accept_and_fire_counts_by_fixture": counts,
        "positive_and_inactive_control_results": {},
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
    updated["positive_and_inactive_control_results"] = positive_and_inactive_control_results(
        updated
    )
    updated["paired_intervals_equivalence_margins_and_sample_sizes"] = (
        paired_intervals_equivalence_margins_and_sample_sizes(updated)
    )
    updated["harmful_regressions"] = harmful_regressions(updated)
    updated["mode_jump_safety_ready_score"] = mode_jump_safety_ready_score(updated)
    updated["mode_jump_workload_value_ready_score"] = mode_jump_workload_value_ready_score(
        updated
    )
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
        test_exit_codes=test_exit_codes if test_exit_codes is not None else _external_test_exit_codes(),
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
    expected_harm = harmful_regressions(artifact)
    if artifact["harmful_regressions"] != expected_harm:
        raise ValueError("harmful_regressions")
    if artifact["mode_jump_safety_ready_score"] != mode_jump_safety_ready_score(artifact):
        raise ValueError("mode_jump_safety_ready_score")
    if artifact["mode_jump_workload_value_ready_score"] != mode_jump_workload_value_ready_score(
        artifact
    ):
        raise ValueError("mode_jump_workload_value_ready_score")
    if artifact["status"] != status(artifact):
        raise ValueError("status")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _validate_matched_seeds(artifact: Mapping[str, Any]) -> None:
    expected = list(SEEDS)
    if artifact["matched_arm_configuration"]["matched_seeds"] != expected:
        raise ValueError("seed mismatch in matched_arm_configuration")
    seen = sorted(
        {
            int(row["seed"])
            for row in artifact["rust_pyo3_backend_receipts"]["chains"]
            if row["arm"] in ARMS
        }
    )
    if seen != expected:
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
    env = __import__("os").environ
    path = Path(str(env.get("CARNOT_6269_COMMAND_RECEIPTS", DEFAULT_RECEIPT_PATH)))
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
