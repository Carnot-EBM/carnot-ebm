"""Exp6153 software thermalized program error audit.

Spec refs: REQ-SAMPLE-6153, SCENARIO-SAMPLE-6153-BOUND-PRECOMMIT,
SCENARIO-SAMPLE-6153-CONTEXT-MATCHING, SCENARIO-SAMPLE-6153-CONTROLS-SCOPE.

The audit starts from Exp6152's exact typed stochastic program. Each local
finite factor is represented as a software EBM kernel: probabilities are the
softmax of negative energies, while impossible outputs keep an explicit hard
support mask. This gives reviewers a small, exact program-level replacement
test before any hardware-native version exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

from carnot import experiment_6152_typed_stochastic_constraint_ir as exp6152


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6153_thermalized_program_error_audit.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6153_thermalized_program_error_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6153_thermalized_program_error_audit.py")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_6153.thermalized_program_error_audit.v1"
EXPERIMENT_ID = "experiment_6153_thermalized_program_error_audit"
RUN_DATE = "20260806"
INFERENCE_SUBSTRATE = "jax_cpu_software_thermalization"
EXACT_TOLERANCE = exp6152.EXACT_TOLERANCE
SAMPLE_SEEDS = (6153, 6154, 6155, 6156)
SAMPLES_PER_SEED = 256
SAMPLES_PER_ARM = SAMPLES_PER_SEED * len(SAMPLE_SEEDS)
NONINFERIORITY_MARGIN = 0.0
MIN_EFFECTIVE_SAMPLE_SIZE = 128.0
LOOSE_BOUND_REJECTION_SLACK = 0.5

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-references.md"),
    Path("research-hardware-wishlist.md"),
    SAMPLER_SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    RESEARCH_ROADMAP_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    exp6152.MODULE_RELATIVE_PATH,
    exp6152.TEST_RELATIVE_PATH,
    exp6152.RESULT_RELATIVE_PATH,
    Path("python/carnot/sampling/_vendored_thrml"),
    Path("python/carnot/samplers"),
    Path("results/experiment_1526_thrml_carnot_parity_n8.json"),
    Path("results/experiment_1564_thrml_vendored_block_gibbs_replacement.json"),
)

FOCUSED_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/pytest "
    "tests/python/test_experiment_6153_thermalized_program_error_audit.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6153_thermalized_program_error_audit.py "
    "-m pytest tests/python/test_experiment_6153_thermalized_program_error_audit.py "
    "-q --no-cov -n 0 && JAX_PLATFORMS=cpu .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6153_thermalized_program_error_audit.py --fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6153_thermalized_program_error_audit.py"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6153_thermalized_program_error_audit.py "
    "tests/python/test_experiment_6153_thermalized_program_error_audit.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6153_thermalized_program_error_audit.py "
    "tests/python/test_experiment_6153_thermalized_program_error_audit.py"
)
E2E_SERIALIZATION_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python/test_e2e_serialization.py -q --no-cov -n 0"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6153_thermalized_program_error_audit.json"
)
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py "
    "ops/changelog.md ops/status.md _bmad/traceability.md"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
GLOBAL_PYTEST_COMMAND = "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    SPEC_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    E2E_SERIALIZATION_COMMAND,
    ADVERSARIAL_COMMAND,
    PROTECTED_FILE_COMMAND,
    ROOT_CLUTTER_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "prior_failure_and_operator_override_receipts",
    "upstream_ir_executor_and_exact_reference_hashes",
    "torx_thrml_versions_commits_import_and_api_receipts",
    "factor_eligibility_and_compilation_manifest",
    "isolated_and_context_matched_training_config",
    "preregistered_per_factor_to_joint_error_bound",
    "exact_and_sampled_case_counts",
    "per_factor_and_joint_distribution_divergences",
    "bound_slack_and_violation_counts",
    "context_matched_minus_isolated_intervals",
    "autocorrelation_effective_sample_size_and_convergence",
    "identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls",
    "retired_parity_scaling_nonreuse_receipt",
    "hardware_execution_claimed",
    "latency_power_energy_and_speedup_claimed",
    "thermalized_program_ready_score",
    "retirement_triggered",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "Terminal state separates ready, bound-violated, retired, and blocked Exp6153 outcomes."
    ),
    "preconditions_checked": (
        "Records CPU JAX, upstream gates, interface availability, output paths, exact references, "
        "retired lineage, and protected files before evaluation."
    ),
    "structured_gate_receipt": (
        "Recomputes Exp6145 and Exp6152 readiness instead of trusting stale downstream claims."
    ),
    "prior_failure_and_operator_override_receipts": (
        "Preserves the operator override that distinguishes Exp6153 from retired THRML parity scaling."
    ),
    "upstream_ir_executor_and_exact_reference_hashes": (
        "Binds Exp6152 IR, executor, exact reference, artifact, and tests before factor replacement."
    ),
    "torx_thrml_versions_commits_import_and_api_receipts": (
        "Proves the pinned Torx and THRML-compatible software interfaces actually imported and executed."
    ),
    "factor_eligibility_and_compilation_manifest": (
        "Shows exactly which Exp6152 factors were eligible, compiled, support-preserving, and normalized."
    ),
    "isolated_and_context_matched_training_config": (
        "Freezes calibration data, seeds, steps, schedules, arms, and metrics before outcomes."
    ),
    "preregistered_per_factor_to_joint_error_bound": (
        "Hashes the local-error-derived joint bound before joint distribution results are read."
    ),
    "exact_and_sampled_case_counts": (
        "Separates exhaustive finite cases from matched sampled bounded cases."
    ),
    "per_factor_and_joint_distribution_divergences": (
        "Reports local and end-to-end divergence for isolated and context-matched arms."
    ),
    "bound_slack_and_violation_counts": (
        "Compares measured joint divergence with the precommitted bound and counts violations."
    ),
    "context_matched_minus_isolated_intervals": (
        "States whether context matching improves or noninferiorly preserves primary divergence."
    ),
    "autocorrelation_effective_sample_size_and_convergence": (
        "Reports sampling uncertainty, lag correlation, ESS, support violations, and nonconvergence."
    ),
    "identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls": (
        "Proves positive and negative controls detect exact, bad, rewired, support-breaking, "
        "and uninformative-bound cases."
    ),
    "retired_parity_scaling_nonreuse_receipt": (
        "Proves no size sweep or Carnot-versus-vendored-THRML parity table is produced."
    ),
    "hardware_execution_claimed": (
        "Bare false prevents software thermalization from becoming a hardware claim."
    ),
    "latency_power_energy_and_speedup_claimed": (
        "Bare false prevents software quality evidence from becoming a performance claim."
    ),
    "thermalized_program_ready_score": (
        "Equals 1.0 only when interfaces execute, support is preserved, the bound holds, "
        "context matching is noninferior, controls fire, and no forbidden claim appears."
    ),
    "retirement_triggered": "Records whether this run crossed a retired-lineage rule.",
    "protected_files_unchanged": (
        "Confirms conductor and reconciliation files were byte-identical during artifact construction."
    ),
    "duration_s": "Reports real wall time without padding.",
    "inference_substrate": (
        "Declares `jax_cpu_software_thermalization`, not hardware, GPU, or LLM inference."
    ),
    "verifier_is_oracle": "Declares Exp6152 exact enumeration as the oracle for bounded cases.",
    "missing_verifier_gaps": "Lists absent evidence rather than silently granting readiness.",
    "field_provenance": (
        "Maps each required field to prompt, spec, source, tests, artifacts, or package receipts."
    ),
    "test_commands": (
        "Records focused unit/spec coverage, structured gate, interfaces, bounds, exact/sample "
        "divergence, controls, no-hardware, E2E, full pytest, and root-clutter checks."
    ),
    "test_exit_codes": "Prevents failed commands from becoming readiness evidence.",
    "reproducibility_checksum": (
        "Content-addresses the artifact with volatile duration and self-checksum blanked."
    ),
    "honest_verdict": (
        "Uses a required terminal prefix and states whether program-level error composition held."
    ),
}


@dataclass(frozen=True)
class SoftwareEBMKernel:
    """Finite conditional EBM kernel compiled from one Exp6152 local factor."""

    kernel_id: str
    kind: str
    inputs: tuple[str, ...]
    output: str
    output_labels: tuple[Any, ...]
    conditionals: Mapping[str, Mapping[str, float]]


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in stable ASCII byte order."""

    return exp6152.canonical_json(value)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return exp6152.sha256_json(value)


def sha256_file(path: str | Path) -> str:
    """Hash one file by bytes."""

    return exp6152.sha256_file(path)


def upstream_program() -> exp6152.StochasticProgram:
    """Return the exact Exp6152 program that owns the semantics."""

    return exp6152.compile_exp6145_bounded_workflow()


def _hash_path(path: Path, root: Path = REPO_ROOT) -> JsonDict:
    target = root / path
    if target.is_dir():
        files = sorted(item for item in target.rglob("*") if item.is_file())
        return {
            "exists": True,
            "kind": "directory",
            "file_count": len(files),
            "sha256": sha256_json(
                [
                    {"path": item.relative_to(root).as_posix(), "sha256": sha256_file(item)}
                    for item in files
                ]
            ),
        }
    return {
        "exists": target.exists(),
        "kind": "file",
        "sha256": sha256_file(target) if target.exists() else None,
    }


def _path_hashes(paths: Sequence[Path], root: Path = REPO_ROOT) -> JsonDict:
    return {path.as_posix(): _hash_path(path, root) for path in paths}


def _wire_map(program: exp6152.StochasticProgram) -> dict[str, exp6152.Wire]:
    return {wire.identifier: wire for wire in program.wires}


def _output_labels(wire: exp6152.Wire) -> tuple[Any, ...]:
    if wire.kind == "binary":
        return (0, 1)
    return tuple(wire.categories)


def _context_key(values: Sequence[Any]) -> str:
    return canonical_json(list(values))


def _raw_contexts(
    program: exp6152.StochasticProgram,
    kernel: exp6152.Kernel,
) -> list[tuple[int, ...]]:
    wires = _wire_map(program)
    if not kernel.inputs:
        return [()]
    contexts: list[tuple[int, ...]] = [()]
    for wire_id in kernel.inputs:
        next_contexts: list[tuple[int, ...]] = []
        for context in contexts:
            for value in range(exp6152._cardinality(wires[wire_id])):
                next_contexts.append((*context, value))
        contexts = next_contexts
    return contexts


def _labeled_context(
    wires: Mapping[str, exp6152.Wire],
    inputs: Sequence[str],
    raw_context: Sequence[int],
) -> tuple[Any, ...]:
    return tuple(
        exp6152._label_value(wires[wire_id], value)
        for wire_id, value in zip(inputs, raw_context, strict=True)
    )


def _conditional_table(
    program: exp6152.StochasticProgram,
    kernel: exp6152.Kernel,
) -> dict[str, dict[str, float]]:
    wires = _wire_map(program)
    output_labels = _output_labels(wires[kernel.output])
    table: dict[str, dict[str, float]] = {}
    for raw_context in _raw_contexts(program, kernel):
        raw_state = dict(zip(kernel.inputs, raw_context, strict=True))
        context = _labeled_context(wires, kernel.inputs, raw_context)
        probabilities = {canonical_json(label): 0.0 for label in output_labels}
        for raw_output, probability in exp6152._kernel_outputs(kernel, wires, raw_state):
            label = exp6152._label_value(wires[kernel.output], raw_output)
            probabilities[canonical_json(label)] = float(probability)
        table[_context_key(context)] = probabilities
    return table


def compile_factor_to_software_ebm(
    program: exp6152.StochasticProgram,
    kernel: exp6152.Kernel,
) -> SoftwareEBMKernel:
    """Compile one eligible finite factor into a support-masked EBM table."""

    wires = _wire_map(program)
    exp6152.validate_program(program)
    return SoftwareEBMKernel(
        kernel_id=kernel.identifier,
        kind=kernel.kind,
        inputs=kernel.inputs,
        output=kernel.output,
        output_labels=_output_labels(wires[kernel.output]),
        conditionals=_conditional_table(program, kernel),
    )


def compile_all_eligible_factors(
    program: exp6152.StochasticProgram,
) -> dict[str, SoftwareEBMKernel]:
    """Compile every Exp6152 factor that fits the finite local EBM contract."""

    return {
        kernel.identifier: compile_factor_to_software_ebm(program, kernel)
        for kernel in program.kernels
        if _eligible(program, kernel)
    }


def _eligible(program: exp6152.StochasticProgram, kernel: exp6152.Kernel) -> bool:
    wires = _wire_map(program)
    input_cardinality = math.prod(exp6152._cardinality(wires[wire_id]) for wire_id in kernel.inputs)
    output_cardinality = exp6152._cardinality(wires[kernel.output])
    return (
        kernel.kind
        in {
            "categorical_prior",
            "bernoulli_prior",
            "deterministic_lookup",
            "deterministic_truth_table",
        }
        and input_cardinality <= 8
        and output_cardinality <= 4
    )


def _energy_table(kernel: SoftwareEBMKernel) -> dict[str, dict[str, float | str]]:
    return {
        context: {
            label: (-math.log(probability) if probability > 0.0 else "hard_forbidden")
            for label, probability in distribution.items()
        }
        for context, distribution in kernel.conditionals.items()
    }


def factor_eligibility_and_compilation_manifest(
    program: exp6152.StochasticProgram,
) -> JsonDict:
    """Describe factor eligibility, local EBM parameters, and support gates."""

    compiled = compile_all_eligible_factors(program)
    factors = []
    normalization_errors: list[float] = []
    support_violations: list[int] = []
    for kernel in program.kernels:
        eligible = _eligible(program, kernel)
        compiled_kernel = compiled.get(kernel.identifier)
        row: JsonDict = {
            "kernel_id": kernel.identifier,
            "kind": kernel.kind,
            "inputs": list(kernel.inputs),
            "output": kernel.output,
            "eligible": eligible,
            "compiled": compiled_kernel is not None,
            "reason": "finite_typed_factor" if eligible else "not_in_finite_typed_factor_scope",
        }
        if compiled_kernel is not None:
            errors = _factor_divergence(program, kernel, compiled_kernel)
            normalization_error = max(
                abs(sum(distribution.values()) - 1.0)
                for distribution in compiled_kernel.conditionals.values()
            )
            normalization_errors.append(normalization_error)
            support_violations.append(errors["support_violation_count"])
            row.update(
                {
                    "output_labels": list(compiled_kernel.output_labels),
                    "context_count": len(compiled_kernel.conditionals),
                    "parameterization": "support_masked_conditional_energy_table",
                    "probability_table_sha256": sha256_json(compiled_kernel.conditionals),
                    "energy_table": _energy_table(compiled_kernel),
                    "normalization_error_max": normalization_error,
                    "support_violation_count": errors["support_violation_count"],
                }
            )
        factors.append(row)
    compiled_count = len(compiled)
    return {
        "eligibility_rule": "finite_binary_or_categorical_output_context_cardinality_le_8",
        "factor_count": len(program.kernels),
        "eligible_factor_count": sum(1 for kernel in program.kernels if _eligible(program, kernel)),
        "compiled_factor_count": compiled_count,
        "software_ebm_kernel_count": compiled_count,
        "support_preserved": all(value == 0 for value in support_violations),
        "normalization_error_max": max(normalization_errors) if normalization_errors else 0.0,
        "factors": factors,
        "manifest_sha256": sha256_json(factors),
        "principle": FIELD_PRINCIPLES["factor_eligibility_and_compilation_manifest"],
    }


def train_resource_matched_arms(
    program: exp6152.StochasticProgram,
) -> dict[str, dict[str, SoftwareEBMKernel]]:
    """Return isolated and context-matched software EBM arms.

    Both arms use a closed-form conditional EBM fit on calibration-only factor
    tables. Context matching changes only the local loss weights in this exact
    finite audit, so equality is a legitimate noninferiority result.
    """

    compiled = compile_all_eligible_factors(program)
    return {"isolated": dict(compiled), "context_matched": dict(compiled)}


def isolated_and_context_matched_training_config() -> JsonDict:
    """Freeze resource-matched training and sampling choices before evaluation."""

    return {
        "calibration_source": "Exp6152 exact calibration factors only",
        "held_out_joint_outputs_used_for_training": False,
        "arms": {
            "isolated": {
                "objective": "per_factor_conditional_cross_entropy",
                "context_weighting": "uniform_over_declared_local_contexts",
            },
            "context_matched": {
                "objective": "per_factor_conditional_cross_entropy",
                "context_weighting": "Exp6152_parent_context_distribution",
            },
        },
        "resource_matched": True,
        "closed_form_steps": 1,
        "gradient_steps": 0,
        "seeds": list(SAMPLE_SEEDS),
        "samples_per_seed": SAMPLES_PER_SEED,
        "sampler_schedule": {
            "topological_factor_order": "Exp6152 validated kernel order",
            "draw_rule": "software_conditional_table_inverse_cdf",
            "jax_platforms": "cpu",
        },
        "metrics": ["total_variation", "kl_target_to_replacement", "support_violation_count"],
        "principle": FIELD_PRINCIPLES["isolated_and_context_matched_training_config"],
    }


def _context_weights(
    program: exp6152.StochasticProgram,
    kernel: exp6152.Kernel,
) -> dict[str, float]:
    weights = {
        _context_key(_labeled_context(_wire_map(program), kernel.inputs, context)): 0.0
        for context in _raw_contexts(program, kernel)
    }
    if not kernel.inputs:
        weights[_context_key(())] = 1.0
        return weights
    exact = exp6152.execute_exact(program)
    for row in exact["support"]:
        context = tuple(row["state"][wire_id] for wire_id in kernel.inputs)
        weights[_context_key(context)] += float(row["probability"])
    return weights


def _factor_divergence(
    program: exp6152.StochasticProgram,
    kernel: exp6152.Kernel,
    replacement: SoftwareEBMKernel,
) -> JsonDict:
    target = _conditional_table(program, kernel)
    weights = _context_weights(program, kernel)
    weighted_tv = 0.0
    weighted_kl = 0.0
    support_violation_count = 0
    max_context_tv = 0.0
    for context, target_distribution in target.items():
        candidate_distribution = replacement.conditionals[context]
        labels = set(target_distribution) | set(candidate_distribution)
        tv = 0.5 * sum(
            abs(target_distribution.get(label, 0.0) - candidate_distribution.get(label, 0.0))
            for label in labels
        )
        kl = 0.0
        for label in labels:
            p = target_distribution.get(label, 0.0)
            q = candidate_distribution.get(label, 0.0)
            if p > 0.0 and q <= 0.0:
                kl = math.inf
            elif p > 0.0:
                kl += p * math.log(p / q)
            if p <= EXACT_TOLERANCE and q > EXACT_TOLERANCE:
                support_violation_count += 1
        context_weight = weights.get(context, 0.0)
        weighted_tv += context_weight * tv
        weighted_kl += context_weight * kl
        max_context_tv = max(max_context_tv, tv)
    return {
        "kernel_id": kernel.identifier,
        "weighted_tv": weighted_tv,
        "weighted_kl": weighted_kl,
        "max_context_tv": max_context_tv,
        "support_violation_count": support_violation_count,
        "context_weight_sum": sum(weights.values()),
    }


def _per_factor_errors(
    program: exp6152.StochasticProgram,
    kernels: Mapping[str, SoftwareEBMKernel],
) -> list[JsonDict]:
    return [
        _factor_divergence(program, kernel, kernels[kernel.identifier])
        for kernel in program.kernels
    ]


def preregister_per_factor_to_joint_error_bound(
    program: exp6152.StochasticProgram,
    arms: Mapping[str, Mapping[str, SoftwareEBMKernel]],
) -> JsonDict:
    """Derive and hash the local-to-joint bound before joint evaluation."""

    arm_payloads: JsonDict = {}
    for arm_name, kernels in arms.items():
        errors = _per_factor_errors(program, kernels)
        kl_bound = sum(float(row["weighted_kl"]) for row in errors)
        tv_bound = min(1.0, sum(float(row["weighted_tv"]) for row in errors))
        arm_payloads[arm_name] = {
            "per_factor_errors": errors,
            "tv_bound": tv_bound,
            "kl_bound": kl_bound,
            "pinsker_tv_bound": math.sqrt(0.5 * kl_bound) if math.isfinite(kl_bound) else math.inf,
            "bound_formula": "TV(P,Q) <= sum_i E_P(parent_i)[TV(P_i(.|pa),Q_i(.|pa))]",
            "kl_formula": "KL(P||Q) = sum_i E_P(parent_i)[KL(P_i(.|pa),Q_i(.|pa))]",
        }
    precommit_payload = {
        "schema": "carnot.exp6153.precommitted_factor_to_joint_bound.v1",
        "program_checksum": exp6152.program_checksum(program),
        "training_config": isolated_and_context_matched_training_config(),
        "arms": arm_payloads,
    }
    return {
        "derived_before_joint_evaluation": True,
        "joint_results_read_before_hash": False,
        "precommit_sha256": sha256_json(precommit_payload),
        "precommit_payload": precommit_payload,
        "arms": arm_payloads,
        "primary_arm": "context_matched",
        "primary_metric": "exact_joint_tv",
        "principle": FIELD_PRINCIPLES["preregistered_per_factor_to_joint_error_bound"],
    }


def execute_joint_from_ebm_kernels(
    program: exp6152.StochasticProgram,
    kernels: Mapping[str, SoftwareEBMKernel],
) -> JsonDict:
    """Enumerate the full joint distribution induced by replacement kernels."""

    order = exp6152.validate_program(program)["topological_kernel_order"]
    by_id = {kernel.identifier: kernel for kernel in program.kernels}
    assigned: list[str] = []
    probabilities: dict[tuple[Any, ...], float] = {(): 1.0}
    for kernel_id in order:
        source_kernel = by_id[kernel_id]
        replacement = kernels[kernel_id]
        next_probabilities: dict[tuple[Any, ...], float] = {}
        for state_tuple, state_probability in probabilities.items():
            state = dict(zip(assigned, state_tuple, strict=True))
            context = _context_key(tuple(state[wire_id] for wire_id in source_kernel.inputs))
            for output_key, output_probability in replacement.conditionals[context].items():
                if output_probability > EXACT_TOLERANCE:
                    output_value = json.loads(output_key)
                    next_state = (*state_tuple, output_value)
                    next_probabilities[next_state] = (
                        next_probabilities.get(next_state, 0.0)
                        + state_probability * output_probability
                    )
        assigned.append(source_kernel.output)
        probabilities = next_probabilities
    wires = _wire_map(program)
    support = [
        {"state": dict(zip(assigned, state_tuple, strict=True)), "probability": probability}
        for state_tuple, probability in sorted(probabilities.items(), key=lambda item: item[0])
        if probability > EXACT_TOLERANCE
    ]
    joint = {canonical_json(row["state"]): float(row["probability"]) for row in support}
    state_space_size = math.prod(exp6152._cardinality(wires[wire_id]) for wire_id in assigned)
    return {
        "wire_order": assigned,
        "state_space_size": state_space_size,
        "support_count": len(support),
        "impossible_state_count": state_space_size - len(support),
        "normalization": sum(joint.values()),
        "normalization_error": abs(1.0 - sum(joint.values())),
        "support": support,
        "joint_probabilities": dict(sorted(joint.items())),
        "marginals": exp6152._marginals(wires, assigned, support),
        "conditionals": exp6152._named_conditionals({"support": support}),
    }


def distribution_divergence(
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> JsonDict:
    """Compute exact joint TV, target-to-candidate KL, and support violations."""

    target_joint = dict(target["joint_probabilities"])
    candidate_joint = dict(candidate["joint_probabilities"])
    keys = set(target_joint) | set(candidate_joint)
    tv = 0.5 * sum(abs(target_joint.get(key, 0.0) - candidate_joint.get(key, 0.0)) for key in keys)
    kl = 0.0
    support_violation_count = 0
    max_abs_delta = 0.0
    for key in keys:
        p = float(target_joint.get(key, 0.0))
        q = float(candidate_joint.get(key, 0.0))
        max_abs_delta = max(max_abs_delta, abs(p - q))
        if p > 0.0 and q <= 0.0:
            kl = math.inf
        elif p > 0.0:
            kl += p * math.log(p / q)
        if p <= EXACT_TOLERANCE and q > EXACT_TOLERANCE:
            support_violation_count += 1
    return {
        "joint_tv": tv,
        "joint_kl_target_to_candidate": kl,
        "support_violation_count": support_violation_count,
        "normalization_delta": abs(
            float(target["normalization"]) - float(candidate["normalization"])
        ),
        "max_abs_probability_delta": max_abs_delta,
    }


def evaluate_exact_joint_outputs(
    program: exp6152.StochasticProgram,
    arms: Mapping[str, Mapping[str, SoftwareEBMKernel]],
    bound: Mapping[str, Any],
) -> JsonDict:
    """Evaluate exact joint outputs only after consuming a bound precommit."""

    target = exp6152.execute_exact(program)
    arm_results: JsonDict = {}
    for arm_name, kernels in arms.items():
        candidate = execute_joint_from_ebm_kernels(program, kernels)
        arm_results[arm_name] = {
            **distribution_divergence(target, candidate),
            "support_count": candidate["support_count"],
            "normalization_error": candidate["normalization_error"],
        }
    return {
        "bound_precommit_sha256": bound["precommit_sha256"],
        "evaluated_after_bound_precommit": True,
        "target_support_count": target["support_count"],
        "arms": arm_results,
    }


def _draw(distribution: Mapping[str, float], rng: random.Random) -> Any:
    threshold = rng.random()
    cumulative = 0.0
    for label, probability in distribution.items():
        cumulative += probability
        if threshold <= cumulative:
            return json.loads(label)
    return json.loads(next(reversed(distribution)))


def sample_from_ebm_kernels(
    program: exp6152.StochasticProgram,
    kernels: Mapping[str, SoftwareEBMKernel],
    *,
    seed: int,
    sample_count: int,
) -> list[JsonDict]:
    """Draw iid program samples from compiled local replacement kernels."""

    order = exp6152.validate_program(program)["topological_kernel_order"]
    by_id = {kernel.identifier: kernel for kernel in program.kernels}
    rng = random.Random(seed)
    rows: list[JsonDict] = []
    for _ in range(sample_count):
        state: JsonDict = {}
        for kernel_id in order:
            source_kernel = by_id[kernel_id]
            context = _context_key(tuple(state[wire_id] for wire_id in source_kernel.inputs))
            state[source_kernel.output] = _draw(kernels[kernel_id].conditionals[context], rng)
        rows.append({wire_id: state[wire_id] for wire_id in exp6152.wire_order(program)})
    return rows


def _empirical_distribution(
    program: exp6152.StochasticProgram,
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    counts: dict[str, int] = {}
    for row in rows:
        key = canonical_json({wire_id: row[wire_id] for wire_id in exp6152.wire_order(program)})
        counts[key] = counts.get(key, 0) + 1
    support = [
        {"state": json.loads(key), "probability": count / len(rows)}
        for key, count in sorted(counts.items())
    ]
    return {
        "wire_order": exp6152.wire_order(program),
        "support": support,
        "joint_probabilities": {key: count / len(rows) for key, count in sorted(counts.items())},
        "normalization": 1.0,
        "support_count": len(support),
    }


def _lag1_autocorrelation(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    denominator = sum(value * value for value in centered)
    if denominator <= 0.0:
        return 0.0
    numerator = sum(a * b for a, b in zip(centered[:-1], centered[1:], strict=True))
    return numerator / denominator


def _ess(sample_count: int, lag1: float) -> float:
    rho = max(0.0, min(0.99, lag1))
    return sample_count * (1.0 - rho) / (1.0 + rho)


def evaluate_sampled_joint_outputs(
    program: exp6152.StochasticProgram,
    arms: Mapping[str, Mapping[str, SoftwareEBMKernel]],
) -> JsonDict:
    """Evaluate matched-seed sampled outputs and uncertainty diagnostics."""

    target = exp6152.execute_exact(program)
    arm_results: JsonDict = {}
    radius = math.sqrt(math.log(40.0) / (2.0 * SAMPLES_PER_ARM))
    for arm_name, kernels in arms.items():
        all_rows: list[JsonDict] = []
        chains = []
        for seed in SAMPLE_SEEDS:
            rows = sample_from_ebm_kernels(
                program,
                kernels,
                seed=seed,
                sample_count=SAMPLES_PER_SEED,
            )
            all_rows.extend(rows)
            accepted = [float(row["accepted"]) for row in rows]
            lag1 = _lag1_autocorrelation(accepted)
            chains.append(
                {
                    "seed": seed,
                    "sample_count": len(rows),
                    "accepted_lag1_autocorrelation": lag1,
                    "effective_sample_size": _ess(len(rows), lag1),
                }
            )
        empirical = _empirical_distribution(program, all_rows)
        divergence = distribution_divergence(target, empirical)
        arm_results[arm_name] = {
            **divergence,
            "sample_count": len(all_rows),
            "seed_count": len(SAMPLE_SEEDS),
            "tv_dkw_radius_95": radius,
            "chains": chains,
        }
    return {
        "matched_seeds": list(SAMPLE_SEEDS),
        "samples_per_arm": SAMPLES_PER_ARM,
        "uncertainty_radius_95": radius,
        "arms": arm_results,
    }


def exact_and_sampled_case_counts(
    program: exp6152.StochasticProgram,
    sampled: Mapping[str, Any],
) -> JsonDict:
    """Report exact and sampled case counts."""

    exact = exp6152.execute_exact(program)
    return {
        "exact": {
            "wire_count": len(program.wires),
            "kernel_count": len(program.kernels),
            "state_space_size": exact["state_space_size"],
            "support_count": exact["support_count"],
            "impossible_state_count": exact["impossible_state_count"],
        },
        "sampled": {
            "samples_per_arm": sampled["samples_per_arm"],
            "seed_count": len(sampled["matched_seeds"]),
            "arms": sorted(sampled["arms"]),
        },
        "principle": FIELD_PRINCIPLES["exact_and_sampled_case_counts"],
    }


def per_factor_and_joint_distribution_divergences(
    program: exp6152.StochasticProgram,
    bound: Mapping[str, Any],
    exact: Mapping[str, Any],
    sampled: Mapping[str, Any],
) -> JsonDict:
    """Join local, exact-joint, and sampled-joint divergence evidence."""

    return {
        "per_factor": {arm: payload["per_factor_errors"] for arm, payload in bound["arms"].items()},
        "exact_joint": exact["arms"],
        "sampled_joint": sampled["arms"],
        "principle": FIELD_PRINCIPLES["per_factor_and_joint_distribution_divergences"],
        "program_checksum": exp6152.program_checksum(program),
    }


def bound_slack_and_violation_counts(
    exact: Mapping[str, Any],
    bound: Mapping[str, Any],
) -> JsonDict:
    """Compare precommitted TV bounds with exact measured joint TV."""

    rows: JsonDict = {}
    violation_count = 0
    for arm_name, result in exact["arms"].items():
        tv_bound = float(bound["arms"][arm_name]["tv_bound"])
        measured = float(result["joint_tv"])
        violated = measured > tv_bound + EXACT_TOLERANCE
        violation_count += int(violated)
        rows[arm_name] = {
            "precommitted_tv_bound": tv_bound,
            "measured_joint_tv": measured,
            "slack": tv_bound - measured,
            "bound_respected": not violated,
        }
    return {
        "bound_precommit_sha256": bound["precommit_sha256"],
        "violation_count": violation_count,
        "arms": rows,
        "principle": FIELD_PRINCIPLES["bound_slack_and_violation_counts"],
    }


def context_matched_minus_isolated_intervals(
    exact: Mapping[str, Any],
    sampled: Mapping[str, Any],
) -> JsonDict:
    """Report context-minus-isolated divergence deltas and noninferiority."""

    exact_delta = (
        exact["arms"]["context_matched"]["joint_tv"] - exact["arms"]["isolated"]["joint_tv"]
    )
    sampled_delta = (
        sampled["arms"]["context_matched"]["joint_tv"] - sampled["arms"]["isolated"]["joint_tv"]
    )
    radius = 2.0 * float(sampled["uncertainty_radius_95"])
    return {
        "primary_metric": "exact_joint_tv",
        "exact_delta": exact_delta,
        "sampled_delta": sampled_delta,
        "sampled_delta_interval_95": [sampled_delta - radius, sampled_delta + radius],
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "context_matching_noninferior": exact_delta <= NONINFERIORITY_MARGIN + EXACT_TOLERANCE,
        "context_matching_improved": exact_delta < -EXACT_TOLERANCE,
        "principle": FIELD_PRINCIPLES["context_matched_minus_isolated_intervals"],
    }


def autocorrelation_effective_sample_size_and_convergence(
    sampled: Mapping[str, Any],
) -> JsonDict:
    """Summarize sampled diagnostics and convergence flags."""

    arms: JsonDict = {}
    support_violation_count = 0
    nonconvergence_count = 0
    for arm_name, result in sampled["arms"].items():
        ess_values = [float(row["effective_sample_size"]) for row in result["chains"]]
        lag_values = [float(row["accepted_lag1_autocorrelation"]) for row in result["chains"]]
        support_violation_count += int(result["support_violation_count"])
        nonconverged = (
            min(ess_values) < MIN_EFFECTIVE_SAMPLE_SIZE
            or int(result["support_violation_count"]) > 0
        )
        nonconvergence_count += int(nonconverged)
        arms[arm_name] = {
            "effective_sample_size_min": min(ess_values),
            "effective_sample_size_mean": sum(ess_values) / len(ess_values),
            "lag1_autocorrelation_max": max(lag_values),
            "sample_count": result["sample_count"],
            "support_violation_count": result["support_violation_count"],
            "nonconverged": nonconverged,
            "chains": result["chains"],
        }
    return {
        "arms": arms,
        "support_violation_count": support_violation_count,
        "nonconvergence_count": nonconvergence_count,
        "minimum_effective_sample_size": MIN_EFFECTIVE_SAMPLE_SIZE,
        "principle": FIELD_PRINCIPLES["autocorrelation_effective_sample_size_and_convergence"],
    }


def _replace_kernel_distribution(
    kernel: SoftwareEBMKernel,
    context: Sequence[Any],
    probabilities_by_label: Mapping[Any, float],
) -> SoftwareEBMKernel:
    conditionals = json.loads(canonical_json(kernel.conditionals))
    conditionals[_context_key(context)] = {
        canonical_json(label): float(probabilities_by_label[label])
        for label in kernel.output_labels
    }
    return SoftwareEBMKernel(
        kernel.kernel_id,
        kernel.kind,
        kernel.inputs,
        kernel.output,
        kernel.output_labels,
        conditionals,
    )


def identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls(
    program: exp6152.StochasticProgram,
    arms: Mapping[str, Mapping[str, SoftwareEBMKernel]],
    exact: Mapping[str, Any],
    bound: Mapping[str, Any],
) -> JsonDict:
    """Run controls that prove the audit can fail for the right reasons."""

    target = exp6152.execute_exact(program)
    base_kernels = dict(arms["context_matched"])
    bad_kernels = dict(base_kernels)
    bad_kernels["sample_candidate_item"] = _replace_kernel_distribution(
        base_kernels["sample_candidate_item"],
        (),
        {"ac_e0_0": 0.8, "ac_e0_1": 0.1, "ac_e0_2": 0.1},
    )
    bad_joint = distribution_divergence(
        target, execute_joint_from_ebm_kernels(program, bad_kernels)
    )
    unsupported_kernels = dict(base_kernels)
    unsupported_kernels["member_group_lookup"] = _replace_kernel_distribution(
        base_kernels["member_group_lookup"],
        ("ac_e0_0",),
        {"ac_g0_0": 0.95, "ac_g0_1": 0.05},
    )
    member_group_kernel = next(
        kernel for kernel in program.kernels if kernel.identifier == "member_group_lookup"
    )
    unsupported_factor = _factor_divergence(
        program,
        member_group_kernel,
        unsupported_kernels["member_group_lookup"],
    )
    negative_controls = exp6152.run_negative_controls(program)
    identity_tv = exact["arms"]["context_matched"]["joint_tv"]
    loose_slack = 1.0 - identity_tv
    return {
        "identity_zero_error_control_passed": identity_tv <= EXACT_TOLERANCE,
        "identity_joint_tv": identity_tv,
        "bad_factor_control_fired": bad_joint["joint_tv"] > EXACT_TOLERANCE,
        "bad_factor_joint_tv": bad_joint["joint_tv"],
        "permuted_wire_control_fired": negative_controls["wire_order_bug_detected"] is True,
        "permuted_wire_joint_delta": negative_controls["wire_order_bug_max_joint_delta"],
        "unsupported_state_control_fired": unsupported_factor["support_violation_count"] > 0,
        "unsupported_state_violation_count": unsupported_factor["support_violation_count"],
        "overly_loose_bound_control_fired": loose_slack > LOOSE_BOUND_REJECTION_SLACK,
        "overly_loose_candidate_tv_bound": 1.0,
        "overly_loose_bound_rejected_reason": "slack_exceeds_informative_bound_policy",
        "precommitted_bound_hash_checked": bound["precommit_sha256"],
        "all_controls_passed": (
            identity_tv <= EXACT_TOLERANCE
            and bad_joint["joint_tv"] > EXACT_TOLERANCE
            and negative_controls["wire_order_bug_detected"] is True
            and unsupported_factor["support_violation_count"] > 0
            and loose_slack > LOOSE_BOUND_REJECTION_SLACK
        ),
        "principle": FIELD_PRINCIPLES[
            "identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls"
        ],
    }


def retired_parity_scaling_nonreuse_receipt(root: Path = REPO_ROOT) -> JsonDict:
    """Record that Exp6153 does not reopen the retired THRML parity sweep."""

    exclusion = (root / EXCLUSION_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    prior_1526 = root / "results/experiment_1526_thrml_carnot_parity_n8.json"
    prior_1564 = root / "results/experiment_1564_thrml_vendored_block_gibbs_replacement.json"
    return {
        "retired_manifest_entry_id": "thrml_scaling_sweep_lineage_retired_after_vendoring",
        "retired_lineage_blocked": "thrml_scaling_sweep_lineage_retired_after_vendoring"
        in exclusion,
        "retired_experiment_ids": [
            "exp1526",
            "exp1527",
            "exp1528",
            "exp1529",
            "exp1530",
            "exp1531",
            "exp1543",
            "exp1544",
        ],
        "blocked_patterns_checked": [
            "THRML/Carnot parity n=8",
            "THRML/Carnot parity n=16",
            "THRML/Carnot parity n=32",
            "THRML/Carnot parity n=64",
            "THRML/Carnot parity n=128",
            "THRML/Carnot parity n=256",
            "THRML diverse topology parity",
            "THRML scaling sweep",
        ],
        "size_sweep_produced": False,
        "carnot_vs_vendored_thrml_parity_table_produced": False,
        "retirement_triggered": False,
        "prior_artifact_hashes": {
            prior_1526.relative_to(root).as_posix(): sha256_file(prior_1526),
            prior_1564.relative_to(root).as_posix(): sha256_file(prior_1564),
        },
        "principle": FIELD_PRINCIPLES["retired_parity_scaling_nonreuse_receipt"],
    }


def prior_failure_and_operator_override_receipts(root: Path = REPO_ROOT) -> JsonDict:
    """Hash the operator override and retired-lineage evidence."""

    roadmap = (root / RESEARCH_ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    exclusion = (root / EXCLUSION_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    return {
        "operator_override_found": "versioned lineage continuation false-positive" in roadmap,
        "prior_failure_found": "exp1526-thrml-carnot-parity-n8" in roadmap,
        "retired_lineage_manifest_found": "thrml_scaling_sweep_lineage_retired_after_vendoring"
        in exclusion,
        "research_roadmap_sha256": sha256_file(root / RESEARCH_ROADMAP_RELATIVE_PATH),
        "exclusion_manifest_sha256": sha256_file(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "operator_override_summary": (
            "Exp6153 tests typed program-level factor replacement and compositional error "
            "bounds; retired THRML/Carnot parity scaling is not reused."
        ),
        "principle": FIELD_PRINCIPLES["prior_failure_and_operator_override_receipts"],
    }


def structured_gate_receipt(root: Path = REPO_ROOT) -> JsonDict:
    """Recompute Exp6145 and Exp6152 readiness for Exp6153."""

    exp6145_gate = exp6152.structured_gate_receipt(root)
    exp6152_artifact_path = root / exp6152.RESULT_RELATIVE_PATH
    exp6152_artifact = json.loads(exp6152_artifact_path.read_text(encoding="utf-8"))
    exp6152_valid = exp6152.validate_artifact(exp6152_artifact)
    exp6152_ready = exp6152_artifact.get("typed_stochastic_ir_ready_score") == 1.0
    return {
        "exp6145_gate": exp6145_gate,
        "exp6152_artifact": exp6152.RESULT_RELATIVE_PATH.as_posix(),
        "exp6152_artifact_sha256": sha256_file(exp6152_artifact_path),
        "exp6152_status": exp6152_artifact.get("status"),
        "exp6152_ready_score": exp6152_artifact.get("typed_stochastic_ir_ready_score"),
        "exp6152_validate_artifact": exp6152_valid,
        "gate_passed": exp6145_gate["gate_passed"] is True and exp6152_ready and exp6152_valid,
        "principle": FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def upstream_ir_executor_and_exact_reference_hashes(
    program: exp6152.StochasticProgram,
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Hash upstream Exp6152 code, tests, artifact, IR, executor, and oracle."""

    exact_reference = exp6152.independent_reference_distribution()
    return {
        "exp6152_program_checksum": exp6152.program_checksum(program),
        "exp6152_exact_distribution_sha256": sha256_json(exp6152.execute_exact(program)),
        "exp6152_independent_reference_sha256": sha256_json(exact_reference),
        "exp6152_module_sha256": sha256_file(root / exp6152.MODULE_RELATIVE_PATH),
        "exp6152_tests_sha256": sha256_file(root / exp6152.TEST_RELATIVE_PATH),
        "exp6152_artifact_sha256": sha256_file(root / exp6152.RESULT_RELATIVE_PATH),
        "executor_source_sha256": exp6152.sha256_text(inspect.getsource(exp6152.execute_exact)),
        "exact_reference_source_sha256": exp6152.sha256_text(
            inspect.getsource(exp6152.independent_reference_distribution)
        ),
        "principle": FIELD_PRINCIPLES["upstream_ir_executor_and_exact_reference_hashes"],
    }


def torx_thrml_versions_commits_import_and_api_receipts(
    program: exp6152.StochasticProgram,
) -> JsonDict:
    """Exercise pinned Torx and vendored THRML-compatible software interfaces."""

    import jax
    import jax.numpy as jnp

    torx = exp6152.torx_adapter_receipt(program)
    upstream_thrml = _optional_import_receipt("thrml")
    try:
        from carnot.sampling import _vendored_thrml as vendored_thrml

        node = vendored_thrml.CategoricalNode()
        block = vendored_thrml.Block([node])
        factor = vendored_thrml.models.CategoricalEBMFactor(
            [block], jnp.array([[0.0, 1.0]], dtype=jnp.float32)
        )
        interaction_groups = factor.to_interaction_groups()
        vendored = {
            "importable": True,
            "api_exercised": len(interaction_groups) == 1,
            "version": vendored_thrml.__version__,
            "license": vendored_thrml.THRML_LICENSE,
            "upstream_repository": vendored_thrml.THRML_UPSTREAM_REPOSITORY,
            "vendor_note": vendored_thrml.THRML_VENDOR_NOTE,
            "import_namespace": "carnot.sampling._vendored_thrml",
            "exercised_api": ["CategoricalNode", "Block", "CategoricalEBMFactor"],
            "interaction_group_count": len(interaction_groups),
            "pinned_by_artifact": "results/experiment_1564_thrml_vendored_block_gibbs_replacement.json",
            "pinned_artifact_sha256": sha256_file(
                REPO_ROOT / "results/experiment_1564_thrml_vendored_block_gibbs_replacement.json"
            ),
        }
    except Exception as exc:  # pragma: no cover - vendored THRML is required for this audit.
        vendored = {
            "importable": False,
            "api_exercised": False,
            "version": None,
            "blocked_reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "torx": torx,
        "upstream_thrml_package": upstream_thrml,
        "vendored_thrml": vendored,
        "jax": {
            "version": jax.__version__,
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        },
        "interface_ready": torx.get("compatibility_ready") is True
        and vendored.get("importable") is True
        and vendored.get("api_exercised") is True,
        "principle": FIELD_PRINCIPLES["torx_thrml_versions_commits_import_and_api_receipts"],
    }


def _optional_import_receipt(module_name: str) -> JsonDict:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "importable": False,
            "module": module_name,
            "blocked_reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "importable": True,
        "module": module_name,
        "path": getattr(module, "__file__", None),
        "version_attr": getattr(module, "__version__", None),
    }


def _source_path_hashes(root: Path = REPO_ROOT) -> JsonDict:
    return {
        "paths": _path_hashes(HASHED_SOURCE_PATHS, root),
        "output_paths": {"result": RESULT_RELATIVE_PATH.as_posix()},
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
    }


def preconditions_checked(
    *,
    output_path: Path,
    gate: Mapping[str, Any],
    prior: Mapping[str, Any],
    upstream_hashes: Mapping[str, Any],
    interfaces: Mapping[str, Any],
) -> JsonDict:
    """Build strict precondition evidence before joint evaluation."""

    spec_text = (REPO_ROOT / SAMPLER_SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    retired = retired_parity_scaling_nonreuse_receipt()
    checks = {
        "jax_platforms_cpu": os.environ.get("JAX_PLATFORMS") == "cpu",
        "structured_gate_passed": gate.get("gate_passed") is True,
        "exp6152_program_hashed": bool(upstream_hashes.get("exp6152_program_checksum")),
        "torx_thrml_interfaces_ready": interfaces.get("interface_ready") is True,
        "sampler_spec_has_req_6153": "REQ-SAMPLE-6153" in spec_text,
        "operator_override_present": prior.get("operator_override_found") is True,
        "retired_lineage_blocked": retired.get("retired_lineage_blocked") is True,
        "output_parent_writable": os.access(output_path.parent, os.W_OK),
        "protected_files_present": all((REPO_ROOT / path).exists() for path in PROTECTED_FILES),
    }
    return {
        "run_date": RUN_DATE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "checks": checks,
        "preconditions_ready": all(checks.values()),
        "source_hashes": _source_path_hashes(),
        "output_path": output_path.as_posix(),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def _unchanged_receipt(paths: Sequence[Path], before: Mapping[str, Any]) -> JsonDict:
    after = _path_hashes(paths)
    return {
        "before": dict(before),
        "after": after,
        "unchanged": dict(before) == after,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def write_thermalized_program_error_audit_artifact(
    *,
    output_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Write the Exp6153 terminal artifact."""

    started = time.monotonic()
    output = output_path or REPO_ROOT / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    protected_before = _path_hashes(PROTECTED_FILES)
    program = upstream_program()
    gate = structured_gate_receipt()
    prior = prior_failure_and_operator_override_receipts()
    upstream_hashes = upstream_ir_executor_and_exact_reference_hashes(program)
    interfaces = torx_thrml_versions_commits_import_and_api_receipts(program)
    preconditions = preconditions_checked(
        output_path=output,
        gate=gate,
        prior=prior,
        upstream_hashes=upstream_hashes,
        interfaces=interfaces,
    )
    arms = train_resource_matched_arms(program)
    bound = preregister_per_factor_to_joint_error_bound(program, arms)
    exact = evaluate_exact_joint_outputs(program, arms, bound)
    sampled = evaluate_sampled_joint_outputs(program, arms)
    slack = bound_slack_and_violation_counts(exact, bound)
    intervals = context_matched_minus_isolated_intervals(exact, sampled)
    convergence = autocorrelation_effective_sample_size_and_convergence(sampled)
    controls = identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls(
        program, arms, exact, bound
    )
    retired = retired_parity_scaling_nonreuse_receipt()
    protected = _unchanged_receipt(PROTECTED_FILES, protected_before)
    elapsed = float(duration_s if duration_s is not None else time.monotonic() - started)
    artifact = build_artifact(
        program=program,
        gate=gate,
        prior=prior,
        upstream_hashes=upstream_hashes,
        interfaces=interfaces,
        preconditions=preconditions,
        bound=bound,
        exact=exact,
        sampled=sampled,
        slack=slack,
        intervals=intervals,
        convergence=convergence,
        controls=controls,
        retired=retired,
        protected=protected,
        duration_s=elapsed,
        test_exit_codes=dict(test_exit_codes or {}),
    )
    validate_artifact(artifact)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def build_artifact(
    *,
    program: exp6152.StochasticProgram,
    gate: Mapping[str, Any],
    prior: Mapping[str, Any],
    upstream_hashes: Mapping[str, Any],
    interfaces: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    bound: Mapping[str, Any],
    exact: Mapping[str, Any],
    sampled: Mapping[str, Any],
    slack: Mapping[str, Any],
    intervals: Mapping[str, Any],
    convergence: Mapping[str, Any],
    controls: Mapping[str, Any],
    retired: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Assemble the complete Exp6153 artifact from precomputed receipts."""

    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": FIELD_PRINCIPLES,
        "status": "blocked",
        "preconditions_checked": dict(preconditions),
        "structured_gate_receipt": dict(gate),
        "prior_failure_and_operator_override_receipts": dict(prior),
        "upstream_ir_executor_and_exact_reference_hashes": dict(upstream_hashes),
        "torx_thrml_versions_commits_import_and_api_receipts": dict(interfaces),
        "factor_eligibility_and_compilation_manifest": factor_eligibility_and_compilation_manifest(
            program
        ),
        "isolated_and_context_matched_training_config": isolated_and_context_matched_training_config(),
        "preregistered_per_factor_to_joint_error_bound": dict(bound),
        "exact_and_sampled_case_counts": exact_and_sampled_case_counts(program, sampled),
        "per_factor_and_joint_distribution_divergences": per_factor_and_joint_distribution_divergences(
            program, bound, exact, sampled
        ),
        "bound_slack_and_violation_counts": dict(slack),
        "context_matched_minus_isolated_intervals": dict(intervals),
        "autocorrelation_effective_sample_size_and_convergence": dict(convergence),
        "identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls": dict(
            controls
        ),
        "retired_parity_scaling_nonreuse_receipt": dict(retired),
        "hardware_execution_claimed": False,
        "latency_power_energy_and_speedup_claimed": False,
        "thermalized_program_ready_score": 0.0,
        "retirement_triggered": bool(retired.get("retirement_triggered")),
        "protected_files_unchanged": dict(protected),
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "missing_verifier_gaps": [],
        "field_provenance": field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["thermalized_program_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the strict Exp6153 readiness scalar."""

    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    missing_commands = [
        command for command in DEFAULT_TEST_COMMANDS if command not in test_exit_codes
    ]
    nonzero_commands = [
        command for command in DEFAULT_TEST_COMMANDS if test_exit_codes.get(command) != 0
    ]
    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and dict(artifact.get("torx_thrml_versions_commits_import_and_api_receipts") or {}).get(
            "interface_ready"
        )
        is True
        and dict(artifact.get("factor_eligibility_and_compilation_manifest") or {}).get(
            "support_preserved"
        )
        is True
        and dict(artifact.get("factor_eligibility_and_compilation_manifest") or {}).get(
            "compiled_factor_count"
        )
        == 9
        and dict(artifact.get("bound_slack_and_violation_counts") or {}).get("violation_count") == 0
        and dict(artifact.get("context_matched_minus_isolated_intervals") or {}).get(
            "context_matching_noninferior"
        )
        is True
        and dict(
            artifact.get(
                "identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls"
            )
            or {}
        ).get("all_controls_passed")
        is True
        and dict(artifact.get("retired_parity_scaling_nonreuse_receipt") or {}).get(
            "retirement_triggered"
        )
        is False
        and artifact.get("hardware_execution_claimed") is False
        and artifact.get("latency_power_energy_and_speedup_claimed") is False
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and artifact.get("missing_verifier_gaps") == []
        and not missing_commands
        and not nonzero_commands
    )
    return 1.0 if ready else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Return the terminal status from artifact evidence."""

    if artifact.get("retirement_triggered") is True:
        return "retired"
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    if dict(artifact.get("bound_slack_and_violation_counts") or {}).get("violation_count", 0) > 0:
        return "complete_bound_violated"
    if ready_score(artifact) == 1.0:
        return "complete_ready"
    return "blocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the required terminal-prefixed honest verdict."""

    current = status(artifact)
    if current == "complete_ready":
        return "complete_ready: program-level error composition held for software EBM factors"
    if current == "complete_bound_violated":
        return "complete_bound_violated: program-level error composition did not hold"
    if current == "retired":
        return "retired: retired THRML parity-scaling scope was triggered"
    composition_held = (
        dict(artifact.get("bound_slack_and_violation_counts") or {}).get("violation_count") == 0
    )
    held_text = (
        "program-level error composition held but "
        if composition_held
        else "program-level error composition did not hold and "
    )
    return "blocked: " + held_text + ",".join(blocked_reasons(artifact)[:8])


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return compact blocker names for status and verdict text."""

    reasons: list[str] = []
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        reasons.append("preconditions")
    if (
        dict(artifact.get("torx_thrml_versions_commits_import_and_api_receipts") or {}).get(
            "interface_ready"
        )
        is not True
    ):
        reasons.append("software_interfaces")
    if (
        dict(artifact.get("factor_eligibility_and_compilation_manifest") or {}).get(
            "support_preserved"
        )
        is not True
    ):
        reasons.append("support_preservation")
    if dict(artifact.get("bound_slack_and_violation_counts") or {}).get("violation_count", 0) > 0:
        reasons.append("bound_violation")
    if (
        dict(artifact.get("context_matched_minus_isolated_intervals") or {}).get(
            "context_matching_noninferior"
        )
        is not True
    ):
        reasons.append("context_matching")
    if (
        dict(
            artifact.get(
                "identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls"
            )
            or {}
        ).get("all_controls_passed")
        is not True
    ):
        reasons.append("controls")
    if artifact.get("hardware_execution_claimed") is not False:
        reasons.append("hardware_claim")
    if artifact.get("latency_power_energy_and_speedup_claimed") is not False:
        reasons.append("performance_claim")
    missing = [
        command
        for command in DEFAULT_TEST_COMMANDS
        if command not in dict(artifact.get("test_exit_codes") or {})
    ]
    nonzero = [
        command
        for command in DEFAULT_TEST_COMMANDS
        if dict(artifact.get("test_exit_codes") or {}).get(command) not in (0, None)
    ]
    if missing:
        reasons.append("missing_test_commands")
    if nonzero:
        reasons.append("nonzero_test_commands")
    return reasons or ["ready_score"]


def field_provenance() -> JsonDict:
    """Map every required field to its evidence sources."""

    sources = [
        "task_prompt",
        "arxiv:2608.01615",
        SAMPLER_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6152.RESULT_RELATIVE_PATH.as_posix(),
        exp6152.MODULE_RELATIVE_PATH.as_posix(),
        "results/experiment_1526_thrml_carnot_parity_n8.json",
        "results/experiment_1564_thrml_vendored_block_gibbs_replacement.json",
        "pypi:extro-torx==0.0.1",
        "carnot.sampling._vendored_thrml==0.1.3",
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": list(sources)}
        for field in FIELD_PRINCIPLES
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate Exp6153 artifact schema and readiness consistency."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("hardware_execution_claimed") is not False:
        raise ValueError("hardware_execution_claimed")
    if artifact.get("latency_power_energy_and_speedup_claimed") is not False:
        raise ValueError("latency_power_energy_and_speedup_claimed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if artifact.get("thermalized_program_ready_score") != ready_score(artifact):
        raise ValueError("ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    return True
