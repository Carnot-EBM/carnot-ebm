"""Exp6166 mode-jumping factor thermalization.

Spec refs: REQ-SAMPLE-6166, SCENARIO-SAMPLE-6166-MULTIMODAL-CNCE,
SCENARIO-SAMPLE-6166-BOUND-CONTROLS.

The experiment keeps the target small enough that the typed stochastic program
is the oracle. CNCE training sees only sampled conditional pairs, not the exact
log-probability table. Local pairs identify within-mode ratios, while explicit
cross-mode pairs identify the relative offset between separated modes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from carnot import experiment_6153_thermalized_program_error_audit as exp6153


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6166_mode_jumping_factor_thermalization.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6166_mode_jumping_factor_thermalization.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6166_mode_jumping_factor_thermalization.py")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_HARDWARE_RELATIVE_PATH = Path("research-hardware-wishlist.md")
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_6166.mode_jumping_factor_thermalization.v1"
EXPERIMENT_ID = "experiment_6166_mode_jumping_factor_thermalization"
RUN_DATE = "20260806"
INFERENCE_SUBSTRATE = "jax_cpu_software_multimodal_factor_thermalization"
EXACT_TOLERANCE = exp6152.EXACT_TOLERANCE
CNCE_OPENREVIEW_ID = "07OWUWmUHp"
CNCE_OPENREVIEW_URL = f"https://openreview.net/forum?id={CNCE_OPENREVIEW_ID}"
CNCE_PDF_URL = f"https://openreview.net/pdf?id={CNCE_OPENREVIEW_ID}"
CNCE_TITLE = (
    "Conditional Noise-Contrastive Estimation of Energy-Based Models by Jumping Between Modes"
)
CNCE_SOURCE_SUMMARY = (
    "OpenReview metadata describes conditional NCE with deliberate pairs from different "
    "modes so relative energies of separated modes are directly compared."
)

ALL_LABELS = (
    "left_peak",
    "left_shoulder",
    "valley_left",
    "valley_right",
    "right_peak",
    "right_shoulder",
    "unsupported_shadow",
)
SUPPORT_LABELS = ALL_LABELS[:-1]
EXACT_PROBABILITIES = {
    "left_peak": 0.36,
    "left_shoulder": 0.24,
    "valley_left": 0.025,
    "valley_right": 0.025,
    "right_peak": 0.245,
    "right_shoulder": 0.105,
    "unsupported_shadow": 0.0,
}
MODE_LABELS = {
    "left_mode": ("left_peak", "left_shoulder"),
    "right_mode": ("right_peak", "right_shoulder"),
    "valley": ("valley_left", "valley_right"),
}
LOCAL_EDGES = (
    ("left_peak", "left_shoulder"),
    ("left_shoulder", "valley_left"),
    ("right_peak", "right_shoulder"),
    ("right_shoulder", "valley_right"),
)
CROSS_MODE_EDGES = (
    ("left_peak", "right_peak"),
    ("left_shoulder", "right_shoulder"),
)
WRONG_JUMP_EDGES = (
    ("left_peak", "valley_right"),
    ("right_peak", "valley_left"),
)
CROSS_MODE_MIX = 0.25
TRAINING_SEEDS = (6166, 6167, 6168, 6169)
SAMPLES_PER_SEED = 1200
OPTIMIZER_STEPS = 900
LEARNING_RATE = 0.65
L2_REGULARIZATION = 0.002
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
    RESEARCH_REFERENCES_RELATIVE_PATH,
    RESEARCH_HARDWARE_RELATIVE_PATH,
    SAMPLER_SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    exp6152.MODULE_RELATIVE_PATH,
    exp6152.TEST_RELATIVE_PATH,
    exp6152.RESULT_RELATIVE_PATH,
    exp6153.MODULE_RELATIVE_PATH,
    exp6153.TEST_RELATIVE_PATH,
    exp6153.RESULT_RELATIVE_PATH,
    Path("python/carnot/sampling/_vendored_thrml"),
    Path("python/carnot/samplers"),
)

FOCUSED_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/pytest "
    "tests/python/test_experiment_6166_mode_jumping_factor_thermalization.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6166_mode_jumping_factor_thermalization.py "
    "-m pytest tests/python/test_experiment_6166_mode_jumping_factor_thermalization.py "
    "-q --no-cov -n 0 && JAX_PLATFORMS=cpu .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6166_mode_jumping_factor_thermalization.py "
    "--fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6166_mode_jumping_factor_thermalization.py"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6166_mode_jumping_factor_thermalization.py "
    "tests/python/test_experiment_6166_mode_jumping_factor_thermalization.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6166_mode_jumping_factor_thermalization.py "
    "tests/python/test_experiment_6166_mode_jumping_factor_thermalization.py"
)
VERSION_API_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/pytest "
    "tests/python/test_experiment_6166_mode_jumping_factor_thermalization.py "
    "-q --no-cov -n 0 -k exact_factor_support_modes_and_noise_are_frozen"
)
CNCE_ARMS_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/pytest "
    "tests/python/test_experiment_6166_mode_jumping_factor_thermalization.py "
    "-q --no-cov -n 0 -k mode_jump_improves"
)
E2E_SERIALIZATION_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python/test_e2e_serialization.py -q --no-cov -n 0"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6166_mode_jumping_factor_thermalization.json"
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
    VERSION_API_COMMAND,
    CNCE_ARMS_COMMAND,
    E2E_SERIALIZATION_COMMAND,
    ADVERSARIAL_COMMAND,
    PROTECTED_FILE_COMMAND,
    ROOT_CLUTTER_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "prior_failure_and_operator_override_receipts",
    "upstream_ir_executor_bound_and_software_version_hashes",
    "cnce_source_and_algorithm_receipt",
    "exact_multimodal_factor_support_distribution_and_mode_masses",
    "frozen_local_and_cross_mode_noise_distributions",
    "matched_training_configs_seeds_samples_and_parameters",
    "exact_local_only_mode_jump_bad_and_permuted_arm_receipts",
    "factor_and_joint_tv_kl_and_mode_mass_ratio_errors",
    "deliberately_nonzero_error_receipt",
    "preregistered_factor_to_joint_bound",
    "bound_slack_and_violation_counts",
    "seed_intervals_and_convergence",
    "identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls",
    "retired_parity_scaling_nonreuse_receipt",
    "hardware_execution_claimed",
    "latency_power_energy_and_speedup_claimed",
    "mode_jumping_factor_thermalization_ready_score",
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
    "status": "Terminal state separates positive, null, retired, and blocked Exp6166 outcomes.",
    "preconditions_checked": (
        "Hashes CPU mode, upstream typed IR, software interfaces, CNCE source reachability, "
        "retired lineage, output paths, and protected files before fitting."
    ),
    "prior_failure_and_operator_override_receipts": (
        "Records why Exp6153's exact replacement was vacuous and why this nonzero-error "
        "software approximation is allowed without reopening retired parity scaling."
    ),
    "upstream_ir_executor_bound_and_software_version_hashes": (
        "Binds Exp6152 IR/executor, Exp6153 bound/executor code, Torx, vendored THRML, "
        "sampler paths, outputs, and protected files."
    ),
    "cnce_source_and_algorithm_receipt": (
        "Names OpenReview `07OWUWmUHp`, records reachable source metadata, and states the "
        "local conditional pairwise objective used."
    ),
    "exact_multimodal_factor_support_distribution_and_mode_masses": (
        "Freezes labels, support mask, exact probabilities, separated modes, mode masses, "
        "and relative mode-mass ratio before fitting."
    ),
    "frozen_local_and_cross_mode_noise_distributions": (
        "Freezes local pair noise and cross-mode jump noise before training outcomes are read."
    ),
    "matched_training_configs_seeds_samples_and_parameters": (
        "Freezes seeds, samples, optimizer, step count, learning rate, support mask, arm "
        "budgets, and primary endpoints before fitting."
    ),
    "exact_local_only_mode_jump_bad_and_permuted_arm_receipts": (
        "Keeps identity, local-only, mode-jump, deliberately bad, and permuted arms "
        "separately named and lowered through the same software-kernel interface."
    ),
    "factor_and_joint_tv_kl_and_mode_mass_ratio_errors": (
        "Reports exact factor/joint divergence, normalization, support, and mode-mass "
        "ratio error for every arm."
    ),
    "deliberately_nonzero_error_receipt": (
        "Proves approximate divergence is finite and strictly positive while the identity "
        "arm remains zero."
    ),
    "preregistered_factor_to_joint_bound": (
        "Hashes the factor-derived joint TV/KL bound before joint evaluation."
    ),
    "bound_slack_and_violation_counts": (
        "Compares measured joint TV with the precommitted bound and counts violations."
    ),
    "seed_intervals_and_convergence": (
        "Reports deterministic seed intervals, loss trends, finite parameters, and "
        "convergence diagnostics."
    ),
    "identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls": (
        "Proves exact identity, no-jump/local-only weakness, wrong-jump degradation, bad "
        "factor, permuted category wiring, support violation, and loose-bound rejection "
        "controls fire."
    ),
    "retired_parity_scaling_nonreuse_receipt": (
        "Proves no THRML/Carnot scaling sweep or parity table is produced."
    ),
    "hardware_execution_claimed": (
        "Bare false prevents this software semantic result from becoming a hardware claim."
    ),
    "latency_power_energy_and_speedup_claimed": (
        "Bare false prevents quality evidence from becoming a performance claim."
    ),
    "mode_jumping_factor_thermalization_ready_score": (
        "Equals 1.0 only when preconditions pass, nonzero finite error exists, mode jumping "
        "improves local-only, the bound holds, controls fire, protected files are unchanged, "
        "tests pass, and no forbidden claim appears."
    ),
    "retirement_triggered": "Records whether this run crossed a retired-lineage rule.",
    "protected_files_unchanged": (
        "Confirms conductor and reconciliation files were byte-identical during artifact construction."
    ),
    "duration_s": "Reports real wall time without padding.",
    "inference_substrate": (
        "Declares `jax_cpu_software_multimodal_factor_thermalization`, not hardware, GPU, "
        "or LLM inference."
    ),
    "verifier_is_oracle": "Declares exact finite enumeration as the oracle.",
    "missing_verifier_gaps": "Lists absent evidence rather than silently granting readiness.",
    "field_provenance": (
        "Maps each required field to prompt, spec, source, tests, artifacts, or package receipts."
    ),
    "test_commands": (
        "Records focused unit/spec coverage, version/API, multimodal exact reference, CNCE "
        "arms, nonzero-error, bound, controls, no-hardware, JAX CPU, schema, adversarial, "
        "protected-file, E2E, full pytest, and root-clutter checks."
    ),
    "test_exit_codes": "Prevents failed commands from becoming readiness evidence.",
    "reproducibility_checksum": (
        "Content-addresses the artifact with volatile duration and self-checksum blanked."
    ),
    "honest_verdict": (
        "Uses a required terminal prefix and states whether mode jumping improved composition."
    ),
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in stable ASCII byte order."""

    return exp6152.canonical_json(value)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return exp6152.sha256_json(value)


def sha256_file(path: str | Path) -> str:
    """Hash one file by bytes."""

    return exp6152.sha256_file(path)


def build_multimodal_factor_program() -> exp6152.StochasticProgram:
    """Build one typed categorical factor with separated modes and a hard support mask."""

    program = exp6152.StochasticProgram(
        exp6152.IR_SCHEMA_VERSION,
        "exp6166_multimodal_categorical_factor",
        (exp6152.Wire("mode_state", "categorical", ALL_LABELS),),
        (
            exp6152.Kernel(
                "sample_multimodal_mode",
                "categorical_prior",
                (),
                "mode_state",
                {
                    "probabilities": [EXACT_PROBABILITIES[label] for label in ALL_LABELS],
                    "seed_role": "mode_state_root",
                },
            ),
        ),
        {
            "source": "Exp6166 multimodal factor on Exp6152 typed PSC/DFG IR",
            "modes": {name: list(labels) for name, labels in MODE_LABELS.items()},
            "unsupported_state": "unsupported_shadow",
            "relative_mode_mass_ratio": _mode_mass_ratio(EXACT_PROBABILITIES),
        },
    )
    exp6152.validate_program(program)
    return program


def _mode_masses(probabilities: Mapping[str, float]) -> dict[str, float]:
    return {
        mode: sum(float(probabilities[label]) for label in labels)
        for mode, labels in MODE_LABELS.items()
    }


def _mode_mass_ratio(probabilities: Mapping[str, float]) -> float:
    masses = _mode_masses(probabilities)
    right = masses["right_mode"]
    return math.inf if right <= 0.0 else masses["left_mode"] / right


def exact_multimodal_factor_support_distribution_and_mode_masses(
    program: exp6152.StochasticProgram,
) -> JsonDict:
    """Freeze the exact factor distribution and mode masses before fitting."""

    exact = exp6152.execute_exact(program)
    mode_masses = _mode_masses(EXACT_PROBABILITIES)
    return {
        "labels": list(ALL_LABELS),
        "support_labels": list(SUPPORT_LABELS),
        "unsupported_states": [
            label for label in ALL_LABELS if EXACT_PROBABILITIES[label] <= EXACT_TOLERANCE
        ],
        "exact_probabilities": dict(EXACT_PROBABILITIES),
        "state_space_size": exact["state_space_size"],
        "support_count": exact["support_count"],
        "normalization": exact["normalization"],
        "normalization_error": exact["normalization_error"],
        "mode_labels": {name: list(labels) for name, labels in MODE_LABELS.items()},
        "mode_masses": mode_masses,
        "relative_mode_mass_ratio": mode_masses["left_mode"] / mode_masses["right_mode"],
        "separation_policy": "left/right modes communicate only through declared cross-mode noise",
        "exact_joint_sha256": sha256_json(exact),
        "principle": FIELD_PRINCIPLES[
            "exact_multimodal_factor_support_distribution_and_mode_masses"
        ],
    }


def _adjacency(edges: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    adjacency = {label: [] for label in SUPPORT_LABELS}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    return {label: sorted(neighbors) for label, neighbors in adjacency.items()}


def _transition_distribution(
    label: str,
    *,
    extra_edges: Sequence[tuple[str, str]] = (),
    extra_mix: float = CROSS_MODE_MIX,
) -> dict[str, float]:
    local = _adjacency(LOCAL_EDGES)
    extra = _adjacency(extra_edges)
    local_weight = 1.0
    extra_weight = 0.0
    if extra_edges and extra[label]:
        local_weight = 1.0 - extra_mix
        extra_weight = extra_mix
    probabilities: dict[str, float] = {}
    for neighbor in local[label]:
        probabilities[neighbor] = probabilities.get(neighbor, 0.0) + local_weight / len(
            local[label]
        )
    for neighbor in extra[label]:
        probabilities[neighbor] = probabilities.get(neighbor, 0.0) + extra_weight / len(
            extra[label]
        )
    normalizer = sum(probabilities.values())
    return {neighbor: value / normalizer for neighbor, value in sorted(probabilities.items())}


def frozen_local_and_cross_mode_noise_distributions() -> JsonDict:
    """Freeze local and cross-mode noise tables before training."""

    local_transitions = {label: _transition_distribution(label) for label in SUPPORT_LABELS}
    cross_transitions = {
        label: _transition_distribution(label, extra_edges=CROSS_MODE_EDGES)
        for label in SUPPORT_LABELS
    }
    wrong_transitions = {
        label: _transition_distribution(label, extra_edges=WRONG_JUMP_EDGES)
        for label in SUPPORT_LABELS
    }
    return {
        "local_noise": {
            "undirected_edges": [list(edge) for edge in LOCAL_EDGES],
            "transitions": local_transitions,
            "contains_cross_mode_jump": False,
        },
        "cross_mode_noise": {
            "undirected_edges": [list(edge) for edge in CROSS_MODE_EDGES],
            "mixture_weight": CROSS_MODE_MIX,
            "transitions": cross_transitions,
            "contains_cross_mode_jump": True,
        },
        "wrong_jump_noise": {
            "undirected_edges": [list(edge) for edge in WRONG_JUMP_EDGES],
            "mixture_weight": CROSS_MODE_MIX,
            "transitions": wrong_transitions,
            "contains_cross_mode_jump": False,
        },
        "support_mask_excludes": ["unsupported_shadow"],
        "noise_sha256": sha256_json(
            {
                "local": local_transitions,
                "cross": cross_transitions,
                "wrong": wrong_transitions,
            }
        ),
        "principle": FIELD_PRINCIPLES["frozen_local_and_cross_mode_noise_distributions"],
    }


def matched_training_configs_seeds_samples_and_parameters() -> JsonDict:
    """Freeze CNCE training budgets and endpoints before fitting."""

    return {
        "training_data_source": "sampled exact factor states plus declared conditional noise",
        "exact_log_probabilities_copied_into_approximate_arms": False,
        "arms": {
            "local_only": {
                "noise": "local_noise",
                "cross_mode_pairs_enabled": False,
                "total_pair_samples": len(TRAINING_SEEDS) * SAMPLES_PER_SEED,
            },
            "mode_jump": {
                "noise": "local_plus_cross_mode_noise",
                "cross_mode_pairs_enabled": True,
                "total_pair_samples": len(TRAINING_SEEDS) * SAMPLES_PER_SEED,
            },
        },
        "seeds": list(TRAINING_SEEDS),
        "samples_per_seed": SAMPLES_PER_SEED,
        "optimizer": "full_batch_pairwise_logistic_cnce",
        "optimizer_steps": OPTIMIZER_STEPS,
        "learning_rate": LEARNING_RATE,
        "l2_regularization": L2_REGULARIZATION,
        "support_mask": {label: label in SUPPORT_LABELS for label in ALL_LABELS},
        "primary_endpoint": "factor_tv_and_relative_mode_mass_ratio_error_pre_joint_eval",
        "secondary_endpoint": "factor_kl_target_to_candidate_pre_joint_eval",
        "principle": FIELD_PRINCIPLES["matched_training_configs_seeds_samples_and_parameters"],
    }


def _draw_from_distribution(probabilities: Mapping[str, float], rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for label, probability in probabilities.items():
        cumulative += float(probability)
        if threshold <= cumulative:
            return label
    return next(reversed(probabilities))


def _sample_data_label(rng: random.Random) -> str:
    return _draw_from_distribution(
        {label: EXACT_PROBABILITIES[label] for label in SUPPORT_LABELS}, rng
    )


def _sample_pairs(
    *,
    seed: int,
    sample_count: int,
    extra_edges: Sequence[tuple[str, str]],
) -> list[tuple[str, str, float]]:
    rng = random.Random(seed)
    pairs: list[tuple[str, str, float]] = []
    for _ in range(sample_count):
        data_label = _sample_data_label(rng)
        forward = _transition_distribution(data_label, extra_edges=extra_edges)
        noise_label = _draw_from_distribution(forward, rng)
        reverse = _transition_distribution(noise_label, extra_edges=extra_edges)
        pairs.append(
            (data_label, noise_label, math.log(forward[noise_label] / reverse[data_label]))
        )
    return pairs


def _stable_logistic_loss(logit: float) -> float:
    if logit >= 0.0:
        return math.log1p(math.exp(-logit))
    return -logit + math.log1p(math.exp(logit))


def _train_probabilities(
    *,
    arm_name: str,
    extra_edges: Sequence[tuple[str, str]],
    seeds: Sequence[int],
    samples_per_seed: int,
) -> JsonDict:
    pairs = [
        pair
        for seed in seeds
        for pair in _sample_pairs(seed=seed, sample_count=samples_per_seed, extra_edges=extra_edges)
    ]
    scores = {label: 0.0 for label in SUPPORT_LABELS}
    first_loss = 0.0
    final_loss = 0.0
    for step in range(OPTIMIZER_STEPS):
        gradients = {label: L2_REGULARIZATION * scores[label] for label in SUPPORT_LABELS}
        loss = 0.0
        for data_label, noise_label, log_noise_ratio in pairs:
            logit = scores[data_label] - scores[noise_label] + log_noise_ratio
            loss += _stable_logistic_loss(logit)
            gradient = (_sigmoid(logit) - 1.0) / len(pairs)
            gradients[data_label] += gradient
            gradients[noise_label] -= gradient
        for label in SUPPORT_LABELS:
            scores[label] -= LEARNING_RATE * gradients[label]
        mean_score = sum(scores.values()) / len(scores)
        for label in SUPPORT_LABELS:
            scores[label] -= mean_score
        if step == 0:
            first_loss = loss / len(pairs)
        if step == OPTIMIZER_STEPS - 1:
            final_loss = loss / len(pairs)
    exp_scores = {label: math.exp(scores[label]) for label in SUPPORT_LABELS}
    normalizer = sum(exp_scores.values())
    probabilities = {label: exp_scores[label] / normalizer for label in SUPPORT_LABELS}
    probabilities["unsupported_shadow"] = 0.0
    return {
        "arm_name": arm_name,
        "probabilities": probabilities,
        "scores": scores,
        "loss_start": first_loss,
        "loss_end": final_loss,
        "pair_sample_count": len(pairs),
        "pair_samples_sha256": sha256_json(pairs),
        "finite_parameters": all(math.isfinite(value) for value in scores.values()),
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _kernel_from_probabilities(
    name: str,
    probabilities: Mapping[str, float],
) -> exp6153.SoftwareEBMKernel:
    return exp6153.SoftwareEBMKernel(
        kernel_id="sample_multimodal_mode",
        kind="categorical_prior",
        inputs=(),
        output="mode_state",
        output_labels=ALL_LABELS,
        conditionals={
            exp6153._context_key(()): {
                exp6152.canonical_json(label): float(probabilities[label]) for label in ALL_LABELS
            }
        },
    )


def _arm_receipt(
    *,
    name: str,
    probabilities: Mapping[str, float],
    source: str,
    copied_exact: bool,
    training: Mapping[str, Any] | None = None,
) -> JsonDict:
    return {
        "arm_name": name,
        "source": source,
        "lowered_through": "exp6153.SoftwareEBMKernel",
        "kernel_id": "sample_multimodal_mode",
        "probabilities": dict(probabilities),
        "probability_sha256": sha256_json(probabilities),
        "normalization_error": abs(sum(probabilities.values()) - 1.0),
        "support_mask_preserved": probabilities.get("unsupported_shadow", 0.0) == 0.0,
        "copied_exact_log_probabilities": copied_exact,
        "training": dict(training or {}),
    }


def _single_seed_metrics(extra_edges: Sequence[tuple[str, str]], seed: int) -> JsonDict:
    training = _train_probabilities(
        arm_name=f"seed_{seed}",
        extra_edges=extra_edges,
        seeds=(seed,),
        samples_per_seed=SAMPLES_PER_SEED,
    )
    metrics = _distribution_metrics(training["probabilities"])
    return {
        "seed": seed,
        "joint_tv": metrics["joint_tv"],
        "mode_mass_ratio_error": metrics["mode_mass_ratio_error"],
        "loss_start": training["loss_start"],
        "loss_end": training["loss_end"],
        "finite_parameters": training["finite_parameters"],
    }


def train_matched_cnce_arms(
    program: exp6152.StochasticProgram,
) -> dict[str, JsonDict]:
    """Train exact, local-only, cross-mode, and control arms through one interface."""

    exp6152.validate_program(program)
    local_training = _train_probabilities(
        arm_name="local_only",
        extra_edges=(),
        seeds=TRAINING_SEEDS,
        samples_per_seed=SAMPLES_PER_SEED,
    )
    mode_training = _train_probabilities(
        arm_name="mode_jump",
        extra_edges=CROSS_MODE_EDGES,
        seeds=TRAINING_SEEDS,
        samples_per_seed=SAMPLES_PER_SEED,
    )
    bad = {
        label: (1.0 / len(SUPPORT_LABELS) if label in SUPPORT_LABELS else 0.0)
        for label in ALL_LABELS
    }
    permuted = {
        label: EXACT_PROBABILITIES[ALL_LABELS[(index + 2) % len(SUPPORT_LABELS)]]
        if label in SUPPORT_LABELS
        else 0.0
        for index, label in enumerate(ALL_LABELS)
    }
    wrong_jump = dict(mode_training["probabilities"])
    wrong_jump["left_peak"], wrong_jump["right_peak"] = (
        wrong_jump["right_peak"],
        wrong_jump["left_peak"],
    )
    wrong_jump["left_shoulder"], wrong_jump["right_shoulder"] = (
        wrong_jump["right_shoulder"],
        wrong_jump["left_shoulder"],
    )
    unsupported = dict(EXACT_PROBABILITIES)
    unsupported["unsupported_shadow"] = 0.02
    unsupported["right_shoulder"] -= 0.02
    arms = {
        "identity": _make_arm("identity", EXACT_PROBABILITIES, "exact_table_identity", True),
        "local_only": _make_arm(
            "local_only",
            local_training["probabilities"],
            "sampled_local_pair_cnce",
            False,
            local_training,
        ),
        "mode_jump": _make_arm(
            "mode_jump",
            mode_training["probabilities"],
            "sampled_local_plus_cross_mode_pair_cnce",
            False,
            mode_training,
        ),
        "wrong_jump": _make_arm(
            "wrong_jump",
            wrong_jump,
            "deliberately_inverted_cross_mode_pair_orientation_control",
            False,
        ),
        "bad_factor": _make_arm("bad_factor", bad, "deliberately_uniform_bad_factor", False),
        "permuted_wire": _make_arm(
            "permuted_wire",
            permuted,
            "deliberately_permuted_category_order",
            False,
        ),
        "unsupported_state": _make_arm(
            "unsupported_state",
            unsupported,
            "deliberately_support_breaking_factor",
            False,
        ),
    }
    arms["local_only"]["seed_runs"] = [_single_seed_metrics((), seed) for seed in TRAINING_SEEDS]
    arms["mode_jump"]["seed_runs"] = [
        _single_seed_metrics(CROSS_MODE_EDGES, seed) for seed in TRAINING_SEEDS
    ]
    return arms


def _make_arm(
    name: str,
    probabilities: Mapping[str, float],
    source: str,
    copied_exact: bool,
    training: Mapping[str, Any] | None = None,
) -> JsonDict:
    return {
        "kernel": _kernel_from_probabilities(name, probabilities),
        "probabilities": dict(probabilities),
        "receipt": _arm_receipt(
            name=name,
            probabilities=probabilities,
            source=source,
            copied_exact=copied_exact,
            training=training,
        ),
    }


def exact_local_only_mode_jump_bad_and_permuted_arm_receipts(
    arms: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Return JSON-safe arm receipts for the terminal artifact."""

    return {
        "arms": {
            name: dict(payload["receipt"])
            for name, payload in arms.items()
            if name
            in {
                "identity",
                "local_only",
                "mode_jump",
                "wrong_jump",
                "bad_factor",
                "permuted_wire",
                "unsupported_state",
            }
        },
        "shared_lowering_interface": "exp6153.execute_joint_from_ebm_kernels",
        "principle": FIELD_PRINCIPLES["exact_local_only_mode_jump_bad_and_permuted_arm_receipts"],
    }


def _distribution_metrics(probabilities: Mapping[str, float]) -> JsonDict:
    keys = set(EXACT_PROBABILITIES) | set(probabilities)
    tv = 0.5 * sum(
        abs(float(EXACT_PROBABILITIES.get(label, 0.0)) - float(probabilities.get(label, 0.0)))
        for label in keys
    )
    kl = 0.0
    support_violation_count = 0
    for label in keys:
        p = float(EXACT_PROBABILITIES.get(label, 0.0))
        q = float(probabilities.get(label, 0.0))
        if p > 0.0 and q <= 0.0:
            kl = math.inf
        elif p > 0.0:
            kl += p * math.log(p / q)
        if p <= EXACT_TOLERANCE and q > EXACT_TOLERANCE:
            support_violation_count += 1
    exact_ratio = _mode_mass_ratio(EXACT_PROBABILITIES)
    candidate_ratio = _mode_mass_ratio(probabilities)
    return {
        "joint_tv": tv,
        "joint_kl_target_to_candidate": kl,
        "support_violation_count": support_violation_count,
        "mode_mass_ratio": candidate_ratio,
        "mode_mass_ratio_error": abs(candidate_ratio - exact_ratio),
    }


def _factor_divergence(
    program: exp6152.StochasticProgram,
    arm: Mapping[str, Any],
) -> JsonDict:
    kernel = program.kernels[0]
    return exp6153._factor_divergence(program, kernel, arm["kernel"])


def preregistered_factor_to_joint_bound(
    program: exp6152.StochasticProgram,
    arms: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Hash factor-local TV/KL bounds before any joint evaluation is read."""

    arm_payloads: JsonDict = {}
    for arm_name, arm in arms.items():
        factor = _factor_divergence(program, arm)
        arm_payloads[arm_name] = {
            "factor_tv": factor["weighted_tv"],
            "factor_kl": factor["weighted_kl"],
            "tv_bound": min(1.0, factor["weighted_tv"]),
            "kl_bound": factor["weighted_kl"],
            "support_violation_count": factor["support_violation_count"],
            "bound_formula": "single_factor_joint_TV_equals_factor_TV_for_root_categorical_factor",
        }
    precommit_payload = {
        "schema": "carnot.exp6166.precommitted_factor_to_joint_bound.v1",
        "program_checksum": exp6152.program_checksum(program),
        "training_config": matched_training_configs_seeds_samples_and_parameters(),
        "arms": arm_payloads,
    }
    return {
        "derived_before_joint_evaluation": True,
        "joint_results_read_before_hash": False,
        "precommit_sha256": sha256_json(precommit_payload),
        "precommit_payload": precommit_payload,
        "arms": arm_payloads,
        "primary_arm": "mode_jump",
        "primary_metric": "exact_joint_tv",
        "principle": FIELD_PRINCIPLES["preregistered_factor_to_joint_bound"],
    }


def factor_and_joint_tv_kl_and_mode_mass_ratio_errors(
    program: exp6152.StochasticProgram,
    arms: Mapping[str, Mapping[str, Any]],
    bound: Mapping[str, Any],
) -> JsonDict:
    """Evaluate exact factor and joint errors after consuming the bound hash."""

    exact = exp6152.execute_exact(program)
    results: JsonDict = {}
    for arm_name, arm in arms.items():
        factor = _factor_divergence(program, arm)
        candidate = exp6153.execute_joint_from_ebm_kernels(
            program, {"sample_multimodal_mode": arm["kernel"]}
        )
        joint = exp6153.distribution_divergence(exact, candidate)
        mode_masses = _mode_masses(arm["probabilities"])
        mode_ratio = _mode_mass_ratio(arm["probabilities"])
        results[arm_name] = {
            "factor_tv": factor["weighted_tv"],
            "factor_kl_target_to_candidate": factor["weighted_kl"],
            "joint_tv": joint["joint_tv"],
            "joint_kl_target_to_candidate": joint["joint_kl_target_to_candidate"],
            "support_violation_count": joint["support_violation_count"],
            "normalization_error": candidate["normalization_error"],
            "support_count": candidate["support_count"],
            "mode_masses": mode_masses,
            "mode_mass_ratio": mode_ratio,
            "mode_mass_ratio_error": abs(
                mode_ratio
                - exact_multimodal_factor_support_distribution_and_mode_masses(program)[
                    "relative_mode_mass_ratio"
                ]
            ),
            "precommitted_tv_bound": bound["arms"][arm_name]["tv_bound"],
        }
    local = results["local_only"]
    jumped = results["mode_jump"]
    return {
        "bound_precommit_sha256": bound["precommit_sha256"],
        "evaluated_after_bound_precommit": True,
        "arms": results,
        "mode_jump_improved_over_local_only": (
            jumped["joint_tv"] < local["joint_tv"]
            and jumped["mode_mass_ratio_error"] < local["mode_mass_ratio_error"]
        ),
        "primary_metric": "exact_joint_tv",
        "principle": FIELD_PRINCIPLES["factor_and_joint_tv_kl_and_mode_mass_ratio_errors"],
    }


def deliberately_nonzero_error_receipt(errors: Mapping[str, Any]) -> JsonDict:
    """Prove approximations are non-vacuous while identity stays exact."""

    arms = errors["arms"]
    approximate_names = ("local_only", "mode_jump")
    finite_positive = all(
        math.isfinite(float(arms[name]["joint_kl_target_to_candidate"]))
        and float(arms[name]["joint_tv"]) > EXACT_TOLERANCE
        for name in approximate_names
    )
    identity_zero = (
        float(arms["identity"]["joint_tv"]) <= EXACT_TOLERANCE
        and float(arms["identity"]["joint_kl_target_to_candidate"]) <= EXACT_TOLERANCE
    )
    return {
        "approximate_arms": list(approximate_names),
        "approximate_error_finite_and_strictly_positive": finite_positive,
        "identity_exact_table_zero_error": identity_zero,
        "local_only_joint_tv": arms["local_only"]["joint_tv"],
        "mode_jump_joint_tv": arms["mode_jump"]["joint_tv"],
        "identity_joint_tv": arms["identity"]["joint_tv"],
        "principle": FIELD_PRINCIPLES["deliberately_nonzero_error_receipt"],
    }


def bound_slack_and_violation_counts(
    errors: Mapping[str, Any],
    bound: Mapping[str, Any],
) -> JsonDict:
    """Compare measured joint TV with the precommitted TV bound."""

    rows: JsonDict = {}
    violation_count = 0
    for arm_name, result in errors["arms"].items():
        measured = float(result["joint_tv"])
        tv_bound = float(bound["arms"][arm_name]["tv_bound"])
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


def seed_intervals_and_convergence(arms: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Summarize per-seed CNCE diagnostics."""

    local_runs = list(arms["local_only"]["seed_runs"])
    jump_runs = list(arms["mode_jump"]["seed_runs"])
    deltas = [
        {
            "seed": local["seed"],
            "joint_tv_delta_mode_jump_minus_local": jump["joint_tv"] - local["joint_tv"],
            "mode_ratio_error_delta_mode_jump_minus_local": (
                jump["mode_mass_ratio_error"] - local["mode_mass_ratio_error"]
            ),
        }
        for local, jump in zip(local_runs, jump_runs, strict=True)
    ]
    return {
        "seeds": list(TRAINING_SEEDS),
        "local_only": local_runs,
        "mode_jump": jump_runs,
        "deltas": deltas,
        "mode_jump_improved_primary_metric_all_seeds": all(
            delta["joint_tv_delta_mode_jump_minus_local"] < 0.0 for delta in deltas
        ),
        "mode_jump_improved_ratio_error_all_seeds": all(
            delta["mode_ratio_error_delta_mode_jump_minus_local"] < 0.0 for delta in deltas
        ),
        "loss_decreased_all_runs": all(
            run["loss_end"] < run["loss_start"] for run in [*local_runs, *jump_runs]
        ),
        "finite_parameters_all_runs": all(
            run["finite_parameters"] for run in [*local_runs, *jump_runs]
        ),
        "principle": FIELD_PRINCIPLES["seed_intervals_and_convergence"],
    }


def identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls(
    errors: Mapping[str, Any],
    bound: Mapping[str, Any],
) -> JsonDict:
    """Run controls that prove the audit detects exact, bad, and unsupported arms."""

    arms = errors["arms"]
    identity_ok = arms["identity"]["joint_tv"] <= EXACT_TOLERANCE
    no_jump = (
        arms["local_only"]["joint_tv"] > arms["mode_jump"]["joint_tv"]
        and arms["local_only"]["mode_mass_ratio_error"] > arms["mode_jump"]["mode_mass_ratio_error"]
    )
    wrong_jump = (
        arms["wrong_jump"]["mode_mass_ratio_error"] > arms["mode_jump"]["mode_mass_ratio_error"]
    )
    bad_factor = arms["bad_factor"]["joint_tv"] > EXACT_TOLERANCE
    permuted = arms["permuted_wire"]["joint_tv"] > EXACT_TOLERANCE
    unsupported = arms["unsupported_state"]["support_violation_count"] > 0
    loose_slack = 1.0 - float(arms["mode_jump"]["joint_tv"])
    loose = loose_slack > LOOSE_BOUND_REJECTION_SLACK
    return {
        "identity_zero_error_control_passed": identity_ok,
        "identity_joint_tv": arms["identity"]["joint_tv"],
        "no_jump_control_fired": no_jump,
        "no_jump_joint_tv": arms["local_only"]["joint_tv"],
        "wrong_jump_control_fired": wrong_jump,
        "wrong_jump_mode_mass_ratio_error": arms["wrong_jump"]["mode_mass_ratio_error"],
        "bad_factor_control_fired": bad_factor,
        "bad_factor_joint_tv": arms["bad_factor"]["joint_tv"],
        "permuted_wire_control_fired": permuted,
        "permuted_wire_joint_tv": arms["permuted_wire"]["joint_tv"],
        "unsupported_state_control_fired": unsupported,
        "unsupported_state_violation_count": arms["unsupported_state"]["support_violation_count"],
        "loose_bound_control_fired": loose,
        "loose_bound_candidate_tv_bound": 1.0,
        "loose_bound_slack": loose_slack,
        "precommitted_bound_hash_checked": bound["precommit_sha256"],
        "all_controls_passed": all(
            (identity_ok, no_jump, wrong_jump, bad_factor, permuted, unsupported, loose)
        ),
        "principle": FIELD_PRINCIPLES[
            "identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls"
        ],
    }


def cnce_source_and_algorithm_receipt() -> JsonDict:
    """Record the OpenReview source locator and the local pairwise objective."""

    source_metadata = {
        "paper_id": CNCE_OPENREVIEW_ID,
        "title": CNCE_TITLE,
        "forum_url": CNCE_OPENREVIEW_URL,
        "pdf_url": CNCE_PDF_URL,
        "source_summary": CNCE_SOURCE_SUMMARY,
    }
    return {
        **source_metadata,
        "primary_source": "OpenReview EurIPS 2025 PriGM workshop",
        "source_access_status": "openreview_browser_challenge_observed_api_403_recorded",
        "source_metadata_sha256": sha256_json(source_metadata),
        "local_algorithm": {
            "objective": "conditional_pairwise_logistic_cnce",
            "orientation_logit": ("score(data)-score(noise)+log q(noise|data)-log q(data|noise)"),
            "normalizer_handling": "scores_centered_each_step_support_masked_softmax_after_fit",
            "cross_mode_mechanism": "add edges comparing left and right mode representatives",
        },
        "copied_exact_log_probabilities_into_approximate_arms": False,
        "principle": FIELD_PRINCIPLES["cnce_source_and_algorithm_receipt"],
    }


def prior_failure_and_operator_override_receipts(root: Path = REPO_ROOT) -> JsonDict:
    """Hash the Exp6153 zero-divergence context and operator-authorized CNCE hook."""

    exp6153_path = root / exp6153.RESULT_RELATIVE_PATH
    exp6153_artifact = json.loads(exp6153_path.read_text(encoding="utf-8"))
    exact_joint = exp6153_artifact["per_factor_and_joint_distribution_divergences"]["exact_joint"]
    references = (root / RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    return {
        "exp6153_artifact": exp6153.RESULT_RELATIVE_PATH.as_posix(),
        "exp6153_artifact_sha256": sha256_file(exp6153_path),
        "exp6153_zero_divergence_exact_replacement_found": all(
            float(row["joint_tv"]) <= EXACT_TOLERANCE for row in exact_joint.values()
        ),
        "operator_cnce_hook_found": CNCE_OPENREVIEW_ID in references and CNCE_TITLE in references,
        "research_references_sha256": sha256_file(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "operator_override_summary": (
            "Exp6166 intentionally replaces Exp6153's exact-table identity result with "
            "finite-sample CNCE approximations that must have nonzero error."
        ),
        "principle": FIELD_PRINCIPLES["prior_failure_and_operator_override_receipts"],
    }


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


def upstream_ir_executor_bound_and_software_version_hashes(
    program: exp6152.StochasticProgram,
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Hash upstream typed IR, bound code, software interfaces, and source paths."""

    import jax

    upstream_program = exp6152.compile_exp6145_bounded_workflow()
    interfaces = exp6153.torx_thrml_versions_commits_import_and_api_receipts(upstream_program)
    return {
        "exp6166_program_checksum": exp6152.program_checksum(program),
        "exp6166_exact_distribution_sha256": sha256_json(exp6152.execute_exact(program)),
        "exp6152_program_checksum": exp6152.program_checksum(upstream_program),
        "exp6152_module_sha256": sha256_file(root / exp6152.MODULE_RELATIVE_PATH),
        "exp6152_artifact_sha256": sha256_file(root / exp6152.RESULT_RELATIVE_PATH),
        "exp6153_module_sha256": sha256_file(root / exp6153.MODULE_RELATIVE_PATH),
        "exp6153_artifact_sha256": sha256_file(root / exp6153.RESULT_RELATIVE_PATH),
        "exp6152_executor_source_sha256": exp6152.sha256_text(
            inspect.getsource(exp6152.execute_exact)
        ),
        "exp6153_joint_executor_source_sha256": exp6152.sha256_text(
            inspect.getsource(exp6153.execute_joint_from_ebm_kernels)
        ),
        "exp6153_bound_source_sha256": exp6152.sha256_text(
            inspect.getsource(exp6153.preregister_per_factor_to_joint_error_bound)
        ),
        "software_interfaces": interfaces,
        "jax": {
            "version": jax.__version__,
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        },
        "source_hashes": {
            "paths": _path_hashes(HASHED_SOURCE_PATHS, root),
            "output_paths": {"result": RESULT_RELATIVE_PATH.as_posix()},
            "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        },
        "interfaces_ready": interfaces.get("interface_ready") is True,
        "principle": FIELD_PRINCIPLES["upstream_ir_executor_bound_and_software_version_hashes"],
    }


def retired_parity_scaling_nonreuse_receipt(root: Path = REPO_ROOT) -> JsonDict:
    """Record that Exp6166 does not reopen retired THRML parity scaling."""

    receipt = exp6153.retired_parity_scaling_nonreuse_receipt(root)
    receipt["principle"] = FIELD_PRINCIPLES["retired_parity_scaling_nonreuse_receipt"]
    return receipt


def preconditions_checked(
    *,
    output_path: Path,
    prior: Mapping[str, Any],
    upstream_hashes: Mapping[str, Any],
    cnce_source: Mapping[str, Any],
) -> JsonDict:
    """Build strict precondition evidence before fitting."""

    spec_text = (REPO_ROOT / SAMPLER_SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    retired = retired_parity_scaling_nonreuse_receipt()
    checks = {
        "jax_platforms_cpu": os.environ.get("JAX_PLATFORMS") == "cpu",
        "jax_default_backend_cpu": upstream_hashes["jax"]["default_backend"] == "cpu",
        "exp6152_program_validated": exp6152.validate_program(build_multimodal_factor_program())[
            "ok"
        ]
        is True,
        "exp6153_zero_divergence_context_hashed": prior.get(
            "exp6153_zero_divergence_exact_replacement_found"
        )
        is True,
        "cnce_source_metadata_hashed": bool(cnce_source.get("source_metadata_sha256")),
        "software_interfaces_ready": upstream_hashes.get("interfaces_ready") is True,
        "sampler_spec_has_req_6166": "REQ-SAMPLE-6166" in spec_text,
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


def write_mode_jumping_factor_thermalization_artifact(
    *,
    output_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Write the Exp6166 terminal artifact."""

    started = time.monotonic()
    output = output_path or REPO_ROOT / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    protected_before = _path_hashes(PROTECTED_FILES)
    program = build_multimodal_factor_program()
    cnce_source = cnce_source_and_algorithm_receipt()
    prior = prior_failure_and_operator_override_receipts()
    upstream_hashes = upstream_ir_executor_bound_and_software_version_hashes(program)
    preconditions = preconditions_checked(
        output_path=output,
        prior=prior,
        upstream_hashes=upstream_hashes,
        cnce_source=cnce_source,
    )
    arms = train_matched_cnce_arms(program)
    bound = preregistered_factor_to_joint_bound(program, arms)
    errors = factor_and_joint_tv_kl_and_mode_mass_ratio_errors(program, arms, bound)
    nonzero = deliberately_nonzero_error_receipt(errors)
    slack = bound_slack_and_violation_counts(errors, bound)
    intervals = seed_intervals_and_convergence(arms)
    controls = identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls(
        errors, bound
    )
    retired = retired_parity_scaling_nonreuse_receipt()
    protected = _unchanged_receipt(PROTECTED_FILES, protected_before)
    elapsed = float(duration_s if duration_s is not None else time.monotonic() - started)
    artifact = build_artifact(
        program=program,
        preconditions=preconditions,
        prior=prior,
        upstream_hashes=upstream_hashes,
        cnce_source=cnce_source,
        arms=arms,
        bound=bound,
        errors=errors,
        nonzero=nonzero,
        slack=slack,
        intervals=intervals,
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
    preconditions: Mapping[str, Any],
    prior: Mapping[str, Any],
    upstream_hashes: Mapping[str, Any],
    cnce_source: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    bound: Mapping[str, Any],
    errors: Mapping[str, Any],
    nonzero: Mapping[str, Any],
    slack: Mapping[str, Any],
    intervals: Mapping[str, Any],
    controls: Mapping[str, Any],
    retired: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Assemble the complete Exp6166 artifact from precomputed receipts."""

    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": FIELD_PRINCIPLES,
        "status": "blocked",
        "preconditions_checked": dict(preconditions),
        "prior_failure_and_operator_override_receipts": dict(prior),
        "upstream_ir_executor_bound_and_software_version_hashes": dict(upstream_hashes),
        "cnce_source_and_algorithm_receipt": dict(cnce_source),
        "exact_multimodal_factor_support_distribution_and_mode_masses": (
            exact_multimodal_factor_support_distribution_and_mode_masses(program)
        ),
        "frozen_local_and_cross_mode_noise_distributions": (
            frozen_local_and_cross_mode_noise_distributions()
        ),
        "matched_training_configs_seeds_samples_and_parameters": (
            matched_training_configs_seeds_samples_and_parameters()
        ),
        "exact_local_only_mode_jump_bad_and_permuted_arm_receipts": (
            exact_local_only_mode_jump_bad_and_permuted_arm_receipts(arms)
        ),
        "factor_and_joint_tv_kl_and_mode_mass_ratio_errors": dict(errors),
        "deliberately_nonzero_error_receipt": dict(nonzero),
        "preregistered_factor_to_joint_bound": dict(bound),
        "bound_slack_and_violation_counts": dict(slack),
        "seed_intervals_and_convergence": dict(intervals),
        "identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls": dict(
            controls
        ),
        "retired_parity_scaling_nonreuse_receipt": dict(retired),
        "hardware_execution_claimed": False,
        "latency_power_energy_and_speedup_claimed": False,
        "mode_jumping_factor_thermalization_ready_score": 0.0,
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
    artifact["mode_jumping_factor_thermalization_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the strict Exp6166 readiness scalar."""

    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    missing_commands = [
        command for command in DEFAULT_TEST_COMMANDS if command not in test_exit_codes
    ]
    nonzero_commands = [
        command for command in DEFAULT_TEST_COMMANDS if test_exit_codes.get(command) != 0
    ]
    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and dict(artifact.get("deliberately_nonzero_error_receipt") or {}).get(
            "approximate_error_finite_and_strictly_positive"
        )
        is True
        and dict(artifact.get("deliberately_nonzero_error_receipt") or {}).get(
            "identity_exact_table_zero_error"
        )
        is True
        and dict(artifact.get("factor_and_joint_tv_kl_and_mode_mass_ratio_errors") or {}).get(
            "mode_jump_improved_over_local_only"
        )
        is True
        and dict(artifact.get("bound_slack_and_violation_counts") or {}).get("violation_count") == 0
        and dict(artifact.get("seed_intervals_and_convergence") or {}).get(
            "mode_jump_improved_primary_metric_all_seeds"
        )
        is True
        and dict(
            artifact.get(
                "identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls"
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
        return "blocked"
    if ready_score(artifact) == 1.0:
        return "complete_positive"
    if (
        dict(artifact.get("factor_and_joint_tv_kl_and_mode_mass_ratio_errors") or {}).get(
            "mode_jump_improved_over_local_only"
        )
        is False
    ):
        return "complete_null"
    return "blocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the required terminal-prefixed honest verdict."""

    current = status(artifact)
    improved = (
        dict(artifact.get("factor_and_joint_tv_kl_and_mode_mass_ratio_errors") or {}).get(
            "mode_jump_improved_over_local_only"
        )
        is True
    )
    if current == "complete_positive":
        return "complete_positive: mode jumping improved composition with finite nonzero error"
    if current == "complete_null":
        return "complete_null: mode jumping did not improve composition"
    if current == "retired":
        return "retired: retired THRML parity-scaling scope was triggered"
    prefix = "mode jumping improved but " if improved else "mode jumping did not improve and "
    return "blocked: " + prefix + ",".join(blocked_reasons(artifact)[:8])


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return compact blocker names for status and verdict text."""

    reasons: list[str] = []
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        reasons.append("preconditions")
    if (
        dict(artifact.get("deliberately_nonzero_error_receipt") or {}).get(
            "approximate_error_finite_and_strictly_positive"
        )
        is not True
    ):
        reasons.append("nonzero_error")
    if (
        dict(artifact.get("factor_and_joint_tv_kl_and_mode_mass_ratio_errors") or {}).get(
            "mode_jump_improved_over_local_only"
        )
        is not True
    ):
        reasons.append("mode_jump_improvement")
    if dict(artifact.get("bound_slack_and_violation_counts") or {}).get("violation_count", 0) > 0:
        reasons.append("bound_violation")
    if (
        dict(
            artifact.get(
                "identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls"
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
        CNCE_OPENREVIEW_URL,
        SAMPLER_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6152.RESULT_RELATIVE_PATH.as_posix(),
        exp6152.MODULE_RELATIVE_PATH.as_posix(),
        exp6153.RESULT_RELATIVE_PATH.as_posix(),
        exp6153.MODULE_RELATIVE_PATH.as_posix(),
        RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
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
    """Validate Exp6166 artifact schema and readiness consistency."""

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
    if artifact.get("mode_jumping_factor_thermalization_ready_score") != ready_score(artifact):
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
