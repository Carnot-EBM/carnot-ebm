"""Exp6237 activated mode-jump sampler A/B.

Spec refs: REQ-SAMPLER-6237,
SCENARIO-SAMPLER-6237-ACTIVATED-EQUIVALENCE,
SCENARIO-SAMPLER-6237-CONTROLS-FAIL-CLOSED.

This harness measures whether the mode-jump runtime was active before it
reports quality. This matters because a sampler comparison is not meaningful
when the step that distinguishes the treatment never runs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

import numpy as np

from carnot import experiment_6208_mode_jump_runtime_integration as exp6208
from carnot.samplers.mode_jump_rust_backend import (
    ACTIVE_PYTHON_FALLBACK,
    ACTIVE_RUST_BACKEND,
    MODE_JUMP_ALGORITHM,
    MODE_JUMP_TOPOLOGY,
    ModeJumpRustBackend,
    checkpoint_checksum,
    descriptor_for_run,
    frozen_mode_jump_inputs,
    sha256_json,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6237_activated_mode_jump_sampler_ab.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6237_activated_mode_jump_sampler_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6237_activated_mode_jump_sampler_ab.py")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
BACKEND_RELATIVE_PATH = Path("python/carnot/samplers/mode_jump_rust_backend.py")
FACTORY_RELATIVE_PATH = Path("python/carnot/samplers/backend.py")
RUST_KERNEL_RELATIVE_PATH = Path("crates/carnot-samplers/src/mode_jump.rs")
PYO3_BINDING_RELATIVE_PATH = Path("crates/carnot-python/src/mode_jump.rs")
EXP6166_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6166_mode_jumping_factor_thermalization.json"
)
EXP6208_RESULT_RELATIVE_PATH = Path("results/experiment_6208_mode_jump_runtime_integration.json")
EXP6220_RESULT_RELATIVE_PATH = Path("results/experiment_6220_mode_jump_runtime_ab.json")

SCHEMA = "carnot.experiment_6237.activated_mode_jump_sampler_ab.v1"
EXPERIMENT_ID = "experiment_6237_activated_mode_jump_sampler_ab"
RUN_DATE = "20260809"
INFERENCE_SUBSTRATE = "local_cpu_software_activated_mode_jump_sampler_ab"
DEFAULT_RECEIPT_PATH = Path("/tmp/carnot_6237_command_receipts.json")

SEEDS = (6237, 6238, 6239)
BURN_IN = 128
RETAINED_SAMPLE_COUNT = 4096
WALL_BUDGET_S = 5.0
RESTART_PREFIX_COUNT = 32
RESTART_SUFFIX_COUNT = 32
MAX_ACF_LAG = 200

MODE_LABELS: dict[str, tuple[str, ...]] = {
    "left_mode": ("left_peak", "left_shoulder"),
    "right_mode": ("right_peak", "right_shoulder"),
    "valley": ("valley_left", "valley_right"),
}
CROSS_MODE_EDGES = {
    frozenset(("left_peak", "right_peak")),
    frozenset(("left_shoulder", "right_shoulder")),
}

QUALITY_TOLERANCES: dict[str, float] = {
    "target_tv": 0.035,
    "target_kl": 0.006,
    "energy_mean_abs_error": 0.07,
    "energy_variance_abs_error": 0.12,
    "mode_mass_abs_error": 0.05,
    "ess_min": 750.0,
    "autocorrelation_abs_max": 0.85,
}
EQUIVALENCE_BOUNDS: dict[str, float] = {
    "total_variation_to_target_delta": 0.01,
    "kl_target_to_empirical_delta": 0.002,
    "energy_mean_abs_error_delta": 0.02,
    "energy_variance_abs_error_delta": 0.04,
    "max_mode_mass_abs_error_delta": 0.02,
    "effective_sample_size_delta": 250.0,
    "lag1_autocorrelation_delta": 0.05,
    "mode_coverage_fraction_delta": 0.01,
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_paths_hashes_and_determinations",
    "literature_control_path_and_hash",
    "preregistered_fixture_seed_budget_matrix",
    "exact_reference_distribution_receipts",
    "arm_support_matrix",
    "matched_arm_configuration",
    "jump_proposal_acceptance_and_transition_counts",
    "treatment_activation_score",
    "multimodal_positive_control",
    "fallback_parity_control",
    "distribution_quality_by_fixture_arm",
    "mode_coverage_ess_autocorrelation_by_fixture_arm",
    "wall_and_transition_costs",
    "paired_intervals",
    "equivalence_bounds_and_decision",
    "unsupported_or_failed_cells",
    "task_owned_and_preexisting_nonzero_command_ledger",
    "default_off_preserved",
    "hardware_claim_count",
    "sampler_runtime_ready_score",
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
    "status": "Separates supported equivalence evidence from inconclusive, blocked, and instrument-failure outcomes.",
    "upstream_paths_hashes_and_determinations": "Pins Exp6166, Exp6208, Exp6220, and sampler source evidence before this A/B is trusted.",
    "literature_control_path_and_hash": "Pins the local arXiv:2608.05025 warning that inactive treatments cannot support sampler conclusions.",
    "preregistered_fixture_seed_budget_matrix": "Freezes fixtures, seeds, transition budgets, wall budgets, metrics, margins, and positive controls before compute.",
    "exact_reference_distribution_receipts": "Records the exact finite target, modes, support, and hashes used as the oracle.",
    "arm_support_matrix": "Shows that every preregistered main fixture arm is supported and that controls fail closed.",
    "matched_arm_configuration": "Proves the fallback and mode-jump arms share target, proposal, seed, initial state, burn-in, retained samples, and wall budget.",
    "jump_proposal_acceptance_and_transition_counts": "Stores chain-level samples and transition counts before aggregation.",
    "treatment_activation_score": "Equals 1.0 only when Rust/PyO3 ran and nonzero mode-jump proposals and acceptances occurred.",
    "multimodal_positive_control": "Proves the fixture can exercise cross-mode moves before quality is interpreted.",
    "fallback_parity_control": "Proves seeded exact fallback and active Rust/PyO3 replay the same transition stream.",
    "distribution_quality_by_fixture_arm": "Reports exact-distribution error and energy moments for each supported fixture arm.",
    "mode_coverage_ess_autocorrelation_by_fixture_arm": "Reports mode coverage, ESS, and autocorrelation instead of using frequency alone.",
    "wall_and_transition_costs": "Reports matched transition budgets and diagnostic wall costs without making a speed claim.",
    "paired_intervals": "Stores paired quality and cost intervals across seeds.",
    "equivalence_bounds_and_decision": "Applies preregistered margins and emits positive, negative, equivalence-supported, inconclusive, or instrument-failure decisions.",
    "unsupported_or_failed_cells": "Keeps unsupported fixtures, degenerate chains, interruption, and activation failures visible.",
    "task_owned_and_preexisting_nonzero_command_ledger": "Separates task-owned command failures from separately classified repository-wide nonzero commands.",
    "default_off_preserved": "Bare true only when the production default remains CPU and mode-jump Rust execution requires explicit opt-in.",
    "hardware_claim_count": "Bare integer `0` prevents this software A/B from becoming a hardware claim.",
    "sampler_runtime_ready_score": "Summarizes activation, support, parity, equivalence, default-off, protected-file, and command gates.",
    "protected_files_unchanged": "Confirms conductor and reconciler-owned files stayed unchanged.",
    "preconditions_checked": "Records that preregistration and protected hashes were captured before sampler chains ran.",
    "inference_substrate": "Declares local CPU software mode-jump sampling, not hardware, GPU, TSU, or LLM inference.",
    "verifier_is_oracle": "States that exact finite enumeration and transition receipts are the verifier.",
    "field_provenance": "Maps each required field to prompt, spec, local source, upstream artifact, command receipt, or computed chain evidence.",
    "field_principles": "Explains why each required field exists before a reviewer trusts the artifact.",
    "test_commands": "Records focused tests, coverage, experiment, E2E, adversarial, and separately run suite receipts.",
    "test_exit_codes": "Stores exit codes so failed commands cannot become readiness evidence.",
    "duration_s": "Reports real wall time without padding.",
    "reproducibility_checksum": "Content-addresses the artifact after blanking volatile duration and the checksum field.",
    "honest_verdict": "Uses a terminal prefix and states activation, equivalence, unsupported controls, commands, default-off, and no hardware claim.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_float(value: Any) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return _stable_float(statistics.mean(values))


def _interval(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [_stable_float(values[0]), _stable_float(values[0])]
    mean = statistics.mean(values)
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [_stable_float(mean - half_width), _stable_float(mean + half_width)]


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected at {path}")
    return payload


def _path_hashes(root: Path, paths: Sequence[Path]) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for path in paths:
        full = root / path
        rows[path.as_posix()] = {
            "exists": full.exists(),
            "sha256": sha256_file(full) if full.exists() else None,
            "size_bytes": full.stat().st_size if full.exists() else None,
        }
    return rows


def _fixture(root: Path = REPO_ROOT) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    payload = {
        "name": "exp6166_multimodal_exact",
        "source": EXP6166_RESULT_RELATIVE_PATH.as_posix(),
        "labels": labels,
        "target_probabilities": target.astype(float).tolist(),
        "proposal_probabilities": proposal.astype(float).tolist(),
        "mode_labels": {key: list(value) for key, value in MODE_LABELS.items()},
        "cross_mode_edges": [sorted(edge) for edge in sorted(CROSS_MODE_EDGES, key=sorted)],
        "expected_adapter_support": True,
    }
    payload["fixture_sha256"] = sha256_json(
        {
            "labels": payload["labels"],
            "target_probabilities": payload["target_probabilities"],
            "proposal_probabilities": payload["proposal_probabilities"],
            "mode_labels": payload["mode_labels"],
        }
    )
    return payload


def preregistered_fixture_seed_budget_matrix(root: Path = REPO_ROOT) -> JsonDict:
    fixture = _fixture(root)
    matrix = {
        "fixtures": [
            {
                "name": fixture["name"],
                "fixture_sha256": fixture["fixture_sha256"],
                "source": fixture["source"],
                "expected_adapter_support": True,
            }
        ],
        "arms": ["seeded_fallback", "mode_jump_runtime"],
        "seeds": list(SEEDS),
        "transition_budget": {
            "burn_in": BURN_IN,
            "retained_samples": RETAINED_SAMPLE_COUNT,
            "total_transitions": BURN_IN + RETAINED_SAMPLE_COUNT,
        },
        "wall_budget_s_per_cell": WALL_BUDGET_S,
        "observables": [
            "sample_labels",
            "transition_counts",
            "mode_jump_proposals",
            "mode_jump_acceptances",
            "total_variation_to_target",
            "kl_target_to_empirical",
            "energy_mean",
            "energy_variance",
            "mode_coverage",
            "effective_sample_size",
            "lag1_autocorrelation",
            "wall_time_s",
        ],
        "quality_tolerances": dict(QUALITY_TOLERANCES),
        "equivalence_bounds": dict(EQUIVALENCE_BOUNDS),
        "positive_control": "nonzero cross-mode proposal and acceptance on multimodal fixture",
        "unsupported_controls": [
            "unsupported_fixture_control",
            "zero_activation_control",
            "degenerate_chain_control",
            "interruption_restart_control",
        ],
        "recursive_repository_suite_inside_cli": False,
        "principle": FIELD_PRINCIPLES["preregistered_fixture_seed_budget_matrix"],
    }
    matrix["matrix_sha256"] = sha256_json(
        {key: value for key, value in matrix.items() if key != "principle"}
    )
    return matrix


def preconditions_checked(root: Path, matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "computed_before_sampler_chains": True,
        "run_date": RUN_DATE,
        "matrix_sha256": matrix["matrix_sha256"],
        "focused_commands_frozen": [
            ".venv/bin/pytest tests/python/test_experiment_6237_activated_mode_jump_sampler_ab.py -q -o addopts=",
            ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6237_activated_mode_jump_sampler_ab.py -m pytest tests/python/test_experiment_6237_activated_mode_jump_sampler_ab.py -q --no-cov -o addopts=",
            ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6237_activated_mode_jump_sampler_ab.py --fail-under=100",
        ],
        "protected_hashes_before_compute": _path_hashes(root, PROTECTED_FILES),
        "source_hashes_before_compute": _path_hashes(
            root,
            (
                MODULE_RELATIVE_PATH,
                TEST_RELATIVE_PATH,
                SAMPLER_SPEC_RELATIVE_PATH,
                BACKEND_RELATIVE_PATH,
                RUST_KERNEL_RELATIVE_PATH,
                PYO3_BINDING_RELATIVE_PATH,
            ),
        ),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def upstream_paths_hashes_and_determinations(root: Path = REPO_ROOT) -> JsonDict:
    artifact_paths = (
        EXP6166_RESULT_RELATIVE_PATH,
        EXP6208_RESULT_RELATIVE_PATH,
        EXP6220_RESULT_RELATIVE_PATH,
    )
    determinations: dict[str, JsonDict] = {}
    for path in artifact_paths:
        payload = _read_json(root / path)
        determinations[path.as_posix()] = {
            "status": payload.get("status"),
            "honest_verdict": payload.get("honest_verdict"),
            "sha256": sha256_file(root / path),
        }
    return {
        "paths": _path_hashes(
            root,
            (
                EXP6166_RESULT_RELATIVE_PATH,
                EXP6208_RESULT_RELATIVE_PATH,
                EXP6220_RESULT_RELATIVE_PATH,
                BACKEND_RELATIVE_PATH,
                FACTORY_RELATIVE_PATH,
                RUST_KERNEL_RELATIVE_PATH,
                PYO3_BINDING_RELATIVE_PATH,
            ),
        ),
        "determinations": determinations,
        "exp6220_limitation_recorded": "activation was not a prerequisite for quality conclusions",
        "principle": FIELD_PRINCIPLES["upstream_paths_hashes_and_determinations"],
    }


def literature_control_path_and_hash(root: Path = REPO_ROOT) -> JsonDict:
    path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    text = path.read_text(encoding="utf-8")
    marker = "arXiv:2608.05025"
    return {
        "path": RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "contains_arxiv_2608_05025_note": marker in text,
        "control_rule": "prove treatment activation before sampler quality conclusions",
        "principle": FIELD_PRINCIPLES["literature_control_path_and_hash"],
    }


def exact_reference_distribution_receipts(root: Path = REPO_ROOT) -> JsonDict:
    fixture = _fixture(root)
    labels = [str(label) for label in fixture["labels"]]
    target = np.asarray(fixture["target_probabilities"], dtype=np.float64)
    probabilities = {label: _stable_float(target[index]) for index, label in enumerate(labels)}
    mode_masses = {
        mode: _stable_float(sum(probabilities[label] for label in members))
        for mode, members in MODE_LABELS.items()
    }
    return {
        "fixture": fixture["name"],
        "fixture_sha256": fixture["fixture_sha256"],
        "labels": labels,
        "support": [label for label in labels if probabilities[label] > 0.0],
        "target_probabilities": probabilities,
        "proposal_sha256": sha256_json(fixture["proposal_probabilities"]),
        "mode_labels": {key: list(value) for key, value in MODE_LABELS.items()},
        "mode_masses": mode_masses,
        "normalization_error": _stable_float(abs(float(target.sum()) - 1.0)),
        "oracle": "exact finite categorical distribution from Exp6166",
        "principle": FIELD_PRINCIPLES["exact_reference_distribution_receipts"],
    }


def matched_arm_configuration(root: Path = REPO_ROOT) -> JsonDict:
    fixture = _fixture(root)
    return {
        "fixture": fixture["name"],
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
        "matched_seeds": list(SEEDS),
        "matched_algorithm": MODE_JUMP_ALGORITHM,
        "matched_topology": MODE_JUMP_TOPOLOGY,
        "matched_target_hash": sha256_json(fixture["target_probabilities"]),
        "matched_proposal_hash": sha256_json(fixture["proposal_probabilities"]),
        "matched_initial_label": "left_peak",
        "matched_burn_in": BURN_IN,
        "matched_retained_sample_count": RETAINED_SAMPLE_COUNT,
        "matched_total_transitions": BURN_IN + RETAINED_SAMPLE_COUNT,
        "matched_wall_budget_s_per_cell": WALL_BUDGET_S,
        "principle": FIELD_PRINCIPLES["matched_arm_configuration"],
    }


def _descriptor(
    seed: int, *, return_trace: bool = True, checkpoint: Mapping[str, Any] | None = None
) -> JsonDict:
    labels = [str(label) for label in _fixture()["labels"]]
    descriptor = descriptor_for_run(
        labels=labels,
        seed=seed,
        burn_in=BURN_IN,
        enable_mode_jump_runtime=True,
    )
    descriptor["return_trace"] = bool(return_trace)
    if checkpoint is not None:
        descriptor["checkpoint"] = checkpoint
    return descriptor


def _support_error(exc: BaseException) -> JsonDict:
    return {"support_valid": False, "error_type": type(exc).__name__, "message": str(exc)}


def _run_chain(root: Path, seed: int, arm: str, *, return_trace: bool = True) -> JsonDict:
    fixture = _fixture(root)
    target = np.asarray(fixture["target_probabilities"], dtype=np.float64)
    proposal = np.asarray(fixture["proposal_probabilities"], dtype=np.float64)
    prefer_rust = arm == "mode_jump_runtime"
    backend = ModeJumpRustBackend(seed=seed, prefer_rust=prefer_rust)
    started = time.perf_counter()
    try:
        result = backend.run_descriptor(
            target,
            proposal,
            n_samples=RETAINED_SAMPLE_COUNT,
            config=_descriptor(seed, return_trace=return_trace),
        )
    except Exception as exc:  # pragma: no cover - main preregistered cells should be supported.
        elapsed = _stable_float(time.perf_counter() - started)
        return {
            "success": False,
            "fixture": fixture["name"],
            "seed": seed,
            "arm": arm,
            "elapsed_s": elapsed,
            "wall_budget_s": WALL_BUDGET_S,
            **_support_error(exc),
        }
    elapsed = _stable_float(time.perf_counter() - started)
    metrics = _distribution_metrics(
        fixture["labels"],
        np.asarray(fixture["target_probabilities"], dtype=np.float64),
        result["sample_labels"],
    )
    quality = _quality_from_labels(result["sample_labels"])
    mode_counts = _mode_counts(result["sample_labels"])
    transition_counts = _transition_counts(result["decision_log"])
    jump_counts = _mode_jump_counts(result["decision_log"])
    return {
        "success": True,
        "support_valid": True,
        "fixture": fixture["name"],
        "seed": seed,
        "arm": arm,
        "elapsed_s": elapsed,
        "wall_budget_s": WALL_BUDGET_S,
        "wall_budget_met": elapsed <= WALL_BUDGET_S,
        "active_backend": result["receipt"]["active_backend"],
        "fallback_reason": result["receipt"]["fallback_reason"],
        "receipt": result["receipt"],
        "sample_labels": [str(label) for label in result["sample_labels"]],
        "decision_log": result["decision_log"],
        "checkpoint": result["checkpoint"],
        "distribution_metrics": metrics,
        "quality_metrics": quality,
        "mode_counts": mode_counts,
        "transition_counts": transition_counts,
        **jump_counts,
    }


def _measure_main_chains(root: Path) -> list[JsonDict]:
    chains: list[JsonDict] = []
    for seed in SEEDS:
        chains.append(_run_chain(root, seed, "seeded_fallback"))
        chains.append(_run_chain(root, seed, "mode_jump_runtime"))
    return chains


def _mode_counts(sample_labels: Sequence[str]) -> dict[str, int]:
    return {
        mode: sum(str(label) in set(members) for label in sample_labels)
        for mode, members in MODE_LABELS.items()
    }


def _mode_coverage_fraction(sample_labels: Sequence[str]) -> float:
    if not sample_labels:
        return 0.0
    covered = sum(count > 0 for count in _mode_counts(sample_labels).values())
    return _stable_float(covered / len(MODE_LABELS))


def _transition_counts(decision_log: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in decision_log:
        before = str(event["state_before"]["current_label"])
        after = str(event["state_after"]["current_label"])
        counts[f"{before}->{after}"] += 1
    return dict(sorted(counts.items()))


def _mode_jump_counts(decision_log: Sequence[Mapping[str, Any]]) -> JsonDict:
    proposals = 0
    acceptances = 0
    for event in decision_log:
        before = str(event["state_before"]["current_label"])
        proposed = str(event["proposed_label"])
        if frozenset((before, proposed)) in CROSS_MODE_EDGES:
            proposals += 1
            acceptances += int(bool(event["accepted"]))
    return {
        "mode_jump_proposal_count": proposals,
        "mode_jump_acceptance_count": acceptances,
    }


def _energy_values(labels: Sequence[str], target: np.ndarray) -> dict[str, float]:
    return {str(label): -math.log(float(target[index])) for index, label in enumerate(labels)}


def _distribution_metrics(
    labels: Sequence[str],
    target: np.ndarray,
    sample_labels: Sequence[str],
) -> JsonDict:
    total = len(sample_labels)
    counts = Counter(str(label) for label in sample_labels)
    frequencies = {
        str(label): _stable_float(counts[str(label)] / total) if total else 0.0 for label in labels
    }
    tv = 0.5 * sum(
        abs(frequencies[str(label)] - float(target[index])) for index, label in enumerate(labels)
    )
    kl = 0.0
    for index, label in enumerate(labels):
        probability = float(target[index])
        frequency = frequencies[str(label)]
        if probability > 0.0 and frequency > 0.0:
            kl += probability * math.log(probability / frequency)
        elif probability > 0.0:
            kl = float("inf")
    energies = _energy_values(labels, target)
    exact_mean = sum(
        float(target[index]) * energies[str(label)] for index, label in enumerate(labels)
    )
    exact_second = sum(
        float(target[index]) * energies[str(label)] * energies[str(label)]
        for index, label in enumerate(labels)
    )
    sample_energy = [energies[str(label)] for label in sample_labels]
    sample_mean = float(np.mean(sample_energy)) if sample_energy else 0.0
    sample_variance = float(np.var(sample_energy)) if sample_energy else 0.0
    exact_variance = exact_second - exact_mean * exact_mean
    exact_modes = {
        mode: sum(float(target[list(labels).index(label)]) for label in members)
        for mode, members in MODE_LABELS.items()
    }
    sample_modes = {
        mode: _stable_float(sum(counts[label] for label in members) / total) if total else 0.0
        for mode, members in MODE_LABELS.items()
    }
    mode_errors = {
        mode: _stable_float(abs(sample_modes[mode] - exact_modes[mode])) for mode in MODE_LABELS
    }
    return {
        "sample_count": total,
        "frequencies": frequencies,
        "total_variation_to_target": _stable_float(tv),
        "kl_target_to_empirical": _stable_float(kl),
        "exact_energy_mean": _stable_float(exact_mean),
        "sample_energy_mean": _stable_float(sample_mean),
        "energy_mean_abs_error": _stable_float(abs(sample_mean - exact_mean)),
        "exact_energy_variance": _stable_float(exact_variance),
        "sample_energy_variance": _stable_float(sample_variance),
        "energy_variance_abs_error": _stable_float(abs(sample_variance - exact_variance)),
        "mode_masses_exact": {mode: _stable_float(value) for mode, value in exact_modes.items()},
        "mode_masses_empirical": sample_modes,
        "mode_mass_abs_errors": mode_errors,
        "max_mode_mass_abs_error": _stable_float(max(mode_errors.values())),
        "mode_coverage_fraction": _mode_coverage_fraction(sample_labels),
    }


def _quality_from_labels(sample_labels: Sequence[str]) -> JsonDict:
    if not sample_labels:
        return {
            "degenerate": True,
            "lag1_autocorrelation": 0.0,
            "integrated_autocorrelation_time": 1.0,
            "effective_sample_size": 0.0,
        }
    indicator = [1.0 if str(label) == "left_peak" else 0.0 for label in sample_labels]
    mean = sum(indicator) / len(indicator)
    denom = sum((value - mean) ** 2 for value in indicator)
    if len(indicator) < 2 or denom == 0.0:
        return {
            "degenerate": len(set(sample_labels)) <= 1,
            "lag1_autocorrelation": 0.0,
            "integrated_autocorrelation_time": 1.0,
            "effective_sample_size": _stable_float(len(indicator)),
        }
    lag1 = _autocorrelation(indicator, mean, denom, 1)
    positive_sum = 0.0
    for lag in range(1, min(MAX_ACF_LAG, len(indicator) - 1) + 1):
        rho = _autocorrelation(indicator, mean, denom, lag)
        if rho <= 0.0:
            break
        positive_sum += rho
    iact = max(1.0, 1.0 + 2.0 * positive_sum)
    return {
        "degenerate": False,
        "lag1_autocorrelation": _stable_float(lag1),
        "integrated_autocorrelation_time": _stable_float(iact),
        "effective_sample_size": _stable_float(len(indicator) / iact),
    }


def _autocorrelation(values: Sequence[float], mean: float, denom: float, lag: int) -> float:
    return (
        sum(
            (values[index] - mean) * (values[index - lag] - mean)
            for index in range(lag, len(values))
        )
        / denom
    )


def arm_support_matrix(
    chains: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]]
) -> JsonDict:
    main_cells = [
        {
            "fixture": chain["fixture"],
            "seed": chain["seed"],
            "arm": chain["arm"],
            "support_valid": bool(chain["support_valid"]),
            "active_backend": chain.get("active_backend"),
            "fallback_reason": chain.get("fallback_reason"),
            "message": chain.get("message"),
        }
        for chain in chains
    ]
    failed = [cell for cell in main_cells if cell["support_valid"] is not True]
    return {
        "main_cells": main_cells,
        "all_main_fixture_arms_supported": not failed,
        "main_unsupported_or_failed_cells": failed,
        "unsupported_controls_fail_closed": all(
            row.get("fail_closed") is True
            for row in controls
            if row.get("classification") == "unsupported_control"
        ),
        "principle": FIELD_PRINCIPLES["arm_support_matrix"],
    }


def jump_proposal_acceptance_and_transition_counts(chains: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = []
    for chain in chains:
        rows.append(
            {
                "fixture": chain["fixture"],
                "seed": chain["seed"],
                "arm": chain["arm"],
                "active_backend": chain.get("active_backend"),
                "support_valid": bool(chain["support_valid"]),
                "sample_labels": chain.get("sample_labels", []),
                "transition_counts": chain.get("transition_counts", {}),
                "accepted_count": int(chain["receipt"]["final_state"]["accepted_count"])
                if chain.get("success")
                else 0,
                "attempted_count": int(chain["receipt"]["transition_budget"]["total_steps"])
                if chain.get("success")
                else 0,
                "mode_jump_proposal_count": int(chain.get("mode_jump_proposal_count", 0)),
                "mode_jump_acceptance_count": int(chain.get("mode_jump_acceptance_count", 0)),
                "sample_labels_sha256": sha256_json(chain.get("sample_labels", [])),
            }
        )
    return {
        "chains": rows,
        "main_cells_all_recorded": all(
            row["support_valid"] and row["sample_labels"] for row in rows
        ),
        "total_mode_jump_proposals": sum(row["mode_jump_proposal_count"] for row in rows),
        "total_mode_jump_acceptances": sum(row["mode_jump_acceptance_count"] for row in rows),
        "principle": FIELD_PRINCIPLES["jump_proposal_acceptance_and_transition_counts"],
    }


def treatment_activation_score(chains: Sequence[Mapping[str, Any]]) -> JsonDict:
    treatment = [chain for chain in chains if chain["arm"] == "mode_jump_runtime"]
    rust_active = all(chain.get("active_backend") == ACTIVE_RUST_BACKEND for chain in treatment)
    proposals = sum(int(chain.get("mode_jump_proposal_count", 0)) for chain in treatment)
    acceptances = sum(int(chain.get("mode_jump_acceptance_count", 0)) for chain in treatment)
    passed = bool(treatment and rust_active and proposals > 0 and acceptances > 0)
    return {
        "score": 1.0 if passed else 0.0,
        "activation_passed": passed,
        "all_treatment_chains_used_rust_pyo3": rust_active,
        "mode_jump_proposal_count": proposals,
        "mode_jump_acceptance_count": acceptances,
        "instrument_failure_if_false": True,
        "principle": FIELD_PRINCIPLES["treatment_activation_score"],
    }


def multimodal_positive_control(
    chains: Sequence[Mapping[str, Any]], activation: Mapping[str, Any]
) -> JsonDict:
    treatment_samples = [
        label
        for chain in chains
        if chain["arm"] == "mode_jump_runtime"
        for label in chain.get("sample_labels", [])
    ]
    mode_counts = _mode_counts(treatment_samples)
    left_right_covered = mode_counts["left_mode"] > 0 and mode_counts["right_mode"] > 0
    passed = bool(activation.get("activation_passed") is True and left_right_covered)
    return {
        "fixture": "exp6166_multimodal_exact",
        "passed": passed,
        "mode_counts": mode_counts,
        "left_and_right_modes_covered": left_right_covered,
        "nonzero_jump_proposals": int(activation.get("mode_jump_proposal_count", 0)) > 0,
        "nonzero_jump_acceptances": int(activation.get("mode_jump_acceptance_count", 0)) > 0,
        "quality_conclusions_allowed": passed,
        "principle": FIELD_PRINCIPLES["multimodal_positive_control"],
    }


def fallback_parity_control(chains: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_seed = {
        seed: {
            chain["arm"]: chain
            for chain in chains
            if chain.get("seed") == seed and chain.get("success") is True
        }
        for seed in SEEDS
    }
    rows = []
    for seed, arms in by_seed.items():
        fallback = arms["seeded_fallback"]
        runtime = arms["mode_jump_runtime"]
        rows.append(
            {
                "seed": seed,
                "sample_labels_match": fallback["sample_labels"] == runtime["sample_labels"],
                "final_state_match": fallback["checkpoint"]["state"]
                == runtime["checkpoint"]["state"],
                "decision_log_match": fallback["decision_log"] == runtime["decision_log"],
                "fallback_active_backend": fallback["active_backend"],
                "runtime_active_backend": runtime["active_backend"],
            }
        )
    return {
        "pairs": rows,
        "exact_replay_match": all(
            row["sample_labels_match"] and row["final_state_match"] for row in rows
        ),
        "rust_python_decision_logs_match": all(row["decision_log_match"] for row in rows),
        "principle": FIELD_PRINCIPLES["fallback_parity_control"],
    }


def _quality_pass(metrics: Mapping[str, Any], quality: Mapping[str, Any]) -> bool:
    return bool(
        metrics["total_variation_to_target"] <= QUALITY_TOLERANCES["target_tv"]
        and metrics["kl_target_to_empirical"] <= QUALITY_TOLERANCES["target_kl"]
        and metrics["energy_mean_abs_error"] <= QUALITY_TOLERANCES["energy_mean_abs_error"]
        and metrics["energy_variance_abs_error"] <= QUALITY_TOLERANCES["energy_variance_abs_error"]
        and metrics["max_mode_mass_abs_error"] <= QUALITY_TOLERANCES["mode_mass_abs_error"]
        and quality["effective_sample_size"] >= QUALITY_TOLERANCES["ess_min"]
        and abs(quality["lag1_autocorrelation"]) <= QUALITY_TOLERANCES["autocorrelation_abs_max"]
        and quality["degenerate"] is False
    )


def distribution_quality_by_fixture_arm(
    chains: Sequence[Mapping[str, Any]],
    positive_control: Mapping[str, Any],
) -> JsonDict:
    grouped = _group_by_fixture_arm(chains)
    rows: dict[str, JsonDict] = {}
    all_pass = True
    for fixture, arms in grouped.items():
        rows[fixture] = {}
        for arm, arm_chains in arms.items():
            metrics = [chain["distribution_metrics"] for chain in arm_chains]
            qualities = [chain["quality_metrics"] for chain in arm_chains]
            chain_rows = []
            for chain, metric, quality in zip(arm_chains, metrics, qualities, strict=True):
                passed = _quality_pass(metric, quality)
                all_pass = all_pass and passed
                chain_rows.append(
                    {
                        "seed": chain["seed"],
                        **metric,
                        "quality_pass": passed,
                    }
                )
            rows[fixture][arm] = {
                "chains": chain_rows,
                "mean_total_variation_to_target": _mean(
                    [metric["total_variation_to_target"] for metric in metrics]
                ),
                "mean_kl_target_to_empirical": _mean(
                    [metric["kl_target_to_empirical"] for metric in metrics]
                ),
                "max_mode_mass_abs_error": _stable_float(
                    max(metric["max_mode_mass_abs_error"] for metric in metrics)
                ),
            }
    return {
        "fixtures": rows,
        "quality_tolerances": dict(QUALITY_TOLERANCES),
        "all_supported_quality_passed": all_pass,
        "quality_interpretable": bool(positive_control.get("quality_conclusions_allowed") is True),
        "principle": FIELD_PRINCIPLES["distribution_quality_by_fixture_arm"],
    }


def mode_coverage_ess_autocorrelation_by_fixture_arm(
    chains: Sequence[Mapping[str, Any]],
) -> JsonDict:
    grouped = _group_by_fixture_arm(chains)
    rows: dict[str, JsonDict] = {}
    all_pass = True
    for fixture, arms in grouped.items():
        rows[fixture] = {}
        for arm, arm_chains in arms.items():
            chain_rows = []
            for chain in arm_chains:
                quality = chain["quality_metrics"]
                passed = (
                    chain["distribution_metrics"]["mode_coverage_fraction"] == 1.0
                    and quality["effective_sample_size"] >= QUALITY_TOLERANCES["ess_min"]
                    and abs(quality["lag1_autocorrelation"])
                    <= QUALITY_TOLERANCES["autocorrelation_abs_max"]
                    and quality["degenerate"] is False
                )
                all_pass = all_pass and passed
                chain_rows.append(
                    {
                        "seed": chain["seed"],
                        "mode_counts": chain["mode_counts"],
                        "mode_coverage_fraction": chain["distribution_metrics"][
                            "mode_coverage_fraction"
                        ],
                        **quality,
                        "mixing_pass": passed,
                    }
                )
            rows[fixture][arm] = {
                "chains": chain_rows,
                "mean_effective_sample_size": _mean(
                    [row["effective_sample_size"] for row in chain_rows]
                ),
                "mean_lag1_autocorrelation": _mean(
                    [row["lag1_autocorrelation"] for row in chain_rows]
                ),
            }
    return {
        "fixtures": rows,
        "all_supported_mixing_passed": all_pass,
        "principle": FIELD_PRINCIPLES["mode_coverage_ess_autocorrelation_by_fixture_arm"],
    }


def _group_by_fixture_arm(
    chains: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for chain in chains:
        grouped.setdefault(str(chain["fixture"]), {}).setdefault(str(chain["arm"]), []).append(
            chain
        )
    return grouped


def wall_and_transition_costs(chains: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [
        {
            "fixture": chain["fixture"],
            "seed": chain["seed"],
            "arm": chain["arm"],
            "elapsed_s": chain["elapsed_s"],
            "wall_budget_s": chain["wall_budget_s"],
            "wall_budget_met": chain["wall_budget_met"],
            "transition_count": chain["receipt"]["transition_budget"]["total_steps"],
            "retained_sample_count": chain["receipt"]["transition_budget"]["retained_samples"],
            "transitions_per_second": _stable_float(
                chain["receipt"]["transition_budget"]["total_steps"] / chain["elapsed_s"]
            ),
        }
        for chain in chains
        if chain.get("success") is True
    ]
    transition_counts = {row["transition_count"] for row in rows}
    return {
        "chains": rows,
        "matched_transition_budget": len(transition_counts) == 1,
        "wall_budget_s_per_cell": WALL_BUDGET_S,
        "all_main_cells_within_wall_budget": all(row["wall_budget_met"] for row in rows),
        "timing_is_cost_receipt_not_speed_claim": True,
        "platform": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "os_cpu_count": os.cpu_count(),
        },
        "principle": FIELD_PRINCIPLES["wall_and_transition_costs"],
    }


def paired_intervals(chains: Sequence[Mapping[str, Any]]) -> JsonDict:
    metrics = [
        "total_variation_to_target",
        "kl_target_to_empirical",
        "energy_mean_abs_error",
        "energy_variance_abs_error",
        "max_mode_mass_abs_error",
        "effective_sample_size",
        "lag1_autocorrelation",
        "mode_coverage_fraction",
        "wall_time_s",
    ]
    by_seed = {
        seed: {
            chain["arm"]: chain
            for chain in chains
            if chain.get("seed") == seed and chain.get("success") is True
        }
        for seed in SEEDS
    }
    deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
    for arms in by_seed.values():
        fallback = arms["seeded_fallback"]
        runtime = arms["mode_jump_runtime"]
        for metric in metrics:
            if metric == "wall_time_s":
                left = float(runtime["elapsed_s"])
                right = float(fallback["elapsed_s"])
            elif metric in {"effective_sample_size", "lag1_autocorrelation"}:
                left = float(runtime["quality_metrics"][metric])
                right = float(fallback["quality_metrics"][metric])
            else:
                left = float(runtime["distribution_metrics"][metric])
                right = float(fallback["distribution_metrics"][metric])
            deltas[metric].append(_stable_float(left - right))
    return {
        "paired_seed_count": len(by_seed),
        "delta_definition": "mode_jump_runtime minus seeded_fallback",
        "intervals": {
            f"{metric}_delta": {
                "values": values,
                "mean": _mean(values),
                "mean_95_interval": _interval(values),
            }
            for metric, values in deltas.items()
        },
        "principle": FIELD_PRINCIPLES["paired_intervals"],
    }


def equivalence_bounds_and_decision(artifact: Mapping[str, Any]) -> JsonDict:
    activation = dict(artifact.get("treatment_activation_score") or {})
    positive = dict(artifact.get("multimodal_positive_control") or {})
    support = dict(artifact.get("arm_support_matrix") or {})
    quality = dict(artifact.get("distribution_quality_by_fixture_arm") or {})
    mixing = dict(artifact.get("mode_coverage_ess_autocorrelation_by_fixture_arm") or {})
    intervals = dict(dict(artifact.get("paired_intervals") or {}).get("intervals") or {})
    if activation.get("activation_passed") is not True or positive.get("passed") is not True:
        decision = "instrument_failure"
        reason = "treatment inactive or positive control failed"
    elif support.get("all_main_fixture_arms_supported") is not True:
        decision = "inconclusive"
        reason = "not all main fixture arms were supported"
    elif (
        quality.get("all_supported_quality_passed") is not True
        or mixing.get("all_supported_mixing_passed") is not True
    ):
        decision = "negative"
        reason = "mode-jump quality or mixing gate failed"
    elif _all_quality_intervals_within_bounds(intervals):
        decision = "equivalence_supported"
        reason = "paired quality intervals stayed inside preregistered equivalence bounds"
    else:
        decision = "inconclusive"
        reason = "paired quality intervals crossed an equivalence bound"
    return {
        "bounds": dict(EQUIVALENCE_BOUNDS),
        "decision": decision,
        "decision_reason": reason,
        "outcome_labels_supported": [
            "positive",
            "negative",
            "equivalence_supported",
            "inconclusive",
            "instrument_failure",
        ],
        "quality_conclusion_allowed": decision != "instrument_failure",
        "cost_conclusion": "descriptive_only",
        "principle": FIELD_PRINCIPLES["equivalence_bounds_and_decision"],
    }


def _all_quality_intervals_within_bounds(intervals: Mapping[str, Any]) -> bool:
    for metric, margin in EQUIVALENCE_BOUNDS.items():
        row = dict(intervals.get(metric) or {})
        low, high = row.get("mean_95_interval", [float("inf"), float("-inf")])
        if abs(float(low)) > margin or abs(float(high)) > margin:
            return False
    return True


def unsupported_or_failed_cells(root: Path, chains: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows = [
        {
            "cell": "main_fixture_cell",
            "fixture": chain["fixture"],
            "seed": chain["seed"],
            "arm": chain["arm"],
            "recorded": True,
            "classification": "main_cell_failure",
            "fail_closed": True,
            "message": chain.get("message"),
        }
        for chain in chains
        if chain.get("success") is not True
    ]
    rows.append(_unsupported_fixture_control(root))
    rows.append(
        {
            "cell": "zero_activation_control",
            "recorded": True,
            "classification": "activation_control",
            "decision": "instrument_failure",
            "null_sampler_verdict_allowed": False,
            "fail_closed": True,
        }
    )
    degenerate = _quality_from_labels(["left_peak", "left_peak", "left_peak"])
    rows.append(
        {
            "cell": "degenerate_chain_control",
            "recorded": True,
            "classification": "degenerate_control",
            "degenerate": degenerate["degenerate"],
            "effective_sample_size": degenerate["effective_sample_size"],
            "fail_closed": True,
        }
    )
    rows.append(_interruption_restart_control(root))
    return rows


def _unsupported_fixture_control(root: Path) -> JsonDict:
    labels, _target, proposal = frozen_mode_jump_inputs(root)
    unsupported_target = np.asarray([0.86, 0.06, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
    descriptor = descriptor_for_run(
        labels=labels,
        seed=SEEDS[0],
        burn_in=1,
        enable_mode_jump_runtime=True,
    )
    try:
        ModeJumpRustBackend(seed=SEEDS[0]).run_descriptor(
            unsupported_target,
            proposal,
            n_samples=2,
            config=descriptor,
        )
    except ValueError as exc:
        return {
            "cell": "unsupported_fixture_control",
            "recorded": True,
            "classification": "unsupported_control",
            "fail_closed": True,
            **_support_error(exc),
        }
    return {
        "cell": "unsupported_fixture_control",
        "recorded": True,
        "classification": "unsupported_control",
        "fail_closed": False,
        "support_valid": True,
    }


def _interruption_restart_control(root: Path) -> JsonDict:
    labels, target, proposal = frozen_mode_jump_inputs(root)
    descriptor = descriptor_for_run(
        labels=labels,
        seed=SEEDS[0],
        burn_in=1,
        enable_mode_jump_runtime=True,
    )
    interrupted = False
    try:
        ModeJumpRustBackend(seed=SEEDS[0]).run_descriptor(
            target,
            proposal,
            n_samples=4,
            config={**descriptor, "cancel_after_steps": 0},
        )
    except TimeoutError:
        interrupted = True
    backend = ModeJumpRustBackend(seed=SEEDS[0])
    prefix = backend.run_descriptor(target, proposal, RESTART_PREFIX_COUNT, descriptor)
    checkpoint = backend.save_checkpoint()
    loaded = backend.load_checkpoint(checkpoint, target, proposal, config=descriptor)
    resumed = {**descriptor, "checkpoint": checkpoint}
    runtime = ModeJumpRustBackend(seed=SEEDS[0]).run_descriptor(
        target,
        proposal,
        RESTART_SUFFIX_COUNT,
        resumed,
    )
    fallback = ModeJumpRustBackend(seed=SEEDS[0], prefer_rust=False).run_descriptor(
        target,
        proposal,
        RESTART_SUFFIX_COUNT,
        resumed,
    )
    restart_equivalence = (
        loaded == checkpoint["state"]
        and checkpoint["payload_checksum"] == checkpoint_checksum(checkpoint)
        and runtime["sample_labels"] == fallback["sample_labels"]
        and runtime["decision_log"] == fallback["decision_log"]
    )
    return {
        "cell": "interruption_restart_control",
        "recorded": True,
        "classification": "restart_control",
        "interruption_recorded": interrupted,
        "prefix_sample_count": len(prefix["sample_labels"]),
        "checkpoint_checksum_valid": checkpoint["payload_checksum"]
        == checkpoint_checksum(checkpoint),
        "restart_equivalence": restart_equivalence,
        "fail_closed": interrupted and restart_equivalence,
    }


def task_owned_and_preexisting_nonzero_command_ledger(
    command_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    receipts = [dict(row) for row in command_receipts]
    task_owned = [row for row in receipts if bool(row.get("task_owned", True)) is True]
    task_failures = [row for row in task_owned if int(row.get("exit_code", -999)) != 0]
    preexisting = [
        row
        for row in receipts
        if int(row.get("exit_code", 0)) != 0 and bool(row.get("task_owned", True)) is False
    ]
    return {
        "command_receipts": receipts,
        "task_owned_failure_count": len(task_failures),
        "task_owned_failures": task_failures,
        "all_task_owned_commands_passed": len(task_failures) == 0,
        "preexisting_repository_wide_nonzero_commands": preexisting,
        "preexisting_nonzero_count": len(preexisting),
        "experiment_cli_runs_recursive_repository_suite": False,
        "principle": FIELD_PRINCIPLES["task_owned_and_preexisting_nonzero_command_ledger"],
    }


def default_off_preserved(root: Path = REPO_ROOT) -> bool:
    return bool(exp6208.default_off_receipt(root)["default_off_pass"])


def protected_files_unchanged(root: Path = REPO_ROOT) -> JsonDict:
    diff = subprocess.run(
        ["git", "diff", "--quiet", "--", *[path.as_posix() for path in PROTECTED_FILES]],
        cwd=root,
        check=False,
    )
    return {
        "paths": [path.as_posix() for path in PROTECTED_FILES],
        "hashes": _path_hashes(root, PROTECTED_FILES),
        "unchanged": diff.returncode == 0,
        "git_diff_exit_code": diff.returncode,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def verifier_is_oracle() -> JsonDict:
    return {
        "value": True,
        "oracle": "exact finite categorical target plus fixed transition receipts",
        "not_oracle_for": ["hardware", "power", "energy efficiency", "unseen target families"],
        "principle": FIELD_PRINCIPLES["verifier_is_oracle"],
    }


def field_provenance() -> JsonDict:
    sources = {
        "status": "computed terminal gates",
        "upstream_paths_hashes_and_determinations": "upstream artifact and source hashes",
        "literature_control_path_and_hash": "research-references.md V539 note",
        "preregistered_fixture_seed_budget_matrix": "frozen constants in this module",
        "exact_reference_distribution_receipts": "Exp6166 exact target through Exp6208 adapter inputs",
        "arm_support_matrix": "main sampler chain support receipts and controls",
        "matched_arm_configuration": "ModeJumpRustBackend descriptors",
        "jump_proposal_acceptance_and_transition_counts": "chain decision logs and samples",
        "treatment_activation_score": "mode-jump treatment decision logs",
        "multimodal_positive_control": "mode coverage and jump activation counts",
        "fallback_parity_control": "paired fallback and Rust/PyO3 chain replay",
        "distribution_quality_by_fixture_arm": "exact target and sample labels",
        "mode_coverage_ess_autocorrelation_by_fixture_arm": "sample labels and chain indicators",
        "wall_and_transition_costs": "perf_counter and transition receipts",
        "paired_intervals": "paired seed deltas",
        "equivalence_bounds_and_decision": "preregistered bounds and paired intervals",
        "unsupported_or_failed_cells": "unsupported, zero-activation, degenerate, and restart controls",
        "task_owned_and_preexisting_nonzero_command_ledger": "external command receipts",
        "default_off_preserved": "Exp6208 default-off receipt",
        "hardware_claim_count": "prompt and spec invariant",
        "sampler_runtime_ready_score": "computed readiness gates",
        "protected_files_unchanged": "git diff and file hashes",
        "preconditions_checked": "pre-chain preregistration hash receipt",
        "inference_substrate": "prompt and spec invariant",
        "verifier_is_oracle": "exact finite target",
        "field_provenance": "this provenance map",
        "field_principles": "OpenSpec required field principles",
        "test_commands": "external command receipts",
        "test_exit_codes": "external command receipts",
        "duration_s": "wall-clock measurement",
        "reproducibility_checksum": "deterministic artifact hash",
        "honest_verdict": "computed verdict",
    }
    return {
        field: {"source": sources[field], "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_commands(command_receipts: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("command", "")) for row in command_receipts]


def _test_exit_codes(command_receipts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        str(row.get("command", "")): int(row.get("exit_code", -999)) for row in command_receipts
    }


def sampler_runtime_ready_score(artifact: Mapping[str, Any]) -> float:
    ledger = dict(artifact.get("task_owned_and_preexisting_nonzero_command_ledger") or {})
    decision = dict(artifact.get("equivalence_bounds_and_decision") or {}).get("decision")
    gates = [
        decision in {"positive", "negative", "equivalence_supported"},
        artifact.get("default_off_preserved") is True,
        type(artifact.get("hardware_claim_count")) is int
        and artifact.get("hardware_claim_count") == 0,
        dict(artifact.get("treatment_activation_score") or {}).get("activation_passed") is True,
        dict(artifact.get("multimodal_positive_control") or {}).get("passed") is True,
        dict(artifact.get("arm_support_matrix") or {}).get("all_main_fixture_arms_supported")
        is True,
        dict(artifact.get("fallback_parity_control") or {}).get("exact_replay_match") is True,
        dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True,
        ledger.get("all_task_owned_commands_passed") is True,
    ]
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    ledger = dict(artifact.get("task_owned_and_preexisting_nonzero_command_ledger") or {})
    decision = dict(artifact.get("equivalence_bounds_and_decision") or {}).get("decision")
    if ledger.get("task_owned_failure_count", 0) > 0:
        return "blocked"
    if decision == "instrument_failure":
        return "instrument_failure"
    if decision == "equivalence_supported" and sampler_runtime_ready_score(artifact) == 1.0:
        return "complete_equivalence_supported"
    if decision == "positive" and sampler_runtime_ready_score(artifact) == 1.0:
        return "complete_positive"
    if decision == "negative" and sampler_runtime_ready_score(artifact) == 1.0:
        return "complete_negative"
    return "complete_inconclusive"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    ledger = dict(artifact.get("task_owned_and_preexisting_nonzero_command_ledger") or {})
    decision = dict(artifact.get("equivalence_bounds_and_decision") or {}).get("decision")
    task_failures = ledger.get("task_owned_failures", [])
    prefix = status(artifact)
    return (
        f"{prefix}: decision={decision}; "
        f"activation={dict(artifact.get('treatment_activation_score') or {}).get('activation_passed')}; "
        f"positive_control={dict(artifact.get('multimodal_positive_control') or {}).get('passed')}; "
        f"default_off={artifact.get('default_off_preserved') is True}; "
        f"hardware_claim_count={artifact.get('hardware_claim_count')}; "
        f"task_owned_failures={json.dumps(task_failures, sort_keys=True)}"
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    command_receipts: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
) -> JsonDict:
    matrix = preregistered_fixture_seed_budget_matrix(root)
    preconditions = preconditions_checked(root, matrix)
    chains = _measure_main_chains(root)
    controls = unsupported_or_failed_cells(root, chains)
    activation = treatment_activation_score(chains)
    positive = multimodal_positive_control(chains, activation)
    artifact: JsonDict = {
        "status": "blocked",
        "upstream_paths_hashes_and_determinations": upstream_paths_hashes_and_determinations(root),
        "literature_control_path_and_hash": literature_control_path_and_hash(root),
        "preregistered_fixture_seed_budget_matrix": matrix,
        "exact_reference_distribution_receipts": exact_reference_distribution_receipts(root),
        "arm_support_matrix": arm_support_matrix(chains, controls),
        "matched_arm_configuration": matched_arm_configuration(root),
        "jump_proposal_acceptance_and_transition_counts": jump_proposal_acceptance_and_transition_counts(
            chains
        ),
        "treatment_activation_score": activation,
        "multimodal_positive_control": positive,
        "fallback_parity_control": fallback_parity_control(chains),
        "distribution_quality_by_fixture_arm": distribution_quality_by_fixture_arm(
            chains, positive
        ),
        "mode_coverage_ess_autocorrelation_by_fixture_arm": mode_coverage_ess_autocorrelation_by_fixture_arm(
            chains
        ),
        "wall_and_transition_costs": wall_and_transition_costs(chains),
        "paired_intervals": paired_intervals(chains),
        "equivalence_bounds_and_decision": {},
        "unsupported_or_failed_cells": controls,
        "task_owned_and_preexisting_nonzero_command_ledger": task_owned_and_preexisting_nonzero_command_ledger(
            command_receipts
        ),
        "default_off_preserved": default_off_preserved(root),
        "hardware_claim_count": 0,
        "sampler_runtime_ready_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(root),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": _test_commands(command_receipts),
        "test_exit_codes": _test_exit_codes(command_receipts),
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: pending",
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "schema": SCHEMA,
    }
    artifact["equivalence_bounds_and_decision"] = equivalence_bounds_and_decision(artifact)
    artifact["sampler_runtime_ready_score"] = sampler_runtime_ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    output_path: Path,
    root: Path = REPO_ROOT,
    command_receipts: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
) -> JsonDict:
    artifact = build_artifact(
        root=root,
        command_receipts=command_receipts,
        duration_s=duration_s,
        run_date=run_date,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required: {missing}")
    if (
        type(artifact.get("hardware_claim_count")) is not int
        or artifact.get("hardware_claim_count") != 0
    ):
        raise ValueError("hardware_claim_count")
    if artifact.get("default_off_preserved") is not True:
        raise ValueError("default_off_preserved")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    decision = dict(artifact.get("equivalence_bounds_and_decision") or {}).get("decision")
    if dict(artifact.get("treatment_activation_score") or {}).get("activation_passed") is not True:
        if decision != "instrument_failure" or "null" in str(artifact.get("honest_verdict")):
            raise ValueError("inactive_treatment_instrument_failure")
    if artifact.get("equivalence_bounds_and_decision") != equivalence_bounds_and_decision(artifact):
        raise ValueError("equivalence_bounds_and_decision")
    if artifact.get("sampler_runtime_ready_score") != sampler_runtime_ready_score(artifact):
        raise ValueError("sampler_runtime_ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field in REQUIRED_ARTIFACT_FIELDS:
        row = provenance.get(field)
        if not isinstance(row, Mapping) or not row.get("source") or not row.get("principle"):
            raise ValueError("field_provenance")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _external_command_receipts() -> list[JsonDict]:
    receipt_path = Path(os.environ.get("CARNOT_6237_COMMAND_RECEIPTS", DEFAULT_RECEIPT_PATH))
    if not receipt_path.exists():
        return []
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("command receipt payload must be a list")
    return [dict(row) for row in payload]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    artifact = write_artifact(
        output_path=REPO_ROOT / RESULT_RELATIVE_PATH,
        root=REPO_ROOT,
        command_receipts=_external_command_receipts(),
        duration_s=time.perf_counter() - started,
        run_date=args.date,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
