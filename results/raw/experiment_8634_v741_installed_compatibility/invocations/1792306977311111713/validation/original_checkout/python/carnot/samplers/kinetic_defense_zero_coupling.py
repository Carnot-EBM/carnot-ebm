"""Exp 1561 kinetic-defense zero-coupling null-space audit.

This module implements the DT-MCMC-NULL falsifiable predicate on a planted
64-bit landscape with 15 independent four-bit verifier blocks and four inert
null bits. It compares single-site Metropolis-Hastings, single-site Glauber
Gibbs, and THRML 0.1.3 graph-color block-Gibbs semantics using the same
factorized simulator so the measured difference is the update rule rather than
Python/JAX runtime overhead.

Spec refs: REQ-SAMPLE-057, SCENARIO-SAMPLE-085.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1561_kinetic_defense_zero_coupling_test.json"

EXPERIMENT_ID = 1561
RUN_DATE = "20260508"
SCHEMA = "kinetic_defense_zero_coupling_v1"
SAMPLERS = ("mh", "single_site_gibbs", "thrml_block_gibbs")
TERMINAL_VERDICT_PREFIXES = ("complete:", "complete_", "complete-")
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "kinetic_defense_in_depth_validated",
    "mh_hitting_time_steps_per_block",
    "single_site_gibbs_hitting_time_steps_per_block",
    "thrml_block_gibbs_hitting_time_steps_per_block",
    "thrml_security_parity_with_single_site_gibbs",
    "p_n_at_k100_mh",
    "p_n_at_k100_single_site_gibbs",
    "p_n_at_k100_thrml_block_gibbs",
    "blockers",
    "honest_verdict",
}


@dataclass(frozen=True)
class ZeroCouplingConfig:
    """Configuration for the planted verifier-null-space experiment."""

    n_bits: int = 64
    n_blocks: int = 15
    block_size: int = 4
    free_bits: int = 4
    reward: float = 10.0
    temperature: float = 1.0
    n_chains: int = 10_000
    checkpoints: tuple[int, ...] = (10, 50, 100, 500, 1000)
    seed: int = 1561
    single_block_max_steps: int = 1000


def target_block_state(config: ZeroCouplingConfig) -> int:
    """Return the integer encoding of the all-ones verifier target block."""

    return (1 << config.block_size) - 1


def energy_from_block_states(block_states: np.ndarray, config: ZeroCouplingConfig) -> np.ndarray:
    """Return one energy value per chain for encoded verifier-block states."""

    states = np.asarray(block_states, dtype=np.uint8)
    satisfied = states == target_block_state(config)
    return -float(config.reward) * satisfied.sum(axis=1, dtype=np.int16).astype(np.float64)


def in_null_space(block_states: np.ndarray, config: ZeroCouplingConfig) -> np.ndarray:
    """Return whether each chain is in the planted null space N."""

    states = np.asarray(block_states, dtype=np.uint8)
    return np.all(states == target_block_state(config), axis=1)


def _sigmoid_reward(config: ZeroCouplingConfig) -> float:
    return float(1.0 / (1.0 + np.exp(-config.reward / config.temperature)))


def _gibbs_update(
    states: np.ndarray,
    bit_indices: np.ndarray,
    rng: np.random.Generator,
    config: ZeroCouplingConfig,
) -> np.ndarray:
    target = np.uint8(target_block_state(config))
    full_mask = np.uint8(target_block_state(config))
    masks = np.left_shift(np.uint8(1), bit_indices).astype(np.uint8)
    other_masks = (~masks & full_mask).astype(np.uint8)
    other_bits_match = (states & other_masks) == (target & other_masks)
    probabilities = np.where(other_bits_match, _sigmoid_reward(config), 0.5)
    sampled_target = rng.random(states.shape) < probabilities
    set_target = states | masks
    clear_target = states & other_masks
    return np.where(sampled_target, set_target, clear_target).astype(np.uint8)


def _mh_update(
    states: np.ndarray,
    bit_indices: np.ndarray,
    rng: np.random.Generator,
    config: ZeroCouplingConfig,
) -> np.ndarray:
    target = target_block_state(config)
    masks = np.left_shift(np.uint8(1), bit_indices).astype(np.uint8)
    proposed = (states ^ masks).astype(np.uint8)
    current_target = states == target
    proposed_target = proposed == target
    uphill = current_target & ~proposed_target
    accept = np.ones(states.shape, dtype=bool)
    uphill_count = int(np.count_nonzero(uphill))
    uphill_acceptance = float(np.exp(-config.reward / config.temperature))
    accept[uphill] = rng.random(uphill_count) < uphill_acceptance
    return np.where(accept, proposed, states).astype(np.uint8)


def _single_site_step(
    sampler: str,
    states: np.ndarray,
    rng: np.random.Generator,
    config: ZeroCouplingConfig,
) -> np.ndarray:
    bit_indices = rng.integers(0, config.block_size, size=states.shape, dtype=np.uint8)
    if sampler == "mh":
        return _mh_update(states, bit_indices, rng, config)
    return _gibbs_update(states, bit_indices, rng, config)


def _thrml_color_step(
    states: np.ndarray,
    step_index: int,
    rng: np.random.Generator,
    config: ZeroCouplingConfig,
) -> np.ndarray:
    bit_index = np.uint8(step_index % config.block_size)
    bit_indices = np.full(states.shape, bit_index, dtype=np.uint8)
    return _gibbs_update(states, bit_indices, rng, config)


def simulate_single_block_hitting(
    sampler: str,
    config: ZeroCouplingConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    """Estimate first-hit steps for one four-bit verifier block.

    `mh` and `single_site_gibbs` use random single-site bit selection. The
    THRML lane uses graph-color semantics: one deterministic bit color is
    updated per step, matching THRML block-Gibbs color phases for a K4 verifier
    block while parallelizing independent blocks only as a compute detail.
    """

    rng = np.random.default_rng(seed)
    states = np.zeros(config.n_chains, dtype=np.uint8)
    first_hit = np.zeros(config.n_chains, dtype=np.int32)
    active = np.ones(config.n_chains, dtype=bool)
    target = target_block_state(config)

    for step in range(1, config.single_block_max_steps + 1):
        if sampler == "thrml_block_gibbs":
            states = _thrml_color_step(states, step - 1, rng, config)
        else:
            states = _single_site_step(sampler, states, rng, config)
        newly_hit = active & (states == target)
        first_hit[newly_hit] = step
        active[newly_hit] = False
        if not bool(active.any()):
            break

    observed = first_hit[first_hit > 0]
    return {
        "sampler": sampler,
        "chains": int(config.n_chains),
        "mean_hitting_time_steps": _round_float(float(observed.mean())),
        "censored_fraction": _round_float(float(active.mean())),
        "max_observed_hitting_step": int(observed.max()),
        "target_block": format(target, f"0{config.block_size}b"),
    }


def _energy_distribution(energies: np.ndarray) -> list[dict[str, float | int]]:
    values, counts = np.unique(energies.astype(np.int16), return_counts=True)
    total = float(energies.shape[0])
    return [
        {
            "energy": float(value),
            "count": int(count),
            "fraction": _round_float(float(count) / total),
        }
        for value, count in zip(values, counts, strict=True)
    ]


def simulate_global_null_space(
    sampler: str,
    config: ZeroCouplingConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run the 15-block planted-null-space experiment for one sampler."""

    rng = np.random.default_rng(seed)
    states = np.zeros((config.n_chains, config.n_blocks), dtype=np.uint8)
    first_hit_sweep = np.zeros(config.n_chains, dtype=np.int32)
    ever_hit = np.zeros(config.n_chains, dtype=bool)
    checkpoints = set(config.checkpoints)
    max_sweep = max(config.checkpoints)
    current_mass: dict[str, float] = {}
    cumulative_mass: dict[str, float] = {}
    energy_distributions: dict[str, list[dict[str, float | int]]] = {}
    mean_hitting_sweeps_at_k: dict[str, float | None] = {}

    for sweep in range(1, max_sweep + 1):
        if sampler == "thrml_block_gibbs":
            for phase in range(config.block_size):
                states = _thrml_color_step(states, phase, rng, config)
        else:
            for _ in range(config.block_size):
                states = _single_site_step(sampler, states, rng, config)

        in_null = in_null_space(states, config)
        newly_hit = (first_hit_sweep == 0) & in_null
        first_hit_sweep[newly_hit] = sweep
        ever_hit |= in_null

        if sweep in checkpoints:
            key = str(sweep)
            energies = energy_from_block_states(states, config)
            observed = first_hit_sweep[first_hit_sweep > 0]
            current_mass[key] = _round_float(float(in_null.mean()))
            cumulative_mass[key] = _round_float(float(ever_hit.mean()))
            energy_distributions[key] = _energy_distribution(energies)
            mean_hitting_sweeps_at_k[key] = (
                _round_float(float(observed.mean())) if observed.size else None
            )

    observed_final = first_hit_sweep[first_hit_sweep > 0]
    return {
        "sampler": sampler,
        "chains": int(config.n_chains),
        "current_null_mass_by_sweep": current_mass,
        "cumulative_null_hit_mass_by_sweep": cumulative_mass,
        "energy_distributions": energy_distributions,
        "mean_hitting_sweeps_by_checkpoint": mean_hitting_sweeps_at_k,
        "mean_global_hitting_sweeps": _round_float(float(observed_final.mean())),
        "unhit_fraction_at_final_checkpoint": _round_float(float((first_hit_sweep == 0).mean())),
    }


def classify_kinetic_defense(
    *,
    mh_steps: float,
    gibbs_steps: float,
    thrml_steps: float,
) -> dict[str, Any]:
    """Classify the DT-MCMC-NULL acceptance gate from hitting-time estimates."""

    mh_faster_than_gibbs = mh_steps < gibbs_steps
    thrml_security_parity = thrml_steps >= gibbs_steps
    thrml_hits_at_mh_class_rate = thrml_steps <= mh_steps * 1.10
    validated = bool(mh_faster_than_gibbs and thrml_security_parity)
    mitigation = (
        "Do not use graph-color block-Gibbs as the security argument on sparse "
        "zero-coupling null-space plateaus without an added kinetic throttle: "
        "randomize color order with replacement, cap color phases per inference "
        "budget, or audit candidate-warm-start null-space mass before adoption."
    )
    falsification_note = (
        "THRML graph-color block-Gibbs reached the planted null space at an "
        "MH-class rate, so parallel color scheduling is an attack-surface risk "
        "on this zero-coupling predicate."
        if thrml_hits_at_mh_class_rate
        else "THRML block-Gibbs was faster than single-site Gibbs on this predicate."
    )
    return {
        "mh_faster_than_single_site_gibbs": bool(mh_faster_than_gibbs),
        "thrml_security_parity_with_single_site_gibbs": bool(thrml_security_parity),
        "thrml_hits_at_mh_class_rate": bool(thrml_hits_at_mh_class_rate),
        "kinetic_defense_in_depth_validated": validated,
        "falsification_note": None if validated else falsification_note,
        "mitigation": None if validated else mitigation,
        "honest_verdict": (
            "complete_kinetic_defense_validated"
            if validated
            else "complete_thrml_block_gibbs_falsifies_kinetic_security_parity"
        ),
    }


def probe_thrml_metadata(
    importer: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    """Record direct THRML import provenance for the block-Gibbs semantics lane."""

    try:
        thrml = importer("thrml")
    except Exception as exc:
        return {
            "thrml_import_ready": False,
            "thrml_version": None,
            "thrml_import_path": None,
            "thrml_import_error": f"{type(exc).__name__}: {exc}",
            "thrml_execution_mode": "local_graph_color_semantics_no_import",
            "hardware_claim_allowed": False,
        }
    return {
        "thrml_import_ready": True,
        "thrml_version": getattr(thrml, "__version__", "unknown"),
        "thrml_import_path": str(getattr(thrml, "__file__", "unknown")),
        "thrml_import_error": None,
        "thrml_execution_mode": "thrml_0_1_3_graph_color_semantics",
        "hardware_claim_allowed": False,
    }


def build_artifact(
    *,
    config: ZeroCouplingConfig,
    single_block_results: Mapping[str, Mapping[str, Any]],
    global_results: Mapping[str, Mapping[str, Any]],
    thrml_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate the terminal Exp 1561 artifact."""

    mh_steps = float(single_block_results["mh"]["mean_hitting_time_steps"])
    gibbs_steps = float(single_block_results["single_site_gibbs"]["mean_hitting_time_steps"])
    thrml_steps = float(single_block_results["thrml_block_gibbs"]["mean_hitting_time_steps"])
    gate = classify_kinetic_defense(
        mh_steps=mh_steps,
        gibbs_steps=gibbs_steps,
        thrml_steps=thrml_steps,
    )
    blockers = []
    if not gate["kinetic_defense_in_depth_validated"]:
        blockers.append(
            {
                "blocker": "thrml_security_parity_failed",
                "detail": str(gate["falsification_note"]),
            }
        )

    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "spec_refs": ["REQ-SAMPLE-057", "SCENARIO-SAMPLE-085"],
            "dt_predicate": "DT-MCMC-NULL",
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "hardware_claim_allowed": False,
            "thrml": dict(thrml_metadata),
        },
        "status": "complete",
        "n_bits": config.n_bits,
        "n_blocks": config.n_blocks,
        "block_size": config.block_size,
        "free_bits": config.free_bits,
        "target_block": format(target_block_state(config), f"0{config.block_size}b"),
        "reward": float(config.reward),
        "temperature": float(config.temperature),
        "n_chains": int(config.n_chains),
        "checkpoints": list(config.checkpoints),
        "sampler_results": {name: dict(global_results[name]) for name in SAMPLERS},
        "single_block_hitting_results": {name: dict(single_block_results[name]) for name in SAMPLERS},
        "mh_hitting_time_steps_per_block": mh_steps,
        "single_site_gibbs_hitting_time_steps_per_block": gibbs_steps,
        "thrml_block_gibbs_hitting_time_steps_per_block": thrml_steps,
        "p_n_at_k100_mh": float(global_results["mh"]["current_null_mass_by_sweep"]["100"]),
        "p_n_at_k100_single_site_gibbs": float(
            global_results["single_site_gibbs"]["current_null_mass_by_sweep"]["100"]
        ),
        "p_n_at_k100_thrml_block_gibbs": float(
            global_results["thrml_block_gibbs"]["current_null_mass_by_sweep"]["100"]
        ),
        "mh_faster_than_single_site_gibbs": gate["mh_faster_than_single_site_gibbs"],
        "thrml_security_parity_with_single_site_gibbs": gate[
            "thrml_security_parity_with_single_site_gibbs"
        ],
        "thrml_hits_at_mh_class_rate": gate["thrml_hits_at_mh_class_rate"],
        "kinetic_defense_in_depth_validated": gate["kinetic_defense_in_depth_validated"],
        "falsification_note": gate["falsification_note"],
        "mitigation": gate["mitigation"],
        "blockers": blockers,
        "honest_verdict": gate["honest_verdict"],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required terminal schema fields for Exp 1561."""

    missing = REQUIRED_ARTIFACT_FIELDS - artifact.keys()
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("status") != "complete":
        raise ValueError("status must be complete")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    output_path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads(output_path.read_text(encoding="utf-8"))


def _write_in_progress(path: str | Path, config: ZeroCouplingConfig) -> None:
    _write_json(
        path,
        {
            "metadata": {
                "experiment_id": EXPERIMENT_ID,
                "schema": SCHEMA,
                "run_date": RUN_DATE,
                "spec_refs": ["REQ-SAMPLE-057", "SCENARIO-SAMPLE-085"],
            },
            "status": "in_progress",
            "n_chains": int(config.n_chains),
            "honest_verdict": "in_progress_exp1561_kinetic_defense_audit",
        },
    )


def run_experiment(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    config: ZeroCouplingConfig = ZeroCouplingConfig(),
    importer: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    """Run Exp 1561 and write the terminal deliverable JSON."""

    _write_in_progress(output_path, config)
    thrml_metadata = probe_thrml_metadata(importer)
    single_block_results = {
        "mh": simulate_single_block_hitting("mh", config, seed=config.seed + 101),
        "single_site_gibbs": simulate_single_block_hitting(
            "single_site_gibbs", config, seed=config.seed + 102
        ),
        "thrml_block_gibbs": simulate_single_block_hitting(
            "thrml_block_gibbs", config, seed=config.seed + 103
        ),
    }
    global_results = {
        "mh": simulate_global_null_space("mh", config, seed=config.seed + 111),
        "single_site_gibbs": simulate_global_null_space(
            "single_site_gibbs", config, seed=config.seed + 112
        ),
        "thrml_block_gibbs": simulate_global_null_space(
            "thrml_block_gibbs", config, seed=config.seed + 113
        ),
    }
    artifact = build_artifact(
        config=config,
        single_block_results=single_block_results,
        global_results=global_results,
        thrml_metadata=thrml_metadata,
    )
    return _write_json(output_path, artifact)
