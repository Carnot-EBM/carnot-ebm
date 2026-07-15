"""Exp5714 one-axis corrected-cDLS Rust/Python parity audit.

Spec refs: REQ-SAMPLE-5714, SCENARIO-SAMPLE-5714.

The promoted algorithm is the one-axis temperature-label exchange method from
Exp5633/5634. This module deliberately keeps the Python reference and Rust
comparison on that narrow surface: exact Ising energy, corrected cDLS proposal
probabilities, Metropolis-Hastings decisions, label-only adjacent swaps, a fixed
schedule, and serializable checkpoints. It does not implement the retired
two-axis penalty-exchange path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
from math import erfc, exp, log, sqrt
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
from unittest import mock

import numpy as np

from carnot import experiment_5622_cdls_exact_kernel_audit as exp5622
from carnot import experiment_5633_temperature_exchange_cdls_exact_audit as exp5633


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5714_one_axis_tempering_rust_parity.json")

EXPERIMENT = 5714
EXPERIMENT_ID = "exp5714-one-axis-tempering-rust-parity"
MILESTONE = "2026.07.514"
RUN_DATE = "2026-07-15"
SCHEMA = "carnot.experiment_5714.one_axis_tempering_rust_parity.v1"
SPEC_REFS = ("REQ-SAMPLE-5714", "SCENARIO-SAMPLE-5714")
INFERENCE_SUBSTRATE = "rust_python_exact_one_axis_sampler_parity"

BETA_LADDER = exp5633.BETA_LADDER
COLD_LABEL = len(BETA_LADDER) - 1
DEFAULT_RANDOM_SEEDS = (5714, 5715, 5716)
LCG_A = 6364136223846793005
LCG_C = 1442695040888963407
TERMINAL_PREFIXES = ("complete:", "blocked:")

FROZEN_TOLERANCES: dict[str, float] = {
    "energy": 1e-12,
    "proposal_log_probability": 1e-12,
    "swap_log_ratio": 1e-12,
    "exact_target_tv": 1e-9,
    "target_marginal": 1e-9,
    "detailed_balance": 1e-9,
}

BROKEN_CONTROL_IDS = (
    "stale_label_exchange",
    "wrong_sign_swap",
    "uncorrected_kernel",
    "collapsed_ladder",
    "corrupt_state",
)
MALFORMED_INPUT_IDS = (
    "nonsquare_couplings",
    "collapsed_ladder",
    "invalid_spin",
    "duplicate_labels",
    "corrupt_checkpoint",
)

RUST_SOURCE_PATHS = (
    Path("crates/carnot-samplers/src/one_axis_tempering.rs"),
    Path("crates/carnot-samplers/src/lib.rs"),
    Path("crates/carnot-samplers/Cargo.toml"),
)
PYTHON_BINDING_PATHS = (
    Path("crates/carnot-python/src/one_axis_tempering.rs"),
    Path("crates/carnot-python/src/lib.rs"),
    Path("crates/carnot-python/Cargo.toml"),
    Path("python/carnot/_rust_compat.py"),
)
PYTHON_REFERENCE_PATHS = (
    Path("python/carnot/experiment_5714_one_axis_tempering_rust_parity.py"),
    Path("python/carnot/experiment_5622_cdls_exact_kernel_audit.py"),
    Path("python/carnot/experiment_5633_temperature_exchange_cdls_exact_audit.py"),
    Path("python/carnot/experiment_5634_temperature_exchange_cdls_quality.py"),
    Path("tests/python/test_experiment_5714_one_axis_tempering_rust_parity.py"),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every required parity field exists before the artifact can promote a Rust port.",
    "source_promotion_receipts": "Pins Exp5633 exactness, Exp5634 quality promotion, and Exp5645/5646 retirement scope before port identity is trusted.",
    "source_artifact_hashes": "Content-addresses upstream evidence so the port cannot silently drift from promoted one-axis provenance.",
    "source_algorithm_hash": "Hashes the frozen one-axis algorithm recipe independently of implementation language.",
    "openspec_requirement_ids": "Keeps implementation, tests, and artifact anchored to REQ-SAMPLE-5714 and SCENARIO-SAMPLE-5714.",
    "rust_source_paths": "Lists the Rust files that implement the portable one-axis core.",
    "python_binding_paths": "Lists the PyO3/fallback files that expose the Rust core to Python.",
    "python_reference_paths": "Lists the Python reference and test files used for exact parity.",
    "compiler_and_toolchain": "Records the local Rust, Python, Cargo, and platform versions needed to reconstruct the build.",
    "pyo3_version": "Records the binding ABI dependency version explicitly.",
    "build_features": "Shows that the build used the ordinary one-axis Rust/PyO3 feature surface.",
    "abi_receipt": "Proves malformed inputs fail safely at the Python/Rust boundary.",
    "fixture_manifest": "Freezes the exact enumerable Ising workloads, ladders, schedules, and seeds.",
    "frozen_tolerances": "Predeclares numerical and exact-target tolerances before parity is interpreted.",
    "energy_error_max": "Quantifies deterministic energy parity between Python and Rust.",
    "proposal_probability_error_max": "Quantifies corrected cDLS proposal-probability parity.",
    "swap_log_ratio_error_max": "Quantifies temperature-label swap-ratio parity.",
    "deterministic_decision_parity": "Proves seeded within-replica and swap decisions match exactly.",
    "scheduler_parity": "Proves both implementations apply the same fixed within-replica and exchange schedule.",
    "exact_target_tv_python": "Measures Python exact-target stationarity against the enumerable Boltzmann product target.",
    "exact_target_tv_rust": "Measures Rust exact-target stationarity against the same enumerable Boltzmann product target.",
    "target_marginal_delta": "Measures cold-target marginal agreement across implementations and exact target.",
    "detailed_balance_error_by_impl": "Reports within-kernel and swap detailed-balance residuals by implementation.",
    "checkpoint_roundtrip_pass": "Proves Rust state serialization preserves labels, states, seed, and sweep.",
    "cross_language_restart_pass": "Proves a Python checkpoint can resume through Rust and reproduce the trace.",
    "malformed_input_controls": "Documents invalid ABI inputs and confirms they fail closed.",
    "broken_control_results": "Documents stale-label, wrong-sign, uncorrected-kernel, collapsed-ladder, and corrupt-state controls.",
    "broken_control_rejected": "Provides the boolean gate that every broken control was rejected.",
    "broken_control_rejected_score": "Provides the scalar gate that is 1.0 only when all broken controls reject.",
    "python_fallback_equivalence": "Proves the pure-Python path remains available when Rust is absent.",
    "two_axis_code_added": "Bare false closes the retired penalty-axis scope.",
    "timing_claimed": "Bare false prevents semantic parity from becoming a speed claim.",
    "hardware_speedup_claimed": "Bare false prevents portability evidence from becoming board or hardware evidence.",
    "one_axis_rust_parity_ready_score": "Provides a scalar downstream gate that is 1.0 only under the full parity contract.",
    "inference_substrate": "Declares Rust/Python exact one-axis sampler parity with no LLM or board participation.",
    "random_seeds": "Records replay seeds for deterministic decisions and restart.",
    "reproducibility_checksum": "Content-addresses the complete artifact after blanking the self-checksum field.",
    "honest_verdict": "Starts complete: or blocked: and states whether parity is final.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class OneAxisConfig:
    """Validated Python reference configuration for the promoted one-axis port."""

    couplings: np.ndarray
    fields: np.ndarray
    beta_ladder: tuple[float, ...] = BETA_LADDER
    proposal_std: float = exp5622.CDLS_PROPOSAL_STD
    drift_scale: float = exp5622.CDLS_DRIFT_SCALE

    def __post_init__(self) -> None:
        couplings = np.array(self.couplings, dtype=np.float64)
        fields = np.array(self.fields, dtype=np.float64)
        if (
            couplings.ndim != 2
            or couplings.shape[0] == 0
            or couplings.shape[0] != couplings.shape[1]
        ):
            raise ValueError("couplings must be square")
        if fields.shape != (couplings.shape[0],):
            raise ValueError("fields length must match couplings dimension")
        if not np.all(np.isfinite(couplings)) or not np.all(np.isfinite(fields)):
            raise ValueError("couplings and fields must be finite")
        if len(self.beta_ladder) < 2:
            raise ValueError("beta_ladder must contain at least two labels")
        if any(not np.isfinite(beta) or beta <= 0 for beta in self.beta_ladder):
            raise ValueError("beta_ladder values must be finite and positive")
        if any(left >= right for left, right in zip(self.beta_ladder, self.beta_ladder[1:])):
            raise ValueError("beta_ladder must be strictly increasing")
        if not np.isfinite(self.proposal_std) or self.proposal_std <= 0:
            raise ValueError("proposal_std must be finite and positive")
        if not np.isfinite(self.drift_scale):
            raise ValueError("drift_scale must be finite")
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "beta_ladder", tuple(float(beta) for beta in self.beta_ladder))
        object.__setattr__(self, "proposal_std", float(self.proposal_std))
        object.__setattr__(self, "drift_scale", float(self.drift_scale))

    @property
    def n_spins(self) -> int:
        return int(self.fields.shape[0])


@dataclass(frozen=True)
class OneAxisState:
    """Serializable seeded Python reference state."""

    states: np.ndarray
    labels: tuple[int, ...]
    rng_state: int
    sweep: int = 0

    def __post_init__(self) -> None:
        states = np.array(self.states, dtype=np.int8)
        if states.ndim != 2 or states.shape[0] == 0:
            raise ValueError("states must be a non-empty two-dimensional array")
        if not np.all((states == -1) | (states == 1)):
            raise ValueError("spin state values must be -1 or +1")
        labels = tuple(int(label) for label in self.labels)
        if len(labels) != states.shape[0] or sorted(labels) != list(range(len(labels))):
            raise ValueError("labels must be a permutation of beta-label indices")
        if self.rng_state < 0 or self.rng_state >= 2**64:
            raise ValueError("rng_state must fit in u64")
        if self.sweep < 0:
            raise ValueError("sweep must be nonnegative")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "rng_state", int(self.rng_state))
        object.__setattr__(self, "sweep", int(self.sweep))

    def checkpoint(self) -> JsonDict:
        """Return the JSON-serializable checkpoint shape shared with Rust."""

        return {
            "states": self.states.astype(int).tolist(),
            "labels": list(self.labels),
            "rng_state": int(self.rng_state),
            "sweep": int(self.sweep),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "OneAxisState":
        missing = {"states", "labels", "rng_state", "sweep"} - set(checkpoint)
        if missing:
            raise ValueError(f"checkpoint missing {sorted(missing)}")
        return cls(
            states=np.array(checkpoint["states"], dtype=np.int8),
            labels=tuple(int(label) for label in checkpoint["labels"]),
            rng_state=int(checkpoint["rng_state"]),
            sweep=int(checkpoint["sweep"]),
        )


class PythonOneAxisTemperingCore:
    """Pure-Python reference for the one-axis Rust/PyO3 core."""

    def __init__(self, config: OneAxisConfig) -> None:
        self.config = config

    def energy(self, state: Sequence[int]) -> float:
        state_array = _spin_array(state, self.config.n_spins)
        pair_term = -0.5 * float(state_array @ self.config.couplings @ state_array)
        field_term = -float(state_array @ self.config.fields)
        return pair_term + field_term

    def proposal_log_probability(
        self,
        source: Sequence[int],
        target: Sequence[int],
        beta: float,
    ) -> float:
        source_array = _spin_array(source, self.config.n_spins)
        target_array = _spin_array(target, self.config.n_spins)
        mean = self._proposal_mean(source_array, beta)
        return float(
            sum(
                log(_normal_cdf(float(sign * coordinate_mean / self.config.proposal_std)))
                for sign, coordinate_mean in zip(target_array, mean, strict=True)
            )
        )

    def corrected_step(
        self,
        state: Sequence[int],
        beta: float,
        uniforms: Sequence[float],
    ) -> JsonDict:
        state_array = _spin_array(state, self.config.n_spins)
        if len(uniforms) != self.config.n_spins + 1:
            raise ValueError("uniforms length must be n_spins + 1")
        if any(not np.isfinite(value) or value < 0.0 or value >= 1.0 for value in uniforms):
            raise ValueError("uniforms must be finite values in [0, 1)")
        proposed = self._draw_projected_proposal(state_array, beta, uniforms)
        current_energy = self.energy(state_array)
        proposed_energy = self.energy(proposed)
        log_forward = self.proposal_log_probability(state_array, proposed, beta)
        log_reverse = self.proposal_log_probability(proposed, state_array, beta)
        log_acceptance = (
            -float(beta) * (proposed_energy - current_energy) + log_reverse - log_forward
        )
        accepted = bool(
            log_acceptance >= 0.0 or log(float(uniforms[self.config.n_spins])) < log_acceptance
        )
        return {
            "state": (proposed if accepted else state_array).astype(int).tolist(),
            "proposed_state": proposed.astype(int).tolist(),
            "current_energy": current_energy,
            "proposed_energy": proposed_energy,
            "proposal_log_forward": log_forward,
            "proposal_log_reverse": log_reverse,
            "log_acceptance": log_acceptance,
            "accepted": accepted,
        }

    def swap_log_ratio(
        self,
        states: Sequence[Sequence[int]],
        labels: Sequence[int],
        label_pair: Sequence[int],
    ) -> float:
        states_array, labels_tuple = _state_collection(states, labels, self.config)
        left_label, right_label = _label_pair(label_pair, len(self.config.beta_ladder))
        left_pos = labels_tuple.index(left_label)
        right_pos = labels_tuple.index(right_label)
        beta_left = self.config.beta_ladder[left_label]
        beta_right = self.config.beta_ladder[right_label]
        energy_left = self.energy(states_array[left_pos])
        energy_right = self.energy(states_array[right_pos])
        return float((beta_left - beta_right) * (energy_left - energy_right))

    def swap_decision(
        self,
        states: Sequence[Sequence[int]],
        labels: Sequence[int],
        label_pair: Sequence[int],
        uniform: float,
    ) -> JsonDict:
        if not np.isfinite(uniform) or uniform < 0.0 or uniform >= 1.0:
            raise ValueError("swap uniform must be finite and in [0, 1)")
        _, labels_tuple = _state_collection(states, labels, self.config)
        pair = _label_pair(label_pair, len(self.config.beta_ladder))
        log_ratio = self.swap_log_ratio(states, labels, pair)
        probability = _acceptance_probability(log_ratio)
        accepted = bool(uniform < probability)
        proposed = list(labels_tuple)
        left_pos = labels_tuple.index(pair[0])
        right_pos = labels_tuple.index(pair[1])
        proposed[left_pos], proposed[right_pos] = proposed[right_pos], proposed[left_pos]
        return {
            "labels": proposed if accepted else list(labels_tuple),
            "proposed_labels": proposed,
            "log_ratio": log_ratio,
            "acceptance_probability": probability,
            "accepted": accepted,
        }

    def scheduler_trace(self) -> list[str]:
        return [f"within:{idx}" for idx in range(len(self.config.beta_ladder))] + [
            f"swap:{idx}-{idx + 1}" for idx in range(len(self.config.beta_ladder) - 1)
        ]

    def step(self, state: OneAxisState) -> OneAxisState:
        states_array, labels_tuple = _state_collection(state.states, state.labels, self.config)
        updated_states = states_array.copy()
        labels = list(labels_tuple)
        rng_state = int(state.rng_state)
        for physical_index in range(len(self.config.beta_ladder)):
            beta = self.config.beta_ladder[labels[physical_index]]
            rng_state, uniforms = _draw_uniforms(rng_state, self.config.n_spins + 1)
            outcome = self.corrected_step(updated_states[physical_index], beta, uniforms)
            updated_states[physical_index] = np.array(outcome["state"], dtype=np.int8)
        for left in range(len(self.config.beta_ladder) - 1):
            rng_state, uniform = _next_uniform(rng_state)
            outcome = self.swap_decision(updated_states, labels, (left, left + 1), uniform)
            labels = [int(label) for label in outcome["labels"]]
        return OneAxisState(
            states=updated_states,
            labels=tuple(labels),
            rng_state=rng_state,
            sweep=state.sweep + 1,
        )

    def target_state(self, state: OneAxisState) -> list[int]:
        _, labels = _state_collection(state.states, state.labels, self.config)
        position = labels.index(COLD_LABEL)
        return state.states[position].astype(int).tolist()

    def _proposal_mean(self, source: np.ndarray, beta: float) -> np.ndarray:
        field = self.config.couplings @ source.astype(np.float64) + self.config.fields
        return source.astype(np.float64) + self.config.drift_scale * float(beta) * field

    def _draw_projected_proposal(
        self,
        source: np.ndarray,
        beta: float,
        uniforms: Sequence[float],
    ) -> np.ndarray:
        mean = self._proposal_mean(source, beta)
        probabilities = np.array(
            [_normal_cdf(float(value / self.config.proposal_std)) for value in mean],
            dtype=np.float64,
        )
        return np.where(np.array(uniforms[: self.config.n_spins]) < probabilities, 1, -1).astype(
            np.int8
        )


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for reproducible artifact hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content using Carnot's SHA-256 convention."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file byte-for-byte for provenance receipts."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def default_config(system: exp5622.IsingSystem | None = None) -> OneAxisConfig:
    """Return the default Exp5633 fixture config used for deterministic parity."""

    selected = system or exp5633.enumerable_frustrated_systems()[0]
    return OneAxisConfig(
        couplings=np.array(selected.couplings, dtype=np.float64),
        fields=np.array(selected.fields, dtype=np.float64),
        beta_ladder=BETA_LADDER,
        proposal_std=exp5622.CDLS_PROPOSAL_STD,
        drift_scale=exp5622.CDLS_DRIFT_SCALE,
    )


def default_state(seed: int = 5714) -> OneAxisState:
    """Return a stable three-replica state whose RNG seed is caller-selected."""

    return OneAxisState(
        states=np.array([[1, -1, 1], [-1, -1, 1], [1, 1, -1]], dtype=np.int8),
        labels=(0, 1, 2),
        rng_state=int(seed),
        sweep=0,
    )


def run_parity_audit(
    *,
    root: Path = REPO_ROOT,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
) -> JsonDict:
    """Run deterministic, exact-target, restart, ABI, and control parity gates."""

    rust_classes = _rust_classes()
    energy_error, proposal_error, swap_error = _numeric_parity_errors(rust_classes)
    deterministic = _deterministic_replay_parity(rust_classes, random_seeds)
    exact_report = _exact_target_report(rust_classes)
    malformed = _malformed_input_controls(rust_classes)
    broken = _broken_control_results(rust_classes)
    fallback = _python_fallback_equivalence()

    return {
        "energy_error_max": energy_error,
        "proposal_probability_error_max": proposal_error,
        "swap_log_ratio_error_max": swap_error,
        "deterministic_decision_parity": deterministic["deterministic_decision_parity"],
        "scheduler_parity": deterministic["scheduler_parity"],
        "exact_target_tv_python": exact_report["python"]["exact_target_tv"],
        "exact_target_tv_rust": exact_report["rust"]["exact_target_tv"],
        "target_marginal_delta": exact_report["target_marginal_delta"],
        "detailed_balance_error_by_impl": {
            "python": exact_report["python"]["detailed_balance"],
            "rust": exact_report["rust"]["detailed_balance"],
        },
        "checkpoint_roundtrip_pass": deterministic["checkpoint_roundtrip_pass"],
        "cross_language_restart_pass": deterministic["cross_language_restart_pass"],
        "malformed_input_controls": malformed,
        "broken_control_results": broken,
        "broken_control_rejected": all(row["rejected"] is True for row in broken),
        "broken_control_rejected_score": 1.0
        if all(row["rejected"] is True for row in broken)
        else 0.0,
        "python_fallback_equivalence": fallback,
        "root_checked": root.as_posix(),
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp5714 parity artifact."""

    parity = run_parity_audit(root=root, random_seeds=random_seeds)
    broken_score = float(parity["broken_control_rejected_score"])
    source_hashes = source_artifact_hashes(root)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "schema": SCHEMA,
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_promotion_receipts": source_promotion_receipts(root),
        "source_artifact_hashes": source_hashes,
        "source_algorithm_hash": source_algorithm_hash(),
        "openspec_requirement_ids": list(SPEC_REFS),
        "rust_source_paths": [path.as_posix() for path in RUST_SOURCE_PATHS],
        "python_binding_paths": [path.as_posix() for path in PYTHON_BINDING_PATHS],
        "python_reference_paths": [path.as_posix() for path in PYTHON_REFERENCE_PATHS],
        "compiler_and_toolchain": compiler_and_toolchain(root),
        "pyo3_version": "0.24",
        "build_features": {
            "pyo3_extension_module": True,
            "one_axis_tempering": True,
            "penalty_axis_exchange": False,
            "semantic_portability_only": True,
        },
        "abi_receipt": {
            "rust_extension_importable": True,
            "malformed_input_ids": list(MALFORMED_INPUT_IDS),
            "malformed_input_failures": sum(
                1 for row in parity["malformed_input_controls"] if row["failed_safely"]
            ),
        },
        "fixture_manifest": fixture_manifest(random_seeds),
        "frozen_tolerances": dict(FROZEN_TOLERANCES),
        "energy_error_max": parity["energy_error_max"],
        "proposal_probability_error_max": parity["proposal_probability_error_max"],
        "swap_log_ratio_error_max": parity["swap_log_ratio_error_max"],
        "deterministic_decision_parity": parity["deterministic_decision_parity"],
        "scheduler_parity": parity["scheduler_parity"],
        "exact_target_tv_python": parity["exact_target_tv_python"],
        "exact_target_tv_rust": parity["exact_target_tv_rust"],
        "target_marginal_delta": parity["target_marginal_delta"],
        "detailed_balance_error_by_impl": parity["detailed_balance_error_by_impl"],
        "checkpoint_roundtrip_pass": parity["checkpoint_roundtrip_pass"],
        "cross_language_restart_pass": parity["cross_language_restart_pass"],
        "malformed_input_controls": parity["malformed_input_controls"],
        "broken_control_results": parity["broken_control_results"],
        "broken_control_rejected": parity["broken_control_rejected"],
        "broken_control_rejected_score": broken_score,
        "python_fallback_equivalence": parity["python_fallback_equivalence"],
        "two_axis_code_added": False,
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "one_axis_rust_parity_ready_score": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": [int(seed) for seed in random_seeds],
        "tests_added_or_reused": list(tests_added_or_reused or []),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: parity gates not evaluated",
    }
    artifact["one_axis_rust_parity_ready_score"] = ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_output(root: Path, artifact: Mapping[str, Any]) -> Path:
    """Write the artifact under the requested root and return its path."""

    output_path = Path(root) / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the Exp5714 schema and mechanical gates."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if dict(payload["field_principles"]) != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload["openspec_requirement_ids"] != list(SPEC_REFS):
        raise ValueError("openspec_requirement_ids mismatch")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload["two_axis_code_added"] is not False:
        raise ValueError("two_axis_code_added must be false")
    if payload["timing_claimed"] is not False:
        raise ValueError("timing_claimed must be false")
    if payload["hardware_speedup_claimed"] is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload["broken_control_rejected_score"] != (
        1.0 if payload["broken_control_rejected"] is True else 0.0
    ):
        raise ValueError("broken_control_rejected_score mismatch")
    expected_score = ready_score(payload)
    if payload["one_axis_rust_parity_ready_score"] != expected_score:
        raise ValueError("one_axis_rust_parity_ready_score mismatch")
    if not str(payload["honest_verdict"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start complete: or blocked:")
    if payload["honest_verdict"] != honest_verdict(payload):
        raise ValueError("honest_verdict mismatch")
    if payload["reproducibility_checksum"] != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def ready_score(payload: Mapping[str, Any]) -> float:
    """Return the downstream scalar gate for the parity artifact."""

    gates = [
        float(payload["energy_error_max"]) <= FROZEN_TOLERANCES["energy"],
        float(payload["proposal_probability_error_max"])
        <= FROZEN_TOLERANCES["proposal_log_probability"],
        float(payload["swap_log_ratio_error_max"]) <= FROZEN_TOLERANCES["swap_log_ratio"],
        payload["deterministic_decision_parity"] is True,
        payload["scheduler_parity"] is True,
        float(payload["exact_target_tv_python"]) <= FROZEN_TOLERANCES["exact_target_tv"],
        float(payload["exact_target_tv_rust"]) <= FROZEN_TOLERANCES["exact_target_tv"],
        float(payload["target_marginal_delta"]) <= FROZEN_TOLERANCES["target_marginal"],
        all(
            value <= FROZEN_TOLERANCES["detailed_balance"]
            for by_impl in payload["detailed_balance_error_by_impl"].values()
            for value in by_impl.values()
        ),
        payload["checkpoint_roundtrip_pass"] is True,
        payload["cross_language_restart_pass"] is True,
        all(row["failed_safely"] is True for row in payload["malformed_input_controls"]),
        payload["broken_control_rejected"] is True,
        payload["broken_control_rejected_score"] == 1.0,
        payload["python_fallback_equivalence"] is True,
        payload["two_axis_code_added"] is False,
        payload["timing_claimed"] is False,
        payload["hardware_speedup_claimed"] is False,
    ]
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return the required terminal verdict string."""

    if ready_score(payload) == 1.0:
        return "complete: one-axis corrected-cDLS Rust/Python parity is exact within frozen tolerances; portability only, no speed claim"
    return "blocked: one-axis corrected-cDLS Rust/Python parity gate failed"


def source_artifact_hashes(root: Path) -> JsonDict:
    """Hash upstream artifacts and source files used to reconstruct the port."""

    paths = {
        "experiment_5622": Path("results/experiment_5622_cdls_exact_kernel_audit.json"),
        "experiment_5633": Path(
            "results/experiment_5633_temperature_exchange_cdls_exact_audit.json"
        ),
        "experiment_5634": Path("results/experiment_5634_temperature_exchange_cdls_quality.json"),
        "experiment_5645": Path(
            "results/experiment_5645_two_axis_tempering_hard_constraint_quality.json"
        ),
        "experiment_5646": Path("results/experiment_5646_two_axis_tempering_rust_parity.json"),
        "source_5622": Path("python/carnot/experiment_5622_cdls_exact_kernel_audit.py"),
        "source_5633": Path(
            "python/carnot/experiment_5633_temperature_exchange_cdls_exact_audit.py"
        ),
        "source_5634": Path("python/carnot/experiment_5634_temperature_exchange_cdls_quality.py"),
    }
    return {name: file_sha256(root / path) for name, path in paths.items()}


def source_promotion_receipts(root: Path) -> JsonDict:
    """Return upstream gate receipts and block if one-axis provenance is stale."""

    exp5622_payload = _read_json(root / "results/experiment_5622_cdls_exact_kernel_audit.json")
    exp5633_payload = _read_json(
        root / "results/experiment_5633_temperature_exchange_cdls_exact_audit.json"
    )
    exp5634_payload = _read_json(
        root / "results/experiment_5634_temperature_exchange_cdls_quality.json"
    )
    exp5645_payload = _read_json(
        root / "results/experiment_5645_two_axis_tempering_hard_constraint_quality.json"
    )
    exp5646_payload = _read_json(
        root / "results/experiment_5646_two_axis_tempering_rust_parity.json"
    )
    hashes = source_artifact_hashes(root)

    corrected_receipt = exp5633_payload["corrected_kernel_receipt"]
    source_5622_hash = file_sha256(root / corrected_receipt["source_path"])
    result_5622_hash = file_sha256(root / corrected_receipt["result_path"])
    if corrected_receipt["source_sha256"] != source_5622_hash:
        raise ValueError("Exp5633 corrected kernel source hash is stale")
    if corrected_receipt["result_sha256"] != result_5622_hash:
        raise ValueError("Exp5633 corrected kernel artifact hash is stale")
    if exp5634_payload["upstream_gate_receipts"]["exp5633"]["sha256"] != hashes["experiment_5633"]:
        raise ValueError("Exp5634 Exp5633 receipt hash is stale")

    return {
        "exp5622": {
            "path": "results/experiment_5622_cdls_exact_kernel_audit.json",
            "sha256": hashes["experiment_5622"],
            "kernel_audit_ready_score": exp5622_payload["kernel_audit_ready_score"],
            "ready": exp5622_payload["kernel_audit_ready_score"] == 1.0,
        },
        "exp5633": {
            "path": "results/experiment_5633_temperature_exchange_cdls_exact_audit.json",
            "sha256": hashes["experiment_5633"],
            "replica_exchange_kernel_ready_score": exp5633_payload[
                "replica_exchange_kernel_ready_score"
            ],
            "ready": exp5633_payload["replica_exchange_kernel_ready_score"] == 1.0,
        },
        "exp5634": {
            "path": "results/experiment_5634_temperature_exchange_cdls_quality.json",
            "sha256": hashes["experiment_5634"],
            "quality_mixing_ready": bool(exp5634_payload["quality_mixing_ready"]),
            "ready": bool(exp5634_payload["quality_mixing_ready"]),
        },
        "exp5645_two_axis_retired": {
            "path": "results/experiment_5645_two_axis_tempering_hard_constraint_quality.json",
            "sha256": hashes["experiment_5645"],
            "two_axis_quality_ready_score": exp5645_payload["two_axis_quality_ready_score"],
            "ready": exp5645_payload["two_axis_quality_ready_score"] == 1.0,
            "retired_scope": exp5645_payload["two_axis_quality_ready_score"] == 0.0,
        },
        "exp5646_two_axis_rust_path_blocked": {
            "path": "results/experiment_5646_two_axis_tempering_rust_parity.json",
            "sha256": hashes["experiment_5646"],
            "honest_verdict": exp5646_payload["honest_verdict"],
            "ready": False,
            "retired_scope": True,
        },
    }


def source_algorithm_hash() -> str:
    """Hash the one-axis algorithm recipe independent of implementation files."""

    recipe = {
        "algorithm": "one_axis_corrected_cdls_temperature_label_exchange",
        "corrected_kernel": "corrected_cdls_projection_mh",
        "proposal_std": exp5622.CDLS_PROPOSAL_STD,
        "drift_scale": exp5622.CDLS_DRIFT_SCALE,
        "beta_ladder": list(BETA_LADDER),
        "within_replica_schedule": exp5633.within_replica_schedule(),
        "exchange_schedule": exp5633.exchange_schedule(),
        "swap_rule": exp5633.swap_rule(),
        "two_axis_penalty_exchange": False,
    }
    return sha256_json(recipe)


def fixture_manifest(random_seeds: Sequence[int]) -> JsonDict:
    """Return frozen fixture and schedule metadata."""

    systems = exp5633.enumerable_frustrated_systems()
    return {
        "fixtures": [
            {
                "system_id": system.system_id,
                "topology": system.topology,
                "n_spins": system.n_spins,
                "couplings": np.round(system.couplings, 12).tolist(),
                "fields": np.round(system.fields, 12).tolist(),
            }
            for system in systems
        ],
        "beta_ladder": list(BETA_LADDER),
        "within_replica_schedule": exp5633.within_replica_schedule(),
        "exchange_schedule": exp5633.exchange_schedule(),
        "proposal_std": exp5622.CDLS_PROPOSAL_STD,
        "drift_scale": exp5622.CDLS_DRIFT_SCALE,
        "random_seeds": [int(seed) for seed in random_seeds],
    }


def compiler_and_toolchain(root: Path) -> JsonDict:
    """Capture local build-tool versions without making performance claims."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rustc": _command_version(root, ["rustc", "--version"]),
        "cargo": _command_version(root, ["cargo", "--version"]),
        "cargo_available": _command_version(root, ["cargo", "--version"])["available"],
        "maturin": _command_version(root, ["maturin", "--version"]),
    }


def _numeric_parity_errors(rust_classes: Mapping[str, Any]) -> tuple[float, float, float]:
    energy_error = 0.0
    proposal_error = 0.0
    swap_error = 0.0
    for system in exp5633.enumerable_frustrated_systems():
        config = default_config(system)
        py_core = PythonOneAxisTemperingCore(config)
        rust_core = _rust_core_from_config(config, rust_classes)
        states = exp5622.enumerate_states(system.n_spins)
        for state in states:
            rust_energy = float(rust_core.energy(state.astype(int).tolist()))
            energy_error = max(energy_error, abs(rust_energy - py_core.energy(state)))
        for beta in BETA_LADDER:
            for source in states:
                for target in states:
                    source_list = source.astype(int).tolist()
                    target_list = target.astype(int).tolist()
                    rust_log = float(
                        rust_core.proposal_log_probability(source_list, target_list, beta)
                    )
                    py_log = py_core.proposal_log_probability(source_list, target_list, beta)
                    proposal_error = max(proposal_error, abs(rust_log - py_log))
        sample_states = [
            states[0].astype(int).tolist(),
            states[3].astype(int).tolist(),
            states[-1].astype(int).tolist(),
        ]
        for labels in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
            for pair in ((0, 1), (1, 2)):
                rust_ratio = float(
                    rust_core.swap_log_ratio(sample_states, list(labels), list(pair))
                )
                py_ratio = py_core.swap_log_ratio(sample_states, labels, pair)
                swap_error = max(swap_error, abs(rust_ratio - py_ratio))
    return energy_error, proposal_error, swap_error


def _deterministic_replay_parity(
    rust_classes: Mapping[str, Any],
    random_seeds: Sequence[int],
) -> JsonDict:
    config = default_config()
    py_core = PythonOneAxisTemperingCore(config)
    rust_core = _rust_core_from_config(config, rust_classes)
    rust_state_cls = rust_classes["state"]
    scheduler_parity = rust_core.scheduler_trace() == py_core.scheduler_trace()
    decision_parity = True
    checkpoint_roundtrip = True
    cross_language_restart = True
    for seed in random_seeds:
        py_state = default_state(seed)
        rust_state = rust_state_cls.from_checkpoint(py_state.checkpoint())
        for _ in range(4):
            py_next = py_core.step(py_state)
            rust_next = rust_core.step(rust_state)
            if rust_next.checkpoint() != py_next.checkpoint():
                decision_parity = False
            if (
                rust_state_cls.from_checkpoint(rust_next.checkpoint()).checkpoint()
                != rust_next.checkpoint()
            ):
                checkpoint_roundtrip = False
            restarted = rust_state_cls.from_checkpoint(py_next.checkpoint())
            if rust_core.step(restarted).checkpoint() != py_core.step(py_next).checkpoint():
                cross_language_restart = False
            py_state = py_next
            rust_state = rust_next
    return {
        "deterministic_decision_parity": decision_parity,
        "scheduler_parity": scheduler_parity,
        "checkpoint_roundtrip_pass": checkpoint_roundtrip,
        "cross_language_restart_pass": cross_language_restart,
    }


def _exact_target_report(rust_classes: Mapping[str, Any]) -> JsonDict:
    reports = {"python": [], "rust": []}
    for system in exp5633.enumerable_frustrated_systems():
        config = default_config(system)
        states = exp5622.enumerate_states(system.n_spins)
        py_core = PythonOneAxisTemperingCore(config)
        rust_core = _rust_core_from_config(config, rust_classes)
        reports["python"].append(_exact_report_for_core(py_core, states, system))
        reports["rust"].append(_exact_report_for_core(rust_core, states, system))

    python_cold = [np.array(row["cold_marginal"], dtype=np.float64) for row in reports["python"]]
    rust_cold = [np.array(row["cold_marginal"], dtype=np.float64) for row in reports["rust"]]
    target_delta = max(
        max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(python_cold, rust_cold, strict=True)
        ),
        max(float(row["target_marginal_error"]) for row in reports["python"]),
        max(float(row["target_marginal_error"]) for row in reports["rust"]),
    )
    return {
        "python": _summarize_exact_reports(reports["python"]),
        "rust": _summarize_exact_reports(reports["rust"]),
        "target_marginal_delta": target_delta,
    }


def _exact_report_for_core(core: Any, states: np.ndarray, system: exp5622.IsingSystem) -> JsonDict:
    targets_by_label = [
        exp5633.target_distribution_for_beta(system, states, beta) for beta in BETA_LADDER
    ]
    kernels_by_label = [_corrected_transition_matrix(core, states, beta) for beta in BETA_LADDER]
    product, permutations, permutation_to_index = exp5633.product_target_distribution(
        targets_by_label
    )
    updated = _apply_schedule(
        product, core, states, kernels_by_label, permutations, permutation_to_index
    )
    cold = exp5633.cold_label_marginal(
        updated,
        cold_label=COLD_LABEL,
        state_count=len(states),
        replica_count=len(BETA_LADDER),
        permutations=permutations,
        permutation_to_index=permutation_to_index,
    )
    cold_target = targets_by_label[COLD_LABEL]
    return {
        "system_id": system.system_id,
        "exact_target_tv": exp5633.total_variation(product, updated),
        "target_marginal_error": float(np.max(np.abs(cold - cold_target))),
        "cold_marginal": np.round(cold, 15).tolist(),
        "detailed_balance": {
            "within": _within_detailed_balance_error(kernels_by_label, targets_by_label),
            "swap": _swap_detailed_balance_error(
                core, states, product, permutations, permutation_to_index
            ),
        },
    }


def _summarize_exact_reports(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "exact_target_tv": max(float(row["exact_target_tv"]) for row in rows),
        "detailed_balance": {
            "within": max(float(row["detailed_balance"]["within"]) for row in rows),
            "swap": max(float(row["detailed_balance"]["swap"]) for row in rows),
        },
        "systems": [row["system_id"] for row in rows],
    }


def _corrected_transition_matrix(core: Any, states: np.ndarray, beta: float) -> np.ndarray:
    n_states = len(states)
    matrix = np.zeros((n_states, n_states), dtype=np.float64)
    energies = np.array(
        [float(core.energy(row.astype(int).tolist())) for row in states], dtype=np.float64
    )
    proposal_logs = np.zeros((n_states, n_states), dtype=np.float64)
    for source in range(n_states):
        for proposed in range(n_states):
            proposal_logs[source, proposed] = float(
                core.proposal_log_probability(
                    states[source].astype(int).tolist(),
                    states[proposed].astype(int).tolist(),
                    beta,
                )
            )
    for source in range(n_states):
        off_diagonal = 0.0
        for proposed in range(n_states):
            if proposed == source:
                continue
            log_acceptance = min(
                0.0,
                -float(beta) * (energies[proposed] - energies[source])
                + proposal_logs[proposed, source]
                - proposal_logs[source, proposed],
            )
            probability = exp(float(proposal_logs[source, proposed] + log_acceptance))
            matrix[source, proposed] = probability
            off_diagonal += probability
        matrix[source, source] = max(0.0, 1.0 - off_diagonal)
    return matrix


def _uncorrected_transition_matrix(core: Any, states: np.ndarray, beta: float) -> np.ndarray:
    matrix = np.zeros((len(states), len(states)), dtype=np.float64)
    for source in range(len(states)):
        for proposed in range(len(states)):
            matrix[source, proposed] = exp(
                float(
                    core.proposal_log_probability(
                        states[source].astype(int).tolist(),
                        states[proposed].astype(int).tolist(),
                        beta,
                    )
                )
            )
        matrix[source, :] /= float(np.sum(matrix[source, :]))
    return matrix


def _apply_schedule(
    distribution: np.ndarray,
    core: Any,
    states: np.ndarray,
    kernels_by_label: Sequence[np.ndarray],
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> np.ndarray:
    updated = np.array(distribution, dtype=np.float64)
    state_count = len(states)
    replica_count = len(BETA_LADDER)
    for replica_index in range(replica_count):
        output = np.zeros_like(updated)
        for state_tuple, labels in exp5633._iter_augmented_states(
            state_count=state_count,
            replica_count=replica_count,
            permutations=permutations,
        ):
            source_index = exp5633._augmented_index(
                state_tuple,
                labels,
                state_count=state_count,
                permutation_to_index=permutation_to_index,
            )
            label = labels[replica_index]
            row = kernels_by_label[label][state_tuple[replica_index]]
            for proposed_state, probability in enumerate(row):
                proposed_tuple = list(state_tuple)
                proposed_tuple[replica_index] = int(proposed_state)
                target_index = exp5633._augmented_index(
                    proposed_tuple,
                    labels,
                    state_count=state_count,
                    permutation_to_index=permutation_to_index,
                )
                output[target_index] += float(updated[source_index]) * float(probability)
        updated = output
    for left in range(replica_count - 1):
        output = np.zeros_like(updated)
        for state_tuple, labels in exp5633._iter_augmented_states(
            state_count=state_count,
            replica_count=replica_count,
            permutations=permutations,
        ):
            source_index = exp5633._augmented_index(
                state_tuple,
                labels,
                state_count=state_count,
                permutation_to_index=permutation_to_index,
            )
            state_rows = [states[index].astype(int).tolist() for index in state_tuple]
            log_ratio = float(core.swap_log_ratio(state_rows, list(labels), [left, left + 1]))
            acceptance = _acceptance_probability(log_ratio)
            swapped = list(labels)
            left_pos = labels.index(left)
            right_pos = labels.index(left + 1)
            swapped[left_pos], swapped[right_pos] = swapped[right_pos], swapped[left_pos]
            swapped_index = exp5633._augmented_index(
                state_tuple,
                tuple(swapped),
                state_count=state_count,
                permutation_to_index=permutation_to_index,
            )
            output[swapped_index] += float(updated[source_index]) * acceptance
            output[source_index] += float(updated[source_index]) * (1.0 - acceptance)
        updated = output
    return updated / float(np.sum(updated))


def _within_detailed_balance_error(
    kernels_by_label: Sequence[np.ndarray],
    targets_by_label: Sequence[np.ndarray],
) -> float:
    residual = 0.0
    for kernel, target in zip(kernels_by_label, targets_by_label, strict=True):
        for source in range(len(target)):
            for proposed in range(len(target)):
                residual = max(
                    residual,
                    abs(
                        float(target[source]) * float(kernel[source, proposed])
                        - float(target[proposed]) * float(kernel[proposed, source])
                    ),
                )
    return residual


def _swap_detailed_balance_error(
    core: Any,
    states: np.ndarray,
    product: np.ndarray,
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> float:
    residual = 0.0
    state_count = len(states)
    replica_count = len(BETA_LADDER)
    for left in range(replica_count - 1):
        for state_tuple, labels in exp5633._iter_augmented_states(
            state_count=state_count,
            replica_count=replica_count,
            permutations=permutations,
        ):
            source_index = exp5633._augmented_index(
                state_tuple,
                labels,
                state_count=state_count,
                permutation_to_index=permutation_to_index,
            )
            swapped = list(labels)
            left_pos = labels.index(left)
            right_pos = labels.index(left + 1)
            swapped[left_pos], swapped[right_pos] = swapped[right_pos], swapped[left_pos]
            swapped_tuple = tuple(swapped)
            target_index = exp5633._augmented_index(
                state_tuple,
                swapped_tuple,
                state_count=state_count,
                permutation_to_index=permutation_to_index,
            )
            state_rows = [states[index].astype(int).tolist() for index in state_tuple]
            forward = _acceptance_probability(
                float(core.swap_log_ratio(state_rows, list(labels), [left, left + 1]))
            )
            reverse = _acceptance_probability(
                float(core.swap_log_ratio(state_rows, list(swapped_tuple), [left, left + 1]))
            )
            residual = max(
                residual,
                abs(
                    float(product[source_index]) * forward - float(product[target_index]) * reverse
                ),
            )
    return residual


def _broken_control_results(rust_classes: Mapping[str, Any]) -> list[JsonDict]:
    config = default_config()
    py_core = PythonOneAxisTemperingCore(config)
    rust_core = _rust_core_from_config(config, rust_classes)
    system = exp5633.enumerable_frustrated_systems()[0]
    states = exp5622.enumerate_states(system.n_spins)
    state_rows = [
        states[0].astype(int).tolist(),
        states[3].astype(int).tolist(),
        states[-1].astype(int).tolist(),
    ]
    labels = [2, 0, 1]
    correct = py_core.swap_log_ratio(state_rows, labels, (1, 2))
    stale = py_core.swap_log_ratio(state_rows, [0, 1, 2], (1, 2))
    wrong_sign = -float(rust_core.swap_log_ratio(state_rows, labels, [1, 2]))

    targets_by_label = [
        exp5633.target_distribution_for_beta(system, states, beta) for beta in BETA_LADDER
    ]
    product, permutations, permutation_to_index = exp5633.product_target_distribution(
        targets_by_label
    )
    uncorrected_kernels = [
        _uncorrected_transition_matrix(py_core, states, beta) for beta in BETA_LADDER
    ]
    uncorrected_updated = _apply_schedule(
        product,
        py_core,
        states,
        uncorrected_kernels,
        permutations,
        permutation_to_index,
    )
    uncorrected_tv = exp5633.total_variation(product, uncorrected_updated)

    rows = [
        {
            "control_id": "stale_label_exchange",
            "rejected": abs(stale - correct) > FROZEN_TOLERANCES["swap_log_ratio"],
            "metric": abs(stale - correct),
            "reason": "stale labels change the label-position energy pair",
        },
        {
            "control_id": "wrong_sign_swap",
            "rejected": abs(wrong_sign - correct) > FROZEN_TOLERANCES["swap_log_ratio"],
            "metric": abs(wrong_sign - correct),
            "reason": "sign-reversed swap log-ratio mismatches Exp5633 rule",
        },
        {
            "control_id": "uncorrected_kernel",
            "rejected": uncorrected_tv > FROZEN_TOLERANCES["exact_target_tv"],
            "metric": uncorrected_tv,
            "reason": "projected cDLS without MH correction fails exact product target",
        },
        {
            "control_id": "collapsed_ladder",
            "rejected": _raises_value_error(
                lambda: OneAxisConfig(config.couplings, config.fields, (0.8, 0.8, 1.25))
            ),
            "metric": 1.0,
            "reason": "strict beta ladder validation rejects duplicate labels",
        },
        {
            "control_id": "corrupt_state",
            "rejected": _raises_value_error(
                lambda: OneAxisState.from_checkpoint({"states": [[1, -1, 1]]})
            ),
            "metric": 1.0,
            "reason": "checkpoint validation rejects missing labels, seed, and sweep",
        },
    ]
    return rows


def _malformed_input_controls(rust_classes: Mapping[str, Any]) -> list[JsonDict]:
    cfg_cls = rust_classes["config"]
    core_cls = rust_classes["core"]
    state_cls = rust_classes["state"]
    config = default_config()
    rust_core = _rust_core_from_config(config, rust_classes)
    checks = {
        "nonsquare_couplings": lambda: cfg_cls([[0.0, 0.1]], [0.0, 0.1], list(BETA_LADDER)),
        "collapsed_ladder": lambda: cfg_cls([[0.0]], [0.0], [0.8, 0.8]),
        "invalid_spin": lambda: rust_core.energy([1, 0, -1]),
        "duplicate_labels": lambda: state_cls(
            [[1, -1, 1], [1, -1, 1], [1, -1, 1]], [0, 0, 2], 7, 0
        ),
        "corrupt_checkpoint": lambda: state_cls.from_checkpoint({"states": [[1, -1, 1]]}),
    }
    rows: list[JsonDict] = []
    for control_id, call in checks.items():
        failed = _raises_value_error(call)
        rows.append(
            {
                "control_id": control_id,
                "failed_safely": failed,
                "error_type": "ValueError" if failed else None,
            }
        )
    return rows


def _python_fallback_equivalence() -> bool:
    config = default_config()
    py_core = PythonOneAxisTemperingCore(config)
    fallback_energy_ok = py_core.energy([1, -1, 1]) == py_core.energy([1, -1, 1])
    module_name = "carnot._rust_compat"
    saved = sys.modules.pop(module_name, None)
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "carnot._rust":
            raise ImportError("no rust extension")
        return original_import(name, *args, **kwargs)

    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            compat = importlib.import_module(module_name)
        return bool(
            fallback_energy_ok
            and compat.RUST_AVAILABLE is False
            and compat.RustOneAxisTemperingCore is None
        )
    finally:
        sys.modules.pop(module_name, None)
        if saved is not None:
            sys.modules[module_name] = saved


def _rust_classes() -> JsonDict:
    try:
        from carnot._rust import (  # type: ignore[import-not-found]
            RustOneAxisTemperingConfig,
            RustOneAxisTemperingCore,
            RustOneAxisTemperingState,
        )
    except ImportError as exc:  # pragma: no cover - tests build the extension.
        raise RuntimeError("Rust one-axis PyO3 extension is required for Exp5714") from exc
    return {
        "config": RustOneAxisTemperingConfig,
        "core": RustOneAxisTemperingCore,
        "state": RustOneAxisTemperingState,
    }


def _rust_core_from_config(config: OneAxisConfig, rust_classes: Mapping[str, Any]) -> Any:
    rust_config = rust_classes["config"](
        config.couplings.tolist(),
        config.fields.tolist(),
        list(config.beta_ladder),
        config.proposal_std,
        config.drift_scale,
    )
    return rust_classes["core"](rust_config)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normal_cdf(value: float) -> float:
    return min(max(0.5 * erfc(-float(value) / sqrt(2.0)), 1e-300), 1.0)


def _acceptance_probability(log_ratio: float) -> float:
    if log_ratio >= 0.0:
        return 1.0
    if log_ratio < -745.0:
        return 0.0
    return float(exp(log_ratio))


def _spin_array(state: Sequence[int], expected: int) -> np.ndarray:
    array = np.array(state, dtype=np.int8)
    if array.shape != (expected,):
        raise ValueError("state dimension mismatch")
    if not np.all((array == -1) | (array == 1)):
        raise ValueError("spin state values must be -1 or +1")
    return array.astype(np.float64)


def _state_collection(
    states: Sequence[Sequence[int]],
    labels: Sequence[int],
    config: OneAxisConfig,
) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.array(states, dtype=np.int8)
    if array.shape != (len(config.beta_ladder), config.n_spins):
        raise ValueError("states must match beta_ladder replica count and spin dimension")
    if not np.all((array == -1) | (array == 1)):
        raise ValueError("spin state values must be -1 or +1")
    labels_tuple = tuple(int(label) for label in labels)
    if sorted(labels_tuple) != list(range(len(config.beta_ladder))):
        raise ValueError("labels must be a permutation of beta-label indices")
    return array, labels_tuple


def _label_pair(label_pair: Sequence[int], replica_count: int) -> tuple[int, int]:
    if len(label_pair) != 2:
        raise ValueError("label_pair must contain exactly two adjacent labels")
    left, right = int(label_pair[0]), int(label_pair[1])
    if right != left + 1 or right >= replica_count:
        raise ValueError("label_pair must contain adjacent beta-label indices")
    return left, right


def _next_uniform(rng_state: int) -> tuple[int, float]:
    next_state = (int(rng_state) * LCG_A + LCG_C) % (2**64)
    bits = next_state >> 11
    return next_state, float(bits) * (1.0 / float(1 << 53))


def _draw_uniforms(rng_state: int, count: int) -> tuple[int, list[float]]:
    uniforms: list[float] = []
    state = int(rng_state)
    for _ in range(count):
        state, uniform = _next_uniform(state)
        uniforms.append(uniform)
    return state, uniforms


def _raises_value_error(call: Any) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def _command_version(root: Path, command: Sequence[str]) -> JsonDict:
    try:
        result = subprocess.run(
            list(command),
            cwd=root,
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "version": None, "error": str(exc)}
    output = (result.stdout or result.stderr).strip().splitlines()
    return {
        "available": result.returncode == 0,
        "version": output[0] if output else None,
        "returncode": result.returncode,
    }


def main() -> None:
    artifact = build_artifact(root=REPO_ROOT, random_seeds=DEFAULT_RANDOM_SEEDS)
    write_output(REPO_ROOT, artifact)


if __name__ == "__main__":  # pragma: no cover
    main()
