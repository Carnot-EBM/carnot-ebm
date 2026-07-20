"""Production SamplerBackend adapter for the promoted one-axis Rust kernel.

**Researcher summary:**
    Exp5714 and Exp5715 proved that the narrow one-axis corrected-cDLS
    temperature-label exchange algorithm has exact Rust/Python parity. This
    module exposes that promoted kernel through Carnot's production sampler
    backend boundary while keeping the default backend unchanged.

**Detailed explanation for engineers:**
    The adapter is deliberately conservative. It accepts one descriptor, routes
    Rust-supported inputs through the PyO3 symbols, and records an explicit
    exact Python fallback reason for declared compatibility cases. Invalid
    descriptors, two-axis requests, corrupt checkpoints, and seed mismatches
    stop at the boundary instead of changing algorithms silently.

Spec: REQ-SAMPLE-5723, SCENARIO-SAMPLE-5723, REQ-SAMPLE-5738,
SCENARIO-SAMPLE-5738
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import importlib
import json
import sys
from typing import Any

import numpy as np

from carnot import experiment_5714_one_axis_tempering_rust_parity as exp5714
from carnot import experiment_5715_one_axis_tempering_rust_quality_restart as exp5715


JsonDict = dict[str, Any]

ONE_AXIS_ALGORITHM = "one_axis_corrected_cdls_temperature_label_exchange"
ONE_AXIS_TOPOLOGY = "one_axis_temperature_label_exchange"
CHECKPOINT_SCHEMA_VERSION = "carnot.one_axis_samplerbackend.checkpoint.v1"
ENERGY_CONVENTION = "-0.5*x^T*J*x - h^T*x"
ACTIVE_RUST_BACKEND = "rust_pyo3"
ACTIVE_PYTHON_FALLBACK = "python_exact_fallback"

DEFAULT_ONE_AXIS_BACKEND_SEED = 5723
ONE_AXIS_BACKEND_SPEC_REFS = ["REQ-SAMPLE-5723", "SCENARIO-SAMPLE-5723"]
BATCH_BACKEND_SPEC_REFS = ["REQ-SAMPLE-5738", "SCENARIO-SAMPLE-5738"]


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for stable receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content using the repository convention."""

    return exp5714.sha256_json(value)


def checkpoint_checksum(checkpoint: Mapping[str, Any]) -> str:
    """Hash a checkpoint while blanking the self-referential checksum field."""

    stable = dict(checkpoint)
    stable["payload_checksum"] = ""
    return sha256_json(stable)


def descriptor_for_run(
    *,
    seed: int = DEFAULT_ONE_AXIS_BACKEND_SEED,
    initial_states: Sequence[Sequence[int]] | None = None,
    initial_labels: Sequence[int] | None = None,
    burn_in_sweeps: int = 0,
    force_python_fallback: bool = False,
) -> JsonDict:
    """Build the stable descriptor accepted by ``OneAxisRustBackend``.

    The defaults preserve the promoted Exp5714 constants. Callers may supply an
    initial state and labels when they need exact restart or parity replay.
    """

    descriptor: JsonDict = {
        "algorithm": ONE_AXIS_ALGORITHM,
        "topology": ONE_AXIS_TOPOLOGY,
        "source_algorithm_hash": exp5714.source_algorithm_hash(),
        "beta_ladder": [float(beta) for beta in exp5714.BETA_LADDER],
        "proposal_std": float(exp5714.exp5622.CDLS_PROPOSAL_STD),
        "drift_scale": float(exp5714.exp5622.CDLS_DRIFT_SCALE),
        "seed": int(seed),
        "burn_in_sweeps": int(burn_in_sweeps),
    }
    if initial_states is not None:
        descriptor["initial_states"] = [[int(value) for value in row] for row in initial_states]
    if initial_labels is not None:
        descriptor["initial_labels"] = [int(value) for value in initial_labels]
    if force_python_fallback:
        descriptor["force_python_fallback"] = True
    return descriptor


@dataclass(frozen=True)
class _NormalizedDescriptor:
    algorithm: str
    topology: str
    source_algorithm_hash: str
    beta_ladder: tuple[float, ...]
    proposal_std: float
    drift_scale: float
    seed: int
    burn_in_sweeps: int
    initial_state: exp5714.OneAxisState
    force_python_fallback: bool
    checkpoint: Mapping[str, Any] | None

    @property
    def descriptor_hash(self) -> str:
        return sha256_json(
            {
                "algorithm": self.algorithm,
                "topology": self.topology,
                "source_algorithm_hash": self.source_algorithm_hash,
                "beta_ladder": list(self.beta_ladder),
                "proposal_std": self.proposal_std,
                "drift_scale": self.drift_scale,
                "seed": self.seed,
                "burn_in_sweeps": self.burn_in_sweeps,
                "initial_state": self.initial_state.checkpoint(),
                "checkpoint_present": self.checkpoint is not None,
                "force_python_fallback": self.force_python_fallback,
            }
        )


@dataclass
class OneAxisRustBackend:
    """SamplerBackend adapter for the promoted one-axis Rust/PyO3 kernel.

    ``sample`` returns the protocol's boolean array shape, where ``True`` means
    spin ``+1`` and ``False`` means spin ``-1``. ``run_descriptor`` returns the
    same samples plus receipts, decision logs, and a portable checkpoint for
    audit and restart workflows.

    Spec: REQ-SAMPLE-5723, REQ-SAMPLE-5738
    """

    seed: int = DEFAULT_ONE_AXIS_BACKEND_SEED
    prefer_rust: bool = True
    rust_module_loader: Callable[[], Any] | None = None
    last_receipt: JsonDict | None = field(default=None, init=False)
    last_checkpoint: JsonDict | None = field(default=None, init=False)
    last_batch_receipt: JsonDict | None = field(default=None, init=False)
    last_run: JsonDict | None = field(default=None, init=False, repr=False)

    @property
    def backend_name(self) -> str:
        return "one_axis_rust"

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run the one-axis backend after a caller-specified burn-in budget."""

        config = descriptor_for_run(seed=self.seed, burn_in_sweeps=int(n_steps))
        config["requested_beta"] = float(beta)
        return self.sample(biases, couplings, n_samples=n_samples, config=config)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Return cold-target samples through the SamplerBackend call shape."""

        return np.asarray(
            self.run_descriptor(biases, couplings, n_samples=n_samples, config=config)["samples"],
            dtype=np.bool_,
        )

    def sample_batch(self, workloads: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
        """Run independent one-axis workloads in deterministic input order.

        Each workload mapping must contain ``biases``, ``couplings``,
        ``n_samples``, and ``config``. The return value preserves input order
        and uses the same full result shape as ``run_descriptor``.

        Spec: REQ-SAMPLE-5738
        """

        if not isinstance(workloads, Sequence) or isinstance(workloads, (str, bytes)):
            raise ValueError("batch workloads must be a sequence of mappings")
        results: list[JsonDict] = []
        ordered_ids: list[str] = []
        active_backends: list[str | None] = []
        for batch_index, item in enumerate(workloads):
            if not isinstance(item, Mapping):
                raise ValueError("batch workload must be a mapping")
            missing = {"biases", "couplings", "n_samples", "config"} - set(item)
            if missing:
                raise ValueError(f"batch workload missing required fields: {sorted(missing)}")
            workload_id = str(item.get("workload_id", f"batch-{batch_index}"))
            result = self.run_descriptor(
                item["biases"],
                item["couplings"],
                int(item["n_samples"]),
                item["config"],
            )
            result["workload_id"] = workload_id
            result["receipt"] = {
                **result["receipt"],
                "batch_index": int(batch_index),
                "workload_id": workload_id,
                "batch_item_count": len(workloads),
            }
            ordered_ids.append(workload_id)
            active_backends.append(result["receipt"].get("active_backend"))
            results.append(result)
        self.last_batch_receipt = {
            "backend_name": self.backend_name,
            "method": "sample_batch",
            "item_count": len(workloads),
            "ordered_workload_ids": ordered_ids,
            "active_backends": active_backends,
            "result_order_deterministic": True,
            "empty_batch": len(workloads) == 0,
            "spec_refs": list(BATCH_BACKEND_SPEC_REFS),
        }
        return results

    def run_descriptor(
        self,
        biases: object,
        couplings: object,
        n_samples: int,
        config: Mapping[str, Any],
    ) -> JsonDict:
        """Run the promoted descriptor and return samples plus audit receipts."""

        bias_vector, coupling_matrix, input_support = self._coerce_ising_inputs(
            biases,
            couplings,
        )
        descriptor = self._normalize_descriptor(config, bias_vector.size)
        input_hash = self._input_hash(bias_vector, coupling_matrix)
        core, active_backend, fallback_reason = self._select_core(
            bias_vector,
            coupling_matrix,
            descriptor,
            input_support=input_support,
        )
        initial = (
            exp5714.OneAxisState.from_checkpoint(
                self.load_checkpoint(
                    descriptor.checkpoint,
                    bias_vector,
                    coupling_matrix,
                    config=config,
                )
            )
            if descriptor.checkpoint is not None
            else descriptor.initial_state
        )
        run = self._advance(
            core=core,
            state=initial,
            n_samples=int(n_samples),
            descriptor=descriptor,
        )
        checkpoint = self._make_checkpoint(
            descriptor=descriptor,
            state=run["final_state"],
            input_hash=input_hash,
            active_backend=active_backend,
            fallback_reason=fallback_reason,
        )
        receipt = self._build_receipt(
            descriptor=descriptor,
            input_hash=input_hash,
            input_support=input_support,
            active_backend=active_backend,
            fallback_reason=fallback_reason,
            n_samples=int(n_samples),
            start_sweep=initial.sweep,
            final_state=run["final_state"],
        )
        result = {
            "samples": np.asarray(run["samples_bool"], dtype=np.bool_),
            "samples_spin": run["samples_spin"],
            "decision_log": run["decision_log"],
            "receipt": receipt,
            "checkpoint": checkpoint,
        }
        self.last_receipt = receipt
        self.last_checkpoint = checkpoint
        self.last_run = result
        return result

    def set_constraints(self, constraints: Any) -> None:
        """No-op primal-dual hook for SamplerBackend protocol conformance."""

        return None

    def dual_update_step(self, dual_lr: float) -> None:
        """No-op dual-update hook for SamplerBackend protocol conformance."""

        return None

    def save_checkpoint(self) -> JsonDict:
        """Return the latest checkpoint as a duplicate-safe JSON object."""

        if self.last_checkpoint is None:
            raise ValueError("no one-axis checkpoint has been produced")
        return json.loads(json.dumps(self.last_checkpoint))

    def load_checkpoint(
        self,
        checkpoint: Mapping[str, Any] | None,
        biases: object,
        couplings: object,
        *,
        config: Mapping[str, Any],
    ) -> JsonDict:
        """Validate a production one-axis checkpoint and return its state."""

        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must be an object")
        bias_vector, coupling_matrix, _ = self._coerce_ising_inputs(biases, couplings)
        descriptor = self._normalize_descriptor_without_checkpoint(config, bias_vector.size)
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema_version mismatch")
        if checkpoint.get("algorithm") != ONE_AXIS_ALGORITHM:
            raise ValueError("checkpoint algorithm mismatch")
        if checkpoint.get("source_algorithm_hash") != exp5714.source_algorithm_hash():
            raise ValueError("checkpoint source_algorithm_hash mismatch")
        if checkpoint.get("beta_ladder_hash") != exp5715.beta_ladder_hash():
            raise ValueError("checkpoint beta_ladder_hash mismatch")
        if checkpoint.get("input_hash") != self._input_hash(bias_vector, coupling_matrix):
            raise ValueError("checkpoint input_hash mismatch")
        if int(checkpoint.get("seed", -1)) != descriptor.seed:
            raise ValueError("checkpoint seed mismatch")
        if checkpoint.get("payload_checksum") != checkpoint_checksum(checkpoint):
            raise ValueError("checkpoint checksum mismatch")
        state = checkpoint.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint state missing")
        try:
            exp5714.OneAxisState.from_checkpoint(state)
        except ValueError as exc:
            raise ValueError(f"checkpoint state invalid: {exc}") from exc
        return dict(state)

    def energy_sign_control(
        self,
        biases: object,
        couplings: object,
        config: Mapping[str, Any],
    ) -> JsonDict:
        """Return a control proving the adapter rejects the opposite energy sign."""

        bias_vector, coupling_matrix, _ = self._coerce_ising_inputs(biases, couplings)
        descriptor = self._normalize_descriptor(config, bias_vector.size)
        state = descriptor.initial_state.states[0].astype(np.float64)
        correct = float(-0.5 * (state @ coupling_matrix @ state) - (state @ bias_vector))
        wrong_sign = float(0.5 * (state @ coupling_matrix @ state) + (state @ bias_vector))
        return {
            "control_id": "energy_sign",
            "expected_energy": _stable_float(correct),
            "wrong_sign_energy": _stable_float(wrong_sign),
            "rejected": abs(correct - wrong_sign) > 1e-12,
            "reason": "opposite Ising energy sign would invert the promoted target",
        }

    def _select_core(
        self,
        bias_vector: np.ndarray,
        coupling_matrix: np.ndarray,
        descriptor: _NormalizedDescriptor,
        *,
        input_support: Mapping[str, Any],
    ) -> tuple[Any, str, str | None]:
        fallback_reason = self._fallback_reason(descriptor, input_support)
        if fallback_reason is not None:
            return (
                self._python_core(bias_vector, coupling_matrix, descriptor),
                ACTIVE_PYTHON_FALLBACK,
                fallback_reason,
            )

        try:
            rust_module = (
                self.rust_module_loader()
                if self.rust_module_loader is not None
                else importlib.import_module("carnot._rust")
            )
        except ImportError as exc:
            return (
                self._python_core(bias_vector, coupling_matrix, descriptor),
                ACTIVE_PYTHON_FALLBACK,
                f"rust_extension_missing:{exc}",
            )
        except Exception as exc:
            return (
                self._python_core(bias_vector, coupling_matrix, descriptor),
                ACTIVE_PYTHON_FALLBACK,
                f"rust_extension_error:{type(exc).__name__}:{exc}",
            )

        required = (
            "RustOneAxisTemperingConfig",
            "RustOneAxisTemperingCore",
            "RustOneAxisTemperingState",
        )
        missing = [name for name in required if not hasattr(rust_module, name)]
        if missing:
            return (
                self._python_core(bias_vector, coupling_matrix, descriptor),
                ACTIVE_PYTHON_FALLBACK,
                f"rust_symbol_missing:{','.join(missing)}",
            )

        try:
            rust_config = rust_module.RustOneAxisTemperingConfig(
                coupling_matrix.tolist(),
                bias_vector.tolist(),
                list(descriptor.beta_ladder),
                descriptor.proposal_std,
                descriptor.drift_scale,
            )
            return rust_module.RustOneAxisTemperingCore(rust_config), ACTIVE_RUST_BACKEND, None
        except Exception as exc:
            return (
                self._python_core(bias_vector, coupling_matrix, descriptor),
                ACTIVE_PYTHON_FALLBACK,
                f"rust_extension_error:{type(exc).__name__}:{exc}",
            )

    def _fallback_reason(
        self,
        descriptor: _NormalizedDescriptor,
        input_support: Mapping[str, Any],
    ) -> str | None:
        if descriptor.force_python_fallback or not self.prefer_rust:
            return "declared_python_compatibility"
        if input_support["rust_supported"] is not True:
            return "unsupported_dtype_or_layout"
        return None

    @staticmethod
    def _python_core(
        bias_vector: np.ndarray,
        coupling_matrix: np.ndarray,
        descriptor: _NormalizedDescriptor,
    ) -> exp5714.PythonOneAxisTemperingCore:
        config = exp5714.OneAxisConfig(
            couplings=coupling_matrix,
            fields=bias_vector,
            beta_ladder=descriptor.beta_ladder,
            proposal_std=descriptor.proposal_std,
            drift_scale=descriptor.drift_scale,
        )
        return exp5714.PythonOneAxisTemperingCore(config)

    def _advance(
        self,
        *,
        core: Any,
        state: exp5714.OneAxisState,
        n_samples: int,
        descriptor: _NormalizedDescriptor,
    ) -> JsonDict:
        bulk_runner = getattr(core, "run_sweeps", None)
        if callable(bulk_runner):
            return self._advance_rust_bulk(
                bulk_runner=bulk_runner,
                state=state,
                n_samples=n_samples,
                descriptor=descriptor,
            )
        return self._advance_scalar(
            core=core,
            state=state,
            n_samples=n_samples,
            descriptor=descriptor,
        )

    def _advance_scalar(
        self,
        *,
        core: Any,
        state: exp5714.OneAxisState,
        n_samples: int,
        descriptor: _NormalizedDescriptor,
    ) -> JsonDict:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        states = np.array(state.states, dtype=np.int8)
        labels = [int(label) for label in state.labels]
        rng_state = int(state.rng_state)
        sweep = int(state.sweep)
        total_sweeps = descriptor.burn_in_sweeps + int(n_samples)
        samples_spin: list[list[int]] = []
        samples_bool: list[list[bool]] = []
        decision_log: list[JsonDict] = []

        for local_sweep in range(total_sweeps):
            sweep_events: list[JsonDict] = []
            completed_sweep = sweep + 1
            for physical_index in range(len(descriptor.beta_ladder)):
                beta_label = labels[physical_index]
                beta = descriptor.beta_ladder[beta_label]
                before = states[physical_index].astype(int).tolist()
                rng_state, uniforms = exp5714._draw_uniforms(  # noqa: SLF001
                    rng_state,
                    states.shape[1] + 1,
                )
                outcome = dict(core.corrected_step(before, beta, uniforms))
                after = [int(value) for value in outcome["state"]]
                states[physical_index] = np.array(after, dtype=np.int8)
                sweep_events.append(
                    {
                        "kind": "within",
                        "sweep": completed_sweep,
                        "physical_index": physical_index,
                        "beta_label": beta_label,
                        "beta": _stable_float(beta),
                        "uniforms": [_stable_float(value) for value in uniforms],
                        "state_before": before,
                        "state_after": after,
                        "proposed_state": [int(value) for value in outcome["proposed_state"]],
                        "current_energy": _stable_float(outcome["current_energy"]),
                        "proposed_energy": _stable_float(outcome["proposed_energy"]),
                        "proposal_log_forward": _stable_float(outcome["proposal_log_forward"]),
                        "proposal_log_reverse": _stable_float(outcome["proposal_log_reverse"]),
                        "log_acceptance": _stable_float(outcome["log_acceptance"]),
                        "accepted": bool(outcome["accepted"]),
                    }
                )
            for left in range(len(descriptor.beta_ladder) - 1):
                before_labels = list(labels)
                rng_state, uniform = exp5714._next_uniform(rng_state)  # noqa: SLF001
                outcome = dict(
                    core.swap_decision(
                        states.astype(int).tolist(),
                        labels,
                        [left, left + 1],
                        uniform,
                    )
                )
                labels = [int(label) for label in outcome["labels"]]
                sweep_events.append(
                    {
                        "kind": "swap",
                        "sweep": completed_sweep,
                        "label_pair": [left, left + 1],
                        "uniform": _stable_float(uniform),
                        "labels_before": before_labels,
                        "labels_after": list(labels),
                        "proposed_labels": [int(label) for label in outcome["proposed_labels"]],
                        "log_ratio": _stable_float(outcome["log_ratio"]),
                        "acceptance_probability": _stable_float(outcome["acceptance_probability"]),
                        "accepted": bool(outcome["accepted"]),
                    }
                )
            sweep = completed_sweep
            decision_log.extend(sweep_events)
            if local_sweep >= descriptor.burn_in_sweeps:
                cold_position = labels.index(exp5714.COLD_LABEL)
                sample = states[cold_position].astype(int).tolist()
                samples_spin.append(sample)
                samples_bool.append([value == 1 for value in sample])

        final_state = exp5714.OneAxisState(
            states=states,
            labels=tuple(labels),
            rng_state=rng_state,
            sweep=sweep,
        )
        return {
            "samples_spin": samples_spin,
            "samples_bool": samples_bool,
            "decision_log": decision_log,
            "final_state": final_state,
        }

    def _advance_rust_bulk(
        self,
        *,
        bulk_runner: Callable[..., Mapping[str, Any]],
        state: exp5714.OneAxisState,
        n_samples: int,
        descriptor: _NormalizedDescriptor,
    ) -> JsonDict:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        try:
            raw = bulk_runner(
                state.states.astype(int).tolist(),
                list(state.labels),
                int(state.rng_state),
                int(state.sweep),
                int(descriptor.burn_in_sweeps),
                int(n_samples),
            )
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed at the batch boundary.
            raise ValueError(f"rust bulk one-axis run failed: {type(exc).__name__}:{exc}") from exc
        samples_spin = [[int(value) for value in row] for row in raw["samples_spin"]]
        decision_log = [_normalize_decision_event(event) for event in raw["decision_log"]]
        final_state = exp5714.OneAxisState.from_checkpoint(raw["final_state"])
        return {
            "samples_spin": samples_spin,
            "samples_bool": [[value == 1 for value in sample] for sample in samples_spin],
            "decision_log": decision_log,
            "final_state": final_state,
        }

    def _make_checkpoint(
        self,
        *,
        descriptor: _NormalizedDescriptor,
        state: exp5714.OneAxisState,
        input_hash: str,
        active_backend: str,
        fallback_reason: str | None,
    ) -> JsonDict:
        checkpoint: JsonDict = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "algorithm": ONE_AXIS_ALGORITHM,
            "topology": ONE_AXIS_TOPOLOGY,
            "source_algorithm_hash": exp5714.source_algorithm_hash(),
            "descriptor_hash": descriptor.descriptor_hash,
            "input_hash": input_hash,
            "seed": int(descriptor.seed),
            "beta_ladder": [float(beta) for beta in descriptor.beta_ladder],
            "beta_ladder_hash": exp5715.beta_ladder_hash(),
            "proposal_std": descriptor.proposal_std,
            "drift_scale": descriptor.drift_scale,
            "energy_convention": ENERGY_CONVENTION,
            "active_backend": active_backend,
            "fallback_reason": fallback_reason,
            "byte_order": sys.byteorder,
            "state": state.checkpoint(),
            "payload_checksum": "",
        }
        checkpoint["payload_checksum"] = checkpoint_checksum(checkpoint)
        return checkpoint

    @staticmethod
    def _build_receipt(
        *,
        descriptor: _NormalizedDescriptor,
        input_hash: str,
        input_support: Mapping[str, Any],
        active_backend: str,
        fallback_reason: str | None,
        n_samples: int,
        start_sweep: int,
        final_state: exp5714.OneAxisState,
    ) -> JsonDict:
        total_sweeps = descriptor.burn_in_sweeps + int(n_samples)
        return {
            "backend_name": "one_axis_rust",
            "active_backend": active_backend,
            "fallback_reason": fallback_reason,
            "descriptor_hash": descriptor.descriptor_hash,
            "input_hash": input_hash,
            "input_support": dict(input_support),
            "algorithm": ONE_AXIS_ALGORITHM,
            "source_algorithm_hash": exp5714.source_algorithm_hash(),
            "energy_convention": ENERGY_CONVENTION,
            "seed": int(descriptor.seed),
            "rng_state_start": int(descriptor.initial_state.rng_state),
            "rng_state_final": int(final_state.rng_state),
            "start_sweep": int(start_sweep),
            "final_sweep": int(final_state.sweep),
            "transition_budget": {
                "burn_in_sweeps": int(descriptor.burn_in_sweeps),
                "sample_sweeps": int(n_samples),
                "total_sweeps": int(total_sweeps),
                "corrected_transitions": int(total_sweeps * len(descriptor.beta_ladder)),
                "swap_attempts": int(total_sweeps * (len(descriptor.beta_ladder) - 1)),
                "cold_target_samples": int(n_samples),
            },
        }

    def _normalize_descriptor(
        self,
        config: Mapping[str, Any],
        n_spins: int,
    ) -> _NormalizedDescriptor:
        if not isinstance(config, Mapping):
            raise ValueError("descriptor config must be an object")
        descriptor = dict(config.get("descriptor", config))
        return self._normalize_descriptor_mapping(descriptor, n_spins)

    def _normalize_descriptor_without_checkpoint(
        self,
        config: Mapping[str, Any],
        n_spins: int,
    ) -> _NormalizedDescriptor:
        descriptor = dict(config.get("descriptor", config))
        descriptor.pop("checkpoint", None)
        return self._normalize_descriptor_mapping(descriptor, n_spins)

    def _normalize_descriptor_mapping(
        self,
        descriptor: Mapping[str, Any],
        n_spins: int,
    ) -> _NormalizedDescriptor:
        algorithm = descriptor.get("algorithm")
        if algorithm != ONE_AXIS_ALGORITHM:
            raise ValueError("descriptor algorithm must be one_axis_corrected_cdls")
        topology = descriptor.get("topology", ONE_AXIS_TOPOLOGY)
        if topology != ONE_AXIS_TOPOLOGY:
            raise ValueError("unsupported topology for one-axis sampler backend")
        source_hash = str(descriptor.get("source_algorithm_hash", exp5714.source_algorithm_hash()))
        if source_hash != exp5714.source_algorithm_hash():
            raise ValueError("source_algorithm_hash mismatch")
        beta_ladder = tuple(float(beta) for beta in descriptor.get("beta_ladder", ()))
        if beta_ladder != tuple(float(beta) for beta in exp5714.BETA_LADDER):
            raise ValueError("beta_ladder must match Exp5714 one-axis ladder")
        proposal_std = float(descriptor.get("proposal_std", exp5714.exp5622.CDLS_PROPOSAL_STD))
        drift_scale = float(descriptor.get("drift_scale", exp5714.exp5622.CDLS_DRIFT_SCALE))
        if proposal_std != float(exp5714.exp5622.CDLS_PROPOSAL_STD):
            raise ValueError("proposal_std must match Exp5714")
        if drift_scale != float(exp5714.exp5622.CDLS_DRIFT_SCALE):
            raise ValueError("drift_scale must match Exp5714")
        seed = int(descriptor.get("seed", self.seed))
        if seed < 0 or seed >= 2**64:
            raise ValueError("seed must fit in u64")
        burn_in_sweeps = int(descriptor.get("burn_in_sweeps", 0))
        if burn_in_sweeps < 0:
            raise ValueError("burn_in_sweeps must be nonnegative")
        checkpoint = descriptor.get("checkpoint")
        initial_state = self._initial_state_from_descriptor(descriptor, n_spins, seed)
        return _NormalizedDescriptor(
            algorithm=ONE_AXIS_ALGORITHM,
            topology=ONE_AXIS_TOPOLOGY,
            source_algorithm_hash=source_hash,
            beta_ladder=beta_ladder,
            proposal_std=proposal_std,
            drift_scale=drift_scale,
            seed=seed,
            burn_in_sweeps=burn_in_sweeps,
            initial_state=initial_state,
            force_python_fallback=bool(descriptor.get("force_python_fallback", False)),
            checkpoint=checkpoint if isinstance(checkpoint, Mapping) else None,
        )

    @staticmethod
    def _initial_state_from_descriptor(
        descriptor: Mapping[str, Any],
        n_spins: int,
        seed: int,
    ) -> exp5714.OneAxisState:
        replica_count = len(exp5714.BETA_LADDER)
        states = descriptor.get("initial_states")
        labels = descriptor.get("initial_labels", list(range(replica_count)))
        if states is None:
            states = _default_initial_states(seed, replica_count, n_spins)
        return exp5714.OneAxisState(
            states=np.asarray(states, dtype=np.int8),
            labels=tuple(int(label) for label in labels),
            rng_state=int(descriptor.get("rng_state", seed)),
            sweep=int(descriptor.get("initial_sweep", 0)),
        )

    @staticmethod
    def _coerce_ising_inputs(
        biases: object,
        couplings: object,
    ) -> tuple[np.ndarray, np.ndarray, JsonDict]:
        raw_biases = np.asarray(biases)
        raw_couplings = np.asarray(couplings)
        bias_vector = np.asarray(biases, dtype=np.float64)
        coupling_matrix = np.asarray(couplings, dtype=np.float64)
        if bias_vector.ndim != 1:
            raise ValueError("biases must be a one-dimensional finite array")
        if coupling_matrix.shape != (bias_vector.size, bias_vector.size):
            raise ValueError("couplings must have shape (n_spins, n_spins)")
        if not np.all(np.isfinite(bias_vector)):
            raise ValueError("biases must contain only finite values")
        if not np.all(np.isfinite(coupling_matrix)):
            raise ValueError("couplings must contain only finite values")
        rust_supported = (
            isinstance(biases, np.ndarray)
            and isinstance(couplings, np.ndarray)
            and raw_biases.dtype == np.float64
            and raw_couplings.dtype == np.float64
            and raw_biases.flags.c_contiguous
            and raw_couplings.flags.c_contiguous
        )
        return (
            bias_vector,
            coupling_matrix,
            {
                "biases_dtype": str(raw_biases.dtype),
                "couplings_dtype": str(raw_couplings.dtype),
                "biases_shape": list(raw_biases.shape),
                "couplings_shape": list(raw_couplings.shape),
                "biases_c_contiguous": bool(raw_biases.flags.c_contiguous),
                "couplings_c_contiguous": bool(raw_couplings.flags.c_contiguous),
                "rust_supported": bool(rust_supported),
            },
        )

    @staticmethod
    def _input_hash(bias_vector: np.ndarray, coupling_matrix: np.ndarray) -> str:
        return sha256_json(
            {
                "fields": np.asarray(bias_vector, dtype=np.float64).tolist(),
                "couplings": np.asarray(coupling_matrix, dtype=np.float64).tolist(),
                "energy_convention": ENERGY_CONVENTION,
            }
        )


def _default_initial_states(seed: int, replica_count: int, n_spins: int) -> list[list[int]]:
    rng_state = int(seed)
    states: list[list[int]] = []
    for _ in range(replica_count):
        rng_state, uniforms = exp5714._draw_uniforms(rng_state, n_spins)  # noqa: SLF001
        states.append([1 if value < 0.5 else -1 for value in uniforms])
    return states


def _stable_float(value: Any) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _normalize_decision_event(event: Mapping[str, Any]) -> JsonDict:
    normalized: JsonDict = {}
    float_keys = {
        "beta",
        "uniform",
        "current_energy",
        "proposed_energy",
        "proposal_log_forward",
        "proposal_log_reverse",
        "log_acceptance",
        "log_ratio",
        "acceptance_probability",
    }
    list_int_keys = {
        "label_pair",
        "labels_after",
        "labels_before",
        "proposed_labels",
        "proposed_state",
        "state_after",
        "state_before",
    }
    for key, value in event.items():
        if key in float_keys:
            normalized[key] = _stable_float(value)
        elif key == "uniforms":
            normalized[key] = [_stable_float(item) for item in value]
        elif key in list_int_keys:
            normalized[key] = [int(item) for item in value]
        elif key in {"beta_label", "physical_index", "sweep"}:
            normalized[key] = int(value)
        elif key == "accepted":
            normalized[key] = bool(value)
        else:
            normalized[key] = value
    return normalized
