"""Runtime SamplerBackend adapter for the fixed mode-jump kernel.

Exp6194 qualified one finite categorical Metropolis-Hastings transition across
Python, Rust, and PyO3. This module exposes that exact transition through the
existing sampler backend factory while keeping Rust execution opt-in and
falling back to the exact Python path for declared compatibility cases.

Spec: REQ-SAMPLE-6208, SCENARIO-SAMPLE-6208-DEFAULT-OFF-FALLBACK,
SCENARIO-SAMPLE-6208-RUNTIME-PARITY, SCENARIO-SAMPLE-6208-BOUNDARY-ERRORS.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot import experiment_6194_mode_jump_rust_pyo3_parity as exp6194


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
MODE_JUMP_ALGORITHM = "exp6166_cross_mode_categorical_mh_v1"
MODE_JUMP_TOPOLOGY = "fixed_categorical_mode_jump"
VARIABLE_CARDINALITY_TOPOLOGY = "typed_variable_cardinality_mode_jump"
TYPED_STATE_METADATA_SCHEMA_VERSION = "carnot.mode_jump.typed_state_metadata.v1"
CHECKPOINT_SCHEMA_VERSION = "carnot.mode_jump_samplerbackend.checkpoint.v1"
ACTIVE_RUST_BACKEND = "rust_pyo3"
ACTIVE_PYTHON_FALLBACK = "python_exact_fallback"
FEATURE_ENV_VAR = "CARNOT_ENABLE_MODE_JUMP_RUNTIME"
DEFAULT_MODE_JUMP_BACKEND_SEED = 6194
MODE_JUMP_BACKEND_SPEC_REFS = [
    "REQ-SAMPLE-6208",
    "REQ-SAMPLER-6280",
    "SCENARIO-SAMPLE-6208-DEFAULT-OFF-FALLBACK",
    "SCENARIO-SAMPLE-6208-RUNTIME-PARITY",
    "SCENARIO-SAMPLE-6208-BOUNDARY-ERRORS",
    "SCENARIO-SAMPLER-6280-METADATA-ROUNDTRIP",
    "SCENARIO-SAMPLER-6280-PROPOSAL-PARITY",
]


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible receipts deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content using the repository convention."""

    return exp6194.sha256_json(value)


def checkpoint_checksum(checkpoint: Mapping[str, Any]) -> str:
    """Hash a checkpoint while blanking the self-referential checksum field."""

    stable = dict(checkpoint)
    stable["payload_checksum"] = ""
    return sha256_json(stable)


def frozen_mode_jump_inputs(root: Path = REPO_ROOT) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return the frozen Exp6194 label order, target, and proposal table."""

    config = exp6194.fixed_algorithm_equations_config_and_seed(root)
    labels = [str(label) for label in config["labels"]]
    target = np.asarray(
        [float(config["target_probabilities"][label]) for label in labels],
        dtype=np.float64,
    )
    proposal = np.asarray(config["proposal_probabilities"], dtype=np.float64)
    return labels, target, proposal


def descriptor_for_run(
    *,
    labels: Sequence[str] | None = None,
    seed: int = DEFAULT_MODE_JUMP_BACKEND_SEED,
    initial_label: str = exp6194.INITIAL_LABEL,
    burn_in: int = 0,
    enable_mode_jump_runtime: bool = False,
    force_python_fallback: bool = False,
    typed_state_metadata: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the stable descriptor accepted by ``ModeJumpRustBackend``."""

    frozen_labels = frozen_mode_jump_inputs()[0]
    active_labels = [str(label) for label in (labels if labels is not None else frozen_labels)]
    descriptor: JsonDict = {
        "algorithm": MODE_JUMP_ALGORITHM,
        "topology": (
            VARIABLE_CARDINALITY_TOPOLOGY
            if typed_state_metadata is not None
            else MODE_JUMP_TOPOLOGY
        ),
        "labels": active_labels,
        "seed": int(seed),
        "initial_label": str(initial_label),
        "burn_in": int(burn_in),
        "enable_mode_jump_runtime": bool(enable_mode_jump_runtime),
    }
    if typed_state_metadata is not None:
        descriptor["typed_state_metadata"] = dict(typed_state_metadata)
    if force_python_fallback:
        descriptor["force_python_fallback"] = True
    return descriptor


@dataclass(frozen=True)
class _NormalizedDescriptor:
    algorithm: str
    topology: str
    labels: tuple[str, ...]
    typed_state_metadata: Mapping[str, Any] | None
    seed: int
    initial_label: str
    burn_in: int
    enable_mode_jump_runtime: bool
    force_python_fallback: bool
    checkpoint: Mapping[str, Any] | None
    cancel_after_steps: int | None
    timeout_s: float | None
    return_trace: bool

    @property
    def descriptor_hash(self) -> str:
        return sha256_json(
            {
                "algorithm": self.algorithm,
                "topology": self.topology,
                "labels": list(self.labels),
                "typed_state_metadata_hash": (
                    sha256_json(self.typed_state_metadata)
                    if self.typed_state_metadata is not None
                    else None
                ),
                "seed": self.seed,
                "initial_label": self.initial_label,
                "burn_in": self.burn_in,
                "enable_mode_jump_runtime": self.enable_mode_jump_runtime,
                "force_python_fallback": self.force_python_fallback,
                "checkpoint_present": self.checkpoint is not None,
                "cancel_after_steps": self.cancel_after_steps,
                "timeout_s": self.timeout_s,
                "return_trace": self.return_trace,
            }
        )


@dataclass
class ModeJumpRustBackend:
    """SamplerBackend adapter for the fixed categorical mode-jump kernel.

    The protocol ``sample`` return is a one-hot boolean matrix over the frozen
    label order. The richer ``run_descriptor`` surface returns the same samples
    plus labels, state, diagnostics, checkpoint, and fallback receipts.
    """

    seed: int = DEFAULT_MODE_JUMP_BACKEND_SEED
    prefer_rust: bool = True
    rust_module_loader: Callable[[], Any] | None = None
    last_receipt: JsonDict | None = field(default=None, init=False)
    last_checkpoint: JsonDict | None = field(default=None, init=False)
    last_run: JsonDict | None = field(default=None, init=False, repr=False)

    @property
    def backend_name(self) -> str:
        return "mode_jump_rust"

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Return categorical samples after using ``n_steps`` as burn-in."""

        labels, _, _ = frozen_mode_jump_inputs()
        descriptor = descriptor_for_run(
            labels=labels,
            seed=self.seed,
            burn_in=int(n_steps),
            enable_mode_jump_runtime=_env_flag_enabled(),
        )
        descriptor["requested_beta"] = float(beta)
        return self.sample(biases, couplings, n_samples=n_samples, config=descriptor)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Return one-hot categorical mode samples through SamplerBackend."""

        result = self.run_descriptor(biases, couplings, n_samples=n_samples, config=config)
        return np.asarray(result["samples"], dtype=np.bool_)

    def run_descriptor(
        self,
        target_probabilities: object,
        proposal_probabilities: object,
        n_samples: int,
        config: Mapping[str, Any],
    ) -> JsonDict:
        """Run the fixed mode-jump descriptor and return audit receipts."""

        if int(n_samples) <= 0:
            raise ValueError("n_samples must be positive")
        target, proposal, input_support = self._coerce_mode_jump_inputs(
            target_probabilities,
            proposal_probabilities,
            config=config,
        )
        descriptor = self._normalize_descriptor(config, target.size)
        input_hash = self._input_hash(
            descriptor.labels,
            target,
            proposal,
            typed_state_metadata=descriptor.typed_state_metadata,
        )
        active_backend, fallback_reason, core, rust_module = self._select_core(
            target,
            proposal,
            descriptor,
            input_support=input_support,
        )
        initial_state = (
            self.load_checkpoint(
                descriptor.checkpoint,
                target,
                proposal,
                config=config,
            )
            if descriptor.checkpoint is not None
            else {
                "current_label": descriptor.initial_label,
                "rng_state": int(descriptor.seed),
                "step": 0,
                "accepted_count": 0,
            }
        )
        run = self._advance(
            core=core,
            rust_module=rust_module,
            state=initial_state,
            target=target,
            proposal=proposal,
            descriptor=descriptor,
            n_samples=int(n_samples),
            active_backend=active_backend,
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
            initial_state=initial_state,
            final_state=run["final_state"],
        )
        result = {
            "samples": run["samples"],
            "sample_labels": run["sample_labels"],
            "decision_log": run["decision_log"],
            "metrics": run["metrics"],
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
            raise ValueError("no mode-jump checkpoint has been produced")
        return json.loads(json.dumps(self.last_checkpoint))

    def load_checkpoint(
        self,
        checkpoint: Mapping[str, Any] | None,
        target_probabilities: object,
        proposal_probabilities: object,
        *,
        config: Mapping[str, Any],
    ) -> JsonDict:
        """Validate a mode-jump checkpoint and return its portable state."""

        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must be an object")
        target, proposal, _ = self._coerce_mode_jump_inputs(
            target_probabilities,
            proposal_probabilities,
            config=config,
        )
        descriptor = self._normalize_descriptor_without_checkpoint(config, target.size)
        if checkpoint.get("payload_checksum") != checkpoint_checksum(checkpoint):
            raise ValueError("checkpoint checksum mismatch")
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema_version mismatch")
        if checkpoint.get("algorithm") != MODE_JUMP_ALGORITHM:
            raise ValueError("checkpoint algorithm mismatch")
        if checkpoint.get("topology") != descriptor.topology:
            raise ValueError("checkpoint topology mismatch")
        if checkpoint.get("labels") != list(descriptor.labels):
            raise ValueError("checkpoint labels mismatch")
        expected_metadata_hash = (
            sha256_json(descriptor.typed_state_metadata)
            if descriptor.typed_state_metadata is not None
            else None
        )
        if checkpoint.get("typed_state_metadata_hash") != expected_metadata_hash:
            raise ValueError("checkpoint typed_state_metadata_hash mismatch")
        if checkpoint.get("input_hash") != self._input_hash(
            descriptor.labels,
            target,
            proposal,
            typed_state_metadata=descriptor.typed_state_metadata,
        ):
            raise ValueError("checkpoint input_hash mismatch")
        if int(checkpoint.get("seed", -1)) != descriptor.seed:
            raise ValueError("checkpoint seed mismatch")
        state = checkpoint.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint state missing")
        normalized = _normalize_state(state)
        self._validate_state(normalized, descriptor.labels)
        if checkpoint.get("serialized_state") != _serialize_state(normalized):
            raise ValueError("checkpoint serialized_state mismatch")
        return normalized

    def _select_core(
        self,
        target: np.ndarray,
        proposal: np.ndarray,
        descriptor: _NormalizedDescriptor,
        *,
        input_support: Mapping[str, Any],
    ) -> tuple[str, str | None, Any, Any | None]:
        fallback_reason = self._fallback_reason(descriptor, input_support)
        if fallback_reason is not None:
            return ACTIVE_PYTHON_FALLBACK, fallback_reason, None, None

        try:
            rust_module = (
                self.rust_module_loader()
                if self.rust_module_loader is not None
                else importlib.import_module("carnot._rust")
            )
        except ImportError as exc:
            return ACTIVE_PYTHON_FALLBACK, f"rust_extension_missing:{exc}", None, None
        except Exception as exc:
            return (
                ACTIVE_PYTHON_FALLBACK,
                f"rust_extension_error:{type(exc).__name__}:{exc}",
                None,
                None,
            )

        required = ["RustModeJumpConfig", "RustModeJumpCore", "RustModeJumpState"]
        if descriptor.typed_state_metadata is not None:
            required.append("RustModeJumpStateMetadata")
        missing = [name for name in required if not hasattr(rust_module, name)]
        if missing:
            return ACTIVE_PYTHON_FALLBACK, f"rust_symbol_missing:{','.join(missing)}", None, None

        try:
            if descriptor.typed_state_metadata is None:
                rust_config = rust_module.RustModeJumpConfig(
                    list(descriptor.labels),
                    target.astype(float).tolist(),
                    proposal.astype(float).tolist(),
                )
            else:
                metadata = descriptor.typed_state_metadata
                rust_metadata = rust_module.RustModeJumpStateMetadata(
                    str(metadata["schema"]),
                    [int(value) for value in metadata["shape"]],
                    [int(value) for value in metadata["cardinalities"]],
                    str(metadata["encoding"]),
                    [str(value) for value in metadata["state_labels"]],
                    [[int(item) for item in row] for row in metadata["state_values"]],
                    str(metadata["proposal_domain"]),
                    int(metadata["state_space_size"]),
                )
                rust_config = rust_module.RustModeJumpConfig.with_metadata(
                    list(descriptor.labels),
                    target.astype(float).tolist(),
                    proposal.astype(float).tolist(),
                    rust_metadata,
                )
            return ACTIVE_RUST_BACKEND, None, rust_module.RustModeJumpCore(rust_config), rust_module
        except Exception as exc:
            return (
                ACTIVE_PYTHON_FALLBACK,
                f"rust_extension_error:{type(exc).__name__}:{exc}",
                None,
                None,
            )

    def _fallback_reason(
        self,
        descriptor: _NormalizedDescriptor,
        input_support: Mapping[str, Any],
    ) -> str | None:
        if not (descriptor.enable_mode_jump_runtime or _env_flag_enabled()):
            return "feature_flag_disabled"
        if descriptor.force_python_fallback or not self.prefer_rust:
            return "declared_python_compatibility"
        if input_support["rust_supported"] is not True:
            return "unsupported_dtype_or_layout"
        return None

    def _advance(
        self,
        *,
        core: Any,
        rust_module: Any | None,
        state: Mapping[str, Any],
        target: np.ndarray,
        proposal: np.ndarray,
        descriptor: _NormalizedDescriptor,
        n_samples: int,
        active_backend: str,
    ) -> JsonDict:
        total_steps = descriptor.burn_in + n_samples
        start = time.perf_counter()
        current = _normalize_state(state)
        counts: Counter[str] = Counter()
        sample_labels: list[str] = []
        decision_log: list[JsonDict] = []
        accepted_before = int(current["accepted_count"])
        step_before = int(current["step"])

        for local_step in range(total_steps):
            if (
                descriptor.cancel_after_steps is not None
                and local_step >= descriptor.cancel_after_steps
            ):
                raise TimeoutError("mode-jump run cancelled by cancel_after_steps")
            if (
                descriptor.timeout_s is not None
                and time.perf_counter() - start >= descriptor.timeout_s
            ):
                raise TimeoutError("mode-jump runtime timeout")

            outcome = (
                self._rust_step(core, rust_module, current)
                if active_backend == ACTIVE_RUST_BACKEND
                else self._python_step(descriptor, target, proposal, current)
            )
            current = _normalize_state(outcome["state_after"])
            if descriptor.return_trace:
                decision_log.append(_normalize_decision(outcome))
            if local_step >= descriptor.burn_in:
                label = str(current["current_label"])
                sample_labels.append(label)
                counts[label] += 1

        samples = _one_hot(sample_labels, descriptor.labels)
        metrics = _metrics_from_labels(
            sample_labels=sample_labels,
            labels=descriptor.labels,
            target=target,
            accepted_count=int(current["accepted_count"]) - accepted_before,
            attempted_count=int(current["step"]) - step_before,
            final_state=current,
        )
        return {
            "samples": samples,
            "sample_labels": sample_labels,
            "counts": dict(counts),
            "decision_log": decision_log,
            "metrics": metrics,
            "final_state": current,
        }

    @staticmethod
    def _rust_step(core: Any, rust_module: Any, state: Mapping[str, Any]) -> JsonDict:
        rust_state = rust_module.RustModeJumpState.from_snapshot(dict(state))
        return dict(core.step_trace(rust_state))

    @staticmethod
    def _python_step(
        descriptor: _NormalizedDescriptor,
        target: np.ndarray,
        proposal: np.ndarray,
        state: Mapping[str, Any],
    ) -> JsonDict:
        config = {
            "labels": list(descriptor.labels),
            "target_probabilities": {
                label: float(target[index]) for index, label in enumerate(descriptor.labels)
            },
            "proposal_probabilities": proposal.astype(float).tolist(),
        }
        return exp6194._python_step(config, dict(state))  # noqa: SLF001

    def _normalize_descriptor(
        self,
        config: Mapping[str, Any],
        label_count: int,
    ) -> _NormalizedDescriptor:
        if not isinstance(config, Mapping):
            raise ValueError("descriptor config must be an object")
        descriptor = dict(config.get("descriptor", config))
        return self._normalize_descriptor_mapping(descriptor, label_count)

    def _normalize_descriptor_without_checkpoint(
        self,
        config: Mapping[str, Any],
        label_count: int,
    ) -> _NormalizedDescriptor:
        descriptor = dict(config.get("descriptor", config))
        descriptor.pop("checkpoint", None)
        return self._normalize_descriptor_mapping(descriptor, label_count)

    def _normalize_descriptor_mapping(
        self,
        descriptor: Mapping[str, Any],
        label_count: int,
    ) -> _NormalizedDescriptor:
        algorithm = descriptor.get("algorithm")
        if algorithm != MODE_JUMP_ALGORITHM:
            raise ValueError("descriptor algorithm must be exp6166_cross_mode_categorical_mh_v1")
        raw_metadata = descriptor.get("typed_state_metadata")
        typed_state_metadata = (
            normalize_typed_state_metadata(raw_metadata, label_count=label_count)
            if isinstance(raw_metadata, Mapping)
            else None
        )
        default_topology = (
            VARIABLE_CARDINALITY_TOPOLOGY
            if typed_state_metadata is not None
            else MODE_JUMP_TOPOLOGY
        )
        topology = str(descriptor.get("topology", default_topology))
        if typed_state_metadata is None and topology != MODE_JUMP_TOPOLOGY:
            raise ValueError("unsupported topology for mode-jump sampler backend")
        if typed_state_metadata is not None and topology != VARIABLE_CARDINALITY_TOPOLOGY:
            raise ValueError("typed state metadata requires variable-cardinality topology")
        frozen_labels = tuple(frozen_mode_jump_inputs()[0])
        if typed_state_metadata is None:
            labels = tuple(str(label) for label in descriptor.get("labels", frozen_labels))
            if labels != frozen_labels or len(labels) != label_count:
                raise ValueError("descriptor labels must match frozen Exp6194 labels")
        else:
            metadata_labels = tuple(str(label) for label in typed_state_metadata["state_labels"])
            labels = tuple(str(label) for label in descriptor.get("labels", metadata_labels))
            if labels != metadata_labels or len(labels) != label_count:
                raise ValueError("descriptor labels must match typed state metadata labels")
        seed = int(descriptor.get("seed", self.seed))
        if seed < 0 or seed >= 2**64:
            raise ValueError("seed must fit in u64")
        initial_default = exp6194.INITIAL_LABEL if typed_state_metadata is None else labels[0]
        initial_label = str(descriptor.get("initial_label", initial_default))
        if initial_label not in labels:
            raise ValueError("initial_label must be one of the frozen labels")
        burn_in = int(descriptor.get("burn_in", 0))
        if burn_in < 0:
            raise ValueError("burn_in must be nonnegative")
        checkpoint = descriptor.get("checkpoint")
        cancel_after_steps = descriptor.get("cancel_after_steps")
        timeout_s = descriptor.get("timeout_s")
        return _NormalizedDescriptor(
            algorithm=MODE_JUMP_ALGORITHM,
            topology=topology,
            labels=labels,
            typed_state_metadata=typed_state_metadata,
            seed=seed,
            initial_label=initial_label,
            burn_in=burn_in,
            enable_mode_jump_runtime=bool(descriptor.get("enable_mode_jump_runtime", False)),
            force_python_fallback=bool(descriptor.get("force_python_fallback", False)),
            checkpoint=checkpoint if isinstance(checkpoint, Mapping) else None,
            cancel_after_steps=None if cancel_after_steps is None else int(cancel_after_steps),
            timeout_s=None if timeout_s is None else float(timeout_s),
            return_trace=bool(descriptor.get("return_trace", True)),
        )

    @staticmethod
    def _coerce_mode_jump_inputs(
        target_probabilities: object,
        proposal_probabilities: object,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, JsonDict]:
        raw_metadata = _typed_state_metadata_from_config(config)
        if raw_metadata is not None:
            raw_target = np.asarray(target_probabilities)
            raw_proposal = np.asarray(proposal_probabilities)
            target = np.asarray(target_probabilities, dtype=np.float64)
            proposal = np.asarray(proposal_probabilities, dtype=np.float64)
            if target.ndim != 1:
                raise ValueError("target probabilities must be a rank-1 vector")
            metadata = normalize_typed_state_metadata(raw_metadata, label_count=int(target.size))
            label_count = len(metadata["state_labels"])
            if target.shape != (label_count,):
                raise ValueError("target probabilities must match typed metadata labels")
            if proposal.shape != (label_count, label_count):
                raise ValueError("proposal probabilities must match typed metadata labels")
            _validate_probability_inputs(target, proposal, target_name="target probabilities")
            rust_supported = (
                isinstance(target_probabilities, np.ndarray)
                and isinstance(proposal_probabilities, np.ndarray)
                and raw_target.dtype == np.float64
                and raw_proposal.dtype == np.float64
                and raw_target.flags.c_contiguous
                and raw_proposal.flags.c_contiguous
            )
            return (
                target.copy(),
                proposal.copy(),
                {
                    "target_dtype": str(raw_target.dtype),
                    "proposal_dtype": str(raw_proposal.dtype),
                    "target_shape": list(raw_target.shape),
                    "proposal_shape": list(raw_proposal.shape),
                    "target_c_contiguous": bool(raw_target.flags.c_contiguous),
                    "proposal_c_contiguous": bool(raw_proposal.flags.c_contiguous),
                    "rust_supported": bool(rust_supported),
                    "typed_state_metadata_schema": metadata["schema"],
                    "typed_state_metadata_hash": sha256_json(metadata),
                    "state_space_size": int(metadata["state_space_size"]),
                    "support_count": int(metadata["support_count"]),
                    "encoding": metadata["encoding"],
                },
            )
        frozen_labels, frozen_target, frozen_proposal = frozen_mode_jump_inputs()
        raw_target = np.asarray(target_probabilities)
        raw_proposal = np.asarray(proposal_probabilities)
        target = np.asarray(target_probabilities, dtype=np.float64)
        proposal = np.asarray(proposal_probabilities, dtype=np.float64)
        if target.ndim != 1 or target.shape != (len(frozen_labels),):
            raise ValueError("target probabilities must have frozen shape (6,)")
        if proposal.shape != (len(frozen_labels), len(frozen_labels)):
            raise ValueError("proposal probabilities must have frozen shape (6, 6)")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(proposal)):
            raise ValueError("mode-jump probabilities must contain only finite values")
        if np.any(target <= 0.0):
            raise ValueError("target probabilities must be positive on frozen support")
        _validate_probability_inputs(target, proposal, target_name="target probabilities")
        if not np.allclose(target, frozen_target, atol=1e-7, rtol=0.0):
            raise ValueError("target probabilities must match frozen Exp6194 target")
        if not np.allclose(proposal, frozen_proposal, atol=1e-7, rtol=0.0):
            raise ValueError("proposal probabilities must match frozen Exp6194 proposal")
        rust_supported = (
            isinstance(target_probabilities, np.ndarray)
            and isinstance(proposal_probabilities, np.ndarray)
            and raw_target.dtype == np.float64
            and raw_proposal.dtype == np.float64
            and raw_target.flags.c_contiguous
            and raw_proposal.flags.c_contiguous
        )
        return (
            frozen_target.copy(),
            frozen_proposal.copy(),
            {
                "target_dtype": str(raw_target.dtype),
                "proposal_dtype": str(raw_proposal.dtype),
                "target_shape": list(raw_target.shape),
                "proposal_shape": list(raw_proposal.shape),
                "target_c_contiguous": bool(raw_target.flags.c_contiguous),
                "proposal_c_contiguous": bool(raw_proposal.flags.c_contiguous),
                "rust_supported": bool(rust_supported),
            },
        )

    @staticmethod
    def _input_hash(
        labels: Sequence[str],
        target: np.ndarray,
        proposal: np.ndarray,
        *,
        typed_state_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        return sha256_json(
            {
                "labels": list(labels),
                "target_probabilities": np.asarray(target, dtype=np.float64).tolist(),
                "proposal_probabilities": np.asarray(proposal, dtype=np.float64).tolist(),
                "algorithm": MODE_JUMP_ALGORITHM,
                "topology": (
                    VARIABLE_CARDINALITY_TOPOLOGY
                    if typed_state_metadata is not None
                    else MODE_JUMP_TOPOLOGY
                ),
                "typed_state_metadata": (
                    dict(typed_state_metadata) if typed_state_metadata is not None else None
                ),
            }
        )

    @staticmethod
    def _validate_state(state: Mapping[str, Any], labels: Sequence[str]) -> None:
        if str(state["current_label"]) not in set(labels):
            raise ValueError("checkpoint state label is not in frozen labels")
        if int(state["accepted_count"]) > int(state["step"]):
            raise ValueError("checkpoint state accepted_count exceeds step")

    @staticmethod
    def _make_checkpoint(
        *,
        descriptor: _NormalizedDescriptor,
        state: Mapping[str, Any],
        input_hash: str,
        active_backend: str,
        fallback_reason: str | None,
    ) -> JsonDict:
        normalized = _normalize_state(state)
        checkpoint: JsonDict = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "algorithm": MODE_JUMP_ALGORITHM,
            "topology": descriptor.topology,
            "labels": list(descriptor.labels),
            "typed_state_metadata_hash": (
                sha256_json(descriptor.typed_state_metadata)
                if descriptor.typed_state_metadata is not None
                else None
            ),
            "descriptor_hash": descriptor.descriptor_hash,
            "input_hash": input_hash,
            "seed": int(descriptor.seed),
            "state": normalized,
            "serialized_state": _serialize_state(normalized),
            "active_backend": active_backend,
            "fallback_reason": fallback_reason,
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
        initial_state: Mapping[str, Any],
        final_state: Mapping[str, Any],
    ) -> JsonDict:
        return {
            "backend_name": "mode_jump_rust",
            "active_backend": active_backend,
            "fallback_reason": fallback_reason,
            "algorithm": MODE_JUMP_ALGORITHM,
            "topology": descriptor.topology,
            "descriptor_hash": descriptor.descriptor_hash,
            "typed_state_metadata_hash": (
                sha256_json(descriptor.typed_state_metadata)
                if descriptor.typed_state_metadata is not None
                else None
            ),
            "input_hash": input_hash,
            "input_support": dict(input_support),
            "feature_enabled_by_config": bool(descriptor.enable_mode_jump_runtime),
            "feature_env_var": FEATURE_ENV_VAR,
            "feature_enabled_by_env": _env_flag_enabled(),
            "seed": int(descriptor.seed),
            "initial_state": _normalize_state(initial_state),
            "final_state": _normalize_state(final_state),
            "transition_budget": {
                "burn_in": int(descriptor.burn_in),
                "retained_samples": int(n_samples),
                "total_steps": int(descriptor.burn_in + n_samples),
            },
            "spec_refs": list(MODE_JUMP_BACKEND_SPEC_REFS),
        }


def _env_flag_enabled() -> bool:
    return os.environ.get(FEATURE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def complete_support_proposal(label_count: int) -> np.ndarray:
    """Build a finite support proposal table for ABI parity controls."""

    if int(label_count) <= 1:
        raise ValueError("proposal support requires at least two labels")
    proposal = np.full(
        (int(label_count), int(label_count)),
        1.0 / float(int(label_count) - 1),
        dtype=np.float64,
    )
    np.fill_diagonal(proposal, 0.0)
    return proposal


def mode_jump_inputs_from_fixture_receipt(
    receipt: Mapping[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray, JsonDict]:
    """Return labels, target, proposal, and metadata for one Exp6268 receipt."""

    metadata = typed_state_metadata_from_fixture_receipt(receipt)
    labels = [str(label) for label in metadata["state_labels"]]
    if str(receipt["target_type"]) == "categorical_mode_jump":
        definition = dict(receipt["definition"])
        target = np.asarray(definition["target_probabilities"], dtype=np.float64)
        proposal = np.asarray(definition["proposal_probabilities"], dtype=np.float64)
    else:
        probability_by_label = {
            str(row["state_label"]): float(row["probability"]) for row in receipt["support"]
        }
        target = np.asarray([probability_by_label[label] for label in labels], dtype=np.float64)
        proposal = complete_support_proposal(len(labels))
    return labels, target, proposal, metadata


def typed_state_metadata_from_fixture_receipt(receipt: Mapping[str, Any]) -> JsonDict:
    """Build typed rank-1 state metadata from an Exp6268 exact fixture receipt."""

    target_type = str(receipt["target_type"])
    if target_type == "categorical_mode_jump":
        return _categorical_metadata(receipt)
    if target_type == "ising":
        return _comma_state_metadata(
            receipt,
            cardinalities=[2] * int(receipt["definition"]["n_spins"]),
            encoding="ising_pm_one_rank1",
            categories=[["-1", "+1"] for _ in range(int(receipt["definition"]["n_spins"]))],
        )
    if target_type == "potts":
        n_spins = int(receipt["definition"]["n_spins"])
        q_states = int(receipt["definition"]["q_states"])
        return _comma_state_metadata(
            receipt,
            cardinalities=[q_states] * n_spins,
            encoding="potts_zero_based_rank1",
            categories=[list(receipt["definition"]["state_labels"]) for _ in range(n_spins)],
        )
    if target_type == "typed_factor":
        return _typed_factor_metadata(receipt)
    raise ValueError(f"unsupported typed state fixture target_type: {target_type}")


def normalize_typed_state_metadata(
    metadata: Mapping[str, Any],
    *,
    label_count: int,
) -> JsonDict:
    """Validate and normalize explicit typed state metadata."""

    if not isinstance(metadata, Mapping):
        raise ValueError("typed state metadata must be an object")
    schema = str(metadata.get("schema"))
    if schema != TYPED_STATE_METADATA_SCHEMA_VERSION:
        raise ValueError("typed state metadata schema mismatch")
    shape = [int(value) for value in metadata.get("shape", [])]
    cardinalities = [int(value) for value in metadata.get("cardinalities", [])]
    if len(shape) != 1:
        raise ValueError("typed state metadata shape must be rank-1")
    if not cardinalities or shape[0] != len(cardinalities):
        raise ValueError("typed state metadata shape must match cardinality count")
    if any(cardinality < 2 for cardinality in cardinalities):
        raise ValueError("typed state metadata cardinality values must be at least 2")
    state_space_size = int(metadata.get("state_space_size", 0))
    expected_size = _product(cardinalities)
    if state_space_size != expected_size:
        raise ValueError(
            "typed state metadata cardinality state_space_size must match cardinalities"
        )
    labels = [str(label) for label in metadata.get("state_labels", [])]
    if len(labels) != int(label_count) or len(set(labels)) != len(labels):
        raise ValueError("typed state metadata labels must match target labels and be unique")
    raw_values = metadata.get("state_values", [])
    state_values = [[int(item) for item in row] for row in raw_values]
    if len(state_values) != len(labels):
        raise ValueError("typed state metadata state_values length must match labels")
    seen_values: set[tuple[int, ...]] = set()
    for state_value in state_values:
        if len(state_value) != len(cardinalities):
            raise ValueError("typed state metadata state value length must match cardinalities")
        for coordinate, cardinality in zip(state_value, cardinalities, strict=True):
            if coordinate < 0 or coordinate >= cardinality:
                raise ValueError("typed state metadata state value exceeds cardinality")
        value_key = tuple(state_value)
        if value_key in seen_values:
            raise ValueError("typed state metadata state values must be unique")
        seen_values.add(value_key)
    if len(labels) > state_space_size:
        raise ValueError("typed state metadata support exceeds state space")
    proposal_domain = str(metadata.get("proposal_domain"))
    if proposal_domain not in {"explicit_support_complete_no_self", "explicit_support_table"}:
        raise ValueError("typed state metadata proposal_domain is unsupported")
    variables = list(metadata.get("variables", []))
    normalized: JsonDict = {
        "schema": schema,
        "shape": shape,
        "cardinalities": cardinalities,
        "encoding": str(metadata.get("encoding")),
        "variables": variables,
        "state_labels": labels,
        "state_values": state_values,
        "label_to_index": {label: index for index, label in enumerate(labels)},
        "proposal_domain": proposal_domain,
        "state_space_size": state_space_size,
        "support_count": len(labels),
        "roundtrip_rule": "state_label -> support_index -> state_label; state_value stays within per-variable cardinality",
    }
    return normalized


def _typed_state_metadata_from_config(config: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(config, Mapping):
        return None
    descriptor = config.get("descriptor", config)
    if not isinstance(descriptor, Mapping):
        return None
    metadata = descriptor.get("typed_state_metadata")
    return metadata if isinstance(metadata, Mapping) else None


def _validate_probability_inputs(
    target: np.ndarray,
    proposal: np.ndarray,
    *,
    target_name: str,
) -> None:
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(proposal)):
        raise ValueError("mode-jump probabilities must contain only finite values")
    if np.any(target <= 0.0):
        raise ValueError(f"{target_name} must be positive on support")
    if abs(float(np.sum(target)) - 1.0) > 1.0e-7:
        raise ValueError(f"{target_name} must sum to 1.0")
    if np.any(proposal < 0.0):
        raise ValueError("proposal probabilities must be nonnegative")
    for row_index, row in enumerate(proposal):
        if abs(float(np.sum(row)) - 1.0) > 1.0e-7:
            raise ValueError(f"proposal row {row_index} must sum to 1.0")
    support = proposal > 0.0
    if not np.array_equal(support, support.T):
        raise ValueError("proposal support must be symmetric for MH correction")
    if np.any(np.diag(support)) and np.any(~np.eye(proposal.shape[0], dtype=bool) & support):
        return


def _categorical_metadata(receipt: Mapping[str, Any]) -> JsonDict:
    definition = dict(receipt["definition"])
    labels = [str(label) for label in definition["labels"]]
    variables = [
        {
            "id": "label",
            "kind": "categorical",
            "cardinality": len(labels),
            "categories": labels,
        }
    ]
    return normalize_typed_state_metadata(
        {
            "schema": TYPED_STATE_METADATA_SCHEMA_VERSION,
            "shape": [1],
            "cardinalities": [len(labels)],
            "encoding": "categorical_label_rank1",
            "variables": variables,
            "state_labels": labels,
            "state_values": [[index] for index in range(len(labels))],
            "proposal_domain": "explicit_support_table",
            "state_space_size": int(receipt["state_space_size"]),
        },
        label_count=len(labels),
    )


def _comma_state_metadata(
    receipt: Mapping[str, Any],
    *,
    cardinalities: Sequence[int],
    encoding: str,
    categories: Sequence[Sequence[str]],
) -> JsonDict:
    labels = [str(row["state_label"]) for row in receipt["support"]]
    state_values = []
    for label in labels:
        parts = label.split(",")
        encoded = []
        for raw, cardinality in zip(parts, cardinalities, strict=True):
            if raw == "-1":
                encoded.append(0)
            elif raw == "+1":
                encoded.append(1)
            else:
                value = int(raw)
                if value < 0 or value >= int(cardinality):
                    raise ValueError("state label exceeds cardinality")
                encoded.append(value)
        state_values.append(encoded)
    variables = [
        {
            "id": f"x{index}",
            "kind": "categorical",
            "cardinality": int(cardinality),
            "categories": list(category),
        }
        for index, (cardinality, category) in enumerate(zip(cardinalities, categories, strict=True))
    ]
    return normalize_typed_state_metadata(
        {
            "schema": TYPED_STATE_METADATA_SCHEMA_VERSION,
            "shape": [len(cardinalities)],
            "cardinalities": [int(value) for value in cardinalities],
            "encoding": encoding,
            "variables": variables,
            "state_labels": labels,
            "state_values": state_values,
            "proposal_domain": "explicit_support_complete_no_self",
            "state_space_size": int(receipt["state_space_size"]),
        },
        label_count=len(labels),
    )


def _typed_factor_metadata(receipt: Mapping[str, Any]) -> JsonDict:
    definition = dict(receipt["definition"])
    wires = {str(row["id"]): dict(row) for row in definition["program_payload"]["wires"]}
    wire_order = [str(value) for value in receipt.get("wire_order", wires.keys())]
    variables = []
    cardinalities = []
    for wire_id in wire_order:
        wire = wires[wire_id]
        categories = [str(value) for value in wire.get("categories", [])]
        cardinality = len(categories) if categories else 2
        variables.append(
            {
                "id": wire_id,
                "kind": str(wire.get("kind", "categorical")),
                "cardinality": int(cardinality),
                "categories": categories,
            }
        )
        cardinalities.append(int(cardinality))
    labels = [str(row["state_label"]) for row in receipt["support"]]
    state_values = []
    for support_row in receipt["support"]:
        state = dict(support_row["state"])
        encoded = []
        for variable in variables:
            value = state[variable["id"]]
            categories = list(variable["categories"])
            if categories:
                encoded.append(categories.index(str(value)))
            else:
                encoded.append(int(value))
        state_values.append(encoded)
    return normalize_typed_state_metadata(
        {
            "schema": TYPED_STATE_METADATA_SCHEMA_VERSION,
            "shape": [len(cardinalities)],
            "cardinalities": cardinalities,
            "encoding": "typed_factor_wire_order_rank1",
            "variables": variables,
            "state_labels": labels,
            "state_values": state_values,
            "proposal_domain": "explicit_support_complete_no_self",
            "state_space_size": int(receipt["state_space_size"]),
        },
        label_count=len(labels),
    )


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _normalize_state(state: Mapping[str, Any]) -> JsonDict:
    return {
        "current_label": str(state["current_label"]),
        "rng_state": int(state["rng_state"]),
        "step": int(state["step"]),
        "accepted_count": int(state["accepted_count"]),
    }


def _serialize_state(state: Mapping[str, Any]) -> str:
    normalized = _normalize_state(state)
    return (
        "mode_jump_state_v1|"
        f"{normalized['current_label']}|"
        f"{normalized['rng_state']}|"
        f"{normalized['step']}|"
        f"{normalized['accepted_count']}"
    )


def _normalize_decision(outcome: Mapping[str, Any]) -> JsonDict:
    normalized: JsonDict = {}
    for key, value in outcome.items():
        if key == "current_label":
            continue
        if key in {"state_before", "state_after"}:
            normalized[key] = _normalize_state(value)
        elif key in {
            "proposal_uniform",
            "acceptance_uniform",
            "current_energy",
            "proposed_energy",
            "proposal_log_forward",
            "proposal_log_reverse",
            "log_acceptance",
            "acceptance_probability",
        }:
            normalized[key] = _stable_float(value)
        elif key in {"accepted"}:
            normalized[key] = bool(value)
        elif key in {"rng_state_after"}:
            normalized[key] = int(value)
        else:
            normalized[key] = value
    return normalized


def _one_hot(sample_labels: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    samples = np.zeros((len(sample_labels), len(labels)), dtype=np.bool_)
    for row, label in enumerate(sample_labels):
        samples[row, index[label]] = True
    return samples


def _metrics_from_labels(
    *,
    sample_labels: Sequence[str],
    labels: Sequence[str],
    target: np.ndarray,
    accepted_count: int,
    attempted_count: int,
    final_state: Mapping[str, Any],
) -> JsonDict:
    counts = Counter(sample_labels)
    sample_count = len(sample_labels)
    frequencies = {
        label: {
            "count": int(counts[label]),
            "frequency": counts[label] / sample_count,
            "target_probability": float(target[index]),
            "wald_95_interval": _wald_interval(counts[label] / sample_count, sample_count),
        }
        for index, label in enumerate(labels)
    }
    tv = 0.5 * sum(
        abs(frequencies[label]["frequency"] - frequencies[label]["target_probability"])
        for label in labels
    )
    kl = sum(
        0.0
        if float(target[index]) == 0.0
        else float(target[index])
        * (
            float("inf")
            if frequencies[label]["frequency"] == 0.0
            else np.log(float(target[index]) / frequencies[label]["frequency"])
        )
        for index, label in enumerate(labels)
    )
    indicator = [1.0 if label == labels[0] else 0.0 for label in sample_labels]
    lag1, iact, ess = _quality_from_indicator(indicator)
    return {
        "sample_count": sample_count,
        "frequencies": frequencies,
        "total_variation_to_target": _stable_float(tv),
        "kl_target_to_empirical": _stable_float(kl),
        "accepted_count": int(accepted_count),
        "attempted_count": int(attempted_count),
        "acceptance_rate": _stable_float(accepted_count / attempted_count),
        "lag1_autocorrelation": _stable_float(lag1),
        "integrated_autocorrelation_time": _stable_float(iact),
        "effective_sample_size": _stable_float(ess),
        "final_state": _normalize_state(final_state),
        "serialized_final_state": _serialize_state(final_state),
    }


def _quality_from_indicator(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        return 0.0, 1.0, float(len(values))
    mean = sum(values) / len(values)
    denom = sum((value - mean) ** 2 for value in values)
    if denom == 0.0:
        return 0.0, 1.0, float(len(values))
    lag1 = _autocorrelation(values, mean, denom, 1)
    positive_sum = 0.0
    for lag in range(1, min(exp6194.MAX_ACF_LAG, len(values) - 1) + 1):
        rho = _autocorrelation(values, mean, denom, lag)
        if rho <= 0.0:
            break
        positive_sum += rho
    iact = max(1.0, 1.0 + 2.0 * positive_sum)
    return lag1, iact, len(values) / iact


def _autocorrelation(values: Sequence[float], mean: float, denom: float, lag: int) -> float:
    return (
        sum(
            (values[index] - mean) * (values[index - lag] - mean)
            for index in range(lag, len(values))
        )
        / denom
    )


def _wald_interval(p: float, n: int) -> list[float]:
    half_width = 1.96 * float(np.sqrt(max(p * (1.0 - p), 0.0) / n))
    return [_stable_float(max(0.0, p - half_width)), _stable_float(min(1.0, p + half_width))]


def _stable_float(value: Any) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded
