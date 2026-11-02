"""Exp 1598 Z1 drift compatibility test for the SamplerBackend boundary.

Spec traces: REQ-SAMPLE-065, SCENARIO-SAMPLE-093.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from carnot.samplers.backend import SamplerBackend
from carnot.sampling.z1_drift_correction import (
    DriftCorrectionConfig,
    SyntheticDriftIsingBackend,
    build_bipartite_ring_problem,
    make_beta_drift,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_PATH = PROJECT_ROOT / "results/experiment_1598_z1_drift.json"
RUN_DATE = "20260509"
EXP_ID = 1598

STRICT_TRANSCRIPT_ABSENT_FIELDS = (
    "transcript_schema_version",
    "authenticated_access_proof",
    "access_grant_reference",
    "provider_or_lab_operator",
    "device_family",
    "device_identifier",
    "device_firmware_or_runtime",
    "sdk_package_name",
    "sdk_version",
    "thrml_version",
    "device_discovery_command",
    "execution_timestamp_utc",
    "host_identifier",
    "benchmark_case_id",
    "schedule_id",
    "topology",
    "sample_count",
    "state_encoding",
    "output_samples_sha256",
    "energy_trace_sha256",
    "energy_metric_fields",
    "latency_metric_fields",
    "simulator_fallback_used",
    "claim_boundary_acknowledged",
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "experiment_id",
    "run_date",
    "compatibility_test_ready",
    "z1_drift_compatibility_test_passed",
    "sampler_backend_protocol",
    "sampler_backend_boundary",
    "backend_name",
    "n_spins",
    "n_samples",
    "minimize_samples",
    "beta",
    "drift_std",
    "sample_shape",
    "minimize_shape",
    "sample_dtype",
    "minimize_dtype",
    "sample_acceptance_rate",
    "minimize_acceptance_rate",
    "strict_transcript_fields_absent",
    "strict_transcript_absent_fields",
    "transcript_schema_path",
    "hardware_execution_performed",
    "hardware_claim_allowed",
    "z1_hardware_execution",
    "tsu_hardware_execution",
    "simulator_only_no_hardware_claim",
    "spec_refs",
    "source_spec_refs",
    "honest_verdict",
}


@dataclass(frozen=True)
class CompatibilityConfig:
    """Small deterministic simulator-only configuration for Exp 1598."""

    n_spins: int = 128
    beta: float = 0.85
    drift_std: float = 0.05
    n_samples: int = 24
    n_warmup_sweeps: int = 12
    sweeps_per_sample: int = 1
    seed: int = 1598
    minimize_samples: int = 6
    minimize_steps: int = 8

    def as_drift_config(self) -> DriftCorrectionConfig:
        """Return the Exp 1583 simulator config used by the compatibility test."""

        return DriftCorrectionConfig(
            n_spins=self.n_spins,
            beta=self.beta,
            drift_std=self.drift_std,
            n_samples=self.n_samples,
            n_warmup_sweeps=self.n_warmup_sweeps,
            sweeps_per_sample=self.sweeps_per_sample,
            seed=self.seed,
        )


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _dtype_name(values: np.ndarray) -> str:
    return np.dtype(values.dtype).name


def build_artifact(config: CompatibilityConfig = CompatibilityConfig()) -> dict[str, Any]:
    """Run the corrected synthetic drift backend as a SamplerBackend compatibility test."""

    drift_config = config.as_drift_config()
    biases, couplings = build_bipartite_ring_problem(drift_config)
    drift = make_beta_drift(drift_config)
    backend = SyntheticDriftIsingBackend(config=drift_config, beta_multipliers=drift, corrected=True)
    sample_config = {"beta": config.beta}
    samples = backend.sample(biases, couplings, config.n_samples, sample_config)
    sample_acceptance_rate = float(backend.last_acceptance_rate)
    minimized = backend.minimize_energy(
        biases,
        couplings,
        n_samples=config.minimize_samples,
        n_steps=config.minimize_steps,
        beta=config.beta,
    )
    minimize_acceptance_rate = float(backend.last_acceptance_rate)
    protocol_ok = isinstance(backend, SamplerBackend)
    compatibility_passed = bool(
        protocol_ok
        and samples.shape == (config.n_samples, config.n_spins)
        and minimized.shape == (config.minimize_samples, config.n_spins)
        and samples.dtype == np.dtype(bool)
        and minimized.dtype == np.dtype(bool)
        and 0.0 <= sample_acceptance_rate <= 1.0
        and 0.0 <= minimize_acceptance_rate <= 1.0
    )
    artifact = {
        "status": "complete",
        "experiment_id": EXP_ID,
        "run_date": RUN_DATE,
        "compatibility_test_ready": compatibility_passed,
        "z1_drift_compatibility_test_passed": compatibility_passed,
        "sampler_backend_protocol": "carnot.samplers.backend.SamplerBackend",
        "sampler_backend_boundary": (
            "carnot.sampling.z1_drift_correction.SyntheticDriftIsingBackend"
        ),
        "backend_name": backend.backend_name,
        "n_spins": int(config.n_spins),
        "n_samples": int(config.n_samples),
        "minimize_samples": int(config.minimize_samples),
        "beta": float(config.beta),
        "drift_std": float(drift.std(ddof=0)),
        "sample_shape": list(samples.shape),
        "minimize_shape": list(minimized.shape),
        "sample_dtype": _dtype_name(samples),
        "minimize_dtype": _dtype_name(minimized),
        "sample_acceptance_rate": sample_acceptance_rate,
        "minimize_acceptance_rate": minimize_acceptance_rate,
        "strict_transcript_fields_absent": True,
        "strict_transcript_absent_fields": list(STRICT_TRANSCRIPT_ABSENT_FIELDS),
        "transcript_schema_path": "ops/extropic_z1_transcript_schema.json",
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
        "z1_hardware_execution": False,
        "tsu_hardware_execution": False,
        "simulator_only_no_hardware_claim": True,
        "spec_refs": ["REQ-SAMPLE-065", "SCENARIO-SAMPLE-093"],
        "source_spec_refs": ["REQ-SAMPLE-063", "SCENARIO-SAMPLE-091"],
        "honest_verdict": (
            "complete: z1_drift_samplerbackend_compatibility_simulator_only_no_execution_claim"
        ),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 1598 schema and no-hardware-claim boundary."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - artifact.keys())
    forbidden_present = sorted(set(STRICT_TRANSCRIPT_ABSENT_FIELDS).intersection(artifact))
    expected_sample_shape = [artifact.get("n_samples"), artifact.get("n_spins")]
    expected_minimize_shape = [artifact.get("minimize_samples"), artifact.get("n_spins")]
    _ensure(not missing, f"missing required artifact fields: {missing}")
    _ensure(not forbidden_present, f"strict transcript fields present: {forbidden_present}")
    _ensure(artifact.get("status") == "complete", "status must be complete")
    _ensure(artifact.get("compatibility_test_ready") is True, "compatibility_test_ready")
    _ensure(
        artifact.get("z1_drift_compatibility_test_passed") is True,
        "z1_drift_compatibility_test_passed",
    )
    _ensure(artifact.get("backend_name") == "synthetic-drift-ising-hastings", "backend_name")
    _ensure(artifact.get("sample_shape") == expected_sample_shape, "sample_shape")
    _ensure(artifact.get("minimize_shape") == expected_minimize_shape, "minimize_shape")
    _ensure(artifact.get("sample_dtype") == "bool", "sample_dtype")
    _ensure(artifact.get("minimize_dtype") == "bool", "minimize_dtype")
    _ensure(
        0.0 <= float(artifact.get("sample_acceptance_rate", -1.0)) <= 1.0,
        "sample_acceptance_rate",
    )
    _ensure(
        0.0 <= float(artifact.get("minimize_acceptance_rate", -1.0)) <= 1.0,
        "minimize_acceptance_rate",
    )
    _ensure(
        artifact.get("strict_transcript_fields_absent") is True,
        "strict_transcript_fields_absent must remain true",
    )
    _ensure(
        artifact.get("strict_transcript_absent_fields") == list(STRICT_TRANSCRIPT_ABSENT_FIELDS),
        "strict_transcript_absent_fields",
    )
    _ensure(artifact.get("hardware_execution_performed") is False, "hardware_execution_performed")
    _ensure(artifact.get("hardware_claim_allowed") is False, "hardware_claim_allowed")
    _ensure(artifact.get("z1_hardware_execution") is False, "z1_hardware_execution")
    _ensure(artifact.get("tsu_hardware_execution") is False, "tsu_hardware_execution")
    _ensure(
        artifact.get("simulator_only_no_hardware_claim") is True,
        "simulator_only_no_hardware_claim",
    )
    _ensure(
        str(artifact.get("honest_verdict", "")).startswith("complete:"),
        "honest_verdict",
    )


def write_artifact(
    path: Path = DEFAULT_RESULT_PATH,
    config: CompatibilityConfig = CompatibilityConfig(),
) -> dict[str, Any]:
    """Write the Exp 1598 compatibility artifact."""

    artifact = build_artifact(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    """CLI entry point for generating the Exp 1598 deliverable."""

    artifact = write_artifact(DEFAULT_RESULT_PATH)
    print(
        artifact.get("compatibility_test_ready"),
        artifact.get("hardware_claim_allowed"),
        artifact.get("honest_verdict"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
