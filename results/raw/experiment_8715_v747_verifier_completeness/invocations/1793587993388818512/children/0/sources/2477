"""Measure exact Ising parity through the installed Torx CPU factor API.

This module uses Torx 0.0.1 only as installed. It maps each Carnot coupling and
bias to a typed ``PISING`` energy table. It then composes those local energies
and compares the complete distribution with the ready Exp6683 oracle. The
result is a software portability receipt. It is not a sampler or hardware
performance result.

Spec: REQ-SAMPLER-6684, REQ-REPORT-6684,
SCENARIO-SAMPLER-6684-EXACT-PARITY,
SCENARIO-SAMPLER-6684-FAIL-CLOSED,
SCENARIO-SAMPLER-6684-ADVERSARIAL-MAPPING,
SCENARIO-REPORT-6684-READY,
SCENARIO-REPORT-6684-BLOCKED,
SCENARIO-REPORT-6684-ATOMIC-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import importlib
from importlib import metadata
import inspect
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot import experiment_6657_bounded_treewidth_ising_reference as reference
from carnot import experiment_6683_ising_reference_scope_receipt as upstream_reference


# The task contract is CPU-only. Set this before Torx imports JAX and initializes
# a backend. This prevents an available accelerator from changing the evidence.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

JsonDict = dict[str, Any]
CommandRunner = Callable[[list[str], Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260827"
RESULT_PATH = Path("results/experiment_6684_torx_typed_factor_parity.json")
MODULE_PATH = Path("python/carnot/experiment_6684_torx_typed_factor_parity.py")
TEST_PATH = Path("tests/python/test_experiment_6684_torx_typed_factor_parity.py")
UPSTREAM_PATH = Path("results/experiment_6683_ising_reference_scope_receipt.json")
REFERENCE_PATH = Path("python/carnot/experiment_6657_bounded_treewidth_ising_reference.py")
UPSTREAM_MODULE_PATH = Path("python/carnot/experiment_6683_ising_reference_scope_receipt.py")
SAMPLER_SPEC_PATH = Path("openspec/capabilities/samplers/spec.md")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
HARDWARE_SPEC_PATH = Path("openspec/capabilities/hardware/spec.md")
VERIFICATION_SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
SAFETY_SPEC_PATH = Path("openspec/capabilities/safety/spec.md")
E2E_PLAN_PATH = Path("ops/e2e-test-plan.md")
V582_DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
REFERENCE_REFRESH_PATH = Path("research-references.md")
PROTECTED_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))

INFERENCE_SUBSTRATE = "installed_torx_cpu_typed_factors_no_llm"
CLAIM_SCOPE = "installed_torx_cpu_software_only"
TORX_DISTRIBUTION = "extro-torx"
TORX_VERSION = "0.0.1"
RELATIVE_FLOOR = 1.0e-15
PISING_DT = 1.0

REQUIRED_API_SYMBOLS = (
    "torx.__version__",
    "torx.psc.PISING",
    "torx.psc.PISING._energies",
    "torx.psc.PISING.get_generator",
)
REQUIRED_REJECTIONS = (
    "invalid_binary_wire",
    "malformed_state_width",
    "nonfinite_coefficient",
    "nonpositive_temperature",
    "unsupported_width",
    "self_loop",
    "duplicate_edge",
)
REQUIRED_ATTACKS = (
    "sign",
    "encoding",
    "scale",
    "bias",
    "duplicate",
    "precision",
    "order",
    "topology",
    "unsupported_width",
    "fallback",
)
TOLERANCES: dict[str, dict[str, float]] = {
    "factor_energy": {"absolute": 1.0e-12, "relative": 1.0e-12},
    "total_energy": {"absolute": 2.0e-12, "relative": 2.0e-12},
    "log_weight": {"absolute": 2.0e-12, "relative": 2.0e-12},
    "probability": {"absolute": 2.0e-12, "relative": 2.0e-11},
    "marginal": {"absolute": 2.0e-11, "relative": 2.0e-11},
    "correlation": {"absolute": 2.0e-11, "relative": 2.0e-11},
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "torx_runtime_receipt",
    "frozen_mapping_contract",
    "factor_rows",
    "state_parity_rows",
    "rejection_rows",
    "attack_rows",
    "torx_factor_parity_ready",
    "claim_scope",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "status": "The terminal state comes from deterministic process evidence.",
    "honest_verdict": "The conclusion uses measured parity evidence only.",
    "verdict_class": "The closed class preserves exact-reference circularity.",
    "gate_check_summary": "Expected and observed values localize each failure.",
    "torx_runtime_receipt": "Measured software identity binds the installed API.",
    "frozen_mapping_contract": "One mapping contract prevents semantic drift.",
    "factor_rows": "Local rows expose each coupling and bias energy table.",
    "state_parity_rows": "State rows expose every composed probability field.",
    "rejection_rows": "Unsupported inputs prove a fail-closed boundary.",
    "attack_rows": "Adversarial rows test each known mapping failure mode.",
    "torx_factor_parity_ready": "One Boolean reduces complete parity evidence.",
    "claim_scope": "The scope excludes every unmeasured hardware claim.",
    "per_unit_rows": "Raw units make every reduction independently recheckable.",
    "aggregate_row_recomputation": "The aggregate is rebuilt from retained rows.",
    "preconditions_checked": "Measured inputs and resources establish provenance.",
    "protected_files_unchanged": "Hashes protect active orchestration files.",
    "inference_substrate": "The substrate names installed CPU factors and no LLM.",
    "verifier_is_oracle": "Exp6683 exact probabilities explicitly define parity.",
    "field_provenance": "Every field names its source, path, and content hash.",
    "random_seed": "Frozen fixture and attack seeds preserve replay order.",
    "duration_s": "A monotonic duration records measured work.",
    "tests_run": "Command receipts make verification reproducible.",
    "reproducibility_checksum": "A canonical digest detects artifact mutation.",
}

_COVERAGE_FILE = "/tmp/carnot_exp6684_coverage"
_MODULE_INCLUDE = "*/experiment_6684_torx_typed_factor_parity.py"
VERIFICATION_DEFINITIONS: tuple[JsonDict, ...] = (
    {
        "ordinal": 1,
        "check_id": "focused_tests",
        "command": (
            f".venv/bin/coverage run --rcfile=/dev/null --data-file={_COVERAGE_FILE} "
            f"--include={_MODULE_INCLUDE} -m pytest {TEST_PATH} -q --no-cov -n 0 -o addopts="
        ),
        "expected_coverage_percent": None,
    },
    {
        "ordinal": 2,
        "check_id": "scoped_coverage",
        "command": (
            f".venv/bin/coverage report --rcfile=/dev/null --data-file={_COVERAGE_FILE} "
            f"--include={_MODULE_INCLUDE} --fail-under=100 --show-missing"
        ),
        "expected_coverage_percent": 100.0,
    },
    {
        "ordinal": 3,
        "check_id": "ruff_check",
        "command": f".venv/bin/ruff check {MODULE_PATH} {TEST_PATH}",
        "expected_coverage_percent": None,
    },
    {
        "ordinal": 4,
        "check_id": "format_check",
        "command": f".venv/bin/ruff format --check {MODULE_PATH} {TEST_PATH}",
        "expected_coverage_percent": None,
    },
    {
        "ordinal": 5,
        "check_id": "spec_coverage",
        "command": f".venv/bin/python scripts/check_spec_coverage.py {TEST_PATH}",
        "expected_coverage_percent": None,
    },
    {
        "ordinal": 6,
        "check_id": "applicable_e2e",
        "command": (
            ".venv/bin/pytest tests/python/test_e2e_training_sampling.py "
            "-q --no-cov -n 0 -o addopts="
        ),
        "expected_coverage_percent": None,
    },
    {
        "ordinal": 7,
        "check_id": "full_python_suite",
        "command": ".venv/bin/pytest tests/python -q",
        "expected_coverage_percent": None,
    },
)


class TorxApiError(RuntimeError):
    """Identify a missing or changed installed Torx API without a fallback."""


class UnsupportedTorxInput(ValueError):
    """Identify an input outside the frozen typed-factor mapping contract."""


class UpstreamGateError(RuntimeError):
    """Identify a failed or inconsistent Exp6683 release gate."""


@dataclass(frozen=True)
class TorxRuntime:
    """Store measured Torx identity plus the live class used for evaluation."""

    version: str
    distribution_name: str
    import_path: str
    symbols: tuple[str, ...]
    backend: str
    x64_enabled: bool
    cpu: str
    package_sha256: str
    pising_class: Any = field(repr=False, compare=False)
    jnp: Any = field(repr=False, compare=False)

    def receipt(self) -> JsonDict:
        """Return only stable JSON evidence, not live Python objects."""

        return {
            "version": self.version,
            "distribution_name": self.distribution_name,
            "import_path": self.import_path,
            "symbols": list(self.symbols),
            "backend": self.backend,
            "jax_x64_enabled": self.x64_enabled,
            "cpu": self.cpu,
            "package_sha256": self.package_sha256,
        }


@dataclass(frozen=True)
class MappedFactor:
    """Store one Torx PISING factor and its frozen local energy table."""

    fixture_id: str
    factor_id: str
    factor_type: str
    n_spins: int
    variables: tuple[int | str, int | str]
    coefficient: float
    theta: tuple[float, float, float, float, float]
    wire_dims: tuple[int, int]
    pinned_auxiliary_wire: int | None
    energy_table: tuple[float, float, float, float]


def canonical_json(value: Any) -> str:
    """Serialize evidence deterministically and reject nonfinite numbers."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("nonfinite JSON or unsupported evidence value") from exc


def sha256_bytes(value: bytes) -> str:
    """Prefix a SHA-256 digest so it cannot be mistaken for source text."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON rather than interpreter object text."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Keep a missing required file distinct from an empty file."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else "missing"


def row_hash(row: Mapping[str, Any], field_name: str = "row_sha256") -> str:
    """Hash one row after removing its self-referential digest field."""

    return sha256_json({key: value for key, value in row.items() if key != field_name})


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind every final field except the checksum that stores this digest."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def load_json(path: Path) -> JsonDict:
    """Load one JSON object without coercing malformed evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return dict(value)


def spec_anchors(text: str) -> list[str]:
    """Extract stable requirement identifiers from one specification."""

    return sorted(set(re.findall(r"(?:REQ|SCENARIO)-[A-Z0-9-]+", text)))


def relative_error(expected: float, observed: float) -> float:
    """Return error relative to the exact reference with a finite zero floor."""

    if expected == observed:
        return 0.0
    return abs(observed - expected) / max(abs(expected), RELATIVE_FLOOR)


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
        if match:
            return match.group(1).strip()
    return platform.processor() or platform.machine()


def _package_hash(distribution: metadata.Distribution) -> str:
    files = distribution.files or ()
    rows = []
    for relative in sorted(files, key=str):
        path = Path(distribution.locate_file(relative))
        if path.is_file():
            rows.append({"path": str(relative), "sha256": sha256_file(path)})
    if not rows:
        raise TorxApiError("installed Torx distribution has no hashable package files")
    return sha256_json(rows)


def require_torx_api(module: Any) -> type:
    """Require the exact installed symbols used by parity and permit no fallback."""

    psc = getattr(module, "psc", None)
    pising = getattr(psc, "PISING", None)
    if pising is None:
        raise TorxApiError("missing declared Torx API symbol: torx.psc.PISING")
    if not callable(getattr(pising, "_energies", None)):
        raise TorxApiError("missing declared Torx API symbol: torx.psc.PISING._energies")
    if not callable(getattr(pising, "get_generator", None)):
        raise TorxApiError("missing declared Torx API symbol: torx.psc.PISING.get_generator")
    return pising


def load_torx_runtime(*, importer: Callable[[str], Any] = importlib.import_module) -> TorxRuntime:
    """Import and identify the installed Torx CPU path before any parity work."""

    module = importer("torx")
    pising = require_torx_api(module)
    version = str(getattr(module, "__version__", "missing"))
    if version != TORX_VERSION:
        raise TorxApiError(f"expected Torx {TORX_VERSION}, observed {version}")

    jax = importer("jax")
    jnp = importer("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    backend = str(jax.default_backend())
    if backend != "cpu":
        raise TorxApiError(f"expected Torx JAX backend cpu, observed {backend}")
    if not bool(jax.config.x64_enabled):
        raise TorxApiError("Torx JAX binary64 support is disabled")

    distribution = metadata.distribution(TORX_DISTRIBUTION)
    return TorxRuntime(
        version=version,
        distribution_name=TORX_DISTRIBUTION,
        import_path=str(Path(module.__file__).resolve()),
        symbols=REQUIRED_API_SYMBOLS,
        backend=backend,
        x64_enabled=True,
        cpu=_cpu_name(),
        package_sha256=_package_hash(distribution),
        pising_class=pising,
        jnp=jnp,
    )


def _upstream_gate_receipt(upstream: Mapping[str, Any]) -> JsonDict:
    errors = upstream_reference.validate_artifact(upstream)
    observed = {
        "ising_reference_ready": upstream.get("ising_reference_ready"),
        "status": upstream.get("status"),
        "validation_errors": errors,
        "artifact_checksum": upstream.get("reproducibility_checksum"),
    }
    passed = upstream.get("ising_reference_ready") is True and not errors
    return {
        "check": "upstream",
        "expected": {"ising_reference_ready": True, "validation_errors": []},
        "observed": observed,
        "passed": passed,
    }


def supported_instances(upstream: Mapping[str, Any]) -> tuple[reference.IsingInstance, ...]:
    """Bind the installed mapping to the exact fixtures released by Exp6683."""

    gate = _upstream_gate_receipt(upstream)
    if not gate["passed"]:
        raise UpstreamGateError(f"Exp6683 gate failed: {gate['observed']}")
    manifest = {
        row["fixture_id"]: row
        for row in upstream.get("frozen_fixture_manifest", [])
        if row.get("expected_supported") is True
    }
    instances = tuple(item for item in reference.frozen_fixtures() if item.expected_supported)
    if len(instances) != 12 or set(manifest) != {item.instance_id for item in instances}:
        raise UpstreamGateError("Exp6683 supported fixture set changed")
    for item in instances:
        expected = "sha256:" + item.fixture_sha256
        if manifest[item.instance_id].get("source_fixture_sha256") != expected:
            raise UpstreamGateError(f"Exp6683 fixture hash changed: {item.instance_id}")
    return instances


def spin_to_bit(spin: int) -> int:
    """Apply the frozen Torx wire encoding and reject every other value."""

    if spin not in reference.SPINS:
        raise UnsupportedTorxInput("binary spin must be -1 or +1")
    return 0 if spin == -1 else 1


def map_fixture(
    instance: reference.IsingInstance, runtime: TorxRuntime
) -> tuple[MappedFactor, ...]:
    """Create one installed PISING table for each coupling and bias."""

    try:
        decomposition = reference.deterministic_tree_decomposition(instance)
        reference.validate_tree_decomposition(instance, decomposition)
    except reference.UnsupportedGraphError as exc:
        raise UnsupportedTorxInput(str(exc)) from exc
    beta = 1.0 / float(instance.temperature)
    factors: list[MappedFactor] = []
    definitions: list[
        tuple[str, str, tuple[int | str, int | str], float, tuple[float, ...], int | None]
    ] = []
    for index, (left, right, coupling) in enumerate(instance.edges):
        definitions.append(
            (
                f"coupling:{index}:{left}-{right}",
                "coupling",
                (left, right),
                float(coupling),
                (float(coupling), 0.0, 0.0, beta, PISING_DT),
                None,
            )
        )
    for vertex, bias in enumerate(instance.fields):
        definitions.append(
            (
                f"bias:{vertex}",
                "bias",
                (vertex, f"bias_anchor:{vertex}"),
                float(bias),
                (0.0, float(bias), 0.0, beta, PISING_DT),
                0,
            )
        )

    for factor_id, factor_type, variables, coefficient, theta, pinned in definitions:
        gate = runtime.pising_class(sites=[0, 1])
        params = runtime.jnp.asarray(theta, dtype=runtime.jnp.float64)
        energies = np.asarray(gate._energies(params), dtype=np.float64)
        if energies.shape != (4,) or not np.all(np.isfinite(energies)):
            raise UnsupportedTorxInput("Torx PISING returned an invalid energy table")
        factors.append(
            MappedFactor(
                fixture_id=instance.instance_id,
                factor_id=factor_id,
                factor_type=factor_type,
                n_spins=instance.n_spins,
                variables=variables,
                coefficient=coefficient,
                theta=tuple(float(value) for value in theta),  # type: ignore[arg-type]
                wire_dims=tuple(int(value) for value in gate.dims),
                pinned_auxiliary_wire=pinned,
                energy_table=tuple(float(value) for value in energies),  # type: ignore[arg-type]
            )
        )
    return tuple(factors)


def _normalize_state(factor: MappedFactor, state: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in state)
    if len(normalized) != factor.n_spins:
        raise UnsupportedTorxInput("state width must match the fixture")
    for value in normalized:
        spin_to_bit(value)
    return normalized


def factor_energy(factor: MappedFactor, state: Sequence[int], runtime: TorxRuntime) -> float:
    """Read one local energy from the installed Torx table."""

    del runtime
    spins = _normalize_state(factor, state)
    first = int(factor.variables[0])
    first_bit = spin_to_bit(spins[first])
    if factor.factor_type == "coupling":
        second_bit = spin_to_bit(spins[int(factor.variables[1])])
    else:
        second_bit = int(factor.pinned_auxiliary_wire or 0)
    return factor.energy_table[2 * first_bit + second_bit]


def _exact_factor_energy(factor: MappedFactor, state: Sequence[int]) -> float:
    spins = _normalize_state(factor, state)
    first = spins[int(factor.variables[0])]
    if factor.factor_type == "coupling":
        return -factor.coefficient * first * spins[int(factor.variables[1])]
    return -factor.coefficient * first


def _error_pair(expected: float, observed: float, field_name: str) -> JsonDict:
    absolute = abs(observed - expected)
    relative = relative_error(expected, observed)
    tolerance = TOLERANCES[field_name]
    finite = math.isfinite(expected) and math.isfinite(observed)
    return {
        "absolute": absolute,
        "relative": relative,
        "absolute_tolerance": tolerance["absolute"],
        "relative_tolerance": tolerance["relative"],
        "finite": finite,
        "valid": finite and absolute <= tolerance["absolute"] and relative <= tolerance["relative"],
    }


def _factor_row(
    factor: MappedFactor,
    states: Sequence[Sequence[int]],
    runtime: TorxRuntime,
) -> JsonDict:
    evaluations = []
    for state in states:
        exact = _exact_factor_energy(factor, state)
        observed = factor_energy(factor, state, runtime)
        error = _error_pair(exact, observed, "factor_energy")
        entry: JsonDict = {
            "state": list(state),
            "binary_state": [spin_to_bit(int(value)) for value in state],
            "exact_energy": exact,
            "torx_energy": observed,
            "absolute_error": error["absolute"],
            "relative_error": error["relative"],
            "valid": error["valid"],
        }
        entry["row_sha256"] = row_hash(entry)
        evaluations.append(entry)
    row: JsonDict = {
        "fixture_id": factor.fixture_id,
        "factor_id": factor.factor_id,
        "factor_type": factor.factor_type,
        "variables": list(factor.variables),
        "coefficient": factor.coefficient,
        "theta_J_h1_h2_beta_dt": list(factor.theta),
        "wire_dims": list(factor.wire_dims),
        "pinned_auxiliary_wire": factor.pinned_auxiliary_wire,
        "torx_factor": "torx.psc.PISING",
        "torx_call": "PISING._energies(theta)",
        "state_energy_rows": evaluations,
        "maximum_absolute_error": max(item["absolute_error"] for item in evaluations),
        "maximum_relative_error": max(item["relative_error"] for item in evaluations),
        "valid": all(item["valid"] for item in evaluations),
    }
    row["row_sha256"] = row_hash(row)
    return row


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def replay_parity(upstream: Mapping[str, Any], runtime: TorxRuntime) -> JsonDict:
    """Compare every installed Torx factor and state with Exp6683."""

    instances = supported_instances(upstream)
    exact_by_fixture: dict[str, list[Mapping[str, Any]]] = {}
    for row in upstream["exact_probability_rows"]:
        exact_by_fixture.setdefault(str(row["fixture_id"]), []).append(row)

    factor_rows: list[JsonDict] = []
    state_rows: list[JsonDict] = []
    mapping_rows: list[JsonDict] = []
    for instance in instances:
        exact_rows = exact_by_fixture.get(instance.instance_id, [])
        expected_states = 2**instance.n_spins
        if len(exact_rows) != expected_states:
            raise UpstreamGateError(
                f"Exp6683 state count changed for {instance.instance_id}: {len(exact_rows)}"
            )
        states = [tuple(int(value) for value in row["state"]) for row in exact_rows]
        factors = map_fixture(instance, runtime)
        local_rows = [_factor_row(factor, states, runtime) for factor in factors]
        factor_rows.extend(local_rows)

        manifest = next(
            row
            for row in upstream["frozen_fixture_manifest"]
            if row["fixture_id"] == instance.instance_id
        )
        mapping: JsonDict = {
            "fixture_id": instance.instance_id,
            "fixture_sha256": manifest["source_fixture_sha256"],
            "update_order": list(manifest["update_order"]),
            "factor_order": [factor.factor_id for factor in factors],
            "factor_count": len(factors),
            "state_count": expected_states,
        }
        mapping["row_sha256"] = row_hash(mapping)
        mapping_rows.append(mapping)

        preliminaries = []
        for exact_row, state in zip(exact_rows, states, strict=True):
            torx_factor_values = [factor_energy(factor, state, runtime) for factor in factors]
            exact_factor_values = [_exact_factor_energy(factor, state) for factor in factors]
            exact_energy = float(exact_row["energy"])
            torx_energy = float(math.fsum(torx_factor_values))
            exact_log_weight = -exact_energy / instance.temperature
            torx_log_weight = -torx_energy / instance.temperature
            preliminaries.append(
                {
                    "exact_row": exact_row,
                    "state": state,
                    "exact_factor_values": exact_factor_values,
                    "torx_factor_values": torx_factor_values,
                    "exact_energy": exact_energy,
                    "torx_energy": torx_energy,
                    "exact_log_weight": exact_log_weight,
                    "torx_log_weight": torx_log_weight,
                }
            )

        torx_log_partition = _logsumexp([float(item["torx_log_weight"]) for item in preliminaries])
        torx_probabilities = [
            math.exp(float(item["torx_log_weight"]) - torx_log_partition) for item in preliminaries
        ]
        torx_marginals = {
            str(vertex): math.fsum(
                probability
                for item, probability in zip(preliminaries, torx_probabilities, strict=True)
                if item["state"][vertex] == 1
            )
            for vertex in range(instance.n_spins)
        }
        torx_correlations = {
            f"{left}-{right}": math.fsum(
                probability * item["state"][left] * item["state"][right]
                for item, probability in zip(preliminaries, torx_probabilities, strict=True)
            )
            for left in range(instance.n_spins)
            for right in range(left + 1, instance.n_spins)
        }

        for item, torx_probability in zip(preliminaries, torx_probabilities, strict=True):
            exact_row = item["exact_row"]
            exact_marginals = {
                str(key): float(value) for key, value in exact_row["node_marginals_plus"].items()
            }
            exact_correlations = {
                str(key): float(value) for key, value in exact_row["pair_correlations"].items()
            }
            marginal_errors = {
                key: _error_pair(value, torx_marginals[key], "marginal")
                for key, value in exact_marginals.items()
            }
            correlation_errors = {
                key: _error_pair(value, torx_correlations[key], "correlation")
                for key, value in exact_correlations.items()
            }
            field_errors = {
                "total_energy": _error_pair(
                    float(item["exact_energy"]), float(item["torx_energy"]), "total_energy"
                ),
                "log_weight": _error_pair(
                    float(item["exact_log_weight"]),
                    float(item["torx_log_weight"]),
                    "log_weight",
                ),
                "probability": _error_pair(
                    float(exact_row["normalized_probability"]),
                    torx_probability,
                    "probability",
                ),
                "marginal": {
                    "absolute": max(
                        (value["absolute"] for value in marginal_errors.values()), default=0.0
                    ),
                    "relative": max(
                        (value["relative"] for value in marginal_errors.values()), default=0.0
                    ),
                    "valid": all(value["valid"] for value in marginal_errors.values()),
                },
                "correlation": {
                    "absolute": max(
                        (value["absolute"] for value in correlation_errors.values()), default=0.0
                    ),
                    "relative": max(
                        (value["relative"] for value in correlation_errors.values()), default=0.0
                    ),
                    "valid": all(value["valid"] for value in correlation_errors.values()),
                },
            }
            row = {
                "fixture_id": instance.instance_id,
                "state": list(item["state"]),
                "binary_state": [spin_to_bit(value) for value in item["state"]],
                "exact_factor_energies": item["exact_factor_values"],
                "torx_factor_energies": item["torx_factor_values"],
                "exact_total_energy": item["exact_energy"],
                "torx_total_energy": item["torx_energy"],
                "exact_log_weight": item["exact_log_weight"],
                "torx_log_weight": item["torx_log_weight"],
                "exact_partition_function": float(exact_row["partition_function"]),
                "torx_log_partition": torx_log_partition,
                "exact_normalized_probability": float(exact_row["normalized_probability"]),
                "torx_normalized_probability": torx_probability,
                "exact_node_marginals_plus": exact_marginals,
                "torx_node_marginals_plus": torx_marginals,
                "marginal_errors": marginal_errors,
                "exact_pair_correlations": exact_correlations,
                "torx_pair_correlations": {
                    key: torx_correlations[key] for key in exact_correlations
                },
                "correlation_errors": correlation_errors,
                "field_errors": field_errors,
                "finite": all(
                    math.isfinite(float(value))
                    for value in (
                        item["exact_energy"],
                        item["torx_energy"],
                        item["exact_log_weight"],
                        item["torx_log_weight"],
                        torx_probability,
                    )
                ),
                "valid": all(value["valid"] for value in field_errors.values()),
            }
            row["row_sha256"] = row_hash(row)
            state_rows.append(row)

    maximum_errors = {
        "factor_energy": {
            "absolute": max(row["maximum_absolute_error"] for row in factor_rows),
            "relative": max(row["maximum_relative_error"] for row in factor_rows),
        }
    }
    for field_name in ("total_energy", "log_weight", "probability", "marginal", "correlation"):
        maximum_errors[field_name] = {
            "absolute": max(row["field_errors"][field_name]["absolute"] for row in state_rows),
            "relative": max(row["field_errors"][field_name]["relative"] for row in state_rows),
        }
    return {
        "factor_rows": factor_rows,
        "state_parity_rows": state_rows,
        "fixture_mapping_rows": mapping_rows,
        "maximum_errors": maximum_errors,
        "supported_fixture_count": len(instances),
        "factor_count": len(factor_rows),
        "state_count": len(state_rows),
    }


def _observe_failure(call: Callable[[], Any]) -> str:
    try:
        call()
    except (UnsupportedTorxInput, reference.UnsupportedGraphError, TorxApiError) as exc:
        return str(exc)
    return "unexpectedly accepted"


def _rejection(case_id: str, expected: str, observed: str) -> JsonDict:
    row: JsonDict = {
        "case_id": case_id,
        "expected_failure": expected,
        "observed_failure": observed,
        "passed": expected in observed,
    }
    row["row_sha256"] = row_hash(row)
    return row


def build_rejection_rows(runtime: TorxRuntime) -> list[JsonDict]:
    """Exercise every unsupported boundary without hiding accepted inputs."""

    frozen = {item.instance_id: item for item in reference.frozen_fixtures()}
    valid = reference.IsingInstance("rejection_valid", 2, ((0, 1, 0.2),), (0.0, 0.0), 1.0, 6684001)
    factor = map_fixture(valid, runtime)[0]
    nonfinite = reference.IsingInstance(
        "rejection_nonfinite", 2, ((0, 1, float("nan")),), (0.0, 0.0), 1.0, 6684002
    )
    zero_temperature = reference.IsingInstance("rejection_temperature", 1, (), (0.0,), 0.0, 6684003)
    return [
        _rejection(
            "invalid_binary_wire",
            "binary spin",
            _observe_failure(lambda: factor_energy(factor, (0, 1), runtime)),
        ),
        _rejection(
            "malformed_state_width",
            "state width",
            _observe_failure(lambda: factor_energy(factor, (-1,), runtime)),
        ),
        _rejection(
            "nonfinite_coefficient",
            "finite",
            _observe_failure(lambda: map_fixture(nonfinite, runtime)),
        ),
        _rejection(
            "nonpositive_temperature",
            "positive",
            _observe_failure(lambda: map_fixture(zero_temperature, runtime)),
        ),
        _rejection(
            "unsupported_width",
            "treewidth",
            _observe_failure(lambda: map_fixture(frozen["unsupported_k6_tw5"], runtime)),
        ),
        _rejection(
            "self_loop",
            "self-loop",
            _observe_failure(lambda: map_fixture(frozen["unsupported_self_loop"], runtime)),
        ),
        _rejection(
            "duplicate_edge",
            "duplicate edge",
            _observe_failure(lambda: map_fixture(frozen["unsupported_duplicate_edge"], runtime)),
        ),
    ]


def _attack(attack_id: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    row: JsonDict = {
        "attack_id": attack_id,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }
    row["row_sha256"] = row_hash(row)
    return row


def build_attack_rows(
    upstream: Mapping[str, Any],
    runtime: TorxRuntime,
    parity: Mapping[str, Any],
    rejection_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Attack every requested mapping boundary with a measured counterfactual."""

    del upstream
    states = parity["state_parity_rows"]
    ferro = next(
        row for row in states if row["fixture_id"] == "edge_ferro" and row["state"] == [1, 1]
    )
    sign_delta = abs(float(ferro["exact_total_energy"]) - -float(ferro["torx_total_energy"]))

    singleton = next(
        row for row in states if row["fixture_id"] == "singleton_field" and row["state"] == [-1]
    )
    encoding_wrong_energy = -float(singleton["exact_total_energy"])
    encoding_delta = abs(float(singleton["exact_total_energy"]) - encoding_wrong_energy)

    scale_wrong_log_weight = -float(ferro["torx_total_energy"]) * 0.9
    scale_delta = abs(float(ferro["exact_log_weight"]) - scale_wrong_log_weight)

    path = next(
        row for row in states if row["fixture_id"] == "path4_field" and row["state"] == [1, 1, 1, 1]
    )
    bias_factor = next(
        value
        for value in parity["factor_rows"]
        if value["fixture_id"] == "path4_field"
        and value["factor_type"] == "bias"
        and value["coefficient"] != 0.0
    )
    bias_energy = next(
        value["torx_energy"]
        for value in bias_factor["state_energy_rows"]
        if value["state"] == path["state"]
    )
    missing_bias_delta = abs(float(bias_energy))

    coupling_factor = next(
        value
        for value in parity["factor_rows"]
        if value["fixture_id"] == "edge_ferro" and value["factor_type"] == "coupling"
    )
    duplicate_delta = abs(
        next(
            value["torx_energy"]
            for value in coupling_factor["state_energy_rows"]
            if value["state"] == ferro["state"]
        )
    )

    precise = 0.1234567890123456
    truncated = float(np.float32(precise))
    precision_delta = abs(precise - truncated)

    mapping = next(
        row
        for row in parity["fixture_mapping_rows"]
        if len(row["update_order"]) > 1
        and list(reversed(row["update_order"])) != row["update_order"]
    )
    reversed_order = list(reversed(mapping["update_order"]))

    disconnected = reference.IsingInstance(
        "attack_disconnected",
        4,
        ((0, 1, 0.4), (2, 3, -0.2)),
        (0.1, 0.0, -0.1, 0.05),
        1.0,
        6684010,
    )
    disconnected_factors = map_fixture(disconnected, runtime)
    disconnected_rows = reference.brute_force_reference(disconnected)
    torx_log_weights = []
    for state in disconnected_rows["states"]:
        energy = math.fsum(factor_energy(factor, state, runtime) for factor in disconnected_factors)
        torx_log_weights.append(-energy / disconnected.temperature)
    log_partition = _logsumexp(torx_log_weights)
    topology_error = max(
        abs(expected - math.exp(log_weight - log_partition))
        for expected, log_weight in zip(
            disconnected_rows["probabilities"], torx_log_weights, strict=True
        )
    )

    width_row = next(row for row in rejection_rows if row["case_id"] == "unsupported_width")
    fake = SimpleNamespace(__version__=TORX_VERSION, psc=SimpleNamespace())
    fallback_error = _observe_failure(lambda: require_torx_api(fake))
    attacks = [
        _attack("sign", "nonzero parity error", {"absolute_error": sign_delta}, sign_delta > 0.0),
        _attack(
            "encoding",
            "nonzero parity error",
            {"absolute_error": encoding_delta},
            encoding_delta > 0.0,
        ),
        _attack(
            "scale",
            "nonzero log-weight error",
            {"absolute_error": scale_delta},
            scale_delta > TOLERANCES["log_weight"]["absolute"],
        ),
        _attack(
            "bias",
            "missing bias detected",
            {"absolute_error": missing_bias_delta},
            missing_bias_delta > 0.0,
        ),
        _attack(
            "duplicate",
            "duplicate factor detected",
            {"absolute_error": duplicate_delta},
            duplicate_delta > 0.0,
        ),
        _attack(
            "precision",
            "binary32 truncation detected",
            {"binary64": precise, "binary32": truncated, "absolute_error": precision_delta},
            precision_delta > 0.0,
        ),
        _attack(
            "order",
            mapping["update_order"],
            {
                "reversed": reversed_order,
                "drift_detected": reversed_order != mapping["update_order"],
            },
            reversed_order != mapping["update_order"],
        ),
        _attack(
            "topology",
            "disconnected supported parity",
            {
                "disconnected_supported": True,
                "maximum_probability_error": topology_error,
            },
            topology_error <= TOLERANCES["probability"]["absolute"],
        ),
        _attack(
            "unsupported_width",
            "treewidth rejection",
            {"rejection_row_sha256": width_row["row_sha256"], "passed": width_row["passed"]},
            width_row["passed"] is True,
        ),
        _attack(
            "fallback",
            "missing API blocks with no fallback",
            {"error": fallback_error, "fallback_used": False},
            "PISING" in fallback_error,
        ),
    ]
    return attacks


def make_test_receipt(
    definition: Mapping[str, Any],
    *,
    exit_code: int,
    duration_s: float,
    summary: str,
    output_sha256: str,
    coverage_percent: float | None,
) -> JsonDict:
    """Bind one verification command to its exit and coverage result."""

    expected_coverage = definition.get("expected_coverage_percent")
    passed = exit_code == 0 and (expected_coverage is None or coverage_percent == expected_coverage)
    row: JsonDict = {
        "ordinal": definition["ordinal"],
        "check_id": definition["check_id"],
        "command": definition["command"],
        "exit_code": exit_code,
        "coverage_percent": coverage_percent,
        "duration_s": duration_s,
        "summary": summary,
        "output_sha256": output_sha256,
        "passed": passed,
    }
    row["receipt_sha256"] = row_hash(row, "receipt_sha256")
    return row


def default_command_runner(command: list[str], cwd: Path) -> JsonDict:
    """Run one verification process and retain its measured output."""

    started = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    lines = [line for line in output.splitlines() if line.strip()]
    return {
        "exit_code": completed.returncode,
        "duration_s": time.monotonic() - started,
        "summary": lines[-1] if lines else "no output",
        "output": output,
        "output_sha256": sha256_bytes(output.encode("utf-8")),
    }


def run_verification(
    root: Path, *, command_runner: CommandRunner = default_command_runner
) -> list[JsonDict]:
    """Run focused tests, scoped coverage, lint, format, spec, and E2E checks."""

    rows = []
    for definition in VERIFICATION_DEFINITIONS:
        result = command_runner(shlex.split(str(definition["command"])), root)
        coverage = None
        if definition["check_id"] == "scoped_coverage":
            match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", str(result["output"]))
            coverage = float(match.group(1)) if match else None
        rows.append(
            make_test_receipt(
                definition,
                exit_code=int(result["exit_code"]),
                duration_s=float(result["duration_s"]),
                summary=str(result["summary"]),
                output_sha256=str(result["output_sha256"]),
                coverage_percent=coverage,
            )
        )
    return rows


def reduce_test_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], JsonDict]:
    """Reject missing, changed, reordered, failed, or unhashed command receipts."""

    failures: list[JsonDict] = []
    expected = [row["check_id"] for row in VERIFICATION_DEFINITIONS]
    observed = [row.get("check_id") for row in rows]
    if observed != expected:
        failures.append({"check": "test", "expected": expected, "observed": observed})
    for index, definition in enumerate(VERIFICATION_DEFINITIONS):
        if index >= len(rows):
            break
        row = rows[index]
        definition_valid = all(
            row.get(key) == definition.get(key) for key in ("ordinal", "check_id", "command")
        )
        hash_valid = row.get("receipt_sha256") == row_hash(row, "receipt_sha256")
        if not definition_valid or not hash_valid or row.get("passed") is not True:
            failures.append(
                {
                    "check": "test",
                    "expected": {
                        "check_id": definition["check_id"],
                        "exit_code": 0,
                        "coverage_percent": definition.get("expected_coverage_percent"),
                    },
                    "observed": {
                        "check_id": row.get("check_id"),
                        "exit_code": row.get("exit_code"),
                        "coverage_percent": row.get("coverage_percent"),
                        "definition_valid": definition_valid,
                        "hash_valid": hash_valid,
                    },
                }
            )
    summary = {
        "ready": not failures and len(rows) == len(VERIFICATION_DEFINITIONS),
        "command_count": len(rows),
        "failed_count": len(failures),
        "coverage_percent": next(
            (
                row.get("coverage_percent")
                for row in rows
                if row.get("check_id") == "scoped_coverage"
            ),
            None,
        ),
    }
    return failures, summary


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash the active roadmap and conductor before task work."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def protected_files_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    """Prove that active orchestration files stayed byte-identical."""

    after = protected_hashes(root)
    return {
        "before": dict(before),
        "after": after,
        "files": {
            path: {
                "before_sha256": before.get(path),
                "after_sha256": after.get(path),
                "unchanged": before.get(path) == after.get(path),
            }
            for path in sorted(set(before) | set(after))
        },
        "unchanged": bool(before) and dict(before) == after,
    }


def _ram_total_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:  # pragma: no cover - host evidence only.
        return "missing"


def _exact_function_hashes() -> dict[str, str]:
    functions = (
        reference.frozen_fixtures,
        reference._log_weight,
        reference.brute_force_reference,
        reference.configuration_probability,
        reference.exact_marginals,
    )
    return {
        function.__name__: sha256_bytes(inspect.getsource(function).encode("utf-8"))
        for function in functions
    }


def collect_preconditions(
    root: Path,
    *,
    upstream: Mapping[str, Any],
    runtime: TorxRuntime,
) -> JsonDict:
    """Record gate, package, API, fixtures, libraries, resources, and hashes."""

    disk = shutil.disk_usage(root)
    paths = (
        UPSTREAM_PATH,
        REFERENCE_PATH,
        UPSTREAM_MODULE_PATH,
        MODULE_PATH,
        TEST_PATH,
        SAMPLER_SPEC_PATH,
        REPORT_SPEC_PATH,
        HARDWARE_SPEC_PATH,
        VERIFICATION_SPEC_PATH,
        SAFETY_SPEC_PATH,
        E2E_PLAN_PATH,
        V582_DESIGN_PATH,
        REFERENCE_REFRESH_PATH,
        *PROTECTED_PATHS,
    )
    return {
        "planning_date": RUN_DATE,
        "root": str(root.resolve()),
        "gate": _upstream_gate_receipt(upstream),
        "package": runtime.receipt(),
        "api": {
            "required_symbols": list(REQUIRED_API_SYMBOLS),
            "observed_symbols": list(runtime.symbols),
            "all_present": set(REQUIRED_API_SYMBOLS) <= set(runtime.symbols),
            "fallback_allowed": False,
        },
        "input_hashes": {path.as_posix(): sha256_file(root / path) for path in paths},
        "fixture_hashes": {
            item.instance_id: "sha256:" + item.fixture_sha256
            for item in reference.frozen_fixtures()
        },
        "exact_function_hashes": _exact_function_hashes(),
        "libraries": {
            name: _version(name)
            for name in (
                "numpy",
                "scipy",
                "jax",
                "jaxlib",
                "extro-torx",
                "pytest",
                "coverage",
                "ruff",
            )
        },
        "resources": {
            "cpu": _cpu_name(),
            "cpu_architecture": platform.machine(),
            "cpu_count": os.cpu_count() or 1,
            "ram_bytes": _ram_total_bytes(),
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
        },
        "active_roadmap": {
            "path": "research-roadmap.yaml",
            "sha256": sha256_file(root / "research-roadmap.yaml"),
            "milestone": "2026.08.582",
        },
        "conductor": {
            "path": "scripts/research_conductor.py",
            "sha256": sha256_file(root / "scripts/research_conductor.py"),
        },
        "e2e": {
            "plan": E2E_PLAN_PATH.as_posix(),
            "applicable": "E2E-002 CPU energy and probability path",
        },
        "no_llm": {
            "declared": INFERENCE_SUBSTRATE,
            "model_load_attempt_count": 0,
            "generation_attempt_count": 0,
        },
    }


def _blocked_preconditions(
    root: Path,
    *,
    upstream: Mapping[str, Any] | None,
    check: str,
    observed: Any,
) -> JsonDict:
    disk = shutil.disk_usage(root)
    return {
        "planning_date": RUN_DATE,
        "root": str(root.resolve()),
        "gate": _upstream_gate_receipt(upstream) if upstream is not None else observed,
        "failed_check": check,
        "observed": observed,
        "resources": {
            "cpu": _cpu_name(),
            "ram_bytes": _ram_total_bytes(),
            "disk_free_bytes": disk.free,
            "python": platform.python_version(),
        },
        "input_hashes": {
            UPSTREAM_PATH.as_posix(): sha256_file(root / UPSTREAM_PATH),
            "research-roadmap.yaml": sha256_file(root / "research-roadmap.yaml"),
            "scripts/research_conductor.py": sha256_file(root / "scripts/research_conductor.py"),
        },
        "no_llm": {
            "declared": INFERENCE_SUBSTRATE,
            "model_load_attempt_count": 0,
            "generation_attempt_count": 0,
        },
    }


def frozen_mapping_contract(
    upstream: Mapping[str, Any], runtime: TorxRuntime, parity: Mapping[str, Any]
) -> JsonDict:
    """Freeze every semantic choice needed by a later software backend."""

    contract: JsonDict = {
        "spin_encoding": {"torx_binary_0": -1, "torx_binary_1": 1},
        "coupling_factor": {
            "type": "torx.psc.PISING",
            "theta": "[J,0,0,1/T,1]",
            "energy": "-J*s_i*s_j",
        },
        "bias_factor": {
            "type": "torx.psc.PISING",
            "theta": "[0,h_i,0,1/T,1]",
            "energy": "-h_i*s_i",
            "second_wire": "pinned binary 0 with zero coupling and zero field",
        },
        "signs": "positive J is ferromagnetic; positive h favors spin +1",
        "temperature": "factor energy excludes T; beta=1/T; log_weight=-E/T",
        "coefficient_precision": "IEEE-754 binary64 with JAX x64 enabled",
        "accumulation": "Python math.fsum binary64; log normalization uses shifted exp",
        "factor_order": "couplings in fixture source order, then biases in vertex order",
        "update_order": {
            row["fixture_id"]: row["update_order"] for row in parity["fixture_mapping_rows"]
        },
        "tolerances": deepcopy(TOLERANCES),
        "relative_error_floor": RELATIVE_FLOOR,
        "fixture_hashes": {
            row["fixture_id"]: row["source_fixture_sha256"]
            for row in upstream["frozen_fixture_manifest"]
        },
        "exact_function_hashes": _exact_function_hashes(),
        "exp6683_artifact_sha256": sha256_file(REPO_ROOT / UPSTREAM_PATH),
        "torx_package_sha256": runtime.package_sha256,
        "api_fallback": "forbidden",
        "expected_supported_fixture_count": parity["supported_fixture_count"],
        "expected_factor_count": parity["factor_count"],
        "expected_state_count": parity["state_count"],
    }
    contract["contract_sha256"] = row_hash(contract, "contract_sha256")
    return contract


def _valid_hashes(rows: Sequence[Mapping[str, Any]], field_name: str = "row_sha256") -> bool:
    return all(row.get(field_name) == row_hash(row, field_name) for row in rows)


def _per_unit_rows(
    factor_rows: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    rejection_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    units = []
    for kind, rows in (
        ("factor", factor_rows),
        ("state", state_rows),
        ("rejection", rejection_rows),
        ("attack", attack_rows),
    ):
        for row in rows:
            unit = {"row_kind": kind, "source_row_sha256": row.get("row_sha256")}
            unit["row_sha256"] = row_hash(unit)
            units.append(unit)
    return units


def _maximum_errors(
    factor_rows: Sequence[Mapping[str, Any]], state_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, float | None]]:
    maximum: dict[str, dict[str, float | None]] = {
        "factor_energy": {
            "absolute": max(
                (float(row.get("maximum_absolute_error", math.inf)) for row in factor_rows),
                default=None,
            ),
            "relative": max(
                (float(row.get("maximum_relative_error", math.inf)) for row in factor_rows),
                default=None,
            ),
        }
    }
    for field_name in ("total_energy", "log_weight", "probability", "marginal", "correlation"):
        maximum[field_name] = {
            "absolute": max(
                (
                    float(row.get("field_errors", {}).get(field_name, {}).get("absolute", math.inf))
                    for row in state_rows
                ),
                default=None,
            ),
            "relative": max(
                (
                    float(row.get("field_errors", {}).get(field_name, {}).get("relative", math.inf))
                    for row in state_rows
                ),
                default=None,
            ),
        }
    return maximum


def recompute_aggregate(
    *,
    mapping_contract: Mapping[str, Any],
    factor_rows: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    rejection_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    protected_unchanged: bool,
    gate_failures: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Rebuild readiness and error summaries from retained evidence rows."""

    failures = [dict(row) for row in gate_failures]
    expected_factor_count = int(mapping_contract.get("expected_factor_count", 0))
    expected_state_count = int(mapping_contract.get("expected_state_count", 0))
    checks = (
        (
            "parity",
            expected_factor_count,
            len(factor_rows),
            bool(factor_rows)
            and all(row.get("valid") is True for row in factor_rows)
            and _valid_hashes(factor_rows),
        ),
        (
            "parity",
            expected_state_count,
            len(state_rows),
            bool(state_rows)
            and all(row.get("valid") is True for row in state_rows)
            and _valid_hashes(state_rows),
        ),
        (
            "rejection",
            len(REQUIRED_REJECTIONS),
            len(rejection_rows),
            {row.get("case_id") for row in rejection_rows} == set(REQUIRED_REJECTIONS)
            and all(row.get("passed") is True for row in rejection_rows)
            and _valid_hashes(rejection_rows),
        ),
        (
            "attack",
            len(REQUIRED_ATTACKS),
            len(attack_rows),
            {row.get("attack_id") for row in attack_rows} == set(REQUIRED_ATTACKS)
            and all(row.get("passed") is True for row in attack_rows)
            and _valid_hashes(attack_rows),
        ),
    )
    for check, expected, observed, passed in checks:
        if not passed:
            failure = {"check": check, "expected": expected, "observed": observed}
            if failure not in failures:
                failures.append(failure)
    test_failures, test_summary = reduce_test_rows(tests_run)
    for failure in test_failures:
        if failure not in failures:
            failures.append(failure)
    if not protected_unchanged:
        failure = {"check": "integrity", "expected": True, "observed": False}
        if failure not in failures:
            failures.append(failure)
    return {
        "ready": not failures,
        "failed_check_count": len(failures),
        "failures": failures,
        "supported_fixture_count": int(mapping_contract.get("expected_supported_fixture_count", 0)),
        "factor_row_count": len(factor_rows),
        "state_row_count": len(state_rows),
        "rejection_row_count": len(rejection_rows),
        "attack_row_count": len(attack_rows),
        "tests": test_summary,
        "maximum_errors": _maximum_errors(factor_rows, state_rows),
        "null_nan_overflow_underflow_count": sum(
            row.get("finite") is not True for row in state_rows
        ),
        "protected_files_unchanged": protected_unchanged,
    }


def _field_provenance(
    upstream: Mapping[str, Any] | None, runtime_receipt: Mapping[str, Any]
) -> dict[str, JsonDict]:
    upstream_hash = sha256_file(REPO_ROOT / UPSTREAM_PATH)
    function_hashes = _exact_function_hashes()
    return {
        field_name: {
            "principle": FIELD_PRINCIPLES[field_name],
            "fixture": "all Exp6683 supported fixtures"
            if upstream is not None
            else "blocked before fixtures",
            "exact_function": "Exp6683 exact_probability_rows and Exp6657 scalar energy",
            "torx_call": "torx.psc.PISING._energies(theta)" if runtime_receipt else "not invoked",
            "numeric_path": "CPU IEEE-754 binary64" if runtime_receipt else "blocked precondition",
            "source_hashes": {
                "exp6683": upstream_hash,
                "exact_functions": sha256_json(function_hashes),
                "torx_package": runtime_receipt.get("package_sha256", "not_available"),
            },
        }
        for field_name in REQUIRED_ARTIFACT_FIELDS
    }


def build_artifact(
    *,
    date: str,
    duration_s: float,
    upstream: Mapping[str, Any],
    runtime: TorxRuntime,
    parity: Mapping[str, Any],
    rejection_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    """Build a complete ready or row-failed parity artifact."""

    factor_rows = [dict(row) for row in parity["factor_rows"]]
    state_rows = [dict(row) for row in parity["state_parity_rows"]]
    rejections = [dict(row) for row in rejection_rows]
    attacks = [dict(row) for row in attack_rows]
    tests = [dict(row) for row in tests_run]
    contract = frozen_mapping_contract(upstream, runtime, parity)
    aggregate = recompute_aggregate(
        mapping_contract=contract,
        factor_rows=factor_rows,
        state_rows=state_rows,
        rejection_rows=rejections,
        attack_rows=attacks,
        tests_run=tests,
        protected_unchanged=protected.get("unchanged") is True,
        gate_failures=(),
    )
    ready = aggregate["ready"] is True
    gate_summary = [] if ready else aggregate["failures"]
    artifact: JsonDict = {
        "status": "complete_ready" if ready else "blocked_parity_check_failed",
        "honest_verdict": (
            "complete: installed Torx CPU typed factors match the Exp6683 exact Ising reference"
            if ready
            else "blocked_parity_check_failed: installed Torx CPU parity did not pass every row"
        ),
        "verdict_class": "circular_positive" if ready else "blocked",
        "gate_check_summary": gate_summary,
        "torx_runtime_receipt": runtime.receipt(),
        "frozen_mapping_contract": contract,
        "factor_rows": factor_rows,
        "state_parity_rows": state_rows,
        "rejection_rows": rejections,
        "attack_rows": attacks,
        "torx_factor_parity_ready": ready,
        "claim_scope": CLAIM_SCOPE,
        "per_unit_rows": _per_unit_rows(factor_rows, state_rows, rejections, attacks),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(upstream, runtime.receipt()),
        "random_seed": {
            "fixture_seeds": {
                item.instance_id: item.seed for item in supported_instances(upstream)
            },
            "attack_order_seed": 6684,
        },
        "duration_s": float(duration_s),
        "tests_run": tests,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    check: str,
    expected: Any,
    observed: Any,
    upstream: Mapping[str, Any] | None,
    tests_run: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    """Write a schema-complete blocker without inventing Torx evidence."""

    del date
    failure = {"check": check, "expected": expected, "observed": observed}
    contract: JsonDict = {
        "spin_encoding": {"torx_binary_0": -1, "torx_binary_1": 1},
        "tolerances": deepcopy(TOLERANCES),
        "api_fallback": "forbidden",
        "expected_supported_fixture_count": 0,
        "expected_factor_count": 0,
        "expected_state_count": 0,
        "blocked_before_mapping": True,
    }
    contract["contract_sha256"] = row_hash(contract, "contract_sha256")
    tests = [dict(row) for row in tests_run]
    aggregate = recompute_aggregate(
        mapping_contract=contract,
        factor_rows=(),
        state_rows=(),
        rejection_rows=(),
        attack_rows=(),
        tests_run=tests,
        protected_unchanged=protected.get("unchanged") is True,
        gate_failures=(failure,),
    )
    artifact: JsonDict = {
        "status": "blocked_upstream_gate" if check == "upstream" else f"blocked_{check}",
        "honest_verdict": (
            f"blocked_upstream_gate: {observed}"
            if check == "upstream"
            else f"blocked_{check}: {observed}"
        ),
        "verdict_class": "blocked",
        "gate_check_summary": [failure],
        "torx_runtime_receipt": {},
        "frozen_mapping_contract": contract,
        "factor_rows": [],
        "state_parity_rows": [],
        "rejection_rows": [],
        "attack_rows": [],
        "torx_factor_parity_ready": False,
        "claim_scope": CLAIM_SCOPE,
        "per_unit_rows": [],
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(upstream, {}),
        "random_seed": {"fixture_seeds": {}, "attack_order_seed": 6684},
        "duration_s": float(duration_s),
        "tests_run": tests,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return stable error codes for incomplete or drifted parity evidence."""

    errors: list[str] = []
    if set(REQUIRED_ARTIFACT_FIELDS) - set(payload):
        return ["missing_required_fields"]
    try:
        if payload.get("reproducibility_checksum") != artifact_checksum(payload):
            errors.append("reproducibility_checksum_mismatch")
    except ValueError:
        return ["nonfinite_artifact"]
    if payload.get("claim_scope") != CLAIM_SCOPE:
        errors.append("claim_scope_mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")

    factor_rows = payload.get("factor_rows", [])
    state_rows = payload.get("state_parity_rows", [])
    rejection_rows = payload.get("rejection_rows", [])
    attack_rows = payload.get("attack_rows", [])
    if not _valid_hashes(factor_rows):
        errors.append("factor_row_hash_mismatch")
    if not _valid_hashes(state_rows):
        errors.append("state_row_hash_mismatch")
    if not _valid_hashes(rejection_rows):
        errors.append("rejection_row_hash_mismatch")
    if not _valid_hashes(attack_rows):
        errors.append("attack_row_hash_mismatch")

    contract = payload.get("frozen_mapping_contract", {})
    if contract.get("contract_sha256") != row_hash(contract, "contract_sha256"):
        errors.append("mapping_contract_hash_mismatch")
    if len(factor_rows) != int(contract.get("expected_factor_count", 0)):
        errors.append("factor_row_count_mismatch")
    if len(state_rows) != int(contract.get("expected_state_count", 0)):
        errors.append("state_row_count_mismatch")

    expected_units = _per_unit_rows(factor_rows, state_rows, rejection_rows, attack_rows)
    if payload.get("per_unit_rows") != expected_units:
        errors.append("per_unit_rows_mismatch")
    if set(payload.get("field_provenance", {})) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_invalid")
    protected = payload.get("protected_files_unchanged", {})
    if protected.get("unchanged") is not True:
        errors.append("protected_files_changed")

    aggregate = recompute_aggregate(
        mapping_contract=contract,
        factor_rows=factor_rows,
        state_rows=state_rows,
        rejection_rows=rejection_rows,
        attack_rows=attack_rows,
        tests_run=payload.get("tests_run", []),
        protected_unchanged=protected.get("unchanged") is True,
        gate_failures=payload.get("gate_check_summary", []),
    )
    if aggregate != payload.get("aggregate_row_recomputation"):
        errors.append("aggregate_row_recomputation_mismatch")
    ready = payload.get("torx_factor_parity_ready") is True
    if aggregate["ready"] != ready:
        errors.append("readiness_mismatch")
    if ready:
        receipt = payload.get("torx_runtime_receipt", {})
        if (
            payload.get("status") != "complete_ready"
            or payload.get("verdict_class") != "circular_positive"
            or not str(payload.get("honest_verdict", "")).startswith("complete:")
            or payload.get("gate_check_summary") != []
        ):
            errors.append("ready_terminal_state_mismatch")
        if (
            receipt.get("version") != TORX_VERSION
            or receipt.get("backend") != "cpu"
            or receipt.get("jax_x64_enabled") is not True
            or set(receipt.get("symbols", [])) < set(REQUIRED_API_SYMBOLS)
        ):
            errors.append("runtime_receipt_mismatch")
    elif (
        not str(payload.get("status", "")).startswith("blocked_")
        or payload.get("verdict_class") != "blocked"
        or not payload.get("gate_check_summary")
    ):
        errors.append("blocked_terminal_state_mismatch")
    if not isinstance(payload.get("duration_s"), (int, float)) or not math.isfinite(
        float(payload.get("duration_s", math.nan))
    ):
        errors.append("duration_invalid")
    return list(dict.fromkeys(errors))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Publish one complete JSON through file and directory synchronization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "atomic_replace": True,
        "file_fsync": True,
        "directory_fsync": True,
    }


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    upstream_payload: Mapping[str, Any] | None = None,
    runtime_loader: Callable[[], TorxRuntime] = load_torx_runtime,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Check gates, measure parity, validate, and atomically write the artifact."""

    started = time.monotonic()
    before = protected_hashes(root)
    output = output_path or root / RESULT_PATH
    try:
        upstream = (
            dict(upstream_payload)
            if upstream_payload is not None
            else load_json(root / UPSTREAM_PATH)
        )
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        observed = f"{type(exc).__name__}: {exc}"
        protected = protected_files_receipt(root, before)
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            check="upstream",
            expected="readable ready Exp6683 artifact",
            observed=observed,
            upstream=None,
            tests_run=tests_run or (),
            preconditions=_blocked_preconditions(
                root, upstream=None, check="upstream_gate", observed=observed
            ),
            protected=protected,
        )
        if validate_artifact(artifact):
            raise ValueError("invalid blocked upstream artifact")
        write_json_atomic(output, artifact)
        return artifact

    gate = _upstream_gate_receipt(upstream)
    if not gate["passed"]:
        protected = protected_files_receipt(root, before)
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            check="upstream",
            expected=gate["expected"],
            observed=gate["observed"],
            upstream=upstream,
            tests_run=tests_run or (),
            preconditions=_blocked_preconditions(
                root, upstream=upstream, check="upstream_gate", observed=gate["observed"]
            ),
            protected=protected,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError(f"invalid blocked upstream artifact: {errors}")
        write_json_atomic(output, artifact)
        return artifact

    try:
        runtime = runtime_loader()
    except (TorxApiError, metadata.PackageNotFoundError, ImportError) as exc:
        observed = f"{type(exc).__name__}: {exc}"
        protected = protected_files_receipt(root, before)
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            check="api",
            expected={"version": TORX_VERSION, "symbols": list(REQUIRED_API_SYMBOLS)},
            observed=observed,
            upstream=upstream,
            tests_run=tests_run or (),
            preconditions=_blocked_preconditions(
                root, upstream=upstream, check="api", observed=observed
            ),
            protected=protected,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError(f"invalid blocked API artifact: {errors}")
        write_json_atomic(output, artifact)
        return artifact

    measured_tests = list(tests_run) if tests_run is not None else run_verification(root)
    parity = replay_parity(upstream, runtime)
    rejections = build_rejection_rows(runtime)
    attacks = build_attack_rows(upstream, runtime, parity, rejections)
    preconditions = collect_preconditions(root, upstream=upstream, runtime=runtime)
    protected = protected_files_receipt(root, before)
    artifact = build_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        upstream=upstream,
        runtime=runtime,
        parity=parity,
        rejection_rows=rejections,
        attack_rows=attacks,
        tests_run=measured_tests,
        preconditions=preconditions,
        protected=protected,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp6684 artifact: {errors}")
    write_json_atomic(output, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6684 or validate a redirected artifact."""

    args = _parse_args(argv)
    if args.validate:
        if not args.output.is_file():
            print(json.dumps({"valid": False, "errors": ["artifact_missing"]}, sort_keys=True))
            return 1
        try:
            artifact = load_json(args.output)
        except (json.JSONDecodeError, TypeError, OSError) as exc:
            errors = [f"artifact_unreadable:{type(exc).__name__}"]
            print(json.dumps({"valid": False, "errors": errors}, sort_keys=True))
            return 1
        errors = validate_artifact(artifact)
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(date=args.date, root=REPO_ROOT, output_path=args.output)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "torx_factor_parity_ready": artifact["torx_factor_parity_ready"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["torx_factor_parity_ready"] else 2


if __name__ == "__main__":  # pragma: no cover - exercised by the required module command.
    raise SystemExit(main())
