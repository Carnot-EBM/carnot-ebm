"""Qualify the Exp6657 exact Ising reference with owned evidence.

The repository-wide suite remains visible as a diagnostic. Only focused tests,
exact rows, rejection controls, and attacks can release this CPU reference.
The Exp6657 algorithm remains the source of every probability and sample.

Spec: REQ-REPORT-6683 and SCENARIO-REPORT-6683-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
from importlib import metadata
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
from typing import Any

import numpy as np

from carnot import experiment_6657_bounded_treewidth_ising_reference as reference


JsonDict = dict[str, Any]
CommandRunner = Callable[[list[str], Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260827"
INFERENCE_SUBSTRATE = "cpu_bounded_treewidth_exact_inference_no_llm"
RESULT_PATH = Path("results/experiment_6683_ising_reference_scope_receipt.json")
MODULE_PATH = Path("python/carnot/experiment_6683_ising_reference_scope_receipt.py")
TEST_PATH = Path("tests/python/test_experiment_6683_ising_reference_scope_receipt.py")
REFERENCE_SOURCE_PATH = Path("python/carnot/experiment_6657_bounded_treewidth_ising_reference.py")
REFERENCE_TEST_PATH = Path("tests/python/test_experiment_6657_bounded_treewidth_ising_reference.py")
REFERENCE_ARTIFACT_PATH = Path("results/experiment_6657_bounded_treewidth_ising_reference.json")
EXP6658_ARTIFACT_PATH = Path("results/experiment_6658_thermodynamic_schedule_ab.json")
EXP6639_SOURCE_PATH = Path("python/carnot/experiment_6639_kac_ward_planar_reference.py")
EXP6639_TEST_PATH = Path("tests/python/test_experiment_6639_kac_ward_planar_reference.py")
REPORT_SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REPORT_SPEC_PATH = REPO_ROOT / REPORT_SPEC_RELATIVE_PATH
SAMPLER_SPEC_PATH = Path("openspec/capabilities/samplers/spec.md")
HARDWARE_SPEC_PATH = Path("openspec/capabilities/hardware/spec.md")
PIPELINE_SPEC_PATH = Path("openspec/capabilities/pipeline/spec.md")
VERIFICATION_SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
V582_DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
REFERENCE_REFRESH_PATH = Path("research-references.md")
GLOBAL_CACHE_PATH = Path(".pytest_cache/v/cache/lastfailed")
PROTECTED_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))

OWNED_NODE_PREFIXES = (REFERENCE_TEST_PATH.as_posix(), TEST_PATH.as_posix())
EXPECTED_OWNED_NODE_COUNT = 66
REQUIRED_SPEC_ANCHORS = (
    "REQ-REPORT-6683",
    "SCENARIO-REPORT-6683-OWNED-READY",
    "SCENARIO-REPORT-6683-GLOBAL-DIAGNOSTIC",
    "SCENARIO-REPORT-6683-FAIL-CLOSED",
    "SCENARIO-REPORT-6683-ATOMIC-PROVENANCE",
)

_TEST_TARGETS = f"{REFERENCE_TEST_PATH.as_posix()} {TEST_PATH.as_posix()}"
_SOURCE_INCLUDE = (
    "*/experiment_6657_bounded_treewidth_ising_reference.py,"
    "*/experiment_6683_ising_reference_scope_receipt.py"
)
COLLECT_COMMAND = f".venv/bin/pytest --collect-only -q {_TEST_TARGETS} --no-cov -n 0 -o addopts="
FOCUSED_TEST_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--data-file=/tmp/carnot_exp6683_coverage "
    f"--include={_SOURCE_INCLUDE} -m pytest {_TEST_TARGETS} -q --no-cov -n 0 -o addopts="
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--data-file=/tmp/carnot_exp6683_coverage "
    f"--include={_SOURCE_INCLUDE} --fail-under=100 --show-missing"
)
RUFF_COMMAND = (
    f".venv/bin/ruff check {REFERENCE_SOURCE_PATH.as_posix()} {MODULE_PATH.as_posix()} "
    f"{REFERENCE_TEST_PATH.as_posix()} {TEST_PATH.as_posix()}"
)
FORMAT_COMMAND = (
    f".venv/bin/ruff format --check {REFERENCE_SOURCE_PATH.as_posix()} {MODULE_PATH.as_posix()} "
    f"{REFERENCE_TEST_PATH.as_posix()} {TEST_PATH.as_posix()}"
)
SPEC_COMMAND = (
    f".venv/bin/python scripts/check_spec_coverage.py {REFERENCE_TEST_PATH.as_posix()} "
    f"{TEST_PATH.as_posix()}"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"


def _definition(ordinal: int, check_id: str, command: str) -> JsonDict:
    return {
        "ordinal": ordinal,
        "check_id": check_id,
        "command": command,
        "expected_exit_code": 0,
        "expected_node_count": EXPECTED_OWNED_NODE_COUNT,
        "expected_coverage_percent": 100.0 if check_id == "scoped_coverage" else None,
    }


OWNED_CHECK_DEFINITIONS = (
    _definition(1, "focused_tests", FOCUSED_TEST_COMMAND),
    _definition(2, "scoped_coverage", COVERAGE_COMMAND),
    _definition(3, "ruff_check", RUFF_COMMAND),
    _definition(4, "format_check", FORMAT_COMMAND),
    _definition(5, "spec_coverage", SPEC_COMMAND),
)

REQUIRED_ATTACKS = {
    "disconnected_graph",
    "repeated_edge",
    "coupling_sign_change",
    "field_term",
    "low_positive_temperature",
    "high_positive_temperature",
    "zero_temperature",
    "invalid_decomposition",
    "unsupported_width",
    "precision_truncation",
    "update_order_drift",
    "missing_normalization",
    "degenerate_exact_sample",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "frozen_fixture_manifest",
    "decomposition_rows",
    "exact_probability_rows",
    "rejection_rows",
    "numeric_contract",
    "owned_test_rows",
    "global_suite_diagnostic",
    "attack_rows",
    "ising_reference_ready",
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
    "honest_verdict": "The verdict uses measured exact-reference evidence only.",
    "verdict_class": "A closed class keeps a ready reference as null infrastructure.",
    "gate_check_summary": "Observed values localize each failed owned gate.",
    "frozen_fixture_manifest": "Hashes bind immutable graph and numeric inputs.",
    "decomposition_rows": "Structural rows prove each tree-decomposition certificate.",
    "exact_probability_rows": "Enumeration rows provide recheckable exact authority.",
    "rejection_rows": "Expected and observed errors keep boundaries fail closed.",
    "numeric_contract": "Typed precision and order define portable semantics.",
    "owned_test_rows": "Commands and nodes define the task verification boundary.",
    "global_suite_diagnostic": "Repository state stays visible without gating readiness.",
    "attack_rows": "Adversarial controls test topology and numeric drift.",
    "ising_reference_ready": "One Boolean reduces complete owned evidence.",
    "per_unit_rows": "Raw units preserve every fixture, state, and check.",
    "aggregate_row_recomputation": "Readiness is rebuilt from retained rows.",
    "preconditions_checked": "Measured tools, inputs, and resources establish provenance.",
    "protected_files_unchanged": "Before and after hashes protect active operations.",
    "inference_substrate": "The CPU-only declaration prevents a model claim.",
    "verifier_is_oracle": "Exact inference explicitly defines the reference.",
    "field_provenance": "Each field names its source and numeric path.",
    "random_seed": "Frozen seeds preserve sampling and attack replay.",
    "duration_s": "Monotonic time records measured work.",
    "tests_run": "Command receipts make verification reproducible.",
    "reproducibility_checksum": "A canonical digest detects artifact changes.",
}

FROZEN_PATHS = (
    REFERENCE_ARTIFACT_PATH,
    REFERENCE_SOURCE_PATH,
    REFERENCE_TEST_PATH,
    EXP6658_ARTIFACT_PATH,
    EXP6639_SOURCE_PATH,
    EXP6639_TEST_PATH,
    MODULE_PATH,
    TEST_PATH,
    REPORT_SPEC_RELATIVE_PATH,
    SAMPLER_SPEC_PATH,
    HARDWARE_SPEC_PATH,
    PIPELINE_SPEC_PATH,
    VERIFICATION_SPEC_PATH,
    V582_DESIGN_PATH,
    REFERENCE_REFRESH_PATH,
    *PROTECTED_PATHS,
)


def canonical_json(value: Any) -> str:
    """Serialize one receipt with stable keys and no nonfinite values."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    """Prefix hashes so a digest cannot be mistaken for source text."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON instead of interpreter-specific object text."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Keep a missing required file distinct from an empty file."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else "missing"


def receipt_hash(value: Any, *, excluded: Sequence[str] = ()) -> str:
    """Hash a row after removing only its named self-referential fields."""

    if isinstance(value, Mapping):
        ignored = set(excluded)
        value = {key: item for key, item in value.items() if key not in ignored}
    return sha256_json(value)


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind all final fields except the checksum that stores this digest."""

    return receipt_hash(payload, excluded=("reproducibility_checksum",))


def load_json(path: Path) -> JsonDict:
    """Load one JSON object without repairing malformed evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return dict(value)


def _row_hash(row: Mapping[str, Any], field: str = "row_sha256") -> str:
    return receipt_hash(row, excluded=(field,))


def _energy(instance: reference.IsingInstance, state: Sequence[int]) -> float:
    return -instance.temperature * reference._log_weight(instance, state)


def _pair_correlations(brute: Mapping[str, Any]) -> dict[str, float]:
    states = brute["states"]
    probabilities = brute["probabilities"]
    n_spins = len(states[0])
    return {
        f"{left}-{right}": float(
            sum(
                probability * state[left] * state[right]
                for state, probability in zip(states, probabilities, strict=True)
            )
        )
        for left in range(n_spins)
        for right in range(left + 1, n_spins)
    }


def _decomposition_for_manifest(
    instance: reference.IsingInstance,
) -> reference.TreeDecomposition | None:
    try:
        return reference.deterministic_tree_decomposition(instance)
    except reference.UnsupportedGraphError:
        return None


def replay_reference() -> JsonDict:
    """Replay every frozen fixture through the unchanged Exp6657 functions."""

    manifests: list[JsonDict] = []
    decompositions: list[JsonDict] = []
    exact_rows: list[JsonDict] = []
    marginal_rows: list[JsonDict] = []
    correlation_rows: list[JsonDict] = []
    rejection_rows: list[JsonDict] = []

    for instance in reference.frozen_fixtures():
        decomposition = _decomposition_for_manifest(instance)
        manifest: JsonDict = {
            "fixture_id": instance.instance_id,
            "graph": {
                "n_spins": instance.n_spins,
                "edges": [list(edge[:2]) for edge in instance.edges],
            },
            "width": decomposition.width if decomposition is not None else None,
            "couplings": [float(edge[2]) for edge in instance.edges],
            "biases": list(instance.fields),
            "temperature": instance.temperature,
            "precision": "IEEE-754 binary64 via numpy.float64",
            "update_order": (
                list(decomposition.elimination_order) if decomposition is not None else []
            ),
            "seed": instance.seed,
            "expected_supported": instance.expected_supported,
            "expected_rejection": instance.expected_rejection,
            "source_fixture_sha256": "sha256:" + instance.fixture_sha256,
        }
        manifest["manifest_sha256"] = _row_hash(manifest, "manifest_sha256")
        manifests.append(manifest)

        if not instance.expected_supported:
            expected = instance.expected_rejection or "rejection"
            observed = "unexpectedly accepted"
            try:
                reference.solve_exact(instance)
            except reference.UnsupportedGraphError as exc:
                observed = str(exc)
            rejection: JsonDict = {
                "fixture_id": instance.instance_id,
                "input_class": "frozen_unsupported_fixture",
                "expected_failure": expected,
                "observed_failure": observed,
                "passed": expected in observed,
            }
            rejection["row_sha256"] = _row_hash(rejection)
            rejection_rows.append(rejection)
            continue

        assert decomposition is not None
        validation = reference.validate_tree_decomposition(instance, decomposition)
        separators = [
            sorted(set(decomposition.bags[left]) & set(decomposition.bags[right]))
            for left, right in decomposition.tree_edges
        ]
        decomposition_row: JsonDict = {
            "fixture_id": instance.instance_id,
            "width": decomposition.width,
            "bags": [list(bag) for bag in decomposition.bags],
            "separators": separators,
            "tree_edges": [list(edge) for edge in decomposition.tree_edges],
            "update_order": list(decomposition.elimination_order),
            "running_intersection": validation["running_intersection"],
            "passed": all(
                validation[key] for key in ("valid", "tree", "vertex_coverage", "edge_coverage")
            )
            and validation["running_intersection"]
            and validation["width"] <= reference.MAX_TREEWIDTH,
        }
        decomposition_row["row_sha256"] = _row_hash(decomposition_row)
        decompositions.append(decomposition_row)

        solution = reference.solve_exact(instance)
        brute = reference.brute_force_reference(instance)
        exact_marginals = reference.exact_marginals(instance, solution)
        correlations = _pair_correlations(brute)
        node_map = {str(index): float(value) for index, value in enumerate(brute["node_plus"])}
        algorithm_probabilities: list[float] = []
        for state, expected in zip(brute["states"], brute["probabilities"], strict=True):
            observed = reference.configuration_probability(instance, state, solution)
            algorithm_probabilities.append(observed)
            energy = _energy(instance, state)
            row: JsonDict = {
                "fixture_id": instance.instance_id,
                "state": list(state),
                "energy": energy,
                "unnormalized_weight": math.exp(-energy / instance.temperature),
                "partition_function": math.exp(solution.log_partition),
                "normalized_probability": float(expected),
                "algorithm_probability": observed,
                "probability_error": abs(observed - float(expected)),
                "node_marginals_plus": node_map,
                "pair_correlations": correlations,
                "passed": abs(observed - float(expected))
                <= reference.EXACT_TOLERANCES["probability"],
            }
            row["row_sha256"] = _row_hash(row)
            exact_rows.append(row)

        for index, expected in enumerate(brute["node_plus"]):
            observed = float(exact_marginals["node_plus"][index])
            row = {
                "fixture_id": instance.instance_id,
                "node": index,
                "expected_marginal_plus": float(expected),
                "observed_marginal_plus": observed,
                "error": abs(observed - float(expected)),
                "passed": abs(observed - float(expected)) <= reference.EXACT_TOLERANCES["marginal"],
            }
            row["row_sha256"] = _row_hash(row)
            marginal_rows.append(row)

        for pair, expected in correlations.items():
            left, right = (int(value) for value in pair.split("-"))
            observed = float(
                sum(
                    probability * state[left] * state[right]
                    for state, probability in zip(
                        brute["states"], algorithm_probabilities, strict=True
                    )
                )
            )
            row = {
                "fixture_id": instance.instance_id,
                "pair": pair,
                "expected_correlation": expected,
                "observed_correlation": observed,
                "error": abs(observed - expected),
                "passed": abs(observed - expected) <= reference.EXACT_TOLERANCES["marginal"],
            }
            row["row_sha256"] = _row_hash(row)
            correlation_rows.append(row)

    return {
        "frozen_fixture_manifest": manifests,
        "decomposition_rows": decompositions,
        "exact_probability_rows": exact_rows,
        "marginal_rows": marginal_rows,
        "correlation_rows": correlation_rows,
        "rejection_rows": rejection_rows,
    }


def _attack(attack_id: str, category: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    row: JsonDict = {
        "attack_id": attack_id,
        "category": category,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }
    row["row_sha256"] = _row_hash(row)
    return row


def _observed_rejection(instance: reference.IsingInstance) -> str:
    try:
        reference.solve_exact(instance)
    except reference.UnsupportedGraphError as exc:
        return str(exc)
    return "unexpectedly accepted"


def build_attack_rows(replay: Mapping[str, Any]) -> list[JsonDict]:
    """Run deterministic attacks against every requested semantic boundary."""

    disconnected = reference.IsingInstance(
        "attack_disconnected",
        4,
        ((0, 1, 0.4), (2, 3, -0.2)),
        (0.1, 0.0, -0.1, 0.05),
        1.0,
        6683001,
    )
    disconnected_row = reference.cross_check_fixture(disconnected)

    repeated = reference.IsingInstance(
        "attack_repeated", 2, ((0, 1, 0.2), (1, 0, -0.1)), (0.0, 0.0), 1.0, 6683002
    )
    repeated_error = _observed_rejection(repeated)

    positive = reference.IsingInstance(
        "attack_sign_positive", 2, ((0, 1, 0.4),), (0.0, 0.0), 1.0, 6683003
    )
    negative = reference.IsingInstance(
        "attack_sign_negative", 2, ((0, 1, -0.4),), (0.0, 0.0), 1.0, 6683004
    )
    positive_brute = reference.brute_force_reference(positive)
    negative_brute = reference.brute_force_reference(negative)
    positive_corr = _pair_correlations(positive_brute)["0-1"]
    negative_corr = _pair_correlations(negative_brute)["0-1"]
    sign_observed = {
        "partition_delta": abs(
            positive_brute["partition_function"] - negative_brute["partition_function"]
        ),
        "correlation_sum": positive_corr + negative_corr,
    }

    field = reference.IsingInstance("attack_field", 1, (), (0.3,), 0.7, 6683005)
    field_observed = reference.exact_marginals(field)["node_plus"][0]
    field_expected = 1.0 / (1.0 + math.exp(-2.0 * 0.3 / 0.7))

    low_temperature = reference.IsingInstance(
        "attack_low_temperature", 2, ((0, 1, 0.001),), (0.0, 0.0), 0.01, 6683006
    )
    high_temperature = reference.IsingInstance(
        "attack_high_temperature", 2, ((0, 1, 0.7),), (0.2, -0.1), 1.0e6, 6683007
    )
    low_row = reference.cross_check_fixture(low_temperature)
    high_row = reference.cross_check_fixture(high_temperature)

    zero_temperature = reference.IsingInstance(
        "attack_zero_temperature", 1, (), (0.0,), 0.0, 6683008
    )
    zero_error = _observed_rejection(zero_temperature)

    path = reference.IsingInstance(
        "attack_invalid_decomposition",
        3,
        ((0, 1, 0.2), (1, 2, 0.2)),
        (0.0, 0.0, 0.0),
        1.0,
        6683009,
    )
    invalid = reference.TreeDecomposition(((0, 1), (1, 2), (0, 2)), ((0, 1), (1, 2)), (0, 1, 2), 1)
    try:
        reference.validate_tree_decomposition(path, invalid)
        invalid_error = "unexpectedly accepted"
    except reference.UnsupportedGraphError as exc:
        invalid_error = str(exc)

    unsupported = next(
        item for item in reference.frozen_fixtures() if item.instance_id == "unsupported_k6_tw5"
    )
    width_error = _observed_rejection(unsupported)

    precise = reference.IsingInstance(
        "attack_precision", 2, ((0, 1, 0.1234567890123456),), (0.0, 0.0), 1.0, 6683010
    )
    truncated = reference.IsingInstance(
        "attack_precision_truncated",
        2,
        ((0, 1, float(np.float32(0.1234567890123456))),),
        (0.0, 0.0),
        1.0,
        6683010,
    )
    precise_probability = reference.brute_force_reference(precise)["probabilities"][0]
    truncated_probability = reference.brute_force_reference(truncated)["probabilities"][0]
    precision_observed = {
        "fixture_hash_changed": precise.fixture_sha256 != truncated.fixture_sha256,
        "probability_delta": abs(precise_probability - truncated_probability),
    }

    order_row = next(row for row in replay["decomposition_rows"] if len(row["update_order"]) > 1)
    drifted_order = list(reversed(order_row["update_order"]))
    order_observed = {
        "frozen_order": order_row["update_order"],
        "drifted_order": drifted_order,
        "drift_detected": drifted_order != order_row["update_order"],
    }

    singleton_rows = [
        row for row in replay["exact_probability_rows"] if row["fixture_id"] == "singleton_field"
    ]
    incomplete_mass = sum(row["normalized_probability"] for row in singleton_rows[:-1])

    degenerate = reference.IsingInstance("attack_degenerate", 2, (), (0.0, 0.0), 1.0, 6683011)
    sample_a = reference.independent_samples(degenerate, 256, 6683011)
    sample_b = reference.independent_samples(degenerate, 256, 6683011)
    sample_observed = {
        "replay_equal": sample_a["sample_sha256"] == sample_b["sample_sha256"],
        "unique_state_count": len({tuple(row) for row in sample_a["samples"]}),
        "sample_sha256": "sha256:" + sample_a["sample_sha256"],
    }

    return [
        _attack(
            "disconnected_graph", "topology", True, disconnected_row, disconnected_row["passed"]
        ),
        _attack(
            "repeated_edge",
            "topology",
            "duplicate edge",
            repeated_error,
            "duplicate edge" in repeated_error,
        ),
        _attack(
            "coupling_sign_change",
            "coupling",
            {"partition_delta": 0.0, "correlation_sum": 0.0},
            sign_observed,
            sign_observed["partition_delta"] <= 1.0e-14
            and abs(sign_observed["correlation_sum"]) <= 1.0e-14,
        ),
        _attack(
            "field_term",
            "field",
            field_expected,
            field_observed,
            abs(field_expected - field_observed) <= 1.0e-14,
        ),
        _attack("low_positive_temperature", "temperature", True, low_row, low_row["passed"]),
        _attack("high_positive_temperature", "temperature", True, high_row, high_row["passed"]),
        _attack(
            "zero_temperature", "temperature", "positive", zero_error, "positive" in zero_error
        ),
        _attack(
            "invalid_decomposition",
            "topology",
            "running intersection",
            invalid_error,
            "running intersection" in invalid_error,
        ),
        _attack(
            "unsupported_width", "topology", "treewidth", width_error, "treewidth" in width_error
        ),
        _attack(
            "precision_truncation",
            "precision",
            "detected",
            precision_observed,
            precision_observed["fixture_hash_changed"]
            and precision_observed["probability_delta"] > 0.0,
        ),
        _attack(
            "update_order_drift", "order", True, order_observed, order_observed["drift_detected"]
        ),
        _attack(
            "missing_normalization",
            "normalization",
            "mass differs from one",
            incomplete_mass,
            abs(incomplete_mass - 1.0) > reference.EXACT_TOLERANCES["normalization"],
        ),
        _attack(
            "degenerate_exact_sample",
            "sampling",
            {"replay_equal": True, "unique_state_count": 4},
            sample_observed,
            sample_observed["replay_equal"] and sample_observed["unique_state_count"] == 4,
        ),
    ]


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash the active roadmap and conductor before task work."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def protected_files_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    """Prove the active roadmap and conductor stayed byte-identical."""

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


def capture_frozen_hashes(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash every task input before verification changes process state."""

    return {path.as_posix(): sha256_file(root / path) for path in FROZEN_PATHS}


def _ram_total_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:  # pragma: no cover - depends on the host environment.
        return "missing"


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
        if match:
            return match.group(1).strip()
    return platform.processor() or platform.machine()


def collect_preconditions(root: Path, frozen_before: Mapping[str, str]) -> JsonDict:
    """Record exact inputs, tools, resources, libraries, and no-LLM use."""

    disk = shutil.disk_usage(root)
    return {
        "planning_date": RUN_DATE,
        "root": str(root.resolve()),
        "input_hashes": dict(frozen_before),
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
        "libraries": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "pytest", "coverage", "ruff")
        },
        "tools": {
            "pytest": (root / ".venv/bin/pytest").is_file(),
            "coverage": (root / ".venv/bin/coverage").is_file(),
            "ruff": (root / ".venv/bin/ruff").is_file(),
            "spec_coverage": (root / "scripts/check_spec_coverage.py").is_file(),
            "adversarial_verify": (root / "scripts/adversarial_verify.py").is_file(),
        },
        "e2e_plan": {
            "path": "ops/e2e-test-plan.md",
            "applicable_id": "E2E-001/E2E-002 sampling statistics",
            "focused_existing_test": "tests/python/test_e2e_training_sampling.py",
        },
        "no_llm": {
            "declared": INFERENCE_SUBSTRATE,
            "model_load_attempt_count": 0,
            "generation_attempt_count": 0,
            "exact_cpu_functions_only": True,
        },
    }


def _source_global_receipt(root: Path) -> JsonDict:
    source = load_json(root / REFERENCE_ARTIFACT_PATH)
    return dict(
        next(
            row
            for row in source.get("tests_run", [])
            if isinstance(row, Mapping) and row.get("scope") == "full_python_suite"
        )
    )


def load_global_suite_diagnostic(
    root: Path = REPO_ROOT, *, cache_path: Path | None = None
) -> JsonDict:
    """Keep repository-suite state visible without adding it to readiness."""

    path = cache_path or root / GLOBAL_CACHE_PATH
    cache_error: str | None = None
    if path.is_file():
        try:
            cache = load_json(path)
        except (json.JSONDecodeError, TypeError) as exc:
            cache = {}
            cache_error = f"{type(exc).__name__}: {exc}"
    else:
        cache = {}
        cache_error = f"FileNotFoundError: {path}"
    nodes = sorted(str(node) for node in cache)
    owned = [
        node
        for node in nodes
        if any(node.startswith(prefix + "::") for prefix in OWNED_NODE_PREFIXES)
    ]
    unrelated = [node for node in nodes if node not in set(owned)]
    source_receipt = _source_global_receipt(root)
    failed = bool(nodes) or source_receipt.get("exit_code") not in (0, None)
    row: JsonDict = {
        "command": FULL_SUITE_COMMAND,
        "failure_state": "failed" if failed else "passed",
        "source_exit_code": source_receipt.get("exit_code"),
        "source_summary": source_receipt.get("summary"),
        "failure_count": len(nodes),
        "owned_node_count": len(owned),
        "owned_failure_nodes": owned,
        "unrelated_failure_nodes": unrelated,
        "node_attribution_complete": len(nodes) == len(owned) + len(unrelated),
        "cache_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "cache_sha256": sha256_file(path),
        "cache_read_error": cache_error,
        "source_artifact": REFERENCE_ARTIFACT_PATH.as_posix(),
        "source_artifact_sha256": sha256_file(root / REFERENCE_ARTIFACT_PATH),
        "known_issue": "ops/known-issues.md:91",
        "gating": False,
        "readiness_influence": False,
    }
    row["receipt_sha256"] = receipt_hash(row)
    return row


def make_owned_test_row(
    definition: Mapping[str, Any],
    *,
    node_set: Sequence[str],
    exit_code: int,
    coverage_percent: float | None,
    duration_s: float,
    summary: str,
    output_sha256: str,
    spec_anchors: Sequence[str],
) -> JsonDict:
    """Bind one measured command to the exact focused test nodes."""

    nodes = list(node_set)
    passed = (
        exit_code == definition.get("expected_exit_code")
        and len(nodes) == definition.get("expected_node_count")
        and all(
            any(node.startswith(prefix + "::") for prefix in OWNED_NODE_PREFIXES) for node in nodes
        )
        and set(REQUIRED_SPEC_ANCHORS) <= set(spec_anchors)
        and (
            definition.get("expected_coverage_percent") is None
            or coverage_percent == definition.get("expected_coverage_percent")
        )
    )
    row: JsonDict = {
        "row_kind": "owned_test_command",
        "ordinal": definition.get("ordinal"),
        "check_id": definition.get("check_id"),
        "command": definition.get("command"),
        "node_count": len(nodes),
        "node_set": nodes,
        "exit_code": exit_code,
        "coverage_percent": coverage_percent,
        "spec_anchors": list(spec_anchors),
        "duration_s": duration_s,
        "summary": summary,
        "output_sha256": output_sha256,
        "passed": passed,
    }
    row["receipt_sha256"] = receipt_hash(row)
    return row


def reduce_owned_test_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], JsonDict]:
    """Reject missing, changed, duplicated, or reordered command receipts."""

    failures: list[JsonDict] = []
    expected_ids = [row["check_id"] for row in OWNED_CHECK_DEFINITIONS]
    observed_ids = [row.get("check_id") for row in rows]
    if len(rows) < len(OWNED_CHECK_DEFINITIONS):
        failures.append({"reason": "missing_receipt", "observed": observed_ids})
    if len(set(observed_ids)) != len(observed_ids):
        failures.append({"reason": "duplicate_receipt", "observed": observed_ids})
    if observed_ids != expected_ids and len(rows) == len(OWNED_CHECK_DEFINITIONS):
        failures.append({"reason": "receipt_order_mismatch", "observed": observed_ids})

    for index, definition in enumerate(OWNED_CHECK_DEFINITIONS):
        if index >= len(rows):
            continue
        row = rows[index]
        definition_matches = all(
            row.get(key) == definition.get(key) for key in ("ordinal", "check_id", "command")
        )
        if not definition_matches:
            failures.append({"reason": "definition_mismatch", "check_id": definition["check_id"]})
        receipt_valid = row.get("receipt_sha256") == receipt_hash(row, excluded=("receipt_sha256",))
        if not receipt_valid or row.get("passed") is not True:
            failures.append(
                {
                    "reason": "observed_value_mismatch",
                    "check_id": definition["check_id"],
                    "observed": {
                        "exit_code": row.get("exit_code"),
                        "coverage_percent": row.get("coverage_percent"),
                        "node_count": row.get("node_count"),
                        "receipt_valid": receipt_valid,
                    },
                }
            )

    node_sets = [tuple(row.get("node_set", [])) for row in rows]
    if node_sets and any(node_set != node_sets[0] for node_set in node_sets[1:]):
        failures.append({"reason": "node_set_mismatch", "observed": len(set(node_sets))})
    coverage = next(
        (row.get("coverage_percent") for row in rows if row.get("check_id") == "scoped_coverage"),
        None,
    )
    node_count = len(node_sets[0]) if node_sets else 0
    summary = {
        "ready": not failures
        and len(rows) == len(OWNED_CHECK_DEFINITIONS)
        and node_count == EXPECTED_OWNED_NODE_COUNT
        and coverage == 100.0,
        "command_count": len(rows),
        "node_count": node_count,
        "coverage_percent": coverage,
        "failed_count": len(failures),
    }
    return failures, summary


def default_command_runner(command: list[str], cwd: Path) -> JsonDict:
    """Run one process and retain measured output, exit, hash, and duration."""

    started = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    duration = time.monotonic() - started
    output = completed.stdout + completed.stderr
    lines = [line for line in output.splitlines() if line.strip()]
    return {
        "exit_code": completed.returncode,
        "output": output,
        "summary": lines[-1] if lines else "no output",
        "output_sha256": sha256_bytes(output.encode("utf-8")),
        "duration_s": duration,
    }


def _spec_anchors(root: Path) -> list[str]:
    del root
    text = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8") for path in (REFERENCE_TEST_PATH, TEST_PATH)
    )
    return sorted(set(re.findall(r"(?:REQ|SCENARIO)-[A-Z0-9-]+", text)))


def run_owned_verification(
    root: Path, *, command_runner: CommandRunner = default_command_runner
) -> list[JsonDict]:
    """Run the frozen focused tests, coverage, lint, format, and spec checks."""

    collection = command_runner(shlex.split(COLLECT_COMMAND), root)
    nodes = [
        line.strip()
        for line in str(collection.get("output", "")).splitlines()
        if any(line.startswith(prefix + "::") for prefix in OWNED_NODE_PREFIXES)
    ]
    anchors = _spec_anchors(root)
    rows: list[JsonDict] = []
    for definition in OWNED_CHECK_DEFINITIONS:
        result = command_runner(shlex.split(str(definition["command"])), root)
        coverage_percent = None
        if definition["check_id"] == "scoped_coverage":
            match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", str(result["output"]))
            coverage_percent = float(match.group(1)) if match else None
        rows.append(
            make_owned_test_row(
                definition,
                node_set=nodes,
                exit_code=int(result["exit_code"]),
                coverage_percent=coverage_percent,
                duration_s=float(result["duration_s"]),
                summary=str(result["summary"]),
                output_sha256=str(result["output_sha256"]),
                spec_anchors=anchors,
            )
        )
    return rows


def numeric_contract() -> JsonDict:
    """State the portable coefficient, accumulation, and update semantics."""

    return {
        "coefficient_type": "numpy.float64",
        "precision_bits": 64,
        "coefficient_input_contract": "finite Python real converted to IEEE-754 binary64",
        "tolerances": dict(reference.EXACT_TOLERANCES),
        "sampling_tolerances": dict(reference.SAMPLE_TOLERANCES),
        "accumulation": "binary64 factor sums with numpy.logaddexp elimination",
        "update_order": "deterministic min-fill with vertex-id tie break",
        "sampling_order": "reverse elimination order; one ancestral traversal per draw",
        "rng": "NumPy PCG64 with one fresh generator per fixture seed",
        "energy": "E=-sum(J_ij*s_i*s_j)-sum(h_i*s_i); each undirected edge appears once",
        "temperature": "finite positive binary64; probability=exp(-E/T)/Z",
    }


def _rows_have_valid_hash(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    return all(row.get(field) == _row_hash(row, field) for row in rows)


def recompute_aggregate(
    *,
    fixture_manifest: Sequence[Mapping[str, Any]],
    decomposition_rows: Sequence[Mapping[str, Any]],
    exact_probability_rows: Sequence[Mapping[str, Any]],
    marginal_rows: Sequence[Mapping[str, Any]],
    correlation_rows: Sequence[Mapping[str, Any]],
    rejection_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    owned_test_rows: Sequence[Mapping[str, Any]],
    frozen_inputs_unchanged: bool,
    protected_files_unchanged: bool,
    global_failure_count: int,
) -> JsonDict:
    """Rebuild readiness without consulting the global-suite result."""

    _, owned_summary = reduce_owned_test_rows(owned_test_rows)
    masses: dict[str, float] = defaultdict(float)
    for row in exact_probability_rows:
        masses[str(row.get("fixture_id"))] += float(row.get("normalized_probability", 0.0))
    checks = {
        "fixture_support": len(fixture_manifest) == 15
        and sum(bool(row.get("expected_supported")) for row in fixture_manifest) == 12
        and _rows_have_valid_hash(fixture_manifest, "manifest_sha256"),
        "decompositions": len(decomposition_rows) == 12
        and all(row.get("passed") is True for row in decomposition_rows)
        and _rows_have_valid_hash(decomposition_rows, "row_sha256"),
        "exact_fields": bool(exact_probability_rows)
        and all(row.get("passed") is True for row in exact_probability_rows)
        and _rows_have_valid_hash(exact_probability_rows, "row_sha256"),
        "normalization": len(masses) == 12
        and all(
            abs(mass - 1.0) <= reference.EXACT_TOLERANCES["normalization"]
            for mass in masses.values()
        ),
        "marginals": bool(marginal_rows)
        and all(row.get("passed") is True for row in marginal_rows)
        and _rows_have_valid_hash(marginal_rows, "row_sha256"),
        "correlations": bool(correlation_rows)
        and all(row.get("passed") is True for row in correlation_rows)
        and _rows_have_valid_hash(correlation_rows, "row_sha256"),
        "rejections": len(rejection_rows) >= 3
        and all(row.get("passed") is True for row in rejection_rows)
        and _rows_have_valid_hash(rejection_rows, "row_sha256"),
        "owned_tests": owned_summary["ready"],
        "attacks": REQUIRED_ATTACKS <= {row.get("attack_id") for row in attack_rows}
        and all(row.get("passed") is True for row in attack_rows)
        and _rows_have_valid_hash(attack_rows, "row_sha256"),
        "frozen_inputs": frozen_inputs_unchanged,
        "protected_files": protected_files_unchanged,
    }
    counts = {
        "fixtures": len(fixture_manifest),
        "decompositions": len(decomposition_rows),
        "states": len(exact_probability_rows),
        "marginals": len(marginal_rows),
        "correlations": len(correlation_rows),
        "rejections": len(rejection_rows),
        "owned_test_rows": len(owned_test_rows),
        "owned_test_nodes": owned_summary["node_count"],
        "attacks": len(attack_rows),
        "global_failures": global_failure_count,
    }
    return {
        "checks": checks,
        "counts": counts,
        "maximum_errors": {
            "probability": max(
                (float(row["probability_error"]) for row in exact_probability_rows),
                default=None,
            ),
            "marginal": max((float(row["error"]) for row in marginal_rows), default=None),
            "correlation": max((float(row["error"]) for row in correlation_rows), default=None),
            "normalization": max((abs(mass - 1.0) for mass in masses.values()), default=None),
        },
        "global_suite_in_reducer": False,
        "ready": all(checks.values()),
    }


def _per_unit_rows(
    replay: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
    owned_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    units: list[JsonDict] = []
    families = (
        ("fixture", replay["frozen_fixture_manifest"]),
        ("state", replay["exact_probability_rows"]),
        ("marginal", replay["marginal_rows"]),
        ("correlation", replay["correlation_rows"]),
        ("rejection", replay["rejection_rows"]),
        ("attack", attacks),
    )
    for unit_type, rows in families:
        for row in rows:
            unit = {"unit_type": unit_type, "row": deepcopy(row)}
            unit["unit_sha256"] = _row_hash(unit, "unit_sha256")
            units.append(unit)
    _, summary = reduce_owned_test_rows(owned_rows)
    nodes = list(owned_rows[0].get("node_set", [])) if owned_rows else []
    for node in nodes:
        unit = {"unit_type": "test", "node_id": node, "passed": summary["ready"]}
        unit["unit_sha256"] = _row_hash(unit, "unit_sha256")
        units.append(unit)
    return units


def _field_provenance(root: Path) -> dict[str, JsonDict]:
    source_hash = sha256_file(root / MODULE_PATH)
    reference_hash = sha256_file(root / REFERENCE_SOURCE_PATH)
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "fixture": "Exp6657 frozen fixtures or task-owned process receipt",
            "exact_function": "experiment_6657 exact functions and experiment_6683 row reducer",
            "numeric_path": "CPU NumPy binary64; no LLM or accelerator",
            "source": MODULE_PATH.as_posix(),
            "source_sha256": source_hash,
            "reference_sha256": reference_hash,
            "schema_anchor": "REQ-REPORT-6683",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_artifact(
    *,
    root: Path,
    date: str,
    duration_s: float,
    owned_test_rows: Sequence[Mapping[str, Any]],
    global_suite_diagnostic: Mapping[str, Any],
    frozen_before: Mapping[str, str],
    protected_before: Mapping[str, str],
    replay: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build one terminal artifact from exact rows and owned command receipts."""

    rows = dict(replay or replay_reference())
    attacks = build_attack_rows(rows)
    frozen_after = capture_frozen_hashes(root)
    frozen_unchanged = bool(frozen_before) and dict(frozen_before) == frozen_after
    protection = protected_files_receipt(root, protected_before)
    aggregate = recompute_aggregate(
        fixture_manifest=rows["frozen_fixture_manifest"],
        decomposition_rows=rows["decomposition_rows"],
        exact_probability_rows=rows["exact_probability_rows"],
        marginal_rows=rows["marginal_rows"],
        correlation_rows=rows["correlation_rows"],
        rejection_rows=rows["rejection_rows"],
        attack_rows=attacks,
        owned_test_rows=owned_test_rows,
        frozen_inputs_unchanged=frozen_unchanged,
        protected_files_unchanged=protection["unchanged"],
        global_failure_count=int(global_suite_diagnostic.get("failure_count", 0)),
    )
    failures = [
        {"check": check, "expected": True, "observed_value": value}
        for check, value in aggregate["checks"].items()
        if value is not True
    ]
    ready = aggregate["ready"]
    status = "complete_ready" if ready else f"blocked_{failures[0]['check']}_failed"
    preconditions = collect_preconditions(root, frozen_before)
    preconditions["frozen_inputs"] = {
        "before": dict(frozen_before),
        "after": frozen_after,
        "unchanged": frozen_unchanged,
    }
    artifact: JsonDict = {
        "schema": "carnot.experiment_6683.ising_reference_scope_receipt.v1",
        "planning_date": date,
        "status": status,
        "honest_verdict": (
            "complete: bounded-treewidth exact Ising reference is ready under task-owned evidence"
            if ready
            else f"blocked_{failures[0]['check']}_failed: exact-reference readiness is not established"
        ),
        "verdict_class": None if ready else "blocked",
        "gate_check_summary": failures,
        "frozen_fixture_manifest": deepcopy(rows["frozen_fixture_manifest"]),
        "decomposition_rows": deepcopy(rows["decomposition_rows"]),
        "exact_probability_rows": deepcopy(rows["exact_probability_rows"]),
        "rejection_rows": deepcopy(rows["rejection_rows"]),
        "numeric_contract": numeric_contract(),
        "owned_test_rows": [dict(row) for row in owned_test_rows],
        "global_suite_diagnostic": dict(global_suite_diagnostic),
        "attack_rows": attacks,
        "ising_reference_ready": ready,
        "per_unit_rows": _per_unit_rows(rows, attacks, owned_test_rows),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protection,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(root),
        "random_seed": {
            "fixture_seeds": {
                row["fixture_id"]: row["seed"] for row in rows["frozen_fixture_manifest"]
            },
            "attack_order_seed": 6683,
            "degenerate_sample_seed": 6683011,
        },
        "duration_s": float(duration_s),
        "tests_run": [dict(row) for row in owned_test_rows],
        "reproducibility_checksum": "",
        "_marginal_rows": deepcopy(rows["marginal_rows"]),
        "_correlation_rows": deepcopy(rows["correlation_rows"]),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return stable error codes for incomplete or drifted evidence."""

    errors: list[str] = []
    if set(REQUIRED_ARTIFACT_FIELDS) - set(payload):
        return ["missing_required_fields"]
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("numeric_contract") != numeric_contract():
        errors.append("numeric_contract_mismatch")
    global_row = payload.get("global_suite_diagnostic", {})
    if global_row.get("gating") is not False or global_row.get("readiness_influence") is not False:
        errors.append("global_diagnostic_gating")
    owned_rows = payload.get("owned_test_rows", [])
    expected_ids = [row["check_id"] for row in OWNED_CHECK_DEFINITIONS]
    structural_owned_receipts_valid = [
        row.get("check_id") for row in owned_rows
    ] == expected_ids and all(
        row.get("receipt_sha256") == receipt_hash(row, excluded=("receipt_sha256",))
        for row in owned_rows
    )
    if not structural_owned_receipts_valid:
        errors.append("owned_test_receipts_invalid")
    if any(row.get("passed") is not True for row in payload.get("exact_probability_rows", [])):
        errors.append("exact_rows_failed")
    if any(row.get("passed") is not True for row in payload.get("attack_rows", [])):
        errors.append("attack_rows_failed")
    if payload.get("protected_files_unchanged", {}).get("unchanged") is not True:
        errors.append("protected_files_changed")
    if set(payload.get("field_provenance", {})) < set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_invalid")
    frozen = payload.get("preconditions_checked", {}).get("frozen_inputs", {})
    aggregate = recompute_aggregate(
        fixture_manifest=payload.get("frozen_fixture_manifest", []),
        decomposition_rows=payload.get("decomposition_rows", []),
        exact_probability_rows=payload.get("exact_probability_rows", []),
        marginal_rows=payload.get("_marginal_rows", []),
        correlation_rows=payload.get("_correlation_rows", []),
        rejection_rows=payload.get("rejection_rows", []),
        attack_rows=payload.get("attack_rows", []),
        owned_test_rows=payload.get("owned_test_rows", []),
        frozen_inputs_unchanged=frozen.get("unchanged") is True,
        protected_files_unchanged=payload.get("protected_files_unchanged", {}).get("unchanged")
        is True,
        global_failure_count=int(global_row.get("failure_count", 0)),
    )
    if aggregate != payload.get("aggregate_row_recomputation"):
        errors.append("aggregate_row_recomputation_mismatch")
    ready = payload.get("ising_reference_ready") is True
    if aggregate["ready"] != ready:
        errors.append("readiness_mismatch")
    if ready:
        if payload.get("status") != "complete_ready" or payload.get("verdict_class") is not None:
            errors.append("ready_terminal_state_mismatch")
        if not str(payload.get("honest_verdict", "")).startswith("complete:"):
            errors.append("honest_verdict_mismatch")
        if payload.get("gate_check_summary") != []:
            errors.append("ready_gate_summary_mismatch")
    else:
        if not str(payload.get("status", "")).startswith("blocked_"):
            errors.append("blocked_terminal_state_mismatch")
        if payload.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if not payload.get("gate_check_summary"):
            errors.append("blocked_gate_summary_mismatch")
    if not isinstance(payload.get("duration_s"), (int, float)) or payload.get("duration_s", -1) < 0:
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
        if temporary.exists():  # pragma: no cover - only a failed replace leaves this file.
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
    owned_test_rows: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Measure owned checks, validate the receipt, and write it atomically."""

    started = time.monotonic()
    frozen_before = capture_frozen_hashes(root)
    protected_before = protected_hashes(root)
    measured_rows = (
        list(owned_test_rows) if owned_test_rows is not None else run_owned_verification(root)
    )
    replay = replay_reference()
    diagnostic = load_global_suite_diagnostic(root)
    artifact = build_artifact(
        root=root,
        date=date,
        duration_s=time.monotonic() - started,
        owned_test_rows=measured_rows,
        global_suite_diagnostic=diagnostic,
        frozen_before=frozen_before,
        protected_before=protected_before,
        replay=replay,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp6683 artifact: {errors}")
    write_json_atomic(output_path or root / RESULT_PATH, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6683 or validate a redirected artifact."""

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
                "ising_reference_ready": artifact["ising_reference_ready"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["ising_reference_ready"] else 2


if __name__ == "__main__":  # pragma: no cover - exercised by the required module command.
    raise SystemExit(main())
