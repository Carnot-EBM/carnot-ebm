"""Exp6152 typed stochastic constraint IR.

Spec refs: REQ-SAMPLE-6152, SCENARIO-SAMPLE-6152-VALIDATION,
SCENARIO-SAMPLE-6152-EXACT, SCENARIO-SAMPLE-6152-SERIALIZATION-TORX.

The experiment keeps the stochastic program small enough to enumerate exactly.
It models one Exp6145 access-control workflow: a candidate item is proposed,
the deterministic Horn-style facts derive eligibility, and a stochastic
strategy-clean bit gates final admission. The exact finite enumerator is the
semantic oracle; the Torx path is a real pinned API smoke, not a replacement
for Carnot's workflow executor.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_6145_constraint_shift_stream as exp6145


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6152_typed_stochastic_constraint_ir.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6152_typed_stochastic_constraint_ir.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6152_typed_stochastic_constraint_ir.py")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
IR_SCHEMA_VERSION = "carnot.typed_stochastic_constraint_ir.v1"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_6152.typed_stochastic_constraint_ir.v1"
EXPERIMENT_ID = "experiment_6152_typed_stochastic_constraint_ir"
RUN_DATE = "20260805"
EXACT_TOLERANCE = 1.0e-12
TORX_PINNED_PACKAGE_VERSION = "0.0.1"
TORX_PINNED_REPOSITORY = "https://github.com/extropic-ai/torx"
TORX_PINNED_REPOSITORY_COMMIT = "f1fc858ed950ecd41935d15c06d0ec7c5e0674ae"
TORX_PINNED_WHEEL_SHA256 = "e51d6efe0a8bc62fb4b2b417d5e4ac8190e3fb22c9d14d9342c207afdc64a23c"
TORX_PINNED_SDIST_SHA256 = "c7bbeb0c39e5c7f9a241b9d3f68c224b8dddbc29674ffc15c78d21d05eb0149a"
JAX_CPU_SUBSTRATE = "jax_cpu_exact_stochastic_program"
CARNOT_ONLY_SUBSTRATE = "carnot_only_blocked_torx_compatibility"
VERIFIER_IS_ORACLE = True

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
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6145_constraint_shift_stream.py"),
    Path("python/carnot/experiment_5896_typed_constraint_ir_fixture.py"),
    Path("python/carnot/samplers/backend.py"),
    Path("python/carnot/samplers/thrml_init.py"),
    Path("python/carnot/samplers"),
    Path("python/carnot/constraints"),
    Path("crates/carnot-ising"),
    Path("crates/carnot-gibbs"),
    Path("results/experiment_6145_constraint_shift_stream.json"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
)

FOCUSED_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/pytest "
    "tests/python/test_experiment_6152_typed_stochastic_constraint_ir.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6152_typed_stochastic_constraint_ir.py "
    "-m pytest tests/python/test_experiment_6152_typed_stochastic_constraint_ir.py "
    "-q --no-cov -n 0 && JAX_PLATFORMS=cpu .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6152_typed_stochastic_constraint_ir.py --fail-under=100"
)
GLOBAL_PYTEST_COMMAND = "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6152_typed_stochastic_constraint_ir.py"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6152_typed_stochastic_constraint_ir.py "
    "tests/python/test_experiment_6152_typed_stochastic_constraint_ir.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6152_typed_stochastic_constraint_ir.py "
    "tests/python/test_experiment_6152_typed_stochastic_constraint_ir.py"
)
E2E_SERIALIZATION_COMMAND = (
    "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python/test_e2e_serialization.py -q --no-cov -n 0"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6152_typed_stochastic_constraint_ir.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py "
    "ops/changelog.md ops/status.md _bmad/traceability.md"
)
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
    "source_workflow_validator_sampler_and_exclusion_hashes",
    "torx_package_version_commit_import_and_api_receipts",
    "ir_schema_version_types_kernels_and_graph_contract",
    "compiler_executor_adapter_and_test_paths",
    "exact_enumeration_case_counts",
    "support_conditional_joint_normalization_and_marginal_deltas",
    "deterministic_impossible_batch_seed_and_serialization_controls",
    "wire_order_category_type_cycle_dangling_and_invalid_mass_negative_controls",
    "torx_compatibility_scope",
    "deterministic_rebuild_checksum",
    "typed_stochastic_ir_ready_score",
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
        "A terminal state distinguishes ready, Carnot-only, null, and blocked compiler boundaries."
    ),
    "preconditions_checked": (
        "Hashes JAX CPU mode, Exp6145 workflow evidence, validators, sampler APIs, Torx metadata, "
        "exclusions, outputs, and protected files before construction."
    ),
    "structured_gate_receipt": (
        "Exp6145 readiness and sidecar replay must pass before the stochastic workflow is compiled."
    ),
    "source_workflow_validator_sampler_and_exclusion_hashes": (
        "Content hashes bind the source workflow, exact validators, sampler APIs, Torx package "
        "metadata or commit, exclusions, output paths, and protected files."
    ),
    "torx_package_version_commit_import_and_api_receipts": (
        "Real importable Torx package/version/commit/API evidence is recorded; resemblance to "
        "public docs is not compatibility."
    ),
    "ir_schema_version_types_kernels_and_graph_contract": (
        "The versioned IR contract names wire types, kernel schemas, edge rules, state shapes, "
        "and normalization gates."
    ),
    "compiler_executor_adapter_and_test_paths": (
        "Paths identify the compiler, exact executor, optional adapter, tests, and artifact boundary."
    ),
    "exact_enumeration_case_counts": (
        "Finite state counts prove bounded exhaustive enumeration rather than sampled evidence."
    ),
    "support_conditional_joint_normalization_and_marginal_deltas": (
        "Exact enumeration, not sampling, is authoritative on bounded cases."
    ),
    "deterministic_impossible_batch_seed_and_serialization_controls": (
        "Deterministic factors, impossible states, batch shape, seed replay, and JSON round trips "
        "are checked together."
    ),
    "wire_order_category_type_cycle_dangling_and_invalid_mass_negative_controls": (
        "Negative controls prove validators and the independent reference catch subtle graph and "
        "indexing failures."
    ),
    "torx_compatibility_scope": (
        "Names the real exercised version/API; source resemblance is not compatibility."
    ),
    "deterministic_rebuild_checksum": (
        "A second construction must reproduce the same program and exact-semantics commitment."
    ),
    "typed_stochastic_ir_ready_score": (
        "Exactly one only when exact semantics, controls, serialization, protected files, tests, "
        "and real Torx compatibility pass."
    ),
    "protected_files_unchanged": (
        "Conductor and reconciler-owned files remain byte-identical during artifact construction."
    ),
    "duration_s": "Measured deterministic construction time is reported without padding.",
    "inference_substrate": (
        "Use `jax_cpu_exact_stochastic_program` or `carnot_only_blocked_torx_compatibility` honestly."
    ),
    "verifier_is_oracle": (
        "The independent exact finite enumerator is the probability oracle for bounded cases."
    ),
    "missing_verifier_gaps": "Any absent Torx or non-enumerated semantic evidence is explicit.",
    "field_provenance": (
        "Every field traces to the prompt, spec, workflow, source, tests, command receipts, or "
        "package metadata."
    ),
    "test_commands": (
        "Commands document focused unit/spec coverage, structured gate, exact probability, "
        "Torx adapter, serialization, negative controls, protected-file, E2E, global pytest, "
        "and root-clutter checks."
    ),
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": (
        "The artifact hash detects source, schema, probability, adapter, test, or protected-file drift."
    ),
    "honest_verdict": (
        "Use `complete_ready:`, `complete_carnot_only:`, `complete_null:`, or `blocked:` and state "
        "the exact compiler boundary."
    ),
}


class TypedStochasticIRValidationError(ValueError):
    """Raised when a stochastic IR payload has ambiguous or invalid semantics."""


class TypedStochasticIRReplayError(ValueError):
    """Raised when a written Exp6152 artifact no longer replays exactly."""


@dataclass(frozen=True)
class Wire:
    identifier: str
    kind: str
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class Kernel:
    identifier: str
    kind: str
    inputs: tuple[str, ...]
    output: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class StochasticProgram:
    schema_version: str
    program_id: str
    wires: tuple[Wire, ...]
    kernels: tuple[Kernel, ...]
    metadata: Mapping[str, Any]


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in stable ASCII byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash one file by bytes, independent of path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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


def compile_exp6145_bounded_workflow() -> StochasticProgram:
    """Compile one bounded Exp6145 access-control workflow into PSC/DFG-style IR."""

    config = exp6145.FAMILY_CONFIGS[0]
    source_ir = exp6145._variant_ir(config, 0, "canonical")
    certificate = exp5896.certify_ir(source_ir)
    wires = (
        Wire("candidate_item", "categorical", ("ac_e0_0", "ac_e0_1", "ac_e0_2")),
        Wire("strategy_clean", "binary"),
        Wire("member_group", "categorical", ("ac_g0_0", "ac_g0_1")),
        Wire(
            "clearance", "categorical", ("clearance_1", "clearance_2", "clearance_3", "clearance_4")
        ),
        Wire("gate_open", "binary"),
        Wire("blocked", "binary"),
        Wire("clearance_ok", "binary"),
        Wire("eligible", "binary"),
        Wire("accepted", "binary"),
    )
    kernels = (
        Kernel(
            "sample_candidate_item",
            "categorical_prior",
            (),
            "candidate_item",
            {"probabilities": [0.4, 0.5, 0.1], "seed_role": "candidate_item_root"},
        ),
        Kernel(
            "sample_strategy_clean",
            "bernoulli_prior",
            (),
            "strategy_clean",
            {"p_true": 0.9, "seed_role": "strategy_clean_root"},
        ),
        Kernel(
            "member_group_lookup",
            "deterministic_lookup",
            ("candidate_item",),
            "member_group",
            {"table": [0, 1, 0]},
        ),
        Kernel(
            "clearance_lookup",
            "deterministic_lookup",
            ("candidate_item",),
            "clearance",
            {"table": [2, 3, 1]},
        ),
        Kernel(
            "gate_open_lookup",
            "deterministic_lookup",
            ("member_group",),
            "gate_open",
            {"table": [1, 1]},
        ),
        Kernel(
            "blocked_lookup",
            "deterministic_lookup",
            ("candidate_item",),
            "blocked",
            {"table": [0, 1, 0]},
        ),
        Kernel(
            "clearance_ok_lookup",
            "deterministic_lookup",
            ("clearance",),
            "clearance_ok",
            {"table": [0, 1, 1, 1]},
        ),
        Kernel(
            "eligible_truth_table",
            "deterministic_truth_table",
            ("gate_open", "blocked", "clearance_ok"),
            "eligible",
            {"table": [0, 0, 0, 0, 0, 1, 0, 0]},
        ),
        Kernel(
            "accepted_truth_table",
            "deterministic_truth_table",
            ("eligible", "strategy_clean"),
            "accepted",
            {"table": [0, 0, 0, 1]},
        ),
    )
    metadata = {
        "source": "Exp6145 access_control canonical template t00 with stochastic strategy-clean gate",
        "source_event_id": "exp6145-event-000000",
        "source_base_template_id": "exp6145.access_control.t00",
        "source_family": config.family,
        "source_constraint_ir_hash": exp6145.sha256_json(source_ir),
        "source_behavior_hash": certificate["python"].get("behavior_hash"),
        "source_exact_answer": certificate["python"].get("query_bindings"),
        "strategy_clean_probability": 0.9,
        "candidate_item_probabilities": [0.4, 0.5, 0.1],
    }
    program = StochasticProgram(
        IR_SCHEMA_VERSION,
        "exp6145_access_control_t00_stochastic_admission",
        wires,
        kernels,
        metadata,
    )
    validate_program(program)
    return program


def program_to_payload(program: StochasticProgram) -> JsonDict:
    """Turn the dataclass representation into strict JSON-compatible payload."""

    return {
        "schema_version": program.schema_version,
        "program_id": program.program_id,
        "wires": [
            {"id": wire.identifier, "kind": wire.kind, "categories": list(wire.categories)}
            for wire in program.wires
        ],
        "kernels": [
            {
                "id": kernel.identifier,
                "kind": kernel.kind,
                "inputs": list(kernel.inputs),
                "output": kernel.output,
                "params": json.loads(canonical_json(kernel.params)),
            }
            for kernel in program.kernels
        ],
        "metadata": json.loads(canonical_json(program.metadata)),
    }


def program_from_payload(payload: Mapping[str, Any]) -> StochasticProgram:
    """Parse and validate a JSON-compatible typed stochastic program."""

    required = {"schema_version", "program_id", "wires", "kernels", "metadata"}
    missing = sorted(required - set(payload))
    if missing:
        raise TypedStochasticIRValidationError(f"missing top-level fields: {missing}")
    if payload["schema_version"] != IR_SCHEMA_VERSION:
        raise TypedStochasticIRValidationError("unsupported schema_version")
    wires = tuple(
        Wire(
            identifier=str(item["id"]),
            kind=str(item["kind"]),
            categories=tuple(str(value) for value in item.get("categories", ())),
        )
        for item in _require_list(payload["wires"], "wires")
    )
    kernels = tuple(
        Kernel(
            identifier=str(item["id"]),
            kind=str(item["kind"]),
            inputs=tuple(str(value) for value in item.get("inputs", ())),
            output=str(item["output"]),
            params=dict(item.get("params") or {}),
        )
        for item in _require_list(payload["kernels"], "kernels")
    )
    program = StochasticProgram(
        schema_version=str(payload["schema_version"]),
        program_id=str(payload["program_id"]),
        wires=wires,
        kernels=kernels,
        metadata=dict(payload["metadata"]),
    )
    validate_program(program)
    return program


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypedStochasticIRValidationError(f"{context} must be a list")
    return value


def program_checksum(program: StochasticProgram) -> str:
    """Hash the stable program payload."""

    return sha256_json(program_to_payload(program))


def wire_order(program: StochasticProgram) -> list[str]:
    """Return the declared wire order used for state shape checks."""

    return [wire.identifier for wire in program.wires]


def validate_program(program: StochasticProgram) -> JsonDict:
    """Validate typed wires, local kernels, graph edges, probabilities, and seeds."""

    if program.schema_version != IR_SCHEMA_VERSION:
        raise TypedStochasticIRValidationError("unsupported schema_version")
    wires = _wire_map(program)
    _validate_wires(program.wires)
    _validate_kernels(program.kernels, wires)
    order = _topological_kernel_order(program.kernels)
    produced = {kernel.output for kernel in program.kernels}
    dangling_wires = sorted(set(wires) - produced)
    if dangling_wires:
        raise TypedStochasticIRValidationError(f"dangling wires without producer: {dangling_wires}")
    return {
        "ok": True,
        "schema_version": program.schema_version,
        "wire_count": len(program.wires),
        "kernel_count": len(program.kernels),
        "wire_order": wire_order(program),
        "wire_type_counts": dict(sorted(Counter(wire.kind for wire in program.wires).items())),
        "topological_kernel_order": order,
        "state_shape_contract": {
            "rank": 1,
            "wire_count": len(program.wires),
            "batch_axis": "optional_leading_axis",
        },
    }


def _wire_map(program: StochasticProgram) -> dict[str, Wire]:
    return {wire.identifier: wire for wire in program.wires}


def _validate_wires(wires: Sequence[Wire]) -> None:
    seen: set[str] = set()
    for wire in wires:
        if not wire.identifier:
            raise TypedStochasticIRValidationError("wire id must be non-empty")
        if wire.identifier in seen:
            raise TypedStochasticIRValidationError(f"duplicate wire: {wire.identifier}")
        seen.add(wire.identifier)
        if wire.kind == "binary":
            if wire.categories:
                raise TypedStochasticIRValidationError("binary wires may not declare categories")
        elif wire.kind == "categorical":
            if not wire.categories or len(set(wire.categories)) != len(wire.categories):
                raise TypedStochasticIRValidationError("categorical wires need unique categories")
        else:
            raise TypedStochasticIRValidationError(f"unsupported wire kind: {wire.kind}")


def _validate_kernels(kernels: Sequence[Kernel], wires: Mapping[str, Wire]) -> None:
    seen: set[str] = set()
    seed_roles: list[str] = []
    for kernel in kernels:
        if not kernel.identifier:
            raise TypedStochasticIRValidationError("kernel id must be non-empty")
        if kernel.identifier in seen:
            raise TypedStochasticIRValidationError(f"duplicate kernel: {kernel.identifier}")
        seen.add(kernel.identifier)
        if kernel.output not in wires:
            raise TypedStochasticIRValidationError(f"dangling wire output: {kernel.output}")
        for wire_id in kernel.inputs:
            if wire_id not in wires:
                raise TypedStochasticIRValidationError(f"dangling wire input: {wire_id}")
        if kernel.kind in {"categorical_prior", "bernoulli_prior"}:
            role = kernel.params.get("seed_role")
            if not isinstance(role, str) or not role:
                raise TypedStochasticIRValidationError("ambiguous seed role for stochastic kernel")
            seed_roles.append(role)
        _validate_kernel_schema(kernel, wires)
    if len(seed_roles) != len(set(seed_roles)):
        raise TypedStochasticIRValidationError("ambiguous seed role reuse")


def _validate_kernel_schema(kernel: Kernel, wires: Mapping[str, Wire]) -> None:
    output = wires[kernel.output]
    if kernel.kind == "categorical_prior":
        if kernel.inputs or output.kind != "categorical":
            raise TypedStochasticIRValidationError("type mismatch for categorical_prior")
        _validate_probabilities(kernel.params.get("probabilities"), _cardinality(output))
    elif kernel.kind == "bernoulli_prior":
        if kernel.inputs or output.kind != "binary":
            raise TypedStochasticIRValidationError("type mismatch for bernoulli_prior")
        p_true = kernel.params.get("p_true")
        if not isinstance(p_true, int | float) or not 0.0 <= float(p_true) <= 1.0:
            raise TypedStochasticIRValidationError("probability mass for bernoulli_prior")
    elif kernel.kind == "deterministic_lookup":
        if len(kernel.inputs) != 1 or wires[kernel.inputs[0]].kind != "categorical":
            raise TypedStochasticIRValidationError("type mismatch for deterministic_lookup")
        _validate_table(
            kernel.params.get("table"), _cardinality(wires[kernel.inputs[0]]), _cardinality(output)
        )
    elif kernel.kind == "deterministic_truth_table":
        if not kernel.inputs or output.kind != "binary":
            raise TypedStochasticIRValidationError("type mismatch for deterministic_truth_table")
        if any(wires[wire_id].kind != "binary" for wire_id in kernel.inputs):
            raise TypedStochasticIRValidationError("type mismatch for deterministic_truth_table")
        table_size = math.prod(_cardinality(wires[wire_id]) for wire_id in kernel.inputs)
        _validate_table(kernel.params.get("table"), table_size, 2)
    else:
        raise TypedStochasticIRValidationError(f"unsupported kernel kind: {kernel.kind}")


def _validate_probabilities(raw: Any, expected_len: int) -> None:
    if not isinstance(raw, list) or len(raw) != expected_len:
        raise TypedStochasticIRValidationError("probability mass length mismatch")
    values = [float(value) for value in raw]
    if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > EXACT_TOLERANCE:
        raise TypedStochasticIRValidationError("probability mass must be normalized")


def _validate_table(raw: Any, expected_len: int, output_cardinality: int) -> None:
    if not isinstance(raw, list) or len(raw) != expected_len:
        raise TypedStochasticIRValidationError("deterministic table length mismatch")
    if any(not isinstance(value, int) or value < 0 or value >= output_cardinality for value in raw):
        raise TypedStochasticIRValidationError("category index outside output domain")


def _cardinality(wire: Wire) -> int:
    return 2 if wire.kind == "binary" else len(wire.categories)


def _topological_kernel_order(kernels: Sequence[Kernel]) -> list[str]:
    producer: dict[str, str] = {}
    by_id: dict[str, Kernel] = {}
    for kernel in kernels:
        if kernel.output in producer:
            raise TypedStochasticIRValidationError(f"duplicate producer for wire: {kernel.output}")
        producer[kernel.output] = kernel.identifier
        by_id[kernel.identifier] = kernel
    deps = {
        kernel.identifier: {
            producer[input_id] for input_id in kernel.inputs if input_id in producer
        }
        for kernel in kernels
    }
    ready = deque(kernel.identifier for kernel in kernels if not deps[kernel.identifier])
    order: list[str] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for kernel_id in by_id:
            if current in deps[kernel_id]:
                deps[kernel_id].remove(current)
                if not deps[kernel_id]:
                    ready.append(kernel_id)
    if len(order) != len(kernels):
        raise TypedStochasticIRValidationError("unsupported cycle in stochastic graph")
    return order


def execute_exact(program: StochasticProgram) -> JsonDict:
    """Enumerate the full finite stochastic program exactly."""

    receipt = validate_program(program)
    wires = _wire_map(program)
    kernels = {kernel.identifier: kernel for kernel in program.kernels}
    assigned: list[str] = []
    probabilities: dict[tuple[int, ...], float] = {(): 1.0}
    for kernel_id in receipt["topological_kernel_order"]:
        kernel = kernels[kernel_id]
        next_probabilities: dict[tuple[int, ...], float] = {}
        for state_tuple, state_probability in probabilities.items():
            state = dict(zip(assigned, state_tuple, strict=True))
            for output_value, output_probability in _kernel_outputs(kernel, wires, state):
                next_state = (*state_tuple, output_value)
                next_probabilities[next_state] = (
                    next_probabilities.get(next_state, 0.0) + state_probability * output_probability
                )
        assigned.append(kernel.output)
        probabilities = next_probabilities

    support = []
    joint_probabilities: dict[str, float] = {}
    for state_tuple, probability in sorted(probabilities.items()):
        if probability > EXACT_TOLERANCE:
            labeled = _label_state(wires, assigned, state_tuple)
            key = canonical_json(labeled)
            joint_probabilities[key] = probability
            support.append({"state": labeled, "probability": probability})
    normalization = sum(joint_probabilities.values())
    marginals = _marginals(wires, assigned, support)
    state_space_size = math.prod(_cardinality(wires[wire_id]) for wire_id in assigned)
    return {
        "wire_order": assigned,
        "state_space_size": state_space_size,
        "support_count": len(support),
        "impossible_state_count": state_space_size - len(support),
        "normalization": normalization,
        "normalization_error": abs(1.0 - normalization),
        "support": support,
        "joint_probabilities": joint_probabilities,
        "marginals": marginals,
        "conditionals": _named_conditionals({"support": support}),
    }


def _kernel_outputs(
    kernel: Kernel,
    wires: Mapping[str, Wire],
    state: Mapping[str, int],
) -> list[tuple[int, float]]:
    if kernel.kind == "categorical_prior":
        return [(index, float(prob)) for index, prob in enumerate(kernel.params["probabilities"])]
    if kernel.kind == "bernoulli_prior":
        p_true = float(kernel.params["p_true"])
        return [(0, 1.0 - p_true), (1, p_true)]
    if kernel.kind == "deterministic_lookup":
        input_value = state[kernel.inputs[0]]
        return [(int(kernel.params["table"][input_value]), 1.0)]
    table_index = 0
    for input_id in kernel.inputs:
        table_index = table_index * _cardinality(wires[input_id]) + state[input_id]
    return [(int(kernel.params["table"][table_index]), 1.0)]


def _label_state(
    wires: Mapping[str, Wire],
    order: Sequence[str],
    values: Sequence[int],
) -> JsonDict:
    return {
        wire_id: _label_value(wires[wire_id], value)
        for wire_id, value in zip(order, values, strict=True)
    }


def _label_value(wire: Wire, value: int) -> str | int:
    return value if wire.kind == "binary" else wire.categories[value]


def _marginals(
    wires: Mapping[str, Wire],
    order: Sequence[str],
    support: Sequence[Mapping[str, Any]],
) -> JsonDict:
    result: JsonDict = {}
    for wire_id in order:
        labels = ["0", "1"] if wires[wire_id].kind == "binary" else list(wires[wire_id].categories)
        result[wire_id] = {label: 0.0 for label in labels}
    for row in support:
        state = row["state"]
        probability = float(row["probability"])
        for wire_id in order:
            result[wire_id][str(state[wire_id])] += probability
    return result


def independent_reference_distribution() -> JsonDict:
    """Compute the same workflow without using the IR or kernel executor."""

    item_probabilities = {"ac_e0_0": 0.4, "ac_e0_1": 0.5, "ac_e0_2": 0.1}
    clean_probabilities = {0: 1.0 - 0.9, 1: 0.9}
    group = {"ac_e0_0": "ac_g0_0", "ac_e0_1": "ac_g0_1", "ac_e0_2": "ac_g0_0"}
    clearance = {"ac_e0_0": "clearance_3", "ac_e0_1": "clearance_4", "ac_e0_2": "clearance_2"}
    blocked = {"ac_e0_0": 0, "ac_e0_1": 1, "ac_e0_2": 0}
    support = []
    for item, item_probability in item_probabilities.items():
        for clean, clean_probability in clean_probabilities.items():
            clearance_ok = int(clearance[item] in {"clearance_2", "clearance_3", "clearance_4"})
            eligible = int(
                group[item] in {"ac_g0_0", "ac_g0_1"} and clearance_ok and not blocked[item]
            )
            accepted = int(eligible and clean)
            support.append(
                {
                    "state": {
                        "candidate_item": item,
                        "strategy_clean": clean,
                        "member_group": group[item],
                        "clearance": clearance[item],
                        "gate_open": 1,
                        "blocked": blocked[item],
                        "clearance_ok": clearance_ok,
                        "eligible": eligible,
                        "accepted": accepted,
                    },
                    "probability": item_probability * clean_probability,
                }
            )
    joint = {canonical_json(row["state"]): float(row["probability"]) for row in support}
    wires = _wire_map(compile_exp6145_bounded_workflow())
    order = wire_order(compile_exp6145_bounded_workflow())
    return {
        "wire_order": order,
        "support": sorted(support, key=lambda row: canonical_json(row["state"])),
        "support_count": len(support),
        "joint_probabilities": dict(sorted(joint.items())),
        "normalization": sum(joint.values()),
        "marginals": _marginals(wires, order, support),
        "conditionals": _named_conditionals({"support": support}),
    }


def probability_of(distribution: Mapping[str, Any], filters: Mapping[str, Any]) -> float:
    """Return the exact probability of states matching all filter values."""

    return sum(
        float(row["probability"])
        for row in distribution["support"]
        if all(row["state"].get(wire_id) == expected for wire_id, expected in filters.items())
    )


def conditional_probability(
    distribution: Mapping[str, Any],
    event: Mapping[str, Any],
    given: Mapping[str, Any],
) -> float:
    """Return P(event | given) from an exact finite distribution."""

    denominator = probability_of(distribution, given)
    numerator = probability_of(distribution, {**given, **event})
    return numerator / denominator if denominator else 0.0


def _named_conditionals(distribution: Mapping[str, Any]) -> JsonDict:
    return {
        "p_accepted": probability_of(distribution, {"accepted": 1}),
        "p_eligible": probability_of(distribution, {"eligible": 1}),
        "p_accepted_given_eligible": conditional_probability(
            distribution, {"accepted": 1}, {"eligible": 1}
        ),
        "p_item2_given_accepted": conditional_probability(
            distribution, {"candidate_item": "ac_e0_2"}, {"accepted": 1}
        ),
    }


def compare_exact_semantics(program: StochasticProgram) -> JsonDict:
    """Compare Carnot execution to the independent exact finite reference."""

    exact = execute_exact(program)
    reference = independent_reference_distribution()
    joint_keys = set(exact["joint_probabilities"]) | set(reference["joint_probabilities"])
    marginal_deltas = [
        abs(
            float(exact["marginals"][wire_id].get(label, 0.0))
            - float(reference["marginals"][wire_id].get(label, 0.0))
        )
        for wire_id in exact["marginals"]
        for label in set(exact["marginals"][wire_id]) | set(reference["marginals"][wire_id])
    ]
    conditional_deltas = [
        abs(float(exact["conditionals"][name]) - float(reference["conditionals"][name]))
        for name in exact["conditionals"]
    ]
    joint_deltas = [
        abs(
            float(exact["joint_probabilities"].get(key, 0.0))
            - float(reference["joint_probabilities"].get(key, 0.0))
        )
        for key in joint_keys
    ]
    return {
        "principle": FIELD_PRINCIPLES[
            "support_conditional_joint_normalization_and_marginal_deltas"
        ],
        "support_match": set(exact["joint_probabilities"]) == set(reference["joint_probabilities"]),
        "support_delta_count": len(
            set(exact["joint_probabilities"]) ^ set(reference["joint_probabilities"])
        ),
        "max_joint_delta": max(joint_deltas) if joint_deltas else 0.0,
        "max_conditional_delta": max(conditional_deltas) if conditional_deltas else 0.0,
        "max_marginal_delta": max(marginal_deltas) if marginal_deltas else 0.0,
        "normalization_delta": abs(
            float(exact["normalization"]) - float(reference["normalization"])
        ),
        "tolerance": EXACT_TOLERANCE,
        "exact_conditionals": exact["conditionals"],
        "reference_conditionals": reference["conditionals"],
    }


def sample_batch(
    program: StochasticProgram,
    *,
    batch_size: int,
    seed: int | None,
) -> list[JsonDict]:
    """Draw replayable samples from local kernels for seed/batch controls only."""

    if seed is None:
        raise TypedStochasticIRValidationError("ambiguous seed")
    if batch_size <= 0:
        raise TypedStochasticIRValidationError("batch_size must be positive")
    validate_program(program)
    wires = _wire_map(program)
    kernels = {kernel.identifier: kernel for kernel in program.kernels}
    order = validate_program(program)["topological_kernel_order"]
    rng = random.Random(seed)
    rows: list[JsonDict] = []
    for _ in range(batch_size):
        state: dict[str, int] = {}
        for kernel_id in order:
            outputs = _kernel_outputs(kernels[kernel_id], wires, state)
            state[kernels[kernel_id].output] = _draw(outputs, rng)
        rows.append(
            _label_state(wires, wire_order(program), [state[name] for name in wire_order(program)])
        )
    return rows


def _draw(outputs: Sequence[tuple[int, float]], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for value, probability in outputs:
        cumulative += probability
        if threshold <= cumulative:
            return value
    return outputs[-1][0]


def batch_shape_contract(
    program: StochasticProgram, batch: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Check the simple state shape contract for sampled batches."""

    order = wire_order(program)
    return {
        "batch_size": len(batch),
        "wire_count": len(order),
        "ok": all(list(row) == order for row in batch),
    }


def run_negative_controls(program: StochasticProgram) -> JsonDict:
    """Run positive and negative controls that catch graph and indexing bugs."""

    payload = program_to_payload(program)
    wire_order_bug = deepcopy_json(payload)
    wire_order_bug["kernels"][7]["inputs"] = ["blocked", "gate_open", "clearance_ok"]
    wire_order_delta = compare_exact_semantics(program_from_payload(wire_order_bug))[
        "max_joint_delta"
    ]

    category_bug = deepcopy_json(payload)
    category_bug["kernels"][3]["params"]["table"] = [2, 3, 0]
    category_delta = compare_exact_semantics(program_from_payload(category_bug))["max_joint_delta"]

    controls = {
        "wire_order_bug_detected": wire_order_delta > EXACT_TOLERANCE,
        "wire_order_bug_max_joint_delta": wire_order_delta,
        "category_index_bug_detected": category_delta > EXACT_TOLERANCE,
        "category_index_bug_max_joint_delta": category_delta,
        "invalid_category_index_rejected": _rejects_with(
            payload, _invalid_category_payload, "category index"
        ),
        "type_mismatch_rejected": _rejects_with(payload, _type_mismatch_payload, "type mismatch"),
        "cycle_rejected": _rejects_with(payload, _cycle_payload, "cycle"),
        "dangling_wire_rejected": _rejects_with(payload, _dangling_payload, "dangling wire"),
        "invalid_mass_rejected": _rejects_with(payload, _invalid_mass_payload, "probability mass"),
        "ambiguous_seed_rejected": _rejects_with(
            payload, _ambiguous_seed_payload, "ambiguous seed"
        ),
        "principle": FIELD_PRINCIPLES[
            "wire_order_category_type_cycle_dangling_and_invalid_mass_negative_controls"
        ],
    }
    controls["all_negative_controls_passed"] = all(
        value is True
        for key, value in controls.items()
        if key.endswith("_rejected") or key.endswith("_detected")
    )
    return controls


def deepcopy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _rejects_with(payload: Mapping[str, Any], mutator: Any, expected: str) -> bool:
    mutated = mutator(deepcopy_json(payload))
    try:
        program_from_payload(mutated)
    except TypedStochasticIRValidationError as exc:
        return expected in str(exc)
    return False


def _invalid_category_payload(payload: JsonDict) -> JsonDict:
    payload["kernels"][2]["params"]["table"][0] = 99
    return payload


def _type_mismatch_payload(payload: JsonDict) -> JsonDict:
    payload["kernels"][2]["inputs"] = ["strategy_clean"]
    return payload


def _cycle_payload(payload: JsonDict) -> JsonDict:
    payload["kernels"][2]["inputs"] = ["member_group"]
    payload["kernels"][2]["params"]["table"] = [0, 1]
    return payload


def _dangling_payload(payload: JsonDict) -> JsonDict:
    payload["kernels"][2]["inputs"] = ["missing_wire"]
    return payload


def _invalid_mass_payload(payload: JsonDict) -> JsonDict:
    payload["kernels"][0]["params"]["probabilities"] = [0.4, 0.4, 0.1]
    return payload


def _ambiguous_seed_payload(payload: JsonDict) -> JsonDict:
    del payload["kernels"][0]["params"]["seed_role"]
    return payload


def deterministic_batch_seed_serialization_controls(program: StochasticProgram) -> JsonDict:
    """Collect deterministic-factor, impossible-state, seed, batch, and JSON controls."""

    exact = execute_exact(program)
    payload = program_to_payload(program)
    restored = program_from_payload(json.loads(canonical_json(payload)))
    batch = sample_batch(program, batch_size=16, seed=6152)
    replay = sample_batch(program, batch_size=16, seed=6152)
    return {
        "deterministic_factor_outputs": {
            "item0_eligible": probability_of(exact, {"candidate_item": "ac_e0_0", "eligible": 1}),
            "item1_eligible": probability_of(exact, {"candidate_item": "ac_e0_1", "eligible": 1}),
            "item2_eligible": probability_of(exact, {"candidate_item": "ac_e0_2", "eligible": 1}),
        },
        "impossible_state_probability": probability_of(
            exact, {"candidate_item": "ac_e0_0", "blocked": 1}
        ),
        "batch_shape_contract": batch_shape_contract(program, batch),
        "seed_replay_equal": batch == replay,
        "serialization_round_trip_checksum_equal": program_checksum(restored)
        == program_checksum(program),
        "serialization_round_trip_max_joint_delta": compare_exact_semantics(restored)[
            "max_joint_delta"
        ],
        "principle": FIELD_PRINCIPLES[
            "deterministic_impossible_batch_seed_and_serialization_controls"
        ],
    }


def torx_adapter_receipt(program: StochasticProgram) -> JsonDict:
    """Exercise the real pinned Torx PSC API when importable."""

    base = {
        "package_name": "extro-torx",
        "import_namespace": "torx",
        "pinned_repository": TORX_PINNED_REPOSITORY,
        "pinned_repository_commit": TORX_PINNED_REPOSITORY_COMMIT,
        "pinned_package_version": TORX_PINNED_PACKAGE_VERSION,
        "pinned_wheel_sha256": TORX_PINNED_WHEEL_SHA256,
        "pinned_sdist_sha256": TORX_PINNED_SDIST_SHA256,
        "program_checksum": program_checksum(program),
        "adapter_boundary": "psc_api_smoke_for_binary_and_categorical_wires_not_full_semantic_lowering",
        "principle": FIELD_PRINCIPLES["torx_package_version_commit_import_and_api_receipts"],
    }
    try:
        import jax
        import jax.numpy as jnp
        from torx import psc
    except Exception as exc:  # pragma: no cover - depends on optional Torx install.
        return {
            **base,
            "importable": False,
            "installed_version": None,
            "api_exercised": False,
            "compatibility_ready": False,
            "blocked_reason": f"{type(exc).__name__}: {exc}",
            "exercised_api": [],
        }

    version = importlib.metadata.version("extro-torx")
    circuit = psc.DiscretePCircuit([psc.PNOT(0), psc.PditShift(1, dims=3)])
    thetas = [jnp.array([math.log(0.2 / 0.8)]), jnp.array([0.0])]
    simulator = psc.StateVectorSimulator()
    compiled = simulator.build_circuit(circuit, thetas)
    state = jnp.zeros((6,), dtype=jnp.float32).at[0].set(1.0)
    density = simulator.density(compiled, state)
    expected = [0.4, 0.4, 0.0, 0.1, 0.1, 0.0]
    observed = [float(value) for value in density]
    max_delta = max(abs(a - b) for a, b in zip(observed, expected, strict=True))
    sum_delta = abs(float(jnp.sum(density)) - 1.0)
    return {
        **base,
        "importable": True,
        "installed_version": version,
        "torx_version_attr": getattr(__import__("torx"), "__version__", None),
        "api_exercised": True,
        "compatibility_ready": version == TORX_PINNED_PACKAGE_VERSION and max_delta <= 1.0e-7,
        "blocked_reason": None,
        "exercised_api": ["DiscretePCircuit", "StateVectorSimulator", "PNOT", "PditShift"],
        "psc_dims": list(circuit.dims),
        "psc_density": observed,
        "psc_expected_density": expected,
        "psc_density_max_delta": max_delta,
        "psc_density_sum_delta": sum_delta,
        "jax_version": jax.__version__,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def structured_gate_receipt(root: Path = REPO_ROOT) -> JsonDict:
    """Replay the Exp6145 gate that makes this bounded compiler input valid."""

    artifact_path = root / exp6145.RESULT_RELATIVE_PATH
    row_path = root / exp6145.ROW_FILE_RELATIVE_PATH
    split_path = root / exp6145.SPLIT_FILE_RELATIVE_PATH
    outcome_path = root / exp6145.OUTCOME_FILE_RELATIVE_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    replay = exp6145.replay_sidecars(row_path, split_path, outcome_path)
    return {
        "exp6145_artifact": exp6145.RESULT_RELATIVE_PATH.as_posix(),
        "exp6145_ready_score": artifact.get("constraint_shift_stream_ready_score"),
        "exp6145_status": artifact.get("status"),
        "exp6145_honest_verdict": artifact.get("honest_verdict"),
        "sidecar_replay_ok": replay["ok"],
        "row_sha256": replay["row_sha256"],
        "split_sha256": replay["split_sha256"],
        "outcome_sha256": replay["outcome_sha256"],
        "gate_passed": artifact.get("constraint_shift_stream_ready_score") == 1.0
        and replay["ok"] is True,
        "principle": FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def source_workflow_validator_sampler_and_exclusion_hashes(root: Path = REPO_ROOT) -> JsonDict:
    """Hash workflow, validators, sampler APIs, exclusions, Torx metadata, and outputs."""

    paths = _path_hashes(HASHED_SOURCE_PATHS, root)
    return {
        "paths": paths,
        "source_workflow": exp6145.RESULT_RELATIVE_PATH.as_posix(),
        "exact_validators": [
            "python/carnot/experiment_5896_typed_constraint_ir_fixture.py",
            "python/carnot/experiment_6145_constraint_shift_stream.py",
        ],
        "sampler_api_paths": [
            "python/carnot/samplers/backend.py",
            "python/carnot/samplers",
            "crates/carnot-ising",
            "crates/carnot-gibbs",
        ],
        "torx_reference": {
            "package": "extro-torx",
            "version": TORX_PINNED_PACKAGE_VERSION,
            "repository": TORX_PINNED_REPOSITORY,
            "commit": TORX_PINNED_REPOSITORY_COMMIT,
            "wheel_sha256": TORX_PINNED_WHEEL_SHA256,
            "sdist_sha256": TORX_PINNED_SDIST_SHA256,
        },
        "output_paths": {"result": RESULT_RELATIVE_PATH.as_posix()},
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "principle": FIELD_PRINCIPLES["source_workflow_validator_sampler_and_exclusion_hashes"],
    }


def preconditions_checked(
    *,
    output_path: Path,
    source_hashes: Mapping[str, Any],
    torx_receipt: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> JsonDict:
    """Build the precondition receipt used by the terminal artifact."""

    checks = {
        "jax_platforms_cpu": os.environ.get("JAX_PLATFORMS") == "cpu",
        "exp6145_structured_gate_passed": gate.get("gate_passed") is True,
        "exp6145_workflow_present": (REPO_ROOT / exp6145.RESULT_RELATIVE_PATH).exists(),
        "validator_callable": callable(exp5896.certify_ir),
        "sampler_spec_has_req": "REQ-SAMPLE-6152"
        in (REPO_ROOT / SAMPLER_SPEC_RELATIVE_PATH).read_text(encoding="utf-8"),
        "torx_importable": torx_receipt.get("importable") is True,
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
        "source_hash_receipt_sha256": sha256_json(source_hashes),
        "torx_receipt_sha256": sha256_json(torx_receipt),
        "output_path": output_path.as_posix(),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def ir_schema_version_types_kernels_and_graph_contract(program: StochasticProgram) -> JsonDict:
    """Describe the local IR schema and graph contract."""

    validation = validate_program(program)
    return {
        "schema_version": IR_SCHEMA_VERSION,
        "wire_types": {
            wire.identifier: {
                "kind": wire.kind,
                "categories": list(wire.categories) if wire.kind == "categorical" else [0, 1],
            }
            for wire in program.wires
        },
        "kernel_kinds": sorted({kernel.kind for kernel in program.kernels}),
        "kernels": [
            {
                "id": kernel.identifier,
                "kind": kernel.kind,
                "inputs": list(kernel.inputs),
                "output": kernel.output,
            }
            for kernel in program.kernels
        ],
        "graph_validation": validation,
        "normalization_tolerance": EXACT_TOLERANCE,
        "dependency_policy": "torx_optional_carnot_exact_executor_required",
        "principle": FIELD_PRINCIPLES["ir_schema_version_types_kernels_and_graph_contract"],
    }


def compiler_executor_adapter_and_test_paths() -> JsonDict:
    """Record implementation paths for the compiler, executor, adapter, and tests."""

    return {
        "module": MODULE_RELATIVE_PATH.as_posix(),
        "tests": TEST_RELATIVE_PATH.as_posix(),
        "spec": SAMPLER_SPEC_RELATIVE_PATH.as_posix(),
        "compiler": "compile_exp6145_bounded_workflow",
        "exact_executor": "execute_exact",
        "independent_reference": "independent_reference_distribution",
        "torx_adapter": "torx_adapter_receipt",
        "artifact": RESULT_RELATIVE_PATH.as_posix(),
        "principle": FIELD_PRINCIPLES["compiler_executor_adapter_and_test_paths"],
    }


def exact_enumeration_case_counts(program: StochasticProgram) -> JsonDict:
    """Summarize bounded exhaustive enumeration counts."""

    exact = execute_exact(program)
    return {
        "wire_count": len(program.wires),
        "kernel_count": len(program.kernels),
        "state_space_size": exact["state_space_size"],
        "support_count": exact["support_count"],
        "impossible_state_count": exact["impossible_state_count"],
        "conditional_count": len(exact["conditionals"]),
        "marginal_wire_count": len(exact["marginals"]),
        "principle": FIELD_PRINCIPLES["exact_enumeration_case_counts"],
    }


def deterministic_rebuild_receipt() -> JsonDict:
    """Build the program twice and hash exact semantics for deterministic replay."""

    first = compile_exp6145_bounded_workflow()
    second = compile_exp6145_bounded_workflow()
    first_checksum = sha256_json(
        {"program": program_to_payload(first), "exact": execute_exact(first)}
    )
    second_checksum = sha256_json(
        {"program": program_to_payload(second), "exact": execute_exact(second)}
    )
    return {
        "checksum": first_checksum,
        "second_checksum": second_checksum,
        "matches": first_checksum == second_checksum,
    }


def write_typed_stochastic_ir_artifact(
    *,
    output_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Write the terminal Exp6152 artifact."""

    started = time.monotonic()
    output = output_path or REPO_ROOT / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    protected_before = _path_hashes(PROTECTED_FILES)
    program = compile_exp6145_bounded_workflow()
    torx_receipt = torx_adapter_receipt(program)
    gate = structured_gate_receipt()
    source_hashes = source_workflow_validator_sampler_and_exclusion_hashes()
    preconditions = preconditions_checked(
        output_path=output,
        source_hashes=source_hashes,
        torx_receipt=torx_receipt,
        gate=gate,
    )
    protected = _unchanged_receipt(PROTECTED_FILES, protected_before)
    elapsed = float(duration_s if duration_s is not None else time.monotonic() - started)
    artifact = build_artifact(
        program=program,
        preconditions=preconditions,
        gate=gate,
        source_hashes=source_hashes,
        torx_receipt=torx_receipt,
        protected=protected,
        duration_s=elapsed,
        test_exit_codes=dict(test_exit_codes or {}),
    )
    validate_artifact(artifact)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def build_artifact(
    *,
    program: StochasticProgram,
    preconditions: Mapping[str, Any],
    gate: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    torx_receipt: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Assemble the terminal artifact before writing it."""

    comparison = compare_exact_semantics(program)
    controls = deterministic_batch_seed_serialization_controls(program)
    negative_controls = run_negative_controls(program)
    rebuild = deterministic_rebuild_receipt()
    torx_scope = {
        "package": torx_receipt["package_name"],
        "installed_version": torx_receipt.get("installed_version"),
        "pinned_repository_commit": torx_receipt["pinned_repository_commit"],
        "adapter_boundary": torx_receipt["adapter_boundary"],
        "exercised_api": list(torx_receipt.get("exercised_api") or []),
        "compatibility_ready": torx_receipt.get("compatibility_ready") is True,
        "principle": FIELD_PRINCIPLES["torx_compatibility_scope"],
    }
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": FIELD_PRINCIPLES,
        "status": "complete_null",
        "preconditions_checked": dict(preconditions),
        "structured_gate_receipt": dict(gate),
        "source_workflow_validator_sampler_and_exclusion_hashes": dict(source_hashes),
        "torx_package_version_commit_import_and_api_receipts": dict(torx_receipt),
        "ir_schema_version_types_kernels_and_graph_contract": ir_schema_version_types_kernels_and_graph_contract(
            program
        ),
        "compiler_executor_adapter_and_test_paths": compiler_executor_adapter_and_test_paths(),
        "exact_enumeration_case_counts": exact_enumeration_case_counts(program),
        "support_conditional_joint_normalization_and_marginal_deltas": comparison,
        "deterministic_impossible_batch_seed_and_serialization_controls": controls,
        "wire_order_category_type_cycle_dangling_and_invalid_mass_negative_controls": negative_controls,
        "torx_compatibility_scope": torx_scope,
        "deterministic_rebuild_checksum": rebuild["checksum"],
        "typed_stochastic_ir_ready_score": 0.0,
        "protected_files_unchanged": dict(protected),
        "duration_s": round(duration_s, 6),
        "inference_substrate": JAX_CPU_SUBSTRATE
        if torx_scope["compatibility_ready"]
        else CARNOT_ONLY_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": []
        if torx_scope["compatibility_ready"]
        else ["torx_compatibility"],
        "field_provenance": field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["typed_stochastic_ir_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _unchanged_receipt(paths: Sequence[Path], before: Mapping[str, Any]) -> JsonDict:
    after = _path_hashes(paths)
    return {
        "before": dict(before),
        "after": after,
        "unchanged": dict(before) == after,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the strict readiness scalar."""

    deltas = dict(artifact.get("support_conditional_joint_normalization_and_marginal_deltas") or {})
    controls = dict(
        artifact.get("deterministic_impossible_batch_seed_and_serialization_controls") or {}
    )
    negatives = dict(
        artifact.get("wire_order_category_type_cycle_dangling_and_invalid_mass_negative_controls")
        or {}
    )
    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    missing_commands = [
        command for command in DEFAULT_TEST_COMMANDS if command not in test_exit_codes
    ]
    nonzero_commands = [
        command for command in DEFAULT_TEST_COMMANDS if test_exit_codes.get(command) != 0
    ]
    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and dict(artifact.get("structured_gate_receipt") or {}).get("gate_passed") is True
        and deltas.get("support_match") is True
        and float(deltas.get("max_joint_delta", 1.0)) <= EXACT_TOLERANCE
        and float(deltas.get("max_conditional_delta", 1.0)) <= EXACT_TOLERANCE
        and float(deltas.get("max_marginal_delta", 1.0)) <= EXACT_TOLERANCE
        and float(deltas.get("normalization_delta", 1.0)) <= EXACT_TOLERANCE
        and controls.get("seed_replay_equal") is True
        and controls.get("serialization_round_trip_checksum_equal") is True
        and controls.get("impossible_state_probability") == 0.0
        and negatives.get("all_negative_controls_passed") is True
        and dict(artifact.get("torx_compatibility_scope") or {}).get("compatibility_ready") is True
        and artifact.get("deterministic_rebuild_checksum")
        == deterministic_rebuild_receipt()["checksum"]
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("inference_substrate") == JAX_CPU_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and not missing_commands
        and not nonzero_commands
    )
    return 1.0 if ready else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Return terminal artifact status from the evidence."""

    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    if ready_score(artifact) == 1.0:
        return "complete_ready"
    if dict(artifact.get("torx_compatibility_scope") or {}).get("compatibility_ready") is not True:
        return "complete_carnot_only"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the required terminal-prefixed verdict."""

    current = status(artifact)
    if current == "complete_ready":
        return "complete_ready: exact_carnot_semantics_and_pinned_torx_api_smoke"
    if current == "complete_carnot_only":
        return "complete_carnot_only: exact_semantics_passed_torx_compatibility_blocked"
    if current == "blocked":
        return "blocked: " + ",".join(blocked_reasons(artifact)[:8])
    return "complete_null: " + ",".join(blocked_reasons(artifact)[:8])


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return compact blocker names for the honest verdict."""

    reasons: list[str] = []
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        reasons.append("preconditions")
    if dict(artifact.get("structured_gate_receipt") or {}).get("gate_passed") is not True:
        reasons.append("structured_gate")
    if dict(artifact.get("torx_compatibility_scope") or {}).get("compatibility_ready") is not True:
        reasons.append("torx_compatibility")
    if dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is not True:
        reasons.append("protected_files")
    missing = [
        command
        for command in DEFAULT_TEST_COMMANDS
        if command not in dict(artifact.get("test_exit_codes") or {})
    ]
    if missing:
        reasons.append("missing_test_commands")
    return reasons or ["ready_score"]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def field_provenance() -> JsonDict:
    """Map every required field to the evidence classes that produced it."""

    sources = [
        "task_prompt",
        SAMPLER_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6145.RESULT_RELATIVE_PATH.as_posix(),
        "python/carnot/experiment_6145_constraint_shift_stream.py",
        "python/carnot/experiment_5896_typed_constraint_ir_fixture.py",
        "pypi:extro-torx==0.0.1",
        TORX_PINNED_REPOSITORY + "@" + TORX_PINNED_REPOSITORY_COMMIT,
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": list(sources)}
        for field in FIELD_PRINCIPLES
    }


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal artifact schema and readiness consistency."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if artifact.get("typed_stochastic_ir_ready_score") != ready_score(artifact):
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
