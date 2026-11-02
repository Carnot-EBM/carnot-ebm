"""Measure native lossless packed-belief parity and boundary cost.

The experiment ports only the decision-bearing survivor and vote core. Python
keeps the study protocol, provenance checks, and artifact validation.

Spec refs: REQ-CL-7230, SCENARIO-CL-7230-*, REQ-RUSTPY-7230, and
SCENARIO-RUSTPY-7230-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
from types import ModuleType
from typing import Any

import numpy as np
import yaml

from carnot import experiment_7217_v635_abi_board_readiness as exp7217
from carnot import experiment_7226_v636_belief_compiler as exp7226


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7230
TASK_ID = "exp7230-native-belief"
SCHEMA = "carnot.exp7230.v636_native_belief.v1"
MILESTONE = "2026.09.636"
RUN_DATE = "20260912"
RANDOM_SEED = 7_230_000
SEQUENCE_SEEDS = tuple(range(7_230_001, 7_230_021))
BATCH_SIZES = (1, 32, 256)
REPETITIONS = 10
MEASUREMENT_TIMEOUT_S = 300.0
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
FAMILY_NAMES = tuple(exp7226.FAMILIES)
FAMILY_CODES = {name: index for index, name in enumerate(FAMILY_NAMES)}
DOMAIN_SIZE = len(exp7226.PARAMETER_DOMAIN)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE = Path("results/experiment_7230_v636_native_belief.json")
CHECKPOINT_RELATIVE = Path("results/checkpoints/experiment_7230_v636_native_belief.json")
UPSTREAM_RELATIVE = Path("results/experiment_7226_v636_belief_compiler.json")
UPSTREAM_PATH = REPO_ROOT / UPSTREAM_RELATIVE
ABI_RELATIVE = Path("results/experiment_7217_v635_abi_board_readiness.json")
ROADMAP_RELATIVE = Path("research-roadmap.yaml")
EXCLUSION_RELATIVE = Path("ops/exclusion_manifest.yaml")
CL_SPEC_RELATIVE = Path("openspec/capabilities/continuous-learning/spec.md")
RUST_SPEC_RELATIVE = Path("openspec/capabilities/rust-python-boundary/spec.md")
TARGET_RELATIVE = Path("target/experiment-7230-interpreter-bound")
LOAD_RELATIVE = Path("target/experiment-7230-load")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ROADMAP_RELATIVE,
    EXCLUSION_RELATIVE,
    Path("ops/e2e-test-plan.md"),
    Path("crates/carnot-python/Cargo.toml"),
    Path("crates/carnot-python/src/lib.rs"),
    Path("crates/carnot-python/src/packed_belief.rs"),
    Path("python/carnot/experiment_7201_v634_slice_pyo3.py"),
    Path("python/carnot/experiment_7217_v635_abi_board_readiness.py"),
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/experiment_7230_v636_native_belief.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    ABI_RELATIVE,
    UPSTREAM_RELATIVE,
    CL_SPEC_RELATIVE,
    RUST_SPEC_RELATIVE,
    Path("scripts/experiments/experiment_7230_v636_native_belief.py"),
    Path("tests/python/test_experiment_7230_v636_native_belief.py"),
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed ID binds this evidence to one task.",
    "milestone": "The milestone binds the result to the V636 execution contract.",
    "field_principles": (
        "Annotate actual values in this map; do not wrap arbitrary dictionaries as "
        "principle/value records."
    ),
    "status": (
        "Write a terminal artifact only when done or externally blocked; running checkpoints "
        "use a different path."
    ),
    "run_date": "Use 20260912 and record actual UTC timestamps, never copy an upstream run date.",
    "started_at_utc": "Record the actual UTC start separately from the fixed run date.",
    "completed_at_utc": "Record the actual UTC completion separately from the fixed run date.",
    "preconditions_checked": "Actual code, resource, identity and gate observations before expensive work.",
    "inference_substrate": (
        "Use the recognized literal for the work actually executed; custom free text caused the "
        "Exp7208 quarantine."
    ),
    "inference_substrate_class": (
        "Match actual generation, load-only, CPU or aggregation work and its duration floor."
    ),
    "execution_venue": (
        "Exactly host, kv260, gatemate or polarfire; the top-level orchestration here is host."
    ),
    "execution_host": "Actual hostname separate from venue.",
    "duration_s": "Measured monotonic work time; no padding or reclassification to evade a floor.",
    "source_artifact_hashes": "Bind code, source documents, manifests and raw evidence to claims.",
    "rows": (
        "Per unit/arm/seed metric, error and abstention for every comparison; retain full "
        "denominators."
    ),
    "sample_size_budget": (
        "Planned, attempted, completed, censored and independent units; no silent removal."
    ),
    "random_seed": "Freeze random choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash exact source, inputs, settings and raw unit rows.",
    "gate_check_summary": (
        "Every blocked_* verdict names failed check, upstream, field, expected and observed value."
    ),
    "verifier_is_oracle": (
        "True when correctness authority is reused as the verifier; independent code alone is "
        "not distinct authority."
    ),
    "verdict_class": (
        "Closed enum positive | circular_positive | null | blocked | disqualified | partial. "
        "partial means unfinished own work only."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings; blocked_* for external absence. "
        "A failed acceptance gate forbids positive."
    ),
    "MODEL_SPECS": "Only models actually invoked; [] for CPU/aggregation, mandated Qwen3.8 for every model task.",
    "model_invoked": "True only for actual model execution; upstream model outputs are cached evidence.",
    "native_belief_ready_score": "Actual compiled execution, exact parity and state restoration.",
    "native_cost_value_score": "Matched end-to-end cost improvement with exact semantics.",
    "native_identity_receipt": "Executing interpreter, build flags, shared library hash and native entrypoint.",
    "parity_rows": "Native/reference decisions, energies and state roundtrips.",
    "cost_rows": "Per trace, batch, repetition and arm; full boundary costs.",
    "cost_summary": "Paired speed ratios and fixed confidence bounds derive the cost gates.",
    "nfr_01_10x_met": "Derived from measured lower confidence bound, never presumed.",
    "scientific_value_inherited": "No new learning or semantic value follows from speed alone.",
    "checkpoint_path": "Provisional parity evidence stays under results/checkpoints.",
    "sampler_nfr_history": "The failed sampler 10x claim remains closed and separate.",
    "spec_refs": "Exact driving requirements connect the artifact to tests and code.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

_RESTORE_HELPER = r"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

extension = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("carnot._rust", extension)
if spec is None or spec.loader is None:
    raise RuntimeError("native loader unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
controller = module.RustPackedBeliefController()
checkpoint = sys.stdin.read()
controller.load_state(checkpoint)
result = dict(controller.query_batch(
    np.ascontiguousarray([0, 1, 2, 3], dtype=np.uint8),
    np.ascontiguousarray([0, 8, 16, 24], dtype=np.int64),
    np.ascontiguousarray([1, 0, 1, 0], dtype=np.int8),
))
print("__CARNOT_JSON__" + json.dumps({
    "module_file": str(Path(module.__file__).resolve()),
    "serialized_state": controller.serialize_state(),
    "decisions": result["decisions"],
}, allow_nan=False, sort_keys=True), flush=True)
"""


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep provisional parity bytes separate from the terminal result."""

    checkpoint: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return repository destinations for the public command."""

        return cls(REPO_ROOT / CHECKPOINT_RELATIVE, REPO_ROOT / RESULT_RELATIVE)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put all test outputs below one caller-owned temporary directory."""

        return cls(root / "checkpoints/exp7230.json", root / "experiment_7230.json")


def progress(phase: int, boundary: str, operation: str) -> None:
    """Flush one numbered boundary so the conductor sees real state."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def canonical_json(value: Any) -> str:
    """Encode deterministic finite JSON for state and evidence identity."""

    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def sha256_file(path: Path) -> str:
    """Hash actual file bytes without loading a large artifact at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum itself."""

    material = dict(artifact)
    material.pop("reproducibility_checksum", None)
    return "sha256:" + hashlib.sha256(canonical_json(material).encode()).hexdigest()


def _finish_row(row: JsonDict) -> JsonDict:
    """Add the shared full-denominator row fields and stable identity."""

    row.pop("row_sha256", None)
    row.setdefault("arm", "not_applicable")
    row.setdefault("seed", None)
    row.setdefault("metric", 0.0)
    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row["row_sha256"] = "sha256:" + hashlib.sha256(canonical_json(row).encode()).hexdigest()
    return row


def unwrap_principled_value(value: Any) -> Any:
    """Unwrap only the exact principle/value record allowed by the contract."""

    return exp7217.unwrap_principled_value(value)


def check(
    name: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Return one uniform precondition observation."""

    return {
        "check": name,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every check and expose the first exact failed gate."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "checks": [dict(row) for row in checks],
        "failed_check": None if failed is None else failed.get("check"),
        "upstream": None if failed is None else failed.get("upstream"),
        "field": None if failed is None else failed.get("field"),
        "expected_value": None if failed is None else failed.get("expected_value"),
        "observed_value": None if failed is None else failed.get("observed_value"),
    }


def _read_object(path: Path) -> JsonDict:
    """Read one JSON object while malformed evidence remains a failed gate."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _writable(path: Path) -> bool:
    """Check the nearest existing parent without creating result bytes."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_contract(root: Path) -> JsonDict | None:
    """Parse the real roadmap and return only fields that authorize Exp7230."""

    try:
        document = yaml.safe_load((root / ROADMAP_RELATIVE).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    tasks = document.get("tasks") if isinstance(document, Mapping) else None
    if not isinstance(tasks, list):
        return None
    task = next(
        (row for row in tasks if isinstance(row, Mapping) and row.get("id") == TASK_ID), None
    )
    if not isinstance(task, Mapping):
        return None
    gate = task.get("gated_on")
    return {
        "id": task.get("id"),
        "milestone": task.get("milestone"),
        "deliverable": task.get("deliverable"),
        "gated_on": gate,
    }


def collect_preconditions(
    root: Path,
    paths: ExperimentPaths,
    *,
    upstream_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Authenticate sources, the ready producer, identity recipe, and outputs."""

    progress(0, "start", "check source bytes, requirements, imports, gates, and outputs")
    upstream_file = upstream_path or root / UPSTREAM_RELATIVE
    upstream = _read_object(upstream_file)
    abi = _read_object(root / ABI_RELATIVE)
    checks: list[JsonDict] = []
    sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    checks.append(
        check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            "all nonempty",
            sizes,
            all(size is not None and size > 0 for size in sizes.values()),
        )
    )
    cl_text = (root / CL_SPEC_RELATIVE).read_text(encoding="utf-8")
    rust_text = (root / RUST_SPEC_RELATIVE).read_text(encoding="utf-8")
    spec_observed = {
        "REQ-CL-7230": "REQ-CL-7230" in cl_text,
        "SCENARIO-CL-7230": "SCENARIO-CL-7230-" in cl_text,
        "REQ-RUSTPY-7230": "REQ-RUSTPY-7230" in rust_text,
        "SCENARIO-RUSTPY-7230": "SCENARIO-RUSTPY-7230-" in rust_text,
    }
    checks.append(
        check(
            "driving_capability_specs",
            f"{CL_SPEC_RELATIVE},{RUST_SPEC_RELATIVE}",
            "REQ-* and SCENARIO-*",
            {name: True for name in spec_observed},
            spec_observed,
            all(spec_observed.values()),
        )
    )
    expected_contract = {
        "id": TASK_ID,
        "milestone": MILESTONE,
        "deliverable": str(RESULT_RELATIVE),
        "gated_on": [
            {
                "upstream": "exp7226-belief-compiler",
                "artifact_field": "belief_compiler_ready_score",
                "op": "==",
                "value": 1,
            }
        ],
    }
    observed_contract = _task_contract(root)
    checks.append(
        check(
            "roadmap_task_contract",
            str(ROADMAP_RELATIVE),
            TASK_ID,
            expected_contract,
            observed_contract,
            observed_contract == expected_contract,
        )
    )
    tools = {
        "python": str(Path(sys.executable).absolute()),
        "cargo": shutil.which("cargo"),
        "rustc": shutil.which("rustc"),
        "ldd": shutil.which("ldd"),
        "numpy": np.__version__,
    }
    outputs = {"artifact": _writable(paths.artifact), "checkpoint": _writable(paths.checkpoint)}
    checks.append(
        check(
            "imports_tools_and_outputs",
            "host",
            "python,cargo,rustc,ldd,numpy,raw/checkpoint outputs",
            "all available and writable",
            {**tools, **outputs},
            all(tools[name] for name in ("cargo", "rustc", "ldd")) and all(outputs.values()),
        )
    )
    exclusion = None
    try:
        exclusion = yaml.safe_load((root / EXCLUSION_RELATIVE).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        pass
    manifest_match = exp7217._manifest_mentions_experiment(exclusion, "7226")
    quarantine = exp7217.upstream_quarantine_observation(upstream, manifest_match=manifest_match)
    checks.append(
        check(
            "exp7226_not_quarantined",
            str(upstream_file),
            "quarantined",
            False,
            quarantine,
            quarantine["quarantined"] is False,
        )
    )
    upstream_errors = (
        exp7226.validate_artifact(upstream)
        if upstream and quarantine["quarantined"] is False
        else ["not_authenticated"]
    )
    checks.append(
        check(
            "exp7226_shipped_validator",
            str(upstream_file),
            "validator_errors",
            [],
            upstream_errors,
            upstream_errors == [],
        )
    )
    gate = exp7217.gated_upstream_value(upstream, quarantine, "belief_compiler_ready_score")
    checks.append(
        check(
            "exp7226_ready_gate",
            str(upstream_file),
            "belief_compiler_ready_score",
            1,
            gate,
            upstream_errors == [] and gate == 1,
        )
    )
    raw_receipts = [
        *upstream.get("stream_manifest_path", {}).values(),
        upstream.get("compiler_state_path", {}),
    ]
    raw_observed = []
    for receipt in raw_receipts:
        path = Path(str(receipt.get("path", ""))) if isinstance(receipt, Mapping) else Path()
        actual = sha256_file(path) if path.is_file() else None
        raw_observed.append(
            {"path": str(path), "expected": receipt.get("sha256"), "actual": actual}
            if isinstance(receipt, Mapping)
            else {"path": "", "expected": None, "actual": None}
        )
    checks.append(
        check(
            "exp7226_raw_evidence_hashes",
            str(upstream_file),
            "stream_manifest_path and compiler_state_path",
            "all exact hashes",
            raw_observed,
            len(raw_observed) == 5
            and all(
                row["expected"] is not None and row["expected"] == row["actual"]
                for row in raw_observed
            ),
        )
    )
    abi_quarantine = exp7217.upstream_quarantine_observation(
        abi,
        manifest_match=exp7217._manifest_mentions_experiment(exclusion, "7217"),
    )
    abi_errors = exp7217.validate_artifact(abi) if abi and not abi_quarantine["quarantined"] else []
    abi_observed = {
        "quarantined": abi_quarantine["quarantined"],
        "validator_errors": abi_errors,
        "native_abi_ready_score": exp7217.gated_upstream_value(
            abi, abi_quarantine, "native_abi_ready_score"
        ),
    }
    checks.append(
        check(
            "exp7217_interpreter_recipe",
            str(ABI_RELATIVE),
            "authenticated native ABI receipt",
            {"quarantined": False, "validator_errors": [], "native_abi_ready_score": 1},
            abi_observed,
            abi_observed
            == {"quarantined": False, "validator_errors": [], "native_abi_ready_score": 1},
        )
    )
    hashes = {
        str(path): sha256_file(root / path) for path in SOURCE_PATHS if (root / path).is_file()
    }
    upstream_key = (
        str(UPSTREAM_RELATIVE) if upstream_file == root / UPSTREAM_RELATIVE else str(upstream_file)
    )
    if upstream_file.is_file():
        hashes[upstream_key] = sha256_file(upstream_file)
    progress(
        0,
        "end",
        f"precondition checks={len(checks)} failed={sum(not row['passed'] for row in checks)}",
    )
    return checks, hashes


def build_native_extension(root: Path, *, show_progress: bool = True) -> tuple[Path, JsonDict]:
    """Build only the PyO3 crate for this interpreter in an isolated target."""

    target = root / TARGET_RELATIVE
    environment = exp7217.interpreter_build_environment(Path(sys.executable), target)
    if show_progress:
        progress(3, "before", "interpreter-bound carnot-python subprocess")
    receipt = exp7217._stream_process(
        ["cargo", "build", "--release", "-p", "carnot-python"],
        root=root,
        environment=environment,
        operation="Exp7230 interpreter-bound carnot-python build",
        timeout_s=900,
    )
    library = target / "release/libcarnot_python.so"
    if not library.is_file():
        raise RuntimeError(f"native build output missing: {library}")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python extension suffix unavailable")
    load_dir = root / LOAD_RELATIVE
    load_dir.mkdir(parents=True, exist_ok=True)
    destination = load_dir / f"_rust{suffix}"
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=load_dir)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(library, temporary)
        os.replace(temporary, destination)
        destination.chmod(0o755)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt.update(
        {
            "PYO3_PYTHON": environment["PYO3_PYTHON"],
            "CARGO_TARGET_DIR": environment["CARGO_TARGET_DIR"],
            "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": None,
            "built_library": str(library.resolve()),
            "loaded_copy": str(destination.resolve()),
            "binary_sha256": sha256_file(destination),
        }
    )
    if show_progress:
        progress(3, "after", f"native build selected={destination.resolve()}")
    return destination, receipt


def load_native_extension(extension: Path) -> ModuleType:
    """Load the exact copied binary and reject a missing packed entrypoint."""

    resolved = extension.resolve()
    existing = sys.modules.get("carnot._rust")
    if existing is not None and Path(str(existing.__file__)).resolve() == resolved:
        module = existing
    else:
        sys.modules.pop("carnot._rust", None)
        spec = importlib.util.spec_from_file_location("carnot._rust", resolved)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot create native loader for {resolved}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    if not hasattr(module, "RustPackedBeliefController"):
        raise RuntimeError("selected extension lacks RustPackedBeliefController")
    return module


def reference_semantic_state(controller: exp7226.PackedBeliefController) -> JsonDict:
    """Project Exp7226 state to exactly the decision-bearing native fields."""

    state = controller.state_dict()
    families = state["families"]
    return {
        "epochs": [int(families[name]["epoch"]) for name in FAMILY_NAMES],
        "survivor_masks": [int(families[name]["survivor_mask"]) for name in FAMILY_NAMES],
        "version": int(state["version"]),
        "vote_counts": [list(families[name]["vote_counts"]) for name in FAMILY_NAMES],
    }


def initial_semantic_state_json() -> str:
    """Return the canonical checkpoint for the full initial version space."""

    return canonical_json(reference_semantic_state(exp7226.PackedBeliefController()))


def reference_query(
    state: Mapping[str, Any],
    families: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
) -> JsonDict:
    """Evaluate semantic state independently of the PyO3 implementation."""

    decisions: list[int] = []
    disagreements: list[float] = []
    statuses: list[int] = []
    energies: list[float | None] = []
    masks = state["survivor_masks"]
    votes = state["vote_counts"]
    for raw_family, raw_value, raw_label in zip(families, values, labels, strict=True):
        family = int(raw_family)
        label = int(raw_label)
        if family >= len(FAMILY_NAMES):
            decisions.append(-2)
            disagreements.append(0.0)
            statuses.append(-2)
            energies.append(None)
            continue
        value = int(raw_value) % DOMAIN_SIZE
        survivor_count = int(masks[family]).bit_count()
        if survivor_count == 0:
            decisions.append(-1)
            disagreements.append(0.0)
            statuses.append(0)
            energies.append(None)
            continue
        accepts = int(votes[family][value])
        decisions.append(1 if accepts > survivor_count / 2 else 0)
        disagreements.append(min(accepts, survivor_count - accepts) / survivor_count)
        if label not in (0, 1):
            statuses.append(-1)
            energies.append(None)
        else:
            disagrees = survivor_count - accepts if label == 1 else accepts
            statuses.append(1)
            energies.append(disagrees / survivor_count)
    return {
        "decisions": decisions,
        "disagreements": disagreements,
        "energy_status": statuses,
        "energies": energies,
    }


def release(event_id: str, family: int, value: int, label: int, role: int) -> JsonDict:
    """Create one due public release for the Exp7226 reference controller."""

    return {
        "event_id": event_id,
        "family_id": FAMILY_NAMES[family],
        "numeric_value": value,
        "observed_label": "accept" if label == 1 else "reject",
        "role": "support" if role == 1 else "validation",
        "request_index": 0,
        "release_index": 0,
    }


def _query_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all supported inputs with alternating energy labels."""

    families = np.repeat(np.arange(len(FAMILY_NAMES), dtype=np.uint8), DOMAIN_SIZE)
    values = np.tile(np.arange(DOMAIN_SIZE, dtype=np.int64), len(FAMILY_NAMES))
    labels = (values % 2).astype(np.int8)
    return tuple(np.ascontiguousarray(array) for array in (families, values, labels))


def run_exhaustive_parity(binding: ModuleType, *, max_subset_size: int = 2) -> JsonDict:
    """Compare every finite input for empty, singleton, and pair survivor sets."""

    if max_subset_size not in (1, 2):
        raise ValueError("max_subset_size must be one or two")
    mismatch_count = 0
    case_count = 0
    for family, family_name in enumerate(FAMILY_NAMES):
        for subset_size in range(max_subset_size + 1):
            for subset in itertools.combinations(exp7226.PARAMETER_DOMAIN, subset_size):
                python = exp7226.PackedBeliefController.from_survivors({family_name: set(subset)})
                state = reference_semantic_state(python)
                controller = binding.RustPackedBeliefController()
                controller.load_state(canonical_json(state))
                families = np.ascontiguousarray([family] * DOMAIN_SIZE, dtype=np.uint8)
                values = np.ascontiguousarray(range(DOMAIN_SIZE), dtype=np.int64)
                for label in (0, 1):
                    labels = np.ascontiguousarray([label] * DOMAIN_SIZE, dtype=np.int8)
                    native = dict(controller.query_batch(families, values, labels))
                    reference_result = reference_query(state, families, values, labels)
                    mismatch_count += int(
                        any(native[name] != reference_result[name] for name in reference_result)
                    )
                case_count += DOMAIN_SIZE
    return _finish_row(
        {
            "unit_id": "exhaustive-small-domain",
            "arm": "native_pyo3_vs_exp7226_python",
            "seed": RANDOM_SEED,
            "metric": mismatch_count,
            "error": None,
            "abstention": False,
            "max_subset_size": max_subset_size,
            "case_count": case_count,
            "energy_case_count": case_count * 2,
            "mismatch_count": mismatch_count,
            "passed": mismatch_count == 0,
        }
    )


def _sequence_trace(seed: int, steps: int) -> list[tuple[int, int, int, int]]:
    """Freeze one update sequence before either implementation evaluates it."""

    rng = random.Random(seed)
    return [
        (rng.randrange(4), rng.randrange(-33, 66), rng.randrange(2), int(index % 7 != 0))
        for index in range(steps)
    ]


def run_sequence_parity(
    binding: ModuleType,
    *,
    seeds: Sequence[int] = SEQUENCE_SEEDS,
    steps: int = 128,
) -> list[JsonDict]:
    """Compare fixed update, reload, decision, energy, reset, and state sequences."""

    rows: list[JsonDict] = []
    probes = _query_arrays()
    phase_started = time.monotonic()
    for sequence_index, seed in enumerate(seeds):
        trace = _sequence_trace(seed, steps)
        python = exp7226.PackedBeliefController()
        controller = binding.RustPackedBeliefController()
        mismatch_count = 0
        native_decisions: list[int] = []
        reference_decisions: list[int] = []
        energy_deltas: list[float] = []
        reset_count = 0
        for batch_start in range(0, steps, 8):
            batch = trace[batch_start : batch_start + 8]
            native_query = dict(controller.query_batch(*probes))
            state = reference_semantic_state(python)
            expected_query = reference_query(state, *probes)
            native_decisions.extend(native_query["decisions"])
            reference_decisions.extend(expected_query["decisions"])
            for actual, expected in zip(
                native_query["energies"], expected_query["energies"], strict=True
            ):
                energy_deltas.append(
                    0.0
                    if actual is None and expected is None
                    else abs(float(actual) - float(expected))
                )
            mismatch_count += sum(
                actual != expected
                for actual, expected in zip(
                    native_query["decisions"], expected_query["decisions"], strict=True
                )
            )
            families = np.ascontiguousarray([row[0] for row in batch], dtype=np.uint8)
            values = np.ascontiguousarray([row[1] for row in batch], dtype=np.int64)
            labels = np.ascontiguousarray([row[2] for row in batch], dtype=np.int8)
            roles = np.ascontiguousarray([row[3] for row in batch], dtype=np.uint8)
            native_receipt = dict(controller.update_batch(families, values, labels, roles))
            reference_receipt = python.commit_batch(
                [
                    release(f"{seed}:{batch_start + offset}", *row)
                    for offset, row in enumerate(batch)
                ],
                current_cycle=0,
                expected_parent_hash=python.state_hash(),
            )
            expected_resets = sum(
                int(operation["empty_reset"]) for operation in reference_receipt["operations"]
            )
            reset_count += expected_resets
            mismatch_count += int(native_receipt["reset_count"] != expected_resets)
            expected_state = canonical_json(reference_semantic_state(python))
            mismatch_count += int(controller.serialize_state() != expected_state)
            restored = binding.RustPackedBeliefController()
            restored.load_state(controller.serialize_state())
            controller = restored
        native_bytes = controller.serialize_state().encode()
        reference_bytes = canonical_json(reference_semantic_state(python)).encode()
        mismatch_count += int(native_bytes != reference_bytes)
        rows.append(
            _finish_row(
                {
                    "unit_id": f"sequence:{seed}",
                    "arm": "native_pyo3_vs_exp7226_python",
                    "seed": seed,
                    "metric": mismatch_count,
                    "error": None,
                    "abstention": native_decisions.count(-1),
                    "steps": steps,
                    "reload_count": (steps + 7) // 8,
                    "reset_count": reset_count,
                    "native_state_b64": base64.b64encode(native_bytes).decode(),
                    "reference_state_b64": base64.b64encode(reference_bytes).decode(),
                    "native_decisions": native_decisions,
                    "reference_decisions": reference_decisions,
                    "energy_deltas": energy_deltas,
                    "maximum_energy_delta": max(energy_deltas, default=0.0),
                    "mismatch_count": mismatch_count,
                    "passed": mismatch_count == 0 and not any(energy_deltas),
                }
            )
        )
        if time.monotonic() - phase_started >= 60 or sequence_index + 1 == len(seeds):
            print(
                f"[phase 5 progress] completed_sequences={sequence_index + 1}/{len(seeds)} "
                f"elapsed_s={time.monotonic() - phase_started:.3f}",
                flush=True,
            )
            phase_started = time.monotonic()
    return rows


def run_cross_process_restore(extension: Path, checkpoint: str) -> JsonDict:
    """Restore one native checkpoint in a bounded fresh Python process."""

    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        "-c",
        _RESTORE_HELPER,
        str(extension.resolve()),
    ]
    progress(5, "before", "fresh-process native checkpoint restore")
    started = time.monotonic()
    with exp7217._Heartbeat("Exp7230 cross-process native restore", interval_s=30):
        completed = subprocess.run(
            command,
            input=checkpoint,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    result: JsonDict = {}
    marker = "__CARNOT_JSON__"
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            value = json.loads(line[len(marker) :])
            if isinstance(value, Mapping):
                result = dict(value)
    passed = (
        completed.returncode == 0
        and result.get("module_file") == str(extension.resolve())
        and result.get("serialized_state") == checkpoint
    )
    receipt = {
        "unit_id": "cross-process-restore",
        "arm": "native_pyo3_fresh_process",
        "seed": SEQUENCE_SEEDS[-1],
        "metric": int(not passed),
        "error": None if completed.returncode == 0 else completed.stderr,
        "abstention": False,
        "command": command,
        "exit_code": completed.returncode,
        "duration_s": time.monotonic() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "module_file": result.get("module_file"),
        "serialized_state": result.get("serialized_state"),
        "decisions": result.get("decisions"),
        "python_fallback_used": False,
        "passed": passed,
    }
    progress(5, "after", f"fresh-process native checkpoint restore passed={passed}")
    return _finish_row(receipt)


def _benchmark_trace(batch_size: int, seed: int) -> list[tuple[int, int, int, int]]:
    """Freeze one matched operation trace before randomized arm execution."""

    return _sequence_trace(seed + batch_size, batch_size)


def _native_trace(binding: ModuleType, trace: Sequence[tuple[int, int, int, int]]) -> JsonDict:
    """Measure native marshaling, two queries, update, checkpoint, and output."""

    started = time.perf_counter_ns()
    controller = binding.RustPackedBeliefController()
    families = np.ascontiguousarray([row[0] for row in trace], dtype=np.uint8)
    values = np.ascontiguousarray([row[1] for row in trace], dtype=np.int64)
    labels = np.ascontiguousarray([row[2] for row in trace], dtype=np.int8)
    roles = np.ascontiguousarray([row[3] for row in trace], dtype=np.uint8)
    before = dict(controller.query_batch(families, values, labels))
    update = dict(controller.update_batch(families, values, labels, roles))
    after = dict(controller.query_batch(families, values, labels))
    checkpoint = controller.serialize_state()
    output = canonical_json(
        {
            "before": {name: value for name, value in before.items() if name != "kernel_ns"},
            "update": {name: value for name, value in update.items() if name != "kernel_ns"},
            "after": {name: value for name, value in after.items() if name != "kernel_ns"},
        }
    )
    output_hash = hashlib.sha256(output.encode()).hexdigest()
    return {
        "end_to_end_ns": time.perf_counter_ns() - started,
        "kernel_ns": int(before["kernel_ns"]) + int(update["kernel_ns"]) + int(after["kernel_ns"]),
        "checkpoint_bytes": len(checkpoint.encode()),
        "output_bytes": len(output.encode()),
        "output_sha256": "sha256:" + output_hash,
    }


def _python_trace(trace: Sequence[tuple[int, int, int, int]]) -> JsonDict:
    """Measure the shipped Python controller over the identical operation trace."""

    started = time.perf_counter_ns()
    controller = exp7226.PackedBeliefController()
    events = [
        {"event_id": f"cost:{index}", "family_id": FAMILY_NAMES[row[0]], "numeric_value": row[1]}
        for index, row in enumerate(trace)
    ]
    kernel_started = time.perf_counter_ns()
    before = [
        {
            "prediction": controller.predict(event),
            "energy": controller.energy("accept" if row[2] else "reject", event),
        }
        for event, row in zip(events, trace, strict=True)
    ]
    receipt = controller.commit_batch(
        [release(f"cost:{index}", *row) for index, row in enumerate(trace)],
        current_cycle=0,
        expected_parent_hash=controller.state_hash(),
    )
    after = [
        {
            "prediction": controller.predict(event),
            "energy": controller.energy("accept" if row[2] else "reject", event),
        }
        for event, row in zip(events, trace, strict=True)
    ]
    checkpoint = controller.state_bytes()
    kernel_ns = time.perf_counter_ns() - kernel_started
    output = canonical_json(
        {
            "before": before,
            "update": {
                "release_count": receipt["release_count"],
                "operations": receipt["operations"],
            },
            "after": after,
        }
    )
    output_hash = hashlib.sha256(output.encode()).hexdigest()
    return {
        "end_to_end_ns": time.perf_counter_ns() - started,
        "kernel_ns": kernel_ns,
        "checkpoint_bytes": len(checkpoint),
        "output_bytes": len(output.encode()),
        "output_sha256": "sha256:" + output_hash,
    }


def run_cost_benchmark(
    binding: ModuleType,
    *,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    repetitions: int = REPETITIONS,
    seed: int = RANDOM_SEED,
    timeout_s: float = MEASUREMENT_TIMEOUT_S,
) -> list[JsonDict]:
    """Measure fixed paired traces after separate warmup and before the hard cap."""

    if repetitions <= 0 or not batch_sizes:
        raise ValueError("benchmark roster must be nonempty")
    traces = {batch: _benchmark_trace(batch, seed) for batch in batch_sizes}
    progress(6, "before", "separated native and Python benchmark warmup")
    for batch in batch_sizes:
        _native_trace(binding, traces[batch])
        _python_trace(traces[batch])
    progress(6, "after", "separated native and Python benchmark warmup")
    rng = random.Random(seed)
    orders = {}
    for batch in batch_sizes:
        for repetition in range(repetitions):
            order = ["native_pyo3", "python_reference"]
            rng.shuffle(order)
            orders[(batch, repetition)] = order
    rows: list[JsonDict] = []
    started = time.monotonic()
    progress(6, "before", "paired end-to-end cost benchmark")
    for batch in batch_sizes:
        for repetition in range(repetitions):
            if time.monotonic() - started > timeout_s:
                raise TimeoutError("cost measurement exceeded fixed timeout")
            for order_index, arm in enumerate(orders[(batch, repetition)]):
                measured = (
                    _native_trace(binding, traces[batch])
                    if arm == "native_pyo3"
                    else _python_trace(traces[batch])
                )
                rows.append(
                    _finish_row(
                        {
                            "unit_id": f"cost:{batch}:{repetition}:{arm}",
                            "arm": arm,
                            "seed": seed,
                            "metric": measured["end_to_end_ns"] / batch,
                            "error": None,
                            "abstention": False,
                            "batch_size": batch,
                            "repetition": repetition,
                            "arm_order": order_index,
                            "end_to_end_ns": measured["end_to_end_ns"],
                            "amortized_end_to_end_ns": measured["end_to_end_ns"] / batch,
                            "kernel_ns": measured["kernel_ns"],
                            "checkpoint_bytes": measured["checkpoint_bytes"],
                            "output_bytes": measured["output_bytes"],
                            "output_sha256": measured["output_sha256"],
                            "trace_sha256": "sha256:"
                            + hashlib.sha256(canonical_json(traces[batch]).encode()).hexdigest(),
                        }
                    )
                )
    progress(6, "after", f"paired cost rows={len(rows)} elapsed_s={time.monotonic() - started:.3f}")
    return rows


def _bootstrap_lower(values: Sequence[float], *, seed: int, draws: int = 10_000) -> float:
    """Return the fixed paired bootstrap lower percentile for mean speed ratio."""

    if not values:
        raise ValueError("paired ratios are empty")
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws)]
    means.sort()
    return means[int(0.025 * draws)]


def summarize_cost(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_batches: Sequence[int] = BATCH_SIZES,
    repetitions: int = REPETITIONS,
) -> JsonDict:
    """Derive paired confidence bounds and both fixed performance gates."""

    cells: list[JsonDict] = []
    for batch in required_batches:
        ratios: list[float] = []
        for repetition in range(repetitions):
            pair = {
                str(row["arm"]): float(row["end_to_end_ns"])
                for row in rows
                if row.get("batch_size") == batch and row.get("repetition") == repetition
            }
            if set(pair) != {"native_pyo3", "python_reference"}:
                raise ValueError(f"incomplete paired cost cell:{batch}:{repetition}")
            ratios.append(pair["python_reference"] / pair["native_pyo3"])
        lower = _bootstrap_lower(ratios, seed=RANDOM_SEED + batch)
        cells.append(
            {
                "batch_size": batch,
                "paired_repetitions": repetitions,
                "speed_ratio_python_over_native": sum(ratios) / len(ratios),
                "paired_lower_ci95": lower,
                "paired_ratios": ratios,
                "native_faster_gate": lower > 1.0,
                "nfr_01_10x_gate": lower >= 10.0,
            }
        )
    return {
        "cells": cells,
        "native_cost_value_score": int(all(cell["native_faster_gate"] for cell in cells)),
        "nfr_01_10x_met": all(cell["nfr_01_10x_gate"] for cell in cells),
    }


def synthetic_cost_rows(
    batches: Sequence[int],
    repetitions: int,
    *,
    python_ns: int,
    native_ns: int,
) -> list[JsonDict]:
    """Create deterministic paired rows for gate and validator tests."""

    rows = []
    for batch in batches:
        for repetition in range(repetitions):
            for arm, duration in (("native_pyo3", native_ns), ("python_reference", python_ns)):
                rows.append(
                    _finish_row(
                        {
                            "unit_id": f"fixture:{batch}:{repetition}:{arm}",
                            "arm": arm,
                            "seed": RANDOM_SEED,
                            "metric": duration / batch,
                            "error": None,
                            "abstention": False,
                            "batch_size": batch,
                            "repetition": repetition,
                            "arm_order": int(arm == "python_reference"),
                            "end_to_end_ns": duration,
                            "amortized_end_to_end_ns": duration / batch,
                            "kernel_ns": duration,
                            "checkpoint_bytes": 1,
                            "output_bytes": 1,
                            "output_sha256": "sha256:fixture",
                            "trace_sha256": "sha256:fixture",
                        }
                    )
                )
    return rows


def sequence_fixture_row(seed: int) -> JsonDict:
    """Create one exact sequence row for artifact validator tests."""

    encoded = base64.b64encode(initial_semantic_state_json().encode()).decode()
    return _finish_row(
        {
            "unit_id": f"sequence:{seed}",
            "arm": "native_pyo3_vs_exp7226_python",
            "seed": seed,
            "metric": 0,
            "error": None,
            "abstention": 0,
            "steps": 1,
            "reload_count": 1,
            "reset_count": 0,
            "native_state_b64": encoded,
            "reference_state_b64": encoded,
            "native_decisions": [0],
            "reference_decisions": [0],
            "energy_deltas": [0.0],
            "maximum_energy_delta": 0.0,
            "mismatch_count": 0,
            "passed": True,
        }
    )


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
    paths: ExperimentPaths,
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create all fields before the task chooses a terminal outcome."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_parity_sequences": len(SEQUENCE_SEEDS),
            "attempted_parity_sequences": 0,
            "completed_parity_sequences": 0,
            "censored_parity_sequences": len(SEQUENCE_SEEDS),
            "planned_cost_rows": len(BATCH_SIZES) * REPETITIONS * 2,
            "attempted_cost_rows": 0,
            "completed_cost_rows": 0,
            "censored_cost_rows": len(BATCH_SIZES) * REPETITIONS * 2,
            "independent_sequence_units": 0,
            "paired_cost_units": 0,
        },
        "random_seed": {
            "global": RANDOM_SEED,
            "sequences": list(SEQUENCE_SEEDS),
            "arm_order": RANDOM_SEED,
            "bootstrap": RANDOM_SEED,
        },
        "reproducibility_checksum": None,
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external:unknown_precondition",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "native_belief_ready_score": 0,
        "native_cost_value_score": 0,
        "native_identity_receipt": {},
        "parity_rows": [],
        "cost_rows": [],
        "cost_summary": {"cells": [], "native_cost_value_score": 0, "nfr_01_10x_met": False},
        "nfr_01_10x_met": False,
        "scientific_value_inherited": False,
        "checkpoint_path": {"path": str(paths.checkpoint), "sha256": None, "bytes": 0},
        "sampler_nfr_history": {
            "upstream": "results/experiment_7202_v634_slice_cost_quality.json",
            "nfr_01_10x_met": False,
            "reopened": False,
        },
        "spec_refs": ["REQ-CL-7230", "REQ-RUSTPY-7230"],
    }


def blocked_artifact_for_test(failed: Mapping[str, Any]) -> JsonDict:
    """Return a schema-complete row-free external-block fixture."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        [failed], {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=0.0
    )
    artifact["honest_verdict"] = f"blocked_external:{failed.get('check')}"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
    paths: ExperimentPaths,
    *,
    started_at: str,
    duration_s: float,
) -> JsonDict:
    """Publish exact diagnosis only for an unchanged external precondition."""

    artifact = _base_artifact(
        checks,
        hashes,
        paths,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=duration_s,
    )
    failed = artifact["gate_check_summary"]["failed_check"]
    artifact["honest_verdict"] = f"blocked_external:{failed}"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _native_identity(
    root: Path,
    extension: Path,
    binding: ModuleType,
    build_receipt: Mapping[str, Any],
) -> JsonDict:
    """Bind the executing interpreter and actual loaded extension identity."""

    controller = binding.RustPackedBeliefController()
    return {
        "interpreter": str(Path(sys.executable).absolute()),
        "interpreter_resolved": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "pyo3_configuration": exp7217.interpreter_metadata(root)["cargo"],
        "build_flags": {
            "command": build_receipt.get("command"),
            "PYO3_PYTHON": build_receipt.get("PYO3_PYTHON"),
            "CARGO_TARGET_DIR": build_receipt.get("CARGO_TARGET_DIR"),
            "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": build_receipt.get(
                "PYO3_USE_ABI3_FORWARD_COMPATIBILITY"
            ),
        },
        "module_file": str(Path(binding.__file__).resolve()),
        "shared_library_sha256": sha256_file(extension),
        "native_entrypoint": "carnot._rust.RustPackedBeliefController",
        "native_methods": [
            name
            for name in (
                "query_batch",
                "update_batch",
                "serialize_state",
                "load_state",
                "rollback",
                "reset",
            )
            if hasattr(controller, name)
        ],
        "linkage": exp7217._linkage_receipt(extension),
        "compiled_execution": True,
        "python_fallback_used": False,
    }


def build_artifact(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Run authenticated native parity and one fixed paired measurement roster."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    checks, hashes = collect_preconditions(root, paths)
    if not all(row["passed"] for row in checks):
        return _blocked_artifact(
            checks, hashes, paths, started_at=started_at, duration_s=time.monotonic() - started
        )
    progress(1, "start", "activate flushed progress and bounded subprocess reporting")
    progress(1, "end", "progress contract active")
    progress(2, "start", "CPU exact solver or simulator; MODEL_SPECS is empty")
    progress(2, "end", "no model load or generation occurred")
    progress(3, "start", "build and load the interpreter-bound compiled extension")
    extension, build_receipt = build_native_extension(root)
    progress(3, "before", "load exact compiled extension")
    binding = load_native_extension(extension)
    progress(3, "after", f"loaded module_file={Path(binding.__file__).resolve()}")
    identity = _native_identity(root, extension, binding, build_receipt)
    hashes[str(extension.resolve())] = sha256_file(extension)
    progress(3, "end", "compiled identity receipt complete")

    progress(4, "start", "confirm scoped persistent packed survivor and vote core")
    progress(4, "end", f"native methods={len(identity['native_methods'])} harness remains Python")
    progress(5, "start", "exhaustive small domains and twenty fixed update/reload sequences")
    exhaustive = run_exhaustive_parity(binding)
    sequences = run_sequence_parity(binding)
    checkpoint = sequences[-1]["native_state_b64"]
    checkpoint_text = base64.b64decode(checkpoint).decode()
    restore = run_cross_process_restore(extension, checkpoint_text)
    parity_rows = []
    for index, row in enumerate((exhaustive, *sequences, restore)):
        normalized = dict(row)
        normalized.setdefault("unit_id", f"parity:{index}")
        parity_rows.append(_finish_row(normalized))
    parity_mismatches = sum(
        int(row.get("mismatch_count", row.get("metric", 0))) for row in parity_rows
    )
    ready = int(
        parity_mismatches == 0
        and len(sequences) == len(SEQUENCE_SEEDS)
        and all(row.get("passed") is True for row in parity_rows)
        and restore["module_file"] == str(extension.resolve())
    )
    progress(5, "end", f"parity rows={len(parity_rows)} mismatches={parity_mismatches}")
    if ready != 1:
        raise RuntimeError("owned native parity or process restore failed")

    checkpoint_payload = {
        "schema": "carnot.exp7230.checkpoint.v1",
        "selected_extension": str(extension.resolve()),
        "selected_extension_sha256": sha256_file(extension),
        "restored_state": checkpoint_text,
        "sequence_rows": sequences,
    }
    checkpoint_receipt = atomic_write(paths.checkpoint, checkpoint_payload)
    hashes[str(paths.checkpoint.resolve())] = checkpoint_receipt["sha256"]

    progress(6, "start", "fixed randomized paired cost roster")
    cost_rows = run_cost_benchmark(binding)
    progress(7, "start", "derive paired confidence gates without extra repetitions")
    cost_summary = summarize_cost(cost_rows)
    progress(
        6, "end", f"cost rows={len(cost_rows)} value={cost_summary['native_cost_value_score']}"
    )
    progress(
        7,
        "end",
        f"native_cost_value={cost_summary['native_cost_value_score']} "
        f"nfr_01_10x={cost_summary['nfr_01_10x_met']}",
    )
    duration = time.monotonic() - started
    verdict_class = "positive" if cost_summary["native_cost_value_score"] == 1 else "null"
    verdict = (
        "complete: native packed belief matched the Python reference and restored across a "
        "fresh process; the paired end-to-end native cost gate passed. This does not supply "
        "new learning value or reopen the sampler claim."
        if verdict_class == "positive"
        else "complete_null: native packed belief matched the Python reference and restored "
        "across a fresh process, but the paired end-to-end native cost gate did not pass. "
        "This does not supply new learning value or reopen the sampler claim."
    )
    artifact = _base_artifact(
        checks,
        hashes,
        paths,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=duration,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": [*parity_rows, *cost_rows],
            "sample_size_budget": {
                "planned_parity_sequences": len(SEQUENCE_SEEDS),
                "attempted_parity_sequences": len(SEQUENCE_SEEDS),
                "completed_parity_sequences": len(sequences),
                "censored_parity_sequences": 0,
                "planned_cost_rows": len(BATCH_SIZES) * REPETITIONS * 2,
                "attempted_cost_rows": len(cost_rows),
                "completed_cost_rows": len(cost_rows),
                "censored_cost_rows": 0,
                "independent_sequence_units": len(sequences),
                "paired_cost_units": len(BATCH_SIZES) * REPETITIONS,
            },
            "verdict_class": verdict_class,
            "honest_verdict": verdict,
            "native_belief_ready_score": ready,
            "native_cost_value_score": cost_summary["native_cost_value_score"],
            "native_identity_receipt": identity,
            "parity_rows": parity_rows,
            "cost_rows": cost_rows,
            "cost_summary": cost_summary,
            "nfr_01_10x_met": cost_summary["nfr_01_10x_met"],
            "checkpoint_path": checkpoint_receipt,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _row_hash_valid(row: Mapping[str, Any]) -> bool:
    """Recompute one row identity without trusting the stored digest."""

    material = dict(row)
    stored = material.pop("row_sha256", None)
    expected = "sha256:" + hashlib.sha256(canonical_json(material).encode()).hexdigest()
    return stored == expected


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    check_files: bool = False,
    root: Path = REPO_ROOT,
) -> list[str]:
    """Cold-check schema, native provenance, parity, cost gates, and hashes."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_FIELDS.issubset(artifact), "missing_fields")
    if not REQUIRED_FIELDS.issubset(artifact):
        return errors
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id")
    add(artifact.get("milestone") != MILESTONE, "milestone")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(not artifact.get("execution_host"), "execution_host")
    add(
        artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False,
        "model_declaration",
    )
    add(artifact.get("verifier_is_oracle") is not True, "verifier_authority")
    add(artifact.get("scientific_value_inherited") is not False, "scientific_value")
    add(artifact.get("sampler_nfr_history", {}).get("reopened") is not False, "sampler_history")
    try:
        checksum_valid = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_valid = False
    add(not checksum_valid, "reproducibility_checksum")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        gate = artifact.get("gate_check_summary", {})
        add(artifact.get("status") != "blocked", "blocked_status")
        add(artifact.get("inference_substrate") != "blocked_no_run", "blocked_substrate")
        add(artifact.get("inference_substrate_class") != "blocked_no_run", "blocked_class")
        add(
            any(artifact.get(name) for name in ("rows", "parity_rows", "cost_rows")),
            "blocked_rows",
        )
        add(artifact.get("native_belief_ready_score") != 0, "blocked_ready")
        add(artifact.get("native_cost_value_score") != 0, "blocked_cost")
        add(
            not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or any(
                name not in gate
                for name in (
                    "failed_check",
                    "upstream",
                    "field",
                    "expected_value",
                    "observed_value",
                )
            ),
            "blocked_gate",
        )
        return errors

    parity_rows = artifact.get("parity_rows", [])
    cost_rows = artifact.get("cost_rows", [])
    add(artifact.get("status") != "complete", "status")
    add(artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate")
    add(
        artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate_class",
    )
    add(artifact.get("gate_check_summary", {}).get("passed") is not True, "preconditions")
    add(
        not isinstance(parity_rows, list)
        or len(parity_rows) != len(SEQUENCE_SEEDS) + 2
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True for row in parity_rows
        ),
        "parity_rows",
    )
    add(
        not isinstance(cost_rows, list) or len(cost_rows) != len(BATCH_SIZES) * REPETITIONS * 2,
        "cost_rows",
    )
    all_rows = artifact.get("rows", [])
    add(
        not isinstance(all_rows, list)
        or len(all_rows) != len(parity_rows) + len(cost_rows)
        or any(
            not isinstance(row, Mapping)
            or not {"unit_id", "arm", "seed", "metric", "error", "abstention", "row_sha256"}
            <= set(row)
            or not _row_hash_valid(row)
            for row in all_rows
        ),
        "rows",
    )
    identity = artifact.get("native_identity_receipt", {})
    add(
        not isinstance(identity, Mapping)
        or identity.get("compiled_execution") is not True
        or identity.get("python_fallback_used") is not False
        or not str(identity.get("module_file", "")).endswith(
            str(sysconfig.get_config_var("EXT_SUFFIX"))
        )
        or not str(identity.get("shared_library_sha256", "")).startswith("sha256:")
        or identity.get("native_entrypoint") != "carnot._rust.RustPackedBeliefController",
        "native_identity",
    )
    ready_expected = int(
        isinstance(parity_rows, list)
        and len(parity_rows) == len(SEQUENCE_SEEDS) + 2
        and all(row.get("passed") is True for row in parity_rows if isinstance(row, Mapping))
    )
    add(artifact.get("native_belief_ready_score") != ready_expected, "ready_score")
    try:
        cost_expected = summarize_cost(cost_rows)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        cost_expected = None
    add(cost_expected is None or artifact.get("cost_summary") != cost_expected, "cost_summary")
    add(
        cost_expected is None
        or artifact.get("native_cost_value_score") != cost_expected["native_cost_value_score"],
        "cost_gate",
    )
    add(
        cost_expected is None or artifact.get("nfr_01_10x_met") != cost_expected["nfr_01_10x_met"],
        "nfr_gate",
    )
    add(
        artifact.get("verdict_class")
        != ("positive" if artifact.get("native_cost_value_score") == 1 else "null"),
        "verdict_class",
    )
    add(not str(artifact.get("honest_verdict", "")).startswith("complete"), "honest_verdict")
    budget = artifact.get("sample_size_budget", {})
    add(
        budget.get("completed_parity_sequences") != len(SEQUENCE_SEEDS)
        or budget.get("censored_parity_sequences") != 0
        or budget.get("completed_cost_rows") != len(BATCH_SIZES) * REPETITIONS * 2
        or budget.get("censored_cost_rows") != 0,
        "sample_size_budget",
    )
    if check_files:
        for path_text, expected in artifact.get("source_artifact_hashes", {}).items():
            path = Path(path_text)
            resolved = path if path.is_absolute() else root / path
            add(not resolved.is_file() or sha256_file(resolved) != expected, "source_hashes")
        module_file = Path(str(identity.get("module_file", "")))
        add(
            not module_file.is_file()
            or sha256_file(module_file) != identity.get("shared_library_sha256"),
            "native_binary_hash",
        )
    return errors


def complete_artifact_fixture_for_test() -> JsonDict:
    """Create a complete measured-shape artifact for closed validator tests."""

    now = datetime.now(UTC).isoformat()
    checks = [check("fixture", "fixture", "value", 1, 1, True)]
    artifact = _base_artifact(
        checks, {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=1.0
    )
    exhaustive = _finish_row(
        {
            "unit_id": "exhaustive-small-domain",
            "arm": "native_pyo3_vs_exp7226_python",
            "seed": RANDOM_SEED,
            "metric": 0,
            "error": None,
            "abstention": False,
            "case_count": 1,
            "mismatch_count": 0,
            "passed": True,
        }
    )
    sequences = [sequence_fixture_row(seed) for seed in SEQUENCE_SEEDS]
    restore = _finish_row(
        {
            "unit_id": "cross-process-restore",
            "arm": "native_pyo3_fresh_process",
            "seed": SEQUENCE_SEEDS[-1],
            "metric": 0,
            "error": None,
            "abstention": False,
            "module_file": f"/tmp/_rust{sysconfig.get_config_var('EXT_SUFFIX')}",
            "serialized_state": initial_semantic_state_json(),
            "python_fallback_used": False,
            "passed": True,
        }
    )
    parity = [exhaustive, *sequences, restore]
    cost = synthetic_cost_rows(BATCH_SIZES, REPETITIONS, python_ns=100, native_ns=50)
    summary = summarize_cost(cost)
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": [*parity, *cost],
            "sample_size_budget": {
                "planned_parity_sequences": len(SEQUENCE_SEEDS),
                "attempted_parity_sequences": len(SEQUENCE_SEEDS),
                "completed_parity_sequences": len(SEQUENCE_SEEDS),
                "censored_parity_sequences": 0,
                "planned_cost_rows": len(cost),
                "attempted_cost_rows": len(cost),
                "completed_cost_rows": len(cost),
                "censored_cost_rows": 0,
                "independent_sequence_units": len(SEQUENCE_SEEDS),
                "paired_cost_units": len(BATCH_SIZES) * REPETITIONS,
            },
            "verdict_class": "positive",
            "honest_verdict": "complete: validator fixture",
            "native_belief_ready_score": 1,
            "native_cost_value_score": summary["native_cost_value_score"],
            "native_identity_receipt": {
                "module_file": f"/tmp/_rust{sysconfig.get_config_var('EXT_SUFFIX')}",
                "shared_library_sha256": "sha256:fixture",
                "native_entrypoint": "carnot._rust.RustPackedBeliefController",
                "compiled_execution": True,
                "python_fallback_used": False,
            },
            "parity_rows": parity,
            "cost_rows": cost,
            "cost_summary": summary,
            "nfr_01_10x_met": summary["nfr_01_10x_met"],
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def atomic_write(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Publish complete JSON through one same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(root: Path, output: Path, run_date: str) -> JsonDict:
    """Build, cold-validate, and atomically publish the terminal artifact."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    paths = ExperimentPaths(REPO_ROOT / CHECKPOINT_RELATIVE, output)
    artifact = build_artifact(root, paths)
    progress(8, "before", "final cold validation")
    errors = validate_artifact(
        artifact,
        check_files=artifact.get("status") == "complete"
        and bool(artifact.get("source_artifact_hashes")),
        root=root,
    )
    progress(8, "after", f"final cold validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7230 artifact:{errors}")
    progress(9, "before", "required-field check and atomic terminal write")
    receipt = atomic_write(output, artifact)
    progress(9, "after", f"atomic terminal write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and optional read-only validation path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded study or validate existing bytes without mutation."""

    args = _parse_args(argv)
    if args.validate is not None:
        progress(8, "before", f"read-only validation path={args.validate}")
        try:
            artifact = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(artifact)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"validation_error: {exc}", flush=True)
            progress(8, "after", "read-only validation errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        progress(8, "after", f"read-only validation errors={len(errors)}")
        return 0 if not errors else 2
    if args.date != RUN_DATE:
        print(f"experiment_error: run date must be {RUN_DATE}", flush=True)
        return 2
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        run_experiment(REPO_ROOT, output, args.date)
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"experiment_error: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
