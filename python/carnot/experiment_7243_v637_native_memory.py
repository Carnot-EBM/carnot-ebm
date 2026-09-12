"""Measure native packed-state work inside the recurring archive controller.

The archive policy stays in the Exp7240 Python controller for both arms. The
native arm replaces only active packed prediction, energy, update, and state
conversion with the existing PyO3 kernel.

Spec refs: REQ-CL-7243, SCENARIO-CL-7243-*, REQ-RUSTPY-7243, and
SCENARIO-RUSTPY-7243-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
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
from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7243
TASK_ID = "exp7243-native-memory"
SCHEMA = "carnot.exp7243.v637_native_memory.v1"
MILESTONE = "2026.09.637"
RUN_DATE = "20260912"
RANDOM_SEED = 7_243_000
STREAM_SEEDS = exp7240.STREAM_SEEDS
ARCHIVE_CAPACITIES = (1, 2, 4)
BATCH_SIZES = (1, 16, 128)
PAIRED_BLOCKS = 30
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE = Path("results/experiment_7243_v637_native_memory.json")
CHECKPOINT_RELATIVE = Path("results/checkpoints/experiment_7243_v637_native_memory.json")
UPSTREAM_RELATIVE = Path("results/experiment_7240_v637_recurrence_fixture.json")
ABI_RELATIVE = Path("results/experiment_7217_v635_abi_board_readiness.json")
HISTORY_RELATIVE = Path("results/experiment_7230_v636_native_belief.json")
ROADMAP_RELATIVE = Path("research-roadmap.yaml")
EXCLUSION_RELATIVE = Path("ops/exclusion_manifest.yaml")
CL_SPEC_RELATIVE = Path("openspec/capabilities/continuous-learning/spec.md")
RUST_SPEC_RELATIVE = Path("openspec/capabilities/rust-python-boundary/spec.md")
TARGET_RELATIVE = Path("target/experiment-7243-interpreter-bound")
LOAD_RELATIVE = Path("target/experiment-7243-load")
EXPECTED_EXP7240_SHA256 = "sha256:0f12abc839f3d70006698ebaa5169f12ab1cae9dc7526c6d124078ec7a6baf74"
EXPECTED_EXP7217_SHA256 = "sha256:49e522b997c3d38aaf0d2ebdd6bad10db4696dd7f9ef1f81025775896b8e44e8"
EXPECTED_EXP7230_SHA256 = "sha256:dd49b1bb5a61898ee3d5a7f177b5ef1ec1e2c4c42c21e88ed6ec2e40777cbe03"

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
    Path("python/carnot/experiment_7217_v635_abi_board_readiness.py"),
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/experiment_7230_v636_native_belief.py"),
    Path("python/carnot/experiment_7240_v637_recurrence_fixture.py"),
    Path("python/carnot/experiment_7243_v637_native_memory.py"),
    Path("scripts/experiments/experiment_7243_v637_native_memory.py"),
    Path("tests/python/test_experiment_7243_v637_native_memory.py"),
    CL_SPEC_RELATIVE,
    RUST_SPEC_RELATIVE,
    UPSTREAM_RELATIVE,
    ABI_RELATIVE,
    HISTORY_RELATIVE,
)

FIELD_PRINCIPLES = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "experiment_id": "A fixed identifier prevents another task from supplying this evidence.",
    "milestone": "The milestone binds this result to the V637 execution contract.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "started_at_utc": "Record the actual UTC start separately from the fixed run date.",
    "completed_at_utc": "Record the actual UTC completion separately from the fixed run date.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources and injected fixtures are separate.",
    "source_artifact_hashes": "Hash source code, public inputs, private evaluator inputs and raw output files.",
    "rows": "Every comparison retains one row per independent unit and arm, with errors and abstentions.",
    "sample_size_budget": "Predeclared independent units, attempted/completed/censored units, and stopping rule.",
    "random_seed": "Freeze seeds and schedules before observing evaluation labels.",
    "reproducibility_checksum": "Hash the exact settings, inputs and raw rows supporting the result.",
    "gate_check_summary": "Every blocked_* verdict names check, upstream, artifact_field, expected and observed value.",
    "verifier_is_oracle": "True when the verification authority also defines correctness; separate code is insufficient independence.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial.",
    "honest_verdict": "Completed findings start complete_ or complete:. External absence starts blocked_.",
    "acceptance_gate_results": "Preserve each frozen criterion, actual value and pass/fail independently of task completion.",
    "native_archive_ready_score": "Native execution and exact Python/native/restart parity on all scheduled cases.",
    "native_archive_cost_value_score": "Paired lower CI95 speedup exceeds one for complete batch-1 event cost.",
    "parity_rows": "Per stream and implementation predictions, queries, hashes and mismatch counts.",
    "cost_rows": "Each of 30 blocks, capacity, batch, arm and all timed boundary components.",
    "native_binary_receipt": "Interpreter path/version, module __file__, binary hash, build flags and extension origin.",
    "hardware_target_gaps": "Measured 10x and 100x lower-bound status plus byte and transfer requirements.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep provisional measurements apart from the terminal deliverable."""

    checkpoint: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the fixed repository paths used by the public command."""

        return cls(REPO_ROOT / CHECKPOINT_RELATIVE, REPO_ROOT / RESULT_RELATIVE)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test evidence below one caller-owned temporary directory."""

        return cls(root / "checkpoints/exp7243.json", root / "experiment_7243.json")


def progress(phase: int, boundary: str, operation: str) -> None:
    """Flush a truthful phase boundary for external liveness monitoring."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def canonical_json(value: Any) -> str:
    """Encode finite JSON with stable key order for evidence identity."""

    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def sha256_file(path: Path) -> str:
    """Hash actual file bytes in bounded chunks."""

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
    """Add common denominators and a stable row identity."""

    row.pop("row_sha256", None)
    row.setdefault("arm", "not_applicable")
    row.setdefault("seed", None)
    row.setdefault("metric", 0.0)
    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row["row_sha256"] = "sha256:" + hashlib.sha256(canonical_json(row).encode()).hexdigest()
    return row


def unwrap_principled_value(value: Any) -> Any:
    """Unwrap only an exact two-key principle and value record."""

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


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all checks and expose the first failed gate."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "checks": [dict(row) for row in checks],
        "failed_check": None if failed is None else failed.get("check"),
        "upstream": None if failed is None else failed.get("upstream"),
        "artifact_field": None if failed is None else failed.get("field"),
        "expected_value": None if failed is None else failed.get("expected_value"),
        "observed_value": None if failed is None else failed.get("observed_value"),
    }


def read_object(path: Path) -> JsonDict:
    """Read one JSON object while malformed bytes remain a failed check."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _writable(path: Path) -> bool:
    """Check the nearest existing parent without creating terminal bytes."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _task_contract(root: Path) -> JsonDict | None:
    """Read only the V637 task fields that authorize Exp7243."""

    try:
        document = yaml.safe_load((root / ROADMAP_RELATIVE).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    tasks = document.get("tasks") if isinstance(document, Mapping) else None
    task = next(
        (row for row in tasks or [] if isinstance(row, Mapping) and row.get("id") == TASK_ID),
        None,
    )
    if not isinstance(task, Mapping):
        return None
    return {
        "id": task.get("id"),
        "milestone": task.get("milestone"),
        "deliverable": task.get("deliverable"),
        "gated_on": task.get("gated_on"),
    }


def _receipt_hashes_match(root: Path, artifact: Mapping[str, Any]) -> bool:
    """Authenticate the four Exp7240 stream receipts from their exact paths."""

    receipts = artifact.get("stream_receipts")
    if not isinstance(receipts, Mapping) or len(receipts) != 4:
        return False
    for receipt in receipts.values():
        if not isinstance(receipt, Mapping):
            return False
        path = Path(str(receipt.get("path", "")))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != receipt.get("sha256"):
            return False
    return True


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str], dict[str, JsonDict]]:
    """Authenticate requirements, exact producers, quarantine, tools, and paths."""

    progress(0, "start", "authenticate sources, V637 gate, quarantine, tools, and outputs")
    upstream = read_object(root / UPSTREAM_RELATIVE)
    abi = read_object(root / ABI_RELATIVE)
    history = read_object(root / HISTORY_RELATIVE)
    sources = {"exp7240": upstream, "exp7217": abi, "exp7230": history}
    hashes = {
        str(path): sha256_file(root / path) for path in SOURCE_PATHS if (root / path).is_file()
    }
    sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    cl_text = (root / CL_SPEC_RELATIVE).read_text(encoding="utf-8")
    rust_text = (root / RUST_SPEC_RELATIVE).read_text(encoding="utf-8")
    expected_contract = {
        "id": TASK_ID,
        "milestone": MILESTONE,
        "deliverable": str(RESULT_RELATIVE),
        "gated_on": [
            {
                "upstream": "exp7240-recurrence-fixture",
                "artifact_field": "recurrence_fixture_ready_score",
                "op": "==",
                "value": 1,
            }
        ],
    }
    try:
        exclusion = yaml.safe_load((root / EXCLUSION_RELATIVE).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        exclusion = None
    upstream_quarantine = exp7217.upstream_quarantine_observation(
        upstream,
        manifest_match=exp7217._manifest_mentions_experiment(exclusion, "7240"),
    )
    history_quarantine = exp7217.upstream_quarantine_observation(
        history,
        manifest_match=exp7217._manifest_mentions_experiment(exclusion, "7230"),
    )
    gate = exp7217.gated_upstream_value(
        upstream, upstream_quarantine, "recurrence_fixture_ready_score"
    )
    abi_quarantine = exp7217.upstream_quarantine_observation(
        abi,
        manifest_match=exp7217._manifest_mentions_experiment(exclusion, "7217"),
    )
    try:
        upstream_checksum = exp7240.reproducibility_checksum(upstream)
    except (KeyError, TypeError, ValueError):
        upstream_checksum = None
    tools = {
        "python": str(Path(sys.executable).absolute()),
        "cargo": shutil.which("cargo"),
        "rustc": shutil.which("rustc"),
        "ldd": shutil.which("ldd"),
        "numpy": np.__version__,
    }
    outputs = {"artifact": _writable(paths.artifact), "checkpoint": _writable(paths.checkpoint)}
    checks = [
        check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            "all nonempty",
            sizes,
            all(size is not None and size > 0 for size in sizes.values()),
        ),
        check(
            "driving_capability_specs",
            f"{CL_SPEC_RELATIVE},{RUST_SPEC_RELATIVE}",
            "REQ-CL-7243,REQ-RUSTPY-7243",
            {"REQ-CL-7243": True, "REQ-RUSTPY-7243": True},
            {
                "REQ-CL-7243": "REQ-CL-7243" in cl_text,
                "REQ-RUSTPY-7243": "REQ-RUSTPY-7243" in rust_text,
            },
            "REQ-CL-7243" in cl_text and "REQ-RUSTPY-7243" in rust_text,
        ),
        check(
            "roadmap_task_contract",
            str(ROADMAP_RELATIVE),
            TASK_ID,
            expected_contract,
            _task_contract(root),
            _task_contract(root) == expected_contract,
        ),
        check(
            "imports_tools_and_outputs",
            "host",
            "python,cargo,rustc,ldd,numpy,raw/checkpoint outputs",
            "all available and writable",
            {**tools, **outputs},
            all(tools[name] for name in ("cargo", "rustc", "ldd")) and all(outputs.values()),
        ),
        check(
            "exp7240_artifact_hash",
            str(UPSTREAM_RELATIVE),
            "sha256",
            EXPECTED_EXP7240_SHA256,
            hashes.get(str(UPSTREAM_RELATIVE)),
            hashes.get(str(UPSTREAM_RELATIVE)) == EXPECTED_EXP7240_SHA256,
        ),
        check(
            "exp7240_not_quarantined",
            str(UPSTREAM_RELATIVE),
            "quarantined",
            False,
            upstream_quarantine,
            upstream_quarantine["quarantined"] is False,
        ),
        check(
            "exp7240_checksum_and_stream_hashes",
            str(UPSTREAM_RELATIVE),
            "reproducibility_checksum,stream_receipts",
            True,
            upstream_checksum == upstream.get("reproducibility_checksum")
            and _receipt_hashes_match(root, upstream),
            upstream_checksum == upstream.get("reproducibility_checksum")
            and _receipt_hashes_match(root, upstream),
        ),
        check(
            "exp7240_ready_gate",
            "exp7240-recurrence-fixture",
            "recurrence_fixture_ready_score",
            1,
            gate,
            gate == 1,
        ),
        check(
            "exp7217_interpreter_recipe",
            str(ABI_RELATIVE),
            "hash,quarantine,native_abi_ready_score",
            {"hash": EXPECTED_EXP7217_SHA256, "quarantined": False, "ready": 1},
            {
                "hash": hashes.get(str(ABI_RELATIVE)),
                "quarantined": abi_quarantine["quarantined"],
                "ready": exp7217.gated_upstream_value(
                    abi, abi_quarantine, "native_abi_ready_score"
                ),
            },
            hashes.get(str(ABI_RELATIVE)) == EXPECTED_EXP7217_SHA256
            and abi_quarantine["quarantined"] is False
            and exp7217.gated_upstream_value(abi, abi_quarantine, "native_abi_ready_score") == 1,
        ),
        check(
            "exp7230_quarantined_history",
            str(HISTORY_RELATIVE),
            "sha256,quarantined,verifier_is_oracle,verdict_class",
            {
                "hash": EXPECTED_EXP7230_SHA256,
                "quarantined": True,
                "verifier_is_oracle": True,
                "verdict_class": "positive",
            },
            {
                "hash": hashes.get(str(HISTORY_RELATIVE)),
                "quarantined": history_quarantine["quarantined"],
                "verifier_is_oracle": history.get("verifier_is_oracle"),
                "verdict_class": history.get("verdict_class"),
            },
            hashes.get(str(HISTORY_RELATIVE)) == EXPECTED_EXP7230_SHA256
            and history_quarantine["quarantined"] is True
            and history.get("verifier_is_oracle") is True
            and history.get("verdict_class") == "positive",
        ),
    ]
    progress(
        0,
        "end",
        f"precondition checks={len(checks)} failed={sum(row['passed'] is not True for row in checks)}",
    )
    return checks, hashes, sources


def build_native_extension(root: Path, *, show_progress: bool = True) -> tuple[Path, JsonDict]:
    """Build only carnot-python for this interpreter in an isolated target."""

    target = root / TARGET_RELATIVE
    environment = exp7217.interpreter_build_environment(Path(sys.executable), target)
    if show_progress:
        progress(3, "before", "interpreter-bound carnot-python build subprocess")
    receipt = exp7217._stream_process(
        ["cargo", "build", "--release", "-p", "carnot-python"],
        root=root,
        environment=environment,
        operation="Exp7243 interpreter-bound carnot-python build",
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=load_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
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


def _native_semantic_state(state: Mapping[str, Any]) -> JsonDict:
    """Project Python metadata to the decision-bearing native state schema."""

    controller = exp7226.PackedBeliefController.from_state(state)
    return exp7230.reference_semantic_state(controller)


class NativePackedActive:
    """Execute active packed operations in Rust while retaining Python metadata."""

    def __init__(
        self,
        binding: ModuleType,
        state: Mapping[str, Any],
        timing: JsonDict,
    ) -> None:
        self._binding = binding
        self._state = exp7226.PackedBeliefController.from_state(state).state_dict()
        self._timing = timing
        self._native = binding.RustPackedBeliefController()
        self._native.load_state(canonical_json(_native_semantic_state(self._state)))
        self._timing["native_call_count"] += 1

    @classmethod
    def from_masks(
        cls,
        binding: ModuleType,
        masks: Mapping[str, Any],
        timing: JsonDict,
    ) -> NativePackedActive:
        """Create a real native active state from one nominated archive."""

        survivors = {
            family: {
                parameter
                for parameter in exp7226.PARAMETER_DOMAIN
                if int(masks[family]) & (1 << parameter)
            }
            for family in exp7226.FAMILIES
        }
        return cls(
            binding, exp7226.PackedBeliefController.from_survivors(survivors).state_dict(), timing
        )

    def state_dict(self) -> JsonDict:
        """Return detached metadata after native semantic state validation."""

        native = json.loads(self._native.serialize_state())
        if native != _native_semantic_state(self._state):
            raise RuntimeError("native_shadow_state_mismatch")
        self._timing["native_call_count"] += 1
        return deepcopy(self._state)

    def state_bytes(self) -> bytes:
        """Serialize the same complete active state used by the Python arm."""

        return transactional.canonical_json_bytes(self.state_dict())

    def state_hash(self) -> str:
        """Hash complete active metadata, not only the packed kernel bytes."""

        return transactional.sha256_bytes(self.state_bytes())

    def family_state(self, family: str) -> JsonDict:
        """Expose one detached family for shared archive contradiction checks."""

        return deepcopy(self._state["families"][family])

    def _query(self, event: Mapping[str, Any], label: int) -> JsonDict | None:
        """Convert one public event and call the real native query boundary."""

        coordinates = exp7226.PackedBeliefController._public_coordinates(event)
        if coordinates is None:
            return None
        family, value = coordinates
        conversion_started = time.perf_counter_ns()
        families = np.ascontiguousarray([exp7230.FAMILY_CODES[family]], dtype=np.uint8)
        values = np.ascontiguousarray([value], dtype=np.int64)
        labels = np.ascontiguousarray([label], dtype=np.int8)
        self._timing["binding_conversion_ns"] += time.perf_counter_ns() - conversion_started
        result = dict(self._native.query_batch(families, values, labels))
        self._timing["native_kernel_ns"] += int(result["kernel_ns"])
        self._timing["native_call_count"] += 1
        return result

    def predict(self, event: Mapping[str, Any]) -> tuple[str, float]:
        """Translate one native majority result to the public decision contract."""

        result = self._query(event, 1)
        if result is None or result["decisions"][0] < 0:
            return "abstain", 0.0
        return (
            "accept" if result["decisions"][0] == 1 else "reject",
            float(result["disagreements"][0]),
        )

    def energy(self, label: str, event: Mapping[str, Any]) -> JsonDict:
        """Translate native disagreement energy without treating it as truth."""

        if label not in {"accept", "reject"}:
            return {
                "status": "unknown_label",
                "value": None,
                "survivor_count": None,
                "disagree_count": None,
            }
        coordinates = exp7226.PackedBeliefController._public_coordinates(event)
        if coordinates is None:
            return {
                "status": "unknown_input",
                "value": None,
                "survivor_count": None,
                "disagree_count": None,
            }
        result = self._query(event, int(label == "accept"))
        family, _ = coordinates
        count = int(self._state["families"][family]["survivor_mask"]).bit_count()
        if result is None or result["energy_status"][0] != 1:
            return {
                "status": "empty",
                "value": None,
                "survivor_count": 0,
                "disagree_count": None,
            }
        value = float(result["energies"][0])
        return {
            "status": "known",
            "value": value,
            "survivor_count": count,
            "disagree_count": round(value * count),
        }

    def select_request(
        self,
        block: Sequence[Mapping[str, Any]],
        tie_ranks: Mapping[str, int],
    ) -> Mapping[str, Any]:
        """Use the unchanged public selector with native predictions."""

        return exp7226.exp7199.select_request(block, "priority_admission", tie_ranks, self)

    def commit_batch(
        self,
        releases: Sequence[Mapping[str, Any]],
        *,
        current_cycle: int,
        expected_parent_hash: str,
        state_path: Path | None = None,
    ) -> JsonDict:
        """Validate in Python, mutate packed semantics in Rust, and retain metadata."""

        parent_bytes = transactional.canonical_json_bytes(self._state)
        parent_hash = transactional.sha256_bytes(parent_bytes)
        if expected_parent_hash != parent_hash:
            raise exp7226.CommitRejected("stale_parent")
        normalized = [
            exp7226.PackedBeliefController._validate_release(row, current_cycle) for row in releases
        ]
        existing = {
            str(item["event_id"])
            for family in exp7226.FAMILIES
            for item in self._state["families"][family]["provenance"]
        }
        ids = [str(row["event_id"]) for row in normalized]
        if len(set(ids)) != len(ids) or existing.intersection(ids):
            raise exp7226.CommitRejected("duplicate_release")
        before_masks = {
            family: int(self._state["families"][family]["survivor_mask"])
            for family in exp7226.FAMILIES
        }
        conversion_started = time.perf_counter_ns()
        families = np.ascontiguousarray(
            [exp7230.FAMILY_CODES[str(row["family_id"])] for row in normalized], dtype=np.uint8
        )
        values = np.ascontiguousarray([row["numeric_value"] for row in normalized], dtype=np.int64)
        labels = np.ascontiguousarray(
            [int(row["observed_label"] == "accept") for row in normalized], dtype=np.int8
        )
        roles = np.ascontiguousarray(
            [int(row["role"] == "support") for row in normalized], dtype=np.uint8
        )
        self._timing["binding_conversion_ns"] += time.perf_counter_ns() - conversion_started
        native_receipt = dict(self._native.update_batch(families, values, labels, roles))
        self._timing["native_kernel_ns"] += int(native_receipt["kernel_ns"])
        self._timing["native_call_count"] += 1
        semantic = json.loads(self._native.serialize_state())
        self._timing["native_call_count"] += 1
        candidate = deepcopy(self._state)
        candidate["version"] = int(semantic["version"])
        candidate["parent_hash"] = parent_hash
        for index, family in enumerate(exp7226.FAMILIES):
            candidate["families"][family]["survivor_mask"] = int(semantic["survivor_masks"][index])
            candidate["families"][family]["vote_counts"] = list(semantic["vote_counts"][index])
            candidate["families"][family]["epoch"] = int(semantic["epochs"][index])
        for release in normalized:
            candidate["families"][str(release["family_id"])]["provenance"].append(deepcopy(release))
        admitted = exp7226.PackedBeliefController.from_state(candidate)
        self._state = admitted.state_dict()
        new_bytes = transactional.canonical_json_bytes(self._state)
        operations = []
        for release in normalized:
            family = str(release["family_id"])
            after = int(self._state["families"][family]["survivor_mask"])
            operations.append(
                {
                    "event_id": release["event_id"],
                    "family_id": family,
                    "role": release["role"],
                    "survivor_mask_before": before_masks[family],
                    "survivor_mask_after": after,
                    "empty_reset": bool(native_receipt["reset_count"]),
                    "validation_used_for_elimination": False,
                }
            )
        if state_path is not None:
            transactional._atomic_write(state_path, new_bytes)
        return {
            "parent_hash": parent_hash,
            "new_state_hash": transactional.sha256_bytes(new_bytes),
            "parent_bytes_b64": transactional.encode_bytes(parent_bytes),
            "new_state_bytes_b64": transactional.encode_bytes(new_bytes),
            "state_version": candidate["version"],
            "release_count": len(normalized),
            "release_order": ids,
            "operations": operations,
            "atomic_write": None,
        }


PythonArchiveController = exp7240.ArchivedBeliefController


class NativeArchiveController(exp7240.ArchivedBeliefController):
    """Use the Exp7240 archive policy with a real native active-state backend."""

    def __init__(self, binding: ModuleType, *, archive_cap: int = 4) -> None:
        self._binding = binding
        self._native_timing: JsonDict = {
            "binding_conversion_ns": 0,
            "native_kernel_ns": 0,
            "native_call_count": 0,
        }
        super().__init__(archive_cap=archive_cap, nomination_mode="validated")

    @classmethod
    def from_state_with_binding(
        cls, binding: ModuleType, value: Mapping[str, Any]
    ) -> NativeArchiveController:
        """Restore checked archive bytes and retain the selected native module."""

        validated = exp7240.ArchivedBeliefController.from_state(value)
        controller = cls.__new__(cls)
        controller._binding = binding
        controller._native_timing = {
            "binding_conversion_ns": 0,
            "native_kernel_ns": 0,
            "native_call_count": 0,
        }
        controller._state = validated.state_dict()
        controller._active().state_dict()
        return controller

    @property
    def native_call_count(self) -> int:
        """Expose actual PyO3 calls for provenance and tests."""

        return int(self._native_timing["native_call_count"])

    @property
    def native_module_file(self) -> str:
        """Identify the exact extension used by this live controller."""

        return str(Path(self._binding.__file__).resolve())

    def native_timing(self) -> JsonDict:
        """Return detached conversion and kernel counters for cost rows."""

        return deepcopy(self._native_timing)

    def _active_from_state(self, value: Mapping[str, Any]) -> NativePackedActive:
        """Load one real native active state under the shared archive policy."""

        return NativePackedActive(self._binding, value, self._native_timing)

    def _active_from_masks(self, masks: Mapping[str, Any]) -> NativePackedActive:
        """Reactivate an archive as one real native packed state."""

        return NativePackedActive.from_masks(self._binding, masks, self._native_timing)

    def _validate_live_state(self, value: Mapping[str, Any]) -> None:
        """Use the canonical archive validator before any native mutation."""

        exp7240.ArchivedBeliefController.from_state(value)

    def _admit_state(self, value: Mapping[str, Any]) -> NativeArchiveController:
        """Admit the next archive state with this exact native binding."""

        return type(self).from_state_with_binding(self._binding, value)

    def _load_durable_state(self, path: Path) -> NativeArchiveController:
        """Load durable archive bytes with the same native extension."""

        return type(self).from_state_with_binding(
            self._binding, json.loads(path.read_text(encoding="utf-8"))
        )


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read authenticated chronological JSON objects from one sidecar."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _stream_sources(root: Path) -> tuple[list[JsonDict], dict[str, JsonDict], dict[str, JsonDict]]:
    """Load the exact Exp7240 public, authority, and release stream views."""

    public = _read_jsonl(root / "results/streams/experiment_7240/public_stream.jsonl")
    authority_rows = _read_jsonl(root / "results/streams/experiment_7240/private_evaluator.jsonl")
    release_rows = _read_jsonl(root / "results/streams/experiment_7240/release_schedule.jsonl")
    return (
        public,
        {str(row["event_id"]): row for row in authority_rows},
        {str(row["event_id"]): row for row in release_rows},
    )


def _continuation_fixture(state: Mapping[str, Any], seed: int) -> JsonDict:
    """Attach deterministic future probes and delayed releases to one checkpoint."""

    events = [
        {
            "event_id": f"continuation-{seed}-{index}",
            "family_id": exp7226.FAMILIES[index % len(exp7226.FAMILIES)],
            "numeric_value": (seed + index * 7) % len(exp7226.PARAMETER_DOMAIN),
        }
        for index in range(8)
    ]
    releases = [
        {
            **event,
            "observed_label": "accept" if index % 2 else "reject",
            "role": "support",
            "request_index": 2_000 + index,
            "release_index": 2_004 + index,
        }
        for index, event in enumerate(events[:4])
    ]
    return {"state": deepcopy(dict(state)), "events": events, "releases": releases}


def run_stream_parity(
    binding: ModuleType,
    *,
    stream_limit: int = len(STREAM_SEEDS),
    event_limit: int = exp7240.EVENTS_PER_STREAM,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Replay matched Exp7240 streams through Python and native active states."""

    public, authority, schedules = _stream_sources(REPO_ROOT)
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    started = time.monotonic()
    for stream_offset, seed in enumerate(STREAM_SEEDS[:stream_limit]):
        stream_id = f"stream-{stream_offset + 1:02d}"
        events = [row for row in public if row["stream_id"] == stream_id][:event_limit]
        python = PythonArchiveController(archive_cap=4, nomination_mode="validated")
        native = NativeArchiveController(binding, archive_cap=4)
        arm_values: dict[str, JsonDict] = {
            "python": {
                "predictions": [],
                "energies": [],
                "queries": [],
                "state_hashes": [],
                "archive_state_hashes": [],
            },
            "native_pyo3": {
                "predictions": [],
                "energies": [],
                "queries": [],
                "state_hashes": [],
                "archive_state_hashes": [],
            },
        }
        pending: list[JsonDict] = []
        mismatch_count = 0
        query_count = 0
        for block_index, offset in enumerate(range(0, len(events), exp7240.QUERY_BLOCK_SIZE)):
            block = events[offset : offset + exp7240.QUERY_BLOCK_SIZE]
            tie_ranks = exp7240.exp7199.seeded_tie_ranks(seed, block_index, block)
            python_selected = python.select_request(block, tie_ranks)
            native_selected = native.select_request(block, tie_ranks)
            mismatch_count += int(python_selected != native_selected)
            selected_id = str(python_selected["event_id"])
            for event in block:
                event_id = str(event["event_id"])
                chronology = int(event["chronology_index"])
                py_prediction = python.predict(event)
                native_prediction = native.predict(event)
                py_energy = [python.energy(label, event) for label in ("accept", "reject")]
                native_energy = [native.energy(label, event) for label in ("accept", "reject")]
                mismatch_count += int(py_prediction != native_prediction)
                mismatch_count += int(py_energy != native_energy)
                py_hash = python.state_hash()
                native_hash = native.state_hash()
                mismatch_count += int(py_hash != native_hash)
                for name, controller, prediction, energies in (
                    ("python", python, py_prediction, py_energy),
                    ("native_pyo3", native, native_prediction, native_energy),
                ):
                    arm_values[name]["predictions"].append(prediction)
                    arm_values[name]["energies"].append(energies)
                    arm_values[name]["state_hashes"].append(controller.state_hash())
                    arm_values[name]["archive_state_hashes"].append(
                        [row["state_hash"] for row in controller.archives()]
                    )
                will_query = (
                    event_id == selected_id
                    and query_count < exp7240.QUERY_CEILING
                    and len(pending) < exp7240.PENDING_CAPACITY
                )
                if will_query:
                    query_count += 1
                    pending.append(
                        {
                            "public": deepcopy(event),
                            "observed_label": authority[event_id]["exact_label"],
                            "request_index": chronology,
                            "release_index": chronology + int(schedules[event_id]["delay"]),
                        }
                    )
                    for values in arm_values.values():
                        values["queries"].append(event_id)
                due = sorted(
                    [row for row in pending if row["release_index"] <= chronology],
                    key=lambda row: (row["release_index"], row["request_index"]),
                )
                if due:
                    releases = [exp7240._support_release(row) for row in due]
                    python.commit_batch(
                        releases,
                        current_cycle=chronology,
                        expected_parent_hash=python.state_hash(),
                    )
                    native.commit_batch(
                        releases,
                        current_cycle=chronology,
                        expected_parent_hash=native.state_hash(),
                    )
                    mismatch_count += int(python.state_bytes() != native.state_bytes())
                    pending = [row for row in pending if row not in due]
        mismatch_count += int(python.state_bytes() != native.state_bytes())
        for name, controller in (("python", python), ("native_pyo3", native)):
            values = arm_values[name]
            rows.append(
                _finish_row(
                    {
                        "unit_id": f"parity:{stream_id}:{name}",
                        "stream_id": stream_id,
                        "implementation": name,
                        "arm": name,
                        "seed": seed,
                        "metric": mismatch_count,
                        "error": None,
                        "abstention": sum(
                            int(prediction[0] == "abstain") for prediction in values["predictions"]
                        ),
                        "event_count": len(events),
                        "predictions": values["predictions"],
                        "energies": values["energies"],
                        "queries": values["queries"],
                        "state_hashes": values["state_hashes"],
                        "archive_state_hashes": values["archive_state_hashes"],
                        "final_state_hash": controller.state_hash(),
                        "mismatch_count": mismatch_count,
                        "native_call_count": native.native_call_count,
                        "passed": mismatch_count == 0,
                    }
                )
            )
        checkpoints.append(_continuation_fixture(python.state_dict(), seed))
        print(
            f"[phase 4 progress] completed_streams={stream_offset + 1}/{stream_limit} "
            f"elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
    return rows, checkpoints


def _continue_controller(controller: Any, checkpoint: Mapping[str, Any]) -> JsonDict:
    """Continue fixed probes and delayed releases from one restored checkpoint."""

    outputs = []
    events = checkpoint["events"]
    for index, event in enumerate(events):
        tie_ranks = {str(event["event_id"]): 0}
        outputs.append(
            {
                "prediction": controller.predict(event),
                "energies": [controller.energy(label, event) for label in ("accept", "reject")],
                "query": controller.select_request([event], tie_ranks)["event_id"],
                "state_hash": controller.state_hash(),
            }
        )
        if index < len(checkpoint["releases"]):
            release = checkpoint["releases"][index]
            controller.commit_batch(
                [release],
                current_cycle=int(release["release_index"]),
                expected_parent_hash=controller.state_hash(),
            )
    return {
        "outputs": outputs,
        "final_state_hash": controller.state_hash(),
        "final_state": controller.state_dict(),
        "delayed_updates_continued": len(checkpoint["releases"]),
    }


_CONTINUATION_HELPER = r"""
import importlib.util
import json
from pathlib import Path
import sys
from carnot import experiment_7243_v637_native_memory as experiment

extension = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("carnot._rust", extension)
if spec is None or spec.loader is None:
    raise RuntimeError("native loader unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
checkpoint = json.loads(sys.stdin.read())
controller = experiment.NativeArchiveController.from_state_with_binding(
    module, checkpoint["state"]
)
result = experiment._continue_controller(controller, checkpoint)
result["module_file"] = str(Path(module.__file__).resolve())
print("__CARNOT_JSON__" + json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
"""


def run_fresh_process_continuation(extension: Path, checkpoint: Mapping[str, Any]) -> JsonDict:
    """Restore the full archive in a fresh interpreter and continue delayed updates."""

    progress(5, "before", "fresh-process native archive restore and delayed continuation")
    python = PythonArchiveController.from_state(checkpoint["state"])
    expected = json.loads(canonical_json(_continue_controller(python, checkpoint)))
    started = time.monotonic()
    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        "-c",
        _CONTINUATION_HELPER,
        str(extension),
    ]
    with exp7217._Heartbeat("Exp7243 fresh-process continuation", interval_s=30):
        completed = subprocess.run(
            command,
            input=canonical_json(checkpoint),
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO_ROOT / "python")},
        )
    observed: JsonDict = {}
    for line in completed.stdout.splitlines():
        if line.startswith("__CARNOT_JSON__"):
            value = json.loads(line.removeprefix("__CARNOT_JSON__"))
            if isinstance(value, Mapping):
                observed = dict(value)
    mismatch_count = int(
        completed.returncode != 0
        or observed.get("outputs") != expected["outputs"]
        or observed.get("final_state_hash") != expected["final_state_hash"]
        or observed.get("final_state") != expected["final_state"]
        or observed.get("module_file") != str(extension.resolve())
    )
    receipt = _finish_row(
        {
            "unit_id": "fresh-process-delayed-continuation",
            "arm": "native_pyo3_fresh_process",
            "seed": STREAM_SEEDS[-1],
            "metric": mismatch_count,
            "error": None if completed.returncode == 0 else completed.stderr,
            "abstention": False,
            "command": command,
            "exit_code": completed.returncode,
            "duration_s": time.monotonic() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "module_file": observed.get("module_file"),
            "final_state_hash": observed.get("final_state_hash"),
            "delayed_updates_continued": observed.get("delayed_updates_continued", 0),
            "mismatch_count": mismatch_count,
            "python_fallback_used": False,
            "passed": mismatch_count == 0,
        }
    )
    progress(5, "after", f"fresh-process continuation passed={receipt['passed']}")
    return receipt


def _cost_release(block: int, index: int) -> JsonDict:
    """Create one fixed released event for complete-boundary timing."""

    family = exp7226.FAMILIES[index % len(exp7226.FAMILIES)]
    return {
        "event_id": f"cost-{block}-{index}",
        "family_id": family,
        "numeric_value": (block * 5 + index * 7) % len(exp7226.PARAMETER_DOMAIN),
        "observed_label": "accept" if (block + index) % 2 else "reject",
        "role": "support",
        "request_index": 10_000 + block * 1_000 + index,
        "release_index": 10_000 + block * 1_000 + index,
    }


def _seed_cost_state(capacity: int) -> JsonDict:
    """Build the same nonempty archive and validation window before both arms."""

    controller = PythonArchiveController(archive_cap=capacity, nomination_mode="validated")
    for index in range(18):
        release = {
            "event_id": f"seed-{capacity}-{index}",
            "family_id": "lower_bound",
            "numeric_value": 0,
            "observed_label": "accept" if index % 2 == 0 else "reject",
            "role": "support",
            "request_index": index,
            "release_index": index,
        }
        controller.commit_batch(
            [release], current_cycle=index, expected_parent_hash=controller.state_hash()
        )
    return controller.state_dict()


def _measure_cost_arm(
    binding: ModuleType,
    arm: str,
    state: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Time every operation inside one complete archive-controller block."""

    controller: Any = (
        NativeArchiveController.from_state_with_binding(binding, state)
        if arm == "native_pyo3"
        else PythonArchiveController.from_state(state)
    )
    total_started = time.perf_counter_ns()
    lookup_ns = 0
    query_ns = 0
    update_ns = 0
    for release in trace:
        event = {
            "event_id": release["event_id"],
            "family_id": release["family_id"],
            "numeric_value": release["numeric_value"],
        }
        started = time.perf_counter_ns()
        controller.predict(event)
        controller.energy("accept", event)
        controller.energy("reject", event)
        lookup_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        controller.select_request([event], {str(event["event_id"]): 0})
        query_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        controller.commit_batch(
            [release],
            current_cycle=int(release["release_index"]),
            expected_parent_hash=controller.state_hash(),
        )
        update_ns += time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    serialized = controller.state_bytes()
    serialization_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    restored: Any = (
        NativeArchiveController.from_state_with_binding(binding, json.loads(serialized))
        if arm == "native_pyo3"
        else PythonArchiveController.from_state(json.loads(serialized))
    )
    restored.predict(
        {
            "event_id": trace[-1]["event_id"],
            "family_id": trace[-1]["family_id"],
            "numeric_value": trace[-1]["numeric_value"],
        }
    )
    restore_ns = time.perf_counter_ns() - started
    total = time.perf_counter_ns() - total_started
    native_timing = (
        controller.native_timing() if isinstance(controller, NativeArchiveController) else {}
    )
    accounted = lookup_ns + query_ns + update_ns + serialization_ns + restore_ns
    components = {
        "python_dispatch_ns": max(total - accounted, 0),
        "binding_conversion_ns": int(native_timing.get("binding_conversion_ns", 0)),
        "lookup_ns": lookup_ns,
        "query_ns": query_ns,
        "validation_archive_update_ns": update_ns,
        "serialization_ns": serialization_ns,
        "restore_ns": restore_ns,
        "total_event_ns": total / len(trace),
    }
    return {
        "total_block_ns": total,
        "total_event_ns": total / len(trace),
        "component_ns": components,
        "native_kernel_ns": int(native_timing.get("native_kernel_ns", 0)),
        "native_call_count": int(native_timing.get("native_call_count", 0)),
        "serialized_bytes": len(serialized),
        "final_state_hash": restored.state_hash(),
    }


def run_cost_benchmark(
    binding: ModuleType,
    *,
    capacities: Sequence[int] = ARCHIVE_CAPACITIES,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    blocks: int = PAIRED_BLOCKS,
    seed: int = RANDOM_SEED,
) -> list[JsonDict]:
    """Run fixed interleaved paired blocks after a separate warmup."""

    if not capacities or not batch_sizes or blocks <= 0:
        raise ValueError("cost roster must be nonempty")
    states = {capacity: _seed_cost_state(capacity) for capacity in capacities}
    traces = {
        (capacity, batch, block): [_cost_release(block, index) for index in range(batch)]
        for capacity in capacities
        for batch in batch_sizes
        for block in range(blocks)
    }
    progress(6, "before", "separate complete-boundary warmup")
    for capacity in capacities:
        for batch in batch_sizes:
            trace = traces[(capacity, batch, 0)]
            for arm in ("python_reference", "native_pyo3"):
                _measure_cost_arm(binding, arm, states[capacity], trace)
    progress(6, "after", "separate complete-boundary warmup")
    rng = random.Random(seed)
    orders = {}
    for capacity in capacities:
        for batch in batch_sizes:
            for block in range(blocks):
                order = ["python_reference", "native_pyo3"]
                rng.shuffle(order)
                orders[(capacity, batch, block)] = order
    rows: list[JsonDict] = []
    started = time.monotonic()
    progress(6, "before", "interleaved paired archive cost blocks")
    for capacity in capacities:
        for batch in batch_sizes:
            for block in range(blocks):
                trace = traces[(capacity, batch, block)]
                trace_hash = "sha256:" + hashlib.sha256(canonical_json(trace).encode()).hexdigest()
                for order_index, arm in enumerate(orders[(capacity, batch, block)]):
                    measured = _measure_cost_arm(binding, arm, states[capacity], trace)
                    rows.append(
                        _finish_row(
                            {
                                "unit_id": f"cost:{capacity}:{batch}:{block}:{arm}",
                                "arm": arm,
                                "seed": seed,
                                "metric": measured["total_event_ns"],
                                "error": None,
                                "abstention": False,
                                "archive_capacity": capacity,
                                "batch_size": batch,
                                "block": block,
                                "arm_order": order_index,
                                "trace_sha256": trace_hash,
                                **measured,
                            }
                        )
                    )
                if time.monotonic() - started >= 60:
                    print(
                        f"[phase 6 progress] completed_blocks={block + 1}/{blocks} "
                        f"capacity={capacity} batch={batch} elapsed_s={time.monotonic() - started:.3f}",
                        flush=True,
                    )
                    started = time.monotonic()
    progress(6, "after", f"paired cost rows={len(rows)}")
    return rows


def _bootstrap_lower(values: Sequence[float], seed: int, draws: int = 10_000) -> float:
    """Return the fixed paired bootstrap lower percentile for a mean ratio."""

    if not values:
        raise ValueError("paired ratios are empty")
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws)]
    means.sort()
    return means[int(0.025 * draws)]


def summarize_cost(
    rows: Sequence[Mapping[str, Any]],
    *,
    capacities: Sequence[int] = ARCHIVE_CAPACITIES,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    blocks: int = PAIRED_BLOCKS,
) -> JsonDict:
    """Derive paired cell intervals and batch-one 1x, 10x, and 100x gates."""

    cells = []
    for capacity in capacities:
        for batch in batch_sizes:
            ratios = []
            for block in range(blocks):
                pair = {
                    str(row["arm"]): float(row["total_event_ns"])
                    for row in rows
                    if row.get("archive_capacity") == capacity
                    and row.get("batch_size") == batch
                    and row.get("block") == block
                }
                if set(pair) != {"python_reference", "native_pyo3"}:
                    raise ValueError(f"incomplete paired cost cell:{capacity}:{batch}:{block}")
                ratios.append(pair["python_reference"] / pair["native_pyo3"])
            lower = _bootstrap_lower(ratios, RANDOM_SEED + capacity * 1_000 + batch)
            cells.append(
                {
                    "archive_capacity": capacity,
                    "batch_size": batch,
                    "paired_blocks": blocks,
                    "speedup_python_over_native": sum(ratios) / len(ratios),
                    "paired_lower_ci95": lower,
                    "paired_ratios": ratios,
                }
            )
    batch_one_lower = min(cell["paired_lower_ci95"] for cell in cells if cell["batch_size"] == 1)
    return {
        "cells": cells,
        "batch_one_lower_ci95": batch_one_lower,
        "native_archive_cost_value_score": int(batch_one_lower > 1.0),
        "nfr_01_10x_met": batch_one_lower >= 10.0,
        "research_program_100x_met": batch_one_lower >= 100.0,
    }


def synthetic_cost_rows(
    *,
    capacities: Sequence[int],
    batch_sizes: Sequence[int],
    blocks: int,
    python_ns: int,
    native_ns: int,
) -> list[JsonDict]:
    """Build deterministic complete-shape rows for gate and validator tests."""

    rows = []
    components = {
        "python_dispatch_ns": 0,
        "binding_conversion_ns": 0,
        "lookup_ns": 1,
        "query_ns": 1,
        "validation_archive_update_ns": 1,
        "serialization_ns": 1,
        "restore_ns": 1,
    }
    for capacity in capacities:
        for batch in batch_sizes:
            for block in range(blocks):
                for order, (arm, duration) in enumerate(
                    (("python_reference", python_ns), ("native_pyo3", native_ns))
                ):
                    rows.append(
                        _finish_row(
                            {
                                "unit_id": f"fixture:{capacity}:{batch}:{block}:{arm}",
                                "arm": arm,
                                "seed": RANDOM_SEED,
                                "metric": duration,
                                "error": None,
                                "abstention": False,
                                "archive_capacity": capacity,
                                "batch_size": batch,
                                "block": block,
                                "arm_order": order,
                                "trace_sha256": "sha256:fixture",
                                "total_block_ns": duration * batch,
                                "total_event_ns": duration,
                                "component_ns": dict(components),
                                "native_kernel_ns": duration if arm == "native_pyo3" else 0,
                                "native_call_count": 1 if arm == "native_pyo3" else 0,
                                "serialized_bytes": 1,
                                "final_state_hash": "sha256:fixture",
                            }
                        )
                    )
    return rows


def _native_identity(
    root: Path,
    extension: Path,
    binding: ModuleType,
    build_receipt: Mapping[str, Any],
    *,
    build_duration_s: float,
    import_duration_s: float,
) -> JsonDict:
    """Bind the executing interpreter and exact loaded native binary."""

    controller = binding.RustPackedBeliefController()
    return {
        "interpreter": str(Path(sys.executable).absolute()),
        "interpreter_resolved": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "extension_origin": "task_specific_interpreter_bound_carnot_python_build",
        "module_file": str(Path(binding.__file__).resolve()),
        "binary_sha256": sha256_file(extension),
        "build_flags": {
            "command": build_receipt.get("command"),
            "PYO3_PYTHON": build_receipt.get("PYO3_PYTHON"),
            "CARGO_TARGET_DIR": build_receipt.get("CARGO_TARGET_DIR"),
            "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": build_receipt.get(
                "PYO3_USE_ABI3_FORWARD_COMPATIBILITY"
            ),
        },
        "linkage": exp7217._linkage_receipt(extension),
        "native_class": "carnot._rust.RustPackedBeliefController",
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
        "compiled_execution": True,
        "python_fallback_used": False,
        "cold_build_duration_s": build_duration_s,
        "cold_import_duration_s": import_duration_s,
    }


def _sample_budget(complete: bool) -> JsonDict:
    """Declare fixed units, censoring, and the no-extension stopping rule."""

    parity = len(STREAM_SEEDS) * 2 + 1
    costs = len(ARCHIVE_CAPACITIES) * len(BATCH_SIZES) * PAIRED_BLOCKS * 2
    return {
        "independent_stream_units_planned": len(STREAM_SEEDS),
        "independent_stream_units_attempted": len(STREAM_SEEDS) if complete else 0,
        "independent_stream_units_completed": len(STREAM_SEEDS) if complete else 0,
        "independent_stream_units_censored": 0 if complete else len(STREAM_SEEDS),
        "parity_rows_planned": parity,
        "parity_rows_completed": parity if complete else 0,
        "cost_rows_planned": costs,
        "cost_rows_completed": costs if complete else 0,
        "paired_blocks_per_cell": PAIRED_BLOCKS,
        "stopping_rule": "all 32 streams and all 30 paired blocks once; no outcome extension",
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
    paths: ExperimentPaths,
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create every required field before terminal classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "phase_spans_s": {},
        "MODEL_SPECS": [],
        "model_invoked": False,
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "current_inference_count": 0,
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(False),
        "random_seed": {
            "global": RANDOM_SEED,
            "streams": list(STREAM_SEEDS),
            "arm_order": RANDOM_SEED,
            "bootstrap": RANDOM_SEED,
        },
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external:unknown_precondition",
        "acceptance_gate_results": {},
        "native_archive_ready_score": 0,
        "native_archive_cost_value_score": 0,
        "parity_rows": [],
        "cost_rows": [],
        "cost_summary": {
            "cells": [],
            "batch_one_lower_ci95": None,
            "native_archive_cost_value_score": 0,
            "nfr_01_10x_met": False,
            "research_program_100x_met": False,
        },
        "native_binary_receipt": {},
        "hardware_target_gaps": {
            "nfr_01_10x": {"target": 10.0, "measured_lower_bound": None, "met": False},
            "research_program_100x": {
                "target": 100.0,
                "measured_lower_bound": None,
                "met": False,
            },
            "controller_state_bytes": None,
            "host_transfer_requirement": "one complete serialized controller state per restore",
        },
        "checkpoint_receipt": {"path": str(paths.checkpoint), "sha256": None, "bytes": 0},
        "cold_setup_and_amortization": {},
        "historical_source_receipt": {
            "path": str(HISTORY_RELATIVE),
            "sha256": hashes.get(str(HISTORY_RELATIVE)),
            "quarantined": True,
            "promoted": False,
        },
        "default_pipeline_modified": False,
        "publication_performed": False,
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


def _fixture_parity_rows() -> list[JsonDict]:
    """Build all planned zero-mismatch rows for closed validator tests."""

    rows = []
    for index, seed in enumerate(STREAM_SEEDS):
        for arm in ("python", "native_pyo3"):
            rows.append(
                _finish_row(
                    {
                        "unit_id": f"parity:stream-{index + 1:02d}:{arm}",
                        "stream_id": f"stream-{index + 1:02d}",
                        "implementation": arm,
                        "arm": arm,
                        "seed": seed,
                        "metric": 0,
                        "error": None,
                        "abstention": 0,
                        "event_count": exp7240.EVENTS_PER_STREAM,
                        "predictions": [["accept", 0.0]],
                        "energies": [[{"status": "known", "value": 0.0}]],
                        "queries": ["fixture"],
                        "state_hashes": ["sha256:fixture"],
                        "archive_state_hashes": [[]],
                        "final_state_hash": "sha256:fixture",
                        "mismatch_count": 0,
                        "native_call_count": int(arm == "native_pyo3"),
                        "passed": True,
                    }
                )
            )
    rows.append(
        _finish_row(
            {
                "unit_id": "fresh-process-delayed-continuation",
                "arm": "native_pyo3_fresh_process",
                "seed": STREAM_SEEDS[-1],
                "metric": 0,
                "error": None,
                "abstention": False,
                "module_file": f"/tmp/_rust{sysconfig.get_config_var('EXT_SUFFIX')}",
                "delayed_updates_continued": 4,
                "mismatch_count": 0,
                "python_fallback_used": False,
                "passed": True,
            }
        )
    )
    return rows


def complete_artifact_fixture_for_test(*, cost_pass: bool) -> JsonDict:
    """Create one structurally complete circular-positive or null artifact."""

    now = datetime.now(UTC).isoformat()
    checks = [check("fixture", "fixture", "value", 1, 1, True)]
    artifact = _base_artifact(
        checks, {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=1.0
    )
    parity = _fixture_parity_rows()
    costs = synthetic_cost_rows(
        capacities=ARCHIVE_CAPACITIES,
        batch_sizes=BATCH_SIZES,
        blocks=PAIRED_BLOCKS,
        python_ns=200 if cost_pass else 10,
        native_ns=1 if cost_pass else 20,
    )
    summary = summarize_cost(costs)
    value = summary["native_archive_cost_value_score"]
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": [*parity, *costs],
            "sample_size_budget": _sample_budget(True),
            "verdict_class": "circular_positive" if value else "null",
            "honest_verdict": (
                "complete_circular_positive: exact oracle parity and batch-one cost gate passed"
                if value
                else "complete_null: exact oracle parity passed but batch-one cost gate failed"
            ),
            "acceptance_gate_results": {
                "native_archive_ready": {"expected": 1, "actual": 1, "pass": True},
                "batch_one_speedup": {
                    "expected": ">1",
                    "actual": summary["batch_one_lower_ci95"],
                    "pass": bool(value),
                },
            },
            "native_archive_ready_score": 1,
            "native_archive_cost_value_score": value,
            "parity_rows": parity,
            "cost_rows": costs,
            "cost_summary": summary,
            "native_binary_receipt": {
                "module_file": f"/tmp/_rust{sysconfig.get_config_var('EXT_SUFFIX')}",
                "binary_sha256": "sha256:fixture",
                "native_class": "carnot._rust.RustPackedBeliefController",
                "compiled_execution": True,
                "python_fallback_used": False,
            },
            "hardware_target_gaps": {
                "nfr_01_10x": {
                    "target": 10.0,
                    "measured_lower_bound": summary["batch_one_lower_ci95"],
                    "met": summary["nfr_01_10x_met"],
                },
                "research_program_100x": {
                    "target": 100.0,
                    "measured_lower_bound": summary["batch_one_lower_ci95"],
                    "met": summary["research_program_100x_met"],
                },
                "controller_state_bytes": 1,
                "host_transfer_requirement": "one complete serialized controller state per restore",
            },
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _row_hash_valid(row: Mapping[str, Any]) -> bool:
    """Recompute a row hash without trusting its stored digest."""

    material = dict(row)
    stored = material.pop("row_sha256", None)
    expected = "sha256:" + hashlib.sha256(canonical_json(material).encode()).hexdigest()
    return stored == expected


def validate_artifact(
    artifact: Mapping[str, Any], *, check_files: bool = False, root: Path = REPO_ROOT
) -> list[str]:
    """Cold-check identity, provenance, parity, paired costs, gates, and hashes."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_FIELDS.issubset(artifact), "missing_fields")
    if not REQUIRED_FIELDS.issubset(artifact):
        return errors
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID,
        "identity",
    )
    add(artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(not artifact.get("execution_host"), "execution_host")
    add(
        artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False,
        "model_invocation",
    )
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    try:
        checksum_valid = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_valid = False
    add(not checksum_valid, "reproducibility_checksum")
    if artifact.get("status") == "blocked":
        gate = artifact.get("gate_check_summary", {})
        add(artifact.get("verdict_class") != "blocked", "blocked_class")
        add(artifact.get("inference_substrate") != "blocked_no_run", "blocked_substrate")
        add(
            artifact.get("inference_substrate_class") != "blocked_no_run", "blocked_substrate_class"
        )
        add(
            any(artifact.get(name) for name in ("rows", "parity_rows", "cost_rows")), "blocked_rows"
        )
        add(artifact.get("native_archive_ready_score") != 0, "blocked_ready")
        add(not isinstance(gate, Mapping) or gate.get("passed") is not False, "blocked_gate")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked_"), "blocked_verdict")
        return errors
    parity = artifact.get("parity_rows", [])
    costs = artifact.get("cost_rows", [])
    add(artifact.get("status") != "complete", "status")
    add(artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate")
    add(
        artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate_class",
    )
    add(artifact.get("gate_check_summary", {}).get("passed") is not True, "preconditions")
    add(
        not isinstance(parity, list)
        or len(parity) != len(STREAM_SEEDS) * 2 + 1
        or any(row.get("passed") is not True or row.get("mismatch_count") != 0 for row in parity),
        "parity_rows",
    )
    add(
        not isinstance(costs, list)
        or len(costs) != len(ARCHIVE_CAPACITIES) * len(BATCH_SIZES) * PAIRED_BLOCKS * 2,
        "cost_rows",
    )
    all_rows = artifact.get("rows", [])
    add(
        not isinstance(all_rows, list)
        or len(all_rows) != len(parity) + len(costs)
        or any(not isinstance(row, Mapping) or not _row_hash_valid(row) for row in all_rows),
        "rows",
    )
    identity = artifact.get("native_binary_receipt", {})
    add(
        not isinstance(identity, Mapping)
        or identity.get("compiled_execution") is not True
        or identity.get("python_fallback_used") is not False
        or identity.get("native_class") != "carnot._rust.RustPackedBeliefController"
        or not str(identity.get("module_file", "")).endswith(
            str(sysconfig.get_config_var("EXT_SUFFIX"))
        )
        or not str(identity.get("binary_sha256", "")).startswith("sha256:"),
        "native_binary_receipt",
    )
    ready = int(
        isinstance(parity, list)
        and len(parity) == len(STREAM_SEEDS) * 2 + 1
        and all(row.get("passed") is True and row.get("mismatch_count") == 0 for row in parity)
        and isinstance(identity, Mapping)
        and identity.get("compiled_execution") is True
        and identity.get("python_fallback_used") is False
    )
    add(artifact.get("native_archive_ready_score") != ready, "ready_score")
    try:
        summary = summarize_cost(costs)
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        summary = None
    add(summary is None or artifact.get("cost_summary") != summary, "cost_summary")
    expected_value = (
        0 if summary is None else int(ready == 1 and summary["batch_one_lower_ci95"] > 1)
    )
    add(artifact.get("native_archive_cost_value_score") != expected_value, "cost_gate")
    expected_class = "circular_positive" if expected_value == 1 else "null"
    add(artifact.get("verdict_class") != expected_class, "verdict_class")
    add(not str(artifact.get("honest_verdict", "")).startswith("complete"), "honest_verdict")
    budget = artifact.get("sample_size_budget", {})
    add(
        budget.get("independent_stream_units_completed") != len(STREAM_SEEDS)
        or budget.get("cost_rows_completed")
        != len(ARCHIVE_CAPACITIES) * len(BATCH_SIZES) * PAIRED_BLOCKS * 2,
        "sample_size_budget",
    )
    if check_files:
        for path_text, expected_hash in artifact.get("source_artifact_hashes", {}).items():
            path = Path(path_text)
            resolved = path if path.is_absolute() else root / path
            add(
                not resolved.is_file() or sha256_file(resolved) != expected_hash,
                "source_artifact_hashes",
            )
        module_file = Path(str(identity.get("module_file", "")))
        add(
            not module_file.is_file() or sha256_file(module_file) != identity.get("binary_sha256"),
            "native_binary_hash",
        )
    return errors


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


def _amortization(
    costs: Sequence[Mapping[str, Any]], native_receipt: Mapping[str, Any]
) -> JsonDict:
    """Keep cold setup separate and report measured break-even without a claim."""

    python_values = [
        float(row["total_event_ns"])
        for row in costs
        if row["batch_size"] == 1 and row["arm"] == "python_reference"
    ]
    native_values = [
        float(row["total_event_ns"])
        for row in costs
        if row["batch_size"] == 1 and row["arm"] == "native_pyo3"
    ]
    saving = sum(python_values) / len(python_values) - sum(native_values) / len(native_values)
    cold_ns = (
        float(native_receipt["cold_build_duration_s"])
        + float(native_receipt["cold_import_duration_s"])
    ) * 1_000_000_000
    return {
        "cold_build_duration_s": native_receipt["cold_build_duration_s"],
        "cold_import_duration_s": native_receipt["cold_import_duration_s"],
        "mean_batch_one_event_saving_ns": saving,
        "break_even_events": None if saving <= 0 else cold_ns / saving,
        "cold_cost_in_timed_rows": False,
    }


def build_artifact(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Run authenticated parity, restart, cost measurement, and cold validation."""

    monotonic_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    phase_started = time.monotonic()
    checks, hashes, sources = collect_preconditions(root, paths)
    spans["preconditions"] = time.monotonic() - phase_started
    if gate_summary(checks)["passed"] is not True:
        artifact = _base_artifact(
            checks,
            hashes,
            paths,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - monotonic_start,
        )
        artifact["honest_verdict"] = "blocked_external:" + str(
            artifact["gate_check_summary"]["failed_check"]
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    progress(1, "start", "flushed boundaries and native subprocess heartbeat active")
    progress(1, "end", "progress contract active")
    progress(2, "start", "MODEL_SPECS empty; no model load or generation")
    progress(2, "end", "current invocation counters remain zero")
    progress(3, "start", "build and import exact interpreter-bound native extension")
    phase_started = time.monotonic()
    build_started = time.monotonic()
    extension, build_receipt = build_native_extension(root)
    build_duration = time.monotonic() - build_started
    progress(3, "before", "load exact compiled extension")
    import_started = time.monotonic()
    binding = exp7230.load_native_extension(extension)
    import_duration = time.monotonic() - import_started
    progress(3, "after", f"loaded module_file={Path(binding.__file__).resolve()}")
    native_receipt = _native_identity(
        root,
        extension,
        binding,
        build_receipt,
        build_duration_s=build_duration,
        import_duration_s=import_duration,
    )
    hashes[str(extension.resolve())] = sha256_file(extension)
    spans["native_build_and_import"] = time.monotonic() - phase_started
    progress(3, "end", "native binary receipt complete")

    progress(4, "before", "32-stream Python/native archive parity")
    phase_started = time.monotonic()
    parity, checkpoints = run_stream_parity(binding)
    spans["stream_parity"] = time.monotonic() - phase_started
    progress(4, "after", f"stream parity rows={len(parity)}")
    fresh = run_fresh_process_continuation(extension, checkpoints[-1])
    parity.append(fresh)
    ready = int(
        len(parity) == len(STREAM_SEEDS) * 2 + 1
        and all(row["passed"] is True and row["mismatch_count"] == 0 for row in parity)
        and native_receipt["compiled_execution"] is True
    )
    if ready != 1:
        raise RuntimeError("owned native archive parity or fresh continuation failed")

    progress(6, "start", "30 paired blocks across all capacity and batch cells")
    phase_started = time.monotonic()
    costs = run_cost_benchmark(binding)
    spans["paired_cost_benchmark"] = time.monotonic() - phase_started
    summary = summarize_cost(costs)
    value = int(ready == 1 and summary["native_archive_cost_value_score"] == 1)
    progress(
        6,
        "end",
        f"cost rows={len(costs)} batch_one_lower_ci95={summary['batch_one_lower_ci95']:.6f}",
    )

    checkpoint_payload = {
        "schema": "carnot.exp7243.checkpoint.v1",
        "selected_extension": str(extension.resolve()),
        "selected_extension_sha256": sha256_file(extension),
        "parity_rows": parity,
        "cost_rows": costs,
        "continuation_checkpoint": checkpoints[-1],
    }
    checkpoint_receipt = atomic_write(paths.checkpoint, checkpoint_payload)
    hashes[str(paths.checkpoint.resolve())] = checkpoint_receipt["sha256"]
    controller_bytes = max(
        len(canonical_json(checkpoint["state"]).encode()) for checkpoint in checkpoints
    )
    artifact = _base_artifact(
        checks,
        hashes,
        paths,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - monotonic_start,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "rows": [*parity, *costs],
            "sample_size_budget": _sample_budget(True),
            "verdict_class": "circular_positive" if value else "null",
            "honest_verdict": (
                "complete_circular_positive: exact oracle parity and batch-one full-event cost gate passed"
                if value
                else "complete_null: exact oracle parity passed but batch-one full-event cost gate failed"
            ),
            "acceptance_gate_results": {
                "native_execution_and_exact_fresh_parity": {
                    "expected": 1,
                    "actual": ready,
                    "pass": ready == 1,
                },
                "batch_one_total_event_lower_ci95": {
                    "expected": ">1",
                    "actual": summary["batch_one_lower_ci95"],
                    "pass": summary["batch_one_lower_ci95"] > 1,
                },
                "nfr_01_10x": {
                    "expected": ">=10",
                    "actual": summary["batch_one_lower_ci95"],
                    "pass": summary["nfr_01_10x_met"],
                },
                "research_program_100x": {
                    "expected": ">=100",
                    "actual": summary["batch_one_lower_ci95"],
                    "pass": summary["research_program_100x_met"],
                },
            },
            "native_archive_ready_score": ready,
            "native_archive_cost_value_score": value,
            "parity_rows": parity,
            "cost_rows": costs,
            "cost_summary": summary,
            "native_binary_receipt": native_receipt,
            "hardware_target_gaps": {
                "nfr_01_10x": {
                    "target": 10.0,
                    "measured_lower_bound": summary["batch_one_lower_ci95"],
                    "met": summary["nfr_01_10x_met"],
                },
                "research_program_100x": {
                    "target": 100.0,
                    "measured_lower_bound": summary["batch_one_lower_ci95"],
                    "met": summary["research_program_100x_met"],
                },
                "controller_state_bytes": controller_bytes,
                "packed_active_masks_bytes": len(exp7226.FAMILIES) * 8,
                "archive_mask_bytes_at_capacity_four": len(exp7226.FAMILIES) * 8 * 4,
                "host_transfer_requirement": "one complete serialized controller state per restore",
                "board_execution_measured": False,
            },
            "checkpoint_receipt": checkpoint_receipt,
            "cold_setup_and_amortization": _amortization(costs, native_receipt),
            "historical_source_receipt": {
                "path": str(HISTORY_RELATIVE),
                "sha256": hashes[str(HISTORY_RELATIVE)],
                "quarantined": True,
                "flagged_adversarial": sources["exp7230"].get("flagged_adversarial"),
                "verifier_is_oracle": sources["exp7230"].get("verifier_is_oracle"),
                "verdict_class": sources["exp7230"].get("verdict_class"),
                "native_cost_value_score": sources["exp7230"].get("native_cost_value_score"),
                "promoted": False,
            },
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path, output: Path, run_date: str) -> JsonDict:
    """Build, cold-validate, and atomically write the terminal artifact."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    paths = ExperimentPaths(root / CHECKPOINT_RELATIVE, output)
    artifact = build_artifact(root, paths)
    progress(7, "before", "final artifact validation")
    errors = validate_artifact(
        artifact,
        check_files=artifact.get("status") == "complete",
        root=root,
    )
    progress(7, "after", f"final artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7243 artifact:{errors}")
    progress(8, "before", "atomic terminal write")
    receipt = atomic_write(output, artifact)
    progress(8, "after", f"atomic terminal write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed run date and optional read-only validation path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded study or validate existing bytes without mutation."""

    args = _parse_args(argv)
    if args.validate is not None:
        progress(7, "before", f"read-only validation path={args.validate}")
        try:
            artifact = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(artifact)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"validation_error: {error}", flush=True)
            progress(7, "after", "read-only validation errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        progress(7, "after", f"read-only validation errors={len(errors)}")
        return 0 if not errors else 2
    if args.date != RUN_DATE:
        print(f"experiment_error: run date must be {RUN_DATE}", flush=True)
        return 2
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        run_experiment(REPO_ROOT, output, args.date)
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(f"experiment_error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
