"""Establish interpreter-bound PyO3 readiness and preserve board receipts.

This task diagnoses one native deployment boundary. It replays a small shipped
sampler fixture and carries forward read-only board evidence. It does not rerun
the prior throughput sweep or claim device performance.

Spec: REQ-ISING-7217 and SCENARIO-ISING-7217-ABI.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import tomllib
from typing import Any, Mapping, Sequence

import yaml

from carnot import experiment_7187_v633_slice_sampler as slices
from carnot import experiment_7189_v633_rust_slice_parity as exp7189
from carnot.experiment_7201_v634_slice_pyo3 import validate_artifact as validate_exp7201
from carnot.experiment_7203_v634_hardware_correction import (
    validate_artifact as validate_exp7203,
)


JsonDict = dict[str, Any]

RUN_DATE = "20260911"
TASK_ID = "exp7217-abi-board-readiness"
MILESTONE = "2026.09.635"
RESULT_PATH = Path("results/experiment_7217_v635_abi_board_readiness.json")
CHECKPOINT_DIR = Path("results/checkpoints")
SPEC_PATH = Path("openspec/capabilities/ising-backend/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
UPSTREAM_NATIVE_PATH = Path("results/experiment_7201_v634_slice_pyo3.json")
UPSTREAM_NULL_PATH = Path("results/experiment_7202_v634_slice_cost_quality.json")
UPSTREAM_BOARD_PATH = Path("results/experiment_7203_v634_hardware_correction.json")
TASK_TARGET_DIR = Path("target/experiment-7217-interpreter-bound")
TASK_LOAD_DIR = Path("target/experiment-7217-load")
REPLAY_SEED = 7217001
REPLAY_STEPS = 8
NATIVE_TIMEOUT_S = 60
BUILD_TIMEOUT_S = 900
TOLERANCE = 1.0e-12
MODEL_SPECS: list[JsonDict] = []

EXPECTED_TASK_CONTRACT: JsonDict = {
    "id": TASK_ID,
    "title": "Interpreter-bound PyO3 recovery and attached-board continuity",
    "milestone": MILESTONE,
    "deliverable": str(RESULT_PATH),
    "gated_on": None,
    "prior_failures": [
        {
            "experiment_id": "exp7202-slice-cost-quality",
            "verdict": (
                "complete: all fixed boundary, law, control, and long-chain quality rows were "
                "measured. Sample-quality evidence was insufficient. The primary local boundary "
                "gate did not pass. The unchanged NFR-01 10x target was not met."
            ),
            "addressed_by": (
                "The subsequent reproduction failed native import with "
                "Py_GetConstantBorrowed; rebuild for the actual interpreter and test correctness "
                "only, without reopening the failed 10x claim."
            ),
            "retire_if_same_verdict": True,
        },
        {
            "experiment_id": "exp7146-gatemate-changed-state-continuity",
            "verdict": "blocked_no_new_operator_physical_state_receipt_after_exp6559",
            "addressed_by": (
                "Carry the current read-only physical-state disposition; no unchanged JTAG "
                "attempt. New host work diagnoses the independently observed ABI failure."
            ),
            "retire_if_same_verdict": True,
        },
    ],
    "operator_override": (
        "2026-05-29 operator directive (standing): active hardware continuity versus exp7146; "
        "retain all board dispositions without retrying unchanged physical state. The "
        "2026-09-11 known-issues entry supplies the new host ABI diagnosis."
    ),
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ROADMAP_PATH,
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/known-issues.md"),
    Path("ops/hardware-bringup-prep.md"),
    Path("research-hardware-wishlist.md"),
    UPSTREAM_NATIVE_PATH,
    UPSTREAM_NULL_PATH,
    UPSTREAM_BOARD_PATH,
    Path("results/experiment_7190_v633_board_placement_receipt.json"),
    Path("results/experiment_5861_attached_board_state_receipts.json"),
    Path("results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"),
    Path("results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json"),
    Path("results/experiment_3867_polarfire_soc_smoke_v4.json"),
    Path("results/experiment_6559_gatemate_changed_state_continuity.json"),
    Path("results/experiment_7146_v627_gatemate_changed_state.json"),
    Path("python/carnot/experiment_7187_v633_slice_sampler.py"),
    Path("python/carnot/experiment_7189_v633_rust_slice_parity.py"),
    Path("python/carnot/experiment_7201_v634_slice_pyo3.py"),
    Path("python/carnot/experiment_7203_v634_hardware_correction.py"),
    Path("python/carnot/experiment_7217_v635_abi_board_readiness.py"),
    Path("crates/carnot-python/Cargo.toml"),
    Path("crates/carnot-python/src/fixed_cardinality.rs"),
    Path("crates/carnot-python/src/lib.rs"),
    Path("crates/carnot-samplers/src/fixed_cardinality.rs"),
    SPEC_PATH,
    Path("scripts/experiments/experiment_7217_v635_abi_board_readiness.py"),
    Path("tests/python/test_experiment_7217_v635_abi_board_readiness.py"),
)

FIELD_PRINCIPLES = {
    "field_principles": "Echo the reason for each field beside its actual evidence.",
    "status": "Write a terminal artifact only after completion or a diagnosed external block.",
    "run_date": "Use 20260911; do not substitute an upstream experiment date.",
    "preconditions_checked": "Record the actual resource, code and gate observations.",
    "inference_substrate": "Describe executed computation, not the intended workload.",
    "inference_substrate_class": "The actual operation determines its duration floor.",
    "execution_venue": (
        "Use exactly host, kv260, gatemate or polarfire; these tasks execute on host."
    ),
    "execution_host": "Put the actual hostname here, never inside execution_venue.",
    "duration_s": "Measure monotonic work time; do not pad it to pass a floor.",
    "source_artifact_hashes": "Bind source code, input data and frozen contracts to the claim.",
    "rows": (
        "Keep unit_id, arm, seed, metric, error and abstention for each comparison; do not "
        "replace numeric rows with a task roster."
    ),
    "sample_size_budget": (
        "Retain planned, attempted, completed, censored and independent-unit counts."
    ),
    "random_seed": "Freeze stochastic choices before held-out outcomes are read.",
    "reproducibility_checksum": "Hash the inputs, code, settings and raw rows.",
    "gate_check_summary": (
        "Every blocked_* verdict names failed check, upstream, field, expected and observed value."
    ),
    "verifier_is_oracle": (
        "Same correctness authority remains circular even with a separate implementation."
    ),
    "verdict_class": (
        "Use exactly positive | circular_positive | null | blocked | disqualified | partial; "
        "only incomplete own work can be partial."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings; blocked_* for external absence. "
        "Readiness is not scientific value."
    ),
    "native_abi_ready_score": (
        "One requires importing and executing the actual native binary in the chosen interpreter."
    ),
    "abi_board_receipt_complete_score": (
        "The complete receipt preserves independent host and board states."
    ),
    "abi_rows": "Interpreter, extension, linker and build settings diagnose reproducibility.",
    "e2e_receipts": "Fresh-process native execution and serialization must run.",
    "board_rows": "Each board has a dated source, terminal criterion and exact next prerequisite.",
    "operator_state_receipt": "Physical-state claims require operator-authored evidence.",
    "hardware_operations_issued": "Empty because this task performs no board mutation or probe.",
    "topology_fit": "Unknown without an explicit mapping to the published parent graph.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is only cited.",
}

REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES) | {
    "task_id",
    "milestone",
    "native_execution_receipt",
    "upstream_failed_value",
    "operation_map",
    "device_latency",
    "device_power",
    "hardware_performance_claimed",
    "throughput_sweep_rerun",
    "spec_refs",
}

_QUARANTINE_FIELDS = (
    "flagged_adversarial",
    "quarantined",
    "quarantine",
    "quarantine_flags",
    "disqualified",
    "invalidated",
)

_NATIVE_HELPER = r"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

extension = Path(sys.argv[1]).resolve()
action = sys.argv[2]
spec = importlib.util.spec_from_file_location("carnot._rust", extension)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot create native loader for {extension}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
base = {
    "module_file": str(Path(module.__file__).resolve()),
    "module_name": module.__name__,
    "module_version": getattr(module, "__version__", None),
    "class_present": hasattr(module, "RustFixedCardinalitySampler"),
    "interpreter": str(Path(sys.executable).absolute()),
    "python_version": sys.version,
}
if action == "import":
    print("__CARNOT_JSON__" + json.dumps(base, allow_nan=False, sort_keys=True))
    raise SystemExit(0)

payload = json.loads(sys.stdin.read())
sampler = module.RustFixedCardinalitySampler(
    [tuple(edge) for edge in payload["edges"]],
    payload["fields"],
    payload["cardinality"],
    payload["beta"],
)

def replay(state, tape):
    states = np.asarray([state], dtype=np.int8)
    positive = np.asarray([[draw["positive_index"] for draw in tape]], dtype=np.uintp)
    negative = np.asarray([[draw["negative_index"] for draw in tape]], dtype=np.uintp)
    uniforms = np.asarray([[draw["uniform"] for draw in tape]], dtype=np.float64)
    return sampler.replay_batch(states, positive, negative, uniforms)[0]

if action == "replay":
    replay_result = replay(payload["initial_state"], payload["tape"])
    seeded = sampler.run_seeded_batch(
        np.asarray([payload["initial_state"]], dtype=np.int8),
        [payload["seeded_stream_seed"]],
        payload["burn_in"],
        payload["retained"],
    )[0]
    serialized = sampler.serialize_state(seeded["final_state"])
    result = {
        **base,
        "replay": replay_result,
        "seeded": seeded,
        "serialized_state": serialized,
        "buffer_receipt": dict(sampler.buffer_receipt()),
    }
elif action == "restore":
    restored = dict(sampler.deserialize_state(payload["serialized_state"]))
    result = {
        **base,
        "restored_state": restored,
        "reserialized_state": sampler.serialize_state(restored),
        "continued_replay": replay(restored["spins"], payload["continuation_tape"]),
    }
else:
    raise ValueError(f"unsupported native helper action: {action}")
print("__CARNOT_JSON__" + json.dumps(result, allow_nan=False, sort_keys=True))
"""


def progress(phase: int, boundary: str, operation: str) -> None:
    """Flush one numbered boundary so an external runner sees real state."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def canonical_json(value: Any) -> str:
    """Encode stable finite JSON for evidence hashes."""

    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON value and retain the algorithm label."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash the actual bytes of one source or native binary."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind all stable terminal fields except the checksum itself."""

    material = dict(payload)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def _finish_row(row: JsonDict) -> JsonDict:
    """Add the common comparison contract before hashing the row."""

    row.setdefault("arm", "not_applicable")
    row.setdefault("seed", None)
    row.setdefault("metric", 0.0)
    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row["row_sha256"] = sha256_json(row)
    return row


def unwrap_principled_value(value: Any) -> Any:
    """Unwrap only the exact two-key principle/value representation."""

    if (
        isinstance(value, Mapping)
        and set(value) == {"principle", "value"}
        and isinstance(value.get("principle"), str)
    ):
        return value["value"]
    return value


def upstream_quarantine_observation(
    upstream: Mapping[str, Any], *, manifest_match: bool
) -> JsonDict:
    """Combine artifact and manifest quarantine signals before gate access."""

    observed = {name: upstream.get(name) for name in _QUARANTINE_FIELDS}
    active = [name for name, value in observed.items() if value not in (None, False, "", [], {})]
    if manifest_match:
        active.append("exclusion_manifest")
    return {
        **observed,
        "exclusion_manifest_match": manifest_match,
        "active_flags": active,
        "quarantined": bool(active),
    }


def gated_upstream_value(
    upstream: Mapping[str, Any], quarantine: Mapping[str, Any], field_name: str
) -> Any:
    """Reject quarantine before reading or unwrapping a producer value."""

    if quarantine.get("quarantined") is True:
        return "not_consumed_due_to_quarantine"
    return unwrap_principled_value(upstream.get(field_name))


def _manifest_mentions_experiment(value: Any, experiment_number: str) -> bool:
    """Search identifier fields without treating narrative dates as exclusions."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"experiment_id", "experiment_ids"}:
                identifiers = child if isinstance(child, list) else [child]
                if any(experiment_number in str(identifier) for identifier in identifiers):
                    return True
            if _manifest_mentions_experiment(child, experiment_number):
                return True
    elif isinstance(value, list):
        return any(_manifest_mentions_experiment(child, experiment_number) for child in value)
    return False


def _read_json(path: Path) -> JsonDict:
    """Read one required JSON object and return an empty object on invalid bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _task_contract(root: Path) -> JsonDict | None:
    """Read the exact roadmap fields that authorize this task."""

    try:
        document = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(document, Mapping) or not isinstance(document.get("tasks"), list):
        return None
    task = next(
        (row for row in document["tasks"] if isinstance(row, Mapping) and row.get("id") == TASK_ID),
        None,
    )
    if task is None:
        return None
    return {key: task.get(key) for key in EXPECTED_TASK_CONTRACT}


def _check(
    name: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Use one exact diagnostic shape for every precondition."""

    return {
        "check": name,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _cargo_configuration(root: Path) -> JsonDict:
    """Read the declared PyO3 features before selecting build flags."""

    cargo_path = root / "crates/carnot-python/Cargo.toml"
    try:
        cargo = tomllib.loads(cargo_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    declaration = cargo.get("dependencies", {}).get("pyo3", {})
    features = declaration.get("features", []) if isinstance(declaration, Mapping) else []
    return {
        "pyo3_declaration": declaration,
        "pyo3_features": list(features) if isinstance(features, list) else [],
        "extension_module_enabled": "extension-module" in features,
        "abi3_feature_enabled": any(str(feature).startswith("abi3") for feature in features),
        "abi3_forward_compatibility_enabled": False,
    }


def interpreter_metadata(root: Path) -> JsonDict:
    """Record the interpreter, ABI suffix, Python library, and PyO3 features."""

    return {
        "interpreter": str(Path(sys.executable).absolute()),
        "python_version": platform.python_version(),
        "python_version_long": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "libpython": {
            "LDLIBRARY": sysconfig.get_config_var("LDLIBRARY"),
            "LIBRARY": sysconfig.get_config_var("LIBRARY"),
            "LIBDIR": sysconfig.get_config_var("LIBDIR"),
            "Py_ENABLE_SHARED": sysconfig.get_config_var("Py_ENABLE_SHARED"),
        },
        "cargo": _cargo_configuration(root),
    }


def collect_preconditions(
    root: Path,
    *,
    result_path: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Print first and then retain actual bytes, gates, tools, and authentication."""

    result = result_path or root / RESULT_PATH
    checkpoints = checkpoint_dir or root / CHECKPOINT_DIR
    result.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    def announce(name: str) -> None:
        print(f"[phase 0 check start] {name}", flush=True)

    checks: list[JsonDict] = []
    announce("required source bytes and hashes")
    sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    hashes = {
        str(path): sha256_file(root / path)
        for path in REQUIRED_SOURCE_PATHS
        if sizes[str(path)] not in (None, 0)
    }
    checks.append(
        _check(
            "required_source_bytes",
            "repository",
            "byte_count",
            "every required source is nonempty",
            sizes,
            all(size is not None and size > 0 for size in sizes.values()),
        )
    )

    announce("driving capability specification")
    spec_file = root / SPEC_PATH
    spec_text = spec_file.read_text(encoding="utf-8") if spec_file.is_file() else ""
    spec_observed = {
        "exists": spec_file.is_file(),
        "requirement": "REQ-ISING-7217" in spec_text,
        "scenarios": "SCENARIO-ISING-7217-" in spec_text,
    }
    checks.append(
        _check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-* and SCENARIO-*",
            {"exists": True, "requirement": True, "scenarios": True},
            spec_observed,
            all(spec_observed.values()),
        )
    )

    announce("exact roadmap task contract")
    task_contract = _task_contract(root)
    checks.append(
        _check(
            "roadmap_task_contract",
            str(ROADMAP_PATH),
            "exp7217 task fields",
            EXPECTED_TASK_CONTRACT,
            task_contract,
            task_contract == EXPECTED_TASK_CONTRACT,
        )
    )

    announce("required imports, tools, and output directories")
    cargo = shutil.which("cargo")
    rustc = shutil.which("rustc")
    ldd = shutil.which("ldd")
    tool_observed = {
        "python": str(Path(sys.executable).absolute()),
        "cargo": cargo,
        "rustc": rustc,
        "ldd": ldd,
        "yaml": yaml.__version__,
        "output_parent": result.parent.is_dir() and os.access(result.parent, os.W_OK),
        "checkpoint_dir": checkpoints.is_dir() and os.access(checkpoints, os.W_OK),
    }
    checks.append(
        _check(
            "imports_tools_and_directories",
            "host",
            "python_cargo_rustc_ldd_yaml_storage",
            "all available",
            tool_observed,
            bool(cargo and rustc and ldd)
            and tool_observed["output_parent"] is True
            and tool_observed["checkpoint_dir"] is True,
        )
    )

    announce("PyO3 and Cargo feature configuration")
    cargo_config = _cargo_configuration(root)
    checks.append(
        _check(
            "pyo3_feature_configuration",
            "crates/carnot-python/Cargo.toml",
            "extension-module without abi3 workaround",
            {"extension_module_enabled": True, "abi3_forward_compatibility_enabled": False},
            cargo_config,
            cargo_config.get("extension_module_enabled") is True
            and cargo_config.get("abi3_forward_compatibility_enabled") is False,
        )
    )

    announce("exclusion manifest before producer gates")
    try:
        exclusion = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        exclusion = None
    native = _read_json(root / UPSTREAM_NATIVE_PATH)
    historical_null = _read_json(root / UPSTREAM_NULL_PATH)
    board = _read_json(root / UPSTREAM_BOARD_PATH)
    native_quarantine = upstream_quarantine_observation(
        native, manifest_match=_manifest_mentions_experiment(exclusion, "7201")
    )
    null_quarantine = upstream_quarantine_observation(
        historical_null, manifest_match=_manifest_mentions_experiment(exclusion, "7202")
    )
    board_quarantine = upstream_quarantine_observation(
        board, manifest_match=_manifest_mentions_experiment(exclusion, "7203")
    )
    for label, path, observation in (
        ("exp7201", UPSTREAM_NATIVE_PATH, native_quarantine),
        ("exp7202", UPSTREAM_NULL_PATH, null_quarantine),
        ("exp7203", UPSTREAM_BOARD_PATH, board_quarantine),
    ):
        checks.append(
            _check(
                f"{label}_not_quarantined",
                str(path),
                "quarantined",
                False,
                observation,
                observation["quarantined"] is False,
            )
        )

    announce("Exp7201 producer authentication and exact native gate")
    native_auth = (
        validate_exp7201(native) if native and native_quarantine["quarantined"] is False else []
    )
    checks.append(
        _check(
            "exp7201_producer_authentication",
            str(UPSTREAM_NATIVE_PATH),
            "shipped_validator_errors",
            [],
            native_auth,
            bool(native) and native_quarantine["quarantined"] is False and native_auth == [],
        )
    )
    native_gate = gated_upstream_value(native, native_quarantine, "pyo3_slice_ready_score")
    checks.append(
        _check(
            "exp7201_native_gate",
            str(UPSTREAM_NATIVE_PATH),
            "pyo3_slice_ready_score",
            1,
            native_gate,
            native_auth == [] and native_gate == 1,
        )
    )

    announce("Exp7203 board producer authentication and exact receipt gate")
    board_auth = (
        validate_exp7203(board) if board and board_quarantine["quarantined"] is False else []
    )
    checks.append(
        _check(
            "exp7203_board_producer_authentication",
            str(UPSTREAM_BOARD_PATH),
            "shipped_validator_errors",
            [],
            board_auth,
            bool(board) and board_quarantine["quarantined"] is False and board_auth == [],
        )
    )
    board_gate = gated_upstream_value(board, board_quarantine, "hardware_envelope_complete_score")
    board_names = sorted(
        row.get("board")
        for row in board.get("board_rows", [])
        if isinstance(row, Mapping) and isinstance(row.get("board"), str)
    )
    board_gate_observed = {"hardware_envelope_complete_score": board_gate, "boards": board_names}
    board_gate_expected = {
        "hardware_envelope_complete_score": 1,
        "boards": ["GateMate", "KV260", "PolarFire"],
    }
    checks.append(
        _check(
            "exp7203_board_receipt_gate",
            str(UPSTREAM_BOARD_PATH),
            "hardware_envelope_complete_score and board names",
            board_gate_expected,
            board_gate_observed,
            board_auth == [] and board_gate_observed == board_gate_expected,
        )
    )

    announce("Exp7202 failed value remains failed and does not authorize work")
    failed_value = gated_upstream_value(historical_null, null_quarantine, "nfr_01_10x_met")
    checks.append(
        _check(
            "exp7202_failed_value_preserved",
            str(UPSTREAM_NULL_PATH),
            "nfr_01_10x_met",
            False,
            failed_value,
            null_quarantine["quarantined"] is False and failed_value is False,
        )
    )
    checks.append(
        _check(
            "source_artifact_hashes",
            "required_source_bytes",
            "sha256",
            len(REQUIRED_SOURCE_PATHS),
            len(hashes),
            len(hashes) == len(REQUIRED_SOURCE_PATHS),
        )
    )
    return checks, hashes


def interpreter_build_environment(
    python_executable: Path, target_dir: Path, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Bind PyO3 to one interpreter and remove the generic abi3 workaround."""

    environment = dict(os.environ if base is None else base)
    environment.pop("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", None)
    environment["PYO3_PYTHON"] = str(python_executable.absolute())
    environment["CARGO_TARGET_DIR"] = str(target_dir.resolve())
    return environment


class _Heartbeat:
    """Report elapsed blocking-call state without claiming hidden progress."""

    def __init__(self, operation: str, interval_s: float = 30.0) -> None:
        self.operation = operation
        self.interval_s = interval_s
        self.started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _Heartbeat:
        self.started = time.monotonic()

        def report() -> None:
            while not self._stop.wait(self.interval_s):
                print(
                    f"[heartbeat] operation={self.operation} state=waiting "
                    f"elapsed_s={time.monotonic() - self.started:.3f}",
                    flush=True,
                )

        self._thread = threading.Thread(target=report, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _bounded_native_process(
    extension: Path,
    action: str,
    payload: Mapping[str, Any] | None = None,
    *,
    timeout_s: int = NATIVE_TIMEOUT_S,
) -> JsonDict:
    """Load one exact binary in a fresh bounded process and retain raw output."""

    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        "-c",
        _NATIVE_HELPER,
        str(extension),
        action,
    ]
    started = time.monotonic()
    with _Heartbeat(f"native {action}"):
        try:
            completed = subprocess.run(
                command,
                input=canonical_json(payload or {}),
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return {
                "command": command,
                "exit_code": None,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": True,
                "duration_s": time.monotonic() - started,
                "result": None,
                "undefined_symbol": _undefined_symbol(stderr),
            }
    result: JsonDict | None = None
    marker = "__CARNOT_JSON__"
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            candidate = json.loads(line[len(marker) :])
            if isinstance(candidate, Mapping):
                result = dict(candidate)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": timed_out,
        "duration_s": time.monotonic() - started,
        "result": result,
        "undefined_symbol": _undefined_symbol(completed.stderr),
    }


def _undefined_symbol(stderr: str) -> str | None:
    """Extract the loader's actual undefined symbol without inventing history."""

    match = re.search(r"undefined symbol:\s*([^\s]+)", stderr)
    return match.group(1) if match else None


def _stream_process(
    command: Sequence[str],
    *,
    root: Path,
    environment: Mapping[str, str],
    operation: str,
    timeout_s: int,
) -> JsonDict:
    """Stream build output, report waiting state, and enforce one hard deadline."""

    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=root,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError(f"{operation} did not expose build output")
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    captured: list[str] = []
    last_heartbeat = started
    stream_finished = False
    while not stream_finished or process.poll() is None:
        now = time.monotonic()
        if now - started > timeout_s:
            process.kill()
            process.wait(timeout=10)
            raise subprocess.TimeoutExpired(command, timeout_s, output="".join(captured))
        try:
            item = lines.get(timeout=1.0)
        except queue.Empty:
            item = ""
        if item is None:
            stream_finished = True
        elif item:
            captured.append(item)
            print(item.rstrip(), flush=True)
        if now - last_heartbeat >= 30.0:
            print(
                f"[heartbeat] operation={operation} state=process_running "
                f"elapsed_s={now - started:.3f}",
                flush=True,
            )
            last_heartbeat = now
    reader.join(timeout=1.0)
    returncode = process.wait(timeout=10)
    receipt = {
        "command": list(command),
        "exit_code": returncode,
        "duration_s": time.monotonic() - started,
        "output": "".join(captured),
    }
    if returncode != 0:
        raise RuntimeError(f"{operation} failed with exit {returncode}")
    return receipt


def _extension_suffix() -> str:
    """Use this interpreter's complete extension suffix."""

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python extension suffix is unavailable")
    return suffix


def historical_extension_path(root: Path) -> Path:
    """Return the exact path named by the 2026-09-11 failure report."""

    return root / "target/exp7201-pyo3-load" / f"_rust{_extension_suffix()}"


def build_interpreter_bound_extension(root: Path) -> tuple[Path, JsonDict]:
    """Build in a task-specific target and copy only the resulting library."""

    target = root / TASK_TARGET_DIR
    environment = interpreter_build_environment(Path(sys.executable), target)
    command = ["cargo", "build", "--release", "-p", "carnot-python"]
    build_receipt = _stream_process(
        command,
        root=root,
        environment=environment,
        operation="interpreter-bound carnot-python build",
        timeout_s=BUILD_TIMEOUT_S,
    )
    library = target / "release/libcarnot_python.so"
    if not library.is_file():
        raise RuntimeError(f"interpreter-bound build output is missing: {library}")
    load_dir = root / TASK_LOAD_DIR
    load_dir.mkdir(parents=True, exist_ok=True)
    destination = load_dir / f"_rust{_extension_suffix()}"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=load_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(library, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    build_receipt.update(
        {
            "PYO3_PYTHON": environment["PYO3_PYTHON"],
            "CARGO_TARGET_DIR": environment["CARGO_TARGET_DIR"],
            "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": None,
            "built_library": str(library.resolve()),
            "loaded_copy": str(destination.resolve()),
            "binary_sha256": sha256_file(destination),
        }
    )
    return destination, build_receipt


def native_fixture() -> JsonDict:
    """Create one explicit tape and expected replay from the Exp7187 authority."""

    instance = slices.make_frustrated_instance(8, 718701)
    cardinality = 2
    initial = slices.enumerate_slice(8, cardinality)[3]
    tape = exp7189.make_replay_tape(8, cardinality, seed=REPLAY_SEED, steps=REPLAY_STEPS)
    expected = exp7189.python_replay(instance, cardinality, 2.0, initial, tape)
    return {
        "authority": "experiment_7187_v633_slice_sampler",
        "edges": [list(edge) for edge in instance.edges],
        "fields": list(instance.fields),
        "cardinality": cardinality,
        "beta": 2.0,
        "initial_state": list(initial),
        "tape": tape,
        "expected_replay": json.loads(canonical_json(expected)),
        "seeded_stream_seed": REPLAY_SEED + 1,
        "burn_in": 2,
        "retained": 3,
    }


def _linkage_receipt(extension: Path) -> JsonDict:
    """Record the selected binary's dynamic linker view."""

    with _Heartbeat("ldd selected native extension"):
        completed = subprocess.run(
            ["ldd", str(extension)],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    return {
        "command": ["ldd", str(extension)],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "libpython_lines": [
            line.strip() for line in completed.stdout.splitlines() if "libpython" in line
        ],
    }


def run_native_e2e(extension: Path) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    """Run explicit replay and state restoration in two fresh processes."""

    fixture = native_fixture()
    first = _bounded_native_process(extension, "replay", fixture)
    if first["exit_code"] != 0 or not isinstance(first.get("result"), Mapping):
        raise RuntimeError(f"fresh native replay failed: {first['stderr']}")
    first_result = dict(first["result"])
    native_replay = first_result.get("replay")
    if not isinstance(native_replay, Mapping):
        raise RuntimeError("fresh native replay did not return transition output")
    serialized = first_result.get("serialized_state")
    seeded = first_result.get("seeded")
    if not isinstance(serialized, str) or not isinstance(seeded, Mapping):
        raise RuntimeError("fresh native replay did not return restart state")
    final_state = seeded.get("final_state")
    if not isinstance(final_state, Mapping):
        raise RuntimeError("fresh native replay returned malformed restart state")
    compared = exp7189.compare_replays(
        "exp7217:fresh-process", fixture["tape"], fixture["expected_replay"], native_replay
    )
    rows: list[JsonDict] = []
    for index, comparison in enumerate(compared):
        comparison.pop("row_sha256", None)
        comparison.update(
            {
                "unit_id": f"native-transition:{index}",
                "arm": "interpreter_bound_pyo3_vs_exp7187_python",
                "seed": REPLAY_SEED,
                "metric": float(comparison.get("delta_energy_error", 0.0)),
                "error": None,
                "abstention": False,
                "passed": comparison.get("passed") is True
                and comparison.get("delta_energy_error", float("inf")) <= TOLERANCE,
            }
        )
        rows.append(_finish_row(comparison))
    continuation_tape = exp7189.make_replay_tape(8, 2, seed=REPLAY_SEED + 2, steps=REPLAY_STEPS)
    second_payload = {
        **{key: fixture[key] for key in ("edges", "fields", "cardinality", "beta")},
        "serialized_state": serialized,
        "continuation_tape": continuation_tape,
    }
    second = _bounded_native_process(extension, "restore", second_payload)
    if second["exit_code"] != 0 or not isinstance(second.get("result"), Mapping):
        raise RuntimeError(f"fresh native restore failed: {second['stderr']}")
    second_result = dict(second["result"])
    restored = second_result.get("restored_state")
    continued = second_result.get("continued_replay")
    if not isinstance(restored, Mapping) or not isinstance(continued, Mapping):
        raise RuntimeError("fresh native restore did not return state and continuation")
    expected_continuation = exp7189.python_replay(
        slices.make_frustrated_instance(8, 718701),
        2,
        2.0,
        restored["spins"],
        continuation_tape,
    )
    continuation_rows = exp7189.compare_replays(
        "exp7217:restored-process", continuation_tape, expected_continuation, continued
    )
    continuation_passed = all(row.get("passed") is True for row in continuation_rows)
    state_passed = (
        dict(restored) == dict(final_state)
        and second_result.get("reserialized_state") == serialized
        and continuation_passed
    )
    rows.append(
        _finish_row(
            {
                "unit_id": "native-state:cross-process-restore",
                "arm": "serialized_rust_state_round_trip",
                "seed": REPLAY_SEED + 1,
                "metric": int(state_passed),
                "error": None if state_passed else "restored state or continuation mismatch",
                "abstention": False,
                "passed": state_passed,
                "serialized_state": serialized,
                "restored_state": dict(restored),
                "continuation_python_sha256": sha256_json(expected_continuation),
                "continuation_rust_sha256": sha256_json(continued),
            }
        )
    )
    e2e_receipts = [
        {
            "scenario": "native_import_fresh_process",
            "spec_ref": "SCENARIO-ISING-7217-ABI",
            "passed": first["exit_code"] == 0 and first_result.get("class_present") is True,
            "exit_code": first["exit_code"],
            "module_file": first_result.get("module_file"),
            "interpreter": first_result.get("interpreter"),
        },
        {
            "scenario": "E2E-003",
            "spec_ref": "REQ-ISING-7217",
            "passed": bool(compared) and all(row["passed"] for row in rows[:-1]),
            "transition_count": len(compared),
            "maximum_energy_error": max(
                float(row.get("delta_energy_error", 0.0)) for row in compared
            ),
        },
        {
            "scenario": "E2E-004",
            "spec_ref": "SCENARIO-ISING-7217-ABI",
            "passed": state_passed and second["exit_code"] == 0,
            "exit_code": second["exit_code"],
            "serialized_state": serialized,
            "restored_state": dict(restored),
            "second_process_module_file": second_result.get("module_file"),
        },
    ]
    native_receipt = {
        "binary_sha256": sha256_file(extension),
        "module_file": first_result.get("module_file"),
        "interpreter": first_result.get("interpreter"),
        "import_exit_code": first["exit_code"],
        "restore_exit_code": second["exit_code"],
        "explicit_inputs": fixture,
        "exact_outputs": {
            "first_process": first_result,
            "second_process": second_result,
        },
        "raw_process_receipts": {"first": first, "second": second},
        "linked_libraries": _linkage_receipt(extension),
        "compiled_execution": True,
        "python_fallback_used": False,
    }
    return rows, e2e_receipts, native_receipt


def load_board_evidence(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Copy each authenticated board row and verify its cited receipt bytes."""

    artifact = _read_json(root / UPSTREAM_BOARD_PATH)
    source_rows = artifact.get("board_rows", [])
    if not isinstance(source_rows, list):
        raise ValueError("Exp7203 board_rows must be a list")
    rows: list[JsonDict] = []
    for source in source_rows:
        if not isinstance(source, Mapping) or not isinstance(source.get("board"), str):
            raise ValueError("Exp7203 board row is malformed")
        row = deepcopy(dict(source))
        evidence = root / str(row.get("evidence_path", ""))
        if not evidence.is_file() or sha256_file(evidence) != row.get("source_hash"):
            raise ValueError(f"board evidence hash mismatch: {evidence}")
        row["receipt_authenticated"] = True
        row["receipt_date"] = row.get("receipt_date") or row.get("recorded_date")
        row["exact_next_prerequisite"] = row.get("exact_next_prerequisite") or row.get(
            "unresolved_prerequisite"
        )
        row["hardware_command_count"] = 0
        rows.append(row)
    by_board = {row["board"]: row for row in rows}
    if set(by_board) != {"KV260", "GateMate", "PolarFire"}:
        raise ValueError("Exp7203 must supply exactly the three attached boards")
    expected_dispositions = {
        "KV260": "graduated_preserved",
        "GateMate": "blocked_inherited_no_new_physical_state",
        "PolarFire": "blocked_missing_raw_dispatch_transcript",
    }
    for board, disposition in expected_dispositions.items():
        if by_board[board].get("disposition") != disposition:
            raise ValueError(f"{board} disposition changed from the authenticated receipt")
    operator_source = artifact.get("operator_state_receipt", {})
    if not isinstance(operator_source, Mapping):
        raise ValueError("Exp7203 operator_state_receipt is malformed")
    operator = {
        **dict(operator_source),
        "compared_to_experiment": "Exp6559",
        "newer_than_exp6559": operator_source.get("newer_than_exp6559") is True,
        "hardware_command_count": 0,
        "hardware_operations_issued": [],
    }
    return rows, operator


def operation_map() -> list[JsonDict]:
    """Separate potential deployment operations from measured host work."""

    return [
        {
            "operation": "down_up_remove_member",
            "executed_here": False,
            "current_mapping": "host_cpu_candidate",
            "potential_targets": ["cpu", "gpu", "fpga", "tsu"],
            "device_measurement": "unknown",
        },
        {
            "operation": "down_up_replacement_energy_and_logsumexp_normalization",
            "executed_here": False,
            "current_mapping": "host_work_until_explicit_device_mapping",
            "potential_targets": ["cpu", "gpu", "fpga", "tsu"],
            "device_measurement": "unknown",
        },
        {
            "operation": "compiled_predicate_lookup_pair_swap_replay",
            "executed_here": True,
            "current_mapping": "host_cpu_pyo3",
            "potential_targets": ["cpu", "gpu", "fpga", "tsu"],
            "device_measurement": "host_only",
        },
        {
            "operation": "z1_fixed_parent_graph_embedding",
            "executed_here": False,
            "current_mapping": "unmapped",
            "degree_bound": "maximum_degree<=16_is_necessary_not_sufficient",
            "device_measurement": "unknown",
        },
    ]


def _base_artifact(
    root: Path, checks: list[JsonDict], hashes: Mapping[str, str], run_date: str
) -> JsonDict:
    """Create all required fields before choosing a terminal outcome."""

    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": checks,
        "inference_substrate": "not_started",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown-host",
        "duration_s": 0.0,
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": {
            "planned": REPLAY_STEPS + 1,
            "attempted": 0,
            "completed": 0,
            "censored": 0,
            "independent_unit_count": 1,
            "board_receipts_planned": 3,
            "board_receipts_completed": 0,
        },
        "random_seed": REPLAY_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "running",
        "native_abi_ready_score": 0,
        "abi_board_receipt_complete_score": 0,
        "abi_rows": [],
        "e2e_receipts": [],
        "board_rows": [],
        "operator_state_receipt": {},
        "hardware_operations_issued": [],
        "topology_fit": "unknown",
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_invoked": False,
        "native_execution_receipt": {},
        "upstream_failed_value": {
            "artifact": str(UPSTREAM_NULL_PATH),
            "field": "nfr_01_10x_met",
            "observed_value": False,
            "promoted": False,
        },
        "operation_map": operation_map(),
        "device_latency": {"gpu": "unknown", "fpga": "unknown", "tsu": "unknown"},
        "device_power": {"gpu": "unknown", "fpga": "unknown", "tsu": "unknown"},
        "hardware_performance_claimed": False,
        "throughput_sweep_rerun": False,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "spec_refs": [
            "REQ-ISING-7217",
            "SCENARIO-ISING-7217-PREFLIGHT",
            "SCENARIO-ISING-7217-ABI",
            "SCENARIO-ISING-7217-REBUILD",
            "SCENARIO-ISING-7217-BOARDS",
            "SCENARIO-ISING-7217-ARTIFACT",
        ],
        "root": str(root.resolve()),
    }


def _blocked_artifact(
    artifact: JsonDict,
    failed: Mapping[str, Any],
    started: float,
    *,
    qualifying_work: bool = False,
) -> JsonDict:
    """Publish one exact external block without invented native rows."""

    artifact.update(
        {
            "status": "blocked_external_precondition",
            "inference_substrate": (
                "aggregation_from_upstream_artifacts" if qualifying_work else "blocked_no_run"
            ),
            "inference_substrate_class": "aggregation" if qualifying_work else "blocked_no_run",
            "duration_s": time.monotonic() - started,
            "gate_check_summary": {
                "passed": False,
                "failed_check": failed.get("check"),
                "upstream": failed.get("upstream"),
                "field": failed.get("field"),
                "expected_value": failed.get("expected_value"),
                "observed_value": failed.get("observed_value"),
            },
            "verdict_class": "blocked",
            "honest_verdict": (
                f"blocked_external_precondition: {failed.get('check')} failed before native "
                "readiness could be established"
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(root: Path, run_date: str = RUN_DATE) -> JsonDict:
    """Build a measured native receipt or one diagnosed external block."""

    started = time.monotonic()
    progress(0, "before", "source, contract, quarantine, gate, tool, and directory checks")
    checks, hashes = collect_preconditions(root)
    progress(0, "after", "source, contract, quarantine, gate, tool, and directory checks")
    artifact = _base_artifact(root, checks, hashes, run_date)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        return _blocked_artifact(artifact, failed, started)
    artifact["gate_check_summary"] = {
        "passed": True,
        "failed_check": None,
        "upstream": "repository_preflight",
        "field": "all_required_preconditions",
        "expected_value": "all pass",
        "observed_value": "all pass",
    }

    progress(1, "start", "interpreter, SOABI, extension suffix, library, and Cargo features")
    metadata = interpreter_metadata(root)
    print(canonical_json(metadata), flush=True)
    artifact["abi_rows"].append({"row_type": "interpreter_configuration", **metadata})
    progress(1, "end", "interpreter and build configuration recorded")

    progress(2, "start", "bounded historical extension import")
    historical = historical_extension_path(root)
    if historical.is_file():
        historical_probe = _bounded_native_process(historical, "import")
    else:
        historical_probe = {
            "command": [str(Path(sys.executable).absolute()), "native-import", str(historical)],
            "exit_code": None,
            "stdout": "",
            "stderr": "historical extension path is absent",
            "timed_out": False,
            "duration_s": 0.0,
            "result": None,
            "undefined_symbol": None,
        }
    artifact["abi_rows"].append(
        {"row_type": "historical_import_probe", "path": str(historical), **historical_probe}
    )
    progress(
        2,
        "end",
        f"historical import exit_code={historical_probe['exit_code']} "
        f"undefined_symbol={historical_probe['undefined_symbol']}",
    )

    selected = historical
    build_receipt: JsonDict = {
        "build_required": False,
        "fast_path": True,
        "reason": "historical extension imported in the selected interpreter",
    }
    if historical_probe.get("exit_code") != 0 or not isinstance(
        historical_probe.get("result"), Mapping
    ):
        progress(3, "start", "interpreter-bound carnot-python build")
        try:
            selected, measured_build = build_interpreter_bound_extension(root)
            build_receipt = {"build_required": True, "fast_path": False, **measured_build}
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            progress(3, "end", f"interpreter-bound build failed error={exc}")
            artifact["abi_rows"].append(
                {
                    "row_type": "interpreter_bound_build",
                    "build_required": True,
                    "error": str(exc),
                    "PYO3_PYTHON": str(Path(sys.executable).absolute()),
                    "CARGO_TARGET_DIR": str((root / TASK_TARGET_DIR).resolve()),
                    "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": None,
                }
            )
            return _blocked_artifact(
                artifact,
                {
                    "check": "interpreter_bound_native_build",
                    "upstream": "host cargo/PyO3 toolchain",
                    "field": "carnot-python cdylib",
                    "expected_value": "build exits 0 within 900 seconds",
                    "observed_value": str(exc),
                },
                started,
            )
        progress(3, "end", f"interpreter-bound build selected={selected}")
    else:
        progress(3, "start", "verified fast path; no rebuild required")
        progress(3, "end", f"verified fast path selected={selected}")
    artifact["abi_rows"].append({"row_type": "build_selection", **build_receipt})

    progress(4, "start", "fresh-process compiled replay and Exp7187 parity")
    try:
        rows, e2e_receipts, native_receipt = run_native_e2e(selected)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        progress(4, "end", f"fresh-process native execution failed error={exc}")
        return _blocked_artifact(
            artifact,
            {
                "check": "fresh_process_native_execution",
                "upstream": str(selected),
                "field": "import_replay_serialize_restore",
                "expected_value": "two fresh processes pass native parity and state restore",
                "observed_value": str(exc),
            },
            started,
        )
    artifact["rows"] = rows
    artifact["e2e_receipts"] = e2e_receipts
    artifact["native_execution_receipt"] = native_receipt
    artifact["abi_rows"].append(
        {
            "row_type": "selected_extension",
            "path": str(selected.resolve()),
            "binary_sha256": sha256_file(selected),
            "module_file": native_receipt["module_file"],
            "interpreter": native_receipt["interpreter"],
            "linked_libraries": native_receipt["linked_libraries"],
        }
    )
    progress(4, "end", f"fresh-process native comparison rows={len(rows)}")

    progress(5, "start", "authenticated read-only KV260, GateMate, and PolarFire receipts")
    board_rows, operator_receipt = load_board_evidence(root)
    artifact["board_rows"] = board_rows
    artifact["operator_state_receipt"] = operator_receipt
    progress(5, "end", f"authenticated board receipts={len(board_rows)} operations=0")

    progress(6, "start", "host and potential device operation mapping")
    native_ready = (
        bool(rows)
        and all(row.get("passed") is True for row in rows)
        and len(e2e_receipts) == 3
        and all(row.get("passed") is True for row in e2e_receipts)
        and native_receipt.get("compiled_execution") is True
        and native_receipt.get("python_fallback_used") is False
        and native_receipt.get("module_file") == str(selected.resolve())
    )
    board_terminal = (
        len(board_rows) == 3
        and all(row.get("receipt_authenticated") is True for row in board_rows)
        and all(row.get("hardware_command_count") == 0 for row in board_rows)
    )
    artifact["native_abi_ready_score"] = int(native_ready)
    artifact["abi_board_receipt_complete_score"] = int(native_ready and board_terminal)
    artifact["sample_size_budget"].update(
        {
            "attempted": REPLAY_STEPS + 1,
            "completed": len(rows),
            "censored": 0,
            "board_receipts_completed": len(board_rows),
        }
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": (
                "host CPU execution of the compiled fixed-cardinality pair-swap replay, Exp7187 "
                "Python energy/transition scoring, and read-only board-receipt aggregation"
            ),
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "duration_s": time.monotonic() - started,
            "verdict_class": "circular_positive",
            "honest_verdict": (
                "complete: the selected interpreter imported and executed the genuine compiled "
                "sampler, explicit transitions matched the Exp7187 authority, and serialized "
                "state survived a second process. KV260 graduation, GateMate's unchanged "
                "physical-state block, and PolarFire dispatch uncertainty remain separate. "
                "This readiness receipt does not reopen the failed 10x claim or establish device "
                "performance."
            ),
        }
    )
    progress(6, "end", "operation mapping complete; device latency, power, and topology unknown")
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Recompute terminal state, native provenance, boards, limits, and hashes."""

    errors: list[str] = []
    if not REQUIRED_ARTIFACT_FIELDS.issubset(artifact):
        errors.append("missing_required_fields")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    try:
        checksum_valid = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_valid = False
    if not checksum_valid:
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_declaration_invalid")
    if artifact.get("execution_venue") != "host" or not artifact.get("execution_host"):
        errors.append("execution_identity_invalid")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_authority_invalid")
    if artifact.get("hardware_operations_issued") != []:
        errors.append("hardware_operations_invalid")
    if artifact.get("topology_fit") != "unknown":
        errors.append("topology_fit_invalid")
    if artifact.get("hardware_performance_claimed") is not False:
        errors.append("hardware_claim_invalid")
    if artifact.get("throughput_sweep_rerun") is not False:
        errors.append("throughput_rerun_invalid")
    if artifact.get("upstream_failed_value", {}).get("promoted") is not False:
        errors.append("upstream_failed_value_promoted")

    if artifact.get("verdict_class") == "blocked":
        gate = artifact.get("gate_check_summary", {})
        if (
            artifact.get("status") != "blocked_external_precondition"
            or artifact.get("native_abi_ready_score") != 0
            or artifact.get("abi_board_receipt_complete_score") != 0
            or artifact.get("rows") != []
            or not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or not all(
                gate.get(name) is not None
                for name in (
                    "failed_check",
                    "upstream",
                    "field",
                    "expected_value",
                    "observed_value",
                )
            )
        ):
            errors.append("blocked_state_invalid")
        return list(dict.fromkeys(errors))

    rows = artifact.get("rows", [])
    receipts = artifact.get("e2e_receipts", [])
    boards = artifact.get("board_rows", [])
    rows_valid = (
        isinstance(rows, list)
        and len(rows) == REPLAY_STEPS + 1
        and all(
            isinstance(row, Mapping)
            and all(
                name in row for name in ("unit_id", "arm", "seed", "metric", "error", "abstention")
            )
            and row.get("passed") is True
            and row.get("abstention") is False
            and row.get("row_sha256")
            == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
            for row in rows
        )
    )
    if not rows_valid:
        errors.append("rows_invalid")
    receipts_valid = (
        isinstance(receipts, list)
        and len(receipts) == 3
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in receipts)
    )
    if not receipts_valid:
        errors.append("e2e_receipts_invalid")
    native = artifact.get("native_execution_receipt", {})
    native_valid = (
        isinstance(native, Mapping)
        and native.get("compiled_execution") is True
        and native.get("python_fallback_used") is False
        and native.get("import_exit_code") == 0
        and native.get("restore_exit_code") == 0
        and isinstance(native.get("binary_sha256"), str)
        and str(native.get("binary_sha256")).startswith("sha256:")
        and isinstance(native.get("module_file"), str)
        and isinstance(native.get("interpreter"), str)
    )
    if not native_valid:
        errors.append("native_execution_receipt_invalid")
    board_valid = False
    if isinstance(boards, list):
        by_board = {
            row.get("board"): row for row in boards if isinstance(row, Mapping) and row.get("board")
        }
        board_valid = (
            set(by_board) == {"KV260", "GateMate", "PolarFire"}
            and by_board["KV260"].get("disposition") == "graduated_preserved"
            and by_board["GateMate"].get("disposition") == "blocked_inherited_no_new_physical_state"
            and by_board["PolarFire"].get("disposition")
            == "blocked_missing_raw_dispatch_transcript"
            and all(row.get("receipt_authenticated") is True for row in by_board.values())
            and all(row.get("hardware_command_count") == 0 for row in by_board.values())
            and all(row.get("receipt_date") for row in by_board.values())
            and all(row.get("terminal_criterion") for row in by_board.values())
            and all(row.get("exact_next_prerequisite") for row in by_board.values())
        )
    if not board_valid:
        errors.append("board_rows_invalid")
    operator = artifact.get("operator_state_receipt", {})
    if (
        not isinstance(operator, Mapping)
        or operator.get("newer_than_exp6559") is not False
        or operator.get("hardware_operations_issued") != []
    ):
        errors.append("operator_state_receipt_invalid")
    budget = artifact.get("sample_size_budget", {})
    if (
        not isinstance(budget, Mapping)
        or budget.get("planned") != REPLAY_STEPS + 1
        or budget.get("attempted") != REPLAY_STEPS + 1
        or budget.get("completed") != REPLAY_STEPS + 1
        or budget.get("censored") != 0
        or budget.get("independent_unit_count") != 1
        or budget.get("board_receipts_completed") != 3
    ):
        errors.append("sample_size_budget_invalid")
    complete_valid = rows_valid and receipts_valid and native_valid and board_valid
    if artifact.get("native_abi_ready_score") != int(
        rows_valid and receipts_valid and native_valid
    ):
        errors.append("native_readiness_invalid")
    if artifact.get("abi_board_receipt_complete_score") != int(complete_valid):
        errors.append("complete_receipt_score_invalid")
    if (
        artifact.get("status") != "complete"
        or artifact.get("verdict_class") != "circular_positive"
        or not str(artifact.get("honest_verdict", "")).startswith("complete:")
        or artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
        or artifact.get("gate_check_summary", {}).get("passed") is not True
    ):
        errors.append("terminal_state_invalid")
    if not isinstance(artifact.get("abi_rows"), list) or not artifact.get("abi_rows"):
        errors.append("abi_rows_invalid")
    if root is not None:
        recorded = artifact.get("source_artifact_hashes", {})
        expected_paths = {str(path) for path in REQUIRED_SOURCE_PATHS}
        if (
            not isinstance(recorded, Mapping)
            or set(recorded) != expected_paths
            or any(
                not (root / path).is_file() or recorded[path] != sha256_file(root / path)
                for path in expected_paths
            )
        ):
            errors.append("source_artifact_hashes_invalid")
        if native_valid:
            module_file = Path(str(native["module_file"]))
            if not module_file.is_file() or native["binary_sha256"] != sha256_file(module_file):
                errors.append("native_binary_hash_invalid")
    return list(dict.fromkeys(errors))


def complete_artifact_fixture_for_test(root: Path) -> JsonDict:
    """Create a small structurally complete receipt for validator unit tests."""

    artifact = _base_artifact(root, [], {"fixture": "sha256:test"}, RUN_DATE)
    board_rows, operator = load_board_evidence(root)
    rows = [
        _finish_row(
            {
                "unit_id": f"fixture:{index}",
                "arm": "compiled_fixture",
                "seed": REPLAY_SEED,
                "metric": 0.0 if index < REPLAY_STEPS else 1,
                "error": None,
                "abstention": False,
                "passed": True,
            }
        )
        for index in range(REPLAY_STEPS + 1)
    ]
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": "host CPU compiled fixture",
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "duration_s": 1.0,
            "rows": rows,
            "sample_size_budget": {
                "planned": REPLAY_STEPS + 1,
                "attempted": REPLAY_STEPS + 1,
                "completed": REPLAY_STEPS + 1,
                "censored": 0,
                "independent_unit_count": 1,
                "board_receipts_planned": 3,
                "board_receipts_completed": 3,
            },
            "gate_check_summary": {"passed": True},
            "verdict_class": "circular_positive",
            "honest_verdict": "complete: validator fixture",
            "native_abi_ready_score": 1,
            "abi_board_receipt_complete_score": 1,
            "abi_rows": [{"row_type": "fixture"}],
            "e2e_receipts": [{"passed": True}, {"passed": True}, {"passed": True}],
            "board_rows": board_rows,
            "operator_state_receipt": operator,
            "native_execution_receipt": {
                "compiled_execution": True,
                "python_fallback_used": False,
                "import_exit_code": 0,
                "restore_exit_code": 0,
                "binary_sha256": "sha256:test",
                "module_file": "/tmp/test-extension.so",
                "interpreter": str(Path(sys.executable).absolute()),
            },
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def blocked_artifact_for_test(root: Path, failed: Mapping[str, Any]) -> JsonDict:
    """Create one blocked receipt for exact gate-summary tests."""

    artifact = _base_artifact(root, [dict(failed)], {"fixture": "sha256:test"}, RUN_DATE)
    return _blocked_artifact(artifact, failed, time.monotonic())


def atomic_write(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Write complete JSON through one same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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
    """Build, independently validate, and atomically publish Exp7217."""

    artifact = build_artifact(root, run_date)
    progress(7, "before", "terminal artifact validation")
    errors = validate_artifact(artifact, root=root)
    progress(7, "after", f"terminal artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7217 artifact: {errors}")
    progress(8, "before", "final atomic artifact write")
    receipt = atomic_write(output, artifact)
    progress(8, "after", f"final atomic artifact write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and optional read-only validation path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded study or validate existing bytes without mutation."""

    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.validate is not None:
        progress(7, "before", f"read-only validation path={args.validate}")
        try:
            artifact = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(artifact)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"validation_error: {exc}", flush=True)
            progress(7, "after", "read-only validation errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        progress(7, "after", f"read-only validation errors={len(errors)}")
        return 0 if not errors else 2
    if args.date != RUN_DATE:
        print(f"experiment_error: run date must be {RUN_DATE}", flush=True)
        return 2
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        run_experiment(root, output, args.date)
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"experiment_error: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
