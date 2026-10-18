"""Preserve three-board evidence and recover one PolarFire CPU transcript.

The controller runs on the host. It authenticates existing receipts before it
opens one bounded PolarFire SSH transport. It never contacts KV260 or GateMate,
and it never treats a board CPU process as FPGA fabric execution.

Spec: REQ-ISING-7231 and SCENARIO-ISING-7231-POLARFIRE.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any

import yaml

from carnot import experiment_7217_v635_abi_board_readiness as exp7217
from carnot import experiment_7226_v636_belief_compiler as exp7226


JsonDict = dict[str, Any]
CommandRunner = Callable[[tuple[str, ...], float], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260912"
TASK_ID = "exp7231-board-continuity"
EXPERIMENT_ID = 7231
MILESTONE = "2026.09.636"
SCHEMA = "carnot.exp7231.v636.board_continuity.v1"
RESULT_PATH = Path("results/experiment_7231_v636_board_continuity.json")
RAW_TRANSCRIPT_PATH = Path("results/raw/experiment_7231/polarfire_dispatch.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7231_v636_board_continuity.json")
SPEC_PATH = Path("openspec/capabilities/ising-backend/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
BOARD_UPSTREAM_PATH = Path("results/experiment_7217_v635_abi_board_readiness.json")
CONTROLLER_UPSTREAM_PATH = Path("results/experiment_7226_v636_belief_compiler.json")

KV260_TERMINAL_CRITERION = (
    "board-level programmable-logic latency transcript and successful KV260 synthesis"
)
POLARFIRE_BINARY = "/usr/bin/carnot"
POLARFIRE_INPUT_BYTES = b"/usr/bin/carnot --help\n"
REMOTE_BEGIN = "CARNOT_EXP7231_BEGIN"
REMOTE_END = "CARNOT_EXP7231_END"
CONNECTION_TIMEOUT_S = 10
REMOTE_WORKLOAD_TIMEOUT_S = 60
LOCAL_TIMEOUT_S = 75.0

_REMOTE_SCRIPT = r"""
binary=/usr/bin/carnot
input_sha256=$(printf '%s\n' '/usr/bin/carnot --help' | sha256sum | awk '{print $1}')
printf 'CARNOT_EXP7231_BEGIN\n'
printf 'input_sha256=%s\n' "$input_sha256"
if [ ! -x "$binary" ]; then
    printf 'availability=missing\n'
    printf 'CARNOT_EXP7231_END\n'
    exit 3
fi
binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
output_file=$(mktemp /tmp/carnot-exp7231.XXXXXX)
trap 'rm -f "$output_file"' EXIT HUP INT TERM
started_ns=$(date +%s%N)
timeout 60 "$binary" --help >"$output_file" 2>&1
workload_exit_code=$?
completed_ns=$(date +%s%N)
output_sha256=$(sha256sum "$output_file" | awk '{print $1}')
output_bytes=$(wc -c <"$output_file" | tr -d ' ')
output_base64=$(base64 <"$output_file" | tr -d '\n')
printf 'availability=available\n'
printf 'binary_sha256=%s\n' "$binary_sha256"
printf 'workload_exit_code=%s\n' "$workload_exit_code"
printf 'output_sha256=%s\n' "$output_sha256"
printf 'output_bytes=%s\n' "$output_bytes"
printf 'output_base64=%s\n' "$output_base64"
printf 'elapsed_ns=%s\n' "$((completed_ns - started_ns))"
printf 'CARNOT_EXP7231_END\n'
exit "$workload_exit_code"
""".strip()

POLARFIRE_COMMAND = (
    "ssh",
    "-T",
    "-o",
    "BatchMode=yes",
    "-o",
    f"ConnectTimeout={CONNECTION_TIMEOUT_S}",
    "polarfire",
    _REMOTE_SCRIPT,
)

EXPECTED_TASK_CONTRACT: JsonDict = {
    "id": TASK_ID,
    "title": "KV260 GateMate and PolarFire evidence continuity",
    "track": "hardware",
    "priority": "high",
    "requires_gpu": False,
    "max_turns": 100,
    "estimated_wall_time_min": 35,
    "per_unit_rows": True,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
    "agent_type": "claude",
    "model": "opus",
    "prior_failures": [
        {
            "experiment_id": "exp7217-abi-board-readiness",
            "verdict": (
                "complete: the selected interpreter imported and executed the genuine compiled "
                "sampler, explicit transitions matched the Exp7187 authority, and serialized "
                "state survived a second process. KV260 graduation, GateMate's unchanged "
                "physical-state block, and PolarFire dispatch uncertainty remain separate. This "
                "readiness receipt does not reopen the failed 10x claim or establish device "
                "performance."
            ),
            "addressed_by": (
                "The prior complete host-readiness receipt retained PolarFire "
                "blocked_missing_raw_dispatch_transcript and the GateMate physical block. "
                "Recover one existing PolarFire CPU dispatch with raw bytes; keep GateMate "
                "read-only. No retry of native ABI readiness or sampler performance."
            ),
            "retire_if_same_verdict": True,
        }
    ],
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-hardware-wishlist.md"),
    ROADMAP_PATH,
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/known-issues.md"),
    SPEC_PATH,
    Path("python/carnot/experiment_7217_v635_abi_board_readiness.py"),
    Path("python/carnot/experiment_7226_v636_belief_compiler.py"),
    Path("python/carnot/experiment_7231_v636_board_continuity.py"),
    Path("scripts/experiments/experiment_7190_v633_board_placement_receipt.py"),
    Path("scripts/experiments/experiment_7231_v636_board_continuity.py"),
    Path("tests/python/test_experiment_7231_v636_board_continuity.py"),
    BOARD_UPSTREAM_PATH,
    CONTROLLER_UPSTREAM_PATH,
    Path("results/experiment_7190_v633_board_placement_receipt.json"),
    Path("results/experiment_7203_v634_hardware_correction.json"),
    Path("results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"),
    Path("results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json"),
    Path("results/experiment_3867_polarfire_soc_smoke_v4.json"),
    Path("results/experiment_5861_attached_board_state_receipts.json"),
    Path("results/experiment_6559_gatemate_changed_state_continuity.json"),
    Path("results/experiment_7146_v627_gatemate_changed_state.json"),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed ID binds this evidence to one task.",
    "task_id": "The roadmap task identity prevents another task from supplying this result.",
    "milestone": "The milestone binds this result to the V636 execution contract.",
    "spec_refs": "Requirement and scenario IDs connect tests and evidence to this contract.",
    "field_principles": (
        "Annotate actual values in this map; do not wrap arbitrary dictionaries as "
        "principle/value records."
    ),
    "status": (
        "Write a terminal artifact only when done or externally blocked; running checkpoints "
        "use a different path."
    ),
    "run_date": "Use 20260912 and record actual UTC timestamps, never copy an upstream run date.",
    "started_at_utc": "Record the actual UTC start separately from the fixed execution date.",
    "completed_at_utc": "Record the actual UTC completion separately from the fixed execution date.",
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
    "rows": "Per unit/arm/seed metric, error and abstention for every comparison; retain full denominators.",
    "sample_size_budget": "Planned, attempted, completed, censored and independent units; no silent removal.",
    "random_seed": "Freeze random choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash exact source, inputs, settings and raw unit rows.",
    "gate_check_summary": (
        "Every blocked_* verdict names failed check, upstream, field, expected and observed value."
    ),
    "verifier_is_oracle": (
        "True when correctness authority is reused as the verifier; independent code alone is not "
        "distinct authority."
    ),
    "verdict_class": (
        "Closed enum positive | circular_positive | null | blocked | disqualified | partial. "
        "partial means unfinished own work only."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings; blocked_* for external absence. A "
        "failed acceptance gate forbids positive."
    ),
    "MODEL_SPECS": "Only models actually invoked; [] for CPU/aggregation, mandated Qwen3.8 for every model task.",
    "model_invoked": "True only for actual model execution; upstream model outputs are cached evidence.",
    "board_continuity_complete_score": "All three board dispositions, not all boards successful.",
    "board_rows": "Board, latest source, terminal criterion, exact observation and next prerequisite.",
    "hardware_operations_issued": "Exact bounded commands or empty list.",
    "raw_dispatch_transcript_path": "Actual PolarFire stdout/stderr and hashes, or null with a reason.",
    "raw_dispatch_transcript_absence_reason": "Explain null raw evidence without inventing a result.",
    "dispatch_receipt": "Transport, binary, input, output and timing evidence for the one bounded attempt.",
    "operator_state_receipt": "GateMate physical change source or explicit absence.",
    "operation_map": "Host/board/FPGA/TSU work and unknown quantities distinguished.",
    "topology_fit": "Unknown until an explicit device topology mapping exists.",
    "device_power": "Unknown until a device power measurement exists.",
    "device_speed": "Unknown until a comparable device timing measurement exists.",
    "hardware_performance_claimed": "A CPU smoke and historical receipts do not establish performance.",
    "programmable_logic_sampling_claimed": "PolarFire CPU execution is not FPGA sampling.",
    "purchases_or_vendor_contact": "This audit does not buy hardware or contact a vendor.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep terminal, raw, and checkpoint outputs distinct."""

    artifact: Path
    raw_transcript: Path
    checkpoint: Path

    @classmethod
    def defaults(cls, root: Path = REPO_ROOT) -> ExperimentPaths:
        """Resolve production destinations under one explicit repository root."""

        return cls(root / RESULT_PATH, root / RAW_TRANSCRIPT_PATH, root / CHECKPOINT_PATH)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Give tests and replays private destinations that cannot rewrite research state."""

        return cls(
            root / RESULT_PATH,
            root / RAW_TRANSCRIPT_PATH,
            root / CHECKPOINT_PATH,
        )


def progress(phase: int, boundary: str, operation: str) -> None:
    """Flush a numbered boundary so a slow transport never looks silent."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def canonical_json(value: Any) -> str:
    """Use the shipped stable JSON encoding for all evidence hashes."""

    return exp7217.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes and retain the algorithm name."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Use the shipped streaming file hash implementation."""

    return exp7217.sha256_file(path)


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum that stores this digest."""

    return exp7217.artifact_checksum(payload)


def unwrap_principled_value(value: Any) -> Any:
    """Unwrap only the exact two-key producer representation."""

    return exp7217.unwrap_principled_value(value)


def check(
    name: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Give every precondition one stable diagnostic shape."""

    return {
        "check": name,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _read_json(path: Path) -> JsonDict:
    """Return an empty object when optional bytes are absent or malformed."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _task_contract(root: Path) -> JsonDict | None:
    """Read only the frozen roadmap fields that control this task."""

    try:
        roadmap = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    tasks = roadmap.get("tasks") if isinstance(roadmap, Mapping) else None
    if not isinstance(tasks, list):
        return None
    for task in tasks:
        if isinstance(task, Mapping) and task.get("id") == TASK_ID:
            return {key: deepcopy(task.get(key)) for key in EXPECTED_TASK_CONTRACT}
    return None


def _writable_destination(path: Path) -> bool:
    """Probe a destination directory and remove the probe immediately."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp7231-write-", dir=path.parent)
        os.close(descriptor)
        Path(name).unlink()
    except OSError:
        return False
    return True


def _manifest_match(value: Any, experiment_number: str) -> bool:
    """Reuse the identifier-only exclusion search from the board producer."""

    return exp7217._manifest_mentions_experiment(value, experiment_number)


def collect_preconditions(
    root: Path,
    paths: ExperimentPaths,
    *,
    board_upstream_path: Path | None = None,
    controller_upstream_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str], dict[str, JsonDict]]:
    """Print first, then authenticate sources, gates, imports, and outputs."""

    print("[phase 0 check start] required source bytes", flush=True)
    checks: list[JsonDict] = []
    sizes = {
        path.as_posix(): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    hashes = {
        path.as_posix(): sha256_file(root / path)
        for path in REQUIRED_SOURCE_PATHS
        if sizes[path.as_posix()] not in (None, 0)
    }
    checks.append(
        check(
            "required_source_bytes",
            "repository",
            "REQUIRED_SOURCE_PATHS",
            "all nonempty",
            sizes,
            all(size is not None and size > 0 for size in sizes.values()),
        )
    )

    print("[phase 0 check start] driving specification and roadmap", flush=True)
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    spec_observed = {
        "requirement": "REQ-ISING-7231" in spec,
        "scenarios": "SCENARIO-ISING-7231-" in spec,
    }
    checks.append(
        check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-ISING-7231 and scenarios",
            {"requirement": True, "scenarios": True},
            spec_observed,
            all(spec_observed.values()),
        )
    )
    contract = _task_contract(root)
    checks.append(
        check(
            "roadmap_task_contract",
            ROADMAP_PATH.as_posix(),
            "exp7231 task fields",
            EXPECTED_TASK_CONTRACT,
            contract,
            contract == EXPECTED_TASK_CONTRACT,
        )
    )

    print("[phase 0 check start] imports, SSH, raw, checkpoint, and result outputs", flush=True)
    tools = {
        "python": str(Path(os.sys.executable).absolute()),
        "yaml": yaml.__version__,
        "ssh": shutil.which("ssh"),
        "artifact_writable": _writable_destination(paths.artifact),
        "raw_writable": _writable_destination(paths.raw_transcript),
        "checkpoint_writable": _writable_destination(paths.checkpoint),
        "exp7217_import": callable(exp7217.validate_artifact),
        "exp7226_import": callable(exp7226.validate_artifact),
    }
    checks.append(
        check(
            "imports_tools_and_outputs",
            "host",
            "python,yaml,ssh,raw,checkpoint,result",
            "all available",
            tools,
            bool(tools["ssh"])
            and all(
                tools[name] is True
                for name in (
                    "artifact_writable",
                    "raw_writable",
                    "checkpoint_writable",
                    "exp7217_import",
                    "exp7226_import",
                )
            ),
        )
    )

    print("[phase 0 check start] exclusion manifest before producer gates", flush=True)
    try:
        exclusion = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        exclusion = None
    board_path = board_upstream_path or root / BOARD_UPSTREAM_PATH
    controller_path = controller_upstream_path or root / CONTROLLER_UPSTREAM_PATH
    board = _read_json(board_path)
    controller = _read_json(controller_path)
    board_quarantine = exp7217.upstream_quarantine_observation(
        board,
        manifest_match=_manifest_match(exclusion, "7217"),
    )
    controller_quarantine = exp7217.upstream_quarantine_observation(
        controller,
        manifest_match=_manifest_match(exclusion, "7226"),
    )
    for label, upstream_path, quarantine in (
        ("exp7217", board_path, board_quarantine),
        ("exp7226", controller_path, controller_quarantine),
    ):
        checks.append(
            check(
                f"{label}_not_quarantined",
                str(upstream_path),
                "quarantined",
                False,
                quarantine,
                quarantine["quarantined"] is False,
            )
        )

    print("[phase 0 check start] Exp7217 board producer authentication", flush=True)
    board_auth = (
        exp7217.validate_artifact(board)
        if board and board_quarantine["quarantined"] is False
        else ["not_authenticated"]
    )
    checks.append(
        check(
            "exp7217_producer_authentication",
            str(board_path),
            "shipped_validator_errors",
            [],
            board_auth,
            board_auth == [],
        )
    )
    board_gate = exp7217.gated_upstream_value(
        board, board_quarantine, "abi_board_receipt_complete_score"
    )
    if board_gate == "not_consumed_due_to_quarantine":
        board_observed: Any = board_gate
    else:
        names = sorted(
            row.get("board")
            for row in board.get("board_rows", [])
            if isinstance(row, Mapping) and isinstance(row.get("board"), str)
        )
        board_observed = {"score": board_gate, "boards": names}
    checks.append(
        check(
            "exp7217_board_gate",
            str(board_path),
            "abi_board_receipt_complete_score and boards",
            {"score": 1, "boards": ["GateMate", "KV260", "PolarFire"]},
            board_observed,
            board_auth == []
            and board_observed
            == {
                "score": 1,
                "boards": ["GateMate", "KV260", "PolarFire"],
            },
        )
    )

    print("[phase 0 check start] Exp7226 compact-controller authentication", flush=True)
    controller_auth = (
        exp7226.validate_artifact(controller)
        if controller and controller_quarantine["quarantined"] is False
        else ["not_authenticated"]
    )
    checks.append(
        check(
            "exp7226_producer_authentication",
            str(controller_path),
            "shipped_validator_errors",
            [],
            controller_auth,
            controller_auth == [],
        )
    )
    controller_gate = exp7217.gated_upstream_value(
        controller, controller_quarantine, "belief_compiler_ready_score"
    )
    future = controller.get("future_hardware_path", {})
    controller_observed = (
        "not_consumed_due_to_quarantine"
        if controller_gate == "not_consumed_due_to_quarantine"
        else {
            "belief_compiler_ready_score": controller_gate,
            "cpu_packed_kernel_bytes": future.get("cpu_packed_kernel_bytes")
            if isinstance(future, Mapping)
            else None,
            "fpga_table_bits": future.get("fpga_table_bits")
            if isinstance(future, Mapping)
            else None,
        }
    )
    expected_controller = {
        "belief_compiler_ready_score": 1,
        "cpu_packed_kernel_bytes": 152,
        "fpga_table_bits": 1216,
    }
    checks.append(
        check(
            "exp7226_controller_gate",
            str(controller_path),
            "readiness and compact footprint",
            expected_controller,
            controller_observed,
            controller_auth == [] and controller_observed == expected_controller,
        )
    )
    return checks, hashes, {"board": board, "controller": controller}


def _first_failed(checks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the first failed observation so a block has one exact cause."""

    return next((row for row in checks if row.get("passed") is not True), None)


def _finish_row(row: JsonDict) -> JsonDict:
    """Add full-denominator fields and hash the final row contents."""

    row.setdefault("arm", "authenticated_receipt_continuity")
    row.setdefault("seed", None)
    row.setdefault("metric", False)
    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row.pop("row_sha256", None)
    row["row_sha256"] = exp7217.sha256_json(row)
    return row


def load_latest_board_rows(
    root: Path, upstream: Mapping[str, Any]
) -> tuple[list[JsonDict], JsonDict]:
    """Select and authenticate each board row from the latest terminal producer."""

    source_rows = upstream.get("board_rows")
    if not isinstance(source_rows, list):
        raise ValueError("Exp7217 board_rows must be a list")
    latest_hash = sha256_file(root / BOARD_UPSTREAM_PATH)
    latest_date = upstream.get("run_date")
    rows: list[JsonDict] = []
    for source in source_rows:
        if not isinstance(source, Mapping) or source.get("board") not in {
            "KV260",
            "GateMate",
            "PolarFire",
        }:
            raise ValueError("Exp7217 board row is malformed")
        row = deepcopy(dict(source))
        evidence_path = root / str(row.get("evidence_path", ""))
        if not evidence_path.is_file() or row.get("source_hash") != sha256_file(evidence_path):
            raise ValueError(f"board evidence hash mismatch: {evidence_path}")
        board = str(row["board"])
        row.update(
            {
                "unit_id": f"board:{board}",
                "latest_receipt_path": BOARD_UPSTREAM_PATH.as_posix(),
                "latest_receipt_date": latest_date,
                "latest_receipt_hash": latest_hash,
                "latest_receipt_authenticated": True,
                "execution_venue": board.lower(),
                "processor_class": {
                    "KV260": "fpga_fabric",
                    "GateMate": "unavailable",
                    "PolarFire": "cpu",
                }[board],
                "hardware_operations_issued": [],
                "hardware_command_count": 0,
            }
        )
        rows.append(_finish_row(row))
    by_board = {row["board"]: row for row in rows}
    if set(by_board) != {"KV260", "GateMate", "PolarFire"}:
        raise ValueError("Exp7217 must supply exactly three board rows")
    if (
        by_board["KV260"].get("disposition") != "graduated_preserved"
        or by_board["KV260"].get("terminal_criterion") != KV260_TERMINAL_CRITERION
        or by_board["GateMate"].get("disposition") != "blocked_inherited_no_new_physical_state"
    ):
        raise ValueError("authenticated board disposition changed")
    operator_source = upstream.get("operator_state_receipt")
    if not isinstance(operator_source, Mapping):
        raise ValueError("Exp7217 operator_state_receipt is malformed")
    operator = {
        **deepcopy(dict(operator_source)),
        "compared_to_experiment": "Exp6559",
        "hardware_operations_issued": [],
        "hardware_command_count": 0,
    }
    return rows, operator


class _Heartbeat:
    """Report only elapsed waiting time while a native transport is blocked."""

    def __init__(self, operation: str, interval_s: float) -> None:
        self.operation = operation
        self.interval_s = interval_s
        self.started = 0.0
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> _Heartbeat:
        self.started = time.monotonic()

        def report() -> None:
            while not self.stop.wait(self.interval_s):
                elapsed = time.monotonic() - self.started
                print(
                    f"[heartbeat] operation={self.operation} state=waiting elapsed_s={elapsed:.1f}",
                    flush=True,
                )

        self.thread = threading.Thread(target=report, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=max(self.interval_s * 2, 0.1))


def _run_subprocess(
    command: tuple[str, ...],
    timeout_s: float,
    *,
    heartbeat_interval_s: float = 30.0,
) -> JsonDict:
    """Stream exact child bytes, report elapsed waits, and enforce one deadline."""

    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    observed: queue.Queue[tuple[str, bytes]] = queue.Queue()

    def pump(name: str, stream: Any) -> None:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            chunks[name].append(chunk)
            observed.put((name, chunk))

    readers = [
        threading.Thread(target=pump, args=(name, stream), daemon=True)
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr))
    ]
    for reader in readers:
        reader.start()
    timed_out = False
    with _Heartbeat("PolarFire SSH CPU smoke", heartbeat_interval_s):
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
    for reader in readers:
        reader.join()
    while not observed.empty():
        name, chunk = observed.get_nowait()
        text = chunk.decode("utf-8", errors="replace")
        print(f"[subprocess {name}] {text}", end="" if text.endswith("\n") else "\n", flush=True)
    return {
        "command": list(command),
        "exit_code": process.returncode,
        "stdout": b"".join(chunks["stdout"]),
        "stderr": b"".join(chunks["stderr"]),
        "timed_out": timed_out,
        "transport_duration_s": time.monotonic() - started,
    }


def _tag_hash(value: Any) -> str | None:
    """Normalize a bare or tagged SHA-256 without accepting another shape."""

    if not isinstance(value, str):
        return None
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        return None
    return "sha256:" + digest


def _remote_fields(stdout: bytes) -> JsonDict:
    """Parse only the delimited key-value block from the remote transport."""

    text = stdout.decode("utf-8", errors="replace")
    lines = text.splitlines()
    try:
        start = lines.index(REMOTE_BEGIN)
        end = lines.index(REMOTE_END, start + 1)
    except ValueError:
        return {}
    fields: JsonDict = {}
    for line in lines[start + 1 : end]:
        if "=" not in line:
            return {}
        key, value = line.split("=", 1)
        fields[key] = value
    return fields


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write raw or terminal evidence through one same-directory replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_polarfire_smoke(
    raw_path: Path,
    *,
    command_runner: CommandRunner = _run_subprocess,
) -> JsonDict:
    """Run one conditional deployed binary and retain exact transport bytes."""

    transport = command_runner(POLARFIRE_COMMAND, LOCAL_TIMEOUT_S)
    stdout = transport.get("stdout", b"")
    stderr = transport.get("stderr", b"")
    stdout_bytes = stdout if isinstance(stdout, bytes) else str(stdout).encode()
    stderr_bytes = stderr if isinstance(stderr, bytes) else str(stderr).encode()
    fields = _remote_fields(stdout_bytes)
    block_reason: str | None = None
    output = b""
    output_hash = _tag_hash(fields.get("output_sha256"))
    input_hash = _tag_hash(fields.get("input_sha256"))
    binary_hash = _tag_hash(fields.get("binary_sha256"))
    try:
        output = base64.b64decode(str(fields.get("output_base64", "")), validate=True)
    except (ValueError, base64.binascii.Error):
        output = b""
    expected_input_hash = sha256_bytes(POLARFIRE_INPUT_BYTES)
    input_match = input_hash == expected_input_hash
    output_match = output_hash is not None and output_hash == sha256_bytes(output)
    try:
        workload_exit = int(fields["workload_exit_code"])
    except (KeyError, TypeError, ValueError):
        workload_exit = None
    if transport.get("timed_out") is True:
        block_reason = "transport_timeout"
    elif not fields:
        block_reason = (
            "ssh_transport_unavailable"
            if transport.get("exit_code") not in (0, None)
            else "invalid_dispatch_transcript"
        )
    elif fields.get("availability") == "missing":
        block_reason = "missing_deployed_workload"
    elif fields.get("availability") != "available" or not (
        binary_hash and input_match and output_match
    ):
        block_reason = "invalid_dispatch_transcript"
    elif workload_exit != 0:
        block_reason = "deployed_workload_failed"
    dispatch_completed = block_reason is None
    raw = {
        "schema": "carnot.exp7231.polarfire_raw_dispatch.v1",
        "run_date": RUN_DATE,
        "command": list(POLARFIRE_COMMAND),
        "connection_timeout_s": CONNECTION_TIMEOUT_S,
        "remote_workload_timeout_s": REMOTE_WORKLOAD_TIMEOUT_S,
        "local_timeout_s": LOCAL_TIMEOUT_S,
        "transport_exit_code": transport.get("exit_code"),
        "transport_duration_s": transport.get("transport_duration_s"),
        "transport_timed_out": transport.get("timed_out") is True,
        "transport_stdout": stdout_bytes.decode("utf-8", errors="replace"),
        "transport_stderr": stderr_bytes.decode("utf-8", errors="replace"),
        "transport_stdout_base64": base64.b64encode(stdout_bytes).decode(),
        "transport_stderr_base64": base64.b64encode(stderr_bytes).decode(),
        "transport_stdout_sha256": sha256_bytes(stdout_bytes),
        "transport_stderr_sha256": sha256_bytes(stderr_bytes),
        "deployed_binary": POLARFIRE_BINARY if fields.get("availability") == "available" else None,
        "binary_sha256": binary_hash,
        "input_sha256": input_hash,
        "expected_input_sha256": expected_input_hash,
        "input_hash_matches": input_match,
        "workload_exit_code": workload_exit,
        "workload_elapsed_ns": int(fields["elapsed_ns"])
        if str(fields.get("elapsed_ns", "")).isdigit()
        else None,
        "workload_output_base64": base64.b64encode(output).decode(),
        "workload_output_sha256": output_hash,
        "workload_output_bytes": len(output),
        "output_hash_matches": output_match,
        "dispatch_completed": dispatch_completed,
        "block_reason": block_reason,
        "execution_venue": "polarfire",
        "processor_class": "cpu",
        "programmable_logic_sampling_observed": False,
        "persistent_configuration_changed": False,
        "upload_or_install_performed": False,
    }
    _atomic_json(raw_path, raw)
    return {**raw, "raw_path": str(raw_path), "raw_sha256": sha256_file(raw_path)}


def operation_map(controller: Mapping[str, Any], *, polarfire_executed: bool) -> list[JsonDict]:
    """Keep measured CPU work and possible device table paths separate."""

    future = controller.get("future_hardware_path", {})
    future_map = future if isinstance(future, Mapping) else {}
    memory_bytes = future_map.get("cpu_packed_kernel_bytes")
    memory_bits = future_map.get("fpga_table_bits")
    return [
        {
            "target": "host_cpu",
            "operation": "receipt authentication and compact-controller table lookup",
            "executed_here": True,
            "memory_table_bytes": memory_bytes,
            "memory_table_bits": memory_bits,
            "topology_fit": "unknown",
            "power": "unknown",
            "speed": "unknown",
        },
        {
            "target": "polarfire_cpu",
            "operation": "already-deployed /usr/bin/carnot --help smoke",
            "executed_here": polarfire_executed,
            "memory_table_bytes": "unknown",
            "memory_table_bits": "unknown",
            "topology_fit": "unknown",
            "power": "unknown",
            "speed": "unknown",
        },
        {
            "target": "fpga_fabric",
            "operation": future_map.get("fpga_table_path", "unknown"),
            "executed_here": False,
            "memory_table_bytes": memory_bytes,
            "memory_table_bits": memory_bits,
            "topology_fit": "unknown",
            "power": "unknown",
            "speed": "unknown",
        },
        {
            "target": "tsu",
            "operation": "unmapped",
            "executed_here": False,
            "memory_table_bytes": "unknown",
            "memory_table_bits": "unknown",
            "topology_fit": "unknown",
            "power": "unknown",
            "speed": "unknown",
        },
    ]


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve all checks and expose one exact terminal failure."""

    failed = _first_failed(checks)
    if failed is None:
        return {
            "passed": True,
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": None,
            "observed_value": None,
            "checks": [dict(row) for row in checks],
        }
    return {
        "passed": False,
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "field": failed.get("field"),
        "expected_value": failed.get("expected_value"),
        "observed_value": failed.get("observed_value"),
        "checks": [dict(row) for row in checks],
    }


def _base_artifact(
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
) -> JsonDict:
    """Create every required field before terminal state is selected."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "spec_refs": [
            "REQ-ISING-7231",
            "SCENARIO-ISING-7231-PREFLIGHT",
            "SCENARIO-ISING-7231-BOARDS",
            "SCENARIO-ISING-7231-GATEMATE",
            "SCENARIO-ISING-7231-POLARFIRE",
            "SCENARIO-ISING-7231-PLACEMENT",
            "SCENARIO-ISING-7231-ARTIFACT",
        ],
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node(),
        "duration_s": duration_s,
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": {
            "planned": 3,
            "attempted": 0,
            "completed": 0,
            "censored": 0,
            "independent_units_planned": 3,
            "independent_units_completed": 0,
            "polarfire_dispatch_planned": 1,
            "polarfire_dispatch_attempted": 0,
            "polarfire_dispatch_completed": 0,
        },
        "random_seed": EXPERIMENT_ID,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial: terminal state not selected",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "board_continuity_complete_score": 0,
        "board_rows": [],
        "hardware_operations_issued": [],
        "raw_dispatch_transcript_path": None,
        "raw_dispatch_transcript_absence_reason": "not_attempted",
        "dispatch_receipt": None,
        "operator_state_receipt": None,
        "operation_map": [],
        "topology_fit": "unknown",
        "device_power": "unknown",
        "device_speed": "unknown",
        "hardware_performance_claimed": False,
        "programmable_logic_sampling_claimed": False,
        "purchases_or_vendor_contact": [],
    }


def _blocked_artifact(
    *,
    started_at: str,
    started: float,
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
) -> JsonDict:
    """Finish one precondition block without any board command or row."""

    failed = _first_failed(checks)
    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        started_at=started_at,
        completed_at=completed_at,
        duration_s=time.monotonic() - started,
        checks=checks,
        hashes=hashes,
    )
    artifact.update(
        {
            "status": "blocked_external_precondition",
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_external:{failed.get('check') if failed else 'unknown'}",
            "raw_dispatch_transcript_absence_reason": "blocked_before_board_access",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _apply_polarfire_receipt(rows: list[JsonDict], receipt: Mapping[str, Any]) -> None:
    """Replace only the PolarFire observation with this run's CPU outcome."""

    row = next(item for item in rows if item["board"] == "PolarFire")
    completed = receipt.get("dispatch_completed") is True
    reason = receipt.get("block_reason")
    row.update(
        {
            "hardware_operations_issued": [list(POLARFIRE_COMMAND)],
            "hardware_command_count": 1,
            "raw_dispatch_transcript_path": receipt.get("raw_path"),
            "raw_dispatch_transcript_hash": receipt.get("raw_sha256"),
            "dispatch_receipt": {
                "dispatch_completed": completed,
                "block_reason": reason,
                "binary_sha256": receipt.get("binary_sha256"),
                "input_sha256": receipt.get("input_sha256"),
                "input_hash_matches": receipt.get("input_hash_matches"),
                "output_sha256": receipt.get("workload_output_sha256"),
                "output_hash_matches": receipt.get("output_hash_matches"),
                "transport_exit_code": receipt.get("transport_exit_code"),
                "transport_duration_s": receipt.get("transport_duration_s"),
                "workload_exit_code": receipt.get("workload_exit_code"),
                "workload_elapsed_ns": receipt.get("workload_elapsed_ns"),
                "execution_venue": "polarfire",
                "processor_class": "cpu",
            },
            "disposition": (
                "terminal_cpu_dispatch_raw_transcript_retained"
                if completed
                else f"blocked_{reason}"
            ),
            "terminal_criterion_met": completed,
            "last_observed_value": (
                "bounded_deployed_cpu_binary_executed_and_raw_hashes_verified"
                if completed
                else f"blocked_{reason}"
            ),
            "exact_next_prerequisite": (
                "none for CPU dispatch continuity; programmable-logic sampling remains separate"
                if completed
                else {
                    "missing_deployed_workload": "restore the already deployed /usr/bin/carnot workload",
                    "ssh_transport_unavailable": "restore ssh polarfire reachability",
                    "transport_timeout": "restore bounded ssh polarfire responsiveness",
                    "deployed_workload_failed": "repair the existing deployed workload without installing during this audit",
                    "invalid_dispatch_transcript": "return the required binary, input, output, and timing hashes",
                }.get(str(reason), "supply one authenticated bounded CPU dispatch transcript")
            ),
            "metric": completed,
            "error": None if completed else str(reason),
            "abstention": not completed,
            "board_cpu_work_observed": completed,
            "programmable_logic_sampling_observed": False,
            "processor_class": "cpu",
            "execution_venue": "polarfire",
        }
    )
    if completed:
        row.update(
            {
                "latest_receipt_path": receipt.get("raw_path"),
                "latest_receipt_date": RUN_DATE,
                "latest_receipt_hash": receipt.get("raw_sha256"),
                "latest_receipt_authenticated": True,
            }
        )
    _finish_row(row)


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    *,
    command_runner: CommandRunner = _run_subprocess,
) -> JsonDict:
    """Authenticate inputs, run one bounded smoke, and assemble terminal evidence."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "preconditions before any board access")
    checks, hashes, upstreams = collect_preconditions(root, paths)
    progress(0, "end", f"preconditions failed={int(_first_failed(checks) is not None)}")
    if _first_failed(checks) is not None:
        return _blocked_artifact(
            started_at=started_at,
            started=started,
            checks=checks,
            hashes=hashes,
        )

    progress(1, "start", "independent authenticated board receipt selection")
    rows, operator = load_latest_board_rows(root, upstreams["board"])
    progress(1, "end", f"authenticated board dispositions={len(rows)}")

    progress(2, "start", "GateMate Exp6559 changed-state boundary")
    if operator.get("newer_than_exp6559") is True:
        gate_row = next(row for row in rows if row["board"] == "GateMate")
        gate_row.update(
            {
                "disposition": "authorized_later_action",
                "exact_next_prerequisite": (
                    "in a later task, run one bounded detect action authorized by the operator receipt"
                ),
                "last_observed_value": "new_operator_physical_state_receipt_recorded",
            }
        )
        _finish_row(gate_row)
    progress(2, "end", "GateMate commands issued=0")

    progress(3, "before", "one conditional PolarFire SSH CPU smoke")
    receipt = run_polarfire_smoke(paths.raw_transcript, command_runner=command_runner)
    progress(
        3,
        "after",
        "PolarFire dispatch "
        + ("completed" if receipt["dispatch_completed"] else f"blocked={receipt['block_reason']}"),
    )
    _apply_polarfire_receipt(rows, receipt)
    hashes[str(paths.raw_transcript)] = receipt["raw_sha256"]

    progress(4, "start", "host, board CPU, FPGA, and TSU operation mapping")
    mapping = operation_map(
        upstreams["controller"],
        polarfire_executed=receipt["dispatch_completed"] is True,
    )
    progress(4, "end", "topology, power, and speed remain unknown")

    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        started_at=started_at,
        completed_at=completed_at,
        duration_s=time.monotonic() - started,
        checks=checks,
        hashes=hashes,
    )
    dispatch_completed = receipt["dispatch_completed"] is True
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": (
                "cpu_exact_solver_or_simulator" if dispatch_completed else "aggregation"
            ),
            "inference_substrate_class": (
                "cpu_exact_solver_or_simulator" if dispatch_completed else "aggregation"
            ),
            "rows": rows,
            "sample_size_budget": {
                "planned": 3,
                "attempted": 3,
                "completed": 3,
                "censored": 0,
                "independent_units_planned": 3,
                "independent_units_completed": 3,
                "polarfire_dispatch_planned": 1,
                "polarfire_dispatch_attempted": 1,
                "polarfire_dispatch_completed": int(dispatch_completed),
            },
            "verdict_class": "positive",
            "honest_verdict": (
                "complete: all three board dispositions are authenticated. KV260 graduation "
                "and its exact terminal criterion remain preserved. GateMate remains read-only "
                "without a post-Exp6559 physical-state receipt. "
                + (
                    "The deployed PolarFire CPU binary executed once and its raw transcript and "
                    "hashes are retained. This is not programmable-logic sampling."
                    if dispatch_completed
                    else f"PolarFire remains {receipt['block_reason']}; no result was invented."
                )
            ),
            "board_continuity_complete_score": 1,
            "board_rows": rows,
            "hardware_operations_issued": [
                {
                    "board": "PolarFire",
                    "execution_venue": "polarfire",
                    "processor_class": "cpu",
                    "command": list(POLARFIRE_COMMAND),
                    "connection_timeout_s": CONNECTION_TIMEOUT_S,
                    "remote_workload_timeout_s": REMOTE_WORKLOAD_TIMEOUT_S,
                    "persistent_configuration_changed": False,
                }
            ],
            "raw_dispatch_transcript_path": str(paths.raw_transcript),
            "raw_dispatch_transcript_absence_reason": None,
            "dispatch_receipt": {
                key: receipt.get(key)
                for key in (
                    "dispatch_completed",
                    "block_reason",
                    "raw_path",
                    "raw_sha256",
                    "transport_exit_code",
                    "transport_duration_s",
                    "transport_timed_out",
                    "binary_sha256",
                    "input_sha256",
                    "input_hash_matches",
                    "workload_exit_code",
                    "workload_elapsed_ns",
                    "workload_output_sha256",
                    "workload_output_bytes",
                    "output_hash_matches",
                    "execution_venue",
                    "processor_class",
                    "programmable_logic_sampling_observed",
                    "persistent_configuration_changed",
                    "upload_or_install_performed",
                )
            },
            "operator_state_receipt": operator,
            "operation_map": mapping,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(5, "end", "terminal artifact assembled")
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> list[str]:
    """Recompute terminal state, board boundaries, hashes, and footprint."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id")
    add(artifact.get("task_id") != TASK_ID or artifact.get("milestone") != MILESTONE, "identity")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(
        artifact.get("execution_venue") != "host" or not artifact.get("execution_host"),
        "execution_identity",
    )
    add(
        not isinstance(artifact.get("duration_s"), (int, float)) or artifact["duration_s"] < 0,
        "duration",
    )
    for name in ("started_at_utc", "completed_at_utc"):
        try:
            datetime.fromisoformat(str(artifact.get(name)))
        except ValueError:
            add(True, name)
    add(
        artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False,
        "model_declaration",
    )
    add(artifact.get("verifier_is_oracle") is not False, "verifier_authority")
    add(
        artifact.get("topology_fit") != "unknown"
        or artifact.get("device_power") != "unknown"
        or artifact.get("device_speed") != "unknown"
        or artifact.get("hardware_performance_claimed") is not False
        or artifact.get("programmable_logic_sampling_claimed") is not False,
        "hardware_claim_boundary",
    )
    add(artifact.get("purchases_or_vendor_contact") != [], "external_action_boundary")
    try:
        checksum_valid = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_valid = False
    add(not checksum_valid, "reproducibility_checksum")

    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        summary = artifact.get("gate_check_summary", {})
        add(artifact.get("status") != "blocked_external_precondition", "blocked_status")
        add(
            artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run",
            "blocked_substrate",
        )
        add(artifact.get("rows") != [] or artifact.get("board_rows") != [], "blocked_rows")
        add(artifact.get("hardware_operations_issued") != [], "blocked_hardware_operation")
        add(artifact.get("board_continuity_complete_score") != 0, "blocked_completion_score")
        add(
            not isinstance(summary, Mapping)
            or summary.get("passed") is not False
            or any(
                summary.get(name) is None
                for name in (
                    "failed_check",
                    "upstream",
                    "field",
                    "expected_value",
                    "observed_value",
                )
            ),
            "blocked_gate_summary",
        )
        return errors

    rows = artifact.get("board_rows", [])
    by_board = (
        {
            row.get("board"): row
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("board"), str)
        }
        if isinstance(rows, list)
        else {}
    )
    row_schema = all(
        {"unit_id", "arm", "seed", "metric", "error", "abstention", "row_sha256"} <= set(row)
        and row.get("row_sha256")
        == exp7217.sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        and row.get("latest_receipt_authenticated") is True
        and row.get("latest_receipt_date")
        and row.get("latest_receipt_hash")
        for row in by_board.values()
    )
    board_valid = (
        set(by_board) == {"KV260", "GateMate", "PolarFire"}
        and row_schema
        and by_board["KV260"].get("disposition") == "graduated_preserved"
        and by_board["KV260"].get("terminal_criterion") == KV260_TERMINAL_CRITERION
        and by_board["KV260"].get("hardware_command_count") == 0
        and by_board["GateMate"].get("hardware_command_count") == 0
        and by_board["GateMate"].get("disposition")
        in {"blocked_inherited_no_new_physical_state", "authorized_later_action"}
        and by_board["PolarFire"].get("processor_class") == "cpu"
        and by_board["PolarFire"].get("programmable_logic_sampling_observed") is False
        and by_board["PolarFire"].get("hardware_command_count") == 1
    )
    add(not board_valid, "board_rows")
    add(artifact.get("rows") != rows, "rows")
    add(artifact.get("board_continuity_complete_score") != int(board_valid), "completion_score")
    operator = artifact.get("operator_state_receipt", {})
    add(
        not isinstance(operator, Mapping)
        or operator.get("hardware_operations_issued") != []
        or operator.get("hardware_command_count") != 0,
        "operator_state_receipt",
    )
    operations = artifact.get("hardware_operations_issued", [])
    operation_valid = (
        isinstance(operations, list)
        and len(operations) == 1
        and operations[0].get("board") == "PolarFire"
        and operations[0].get("processor_class") == "cpu"
        and operations[0].get("command") == list(POLARFIRE_COMMAND)
        and operations[0].get("connection_timeout_s") == CONNECTION_TIMEOUT_S
        and operations[0].get("remote_workload_timeout_s") == REMOTE_WORKLOAD_TIMEOUT_S
        and operations[0].get("persistent_configuration_changed") is False
    )
    add(not operation_valid, "hardware_operations")

    receipt = artifact.get("dispatch_receipt", {})
    dispatch_completed = isinstance(receipt, Mapping) and receipt.get("dispatch_completed") is True
    receipt_valid = (
        isinstance(receipt, Mapping)
        and receipt.get("raw_path") == artifact.get("raw_dispatch_transcript_path")
        and isinstance(receipt.get("raw_sha256"), str)
        and receipt.get("execution_venue") == "polarfire"
        and receipt.get("processor_class") == "cpu"
        and receipt.get("programmable_logic_sampling_observed") is False
        and receipt.get("persistent_configuration_changed") is False
        and receipt.get("upload_or_install_performed") is False
    )
    if dispatch_completed:
        receipt_valid = receipt_valid and all(
            (
                receipt.get("binary_sha256"),
                receipt.get("input_hash_matches") is True,
                receipt.get("output_hash_matches") is True,
                receipt.get("workload_exit_code") == 0,
            )
        )
    else:
        receipt_valid = receipt_valid and bool(receipt.get("block_reason"))
    add(not receipt_valid, "dispatch_receipt")
    expected_substrate = "cpu_exact_solver_or_simulator" if dispatch_completed else "aggregation"
    add(
        artifact.get("inference_substrate") != expected_substrate
        or artifact.get("inference_substrate_class") != expected_substrate,
        "substrate",
    )
    mapping = artifact.get("operation_map", [])
    mapped = (
        {
            row.get("target"): row
            for row in mapping
            if isinstance(row, Mapping) and isinstance(row.get("target"), str)
        }
        if isinstance(mapping, list)
        else {}
    )
    mapping_valid = (
        set(mapped) == {"host_cpu", "polarfire_cpu", "fpga_fabric", "tsu"}
        and mapped["host_cpu"].get("memory_table_bytes") == 152
        and mapped["fpga_fabric"].get("memory_table_bits") == 1216
        and mapped["polarfire_cpu"].get("executed_here") is dispatch_completed
        and all(row.get("topology_fit") == "unknown" for row in mapped.values())
        and all(row.get("power") == "unknown" for row in mapped.values())
        and all(row.get("speed") == "unknown" for row in mapped.values())
    )
    add(not mapping_valid, "operation_map")
    budget = artifact.get("sample_size_budget", {})
    add(
        not isinstance(budget, Mapping)
        or budget.get("planned") != 3
        or budget.get("attempted") != 3
        or budget.get("completed") != 3
        or budget.get("censored") != 0
        or budget.get("independent_units_planned") != 3
        or budget.get("independent_units_completed") != 3
        or budget.get("polarfire_dispatch_attempted") != 1
        or budget.get("polarfire_dispatch_completed") != int(dispatch_completed),
        "sample_size_budget",
    )
    add(
        artifact.get("status") != "complete"
        or artifact.get("verdict_class") != "positive"
        or not str(artifact.get("honest_verdict", "")).startswith("complete:")
        or artifact.get("gate_check_summary", {}).get("passed") is not True,
        "terminal_state",
    )
    if root is not None:
        hashes = artifact.get("source_artifact_hashes", {})
        add(
            not isinstance(hashes, Mapping)
            or any(
                not (Path(path) if Path(path).is_absolute() else root / path).is_file()
                or sha256_file(Path(path) if Path(path).is_absolute() else root / path) != digest
                for path, digest in hashes.items()
            ),
            "source_artifact_hashes",
        )
    return errors


def atomic_write(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and atomically publish only terminal evidence."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp7231 artifact: {errors}")
    _atomic_json(path, artifact)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(
    root: Path,
    paths: ExperimentPaths,
    *,
    command_runner: CommandRunner = _run_subprocess,
) -> JsonDict:
    """Build, independently validate, and publish the V636 continuity receipt."""

    artifact = build_artifact(root, paths, command_runner=command_runner)
    progress(6, "before", "final artifact validation")
    errors = validate_artifact(artifact, root=root)
    progress(6, "after", f"final artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7231 artifact: {errors}")
    progress(7, "before", "atomic terminal write")
    receipt = atomic_write(paths.artifact, artifact)
    progress(7, "after", f"atomic terminal write bytes={receipt['bytes']}")
    return artifact


def _parser() -> argparse.ArgumentParser:
    """Keep direct execution and read-only validation on one parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-transcript", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded audit or validate existing bytes without mutation."""

    args = _parser().parse_args(argv)
    try:
        if args.validate is not None:
            progress(6, "before", f"read-only validation path={args.validate}")
            artifact = _read_json(args.validate)
            errors = ["artifact_not_json_object"] if not artifact else validate_artifact(artifact)
            progress(6, "after", f"read-only validation errors={len(errors)}")
            if errors:
                print(f"validation_failed errors={errors}", flush=True)
                return 2
            print("validation_passed", flush=True)
            return 0
        if args.date != RUN_DATE:
            raise ValueError(f"run date must be {RUN_DATE}")
        root = args.root.resolve()
        paths = ExperimentPaths.defaults(root)
        paths = ExperimentPaths(
            args.output or paths.artifact,
            args.raw_transcript or paths.raw_transcript,
            args.checkpoint or paths.checkpoint,
        )
        paths = ExperimentPaths(
            paths.artifact if paths.artifact.is_absolute() else root / paths.artifact,
            paths.raw_transcript
            if paths.raw_transcript.is_absolute()
            else root / paths.raw_transcript,
            paths.checkpoint if paths.checkpoint.is_absolute() else root / paths.checkpoint,
        )
        artifact = run_experiment(root, paths)
        print(
            f"experiment_complete status={artifact['status']} "
            f"score={artifact['board_continuity_complete_score']} output={paths.artifact}",
            flush=True,
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"experiment_error type={type(exc).__name__} message={exc}", flush=True)
        return 2
