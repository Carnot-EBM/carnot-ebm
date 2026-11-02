#!/usr/bin/env python3
"""Exp5506: multi-board hardware smoke receipts for descriptor continuity.

Spec refs: REQ-VERIFY-5506, SCENARIO-VERIFY-5506.

This module records whether each currently relevant substrate can receive a
small descriptor-smoke payload. The receipt is deliberately narrower than a
benchmark: it checks identity paths and hash continuity, records blockers, and
keeps ``hardware_speedup_claim`` false because no authenticated matched timing
harness exists for equivalent work in this task.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5420_pbit_hardware_transfer_preflight_v493 as exp5420


JsonDict = dict[str, Any]
Clock = Callable[[], float]
CommandProbe = exp5420.CommandProbe
CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5506_hardware_multiboard_receipts_v499.json")
EXP5505_RELATIVE_PATH = Path("results/experiment_5505_active_constraint_milp_descriptor_v499.json")
EXP5491_FALLBACK_RELATIVE_PATH = Path(
    "results/experiment_5491_active_constraint_subproblem_descriptor_v498.json"
)

EXPERIMENT = 5506
EXPERIMENT_ID = "exp5506-hardware-multiboard-receipts-v499"
MILESTONE = "2026.07.499"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5506
SCHEMA = "carnot.experiment_5506.hardware_multiboard_receipts.v499"
SPEC_REFS = ("REQ-VERIFY-5506", "SCENARIO-VERIFY-5506")
INFERENCE_SUBSTRATE = "hardware_smoke"
SMOKE_DESCRIPTOR_LIMIT = 2
TERMINAL_PREFIXES = ("complete:", "blocked:")
STATUS_VALUES = (
    "reachable",
    "blocked_identity",
    "blocked_toolchain",
    "blocked_descriptor",
    "not_attempted_with_reason",
)

LOCAL_TIMEOUT_S = 10.0
SSH_TIMEOUT_S = 5.0
GATEMATE_TIMEOUT_S = 10.0

KV260_IDENTITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "printf 'board_identity=kv260\\nhostname=' && hostname && printf '\\nmachine=' && uname -m",
)
POLARFIRE_IDENTITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "printf 'board_identity=polarfire\\nhostname=' && hostname && printf '\\nmachine=' && uname -m",
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")

HOST_STORAGE_MARKERS = ("/dev/mmcblk", "/dev/disk")
FORBIDDEN_COMMAND_TERMS = ("rm -rf", "mkfs", "dd ", "--write", "program", "flash")

FIELD_PRINCIPLES: dict[str, str] = {
    "descriptor_source": "names Exp5505 when ready or records the Exp5491 fallback.",
    "descriptor_source_ready": "prevents board receipts from implying descriptor readiness.",
    "polar_fire_status": "per-board terminal status from SSH identity and smoke evidence.",
    "kv260_status": "SSH-only KV260 status; never host SD-card storage.",
    "gatemate_status": "DirtyJTAG detect-only status without flash or workload overclaim.",
    "cuda_status": "local CUDA smoke status, not a matched speedup benchmark.",
    "cpu_status": "local CPU descriptor-smoke status used as a hash reference.",
    "command_receipts": (
        "bounded command transcripts with exit codes, summaries, hashes, timing, and blockers."
    ),
    "matched_hashes": "only hash matches over the selected descriptor smoke workload.",
    "matched_timing_available": "false until authenticated equivalent-work timing exists.",
    "hardware_speedup_claim": "must remain false without authenticated matched timing.",
    "conductor_unchanged": "confirms scripts/research_conductor.py was not modified.",
    "inference_substrate": "hardware_smoke, not live LLM inference or benchmark acceleration.",
    "honest_verdict": "terminal status with no speedup overclaim.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically so receipt hashes are portable."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Hash command text and stdout/stderr summaries without storing large blobs."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(payload))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while ignoring its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def command_to_string(command: Sequence[str]) -> str:
    """Render command tuples the same way as the existing hardware receipt helpers."""

    return exp5420.command_to_string(tuple(command))


def run_command(command: tuple[str, ...], timeout_s: float = LOCAL_TIMEOUT_S) -> CommandProbe:
    """Run one bounded command and convert expected hardware failures to receipts."""

    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return CommandProbe(
            command=tuple(command),
            exit_code=int(result.returncode),
            stdout=result.stdout,
            stderr=result.stderr,
            duration_s=round(time.perf_counter() - started, 6),
        )
    except FileNotFoundError as exc:
        return CommandProbe(
            command=tuple(command),
            exit_code=127,
            stderr=str(exc),
            duration_s=round(time.perf_counter() - started, 6),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
        return CommandProbe(
            command=tuple(command),
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            duration_s=round(time.perf_counter() - started, 6),
        )


def load_descriptor_source(root: str | Path = REPO_ROOT) -> JsonDict:
    """Select Exp5505 descriptors when ready, otherwise the Exp5491 fallback."""

    root_path = Path(root)
    primary = _read_json(root_path / EXP5505_RELATIVE_PATH)
    primary_blockers: list[str] = []
    if primary is not None:
        primary_ready = bool(primary.get("descriptor_ready_for_hardware"))
        primary_rows = primary.get("descriptor_rows")
        if primary_ready and isinstance(primary_rows, list) and primary_rows:
            return {
                "descriptor_source": EXP5505_RELATIVE_PATH.as_posix(),
                "descriptor_source_ready": True,
                "fallback_used": False,
                "descriptor_family": "exp5505_milp_maxsat_csp",
                "source_experiment_id": str(primary.get("experiment_id", "")),
                "descriptors": [dict(row) for row in primary_rows],
                "readiness_blockers": [],
                "primary_readiness_blockers": [],
            }
        primary_blockers = [str(item) for item in primary.get("readiness_blockers", [])]
        if not primary_blockers:
            primary_blockers = ["exp5505_descriptor_not_ready"]

    fallback = _read_json(root_path / EXP5491_FALLBACK_RELATIVE_PATH)
    if fallback is not None:
        fallback_ready = bool(fallback.get("subproblem_descriptor_ready"))
        fallback_rows = fallback.get("descriptors")
        if fallback_ready and isinstance(fallback_rows, list) and fallback_rows:
            return {
                "descriptor_source": EXP5491_FALLBACK_RELATIVE_PATH.as_posix(),
                "descriptor_source_ready": True,
                "fallback_used": True,
                "descriptor_family": "exp5491_active_constraint_subproblem",
                "source_experiment_id": str(fallback.get("experiment_id", "")),
                "descriptors": [dict(row) for row in fallback_rows],
                "readiness_blockers": [],
                "primary_readiness_blockers": primary_blockers,
            }
        return _missing_descriptor_source(["exp5491_fallback_not_ready", *primary_blockers])
    return _missing_descriptor_source(["descriptor_source_missing", *primary_blockers])


def _read_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _missing_descriptor_source(blockers: Sequence[str]) -> JsonDict:
    return {
        "descriptor_source": "missing",
        "descriptor_source_ready": False,
        "fallback_used": False,
        "descriptor_family": "missing",
        "source_experiment_id": "",
        "descriptors": [],
        "readiness_blockers": list(dict.fromkeys(blockers)),
        "primary_readiness_blockers": [],
    }


def build_smoke_workload(source: Mapping[str, Any]) -> JsonDict:
    """Build the small hash-only descriptor payload used by CPU, CUDA, and boards."""

    descriptors = list(source.get("descriptors", []))[:SMOKE_DESCRIPTOR_LIMIT]
    smokes: list[JsonDict] = []
    for index, descriptor in enumerate(descriptors):
        smokes.append(_descriptor_smoke(index, descriptor))
    aggregate_input_hash = sha256_json({"input_hashes": [row["input_hash"] for row in smokes]})
    aggregate_expected_output_hash = sha256_json(
        {"expected_output_hashes": [row["expected_output_hash"] for row in smokes]}
    )
    return {
        "schema": "carnot.hardware_descriptor_smoke_workload.v1",
        "descriptor_source": str(source.get("descriptor_source", "")),
        "descriptor_family": str(source.get("descriptor_family", "")),
        "descriptor_smokes": smokes,
        "aggregate_input_hash": aggregate_input_hash,
        "aggregate_expected_output_hash": aggregate_expected_output_hash,
        "smoke_workload_hash": sha256_json(
            {
                "descriptor_source": source.get("descriptor_source", ""),
                "aggregate_input_hash": aggregate_input_hash,
                "aggregate_expected_output_hash": aggregate_expected_output_hash,
            }
        ),
    }


def _descriptor_smoke(index: int, descriptor: Mapping[str, Any]) -> JsonDict:
    descriptor_id = str(descriptor.get("descriptor_id", f"descriptor_{index}"))
    partition_id = str(descriptor.get("partition_id", ""))
    partition_update = descriptor.get("partition_update")
    exact = descriptor.get("exact_fallback", {})
    expected = descriptor.get("expected_outputs")
    if not isinstance(expected, Mapping):
        expected = {
            "status": exact.get("status"),
            "solution": exact.get("solution"),
            "objective_score": exact.get("objective_score"),
            "solution_hash": exact.get("solution_hash"),
        }
    input_payload = {
        "descriptor_id": descriptor_id,
        "partition_id": partition_id,
        "descriptor_style": descriptor.get("descriptor_style", descriptor.get("coupling_type", "")),
        "domains": descriptor.get("domains", {}),
        "hard_constraints": descriptor.get("hard_constraints", []),
        "soft_preferences": descriptor.get("soft_preferences", []),
        "update_schedule": descriptor.get("update_schedule", {}),
    }
    input_hash = (
        str(partition_update["descriptor_input_hash"])
        if isinstance(partition_update, Mapping)
        and isinstance(partition_update.get("descriptor_input_hash"), str)
        else sha256_json(input_payload)
    )
    output_hash = (
        str(partition_update["expected_output_hash"])
        if isinstance(partition_update, Mapping)
        and isinstance(partition_update.get("expected_output_hash"), str)
        else sha256_json({"expected_outputs": dict(expected)})
    )
    return {
        "descriptor_id": descriptor_id,
        "partition_id": partition_id,
        "input_hash": input_hash,
        "expected_output_hash": output_hash,
        "expected_status": str(expected.get("status", exact.get("status", "unknown"))),
        "expected_solution_hash": expected.get("solution_hash", exact.get("solution_hash")),
    }


def smoke_receipt_stdout(substrate: str, workload: Mapping[str, Any], *, runtime: str) -> str:
    """Return the compact JSON stdout line expected from a descriptor-smoke command."""

    receipt = smoke_receipt(substrate, workload, runtime=runtime, wall_time_s=0.0005)
    return json.dumps(receipt, sort_keys=True, ensure_ascii=True) + "\n"


def smoke_receipt(
    substrate: str,
    workload: Mapping[str, Any],
    *,
    runtime: str,
    wall_time_s: float,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the hash receipt emitted by local and remote smoke commands."""

    payload: JsonDict = {
        "schema": "carnot.hardware_smoke_receipt.v1",
        "substrate": substrate,
        "runtime": runtime,
        "descriptor_source": workload["descriptor_source"],
        "descriptor_count": len(workload["descriptor_smokes"]),
        "aggregate_input_hash": workload["aggregate_input_hash"],
        "aggregate_expected_output_hash": workload["aggregate_expected_output_hash"],
        "smoke_workload_hash": workload["smoke_workload_hash"],
        "descriptor_hashes": [
            {
                "descriptor_id": row["descriptor_id"],
                "input_hash": row["input_hash"],
                "expected_output_hash": row["expected_output_hash"],
            }
            for row in workload["descriptor_smokes"]
        ],
        "wall_time_s": round(float(wall_time_s), 9),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def local_smoke_command(substrate: str, workload: Mapping[str, Any]) -> tuple[str, ...]:
    """Build a local CPU or CUDA descriptor-smoke command."""

    if substrate not in {"cpu", "cuda"}:
        raise ValueError("local substrate")
    payload = base64.urlsafe_b64encode(
        canonical_json(_workload_command_payload(workload)).encode("utf-8")
    ).decode("ascii")
    return (
        sys.executable,
        "-m",
        __name__,
        "--emit-local-smoke",
        substrate,
        "--payload-b64",
        payload,
    )


def remote_smoke_command(board: str, workload: Mapping[str, Any]) -> tuple[str, ...]:
    """Build a safe SSH command for a board-local descriptor-smoke receipt."""

    if board != "polarfire":
        raise ValueError("remote board")
    remote = "python3 - <<'PY'\n" + _remote_smoke_source(board, workload) + "\nPY"
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        "polarfire",
        remote,
    )


def _remote_smoke_source(substrate: str, workload: Mapping[str, Any]) -> str:
    payload_json = json.dumps(_workload_command_payload(workload), sort_keys=True)
    return "\n".join(
        [
            "import hashlib,json,time",
            f"payload=json.loads({payload_json!r})",
            "started=time.perf_counter()",
            "extra={'board_local':True}",
            "runtime='remote_python'",
            *_receipt_source_tail(substrate),
        ]
    )


def emit_local_smoke(substrate: str, payload: Mapping[str, Any]) -> int:
    """Emit one local descriptor-smoke JSON line for the CLI helper path."""

    started = time.perf_counter()
    if substrate == "cuda":
        try:
            import torch as gpu_runtime
        except Exception as exc:
            print("gpu_runtime_import_failed=" + type(exc).__name__)
            return 42
        if not gpu_runtime.cuda.is_available():
            print("gpu_available=false")
            return 43
        device = gpu_runtime.device("cuda")
        tensor = gpu_runtime.tensor([len(payload["descriptor_smokes"])], device=device)
        gpu_runtime.cuda.synchronize()
        extra = {
            "gpu_device": gpu_runtime.cuda.get_device_name(device),
            "gpu_tensor_value": int(tensor.cpu().item()),
        }
        runtime = "local_gpu_smoke"
    elif substrate == "cpu":
        extra = {}
        runtime = "python_cpu"
    else:
        print("unknown_substrate")
        return 2
    receipt = smoke_receipt(
        substrate,
        payload,
        runtime=runtime,
        wall_time_s=max(time.perf_counter() - started, 0.0),
        extra=extra,
    )
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=True))
    return 0


def _workload_command_payload(workload: Mapping[str, Any]) -> JsonDict:
    return {
        "descriptor_source": workload["descriptor_source"],
        "descriptor_smokes": [dict(row) for row in workload["descriptor_smokes"]],
        "aggregate_input_hash": workload["aggregate_input_hash"],
        "aggregate_expected_output_hash": workload["aggregate_expected_output_hash"],
        "smoke_workload_hash": workload["smoke_workload_hash"],
    }


def _receipt_source_tail(substrate: str) -> list[str]:
    return [
        "receipt={",
        " 'schema':'carnot.hardware_smoke_receipt.v1',",
        f" 'substrate':{substrate!r},",
        " 'runtime':runtime,",
        " 'descriptor_source':payload['descriptor_source'],",
        " 'descriptor_count':len(payload['descriptor_smokes']),",
        " 'aggregate_input_hash':payload['aggregate_input_hash'],",
        " 'aggregate_expected_output_hash':payload['aggregate_expected_output_hash'],",
        " 'smoke_workload_hash':payload['smoke_workload_hash'],",
        " 'descriptor_hashes':[{'descriptor_id':row['descriptor_id'],'input_hash':row['input_hash'],'expected_output_hash':row['expected_output_hash']} for row in payload['descriptor_smokes']],",
        " 'wall_time_s':round(time.perf_counter()-started,9),",
        "}",
        "receipt.update(extra)",
        "print(json.dumps(receipt,sort_keys=True))",
    ]


def parse_smoke_stdout(
    stdout: str,
    *,
    substrate: str,
    workload: Mapping[str, Any],
) -> tuple[JsonDict | None, str | None]:
    """Parse one smoke stdout stream and verify it matches the selected workload."""

    parsed: Any | None = None
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            break
    if not isinstance(parsed, Mapping):
        return None, "smoke stdout is not valid JSON"
    receipt = dict(parsed)
    errors: list[str] = []
    if receipt.get("substrate") != substrate:
        errors.append("substrate mismatch")
    for field in (
        "descriptor_source",
        "aggregate_input_hash",
        "aggregate_expected_output_hash",
        "smoke_workload_hash",
    ):
        if receipt.get(field) != workload.get(field):
            errors.append(f"{field} mismatch")
    if receipt.get("descriptor_count") != len(workload["descriptor_smokes"]):
        errors.append("descriptor_count mismatch")
    if not isinstance(receipt.get("wall_time_s"), int | float) or receipt["wall_time_s"] < 0:
        errors.append("wall_time_s invalid")
    return receipt, "; ".join(errors) if errors else None


def collect_receipts(
    *,
    workload: Mapping[str, Any],
    command_runner: CommandRunner,
) -> JsonDict:
    """Collect CPU/CUDA smoke receipts and safe board identity/smoke receipts."""

    command_receipts: list[JsonDict] = []
    matched_hashes: list[JsonDict] = []
    statuses: JsonDict = {}

    cpu_command = local_smoke_command("cpu", workload)
    cpu_probe = command_runner(cpu_command, LOCAL_TIMEOUT_S)
    cpu_status, cpu_receipt, cpu_blocker = _classify_smoke_probe(
        cpu_probe,
        substrate="cpu",
        workload=workload,
        unavailable_reason="cpu_smoke_failed",
    )
    statuses["cpu_status"] = cpu_status
    command_receipts.append(
        _command_receipt(
            cpu_probe,
            kind="cpu_descriptor_smoke",
            timeout_s=LOCAL_TIMEOUT_S,
            outcome=cpu_status,
            blocked_reason=cpu_blocker,
            parsed_receipt=cpu_receipt,
        )
    )
    matched_hashes.extend(_matched_hash_rows("cpu", cpu_receipt, workload))

    cuda_command = local_smoke_command("cuda", workload)
    cuda_probe = command_runner(cuda_command, LOCAL_TIMEOUT_S)
    cuda_status, cuda_receipt, cuda_blocker = _classify_smoke_probe(
        cuda_probe,
        substrate="cuda",
        workload=workload,
        unavailable_reason=_cuda_blocked_reason(cuda_probe),
    )
    statuses["cuda_status"] = cuda_status
    command_receipts.append(
        _command_receipt(
            cuda_probe,
            kind="cuda_descriptor_smoke",
            timeout_s=LOCAL_TIMEOUT_S,
            outcome=cuda_status,
            blocked_reason=cuda_blocker,
            parsed_receipt=cuda_receipt,
        )
    )
    matched_hashes.extend(_matched_hash_rows("cuda", cuda_receipt, workload))

    kv_probe = command_runner(KV260_IDENTITY_COMMAND, SSH_TIMEOUT_S)
    kv_status, kv_blocker = _identity_status(kv_probe, board="kv260")
    statuses["kv260_status"] = kv_status
    command_receipts.append(
        _command_receipt(
            kv_probe,
            kind="kv260_ssh_identity",
            timeout_s=SSH_TIMEOUT_S,
            outcome=kv_status,
            blocked_reason=kv_blocker,
        )
    )

    gate_probe = command_runner(GATEMATE_DETECT_COMMAND, GATEMATE_TIMEOUT_S)
    gate_status, gate_blocker = _gatemate_status(gate_probe)
    statuses["gatemate_status"] = gate_status
    command_receipts.append(
        _command_receipt(
            gate_probe,
            kind="gatemate_dirtyjtag_detect",
            timeout_s=GATEMATE_TIMEOUT_S,
            outcome=gate_status,
            blocked_reason=gate_blocker,
        )
    )

    pf_identity_probe = command_runner(POLARFIRE_IDENTITY_COMMAND, SSH_TIMEOUT_S)
    pf_identity_status, pf_identity_blocker = _identity_status(
        pf_identity_probe,
        board="polarfire",
    )
    command_receipts.append(
        _command_receipt(
            pf_identity_probe,
            kind="polarfire_ssh_identity",
            timeout_s=SSH_TIMEOUT_S,
            outcome=pf_identity_status,
            blocked_reason=pf_identity_blocker,
        )
    )
    if pf_identity_status == "reachable":
        pf_command = remote_smoke_command("polarfire", workload)
        pf_probe = command_runner(pf_command, SSH_TIMEOUT_S)
        pf_status, pf_receipt, pf_blocker = _classify_smoke_probe(
            pf_probe,
            substrate="polarfire",
            workload=workload,
            unavailable_reason="polarfire_smoke_failed",
        )
        statuses["polar_fire_status"] = pf_status
        command_receipts.append(
            _command_receipt(
                pf_probe,
                kind="polarfire_descriptor_smoke",
                timeout_s=SSH_TIMEOUT_S,
                outcome=pf_status,
                blocked_reason=pf_blocker,
                parsed_receipt=pf_receipt,
            )
        )
        matched_hashes.extend(_matched_hash_rows("polarfire", pf_receipt, workload))
    else:
        statuses["polar_fire_status"] = pf_identity_status

    return {
        **statuses,
        "command_receipts": command_receipts,
        "matched_hashes": matched_hashes,
    }


def _classify_smoke_probe(
    probe: CommandProbe,
    *,
    substrate: str,
    workload: Mapping[str, Any],
    unavailable_reason: str,
) -> tuple[str, JsonDict | None, str | None]:
    receipt, parse_error = parse_smoke_stdout(probe.stdout, substrate=substrate, workload=workload)
    if probe.exit_code == 0 and receipt is not None and parse_error is None:
        return "reachable", receipt, None
    if substrate == "cuda":
        return "blocked_toolchain", None, unavailable_reason
    if probe.exit_code in {42, 43, 127}:
        return "blocked_toolchain", None, unavailable_reason
    return "blocked_toolchain", receipt, parse_error or unavailable_reason


def _cuda_blocked_reason(probe: CommandProbe) -> str:
    text = probe.combined_output.lower()
    if "cuda_available=false" in text or probe.exit_code == 43:
        return "cuda_unavailable"
    if "torch_import_failed" in text or probe.exit_code in {42, 127}:
        return "cuda_toolchain_unavailable"
    return "cuda_smoke_failed"


def _identity_status(probe: CommandProbe, *, board: str) -> tuple[str, str | None]:
    if probe.exit_code != 0:
        return "blocked_identity", f"blocked_{board}_ssh_identity"
    identity = parse_identity_stdout(probe.stdout)
    if identity.get("board_identity") != board:
        return "blocked_identity", f"blocked_{board}_ssh_identity"
    return "reachable", None


def parse_identity_stdout(stdout: str) -> JsonDict:
    """Parse key-value board identity output from SSH precondition commands."""

    values: JsonDict = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _gatemate_status(probe: CommandProbe) -> tuple[str, str | None]:
    if probe.exit_code == 127:
        return "blocked_toolchain", "gatemate_toolchain_unavailable"
    detected = probe.exit_code == 0 and any(
        marker in probe.stdout for marker in ("IDCODE", "GateMate", "GM1A")
    )
    if detected:
        return "reachable", None
    return "blocked_identity", "blocked_gatemate_dirtyjtag_identity"


def _command_receipt(
    probe: CommandProbe,
    *,
    kind: str,
    timeout_s: float,
    outcome: str,
    blocked_reason: str | None = None,
    parsed_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    receipt = exp5420.command_receipt(
        probe,
        kind=kind,
        timeout_s=timeout_s,
        outcome=outcome,
    )
    if blocked_reason:
        receipt["blocked_reason"] = blocked_reason
    if parsed_receipt is not None:
        receipt["hash_receipt"] = dict(parsed_receipt)
        receipt["hash_receipt_sha256"] = sha256_json(parsed_receipt)
    return receipt


def _matched_hash_rows(
    substrate: str,
    receipt: Mapping[str, Any] | None,
    workload: Mapping[str, Any],
) -> list[JsonDict]:
    if receipt is None:
        return []
    if receipt.get("aggregate_input_hash") != workload.get("aggregate_input_hash") or receipt.get(
        "aggregate_expected_output_hash"
    ) != workload.get("aggregate_expected_output_hash"):
        return []
    return [
        {
            "substrate": substrate,
            "descriptor_id": row["descriptor_id"],
            "input_hash": row["input_hash"],
            "expected_output_hash": row["expected_output_hash"],
            "matched": True,
        }
        for row in workload["descriptor_smokes"]
    ]


def conductor_unchanged(root: str | Path = REPO_ROOT) -> bool:
    """Return false only when git reports a diff for scripts/research_conductor.py."""

    root_path = Path(root)
    conductor = root_path / "scripts/research_conductor.py"
    if not conductor.exists():
        return True
    commands = (
        ("git", "-C", str(root_path), "diff", "--quiet", "--", "scripts/research_conductor.py"),
        (
            "git",
            "-C",
            str(root_path),
            "diff",
            "--cached",
            "--quiet",
            "--",
            "scripts/research_conductor.py",
        ),
    )
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 1:
            return False
    return True


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the Exp5506 terminal artifact from real or mocked receipt commands."""

    started = clock()
    source = load_descriptor_source(root)
    descriptor_ready = bool(source["descriptor_source_ready"])
    if descriptor_ready:
        workload = build_smoke_workload(source)
        collected = collect_receipts(workload=workload, command_runner=command_runner)
    else:
        workload = build_smoke_workload(source)
        collected = {
            "cpu_status": "blocked_descriptor",
            "cuda_status": "blocked_descriptor",
            "kv260_status": "blocked_descriptor",
            "gatemate_status": "blocked_descriptor",
            "polar_fire_status": "blocked_descriptor",
            "command_receipts": [],
            "matched_hashes": [],
        }
    ready = descriptor_ready and collected["cpu_status"] == "reachable"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(clock() - started, 0.0), 6),
        "descriptor_source": source["descriptor_source"],
        "descriptor_source_ready": descriptor_ready,
        "descriptor_family": source["descriptor_family"],
        "fallback_used": bool(source["fallback_used"]),
        "source_experiment_id": source["source_experiment_id"],
        "selected_descriptor_count": len(workload["descriptor_smokes"]),
        "smoke_workload_hash": workload["smoke_workload_hash"],
        "aggregate_input_hash": workload["aggregate_input_hash"],
        "aggregate_expected_output_hash": workload["aggregate_expected_output_hash"],
        "polar_fire_status": collected["polar_fire_status"],
        "kv260_status": collected["kv260_status"],
        "gatemate_status": collected["gatemate_status"],
        "cuda_status": collected["cuda_status"],
        "cpu_status": collected["cpu_status"],
        "command_receipts": collected["command_receipts"],
        "matched_hashes": collected["matched_hashes"],
        "matched_timing_available": False,
        "hardware_speedup_claim": False,
        "conductor_unchanged": conductor_unchanged(root),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(
            ready=ready,
            descriptor_ready=descriptor_ready,
            statuses=collected,
            blockers=source["readiness_blockers"],
        ),
        "readiness_blockers": readiness_blockers(
            descriptor_ready=descriptor_ready,
            cpu_status=str(collected["cpu_status"]),
            source_blockers=source["readiness_blockers"],
        ),
        "claim_limits": [
            "descriptor smoke hashes only",
            "board identity reachability is not a hardware acceleration claim",
            "cuda substrate smoke is local availability only, not matched timing",
            "KV260 uses SSH identity only; host /dev/mmcblk paths are forbidden",
            "GateMate uses dirtyJtag detect only; no flash or workload execution",
            "hardware_speedup_claim=false without authenticated matched timing",
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalize_tests(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def readiness_blockers(
    *,
    descriptor_ready: bool,
    cpu_status: str,
    source_blockers: Sequence[str],
) -> list[str]:
    """Explain why the receipt artifact is blocked while preserving partial evidence."""

    blockers: list[str] = []
    if not descriptor_ready:
        blockers.extend(source_blockers or ["descriptor_source_not_ready"])
    if descriptor_ready and cpu_status != "reachable":
        blockers.append("cpu_descriptor_smoke_not_reachable")
    return list(dict.fromkeys(blockers))


def honest_verdict(
    *,
    ready: bool,
    descriptor_ready: bool,
    statuses: Mapping[str, Any],
    blockers: Sequence[str],
) -> str:
    """Return a terminal verdict that preserves blocked boards without overclaim."""

    if ready:
        blocked = sorted(
            name.removesuffix("_status")
            for name, status in statuses.items()
            if name.endswith("_status") and status != "reachable"
        )
        blocked_text = ",".join(blocked) if blocked else "none"
        return (
            "complete: descriptor smoke receipts collected with honest blocked "
            f"statuses ({blocked_text}); matched_timing_available=false; "
            "hardware_speedup_claim=false"
        )
    if not descriptor_ready:
        joined = ",".join(blockers) if blockers else "descriptor_source_not_ready"
        return f"blocked: {joined}; hardware_speedup_claim=false"
    return "blocked: cpu_descriptor_smoke_not_reachable; hardware_speedup_claim=false"


def _normalize_tests(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    if tests_run is None:
        return [{"command": "verification not yet attached", "outcome": "pending"}]
    return [dict(item) for item in tests_run]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on schema drift, unsafe commands, or speedup overclaim."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("schema") == SCHEMA, "schema")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(isinstance(artifact.get("descriptor_source"), str), "descriptor_source")
    _require(isinstance(artifact.get("descriptor_source_ready"), bool), "descriptor_source_ready")
    for field in (
        "polar_fire_status",
        "kv260_status",
        "gatemate_status",
        "cuda_status",
        "cpu_status",
    ):
        _require(artifact.get(field) in STATUS_VALUES, field)
    _require(artifact.get("matched_timing_available") is False, "matched_timing_available")
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(artifact.get("conductor_unchanged") is True, "conductor_unchanged")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("hardware_speedup_claim=false" in verdict, "honest_verdict")
    _validate_command_receipts(artifact.get("command_receipts"))
    _validate_matched_hashes(artifact.get("matched_hashes"))
    _validate_tests_run(artifact.get("tests_run"))
    if artifact.get("descriptor_source_ready") is False:
        _require(artifact.get("command_receipts") == [], "command_receipts")
        for field in (
            "polar_fire_status",
            "kv260_status",
            "gatemate_status",
            "cuda_status",
            "cpu_status",
        ):
            _require(artifact.get(field) == "blocked_descriptor", field)
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def _validate_command_receipts(receipts: Any) -> None:
    _require(isinstance(receipts, list), "command_receipts")
    for receipt in receipts:
        _require(isinstance(receipt, Mapping), "command_receipts")
        command = receipt.get("command")
        _require(isinstance(command, str) and command, "command_receipts")
        _require(not _command_uses_host_storage(command), "host storage command")
        _require(not _command_is_destructive(command), "destructive command")
        _require(receipt.get("command_sha256") == sha256_text(command), "command_sha256")
        _require(isinstance(receipt.get("exit_code"), int), "exit_code")
        _require(isinstance(receipt.get("stdout_sha256"), str), "stdout_sha256")
        _require(isinstance(receipt.get("stderr_sha256"), str), "stderr_sha256")
        if "hash_receipt" in receipt:
            _require(
                receipt.get("hash_receipt_sha256") == sha256_json(receipt["hash_receipt"]),
                "hash_receipt_sha256",
            )


def _validate_matched_hashes(rows: Any) -> None:
    _require(isinstance(rows, list), "matched_hashes")
    for row in rows:
        _require(isinstance(row, Mapping), "matched_hashes")
        _require(row.get("substrate") in {"cpu", "cuda", "polarfire"}, "matched_hashes")
        _require(
            isinstance(row.get("descriptor_id"), str) and row["descriptor_id"], "matched_hashes"
        )
        _require(
            isinstance(row.get("input_hash"), str) and len(row["input_hash"]) == 64, "input_hash"
        )
        _require(
            isinstance(row.get("expected_output_hash"), str)
            and len(row["expected_output_hash"]) == 64,
            "expected_output_hash",
        )
        _require(row.get("matched") is True, "matched_hashes")


def _validate_tests_run(tests_run: Any) -> None:
    _require(isinstance(tests_run, list) and tests_run, "tests_run")
    for item in tests_run:
        _require(isinstance(item, Mapping), "tests_run")
        _require(isinstance(item.get("command"), str) and item["command"], "tests_run")
        _require(isinstance(item.get("outcome"), str) and item["outcome"], "tests_run")


def _command_is_destructive(command_text: str) -> bool:
    lowered = command_text.lower()
    return any(term in lowered for term in FORBIDDEN_COMMAND_TERMS)


def _command_uses_host_storage(command_text: str) -> bool:
    return any(marker in command_text for marker in HOST_STORAGE_MARKERS)


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal artifact under ``root`` and return the path."""

    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    descriptor_root: str | Path | None = None,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Build, validate, and write Exp5506's JSON deliverable."""

    artifact = build_artifact(
        root=descriptor_root if descriptor_root is not None else repo_root,
        command_runner=command_runner,
        clock=clock,
        tests_run=tests_run,
    )
    return write_output(repo_root, artifact)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-local-smoke", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--payload-b64", default="")
    parser.add_argument("--output-root", default=str(REPO_ROOT))
    parser.add_argument("--descriptor-root", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    args = parser.parse_args(argv)
    if args.emit_local_smoke is not None:
        payload = json.loads(base64.urlsafe_b64decode(args.payload_b64).decode("utf-8"))
        return emit_local_smoke(args.emit_local_smoke, payload)
    tests = [{"command": command, "outcome": "passed"} for command in args.test_run] or None
    run_experiment(
        repo_root=args.output_root,
        descriptor_root=args.descriptor_root,
        tests_run=tests,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main(sys.argv[1:]))
