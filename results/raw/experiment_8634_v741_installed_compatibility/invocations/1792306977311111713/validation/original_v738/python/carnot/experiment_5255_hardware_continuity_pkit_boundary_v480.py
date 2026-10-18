#!/usr/bin/env python3
"""Exp 5255: hardware continuity plus p-kit/Extropic/Kona boundary notes.

Spec refs: REQ-HW-5255, SCENARIO-HW-5255.

This module refreshes continuity receipts only. KV260 and PolarFire are checked
with SSH reachability plus a tiny board-local hash/correctness smoke when SSH
works. GateMate keeps the known physical/JTAG blocker unless the operator says
the setup changed. The notes explain why IBM p-kit, Extropic, Kona, and Aleph
are public reference material rather than local acceleration evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]
Clock = Callable[[], float]

RUN_DATE = "20260705"
EXPERIMENT_ID = "exp5255-hardware-continuity-pkit-boundary-v480"
EXPERIMENT_NAME = "experiment_5255_hardware_continuity_pkit_boundary"
MILESTONE = "2026.07.480"
SCHEMA = "carnot.experiment_5255.hardware_continuity_pkit_boundary.v480"
SPEC_REFS = ("REQ-HW-5255", "SCENARIO-HW-5255")
RANDOM_SEED = 5255
INFERENCE_SUBSTRATE = "hardware_probe_no_speedup_claim"
RESULT_RELATIVE_PATH = Path("results/experiment_5255_hardware_continuity_pkit_boundary_v480.json")
PKIT_NOTE_RELATIVE_PATH = Path("docs/research-notes/experiment_5255_pkit_boundary_v480.md")
EXTROPIC_KONA_NOTE_RELATIVE_PATH = Path(
    "docs/research-notes/experiment_5255_extropic_kona_boundary_v480.md"
)
TERMINAL_PREFIXES = ("complete:", "blocked_")

KV260_SSH_COMMAND = ("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "kria", "true")
POLARFIRE_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)

PUBLIC_SOURCE_URLS = {
    "ibm_pkit": "https://github.com/IBM/p-kit",
    "extropic_hardware": "https://extropic.ai/hardware",
    "extropic_tsu_101": "https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware",
    "extropic_xtr0": "https://extropic.ai/writing/inside-x0-and-xtr-0",
    "logical_kona": "https://logicalintelligence.com/blog/energy-based-model-sudoku-demo",
    "logical_aleph": "https://logicalintelligence.com/blog/aleph-leading-benchmarks",
}

HONEST_VERDICT_PRINCIPLE = (
    "Value starts with complete: or blocked_ and states KV260, PolarFire, GateMate, "
    "and no-speedup status."
)
FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": HONEST_VERDICT_PRINCIPLE,
    "inference_substrate": f"Must be {INFERENCE_SUBSTRATE}; this is a receipt, not acceleration evidence.",
    "kv260_status": "Reachable only when SSH and the board-local hash/correctness smoke both pass.",
    "kv260_ssh_only_confirmed": "True only when KV260 is evaluated without host block-device preconditions.",
    "polarfire_status": "Reachable only when SSH and the board-local hash/correctness smoke both pass.",
    "gatemate_status": "Carry forward blocked_physical_jtag unless the operator changed the physical setup.",
    "physical_setup_changed": "Boolean operator setup input controlling whether GateMate probing is allowed.",
    "workload_hashes": "Commit, workload, executable or bitstream, and output hashes are receipt fields only.",
    "pkit_boundary_note_path": "Repository path to the IBM p-kit local-claim boundary note.",
    "extropic_kona_boundary_note_path": "Repository path to the Extropic/Kona/Aleph boundary note.",
    "speedup_claimed": "False because this experiment is continuity and boundary documentation only.",
}
REQUIRED_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
BOARD_STATUSES = {"reachable", "blocked", "not_checked"}
GATEMATE_STATUSES = {"blocked_physical_jtag", "reachable", "not_checked"}

HASH_SMOKE_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "workload": "exp5255_inline_hash_ising_smoke",
    "spins": [1, -1, -1, 1, 1, -1, 1, -1],
    "edges": [[0, 1, 1], [1, 2, -1], [2, 3, 1], [3, 4, -1], [4, 5, 1], [6, 7, 1]],
}
HASH_SMOKE_PROGRAM_TEMPLATE = (
    "exp5255_inline_board_hash_smoke_v1: parse JSON spins/edges, compute integer "
    "Ising energy, emit receipt hashes"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


HASH_SMOKE_EXECUTABLE_HASH = sha256_text(HASH_SMOKE_PROGRAM_TEMPLATE)


def ising_energy(payload: Mapping[str, Any]) -> int:
    """Return the tiny deterministic Ising energy used as a correctness smoke."""

    spins = [int(value) for value in payload["spins"]]
    total = 0
    for row, col, coupling in payload["edges"]:
        total -= int(coupling) * spins[int(row)] * spins[int(col)]
    return total


HASH_SMOKE_EXPECTED_ENERGY = ising_energy(HASH_SMOKE_BASE)
HASH_SMOKE_WORKLOAD = dict(HASH_SMOKE_BASE, expected_energy=HASH_SMOKE_EXPECTED_ENERGY)


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


HASH_SMOKE_WORKLOAD_HASH = sha256_json(HASH_SMOKE_WORKLOAD)


def output_hash(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "output_sha256"}
    return sha256_json(stable)


def command_to_string(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def hash_smoke_command(host: str, board: str) -> tuple[str, ...]:
    payload_json = json.dumps(HASH_SMOKE_WORKLOAD, sort_keys=True)
    remote = (
        "python3 - <<'PY'\n"
        "import hashlib, json\n"
        f"payload = {payload_json!r}\n"
        "data = json.loads(payload)\n"
        "spins = [int(value) for value in data['spins']]\n"
        "energy = 0\n"
        "for row, col, coupling in data['edges']:\n"
        "    energy -= int(coupling) * spins[int(row)] * spins[int(col)]\n"
        "out = {\n"
        f"    'board': {board!r},\n"
        f"    'workload_sha256': {HASH_SMOKE_WORKLOAD_HASH!r},\n"
        f"    'binary_or_bitstream_sha256': {HASH_SMOKE_EXECUTABLE_HASH!r},\n"
        f"    'inference_substrate': {INFERENCE_SUBSTRATE!r},\n"
        "    'energy': energy,\n"
        "    'correctness': {'energy_matches_expected': energy == data['expected_energy']},\n"
        "}\n"
        "encoded = json.dumps(out, sort_keys=True, separators=(',', ':')).encode()\n"
        "out['output_sha256'] = hashlib.sha256(encoded).hexdigest()\n"
        "print(json.dumps(out, sort_keys=True))\n"
        "PY"
    )
    return ("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, remote)


KV260_HASH_SMOKE_COMMAND = hash_smoke_command("kria", "kv260")
POLARFIRE_HASH_SMOKE_COMMAND = hash_smoke_command("polarfire", "polarfire")


@dataclass(frozen=True)
class CommandProbe:
    """One bounded command transcript preserved as receipt evidence."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_dict(self) -> JsonDict:
        return {
            "command": command_to_string(self.command),
            "exit_code": int(self.exit_code),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "combined_output": self.combined_output,
            "duration_s": round_duration(self.duration_s),
        }


CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandProbe:  # pragma: no cover
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
        return CommandProbe(
            tuple(command),
            int(completed.returncode),
            completed.stdout,
            completed.stderr,
            time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandProbe(
            tuple(command),
            124,
            exc.stdout or "",
            exc.stderr or f"command timed out after {timeout_s}s",
            time.perf_counter() - started,
        )
    except OSError as exc:
        return CommandProbe(tuple(command), 127, "", f"{type(exc).__name__}: {exc}", time.perf_counter() - started)


def get_git_commit(repo_root: str | Path) -> str:  # pragma: no cover
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "unknown"


def round_duration(duration_s: float) -> float:
    return round(max(float(duration_s), 0.000001), 6)


def wrap_field(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_last_json(text: str) -> JsonDict:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def board_receipt(
    *,
    board: str,
    ssh_probe: CommandProbe,
    smoke_probe: CommandProbe | None,
    commit: str,
) -> JsonDict:
    if ssh_probe.exit_code != 0:
        return {
            "status": "blocked",
            "blocked_reason": f"blocked_{board}_ssh",
            "command": command_to_string(ssh_probe.command),
            "commit": commit,
            "workload_sha256": None,
            "binary_or_bitstream_sha256": None,
            "output_hash": None,
            "hash_verified": False,
            "correctness_ok": False,
            "wall_clock_s": round_duration(ssh_probe.duration_s),
            "error": ssh_probe.combined_output,
        }
    if smoke_probe is None:
        return {
            "status": "blocked",
            "blocked_reason": f"blocked_{board}_hash_smoke_missing",
            "command": "",
            "commit": commit,
            "workload_sha256": None,
            "binary_or_bitstream_sha256": None,
            "output_hash": None,
            "hash_verified": False,
            "correctness_ok": False,
            "wall_clock_s": 0.0,
            "error": "reachable SSH precondition did not produce a hash smoke receipt",
        }
    if smoke_probe.exit_code != 0:
        return {
            "status": "blocked",
            "blocked_reason": f"blocked_{board}_hash_smoke",
            "command": command_to_string(smoke_probe.command),
            "commit": commit,
            "workload_sha256": None,
            "binary_or_bitstream_sha256": None,
            "output_hash": None,
            "hash_verified": False,
            "correctness_ok": False,
            "wall_clock_s": round_duration(smoke_probe.duration_s),
            "error": smoke_probe.combined_output,
        }

    parsed = parse_last_json(smoke_probe.combined_output)
    correctness = parsed.get("correctness") if isinstance(parsed.get("correctness"), Mapping) else {}
    expected_output_hash = output_hash(parsed)
    hash_verified = (
        parsed.get("board") == board
        and parsed.get("workload_sha256") == HASH_SMOKE_WORKLOAD_HASH
        and parsed.get("binary_or_bitstream_sha256") == HASH_SMOKE_EXECUTABLE_HASH
        and parsed.get("inference_substrate") == INFERENCE_SUBSTRATE
        and parsed.get("output_sha256") == expected_output_hash
    )
    correctness_ok = (
        correctness.get("energy_matches_expected") is True
        and parsed.get("energy") == HASH_SMOKE_EXPECTED_ENERGY
    )
    ok = hash_verified and correctness_ok
    return {
        "status": "reachable" if ok else "blocked",
        "blocked_reason": None if ok else f"blocked_{board}_hash_smoke",
        "command": command_to_string(smoke_probe.command),
        "commit": commit,
        "workload_sha256": parsed.get("workload_sha256"),
        "expected_workload_sha256": HASH_SMOKE_WORKLOAD_HASH,
        "binary_or_bitstream_sha256": parsed.get("binary_or_bitstream_sha256"),
        "expected_binary_or_bitstream_sha256": HASH_SMOKE_EXECUTABLE_HASH,
        "output_hash": parsed.get("output_sha256"),
        "expected_output_hash": expected_output_hash,
        "hash_verified": hash_verified,
        "correctness_ok": correctness_ok,
        "wall_clock_s": round_duration(smoke_probe.duration_s),
        "error": "" if ok else smoke_probe.combined_output,
        "output": parsed,
    }


def status_from_receipt(receipt: Mapping[str, Any]) -> str:
    return "reachable" if receipt.get("status") == "reachable" else "blocked"


def blocked_precondition(
    *, board: str, status: str, ssh_probe: CommandProbe, receipt: Mapping[str, Any]
) -> JsonDict | None:
    if status != "blocked":
        return None
    if ssh_probe.exit_code != 0:
        return {
            "reason": f"blocked_{board}_ssh",
            "command": command_to_string(ssh_probe.command),
            "exit_code": int(ssh_probe.exit_code),
            "error": ssh_probe.combined_output,
            "wall_clock_s": round_duration(ssh_probe.duration_s),
        }
    return {
        "reason": str(receipt.get("blocked_reason") or f"blocked_{board}_hash_smoke"),
        "command": str(receipt.get("command") or ""),
        "exit_code": 0,
        "error": str(receipt.get("error") or receipt.get("blocked_reason") or ""),
        "wall_clock_s": float(receipt.get("wall_clock_s") or 0.0),
    }


def command_receipt(*, board: str, kind: str, probe: CommandProbe, passed: bool) -> JsonDict:
    return {
        "board": board,
        "kind": kind,
        "command": command_to_string(probe.command),
        "exit_code": int(probe.exit_code),
        "passed": bool(passed),
        "stdout": probe.stdout,
        "stderr": probe.stderr,
        "error": "" if passed else probe.combined_output,
        "duration_s": round_duration(probe.duration_s),
    }


def build_honest_verdict(*, kv260_status: str, polarfire_status: str, gatemate_status: str) -> str:
    prefix = "blocked_precondition:" if "blocked" in {kv260_status, polarfire_status} else "complete:"
    return (
        f"{prefix} kv260={kv260_status} polarfire={polarfire_status} "
        f"gatemate={gatemate_status} no_speedup_claim"
    )


def build_pkit_boundary_note() -> str:
    return f"""# Exp 5255 IBM p-kit boundary note

Status: software reference only. No local p-bit hardware claim is allowed.

Public source reviewed:
- IBM/p-kit: {PUBLIC_SOURCE_URLS["ibm_pkit"]}

Boundary:
- IBM/p-kit is a Python library and GitHub watch reference for simulating probabilistic circuits.
- Carnot can use p-kit as terminology and software-comparison context for p-bits, probabilistic circuits, and Boltzmann-style Hamiltonians.
- Carnot cannot claim p-kit hardware execution, p-bit acceleration, local IBM p-bit silicon, or measured throughput from this repository reference.
- A future valid claim would need an authenticated local device or remote hardware transcript with workload hash, executable or bitstream hash, output hash, correctness parity, and end-to-end wall clock.

Local status: not local p-bit hardware.
"""


def build_extropic_kona_boundary_note() -> str:
    return f"""# Exp 5255 Extropic, Kona, and Aleph boundary note

Status: public architecture references only. No speedup claim is allowed.

Public sources reviewed:
- Extropic hardware: {PUBLIC_SOURCE_URLS["extropic_hardware"]}
- Extropic TSU 101: {PUBLIC_SOURCE_URLS["extropic_tsu_101"]}
- Extropic X0/XTR-0: {PUBLIC_SOURCE_URLS["extropic_xtr0"]}
- Logical Intelligence Kona Sudoku note: {PUBLIC_SOURCE_URLS["logical_kona"]}
- Logical Intelligence Aleph benchmarks note: {PUBLIC_SOURCE_URLS["logical_aleph"]}

Boundary:
- Extropic TSU, XTR-0, and Z1 materials motivate probabilistic and thermodynamic sampling research, but Carnot has no authenticated local TSU, XTR-0, or Z1 run.
- Kona motivates non-autoregressive, energy-scored constraint reasoning, but Carnot has no local Kona model, Kona service transcript, or Kona-equivalent benchmark claim.
- Aleph motivates formal verification and tool-orchestrated correctness, but Carnot has no local Aleph system or Aleph benchmark result.
- KV260 and PolarFire receipts in Exp 5255 are continuity/hash receipts only; GateMate remains a carried-forward physical/JTAG block.

No local Carnot claim: no Extropic hardware execution, no Kona/Aleph execution, and no acceleration result.
"""


def write_boundary_notes(repo_root: str | Path) -> tuple[Path, Path]:
    root = Path(repo_root)
    pkit_path = root / PKIT_NOTE_RELATIVE_PATH
    extropic_kona_path = root / EXTROPIC_KONA_NOTE_RELATIVE_PATH
    pkit_path.parent.mkdir(parents=True, exist_ok=True)
    pkit_path.write_text(build_pkit_boundary_note(), encoding="utf-8")
    extropic_kona_path.write_text(build_extropic_kona_boundary_note(), encoding="utf-8")
    return pkit_path, extropic_kona_path


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    notes_written: bool = True,
    physical_setup_changed: bool = False,
) -> JsonDict:
    started = clock()

    kv260_ssh = command_runner(KV260_SSH_COMMAND, 10.0)
    kv260_smoke_probe = (
        command_runner(KV260_HASH_SMOKE_COMMAND, 30.0) if kv260_ssh.exit_code == 0 else None
    )
    kv260_receipt = board_receipt(
        board="kv260", ssh_probe=kv260_ssh, smoke_probe=kv260_smoke_probe, commit=commit
    )
    kv260_status = status_from_receipt(kv260_receipt)

    polarfire_ssh = command_runner(POLARFIRE_SSH_COMMAND, 10.0)
    polarfire_smoke_probe = (
        command_runner(POLARFIRE_HASH_SMOKE_COMMAND, 30.0)
        if polarfire_ssh.exit_code == 0
        else None
    )
    polarfire_receipt = board_receipt(
        board="polarfire",
        ssh_probe=polarfire_ssh,
        smoke_probe=polarfire_smoke_probe,
        commit=commit,
    )
    polarfire_status = status_from_receipt(polarfire_receipt)

    gatemate_status = "not_checked" if physical_setup_changed else "blocked_physical_jtag"
    safe_commands = [
        command_receipt(
            board="kv260",
            kind="ssh_reachability",
            probe=kv260_ssh,
            passed=kv260_ssh.exit_code == 0,
        )
    ]
    if kv260_smoke_probe is not None:
        safe_commands.append(
            command_receipt(
                board="kv260",
                kind="board_local_hash_smoke",
                probe=kv260_smoke_probe,
                passed=kv260_receipt.get("status") == "reachable",
            )
        )
    safe_commands.append(
        command_receipt(
            board="polarfire",
            kind="ssh_reachability",
            probe=polarfire_ssh,
            passed=polarfire_ssh.exit_code == 0,
        )
    )
    if polarfire_smoke_probe is not None:
        safe_commands.append(
            command_receipt(
                board="polarfire",
                kind="board_local_hash_smoke",
                probe=polarfire_smoke_probe,
                passed=polarfire_receipt.get("status") == "reachable",
            )
        )

    output_hashes = {
        "kv260": kv260_receipt.get("output_hash"),
        "polarfire": polarfire_receipt.get("output_hash"),
    }
    workload_hashes = {
        "commit": commit,
        "workload_sha256": HASH_SMOKE_WORKLOAD_HASH,
        "binary_or_bitstream_sha256": HASH_SMOKE_EXECUTABLE_HASH,
        "output_hashes": output_hashes,
    }

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "duration_s": round_duration(clock() - started),
        "honest_verdict": wrap_field(
            "honest_verdict",
            build_honest_verdict(
                kv260_status=kv260_status,
                polarfire_status=polarfire_status,
                gatemate_status=gatemate_status,
            ),
        ),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "kv260_status": wrap_field("kv260_status", kv260_status),
        "kv260_ssh_only_confirmed": wrap_field("kv260_ssh_only_confirmed", True),
        "polarfire_status": wrap_field("polarfire_status", polarfire_status),
        "gatemate_status": wrap_field("gatemate_status", gatemate_status),
        "physical_setup_changed": wrap_field("physical_setup_changed", bool(physical_setup_changed)),
        "workload_hashes": wrap_field("workload_hashes", workload_hashes),
        "pkit_boundary_note_path": wrap_field(
            "pkit_boundary_note_path",
            str(PKIT_NOTE_RELATIVE_PATH) if notes_written else None,
        ),
        "extropic_kona_boundary_note_path": wrap_field(
            "extropic_kona_boundary_note_path",
            str(EXTROPIC_KONA_NOTE_RELATIVE_PATH) if notes_written else None,
        ),
        "speedup_claimed": wrap_field("speedup_claimed", False),
        "field_principles": dict(FIELD_PRINCIPLES),
        "board_receipts": {"kv260": kv260_receipt, "polarfire": polarfire_receipt},
        "kv260_blocked_precondition": blocked_precondition(
            board="kv260", status=kv260_status, ssh_probe=kv260_ssh, receipt=kv260_receipt
        ),
        "polarfire_blocked_precondition": blocked_precondition(
            board="polarfire",
            status=polarfire_status,
            ssh_probe=polarfire_ssh,
            receipt=polarfire_receipt,
        ),
        "gatemate_carry_forward": {
            "timestamp": run_date,
            "status": gatemate_status,
            "physical_setup_changed": bool(physical_setup_changed),
            "evidence": [
                "results/experiment_5243_hardware_continuity_kan_pbit_boundary_v479.json",
                "results/experiment_5231_hardware_continuity_pbit_boundary_v478.json",
                "research-hardware-wishlist.md exp5231-pbit-boundary-plan block",
            ],
            "rationale": (
                "no operator cable, port, power, or board setup change was provided"
                if not physical_setup_changed
                else "operator setup changed, but this bounded run did not perform physical/JTAG probing"
            ),
        },
        "safe_commands_run": safe_commands,
        "command_probes": {
            "kv260_ssh": kv260_ssh.as_dict(),
            "kv260_hash_smoke": kv260_smoke_probe.as_dict() if kv260_smoke_probe else None,
            "polarfire_ssh": polarfire_ssh.as_dict(),
            "polarfire_hash_smoke": polarfire_smoke_probe.as_dict()
            if polarfire_smoke_probe
            else None,
            "gatemate_physical_jtag": None,
        },
        "reviewed_inputs": [
            "CLAUDE.md hardware continuity and KV260 SSH-only sections",
            "CODEX.md required workflow",
            "research-hardware-wishlist.md active tracks and exp5231 carry-forward note",
            "research-references.md Extropic/Logical/GitHub watch entries",
            "results/experiment_5243_hardware_continuity_kan_pbit_boundary_v479.json",
            "ops/exclusion_manifest.yaml",
        ],
        "reviewed_public_sources": dict(PUBLIC_SOURCE_URLS),
        "conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def no_host_storage_markers(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    return "mmcblk" not in encoded and "/dev/disk" not in encoded


def validate_wrapped_field(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact.get(field)
    require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
    require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
    require("value" in wrapped, f"{field} missing value")
    return wrapped["value"]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = set(REQUIRED_WRAPPED_FIELDS) - set(artifact)
    require(not missing, f"missing required field: {sorted(missing)}")
    require(artifact.get("schema") == SCHEMA, "schema mismatch")
    require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id mismatch")
    require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs mismatch")
    require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    verdict = validate_wrapped_field(artifact, "honest_verdict")
    require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict prefix")
    for token in ("kv260=", "polarfire=", "gatemate=", "no_speedup"):
        require(token in verdict, f"honest_verdict missing {token}")
    require(
        validate_wrapped_field(artifact, "inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate mismatch",
    )
    require(validate_wrapped_field(artifact, "kv260_status") in BOARD_STATUSES, "kv260_status invalid")
    require(
        validate_wrapped_field(artifact, "kv260_ssh_only_confirmed") is True,
        "kv260 ssh-only discipline mismatch",
    )
    require(
        validate_wrapped_field(artifact, "polarfire_status") in BOARD_STATUSES,
        "polarfire_status invalid",
    )
    require(
        validate_wrapped_field(artifact, "gatemate_status") in GATEMATE_STATUSES,
        "gatemate_status invalid",
    )
    require(
        isinstance(validate_wrapped_field(artifact, "physical_setup_changed"), bool),
        "physical_setup_changed type mismatch",
    )
    hashes = validate_wrapped_field(artifact, "workload_hashes")
    require(isinstance(hashes, Mapping), "workload_hashes must be a mapping")
    for key in ("commit", "workload_sha256", "binary_or_bitstream_sha256", "output_hashes"):
        require(key in hashes, f"workload_hashes missing {key}")
    require(validate_wrapped_field(artifact, "speedup_claimed") is False, "speedup_claimed must be false")
    pkit_path = validate_wrapped_field(artifact, "pkit_boundary_note_path")
    extropic_kona_path = validate_wrapped_field(artifact, "extropic_kona_boundary_note_path")
    require(pkit_path is None or isinstance(pkit_path, str), "pkit_boundary_note_path invalid")
    require(
        extropic_kona_path is None or isinstance(extropic_kona_path, str),
        "extropic_kona_boundary_note_path invalid",
    )
    require(no_host_storage_markers(artifact), "host storage marker present")
    require(artifact.get("conductor_modified") is False, "conductor_modified mismatch")
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "checksum mismatch",
    )


def write_artifact(repo_root: str | Path, artifact: Mapping[str, Any]) -> Path:
    validate_artifact(artifact)
    out_path = Path(repo_root) / RESULT_RELATIVE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str | None = None,
    physical_setup_changed: bool = False,
) -> Path:
    pkit_path, extropic_kona_path = write_boundary_notes(repo_root)
    artifact = build_artifact(
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit or get_git_commit(repo_root),
        notes_written=pkit_path.exists() and extropic_kona_path.exists(),
        physical_setup_changed=physical_setup_changed,
    )
    return write_artifact(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument(
        "--physical-setup-changed",
        action="store_true",
        help="Record a changed GateMate setup; this run still avoids unbounded physical/JTAG probing.",
    )
    args = parser.parse_args(argv)
    print(
        run_experiment(
            repo_root=Path("."),
            run_date=args.date,
            physical_setup_changed=args.physical_setup_changed,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
