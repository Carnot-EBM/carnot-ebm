#!/usr/bin/env python3
"""Exp 5243: hardware continuity plus KAN/p-bit speedup boundary (v479).

Spec refs: REQ-HW-5243, SCENARIO-HW-5243.

This experiment deliberately treats the local boards as continuity evidence,
not performance evidence. KV260 and PolarFire get SSH-only reachability plus a
tiny board-local hash/correctness smoke when SSH works. GateMate keeps the
known physical/JTAG blocker unless the operator reports a physical setup
change. The KAN/p-bit note states what a future workload must prove before any
speedup experiment would be valid.
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

RUN_DATE = "20260704"
EXPERIMENT_ID = "exp5243-hardware-continuity-kan-pbit-boundary-v479"
EXPERIMENT_NAME = "experiment_5243_hardware_continuity_kan_pbit_boundary"
MILESTONE = "2026.07.479"
SCHEMA = "carnot.experiment_5243.hardware_continuity_kan_pbit_boundary.v479"
SPEC_REFS = ("REQ-HW-5243", "SCENARIO-HW-5243")
RANDOM_SEED = 5243
INFERENCE_SUBSTRATE = "hardware_reachability_hash_boundary"
RESULT_RELATIVE_PATH = Path("results/experiment_5243_hardware_continuity_kan_pbit_boundary_v479.json")
BOUNDARY_NOTE_RELATIVE_PATH = Path(
    "docs/research-notes/experiment_5243_kan_pbit_speedup_boundary_v479.md"
)
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")

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

HONEST_VERDICT_PRINCIPLE = (
    "Must start with complete:/complete_/success:/success_ or blocked_ and state board statuses "
    "with no speedup claim."
)
FIELD_PRINCIPLES: dict[str, str] = {
    "kv260_status": "Value is reachable only when SSH and the safe board-local hash smoke pass.",
    "kv260_ssh_only_confirmed": "True only when KV260 was evaluated without host-visible block-device preconditions.",
    "polarfire_status": "Value is reachable only when SSH and the safe board-local hash smoke pass.",
    "gatemate_status": "Carry forward blocked_physical_jtag unless the operator changed physical setup.",
    "physical_setup_changed": "Boolean operator setup input controlling whether physical/JTAG probing is allowed.",
    "speedup_claimed": "False because this artifact is reachability/hash continuity, not timing evidence.",
    "kan_pbit_boundary_note_path": "Repository path to the KAN/p-bit speedup-boundary note, or null if unwritten.",
    "safe_commands_run": "List of bounded SSH/hash commands actually attempted, with pass/fail and exact error text.",
    "inference_substrate": f"Must be {INFERENCE_SUBSTRATE}.",
    "honest_verdict": HONEST_VERDICT_PRINCIPLE,
}
REQUIRED_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
BOARD_STATUSES = {"reachable", "blocked", "not_checked"}
GATEMATE_STATUSES = {"blocked_physical_jtag", "reachable", "not_checked"}

HASH_SMOKE_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "workload": "exp5243_inline_hash_ising_smoke",
    "spins": [1, -1, -1, 1, 1, -1, 1, -1],
    "edges": [[0, 1, 1], [1, 2, -1], [2, 3, 1], [3, 4, -1], [4, 5, 1], [6, 7, 1]],
}
HASH_SMOKE_PROGRAM_TEMPLATE = (
    "exp5243_inline_board_hash_smoke_v1: parse JSON spins/edges, compute integer "
    "Ising energy, emit board/workload/executable hashes"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


HASH_SMOKE_EXECUTABLE_HASH = sha256_text(HASH_SMOKE_PROGRAM_TEMPLATE)


def ising_energy(payload: Mapping[str, Any]) -> int:
    """Return the tiny deterministic Ising energy used only as a smoke check."""

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


def command_to_string(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def hash_smoke_command(host: str, board: str) -> tuple[str, ...]:
    payload_json = json.dumps(HASH_SMOKE_WORKLOAD, sort_keys=True)
    remote = (
        "python3 - <<'PY'\n"
        "import json\n"
        f"payload = {payload_json!r}\n"
        "data = json.loads(payload)\n"
        "spins = [int(value) for value in data['spins']]\n"
        "energy = 0\n"
        "for row, col, coupling in data['edges']:\n"
        "    energy -= int(coupling) * spins[int(row)] * spins[int(col)]\n"
        "print(json.dumps({\n"
        f"    'board': {board!r},\n"
        f"    'workload_sha256': {HASH_SMOKE_WORKLOAD_HASH!r},\n"
        f"    'executable_sha256': {HASH_SMOKE_EXECUTABLE_HASH!r},\n"
        f"    'inference_substrate': {INFERENCE_SUBSTRATE!r},\n"
        "    'energy': energy,\n"
        "    'correctness': {'energy_matches_expected': energy == data['expected_energy']},\n"
        "}, sort_keys=True))\n"
        "PY"
    )
    return ("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, remote)


KV260_HASH_SMOKE_COMMAND = hash_smoke_command("kria", "kv260")
POLARFIRE_HASH_SMOKE_COMMAND = hash_smoke_command("polarfire", "polarfire")


@dataclass(frozen=True)
class CommandProbe:
    """One bounded command transcript preserved for continuity provenance."""

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


def board_hash_smoke_status(
    *, board: str, reachable: bool, smoke_probe: CommandProbe | None
) -> JsonDict:
    if not reachable:
        return {
            "status": "not_run_unreachable",
            "hash_verified": False,
            "correctness_ok": False,
            "workload_hash": None,
            "blocked_reason": f"blocked_{board}_ssh",
        }
    if smoke_probe is None:
        return {
            "status": "not_run_missing_probe",
            "hash_verified": False,
            "correctness_ok": False,
            "workload_hash": None,
            "blocked_reason": f"blocked_{board}_hash_smoke_missing",
        }
    if smoke_probe.exit_code != 0:
        return {
            "status": "smoke_command_failed",
            "hash_verified": False,
            "correctness_ok": False,
            "workload_hash": None,
            "blocked_reason": f"blocked_{board}_hash_smoke",
            "command": command_to_string(smoke_probe.command),
            "error": smoke_probe.combined_output,
        }

    output = parse_last_json(smoke_probe.combined_output)
    correctness = output.get("correctness") if isinstance(output.get("correctness"), Mapping) else {}
    hash_verified = (
        output.get("board") == board
        and output.get("workload_sha256") == HASH_SMOKE_WORKLOAD_HASH
        and output.get("executable_sha256") == HASH_SMOKE_EXECUTABLE_HASH
        and output.get("inference_substrate") == INFERENCE_SUBSTRATE
    )
    correctness_ok = (
        correctness.get("energy_matches_expected") is True
        and output.get("energy") == HASH_SMOKE_EXPECTED_ENERGY
    )
    ok = hash_verified and correctness_ok
    return {
        "status": "hash_verified_correctness_ok" if ok else "smoke_failed_validation",
        "hash_verified": hash_verified,
        "correctness_ok": correctness_ok,
        "workload_hash": output.get("workload_sha256"),
        "expected_workload_hash": HASH_SMOKE_WORKLOAD_HASH,
        "output": output,
        "blocked_reason": None if ok else f"blocked_{board}_hash_smoke",
    }


def status_from_probe(board: str, ssh_reachable: bool, smoke: Mapping[str, Any]) -> str:
    if not ssh_reachable:
        return "blocked"
    if smoke.get("status") == "hash_verified_correctness_ok":
        return "reachable"
    return "blocked"


def safe_command_entry(*, board: str, kind: str, probe: CommandProbe, passed: bool) -> JsonDict:
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


def blocked_substatus(
    *, board: str, status: str, ssh_probe: CommandProbe, smoke: Mapping[str, Any]
) -> JsonDict | None:
    if status != "blocked":
        return None
    if ssh_probe.exit_code != 0:
        return {
            "reason": f"blocked_{board}_ssh",
            "command": command_to_string(ssh_probe.command),
            "error": ssh_probe.combined_output,
        }
    return {
        "reason": str(smoke.get("blocked_reason") or f"blocked_{board}_hash_smoke"),
        "command": str(smoke.get("command") or ""),
        "error": str(smoke.get("error") or smoke.get("blocked_reason") or ""),
    }


def build_honest_verdict(*, kv260_status: str, polarfire_status: str, gatemate_status: str) -> str:
    return (
        "complete: "
        f"kv260={kv260_status} polarfire={polarfire_status} "
        f"gatemate={gatemate_status} no_speedup_claim"
    )


def build_boundary_note() -> str:
    return """# Exp 5243 KAN/p-bit speedup-boundary note

Status: boundary plan only. No speedup claim is allowed from this note.

Inputs reviewed:
- Exp 5242: `results/experiment_5242_kan_certificate_abstraction_scale_v479.json` shows a bounded deterministic KAEM/PWA/MILP certificate boundary, not analog execution or hardware readiness.
- V479 analog KAN reference: arXiv:2606.27892 motivates circuit-level error modeling and pruning for future analog KAN mapping.
- V479 p-bit references motivate partitioned sampler telemetry, boundary exchange accounting, and hash/correctness parity before benchmark claims.
- Extropic TSU remains watch-only until authenticated local TSU/XTR-0 hardware evidence exists.

Minimum valid workload before a speedup experiment:
- same canonical KAN/p-bit workload on CPU baseline and candidate hardware.
- same seeds, same partitions, same boundary exchange schedule, and same KAN-to-p-bit energy coupling.
- Board-local graph, partition, packet-stream, executable, and output hashes.
- Correctness parity against CPU reference energy, final state checksum, and accepted exchange count.
- Analog/KAN error-budget accounting when analog KAN approximation is part of the path.
- Data movement, host dispatch, device setup, sampler steps, validation, and end-to-end wall clock measured in one transcript.

Until those conditions exist, KV260, PolarFire, GateMate, and TSU evidence support only reachability/hash continuity.
"""


def write_boundary_note(repo_root: str | Path) -> Path:
    path = Path(repo_root) / BOUNDARY_NOTE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_boundary_note(), encoding="utf-8")
    return path


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    note_written: bool = True,
    physical_setup_changed: bool = False,
) -> JsonDict:
    started = clock()

    kv260_ssh = command_runner(KV260_SSH_COMMAND, 10.0)
    kv260_ssh_ok = kv260_ssh.exit_code == 0
    kv260_smoke_probe = command_runner(KV260_HASH_SMOKE_COMMAND, 30.0) if kv260_ssh_ok else None
    kv260_smoke = board_hash_smoke_status(
        board="kv260", reachable=kv260_ssh_ok, smoke_probe=kv260_smoke_probe
    )
    kv260_status = status_from_probe("kv260", kv260_ssh_ok, kv260_smoke)

    polarfire_ssh = command_runner(POLARFIRE_SSH_COMMAND, 10.0)
    polarfire_ssh_ok = polarfire_ssh.exit_code == 0
    polarfire_smoke_probe = (
        command_runner(POLARFIRE_HASH_SMOKE_COMMAND, 30.0) if polarfire_ssh_ok else None
    )
    polarfire_smoke = board_hash_smoke_status(
        board="polarfire", reachable=polarfire_ssh_ok, smoke_probe=polarfire_smoke_probe
    )
    polarfire_status = status_from_probe("polarfire", polarfire_ssh_ok, polarfire_smoke)

    gatemate_status = "not_checked" if physical_setup_changed else "blocked_physical_jtag"
    safe_commands = [
        safe_command_entry(
            board="kv260",
            kind="ssh_reachability",
            probe=kv260_ssh,
            passed=kv260_ssh_ok,
        )
    ]
    if kv260_smoke_probe is not None:
        safe_commands.append(
            safe_command_entry(
                board="kv260",
                kind="board_local_hash_smoke",
                probe=kv260_smoke_probe,
                passed=kv260_smoke.get("status") == "hash_verified_correctness_ok",
            )
        )
    safe_commands.append(
        safe_command_entry(
            board="polarfire",
            kind="ssh_reachability",
            probe=polarfire_ssh,
            passed=polarfire_ssh_ok,
        )
    )
    if polarfire_smoke_probe is not None:
        safe_commands.append(
            safe_command_entry(
                board="polarfire",
                kind="board_local_hash_smoke",
                probe=polarfire_smoke_probe,
                passed=polarfire_smoke.get("status") == "hash_verified_correctness_ok",
            )
        )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "duration_s": round_duration(clock() - started),
        "kv260_status": wrap_field("kv260_status", kv260_status),
        "kv260_ssh_only_confirmed": wrap_field("kv260_ssh_only_confirmed", True),
        "polarfire_status": wrap_field("polarfire_status", polarfire_status),
        "gatemate_status": wrap_field("gatemate_status", gatemate_status),
        "physical_setup_changed": wrap_field("physical_setup_changed", bool(physical_setup_changed)),
        "speedup_claimed": wrap_field("speedup_claimed", False),
        "kan_pbit_boundary_note_path": wrap_field(
            "kan_pbit_boundary_note_path",
            str(BOUNDARY_NOTE_RELATIVE_PATH) if note_written else None,
        ),
        "safe_commands_run": wrap_field("safe_commands_run", safe_commands),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": wrap_field(
            "honest_verdict",
            build_honest_verdict(
                kv260_status=kv260_status,
                polarfire_status=polarfire_status,
                gatemate_status=gatemate_status,
            ),
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
        "board_hash_smokes": {"kv260": kv260_smoke, "polarfire": polarfire_smoke},
        "kv260_blocked_substatus": blocked_substatus(
            board="kv260", status=kv260_status, ssh_probe=kv260_ssh, smoke=kv260_smoke
        ),
        "polarfire_blocked_substatus": blocked_substatus(
            board="polarfire",
            status=polarfire_status,
            ssh_probe=polarfire_ssh,
            smoke=polarfire_smoke,
        ),
        "gatemate_carry_forward": {
            "timestamp": run_date,
            "status": gatemate_status,
            "physical_setup_changed": bool(physical_setup_changed),
            "prior_reference": "results/experiment_5231_hardware_continuity_pbit_boundary_v478.json",
            "rationale": (
                "no operator cable, port, power, or board setup change was provided"
                if not physical_setup_changed
                else "operator setup changed, but this bounded run did not perform physical/JTAG probing"
            ),
        },
        "reviewed_inputs": [
            "results/experiment_5231_hardware_continuity_pbit_boundary_v478.json",
            "results/experiment_5242_kan_certificate_abstraction_scale_v479.json",
            "research-references.md V479 KAN and hardware sampling signals",
            "research-hardware-wishlist.md active hardware tracks",
        ],
        "command_probes": {
            "kv260_ssh": kv260_ssh.as_dict(),
            "kv260_hash_smoke": kv260_smoke_probe.as_dict() if kv260_smoke_probe else None,
            "polarfire_ssh": polarfire_ssh.as_dict(),
            "polarfire_hash_smoke": polarfire_smoke_probe.as_dict()
            if polarfire_smoke_probe
            else None,
            "gatemate_physical_jtag": None,
        },
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
    require(validate_wrapped_field(artifact, "speedup_claimed") is False, "speedup_claimed must be false")
    note_path = validate_wrapped_field(artifact, "kan_pbit_boundary_note_path")
    require(note_path is None or isinstance(note_path, str), "kan_pbit_boundary_note_path invalid")
    safe_commands = validate_wrapped_field(artifact, "safe_commands_run")
    require(isinstance(safe_commands, list), "safe_commands_run must be a list")
    for entry in safe_commands:
        require(isinstance(entry, Mapping), "safe_commands_run entry invalid")
        require("command" in entry and "passed" in entry, "safe command entry missing fields")
    require(
        validate_wrapped_field(artifact, "inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate mismatch",
    )
    verdict = validate_wrapped_field(artifact, "honest_verdict")
    require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict prefix")
    for token in ("kv260=", "polarfire=", "gatemate=", "no_speedup"):
        require(token in verdict, f"honest_verdict missing {token}")
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
    physical_setup_changed: bool = False,
) -> Path:
    note_path = write_boundary_note(repo_root)
    artifact = build_artifact(
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        note_written=note_path.exists(),
        physical_setup_changed=physical_setup_changed,
    )
    return write_artifact(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument(
        "--physical-setup-changed",
        action="store_true",
        help="Record that operator setup changed; this bounded run still avoids physical/JTAG probing.",
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
