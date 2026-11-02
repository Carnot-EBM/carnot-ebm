#!/usr/bin/env python3
"""Exp 5231: hardware continuity plus p-bit boundary-exchange plan (v478).

Spec refs: REQ-HW-5231, SCENARIO-HW-5231.

This experiment is intentionally narrow. KV260 is checked by SSH reachability
only, PolarFire gets the same SSH precondition plus a tiny hash-verified smoke
when reachable, and GateMate preserves the v477 physical/JTAG block unless the
operator explicitly says the cable, port, or board setup changed. The p-bit work
is a future measurement plan, not a benchmark result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]
Clock = Callable[[], float]

EXPERIMENT_ID = "exp5231-hardware-continuity-pbit-boundary-v478"
EXPERIMENT_NAME = "experiment_5231_hardware_continuity_pbit_boundary"
MILESTONE = "2026.07.478"
SCHEMA = "carnot.experiment_5231_hardware_continuity_pbit_boundary.v478"
SPEC_REFS = ["REQ-HW-5231", "SCENARIO-HW-5231"]
OUTPUT_REL_PATH = Path("results") / "experiment_5231_hardware_continuity_pbit_boundary_v478.json"
PBIT_PLAN_REL_PATH = (
    Path("docs") / "research-notes" / "experiment_5231_pbit_boundary_exchange_timing_ratio_plan.md"
)
RANDOM_SEED = 5231
INFERENCE_SUBSTRATE = "hardware_reachability_and_sampler_boundary_plan"
NO_SPEEDUP_PRINCIPLE = (
    "No .478 hardware task may claim speedup without a real end-to-end workload."
)
HONEST_VERDICT_PRINCIPLE = (
    "Must start with complete:/complete_/success:/success_ and state reachability plus no-speedup status."
)

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
GATEMATE_DEBUG_DETECT_COMMAND = (
    "openFPGALoader",
    "-c",
    "dirtyJtag",
    "--detect",
    "--verbose-level",
    "2",
)
GATEMATE_EXPECTED_IDCODE = "0x20000001"
PRIOR_GATEMATE_RAW_IDCODE = "0xffffffff"

PBIT_FUTURE_WORKLOAD = "distributed_sparse_pbit_boundary_exchange_n1024x4"
PBIT_PARTITIONS = "4 x 256 p-bits"

INLINE_PROGRAM_TEMPLATE = (
    "exp5231_inline_polarfire_ising_smoke_v1: parse JSON spins/edges, compute "
    "integer Ising energy, emit workload and executable hashes"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


INLINE_EXECUTABLE_HASH = sha256_text(INLINE_PROGRAM_TEMPLATE)

POLARFIRE_SMOKE_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "polarfire",
    "workload": "exp5231_inline_ising_energy_smoke",
    "spins": [1, -1, -1, 1, 1, -1, 1, -1],
    "edges": [[0, 1, 1], [1, 2, -1], [2, 3, 1], [3, 4, -1], [4, 5, 1], [6, 7, 1]],
}


def ising_energy(payload: Mapping[str, Any]) -> int:
    """Return the integer Ising energy for a tiny reproducible edge-list smoke."""

    spins = [int(value) for value in payload["spins"]]
    total = 0
    for row, col, coupling in payload["edges"]:
        total -= int(coupling) * spins[int(row)] * spins[int(col)]
    return total


POLARFIRE_EXPECTED_ENERGY = ising_energy(POLARFIRE_SMOKE_BASE)
POLARFIRE_SMOKE_WORKLOAD = dict(
    POLARFIRE_SMOKE_BASE, expected_energy=POLARFIRE_EXPECTED_ENERGY
)


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


POLARFIRE_SMOKE_WORKLOAD_HASH = sha256_json(POLARFIRE_SMOKE_WORKLOAD)


def command_to_string(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def polarfire_smoke_command() -> tuple[str, ...]:
    """Build the bounded board-local smoke command with embedded hash evidence."""

    payload_json = json.dumps(POLARFIRE_SMOKE_WORKLOAD, sort_keys=True)
    remote = (
        "python3 - <<'PY'\n"
        "import json, math\n"
        f"payload = {payload_json!r}\n"
        "data = json.loads(payload)\n"
        "spins = [int(value) for value in data['spins']]\n"
        "energy = 0\n"
        "for row, col, coupling in data['edges']:\n"
        "    energy -= int(coupling) * spins[int(row)] * spins[int(col)]\n"
        "print(json.dumps({\n"
        f"    'workload_sha256': {POLARFIRE_SMOKE_WORKLOAD_HASH!r},\n"
        f"    'executable_sha256': {INLINE_EXECUTABLE_HASH!r},\n"
        f"    'inference_substrate': {INFERENCE_SUBSTRATE!r},\n"
        "    'energy': energy,\n"
        "    'sample_quality': {'sample_count': len(spins), 'finite_energy': math.isfinite(float(energy))},\n"
        "    'correctness': {'energy_matches_expected': energy == data['expected_energy']},\n"
        "}, sort_keys=True))\n"
        "PY"
    )
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        "polarfire",
        remote,
    )


POLARFIRE_SMOKE_COMMAND = polarfire_smoke_command()

FIELD_PRINCIPLES: dict[str, str] = {
    "kv260_reachable": "true only when the KV260 SSH BatchMode precondition exits zero.",
    "kv260_check_method": "records the SSH-only method so host-storage preconditions cannot creep back in.",
    "polarfire_reachable": "true only when the PolarFire SSH precondition exits zero.",
    "gatemate_status": "preserve the physical/JTAG blocker unless the operator changed the setup.",
    "gatemate_idcode_raw": "preserve the raw v477 IDCODE evidence or record the bounded recheck result.",
    "pbit_boundary_plan_path": "points to the future sampler timing-ratio plan, not a benchmark result.",
    "speedup_claimed": NO_SPEEDUP_PRINCIPLE,
    "hardware_docs_updated": "true only when the p-bit plan and hardware wishlist note were written.",
    "inference_substrate": "declares this as reachability plus sampler-boundary planning evidence.",
    "honest_verdict": HONEST_VERDICT_PRINCIPLE,
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "spec_refs",
    "run_date",
    "random_seed",
    "duration_s",
    "kv260_reachable",
    "kv260_check_method",
    "polarfire_reachable",
    "gatemate_status",
    "gatemate_idcode_raw",
    "pbit_boundary_plan_path",
    "speedup_claimed",
    "hardware_docs_updated",
    "inference_substrate",
    "honest_verdict",
    "field_principles",
    "preconditions_checked",
    "polarfire_smoke",
    "gatemate_rechecked",
    "gatemate_check_note",
    "gatemate_next_physical_action",
    "pbit_boundary_plan",
    "command_probes",
    "conductor_modified",
    "reproducibility_checksum",
)
GATEMATE_STATUSES = {"reachable", "blocked_physical_jtag", "not_checked"}


@dataclass(frozen=True)
class CommandProbe:
    """One bounded command transcript used for artifact provenance."""

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


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandProbe:
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


def round_duration(duration_s: float) -> float:
    return round(max(float(duration_s), 0.000001), 6)


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


def polarfire_smoke_status(
    *, reachable: bool, smoke_probe: CommandProbe | None
) -> JsonDict:
    if not reachable:
        return {
            "status": "not_run_unreachable",
            "hash_verified": False,
            "correctness_ok": False,
            "workload_hash": None,
            "blocked_reason": "blocked_polarfire_ssh",
        }
    if smoke_probe is None:
        return {
            "status": "not_run_missing_probe",
            "hash_verified": False,
            "correctness_ok": False,
            "workload_hash": None,
            "blocked_reason": "blocked_polarfire_smoke_missing",
        }

    output = parse_last_json(smoke_probe.combined_output) if smoke_probe.exit_code == 0 else {}
    correctness = output.get("correctness") if isinstance(output.get("correctness"), Mapping) else {}
    hash_verified = (
        output.get("workload_sha256") == POLARFIRE_SMOKE_WORKLOAD_HASH
        and output.get("executable_sha256") == INLINE_EXECUTABLE_HASH
        and output.get("inference_substrate") == INFERENCE_SUBSTRATE
    )
    correctness_ok = (
        correctness.get("energy_matches_expected") is True
        and output.get("energy") == POLARFIRE_EXPECTED_ENERGY
    )
    return {
        "status": "hash_verified_correctness_ok"
        if hash_verified and correctness_ok
        else "smoke_failed_validation",
        "hash_verified": hash_verified,
        "correctness_ok": correctness_ok,
        "workload_hash": output.get("workload_sha256"),
        "expected_workload_hash": POLARFIRE_SMOKE_WORKLOAD_HASH,
        "output": output,
        "blocked_reason": None if hash_verified and correctness_ok else "blocked_polarfire_smoke",
    }


def raw_idcode_from_text(text: str) -> str | None:
    match = re.search(
        r"(?:raw idcode.*?->\s*|idcode\s+)(0x[0-9a-fA-F]+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).lower() if match else None


def is_floating_tdo(raw_idcode: str | None) -> bool:
    if raw_idcode is None:
        return False
    body = raw_idcode.lower().removeprefix("0x")
    return bool(body) and (set(body) <= {"f"} or set(body) <= {"0"})


def gatemate_from_probe(
    *, setup_changed: bool, probe: CommandProbe | None
) -> JsonDict:
    if not setup_changed:
        return {
            "status": "blocked_physical_jtag",
            "raw_idcode": PRIOR_GATEMATE_RAW_IDCODE,
            "rechecked": False,
            "note": "preserved v477 physical/JTAG block; no operator cable, port, or board setup change was provided",
            "next_action": "operator reseat or replace GateMate JTAG cable/port path and verify board power before another IDCODE loop",
        }
    if probe is None or probe.exit_code != 0:
        return {
            "status": "not_checked",
            "raw_idcode": None,
            "rechecked": probe is not None,
            "note": "operator setup changed, but the bounded GateMate debug detect did not produce usable IDCODE evidence",
            "next_action": "operator capture a debug-level IDCODE transcript after cable/port/board setup is stable",
        }

    raw_idcode = raw_idcode_from_text(probe.combined_output)
    if raw_idcode == GATEMATE_EXPECTED_IDCODE:
        return {
            "status": "reachable",
            "raw_idcode": raw_idcode,
            "rechecked": True,
            "note": "operator setup changed and the bounded debug detect read the expected GM1Ax IDCODE",
            "next_action": "resume GateMate Ising tile smoke only after preserving this restored IDCODE transcript",
        }
    if is_floating_tdo(raw_idcode):
        return {
            "status": "blocked_physical_jtag",
            "raw_idcode": raw_idcode,
            "rechecked": True,
            "note": "operator setup changed, but raw all-ones/all-zeros TDO still indicates the physical/JTAG block",
            "next_action": "operator continue physical cable/port/power diagnosis before more software probing",
        }
    return {
        "status": "not_checked",
        "raw_idcode": raw_idcode,
        "rechecked": True,
        "note": "operator setup changed, but the bounded recheck did not confirm the expected IDCODE or the preserved physical signature",
        "next_action": "operator inspect the raw IDCODE transcript before classifying a new GateMate failure layer",
    }


def precondition_entry(
    *, board: str, resource: str, probe: CommandProbe, available: bool, discipline: str
) -> JsonDict:
    return {
        "board": board,
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "exit_code": int(probe.exit_code),
        "duration_s": round_duration(probe.duration_s),
        "observed": probe.combined_output,
        "discipline": discipline,
    }


def build_pbit_boundary_plan() -> str:
    """Return the short future p-bit timing-ratio plan written by this task."""

    return f"""# Exp 5231 p-bit boundary-exchange timing-ratio plan

Status: future sampler plan only. No speedup claim is allowed from this document.

Future workload: `{PBIT_FUTURE_WORKLOAD}`.

partitions: {PBIT_PARTITIONS}. Each partition owns a sparse Ising subgraph and emits a deterministic boundary packet after each exchange interval. The first implementation target is CPU reference plus board-local or simulator parity; million-p-bit and TSU/XTR-0 references remain motivation only until real device evidence exists.

hash/correctness checks:
- Hash the graph CSR, partition map, seeds, exchange interval, and each boundary packet stream.
- Compare CPU reference energy, final state checksum, per-partition boundary packet hashes, and accepted exchange count.
- Record the exact command transcript for each substrate and reject any run with missing hash or correctness fields.

Measurement needed before speedup:
- Same workload, same seeds, same partition map, and same exchange schedule on CPU baseline and hardware candidate.
- End-to-end wall clock including partition setup, boundary exchange, device transfer, sampler steps, and result validation.
- Only a hash/correctness-passing end-to-end workload may support a timing-ratio or speedup statement.
"""


def pbit_boundary_plan_summary() -> JsonDict:
    return {
        "future_workload": PBIT_FUTURE_WORKLOAD,
        "partitions": PBIT_PARTITIONS,
        "hash_correctness_checks": [
            "graph_csr_hash",
            "partition_map_hash",
            "seed_hash",
            "boundary_packet_stream_hash",
            "cpu_reference_energy",
            "final_state_checksum",
        ],
        "measurement_before_speedup": (
            "same workload/seeds/partitions/exchange schedule, end-to-end wall clock, "
            "boundary-transfer overhead, and hash/correctness parity on each substrate"
        ),
        "speedup_claim_allowed": False,
    }


def build_honest_verdict(
    *, kv260_reachable: bool, polarfire_reachable: bool, gatemate_status: str
) -> str:
    kv = "reachable" if kv260_reachable else "blocked_kv260_ssh"
    pf = "reachable" if polarfire_reachable else "blocked_polarfire_ssh"
    return (
        "complete_hardware_continuity_pbit_boundary_v478_"
        f"kv260:{kv}_polarfire:{pf}_gatemate:{gatemate_status}_no_speedup"
    )


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = "20260704",
    hardware_docs_updated: bool = True,
    gatemate_setup_changed: bool = False,
) -> JsonDict:
    """Run bounded checks and return the v478 artifact."""

    started = clock()
    kv260_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    polarfire_probe = command_runner(POLARFIRE_SSH_COMMAND, 10.0)
    polarfire_reachable = polarfire_probe.exit_code == 0
    polarfire_smoke_probe = (
        command_runner(POLARFIRE_SMOKE_COMMAND, 30.0) if polarfire_reachable else None
    )
    gatemate_probe = (
        command_runner(GATEMATE_DEBUG_DETECT_COMMAND, 30.0) if gatemate_setup_changed else None
    )
    gatemate = gatemate_from_probe(setup_changed=gatemate_setup_changed, probe=gatemate_probe)
    kv260_reachable = kv260_probe.exit_code == 0

    preconditions = [
        precondition_entry(
            board="kv260",
            resource="kv260_ssh",
            probe=kv260_probe,
            available=kv260_reachable,
            discipline="ssh_only",
        ),
        precondition_entry(
            board="polarfire",
            resource="polarfire_ssh",
            probe=polarfire_probe,
            available=polarfire_reachable,
            discipline="ssh_only",
        ),
    ]
    if gatemate_probe is not None:
        preconditions.append(
            precondition_entry(
                board="gatemate",
                resource="gatemate_dirtyjtag_debug_idcode",
                probe=gatemate_probe,
                available=gatemate["status"] == "reachable",
                discipline="operator_setup_changed_only",
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
        "kv260_reachable": kv260_reachable,
        "kv260_check_method": "ssh_only",
        "polarfire_reachable": polarfire_reachable,
        "gatemate_status": gatemate["status"],
        "gatemate_idcode_raw": gatemate["raw_idcode"],
        "pbit_boundary_plan_path": str(PBIT_PLAN_REL_PATH),
        "speedup_claimed": False,
        "hardware_docs_updated": bool(hardware_docs_updated),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": build_honest_verdict(
            kv260_reachable=kv260_reachable,
            polarfire_reachable=polarfire_reachable,
            gatemate_status=gatemate["status"],
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "polarfire_smoke": polarfire_smoke_status(
            reachable=polarfire_reachable, smoke_probe=polarfire_smoke_probe
        ),
        "gatemate_rechecked": bool(gatemate["rechecked"]),
        "gatemate_check_note": gatemate["note"],
        "gatemate_next_physical_action": gatemate["next_action"],
        "gatemate_operator_setup_changed": bool(gatemate_setup_changed),
        "prior_gatemate_reference": {
            "path": "results/experiment_5217_hardware_continuity_v477.json",
            "raw_idcode": PRIOR_GATEMATE_RAW_IDCODE,
            "status": "physical/JTAG block preserved unless operator setup changed",
        },
        "pbit_boundary_plan": pbit_boundary_plan_summary(),
        "command_probes": {
            "kv260_ssh": kv260_probe.as_dict(),
            "polarfire_ssh": polarfire_probe.as_dict(),
            "polarfire_smoke": polarfire_smoke_probe.as_dict()
            if polarfire_smoke_probe is not None
            else None,
            "gatemate_debug_detect": gatemate_probe.as_dict() if gatemate_probe is not None else None,
        },
        "conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def terminal_prefix_ok(verdict: str) -> bool:
    return verdict.startswith(("complete:", "complete_", "success:", "success_"))


def no_host_storage_markers(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    return "mmcblk" not in encoded and "/dev/disk" not in encoded


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
        return errors
    expect(errors, artifact.get("schema") == SCHEMA, "schema mismatch")
    expect(errors, artifact.get("experiment") == EXPERIMENT_NAME, "experiment mismatch")
    expect(errors, artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id mismatch")
    expect(errors, artifact.get("milestone") == MILESTONE, "milestone mismatch")
    expect(errors, artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    expect(errors, artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    expect(errors, isinstance(artifact.get("kv260_reachable"), bool), "kv260_reachable type")
    expect(errors, artifact.get("kv260_check_method") == "ssh_only", "kv260_check_method mismatch")
    expect(errors, isinstance(artifact.get("polarfire_reachable"), bool), "polarfire_reachable type")
    expect(errors, artifact.get("gatemate_status") in GATEMATE_STATUSES, "gatemate_status invalid")
    expect(
        errors,
        artifact.get("gatemate_idcode_raw") is None
        or isinstance(artifact.get("gatemate_idcode_raw"), str),
        "gatemate_idcode_raw type",
    )
    expect(
        errors,
        artifact.get("pbit_boundary_plan_path") == str(PBIT_PLAN_REL_PATH),
        "pbit_boundary_plan_path mismatch",
    )
    expect(errors, artifact.get("speedup_claimed") is False, "speedup_claimed must be false")
    expect(
        errors,
        artifact.get("hardware_docs_updated") is True,
        "hardware_docs_updated must be true",
    )
    expect(
        errors,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate mismatch",
    )
    expect(errors, artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    verdict = str(artifact.get("honest_verdict", ""))
    expect(errors, terminal_prefix_ok(verdict), "honest_verdict prefix mismatch")
    expect(errors, "no_speedup" in verdict, "honest_verdict must state no_speedup")
    validate_preconditions(errors, artifact)
    validate_polarfire_smoke(errors, artifact)
    validate_pbit_plan(errors, artifact)
    expect(errors, artifact.get("conductor_modified") is False, "conductor_modified mismatch")
    expect(errors, no_host_storage_markers(artifact), "host storage marker present")
    expect(errors, artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum mismatch")
    return errors


def validate_preconditions(errors: list[str], artifact: Mapping[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list) or len(preconditions) < 2:
        errors.append("preconditions_checked invalid")
        return
    first = preconditions[0]
    second = preconditions[1]
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        errors.append("preconditions_checked entries invalid")
        return
    expect(errors, first.get("resource") == "kv260_ssh", "kv260 precondition resource mismatch")
    expect(errors, first.get("command") == command_to_string(KV260_SSH_COMMAND), "kv260 command mismatch")
    expect(errors, first.get("discipline") == "ssh_only", "kv260 discipline mismatch")
    expect(errors, first.get("available") is artifact.get("kv260_reachable"), "kv260 availability mismatch")
    expect(errors, second.get("resource") == "polarfire_ssh", "polarfire precondition resource mismatch")
    expect(
        errors,
        second.get("available") is artifact.get("polarfire_reachable"),
        "polarfire availability mismatch",
    )


def validate_polarfire_smoke(errors: list[str], artifact: Mapping[str, Any]) -> None:
    smoke = artifact.get("polarfire_smoke")
    if not isinstance(smoke, Mapping):
        errors.append("polarfire_smoke invalid")
        return
    if artifact.get("polarfire_reachable") is False:
        expect(errors, smoke.get("status") == "not_run_unreachable", "polarfire smoke should be skipped")
        return
    expect(errors, "hash_verified" in smoke, "polarfire smoke missing hash_verified")
    expect(errors, "correctness_ok" in smoke, "polarfire smoke missing correctness_ok")


def validate_pbit_plan(errors: list[str], artifact: Mapping[str, Any]) -> None:
    plan = artifact.get("pbit_boundary_plan")
    if not isinstance(plan, Mapping):
        errors.append("pbit_boundary_plan invalid")
        return
    expect(errors, plan.get("future_workload") == PBIT_FUTURE_WORKLOAD, "future workload mismatch")
    expect(errors, plan.get("partitions") == PBIT_PARTITIONS, "partitions mismatch")
    expect(errors, plan.get("speedup_claim_allowed") is False, "plan speedup flag mismatch")


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_pbit_boundary_plan(repo_root: str | Path) -> Path:
    out_path = Path(repo_root) / PBIT_PLAN_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_pbit_boundary_plan(), encoding="utf-8")
    return out_path


def update_hardware_wishlist(repo_root: str | Path) -> bool:
    path = Path(repo_root) / "research-hardware-wishlist.md"
    if not path.exists():
        path.write_text("# Hardware wishlist\n\n", encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    start = "<!-- exp5231-pbit-boundary-plan:start -->"
    end = "<!-- exp5231-pbit-boundary-plan:end -->"
    note = (
        f"{start}\n"
        "### Exp 5231 hardware continuity + p-bit boundary plan (2026-07-04)\n"
        "\n"
        "- KV260 continuity remains SSH-only; no benchmark speedup is claimed.\n"
        "- PolarFire continuity uses SSH plus a bounded hash/correctness smoke when reachable.\n"
        "- GateMate preserves the v477 physical/JTAG block (`0xffffffff`) until the operator changes cable, port, or board power setup.\n"
        f"- Future sampler plan: `{PBIT_PLAN_REL_PATH}` for `{PBIT_FUTURE_WORKLOAD}`.\n"
        f"{end}\n"
    )
    if start in text and end in text:
        text = re.sub(f"{re.escape(start)}.*?{re.escape(end)}\n?", note, text, flags=re.DOTALL)
    else:
        suffix = "" if text.endswith("\n") else "\n"
        text = f"{text}{suffix}\n{note}"
    path.write_text(text, encoding="utf-8")
    return True


def write_artifact(repo_root: str | Path, artifact: Mapping[str, Any]) -> Path:
    validate_artifact(artifact)
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = "20260704",
    hardware_docs_updated: bool | None = None,
    gatemate_setup_changed: bool = False,
) -> Path:
    plan_written = write_pbit_boundary_plan(repo_root).exists()
    wishlist_updated = update_hardware_wishlist(repo_root)
    docs_updated = plan_written and wishlist_updated if hardware_docs_updated is None else hardware_docs_updated
    artifact = build_artifact(
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        hardware_docs_updated=docs_updated,
        gatemate_setup_changed=gatemate_setup_changed,
    )
    return write_artifact(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260704")
    parser.add_argument(
        "--gatemate-setup-changed",
        action="store_true",
        help="Run the bounded GateMate debug IDCODE recheck because physical setup changed.",
    )
    args = parser.parse_args(argv)
    out_path = run_experiment(
        repo_root=Path("."),
        run_date=args.date,
        gatemate_setup_changed=args.gatemate_setup_changed,
    )
    print(out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI in integration use.
    raise SystemExit(main())
