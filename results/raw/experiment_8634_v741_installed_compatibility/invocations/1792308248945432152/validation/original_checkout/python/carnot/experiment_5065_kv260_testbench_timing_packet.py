#!/usr/bin/env python3
"""Exp 5065 KV260 transcript-backed testbench timing packet.

Spec refs: REQ-HW-5065, SCENARIO-HW-5065.

This experiment extends Exp 5052 by writing the exact SSH command transcript
that supports the board-side parity and timing evidence. The result is scoped
to a local SSH-attached KV260 Python testbench after the board confirms the
existing `carnot_ising_v2_n64` overlay path; it is not a generalized FPGA
speedup claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_5052_kv260_pbit_timing_ratio as exp5052


JsonDict = dict[str, Any]
Clock = Callable[[], float]

EXPERIMENT_ID = 5065
SCHEMA = "carnot.kv260_testbench_timing_packet.v1"
SPEC_REFS = ["REQ-HW-5065", "SCENARIO-HW-5065"]
OUTPUT_REL_PATH = Path("results") / "experiment_5065_kv260_testbench_timing_packet.json"
TRANSCRIPT_REL_PATH = (
    Path("results") / "experiment_5065_kv260_testbench_timing_packet.transcript.jsonl"
)
INFERENCE_SUBSTRATE = "hardware_smoke"
EXPECTED_OVERLAY = "carnot_ising_v2_n64"
RANDOM_SEED = 5065
WORKLOAD_SEED = exp5052.RANDOM_SEED
WORKLOAD_NAME = exp5052.WORKLOAD_NAME
N_VARIABLES = exp5052.N_VARIABLES
ITERATIONS = exp5052.ITERATIONS
LOCAL_CLAIM_SCOPE = (
    "local_ssh_attached_kv260_python_testbench_on_confirmed_carnot_overlay_only_"
    "no_general_fpga_speedup_claim_no_gpu_benchmark_claim_no_external_2026_paper_claim"
)

CommandProbe = exp5052.CommandProbe
CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]

KV260_SSH_COMMAND = exp5052.KV260_SSH_COMMAND
KV260_LISTAPPS_COMMAND = exp5052.KV260_LISTAPPS_COMMAND
KV260_LISTAPPS_SUDO_COMMAND = exp5052.KV260_LISTAPPS_SUDO_COMMAND
KV260_UIO_COMMAND = exp5052.KV260_UIO_COMMAND
KV260_TESTBENCH_COMMAND = exp5052.KV260_PBIT_WORKLOAD_COMMAND

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "kv260_ssh_reachable",
    "overlay_loaded",
    "loaded_overlay",
    "cpu_reference_ok",
    "kv260_result_ok",
    "timing_ratio_packet_built",
    "board_transcript_path",
    "transcript_sha256",
    "structured_testbench_evidence",
    "local_claim_scope",
    "optional_board_prechecks",
    "schema",
    "experiment",
    "spec_refs",
    "field_principles",
    "inference_substrate",
    "preconditions_checked",
    "overlay_state",
    "uio_devices",
    "xmutil_requires_sudo",
    "cpu_reference",
    "kv260_workload",
    "timing_ratio_packet",
    "duration_s",
    "command_probes",
    "verifier_is_oracle",
    "random_seed",
    "workload_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix for blocked SSH, overlay-not-confirmed, parity failure, or packet-built outcomes.",
    "kv260_ssh_reachable": "true only when the SSH BatchMode precondition exits zero; never host SD-card.",
    "overlay_loaded": f"true only when the board transcript confirms {EXPECTED_OVERLAY} is loaded.",
    "loaded_overlay": "the board-reported loaded Carnot overlay, or null when none is confirmed.",
    "cpu_reference_ok": "true only when the deterministic local CPU reference is structurally valid.",
    "kv260_result_ok": "true only when the board testbench exits zero and matches the CPU reference.",
    "timing_ratio_packet_built": "true only when CPU and KV260 parity are both valid.",
    "board_transcript_path": "relative path to the JSONL command transcript backing board evidence.",
    "transcript_sha256": "SHA-256 digest of the exact board transcript bytes.",
    "structured_testbench_evidence": "machine-readable CPU parity, board result, timing, mismatch, and transcript evidence.",
    "local_claim_scope": "limits claims to local SSH evidence without generalized speedup or external paper claims.",
    "optional_board_prechecks": "GateMate/PolarFire may appear only as non-flashing reachability prechecks.",
}


@dataclass(frozen=True)
class BuiltPacket:
    """Validated Exp 5065 artifact plus the exact transcript text to write."""

    artifact: JsonDict
    transcript_text: str


command_to_string = exp5052.command_to_string
payload_checksum = exp5052.payload_checksum
run_command = exp5052.run_command


def run_cpu_reference(*, clock: Clock = time.perf_counter) -> JsonDict:
    """Run the same deterministic CPU reference used by the Exp 5052 packet."""

    return exp5052.run_pbit_reference(clock=clock)


def parse_testbench_stdout(stdout: str) -> JsonDict | None:
    """Extract the final JSON object printed by the SSH testbench command."""

    return exp5052.parse_workload_stdout(stdout)


def parse_uio_devices(text: str) -> list[str]:
    """Return unique `/dev/uio*` device names from the SSH UIO probe text."""

    return exp5052.parse_uio_devices(text)


def confirmed_existing_overlay(text: str) -> tuple[bool, str | None]:
    """Report whether the board confirmed the existing KV260 overlay path."""

    loaded_overlay = exp5052.loaded_overlay_from_xmutil(text)
    return loaded_overlay == EXPECTED_OVERLAY, loaded_overlay


def build_packet(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> BuiltPacket:
    """Build the transcript-backed Exp 5065 packet from local and SSH evidence."""

    started = clock()
    cpu_reference = run_cpu_reference(clock=clock)
    ssh_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    artifact = _base_artifact(
        cpu_reference=cpu_reference,
        ssh_probe=ssh_probe,
        duration_s=clock() - started,
    )

    if ssh_probe.exit_code != 0:
        return _finalize(artifact)

    command_probes = _command_probes(artifact)
    listapps_probe = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
    command_probes["kv260_xmutil_listapps"] = listapps_probe.as_dict()
    listapps_text = listapps_probe.combined_output if listapps_probe.exit_code == 0 else ""
    requires_sudo = _xmutil_requires_root(listapps_probe)
    if requires_sudo:
        sudo_probe = command_runner(KV260_LISTAPPS_SUDO_COMMAND, 30.0)
        command_probes["kv260_xmutil_listapps_sudo"] = sudo_probe.as_dict()
        if sudo_probe.exit_code == 0:
            listapps_text = sudo_probe.combined_output

    uio_probe = command_runner(KV260_UIO_COMMAND, 10.0)
    command_probes["kv260_uio_devices"] = uio_probe.as_dict()
    overlay_confirmed, loaded_overlay = confirmed_existing_overlay(listapps_text)
    uio_devices = parse_uio_devices(uio_probe.combined_output if uio_probe.exit_code == 0 else "")
    artifact.update(
        {
            "honest_verdict": "blocked_kv260_expected_overlay_not_confirmed",
            "kv260_ssh_reachable": True,
            "overlay_loaded": overlay_confirmed,
            "overlay_state": _overlay_state(
                listapps_probe=listapps_probe,
                listapps_text=listapps_text,
                loaded_overlay=loaded_overlay,
                overlay_confirmed=overlay_confirmed,
                uio_probe=uio_probe,
                uio_devices=uio_devices,
                requires_sudo=requires_sudo,
            ),
            "loaded_overlay": loaded_overlay,
            "uio_devices": uio_devices,
            "xmutil_requires_sudo": requires_sudo,
            "duration_s": _duration_floor(clock() - started),
        }
    )

    if not overlay_confirmed:
        return _finalize(artifact)

    testbench_probe = command_runner(KV260_TESTBENCH_COMMAND, 30.0)
    command_probes["kv260_testbench_workload"] = testbench_probe.as_dict()
    kv260_workload = (
        parse_testbench_stdout(testbench_probe.stdout) if testbench_probe.exit_code == 0 else None
    )
    kv260_result_ok = _workload_matches(cpu_reference, kv260_workload)
    timing_ratio_packet = (
        _timing_ratio_packet(cpu_reference, kv260_workload, testbench_probe)
        if artifact["cpu_reference_ok"] and kv260_result_ok
        else None
    )
    artifact.update(
        {
            "honest_verdict": "success_kv260_testbench_timing_packet_built"
            if timing_ratio_packet is not None
            else "blocked_kv260_testbench_parity_failed",
            "kv260_result_ok": kv260_result_ok,
            "kv260_workload": kv260_workload,
            "timing_ratio_packet_built": timing_ratio_packet is not None,
            "timing_ratio_packet": timing_ratio_packet,
            "duration_s": _duration_floor(clock() - started),
        }
    )
    return _finalize(artifact)


def write_packet(repo_root: str | Path, packet: BuiltPacket) -> Path:
    """Write the validated artifact and the transcript file it hashes."""

    validate_artifact(packet.artifact, transcript_text=packet.transcript_text)
    root = Path(repo_root)
    transcript_path = root / TRANSCRIPT_REL_PATH
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(packet.transcript_text, encoding="utf-8")
    out_path = root / OUTPUT_REL_PATH
    out_path.write_text(
        json.dumps(packet.artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 5065 and write the JSON artifact plus transcript."""

    packet = build_packet(command_runner=command_runner, clock=clock)
    return write_packet(repo_root, packet)


def validate_artifact(artifact: JsonDict, *, transcript_text: str | None = None) -> None:
    """Validate Exp 5065's schema, claim scope, parity gates, and transcript hash."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(artifact.get("schema") == SCHEMA, "schema mismatch")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment mismatch")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle must be false")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    _require(artifact.get("workload_seed") == WORKLOAD_SEED, "workload_seed mismatch")
    _require(float(artifact.get("duration_s", 0.0)) >= 0.0001, "duration_s below floor")
    _require(artifact.get("local_claim_scope") == LOCAL_CLAIM_SCOPE, "local_claim_scope mismatch")
    _require(_cpu_reference_ok(artifact.get("cpu_reference")), "cpu_reference invalid")
    _require(artifact.get("cpu_reference_ok") is True, "cpu_reference_ok must be true")
    _require(
        artifact.get("board_transcript_path") == str(TRANSCRIPT_REL_PATH),
        "board_transcript_path mismatch",
    )
    _require(_is_sha256(artifact.get("transcript_sha256")), "transcript_sha256 invalid")
    _validate_no_host_storage(artifact)
    _validate_precondition(artifact)
    _validate_optional_prechecks(artifact)
    _validate_overlay_and_workload(artifact)
    _validate_structured_evidence(artifact)
    if transcript_text is not None:
        _require(
            artifact.get("transcript_sha256") == _sha256_text(transcript_text),
            "transcript hash mismatch",
        )
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _base_artifact(
    *, cpu_reference: JsonDict, ssh_probe: CommandProbe, duration_s: float
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": "blocked_kv260_ssh_unreachable",
        "kv260_ssh_reachable": ssh_probe.exit_code == 0,
        "overlay_loaded": False,
        "loaded_overlay": None,
        "cpu_reference_ok": _cpu_reference_ok(cpu_reference),
        "kv260_result_ok": False,
        "timing_ratio_packet_built": False,
        "board_transcript_path": str(TRANSCRIPT_REL_PATH),
        "transcript_sha256": "",
        "structured_testbench_evidence": {},
        "local_claim_scope": LOCAL_CLAIM_SCOPE,
        "optional_board_prechecks": _optional_board_prechecks(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": [_precondition_entry(ssh_probe)],
        "overlay_state": _empty_overlay_state(),
        "uio_devices": [],
        "xmutil_requires_sudo": False,
        "cpu_reference": dict(cpu_reference),
        "kv260_workload": None,
        "timing_ratio_packet": None,
        "duration_s": _duration_floor(duration_s),
        "command_probes": {
            "kv260_ssh": ssh_probe.as_dict(),
            "kv260_xmutil_listapps": None,
            "kv260_xmutil_listapps_sudo": None,
            "kv260_uio_devices": None,
            "kv260_testbench_workload": None,
        },
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "workload_seed": WORKLOAD_SEED,
        "reproducibility_checksum": "",
    }


def _empty_overlay_state() -> JsonDict:
    return {
        "expected_overlay": EXPECTED_OVERLAY,
        "expected_overlay_confirmed": False,
        "loaded_overlay": None,
        "overlay_loaded": False,
        "xmutil_listapps_output": None,
        "xmutil_listapps_exit_code": None,
        "xmutil_requires_sudo": False,
        "uio_devices": [],
        "uio_output": None,
        "uio_exit_code": None,
    }


def _overlay_state(
    *,
    listapps_probe: CommandProbe,
    listapps_text: str,
    loaded_overlay: str | None,
    overlay_confirmed: bool,
    uio_probe: CommandProbe,
    uio_devices: list[str],
    requires_sudo: bool,
) -> JsonDict:
    return {
        "expected_overlay": EXPECTED_OVERLAY,
        "expected_overlay_confirmed": overlay_confirmed,
        "loaded_overlay": loaded_overlay,
        "overlay_loaded": overlay_confirmed,
        "xmutil_listapps_output": listapps_text,
        "xmutil_listapps_exit_code": listapps_probe.exit_code,
        "xmutil_requires_sudo": requires_sudo,
        "uio_devices": list(uio_devices),
        "uio_output": uio_probe.combined_output,
        "uio_exit_code": uio_probe.exit_code,
    }


def _optional_board_prechecks() -> JsonDict:
    return {
        "gatemate": {
            "status": "not_run_scope_guard",
            "scope": "reachability_precheck_only_no_flash_no_latency_claim",
        },
        "polarfire": {
            "status": "not_run_scope_guard",
            "scope": "reachability_precheck_only_no_flash_no_latency_claim",
        },
    }


def _precondition_entry(ssh_probe: CommandProbe) -> JsonDict:
    return {
        "resource": "kv260_ssh",
        "available": ssh_probe.exit_code == 0,
        "command": command_to_string(KV260_SSH_COMMAND),
        "exit_code": ssh_probe.exit_code,
        "duration_s": float(ssh_probe.duration_s),
        "observed": _observed_first_line(ssh_probe),
        "discipline": "ssh_only_no_host_sd_card",
    }


def _structured_testbench_evidence(artifact: JsonDict, transcript_sha256: str) -> JsonDict:
    mismatches = _mismatch_fields(artifact["cpu_reference"], artifact.get("kv260_workload"))
    status = artifact["honest_verdict"].removeprefix("success_kv260_testbench_timing_")
    if artifact.get("timing_ratio_packet_built") is True:
        status = "packet_built"
    return {
        "schema": "carnot.kv260.structured_testbench_evidence.v1",
        "status": status,
        "spec_refs": list(SPEC_REFS),
        "workload_name": WORKLOAD_NAME,
        "cpu_reference": dict(artifact["cpu_reference"]),
        "board_result": artifact.get("kv260_workload"),
        "parity": {
            "cpu_reference_ok": artifact.get("cpu_reference_ok"),
            "kv260_result_ok": artifact.get("kv260_result_ok"),
            "checked_fields": _PARITY_FIELDS,
            "mismatches": mismatches,
        },
        "timing_ratio_packet": artifact.get("timing_ratio_packet"),
        "transcript": {
            "path": artifact["board_transcript_path"],
            "sha256": transcript_sha256,
        },
        "claim_scope": artifact["local_claim_scope"],
    }


_PARITY_FIELDS = (
    "workload_name",
    "n_variables",
    "iterations",
    "flips",
    "energy",
    "final_state_checksum",
)


def _timing_ratio_packet(
    cpu_reference: JsonDict, kv260_workload: JsonDict, probe: CommandProbe
) -> JsonDict:
    cpu_s = _duration_positive(cpu_reference["duration_s"])
    command_s = _duration_positive(probe.duration_s)
    board_s = _duration_positive(kv260_workload["duration_s"])
    return {
        "workload_name": WORKLOAD_NAME,
        "n_variables": N_VARIABLES,
        "iterations": int(cpu_reference["iterations"]),
        "flips": int(cpu_reference["flips"]),
        "cpu_wall_clock_s": cpu_s,
        "kv260_command_wall_clock_s": command_s,
        "kv260_board_reported_workload_s": board_s,
        "cpu_to_kv260_command_wall_ratio": round(cpu_s / command_s, 12),
        "cpu_to_kv260_board_workload_ratio": round(cpu_s / board_s, 12),
        "parity_match": True,
        "ratio_claim_scope": LOCAL_CLAIM_SCOPE,
    }


def _transcript_text(artifact: JsonDict) -> str:
    lines = [json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in _transcript_entries(artifact)]
    return "\n".join(lines) + "\n"


def _transcript_entries(artifact: JsonDict) -> list[JsonDict]:
    probes = _command_probes(artifact)
    entries = [
        {
            "label": "local_cpu_reference",
            "workload_name": WORKLOAD_NAME,
            "cpu_reference_ok": artifact["cpu_reference_ok"],
            "cpu_reference": artifact["cpu_reference"],
        }
    ]
    for label in (
        "kv260_ssh",
        "kv260_xmutil_listapps",
        "kv260_xmutil_listapps_sudo",
        "kv260_uio_devices",
        "kv260_testbench_workload",
    ):
        probe = probes.get(label)
        if probe is not None:
            entries.append({"label": label, **dict(probe)})
    return entries


def _cpu_reference_ok(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("workload_name") == WORKLOAD_NAME
        and payload.get("n_variables") == N_VARIABLES
        and payload.get("iterations") == ITERATIONS
        and isinstance(payload.get("flips"), int)
        and isinstance(payload.get("energy"), int)
        and isinstance(payload.get("final_state_checksum"), str)
        and len(str(payload.get("final_state_checksum"))) == 64
        and float(payload.get("duration_s", 0.0)) > 0.0
    )


def _workload_matches(cpu_reference: JsonDict, kv260_workload: JsonDict | None) -> bool:
    if not isinstance(kv260_workload, Mapping):
        return False
    return (
        all(kv260_workload.get(field) == cpu_reference.get(field) for field in _PARITY_FIELDS)
        and float(kv260_workload.get("duration_s", 0.0)) > 0.0
    )


def _mismatch_fields(cpu_reference: JsonDict, kv260_workload: Any) -> list[str]:
    if not isinstance(kv260_workload, Mapping):
        return list(_PARITY_FIELDS)
    return [field for field in _PARITY_FIELDS if kv260_workload.get(field) != cpu_reference.get(field)]


def _validate_no_host_storage(artifact: JsonDict) -> None:
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require(
        "mmcblk" not in encoded and "/dev/disk" not in encoded, "forbidden host storage marker"
    )


def _validate_precondition(artifact: JsonDict) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(
        isinstance(preconditions, list) and len(preconditions) == 1, "bad preconditions_checked"
    )
    entry = preconditions[0]
    _require(isinstance(entry, Mapping), "bad precondition entry")
    _require(entry.get("resource") == "kv260_ssh", "bad KV260 precondition resource")
    _require(entry.get("command") == command_to_string(KV260_SSH_COMMAND), "bad KV260 SSH command")
    _require(entry.get("discipline") == "ssh_only_no_host_sd_card", "bad KV260 discipline")
    _require(entry.get("available") is artifact.get("kv260_ssh_reachable"), "precondition mismatch")


def _validate_optional_prechecks(artifact: JsonDict) -> None:
    prechecks = artifact.get("optional_board_prechecks")
    _require(isinstance(prechecks, Mapping), "optional_board_prechecks must be a dict")
    encoded = json.dumps(prechecks, sort_keys=True).lower()
    _require("flash" not in encoded.replace("no_flash", ""), "optional prechecks cannot flash")
    _require("latency" not in encoded.replace("no_latency_claim", ""), "optional prechecks cannot claim latency")


def _validate_overlay_and_workload(artifact: JsonDict) -> None:
    probes = _command_probes(artifact)
    if artifact.get("kv260_ssh_reachable") is False:
        _require(artifact.get("honest_verdict") == "blocked_kv260_ssh_unreachable", "bad SSH verdict")
        _require(artifact.get("overlay_loaded") is False, "blocked SSH cannot confirm overlay")
        _require(artifact.get("kv260_workload") is None, "blocked SSH cannot have KV260 workload")
        _require(probes.get("kv260_xmutil_listapps") is None, "blocked SSH cannot run xmutil")
        _require(probes.get("kv260_testbench_workload") is None, "blocked SSH cannot run testbench")
        return

    _require(probes.get("kv260_xmutil_listapps") is not None, "reachable SSH requires xmutil")
    _require(probes.get("kv260_uio_devices") is not None, "reachable SSH requires UIO probe")
    overlay_state = artifact.get("overlay_state")
    _require(isinstance(overlay_state, Mapping), "overlay_state must be a dict")
    _require(overlay_state.get("loaded_overlay") == artifact.get("loaded_overlay"), "overlay mismatch")
    _require(overlay_state.get("uio_devices") == artifact.get("uio_devices"), "UIO mismatch")
    _require(overlay_state.get("overlay_loaded") is artifact.get("overlay_loaded"), "overlay flag mismatch")
    if artifact.get("overlay_loaded") is False:
        _require(
            artifact.get("honest_verdict") == "blocked_kv260_expected_overlay_not_confirmed",
            "bad overlay verdict",
        )
        _require(artifact.get("timing_ratio_packet_built") is False, "no overlay cannot build timing")
        _require(artifact.get("kv260_result_ok") is False, "no overlay cannot have KV260 result")
        _require(artifact.get("kv260_workload") is None, "no overlay cannot run workload")
        _require(probes.get("kv260_testbench_workload") is None, "no overlay cannot run testbench")
        return

    _require(artifact.get("loaded_overlay") == EXPECTED_OVERLAY, "expected overlay mismatch")
    _require(
        probes.get("kv260_testbench_workload") is not None, "loaded overlay requires testbench"
    )
    _require(
        artifact.get("kv260_result_ok")
        is _workload_matches(artifact["cpu_reference"], artifact.get("kv260_workload")),
        "parity mismatch",
    )
    if artifact.get("timing_ratio_packet_built"):
        packet = artifact.get("timing_ratio_packet")
        _require(artifact.get("honest_verdict") == "success_kv260_testbench_timing_packet_built", "bad success verdict")
        _require(artifact.get("kv260_result_ok") is True, "timing packet requires KV260 result")
        _require(
            isinstance(packet, Mapping) and packet.get("parity_match") is True, "bad timing packet"
        )
    else:
        _require(artifact.get("honest_verdict") == "blocked_kv260_testbench_parity_failed", "bad parity verdict")
        _require(artifact.get("timing_ratio_packet") is None, "failed parity cannot keep timing")


def _validate_structured_evidence(artifact: JsonDict) -> None:
    evidence = artifact.get("structured_testbench_evidence")
    _require(isinstance(evidence, Mapping), "structured_testbench_evidence must be a dict")
    _require(evidence.get("schema") == "carnot.kv260.structured_testbench_evidence.v1", "bad evidence schema")
    _require(evidence.get("spec_refs") == SPEC_REFS, "bad evidence spec refs")
    _require(evidence.get("claim_scope") == LOCAL_CLAIM_SCOPE, "bad evidence claim scope")
    transcript = evidence.get("transcript")
    _require(isinstance(transcript, Mapping), "bad evidence transcript")
    _require(transcript.get("path") == artifact.get("board_transcript_path"), "bad evidence transcript path")
    _require(transcript.get("sha256") == artifact.get("transcript_sha256"), "bad evidence transcript hash")
    parity = evidence.get("parity")
    _require(isinstance(parity, Mapping), "bad evidence parity")
    _require(parity.get("cpu_reference_ok") is artifact.get("cpu_reference_ok"), "bad CPU parity flag")
    _require(parity.get("kv260_result_ok") is artifact.get("kv260_result_ok"), "bad KV260 parity flag")


def _command_probes(artifact: JsonDict) -> JsonDict:
    probes = artifact.get("command_probes")
    _require(isinstance(probes, Mapping), "command_probes must be a dict")
    return probes


def _xmutil_requires_root(probe: CommandProbe) -> bool:
    lowered = probe.combined_output.lower()
    return "root privileges" in lowered or "using 'sudo'" in lowered


def _observed_first_line(probe: CommandProbe) -> str:
    observed = probe.combined_output.strip() or f"returncode={probe.exit_code}"
    return observed.splitlines()[0][:300]


def _duration_floor(duration_s: float) -> float:
    return round(max(float(duration_s), 0.0001), 4)


def _duration_positive(duration_s: float) -> float:
    return round(max(float(duration_s), 0.0001), 12)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _finalize(artifact: JsonDict) -> BuiltPacket:
    transcript_text = _transcript_text(artifact)
    transcript_sha256 = _sha256_text(transcript_text)
    artifact["transcript_sha256"] = transcript_sha256
    artifact["structured_testbench_evidence"] = _structured_testbench_evidence(
        artifact,
        transcript_sha256,
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact, transcript_text=transcript_text)
    return BuiltPacket(artifact=artifact, transcript_text=transcript_text)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:  # pragma: no cover - live hardware entrypoint
    out_path = run_experiment(repo_root=REPO_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "result": str(out_path),
                "transcript": artifact["board_transcript_path"],
                "transcript_sha256": artifact["transcript_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint
    raise SystemExit(main())
