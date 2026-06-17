"""Exp 4356 KV260 SSH reachability and loaded-bitstream continuity check.

This module keeps the KV260 hardware path visible without turning a continuity
check into a hardware build. The precondition is board SSH reachability, not a
host storage device. A failed SSH precondition writes an honest blocked verdict;
a reachable board records the `xmutil listapps` and `/dev/uio*` transcripts
needed to see whether the Carnot Ising overlay is still present.

Spec refs: REQ-HW-4356, SCENARIO-HW-4356.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_4345_hardware_continuity as _previous


EXPERIMENT_ID = 4356
SCHEMA = "carnot.hardware_continuity_kv260_ssh_overlay.v1"
SPEC_REFS = ["REQ-HW-4356", "SCENARIO-HW-4356"]
OUTPUT_REL_PATH = Path("results") / "experiment_4356_hardware_continuity_kv260.json"
RANDOM_SEED = 4356
INFERENCE_SUBSTRATE = "hardware_smoke"

KV260_SSH_PRECONDITION = _previous.KV260_SSH_PRECONDITION
KV260_LISTAPPS_COMMAND = _previous.KV260_LISTAPPS_COMMAND
KV260_UIO_COMMAND = ("ssh", "kria", "ls /dev/uio*")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "kv260_reachable",
    "loaded_overlay",
    "kv260_terminal_state_reached",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (success_kv260_continuity_<state> or "
        "blocked_kv260_ssh_unreachable). A reachable continuity check and a clean "
        "unreachable skip are BOTH honest terminal states."
    ),
    "kv260_reachable": (
        "BARE bool: SSH reachability -- the only valid KV260 precondition "
        "(NEVER host SD-card presence)."
    ),
    "loaded_overlay": (
        "The xmutil listapps result -- records whether the carnot_ising bitstream "
        "is loaded (board continuity)."
    ),
    "kv260_terminal_state_reached": (
        "BARE bool: true iff the board-latency transcript + kv260_synthesis_succeeded "
        "terminal state is met (north-star \u00a73) -- after which the per-milestone "
        "KV260 mandate lifts."
    ),
    "preconditions_checked": (
        "Records the SSH-reachability check (NOT host SD card); pre-empts the "
        "wrong-mechanism SD-card confusion + the silent-missing-resource fabrication mode."
    ),
}

CommandProbe = _previous.CommandProbe
CommandRunner = _previous.CommandRunner
Clock = _previous.Clock

run_command = _previous.run_command
prepend_oss_cad_suite = _previous.prepend_oss_cad_suite
command_to_string = _previous.command_to_string
payload_checksum = _previous.payload_checksum


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4356 artifact from live or injected KV260 SSH probes."""
    root = Path(repo_root)
    started = clock()

    ssh_probe = command_runner(KV260_SSH_PRECONDITION, 10.0)
    kv260_reachable = ssh_probe.exit_code == 0
    preconditions = [_precondition_entry(ssh_probe, kv260_reachable)]
    transcript = [_transcript_entry("kv260_ssh_precondition", ssh_probe)]
    terminal_evidence = _terminal_state_evidence(root)

    if kv260_reachable:
        overlay_probe = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
        uio_probe = command_runner(KV260_UIO_COMMAND, 30.0)
        loaded_overlay = _loaded_overlay(overlay_probe)
        uio_presence = _uio_device_presence(uio_probe)
        transcript.append(_transcript_entry("kv260_xmutil_listapps", overlay_probe))
        transcript.append(_transcript_entry("kv260_uio_device_presence", uio_probe))
        terminal_reached = bool(terminal_evidence["terminal_state_reached"])
    else:
        loaded_overlay = _not_run_overlay()
        uio_presence = _not_run_uio()
        terminal_reached = False

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "honest_verdict": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_context": _source_context(root),
        "kv260_reachable": kv260_reachable,
        "loaded_overlay": loaded_overlay,
        "uio_device_presence": uio_presence,
        "kv260_terminal_state_reached": terminal_reached,
        "terminal_state_evidence": terminal_evidence,
        "preconditions_checked": preconditions,
        "board_state_transcript": transcript,
        "duration_s": _round_duration(clock() - started),
        "fabric_acceleration_claimed": False,
        "speedup_claim_made": False,
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return the terminal-prefixed Exp 4356 verdict."""
    if artifact.get("kv260_reachable") is False:
        return "blocked_kv260_ssh_unreachable"
    if artifact.get("kv260_terminal_state_reached") is True:
        return "success_kv260_continuity_terminal_reached"
    loaded = artifact.get("loaded_overlay", {})
    if isinstance(loaded, dict) and loaded.get("carnot_ising_loaded") is True:
        return "success_kv260_continuity_overlay_loaded_terminal_pending"
    if isinstance(loaded, dict) and str(loaded.get("status", "")).startswith("xmutil_listapps"):
        return "success_kv260_continuity_overlay_unknown_terminal_pending"
    return "success_kv260_continuity_overlay_absent_terminal_pending"


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4356 and write the requested KV260 continuity JSON artifact."""
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4356")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4356")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4356")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4356")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_principles(artifact)
    _validate_bare_bool(artifact, "kv260_reachable")
    _validate_bare_bool(artifact, "kv260_terminal_state_reached")
    _validate_preconditions(artifact)
    _validate_overlay(artifact)
    _validate_uio(artifact)
    _validate_transcript(artifact)
    _validate_source_context(artifact)
    _require(
        "mmcblk" not in json.dumps(artifact, sort_keys=True, default=str).lower(),
        "SD-card precondition marker is forbidden",
    )
    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be hardware_smoke",
    )
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale verdict")
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric acceleration claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _precondition_entry(probe: CommandProbe, available: bool) -> dict[str, Any]:
    return {
        "resource": "kv260_ssh",
        "mechanism": "ssh",
        "available": bool(available),
        "command": command_to_string(probe.command),
        "duration_s": _round_duration(probe.duration_s),
        "exit_code": probe.exit_code,
        "observed": _observed(probe),
    }


def _transcript_entry(stage: str, probe: CommandProbe) -> dict[str, Any]:
    return {
        "board": "kv260",
        "stage": stage,
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "duration_s": _round_duration(probe.duration_s),
        "output_excerpt": _observed(probe)[:1000],
    }


def _loaded_overlay(probe: CommandProbe) -> dict[str, Any]:
    observed = _observed(probe)
    names = _overlay_names(observed) if probe.exit_code == 0 else []
    loaded = any(name.startswith("carnot_ising") for name in names)
    status = "carnot_ising_loaded" if loaded else "carnot_ising_not_seen"
    if probe.exit_code != 0:
        status = f"xmutil_listapps_returncode_{probe.exit_code}"
    return {
        "status": status,
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "output_excerpt": observed[:2000],
        "overlay_names": names,
        "carnot_ising_loaded": loaded,
    }


def _uio_device_presence(probe: CommandProbe) -> dict[str, Any]:
    observed = _observed(probe)
    devices = _uio_devices(observed) if probe.exit_code == 0 else []
    present = bool(devices)
    status = "uio_devices_present" if present else "uio_devices_absent"
    if probe.exit_code != 0:
        status = f"uio_list_returncode_{probe.exit_code}"
    return {
        "status": status,
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "output_excerpt": observed[:2000],
        "devices": devices,
        "uio_devices_present": present,
    }


def _not_run_overlay() -> dict[str, Any]:
    return {
        "status": "not_run_kv260_ssh_unreachable",
        "command": None,
        "exit_code": None,
        "output_excerpt": "not_run_kv260_ssh_unreachable",
        "overlay_names": [],
        "carnot_ising_loaded": False,
    }


def _not_run_uio() -> dict[str, Any]:
    return {
        "status": "not_run_kv260_ssh_unreachable",
        "command": None,
        "exit_code": None,
        "output_excerpt": "not_run_kv260_ssh_unreachable",
        "devices": [],
        "uio_devices_present": False,
    }


def _overlay_names(text: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(r"\bcarnot_ising[0-9A-Za-z_]*\b", text):
        name = match.group(0)
        if name not in names:
            names.append(name)
    return names


def _uio_devices(text: str) -> list[str]:
    devices: list[str] = []
    for match in re.finditer(r"/dev/uio\d+\b", text):
        device = match.group(0)
        if device not in devices:
            devices.append(device)
    return devices


def _terminal_state_evidence(repo_root: Path) -> dict[str, Any]:
    rel_path = Path("results") / "experiment_2742_kv260_latency_transcript_terminal.json"
    payload = _read_json(repo_root / rel_path)
    synthesis = payload.get("kv260_synthesis_succeeded") is True
    latency = (
        payload.get("kv260_terminal") is True and int(payload.get("n_cycles_measured") or 0) > 0
    )
    return {
        "source": str(rel_path),
        "source_read": bool(payload),
        "kv260_synthesis_succeeded": synthesis,
        "board_latency_transcript_present": latency,
        "terminal_state_reached": bool(synthesis and latency),
        "latency_mean_us": payload.get("kv260_latency_mean_us"),
        "n_cycles_measured": payload.get("n_cycles_measured"),
    }


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4345_hardware_continuity.json"
    prior_payload = _read_json(prior_path)
    return {
        "previous_experiment": 4345,
        "most_recent_hardware_continuity_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_kv260_reachable": prior_payload.get("kv260_reachable"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact.
        return {}


def _observed(probe: CommandProbe) -> str:
    text = probe.combined_output.strip()
    return text if text else f"returncode={probe.exit_code}"


def _validate_principles(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(isinstance(principles, dict), "field_principles must be dict")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(principles.get(field) == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
        _require(
            not (
                isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}
            ),
            f"{field} must remain a bare value",
        )


def _validate_bare_bool(artifact: dict[str, Any], field: str) -> None:
    _require(isinstance(artifact.get(field), bool), f"{field} must be a bare bool")


def _validate_preconditions(artifact: dict[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list), "preconditions_checked must be a list")
    _require(len(preconditions) == 1, "exactly one KV260 SSH precondition is required")
    entry = preconditions[0]
    _require(isinstance(entry, dict), "precondition entry must be a dict")
    _require(entry.get("resource") == "kv260_ssh", "precondition resource must be kv260_ssh")
    _require(entry.get("mechanism") == "ssh", "precondition mechanism must be ssh")
    _require(entry.get("command") == command_to_string(KV260_SSH_PRECONDITION), "bad SSH command")
    _require(isinstance(entry.get("available"), bool), "precondition available must be bool")


def _validate_overlay(artifact: dict[str, Any]) -> None:
    overlay = artifact.get("loaded_overlay")
    _require(isinstance(overlay, dict), "loaded_overlay must be a dict")
    if artifact["kv260_reachable"]:
        _require(overlay.get("command") == command_to_string(KV260_LISTAPPS_COMMAND), "")
        _require(isinstance(overlay.get("exit_code"), int), "overlay exit_code must be int")
    else:
        _require(overlay == _not_run_overlay(), "blocked artifact must not run overlay probe")
    _require(isinstance(overlay.get("carnot_ising_loaded"), bool), "overlay loaded must be bool")
    _require(isinstance(overlay.get("overlay_names"), list), "overlay_names must be a list")


def _validate_uio(artifact: dict[str, Any]) -> None:
    uio = artifact.get("uio_device_presence")
    _require(isinstance(uio, dict), "uio_device_presence must be a dict")
    if artifact["kv260_reachable"]:
        _require(uio.get("command") == command_to_string(KV260_UIO_COMMAND), "bad UIO command")
        _require(isinstance(uio.get("exit_code"), int), "UIO exit_code must be int")
    else:
        _require(uio == _not_run_uio(), "blocked artifact must not run UIO probe")
    _require(isinstance(uio.get("uio_devices_present"), bool), "UIO presence must be bool")
    _require(isinstance(uio.get("devices"), list), "UIO devices must be a list")


def _validate_transcript(artifact: dict[str, Any]) -> None:
    transcript = artifact.get("board_state_transcript")
    _require(isinstance(transcript, list), "board_state_transcript must be a list")
    expected = (
        ["kv260_ssh_precondition", "kv260_xmutil_listapps", "kv260_uio_device_presence"]
        if artifact["kv260_reachable"]
        else ["kv260_ssh_precondition"]
    )
    stages = [entry.get("stage") for entry in transcript if isinstance(entry, dict)]
    _require(stages == expected, "board_state_transcript has wrong stages")


def _validate_source_context(artifact: dict[str, Any]) -> None:
    source = artifact.get("source_context")
    _require(isinstance(source, dict), "source_context must be present")
    _require(source.get("previous_experiment") == 4345, "source_context must read Exp 4345")
    _require(
        str(source.get("most_recent_hardware_continuity_artifact", "")).endswith(
            "experiment_4345_hardware_continuity.json"
        ),
        "source_context must point at Exp 4345",
    )


def _round_duration(value: float) -> float:
    return round(max(float(value), 0.000001), 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"loaded_overlay: {artifact['loaded_overlay']}")
    print(f"kv260_terminal_state_reached: {artifact['kv260_terminal_state_reached']}")


if __name__ == "__main__":  # pragma: no cover
    main()
