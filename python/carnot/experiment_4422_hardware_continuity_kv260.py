"""Exp 4422 KV260 SSH reachability and loaded-bitstream continuity check.

This is an opportunistic continuity probe, not a board bring-up task. It checks
board SSH reachability first; unreachable SSH writes a clean blocked artifact
and stops. Reachable SSH records the loaded Carnot overlay string and whether
any UIO device is present.

Spec refs: REQ-HW-4422, SCENARIO-HW-4422.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_4356_hardware_continuity_kv260 as _base
from carnot import experiment_4411_hardware_continuity_kv260 as _previous


EXPERIMENT_ID = 4422
SCHEMA = "carnot.hardware_continuity_kv260_ssh_overlay.v2"
SPEC_REFS = ["REQ-HW-4422", "SCENARIO-HW-4422"]
OUTPUT_REL_PATH = Path("results") / "experiment_4422_hardware_continuity_kv260.json"
RANDOM_SEED = 4422
INFERENCE_SUBSTRATE = _previous.INFERENCE_SUBSTRATE

KV260_SSH_PRECONDITION = _previous.KV260_SSH_PRECONDITION
KV260_LISTAPPS_COMMAND = _previous.KV260_LISTAPPS_COMMAND
KV260_UIO_COMMAND = _previous.KV260_UIO_COMMAND

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "kv260_reachable",
    "loaded_overlay",
    "uio_present",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (success_kv260_reachable_overlay_<name> or "
        "blocked_kv260_ssh_unreachable -- a clean documented skip is honest, "
        "not a fabrication)."
    ),
    "kv260_reachable": (
        "BARE bool: SSH-reachability (the ONLY valid KV260 precondition; never "
        "host SD-card presence)."
    ),
    "loaded_overlay": (
        "str|null: the xmutil listapps loaded overlay if reachable "
        "(the sovereignty-story continuity record)."
    ),
    "uio_present": (
        "bool|null: the carnot_ising bitstream UIO device present if reachable "
        "(the liveness check)."
    ),
    "preconditions_checked": (
        "Records the SSH-reachability check (NOT a host SD-card check); pre-empts "
        "the retired-mechanism + fabrication modes."
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
    """Build the Exp 4422 artifact from live or injected KV260 SSH probes."""
    root = Path(repo_root)
    started = clock()

    ssh_probe = command_runner(KV260_SSH_PRECONDITION, 10.0)
    kv260_reachable = ssh_probe.exit_code == 0
    preconditions = [_precondition_entry(ssh_probe, kv260_reachable)]
    transcript = [_transcript_entry("kv260_ssh_precondition", ssh_probe)]

    if kv260_reachable:
        overlay_probe_raw = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
        uio_probe_raw = command_runner(KV260_UIO_COMMAND, 30.0)
        overlay_probe = _overlay_probe(overlay_probe_raw)
        uio_probe = _uio_probe(uio_probe_raw)
        transcript.append(_transcript_entry("kv260_xmutil_listapps", overlay_probe_raw))
        transcript.append(_transcript_entry("kv260_uio_device_presence", uio_probe_raw))
        loaded_overlay = overlay_probe["loaded_overlay"]
        uio_present = uio_probe["uio_present"]
    else:
        overlay_probe = _not_run_overlay_probe()
        uio_probe = _not_run_uio_probe()
        loaded_overlay = None
        uio_present = None

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
        "uio_present": uio_present,
        "overlay_probe": overlay_probe,
        "uio_probe": uio_probe,
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
    """Return the terminal-prefixed Exp 4422 verdict."""
    if artifact.get("kv260_reachable") is False:
        return "blocked_kv260_ssh_unreachable"
    overlay = artifact.get("loaded_overlay") or "unknown"
    return f"success_kv260_reachable_overlay_{_verdict_token(str(overlay))}"


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4422 and write the requested KV260 continuity JSON artifact."""
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4422")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4422")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4422")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4422")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_bare_bool(artifact, "kv260_reachable")
    _validate_principles(artifact)
    _validate_overlay_contract(artifact)
    _validate_uio_contract(artifact)
    _validate_preconditions(artifact)
    _validate_probe_details(artifact)
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


def _overlay_probe(probe: CommandProbe) -> dict[str, Any]:
    observed = _observed(probe)
    names = _overlay_names(observed) if probe.exit_code == 0 else []
    loaded_overlay = names[0] if names else None
    return {
        "status": "overlay_parsed" if loaded_overlay else _returncode_status("overlay", probe),
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "output_excerpt": observed[:2000],
        "overlay_names": names,
        "loaded_overlay": loaded_overlay,
    }


def _uio_probe(probe: CommandProbe) -> dict[str, Any]:
    observed = _observed(probe)
    devices = _uio_devices(observed) if probe.exit_code == 0 else []
    return {
        "status": "uio_devices_present" if devices else _returncode_status("uio", probe),
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "output_excerpt": observed[:2000],
        "devices": devices,
        "uio_present": bool(devices) if probe.exit_code == 0 else False,
    }


def _not_run_overlay_probe() -> dict[str, Any]:
    return {
        "status": "not_run_kv260_ssh_unreachable",
        "command": None,
        "exit_code": None,
        "output_excerpt": "not_run_kv260_ssh_unreachable",
        "overlay_names": [],
        "loaded_overlay": None,
    }


def _not_run_uio_probe() -> dict[str, Any]:
    return {
        "status": "not_run_kv260_ssh_unreachable",
        "command": None,
        "exit_code": None,
        "output_excerpt": "not_run_kv260_ssh_unreachable",
        "devices": [],
        "uio_present": None,
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


def _returncode_status(kind: str, probe: CommandProbe) -> str:
    if probe.exit_code == 0:
        return f"{kind}_not_seen"
    return f"{kind}_returncode_{probe.exit_code}"


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4411_hardware_continuity_kv260.json"
    prior_payload = _base._read_json(prior_path)
    return {
        "previous_experiment": 4411,
        "most_recent_hardware_continuity_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_kv260_reachable": prior_payload.get("kv260_reachable"),
    }


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


def _validate_overlay_contract(artifact: dict[str, Any]) -> None:
    overlay = artifact.get("loaded_overlay")
    if artifact["kv260_reachable"]:
        _require(overlay is None or isinstance(overlay, str), "loaded_overlay must be str|null")
    else:
        _require(overlay is None, "blocked artifact must set loaded_overlay null")


def _validate_uio_contract(artifact: dict[str, Any]) -> None:
    uio_present = artifact.get("uio_present")
    if artifact["kv260_reachable"]:
        _require(isinstance(uio_present, bool), "uio_present must be bool|null")
    else:
        _require(uio_present is None, "blocked artifact must set uio_present null")


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


def _validate_probe_details(artifact: dict[str, Any]) -> None:
    overlay = artifact.get("overlay_probe")
    uio = artifact.get("uio_probe")
    _require(isinstance(overlay, dict), "overlay_probe must be a dict")
    _require(isinstance(uio, dict), "uio_probe must be a dict")
    if artifact["kv260_reachable"]:
        _require(overlay.get("command") == command_to_string(KV260_LISTAPPS_COMMAND), "bad overlay command")
        _require(uio.get("command") == command_to_string(KV260_UIO_COMMAND), "bad UIO command")
        _require(overlay.get("loaded_overlay") == artifact.get("loaded_overlay"), "overlay mismatch")
        _require(uio.get("uio_present") == artifact.get("uio_present"), "UIO mismatch")
    else:
        _require(overlay == _not_run_overlay_probe(), "blocked artifact must not run overlay probe")
        _require(uio == _not_run_uio_probe(), "blocked artifact must not run UIO probe")


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
    _require(source.get("previous_experiment") == 4411, "source_context must read Exp 4411")
    _require(
        str(source.get("most_recent_hardware_continuity_artifact", "")).endswith(
            "experiment_4411_hardware_continuity_kv260.json"
        ),
        "source_context must point at Exp 4411",
    )


def _verdict_token(text: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_").lower()
    return token or "unknown"


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
    print(f"uio_present: {artifact['uio_present']}")


if __name__ == "__main__":  # pragma: no cover
    main()
