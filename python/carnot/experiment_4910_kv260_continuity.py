"""Exp 4910 KV260 SSH-only overlay and UIO continuity artifact.

Spec refs: REQ-HW-4910, SCENARIO-HW-4910.

This check keeps the graduated KV260 in the hardware-continuity rotation. The
only precondition is SSH reachability; when SSH is reachable, the artifact
records the loaded overlay transcript and the visible UIO device list over SSH.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot import experiment_4899_kv260_continuity as base


EXPERIMENT_ID = 4910
SCHEMA = "carnot.kv260_ssh_continuity.v15"
SPEC_REFS = ["REQ-HW-4910", "SCENARIO-HW-4910"]
OUTPUT_REL_PATH = Path("results") / "experiment_4910_kv260_continuity.json"
RANDOM_SEED = 4910
INFERENCE_SUBSTRATE = "hardware_smoke"

CommandProbe = base.CommandProbe
CommandRunner = base.CommandRunner
Clock = base.Clock

KV260_SSH_COMMAND = base.KV260_SSH_COMMAND
KV260_LISTAPPS_COMMAND = base.KV260_LISTAPPS_COMMAND
KV260_LISTAPPS_SUDO_COMMAND = base.KV260_LISTAPPS_SUDO_COMMAND
KV260_UIO_COMMAND = ("ssh", "kria", "ls /dev/uio*")

VALID_OVERLAYS = (
    "carnot_ising_v4_n64",
    "carnot_ising_v2_n64",
    "carnot_ising_v4",
    "carnot_ising",
)

SUCCESS_VERDICT = "success_kv260_continuity_ok"
BLOCKED_SSH_VERDICT = "blocked_kv260_ssh_unreachable"

REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "kv260_ssh_reachable",
    "loaded_overlay",
    "uio_devices",
    "inference_substrate",
    "preconditions_checked",
)
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_PRINCIPLE_FIELDS,
    "schema",
    "experiment",
    "spec_refs",
    "field_principles",
    "duration_s",
    "command_probes",
    "xmutil_requires_sudo",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success_kv260_continuity_ok or "
        "blocked_kv260_ssh_unreachable."
    ),
    "kv260_ssh_reachable": (
        "true iff `ssh kria true` exit 0 -- the SSH-only precondition "
        "(NEVER host SD-card)."
    ),
    "loaded_overlay": (
        "the xmutil listapps overlay (the flashed carnot_ising bitstream state)."
    ),
    "uio_devices": "the /dev/uio* list (the energy-evaluator interface presence).",
    "inference_substrate": "hardware_smoke (SSH-attached board; per-board floor).",
    "preconditions_checked": (
        "records the SSH reachability check; unreachable emits blocked_, never "
        "a fabricated record."
    ),
}

command_to_string = base.command_to_string
payload_checksum = base.payload_checksum
run_command = base.run_command


def parse_uio_devices(text: str) -> list[str]:
    """Return unique `/dev/uioN` paths from an SSH transcript."""
    devices: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"/dev/uio\d+", text):
        device = match.group(0)
        if device not in seen:
            devices.append(device)
            seen.add(device)
    return devices


def loaded_overlay_from_xmutil(text: str) -> str | None:
    """Return the active Carnot overlay from an `xmutil listapps` transcript."""
    for line in text.splitlines():
        lowered = line.lower()
        for overlay in VALID_OVERLAYS:
            if overlay not in line:
                continue
            if "running" in lowered or "loaded" in lowered or "slot_handle 0" in lowered:
                return overlay
            if "->" in line and not line.rstrip().endswith("-1"):
                return overlay
    return None


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, object]:
    """Build the Exp 4910 artifact from live or injected SSH board probes."""
    started = clock()
    ssh_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    artifact = _base_artifact(ssh_probe=ssh_probe, duration_s=clock() - started)

    if ssh_probe.exit_code != 0:
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
        validate_artifact(artifact)
        return artifact

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
    uio_text = uio_probe.combined_output if uio_probe.exit_code == 0 else ""

    artifact.update(
        {
            "honest_verdict": SUCCESS_VERDICT,
            "kv260_ssh_reachable": True,
            "loaded_overlay": loaded_overlay_from_xmutil(listapps_text),
            "uio_devices": parse_uio_devices(uio_text),
            "xmutil_requires_sudo": requires_sudo,
            "duration_s": _duration_floor(clock() - started),
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(repo_root: str | Path, artifact: dict[str, object]) -> Path:
    """Write a validated Exp 4910 artifact under the requested repository root."""
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4910 and write `results/experiment_4910_kv260_continuity.json`."""
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate the Exp 4910 SSH-only overlay and UIO continuity schema."""
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(artifact.get("schema") == SCHEMA, "schema mismatch")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment mismatch")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle must be false")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    _require(float(artifact.get("duration_s", 0.0)) >= 0.0001, "duration_s below floor")
    for field in REQUIRED_PRINCIPLE_FIELDS:
        _require(field in artifact, f"missing principle field: {field}")
        _require(
            not (isinstance(artifact[field], dict) and "principle" in artifact[field]),
            f"{field} must remain a bare value",
        )
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require(
        "mmcblk" not in encoded and "/dev/disk" not in encoded,
        "forbidden host storage marker",
    )
    _validate_precondition(artifact)
    _validate_verdict(artifact)
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "bad checksum",
    )


def _base_artifact(ssh_probe: CommandProbe, duration_s: float) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": BLOCKED_SSH_VERDICT,
        "kv260_ssh_reachable": ssh_probe.exit_code == 0,
        "loaded_overlay": None,
        "uio_devices": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": [_precondition_entry(ssh_probe)],
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": _duration_floor(duration_s),
        "command_probes": {
            "kv260_ssh": ssh_probe.as_dict(),
            "kv260_xmutil_listapps": None,
            "kv260_xmutil_listapps_sudo": None,
            "kv260_uio_devices": None,
        },
        "xmutil_requires_sudo": False,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }


def _precondition_entry(ssh_probe: CommandProbe) -> dict[str, object]:
    return {
        "resource": "kv260_ssh",
        "available": ssh_probe.exit_code == 0,
        "command": command_to_string(KV260_SSH_COMMAND),
        "exit_code": ssh_probe.exit_code,
        "duration_s": ssh_probe.duration_s,
        "observed": _observed_first_line(ssh_probe),
        "discipline": "ssh_only_no_host_sd_card",
    }


def _duration_floor(duration_s: float) -> float:
    return round(max(float(duration_s), 0.0001), 4)


def _xmutil_requires_root(probe: CommandProbe) -> bool:
    lowered = probe.combined_output.lower()
    return "root privileges" in lowered or "using 'sudo'" in lowered


def _observed_first_line(probe: CommandProbe) -> str:
    observed = probe.combined_output.strip() or f"returncode={probe.exit_code}"
    return observed.splitlines()[0][:300]


def _validate_precondition(artifact: dict[str, object]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list) and len(preconditions) == 1, "bad preconditions_checked")
    entry = preconditions[0]
    _require(isinstance(entry, dict), "bad precondition entry")
    _require(entry.get("resource") == "kv260_ssh", "bad KV260 precondition resource")
    _require(entry.get("command") == command_to_string(KV260_SSH_COMMAND), "bad KV260 SSH command")
    _require(entry.get("discipline") == "ssh_only_no_host_sd_card", "bad KV260 discipline")
    _require(entry.get("available") is artifact.get("kv260_ssh_reachable"), "precondition mismatch")


def _validate_verdict(artifact: dict[str, object]) -> None:
    probes = _command_probes(artifact)
    if artifact.get("kv260_ssh_reachable") is False:
        _require(artifact.get("honest_verdict") == BLOCKED_SSH_VERDICT, "bad blocked verdict")
        _require(artifact.get("loaded_overlay") is None, "blocked SSH cannot report overlay")
        _require(artifact.get("uio_devices") == [], "blocked SSH cannot report UIO devices")
        _require(probes.get("kv260_xmutil_listapps") is None, "blocked SSH cannot run xmutil")
        _require(probes.get("kv260_uio_devices") is None, "blocked SSH cannot run UIO listing")
        return

    _require(artifact.get("kv260_ssh_reachable") is True, "kv260_ssh_reachable must be bool")
    _require(artifact.get("honest_verdict") == SUCCESS_VERDICT, "bad success verdict")
    loaded_overlay = artifact.get("loaded_overlay")
    _require(loaded_overlay is None or loaded_overlay in VALID_OVERLAYS, "invalid loaded overlay")
    uio_devices = artifact.get("uio_devices")
    _require(
        isinstance(uio_devices, list) and all(isinstance(item, str) for item in uio_devices),
        "uio_devices must be a list of strings",
    )
    _require(probes.get("kv260_xmutil_listapps") is not None, "success requires xmutil probe")
    _require(probes.get("kv260_uio_devices") is not None, "success requires UIO probe")
    if artifact.get("xmutil_requires_sudo"):
        _require(probes.get("kv260_xmutil_listapps_sudo") is not None, "sudo fallback missing")


def _command_probes(artifact: dict[str, object]) -> dict[str, Any]:
    probes = artifact.get("command_probes")
    _require(isinstance(probes, dict), "command_probes must be a dict")
    return probes


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"loaded_overlay: {artifact['loaded_overlay']}")
    print(f"uio_devices: {artifact['uio_devices']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
