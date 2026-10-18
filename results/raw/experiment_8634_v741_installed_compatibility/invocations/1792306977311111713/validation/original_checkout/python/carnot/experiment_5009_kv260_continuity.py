"""Exp 5009 KV260 SSH-only overlay/UIO and energy continuity artifact.

Spec refs: REQ-HW-5009, SCENARIO-HW-5009.

This check keeps the KV260 hardware-continuity slot honest. The only
precondition is SSH reachability. If the board is reachable, the run records
overlay and UIO state over SSH, then runs a tiny on-board Ising energy smoke
only when a Carnot Ising overlay is actually loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
import time
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot import experiment_4910_kv260_continuity as base


EXPERIMENT_ID = 5009
SCHEMA = "carnot.kv260_ssh_continuity.v17"
SPEC_REFS = ["REQ-HW-5009", "SCENARIO-HW-5009"]
OUTPUT_REL_PATH = Path("results") / "experiment_5009_kv260_continuity.json"
RANDOM_SEED = 5009
INFERENCE_SUBSTRATE = "hardware_smoke"

CommandProbe = base.CommandProbe
CommandRunner = base.CommandRunner
Clock = base.Clock

KV260_SSH_COMMAND = base.KV260_SSH_COMMAND
KV260_LISTAPPS_COMMAND = base.KV260_LISTAPPS_COMMAND
KV260_LISTAPPS_SUDO_COMMAND = base.KV260_LISTAPPS_SUDO_COMMAND
KV260_UIO_COMMAND = base.KV260_UIO_COMMAND
VALID_OVERLAYS = base.VALID_OVERLAYS

ENERGY_SMOKE_PROBLEM = "tiny_quadratic_ising_constraint_energy"
ENERGY_SMOKE_EXPECTED = -7
ENERGY_SMOKE_CODE = (
    "import json,time;"
    "started=time.perf_counter();"
    "spins=[1,-1,1,-1];"
    "edges=[(0,1,2),(1,2,-1),(2,3,3),(0,3,1)];"
    "bias=[1,0,-1,2];"
    "energy=sum(w*spins[i]*spins[j] for i,j,w in edges)+"
    "sum(b*s for b,s in zip(bias,spins));"
    "print(json.dumps({"
    "'problem':'tiny_quadratic_ising_constraint_energy',"
    "'energy':energy,"
    "'expected_energy':-7,"
    "'duration_s':time.perf_counter()-started"
    "}))"
)
KV260_ENERGY_COMMAND = ("ssh", "kria", f"python3 -c {shlex.quote(ENERGY_SMOKE_CODE)}")

BLOCKED_SSH_VERDICT = "blocked_kv260_ssh_unreachable"

REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "kv260_ssh_reachable",
    "overlay_state",
    "on_board_energy_duration_s",
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
    "kv260_ssh_exit_code",
    "loaded_overlay",
    "uio_devices",
    "xmutil_requires_sudo",
    "energy_smoke",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success_kv260_reachable_overlay_<state> or blocked_kv260_ssh_unreachable."
    ),
    "kv260_ssh_reachable": (
        "the SSH reachability exit code (the SSH-only precondition; never host SD-card)."
    ),
    "overlay_state": (
        "xmutil listapps output -- whether the carnot_ising overlay is loaded "
        "(the continuity signal)."
    ),
    "on_board_energy_duration_s": (
        "real wall-clock of the on-board energy smoke if the overlay is loaded; "
        "null if reachability-only."
    ),
    "inference_substrate": "hardware_smoke (SSH-attached board test).",
    "preconditions_checked": (
        "records the SSH reachability check (never the host SD-card device path); "
        "a non-zero SSH emits blocked_kv260_ssh_unreachable."
    ),
}

command_to_string = base.command_to_string
payload_checksum = base.payload_checksum
run_command = base.run_command


def parse_uio_devices(text: str) -> list[str]:
    """REQ-HW-5009: parse unique UIO device paths from an SSH transcript."""
    return base.parse_uio_devices(text)


def loaded_overlay_from_xmutil(text: str) -> str | None:
    """REQ-HW-5009: return the loaded Carnot overlay from `xmutil listapps`."""
    return base.loaded_overlay_from_xmutil(text)


def parse_energy_smoke_stdout(stdout: str) -> dict[str, Any] | None:
    """SCENARIO-HW-5009: extract the final JSON energy-smoke payload."""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    return None


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, object]:
    """Build the Exp 5009 artifact from live or injected SSH board probes."""
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

    loaded_overlay = loaded_overlay_from_xmutil(listapps_text)
    uio_devices = parse_uio_devices(uio_text)
    overlay_state = _overlay_state(
        listapps_probe=listapps_probe,
        listapps_text=listapps_text,
        loaded_overlay=loaded_overlay,
        uio_probe=uio_probe,
        uio_devices=uio_devices,
        requires_sudo=requires_sudo,
    )

    energy_smoke: dict[str, object] | None = None
    energy_duration_s: float | None = None
    overlay_token = "not_loaded"
    if loaded_overlay is not None:
        energy_probe = command_runner(KV260_ENERGY_COMMAND, 30.0)
        command_probes["kv260_energy_smoke"] = energy_probe.as_dict()
        energy_duration_s = _duration_round(energy_probe.duration_s)
        energy_smoke = _energy_smoke_from_probe(energy_probe)
        overlay_token = "loaded_energy_ok" if energy_smoke["success"] else "loaded_energy_failed"

    artifact.update(
        {
            "honest_verdict": f"success_kv260_reachable_overlay_{overlay_token}",
            "kv260_ssh_reachable": True,
            "kv260_ssh_exit_code": ssh_probe.exit_code,
            "overlay_state": overlay_state,
            "loaded_overlay": loaded_overlay,
            "uio_devices": uio_devices,
            "xmutil_requires_sudo": requires_sudo,
            "on_board_energy_duration_s": energy_duration_s,
            "energy_smoke": energy_smoke,
            "duration_s": _duration_floor(clock() - started),
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(repo_root: str | Path, artifact: dict[str, object]) -> Path:
    """Write a validated Exp 5009 artifact under the requested repository root."""
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
    """Run Exp 5009 and write `results/experiment_5009_kv260_continuity.json`."""
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate the Exp 5009 SSH-only overlay/UIO energy schema."""
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
    _validate_overlay_state(artifact)
    _validate_energy_contract(artifact)
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
        "kv260_ssh_exit_code": ssh_probe.exit_code,
        "overlay_state": _empty_overlay_state(),
        "loaded_overlay": None,
        "uio_devices": [],
        "on_board_energy_duration_s": None,
        "energy_smoke": None,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": [_precondition_entry(ssh_probe)],
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": _duration_floor(duration_s),
        "command_probes": {
            "kv260_ssh": ssh_probe.as_dict(),
            "kv260_xmutil_listapps": None,
            "kv260_xmutil_listapps_sudo": None,
            "kv260_uio_devices": None,
            "kv260_energy_smoke": None,
        },
        "xmutil_requires_sudo": False,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }


def _empty_overlay_state() -> dict[str, object]:
    return {
        "xmutil_listapps_output": None,
        "xmutil_listapps_exit_code": None,
        "xmutil_requires_sudo": False,
        "loaded_overlay": None,
        "carnot_ising_overlay_loaded": False,
        "uio_devices": [],
        "uio_output": None,
        "uio_exit_code": None,
    }


def _overlay_state(
    *,
    listapps_probe: CommandProbe,
    listapps_text: str,
    loaded_overlay: str | None,
    uio_probe: CommandProbe,
    uio_devices: list[str],
    requires_sudo: bool,
) -> dict[str, object]:
    return {
        "xmutil_listapps_output": listapps_text,
        "xmutil_listapps_exit_code": listapps_probe.exit_code,
        "xmutil_requires_sudo": requires_sudo,
        "loaded_overlay": loaded_overlay,
        "carnot_ising_overlay_loaded": loaded_overlay is not None,
        "uio_devices": list(uio_devices),
        "uio_output": uio_probe.combined_output,
        "uio_exit_code": uio_probe.exit_code,
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


def _energy_smoke_from_probe(probe: CommandProbe) -> dict[str, object]:
    payload = parse_energy_smoke_stdout(probe.stdout) if probe.exit_code == 0 else None
    energy = payload.get("energy") if payload is not None else None
    expected = payload.get("expected_energy") if payload is not None else ENERGY_SMOKE_EXPECTED
    return {
        "problem": payload.get("problem") if payload is not None else ENERGY_SMOKE_PROBLEM,
        "energy": energy,
        "expected_energy": expected,
        "success": probe.exit_code == 0 and energy == ENERGY_SMOKE_EXPECTED,
        "board_reported_duration_s": payload.get("duration_s") if payload is not None else None,
        "command_exit_code": probe.exit_code,
    }


def _duration_floor(duration_s: float) -> float:
    return round(max(float(duration_s), 0.0001), 4)


def _duration_round(duration_s: float) -> float:
    return round(max(float(duration_s), 0.000001), 6)


def _xmutil_requires_root(probe: CommandProbe) -> bool:
    lowered = probe.combined_output.lower()
    return "root privileges" in lowered or "using 'sudo'" in lowered


def _observed_first_line(probe: CommandProbe) -> str:
    observed = probe.combined_output.strip() or f"returncode={probe.exit_code}"
    return observed.splitlines()[0][:300]


def _validate_precondition(artifact: dict[str, object]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(
        isinstance(preconditions, list) and len(preconditions) == 1, "bad preconditions_checked"
    )
    entry = preconditions[0]
    _require(isinstance(entry, dict), "bad precondition entry")
    _require(entry.get("resource") == "kv260_ssh", "bad KV260 precondition resource")
    _require(entry.get("command") == command_to_string(KV260_SSH_COMMAND), "bad KV260 SSH command")
    _require(entry.get("discipline") == "ssh_only_no_host_sd_card", "bad KV260 discipline")
    _require(entry.get("exit_code") == artifact.get("kv260_ssh_exit_code"), "bad SSH exit code")
    _require(entry.get("available") is artifact.get("kv260_ssh_reachable"), "precondition mismatch")


def _validate_overlay_state(artifact: dict[str, object]) -> None:
    probes = _command_probes(artifact)
    overlay_state = artifact.get("overlay_state")
    _require(isinstance(overlay_state, dict), "overlay_state must be a dict")
    if artifact.get("kv260_ssh_reachable") is False:
        _require(artifact.get("honest_verdict") == BLOCKED_SSH_VERDICT, "bad blocked verdict")
        _require(overlay_state == _empty_overlay_state(), "blocked SSH cannot report overlay state")
        _require(probes.get("kv260_xmutil_listapps") is None, "blocked SSH cannot run xmutil")
        _require(probes.get("kv260_uio_devices") is None, "blocked SSH cannot run UIO listing")
        return

    _require(artifact.get("kv260_ssh_reachable") is True, "kv260_ssh_reachable must be bool")
    _require(
        str(artifact.get("honest_verdict", "")).startswith("success_kv260_reachable_overlay_"),
        "bad success verdict",
    )
    _require(probes.get("kv260_xmutil_listapps") is not None, "success requires xmutil probe")
    _require(probes.get("kv260_uio_devices") is not None, "success requires UIO probe")
    if artifact.get("xmutil_requires_sudo"):
        _require(probes.get("kv260_xmutil_listapps_sudo") is not None, "sudo fallback missing")
    _require(
        overlay_state.get("loaded_overlay") == artifact.get("loaded_overlay"), "overlay mismatch"
    )
    _require(overlay_state.get("uio_devices") == artifact.get("uio_devices"), "UIO mismatch")


def _validate_energy_contract(artifact: dict[str, object]) -> None:
    probes = _command_probes(artifact)
    loaded_overlay = artifact.get("loaded_overlay")
    energy_duration = artifact.get("on_board_energy_duration_s")
    energy_smoke = artifact.get("energy_smoke")
    if artifact.get("kv260_ssh_reachable") is False or loaded_overlay is None:
        _require(energy_duration is None, "energy duration must be null without a loaded overlay")
        _require(energy_smoke is None, "energy smoke must be null without a loaded overlay")
        _require(probes.get("kv260_energy_smoke") is None, "energy command must be skipped")
        return

    _require(
        isinstance(energy_duration, (int, float)) and energy_duration > 0.0, "bad energy duration"
    )
    _require(isinstance(energy_smoke, dict), "energy_smoke must be a dict")
    _require(probes.get("kv260_energy_smoke") is not None, "energy command missing")
    if energy_smoke.get("success") is True:
        _require(energy_smoke.get("energy") == ENERGY_SMOKE_EXPECTED, "energy mismatch")


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
    print(f"kv260_ssh_reachable: {artifact['kv260_ssh_reachable']}")
    print(f"loaded_overlay: {artifact['overlay_state']['loaded_overlay']}")
    print(f"on_board_energy_duration_s: {artifact['on_board_energy_duration_s']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
