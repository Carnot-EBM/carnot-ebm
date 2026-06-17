"""Exp 4345 opportunistic KV260 SSH and board reachability continuity.

This module records KV260 through the current SSH-only mechanism. If KV260 is
not reachable over SSH, the artifact stops at an honest blocked verdict instead
of inventing GateMate or PolarFire state. When KV260 is reachable, the module
records the KV260 `xmutil listapps` state, then opportunistically probes
GateMate DirtyJTAG and PolarFire SSH reachability.

Spec refs: REQ-HW-4345, SCENARIO-HW-4345.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4334_hardware_continuity as _baseline


EXPERIMENT_ID = 4345
SCHEMA = "carnot.hardware_continuity_opportunistic_reachability.v1"
SPEC_REFS = ["REQ-HW-4345", "SCENARIO-HW-4345"]
OUTPUT_REL_PATH = Path("results") / "experiment_4345_hardware_continuity.json"
RANDOM_SEED = 4345
INFERENCE_SUBSTRATE = "hardware_smoke"
BOARD_NAMES = ("kv260", "gatemate", "polarfire")

KV260_SSH_PRECONDITION = _baseline.KV260_SSH_PRECONDITION
KV260_LISTAPPS_COMMAND = _baseline.KV260_LISTAPPS_COMMAND
GATEMATE_DETECT_COMMAND = _baseline.GATEMATE_DETECT_COMMAND
POLARFIRE_SSH_PRECONDITION = _baseline.POLARFIRE_SSH_PRECONDITION

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "kv260_reachable",
    "boards_probed",
    "preconditions_checked",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records the KV260 SSH continuity + opportunistic board "
        "reachability (or blocked_kv260_ssh_unreachable -- an honest non-fabrication)."
    ),
    "kv260_reachable": (
        "BARE bool: KV260 SSH-reachable (the sovereignty story's board-state; the SSH "
        "precondition, NEVER host SD-card)."
    ),
    "boards_probed": (
        "list: each board {name, reachable, state} -- keeps the boards visible in the "
        "milestone retro per the continuity discipline."
    ),
    "preconditions_checked": (
        "Records the SSH reachability checks (NEVER a host-SD-card device-node check); "
        "pre-empts the silent-missing-resource fabrication mode + the retired-mechanism trap."
    ),
}

CommandProbe = _baseline.CommandProbe
CommandRunner = _baseline.CommandRunner
Clock = _baseline.Clock

run_command = _baseline.run_command
prepend_oss_cad_suite = _baseline.prepend_oss_cad_suite
command_to_string = _baseline.command_to_string
payload_checksum = _baseline.payload_checksum
state_token = _baseline.state_token


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4345 artifact from live or injected board probes."""
    root = Path(repo_root)
    started = clock()
    preconditions: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []

    kv260_probe = command_runner(KV260_SSH_PRECONDITION, 10.0)
    kv260_reachable = kv260_probe.exit_code == 0
    preconditions.append(_precondition_entry("kv260_ssh", "ssh", kv260_probe, kv260_reachable))
    transcript.append(_transcript_entry("kv260", "kv260_ssh_precondition", kv260_probe))

    boards_probed = [
        {
            "name": "kv260",
            "reachable": kv260_reachable,
            "state": "kv260_ssh_reachable" if kv260_reachable else "blocked_kv260_ssh_unreachable",
        }
    ]

    if kv260_reachable:
        xmutil_probe = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
        boards_probed[0]["state"] = _kv260_listapps_state(xmutil_probe)
        transcript.append(_transcript_entry("kv260", "kv260_xmutil_listapps", xmutil_probe))

        gatemate_probe = command_runner(GATEMATE_DETECT_COMMAND, 30.0)
        gatemate_idcode = _gatemate_idcode(gatemate_probe)
        gatemate_reachable = bool(gatemate_idcode)
        preconditions.append(
            _precondition_entry("gatemate_jtag_detect", "usb_dirtyjtag", gatemate_probe, gatemate_reachable)
        )
        transcript.append(_transcript_entry("gatemate", "gatemate_dirtyjtag_detect", gatemate_probe))
        boards_probed.append(
            {
                "name": "gatemate",
                "reachable": gatemate_reachable,
                "state": (
                    f"gatemate_idcode_{gatemate_idcode}"
                    if gatemate_reachable
                    else "blocked_gatemate_unreachable"
                ),
            }
        )

        polarfire_probe = command_runner(POLARFIRE_SSH_PRECONDITION, 10.0)
        polarfire_reachable = polarfire_probe.exit_code == 0
        preconditions.append(
            _precondition_entry("polarfire_ssh", "ssh", polarfire_probe, polarfire_reachable)
        )
        transcript.append(_transcript_entry("polarfire", "polarfire_ssh_precondition", polarfire_probe))
        boards_probed.append(
            {
                "name": "polarfire",
                "reachable": polarfire_reachable,
                "state": "polarfire_ssh_reachable"
                if polarfire_reachable
                else "blocked_polarfire_unreachable",
            }
        )

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
        "boards_probed": boards_probed,
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
    """Return the terminal-prefixed Exp 4345 verdict."""
    if artifact.get("kv260_reachable") is False:
        return "blocked_kv260_ssh_unreachable"
    states = {entry["name"]: entry["state"] for entry in artifact["boards_probed"]}
    return (
        "complete: hardware_continuity_4345_"
        f"kv260_{state_token(str(states['kv260']))}_"
        f"gatemate_{state_token(str(states['gatemate']))}_"
        f"polarfire_{state_token(str(states['polarfire']))}_"
        "ssh_only_opportunistic"
    )


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4345 and write `results/experiment_4345_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4345")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4345")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4345")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4345")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _validate_principles(artifact)
    _require(isinstance(artifact.get("kv260_reachable"), bool), "kv260_reachable must be bare bool")
    _validate_boards_probed(artifact)
    _validate_preconditions(artifact)
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


def _precondition_entry(
    resource: str,
    mechanism: str,
    probe: CommandProbe,
    available: bool,
) -> dict[str, Any]:
    return {
        "resource": resource,
        "mechanism": mechanism,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "duration_s": _round_duration(probe.duration_s),
        "exit_code": probe.exit_code,
        "observed": _observed(probe),
    }


def _transcript_entry(board: str, stage: str, probe: CommandProbe) -> dict[str, Any]:
    return {
        "board": board,
        "stage": stage,
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "duration_s": _round_duration(probe.duration_s),
        "output_excerpt": _observed(probe)[:1000],
    }


def _kv260_listapps_state(probe: CommandProbe) -> str:
    if probe.exit_code != 0:
        return f"kv260_xmutil_listapps_blocked_returncode_{probe.exit_code}"
    observed = _observed(probe).lower()
    return (
        "kv260_carnot_ising_listapps_seen"
        if "carnot_ising" in observed
        else "kv260_xmutil_listapps_no_carnot_ising_seen"
    )


def _gatemate_idcode(probe: CommandProbe) -> str:
    return "0x20000001" if "0x20000001" in _observed(probe).lower() else ""


def _observed(probe: CommandProbe) -> str:
    text = probe.combined_output.strip()
    return text if text else f"returncode={probe.exit_code}"


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4334_hardware_continuity.json"
    prior_payload = _read_json(prior_path)
    return {
        "previous_experiment": 4334,
        "most_recent_hardware_continuity_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_kv260_reachable": prior_payload.get("kv260_reachable"),
        "previous_per_board_reachability": prior_payload.get("per_board_reachability"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact.
        return {}


def _validate_principles(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(isinstance(principles, dict), "field_principles must be dict")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(principles.get(field) == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
        )


def _validate_boards_probed(artifact: dict[str, Any]) -> None:
    boards = artifact.get("boards_probed")
    _require(isinstance(boards, list), "boards_probed must be a list")
    names = [entry.get("name") for entry in boards if isinstance(entry, dict)]
    expected = list(BOARD_NAMES) if artifact["kv260_reachable"] else ["kv260"]
    _require(names == expected, "boards_probed has wrong board order")
    for entry in boards:
        _require(isinstance(entry, dict), "board entries must be dicts")
        _require(set(entry) == {"name", "reachable", "state"}, "board entries must stay minimal")
        _require(isinstance(entry["reachable"], bool), "board reachable must be bool")
        _require(isinstance(entry["state"], str) and entry["state"], "board state must be non-empty")
    if artifact["kv260_reachable"]:
        _require(boards[0]["reachable"] is True, "reachable artifact must have KV260 reachable")
    else:
        _require(boards == [{"name": "kv260", "reachable": False, "state": "blocked_kv260_ssh_unreachable"}], "blocked KV260 board mismatch")


def _validate_preconditions(artifact: dict[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list), "preconditions_checked must be a list")
    expected_resources = (
        ["kv260_ssh", "gatemate_jtag_detect", "polarfire_ssh"]
        if artifact["kv260_reachable"]
        else ["kv260_ssh"]
    )
    expected_commands = {
        "kv260_ssh": command_to_string(KV260_SSH_PRECONDITION),
        "gatemate_jtag_detect": command_to_string(GATEMATE_DETECT_COMMAND),
        "polarfire_ssh": command_to_string(POLARFIRE_SSH_PRECONDITION),
    }
    _require(
        [entry.get("resource") for entry in preconditions if isinstance(entry, dict)] == expected_resources,
        "preconditions_checked has wrong resources",
    )
    for entry in preconditions:
        _require(isinstance(entry, dict), "precondition entries must be dicts")
        _require({"resource", "available", "command", "exit_code"} <= set(entry), "missing precondition keys")
        _require(isinstance(entry["available"], bool), "precondition available must be bool")
        _require(entry["command"] == expected_commands[entry["resource"]], "invalid precondition command")


def _validate_transcript(artifact: dict[str, Any]) -> None:
    transcript = artifact.get("board_state_transcript")
    _require(isinstance(transcript, list), "board_state_transcript must be a list")
    expected_stages = (
        [
            "kv260_ssh_precondition",
            "kv260_xmutil_listapps",
            "gatemate_dirtyjtag_detect",
            "polarfire_ssh_precondition",
        ]
        if artifact["kv260_reachable"]
        else ["kv260_ssh_precondition"]
    )
    _require(
        [entry.get("stage") for entry in transcript if isinstance(entry, dict)] == expected_stages,
        "board_state_transcript has wrong stages",
    )
    for entry in transcript:
        _require(isinstance(entry, dict), "transcript entries must be dicts")
        _require({"board", "stage", "command", "exit_code", "output_excerpt"} <= set(entry), "bad transcript entry")


def _validate_source_context(artifact: dict[str, Any]) -> None:
    source = artifact.get("source_context")
    _require(isinstance(source, dict), "source_context must be present")
    _require(source.get("previous_experiment") == 4334, "source_context must read Exp 4334")
    _require(
        str(source.get("most_recent_hardware_continuity_artifact", "")).endswith(
            "experiment_4334_hardware_continuity.json"
        ),
        "source_context must point at Exp 4334",
    )


def _round_duration(value: float) -> float:
    return round(max(float(value), 0.000001), 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)  # pragma: no cover


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"boards_probed: {artifact['boards_probed']}")


if __name__ == "__main__":  # pragma: no cover
    main()
