"""Exp 4451 precondition-gated hardware continuity across attached boards.

This experiment keeps the hardware task visible without pretending a missing
board worked. Each board gets its own precondition first. Reachable boards take
one bounded next step, and blocked boards get a one-line audit that explains
why the step was not run.

Spec refs: REQ-HW-4451, SCENARIO-HW-4451.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot import experiment_4439_hardware_continuity as _base  # noqa: E402


EXPERIMENT_ID = 4451
SCHEMA = "carnot.hardware_continuity_precondition_gated.v3"
SPEC_REFS = ["REQ-HW-4451", "SCENARIO-HW-4451"]
OUTPUT_REL_PATH = Path("results") / "experiment_4451_hardware_continuity.json"
RANDOM_SEED = 4451
INFERENCE_SUBSTRATE = "hardware_smoke"
BOARD_NAMES = _base.BOARD_NAMES

KV260_SSH_PRECONDITION = _base.KV260_SSH_PRECONDITION
GATEMATE_NEXTPNR_PRECONDITION = _base.GATEMATE_NEXTPNR_PRECONDITION
GATEMATE_DETECT_COMMAND = _base.GATEMATE_DETECT_COMMAND
POLARFIRE_SSH_PRECONDITION = ("ssh", "-o", "ConnectTimeout=5", "polarfire", "true")

KV260_LATENCY_TRANSCRIPT_COMMAND = _base.KV260_LATENCY_TRANSCRIPT_COMMAND
GATEMATE_GMPACK_LOOKUP_COMMAND = _base.GATEMATE_GMPACK_LOOKUP_COMMAND
POLARFIRE_SAMPLER_SMOKE_COMMAND = _base.POLARFIRE_SAMPLER_SMOKE_COMMAND

REQUIRED_ARTIFACT_FIELDS = (
    "preconditions_checked",
    "per_board_status",
    "honest_verdict",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "preconditions_checked": (
        "list of {resource, available}; pre-empts the fabricate-when-resource-missing mode"
    ),
    "per_board_status": (
        "one status per attached board so each stays visible in the milestone "
        "retro (Hardware-Task Continuity)"
    ),
    "honest_verdict": (
        "terminal-prefixed; a clean SSH-unreachable is blocked_, an audit-only "
        "step is complete_"
    ),
    "inference_substrate": "hardware_smoke -- SSH-attached board test; per-board floor",
}

TERMINAL_PREFIXES = _base.TERMINAL_PREFIXES

CommandProbe = _base.CommandProbe
CommandRunner = _base.CommandRunner
Clock = _base.Clock

command_to_string = _base.command_to_string
payload_checksum = _base.payload_checksum
prepend_oss_cad_suite = _base.prepend_oss_cad_suite
run_command = _base.run_command
state_token = _base.state_token


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4451 artifact from live or injected board probes."""
    root = Path(repo_root)
    started = clock()
    preconditions = check_preconditions(command_runner)
    availability = {entry["resource"]: bool(entry["available"]) for entry in preconditions}

    per_board_status = {
        "kv260": _base._kv260_status(command_runner, availability["kv260_ssh"]),
        "gatemate": _gatemate_status(
            root,
            command_runner,
            nextpnr_available=availability["gatemate_nextpnr_himbaechel"],
            dirtyjtag_available=availability["gatemate_dirtyjtag_detect"],
        ),
        "polarfire": _base._polarfire_status(command_runner, availability["polarfire_ssh"]),
    }

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "honest_verdict": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_context": _source_context(root),
        "preconditions_checked": preconditions,
        "per_board_status": per_board_status,
        "duration_s": _base._round_duration(clock() - started),
        "fabric_acceleration_claimed": False,
        "speedup_claim_made": False,
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def check_preconditions(command_runner: CommandRunner) -> list[dict[str, Any]]:
    """Run the required preconditions before any concrete hardware step."""
    specs = (
        ("kv260_ssh", KV260_SSH_PRECONDITION, _base._exit_zero, 10.0),
        ("gatemate_nextpnr_himbaechel", GATEMATE_NEXTPNR_PRECONDITION, _base._nextpnr_seen, 10.0),
        ("gatemate_dirtyjtag_detect", GATEMATE_DETECT_COMMAND, _base._gatemate_detected, 30.0),
        ("polarfire_ssh", POLARFIRE_SSH_PRECONDITION, _base._exit_zero, 10.0),
    )
    entries: list[dict[str, Any]] = []
    for resource, command, predicate, timeout_s in specs:
        probe = command_runner(command, timeout_s)
        entries.append(_base._precondition_entry(resource, probe, bool(predicate(probe))))
    return entries


def honest_verdict(artifact: dict[str, Any]) -> str:
    """Return the overall terminal-prefixed verdict for the continuity artifact."""
    statuses = artifact["per_board_status"]
    if statuses["kv260"]["status"] == "blocked_kv260_ssh_unreachable":
        return "blocked_kv260_ssh_unreachable"
    return (
        "complete: hardware_continuity_4451_"
        f"kv260_{state_token(str(statuses['kv260']['status']))}_"
        f"gatemate_{state_token(str(statuses['gatemate']['status']))}_"
        f"polarfire_{state_token(str(statuses['polarfire']['status']))}"
    )


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run Exp 4451 and write `results/experiment_4451_hardware_continuity.json`."""
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    """Write the JSON artifact with stable formatting for audit diffs."""
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Reject artifacts that drift from the precondition-gated hardware contract."""
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4451")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4451")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4451")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4451")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing {field}")
    _validate_principles(artifact)
    _validate_preconditions(artifact)
    _base._validate_per_board_status(artifact)
    _validate_source_context(artifact)
    encoded = json.dumps(artifact, sort_keys=True, default=str)
    _require("mmcblk" not in encoded.lower(), "SD-card precondition marker is forbidden")
    _require("nextpnr-gatemate" not in encoded, "obsolete nextpnr-gatemate is forbidden")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "wrong substrate")
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    verdict = str(artifact.get("honest_verdict", ""))
    _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict terminal prefix")
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale verdict")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _gatemate_status(
    repo_root: Path,
    command_runner: CommandRunner,
    *,
    nextpnr_available: bool,
    dirtyjtag_available: bool,
) -> dict[str, Any]:
    if not nextpnr_available:
        return _base._blocked_status(
            "gatemate",
            "blocked_gatemate_nextpnr_himbaechel_unavailable",
            "GateMate nextpnr-himbaechel precondition failed; bitstream pack was not run.",
        )
    if not dirtyjtag_available:
        return _base._blocked_status(
            "gatemate",
            "blocked_gatemate_dirtyjtag_unreachable",
            "GateMate DirtyJTAG detect failed; bitstream pack was not run.",
        )
    cfg = _base._first_existing(_base._candidate_gatemate_configs(repo_root))
    if cfg is None:  # pragma: no cover - repo normally carries routed configs.
        return _base._blocked_status(
            "gatemate",
            "blocked_gatemate_config_missing",
            "GateMate was reachable, but no routed config was available to pack.",
        )

    packer_probe = command_runner(GATEMATE_GMPACK_LOOKUP_COMMAND, 10.0)
    if packer_probe.exit_code != 0:
        return _base._step_blocked_status(
            "gatemate",
            "bitstream_pack",
            "blocked_gatemate_gmpack_unavailable",
            "GateMate was reachable, but gmpack was unavailable for bitstream pack.",
            _base._step_transcript("gmpack_lookup", packer_probe),
        )

    packer = _base._observed(packer_probe).splitlines()[0]
    bitstream = _gatemate_output_bitstream(repo_root)
    pack_command = (packer, str(cfg), str(bitstream))
    pack_probe = command_runner(pack_command, 120.0)
    transcript = [
        _base._step_transcript("gmpack_lookup", packer_probe),
        _base._step_transcript("bitstream_pack", pack_probe),
    ]
    if pack_probe.exit_code != 0:
        return _base._step_blocked_status(
            "gatemate",
            "bitstream_pack",
            "blocked_gatemate_bitstream_pack_failed",
            "GateMate was reachable, but gmpack returned a non-zero status.",
            transcript,
        )
    if not bitstream.exists():  # pragma: no cover - defensive live guard.
        return _base._step_blocked_status(
            "gatemate",
            "bitstream_pack",
            "blocked_gatemate_bitstream_pack_output_missing",
            "GateMate pack command succeeded, but no output bitstream was observed.",
            transcript,
        )
    return {
        "board": "gatemate",
        "reachable": True,
        "step": "bitstream_pack",
        "status": "gatemate_bitstream_pack_succeeded",
        "honest_verdict": "complete: gatemate_bitstream_pack_succeeded",
        "audit": "GateMate preconditions passed; himbaechel-routed config was packed.",
        "config_path": str(cfg),
        "bitstream_path": str(bitstream),
        "bitstream_sha256": _base._sha256_file(bitstream),
        "step_transcript": transcript,
    }


def _gatemate_output_bitstream(repo_root: Path) -> Path:
    return (
        repo_root
        / "build"
        / "gatemate"
        / "experiment_4451_hardware_continuity"
        / "gatemate_ising_n16_exp4451.bit"
    )


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4439_hardware_continuity.json"
    prior_payload = _read_json(prior_path)
    prior_status = prior_payload.get("per_board_status", {})
    if not isinstance(prior_status, dict):
        prior_status = {}
    return {
        "previous_experiment": 4439,
        "most_recent_hardware_continuity_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_per_board_status": prior_status,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact guard.
        return {}
    return payload if isinstance(payload, dict) else {}


def _validate_principles(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(isinstance(principles, dict), "field_principles must be a dict")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(principles.get(field) == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
        )


def _validate_preconditions(artifact: dict[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list), "preconditions_checked must be a list")
    resources = [entry.get("resource") for entry in preconditions if isinstance(entry, dict)]
    _require(
        resources
        == [
            "kv260_ssh",
            "gatemate_nextpnr_himbaechel",
            "gatemate_dirtyjtag_detect",
            "polarfire_ssh",
        ],
        "precondition resources mismatch",
    )
    expected_commands = {
        "kv260_ssh": command_to_string(KV260_SSH_PRECONDITION),
        "gatemate_nextpnr_himbaechel": command_to_string(GATEMATE_NEXTPNR_PRECONDITION),
        "gatemate_dirtyjtag_detect": command_to_string(GATEMATE_DETECT_COMMAND),
        "polarfire_ssh": command_to_string(POLARFIRE_SSH_PRECONDITION),
    }
    for entry in preconditions:
        _require(isinstance(entry, dict), "precondition entries must be dicts")
        _require(isinstance(entry.get("available"), bool), "available must be a bare bool")
        _require(isinstance(entry.get("resource"), str), "resource must be a string")
        _require(entry.get("command") == expected_commands[entry["resource"]], "precondition command mismatch")


def _validate_source_context(artifact: dict[str, Any]) -> None:
    source = artifact.get("source_context")
    _require(isinstance(source, dict), "source_context must be a dict")
    _require(source.get("previous_experiment") == 4439, "source_context must read Exp 4439")
    _require(
        str(source.get("most_recent_hardware_continuity_artifact", "")).endswith(
            "experiment_4439_hardware_continuity.json"
        ),
        "source_context must point at Exp 4439",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover - exercised by the required run command.
    repo_root = Path(__file__).resolve().parents[2]
    out_path = run_experiment(repo_root=repo_root)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path.relative_to(repo_root)}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"preconditions_checked: {artifact['preconditions_checked']}")
    print(f"per_board_status: {artifact['per_board_status']}")


if __name__ == "__main__":  # pragma: no cover - direct script invocation.
    main()
