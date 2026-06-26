"""Exp 4757 KV260 SSH-only continuity artifact.

Spec refs: REQ-HW-4757, SCENARIO-HW-4757.

This module is intentionally a state audit, not a hardware benchmark. The KV260
has already graduated its terminal latency criteria, so this per-milestone check
keeps the board visible by proving SSH reachability, preserving the `xmutil`
overlay transcript, and recording the next concrete board step. The guardrail is
simple but important: KV260 continuity lives on SSH, never on host SD-card device
nodes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXPERIMENT_ID = 4757
SCHEMA = "carnot.kv260_ssh_continuity.v1"
SPEC_REFS = ["REQ-HW-4757", "SCENARIO-HW-4757"]
OUTPUT_REL_PATH = Path("results") / "experiment_4757_kv260_continuity.json"
RANDOM_SEED = 4757
INFERENCE_SUBSTRATE = "hardware_smoke"

KV260_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
KV260_LISTAPPS_COMMAND = ("ssh", "kria", "xmutil listapps")
KV260_LISTAPPS_SUDO_COMMAND = ("ssh", "kria", "sudo xmutil listapps")
KV260_BOARD_STATE_COMMAND = (
    "ssh",
    "kria",
    "hostname; uname -a; uptime; ls /dev/uio* 2>/dev/null | wc -l",
)

VALID_OVERLAYS = ("carnot_ising_v4_n64", "carnot_ising_v2_n64", "carnot_ising_v4")

SUCCESS_VERDICT = "success: kv260_continuity_recorded"
BLOCKED_SSH_VERDICT = "complete:/blocked_kv260_ssh_unreachable"
REACHABLE_NEXT_FORWARD_STEP = (
    "GRADUATED: KV260 terminal criteria met; continue per-milestone "
    "SSH-only continuity checks."
)
BLOCKED_NEXT_FORWARD_STEP = (
    "Restore KV260 SSH reachability, then rerun the SSH-only continuity check; "
    "do not use host SD-card device nodes."
)

REQUIRED_OPERATOR_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "kv260_ssh_reachable",
    "next_forward_step",
)
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_OPERATOR_FIELDS,
    "schema",
    "experiment",
    "spec_refs",
    "field_principles",
    "duration_s",
    "command_probes",
    "loaded_overlay",
    "xmutil_requires_sudo",
    "board_state",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; blocked_kv260_ssh_unreachable is an honest non-terminal "
        "block, a continuity-recorded run is complete_."
    ),
    "inference_substrate": "hardware_smoke (SSH-attached board).",
    "preconditions_checked": (
        "records the SSH-reachability check (NEVER host SD-card) -- the KV260 "
        "wrong-mechanism guard."
    ),
    "kv260_ssh_reachable": (
        "the board's SSH reachability -- the only valid KV260 precondition."
    ),
    "next_forward_step": (
        "the next concrete board step recorded so the board stays visible in retros "
        "(the forget-pattern guard)."
    ),
}


@dataclass(frozen=True)
class CommandProbe:
    command: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_dict(self) -> dict[str, object]:
        return {
            "command": command_to_string(self.command),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "combined_output": self.combined_output,
        }


CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]
Clock = Callable[[], float]


def command_to_string(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandProbe:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - live timeout path.
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else f"timeout after {timeout_s}s"
        return CommandProbe(command, 124, stdout, stderr, time.perf_counter() - started)
    except OSError as exc:  # pragma: no cover - host process failure path.
        return CommandProbe(command, 127, "", str(exc), time.perf_counter() - started)
    return CommandProbe(
        command,
        completed.returncode,
        completed.stdout,
        completed.stderr,
        time.perf_counter() - started,
    )


def payload_checksum(payload: dict[str, object]) -> str:
    checksum_payload = dict(payload)
    checksum_payload.pop("reproducibility_checksum", None)
    encoded = json.dumps(
        checksum_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def loaded_overlay_from_xmutil(text: str) -> str | None:
    for line in text.splitlines():
        for overlay in VALID_OVERLAYS:
            if overlay not in line:
                continue
            lowered = line.lower()
            if "running" in lowered or "loaded" in lowered or "slot_handle 0" in lowered:
                return overlay
            if "->" in line and not line.rstrip().endswith("-1"):
                return overlay
    return None


def _xmutil_requires_root(probe: CommandProbe) -> bool:
    lowered = probe.combined_output.lower()
    return "root privileges" in lowered or "using 'sudo'" in lowered


def _observed_first_line(probe: CommandProbe) -> str:
    observed = probe.combined_output.strip() or f"returncode={probe.exit_code}"
    return observed.splitlines()[0][:300]


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


def _empty_command_probes(ssh_probe: CommandProbe) -> dict[str, object]:
    return {
        "kv260_ssh": ssh_probe.as_dict(),
        "kv260_xmutil_listapps": None,
        "kv260_xmutil_listapps_sudo": None,
        "kv260_board_state": None,
    }


def _parse_board_state(probe: CommandProbe) -> dict[str, object]:
    if probe.exit_code != 0:
        return {
            "captured": False,
            "exit_code": probe.exit_code,
            "observed": _observed_first_line(probe),
        }
    lines = probe.stdout.splitlines()
    uio_text = lines[3].strip() if len(lines) > 3 else ""
    return {
        "captured": True,
        "hostname": lines[0].strip() if len(lines) > 0 else "",
        "kernel": lines[1].strip() if len(lines) > 1 else "",
        "uptime": lines[2].strip() if len(lines) > 2 else "",
        "uio_device_count": int(uio_text) if uio_text.isdigit() else None,
        "raw_stdout": probe.stdout,
    }


def _base_artifact(ssh_probe: CommandProbe, duration_s: float) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": BLOCKED_SSH_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": [_precondition_entry(ssh_probe)],
        "kv260_ssh_reachable": ssh_probe.exit_code == 0,
        "next_forward_step": BLOCKED_NEXT_FORWARD_STEP,
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(max(float(duration_s), 0.0001), 4),
        "command_probes": _empty_command_probes(ssh_probe),
        "loaded_overlay": None,
        "xmutil_requires_sudo": False,
        "board_state": {"captured": False, "reason": "kv260_ssh_unreachable"},
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, object]:
    started = clock()
    ssh_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    artifact = _base_artifact(ssh_probe, clock() - started)

    if ssh_probe.exit_code != 0:
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
        validate_artifact(artifact)
        return artifact

    command_probes = artifact["command_probes"]
    list_probe = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
    command_probes["kv260_xmutil_listapps"] = list_probe.as_dict()
    listapps_text = list_probe.combined_output if list_probe.exit_code == 0 else ""
    requires_sudo = _xmutil_requires_root(list_probe)

    if requires_sudo:
        sudo_probe = command_runner(KV260_LISTAPPS_SUDO_COMMAND, 30.0)
        command_probes["kv260_xmutil_listapps_sudo"] = sudo_probe.as_dict()
        if sudo_probe.exit_code == 0:
            listapps_text = sudo_probe.combined_output

    board_probe = command_runner(KV260_BOARD_STATE_COMMAND, 30.0)
    command_probes["kv260_board_state"] = board_probe.as_dict()

    artifact.update(
        {
            "honest_verdict": SUCCESS_VERDICT,
            "loaded_overlay": loaded_overlay_from_xmutil(listapps_text),
            "xmutil_requires_sudo": requires_sudo,
            "board_state": _parse_board_state(board_probe),
            "next_forward_step": REACHABLE_NEXT_FORWARD_STEP,
            "duration_s": round(max(float(clock() - started), 0.0001), 4),
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(repo_root: str | Path, artifact: dict[str, object]) -> Path:
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, object]) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(artifact.get("schema") == SCHEMA, "schema mismatch")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment mismatch")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle must be false")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    _require(float(artifact.get("duration_s", 0.0)) > 0.0, "duration_s must be positive")
    for field in REQUIRED_OPERATOR_FIELDS:
        _require(field in artifact, f"missing operator field: {field}")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
        )
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require("mmcblk" not in encoded and "/dev/disk" not in encoded, "forbidden host storage marker")
    _validate_precondition(artifact)

    if artifact.get("kv260_ssh_reachable") is False:
        _require(artifact.get("honest_verdict") == BLOCKED_SSH_VERDICT, "bad blocked SSH verdict")
        _require(artifact.get("loaded_overlay") is None, "blocked SSH cannot report overlay")
        _require(artifact.get("next_forward_step") == BLOCKED_NEXT_FORWARD_STEP, "bad blocked next step")
        probes = artifact.get("command_probes")
        _require(probes.get("kv260_xmutil_listapps") is None, "blocked SSH cannot run xmutil")
        _require(probes.get("kv260_board_state") is None, "blocked SSH cannot run board state")
        _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")
        return

    _require(artifact.get("kv260_ssh_reachable") is True, "kv260_ssh_reachable must be bool")
    _require(artifact.get("honest_verdict") == SUCCESS_VERDICT, "bad success verdict")
    _require(artifact.get("next_forward_step") == REACHABLE_NEXT_FORWARD_STEP, "bad reachable next step")
    loaded_overlay = artifact.get("loaded_overlay")
    _require(loaded_overlay is None or loaded_overlay in VALID_OVERLAYS, "invalid loaded overlay")
    _require(isinstance(artifact.get("xmutil_requires_sudo"), bool), "xmutil_requires_sudo must be bool")
    probes = artifact.get("command_probes")
    _require(probes.get("kv260_xmutil_listapps") is not None, "success requires xmutil probe")
    if artifact.get("xmutil_requires_sudo"):
        _require(probes.get("kv260_xmutil_listapps_sudo") is not None, "sudo fallback missing")
    _require(probes.get("kv260_board_state") is not None, "success requires board state probe")
    board_state = artifact.get("board_state")
    _require(isinstance(board_state, dict), "board_state must be a dict")
    _require(board_state.get("captured") is True, "success requires captured board state")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _validate_precondition(artifact: dict[str, object]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list) and len(preconditions) == 1, "bad preconditions_checked")
    entry = preconditions[0]
    _require(entry.get("resource") == "kv260_ssh", "bad KV260 precondition resource")
    _require(entry.get("command") == command_to_string(KV260_SSH_COMMAND), "bad KV260 SSH command")
    _require(entry.get("discipline") == "ssh_only_no_host_sd_card", "bad KV260 discipline")
    _require(entry.get("available") is artifact.get("kv260_ssh_reachable"), "precondition mismatch")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"loaded_overlay: {artifact['loaded_overlay']}")
    print(f"next_forward_step: {artifact['next_forward_step']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
