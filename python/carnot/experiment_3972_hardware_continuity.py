"""Owed KV260, GateMate, and PolarFire continuity rerun for Exp 3972.

The experiment is deliberately narrow: it answers whether each board is still
reachable through the allowed board path, records a small continuity transcript
for reachable boards, and names the next concrete forward step. It does not
claim FPGA acceleration or model inference. The separate per-board timers are
kept because prior hardware artifacts were flagged when one duration was reused
as if it described multiple boards.

Spec refs: REQ-HW-3972, SCENARIO-HW-3972.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any


EXPERIMENT_ID = 3972
SCHEMA = "carnot.hardware_continuity_owed_rerun.v1"
SPEC_REFS = ["REQ-HW-3972", "SCENARIO-HW-3972"]
OUTPUT_REL_PATH = Path("results") / "experiment_3972_hardware_continuity.json"
RANDOM_SEED = 3972
INFERENCE_SUBSTRATE = "hardware_smoke"

KV260_SSH_PRECONDITION = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
POLARFIRE_SSH_PRECONDITION = ("ssh", "-o", "ConnectTimeout=5", "polarfire", "true")

KV260_LISTAPPS_COMMAND = ("ssh", "kria", "xmutil listapps")
KV260_UIO_COMMAND = ("ssh", "kria", "ls /dev/uio*")
POLARFIRE_CONTINUITY_COMMAND = ("ssh", "polarfire", "uname -a && uptime")

BOARD_NAMES = ("kv260", "gatemate", "polarfire")
REQUIRED_ARTIFACT_FIELDS = (
    "kv260_reachable",
    "gatemate_reachable",
    "polarfire_reachable",
    "per_board_next_step",
    "per_board_duration_s",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)
FIELD_PRINCIPLES = {
    "kv260_reachable": (
        "BARE BOOLs -- per-board SSH/USB reachability (the continuity signal; "
        "keeps boards from being silently dropped)."
    ),
    "gatemate_reachable": (
        "BARE BOOLs -- per-board SSH/USB reachability (the continuity signal; "
        "keeps boards from being silently dropped)."
    ),
    "polarfire_reachable": (
        "BARE BOOLs -- per-board SSH/USB reachability (the continuity signal; "
        "keeps boards from being silently dropped)."
    ),
    "per_board_next_step": "The next concrete forward step per reachable board.",
    "per_board_duration_s": (
        "DISTINCT wall-clock per board -- identical durations across boards is a "
        "TAUTOLOGY flag (exp3866)."
    ),
    "preconditions_checked": (
        "list of {resource, available} -- records WHICH checks ran (pre-empts "
        "the fabrication mode)."
    ),
    "honest_verdict": "Terminal-prefix verdict plus hardware_smoke substrate.",
    "duration_s": "Terminal-prefix verdict plus hardware_smoke substrate.",
    "inference_substrate": "Terminal-prefix verdict plus hardware_smoke substrate.",
}
VALID_KV260_OVERLAYS = ("carnot_ising_v4", "carnot_ising_v2_n64", "carnot_ising")


@dataclass(frozen=True)
class CommandResult:
    """Small command transcript used by live runs and deterministic tests."""

    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": command_to_string(self.command),
            "exit_code": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": round_timer(self.duration_s),
            "combined_output": self.combined_output,
        }


CommandRunner = Callable[[tuple[str, ...], float], CommandResult]
Clock = Callable[[], float]


def command_to_string(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandResult:  # pragma: no cover - live subprocess boundary
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else f"timeout after {timeout_s}s"
        return CommandResult(command, 124, stdout, stderr, time.perf_counter() - started)
    except OSError as exc:
        return CommandResult(command, 127, "", str(exc), time.perf_counter() - started)
    return CommandResult(
        command,
        completed.returncode,
        completed.stdout,
        completed.stderr,
        time.perf_counter() - started,
    )


def payload_checksum(payload: dict[str, Any]) -> str:
    stable = dict(payload)
    stable.pop("reproducibility_checksum", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Run the three board preconditions and build the continuity artifact."""
    del repo_root
    started = clock()

    kv260_record, kv260_duration = measure_board(clock, lambda: check_kv260(command_runner))
    gatemate_record, gatemate_duration = measure_board(
        clock,
        lambda: check_gatemate(command_runner),
    )
    polarfire_record, polarfire_duration = measure_board(
        clock,
        lambda: check_polarfire(command_runner),
    )

    reachable = {
        "kv260": kv260_record["reachable"],
        "gatemate": gatemate_record["reachable"],
        "polarfire": polarfire_record["reachable"],
    }
    next_steps = {
        "kv260": kv260_record["next_step"],
        "gatemate": gatemate_record["next_step"],
        "polarfire": polarfire_record["next_step"],
    }
    durations = {
        "kv260": kv260_duration,
        "gatemate": gatemate_duration,
        "polarfire": polarfire_duration,
    }
    total_duration = round_timer(clock() - started)

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict(
            kv260_reachable=reachable["kv260"],
            gatemate_reachable=reachable["gatemate"],
            polarfire_reachable=reachable["polarfire"],
            kv260_state=kv260_record["state"],
            gatemate_state=gatemate_record["state"],
            polarfire_state=polarfire_record["state"],
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "kv260_reachable": reachable["kv260"],
        "gatemate_reachable": reachable["gatemate"],
        "polarfire_reachable": reachable["polarfire"],
        "per_board_next_step": next_steps,
        "per_board_duration_s": durations,
        "duration_s": total_duration,
        "preconditions_checked": [
            kv260_record["precondition"],
            gatemate_record["precondition"],
            polarfire_record["precondition"],
        ],
        "kv260_state": kv260_record["state"],
        "kv260_loaded_overlay": kv260_record["loaded_overlay"],
        "kv260_uio_devices": kv260_record["uio_devices"],
        "kv260_command_transcripts": kv260_record["command_transcripts"],
        "gatemate_state": gatemate_record["state"],
        "gatemate_detect_output": gatemate_record["detect_output"],
        "gatemate_command_transcript": gatemate_record["command_transcript"],
        "polarfire_state": polarfire_record["state"],
        "polarfire_continuity_output": polarfire_record["continuity_output"],
        "polarfire_command_transcripts": polarfire_record["command_transcripts"],
        "fabric_acceleration_claimed": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def measure_board(clock: Clock, action: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], float]:
    started = clock()
    record = action()
    return record, round_timer(clock() - started)


def check_kv260(command_runner: CommandRunner) -> dict[str, Any]:
    probe = command_runner(KV260_SSH_PRECONDITION, 10.0)
    reachable = probe.returncode == 0
    listapps_result: CommandResult | None = None
    uio_result: CommandResult | None = None
    overlay: str | None = None
    uio_devices: list[str] = []
    if reachable:
        listapps_result = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
        uio_result = command_runner(KV260_UIO_COMMAND, 10.0)
        overlay = (
            parse_kv260_loaded_overlay(listapps_result.combined_output)
            if listapps_result.returncode == 0
            else None
        )
        uio_devices = (
            parse_uio_devices(uio_result.combined_output) if uio_result.returncode == 0 else []
        )
    state = kv260_state(reachable, overlay, uio_devices)
    return {
        "reachable": reachable,
        "state": state,
        "next_step": kv260_next_step(reachable, overlay, uio_devices),
        "loaded_overlay": overlay,
        "uio_devices": uio_devices,
        "precondition": precondition_entry("kv260_ssh", probe),
        "command_transcripts": {
            "ssh_true": probe.as_dict(),
            "xmutil_listapps": listapps_result.as_dict() if listapps_result is not None else None,
            "uio_list": uio_result.as_dict() if uio_result is not None else None,
        },
    }


def check_gatemate(command_runner: CommandRunner) -> dict[str, Any]:
    detect_result = command_runner(GATEMATE_DETECT_COMMAND, 30.0)
    reachable = detects_gatemate(detect_result)
    state = "reachable_detected_gatemate_idcode" if reachable else "blocked_gatemate_unreachable"
    return {
        "reachable": reachable,
        "state": state,
        "next_step": (
            "gatemate_forward_step_run_minimal_ising_tile_smoke"
            if reachable
            else "blocked_gatemate_unreachable"
        ),
        "detect_output": excerpt(command_text(detect_result)),
        "precondition": precondition_entry("gatemate_jtag_detect", detect_result),
        "command_transcript": detect_result.as_dict(),
    }


def check_polarfire(command_runner: CommandRunner) -> dict[str, Any]:
    probe = command_runner(POLARFIRE_SSH_PRECONDITION, 10.0)
    reachable = probe.returncode == 0
    continuity_result: CommandResult | None = None
    if reachable:
        continuity_result = command_runner(POLARFIRE_CONTINUITY_COMMAND, 15.0)
    state = (
        "reachable_ssh_continuity_recorded"
        if reachable
        else "blocked_polarfire_unreachable"
    )
    return {
        "reachable": reachable,
        "state": state,
        "next_step": (
            "polarfire_forward_step_run_hash_verified_soft_cpu_dispatch"
            if reachable
            else "blocked_polarfire_unreachable"
        ),
        "continuity_output": (
            excerpt(command_text(continuity_result))
            if continuity_result is not None
            else "skipped: polarfire unreachable"
        ),
        "precondition": precondition_entry("polarfire_ssh", probe),
        "command_transcripts": {
            "ssh_true": probe.as_dict(),
            "continuity": continuity_result.as_dict() if continuity_result is not None else None,
        },
    }


def parse_kv260_loaded_overlay(text: str) -> str | None:
    for overlay in VALID_KV260_OVERLAYS:
        if overlay in text:
            return overlay
    return None


def parse_uio_devices(text: str) -> list[str]:
    devices: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"/dev/uio\d+", text):
        device = match.group(0)
        if device not in seen:
            devices.append(device)
            seen.add(device)
    return devices


def detects_gatemate(result: CommandResult) -> bool:
    text = command_text(result).lower()
    return (
        result.returncode == 0
        and "idcode" in text
        and ("colognechip" in text or "gatemate" in text or "gm1a" in text)
    )


def kv260_state(reachable: bool, overlay: str | None, uio_devices: list[str]) -> str:
    if not reachable:
        return "blocked_kv260_unreachable"
    if overlay and uio_devices:
        return "reachable_overlay_loaded_uio_present"
    if overlay:
        return "reachable_overlay_loaded_uio_absent"
    return "reachable_overlay_absent"


def kv260_next_step(reachable: bool, overlay: str | None, uio_devices: list[str]) -> str:
    if not reachable:
        return "blocked_kv260_unreachable"
    if overlay and uio_devices:
        return "kv260_forward_step_run_overlay_latency_smoke"
    if overlay:
        return "kv260_forward_step_restore_uio_binding_then_latency_smoke"
    return "kv260_forward_step_load_carnot_overlay_then_latency_smoke"


def honest_verdict(
    *,
    kv260_reachable: bool,
    gatemate_reachable: bool,
    polarfire_reachable: bool,
    kv260_state: str,
    gatemate_state: str,
    polarfire_state: str,
) -> str:
    if not (kv260_reachable or gatemate_reachable or polarfire_reachable):
        return "blocked_all_boards_unreachable"
    return (
        "complete: hardware_continuity_3972_"
        f"kv{state_token(kv260_state)}_"
        f"gm{state_token(gatemate_state)}_"
        f"pf{state_token(polarfire_state)}"
    )


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema") != SCHEMA:
        raise ValueError("schema must identify the Exp 3972 continuity artifact")
    if artifact.get("experiment") != EXPERIMENT_ID:
        raise ValueError("experiment must be 3972")
    if artifact.get("spec_refs") != SPEC_REFS:
        raise ValueError("spec_refs must reference REQ-HW-3972 and SCENARIO-HW-3972")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3972")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        raise ValueError("field_principles must be a dict")
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(principles)
    if missing_principles:
        raise ValueError(f"field_principles missing required fields: {sorted(missing_principles)}")
    for field in REQUIRED_ARTIFACT_FIELDS:
        value = artifact[field]
        if isinstance(value, dict) and set(value) == {"value", "principle"}:
            raise ValueError(f"{field} must remain a bare value, not a principle wrapper")
    for field in ("kv260_reachable", "gatemate_reachable", "polarfire_reachable"):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be bool")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be hardware_smoke")
    if artifact.get("fabric_acceleration_claimed") is not False:
        raise ValueError("fabric_acceleration_claimed must be false")
    validate_durations(artifact)
    validate_next_steps(artifact)
    validate_preconditions(artifact)
    validate_verdict(artifact)
    text = json.dumps(artifact, sort_keys=True, default=str).lower()
    if "/dev/mmcblk" in text:
        raise ValueError("artifact contains forbidden KV260 host storage marker")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match artifact content")


def validate_durations(artifact: dict[str, Any]) -> None:
    duration = positive_number(artifact.get("duration_s"))
    if duration <= 0.0:
        raise ValueError("duration_s must be positive")
    durations = artifact.get("per_board_duration_s")
    if not isinstance(durations, dict) or set(durations) != set(BOARD_NAMES):
        raise ValueError("per_board_duration_s keys must be kv260, gatemate, and polarfire")
    values = [positive_number(durations[board]) for board in BOARD_NAMES]
    if any(value <= 0.0 for value in values):
        raise ValueError("per_board_duration_s values must be positive")
    if len(set(values)) != len(values):
        raise ValueError("per_board_duration_s values must be distinct")


def validate_next_steps(artifact: dict[str, Any]) -> None:
    next_steps = artifact.get("per_board_next_step")
    if not isinstance(next_steps, dict) or set(next_steps) != set(BOARD_NAMES):
        raise ValueError("per_board_next_step keys must be kv260, gatemate, and polarfire")
    for board in BOARD_NAMES:
        step = next_steps[board]
        if not isinstance(step, str) or not step:
            raise ValueError("per_board_next_step values must be non-empty strings")
        reachable = bool(artifact[f"{board}_reachable"])
        blocked = f"blocked_{board}_unreachable"
        if reachable and step == blocked:
            raise ValueError(f"reachable board {board} cannot use {blocked}")
        if not reachable and step != blocked:
            raise ValueError(f"unreachable board {board} must use {blocked}")


def validate_preconditions(artifact: dict[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list):
        raise ValueError("preconditions_checked must be a list")
    expected = {"kv260_ssh", "gatemate_jtag_detect", "polarfire_ssh"}
    seen: set[str] = set()
    for entry in preconditions:
        if not isinstance(entry, dict) or not {"resource", "available"} <= set(entry):
            raise ValueError("preconditions_checked entries must include resource and available")
        if not isinstance(entry["available"], bool):
            raise ValueError("preconditions_checked available must be bool")
        seen.add(str(entry["resource"]))
    if seen != expected:
        raise ValueError("preconditions_checked must include kv260, gatemate, and polarfire")


def validate_verdict(artifact: dict[str, Any]) -> None:
    verdict = str(artifact.get("honest_verdict", ""))
    any_reachable = any(bool(artifact.get(f"{board}_reachable")) for board in BOARD_NAMES)
    if not verdict.startswith(("success:", "complete:", "blocked_")):
        raise ValueError("honest_verdict must start with a terminal prefix or blocked_")
    if any_reachable:
        if not verdict.startswith(("success:", "complete:")):
            raise ValueError("honest_verdict must use a terminal prefix for reachable boards")
    elif verdict != "blocked_all_boards_unreachable":
        raise ValueError("honest_verdict must be blocked_all_boards_unreachable")


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def precondition_entry(resource: str, result: CommandResult) -> dict[str, Any]:
    return {
        "resource": resource,
        "available": result.returncode == 0 if resource != "gatemate_jtag_detect" else detects_gatemate(result),
        "command": command_to_string(result.command),
        "exit_code": result.returncode,
        "observed": observed(result),
        "duration_s": round_timer(result.duration_s),
        "checked_before_board_operations": True,
    }


def command_text(result: CommandResult) -> str:
    return result.combined_output.strip()


def observed(result: CommandResult) -> str:
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        return excerpt(stdout)
    if stderr:
        return excerpt(stderr)
    return f"returncode={result.returncode}"


def excerpt(text: str, limit: int = 500) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def state_token(state: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", state.lower()).strip("_")
    return token or "unknown"


def positive_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number <= 0.0:
        return 0.0
    return number


def round_timer(value: Any) -> float:
    return round(float(value), 6)


def main() -> None:  # pragma: no cover - CLI wrapper
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"gatemate_reachable: {artifact['gatemate_reachable']}")
    print(f"polarfire_reachable: {artifact['polarfire_reachable']}")
    print(f"duration_s: {artifact['duration_s']}")
    print(f"per_board_duration_s: {artifact['per_board_duration_s']}")


if __name__ == "__main__":  # pragma: no cover
    main()
