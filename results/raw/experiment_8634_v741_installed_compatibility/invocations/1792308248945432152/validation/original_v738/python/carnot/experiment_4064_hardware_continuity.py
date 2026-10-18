"""Exp 4064 hardware continuity with GateMate/PolarFire forward progress.

Spec refs: REQ-HW-4064, SCENARIO-HW-4064.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from typing import Any


EXPERIMENT_ID = 4064
SCHEMA = "carnot.hardware_continuity_drive_to_terminal.v1"
SPEC_REFS = ["REQ-HW-4064", "SCENARIO-HW-4064"]
OUTPUT_REL_PATH = Path("results") / "experiment_4064_hardware_continuity.json"
RANDOM_SEED = 4064
INFERENCE_SUBSTRATE = "hardware_smoke"
BOARD_NAMES = ("kv260", "gatemate", "polarfire")

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

GATEMATE_FLASH_BOARD = "olimex_gatemateevb"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "per_board_reachability",
    "gatemate_step_taken",
    "polarfire_step_taken",
    "kv260_terminal_confirmed",
    "preconditions_checked",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix naming the GateMate and PolarFire drive-to-terminal steps "
        "while KV260 remains opportunistic"
    ),
    "per_board_reachability": (
        "principle: the continuity record that keeps boards visible in retros"
    ),
    "gatemate_step_taken": (
        "principle: the drive-to-terminal progress for the non-terminal GateMate board"
    ),
    "polarfire_step_taken": (
        "principle: the drive-to-terminal progress for the non-terminal PolarFire board"
    ),
    "kv260_terminal_confirmed": (
        "principle: KV260 is done -- opportunistic confirm only"
    ),
    "preconditions_checked": (
        "principle: records which board accesses were verified before the smoke, "
        "pre-empting the fabricate-when-unreachable failure mode"
    ),
    "inference_substrate": (
        "declares this artifact as hardware_smoke rather than model inference"
    ),
}


@dataclass(frozen=True)
class CommandProbe:
    """Captured command transcript for preconditions and bounded hardware steps."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def combined_output(self) -> str:
        return "\n".join(part.rstrip() for part in (self.stdout, self.stderr) if part.rstrip())


@dataclass(frozen=True)
class StepOutcome:
    """A single board forward-step result used by the continuity artifact."""

    step_taken: str
    terminal_state: str
    success: bool
    duration_s: float
    details: dict[str, Any]


CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]
Clock = Callable[[], float]
StepRunner = Callable[..., StepOutcome]


def prepend_oss_cad_suite() -> None:  # pragma: no cover - environment dependent
    """Expose the standard OSS CAD Suite tools when installed under /opt."""
    candidate = Path("/opt/oss-cad-suite/bin")
    if not (candidate / "openFPGALoader").exists():
        return
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if str(candidate) not in parts:
        os.environ["PATH"] = os.pathsep.join([str(candidate), *parts])


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandProbe:  # pragma: no cover
    """Run a local command and return a bounded transcript."""
    started = time.perf_counter()
    try:  # pragma: no cover - exercised by the live hardware script.
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandProbe(
            command=tuple(command),
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=_positive_duration(time.perf_counter() - started),
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
        return CommandProbe(tuple(command), 124, stdout, stderr, _positive_duration(time.perf_counter() - started))
    except OSError as exc:  # pragma: no cover
        return CommandProbe(tuple(command), 127, "", str(exc), _positive_duration(time.perf_counter() - started))


def command_to_string(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def payload_checksum(artifact: dict[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    gatemate_step_runner: StepRunner | None = None,
    polarfire_step_runner: StepRunner | None = None,
) -> dict[str, Any]:
    """Build the consolidated Exp 4064 hardware-continuity artifact."""
    root = Path(repo_root)
    started = clock()
    preconditions, reachability, precondition_durations = check_preconditions(command_runner)
    per_board_duration_s = dict(precondition_durations)
    terminal_state = {
        "kv260": (
            "opportunistic_terminal_confirmed_ssh_only"
            if reachability["kv260"]
            else "blocked_kv260_unreachable"
        ),
        "gatemate": "blocked_gatemate_unreachable",
        "polarfire": "blocked_polarfire_unreachable",
    }

    kv260_terminal_confirmed = bool(reachability["kv260"])
    gate_runner = gatemate_step_runner or run_gatemate_forward_step
    polar_runner = polarfire_step_runner or run_polarfire_forward_step
    gatemate_step = _blocked_step("gatemate")
    polarfire_step = _blocked_step("polarfire")

    if reachability["gatemate"]:
        gatemate_outcome = gate_runner(
            repo_root=root,
            command_runner=command_runner,
            clock=clock,
        )
        gatemate_step = gatemate_outcome.details
        terminal_state["gatemate"] = gatemate_outcome.terminal_state
        per_board_duration_s["gatemate"] = _round_duration(
            per_board_duration_s["gatemate"] + gatemate_outcome.duration_s
        )
        gatemate_step_taken = gatemate_outcome.step_taken
    else:
        gatemate_step_taken = "blocked_gatemate_unreachable"

    if reachability["polarfire"]:
        polarfire_outcome = polar_runner(
            repo_root=root,
            command_runner=command_runner,
            clock=clock,
        )
        polarfire_step = polarfire_outcome.details
        terminal_state["polarfire"] = polarfire_outcome.terminal_state
        per_board_duration_s["polarfire"] = _round_duration(
            per_board_duration_s["polarfire"] + polarfire_outcome.duration_s
        )
        polarfire_step_taken = polarfire_outcome.step_taken
    else:
        polarfire_step_taken = "blocked_polarfire_unreachable"

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "honest_verdict": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "per_board_reachability": dict(reachability),
        "kv260_reachable": bool(reachability["kv260"]),
        "gatemate_reachable": bool(reachability["gatemate"]),
        "polarfire_reachable": bool(reachability["polarfire"]),
        "kv260_terminal_confirmed": kv260_terminal_confirmed,
        "kv260_step_taken": (
            "kv260_opportunistic_ssh_confirm_only"
            if kv260_terminal_confirmed
            else "blocked_kv260_unreachable"
        ),
        "gatemate_step_taken": gatemate_step_taken,
        "polarfire_step_taken": polarfire_step_taken,
        "gatemate_step": gatemate_step,
        "polarfire_step": polarfire_step,
        "per_board_terminal_state": terminal_state,
        "per_board_next_step": {
            "kv260": (
                "kv260_terminal_opportunistic_confirm_only"
                if kv260_terminal_confirmed
                else "blocked_kv260_unreachable"
            ),
            "gatemate": gatemate_step_taken,
            "polarfire": polarfire_step_taken,
        },
        "per_board_duration_s": {
            board: _round_duration(per_board_duration_s[board]) for board in BOARD_NAMES
        },
        "duration_s": _elapsed(started, clock),
        "fabric_acceleration_claimed": False,
        "speedup_claim_made": False,
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def check_preconditions(
    command_runner: CommandRunner,
) -> tuple[list[dict[str, Any]], dict[str, bool], dict[str, float]]:
    """Run every board-access precondition before any smoke step."""
    specs = (
        ("kv260", "kv260_ssh", KV260_SSH_PRECONDITION, _ssh_available, 10.0),
        ("gatemate", "gatemate_jtag_detect", GATEMATE_DETECT_COMMAND, _gatemate_available, 30.0),
        ("polarfire", "polarfire_ssh", POLARFIRE_SSH_PRECONDITION, _ssh_available, 10.0),
    )
    entries: list[dict[str, Any]] = []
    reachability: dict[str, bool] = {}
    durations: dict[str, float] = {}
    for board, resource, command, predicate, timeout_s in specs:
        probe = command_runner(command, timeout_s)
        available = bool(predicate(probe))
        entries.append(_precondition_entry(resource, probe, available))
        reachability[board] = available
        durations[board] = _round_duration(probe.duration_s)
    return entries, reachability, durations


def run_gatemate_forward_step(
    *,
    repo_root: Path,
    command_runner: CommandRunner,
    clock: Clock = time.perf_counter,
) -> StepOutcome:
    """Flash an existing n=16 GateMate bitstream and re-detect the board."""
    started = clock()
    candidates = _candidate_gatemate_bitstreams(repo_root)
    bitstream = next((candidate for candidate in candidates if candidate.exists()), None)
    if bitstream is None:
        return StepOutcome(
            step_taken="blocked_gatemate_no_existing_n16_bitstream",
            terminal_state="reachable_step_blocked_no_existing_n16_bitstream",
            success=False,
            duration_s=_elapsed(started, clock),
            details={
                "step": "flash_existing_n16_bitstream_plus_dirtyjtag_detect_smoke",
                "candidate_paths_checked": [str(path) for path in candidates],
                "blocker": "blocked_gatemate_no_existing_n16_bitstream",
            },
        )

    flash_command = (
        "openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        GATEMATE_FLASH_BOARD,
        str(bitstream),
    )
    flash_probe = command_runner(flash_command, 120.0)
    detect_probe = command_runner(GATEMATE_DETECT_COMMAND, 30.0)
    flash_ok = flash_probe.exit_code == 0
    detect_ok = _gatemate_available(detect_probe)
    success = flash_ok and detect_ok
    if success:
        step_taken = "gatemate_existing_n16_bitstream_flash_detect_smoke_succeeded"
        terminal = "reachable_n16_bitstream_flashed_detect_smoke_recorded"
    elif not flash_ok:  # pragma: no cover - live failure branch.
        step_taken = f"gatemate_existing_n16_bitstream_flash_blocked_returncode_{flash_probe.exit_code}"
        terminal = "reachable_n16_bitstream_flash_blocked"
    else:  # pragma: no cover - live failure branch.
        step_taken = "gatemate_existing_n16_bitstream_post_flash_detect_blocked"
        terminal = "reachable_n16_bitstream_post_flash_detect_blocked"
    return StepOutcome(
        step_taken=step_taken,
        terminal_state=terminal,
        success=success,
        duration_s=_elapsed(started, clock),
        details={
            "step": "flash_existing_n16_bitstream_plus_dirtyjtag_detect_smoke",
            "bitstream_path": str(bitstream),
            "bitstream_sha256": _sha256_file(bitstream),
            "flash_command": command_to_string(flash_command),
            "post_flash_detect_command": command_to_string(GATEMATE_DETECT_COMMAND),
            "flash_exit_code": flash_probe.exit_code,
            "post_flash_detect_exit_code": detect_probe.exit_code,
            "post_flash_detected_gatemate": detect_ok,
            "command_transcripts": [
                _command_transcript("flash_existing_bitstream", flash_probe),
                _command_transcript("post_flash_detect_smoke", detect_probe),
            ],
        },
    )


def run_polarfire_forward_step(
    *,
    repo_root: Path,
    command_runner: CommandRunner,
    clock: Clock = time.perf_counter,
) -> StepOutcome:
    """Run a short PolarFire CPU dispatch and require a host/board hash match."""
    del repo_root
    from carnot import experiment_3867_polarfire_soc_smoke_v4 as polarfire_helper

    started = clock()
    python_command = ("ssh", "polarfire", "python3 --version")
    python_probe = command_runner(python_command, 10.0)
    helper_preconditions = [_precondition_entry("polarfire_python", python_probe, python_probe.exit_code == 0)]
    if python_probe.exit_code != 0:  # pragma: no cover - live failure branch.
        return StepOutcome(
            step_taken="blocked_polarfire_python_unavailable",
            terminal_state="reachable_cpu_dispatch_blocked_python_unavailable",
            success=False,
            duration_s=_elapsed(started, clock),
            details={
                "step": "hash_verified_cpu_dispatch_smoke",
                "preconditions_checked": helper_preconditions,
                "blocker": "blocked_polarfire_python_unavailable",
            },
        )

    workload = polarfire_helper.build_ising_workload(RANDOM_SEED)
    cpu_reference = polarfire_helper.evaluate_ising_workload(workload)
    cpu_reference_sha256 = polarfire_helper.sha256_json(cpu_reference)

    def helper_runner(args: Sequence[str], timeout: float | None = None) -> Any:
        probe = command_runner(tuple(str(part) for part in args), timeout or 60.0)
        return polarfire_helper.CommandResult(
            args=tuple(str(part) for part in args),
            returncode=probe.exit_code,
            stdout=probe.stdout,
            stderr=probe.stderr,
        )

    remote_dir = f"/tmp/carnot_exp4064_{os.getpid()}_{int(time.time())}"
    command_transcript: list[dict[str, Any]]
    try:
        with tempfile.TemporaryDirectory(prefix="carnot_exp4064_") as tmpdir:
            tmp = Path(tmpdir)
            workload_path = tmp / "workload.json"
            runner_path = tmp / "runner.py"
            workload_path.write_text(json.dumps(workload, sort_keys=True, indent=2), encoding="utf-8")
            runner_path.write_text(polarfire_helper.build_remote_runner_source(), encoding="utf-8")
            board_result, command_transcript = polarfire_helper.dispatch_to_board(
                runner=helper_runner,
                host="polarfire",
                workload_path=workload_path,
                runner_path=runner_path,
                remote_dir=remote_dir,
                min_remote_runtime_s=0.0,
            )
        soc_temp_max_c, thermal_transcript = polarfire_helper.probe_soc_temp_max_c(
            helper_runner,
            "polarfire",
        )
        command_transcript.append(thermal_transcript)
    except polarfire_helper.DispatchError as exc:  # pragma: no cover - live failure branch.
        return StepOutcome(
            step_taken="blocked_polarfire_cpu_dispatch_failed",
            terminal_state="reachable_cpu_dispatch_blocked",
            success=False,
            duration_s=_elapsed(started, clock),
            details={
                "step": "hash_verified_cpu_dispatch_smoke",
                "preconditions_checked": helper_preconditions,
                "blocker": exc.verdict,
                "failure_detail": exc.detail,
                "command_transcript": exc.transcript,
            },
        )

    board_result_sha256 = polarfire_helper.sha256_json(board_result)
    result_hash_match = board_result_sha256 == cpu_reference_sha256
    success = result_hash_match
    if success:
        step_taken = "polarfire_hash_verified_cpu_dispatch_succeeded"
        terminal = "reachable_hash_verified_cpu_dispatch_recorded"
    else:  # pragma: no cover - live failure branch.
        step_taken = "polarfire_hash_verified_cpu_dispatch_hash_mismatch"
        terminal = "reachable_cpu_dispatch_hash_mismatch"
    return StepOutcome(
        step_taken=step_taken,
        terminal_state=terminal,
        success=success,
        duration_s=_elapsed(started, clock),
        details={
            "step": "hash_verified_cpu_dispatch_smoke",
            "preconditions_checked": helper_preconditions,
            "result_hash_match": result_hash_match,
            "board_result_sha256": board_result_sha256,
            "cpu_reference_sha256": cpu_reference_sha256,
            "workload_sha256": polarfire_helper.sha256_json(workload),
            "soc_temp_max_c": soc_temp_max_c,
            "command_transcript": command_transcript,
        },
    )


def honest_verdict(artifact: dict[str, Any]) -> str:
    if not any(bool(artifact[f"{board}_reachable"]) for board in BOARD_NAMES):
        return "blocked_all_boards_unreachable"
    return (
        "complete: hardware_continuity_"
        f"gatemate_{state_token(str(artifact['gatemate_step_taken']))}_"
        f"polarfire_{state_token(str(artifact['polarfire_step_taken']))}_"
        f"kv260_{state_token(str(artifact['per_board_terminal_state']['kv260']))}"
    )


def state_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    return "_".join(part for part in token.split("_") if part)


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4064")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4064")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4064")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4064")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _validate_principles(artifact)
    _validate_preconditions(artifact)
    _validate_boards(artifact)
    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be hardware_smoke",
    )
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    _require("mmcblk" not in json.dumps(artifact, sort_keys=True, default=str).lower(), "")
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale verdict")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


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
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def _precondition_entry(resource: str, probe: CommandProbe, available: bool) -> dict[str, Any]:
    return {
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "duration_s": _round_duration(probe.duration_s),
        "exit_code": probe.exit_code,
        "observed": _observed(probe),
    }


def _ssh_available(probe: CommandProbe) -> bool:
    return probe.exit_code == 0


def _gatemate_available(probe: CommandProbe) -> bool:
    text = probe.combined_output.lower()
    return (
        probe.exit_code == 0
        and "idcode" in text
        and ("gatemate" in text or "colognechip" in text or "gm1a" in text)
    )


def _observed(probe: CommandProbe) -> str:
    text = probe.combined_output.strip()
    return text if text else f"returncode={probe.exit_code}"


def _command_transcript(stage: str, probe: CommandProbe) -> dict[str, Any]:
    return {
        "stage": stage,
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "output_excerpt": probe.combined_output[-1200:],
        "duration_s": _round_duration(probe.duration_s),
    }


def _candidate_gatemate_bitstreams(repo_root: Path) -> list[Path]:
    candidates = [
        repo_root
        / "build"
        / "gatemate"
        / "experiment_3866_gatemate_ising_tile_flash_v2"
        / "gatemate_ising_n16.bit",
        repo_root / "rtl" / "gatemate_ising_n16.bit",
        repo_root / "rtl" / "gatemate_ising_n16_packed.bit",
    ]
    prior = repo_root / "results" / "experiment_3866_gatemate_ising_tile_flash_v2.json"
    if prior.exists():
        try:
            bitstream = json.loads(prior.read_text(encoding="utf-8")).get("bitstream_path", "")
            if bitstream:
                candidates.insert(0, Path(bitstream))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt prior artifact.
            pass
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blocked_step(board: str) -> dict[str, Any]:
    return {"step": f"blocked_{board}_unreachable", "blocker": f"blocked_{board}_unreachable"}


def _validate_principles(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(isinstance(principles, dict), "field_principles must be dict")
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
    _require(resources == ["kv260_ssh", "gatemate_jtag_detect", "polarfire_ssh"], "")
    for entry in preconditions:
        _require(isinstance(entry, dict), "precondition entries must be dicts")
        _require({"resource", "available"} <= set(entry), "precondition missing required keys")
        _require(isinstance(entry["available"], bool), "available must be bool")


def _validate_boards(artifact: dict[str, Any]) -> None:
    reachability = artifact.get("per_board_reachability")
    terminal_state = artifact.get("per_board_terminal_state")
    next_step = artifact.get("per_board_next_step")
    durations = artifact.get("per_board_duration_s")
    for mapping, name in (
        (reachability, "per_board_reachability"),
        (terminal_state, "per_board_terminal_state"),
        (next_step, "per_board_next_step"),
        (durations, "per_board_duration_s"),
    ):
        _require(isinstance(mapping, dict), f"{name} must be a dict")
        _require(set(mapping) == set(BOARD_NAMES), f"{name} must be keyed by all boards")
    for board in BOARD_NAMES:
        _require(isinstance(reachability[board], bool), "reachability values must be bool")
        _require(reachability[board] is artifact[f"{board}_reachable"], "scalar mismatch")
        _require(isinstance(terminal_state[board], str) and terminal_state[board], "")
        _require(isinstance(next_step[board], str) and next_step[board], "")
        _require(float(durations[board]) > 0.0, "per-board duration must be positive")
        if not reachability[board]:
            _require(next_step[board] == f"blocked_{board}_unreachable", "")
    _require(isinstance(artifact["kv260_terminal_confirmed"], bool), "kv260 gate must be bool")
    _require(artifact["kv260_terminal_confirmed"] is reachability["kv260"], "")
    _require(isinstance(artifact["gatemate_step_taken"], str), "GateMate step must be str")
    _require(isinstance(artifact["polarfire_step_taken"], str), "PolarFire step must be str")
    _require(float(artifact.get("duration_s", 0.0)) > 0.0, "duration_s must be positive")


def _positive_duration(value: float) -> float:
    return max(float(value), 0.000001)


def _round_duration(value: float) -> float:
    return round(_positive_duration(value), 6)


def _elapsed(started: float, clock: Clock) -> float:
    return _round_duration(clock() - started)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)  # pragma: no cover


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"per_board_reachability: {artifact['per_board_reachability']}")
    print(f"gatemate_step_taken: {artifact['gatemate_step_taken']}")
    print(f"polarfire_step_taken: {artifact['polarfire_step_taken']}")
    print(f"kv260_terminal_confirmed: {artifact['kv260_terminal_confirmed']}")


if __name__ == "__main__":  # pragma: no cover
    main()
