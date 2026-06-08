"""Clean GateMate, PolarFire, and KV260 continuity rerun for Exp 3931.

Exp 3922 already had the right per-board shape, but its top-level timers could
remain zero when GateMate was blocked and another board was reachable. This
rerun keeps the same board preconditions and claim boundary while replacing the
top-level timing method: `duration_s` is the full clean-rerun wall clock and
`run_duration_s` is measured board-command or board-dispatch time.

Spec refs: REQ-HW-3931, SCENARIO-HW-3931.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any

from carnot import experiment_3922_hardware_continuity_consolidated as _base


EXPERIMENT_ID = 3931
SCHEMA = "carnot.hardware_continuity_clean_rerun.v1"
SPEC_REFS = ["REQ-HW-3931", "SCENARIO-HW-3931"]
OUTPUT_REL_PATH = Path("results") / "experiment_3931_hardware_continuity_clean_rerun.json"
RANDOM_SEED = 3931
INFERENCE_SUBSTRATE = "hardware_smoke"

CommandResult = _base.CommandResult
CommandRunner = _base.CommandRunner
Clock = _base.Clock
PolarfireDispatcher = _base.PolarfireDispatcher
GateMateBuilder = _base.GateMateBuilder
GateMateRunCommand = _base.GateMateRunCommand
GateMateClock = _base.GateMateClock
WhichFunc = _base.WhichFunc

POLARFIRE_SSH_PRECONDITION = _base.POLARFIRE_SSH_PRECONDITION
KV260_SSH_PRECONDITION = _base.KV260_SSH_PRECONDITION
KV260_LISTAPPS_COMMAND = _base.KV260_LISTAPPS_COMMAND
KV260_UIO_COMMAND = _base.KV260_UIO_COMMAND

run_command = _base.run_command
run_polarfire_dispatch = _base.run_polarfire_dispatch
run_gatemate_confirmation = _base.run_gatemate_confirmation

BaseBuilder = Callable[..., dict[str, Any]]

REQUIRED_ARTIFACT_FIELDS = (
    "gatemate_reachable",
    "polarfire_reachable",
    "kv260_reachable",
    "gatemate_terminal_state_reached",
    "duration_s",
    "run_duration_s",
    "polarfire_state",
    "kv260_state",
    "fabric_acceleration_claimed",
    "preconditions_checked",
    "reproducibility_checksum",
    "inference_substrate",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "gatemate_reachable": (
        "Per-board reachability: GateMate via JTAG detect, PolarFire and KV260 via SSH."
    ),
    "polarfire_reachable": (
        "Per-board reachability: GateMate via JTAG detect, PolarFire and KV260 via SSH."
    ),
    "kv260_reachable": (
        "Per-board reachability: KV260 is checked through board SSH only, not host storage."
    ),
    "gatemate_terminal_state_reached": (
        "Bare bool: flashed plus smoke plus readback-or-unsupported plus distinct timers."
    ),
    "duration_s": "Full clean-rerun wall-clock timer.",
    "run_duration_s": "Measured board-command or board-dispatch timer, distinct from duration_s.",
    "polarfire_state": "Hash-verified soft-CPU SSH dispatch only, with no fabric claim.",
    "kv260_state": "Loaded overlay plus UIO presence records terminal or non-terminal state.",
    "fabric_acceleration_claimed": "Must remain false for this continuity audit.",
    "preconditions_checked": "Hardware-smoke methodology with real per-board preconditions first.",
    "reproducibility_checksum": "Content hash over the cleaned artifact for drift checks.",
    "inference_substrate": "Declares this as hardware_smoke board interaction.",
    "honest_verdict": "Success or blocked prefix records the continuity falsification gate.",
}

BANNED_ARTIFACT_MARKERS = (
    "/dev/mmcblk",
    "gguf",
    "cuda",
    "torch.cuda",
    ".cuda(",
    "dualgpu",
    "llama.cpp",
    "live model",
)

PRIOR_EXP3922_DIAGNOSIS = (
    "Exp 3922 recorded zero top-level timers on a hardware_smoke artifact; "
    "Exp 3931 records distinct measured timers and keeps the no-fabric claim."
)


class DurationRecordingRunner:
    """Command-runner adapter that preserves measured subprocess durations."""

    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner
        self.results: list[CommandResult] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> CommandResult:
        result = self.runner(command, timeout_s)
        self.results.append(result)
        return result


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
    polarfire_dispatcher: PolarfireDispatcher = run_polarfire_dispatch,
    gatemate_builder: GateMateBuilder = run_gatemate_confirmation,
    gatemate_run_command: GateMateRunCommand = _base._default_gatemate_run_command,
    which_func: WhichFunc = shutil.which,
    clock: Clock = time.perf_counter,
    gatemate_monotonic: GateMateClock = time.monotonic,
    base_builder: BaseBuilder = _base.build_artifact,
    duration_s: float | None = None,
    run_duration_s: float | None = None,
) -> dict[str, Any]:
    """Build the clean rerun artifact from the Exp 3922 per-board audit."""
    root = Path(repo_root)
    started = clock()
    recording_runner = DurationRecordingRunner(command_runner)
    base_artifact = base_builder(
        repo_root=root,
        command_runner=recording_runner,
        polarfire_dispatcher=polarfire_dispatcher,
        gatemate_builder=gatemate_builder,
        gatemate_run_command=gatemate_run_command,
        which_func=which_func,
        clock=clock,
        gatemate_monotonic=gatemate_monotonic,
    )
    measured_duration = duration_s if duration_s is not None else clock() - started
    measured_run_duration = (
        run_duration_s
        if run_duration_s is not None
        else measured_board_duration(base_artifact, recording_runner.results)
    )
    artifact = project_clean_artifact(
        base_artifact,
        duration_s=measured_duration,
        run_duration_s=measured_run_duration,
    )
    validate_artifact(artifact)
    return artifact


def project_clean_artifact(
    base_artifact: dict[str, Any],
    *,
    duration_s: float,
    run_duration_s: float,
) -> dict[str, Any]:
    """Project an Exp 3922-shaped artifact into the clean Exp 3931 schema."""
    artifact = json.loads(json.dumps(base_artifact, sort_keys=True, default=str))
    artifact.pop("flagged_adversarial", None)
    artifact.pop("corrigendum_pending", None)
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment": EXPERIMENT_ID,
            "spec_refs": list(SPEC_REFS),
            "random_seed": RANDOM_SEED,
            "duration_s": _round_timer(duration_s),
            "run_duration_s": _round_timer(run_duration_s),
            "field_principles": dict(FIELD_PRINCIPLES),
            "prior_exp3922_diagnosis": PRIOR_EXP3922_DIAGNOSIS,
            "honest_verdict": clean_honest_verdict(artifact),
            "reproducibility_checksum": "",
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def measured_board_duration(
    base_artifact: dict[str, Any],
    command_results: list[CommandResult],
) -> float:
    """Return measured board-operation time from command results or summaries."""
    command_total = sum(_positive_number(getattr(result, "duration_s", 0.0)) for result in command_results)
    gatemate_run = _summary_duration(base_artifact.get("gatemate_summary"))
    if command_total > 0.0:
        return _round_timer(command_total + gatemate_run)

    fallback_total = gatemate_run
    fallback_total += _summary_duration(base_artifact.get("polarfire_dispatch_summary"))
    transcripts = base_artifact.get("kv260_command_transcripts")
    if isinstance(transcripts, dict):
        for transcript in transcripts.values():
            fallback_total += _summary_duration(transcript)
    return _round_timer(fallback_total)


def clean_honest_verdict(artifact: dict[str, Any]) -> str:
    gatemate_reachable = bool(artifact.get("gatemate_reachable"))
    polarfire_reachable = bool(artifact.get("polarfire_reachable"))
    kv260_reachable = bool(artifact.get("kv260_reachable"))
    if not (gatemate_reachable or polarfire_reachable or kv260_reachable):
        return "blocked_all_boards_unreachable"
    return (
        "success: hardware_continuity_clean_"
        f"gatemate{state_token(str(artifact.get('gatemate_state', 'unknown')))}_"
        f"pf{state_token(str(artifact.get('polarfire_state', 'unknown')))}_"
        f"kv{state_token(str(artifact.get('kv260_state', 'unknown')))}_"
        "distinct_timers_no_fabric_claim"
    )


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema") != SCHEMA:
        raise ValueError("schema must identify the Exp 3931 clean rerun artifact")
    if artifact.get("experiment") != EXPERIMENT_ID:
        raise ValueError("experiment must be 3931")
    if artifact.get("spec_refs") != SPEC_REFS:
        raise ValueError("spec_refs must reference REQ-HW-3931 and SCENARIO-HW-3931")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3931")
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
            raise ValueError(f"{field} must be a bare scalar/list field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be hardware_smoke")
    if artifact.get("fabric_acceleration_claimed") is not False:
        raise ValueError("fabric_acceleration_claimed must be false")
    _validate_verdict_and_timers(artifact)
    _validate_no_forbidden_markers(artifact)
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match artifact content")


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    polarfire_dispatcher: PolarfireDispatcher = run_polarfire_dispatch,
    gatemate_builder: GateMateBuilder = run_gatemate_confirmation,
    gatemate_run_command: GateMateRunCommand = _base._default_gatemate_run_command,
    which_func: WhichFunc = shutil.which,
    clock: Clock = time.perf_counter,
    gatemate_monotonic: GateMateClock = time.monotonic,
    base_builder: BaseBuilder = _base.build_artifact,
    duration_s: float | None = None,
    run_duration_s: float | None = None,
) -> Path:
    artifact = build_artifact(
        repo_root=repo_root,
        command_runner=command_runner,
        polarfire_dispatcher=polarfire_dispatcher,
        gatemate_builder=gatemate_builder,
        gatemate_run_command=gatemate_run_command,
        which_func=which_func,
        clock=clock,
        gatemate_monotonic=gatemate_monotonic,
        base_builder=base_builder,
        duration_s=duration_s,
        run_duration_s=run_duration_s,
    )
    return write_artifact(repo_root, artifact)


def state_token(state: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", state.lower()).strip("_")
    return token or "unknown"


def _validate_verdict_and_timers(artifact: dict[str, Any]) -> None:
    verdict = str(artifact.get("honest_verdict", ""))
    any_reachable = any(
        bool(artifact.get(field))
        for field in ("gatemate_reachable", "polarfire_reachable", "kv260_reachable")
    )
    if any_reachable and not (
        verdict.startswith("success: hardware_continuity_clean_gatemate")
        and verdict.endswith("_distinct_timers_no_fabric_claim")
    ):
        raise ValueError("reachable artifacts must use the clean success prefix")
    if not any_reachable and verdict != "blocked_all_boards_unreachable":
        raise ValueError("all-unreachable artifacts must use blocked_all_boards_unreachable")

    duration = _positive_number(artifact.get("duration_s"))
    run_duration = _positive_number(artifact.get("run_duration_s"))
    if duration <= 0.0 or run_duration <= 0.0:
        raise ValueError("duration_s and run_duration_s must be positive")
    if duration == run_duration:
        raise ValueError("clean rerun requires distinct timers")


def _validate_no_forbidden_markers(artifact: dict[str, Any]) -> None:
    text = json.dumps(artifact, sort_keys=True, default=str).lower()
    for marker in BANNED_ARTIFACT_MARKERS:
        if marker in text:
            raise ValueError(f"artifact contains forbidden substrate marker: {marker}")


def _summary_duration(summary: Any) -> float:
    if isinstance(summary, dict):
        return _positive_number(summary.get("run_duration_s", summary.get("duration_s")))
    return 0.0


def _positive_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number <= 0.0:
        return 0.0
    return number


def _round_timer(value: Any) -> float:
    return round(float(value), 6)


def main() -> None:  # pragma: no cover - CLI wrapper
    _base._gatemate.resolve_toolchain_path()
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"gatemate_reachable: {artifact['gatemate_reachable']}")
    print(f"polarfire_reachable: {artifact['polarfire_reachable']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"duration_s: {artifact['duration_s']}")
    print(f"run_duration_s: {artifact['run_duration_s']}")
    print(f"fabric_acceleration_claimed: {artifact['fabric_acceleration_claimed']}")


if __name__ == "__main__":  # pragma: no cover
    main()
