"""Consolidated GateMate, PolarFire, and KV260 continuity audit for Exp 3922.

The artifact checks each attached board independently. GateMate evidence is the
Exp 3900 n=16 flash confirmation path with distinct timers. PolarFire evidence
is the Exp 3867 hash-verified soft-CPU SSH dispatch. KV260 evidence is only SSH
reachability, `xmutil listapps`, and `/dev/uio*` presence. None of those checks
prove FPGA compute acceleration, so the fabric acceleration claim is always
false.

Spec refs: REQ-HW-3922, SCENARIO-HW-3922.
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

from carnot import experiment_3900_gatemate_terminal_confirmation as _gatemate
from carnot import experiment_3901_polarfire_kv260_continuity as _pfkv
from carnot.experiment_3866_gatemate_ising_tile_flash_v2 import (
    ClockFunc as GateMateClock,
    CommandResult as GateMateCommandResult,
    RunCommand as GateMateRunCommand,
    WhichFunc,
    _command_text as _gatemate_command_text,
    _default_run_command as _default_gatemate_run_command,
)


EXPERIMENT_ID = 3922
SCHEMA = "carnot.hardware_continuity_consolidated.v1"
SPEC_REFS = ["REQ-HW-3922", "SCENARIO-HW-3922"]
OUTPUT_REL_PATH = Path("results") / "experiment_3922_hardware_continuity_consolidated.json"
RANDOM_SEED = 3922
INFERENCE_SUBSTRATE = "hardware_smoke"

GATEMATE_TOOLS = ("nextpnr-himbaechel", "yosys", "openFPGALoader")
GATEMATE_TOOLCHAIN_COMMAND = (
    "command -v nextpnr-himbaechel && command -v yosys && command -v openFPGALoader"
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")

CommandResult = _pfkv.CommandResult
CommandRunner = _pfkv.CommandRunner
Clock = _pfkv.Clock
PolarfireDispatcher = _pfkv.PolarfireDispatcher

POLARFIRE_SSH_PRECONDITION = _pfkv.POLARFIRE_SSH_PRECONDITION
KV260_SSH_PRECONDITION = _pfkv.KV260_SSH_PRECONDITION
KV260_LISTAPPS_COMMAND = _pfkv.KV260_LISTAPPS_COMMAND
KV260_UIO_COMMAND = _pfkv.KV260_UIO_COMMAND

command_to_string = _pfkv.command_to_string
run_command = _pfkv.run_command
run_polarfire_dispatch = _pfkv.run_polarfire_dispatch
summarize_polarfire_dispatch = _pfkv.summarize_polarfire_dispatch
classify_kv260_state = _pfkv.classify_kv260_state

GateMateBuilder = Callable[
    [Path, GateMateRunCommand, WhichFunc, GateMateClock],
    dict[str, Any],
]

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
        "Per-board reachability -- GateMate via JTAG detect, the others via SSH "
        "(KV260 never via host SD card)."
    ),
    "polarfire_reachable": (
        "Per-board reachability -- GateMate via JTAG detect, the others via SSH "
        "(KV260 never via host SD card)."
    ),
    "kv260_reachable": (
        "Per-board reachability -- GateMate via JTAG detect, the others via SSH "
        "(KV260 never via host SD card)."
    ),
    "gatemate_terminal_state_reached": (
        "BARE BOOL -- flashed + smoke + readback-or-unsupported + distinct timers "
        "(no tautology); the graduation signal."
    ),
    "duration_s": (
        "GateMate timers MUST be distinct (the exp3866 TAUTOLOGY corrigendum)."
    ),
    "run_duration_s": (
        "GateMate timers MUST be distinct (the exp3866 TAUTOLOGY corrigendum)."
    ),
    "polarfire_state": (
        "Terminal hash-verified dispatch, soft-CPU only -- NO fabric-acceleration claim."
    ),
    "kv260_state": "Loaded overlay + UIO presence -- terminal/non-terminal record.",
    "fabric_acceleration_claimed": (
        "MUST be false -- neither board demonstrates compute acceleration; the "
        "honest narrowing (north-star section 3)."
    ),
    "preconditions_checked": (
        "Hardware-smoke methodology -- real board interaction; no GGUF/CUDA markers."
    ),
    "reproducibility_checksum": (
        "Hardware-smoke methodology -- real board interaction; no GGUF/CUDA markers."
    ),
    "inference_substrate": (
        "Hardware-smoke methodology -- real board interaction; no GGUF/CUDA markers."
    ),
    "honest_verdict": (
        "Terminal prefix or blocked_resource prefix records the continuity gate."
    ),
}


def run_gatemate_confirmation(
    repo_root: Path,
    run_command: GateMateRunCommand,
    which_func: WhichFunc,
    monotonic: GateMateClock,
) -> dict[str, Any]:  # pragma: no cover - live hardware delegate
    return _gatemate.build_artifact(
        repo_root=repo_root,
        run_command=run_command,
        which_func=which_func,
        monotonic=monotonic,
    )


def payload_checksum(payload: dict[str, Any]) -> str:
    stable = dict(payload)
    stable.pop("reproducibility_checksum", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def check_gatemate_preconditions(
    *,
    run_command: GateMateRunCommand,
    which_func: WhichFunc,
) -> tuple[bool, str, list[dict[str, Any]]]:
    tool_paths = {tool: which_func(tool) or "" for tool in GATEMATE_TOOLS}
    toolchain_available = all(bool(path) for path in tool_paths.values())
    toolchain = {
        "resource": "gatemate_toolchain",
        "command": GATEMATE_TOOLCHAIN_COMMAND,
        "available": toolchain_available,
        "blocker": "" if toolchain_available else "blocked_gatemate_toolchain_missing",
        "detail": tool_paths,
        "checked_before_board_operations": True,
    }
    if not toolchain_available:
        detect = {
            "resource": "gatemate_board_detect",
            "command": "openFPGALoader -c dirtyJtag --detect",
            "available": False,
            "blocker": "blocked_gatemate_toolchain_missing",
            "exit_code": None,
            "observed": "skipped: GateMate toolchain precondition failed",
            "checked_before_board_operations": True,
        }
        return False, "blocked_gatemate_toolchain_missing", [toolchain, detect]

    loader_path = tool_paths["openFPGALoader"]
    detect_result = run_command([loader_path, "-c", "dirtyJtag", "--detect"], 30.0)
    detected = _detects_gatemate(detect_result)
    detect = {
        "resource": "gatemate_board_detect",
        "command": "openFPGALoader -c dirtyJtag --detect",
        "available": detected,
        "blocker": "" if detected else "blocked_gatemate_board_unreachable",
        "exit_code": detect_result.returncode,
        "observed": _observed_gatemate(detect_result),
        "checked_before_board_operations": True,
    }
    if not detected:
        return False, "blocked_gatemate_board_unreachable", [toolchain, detect]
    return True, "", [toolchain, detect]


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    polarfire_dispatcher: PolarfireDispatcher = run_polarfire_dispatch,
    gatemate_builder: GateMateBuilder = run_gatemate_confirmation,
    gatemate_run_command: GateMateRunCommand = _default_gatemate_run_command,
    which_func: WhichFunc = shutil.which,
    clock: Clock = time.perf_counter,
    gatemate_monotonic: GateMateClock = time.monotonic,
    audit_duration_s: float | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    started = clock()

    gatemate_reachable, gatemate_blocker, gatemate_preconditions = check_gatemate_preconditions(
        run_command=gatemate_run_command,
        which_func=which_func,
    )
    polarfire_probe = command_runner(POLARFIRE_SSH_PRECONDITION, 10.0)
    kv260_probe = command_runner(KV260_SSH_PRECONDITION, 10.0)
    polarfire_reachable = polarfire_probe.returncode == 0
    kv260_reachable = kv260_probe.returncode == 0

    gatemate_artifact: dict[str, Any] | None = None
    if gatemate_reachable:
        gatemate_artifact = gatemate_builder(
            root,
            gatemate_run_command,
            which_func,
            gatemate_monotonic,
        )
    (
        gatemate_state,
        gatemate_summary,
        gatemate_terminal,
        gatemate_duration_s,
        gatemate_run_duration_s,
    ) = summarize_gatemate(gatemate_reachable, gatemate_blocker, gatemate_artifact)

    dispatch_artifact: dict[str, Any] | None = None
    if polarfire_reachable:
        dispatch_artifact = polarfire_dispatcher(root, command_runner, clock)
    polarfire_state, polarfire_summary = summarize_polarfire_dispatch(dispatch_artifact)

    kv260_listapps_result: CommandResult | None = None
    kv260_uio_result: CommandResult | None = None
    if kv260_reachable:
        kv260_listapps_result = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
        kv260_uio_result = command_runner(KV260_UIO_COMMAND, 10.0)
    kv260_state, kv260_overlay, kv260_active, kv260_uio_devices = classify_kv260_state(
        reachable=kv260_reachable,
        listapps_result=kv260_listapps_result,
        uio_result=kv260_uio_result,
    )

    elapsed = audit_duration_s if audit_duration_s is not None else clock() - started
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _honest_verdict(
            gatemate_reachable=gatemate_reachable,
            polarfire_reachable=polarfire_reachable,
            kv260_reachable=kv260_reachable,
            gatemate_state=gatemate_state,
            polarfire_state=polarfire_state,
            kv260_state=kv260_state,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "gatemate_reachable": gatemate_reachable,
        "polarfire_reachable": polarfire_reachable,
        "kv260_reachable": kv260_reachable,
        "gatemate_state": gatemate_state,
        "gatemate_terminal_state_reached": gatemate_terminal,
        "duration_s": gatemate_duration_s,
        "run_duration_s": gatemate_run_duration_s,
        "polarfire_state": polarfire_state,
        "kv260_state": kv260_state,
        "fabric_acceleration_claimed": False,
        "preconditions_checked": [
            *gatemate_preconditions,
            _ssh_precondition_entry("polarfire_ssh", polarfire_probe),
            _ssh_precondition_entry("kv260_ssh", kv260_probe),
        ],
        "gatemate_summary": gatemate_summary,
        "polarfire_dispatch_summary": polarfire_summary,
        "kv260_loaded_overlay": kv260_overlay,
        "kv260_carnot_ising_active": kv260_active,
        "kv260_uio_devices": kv260_uio_devices,
        "kv260_command_transcripts": {
            "xmutil_listapps": (
                kv260_listapps_result.as_dict() if kv260_listapps_result is not None else None
            ),
            "uio_list": kv260_uio_result.as_dict() if kv260_uio_result is not None else None,
        },
        "audit_duration_s": round(float(elapsed), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def summarize_gatemate(
    reachable: bool,
    blocker: str,
    artifact: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None, bool, float, float]:
    if not reachable:
        return blocker, None, False, 0.0, 0.0
    if artifact is None:
        return "nonterminal_missing_gatemate_confirmation", None, False, 0.0, 0.0

    duration_s = float(artifact.get("duration_s", 0.0))
    run_duration_s = float(artifact.get("run_duration_s", 0.0))
    no_tautology = duration_s != run_duration_s
    terminal = bool(artifact.get("terminal_state_reached")) and no_tautology
    verdict = str(artifact.get("honest_verdict", ""))
    if terminal:
        state = "terminal_reached"
    elif verdict.startswith("blocked_"):
        state = _state_token(verdict)
    elif not no_tautology:
        state = "nonterminal_timer_tautology"
    elif artifact.get("gatemate_bitstream_flashed"):
        state = "nonterminal_flashed_readback_inconclusive"
    else:
        state = "nonterminal_gate_smoke_incomplete"

    summary = {
        "exp3900_honest_verdict": verdict,
        "gatemate_bitstream_flashed": bool(artifact.get("gatemate_bitstream_flashed")),
        "smoke_ok": bool(artifact.get("smoke_ok")),
        "readback_supported": bool(artifact.get("readback_supported")),
        "readback_verified": bool(artifact.get("readback_verified")),
        "terminal_state_reached": terminal,
        "no_tautology": no_tautology,
        "duration_s": duration_s,
        "run_duration_s": run_duration_s,
        "reproducibility_checksum": artifact.get("reproducibility_checksum"),
    }
    return state, summary, terminal, duration_s, run_duration_s


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema") != SCHEMA:
        raise ValueError("schema must identify the Exp 3922 continuity artifact")
    if artifact.get("experiment") != EXPERIMENT_ID:
        raise ValueError("experiment must be 3922")
    if artifact.get("spec_refs") != SPEC_REFS:
        raise ValueError("spec_refs must reference REQ-HW-3922 and SCENARIO-HW-3922")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3922")
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
    verdict = str(artifact.get("honest_verdict", ""))
    any_reachable = any(
        bool(artifact.get(field))
        for field in ("gatemate_reachable", "polarfire_reachable", "kv260_reachable")
    )
    if any_reachable and not (
        verdict.startswith("success: hardware_continuity_gatemate")
        and verdict.endswith("_no_fabric_claim")
    ):
        raise ValueError("reachable continuity artifacts must use the success hardware prefix")
    if not any_reachable and verdict != "blocked_all_boards_unreachable":
        raise ValueError("all-unreachable artifacts must use blocked_all_boards_unreachable")
    if artifact.get("gatemate_terminal_state_reached") and (
        artifact.get("duration_s") == artifact.get("run_duration_s")
    ):
        raise ValueError("gatemate terminal state requires distinct timers")
    banned_text = json.dumps(artifact, sort_keys=True).lower()
    for marker in ("/dev/mmcblk", "cuda_device", "gguf_model"):
        if marker in banned_text:
            raise ValueError(f"artifact contains retired or non-hardware marker: {marker}")
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
    gatemate_run_command: GateMateRunCommand = _default_gatemate_run_command,
    which_func: WhichFunc = shutil.which,
    clock: Clock = time.perf_counter,
    gatemate_monotonic: GateMateClock = time.monotonic,
    audit_duration_s: float | None = None,
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
        audit_duration_s=audit_duration_s,
    )
    return write_artifact(repo_root, artifact)


def _detects_gatemate(result: GateMateCommandResult) -> bool:
    text = _gatemate_command_text(result).lower()
    return (
        result.returncode == 0
        and "idcode" in text
        and ("colognechip" in text or "gatemate" in text or "gm1a" in text)
    )


def _observed_gatemate(result: GateMateCommandResult) -> str:
    text = _gatemate_command_text(result).strip()
    return _excerpt(text) if text else f"returncode={result.returncode}"


def _observed(result: CommandResult) -> str:
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        return stdout
    if stderr:
        return stderr
    return f"returncode={result.returncode}"


def _ssh_precondition_entry(resource: str, result: CommandResult) -> dict[str, Any]:
    return {
        "resource": resource,
        "command": command_to_string(result.command),
        "exit_code": result.returncode,
        "available": result.returncode == 0,
        "observed": _observed(result),
        "checked_before_board_operations": True,
    }


def _excerpt(text: str, limit: int = 500) -> str:
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _state_token(state: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", state.lower()).strip("_")
    return token or "unknown"


def _honest_verdict(
    *,
    gatemate_reachable: bool,
    polarfire_reachable: bool,
    kv260_reachable: bool,
    gatemate_state: str,
    polarfire_state: str,
    kv260_state: str,
) -> str:
    if not (gatemate_reachable or polarfire_reachable or kv260_reachable):
        return "blocked_all_boards_unreachable"
    return (
        "success: hardware_continuity_"
        f"gatemate{_state_token(gatemate_state)}_"
        f"pf{_state_token(polarfire_state)}_"
        f"kv{_state_token(kv260_state)}_"
        "no_fabric_claim"
    )


def main() -> None:  # pragma: no cover - CLI wrapper
    _gatemate.resolve_toolchain_path()
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"gatemate_reachable: {artifact['gatemate_reachable']}")
    print(f"polarfire_reachable: {artifact['polarfire_reachable']}")
    print(f"kv260_reachable: {artifact['kv260_reachable']}")
    print(f"gatemate_state: {artifact['gatemate_state']}")
    print(f"polarfire_state: {artifact['polarfire_state']}")
    print(f"kv260_state: {artifact['kv260_state']}")
    print(f"fabric_acceleration_claimed: {artifact['fabric_acceleration_claimed']}")


if __name__ == "__main__":  # pragma: no cover
    main()
