"""Consolidated PolarFire plus KV260 continuity audit for Exp 3890.

This module keeps the hardware claim deliberately narrow. PolarFire evidence is
only a soft-CPU SSH dispatch that reuses Exp 3867's hash-matched Ising workload.
KV260 evidence is only SSH reachability, `xmutil listapps`, and `/dev/uio*`
presence. Those checks are useful continuity signals, but they do not prove
FPGA compute acceleration, so the artifact always leaves the fabric acceleration
claim false.

Spec refs: REQ-HW-3890, SCENARIO-HW-3890.
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
import tempfile
import time
from typing import Any

from carnot import experiment_3867_polarfire_soc_smoke_v4 as exp3867


EXPERIMENT_ID = 3890
SCHEMA = "carnot.polarfire_kv260_continuity.v1"
SPEC_REFS = ["REQ-HW-3890", "SCENARIO-HW-3890"]
OUTPUT_REL_PATH = Path("results") / "experiment_3890_polarfire_kv260_continuity.json"
RANDOM_SEED = 3890
INFERENCE_SUBSTRATE = "hardware_smoke"

POLARFIRE_SSH_PRECONDITION = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
KV260_SSH_PRECONDITION = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
KV260_LISTAPPS_COMMAND = ("ssh", "kria", "xmutil listapps")
KV260_UIO_COMMAND = ("ssh", "kria", "ls /dev/uio*")

VALID_KV260_OVERLAYS = ("carnot_ising_v2_n64", "carnot_ising_v4", "carnot_ising")

REQUIRED_ARTIFACT_FIELDS = (
    "polarfire_reachable",
    "kv260_reachable",
    "polarfire_state",
    "kv260_state",
    "fabric_acceleration_claimed",
    "preconditions_checked",
    "reproducibility_checksum",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "polarfire_reachable": (
        "Per-board SSH reachability is the honest precondition; a PolarFire miss "
        "does not suppress the KV260 audit."
    ),
    "kv260_reachable": (
        "Per-board SSH reachability is the honest KV260 precondition; host storage "
        "checks are the retired wrong mechanism."
    ),
    "polarfire_state": (
        "Terminal hash-verified dispatch is soft-CPU over SSH only; it is not a "
        "fabric acceleration claim."
    ),
    "kv260_state": (
        "Loaded-overlay plus UIO evidence records terminal or non-terminal KV260 "
        "state for Hardware-Task Continuity."
    ),
    "fabric_acceleration_claimed": (
        "Must remain false because this continuity audit does not demonstrate "
        "compute acceleration on either board."
    ),
    "preconditions_checked": (
        "Hardware-smoke methodology records real SSH board interaction before any "
        "board-specific command."
    ),
    "reproducibility_checksum": (
        "Hardware-smoke methodology gives a content hash so later drift is visible "
        "without pretending live-model inference ran."
    ),
    "inference_substrate": (
        "Hardware-smoke methodology identifies this as SSH board interaction, not "
        "a model-inference substrate."
    ),
}


@dataclass(frozen=True)
class CommandResult:
    """Bounded command transcript used by live runs and synthetic tests."""

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
            "duration_s": round(float(self.duration_s), 6),
            "combined_output": self.combined_output,
        }


CommandRunner = Callable[[tuple[str, ...], float], CommandResult]
Clock = Callable[[], float]
PolarfireDispatcher = Callable[[Path, CommandRunner, Clock], dict[str, Any]]


def command_to_string(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandResult:
    """Run a subprocess with a timeout and keep enough evidence for the artifact."""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - environment timing
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else f"timeout after {timeout_s}s"
        return CommandResult(command, 124, stdout, stderr, time.perf_counter() - started)
    except OSError as exc:  # pragma: no cover - shell availability
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
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_kv260_listapps(text: str) -> str | None:
    """Return the listed Carnot Ising overlay name, if `xmutil` reports one."""
    for overlay in VALID_KV260_OVERLAYS:
        if overlay in text:
            return overlay
    return None


def parse_uio_devices(text: str) -> list[str]:
    """Extract remote `/dev/uioN` paths while preserving first-seen order."""
    devices: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"/dev/uio\d+", text):
        device = match.group(0)
        if device not in seen:
            devices.append(device)
            seen.add(device)
    return devices


def classify_kv260_state(
    *,
    reachable: bool,
    listapps_result: CommandResult | None,
    uio_result: CommandResult | None,
) -> tuple[str, str | None, bool, list[str]]:
    """Classify KV260 continuity from only the allowed SSH board commands."""
    if not reachable:
        return "blocked_kv260_ssh_unreachable", None, False, []

    overlay = (
        parse_kv260_listapps(listapps_result.combined_output)
        if listapps_result is not None and listapps_result.returncode == 0
        else None
    )
    uio_devices = (
        parse_uio_devices(uio_result.combined_output)
        if uio_result is not None and uio_result.returncode == 0
        else []
    )
    carnot_active = overlay is not None and bool(uio_devices)

    if carnot_active:
        state = "terminal_carnot_ising_active_uio_present"
    elif overlay is not None:
        state = "nonterminal_carnot_ising_listed_uio_absent"
    elif uio_devices:
        state = "nonterminal_carnot_ising_inactive_uio_present"
    else:
        state = "nonterminal_carnot_ising_inactive_uio_absent"
    return state, overlay, carnot_active, uio_devices


def run_polarfire_dispatch(
    repo_root: Path,
    command_runner: CommandRunner,
    clock: Clock,
) -> dict[str, Any]:
    """Delegate the PolarFire reconfirmation to Exp 3867's hash verifier.

    Exp 3867 expects its own command-result dataclass. The adapter keeps this
    module's runner injectable while preserving Exp 3867's real SSH/SCP path.
    """

    def exp3867_runner(args: Any, timeout: float | None = None) -> exp3867.CommandResult:
        command = tuple(str(part) for part in args)
        result = command_runner(command, 60.0 if timeout is None else float(timeout))
        return exp3867.CommandResult(
            args=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    with tempfile.TemporaryDirectory(prefix="carnot_exp3890_polarfire_") as tmpdir:
        output_path = Path(tmpdir) / "experiment_3890_polarfire_reconfirm.json"
        return exp3867.run_experiment(
            repo_root=repo_root,
            runner=exp3867_runner,
            clock=clock,
            output_path=output_path,
        )


def summarize_polarfire_dispatch(dispatch_artifact: dict[str, Any] | None) -> tuple[str, Any]:
    if dispatch_artifact is None:
        return "blocked_polarfire_ssh_unreachable", None

    hash_match = bool(dispatch_artifact.get("result_hash_match"))
    validated = bool(dispatch_artifact.get("polarfire_workload_validated"))
    verdict = str(dispatch_artifact.get("honest_verdict", ""))
    if hash_match and validated and verdict.startswith("success:"):
        state = "terminal_hash_verified_soft_cpu_ssh_dispatch"
    elif verdict.startswith("blocked_"):
        state = _state_token(verdict)
    else:
        state = "nonterminal_hash_verification_not_confirmed"

    summary = {
        "exp3867_honest_verdict": verdict,
        "polarfire_workload_validated": validated,
        "result_hash_match": hash_match,
        "board_result_sha256": dispatch_artifact.get("board_result_sha256"),
        "cpu_reference_sha256": dispatch_artifact.get("cpu_reference_sha256"),
        "run_duration_s": dispatch_artifact.get("run_duration_s"),
        "inference_substrate": dispatch_artifact.get("inference_substrate"),
        "claim_boundary": "soft_cpu_ssh_dispatch_no_fpga_fabric_acceleration",
    }
    return state, summary


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    polarfire_dispatcher: PolarfireDispatcher = run_polarfire_dispatch,
    clock: Clock = time.perf_counter,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Run the per-board SSH gates and build the terminal continuity artifact."""
    root = Path(repo_root)
    started = clock()

    polarfire_probe = command_runner(POLARFIRE_SSH_PRECONDITION, 10.0)
    kv260_probe = command_runner(KV260_SSH_PRECONDITION, 10.0)
    polarfire_reachable = polarfire_probe.returncode == 0
    kv260_reachable = kv260_probe.returncode == 0

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

    elapsed = duration_s if duration_s is not None else clock() - started
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _honest_verdict(
            polarfire_reachable=polarfire_reachable,
            kv260_reachable=kv260_reachable,
            polarfire_state=polarfire_state,
            kv260_state=kv260_state,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "polarfire_reachable": polarfire_reachable,
        "kv260_reachable": kv260_reachable,
        "polarfire_state": polarfire_state,
        "kv260_state": kv260_state,
        "fabric_acceleration_claimed": False,
        "preconditions_checked": [
            _precondition_entry("polarfire_ssh", polarfire_probe),
            _precondition_entry("kv260_ssh", kv260_probe),
        ],
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
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(elapsed), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        raise ValueError("field_principles must be a dict")
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(principles)
    if missing_principles:
        raise ValueError(f"field_principles missing required fields: {sorted(missing_principles)}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be hardware_smoke")
    if artifact.get("fabric_acceleration_claimed") is not False:
        raise ValueError("fabric_acceleration_claimed must be false")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (
        verdict.startswith(("success:", "complete:", "failed:", "retired:"))
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must start with a terminal prefix or blocked_")
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
    clock: Clock = time.perf_counter,
    duration_s: float | None = None,
) -> Path:
    artifact = build_artifact(
        repo_root=repo_root,
        command_runner=command_runner,
        polarfire_dispatcher=polarfire_dispatcher,
        clock=clock,
        duration_s=duration_s,
    )
    return write_artifact(repo_root, artifact)


def _observed(result: CommandResult) -> str:
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        return stdout
    if stderr:
        return stderr
    return f"returncode={result.returncode}"


def _precondition_entry(resource: str, result: CommandResult) -> dict[str, Any]:
    return {
        "resource": resource,
        "command": command_to_string(result.command),
        "exit_code": result.returncode,
        "available": result.returncode == 0,
        "observed": _observed(result),
        "checked_before_board_operations": True,
    }


def _state_token(state: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", state.lower()).strip("_")
    return token or "unknown"


def _honest_verdict(
    *,
    polarfire_reachable: bool,
    kv260_reachable: bool,
    polarfire_state: str,
    kv260_state: str,
) -> str:
    if not polarfire_reachable and not kv260_reachable:
        return "blocked_polarfire_and_kv260_ssh_unreachable"
    return (
        "success: polarfire_kv260_continuity_"
        f"pf{_state_token(polarfire_state)}_"
        f"kv{_state_token(kv260_state)}_"
        "no_fabric_claim"
    )


def main() -> None:  # pragma: no cover
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"polarfire_state: {artifact['polarfire_state']}")
    print(f"kv260_state: {artifact['kv260_state']}")
    print(f"fabric_acceleration_claimed: {artifact['fabric_acceleration_claimed']}")


if __name__ == "__main__":  # pragma: no cover
    main()
