"""Build the Exp 3730 KV260 opportunistic terminal-state audit artifact.

Spec refs: REQ-HW-3730, SCENARIO-HW-3730.

This audit is intentionally small. Exp 3709 already captured the board-latency
terminal transcript, so this module only asks whether the board is still
reachable over SSH and whether the Carnot accelerator overlay is still listed
by the board. It does not rerun the sampler or make any speedup claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

EXPERIMENT_ID = "exp3730"
TASK_ID = "exp3730-kv260-opportunistic-continuity-audit"
SCHEMA = "carnot.kv260_opportunistic_continuity_audit.v1"
OUTPUT_REL_PATH = Path("results/experiment_3730_kv260_opportunistic_continuity_audit.json")
RANDOM_SEED = 3730

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

VALID_KV260_OVERLAYS = ("carnot_ising_v4", "carnot_ising_v2_n64", "carnot_ising")

SUCCESS_VERDICT = (
    "complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"
)
BLOCKED_VERDICT = "blocked_kv260_ssh_unreachable"
REGRESSION_VERDICT = (
    "complete: kv260_terminal_state_regressed_accelerator_not_listed_opportunistic_audit"
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix; blocked_kv260_ssh_unreachable if the board is unreachable."
    ),
    "inference_substrate": "An SSH board check, not live inference.",
    "kv260_ssh_reachable": "The SSH-reachability fact, not host SD-card presence.",
    "terminal_state_holds": (
        "Confirms the .340 terminal state did not regress; opportunistic, not mandated."
    ),
    "preconditions_checked": ("Records the SSH check was actually run before any board operation."),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


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


def command_to_string(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandProbe:
    """Run one bounded subprocess and keep enough transcript for audit."""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else f"timeout after {timeout_s}s"
        return CommandProbe(
            command=command,
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            duration_s=time.perf_counter() - started,
        )
    except OSError as exc:  # pragma: no cover
        return CommandProbe(
            command=command,
            exit_code=127,
            stderr=str(exc),
            duration_s=time.perf_counter() - started,
        )
    return CommandProbe(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_s=time.perf_counter() - started,
    )


def sha256_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def detect_carnot_overlay(text: str) -> str | None:
    for overlay in VALID_KV260_OVERLAYS:
        if overlay in text:
            return overlay
    return None


def xmutil_requires_root(probe: CommandProbe) -> bool:
    lowered = probe.combined_output.lower()
    return "root privileges" in lowered or "using 'sudo'" in lowered


def _precondition_entry(probe: CommandProbe) -> dict[str, object]:
    return {
        "resource": "kv260_ssh",
        "command": command_to_string(KV260_SSH_COMMAND),
        "available": probe.exit_code == 0,
        "exit_code": probe.exit_code,
        "checked_before_board_operations": True,
    }


def _honest_verdict(kv260_ssh_reachable: bool, terminal_state_holds: bool) -> str:
    if not kv260_ssh_reachable:
        return BLOCKED_VERDICT
    if terminal_state_holds:
        return SUCCESS_VERDICT
    return REGRESSION_VERDICT


def _operator_regression_note(
    kv260_ssh_reachable: bool,
    terminal_state_holds: bool,
) -> str:
    if not kv260_ssh_reachable:
        return "kv260_ssh_unreachable"
    if not terminal_state_holds:
        return "carnot_overlay_not_listed"
    return "none"


def validate_artifact(payload: dict[str, object]) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(payload)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")


def build_artifact(
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> dict[str, object]:
    """Build the audit artifact from SSH reachability and board overlay listing."""
    started = time.perf_counter()
    kv260_ssh_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    kv260_ssh_reachable = kv260_ssh_probe.exit_code == 0
    kv260_listapps_probe: CommandProbe | None = None
    kv260_listapps_sudo_probe: CommandProbe | None = None
    kv260_overlay_name: str | None = None

    if kv260_ssh_reachable:
        kv260_listapps_probe = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
        if kv260_listapps_probe.exit_code == 0:
            kv260_overlay_name = detect_carnot_overlay(kv260_listapps_probe.combined_output)
        if kv260_overlay_name is None and xmutil_requires_root(kv260_listapps_probe):
            kv260_listapps_sudo_probe = command_runner(KV260_LISTAPPS_SUDO_COMMAND, 30.0)
            if kv260_listapps_sudo_probe.exit_code == 0:
                kv260_overlay_name = detect_carnot_overlay(
                    kv260_listapps_sudo_probe.combined_output
                )

    kv260_overlay_loadable = kv260_overlay_name is not None
    terminal_state_holds = kv260_ssh_reachable and kv260_overlay_loadable
    raw_elapsed = duration_s if duration_s is not None else time.perf_counter() - started
    elapsed = round(max(float(raw_elapsed), 0.0001), 4)

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "honest_verdict": _honest_verdict(
            kv260_ssh_reachable,
            terminal_state_holds,
        ),
        "inference_substrate": "hardware_smoke",
        "kv260_ssh_reachable": kv260_ssh_reachable,
        "terminal_state_holds": terminal_state_holds,
        "preconditions_checked": [_precondition_entry(kv260_ssh_probe)],
        "kv260_overlay_loadable": kv260_overlay_loadable,
        "kv260_overlay_name": kv260_overlay_name,
        "operator_regression_note": _operator_regression_note(
            kv260_ssh_reachable,
            terminal_state_holds,
        ),
        "terminal_anchor_experiment": "exp3709",
        "mandate_status": "opportunistic_not_mandated",
        "latency_harness_rerun": False,
        "speedup_claim_made": False,
        "command_probes": {
            "kv260_ssh": kv260_ssh_probe.as_dict(),
            "kv260_xmutil_listapps": (
                kv260_listapps_probe.as_dict() if kv260_listapps_probe is not None else None
            ),
            "kv260_xmutil_listapps_sudo": (
                kv260_listapps_sudo_probe.as_dict()
                if kv260_listapps_sudo_probe is not None
                else None
            ),
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed,
    }
    payload["reproducibility_checksum"] = sha256_payload(payload)
    validate_artifact(payload)
    return payload


def write_artifact(repo_root: str | Path, payload: dict[str, object]) -> Path:
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def run_experiment(
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> Path:
    payload = build_artifact(command_runner=command_runner, duration_s=duration_s)
    return write_artifact(repo_root, payload)


def main() -> None:  # pragma: no cover
    run_experiment(Path("."))


if __name__ == "__main__":  # pragma: no cover
    main()
