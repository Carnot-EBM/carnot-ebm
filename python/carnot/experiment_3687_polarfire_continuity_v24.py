"""Build the Exp 3687 PolarFire continuity v24 artifact.

Spec refs: REQ-HW-3687, SCENARIO-HW-3687.
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

EXPERIMENT_ID = "exp3687"
TASK_ID = "exp3687-polarfire-continuity-v24"
SCHEMA = "carnot.polarfire_continuity.v24"
OUTPUT_REL_PATH = Path("results/experiment_3687_polarfire_continuity_v24.json")
RANDOM_SEED = 3687

POLARFIRE_SSH_COMMAND = ("ssh", "-o", "ConnectTimeout=5", "polarfire", "true")
POLARFIRE_UPTIME_COMMAND = ("ssh", "polarfire", "uptime")
POLARFIRE_DISPATCH_COMMAND = ("ssh", "polarfire", "which carnot")

REACHABLE_VERDICT = "complete: polarfire_continuity_confirmed_reachable"
BLOCKED_VERDICT = "complete: blocked_polarfire_ssh_timeout"

CURRENT_MILESTONE = ".337"
LAST_RECORDED_REACHABLE_MILESTONE = ".331"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "polarfire_ssh_reachable",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "SSH-attached board test.",
    "preconditions_checked": ("Records the SSH-reachability check before claiming continuity."),
    "polarfire_ssh_reachable": "Honest board state.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
    "polarfire_uptime": (
        "Live board continuity value collected only after SSH reachability passes."
    ),
    "polarfire_carnot_dispatch_path": (
        "Live Carnot dispatch-path value collected only after SSH reachability passes."
    ),
    "polarfire_continuity_state": (
        "Deflagged summary derived from the distinct PolarFire uptime and dispatch fields."
    ),
}


@dataclass(frozen=True)
class CommandProbe:
    command: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_dict(self) -> dict[str, object]:
        return {
            "command": command_to_string(self.command),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "combined_output": self.combined_output,
        }


CommandRunner = Callable[[tuple[str, ...]], CommandProbe]


def command_to_string(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def run_command(command: tuple[str, ...]) -> CommandProbe:
    """Run one command and preserve enough transcript to audit board continuity."""
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return CommandProbe(command=command, exit_code=127, stderr=str(exc))
    return CommandProbe(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def sha256_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _precondition_entry(ssh_probe: CommandProbe) -> dict[str, object]:
    return {
        "resource": "polarfire_ssh",
        "command": command_to_string(POLARFIRE_SSH_COMMAND),
        "available": ssh_probe.exit_code == 0,
        "exit_code": ssh_probe.exit_code,
    }


def _uptime_value(probe: CommandProbe | None) -> str | None:
    if probe is None:
        return None
    if probe.exit_code != 0:
        return "unknown"
    stripped = probe.stdout.strip()
    return stripped or "unknown"


def _dispatch_value(probe: CommandProbe | None) -> str | None:
    if probe is None:
        return None
    if probe.exit_code != 0:
        return "not_found"
    stripped = probe.stdout.strip()
    return stripped or "not_found"


def _continuity_state(
    ssh_reachable: bool,
    uptime_probe: CommandProbe | None,
    dispatch_probe: CommandProbe | None,
) -> str:
    if not ssh_reachable:
        return "blocked_ssh_timeout"
    if (
        uptime_probe is not None
        and dispatch_probe is not None
        and uptime_probe.exit_code == 0
        and dispatch_probe.exit_code == 0
    ):
        return "reachable_uptime_and_dispatch_path_recorded_deflagged"
    return "reachable_probe_values_incomplete"


def build_artifact(
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> dict[str, object]:
    """Build a continuity artifact from the SSH precondition and board probes."""
    started = time.perf_counter()
    ssh_probe = command_runner(POLARFIRE_SSH_COMMAND)
    ssh_reachable = ssh_probe.exit_code == 0

    uptime_probe = command_runner(POLARFIRE_UPTIME_COMMAND) if ssh_reachable else None
    dispatch_probe = command_runner(POLARFIRE_DISPATCH_COMMAND) if ssh_reachable else None
    elapsed = duration_s if duration_s is not None else max(time.perf_counter() - started, 0.0001)

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "honest_verdict": REACHABLE_VERDICT if ssh_reachable else BLOCKED_VERDICT,
        "inference_substrate": "hardware_smoke",
        "preconditions_checked": [_precondition_entry(ssh_probe)],
        "polarfire_ssh_reachable": ssh_reachable,
        "polarfire_uptime": _uptime_value(uptime_probe),
        "polarfire_carnot_dispatch_path": _dispatch_value(dispatch_probe),
        "polarfire_continuity_state": _continuity_state(
            ssh_reachable,
            uptime_probe,
            dispatch_probe,
        ),
        "continuity_history": {
            "current_milestone": CURRENT_MILESTONE,
            "last_recorded_reachable_milestone": LAST_RECORDED_REACHABLE_MILESTONE,
        },
        "command_probes": {
            "polarfire_ssh": ssh_probe.as_dict(),
            "polarfire_uptime": uptime_probe.as_dict() if uptime_probe is not None else None,
            "polarfire_carnot_dispatch_path": (
                dispatch_probe.as_dict() if dispatch_probe is not None else None
            ),
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(elapsed), 4),
    }
    payload["reproducibility_checksum"] = sha256_payload(payload)
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
