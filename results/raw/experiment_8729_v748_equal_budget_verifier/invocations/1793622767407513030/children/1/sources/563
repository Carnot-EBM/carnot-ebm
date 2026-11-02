"""Build the Exp 3686 KV260 continuity v24 artifact.

Spec refs: REQ-HW-3686, SCENARIO-HW-3686.
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

EXPERIMENT_ID = "exp3686"
TASK_ID = "exp3686-kv260-continuity-v24"
SCHEMA = "carnot.kv260_continuity.v24"
OUTPUT_REL_PATH = Path("results/experiment_3686_kv260_continuity_v24.json")
RANDOM_SEED = 3686

KV260_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
KV260_OVERLAY_COMMAND = ("ssh", "kria", "xmutil listapps")

PREVIOUS_UNREACHABLE_MILESTONES = (".331", ".332", ".333", ".334", ".335", ".336")
CURRENT_MILESTONE = ".337"
BLOCKED_STREAK_COUNT = 7

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "kv260_ssh_reachable",
    "kv260_overlay_loaded",
    "consecutive_unreachable_milestones",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "SSH-attached board test; per-board duration floor.",
    "preconditions_checked": (
        "Records the SSH-reachability check -- the correct KV260 precondition, "
        "not host SD card."
    ),
    "kv260_ssh_reachable": (
        "The honest board state; an unreachable board is a blocked_*, "
        "not a fabricated pass."
    ),
    "kv260_overlay_loaded": "Continuity state if reachable.",
    "consecutive_unreachable_milestones": (
        "Tracks the .331->.336 unreachable streak so a persistent outage "
        "surfaces as an operator-action item."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}

REACHABLE_VERDICT = "complete: kv260_continuity_confirmed_reachable"
BLOCKED_VERDICT = "complete: blocked_kv260_ssh_unreachable"


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
    """Run one command and preserve its transcript for later audit."""
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


def _overlay_value(probe: CommandProbe | None) -> str | None:
    if probe is None or probe.exit_code != 0:
        return None
    stripped = probe.stdout.strip()
    return stripped or None


def _precondition_entry(ssh_probe: CommandProbe) -> dict[str, object]:
    return {
        "resource": "kv260_ssh",
        "command": command_to_string(KV260_SSH_COMMAND),
        "available": ssh_probe.exit_code == 0,
        "exit_code": ssh_probe.exit_code,
    }


def _operator_action_item(ssh_reachable: bool) -> str:
    if ssh_reachable:
        return "none"
    return (
        "KV260 SSH is unreachable for milestones .331, .332, .333, .334, .335, "
        ".336, and .337; operator action required after 7 consecutive milestones "
        "to restore kria/kv260.local DNS, network, or SSH access before the next "
        "hardware continuity task."
    )


def _continuity_state(ssh_reachable: bool, overlay_probe: CommandProbe | None) -> str:
    if not ssh_reachable:
        return "blocked_ssh_unreachable"
    if overlay_probe is None:
        return "reachable_overlay_not_checked"
    if overlay_probe.exit_code == 0:
        return "reachable_overlay_state_recorded"
    return "reachable_overlay_query_failed"


def build_artifact(
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> dict[str, object]:
    """Build the continuity artifact from SSH state, not local removable media."""
    started = time.perf_counter()
    ssh_probe = command_runner(KV260_SSH_COMMAND)
    ssh_reachable = ssh_probe.exit_code == 0

    overlay_probe = command_runner(KV260_OVERLAY_COMMAND) if ssh_reachable else None
    elapsed = duration_s if duration_s is not None else max(time.perf_counter() - started, 0.0001)
    consecutive_unreachable = 0 if ssh_reachable else BLOCKED_STREAK_COUNT

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "honest_verdict": REACHABLE_VERDICT if ssh_reachable else BLOCKED_VERDICT,
        "inference_substrate": "hardware_smoke",
        "preconditions_checked": [_precondition_entry(ssh_probe)],
        "kv260_ssh_reachable": ssh_reachable,
        "kv260_overlay_loaded": _overlay_value(overlay_probe),
        "kv260_continuity_state": _continuity_state(ssh_reachable, overlay_probe),
        "consecutive_unreachable_milestones": consecutive_unreachable,
        "operator_action_required": not ssh_reachable,
        "operator_action_item": _operator_action_item(ssh_reachable),
        "continuity_history": {
            "previous_unreachable_milestones": list(PREVIOUS_UNREACHABLE_MILESTONES),
            "current_milestone": CURRENT_MILESTONE,
            "current_unreachable_streak_if_blocked": BLOCKED_STREAK_COUNT,
        },
        "retired_precondition_policy": (
            "Host SD-card device-node checks were not used; KV260 precondition is SSH only."
        ),
        "command_probes": {
            "kv260_ssh": ssh_probe.as_dict(),
            "kv260_xmutil_listapps": (
                overlay_probe.as_dict() if overlay_probe is not None else None
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
