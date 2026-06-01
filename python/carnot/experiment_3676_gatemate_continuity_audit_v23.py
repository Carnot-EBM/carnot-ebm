"""Build the Exp 3676 GateMate continuity audit v23 artifact.

Spec refs: REQ-HW-3676, SCENARIO-HW-3676.

This audit records what the host can honestly know without entering the
GateMate flash/smoke path that is already known to hang. A positive JTAG detect
means the board is visible to DirtyJTAG; it is not a host-visible smoke pass.
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

EXPERIMENT_ID = "exp3676"
TASK_ID = "exp3676-gatemate-continuity-audit-v23"
SCHEMA = "carnot.gatemate_continuity_audit.v23"
OUTPUT_REL_PATH = Path("results/experiment_3676_gatemate_continuity_audit_v23.json")
RANDOM_SEED = 3676

OPENFPGALOADER_WHICH_COMMAND = ("bash", "-lc", "command -v openFPGALoader")
GATEMATE_DETECT_COMMAND = (
    "timeout",
    "--kill-after=2s",
    "5s",
    "openFPGALoader",
    "-c",
    "dirtyJtag",
    "--detect",
)

TERMINAL_VERDICT = (
    "complete: "
    "gatemate_continuity_audit_recorded_openfpgaloader_state_"
    "flash_smoke_host_io_hang_known_blocker"
)
HOST_IO_HANG_BLOCKER = "flash/smoke host-IO hang"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "openfpgaloader_installed",
    "gatemate_idcode_detected",
    "known_blocker",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "JTAG detect only; no flash.",
    "openfpgaloader_installed": (
        "Records the root blocker (tool missing) honestly so the audit trail is current."
    ),
    "gatemate_idcode_detected": (
        "Honest detect result (bool or null) -- not a fabricated flash pass."
    ),
    "known_blocker": (
        "Names the missing-tool / flash-smoke host-IO hang so the audit trail stays current."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
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
    """Run one bounded probe and preserve the output that explains the verdict."""
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


def _detect_reports_gatemate_idcode(probe: CommandProbe) -> bool:
    text = probe.combined_output.lower()
    has_idcode = "idcode" in text
    names_gatemate = "gatemate" in text or "gm1a" in text or "colognechip" in text
    return probe.exit_code == 0 and has_idcode and names_gatemate


def _known_blocker(openfpgaloader_installed: bool, detect_probe: CommandProbe | None) -> str:
    if not openfpgaloader_installed:
        return f"openFPGALoader not found on PATH; {HOST_IO_HANG_BLOCKER}"
    if detect_probe is not None and detect_probe.exit_code == 124:
        return f"openFPGALoader detect timed out; {HOST_IO_HANG_BLOCKER}"
    if detect_probe is not None and detect_probe.exit_code != 0:
        return (
            f"openFPGALoader detect failed exit_code={detect_probe.exit_code}; "
            f"{HOST_IO_HANG_BLOCKER}"
        )
    if detect_probe is not None and not _detect_reports_gatemate_idcode(detect_probe):
        return f"openFPGALoader detect did not report a GateMate IDCODE; {HOST_IO_HANG_BLOCKER}"
    return HOST_IO_HANG_BLOCKER


def _preconditions_checked(
    which_probe: CommandProbe,
    detect_probe: CommandProbe | None,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = [
        {
            "resource": "openFPGALoader",
            "command": "command -v openFPGALoader",
            "available": which_probe.exit_code == 0,
            "exit_code": which_probe.exit_code,
            "path": which_probe.stdout.strip() or None,
        }
    ]
    if detect_probe is not None:
        entries.append(
            {
                "resource": "gatemate_jtag_detect",
                "command": command_to_string(GATEMATE_DETECT_COMMAND),
                "available": _detect_reports_gatemate_idcode(detect_probe),
                "exit_code": detect_probe.exit_code,
            }
        )
    return entries


def build_artifact(
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> dict[str, object]:
    """Build the audit artifact from tool discovery and bounded JTAG detect only."""
    started = time.perf_counter()
    which_probe = command_runner(OPENFPGALOADER_WHICH_COMMAND)
    openfpgaloader_installed = which_probe.exit_code == 0

    detect_probe = (
        command_runner(GATEMATE_DETECT_COMMAND) if openfpgaloader_installed else None
    )
    gatemate_idcode_detected = (
        _detect_reports_gatemate_idcode(detect_probe) if detect_probe is not None else None
    )
    elapsed = duration_s if duration_s is not None else max(time.perf_counter() - started, 0.0001)

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": "hardware_smoke",
        "preconditions_checked": _preconditions_checked(which_probe, detect_probe),
        "openfpgaloader_installed": openfpgaloader_installed,
        "openfpgaloader_path": which_probe.stdout.strip() or None,
        "gatemate_idcode_detected": gatemate_idcode_detected,
        "known_blocker": _known_blocker(openfpgaloader_installed, detect_probe),
        "continuity_history": {
            "current_milestone": "2026.06.336",
            "scope": "JTAG detect only; flash/smoke host-IO path intentionally not run.",
        },
        "command_probes": {
            "openfpgaloader_which": which_probe.as_dict(),
            "gatemate_jtag_detect": detect_probe.as_dict() if detect_probe is not None else None,
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
