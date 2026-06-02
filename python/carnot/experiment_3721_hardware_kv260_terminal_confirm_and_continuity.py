"""Build the Exp 3721 consolidated hardware-continuity artifact.

Spec refs: REQ-HW-3721, SCENARIO-HW-3721.

The experiment records what the current host can honestly verify. KV260
terminal state requires both a live SSH overlay check and the prior Exp 3709
board-latency transcript on disk; PolarFire and GateMate are opportunistic
continuity checks that should not block the milestone when unavailable.
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
from typing import Any

EXPERIMENT_ID = "exp3721"
TASK_ID = "exp3721-hardware-kv260-terminal-confirm-and-continuity"
SCHEMA = "carnot.hardware_kv260_terminal_confirm_and_continuity.v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json"
)
EXP3709_REL_PATH = Path(
    "results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"
)
RANDOM_SEED = 3721

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
POLARFIRE_SSH_COMMAND = ("ssh", "-o", "ConnectTimeout=5", "polarfire", "true")
POLARFIRE_UPTIME_COMMAND = ("ssh", "polarfire", "uptime")
POLARFIRE_DISPATCH_COMMAND = ("ssh", "polarfire", "which carnot")
GATEMATE_OPENFPGALOADER_COMMAND = ("bash", "-lc", "command -v openFPGALoader")

VALID_KV260_OVERLAYS = ("carnot_ising_v4", "carnot_ising_v2_n64", "carnot_ising")

KV260_TERMINAL_VERDICT = (
    "complete: "
    "kv260_terminal_confirmed_mandate_lift_recommended_polarfire_gatemate_audited"
)
KV260_UNREACHABLE_VERDICT = "complete: kv260_unreachable_polarfire_gatemate_audited"
PARTIAL_VERDICT = "complete: hardware_continuity_partial_recorded"
TERMINAL_VERDICTS = {
    KV260_TERMINAL_VERDICT,
    KV260_UNREACHABLE_VERDICT,
    PARTIAL_VERDICT,
}

MANDATE_LIFT_RECOMMENDATION = (
    "recommend_operator_lift_per_milestone_kv260_mandate"
)
NO_LIFT_KV260_UNREACHABLE_RECOMMENDATION = (
    "no_lift_recommendation_kv260_ssh_unreachable"
)
NO_LIFT_PARTIAL_RECOMMENDATION = (
    "no_lift_recommendation_terminal_condition_not_confirmed"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "kv260_ssh_reachable",
    "kv260_overlay_loaded",
    "kv260_terminal_transcript_present",
    "kv260_terminal_condition_confirmed",
    "kv260_mandate_lift_recommendation",
    "polarfire_ssh_reachable",
    "gatemate_openfpgaloader_installed",
    "speedup_claim_avoided_assert",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "SSH-attached board tests plus a disk check; no live model inference."
    ),
    "preconditions_checked": (
        "Records per-board reachability checks, including the correct KV260 SSH "
        "precondition rather than host SD-card state."
    ),
    "kv260_ssh_reachable": "Honest KV260 board state.",
    "kv260_overlay_loaded": (
        "Confirms the carnot_ising overlay is the latest real-board-deployable "
        "bitstream."
    ),
    "kv260_terminal_transcript_present": (
        "True iff the Exp 3709 non-fabricated board-latency transcript exists "
        "on disk with at least 30 samples."
    ),
    "kv260_terminal_condition_confirmed": (
        "Bare bool: overlay confirmation plus non-fabricated transcript means "
        "KV260 is terminal."
    ),
    "kv260_mandate_lift_recommendation": (
        "Operator-action recommendation for lifting the per-milestone KV260 "
        "mandate once terminal."
    ),
    "polarfire_ssh_reachable": "Honest PolarFire board state.",
    "gatemate_openfpgaloader_installed": "Records the GateMate root blocker honestly.",
    "speedup_claim_avoided_assert": (
        "Asserts no thermalization, equilibrium, or comparative hardware-speedup "
        "claim is made."
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
    """Run one bounded command while preserving transcript data for audit."""
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
    except OSError as exc:
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


def _is_positive_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and float(value) > 0.0


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_exp3709_terminal_transcript(repo_root: str | Path = ".") -> dict[str, object]:
    """Inspect prior KV260 transcript evidence without synthesizing new samples."""
    path = Path(repo_root) / EXP3709_REL_PATH
    evidence: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "non_fabricated": False,
        "sample_count": 0,
        "median_ms": None,
        "sha256": None,
        "overlay_name": None,
        "validation_reasons": [],
    }
    reasons = evidence["validation_reasons"]
    if not path.exists():
        reasons.append("missing_exp3709_artifact")
        return evidence

    evidence["sha256"] = _file_sha256(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append("invalid_json")
        return evidence

    samples = payload.get("board_latency_samples")
    if isinstance(samples, list):
        evidence["sample_count"] = len(samples)
    if not isinstance(samples, list) or len(samples) < 30:
        reasons.append("latency_sample_count_below_30")

    positive_samples = isinstance(samples, list) and all(
        _is_positive_number(sample) for sample in samples
    )
    if not positive_samples:
        reasons.append("latency_samples_not_positive")
    if positive_samples and isinstance(samples, list):
        numeric_samples = [float(sample) for sample in samples]
        evidence["median_ms"] = float(payload.get("board_latency_median_ms") or 0.0)
        if evidence["median_ms"] <= 0.0:
            evidence["median_ms"] = sorted(numeric_samples)[len(numeric_samples) // 2]

    command_probes = payload.get("command_probes") or {}
    latency_probe: Any = command_probes.get("kv260_latency_harness")
    probe_text = ""
    if isinstance(latency_probe, dict):
        probe_text = str(latency_probe.get("combined_output") or latency_probe.get("stdout") or "")
    has_timing_evidence = (
        isinstance(latency_probe, dict)
        and latency_probe.get("exit_code") == 0
        and "BOARD_HARNESS_START exp3709" in probe_text
        and "per_sample_wall_ms" in probe_text
    )
    if not has_timing_evidence:
        reasons.append("board_harness_timing_evidence_missing")

    if payload.get("terminal_condition_met") is not True:
        reasons.append("exp3709_terminal_condition_not_met")
    if payload.get("inference_substrate") != "hardware_smoke":
        reasons.append("exp3709_inference_substrate_not_hardware_smoke")

    overlay_name = payload.get("kv260_overlay_loaded")
    evidence["overlay_name"] = overlay_name if isinstance(overlay_name, str) else None
    evidence["non_fabricated"] = not reasons
    return evidence


def _detect_kv260_overlay(text: str) -> str | None:
    for overlay in VALID_KV260_OVERLAYS:
        if overlay in text:
            return overlay
    return None


def _xmutil_requires_root(probe: CommandProbe) -> bool:
    lowered = probe.combined_output.lower()
    return "root privileges" in lowered or "using 'sudo'" in lowered


def _precondition_entry(
    resource: str,
    command: tuple[str, ...],
    probe: CommandProbe,
    path: str | None = None,
    command_text: str | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "resource": resource,
        "command": command_text or command_to_string(command),
        "available": probe.exit_code == 0,
        "exit_code": probe.exit_code,
    }
    if path is not None:
        entry["path"] = path
    return entry


def _polarfire_uptime_value(probe: CommandProbe | None) -> str | None:
    if probe is None:
        return None
    if probe.exit_code != 0:
        return "unknown"
    return probe.stdout.strip() or "unknown"


def _polarfire_dispatch_value(probe: CommandProbe | None) -> str | None:
    if probe is None:
        return None
    if probe.exit_code != 0:
        return "not_found"
    return probe.stdout.strip() or "not_found"


def _polarfire_continuity_state(
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
        return "reachable_uptime_and_dispatch_path_recorded"
    return "reachable_probe_values_incomplete"


def _gatemate_known_blocker(openfpgaloader_installed: bool) -> str:
    if openfpgaloader_installed:
        return "flash/smoke host-IO hang blocker recorded; flash and smoke not run"
    return "openFPGALoader not found on PATH; flash/smoke host-IO hang blocker remains"


def _honest_verdict(kv260_ssh_reachable: bool, terminal_confirmed: bool) -> str:
    if not kv260_ssh_reachable:
        return KV260_UNREACHABLE_VERDICT
    if terminal_confirmed:
        return KV260_TERMINAL_VERDICT
    return PARTIAL_VERDICT


def _mandate_lift_recommendation(
    kv260_ssh_reachable: bool,
    terminal_confirmed: bool,
) -> str:
    if terminal_confirmed:
        return MANDATE_LIFT_RECOMMENDATION
    if not kv260_ssh_reachable:
        return NO_LIFT_KV260_UNREACHABLE_RECOMMENDATION
    return NO_LIFT_PARTIAL_RECOMMENDATION


def validate_artifact(payload: dict[str, object]) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(payload)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")


def build_artifact(
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> dict[str, object]:
    """Build a consolidated artifact from live reachability and disk evidence."""
    started = time.perf_counter()
    transcript_evidence = inspect_exp3709_terminal_transcript(repo_root)

    kv260_ssh_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    kv260_ssh_reachable = kv260_ssh_probe.exit_code == 0
    kv260_listapps_probe = None
    kv260_listapps_sudo_probe = None
    kv260_overlay_name = None

    if kv260_ssh_reachable:
        kv260_listapps_probe = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
        if kv260_listapps_probe.exit_code == 0:
            kv260_overlay_name = _detect_kv260_overlay(kv260_listapps_probe.combined_output)
        if kv260_overlay_name is None and _xmutil_requires_root(kv260_listapps_probe):
            kv260_listapps_sudo_probe = command_runner(KV260_LISTAPPS_SUDO_COMMAND, 30.0)
            if kv260_listapps_sudo_probe.exit_code == 0:
                kv260_overlay_name = _detect_kv260_overlay(
                    kv260_listapps_sudo_probe.combined_output
                )

    polarfire_ssh_probe = command_runner(POLARFIRE_SSH_COMMAND, 10.0)
    polarfire_ssh_reachable = polarfire_ssh_probe.exit_code == 0
    polarfire_uptime_probe = (
        command_runner(POLARFIRE_UPTIME_COMMAND, 30.0) if polarfire_ssh_reachable else None
    )
    polarfire_dispatch_probe = (
        command_runner(POLARFIRE_DISPATCH_COMMAND, 30.0)
        if polarfire_ssh_reachable
        else None
    )

    gatemate_which_probe = command_runner(GATEMATE_OPENFPGALOADER_COMMAND, 10.0)
    gatemate_path = gatemate_which_probe.stdout.strip() or None
    gatemate_openfpgaloader_installed = gatemate_which_probe.exit_code == 0

    kv260_overlay_loaded = kv260_overlay_name is not None
    kv260_terminal_transcript_present = bool(transcript_evidence["non_fabricated"])
    kv260_terminal_condition_confirmed = (
        kv260_overlay_loaded and kv260_terminal_transcript_present
    )
    raw_elapsed = duration_s if duration_s is not None else time.perf_counter() - started
    elapsed = round(max(float(raw_elapsed), 0.0001), 4)

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "honest_verdict": _honest_verdict(
            kv260_ssh_reachable,
            kv260_terminal_condition_confirmed,
        ),
        "inference_substrate": "hardware_smoke",
        "preconditions_checked": [
            _precondition_entry("kv260_ssh", KV260_SSH_COMMAND, kv260_ssh_probe),
            _precondition_entry(
                "polarfire_ssh",
                POLARFIRE_SSH_COMMAND,
                polarfire_ssh_probe,
            ),
            _precondition_entry(
                "gatemate_openfpgaloader",
                GATEMATE_OPENFPGALOADER_COMMAND,
                gatemate_which_probe,
                path=gatemate_path,
                command_text="command -v openFPGALoader",
            ),
        ],
        "kv260_ssh_reachable": kv260_ssh_reachable,
        "kv260_overlay_loaded": kv260_overlay_loaded,
        "kv260_overlay_name": kv260_overlay_name,
        "kv260_terminal_transcript_present": kv260_terminal_transcript_present,
        "kv260_terminal_transcript_path": transcript_evidence["path"],
        "kv260_terminal_transcript_sha256": transcript_evidence["sha256"],
        "kv260_terminal_transcript_sample_count": transcript_evidence["sample_count"],
        "kv260_terminal_transcript_median_ms": transcript_evidence["median_ms"],
        "kv260_terminal_transcript_validation": transcript_evidence,
        "kv260_terminal_condition_confirmed": kv260_terminal_condition_confirmed,
        "kv260_mandate_lift_recommendation": _mandate_lift_recommendation(
            kv260_ssh_reachable,
            kv260_terminal_condition_confirmed,
        ),
        "polarfire_ssh_reachable": polarfire_ssh_reachable,
        "polarfire_uptime": _polarfire_uptime_value(polarfire_uptime_probe),
        "polarfire_carnot_dispatch_path": _polarfire_dispatch_value(
            polarfire_dispatch_probe
        ),
        "polarfire_continuity_state": _polarfire_continuity_state(
            polarfire_ssh_reachable,
            polarfire_uptime_probe,
            polarfire_dispatch_probe,
        ),
        "gatemate_openfpgaloader_installed": gatemate_openfpgaloader_installed,
        "gatemate_openfpgaloader_path": gatemate_path,
        "gatemate_known_blocker": _gatemate_known_blocker(
            gatemate_openfpgaloader_installed
        ),
        "speedup_claim_avoided_assert": True,
        "narrowing_scope": "POC functional latency anchor only; no comparative CPU claim.",
        "command_probes": {
            "kv260_ssh": kv260_ssh_probe.as_dict(),
            "kv260_xmutil_listapps": (
                kv260_listapps_probe.as_dict()
                if kv260_listapps_probe is not None
                else None
            ),
            "kv260_xmutil_listapps_sudo": (
                kv260_listapps_sudo_probe.as_dict()
                if kv260_listapps_sudo_probe is not None
                else None
            ),
            "polarfire_ssh": polarfire_ssh_probe.as_dict(),
            "polarfire_uptime": (
                polarfire_uptime_probe.as_dict()
                if polarfire_uptime_probe is not None
                else None
            ),
            "polarfire_carnot_dispatch_path": (
                polarfire_dispatch_probe.as_dict()
                if polarfire_dispatch_probe is not None
                else None
            ),
            "gatemate_openfpgaloader_which": gatemate_which_probe.as_dict(),
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed,
    }
    checksum_payload = dict(payload)
    payload["reproducibility_checksum"] = sha256_payload(checksum_payload)
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
    payload = build_artifact(
        repo_root=repo_root,
        command_runner=command_runner,
        duration_s=duration_s,
    )
    return write_artifact(repo_root, payload)


def main() -> None:  # pragma: no cover
    run_experiment(Path("."))


if __name__ == "__main__":  # pragma: no cover
    main()
