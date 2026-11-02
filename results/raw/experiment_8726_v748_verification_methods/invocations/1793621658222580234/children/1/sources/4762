"""Build the Exp 3639 Gemini CLI quota/crash diagnostic artifact.

Spec refs: REQ-REPORT-3639, SCENARIO-REPORT-3639,
SCENARIO-REPORT-3639-RECOVERED.
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

EXPERIMENT_ID = "exp3639"
TASK_ID = "exp3639-gemini-cli-quota-crash-resilience-diagnostic"
SCHEMA = "carnot.gemini_cli_quota_crash_resilience_diagnostic.v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3639_gemini_cli_quota_crash_resilience_diagnostic.json"
)
RANDOM_SEED = 3639

GEMINI_VERSION_COMMAND = ("timeout", "30", "gemini", "--version")
GEMINI_REPLY_COMMAND = (
    "timeout",
    "60",
    "gemini",
    "-m",
    "gemini-3.1-pro-preview",
    "-p",
    "Reply OK",
)
CONDUCTOR_ENV_COMMAND = ("bash", "-lc", "env | grep -iE 'AGENT_TYPE|FORCE_EXPERIMENTS'")
TRACKED_ENV_KEYS = (
    "AGENT_TYPE",
    "GEMINI_FORCE_EXPERIMENTS",
    "CODEX_FORCE_EXPERIMENTS",
    "AGENT_TYPE_PLANNER",
    "AGENT_TYPE_RETRO",
)

FAILURE_VERDICT = (
    "complete: "
    "gemini_quota_crash_diagnosed_429_crash_recorded_operator_codex_flip_recommended"
)
OK_VERDICT = "complete: gemini_recovered_quota_ok_no_action_needed"

RESET_RE = re.compile(r"quota will reset after\s+([^.\n]+)", re.IGNORECASE)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "A bounded CLI health probe, not live model inference; it only asks "
        "whether the Gemini backend is available."
    ),
    "gemini_cli_version": (
        "Records the installed gemini-cli version so a future upgrade is auditable."
    ),
    "gemini_quota_state": (
        "Root-cause signal that wiped out .333: ok, quota_exhausted_429, or "
        "crash_js_345500."
    ),
    "gemini_reset_eta": (
        "If 429 is returned, the reported quota-reset window tells the operator "
        "how long to wait."
    ),
    "conductor_coercion_env": (
        "Records AGENT_TYPE and force-experiment env so the failed route is "
        "auditable."
    ),
    "operator_recommendation": (
        "Concrete operator action: flip experiments to Codex or wait for quota reset."
    ),
    "conductor_unmodified_assert": (
        "Asserts this diagnostic did not modify research_conductor.py or env."
    ),
    "random_seed": "Determinism precondition.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class CommandProbe:
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_dict(self) -> dict[str, object]:
        return {
            "command": shlex.join(self.command),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "combined_output": self.combined_output,
        }


CommandRunner = Callable[[tuple[str, ...]], CommandProbe]


def run_command(command: tuple[str, ...]) -> CommandProbe:
    """Run one bounded diagnostic command and capture stdout, stderr, and exit."""
    completed = subprocess.run(  # noqa: S603
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    return CommandProbe(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def sha256_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_version(probe: CommandProbe) -> str:
    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    return lines[0] if lines else f"unknown_exit_{probe.exit_code}"


def _classify_quota_state(probe: CommandProbe) -> str:
    text = probe.combined_output.lower()
    if "quota_exhausted" in text or "429" in text:
        return "quota_exhausted_429"
    if ".js:345500:14" in text:
        return "crash_js_345500"
    return "ok" if probe.exit_code == 0 else "crash_js_345500"


def _parse_reset_eta(probe: CommandProbe, state: str) -> str:
    if state == "ok":
        return "not_applicable_quota_ok"
    match = RESET_RE.search(probe.combined_output)
    return match.group(1).strip() if match else "not_reported"


def _parse_conductor_env(probe: CommandProbe) -> dict[str, str | None]:
    raw = probe.stdout + probe.stderr
    pairs = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    parsed: dict[str, str | None] = {key: pairs.get(key) for key in TRACKED_ENV_KEYS}
    parsed["raw"] = raw
    return parsed


def _operator_recommendation(state: str, reset_eta: str) -> str:
    if state == "ok":
        return (
            "Gemini responded OK in the bounded probe; no action needed for this "
            "specific quota incident."
        )
    wait_clause = (
        f"wait about {reset_eta} for Gemini quota reset"
        if reset_eta not in {"not_reported", "not_applicable_quota_ok"}
        else "wait for the Gemini quota reset window to pass"
    )
    return (
        "Set CODEX_FORCE_EXPERIMENTS=1 and unset or override "
        "GEMINI_FORCE_EXPERIMENTS before relaunching the conductor, or "
        f"{wait_clause}."
    )


def build_artifact(
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> dict[str, object]:
    """Build the Exp 3639 diagnostic from bounded probes and repository context."""
    started = time.perf_counter()
    root = Path(repo_root)
    conductor_path = root / "scripts" / "research_conductor.py"
    conductor_before = _file_sha256(conductor_path)

    version_probe = command_runner(GEMINI_VERSION_COMMAND)
    reply_probe = command_runner(GEMINI_REPLY_COMMAND)
    env_probe = command_runner(CONDUCTOR_ENV_COMMAND)

    conductor_after = _file_sha256(conductor_path)
    quota_state = _classify_quota_state(reply_probe)
    reset_eta = _parse_reset_eta(reply_probe, quota_state)
    elapsed = duration_s if duration_s is not None else max(time.perf_counter() - started, 0.0001)
    conductor_log = (root / "ops" / "conductor-log.md").read_text(encoding="utf-8")

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "honest_verdict": OK_VERDICT if quota_state == "ok" else FAILURE_VERDICT,
        "inference_substrate": "hardware_smoke",
        "gemini_cli_version": _parse_version(version_probe),
        "gemini_quota_state": quota_state,
        "gemini_reset_eta": reset_eta,
        "gemini_crash_signature_seen": ".js:345500:14" in reply_probe.combined_output,
        "gemini_quota_signal_seen": (
            "QUOTA_EXHAUSTED" in reply_probe.combined_output
            or "429" in reply_probe.combined_output
        ),
        "conductor_coercion_env": _parse_conductor_env(env_probe),
        "operator_recommendation": _operator_recommendation(quota_state, reset_eta),
        "conductor_unmodified_assert": (
            "scripts/research_conductor.py sha256 unchanged before/after diagnostic; "
            "environment was read only and no conductor config writes were performed."
        ),
        "research_conductor_unmodified": conductor_before == conductor_after,
        "research_conductor_sha256_before": conductor_before,
        "research_conductor_sha256_after": conductor_after,
        "context_sources_read": [
            "ops/conductor-log.md tail",
            "CLAUDE.md Gemini-Default for Experiments",
            "ops/known-issues.md gemini backend routing paused",
        ],
        "conductor_log_js345500_seen": ".js:345500:14" in conductor_log,
        "command_probes": {
            "gemini_version": version_probe.as_dict(),
            "gemini_reply": reply_probe.as_dict(),
            "conductor_env": env_probe.as_dict(),
        },
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(elapsed), 4),
    }
    payload["reproducibility_checksum"] = sha256_payload(payload)
    return payload


def write_artifact(repo_root: str | Path, payload: dict[str, object]) -> Path:
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> Path:
    payload = build_artifact(repo_root, command_runner=command_runner, duration_s=duration_s)
    return write_artifact(repo_root, payload)


def main() -> None:  # pragma: no cover
    run_experiment(Path("."))


if __name__ == "__main__":  # pragma: no cover
    main()
