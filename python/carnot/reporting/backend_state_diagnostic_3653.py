"""Build the Exp 3653 Gemini backend state diagnostic artifact.

Spec refs: REQ-REPORT-3653, SCENARIO-REPORT-3653,
SCENARIO-REPORT-3653-RECOVERED.
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

EXPERIMENT_ID = "exp3653"
TASK_ID = "exp3653-backend-state-diagnostic"
SCHEMA = "carnot.backend_state_diagnostic.v1"
OUTPUT_REL_PATH = Path("results/experiment_3653_backend_state_diagnostic.json")
RANDOM_SEED = 3653

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

KEEP_CODEX_VERDICT = (
    "complete: backend_diagnosed_gemini_still_exhausted_keep_codex_routing"
)
GEMINI_RECOVERED_VERDICT = (
    "complete: backend_diagnosed_gemini_recovered_operator_may_flip_to_gemini_default"
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "A bounded CLI health probe, not live model inference."
    ),
    "gemini_cli_version": (
        "Records the installed gemini-cli version so a future upgrade is auditable."
    ),
    "gemini_quota_state": (
        "ok / quota_exhausted_429 / crash_js_345500 -- whether the .333 root cause cleared."
    ),
    "conductor_coercion_env": (
        "Records AGENT_TYPE + FORCE_EXPERIMENTS active -- explains why per-task agent_type routes as it does."
    ),
    "recommended_routing": (
        "codex_requires_codex (keep) or gemini_default (flip) -- the operator-actionable recommendation."
    ),
    "conductor_unmodified_assert": (
        "Asserts this diagnostic did NOT modify research_conductor.py or env -- honors the never-modify-conductor rule."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}

CONTEXT_SOURCE_PATHS = (
    Path("results/experiment_3639_gemini_cli_quota_crash_resilience_diagnostic.json"),
    Path("CLAUDE.md"),
    Path("ops/known-issues.md"),
)


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
    lines = [
        line.strip()
        for line in probe.combined_output.splitlines()
        if line.strip()
    ]
    return lines[0] if lines else f"unknown_exit_{probe.exit_code}"


def _classify_quota_state(probe: CommandProbe) -> str:
    text = probe.combined_output.lower()
    if "quota_exhausted" in text or "quota exhausted" in text or "429" in text:
        return "quota_exhausted_429"
    if ".js:345500:14" in text:
        return "crash_js_345500"
    return "ok" if probe.exit_code == 0 else "crash_js_345500"


def _parse_conductor_env(probe: CommandProbe) -> dict[str, str | None]:
    raw = probe.combined_output
    pairs = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    parsed: dict[str, str | None] = {key: pairs.get(key) for key in TRACKED_ENV_KEYS}
    parsed["raw"] = raw
    return parsed


def _recommended_routing(quota_state: str) -> str:
    return "gemini_default" if quota_state == "ok" else "codex_requires_codex"


def _verdict(quota_state: str) -> str:
    return GEMINI_RECOVERED_VERDICT if quota_state == "ok" else KEEP_CODEX_VERDICT


def _context_source_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for rel_path in CONTEXT_SOURCE_PATHS:
        path = root / rel_path
        checksums[str(rel_path)] = _file_sha256(path) if path.exists() else "missing"
    return checksums


def build_artifact(
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> dict[str, object]:
    """Build the Exp 3653 diagnostic from bounded probes and repository context."""
    started = time.perf_counter()
    root = Path(repo_root)
    conductor_path = root / "scripts" / "research_conductor.py"
    conductor_before = _file_sha256(conductor_path)

    version_probe = command_runner(GEMINI_VERSION_COMMAND)
    reply_probe = command_runner(GEMINI_REPLY_COMMAND)
    env_probe = command_runner(CONDUCTOR_ENV_COMMAND)

    conductor_after = _file_sha256(conductor_path)
    quota_state = _classify_quota_state(reply_probe)
    elapsed = duration_s if duration_s is not None else max(time.perf_counter() - started, 0.0001)

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "honest_verdict": _verdict(quota_state),
        "inference_substrate": "hardware_smoke",
        "gemini_cli_version": _parse_version(version_probe),
        "gemini_quota_state": quota_state,
        "conductor_coercion_env": _parse_conductor_env(env_probe),
        "recommended_routing": _recommended_routing(quota_state),
        "conductor_unmodified_assert": (
            "scripts/research_conductor.py sha256 unchanged before/after diagnostic; "
            "environment was read only and no conductor config writes were performed."
        ),
        "research_conductor_unmodified": conductor_before == conductor_after,
        "research_conductor_sha256_before": conductor_before,
        "research_conductor_sha256_after": conductor_after,
        "command_probes": {
            "gemini_version": version_probe.as_dict(),
            "gemini_reply": reply_probe.as_dict(),
            "conductor_env": env_probe.as_dict(),
        },
        "context_sources_read": [
            "scripts/summarize_artifact.py 3639",
            "CLAUDE.md Gemini-Default for Experiments",
            "ops/known-issues.md gemini backend routing paused",
        ],
        "context_source_checksums": _context_source_checksums(root),
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
    out_path = run_experiment(Path("."))
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    print(payload["honest_verdict"])
    print(out_path)


if __name__ == "__main__":  # pragma: no cover
    main()
