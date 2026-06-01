"""Build the Exp 3679 Gemini backend state diagnostic v3 artifact.

Spec refs: REQ-REPORT-3679, SCENARIO-REPORT-3679,
SCENARIO-REPORT-3679-UNSTABLE.
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

EXPERIMENT_ID = "exp3679"
TASK_ID = "exp3679-backend-state-diagnostic-v3"
SCHEMA = "carnot.backend_state_diagnostic.v3"
OUTPUT_REL_PATH = Path("results/experiment_3679_backend_state_diagnostic_v3.json")
RANDOM_SEED = 3679

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

STABLE_VERDICT = (
    "complete: backend_diagnosed_gemini_stable_3rd_probe_v338_may_flip_to_gemini_default"
)
UNSTABLE_VERDICT = (
    "complete: backend_diagnosed_gemini_still_unstable_keep_codex_routing"
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "hardware_smoke (principle: a bounded CLI health probe, not live model inference)."
    ),
    "gemini_cli_version": (
        "Records the installed gemini-cli version so a future upgrade is auditable."
    ),
    "gemini_quota_state": (
        "ok / quota_exhausted_429 / crash_js_345500 -- whether gemini is stable on a 3rd consecutive probe."
    ),
    "consecutive_stable_probes": (
        "Counts consecutive milestones gemini probed OK (was 2 at exp3666); 3+ supports a .338 flip."
    ),
    "conductor_coercion_env": (
        "Records AGENT_TYPE + FORCE_EXPERIMENTS active -- explains why per-task agent_type routes as it does."
    ),
    "recommended_routing": (
        "codex_requires_codex (keep) or gemini_default_eligible_for_v338 -- the operator-actionable recommendation."
    ),
    "conductor_unmodified_assert": (
        "Asserts this diagnostic did NOT modify research_conductor.py or env."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}

CONTEXT_SOURCE_PATHS = (
    Path("results/experiment_3653_backend_state_diagnostic.json"),
    Path("results/experiment_3666_backend_state_diagnostic_v2.json"),
    Path("scripts/summarize_artifact.py"),
    Path("CLAUDE.md"),
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


def _has_ok_reply(probe: CommandProbe) -> bool:
    return any(line.strip().upper() == "OK" for line in probe.stdout.splitlines())


def _classify_quota_state(probe: CommandProbe) -> str:
    text = probe.combined_output.lower()
    if "quota_exhausted" in text or "quota exhausted" in text or "429" in text:
        return "quota_exhausted_429"
    if probe.exit_code == 0 and _has_ok_reply(probe):
        return "ok"
    return "crash_js_345500"


def _parse_conductor_env(probe: CommandProbe) -> dict[str, str | None]:
    raw = probe.combined_output
    pairs = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    parsed: dict[str, str | None] = {key: pairs.get(key) for key in TRACKED_ENV_KEYS}
    parsed["raw"] = raw
    return parsed


def _previous_probe_states(root: Path) -> dict[str, str]:
    paths = {
        "exp3653": root / "results" / "experiment_3653_backend_state_diagnostic.json",
        "exp3666": root / "results" / "experiment_3666_backend_state_diagnostic_v2.json",
    }
    states: dict[str, str] = {}
    for key, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        states[key] = str(payload.get("gemini_quota_state", "missing"))
    return states


def _consecutive_stable_probes(
    previous_states: dict[str, str],
    current_state: str,
) -> int:
    return (
        3
        if previous_states == {"exp3653": "ok", "exp3666": "ok"}
        and current_state == "ok"
        else 0
    )


def _recommended_routing(quota_state: str, consecutive_stable_probes: int) -> str:
    return (
        "gemini_default_eligible_for_v338"
        if quota_state == "ok" and consecutive_stable_probes >= 3
        else "codex_requires_codex"
    )


def _verdict(recommended_routing: str) -> str:
    return (
        STABLE_VERDICT
        if recommended_routing == "gemini_default_eligible_for_v338"
        else UNSTABLE_VERDICT
    )


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
    """Build the Exp 3679 diagnostic from bounded probes and repository context."""
    started = time.perf_counter()
    root = Path(repo_root)
    conductor_path = root / "scripts" / "research_conductor.py"
    conductor_before = _file_sha256(conductor_path)

    previous_states = _previous_probe_states(root)
    version_probe = command_runner(GEMINI_VERSION_COMMAND)
    reply_probe = command_runner(GEMINI_REPLY_COMMAND)
    env_probe = command_runner(CONDUCTOR_ENV_COMMAND)

    conductor_after = _file_sha256(conductor_path)
    quota_state = _classify_quota_state(reply_probe)
    stable_probe_count = _consecutive_stable_probes(previous_states, quota_state)
    recommended_routing = _recommended_routing(quota_state, stable_probe_count)
    elapsed = duration_s if duration_s is not None else max(time.perf_counter() - started, 0.0001)

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "honest_verdict": _verdict(recommended_routing),
        "inference_substrate": "hardware_smoke",
        "gemini_cli_version": _parse_version(version_probe),
        "gemini_quota_state": quota_state,
        "previous_gemini_probe_states": previous_states,
        "consecutive_stable_probes": stable_probe_count,
        "conductor_coercion_env": _parse_conductor_env(env_probe),
        "recommended_routing": recommended_routing,
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
            "scripts/summarize_artifact.py results/experiment_3653_backend_state_diagnostic.json",
            "scripts/summarize_artifact.py results/experiment_3666_backend_state_diagnostic_v2.json",
            "CLAUDE.md Gemini-Default for Experiments",
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
