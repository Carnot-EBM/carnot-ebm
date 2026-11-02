"""Exp 1584 Tenstorrent Wormhole n150d block-Gibbs preflight.

The preflight is deliberately conservative: it checks for local TT-Metalium
tooling, local or remote Wormhole access signals, and public TT-Metal source
reachability without installing packages or changing system state. It writes a
blocked artifact when access is absent and never claims Wormhole execution
without a transcript.

Spec traces: REQ-SAMPLE-064, SCENARIO-SAMPLE-092.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DATE = "20260509"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "experiment_1584_tenstorrent_wormhole_n150d_block_gibbs_preflight.json"
)
DEFAULT_TRANSCRIPT_PATH = (
    PROJECT_ROOT / "logs" / "experiment_1584_tenstorrent_wormhole_preflight_transcript.txt"
)
DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT / "docs" / "research-notes" / "tenstorrent-wormhole-block-gibbs-preflight.md"
)
DEFAULT_PYTHON = "python3"

SPEC_REFS = ["REQ-SAMPLE-064", "SCENARIO-SAMPLE-092"]

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "wormhole_access_available",
    "tt_metalium_available",
    "hardware_transcript_path",
    "benchmark_protocol_path",
    "wormhole_preflight_ready",
    "blocked_reason",
    "no_hardware_claim_without_transcript",
    "honest_verdict",
}

TT_ENV_VARS = (
    "TT_METAL_HOME",
    "TT_METALIUM_HOME",
    "TT_HOME",
    "ARCH_NAME",
)

WORMHOLE_CLOUD_ENV_VARS = (
    "TENSTORRENT_CLOUD_API_KEY",
    "TENSTORRENT_CLOUD_TOKEN",
    "TT_CLOUD_API_KEY",
    "TT_CLOUD_TOKEN",
    "TT_CLOUD_URL",
    "TT_REMOTE_HOST",
    "TT_METAL_REMOTE_HOST",
    "KOYEB_TOKEN",
)

TT_IMPORT_PROBE_CODE = """\
import importlib
import json

modules = []
for name in ("ttnn", "tt_lib", "tt_metal"):
    try:
        importlib.import_module(name)
    except Exception:
        continue
    modules.append(name)
print(json.dumps(modules))
"""


@dataclass(frozen=True)
class CommandResult:
    """Bounded command result preserved for transcript evidence."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


@dataclass(frozen=True)
class AvailabilitySummary:
    """Non-mutating TT-Metalium and Wormhole availability summary."""

    tt_metalium_available: bool
    wormhole_access_available: bool
    tt_metalium_signals: list[dict[str, Any]]
    wormhole_access_signals: list[dict[str, Any]]
    remote_reachability: dict[str, Any]


CommandRunner = Callable[[list[str], float], CommandResult]
CommandLookup = Callable[[str], str | None]


def _compact_text(text: str, *, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def command_payload(result: CommandResult) -> dict[str, Any]:
    """Convert command output into a stable, bounded JSON/transcript payload."""

    return {
        "command": result.command,
        "returncode": int(result.returncode),
        "stdout": _compact_text(result.stdout),
        "stderr": _compact_text(result.stderr),
        "timed_out": bool(result.timed_out),
        "duration_s": round(float(result.duration_s), 6),
    }


def run_command(command: list[str], timeout_s: float) -> CommandResult:
    """Run one non-mutating command and return captured output."""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - timing-sensitive fallback
        duration_s = time.perf_counter() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = f"{stderr}\ncommand timed out after {timeout_s} seconds".strip()
        return CommandResult(
            command=command,
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            duration_s=duration_s,
        )

    return CommandResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
        duration_s=time.perf_counter() - started,
    )


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    artifact_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def write_in_progress_artifact(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """REQ-SAMPLE-064: write the startup marker before availability probing."""

    marker: dict[str, Any] = {
        "status": "in_progress",
        "wormhole_access_available": False,
        "tt_metalium_available": False,
        "hardware_transcript_path": None,
        "benchmark_protocol_path": None,
        "wormhole_preflight_ready": False,
        "blocked_reason": "Preflight checks have not run yet.",
        "no_hardware_claim_without_transcript": True,
        "honest_verdict": "in_progress",
    }
    return _write_json(path, marker)


def _json_list_from_stdout(stdout: str) -> list[str]:
    try:
        parsed = json.loads(stdout.strip() or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _existing_source_paths(project_root: Path, env: Mapping[str, str]) -> list[str]:
    candidates: list[Path] = []
    for name in ("TT_METAL_HOME", "TT_METALIUM_HOME"):
        value = env.get(name)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            project_root.parent / "tt-metal",
            project_root.parent / "tt-metalium",
            project_root.parent / "tenstorrent" / "tt-metal",
        ]
    )
    seen: set[str] = set()
    existing: list[str] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen and candidate.exists():
            existing.append(key)
            seen.add(key)
    return existing


def _present_env_names(env: Mapping[str, str], names: Iterable[str]) -> list[str]:
    return [name for name in names if bool(str(env.get(name, "")).strip())]


def _device_path_strings(device_paths: Iterable[str | Path] | None) -> list[str]:
    if device_paths is not None:
        return [str(path) for path in device_paths]
    return [str(path) for path in Path("/dev").glob("tenstorrent*")]


def _probe_lspci(
    *,
    runner: CommandRunner,
    command_lookup: CommandLookup,
    timeout_s: float,
) -> dict[str, Any]:
    if command_lookup("lspci") is None:
        return {"name": "lspci Wormhole PCI probe", "available": False, "reason": "lspci not found"}
    result = runner(["lspci", "-nn"], timeout_s)
    payload = command_payload(result)
    text = f"{result.stdout}\n{result.stderr}".lower()
    available = result.returncode == 0 and (
        "tenstorrent" in text or "wormhole" in text or "1e52" in text
    )
    return {
        "name": "lspci Wormhole PCI probe",
        "available": bool(available),
        "command_result": payload,
    }


def _probe_tt_smi(
    *,
    runner: CommandRunner,
    command_lookup: CommandLookup,
    timeout_s: float,
) -> dict[str, Any]:
    if command_lookup("tt-smi") is None:
        return {"name": "tt-smi device probe", "available": False, "reason": "tt-smi not found"}
    result = runner(["tt-smi"], timeout_s)
    text = f"{result.stdout}\n{result.stderr}".lower()
    available = result.returncode == 0 and (
        bool(text.strip()) or "wormhole" in text or "tenstorrent" in text
    )
    return {
        "name": "tt-smi device probe",
        "available": bool(available),
        "command_result": command_payload(result),
    }


def _probe_remote_tt_metal(
    *,
    runner: CommandRunner,
    command_lookup: CommandLookup,
    timeout_s: float,
) -> dict[str, Any]:
    if command_lookup("git") is None:
        return {
            "reachable": False,
            "status": "skipped_git_not_found",
            "url": "https://github.com/tenstorrent/tt-metal.git",
        }
    result = runner(
        ["git", "ls-remote", "--heads", "https://github.com/tenstorrent/tt-metal.git", "main"],
        timeout_s,
    )
    return {
        "reachable": result.returncode == 0,
        "status": "reachable" if result.returncode == 0 else "unreachable",
        "url": "https://github.com/tenstorrent/tt-metal.git",
        "command_result": command_payload(result),
    }


def probe_availability(
    *,
    project_root: str | Path = PROJECT_ROOT,
    env: Mapping[str, str] | None = None,
    runner: CommandRunner = run_command,
    command_lookup: CommandLookup = shutil.which,
    python_executable: str = DEFAULT_PYTHON,
    device_paths: Iterable[str | Path] | None = None,
    timeout_s: float = 20.0,
) -> AvailabilitySummary:
    """REQ-SAMPLE-064: check TT-Metalium and Wormhole access without mutation."""

    root = Path(project_root)
    active_env = os.environ if env is None else env

    import_result = runner([python_executable, "-c", TT_IMPORT_PROBE_CODE], timeout_s)
    importable_modules = _json_list_from_stdout(import_result.stdout) if import_result.returncode == 0 else []
    tt_env_names = _present_env_names(active_env, TT_ENV_VARS)
    source_paths = _existing_source_paths(root, active_env)
    tt_executables = [
        name for name in ("tt-smi", "tt-metal", "tt-run") if command_lookup(name) is not None
    ]
    tt_signals = [
        {
            "name": "TT-Metalium import probe",
            "available": bool(importable_modules),
            "modules": importable_modules,
            "command_result": command_payload(import_result),
        },
        {
            "name": "TT-Metalium environment variables",
            "available": bool(tt_env_names),
            "env_var_names": tt_env_names,
        },
        {
            "name": "TT-Metalium source checkout",
            "available": bool(source_paths),
            "paths": source_paths,
        },
        {
            "name": "TT-Metalium executable lookup",
            "available": bool(tt_executables),
            "executables": tt_executables,
        },
    ]

    device_signal = {
        "name": "/dev/tenstorrent device nodes",
        "available": bool(_device_path_strings(device_paths)),
        "paths": _device_path_strings(device_paths),
    }
    cloud_env_names = _present_env_names(active_env, WORMHOLE_CLOUD_ENV_VARS)
    cloud_signal = {
        "name": "Wormhole cloud or remote access environment",
        "available": bool(cloud_env_names),
        "env_var_names": cloud_env_names,
    }
    wormhole_signals = [
        device_signal,
        cloud_signal,
        _probe_lspci(runner=runner, command_lookup=command_lookup, timeout_s=timeout_s),
        _probe_tt_smi(runner=runner, command_lookup=command_lookup, timeout_s=timeout_s),
    ]
    remote_reachability = {
        "tt_metal_github": _probe_remote_tt_metal(
            runner=runner,
            command_lookup=command_lookup,
            timeout_s=timeout_s,
        )
    }

    return AvailabilitySummary(
        tt_metalium_available=any(signal["available"] for signal in tt_signals),
        wormhole_access_available=any(signal["available"] for signal in wormhole_signals),
        tt_metalium_signals=tt_signals,
        wormhole_access_signals=wormhole_signals,
        remote_reachability=remote_reachability,
    )


def run_non_destructive_smoke(
    *,
    summary: AvailabilitySummary,
    runner: CommandRunner = run_command,
    command_lookup: CommandLookup = shutil.which,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Run a minimal smoke only when both access and TT-Metalium are present."""

    if not (summary.tt_metalium_available and summary.wormhole_access_available):
        return {
            "status": "skipped_unavailable",
            "successful": False,
            "reason": "TT-Metalium and Wormhole access are both required before smoke.",
        }
    if command_lookup("tt-smi") is None:
        return {
            "status": "skipped_no_safe_smoke_command",
            "successful": False,
            "reason": "tt-smi was not found; no safe local Wormhole smoke command is configured.",
        }
    result = runner(["tt-smi"], timeout_s)
    return {
        "status": "smoke_passed" if result.returncode == 0 else "smoke_failed",
        "successful": result.returncode == 0,
        "command": ["tt-smi"],
        "command_result": command_payload(result),
    }


def _blocked_reason(summary: AvailabilitySummary, smoke_result: Mapping[str, Any]) -> str:
    missing: list[str] = []
    if not summary.tt_metalium_available:
        missing.append("TT-Metalium was not detected")
    if not summary.wormhole_access_available:
        missing.append("Wormhole hardware or cloud access was not detected")
    if missing:
        return "; ".join(missing) + "."
    if not smoke_result.get("successful"):
        return "Wormhole and TT-Metalium were detected, but the non-destructive smoke did not pass."
    return "not_blocked"


def _acquisition_next_steps() -> list[str]:
    return [
        "Order or allocate a Tenstorrent Wormhole n150d host with PCIe Gen4 slot, EPS12V power, supported Linux kernel, and cooling headroom for the 160 W board.",
        "If local procurement is slower, request Tenstorrent Cloud or Koyeb Tenstorrent Wormhole access and record the instance type before running Carnot benchmarks.",
        "Install TT-Metalium from Tenstorrent's open-source tt-metal release instructions in an isolated user environment; do not treat a dry-run or source checkout as hardware execution.",
        "Confirm access with a transcript showing device enumeration, TT-Metalium version or commit, and a non-destructive smoke command such as tt-smi.",
        "Only after the smoke passes, run the block-Gibbs benchmark protocol and compare against the vendored THRML reference path.",
    ]


def _benchmark_protocol() -> dict[str, Any]:
    return {
        "workload": "THRML-compatible even/odd block-Gibbs Ising sampling at n=16 exact, n=128 sampled, and n=256 stress sizes with candidate warm starts.",
        "baseline": "Vendored THRML 0.1.3 CPU/JAX block-Gibbs transition operator on the same seeds and beta schedule.",
        "acceptance_gates": {
            "KL to THRML": "n=16 exact-state KL <= 1e-3 and n>=128 energy-histogram KL <= 0.05 with matched burn-in/sweeps.",
            "samples/sec": "Report raw samples/sec and pass prototype only when Wormhole is at least 2x the local THRML CPU baseline at n=128.",
            "samples/W": "Report board-power-normalized samples/W from tt-smi or platform telemetry and pass only when it is at least 2x the CPU baseline samples/J.",
            "open-toolchain reproducibility": "Record tt-metal commit or release, Carnot commit, build flags, kernel source, seed schedule, and transcript paths from a clean checkout.",
        },
    }


def build_artifact(
    *,
    summary: AvailabilitySummary,
    smoke_result: Mapping[str, Any],
    hardware_transcript_path: str | Path,
    benchmark_protocol_path: str | Path,
    run_date: str = DEFAULT_RUN_DATE,
) -> dict[str, Any]:
    """Build the terminal Exp 1584 artifact from probe evidence."""

    ready = bool(
        summary.tt_metalium_available
        and summary.wormhole_access_available
        and smoke_result.get("successful")
        and str(hardware_transcript_path)
    )
    blocked_reason = _blocked_reason(summary, smoke_result)
    artifact: dict[str, Any] = {
        "status": "complete" if ready else "blocked",
        "experiment_id": 1584,
        "run_date": run_date,
        "spec_refs": SPEC_REFS,
        "wormhole_access_available": bool(summary.wormhole_access_available),
        "tt_metalium_available": bool(summary.tt_metalium_available),
        "hardware_transcript_path": str(hardware_transcript_path),
        "benchmark_protocol_path": str(benchmark_protocol_path),
        "wormhole_preflight_ready": ready,
        "blocked_reason": blocked_reason,
        "no_hardware_claim_without_transcript": True,
        "availability_checks": {
            "tt_metalium_signals": summary.tt_metalium_signals,
            "wormhole_access_signals": summary.wormhole_access_signals,
            "remote_reachability": summary.remote_reachability,
        },
        "smoke_result": dict(smoke_result),
        "acquisition_or_cloud_next_steps": _acquisition_next_steps(),
        "expected_benchmark_protocol": _benchmark_protocol(),
        "honest_verdict": (
            "complete: wormhole_preflight_ready_smoke_transcript_recorded"
            if ready
            else "complete: wormhole_preflight_blocked_no_access_no_hardware_claim"
        ),
    }
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and the no-claim-without-transcript guard."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["no_hardware_claim_without_transcript"] is not True:
        raise ValueError("no_hardware_claim_without_transcript must be true")
    if not str(artifact["benchmark_protocol_path"]).strip():
        raise ValueError("benchmark_protocol_path is required")
    ready = bool(artifact["wormhole_preflight_ready"])
    if ready:
        if artifact["status"] != "complete":
            raise ValueError("ready artifacts must have status=complete")
        if not (artifact["wormhole_access_available"] and artifact["tt_metalium_available"]):
            raise ValueError("ready artifacts require both access and TT-Metalium")
        if not str(artifact["hardware_transcript_path"]).strip():
            raise ValueError("ready artifacts require a hardware transcript")
        if artifact["blocked_reason"] != "not_blocked":
            raise ValueError("ready artifacts must use blocked_reason=not_blocked")
    else:
        if artifact["status"] not in {"blocked", "complete"}:
            raise ValueError("terminal artifacts must have status blocked or complete")
        if not str(artifact["blocked_reason"]).strip():
            raise ValueError("blocked artifacts require blocked_reason")
    verdict = str(artifact["honest_verdict"])
    if not (
        verdict.startswith("complete:")
        or verdict.startswith("complete_")
        or verdict.startswith("success:")
        or verdict.startswith("passed:")
        or verdict.startswith("shipped:")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")


def write_transcript(
    *,
    path: str | Path,
    summary: AvailabilitySummary,
    smoke_result: Mapping[str, Any],
) -> Path:
    """Write a human-readable transcript of all non-mutating checks."""

    transcript_path = Path(path)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exp 1584 Tenstorrent Wormhole Preflight Transcript",
        "",
        "No privileged installation, driver change, or package mutation was performed.",
        "",
        "## TT-Metalium checks",
    ]
    for signal in summary.tt_metalium_signals:
        lines.append(f"- {signal['name']}: available={signal['available']}")
        if signal["name"] == "TT-Metalium import probe":
            lines.append("  TT-Metalium import probe command/result:")
        lines.append("  " + json.dumps(signal, sort_keys=True))
    lines.extend(["", "## Wormhole access checks"])
    for signal in summary.wormhole_access_signals:
        lines.append(f"- {signal['name']}: available={signal['available']}")
        lines.append("  " + json.dumps(signal, sort_keys=True))
    lines.extend(
        [
            "",
            "## Remote reachability checks",
            json.dumps(summary.remote_reachability, indent=2, sort_keys=True),
            "",
            "## Non-destructive smoke",
            json.dumps(dict(smoke_result), indent=2, sort_keys=True),
            "",
        ]
    )
    transcript_path.write_text("\n".join(lines), encoding="utf-8")
    return transcript_path


def write_benchmark_protocol(
    *,
    path: str | Path,
    artifact: Mapping[str, Any],
) -> Path:
    """Write the prototype plan and acceptance gates."""

    protocol_path = Path(path)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = _benchmark_protocol()
    gates = protocol["acceptance_gates"]
    text = f"""# Tenstorrent Wormhole Block-Gibbs Preflight

Spec refs: REQ-SAMPLE-064, SCENARIO-SAMPLE-092.

## Scope

This is a preflight/prototype plan for Tenstorrent Wormhole n150d as a possible
open-toolchain sovereignty platform for Carnot block-Gibbs sampling. It is not a
hardware execution result. The current artifact verdict is:
`{artifact["honest_verdict"]}`.

## Availability Verdict

- `wormhole_access_available`: `{artifact["wormhole_access_available"]}`
- `tt_metalium_available`: `{artifact["tt_metalium_available"]}`
- `wormhole_preflight_ready`: `{artifact["wormhole_preflight_ready"]}`
- `blocked_reason`: `{artifact["blocked_reason"]}`
- `hardware_transcript_path`: `{artifact["hardware_transcript_path"]}`

## Acquisition Or Cloud Next Steps

1. Order or allocate a Tenstorrent Wormhole n150d host with suitable PCIe,
   power, cooling, and Linux support.
2. If local n150d access is unavailable, request Tenstorrent Cloud access or a
   Koyeb Tenstorrent Wormhole instance for a chip-family smoke; record the
   instance type and do not relabel it as n150d hardware.
3. Build TT-Metalium from the official open-source tt-metal instructions in an
   isolated user environment and record the release or commit SHA.
4. Capture a transcript for device enumeration, TT-Metalium import/version, and
   a non-destructive smoke such as `tt-smi`.
5. Run the benchmark protocol below only after the smoke transcript exists.

## Benchmark Protocol

- Workload: {protocol["workload"]}
- Baseline: {protocol["baseline"]}

## Acceptance Gates

- KL to THRML: {gates["KL to THRML"]}
- samples/sec: {gates["samples/sec"]}
- samples/W: {gates["samples/W"]}
- open-toolchain reproducibility: {gates["open-toolchain reproducibility"]}

## Claim Boundary

No Wormhole execution, throughput, samples/W, or kernel result may be claimed
unless `hardware_transcript_path` points to a successful TT-Metalium/Wormhole
smoke transcript. Public TT-Metal reachability only proves that source is
reachable; it is not hardware access.

## Reference Links

- Tenstorrent Wormhole product page: https://future.tenstorrent.com/hardware/wormhole
- TT-Metalium source: https://github.com/tenstorrent/tt-metal
- Tenstorrent Cloud: https://tenstorrent.com/en/hardware/cloud
- Koyeb Tenstorrent n300s docs: https://www.koyeb.com/docs/hardware/tenstorrent-n300
"""
    protocol_path.write_text(text, encoding="utf-8")
    return protocol_path


def run_preflight(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    hardware_transcript_path: str | Path = DEFAULT_TRANSCRIPT_PATH,
    benchmark_protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    project_root: str | Path = PROJECT_ROOT,
    env: Mapping[str, str] | None = None,
    runner: CommandRunner = run_command,
    command_lookup: CommandLookup = shutil.which,
    device_paths: Iterable[str | Path] | None = None,
    run_date: str = DEFAULT_RUN_DATE,
) -> dict[str, Any]:
    """Run the non-mutating preflight and persist all required artifacts."""

    write_in_progress_artifact(output_path)
    summary = probe_availability(
        project_root=project_root,
        env=env,
        runner=runner,
        command_lookup=command_lookup,
        device_paths=device_paths,
    )
    smoke_result = run_non_destructive_smoke(
        summary=summary,
        runner=runner,
        command_lookup=command_lookup,
    )
    write_transcript(path=hardware_transcript_path, summary=summary, smoke_result=smoke_result)
    artifact = build_artifact(
        summary=summary,
        smoke_result=smoke_result,
        hardware_transcript_path=hardware_transcript_path,
        benchmark_protocol_path=benchmark_protocol_path,
        run_date=run_date,
    )
    validate_artifact(artifact)
    write_benchmark_protocol(path=benchmark_protocol_path, artifact=artifact)
    return _write_json(output_path, artifact)


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run_preflight()
    print(
        json.dumps(
            {
                "wormhole_preflight_ready": artifact["wormhole_preflight_ready"],
                "wormhole_access_available": artifact["wormhole_access_available"],
                "blocked_reason": artifact["blocked_reason"],
                "honest_verdict": artifact["honest_verdict"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
