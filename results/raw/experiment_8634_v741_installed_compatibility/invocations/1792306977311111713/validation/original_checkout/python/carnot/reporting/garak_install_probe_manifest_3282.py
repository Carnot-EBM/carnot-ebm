"""Build the Exp 3282 Garak install/probe manifest.

Spec refs: REQ-REPORT-3282, SCENARIO-REPORT-3282.

This module does only toolchain probing. It checks import, CLI, and probe
inventory surfaces, then records how downstream work should connect Garak to a
local llama.cpp server for GGUF targets. It deliberately does not start
llama.cpp, load a GGUF, call a hosted model, or run a full Garak benchmark.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]
RunnerFn = Callable[[Sequence[str], int], "CommandResult"]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.garak_install_probe_manifest.v1"
EXPERIMENT_ID = "exp3282"
TASK_ID = "exp3282-garak-install-and-probe-manifest-v1"
ARTIFACT = "experiment_3282_garak_install_and_probe_manifest_v1"
MILESTONE = "2026.05.304"
RUN_DATE = "20260528"
RANDOM_SEED = 3282

OUTPUT_REL_PATH = Path("results/experiment_3282_garak_install_and_probe_manifest_v1.json")
EXP3274_REL_PATH = Path(
    "results/experiment_3274_prompt_injection_v4_garak_dataflip_redteam_eval_v1.json"
)
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3282_garak_install_and_probe_manifest_v1.py"

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
IMPORT_MARKER = "GARAK_IMPORT_PROBE_V1"
INVENTORY_MARKER = "GARAK_PROBE_INVENTORY_V1"

MANDATED_TARGET_MODELS = (
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)

REQUIRED_ARTIFACT_FIELDS = {
    "garak_install_probe_manifest_ready",
    "garak_runner_ready",
    "garak_available",
    "garak_version",
    "garak_import_command",
    "garak_cli_command",
    "probe_inventory",
    "promptinject_probe_count",
    "local_target_adapter_plan",
    "install_blockers",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}

STATIC_PROBE_FAMILIES = (
    (
        "promptinject",
        "garak.probes.promptinject",
        ("HijackHateHumans", "HijackKillHumans", "HijackLongPrompt"),
    ),
    ("jailbreak", "garak.probes.dan", ("DAN", "DANInTheWild", "DANJailbreak")),
    ("encoding", "garak.probes.encoding", ("InjectBase64", "InjectHex", "InjectROT13")),
    ("leakage", "garak.probes.leakreplay", ("ReplayLeak", "PromptLeak", "SystemLeak")),
    ("hallucination", "garak.probes.snowball", ("Snowball", "MisleadingFalsehood")),
)

GARAK_IMPORT_CODE = (
    f"import json, importlib.metadata; import garak; print(json.dumps({{'marker': "
    f"'{IMPORT_MARKER}', 'version': importlib.metadata.version('garak')}}))"
)

GARAK_INVENTORY_CODE = r"""
import importlib
import inspect
import json
import pkgutil

import garak.probes as probes

INTERESTING = ("promptinject", "jailbreak", "dan", "encoding", "leak", "hallucination", "snowball")
inventory = []
for module_info in pkgutil.walk_packages(probes.__path__, probes.__name__ + "."):
    module_name = module_info.name
    if not any(token in module_name.lower() for token in INTERESTING):
        continue
    try:
        module = importlib.import_module(module_name)
    except Exception:
        inventory.append({"module": module_name, "classes": []})
        continue
    classes = [
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if getattr(obj, "__module__", "") == module.__name__
    ]
    inventory.append({"module": module_name, "classes": sorted(classes)})
print(json.dumps(inventory, sort_keys=True))
""".strip() + f"\n# {INVENTORY_MARKER}"


@dataclass(frozen=True)
class CommandResult:
    """Small subprocess result shape that is easy to fake in tests."""

    returncode: int
    stdout: str
    stderr: str


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    output_path: str | Path = OUTPUT_REL_PATH,
    runner: RunnerFn = None,  # type: ignore[assignment]
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """REQ-REPORT-3282: write a stable Garak toolchain contract artifact."""

    start = monotonic()
    root = Path(project_root)
    command_runner = runner or run_command
    prior_exp3274 = read_json_object(root / EXP3274_REL_PATH)

    project_import = probe_import(project_import_command(root), command_runner, "project_garak_import")
    project_cli = probe_cli(("garak", "--version"), command_runner, "project_garak_cli")
    checks = [prior_exp3274_check(prior_exp3274), project_import, project_cli]

    selected_import = project_import
    selected_cli = project_cli
    if not command_check_passed(project_import, project_cli):
        uv_checks = isolated_uv_checks(command_runner)
        checks.extend(uv_checks)
        if len(uv_checks) == 2 and command_check_passed(uv_checks[0], uv_checks[1]):
            selected_import, selected_cli = uv_checks

    inventory_check, probe_inventory = probe_inventory_for(selected_import, command_runner)
    checks.append(inventory_check)
    promptinject_probe_count = count_promptinject_available(probe_inventory)
    garak_runner_ready = command_check_passed(selected_import, selected_cli) and promptinject_probe_count > 0
    garak_available = command_check_passed(selected_import, selected_cli)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "garak_install_probe_manifest_ready": True,
        "garak_runner_ready": garak_runner_ready,
        "garak_available": garak_available,
        "garak_version": garak_version(selected_import, selected_cli),
        "garak_import_command": str(selected_import.get("command") or ""),
        "garak_cli_command": str(selected_cli.get("command") or ""),
        "probe_inventory": probe_inventory,
        "promptinject_probe_count": promptinject_probe_count,
        "local_target_adapter_plan": local_target_adapter_plan(),
        "install_blockers": install_blockers(garak_available, garak_runner_ready, checks),
        "preconditions_checked": checks,
        "output_paths": [path_as_artifact_string(output_path), EXP3274_REL_PATH.as_posix()],
        "no_full_garak_benchmark_run": True,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": duration(start, monotonic()),
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)

    out_path = resolve_output_path(root, output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def probe_import(command: Sequence[str], runner: RunnerFn, name: str) -> JsonDict:
    """Run an import command and normalize the result into a precondition row."""

    result = runner(command, 120)
    version = parse_import_probe_stdout(result.stdout) if result.returncode == 0 else ""
    return command_row(
        name=name,
        command=command,
        result=result,
        passed=result.returncode == 0 and bool(version),
        version=version,
        blocked_reason="blocked_garak_import_unavailable",
    )


def probe_cli(command: Sequence[str], runner: RunnerFn, name: str) -> JsonDict:
    """Run a Garak CLI version command and normalize the result."""

    result = runner(command, 120)
    version = first_nonempty_line(result.stdout, result.stderr) if result.returncode == 0 else ""
    return command_row(
        name=name,
        command=command,
        result=result,
        passed=result.returncode == 0,
        version=version,
        blocked_reason="blocked_garak_cli_unavailable",
    )


def isolated_uv_checks(runner: RunnerFn) -> list[JsonDict]:
    """Try the isolated uv path only when uv is present on this machine."""

    if not shutil.which("uv"):
        return []
    return [
        probe_import(isolated_import_command(), runner, "isolated_uv_garak_import"),
        probe_cli(isolated_cli_command(), runner, "isolated_uv_garak_cli"),
    ]


def probe_inventory_for(selected_import: Mapping[str, Any], runner: RunnerFn) -> tuple[JsonDict, list[JsonDict]]:
    """Inventory relevant Garak probes through the same command path as import."""

    inventory_command = inventory_command_from_import(selected_import)
    if not inventory_command:
        row = command_row(
            name="garak_probe_inventory",
            command=("<unavailable>",),
            result=CommandResult(1, "", "no successful Garak import command"),
            passed=False,
            version="",
            blocked_reason="blocked_probe_inventory_unavailable",
        )
        return row, static_probe_inventory()

    result = runner(inventory_command, 120)
    live_inventory = parse_probe_inventory(result.stdout) if result.returncode == 0 else []
    passed = bool(live_inventory)
    row = command_row(
        name="garak_probe_inventory",
        command=inventory_command,
        result=result,
        passed=passed,
        version="",
        blocked_reason="blocked_probe_inventory_unavailable",
    )
    return row, live_inventory if passed else static_probe_inventory()


def command_row(
    *,
    name: str,
    command: Sequence[str],
    result: CommandResult,
    passed: bool,
    version: str,
    blocked_reason: str,
) -> JsonDict:
    """Create a reproducible command precondition with terse stderr evidence."""

    return {
        "name": name,
        "passed": bool(passed),
        "command": command_to_string(command),
        "returncode": int(result.returncode),
        "stdout_summary": stderr_summary(result.stdout),
        "stderr_summary": stderr_summary(result.stderr),
        "version": version,
        "blocked_reason": "" if passed else blocked_reason,
    }


def prior_exp3274_check(prior_exp3274: Mapping[str, Any]) -> JsonDict:
    """Preserve the Exp 3274 Garak availability field as a precondition."""

    path = EXP3274_REL_PATH.as_posix()
    return {
        "name": "prior_exp3274_garak_available",
        "passed": prior_exp3274.get("garak_available") is True,
        "path": path,
        "exists": bool(prior_exp3274),
        "garak_available": bool(prior_exp3274.get("garak_available")),
        "blocked_reasons": list(prior_exp3274.get("blocked_reasons") or []),
    }


def install_blockers(
    garak_available: bool,
    garak_runner_ready: bool,
    checks: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return active install blockers, not resolved fallback attempts."""

    if garak_runner_ready:
        return []
    active_names = {"garak_probe_inventory"}
    if not garak_available:
        active_names.update({"project_garak_import", "project_garak_cli"})
        if any(str(check.get("name", "")).startswith("isolated_uv_") for check in checks):
            active_names.update({"isolated_uv_garak_import", "isolated_uv_garak_cli"})

    blockers: list[JsonDict] = []
    for check in checks:
        if check.get("passed") is True or check.get("name") not in active_names:
            continue
        blockers.append(
            {
                "reason": str(check.get("blocked_reason") or "blocked_garak_toolchain"),
                "command": str(check.get("command") or check.get("path") or ""),
                "returncode": int(check.get("returncode", 1)),
                "stderr_summary": str(check.get("stderr_summary") or ""),
                "next_action": next_action_for(str(check.get("blocked_reason") or "")),
            }
        )
    return blockers


def next_action_for(blocked_reason: str) -> str:
    """Map a blocker to a concrete operator action for the next milestone."""

    if "inventory" in blocked_reason:
        return "rerun the probe inventory command after Garak import and CLI pass"
    if "cli" in blocked_reason:
        return "install or expose the Garak console script, then rerun garak --version"
    if "import" in blocked_reason:
        return "install Garak in the project environment or use uv run --no-project --with garak"
    return "inspect the command row and rerun the failed toolchain probe"


def static_probe_inventory() -> list[JsonDict]:
    """Return expected probe families when live Garak inventory is blocked."""

    rows: list[JsonDict] = []
    for family, module, classes in STATIC_PROBE_FAMILIES:
        for class_name in classes:
            rows.append(
                {
                    "family": family,
                    "module": module,
                    "class_name": class_name,
                    "probe": f"{module}.{class_name}",
                    "available": False,
                    "source": "static_expected_recheck",
                    "relevance": relevance_for_family(family),
                }
            )
    return rows


def parse_probe_inventory(stdout: str) -> list[JsonDict]:
    """Parse live probe inventory JSON into relevant prompt-injection families."""

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    rows: list[JsonDict] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        module = str(item.get("module") or "")
        classes = item.get("classes")
        if not isinstance(classes, list):
            continue
        for class_name_raw in classes:
            class_name = str(class_name_raw)
            family = classify_probe_family(module, class_name)
            if family == "other":
                continue
            rows.append(
                {
                    "family": family,
                    "module": module,
                    "class_name": class_name,
                    "probe": f"{module}.{class_name}",
                    "available": True,
                    "source": "live_garak_inventory",
                    "relevance": relevance_for_family(family),
                }
            )
    return sorted(rows, key=lambda row: (row["family"], row["module"], row["class_name"]))


def classify_probe_family(module: str, class_name: str) -> str:
    """Classify Garak probe names into the families required by Exp 3282."""

    text = f"{module}.{class_name}".lower()
    if "promptinject" in text:
        return "promptinject"
    if any(token in text for token in ("jailbreak", "dan", "doanything", "grandma")):
        return "jailbreak"
    if any(token in text for token in ("encoding", "base64", "rot13", "unicode", "hex")):
        return "encoding"
    if any(token in text for token in ("leak", "lmrc", "replay", "systemprompt")):
        return "leakage"
    if any(token in text for token in ("hallucination", "snowball", "misleading")):
        return "hallucination"
    return "other"


def relevance_for_family(family: str) -> str:
    """Explain why a probe family belongs in a prompt-injection readiness manifest."""

    return {
        "promptinject": "direct prompt-injection and instruction-hijack probes",
        "jailbreak": "policy-override probes that pressure instruction hierarchy",
        "encoding": "obfuscated instruction probes for encoded prompt injection",
        "leakage": "secret/system-prompt leakage probes triggered by hostile instructions",
        "hallucination": "false-claim probes that expose unsafe compliance under attack",
    }.get(family, "related probe family")


def local_target_adapter_plan() -> JsonDict:
    """Describe the downstream local GGUF adapter without launching a model."""

    probe_spec = "promptinject,dan,encoding,leakreplay,snowball,packagehallucination"
    generator_options = {
        "openai": {
            "OpenAICompatible": {
                "uri": "http://127.0.0.1:8080/v1",
                "max_tokens": 256,
                "temperature": 0.0,
            }
        }
    }
    return {
        "adapter_kind": "llama_cpp_openai_compatible_rest",
        "mandated_targets": list(MANDATED_TARGET_MODELS),
        "does_not_run_model": True,
        "llama_cpp_server_command": [
            "llama-server",
            "--model",
            "<GGUF_PATH>",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--ctx-size",
            "8192",
            "--n-gpu-layers",
            "-1",
        ],
        "llama_cpp_server_healthcheck": "curl -fsS http://127.0.0.1:8080/v1/models",
        "openai_compatible_base_url": "http://127.0.0.1:8080/v1",
        "openai_compatible_env": {
            "OPENAICOMPATIBLE_API_KEY": "garak-local-placeholder",
        },
        "garak_probe_spec": probe_spec,
        "probe_family_to_garak_modules": {
            "promptinject": ["promptinject"],
            "jailbreak": ["dan", "visual_jailbreak"],
            "encoding": ["encoding"],
            "leakage": ["leakreplay"],
            "hallucination": ["snowball", "packagehallucination"],
        },
        "generator_options_json": generator_options,
        "garak_generator_command": (
            "garak --target_type openai.OpenAICompatible --target_name <MODEL_ID> "
            "--generator_options "
            + shlex.quote(json.dumps(generator_options, sort_keys=True, separators=(",", ":")))
            + f" --probes {probe_spec}"
        ),
        "smoke_eval_handoff": (
            "run one low-generation promptinject probe against the llama.cpp OpenAI-compatible "
            "server before any full red-team sweep"
        ),
        "full_eval_handoff": (
            "reuse the same target_type/target_name/options contract, then expand probes and "
            "generations only after the smoke artifact passes"
        ),
    }


def count_promptinject_available(probe_inventory: Sequence[Mapping[str, Any]]) -> int:
    """Count live PromptInject probes so readiness is explicit."""

    return sum(
        1
        for row in probe_inventory
        if row.get("family") == "promptinject" and row.get("available") is True
    )


def garak_version(import_check: Mapping[str, Any], cli_check: Mapping[str, Any]) -> str:
    """Prefer the import package version, falling back to the CLI version line."""

    return str(import_check.get("version") or cli_check.get("version") or "")


def command_check_passed(import_check: Mapping[str, Any], cli_check: Mapping[str, Any]) -> bool:
    """Return true when import and CLI checks both passed."""

    return import_check.get("passed") is True and cli_check.get("passed") is True


def project_import_command(root: Path) -> tuple[str, ...]:
    """Use the project Python when present so the import command is reproducible."""

    return (project_python(root), "-c", GARAK_IMPORT_CODE)


def project_python(root: Path) -> str:
    """Resolve the local project interpreter with a sys.executable fallback."""

    candidate = root / ".venv" / "bin" / "python"
    return candidate.as_posix() if candidate.exists() else sys.executable


def isolated_import_command() -> tuple[str, ...]:
    """Return the non-mutating uv command for importing Garak in an isolated env."""

    return ("uv", "run", "--no-project", "--with", "garak", "python", "-c", GARAK_IMPORT_CODE)


def isolated_cli_command() -> tuple[str, ...]:
    """Return the non-mutating uv command for invoking Garak's CLI."""

    return ("uv", "run", "--no-project", "--with", "garak", "garak", "--version")


def inventory_command_from_import(import_check: Mapping[str, Any]) -> tuple[str, ...]:
    """Reuse project or uv command shape for probe inventory."""

    if import_check.get("passed") is not True:
        return ()
    command = str(import_check.get("command") or "")
    if command.startswith("uv run --no-project --with garak"):
        return ("uv", "run", "--no-project", "--with", "garak", "python", "-c", GARAK_INVENTORY_CODE)
    return (command.split(" -c ", 1)[0], "-c", GARAK_INVENTORY_CODE)


def run_command(command: Sequence[str], timeout_s: int) -> CommandResult:
    """Run a command without raising; callers store stderr as artifact evidence."""

    try:
        completed = subprocess.run(
            list(command),
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - host dependent
        return CommandResult(returncode=127, stdout="", stderr=f"{type(exc).__name__}: {exc}")
    return CommandResult(
        returncode=int(completed.returncode),
        stdout=str(completed.stdout or ""),
        stderr=str(completed.stderr or ""),
    )


def parse_import_probe_stdout(stdout: str) -> str:
    """Extract a package version from the import probe output."""

    stripped = stdout.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(payload, Mapping):
        return str(payload.get("version") or "")
    return stripped


def first_nonempty_line(*values: str) -> str:
    """Return the first non-empty stdout/stderr line from a version command."""

    for value in values:
        for line in value.splitlines():
            if line.strip():
                return line.strip()
    return ""


def stderr_summary(value: str, *, limit: int = 400) -> str:
    """Collapse command output into a short artifact-safe summary."""

    compact = " ".join(value.strip().split())
    return compact[: max(0, limit - 3)] + "..." if len(compact) > limit else compact


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict consumed by conductor and matrix tasks."""

    if artifact.get("garak_runner_ready") is True:
        return (
            "complete: garak_install_probe_manifest_ready=true; "
            "garak_runner_ready=true; "
            f"promptinject_probe_count={artifact.get('promptinject_probe_count')}"
        )
    reasons = ",".join(item["reason"] for item in artifact.get("install_blockers", [])) or "none"
    return (
        "complete: garak_install_probe_manifest_ready=true; "
        "garak_runner_ready=false; "
        f"garak_available={str(artifact.get('garak_available')).lower()}; "
        f"install_blockers={reasons}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fields that downstream Garak smoke tasks depend on."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if int(artifact.get("promptinject_probe_count", -1)) < 0:
        raise ValueError("promptinject_probe_count must be non-negative")
    if not isinstance(artifact.get("probe_inventory"), list):
        raise ValueError("probe_inventory must be a list")  # pragma: no cover
    if not isinstance(artifact.get("local_target_adapter_plan"), Mapping):
        raise ValueError("local_target_adapter_plan must be a dict")  # pragma: no cover
    if not isinstance(artifact.get("install_blockers"), list):
        raise ValueError("install_blockers must be a list")  # pragma: no cover
    if not isinstance(artifact.get("preconditions_checked"), list):
        raise ValueError("preconditions_checked must be a list")  # pragma: no cover


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while ignoring runtime-only fields."""

    stable = json.loads(json.dumps(artifact, sort_keys=True, default=str))
    stable["reproducibility_checksum"] = ""
    stable["honest_verdict"] = ""
    stable["duration_s"] = 0.0
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def duration(start: float, end: float) -> float:
    """Return non-negative rounded duration for artifact timing evidence."""

    return round(max(0.0, float(end) - float(start)), 6)


def command_to_string(command: Sequence[str]) -> str:
    """Render a command in a shell-copyable form for the artifact."""

    return " ".join(shlex.quote(str(part)) for part in command)


def path_as_artifact_string(path: str | Path) -> str:
    """Preserve relative artifact paths for downstream matrix tasks."""

    return Path(path).as_posix()


def resolve_output_path(root: Path, path: str | Path) -> Path:
    """Resolve a relative output path under the supplied project root."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def read_json_object(path: Path) -> JsonDict:
    """Read JSON object input, returning an empty object for missing artifacts."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def main() -> int:  # pragma: no cover
    artifact = run_experiment(project_root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
