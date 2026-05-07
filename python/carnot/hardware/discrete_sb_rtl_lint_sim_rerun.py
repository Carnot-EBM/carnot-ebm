"""Exp 1451 bounded Discrete SB RTL lint/simulation rerun.

Exp 1437 stopped honestly because the planned Discrete SB RTL source was
missing.  Exp 1441 created that source and a smoke testbench.  This module is
the rerun gate between those two facts and any future KV260 work: it checks the
source prerequisite, probes local HDL tools, runs only source-level lint and
simulation commands, and keeps hardware claims disabled unless a real board run
is evidenced in the same artifact.

Spec refs: REQ-ISING-026, SCENARIO-ISING-036.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DATE = "20260507"
EXPERIMENT_ID = 1451
DEFAULT_EXP1441_PATH = (
    PROJECT_ROOT / "results" / "experiment_1441_discrete_sb_rtl_source_implementation.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "experiment_1451_discrete_sb_rtl_lint_sim_rerun.json"
)
SOURCE_REL = Path("hardware/kv260/discrete_sb_256.v")
TESTBENCH_REL = Path("hardware/kv260/discrete_sb_256_tb.v")
SIM_OUTPUT = "/tmp/discrete_sb_256_tb_1451.vvp"
SUMMARY_LIMIT = 800

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "rtl_source_present",
    "rtl_lint_complete",
    "simulation_complete",
    "tools_available",
    "lint_command",
    "simulation_command",
    "lint_errors",
    "simulation_errors",
    "hardware_claim_allowed",
    "commands_run",
    "honest_verdict",
}

TOOL_VERSION_COMMANDS: dict[str, list[str]] = {
    "verilator": ["verilator", "--version"],
    "iverilog": ["iverilog", "-V"],
    "yosys": ["yosys", "--version"],
    "vivado": ["vivado", "-version"],
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


def write_in_progress_artifact(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, str]:
    """Write the minimal startup marker before any inspection can fail.

    The research conductor treats result files as operational evidence.  A
    small `in_progress` marker makes interrupted runs distinguishable from
    never-started runs while avoiding fabricated terminal fields.
    """

    artifact = {"status": "in_progress"}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run_command(
    command: Sequence[str],
    *,
    cwd: str | Path,
    runner: Runner = subprocess.run,
    timeout: int = 30,
) -> dict[str, Any]:
    """Run one bounded command and return a compact JSON-safe transcript.

    HDL tools can print long banners, warnings, and simulator traces.  The
    artifact needs exact commands plus enough output to classify failures, so
    summaries are capped while return codes and error classes stay explicit.
    """

    cmd = [str(part) for part in command]
    try:
        completed = runner(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return _command_result(
            cmd,
            returncode=int(completed.returncode),
            stdout=str(completed.stdout or ""),
            stderr=str(completed.stderr or ""),
            error_class="none" if completed.returncode == 0 else "nonzero_exit",
            error="",
        )
    except FileNotFoundError as exc:
        return _command_result(
            cmd,
            returncode=None,
            stdout="",
            stderr="",
            error_class="not_found",
            error=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        return _command_result(
            cmd,
            returncode=None,
            stdout=str(exc.output or ""),
            stderr=str(exc.stderr or ""),
            error_class="timeout",
            error=f"timeout_after_{timeout}s",
        )
    except OSError as exc:
        return _command_result(
            cmd,
            returncode=None,
            stdout="",
            stderr="",
            error_class="os_error",
            error=str(exc),
        )


def run_experiment(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    exp1441_path: str | Path = DEFAULT_EXP1441_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run the gated Exp 1451 source-level lint and simulation flow."""

    root = Path(project_root)
    output = Path(output_path)
    write_in_progress_artifact(output)

    exp1441 = load_exp1441_summary(exp1441_path)
    rtl_source_present = (root / SOURCE_REL).is_file()
    testbench_present = (root / TESTBENCH_REL).is_file()
    tool_probes = probe_tools(root, runner=runner)
    commands_run = [probe["probe"] for probe in tool_probes.values()]
    tools_available = {tool: bool(probe["available"]) for tool, probe in tool_probes.items()}

    lint_command = ""
    simulation_command = ""
    lint_errors: list[dict[str, Any]] = []
    simulation_errors: list[dict[str, Any]] = []
    rtl_lint_complete = False
    simulation_complete = False
    source_confirmed = rtl_source_present and bool(exp1441["rtl_source_created"])

    if not source_confirmed:
        prerequisite_error = _manual_error(
            "prerequisite",
            "prerequisite_missing",
            "hardware/kv260/discrete_sb_256.v is absent or Exp 1441 did not report rtl_source_created=true",
        )
        lint_errors.append(prerequisite_error)
        simulation_errors.append(prerequisite_error)
    else:
        lint = choose_lint_command(tools_available, testbench_present)
        if lint:
            lint_command = _format_command(lint)
            lint_result = run_command(lint, cwd=root, runner=runner, timeout=30)
            commands_run.append(lint_result)
            rtl_lint_complete = lint_result["returncode"] == 0
            lint_errors.extend(_errors_from_result("rtl_lint", lint_result))
        else:
            lint_errors.append(_manual_error("rtl_lint", "no_lint_tool", "no local lint tool found"))

        sim_compile, sim_run = choose_simulation_commands(tools_available, testbench_present)
        if sim_compile and sim_run:
            simulation_command = f"{_format_command(sim_compile)} && {_format_command(sim_run)}"
            compile_result = run_command(sim_compile, cwd=root, runner=runner, timeout=60)
            commands_run.append(compile_result)
            simulation_errors.extend(_errors_from_result("simulation_compile", compile_result))
            if compile_result["returncode"] == 0:
                sim_result = run_command(sim_run, cwd=root, runner=runner, timeout=60)
                commands_run.append(sim_result)
                simulation_complete = sim_result["returncode"] == 0
                simulation_errors.extend(_errors_from_result("simulation", sim_result))
        elif not testbench_present:
            simulation_errors.append(
                _manual_error("simulation", "testbench_missing", TESTBENCH_REL.as_posix())
            )
        else:
            simulation_errors.append(
                _manual_error("simulation", "no_simulator", "no local iverilog simulator found")
            )

    honest_verdict = classify_verdict(
        source_confirmed=source_confirmed,
        lint_complete=rtl_lint_complete,
        simulation_complete=simulation_complete,
        lint_errors=lint_errors,
        simulation_errors=simulation_errors,
    )
    artifact = {
        "status": "complete",
        "run_date": run_date,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": 1441,
        "rtl_source_present": rtl_source_present,
        "testbench_present": testbench_present,
        "exp1441_rtl_source_created": bool(exp1441["rtl_source_created"]),
        "exp1441_summary": exp1441,
        "rtl_lint_complete": rtl_lint_complete,
        "simulation_complete": simulation_complete,
        "tools_available": tools_available,
        "tool_probe_results": tool_probes,
        "lint_command": lint_command,
        "simulation_command": simulation_command,
        "lint_errors": lint_errors,
        "simulation_errors": simulation_errors,
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
        "commands_run": commands_run,
        "honest_verdict": honest_verdict,
    }
    validate_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def load_exp1441_summary(path: str | Path) -> dict[str, Any]:
    """Load the source-creation artifact fields that gate this rerun."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    return {
        "path": Path(path).as_posix(),
        "exists": bool(artifact),
        "status": artifact.get("status"),
        "rtl_source_created": bool(artifact.get("rtl_source_created", False)),
        "rtl_source_path": artifact.get("rtl_source_path"),
        "testbench_created": bool(artifact.get("testbench_created", False)),
        "testbench_path": artifact.get("testbench_path"),
        "honest_verdict": artifact.get("honest_verdict"),
    }


def probe_tools(project_root: str | Path, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    """Probe HDL tools through version commands without installing anything."""

    probes: dict[str, Any] = {}
    for tool, command in TOOL_VERSION_COMMANDS.items():
        result = run_command(command, cwd=project_root, runner=runner, timeout=10)
        probes[tool] = {"available": result["returncode"] == 0, "probe": result}
    return probes


def choose_lint_command(
    tools_available: Mapping[str, bool],
    testbench_present: bool,
) -> list[str]:
    """Choose one narrow source-level lint command from available tools."""

    source = SOURCE_REL.as_posix()
    testbench = TESTBENCH_REL.as_posix()
    verilator_inputs = [source, testbench] if testbench_present else [source]
    if tools_available["verilator"]:
        return [
            "verilator",
            "--lint-only",
            "--timing",
            "-Wall",
            "-Wno-DECLFILENAME",
            "-Wno-BLKSEQ",
            "-Wno-WIDTH",
            *verilator_inputs,
        ]
    if tools_available["yosys"]:
        return ["yosys", "-q", "-p", f"read_verilog -sv {source}; hierarchy -check"]
    if tools_available["iverilog"]:
        return ["iverilog", "-tnull", "-g2012", source]
    return []


def choose_simulation_commands(
    tools_available: Mapping[str, bool],
    testbench_present: bool,
) -> tuple[list[str], list[str]]:
    """Return Icarus compile/run commands when a local simulator is available."""

    if tools_available["iverilog"] and testbench_present:
        compile_cmd = [
            "iverilog",
            "-g2012",
            "-o",
            SIM_OUTPUT,
            SOURCE_REL.as_posix(),
            TESTBENCH_REL.as_posix(),
        ]
        return compile_cmd, ["vvp", SIM_OUTPUT]
    return [], []


def classify_verdict(
    *,
    source_confirmed: bool,
    lint_complete: bool,
    simulation_complete: bool,
    lint_errors: Sequence[Mapping[str, Any]],
    simulation_errors: Sequence[Mapping[str, Any]],
) -> str:
    """Convert observed evidence into a claim-safe terminal verdict."""

    if not source_confirmed:
        return "blocked_missing_or_unconfirmed_discrete_sb_rtl_source_no_hardware_claim"
    if _has_error(lint_errors, "no_lint_tool") and _has_error(simulation_errors, "no_simulator"):
        return "blocked_no_local_lint_or_simulation_tool_no_hardware_claim"
    if lint_complete and simulation_complete:
        return "rtl_lint_and_simulation_complete_no_hardware_execution_no_kv260_claim"
    if lint_complete:
        return "rtl_lint_complete_simulation_not_complete_no_hardware_execution_no_kv260_claim"
    return "blocked_rtl_lint_and_simulation_failed_no_hardware_claim"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal schema and hardware-claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    if artifact.get("hardware_execution_performed") is True:
        raise ValueError("hardware_execution_performed requires real KV260 board evidence")
    if artifact["hardware_claim_allowed"] is not False:
        raise ValueError("hardware_claim_allowed must remain false without KV260 board evidence")


def _command_result(
    command: Sequence[str],
    *,
    returncode: int | None,
    stdout: str,
    stderr: str,
    error_class: str,
    error: str,
) -> dict[str, Any]:
    return {
        "command": [str(part) for part in command],
        "command_string": _format_command(command),
        "returncode": returncode,
        "stdout_summary": _summarize(stdout),
        "stderr_summary": _summarize(stderr),
        "error_class": error_class,
        "error": error,
    }


def _errors_from_result(stage: str, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    if result["returncode"] == 0:
        return []
    return [
        {
            "stage": stage,
            "error_class": result["error_class"],
            "returncode": result["returncode"],
            "stdout_summary": result["stdout_summary"],
            "stderr_summary": result["stderr_summary"],
            "error": result["error"],
        }
    ]


def _manual_error(stage: str, error_class: str, detail: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_class": error_class,
        "returncode": None,
        "stdout_summary": "",
        "stderr_summary": detail,
        "error": detail,
    }


def _has_error(errors: Sequence[Mapping[str, Any]], error_class: str) -> bool:
    return any(error.get("error_class") == error_class for error in errors)


def _format_command(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def _summarize(text: str) -> str:
    summary = str(text or "").strip()
    if len(summary) <= SUMMARY_LIMIT:
        return summary
    return summary[:SUMMARY_LIMIT].rstrip() + "\n...[truncated]"
