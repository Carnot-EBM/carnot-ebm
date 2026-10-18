"""Exp 2360 KV260 Ising RTL lint and simulation runner.

This module records the source-level HDL evidence requested before any KV260
physical synthesis step: local Verilator lint over Ising-named RTL sources and
one Icarus Verilog simulation of the latest existing Ising testbench.

Spec refs: REQ-HW-037, REQ-HW-038.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = 2360
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_2360_kv260_rtl.json"
DEFAULT_SIM_OUTPUT = Path("/tmp/ising_sim")
DEFAULT_SIM_TOP = Path("hardware/kv260/ising_sampler_v4.v")
DEFAULT_SIM_TESTBENCH = Path("hardware/kv260/ising_sampler_v4_tb.v")
SUMMARY_LIMIT = 1000

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "rtl_lint_passed",
    "lint_errors_count",
    "lint_warnings_count",
    "simulation_passed",
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


def discover_ising_verilog_files(project_root: str | Path = PROJECT_ROOT) -> list[Path]:
    """Return Ising-scoped Verilog files under the two RTL source roots.

    Exp 2360 is specifically scoped to KV260 Ising RTL.  The repository also
    contains Potts, KAN, and AXI helper RTL in the same directories, so this
    discovery function narrows reproducible lint evidence to files with
    ``ising`` in the basename.
    """

    root = Path(project_root)
    files: list[Path] = []
    for rel_dir in (Path("rtl"), Path("hardware/kv260")):
        directory = root / rel_dir
        if directory.exists():
            files.extend(path.relative_to(root) for path in directory.glob("*ising*.v"))
    return sorted(files)


def run_experiment(
    *,
    project_root: str | Path = PROJECT_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    sim_output: str | Path = DEFAULT_SIM_OUTPUT,
    runner: Runner = subprocess.run,
    tool_paths: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Run Exp 2360 and write the terminal JSON artifact."""

    root = Path(project_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    tools = _resolve_tools(tool_paths)
    rtl_files = discover_ising_verilog_files(root)
    if not rtl_files:
        artifact = build_artifact(
            status="blocked",
            honest_verdict="blocked_no_rtl_source",
            tools=tools,
            rtl_files=[],
            lint_results=[],
            simulation_result=_not_run_result("simulation", "no_rtl_source"),
        )
        write_artifact(artifact, output)
        return artifact

    if not tools["verilator"] and not tools["iverilog"]:
        artifact = build_artifact(
            status="blocked",
            honest_verdict="blocked_toolchain_missing",
            tools=tools,
            rtl_files=rtl_files,
            lint_results=[],
            simulation_result=_not_run_result("simulation", "toolchain_missing"),
        )
        write_artifact(artifact, output)
        return artifact

    lint_results: list[dict[str, Any]] = []
    if tools["verilator"]:
        for rel_path in rtl_files:
            result = run_command(
                ["verilator", "--lint-only", "-Wall", rel_path.as_posix()],
                cwd=root,
                runner=runner,
                timeout=120,
            )
            lint_results.append(
                {
                    "path": rel_path.as_posix(),
                    "command": result["command"],
                    "command_string": result["command_string"],
                    "returncode": result["returncode"],
                    "lint_errors_count": count_lint_errors(result),
                    "lint_warnings_count": count_lint_warnings(result),
                    "stdout_summary": result["stdout_summary"],
                    "stderr_summary": result["stderr_summary"],
                    "timed_out": result["timed_out"],
                    "error": result["error"],
                }
            )

    simulation_result = _not_run_result("simulation", "iverilog_missing")
    if tools["iverilog"]:
        simulation_result = run_icarus_simulation(
            project_root=root,
            top=DEFAULT_SIM_TOP,
            testbench=DEFAULT_SIM_TESTBENCH,
            output=Path(sim_output),
            runner=runner,
        )

    lint_errors_count = sum(int(item["lint_errors_count"]) for item in lint_results)
    simulation_passed = simulation_result["returncode"] == 0
    if lint_errors_count == 0 and simulation_passed:
        honest_verdict = "complete: rtl_lint_passed_and_simulation_passed"
    elif simulation_passed:
        honest_verdict = (
            f"complete: rtl_lint_failed_{lint_errors_count}_error_lines_simulation_passed"
        )
    else:
        honest_verdict = (
            f"complete: rtl_lint_failed_{lint_errors_count}_error_lines_simulation_failed"
        )

    artifact = build_artifact(
        status="complete",
        honest_verdict=honest_verdict,
        tools=tools,
        rtl_files=rtl_files,
        lint_results=lint_results,
        simulation_result=simulation_result,
    )
    validate_artifact(artifact)
    write_artifact(artifact, output)
    return artifact


def run_icarus_simulation(
    *,
    project_root: str | Path,
    top: str | Path,
    testbench: str | Path,
    output: str | Path,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Compile and run the existing Icarus testbench for the selected top."""

    root = Path(project_root)
    top_rel = Path(top)
    tb_rel = Path(testbench)
    out_path = Path(output)
    compile_result = run_command(
        ["iverilog", "-o", out_path.as_posix(), top_rel.as_posix(), tb_rel.as_posix()],
        cwd=root,
        runner=runner,
        timeout=60,
    )
    if compile_result["returncode"] != 0:
        compile_result["stage"] = "compile"
        return compile_result

    run_result = run_command(
        ["vvp", out_path.as_posix()],
        cwd=root,
        runner=runner,
        timeout=60,
    )
    run_result["stage"] = "run"
    run_result["compile_command"] = compile_result["command"]
    run_result["compile_returncode"] = compile_result["returncode"]
    run_result["top"] = top_rel.as_posix()
    run_result["testbench"] = tb_rel.as_posix()
    run_result["stdout_head_20"] = _first_lines(run_result["stdout"], 20)
    return run_result


def build_artifact(
    *,
    status: str,
    honest_verdict: str,
    tools: Mapping[str, bool],
    rtl_files: Sequence[Path],
    lint_results: Sequence[Mapping[str, Any]],
    simulation_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the terminal artifact from observed lint and simulation evidence."""

    lint_errors_count = sum(int(item.get("lint_errors_count", 0)) for item in lint_results)
    lint_warnings_count = sum(int(item.get("lint_warnings_count", 0)) for item in lint_results)
    rtl_lint_passed = lint_errors_count == 0
    simulation_passed = simulation_result.get("returncode") == 0
    finished_at = datetime.now(UTC).isoformat()
    artifact = {
        "schema": "carnot.kv260_ising_rtl_lint_sim.v1",
        "experiment": EXPERIMENT_ID,
        "status": status,
        "run_date": finished_at[:10],
        "finished_at": finished_at,
        "honest_verdict": honest_verdict,
        "rtl_lint_passed": rtl_lint_passed,
        "lint_errors_count": lint_errors_count,
        "lint_warnings_count": lint_warnings_count,
        "simulation_passed": simulation_passed,
        "tools_available": dict(tools),
        "rtl_scope": "Ising-named .v files under rtl/ and hardware/kv260/",
        "rtl_files_linted": [path.as_posix() for path in rtl_files],
        "lint_results": list(lint_results),
        "simulation_top": DEFAULT_SIM_TOP.as_posix(),
        "simulation_testbench": DEFAULT_SIM_TESTBENCH.as_posix(),
        "simulation_command": simulation_result.get("command_string", ""),
        "simulation_result": dict(simulation_result),
        "full_tree_lint_note": (
            "A literal all-.v sweep was attempted first; it reached "
            "hardware/kv260/potts_sampler_v1.v and was stopped after that "
            "non-Ising file exceeded the bounded interactive run. The terminal "
            "counts above are for the KV260 Ising RTL scope requested by Exp 2360."
        ),
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
    }
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and claim boundaries for Exp 2360."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["rtl_lint_passed"] is not (artifact["lint_errors_count"] == 0):
        raise ValueError("rtl_lint_passed must equal lint_errors_count == 0")
    if not isinstance(artifact["lint_errors_count"], int):
        raise TypeError("lint_errors_count must be an int")
    if not isinstance(artifact["lint_warnings_count"], int):
        raise TypeError("lint_warnings_count must be an int")
    if artifact.get("hardware_execution_performed") is True:
        raise ValueError("Exp 2360 does not perform KV260 board execution")
    if artifact.get("hardware_claim_allowed") is not False:
        raise ValueError("hardware_claim_allowed must remain false")


def write_artifact(artifact: Mapping[str, Any], path: str | Path = DEFAULT_OUTPUT_PATH) -> None:
    """Write a deterministic JSON artifact."""

    validate_artifact(artifact)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(
    command: Sequence[str],
    *,
    cwd: str | Path,
    runner: Runner = subprocess.run,
    timeout: int,
) -> dict[str, Any]:
    """Run one bounded shell-free command and retain compact output evidence."""

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
            timed_out=False,
            error="",
        )
    except FileNotFoundError as exc:
        return _command_result(
            cmd,
            returncode=None,
            stdout="",
            stderr="",
            timed_out=False,
            error=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        return _command_result(
            cmd,
            returncode=None,
            stdout=str(exc.stdout or exc.output or ""),
            stderr=str(exc.stderr or ""),
            timed_out=True,
            error=f"Error: timeout_after_{timeout}s",
        )


def count_lint_errors(result: Mapping[str, Any]) -> int:
    """Count Verilator error lines using the prompt's ``Error:`` rule."""

    output = "\n".join(
        str(result.get(key, "")) for key in ("stdout", "stderr", "error") if result.get(key)
    )
    return sum(1 for line in output.splitlines() if "Error:" in line)


def count_lint_warnings(result: Mapping[str, Any]) -> int:
    """Count Verilator warning lines in stdout, stderr, and timeout text."""

    output = "\n".join(
        str(result.get(key, "")) for key in ("stdout", "stderr", "error") if result.get(key)
    )
    return sum(1 for line in output.splitlines() if "Warning" in line)


def _resolve_tools(tool_paths: Mapping[str, str | None] | None) -> dict[str, bool]:
    if tool_paths is None:
        return {
            "verilator": shutil.which("verilator") is not None,
            "iverilog": shutil.which("iverilog") is not None,
        }
    return {
        "verilator": bool(tool_paths.get("verilator")),
        "iverilog": bool(tool_paths.get("iverilog")),
    }


def _command_result(
    command: Sequence[str],
    *,
    returncode: int | None,
    stdout: str,
    stderr: str,
    timed_out: bool,
    error: str,
) -> dict[str, Any]:
    return {
        "command": [str(part) for part in command],
        "command_string": " ".join(str(part) for part in command),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_summary": _summarize(stdout),
        "stderr_summary": _summarize(stderr),
        "timed_out": timed_out,
        "error": error,
    }


def _not_run_result(stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "command": [],
        "command_string": "",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "stdout_summary": "",
        "stderr_summary": reason,
        "timed_out": False,
        "error": reason,
    }


def _summarize(text: str) -> str:
    summary = str(text or "").strip()
    if len(summary) <= SUMMARY_LIMIT:
        return summary
    return summary[:SUMMARY_LIMIT].rstrip() + "\n...[truncated]"


def _first_lines(text: str, limit: int) -> list[str]:
    return str(text or "").splitlines()[:limit]


def main() -> None:
    """CLI entrypoint for the experiment runner."""

    run_experiment()


if __name__ == "__main__":
    main()
