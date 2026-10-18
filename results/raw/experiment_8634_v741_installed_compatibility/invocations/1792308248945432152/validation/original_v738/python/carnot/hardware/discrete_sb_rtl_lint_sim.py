"""Exp 1437 bounded RTL lint/simulation runner for Discrete SB KV260.

Exp 1422 documented the Discrete Simulated Bifurcation RTL plan, but it did
not leave behind a Verilog source, simulation transcript, bitfile, or board
run.  This module records the next honest local hardware step: inspect the
planned source path, probe local RTL tools without installing anything, run the
cheapest bounded lint/syntax/simulation command only when that is possible, and
keep hardware claims disabled unless real KV260 evidence exists.

Spec refs: REQ-ISING-024, SCENARIO-ISING-034.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DATE = "20260506"
EXPERIMENT_ID = 1437
DEFAULT_EXP1422_PATH = PROJECT_ROOT / "results" / "experiment_1422_discrete_sb_kv260_rtl_spec.json"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "experiment_1437_discrete_sb_kv260_rtl_lint_sim.json"
)
EXPECTED_SOURCE_REL = Path("hardware/kv260/discrete_sb_256.v")
EXPECTED_TESTBENCH_REL = Path("hardware/kv260/discrete_sb_256_tb.v")
SUMMARY_LIMIT = 800

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "rtl_sources_checked",
    "rtl_lint_complete",
    "simulation_complete",
    "synthesis_attempted",
    "yosys_available",
    "verilator_available",
    "vivado_available",
    "hardware_execution_performed",
    "hardware_claim_allowed",
    "next_bitfile_step",
    "honest_verdict",
}

TOOL_VERSION_COMMANDS: dict[str, list[str]] = {
    "yosys": ["yosys", "--version"],
    "verilator": ["verilator", "--version"],
    "iverilog": ["iverilog", "-V"],
    "vivado": ["vivado", "-version"],
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


def write_in_progress_artifact(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Write the bootstrap marker before source and tool inspection.

    The conductor may be interrupted between startup and the final write.  A
    visible `in_progress` artifact makes that interruption auditable instead of
    leaving operators to infer whether Exp 1437 ever started.
    """

    artifact = {
        "status": "in_progress",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": "in_progress",
    }
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
    """Run a bounded subprocess and return a compact JSON-safe transcript.

    RTL tools can produce very large logs.  The artifact needs enough evidence
    to explain a pass or blocker, not a full transcript, so stdout and stderr
    are summarized while the exact command and return code stay intact.
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
            timed_out=False,
            error="",
        )
    except FileNotFoundError as exc:
        return _command_result(
            cmd, returncode=None, stdout="", stderr="", timed_out=False, error=str(exc)
        )
    except subprocess.TimeoutExpired as exc:
        return _command_result(
            cmd,
            returncode=None,
            stdout=str(exc.output or ""),
            stderr=str(exc.stderr or ""),
            timed_out=True,
            error=f"timeout_after_{timeout}s",
        )
    except OSError as exc:
        return _command_result(
            cmd, returncode=None, stdout="", stderr="", timed_out=False, error=str(exc)
        )


def run_experiment(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    exp1422_path: str | Path = DEFAULT_EXP1422_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run the bounded Exp 1437 inspection/lint/simulation flow."""

    root = Path(project_root)
    output = Path(output_path)
    write_in_progress_artifact(output)
    exp1422_summary = load_exp1422_summary(exp1422_path)
    rtl_sources_checked = discover_rtl_sources(root)
    tool_probes = probe_tools(root, runner=runner)
    source_paths = [
        entry["path"]
        for entry in rtl_sources_checked
        if bool(entry["exists"]) and not entry["path"].endswith("_tb.v")
    ]
    testbench_exists = (root / EXPECTED_TESTBENCH_REL).is_file()
    command_results: dict[str, Any] = {
        "rtl_lint": _not_run_result("rtl_lint"),
        "simulation_compile": _not_run_result("simulation_compile"),
        "simulation": _not_run_result("simulation"),
    }

    if source_paths:
        lint_command = choose_lint_command(source_paths, tool_probes)
        if lint_command:
            command_results["rtl_lint"] = run_command(
                lint_command,
                cwd=root,
                runner=runner,
                timeout=30,
            )
            if command_results["rtl_lint"]["returncode"] == 0 and testbench_exists:
                sim_compile = [
                    "iverilog",
                    "-g2012",
                    "-o",
                    str(
                        (
                            root / "hardware" / "kv260" / "sim_work" / "discrete_sb_1437_sim"
                        ).relative_to(root)
                    ),
                    *source_paths,
                    EXPECTED_TESTBENCH_REL.as_posix(),
                ]
                command_results["simulation_compile"] = run_command(
                    sim_compile,
                    cwd=root,
                    runner=runner,
                    timeout=30,
                )
                if command_results["simulation_compile"]["returncode"] == 0:
                    command_results["simulation"] = run_command(
                        [str(part) for part in ["vvp", sim_compile[3]]],
                        cwd=root,
                        runner=runner,
                        timeout=30,
                    )

    artifact = build_artifact(
        run_date=run_date,
        exp1422_summary=exp1422_summary,
        rtl_sources_checked=rtl_sources_checked,
        tool_probes=tool_probes,
        command_results=command_results,
        testbench_exists=testbench_exists,
    )
    validate_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def load_exp1422_summary(path: str | Path) -> dict[str, Any]:
    """Load the prior RTL-spec artifact fields needed for provenance."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    return {
        "path": Path(path).as_posix(),
        "exists": bool(artifact),
        "status": artifact.get("status"),
        "rtl_spec_path": artifact.get("rtl_spec_path"),
        "hardware_execution_performed": bool(artifact.get("hardware_execution_performed", False)),
        "hardware_claim_allowed": bool(artifact.get("hardware_claim_allowed", False)),
        "honest_verdict": artifact.get("honest_verdict"),
    }


def discover_rtl_sources(project_root: str | Path) -> list[dict[str, Any]]:
    """Inspect the expected Discrete SB RTL source path and local candidates."""

    root = Path(project_root)
    candidates = [EXPECTED_SOURCE_REL]
    kv260_dir = root / "hardware" / "kv260"
    candidates.extend(
        sorted(
            path.relative_to(root)
            for path in kv260_dir.glob("*discrete*sb*.v")
            if path.name != EXPECTED_TESTBENCH_REL.name
        )
    )
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for rel in candidates:
        rel_posix = rel.as_posix()
        if rel_posix not in seen:
            seen.add(rel_posix)
            checked.append(
                {
                    "path": rel_posix,
                    "exists": (root / rel).is_file(),
                    "source": "exp1422_planned_discrete_sb_source",
                }
            )
    return checked


def probe_tools(project_root: str | Path, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    """Probe local RTL tool availability without installing new toolchains."""

    probes: dict[str, Any] = {}
    for tool, command in TOOL_VERSION_COMMANDS.items():
        result = run_command(command, cwd=project_root, runner=runner, timeout=10)
        probes[tool] = {
            "available": result["returncode"] == 0,
            "probe": result,
        }
    return probes


def choose_lint_command(source_paths: Sequence[str], tool_probes: Mapping[str, Any]) -> list[str]:
    """Pick the cheapest available syntax/lint command for the source set."""

    if bool(tool_probes["verilator"]["available"]):
        return ["verilator", "--lint-only", "--timing", *source_paths]
    if bool(tool_probes["yosys"]["available"]):
        return [
            "yosys",
            "-q",
            "-p",
            "read_verilog -sv " + " ".join(source_paths) + "; hierarchy -check",
        ]
    if bool(tool_probes["iverilog"]["available"]):
        return ["iverilog", "-tnull", "-g2012", *source_paths]
    return []


def build_artifact(
    *,
    run_date: str,
    exp1422_summary: Mapping[str, Any],
    rtl_sources_checked: Sequence[Mapping[str, Any]],
    tool_probes: Mapping[str, Any],
    command_results: Mapping[str, Any],
    testbench_exists: bool,
) -> dict[str, Any]:
    """Build the final JSON artifact from inspected source/tool evidence."""

    rtl_lint_complete = command_results["rtl_lint"]["returncode"] == 0
    simulation_complete = command_results["simulation"]["returncode"] == 0
    source_found = any(bool(source["exists"]) for source in rtl_sources_checked)
    tool_found = any(
        bool(tool_probes[name]["available"]) for name in ("yosys", "verilator", "iverilog")
    )
    status, honest_verdict, next_bitfile_step = _status_verdict_and_next_step(
        source_found=source_found,
        tool_found=tool_found,
        rtl_lint_complete=rtl_lint_complete,
        simulation_complete=simulation_complete,
        testbench_exists=testbench_exists,
        lint_attempted=bool(command_results["rtl_lint"]["attempted"]),
    )
    return {
        "status": status,
        "run_date": run_date,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": 1422,
        "exp1422_summary": dict(exp1422_summary),
        "rtl_sources_checked": [dict(source) for source in rtl_sources_checked],
        "rtl_lint_complete": rtl_lint_complete,
        "simulation_complete": simulation_complete,
        "synthesis_attempted": False,
        "yosys_available": bool(tool_probes["yosys"]["available"]),
        "verilator_available": bool(tool_probes["verilator"]["available"]),
        "iverilog_available": bool(tool_probes["iverilog"]["available"]),
        "vivado_available": bool(tool_probes["vivado"]["available"]),
        "tool_probes": dict(tool_probes),
        "command_results": dict(command_results),
        "testbench_exists": testbench_exists,
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
        "next_bitfile_step": next_bitfile_step,
        "honest_verdict": honest_verdict,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate claim gating and required Exp 1437 schema fields."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("status must be complete or blocked")
    if artifact["hardware_execution_performed"] is not False:
        raise ValueError(
            "hardware_execution_performed must remain false without KV260 board evidence"
        )
    if artifact["hardware_claim_allowed"] is not False:
        raise ValueError("hardware_claim_allowed must remain false without KV260 board evidence")


def _status_verdict_and_next_step(
    *,
    source_found: bool,
    tool_found: bool,
    rtl_lint_complete: bool,
    simulation_complete: bool,
    testbench_exists: bool,
    lint_attempted: bool,
) -> tuple[str, str, str]:
    if not source_found:
        return (
            "blocked",
            "blocked_missing_discrete_sb_rtl_source",
            "Implement hardware/kv260/discrete_sb_256.v from hardware/kv260/discrete_sb_rtl_spec.md, then rerun Exp 1437 lint/sim before Vivado synthesis.",
        )
    if not tool_found:
        return (
            "blocked",
            "blocked_no_rtl_lint_or_sim_tool",
            "Expose yosys/verilator/iverilog on PATH or run on a tool-equipped host, then rerun bounded lint/sim.",
        )
    if not lint_attempted or not rtl_lint_complete:
        return (
            "blocked",
            "blocked_rtl_lint_failed",
            "Fix the Discrete SB RTL syntax/lint errors, then rerun Exp 1437 before attempting synthesis.",
        )
    if testbench_exists and not simulation_complete:
        return (
            "blocked",
            "blocked_simulation_failed",
            "Fix the Discrete SB testbench or simulator failure, then rerun Exp 1437 before Vivado synthesis.",
        )
    if simulation_complete:
        return (
            "complete",
            "rtl_lint_and_simulation_complete_no_hardware_execution",
            "Run Vivado synthesis for hardware/kv260/discrete_sb_256.v on a Vivado-equipped host, then generate and validate a KV260 bitfile.",
        )
    return (
        "complete",
        "rtl_lint_complete_simulation_not_run_no_testbench",
        "Add hardware/kv260/discrete_sb_256_tb.v for behavioral simulation, then run Vivado synthesis on a tool-equipped host.",
    )


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
        "attempted": True,
        "command": [str(part) for part in command],
        "returncode": returncode,
        "stdout_summary": _summarize(stdout),
        "stderr_summary": _summarize(stderr),
        "timed_out": timed_out,
        "error": error,
    }


def _not_run_result(name: str) -> dict[str, Any]:
    return {
        "attempted": False,
        "command": [],
        "returncode": None,
        "stdout_summary": "",
        "stderr_summary": "",
        "timed_out": False,
        "error": f"{name}_not_run",
    }


def _summarize(text: str) -> str:
    summary = str(text or "").strip()
    if len(summary) <= SUMMARY_LIMIT:
        return summary
    return summary[:SUMMARY_LIMIT].rstrip() + "\n...[truncated]"
