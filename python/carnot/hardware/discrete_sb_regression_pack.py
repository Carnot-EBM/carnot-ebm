"""Exp 1476 KV260 Discrete SB RTL regression manifest packaging.

Exp 1451 proved that the Discrete SB RTL source and testbench can pass local
Verilator lint and Icarus simulation. Exp 1460 then narrowed the KV260 track to
source-level RTL evidence only. This module packages that narrow evidence into
a repeatable manifest and terminal JSON artifact without claiming Vivado
synthesis, bitfile generation, KV260 board execution, or latency.

Spec refs: REQ-ISING-027, SCENARIO-ISING-037.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DATE = "20260507"
EXPERIMENT_ID = 1476
DEFAULT_EXP1451_PATH = (
    PROJECT_ROOT / "results" / "experiment_1451_discrete_sb_rtl_lint_sim_rerun.json"
)
DEFAULT_EXP1460_PATH = (
    PROJECT_ROOT / "results" / "experiment_1460_hardware_portfolio_narrowing.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "experiment_1476_kv260_discrete_sb_rtl_regression_pack.json"
)
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "hardware" / "kv260" / "discrete_sb_regression_manifest.md"
SOURCE_REL = Path("hardware/kv260/discrete_sb_256.v")
TESTBENCH_REL = Path("hardware/kv260/discrete_sb_256_tb.v")
SIM_OUTPUT = "/tmp/discrete_sb_256_tb_1476.vvp"
SUMMARY_LIMIT = 800

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "rtl_files",
    "testbench_files",
    "rtl_regression_complete",
    "verilator_lint_passed",
    "icarus_sim_passed",
    "yosys_available",
    "board_execution_performed",
    "bitfile_produced",
    "latency_claimed",
    "regression_manifest_path",
    "honest_verdict",
}

TOOL_VERSION_COMMANDS: dict[str, list[str]] = {
    "verilator": ["verilator", "--version"],
    "iverilog": ["iverilog", "-V"],
    "yosys": ["yosys", "--version"],
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


def write_in_progress_artifact(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Write the Exp 1476 startup marker before source/tool inspection.

    The in-progress record already includes the no-board, no-bitfile, and
    no-latency fields because interrupted runs must never be mistaken for
    hardware evidence. Terminal fields are replaced after the regression
    commands and manifest write complete.
    """

    artifact: dict[str, Any] = {
        "status": "in_progress",
        "rtl_files": [],
        "testbench_files": [],
        "rtl_regression_complete": False,
        "verilator_lint_passed": False,
        "icarus_sim_passed": False,
        "yosys_available": False,
        "board_execution_performed": False,
        "bitfile_produced": False,
        "latency_claimed": False,
        "regression_manifest_path": "",
        "honest_verdict": (
            "in_progress: KV260 Discrete SB RTL regression packaging started; "
            "no board, bitfile, or latency claim made."
        ),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run_regression_pack(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    exp1451_path: str | Path = DEFAULT_EXP1451_PATH,
    exp1460_path: str | Path = DEFAULT_EXP1460_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run local RTL checks and write the Exp 1476 JSON plus manifest."""

    root = Path(project_root)
    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output)

    exp1451 = _load_json(exp1451_path)
    exp1460 = _load_json(exp1460_path)
    prior_boundary_preserved = _prior_boundary_preserved(exp1451, exp1460)
    rtl_present = (root / SOURCE_REL).is_file()
    testbench_present = (root / TESTBENCH_REL).is_file()

    tool_probes = _probe_tools(root, runner=runner)
    commands_run = [probe["probe"] for probe in tool_probes.values()]
    tools_available = {tool: bool(probe["available"]) for tool, probe in tool_probes.items()}

    rtl_errors: list[dict[str, Any]] = []
    verilator_lint_passed = False
    icarus_sim_passed = False
    yosys_probe_passed = False
    verilator_lint_command = _format_command(_verilator_lint_command())
    icarus_sim_command = (
        f"{_format_command(_icarus_compile_command())} && {_format_command(_icarus_run_command())}"
    )
    yosys_probe_command = _format_command(_yosys_probe_command())

    if not rtl_present or not testbench_present:
        rtl_errors.append(
            _manual_error(
                "source_scan",
                "rtl_or_testbench_missing",
                f"{SOURCE_REL.as_posix()} present={rtl_present}; "
                f"{TESTBENCH_REL.as_posix()} present={testbench_present}",
            )
        )
    elif not prior_boundary_preserved:
        rtl_errors.append(
            _manual_error(
                "prior_artifacts",
                "prior_claim_boundary_not_preserved",
                "Exp 1451/1460 do not preserve the source-only KV260 claim boundary.",
            )
        )
    else:
        if tools_available["verilator"]:
            lint_result = _run_command(
                _verilator_lint_command(), cwd=root, runner=runner, timeout=30
            )
            commands_run.append(lint_result)
            verilator_lint_passed = lint_result["returncode"] == 0
            rtl_errors.extend(_errors_from_result("verilator_lint", lint_result))
        else:
            rtl_errors.append(_manual_error("verilator_lint", "verilator_unavailable", "verilator"))

        if tools_available["iverilog"]:
            compile_result = _run_command(
                _icarus_compile_command(), cwd=root, runner=runner, timeout=60
            )
            commands_run.append(compile_result)
            rtl_errors.extend(_errors_from_result("icarus_compile", compile_result))
            if compile_result["returncode"] == 0:
                sim_result = _run_command(
                    _icarus_run_command(), cwd=root, runner=runner, timeout=60
                )
                commands_run.append(sim_result)
                icarus_sim_passed = sim_result["returncode"] == 0
                rtl_errors.extend(_errors_from_result("icarus_simulation", sim_result))
        else:
            rtl_errors.append(
                _manual_error("icarus_simulation", "iverilog_unavailable", "iverilog")
            )

        if tools_available["yosys"]:
            yosys_result = _run_command(_yosys_probe_command(), cwd=root, runner=runner, timeout=30)
            commands_run.append(yosys_result)
            yosys_probe_passed = yosys_result["returncode"] == 0
            rtl_errors.extend(_errors_from_result("yosys_probe", yosys_result))

    rtl_regression_complete = (
        rtl_present
        and testbench_present
        and prior_boundary_preserved
        and verilator_lint_passed
        and icarus_sim_passed
    )
    artifact = {
        "status": "complete",
        "run_date": run_date,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_ids": [1451, 1460],
        "rtl_files": [SOURCE_REL.as_posix()] if rtl_present else [],
        "testbench_files": [TESTBENCH_REL.as_posix()] if testbench_present else [],
        "rtl_regression_complete": rtl_regression_complete,
        "verilator_lint_passed": verilator_lint_passed,
        "icarus_sim_passed": icarus_sim_passed,
        "yosys_available": tools_available["yosys"],
        "yosys_probe_passed": yosys_probe_passed,
        "board_execution_performed": False,
        "bitfile_produced": False,
        "latency_claimed": False,
        "regression_manifest_path": _relative_to_root(manifest, root),
        "prior_boundary_preserved": prior_boundary_preserved,
        "tools_available": tools_available,
        "tool_probe_results": tool_probes,
        "verilator_lint_command": verilator_lint_command,
        "icarus_sim_command": icarus_sim_command,
        "yosys_probe_command": yosys_probe_command,
        "commands_run": commands_run,
        "rtl_errors": rtl_errors,
        "honest_verdict": _classify_verdict(
            rtl_present=rtl_present,
            testbench_present=testbench_present,
            prior_boundary_preserved=prior_boundary_preserved,
            rtl_regression_complete=rtl_regression_complete,
        ),
    }
    validate_artifact(artifact)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(_render_manifest(artifact), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal schema and the Exp 1476 claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    for claim_field in ("board_execution_performed", "bitfile_produced", "latency_claimed"):
        if artifact[claim_field] is not False:
            raise ValueError(f"{claim_field} must remain false without same-run evidence")


def _load_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    return json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else {}


def _prior_boundary_preserved(exp1451: Mapping[str, Any], exp1460: Mapping[str, Any]) -> bool:
    exp1451_safe = (
        exp1451.get("status") == "complete"
        and exp1451.get("hardware_execution_performed") is False
        and exp1451.get("hardware_claim_allowed") is False
    )
    return bool(exp1451_safe and _exp1460_keeps_kv260_source_only(exp1460))


def _exp1460_keeps_kv260_source_only(exp1460: Mapping[str, Any]) -> bool:
    for track in exp1460.get("active_hardware_tracks", []):
        if track.get("track_id") == "kv260_discrete_sb_rtl_sim":
            boundary = str(track.get("claim_boundary", "")).lower()
            return all(token in boundary for token in ("no kv260 board", "bitfile", "latency"))
    return False


def _probe_tools(project_root: str | Path, *, runner: Runner) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    for tool, command in TOOL_VERSION_COMMANDS.items():
        result = _run_command(command, cwd=project_root, runner=runner, timeout=10)
        probes[tool] = {"available": result["returncode"] == 0, "probe": result}
    return probes


def _verilator_lint_command() -> list[str]:
    return [
        "verilator",
        "--lint-only",
        "--timing",
        "-Wall",
        "-Wno-DECLFILENAME",
        "-Wno-BLKSEQ",
        "-Wno-WIDTH",
        SOURCE_REL.as_posix(),
        TESTBENCH_REL.as_posix(),
    ]


def _icarus_compile_command() -> list[str]:
    return [
        "iverilog",
        "-g2012",
        "-o",
        SIM_OUTPUT,
        SOURCE_REL.as_posix(),
        TESTBENCH_REL.as_posix(),
    ]


def _icarus_run_command() -> list[str]:
    return ["vvp", SIM_OUTPUT]


def _yosys_probe_command() -> list[str]:
    return ["yosys", "-q", "-p", "help read_verilog"]


def _run_command(
    command: Sequence[str],
    *,
    cwd: str | Path,
    runner: Runner,
    timeout: int,
) -> dict[str, Any]:
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
            cmd, returncode=None, stdout="", stderr="", error_class="not_found", error=str(exc)
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
            cmd, returncode=None, stdout="", stderr="", error_class="os_error", error=str(exc)
        )


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


def _classify_verdict(
    *,
    rtl_present: bool,
    testbench_present: bool,
    prior_boundary_preserved: bool,
    rtl_regression_complete: bool,
) -> str:
    if not rtl_present or not testbench_present:
        return "blocked_missing_discrete_sb_rtl_or_testbench_no_hardware_claim"
    if not prior_boundary_preserved:
        return "blocked_prior_artifact_claim_boundary_not_preserved"
    if rtl_regression_complete:
        return (
            "rtl_regression_manifest_complete_source_level_only_no_board_bitfile_or_latency_claim"
        )
    return "blocked_rtl_regression_checks_failed_no_hardware_claim"


def _render_manifest(artifact: Mapping[str, Any]) -> str:
    return f"""# KV260 Discrete SB RTL Regression Manifest

Spec traces: REQ-ISING-027, SCENARIO-ISING-037

Run date: {artifact["run_date"]}
Result artifact: `results/experiment_1476_kv260_discrete_sb_rtl_regression_pack.json`

## Source Files

- RTL files: `{", ".join(artifact["rtl_files"]) or "missing"}`
- Testbench files: `{", ".join(artifact["testbench_files"]) or "missing"}`

## Tool Availability

- Verilator available: `{str(artifact["tools_available"]["verilator"]).lower()}`
- Icarus Verilog available: `{str(artifact["tools_available"]["iverilog"]).lower()}`
- Yosys available: `{str(artifact["yosys_available"]).lower()}`

## Commands

1. Verilator lint
   - Command: `{artifact["verilator_lint_command"]}`
   - Expected output: return code 0 and no lint errors.
   - Observed pass: `{str(artifact["verilator_lint_passed"]).lower()}`
2. Icarus simulation
   - Command: `{artifact["icarus_sim_command"]}`
   - Expected output: return code 0 and `SIMULATION RESULT: PASS`.
   - Observed pass: `{str(artifact["icarus_sim_passed"]).lower()}`
3. Yosys availability probe
   - Command: `{artifact["yosys_probe_command"]}`
   - Expected output: return code 0 when Yosys is installed; unavailable tools are recorded.
   - Observed available: `{str(artifact["yosys_available"]).lower()}`
   - Observed probe pass: `{str(artifact["yosys_probe_passed"]).lower()}`

## Claim Boundary

Board execution performed: `{str(artifact["board_execution_performed"]).lower()}`
Bitfile produced: `{str(artifact["bitfile_produced"]).lower()}`
Latency claimed: `{str(artifact["latency_claimed"]).lower()}`

This manifest is source-level RTL regression evidence only. It does not claim
Vivado synthesis, a KV260 bitfile, KV260 board execution, or measured latency.

Honest verdict: `{artifact["honest_verdict"]}`
"""


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _format_command(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def _summarize(text: str) -> str:
    summary = str(text or "").strip()
    if len(summary) <= SUMMARY_LIMIT:
        return summary
    return summary[:SUMMARY_LIMIT].rstrip() + "\n...[truncated]"
