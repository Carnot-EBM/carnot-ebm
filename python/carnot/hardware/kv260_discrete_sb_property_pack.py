"""Exp 1517 KV260 Discrete SB source-level RTL property pack.

This module packages source-level properties for the KV260 Discrete Simulated
Bifurcation RTL.  It is deliberately limited to file inspection, Verilator
lint, Icarus simulation, and Yosys parse probes.  It never invokes Vivado
bitfile generation, board programming, SSH, PYNQ, or latency measurement.

Spec refs: REQ-ISING-028, SCENARIO-ISING-038.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DATE = "20260508"
EXPERIMENT_ID = 1517
DEFAULT_EXP1506_PATH = (
    PROJECT_ROOT / "results" / "experiment_1506_115_completion_archive_116_activation.json"
)
DEFAULT_EXP1460_REQUESTED_PATH = (
    PROJECT_ROOT / "results" / "experiment_1460_hardware_track_priority_retro.json"
)
DEFAULT_EXP1460_FALLBACK_PATH = (
    PROJECT_ROOT / "results" / "experiment_1460_hardware_portfolio_narrowing.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "experiment_1517_kv260_discrete_sb_rtl_property_pack_v2.json"
)
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "results" / "kv260_discrete_sb_property_manifest_1517.json"

SOURCE_REL = Path("hardware/kv260/discrete_sb_256.v")
SMOKE_TB_REL = Path("hardware/kv260/discrete_sb_256_tb.v")
PROPERTY_TB_REL = Path("hardware/kv260/discrete_sb_256_property_tb.sv")
REGRESSION_MANIFEST_REL = Path("hardware/kv260/discrete_sb_regression_manifest.md")
SIM_OUTPUT = "/tmp/discrete_sb_256_property_tb_1517.vvp"
SUMMARY_LIMIT = 800

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "kv260_property_pack_ready",
    "gated_inputs_present",
    "source_level_only",
    "no_board_execution",
    "no_bitstream_claim",
    "rtl_files_checked",
    "properties_defined",
    "simulations_run",
    "lint_or_parse_results",
    "property_manifest_path",
    "blockers",
    "honest_verdict",
}

HONEST_VERDICT_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

TOOL_VERSION_COMMANDS: dict[str, list[str]] = {
    "verilator": ["verilator", "--version"],
    "iverilog": ["iverilog", "-V"],
    "yosys": ["yosys", "--version"],
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


def write_in_progress_artifact(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Write the bootstrap artifact before source and tool inspection.

    Exp 1517 has a strict claim boundary.  Even an interrupted run must say that
    the work was source-level only and made no board or bitstream claim.
    """

    artifact = _base_terminal_artifact(status="in_progress")
    artifact["honest_verdict"] = "complete_in_progress_kv260_property_pack_source_level_only"
    _write_json(path, artifact)
    return artifact


def run_command(
    command: Sequence[str],
    *,
    cwd: str | Path,
    runner: Runner = subprocess.run,
    timeout: int = 60,
) -> dict[str, Any]:
    """Run one bounded local source-level command and summarize its transcript."""

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
            command=cmd,
            returncode=int(completed.returncode),
            stdout=str(completed.stdout or ""),
            stderr=str(completed.stderr or ""),
            error_class="none" if completed.returncode == 0 else "nonzero_exit",
            error="",
        )
    except FileNotFoundError as exc:
        return _command_result(
            command=cmd,
            returncode=None,
            stdout="",
            stderr="",
            error_class="not_found",
            error=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        return _command_result(
            command=cmd,
            returncode=None,
            stdout=str(exc.output or ""),
            stderr=str(exc.stderr or ""),
            error_class="timeout",
            error=f"timeout_after_{timeout}s",
        )
    except OSError as exc:
        return _command_result(
            command=cmd,
            returncode=None,
            stdout="",
            stderr="",
            error_class="os_error",
            error=str(exc),
        )


def run_property_pack(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    exp1506_path: str | Path = DEFAULT_EXP1506_PATH,
    exp1460_requested_path: str | Path = DEFAULT_EXP1460_REQUESTED_PATH,
    exp1460_fallback_path: str | Path = DEFAULT_EXP1460_FALLBACK_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run the Exp 1517 source-only property-pack flow."""

    root = Path(project_root)
    output = Path(output_path)
    manifest = Path(manifest_path)
    exp1506_path = _rebase_default_path(exp1506_path, DEFAULT_EXP1506_PATH, root)
    exp1460_requested_path = _rebase_default_path(
        exp1460_requested_path,
        DEFAULT_EXP1460_REQUESTED_PATH,
        root,
    )
    exp1460_fallback_path = _rebase_default_path(
        exp1460_fallback_path,
        DEFAULT_EXP1460_FALLBACK_PATH,
        root,
    )
    write_in_progress_artifact(output)

    exp1506 = _load_json(exp1506_path)
    if exp1506.get("prior_kv260_source_track_active") is not True:
        artifact = _base_terminal_artifact(status="complete")
        artifact["blockers"] = [
            _manual_error(
                "prior_gate",
                "prior_kv260_source_track_inactive",
                "Exp 1506 did not report prior_kv260_source_track_active=true.",
            )
        ]
        artifact["honest_verdict"] = "complete: gated_prior_kv260_source_track_inactive"
        validate_terminal_artifact(artifact)
        _write_json(output, artifact)
        return artifact

    exp1460_path, path_mismatches = _resolve_exp1460_path(
        exp1460_requested_path,
        exp1460_fallback_path,
        root,
    )
    exp1460 = _load_json(exp1460_path)
    rtl_files_checked = inventory_source_files(root)
    properties_defined = evaluate_properties(root)
    tool_probes = probe_tools(root, runner=runner)
    tools_available = {tool: bool(probe["available"]) for tool, probe in tool_probes.items()}

    blockers = _initial_blockers(exp1460, rtl_files_checked, properties_defined)
    lint_or_parse_results: list[dict[str, Any]] = []
    simulations_run: list[dict[str, Any]] = []
    source_bundle_present = _source_bundle_present(rtl_files_checked)

    if source_bundle_present:
        lint_or_parse_results.append(
            _run_or_not(
                tools_available["verilator"],
                _verilator_property_lint_command(),
                stage="verilator_property_lint",
                missing_error_class="no_verilator",
                root=root,
                runner=runner,
            )
        )
        simulations_run.append(
            _run_icarus_property_simulation(
                tools_available["iverilog"],
                root=root,
                runner=runner,
            )
        )
        lint_or_parse_results.append(
            _run_or_not(
                tools_available["iverilog"],
                _iverilog_parse_command(),
                stage="iverilog_property_parse",
                missing_error_class="no_iverilog",
                root=root,
                runner=runner,
            )
        )
    else:
        lint_or_parse_results.extend(
            [
                _not_run_result("verilator_property_lint", "source_bundle_missing"),
                _not_run_result("iverilog_property_parse", "source_bundle_missing"),
            ]
        )
        simulations_run.append(
            _not_run_result("icarus_property_simulation", "source_bundle_missing")
        )

    blockers.extend(_blockers_from_tool_results(lint_or_parse_results, simulations_run))
    if not _any_source_command_executed(lint_or_parse_results, simulations_run):
        blockers.append(
            _manual_error(
                "source_checks",
                "no_source_level_command_executed",
                "No local source-level lint, parse, or simulation command executed.",
            )
        )

    manifest_payload = _build_manifest(
        run_date=run_date,
        root=root,
        exp1506_path=Path(exp1506_path),
        exp1506=exp1506,
        exp1460_path=exp1460_path,
        exp1460=exp1460,
        path_mismatches=path_mismatches,
        rtl_files_checked=rtl_files_checked,
        properties_defined=properties_defined,
        tool_probes=tool_probes,
        lint_or_parse_results=lint_or_parse_results,
        simulations_run=simulations_run,
        blockers=blockers,
    )
    _write_json(manifest, manifest_payload)

    property_pack_ready = (
        _exp1460_keeps_source_only(exp1460)
        and all(prop["passed"] for prop in properties_defined)
        and _any_source_command_executed(lint_or_parse_results, simulations_run)
        and not blockers
    )
    artifact = _base_terminal_artifact(status="complete")
    artifact.update(
        {
            "kv260_property_pack_ready": property_pack_ready,
            "gated_inputs_present": bool(_exp1460_keeps_source_only(exp1460)),
            "rtl_files_checked": rtl_files_checked,
            "properties_defined": properties_defined,
            "simulations_run": simulations_run,
            "lint_or_parse_results": lint_or_parse_results,
            "property_manifest_path": _relative_to_root(manifest, root),
            "blockers": blockers,
            "honest_verdict": _classify_verdict(property_pack_ready, blockers),
        }
    )
    validate_terminal_artifact(artifact)
    _write_json(output, artifact)
    return artifact


def inventory_source_files(project_root: str | Path) -> list[dict[str, Any]]:
    """Inventory the source, property, manifest, and script paths used by Exp 1517."""

    root = Path(project_root)
    files = [
        (SOURCE_REL, "rtl_source"),
        (SMOKE_TB_REL, "smoke_testbench"),
        (PROPERTY_TB_REL, "property_testbench"),
        (REGRESSION_MANIFEST_REL, "existing_regression_manifest"),
        (Path("python/carnot/hardware/discrete_sb_regression_pack.py"), "python_helper"),
        (Path("python/carnot/hardware/discrete_sb_rtl_lint_sim_rerun.py"), "python_helper"),
    ]
    return [
        {
            "path": rel.as_posix(),
            "kind": kind,
            "exists": (root / rel).is_file(),
        }
        for rel, kind in files
    ]


def evaluate_properties(project_root: str | Path) -> list[dict[str, Any]]:
    """Evaluate the source-level property definitions against local source text."""

    root = Path(project_root)
    source_cache: dict[str, str] = {}
    evaluated = []
    for prop in _property_definitions():
        checks = []
        for check in prop["source_checks"]:
            rel = str(check["path"])
            if rel not in source_cache:
                candidate = root / rel
                source_cache[rel] = (
                    candidate.read_text(encoding="utf-8") if candidate.exists() else ""
                )
            checks.append(
                {
                    "path": rel,
                    "token": check["token"],
                    "passed": check["token"] in source_cache[rel],
                }
            )
        evaluated.append(
            {
                "id": prop["id"],
                "category": prop["category"],
                "spec_refs": prop["spec_refs"],
                "description": prop["description"],
                "checks": checks,
                "passed": all(check["passed"] for check in checks),
            }
        )
    return evaluated


def probe_tools(project_root: str | Path, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    """Probe only source-level HDL tools; Vivado and board tools are out of scope."""

    return {
        tool: {
            "available": (
                result := run_command(command, cwd=project_root, runner=runner, timeout=10)
            )["returncode"]
            == 0,
            "probe": result,
        }
        for tool, command in TOOL_VERSION_COMMANDS.items()
    }


def validate_terminal_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal schema and source-level claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["status"] not in {"in_progress", "complete"}:
        raise ValueError("status must be in_progress or complete")
    for field in ("source_level_only", "no_board_execution", "no_bitstream_claim"):
        if artifact[field] is not True:
            raise ValueError(f"{field} must remain true for Exp 1517")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(HONEST_VERDICT_PREFIXES):
        raise ValueError("honest_verdict has an unsupported terminal prefix")


def _base_terminal_artifact(*, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "kv260_property_pack_ready": False,
        "gated_inputs_present": False,
        "source_level_only": True,
        "no_board_execution": True,
        "no_bitstream_claim": True,
        "rtl_files_checked": [],
        "properties_defined": [],
        "simulations_run": [],
        "lint_or_parse_results": [],
        "property_manifest_path": "",
        "blockers": [],
        "honest_verdict": "complete_in_progress_kv260_property_pack_source_level_only",
    }


def _property_definitions() -> list[dict[str, Any]]:
    return [
        {
            "id": "KV260-DSB-PROP-BOUNDED-001",
            "category": "bounded_behavior",
            "spec_refs": ["REQ-ISING-028", "SCENARIO-ISING-038"],
            "description": "A bounded one-step run must reach DONE after row 255.",
            "source_checks": [
                {"path": SOURCE_REL.as_posix(), "token": "STATE_ROW"},
                {"path": SOURCE_REL.as_posix(), "token": "STATE_COMMIT"},
                {
                    "path": SOURCE_REL.as_posix(),
                    "token": "(step_count + 16'd1) >= max_steps_active",
                },
                {"path": PROPERTY_TB_REL.as_posix(), "token": "PROP_BOUNDED_ONE_STEP_DONE"},
            ],
        },
        {
            "id": "KV260-DSB-PROP-RESET-001",
            "category": "reset_behavior",
            "spec_refs": ["REQ-ISING-028", "SCENARIO-ISING-038"],
            "description": "Reset must drive IDLE, clear counters, and expose known spins.",
            "source_checks": [
                {"path": SOURCE_REL.as_posix(), "token": "if (rst) begin"},
                {"path": SOURCE_REL.as_posix(), "token": "state <= STATE_IDLE;"},
                {"path": SOURCE_REL.as_posix(), "token": "busy <= 1'b0;"},
                {"path": SOURCE_REL.as_posix(), "token": "step_count <= 16'd0;"},
                {"path": PROPERTY_TB_REL.as_posix(), "token": "PROP_RESET_KNOWN_STATE"},
            ],
        },
        {
            "id": "KV260-DSB-PROP-ORDER-001",
            "category": "deterministic_update_ordering",
            "spec_refs": ["REQ-ISING-028", "SCENARIO-ISING-038"],
            "description": "Rows must read spin_snapshot and commit spin_next only after a sweep.",
            "source_checks": [
                {"path": SOURCE_REL.as_posix(), "token": "spin_snapshot <= spin_cur;"},
                {"path": SOURCE_REL.as_posix(), "token": "spin_next[row_idx]"},
                {"path": SOURCE_REL.as_posix(), "token": "spin_cur <= spin_next;"},
                {"path": SOURCE_REL.as_posix(), "token": "spin_out <= spin_next;"},
                {
                    "path": PROPERTY_TB_REL.as_posix(),
                    "token": "PROP_SNAPSHOT_STABLE_DURING_ROW_UPDATE",
                },
            ],
        },
        {
            "id": "KV260-DSB-PROP-SHAPE-001",
            "category": "shape_width_assumptions",
            "spec_refs": ["REQ-ISING-028", "SCENARIO-ISING-038"],
            "description": "The source must preserve 256 variables and int8 dense couplings.",
            "source_checks": [
                {"path": SOURCE_REL.as_posix(), "token": "parameter integer N_VARIABLES = 256"},
                {"path": SOURCE_REL.as_posix(), "token": "parameter integer COUPLING_BITS = 8"},
                {
                    "path": SOURCE_REL.as_posix(),
                    "token": "COUPLING_COUNT = N_VARIABLES * N_VARIABLES",
                },
                {"path": SOURCE_REL.as_posix(), "token": "[15:0]"},
                {"path": SOURCE_REL.as_posix(), "token": "[4:0]"},
                {"path": SOURCE_REL.as_posix(), "token": "output reg [N_VARIABLES-1:0]"},
                {"path": PROPERTY_TB_REL.as_posix(), "token": "PROP_SHAPE_WIDTH_DEFAULTS"},
            ],
        },
    ]


def _resolve_exp1460_path(
    requested_path: str | Path,
    fallback_path: str | Path,
    root: Path,
) -> tuple[Path, list[dict[str, str]]]:
    requested = Path(requested_path)
    fallback = Path(fallback_path)
    if requested.is_file():
        return requested, []
    if fallback.is_file():
        return fallback, [
            {
                "requested": _relative_to_root(requested, root),
                "actual": _relative_to_root(fallback, root),
            }
        ]
    return requested, []


def _initial_blockers(
    exp1460: Mapping[str, Any],
    rtl_files_checked: Sequence[Mapping[str, Any]],
    properties_defined: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not _exp1460_keeps_source_only(exp1460):
        blockers.append(
            _manual_error(
                "exp1460_boundary",
                "exp1460_source_only_boundary_missing",
                "Exp 1460 did not preserve the KV260 source-only claim boundary.",
            )
        )
    for checked in rtl_files_checked:
        if checked["kind"] in {"rtl_source", "property_testbench"} and not checked["exists"]:
            blockers.append(
                _manual_error("source_inventory", "source_file_missing", str(checked["path"]))
            )
    for prop in properties_defined:
        if not prop["passed"]:
            blockers.append(_manual_error("property_scan", "property_tokens_missing", prop["id"]))
    return blockers


def _run_or_not(
    should_run: bool,
    command: Sequence[str],
    *,
    stage: str,
    missing_error_class: str,
    root: Path,
    runner: Runner,
) -> dict[str, Any]:
    if not should_run:
        return _not_run_result(stage, missing_error_class)
    result = run_command(command, cwd=root, runner=runner, timeout=60)
    result["stage"] = stage
    return result


def _run_icarus_property_simulation(
    should_run: bool,
    *,
    root: Path,
    runner: Runner,
) -> dict[str, Any]:
    stage = "icarus_property_simulation"
    if not should_run:
        return _not_run_result(stage, "no_iverilog")
    compile_result = run_command(_icarus_compile_command(), cwd=root, runner=runner, timeout=60)
    if compile_result["returncode"] == 0:
        run_result = run_command(_icarus_run_command(), cwd=root, runner=runner, timeout=60)
    else:
        run_result = _not_run_result("vvp_property_run", "compile_failed")
    return {
        "stage": stage,
        "command_string": f"{_format_command(_icarus_compile_command())} && "
        f"{_format_command(_icarus_run_command())}",
        "returncode": run_result["returncode"]
        if compile_result["returncode"] == 0
        else compile_result["returncode"],
        "error_class": run_result["error_class"]
        if compile_result["returncode"] == 0
        else compile_result["error_class"],
        "compile_result": compile_result,
        "run_result": run_result,
        "stdout_summary": run_result["stdout_summary"],
        "stderr_summary": run_result["stderr_summary"],
        "error": run_result["error"]
        if compile_result["returncode"] == 0
        else compile_result["error"],
    }


def _blockers_from_tool_results(
    lint_or_parse_results: Sequence[Mapping[str, Any]],
    simulations_run: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for result in [*lint_or_parse_results, *simulations_run]:
        if result["returncode"] not in (0, None):
            blockers.append(
                _manual_error(
                    str(result["stage"]),
                    str(result["error_class"]),
                    str(result.get("stderr_summary") or result.get("error") or "nonzero exit"),
                )
            )
        if result["returncode"] is None and str(result["error_class"]).startswith("no_"):
            blockers.append(
                _manual_error(str(result["stage"]), str(result["error_class"]), "tool unavailable")
            )
    return blockers


def _build_manifest(
    *,
    run_date: str,
    root: Path,
    exp1506_path: Path,
    exp1506: Mapping[str, Any],
    exp1460_path: Path,
    exp1460: Mapping[str, Any],
    path_mismatches: Sequence[Mapping[str, str]],
    rtl_files_checked: Sequence[Mapping[str, Any]],
    properties_defined: Sequence[Mapping[str, Any]],
    tool_probes: Mapping[str, Any],
    lint_or_parse_results: Sequence[Mapping[str, Any]],
    simulations_run: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "complete",
        "run_date": run_date,
        "experiment_id": EXPERIMENT_ID,
        "source_level_only": True,
        "no_board_execution": True,
        "no_bitstream_claim": True,
        "prior_artifacts": {
            "exp1506": {
                "path": _relative_to_root(exp1506_path, root),
                "prior_kv260_source_track_active": bool(
                    exp1506.get("prior_kv260_source_track_active")
                ),
            },
            "exp1460": {
                "path": _relative_to_root(exp1460_path, root),
                "kv260_source_only_boundary": _exp1460_keeps_source_only(exp1460),
            },
        },
        "path_mismatches": list(path_mismatches),
        "rtl_files_checked": list(rtl_files_checked),
        "properties_defined": list(properties_defined),
        "tool_probe_results": dict(tool_probes),
        "lint_or_parse_results": list(lint_or_parse_results),
        "simulations_run": list(simulations_run),
        "claim_boundaries": {
            "vivado_bitfile_generation": "not_run",
            "kv260_board_programming": "not_run",
            "hardware_latency_measurement": "not_run",
        },
        "blockers": list(blockers),
    }


def _exp1460_keeps_source_only(exp1460: Mapping[str, Any]) -> bool:
    for track in exp1460.get("active_hardware_tracks", []):
        if track.get("track_id") == "kv260_discrete_sb_rtl_sim":
            boundary = str(track.get("claim_boundary", "")).lower()
            return all(token in boundary for token in ("no kv260 board", "bitfile", "latency"))
    return False


def _source_bundle_present(rtl_files_checked: Sequence[Mapping[str, Any]]) -> bool:
    required = {"rtl_source", "property_testbench"}
    present = {str(item["kind"]) for item in rtl_files_checked if item["exists"]}
    return required <= present


def _any_source_command_executed(
    lint_or_parse_results: Sequence[Mapping[str, Any]],
    simulations_run: Sequence[Mapping[str, Any]],
) -> bool:
    return any(
        result["returncode"] is not None for result in [*lint_or_parse_results, *simulations_run]
    )


def _classify_verdict(property_pack_ready: bool, blockers: Sequence[Mapping[str, Any]]) -> str:
    if property_pack_ready:
        return "complete: kv260_discrete_sb_property_pack_ready_source_level_only"
    if any(blocker["error_class"] == "no_source_level_command_executed" for blocker in blockers):
        return "complete: blocked_no_source_level_command_executed"
    return "complete: blocked_kv260_property_pack_source_level_issues"


def _verilator_property_lint_command() -> list[str]:
    return [
        "verilator",
        "--lint-only",
        "--timing",
        "-Wall",
        "-Wno-DECLFILENAME",
        "-Wno-BLKSEQ",
        "-Wno-WIDTH",
        "--top-module",
        "discrete_sb_256_property_tb",
        SOURCE_REL.as_posix(),
        PROPERTY_TB_REL.as_posix(),
    ]


def _icarus_compile_command() -> list[str]:
    return [
        "iverilog",
        "-g2012",
        "-o",
        SIM_OUTPUT,
        SOURCE_REL.as_posix(),
        PROPERTY_TB_REL.as_posix(),
    ]


def _icarus_run_command() -> list[str]:
    return ["vvp", SIM_OUTPUT]


def _iverilog_parse_command() -> list[str]:
    return [
        "iverilog",
        "-tnull",
        "-g2012",
        SOURCE_REL.as_posix(),
        PROPERTY_TB_REL.as_posix(),
    ]


def _command_result(
    *,
    command: Sequence[str],
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


def _not_run_result(stage: str, error_class: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "command": [],
        "command_string": "",
        "returncode": None,
        "stdout_summary": "",
        "stderr_summary": "",
        "error_class": error_class,
        "error": error_class,
    }


def _manual_error(stage: str, error_class: str, detail: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_class": error_class,
        "returncode": None,
        "stdout_summary": "",
        "stderr_summary": detail,
        "error": detail,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    return json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else {}


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _rebase_default_path(path: str | Path, default_path: Path, root: Path) -> Path:
    candidate = Path(path)
    if candidate == default_path:
        return root / default_path.relative_to(PROJECT_ROOT)
    return candidate


def _format_command(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def _summarize(text: str) -> str:
    summary = str(text or "").strip()
    if len(summary) <= SUMMARY_LIMIT:
        return summary
    return summary[:SUMMARY_LIMIT].rstrip() + "\n...[truncated]"
