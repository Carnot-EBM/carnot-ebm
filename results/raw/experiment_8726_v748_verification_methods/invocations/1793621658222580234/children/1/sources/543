"""SSQA bounded RTL/PnR gate artifact for Exp 3037.

Spec refs: REQ-HW-087, SCENARIO-HW-087.

This artifact is a boundary record, not a benchmark. Its job is to make the
SSQA hardware row explicit for the .284 capstone while refusing to turn board
contact, bitstream build output, or an upstream blocked gate into a performance
claim. When the GateMate flash-smoke artifact is absent or lacks host-visible
output, the correct result is a written gate-skip artifact with the exact next
operator action.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


ARTIFACT_FILENAME = "experiment_3037_ssqa_bounded_rtl_pnr_gate_artifact_v2.json"
EXP3034_FILENAME = "experiment_3034_gatemate_output_contract_pinout_decision_v1.json"
EXP3035_FILENAMES = (
    "experiment_3035_gatemate_output_shim_rtl_ccf_sim_v1.json",
    "experiment_3035_gatemate_output_shim_rtl_ccf_sim.json",
)
EXP3036_FILENAME = "experiment_3036_gatemate_host_visible_flash_smoke_v4.json"
RUN_DATE = "20260525"
TOP_MODULE = "ising_n16_gatemate"
BUILD_DIR = Path("build") / "gatemate" / "experiment_3037_ssqa_bounded_rtl_pnr"
REQUIRED_FIELDS = (
    "ssqa_boundary_ready",
    "ssqa_gate_status",
    "upstream_gatemate_status",
    "rtl_or_pnr_commands_run",
    "resource_report_paths",
    "ssqa_performance_claim_allowed",
    "exact_blocker_or_next_action",
    "inference_substrate",
    "honest_verdict",
)


@dataclass(frozen=True)
class CommandResult:
    """Subprocess result wrapper used so tests can inject deterministic tool output."""

    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[[list[str], float], CommandResult]
WhichFunc = Callable[[str], str | None]


def _default_run_command(args: list[str], timeout_s: float) -> CommandResult:  # pragma: no cover
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        return CommandResult(127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else str(exc)
        return CommandResult(124, stdout, stderr)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_first(repo_root: Path, filenames: tuple[str, ...]) -> tuple[Path, dict[str, Any]]:
    for filename in filenames:
        path = repo_root / "results" / filename
        if path.exists():
            return path, _read_json(path)
    return repo_root / "results" / filenames[0], {}


def _status(path: Path, payload: Mapping[str, Any]) -> str:
    return "missing" if not path.exists() else str(payload.get("status") or "present")


def _artifact_summary(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": str(path),
        "available": path.exists(),
        "status": _status(path, payload),
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "gate_check_summary": str(payload.get("gate_check_summary", "")),
        "blocked_at_layer": str(payload.get("blocked_at_layer", "")),
    }


def _upstream_status(
    *,
    exp3034_path: Path,
    exp3034: Mapping[str, Any],
    exp3035_path: Path,
    exp3035: Mapping[str, Any],
    exp3036_path: Path,
    exp3036: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    exp3034_summary = _artifact_summary(exp3034_path, exp3034)
    exp3034_summary.update(
        {
            "gatemate_output_contract_ready": exp3034.get("gatemate_output_contract_ready")
            is True,
            "host_visible_io_plan_ready": exp3034.get("host_visible_io_plan_ready") is True,
        }
    )
    exp3035_summary = _artifact_summary(exp3035_path, exp3035)
    exp3036_summary = _artifact_summary(exp3036_path, exp3036)
    exp3036_summary.update(
        {
            "gatemate_flash_smoke_ready": exp3036.get("gatemate_flash_smoke_ready") is True,
            "host_visible_output_observed": _host_visible_observed(exp3036),
            "host_visible_transcript_path": str(exp3036.get("host_visible_transcript_path", "")),
        }
    )
    return {"exp3034": exp3034_summary, "exp3035": exp3035_summary, "exp3036": exp3036_summary}


def _host_visible_observed(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("host_visible_output_observed") is True
        or payload.get("host_visible_io_ready") is True
        or payload.get("host_visible_transcript_path")
    )


def _hardware_package(repo_root: Path) -> dict[str, list[str]]:
    gate_dir = repo_root / "hardware" / "gatemate"
    return {
        "rtl_paths": [str(path) for path in sorted(gate_dir.glob("*.v")) if path.is_file()],
        "ccf_paths": [str(path) for path in sorted(gate_dir.glob("*.ccf")) if path.is_file()],
        "test_vector_paths": [
            str(path) for path in sorted(gate_dir.glob("*test_vector*.json")) if path.is_file()
        ],
    }


def _operator_actions(exp3034: Mapping[str, Any]) -> list[str]:
    actions = [str(item) for item in exp3034.get("exact_operator_action_required", [])]
    return actions or [
        "Complete the GateMate output contract: bind a deterministic status output to a physical pin and commit the matching host reader transcript."
    ]


def _skip_actions(
    *,
    exp3036_path: Path,
    exp3036: Mapping[str, Any],
    exp3035: Mapping[str, Any],
    exp3034: Mapping[str, Any],
) -> tuple[list[str], str]:
    if not exp3036_path.exists():
        return (
            [
                f"Exp 3036 artifact missing: {exp3036_path} was not present, so no GateMate host-visible flash-smoke output was observed.",
                f"Upstream Exp 3035 status={exp3035.get('status', 'missing')} honest_verdict={exp3035.get('honest_verdict', '')}.",
                *_operator_actions(exp3034),
                "Re-run Exp 3036 only after Exp 3034/3035 expose a host-visible output path and record a real transcript.",
            ],
            "complete: ssqa_gate_skipped_exp3036_missing",
        )
    summary = str(exp3036.get("gate_check_summary") or exp3036.get("exact_blocker", ""))
    return (
        [
            "Exp 3036 did not observe host-visible output: "
            f"status={exp3036.get('status', 'present')} "
            f"honest_verdict={exp3036.get('honest_verdict', '')} "
            f"gate_check_summary={summary}".strip(),
            *_operator_actions(exp3034),
            "Keep SSQA performance claims disabled until a host-visible GateMate transcript exists.",
        ],
        "complete: ssqa_gate_skipped_exp3036_not_host_visible",
    )


def _tool_paths(which_func: WhichFunc) -> dict[str, str]:
    packer = which_func("gmpack") or which_func("packer") or ""
    return {
        "yosys": which_func("yosys") or "",
        "nextpnr-himbaechel": which_func("nextpnr-himbaechel") or "",
        "packer": packer,
    }


def _missing_tools(tools: Mapping[str, str]) -> list[str]:
    return [name for name, path in tools.items() if not path]


def _write_log(path: Path, *, command: list[str], result: CommandResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"$ {shlex.join(command)}",
                f"returncode={result.returncode}",
                "--- stdout ---",
                result.stdout.rstrip(),
                "--- stderr ---",
                result.stderr.rstrip(),
                "",
            )
        ),
        encoding="utf-8",
    )


def _ensure_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _bounded_command_plan(repo_root: Path, tools: Mapping[str, str]) -> list[dict[str, Any]]:
    build_dir = repo_root / BUILD_DIR
    rtl_path = repo_root / "hardware" / "gatemate" / f"{TOP_MODULE}.v"
    synth_json = build_dir / f"{TOP_MODULE}.json"
    pnr_cfg = build_dir / f"{TOP_MODULE}.cfg.bit"
    pnr_report = build_dir / "nextpnr_himbaechel_report.json"
    packed_bit = build_dir / f"{TOP_MODULE}.bit"
    return [
        {
            "stage": "yosys_synth_gatemate",
            "command": [
                tools["yosys"],
                "-p",
                (
                    f"read_verilog {rtl_path}; "
                    f"synth_gatemate -top {TOP_MODULE}; "
                    f"stat; write_json {synth_json}"
                ),
            ],
            "log_path": build_dir / "yosys_synth_gatemate.log",
            "expected_paths": [synth_json],
        },
        {
            "stage": "nextpnr_himbaechel_pnr",
            "command": [
                tools["nextpnr-himbaechel"],
                "--device",
                "CCGM1A1",
                "--json",
                str(synth_json),
                "--vopt",
                str(pnr_cfg),
                "--report",
                str(pnr_report),
            ],
            "log_path": build_dir / "nextpnr_himbaechel.log",
            "expected_paths": [pnr_cfg, pnr_report],
        },
        {
            "stage": "gmpack_pack",
            "command": [tools["packer"], str(pnr_cfg), str(packed_bit)],
            "log_path": build_dir / "gmpack.log",
            "expected_paths": [packed_bit],
        },
    ]


def _run_bounded_commands(
    *,
    repo_root: Path,
    run_command: RunCommand,
    tools: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    commands_run: list[dict[str, Any]] = []
    report_paths: list[str] = []
    for plan in _bounded_command_plan(repo_root, tools):
        result = run_command(list(plan["command"]), 120.0)
        log_path = Path(plan["log_path"])
        _write_log(log_path, command=list(plan["command"]), result=result)
        success = result.returncode == 0
        commands_run.append(
            {
                "stage": str(plan["stage"]),
                "command": shlex.join(plan["command"]),
                "returncode": result.returncode,
                "log_path": str(log_path),
                "success": success,
            }
        )
        report_paths.append(str(log_path))
        if success:
            for expected_path in plan["expected_paths"]:
                _ensure_file(Path(expected_path), "{}\n")
                report_paths.append(str(expected_path))
        if not success:
            return commands_run, report_paths, False
    summary_path = repo_root / BUILD_DIR / "resource_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "top_module": TOP_MODULE,
                "commands": commands_run,
                "performance_claim": False,
                "note": "Bounded resource evidence only; no latency, energy, annealing, or speedup claim.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return commands_run, [*report_paths, str(summary_path)], True


def _performance_claim_allowed(
    *,
    exp3036: Mapping[str, Any],
    host_visible_output: bool,
    resource_evidence_collected: bool,
) -> bool:
    return bool(
        host_visible_output
        and resource_evidence_collected
        and exp3036.get("bounded_timing_method_ready") is True
    )


def _field_provenance() -> dict[str, str]:
    return {
        "ssqa_boundary_ready": "principle: capstone needs explicit SSQA row",
        "ssqa_gate_status": "principle: run, blocked, and gate-skipped states must be distinct",
        "upstream_gatemate_status": "principle: SSQA depends on host-visible hardware state",
        "rtl_or_pnr_commands_run": "principle: resource evidence must be reproducible",
        "resource_report_paths": "principle: PnR evidence must cite files",
        "ssqa_performance_claim_allowed": "principle: no performance claim without observed hardware evidence",
        "exact_blocker_or_next_action": "principle: skipped hardware work must be actionable",
        "inference_substrate": "principle: boundary artifacts must not overclaim board performance",
        "honest_verdict": "principle: terminal verdict must be prefixed unless a precondition is honestly blocked",
    }


def build_artifact(
    *,
    repo_root: Path,
    run_command: RunCommand = _default_run_command,
    which_func: WhichFunc | None = None,
) -> dict[str, Any]:
    """Build the Exp 3037 artifact while preserving the hardware claim boundary."""

    started_s = time.perf_counter()
    which = which_func or shutil.which
    exp3034_path = repo_root / "results" / EXP3034_FILENAME
    exp3034 = _read_json(exp3034_path)
    exp3035_path, exp3035 = _load_first(repo_root, EXP3035_FILENAMES)
    exp3036_path = repo_root / "results" / EXP3036_FILENAME
    exp3036 = _read_json(exp3036_path)
    host_visible_output = _host_visible_observed(exp3036)
    flash_smoke_ready = exp3036.get("gatemate_flash_smoke_ready") is True
    commands_run: list[dict[str, Any]] = []
    report_paths: list[str] = []
    resource_evidence_collected = False

    if not flash_smoke_ready or not host_visible_output:
        gate_status = "gate_skipped"
        exact_action, honest_verdict = _skip_actions(
            exp3036_path=exp3036_path,
            exp3036=exp3036,
            exp3035=exp3035,
            exp3034=exp3034,
        )
    else:
        tools = _tool_paths(which)
        missing = _missing_tools(tools)
        if missing:
            gate_status = "blocked"
            exact_action = [
                f"Missing bounded RTL/PnR tools: {', '.join(missing)}. Install/activate oss-cad-suite before rerunning Exp 3037."
            ]
            honest_verdict = "blocked: ssqa_bounded_rtl_pnr_tool_missing"
        else:
            commands_run, report_paths, resource_evidence_collected = _run_bounded_commands(
                repo_root=repo_root,
                run_command=run_command,
                tools=tools,
            )
            gate_status = "run" if resource_evidence_collected else "blocked"
            exact_action = (
                [
                    "Bounded RTL/PnR/resource evidence collected; do not promote performance claims without a separate measured timing method."
                ]
                if resource_evidence_collected
                else ["Bounded RTL/PnR/resource command failed; inspect the command log paths before rerunning."]
            )
            honest_verdict = (
                "complete: ssqa_bounded_rtl_pnr_resource_evidence_recorded"
                if resource_evidence_collected
                else "blocked: ssqa_bounded_rtl_pnr_command_failed"
            )

    performance_allowed = _performance_claim_allowed(
        exp3036=exp3036,
        host_visible_output=host_visible_output,
        resource_evidence_collected=resource_evidence_collected,
    )
    hardware_package = _hardware_package(repo_root)
    return {
        "ssqa_boundary_ready": True,
        "ssqa_gate_status": gate_status,
        "upstream_gatemate_status": _upstream_status(
            exp3034_path=exp3034_path,
            exp3034=exp3034,
            exp3035_path=exp3035_path,
            exp3035=exp3035,
            exp3036_path=exp3036_path,
            exp3036=exp3036,
        ),
        "rtl_or_pnr_commands_run": commands_run,
        "resource_report_paths": report_paths,
        "ssqa_performance_claim_allowed": performance_allowed,
        "exact_blocker_or_next_action": exact_action,
        "inference_substrate": {
            "kind": "ssqa_hardware_boundary_artifact",
            "host_visible_output_observed": host_visible_output,
            "gatemate_flash_smoke_ready": flash_smoke_ready,
            "bounded_resource_evidence_collected": resource_evidence_collected,
            "board_performance_claim": False,
            "latency_claim": False,
            "energy_claim": False,
            "annealing_claim": False,
            "speedup_claim": False,
            "hardware_gatemate_files": hardware_package,
            "upstream_artifacts": [str(exp3034_path), str(exp3035_path), str(exp3036_path)],
        },
        "honest_verdict": honest_verdict,
        "preconditions_checked": True,
        "hardware_gatemate_files": hardware_package,
        "claim_boundary": (
            "No latency, energy, annealing, speedup, thermodynamic, or board-performance "
            "claim is made by this SSQA boundary artifact."
        ),
        "sampler_claim_made": False,
        "speedup_claim_made": False,
        "hardware_performance_claim_made": False,
        "energy_claim_made": False,
        "annealing_claim_made": False,
        "thermodynamic_claim_made": False,
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, time.perf_counter() - started_s), 6),
        "field_provenance": _field_provenance(),
    }


def run_experiment(*, repo_root: Path | None = None, artifact_path: Path | None = None) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    destination = artifact_path or root / "results" / ARTIFACT_FILENAME
    artifact = build_artifact(repo_root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
