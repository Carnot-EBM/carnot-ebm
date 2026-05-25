"""Tests for Exp 3037 SSQA bounded RTL/PnR gate artifact.

Spec refs: REQ-HW-087, SCENARIO-HW-087.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.experiment_3037_ssqa_bounded_rtl_pnr_gate_artifact import (
    ARTIFACT_FILENAME,
    REQUIRED_FIELDS,
    CommandResult,
    build_artifact,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gate_package(repo_root: Path) -> None:
    gate_dir = repo_root / "hardware" / "gatemate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "ising_n16_gatemate.v").write_text(
        "module ising_n16_gatemate(input clk, output done, output [15:0] spin_out); endmodule\n",
        encoding="utf-8",
    )
    (gate_dir / "ising_n16_gatemate.ccf").write_text(
        "# build-only constraints; no physical output binding\n",
        encoding="utf-8",
    )
    (gate_dir / "ising_n16_gatemate_test_vector.json").write_text(
        json.dumps({"schema": "carnot.gatemate.ising_n16_test_vector.v1"}) + "\n",
        encoding="utf-8",
    )


def _write_blocked_3034_3035(repo_root: Path) -> None:
    _write_json(
        repo_root / "results" / "experiment_3034_gatemate_output_contract_pinout_decision_v1.json",
        {
            "gatemate_output_contract_ready": False,
            "host_visible_io_plan_ready": False,
            "honest_verdict": "complete: blocked_gatemate_output_contract_pinout_missing",
            "exact_operator_action_required": [
                "Provide an authoritative GateMate A1-EVB-2M output pinout.",
                "Choose and commit the matching host reader command.",
            ],
        },
    )
    _write_json(
        repo_root / "results" / "experiment_3035_gatemate_output_shim_rtl_ccf_sim.json",
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp3034-gatemate-output-contract-pinout-decision."
                "gatemate_output_contract_ready"
            ),
        },
    )


def _write_ready_3036(repo_root: Path, **extra: object) -> None:
    payload: dict[str, object] = {
        "status": "complete",
        "gatemate_flash_smoke_ready": True,
        "host_visible_output_observed": True,
        "host_visible_transcript_path": str(repo_root / "logs" / "exp3036" / "smoke.txt"),
        "honest_verdict": "complete: gatemate_host_visible_flash_smoke_ready",
    }
    payload.update(extra)
    _write_json(
        repo_root / "results" / "experiment_3036_gatemate_host_visible_flash_smoke_v4.json",
        payload,
    )


def _which_from(paths: dict[str, str]):
    def which(name: str) -> str | None:
        return paths.get(name)

    return which


def _successful_runner():
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        assert timeout_s > 0
        calls.append(tuple(args))
        exe = Path(args[0]).name
        if exe == "yosys":
            return CommandResult(0, "Yosys synth_gatemate complete\nNumber of cells: 42\n", "")
        if exe == "nextpnr-himbaechel":
            return CommandResult(0, "Info: Device utilisation: 42 LUTs\n", "")
        if exe == "gmpack":
            return CommandResult(0, "GateMate pack complete\n", "")
        raise AssertionError(f"unexpected command: {args}")

    return run, calls


def _failing_nextpnr_runner():
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        assert timeout_s > 0
        calls.append(tuple(args))
        exe = Path(args[0]).name
        if exe == "yosys":
            return CommandResult(0, "Yosys synth_gatemate complete\n", "")
        if exe == "nextpnr-himbaechel":
            return CommandResult(1, "", "routing failed\n")
        raise AssertionError(f"unexpected command after failed PnR: {args}")

    return run, calls


def test_req_hw_087_spec_entry_present() -> None:
    """REQ-HW-087: the FPGA spec anchors the Exp 3037 boundary artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-087" in spec
    assert "SCENARIO-HW-087" in spec
    assert ARTIFACT_FILENAME in spec


def test_scenario_hw_087_missing_exp3036_writes_gate_skipped_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-087: missing flash-smoke output yields an explicit SSQA skip."""
    _write_gate_package(tmp_path)
    _write_blocked_3034_3035(tmp_path)

    artifact = build_artifact(repo_root=tmp_path)

    assert [field for field in REQUIRED_FIELDS if field not in artifact] == []
    assert artifact["ssqa_boundary_ready"] is True
    assert artifact["ssqa_gate_status"] == "gate_skipped"
    assert artifact["upstream_gatemate_status"]["exp3034"]["available"] is True
    assert artifact["upstream_gatemate_status"]["exp3035"]["status"] == "blocked"
    assert artifact["upstream_gatemate_status"]["exp3036"]["available"] is False
    assert artifact["upstream_gatemate_status"]["exp3036"]["status"] == "missing"
    assert artifact["rtl_or_pnr_commands_run"] == []
    assert artifact["resource_report_paths"] == []
    assert artifact["ssqa_performance_claim_allowed"] is False
    assert artifact["inference_substrate"]["host_visible_output_observed"] is False
    assert artifact["inference_substrate"]["board_performance_claim"] is False
    assert any("Exp 3036 artifact missing" in item for item in artifact["exact_blocker_or_next_action"])
    assert any("GateMate A1-EVB-2M output pinout" in item for item in artifact["exact_blocker_or_next_action"])
    assert artifact["honest_verdict"] == "complete: ssqa_gate_skipped_exp3036_missing"


def test_req_hw_087_blocked_exp3036_without_host_visible_output_runs_no_commands(
    tmp_path: Path,
) -> None:
    """REQ-HW-087: blocked Exp 3036 preserves the no-performance-claim boundary."""
    _write_gate_package(tmp_path)
    _write_blocked_3034_3035(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3036_gatemate_host_visible_flash_smoke_v4.json",
        {
            "status": "blocked",
            "gatemate_flash_smoke_ready": False,
            "host_visible_output_observed": False,
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "exp3035 output shim was blocked",
        },
    )

    artifact = build_artifact(repo_root=tmp_path)

    assert artifact["ssqa_gate_status"] == "gate_skipped"
    assert artifact["upstream_gatemate_status"]["exp3036"]["available"] is True
    assert artifact["upstream_gatemate_status"]["exp3036"]["gatemate_flash_smoke_ready"] is False
    assert artifact["rtl_or_pnr_commands_run"] == []
    assert artifact["resource_report_paths"] == []
    assert artifact["ssqa_performance_claim_allowed"] is False
    assert "blocked_gate_check_failed" in " ".join(artifact["exact_blocker_or_next_action"])
    assert artifact["honest_verdict"] == "complete: ssqa_gate_skipped_exp3036_not_host_visible"


def test_req_hw_087_ready_exp3036_runs_bounded_resource_commands(tmp_path: Path) -> None:
    """REQ-HW-087: ready flash smoke permits only bounded RTL/PnR evidence."""
    _write_gate_package(tmp_path)
    _write_ready_3036(tmp_path)
    runner, calls = _successful_runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from(
            {
                "yosys": "/suite/bin/yosys",
                "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
                "gmpack": "/suite/bin/gmpack",
            }
        ),
    )

    assert artifact["ssqa_gate_status"] == "run"
    assert artifact["ssqa_boundary_ready"] is True
    assert [Path(call[0]).name for call in calls] == ["yosys", "nextpnr-himbaechel", "gmpack"]
    assert len(artifact["rtl_or_pnr_commands_run"]) == 3
    assert all(item["returncode"] == 0 for item in artifact["rtl_or_pnr_commands_run"])
    assert any("synth_gatemate" in item["command"] for item in artifact["rtl_or_pnr_commands_run"])
    assert len(artifact["resource_report_paths"]) >= 3
    assert all(Path(path).exists() for path in artifact["resource_report_paths"])
    assert artifact["ssqa_performance_claim_allowed"] is False
    assert artifact["inference_substrate"]["bounded_resource_evidence_collected"] is True
    assert artifact["inference_substrate"]["latency_claim"] is False
    assert artifact["exact_blocker_or_next_action"] == [
        "Bounded RTL/PnR/resource evidence collected; do not promote performance claims without a separate measured timing method."
    ]
    assert artifact["honest_verdict"] == "complete: ssqa_bounded_rtl_pnr_resource_evidence_recorded"


def test_req_hw_087_ready_exp3036_blocks_when_tool_missing(tmp_path: Path) -> None:
    """REQ-HW-087: ready upstream still blocks if bounded PnR tools are absent."""
    _write_gate_package(tmp_path)
    _write_ready_3036(tmp_path)

    artifact = build_artifact(repo_root=tmp_path, which_func=_which_from({}))

    assert artifact["ssqa_gate_status"] == "blocked"
    assert artifact["rtl_or_pnr_commands_run"] == []
    assert artifact["resource_report_paths"] == []
    assert artifact["ssqa_performance_claim_allowed"] is False
    assert "Missing bounded RTL/PnR tools" in artifact["exact_blocker_or_next_action"][0]
    assert artifact["honest_verdict"] == "blocked: ssqa_bounded_rtl_pnr_tool_missing"


def test_req_hw_087_failed_bounded_command_is_blocked_with_logs(tmp_path: Path) -> None:
    """REQ-HW-087: failed bounded resource commands are blocked, not upgraded."""
    _write_gate_package(tmp_path)
    _write_ready_3036(tmp_path)
    runner, calls = _failing_nextpnr_runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from(
            {
                "yosys": "/suite/bin/yosys",
                "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
                "gmpack": "/suite/bin/gmpack",
            }
        ),
    )

    assert [Path(call[0]).name for call in calls] == ["yosys", "nextpnr-himbaechel"]
    assert artifact["ssqa_gate_status"] == "blocked"
    assert len(artifact["rtl_or_pnr_commands_run"]) == 2
    assert artifact["rtl_or_pnr_commands_run"][-1]["returncode"] == 1
    assert artifact["ssqa_performance_claim_allowed"] is False
    assert all(Path(path).exists() for path in artifact["resource_report_paths"])
    assert artifact["exact_blocker_or_next_action"] == [
        "Bounded RTL/PnR/resource command failed; inspect the command log paths before rerunning."
    ]
    assert artifact["honest_verdict"] == "blocked: ssqa_bounded_rtl_pnr_command_failed"


def test_scenario_hw_087_run_experiment_writes_stable_json(tmp_path: Path) -> None:
    """SCENARIO-HW-087: run_experiment writes the required v2 artifact."""
    _write_gate_package(tmp_path)
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(repo_root=tmp_path, artifact_path=destination)

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert [field for field in REQUIRED_FIELDS if field not in loaded] == []
    assert loaded["ssqa_boundary_ready"] is True
    assert loaded["ssqa_gate_status"] == "gate_skipped"
