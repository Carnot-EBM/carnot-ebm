"""Tests for Exp 3023 explicit SSQA gate artifact.

Spec refs: REQ-HW-085, SCENARIO-HW-085.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.experiment_3023_ssqa_explicit_gate_artifact_and_rtl_report import (
    ARTIFACT_FILENAME,
    REQUIRED_FIELDS,
    build_artifact,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gate_package(repo_root: Path) -> None:
    hw_dir = repo_root / "hardware" / "gatemate"
    hw_dir.mkdir(parents=True, exist_ok=True)
    (hw_dir / "ising_n16_gatemate.v").write_text(
        "module ising_n16_gatemate(input clk, output done, output [15:0] spin_out); endmodule\n",
        encoding="utf-8",
    )
    (hw_dir / "ising_n16_gatemate.ccf").write_text(
        "# build-only constraints; no physical output pin\n",
        encoding="utf-8",
    )
    (hw_dir / "ising_n16_gatemate_test_vector.json").write_text(
        json.dumps({"schema": "carnot.gatemate.ising_n16_test_vector.v1"}),
        encoding="utf-8",
    )


def test_req_hw_085_spec_entry_present() -> None:
    """REQ-HW-085: the FPGA spec anchors the Exp 3023 artifact contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-085" in spec
    assert "SCENARIO-HW-085" in spec
    assert ARTIFACT_FILENAME in spec


def test_scenario_hw_085_blocked_exp3022_writes_gate_skipped_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-085: blocked host-visible IO yields explicit SSQA gate skip."""
    _write_gate_package(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3022_gatemate_transport_flash_smoke_v3.json",
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp3021-gatemate-rtl-ccf-host-visible-transport-shim",
                    "artifact_field": "gatemate_transport_rtl_ready",
                    "actual": False,
                    "expected": True,
                    "passed": False,
                }
            ],
        },
    )
    _write_json(
        tmp_path
        / "results"
        / "experiment_3021_gatemate_rtl_ccf_host_visible_transport_shim_v1.json",
        {
            "gatemate_transport_rtl_ready": False,
            "host_visible_io_plan_ready": False,
            "io_transport_path": "blocked:gatemate_pinout_missing_no_physical_pinout_for_done_spin_out",
            "rtl_paths": [str(tmp_path / "hardware" / "gatemate" / "ising_n16_gatemate.v")],
            "transcript_paths": [str(tmp_path / "logs" / "exp3021" / "yosys_lint.txt")],
            "honest_verdict": "complete: blocked_gatemate_transport_pinout_missing",
        },
    )

    artifact = build_artifact(repo_root=tmp_path)

    assert [field for field in REQUIRED_FIELDS if field not in artifact] == []
    assert artifact["ssqa_artifact_written"] is True
    assert artifact["preconditions_checked"] is True
    assert artifact["upstream_host_visible_io_ready"] is False
    assert artifact["ssqa_gate_status"] == "gate_skipped"
    assert artifact["ssqa_rtl_pnr_report_ready"] is False
    assert artifact["projection_only"] is True
    assert artifact["sampler_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["boltzmann_claim_made"] is False
    assert artifact["thermodynamic_claim_made"] is False
    assert artifact["fpga_acceleration_claim_made"] is False
    assert artifact["pnr_or_synthesis_attempted"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert "exp3022" in artifact["honest_verdict"]
    assert artifact["upstream_status"] == "blocked"
    assert artifact["upstream_blocked_at_layer"] == "conductor_pre_gate"
    assert "1 of 1 gate(s) failed" in artifact["upstream_gate_check_summary"]
    assert artifact["rtl_path"].endswith("hardware/gatemate/ising_n16_gatemate.v")
    assert artifact["pnr_report_path"] == ""
    assert artifact["resource_report_path"] == ""
    assert artifact["smoke_hook_paths"] == [
        str(tmp_path / "hardware" / "gatemate" / "ising_n16_gatemate_test_vector.json")
    ]


def test_req_hw_085_missing_exp3022_is_still_a_written_gate_skip(tmp_path: Path) -> None:
    """REQ-HW-085: a missing upstream artifact is explicit, not silent."""
    _write_gate_package(tmp_path)

    artifact = build_artifact(repo_root=tmp_path)

    assert artifact["ssqa_artifact_written"] is True
    assert artifact["ssqa_gate_status"] == "gate_skipped"
    assert artifact["upstream_artifact_available"] is False
    assert artifact["upstream_host_visible_io_ready"] is False
    assert artifact["upstream_status"] == "missing"
    assert artifact["honest_verdict"] == "complete: ssqa_gate_skipped_exp3022_missing"
    assert artifact["sampler_claim_made"] is False
    assert artifact["speedup_claim_made"] is False


def test_req_hw_085_ready_upstream_requires_inspectable_rtl_pnr_resource_paths(
    tmp_path: Path,
) -> None:
    """REQ-HW-085: ready status is allowed only with concrete evidence paths."""
    _write_gate_package(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3022_gatemate_transport_flash_smoke_v3.json",
        {
            "status": "complete",
            "host_visible_io_ready": True,
            "io_transport_path": "uart_tx:/tmp/gatemate_uart_reader.py",
            "honest_verdict": "complete: host_visible_io_ready",
        },
    )
    rtl_path = tmp_path / "hardware" / "gatemate" / "ssqa_dual_bram_register_map.v"
    pnr_path = tmp_path / "build" / "gatemate" / "experiment_3023_ssqa" / "ssqa_dual_bram.pnr.log"
    resource_path = (
        tmp_path / "build" / "gatemate" / "experiment_3023_ssqa" / "ssqa_dual_bram.resources.json"
    )
    rtl_path.write_text("module ssqa_dual_bram_register_map(input clk); endmodule\n", encoding="utf-8")
    pnr_path.parent.mkdir(parents=True, exist_ok=True)
    pnr_path.write_text("Info: bounded PnR transcript\n", encoding="utf-8")
    resource_path.write_text('{"luts": 12, "bram": 2}\n', encoding="utf-8")

    artifact = build_artifact(repo_root=tmp_path)

    assert artifact["upstream_host_visible_io_ready"] is True
    assert artifact["ssqa_gate_status"] == "rtl_pnr_report_ready"
    assert artifact["ssqa_rtl_pnr_report_ready"] is True
    assert artifact["projection_only"] is False
    assert artifact["rtl_path"] == str(rtl_path)
    assert artifact["pnr_report_path"] == str(pnr_path)
    assert artifact["resource_report_path"] == str(resource_path)
    assert artifact["pnr_or_synthesis_attempted"] is False
    assert artifact["honest_verdict"] == "complete: ssqa_rtl_pnr_resource_report_ready"
    assert artifact["sampler_claim_made"] is False
    assert artifact["speedup_claim_made"] is False


def test_req_hw_085_ready_upstream_without_reports_stays_projection_only(tmp_path: Path) -> None:
    """REQ-HW-085: host IO readiness alone is not an SSQA RTL/PnR claim."""
    _write_gate_package(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_3022_gatemate_transport_flash_smoke_v3.json",
        {
            "status": "complete",
            "host_visible_io_ready": True,
            "honest_verdict": "complete: host_visible_io_ready",
        },
    )

    artifact = build_artifact(repo_root=tmp_path)

    assert artifact["upstream_host_visible_io_ready"] is True
    assert artifact["ssqa_gate_status"] == "rtl_pnr_report_missing"
    assert artifact["ssqa_rtl_pnr_report_ready"] is False
    assert artifact["projection_only"] is True
    assert artifact["pnr_report_path"] == ""
    assert artifact["resource_report_path"] == ""
    assert artifact["honest_verdict"] == (
        "complete: ssqa_host_visible_io_ready_but_rtl_pnr_resource_evidence_missing"
    )
    assert artifact["sampler_claim_made"] is False
    assert artifact["speedup_claim_made"] is False


def test_scenario_hw_085_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-085: run_experiment writes the stable v1 JSON artifact."""
    _write_gate_package(tmp_path)
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(repo_root=tmp_path, artifact_path=destination)

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert [field for field in REQUIRED_FIELDS if field not in loaded] == []
    assert loaded["ssqa_artifact_written"] is True
    assert loaded["ssqa_gate_status"] == "gate_skipped"
