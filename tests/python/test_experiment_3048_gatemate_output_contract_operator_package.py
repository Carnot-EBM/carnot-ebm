"""Tests for Exp 3048 GateMate output contract operator package.

Spec refs: REQ-HW-088, SCENARIO-HW-088.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.experiment_3048_gatemate_output_contract_operator_package import (
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


def _write_exp3034(repo_root: Path, *, ready: bool = False) -> None:
    row = {
        "signal_name": "done",
        "rtl_source": str(repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v") + ":output done",
        "ccf_binding": "IO_EB_B7" if ready else "",
        "host_read_command": (
            ".venv/bin/python scripts/gatemate_done_gpio_reader.py --expect done=1"
            if ready
            else ""
        ),
        "expected_transcript": ["done=1 PASS"] if ready else "blocked: no expected transcript",
        "blocker_status": "ready" if ready else "blocked_missing_physical_pinout",
    }
    _write_json(
        repo_root / "results" / "experiment_3034_gatemate_output_contract_pinout_decision_v1.json",
        {
            "gatemate_output_contract_ready": ready,
            "host_visible_io_plan_ready": ready,
            "selected_output_path": "led_gpio_done_status" if ready else "explicit_no_ready_contract",
            "pinout_table": [
                row,
                {
                    "signal_name": "spin_out[15:0]",
                    "rtl_source": str(repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v")
                    + ":output spin_out",
                    "ccf_binding": "",
                    "host_read_command": "",
                    "expected_transcript": "blocked: no expected transcript",
                    "blocker_status": "blocked_missing_physical_pinout",
                },
            ],
            "host_reader_command": row["host_read_command"]
            if ready
            else "blocked_no_host_reader_command: explicit_no_ready_contract",
            "exact_operator_action_required": []
            if ready
            else [
                "Provide an authoritative GateMate A1-EVB-2M output pinout and commit a CCF Pin_out binding for done or a deterministic status bit.",
                "Choose and commit the matching host reader command: GPIO/LED read, UART serial decode, or JTAG-readable status command.",
                "Keep downstream flash smoke gated until the reader command has an expected pass/fail transcript.",
            ],
            "honest_verdict": "complete: gatemate_output_contract_ready"
            if ready
            else "complete: blocked_gatemate_output_contract_pinout_missing",
        },
    )


def _write_gate_package(repo_root: Path, *, bound: bool = False, reader: bool = False) -> None:
    gate_dir = repo_root / "hardware" / "gatemate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "ising_n16_gatemate.v").write_text(
        "module ising_n16_gatemate(input clk, output done, output [15:0] spin_out); endmodule\n",
        encoding="utf-8",
    )
    (gate_dir / "ising_n16_gatemate.ccf").write_text(
        (
            "# Authoritative GateMate A1-EVB-2M output pinout for done status.\n"
            "Pin_out done Loc = IO_EB_B7\n"
        )
        if bound
        else (
            "# GateMate CCGM1A1 build-only constraints.\n"
            "# This repository does not yet contain an authoritative GateMate A1-EVB-2M pin map.\n"
        ),
        encoding="utf-8",
    )
    if reader:
        scripts = repo_root / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "gatemate_done_gpio_reader.py").write_text(
            "read_gpio('done')\n", encoding="utf-8"
        )


def _which_from(paths: dict[str, str]):
    def which(name: str) -> str | None:
        return paths.get(name)

    return which


def test_req_hw_088_spec_entry_present() -> None:
    """REQ-HW-088: the FPGA spec anchors the Exp 3048 operator package."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-088" in spec
    assert "SCENARIO-HW-088" in spec
    assert ARTIFACT_FILENAME in spec


def test_scenario_hw_088_missing_authority_writes_blocked_operator_package(tmp_path: Path) -> None:
    """SCENARIO-HW-088: missing CCF pinout and reader become a clean skip package."""
    _write_gate_package(tmp_path)
    _write_exp3034(tmp_path, ready=False)

    artifact = build_artifact(
        repo_root=tmp_path,
        which_func=_which_from({"openFPGALoader": "/suite/bin/openFPGALoader"}),
    )

    assert [field for field in REQUIRED_FIELDS if field not in artifact] == []
    assert artifact["gatemate_output_contract_ready"] is False
    assert artifact["host_visible_io_plan_ready"] is False
    assert artifact["selected_output_signal"] == "done"
    assert artifact["ccf_binding"] == {}
    assert artifact["host_reader_command"] == ""
    assert artifact["expected_transcript"] == []
    assert any("GateMate A1-EVB-2M output pinout" in item for item in artifact["missing_operator_actions"])
    assert any("host reader command" in item for item in artifact["missing_operator_actions"])
    assert artifact["hardware_execution_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["inference_substrate"]["model_inference"] is False
    assert artifact["inference_substrate"]["flash_attempted"] is False
    assert artifact["safety_limits"]["downstream_flash_gate_open"] is False
    assert artifact["authority_search"]["tool_availability"]["openFPGALoader"] == "/suite/bin/openFPGALoader"
    assert artifact["honest_verdict"] == "complete: blocked_gatemate_output_contract_authority_missing"


def test_req_hw_088_ready_package_requires_binding_reader_and_transcript(tmp_path: Path) -> None:
    """REQ-HW-088: a ready package requires concrete CCF, reader, and transcript."""
    _write_gate_package(tmp_path, bound=True, reader=True)
    _write_exp3034(tmp_path, ready=True)

    artifact = build_artifact(
        repo_root=tmp_path,
        which_func=_which_from(
            {
                "openFPGALoader": "/suite/bin/openFPGALoader",
                "yosys": "/suite/bin/yosys",
                "gmpack": "/suite/bin/gmpack",
            }
        ),
    )

    assert artifact["gatemate_output_contract_ready"] is True
    assert artifact["host_visible_io_plan_ready"] is True
    assert artifact["selected_output_signal"] == "done"
    assert artifact["ccf_binding"]["pin"] == "IO_EB_B7"
    assert artifact["ccf_binding"]["line"] == "Pin_out done Loc = IO_EB_B7"
    assert artifact["host_reader_command"] == ".venv/bin/python scripts/gatemate_done_gpio_reader.py --expect done=1"
    assert artifact["expected_transcript"] == ["done=1 PASS"]
    assert artifact["missing_operator_actions"] == []
    assert artifact["safety_limits"]["downstream_flash_gate_open"] is True
    assert artifact["safety_limits"]["exp3049_gate"] == "require_gatemate_output_contract_ready_true"
    assert artifact["hardware_execution_claim_made"] is False
    assert artifact["honest_verdict"] == "complete: gatemate_output_contract_operator_package_ready"


def test_req_hw_088_bound_signal_without_reader_stays_not_host_visible(tmp_path: Path) -> None:
    """REQ-HW-088: physical binding alone never opens the host-visible IO plan."""
    _write_gate_package(tmp_path, bound=True, reader=False)
    _write_exp3034(tmp_path, ready=False)

    artifact = build_artifact(repo_root=tmp_path, which_func=_which_from({}))

    assert artifact["selected_output_signal"] == "done"
    assert artifact["ccf_binding"]["pin"] == "IO_EB_B7"
    assert artifact["host_reader_command"] == ""
    assert artifact["expected_transcript"] == []
    assert artifact["gatemate_output_contract_ready"] is False
    assert artifact["host_visible_io_plan_ready"] is False
    assert any("host reader command" in item for item in artifact["missing_operator_actions"])
    assert artifact["authority_search"]["host_reader_candidates"] == []


def test_req_hw_088_repo_reader_candidate_can_complete_bound_signal(tmp_path: Path) -> None:
    """REQ-HW-088: a committed reader script can provide the concrete command."""
    _write_gate_package(tmp_path, bound=True, reader=True)
    _write_exp3034(tmp_path, ready=False)

    artifact = build_artifact(repo_root=tmp_path, which_func=_which_from({}))

    assert artifact["gatemate_output_contract_ready"] is True
    assert artifact["host_visible_io_plan_ready"] is True
    assert artifact["host_reader_command"].endswith("scripts/gatemate_done_gpio_reader.py --expect done=1")
    assert artifact["expected_transcript"] == ["done=1 PASS"]
    assert artifact["authority_search"]["host_reader_candidates"]


def test_req_hw_088_malformed_or_missing_exp3034_preserves_blocker(tmp_path: Path) -> None:
    """REQ-HW-088: unreadable upstream evidence cannot become a ready contract."""
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "experiment_3034_gatemate_output_contract_pinout_decision_v1.json").write_text(
        "{not-json", encoding="utf-8"
    )

    artifact = build_artifact(repo_root=tmp_path, which_func=_which_from({}))

    assert artifact["selected_output_signal"] == "done"
    assert artifact["upstream_exp3034"]["available"] is False
    assert artifact["ccf_binding"] == {}
    assert artifact["host_reader_command"] == ""
    assert artifact["expected_transcript"] == []
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_hw_088_reader_without_binding_still_requests_transcript(tmp_path: Path) -> None:
    """REQ-HW-088: partial reader evidence does not bypass the CCF binding gate."""
    (tmp_path / "CLAUDE.md").write_text("GateMate A1-EVB local operator notes\n", encoding="utf-8")
    gate_dir = tmp_path / "hardware" / "gatemate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "status.ccf").write_text("# no Pin_out yet\n", encoding="utf-8")
    scripts = tmp_path / "scripts"
    (scripts / "cache").mkdir(parents=True)
    (scripts / "gatemate_status_flag.tmp").write_text("read_gpio('status_flag')\n", encoding="utf-8")
    (scripts / "unrelated.py").write_text("print('no status reader here')\n", encoding="utf-8")
    _write_json(
        tmp_path / "results" / "experiment_3034_gatemate_output_contract_pinout_decision_v1.json",
        {
            "pinout_table": [
                {
                    "signal_name": "status_flag",
                    "host_read_command": ".venv/bin/python scripts/gatemate_status_reader.py --expect status_flag=1",
                    "expected_transcript": "blocked: no expected transcript",
                }
            ],
            "host_reader_command": "",
            "honest_verdict": "complete: custom_status_flag_reader_partial",
        },
    )

    artifact = build_artifact(repo_root=tmp_path, which_func=_which_from({}))

    assert artifact["selected_output_signal"] == "status_flag"
    assert artifact["ccf_binding"] == {}
    assert artifact["host_reader_command"] == ""
    assert any("expected pass/fail transcript" in item for item in artifact["missing_operator_actions"])
    assert str(tmp_path / "CLAUDE.md") in artifact["authority_search"]["local_docs_scanned"]


def test_scenario_hw_088_run_experiment_writes_stable_json(tmp_path: Path) -> None:
    """SCENARIO-HW-088: run_experiment writes the required v1 artifact."""
    _write_gate_package(tmp_path)
    _write_exp3034(tmp_path, ready=False)
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        which_func=_which_from({"openFPGALoader": "/suite/bin/openFPGALoader"}),
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert [field for field in REQUIRED_FIELDS if field not in loaded] == []
