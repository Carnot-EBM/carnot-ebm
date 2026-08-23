"""Tests for Exp6559 GateMate changed-state continuity.

Spec refs: REQ-HW-6559, SCENARIO-HW-6559-1, SCENARIO-HW-6559-2,
SCENARIO-HW-6559-3, SCENARIO-HW-6559-4.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6559_gatemate_changed_state_continuity as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "hardware" / "spec.md"


class RecordingRunner:
    """REQ-HW-6559 fake runner that proves the one-action budget."""

    def __init__(self, receipts: dict[tuple[str, ...], list[mod.CommandReceipt]] | None = None):
        self.receipts = {key: list(value) for key, value in (receipts or {}).items()}
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...], timeout_s: float) -> mod.CommandReceipt:
        assert timeout_s > 0
        argv = tuple(argv)
        self.calls.append(argv)
        if argv not in self.receipts or not self.receipts[argv]:
            raise AssertionError(f"unexpected hardware command: {argv!r}")
        return self.receipts[argv].pop(0)


class StepClock:
    """Deterministic monotonic clock for REQ-HW-6559 action timing."""

    def __init__(self, *values: float):
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def _receipt(**updates: object) -> dict[str, Any]:
    receipt = deepcopy(mod.DEFAULT_VALID_TEST_RECEIPT)
    receipt.update(updates)
    return receipt


def _command_receipt(
    argv: tuple[str, ...],
    *,
    stdout: str = "",
    stderr: str = "",
    exit_status: int = 0,
    timeout: bool = False,
) -> mod.CommandReceipt:
    return mod.CommandReceipt(
        argv=argv,
        exit_status=exit_status,
        stdout=stdout,
        stderr=stderr,
        duration_s=0.25,
        timeout=timeout,
    )


def _detect_hit() -> str:
    return (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        "\tidcode 0x20000001\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
    )


def _tool_identities(*, present: bool = True) -> dict[str, dict[str, Any]]:
    return {
        "openFPGALoader": {
            "present": present,
            "path": "/usr/bin/openFPGALoader" if present else None,
            "sha256": "sha256:" + "0" * 64 if present else None,
            "identity_source": "path_hash_no_hardware_command",
        },
        "yosys": {"present": False, "path": None, "sha256": None},
        "nextpnr-himbaechel": {"present": False, "path": None, "sha256": None},
        "gmpack": {"present": False, "path": None, "sha256": None},
    }


def _blocked_artifact() -> dict[str, Any]:
    return mod.build_artifact(
        root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(10.0, 10.75),
        run_date="20260823",
        receipt_candidates=[],
        protected_before_hashes=mod.protected_file_hashes(REPO_ROOT),
        git_status_text=" M openspec/capabilities/hardware/spec.md\n",
        current_time_utc="2026-08-23T12:00:00Z",
        tool_identities=_tool_identities(),
    )


def test_req_hw_6559_spec_declares_exp6525_boundary_contract() -> None:
    """REQ-HW-6559: OpenSpec owns the V567 continuity contract."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("### REQ-HW-6559", maxsplit=1)[1]
    for marker in (
        "REQ-HW-6559",
        "SCENARIO-HW-6559-1",
        "SCENARIO-HW-6559-2",
        "SCENARIO-HW-6559-3",
        "SCENARIO-HW-6559-4",
        "newer than Exp6525",
        mod.OUTPUT_REL_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_hw_6559_no_new_receipt_runs_zero_hardware_commands() -> None:
    """SCENARIO-HW-6559-1: absent post-Exp6525 receipt blocks cleanly."""

    stale = _receipt(receipt_date="20260823")
    planner = _receipt(
        receipt_date="20260824",
        source="planner note: next GateMate task should move the cable",
        operator_authored=False,
    )
    undated = _receipt(receipt_date=None)
    usb_only = _receipt(
        receipt_date="20260824",
        source="operator note 2026-08-24: DirtyJTAG USB enumeration observed",
        changes=[{"field": "usb", "description": "1209:c0ca enumerated"}],
        usb_only=True,
    )
    runner = RecordingRunner()

    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(20.0, 20.75),
        run_date="20260823",
        receipt_candidates=[stale, planner, undated, usb_only],
        protected_before_hashes=mod.protected_file_hashes(REPO_ROOT),
        git_status_text=" M openspec/capabilities/hardware/spec.md\n",
        current_time_utc="2026-08-23T12:00:00Z",
        tool_identities=_tool_identities(),
    )

    assert runner.calls == []
    assert artifact["status"] == "blocked_missing_new_physical_receipt"
    assert artifact["honest_verdict"].startswith("blocked_missing_new_physical_receipt:")
    assert "newer than Exp6525" in artifact["honest_verdict"]
    assert artifact["verdict_class"] == "blocked"
    assert artifact["operator_physical_state_receipt"]["exists"] is False
    assert artifact["hardware_action_rows"] == []
    assert artifact["terminal_command_receipt"] is None
    assert artifact["zero_command_block_receipt"]["hardware_command_count"] == 0
    assert artifact["zero_command_block_receipt"]["blocked_check"] == (
        "operator_physical_state_receipt.newer_than_exp6525"
    )
    assert artifact["zero_command_block_receipt"]["latest_receipt_date"] == "20260824"
    assert artifact["gatemate_changed_state_slot_complete_score"] == 1.0
    assert artifact["gatemate_hardware_advanced_score"] == 0.0
    assert artifact["inference_substrate"] == mod.NO_COMMAND_INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["claim_boundary"]["performance_claim_made"] is False
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["preconditions_checked"]["git_status"]["short"] == (
        " M openspec/capabilities/hardware/spec.md\n"
    )
    assert artifact["preconditions_checked"]["prior_failure_artifact"]["path"] == (
        "results/experiment_6525_gatemate_changed_state_continuity.json"
    )
    assert (
        artifact["preconditions_checked"]["usb_enumeration_from_existing_receipts"][
            "live_usb_command_run"
        ]
        is False
    )
    assert artifact["gate_check_summary"]["failed_check"] == (
        "operator_physical_state_receipt.newer_than_exp6525"
    )
    assert artifact["aggregate_row_recomputation"]["hardware_command_count_recomputed"] == 0
    assert {
        row["reject_reason"]
        for row in artifact["operator_physical_state_receipt"]["candidate_rows"]
    } == {
        "stale_or_not_newer_than_exp6525",
        "not_operator_authored",
        "undated",
        "usb_only_evidence",
    }
    mod.validate_artifact(artifact)


def test_scenario_hw_6559_valid_receipt_runs_exactly_one_detect_and_stops() -> None:
    """SCENARIO-HW-6559-3: one valid changed-state receipt permits one detect."""

    runner = RecordingRunner(
        {mod.DETECT_COMMAND: [_command_receipt(mod.DETECT_COMMAND, stdout=_detect_hit())]}
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(30.0, 31.0, 32.0, 33.0),
        run_date="20260823",
        receipt_candidates=[_receipt(receipt_date="20260824")],
        git_status_text="",
        current_time_utc="2026-08-23T12:00:00Z",
        tool_identities=_tool_identities(),
    )

    assert runner.calls == [mod.DETECT_COMMAND]
    assert artifact["status"] == "complete_terminal_detect"
    assert artifact["honest_verdict"].startswith("complete_terminal_detect:")
    assert artifact["verdict_class"] is None
    assert artifact["hardware_action_rows"][0]["argv"] == list(mod.DETECT_COMMAND)
    assert artifact["hardware_action_rows"][0]["exit_status"] == 0
    assert artifact["hardware_action_rows"][0]["stdout_sha256"] == mod.sha256_text(_detect_hit())
    assert (
        artifact["hardware_action_rows"][0]["device_identity"]["expected_gatemate_idcode_seen"]
        is True
    )
    assert artifact["terminal_command_receipt"]["terminal_disposition"] == "detect_visible_terminal"
    assert artifact["safe_target_validation_receipt"]["target_ok"] is True
    assert artifact["operator_physical_state_receipt"]["receipt_date"] == "20260824"
    assert artifact["inference_substrate"] == mod.HARDWARE_COMMAND_INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["gatemate_hardware_advanced_score"] == 1.0
    assert artifact["claim_boundary"]["speed_claimed"] is False
    assert artifact["aggregate_row_recomputation"]["hardware_command_count_recomputed"] == 1
    mod.validate_artifact(artifact)


def test_scenario_hw_6559_terminal_failures_and_flash_are_bounded() -> None:
    """SCENARIO-HW-6559-3: failures and flash receipts still stop after one action."""

    failed_runner = RecordingRunner(
        {
            mod.DETECT_COMMAND: [
                _command_receipt(
                    mod.DETECT_COMMAND, stderr="dirtyjtag open failed\n", exit_status=1
                )
            ]
        }
    )
    failed = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=failed_runner,
        clock=StepClock(40.0, 41.0, 42.0, 43.0),
        run_date="20260823",
        receipt_candidates=[_receipt(receipt_date="20260824")],
        tool_identities=_tool_identities(),
    )

    bitstream = "rtl/gatemate_ising_n16.bit"
    flash_command = mod.flash_command_for(bitstream)
    flash_runner = RecordingRunner(
        {flash_command: [_command_receipt(flash_command, stdout="JTAG chain ok\nload done\n")]}
    )
    flashed = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=flash_runner,
        clock=StepClock(50.0, 51.0, 52.0, 53.0),
        run_date="20260823",
        receipt_candidates=[
            _receipt(receipt_date="20260824", action="flash", bitstream_path=bitstream)
        ],
        tool_identities=_tool_identities(),
    )

    assert failed_runner.calls == [mod.DETECT_COMMAND]
    assert failed["status"] == "partial_terminal_action_failed"
    assert failed["verdict_class"] == "partial"
    assert failed["hardware_action_rows"][0]["retry_count"] == 0
    assert failed["gatemate_hardware_advanced_score"] == 1.0
    assert flash_runner.calls == [flash_command]
    assert flashed["status"] == "complete_terminal_flash"
    assert flashed["verdict_class"] is None
    assert flashed["terminal_command_receipt"]["flash_receipt"]["bitstream_sha256"]
    assert flashed["hardware_action_rows"][0]["terminal_disposition"] == "flash_succeeded_terminal"
    mod.validate_artifact(failed)
    mod.validate_artifact(flashed)


def test_scenario_hw_6559_validation_fails_closed_on_budget_target_and_claim_drift() -> None:
    """SCENARIO-HW-6559-4: invalid budgets, provenance, or claims cannot graduate."""

    artifact = _blocked_artifact()

    missing = deepcopy(artifact)
    del missing["status"]
    assert any("missing required fields" in err for err in mod.artifact_schema_errors(missing))

    overclaim = deepcopy(artifact)
    overclaim["claim_boundary"]["speed_claimed"] = True
    assert any("claim_boundary" in err for err in mod.artifact_schema_errors(overclaim))

    command_without_auth = deepcopy(artifact)
    command_without_auth["hardware_action_rows"] = [
        {
            "argv": list(mod.DETECT_COMMAND),
            "terminal_disposition": "detect_failed",
            "retry_count": 0,
        }
    ]
    errors = mod.artifact_schema_errors(command_without_auth)
    assert any("command count mismatch" in err for err in errors)
    assert any("unauthorized command rows" in err for err in errors)

    two_commands = deepcopy(artifact)
    two_commands["safe_target_validation_receipt"]["authorized"] = True
    two_commands["hardware_action_rows"] = [
        {
            "argv": list(mod.DETECT_COMMAND),
            "retry_count": 0,
            "stdout_sha256": "x",
            "stderr_sha256": "y",
        },
        {
            "argv": list(mod.DETECT_COMMAND),
            "retry_count": 0,
            "stdout_sha256": "x",
            "stderr_sha256": "y",
        },
    ]
    assert any("single action budget" in err for err in mod.artifact_schema_errors(two_commands))

    bad_argv = deepcopy(artifact)
    bad_argv["safe_target_validation_receipt"]["authorized"] = True
    bad_argv["safe_target_validation_receipt"]["target_ok"] = True
    bad_argv["hardware_action_rows"] = [
        {
            "argv": ["openFPGALoader", "--scan-usb"],
            "retry_count": 0,
            "stdout_sha256": "x",
            "stderr_sha256": "y",
        }
    ]
    bad_argv["terminal_command_receipt"] = {"terminal_disposition": "detect_failed"}
    assert any("allowlisted" in err for err in mod.artifact_schema_errors(bad_argv))

    protected = deepcopy(artifact)
    protected["protected_files_unchanged"]["all_unchanged"] = False
    assert any("protected files" in err for err in mod.artifact_schema_errors(protected))

    exp3866 = deepcopy(artifact)
    exp3866["exp3866_exclusion_preserved"]["preserved"] = False
    assert any("Exp3866" in err for err in mod.artifact_schema_errors(exp3866))

    oracle = deepcopy(artifact)
    oracle["verifier_is_oracle"] = True
    assert any("verifier_is_oracle" in err for err in mod.artifact_schema_errors(oracle))

    provenance = deepcopy(artifact)
    del provenance["field_provenance"]["status"]
    assert any("field_provenance" in err for err in mod.artifact_schema_errors(provenance))

    score = deepcopy(artifact)
    score["gatemate_changed_state_slot_complete_score"] = 0.0
    assert any("changed_state_slot" in err for err in mod.artifact_schema_errors(score))

    metadata = deepcopy(artifact)
    metadata["schema"] = "wrong"
    metadata["spec_refs"] = []
    metadata["random_seed"] = 0
    errors = mod.artifact_schema_errors(metadata)
    assert "schema mismatch" in errors
    assert "spec_refs mismatch" in errors
    assert "random_seed mismatch" in errors

    terminal_without_command = deepcopy(artifact)
    terminal_without_command["terminal_command_receipt"] = {"terminal_disposition": "detect_failed"}
    assert any(
        "terminal_command_receipt" in err
        for err in mod.artifact_schema_errors(terminal_without_command)
    )

    missing_zero_block = deepcopy(artifact)
    missing_zero_block["zero_command_block_receipt"] = None
    assert any(
        "zero_command_block_receipt" in err
        for err in mod.artifact_schema_errors(missing_zero_block)
    )

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = mod.HARDWARE_COMMAND_INFERENCE_SUBSTRATE
    assert any("inference_substrate" in err for err in mod.artifact_schema_errors(bad_substrate))

    advanced = deepcopy(artifact)
    advanced["gatemate_hardware_advanced_score"] = 1.0
    assert any(
        "gatemate_hardware_advanced_score" in err for err in mod.artifact_schema_errors(advanced)
    )

    retry = deepcopy(artifact)
    retry["safe_target_validation_receipt"]["authorized"] = True
    retry["hardware_action_rows"] = [
        {
            "argv": list(mod.DETECT_COMMAND),
            "retry_count": 1,
            "stdout_sha256": "x",
            "stderr_sha256": "y",
        }
    ]
    assert any("retry_count" in err for err in mod.artifact_schema_errors(retry))

    with pytest.raises(ValueError, match="claim_boundary"):
        mod.validate_artifact(overclaim)


def test_req_hw_6559_receipt_parser_search_and_helper_fallbacks(tmp_path: Path) -> None:
    """SCENARIO-HW-6559-2: parser rejects bad rows and accepts material ones."""

    known = tmp_path / "ops" / "known-issues.md"
    known.parent.mkdir(parents=True)
    known.write_text(
        "\n".join(
            [
                "## 2026-08-24 operator GateMate physical action",
                "```json",
                json.dumps(_receipt(receipt_date="20260824"), sort_keys=True),
                "```",
                "## 2026-08-24 agent GateMate idea",
                "Agent plan: maybe move the GateMate cable later.",
                "## 2026-08-24 unrelated note",
                "No hardware receipt content here.",
            ]
        ),
        encoding="utf-8",
    )
    wishlist = tmp_path / "research-hardware-wishlist.md"
    wishlist.write_text(
        "GateMate preserves the physical block until operator action.\n", encoding="utf-8"
    )

    rows = mod.search_dated_receipts(tmp_path, "20260823")

    assert [row["path"] for row in rows] == [
        "ops/known-issues.md",
        "ops/known-issues.md",
        "research-hardware-wishlist.md",
    ]
    assert rows[0]["valid"] is True
    assert rows[0]["receipt_date"] == "20260824"
    assert rows[1]["valid"] is False
    assert rows[1]["reject_reason"] == "not_operator_authored"
    assert rows[2]["valid"] is False
    assert rows[2]["reject_reason"] == "undated"
    assert mod.select_physical_state_receipt(rows)["receipt_date"] == "20260824"
    assert mod.parse_receipt_json_block("no json") == {}
    assert mod.parse_receipt_json_block('```json\n{"bad":}\n```') == {}

    wrong_target = mod.row_from_candidate(
        "unit",
        2,
        "operator directive 2026-08-24T12:00:00Z: KV260 power changed",
        {
            "exists": True,
            "receipt_date": "20260824",
            "operator_authored": True,
            "board": "AMD Xilinx KV260",
            "changes": [{"field": "power", "description": "KV260 power changed"}],
        },
        "20260823",
    )
    assert wrong_target["reject_reason"] == "wrong_or_ambiguous_target"

    no_material = mod.row_from_candidate(
        "unit",
        3,
        "operator directive 2026-08-24T12:00:00Z: GateMate note only with DirtyJTAG named",
        {
            "exists": True,
            "receipt_date": "20260824",
            "operator_authored": True,
            "source": "operator directive 2026-08-24T12:00:00Z: GateMate note only with DirtyJTAG named",
        },
        "20260823",
    )
    assert no_material["reject_reason"] == "no_material_physical_change"

    assert mod.path_receipt(tmp_path, "missing.json") == {
        "path": "missing.json",
        "present": False,
        "bytes": 0,
        "sha256": None,
    }
    assert mod.read_json_object(tmp_path, "missing.json") == {}
    assert mod.bitstream_identity(tmp_path, None) == {
        "path": None,
        "present": False,
        "safe_relative_path": False,
        "sha256": None,
    }

    assert (
        mod.terminal_from_command(
            "detect",
            _command_receipt(mod.DETECT_COMMAND, timeout=True),
            mod.parse_device_identity("", ""),
        )
        == "timeout"
    )
    assert (
        mod.terminal_from_command(
            "detect",
            _command_receipt(mod.DETECT_COMMAND, stdout="found 0 devices\n"),
            mod.parse_device_identity("", ""),
        )
        == "detect_missing_idcode"
    )
    assert (
        mod.terminal_from_command(
            "flash",
            _command_receipt(
                mod.flash_command_for("rtl/gatemate_ising_n16.bit"), stdout="no load\n"
            ),
            mod.parse_device_identity("", ""),
        )
        == "flash_failed"
    )
    assert mod.verdict_class_for_status("disqualified_unauthorized_command") == "disqualified"

    unsupported = mod.authorization_decision(
        REPO_ROOT,
        {"exists": True, "action": "erase", "receipt_date": "20260824", "raw_receipt": {}},
        _tool_identities(),
    )
    assert unsupported["authorized"] is False
    assert unsupported["reason"] == "unsupported_predeclared_action"

    missing_tool = mod.authorization_decision(
        REPO_ROOT,
        {"exists": True, "action": "detect", "receipt_date": "20260824", "raw_receipt": {}},
        _tool_identities(present=False),
    )
    assert missing_tool["authorized"] is False
    assert missing_tool["reason"] == "openfpgaloader_missing"

    safe_block = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(90.0, 90.5),
        run_date="20260823",
        receipt_candidates=[_receipt(receipt_date="20260824")],
        tool_identities=_tool_identities(present=False),
    )
    assert safe_block["status"] == "blocked_safe_target_validation"
    assert (
        safe_block["gate_check_summary"]["failed_check"]
        == "safe_target_validation.openfpgaloader_missing"
    )
    mod.validate_artifact(safe_block)

    usb_root = tmp_path / "usb-root"
    (usb_root / "ops").mkdir(parents=True)
    (usb_root / "ops" / "hardware-bringup-prep.md").write_text(
        "DirtyJTAG programmer enumerated as 1209:c0ca\n",
        encoding="utf-8",
    )
    usb_receipt = mod.usb_enumeration_from_existing_receipts(usb_root)
    assert usb_receipt["live_usb_command_run"] is False
    assert usb_receipt["observed_in_receipts"] is True


def test_req_hw_6559_run_experiment_writes_artifact_and_main_prints(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """REQ-HW-6559: run_experiment and CLI write the required deliverable."""

    out = mod.run_experiment(
        repo_root=tmp_path,
        source_root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(70.0, 70.5),
        run_date="20260823",
        receipt_candidates=[],
        git_status_text="",
        current_time_utc="2026-08-23T12:00:00Z",
        tool_identities=_tool_identities(),
    )
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert out == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["status"] == "blocked_missing_new_physical_receipt"
    assert artifact["result_path"] == mod.OUTPUT_REL_PATH.as_posix()
    mod.validate_artifact(artifact)

    real_run_experiment = mod.run_experiment

    def fake_run_experiment(*, repo_root: Path, run_date: str) -> Path:
        return real_run_experiment(
            repo_root=tmp_path,
            source_root=REPO_ROOT,
            command_runner=RecordingRunner(),
            clock=StepClock(80.0, 80.5),
            run_date=run_date,
            receipt_candidates=[],
            git_status_text="",
            current_time_utc="2026-08-23T12:00:00Z",
            tool_identities=_tool_identities(),
        )

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    rc = mod.main(["--date", "20260823", "--repo-root", str(tmp_path)])
    captured = capsys.readouterr().out
    assert rc == 0
    assert mod.OUTPUT_REL_PATH.name in captured
    assert "hardware_action_count: 0" in captured
    assert "blocked_missing_new_physical_receipt" in captured
