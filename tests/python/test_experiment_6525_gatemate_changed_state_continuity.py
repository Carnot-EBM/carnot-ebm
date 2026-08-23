"""Tests for Exp6525 GateMate changed-state continuity.

Spec refs: REQ-HW-6525, SCENARIO-HW-6525-1, SCENARIO-HW-6525-2,
SCENARIO-HW-6525-3, SCENARIO-HW-6525-4.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6525_gatemate_changed_state_continuity as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "hardware" / "spec.md"


class RecordingRunner:
    """REQ-HW-6525 fake runner that proves the command budget mechanically."""

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
    """Deterministic clock used for duration and command UTC ordering."""

    def __init__(self, *values: float):
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def _receipt(**updates: object) -> dict:
    receipt = deepcopy(mod.DEFAULT_VALID_TEST_RECEIPT)
    receipt.update(updates)
    return receipt


def _command_receipt(
    argv: tuple[str, ...],
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    timeout: bool = False,
) -> mod.CommandReceipt:
    return mod.CommandReceipt(
        argv=argv,
        exit_code=exit_code,
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


def test_req_hw_6525_spec_defines_v564_continuity_contract() -> None:
    """REQ-HW-6525: OpenSpec anchors the V564 continuity artifact."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("### REQ-HW-6525", maxsplit=1)[1]
    for marker in (
        "REQ-HW-6525",
        "SCENARIO-HW-6525-1",
        "SCENARIO-HW-6525-2",
        "SCENARIO-HW-6525-3",
        "SCENARIO-HW-6525-4",
        mod.OUTPUT_REL_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_hw_6525_no_new_receipt_runs_zero_hardware_commands() -> None:
    """SCENARIO-HW-6525-1: absent post-Exp6325 physical receipt blocks cleanly."""

    stale = _receipt(receipt_date="20260811")
    planner = _receipt(
        receipt_date="20260823",
        source="planner note: next GateMate task should move the cable",
        operator_authored=False,
    )
    undated = _receipt(receipt_date=None)
    usb_only = _receipt(
        receipt_date="20260823",
        source="operator note 2026-08-23: DirtyJTAG USB enumeration observed",
        changes=[{"field": "usb", "description": "1209:c0ca enumerated"}],
        usb_only=True,
    )
    runner = RecordingRunner()

    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(10.0, 10.75),
        run_date="20260823",
        receipt_candidates=[stale, planner, undated, usb_only],
        protected_before_hashes=mod.protected_file_hashes(REPO_ROOT),
        git_status_text=" M openspec/capabilities/hardware/spec.md\n",
        current_time_utc="2026-08-23T12:00:00Z",
    )

    assert runner.calls == []
    assert artifact["status"] == "blocked_missing_new_physical_receipt"
    assert artifact["honest_verdict"].startswith("blocked_missing_new_physical_receipt:")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["hardware_command_count"] == 0
    assert artifact["command_rows"] == []
    assert artifact["terminal_disposition"] == "blocked_missing_new_physical_receipt"
    assert artifact["gatemate_continuity_slot_complete_score"] == 1.0
    assert artifact["gatemate_bitstream_flashed"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["inference_substrate"] == mod.NO_COMMAND_INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["changed_state_receipt"]["exists"] is False
    assert {row["reject_reason"] for row in artifact["dated_receipt_search_rows"]} == {
        "stale_or_not_newer_than_exp6325",
        "not_operator_authored",
        "undated",
        "usb_only_evidence",
    }
    assert artifact["preconditions_checked"]["git_status"]["short"] == (
        " M openspec/capabilities/hardware/spec.md\n"
    )
    assert "results/experiment_6325_gatemate_dated_receipt_single_detect.json" in (
        artifact["preconditions_checked"]["historical_artifact_paths_and_hashes"]
    )
    assert artifact["preconditions_checked"]["exclusion_state"]["exp3866_preserved"] is True
    assert artifact["gate_check_summary"]["no_hardware_commands_without_new_receipt"] is True
    assert artifact["aggregate_row_recomputation"]["hardware_command_count_recomputed"] == 0
    mod.validate_artifact(artifact)


def test_scenario_hw_6525_valid_receipt_runs_exactly_one_detect_and_stops() -> None:
    """SCENARIO-HW-6525-3: a valid changed-state receipt permits one bounded detect."""

    runner = RecordingRunner({mod.DETECT_COMMAND: [_command_receipt(mod.DETECT_COMMAND, stdout=_detect_hit())]})
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(20.0, 21.0, 22.0, 23.0),
        run_date="20260823",
        receipt_candidates=[_receipt()],
        git_status_text="",
        current_time_utc="2026-08-23T12:00:00Z",
    )

    assert runner.calls == [mod.DETECT_COMMAND]
    assert artifact["status"] == "partial_detect_visible"
    assert artifact["verdict_class"] == "partial"
    assert artifact["hardware_command_count"] == 1
    assert artifact["command_rows"][0]["argv"] == list(mod.DETECT_COMMAND)
    assert artifact["command_rows"][0]["exit_code"] == 0
    assert artifact["command_rows"][0]["stdout_sha256"] == mod.sha256_text(_detect_hit())
    assert artifact["command_rows"][0]["device_identity"]["expected_gatemate_idcode_seen"] is True
    assert artifact["terminal_disposition"] == "detect_visible_nonterminal"
    assert artifact["authorization_decision"]["authorized"] is True
    assert artifact["changed_state_receipt"]["receipt_date"] == "20260823"
    assert artifact["inference_substrate"] == mod.HARDWARE_COMMAND_INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["gatemate_bitstream_flashed"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["aggregate_row_recomputation"]["hardware_command_count_recomputed"] == 1
    mod.validate_artifact(artifact)


def test_scenario_hw_6525_detect_failure_and_timeout_are_terminal_without_retry() -> None:
    """SCENARIO-HW-6525-3: failed and timed-out actions stay blocked and do not retry."""

    failed_runner = RecordingRunner(
        {mod.DETECT_COMMAND: [_command_receipt(mod.DETECT_COMMAND, stderr="dirtyjtag open failed\n", exit_code=1)]}
    )
    failed = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=failed_runner,
        clock=StepClock(30.0, 31.0, 32.0, 33.0),
        run_date="20260823",
        receipt_candidates=[_receipt()],
    )

    timeout_runner = RecordingRunner(
        {
            mod.DETECT_COMMAND: [
                _command_receipt(
                    mod.DETECT_COMMAND,
                    stderr="timed out after 30s",
                    exit_code=124,
                    timeout=True,
                )
            ]
        }
    )
    timeout = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=timeout_runner,
        clock=StepClock(40.0, 41.0, 42.0, 43.0),
        run_date="20260823",
        receipt_candidates=[_receipt()],
    )

    assert failed_runner.calls == [mod.DETECT_COMMAND]
    assert timeout_runner.calls == [mod.DETECT_COMMAND]
    assert failed["status"] == "blocked_action_failed"
    assert failed["verdict_class"] == "blocked"
    assert failed["terminal_disposition"] == "detect_failed"
    assert failed["command_rows"][0]["retry_count"] == 0
    assert timeout["status"] == "blocked_action_timeout"
    assert timeout["terminal_disposition"] == "timeout"
    assert timeout["command_rows"][0]["timeout"] is True
    assert timeout["command_rows"][0]["retry_count"] == 0
    mod.validate_artifact(failed)
    mod.validate_artifact(timeout)


def test_scenario_hw_6525_valid_flash_sets_bitstream_flag_only_with_same_run_evidence() -> None:
    """REQ-HW-6525: same-run authenticated flash evidence is required for flash=true."""

    bitstream = "rtl/gatemate_ising_n16.bit"
    receipt = _receipt(action="flash", bitstream_path=bitstream)
    flash_command = mod.flash_command_for(bitstream)
    runner = RecordingRunner(
        {flash_command: [_command_receipt(flash_command, stdout="JTAG chain ok\nload done\n")]}
    )
    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=runner,
        clock=StepClock(50.0, 51.0, 52.0, 53.0),
        run_date="20260823",
        receipt_candidates=[receipt],
    )

    assert runner.calls == [flash_command]
    assert artifact["status"] == "circular_positive_flash_evidence"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["gatemate_bitstream_flashed"] is True
    assert artifact["command_rows"][0]["action"] == "flash"
    assert artifact["command_rows"][0]["bitstream_identity"]["path"] == bitstream
    assert artifact["command_rows"][0]["bitstream_identity"]["present"] is True
    assert artifact["terminal_disposition"] == "flash_succeeded_same_run"
    assert artifact["hardware_speedup_claim"] is False
    mod.validate_artifact(artifact)


def test_scenario_hw_6525_validation_fails_closed_on_budget_target_and_claim_drift() -> None:
    """SCENARIO-HW-6525-4: invalid command budgets or claims cannot graduate."""

    artifact = mod.build_artifact(
        root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(60.0, 60.5),
        run_date="20260823",
        receipt_candidates=[],
    )

    missing = deepcopy(artifact)
    del missing["status"]
    assert any("missing required fields" in err for err in mod.artifact_schema_errors(missing))

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    assert any("hardware_speedup_claim" in err for err in mod.artifact_schema_errors(speedup))

    flash = deepcopy(artifact)
    flash["gatemate_bitstream_flashed"] = True
    assert any("same-run flash" in err for err in mod.artifact_schema_errors(flash))

    command_without_auth = deepcopy(artifact)
    command_without_auth["hardware_command_count"] = 1
    command_without_auth["command_rows"] = [
        {"argv": list(mod.DETECT_COMMAND), "terminal_disposition": "detect_failed"}
    ]
    errors = mod.artifact_schema_errors(command_without_auth)
    assert any("command count mismatch" in err for err in errors)
    assert any("unauthorized command rows" in err for err in errors)

    two_commands = deepcopy(artifact)
    two_commands["authorization_decision"]["authorized"] = True
    two_commands["hardware_command_count"] = 2
    assert any("single command budget" in err for err in mod.artifact_schema_errors(two_commands))

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = mod.HARDWARE_COMMAND_INFERENCE_SUBSTRATE
    assert any("inference_substrate" in err for err in mod.artifact_schema_errors(bad_substrate))

    protected = deepcopy(artifact)
    protected["protected_files_unchanged"]["all_unchanged"] = False
    assert any("protected files" in err for err in mod.artifact_schema_errors(protected))

    score = deepcopy(artifact)
    score["gatemate_continuity_slot_complete_score"] = 0.0
    assert any("continuity_slot" in err for err in mod.artifact_schema_errors(score))

    principles = deepcopy(artifact)
    del principles["field_principles"]["status"]
    assert any("field_principles" in err for err in mod.artifact_schema_errors(principles))

    provenance = deepcopy(artifact)
    del provenance["field_provenance"]["status"]
    assert any("field_provenance" in err for err in mod.artifact_schema_errors(provenance))

    oracle = deepcopy(artifact)
    oracle["verifier_is_oracle"] = True
    assert any("verifier_is_oracle" in err for err in mod.artifact_schema_errors(oracle))

    metadata = deepcopy(artifact)
    metadata["schema"] = "wrong"
    metadata["spec_refs"] = []
    metadata["random_seed"] = 0
    errors = mod.artifact_schema_errors(metadata)
    assert "schema mismatch" in errors
    assert "spec_refs mismatch" in errors
    assert "random_seed mismatch" in errors

    bad_argv = deepcopy(artifact)
    bad_argv["authorization_decision"]["authorized"] = True
    bad_argv["hardware_command_count"] = 1
    bad_argv["aggregate_row_recomputation"]["hardware_command_count_recomputed"] = 1
    bad_argv["inference_substrate"] = mod.HARDWARE_COMMAND_INFERENCE_SUBSTRATE
    bad_argv["verifier_is_oracle"] = True
    bad_argv["command_rows"] = [
        {
            "argv": ["openFPGALoader", "--scan-usb"],
            "retry_count": 0,
            "stdout_sha256": mod.sha256_text(""),
            "stderr_sha256": mod.sha256_text(""),
        }
    ]
    assert any("allowlisted" in err for err in mod.artifact_schema_errors(bad_argv))

    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)


def test_req_hw_6525_receipt_parser_and_search_locations(tmp_path: Path) -> None:
    """SCENARIO-HW-6525-2: receipt search rejects bad rows and accepts material ones."""

    known = tmp_path / "ops" / "known-issues.md"
    known.parent.mkdir(parents=True)
    known.write_text(
        "\n".join(
            [
                "## 2026-08-23 operator GateMate physical action",
                "```json",
                json.dumps(_receipt(), sort_keys=True),
                "```",
                "## 2026-08-24 planner GateMate idea",
                "Planner-created USB-only note: 1209:c0ca was visible.",
                "## 2026-08-24 unrelated note",
                "No hardware receipt content here.",
            ]
        ),
        encoding="utf-8",
    )
    wishlist = tmp_path / "research-hardware-wishlist.md"
    wishlist.write_text("GateMate preserves the physical block until operator action.\n", encoding="utf-8")

    rows = mod.search_dated_receipts(tmp_path)

    assert [row["path"] for row in rows] == [
        "ops/known-issues.md",
        "ops/known-issues.md",
        "research-hardware-wishlist.md",
    ]
    assert rows[0]["valid"] is True
    assert rows[0]["receipt_date"] == "20260823"
    assert rows[1]["valid"] is False
    assert rows[1]["reject_reason"] == "not_operator_authored"
    assert rows[2]["valid"] is False
    assert rows[2]["reject_reason"] == "undated"
    assert mod.select_changed_state_receipt(rows)["receipt_date"] == "20260823"
    assert mod.parse_receipt_json_block("no json") == {}
    assert mod.parse_receipt_json_block("```json\n{\"bad\":}\n```") == {}


def test_req_hw_6525_helper_fallbacks_and_terminal_classifiers(tmp_path: Path) -> None:
    """REQ-HW-6525: helper fallbacks stay explicit and conservative."""

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

    no_material = mod.row_from_candidate(
        "unit",
        1,
        "operator directive 2026-08-23T12:00:00Z: GateMate note only",
        {
            "exists": True,
            "receipt_date": "20260823",
            "operator_authored": True,
            "source": "operator directive 2026-08-23T12:00:00Z: GateMate note only",
        },
    )
    wrong_target = mod.row_from_candidate(
        "unit",
        2,
        "operator directive 2026-08-23T12:00:00Z: KV260 power changed",
        {
            "exists": True,
            "receipt_date": "20260823",
            "operator_authored": True,
            "board": "AMD Xilinx KV260",
            "changes": [{"field": "power", "description": "KV260 power changed"}],
        },
    )
    assert no_material["reject_reason"] == "no_material_physical_change"
    assert wrong_target["reject_reason"] == "wrong_or_ambiguous_target"

    unsupported = mod.authorization_decision(
        REPO_ROOT,
        {
            "exists": True,
            "action": "erase",
            "receipt_date": "20260823",
            "raw_receipt": {},
        },
    )
    assert unsupported["authorized"] is False
    assert unsupported["reason"] == "unsupported_predeclared_action"

    detect_no_id = mod.terminal_from_command(
        "detect",
        _command_receipt(mod.DETECT_COMMAND, stdout="found 0 devices\n"),
        mod.parse_device_identity("", ""),
    )
    flash_failed = mod.terminal_from_command(
        "flash",
        _command_receipt(mod.flash_command_for("rtl/gatemate_ising_n16.bit"), stdout="no load\n"),
        mod.parse_device_identity("", ""),
    )
    assert detect_no_id == "detect_no_idcode"
    assert flash_failed == "flash_failed"
    assert mod.verdict_class_for_status("disqualified_unauthorized_command") == "disqualified"


def test_req_hw_6525_run_experiment_writes_artifact_and_main_prints(tmp_path: Path, monkeypatch, capsys) -> None:
    """REQ-HW-6525: run_experiment and CLI write the required deliverable."""

    out = mod.run_experiment(
        repo_root=tmp_path,
        source_root=REPO_ROOT,
        command_runner=RecordingRunner(),
        clock=StepClock(70.0, 70.5),
        run_date="20260823",
        receipt_candidates=[],
        git_status_text="",
        current_time_utc="2026-08-23T12:00:00Z",
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
        )

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    rc = mod.main(["--date", "20260823", "--repo-root", str(tmp_path)])
    captured = capsys.readouterr().out
    assert rc == 0
    assert mod.OUTPUT_REL_PATH.name in captured
    assert "hardware_command_count: 0" in captured
    assert "blocked_missing_new_physical_receipt" in captured
