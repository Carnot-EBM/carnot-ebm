"""Focused tests for the V627 GateMate changed-state continuity receipt.

Spec refs: REQ-HARDWARE-7146, SCENARIO-HARDWARE-7146-1,
SCENARIO-HARDWARE-7146-2, SCENARIO-HARDWARE-7146-3,
SCENARIO-HARDWARE-7146-4.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import experiment_7146_v627_gatemate_changed_state as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/hardware/spec.md"


class RecordingRunner:
    """Record fake board calls so tests can prove the action boundary."""

    def __init__(self, receipts: list[exp.CommandResult] | None = None) -> None:
        self.receipts = list(receipts or [])
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(self, argv: tuple[str, ...], timeout_s: float) -> exp.CommandResult:
        self.calls.append((tuple(argv), timeout_s))
        if not self.receipts:
            raise AssertionError(f"unexpected hardware command: {argv!r}")
        return self.receipts.pop(0)


class RecordingWriter:
    """Keep every artifact checkpoint and also write valid JSON."""

    def __init__(self) -> None:
        self.checkpoints: list[dict[str, Any]] = []

    def __call__(self, artifact: dict[str, Any], path: Path) -> Path:
        snapshot = deepcopy(artifact)
        self.checkpoints.append(snapshot)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def _command(
    *,
    return_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    timeout: bool = False,
) -> exp.CommandResult:
    return exp.CommandResult(
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        timeout=timeout,
        duration_s=0.25,
    )


def _detect_output(idcode: str = exp.EXPECTED_IDCODE) -> str:
    return (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        f"\tidcode {idcode}\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
    )


def _receipt(**updates: object) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "receipt_date": "20260908",
        "operator_authored": True,
        "source": "operator directive 2026-09-08T09:00:00Z",
        "board": exp.EXPECTED_BOARD,
        "board_present": True,
        "power_state": "operator power-cycled GateMate after Exp6559",
        "usb_jtag_cable_state": "operator reseated the onboard DirtyJTAG USB-C cable",
        "host_path": "/dev/bus/usb/003/011",
        "intended_recovery_action": "detect_then_existing_n16_smoke",
        "changed_physical_fields": ["power", "usb_jtag_cable_state"],
    }
    receipt.update(updates)
    return receipt


def _root(tmp_path: Path, *, receipt_text: str = "") -> Path:
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "rtl").mkdir()
    bitstream = root / exp.SMOKE_BITSTREAM_REL_PATH
    bitstream.parent.mkdir(parents=True)
    bitstream.write_bytes(b"fixed-gatemate-n16-bitstream")
    (root / exp.SMOKE_RTL_REL_PATH).write_text("module gatemate_ising_n16; endmodule\n")
    (root / exp.EXP6559_REL_PATH).write_text(
        json.dumps({"run_date": "20260823", "honest_verdict": "blocked_missing_receipt"}),
        encoding="utf-8",
    )
    (root / exp.EXP3866_REL_PATH).write_text(
        json.dumps(
            {
                "bitstream_path": str(bitstream),
                "bitstream_sha256": exp.sha256_bytes(bitstream.read_bytes()).removeprefix(
                    "sha256:"
                ),
                "honest_verdict": "excluded_historical_evidence",
            }
        ),
        encoding="utf-8",
    )
    for relative in (exp.EXP6325_REL_PATH, exp.EXP6525_REL_PATH):
        (root / relative).write_text("{}\n", encoding="utf-8")
    (root / exp.EXCLUSION_REL_PATH).write_text("retired: []\n", encoding="utf-8")
    (root / "ops/known-issues.md").write_text(receipt_text, encoding="utf-8")
    (root / "research-hardware-wishlist.md").write_text("", encoding="utf-8")
    (root / "ops/hardware-bringup-prep.md").write_text("", encoding="utf-8")
    (root / "ops/operator-followup.md").write_text("", encoding="utf-8")
    return root


def _build(
    root: Path,
    *,
    candidates: list[dict[str, Any]] | None,
    runner: RecordingRunner | None = None,
    writer: RecordingWriter | None = None,
    tool_present: bool = True,
) -> dict[str, Any]:
    output = root / exp.RESULT_PATH
    return exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=output,
        receipt_candidates=candidates,
        command_runner=runner or RecordingRunner(),
        checkpoint_writer=writer or RecordingWriter(),
        tool_identity={
            "present": tool_present,
            "path": "/usr/bin/openFPGALoader" if tool_present else None,
            "sha256": "sha256:" + "0" * 64 if tool_present else None,
        },
        utc_now=iter(
            [
                "2026-09-08T09:00:01Z",
                "2026-09-08T09:00:02Z",
                "2026-09-08T09:00:03Z",
                "2026-09-08T09:00:04Z",
            ]
        ).__next__,
    )


def test_req_hardware_7146_spec_declares_the_full_contract() -> None:
    """REQ-HARDWARE-7146 owns every required field and scenario."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("### REQ-HARDWARE-7146", maxsplit=1)[1]
    for marker in (
        "SCENARIO-HARDWARE-7146-1",
        "SCENARIO-HARDWARE-7146-2",
        "SCENARIO-HARDWARE-7146-3",
        "SCENARIO-HARDWARE-7146-4",
        "openFPGALoader -c dirtyJtag --detect",
        "newer than Exp6559",
        exp.RESULT_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp.PROMPT_REQUIRED_FIELDS:
        assert f"`{field}`" in section


def test_scenario_hardware_7146_no_receipt_writes_first_and_runs_zero_commands(
    tmp_path: Path,
) -> None:
    """SCENARIO-HARDWARE-7146-1 writes a complete zero-command block."""

    root = _root(tmp_path)
    runner = RecordingRunner()
    writer = RecordingWriter()
    artifact = _build(root, candidates=[], runner=runner, writer=writer)

    assert runner.calls == []
    assert len(writer.checkpoints) >= 3
    first = writer.checkpoints[0]
    assert set(first) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert first["command_rows"] == []
    assert first["hardware_command_count"] == 0
    assert first["inference_substrate_class"] == "blocked_no_run"
    assert all(checkpoint["command_rows"] == [] for checkpoint in writer.checkpoints)
    assert artifact["receipt_cutoff_experiment"]["experiment"] == "Exp6559"
    assert artifact["receipt_cutoff_experiment"]["run_date"] == "20260823"
    assert artifact["receipt_newer_than_exp6559_score"] == 0.0
    assert artifact["hardware_command_count"] == 0
    assert artifact["detect_rows"] == []
    assert artifact["identity_rows"] == []
    assert artifact["smoke_rows"] == []
    assert artifact["inference_substrate"] == exp.NO_COMMAND_SUBSTRATE
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["execution_venue"] == "host"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"] == {
        "failed_check": "receipt_newer_than_exp6559",
        "expected_value": 1.0,
        "observed_value": 0.0,
        "passed": False,
        "receipt_cutoff_experiment": "Exp6559",
        "receipt_cutoff_date": "20260823",
    }
    assert artifact["first_failure_stop_score"] == 1.0
    assert artifact["bitstream_redesigned"] is False
    assert artifact["exclusion_manifest_modified"] is False
    assert artifact["gatemate_terminal_receipt_score"] == 1.0
    assert artifact["verifier_is_oracle"] is False
    assert not exp.validate_artifact(artifact)


def test_scenario_hardware_7146_dry_run_parser_rejects_false_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-HARDWARE-7146-2 rejects stale, agent, and incomplete rows."""

    root = _root(tmp_path)
    candidates = [
        _receipt(receipt_date="20260823"),
        _receipt(operator_authored=False, source="agent-authored GateMate plan"),
        _receipt(changed_physical_fields=["software"], power_state="tool version changed"),
        _receipt(usb_jtag_cable_state=None),
        _receipt(host_path=None),
        _receipt(board_present=False),
        _receipt(intended_recovery_action="flash_redesigned_bitstream"),
        _receipt(board="AMD/Xilinx KV260"),
        _receipt(receipt_date="20260909"),
        _receipt(),
    ]

    rows, selected = exp.audit_receipts(
        root,
        cutoff_date="20260823",
        run_date=exp.RUN_DATE,
        candidates=candidates,
        dry_run=True,
    )

    assert [row["reject_reason"] for row in rows[:-1]] == [
        "stale_or_not_newer_than_exp6559",
        "not_operator_authored",
        "no_changed_physical_state",
        "missing_usb_jtag_cable_state",
        "missing_host_path",
        "board_not_present",
        "invalid_recovery_action",
        "wrong_board",
        "future_dated_receipt",
    ]
    assert rows[-1]["valid"] is True
    assert selected["exists"] is True
    assert selected["receipt_date"] == exp.RUN_DATE
    with pytest.raises(ValueError, match="dry_run"):
        exp.audit_receipts(
            root,
            cutoff_date="20260823",
            run_date=exp.RUN_DATE,
            candidates=[],
            dry_run=False,
        )

    markdown = "\n".join(
        [
            "## Agent GateMate plan 2026-09-08",
            "The agent says the operator should move a cable later.",
            "## Operator GateMate receipt 2026-09-08",
            "```json",
            json.dumps(_receipt()),
            "```",
        ]
    )
    (root / "ops/known-issues.md").write_text(markdown, encoding="utf-8")
    scanned, scanned_selected = exp.audit_receipts(
        root, cutoff_date="20260823", run_date=exp.RUN_DATE
    )
    assert scanned[0]["reject_reason"] == "not_structured_receipt"
    assert scanned[1]["valid"] is True
    assert scanned_selected["exists"] is True


@pytest.mark.parametrize(
    ("result", "failure_reason"),
    [
        (_command(return_code=1, stderr="DirtyJTAG open failed"), "detect_return_code"),
        (_command(timeout=True, stderr="timeout"), "detect_timeout"),
        (_command(stdout="found 0 devices\n"), "unexpected_gatemate_identity"),
        (_command(stdout=_detect_output("0xffffffff")), "unexpected_gatemate_identity"),
        (_command(stdout=_detect_output() + _detect_output()), "unexpected_gatemate_identity"),
    ],
)
def test_scenario_hardware_7146_detect_failure_stops_after_one_action(
    tmp_path: Path,
    result: exp.CommandResult,
    failure_reason: str,
) -> None:
    """SCENARIO-HARDWARE-7146-3 never retries an unclean detect."""

    root = _root(tmp_path)
    runner = RecordingRunner([result])
    artifact = _build(root, candidates=[_receipt()], runner=runner)

    assert [call[0] for call in runner.calls] == [exp.DETECT_COMMAND]
    assert artifact["hardware_command_count"] == 1
    assert artifact["command_rows"] == artifact["detect_rows"]
    assert artifact["smoke_rows"] == []
    assert artifact["command_rows"][0]["failure_reason"] == failure_reason
    assert artifact["command_rows"][0]["stdout"] == result.stdout
    assert artifact["command_rows"][0]["stderr"] == result.stderr
    assert artifact["command_rows"][0]["return_code"] == result.return_code
    assert artifact["command_rows"][0]["timeout"] is result.timeout
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("partial_")
    assert artifact["first_failure_stop_score"] == 1.0
    assert artifact["execution_venue"] == "gatemate"
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert not exp.validate_artifact(artifact)


def test_scenario_hardware_7146_clean_detect_runs_only_fixed_smoke(
    tmp_path: Path,
) -> None:
    """SCENARIO-HARDWARE-7146-4 appends detect then the fixed n=16 smoke."""

    root = _root(tmp_path)
    runner = RecordingRunner(
        [_command(stdout=_detect_output()), _command(stdout="Load SRAM via JTAG: 100%\nDone\n")]
    )
    writer = RecordingWriter()
    artifact = _build(root, candidates=[_receipt()], runner=runner, writer=writer)

    expected_smoke = exp.smoke_command(root)
    assert [call[0] for call in runner.calls] == [exp.DETECT_COMMAND, expected_smoke]
    assert [row["action"] for row in artifact["command_rows"]] == ["detect", "n16_smoke"]
    assert [row["action_index"] for row in artifact["command_rows"]] == [1, 2]
    assert artifact["detect_rows"] == [artifact["command_rows"][0]]
    assert artifact["smoke_rows"] == [artifact["command_rows"][1]]
    assert artifact["identity_rows"][0]["clean_expected_identity"] is True
    assert (
        artifact["smoke_rows"][0]["bitstream_sha256"]
        == artifact["source_artifact_hashes"][exp.SMOKE_BITSTREAM_REL_PATH.as_posix()]["sha256"]
    )
    assert artifact["hardware_command_count"] == 2
    assert artifact["receipt_newer_than_exp6559_score"] == 1.0
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "gatemate"
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("positive_")
    assert artifact["gatemate_terminal_receipt_score"] == 1.0
    assert artifact["bitstream_redesigned"] is False
    lengths = [len(item["command_rows"]) for item in writer.checkpoints]
    assert lengths == sorted(lengths)
    assert lengths[-1] == 2
    assert not exp.validate_artifact(artifact)

    ledger = exp.ActionLedger()
    ledger.append({"action": "detect", "success": False})
    with pytest.raises(RuntimeError, match="first failed action"):
        ledger.append({"action": "n16_smoke", "success": True})


def test_req_hardware_7146_tool_and_smoke_failures_remain_bounded(tmp_path: Path) -> None:
    """REQ-HARDWARE-7146 blocks missing tools and stops after smoke failure."""

    root = _root(tmp_path)
    no_tool_runner = RecordingRunner()
    no_tool = _build(
        root,
        candidates=[_receipt()],
        runner=no_tool_runner,
        tool_present=False,
    )
    assert no_tool_runner.calls == []
    assert no_tool["verdict_class"] == "blocked"
    assert no_tool["gate_check_summary"]["failed_check"] == "openfpgaloader_available"
    assert no_tool["inference_substrate_class"] == "blocked_no_run"

    runner = RecordingRunner(
        [_command(stdout=_detect_output()), _command(return_code=1, stderr="flash failed")]
    )
    failed = _build(root, candidates=[_receipt()], runner=runner)
    assert len(runner.calls) == 2
    assert failed["smoke_rows"][0]["success"] is False
    assert failed["smoke_rows"][0]["failure_reason"] == "n16_smoke_return_code"
    assert failed["verdict_class"] == "partial"
    assert failed["first_failure_stop_score"] == 1.0
    assert not exp.validate_artifact(no_tool)
    assert not exp.validate_artifact(failed)


def test_req_hardware_7146_validation_and_cli_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-HARDWARE-7146 rejects forged ledgers, fields, and CLI dates."""

    root = _root(tmp_path)
    artifact = _build(root, candidates=[])
    for field, value in (
        ("schema", "wrong"),
        ("random_seed", 0),
        ("bitstream_redesigned", True),
        ("exclusion_manifest_modified", True),
        ("verifier_is_oracle", True),
        ("hardware_command_count", 1),
        ("first_failure_stop_score", 0.0),
        ("gatemate_terminal_receipt_score", 0.0),
        ("reproducibility_checksum", "sha256:bad"),
    ):
        forged = deepcopy(artifact)
        forged[field] = value
        assert exp.validate_artifact(forged), field

    missing = deepcopy(artifact)
    del missing["receipt_rows"]
    assert "missing required fields" in exp.validate_artifact(missing)[0]

    bad_principles = deepcopy(artifact)
    del bad_principles["field_principles"]["command_rows"]
    assert any("field_principles" in error for error in exp.validate_artifact(bad_principles))

    bad_block = deepcopy(artifact)
    bad_block["gate_check_summary"] = {}
    assert any("blocked gate_check_summary" in error for error in exp.validate_artifact(bad_block))

    bad_command = deepcopy(artifact)
    bad_command["command_rows"] = [
        {"action": "detect", "argv": ["openFPGALoader", "--scan-usb"], "success": True}
    ]
    bad_command["hardware_command_count"] = 1
    assert any("exact detect command" in error for error in exp.validate_artifact(bad_command))

    bad_class = deepcopy(artifact)
    bad_class["verdict_class"] = "positive"
    assert any("honest_verdict" in error for error in exp.validate_artifact(bad_class))

    for field, value, message in (
        ("spec_refs", [], "spec_refs"),
        ("run_date", "20260909", "run_date"),
        ("inference_substrate", "hardware_smoke", "zero-command inference_substrate"),
        ("inference_substrate_class", "no_model_load", "zero-command inference_substrate_class"),
        ("execution_venue", "gatemate", "zero-command execution_venue"),
        ("receipt_cutoff_experiment", {}, "receipt_cutoff_experiment"),
    ):
        forged = deepcopy(artifact)
        forged[field] = value
        assert any(message in error for error in exp.validate_artifact(forged))

    non_list = deepcopy(artifact)
    non_list["command_rows"] = "not-a-list"
    assert any("command_rows must be a list" in error for error in exp.validate_artifact(non_list))

    too_many = deepcopy(artifact)
    too_many["command_rows"] = [
        {"action": "wrong", "argv": list(exp.DETECT_COMMAND), "success": True},
        {"action": "n16_smoke", "argv": ["wrong"], "success": True},
        {"action": "n16_smoke", "argv": ["wrong"], "success": True},
    ]
    too_many["hardware_command_count"] = 3
    too_many["detect_rows"] = []
    too_many["smoke_rows"] = too_many["command_rows"][1:]
    assert any("exceeds" in error for error in exp.validate_artifact(too_many))
    assert any("first action is not detect" in error for error in exp.validate_artifact(too_many))

    positive_runner = RecordingRunner(
        [_command(stdout=_detect_output()), _command(stdout="Done\n")]
    )
    positive = _build(root, candidates=[_receipt()], runner=positive_runner)
    bad_second = deepcopy(positive)
    bad_second["command_rows"][1]["argv"] = ["openFPGALoader", "--wrong"]
    bad_second["smoke_rows"][0]["argv"] = ["openFPGALoader", "--wrong"]
    assert any("fixed n16 smoke" in error for error in exp.validate_artifact(bad_second))
    no_identity = deepcopy(positive)
    no_identity["identity_rows"] = []
    assert any("clean expected detect" in error for error in exp.validate_artifact(no_identity))
    smoke_mismatch = deepcopy(positive)
    smoke_mismatch["smoke_rows"] = []
    assert any("smoke_rows mismatch" in error for error in exp.validate_artifact(smoke_mismatch))

    output = root / "results/cli.json"
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["hardware_command_count"] == 0
    assert exp.main(["--date", "20260909", "--root", str(root)]) == 2
    assert exp.main(["--validate", str(output)]) == 0
    invalid = root / "results/invalid.json"
    invalid.write_text("{}\n", encoding="utf-8")
    assert exp.main(["--validate", str(invalid)]) == 1

    invalid_from_build = deepcopy(artifact)
    invalid_from_build["schema"] = "wrong"
    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: invalid_from_build)
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(root)]) == 1
    assert "blocked_no_new_operator" in capsys.readouterr().out


def test_req_hardware_7146_helper_fallbacks_and_terminal_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HARDWARE-7146 covers missing inputs without weakening the gate."""

    root = _root(tmp_path)
    missing_receipt = exp.path_receipt(root, Path("missing"))
    assert missing_receipt == {
        "path": "missing",
        "present": False,
        "bytes": 0,
        "sha256": None,
    }
    malformed = root / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert exp.read_json_object(malformed) == {}
    assert exp.read_json_object(root / "absent.json") == {}
    non_object = root / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    assert exp.read_json_object(non_object) == {}
    assert exp.normalized_date("2026-09-08") == exp.RUN_DATE
    assert exp.normalized_date(None) is None

    changed = _receipt(
        receipt_date=None,
        source="operator directive 2026-09-08T09:00:00Z",
        changed_physical_fields=[],
        changes=[{"field": "power"}],
    )
    changed_row = exp.receipt_row(
        changed,
        source_path="unit",
        row_index=1,
        cutoff_date="20260823",
        run_date=exp.RUN_DATE,
        structured=True,
    )
    assert changed_row["valid"] is True
    undated = exp.receipt_row(
        _receipt(receipt_date=None, source="operator directive without a date"),
        source_path="unit",
        row_index=2,
        cutoff_date="20260823",
        run_date=exp.RUN_DATE,
        structured=True,
    )
    assert undated["reject_reason"] == "undated_receipt"
    no_power = exp.receipt_row(
        _receipt(power_state=None),
        source_path="unit",
        row_index=3,
        cutoff_date="20260823",
        run_date=exp.RUN_DATE,
        structured=True,
    )
    assert no_power["reject_reason"] == "missing_power_state"
    assert exp._json_receipts("```json\n{bad}\n```") == []

    (root / "ops/operator-followup.md").unlink()
    (root / "research-hardware-wishlist.md").write_text(
        "## Unrelated note\nNo physical device appears here.\n", encoding="utf-8"
    )
    rows, selected = exp.audit_receipts(root, cutoff_date="20260823", run_date=exp.RUN_DATE)
    assert rows == []
    assert selected["exists"] is False

    monkeypatch.setattr(exp.shutil, "which", lambda name: None)
    assert exp.binary_identity("openFPGALoader") == {
        "present": False,
        "path": None,
        "sha256": None,
    }
    binary = root / "openFPGALoader"
    binary.write_bytes(b"binary")
    monkeypatch.setattr(exp.shutil, "which", lambda name: str(binary))
    assert exp.binary_identity("openFPGALoader")["sha256"] == exp.sha256_bytes(b"binary")
    assert exp.utc_now().endswith("Z")

    (root / exp.EXP6559_REL_PATH).unlink()
    cutoff_writer = RecordingWriter()
    cutoff_block = exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=Path("results/relative.json"),
        receipt_candidates=[],
        command_runner=RecordingRunner(),
        checkpoint_writer=cutoff_writer,
    )
    assert cutoff_block["honest_verdict"] == "blocked_exp6559_receipt_cutoff_unavailable"
    assert cutoff_block["gate_check_summary"]["failed_check"] == "exp6559_cutoff_available"

    root = _root(tmp_path / "invalid-smoke")
    (root / exp.SMOKE_BITSTREAM_REL_PATH).write_bytes(b"changed-bitstream")
    runner = RecordingRunner([_command(stdout=_detect_output())])
    source_block = _build(root, candidates=[_receipt()], runner=runner)
    assert len(runner.calls) == 1
    assert source_block["honest_verdict"] == (
        "partial_detect_clean_existing_n16_smoke_source_invalid"
    )
    assert source_block["smoke_rows"] == []
    assert not exp.validate_artifact(cutoff_block)
    assert not exp.validate_artifact(source_block)
