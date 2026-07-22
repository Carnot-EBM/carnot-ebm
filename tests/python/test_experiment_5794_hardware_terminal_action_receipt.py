"""Tests for Exp5794 cached hardware terminal-action receipts.

Spec refs: REQ-HW-5794, SCENARIO-HW-5794.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_5794_hardware_terminal_action_receipt as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/hardware/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5794_hardware_terminal_action_receipt.py")


class RecordingRunner:
    """SCENARIO-HW-5794 fake runner that refuses unexpected board commands."""

    def __init__(self, probes: dict[tuple[str, ...], mod.CommandProbe] | None = None) -> None:
        self.probes = dict(probes or {})
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> mod.CommandProbe:
        assert timeout_s > 0.0
        rendered = mod.command_to_string(command)
        assert "mmcblk" not in rendered.lower()
        assert "/dev/disk" not in rendered.lower()
        assert " flash" not in f" {rendered.lower()} "
        assert "--write" not in rendered.lower()
        self.commands.append(command)
        if command not in self.probes:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command]


def _probe(
    command: tuple[str, ...],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _test_exit_codes() -> dict[str, int]:
    return {
        TEST_PATH.as_posix(): 0,
        ".venv/bin/pytest tests/python -q": 0,
        "python scripts/check_spec_coverage.py": 0,
        ".venv/bin/python scripts/root_clutter_sweep.py --check": 0,
    }


def _unchanged_artifact() -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        command_runner=RecordingRunner(),
        run_date="20260722",
        test_exit_codes=_test_exit_codes(),
    )


def test_req_hw_5794_spec_declares_exact_cached_receipt_contract() -> None:
    """REQ-HW-5794: OpenSpec anchors exact paths, no-repeat, and no-claim fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5794") : spec.index("### SCENARIO-HW-5794")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-HW-5794",
        str(mod.RESULT_RELATIVE_PATH),
        "declared paths and declared byte hashes",
        "If a board's hash is unchanged",
        "commands_run=[]",
        "no_authenticated_local_execution_surface",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_hw_5794_unchanged_hashes_skip_all_hardware_commands() -> None:
    """SCENARIO-HW-5794: unchanged precondition hashes produce cached reconciliation only."""

    runner = RecordingRunner()
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        run_date="20260722",
        test_exit_codes=_test_exit_codes(),
    )

    assert runner.commands == []
    assert artifact["status"] == "complete_cached_hardware_reconciliation_no_board_commands"
    assert artifact["spec_refs"] == ["REQ-HW-5794", "SCENARIO-HW-5794"]
    assert artifact["random_seed"] == 5794
    assert artifact["milestone"] == "2026.07.516"
    assert artifact["commands_run"] == []
    assert {row["board"] for row in artifact["commands_skipped"]} == {
        "kv260",
        "polarfire",
        "gatemate",
    }
    assert artifact["changed_preconditions"] == {
        "kv260": False,
        "polarfire": False,
        "gatemate": False,
    }
    assert artifact["precondition_hashes_previous"] == artifact["precondition_hashes_current"]
    assert artifact["storage_write_count"] == 0
    assert artifact["flash_write_count"] == 0
    assert artifact["speedup_claimed"] is False
    assert artifact["energy_claimed"] is False
    assert artifact["production_ready_claimed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert "no_speedup_claim" in artifact["honest_verdict"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_5794_resolves_canonical_artifacts_by_exact_declared_path_and_hash() -> None:
    """REQ-HW-5794: canonical board evidence is exact-path and byte-hash pinned."""

    artifact = _unchanged_artifact()
    canonical = artifact["canonical_hardware_artifacts"]
    hashes = artifact["hardware_artifact_hashes"]

    assert set(canonical) == {"kv260", "polarfire", "gatemate"}
    for board, declaration in mod.CANONICAL_BOARD_ARTIFACTS.items():
        row = canonical[board]
        expected_path = declaration["path"].as_posix()
        expected_hash = mod.file_sha256(REPO / declaration["path"])

        assert row["path"] == expected_path
        assert row["selection_method"] == "declared_exact_path_no_glob_no_mtime"
        assert row["sha256"] == expected_hash
        assert hashes[expected_path] == expected_hash
        assert row["board"] == board

    assert artifact["preconditions_checked"][0]["resource"] == "safety_boundaries_recorded"
    assert all(row["selection_method"] != "mtime" for row in canonical.values())
    mod.validate_artifact(artifact)


def test_req_hw_5794_per_board_state_machines_match_cached_evidence() -> None:
    """REQ-HW-5794: KV260, PolarFire, and GateMate states remain independent."""

    artifact = _unchanged_artifact()

    assert artifact["board_state_machine"]["kv260"]["current"] == (
        "cached_ssh_bitstream_poc_no_performance_claim"
    )
    assert artifact["kv260_state"]["ssh_state"] == "cached_reachable"
    assert artifact["kv260_state"]["bitstream_state"] == (
        "cached_carnot_ising_v4_alias_carnot_ising_v2_n64"
    )
    assert artifact["kv260_state"]["host_storage_or_block_device_accessed"] is False
    assert artifact["kv260_state"]["host_storage_access_prohibited"] is True

    assert artifact["board_state_machine"]["polarfire"]["current"] == (
        "cached_ssh_reachable_terminal_workload_missing_passive_cooling_limited"
    )
    assert artifact["polarfire_state"]["authentication_state"] == "cached_ssh_reachable"
    assert artifact["polarfire_state"]["terminal_carnot_workload_state"] == (
        "missing_terminal_hash_verified_dispatch"
    )
    assert artifact["polarfire_state"]["passive_cooling"]["max_unaided_duration_s"] == 300
    assert artifact["temperature_duration_receipts"]["polarfire"]["command_run"] is False

    assert artifact["board_state_machine"]["gatemate"]["current"] == (
        "cached_dirtyjtag_cable_or_port_block_no_flash"
    )
    assert artifact["gatemate_state"]["raw_idcode"] == "0xffffffff"
    assert artifact["gatemate_state"]["dirtyjtag_state"] == "cached_visible_no_gm1ax_idcode"
    assert artifact["gatemate_state"]["flash_state"] == "not_authorized_not_run"
    mod.validate_artifact(artifact)


def test_req_hw_5794_changed_precondition_runs_only_small_authorized_check() -> None:
    """REQ-HW-5794: a changed GateMate physical setup permits only bounded detect."""

    baseline = _unchanged_artifact()
    runner = RecordingRunner(
        {
            mod.SAFE_PROBE_COMMANDS["gatemate"].command: _probe(
                mod.SAFE_PROBE_COMMANDS["gatemate"].command,
                stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                stderr="no idcode found\n",
                duration_s=0.02,
            )
        }
    )
    artifact = mod.build_artifact(
        root=REPO,
        command_runner=runner,
        run_date="20260722",
        previous_precondition_hashes=baseline["precondition_hashes_current"],
        operator_authorization={
            "gatemate": {
                "physical_setup_changed": True,
                "bounded_non_destructive_check_authorized": True,
            }
        },
        test_exit_codes=_test_exit_codes(),
    )

    assert runner.commands == [mod.SAFE_PROBE_COMMANDS["gatemate"].command]
    assert artifact["changed_preconditions"] == {
        "kv260": False,
        "polarfire": False,
        "gatemate": True,
    }
    assert [row["board"] for row in artifact["commands_run"]] == ["gatemate"]
    receipt = artifact["commands_run"][0]
    assert receipt["command"] == mod.command_to_string(mod.SAFE_PROBE_COMMANDS["gatemate"].command)
    assert receipt["target"] == "dirtyjtag_idcode"
    assert receipt["timeout_s"] == mod.SAFE_PROBE_COMMANDS["gatemate"].timeout_s
    assert receipt["stdout_sha256"] == mod.sha256_text(
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
    )
    assert receipt["stderr_sha256"] == mod.sha256_text("no idcode found\n")
    assert receipt["stop_state"] == "stopped_after_non_destructive_check"
    assert artifact["commands_skipped"][0]["board"] == "kv260"
    assert artifact["commands_skipped"][1]["board"] == "polarfire"
    assert artifact["storage_write_count"] == 0
    assert artifact["flash_write_count"] == 0
    mod.validate_artifact(artifact)


def test_req_hw_5794_operator_packets_and_proprietary_access_are_precise() -> None:
    """REQ-HW-5794: blocked and proprietary lanes become action packets, not probes."""

    artifact = _unchanged_artifact()
    packets = artifact["operator_action_packets"]

    assert packets["kv260"]["next_action"] == (
        "provide a new explicit bitstream/workload directive before any SSH recheck"
    )
    assert "host storage" in packets["kv260"]["do_not_do"]
    assert packets["polarfire"]["next_action"] == (
        "add active cooling or authorize a bounded terminal Carnot workload with temperature monitoring"
    )
    assert packets["gatemate"]["next_action"] == (
        "change or reseat cable/port/power path and provide a new physical-setup message"
    )
    assert packets["extropic"]["next_action"] == (
        "provide authenticated local Extropic TSU/Z1 execution credentials or hardware"
    )
    assert packets["kona"]["next_action"] == "provide an authenticated local Kona execution route"

    assert artifact["extropic_access_state"] == {
        "state": "no_authenticated_local_execution_surface",
        "commands_run": [],
        "public_services_probed": False,
        "performance_inferred_from_marketing": False,
    }
    assert artifact["kona_access_state"] == {
        "state": "no_authenticated_local_execution_surface",
        "commands_run": [],
        "public_services_probed": False,
        "performance_inferred_from_marketing": False,
    }
    mod.validate_artifact(artifact)


def test_req_hw_5794_schema_rejects_adversarial_drift() -> None:
    """REQ-HW-5794: schema guard rejects overclaims, unsafe commands, and checksum drift."""

    base = _unchanged_artifact()

    mutations = [
        (lambda a: a.update(speedup_claimed=True), "speedup_claimed"),
        (lambda a: a.update(energy_claimed=True), "energy_claimed"),
        (lambda a: a.update(production_ready_claimed=True), "production_ready_claimed"),
        (lambda a: a.update(storage_write_count=1), "storage_write_count"),
        (lambda a: a.update(flash_write_count=1), "flash_write_count"),
        (lambda a: a.update(inference_substrate="hardware_smoke"), "inference_substrate"),
        (lambda a: a.update(honest_verdict="cached ok"), "honest_verdict"),
        (
            lambda a: a["commands_run"].append(  # type: ignore[index,union-attr]
                {
                    "board": "kv260",
                    "command": "dd if=image of=/dev/mmcblk0",
                    "stdout_sha256": "0" * 64,
                    "stderr_sha256": "0" * 64,
                    "timeout_s": 5.0,
                    "target": "unsafe",
                    "exit_code": 0,
                    "duration_s": 0.1,
                    "stop_state": "unsafe",
                }
            ),
            "unsafe command",
        ),
        (
            lambda a: a.update(
                extropic_access_state={
                    "state": "public_marketing_checked",
                    "commands_run": [],
                    "public_services_probed": True,
                    "performance_inferred_from_marketing": True,
                }
            ),
            "extropic_access_state",
        ),
        (lambda a: a.update(schema="bad"), "schema mismatch"),
        (lambda a: a.update(experiment_id="bad"), "experiment_id mismatch"),
        (lambda a: a.update(milestone="bad"), "milestone mismatch"),
        (lambda a: a.update(random_seed=1), "random_seed mismatch"),
        (lambda a: a.update(spec_refs=["REQ-HW-5794"]), "spec_refs mismatch"),
        (lambda a: a.update(field_principles={}), "field_principles mismatch"),
    ]

    for mutate, needle in mutations:
        artifact = deepcopy(base)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        assert any(needle in error for error in mod.artifact_schema_errors(artifact))

    artifact = deepcopy(base)
    artifact["status"] = "tampered"
    assert any("checksum" in error for error in mod.artifact_schema_errors(artifact))

    artifact = deepcopy(base)
    del artifact["status"]
    assert any("missing required fields" in error for error in mod.artifact_schema_errors(artifact))

    artifact = deepcopy(base)
    artifact["honest_verdict"] = "complete: speedup=true"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    assert any("performance overclaim" in error for error in mod.artifact_schema_errors(artifact))


def test_req_hw_5794_schema_rejects_nested_receipt_drift() -> None:
    """REQ-HW-5794: nested canonical, state, and probe validators fail closed."""

    base = _unchanged_artifact()

    nested_mutations = [
        (lambda a: a.update(canonical_hardware_artifacts=[]), "canonical_hardware_artifacts"),
        (
            lambda a: a["canonical_hardware_artifacts"].pop("kv260"),  # type: ignore[index,union-attr]
            "kv260 canonical artifact missing",
        ),
        (
            lambda a: a["canonical_hardware_artifacts"]["kv260"].update(path="results/wrong.json"),  # type: ignore[index,union-attr]
            "kv260 canonical artifact exact path",
        ),
        (
            lambda a: a["canonical_hardware_artifacts"]["kv260"].update(selection_method="mtime"),  # type: ignore[index,union-attr]
            "selection_method",
        ),
        (
            lambda a: a["hardware_artifact_hashes"].update(  # type: ignore[index,union-attr]
                {
                    mod.CANONICAL_BOARD_ARTIFACTS["kv260"]["path"].as_posix(): "bad",
                }
            ),
            "hardware_artifact_hashes mismatch",
        ),
        (
            lambda a: a.update(kv260_state={"host_storage_or_block_device_accessed": True}),
            "kv260_state host storage",
        ),
        (
            lambda a: a["kv260_state"].update(host_storage_access_prohibited=False),  # type: ignore[index,union-attr]
            "kv260_state must prohibit",
        ),
        (lambda a: a.update(polarfire_state=[]), "polarfire_state invalid"),
        (
            lambda a: a["polarfire_state"].update(  # type: ignore[index,union-attr]
                terminal_carnot_workload_state="claimed_terminal"
            ),
            "polarfire_state terminal workload",
        ),
        (lambda a: a.update(gatemate_state=[]), "gatemate_state invalid"),
        (
            lambda a: a["gatemate_state"].update(flash_state="flashed"),  # type: ignore[index,union-attr]
            "gatemate_state flash_state",
        ),
        (lambda a: a.update(temperature_duration_receipts=[]), "temperature_duration_receipts"),
        (
            lambda a: a["temperature_duration_receipts"]["polarfire"].update(  # type: ignore[index,union-attr]
                max_unaided_duration_s=999
            ),
            "temperature_duration_receipts polarfire duration",
        ),
        (lambda a: a.update(changed_preconditions={"kv260": False}), "changed_preconditions"),
        (lambda a: a.update(commands_run="bad"), "commands_run or commands_skipped"),
        (
            lambda a: a["commands_run"].append("bad"),  # type: ignore[index,union-attr]
            "commands_run entry invalid",
        ),
        (
            lambda a: a["commands_run"].append(  # type: ignore[index,union-attr]
                {
                    "board": "kv260",
                    "command": "ssh kria true",
                    "stdout_sha256": "short",
                    "stderr_sha256": "0" * 64,
                    "timeout_s": 5.0,
                    "target": "ssh",
                    "exit_code": 0,
                    "duration_s": 0.1,
                    "stop_state": "done",
                }
            ),
            "commands_run stdout_sha256",
        ),
        (lambda a: a.update(probe_decisions={"kv260": {}}), "probe_decisions"),
    ]

    for mutate, needle in nested_mutations:
        artifact = deepcopy(base)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        assert any(needle in error for error in mod.artifact_schema_errors(artifact))

    artifact = deepcopy(base)
    artifact["commands_run"].append(  # type: ignore[index,union-attr]
        {
            "board": "kv260",
            "command": "ssh kria true",
            "stdout_sha256": "0" * 64,
            "stderr_sha256": "0" * 64,
            "timeout_s": 5.0,
            "target": "ssh",
            "exit_code": 0,
            "duration_s": 0.1,
            "stop_state": "done",
        }
    )
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    errors = mod.artifact_schema_errors(artifact)
    assert any("commands_run present for unchanged_preconditions" in error for error in errors)
    assert any("commands_run must be empty" in error for error in errors)

    with pytest.raises(ValueError, match="speedup_claimed"):
        invalid = deepcopy(base)
        invalid["speedup_claimed"] = True
        invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
        mod.validate_artifact(invalid)


def test_req_hw_5794_run_experiment_writes_valid_json(tmp_path: Path) -> None:
    """REQ-HW-5794: run_experiment writes the exact deliverable shape."""

    artifact = _unchanged_artifact()
    out_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert saved == artifact
    assert saved["test_exit_codes"] == _test_exit_codes()
    mod.validate_artifact(saved)

    run_path = mod.run_experiment(
        repo_root=REPO,
        run_date="20260722",
        test_exit_codes=_test_exit_codes(),
    )
    assert run_path == REPO / mod.RESULT_RELATIVE_PATH
    saved_live = json.loads(run_path.read_text(encoding="utf-8"))
    assert saved_live["reproducibility_checksum"] == mod.payload_checksum(saved_live)


def test_req_hw_5794_helper_branches_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """SCENARIO-HW-5794: helper failures and CLI argument plumbing stay deterministic."""

    assert mod.unwrap_field({"value": "complete:"}) == "complete:"
    assert mod.unwrap_field("bare") == "bare"
    assert mod.hash_source_file(REPO / "missing-file-does-not-exist")["present"] is False
    assert mod.extract_exp5794_prompt("no task here") == ""
    assert mod.extract_exp5794_prompt("id: exp5794-hardware-terminal-action-receipt\nprompt") == (
        "id: exp5794-hardware-terminal-action-receipt\nprompt"
    )
    assert mod.source_hashes(tmp_path)["roadmap_exp5794_operator_message"]["present"] is False

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object expected"):
        mod.read_json(list_json)

    fallback_pf = mod.build_polarfire_state({"polarfire_status": {"value": "reachable"}})
    assert fallback_pf["authentication_state"] == "cached_ssh_reachable"

    baseline = _unchanged_artifact()
    decisions, skipped = mod.build_probe_decisions(
        current_hashes={"kv260": "1", "polarfire": "2", "gatemate": "3"},
        previous_hashes=baseline["precondition_hashes_current"],
        authorization=mod.DEFAULT_OPERATOR_AUTHORIZATION,
    )
    assert decisions["kv260"]["decision"] == "skip_changed_without_authorization"
    assert {row["reason"] for row in skipped} == {"changed_precondition_without_bounded_authorization"}

    class Completed:
        returncode = 7
        stdout = "out\n"
        stderr = "err\n"

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Completed())
    probe = mod.run_command(("cmd",), 1.0)
    assert probe.exit_code == 7
    assert probe.stdout == "out\n"

    def raise_not_found(*args, **kwargs):
        raise FileNotFoundError("missing cmd")

    monkeypatch.setattr(mod.subprocess, "run", raise_not_found)
    missing_probe = mod.run_command(("missing",), 1.0)
    assert missing_probe.exit_code == 127
    assert "missing cmd" in missing_probe.stderr

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(("cmd",), 1.0, output="partial\n")

    monkeypatch.setattr(mod.subprocess, "run", raise_timeout)
    timeout_probe = mod.run_command(("cmd",), 1.0)
    assert timeout_probe.exit_code == 124
    assert "timeout" in timeout_probe.stderr

    with pytest.raises(argparse.ArgumentTypeError):
        mod.parse_test_results_json("[]")

    calls: list[dict[str, object]] = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return Path("results/fake5794.json")

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert mod.main(["--date", "20260722", "--test-results-json", json.dumps(_test_exit_codes())]) == 0
    assert calls[0]["run_date"] == "20260722"
    assert calls[0]["test_exit_codes"] == _test_exit_codes()
    assert "results/fake5794.json" in capsys.readouterr().out
