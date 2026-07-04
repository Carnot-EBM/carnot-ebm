"""Tests for Exp 5231 hardware continuity + p-bit boundary plan.

Spec refs: REQ-HW-5231, SCENARIO-HW-5231.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_5231_hardware_continuity_pbit_boundary_v478 as mod


class RecordingRunner:
    """SCENARIO-HW-5231 command runner with queued board transcripts."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        assert timeout_s > 0.0
        command = tuple(command)
        self.commands.append(command)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


class StepClock:
    """Deterministic increasing clock for stable duration and checksum tests."""

    def __init__(self) -> None:
        self.value = 5231.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


def _probe(
    command: tuple[str, ...],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _polarfire_smoke_stdout(*, workload_hash: str | None = None, ok: bool = True) -> str:
    return (
        json.dumps(
            {
                "workload_sha256": workload_hash or mod.POLARFIRE_SMOKE_WORKLOAD_HASH,
                "executable_sha256": mod.INLINE_EXECUTABLE_HASH,
                "energy": mod.POLARFIRE_EXPECTED_ENERGY if ok else 999,
                "correctness": {"energy_matches_expected": ok},
                "sample_quality": {"sample_count": 8, "finite_energy": True},
                "inference_substrate": mod.INFERENCE_SUBSTRATE,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _base_runner(
    *,
    kv260_exit: int = 0,
    polarfire_exit: int = 0,
    smoke_stdout: str | None = None,
    gatemate_probe: mod.CommandProbe | None = None,
) -> RecordingRunner:
    probes: dict[tuple[str, ...], list[mod.CommandProbe]] = {
        mod.KV260_SSH_COMMAND: [
            _probe(mod.KV260_SSH_COMMAND, exit_code=kv260_exit, duration_s=0.2)
        ],
        mod.POLARFIRE_SSH_COMMAND: [
            _probe(mod.POLARFIRE_SSH_COMMAND, exit_code=polarfire_exit, duration_s=0.3)
        ],
    }
    if polarfire_exit == 0:
        probes[mod.POLARFIRE_SMOKE_COMMAND] = [
            _probe(
                mod.POLARFIRE_SMOKE_COMMAND,
                stdout=smoke_stdout or _polarfire_smoke_stdout(),
                duration_s=0.4,
            )
        ]
    if gatemate_probe is not None:
        probes[mod.GATEMATE_DEBUG_DETECT_COMMAND] = [gatemate_probe]
    return RecordingRunner(probes)


def _success_artifact() -> dict:
    return mod.build_artifact(
        command_runner=_base_runner(),
        clock=StepClock(),
        run_date="20260704",
        hardware_docs_updated=True,
    )


def test_req_hw_5231_spec_declares_artifact_and_no_speedup_contract() -> None:
    """REQ-HW-5231: OpenSpec anchors the v478 artifact contract."""

    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-HW-5231",
        "SCENARIO-HW-5231",
        "experiment_5231_hardware_continuity_pbit_boundary_v478.json",
        "kv260_check_method=\"ssh_only\"",
        "polarfire_reachable",
        "gatemate_status=\"blocked_physical_jtag\"",
        "pbit_boundary_plan_path",
        "No .478 hardware task may claim speedup without a real end-to-end workload.",
    ):
        assert marker in spec


def test_scenario_hw_5231_preserves_gatemate_block_and_builds_plan() -> None:
    """SCENARIO-HW-5231: no setup change skips GateMate recheck and records plan."""

    runner = _base_runner()
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260704",
        hardware_docs_updated=True,
    )

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
        mod.POLARFIRE_SMOKE_COMMAND,
    ]
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == ["REQ-HW-5231", "SCENARIO-HW-5231"]
    assert artifact["kv260_reachable"] is True
    assert artifact["kv260_check_method"] == "ssh_only"
    assert artifact["polarfire_reachable"] is True
    assert artifact["polarfire_smoke"]["hash_verified"] is True
    assert artifact["gatemate_status"] == "blocked_physical_jtag"
    assert artifact["gatemate_idcode_raw"] == "0xffffffff"
    assert artifact["gatemate_rechecked"] is False
    assert artifact["pbit_boundary_plan_path"] == str(mod.PBIT_PLAN_REL_PATH)
    assert artifact["speedup_claimed"] is False
    assert artifact["field_principles"]["speedup_claimed"] == mod.NO_SPEEDUP_PRINCIPLE
    assert artifact["hardware_docs_updated"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete_")
    assert "kv260:reachable" in artifact["honest_verdict"]
    assert "polarfire:reachable" in artifact["honest_verdict"]
    assert "gatemate:blocked_physical_jtag" in artifact["honest_verdict"]
    assert "no_speedup" in artifact["honest_verdict"]
    assert "mmcblk" not in json.dumps(artifact).lower()
    mod.validate_artifact(artifact)


def test_run_experiment_writes_json_plan_and_wishlist_note(tmp_path: Path) -> None:
    """REQ-HW-5231: run_experiment writes the artifact and docs it claims."""

    wishlist = tmp_path / "research-hardware-wishlist.md"
    wishlist.write_text("# Hardware wishlist\n\n", encoding="utf-8")

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_base_runner(),
        clock=StepClock(),
        run_date="20260704",
        hardware_docs_updated=True,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    plan_path = tmp_path / payload["pbit_boundary_plan_path"]
    plan_text = plan_path.read_text(encoding="utf-8")
    wishlist_text = wishlist.read_text(encoding="utf-8")

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert "distributed_sparse_pbit_boundary_exchange_n1024x4" in plan_text
    assert "partitions: 4 x 256 p-bits" in plan_text
    assert "hash/correctness checks" in plan_text
    assert "No speedup claim is allowed" in plan_text
    assert "Exp 5231" in wishlist_text
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    mod.validate_artifact(payload)


def test_polarfire_unreachable_skips_hash_smoke() -> None:
    """REQ-HW-5231: unreachable PolarFire records a blocker without smoke claims."""

    runner = _base_runner(polarfire_exit=255)
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260704",
        hardware_docs_updated=True,
    )

    assert runner.commands == [mod.KV260_SSH_COMMAND, mod.POLARFIRE_SSH_COMMAND]
    assert artifact["polarfire_reachable"] is False
    assert artifact["polarfire_smoke"]["status"] == "not_run_unreachable"
    assert "polarfire:blocked_polarfire_ssh" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_kv260_unreachable_still_uses_ssh_only_method() -> None:
    """REQ-HW-5231: KV260 failures are SSH blockers, not host-storage blockers."""

    artifact = mod.build_artifact(
        command_runner=_base_runner(kv260_exit=255),
        clock=StepClock(),
        run_date="20260704",
        hardware_docs_updated=True,
    )

    assert artifact["kv260_reachable"] is False
    assert artifact["kv260_check_method"] == "ssh_only"
    assert artifact["preconditions_checked"][0]["resource"] == "kv260_ssh"
    assert artifact["preconditions_checked"][0]["discipline"] == "ssh_only"
    assert "kv260:blocked_kv260_ssh" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_gatemate_recheck_only_after_operator_setup_change() -> None:
    """SCENARIO-HW-5231: setup-changed GateMate probe can resolve reachability."""

    debug_stdout = "Raw IDCODE:\n- 0 -> 0x20000001\nmanufacturer colognechip\n"
    runner = _base_runner(
        gatemate_probe=_probe(mod.GATEMATE_DEBUG_DETECT_COMMAND, stdout=debug_stdout)
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260704",
        hardware_docs_updated=True,
        gatemate_setup_changed=True,
    )

    assert mod.GATEMATE_DEBUG_DETECT_COMMAND in runner.commands
    assert artifact["gatemate_status"] == "reachable"
    assert artifact["gatemate_idcode_raw"] == "0x20000001"
    assert artifact["gatemate_rechecked"] is True
    assert "operator setup changed" in artifact["gatemate_check_note"]
    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    "stdout, expected_status, expected_raw",
    [
        ("Raw IDCODE:\n- 0 -> 0xffffffff\nfound 0 devices\n", "blocked_physical_jtag", "0xffffffff"),
        ("found 0 devices\n", "not_checked", None),
    ],
)
def test_gatemate_recheck_blocked_and_not_checked_branches(
    stdout: str, expected_status: str, expected_raw: str | None
) -> None:
    """REQ-HW-5231: GateMate recheck maps raw all-ones and missing raw honestly."""

    runner = _base_runner(
        gatemate_probe=_probe(mod.GATEMATE_DEBUG_DETECT_COMMAND, stdout=stdout)
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260704",
        hardware_docs_updated=True,
        gatemate_setup_changed=True,
    )

    assert artifact["gatemate_status"] == expected_status
    assert artifact["gatemate_idcode_raw"] == expected_raw
    mod.validate_artifact(artifact)


def test_polarfire_hash_or_correctness_mismatch_is_not_verified() -> None:
    """REQ-HW-5231: PolarFire smoke must match both hash and correctness."""

    artifact = mod.build_artifact(
        command_runner=_base_runner(smoke_stdout=_polarfire_smoke_stdout(workload_hash="bad")),
        clock=StepClock(),
        run_date="20260704",
        hardware_docs_updated=True,
    )

    assert artifact["polarfire_reachable"] is True
    assert artifact["polarfire_smoke"]["hash_verified"] is False
    assert artifact["polarfire_smoke"]["correctness_ok"] is True
    assert "polarfire:reachable" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_base_runner(smoke_stdout=_polarfire_smoke_stdout(ok=False)),
        clock=StepClock(),
        run_date="20260704",
        hardware_docs_updated=True,
    )
    assert artifact["polarfire_smoke"]["hash_verified"] is True
    assert artifact["polarfire_smoke"]["correctness_ok"] is False
    mod.validate_artifact(artifact)


def test_helper_fallback_branches_and_live_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-5231: helper fallbacks stay deterministic and command wrapper is bounded."""

    assert mod.parse_last_json("noise\n{bad json}\n") == {}
    assert mod.parse_last_json("noise only\n") == {}
    assert mod.polarfire_smoke_status(reachable=True, smoke_probe=None)["status"] == (
        "not_run_missing_probe"
    )
    assert mod.gatemate_from_probe(setup_changed=True, probe=None)["status"] == "not_checked"
    failed_probe = _probe(mod.GATEMATE_DEBUG_DETECT_COMMAND, exit_code=1, stderr="usb error\n")
    assert mod.gatemate_from_probe(setup_changed=True, probe=failed_probe)["raw_idcode"] is None
    assert mod.raw_idcode_from_text("idcode 0x20000001\n") == "0x20000001"
    assert mod.is_floating_tdo("0x12345678") is False
    assert mod.round_duration(0.0) == pytest.approx(0.000001)

    class Completed:
        returncode = 7
        stdout = "out\n"
        stderr = "err\n"

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Completed())
    probe = mod.run_command(("cmd",), 1.0)
    assert probe.exit_code == 7
    assert probe.combined_output == "out\nerr\n"

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(("cmd",), 1.0, output="partial\n")

    monkeypatch.setattr(mod.subprocess, "run", raise_timeout)
    timeout_probe = mod.run_command(("cmd",), 1.0)
    assert timeout_probe.exit_code == 124
    assert "command timed out" in timeout_probe.stderr


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda a: a.update(kv260_check_method="host_storage"), "kv260_check_method"),
        (lambda a: a.update(speedup_claimed=True), "speedup_claimed"),
        (lambda a: a.update(inference_substrate="hardware_smoke"), "inference_substrate"),
        (lambda a: a.update(hardware_docs_updated=False), "hardware_docs_updated"),
        (lambda a: a.update(honest_verdict="blocked_anything"), "honest_verdict"),
        (lambda a: a.update(gatemate_status="bad"), "gatemate_status"),
        (lambda a: a.update(gatemate_idcode_raw=123), "gatemate_idcode_raw"),
        (lambda a: a.update(pbit_boundary_plan_path="missing.md"), "pbit_boundary_plan_path"),
        (lambda a: a.update(spec_refs=["REQ-HW-5231"]), "spec_refs"),
        (lambda a: a.update(preconditions_checked=[]), "preconditions_checked"),
        (lambda a: a.update(field_principles={}), "field_principles"),
        (lambda a: a.update(extra="/dev/disk/by-id/fake"), "host storage marker"),
    ],
)
def test_schema_rejects_required_field_drift(mutate, needle: str) -> None:
    """REQ-HW-5231: schema guard rejects drift in the required fields."""

    artifact = _success_artifact()
    mutate(artifact)

    assert any(needle in error for error in mod.artifact_schema_errors(artifact))


def test_schema_rejects_malformed_nested_sections() -> None:
    """REQ-HW-5231: nested validators reject malformed sections explicitly."""

    artifact = _success_artifact()
    del artifact["honest_verdict"]
    assert any("missing required fields" in error for error in mod.artifact_schema_errors(artifact))

    artifact = _success_artifact()
    artifact["preconditions_checked"] = ["bad", "bad"]
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    assert any("preconditions_checked entries invalid" in e for e in mod.artifact_schema_errors(artifact))

    artifact = _success_artifact()
    artifact["polarfire_smoke"] = "bad"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    assert any("polarfire_smoke invalid" in e for e in mod.artifact_schema_errors(artifact))

    artifact = _success_artifact()
    artifact["pbit_boundary_plan"] = "bad"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    assert any("pbit_boundary_plan invalid" in e for e in mod.artifact_schema_errors(artifact))

    artifact = _success_artifact()
    artifact["polarfire_smoke"] = {}
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    errors = mod.artifact_schema_errors(artifact)
    assert any("polarfire smoke missing hash_verified" in e for e in errors)
    assert any("polarfire smoke missing correctness_ok" in e for e in errors)


def test_update_wishlist_create_and_replace(tmp_path: Path) -> None:
    """REQ-HW-5231: hardware wishlist note is idempotent."""

    assert mod.update_hardware_wishlist(tmp_path) is True
    wishlist = tmp_path / "research-hardware-wishlist.md"
    first = wishlist.read_text(encoding="utf-8")
    assert "Exp 5231" in first

    assert mod.update_hardware_wishlist(tmp_path) is True
    second = wishlist.read_text(encoding="utf-8")
    assert second.count("Exp 5231 hardware continuity") == 1


def test_checksum_and_write_validation_reject_drift(tmp_path: Path) -> None:
    """REQ-HW-5231: checksum validation catches silent artifact edits."""

    artifact = _success_artifact()
    artifact["duration_s"] += 1.0

    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(artifact)

    artifact = _success_artifact()
    artifact["speedup_claimed"] = True
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)

    with pytest.raises(ValueError, match="speedup_claimed"):
        mod.write_artifact(tmp_path, artifact)


def test_main_uses_run_experiment(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-HW-5231: CLI passes date and GateMate setup-change flag through."""

    calls: list[dict] = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return Path("results/fake.json")

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)

    assert mod.main(["--date", "20260705", "--gatemate-setup-changed"]) == 0
    assert calls[0]["run_date"] == "20260705"
    assert calls[0]["gatemate_setup_changed"] is True
    assert "results/fake.json" in capsys.readouterr().out
