"""Tests for Exp 5255 hardware continuity plus p-kit/Extropic/Kona boundary notes.

Spec refs: REQ-HW-5255, SCENARIO-HW-5255.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5255_hardware_continuity_pkit_boundary_v480 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


class RecordingRunner:
    """SCENARIO-HW-5255 command runner with deterministic board receipts."""

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
    """Deterministic clock for stable wall-clock and checksum assertions."""

    def __init__(self) -> None:
        self.value = 5255.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _probe(
    command: tuple[str, ...],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _smoke_stdout(board: str, *, workload_hash: str | None = None, ok: bool = True) -> str:
    output = {
        "board": board,
        "workload_sha256": workload_hash or mod.HASH_SMOKE_WORKLOAD_HASH,
        "binary_or_bitstream_sha256": mod.HASH_SMOKE_EXECUTABLE_HASH,
        "correctness": {"energy_matches_expected": ok},
        "energy": mod.HASH_SMOKE_EXPECTED_ENERGY if ok else 999,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
    }
    output["output_sha256"] = mod.output_hash(output)
    return json.dumps(output, sort_keys=True) + "\n"


def _runner(
    *,
    kv260_exit: int = 0,
    kv260_stderr: str = "",
    kv260_smoke_stdout: str | None = None,
    polarfire_exit: int = 0,
    polarfire_stderr: str = "",
    polarfire_smoke_stdout: str | None = None,
) -> RecordingRunner:
    probes: dict[tuple[str, ...], list[mod.CommandProbe]] = {
        mod.KV260_SSH_COMMAND: [
            _probe(
                mod.KV260_SSH_COMMAND,
                exit_code=kv260_exit,
                stderr=kv260_stderr,
                duration_s=0.2,
            )
        ],
        mod.POLARFIRE_SSH_COMMAND: [
            _probe(
                mod.POLARFIRE_SSH_COMMAND,
                exit_code=polarfire_exit,
                stderr=polarfire_stderr,
                duration_s=0.3,
            )
        ],
    }
    if kv260_exit == 0:
        probes[mod.KV260_HASH_SMOKE_COMMAND] = [
            _probe(
                mod.KV260_HASH_SMOKE_COMMAND,
                stdout=kv260_smoke_stdout or _smoke_stdout("kv260"),
                duration_s=0.4,
            )
        ]
    if polarfire_exit == 0:
        probes[mod.POLARFIRE_HASH_SMOKE_COMMAND] = [
            _probe(
                mod.POLARFIRE_HASH_SMOKE_COMMAND,
                stdout=polarfire_smoke_stdout or _smoke_stdout("polarfire"),
                duration_s=0.5,
            )
        ]
    return RecordingRunner(probes)


def _success_artifact() -> dict[str, object]:
    return mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        notes_written=True,
    )


def test_req_hw_5255_spec_declares_required_wrapped_fields() -> None:
    """REQ-HW-5255: OpenSpec anchors the v480 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5255") : spec.index("### SCENARIO-HW-4910")]

    for marker in (
        "REQ-HW-5255",
        "SCENARIO-HW-5255",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "kv260_ssh_only_confirmed",
        "workload_hashes",
        "pkit_boundary_note_path",
        "extropic_kona_boundary_note_path",
        "speedup_claimed=false",
        "blocked precondition receipt",
    ):
        assert marker in section
    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section


def test_scenario_hw_5255_success_path_records_receipts_without_gatemate_rerun() -> None:
    """SCENARIO-HW-5255: reachable boards run only SSH/hash receipts."""

    runner = _runner()
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        notes_written=True,
    )

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_HASH_SMOKE_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
        mod.POLARFIRE_HASH_SMOKE_COMMAND,
    ]
    assert _value(artifact, "kv260_status") == "reachable"
    assert _value(artifact, "kv260_ssh_only_confirmed") is True
    assert _value(artifact, "polarfire_status") == "reachable"
    assert _value(artifact, "gatemate_status") == "blocked_physical_jtag"
    assert _value(artifact, "physical_setup_changed") is False
    assert _value(artifact, "speedup_claimed") is False
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "pkit_boundary_note_path") == str(mod.PKIT_NOTE_RELATIVE_PATH)
    assert _value(artifact, "extropic_kona_boundary_note_path") == str(
        mod.EXTROPIC_KONA_NOTE_RELATIVE_PATH
    )
    hashes = _value(artifact, "workload_hashes")
    assert isinstance(hashes, dict)
    assert hashes["commit"] == "abc123"
    assert hashes["workload_sha256"] == mod.HASH_SMOKE_WORKLOAD_HASH
    assert hashes["binary_or_bitstream_sha256"] == mod.HASH_SMOKE_EXECUTABLE_HASH
    assert artifact["board_receipts"]["kv260"]["correctness_ok"] is True
    assert artifact["board_receipts"]["polarfire"]["output_hash"]
    assert artifact["command_probes"]["gatemate_physical_jtag"] is None
    assert "no_speedup_claim" in _value(artifact, "honest_verdict")
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "/dev/mmcblk" not in json.dumps(artifact).lower()
    mod.validate_artifact(artifact)


def test_blocked_precondition_records_exact_connection_receipts() -> None:
    """REQ-HW-5255: unreachable boards produce blocked precondition receipts."""

    runner = _runner(
        kv260_exit=255,
        kv260_stderr="ssh: connect to host kria port 22: timeout\n",
        polarfire_exit=255,
        polarfire_stderr="ssh: connect to host polarfire port 22: No route to host\n",
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        notes_written=True,
    )

    assert runner.commands == [mod.KV260_SSH_COMMAND, mod.POLARFIRE_SSH_COMMAND]
    assert _value(artifact, "kv260_status") == "blocked"
    assert _value(artifact, "polarfire_status") == "blocked"
    assert _value(artifact, "kv260_ssh_only_confirmed") is True
    assert _value(artifact, "honest_verdict").startswith("blocked_precondition")
    assert artifact["kv260_blocked_precondition"]["command"] == mod.command_to_string(
        mod.KV260_SSH_COMMAND
    )
    assert artifact["polarfire_blocked_precondition"]["error"] == (
        "ssh: connect to host polarfire port 22: No route to host\n"
    )
    assert artifact["board_receipts"]["kv260"]["wall_clock_s"] == 0.2
    assert artifact["board_receipts"]["polarfire"]["binary_or_bitstream_sha256"] is None
    mod.validate_artifact(artifact)


def test_hash_smoke_mismatch_blocks_board_without_speedup_claim() -> None:
    """REQ-HW-5255: SSH alone is not inflated into valid hash continuity."""

    artifact = mod.build_artifact(
        command_runner=_runner(kv260_smoke_stdout=_smoke_stdout("kv260", workload_hash="bad")),
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        notes_written=True,
    )

    assert _value(artifact, "kv260_status") == "blocked"
    assert artifact["board_receipts"]["kv260"]["hash_verified"] is False
    assert artifact["board_receipts"]["kv260"]["correctness_ok"] is True
    assert _value(artifact, "speedup_claimed") is False
    assert _value(artifact, "honest_verdict").startswith("blocked_precondition")
    mod.validate_artifact(artifact)


def test_physical_setup_changed_records_not_checked_gate_without_probe() -> None:
    """SCENARIO-HW-5255: setup changes do not authorize an unbounded GateMate rerun."""

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        notes_written=True,
        physical_setup_changed=True,
    )

    assert _value(artifact, "physical_setup_changed") is True
    assert _value(artifact, "gatemate_status") == "not_checked"
    assert artifact["gatemate_carry_forward"]["physical_setup_changed"] is True
    assert artifact["command_probes"]["gatemate_physical_jtag"] is None
    mod.validate_artifact(artifact)


def test_boundary_note_writers_state_local_claim_limits(tmp_path: Path) -> None:
    """SCENARIO-HW-5255: notes distinguish public references from local hardware."""

    artifact_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    pkit_note = (tmp_path / str(_value(artifact, "pkit_boundary_note_path"))).read_text(
        encoding="utf-8"
    )
    extropic_kona_note = (
        tmp_path / str(_value(artifact, "extropic_kona_boundary_note_path"))
    ).read_text(encoding="utf-8")

    assert artifact_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert "IBM/p-kit" in pkit_note
    assert "software reference" in pkit_note
    assert "not local p-bit hardware" in pkit_note
    assert "Extropic" in extropic_kona_note
    assert "XTR-0" in extropic_kona_note
    assert "Kona" in extropic_kona_note
    assert "Aleph" in extropic_kona_note
    assert "No local Carnot claim" in extropic_kona_note
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_helper_edges_and_validation_fail_closed() -> None:
    """REQ-HW-5255: helper fallbacks and schema validation reject bad artifacts."""

    artifact = _success_artifact()
    assert mod.parse_last_json("noise\n{bad json}\n") == {}
    assert mod.parse_last_json("noise only\n") == {}
    assert mod.board_receipt(
        board="kv260",
        ssh_probe=_probe(mod.KV260_SSH_COMMAND, exit_code=1, stderr="down\n"),
        smoke_probe=None,
        commit="abc123",
    )["status"] == "blocked"
    assert mod.board_receipt(
        board="kv260",
        ssh_probe=_probe(mod.KV260_SSH_COMMAND),
        smoke_probe=None,
        commit="abc123",
    )["status"] == "blocked"
    failed = _probe(mod.KV260_HASH_SMOKE_COMMAND, exit_code=1, stderr="python missing\n")
    assert mod.board_receipt(
        board="kv260",
        ssh_probe=_probe(mod.KV260_SSH_COMMAND),
        smoke_probe=failed,
        commit="abc123",
    )["blocked_reason"] == "blocked_kv260_hash_smoke"
    assert mod.status_from_receipt({"status": "reachable"}) == "reachable"
    assert mod.status_from_receipt({"status": "blocked"}) == "blocked"

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "hardware_speedup")
    broken["reproducibility_checksum"] = mod.payload_checksum(broken)
    with pytest.raises(AssertionError, match="inference_substrate"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    del broken["workload_hashes"]
    with pytest.raises(AssertionError, match="missing required field"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["speedup_claimed"] = mod.wrap_field("speedup_claimed", True)
    broken["reproducibility_checksum"] = mod.payload_checksum(broken)
    with pytest.raises(AssertionError, match="speedup_claimed"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_req_hw_5255() -> None:
    """SCENARIO-HW-5255: committed deliverable is stable and claims no speedup."""

    if not RESULT_PATH.exists():
        pytest.skip("deliverable generated after implementation run")
    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert _value(artifact, "speedup_claimed") is False
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "honest_verdict").startswith(("complete:", "blocked_"))
