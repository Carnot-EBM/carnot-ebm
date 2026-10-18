"""REQ-REPORT-8638, REQ-VERIFY-8638: missing costs cannot grant hardware readiness."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_delta_8638 as e
from carnot.reporting import hardware_delta_runner_8638 as runner
from carnot.reporting.current_work_receipt import atomic_json


def cli(tmp_path, *args):
    """Real children prove that the script works outside the checkout."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_blocked_costs_and_unchanged_board(tmp_path):
    """SCENARIO-REPORT-8638-BLOCKED: retain board facts without rerunning history."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, tmp_path, "20261018")
    value = e.build(work, [dict(name="owned", passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["hardware_delta_ready_score"] == value["physical_hardware_ready_score"] == 0
    assert value["NPU_qualified"] == value["TSU_qualified"] == 0
    assert value["new_operation_cost_rows"] == []
    assert all(r["zero_cost_ceiling"] is None for r in value["conditional_ceiling_rows"])
    assert value["device_commands_executed"] == 0
    assert value["MODEL_SPECS"] == []
    assert value["numerical_inference_this_invocation"]["small_head_evaluations"] == 0
    assert value["target_model"]["parameter_count"] == 109
    assert value["kv260_supported_operations"][0]["k_max"] == 5
    assert value["board_reopen_requirements"][2]["expected_idcode"] == "0x20000001"
    assert value["gate_check_summary"]
    assert all(
        set(["upstream", "path", "sha256", "field", "op", "expected", "observed"]) <= set(r)
        for r in value["gate_check_summary"]
    )
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    value["hardware_delta_ready_score"] = 1
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)


def test_missing_and_rehashed_originals(tmp_path):
    """SCENARIO-VERIFY-8638-REPLAY: rehashing edited operands grants no authority."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path, "20261018")
    assert not work["evidence"]["board"]["qualified"]
    assert (
        e.build(work, [dict(passed=False)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))[
            "verdict_class"
        ]
        == "disqualified"
    )
    board = tmp_path / e.BOARD
    board.parent.mkdir(parents=True, exist_ok=True)
    board.write_text("{}")
    assert not e.authenticate(tmp_path)["board"]["qualified"]
    assert not e.replay(tmp_path / "missing")
    work = e.measure(e.ROOT, tmp_path / "real", tmp_path, "20261018")
    value = e.build(work, [dict(passed=True)], tmp_path / "real", tmp_path / (e.NAME + ".json"))
    changed = deepcopy(work)
    changed["evidence"]["board"]["value"]["physical_hardware_ready_score"] = 1
    atomic_json(tmp_path / "real/measurement.json", changed)
    value["work_reference"] = e.reference(tmp_path / "real/measurement.json")
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(tmp_path / "changed.json", value)
    assert not e.replay(tmp_path / "changed.json")


def test_private_cli_and_rejection_paths(tmp_path):
    """SCENARIO-VERIFY-8638-REPLAY: external blockage and owned failure differ."""
    for failed in [False, True]:
        output = tmp_path / str(failed) / (e.NAME + ".json")
        result = cli(
            tmp_path,
            "--private-control",
            "--root",
            tmp_path / "absent",
            "--output",
            output,
            *(["--owned-failure"] if failed else []),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    for date in ["bad", "20260230"]:
        assert cli(tmp_path, "--date", date).returncode == 2
    assert cli(tmp_path, "--private-control").returncode == 2


def test_validation_retains_shards(tmp_path, monkeypatch):
    """REQ-VERIFY-8638: combining coverage preserves each original child shard."""

    def validate(raw):
        runner.previous.child("combine", ["coverage", "combine", str(raw)], raw)
        return [dict(name="owned", passed=True)]

    commands = []
    monkeypatch.setattr(runner.previous, "validate", validate)
    monkeypatch.setattr(runner, "child", lambda name, argv, raw, **kw: commands.append(argv))
    assert runner.validate(tmp_path)[0]["passed"]
    assert commands[0][-2:] == ["--keep", str(tmp_path)]


def test_bad_reference(tmp_path):
    """REQ-VERIFY-8638: missing hashes are rejected before replay reduction."""
    with pytest.raises(ValueError):
        e.checked(dict(path=str(tmp_path / "missing"), sha256="sha256:bad"))


def test_unregistered_cost_and_failed_board_seal(tmp_path, monkeypatch):
    """REQ-REPORT-8638: future summaries cannot self-qualify without original primitives."""
    producer = tmp_path / "results/experiment_8630_v741_causal_brier_stream.json"
    atomic_json(producer, dict(learning_cost_ready_score=1))
    assert not e.authenticate(tmp_path)["branches"][0]["qualified"]
    board = json.loads((e.ROOT / e.BOARD).read_bytes())
    board["required_checks_passed"] = False
    atomic_json(tmp_path / e.BOARD, board)
    monkeypatch.setattr(e, "BOARD_PIN", e.reference(tmp_path / e.BOARD)["sha256"])
    assert not e.authenticate(tmp_path)["board"]["qualified"]


def test_warning_is_still_rejected(tmp_path, monkeypatch):
    """REQ-VERIFY-8638: acknowledging one INFO finding does not suppress other concerns."""
    monkeypatch.setattr(
        runner,
        "original_scan",
        lambda *args: (
            [dict(passed=False, normal_exit=True, actual_exit=0)],
            [dict(kind="bad", severity="warn")],
        ),
    )
    assert not runner.scan(tmp_path, tmp_path)[0][0]["passed"]


def test_real_combine_keeps_original_shards(tmp_path, monkeypatch):
    """REQ-VERIFY-8638: the actual Coverage CLI must retain its original child database."""
    probe = tmp_path / "covered.py"
    probe.write_text("value = 1\n")
    data = str(tmp_path / ".coverage")
    command = [sys.executable, "-m", "coverage"]
    generated = subprocess.run(
        [
            *command,
            "run",
            "--rcfile=/dev/null",
            "--parallel-mode",
            f"--data-file={data}",
            str(probe),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert generated.returncode == 0, generated.stderr
    originals = [e.reference(p) for p in tmp_path.glob(".coverage.*")]
    assert originals

    def validate(raw):
        return [
            runner.previous.child(
                "combine",
                [*command, "combine", "--rcfile=/dev/null", f"--data-file={data}", str(raw)],
                raw,
            )
        ]

    monkeypatch.setattr(runner.previous, "validate", validate)
    assert runner.validate(tmp_path)[0]["passed"]
    assert all(e.checked(ref) for ref in originals)
