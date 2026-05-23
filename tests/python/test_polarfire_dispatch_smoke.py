from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import carnot.hardware.polarfire_dispatch_smoke as mod


REQUIRED_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "polarfire_ssh_uptime_at_run",
    "polarfire_kernel",
    "polarfire_arch",
    "sat_instance_sha256",
    "scorer_output_sha256",
    "scorer_output_hash_verified",
    "per_clause_wall_clock_us",
    "total_wall_clock_s",
    "duration_s",
}


def _success_transcript() -> dict[str, object]:
    instance = mod.generate_sat_instance()
    scorer_output = mod.score_sat_instance(instance)
    return {
        "schema": "carnot.polarfire_sat_scorer_transcript.v1",
        "sat_instance_sha256": mod.sha256_json(instance),
        "scorer_output": scorer_output,
        "scorer_output_sha256": mod.sha256_json(scorer_output),
        "per_clause_wall_clock_us": [1.25 + (idx / 100.0) for idx in range(50)],
        "total_wall_clock_s": 10.25,
        "evaluation_cycles_per_clause": 3,
    }


class FakeRunner:
    def __init__(
        self,
        transcript: dict[str, object] | None = None,
        *,
        arch: str = "riscv64",
        python_version: str = "Python 3.12.12",
        fail_on: str | None = None,
    ) -> None:
        self.transcript = transcript or _success_transcript()
        self.arch = arch
        self.python_version = python_version
        self.fail_on = fail_on
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self, args: Sequence[str], timeout: float | None = None
    ) -> mod.CommandResult:
        del timeout
        command = tuple(args)
        self.commands.append(command)
        joined = " ".join(command)
        if self.fail_on and self.fail_on in joined:
            return mod.CommandResult(command, 1, "", "forced failure")
        if command[:2] == ("scp", "-q") and "transcript.json" in command[2]:
            Path(command[3]).write_text(json.dumps(self.transcript))
            return mod.CommandResult(command, 0, "", "")
        if command[:2] == ("scp", "-q"):
            return mod.CommandResult(command, 0, "", "")
        if command == (
            "ssh",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "BatchMode=yes",
            "polarfire",
            "true",
        ):
            return mod.CommandResult(command, 0, "", "")
        if command == ("ssh", "polarfire", "uname -m"):
            return mod.CommandResult(command, 0, self.arch + "\n", "")
        if command == ("ssh", "polarfire", "python3 --version"):
            return mod.CommandResult(command, 0, self.python_version + "\n", "")
        if command == ("ssh", "polarfire", "uptime"):
            return mod.CommandResult(
                command,
                0,
                " 01:24:00 up 8 days,  7:00,  0 user,  load average: 0.00\n",
                "",
            )
        if command == ("ssh", "polarfire", "uname -r"):
            return mod.CommandResult(
                command, 0, "6.18.17-linux4microchip-2026.04.1\n", ""
            )
        if command[0] == "ssh" and command[1] == "polarfire":
            return mod.CommandResult(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command}")


def test_req_hw_062_sat_instance_is_stable_50_clause_payload() -> None:
    """REQ-HW-062: the dispatched workload is a deterministic 50-clause SAT instance."""
    first = mod.generate_sat_instance()
    second = mod.generate_sat_instance()

    assert first == second
    assert first["num_variables"] == 16
    assert len(first["clauses"]) == 50
    assert mod.sha256_json(first) == mod.sha256_json(second)


def test_scenario_hw_062_generated_scorer_matches_local_hash(tmp_path: Path) -> None:
    """SCENARIO-HW-062: remote scorer output hash matches the local expected hash."""
    instance = mod.generate_sat_instance()
    instance_path = tmp_path / "sat_instance.json"
    transcript_path = tmp_path / "transcript.json"
    scorer_path = tmp_path / "scorer.py"
    instance_path.write_text(json.dumps(instance))
    scorer_path.write_text(mod.build_remote_scorer_source())

    subprocess.run(
        [
            sys.executable,
            str(scorer_path),
            str(instance_path),
            str(transcript_path),
            "--min-runtime-s",
            "0",
            "--min-cycles",
            "1",
        ],
        check=True,
    )

    transcript = json.loads(transcript_path.read_text())
    expected_output = mod.score_sat_instance(instance)
    assert transcript["scorer_output"] == expected_output
    assert transcript["scorer_output_sha256"] == mod.sha256_json(expected_output)
    assert len(transcript["per_clause_wall_clock_us"]) == 50
    assert all(value > 0 for value in transcript["per_clause_wall_clock_us"])


def test_req_hw_062_preconditions_block_unreachable_ssh() -> None:
    """REQ-HW-062: SSH reachability is checked before any architecture probe."""
    fake = FakeRunner(fail_on="BatchMode=yes polarfire true")

    checks, metadata, blocker = mod.check_preconditions(fake)

    assert blocker == "blocked_polarfire_ssh_unreachable"
    assert metadata == {}
    assert checks[0]["resource"] == "polarfire_ssh"
    assert checks[0]["passed"] is False
    assert not any("uname -m" in " ".join(command) for command in fake.commands)


def test_req_hw_062_preconditions_block_wrong_architecture() -> None:
    """REQ-HW-062: non-riscv64 boards are blocked before scorer dispatch."""
    checks, metadata, blocker = mod.check_preconditions(FakeRunner(arch="x86_64"))

    assert blocker == "blocked_polarfire_wrong_architecture"
    assert metadata["polarfire_arch"] == "x86_64"
    assert checks[-1]["resource"] == "polarfire_arch"
    assert checks[-1]["passed"] is False


def test_req_hw_062_preconditions_block_old_python() -> None:
    """REQ-HW-062: Python older than 3.10 is a dispatch blocker."""
    checks, metadata, blocker = mod.check_preconditions(
        FakeRunner(python_version="Python 3.9.18")
    )

    assert blocker == "blocked_polarfire_python_missing"
    assert metadata["polarfire_python"] == "Python 3.9.18"
    assert checks[-1]["resource"] == "polarfire_python"
    assert checks[-1]["passed"] is False
    assert mod.parse_python_version("python missing") is None


def test_scenario_hw_062_success_run_writes_artifact_and_transcript(tmp_path: Path) -> None:
    """SCENARIO-HW-062: successful dispatch writes the required terminal artifact."""
    output_path = tmp_path / "experiment_2900.json"
    transcript_path = tmp_path / "experiment_2900_transcript.json"
    fake = FakeRunner()
    ticks = iter([100.0, 111.5])

    artifact = mod.run_experiment(
        output_path=output_path,
        transcript_output_path=transcript_path,
        runner=fake,
        clock=lambda: next(ticks),
        remote_dir="/tmp/carnot-exp2900-test",
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert json.loads(output_path.read_text()) == artifact
    assert json.loads(transcript_path.read_text()) == fake.transcript
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["polarfire_arch"] == "riscv64"
    assert artifact["scorer_output_hash_verified"] is True
    assert artifact["duration_s"] == 11.5
    assert len(artifact["per_clause_wall_clock_us"]) == 50
    assert any(command[0] == "scp" for command in fake.commands)


def test_req_hw_062_run_experiment_stops_on_precondition_block(tmp_path: Path) -> None:
    """REQ-HW-062: run_experiment writes a blocked artifact before SCP."""
    output_path = tmp_path / "experiment_2900.json"
    transcript_path = tmp_path / "experiment_2900_transcript.json"
    fake = FakeRunner(fail_on="BatchMode=yes polarfire true")
    ticks = iter([7.0, 7.25])

    artifact = mod.run_experiment(
        output_path=output_path,
        transcript_output_path=transcript_path,
        runner=fake,
        clock=lambda: next(ticks),
        remote_dir="/tmp/carnot-exp2900-test",
    )

    assert artifact["honest_verdict"] == "blocked_polarfire_ssh_unreachable"
    assert artifact["duration_s"] == 0.25
    assert json.loads(output_path.read_text()) == artifact
    assert not transcript_path.exists()
    assert not any(command[0] == "scp" for command in fake.commands)


def test_req_hw_062_hash_mismatch_fails_the_terminal_artifact() -> None:
    """REQ-HW-062: the artifact cannot pass when the transcript hash mismatches."""
    transcript = _success_transcript()
    transcript["scorer_output_sha256"] = "0" * 64
    instance = mod.generate_sat_instance()

    artifact = mod.compose_terminal_artifact(
        checks=[],
        metadata={"polarfire_arch": "riscv64", "polarfire_kernel": "k", "uptime": "u"},
        instance=instance,
        transcript=transcript,
        duration_s=12.0,
    )

    assert artifact["honest_verdict"] == "failed_polarfire_scorer_output_hash_mismatch"
    assert artifact["scorer_output_hash_verified"] is False


def test_req_hw_062_duration_gate_fails_fast_success_claims() -> None:
    """REQ-HW-062: successful hash verification still requires duration_s >= 10."""
    artifact = mod.compose_terminal_artifact(
        checks=[],
        metadata={"polarfire_arch": "riscv64", "polarfire_kernel": "k", "uptime": "u"},
        instance=mod.generate_sat_instance(),
        transcript=_success_transcript(),
        duration_s=9.9,
    )

    assert artifact["honest_verdict"] == "failed_polarfire_duration_gate"
    assert artifact["scorer_output_hash_verified"] is True


def test_req_hw_062_dispatch_failure_writes_blocked_artifact(tmp_path: Path) -> None:
    """REQ-HW-062: SCP/remote failures are explicit and preserve preconditions."""
    output_path = tmp_path / "experiment_2900.json"
    transcript_path = tmp_path / "experiment_2900_transcript.json"
    fake = FakeRunner(fail_on="scp")
    ticks = iter([20.0, 21.0])

    artifact = mod.run_experiment(
        output_path=output_path,
        transcript_output_path=transcript_path,
        runner=fake,
        clock=lambda: next(ticks),
        remote_dir="/tmp/carnot-exp2900-test",
    )

    assert artifact["honest_verdict"] == "blocked_polarfire_dispatch_failed"
    assert artifact["scorer_output_hash_verified"] is False
    assert artifact["duration_s"] == 1.0
    assert json.loads(output_path.read_text()) == artifact
    assert not transcript_path.exists()
