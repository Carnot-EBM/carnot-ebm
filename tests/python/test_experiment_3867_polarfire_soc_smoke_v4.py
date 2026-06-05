"""Tests for Exp 3867 PolarFire SoC Ising dispatch v4.

Spec refs: REQ-HW-3867, SCENARIO-HW-3867.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from carnot import experiment_3867_polarfire_soc_smoke_v4 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"


class FakeRunner:
    """Command runner that models SSH/SCP states without touching hardware."""

    def __init__(
        self,
        *,
        fail_ssh: bool = False,
        fail_python: bool = False,
        board_result: dict[str, object] | None = None,
        temp_stdout: str = "42000\n39750\n",
    ) -> None:
        self.fail_ssh = fail_ssh
        self.fail_python = fail_python
        self.board_result = board_result
        self.temp_stdout = temp_stdout
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self, args: Sequence[str], timeout: float | None = None
    ) -> mod.CommandResult:
        del timeout
        command = tuple(args)
        self.commands.append(command)

        if command == mod.POLARFIRE_SSH_PRECONDITION:
            if self.fail_ssh:
                return mod.CommandResult(command, 255, "", "ssh timeout\n")
            return mod.CommandResult(command, 0, "", "")
        if command == mod.POLARFIRE_PYTHON_PRECONDITION:
            if self.fail_python:
                return mod.CommandResult(command, 127, "", "python3: not found\n")
            return mod.CommandResult(command, 0, "Python 3.12.12\n", "")
        if command[0] == "ssh" and command[2].startswith("mkdir -p "):
            return mod.CommandResult(command, 0, "", "")
        if command[0] == "scp":
            return mod.CommandResult(command, 0, "", "")
        if command[0] == "ssh" and "python3 runner.py workload.json" in command[2]:
            payload = self.board_result
            if payload is None:
                workload_path = mod.extract_local_workload_path(command[2])
                workload = json.loads(Path(workload_path).read_text(encoding="utf-8"))
                payload = mod.evaluate_ising_workload(workload)
            return mod.CommandResult(
                command,
                0,
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                "cycles=12\n",
            )
        if command[0] == "ssh" and "thermal_zone" in command[2]:
            return mod.CommandResult(command, 0, self.temp_stdout, "")
        if command[0] == "ssh" and command[2].startswith("rm -rf "):
            return mod.CommandResult(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command!r}")


def _clock(values: list[float]):
    ticks = iter(values)

    def monotonic() -> float:
        return next(ticks)

    return monotonic


def test_req_hw_3867_spec_entry_present() -> None:
    """REQ-HW-3867: OpenSpec declares the PolarFire hash-verified Ising artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-3867" in spec
    assert "SCENARIO-HW-3867" in spec
    assert "experiment_3867_polarfire_soc_smoke_v4.json" in spec
    assert "blocked_polarfire_ssh_timeout" in spec
    assert "blocked_polarfire_no_python" in spec


def test_req_hw_3867_workload_and_remote_runner_are_deterministic(tmp_path: Path) -> None:
    """REQ-HW-3867: CPU and board runner compute the same seed-controlled result hash."""
    workload = mod.build_ising_workload(random_seed=mod.RANDOM_SEED)
    second_workload = mod.build_ising_workload(random_seed=mod.RANDOM_SEED)
    assert workload == second_workload
    assert workload["random_seed"] == mod.RANDOM_SEED
    assert len(workload["spin_configs"]) == 6

    cpu_result = mod.evaluate_ising_workload(workload)
    workload_path = tmp_path / "workload.json"
    result_path = tmp_path / "result.json"
    runner_path = tmp_path / "runner.py"
    workload_path.write_text(json.dumps(workload), encoding="utf-8")
    runner_path.write_text(mod.build_remote_runner_source(), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(runner_path),
            str(workload_path),
            "--min-runtime-s",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result_path.write_text(completed.stdout, encoding="utf-8")
    board_result = json.loads(result_path.read_text(encoding="utf-8"))

    assert board_result == cpu_result
    assert mod.sha256_json(board_result) == mod.sha256_json(cpu_result)


def test_req_hw_3867_ssh_precondition_blocks_before_python_or_scp(tmp_path: Path) -> None:
    """REQ-HW-3867: SSH failure writes blocked_polarfire_ssh_timeout and stops."""
    fake = FakeRunner(fail_ssh=True)
    artifact = mod.run_experiment(
        repo_root=tmp_path,
        runner=fake,
        clock=_clock([10.0, 10.25]),
    )

    assert artifact["honest_verdict"] == "blocked_polarfire_ssh_timeout"
    assert artifact["preconditions_checked"] == [
        {
            "resource": "polarfire_ssh",
            "command": mod.command_to_string(mod.POLARFIRE_SSH_PRECONDITION),
            "passed": False,
            "exit_code": 255,
            "observed": "ssh timeout",
            "principle": mod.FIELD_PRINCIPLES["preconditions_checked"],
        }
    ]
    assert artifact["polarfire_workload_validated"] is False
    assert artifact["result_hash_match"] is False
    assert not any(command[0] == "scp" for command in fake.commands)
    assert mod.POLARFIRE_PYTHON_PRECONDITION not in fake.commands


def test_req_hw_3867_python_precondition_blocks_before_scp(tmp_path: Path) -> None:
    """REQ-HW-3867: missing board Python writes blocked_polarfire_no_python."""
    fake = FakeRunner(fail_python=True)
    artifact = mod.run_experiment(
        repo_root=tmp_path,
        runner=fake,
        clock=_clock([20.0, 20.5]),
    )

    assert artifact["honest_verdict"] == "blocked_polarfire_no_python"
    assert [row["resource"] for row in artifact["preconditions_checked"]] == [
        "polarfire_ssh",
        "polarfire_python",
    ]
    assert artifact["preconditions_checked"][1]["passed"] is False
    assert artifact["board_result_sha256"] == ""
    assert not any(command[0] == "scp" for command in fake.commands)


def test_scenario_hw_3867_success_artifact_hash_verified_terminal(tmp_path: Path) -> None:
    """SCENARIO-HW-3867: hash match and duration >=5 validate the PolarFire workload."""
    fake = FakeRunner()
    artifact = mod.run_experiment(
        repo_root=tmp_path,
        runner=fake,
        clock=_clock([100.0, 106.4]),
        remote_dir="/tmp/carnot-exp3867-test",
        min_remote_runtime_s=5.0,
    )
    on_disk = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert on_disk == artifact
    assert artifact["honest_verdict"].startswith(
        "success: polarfire_carnot_dispatch_hash_verified_terminal_duration6.40s_temp42.0"
    )
    assert artifact["polarfire_workload_validated"] is True
    assert artifact["result_hash_match"] is True
    assert artifact["board_result_sha256"] == artifact["cpu_reference_sha256"]
    assert artifact["run_duration_s"] == 6.4
    assert artifact["soc_temp_max_c"] == 42.0
    assert artifact["thermal_note"] == mod.THERMAL_NOTE
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert len(artifact["reproducibility_checksum"]) == 64
    assert [field for field in mod.REQUIRED_ARTIFACT_FIELDS if field not in artifact] == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert all(
        artifact["field_provenance"][field]["principle"]
        for field in mod.REQUIRED_ARTIFACT_FIELDS
    )
    assert any(command[0] == "scp" for command in fake.commands)


def test_req_hw_3867_hash_mismatch_is_partial_not_validated(tmp_path: Path) -> None:
    """REQ-HW-3867: board result hash mismatch cannot pass the terminal gate."""
    workload = mod.build_ising_workload(random_seed=mod.RANDOM_SEED)
    wrong = mod.evaluate_ising_workload(workload)
    wrong["energies"] = list(wrong["energies"])
    wrong["energies"][0] += 1
    fake = FakeRunner(board_result=wrong)

    artifact = mod.run_experiment(
        repo_root=tmp_path,
        runner=fake,
        clock=_clock([200.0, 207.0]),
    )

    assert artifact["honest_verdict"].startswith(
        "complete: polarfire_dispatch_ran_hash_MISMATCH"
    )
    assert artifact["polarfire_workload_validated"] is False
    assert artifact["result_hash_match"] is False
    assert artifact["board_result_sha256"] != artifact["cpu_reference_sha256"]


def test_req_hw_3867_dispatch_failure_writes_blocked_artifact(tmp_path: Path) -> None:
    """REQ-HW-3867: SCP or remote failures are blocked without validating workload."""

    class ScpFailRunner(FakeRunner):
        def __call__(
            self, args: Sequence[str], timeout: float | None = None
        ) -> mod.CommandResult:
            command = tuple(args)
            if command[0] == "scp":
                self.commands.append(command)
                return mod.CommandResult(command, 1, "", "scp failed\n")
            return super().__call__(args, timeout)

    artifact = mod.run_experiment(
        repo_root=tmp_path,
        runner=ScpFailRunner(),
        clock=_clock([400.0, 401.0]),
    )

    assert artifact["honest_verdict"] == "blocked_polarfire_dispatch_failed"
    assert artifact["polarfire_workload_validated"] is False
    assert artifact["result_hash_match"] is False
    assert "scp_push: rc=1" in artifact["failure_detail"]
    assert artifact["command_transcript"][-1]["stage"] == "remote_cleanup"


def test_req_hw_3867_invalid_remote_json_blocks_dispatch(tmp_path: Path) -> None:
    """REQ-HW-3867: malformed board stdout is a dispatch blocker, not a mismatch."""

    class BadJsonRunner(FakeRunner):
        def __call__(
            self, args: Sequence[str], timeout: float | None = None
        ) -> mod.CommandResult:
            command = tuple(args)
            if command[0] == "ssh" and "python3 runner.py workload.json" in command[2]:
                self.commands.append(command)
                return mod.CommandResult(command, 0, "not-json\n", "")
            return super().__call__(args, timeout)

    artifact = mod.run_experiment(
        repo_root=tmp_path,
        runner=BadJsonRunner(),
        clock=_clock([500.0, 501.0]),
    )

    assert artifact["honest_verdict"] == "blocked_polarfire_dispatch_failed"
    assert "remote_ising_eval_json" in artifact["failure_detail"]


def test_req_hw_3867_duration_gate_prevents_fast_terminal_claim(tmp_path: Path) -> None:
    """REQ-HW-3867: hash match alone is insufficient when run_duration_s is under 5."""
    artifact = mod.run_experiment(
        repo_root=tmp_path,
        runner=FakeRunner(),
        clock=_clock([300.0, 304.99]),
        min_remote_runtime_s=0.0,
    )

    assert artifact["result_hash_match"] is True
    assert artifact["polarfire_workload_validated"] is False
    assert artifact["honest_verdict"] == (
        "complete: polarfire_dispatch_ran_duration_lt5_workload_not_validated"
    )


def test_req_hw_3867_temperature_parser_handles_missing_values() -> None:
    """REQ-HW-3867: SoC temperature is null when sysfs readings are unavailable."""
    assert mod.parse_soc_temp_max_c("42000\n39750\n") == 42.0
    assert mod.parse_soc_temp_max_c("42.5\nnot-a-number\n") == 42.5
    assert mod.parse_soc_temp_max_c("") is None
    assert mod.parse_soc_temp_max_c("cat: no such file\n") is None
    try:
        mod.extract_local_workload_path("python3 runner.py workload.json")
    except ValueError as exc:
        assert "local_workload_path" in str(exc)
    else:  # pragma: no cover - defensive assertion shape.
        raise AssertionError("missing marker should raise")


def test_req_hw_3867_thermal_probe_failure_returns_null() -> None:
    """REQ-HW-3867: failed thermal sysfs probe records null temperature."""

    def runner(args: Sequence[str], timeout: float | None = None) -> mod.CommandResult:
        del timeout
        command = tuple(args)
        return mod.CommandResult(command, 1, "", "no thermal zones\n")

    temp, transcript = mod.probe_soc_temp_max_c(runner, "polarfire")

    assert temp is None
    assert transcript["stage"] == "thermal_probe"
    assert transcript["returncode"] == 1


def test_req_hw_3867_script_wrapper_exists() -> None:
    """SCENARIO-HW-3867: conductor entrypoint delegates to the module."""
    script = Path("scripts/experiments/experiment_3867_polarfire_soc_smoke_v4.py")
    text = script.read_text(encoding="utf-8")
    assert "experiment_3867_polarfire_soc_smoke_v4" in text
    assert "run_experiment" in text
