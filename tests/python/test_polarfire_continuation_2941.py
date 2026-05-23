from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pytest

from carnot.hardware import polarfire_continuation_2941 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

REQUIRED_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "polarfire_ssh_uptime_at_run",
    "n_clauses",
    "per_clause_wall_clock_us_median",
    "per_clause_wall_clock_us_p95",
    "scaling_ratio_vs_exp2900",
    "scorer_output_sha256",
    "scorer_output_hash_verified",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
}


def _exp2900_artifact() -> dict[str, object]:
    return {
        "honest_verdict": "complete: polarfire_riscv64_constraint_scorer_hash_verified",
        "inference_substrate": "hardware_smoke",
        "scorer_output_hash_verified": True,
        "per_clause_wall_clock_us": [40.0, 50.0, 60.0],
    }


def _write_exp2900(root: Path) -> None:
    path = root / exp.EXP2900_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_exp2900_artifact(), sort_keys=True), encoding="utf-8")


def _success_transcript() -> dict[str, object]:
    instance = exp.generate_sat_instance()
    scorer_output = exp.score_sat_instance(instance)
    return {
        "schema": "carnot.polarfire_sat_scorer_transcript.v2",
        "spec_refs": exp.SPEC_REFS,
        "remote_arch": "riscv64",
        "remote_python": "3.12.12",
        "sat_instance_sha256": exp.sha256_json(instance),
        "scorer_output": scorer_output,
        "scorer_output_sha256": exp.sha256_json(scorer_output),
        "per_clause_wall_clock_us": [45.0 + (idx % 11) for idx in range(exp.N_CLAUSES)],
        "total_wall_clock_s": 15.25,
        "evaluation_cycles_per_clause": 19,
    }


class FakeRunner:
    def __init__(
        self,
        transcript: dict[str, object] | None = None,
        *,
        arch: str = "riscv64",
        fail_on: str | None = None,
    ) -> None:
        self.transcript = transcript or _success_transcript()
        self.arch = arch
        self.fail_on = fail_on
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self, args: Sequence[str], timeout: float | None = None
    ) -> exp.CommandResult:
        del timeout
        command = tuple(args)
        self.commands.append(command)
        joined = " ".join(command)
        if self.fail_on and self.fail_on in joined:
            return exp.CommandResult(command, 1, "", "forced failure")
        if command[:2] == ("scp", "-q") and "transcript.json" in command[2]:
            Path(command[3]).write_text(json.dumps(self.transcript), encoding="utf-8")
            return exp.CommandResult(command, 0, "", "")
        if command[:2] == ("scp", "-q"):
            return exp.CommandResult(command, 0, "", "")
        if command == (
            "ssh",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "BatchMode=yes",
            "polarfire",
            "true",
        ):
            return exp.CommandResult(command, 0, "", "")
        if command == ("ssh", "polarfire", "uname -m"):
            return exp.CommandResult(command, 0, self.arch + "\n", "")
        if command == ("ssh", "polarfire", "python3 --version"):
            return exp.CommandResult(command, 0, "Python 3.12.12\n", "")
        if command == ("ssh", "polarfire", "uptime"):
            return exp.CommandResult(
                command,
                0,
                " 10:00:00 up 9 days,  1:02,  0 user,  load average: 0.00\n",
                "",
            )
        if command == ("ssh", "polarfire", "uname -r"):
            return exp.CommandResult(command, 0, "6.18.17-linux4microchip\n", "")
        if command[0] == "ssh" and command[1] == "polarfire":
            return exp.CommandResult(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command}")


def test_req_hw_073_spec_anchor_exists() -> None:
    """REQ-HW-073: OpenSpec anchors the 500-clause PolarFire continuation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-073" in spec
    assert "SCENARIO-HW-073" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "scaling_ratio_vs_exp2900" in spec


def test_req_hw_073_sat_instance_is_stable_500_clause_payload() -> None:
    """REQ-HW-073: Exp 2941 ships a deterministic 500-clause SAT instance."""

    first = exp.generate_sat_instance()
    second = exp.generate_sat_instance()

    assert first == second
    assert first["random_seed"] == exp.RANDOM_SEED
    assert first["num_variables"] == exp.NUM_VARIABLES
    assert len(first["clauses"]) == 500
    assert exp.sha256_json(first) == exp.sha256_json(second)


def test_scenario_hw_073_generated_scorer_matches_local_hash(tmp_path: Path) -> None:
    """SCENARIO-HW-073: remote scorer output hash matches the local expected hash."""

    instance = exp.generate_sat_instance()
    instance_path = tmp_path / "sat_instance.json"
    transcript_path = tmp_path / "transcript.json"
    scorer_path = tmp_path / "scorer.py"
    instance_path.write_text(json.dumps(instance), encoding="utf-8")
    scorer_path.write_text(exp.build_remote_scorer_source(), encoding="utf-8")

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

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    expected_output = exp.score_sat_instance(instance)
    assert transcript["scorer_output"] == expected_output
    assert transcript["scorer_output_sha256"] == exp.sha256_json(expected_output)
    assert len(transcript["per_clause_wall_clock_us"]) == 500


def test_req_hw_073_compose_artifact_records_scaling_and_required_fields() -> None:
    """REQ-HW-073: terminal artifact records timing summary and Exp 2900 ratio."""

    artifact = exp.compose_terminal_artifact(
        checks=[],
        metadata={"polarfire_arch": "riscv64", "polarfire_kernel": "k", "uptime": "u"},
        instance=exp.generate_sat_instance(),
        transcript=_success_transcript(),
        exp2900_median_us=50.0,
        duration_s=16.0,
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["n_clauses"] == 500
    assert artifact["per_clause_wall_clock_us_median"] == pytest.approx(50.0)
    assert artifact["per_clause_wall_clock_us_p95"] == pytest.approx(55.0)
    assert artifact["scaling_ratio_vs_exp2900"] == pytest.approx(1.0)
    assert artifact["scorer_output_hash_verified"] is True
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_hw_073_hash_and_duration_gates_fail_closed() -> None:
    """REQ-HW-073: success requires both hash match and duration_s >= 15."""

    bad_transcript = _success_transcript()
    bad_transcript["scorer_output_sha256"] = "0" * 64
    mismatch = exp.compose_terminal_artifact(
        checks=[],
        metadata={},
        instance=exp.generate_sat_instance(),
        transcript=bad_transcript,
        exp2900_median_us=50.0,
        duration_s=16.0,
    )
    assert mismatch["honest_verdict"] == "failed_polarfire_scorer_output_hash_mismatch"
    assert mismatch["scorer_output_hash_verified"] is False

    too_fast = exp.compose_terminal_artifact(
        checks=[],
        metadata={},
        instance=exp.generate_sat_instance(),
        transcript=_success_transcript(),
        exp2900_median_us=50.0,
        duration_s=14.99,
    )
    assert too_fast["honest_verdict"] == "failed_polarfire_duration_gate"
    assert too_fast["scorer_output_hash_verified"] is True


def test_req_hw_073_exp2900_median_rejects_bad_provenance(tmp_path: Path) -> None:
    """REQ-HW-073: Exp 2900 baseline timing must be hash-verified hardware smoke."""

    with pytest.raises(exp.ContinuationError) as missing:
        exp.load_exp2900_median(tmp_path)
    assert missing.value.verdict == "blocked_exp2900_artifact_missing"
    assert str(missing.value)

    path = tmp_path / exp.EXP2900_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("{", encoding="utf-8")
    with pytest.raises(exp.ContinuationError) as invalid_json:
        exp.load_exp2900_median(tmp_path)
    assert invalid_json.value.verdict == "blocked_exp2900_artifact_invalid_json"

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(exp.ContinuationError) as non_object:
        exp.load_exp2900_median(tmp_path)
    assert non_object.value.verdict == "blocked_exp2900_artifact_invalid_json"

    path.write_text(
        json.dumps(dict(_exp2900_artifact(), inference_substrate="simulation")),
        encoding="utf-8",
    )
    with pytest.raises(exp.ContinuationError) as wrong_substrate:
        exp.load_exp2900_median(tmp_path)
    assert wrong_substrate.value.verdict == "blocked_exp2900_not_hardware_smoke"

    path.write_text(
        json.dumps(dict(_exp2900_artifact(), scorer_output_hash_verified=False)),
        encoding="utf-8",
    )
    with pytest.raises(exp.ContinuationError) as unverified:
        exp.load_exp2900_median(tmp_path)
    assert unverified.value.verdict == "blocked_exp2900_hash_not_verified"

    path.write_text(
        json.dumps(dict(_exp2900_artifact(), per_clause_wall_clock_us=[])),
        encoding="utf-8",
    )
    with pytest.raises(exp.ContinuationError) as no_timing:
        exp.load_exp2900_median(tmp_path)
    assert no_timing.value.verdict == "blocked_exp2900_timing_missing"

    _write_exp2900(tmp_path)
    assert exp.load_exp2900_median(tmp_path) == pytest.approx(50.0)


def test_req_hw_073_validate_artifact_rejects_bad_success_claims() -> None:
    """REQ-HW-073: schema validation refuses incomplete successful artifacts."""

    artifact = exp.compose_terminal_artifact(
        checks=[],
        metadata={},
        instance=exp.generate_sat_instance(),
        transcript=_success_transcript(),
        exp2900_median_us=50.0,
        duration_s=16.0,
    )
    assert exp.percentile_nearest_rank([], 95.0) == 0.0

    missing = dict(artifact)
    del missing["duration_s"]
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad_cases = [
        ("inference_substrate", "simulation", "hardware_smoke"),
        ("n_clauses", 499, "500"),
        ("random_seed", 1, "random_seed"),
        ("reproducibility_checksum", None, "reproducibility_checksum"),
        ("scorer_output_hash_verified", False, "hash verification"),
        ("duration_s", 1.0, "duration_s"),
        ("per_clause_wall_clock_us_median", 0.0, "median"),
        ("per_clause_wall_clock_us_p95", 0.0, "p95"),
        ("scaling_ratio_vs_exp2900", 0.0, "scaling ratio"),
    ]
    for key, value, message in bad_cases:
        bad = dict(artifact)
        bad[key] = value
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(bad)


def test_req_hw_073_run_experiment_stops_on_precondition_block(tmp_path: Path) -> None:
    """REQ-HW-073: failed PolarFire SSH precondition blocks before SCP."""

    fake = FakeRunner(fail_on="BatchMode=yes polarfire true")
    ticks = iter([1.0, 1.25])

    artifact = exp.run_experiment(
        root_path=tmp_path,
        runner=fake,
        clock=lambda: next(ticks),
        remote_dir="/tmp/carnot-exp2941-test",
    )

    assert artifact["honest_verdict"] == "blocked_polarfire_ssh_unreachable"
    assert artifact["duration_s"] == pytest.approx(0.25)
    assert not any(command[0] == "scp" for command in fake.commands)


def test_scenario_hw_073_run_experiment_writes_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-073: successful dispatch writes the 500-clause result JSON."""

    _write_exp2900(tmp_path)
    fake = FakeRunner()
    ticks = iter([100.0, 116.5])

    artifact = exp.run_experiment(
        root_path=tmp_path,
        runner=fake,
        clock=lambda: next(ticks),
        remote_dir="/tmp/carnot-exp2941-test",
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == pytest.approx(16.5)
    assert artifact["scorer_output_hash_verified"] is True
    assert artifact["scaling_ratio_vs_exp2900"] == pytest.approx(1.0)
    assert json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert any(command[0] == "scp" for command in fake.commands)


def test_req_hw_073_dispatch_failure_writes_blocked_artifact(tmp_path: Path) -> None:
    """REQ-HW-073: SCP or remote scorer failures are explicit blocked artifacts."""

    _write_exp2900(tmp_path)
    fake = FakeRunner(fail_on="scp")
    ticks = iter([20.0, 21.0])

    artifact = exp.run_experiment(
        root_path=tmp_path,
        runner=fake,
        clock=lambda: next(ticks),
        remote_dir="/tmp/carnot-exp2941-test",
    )

    assert artifact["honest_verdict"] == "blocked_polarfire_dispatch_failed"
    assert artifact["scorer_output_hash_verified"] is False
    assert artifact["duration_s"] == pytest.approx(1.0)
    assert json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact


def test_req_hw_073_run_experiment_blocks_missing_exp2900(tmp_path: Path) -> None:
    """REQ-HW-073: missing Exp 2900 timing provenance blocks scaling comparison."""

    fake = FakeRunner()
    artifact = exp.run_experiment(
        root_path=tmp_path,
        runner=fake,
        clock=lambda: 10.0,
        remote_dir="/tmp/carnot-exp2941-test",
    )

    assert artifact["honest_verdict"] == "blocked_exp2900_artifact_missing"
    assert artifact["scorer_output_hash_verified"] is False
    assert not any(command[0] == "scp" for command in fake.commands)
