from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

from carnot.hardware import polarfire_1000_clause_scorer_2958 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

REQUIRED_FIELDS = {
    "honest_verdict",
    "preconditions_checked",
    "polarfire_1000_clause_hash_verified",
    "baseline_500_clause_artifact",
    "clause_count",
    "input_sha256",
    "scorer_output_sha256",
    "transcript_paths",
    "elapsed_ms",
    "board_reachable",
    "no_speedup_claim",
    "no_general_acceleration_claim",
    "inference_substrate",
    "duration_s",
}


def _baseline_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: polarfire_500_clause_constraint_scorer_hash_verified",
        "n_clauses": 500,
        "scorer_output_hash_verified": True,
        "sat_instance_sha256": "0fc2fb6c861f763a531ae966ae072ccb796d03f3492431a453c13bd7eb90b3fc",
        "scorer_output_sha256": "3b91df6ed5a3fa14e4810994fb185e6869afe78a3827a2efd8d104e02004ad66",
        "inference_substrate": "hardware_smoke",
    }


def _write_baseline(root: Path) -> None:
    artifact_path = root / exp.BASELINE_500_CLAUSE_REL_PATH
    transcript_path = root / exp.BASELINE_500_TRANSCRIPT_REL_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(_baseline_artifact(), sort_keys=True), encoding="utf-8")
    transcript_path.write_text(
        json.dumps({"schema": "carnot.polarfire_sat_scorer_transcript.v2"}),
        encoding="utf-8",
    )


def _success_transcript() -> dict[str, Any]:
    instance = exp.generate_sat_instance()
    scorer_output = exp.score_sat_instance(instance)
    return {
        "schema": exp.TRANSCRIPT_SCHEMA,
        "spec_refs": exp.SPEC_REFS,
        "remote_arch": "riscv64",
        "remote_python": "3.12.12",
        "sat_instance_sha256": exp.sha256_json(instance),
        "scorer_output": scorer_output,
        "scorer_output_sha256": exp.sha256_json(scorer_output),
        "per_clause_wall_clock_us": [25.0 + (idx % 7) for idx in range(exp.CLAUSE_COUNT)],
        "total_wall_clock_s": 1.25,
        "evaluation_cycles_per_clause": 3,
    }


class FakeRunner:
    def __init__(
        self,
        transcript: dict[str, Any] | None = None,
        *,
        arch: str = "riscv64",
        fail_on: str | None = None,
    ) -> None:
        self.transcript = transcript or _success_transcript()
        self.arch = arch
        self.fail_on = fail_on
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, args: Sequence[str], timeout: float | None = None) -> exp.CommandResult:
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
            return exp.CommandResult(command, 0, " 10:00:00 up 10 days\n", "")
        if command == ("ssh", "polarfire", "uname -r"):
            return exp.CommandResult(command, 0, "6.18.17-linux4microchip\n", "")
        if command[0] == "ssh" and command[1] == "polarfire":
            return exp.CommandResult(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command}")


def test_req_hw_076_spec_anchor_exists() -> None:
    """REQ-HW-076: OpenSpec anchors the 1000-clause PolarFire continuation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-076" in spec
    assert "SCENARIO-HW-076" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "no_general_acceleration_claim=true" in spec


def test_req_hw_076_sat_instance_is_stable_1000_clause_payload() -> None:
    """REQ-HW-076: Exp 2958 ships a deterministic 1000-clause SAT instance."""

    first = exp.generate_sat_instance()
    second = exp.generate_sat_instance()

    assert first == second
    assert first["random_seed"] == exp.RANDOM_SEED
    assert first["spec_refs"] == exp.SPEC_REFS
    assert first["num_variables"] == exp.NUM_VARIABLES
    assert len(first["clauses"]) == 1000
    assert exp.sha256_json(first) == exp.sha256_json(second)


def test_scenario_hw_076_generated_scorer_matches_local_hash(tmp_path: Path) -> None:
    """SCENARIO-HW-076: remote 1000-clause scorer hash matches the local expected hash."""

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
    assert len(transcript["per_clause_wall_clock_us"]) == 1000


def test_req_hw_076_baseline_context_blocks_missing_or_unverified(tmp_path: Path) -> None:
    """REQ-HW-076: Exp 2941 baseline hash context is checked before board dispatch."""

    checks, blocker = exp.check_baseline_context(tmp_path)
    assert blocker == "blocked_baseline_500_clause_artifact_missing"
    assert checks[0]["resource"] == "baseline_500_clause_artifact"
    assert checks[0]["passed"] is False

    artifact_path = tmp_path / exp.BASELINE_500_CLAUSE_REL_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("{", encoding="utf-8")
    checks, blocker = exp.check_baseline_context(tmp_path)
    assert blocker == "blocked_baseline_500_clause_artifact_invalid_json"

    artifact_path.write_text("[]", encoding="utf-8")
    checks, blocker = exp.check_baseline_context(tmp_path)
    assert blocker == "blocked_baseline_500_clause_artifact_invalid_json"

    artifact_path.write_text(
        json.dumps(dict(_baseline_artifact(), scorer_output_hash_verified=False)),
        encoding="utf-8",
    )
    checks, blocker = exp.check_baseline_context(tmp_path)
    assert blocker == "blocked_baseline_500_clause_hash_unverified"

    artifact_path.write_text(json.dumps(_baseline_artifact()), encoding="utf-8")
    checks, blocker = exp.check_baseline_context(tmp_path)
    assert blocker == "blocked_baseline_500_clause_transcript_missing"
    assert checks[-1]["resource"] == "baseline_500_clause_transcript"

    _write_baseline(tmp_path)
    checks, blocker = exp.check_baseline_context(tmp_path)
    assert blocker is None
    assert {check["resource"] for check in checks} == {
        "baseline_500_clause_artifact",
        "baseline_500_clause_transcript",
    }
    assert all(check["passed"] for check in checks)


def test_req_hw_076_compose_success_artifact_has_required_fields() -> None:
    """REQ-HW-076: terminal artifact records hash evidence and explicit no-claim flags."""

    artifact = exp.compose_terminal_artifact(
        checks=[],
        metadata={"polarfire_arch": "riscv64"},
        instance=exp.generate_sat_instance(),
        transcript=_success_transcript(),
        duration_s=2.0,
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["polarfire_1000_clause_hash_verified"] is True
    assert artifact["clause_count"] == 1000
    assert artifact["board_reachable"] is True
    assert artifact["elapsed_ms"] == pytest.approx(1250.0)
    assert artifact["transcript_paths"] == [
        exp.BASELINE_500_TRANSCRIPT_REL_PATH.as_posix(),
        exp.TRANSCRIPT_REL_PATH.as_posix(),
    ]
    assert artifact["no_speedup_claim"] is True
    assert artifact["no_general_acceleration_claim"] is True
    assert artifact["inference_substrate"] == "hardware_smoke"


def test_req_hw_076_hash_mismatch_fails_closed() -> None:
    """REQ-HW-076: a remote hash mismatch cannot become a verified 1000-clause artifact."""

    transcript = _success_transcript()
    transcript["scorer_output_sha256"] = "0" * 64
    artifact = exp.compose_terminal_artifact(
        checks=[],
        metadata={},
        instance=exp.generate_sat_instance(),
        transcript=transcript,
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "failed_polarfire_1000_clause_hash_mismatch"
    assert artifact["polarfire_1000_clause_hash_verified"] is False
    assert artifact["scorer_output_sha256"] == "0" * 64


def test_req_hw_076_validate_artifact_rejects_bad_shapes() -> None:
    """REQ-HW-076: schema validation rejects incomplete or claim-breaking artifacts."""

    artifact = exp.compose_terminal_artifact(
        checks=[],
        metadata={},
        instance=exp.generate_sat_instance(),
        transcript=_success_transcript(),
        duration_s=2.0,
    )

    missing = dict(artifact)
    del missing["duration_s"]
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad_cases = [
        ("clause_count", 999, "1000"),
        ("inference_substrate", "simulation", "hardware_smoke"),
        ("no_speedup_claim", False, "no_speedup_claim"),
        ("no_general_acceleration_claim", False, "no_general_acceleration_claim"),
        ("input_sha256", "bad", "input_sha256"),
        ("scorer_output_sha256", "bad", "scorer_output_sha256"),
        ("transcript_paths", [], "transcript_paths"),
    ]
    for key, value, message in bad_cases:
        bad = dict(artifact)
        bad[key] = value
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(bad)


def test_req_hw_076_run_experiment_blocks_missing_baseline_before_ssh(tmp_path: Path) -> None:
    """REQ-HW-076: missing Exp 2941 baseline blocks before any PolarFire SSH."""

    fake = FakeRunner()
    ticks = iter([1.0, 1.5])
    artifact = exp.run_experiment(
        root_path=tmp_path,
        runner=fake,
        clock=lambda: next(ticks),
        remote_dir="/tmp/carnot-exp2958-test",
    )

    assert artifact["honest_verdict"] == "blocked_baseline_500_clause_artifact_missing"
    assert artifact["board_reachable"] is False
    assert artifact["transcript_paths"] == []
    assert fake.commands == []


def test_scenario_hw_076_run_experiment_writes_blocked_board_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-076: unreachable board preserves baseline transcript evidence."""

    _write_baseline(tmp_path)
    fake = FakeRunner(fail_on="BatchMode=yes polarfire true")
    ticks = iter([10.0, 10.75])
    artifact = exp.run_experiment(
        root_path=tmp_path,
        runner=fake,
        clock=lambda: next(ticks),
        remote_dir="/tmp/carnot-exp2958-test",
    )

    assert artifact["honest_verdict"] == "blocked_polarfire_ssh_unreachable"
    assert artifact["board_reachable"] is False
    assert artifact["polarfire_1000_clause_hash_verified"] is False
    assert artifact["transcript_paths"] == [exp.BASELINE_500_TRANSCRIPT_REL_PATH.as_posix()]
    assert not any(command[0] == "scp" for command in fake.commands)


def test_scenario_hw_076_run_experiment_writes_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-076: reachable board writes the 1000-clause result and transcript."""

    _write_baseline(tmp_path)
    fake = FakeRunner()
    ticks = iter([100.0, 102.0])
    artifact = exp.run_experiment(
        root_path=tmp_path,
        runner=fake,
        clock=lambda: next(ticks),
        remote_dir="/tmp/carnot-exp2958-test",
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["board_reachable"] is True
    assert artifact["polarfire_1000_clause_hash_verified"] is True
    assert artifact["elapsed_ms"] == pytest.approx(1250.0)
    assert json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    written_transcript = json.loads(
        (tmp_path / exp.TRANSCRIPT_REL_PATH).read_text(encoding="utf-8")
    )
    assert written_transcript == fake.transcript
    assert any(command[0] == "scp" for command in fake.commands)


def test_req_hw_076_dispatch_failure_writes_blocked_artifact(tmp_path: Path) -> None:
    """REQ-HW-076: SCP or remote scorer failures are explicit blocked artifacts."""

    _write_baseline(tmp_path)
    fake = FakeRunner(fail_on="scp")
    ticks = iter([20.0, 21.0])
    artifact = exp.run_experiment(
        root_path=tmp_path,
        runner=fake,
        clock=lambda: next(ticks),
        remote_dir="/tmp/carnot-exp2958-test",
    )

    assert artifact["honest_verdict"] == "blocked_polarfire_dispatch_failed"
    assert artifact["board_reachable"] is True
    assert artifact["polarfire_1000_clause_hash_verified"] is False
    assert artifact["failure_detail"]
