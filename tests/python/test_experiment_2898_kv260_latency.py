"""Tests for Exp 2898 KV260 hardware latency transcript.

REQ-HW-060: KV260 latency measurements must be precondition-gated,
transcript-backed, and free of CPU-speedup claims.
SCENARIO-HW-060: three seeded board runs produce reproducible latency
summaries and a provenance-rich hardware-smoke artifact.
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import experiment_2898_kv260_latency as exp  # noqa: E402


def _fake_preconditions() -> list[dict[str, object]]:
    return [
        {"resource": "kv260_ssh", "available": True, "detail": "ssh ok"},
        {"resource": "kv260_overlay", "available": True, "detail": "overlay listed"},
        {"resource": "kv260_uio0", "available": True, "detail": "/dev/uio0 ok"},
    ]


def _fake_board_payload() -> dict[str, object]:
    sweep = []
    for seed in exp.RANDOM_SEEDS:
        for n_samples in exp.N_SAMPLE_COUNTS:
            sweep.append(
                {
                    "seed": seed,
                    "n_samples": n_samples,
                    "per_sample_wall_clock_us_median": float(seed + n_samples / 1000.0),
                    "per_sample_wall_clock_us_p95": float(seed + n_samples / 500.0),
                    "final_energy": -float(seed),
                    "final_spin_words_hex": ["0xffffffff", "0x00000000"],
                    "failed_samples": 0,
                }
            )
    return {
        "selected_uio": "/dev/uio4",
        "uio0_mmap_checked": True,
        "uio_devices": ["/dev/uio0", "/dev/uio1", "/dev/uio2", "/dev/uio3", "/dev/uio4"],
        "runs": sweep,
    }


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys |= _all_keys(child)
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys |= _all_keys(child)
        return keys
    return set()


class TestProblemGeneration:
    def test_q88_clips_to_int16_range(self) -> None:
        assert exp.q88(200.0) == 32767
        assert exp.q88(-200.0) == -32768
        assert exp.q88(0.5) == 128

    def test_generate_ising_problem_is_symmetric_and_zero_field(self) -> None:
        problem = exp.generate_ising_problem(seed=42, n_spins=8)
        assert problem["n_spins"] == 8
        assert problem["random_seed"] == 42
        assert problem["h_vector"] == [0.0] * 8
        for i, row in enumerate(problem["j_matrix"]):
            assert row[i] == 0.0
            for j, value in enumerate(row):
                assert value == pytest.approx(problem["j_matrix"][j][i])

    def test_sparse_upload_uses_top_couplings_and_q88_encoding(self) -> None:
        problem = {
            "n_spins": 4,
            "random_seed": 1,
            "h_vector": [0.0, 0.0, 0.0, 0.0],
            "j_matrix": [
                [0.0, 0.5, -0.25, 0.125],
                [0.5, 0.0, 0.1, -0.9],
                [-0.25, 0.1, 0.0, 0.75],
                [0.125, -0.9, 0.75, 0.0],
            ],
        }
        upload = exp.build_sparse_upload(problem, max_degree=2)
        assert upload["max_degree"] == 2
        assert upload["h_q88"] == [0, 0, 0, 0]
        assert upload["adjacency"][0] == [1, 2]
        assert upload["couplings_q88"][0] == [128, -64]
        assert upload["adjacency"][1] == [3, 0]
        assert upload["couplings_q88"][1] == [-230, 128]

    def test_problem_payload_records_three_seed_hashes(self) -> None:
        payload = exp.build_problem_payload()
        assert payload["random_seeds_used"] == [42, 137, 271]
        assert len(payload["problems"]) == 3
        specs = payload["ising_problem_specs"]
        assert [item["random_seed"] for item in specs] == [42, 137, 271]
        for item in specs:
            assert item["n_spins"] == 64
            assert len(item["j_matrix_sha256"]) == 64
            assert len(item["h_vector_sha256"]) == 64


class TestArtifactAssembly:
    def test_blocked_artifact_has_required_hardware_smoke_fields(self, tmp_path: Path) -> None:
        artifact = exp.build_blocked_artifact(
            verdict="blocked_kv260_ssh_unreachable",
            preconditions_checked=[
                {"resource": "kv260_ssh", "available": False, "detail": "ssh rc=255"}
            ],
            duration_s=1.25,
            transcript_path=tmp_path / "transcript.log",
        )
        assert artifact["honest_verdict"].startswith("blocked_kv260_ssh_unreachable")
        assert artifact["inference_substrate"] == "hardware_smoke"
        assert artifact["random_seeds_used"] == [42, 137, 271]
        assert artifact["per_seed_results"] == []
        for field in exp.REQUIRED_ARTIFACT_FIELDS:
            assert field in artifact

    def test_success_artifact_collapses_to_three_seed_results(self, tmp_path: Path) -> None:
        problem_payload = exp.build_problem_payload()
        artifact = exp.build_success_artifact(
            preconditions_checked=_fake_preconditions(),
            uptime="up 2 hours",
            overlay_loaded="carnot_ising_v2_n64",
            overlay_load_command=exp.OVERLAY_LOAD_COMMAND,
            uio_devices_present=["/dev/uio0", "/dev/uio1", "/dev/uio2", "/dev/uio3", "/dev/uio4"],
            bitstream_sha256="a" * 64,
            problem_payload=problem_payload,
            board_payload=_fake_board_payload(),
            duration_s=31.0,
            transcript_path=tmp_path / "transcript.log",
        )
        assert artifact["honest_verdict"].startswith("complete:")
        assert artifact["kv260_overlay_loaded"] == "carnot_ising_v2_n64"
        assert len(artifact["per_seed_results"]) == 3
        assert [row["seed"] for row in artifact["per_seed_results"]] == [42, 137, 271]
        assert all(row["n_samples"] == 10000 for row in artifact["per_seed_results"])
        assert len(artifact["sample_count_sweep_results"]) == 9
        assert all(
            row["per_sample_wall_clock_us_median"] > 0
            for row in artifact["per_seed_results"]
        )
        assert "speedup" not in {key.lower() for key in _all_keys(artifact)}

    def test_reproducibility_checksum_depends_on_board_identity(self, tmp_path: Path) -> None:
        payload = exp.build_problem_payload()
        first = exp.build_success_artifact(
            preconditions_checked=_fake_preconditions(),
            uptime="up",
            overlay_loaded="carnot_ising_v2_n64",
            overlay_load_command=exp.OVERLAY_LOAD_COMMAND,
            uio_devices_present=["/dev/uio0"],
            bitstream_sha256="a" * 64,
            problem_payload=payload,
            board_payload=_fake_board_payload(),
            duration_s=31.0,
            transcript_path=tmp_path / "a.log",
        )
        second = exp.build_success_artifact(
            preconditions_checked=_fake_preconditions(),
            uptime="up",
            overlay_loaded="carnot_ising_v4",
            overlay_load_command=exp.OVERLAY_LOAD_COMMAND,
            uio_devices_present=["/dev/uio0"],
            bitstream_sha256="b" * 64,
            problem_payload=payload,
            board_payload=_fake_board_payload(),
            duration_s=31.0,
            transcript_path=tmp_path / "b.log",
        )
        assert first["reproducibility_checksum"] != second["reproducibility_checksum"]

    def test_sample_count_sweep_preserves_min_max_when_present(self) -> None:
        payload = {
            "runs": [
                {
                    "seed": 42,
                    "n_samples": 100,
                    "per_sample_wall_clock_us_median": 2.0,
                    "per_sample_wall_clock_us_p95": 3.0,
                    "per_sample_wall_clock_us_min": 1.0,
                    "per_sample_wall_clock_us_max": 4.0,
                    "final_energy": -1.0,
                }
            ]
        }
        rows = exp._sample_count_sweep(payload)
        assert rows[0]["per_sample_wall_clock_us_min"] == 1.0
        assert rows[0]["per_sample_wall_clock_us_max"] == 4.0

    def test_validate_success_artifact_failure_modes(self, tmp_path: Path) -> None:
        payload = exp.build_problem_payload()
        artifact = exp.build_success_artifact(
            preconditions_checked=_fake_preconditions(),
            uptime="up",
            overlay_loaded="carnot_ising_v2_n64",
            overlay_load_command=exp.OVERLAY_LOAD_COMMAND,
            uio_devices_present=["/dev/uio0"],
            bitstream_sha256="a" * 64,
            problem_payload=payload,
            board_payload=_fake_board_payload(),
            duration_s=31.0,
            transcript_path=tmp_path / "a.log",
        )
        exp._validate_success_artifact(artifact)

        missing = dict(artifact)
        missing.pop("run_date")
        with pytest.raises(ValueError, match="missing"):
            exp._validate_success_artifact(missing)

        bad_overlay = dict(artifact, kv260_overlay_loaded="other")
        with pytest.raises(ValueError, match="valid Carnot overlay"):
            exp._validate_success_artifact(bad_overlay)

        bad_seed_count = dict(artifact, per_seed_results=artifact["per_seed_results"][:2])
        with pytest.raises(ValueError, match="exactly three"):
            exp._validate_success_artifact(bad_seed_count)

        bad_median = dict(artifact)
        bad_median["per_seed_results"] = [dict(row) for row in artifact["per_seed_results"]]
        bad_median["per_seed_results"][0]["per_sample_wall_clock_us_median"] = 0.0
        with pytest.raises(ValueError, match="positive"):
            exp._validate_success_artifact(bad_median)

        too_short = dict(artifact, duration_s=29.0)
        with pytest.raises(ValueError, match="30s"):
            exp._validate_success_artifact(too_short)

        no_bitstream = dict(artifact, bitstream_sha256=None)
        with pytest.raises(ValueError, match="bitstream"):
            exp._validate_success_artifact(no_bitstream)


class TestBoardHarnessSource:
    def test_board_harness_uses_uio_mmap_and_documented_registers(self) -> None:
        source = exp.BOARD_HARNESS_SOURCE
        assert '"/dev/uio0"' in source
        assert "mmap.mmap" in source
        assert "ADDR_CONTROL = 0x0000" in source
        assert "ADDR_STATUS = 0x0004" in source
        assert "ADDR_BIAS_BASE = 0x1000" in source
        assert "ADDR_ADJ_BASE = 0x2000" in source
        assert "ADDR_COUPL_BASE = 0x6000" in source
        assert "ADDR_SPOUT_BASE = 0xA010" in source
        assert "STATUS_DONE_MASK = 0x4" in source


class FakeTranscript:
    def __init__(self) -> None:
        self.records: list[tuple[str, exp.CommandResult]] = []

    def record_result(self, label: str, result: exp.CommandResult) -> None:
        self.records.append((label, result))


class TestCommandHelpers:
    def test_transcript_records_stdout_and_stderr(self, tmp_path: Path) -> None:
        transcript = exp.Transcript(tmp_path / "transcript.log")
        transcript.write("line with newline\n")
        transcript.record_result(
            "demo",
            exp.CommandResult(
                cmd=["ssh", "kria", "true"],
                returncode=1,
                stdout="out\n",
                stderr="err\n",
                duration_s=0.5,
            ),
        )
        transcript.record_result(
            "quiet",
            exp.CommandResult(
                cmd=["true"],
                returncode=0,
                stdout="",
                stderr="",
                duration_s=0.1,
            ),
        )
        body = transcript.path.read_text()
        assert "$ demo:" in body
        assert "$ quiet:" in body
        assert "[stdout]" in body
        assert "[stderr]" in body

    def test_write_json_and_repo_relative_path(self, tmp_path: Path) -> None:
        out = tmp_path / "artifact.json"
        exp._write_json(out, {"b": 2, "a": 1})
        assert out.read_text().startswith("{\n")
        assert exp._path_for_artifact(exp.TRANSCRIPT_PATH) == "results/experiment_2898_kv260_transcript.log"

    def test_run_success_timeout_and_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def success(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(exp.subprocess, "run", success)
        assert exp._run(["true"], timeout=1).stdout == "ok"

        def timeout_text(*args: Any, **kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=1, output="partial", stderr="late")

        monkeypatch.setattr(exp.subprocess, "run", timeout_text)
        timed = exp._run(["ssh"], timeout=1)
        assert timed.returncode == 124
        assert timed.stdout == "partial"
        assert timed.stderr == "late"

        def timeout_bytes(*args: Any, **kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=2, output=b"x")

        monkeypatch.setattr(exp.subprocess, "run", timeout_bytes)
        assert "timeout after 2s" in exp._run(["ssh"], timeout=2).stderr

        def os_error(*args: Any, **kwargs: Any) -> None:
            raise OSError("missing")

        monkeypatch.setattr(exp.subprocess, "run", os_error)
        failed = exp._run(["missing"], timeout=1)
        assert failed.returncode == 127
        assert "OSError" in failed.stderr

    def test_ssh_and_scp_build_expected_commands(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], timeout: int | float) -> exp.CommandResult:
            calls.append(cmd)
            return exp.CommandResult(cmd=cmd, returncode=0, stdout="", stderr="", duration_s=0.0)

        monkeypatch.setattr(exp, "_run", fake_run)
        exp._ssh("true", batch_mode=True)
        exp._ssh("true", batch_mode=False)
        exp._scp(Path("local.json"), "/tmp/remote.json")
        assert calls[0][:5] == ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]
        assert calls[1][:3] == ["ssh", "-o", "ConnectTimeout=5"]
        assert calls[2][0] == "scp"


class TestPreconditionsAndProvenance:
    def test_overlay_detection_and_parse_helpers(self) -> None:
        assert exp._detect_overlay("carnot_ising_v4 available") == "carnot_ising_v4"
        assert exp._detect_overlay("none") is None
        sha = "A" * 64
        parsed, path = exp._parse_sha256sum(f"{sha}  /tmp/a.bit\n")
        assert parsed == sha.lower()
        assert path == "/tmp/a.bit"
        assert exp._parse_sha256sum("not a checksum") == (None, None)
        assert exp._precondition("r", 1, "d") == {"resource": "r", "available": True, "detail": "d"}

    def test_preconditions_block_ssh_overlay_and_uio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transcript = FakeTranscript()
        monkeypatch.setattr(
            exp,
            "_run",
            lambda cmd, timeout: exp.CommandResult(cmd, 255, "", "no route", 0.1),
        )
        blocked, preconditions, _ = exp.check_preconditions_and_load_overlay(transcript)
        assert blocked == "blocked_kv260_ssh_unreachable"
        assert preconditions[0]["available"] is False

        transcript = FakeTranscript()
        monkeypatch.setattr(
            exp,
            "_run",
            lambda cmd, timeout: exp.CommandResult(cmd, 0, "", "", 0.1),
        )
        monkeypatch.setattr(
            exp,
            "_ssh",
            lambda remote_cmd, timeout=30, batch_mode=False: exp.CommandResult(
                ["ssh"], 0, "no overlay", "", 0.1
            ),
        )
        blocked, preconditions, _ = exp.check_preconditions_and_load_overlay(transcript)
        assert blocked == "blocked_kv260_overlay_missing"
        assert preconditions[1]["resource"] == "kv260_overlay"

        transcript = FakeTranscript()
        responses = iter(
            [
                exp.CommandResult(["ssh"], 0, "carnot_ising_v2_n64", "", 0.1),
                exp.CommandResult(["ssh"], 0, "loaded", "", 0.1),
                exp.CommandResult(["ssh"], 0, "carnot_ising_v2_n64", "", 0.1),
                exp.CommandResult(["ssh"], 2, "", "missing", 0.1),
            ]
        )
        monkeypatch.setattr(exp, "_ssh", lambda *args, **kwargs: next(responses))
        blocked, preconditions, details = exp.check_preconditions_and_load_overlay(transcript)
        assert blocked == "blocked_kv260_uio_devices_absent"
        assert details["loaded_overlay"] == "carnot_ising_v2_n64"

    def test_preconditions_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transcript = FakeTranscript()
        monkeypatch.setattr(
            exp,
            "_run",
            lambda cmd, timeout: exp.CommandResult(cmd, 0, "", "", 0.1),
        )
        responses = iter(
            [
                exp.CommandResult(["ssh"], 0, "carnot_ising_v4", "", 0.1),
                exp.CommandResult(["ssh"], 0, "carnot_ising_v2_n64: Loaded", "", 0.1),
                exp.CommandResult(["ssh"], 0, "carnot_ising_v2_n64", "", 0.1),
                exp.CommandResult(["ssh"], 0, "/dev/uio0\nok\n", "", 0.1),
            ]
        )
        monkeypatch.setattr(exp, "_ssh", lambda *args, **kwargs: next(responses))
        blocked, preconditions, details = exp.check_preconditions_and_load_overlay(transcript)
        assert blocked is None
        assert all(item["available"] for item in preconditions)
        assert details["loaded_overlay"] == "carnot_ising_v2_n64"

    def test_collect_board_provenance_with_bit_and_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transcript = FakeTranscript()
        sha = "b" * 64

        def with_bit(remote_cmd: str, timeout: int | float = 30, batch_mode: bool = False):
            if remote_cmd == "uptime":
                return exp.CommandResult(["ssh"], 0, "up now\n", "", 0.1)
            if remote_cmd.startswith("ls /dev/uio"):
                return exp.CommandResult(["ssh"], 0, "/dev/uio0\n/dev/uio4\n", "", 0.1)
            return exp.CommandResult(["ssh"], 0, f"{sha}  /lib/firmware/xilinx/carnot_ising_v4/a.bit\n", "", 0.1)

        monkeypatch.setattr(exp, "_ssh", with_bit)
        provenance = exp.collect_board_provenance(transcript)
        assert provenance["bitstream_sha256"] == sha
        assert provenance["uio_devices"] == ["/dev/uio0", "/dev/uio4"]

        transcript = FakeTranscript()
        calls: list[str] = []

        def with_fallback(remote_cmd: str, timeout: int | float = 30, batch_mode: bool = False):
            calls.append(remote_cmd)
            if remote_cmd == "uptime":
                return exp.CommandResult(["ssh"], 0, "up\n", "", 0.1)
            if remote_cmd.startswith("ls /dev/uio"):
                return exp.CommandResult(["ssh"], 0, "/dev/uio0\n", "", 0.1)
            if "*.bit.bin" in remote_cmd:
                return exp.CommandResult(["ssh"], 0, f"{sha}  /tmp/a.bit.bin\n", "", 0.1)
            return exp.CommandResult(["ssh"], 0, "", "", 0.1)

        monkeypatch.setattr(exp, "_ssh", with_fallback)
        provenance = exp.collect_board_provenance(transcript)
        assert provenance["bitstream_path"] == "/tmp/a.bit.bin"
        assert any("*.bit.bin" in call for call in calls)

    def test_extract_board_json_success_and_failure(self) -> None:
        assert exp._extract_board_json("noise\n{\"ok\": true}\n") == {"ok": True}
        with pytest.raises(ValueError, match="final JSON"):
            exp._extract_board_json("noise only")


class TestBoardHarnessRunner:
    def test_run_board_harness_success_and_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(exp, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(exp, "LOCAL_PROBLEM_PATH", tmp_path / "problem.json")
        monkeypatch.setattr(exp, "LOCAL_HARNESS_PATH", tmp_path / "harness.py")
        transcript = FakeTranscript()
        payload = {"problems": [], "n_sample_counts": []}

        scp_calls = []

        def good_scp(local: Path, remote: str, timeout: int | float = 60):
            scp_calls.append((local, remote))
            return exp.CommandResult(["scp"], 0, "", "", 0.1)

        monkeypatch.setattr(exp, "_scp", good_scp)
        monkeypatch.setattr(
            exp,
            "_ssh",
            lambda *args, **kwargs: exp.CommandResult(["ssh"], 0, "log\n{\"runs\": []}\n", "", 0.1),
        )
        assert exp.run_board_harness(payload, transcript) == {"runs": []}
        assert len(scp_calls) == 2

        monkeypatch.setattr(
            exp,
            "_scp",
            lambda *args, **kwargs: exp.CommandResult(["scp"], 1, "", "copy failed", 0.1),
        )
        with pytest.raises(RuntimeError, match="problem scp failed"):
            exp.run_board_harness(payload, transcript)

        calls = iter(
            [
                exp.CommandResult(["scp"], 0, "", "", 0.1),
                exp.CommandResult(["scp"], 1, "", "copy failed", 0.1),
            ]
        )
        monkeypatch.setattr(exp, "_scp", lambda *args, **kwargs: next(calls))
        with pytest.raises(RuntimeError, match="harness scp failed"):
            exp.run_board_harness(payload, transcript)

        calls = iter(
            [
                exp.CommandResult(["scp"], 0, "", "", 0.1),
                exp.CommandResult(["scp"], 0, "", "", 0.1),
            ]
        )
        monkeypatch.setattr(exp, "_scp", lambda *args, **kwargs: next(calls))
        monkeypatch.setattr(
            exp,
            "_ssh",
            lambda *args, **kwargs: exp.CommandResult(["ssh"], 2, "", "run failed", 0.1),
        )
        with pytest.raises(RuntimeError, match="board harness failed"):
            exp.run_board_harness(payload, transcript)


class TestRunExperimentAndMain:
    def test_run_experiment_blocked_and_sha_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(exp, "TRANSCRIPT_PATH", tmp_path / "transcript.log")
        monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / "result.json")
        monkeypatch.setattr(
            exp,
            "check_preconditions_and_load_overlay",
            lambda transcript: (
                "blocked_kv260_ssh_unreachable",
                [{"resource": "kv260_ssh", "available": False, "detail": "no"}],
                {},
            ),
        )
        artifact = exp.run_experiment()
        assert artifact["honest_verdict"] == "blocked_kv260_ssh_unreachable"
        assert (tmp_path / "result.json").exists()

        monkeypatch.setattr(
            exp,
            "check_preconditions_and_load_overlay",
            lambda transcript: (None, _fake_preconditions(), {"loaded_overlay": "carnot_ising_v4"}),
        )
        monkeypatch.setattr(
            exp,
            "collect_board_provenance",
            lambda transcript: {"uptime": "up", "uio_devices": ["/dev/uio0"], "bitstream_sha256": None},
        )
        artifact = exp.run_experiment()
        assert artifact["honest_verdict"] == "blocked_kv260_bitstream_sha256_missing"
        assert artifact["kv260_overlay_loaded"] == "carnot_ising_v4"

    def test_run_experiment_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(exp, "TRANSCRIPT_PATH", tmp_path / "transcript.log")
        monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / "result.json")
        monkeypatch.setattr(
            exp,
            "check_preconditions_and_load_overlay",
            lambda transcript: (
                None,
                _fake_preconditions(),
                {"loaded_overlay": "carnot_ising_v2_n64"},
            ),
        )
        monkeypatch.setattr(
            exp,
            "collect_board_provenance",
            lambda transcript: {
                "uptime": "up",
                "uio_devices": ["/dev/uio0"],
                "bitstream_sha256": "c" * 64,
            },
        )
        monkeypatch.setattr(exp, "run_board_harness", lambda payload, transcript: _fake_board_payload())
        ticks = iter([0.0, 31.0])
        monkeypatch.setattr(exp.time, "perf_counter", lambda: next(ticks))
        artifact = exp.run_experiment()
        assert artifact["honest_verdict"].startswith("complete:")
        assert len(artifact["per_seed_results"]) == 3

    def test_main_outputs_summary_and_result_path(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            exp,
            "run_experiment",
            lambda: {"honest_verdict": "complete: ok"},
        )
        assert exp.main([]) == 0
        assert "complete: ok" in capsys.readouterr().out
        assert exp.main(["--print-result-path"]) == 0
        assert str(exp.RESULT_PATH) in capsys.readouterr().out
