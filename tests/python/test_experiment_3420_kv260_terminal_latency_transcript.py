"""Tests for exp3420 KV260 terminal latency transcript.

Covers REQ-HW-060 (SCENARIO-HW-060) and REQ-HW-061 (SCENARIO-HW-061).

These tests drive the experiment WITHOUT an attached KV260 by injecting a fake
command executor. They lock down:
  * the deterministic Ising problem shaping and Q8.8 fixed-point clamping,
  * the latency-stat reduction (mean/p50/p99),
  * the honest blocked path when the board is unreachable over SSH
    (SCENARIO-HW-060 / SCENARIO-HW-061 negative branch — no fabrication), and
  * the terminal success path that graduates the board
    (SCENARIO-HW-061 positive branch).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "experiment_3420_kv260_terminal_latency_transcript_v1.py"
)

_spec = importlib.util.spec_from_file_location("exp3420", SCRIPT_PATH)
assert _spec is not None
module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Register the dynamically-loaded module in sys.modules BEFORE exec_module.
# Python 3.14's @dataclass resolves field types via
# sys.modules.get(cls.__module__).__dict__; if the module name ("exp3420") is
# absent from sys.modules that lookup returns None and dataclass creation raises
# AttributeError at import time, breaking pytest collection of this file (the
# 2026-05-31 kv260 collection error). Registering first is the standard
# spec_from_file_location + exec_module pattern for modules with dataclasses.
sys.modules[_spec.name] = module
_spec.loader.exec_module(module)


def _cmd_result(returncode=0, stdout="", stderr=""):
    return module.CommandResult(
        cmd=["fake"], returncode=returncode, stdout=stdout, stderr=stderr, duration_s=0.01
    )


class FakeBoard:
    """Programmable executor that answers KV260 commands by substring match."""

    def __init__(
        self, *, ssh_ok=True, overlay_ok=True, uio_ok=True, bitstream_ok=True, bitbin_ok=True
    ):
        self.ssh_ok = ssh_ok
        self.overlay_ok = overlay_ok
        self.uio_ok = uio_ok
        self.bitstream_ok = bitstream_ok
        self.bitbin_ok = bitbin_ok
        self.calls: list[str] = []

    def __call__(self, cmd, timeout):
        joined = " ".join(cmd)
        self.calls.append(joined)
        is_scp = cmd[0] == "scp"
        remote = cmd[-1] if cmd else ""
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return _cmd_result(0 if self.ssh_ok else 255, stderr="" if self.ssh_ok else "no route")
        if "listapps" in joined:
            return _cmd_result(0, stdout="carnot_ising_v2_n64 (loaded)") if self.overlay_ok else _cmd_result(0, stdout="none")
        if joined.endswith("loadapp carnot_ising_v2_n64") or "loadapp" in remote:
            return _cmd_result(0, stdout="carnot_ising_v2_n64 loaded")
        if "ls /dev/uio0" in joined:
            return _cmd_result(0, stdout="/dev/uio0\nok") if self.uio_ok else _cmd_result(0, stdout="")
        if joined.endswith("uptime"):
            return _cmd_result(0, stdout="up 5 days")
        if "ls /dev/uio*" in joined:
            return _cmd_result(0, stdout="/dev/uio0\n/dev/uio1")
        if "sha256sum" in joined and ".bit.bin" in joined:
            if self.bitbin_ok:
                return _cmd_result(0, stdout=("b" * 64) + " /lib/firmware/xilinx/carnot_ising_v4/x.bit.bin")
            return _cmd_result(0, stdout="")
        if "sha256sum" in joined:
            if self.bitstream_ok:
                return _cmd_result(0, stdout=("a" * 64) + " /lib/firmware/xilinx/carnot_ising_v4/x.bit")
            return _cmd_result(0, stdout="")
        if is_scp:
            return _cmd_result(0)
        if "board_harness.py" in joined:
            board_out = {
                "duration_s": 2.5,
                "selected_uio": "/dev/uio0",
                "selected_uio_addr_hex": "0xa0000000",
                "uio_devices": ["/dev/uio0"],
                "runs": [
                    {
                        "seed": seed,
                        "n_samples": module.HEADLINE_SAMPLE_COUNT,
                        "latencies_us": [3.0 + (i % 5) * 0.1 for i in range(20)],
                        "final_energy": -42.0,
                        "final_spin_words_hex": ["0x1", "0x2"],
                        "failed_samples": 0,
                    }
                    for seed in module.RANDOM_SEEDS
                ],
            }
            return _cmd_result(0, stdout="log line\n" + json.dumps(board_out))
        return _cmd_result(0, stdout="")


# ---------------------------------------------------------------------------
# Pure data-shaping (REQ-HW-060)
# ---------------------------------------------------------------------------


def test_generate_ising_problem_is_deterministic_and_symmetric():
    """REQ-HW-060: same seed -> identical symmetric J, zero field/diagonal."""
    a = module.generate_ising_problem(7)
    b = module.generate_ising_problem(7)
    assert a["j_matrix"] == b["j_matrix"]
    assert a["n_spins"] == 64
    j = a["j_matrix"]
    for i in range(a["n_spins"]):
        assert j[i][i] == 0.0
        assert a["h_vector"][i] == 0.0
        for k in range(i + 1, a["n_spins"]):
            assert j[i][k] == j[k][i]


def test_q88_clamps_to_int16():
    """REQ-HW-060: Q8.8 fixed-point clamps to signed 16-bit range."""
    assert module.q88(0.0) == 0
    assert module.q88(1.0) == 256
    assert module.q88(1000.0) == 32767
    assert module.q88(-1000.0) == -32768


def test_build_sparse_upload_respects_max_degree():
    """REQ-HW-060: sparse upload keeps at most MAX_DEGREE couplings per row."""
    problem = module.generate_ising_problem(3)
    upload = module.build_sparse_upload(problem, max_degree=8)
    assert upload["max_degree"] == 8
    assert len(upload["adjacency"]) == problem["n_spins"]
    for row in upload["adjacency"]:
        assert len(row) <= 8
    for row in upload["couplings_q88"]:
        assert all(-32768 <= c <= 32767 for c in row)


def test_problem_spec_and_payload_have_hashes():
    """REQ-HW-060: provenance hashes + payload carry seeds and headline count."""
    spec_obj = module.problem_spec(module.generate_ising_problem(5))
    assert len(spec_obj["j_matrix_sha256"]) == 64
    assert len(spec_obj["h_vector_sha256"]) == 64
    assert spec_obj["random_seed"] == 5
    payload = module.build_problem_payload()
    assert payload["random_seeds_used"] == module.RANDOM_SEEDS
    assert payload["headline_sample_count"] == module.HEADLINE_SAMPLE_COUNT
    assert len(payload["problems"]) == len(module.RANDOM_SEEDS)


# ---------------------------------------------------------------------------
# Latency statistics (REQ-HW-061)
# ---------------------------------------------------------------------------


def test_compute_latency_stats_values():
    """REQ-HW-061: mean/p50/p99 reduction is correct on a known list."""
    stats = module.compute_latency_stats([10.0, 20.0, 30.0, 40.0])
    assert stats["n_iterations"] == 4
    assert stats["mean_us"] == 25.0
    assert stats["min_us"] == 10.0
    assert stats["max_us"] == 40.0
    assert stats["p50_us"] == 20.0
    assert stats["p99_us"] == 40.0


def test_compute_latency_stats_empty_raises():
    """REQ-HW-061: zero iterations is an error, never a fabricated zero."""
    with pytest.raises(ValueError):
        module.compute_latency_stats([])
    with pytest.raises(ValueError):
        module._percentile([], 50.0)


# ---------------------------------------------------------------------------
# Artifact construction + validation
# ---------------------------------------------------------------------------


def test_blocked_artifact_full_schema_and_not_terminal():
    """SCENARIO-HW-061 negative: blocked artifact carries full schema."""
    blocked = module.build_blocked_artifact(
        verdict="blocked_kv260_ssh_unreachable",
        preconditions_checked=[{"resource": "kv260_ssh", "available": False}],
        duration_s=0.5,
        transcript_path=Path("/tmp/x.log"),
    )
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["kv260_terminal_state_reached"] is False
    assert blocked["kv260_synthesis_succeeded"] is False
    assert blocked["inference_substrate"] == "hardware_smoke"
    assert set(module.REQUIRED_ARTIFACT_FIELDS).issubset(blocked.keys())
    module.validate_artifact(blocked)


def test_validate_rejects_blocked_without_prefix():
    blocked = module.build_blocked_artifact(
        verdict="oops_no_prefix",
        preconditions_checked=[],
        duration_s=0.5,
        transcript_path=Path("/tmp/x.log"),
    )
    with pytest.raises(ValueError):
        module.validate_artifact(blocked)


def test_validate_rejects_wrong_substrate():
    blocked = module.build_blocked_artifact(
        verdict="blocked_kv260_ssh_unreachable",
        preconditions_checked=[],
        duration_s=0.5,
        transcript_path=Path("/tmp/x.log"),
    )
    blocked["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError):
        module.validate_artifact(blocked)


def test_validate_rejects_missing_field():
    blocked = module.build_blocked_artifact(
        verdict="blocked_kv260_ssh_unreachable",
        preconditions_checked=[],
        duration_s=0.5,
        transcript_path=Path("/tmp/x.log"),
    )
    del blocked["duration_s"]
    with pytest.raises(ValueError):
        module.validate_artifact(blocked)


def _success_board_payload():
    return {
        "duration_s": 2.5,
        "selected_uio": "/dev/uio0",
        "selected_uio_addr_hex": "0xa0000000",
        "runs": [
            {
                "seed": seed,
                "n_samples": module.HEADLINE_SAMPLE_COUNT,
                "latencies_us": [3.0 + (i % 4) * 0.2 for i in range(25)],
                "failed_samples": 1,
            }
            for seed in module.RANDOM_SEEDS
        ],
    }


def test_success_artifact_graduates_board():
    """SCENARIO-HW-061 positive: success artifact records transcript + graduates."""
    artifact = module.build_success_artifact(
        preconditions_checked=[{"resource": "kv260_ssh", "available": True}],
        uptime="up 5 days",
        overlay_loaded="carnot_ising_v2_n64",
        uio_devices_present=["/dev/uio0"],
        bitstream_sha256="a" * 64,
        problem_payload=module.build_problem_payload(),
        board_payload=_success_board_payload(),
        duration_s=120.0,
        transcript_path=Path("/tmp/x.log"),
        transcript_text="$ precondition_ssh ...\nrc=0\n",
    )
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["kv260_terminal_state_reached"] is True
    assert artifact["kv260_synthesis_succeeded"] is True
    assert artifact["kv260_latency_transcript"]["stats"]["mean_us"] > 0
    assert artifact["per_iteration_latency_us"]
    module.validate_artifact(artifact)


def test_validate_rejects_terminal_without_bitstream():
    artifact = module.build_success_artifact(
        preconditions_checked=[],
        uptime="up 5 days",
        overlay_loaded="carnot_ising_v2_n64",
        uio_devices_present=["/dev/uio0"],
        bitstream_sha256="a" * 64,
        problem_payload=module.build_problem_payload(),
        board_payload=_success_board_payload(),
        duration_s=120.0,
        transcript_path=Path("/tmp/x.log"),
        transcript_text="x",
    )
    artifact["bitstream_sha256"] = ""
    with pytest.raises(ValueError):
        module.validate_artifact(artifact)


def test_extract_board_json_requires_object():
    with pytest.raises(ValueError):
        module._extract_board_json("no json here\njust text")
    assert module._extract_board_json('noise\n{"a": 1}')["a"] == 1


def test_parse_sha256sum():
    sha, path = module._parse_sha256sum(("c" * 64) + "  /firmware/x.bit")
    assert sha == "c" * 64
    assert path == "/firmware/x.bit"
    assert module._parse_sha256sum("garbage") == (None, None)


def test_detect_overlay():
    assert module._detect_overlay("foo carnot_ising_v4 bar") == "carnot_ising_v4"
    assert module._detect_overlay("nothing here") is None


# ---------------------------------------------------------------------------
# End-to-end flow via injected executor
# ---------------------------------------------------------------------------


def test_run_experiment_blocked_when_ssh_unreachable(tmp_path):
    """SCENARIO-HW-060/061: unreachable board -> honest blocked, no fabrication."""
    result_path = tmp_path / "r.json"
    transcript_path = tmp_path / "t.log"
    artifact = module.run_experiment(
        FakeBoard(ssh_ok=False),
        result_path=result_path,
        transcript_path=transcript_path,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert artifact["kv260_terminal_state_reached"] is False
    assert artifact["kv260_latency_transcript"] is None
    assert result_path.exists()


def test_run_experiment_blocked_overlay_missing(tmp_path):
    artifact = module.run_experiment(
        FakeBoard(overlay_ok=False),
        result_path=tmp_path / "r.json",
        transcript_path=tmp_path / "t.log",
    )
    assert artifact["honest_verdict"] == "blocked_kv260_overlay_missing"


def test_run_experiment_blocked_uio_absent(tmp_path):
    artifact = module.run_experiment(
        FakeBoard(uio_ok=False),
        result_path=tmp_path / "r.json",
        transcript_path=tmp_path / "t.log",
    )
    assert artifact["honest_verdict"] == "blocked_kv260_uio_devices_absent"


def test_run_experiment_blocked_bitstream_missing(tmp_path, monkeypatch):
    # The 1.0s hardware-smoke duration floor is a fabrication guard for REAL
    # KV260 runs; a deterministic FakeBoard unit test legitimately completes in
    # <1s, so the floor is inapplicable here (mock the dependency, test the
    # logic — CLAUDE.md "Tests Must Run and Assert").
    monkeypatch.setattr(module, "DURATION_FLOOR_S", 0.0)
    artifact = module.run_experiment(
        FakeBoard(bitstream_ok=False),
        result_path=tmp_path / "r.json",
        transcript_path=tmp_path / "t.log",
    )
    # bitstream .bit missing falls back to .bit.bin in FakeBoard, so this is a
    # success; assert the fallback path produced a terminal artifact.
    assert artifact["kv260_terminal_state_reached"] is True
    assert artifact["bitstream_sha256"] == "b" * 64


def test_run_experiment_blocked_when_no_bitstream_anywhere(tmp_path):
    """Both .bit and .bit.bin absent -> honest blocked, not terminal."""
    artifact = module.run_experiment(
        FakeBoard(bitstream_ok=False, bitbin_ok=False),
        result_path=tmp_path / "r.json",
        transcript_path=tmp_path / "t.log",
    )
    assert artifact["honest_verdict"] == "blocked_kv260_bitstream_sha256_missing"
    assert artifact["kv260_terminal_state_reached"] is False
    assert artifact["kv260_ssh_uptime_at_run"] == "up 5 days"


def _valid_terminal_artifact():
    return module.build_success_artifact(
        preconditions_checked=[],
        uptime="up 5 days",
        overlay_loaded="carnot_ising_v2_n64",
        uio_devices_present=["/dev/uio0"],
        bitstream_sha256="a" * 64,
        problem_payload=module.build_problem_payload(),
        board_payload=_success_board_payload(),
        duration_s=120.0,
        transcript_path=Path("/tmp/x.log"),
        transcript_text="x",
    )


def test_validate_terminal_raises_each_guard():
    """REQ-HW-061: every terminal sanity guard rejects a malformed artifact."""
    bad_verdict = _valid_terminal_artifact()
    bad_verdict["honest_verdict"] = "partial: nope"
    with pytest.raises(ValueError):
        module.validate_artifact(bad_verdict)

    bad_overlay = _valid_terminal_artifact()
    bad_overlay["kv260_overlay_loaded"] = "not_a_real_overlay"
    with pytest.raises(ValueError):
        module.validate_artifact(bad_overlay)

    no_latencies = _valid_terminal_artifact()
    no_latencies["per_iteration_latency_us"] = []
    with pytest.raises(ValueError):
        module.validate_artifact(no_latencies)

    bad_stats = _valid_terminal_artifact()
    bad_stats["kv260_latency_transcript"]["stats"]["mean_us"] = 0.0
    with pytest.raises(ValueError):
        module.validate_artifact(bad_stats)

    short = _valid_terminal_artifact()
    short["duration_s"] = 0.001
    with pytest.raises(ValueError):
        module.validate_artifact(short)


def test_run_experiment_success_full_flow(tmp_path, monkeypatch):
    """SCENARIO-HW-061: reachable board -> terminal transcript recorded."""
    # FakeBoard runs in <1s; neutralize the real-hardware duration floor (a
    # fabrication guard that does not apply to a deterministic unit test).
    monkeypatch.setattr(module, "DURATION_FLOOR_S", 0.0)
    result_path = tmp_path / "r.json"
    artifact = module.run_experiment(
        FakeBoard(),
        result_path=result_path,
        transcript_path=tmp_path / "t.log",
    )
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["kv260_terminal_state_reached"] is True
    assert artifact["kv260_latency_transcript"]["stats"]["p99_us"] > 0
    on_disk = json.loads(result_path.read_text())
    assert on_disk["kv260_synthesis_succeeded"] is True


def test_run_board_harness_raises_on_scp_failure(tmp_path):
    transcript = module.Transcript(tmp_path / "t.log")

    def bad(cmd, timeout):
        return _cmd_result(1) if cmd[0] == "scp" else _cmd_result(0, stdout="{}")

    with pytest.raises(RuntimeError):
        module.run_board_harness(bad, module.build_problem_payload(), transcript)


def test_run_board_harness_raises_on_harness_failure(tmp_path):
    transcript = module.Transcript(tmp_path / "t.log")

    def board(cmd, timeout):
        if cmd[0] == "scp":
            return _cmd_result(0)
        return _cmd_result(2, stderr="boom")

    with pytest.raises(RuntimeError):
        module.run_board_harness(board, module.build_problem_payload(), transcript)


def test_real_run_executor_local_commands():
    """The default subprocess executor handles ok / fail / timeout / oserror."""
    assert module._real_run(["true"], 5).returncode == 0
    assert module._real_run(["false"], 5).returncode == 1
    assert module._real_run(["sleep", "5"], 0.1).returncode == 124
    assert module._real_run(["/nonexistent_binary_xyz_3420"], 5).returncode == 127


def test_transcript_record_includes_streams(tmp_path):
    t = module.Transcript(tmp_path / "t.log")
    t.record_result("label", _cmd_result(0, stdout="out", stderr="err"))
    text = t.read()
    assert "out" in text and "err" in text and "label" in text


def test_main_smoke(tmp_path, monkeypatch, capsys):
    """main() runs end-to-end against the real (unreachable) board path."""
    monkeypatch.setattr(module, "RESULT_PATH", tmp_path / "r.json")
    monkeypatch.setattr(module, "TRANSCRIPT_PATH", tmp_path / "t.log")
    # Force the unreachable path deterministically regardless of bench state.
    monkeypatch.setattr(module, "_real_run", lambda cmd, timeout: _cmd_result(255))
    assert module.main([]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    module.main(["--print-result-path"])


def test_deliverable_artifact_on_disk_is_valid():
    """The committed deliverable JSON exists and passes schema validation."""
    path = (
        REPO_ROOT
        / "results"
        / "experiment_3420_kv260_terminal_latency_transcript_v1.json"
    )
    artifact = json.loads(path.read_text())
    module.validate_artifact(artifact)
    assert set(module.REQUIRED_ARTIFACT_FIELDS).issubset(artifact.keys())
    assert artifact["inference_substrate"] == "hardware_smoke"
