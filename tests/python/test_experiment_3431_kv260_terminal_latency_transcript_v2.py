"""Tests for exp3431 KV260 terminal latency transcript (v2 re-attempt).

Covers REQ-HW-060 (SCENARIO-HW-060) and REQ-HW-061 (SCENARIO-HW-061).

The v2 module reuses the audited v1 hardware logic verbatim and adds only:
  * v2-specific output paths, and
  * a provenance relabel (experiment_id/experiment/run_date/transcript path).

These tests drive the experiment WITHOUT an attached KV260 by injecting a fake
command executor. They lock down ONLY the code v2 adds:
  * the v1 module loads and is reused,
  * the honest blocked path when the board is unreachable over SSH
    (SCENARIO-HW-060 negative branch -- no fabrication), with v2 provenance, and
  * the terminal success path that graduates the board (SCENARIO-HW-061
    positive branch), with v2 provenance, latencies, and schema validity.
"""

from __future__ import annotations

import json
from pathlib import Path

# scripts/ is a package, so import the module by its package path. This lets
# coverage trace the v2 script cleanly (importlib-by-file-path defeats the
# coverage source matcher).
import scripts.experiment_3431_kv260_terminal_latency_transcript_v2 as module

v1 = module.v1


def _cmd_result(returncode=0, stdout="", stderr=""):
    return v1.CommandResult(
        cmd=["fake"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_s=0.01,
    )


class FakeBoard:
    """Programmable executor answering KV260 commands by substring match."""

    def __init__(self, *, ssh_ok=True):
        self.ssh_ok = ssh_ok
        self.calls: list[str] = []

    def __call__(self, cmd, timeout):
        joined = " ".join(cmd)
        self.calls.append(joined)
        is_scp = cmd[0] == "scp"
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return _cmd_result(
                0 if self.ssh_ok else 255,
                stderr="" if self.ssh_ok else "no route to host",
            )
        if "listapps" in joined:
            return _cmd_result(0, stdout="carnot_ising_v2_n64 (loaded)")
        if "loadapp" in joined:
            return _cmd_result(0, stdout="carnot_ising_v2_n64 loaded")
        if "ls /dev/uio0" in joined:
            return _cmd_result(0, stdout="/dev/uio0\nok")
        if joined.endswith("uptime"):
            return _cmd_result(0, stdout="up 9 days")
        if "ls /dev/uio*" in joined:
            return _cmd_result(0, stdout="/dev/uio0\n/dev/uio1")
        if "sha256sum" in joined and ".bit.bin" in joined:
            return _cmd_result(
                0,
                stdout=("b" * 64) + " /lib/firmware/xilinx/carnot_ising_v4/x.bit.bin",
            )
        if "sha256sum" in joined:
            return _cmd_result(
                0, stdout=("a" * 64) + " /lib/firmware/xilinx/carnot_ising_v4/x.bit"
            )
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
                        "n_samples": v1.HEADLINE_SAMPLE_COUNT,
                        "latencies_us": [3.0 + (i % 5) * 0.1 for i in range(20)],
                        "final_energy": -42.0,
                        "final_spin_words_hex": ["0x1", "0x2"],
                        "failed_samples": 0,
                    }
                    for seed in v1.RANDOM_SEEDS
                ],
            }
            return _cmd_result(0, stdout="log line\n" + json.dumps(board_out))
        return _cmd_result(0, stdout="")


# ---------------------------------------------------------------------------
# v2 module wiring
# ---------------------------------------------------------------------------


def test_v2_reuses_v1_module():
    """The v2 module imports and reuses the audited v1 implementation."""
    assert module.EXPERIMENT_ID == 3431
    assert module.RUN_DATE == "20260530"
    assert hasattr(v1, "run_experiment")
    assert v1.EXPERIMENT_ID == 3420


def test_relabel_for_v2_only_touches_identity():
    """relabel_for_v2 rewrites identity labels but preserves measured fields."""
    artifact = v1.build_blocked_artifact(
        verdict="blocked_kv260_ssh_unreachable",
        preconditions_checked=[{"resource": "kv260_ssh", "available": False}],
        duration_s=0.5,
        transcript_path=Path("/tmp/x.log"),
    )
    measured_verdict = artifact["honest_verdict"]
    relabeled = module.relabel_for_v2(artifact)
    assert relabeled["experiment_id"] == 3431
    assert relabeled["experiment"] == module.EXPERIMENT_NAME
    assert relabeled["run_date"] == "20260530"
    # Measured fields untouched.
    assert relabeled["honest_verdict"] == measured_verdict
    assert relabeled["inference_substrate"] == "hardware_smoke"
    assert "experiment_3431_kv260_terminal_transcript_v2.log" in (
        relabeled["board_transcript_path"]
    )


# ---------------------------------------------------------------------------
# End-to-end flow via injected executor (SCENARIO-HW-060 / SCENARIO-HW-061)
# ---------------------------------------------------------------------------


def test_run_experiment_blocked_when_ssh_unreachable(tmp_path):
    """SCENARIO-HW-060: unreachable board -> honest blocked, v2 provenance."""
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
    assert artifact["experiment_id"] == 3431
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert set(v1.REQUIRED_ARTIFACT_FIELDS).issubset(artifact.keys())
    # Persisted to disk and re-validatable.
    assert result_path.exists()
    on_disk = json.loads(result_path.read_text())
    assert on_disk["experiment_id"] == 3431
    v1.validate_artifact(on_disk)


def _success_board_payload():
    return {
        "duration_s": 2.5,
        "selected_uio": "/dev/uio0",
        "selected_uio_addr_hex": "0xa0000000",
        "runs": [
            {
                "seed": seed,
                "n_samples": v1.HEADLINE_SAMPLE_COUNT,
                "latencies_us": [3.0 + (i % 4) * 0.2 for i in range(25)],
                "failed_samples": 0,
            }
            for seed in v1.RANDOM_SEEDS
        ],
    }


def test_relabel_on_success_artifact_preserves_terminal_state(tmp_path):
    """SCENARIO-HW-061: a terminal transcript relabels to v2 and stays valid.

    Built directly (duration_s=120 like the v1 success test) so the
    hardware-smoke duration floor is honored; the fake-executor end-to-end run
    completes instantly and would trip that floor -- which is the v1 logic's
    fabrication guard, not a v2 concern.
    """
    artifact = v1.build_success_artifact(
        preconditions_checked=[{"resource": "kv260_ssh", "available": True}],
        uptime="up 9 days",
        overlay_loaded="carnot_ising_v2_n64",
        uio_devices_present=["/dev/uio0"],
        bitstream_sha256="a" * 64,
        problem_payload=v1.build_problem_payload(),
        board_payload=_success_board_payload(),
        duration_s=120.0,
        transcript_path=tmp_path / "t.log",
        transcript_text="$ precondition_ssh ...\nrc=0\n",
    )
    relabeled = module.relabel_for_v2(artifact)
    assert relabeled["honest_verdict"].startswith("complete:")
    assert relabeled["kv260_terminal_state_reached"] is True
    assert relabeled["kv260_synthesis_succeeded"] is True
    assert relabeled["per_iteration_latency_us"]
    assert relabeled["kv260_latency_transcript"]["stats"]["mean_us"] > 0
    assert relabeled["experiment_id"] == 3431
    assert relabeled["experiment"] == module.EXPERIMENT_NAME
    v1.validate_artifact(relabeled)


def test_main_runs_with_default_real_paths(monkeypatch, capsys, tmp_path):
    """main() drives run_experiment and prints the verdict summary."""
    monkeypatch.setattr(module, "RESULT_PATH", tmp_path / "r.json")
    monkeypatch.setattr(module, "TRANSCRIPT_PATH", tmp_path / "t.log")

    def fake_run_experiment():
        return {
            "honest_verdict": "blocked_kv260_ssh_unreachable",
            "kv260_terminal_state_reached": False,
        }

    monkeypatch.setattr(module, "run_experiment", fake_run_experiment)
    rc = module.main([])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert out["kv260_terminal_state_reached"] is False


def test_main_print_result_path(monkeypatch, capsys):
    monkeypatch.setattr(
        module, "run_experiment", lambda: {"honest_verdict": "x", "kv260_terminal_state_reached": False}
    )
    rc = module.main(["--print-result-path"])
    assert rc == 0
    assert "experiment_3431_kv260_terminal_latency_transcript_v2.json" in (
        capsys.readouterr().out
    )
