"""Tests for exp3568 KV260 terminal latency transcript (v14 re-attempt).

Covers REQ-HW-060 (SCENARIO-HW-060) and REQ-HW-061 (SCENARIO-HW-061).

The v14 module reuses the audited v1 hardware logic verbatim and adds only:
  * v14-specific output paths,
  * a provenance relabel (experiment_id/experiment/run_date/transcript path),
  * three new convenience fields (kv260_ssh_reachable, board_latency_transcript,
    random_seed=20260601), and
  * a Verdict Terminal-Prefix wrap: the honest_verdict carries a ``complete:``
    prefix even on the blocked path.

These tests drive the experiment WITHOUT an attached KV260 by injecting a fake
command executor. They lock down ONLY the code v14 adds:
  * the v1 module loads and is reused,
  * the ``complete:``-prefix wrap of the honest blocked verdict
    (SCENARIO-HW-060 negative branch -- no fabrication), with v14 provenance,
  * the three new v14 convenience fields are present and correct,
  * the terminal success path that graduates the board (SCENARIO-HW-061
    positive branch), with v14 provenance, latencies, and schema validity, and
  * the v14-local ``validate_artifact`` guards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.experiment_3568_kv260_terminal_latency_transcript_v14 as module

v1 = module.v1


class FakeBoard:
    """Programmable executor: reports SSH reachability per the ssh_ok flag."""

    def __init__(self, *, ssh_ok: bool = False):
        self.ssh_ok = ssh_ok

    def __call__(self, cmd, timeout):
        return v1.CommandResult(
            cmd=cmd,
            returncode=0 if self.ssh_ok else 255,
            stdout="",
            stderr="" if self.ssh_ok else "no route to host",
            duration_s=0.01,
        )


# ---------------------------------------------------------------------------
# v14 module wiring
# ---------------------------------------------------------------------------


def test_v14_reuses_v1_module():
    """REQ-HW-060: the v14 module imports and reuses the audited v1 implementation."""
    assert module.EXPERIMENT_ID == 3568
    assert module.RUN_DATE == "20260601"
    assert module.RANDOM_SEED == 20260601
    assert hasattr(v1, "run_experiment")
    assert v1.EXPERIMENT_ID == 3420


def test_prefixed_verdict_wraps_only_non_complete():
    """Blocked verdicts gain a complete: prefix; complete: verdicts are unchanged."""
    assert (
        module._prefixed_verdict("blocked_kv260_ssh_unreachable")
        == "complete: blocked_kv260_ssh_unreachable"
    )
    assert (
        module._prefixed_verdict("complete: already terminal")
        == "complete: already terminal"
    )


def test_relabel_for_v14_identity_and_new_fields():
    """relabel_for_v14 rewrites identity fields and adds v14 convenience fields."""
    artifact = v1.build_blocked_artifact(
        verdict="blocked_kv260_ssh_unreachable",
        preconditions_checked=[{"resource": "kv260_ssh", "available": False}],
        duration_s=0.5,
        transcript_path=Path("/tmp/x.log"),
    )
    relabeled = module.relabel_for_v14(artifact)
    assert relabeled["experiment_id"] == 3568
    assert relabeled["experiment"] == module.EXPERIMENT_NAME
    assert relabeled["run_date"] == "20260601"
    assert relabeled["honest_verdict"] == "complete: blocked_kv260_ssh_unreachable"
    assert relabeled["kv260_terminal_state_reached"] is False
    assert relabeled["inference_substrate"] == "hardware_smoke"
    assert "experiment_3568_kv260_terminal_transcript_v14.log" in (
        relabeled["board_transcript_path"]
    )
    # New v14 fields
    assert relabeled["kv260_ssh_reachable"] is False
    assert relabeled["board_latency_transcript"] is None
    assert relabeled["random_seed"] == 20260601


def test_relabel_ssh_reachable_true_when_precondition_passed():
    """kv260_ssh_reachable is True when preconditions_checked records available=True."""
    artifact = v1.build_blocked_artifact(
        verdict="blocked_kv260_ssh_unreachable",
        preconditions_checked=[{"resource": "kv260_ssh", "available": True}],
        duration_s=0.5,
        transcript_path=Path("/tmp/x.log"),
    )
    relabeled = module.relabel_for_v14(artifact)
    assert relabeled["kv260_ssh_reachable"] is True


# ---------------------------------------------------------------------------
# End-to-end flow via injected executor (SCENARIO-HW-060 / SCENARIO-HW-061)
# ---------------------------------------------------------------------------


def test_run_experiment_blocked_when_ssh_unreachable(tmp_path):
    """SCENARIO-HW-060: unreachable board -> honest blocked, v14 provenance.

    The verdict carries the terminal complete: prefix, the board does NOT
    graduate (kv260_terminal_state_reached stays False), and the new v14
    fields are populated correctly.
    """
    result_path = tmp_path / "r.json"
    transcript_path = tmp_path / "t.log"
    artifact = module.run_experiment(
        FakeBoard(ssh_ok=False),
        result_path=result_path,
        transcript_path=transcript_path,
    )
    assert artifact["honest_verdict"] == "complete: blocked_kv260_ssh_unreachable"
    assert artifact["kv260_terminal_state_reached"] is False
    assert artifact["kv260_latency_transcript"] is None
    assert artifact["board_latency_transcript"] is None
    assert artifact["kv260_ssh_reachable"] is False
    assert artifact["random_seed"] == 20260601
    assert artifact["experiment_id"] == 3568
    assert artifact["inference_substrate"] == "hardware_smoke"
    required = v1.REQUIRED_ARTIFACT_FIELDS | module.V14_EXTRA_FIELDS
    assert required.issubset(artifact.keys())
    assert artifact["preconditions_checked"][0]["resource"] == "kv260_ssh"
    assert artifact["preconditions_checked"][0]["available"] is False
    assert result_path.exists()
    on_disk = json.loads(result_path.read_text())
    assert on_disk["experiment_id"] == 3568
    assert on_disk["random_seed"] == 20260601
    module.validate_artifact(on_disk)


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
    """SCENARIO-HW-061: a terminal transcript relabels to v14 and stays valid."""
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
    relabeled = module.relabel_for_v14(artifact)
    assert relabeled["honest_verdict"].startswith("complete:")
    assert relabeled["kv260_terminal_state_reached"] is True
    assert relabeled["kv260_synthesis_succeeded"] is True
    assert relabeled["per_iteration_latency_us"]
    assert relabeled["kv260_latency_transcript"]["stats"]["mean_us"] > 0
    assert relabeled["board_latency_transcript"] is not None
    assert relabeled["kv260_ssh_reachable"] is True
    assert relabeled["random_seed"] == 20260601
    assert relabeled["experiment_id"] == 3568
    assert relabeled["experiment"] == module.EXPERIMENT_NAME
    module.validate_artifact(relabeled)


# ---------------------------------------------------------------------------
# v14-local validator guards
# ---------------------------------------------------------------------------


def _minimal_blocked_v14_artifact():
    artifact = v1.build_blocked_artifact(
        verdict="blocked_kv260_ssh_unreachable",
        preconditions_checked=[{"resource": "kv260_ssh", "available": False}],
        duration_s=0.5,
        transcript_path=Path("/tmp/x.log"),
    )
    return module.relabel_for_v14(artifact)


def test_validate_rejects_missing_field():
    artifact = _minimal_blocked_v14_artifact()
    del artifact["duration_s"]
    with pytest.raises(ValueError, match="missing required fields"):
        module.validate_artifact(artifact)


def test_validate_rejects_missing_v14_field():
    artifact = _minimal_blocked_v14_artifact()
    del artifact["kv260_ssh_reachable"]
    with pytest.raises(ValueError, match="missing required fields"):
        module.validate_artifact(artifact)


def test_validate_rejects_wrong_substrate():
    artifact = _minimal_blocked_v14_artifact()
    artifact["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="hardware_smoke"):
        module.validate_artifact(artifact)


def test_validate_rejects_verdict_without_complete_prefix():
    artifact = _minimal_blocked_v14_artifact()
    artifact["honest_verdict"] = "blocked_kv260_ssh_unreachable"
    with pytest.raises(ValueError, match="complete: prefix"):
        module.validate_artifact(artifact)


def test_validate_rejects_wrong_random_seed():
    artifact = _minimal_blocked_v14_artifact()
    artifact["random_seed"] = 3535
    with pytest.raises(ValueError, match="random_seed must be"):
        module.validate_artifact(artifact)


def test_validate_rejects_non_bool_ssh_reachable():
    artifact = _minimal_blocked_v14_artifact()
    artifact["kv260_ssh_reachable"] = "yes"
    with pytest.raises(ValueError, match="kv260_ssh_reachable must be a boolean"):
        module.validate_artifact(artifact)


def test_validate_rejects_terminal_without_bitstream():
    artifact = _minimal_blocked_v14_artifact()
    artifact["kv260_terminal_state_reached"] = True
    artifact["kv260_overlay_loaded"] = "carnot_ising_v2_n64"
    artifact["bitstream_sha256"] = None
    with pytest.raises(ValueError, match="bitstream_sha256 missing"):
        module.validate_artifact(artifact)


def test_validate_rejects_terminal_with_bad_overlay():
    artifact = _minimal_blocked_v14_artifact()
    artifact["kv260_terminal_state_reached"] = True
    artifact["kv260_overlay_loaded"] = "not_a_real_overlay"
    with pytest.raises(ValueError, match="valid Carnot overlay"):
        module.validate_artifact(artifact)


def test_validate_rejects_terminal_without_latencies():
    artifact = _minimal_blocked_v14_artifact()
    artifact["kv260_terminal_state_reached"] = True
    artifact["kv260_overlay_loaded"] = "carnot_ising_v2_n64"
    artifact["bitstream_sha256"] = "a" * 64
    artifact["per_iteration_latency_us"] = []
    with pytest.raises(ValueError, match="per-iteration latencies"):
        module.validate_artifact(artifact)


def test_validate_rejects_terminal_with_nonpositive_stats():
    artifact = _minimal_blocked_v14_artifact()
    artifact["kv260_terminal_state_reached"] = True
    artifact["kv260_overlay_loaded"] = "carnot_ising_v2_n64"
    artifact["bitstream_sha256"] = "a" * 64
    artifact["per_iteration_latency_us"] = [1.0]
    artifact["kv260_latency_transcript"] = {"stats": {"mean_us": 0.0, "p99_us": 0.0}}
    with pytest.raises(ValueError, match="latency stats must be positive"):
        module.validate_artifact(artifact)


def test_validate_rejects_terminal_below_duration_floor():
    artifact = _minimal_blocked_v14_artifact()
    artifact["kv260_terminal_state_reached"] = True
    artifact["kv260_overlay_loaded"] = "carnot_ising_v2_n64"
    artifact["bitstream_sha256"] = "a" * 64
    artifact["per_iteration_latency_us"] = [3.0]
    artifact["kv260_latency_transcript"] = {"stats": {"mean_us": 3.0, "p99_us": 3.5}}
    artifact["duration_s"] = 0.001
    with pytest.raises(ValueError, match="duration_s below hardware-smoke floor"):
        module.validate_artifact(artifact)


def test_main_runs_with_default_real_paths(monkeypatch, capsys, tmp_path):
    """main() drives run_experiment and prints the verdict summary."""
    monkeypatch.setattr(module, "RESULT_PATH", tmp_path / "r.json")
    monkeypatch.setattr(module, "TRANSCRIPT_PATH", tmp_path / "t.log")

    def fake_run_experiment():
        return {
            "honest_verdict": "complete: blocked_kv260_ssh_unreachable",
            "kv260_terminal_state_reached": False,
            "kv260_ssh_reachable": False,
            "random_seed": 20260601,
        }

    monkeypatch.setattr(module, "run_experiment", fake_run_experiment)
    rc = module.main([])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["honest_verdict"] == "complete: blocked_kv260_ssh_unreachable"
    assert out["kv260_terminal_state_reached"] is False
    assert out["kv260_ssh_reachable"] is False
    assert out["random_seed"] == 20260601


def test_main_print_result_path(monkeypatch, capsys):
    monkeypatch.setattr(
        module,
        "run_experiment",
        lambda: {
            "honest_verdict": "x",
            "kv260_terminal_state_reached": False,
            "kv260_ssh_reachable": False,
            "random_seed": 20260601,
        },
    )
    rc = module.main(["--print-result-path"])
    assert rc == 0
    assert "experiment_3568_kv260_terminal_latency_transcript_v14.json" in (
        capsys.readouterr().out
    )
