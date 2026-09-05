"""Spec: REQ-ARC-WMTE-7021, REQ-ARC-WMTE-7022

Every eval run declares its provenance and the envelope it actually ran in.

INCIDENT 2026-09-04/05, two halves of one gap.

REQ-7021: the eval wrote no `solve_provenance`. The live self-discovery headline therefore
rested on a structural argument written into ONE consumer -- `generalization_levels()` reasons
that the harness is adapter-free by construction -- while the second source in that same
function demands an explicit stamp and would have credited these runs zero. Any other reader saw
None.

REQ-7022: thirteen of thirteen eval-run artifacts recorded no GPU, no VRAM, no `n_ctx`, no model
path. The `n_ctx=98304` conclusion and the engine comparison rested on runs whose artifacts could
not say what produced them. A third party cannot check a hardware claim against an artifact that
omits the hardware, and G2 asks exactly that.

The envelope records what the DRIVER reports, not what the environment requested. Those
disagreed for a whole run on 2026-09-04: `CARNOT_ARC_GENERATOR_CUDA_GPU=1` still split a 27B
model across both cards, because that variable selects a preferred device and does not mask the
others.
"""

from __future__ import annotations

from pathlib import Path

from carnot.agentic import arc_run_envelope as env_mod

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_the_payload_declares_live_agent_self_discovery() -> None:
    """The stamp the dashboard should not have had to infer."""
    src = (REPO / "scripts" / "arc_leaderboard_eval.py").read_text()
    assert '"solve_provenance": "live_agent_self_discovery"' in src


def test_the_envelope_records_the_requested_context_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_INDUCE_N_CTX", "98304")
    monkeypatch.setattr(env_mod, "_ENVELOPE_CACHE", None, raising=False)
    assert env_mod.run_envelope()["n_ctx_requested"] == "98304"


def test_the_envelope_records_the_requested_gpu_separately_from_what_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requested and held are different facts; conflating them is the 2026-09-04 misreport."""
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
    monkeypatch.setattr(env_mod, "_ENVELOPE_CACHE", None, raising=False)
    env = env_mod.run_envelope()
    assert env["generator_cuda_gpu_requested"] == "1"
    assert "gpu_indices_held" in env


def test_absence_is_recorded_not_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key and a measured 'none' are different facts, so every key is present."""
    for name in (
        "CARNOT_ARC_INDUCE_N_CTX",
        "CARNOT_ARC_GENERATOR_CUDA_GPU",
        "CUDA_VISIBLE_DEVICES",
        "CARNOT_LLAMA_SERVER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_mod, "_ENVELOPE_CACHE", None, raising=False)
    env = env_mod.run_envelope()
    for key in (
        "n_ctx_requested",
        "generator_cuda_gpu_requested",
        "cuda_visible_devices",
        "llama_server_path",
        "model_path",
        "gpus_held",
        "gpu_indices_held",
        "hostname",
    ):
        assert key in env, key
    assert env["n_ctx_requested"] is None


def test_the_envelope_is_cached_so_a_multi_game_run_reports_one_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-reading per game would let two games in one run disagree about the hardware."""
    monkeypatch.setattr(env_mod, "_ENVELOPE_CACHE", None, raising=False)
    first = env_mod.run_envelope()
    monkeypatch.setenv("CARNOT_ARC_INDUCE_N_CTX", "12345")
    assert env_mod.run_envelope() is first
