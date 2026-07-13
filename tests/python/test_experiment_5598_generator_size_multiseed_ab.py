"""Tests for Exp 5598 properly-powered multiseed generator-size A/B (3 arms: current
Qwen3.5-9B-MTP, candidate_27b Qwen3.6-27B-MTP, candidate_35b_moe Qwen3.6-35B-A3B-MTP).

Spec refs: REQ-ARC-WMTE-5598, SCENARIO-ARC-WMTE-5598-MULTISEED-PAIRED-COMPARISON,
SCENARIO-ARC-WMTE-5598-ARM-BATCHED-SERVER-LIFECYCLE.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5598_generator_size_multiseed_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5598_spec_declares_multiseed_contract() -> None:
    """REQ-ARC-WMTE-5598: OpenSpec declares the multiseed A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5598") :]

    for marker in (
        "REQ-ARC-WMTE-5598",
        "SCENARIO-ARC-WMTE-5598-MULTISEED-PAIRED-COMPARISON",
        "SCENARIO-ARC-WMTE-5598-ARM-BATCHED-SERVER-LIFECYCLE",
        "gpu1_unreachable_mid_run_aborting_remaining_draws",
        "No devices were found",
        "paired_vs_current",
    ):
        assert marker in section


def _ok_preconds(root=mod.REPO_ROOT):
    return {
        "offline_arcade_importable": True,
        "offline_arcade_makes_env": True,
        "e3_policy_import": True,
        "gguf_cached_current": True,
        "gguf_cached_candidate_27b": True,
        "gguf_cached_candidate_35b_moe": True,
        "llama_server_binary_present": True,
        "gpu1_free_vram_sufficient": True,
        "ok": True,
    }


def test_scenario_arc_wmte_5598_blocked_precondition_never_runs(monkeypatch) -> None:
    """A missing resource fails closed without attempting any draw."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": False,
            "offline_arcade_makes_env": False,
            "e3_policy_import": True,
            "gguf_cached_current": True,
            "gguf_cached_candidate_27b": True,
            "gguf_cached_candidate_35b_moe": True,
            "llama_server_binary_present": True,
            "gpu1_free_vram_sufficient": True,
            "ok": False,
        },
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("_run_one_draw must not run when a precondition is missing")

    monkeypatch.setattr(mod, "_run_one_draw", _fail_if_called)
    monkeypatch.setattr(mod, "_make_proposer", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["per_draw_results"] == []
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_wmte_5598_gpu1_lost_mid_run_aborts_honestly(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5598-ARM-BATCHED-SERVER-LIFECYCLE (the abort path): if GPU 1 becomes
    unreachable partway through a run, the experiment stops collecting further draws and
    reports a distinct blocked verdict rather than silently continuing on different hardware."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(mod, "_declares_mtp_metadata", lambda repo_substr: True)
    monkeypatch.setattr(
        mod, "_mtp_self_draft_fits_vram", lambda repo_substr: (False, "does not fit")
    )

    class _FakeProposer:
        def stop(self):
            pass

    monkeypatch.setattr(mod, "_make_proposer", lambda *a, **k: _FakeProposer())
    monkeypatch.setattr(mod, "_wait_for_port_down", lambda port: None)

    draws_seen: list[str] = []

    def _fake_run_one_draw(game, *, arm_name, proposer, repeat, explore_budget, total_budget):
        draws_seen.append(f"{arm_name}:{game}:{repeat}")
        return {
            "arm": arm_name,
            "game": game,
            "repeat": repeat,
            "induction_ok": True,
            "heldout_accuracy": 0.5,
        }

    monkeypatch.setattr(mod, "_run_one_draw", _fake_run_one_draw)

    # GPU 1 looks healthy for the first two draws, then vanishes.
    free_mb_sequence = iter([20000, 20000, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    monkeypatch.setattr(mod, "_gpu1_free_mb", lambda: next(free_mb_sequence, -1))

    artifact = mod.build_artifact(roster=("m0r0", "sk48"), n_seeds=3)

    assert artifact["honest_verdict"].startswith(
        "complete: generator_size_multiseed_ab_blocked_gpu1_lost_mid_run_partial_ranked_"
    )
    error_rows = [
        r for r in artifact["per_draw_results"] if r.get("error", "").startswith("gpu1_unreachable")
    ]
    assert len(error_rows) == 1
    # only the "current" arm's first 2 draws ran before the abort; no later arm was reached.
    assert draws_seen == ["current:m0r0:0", "current:m0r0:1"]


def test_scenario_arc_wmte_5598_paired_win_loss_tie_computed(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-5598-MULTISEED-PAIRED-COMPARISON: paired comparisons are keyed on
    (game, repeat), not just arm identity, so a candidate that wins every draw is correctly
    tallied as a clean sweep."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(mod, "_declares_mtp_metadata", lambda repo_substr: True)
    monkeypatch.setattr(
        mod, "_mtp_self_draft_fits_vram", lambda repo_substr: (False, "does not fit")
    )

    class _FakeProposer:
        def stop(self):
            pass

    monkeypatch.setattr(mod, "_make_proposer", lambda *a, **k: _FakeProposer())
    monkeypatch.setattr(mod, "_wait_for_port_down", lambda port: None)
    monkeypatch.setattr(mod, "_gpu1_free_mb", lambda: 20000)

    def _fake_run_one_draw(game, *, arm_name, proposer, repeat, explore_budget, total_budget):
        accuracy = 0.2 if arm_name == "current" else 0.8
        return {
            "arm": arm_name,
            "game": game,
            "repeat": repeat,
            "induction_ok": True,
            "heldout_accuracy": accuracy,
        }

    monkeypatch.setattr(mod, "_run_one_draw", _fake_run_one_draw)

    artifact = mod.build_artifact(roster=("m0r0",), n_seeds=2)

    assert artifact["paired_vs_current"]["candidate_27b"] == {"wins": 2, "losses": 0, "ties": 0}
    assert artifact["paired_vs_current"]["candidate_35b_moe"] == {
        "wins": 2,
        "losses": 0,
        "ties": 0,
    }
    assert artifact["honest_verdict"].startswith("complete: generator_size_multiseed_ab_ranked_")


def test_req_arc_wmte_5598_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-WMTE-5598: the checked-in real run measured all 3 arms on the SAME hardware
    tier across a widened roster with multiple repeats, resolving the exp5596/5597
    contradiction -- both candidates beat current, dense 27b more decisively."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["honest_verdict"] == (
        "complete: generator_size_multiseed_ab_ranked_candidate_27b_gt_candidate_35b_moe_gt_current"
    )
    assert result["inference_substrate"] == "live_llm_inference"
    assert result["solve_provenance"] == "development_proxy"
    assert result["n_seeds"] == 3
    assert set(result["roster"]) == {"m0r0", "sk48", "cd82", "sp80"}
    assert result["arm_mtp_used"] == {
        "current": True,
        "candidate_27b": False,
        "candidate_35b_moe": False,
    }
    # no GPU-1-lost error rows in the successful checked-in run
    assert not any(
        r.get("error", "").startswith("gpu1_unreachable") for r in result["per_draw_results"]
    )
    assert result["per_arm_summary"]["current"]["n_induction_ok"] == 12
    assert result["per_arm_summary"]["candidate_27b"]["n_induction_ok"] == 12
    assert result["per_arm_summary"]["candidate_35b_moe"]["n_induction_ok"] == 11
    assert (
        result["per_arm_summary"]["candidate_27b"]["mean_accuracy"]
        > (result["per_arm_summary"]["current"]["mean_accuracy"])
    )
    assert result["paired_vs_current"]["candidate_27b"]["losses"] == 0
    assert result["duration_s"] > 60.0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
