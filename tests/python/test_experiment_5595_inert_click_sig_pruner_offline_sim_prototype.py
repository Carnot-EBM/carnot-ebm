"""Tests for Exp 5595 InertClickSigPruner offline-sim prototype.

Spec refs: REQ-ARC-FCP-5595, SCENARIO-ARC-FCP-5595-SIGNATURE-CLASSIFIED-ON-REAL-DATA,
SCENARIO-ARC-FCP-5595-RANK-CANDIDATES-SANITY-CHECK.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5595_inert_click_sig_pruner_offline_sim_prototype as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_fcp_5595_spec_declares_pruner_contract() -> None:
    """REQ-ARC-FCP-5595: OpenSpec declares the InertClickSigPruner offline-sim contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5595") :]

    for marker in (
        "REQ-ARC-FCP-5595",
        "SCENARIO-ARC-FCP-5595-SIGNATURE-CLASSIFIED-ON-REAL-DATA",
        "SCENARIO-ARC-FCP-5595-RANK-CANDIDATES-SANITY-CHECK",
        "InertClickSigPruner",
        "click_signature",
        "twin_count",
    ):
        assert marker in section


def test_scenario_arc_fcp_5595_blocked_precondition_never_measures(monkeypatch) -> None:
    """A missing resource fails closed without attempting any game."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": False,
            "offline_arcade_makes_env": False,
            "e3_and_pruner_import": True,
            "gguf_cached": True,
            "llama_server_binary_present": True,
            "port_8920_prewarmed": True,
            "ok": False,
        },
    )

    def _fail_if_called(game, **_kwargs):
        raise AssertionError("_measure_one_game must not run when a precondition is missing")

    monkeypatch.setattr(mod, "_measure_one_game", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["per_game_rows"] == []
    assert artifact["total_click_transitions_observed"] == 0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def _ok_preconds(root=mod.REPO_ROOT):
    return {
        "offline_arcade_importable": True,
        "offline_arcade_makes_env": True,
        "e3_and_pruner_import": True,
        "gguf_cached": True,
        "llama_server_binary_present": True,
        "port_8920_prewarmed": True,
        "ok": True,
    }


def test_scenario_arc_fcp_5595_synthetic_signatures_pruned(monkeypatch) -> None:
    """SCENARIO-ARC-FCP-5595-SIGNATURE-CLASSIFIED-ON-REAL-DATA: at least one confidently
    inert signature is classified as an honest confirmed-pruning verdict."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "_measure_one_game",
        lambda game, **_kwargs: {
            "game": game,
            "transitions_collected": 20,
            "click_transitions": 18,
            "signatures_tracked_after": 3,
            "signatures_confidently_inert_after": 2,
            "rank_candidates_sanity": {"rows_in": 20, "rows_kept": 12, "rows_dropped": 8},
        },
    )

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"] == (
        "complete: inert_click_sig_pruner_prototype_confirmed_2_signatures_pruned_across_1_games"
    )
    assert artifact["total_signatures_confidently_inert"] == 2


def test_scenario_arc_fcp_5595_no_clicks_observed_is_honest_null(monkeypatch) -> None:
    """A roster game that produced zero click transitions is reported honestly, not
    misread as a pruning confirmation."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "_measure_one_game",
        lambda game, **_kwargs: {
            "game": game,
            "transitions_collected": 10,
            "click_transitions": 0,
            "signatures_tracked_after": 0,
            "signatures_confidently_inert_after": 0,
            "rank_candidates_sanity": {"rows_in": 10, "rows_kept": 10, "rows_dropped": 0},
        },
    )

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"] == (
        "complete: inert_click_sig_pruner_prototype_no_click_transitions_observed"
    )


def test_scenario_arc_fcp_5595_ran_but_below_evidence_floor_is_honest_null(monkeypatch) -> None:
    """Clicks were observed but no signature accumulated enough repeated evidence to
    clear the trust+specificity gate -- an honest null, not a fabricated pruning claim."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "_measure_one_game",
        lambda game, **_kwargs: {
            "game": game,
            "transitions_collected": 15,
            "click_transitions": 14,
            "signatures_tracked_after": 6,
            "signatures_confidently_inert_after": 0,
            "rank_candidates_sanity": {"rows_in": 15, "rows_kept": 15, "rows_dropped": 0},
        },
    )

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"] == (
        "complete: inert_click_sig_pruner_prototype_ran_but_no_signature_cleared_evidence_floor"
    )


def test_scenario_arc_fcp_5595_rank_candidates_sanity_check_recorded(monkeypatch) -> None:
    """SCENARIO-ARC-FCP-5595-RANK-CANDIDATES-SANITY-CHECK: the rank_candidates dry-run
    against real collected rows is recorded per game, not silently dropped."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconds)
    monkeypatch.setattr(
        mod,
        "_measure_one_game",
        lambda game, **_kwargs: {
            "game": game,
            "transitions_collected": 22,
            "click_transitions": 21,
            "signatures_tracked_after": 2,
            "signatures_confidently_inert_after": 1,
            "rank_candidates_sanity": {"rows_in": 22, "rows_kept": 17, "rows_dropped": 5},
        },
    )

    artifact = mod.build_artifact()

    row = artifact["per_game_rows"][0]
    assert row["rank_candidates_sanity"]["rows_dropped"] == 5
    assert row["rank_candidates_sanity"]["rows_in"] == (
        row["rank_candidates_sanity"]["rows_kept"] + row["rank_candidates_sanity"]["rows_dropped"]
    )


def test_req_arc_fcp_5595_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-FCP-5595: the checked-in real run measured InertClickSigPruner against
    real click transitions from a real offline-arcade game -- not a fabricated or
    blocked stub."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["honest_verdict"].startswith("complete: inert_click_sig_pruner_prototype_")
    assert result["inference_substrate"] == (
        "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    assert result["solve_provenance"] == "development_proxy"
    assert result["total_click_transitions_observed"] > 0
    assert len(result["per_game_rows"]) >= 1
    assert result["duration_s"] > 5.0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
