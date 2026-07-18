"""Unit tests for the actions-to-progress metric harness.

Spec: REQ-ARC-WMTE-5720 (harness + paired-honesty stats),
      SCENARIO-ARC-WMTE-5720-HARNESS-DRIVES-LIVE-E3,
      SCENARIO-ARC-WMTE-5720-PAIRED-HONEST-STATS,
      SCENARIO-ARC-WMTE-5721-RETRIEVAL-THROUGH-LIVE-WIRING.

These cover the pure logic (arm config + restore, induction aggregation, paired
stats) that does NOT need a GPU or the ARC SDK. The live-drive path
(run_bounded_progress) is exercised end-to-end by the experiment itself.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from carnot.agentic import arc_actions_to_progress as atp


def _mock_proposer():
    return SimpleNamespace(
        no_think_prefix="/no_think\n", max_tokens=4096, tries=3,
        include_playbook_exemplars=False,
    )


# --- REQ-ARC-WMTE-5720: arm config is activated through the real live env-var wiring ---


def test_arm_configs_cover_both_questions():
    # reason question arms
    assert atp.ARM_CONFIGS["frozen"]["codeonly"] == "1"
    assert atp.ARM_CONFIGS["frozen"]["no_think_prefix"] == "/no_think\n"
    assert atp.ARM_CONFIGS["reason"]["codeonly"] == "0"
    assert atp.ARM_CONFIGS["reason"]["no_think_prefix"] == "/think\n"
    # retrieval question arms
    assert atp.ARM_CONFIGS["retrieval"]["retrieval"] == "1"
    assert atp.ARM_CONFIGS["static"]["static"] == "1"
    # none == frozen config (the retrieval control) byte-for-byte on the wiring knobs
    for k in ("codeonly", "no_think_prefix", "retrieval", "static", "max_tokens"):
        assert atp.ARM_CONFIGS["none"][k] == atp.ARM_CONFIGS["frozen"][k]


def test_apply_arm_reason_sets_live_wiring_then_restores():
    prop = _mock_proposer()
    before_codeonly = os.environ.get("CARNOT_ARC_CODEONLY_INDUCE")
    restore = atp.apply_arm(prop, "reason")
    try:
        # the genuine-reasoning arm removes the codeonly fence + flips to /think + bigger budget
        assert os.environ["CARNOT_ARC_CODEONLY_INDUCE"] == "0"
        assert prop.no_think_prefix == "/think\n"
        assert prop.max_tokens == 8192
        assert prop.tries == 1
    finally:
        restore()
    assert prop.no_think_prefix == "/no_think\n"
    assert prop.max_tokens == 4096
    assert os.environ.get("CARNOT_ARC_CODEONLY_INDUCE") == before_codeonly


def test_apply_arm_retrieval_sets_retrieval_env():
    prop = _mock_proposer()
    restore = atp.apply_arm(prop, "retrieval")
    try:
        assert os.environ["CARNOT_ARC_PLAYBOOK_RETRIEVAL"] == "1"
        assert os.environ["CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED"] == "0"
        assert prop.include_playbook_exemplars is False  # E3 self-arms it per stall
    finally:
        restore()
    assert os.environ.get("CARNOT_ARC_PLAYBOOK_RETRIEVAL") in (None, "0")


# --- induction-quality aggregation over the run's multiple attempts ---


def test_summarize_inductions_aggregates_plan_and_heldout():
    events = [
        {"planned": True, "playbook_injection_mode": "retrieval",
         "refinement_rounds": [{"heldout_accuracy": 0.8, "prefix_accuracy": 0.9},
                               {"heldout_accuracy": 0.95, "prefix_accuracy": 0.99}]},
        {"planned": False, "playbook_injection_mode": "none",
         "refinement_rounds": [{"heldout_accuracy": 0.5, "prefix_accuracy": 0.6}]},
    ]
    s = atp._summarize_inductions(events)
    assert s["n_inductions"] == 2
    assert s["n_plans_found"] == 1
    assert s["plan_found_rate"] == 0.5
    # best heldout per event, then mean: (0.95 + 0.5) / 2 = 0.725
    assert abs(s["mean_heldout_accuracy"] - 0.725) < 1e-6
    assert s["playbook_injection_modes"] == ["retrieval", "none"]


def test_summarize_inductions_empty():
    s = atp._summarize_inductions([])
    assert s["n_inductions"] == 0
    assert s["plan_found_rate"] is None
    assert s["mean_heldout_accuracy"] is None


# --- REQ-ARC-WMTE-5720: paired honest stats ---


def _mk(game, seed, arm, hv):
    return atp.ProgressResult(
        game=game, arm=arm, seed=seed, variant=0, start_level=0, reached_level=0,
        levels_gained=0, solved=False, actions_to_first_solve=None, total_actions=10,
        noop_frac=0.0, revisit_frac=0.0, start_hv=10.0, best_hv=10.0 - hv * 10, hv_progress=hv,
        n_inductions=1, n_plans_found=0, plan_found_rate=0.0, mean_heldout_accuracy=0.8,
        mean_prefix_accuracy=0.9, playbook_injection_modes=["none"], wall_s=1.0,
        timed_out=False, hit_induction_cap=False,
    )


def test_paired_summary_treat_wins_all():
    results = []
    for g in ("a", "b", "c"):
        results.append(_mk(g, 1, "reason", hv=0.5))   # treat better
        results.append(_mk(g, 1, "frozen", hv=0.2))   # base worse
    s = atp.paired_summary(results, "reason", "frozen", metric="hv_progress")
    assert s["n_pairs"] == 3
    assert abs(s["mean_delta"] - 0.3) < 1e-6
    assert s["wins_treat"] == 3 and s["losses_treat"] == 0 and s["ties"] == 0
    assert s["sign_test_p"] == 0.25  # 2 * (1/2^3) = 0.25, the min p for 3 concordant pairs


def test_paired_summary_drops_none_metric_pairs():
    r_t = _mk("a", 1, "reason", hv=0.5)
    r_b = _mk("a", 1, "frozen", hv=0.2)
    r_b.plan_found_rate = None  # base missing -> pair dropped for that metric
    s = atp.paired_summary([r_t, r_b], "reason", "frozen", metric="plan_found_rate")
    assert s["n_pairs"] == 0


def test_paired_summary_flags_outlier_fragile():
    # two tiny losses + one huge win -> positive mean, but dropping the win flips it negative
    results = [
        _mk("a", 1, "reason", hv=0.0), _mk("a", 1, "frozen", hv=0.05),  # delta -0.05
        _mk("b", 1, "reason", hv=0.0), _mk("b", 1, "frozen", hv=0.05),  # delta -0.05
        _mk("c", 1, "reason", hv=0.9), _mk("c", 1, "frozen", hv=0.0),   # delta +0.9 (outlier)
    ]
    s = atp.paired_summary(results, "reason", "frozen", metric="hv_progress")
    assert s["mean_delta"] > 0
    assert s["outlier_fragile"] is True


def test_sign_test_p_symmetric_and_bounded():
    assert atp._sign_test_p(0, 0) is None
    assert atp._sign_test_p(5, 0) == atp._sign_test_p(0, 5)
    assert 0.0 <= atp._sign_test_p(4, 1) <= 1.0
    assert atp._sign_test_p(1, 1) == 1.0  # fully discordant -> p=1
