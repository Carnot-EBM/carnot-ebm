"""Tests for the MAX_ACTIONS answer analyser + the LLM-ON contention probe.

REQ-ARC-WMTE-5990 / SCENARIO-ARC-WMTE-5990-*.

Each test is written against a SPECIFIC failure this project has actually shipped, not against a
happy path:

* the false iGPU block (a healthy CUDA-GPU1 server was itself holding the VRAM the resolver's
  headroom guard wanted, so the resolver reported the iGPU fallback for a launch that would never
  happen) -- and the wedged-server case the reuse clause must still REFUSE;
* the one-sided test that makes a reversal read as no effect, and the p-floor that must be published
  when it cannot reach 0.05;
* the unvalidated estimator: a corpus-wide reset-attribution estimate must not be reported without
  its measured error, and its two structural invariants (monotone, capped) must hold;
* the free opening reset (M2): exactly ONE reset comes off the FIRST span and never below zero;
* the "concurrency" arm that did not actually overlap, which would read as a clean null.
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import analyze_arc_max_actions_answer as ans  # noqa: E402
import arc_gateway_rescore as gw  # noqa: E402
import arc_llm_on_contention_probe as probe  # noqa: E402


# --------------------------------------------------------------- SCENARIO-ARC-WMTE-5990-BOTH-TAILS
def test_sign_test_reports_both_tails_and_the_reachable_floor():
    """A 4-pair all-positive support cannot reach p<=0.05 and must say so."""
    r = ans.both_tails_sign_test([0.3, 0.4, 0.2, 0.1])
    assert r["n_nonzero_pairs"] == 4
    assert r["n_positive"] == 4 and r["n_negative"] == 0
    assert r["p_two_sided"] == 0.125
    assert r["min_reachable_two_sided_p_at_this_support"] == 0.125
    assert r["can_ever_reach_0_05"] is False
    assert r["verdict"] == "UNDERPOWERED_p_floor_above_0.05"
    # BOTH tails present, so a reversal cannot be read as a null.
    assert r["p_one_sided_increase"] < r["p_one_sided_decrease"]


def test_sign_test_names_a_reversal_as_a_direction_not_a_null():
    r = ans.both_tails_sign_test([-0.3, -0.4, -0.2, 0.1])
    assert r["direction_favoured"] == "decrease"
    assert r["n_negative"] == 3 and r["n_positive"] == 1
    assert r["p_two_sided"] == 0.625


def test_sign_test_excludes_exact_ties_from_the_support():
    r = ans.both_tails_sign_test([0.0, 0.0, 0.5])
    assert r["n_nonzero_pairs"] == 1


# ---------------------------------------------------------------- SCENARIO-ARC-WMTE-5990-ESTIMATOR
def test_uniform_rate_estimator_is_monotone_and_capped():
    """Cumulative resets can never decrease, nor exceed the run total."""
    spans = [100, 50, 850]
    est = ans._uniform_rate_cum_resets(spans, 1000, 40)
    assert est == sorted(est), est
    assert max(est) <= 40
    assert est[0] == 4  # 40 * 100/1000
    assert est[-1] == 40


def test_uniform_rate_estimator_handles_zero_actions_without_dividing_by_zero():
    assert ans._uniform_rate_cum_resets([0], 0, 7) == [0]


def test_uniform_rate_estimator_error_is_measured_on_the_real_exact_cells():
    """The corpus estimate inherits this error; the artifact must publish it, non-empty."""
    exact = sorted(
        __import__("glob").glob(
            os.path.join(REPO, "results/early_stop_sweep_20260726/rows_exact_attribution*.json")
        )
    )
    assert exact, "the 48 exact-attribution cells are the estimator's only validation set"
    out = ans.part2_score_curve(
        sorted(
            p
            for p in __import__("glob").glob(
                os.path.join(REPO, "results/early_stop_sweep_20260726/rows_b*.json")
            )
            if "g350" not in p
        ),
        exact,
    )
    v = out["estimator_validation"]
    assert v["n_exact_cells_usable"] >= 20
    assert v["convention_identity_holds_on_all"] is True
    assert v["abs_rel_error_median"] is not None
    assert v["direction_test"]["n_nonzero_pairs"] >= 1
    # the dead-baseline channel must be ALIVE, or every charge model agrees at 0 and reads as a null
    assert out["channel_checks"]["baseline_channel_alive"] is True


# ------------------------------------------------------------ SCENARIO-ARC-WMTE-5990-FREE-OPENING
def test_free_opening_reset_removes_exactly_one_from_the_first_span_only():
    spans = [15, 27]
    cum = [3, 5]
    m1 = ans._charged_spans(spans, cum, False)
    m2 = ans._charged_spans(spans, cum, True)
    assert m1 == [18, 29]
    assert m2 == [17, 29]  # one reset off span 1; span 2 untouched
    assert sum(m1) - sum(m2) == 1


def test_free_opening_reset_never_goes_below_the_offline_span():
    spans = [10, 10]
    cum = [0, 0]
    assert ans._charged_spans(spans, cum, True) == [10, 10]


def test_m2_scores_no_lower_than_m1_and_charging_resets_really_costs_score():
    """Anchored on the INSTALLED scorer, not a paraphrase of the formula."""
    baselines = [10, 20]
    m0 = ans._score(baselines, [10, 20], 0, True)
    m1 = ans._score(baselines, ans._charged_spans([10, 20], [5, 5], False), 0, True)
    m2 = ans._score(baselines, ans._charged_spans([10, 20], [5, 5], True), 0, True)
    assert m1 < m0, "charging resets must reduce the score"
    assert m1 <= m2 <= m0, "the bootstrap-free model sits between all-charged and offline"
    chain = gw.crosscheck_reset_charge()
    assert chain["reset_is_charged"] is True


# --------------------------------------------------------------- SCENARIO-ARC-WMTE-5990-OVERLAP
def test_overlap_is_the_intersection_so_a_non_overlapping_batch_reads_zero():
    assert probe._overlap_seconds([(0.0, 10.0), (20.0, 30.0)]) == 0.0
    assert probe._overlap_seconds([(0.0, 10.0), (5.0, 30.0)]) == 5.0
    assert probe._overlap_seconds([(0.0, 10.0)]) == 10.0
    assert probe._overlap_seconds([]) == 0.0


# ------------------------------------------------------------ SCENARIO-ARC-WMTE-5990-DEVICE-WITNESS
def test_reuse_witness_refuses_a_wedged_server_holding_only_a_few_hundred_mib():
    """The observed wedged server had ~296 MiB resident -- i.e. NO weights on the card. Passing the
    device precondition on that would be exactly the silent-wrong-device failure the check exists
    for."""
    port = 65530

    def fake_apps():
        return [{"pid": "111", "gpu_index": 1, "used_mib": "296 MiB"}]

    def fake_ps(cmd, **kw):
        class R:
            stdout = f"  111 llama-server -m x --port {port} --host 127.0.0.1\n"

        return R()

    orig = probe.subprocess.run
    probe.subprocess.run = fake_ps  # type: ignore[assignment]
    try:
        r = probe._existing_server_on_gpu1(port, fake_apps)
    finally:
        probe.subprocess.run = orig  # type: ignore[assignment]
    assert r["reusable_gpu1_server"] is False
    assert r["pids_bound_to_port"] == ["111"]


def test_reuse_witness_passes_only_for_a_pid_bound_to_this_port_with_multi_gib_on_card_1():
    port = 65531

    def fake_apps():
        return [
            {"pid": "222", "gpu_index": 1, "used_mib": "11850 MiB"},  # ours
            {"pid": "999", "gpu_index": 0, "used_mib": "20000 MiB"},  # the conductor's, card 0
        ]

    def fake_ps(cmd, **kw):
        class R:
            stdout = f"  222 llama-server -m x --port {port}\n  999 python train.py\n"

        return R()

    orig = probe.subprocess.run
    probe.subprocess.run = fake_ps  # type: ignore[assignment]
    try:
        r = probe._existing_server_on_gpu1(port, fake_apps)
    finally:
        probe.subprocess.run = orig  # type: ignore[assignment]
    assert r["reusable_gpu1_server"] is True
    assert [m["pid"] for m in r["gpu1_resident_multi_gib"]] == ["222"]


def test_reuse_witness_refuses_when_the_resident_process_is_on_the_conductors_card():
    """VRAM on card 0 is the conductor's; it must never satisfy a GPU-1 precondition."""
    port = 65532

    def fake_apps():
        return [{"pid": "333", "gpu_index": 0, "used_mib": "11850 MiB"}]

    def fake_ps(cmd, **kw):
        class R:
            stdout = f"  333 llama-server -m x --port {port}\n"

        return R()

    orig = probe.subprocess.run
    probe.subprocess.run = fake_ps  # type: ignore[assignment]
    try:
        r = probe._existing_server_on_gpu1(port, fake_apps)
    finally:
        probe.subprocess.run = orig  # type: ignore[assignment]
    assert r["reusable_gpu1_server"] is False


# --------------------------------------------------------- SCENARIO-ARC-WMTE-5990-ARTIFACT-SHAPE
def test_artifact_keeps_the_two_clocks_separate_and_declares_a_READABLE_substrate():
    """`inference_substrate` must be a BARE STRING, and the rows' substrates recorded separately.

    An earlier version of this analyser emitted a DICT (one entry per part), which is what a sibling
    lane's artifact does. It is unreadable to `scripts/adversarial_verify.py`'s substrate classifier:
    with no recognisable value it falls back to the strict live-inference duration floor and
    CRITICAL-flags DURATION_TOO_SHORT on an honest sub-second aggregation. That is exactly what
    happened on this file's first build, so the contract asserted here is the corrected one.
    """
    path = os.path.join(REPO, "results/outer_loop_arc_max_actions_answer_20260726.json")
    if not os.path.exists(path):  # pragma: no cover -- the analyser has not been run yet
        return
    d = json.load(open(path))
    assert d["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert isinstance(d["rows_inference_substrate"], dict)
    assert d["rows_inference_substrate"]["contention_ladder_rows"] == "live_llm_inference"
    assert d["model_specs"]["invoked_by_this_file"] is False
    assert d["measurement_wall_s"] > d["duration_s"] * 10, (
        "the analyser clock must never be published as the measurement clock"
    )
    assert d["what_was_NOT_changed"], "the no-flag-flip contract must be stated on the artifact"
    assert d["honest_verdict"].startswith("complete_"), (
        "terminal-prefix discipline: words like 'died' and 'binds' in the verdict would otherwise "
        "trip the reconciler's partial/blocked token matcher"
    )
