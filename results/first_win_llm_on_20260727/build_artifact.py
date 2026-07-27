#!/usr/bin/env python
"""Build the milestone artifact from analysis.json.

Substrate honesty (the exp5178 lesson): the ARMS are real compute -- a real Qwen3.5-9B
GGUF loaded on a real 3090 with real generation -- so `inference_substrate` is
`live_llm_inference` and `duration_s` reports the MEASUREMENT clock (summed from each row
file's own elapsed_s), NOT this builder's clock. The builder's own cost is published
separately as `artifact_build_s` so a fast build can never make a slow measurement look
fabricated, and a slow build can never pad a fast one.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
OUT = REPO / "results" / "first_win_llm_on_20260727"
ART = REPO / "results" / "outer_loop_arc_first_win_llm_on_eval_concurrency_20260727.json"


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    t0 = time.time()
    analysis = json.loads((OUT / "analysis.json").read_text())
    arms = analysis["arms"]
    comps = {c["comparison"]: c for c in analysis["comparisons"]}
    base = analysis["baseline"]

    runs_raw = analysis.get("runs", {})
    off = arms.get("llm_off", {})
    fix = arms.get("llm_on_fix", {})
    faulty = arms.get("llm_on_16k", {})

    # ------------------------------------------------------------------ witnesses
    # POSITIVE CONTROL / NON-FORCED-GATE WITNESS. If the harness could not record a
    # first_win at all, an "LLM-on scores 0.00" headline would be unfalsifiable -- it would
    # be indistinguishable from a broken win detector. The llm_off arm, run in the SAME
    # tree on the SAME variants through the SAME code path, is that control: it must
    # reproduce the baseline's 0.04 AND name the same winning variants.
    harness_can_detect_a_win = int(off.get("n_first_win") or 0) > 0
    # POINT-ESTIMATE EQUALITY IS NOT A REPRODUCTION TEST (corrected 2026-07-27, adversarial
    # review). `==` on two rates estimated from 100 Bernoulli trials each answers "are these the
    # same number", which is not the question; the question is "are these distinguishable". At
    # 7/100 vs 4/100 they are not: each rate sits inside the OTHER's Clopper-Pearson 95%
    # interval (7/100 -> [0.0286, 0.1389] contains 0.04; 4/100 -> [0.0110, 0.0993] contains
    # 0.07) and Fisher's exact two-sided p = 0.537. Reporting `reproduces_baseline_rate: false`
    # off `==` published a non-reproduction that the data does not support -- and it was never
    # supportable, not even from the earlier 63-cell partial (1/63 -> [0.0004, 0.0853], also
    # containing 0.04, Fisher p = 0.650). So the reproduction verdict is now an interval test,
    # with the raw counts and both intervals published beside it so the reader can recheck.
    reproduces_baseline_rate_point_equal = (
        off.get("first_win_rate") == base["first_win_rate_integrated"]
    )
    _off_k = int(off.get("n_first_win") or 0)
    _off_n = int(off.get("n_cells") or 0)
    _base_k, _base_n = 4, 100  # exp4605: 4 winning held-out variants of 100

    def _clopper_pearson(k: int, n: int) -> list:
        """Exact binomial 95% interval. Exact, not normal-approx: at k=4/n=100 the normal
        approximation puts the lower bound at 0.002 and is simply wrong at this rate."""
        from scipy.stats import beta as _beta

        lo = float(_beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
        hi = float(_beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
        return [round(lo, 6), round(hi, 6)]

    _off_ci = _clopper_pearson(_off_k, _off_n) if _off_n else [0.0, 1.0]
    _base_ci = _clopper_pearson(_base_k, _base_n)
    try:
        from scipy.stats import fisher_exact as _fisher

        _repro_p = round(
            float(_fisher([[_off_k, _off_n - _off_k], [_base_k, _base_n - _base_k]])[1]), 4
        )
    except Exception:
        _repro_p = None
    # "Reproduces" == "is not distinguishable from" the baseline at the 5% level.
    reproduces_baseline_rate = bool(_repro_p is not None and _repro_p >= 0.05)
    off_winners = sorted(off.get("winning_variants") or [])
    reproduces_baseline_winners = off_winners == sorted(base["winning_variants"])
    off_win_games = sorted({s.split("~")[0] for s in off_winners})
    base_win_games = sorted({s.split("~")[0] for s in base["winning_variants"]})
    same_winning_games = off_win_games == base_win_games

    # THE GATE is falsifiability, and nothing more: can this harness record a first_win at
    # all? If it cannot, every "LLM-on scores X" number below is indistinguishable from a
    # broken win detector and the whole measurement is unfalsifiable. This conjunct is about
    # THIS arm only -- it deliberately does NOT encode an assumption about the LLM arms, which
    # is how a previous gate in this project got VOIDED.
    positive_control_passed = bool(harness_can_detect_a_win)

    # CONTENTION-CONTROL WITNESS: the 16k arm has to actually EXHIBIT the fault, otherwise
    # "faulty == fixed" would just mean the fault never fired and the control is inert.
    faulty_live = faulty.get("liveness") or {}
    fixed_live = fix.get("liveness") or {}
    # Keyed on the TARGETED fault class, not the raw error sum: the fixed arms still see a
    # SEPARATE failure (RemoteDisconnected = the server process went away), and summing the
    # two would make the fix look ineffective when it eliminated the class it targets.
    faulty_probe_live = arms.get("llm_on_16k_probe", {}).get("liveness") or {}
    fixed_probe_live = arms.get("llm_on_fix_probe", {}).get("liveness") or {}
    ctx_faulty = int(faulty_live.get("n_context_exceeded_THE_FAULT") or 0) + int(
        faulty_probe_live.get("n_context_exceeded_THE_FAULT") or 0
    )
    ctx_fixed = int(fixed_live.get("n_context_exceeded_THE_FAULT") or 0) + int(
        fixed_probe_live.get("n_context_exceeded_THE_FAULT") or 0
    )
    disc_faulty = int(faulty_live.get("n_remote_disconnected_SEPARATE_FAULT") or 0) + int(
        faulty_probe_live.get("n_remote_disconnected_SEPARATE_FAULT") or 0
    )
    disc_fixed = int(fixed_live.get("n_remote_disconnected_SEPARATE_FAULT") or 0) + int(
        fixed_probe_live.get("n_remote_disconnected_SEPARATE_FAULT") or 0
    )
    fault_exhibited = ctx_faulty > 0
    fix_clean = ctx_fixed == 0

    # LLM-ENGAGEMENT WITNESS: an LLM-on arm whose generator was never called is not an
    # LLM-on arm. This is the channel whose absence made the whole fault invisible.
    fix_engaged = int(fixed_live.get("total_llm_responses") or 0) > 0

    # TREATMENT-APPLIED CHECK. A delta between two arms that are effectively the SAME AGENT is
    # an identity, and stamping it "underpowered" invites "run more cells" -- which would give
    # exactly 0 forever. So both the gate set and the verdict branch ask whether the treatment
    # was applied at all BEFORE interpreting any delta.
    #
    # HOISTED 2026-07-27: this was computed at the verdict branch, ~120 lines BELOW the
    # acceptance-gate dict that reads it, so `bool(applied_any)` raised UnboundLocalError and the
    # builder could not run at all. Same value, defined before its first use.
    ta = analysis.get("treatment_application") or {}
    applied_any = any(v.get("treatment_was_applied") for k, v in ta.items() if isinstance(v, dict))

    payload = {
        "experiment": "outer_loop_arc_first_win_llm_on_eval_concurrency_20260727",
        "schema": "carnot.outer_loop.arc_first_win_llm_on_eval_concurrency.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": analysis["git_head"],
        "inference_substrate": "live_llm_inference",
        # duration_s IS the measurement wall clock. It is published exactly ONCE: an earlier
        # draft also carried it as `measurement_wall_s`, and adversarial_verify correctly
        # flagged that as a TAUTOLOGY -- two names for one number is precisely the redundancy
        # that check exists to catch. Per-arm breakdowns live under arms[*].
        "duration_s": analysis["measurement_wall_s"],
        "duration_s_provenance": (
            "THE MEASUREMENT CLOCK: summed from each per-cell row file's own elapsed_s "
            "(results/first_win_llm_on_20260727/cells/*.json). NOT a per-arm wall clock (K=4 "
            "workers overlap, so a wall sum undercounts by roughly the concurrency factor) "
            "and NOT the analyser's or this builder's clock -- the builder's cost is "
            "published separately as artifact_build_s. NOTE (2026-07-27): because it sums "
            "per-cell clocks across K=4 overlapping workers, it is roughly 4x the elapsed wall "
            "time of the session -- see wall_span_s_approx for the elapsed figure. Reporting "
            "both under one name is what would make this misleading; they are named "
            "differently and mean different things."
        ),
        "wall_span_s_approx": round(analysis["measurement_wall_s"] / 4.0, 1),
        "wall_span_s_approx_provenance": (
            "duration_s divided by the K=4 worker concurrency -- an APPROXIMATION of elapsed "
            "session time, published so duration_s (a summed compute clock, ~9.25h) is not "
            "misread as elapsed time (~2.3h). Not a measurement: the harness did not record a "
            "session start/stop timestamp, and one is not being invented here. The exact "
            "quantity duration_s reports is defined in duration_s_provenance above."
        ),
        "artifact_build_s": None,  # set below
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "submitted_to_leaderboard": False,
        # ------------------------------------------------------------ the question
        "question": (
            "Did the generator concurrency fault materially depress this project's own "
            "measured held-out first-win rate? Re-measure first_win_rate with a WORKING "
            "generator at eval-representative concurrency, using the baseline's exact "
            "definition, so the number is comparable rather than a differently-defined "
            "quantity."
        ),
        # ------------------------------------------------------------ the baseline
        "baseline": base,
        "baseline_provenance_correction": {
            "claim_as_stated": (
                "first_win_rate_integrated = 0.04, CI [0,0] (ops/known-issues.md:3484)"
            ),
            "what_the_ci_actually_is": (
                "[0,0] is the paired bootstrap CI on the DELTA (integrated minus bare), both "
                "arms 0.04, every pair tied -- it is NOT an interval on the 0.04 rate. The "
                "rate's own exact (Clopper-Pearson) 95% interval at 4/100 is "
                "[0.011, 0.0993]. Reporting '0.04, CI [0,0]' reads as a precisely-pinned "
                "rate; it is a precisely-pinned NULL DELTA between two arms that were both "
                "LLM-off."
            ),
            "the_llm_was_off": base["baseline_arms_were_llm_off"],
            "carry_chain": (
                "The 0.04 that gated the programme was not re-measured after exp4605. "
                "exp4950, exp4961, exp4972 and exp4983 all carry it forward "
                "(carry_method=anti_churn_carry_*, live_agent_ran=False) and each names "
                "results/experiment_4605_live_integration_scored_agent.json as its "
                "heldout_proxy_summary.source_artifact_path. exp4983 is stamped "
                "inference_substrate='live_llm_inference' and is labelled the operator's "
                "single final pre-deadline go/no-go artifact "
                "(operator_decision_number.go_no_go='no_go_no_improvement_null'), yet it "
                "resolves through that 4-hop chain to one LLM-OFF, concurrency-1 run."
            ),
            "the_live_agent_ran_true_artifacts_were_also_llm_off": (
                "Six held-out artifacts DO claim live_agent_ran=True with a first-win rate "
                "(exp4875 0.04, exp4886 0.0625, exp4896 0.052632, exp4907 0.05, exp4917 "
                "0.047619, exp4928 0.04) and all six are stamped "
                "inference_substrate='live_llm_inference'. Their generator was nevertheless "
                "NOT engaged. Code path, not inference: they run "
                "experiment_4729_held_out_first_win_readiness.py, whose per-game driver "
                "obtains its runners from "
                "`exp4605.default_variant_runner_factory('integrated')` / `('bare')` "
                "(experiment_4729_held_out_first_win_readiness.py:764-765), which resolve to "
                "exp4605.run_variant_attempt -> exp4605._policy_for_mode -> _NoOpProposer. "
                "So 'live_agent_ran' means the live AGENT (E3AgentPolicy) was stepped "
                "offline; it does NOT mean the LLM tier ran. None of the six carries any "
                "liveness field whatsoever (no call/response/error counter, no "
                "generator-health field), so the claim was unfalsifiable from the artifact. "
                "Corroborating arithmetic: exp4928 reports duration_s=673.601274 for 100 "
                "LLM-on attempts (6.7s each) and exp4875 1137.68s (11.4s each), against a "
                "measured 164-272s per LLM-engaged cell in THIS artifact's arms; and that "
                "same 673.601274 value also appears verbatim in exp4972 and exp4983, i.e. it "
                "is itself a carried number. The family also applies an explicit duration "
                "floor (experiment_4752_held_out_first_win_readiness.py:58, "
                "LIVE_DURATION_FLOOR_S = 60.0), so a duration in that lineage is not by "
                "itself evidence of compute."
            ),
            "consequence_for_the_operator_question": (
                "Because no held-out first-win number in the project's history was measured "
                "with the generator engaged, the concurrency fault CANNOT have depressed the "
                "0.04 baseline -- there was no generator contribution in it to lose. 0.04 is "
                "the LLM-OFF floor, mislabelled as an LLM-on measurement. The live question "
                "is therefore not 'how much did the fault cost us' but 'what does the LLM "
                "tier do at eval concurrency at all', which is what the arms below measure "
                "for the first time."
            ),
        },
        # ------------------------------------------------------------ arms
        "arms": arms,
        "arm_definitions": {
            "llm_off": (
                "_NoOpProposer installed exactly as experiment_4605 does -- reproduces the "
                "baseline definition in this tree. POSITIVE CONTROL."
            ),
            "llm_on_16k": (
                "shipped LocalGGUFProposer with CARNOT_ARC_INDUCE_N_CTX=16384 (the PRE-FIX "
                "context pool). CONTENTION CONTROL: same tree, same binary, same variants, "
                "same concurrency; only the pool reverted."
            ),
            "llm_on_fix": (
                "shipped LocalGGUFProposer at the shipped default n_ctx (81920). THE "
                "TREATMENT: the fixed generator."
            ),
            "llm_on_16k_probe / llm_on_fix_probe": (
                "The same two conditions run on a DIFFERENT cell set -- the cells the llm_off "
                "control actually won. Selecting cells on the control's outcome is biased by "
                "construction, so these arms carry distinct labels and never pool into the "
                "pre-specified slice's rate. Relabelled 2026-07-27 as a NEGATIVE CONTROL FOR "
                "HARNESS STABILITY rather than a directional generator check -- see "
                "control_winner_probe._what_this_probe_IS."
            ),
            "llm_on_fix_cellrecall": (
                "ADDED after the 2026-07-27 review. The fixed generator PLUS "
                "CARNOT_ARC_TRUST_METRIC=cell_recall -- the review's own suggested remedy for "
                "the severed induce->plan path, and the project's own named lever for the "
                "trust gate. Run to TEST that remedy rather than assume it. It does not work, "
                "and is counterproductive here: see trust_gate_margin_measurement."
            ),
            "llm_on_fix_diag": (
                "ADDED after the cell_recall arm produced a result contradicting the lever's "
                "documentation. The fixed generator at the SHIPPED DEFAULT metric ('exact'), "
                "with the new per-attempt gate-margin diagnostics recording. This is the "
                "direct test of whether the treatment is reachable on the path that actually "
                "ships -- the question the cell_recall arm could not answer because it "
                "changed the gating metric."
            ),
        },
        "comparisons": analysis["comparisons"],
        "control_winner_probe": analysis.get("control_winner_probe"),
        # ================================================================================
        # THE FATAL FINDING of the 2026-07-27 adversarial review, stated FIRST because it
        # governs how every number below may be read.
        # ================================================================================
        "treatment_application": analysis.get("treatment_application"),
        "witness_field_corrections_are_forward_only": {
            "llm_enabled_on_the_control_rows": (
                "All 100 llm_off control rows in this measurement carry `llm_enabled: True`, "
                "because that field reads one env var (CARNOT_ARC_DISABLE_INDUCTION) and the "
                "harness installed a stub proposer WITHOUT setting it. The rows are otherwise "
                "honest -- llm.calls/responses/errors carry the -1 UNDETERMINED sentinel, and "
                "the analyser correctly excluded those from its totals rather than summing "
                "them to -100 -- but a consumer keying on `llm_enabled` alone would read this "
                "LLM-OFF arm as an LLM-on arm. That is the same misclassification the witness "
                "exists to prevent, one field up."
            ),
            "what_was_fixed": (
                "arc_competition_agent.py's witness now also emits `generator_is_stub` and "
                "`llm_tier_operational` (a real, instrumented generator was installed -- "
                "distinct from `llm_enabled`, which is policy intent, and from "
                "`llm_on_row_valid`, which is whether it then answered cleanly). "
                "scripts/arc_llm_on_liveness_lint.py keys on `llm_tier_operational` when "
                "present, so a stub arm is read as LLM-OFF while a REAL-but-dead generator "
                "still claims the tier and is still refused."
            ),
            "why_these_rows_still_say_llm_enabled_True": (
                "FORWARD-ONLY. The 174 original cells were written before the field existed "
                "and are NOT rewritten (never-prune). Only the 25 llm_on_fix_cellrecall cells "
                "run after the fix carry the new fields. Running the lint over this whole "
                "directory therefore still reports every row as claiming llm_enabled=True -- "
                "that is history, not a live defect."
            ),
        },
        "treatment_application_headline": {
            "the_defect": (
                "THE TREATMENT WAS NEVER APPLIED. Every LLM-on arm is BIT-IDENTICAL to its "
                "matched llm_off control -- first_win, actions, reached_level and "
                "actions_to_first_levelup all equal on 74/74 cells. Cause, measured not "
                "inferred: induction_attempts_planned == 0 in 174/174 rows. The generator "
                "answered (327 calls, 234 responses across the LLM arms) and every induced "
                "world model was then REJECTED by a POST-generation trust gate before a plan "
                "could be installed (arc_competition_agent.py:5858 / :5877). So the LLM arms "
                "ARE the control agent under a different label."
            ),
            "consequence_for_the_headline": (
                "delta = 0.0, p = 1.0, CI [0,0] and 'zero discordant pairs' are ARITHMETIC "
                "IDENTITIES of comparing an arm to itself, not measurements of the fault's "
                "effect. No generator state -- fixed, faulty or absent -- could have moved "
                "first_win on these cells. The superseded verdict called this "
                "'underpowered', which invites running more cells; more cells of an identity "
                "give exactly 0 forever. The correct stamp is UNFALSIFIABLE."
            ),
            "what_is_therefore_still_unknown": (
                "The operator's gating question -- 'did the concurrency fault depress our "
                "measured first-win rate?' -- remains UNANSWERED. Not answered 'no'. The "
                "fault's first-win effect is UNTESTED, because the mechanism through which a "
                "generator could affect first_win was severed upstream of the comparison."
            ),
            "what_IS_established_by_this_run": (
                "(1) The fix works AT THE GENERATOR: n_context_exceeded 36 -> 0, response "
                "rate 0.67 -> 0.91 on the matched slice. (2) The generator is NOT the binding "
                "constraint on first_win at present -- a strictly upstream gate discards its "
                "output on every cell, so fixing the generator alone cannot move the metric. "
                "(3) The harness is stable at K=4 (see control_winner_probe, now labelled as "
                "the negative control it actually is)."
            ),
        },
        # -------------------------------------------------------- WHY THE OBVIOUS FIX FAILS
        # The review's own suggested remedy was "re-run one arm with
        # CARNOT_ARC_TRUST_METRIC=cell_recall so planned>0, the only configuration in which
        # the question is answerable". That was TESTED here rather than assumed, and it is
        # WRONG -- which is a more useful result than complying would have been.
        "trust_gate_margin_measurement": {
            "the_reviews_suggested_remedy": (
                "'Re-run one arm with CARNOT_ARC_TRUST_METRIC=cell_recall so planned>0, which "
                "is the only configuration in which the question is answerable.'"
            ),
            "was_it_tested": "YES -- and it is WRONG. Measured, not argued.",
            "what_was_run": (
                "A new arm `llm_on_fix_cellrecall`: the FIXED generator (n_ctx=81920, "
                "/props-verified, CONFIRMED_GPU1_BY_PER_PID_RESIDENCY) at K=4 over the same "
                "pre-specified variant-1 slice, with CARNOT_ARC_TRUST_METRIC=cell_recall set "
                "for that arm only (the harness clears the variable for every other arm, so "
                "the five original arms stay byte-identical in behaviour)."
            ),
            "the_lever_reached_the_gate": (
                "CONFIRMED, not assumed: each attempt's own record carries "
                "trust_metric='cell_recall'. This matters because the failure mode being "
                "ruled out is 'the env var never arrived', which would look identical."
            ),
            "result": (
                "planned is STILL 0 and the skip reason is STILL "
                "world_model_accuracy_below_threshold. The gate is not marginally missed -- "
                "the margins are enormous. See per_arm_margins for the measured values "
                "against the 0.5 threshold."
            ),
            "per_arm_margins": (analysis.get("treatment_application") or {}),
            "metric_disagreements_at_the_threshold": (
                (analysis.get("treatment_application") or {}).get(
                    "_metric_disagreements_at_the_0_5_threshold"
                )
            ),
            "AND IT IS WORSE THAN INERT -- IT IS COUNTERPRODUCTIVE HERE": (
                "On lp85 the induced world model scored verify_accuracy = 0.92, comfortably "
                "ABOVE the 0.5 trust threshold, and was gated out ANYWAY because "
                "verify_cell_recall was 0.0. Under the SHIPPED default metric ('exact') that "
                "attempt would have PASSED the gate and installed a plan. So on this corpus "
                "cell_recall is STRICTER than the default, not looser -- the opposite of what "
                "arc_competition_agent.py:5869-5872 describes it as -- and it gated out the "
                "one attempt that cleared the default's bar. This was invisible until the "
                "per-attempt gate margins were instrumented; the skip REASON string is "
                "identical in both cases."
            ),
            "what_this_forces_us_NOT_to_conclude": (
                "'planned is 0 under every configuration' is NOT supported and is not claimed "
                "here. It is supported only for the configurations actually run. The shipped "
                "default may reach planned > 0 on cells where an induced model is "
                "structurally right but changes the wrong cells -- lp85 is an existence proof "
                "that such models occur. See the llm_on_fix_diag arm, which is exactly that "
                "test."
            ),
            "why_the_remedy_fails": (
                "The metric switch changes WHICH number is compared to 0.5; it cannot help "
                "when BOTH numbers are ~0. verify_accuracy is 0.0 and verify_cell_recall "
                "tops out at 0.0181 on the measured cells -- roughly 28x short of the "
                "threshold, not a threshold-tuning distance. The induced world models are "
                "not imperfect-but-useful; on these cells they predict essentially nothing."
            ),
            "scope_limit_measured_not_assumed": (
                "Even in principle the lever could only ever reach 13 of the 25 slice cells. "
                "The `else` branch containing CARNOT_ARC_TRUST_METRIC is only taken for games "
                "NOT in HIDDEN_STATE_GAME_IDS. In the llm_on_fix arm, 13 cells skipped on "
                "world_model_accuracy_below_threshold (the reachable branch), 11 on "
                "hidden_state_trust_below_threshold (a DIFFERENT gate that ignores the "
                "variable entirely), and 1 on proposer_failed."
            ),
            "WHERE_THE_ATTEMPTS_ACTUALLY_DIE_BY_STAGE": (
                (analysis.get("treatment_application") or {}).get(
                    "_where_each_attempt_died_by_stage"
                )
            ),
            "attempts_that_CLEARED_the_trust_gate_and_died_later": (
                (analysis.get("treatment_application") or {}).get(
                    "_attempts_that_CLEARED_the_trust_gate_and_died_later"
                )
            ),
            "the_trust_gate_is_NOT_a_uniform_wall": (
                "The skip-reason histogram alone reads as 'the trust gate rejects "
                "everything'. The gate MARGINS show that is false on the SHIPPED metric: "
                "attempts scoring verify_accuracy 0.92-0.96 CLEAR the 0.5 threshold and then "
                "die one stage further downstream at `no_reachable_plan_after_refinement`. "
                "That distinction is load-bearing for what to fix: a trust-gate rejection is "
                "an INDUCTION-QUALITY problem, while a no-reachable-plan rejection is a "
                "GOAL/PLANNING problem on a world model the system already trusted. Neither "
                "is a generator problem, and neither is a context-pool problem."
            ),
            "what_this_relocates_the_blocker_to": (
                "NOT the generator, and NOT the context pool -- but ALSO not a single "
                "downstream gate. Corrected once already in this artifact's own drafting: an "
                "earlier version of this field said 'INDUCTION QUALITY, upstream of the trust "
                "gate ... which no configuration lever in the current code reaches'. The "
                "shipped-metric arm refuted that: some attempts DO clear the trust gate "
                "(verify_accuracy 0.92-0.96) and die at goal reachability instead. The honest "
                "statement is that the induce->plan path fails at DIFFERENT stages on "
                "different cells -- generation, dynamics trust, hidden-state trust, and goal "
                "reachability all appear -- so there is no single lever, and a claim that one "
                "stage is 'the' blocker is not supported by these rows. See "
                "WHERE_THE_ATTEMPTS_ACTUALLY_DIE_BY_STAGE for the distribution."
            ),
            "this_is_a_diagnosis_not_a_recommendation": (
                "Naming where the blocker sits is not a proposal to reallocate the research "
                "programme -- see no_pivot_recommendation. This artifact states that a "
                "generator fix cannot move first_win while the induced model is discarded "
                "upstream; what to do about that is the operator's call."
            ),
            "how_to_falsify_this": (
                "Produce ANY configuration in which induction_attempts_planned > 0 on a "
                "non-trivial number of cells, then re-run the matched comparison. This block "
                "is refuted by a single such run; it is not a claim that no such "
                "configuration exists, only that the named one does not work."
            ),
        },
        # ------------------------------------------------------- CONCURRENCY CEILING (F3/F4)
        "generator_concurrency_ceiling": {
            "why_this_is_here": (
                "Review finding 3: no artifact stated the ceiling the n_ctx fix LEAVES IN "
                "PLACE. Raising the pool removes the admission failure; it does not raise "
                "the service rate. The failure mode past the ceiling is the SAME silent "
                "degradation -- generate() returns (False, msg) on a client timeout and the "
                "agent proceeds LLM-off -- so it needs naming and a number, not silence."
            ),
            "server_slots": 4,
            "server_slots_source": (
                "llama-server with no explicit --parallel: n_parallel=4 and kv_unified=true "
                "(server.cpp:106-110). Confirmed live on every arm's /props: total_slots=4."
            ),
            "shipped_client_timeout_s": 600,
            "shipped_client_timeout_source": (
                "CARNOT_ARC_INDUCE_TIMEOUT default at BOTH agent call sites -- "
                "arc_competition_agent.py:890 and :5015 (verified by read, this session)."
            ),
            "arrival_concurrency_is_UNBOUNDED_by_the_framework": (
                "swarm.py starts one Thread per game with NO pool and joins them all "
                "(verified by read at /home/ianblenke/arc3_agents/agents/swarm.py:90-99: "
                "`for a in self.agents: self.threads.append(Thread(target=a.main, "
                "daemon=True))`). So the number of games sets the arrival rate; the server "
                "serves 4 at a time and QUEUES the rest."
            ),
            # NUMBERS READ OFF THE PERSISTED ROWS, not reconstructed. Source:
            # results/generator_concurrency_5866/fixprice.json, candidates[0].cells[0]
            # (K=4, all four forced to the full 4096-token budget).
            "measured_service_rate": (
                "One slot-batch of 4 full-budget (4096-token) requests takes ~145-150s at "
                "n_ctx=81920 on this box: the four K=4 requests in exp5866's own raw rows "
                "took 150.05, 147.94, 147.21 and 145.46 seconds. So the service rate is "
                "~4 full-budget requests per ~150s."
            ),
            "what_the_K6_rows_do_NOT_show": (
                "exp5866's K=6 cell cannot be used to measure queue latency: its two QUEUED "
                "requests finished NATURALLY at 898 and 877 tokens (stop_type='eos') in "
                "42.3s and 70.9s, i.e. they never held a full-budget reservation. Any "
                "queue-depth latency figure taken from those two rows would be measuring a "
                "short generation, not a queued full-budget one. Corrected here rather than "
                "repeated -- see results/outer_loop_exp5866_corrigendum_20260727.json."
            ),
            "implied_in_flight_ceiling": (
                "ARITHMETIC FROM THE TWO MEASURED CONSTANTS, flagged as such: a 600s client "
                "timeout divided by a ~150s slot-batch is ~4 batches deep, i.e. ~16 "
                "concurrent induce requests before the tail request times out. This is a "
                "DERIVED bound, not a measured one -- no run in this project has driven "
                "16 concurrent induce calls -- and it assumes strict FIFO queueing with no "
                "per-request slowdown, which would make the real ceiling LOWER, not higher."
            ),
            "what_happens_past_the_ceiling": (
                "The tail request times out; generate() returns (False, msg); the agent "
                "proceeds LLM-off. The SAME silent-degradation shape as the fault this lane "
                "fixed -- different trigger, identical symptom."
            ),
            "is_this_instrumented_now": (
                "YES, but only after the fact: a timeout increments n_server_failures, so it "
                "appears as llm.errors > 0 in the liveness witness and the row fails "
                "llm_on_row_valid. It is detectable; it is not PREVENTED."
            ),
            "not_measured_here": (
                "The ACTUAL induce-call arrival concurrency across a multi-game run was NOT "
                "measured. Induce fires on stalls, not uniformly, so whether arrivals "
                "actually cluster past 16 is unknown. That measurement -- not a further "
                "n_ctx raise -- is the next step if the LLM tier is ever unblocked "
                "downstream. The lever if they do cluster is CARNOT_ARC_INDUCE_TIMEOUT or an "
                "explicit client-side admission cap, NOT more context."
            ),
            "why_a_further_n_ctx_RAISE_is_the_wrong_lever": (
                "The ceiling is a SERVICE-RATE limit, not an admission limit. More context "
                "cells do not create more slots or make a slot finish sooner; the measured "
                "VRAM cost of context is ~0.025 MiB/cell against ~207 MiB/slot, so raising "
                "n_ctx buys admission headroom the fix already has and buys no throughput."
            ),
        },
        # ------------------------------------------------------------ witnesses
        "positive_control_passed": positive_control_passed,
        "positive_control_witness": {
            "harness_can_record_a_first_win": harness_can_detect_a_win,
            "llm_off_arm_n_first_win": off.get("n_first_win"),
            "llm_off_winning_variants": off_winners,
            "could_have_failed": (
                "Yes, and it nearly did: of the baseline's four winning variants "
                "(lp85~color01..04) only a subset still win under today's agent code. A "
                "serial K=1 re-run of the UNPATCHED exp4605 code confirms lp85~color01 now "
                "loses. Had the whole lp85 win disappeared, this conjunct would read False "
                "and every LLM-on number in this artifact would be stamped UNFALSIFIABLE "
                "rather than reported."
            ),
            "principle": (
                "A PASS needs a case that COULD have failed. Without a control arm that "
                "demonstrably records wins, any low LLM-on rate is indistinguishable from a "
                "broken win detector. This conjunct is scoped to the control arm ALONE and "
                "encodes no assumption about the LLM arms -- a gate conjunct that asserted "
                "something about another arm is how a prior gate in this project was VOIDED."
            ),
        },
        # DECLARED-vs-ACTUAL n_ctx CROSS-CHECK. The entire fault under investigation is a gap
        # between what the code DECLARED it was doing and what the server was actually doing,
        # so the two are compared here from two INDEPENDENT reads: the agent-side proposer's
        # own `generator_n_ctx` (per cell, via the liveness witness) against the server's own
        # /props `n_ctx` (read over HTTP at launch). Agreement is a fact about the system;
        # asserting the env var was set would have been a fact about my own arithmetic.
        "declared_vs_actual_n_ctx": {
            arm_name: {
                "agent_side_witness_n_ctx": (arms.get(arm_name, {}).get("liveness") or {}).get(
                    "generator_n_ctx_observed"
                ),
                "server_side_props_n_ctx": (
                    ((runs_raw.get(f"run_{arm_name}.json") or {}).get("server") or {}).get("props")
                    or {}
                ).get("n_ctx_reported"),
                "server_side_props_total_slots": (
                    ((runs_raw.get(f"run_{arm_name}.json") or {}).get("server") or {}).get("props")
                    or {}
                ).get("total_slots"),
            }
            for arm_name in sorted(arms)
        },
        # FIDELITY IS REPORTED, NOT GATED. Conflating "can this harness see a win" with
        # "does it byte-reproduce a 5-week-old run" would let ordinary code drift invalidate
        # a sound measurement -- and would hide the drift instead of naming it.
        "baseline_fidelity": {
            "definition_identical": (
                "Yes. The arms call experiment_4605.run_variant_attempt VERBATIM; the only "
                "monkeypatch is which proposer _policy_for_mode installs. Corpus, budget "
                "(200), variant ids (1,2,3,4 = HELD_OUT_VARIANT_IDS), deepen flag, "
                "target_levels and value_weight are all read from the same SUBMITTED_* "
                "helpers the baseline used."
            ),
            "llm_off_first_win_rate": off.get("first_win_rate"),
            "baseline_first_win_rate": base["first_win_rate_integrated"],
            "reproduces_baseline_rate": reproduces_baseline_rate,
            "reproduces_baseline_rate_method": (
                "CORRECTED 2026-07-27 (adversarial review). This field was previously computed "
                "as point-estimate EQUALITY (`off_rate == baseline_rate`) and published "
                "'false' -- a non-reproduction the data never supported. It is now an interval "
                "test: 'reproduces' means the two rates are NOT DISTINGUISHABLE at the 5% "
                "level by Fisher's exact two-sided test. The winner SET genuinely does differ "
                "(reproduces_baseline_winner_set_exactly stays false, correctly, and the "
                "agent-code drift that causes it is established by serialcheck.json) -- but "
                "differing in WHICH variants win is a different claim from differing in the "
                "RATE, and only the first is supported."
            ),
            "reproduces_baseline_rate_point_equal": reproduces_baseline_rate_point_equal,
            "llm_off_first_win_ci95_clopper_pearson": _off_ci,
            "baseline_first_win_ci95_clopper_pearson": _base_ci,
            "reproduction_fisher_exact_two_sided_p": _repro_p,
            "reproduction_interval_reading": (
                f"llm_off {_off_k}/{_off_n} = {(_off_k / _off_n if _off_n else 0):.4f}, "
                f"CP95 {_off_ci}; baseline {_base_k}/{_base_n} = 0.04, CP95 {_base_ci}. Each "
                f"point estimate lies inside the other's interval and Fisher's exact "
                f"two-sided p = {_repro_p}. The baseline rate REPRODUCES, indistinguishably, "
                "at a different variant composition."
            ),
            "power_caveat_the_intervals_understate": (
                "Both Clopper-Pearson intervals treat the 100 cells as independent Bernoulli "
                "trials, but they are 4 colour-permuted variants of each of 25 games and the "
                "baseline's 4 wins were ALL on one game (lp85~color01..04). The effective n is "
                "closer to the number of distinct GAMES that can win than to 100, so both "
                "intervals are anticonservative and neither 0.04 nor 0.07 should be read as a "
                "per-game first-win rate."
            ),
            "baseline_winning_variants": base["winning_variants"],
            "llm_off_winning_variants": off_winners,
            "reproduces_baseline_winner_set_exactly": reproduces_baseline_winners,
            "same_winning_games": same_winning_games,
            "llm_off_winning_games": off_win_games,
            "baseline_winning_games": base_win_games,
            "trajectory_divergence_measured": (
                "lp85~color02 wins in both, but at actions_to_first_levelup=187 here versus "
                "59 in the baseline; lp85~color01 won in the baseline and LOSES here."
            ),
            "cause_established_not_assumed": (
                "AGENT-CODE DRIFT, not this harness's threading. Discriminating test in "
                "results/first_win_llm_on_20260727/serialcheck.json: the same cells run "
                "strictly SERIALLY (K=1, single thread, single process) through the "
                "UNPATCHED exp4605 code reproduce THIS run's numbers exactly "
                "(lp85~color02 -> 187, lp85~color01 -> loss), not the baseline's. Two "
                "repetitions of each cell were bit-identical, so per-cell determinism is "
                "intact and K=4 threading is exonerated. 77 commits touched "
                "python/carnot/agentic/arc_competition_agent.py between the baseline "
                "artifact (2026-06-25) and today."
            ),
            "consequence": (
                "This artifact measures TODAY's agent with the LLM off versus TODAY's agent "
                "with the LLM on. It is NOT a bit-reproduction of exp4605, and no claim here "
                "depends on it being one: every comparison is against the llm_off arm run in "
                "THIS tree on the SAME cells, never against the June numbers."
            ),
        },
        # WHY THE NUMBER CAME OUT AS IT DID. Measured, from the shipped diagnostics channel --
        # not inferred from the rate. This is the difference between "the LLM's plans did not
        # help" and "the LLM's plans were never used", which have opposite consequences.
        "mechanism_downstream_of_the_generator": {
            "n_cells_where_llm_output_reached_the_policy": (fixed_live or {}).get(
                "n_cells_llm_output_reached_the_policy"
            ),
            "induction_skip_reason_histogram_fixed_arm": (fixed_live or {}).get(
                "induction_skip_reason_histogram"
            ),
            "generator_side_is_healthy": {
                "total_llm_calls": (fixed_live or {}).get("total_llm_calls"),
                "total_llm_responses": (fixed_live or {}).get("total_llm_responses"),
                "total_llm_server_errors": (fixed_live or {}).get("total_llm_server_errors"),
                "total_llm_content_failures": (fixed_live or {}).get("total_llm_content_failures"),
                "dead_generator_cells": (fixed_live or {}).get("dead_generator_cells"),
            },
            "what_this_means": (
                "With the FIXED generator the LLM tier answers essentially every call, yet the "
                "induced world model is REJECTED downstream on every cell and NO plan is ever "
                "installed. Both skip reasons are POST-generation gates, verified by code read: "
                "arc_competition_agent.py:5858 sets 'hidden_state_trust_below_threshold' after "
                "select_trusted_world_model returns trust_pass=False, and :5877 sets "
                "'world_model_accuracy_below_threshold' after "
                "e3.WorldModelVerifier(active_transitions).score(engine) returns a gate value "
                "below the hard 0.5 floor. Both `return` before any plan is installed. So the "
                "generator's output never reaches action selection, and a first-win delta of "
                "zero is what the pipeline MUST produce at default settings -- it is a "
                "structural consequence, not a measurement failure."
            ),
            "the_project_already_suspected_this": (
                "The comment at arc_competition_agent.py:5869-5872 names it: "
                "CARNOT_ARC_TRUST_METRIC=cell_recall is described as 'the coordinated-redesign "
                "lever for the 0.08 wall: exact-match reads ~0 for an imperfect-but-useful "
                "induced model and gates it out -> the induce->plan path is a no-op'. The "
                "default is 'exact'. This measurement is the first direct evidence for that "
                "hypothesis taken with a generator that was verifiably answering: previously "
                "the gate and the generator were confounded, because nothing recorded whether "
                "the generator had answered at all."
            ),
            "not_measured_here": (
                "The numeric MARGIN by which each gate failed (verify_accuracy, "
                "verify_cell_recall, trust_energy, heldout_accuracy) is recorded by the shipped "
                "code onto its per-attempt dict but is not exposed through "
                "generator_liveness_witness, so this harness did not capture it. Adding that "
                "capture mid-run would have instrumented only the arms started afterwards, "
                "producing exactly the asymmetric-instrumentation confound this project's own "
                "measurement discipline forbids -- so it was deliberately NOT added. The "
                "symmetric skip-reason channel above, which every arm carries, is what the "
                "finding rests on."
            ),
        },
        # COMPUTED ADMISSION WITNESS. Derived from the SHIPPED constants (max_tokens=4096 read
        # off LocalGGUFProposer, the two n_ctx values), not assumed: it states in advance that
        # the control arm's pass region is EMPTY and the treatment arm's is non-empty, so
        # neither arm's outcome rests on an untested premise about the other.
        "admission_arithmetic_witness": {
            "shipped_max_tokens": 4096,
            "K": 4,
            "n_ctx_16k_generation_budgets_alone_need_cells": 4 * 4096,
            "n_ctx_16k_pool_cells": 16384,
            "n_ctx_16k_verdict": (
                "4 * 4096 = 16384 == the ENTIRE pool, so at K=4 the four generation budgets "
                "consume every cell and ZERO remain for prompts. The pre-fix arm therefore MUST "
                "fail admission for any non-empty prompt: the control's pass region is empty by "
                "arithmetic, which is what makes observing failures there a live control rather "
                "than luck."
            ),
            "n_ctx_81920_cells_left_per_prompt_at_K4": (81920 - 4 * 4096) // 4,
            "n_ctx_81920_verdict": (
                "leaves 16384 cells per concurrent prompt, above both the ~6k real induce prompt "
                "measured in this project and the 15734-token worst case the shipped default was "
                "sized against. The treatment arm's pass region is non-empty."
            ),
        },
        "contention_control_witness": {
            "fault_exhibited_in_16k_arm": fault_exhibited,
            "n_context_exceeded_pre_fix_all_16k_cells": ctx_faulty,
            "n_context_exceeded_post_fix_all_81920_cells": ctx_fixed,
            "n_remote_disconnected_pre_fix": disc_faulty,
            "n_remote_disconnected_post_fix": disc_fixed,
            "separate_fault_note": (
                "CORRECTED 2026-07-27 (adversarial review). The original text justified setting "
                "RemoteDisconnected aside by asserting it is 'a DIFFERENT failure from the "
                "pool-exhaustion 500 this fix targets ... (the server SURVIVES the overflow)'. "
                "That parenthetical is exp5866's MODE A property, and it is precisely NOT what "
                "RemoteDisconnected is. exp5866's own taxonomy records mode B as "
                "server_survives=False with transport_error 'RemoteDisconnected: Remote end "
                "closed connection without response' and crash_site "
                "'common/sampling.cpp:154 GGML_ASSERT(logits != nullptr) -> ggml_abort'. So the "
                "stated reason for excluding these events cited the wrong mode, and the "
                "excluded events carry the shipped fault's own mode-B signature.\n"
                "WHAT IS AND IS NOT KNOWN. The discriminator was never captured: grepping all "
                "of results/first_win_llm_on_20260727/ for 'ggml_abort' or 'GGML_ASSERT' "
                "returns ZERO hits, because the harness did not keep the server's stderr. So "
                "mode-B-abort and external-SIGTERM cannot be told apart from this record, and "
                "the honest status of these 16 post-fix events is UNRESOLVED, not 'separate'.\n"
                "THE ARGUMENT THAT DOES HOLD, and was never made: at n_ctx=81920 with K=4 and "
                "max_tokens=4096, the four generation budgets reserve 4*4096 = 16384 of 81920 "
                "cells, so the generations CANNOT collectively exhaust the pool -- mode B's own "
                "stated trigger is arithmetically unavailable in the fix arm. That is a reason "
                "to doubt these are mode B; it is not a measurement, and it does not license "
                "reporting them as a settled separate fault.\n"
                "SO THE FIX'S SCOPE IS: mode A eliminated (n_context_exceeded 36 -> 0, and "
                "20 -> 0 / 16 -> 0 within the matched arms), mode C eliminated on the worst "
                "measured prompt (pool_exhaustion_limit == 0 in every fix cell), and mode B "
                "REDUCED 27 -> 16, NOT eliminated. Reported that way rather than as 'the fix "
                "removes all three'.\n"
            ),
            "unresolved_post_fix_transport_deaths": (
                "6 of the 12 llm_on_fix_probe cells carry RemoteDisconnected diagnostics at "
                "generator_n_ctx=81920 (16 diagnostics total), 2 cells end "
                "generator_healthy_after=False, and lp85_color04 is fully LLM-off at "
                "calls=4 / responses=0 / errors=4. Whatever their cause, those cells are not "
                "LLM-on evidence and must not be aggregated as such."
            ),
            "how_to_settle_it": (
                "Re-run a mode-B-specific arm -- many SMALL concurrent prompts over a long "
                "horizon, which is mode B's trigger and the OPPOSITE of the single worst-case "
                "prompt shape the 6-cell HTTP gate fired -- with the external killer excluded "
                "and the server's stderr CAPTURED, so 'ggml_abort' can be read rather than "
                "inferred."
            ),
            "n_server_errors_16k_slice_arm_all_classes": faulty_live.get("total_llm_server_errors"),
            # SCOPED 2026-07-27 (review finding 9). These two were computed over the 25-cell
            # llm_on_fix SLICE alone and read as a property of "the fixed arm", while the
            # report's directional claim leans on llm_on_fix_probe -- which carries 16 server
            # errors, 2 generator_healthy_after=False cells, and only 6/12 valid rows. A
            # witness at a narrower aggregation level than the claim it supports is the scope
            # error this project keeps making, so the key now names its own scope and the
            # whole-condition roll-up sits beside it.
            "n_server_errors_fix_SLICE_ARM_ONLY": fixed_live.get("total_llm_server_errors"),
            "fix_SLICE_arm_server_error_free": fix_clean,
            "fixed_condition_rollup_ALL_ARMS": analysis.get("fixed_condition_liveness_rollup"),
            "scope_correction": (
                "'The fixed arm was server-error free' is TRUE of the 25-cell llm_on_fix "
                "slice and FALSE of the fixed condition as a whole -- the probe arm run at "
                "the same n_ctx carries 16 server errors and 2 dead-generator cells. Read "
                "fixed_condition_rollup_ALL_ARMS before citing it."
            ),
            "server_failure_diagnostics_sample_16k": faulty_live.get(
                "server_failure_diagnostics_sample"
            ),
            "response_rate_16k": faulty_live.get("response_rate"),
            "response_rate_fix": fixed_live.get("response_rate"),
            "principle": (
                "A contention control that never exhibits the fault is inert: 'faulty == "
                "fixed' would then mean 'the fault did not fire', not 'the fault does not "
                "matter'. This block is the computed witness that the control was live."
            ),
        },
        "llm_engagement_witness": {
            "fix_arm_generator_engaged": fix_engaged,
            "fix_arm_total_llm_calls": fixed_live.get("total_llm_calls"),
            "fix_arm_total_llm_responses": fixed_live.get("total_llm_responses"),
            "fix_arm_n_ctx_observed": fixed_live.get("generator_n_ctx_observed"),
            "fix_arm_dead_generator_cells": fixed_live.get("dead_generator_cells"),
            "fix_arm_cells_missing_witness": fixed_live.get("n_cells_missing_witness"),
            "principle": (
                "Every prior LLM-on measurement in this project was taken at concurrency 1 "
                "and had no liveness channel, which is why a generator degraded to LLM-off "
                "was reported as the LLM-on scored path. These counters are that channel; "
                "a missing/zero value here would mean the arm is not evidence about the LLM."
            ),
        },
        # ------------------------------------------------------------ scope
        "scope_and_power": {
            # CORRECTED 2026-07-27 (review finding 8). This said "N cells per arm", which is
            # true only of the CONTROL. The arms have different n, and every comparison runs
            # at the SMALLEST of the pair -- so an unqualified "100 cells per arm" in the one
            # block a reader consults for power overstates the support by 4x.
            "corpus": (
                "25 PUBLIC ARC-AGI-3 games x colour-permuted held-out variants, played "
                "OFFLINE against environment_files -- the same corpus the 0.04 baseline is "
                "defined on. PER-ARM n IS NOT EQUAL: "
                + ", ".join(f"{k}={v.get('n_cells')}" for k, v in sorted(arms.items()))
                + ". The control ran all 4 variants x 25 games; the LLM arms ran the "
                "pre-specified variant-1 slice (25) and the control-winner probe (12). Every "
                "comparison is therefore at n_matched_pairs, NOT at the control's n."
            ),
            "n_cells_per_arm": {k: v.get("n_cells") for k, v in sorted(arms.items())},
            "concurrency": (
                "K=4 worker threads sharing ONE local llama-server. K=4 is the eval-relevant "
                "number because llama.cpp caps its own slot count at 4 and QUEUES the rest "
                "(server.cpp:106-110), so the generator sees at most 4 concurrent induce "
                "requests no matter how many games the framework's one-thread-per-game swarm "
                "starts. K=1 was deliberately NOT used: it is the blind spot that hid the "
                "fault from every prior measurement."
            ),
            "what_this_CANNOT_say": (
                "Nothing here forecasts the HIDDEN leaderboard. These are public games the "
                "project has fully solved and whose mechanics are in "
                "ops/arc_solve_registry.yaml, played offline, with no gateway. The hidden "
                "set is OOD by construction and our scored hardware is not even directly "
                "known (our kernel requests machine_shape NvidiaL4; no scored-run nvidia-smi "
                "exists). The answerable question is the narrow one: did the fault depress "
                "OUR OWN measured first-win rate on OUR OWN proxy?"
            ),
            # CORRECTED 2026-07-27 (review finding 8). The old text anchored the flip count
            # to "a 4/100 baseline" and quoted +/-0.06. But all three entries in `comparisons`
            # run at n_matched_pairs=25, so 6 flips is 6/25 = 0.24 absolute -- the block
            # understated its own MDE by 4x, in the one place a reader goes for power.
            "minimum_detectable_effect": (
                "The paired exact (McNemar) test's smallest reachable two-sided p is 2*0.5^n "
                "at n discordant pairs, so >=6 variants must flip in ONE direction to reach "
                "p<0.05 (2*0.5^6 = 0.03125). AT THIS MEASUREMENT'S ACTUAL SUPPORT -- every "
                "comparison runs at n_matched_pairs=25, not at the control's 100 -- 6 flips "
                "is 6/25 = 0.24 ABSOLUTE. Anything smaller is unresolvable here by "
                "construction and is reported as an effect size with its interval, never as "
                "a null. (The superseded text quoted +/-0.06 by dividing 6 by the CONTROL's "
                "100 cells, understating the MDE 4x.)"
            ),
            "minimum_detectable_effect_per_comparison": {
                c.get("comparison"): {
                    "n_matched_pairs": c.get("n_matched_pairs"),
                    "min_discordant_for_p_lt_0_05": 6,
                    "mde_absolute": (
                        round(6 / c["n_matched_pairs"], 4) if c.get("n_matched_pairs") else None
                    ),
                }
                for c in (analysis.get("comparisons") or [])
            },
            "mde_caveat_when_treatment_not_applied": (
                "These MDEs describe what the DESIGN could resolve. They are moot for any "
                "comparison whose treatment was never applied (see treatment_application): "
                "when both arms are the same agent, the achievable delta is exactly 0 at "
                "every n, so no sample size makes the comparison informative."
            ),
        },
        "preconditions_checked": {},  # filled below
        "field_principles": {
            "duration_s": {
                "principle": (
                    "Real compute takes wall-clock time; this is the MEASUREMENT clock summed from "
                    "per-cell row files, not the artifact builder's clock, so neither can "
                    "disguise the"
                    "other."
                )
            },
            "positive_control_passed": {
                "principle": (
                    "Gates every null claim in this artifact: a low LLM-on rate means nothing "
                    "unless a"
                    "control arm in the same tree demonstrably records wins."
                )
            },
            "contention_control_witness": {
                "principle": (
                    "Proves the pre-fix arm actually exhibited the fault, so a faulty-vs-fixed "
                    "null is"
                    "a statement about the fault's effect and not about the fault's absence."
                )
            },
            "llm_engagement_witness": {
                "principle": (
                    "An uninstrumented arm reads as a clean null; these counters make 'the "
                    "generator"
                    "answered' a measured fact rather than an assumption."
                )
            },
            "honest_verdict": {
                "principle": (
                    "Terminal-prefixed so the conductor reconciler classifies it without re- "
                    "running the"
                    "measurement."
                )
            },
            "random_seed": {
                "principle": (
                    "Determinism is the precondition for reproducibility; the bootstrap and the "
                    "agent's"
                    "own explore RNG are both seeded."
                )
            },
            "reproducibility_checksum": {
                "principle": (
                    "Content hash over the payload so silent drift between this artifact and a "
                    "replication attempt is detectable."
                )
            },
        },
        "random_seed": 4605,
        "model_specs": {
            "name": "Qwen3.5-9B-MTP",
            "model_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
            "model_filename": "Qwen3.5-9B-Q4_K_M.gguf",
            "mtp": True,
            "kv_quant": "q8_0",
            "no_think_prefix": True,
            "max_tokens": 4096,
            "llama_server_kind": "cuda build (~/.cache/llama.cpp-master/build/bin/llama-server)",
            "device": "RTX 3090 GPU 1 (outer-loop card); GPU 0 (conductor) untouched",
            "invoked": True,
        },
        "cited_upstream_artifacts": [
            {
                "experiment_id": 4605,
                "path": base["path"],
                "sha256": base["sha256"],
                "fields_imported": [
                    "first_win_rate_integrated",
                    "first_win_rate_bare",
                    "first_win_ci",
                    "integrated_measurement.variant_attempts",
                    "bare_control_config.llm_arm",
                ],
                "role": "the baseline being reproduced and re-measured against",
            }
        ],
        "row_files_dir": "results/first_win_llm_on_20260727/cells/",
        "analysis_path": "results/first_win_llm_on_20260727/analysis.json",
        "harness_paths": [
            "results/first_win_llm_on_20260727/firstwin.py",
            "results/first_win_llm_on_20260727/analyse.py",
            "results/first_win_llm_on_20260727/build_artifact.py",
        ],
        # DISCLOSED, NOT SILENCED. adversarial_verify raises a WARN
        # (value-routing-cost-control-omitted) on this artifact because it records
        # `submitted_value_weight` in its preconditions alongside "live_" text, which matches its
        # value-routing-context detector; it then asks for `per_node_feature_cost_ms` and
        # `sim_timed_out`. The flag is correct by its own rule and NON-APPLICABLE here, and the
        # honest response is to say why rather than to add fields that would silence it:
        "value_routing_claim_disclosure": {
            "does_this_artifact_attribute_anything_to_value_routing": False,
            "why_the_field_is_present_anyway": (
                "submitted_value_weight is recorded as PROVENANCE -- the parity-relevant config "
                "every arm actually ran with, read from the same SUBMITTED_VALUE_WEIGHT helper "
                "the baseline used. Removing it to dodge a linter would delete real evidence "
                "that the measured agent was the submitted config."
            ),
            "value_weight_used_in_every_arm": 1e-12,
            "why_no_value_routing_effect_is_possible_here": (
                "1e-12 is inert -- the baseline's own validator requires "
                "abs(value_weight) <= 1e-9 -- and it is IDENTICAL across all five arms. A term "
                "that is both inert and constant between arms cannot contribute to any "
                "between-arm delta, so there is no value-routing lift to control for."
            ),
            "per_node_feature_cost_ms_NOT_MEASURED_deliberately_renamed": (
                "NOT MEASURED by this harness, and deliberately NOT invented to clear the flag. "
                "This measurement times whole cells (elapsed_s per row), not per-node feature "
                "extraction."
            ),
            "sim_timed_out_status_deliberately_renamed": (
                "No cell timed out or errored: n_cell_errors is 0 in every arm and every cell "
                "ran its full 200-action budget or terminated on its own first-win break. "
                "Reported here as prose rather than as the bare field name the linter greps for, "
                "because the supportable claim is about cell completion, not about the "
                "simulator-timeout concept that check was written for."
            ),
            "flag_severity": "WARN, not CRITICAL -- it does not set flagged_adversarial and does "
            "not quarantine the result.",
            "why_these_two_keys_are_deliberately_RENAMED": (
                "A first draft of this disclosure used the exact key names "
                "`per_node_feature_cost_ms` and `sim_timed_out`, and the WARN CLEARED -- "
                "adversarial_verify went from 1 flagged to 0. That is because its "
                "`_real_field_values()` (scripts/adversarial_verify.py:4279) collects ANY "
                "value found under the wanted key name and only tests whether the list is "
                "non-empty; a PROSE SENTENCE therefore satisfies a check that is asking for "
                "a finite cost and a boolean. I had explicitly said I would not silence the "
                "flag, and had silenced it by accident. The keys are renamed so the WARN "
                "legitimately returns and is reported. This is also a real weakness in the "
                "QA layer worth surfacing on its own: any artifact can clear this check with "
                "a sentence, so the check cannot currently distinguish a reported control "
                "from an unreported one."
            ),
            "STATUS_UPDATE_2026_07_27_the_QA_hole_is_now_FIXED": (
                "`scripts/adversarial_verify.py` gained `_typed_field_values()`: the four "
                "checks that used bare non-emptiness (value-routing cost control, QD "
                "random-mutation ablation, L2 goal satisfiability, multi-level metric "
                "harness) now require a value of the ASKED-FOR TYPE, recovered through leaf "
                "and principle-wrapper unwrapping so honest per-game dicts and "
                "{'principle','value'} annotations still satisfy them. Regression test: "
                "tests/python/test_adversarial_verify_typed_field_values.py, which uses the "
                "LITERAL prose strings that cleared the WARN. Verified corpus-neutral: "
                "re-running all four checks with the before/after code over 14,847 result "
                "JSONs changed ZERO verdicts. A first attempt at this fix demanded a BARE "
                "bool and over-fired on the real per-game shape "
                "`goal_predicate_satisfiable: {'lp85': true}` -- caught by "
                "tests/python/test_adversarial_verify_hardening_4671.py before shipping, and "
                "now pinned by its own regression test."
            ),
            "consequence_for_THIS_artifact": (
                "The WARN below is now UNCLEARABLE BY PROSE and is left standing on purpose. "
                "It is correct: this measurement times whole cells, not per-node feature "
                "extraction, so it genuinely does not report per_node_feature_cost_ms. "
                "Nothing here attributes anything to value routing (see "
                "does_this_artifact_attribute_anything_to_value_routing: False)."
            ),
        },
        # STALENESS WAS REPORTED AND RESOLVED BY RE-MEASUREMENT, not by rebuilding blind.
        "post_commit_staleness_resolution": {
            "what_happened": (
                "Every measurement cell finished at 10:14 local. Commit 776161963 (THE FIX "
                "itself) landed at 10:19 and its pre-commit hooks rewrote the bytes of "
                "arc_competition_agent.py and arc_executable_world_model.py, so "
                "artifact_freshness_lint and summarize_artifact correctly reported this "
                "artifact STALE against the code now on disk."
            ),
            "why_inspection_was_not_enough": (
                "The fix is verifiably intact (_default_induce_n_ctx still returns 81920, the "
                "liveness witness is still present, git diff against HEAD is empty). But that is "
                "an inspection of the code, not a measurement of its behaviour, and this "
                "project's own rule is that a rebuild must be DIFFED with any moved number "
                "reported rather than waved through."
            ),
            "recheck": "results/first_win_llm_on_20260727/recheck_after_commit.json",
            "n_cells_rechecked": 10,
            "n_cells_identical": 10,
            "n_numbers_that_moved": 0,
            "recheck_coverage": (
                "All 7 win-carrying cells (every cell that could move the headline rate) plus 3 "
                "non-winners as an over-fire control, so a spuriously-appearing NEW win would "
                "also have been caught. first_win, reached_level, actions and "
                "actions_to_first_levelup all bit-identical on 10/10."
            ),
            "residual_limit": (
                "LLM-OFF arm only (seconds, not an hour) and 10 of 174 cells. The argument that "
                "this covers the llm_on arms too is that the generator's output reached the "
                "policy in 0 of 74 LLM-on cells, so those trajectories were produced by the same "
                "code path this recheck exercises -- an argument, not a proof, and labelled as "
                "one."
            ),
        },
        "no_pivot_recommendation": (
            "This artifact deliberately makes NO recommendation about reallocating the "
            "research programme. It produces the number and its uncertainty; the "
            "reallocation is the operator's call."
        ),
    }

    # preconditions, read from what the runs actually recorded (never asserted)
    runs = runs_raw
    pc = {}
    for name, r in runs.items():
        srv = r.get("server") or {}
        dev = srv.get("device") or {}
        pc[name] = {
            "launch_ok": srv.get("launch_ok", None if "llm_off" not in name else True),
            "device_verdict": dev.get("verdict"),
            "server_vram_mib": dev.get("my_vram_mib"),
            "props_total_slots": (srv.get("props") or {}).get("total_slots"),
            "props_n_ctx_reported": (srv.get("props") or {}).get("n_ctx_reported"),
            "server_healthy_after": r.get("server_healthy_after"),
            "teardown": r.get("teardown"),
            "gpu_after_teardown": r.get("gpu_after_teardown"),
            "k_concurrency": r.get("k_concurrency"),
            "n_cells": r.get("n_cells"),
            "deepen_enabled": r.get("deepen_enabled"),
            "submitted_target_levels": r.get("submitted_target_levels"),
            "submitted_value_weight": r.get("submitted_value_weight"),
        }
    payload["preconditions_checked"] = pc

    # ------------------------------------------------------------------ verdict
    # ------------------------------------------------------------------ acceptance gates
    # These gate MEASUREMENT VALIDITY, not the outcome. The deliverable is a number, so a gate
    # on "the number came out favourable" would be nonsense; a gate on "this number is
    # evidence about the system at all" is exactly what was missing from every prior LLM-on
    # measurement here. Each one could genuinely fail, and each is computed from a read of the
    # real object (the witness rows, the server's own /props) rather than from an assumption.
    llm_arms = {k: v for k, v in arms.items() if k.startswith("llm_on")}
    n_ctx_agree = []
    for arm_name in sorted(llm_arms):
        witnessed = (llm_arms[arm_name].get("liveness") or {}).get("generator_n_ctx_observed") or []
        props = (
            ((runs_raw.get(f"run_{arm_name}.json") or {}).get("server") or {}).get("props") or {}
        ).get("n_ctx_reported")
        n_ctx_agree.append(bool(witnessed) and props is not None and set(witnessed) == {props})
    payload_gates = {
        "acceptance_gate_positive_control_passed": positive_control_passed,
        "acceptance_gate_fix_arm_generator_engaged": bool(
            fix_engaged and int(fixed_live.get("dead_generator_cells") or 0) == 0
        ),
        "acceptance_gate_contention_control_exhibited_the_fault": bool(fault_exhibited),
        "acceptance_gate_no_uninstrumented_llm_cells": bool(
            llm_arms
            and all(
                int((v.get("liveness") or {}).get("n_cells_missing_witness") or 0) == 0
                for v in llm_arms.values()
            )
        ),
        "acceptance_gate_declared_n_ctx_matches_server_props": bool(
            n_ctx_agree and all(n_ctx_agree)
        ),
        "acceptance_gate_no_cell_errors": bool(
            all(int(v.get("n_cell_errors") or 0) == 0 for v in arms.values())
        ),
        # ADDED 2026-07-27 (review, FATAL finding). THE MISSING CONJUNCT: every gate above
        # tests the MEASUREMENT APPARATUS (control records wins, generator answered, control
        # arm overflowed, witnesses present, n_ctx agrees, no crashes) and all six passed --
        # while the thing being measured never actually happened. A gate set that can be
        # fully green on a comparison of an arm against itself is not gating the claim.
        "acceptance_gate_treatment_was_applied": bool(applied_any),
        # ADDED 2026-07-27 (review finding 9). The fix-arm liveness gate above was computed
        # over the 25-cell slice ONLY, so the 12-cell probe arm -- which carries the report's
        # directional claim, 16 server errors and 2 dead-generator cells -- was covered by NO
        # gate at all. This one spans every arm run at the fixed n_ctx.
        "acceptance_gate_every_fixed_condition_arm_generator_alive": bool(
            int(
                (analysis.get("fixed_condition_liveness_rollup") or {}).get(
                    "n_dead_generator_cells"
                )
                or 0
            )
            == 0
        ),
    }
    payload_gates["acceptance_gate_measurement_valid"] = all(payload_gates.values())
    payload.update(payload_gates)
    payload["measurement_validity_rationale"] = {
        "acceptance_gate_positive_control_passed": (
            "the control arm demonstrably records a first_win, so a low LLM-on rate is not "
            "just a broken win detector"
        ),
        "acceptance_gate_fix_arm_generator_engaged": (
            "the fixed generator answered and no cell was asked-and-silent; an unengaged arm "
            "is not evidence about the LLM"
        ),
        "acceptance_gate_contention_control_exhibited_the_fault": (
            "the pre-fix arm actually overflowed, so a faulty-vs-fixed null is about the "
            "fault's effect and not the fault's absence"
        ),
        "acceptance_gate_no_uninstrumented_llm_cells": (
            "a missing witness reads as a clean null; every LLM-on cell must carry one"
        ),
        "acceptance_gate_declared_n_ctx_matches_server_props": (
            "the whole fault was a declared-versus-actual gap; two independent reads must agree"
        ),
        "acceptance_gate_no_cell_errors": (
            "a crashed cell excluded silently would bias the rate; there must be none"
        ),
        "acceptance_gate_treatment_was_applied": (
            "the LLM's induced world model must actually reach the policy on at least one "
            "cell; if it never does, both arms are the same agent and every delta, p-value "
            "and CI is an identity rather than a measurement"
        ),
        "acceptance_gate_every_fixed_condition_arm_generator_alive": (
            "a liveness gate scoped to one arm cannot certify a claim that leans on another; "
            "this spans every arm run at the fixed n_ctx, including the probe"
        ),
    }

    fw = fix.get("first_win_rate")
    bl = base["first_win_rate_integrated"]
    c_fix_off = comps.get("llm_on_fix_vs_llm_off", {})
    c_fix_faulty = comps.get("llm_on_fix_vs_llm_on_16k_FAULTY", {})
    m = c_fix_faulty.get("mcnemar_exact", {})
    payload["headline"] = {
        # Every rate is labelled with the N it is over. The LLM-off control ran the full
        # N=100 (so it can be checked against the published baseline cell-for-cell); the LLM
        # arms ran the 25-game variant-1 slice (see RUN_LOG.md for why, and for the wall-clock
        # measurement that forced it). The matched comparisons below are computed ONLY on the
        # per-variant intersection, never on an any-arm union.
        "n_cells_llm_off_arm": off.get("n_cells"),
        "n_cells_llm_on_fixed_arm": fix.get("n_cells"),
        "n_cells_llm_on_faulty_arm": faulty.get("n_cells"),
        "n_matched_pairs_fixed_vs_faulty": c_fix_faulty.get("n_matched_pairs"),
        "n_matched_pairs_fixed_vs_off": c_fix_off.get("n_matched_pairs"),
        "matched_subset_llm_off_rate": c_fix_off.get("control_first_win_rate"),
        "first_win_rate_llm_on_fixed_generator_K4": fw,
        "first_win_rate_llm_on_fixed_generator_ci95": fix.get(
            "first_win_rate_ci95_clopper_pearson"
        ),
        "first_win_rate_llm_on_FAULTY_generator_K4": faulty.get("first_win_rate"),
        "first_win_rate_llm_off_control_K4": off.get("first_win_rate"),
        "baseline_first_win_rate_llm_off_K1": bl,
        "delta_fixed_vs_faulty": c_fix_faulty.get("delta"),
        "delta_fixed_vs_faulty_ci95": (c_fix_faulty.get("paired_bootstrap_delta") or {}).get(
            "ci95"
        ),
        "delta_fixed_vs_llm_off": c_fix_off.get("delta"),
        "delta_fixed_vs_llm_off_ci95": (c_fix_off.get("paired_bootstrap_delta") or {}).get("ci95"),
        "fault_effect_p_two_sided": m.get("p_two_sided"),
        "fault_effect_p_one_sided_fixed_better": m.get("p_one_sided_treatment_better"),
        "fault_effect_p_one_sided_faulty_better": m.get("p_one_sided_control_better"),
        "fault_effect_min_reachable_p": m.get("min_reachable_p_two_sided_at_this_support"),
        "fault_effect_significant_at_0_05": m.get("significant_at_0_05"),
        # THE REACHABLE-WIN CHECK. The variant-1 slice contains only ONE control win, so the
        # slice alone is a weak test of "does the LLM cost us a win". The probe measures EVERY
        # cell the control won, under BOTH generators, at K=4. held/lost only -- no rate, no
        # p-value, because these cells were selected on the control's outcome.
        "control_won_cells_total": len(
            (analysis.get("control_winner_probe") or {}).get("_control_win_set") or []
        ),
        "control_won_cells_HELD_by_fixed_generator": (
            (analysis.get("control_winner_probe") or {}).get("llm_on_fix") or {}
        ).get("n_held"),
        "control_won_cells_LOST_by_fixed_generator": (
            (analysis.get("control_winner_probe") or {}).get("llm_on_fix") or {}
        ).get("n_lost"),
        "control_won_cells_HELD_by_faulty_generator": (
            (analysis.get("control_winner_probe") or {}).get("llm_on_16k") or {}
        ).get("n_held"),
        "control_won_cells_LOST_by_faulty_generator": (
            (analysis.get("control_winner_probe") or {}).get("llm_on_16k") or {}
        ).get("n_lost"),
        "n_llm_on_cells_where_generator_output_reached_the_policy": sum(
            int((v.get("liveness") or {}).get("n_cells_llm_output_reached_the_policy") or 0)
            for k, v in arms.items()
            if k.startswith("llm_on")
        ),
        "n_llm_on_cells_total": sum(
            int(v.get("n_cells") or 0) for k, v in arms.items() if k.startswith("llm_on")
        ),
        # READ THESE THREE BEFORE ANY RATE ABOVE (2026-07-27 review, FATAL finding). They are
        # in the headline block deliberately: a reader who takes delta / p / CI from here
        # without them will read an arithmetic identity as a measured null.
        "TREATMENT_WAS_APPLIED": bool(applied_any),
        "every_llm_on_arm_is_bit_identical_to_its_matched_control": all(
            v.get("arm_is_bit_identical_to_control")
            for k, v in (analysis.get("treatment_application") or {}).items()
            if isinstance(v, dict)
        ),
        "how_to_read_the_deltas_above": (
            "If TREATMENT_WAS_APPLIED is False, every delta / p-value / CI in this block is "
            "an IDENTITY between an arm and itself, not a measurement of the fault's effect. "
            "The fault's first-win effect is then UNTESTED -- not absent, and not merely "
            "underpowered. See treatment_application_headline."
        ),
    }

    d = c_fix_faulty.get("delta")
    if d is not None and not applied_any:
        verdict = "complete_UNFALSIFIABLE_treatment_not_applied_llm_output_never_reached_policy"
    elif d is None:
        verdict = "blocked_incomplete_arms"
    elif m.get("significant_at_0_05") and (d or 0) > 0:
        verdict = f"complete_fault_depressed_first_win_by_{abs(d):.2f}_significant"
    elif m.get("significant_at_0_05") and (d or 0) < 0:
        verdict = f"complete_fixed_generator_first_win_lower_by_{abs(d):.2f}_significant"
    elif abs(d) < 1e-9:
        verdict = (
            "complete_fault_had_no_measurable_effect_on_first_win_rate"
            "_underpowered_exact_zero_delta"
        )
    else:
        verdict = f"complete_fault_effect_on_first_win_{d:+.2f}_underpowered_not_significant"

    # ---------------------------------------------------------------------------------
    # VACUOUS-TEST GUARD (added 2026-07-27, adversarial review). A null is only a finding if
    # the test could have produced something else. TWO computed conditions here say it could
    # not, and both must be read BEFORE any "no effect" wording is allowed to stand:
    #
    #   1. min_reachable_p == 1.0 -- the paired exact test's SMALLEST attainable two-sided p at
    #      this discordant-pair count. At 1.0 there is no outcome on this corpus that could have
    #      reached significance, so "not significant" carries zero information about the world.
    #   2. n_cells_where_llm_output_reached_the_policy == 0 -- with the FIXED generator, every
    #      induced world model is rejected by a POST-generation gate and no plan is ever
    #      installed. The LLM's output never reaches action selection, so a zero first-win delta
    #      is what the pipeline MUST produce regardless of generator health. The channel the
    #      measurement is about is structurally empty.
    #
    # This is exactly the FALSE_NEGATIVE_RISK shape CLAUDE.md names ("a NULL claim is NOT a
    # finding unless a positive control passed"): the arm's positive control shows the WIN
    # DETECTOR works, but nothing shows the LLM->policy channel works, and that is the channel
    # under test. So the verdict is downgraded to name the degeneracy rather than reporting a
    # clean null. The measurement is preserved unchanged; only the claim it licenses shrinks.
    min_p = m.get("min_reachable_p_two_sided_at_this_support")
    reached = (payload.get("mechanism_downstream_of_the_generator") or {}).get(
        "n_cells_where_llm_output_reached_the_policy"
    )
    degenerate_reasons = []
    if isinstance(min_p, (int, float)) and min_p >= 1.0:
        degenerate_reasons.append("min_reachable_p_1.0_no_outcome_could_have_been_significant")
    if reached == 0:
        degenerate_reasons.append("llm_output_reached_the_policy_in_0_cells")
    if degenerate_reasons and verdict.startswith("complete_fault_had_no_measurable_effect"):
        verdict = "complete_fault_effect_UNRESOLVED_degenerate_test_" + "_and_".join(
            degenerate_reasons
        )
    payload["honest_verdict"] = verdict
    payload["false_negative_risk"] = {
        "degenerate": bool(degenerate_reasons),
        "reasons": degenerate_reasons,
        "min_reachable_p_two_sided": min_p,
        "n_cells_where_llm_output_reached_the_policy": reached,
        "what_this_means": (
            "The zero first-win delta between the faulty and fixed generator is NOT evidence "
            "that the concurrency fault had no effect. With the fixed generator, the induced "
            "world model is rejected by a post-generation trust/accuracy gate on every single "
            "cell, so the generator's output never reaches action selection in EITHER arm. A "
            "zero delta is the pipeline's structural output at default settings, and the "
            "paired exact test's smallest attainable two-sided p on this corpus is 1.0 -- the "
            "test could not have come out any other way."
        ),
        "positive_control_that_is_missing": (
            "A run in which LLM output actually reaches action selection -- e.g. relaxing the "
            "post-generation gate (CARNOT_ARC_TRUST_METRIC=cell_recall) or lowering the "
            "world-model accuracy floor -- so that n_cells_where_llm_output_reached_the_policy "
            "> 0. Until that exists, the effect of the fault on first-win rate is UNRESOLVED, "
            "not measured to be zero."
        ),
        "what_IS_established_here": (
            "Independently of the degenerate delta: (a) the 0.04 baseline was itself an LLM-OFF "
            "measurement (_NoOpProposer in both exp4605 arms), so the fault cannot have "
            "depressed it -- there was no generator contribution to lose; (b) the fault "
            "genuinely fired in the 16k arms and is absent from the fixed arms for mode A "
            "(n_context_exceeded 36 -> 0). Those two do not depend on the delta."
        ),
    }

    payload["null_delta_methodology_note"] = (
        "Any zero delta reported here is an HONEST NULL, gated by two computed controls, not "
        "a measurement bug: (1) positive_control_passed shows the llm_off arm reproduces the "
        "baseline rate AND its exact winning variants in this tree, so the win detector "
        "works; (2) contention_control_witness shows the pre-fix arm genuinely exhibited the "
        "fault (non-zero server errors with the server's own 'Context size has been "
        "exceeded.' body) while the fixed arm had none, so the control was live. Power is "
        "stated explicitly: with n discordant pairs the smallest reachable two-sided exact p "
        "is 2*0.5^n, so a non-significant result on few discordant pairs is reported as an "
        "effect size with an interval, never as evidence of no effect."
    )
    payload["false_negative_risk_checked"] = True
    # PROVENANCE, in the shape scripts/artifact_freshness_lint.py verifies: every CODE file
    # whose change could invalidate these numbers, plus every row file they were computed
    # from. Registered so that touching any dependency makes this artifact provably stale and
    # forces a rebuild, rather than leaving a silently-drifted number in the record.
    code_deps = [
        "results/first_win_llm_on_20260727/firstwin.py",
        "results/first_win_llm_on_20260727/analyse.py",
        "results/first_win_llm_on_20260727/build_artifact.py",
        "results/first_win_llm_on_20260727/guardtest.py",
        "results/first_win_llm_on_20260727/recheck_after_commit.py",
        "python/carnot/experiment_4605_live_integration_scored_agent.py",
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_executable_world_model.py",
    ]
    payload["provenance"] = {
        "analyzer": "results/first_win_llm_on_20260727/build_artifact.py",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": [
            {"path": p, "sha256": hashlib.sha256((REPO / p).read_bytes()).hexdigest()}
            for p in code_deps
            if (REPO / p).exists()
        ],
        "rows_sources": [
            {
                "path": str(f.relative_to(REPO)),
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            }
            for f in sorted((OUT / "cells").glob("*.json"))
        ],
    }
    payload["artifact_build_s"] = round(time.time() - t0, 3)
    payload["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
    )
    ART.write_text(json.dumps(payload, indent=1, default=str))
    # REGISTER IN ops/analyzer_artifact_index.json (2026-07-27 review finding 13). The
    # artifact carries its own `provenance` block, and summarize_artifact.py does verify it --
    # but the COMMIT-TIME guard (scripts/artifact_freshness_lint.py, wired as the
    # artifact-freshness-lint pre-commit hook) only checks artifacts that appear in this
    # index. Unregistered, the hook printed OK while saying nothing about this artifact, so
    # an edit to arc_competition_agent.py -- which the FIX commit made 5 minutes after the
    # last cell -- could silently leave it stale. "Not listed" is not "checked and fresh".
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from analyze_scored_path_lever_ab import register_analyzed_artifact

        register_analyzed_artifact(ART, analyzer=Path(__file__))
        print("registered in ops/analyzer_artifact_index.json")
    except Exception as exc:  # never let bookkeeping lose the artifact itself
        print(f"WARNING: index registration failed ({exc!r}); artifact still written")
    print("wrote", ART)
    print(json.dumps(payload["headline"], indent=1))
    print("verdict:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
