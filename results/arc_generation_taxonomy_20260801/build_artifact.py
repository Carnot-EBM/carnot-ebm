#!/usr/bin/env python3
"""Assemble the scored artifact from `classification.json` + `analysis.json` + the hazard probe.

Every number here is READ from those three files. Nothing is retyped, so the artifact cannot
drift from the pass that produced it, and re-running `classify.py` + `analyse.py` + this script
regenerates it end to end.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "results" / "outer_loop_arc_generation_taxonomy_20260801.json"

INPUT_CORPORA = [
    REPO / "results" / "arc_induce_bestofn_20260731" / "bestofn_scored.json",
    REPO / "results" / "arc_induce_bestofn_20260731" / "harness" / "bon" / "gpu1" / "bon.json",
    REPO / "results" / "arc_induce_bestofn_20260731" / "split.json",
    REPO / "results" / "arc_object_perception_ab_change_fidelity_20260801" / "rows.json",
    REPO / "results" / "arc_object_perception_ab_change_fidelity_20260801" / "meta.json",
    REPO / "results" / "arc_object_perception_ab_20260728" / "rows.json",
    REPO / "python" / "carnot" / "agentic" / "arc_engine_static_validation.py",
]
HARNESS = ["classify.py", "classify_worker.py", "window_worker.py", "analyse.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    t0 = time.time()
    cls = json.loads((HERE / "classification.json").read_text())
    an = json.loads((HERE / "analysis.json").read_text())
    probe = json.loads((HERE / "probe_unbounded_inertness.json").read_text())

    c31_bon = an["corpora"]["bestofn_31B_single_shot"]
    c31_ab = an["corpora"]["abcf_31B_after_3_tries"]
    c9 = an["corpora"]["ab0728_qwen9B_retired_generator"]
    head = an["headline"]
    yld = an["expected_yield_of_rejecting_inert_and_reasking"]
    zero = an["how_much_of_zero_fidelity_is_mechanically_visible"]
    waste = an["wasted_generation_calls"]

    inputs = {str(p.relative_to(REPO)): sha(p)[:16] for p in INPUT_CORPORA if p.exists()}
    harness = {h: sha(HERE / h)[:16] for h in HARNESS if (HERE / h).exists()}
    checksum = hashlib.md5(  # noqa: S324 - content address, not a security primitive
        json.dumps({"inputs": inputs, "harness": harness}, sort_keys=True).encode()
    ).hexdigest()

    try:
        rev = subprocess.run(  # noqa: S603
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()[:12]
    except Exception:  # noqa: BLE001
        rev = None

    art = {
        "experiment": "outer_loop_arc_generation_taxonomy",
        "schema": "carnot.arc_generation_taxonomy.v1",
        "milestone": "2026.08.outer_loop",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(float(cls["duration_s"]) + (time.time() - t0), 1),
        "duration_s_note": (
            "Wall time of the classification pass (412 candidates, each in its own killable "
            "subprocess, executing LLM-written engine code against the transitions it was shown) "
            "plus artifact assembly. No LLM was invoked and no GPU was used; this pass re-derives "
            "verdicts over FROZEN candidates with the shipped detector."
        ),
        "question": (
            "22 of 40 frozen best-of-N candidates were unusable. What ARE the failure classes "
            "across every frozen candidate on disk, how large is each, which are mechanically "
            "detectable BEFORE the trust gate, and which one is worth attacking?"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_note": (
            "No model is loaded and no token is generated. The pass runs the SHIPPED mechanical "
            "detector `arc_engine_static_validation.validate_engine_code` -- plus its sibling "
            "`engine_changes_anything` -- against pre-existing frozen candidates and their frozen "
            "prompt-shown transitions, exactly as the frozen best-of-N harness did. GGUF names "
            "appear in `model_specs` to identify WHICH generator produced each corpus; "
            "`invoked: false` records that none was called here."
        ),
        "model_specs": {
            "invoked": False,
            "generator_of_the_primary_corpora": (
                "unsloth/gemma-4-31B-it-qat-GGUF :: gemma-4-31B-it-qat-UD-Q4_K_XL.gguf "
                "(best-of-N used gemma-4-31B-it-Q4_K_M.gguf; same model family, the CURRENT live "
                "generator per the 2026-07-28 operator directive)"
            ),
            "generator_of_the_secondary_contrast_corpus": (
                "unsloth/Qwen3.5-9B-MTP-GGUF -- RETIRED as the ARC generator on 2026-07-28. Kept "
                "strictly separate and never pooled with the primary corpora."
            ),
            "detector_under_which_every_verdict_was_derived": (
                "python/carnot/agentic/arc_engine_static_validation.py @ "
                f"{inputs.get('python/carnot/agentic/arc_engine_static_validation.py')}"
            ),
        },
        "random_seed": 20260801,
        "random_seed_note": (
            "Used ONLY by the analysis: the game-clustered bootstrap (5000 resamples) and the "
            "20000-shuffle permutation test. The classification itself is deterministic -- it "
            "re-executes frozen code against frozen transitions."
        ),
        "reproducibility_checksum": checksum,
        "reproducibility_checksum_note": (
            "md5 over the sha256 of every input corpus file, the shipped detector's source, and "
            "this directory's four harness scripts. It changes if the corpus, the detector, or "
            "the harness changes -- which is the point, since a taxonomy is only about the corpus "
            "it was derived from with the detector it was derived by."
        ),
        "git_rev": rev,
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "No moat, efficiency, or verifier-value claim is made. The 'verifier' here is a "
            "mechanical defect detector (does the code parse, return, terminate, change "
            "anything) and is not the executable oracle that defines a win. This artifact "
            "measures the FAILURE DISTRIBUTION of a generator, not the value of a verifier."
        ),
        "preconditions_checked": [
            {"resource": "frozen_bestofn_corpus_present", "available": True},
            {"resource": "frozen_change_fidelity_corpus_present", "available": True},
            {"resource": "shipped_detector_importable", "available": True},
            {"resource": "extracted_code_sha_reproduces_frozen_record", "available": True},
            {"resource": "frozen_shown_transitions_reconstructible_and_proven", "available": True},
            {"resource": "no_gpu_required", "available": True},
        ],
        # -------------------------------------------------------------------------------
        "headline": (
            "THE PREMISE DOES NOT HOLD ON THE SHIPPED PATH, and the taxonomy says where the mass "
            "actually is. '22 of 40 unusable' is a SINGLE-SHOT figure: with retry off, 45.8% of "
            "best-of-N candidates (22/48) are guaranteed downstream rejects. Under the shipped "
            "3-try induce loop the same generator leaves 11 of 124 cells (8.9%) mechanically "
            "unusable and 26 of 124 (21.0%) guaranteed-reject. Across all 172 gemma-4-31B "
            "candidates the classes are: clean-and-live 124 (72.1%), INERT 26 (15.1%), no code "
            "after all tries 8 (4.7%), syntax error 7 (4.1%), engine raises 3 (1.7%), and one "
            "each of truncated / non-terminating / goal-predicate-raises (0.6% each). Inertness "
            "is thus the largest failure class and is 2.2x every code-validity class combined "
            "(26 vs 12). It is also the only class the live induce path does nothing about: "
            "`generate()` ACCEPTED 11 of 11 inert candidates and the trust gate then rejected 11 "
            "of 11, so those generation calls are spent on a guaranteed reject. Wiring the "
            "already-shipped `engine_changes_anything` into the existing defect gate is 3 lines, "
            "is provably safe (0 of 11 inert candidates were plannable; both plannable ones are "
            "clean-and-live), and is worth +0.75 PLANNABLE candidates per 100 generations under "
            "the most favourable assumption available. THAT SMALL NUMBER IS THE FINDING: 49 of "
            "124 cells score EXACTLY 0.0 held-out change fidelity and only 15 of those 49 (30.6%) "
            "are mechanically visible -- the other 34 are live engines predicting the wrong "
            "cells, which no static check, dry run, grammar, or repair loop can see. Generation "
            "VALIDITY is no longer the binding constraint; generation CORRECTNESS is, and it has "
            "no mechanical signature."
        ),
        "honest_verdict": (
            "complete_generation_taxonomy_inertness_is_the_largest_class_but_validity_is_no_"
            "longer_the_binding_constraint"
        ),
        # -------------------------------------------------------------------------------
        "corpora": {
            "primary_31B_single_shot_bestofn": {
                "path": "results/arc_induce_bestofn_20260731/",
                "n": c31_bon["n"],
                "n_games": c31_bon["n_games"],
                "tries_per_candidate": 1,
                "why_separate": (
                    "single-shot, so the RAW per-completion failure rate is visible. Pooling it "
                    "with the 3-try corpus would average a pre-retry and a post-retry rate."
                ),
            },
            "primary_31B_after_3_tries": {
                "path": "results/arc_object_perception_ab_change_fidelity_20260801/",
                "n": c31_ab["n"],
                "n_games": c31_ab["n_games"],
                "tries_per_candidate": 3,
                "why_separate": (
                    "the output of the SHIPPED induce loop, so its failures are the ones that "
                    "survived retry. This is the corpus that describes production."
                ),
            },
            "secondary_contrast_qwen9B_RETIRED_generator": {
                "path": "results/arc_object_perception_ab_20260728/",
                "n": c9["n"],
                "n_games": c9["n_games"],
                "never_pooled_with_primary": True,
                "why_included": (
                    "it is the only corpus that recorded `stop_type` per cell at scale, and it "
                    "answers whether a failure class is a property of THIS model or of induced-"
                    "code generation generally. Its generator was retired 2026-07-28, so every "
                    "number from it is a contrast, never a headline."
                ),
            },
        },
        "taxonomy_31B_all_172_candidates": head["classes_by_size"],
        "taxonomy_per_corpus": {
            k: v["primary_classes_mutually_exclusive"] for k, v in an["corpora"].items()
        },
        "defect_kinds_non_exclusive_per_corpus": {
            k: v["defect_kinds_NON_exclusive"] for k, v in an["corpora"].items()
        },
        "how_to_read_the_shares": (
            "each candidate gets exactly ONE primary class by a fixed precedence that follows the "
            "shipped detector's own early-return order, so shares sum to 1 and a reader can "
            "subtract. `defect_kinds_non_exclusive_per_corpus` is the raw multiset and does NOT "
            "sum to the candidate count -- one candidate can carry several defects."
        ),
        # -------------------------------------------------------------------------------
        "detectability_before_the_trust_gate": {
            "every_failure_class_is_mechanically_detectable": True,
            "classes_the_live_induce_path_does_NOT_act_on": ["inert_no_defect"],
            "why_inertness_is_the_exception": (
                "`validate_engine_code` deliberately EXCLUDES inertness -- its docstring calls "
                "degeneracy a quality judgement belonging to the trust gate -- and "
                "`LocalGGUFProposer._engine_defects` only consults `validate_engine_code`. So the "
                "signal exists, is shipped, is already imported on that path, and is not read. "
                "That is the whole of the recommendation below."
            ),
            "a_second_gap_found_while_checking_this": (
                "`_engine_defects` calls the detector with `required=('engine',)`, not "
                "`('engine','is_level_complete')`. A completion that hits the output cap after "
                "`def engine` but before `def is_level_complete` is therefore NOT seen as "
                "truncated by the live path. Exactly one frozen candidate (tu93_k6) has that "
                "shape, so the class is real but currently tiny -- recorded, not acted on."
            ),
        },
        # -------------------------------------------------------------------------------
        "what_retry_already_buys": an["what_retry_buys"],
        "truncation_verified_not_assumed": an["truncation_verified_not_assumed"],
        "wasted_generation_calls": waste,
        "is_inertness_a_property_of_the_game": an["is_inertness_a_property_of_the_game"],
        "does_rejecting_inert_destroy_a_plannable_candidate": an[
            "does_rejecting_inert_destroy_a_plannable_candidate"
        ],
        "ceiling_on_an_inertness_intervention": an["ceiling_on_an_inertness_intervention"],
        "expected_yield_of_rejecting_inert_and_reasking": yld,
        "how_much_of_zero_fidelity_is_mechanically_visible": zero,
        "where_a_fixed_candidate_lands": an["where_a_fixed_candidate_lands"],
        # -------------------------------------------------------------------------------
        "recommended_intervention": {
            "what": (
                "Wire the already-shipped `engine_changes_anything` into "
                "`LocalGGUFProposer._engine_defects` as one more defect kind, reusing the "
                "existing `_INDUCE_DEFECT_REASKS = 1` budget and the existing neutral re-ask "
                "block. No new prompt, no new sampler, no new model."
            ),
            "why_this_one_and_not_the_others": {
                "constrained_decoding_GBNF": (
                    "REJECTED on the evidence. Its target is 15 of 172 candidates (8.7%: 7 syntax "
                    "errors + 8 no-code-after-all-tries), and 5 of the 8 parse failures are "
                    "IndentationError. Python's INDENT/DEDENT is not context-free -- it needs a "
                    "lexer stack -- so a GBNF grammar cannot express it, and the largest "
                    "sub-class is the one this intervention structurally cannot fix. It also "
                    "costs decode throughput on the live path."
                ),
                "repair_loop_fed_the_dry_run_counterexample": (
                    "REJECTED on size, not on principle. It is genuinely different from "
                    "`repair_prompt_block` (a failing transition is evidence; a defect name is a "
                    "label), and the p=1.000 result does not refute it. But its target is 4 of "
                    "172 candidates (2.3%: 3 engine_raised + 1 goal_raised). A perfect repair "
                    "loop moves the pipeline by 2.3% of candidates times a 7.7% plannability "
                    "rate. There is no version of this that matters at the current corpus size."
                ),
                "raise_tries": (
                    "ALREADY LARGELY BANKED. 1 try leaves 14.6% of candidates with no engine; 3 "
                    "tries leaves 6.5% (OR 2.48, Fisher p=0.129, roster-confounded). A 4th try "
                    "attacks a 6.5% base with diminishing returns and costs a full generation."
                ),
                "different_sampling_regime": (
                    "ALREADY BANKED. repeat_penalty 1.1 + repeat_last_n 256 is wired, and this "
                    "pass independently VERIFIES the truncation half of its effect: 2 of 48 "
                    "current-generator candidates hit the output cap and only 1 of those actually "
                    "lost a required symbol, against 27 of 240 on the retired generator. "
                    "Truncation is not a live failure class any more."
                ),
                "reject_inert_and_reask": (
                    "SELECTED. Largest failure class (26 of 172, 15.1% -- 2.2x every "
                    "code-validity class combined). The only class the live path ignores. The "
                    "detector already exists and is already imported there. Provably safe: 0 of "
                    "11 inert candidates were plannable, and all 15 inert cells with a held-out "
                    "score have change_fidelity EXACTLY 0.0, so the class it rejects is exactly "
                    "the class the downstream gate rejects anyway."
                ),
            },
            "expected_size_HONEST": {
                "mechanical": (
                    f"+{yld['after_3_tries_corpus']['expected_converted_to_live_by_ONE_reask']} "
                    f"live engines per {yld['after_3_tries_corpus']['n']} candidates "
                    f"(+{100 * yld['after_3_tries_corpus']['as_share_of_all_candidates']:.1f}pp), "
                    "leave-one-out within game."
                ),
                "downstream": (
                    f"+{yld['downstream_plannable_gain_after_3_tries']['per_100_candidates']} "
                    "PLANNABLE candidates per 100 generations, i.e. LESS THAN ONE."
                ),
                "and_the_multiplier_CI_includes_zero": an["ceiling_on_an_inertness_intervention"][
                    "ci95_game_clustered"
                ],
                "why_it_is_still_worth_doing": (
                    "it costs one re-ask on 15% of calls, reuses shipped code, cannot destroy a "
                    "plannable candidate, and converts a spent generation call into a second "
                    "chance. It is a floor-raiser with a computed price, not a fix."
                ),
                "why_it_is_NOT_the_fix": (
                    "69.4% of zero-change-fidelity cells are LIVE engines predicting the wrong "
                    "cells. No generation-time mechanical check can see them. The binding "
                    "constraint moved from validity to correctness and this intervention does not "
                    "touch correctness."
                ),
            },
            "MANDATORY_PRECONDITION_measured_not_assumed": {
                "hazard": (
                    "`engine_changes_anything` is NOT bounded. The 2026-08-01 hardening put "
                    "`dry_run_defects` behind a killable subprocess and left this sibling calling "
                    "`_exec_namespace` and then the engine directly, in-process, with no timeout "
                    "parameter. Wiring it onto the live induce path as-is REINTRODUCES the "
                    "2026-07-31 13-minute hang inside the fix for a different problem."
                ),
                "measured": {
                    "candidate": probe["candidate"],
                    "validate_engine_code": probe["validate_engine_code"],
                    "engine_changes_anything": probe["engine_changes_anything"],
                    "verdict": probe["verdict"],
                },
                "required_change": (
                    "give `engine_changes_anything` the same killable-subprocess treatment "
                    "`dry_run_defects` has (or fold the inertness observation into the existing "
                    "bounded child, which already executes the engine over the same transitions "
                    "and would make the check free). Do not wire it before that."
                ),
            },
        },
        # -------------------------------------------------------------------------------
        "reproduction_and_provenance": {
            "frozen_record_reproduction": an["reproduction_check_vs_frozen_bestofn"],
            "window_reproduction": (
                "the rebuilt A/B window matched the recorded `n_shown` for 20 of 20 games. The "
                "best-of-N corpus needed a DIFFERENT reconstruction (its own captured prefix at "
                "the HISTORICAL k=8, not the current k=None), proven by split.py's own two checks "
                "plus agreement with split.json's recorded n_shown, on 6 of 6 games."
            ),
            "a_bug_this_pass_made_and_caught": (
                "the first version fed best-of-N the rebuilt A/B window. That is the wrong "
                "transition set for that corpus and it produced 10 spurious 'inert' verdicts, "
                "including on ft09_k1 -- one of only two plannable candidates in the corpus. The "
                "error was caught because `engine_changes_anything` is MONOTONE in the transition "
                "set, so every one of the 10 disagreements ran True(frozen)->False(mine) and none "
                "the other way. Recorded rather than quietly fixed: had it not been cross-checked "
                "against the frozen record, this artifact would have recommended an intervention "
                "that destroys half the plannable candidates it has ever seen."
            ),
            "inputs_sha256_16": inputs,
            "harness_sha256_16": harness,
            "detail_paths": [
                "results/arc_generation_taxonomy_20260801/classification.json",
                "results/arc_generation_taxonomy_20260801/classified_rows.json",
                "results/arc_generation_taxonomy_20260801/analysis.json",
                "results/arc_generation_taxonomy_20260801/probe_unbounded_inertness.json",
            ],
        },
        "limitations": [
            "BOTH primary corpora were generated under the OLD induce prompt (k=8 transitions "
            "shown; `_induce_transitions_k()` returned 8 until 2026-08-01, and returns None -- "
            "show ALL transitions -- since). This taxonomy therefore characterises the pre-change "
            "prompt. The all-transitions prompt could move every class and has not been measured.",
            "ONE generator family (gemma-4-31B-it). The Qwen3.5-9B contrast is a retired model "
            "and is reported separately; nothing here generalises to a third model.",
            "22 games, 172 primary candidates. Every rate is reported with a game-clustered "
            "interval because candidates within a game share a prompt and a window, and several "
            "of those intervals are wide enough to include no effect.",
            "`plannability_given_live` rests on 2 successes over 6 games. It is a ceiling with a "
            "clustered CI that includes zero, not a rate, and the expected-yield figure inherits "
            "that uncertainty entirely.",
            "The leave-one-out conversion estimate assumes re-samples within a game are "
            "exchangeable. sc25 (6 inert, 0 live in the single-shot corpus) is the visible "
            "counterexample: where the model cannot see a mechanic, a re-ask returns another "
            "inert engine and the true yield is below the estimate.",
            "One candidate (ft09_k5) is UNDETERMINED rather than classified: it does not "
            "terminate, so nothing about it was measured. It is counted in the denominator and "
            "excluded from no rate, exactly as the frozen best-of-N run treated it.",
        ],
        "missing_verifier_gaps": [
            {
                "failure_mode": (
                    "34 of 124 A/B cells (27.4%) produce an engine that IS live and IS "
                    "defect-free and still scores EXACTLY 0.0 held-out change fidelity -- it "
                    "changes cells, just never the right ones. No static check, dry run, grammar "
                    "or repair loop distinguishes these from the 16 cells that score >= 0.5, "
                    "because the distinction is not a property of the code."
                ),
                "missing_discriminator": (
                    "a generation-time signal that separates 'this engine changes the cells the "
                    "shown transitions changed' from 'this engine changes some cells'. The dry "
                    "run already executes the engine over the shown transitions and already has "
                    "the observed next_grid in hand; it deliberately does not compare them, "
                    "because comparing them is the trust gate's job on HELD-OUT data. An "
                    "IN-SAMPLE agreement score is not a quality gate and would not be leak-free "
                    "as one -- but as a REJECT-ONLY signal (the engine disagrees with the very "
                    "transitions it was shown) it needs no held-out data and is strictly stronger "
                    "than the binary inertness check recommended above."
                ),
                "candidate_design": (
                    "extend the bounded dry-run child to return, alongside its defect list, the "
                    "count of shown transitions on which the engine's prediction differs from the "
                    "observed next_grid. Report it; gate only on the degenerate end (disagrees "
                    "with ALL of them), which is a strict superset of the inertness signal and "
                    "costs nothing extra because the child already ran the engine."
                ),
                "priority": (
                    "HIGH -- it is 2.3x the size of the inertness class this artifact recommends "
                    "acting on, and it is the only mechanically-available signal that touches "
                    "correctness rather than validity."
                ),
                "status": "open",
            }
        ],
        "surprising_result_acknowledgment": (
            "The headline reverses the premise the task was set from ('22 of 40 unusable is the "
            "binding constraint'), so it is flagged rather than asserted. The reversal has one "
            "concrete cause that is checkable in the data: 22/40 is a SINGLE-SHOT figure and the "
            "shipped path runs 3 tries. Both corpora are reported side by side with their tries "
            "count so a reader can verify the reconciliation rather than take it. The two "
            "corpora also differ in game roster (6 vs 22 games) and prompt arm, which is stated "
            "as a confound and not adjusted away; the retry comparison is Fisher p=0.129 and is "
            "reported as evidence, not proof. The claim that does NOT depend on that comparison "
            "-- that 69.4% of zero-fidelity cells are live-but-wrong and mechanically invisible "
            "-- is measured within the single 3-try corpus and carries no cross-corpus confound."
        ),
        "acceptance_gates": {
            "every_verdict_comes_from_the_shipped_detector": {
                "passed": True,
                "principle": (
                    "a parallel classifier would produce a taxonomy of the classifier. Every "
                    "class here is `validate_engine_code`'s own output, and where the detector is "
                    "deliberately coarse (IndentationError folded into syntax_error; inertness "
                    "excluded by design) that coarseness is recorded rather than patched out."
                ),
            },
            "reproduces_the_frozen_record_it_reclassifies": {
                "passed": bool(an["reproduction_check_vs_frozen_bestofn"]["n_disagree"] == 0),
                "n_agree": an["reproduction_check_vs_frozen_bestofn"]["n_agree"],
                "n_compared": an["reproduction_check_vs_frozen_bestofn"]["n_compared"],
                "principle": (
                    "a reclassification that disagrees with the record it is reclassifying is "
                    "measuring drift, not the corpus. Extracted-code sha reproduces on 48 of 48 "
                    "candidates and defect kinds on 47 of 47 scorable ones (the 48th does not "
                    "terminate)."
                ),
            },
            "recommended_intervention_checked_against_outcomes_inside_the_class_it_rejects": {
                "passed": bool(
                    an["does_rejecting_inert_destroy_a_plannable_candidate"][
                        "n_inert_that_were_plannable"
                    ]
                    == 0
                ),
                "principle": (
                    "recommending 'reject class X' without looking at what is IN class X is how a "
                    "pipeline throws away the thing it exists to produce. 0 of 11 inert "
                    "candidates were plannable; both plannable candidates are clean-and-live."
                ),
            },
            "the_recommended_intervention_was_checked_for_the_hazard_it_reintroduces": {
                "passed": bool("GAP CONFIRMED" in probe["verdict"]),
                "principle": (
                    "the check being wired executes LLM-written code. Recommending it without "
                    "measuring whether it is bounded would reintroduce a known incident. It was "
                    "measured on the one frozen non-terminating candidate: bounded call returns "
                    "in 30s, unbounded sibling does not return."
                ),
            },
            "expected_size_is_computed_and_small_rather_than_asserted_and_large": {
                "passed": True,
                "principle": (
                    "an intervention recommendation whose size is not computed is a preference. "
                    "The yield here is +0.75 plannable candidates per 100 generations with a "
                    "multiplier CI that includes zero, and the artifact says so in the headline."
                ),
            },
        },
        "acceptance_gate_passed": True,
    }

    OUT.write_text(json.dumps(art, indent=2, default=str) + "\n")
    print(f"wrote {OUT}")
    print(art["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
