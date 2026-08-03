#!/usr/bin/env python3
"""Build the outer-loop artifact for the induce-prompt audit."""

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A = json.loads((HERE / "out" / "analysis.json").read_text())
E = json.loads((HERE / "out" / "engine_census.json").read_text())
S = json.loads((HERE / "out" / "token_split.json").read_text())
ROWS = json.loads((HERE / "out" / "rows.json").read_text())

# Reproducibility checksum over the per-game prompt shas: this run's entire output is a
# function of the rendered prompts, so hashing them IS hashing the result.
h = hashlib.sha256()
for r in sorted(ROWS, key=lambda x: x["game"]):
    h.update(r["game"].encode())
    for k in ("live_all", "live_k8", "window_all", "anatomy_shown"):
        h.update((r.get("sha", {}).get(k) or "").encode())
CHECKSUM = h.hexdigest()

dur = round(sum(float(r.get("elapsed_s") or 0) for r in ROWS), 1)


def sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, cwd=REPO).stdout.strip()
    except Exception:
        return ""


art = {
    "experiment": "outer_loop_arc_induce_prompt_audit_20260802",
    "title": (
        "Audit of the induce PROMPT: is the model starved, misdirected, or asked the impossible?"
    ),
    "run_date": "2026-08-02",
    "milestone": "outer-loop",
    "inference_substrate": "aggregation_from_upstream_artifacts",
    "inference_substrate_note": (
        "NO LLM WAS RUN. The only compute is (a) stepping the OFFLINE arcade to rebuild each "
        "game's induction window and (b) tokenizing the resulting strings with "
        "`llama_cpp.Llama(model_path=..., vocab_only=True)` -- vocabulary only, no weights, no "
        "GPU, no server, no generation. CUDA_VISIBLE_DEVICES is emptied in every worker before "
        "any carnot import. `aggregation_from_upstream_artifacts` is chosen over "
        "`offline_arcade_live_agent_runtime_self_discovery_no_llm` because no agent policy runs "
        "here either: the arcade is stepped only to reconstruct evidence, and every reported "
        "number is a deterministic property of a rendered STRING."
    ),
    "solve_provenance": "development_proxy",
    "solve_provenance_note": (
        "No level was solved, attempted or claimed. The windows come from "
        "`build_progress_window`, i.e. the offline development twin. Declared because the run "
        "touches ARC solve machinery, not because a solve occurred."
    ),
    "random_seed": 20260802,
    "random_seed_note": (
        "Recorded for form. NOTHING HERE IS STOCHASTIC: prompt rendering is deterministic given "
        "the window, and `build_progress_window` is itself deterministic -- demonstrated, not "
        "asserted, by the 19/19 sha reproduction gate below against a run frozen on a different "
        "day. Re-running reproduces byte-identical prompts."
    ),
    "reproducibility_checksum": CHECKSUM,
    "duration_s": dur,
    "duration_s_note": (
        "Sum of per-game worker wall clock across 25 killable subprocesses (6-way parallel), "
        "dominated by tr87's 337s offline solve. Short because no model weights are loaded."
    ),
    "not_submitted": "no scored or online ARC game was played; submission is operator-only",
    "no_shipped_default_was_changed": (
        "This run DIAGNOSES the prompt. It changes nothing. No env flag was flipped, no prompt "
        "string was edited, and `results/arc_e3` was read and never written (verified with "
        "`git status results/arc_e3`, clean)."
    ),
    "clustering": (
        "GAME. Every quantity is one measurement per game over 24 games. There are no "
        "within-game replicates to mis-pool, because a game's prompt is a single deterministic "
        "string."
    ),
    "min_reachable_p": {
        "value": None,
        "why_none": (
            "STATED BEFORE RESULTS AND DELIBERATELY NULL. This is a CENSUS, not a hypothesis "
            "test: each per-game outcome is a deterministic function of source code plus a "
            "deterministic window, so there is no sampling distribution and a p-value would be "
            "decorative. For the record, had the two roster-wide binaries (24/24 prompts demand "
            "a full grid; 24/24 carry the no-reasoning directive) been treated as a sign test "
            "at game clustering, the smallest reachable two-sided p would be 2^-23 = 1.2e-7 -- "
            "but reporting that would dress a code-path census as evidence, so it is not "
            "claimed."
        ),
    },
    "missing_is_never_zero": (
        "dc22 produced no induction window (`build_progress_window` returned None) and is "
        "recorded as a COVERAGE GAP, excluded from every quantile. It is never counted as a 0. "
        "24 of 25 public games returned."
    ),
    "fidelity_gate": {
        "what": (
            "The prompts audited here must be the prompts the model was really sent. This "
            "harness re-renders them, so it must prove its renderer agrees with a run frozen on "
            "a different day."
        ),
        "method": (
            "Rebuild each game's window, split it with `wmte._split_prefix_heldout` exactly as "
            "`arc_goal_predicate_anatomy_20260801` did, render `induce_prompt` on the shown "
            "half with the object-perception flag unset, and compare sha256 against that run's "
            "recorded `prompt_sha256` for its `off` arm."
        ),
        "frozen_games": 19,
        "reproduced": 19,
        "mismatched": 0,
        "verdict": "PASS -- byte-identical on all 19 games the frozen run recorded",
    },
    "held_out_purity": {
        "applicable": False,
        "why": (
            "NO ENGINE IS SCORED HERE, so no split protects any number and none is claimed to. "
            "Every reported quantity is a property of a prompt STRING: its token count, which "
            "transitions it renders, what its instructions demand. `_split_prefix_heldout` "
            "appears exactly once, to reproduce the frozen run's sha, and never to score. The "
            "primary risk named in the brief -- a scoring transition visible during induction "
            "-- has no surface in this run."
        ),
    },
    "scope_and_branch": {
        "measured_on": "the 25-game PUBLIC roster, offline arcade, development-proxy windows",
        "window_distribution_caveat": (
            "LOAD-BEARING. `build_progress_window` returns the last k actions of a BANKED "
            "WINNING ROUTE cut at the L0->L1 boundary. The LIVE agent inducts on whatever its "
            "stall-triggered exploration buffer holds, which is a strictly weaker and different "
            "sample. Two findings below are distribution-dependent and are flagged as such: the "
            "zero-no-op result, and the per-game action counts. The structural findings (what "
            "the prompt ASKS, what the directive FORBIDS, the goal prompt's emptiness) are "
            "properties of the prompt template and hold on any distribution."
        ),
        "hidden_game_branch": (
            "`HIDDEN_STATE_GAME_IDS` is a hardcoded 11-game PUBLIC tuple, so a hidden Kaggle "
            "game always takes the PLAIN branch. Nothing in this audit is branch-specific: "
            "`induce_prompt` and `_L2_CODEONLY_DIRECTIVE` are shared by both branches, so the "
            "template findings carry to a hidden game. The action-coverage and sparsity "
            "distributions do NOT carry, and are not claimed to."
        ),
        "live_faithfulness": (
            "The audited prompt is rendered with the level-up rows REMOVED. "
            "`_active_transitions()` returns `transitions[_episode_transition_start:]` and that "
            "index is set one past the level-up, so a live induce window cannot contain one. "
            "Rendering the raw offline window instead would fire the WIN TRANSITION block, "
            "which the live path provably never fires (0 of 128 live induce calls, "
            "results/outer_loop_arc_win_transition_exposure_20260802.json). Both shapes are "
            "recorded; the `live_*` fields are the live-faithful ones."
        ),
    },
    "prompt_budget": {
        "tokens": A["prompt_budget_tokens"],
        "derivation": A["budget_derivation"],
        "tokenizer": "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf, vocab_only (the LIVE generator)",
        "tokenizer_rule": (
            "Tokenized through the .gguf path per the CLAUDE.md GGUF tokenizer rule. "
            "AutoTokenizer on a GGUF repo id was NOT used and would have failed -- those repos "
            "ship no HF tokenizer files."
        ),
    },
    "Q2_does_it_truncate": {
        "answer": "NO. Not today, and not close.",
        "as_sent_tokens": A["tokens"]["as_sent_all"],
        "max_pct_of_budget": A["budget_headroom"]["max_pct_of_budget"],
        "n_games_over_budget": A["budget_headroom"]["n_games_over_budget"],
        "n_games_char_budget_bound": A["truncation"]["n_games_char_budget_bound"],
        "reading": (
            "The complete payload the model receives -- code-only directive + induce prompt + "
            "the ```python primer -- has a median of "
            f"{A['tokens']['as_sent_all']['median']:.0f} tokens and a roster maximum of "
            f"{A['budget_headroom']['max_as_sent_tokens']}, against a 16,384-token per-slot "
            "budget. That worst case is 54.1% of budget. The 40,000-char transition budget "
            "never bound on any game. The model is NOT starved by truncation, and the "
            "'context truncation' branch of the three-way diagnosis is REFUTED for the current "
            "default."
        ),
        "which_transitions_would_be_dropped": (
            "Under the PRE-2026-08-01 default (k=8) the picture is different and worth "
            "recording, because that is the configuration most banked results were produced "
            "under. It changes the prompt on 16 of 24 games, dropping a median of 5 "
            "grid-CHANGING transitions -- and the informative ones are exactly what it drops, "
            "since these windows contain ZERO no-ops so `changed[:k-2] + noop[:2]` degenerates "
            "to `changed[:6]` and the two reserved no-op slots are simply wasted."
        ),
        "CORRECTION_to_a_documented_claim": {
            "the_claim": (
                "`_induce_transitions_k`'s docstring states: 'Action coverage was checked and "
                "was NOT being lost -- changed[:6] already covers every distinct action on all "
                "six games, so the stronger story (whole actions were hidden from the model) is "
                "false and is not being told.'"
            ),
            "measured_here": (
                f"FALSE ON THE FULL ROSTER. k=8 loses at least one entire action on "
                f"{A['truncation']['action_coverage_lost_by_k8']} of 24 games. The docstring's "
                "check was run on six games and its conclusion was stated for all. The stronger "
                "story it declined to tell was true on 10 games."
            ),
            "direction": (
                "This STRENGTHENS the 2026-08-01 change to k=all, which shipped on a token "
                "argument. It does not retroactively make that change measured for EFFECT -- "
                "engine accuracy under k=8 vs k=all is still unmeasured, and is not measured "
                "here."
            ),
        },
    },
    "Q3_what_does_it_ask_for": {
        "answer": "A FULL GRID, on every game, with no delta option offered.",
        "engine_returns_full_grid": A["asks"]["engine_returns_full_grid"],
        "evidence_is_delta_encoded": A["asks"]["evidence_is_delta_encoded"],
        "mentions_delta_output_format": A["asks"]["mentions_delta_output_format"],
        "the_asymmetry": (
            "24 of 24 prompts spend ~790 header tokens teaching TWO run-length codecs so the "
            "model can READ the evidence as deltas, then contract the OUTPUT as "
            "'Return the predicted next grid (same shape)'. 0 of 24 offer any way to answer in "
            "the delta language they just taught. The model reads sparse edits and must reply "
            "with a total grid transform."
        ),
        "why_this_favours_identity": (
            "A full-grid answer is graded on all 4096 cells. Measured on the shown transitions, "
            "the median transition changes 1.04% of cells "
            f"(min {A['change_sparsity']['changed_cell_fraction_median']['min']}, max "
            f"{A['change_sparsity']['changed_cell_fraction_median']['max']}), so the identity "
            "function is already cell-wise correct on a median 98.96% of cells (roster range "
            "92.8%-99.9%). Under any cell-wise reading of 'predict the next grid', writing "
            "`return grid` scores ~99%. It is not a lazy answer to that question; it is close "
            "to the best answer available to a model that cannot see the rule."
        ),
        "the_gate_disagrees": (
            "The trust gate scores full-grid EXACT match, where identity scores 0. So the "
            "prompt's phrasing and the gate's metric reward opposite behaviours, and nothing in "
            "the prompt tells the model which one it is being graded on."
        ),
    },
    "Q4_is_identity_invited": {
        "answer": (
            "YES -- by four independent routes, and forbidden by none. This is the strongest "
            "finding in the audit."
        ),
        "route_1_the_directive_forbids_the_task": {
            "codeonly_directive_present": A["identity_surface"]["codeonly_directive_present"],
            "codeonly_forbids_grid_analysis": A["identity_surface"][
                "codeonly_forbids_grid_analysis"
            ],
            "no_think_prefix": A["identity_surface"]["no_think_prefix"],
            "verbatim": (
                "'/no_think' then 'Do NOT analyze the grids. Do NOT describe or reason about "
                "the win state. Do NOT write step-by-step analysis, explanation, or commentary "
                "-- not even as comments.' and '4. Induce SIMPLE, GENERAL rules and write the "
                "requested function(s) directly. Skip all reasoning.'"
            ),
            "reading": (
                "This is prepended to 24 of 24 payloads and is the FIRST thing the model reads. "
                "The task it prefaces is: infer a transition rule from grid deltas. The "
                "directive forbids analysing the grids and instructs the model to skip all "
                "reasoning. A model that complies cannot induce dynamics -- it can only pattern-"
                "match a plausible-looking function, and `return grid` is the most plausible "
                "function for a grid task with no rule in evidence."
            ),
            "it_was_shipped_for_a_real_reason": (
                "NOT a blunder. It exists because Qwen3.5-9B burned its whole budget on "
                "win-state chain-of-thought and emitted 0 code (605s rambling / 450s "
                "truncated, vs 195 tokens / 15.6s with the directive). It is default-on by a "
                "2026-06-25 operator directive. But the generator was swapped to gemma-4-31B on "
                "2026-07-28 and the directive was never re-justified against the new model. "
                "`CARNOT_ARC_CODEONLY_INDUCE=0` already exists as the off switch, and "
                "`arc_actions_to_progress` already defines a `codeonly:0 + /think` arm -- so "
                "this is testable today without writing new code."
            ),
        },
        "route_2_the_evidence_makes_identity_nearly_right": {
            "identity_cellwise_accuracy_median": A["change_sparsity"][
                "identity_cellwise_accuracy_median"
            ],
            "reading": (
                "See Q3. Identity is cell-wise correct on a median 98.96% of cells. Nothing in "
                "the prompt tells the model that the 1% it is getting wrong is the entire point."
            ),
        },
        "route_3_the_action_space_is_mostly_unobserved": {
            "n_observed_actions": A["action_space"]["n_observed_actions"],
            "coverage_fraction": A["action_space"]["coverage_fraction"],
            "n_games_single_action_only": A["action_space"]["n_games_single_action_only"],
            "games_single_action_only": A["action_space"]["games_single_action_only"],
            "n_games_observing_all_7": A["action_space"]["n_games_observing_all_7"],
            "reading": (
                "The prompt DECLARES the action space -- 'Actions are integers 1-7' -- and asks "
                "for `engine(grid, action, data)`, a TOTAL function over it. The evidence covers "
                "a median of 2 of those 7 actions (min 1, max 5); 11 of 24 games show the model "
                "exactly ONE action, and 0 of 24 show all seven. For the median 5 unobserved "
                "actions there is no evidence whatsoever, and `return grid` is the only branch "
                "body the prompt does not contradict. An engine that is identity on 5 of 7 "
                "branches is the CORRECT response to this prompt. Identity on all 7 is one "
                "small generalisation further."
            ),
        },
        "route_4_nothing_forbids_it": {
            "forbids_identity": A["identity_surface"]["forbids_identity"],
            "mentions_word_identity": A["identity_surface"]["mentions_word_identity"],
            "reading": (
                "0 of 24 prompts contain the word 'identity', any instruction that engine() must "
                "change something, or any statement that a do-nothing engine is unacceptable. "
                "The prompt does say 'Prefer SIMPLE GENERAL rules over per-frame special cases' "
                "(24/24) -- and the simplest, most general rule consistent with sparse evidence "
                "over an unobserved action space is that nothing happens."
            ),
        },
        "the_no_op_mechanism_is_INERT_on_this_distribution": {
            "n_noop": A["transitions"]["n_noop"],
            "noop_rendered_as_no_change": A["identity_surface"]["noop_rendered_as_no_change"],
            "n_no_change_examples": A["n_no_change_examples"],
            "first_rendered_transition_is_noop": A["first_rendered_transition_is_noop"],
            "reading": (
                "The brief asked whether `_transitions_block`'s deliberate 'keep 2 no-ops' "
                "helps or teaches the wrong lesson. ON THIS DISTRIBUTION IT DOES NEITHER: these "
                "windows contain ZERO no-ops on all 24 games, the literal '(no change)' never "
                "appears in any prompt, and no prompt opens on a no-op. The question is "
                "UNANSWERED here rather than answered negatively, and answering it needs "
                "exploration-buffer transitions -- see the window caveat. The one thing it does "
                "do on this distribution is waste 2 of the 8 slots under k=8, which is how "
                "`k=8` came to mean `changed[:6]`."
            ),
        },
    },
    "Q5_goal_only_prompt": {
        "answer": "It is not starved. It is empty.",
        "shipped_tokens": A["tokens"]["goal_shipped"],
        "induce_tokens_for_comparison": A["tokens"]["live_all"],
        "shipped_is_evidence_free": A["goal_prompt"]["shipped_is_evidence_free"],
        "shipped_carries_transitions": A["goal_prompt"]["shipped_carries_transitions"],
        "shipped_carries_any_grid": A["goal_prompt"]["shipped_carries_any_grid"],
        "receives_win_transition": A["goal_prompt"]["receives_win_transition"],
        "reading": (
            "The shipped `_goal_only_prompt` is 96 tokens on 23 of 24 games and 97 on the "
            "other -- it is essentially CONSTANT, because it contains nothing about the game "
            "except its name. 0 of 24 carry a grid; 0 of 24 carry a transition. The induce "
            "prompt beside it has a median of 3,717 tokens. The anatomy pass called this "
            "prompt evidence-free; that is confirmed, and the ratio is roughly 39:1."
        ),
        "differences_that_matter": [
            "It never receives `win_transition` (0/24) -- and on the live path neither does the "
            "induce prompt, so no prompt in the system has ever shown the model a self-observed "
            "win event.",
            "On the live path it also never receives `previous_level_complete_grid`: "
            "`arc_competition_agent.py:6497` passes only (game, active_transitions, cell). So "
            "even the one grid this prompt is designed around is absent live, leaving 96 tokens "
            "of pure instruction.",
            "It asks for a win condition while carrying no example of winning and no example of "
            "not-winning. That is not a hard question; it is an underdetermined one, and the "
            "constant predicates the 2026-08-01 taxonomy found (12 of 13 whole-board 'every "
            "cell is one colour') are the expected output of it.",
            "`CARNOT_ARC_GOAL_PROMPT_TRANSITIONS=1` raises it to a median 3,187 tokens and "
            "carries transitions on 24/24. It remains DEFAULT OFF and this run does not flip "
            "it.",
        ],
    },
    "Q1_where_the_tokens_go": {
        "note": (
            "SALIENCE, NOT CAPACITY. Since nothing truncates (Q2), a large layout share is not "
            "costing the model evidence it could otherwise have had. This is an observation "
            "about what dominates the prompt, and it is deliberately not dressed up as a "
            "bottleneck."
        ),
        "pct_static_layout_grid": S["pct_of_prompt_that_is_the_static_layout_grid"],
        "pct_transition_evidence": S["pct_of_prompt_that_is_transition_evidence"],
        "n_games_layout_exceeds_evidence": S["n_games_layout_exceeds_evidence"],
        "header_tokens": S["tok_header"],
        "reading": (
            "A median 40.7% of the payload is ONE static 64x64 grid rendered as 64 run-length "
            "rows, against 35.7% for all the transition deltas combined; on 13 of 24 games the "
            "single layout grid outweighs the entire dynamics evidence. A further ~790 tokens "
            "(near-constant) is header, most of it teaching the two codecs. The part of the "
            "prompt that can say what an ACTION DOES is the minority of it."
        ),
    },
    "engine_census": {
        "what": (
            "Static AST census of the 27 `world_model.py` files currently in results/arc_e3. "
            "READ ONLY -- `git status results/arc_e3` clean after the run. Static rather than "
            "executed because executing these is what wedged a prior session for 13 minutes."
        ),
        "verdict_counts": E["verdict_counts"],
        "n_behaviourally_identity": E["n_behaviourally_identity"],
        "behaviourally_identity_games": E["behaviourally_identity_games"],
        "empty_branch_rate": {k: v for k, v in E["empty_branch_rate"].items() if k != "per_game"},
        "CORRECTION_to_the_briefs_premise": {
            "the_premise": (
                "'surviving on-disk engines are `return grid` on every branch -- IDENTITY. "
                "ft09's is exactly that shape.'"
            ),
            "measured": (
                "TRUE OF ft09, OVERSTATED FOR THE POPULATION. Only 3 of 27 engines are "
                "behaviourally identity (ar25, ft09, g). 12 of 27 return the parameter on every "
                "path but WRITE THROUGH IT in place -- su15 does `grid[py,px] = 15` then "
                "`return grid`, which is a genuine state change, and a census that counted "
                "param-like returns alone would have called it identity. 12 of 27 have a "
                "non-param return. The first version of this census made exactly that mistake "
                "and is recorded here rather than quietly fixed."
            ),
            "what_survives_and_is_stronger": (
                "The EMPTY-BRANCH RATE. 15 of 27 engines return the input on 100% of their "
                "explicit `return` statements (median across engines: 1.0; 9 of 27 sit at "
                "0.0, so the distribution is bimodal, not middling)."
            ),
        },
        "the_shape_that_actually_explains_it": (
            "ft09's engine is not a stub and not a refusal. It dispatches on `action == 6`, "
            "handles `data is None`, extracts px/py, bounds-checks against grid.shape, and then "
            "enumerates EIGHT distinct colours at the clicked cell -- 15, 7, 13, 8, 10, 5, 0, 1 "
            "-- writing `return grid` as the body of every single one, plus a fallthrough, plus "
            "a final `return grid` for all other actions. The model correctly identified the "
            "action, the input format, the coordinate convention, and the discriminating "
            "feature (the colour under the click). It built the entire case analysis and left "
            "every consequent empty. su15 is the same skeleton with exactly ONE consequent "
            "filled in."
        ),
        "diagnosis": (
            "This is NOT 'the model is declining to model'. It is a model that has done the "
            "perception and the dispatch and failed at precisely one step: naming the OUTCOME "
            "of each case. That is the step 'Do NOT analyze the grids / Skip all reasoning' "
            "forbids, and it is the step that a median of 2 observed actions and ~1% changed "
            "cells gives it almost no evidence for."
        ),
    },
    "HEADLINE": {
        "starved": (
            "NOT of context -- worst case 54.1% of budget, nothing truncates, the char budget "
            "never binds. STARVED OF ACTION COVERAGE: a median 2 of the 7 declared actions are "
            "ever observed, 11 of 24 games show exactly one, none shows all seven."
        ),
        "misdirected": (
            "YES, and this is the actionable half. The payload's first instruction "
            "(`/no_think`, 'Do NOT analyze the grids', 'Skip all reasoning') forbids the "
            "inference the task requires; it was justified against a generator retired on "
            "2026-07-28 and never re-justified. Separately, the output contract is full-grid "
            "while all evidence is delta-encoded, and a median 40.7% of tokens is one static "
            "layout grid."
        ),
        "asked_the_impossible": (
            "FOR THE GOAL PROMPT, YES: 96 tokens, zero grids, zero transitions, asked to state "
            "a win condition. FOR THE ENGINE, PARTIALLY: a total function over 7 actions from a "
            "median of 2 observed, where identity is already ~99% cell-wise correct and nothing "
            "forbids it."
        ),
        "single_sentence": (
            "The model is not refusing to model -- ft09's engine proves it builds the correct "
            "click-dispatch skeleton with eight colour cases and leaves every body empty -- so "
            "the binding defect is that the prompt forbids the reasoning needed to fill those "
            "bodies, shows a median 2 of 7 actions, and grades nothing that would tell the "
            "model identity is wrong."
        ),
        "necessary_not_sufficient": (
            "EVERY finding here is about a PROMPT. None of it is a behavioural claim. No "
            "action, plan, level or engine accuracy was measured or moved. A better prompt is a "
            "hypothesis about the generation wall, not a dent in it."
        ),
    },
    "what_this_does_NOT_establish": [
        "That removing the code-only directive improves engine accuracy. UNMEASURED. It "
        "plausibly reintroduces the truncation failure it was shipped to fix, on a model that "
        "is not the one it was measured on.",
        "That k=all is better than k=8 for accuracy. This run shows k=8 dropped whole actions "
        "on 10 of 24 games, which is a mechanism, not an outcome.",
        "That the goal prompt's emptiness CAUSES the constant predicates. The 2026-08-01 "
        "taxonomy's association is consistent with it; this run only confirms the prompt is "
        "empty.",
        "Anything about a HIDDEN game's action-coverage or sparsity distribution. Only the "
        "template findings carry across the branch.",
        "That the on-disk engines were produced by today's prompt. They are an artifact "
        "population of unknown provenance and mixed vintage; the census reads their SHAPE, and "
        "no shape is attributed to a specific prompt version.",
    ],
    "cheapest_next_tests": [
        "ALREADY BUILT, NO NEW CODE: `arc_actions_to_progress` defines a `codeonly:0 + /think, "
        "8192 n_predict` arm beside the frozen live default. Run it against held-out "
        "change_accuracy, clustered at GAME. The directive is the one prompt element with a "
        "shipped off-switch, a stated rationale, and a retired justification.",
        "Ask for a DELTA instead of a full grid, or state the grading metric in the prompt. "
        "Both are one-line prompt edits behind a new default-OFF flag with a both-directions "
        "test, per the standing discipline.",
        "Measure whether the no-op mechanism helps on EXPLORATION-BUFFER transitions, where "
        "no-ops actually exist. This audit could not test it: the dev windows have none.",
    ],
    "preconditions_checked": [
        {"resource": "gemma-4-31B-it-qat GGUF cached", "available": True},
        {"resource": "llama_cpp importable (vocab_only tokenize)", "available": True},
        {"resource": "offline arcade environment_files", "available": True},
        {"resource": "frozen anatomy prompt_witness.json for the sha gate", "available": True},
        {
            "resource": "GPU",
            "available": False,
            "note": "deliberately unavailable: CUDA_VISIBLE_DEVICES emptied in every worker",
        },
    ],
    "artifacts": {
        "rows": "results/arc_induce_prompt_audit_20260802/out/rows.json",
        "analysis": "results/arc_induce_prompt_audit_20260802/out/analysis.json",
        "engine_census": "results/arc_induce_prompt_audit_20260802/out/engine_census.json",
        "token_split": "results/arc_induce_prompt_audit_20260802/out/token_split.json",
        "rendered_prompts": "results/arc_induce_prompt_audit_20260802/out/prompts/<game>/",
    },
    "code_read": [
        "python/carnot/agentic/arc_executable_world_model.py: induce_prompt, "
        "_transitions_block, _select_transitions_for_prompt, _induce_transitions_k, "
        "_L2_CODEONLY_DIRECTIVE, LocalGGUFProposer.induce, _goal_only_prompt",
        "python/carnot/agentic/arc_competition_agent.py:6497 (the live induce call site)",
        "python/carnot/agentic/arc_engine_static_validation.py:engine_changes_anything",
    ],
    "git_head": sh("git", "rev-parse", "HEAD"),
    "honest_verdict": (
        "complete_induce_prompt_audit_not_truncated_54pct_of_budget_but_MISDIRECTED_"
        "codeonly_directive_forbids_the_required_reasoning_on_24of24_and_median_2of7_actions_"
        "observed_and_goal_prompt_is_96_tokens_evidence_free_on_24of24_NO_behavioural_claim"
    ),
}

p = REPO / "results" / "outer_loop_arc_induce_prompt_audit_20260802.json"
p.write_text(json.dumps(art, indent=1))
print(f"wrote {p}")
print(f"duration_s={dur} checksum={CHECKSUM[:16]}")
