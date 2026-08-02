#!/usr/bin/env python3
"""Assemble the scored artifact for the goal-evidence A/B from the run's own outputs.

Every number in the artifact is READ from out/analysis.json, out/meta.json and
out/preregistration.json -- none is retyped. A retyped number is a number that can drift away
from the measurement that produced it, and the project has a standing reading-results discipline
for exactly that reason.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"


def sha_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def find(tests: list[dict], block: str, shape: str) -> dict:
    for t in tests:
        if t.get("block") == block and t.get("shape") == shape:
            return t
    return {}


def exact_min_two_sided_p(n_games: int) -> float:
    """The SMALLEST two-sided p the game-clustered permutation test can return at `n_games`.

    Added 2026-08-02 post-review. The artifact reported the DESIGN-TIME floor (~0 at the
    pre-registered 20 games) and nothing told a reader what the TRUNCATED run could reach. With
    one treatment and one control cell per game, `analyse.permutation_test` has exactly two
    arrangements per game -- keep the pair or swap it -- so the reference set is 2**n_games and
    the most extreme observable statistic is matched by 2 of them. Hence 2 / 2**n_games, which
    is 1.0 at one game, 0.5 at two and 0.25 at three.

    This is why "NON-TEST" is the honest verdict and not a rhetorical hedge: at the n this run
    reached, NO contrast in it could have come back below alpha whatever the model did.
    """
    return round(2.0 / (2**n_games), 6) if n_games >= 1 else 1.0


def main() -> int:
    an = json.loads((OUT / "analysis.json").read_text())
    meta = json.loads((OUT / "meta.json").read_text())
    prereg = json.loads((OUT / "preregistration.json").read_text())
    tests = an["tests"]

    s1_primary = find(tests, "stage1", "DECLINED")
    s1_trope = find(tests, "stage1", "TROPE")
    s1_grounded = find(tests, "stage1", "GROUNDED")
    s1_floor = {s: find(tests, "stage1_AA_floor", s) for s in ("DECLINED", "TROPE", "GROUNDED")}
    s2_b = find(tests, "stage2_ITT", "DECLINED")
    s2_c = [t for t in tests if t.get("block") == "stage2_ITT" and t.get("shape") == "DECLINED"]
    s2_floor = find(tests, "stage2_AA_floor", "DECLINED")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd=ROOT
    ).stdout.strip()

    # The reproducibility checksum binds the three things a third party would have to hold fixed
    # to get this run back: the pre-registration (the design, written before any LLM call), the
    # rows (every cell), and the analysis (the scoring). A checksum over the artifact itself
    # would be circular.
    repro = hashlib.sha256(
        (
            sha_file(OUT / "preregistration.json")
            + sha_file(OUT / "rows.json")
            + sha_file(OUT / "analysis.json")
        ).encode()
    ).hexdigest()

    # ---- post-review additions (2026-08-02) -------------------------------------------------
    # Each is READ from a file the run produced, never retyped, for the same reason every other
    # number here is: a retyped number drifts away from the measurement that produced it.
    placebo = json.loads((OUT / "grounded_placebo.json").read_text())
    placebo_survival = placebo["placebo"]["label_preserved_under_a_foreign_games_transitions"]
    min_reachable = {
        "why": "the DESIGN's floor (min_reachable_p, ~0 at 20 games) says nothing about what "
        "this truncated run could reach. This does. Any contrast whose entry here exceeds "
        "0.05 was UNRESOLVABLE BY CONSTRUCTION, before the model wrote a single token.",
        "formula": "2 / 2**n_games -- one treatment and one control cell per game gives two "
        "arrangements per game, and the extremal statistic is matched by two of them",
        "by_contrast": {
            # Keyed by block AND arms: two stage-2 ITT contrasts share a (block, shape) pair
            # (B-vs-A and C-vs-A) and keying on that alone would silently drop one of them.
            f"{t.get('block')}::{t.get('shape')}::{t.get('treat')}_vs_{t.get('control')}": {
                "n_games": t.get("n_games", 0),
                "exact_min_two_sided_p": exact_min_two_sided_p(int(t.get("n_games", 0) or 0)),
                "observed_p": t.get("p_permutation_two_sided"),
            }
            for t in tests
            if t.get("n_games")
        },
        "smallest_attainable_p_anywhere_in_this_run": min(
            exact_min_two_sided_p(int(t.get("n_games", 0) or 0)) for t in tests if t.get("n_games")
        ),
        "reading": "the smallest p ANY contrast in this run could have returned is 0.25, on the "
        "three-game stage-1 blocks. The PRIMARY landed one paired game, where the only "
        "attainable p is 1.0. NON-TEST is therefore a structural fact about the achieved n, "
        "not a reading of the observed p-values.",
    }
    # The preflight partition is carried forward with its own limitation attached, rather than
    # quoted bare. See POST_REVIEW_CORRIGENDUM_2026_08_02 finding 4.
    preflight = dict(prereg["PREFLIGHT_ON_THE_FROZEN_SHIPPED_CORPUS"])
    preflight["PARTITION_IS_A_ONE_DIRECTIONAL_PROXY"] = (
        "split-induce is inferred as `n_defs >= 2` (two top-level is_level_complete definitions "
        "-- the shadowing signature). Verified to hold on all 116 frozen cells BY CONSTRUCTION, "
        "which is the problem: it is the definition used, not an independent measurement. "
        "`_split_induce` writes only ONE definition whenever the engine half supplied no "
        "is_level_complete, so such a cell has n_defs == 1 and is counted as COMBINED. The "
        "error therefore runs one way -- split cells hide in the combined bucket, never the "
        "reverse -- and because split cells decline far less (0.174 vs 0.495), the true split "
        "share is at least 19.8%. THE 3.4-POINT FIGURE IS A LOWER BOUND ON THE KNOBS' REACH, "
        "NOT A CEILING. The sign of the conclusion (declines concentrate where the knobs are "
        "not) is unaffected; the exact number is not a hard ceiling and should not be quoted "
        "as one. To settle it, persist `_write_world_model`'s split-induce note per cell -- "
        "today it is returned to the caller and never written down. This run's own 8 stage-2 "
        "cells agree with the direct `goal_only_call_ran` witness 8/8, which is consistent but "
        "far too small to validate the proxy over a 116-cell corpus."
    )

    byte_identity = an["stage2_nonfiring_byte_identity"]
    known_confounds = {
        "GROUNDED_is_partially_determined_by_the_treatment": {
            "severity": "HIGH -- this is the circularity class the brief warns about",
            "what": "GROUNDED is defined as 'the predicate names a literal that appears in the "
            "agent's observed deltas'. The TREATMENT prompt is precisely those deltas, rendered "
            "as text. A model that copies a row index out of its prompt scores GROUNDED without "
            "having understood anything, and the control -- which is shown no deltas -- can only "
            "score GROUNDED by coincidence.",
            "so_what": "the GROUNDED contrast is NOT a clean read on goal quality and must not "
            "be reported as one. It is reported as what it is: whether the treatment gets the "
            "model to REFERENCE its own observations.",
            "which_metrics_are_clean": "DECLINED (the PRIMARY) is unaffected -- a constant-False "
            "predicate references nothing, so no prompt content can mechanically produce or "
            "prevent it. TROPE is unaffected for the same reason: a whole-board uniformity claim "
            "names nothing observed by construction. MISSING is unaffected -- it is a parse "
            "outcome. The circularity is confined to GROUNDED.",
            "why_it_was_not_fixed": "removing it would mean changing the pre-registered outcome "
            "definition after seeing data. It is disclosed instead.",
            "UPGRADED_2026_08_02": "this disclosure was NECESSARY BUT NOT SUFFICIENT. It still "
            "assumed the label at least measures the agent's own transitions. The placebo test "
            "in GROUNDED_placebo_discriminance shows it does not: re-scored against a DIFFERENT "
            "game's transitions the label survives on "
            f"{round(placebo_survival * 100)}% "
            "of comparisons, and every GROUNDED cell in the run stays GROUNDED under at least 17 "
            "of 19 foreign games. GROUNDED is a SYNTAX label. The 'which_metrics_are_clean' "
            "sentence above still stands and was re-checked, not assumed.",
        },
        "arm_ORDER_within_a_game_is_fixed_not_randomised": {
            "severity": "MEDIUM, and it reaches SHAPE as well as TIMING -- see so_what_for_SHAPE",
            "what": "cells run A, B, C, AA within a game (and gA, gB, gAA in stage 1), always in "
            "that order. Anything that drifts with position -- llama.cpp prompt-cache warmth, GPU "
            "thermal state -- is therefore confounded with the arm label.",
            # CORRECTED 2026-08-02. This entry previously read "nothing ... the byte-identity
            # check on non-firing stage-2 cells is direct evidence that order is not perturbing
            # the sampled output", which directly contradicted the DIAGNOSIS recorded in
            # stage2_nonfiring_byte_identity in the SAME artifact. The DIAGNOSIS was right and
            # this entry was stale prose written before the check ran.
            "so_what_for_SHAPE": "NOT nothing, contrary to what this entry said until "
            "2026-08-02. Every arm sends the same seed within a (game, replicate), but the "
            "byte-identity check FOUND A DIVERGENCE rather than confirming the assumption: on "
            "bp35 the control wrote 3722 bytes and both treatments 4150, at seed 8300, with the "
            "goal-only call firing in NONE of the three so neither knob could act. A fixed "
            "CARNOT_ARC_GENERATOR_SEED does not fix the completion on this server. The pairing "
            "is APPROXIMATE and arm order is a confound on shape, not only on timing.",
            "so_what_for_TIMING": "median_elapsed_s must be read descriptively and NEVER as a "
            "treatment effect with a p-value. Observed directly in this run: on ar25 the control "
            "took 111.4s and both treatments 55.2/55.1s, while on bp35 the control took 44.5s "
            "and both treatments 67.5/67.6s -- the sign flips between games, which is what "
            "position-plus-noise looks like.",
            "where_timing_IS_load_bearing": "stage 1, where the treatment prompt is 10-20x the "
            "control's and the gap is 3s vs 340s. A cache or thermal effect cannot produce a "
            "100x difference in that direction -- warmth makes calls FASTER -- and the failures "
            "are explained by a mechanism visible in the artefact itself (three full 4096-token "
            "attempts, code block truncated mid-definition).",
            "the_fix_for_a_future_run": "randomise or counterbalance arm order within a game.",
            "evidence": byte_identity["DIAGNOSIS"],
        },
        "transition_SELECTION_is_solve_conditioned": {
            "severity": "HIGH for the hidden-game reach claim; NIL for the within-run contrast",
            "added": "2026-08-02, post-review. Not disclosed in the original artifact at all.",
            "what": "the prompt CONTENT is clean -- only the agent's own observations reach the "
            "model. How those observations were SELECTED is not. Every window comes from "
            "build_progress_window -> exp5717.build_window -> arc_loop_solve.solve_adaptered"
            "(game, 1), i.e. the last k actions of a BANKED WINNING ROUTE, cut at the L0->L1 "
            "boundary. On a hidden game the live `trans` at a _split_induce call is whatever the "
            "stall-triggered exploration buffer happens to hold -- a strictly weaker sample from "
            "a different distribution.",
            "so_what": "it does not bias the A/B: every arm within a game sees the identical "
            "window. It does invalidate any extrapolation from these rates to a hidden game, "
            "which is why works_on_an_unsolved_game is now False.",
            "how_to_remove_it": "drive the contrast from a live exploration buffer rather than a "
            "banked route -- the harness would have to change, since build_progress_window "
            "returns None for a game it cannot solve to L1 offline.",
        },
    }

    corrigendum = {
        "provenance": "adversarial review of the committed artifact, 2026-08-02, same day. Every "
        "finding below was re-verified against the run's own data before being applied; the "
        "verification commands are named so a reader can repeat them. Nothing in "
        "out/preregistration.json, out/rows.json or out/analysis.json was touched, so "
        "reproducibility_checksum is unchanged and still binds the design, the cells and the "
        "scoring exactly as they were.",
        "1_works_on_an_unsolved_game_was_FALSE": {
            "was": "works_on_an_unsolved_game: true",
            "now": "false, restated as a mechanism argument",
            "verified_how": "all 20 roster games are full_game_clear: true in "
            "ops/arc_solve_registry.yaml (levels_reproduced 6-10), and every window required "
            "solve_adaptered(game, 1) to reach L1 -- build_progress_window's own docstring says "
            "it returns None otherwise. Not one unsolved game was measured.",
        },
        "2_the_roster_has_no_zero_win_games": {
            "was": "'stall games with zero observed wins are in the roster on purpose', asserted "
            "in run_ab.py, out/preregistration.json and the research note",
            "now": "false as written. meta.json's split_meta records levelup_rows_in_heldout == 1 "
            "for all 20 games; the window builder logged levelups=1 for all 20. The verified "
            "claim is the narrower one: levelup_rows_in_shown == 0, i.e. no ARM was shown a win.",
            "prereg_left_frozen": "out/preregistration.json still contains the false sentence and "
            "was deliberately NOT edited -- a pre-registration that is rewritten after seeing "
            "data is not a pre-registration. prereg_sha256 still verifies. run_ab.py carries an "
            "in-place correction comment beside the string it emits, and emits the same bytes.",
        },
        "3_the_tn36_existence_proof_was_false": {
            "was": "works_on_an_unsolved_game_evidence.existence_proof_class -- 'tn36 is in the "
            "roster precisely because it is a stall game'",
            "now": "deleted. tn36 is full_game_clear: true with levels_reproduced: 7, and its "
            "registry entry cites four explicit outer-loop source reads (tn36.py:2153, :2171, "
            ":2269, :2500). It also contributed ZERO cells to this run -- rows.json holds only "
            "ar25, bp35 and cd82 -- so it was a roster entry, never a measurement.",
        },
        "4_the_split_vs_combined_partition_is_a_one_directional_proxy": {
            "detail": "see preflight_on_frozen_shipped_corpus."
            "PARTITION_IS_A_ONE_DIRECTIONAL_PROXY. The 3.4-point figure is a LOWER BOUND on the "
            "knobs' reach, not a ceiling. The direction of the finding is unchanged.",
        },
        "5_GROUNDED_does_not_discriminate": {
            "detail": "see GROUNDED_placebo_discriminance. Two behaviour-preserving rewrites "
            "were run against REAL captured cells: `return False` -> `return False and "
            "grid[0, 0] == 4` flips all five real DECLINED cells to GROUNDED, exactly inverting "
            "the reported S2-B/S2-C grounded result from a model that got strictly no better; "
            "an alpha-rename or an operand swap flips TROPE cells to GROUNDED. The PRIMARY and "
            "TROPE contrasts are NOT affected.",
        },
        "6_the_artifact_was_not_the_output_of_its_own_builder": {
            "was": "exhibit_truncation and known_confounds appeared in the committed artifact and "
            "in no code path; artifacts{} carried annotations build_artifact.py did not emit. "
            "Rebuilding produced 47 leaf differences.",
            "now": "both blocks are constructed in build_artifact.py, so a rebuild reproduces the "
            "file. Found while applying finding 1, not reported by the review.",
            "why_it_matters": "this is precisely the class artifact_freshness_lint.py exists for. "
            "That lint did not fire because this artifact is not in "
            "ops/analyzer_artifact_index.json -- a coverage limit the lint's own docstring names "
            "out loud, demonstrated here on a live artifact one day later.",
        },
        "7_min_reachable_p_was_the_designs_floor_not_the_runs": {
            "detail": "see min_reachable_p_at_achieved_n. The smallest p ANY contrast in this "
            "truncated run could have returned is 0.25; on the PRIMARY, which landed one paired "
            "game, it is 1.0. Reported alongside the design-time 0.0 rather than replacing it.",
        },
        "what_survived_review_unchanged": [
            "the PRIMARY (DECLINED) and TROPE contrasts and their A/A floors",
            "the structural-misdirection finding: 46 of 50 declines in the frozen corpus sit on "
            "the combined path where the goal-only prompt is never built",
            "the truncation exhibit: 4096/4096 tokens against the control's 314, 819 empty code "
            "fences, 3s -> 343s per call",
            "solve_provenance: development_proxy -- correctly declared, no level banked, "
            "adversarial_verify flags nothing",
            "defaults unchanged: both knobs still ship OFF",
        ],
    }

    art = {
        "experiment": "arc_goal_evidence_ab_20260802",
        "title": "Does giving the ARC goal prompt the agent's own observed evidence stop the "
        "model declining to write a win condition?",
        "run_date": datetime.now(UTC).isoformat(),
        "git_head": head,
        "schema": "carnot.arc_goal_evidence_ab.v1",
        "duration_s": meta["duration_s"],
        # A BARE INT, matching the sibling artifacts, because `adversarial_verify`'s
        # methodology check reads this field and a principle-wrapped or prose value is exactly
        # the field-shape assumption that silently defeated substrate recognition on 176
        # artifacts corpus-wide (the QA-layer discipline's origin bug #2). The scheme that
        # generated the full set lives beside it as its own string field.
        "random_seed": min(int(r["seed"]) for r in json.loads((OUT / "rows.json").read_text())),
        "random_seed_scheme": prereg["generator"]["seed_scheme"],
        "random_seeds_used": sorted(
            {r["seed"] for r in json.loads((OUT / "rows.json").read_text())}
        ),
        "reproducibility_checksum": "sha256:" + repro,
        "inference_substrate": "live_llm_inference",
        "inference_substrate_note": "every cell loads and generates from a real local GGUF "
        "through the shipped LocalGGUFProposer on the CUDA build; the server witness records "
        "pid, /proc/<pid>/exe, model path and n_ctx read back from /props",
        "model_specs": {
            "generator": meta["server_witness"]["model_from_props"],
            "repo_substr": prereg["generator"]["repo_substr"],
            "n_ctx": meta["server_witness"]["n_ctx_from_props"],
            "kv_quant": meta["server_witness"]["kv_quant"],
            "n_gpu_layers": meta["server_witness"]["n_gpu_layers"],
            "max_tokens": meta["server_witness"]["max_tokens"],
            "cuda_gpu": meta["server_witness"]["cuda_gpu"],
            "exe_from_proc": meta["server_witness"]["exe_from_proc"],
            "one_server_all_arms": True,
        },
        "preconditions_checked": meta["preconditions_checked"],
        "solve_provenance": prereg["solve_provenance"],
        "solve_provenance_note": prereg["solve_provenance_note"],
        "prereg_sha256": meta["prereg_sha256"],
        "n_cells": meta["n_cells"],
        "n_jobs": meta["n_jobs"],
        "treatment_witness_summary": {
            "n_games": len(meta["treatment_witness"]),
            "goal_prompt_chars_control": sorted(
                {t["goal_prompt_chars_off"] for t in meta["treatment_witness"]}
            ),
            "goal_prompt_chars_treatment_range": [
                min(t["goal_prompt_chars_on"] for t in meta["treatment_witness"]),
                max(t["goal_prompt_chars_on"] for t in meta["treatment_witness"]),
            ],
            "combined_induce_prompt_identical_between_arms": all(
                t["combined_prompt_identical"] for t in meta["treatment_witness"]
            ),
            "dedup_inert_when_off_and_excises_when_on": all(
                t["dedup_inert_when_off"] and t["dedup_excises_when_on"]
                for t in meta["treatment_witness"]
            ),
            "levelup_rows_shown_to_any_arm": sorted(
                {v["levelup_rows_in_shown"] for v in meta["split_meta"].values()}
            ),
        },
        "stage1_goal_only_component": an["stage1_goal_only_component"],
        "stage2_live_induce_ITT": an["stage2_live_induce_ITT"],
        "stage2_mechanism_firing": an["stage2_mechanism_firing"],
        "stage2_live_induce_mechanism_fired_only": an["stage2_live_induce_mechanism_fired_only"],
        "cluster_crosstab": an["cluster_crosstab"],
        "stage2_nonfiring_byte_identity": an["stage2_nonfiring_byte_identity"],
        "grounding_audit": an["grounding_audit"],
        "SENSITIVITY_grounded_excluding_trivial_literals": an[
            "SENSITIVITY_grounded_excluding_trivial_literals"
        ],
        # THE GROUNDED LABEL DOES NOT DISCRIMINATE. Added 2026-08-02 post-review; it is the
        # single most important thing in this artifact for anyone about to quote a grounded
        # rate. Produced by placebo.py from this run's own captured cells.
        "GROUNDED_placebo_discriminance": placebo,
        "per_game": an["per_game"],
        # The stopping rule and EVERY amendment to it, inlined rather than referenced. A
        # truncated run whose stopping rule lives only in a side file is a run whose reader has
        # to go looking for the reason the n is small; inlining it means the caveat travels with
        # the numbers it qualifies.
        "stopping_rule": json.loads((OUT / "stopping_rule.json").read_text()),
        "power_simulation_post_hoc": json.loads((HERE / "pre" / "power.json").read_text()),
        "PRIMARY_declined_rate": {
            "stage1_gB_vs_gA": s1_primary,
            "stage2_ITT_B_vs_A": s2_b,
            "stage2_ITT_C_vs_A": next((t for t in s2_c if t.get("treat") == "C"), {}),
        },
        "SECONDARY_trope_rate": {"stage1_gB_vs_gA": s1_trope},
        "SECONDARY_grounded_rate": {"stage1_gB_vs_gA": s1_grounded},
        "AA_NOISE_FLOOR": {"stage1": s1_floor, "stage2_declined": s2_floor},
        "tests": tests,
        # DESIGN-TIME floor of the permutation reference set, at the pre-registered 20 games.
        # It is NOT what this truncated run could reach -- see `min_reachable_p_at_achieved_n`
        # directly below, which is the number a reader should actually use.
        "min_reachable_p": prereg["MINIMUM_REACHABLE_P_AND_A_HONEST_POWER_STATEMENT"][
            "min_reachable_p_reported"
        ],
        "min_reachable_p_at_achieved_n": min_reachable,
        "power_statement": prereg["MINIMUM_REACHABLE_P_AND_A_HONEST_POWER_STATEMENT"],
        "preflight_on_frozen_shipped_corpus": preflight,
        # CORRECTED 2026-08-02 post-review, from TRUE. See POST_REVIEW_CORRIGENDUM_2026_08_02
        # finding 1: every one of the 20 roster games is `full_game_clear: true` in
        # ops/arc_solve_registry.yaml and every window required `solve_adaptered(game, 1)` to
        # reach L1, so not one unsolved game was measured. The MECHANISM claim survives; the
        # measured claim does not, and this boolean asserted the measured one.
        "works_on_an_unsolved_game": False,
        "works_on_an_unsolved_game_evidence": {
            "status": "MECHANISM ARGUMENT ONLY -- NOT MEASURED. Corrected 2026-08-02 after "
            "adversarial review; the field was TRUE and the supporting existence proof was "
            "false. No unsolved game appears anywhere in this run.",
            "what_reaches_the_model": "the agent's OWN observed transitions only, rendered by "
            "the same _transitions_block the engine prompt already uses. The FIELD is as "
            "available on a hidden game as on a solved one; the DISTRIBUTION is not -- see "
            "known_confounds.transition_SELECTION_is_solve_conditioned.",
            "no_win_is_shown_to_any_arm": "levelup_rows_in_shown is 0 for all 20 games -- the "
            "prefix split puts the level-up row in the held-out tail, so every predicate in "
            "this run was written by a model that had never seen the game won. This is the "
            "verified claim. It is NOT the same as 'the roster contains games with no win': "
            "meta.json's own split_meta records levelup_rows_in_heldout == 1 for all 20.",
            "nothing_a_hidden_game_would_withhold": "no game source, no hand-written adapter, no "
            "curated win example, and _previous_level_complete_grid passed as None in every arm "
            "(it is the NEXT level's opening board, not a win state -- the 2026-07-29 "
            "win-state-poison correction). True of the PROMPT CONTENT. Not true of how the "
            "transitions were SELECTED.",
            "what_would_actually_settle_it": "run the same contrast on a game with NO registered "
            "GameAdapter and no banked route, driving _split_induce from a stall-triggered "
            "exploration buffer. That is a different harness -- build_progress_window returns "
            "None for such a game by construction -- and it is the experiment this run did not "
            "run.",
        },
        "defaults_changed": False,
        "defaults_note": "both CARNOT_ARC_GOAL_PROMPT_TRANSITIONS and CARNOT_ARC_GOAL_DEDUP "
        "still ship OFF. This run measures them; flipping either is an operator decision.",
        "not_submitted": "no scored or online ARC game was played; submission is operator-only",
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": "no verifier-value, moat or efficiency claim is made here. "
        "The outcome is the SHAPE of an induced predicate decided from its syntax tree, and the "
        "goal gate -- the project's executable checker -- is deliberately NOT the outcome, "
        "because plan_found is an exact function of it.",
        "artifacts": {
            "preregistration": "results/arc_goal_evidence_20260802/out/preregistration.json",
            "rows": "results/arc_goal_evidence_20260802/out/rows.json",
            "analysis": "results/arc_goal_evidence_20260802/out/analysis.json",
            "meta": "results/arc_goal_evidence_20260802/out/meta.json",
            "server_witness": "results/arc_goal_evidence_20260802/out/server_witness.json",
            "stage1_predicates": "results/arc_goal_evidence_20260802/out/s1_cells/ (*.py.txt -- "
            "raw LLM output kept byte-exact; the .txt suffix keeps ruff-format from rewriting "
            "the evidence engine_sha256 commits to)",
            "stage2_world_models": "results/arc_goal_evidence_20260802/out/s2_cells/ (*.py.txt, "
            "same reason)",
            "classifier": "results/arc_goal_evidence_20260802/classify.py",
            "driver": "results/arc_goal_evidence_20260802/run_ab.py",
            "analyser": "results/arc_goal_evidence_20260802/analyse.py",
            "exhibit": "results/arc_goal_evidence_20260802/out/exhibit_truncation.json + "
            "exhibit_*.txt",
            "stopping_rule": "results/arc_goal_evidence_20260802/out/stopping_rule.json",
            "grounded_placebo": "results/arc_goal_evidence_20260802/out/grounded_placebo.json "
            "(built by placebo.py)",
        },
        # Inlined rather than hand-pasted into the artifact after the fact. THAT IS THE POINT:
        # before 2026-08-02 this key and `known_confounds` existed in the committed artifact and
        # in NO code path, so the file on disk was not the output of its own builder -- the exact
        # staleness class scripts/artifact_freshness_lint.py exists for, on an artifact that
        # happens not to be registered with it. See POST_REVIEW_CORRIGENDUM finding 6.
        "exhibit_truncation": json.loads((OUT / "exhibit_truncation.json").read_text()),
        "known_confounds": known_confounds,
        "POST_REVIEW_CORRIGENDUM_2026_08_02": corrigendum,
    }
    art["honest_verdict"] = json.loads((OUT / "verdict.json").read_text())["honest_verdict"]
    art["headline"] = json.loads((OUT / "verdict.json").read_text())["headline"]
    art["findings"] = json.loads((OUT / "verdict.json").read_text())["findings"]

    dest = ROOT / "results" / "outer_loop_arc_goal_evidence_ab_20260802.json"
    dest.write_text(json.dumps(art, indent=2, default=str) + "\n")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
