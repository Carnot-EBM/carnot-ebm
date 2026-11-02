#!/usr/bin/env python3
"""Build the pre-staged .446 roadmap implementing the operator's "c" directive
(2026-06-26): BOTH in parallel --

  (a) RESEARCH BET -- aim the A1 HEADLINE at the perception/representation root
      cause of the L1-first-contact wall. Grounded (NOT re-derived) in
      docs/research-notes/mechanic-template-and-perception-wall-2026-06-24.md:
      the precise, NARROWED blocker is GENERIC GOAL-PREDICATE GROUNDING --
      recovering stable structural OBJECTS (their identity across frames) from
      the rendered grid PRECISELY enough to evaluate an object-relational
      is_level_complete. The note's own recommended build: a generic,
      grid-grounded module that segments frames into persistent OBJECTS with
      stable identity across transitions tracked by shape/connectivity/MOTION
      (NOT color -- robust to lp85's full recolor, r11l's no-anchor, tu93's
      trail) where the naive color-centroid detector recovered NONE. Built as a
      PROBE/prototype per Phase-Prototype+Validation discipline (the note: "a
      bigger build than a single lever"); full live integration is .447+ iff the
      probe passes.

  (b) REALISTIC SCORED LEVER -- weight A2 (first-win BREADTH / level-up bank),
      A4 (held-out readiness), and B2 (Kaggle submission-package harden) toward
      maximizing OFFLINE banked levels + deadline submission readiness for the
      2026-06-30 ARC-AGI-3 deadline (~3.5 days out).

Energy program CONCLUDED (no live value) -- do NOT re-propose energy stages.
The exploration-prior class is CLOSED (.445 exp4831 fair-test nulled). The .446
frontier is perception/representation, the named root cause.
"""
import yaml

MILESTONE = "2026.06.446"
DATE = "2026-06-27"

OVR = (
    "2026-06-19 operator directive (ARC sprint, standing): routine continuation / "
    "reserved-slot task -- false-positive scope-match vs prior same-class tasks; "
    "structurally never a doomed rerun. Sprint majority-ARC + reserved "
    "infra/hardware/sota/capstone slots."
)


def task(d):
    d.setdefault("milestone", MILESTONE)
    d.setdefault("agent_type", "codex")
    d.setdefault("model", "gpt-5.5")
    return d


tasks = []

# ---------------------------------------------------------------- PHASE 0
tasks.append(task({
    "id": "exp4840-phase0",
    "title": (
        "PHASE 0 (TRANSITION) -- archive .445 -> activate .446; assert YAML parses + pre-test gate "
        "green; RESOLVE any poison pre-test. RECORD the .445 close-state (amortized-prior fair-test "
        "nulled -> exploration-prior class CLOSED; .446 attacks the perception/representation root cause)."
    ),
    "priority": "critical",
    "track": "transition",
    "max_turns": 40,
    "inference_substrate": "aggregation_from_upstream_artifacts",
    "deliverable": "results/experiment_4840_archive_445_activate_446.json",
    "operator_override": OVR + " (archive-v445-activate-v446).",
    "prompt": (
        f"CONTEXT ({DATE}). Milestone 2026.06.445 completed. Mechanically transition to {MILESTONE}, "
        "resolve any poison pre-test, RECORD the true .445 close-state. Routine (codex). Conductor may "
        "have already archived (no-op). You MUST still WRITE the deliverable.\n\n"
        "KEY .445 CLOSE-STATE: the AMORTIZED IN-CONTEXT-EXPLORATION PRIOR fair-test (exp4831) NULLED with "
        "a live archive (the exp4701 silent-bug fix held) -- no held-out first-win lift above 0.04. This "
        "CLOSES the exploration-prior class: ~15 generation/exploration levers tested, ~0 moved the L1 "
        "wall. The named root cause is PERCEPTION/REPRESENTATION. The .446 headline (A1) attacks it: a "
        "generic OBJECT-IDENTITY perception layer for GOAL-GROUNDING. Record "
        "exploration_prior_class_closed=true, energy_program_concluded=true.\n\n"
        "CONCRETE STEPS:\n"
        "  0. PRECONDITIONS: `.venv/bin/python -c \"import yaml; "
        "yaml.safe_load(open('{project_root}/research-roadmap-next.yaml')); print('ok')\"` exits 0; "
        "arc_solver_kit.offline_arcade() exits 0.\n"
        "  1. Archive .445 -> activate .446 (no-op if already done). Record the close-state flags.\n"
        "  2. Run the pre-test gate; resolve any poison test. WRITE the deliverable.\n\n"
        "REQUIRED ARTIFACT FIELDS (principle-annotated):\n"
        "  honest_verdict:\n"
        "    principle: \"terminal prefix; clean transition is complete_445_archived_446_activated_<state>.\"\n"
        "  inference_substrate:\n"
        "    principle: \"aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor).\"\n"
        "  exploration_prior_class_closed:\n"
        "    principle: \"the amortized-prior fair-test nulled with a live archive; the planner must NOT "
        "re-propose exploration-strategy levers -- the .446 frontier is perception/representation.\"\n"
        "  reproducible_total_levels:\n"
        "    principle: \"the authoritative ARC progress metric carried from the registry, not re-counted.\"\n"
    ),
}))

# ---------------------------------------------------------------- PHASE A1 (HEADLINE; research bet)
tasks.append(task({
    "id": "exp4841-a1",
    "title": (
        "PHASE A1 (ARC NORTH STAR; PRIMARY; HEADLINE; operator 'c' RESEARCH BET) -- GENERIC OBJECT-IDENTITY "
        "PERCEPTION LAYER for GOAL-GROUNDING. Build a grid-grounded module that segments each frame into "
        "persistent OBJECTS and tracks their identity across transitions by SHAPE/CONNECTIVITY/MOTION (NOT "
        "color -- robust to recoloring), exposing object-relational features. PROBE per "
        "Phase-Prototype+Validation. Decisive: recover stable cross-frame object correspondence on the "
        "complex-rendering games (lp85 full-recolor, r11l no-anchor) where the naive color-centroid detector "
        "got 0, with tu93 (visible goal) as a positive control. The named L1-wall root cause; a research bet "
        "(may not pay off before 6/30)."
    ),
    "priority": "critical",
    "track": "arc-perception",
    "max_turns": 200,
    "inference_substrate": "live_llm_inference",
    "deliverable": "results/experiment_4841_object_identity_perception_probe.json",
    "operator_override": (
        "2026-06-26 operator chose 'c' (both in parallel): aim the .446 HEADLINE at the "
        "perception/representation root cause as a RESEARCH BET. Grounded in "
        "mechanic-template-and-perception-wall-2026-06-24.md (the note's own recommended build: a generic "
        "object-identity perception layer tracked by shape/connectivity/motion, not color). NARROWED per "
        "that note's RECONCILIATION: the blocker is generic GOAL-PREDICATE GROUNDING (exact object-identity "
        "recovery), NOT value/discrimination (measured-adequate at LOO-AUROC 0.725). Not an energy/"
        "exploration-strategy lever (those are concluded/closed)."
    ),
    "prior_failures": [
        {
            "experiment_id": "lever4-mechanic-template-color-centroid-detector",
            "verdict": (
                "complete: color_centroid_goal_detector_recovers_none_on_complex_rendering_games "
                "(genuine null on real grids)"
            ),
            "addressed_by": (
                "The lever-#4 mechanic-template build (perception-wall note) used a naive COLOR-CENTROID "
                "goal detector: it recovered the structural goal condition on NONE of the three games -- "
                "lp85 (whole board recolors -- no stable color identity), r11l (zero stationary goal "
                "colors), tu93 (trail of new cells). This A1 is a DIFFERENT mechanism: track object "
                "identity by "
                "SHAPE/CONNECTIVITY/MOTION (connected-component + cross-frame correspondence by overlap/"
                "shape, recolor-INVARIANT), the exact build the note recommends ('track by shape/"
                "connectivity/motion, not color -- robust to recoloring'). retire_if_same_verdict: true -- "
                "if a shape/motion-based object tracker ALSO cannot recover stable cross-frame object "
                "correspondence on >=2 of the three games (lp85, r11l, tu93), then object-identity is "
                "genuinely "
                "unrecoverable from the RENDERED grid (the agent would need internal sprite state, which "
                "does not generalize to hidden games) -- a deeper, headline finding that retires "
                "grid-grounded goal-grounding."
            ),
            "retire_if_same_verdict": True,
        },
    ],
    "prompt": (
        f"CONTEXT ({DATE}). OPERATOR-DIRECTED 'c' RESEARCH BET. The energy program is CONCLUDED and the "
        "exploration-prior class is CLOSED (.445 exp4831 nulled with a live archive). ~15 generation/"
        "exploration levers tested, ~0 moved the L1-first-contact wall. The NAMED root cause is "
        "PERCEPTION/REPRESENTATION. This headline attacks it directly.\n\n"
        "THE PRECISE, NARROWED BLOCKER (read "
        "{project_root}/docs/research-notes/mechanic-template-and-perception-wall-2026-06-24.md FIRST, "
        "including its RECONCILIATION section): NOT 'perception' broadly -- the live VALUE/DISCRIMINATION "
        "representation is MEASURED-ADEQUATE (v3 features LOO-AUROC 0.725, exp4545). The blocker is generic "
        "GOAL-PREDICATE GROUNDING: recovering stable structural OBJECTS (their identity across frames) from "
        "the RENDERED grid precisely enough to evaluate an object-relational is_level_complete. The naive "
        "color-centroid detector recovered the goal on NONE of the three games -- lp85 (whole board "
        "RECOLORS -- no stable color identity), r11l (zero stationary goal colors), tu93 (trail of new "
        "cells) -- because object identity is lost under recoloring/trails. Per-game GameAdapters succeed "
        "only by reading INTERNAL "
        "sprite state, which does NOT generalize to hidden games.\n\n"
        "THE BUILD (the note's own recommendation): a generic, grid-grounded OBJECT-IDENTITY perception "
        "module. Segment each frame into persistent OBJECTS via connected-component / shape segmentation, "
        "then track identity ACROSS transitions by SHAPE + CONNECTIVITY + MOTION (overlap/displacement), "
        "NOT by color -- so it is INVARIANT to recoloring. Expose object-relational features (this object "
        "at that object + offset; player overlaps goal). This is the layer BOTH the mechanic-template "
        "goal_energy AND LLM goal-induction need to become grid-grounded-yet-object-aware.\n\n"
        "SCOPE: a PROBE/PROTOTYPE (Phase-Prototype+Validation discipline -- the note: 'a bigger build than "
        "a single lever'). Build the minimal tracker + MEASURE cross-frame object-identity correspondence; "
        "do NOT attempt a full live-solver integration this milestone (that is .447+ iff the probe passes). "
        "It MUST be live-path-reachable in principle (a module under python/carnot/agentic/ importable by "
        "the live agent; arc_orphan_solver_lint must pass -- either wire a thin import hook or allow-list "
        "with a reason).\n\n"
        "verifier_is_oracle: the object-identity tracker is execution-grounded structural perception, NOT "
        "an oracle-distinct learned energy -- declare verifier_is_oracle=true (per the circularity "
        "discipline); this probe is about GROUNDING, not a moat claim.\n\n"
        "CONCRETE STEPS:\n"
        "  0. PRECONDITIONS (BEFORE any other step):\n"
        "     a. `.venv/bin/python -c \"from carnot.agentic import arc_solver_kit as k; k.offline_arcade()\"` "
        "exits 0. Else blocked_offline_arcade_missing, EXIT.\n"
        "     b. The offline frame sources for lp85, r11l, tu93 are present (environment_files / banked "
        "traces). If a game's frames are missing -> record it and proceed with the available games; if "
        "NONE are available -> blocked_no_offline_frames, EXIT.\n"
        "  1. Build the object-identity perception module (connected-component segmentation + "
        "shape/connectivity/motion cross-frame correspondence; recolor-invariant).\n"
        "  2. POSITIVE CONTROL: on tu93 (fully-visible goal) confirm the tracker yields stable "
        "player+goal object identity across the solving transitions (the note flags tu93 as plausibly "
        "grid-groundable with a better-than-color-centroid detector).\n"
        "  3. HARD TEST: on lp85 (full recolor) and r11l (no stationary anchor), measure whether stable "
        "object correspondence is recovered ACROSS frames despite recoloring -- report a quantitative "
        "correspondence score (e.g. fraction of objects with a consistent identity track across the "
        "transition sequence), NOT a binary self-grade. The three test games are lp85, r11l, tu93.\n"
        "  4. WRITE the artifact with solve_provenance=development_proxy (this is an offline perception "
        "PROBE measured on banked frames -- it is NOT a live first-win; declare honestly).\n\n"
        "FALSIFIABLE GATE:\n"
        "  PASS iff (i) the positive control (tu93) yields stable player+goal identity tracks; AND (ii) on "
        ">=2 of the three test games (lp85, r11l, tu93) the shape/motion tracker recovers stable cross-frame "
        "object correspondence (correspondence score materially above a color-centroid baseline run on the SAME "
        "frames -- include that baseline as the comparator). retire_if_same_verdict: true -- if the "
        "shape/motion tracker ALSO fails on >=2 games (no better than color-centroid), object-identity is "
        "genuinely unrecoverable from the rendered grid: a deeper finding (the agent needs internal sprite "
        "state, which does not generalize) that retires grid-grounded goal-grounding and points .447 "
        "elsewhere. EITHER outcome is a real, headline-worthy result for the research bet.\n\n"
        "GUARD (the mechanic-template trap): that build was 'verified on synthetic data ... but on REAL "
        "grids exposed a deeper obstacle.' This probe MUST be measured on REAL game frames (lp85/r11l/tu93), "
        "NOT synthetic align cases. A synthetic-only PASS is a NON-TEST -- B1 will check this.\n\n"
        "REQUIRED ARTIFACT FIELDS (principle-annotated):\n"
        "  honest_verdict:\n"
        "    principle: \"terminal prefix; recovery is success_object_identity_perception_recovers_goal_grounding, "
        "a null is complete_object_identity_unrecoverable_from_rendered_grid_deeper_finding.\"\n"
        "  measured_on_real_frames:\n"
        "    principle: \"true iff the correspondence scores are measured on REAL lp85/r11l/tu93 frames, NOT "
        "synthetic -- the mechanic-template synthetic-only-pass trap; a false here is a NON-TEST.\"\n"
        "  per_game_correspondence:\n"
        "    principle: \"per-game mapping game -> (shape_motion_score, color_centroid_baseline_score, "
        "n_frames) -- the quantitative recovery measure vs the color-centroid comparator on the same "
        "frames.\"\n"
        "  positive_control_tu93_passed:\n"
        "    principle: \"the tu93 visible-goal control must yield stable player+goal identity tracks -- a "
        "Phase-Prototype positive control so a global null is not a harness artifact.\"\n"
        "  games_with_recovery:\n"
        "    principle: \"count of the three test games (lp85, r11l, tu93) where the shape/motion tracker "
        "materially beats color-centroid -- >=2 for a PASS.\"\n"
        "  verifier_is_oracle:\n"
        "    principle: \"true -- execution-grounded structural perception, NOT an oracle-distinct moat "
        "(circularity discipline); this probe is about grounding.\"\n"
        "  live_path_reachable:\n"
        "    principle: \"the module is importable by the live agent (arc_orphan_solver_lint passes) -- a "
        "perception layer the live agent cannot reach is wasted effort.\"\n"
        "  solve_provenance:\n"
        "    principle: \"development_proxy -- an offline perception probe on banked frames, NOT a live "
        "first-win; declared honestly (not live_agent_self_discovery).\"\n"
        "  inference_substrate:\n"
        "    principle: \"live_llm_inference if any LLM induction runs, else "
        "verifier_ensemble_against_cached_candidates -- declare what actually ran (the probe is mostly "
        "CPU segmentation; do not claim live inference if none ran).\"\n"
        "  preconditions_checked:\n"
        "    principle: \"records arcade/frame-availability checks so a missing-resource run emits "
        "blocked_, never a fabricated correspondence score.\"\n"
        "  random_seed:\n"
        "    principle: \"determinism for any stochastic segmentation/tracking step + the baseline.\"\n"
        "  reproducibility_checksum:\n"
        "    principle: \"content hash of (frames, tracker params, baseline) so a replication catches drift.\"\n"
    ),
}))

# ---------------------------------------------------------------- PHASE A2 (level-up; scored lever)
tasks.append(task({
    "id": "exp4842-a2",
    "title": (
        "PHASE A2 (ARC NORTH STAR; LEVEL-UP ATTEMPT GUARANTEE; operator 'c' SCORED LEVER) -- maximize "
        "OFFLINE banked levels: bank >=1 NEW reproducible level on a ROTATED unsolved target "
        "(sb26/lf52/bp35 first-contact, else +1 deepen on the shallowest solved game). offline_reproduced gated."
    ),
    "priority": "critical",
    "track": "arc-north-star",
    "max_turns": 160,
    "inference_substrate": "live_llm_inference",
    "deliverable": "results/experiment_4842_levelup_attempt.json",
    "operator_override": OVR + " (Level-Up Attempt Guarantee; operator 'c' scored lever; rotate sb26/lf52/bp35).",
    "prompt": (
        f"CONTEXT ({DATE}). Level-Up Attempt Guarantee + operator 'c' SCORED LEVER: maximize OFFLINE banked "
        "levels (the realistic scored play for the 6/30 deadline). >=1 task whose gate BANKS a new "
        "reproducible level. ROTATE the target (sb26/lf52/bp35 first-contact -- different from .445's "
        "bp35-class rotation; else +1 deepen on the shallowest solved game).\n\n"
        "CONCRETE STEPS:\n"
        "  0. PRECONDITIONS: arc_solver_kit.offline_arcade() exits 0; Qwen on the iGPU (NEVER the 3090s) if "
        "induction is needed; missing env -> blocked_<game>_offline_env_missing.\n"
        "  1. Read ops/arc_solve_registry.yaml; pick the rotated target; recommend_approach(game).\n"
        "  2. arc_loop_solve.py --game <target>; register a GameAdapter for the DELTA if needed.\n"
        "  3. Reproduction gate; update registry; record DEAD-ENDS.\n\n"
        "FALSIFIABLE GATE: >=1 NEW level offline-reproduced, solve_provenance=live_agent_self_discovery. "
        "retire_if_same_verdict: true -- rotate again if no bank.\n\n"
        "REQUIRED ARTIFACT FIELDS (principle-annotated):\n"
        "  honest_verdict:\n"
        "    principle: \"terminal prefix; banked is success_, no-bank is "
        "complete_<game>_no_new_level_residual_<cause>.\"\n"
        "  solve_provenance:\n"
        "    principle: \"live_agent_self_discovery; NOT outer_loop_re (CRITICAL).\"\n"
        "  offline_reproduced:\n"
        "    principle: \"only reproduced levels count.\"\n"
        "  reproduced_levels:\n"
        "    principle: \"the new reproducible depth; the monotonic ARC metric.\"\n"
        "  inference_substrate:\n"
        "    principle: \"live_llm_inference if induction runs (60s floor).\"\n"
        "  preconditions_checked:\n"
        "    principle: \"records arcade/env/generator checks; a missing resource emits blocked_, never a "
        "fabricated solve.\"\n"
    ),
}))

# ---------------------------------------------------------------- PHASE A3 (self-play; standing)
tasks.append(task({
    "id": "exp4843-a3",
    "title": (
        "PHASE A3 (ARC NORTH STAR; self-play EVERY milestone) -- standing arc_loop_solve loop: "
        "verifier-routed solve -> reproduction gate -> TRAIN + CHECKPOINT the learned verifier."
    ),
    "priority": "high",
    "track": "arc-north-star",
    "max_turns": 140,
    "inference_substrate": "live_llm_inference",
    "deliverable": "results/experiment_4843_self_play_verifier_checkpoint.json",
    "operator_override": OVR + " (self-play-every-milestone).",
    "prompt": (
        f"CONTEXT ({DATE}). Self-play EVERY milestone. Run arc_loop_solve.py on a banked game (warm-started) "
        "to advance +1 level AND refresh the checkpoint.\n\n"
        "CONCRETE STEPS:\n"
        "  0. PRECONDITIONS: arc_solver_kit.offline_arcade() exits 0; target has a banked level.\n"
        "  1. arc_loop_solve.py --game <banked-target> (or --auto): warm-start, verifier-routed search + "
        "reproduction gate, train + checkpoint.\n"
        "  2. Confirm the checkpoint mtime advanced.\n\n"
        "FALSIFIABLE GATE: checkpoint refreshed (mtime advances) AND reproduction gate passes on >=1 level. "
        "A failed run records the residual; does NOT fabricate.\n\n"
        "REQUIRED ARTIFACT FIELDS (principle-annotated):\n"
        "  honest_verdict:\n"
        "    principle: \"terminal prefix; refreshed + gate green is success_.\"\n"
        "  verifier_checkpoint_refreshed:\n"
        "    principle: \"the self-improvement signal.\"\n"
        "  inference_substrate:\n"
        "    principle: \"live_llm_inference (60s floor).\"\n"
        "  solve_provenance:\n"
        "    principle: \"live_agent_self_discovery -- self-play is the agent improving on its own attempts.\"\n"
        "  preconditions_checked:\n"
        "    principle: \"records arcade/registry checks; a missing target emits blocked_, never a "
        "fabricated checkpoint.\"\n"
    ),
}))

# ---------------------------------------------------------------- PHASE A4 (readiness; deadline-relevant)
tasks.append(task({
    "id": "exp4844-a4",
    "title": (
        "PHASE A4 (ARC NORTH STAR; SCORE -- held-out first-win readiness; operator 'c' DEADLINE LANE) -- "
        "measure held-out first-win on the color-permuted variant harness, checkpoint/resume-capable; feeds "
        "the operator submission decision."
    ),
    "priority": "high",
    "track": "arc-north-star",
    "max_turns": 120,
    "inference_substrate": "live_llm_inference",
    "deliverable": "results/experiment_4844_heldout_first_win_readiness.json",
    "operator_override": OVR + " (held-out first-win readiness lane; operator 'c' deadline-submission signal).",
    "prompt": (
        f"CONTEXT ({DATE}). Measure the live E3 agent's held-out first-win rate on the experiment_4605 "
        "variant harness. Checkpoint/resume so a capped run still emits a partial. This is the "
        "deadline-relevant generalization signal that feeds the operator's submission go/no-go.\n\n"
        "CONCRETE STEPS:\n"
        "  0. PRECONDITIONS: arcade + variant harness present; Qwen on the iGPU (NEVER the 3090s). Scope "
        "under the codex cap (or checkpoint/resume).\n"
        "  1. Run with periodic checkpointing; on cap emit the partial.\n"
        "  2. Report rate + CI vs the 0.04 baseline.\n"
        "  3. SUBSTRATE HONESTY: if you RESUME from a cache and do NOT run live, declare "
        "inference_substrate=aggregation_from_upstream_artifacts. For a FLAT null (rate==0.04) add "
        "null_delta_methodology_note + positive_control_passed=true.\n\n"
        "FALSIFIABLE GATE: a non-fabricated rate with CI; a capped run still emits a partial.\n\n"
        "REQUIRED ARTIFACT FIELDS (principle-annotated):\n"
        "  honest_verdict:\n"
        "    principle: \"terminal prefix; a measured rate is complete_/success_.\"\n"
        "  heldout_first_win_rate:\n"
        "    principle: \"the deadline-relevant generalization signal -- held-out.\"\n"
        "  inference_substrate:\n"
        "    principle: \"live_llm_inference (60s floor) if live; aggregation_from_upstream_artifacts if a "
        "cache hit -- declare what ran.\"\n"
        "  null_delta_methodology_note:\n"
        "    principle: \"for a flat null (rate==0.04), why the agreement is a genuine no-improvement, not a "
        "TAUTOLOGY bug.\"\n"
        "  checkpoint_emitted:\n"
        "    principle: \"a capped run must still emit a usable partial.\"\n"
        "  preconditions_checked:\n"
        "    principle: \"records generator/harness checks; a missing resource emits blocked_.\"\n"
    ),
}))

# ---------------------------------------------------------------- PHASE B1 (infra; adversarial audit of A1)
tasks.append(task({
    "id": "exp4845-b1",
    "title": (
        "PHASE B1 (INFRA slot 1; ADVERSARIAL CHECK) -- VERIFY the A1 object-identity perception probe was "
        "GENUINELY exercised on REAL frames (measured_on_real_frames=true, NOT the mechanic-template "
        "synthetic-only-pass trap), the shape/motion tracker actually ran (not a no-op), and per-game "
        "correspondence beats the color-centroid baseline on the SAME frames."
    ),
    "priority": "high",
    "track": "infra",
    "max_turns": 100,
    "inference_substrate": "aggregation_from_upstream_artifacts",
    "deliverable": "results/experiment_4845_perception_probe_audit.json",
    "operator_override": OVR + " (reserved infra slot 1; A1 perception-probe adversarial check -- real-frames + not-a-no-op).",
    "prompt": (
        f"CONTEXT ({DATE}). Reserved infra slot -- the Phase-Prototype+Validation ADVERSARIAL CHECK for the "
        "A1 perception probe (exp4841). The mechanic-template precedent: a build that 'verified on synthetic "
        "data ... but on REAL grids exposed a deeper obstacle' -- a synthetic-only pass is a NON-TEST. "
        "Audit A1 hostilely:\n"
        "  (1) VERIFY measured_on_real_frames=true -- the correspondence scores come from REAL lp85/r11l/tu93 "
        "frames, NOT synthetic align cases. If synthetic-only, A1 is a NON-TEST.\n"
        "  (2) VERIFY the shape/motion tracker materially CHANGED the result vs the color-centroid baseline "
        "(per_game_correspondence shows a real spread, not the tracker silently degenerating to the "
        "baseline -- the S2/S3 no-op lesson applied to perception).\n"
        "  (3) VERIFY the positive control (tu93) genuinely passed AND the >=2-game recovery claim matches "
        "the per-game numbers (no celebratory verdict over failing numbers).\n"
        "  (4) Confirm live_path_reachable (arc_orphan_solver_lint passes) and solve_provenance is the "
        "honest development_proxy (NOT live_agent_self_discovery -- it is an offline probe).\n\n"
        "CONCRETE STEPS:\n"
        "  0. PRECONDITIONS: the .446 A1 artifact present.\n"
        "  1. Read it via scripts/summarize_artifact.py; run the four checks; run "
        "scripts/adversarial_verify.py on it.\n"
        "  2. WRITE the audit appending to ops/arc_null_silent_bug_audit.md.\n\n"
        "REQUIRED ARTIFACT FIELDS (principle-annotated):\n"
        "  honest_verdict:\n"
        "    principle: \"terminal prefix; audit complete is complete_/success_.\"\n"
        "  a1_genuinely_exercised:\n"
        "    principle: \"the load-bearing check -- measured_on_real_frames AND tracker!=baseline no-op AND "
        "positive control real AND verdict matches numbers; else A1 is a non-test (synthetic-only or a "
        "silent degenerate-to-baseline).\"\n"
        "  inference_substrate:\n"
        "    principle: \"aggregation_from_upstream_artifacts (0.0001s floor).\"\n"
    ),
}))

# ---------------------------------------------------------------- PHASE B2 (infra; DEADLINE submission harden)
tasks.append(task({
    "id": "exp4846-b2",
    "title": (
        "PHASE B2 (INFRA slot 2; DEADLINE-RELEVANT; operator 'c' SCORED LEVER) -- SUBMISSION-PACKAGE harden: "
        "verify the Kaggle ARC-AGI-3 package builds + the frozen Qwen3.5-9B-MTP stack loads under ~16GB; "
        "produce the OPERATOR submission checklist (NEVER submits)."
    ),
    "priority": "high",
    "track": "infra",
    "max_turns": 100,
    "inference_substrate": "aggregation_from_upstream_artifacts",
    "deliverable": "results/experiment_4846_submission_package_harden.json",
    "operator_override": OVR + " (reserved infra slot 2; operator 'c' deadline-submission readiness; operator-only submit).",
    "prompt": (
        f"CONTEXT ({DATE}). Reserved infra slot, DEADLINE-RELEVANT (operator 'c' realistic scored lever; 6/30 "
        "is ~3.5 days out). Verify the ARC-AGI-3 Kaggle package builds and the frozen live stack packages "
        "within ~16GB. Produce an OPERATOR checklist -- the task NEVER submits (Operator-Only External "
        "Publication discipline).\n\n"
        "CONCRETE STEPS:\n"
        "  0. PRECONDITIONS: the submission packaging scripts + arc_competition_agent present.\n"
        "  1. Dry-build; verify the agent config + model paths resolve; estimate VRAM vs 16GB (Qwen3.5-9B "
        "Q4 ~5.9GB weights + q8 KV + headroom).\n"
        "  2. Cross-check the package against docs/research-notes/"
        "arc-agi3-kaggle-submission-requirements-2026-06-17.md (the packaging spec).\n"
        "  3. WRITE the operator checklist; the task does not submit.\n\n"
        "REQUIRED ARTIFACT FIELDS (principle-annotated):\n"
        "  honest_verdict:\n"
        "    principle: \"terminal prefix; package builds is success_/complete_.\"\n"
        "  submission_package_ready:\n"
        "    principle: \"True iff ready for the OPERATOR to submit; the task NEVER submits.\"\n"
        "  vram_estimate_gb:\n"
        "    principle: \"must fit ~16GB Kaggle with KV + headroom.\"\n"
        "  inference_substrate:\n"
        "    principle: \"aggregation_from_upstream_artifacts (0.0001s floor).\"\n"
    ),
}))

# ---------------------------------------------------------------- PHASE C (hardware; KV260)
tasks.append(task({
    "id": "exp4847-c",
    "title": (
        "PHASE C (HARDWARE -- KV260) -- SSH-reachability; ALWAYS write the deliverable "
        "(blocked_kv260_ssh_unreachable on failure -- do NOT exit with no file changes). Board known offline "
        "2026-06-26 (no route to 192.168.51.98)."
    ),
    "priority": "medium",
    "track": "hardware",
    "max_turns": 60,
    "inference_substrate": "hardware_smoke",
    "deliverable": "results/experiment_4847_kv260_continuity.json",
    "operator_override": OVR + " (hardware-continuity; KV260; always write a deliverable; board offline noted).",
    "prompt": (
        f"CONTEXT ({DATE}). KV260 hardware-continuity. SSH ONLY -- host SD-card device nodes are PERMANENTLY "
        "retired.\n\n"
        "CRITICAL: the board is known OFFLINE (no route to 192.168.51.98; kria->kv260.local unresolved). "
        "Prior KV260 tasks FAILED 3x with \"No file changes\" because the agent exited on SSH failure "
        "WITHOUT writing the deliverable. You MUST ALWAYS WRITE results/experiment_4847_kv260_continuity.json "
        "-- on SSH failure write it with honest_verdict blocked_kv260_ssh_unreachable. NEVER exit without "
        "writing (that 3-fail-skips). A blocked artifact is the EXPECTED, CORRECT outcome.\n\n"
        "CONCRETE STEPS:\n"
        "  0. PRECONDITIONS: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. If non-zero -> "
        "honest_verdict blocked_kv260_ssh_unreachable + kv260_ssh_reachable=false, GO TO STEP 2 (do NOT "
        "exit before writing).\n"
        "  1. If reachable: report the overlay + board state; record the next step.\n"
        "  2. ALWAYS WRITE the deliverable.\n\n"
        "REQUIRED ARTIFACT FIELDS (principle-annotated):\n"
        "  honest_verdict:\n"
        "    principle: \"terminal prefix; reachable is success_/complete_; unreachable is "
        "blocked_kv260_ssh_unreachable -- artifact MUST be written either way (no 'No file changes' "
        "3-fail-skip).\"\n"
        "  inference_substrate:\n"
        "    principle: \"hardware_smoke (SSH-attached board test).\"\n"
        "  kv260_ssh_reachable:\n"
        "    principle: \"the SSH check -- false when offline; the artifact is still written.\"\n"
        "  preconditions_checked:\n"
        "    principle: \"records the SSH check; a blocked artifact is correct when the board is offline.\"\n"
    ),
}))

# ---------------------------------------------------------------- PHASE D (SOTA-ingestion; .447 frontier)
tasks.append(task({
    "id": "exp4848-d",
    "title": (
        "PHASE D (SOTA-INGESTION -- the .447 frontier) -- ingest object-centric WORLD-MODEL + structured "
        "PLANNING methods that CONSUME an object-identity perception layer (the .446 A1 build), to turn "
        "recovered object structure into a proposable WINNER on a novel game. Real arXiv IDs only; do NOT "
        "re-ingest the nulled exploration-strategy class."
    ),
    "priority": "medium",
    "track": "sota-ingestion",
    "max_turns": 100,
    "inference_substrate": "aggregation_from_upstream_artifacts",
    "deliverable": "results/experiment_4848_sota_ingestion_object_world_model.json",
    "operator_override": OVR + " (reserved SOTA-ingestion slot; .447 frontier = object-centric world-model/planning that consumes the A1 perception layer).",
    "prompt": (
        f"CONTEXT ({DATE}). Reserved SOTA-ingestion slot. The .446 headline (A1) builds a generic "
        "OBJECT-IDENTITY perception layer for goal-grounding. The NEXT-layer question (.447 frontier): given "
        "recovered object structure, how do you turn it into a PROPOSABLE WINNER on a novel game? Ingest "
        "object-centric WORLD-MODELS and structured/relational PLANNING that consume an object-relational "
        "state: object-centric / slot-based world models, relational/graph planners, object-relational "
        "dynamics learning, structured exploration over object affordances. ALSO read the .445 D output "
        "(results/experiment_4838_sota_ingestion_perception_representation.json -- the perception methods "
        "flagged_for_v446) and carry the strongest forward. Use the RELIABLE channel "
        "(sweep_clusters.py / sweep_semscholar.py + low-concurrency WebSearch/WebFetch). Do NOT invoke "
        "/deep-research. Do NOT re-ingest the nulled exploration-strategy class (it is CLOSED).\n\n"
        "CONCRETE STEPS:\n"
        "  0. PRECONDITIONS: research-studying.md / research-references.md present; the .445 D artifact if it "
        "landed.\n"
        "  1. Read the discovered corpus filtered to object-centric world-models + relational planning; "
        "focused fresh sweep (top 5-8 papers).\n"
        "  2. Synthesize a SOTA->experiment mapping note: how each method consumes the A1 object layer to "
        "produce a proposable winner. Cite REAL arXiv IDs.\n"
        "  3. Flag the strongest method(s) for the .447 roadmap; mark ingested in research-studying.md.\n\n"
        "REQUIRED ARTIFACT FIELDS (principle-annotated):\n"
        "  honest_verdict:\n"
        "    principle: \"terminal prefix; mapping emitted is "
        "success_sota_ingestion_object_world_model_mapped.\"\n"
        "  methods_mapped:\n"
        "    principle: \"the strongest 3-5 object-centric world-model/planning methods mapped onto "
        "consuming the A1 perception layer, each with a real arXiv ID.\"\n"
        "  arxiv_ids_cited:\n"
        "    principle: \"every method claim must cite a verifiable arXiv ID.\"\n"
        "  flagged_for_v447:\n"
        "    principle: \"the strongest method(s) flagged so the .447 planner reads the mapping.\"\n"
        "  inference_substrate:\n"
        "    principle: \"aggregation_from_upstream_artifacts (0.0001s floor).\"\n"
    ),
}))

# ---------------------------------------------------------------- PHASE E (capstone)
tasks.append(task({
    "id": "exp4849-e",
    "title": (
        "PHASE E (CAPSTONE .446) -- aggregate the scorecard: the A1 object-identity perception-probe verdict "
        "(recovery -> goal-grounding feasible / genuine null -> object-identity unrecoverable from rendered "
        "grid, a deeper finding, per B1's real-frames check), level-up bank, self-play, readiness, "
        "submission-package state."
    ),
    "priority": "high",
    "track": "capstone",
    "max_turns": 120,
    "inference_substrate": "aggregation_from_upstream_artifacts",
    "deliverable": "results/experiment_4849_capstone_v446.json",
    "operator_override": OVR + " (capstone aggregation; codex default).",
    "prompt": (
        f"CONTEXT ({DATE}). Capstone for .446. The HEADLINE is the A1 object-identity perception-probe "
        "verdict (operator 'c' RESEARCH BET): does the shape/motion object tracker recover stable "
        "cross-frame object correspondence on >=2 of {{lp85, r11l, tu93}} (goal-grounding becomes feasible "
        "-> .447 wires it into the live solver), or is it a GENUINE null (object-identity unrecoverable from "
        "the rendered grid even by shape/motion -> the agent needs internal sprite state, which does not "
        "generalize -- a deeper, headline finding)? ALSO report the operator 'c' SCORED-LEVER state: "
        "level-up bank (A2), held-out readiness (A4), submission-package readiness (B2).\n\n"
        "IMPORTANT: read each upstream artifact via scripts/summarize_artifact.py. The A1 verdict is ONLY "
        "trustworthy if B1 confirmed measured_on_real_frames AND tracker!=baseline-no-op (else it is a "
        "synthetic-only or degenerate NON-TEST, per the mechanic-template precedent).\n\n"
        "CONCRETE STEPS:\n"
        "  0. PRECONDITIONS: the .446 upstream artifacts present.\n"
        "  1. Read each via summarize_artifact; honor the A1 real-frames + not-a-no-op checks (per B1).\n"
        "  2. Aggregate: A1 perception verdict; level-up; self-play; readiness; submission state. Cite "
        "sha256/ids.\n"
        "  3. WRITE the capstone scorecard.\n\n"
        "REQUIRED ARTIFACT FIELDS (principle-annotated):\n"
        "  honest_verdict:\n"
        "    principle: \"terminal prefix; capstone_ready=true is success_/complete_.\"\n"
        "  a1_perception_probe_verdict:\n"
        "    principle: \"the headline -- object-identity recovery (goal-grounding feasible) / genuine null "
        "(unrecoverable from rendered grid, deeper finding) / synthetic-only non-test; honors B1's "
        "real-frames + not-a-no-op checks.\"\n"
        "  scored_lever_state:\n"
        "    principle: \"the operator 'c' deadline track -- {{level_up_banked, heldout_first_win_rate, "
        "submission_package_ready}}; the realistic 6/30 signal.\"\n"
        "  reproducible_total_levels:\n"
        "    principle: \"the monotonic ARC progress metric carried from the registry.\"\n"
        "  cited_upstream_artifacts:\n"
        "    principle: \"list of {{experiment_id, fields_imported, sha256}} -- the audit trail.\"\n"
        "  inference_substrate:\n"
        "    principle: \"aggregation_from_upstream_artifacts (0.0001s floor).\"\n"
    ),
}))

roadmap = {
    "milestone": MILESTONE,
    "planned_by": (
        "outer-loop (Claude Opus 4.8 planner, 2026-06-26). Operator chose 'c' (both in parallel): "
        "(a) RESEARCH BET -- aim the A1 headline at the perception/representation root cause of the "
        "L1-first-contact wall (a generic OBJECT-IDENTITY perception layer for goal-grounding, the "
        "perception-wall note's own recommended build, NARROWED to goal-grounding per its RECONCILIATION); "
        "(b) SCORED LEVER -- weight A2 first-win-breadth + A4 readiness + B2 submission-harden toward "
        "maximizing OFFLINE banked levels + 6/30 submission readiness. Energy program CONCLUDED; "
        "exploration-prior class CLOSED (.445 exp4831 nulled with a live archive)."
    ),
    "milestone_title": (
        "OPERATOR 'c' (BOTH): A1 RESEARCH BET on the perception/representation ROOT CAUSE -- a generic "
        "OBJECT-IDENTITY perception layer (shape/connectivity/MOTION, recolor-invariant) for GOAL-GROUNDING, "
        "the named L1-wall blocker; measured on REAL lp85/r11l/tu93 frames where the color-centroid detector "
        "recovered NONE, tu93 as positive control. PLUS the SCORED LEVER: maximize OFFLINE banked levels "
        "(A2 first-win breadth) + 6/30 submission readiness (A4 held-out, B2 Kaggle package harden). "
        "Exploration-prior class CLOSED; energy CONCLUDED. PASS (>=2 games recover) -> goal-grounding "
        "feasible, .447 wires it live; genuine null -> object-identity unrecoverable from the rendered grid "
        "(needs internal sprite state, does not generalize) -- a deeper headline finding."
    ),
    "milestone_doc": "docs/research-notes/mechanic-template-and-perception-wall-2026-06-24.md",
    "theme": (
        "The two big bets are negative (energy CONCLUDED -- offline discriminator, no live value; "
        "exploration-prior class CLOSED -- .445 exp4831 nulled with a live archive). ~15 generation/"
        "exploration levers tested, ~0 moved the L1-first-contact wall. The NAMED root cause is "
        "PERCEPTION/REPRESENTATION -- precisely (per the perception-wall note's RECONCILIATION) generic "
        "GOAL-PREDICATE GROUNDING: recovering stable object identity from the RENDERED grid (value/"
        "discrimination is measured-adequate at LOO-AUROC 0.725; goal-grounding is the gap). Operator 'c' "
        "runs BOTH tracks: A1 attacks the root cause as a research bet (the object-identity perception "
        "layer, may not pay off before 6/30); A2/A4/B2 maximize the realistic scored lever (offline banked "
        "levels + submission readiness). Phase-Prototype discipline: A1 is a PROBE with a tu93 positive "
        "control + a color-centroid baseline comparator + a B1 real-frames adversarial check (the "
        "mechanic-template synthetic-only-pass trap)."
    ),
    "sprint_forcing_function": (
        "ARC-AGI-3 submission sprint through 2026-06-30 (CLAUDE.md). Majority ARC (A1 perception root-cause "
        "probe + A2 level-up + A3 self-play + A4 readiness); >=1 level-up attempt (Level-Up Attempt "
        "Guarantee -- A2, rotated sb26/lf52/bp35); self-play EVERY milestone (A3); 2 reserved infra (B1 A1 "
        "perception-probe adversarial check, B2 submission harden); 1 hardware continuity (C KV260, "
        "SSH-only, always-write-blocked-artifact); 1 SOTA-ingestion (D object-centric world-model/planning, "
        "the .447 frontier). The energy track is CONCLUDED and the exploration-prior class is CLOSED -- the "
        "planner must NOT re-propose energy stages or exploration-strategy levers."
    ),
    "tasks": tasks,
}

with open("research-roadmap-next.yaml", "w") as f:
    f.write(
        "# Pre-staged .446 roadmap (Pre-Staged Roadmap Convention). Operator chose 'c' (both in parallel):\n"
        "# (a) RESEARCH BET on the perception/representation root cause (A1 object-identity perception layer)\n"
        "# + (b) SCORED LEVER maximizing offline banked levels + 6/30 submission readiness (A2/A4/B2).\n"
        "# Built by scripts/_build_roadmap_446.py.\n"
    )
    yaml.safe_dump(roadmap, f, sort_keys=False, default_flow_style=False, width=110, allow_unicode=True)

print("wrote research-roadmap-next.yaml for", MILESTONE, "with", len(tasks), "tasks")
