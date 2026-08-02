#!/usr/bin/env python3
"""Does giving the GOAL prompt the agent's own evidence stop the model DECLINING to write a win
condition?  Two staged measurements against one generator, both on shipped code paths.

THE DEFECT UNDER TEST. Of 138 induced engines from 21 games, 93 are live and 71 fail on the
GOAL rather than the dynamics.  34 of those 71 are an unconditional `return False`, and four say
why in their own docstring -- "Given no WIN STATE grid was provided...".  `is_level_complete`
appears exactly ONCE in a ~5,000-character induction prompt, as an interface stub, and the
focused goal-only prompt that the split-induce fallback uses carries no transitions at all.
Given zero information about the goal, `return False` is arguably the honest answer.

TWO KNOBS ALREADY EXIST AND BOTH SHIP OFF.  `CARNOT_ARC_GOAL_PROMPT_TRANSITIONS=1` attaches the
observed transitions to `_goal_only_prompt`; `CARNOT_ARC_GOAL_DEDUP=1` stops an
evidence-grounded predicate being SHADOWED by an evidence-free one.  Nobody has measured whether
either helps.  That is this run.  NEITHER DEFAULT IS CHANGED HERE.

WHY TWO STAGES, decided from a PRE-FLIGHT on the frozen shipped-path corpus and written down
before any LLM call.  Classifying all 116 frozen engines with the same AST classifier this run
uses gives: DECLINED 43.1%, and a split-induce rate of 19.8%.  Crucially the declines are NOT
where the knobs are -- declined is 49.5% on COMBINED-path cells and only 17.4% on split-induce
cells.  Both knobs live exclusively in the split fallback, and the goal-only prompt is never
built at all when the combined call succeeds.  So on the live path the maximum arithmetically
reachable ITT movement of the declined rate is 0.198 x 0.174 = 3.4 points, and no design at this
n can resolve that.  STATED UP FRONT, NOT DISCOVERED AFTERWARDS:

  STAGE 2 (the live-path ITT, the arm structure the brief specifies) CANNOT REACH p<0.05 ON THE
  PRIMARY.  It is run anyway, because "the shipped knobs are structurally unable to move the
  headline defect" is a real and reportable answer, and because the per-cell mechanism-fired
  column is what distinguishes an INERT arm from a REFUTED one.

  STAGE 1 exists so the question itself gets a powered answer.  It calls the SAME shipped
  `_goal_only_prompt` and the SAME shipped `generate(required=("is_level_complete",))` that the
  split path calls, directly, once per (game, replicate, arm) -- so the mechanism fires in 100%
  of cells by construction and the contrast is measured rather than diluted to nothing.  It is a
  COMPONENT measurement of a live-path component and is labelled as such; it is not a claim
  about end-to-end rates.

THE PRIMARY IS PREDICATE SHAPE, NOT THE GOAL GATE.  `plan_found` is an exact function of the
goal gate's kind (0 mismatches in 138), so grading a goal intervention with it grades the gate
using the gate; that error sank three experiments this week.  Shape is decided from the syntax
tree by `classify.py`: DECLINED (constant False / no return / raises), TROPE (whole-board
uniformity or colour elimination), GROUNDED (names a region, cell or object that actually
appears in the agent's own observed deltas).  Gate kind is not measured here at all.

NOTHING CROSSES THE LINE.  The only thing added to any prompt is the agent's OWN observed
transitions, rendered by the SAME `_transitions_block` the engine prompt already uses.  No game
source, no adapter, no curated win example, no `_previous_level_complete_grid` asserted as a win
state (it is the NEXT level's opening board -- the win-state poison corrected 2026-07-29; this
harness passes None for it in both arms).  The transitions shown are the PREFIX split, which
excludes the level-up row, so no arm is shown a win.

CORRECTED 2026-08-02 (post-review, same day).  The paragraph above used to end "...Everything
here is exactly as available on a game nobody has ever solved -- which is why stall games with
zero observed wins are in the roster on purpose."  BOTH HALVES OF THAT ARE WRONG and the
retraction is kept here rather than quietly deleted:

  * There are no stall games in the roster.  All 20 are `full_game_clear: true` in
    ops/arc_solve_registry.yaml, and every window is built by `build_progress_window` ->
    `exp5717.build_window` -> `arc_loop_solve.solve_adaptered(game, 1)`, which returns None
    unless the game solves to L1 offline through a registered GameAdapter.  The window builder
    logged `levelups=1` for all 20 and `split_meta` records `levelup_rows_in_heldout: 1` for
    all 20.  The TRUE claim is the narrower one: `levelup_rows_in_shown == 0`, i.e. no ARM was
    shown a win.
  * Prompt CONTENT is clean, but transition SELECTION is solve-conditioned: the window is the
    last k actions of a BANKED WINNING ROUTE cut at the L0->L1 boundary.  On a hidden game the
    live `trans` at a `_split_induce` call is a stall-triggered exploration buffer instead --
    the same FIELD from a different DISTRIBUTION.  So this harness cannot support
    `works_on_an_unsolved_game`, and the artifact now reports it False.

The string this file EMITS into out/preregistration.json is deliberately left byte-identical so
`prereg_sha256` still verifies -- a pre-registration rewritten after seeing data is not a
pre-registration.  The correction lives in the artifact's
POST_REVIEW_CORRIGENDUM_2026_08_02 and in the comment beside the emit site below.

NOT SUBMITTED: no scored or online ARC game is played.  Submission is operator-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Derived, never hardcoded (CLAUDE.md Test-Run Record Integrity rule 4): a baked-in absolute
# path makes a fresh clone write into the operator's checkout.
ROOT = HERE.parents[1]

# ---- isolation: E3_DIR is read at IMPORT time, so it must be set before the import ----------
# `_guard_engine_write` is PYTEST-SCOPED, so a measurement driver is precisely the caller
# nothing protects: a sibling run rewrote results/arc_e3/<game>/world_model.py -- tracked,
# read-only EVIDENCE -- inside 90 seconds. A per-run scratch store also stops the arms sharing
# a store, which would be a cross-arm confound on top of the data loss.
SCRATCH = Path(
    os.environ.get("GEV_SCRATCH")
    or "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/goalev/e3_store"
)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["CARNOT_ARC_E3_DIR"] = str(SCRATCH)

if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HERE))

import classify as gcl  # noqa: E402

PORT = int(os.environ.get("GEV_PORT", "41871"))
GPU = os.environ.get("GEV_GPU", "0")
S1_REPS = int(os.environ.get("GEV_S1_REPS", "5"))
S2_REPS = int(os.environ.get("GEV_S2_REPS", "3"))
SEED_BASE = int(os.environ.get("GEV_SEED_BASE", "8300"))
AA_SEED_BASE = int(os.environ.get("GEV_AA_SEED_BASE", "8600"))
WALL_BUDGET_S = float(os.environ.get("GEV_WALL_BUDGET_S", "39600"))
WINDOWS_PKL = Path(
    os.environ.get("GEV_WINDOWS")
    or "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/goalev/windows.pkl"
)

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
# Captured completions are written as `.py.txt`, NOT `.py`. They are LLM-generated
# EVIDENCE whose exact bytes are what `engine_sha256` commits to, and the repo's
# `ruff-format --check` pre-commit hook would refuse the commit until they were
# reformatted -- which would silently rewrite the artefact the analysis hashes. The
# repo already carves `results/arc_e3/` out of ruff for exactly this reason; a suffix
# the formatter does not claim is the same fix without editing shared config.
(OUT / "s1_cells").mkdir(exist_ok=True)
(OUT / "s2_cells").mkdir(exist_ok=True)

FLAG_PROMPT = "CARNOT_ARC_GOAL_PROMPT_TRANSITIONS"
FLAG_DEDUP = "CARNOT_ARC_GOAL_DEDUP"

ROSTER = [
    "ar25",
    "bp35",
    "cd82",
    "cn04",
    "g50t",
    "ka59",
    "lf52",
    "lp85",
    "ls20",
    "m0r0",
    "re86",
    "s5i5",
    "sb26",
    "sc25",
    "sk48",
    "su15",
    "tn36",
    "tr87",
    "tu93",
    "wa30",
]

# (tag, prompt_flag, dedup_flag, seed_base). `B` and `C` share SEED_BASE with `A`, so within a
# (game, replicate) every arm sends the IDENTICAL seed. `AA` is the SAME arm as `A` at a
# DIFFERENT seed base: the noise floor, run through the identical analysis. This is mandatory,
# not decorative -- the generator sends no seed to the sampler by default and A/A has failed
# repeatedly this week.
#
# CORRECTED 2026-08-02 BY THIS RUN'S OWN MEASUREMENT. This comment used to end "any divergence
# is the knob firing, not the sampler". THAT IS FALSE and the run falsified it: on bp35 the
# control wrote 3722 bytes and BOTH treatments wrote 4150, at the same seed, with zero content
# failures and the goal-only call invoked in NONE of the three arms -- so no knob could have
# acted. B and C match each other on both games, so the divergence tracks POSITION (llama.cpp
# KV-cache state) rather than the knob. A fixed CARNOT_ARC_GENERATOR_SEED narrows the sampler
# but does not pin the completion on this server. The pairing is APPROXIMATE, and arm order --
# fixed here, not counterbalanced -- is therefore a confound on shape and not only on timing.
# A future revision of this harness should randomise arm order within a game.
S2_ARMS = [
    ("A", False, False, SEED_BASE),
    ("B", True, False, SEED_BASE),
    ("C", True, True, SEED_BASE),
    ("AA", False, False, AA_SEED_BASE),
]
# Stage 1 measures the goal-only CALL, so the dedup knob (which is about how two halves are
# COMBINED) has no meaning there and no arm C exists. Saying so explicitly beats leaving a
# reader to wonder whether it was dropped by accident.
S1_ARMS = [("gA", False, SEED_BASE), ("gB", True, SEED_BASE), ("gAA", False, AA_SEED_BASE)]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def server_props(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def listening_pid(port: int) -> int | None:
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if f"127.0.0.1:{port}" in line and "pid=" in line:
            try:
                return int(line.split("pid=")[1].split(",")[0])
            except (IndexError, ValueError):
                return None
    return None


def set_arm(prompt_on: bool, dedup_on: bool) -> None:
    for flag, on in ((FLAG_PROMPT, prompt_on), (FLAG_DEDUP, dedup_on)):
        if on:
            os.environ[flag] = "1"
        else:
            os.environ.pop(flag, None)


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    t0 = time.time()
    from carnot.agentic import arc_executable_world_model as e3

    assert Path(e3.E3_DIR) == SCRATCH, f"E3_DIR isolation failed: {e3.E3_DIR}"

    # ---------------- PRECONDITIONS ---------------------------------------------------------
    gguf = e3._resolve_gguf(e3.ARC_LIVE_GENERATOR_REPO_SUBSTR)  # noqa: SLF001
    conductor = subprocess.run(
        ["systemctl", "--user", "is-active", "carnot-conductor.service"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    free_mb = e3._cuda_gpu_free_mb(int(GPU))  # noqa: SLF001
    dry = os.environ.get("GEV_DRY") == "1"
    pre = [
        {
            "resource": "arc_live_generator_gguf_cached",
            "available": bool(gguf),
            "detail": str(gguf),
            "principle": "the live generator must be on disk; without it the honest verdict is "
            "blocked_model_not_cached, never a fabricated run",
        },
        {
            "resource": "conductor_inactive",
            "available": conductor != "active",
            "detail": f"systemctl is-active -> {conductor!r}",
            "principle": "a live conductor contends for the same card and would interleave its "
            "own induction into this store",
        },
        {
            "resource": f"cuda_gpu_{GPU}_has_headroom",
            "available": free_mb >= 20000 or dry,
            "detail": f"free={free_mb} MiB" + (" (dry run makes no LLM call)" if dry else ""),
            "principle": "a concurrent workflow holds the other card; launching without headroom "
            "would evict work this session does not own",
        },
        {
            "resource": "goal_prompt_transitions_flag_default_off",
            "available": not e3._goal_prompt_transitions_on(),  # noqa: SLF001
            "principle": "the shipped default IS the control arm; if it were already on this "
            "measures ON vs ON, which is the silent-no-op defect exp6013 shipped with",
        },
        {
            "resource": "goal_dedup_flag_default_off",
            "available": not e3._goal_dedup_on(),  # noqa: SLF001
            "principle": "same, for the dedup half",
        },
        {
            "resource": "port_free",
            "available": listening_pid(PORT) is None,
            "detail": f"port {PORT} (non-default on purpose)",
            "principle": "reusing a stale server on the default port is how an arm silently gets "
            "a different model or a different context size",
        },
        {
            "resource": "progress_windows_cached",
            "available": WINDOWS_PKL.exists(),
            "detail": str(WINDOWS_PKL),
            "principle": "windows are built from recorded frames before any LLM call, so every "
            "arm is scored against the identical observations",
        },
    ]
    if not all(p["available"] for p in pre):
        missing = [p["resource"] for p in pre if not p["available"]]
        (OUT / "blocked.json").write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_precondition_" + "_".join(missing)[:110],
                    "preconditions_checked": pre,
                },
                indent=2,
            )
        )
        print("BLOCKED:", missing)
        return 1
    print("preconditions OK")

    windows = pickle.loads(WINDOWS_PKL.read_bytes())
    missing_games = [g for g in ROSTER if g not in windows]
    if missing_games:
        raise SystemExit(f"roster games without a window: {missing_games}")
    split_meta = {}
    for g in ROSTER:
        shown, held, cell = windows[g]
        n_lvl_shown = sum(1 for t in shown if t.level_after > t.level_before)
        split_meta[g] = {
            "n_shown": len(shown),
            "n_heldout": len(held),
            "cell": int(cell),
            # ZERO in every game by construction of the prefix split. Recorded because the whole
            # "works on an unsolved game" claim rests on it: no arm is shown a win.
            "levelup_rows_in_shown": n_lvl_shown,
            "levelup_rows_in_heldout": sum(1 for t in held if t.level_after > t.level_before),
        }
    print(f"windows loaded for {len(ROSTER)} games")

    # ---------------- TREATMENT WITNESS (no LLM yet) -----------------------------------------
    # An intervention that is a silent no-op measured as a null is the exp6013 HUD-mask defect.
    # Four checks per game, all decidable offline:
    #   1. the goal-only prompt DIFFERS between prompt-off and prompt-on, and GROWS
    #   2. the COMBINED induce prompt is byte-identical between arms (the knobs must not leak
    #      into the engine prompt, or the contrast would be part goal and part engine)
    #   3. dedup is inert when off and excises when on, on real corpus-shaped code
    #   4. at analysis time, goal_only_call_ran is recorded per cell
    probe = e3.LocalGGUFProposer(port=PORT)
    dual = (
        "import numpy as np\n\ndef engine(grid, action, data):\n    return grid\n\n"
        "def is_level_complete(grid):\n    return bool(np.all(grid == 3))\n"
    )
    goal_block = "def is_level_complete(grid):\n    return False\n"
    treatment = []
    for game in ROSTER:
        shown, _held, cell = windows[game]
        set_arm(False, False)
        goal_off = probe._goal_only_prompt(game, None, shown)  # noqa: SLF001
        comb_off = e3.induce_prompt(game, shown, cell)
        comb_dedup_off = probe._combine_world_model(dual, goal_block)  # noqa: SLF001
        set_arm(True, True)
        goal_on = probe._goal_only_prompt(game, None, shown)  # noqa: SLF001
        comb_on = e3.induce_prompt(game, shown, cell)
        comb_dedup_on = probe._combine_world_model(dual, goal_block)  # noqa: SLF001
        set_arm(False, False)
        treatment.append(
            {
                "game": game,
                "goal_prompt_chars_off": len(goal_off),
                "goal_prompt_chars_on": len(goal_on),
                "goal_prompt_differs": goal_off != goal_on,
                "goal_prompt_grows": len(goal_on) > len(goal_off),
                "goal_prompt_sha256_off": sha(goal_off),
                "goal_prompt_sha256_on": sha(goal_on),
                "combined_prompt_identical": comb_off == comb_on,
                "combined_prompt_sha256": sha(comb_off),
                "dedup_inert_when_off": comb_dedup_off.count("def is_level_complete") == 2,
                "dedup_excises_when_on": comb_dedup_on.count("def is_level_complete") == 1,
            }
        )
    bad = [
        t
        for t in treatment
        if not (
            t["goal_prompt_differs"]
            and t["goal_prompt_grows"]
            and t["combined_prompt_identical"]
            and t["dedup_inert_when_off"]
            and t["dedup_excises_when_on"]
        )
    ]
    if bad:
        (OUT / "blocked.json").write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_treatment_witness_failed",
                    "failing_games": [b["game"] for b in bad],
                    "treatment": treatment,
                },
                indent=2,
            )
        )
        print("TREATMENT WITNESS FAILED:", [b["game"] for b in bad])
        return 1
    print(f"treatment witness OK on {len(treatment)} games")

    # ---------------- PRE-REGISTRATION (before the first LLM call) ---------------------------
    frozen = json.loads((HERE / "pre" / "frozen_corpus_clusters.json").read_text())
    prereg = {
        "experiment": "arc_goal_evidence_ab",
        "written_before_any_llm_call": True,
        "question": "Does giving the goal prompt the agent's OWN observed transitions (and "
        "stopping an evidence-grounded predicate being shadowed by an evidence-free one) stop "
        "the model DECLINING to write a win condition?",
        "arms_stage2_live_induce": {
            "A": "both knobs unset -- the SHIPPED path, the control",
            "B": f"{FLAG_PROMPT}=1 (evidence in the goal prompt; shadowing still possible)",
            "C": f"{FLAG_PROMPT}=1 and {FLAG_DEDUP}=1 (evidence + dedup)",
            "AA": "identical to A at a different seed base -- the NOISE FLOOR",
            "not_nested": "with dedup on, the goal-only call is SKIPPED whenever the engine half "
            "already carries a usable predicate, so C is not B-plus-something on every cell. "
            "Whether the goal-only call RAN is recorded per cell for exactly this reason.",
        },
        "arms_stage1_goal_only_component": {
            "gA": "shipped `_goal_only_prompt` with no transitions -- evidence-free",
            "gB": f"the same builder with {FLAG_PROMPT}=1 -- the agent's own transitions attached",
            "gAA": "gA at a different seed base -- the noise floor",
            "why_no_gC": "dedup governs how two halves are COMBINED; a single goal-only call has "
            "nothing to combine, so an arm C here would be arm gB relabelled",
        },
        "PRIMARY": {
            "metric": "DECLINED rate",
            "definition": "the LAST top-level `is_level_complete` (the one Python binds) is a "
            "constant False, has no return at all, or raises NameError on a free variable -- "
            "decided from the syntax tree by classify.py, which reuses the 2026-08-01 anatomy "
            "clusters rather than re-deriving them",
            "why_not_plan_found": "plan_found is an EXACT function of the goal gate's kind (0 "
            "mismatches in 138 engines), so scoring a GOAL intervention with it grades the gate "
            "using the gate",
            "why_not_the_goal_gate_at_all": "same circularity one indirection out; shape needs no "
            "environment, no bounded search and no win, and is independent of the gate",
            "test": "stratified permutation, statistic = mean over GAMES of (rate_treat - "
            "rate_control), arm label permuted WITHIN each game, 200000 draws, two-sided",
            "clustering": "GAME. Replicates within a game are not independent trials; treating "
            "them as such inflated a sibling experiment's p from 0.125 to 0.049 on 2026-07-31.",
            "alpha": 0.05,
        },
        "SECONDARIES": {
            "TROPE rate": "whole-board uniformity or colour elimination -- a generic ARC trope "
            "referencing nothing observed. PRE-FLIGHT SAYS THIS IS WHERE THE MECHANISM CAN "
            "ACTUALLY BITE: 52.2% of split-induce cells are tropes against 12.9% of combined.",
            "GROUNDED rate": "names a region, cell or object whose literal appears in the "
            "agent's own observed deltas",
            "goal_only_call_ran": "did the mechanism fire in this cell at all",
            "gate_kind": "NOT MEASURED. Named here so its absence is a decision, not an "
            "omission: it is the thing being changed.",
        },
        "PREFLIGHT_ON_THE_FROZEN_SHIPPED_CORPUS": {
            "source": "results/arc_object_perception_ab_change_fidelity_20260801/engines, 116 "
            "world_model.py from the same 20 games, classified with this run's classifier",
            "n": len(frozen),
            "declined_rate": 0.431,
            "trope_rate": 0.207,
            "split_induce_rate": 0.198,
            "declined_rate_on_split_cells": 0.174,
            "declined_rate_on_combined_cells": 0.495,
        },
        "MINIMUM_REACHABLE_P_AND_A_HONEST_POWER_STATEMENT": {
            "stage2_primary_cannot_reach_0.05": True,
            "arithmetic": "both knobs live ONLY in the split-induce fallback, and the goal-only "
            "prompt is never built when the combined call succeeds. Split-induce is 19.8% of "
            "cells and its declined rate is already only 17.4%, so driving split-cell declines "
            "to ZERO moves the ITT declined rate 0.431 -> 0.397: a 3.4 point ceiling. At 20 "
            "games x 3 replicates the permutation test cannot resolve 3.4 points, so the "
            "STAGE 2 PRIMARY IS DECLARED UNDERPOWERED BEFORE IT IS RUN. A stage-2 null is "
            "evidence that the knobs are STRUCTURALLY MISDIRECTED, not that showing evidence "
            "does not work.",
            "stage1_is_the_powered_test": "the goal-only call is made directly, so the mechanism "
            "fires in 100% of cells and nothing is diluted. At 20 games x 5 replicates per arm, "
            "with the permutation reference set C(2R,R)^20 the attainable minimum p is ~0 and "
            "quoting it would be meaningless reassurance -- the binding constraint is effect "
            "size, not the reference set. The pre-flight base rate for a goal-only-prompt "
            "predicate is taken from the 23 frozen split-induce cells (DECLINED 0.174, TROPE "
            "0.522), so stage 1 is powered for the TROPE contrast and only weakly for DECLINED. "
            "Reported that way rather than the other way round.",
            "min_reachable_p_reported": 0.0,
            "what_that_number_is_not": "it is NOT a claim of power. It is the floor of the "
            "permutation reference set. The honest statement is the effect-size one above.",
        },
        "MISSING_IS_NEVER_ZERO": "a cell whose induce raised, whose server failed, whose "
        "completion was truncated, or which produced no parseable `is_level_complete` is "
        "EXCLUDED and COUNTED PER ARM. It is never scored as a decline and never scored 0.",
        "generator": {
            "repo_substr": e3.ARC_LIVE_GENERATOR_REPO_SUBSTR,
            "gguf": str(gguf),
            "one_server_all_arms": True,
            "port": PORT,
            "cuda_gpu": GPU,
            "seed_scheme": "CARNOT_ARC_GENERATOR_SEED = seed_base + replicate; the SAME seed in "
            "every treatment arm and its control within a (game, replicate)",
        },
        "line_not_crossed": {
            "what_reaches_the_model": "the agent's own observed transitions only, rendered by "
            "the SAME _transitions_block the engine prompt already uses",
            "what_does_not": "game source, hand-written adapters, curated win examples, and "
            "_previous_level_complete_grid asserted as a win state (it is the NEXT level's "
            "opening board -- the win-state poison corrected 2026-07-29; passed as None here in "
            "every arm)",
            # RETRACTED 2026-08-02, and the bytes are NOT changed. The next two entries are
            # false as written (there are no stall games in the roster -- all 20 are
            # `full_game_clear: true`, and every window required `solve_adaptered(game, 1)`),
            # but out/preregistration.json is a frozen pre-registration that `prereg_sha256`
            # commits to, so the emit is left byte-identical and the correction is recorded in
            # the artifact's POST_REVIEW_CORRIGENDUM_2026_08_02 instead. The verified claim is
            # `levelup_rows_in_shown == 0`: no ARM was shown a win. If this harness is ever
            # re-run, FIX THESE TWO STRINGS FIRST -- the freeze protects the record of what was
            # pre-registered, not the sentence itself.
            "no_win_is_shown": "the prefix split puts the level-up row in the HELD-OUT tail, so "
            "levelup_rows_in_shown is 0 for every game; stall games with zero observed wins are "
            "in the roster on purpose",
            "works_on_an_unsolved_game": True,
        },
        "solve_provenance": "development_proxy",
        "solve_provenance_note": "No game is solved and no level is banked. This measures the "
        "SHAPE of an induced goal predicate offline against frozen windows of the agent's own "
        "recorded frames. The intervention itself uses only the agent's own observations, so it "
        "works identically on a game nobody has solved -- but the MEASUREMENT is offline on "
        "public games, so the artifact declares development_proxy, not "
        "live_agent_self_discovery.",
        "defaults_unchanged": "both knobs still ship OFF; flipping either is an operator "
        "decision this run does not take",
        "not_submitted": "no scored or online ARC game is played; submission is operator-only",
    }
    prereg_text = json.dumps(prereg, indent=2, sort_keys=True)
    (OUT / "preregistration.json").write_text(prereg_text)
    prereg_sha = "sha256:" + sha(prereg_text)
    print(f"pre-registration written {prereg_sha}")

    if dry:
        (OUT / "meta_dry.json").write_text(
            json.dumps(
                {
                    "split_meta": split_meta,
                    "treatment_witness": treatment,
                    "prereg_sha256": prereg_sha,
                    "preconditions_checked": pre,
                },
                indent=2,
            )
        )
        print("DRY RUN: stopping before the first LLM call")
        return 0

    # ---------------- SERVER (one, all arms, non-default port) -------------------------------
    os.environ["CARNOT_ARC_INDUCE_N_CTX"] = os.environ.get("GEV_N_CTX", "32768")
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = GPU
    prop = e3.LocalGGUFProposer(port=PORT)
    print(f"launching {prop.repo_substr} on port {PORT} gpu {GPU} n_ctx {prop.n_ctx} ...")
    if not prop._ensure_server():  # noqa: SLF001
        (OUT / "blocked.json").write_text(
            json.dumps({"honest_verdict": "blocked_generator_server_failed_to_start"}, indent=2)
        )
        return 1
    actual_port = prop.port
    pid = listening_pid(actual_port)
    exe = os.readlink(f"/proc/{pid}/exe") if pid else None
    props = server_props(actual_port)
    witness = {
        "pid": pid,
        "exe_from_proc": exe,
        "port_requested": PORT,
        "port_actual": actual_port,
        "cuda_gpu": GPU,
        "n_ctx_declared": prop.n_ctx,
        "n_ctx_from_props": props.get("default_generation_settings", {}).get("n_ctx")
        or props.get("n_ctx"),
        "model_from_props": props.get("model_path") or props.get("model"),
        # The HIP build lives in build-hip/ and is the AMD iGPU path; CUDA is build/. Setting
        # CUDA_VISIBLE_DEVICES alongside CARNOT_ARC_GENERATOR_CUDA_GPU renumbers the cards, the
        # headroom probe finds nothing, and the generator falls back to the iGPU SILENTLY -- ~6x
        # slower and a different substrate. Refuse rather than measure that.
        "is_cuda_build": bool(exe and "build-hip" not in exe and exe.endswith("llama-server")),
        "mtp": prop.mtp,
        "kv_quant": prop.kv_quant,
        "n_gpu_layers": prop.n_gpu_layers,
        "max_tokens": prop.max_tokens,
    }
    if not witness["is_cuda_build"]:
        (OUT / "blocked.json").write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_generator_not_on_the_cuda_build",
                    "server_witness": witness,
                },
                indent=2,
            )
        )
        print("BLOCKED: not the CUDA build ->", exe)
        return 1
    (OUT / "server_witness.json").write_text(json.dumps(witness, indent=2))
    print("server:", json.dumps(witness))

    # ---------------- STAGE 1: the goal-only CALL, mechanism fires 100% -----------------------
    def run_s1(game: str, rep: int, tag: str, prompt_on: bool, seed_base: int) -> dict:
        shown, _held, _cell = windows[game]
        set_arm(prompt_on, False)
        os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(seed_base + rep)
        prompt = prop._goal_only_prompt(game, None, shown)  # noqa: SLF001
        sf0, cf0 = prop.n_server_failures, prop.n_content_failures
        t = time.time()
        try:
            # FIDELITY TO THE SHIPPED SPLIT-INDUCE GOAL CALL, stated rather than assumed. The
            # shipped call is `generate(self._goal_only_prompt(...), ("is_level_complete",),
            # tries=self.tries, codeonly_eligible=True, engine_transitions=trans)`. The one
            # argument omitted here is `engine_transitions`, and it is INERT under the shipped
            # defaults: inside `generate` it is read at exactly one place, guarded by the
            # goal-defect / engine-defect check flags (`CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK`
            # and its engine sibling), both of which DEFAULT OFF and are not set by any arm of
            # this run. It is also omitted identically in every arm, so it cannot confound the
            # contrast even if that reading were wrong. `tries` is pinned to the shipped
            # default of 3 rather than read off the instance so a future default change cannot
            # silently alter what this measurement means.
            ok, code = prop.generate(
                prompt, ("is_level_complete",), tries=3, codeonly_eligible=True
            )
            exc = None
        except Exception as e:  # noqa: BLE001
            ok, code, exc = False, "", f"{type(e).__name__}: {e}"[:300]
        elapsed = time.time() - t
        row = {
            "stage": 1,
            "game": game,
            "replicate": rep,
            "arm": tag,
            "prompt_transitions": prompt_on,
            "seed": seed_base + rep,
            "elapsed_s": round(elapsed, 2),
            "ok": bool(ok),
            "exception": exc,
            "prompt_chars": len(prompt),
            "prompt_sha256": sha(prompt),
            "server_failures_delta": prop.n_server_failures - sf0,
            "content_failures_delta": prop.n_content_failures - cf0,
            # STAGE 1 FIRES BY CONSTRUCTION. Written out anyway so the two stages share a
            # column and the analysis never has to special-case one of them.
            "goal_only_call_ran": True,
        }
        if ok:
            (OUT / "s1_cells" / f"{game}__r{rep}__{tag}.py.txt").write_text(code)
            row["code_sha256"] = sha(code)
            row["code_bytes"] = len(code)
            rec = gcl.classify_predicate(code, shown)
            row.update({f"pred_{k}": v for k, v in rec.items()})
        else:
            row["fail_msg"] = str(code)[:300]
            row["pred_shape"] = None
        return row

    # ---------------- STAGE 2: the live induce path, the ITT ---------------------------------
    def run_s2(game: str, rep: int, tag: str, prompt_on: bool, dedup_on: bool, sb: int) -> dict:
        shown, held, cell = windows[game]
        cell_dir = SCRATCH / f"s2__{game}__r{rep}__{tag}"
        if cell_dir.exists():
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True)
        e3.E3_DIR = cell_dir  # module global, read at call time by write + load
        set_arm(prompt_on, dedup_on)
        os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(sb + rep)

        # THE MOST VALUABLE COLUMN IN THE RUN. An arm whose mechanism never fired is an
        # UNTESTED arm, not a refuted one; last night's sibling produced a false catastrophic
        # verdict for want of exactly this. Wrapping the bound method counts real calls rather
        # than inferring them from a file signature, which dedup would break anyway (it can
        # write a single-import file through a path that DID call the goal generator).
        calls = {"n": 0}
        real_goal_prompt = prop._goal_only_prompt  # noqa: SLF001

        def counting_goal_prompt(*a, **kw):
            calls["n"] += 1
            return real_goal_prompt(*a, **kw)

        prop._goal_only_prompt = counting_goal_prompt  # type: ignore[method-assign]
        sf0, cf0 = prop.n_server_failures, prop.n_content_failures
        t = time.time()
        try:
            ok, msg = prop.induce(game, shown, cell)
            exc = None
        except Exception as e:  # noqa: BLE001
            ok, msg, exc = False, "", f"{type(e).__name__}: {e}"[:300]
        finally:
            prop._goal_only_prompt = real_goal_prompt  # type: ignore[method-assign]
        elapsed = time.time() - t

        row = {
            "stage": 2,
            "game": game,
            "replicate": rep,
            "arm": tag,
            "prompt_transitions": prompt_on,
            "dedup": dedup_on,
            "seed": sb + rep,
            "elapsed_s": round(elapsed, 2),
            "induce_ok": bool(ok),
            "induce_msg": str(msg)[:300],
            "exception": exc,
            "server_failures_delta": prop.n_server_failures - sf0,
            "content_failures_delta": prop.n_content_failures - cf0,
            "goal_only_call_ran": calls["n"] > 0,
            "n_goal_only_calls": calls["n"],
            "n_shown": len(shown),
            "n_heldout": len(held),
            "arm_flags_consistent": (os.environ.get(FLAG_PROMPT) == "1") == prompt_on
            and (os.environ.get(FLAG_DEDUP) == "1") == dedup_on,
        }
        wm = cell_dir / game / "world_model.py"
        row["engine_file_exists"] = wm.exists()
        if wm.exists():
            code = wm.read_text()
            (OUT / "s2_cells" / f"{game}__r{rep}__{tag}.py.txt").write_text(code)
            row["engine_sha256"] = sha(code)
            row["engine_bytes"] = len(code)
            rec = gcl.classify_predicate(code, shown)
            row.update({f"pred_{k}": v for k, v in rec.items()})
        else:
            row["pred_shape"] = None
        return row

    # ORDERED REPLICATE-MAJOR, THEN GAME-MAJOR, WITH BOTH STAGES INTERLEAVED AT THE GAME, and
    # that ordering is load-bearing rather than cosmetic. Truncation is the expected end state
    # of this run, so the only question that matters about the ordering is WHAT A TRUNCATION
    # COSTS:
    #   stage-major   -> loses one ENTIRE stage. The run then silently answers a different
    #                    question than the one it was designed for.
    #   replicate-major, stage-interleaved -> loses trailing replicates from both stages, but
    #                    stage 2 does not begin until stage 1's whole replicate is done, so an
    #                    early stop still yields ZERO stage-2 cells. This is what the first
    #                    ordering did and it was wrong for the same reason, one level down.
    #   THIS: game-major with both stages adjacent -> a stop after game k leaves k games with
    #                    EVERY arm of BOTH stages measured, and 20-k games with none. The
    #                    surviving sample is a balanced prefix of the roster rather than a
    #                    balanced prefix of one stage.
    # The roster order is fixed and alphabetical, so which games survive is decided by the
    # alphabet and not by anything observed -- a truncation cannot select for games where the
    # treatment happens to look good.
    jobs: list[tuple] = []
    for rep in range(max(S1_REPS, S2_REPS)):
        for g in ROSTER:
            if rep < S1_REPS:
                for tag, p_on, sb in S1_ARMS:
                    jobs.append(("s1", g, rep, tag, p_on, False, sb))
            if rep < S2_REPS:
                for tag, p_on, d_on, sb in S2_ARMS:
                    jobs.append(("s2", g, rep, tag, p_on, d_on, sb))

    rows: list[dict] = []
    cache_dir = OUT / "rowcache"
    cache_dir.mkdir(exist_ok=True)
    for i, (stage, g, rep, tag, p_on, d_on, sb) in enumerate(jobs):
        key = cache_dir / f"{stage}__{g}__r{rep}__{tag}.json"
        if key.exists():
            rows.append(json.loads(key.read_text()))
            continue
        if time.time() - t0 > WALL_BUDGET_S:
            print(f"wall budget reached after {i} of {len(jobs)} cells; stopping cleanly")
            break
        row = (
            run_s1(g, rep, tag, p_on, sb) if stage == "s1" else run_s2(g, rep, tag, p_on, d_on, sb)
        )
        key.write_text(json.dumps(row, indent=1, default=str))
        rows.append(row)
        print(
            f"[{i + 1}/{len(jobs)}] {stage} {g} r{rep} {tag}: "
            f"shape={row.get('pred_shape')} ran={row.get('goal_only_call_ran')} "
            f"{row['elapsed_s']}s",
            flush=True,
        )
        if (i + 1) % 20 == 0:
            (OUT / "rows.json").write_text(json.dumps(rows, indent=1, default=str))

    (OUT / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "prereg_sha256": prereg_sha,
                "preconditions_checked": pre,
                "server_witness": witness,
                "split_meta": split_meta,
                "treatment_witness": treatment,
                "n_cells": len(rows),
                "n_jobs": len(jobs),
                "s1_reps": S1_REPS,
                "s2_reps": S2_REPS,
                "duration_s": round(time.time() - t0, 1),
                "liveness_witness": prop.liveness_witness(),
            },
            indent=2,
            default=str,
        )
    )
    print(f"done: {len(rows)}/{len(jobs)} cells in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
