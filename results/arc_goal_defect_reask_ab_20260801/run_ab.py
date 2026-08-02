"""Goal-defect-rejection A/B on the LIVE induce path, scored against REAL OBSERVED FRAMES.

WHY A GOAL INTERVENTION AT ALL. Measured 2026-08-01 over 138 induced engines from 21 games:
of the 93 LIVE engines, 71 (76%) cannot yield a plan because of the GOAL, not the dynamics --
45 because the goal predicate is never true anywhere the bounded search reaches, 24 because it
is degenerate. Every intervention of the preceding week targeted the ENGINE. Nothing had
targeted the goal.

WHY THE OUTCOME IS NOT plan_found, which is the whole methodological point of this run.
plan_found is an EXACT function of the goal gate's kind -- 0 mismatches in 138 -- so scoring a
GOAL intervention against plan_found grades the goal gate with the goal gate. The primary here
is instead measured against frames the agent really observed, and specifically against frames
the induce prompt never contained.

WHY NOT THE PROJECT'S OWN EXISTING WIN-RECOGNITION METRIC EITHER.
`arc_actions_to_progress._levelup_positive_recall` (REQ-ARC-WMTE-5714) scores the predicate on
`next_grid` at a real level-up. `pre/boundary_anatomy.json` measured that frame on all 20
windows BEFORE any outcome was chosen: it is a WHOLESALE BOARD REPLACEMENT, a median 25.8x an
ordinary step's cell-change, i.e. the NEXT level's opening board rather than a picture of the
level just completed. It is the same frame the 2026-07-29 win-state-poison correction was
about. It is reported here as a secondary, with that caveat attached, rather than used as the
primary or quietly dropped.

DEFAULTS ARE NOT TOUCHED. Both knobs ship default OFF; this measures whether they should be
flipped. Flipping is a separate, operator-visible decision.

NOT SUBMITTED: no scored or online ARC game is played. Submission is operator-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Derived, never hardcoded (CLAUDE.md Test-Run Record Integrity rule 4): a baked-in absolute
# path makes a fresh clone write into the operator's checkout. This file lives at
# <repo>/results/<exp>/, so the repo root is two parents up.
ROOT = HERE.parents[1]

# ---- isolation: E3_DIR is read at IMPORT time, so it must be set before the import ----
SCRATCH = Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/goalab/e3_store"
)
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["CARNOT_ARC_E3_DIR"] = str(SCRATCH)

if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

PORT = int(os.environ.get("GDAB_PORT", "41823"))
GPU = os.environ.get("GDAB_GPU", "0")
N_REPLICATES = int(os.environ.get("GDAB_REPLICATES", "3"))
SEED_BASE = int(os.environ.get("GDAB_SEED_BASE", "7100"))
AA_SEED_BASE = int(os.environ.get("GDAB_AA_SEED_BASE", "7200"))
WALL_BUDGET_S = float(os.environ.get("GDAB_WALL_BUDGET_S", "36000"))

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
CELLS = OUT / "cells"
CELLS.mkdir(exist_ok=True)
PREREG_PATH = OUT / "preregistration.json"

FLAG_CHECK = "CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK"
FLAG_PROMPT = "CARNOT_ARC_GOAL_PROMPT_TRANSITIONS"

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

# (arm, tag, seed_base). `on` shares SEED_BASE with `off`, so within a (game, replicate) the
# two arms send the IDENTICAL seed and the IDENTICAL combined prompt -- attempt 0 is the same
# draw in both arms, and any divergence is the intervention firing rather than the sampler.
# `aa` is the SAME arm as `off` at a DIFFERENT seed: the noise floor, run through the identical
# analysis. sampling_seed's docstring records a MEASURED 40% run-to-run divergence when the
# seed is absent, "at least as large as any treatment effect yet measured on this path".
ARMS = [("off", "off", SEED_BASE), ("on", "on", SEED_BASE), ("off", "aa", AA_SEED_BASE)]


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


def main() -> int:  # noqa: C901, PLR0915
    t0 = time.time()
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_world_model_trust_energy as wmte

    assert Path(e3.E3_DIR) == SCRATCH, f"E3_DIR isolation failed: {e3.E3_DIR}"

    # ---------------- PRECONDITIONS -------------------------------------------------
    gguf = e3._resolve_gguf(e3.ARC_LIVE_GENERATOR_REPO_SUBSTR)
    conductor = subprocess.run(
        ["systemctl", "--user", "is-active", "carnot-conductor.service"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    free_mb = e3._cuda_gpu_free_mb(int(GPU))
    pre = [
        {
            "resource": "gemma-4-31B-it-qat gguf cached",
            "available": bool(gguf),
            "detail": str(gguf),
            "principle": "the live generator must be on disk; without it the honest verdict is "
            "blocked_model_not_cached, never a fabricated run",
        },
        {
            "resource": "conductor_inactive",
            "available": conductor != "active",
            "detail": f"systemctl is-active -> {conductor!r}",
            "principle": "a live conductor would contend for the same GPU and interleave its "
            "own induction into this store",
        },
        {
            # SCOPED TO REAL RUNS. A dry run issues no LLM call and binds no card, so demanding
            # VRAM from it would block the witness/pre-registration pass -- the part that is
            # SUPPOSED to run before a GPU is even available -- on a resource it does not use.
            # The check is NOT relaxed for the measured run: `GDAB_DRY` also stops execution
            # before the first LLM call, so there is no path on which this passes vacuously and
            # a model then loads.
            "resource": f"cuda_gpu_{GPU}_has_headroom",
            "available": free_mb >= 20000 or os.environ.get("GDAB_DRY") == "1",
            "detail": f"free={free_mb} MiB"
            + (
                " (not required: dry run makes no LLM call)"
                if os.environ.get("GDAB_DRY") == "1"
                else ""
            ),
            "principle": "a concurrent workflow holds the other card; launching without "
            "headroom would evict work this session does not own",
        },
        {
            "resource": "goal_defect_flag_default_off",
            "available": not e3._goal_defect_check_on(),
            "principle": "the shipped default is the control arm; if it were already on, this "
            "measures ON vs ON -- the silent no-op exp6013 shipped with its HUD mask",
        },
        {
            "resource": "goal_prompt_flag_default_off",
            "available": not e3._goal_prompt_transitions_on(),
            "principle": "same, for the prompt half of the intervention",
        },
        {
            "resource": "port_free",
            "available": listening_pid(PORT) is None,
            "detail": f"port {PORT}",
            "principle": "reusing a stale server on the default port is how an arm silently "
            "gets a different model or context size",
        },
    ]
    if not all(p["available"] for p in pre):
        missing = [p["resource"] for p in pre if not p["available"]]
        (OUT / "blocked.json").write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_precondition_" + "_".join(missing)[:120],
                    "preconditions_checked": pre,
                },
                indent=2,
            )
        )
        print("BLOCKED:", missing)
        return 1
    print("preconditions OK")

    # ---------------- WINDOWS + EXPLICIT SPLIT (no LLM yet) --------------------------
    windows: dict[str, tuple] = {}
    split_meta: dict[str, dict] = {}
    for game in ROSTER:
        w = atp.build_progress_window(game)
        if w is None:
            raise SystemExit(f"roster game {game} lost its window since the probe")
        win, _full, cell = w
        win = list(win)
        shown, held = wmte._split_prefix_heldout(win)  # noqa: SLF001
        n_lvl = sum(1 for t in held if t.level_after > t.level_before)
        windows[game] = (shown, held, int(cell))
        split_meta[game] = {
            "n_transitions": len(win),
            "n_shown": len(shown),
            "n_heldout": len(held),
            "heldout_levelup_rows": n_lvl,
            "cell": int(cell),
        }
        # The primary needs held-out frames to score on. A game with none would be
        # structurally unmeasurable in BOTH arms -- exp6018's instrument floor, per-game.
        assert len(held) >= 1, f"{game} has an empty held-out tail"
    print(f"windows built for {len(windows)} games")

    # ---------------- TREATMENT WITNESS (still no LLM) -------------------------------
    # THIS WITNESS IS SHAPED DIFFERENTLY FROM ITS SIBLINGS, and the difference is the honest
    # part. The object-perception A/B could diff two prompts, because its treatment WAS a
    # prompt block. Half of this treatment is a POST-GENERATION accept check, which leaves the
    # combined induce prompt byte-identical between arms. So a prompt diff alone would show
    # "no difference" and prove nothing. Four checks instead:
    #   1. the combined prompt IS byte-identical between arms (stated, not hidden)
    #   2. the goal-only prompt DIFFERS between arms (the prompt half is armed)
    #   3. the detector is inert with the flag off and bites with it on (the check half is armed)
    #   4. at analysis time, n_goal_defect_reasks > 0 in `on` and == 0 in `off` (it FIRED)
    # Without (4) this could ship as a silent no-op measured as a null, which is the exp6013
    # HUD-mask defect exactly.
    defective = "def is_level_complete(grid):\n    return False\n"
    probe_prop = e3.LocalGGUFProposer(port=PORT)
    treatment = []
    for game, (shown, _held, cell) in windows.items():
        os.environ.pop(FLAG_CHECK, None)
        os.environ.pop(FLAG_PROMPT, None)
        combined_off = e3.induce_prompt(game, shown, cell)
        goal_off = probe_prop._goal_only_prompt(game, None, shown)  # noqa: SLF001
        defects_off = probe_prop._goal_defects(defective, shown)  # noqa: SLF001
        os.environ[FLAG_CHECK] = "1"
        os.environ[FLAG_PROMPT] = "1"
        combined_on = e3.induce_prompt(game, shown, cell)
        goal_on = probe_prop._goal_only_prompt(game, None, shown)  # noqa: SLF001
        defects_on = probe_prop._goal_defects(defective, shown)  # noqa: SLF001
        os.environ.pop(FLAG_CHECK, None)
        os.environ.pop(FLAG_PROMPT, None)
        treatment.append(
            {
                "game": game,
                "combined_prompt_identical": combined_off == combined_on,
                "combined_prompt_sha256": sha(combined_off),
                "goal_prompt_chars_off": len(goal_off),
                "goal_prompt_chars_on": len(goal_on),
                "goal_prompt_differs": goal_off != goal_on,
                "goal_prompt_grows": len(goal_on) > len(goal_off),
                "detector_inert_when_off": defects_off == [],
                "detector_bites_when_on": "goal_constant" in defects_on,
            }
        )
    bad = [
        t
        for t in treatment
        if not (
            t["combined_prompt_identical"]
            and t["goal_prompt_differs"]
            and t["goal_prompt_grows"]
            and t["detector_inert_when_off"]
            and t["detector_bites_when_on"]
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

    # ---------------- PRE-REGISTRATION (before the first LLM call) --------------------
    prereg = {
        "experiment": "arc_goal_defect_reask_heldout_ab",
        "requirement": "REQ-ARC-WMTE-6030 / REQ-ARC-WMTE-6031",
        "written_before_any_llm_call": True,
        "question": "Does rejecting a MECHANICALLY DEFECTIVE induced `is_level_complete` and "
        "re-asking -- plus carrying the agent's own observed transitions into the "
        "goal-only prompt -- produce goal predicates that discriminate on frames the "
        "prompt never saw?",
        "arms": {
            "control_off": "both knobs unset (the shipped path)",
            "treatment_on": f"{FLAG_CHECK}=1 and {FLAG_PROMPT}=1",
            "aa_control": "identical to control_off, different seed base -- the noise floor",
        },
        "PRIMARY": {
            "metric": "O6_pre_win_and_not_open",
            "definition": "the induced is_level_complete returns True on the frame IMMEDIATELY "
            "BEFORE the real level-up -- the last within-level state, the one the winning "
            "action was taken from -- AND False on the level's opening board. The level-up is "
            "the LAST transition of the window in 20/20 games and always lands in the HELD-OUT "
            "tail, so the induce prompt never contained this frame.",
            "why_not_plan_found": "plan_found is an EXACT function of the goal gate's kind (0 "
            "mismatches in 138 engines), so scoring a goal intervention against it grades the "
            "goal gate with the goal gate",
            "why_not_the_gate_at_all": "this is evaluated on REAL OBSERVED FRAMES, never on "
            "states forward-simulated by an induced engine inside "
            "_goal_satisfiability_check's bounded search",
            "control_base_rate_measured": 0.0609,
            "test": "stratified permutation: statistic = mean over GAMES of (rate_on - "
            "rate_off); the arm label is permuted WITHIN each game; 200000 draws; two-sided",
            "clustering": "GAME. Replicates within a game are not independent trials -- "
            "treating them as such inflated a sibling experiment's p from 0.125 to "
            "0.049 on 2026-07-31 and had to be corrected.",
            "alpha": 0.05,
        },
        "AMENDMENT_2_primary_swapped_before_any_llm_call": {
            "when": "after pre/circularity_gap.json and the determinacy cross-tab, before the "
            "first LLM call, before any outcome in this run existed",
            "what_changed": "the PRIMARY was goal_discriminates_heldout (O4) and is now "
            "O6_pre_win_and_not_open. O4 is demoted to a secondary and still reported.",
            "why": "O4 FAILED THE TEST THIS EXPERIMENT SETS ITSELF. The brief was to choose an "
            "outcome that is not the thing being changed, and O4 turned out to be exactly "
            "that: measured over the 115 frozen engines, EVERY predicate the accept check "
            "would KEEP scores O4-positive -- 6 of 6, FN=0. The gate's accept decision "
            "therefore DETERMINES the outcome in the keep direction, which is the same "
            "circularity as scoring against plan_found, one indirection out. O3 has the "
            "identical defect (FN=0). The disjoint-frames argument was not enough: the frame "
            "sets agree 94.8% of the time (pre/circularity_gap.json), so 'disjoint' bought a "
            "5.2% gap and not independence.",
            "why_O6_instead": "the gate's accept decision does NOT determine it -- 2 of the 6 "
            "predicates the gate would keep still FAIL O6 -- and unlike O2 it carries no "
            "constant-True contamination (0 of 7 O6-positives are constant-True, against 1 of "
            "9 for O2). Constant-True matters specifically here because the gate REJECTS "
            "constant-True, so an outcome a `return True` predicate can satisfy would be "
            "pushed DOWN by the treatment for a reason that has nothing to do with goal "
            "quality. O6 is also the sharper claim: it asks whether the predicate recognises "
            "the real near-win state and does not merely fire everywhere.",
            "cost_of_the_swap": "power. The control base rate falls 0.104 -> 0.061, so at 3 "
            "replicates power is ~0.63 for 0.061 -> 0.20 and ~0.87 for 0.061 -> 0.30 "
            "(pre/power_O6.json). A cleaner construct measured with less power is the right "
            "trade against a better-powered construct that the treatment mechanically implies.",
            "honest_residual": "O6 is still not INDEPENDENT of the treatment, only "
            "undetermined by it. A predicate that discriminates on observed frames is more "
            "likely to fire at pre_win than one that does not. The claim is that the "
            "treatment does not mechanically force the outcome, NOT that the two are "
            "unrelated.",
        },
        "SECONDARIES_reported_not_primary": {
            "O4_discriminates_heldout": "the DEMOTED former primary (base 0.104). Reported in "
            "full, with the FN=0 determinacy defect attached, so the swap is auditable rather "
            "than a metric quietly disappearing.",
            "O2_fires_pre_win": "O6 without the not-at-open clause (base 0.078). The delta "
            "between O2 and O6 is exactly the constant-True slice.",
            "O1_fires_post_win": "the frame REQ-ARC-WMTE-5714's _levelup_positive_recall "
            "scores. pre/boundary_anatomy.json measured it as a wholesale board replacement "
            "(median 25.8x an ordinary step) -- the NEXT level's opening board. Reported with "
            "that caveat. Control base rate 0.026.",
            "O7b_all_false_observed": "the raw failure mode: the predicate is False on every "
            "frame the agent ever saw. 88.7% of control predicates. Descriptive, and NOT a "
            "candidate primary -- it is the treatment's own probe restated.",
            "goal_gate_kind_and_plan_found": "SECONDARY ONLY, and explicitly the thing that was "
            "changed. Never the primary.",
        },
        "GUARDRAIL": {
            "metric": "held-out change_fidelity / accuracy of the ENGINE",
            "why": "on the combined call one answer carries both functions, so a goal-triggered "
            "re-ask regenerates the ENGINE too. If the goal improves while the engine "
            "degrades, the intervention is not free and must not be reported as if it were.",
        },
        "ARMEDNESS_GATE": {
            "rule": "if n_goal_defect_reasks is 0 across the whole `on` arm, the treatment never "
            "fired and the run is reported as a NON-TEST, not as a null",
            "why": "exp6013 shipped a HUD-mask factor that was a silent no-op on 162 arms and "
            "was read as 'both mask settings measured'",
        },
        "MISSING_IS_NEVER_ZERO": "a cell whose induce raised, whose server failed, whose "
        "response was truncated, or whose goal scorer timed out is excluded and "
        "COUNTED PER ARM. It is never scored 0.",
        "power": {
            "source": "pre/power_O6.json for the PRIMARY (control base 0.0609); pre/power.json "
            "retained for the demoted O4 (base 0.1043). Both simulated before any GPU time, "
            "with game-level heterogeneity drawn from a Beta so that 'most games flat, a few "
            "carry it' -- the shape the pre-flight actually shows -- rather than a homogeneous "
            "Bernoulli, which would be optimistic.",
            "PRIMARY_O6_unpaired_power_at_3_reps": {
                "p_trt_0.15": 0.403,
                "p_trt_0.20": 0.630,
                "p_trt_0.25": 0.760,
                "p_trt_0.30": 0.870,
                "p_trt_0.40": 0.980,
            },
            "min_reachable_p_statement": "the permutation reference set is C(2R,R)^20 within-"
            "game assignments, so the ATTAINABLE minimum p is ~0 and quoting it would be "
            "meaningless reassurance. The binding constraint is effect size. On the PRIMARY at "
            "3 replicates this design has ~87% power for a 5x effect (0.061 -> 0.30), ~63% for "
            "a 3x effect (0.061 -> 0.20), and ~40% for a 2.5x effect (0.061 -> 0.15). SO A "
            "NULL IS WEAK EVIDENCE AGAINST A SMALL OR MODERATE EFFECT and will be reported as "
            "such, not as 'no effect'. The seeded pairing (both arms share attempt 0, so a "
            "cell where the gate never fires contributes an exactly zero difference) should "
            "beat these unpaired numbers, but that gain is NOT assumed here and is not "
            "credited in advance.",
        },
        "generator": {
            "repo_substr": e3.ARC_LIVE_GENERATOR_REPO_SUBSTR,
            "gguf": str(gguf),
            "one_server_both_arms": True,
            "port": PORT,
            "cuda_gpu": GPU,
            "sampler_seeded": True,
            "seed_scheme": "CARNOT_ARC_GENERATOR_SEED = SEED_BASE + replicate, the SAME seed in "
            "control and treatment of a (game, replicate) pair, so attempt 0 is the "
            "identical draw and the arms diverge only where the gate fires.",
        },
        "AMENDMENT_before_any_llm_call": {
            "when": "after pre/detector_coverage.json, before the first LLM call, before any "
            "outcome in this run existed",
            "measured": "the accept check would reject 109 of 115 frozen engines -- a 94.8% "
            "firing rate. goal_constant accounts for 109, goal_missing_return 3, goal_raises 1. "
            "With the flag off it fires on 0 of 115, so the default-off guarantee holds on real "
            "corpus code and not merely on a fixture.",
            "WHAT THIS CHANGES ABOUT WHAT MAY BE CLAIMED": "at a 94.8% firing rate the "
            "treatment is NOT a selective filter -- it is very close to UNCONDITIONAL "
            "RESAMPLING. With a budget of 2 re-asks inside `tries=3`, on ~95% of cells the "
            "treatment simply takes the THIRD sample instead of the first, under a modified "
            "prompt. So a positive result here must be read as 'resampling the goal under a "
            "nudge helps', NEVER as 'the selectivity of the check helps'. The project already "
            "measured the engine-side analogue of this and found the defect TEXT bought nothing "
            "(p = 1.000 over 5 discordant pairs) -- the second ASK was the whole effect.",
            "why_a_compute_matched_arm_was_NOT_added": "a 100%-firing arm would differ from the "
            "94.8% treatment on ~5% of cells, so it could not separate selectivity from "
            "resampling with any power at this n. The honest move is to state the confound in "
            "the claim rather than to buy a fourth arm that cannot resolve it.",
            "selectivity_cross_tab": "of the 12 frozen engines whose goal DID discriminate on "
            "held-out frames, the gate would reject 6 and keep 6; of the 103 that did not, it "
            "rejects 103 and keeps 0. Recall on non-discriminating goals is perfect, precision "
            "is not: the gate discards half the good ones. The cost is bounded -- a re-ask can "
            "only replace a sample, never fail where the shipped path succeeded -- but it is a "
            "real cost and is reported rather than buried.",
        },
        "solve_provenance": "development_proxy",
        "solve_provenance_note": "No game is solved or level-banked here. This measures the "
        "quality of an induced goal predicate offline against a frozen window. The "
        "intervention itself uses ONLY the agent's own observations -- constancy of a "
        "predicate over frames already seen, and those same frames rendered into the prompt "
        "-- so it carries no fact about any game from outside and works identically on a game "
        "nobody has solved. But this measurement is offline, on public games, so the artifact "
        "declares development_proxy rather than live_agent_self_discovery.",
        "flags_remain_default_off": True,
        "not_submitted": "no scored or online ARC game is played; submission is operator-only",
    }
    prereg_text = json.dumps(prereg, indent=2, sort_keys=True)
    PREREG_PATH.write_text(prereg_text)
    prereg_sha = "sha256:" + sha(prereg_text)
    print(f"pre-registration {PREREG_PATH} {prereg_sha}")

    if os.environ.get("GDAB_DRY") == "1":
        (OUT / "meta_dry.json").write_text(
            json.dumps(
                {
                    "split_meta": split_meta,
                    "treatment_witness": treatment,
                    "prereg_sha256": prereg_sha,
                },
                indent=2,
            )
        )
        print("DRY RUN: stopping before the first LLM call")
        return 0

    # ---------------- SERVER (one, all arms) ----------------------------------------
    # n_ctx 32768, NOT the shipped 81920, declared because an undeclared config deviation is
    # how a measurement silently stops describing the thing it names. The shipped pool exists
    # for CONCURRENCY (4 kv_unified slots); this harness is strictly sequential, where one
    # request needs ~6k prompt + 4096 generated. At 81920 the guard offloads FFN layers to
    # system RAM to fit. Identical in every arm, so it cannot confound the contrast.
    os.environ["CARNOT_ARC_INDUCE_N_CTX"] = os.environ.get("GDAB_N_CTX", "32768")
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
    server_witness = {
        "pid": pid,
        "exe_from_proc": exe,
        "port_requested": PORT,
        "port_actual": actual_port,
        "cuda_gpu": GPU,
        "reuse_refusals": list(getattr(prop, "reuse_refusals", [])),
        "n_ctx_declared": prop.n_ctx,
        "n_ctx_from_props": props.get("default_generation_settings", {}).get("n_ctx")
        or props.get("n_ctx"),
        "model_from_props": props.get("model_path") or props.get("model"),
        # The HIP build lives in build-hip/ and is the AMD iGPU path; CUDA is build/. A run
        # that silently landed on the iGPU would be ~6x slower and a different substrate.
        "is_cuda_build": bool(exe and "build-hip" not in exe and exe.endswith("llama-server")),
        "mtp": prop.mtp,
        "kv_quant": prop.kv_quant,
        "n_gpu_layers": prop.n_gpu_layers,
        "max_tokens": prop.max_tokens,
    }
    if not server_witness["is_cuda_build"]:
        (OUT / "blocked.json").write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_generator_not_on_the_cuda_build",
                    "server_witness": server_witness,
                },
                indent=2,
            )
        )
        print("BLOCKED: not the CUDA build ->", exe)
        return 1
    print("server:", json.dumps(server_witness))
    (OUT / "server_witness.json").write_text(json.dumps(server_witness, indent=2))

    # ---------------- CELLS ----------------------------------------------------------
    def run_cell(game: str, rep: int, arm: str, tag: str, seed_base: int) -> dict:
        shown, held, cell = windows[game]
        cell_dir = SCRATCH / f"{game}__r{rep}__{tag}"
        if cell_dir.exists():
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True)
        e3.E3_DIR = cell_dir  # module global, read at call time by write + load

        if arm == "on":
            os.environ[FLAG_CHECK] = "1"
            os.environ[FLAG_PROMPT] = "1"
        else:
            os.environ.pop(FLAG_CHECK, None)
            os.environ.pop(FLAG_PROMPT, None)
        os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(seed_base + rep)

        prompt = e3.induce_prompt(game, shown, cell)
        # Read these off the instance with NO getattr default: a rename must fail loudly
        # rather than silently returning 0 forever and turning every missing observation into
        # a real zero. (That exact defect is recorded in the sibling harness.)
        sf0, cf0 = prop.n_server_failures, prop.n_content_failures
        gr0, er0 = prop.n_goal_defect_reasks, prop.n_induce_defect_reasks
        t = time.time()
        try:
            ok, msg = prop.induce(game, shown, cell)
            exc = None
        except Exception as e:  # noqa: BLE001
            ok, msg, exc = False, "", f"{type(e).__name__}: {e}"[:300]
        elapsed = time.time() - t

        row = {
            "game": game,
            "replicate": rep,
            "arm": arm,
            "tag": tag,
            "seed": seed_base + rep,
            "elapsed_s": round(elapsed, 2),
            "induce_ok": bool(ok),
            "induce_msg": str(msg)[:300],
            "exception": exc,
            "prompt_chars": len(prompt),
            "prompt_sha256": sha(prompt),
            "server_failures_delta": prop.n_server_failures - sf0,
            "content_failures_delta": prop.n_content_failures - cf0,
            # THE ARMEDNESS EVIDENCE. Must be >0 somewhere in `on` and exactly 0 everywhere in
            # `off`/`aa`, or the treatment never fired and this is a non-test.
            "goal_defect_reasks_delta": prop.n_goal_defect_reasks - gr0,
            "engine_defect_reasks_delta": prop.n_induce_defect_reasks - er0,
            "n_shown": len(shown),
            "n_heldout": len(held),
            "e3_dir": str(cell_dir),
        }
        row["arm_flags_consistent"] = (os.environ.get(FLAG_CHECK) == "1") == (arm == "on")
        wm = cell_dir / game / "world_model.py"
        row["engine_file_exists"] = wm.exists()
        if wm.exists():
            code = wm.read_text()
            row["engine_sha256"] = sha(code)
            row["engine_bytes"] = len(code)
            row["split_induce_signature"] = code.count("import numpy as np") > 1
        return row

    jobs = [
        (g, r, arm, tag, sb) for r in range(N_REPLICATES) for g in ROSTER for (arm, tag, sb) in ARMS
    ]
    rows = []
    for i, (g, r, arm, tag, sb) in enumerate(jobs):
        cache = CELLS / f"{g}__r{r}__{tag}.json"
        if cache.exists():
            rows.append(json.loads(cache.read_text()))
            continue
        if time.time() - t0 > WALL_BUDGET_S:
            print(f"wall budget reached after {i} of {len(jobs)} cells; stopping cleanly")
            break
        row = run_cell(g, r, arm, tag, sb)
        cache.write_text(json.dumps(row, indent=1))
        rows.append(row)
        print(
            f"[{i + 1}/{len(jobs)}] {g} r{r} {tag}: ok={row['induce_ok']} "
            f"{row['elapsed_s']}s goal_reasks={row['goal_defect_reasks_delta']}"
        )

    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "prereg_sha256": prereg_sha,
                "server_witness": server_witness,
                "split_meta": split_meta,
                "treatment_witness": treatment,
                "n_cells": len(rows),
                "n_jobs": len(jobs),
                "duration_s": round(time.time() - t0, 1),
                "liveness_witness": prop.liveness_witness(),
            },
            indent=2,
        )
    )
    print(f"done: {len(rows)}/{len(jobs)} cells in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
