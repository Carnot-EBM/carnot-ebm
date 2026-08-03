"""GENERATION ABLATION: four induce-prompt variants, scored on HELD-OUT engine accuracy.

NO BEHAVIOUR IS MEASURED HERE. No episode is run, no plan is executed, no trust gate is consulted,
no level moves. The outcome is a property of an induced engine evaluated against transitions the
model was never shown. An improvement here would be NECESSARY for the live agent and nowhere near
SUFFICIENT; see the artifact's `necessary_is_not_sufficient` field.

WHY THESE FOUR ARMS -- each is named by a specific finding in the two diagnostic artifacts, and
the arms the diagnosis REFUTED are deliberately absent.

  base   The shipped prompt, unchanged. Mandatory: without it nothing is comparable.

  think  `CARNOT_ARC_CODEONLY_INDUCE=0`, which removes `_L2_CODEONLY_DIRECTIVE` -- the block whose
         first lines are "/no_think", "Do NOT analyze the grids", "Do NOT ... reason about the win
         state", "Skip all reasoning". The prompt audit found this prefaces 24 of 24 payloads on a
         task whose entire content is inferring a transition rule from grid deltas, and that it was
         justified in 2026-06-25 against Qwen3.5-9B -- a generator retired 2026-07-28 -- and never
         re-justified against gemma-4-31B. The engine census supplies the mechanism: ft09's engine
         dispatches on `action == 6`, bounds-checks, enumerates eight distinct colours at the
         clicked cell, and writes `return grid` as the body of EVERY case. The model does the
         perception and the dispatch and fails at exactly one step -- naming the outcome of each
         case -- which is the step "Skip all reasoning" forbids. The audit named this the cheapest
         next test; it is the only prompt element with a shipped off-switch AND a retired rationale.

  antiid The shipped prompt plus an explicit statement that returning the input unchanged is a
         failure. The audit found identity invited by four independent routes and forbidden by
         none: 0 of 24 prompts contain the word "identity", any instruction that engine() must
         change something, or any statement that a do-nothing engine is unacceptable -- while
         24 of 24 say "Prefer SIMPLE GENERAL rules over per-frame special cases", and the simplest
         general rule consistent with sparse evidence over a mostly-unobserved action space is
         that nothing happens. The clause deliberately EXEMPTS unobserved actions, because for
         those identity is the correct answer and forbidding it would be forcing a lie.

  delta  The shipped prompt plus an output-shape instruction in the delta language the header
         already teaches. The audit found the prompt spends ~790 header tokens teaching TWO
         run-length codecs so the model can READ the evidence as deltas (24/24), then offers 0 of
         24 any way to ANSWER in that language: the engine contract is "Return the predicted next
         grid (same shape)". The model reads sparse edits and must reply with a total grid
         transform.

ARMS DELIBERATELY NOT RUN, and why -- spending the budget where the evidence points:

  * TRANSITION SELECTION (more / fewer / differently chosen). REFUTED as a capacity story by the
    audit: the worst-case payload is 54.1% of the 16,384-token slot budget, 0 of 24 games exceed
    it, the 40,000-char transition budget never binds on any game, and the shipped default has
    been "show every transition" since 2026-08-01, so `n_shown == n_changed` on all 24 games.
    There is nothing to un-truncate and nothing informative being dropped.
  * PER-ACTION DECOMPOSITION. The census motivates it (ft09 dispatches on one action and defaults
    elsewhere), but the audit shows the binding scarcity is EVIDENCE, not prompt structure: a
    median of 2 of the 7 declared actions is ever observed, 11 of 24 games show exactly one, and
    0 of 24 show all seven. Splitting one call into seven cannot manufacture evidence for the five
    actions the window never contains. Its cost (7x the calls) is better spent on replicates.

WHAT `change_accuracy` ACTUALLY IS -- a correction to the brief, stated because it changes how
every number below should be read. The brief describes it as "of the cells that TRULY CHANGED in a
held-out transition, what fraction does the engine get right". That is `cell_recall`. The shipped
`change_accuracy` is `n_changes_correct / n_changing` where `n_changes_correct` increments only
inside `np.array_equal(pred_g, g1)` -- a WHOLE-GRID EXACT match, restricted to the changing
transitions. So it is an exact-match rate over changing rows, not a cell fraction, which is why
the census's best clean non-tn36 engine (ls20: 50 of 52 changed cells right on every held-out row,
wrong on two counter cells) scores change_accuracy 0.0000 with cell_recall 0.9615. Both are
reported, and so is change_fidelity, the union-scored cell measure that charges a spurious write
what it charges a miss.

NOT SUBMITTED. No scored or online ARC game is played; submission is operator-only. No shipped
default is flipped: every arm sets its flags in this process's own environment and restores them.
`results/arc_e3` is never written -- E3_DIR is redirected before the import that reads it.
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
ROOT = Path(__file__).resolve().parents[2]

SCRATCH = HERE / "e3_store"
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["CARNOT_ARC_E3_DIR"] = str(SCRATCH)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

SHARD = int(os.environ.get("ABL_SHARD", "0"))
N_SHARDS = int(os.environ.get("ABL_N_SHARDS", "2"))
PORT = int(os.environ.get("ABL_PORT", str(42900 + SHARD)))
GPU = os.environ.get("ABL_GPU", str(SHARD))
N_REPLICATES = int(os.environ.get("ABL_REPLICATES", "3"))
SEED_BASE = int(os.environ.get("ABL_SEED_BASE", "20802"))
WALL_BUDGET_S = float(os.environ.get("ABL_WALL_BUDGET_S", "30000"))
MAX_TOKENS = int(os.environ.get("ABL_MAX_TOKENS", "8192"))
TRIES = int(os.environ.get("ABL_TRIES", "3"))

OUT = HERE / "out"
CELLS = OUT / "cells"
ENGINES = OUT / "engines"
for d in (OUT, CELLS, ENGINES):
    d.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------------------------
# THE ARMS. `suffix` is appended to the base induce_prompt; `codeonly` chooses whether
# `_L2_CODEONLY_DIRECTIVE` is prepended by `generate`. Everything else is held EQUAL across arms
# -- max_tokens, tries, temperature schedule, seed, server, window, split.
#
# DEVIATION FROM THE PROJECT'S OWN `reason` ARM, stated because it is deliberate: that config
# (arc_actions_to_progress.ARM_CONFIGS) pairs codeonly=0 with tries=1 and max_tokens=8192 while
# its controls use tries=3 / 4096. Carrying that over would confound the directive with the retry
# and token budgets, so here EVERY arm gets tries=3 and max_tokens=8192 and the only thing that
# moves is the prompt. The shipped default is 4096; whether lifting it changed `base` at all is
# answerable from the recorded generated-token distribution (if base never approaches 4096, the
# lift is inert for base) rather than assumed.
# ---------------------------------------------------------------------------------------------

ANTIID_MARK = "A DO-NOTHING ENGINE IS A FAILURE"
ANTIID_SUFFIX = f"""

CRITICAL -- {ANTIID_MARK}, NOT A SIMPLE RULE.
Every transition listed above CHANGED the grid. An engine whose body is `return grid` -- or which
returns its input unchanged for an action shown above -- is WRONG and will be rejected. "Nothing
happens" is never the correct rule for an action you were shown changing the grid.
For every action that APPEARS above, engine() MUST return a grid that DIFFERS from its input, in
the way that action was observed to change it. If you are not certain of the exact rule, write
your best SPECIFIC guess: a specific rule that is somewhat wrong is worth more than a do-nothing
rule that cannot be wrong.
For an action that does NOT appear above, returning the grid unchanged is acceptable."""

DELTA_MARK = "ANSWER IN THE SAME DELTA LANGUAGE THE EVIDENCE USES"
DELTA_SUFFIX = f"""

HOW TO WRITE THE BODY -- {DELTA_MARK}.
The evidence above gives you, for each action, ONLY THE CELLS THAT CHANGED. Write engine() the
same way. Do NOT reconstruct or re-derive the whole grid. Start from a copy of the input and
assign ONLY the cells that change:

    def engine(grid, action, data=None):
        out = grid.copy()
        # assign ONLY the cells this action changes, e.g. out[r, c] = v
        return out

A transition typically changes a very small fraction of the grid; every other cell must come out
exactly as it went in. Your job is to name WHICH cells change and WHAT they become -- nothing
else."""

ARMS: dict[str, dict] = {
    "base": {"codeonly": "1", "suffix": "", "mark": None, "desc": "the shipped prompt, unchanged"},
    "think": {
        "codeonly": "0",
        "suffix": "",
        "mark": None,
        "desc": "CARNOT_ARC_CODEONLY_INDUCE=0 -- _L2_CODEONLY_DIRECTIVE removed",
    },
    "antiid": {
        "codeonly": "1",
        "suffix": ANTIID_SUFFIX,
        "mark": ANTIID_MARK,
        "desc": "shipped prompt + explicit anti-identity clause",
    },
    "delta": {
        "codeonly": "1",
        "suffix": DELTA_SUFFIX,
        "mark": DELTA_MARK,
        "desc": "shipped prompt + delta-shaped output instruction",
    },
}
ARM_ORDER = ["base", "think", "antiid", "delta"]

ROSTER = [
    "ls20",
    "s5i5",
    "tu93",
    "cn04",
    "m0r0",
    "sk48",
    "ar25",
    "tr87",
    "g50t",
    "re86",
    "bp35",
    "sb26",
    "lf52",
    "su15",
    "lp85",
    "cd82",
    "wa30",
    "sc25",
    "tn36",
    "ka59",
]
# Failure mode 5: HIDDEN_STATE_GAME_IDS is a hardcoded 11-game PUBLIC tuple and a hidden Kaggle
# game ALWAYS takes the PLAIN branch. The primary scope is therefore the PLAIN-branch games; the
# hidden-state games are still run (they cost the same and add power to the secondary) but are
# reported as their own stratum and never pooled into a claim about the hidden branch.
HIDDEN_STATE = {
    "ar25",
    "cd82",
    "cn04",
    "dc22",
    "g50t",
    "ka59",
    "m0r0",
    "re86",
    "sc25",
    "sk48",
    "wa30",
}


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def min_reachable_two_sided_p(n_disc: int) -> float:
    """Smallest two-sided sign-test p attainable at `n_disc` discordant pairs -- the
    all-one-direction outcome. Stated BEFORE results so a design that cannot reach 0.05 is
    reported as underpowered rather than as a null."""
    if n_disc <= 0:
        return 1.0
    return min(1.0, 2.0 * (0.5**n_disc))


def listening_pid(port: int) -> int | None:
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if f"127.0.0.1:{port}" in line and "pid=" in line:
            try:
                return int(line.split("pid=")[1].split(",")[0])
            except (IndexError, ValueError):
                return None
    return None


def server_props(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=20) as r:
            return json.loads(r.read().decode())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------------------------
# ON-THE-WIRE PAYLOAD RECORDER (failure mode 3: prove a mechanism is REACHED by instrumenting the
# CALLEE and reading what actually crossed the boundary -- never by reading the call site).
#
# `generate` builds `_payload` and POSTs `json.dumps(_payload)` to /completion. Wrapping
# `urllib.request.urlopen` and decoding `req.data` therefore captures the LITERAL prompt string the
# server received, after `_L2_CODEONLY_DIRECTIVE` was or was not prepended and after the arm's
# suffix was or was not appended. Nothing about the arm is inferred from the configuration.
# ---------------------------------------------------------------------------------------------

_WIRE: list[dict] = []
_orig_urlopen = urllib.request.urlopen


def _recording_urlopen(req, *a, **kw):
    try:
        data = getattr(req, "data", None)
        if data:
            payload = json.loads(data.decode())
            if isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
                _WIRE.append(payload)
    except Exception:  # noqa: BLE001 - a recorder must never break the call it observes
        pass
    return _orig_urlopen(req, *a, **kw)


urllib.request.urlopen = _recording_urlopen


def main() -> int:  # noqa: C901 - one linear measurement procedure
    t0 = time.time()
    from carnot.agentic import arc_executable_world_model as e3

    assert Path(e3.E3_DIR) == SCRATCH, f"E3_DIR isolation failed: {e3.E3_DIR}"
    # NOTE ON THE PATCH SITE. `arc_executable_world_model` does `import urllib.request` INSIDE the
    # functions that call it, then `urllib.request.urlopen(...)`. That binds the shared module
    # object and resolves `urlopen` as an attribute AT CALL TIME, so the module-level patch above
    # is seen by the real call path. This is asserted, not assumed: `n_wire_calls` on every cell
    # must be > 0, and a cell that records zero wire calls is treated as un-witnessed.

    with open(OUT / "windows.pkl", "rb") as fh:
        store = pickle.load(fh)
    prep_meta = json.loads((OUT / "prep_meta.json").read_text())

    games = [g for g in ROSTER if g in store]
    mine = [g for i, g in enumerate(games) if i % N_SHARDS == SHARD]
    print(f"shard {SHARD}/{N_SHARDS} gpu {GPU} port {PORT}: {len(mine)} games {mine}", flush=True)

    # ---------------- PRECONDITIONS -------------------------------------------------------
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
            "resource": "gemma-4-31B-it-qat_gguf_cached",
            "available": bool(gguf),
            "detail": str(gguf),
            "principle": "the operator-fixed inducer must be on disk; without it the honest "
            "verdict "
            "honest verdict is blocked_model_not_cached, never a CPU "
            "fallback reported as live",
        },
        {
            "resource": "llama_cpp_gpu_offload",
            "available": _gpu_offload(),
            "principle": "a CPU-only llama-cpp build still clears the 60s live-inference duration "
            "floor, so 'it took long enough' is not evidence of GPU compute",
        },
        {
            "resource": "conductor_inactive",
            "available": conductor != "active",
            "detail": f"systemctl is-active -> {conductor!r}",
            "principle": "a live conductor contends for the same card and interleaves its own "
            "induction into this store",
        },
        {
            "resource": f"cuda_gpu_{GPU}_has_headroom",
            "available": free_mb >= 20000,
            "detail": f"free={free_mb} MiB",
            "principle": "launching without headroom would evict work this session does not own",
        },
        {
            "resource": "port_free",
            "available": listening_pid(PORT) is None,
            "detail": f"port {PORT}",
            "principle": "a stale server on the default port is how an arm silently "
            "gets a different model, context size, or the slow AMD iGPU backend",
        },
        {"resource": "windows_prepped", "available": len(mine) > 0},
    ]
    if not all(p["available"] for p in pre):
        missing = [p["resource"] for p in pre if not p["available"]]
        (OUT / f"blocked_shard{SHARD}.json").write_text(
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
    print("preconditions OK", flush=True)

    # ---------------- TREATMENT WITNESS: the prompts really differ, per arm per game --------
    # Unlike the sibling inert-rejection A/B (whose flag changes no prompt text, so its witness is
    # byte-EQUALITY), every arm here is supposed to MOVE the prompt. The witness is therefore a
    # block diff plus a marker-presence check, computed on the real `induce_prompt` output.
    base_prompt_fn = e3.induce_prompt
    witness = []
    for game in mine:
        s = store[game]
        p_base = base_prompt_fn(game, s["shown"], s["cell"])
        for arm in ARM_ORDER:
            suf = ARMS[arm]["suffix"]
            p_arm = p_base + suf
            mark = ARMS[arm]["mark"]
            witness.append(
                {
                    "game": game,
                    "arm": arm,
                    "prompt_chars": len(p_arm),
                    "delta_chars_vs_base": len(p_arm) - len(p_base),
                    "sha256": sha(p_arm),
                    "is_control": arm == "base",
                    # The CONTROL must be byte-identical to the base prompt; every other
                    # arm must move it. `think` moves it via the directive, which
                    # `induce_prompt` does not emit -- `generate` prepends it -- so its
                    # prompt-text delta is legitimately zero and its movement is
                    # witnessed ON THE WIRE per cell (wire_has_codeonly_directive).
                    "differs_from_base": (
                        (p_arm == p_base)
                        if arm == "base"
                        else ((p_arm != p_base) or ARMS[arm]["codeonly"] == "0")
                    ),
                    "marker_present": (mark in p_arm) if mark else None,
                    "base_is_prefix": p_arm.startswith(p_base),
                }
            )
    bad = [w for w in witness if not w["differs_from_base"]]
    if bad:
        (OUT / f"blocked_shard{SHARD}.json").write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_treatment_witness_arm_did_not_move_the_prompt",
                    "failing": bad[:20],
                },
                indent=2,
            )
        )
        print("TREATMENT WITNESS FAILED:", [(w["game"], w["arm"]) for w in bad][:10])
        return 1
    print(f"treatment witness OK on {len(witness)} (game, arm) prompts", flush=True)

    # ---------------- PRE-REGISTRATION (before the first LLM call) --------------------------
    plain_eligible = [g for g in games if g not in HIDDEN_STATE and g != "tn36"]
    prereg = {
        "experiment": "arc_generation_ablation_heldout_engine_accuracy",
        "written_before_any_llm_call": True,
        "question": "Does any of three prompt interventions -- removing the code-only directive, "
        "forbidding an identity engine, or reshaping the output contract to the delta "
        "language the prompt already teaches -- raise HELD-OUT engine accuracy?",
        "arms": {a: ARMS[a]["desc"] for a in ARM_ORDER},
        "control": "base",
        "PRIMARY": {
            "metric": "held-out change_accuracy on the PRODUCTION TAIL split "
            "(_split_prefix_heldout, the same split the record's 0-of-296 null was "
            "measured on), per game, averaged over replicates",
            "definition_correction": "change_accuracy = n_changes_correct / n_changing where "
            "n_changes_correct counts WHOLE-GRID EXACT matches on "
            "changing rows. It is NOT a cell fraction; cell_recall and "
            "change_fidelity are the cell measures and are secondaries.",
            "target_from_the_brief": ">= 0.5 on any game but tn36, on more than a handful of rows",
            "test": "exact two-sided paired sign test over GAMES, each arm vs base, ties dropped",
            "clustering": "GAME. Replicates within a game are averaged into ONE per-game mean "
            "before pairing. 20 games x 3 replicates is 20 units, not 60.",
            "alpha": 0.05,
        },
        "SECOND_HELD_OUT_SET": {
            "name": "fresh",
            "what": "collect_transitions exploration of the OFFLINE sim from reset at a fixed "
            "seed, deduped against the shown prefix by BOTH a content digest and a "
            "rendered-prompt-line membership test",
            "why": "the production tail leaves ~3 gradable changing rows per game, on which "
            "change_accuracy can only take the values {0, 1/3, 2/3, 1}. The brief asks for "
            "'more than a handful of rows'; the fresh set supplies 120-220 per game.",
            "direction_of_the_bar": "STRICTLY HARDER, not easier: >= 0.5 here means exactly "
            "predicting more than half of ~150 whole 64x64 grids.",
            "distribution_caveat": "fresh rows come from random exploration from reset, the "
            "window from a banked winning route. That is a distribution "
            "shift and is stated as one -- though it is arguably the more "
            "live-faithful of the two, since the live agent induces on its "
            "stall-triggered exploration buffer and must then predict its "
            "own continued exploration.",
        },
        "SECONDARIES": [
            "held-out cell_recall (the quantity the brief's prose describes)",
            "held-out change_fidelity (union-scored: a spurious write costs what a miss costs)",
            "n engines above 0.0 on change_accuracy at all -- pre-registered as the channel that "
            "may matter more than the mean when the record sits at exactly zero",
            "identity rate (engine returns its input on every probed action) -- the antiid arm's "
            "own mechanism witness",
            "induce_ok rate -- removing the code-only directive may reintroduce the truncation "
            "failure it was shipped to fix; that would be a real cost of the think arm",
            "generated tokens, wall seconds, completion calls",
        ],
        "multiplicity": "One primary metric x 3 arm-vs-base contrasts. A p below 0.05 on one "
        "contrast reads against a Bonferroni threshold of 0.05/3 = 0.0167. The "
        "secondaries are exploratory and read against 0.05/6 = 0.00833.",
        "roster": ROSTER,
        "roster_provenance": "reused UNCHANGED from the 2026-08-01 corpora, so it cannot have "
        "been chosen after seeing which games favour an arm",
        "PRIMARY_SCOPE_BRANCH": {
            "plain_branch_games_excluding_tn36": plain_eligible,
            "n": len(plain_eligible),
            "why": "HIDDEN_STATE_GAME_IDS is a hardcoded 11-game PUBLIC tuple, so a hidden Kaggle "
            "game always takes the PLAIN branch. Hidden-state games are run and reported "
            "as their own stratum, never pooled into a hidden-branch claim.",
        },
        "POWER_STATED_UP_FRONT": {
            "min_reachable_two_sided_p_all_20_games_discordant": round(
                min_reachable_two_sided_p(len(ROSTER)), 12
            ),
            "min_reachable_two_sided_p_plain_branch_ex_tn36": round(
                min_reachable_two_sided_p(len(plain_eligible)), 12
            ),
            "n_discordant_needed_for_p_below_0.05": 6,
            "n_discordant_needed_for_p_below_bonferroni_0.0167": 8,
            "HONEST_EXPECTATION": "The record is 0 of 296 clean units at change_accuracy > 0.0000 "
            "across 14 games and both branches. If that floor holds in "
            "every arm the primary is ALL TIES, no test is possible, and "
            "the honest report is 'unmeasurable at this floor' -- NOT 'no "
            "difference'. That is why an instrument-reachability probe runs "
            "BEFORE the arms (see REACHABILITY_PROBE) and why the "
            "continuous secondaries are pre-registered alongside.",
            "CAN_THE_PRIMARY_REACH_0.05": "Only if at least 6 games move. Given the record, the "
            "likeliest outcome is a floored primary and whatever "
            "signal exists appearing on cell_recall / "
            "change_fidelity. Said before any result.",
        },
        "REACHABILITY_PROBE": "Before the arms run, a HAND-WRITTEN lookup oracle -- not an induced "
        "engine and not a claim about anything -- is scored on both held-out "
        "sets. If it does not reach change_accuracy 1.0 the metric is "
        "arithmetically pinned and every 'zero' in this run would be forced "
        "rather than measured. This is the check a prior arm skipped, whose "
        "'0 plans' was later found to be unfalsifiable because a hardcoded "
        "0.5 threshold sat above an achievable maximum of 0.0476.",
        "MISSING_VS_ZERO": {
            "missing": "server failure, HTTP error, harness exception, or a scoring worker "
            "timeout. EXCLUDED and counted -- never coerced to 0.",
            "zero": "a complete response whose engine loads and scores 0. A real failure, "
            "scores 0.",
        },
        "STOPPING_RULE": "every roster game x 4 arms x N replicates, or the wall budget. Analysis "
        "runs ONCE after collection stops. No peeking-and-extending. Only "
        "(game, replicate) pairs where BOTH the arm and base ran enter a "
        "contrast.",
        "n_replicates": N_REPLICATES,
        "seed_base": SEED_BASE,
        "max_tokens_all_arms": MAX_TOKENS,
        "tries_all_arms": TRIES,
        "held_equal_across_arms": [
            "window",
            "split",
            "server",
            "seed",
            "max_tokens",
            "tries",
            "temperature schedule",
            "roster",
        ],
        "flag_remains_default_off": True,
        "not_submitted": "no scored or online ARC game is played; submission is operator-only",
        "no_behavioural_claim": "no action, plan, level or episode is measured or moved",
    }
    prereg_text = json.dumps(prereg, indent=2, sort_keys=True)
    (OUT / f"preregistration_shard{SHARD}.json").write_text(prereg_text)
    prereg_sha = "sha256:" + sha(prereg_text)
    print(f"pre-registration {prereg_sha}", flush=True)
    print(
        f"  min reachable two-sided p (20 games) = "
        f"{min_reachable_two_sided_p(len(ROSTER)):.3e}; p<0.05 needs >=6 discordant",
        flush=True,
    )

    if os.environ.get("ABL_DRY") == "1":
        (OUT / f"meta_dry_shard{SHARD}.json").write_text(
            json.dumps({"witness": witness, "prereg_sha256": prereg_sha, "games": mine}, indent=2)
        )
        print("DRY RUN: stopping before the first LLM call")
        return 0

    # ---------------- SERVER ----------------------------------------------------------------
    os.environ["CARNOT_ARC_INDUCE_N_CTX"] = os.environ.get("ABL_N_CTX", "32768")
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = GPU
    prop = e3.LocalGGUFProposer(port=PORT)
    prop.max_tokens = MAX_TOKENS
    prop.tries = TRIES
    prop.no_think_prefix = ""  # gemma is not a hybrid-thinking model; the live gemma pin is ""
    print(f"launching {prop.repo_substr} port {PORT} gpu {GPU} n_ctx {prop.n_ctx} ...", flush=True)
    if not prop._ensure_server():
        (OUT / f"blocked_shard{SHARD}.json").write_text(
            json.dumps({"honest_verdict": "blocked_generator_server_failed_to_start"}, indent=2)
        )
        return 1
    actual_port = prop.port
    pid = listening_pid(actual_port)
    exe = os.readlink(f"/proc/{pid}/exe") if pid else None
    props = server_props(actual_port)
    sw = {
        "pid": pid,
        "exe_from_proc": exe,
        "port_requested": PORT,
        "port_actual": actual_port,
        "cuda_gpu": GPU,
        "n_ctx_declared": prop.n_ctx,
        "n_ctx_from_props": props.get("default_generation_settings", {}).get("n_ctx")
        or props.get("n_ctx"),
        "model_from_props": props.get("model_path") or props.get("model"),
        # build-hip/ is the AMD iGPU path; a run that silently landed there is ~6x slower and is a
        # different substrate. Proven from /proc/<pid>/exe, not from what we asked for.
        "is_cuda_build": bool(exe and "build-hip" not in exe and exe.endswith("llama-server")),
        "max_tokens": prop.max_tokens,
        "tries": prop.tries,
        "mtp": prop.mtp,
        "kv_quant": prop.kv_quant,
        "n_gpu_layers": prop.n_gpu_layers,
    }
    if not sw["is_cuda_build"]:
        (OUT / f"blocked_shard{SHARD}.json").write_text(
            json.dumps(
                {"honest_verdict": "blocked_generator_not_on_the_cuda_build", "server_witness": sw},
                indent=2,
            )
        )
        print("BLOCKED: not the CUDA build ->", exe)
        return 1
    (OUT / f"server_witness_shard{SHARD}.json").write_text(json.dumps(sw, indent=2))
    print("server:", json.dumps(sw), flush=True)

    # ---------------- CELLS ------------------------------------------------------------------
    def run_cell(game: str, rep: int, arm: str) -> dict:
        s = store[game]
        cell_id = f"{game}__r{rep}__{arm}"
        cell_dir = SCRATCH / cell_id
        if cell_dir.exists():
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True)
        e3.E3_DIR = cell_dir  # module global, read at call time; per-cell isolation

        cfg = ARMS[arm]
        os.environ["CARNOT_ARC_CODEONLY_INDUCE"] = cfg["codeonly"]
        os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(SEED_BASE + rep)

        suffix = cfg["suffix"]

        def patched_induce_prompt(g, trans, c, **kw):
            return base_prompt_fn(g, trans, c, **kw) + suffix

        e3.induce_prompt = patched_induce_prompt
        _WIRE.clear()
        sf0, cf0 = prop.n_server_failures, prop.n_content_failures
        calls0 = prop.n_completion_calls
        t = time.time()
        try:
            ok, msg = prop.induce(game, s["shown"], s["cell"])
            exc = None
        except Exception as e:  # noqa: BLE001
            ok, msg, exc = False, "", f"{type(e).__name__}: {e}"[:300]
        finally:
            e3.induce_prompt = base_prompt_fn
            os.environ.pop("CARNOT_ARC_CODEONLY_INDUCE", None)
        elapsed = time.time() - t

        # ---- ON-THE-WIRE WITNESS: what the server actually received -------------------------
        wire = list(_WIRE)
        directive_head = e3._L2_CODEONLY_DIRECTIVE[:40]
        row = {
            "cell_id": cell_id,
            "game": game,
            "replicate": rep,
            "arm": arm,
            "shard": SHARD,
            "gpu": GPU,
            "port": actual_port,
            "seed": SEED_BASE + rep,
            "elapsed_s": round(elapsed, 2),
            "induce_ok": bool(ok),
            "induce_msg": str(msg)[:300],
            "exception": exc,
            "n_wire_calls": len(wire),
            "wire_prompt_chars": [len(w["prompt"]) for w in wire],
            "wire_n_predict": sorted({int(w.get("n_predict", -1)) for w in wire}),
            "wire_seeds": sorted({int(w["seed"]) for w in wire if "seed" in w}),
            # Was the directive REALLY absent/present in what crossed the wire?
            "wire_has_codeonly_directive": [w["prompt"].startswith(directive_head) for w in wire],
            # Was the arm's own clause REALLY in what crossed the wire? None for arms with no mark.
            "wire_has_arm_marker": (
                [cfg["mark"] in w["prompt"] for w in wire] if cfg["mark"] else None
            ),
            "server_failures_delta": prop.n_server_failures - sf0,
            "content_failures_delta": prop.n_content_failures - cf0,
            "completion_calls_delta": prop.n_completion_calls - calls0,
            "last_stop_type": prop.last_stop_type,
            "generated_tokens": int(prop.last_generated_tokens),
            "prompt_truncated": bool(prop.last_prompt_truncated),
            "n_shown": len(s["shown"]),
            "n_tail": len(s["tail"]),
            "n_fresh": len(s["fresh"]),
            "cell": s["cell"],
        }
        # ARM INTEGRITY. The FIRST wire call is the engine induce; its directive state must match
        # the arm. A mismatch means the arm did not take effect and the cell is not evidence.
        if wire:
            row["arm_directive_consistent"] = row["wire_has_codeonly_directive"][0] == (
                cfg["codeonly"] == "1"
            )
            row["arm_marker_consistent"] = (
                True if not cfg["mark"] else bool(row["wire_has_arm_marker"][0])
            )
        else:
            row["arm_directive_consistent"] = None
            row["arm_marker_consistent"] = None

        wm = cell_dir / game / "world_model.py"
        row["engine_file_exists"] = wm.exists()
        if wm.exists():
            code = wm.read_text()
            row["engine_sha256"] = sha(code)
            row["engine_bytes"] = len(code)
            (ENGINES / f"{cell_id}.py").write_text(code)

        # MISSING vs ZERO. A server-side failure or harness exception is a MISSING observation.
        # A complete response producing bad code is a real zero and is scored as one.
        row["missing"] = bool(row["server_failures_delta"] > 0 or exc is not None)
        row["missing_reason"] = (
            "server_failure"
            if row["server_failures_delta"] > 0
            else ("harness_exception" if exc else None)
        )
        (CELLS / f"{cell_id}.json").write_text(json.dumps(row, indent=2))
        return row

    order: list[tuple[str, int, str]] = []
    for rep in range(N_REPLICATES):
        for game in mine:
            # Rotate arm order so any server drift within a replicate is not confounded with arm.
            rot = (rep + mine.index(game)) % len(ARM_ORDER)
            for a in ARM_ORDER[rot:] + ARM_ORDER[:rot]:
                order.append((game, rep, a))

    rows: list[dict] = []
    print(f"\n{len(order)} cells queued on shard {SHARD}\n", flush=True)
    for i, (game, rep, arm) in enumerate(order, 1):
        if time.time() - t0 > WALL_BUDGET_S:
            print(f"WALL BUDGET reached after {i - 1} cells; stopping", flush=True)
            break
        cp = CELLS / f"{game}__r{rep}__{arm}.json"
        if cp.exists():  # RESUME: a completed cell is evidence already on disk
            rows.append(json.loads(cp.read_text()))
            continue
        r = run_cell(game, rep, arm)
        rows.append(r)
        print(
            f"[{SHARD}:{i}/{len(order)}] {game} r{rep} {arm:6} ok={r['induce_ok']} "
            f"calls={r['completion_calls_delta']} tok={r['generated_tokens']} "
            f"stop={r['last_stop_type']} dir_ok={r['arm_directive_consistent']} "
            f"{r['elapsed_s']}s",
            flush=True,
        )
        (OUT / f"rows_shard{SHARD}.json").write_text(json.dumps(rows, indent=2))

    (OUT / f"meta_shard{SHARD}.json").write_text(
        json.dumps(
            {
                "prereg_sha256": prereg_sha,
                "server_witness": sw,
                "treatment_witness": witness,
                "n_cells_run": len(rows),
                "n_cells_queued": len(order),
                "duration_s": round(time.time() - t0, 2),
                "games": mine,
                "arms": ARM_ORDER,
                "seed_base": SEED_BASE,
                "n_replicates": N_REPLICATES,
                "gguf": str(gguf),
                "prep_meta_ref": str(OUT / "prep_meta.json"),
                "prep_leak_check": prep_meta.get("_prep", {}).get("leak_check_definition"),
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nshard {SHARD} done: {len(rows)} cells in {round(time.time() - t0, 1)}s", flush=True)
    return 0


def _gpu_offload() -> bool:
    try:
        from llama_cpp import llama_cpp as b

        return bool(b.llama_supports_gpu_offload())
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
