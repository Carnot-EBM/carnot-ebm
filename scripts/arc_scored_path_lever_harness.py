"""THE SCORED-PATH HARNESS, WITH THE LLM ON.

WHAT THIS IS. Every lever measured on 2026-07-25 was measured on `CarnotAgentPolicy` with
`CARNOT_ARC_DISABLE_INDUCTION=1` -- i.e. the tier-1 explorer alone, no LLM. The SCORED agent is
`make_carnot_agent(Agent)` -> `E3AgentPolicy`, which runs the full verifier-routed cascade
(explore -> induce with a real local LLM -> verify -> plan -> execute). This harness drives
E3AgentPolicy on the OFFLINE public arcade with the LLM ACTUALLY ON, and instruments the cost.

WHY IT MATTERS THAT THE LLM IS ON, AND HOW THE PREVIOUS HARNESS TURNED IT OFF. The escape hatch
lives at arc_competition_agent.py:5087 -- `if os.environ.get("CARNOT_ARC_DISABLE_INDUCTION") == "1":
attempt["skipped"] = "disabled_by_env"; return`. The earlier e3_probe.py sets that var at import
(`os.environ.setdefault("CARNOT_ARC_DISABLE_INDUCTION", "1")`), so its E3 arm was the scored
POLICY but not the scored CASCADE. This harness explicitly DELETES the var and asserts it is gone.

THE OBSERVE-CHANNEL LESSON, APPLIED. A lever that never fires produces a clean, zero-error NULL
that looks like a finding and is a bug. So every row here carries FIRE COUNTERS that must be
non-zero for the row to mean anything:
  * llm_generate_calls / llm_tokens_predicted -- did the LLM actually run?
  * induction_attempts_llm_reached -- did induction get past the ttt/nav short-circuits to the LLM?
  * nodes_with_previous_frame / nodes_total -- is the observe channel that frame-dependent levers
    read actually populated on THIS path?
  * frontier tier + hud mask diagnostics -- are the seven shipped-lever flags resolved as expected,
    read off the live explorer rather than assumed from the module globals?

COST IS THE DELIVERABLE. Per row: total wall clock, wall clock spent inside the LLM, LLM request
count, prompt+predicted tokens, actions, states expanded. That is what makes an honest
"is 25 games x 3 seeds affordable?" answer possible instead of a guess.

THE EIGHTH GATED FLAG (2026-07-26, REQ-ARC-WMTE-5970). `hazard_move_pruner` -- the NAV-side hazard
move-pruner -- is now wired into the scored explorer DEFAULT-OFF and joins the pinned set. It gets a
first-class fire-counter block and a `lever3_verdict` string, because it is the lever most likely to
produce a misleading null: two pre-wiring censuses say it fits on 0 of 25 / 1 of 15 public games and
prunes nothing. `lever3_verdict` therefore distinguishes FOUR outcomes that all look like
"rows_pruned == 0":
  * UNINTERPRETABLE_NO_OBSERVE   -- the observe channel never fired. A WIRING BUG, not a result.
  * UNINTERPRETABLE_NO_NAV       -- the game issued no keyboard-nav action; no jurisdiction.
  * UNINTERPRETABLE_NOT_FITTED   -- transitions seen, but the hypothesis class did not fit.
  * FIRED_NO_PRUNE               -- fitted and predicted nothing lethal. The only reportable null.
Only FIRED_NO_PRUNE and FIRED_AND_PRUNED are evidence about the lever's value.

BUDGET: THE DEFAULT IS 400 BECAUSE THAT IS WHAT THE SHIPPED AGENT DOES -- NOT BECAUSE THE EVAL
IMPOSES IT. CORRECTED 2026-07-26 after an adversarial review caught this docstring misreading its
own source. The shipped agent is capped at `MAX_ACTIONS = 400` per game (`arc_competition_agent.py`'s
CarnotAgent adapter; the framework loop is `while not done and action_counter <= MAX_ACTIONS`), so
400 is the condition the CURRENT SUBMISSION runs under and that is why it is the default here. But
400 is a SELF-IMPOSED PER-GAME LOOP GUARD that we chose and can raise, not an eval-imposed bound.
The comment directly above that constant says so: "The real bound is the eval's wall-clock budget
(<=12h across all games), NOT this per-game loop guard; Playback overrides it to 1e6 for the same
reason, so it is an INTENDED OVERRIDE POINT. 400 comfortably covers our multi-level replays +
explore while staying well inside the time budget."

WHY THE DISTINCTION IS LOAD-BEARING, WITH NUMBERS. Lever conclusions REVERSE with the budget. At
budget 2000 on the LLM-off dev twin the convention-perturbation battery
(`results/outer_loop_cptb_shipped_lever_convention_transfer_20260726.json`, C0_real) measures the
SHIPPED configuration as the BEST of four arms -- median 12 wins vs 11 frontier-only, 9 HUD-only,
7 all-off. At budget 400 the same shipped configuration wins only 3-4 of 25. The sibling harness
`experiment_5836_frontier_discipline_ab` states the mechanism outright: measured first-win costs on
its baseline span 20 (lp85) to 1747 (cd82) actions, so a budget below ~2000 structurally cannot see
most of the signal. THEREFORE: a budget-400 result is a statement about the CURRENT SUBMISSION'S
configuration, and a budget-2000 result is a statement about the levers themselves. Neither is
"the eval's condition" in the sense of a constraint we cannot change, and neither may be used to
recommend a flag change without the other. Pass `--budget 2000` for the lever-value condition; the
emitted row file records `budget`, `scored_agent_max_actions` and `budget_matches_scored_cap` so a
reader can always tell which one they are holding.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

# LLM ON. Delete rather than set-to-0: the agent's check is `== "1"`, but an explicit delete makes
# the intent unmistakable and survives a future change to truthiness parsing.
os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
# The conductor owns GPU 0 (systemd drop-in 40-arc-generator-3090-20260619.conf sets
# CARNOT_ARC_GENERATOR_CUDA_GPU=0); the outer loop gets GPU 1.
os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")

GAMES_25 = [
    "ar25",
    "bp35",
    "cd82",
    "cn04",
    "dc22",
    "ft09",
    "g50t",
    "ka59",
    "lf52",
    "lp85",
    "ls20",
    "m0r0",
    "r11l",
    "re86",
    "s5i5",
    "sb26",
    "sc25",
    "sk48",
    "sp80",
    "su15",
    "tn36",
    "tr87",
    "tu93",
    "vc33",
    "wa30",
]
SEEDS = [20260724, 20260725, 20260726]

# The seven flags flipped/held on 2026-07-25, read off the LIVE explorer instance.
GATED_FLAG_ATTRS = (
    "tier_exhaustion_enabled",
    "tier_uniform_random_enabled",
    "tier_click_vocab_only",
    "frontier_gradient_enabled",
    "edge_bar_hud_mask_enabled",
    "hud_mask_collapse_guard_enabled",
    "hud_mask_stage2_confirm_enabled",
    # REQ-ARC-WMTE-5970 (2026-07-26): the eighth gated flag.
    "hazard_move_pruner_enabled",
)

# THE SHIPPED CONFIGURATION as of 2026-07-25, as the seven E3AgentPolicy CONSTRUCTOR kwarg names
# (not the attribute names above). Verified equal to what `E3AgentPolicy(game)` resolves from the
# SUBMITTED_* module globals -- pinned EXPLICITLY anyway so an arm can never silently drift when a
# global is flipped, which is exactly the defect that turned exp5836's arm B2 from "the live
# config" into "the pre-HUD-flip config" without anyone editing arm B2.
SHIPPED_2026_07_25 = {
    "tier_exhaustion": True,
    "tier_uniform_random": True,
    "tier_click_vocab_only": True,
    "frontier_gradient": False,
    "edge_bar_hud_mask": True,
    "hud_mask_collapse_guard": True,
    "hud_mask_stage2_confirm": True,
    # REQ-ARC-WMTE-5970: wired 2026-07-26, DEFAULT-OFF, never flipped. Pinned False in the SHIPPED
    # dict so that "S" stays the control even after a future flip -- the whole point of pinning.
    "hazard_move_pruner": False,
}
_FRONTIER_KEYS = (
    "tier_exhaustion",
    "tier_uniform_random",
    "tier_click_vocab_only",
    "frontier_gradient",
)
_HUD_KEYS = ("edge_bar_hud_mask", "hud_mask_collapse_guard", "hud_mask_stage2_confirm")

# EVERY arm pins ALL EIGHT flags. A partially-pinned arm inherits whatever the module globals say
# at run time, so a later flip silently redefines that arm -- the exp5836/B2 failure mode, where the
# HUD A/B's own CONTROL silently became the HUD TREATMENT.
ARMS: dict[str, dict] = {
    # The control: exactly what ships today.
    "S": dict(SHIPPED_2026_07_25),
    # Lever 1 removed (the frontier tier trio), HUD held at shipped.
    "S_minus_frontier": {**SHIPPED_2026_07_25, **{k: False for k in _FRONTIER_KEYS}},
    # Lever 2 removed (the HUD edge-bar trio), frontier held at shipped.
    "S_minus_hud": {**SHIPPED_2026_07_25, **{k: False for k in _HUD_KEYS}},
    # Both removed -- the pre-2026-07-25 agent, for context only (not a single-lever delta).
    "S_minus_both": {
        **SHIPPED_2026_07_25,
        **{k: False for k in _FRONTIER_KEYS},
        **{k: False for k in _HUD_KEYS},
    },
    # Lever 3 ADDED (the nav-side hazard move-pruner). Its control is "S", which differs from it in
    # exactly one flag -- asserted below rather than asserted by eye.
    "S_plus_hazard": {**SHIPPED_2026_07_25, "hazard_move_pruner": True},
    # THE NOISE FLOOR (2026-07-26). Byte-identical configuration to "S" -- deliberately NOT a
    # treatment. With the LLM ON the run is no longer a deterministic function of the seed: the
    # generator's sampling, and the server's slot/checkpoint state, vary between two runs of the
    # SAME config. Without this arm, ANY difference between S and a treatment arm is
    # indistinguishable from that variation, and an A/B on a single seed would report sampling
    # noise as a lever effect. This arm measures the noise directly, on the same games and seeds,
    # so the lever deltas can be read against it instead of against an assumption of determinism.
    "S_replicate": dict(SHIPPED_2026_07_25),
}
assert ARMS["S_replicate"] == ARMS["S"], (
    "S_replicate must be byte-identical to S -- it is the same-config noise floor, not a treatment"
)
assert all(set(v) == set(SHIPPED_2026_07_25) for v in ARMS.values()), (
    "every arm must pin all eight gated flags"
)
assert {k for k in SHIPPED_2026_07_25 if ARMS["S_plus_hazard"][k] != ARMS["S"][k]} == {
    "hazard_move_pruner"
}, "S_plus_hazard must isolate the nav pruner against the live config, nothing else"


def assert_shipped_dict_matches_module_globals() -> dict:
    """Cross-check the pinned SHIPPED dict against the agent's own SUBMITTED_* globals.

    Pinning protects an ARM from a later flip; it does NOT tell you whether the arm named "S" is
    still the live configuration. This does, and it is cheap. If a flag is flipped in
    `arc_competition_agent` and nobody updates this file, "S" silently stops being the control --
    the exact drift that made exp5836's arm B2 the HUD treatment. Returns the comparison so a run
    artifact can record it.
    """

    from carnot.agentic import arc_competition_agent as comp

    live = {
        "tier_exhaustion": comp.SUBMITTED_FRONTIER_TIER_EXHAUSTION_ENABLED,
        "tier_uniform_random": comp.SUBMITTED_FRONTIER_TIER_UNIFORM_RANDOM_ENABLED,
        "tier_click_vocab_only": comp.SUBMITTED_FRONTIER_TIER_CLICK_VOCAB_ONLY_ENABLED,
        "frontier_gradient": comp.SUBMITTED_FRONTIER_DISTANCE_GRADIENT_ENABLED,
        "edge_bar_hud_mask": comp.SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED,
        "hud_mask_collapse_guard": comp.SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED,
        "hud_mask_stage2_confirm": comp.SUBMITTED_HUD_MASK_STAGE2_CONFIRM_ENABLED,
        "hazard_move_pruner": comp.SUBMITTED_HAZARD_MOVE_PRUNER_ENABLED,
    }
    drift = {k: [SHIPPED_2026_07_25[k], live[k]] for k in live if SHIPPED_2026_07_25[k] != live[k]}
    return {"live_submitted_flags": live, "pinned_vs_live_drift": drift}


class InstrumentedProposer:
    """Transparent counting wrapper around LocalGGUFProposer.

    Wraps by DELEGATION rather than subclassing because LocalGGUFProposer is a @dataclass whose
    callers set attributes on it (`include_playbook_exemplars`), so __getattr__/__setattr__
    forwarding keeps the real object authoritative. Counts are taken at the two POST call sites
    (`generate`, `complete_text`) plus `_record_completion_diagnostics`, which the class itself
    calls on EVERY completion response -- that is where llama.cpp's own `timings` block lives, so
    the token counts are the server's, not an estimate.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(
            self,
            "stats",
            {
                "generate_calls": 0,
                "complete_text_calls": 0,
                "responses": 0,
                "tokens_prompt": 0,
                "tokens_predicted": 0,
                "llm_wall_s": 0.0,
                "stop_type_limit": 0,
                "prompt_truncated": 0,
                "errors": 0,
            },
        )
        # PATCH THE INNER INSTANCE, not just this wrapper. Delegation alone under-counts: the agent
        # calls high-level methods (`induce`, `refactor`, `induce_programmatic_experts`) that are
        # defined ON LocalGGUFProposer, so `__getattr__` hands back a method BOUND TO THE INNER
        # object, and `self.generate(...)` inside it resolves to the inner's generate -- never this
        # wrapper's. Measured on the first smoke run: generate_calls=0 while responses=11. Binding
        # the counters to the inner instance's own attributes closes that gap.
        object.__setattr__(self, "_orig_record", inner._record_completion_diagnostics)
        object.__setattr__(self, "_orig_generate", inner.generate)
        object.__setattr__(self, "_orig_complete_text", inner.complete_text)
        inner._record_completion_diagnostics = self._record  # type: ignore[method-assign]
        inner.generate = self._generate_counted  # type: ignore[method-assign]
        inner.complete_text = self._complete_text_counted  # type: ignore[method-assign]

    # -- the token/timings tap ------------------------------------------------------------
    def _record(self, response: dict) -> None:
        s = self.stats
        s["responses"] += 1
        tim = (response or {}).get("timings") or {}
        try:
            s["tokens_prompt"] += int(tim.get("prompt_n") or 0)
            s["tokens_predicted"] += int(tim.get("predicted_n") or 0)
        except Exception:
            pass
        if str((response or {}).get("stop_type") or "") == "limit":
            s["stop_type_limit"] += 1
        if (response or {}).get("truncated"):
            s["prompt_truncated"] += 1
        self._orig_record(response)

    # -- the two request entry points, bound onto the INNER instance -----------------------
    def _generate_counted(self, *a, **kw):
        self.stats["generate_calls"] += 1
        t = time.time()
        try:
            return self._orig_generate(*a, **kw)
        except Exception:
            self.stats["errors"] += 1
            raise
        finally:
            self.stats["llm_wall_s"] += time.time() - t

    def _complete_text_counted(self, *a, **kw):
        self.stats["complete_text_calls"] += 1
        t = time.time()
        try:
            return self._orig_complete_text(*a, **kw)
        except Exception:
            self.stats["errors"] += 1
            raise
        finally:
            self.stats["llm_wall_s"] += time.time() - t

    # Kept so a caller holding the WRAPPER (rather than the inner) is still counted.
    def generate(self, *a, **kw):
        return self._inner.generate(*a, **kw)

    def complete_text(self, *a, **kw):
        return self._inner.complete_text(*a, **kw)

    def forbid_spawn(self) -> None:
        """Make `_ensure_server()` a pure HEALTH CHECK that never launches a server.

        WHY THIS EXISTS -- a measured server-storm, 2026-07-26. `LocalGGUFProposer._healthy()`
        gives `/health` a 2-SECOND timeout, and `_ensure_server()` spawns a brand-new server
        whenever that check fails, with no test for "is something already bound to this port".
        Under the load this harness generates (the E3 cascade's own search sits at 280-600% CPU
        while the server is mid-generation) a 2s health probe can time out on a server that is
        perfectly alive. Observed consequence: THREE llama-servers alive simultaneously, all
        children of the harness (elapsed 2:33 / 0:39 / 0:07), VRAM on the card climbing past
        13.5 GB, and servers dying off as the pile-up collided -- which then silently degraded the
        scored path to LLM-off for the rest of the run. `results`-side symptom is a row with
        `generator_healthy_after: False`, which is why that witness exists.
        (Same mechanism, different symptom, as the 2026-07-21 exp5768 incident recorded in
        `_GENERATOR_CUDA_FREE_RETRY_ATTEMPTS`' docstring: repeated self-heals landing somewhere
        worse than where they started.)
        With spawning forbidden, a transient health blip costs ONE induction (`generate()` returns
        its `(False, "GPU llama-server failed ...")` tuple, the agent logs `proposer_failed`) instead
        of forking a second copy of a 12 GB model onto the card.

        MEASURED 2026-07-26, THE OTHER HALF OF THIS PROBLEM. Forbidding the spawn is necessary but
        not sufficient, because `_healthy()` hard-codes a 2-SECOND urllib timeout. Under this
        harness's own load (two shard processes at ~300% CPU each, plus the server mid-generation)
        that probe can fail on a server that is perfectly alive -- and with spawning forbidden a
        failed probe makes `generate()` return its `(False, ...)` tuple, so the agent logs
        `proposer_failed` and the cell completes with ZERO completions. Observed: an ar25 cell that
        had produced 7 real completions on an earlier run came back with `resp=0` and
        `genok=True->False` while the server was demonstrably up and serving at 165 tok/s. The row
        is correctly stamped `llm_on_row_valid: False`, so it is not silently believed -- but it is
        a WASTED cell, and if the blip is frequent enough the whole run degrades to LLM-off.
        So the replacement is a RETRYING probe with a longer timeout: several attempts, spaced, at
        a timeout appropriate to a loaded box. It still NEVER spawns, so the storm stays impossible;
        it just stops treating one slow response as a dead server. A genuinely dead server (systemd
        restarts it in ~12s, model load included) is still caught -- it simply gets the few seconds
        it needs to come back rather than being declared dead on the first 2s miss.
        """
        inner = self._inner

        def _health_only(attempts: int = 6, timeout: float = 10.0, gap: float = 3.0) -> bool:
            import urllib.request

            for i in range(attempts):
                try:
                    with urllib.request.urlopen(inner._url() + "/health", timeout=timeout) as r:
                        if b"ok" in r.read():
                            return True
                except Exception:
                    pass
                if i < attempts - 1:
                    time.sleep(gap)
            return False

        inner._ensure_server = _health_only  # type: ignore[method-assign]
        # The per-cell liveness WITNESS must use the same tolerant probe, or a cell whose LLM
        # worked fine gets stamped invalid by a 2s blip at the moment the cell happens to end.
        inner._healthy = _health_only  # type: ignore[method-assign]

    def snapshot(self) -> dict:
        return dict(self.stats)

    def reset(self) -> None:
        for k in self.stats:
            self.stats[k] = 0 if not isinstance(self.stats[k], float) else 0.0

    def __getattr__(self, name):  # noqa: D105 -- delegation
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        if name in ("_inner", "stats", "_orig_record"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


def build_proposer(port: int):
    """The FROZEN live generator, constructed exactly as E3AgentPolicy._proposer() does
    (arc_competition_agent.py:4742) -- Qwen3.5-9B-MTP, MTP self-draft, q8 KV, /no_think prefix,
    n_predict 4096, timeout 600, -ngl 999 -- with only the PORT changed, and wrapped for counting.

    Sharing ONE proposer across games mirrors arc_leaderboard_eval.py's own `_PROPOSER` global and
    is also what the real eval effectively gets: each E3AgentPolicy would lazily build its own
    wrapper object, but `_ensure_server()` reuses the healthy server on the port, so the ~14s model
    load is paid once per process either way.
    """
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    inner = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        model_path=os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
        mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0"),
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=int(os.environ.get("CARNOT_ARC_INDUCE_MAX_TOKENS", "4096")),
        timeout=int(os.environ.get("CARNOT_ARC_INDUCE_TIMEOUT", "600")),
        port=port,
        n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
    )
    return InstrumentedProposer(inner)


def _llama_server_count() -> int:
    """How many llama-server processes exist right now. A cell that STARTS with 1 and ENDS with 2+
    was measured during a server storm (see InstrumentedProposer.forbid_spawn) and its wall clock is
    contended -- the number that matters most here -- so the count is recorded per cell."""
    import subprocess

    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=10)
        return sum(1 for ln in out.stdout.splitlines() if "llama-server" in ln)
    except Exception:
        return -1


# THE HUD ROW SCHEMA IS SHARED, NOT DUPLICATED (2026-07-26).
# These four names used to be DEFINED here, and `experiment_5836_frontier_discipline_ab.run_cell`
# hand-rolled its own second projection of the SAME `hud_mask_diagnostics()` dict. Two independent
# projections of one source dict is exactly the defect that produced the two-address split: this
# harness emitted the fields nested, exp5836 emitted a DIFFERENT flat subset (omitting the
# shipped-side comparator and renaming node inflation), and each reader saw `None` on the other's
# rows. They now come from ONE module so a future added field has exactly one place to be added and
# both writers get it. Re-exported under the original names because this module's public surface is
# what `tests/python/test_arc_scored_path_lever_harness.py` and the analyser import.
from carnot.agentic.arc_hud_row_schema import (  # noqa: E402
    HUD_ROW_KEYS,
    LEVER2_FIRE_PREDICATE_VERSION,
    hud_lever_fired,
    hud_row_fields,
    lever2_scoreable,
)

# Historical spelling, kept so already-written call sites and tests do not have to change in the
# same commit that unifies the schema. `HUD_ROW_KEYS` is the canonical name.
HUD_FLAT_ROW_KEYS = HUD_ROW_KEYS

__all_hud_schema__ = (
    "HUD_ROW_KEYS",
    "HUD_FLAT_ROW_KEYS",
    "LEVER2_FIRE_PREDICATE_VERSION",
    "hud_lever_fired",
    "hud_row_fields",
    "lever2_scoreable",
)


def _hazard_verdict(hz: dict, row: dict) -> str:
    """Classify a hazard-pruner cell. THE POINT: `rows_pruned == 0` has four different meanings and
    only two of them are evidence about the lever.

    Ordered most-diagnostic-first, because an earlier condition explains a later zero:
      LEVER_OFF                     -- the control arm. No claim either way.
      ERROR                         -- diagnostics could not be read; the cell proves nothing.
      UNINTERPRETABLE_NO_OBSERVE    -- the observe hook never ran. This is the exp5836 dead-channel
                                       bug (0 of 122 nodes carried `previous_frame`), and it is a
                                       WIRING DEFECT to fix, never a null to report.
      UNINTERPRETABLE_NO_NAV        -- the hook ran but the pruner accepted nothing, i.e. the game
                                       issued no keyboard-nav action (six public games are 100%
                                       click). The lever has no jurisdiction here; excluding such
                                       games from a verdict's DENOMINATOR is mandatory, because
                                       including them dilutes a real effect toward zero.
      UNINTERPRETABLE_NOT_FITTED    -- nav transitions were seen but no hazard model passed the
                                       evidence + trust + specificity gate. Says something about the
                                       HYPOTHESIS CLASS on this game, nothing about pruning value.
      FIRED_NO_PRUNE                -- fitted, and predicted nothing lethal. A REAL null.
      FIRED_AND_PRUNED              -- fitted and withheld moves. A real measurement.
    """

    if not hz or hz.get("error"):
        return "ERROR"
    if not hz.get("enabled"):
        return "LEVER_OFF"
    if not int(hz.get("observe_calls") or 0):
        return "UNINTERPRETABLE_NO_OBSERVE"
    if not int(hz.get("observed_nav_transitions") or 0):
        return "UNINTERPRETABLE_NO_NAV"
    if not hz.get("model_fitted"):
        return "UNINTERPRETABLE_NOT_FITTED"
    return "FIRED_AND_PRUNED" if int(hz.get("rows_pruned") or 0) else "FIRED_NO_PRUNE"


def _action_key(move: dict) -> str:
    kind = move.get("kind")
    data = move.get("data") or {}
    if isinstance(data, dict) and ("x" in data or "y" in data):
        return f"{kind}@{data.get('x')},{data.get('y')}"
    if data:
        return f"{kind}:{json.dumps(data, sort_keys=True)}"
    return str(kind)


def run_cell(
    game: str,
    seed: int,
    *,
    budget: int,
    proposer,
    llm: bool,
    extra_kwargs: dict | None = None,
    arm: str = "E3_shipped",
    early_stop_grace: int | None = None,
) -> dict:
    """One (game, seed, arm) cell on the SCORED path. `llm=False` sets the disable-induction env for
    the duration of THIS cell only, so an LLM-on / LLM-off contrast can be run in one process
    without two harnesses.

    `early_stop_grace` SWEEPS THE PARAMETER WITHOUT FLIPPING THE FLAG. `StepwiseExplorer.__init__`
    already takes `early_stop_grace` (arc_competition_agent.py:1003) and `is_done` already
    implements the window (:3936-3946), but `E3AgentPolicy.__init__` never forwards it, so the
    module global `SUBMITTED_EARLY_STOP_GRACE` is currently DEAD CODE -- it is read nowhere. Setting
    the attribute on the constructed explorer is therefore the only way to exercise the mechanism at
    all, and it deliberately does NOT touch any `SUBMITTED_*` global: the shipped configuration is
    unchanged by this measurement, which is what "sweep by parameter, report, leave the decision to
    the operator" requires. Default None == the shipped behaviour, byte-for-byte.

    PRINCIPLE (why the attribute-set is safe rather than a behaviour change of its own): the three
    pieces of grace state (`early_stop_grace`, `_early_stop_level_mark`, `_early_stop_frame_mark`,
    `early_stopped`) are initialised together at :1344-1347 and are read ONLY inside `is_done`.
    Nothing else in `__init__` branches on the value, so assigning it immediately after construction
    is indistinguishable from having been passed through the constructor. Asserted below rather than
    asserted by eye."""
    import random

    import numpy as np

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    prev_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    if llm:
        os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    else:
        os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

    random.seed(seed)
    np.random.seed(seed % (2**32))

    # GENERATOR-LIVENESS WITNESS, per cell. The llama-server CAN DIE MID-RUN (observed twice on
    # 2026-07-26: the health-check server and its own self-healed replacement, the latter left as a
    # <defunct> zombie). When it does, `generate()` returns `(False, "GPU llama-server failed ...")`
    # instead of raising, so the agent logs `skipped: proposer_failed` and the run CONTINUES --
    # silently degrading the scored path to LLM-off while still emitting rows labelled `llm_on`.
    # That is the observe-channel-dead failure mode exactly: a clean, error-free row that means
    # nothing. Recording health on BOTH sides of the cell makes it detectable per row instead of
    # having to notice a GPU going idle.
    llm0 = proposer.snapshot() if proposer is not None else None
    gen_ok_before = bool(proposer._inner._healthy()) if proposer is not None else None
    servers_before = _llama_server_count()
    t0 = time.time()
    kw = {"frontier_discipline_seed": seed}
    if llm and proposer is not None:
        kw["proposer"] = proposer
    kw.update(extra_kwargs or {})
    policy = E3AgentPolicy(game, **kw)
    ex = policy.explorer
    # SET THE SWEPT PARAMETER, then READ IT BACK off the live explorer (below, `early_stop_grace`
    # in the row) -- never trust that the assignment took. An arm whose parameter silently failed to
    # apply is an UNINSTRUMENTED arm, and it would look like a clean null.
    if early_stop_grace is not None:
        ex.early_stop_grace = int(early_stop_grace)
        ex._early_stop_level_mark = 0
        ex._early_stop_frame_mark = 0
        ex.early_stopped = False
    construct_s = time.time() - t0

    row: dict = {
        "arm": arm,
        "game": game,
        "seed": seed,
        "budget": budget,
        "llm_enabled": bool(llm),
        "construct_s": round(construct_s, 3),
        # RESOLVED gated flags, read off the LIVE explorer -- never assumed from module globals.
        "gated_flags": {a: bool(getattr(ex, a)) for a in GATED_FLAG_ATTRS},
        "auto_hud_mask_enabled": bool(getattr(ex, "auto_hud_mask", None)),
        "explore_budget": int(policy.explore_budget),
        # The nine components whose presence is what makes node frames be RETAINED. Recorded so a
        # frame-dependent lever's null can be separated from a dead observe channel.
        "frame_retention_components": {
            "goal_bias": ex.goal_bias is not None,
            "dense_curiosity": ex.dense_curiosity is not None,
            "action_effect_expansion_prior": ex.action_effect_expansion_prior is not None,
            "qd_generator": ex.qd_generator is not None,
            "controllable_novelty_policy": ex.controllable_novelty_policy is not None,
            "object_centric_proposal_policy": ex.object_centric_proposal_policy is not None,
            "go_explore_archive": ex.go_explore_archive is not None,
            "similarity_retrieval_enabled": bool(ex.similarity_retrieval_enabled),
            "click_pixel_sampling_enabled": bool(ex.click_pixel_sampling_enabled),
        },
    }
    row["retains_node_frames"] = any(row["frame_retention_components"].values())
    # THE SWEPT PARAMETER, READ BACK off the live explorer -- not echoed from the argument. If the
    # attribute-set above ever stopped working (a __slots__, a property, a renamed attribute), this
    # reads None on a treatment arm and the arm is detectably uninstrumented rather than silently
    # equal to the control.
    row["early_stop_grace"] = (
        int(ex.early_stop_grace) if getattr(ex, "early_stop_grace", None) is not None else None
    )
    row["early_stop_grace_requested"] = early_stop_grace
    row["early_stop_grace_applied"] = bool(row["early_stop_grace"] == early_stop_grace)

    t1 = time.time()
    try:
        r = lb.run_game(game, policy, budget=budget, variant=0, reflect=None)
        row["ran"] = True
    except Exception as exc:
        row.update(
            ran=False,
            reason=f"{type(exc).__name__}:{exc}",
            wall_s=round(time.time() - t1, 2),
            states_expanded=(len(ex.graph) if ex is not None else None),
        )
        # THE CRASH PATH MUST BE INSTRUMENTED TOO. A row that omits the hud_* keys entirely reads
        # as `None` to a flat consumer, which is the same unreadable-vs-resolved-nothing ambiguity
        # this projection exists to remove -- and a crashed arm reading as a clean null across a
        # whole condition is a defect this project has already shipped once. The diagnostics are
        # still meaningful after a crash: whatever the explorer ingested before the exception is
        # what it ingested.
        try:
            crash_hud = ex.hud_mask_diagnostics() if ex is not None else {"error": "no_explorer"}
        except Exception as diag_exc:  # instrumentation must never mask the original failure
            crash_hud = {"error": f"{type(diag_exc).__name__}:{diag_exc}"}
        row["hud_diagnostics"] = crash_hud
        row.update(hud_row_fields(crash_hud))
        # Same reasoning as the HUD projection above: a crash row that OMITS these keys reads as
        # `None` to a flat consumer, which is indistinguishable from "measured and empty". Emit the
        # full key set explicitly, with `early_stopped` read off the explorer (it is meaningful
        # after a crash -- whatever the explorer did before the exception is what it did).
        row.update(
            per_level=None,
            level_up_actions=None,
            inter_levelup_gaps=None,
            actions_after_last_levelup=None,
            reached_any_level=None,
            early_stopped=bool(getattr(ex, "early_stopped", False)) if ex is not None else None,
            n_resets=None,
            n_frames=None,
        )
        if prev_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = prev_disable
        return row
    row["wall_s"] = round(time.time() - t1, 2)

    # ---- LLM cost, differenced so a shared proposer still gives per-cell numbers -----------
    if proposer is not None:
        now = proposer.snapshot()
        row["llm"] = {
            k: (round(now[k] - llm0[k], 3) if isinstance(now[k], float) else now[k] - llm0[k])
            for k in now
        }
        row["generator_healthy_before"] = gen_ok_before
        row["generator_healthy_after"] = bool(proposer._inner._healthy())
        row["llama_servers_before"] = servers_before
        row["llama_servers_after"] = _llama_server_count()
        row["server_storm_suspected"] = bool(
            servers_before >= 0 and row["llama_servers_after"] > servers_before
        )
        # A row is VALID as an LLM-on measurement only if the model actually produced tokens for
        # it. Zero responses on an llm=True cell means either the generator was dead or induction
        # never reached it -- in neither case is the row evidence about the LLM tier.
        row["llm_on_row_valid"] = bool(
            (not llm)
            or (
                row["llm"]["responses"] > 0
                and row["generator_healthy_after"]
                and not row["server_storm_suspected"]
            )
        )
    else:
        row["llm"] = None
        row["generator_healthy_before"] = None
        row["generator_healthy_after"] = None
        row["llm_on_row_valid"] = not llm

    # ---- induction accounting -------------------------------------------------------------
    atts = list(getattr(policy, "induction_attempts", []) or [])
    row["induction_attempts"] = len(atts)
    row["induction_reasons"] = dict(collections.Counter(a.get("reason") for a in atts))
    row["induction_skipped"] = dict(
        collections.Counter(a.get("skipped") for a in atts if a.get("skipped"))
    )
    row["induction_planned"] = sum(1 for a in atts if a.get("planned"))
    row["induction_engine_sources"] = dict(
        collections.Counter(a.get("engine_source") for a in atts if a.get("engine_source"))
    )
    # Did induction REACH the LLM, or short-circuit at the ttt-prior / structured-nav tier?
    # KEYED ON `model_specs`, which _induce_and_plan initialises to "offline_dsl_induction_no_llm"
    # and OVERWRITES with the proposer's model name only once an LLM outcome comes back. The first
    # draft of this counter keyed on `skipped`/`engine_source` instead and reported llm=0 on a cell
    # that had in fact issued 11 real completions (35533 predicted tokens) -- `skipped:
    # "proposer_failed"` means the LLM ran and its output was unusable, NOT that it never ran.
    # `llm.responses` is the independent, mechanism-level cross-check on this number.
    row["induction_attempts_llm_reached"] = sum(
        1 for a in atts if str(a.get("model_specs")) != "offline_dsl_induction_no_llm"
    )
    row["induction_model_specs"] = dict(
        collections.Counter(str(a.get("model_specs")) for a in atts)
    )

    # ---- observe-channel witness: do graph nodes carry previous_frame? ---------------------
    nodes = list(ex.graph.values())
    row["nodes_total"] = len(nodes)
    row["nodes_with_frame"] = sum(1 for n in nodes if n.get("frame") is not None)
    row["nodes_with_previous_frame"] = sum(1 for n in nodes if n.get("previous_frame") is not None)

    # ---- shipped-lever fire counters ------------------------------------------------------
    try:
        fd = ex.frontier_discipline_diagnostics()
    except Exception as exc:
        fd = {"error": f"{type(exc).__name__}:{exc}"}
    try:
        hud = ex.hud_mask_diagnostics()
    except Exception as exc:
        hud = {"error": f"{type(exc).__name__}:{exc}"}
    row["frontier_diagnostics"] = fd
    row["hud_diagnostics"] = hud
    # EXPLICIT PER-LEVER FIRE COUNTERS, hoisted to the top level so a null cannot be reported
    # without someone having to look at whether the lever did anything. A lever whose fire counter
    # is zero contributes NO evidence in that cell, positive or negative.
    #
    # Lever 1 (frontier tier trio) fires when the tier machinery actually advances a tier. The
    # click-vocab gate additionally requires the game to have OFFERED a click at least once
    # (tier_click_vocab_seen) -- a zero-click game cannot exercise it, which is a property of the
    # game, not a null result.
    row["lever1_frontier_fire"] = {
        "tier_advances": (fd or {}).get("tier_advances"),
        "tier_deferral_fallbacks": (fd or {}).get("tier_deferral_fallbacks"),
        "tier_click_vocab_seen": (fd or {}).get("tier_click_vocab_seen"),
        "tier_active_effective": (fd or {}).get("tier_active_effective"),
        "active_tier": (fd or {}).get("active_tier"),
    }
    row["lever1_fired"] = bool((fd or {}).get("tier_advances"))
    # Lever 2 (HUD edge-bar trio) fires only when the REPAIRED detector resolves a mask that
    # DIFFERS from the already-shipped auto_hud_mask -- comparing DIGESTS, not cell counts (the
    # 2026-07-25 gate had exactly that defect: masks compared by count, so a same-size different
    # mask read as "no change").
    row["lever2_hud_fire"] = {
        "hud_mask_resolved": (hud or {}).get("hud_mask_resolved"),
        "hud_mask_source": (hud or {}).get("hud_mask_source"),
        "hud_mask_cell_count": (hud or {}).get("hud_mask_cell_count"),
        "hud_mask_digest": (hud or {}).get("hud_mask_digest"),
        "hud_shipped_mask_cell_count": (hud or {}).get("hud_shipped_mask_cell_count"),
        "hud_shipped_mask_digest": (hud or {}).get("hud_shipped_mask_digest"),
        "collapse_guard_refusals": (hud or {}).get("collapse_guard_refusals"),
        "node_inflation_vs_unique_frames": (hud or {}).get("node_inflation_vs_unique_frames"),
        "stage2": (hud or {}).get("stage2"),
    }
    # ALSO emit the SAME fields FLAT at row top level, from ONE projection. Until 2026-07-26 this
    # harness recorded them nested only, so every consumer written against exp5836's flat row schema
    # read `None` for `hud_mask_resolved` / `hud_mask_cell_count` / `hud_mask_source` on all 805
    # recorded scored-path rows -- indistinguishable from "the detector resolved nothing". See
    # `hud_row_fields` for the measurement. `lever2_fired` is set HERE (not separately) so the row's
    # stamp and its flat fields can never come from two different reads of the diagnostics.
    row.update(hud_row_fields(hud))
    # Lever 3 (REQ-ARC-WMTE-5970, the NAV-side hazard move-pruner). This lever needs the most
    # careful fire accounting of the three, because its expected outcome IS zero prunes and there
    # are four structurally different ways to get there. `lever3_verdict` names which one happened,
    # so a cell cannot be read as "the lever does not help" when it never fired.
    try:
        hz = ex.hazard_move_pruner_diagnostics()
    except Exception as exc:
        hz = {"error": f"{type(exc).__name__}:{exc}"}
    row["hazard_diagnostics"] = hz
    row["lever3_hazard_fire"] = {
        "enabled": (hz or {}).get("enabled"),
        "observe_calls": (hz or {}).get("observe_calls"),
        "observed_nav_transitions": (hz or {}).get("observed_nav_transitions"),
        "clicks_skipped": (hz or {}).get("clicks_skipped"),
        "antecedent_from_last_grid": (hz or {}).get("antecedent_from_last_grid"),
        "n_deaths": (hz or {}).get("n_deaths"),
        "model_fitted": (hz or {}).get("model_fitted"),
        "lethal_mode": (hz or {}).get("lethal_mode"),
        "rows_pruned": (hz or {}).get("rows_pruned"),
        "all_pruned_nodes": (hz or {}).get("all_pruned_nodes"),
        "observe_errors": (hz or {}).get("observe_errors"),
        "prune_errors": (hz or {}).get("prune_errors"),
    }
    row["lever3_verdict"] = _hazard_verdict(hz, row)
    row["lever3_fired"] = row["lever3_verdict"] in ("FIRED_AND_PRUNED", "FIRED_NO_PRUNE")
    row["lever3_interpretable"] = row["lever3_verdict"] not in (
        "UNINTERPRETABLE_NO_OBSERVE",
        "UNINTERPRETABLE_NO_NAV",
        "UNINTERPRETABLE_NOT_FITTED",
        "ERROR",
    )

    # ---- outcome + the NAV/CLICK action mix (the TASK-2 target sizing) ---------------------
    recs = []
    for fr in r.get("frame_sequence") or []:
        mv = fr.get("move") or {}
        k = mv.get("kind")
        if k in (None, "RESET"):
            continue
        data = mv.get("data") or {}
        recs.append(
            {
                "key": _action_key(mv),
                "is_click": bool(isinstance(data, dict) and "x" in data),
            }
        )
    # ---- THE QUANTITIES THE SCORER ACTUALLY DIFFERENCES --------------------------------------
    # `actions_to_first_levelup` alone cannot attribute a score delta to a level, and cannot show
    # the post-solve tail at all. The authoritative scorer charges each COMPLETED level
    # `actions_at_level - prev_actions` (arc_agi/scorecard.py:479) -- i.e. it differences a vector
    # of cumulative checkpoints. `run_game` computes that vector but returns only its first element,
    # so it is reconstructed here from `per_level` (whose `agent_actions` ARE those differences, in
    # order) and persisted. With it, any score claim in this sweep is recomputable from the row
    # without re-running the cell, and the inter-level-up GAP distribution -- the quantity that
    # decides whether a given grace value can cost a level -- is computable per row.
    per_level = list(r.get("per_level") or [])
    row["per_level"] = per_level
    _cum, _lua = 0, []
    for _pl in per_level:
        _cum += int(_pl.get("agent_actions") or 0)
        if _pl.get("completed"):
            _lua.append(_cum)
    row["level_up_actions"] = _lua
    row["inter_levelup_gaps"] = [_lua[i] - _lua[i - 1] for i in range(1, len(_lua))]
    # THE TAIL: actions spent after the LAST level-up. This is the quantity early-stop cuts, and
    # (per the resolved charge model) the quantity that costs zero score -- so it is the mechanism's
    # entire benefit, and it must be measured, not inferred from `actions`.
    row["actions_after_last_levelup"] = (
        int(r["actions"]) - _lua[-1] if _lua else (int(r["actions"]) if r.get("actions") else 0)
    )
    row["reached_any_level"] = bool(_lua)
    # DID THE MECHANISM FIRE? Read off the explorer, not inferred from the action count. A treatment
    # arm with zero fires anywhere in the corpus contributes no evidence and must be stamped so.
    row["early_stopped"] = bool(getattr(ex, "early_stopped", False))
    # RESETS. The live gateway charges a RESET one action (arc_agi/scorecard.py:701-704 via
    # update_scorecard); this offline harness charges it zero (arc_leaderboard_eval.py:308-313).
    # Our agent uses RESET-and-replay as a navigation fallback, so offline action counts -- and
    # therefore offline efficiency -- are OPTIMISTIC by exactly this many actions per level. Counted
    # per row so the live-vs-offline gap is visible rather than assumed away. It also converts the
    # grace window's unit: the window counts len(frames) (loop iterations, RESET INCLUDED), so a
    # grace of G frames buys only G * actions/frames actions.
    _n_resets = sum(
        1 for fr in (r.get("frame_sequence") or []) if (fr.get("move") or {}).get("kind") == "RESET"
    )
    row["n_resets"] = _n_resets
    row["n_frames"] = len(r.get("frame_sequence") or [])
    hist = collections.Counter(rc["key"] for rc in recs)
    top = hist.most_common(5)
    n_click = sum(1 for rc in recs if rc["is_click"])
    row.update(
        levels=int(r["levels"]),
        reached=int(r["reached"]),
        actions=int(r["actions"]),
        actions_to_first_levelup=r.get("actions_to_first_levelup"),
        efficiency=r.get("efficiency"),
        states_expanded=len(ex.graph),
        errors=int((fd or {}).get("click_pixel_errors") or 0)
        + int((fd or {}).get("click_pixel_generation_errors") or 0),
        n_actions_counted=len(recs),
        n_click_actions=n_click,
        n_nav_actions=len(recs) - n_click,
        nav_fraction=(round((len(recs) - n_click) / len(recs), 4) if recs else None),
        top_action=(top[0][0] if top else None),
        top_action_count=(int(top[0][1]) if top else 0),
        top_action_is_click=bool(top and "@" in str(top[0][0])),
        top5=[{"action": k, "count": int(v)} for k, v in top],
    )

    if prev_disable is None:
        os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    else:
        os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = prev_disable
    return row


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="lp85")
    ap.add_argument("--seeds", default="20260724")
    # 400 = the SHIPPED agent's own per-game MAX_ACTIONS, i.e. the condition the current submission
    # runs under. It is a self-imposed loop guard and an intended override point, NOT an eval-imposed
    # bound (the eval's bound is <=12h wall clock across all games). Budget choice CHANGES
    # CONCLUSIONS -- see the module docstring for the measured reversal -- so it is explicit here.
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="control arm: same policy, induction disabled (the 2026-07-25 condition)",
    )
    ap.add_argument(
        "--both",
        action="store_true",
        help="run BOTH llm-off and llm-on per cell, round-robin by arm",
    )
    ap.add_argument(
        "--no-spawn",
        action="store_true",
        help="forbid the proposer from launching a server; the caller must have started "
        "one already (start_gen.sh). Prevents the measured server-storm.",
    )
    ap.add_argument(
        "--arms",
        default="",
        help="comma-separated ARMS keys (S, S_minus_frontier, S_minus_hud, "
        "S_minus_both, S_plus_hazard). Empty = the bare default policy, no flag "
        "kwargs pinned.",
    )
    ap.add_argument(
        "--games-all",
        action="store_true",
        help="use all 25 public survey games (overrides --games)",
    )
    a = ap.parse_args(argv)
    if a.games_all:
        a.games = ",".join(GAMES_25)
    games = [g for g in a.games.split(",") if g]
    seeds = [int(s) for s in a.seeds.split(",") if s]
    arms = [x for x in a.arms.split(",") if x] or [""]
    for x in arms:
        if x and x not in ARMS:
            raise SystemExit(f"unknown arm {x!r}; known: {sorted(ARMS)}")

    assert os.environ.get("CARNOT_ARC_DISABLE_INDUCTION") is None, (
        "CARNOT_ARC_DISABLE_INDUCTION must be unset at harness start"
    )

    # Record (and shout about) any drift between this file's pinned SHIPPED dict and the agent's own
    # SUBMITTED_* globals. Pinning keeps an ARM stable; only this check tells you whether the arm
    # named "S" is still the LIVE configuration.
    parity = assert_shipped_dict_matches_module_globals()
    if parity["pinned_vs_live_drift"]:
        print(
            f"[WARNING] pinned SHIPPED dict differs from the live SUBMITTED_* globals: "
            f"{parity['pinned_vs_live_drift']} -- arm 'S' is NOT the live config; "
            f"update SHIPPED_2026_07_25 before treating it as the control.",
            flush=True,
        )
    if a.budget != 400:
        print(
            f"[NOTE] budget={a.budget}; the SHIPPED agent's own cap is MAX_ACTIONS=400. This run "
            f"therefore measures the LEVERS (a budget the agent could be given -- 400 is a "
            f"self-imposed loop guard and a documented override point), NOT the current "
            f"submission's configuration. Report which one you measured.",
            flush=True,
        )

    proposer = build_proposer(a.port)
    t_srv = time.time()
    server_ok = proposer._inner._ensure_server()
    srv_s = round(time.time() - t_srv, 2)
    print(
        f"[generator] ensure_server={server_ok} in {srv_s}s port={a.port} "
        f"gpu_pin={os.environ.get('CARNOT_ARC_GENERATOR_CUDA_GPU')}",
        flush=True,
    )
    if not server_ok:
        print("[generator] FAILED -- refusing to run an LLM-on measurement without a live server")
        return 2
    if a.no_spawn:
        proposer.forbid_spawn()
        print("[generator] spawn FORBIDDEN -- health-check only from here on")
    print(f"[generator] llama-server processes now: {_llama_server_count()}", flush=True)

    modes = [False, True] if a.both else ([False] if a.no_llm else [True])
    rows, t0 = [], time.time()
    for g in games:
        for s in seeds:
            # ROUND-ROBIN BY ARM inside each (game, seed) cell -- never row order, so a run cut
            # short can never leave one arm with more measured cells than another.
            for arm_key in arms:
                for llm in modes:
                    row = run_cell(
                        g,
                        s,
                        budget=a.budget,
                        proposer=proposer,
                        llm=llm,
                        extra_kwargs=(dict(ARMS[arm_key]) if arm_key else None),
                        arm=f"{arm_key or 'E3_default'}_llm{'on' if llm else 'off'}",
                    )
                    rows.append(row)
                    L = row.get("llm") or {}
                    print(
                        f"{row['arm']:26} {g:5} s{s} lv={row.get('levels')} act={row.get('actions')} "
                        f"st={row.get('states_expanded')} wall={row.get('wall_s')}s "
                        f"ind={row.get('induction_attempts')}(llm={row.get('induction_attempts_llm_reached')}) "
                        f"resp={L.get('responses')} gen={L.get('generate_calls')} "
                        f"txt={L.get('complete_text_calls')} "
                        f"tok_out={L.get('tokens_predicted')} tok_in={L.get('tokens_prompt')} "
                        f"llm_wall={L.get('llm_wall_s')}s "
                        f"prevframe={row.get('nodes_with_previous_frame')}/{row.get('nodes_total')} "
                        f"nav={row.get('nav_fraction')} err={row.get('errors')} "
                        f"L1={row.get('lever1_fired')} L2={row.get('lever2_fired')} "
                        f"L3={row.get('lever3_verdict')} "
                        f"genok={row.get('generator_healthy_before')}->{row.get('generator_healthy_after')} "
                        f"srv={row.get('llama_servers_before')}->{row.get('llama_servers_after')} "
                        f"VALID={row.get('llm_on_row_valid')}",
                        flush=True,
                    )
                    Path(a.out).write_text(
                        json.dumps(
                            {
                                "rows": rows,
                                "server_spawn_s": srv_s,
                                "port": a.port,
                                "shipped_config": SHIPPED_2026_07_25,
                                "arms_used": arms,
                                "budget": a.budget,
                                "scored_agent_max_actions": 400,
                                "budget_matches_scored_cap": a.budget == 400,
                                "budget_semantics": (
                                    "400 is the SHIPPED agent's self-imposed per-game MAX_ACTIONS "
                                    "loop guard (an intended override point), NOT an eval-imposed "
                                    "bound -- the eval's bound is <=12h wall clock across all "
                                    "games. A budget-400 row describes the CURRENT SUBMISSION's "
                                    "condition; a budget-2000 row describes the levers themselves. "
                                    "Lever orderings differ between the two."
                                ),
                                "flag_parity_vs_live_globals": parity,
                                "elapsed_s": round(time.time() - t0, 1),
                            },
                            indent=1,
                        )
                    )
    print(f"TOTAL {round(time.time() - t0, 1)}s n={len(rows)}")
    verdicts = collections.Counter(r.get("lever3_verdict") for r in rows)
    print(f"lever3 (hazard pruner) verdicts: {dict(verdicts)}")
    if verdicts.get("UNINTERPRETABLE_NO_OBSERVE"):
        print(
            "[WARNING] some hazard-pruner cells never observed a transition -- that is a WIRING "
            "DEFECT (the exp5836 dead-channel class), not a null. Fix before reporting."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
