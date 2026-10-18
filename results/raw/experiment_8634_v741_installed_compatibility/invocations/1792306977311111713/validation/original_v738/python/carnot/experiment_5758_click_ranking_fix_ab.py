"""REQ-ARC-FCP-5758 -- Live A/B for the click-ranking selection fix ``small_object_first``.

MOTIVATION (the verified gap this follows up on).
exp5757 (REQ-ARC-FCP-5757, candidate-coverage attribution) partitioned Carnot's ARC
score gap on 9 stalled games and found: of 92 real winning-path actions, 98.9% are
already generated AND recognized as frame-changing by the live candidate pipeline. The
ONLY residual single-action gap is a SELECTION/RANKING miss -- 6 object-clicks (r11l x1,
su15 x4, cn04 near-miss) that ARE in the candidate set and ARE frame-changing but rank
>= 12 of ~27-34 candidates (bucket "c"). This experiment attacks that ranking gap.

ROOT-CAUSE DIAGNOSIS (this session, before designing the fix -- see diagnosed_ranking_
failure_mode).
Reproducing the exact gap frames (exp5757's replay mechanism) and dumping per-object
salience features shows the low-ranked winning clicks are consistently VERY SMALL objects
(r11l's rank-22 winner is a single pixel area=1; su15's repeatedly-clicked winner is a
single pixel area=1 color=3; r11l's other winners are area 4 and 12). The shipped default
orders object clicks by VISUAL SALIENCE = ``area * (1 + 1/(1 + global_color_pixels))``,
which is AREA-DOMINATED: a 240-pixel decorative region (salience ~240) always outranks a
1-pixel interactive target (salience ~1), so the tiny winners sink to ranks 13-22 and are
rarely tried within budget. HONEST NUANCE: for the single-pixel-field cases the winner is
ALSO buried under many EQUAL-size, rarer-coloured pixels, so no monotonic-in-area formula
can surface it -- part of this gap is a discriminating-SIGNAL absence (a Missing-Verifier
Gap), not a formula bug.

THE FIX (one, minimal, opt-in, gated OFF by default -- CARNOT_ARC_SMALL_OBJECT_FIRST=1).
``arc_graph_explore._small_object_first_click_points``: reorder the SAME object-click set
into two bands -- a SMALL band (area <= 8) ordered by colour-rarity (rarest first), tried
FIRST, then all remaining objects in the PROVEN salience order. Pure reordering, no click
added/dropped, byte-identical when the flag is off. GENUINELY DIFFERENT from the two
already-NULL reorders: CARNOT_ARC_TIER_SCHEDULE (front-loads MEDIUM-width objects, which
EXCLUDE the 1x1 winners; results/proto_tier_ab.json TIER_NULL_no_win) and the learned
DiscriminativeVerifier candidate_router (exp4556, tested on COLOR VARIANTS first-contact,
verifier_router_no_value_added).

THE DECISIVE TEST.
Matched-budget (200 actions), same 11-game roster as exp5729/exp5732/exp5740, per-game
fixed seed so each arm's game-X starts from an identical RNG state -> any delta isolates
the reorder. The two arms are byte-identical E3AgentPolicy(game) constructions (NO action_
prior -- the true shipped live default); the ONLY difference is the env flag read inside
rich_action_candidates. Primary question: does treatment bank an ADDITIONAL level on r11l
and/or su15 within budget that baseline does not? Secondary: any level gain roster-wide
without a states_expanded (search-cost) blow-up.

DISCIPLINE. inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm
(CARNOT_ARC_DISABLE_INDUCTION=1 -> no GGUF/LLM). verifier_is_oracle=False (the win oracle
is the level counter; the reorder is a perceptual heuristic, never the win check).
solve_provenance=development_proxy (a live-path component A/B on the dev twin; NO new level
banked, offline_reproduced deliberately not claimed). The agent NEVER flips the shipped
default -- recommendation is operator-only.

Spec refs: REQ-ARC-FCP-5758, SCENARIO-ARC-FCP-5758-ELEVEN-GAME-CAPABILITY-AB,
SCENARIO-ARC-FCP-5758-R11L-SU15-DECISIVE, SCENARIO-ARC-FCP-5758-RANK-SHIFT-OFFLINE.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (PYTHON_ROOT, REPO_ROOT, SCRIPTS_ROOT):
    if str(_p) not in sys.path:  # pragma: no cover - direct script guard
        sys.path.insert(0, str(_p))

# No-LLM search substrate: guaranteed before any policy import/construction.
os.environ.setdefault("CARNOT_ARC_DISABLE_INDUCTION", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5758_click_ranking_fix_ab"
RESULT_RELATIVE_PATH = "results/experiment_5758_click_ranking_fix_ab.json"
SCHEMA = "carnot.exp5758.click_ranking_fix_ab.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5758
DEFAULT_BUDGET = 200
BASELINE_ARM = "baseline"
FLAG = "CARNOT_ARC_SMALL_OBJECT_FIRST"
STATES_REGRESSION_REL = 0.20  # exp5729 discipline: >20% states growth is a material regression

# Same 11-game roster as exp5729/exp5732/exp5740 for an apples-to-apples comparison.
DEFAULT_ROSTER = (
    "cd82",
    "cn04",
    "lp85",
    "ls20",
    "m0r0",
    "r11l",
    "sk48",
    "sp80",
    "su15",
    "tu93",
    "wa30",
)
# The two games exp5757 localized the bucket-"c" click-ranking gap to.
DECISIVE_GAMES = ("r11l", "su15")

# (arm_name, flag_value). flag_value is written to os.environ[FLAG] for that arm.
ARMS: tuple[tuple[str, str | None], ...] = (
    ("baseline", None),
    ("small_object_first", "1"),
)

DIAGNOSED_RANKING_FAILURE_MODE = {
    "summary": (
        "AREA-DOMINANCE buries tiny interactive targets. The low-ranked winning object-clicks on "
        "r11l/su15 are consistently VERY SMALL objects (r11l rank-22 winner area=1; su15's "
        "repeatedly-clicked winner area=1 color=3; r11l others area 4 and 12). The shipped default "
        "orders clicks by area*(1+1/(1+global_color_pixels)), which is area-dominated, so a 1-pixel "
        "target (salience ~1) always loses to large decorative regions (salience up to ~240) -- the "
        "winners sink to ranks 13-22 of ~27-34 and are rarely tried within budget."
    ),
    "honest_nuance_not_a_tidy_story": (
        "It is NOT purely 'small button lost to big decoration'. For the single-pixel-field cases the "
        "winner is ALSO outranked by MANY equal-size, rarer-coloured single pixels (r11l idx0: 11 "
        "larger + 12 equal-area rarer pixels rank above it). No monotonic-in-area formula can surface "
        "a 1-pixel winner above a 240-pixel region, and colour-rarity does not identify THE winning "
        "pixel among the single-pixel field. So part of this gap is a discriminating-SIGNAL ABSENCE "
        "(a Missing-Verifier Gap), not a ranking-formula bug -- stated plainly per the task's "
        "instruction not to force a tidy story onto noisy data."
    ),
    "per_gap_action_features": {
        "r11l": [
            {
                "idx": 0,
                "rank": 22,
                "n": 34,
                "win_area": 1,
                "win_color": 15,
                "note": "single pixel; 11 larger + 12 equal-area rarer pixels ranked above",
            },
            {
                "idx": 1,
                "rank": 6,
                "n": 26,
                "win_area": 12,
                "win_color": 3,
                "note": "already handled (rank<12)",
            },
            {
                "idx": 2,
                "rank": 8,
                "n": 26,
                "win_area": 4,
                "win_color": 1,
                "note": "already handled (rank<12)",
            },
        ],
        "su15": [
            {
                "idx": 0,
                "rank": 7,
                "n": 28,
                "win_area": 5,
                "win_color": 0,
                "note": "already handled",
            },
            {
                "idx": 1,
                "rank": 22,
                "n": 27,
                "win_area": 1,
                "win_color": 3,
                "note": "single-pixel winner, bucket c",
            },
            {
                "idx": 2,
                "rank": 19,
                "n": 27,
                "win_area": 1,
                "win_color": 3,
                "note": "same pixel; bucket c",
            },
            {
                "idx": 3,
                "rank": 16,
                "n": 27,
                "win_area": 1,
                "win_color": 3,
                "note": "same pixel; bucket c",
            },
            {
                "idx": 4,
                "rank": 13,
                "n": 27,
                "win_area": 1,
                "win_color": 3,
                "note": "same pixel; bucket c",
            },
            {
                "idx": 5,
                "rank": 10,
                "n": 27,
                "win_area": 1,
                "win_color": 3,
                "note": "climbs as distractor pixels are consumed",
            },
            {
                "idx": 6,
                "rank": 7,
                "n": 27,
                "win_area": 1,
                "win_color": 3,
                "note": "already handled by now",
            },
        ],
    },
}

FIX_DESCRIPTION = {
    "flag": FLAG,
    "default": "off (byte-identical to the shipped salience order)",
    "function": "arc_graph_explore._small_object_first_click_points",
    "mechanism": (
        "Reorder the SAME object-click candidate set into two bands: (1) a SMALL band (object area "
        "<= 8) ordered by colour-rarity (rarest colour first, larger-within-small breaks ties), tried "
        "FIRST; (2) all remaining objects in the PROVEN area*rarity salience order. No click is added "
        "or dropped; the recorded trajectory is still a valid deterministic replay."
    ),
    "small_area_max": 8,
    "why_area_8": (
        "single-digit-pixel = 'tiny interactive target'; covers 3 of the 4 observed winner sizes "
        "(1,4,5); a principled cutoff, not tuned to hit a gate."
    ),
    "genuinely_different_from": (
        "CARNOT_ARC_TIER_SCHEDULE (front-loads MEDIUM-width w,h in [2,32] objects -- EXCLUDES the "
        "1x1 winners here; already A/B-NULL in results/proto_tier_ab.json) and the exp4556 learned "
        "DiscriminativeVerifier candidate_router (tested on colour-variant first-contact, no value)."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "prior_work_extended",
    "diagnosed_ranking_failure_mode",
    "fix_description",
    "levels_gained_total_by_arm",
    "r11l_su15_specifically_fixed",
    "safety_regression_check",
    "any_config_beats_baseline_levels",
    "recommendation",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; a level win, a search-behaviour change with no level gain, and "
        "a total no-op are distinct real outcomes -- the reorder earns a live-default flip only if it "
        "raises banked levels on the same roster+budget WITHOUT a states_expanded blow-up"
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- CARNOT_ARC_DISABLE_"
        "INDUCTION=1 guarantees no GGUF/LLM load; the reorder affects only the candidate ordering the "
        "search expands, isolated from tier-3 induction"
    },
    "verifier_is_oracle": {
        "principle": "False -- the win oracle is the env level counter (frame.levels_completed); the "
        "small-object-first reorder is a perceptual heuristic over object candidates, never the check "
        "that defines a win (no CIRCULAR_MOAT_OVERCLAIM)"
    },
    "solve_provenance": {
        "principle": "development_proxy -- an offline-arcade live-path candidate-ordering A/B on the "
        "dev twin; NOT a self-discovery solve, NO new level banked (any lp85 L1 is a pre-existing "
        "registry solve reached incidentally), so offline_reproduced is deliberately NOT claimed"
    },
    "random_seed": {
        "principle": "determinism precondition; per-game seed = RANDOM_SEED + game_index so "
        "baseline-game-X and treatment-game-X start from an identical RNG state -- any delta isolates "
        "the reorder, not RNG drift"
    },
    "reproducibility_checksum": {
        "principle": "content hash over the arms x roster capability rows; "
        "catches silent corpus/harness drift on replay"
    },
    "duration_s": {
        "principle": "real wall-clock of the search A/B; the no-LLM substrate floor is 0.01s "
        "and this runs 2 arms x 11 games sequentially, so a multi-hundred-second total is expected"
    },
    "prior_work_extended": {
        "principle": "Failed-Experiment Rerun Discipline -- names REQ-ARC-FCP-5757 "
        "(the motivating gap) + exp4556 (the learned-router null) + the tier-schedule null by "
        "id+verdict, and states precisely what is different (a NEW small-object-first reorder, tested "
        "on the base gap games with real level/states metrics, not colour-variant first-contact)"
    },
    "diagnosed_ranking_failure_mode": {
        "principle": "the step-1 root-cause finding stated honestly "
        "even where inconclusive -- area-dominance buries tiny targets AND the single-pixel-field "
        "winners have no clean discriminating feature (a Missing-Verifier Gap)"
    },
    "fix_description": {
        "principle": "the ONE minimal opt-in fix, its flag/default/mechanism/threshold "
        "and why it is genuinely different from the two already-NULL reorders"
    },
    "levels_gained_total_by_arm": {
        "principle": "the raw capability answer: total banked levels per "
        "arm across the roster + per-game breakdown so a third party re-derives every headline number"
    },
    "r11l_su15_specifically_fixed": {
        "principle": "THE decisive test -- do the two games exp5757 "
        "localized the click-ranking gap to actually complete an ADDITIONAL level under the reorder "
        "within budget that baseline does not; a bool + per-game detail"
    },
    "safety_regression_check": {
        "principle": "exp5729 discipline -- a reorder that banks a level while "
        "materially inflating states_expanded (>20% vs baseline) is flagged; a level win is never "
        "reported without its search cost, and a reorder that REGRESSES search elsewhere is surfaced"
    },
    "any_config_beats_baseline_levels": {
        "principle": "the load-bearing capability boolean: True iff "
        "the reorder banks strictly more levels than baseline on the same roster+budget"
    },
    "recommendation": {
        "principle": "reports whether to flip the live default (SUBMITTED_* / the flag); "
        "the agent never self-authorizes flipping the live-stack default -- operator-only"
    },
    "rank_shift_offline": {
        "principle": "mechanistic evidence: the winning-click rank under default vs "
        "the reorder on the exact r11l/su15 gap frames -- shows whether the fix even moves the target "
        "up, independent of whether that converts to a level"
    },
    "levels_gained_headroom_present": {
        "principle": "FALSE_NEGATIVE_RISK discipline -- a no-delta "
        "result is interpretable only if some arm banks a nonzero level somewhere, else the roster had "
        "no level headroom for any reorder at this budget"
    },
    "trajectories_diverge_per_game": {
        "principle": "did the reorder change the search trajectory at all "
        "per game -- distinguishes 'inert' from 'changed search but no level gain'"
    },
    "per_arm_game_rows": {
        "principle": "full per-game rows (levels, states_expanded, actions, "
        "trajectory sha) so the capability comparison is independently auditable"
    },
    "preconditions_checked": {
        "principle": "records the resources verified (offline arcade builds an "
        "env, E3AgentPolicy + the reorder helper import) before the sweep, per Pre-Launch Preconditions"
    },
}

PRIOR_WORK_EXTENDED = {
    "motivating_gap": {
        "experiment_id": "REQ-ARC-FCP-5757 (experiment_5757_candidate_coverage_attribution)",
        "verdict": "complete_attribution_gate_no_threshold_crossed_dominant_gap_c_n92_games9",
        "role": (
            "THE MOTIVATION. Localized the ONLY residual single-action gap to SELECTION/RANKING of "
            "object clicks (bucket c: 6 action-6 clicks in-set + frame-changing but rank>=12; r11l x1, "
            "su15 x4). This experiment attacks exactly that gap on exactly those games."
        ),
    },
    "prior_reorder_nulls": [
        {
            "experiment_id": "exp4556 (experiment_4556_verifier_router_generic_transfer)",
            "prior_verdict": "complete: verifier_router_no_value_added_honest_null_gap_sharpened",
            "diagnosed_root_cause": (
                "The learned DiscriminativeVerifier candidate_router added no transfer value (delta 0.0 "
                "vs 0.04 baseline; random-router control 0.08 > verifier, so the control FAILED)."
            ),
            "what_is_different_here": (
                "exp4556 tested a LEARNED cross-game router on COLOUR-VARIANT first-contact "
                "(generic_solver_offline_variant_env, variant_signature=game~color01) -- NOT the base "
                "r11l/su15 games, and NOT a hand-designed perceptual reorder. THIS is a cheap, "
                "hand-designed small-object-first reorder measured on the BASE gap games with real "
                "level/states metrics. Different roster/corpus + different mechanism -> a legitimate "
                "new test, not a rerun of the learned-router null."
            ),
            "retire_if_same_verdict": False,
        },
        {
            "experiment_id": "proto_tier_ab (CARNOT_ARC_TIER_SCHEDULE grafted just-explore 5-tier salience)",
            "prior_verdict": "TIER_NULL_no_win (tier never beats flat at budget 1000 AND 4000, zero regression)",
            "diagnosed_root_cause": (
                "just-explore's 5-tier schedule front-loads MEDIUM-width (w,h in [2,32]) salient objects. "
                "It EXCLUDES the 1x1 winners this gap is about (they are not medium-width), so it could "
                "never have surfaced them -- candidate ORDERING via that schedule is not the edge."
            ),
            "what_is_different_here": (
                "small_object_first targets the OPPOSITE size band -- the single-pixel / area<=8 objects "
                "the tier schedule excludes -- so it is a genuinely different reorder, not a re-run of "
                "the tier-schedule null."
            ),
            "retire_if_same_verdict": False,
        },
    ],
    "harness_precedent": {
        "experiment_id": "exp5740 (experiment_5740_object_history_salience_11game_ab) / exp5729",
        "role": (
            "THE HARNESS PRECEDENT. Reuses its skeleton: drive E3AgentPolicy/StepwiseExplorer directly "
            "via arc_leaderboard_eval.run_game, 11-game roster, budget 200, per-game fixed seed, "
            "states_expanded = len(policy.explorer.graph), the >20% states-regression discipline."
        ),
    },
    "retire_if_same_verdict": (
        "If the reorder banks NO additional level on r11l/su15 (or roster-wide) at this budget AND the "
        "offline rank-shift is a wash (the winner not reliably surfaced), then a hand-designed "
        "perceptual click reorder is not the live-path lever for this gap -- the residual is a "
        "discriminating-SIGNAL absence (Missing-Verifier Gap). Recommend the operator keep the flag "
        "OFF and log the gap for a learned/goal-conditioned click discriminator; do NOT re-propose "
        "another static perceptual click-reorder without a NEW signal."
    ),
}


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Offline rank-shift diagnostic (mechanistic evidence): reproduce the exact
# r11l/su15 gap frames (exp5757's adapter-replay) and recompute each winning
# click's rank under default vs the reorder. Independent of the live A/B.
# --------------------------------------------------------------------------- #
def _win_action_from_step(step: tuple[str, Optional[dict]]) -> tuple[int, Optional[dict]]:
    name, data = step
    m = re.search(r"(\d+)$", str(name))
    return (int(m.group(1)) if m else 0, data)


class _RecordingEnv:
    def __init__(self, env: Any) -> None:
        self._env = env
        self.calls: list[tuple[str, Optional[dict]]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, *a: Any, **k: Any) -> Any:
        return self._env.reset(*a, **k)

    def step(self, action: Any, data: Any = None, **k: Any) -> Any:
        self.calls.append((getattr(action, "name", str(action)), dict(data) if data else None))
        return self._env.step(action, data=data, **k)


def _winning_click_ranks(game: str) -> list[dict[str, Any]]:
    """Rank of each winning object-click in rich_action_candidates at the gap frame
    (honours the CARNOT_ARC_SMALL_OBJECT_FIRST env var set by the caller)."""
    from carnot.agentic import arc_game_adapters as adapters
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import action_key
    from carnot.agentic.arc_graph_explore import rich_action_candidates
    import arc_loop_solve as loop

    res = loop.solve_adaptered(game, 1)
    labels = res.get("solution_labels") or []
    ad = adapters.get_adapter(game)
    arc = kit.offline_arcade()
    env = _RecordingEnv(arc.make(game, scorecard_id=arc.open_scorecard()))
    f = env.reset()
    if ad.warmup_label is not None:
        f = ad.apply(env, ad.warmup_label, f)
    steps: list[dict[str, Any]] = []
    for i, lbl in enumerate(labels):
        if f is None:
            break
        cands = rich_action_candidates(f)
        cand_keys = [c.key for c in cands]
        n0 = len(env.calls)
        f = ad.apply(env, lbl, f)
        if len(env.calls) == n0:
            continue
        action_id, win_data = _win_action_from_step(env.calls[n0])
        if action_id != 6 or not win_data:
            continue
        wk = action_key(action_id, win_data)
        rank = cand_keys.index(wk) if wk in cand_keys else None
        steps.append({"idx": i, "rank": rank, "n_candidates": len(cands)})
    return steps


def rank_shift_offline() -> JsonDict:
    out: JsonDict = {
        "note": (
            "Winning object-click rank at the exact gap frame under default vs the reorder. "
            "low_rank = rank>=12 (exp5757 bucket-c threshold). This measures whether the fix even "
            "moves the target up, independent of whether it converts to a level."
        )
    }
    for game in DECISIVE_GAMES:
        rec: JsonDict = {}
        for arm, flag in (("default", None), ("small_object_first", "1")):
            prev = os.environ.get(FLAG)
            if flag is None:
                os.environ.pop(FLAG, None)
            else:
                os.environ[FLAG] = flag
            try:
                steps = _winning_click_ranks(game)
            except Exception as exc:  # a diagnostic crash is a datum, not a fatal
                rec[arm] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                if prev is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = prev
                continue
            if prev is None:
                os.environ.pop(FLAG, None)
            else:
                os.environ[FLAG] = prev
            low = sum(1 for s in steps if s["rank"] is not None and s["rank"] >= 12)
            rec[arm] = {"per_step": steps, "n_low_rank_ge12": low}
        out[game] = rec
    # summarise whether the fix reduced the low-rank count
    reduced = 0
    worsened = 0
    for game in DECISIVE_GAMES:
        g = out.get(game, {})
        d = g.get("default", {}).get("n_low_rank_ge12")
        t = g.get("small_object_first", {}).get("n_low_rank_ge12")
        if d is not None and t is not None:
            if t < d:
                reduced += 1
            elif t > d:
                worsened += 1
    out["summary"] = {
        "games_low_rank_reduced": reduced,
        "games_low_rank_worsened": worsened,
        "interpretation": (
            "0 reduced means the reorder does NOT surface the bucket-c winners: they are buried under a "
            "field of equal-size rarer pixels, so promoting small objects reshuffles that field without "
            "picking THE winning pixel -- a wash, confirming the Missing-Verifier framing."
        ),
    }
    return out


# --------------------------------------------------------------------------- #
# Preconditions.
# --------------------------------------------------------------------------- #
def preconditions() -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        checks["offline_arcade_importable"] = True
        checks["offline_arcade_makes_env"] = False
        try:
            env = arc.make(DEFAULT_ROSTER[0], scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401

        checks["e3_policy_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
    try:
        from carnot.agentic.arc_graph_explore import _small_object_first_click_points  # noqa: F401

        checks["reorder_helper_import"] = True
    except Exception:
        checks["reorder_helper_import"] = False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


# --------------------------------------------------------------------------- #
# Live A/B.
# --------------------------------------------------------------------------- #
def _play_one_game(
    game: str, *, arm_name: str, flag: str | None, budget: int, game_index: int
) -> JsonDict:
    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    if flag is None:
        os.environ.pop(FLAG, None)
    else:
        os.environ[FLAG] = flag
    # Per-game fixed seed so baseline-game-X and treatment-game-X start identically.
    _seed_everything(RANDOM_SEED + game_index)

    t0 = time.time()
    policy = E3AgentPolicy(game)  # NO action_prior -- the true shipped live default
    explorer_present = getattr(policy, "explorer", None) is not None
    row = lb.run_game(game, policy, budget=budget)
    dt = round(time.time() - t0, 3)

    transitions = getattr(policy, "transitions", []) or []
    traj = [
        {"action": int(getattr(t, "action", 0)), "data": getattr(t, "data", None)}
        for t in transitions
    ]
    traj_sha = hashlib.sha256(
        json.dumps(traj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    states_expanded = int(len(policy.explorer.graph)) if explorer_present else 0

    if flag is None:
        os.environ.pop(FLAG, None)  # leave env clean

    return {
        "game": game,
        "arm": arm_name,
        "explorer_present": bool(explorer_present),
        "levels": int(row.get("levels", 0)),
        "reached": int(row.get("reached", 0)),
        "actions": int(row.get("actions", 0)),
        "states_expanded": states_expanded,
        "efficiency": float(row.get("efficiency", 0.0) or 0.0),
        "actions_to_first_levelup": row.get("actions_to_first_levelup"),
        "gap_signature": (row.get("gap") or {}).get("signature") if row.get("gap") else None,
        "trajectory_len": len(traj),
        "trajectory_sha": traj_sha,
        "duration_s": dt,
        "_trajectory": traj,
    }


def run_sweep(
    roster: tuple[str, ...], arms: tuple[tuple[str, str | None], ...], *, budget: int
) -> dict[str, dict[str, JsonDict]]:
    out: dict[str, dict[str, JsonDict]] = {}
    for arm_name, flag in arms:
        per_game: dict[str, JsonDict] = {}
        for game_index, game in enumerate(roster):
            per_game[game] = _play_one_game(
                game, arm_name=arm_name, flag=flag, budget=budget, game_index=game_index
            )
        out[arm_name] = per_game
    return out


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    arms: tuple[tuple[str, str | None], ...] = ARMS,
    budget: int = DEFAULT_BUDGET,
) -> JsonDict:
    started_at = time.time()
    preconds = preconditions()
    miss = _first_precondition_miss(preconds)
    arm_names = [a[0] for a in arms]

    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "verifier_is_oracle": False,
            "solve_provenance": "development_proxy",
            "field_principles": FIELD_PRINCIPLES,
            "roster": list(roster),
            "budget": int(budget),
            "arms_tested": arm_names,
            "baseline_arm": BASELINE_ARM,
            "diagnosed_ranking_failure_mode": DIAGNOSED_RANKING_FAILURE_MODE,
            "fix_description": FIX_DESCRIPTION,
            "rank_shift_offline": {},
            "levels_gained_total_by_arm": {},
            "states_expanded_total_by_arm": {},
            "per_arm_game_rows": {},
            "trajectories_diverge_per_game": {},
            "r11l_su15_specifically_fixed": {
                "fixed": False,
                "detail": f"blocked precondition {miss}",
            },
            "levels_gained_headroom_present": False,
            "any_config_beats_baseline_levels": False,
            "safety_regression_check": {},
            "prior_work_extended": PRIOR_WORK_EXTENDED,
            "recommendation": f"blocked precondition {miss}; sweep not run.",
            "random_seed": RANDOM_SEED,
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
            "reproducibility_checksum": "",
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    # Offline rank-shift mechanistic evidence (cheap; self-documents the fix).
    rank_shift = rank_shift_offline()

    sweep = run_sweep(roster, arms, budget=budget)
    baseline_rows = sweep[BASELINE_ARM]
    baseline_levels_total = sum(r["levels"] for r in baseline_rows.values())
    baseline_states_total = sum(r["states_expanded"] for r in baseline_rows.values())

    # Trajectory divergence per game (before dropping heavy trajectories).
    treatment_arms = [a for a in arm_names if a != BASELINE_ARM]
    trajectories_diverge_per_game: JsonDict = {}
    for arm in treatment_arms:
        per_game = {
            g: bool(sweep[arm][g]["_trajectory"] != baseline_rows[g]["_trajectory"]) for g in roster
        }
        trajectories_diverge_per_game[arm] = {
            "per_game": per_game,
            "n_games_diverged": int(sum(1 for v in per_game.values() if v)),
            "games_diverged": sorted(g for g, v in per_game.items() if v),
        }

    per_arm_game_rows: JsonDict = {
        arm: {g: {k: v for k, v in sweep[arm][g].items() if k != "_trajectory"} for g in roster}
        for arm in arm_names
    }

    levels_by_arm: JsonDict = {}
    states_by_arm: JsonDict = {}
    per_arm_results: list[JsonDict] = []
    any_headroom = False
    for arm_name, _flag in arms:
        rows = sweep[arm_name]
        levels_total = sum(r["levels"] for r in rows.values())
        states_total = sum(r["states_expanded"] for r in rows.values())
        if any(r["levels"] > 0 for r in rows.values()):
            any_headroom = True
        levels_by_arm[arm_name] = int(levels_total)
        states_by_arm[arm_name] = int(states_total)
        per_arm_results.append(
            {
                "arm": arm_name,
                "is_baseline": arm_name == BASELINE_ARM,
                "levels_gained_total": int(levels_total),
                "levels_delta_vs_baseline": int(levels_total - baseline_levels_total),
                "states_expanded_total": int(states_total),
                "states_delta_vs_baseline": int(states_total - baseline_states_total),
                "per_game_levels": {g: rows[g]["levels"] for g in roster},
                "per_game_levels_delta_vs_baseline": {
                    g: rows[g]["levels"] - baseline_rows[g]["levels"] for g in roster
                },
                "per_game_states_expanded": {g: rows[g]["states_expanded"] for g in roster},
            }
        )

    # Safety-regression check (exp5729 discipline) -- includes states REGRESSION either way.
    per_config_safety: dict[str, JsonDict] = {}
    for res in per_arm_results:
        arm = res["arm"]
        delta = int(res["states_delta_vs_baseline"])
        rel = (delta / baseline_states_total) if baseline_states_total else 0.0
        regression = bool(arm != BASELINE_ARM and rel > STATES_REGRESSION_REL)
        per_config_safety[arm] = {
            "states_expanded_total": int(res["states_expanded_total"]),
            "states_delta_vs_baseline": delta,
            "states_rel_change_vs_baseline": round(float(rel), 4),
            "states_expanded_regression": regression,
        }
    safety_regression_check = {
        "baseline_states_expanded_total": int(baseline_states_total),
        "states_regression_relative_threshold": STATES_REGRESSION_REL,
        "per_config": per_config_safety,
        "any_config_states_regression": bool(
            any(v["states_expanded_regression"] for v in per_config_safety.values())
        ),
        "interpretation": (
            "states_expanded is the search cost; a reorder that banks a level while materially "
            "inflating states may be luck-under-noise, not a real capability gain. It only earns a "
            "live-default recommendation if it raises levels WITHOUT tripping the >20% regression."
        ),
    }

    # Decisive r11l/su15 test.
    r11l_su15_detail: JsonDict = {}
    any_decisive_fixed = False
    for g in DECISIVE_GAMES:
        base_lv = baseline_rows[g]["levels"] if g in baseline_rows else 0
        treat_lv = max((sweep[a][g]["levels"] for a in treatment_arms if g in sweep[a]), default=0)
        fixed = bool(treat_lv > base_lv)
        any_decisive_fixed = any_decisive_fixed or fixed
        r11l_su15_detail[g] = {
            "baseline_levels": int(base_lv),
            "treatment_levels": int(treat_lv),
            "gained_additional_level": fixed,
        }
    r11l_su15_specifically_fixed = {
        "fixed": bool(any_decisive_fixed),
        "per_game": r11l_su15_detail,
        "detail": (
            "True iff r11l and/or su15 completes an ADDITIONAL level under the reorder within budget "
            "that baseline does not -- the direct point of fixing click-ranking."
        ),
    }

    any_beats = any(levels_by_arm[a] > baseline_levels_total for a in treatment_arms)
    beating_clean = [
        a
        for a in treatment_arms
        if levels_by_arm[a] > baseline_levels_total
        and not per_config_safety[a]["states_expanded_regression"]
    ]
    any_beats_clean = bool(beating_clean)
    total_traj_div = sum(
        trajectories_diverge_per_game[a]["n_games_diverged"] for a in treatment_arms
    )
    rank_reduced = rank_shift.get("summary", {}).get("games_low_rank_reduced", 0)

    # Verdict (terminal-prefixed).
    if any_decisive_fixed and any_beats_clean:
        verdict = (
            "complete: small_object_first_click_reorder_recovers_level_on_decisive_games_"
            f"baseline_{baseline_levels_total}_to_{max(levels_by_arm[a] for a in treatment_arms)}_"
            "no_states_regression"
        )
    elif any_beats_clean:
        verdict = (
            "complete: small_object_first_click_reorder_raises_roster_levels_"
            f"{baseline_levels_total}_to_{max(levels_by_arm[a] for a in treatment_arms)}_"
            "but_not_on_the_decisive_r11l_su15_games_no_states_regression"
        )
    elif any_beats:
        verdict = "complete: small_object_first_click_reorder_raises_levels_but_with_states_expanded_safety_regression"
    elif total_traj_div > 0:
        verdict = (
            "complete: small_object_first_click_reorder_changes_search_on_"
            f"{total_traj_div}_arm_games_but_no_level_gain_over_baseline_{baseline_levels_total}_"
            f"and_offline_rank_shift_is_a_wash_reduced_{rank_reduced}_of_2_decisive_games_"
            "missing_verifier_gap"
        )
    else:
        verdict = (
            "complete: small_object_first_click_reorder_inert_no_level_gain_no_search_change_over_"
            f"baseline_{baseline_levels_total}_click_ranking_is_not_the_live_path_lever_missing_verifier_gap"
        )

    # Recommendation (operator-only).
    if any_beats_clean and any_decisive_fixed:
        recommendation = (
            f"The reorder banks additional levels including on the decisive r11l/su15 games "
            f"({baseline_levels_total} -> {max(levels_by_arm[a] for a in treatment_arms)}) WITHOUT a "
            "states_expanded regression. This is a candidate to flip the live default "
            f"({FLAG}=1 / a SUBMITTED_SMALL_OBJECT_FIRST_ENABLED gate). The operator decides whether "
            "to flip the shipped default (NOT self-authorized); review the per-game deltas + safety "
            "table first."
        )
    elif any_beats:
        recommendation = (
            f"The reorder raises roster levels ({baseline_levels_total} -> "
            f"{max(levels_by_arm[a] for a in treatment_arms)}) but either not on r11l/su15 or with a "
            "states_expanded safety regression. Do NOT flip the live default on this basis; "
            "operator-only whether to investigate with more seeds."
        )
    else:
        recommendation = (
            f"NO arm banks more levels than baseline ({baseline_levels_total}); r11l/su15 do NOT gain "
            "a level under the reorder; and the offline rank-shift is a WASH (the bucket-c winners are "
            f"not surfaced -- reduced low-rank on {rank_reduced} of 2 decisive games). Keep the live "
            f"default OFF ({FLAG} stays unset / a SUBMITTED_SMALL_OBJECT_FIRST_ENABLED gate stays "
            "False). The click-ranking gap is a discriminating-SIGNAL absence (a single winning pixel "
            "indistinguishable within the equal-size pixel field), NOT a formula bug a static "
            "perceptual reorder can close -- log it as a Missing-Verifier Gap for a learned / "
            "goal-conditioned click discriminator (the deeper world-model INDUCTION lever exp5757 "
            "named). Operator-only whether to act."
        )

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "requirements": [
            "REQ-ARC-FCP-5758",
            "SCENARIO-ARC-FCP-5758-ELEVEN-GAME-CAPABILITY-AB",
            "SCENARIO-ARC-FCP-5758-R11L-SU15-DECISIVE",
            "SCENARIO-ARC-FCP-5758-RANK-SHIFT-OFFLINE",
        ],
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "field_principles": FIELD_PRINCIPLES,
        "roster": list(roster),
        "budget": int(budget),
        "arms_tested": arm_names,
        "baseline_arm": BASELINE_ARM,
        "baseline_levels_total": int(baseline_levels_total),
        "baseline_states_expanded_total": int(baseline_states_total),
        "diagnosed_ranking_failure_mode": DIAGNOSED_RANKING_FAILURE_MODE,
        "fix_description": FIX_DESCRIPTION,
        "rank_shift_offline": rank_shift,
        "per_arm_results": per_arm_results,
        "per_arm_game_rows": per_arm_game_rows,
        "levels_gained_total_by_arm": levels_by_arm,
        "states_expanded_total_by_arm": states_by_arm,
        "trajectories_diverge_per_game": trajectories_diverge_per_game,
        "r11l_su15_specifically_fixed": r11l_su15_specifically_fixed,
        "levels_gained_headroom_present": bool(any_headroom),
        "any_config_beats_baseline_levels": bool(any_beats),
        "any_config_beats_baseline_without_safety_regression": bool(any_beats_clean),
        "safety_regression_check": safety_regression_check,
        "prior_work_extended": PRIOR_WORK_EXTENDED,
        "recommendation": recommendation,
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "preconditions_checked": preconds,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
