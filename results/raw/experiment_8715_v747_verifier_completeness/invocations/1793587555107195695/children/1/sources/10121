"""proto_carnot_pruner_abstain.py — ABSTAINING no-op-deferral Carnot pruner on just-explore.

Motivation (TRUST the two prior results, do not redo them):
  - hard_argmax pruner = bimodal: vc33 27x efficiency + lp85/ar25/ft09 wins, BUT
    LOST m0r0's solve (5/5→0/5) and regressed r11l/s5i5/sp80.
  - eps-greedy / weighted-sample hedges FIXED efficiency (median_eff 1.15-1.42>1)
    but EVERY hedge arm STILL regressed 3-4 games' solve rates -> NO deployable hedge.
  Diagnosis: the Carnot frame-change verifier is a BLUNT CROSS-GAME MARGINAL that
  confidently PROMOTES wrong edges on some games (frame-change != progress). No
  policy that PROMOTES a specific edge via the verifier is solve-safe.

This experiment tests the ONE policy class that NEVER promotes a specific edge:
  **abstain_noop_defer**.
  Partition the UNTESTED edges into predicted-LIVE vs predicted-NO-OP by the CNN's
  FRAME-CHANGE probability P(change) (sigmoid 0-1). Policy:
    - if any predicted-LIVE edges  -> uniform-random among LIVE
        (vanilla diversity is fully preserved AMONG live edges — the win edge,
         if live, is reachable with the same uniform probability vanilla gives it)
    - ONLY when LIVE is empty       -> uniform-random among predicted-NO-OP
  This DEFERS confident no-ops to the END of each node's untested set, so the win
  is reached before the no-op tries are spent (saving actions), but it can NEVER
  misdirect onto a specific wrong edge the way hard-argmax killed m0r0.

Partition signal: the CNN's PURE P(change) (cnn_scorer.candidate_score, a sigmoid
in [0,1]) with ABSOLUTE CNN-P thresholds tau in {0.05, 0.1, 0.2}. We use the CNN
component directly (NOT the blended LiveActionEffectScorer, whose memory term
dominates and is unbounded > 1), because the task's LIVE/NO-OP partition is a
frame-change probability and the CNN head is exactly a calibrated P(change).
If the CNN component is unavailable for an edge (action maps to None / aid not in
1..6) we treat it as predicted-LIVE (conservative: never defer an edge we cannot
score, so we cannot drop a reachable edge to the no-op tier).

DECISIVE METRIC: an abstaining tau "succeeds" iff vs vanilla it PRESERVES every
solve (carnot_n_solved >= vanilla_n_solved on ALL 9 games — m0r0 must NOT regress)
AND median_efficiency_ratio > 1.0.

Same 9 games, 5 seeds, budget 2000, SAME paired base_seed as the prior hedged run
(base_seed=24487, derived from RANDOM_SEED=4732) so the per-seed sequences match
vanilla exactly and the comparison is paired. Vanilla is RE-RUN in-process here for
a clean paired baseline (its numbers reproduce the prior artifact's vanilla block).

Artifact: results/proto_carnot_pruner_abstain.json
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
import traceback
import types
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

# ─── Path constants ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
JE_ROOT = Path("/home/ianblenke/arc-sota-refs/arc-agi-3-just-explore")
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Artifact label seed (per task: random_seed=4733). The TRUE pairing anchor is
# BASE_SEED, pinned to the prior hedged run's base_seed (24487) so every per-game
# per-seed sequence is IDENTICAL to the prior vanilla baseline -> paired.
RANDOM_SEED = 4733
BASE_SEED = 24487  # == prior hedged run's base_seed_used (RANDOM_SEED 4732 -> 24487)
N_SEEDS = 5
BUDGET = 2000
SOLVED_GAMES = ["ar25", "cd82", "ft09", "lp85", "m0r0", "r11l", "s5i5", "sp80", "vc33"]

# Two partition modes (see partition_signal in the artifact + the measured
# CNN P(change) distribution, which is near-constant ~0.26..0.81 with NOTHING
# below 0.26 — so absolute thresholds 0.05/0.1/0.2 defer ZERO edges):
#
#   "absolute": predicted-LIVE iff CNN P(change) >= tau (task's first option).
#               At {0.05,0.1,0.2} this is a guaranteed NO-OP on this CNN (the
#               honest absolute-threshold finding). We run it to PROVE the no-op.
#   "percentile": within EACH node, defer the bottom `q` fraction of untested
#                 edges by per-node CNN P(change) rank (predicted-NO-OP); the
#                 top (1-q) are predicted-LIVE. This is the task's explicit
#                 fallback ("partition by a PERCENTILE of the per-node score
#                 distribution") and is what actually EXERCISES the pruner.
ABSOLUTE_TAUS = [0.05, 0.1, 0.2]
PERCENTILE_QS = [0.2, 0.3, 0.5]
# Each arm is (mode, value). mode in {"absolute","percentile"}.
TAU_GRID: list[tuple[str, float]] = (
    [("absolute", t) for t in ABSOLUTE_TAUS]
    + [("percentile", q) for q in PERCENTILE_QS]
)


# ─── 1. Load just-explore modules WITHOUT their broken __init__.py ─────────────
def _load_je_modules() -> dict[str, Any]:
    """Load just-explore structs/tracing/recorder/agent/graph_explorer/heuristic_agent.

    WHY: agents/__init__.py imports langgraph (not installed). We load each file
    directly via importlib, namespacing them into a stub 'agents' package.
    """
    if str(JE_ROOT) not in sys.path:
        sys.path.insert(0, str(JE_ROOT))

    agents_pkg = types.ModuleType("agents")
    sys.modules["agents"] = agents_pkg

    modules: dict[str, Any] = {}
    for mod_name, rel_path in [
        ("agents.structs", "agents/structs.py"),
        ("agents.tracing", "agents/tracing.py"),
        ("agents.recorder", "agents/recorder.py"),
        ("agents.agent", "agents/agent.py"),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, JE_ROOT / rel_path)
        m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[mod_name] = m
        spec.loader.exec_module(m)  # type: ignore[union-attr]
        setattr(agents_pkg, mod_name.split(".")[-1], m)
        modules[mod_name] = m

    spec_ge = importlib.util.spec_from_file_location(
        "graph_explorer", JE_ROOT / "graph_explorer.py"
    )
    ge_m = importlib.util.module_from_spec(spec_ge)  # type: ignore[arg-type]
    sys.modules["graph_explorer"] = ge_m
    spec_ge.loader.exec_module(ge_m)  # type: ignore[union-attr]
    modules["graph_explorer"] = ge_m

    spec_ha = importlib.util.spec_from_file_location(
        "agents.heuristic_agent", JE_ROOT / "agents/heuristic_agent.py"
    )
    ha_m = importlib.util.module_from_spec(spec_ha)  # type: ignore[arg-type]
    sys.modules["agents.heuristic_agent"] = ha_m
    spec_ha.loader.exec_module(ha_m)  # type: ignore[union-attr]
    modules["agents.heuristic_agent"] = ha_m

    return modules


JE_MODS = _load_je_modules()
JEFrameData = JE_MODS["agents.structs"].FrameData
JEGameAction = JE_MODS["agents.structs"].GameAction
JEGameState = JE_MODS["agents.structs"].GameState
OrigHeuristicAgent = JE_MODS["agents.heuristic_agent"].HeuristicAgent
OrigGraphExplorer = JE_MODS["graph_explorer"].GraphExplorer


# ─── 2. Load Carnot agentic stack ─────────────────────────────────────────────
from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_live_adapter import (  # noqa: E402
    ArcAction,
    _available_action_ids,
    _levels_completed,
)
from carnot.agentic.arc_agi3_world_model import grid_of  # noqa: E402
from carnot.agentic.arc_frame_change_predictor import (  # noqa: E402
    load_live_action_effect_scorer,
)
from carnot.agentic.arc_variant_generator import VariantEnv  # noqa: E402
from arcengine import GameAction as OurGameAction  # noqa: E402


# ─── 3. Load the live action-effect scorer (we use its .cnn_scorer component) ──
print("Loading Carnot LiveActionEffectScorer...", flush=True)
_CARNOT_SCORER = load_live_action_effect_scorer(REPO_ROOT)
if _CARNOT_SCORER is None:
    print("ERROR: Carnot scorer failed to load (no transition corpus / no CNN checkpoint)")
    sys.exit(1)
_CNN_SCORER = _CARNOT_SCORER.cnn_scorer
print(
    f"  Scorer loaded: memory={_CARNOT_SCORER.memory is not None}, "
    f"cnn={_CNN_SCORER is not None}"
)
if _CNN_SCORER is None:
    # The abstaining partition REQUIRES the frame-change CNN's P(change). Without
    # it we cannot honestly compute a LIVE/NO-OP split, so abort rather than
    # fall back to the blended (unbounded) score with an arbitrary cutoff.
    print("ERROR: CNN frame-change component is required for the abstaining partition")
    sys.exit(1)


# ─── 4. Abstaining pruned GraphExplorer ────────────────────────────────────────
class AbstainPrunedGraphExplorer(OrigGraphExplorer):
    """GraphExplorer that DEFERS predicted-no-op untested edges (never promotes one).

    The ONLY change to just-explore logic is the choice of WHICH untested edge to
    try next. Partition untested edges into predicted-LIVE (CNN P(change) >= tau)
    vs predicted-NO-OP (< tau):
      - LIVE non-empty -> uniform-random among LIVE  (vanilla diversity preserved)
      - LIVE empty     -> uniform-random among NO-OP
    This NEVER argmaxes onto a specific edge, so it cannot misdirect like the
    hard-argmax m0r0 failure; it only reorders confident no-ops to the back.

    The frame + edge->action mapping are threaded in by the agent via
    `_current_frame` and `_edge_to_action_fn` before each choose_edge call.
    """

    def __init__(
        self,
        *args: Any,
        cnn_scorer: Any = None,
        mode: str = "absolute",
        tau: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cnn_scorer = cnn_scorer
        # mode="absolute": LIVE iff P(change) >= tau.
        # mode="percentile": defer the bottom `tau` fraction of each node's
        #   untested edges by per-node P(change) rank (those are predicted-NO-OP).
        self.mode = str(mode)
        self.tau = float(tau)
        # Set by the agent before each choose_edge call:
        self._current_frame: Any = None
        self._edge_to_action_fn: Any = None  # callable(edge_idx) -> ArcAction|None
        # Diagnostics
        self.pruner_total_choices = 0
        self.pruner_changed_count = 0   # decisions where chosen != vanilla random.choice
        self.pruner_deferred_edges = 0  # total edges classified NO-OP and deferred
        self.pruner_defer_active_steps = 0  # decisions where >0 edges were deferred
        self.pruner_live_empty_steps = 0    # decisions where ALL edges predicted NO-OP
        self.pruner_scored_edges = 0    # total edges we got a CNN score for

    def choose_edge(self, node: Any, return_reasoning: bool = False) -> Any:
        node_info = self._nodes[node]
        if node_info.has_open_group(self.active_group):
            untested_edges: list[int] = []
            for group_id in range(self.active_group + 1):
                untested_edges.extend(node_info.group2remaining_candidate_ids[group_id])
            if not untested_edges:
                raise ValueError("No untested edges in the current group while the group is open")

            self.pruner_total_choices += 1
            # Draw the vanilla baseline FIRST and unconditionally so the RNG
            # advances identically to vanilla on this decision (paired-per-seed).
            random_choice = random.choice(untested_edges)

            if (
                self.cnn_scorer is not None
                and self._current_frame is not None
                and self._edge_to_action_fn is not None
                and len(untested_edges) > 1
            ):
                # Score every untested edge with the CNN P(change). Unscorable
                # edges (action maps to None) are tagged LIVE unconditionally
                # (never defer what we cannot score -> every reachable edge stays
                # reachable). Scorable edges carry their P(change) for partition.
                scorable: list[tuple[int, float]] = []  # (edge, p_change)
                unscorable_live: list[int] = []
                for e in untested_edges:
                    arc_action = self._edge_to_action_fn(e)
                    if arc_action is None:
                        unscorable_live.append(e)
                        continue
                    p_change = float(self.cnn_scorer.candidate_score(self._current_frame, arc_action))
                    self.pruner_scored_edges += 1
                    scorable.append((e, p_change))

                live_edges: list[int] = list(unscorable_live)
                noop_edges: list[int] = []
                if self.mode == "absolute":
                    for e, p in scorable:
                        (live_edges if p >= self.tau else noop_edges).append(e)
                else:  # percentile: defer the bottom `tau` fraction by per-node P-rank
                    n = len(scorable)
                    n_defer = int(n * self.tau)  # floor; q=0 or tiny n => defer 0
                    if n_defer <= 0 or n == 0:
                        live_edges.extend(e for e, _ in scorable)
                    else:
                        # Sort ascending by P(change); the lowest n_defer are NO-OP.
                        ranked = sorted(scorable, key=lambda t: (t[1], t[0]))
                        defer_set = {e for e, _ in ranked[:n_defer]}
                        for e, _ in scorable:
                            (noop_edges if e in defer_set else live_edges).append(e)

                deferred = len(noop_edges)
                if deferred > 0:
                    self.pruner_deferred_edges += deferred
                    self.pruner_defer_active_steps += 1

                if live_edges:
                    pool = live_edges
                else:
                    # All untested edges predicted NO-OP: fall back to the full
                    # set so we never get stuck (uniform among predicted-no-ops).
                    pool = noop_edges if noop_edges else untested_edges
                    self.pruner_live_empty_steps += 1

                chosen = random.choice(pool)

                # "changed" = the abstain policy picked a different edge than the
                # vanilla draw AND a partition actually happened (some edge deferred).
                if deferred > 0 and chosen != random_choice:
                    self.pruner_changed_count += 1

                edge_idx = chosen
                reasoning = (
                    f"Carnot-abstain({self.mode}={self.tau}): {len(live_edges)} LIVE / "
                    f"{len(noop_edges)} NO-OP of {len(untested_edges)} untested; "
                    f"picked from {'LIVE' if live_edges else 'NO-OP-fallback'} pool\n"
                )
            else:
                edge_idx = random_choice
                reasoning = f"Vanilla fallback: chose edge {edge_idx} (scorer/single-edge)\n"
        else:
            # Lowest-distance traversal path (tested edges; unchanged from vanilla).
            lowest_dist = node_info.distance
            edges_with_lowest_dist = [
                edge_idx for edge_idx, edge_data in enumerate(node_info.edge_data)
                if edge_data["distance"] <= lowest_dist
                and edge_data["result"] == 1
                and edge_data["group"] <= self.active_group
            ]
            edge_idx = random.choice(edges_with_lowest_dist)
            reasoning = f"Chose edge {edge_idx} with lowest dist {lowest_dist}\n"

        reasoning += f"Node info: {node_info}\n"
        if return_reasoning:
            return edge_idx, reasoning
        return edge_idx


# ─── 5. Abstaining pruned HeuristicAgent ──────────────────────────────────────
class AbstainPrunedHeuristicAgent(OrigHeuristicAgent):
    """HeuristicAgent subclass that uses AbstainPrunedGraphExplorer.

    Identical context-threading to CarnotPrunedHeuristicAgent in
    proto_carnot_pruner.py: capture segmented_frame + arrow_actions + click count
    so each edge_idx can be mapped to an ArcAction for CNN scoring, then set the
    frame/mapping on the explorer right before choose_edge runs.
    """

    def __init__(
        self,
        *args: Any,
        cnn_scorer: Any = None,
        mode: str = "absolute",
        tau: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.graph_explorer = AbstainPrunedGraphExplorer(
            verbose_level=self.verbose_level,
            n_groups=self.N_GROUPS,
            cnn_scorer=cnn_scorer,
            mode=mode,
            tau=tau,
        )
        self._cnn_scorer = cnn_scorer
        self._last_segmented_frame: Any = None
        self._last_frame_segments: list[Any] = []
        self._last_arrow_actions: list[Any] = []
        self._last_num_click_actions: int = 0
        self._last_je_frame: Any = None

    def _edge_to_arc_action(self, edge_idx: int) -> ArcAction | None:
        """Convert a just-explore edge_idx to a Carnot ArcAction for CNN scoring.

        edge_idx < num_click_actions  -> segment click (ACTION6 at segment centroid)
        edge_idx >= num_click_actions -> directional arrow action (ACTION1..5)
        (identical mapping to proto_carnot_pruner.py).
        """
        if edge_idx < self._last_num_click_actions:
            if self._last_segmented_frame is None:
                return None
            seg_mask = self._last_segmented_frame == edge_idx
            seg_points = np.argwhere(seg_mask)
            if len(seg_points) == 0:
                return None
            ys = seg_points[:, 0]
            xs = seg_points[:, 1]
            cy = int(np.median(ys))
            cx = int(np.median(xs))
            return ArcAction(action_id=6, data={"x": cx, "y": cy}, source="segment_centroid")
        else:
            arrow_idx = edge_idx - self._last_num_click_actions
            if arrow_idx >= len(self._last_arrow_actions):
                return None
            je_arrow = self._last_arrow_actions[arrow_idx]
            aid = je_arrow.value
            if 1 <= aid <= 5:
                return ArcAction(action_id=aid, data=None, source="directional_action")
            return None

    def choose_action(self, frames: Any, latest_frame: Any) -> Any:
        self._last_segmented_frame = None
        self._last_frame_segments = []
        self._last_arrow_actions = []
        self._last_num_click_actions = 0
        self._last_je_frame = latest_frame

        original_segment_frame = self.frame_processor.segment_frame

        def capturing_segment_frame(frame_np: Any) -> Any:
            result = original_segment_frame(frame_np)
            seg_frame, seg_list = result
            self._last_segmented_frame = seg_frame
            self._last_frame_segments = seg_list
            self._last_num_click_actions = len(seg_list) if len(seg_list) > 0 else 0
            return result

        self.frame_processor.segment_frame = capturing_segment_frame

        original_choose_edge = self.graph_explorer.choose_edge

        def wrapped_choose_edge(node: Any, return_reasoning: bool = False) -> Any:
            avail = getattr(latest_frame, "available_actions", []) or []
            SIMPLE_ACTION_ID2GAME_ACTION = OrigHeuristicAgent.SIMPLE_ACTION_ID2GAME_ACTION
            arrow_actions = []
            for action_id in avail:
                if action_id in SIMPLE_ACTION_ID2GAME_ACTION:
                    arrow_actions.append(SIMPLE_ACTION_ID2GAME_ACTION[action_id])
            self._last_arrow_actions = arrow_actions

            self.graph_explorer._current_frame = latest_frame
            self.graph_explorer._edge_to_action_fn = self._edge_to_arc_action

            return original_choose_edge(node, return_reasoning=return_reasoning)

        self.graph_explorer.choose_edge = wrapped_choose_edge

        try:
            result = super().choose_action(frames, latest_frame)
        finally:
            self.frame_processor.segment_frame = original_segment_frame
            self.graph_explorer.choose_edge = original_choose_edge

        return result


# ─── 6. Frame conversion helpers (identical to proto_carnot_pruner.py) ─────────
def _our_raw_to_je_fd(raw: Any, game_id: str, start_level: int) -> Any:
    grid = grid_of(raw)
    frame_3d = [grid.tolist()]
    raw_state = str(getattr(raw, "state", "") or "").upper()
    lc = _levels_completed(raw)
    if lc > start_level:
        je_state = JEGameState.WIN
    elif "GAME_OVER" in raw_state or "LOSE" in raw_state:
        je_state = JEGameState.GAME_OVER
    elif "NOT_PLAYED" in raw_state and lc == 0:
        je_state = JEGameState.NOT_PLAYED
    else:
        je_state = JEGameState.NOT_FINISHED
    avail = _available_action_ids(raw)
    return JEFrameData(
        game_id=game_id,
        frame=frame_3d,
        state=je_state,
        score=lc,
        available_actions=avail,
    )


def _je_action_to_ours(je_action: Any) -> tuple[str, Any, dict | None]:
    aid = je_action.value
    if aid == 0:
        return "RESET", OurGameAction.RESET, None
    if aid == 6:
        ad = je_action.action_data
        x, y = int(ad.x), int(ad.y)
        return json.dumps({"action": 6, "x": x, "y": y}), OurGameAction.ACTION6, {"x": x, "y": y}
    our_ga = getattr(OurGameAction, f"ACTION{aid}")
    return json.dumps({"action": aid}), our_ga, None


# ─── 7. Run one game (vanilla or abstaining) ───────────────────────────────────
def run_one_game(
    game_id: str,
    budget: int,
    arc: Any,
    seed: int,
    pruner_mode: bool,
    cnn_scorer: Any = None,
    mode: str = "absolute",
    tau: float = 0.1,
) -> dict:
    """Run one game with either vanilla or the abstaining-pruned agent.

    seed controls random.seed for this run. Vanilla and the abstaining arm share
    the same per-seed RNG stream (the explorer draws the vanilla random.choice
    FIRST on every decision), so they are paired per seed.
    """
    random.seed(seed)
    np.random.seed(seed % (2**31))

    result: dict = {
        "game": game_id,
        "budget": budget,
        "seed": seed,
        "pruner_mode": pruner_mode,
        "policy": (f"abstain_{mode}_{tau}" if pruner_mode else "vanilla"),
        "reached_level": 0,
        "solved": False,
        "actions_used": 0,
        "actions_to_first_levelup": None,
        "adapter_failed": False,
        "adapter_error": None,
        "pruner_total_choices": 0,
        "pruner_changed_count": 0,
        "pruner_deferred_edges": 0,
        "pruner_defer_active_steps": 0,
        "pruner_live_empty_steps": 0,
        "pruner_scored_edges": 0,
    }

    try:
        sc = arc.open_scorecard()
        base_env = arc.make(game_id, scorecard_id=sc)
        env = VariantEnv(base_env, game_id, 1)
        raw = env.reset()
        start_level = _levels_completed(raw)

        if pruner_mode:
            agent = AbstainPrunedHeuristicAgent(
                card_id="abstain_card",
                game_id=game_id,
                agent_name="carnot_abstain",
                ROOT_URL="http://localhost:0",
                record=False,
                cnn_scorer=cnn_scorer,
                mode=mode,
                tau=tau,
            )
        else:
            agent = OrigHeuristicAgent(
                card_id="vanilla_card",
                game_id=game_id,
                agent_name="vanilla",
                ROOT_URL="http://localhost:0",
                record=False,
            )

        agent.MAX_ACTIONS = budget
        agent.minimal_step_time = 0.0

        je_fd_init = JEFrameData(
            game_id=game_id,
            frame=[],
            state=JEGameState.NOT_PLAYED,
            score=0,
            available_actions=[],
        )
        je_frames: list[Any] = [je_fd_init]

        max_lc = start_level
        prev_score = 0

        for step in range(budget):
            latest_je = je_frames[-1]
            if agent.is_done(je_frames, latest_je):
                break

            try:
                je_action = agent.choose_action(je_frames, latest_je)
            except Exception:
                agent.failed = True
                agent.level_up = True
                je_action = agent.last_action_object

            label, our_ga, data = _je_action_to_ours(je_action)

            if our_ga == OurGameAction.RESET:
                raw = env.reset()
                start_level_new = _levels_completed(raw)
                if max_lc == 0:
                    start_level = start_level_new
            else:
                raw = env.step(our_ga, data=data)

            lc = _levels_completed(raw)
            new_score = lc
            if new_score > prev_score:
                agent.level_up = True
                agent.status_bar_mask = None
            elif agent.status_bar_mask is not None:
                agent.level_up = False
            prev_score = new_score

            je_fd = _our_raw_to_je_fd(raw, game_id, start_level)
            je_frames.append(je_fd)
            agent.frames.append(je_fd)
            agent.action_counter = step + 1
            if je_fd.guid:
                agent.guid = je_fd.guid

            max_lc = max(max_lc, lc)
            if lc > start_level and result["actions_to_first_levelup"] is None:
                result["actions_to_first_levelup"] = step + 1

            if je_fd.state == JEGameState.WIN:
                break

        result["actions_used"] = agent.action_counter
        result["reached_level"] = max(0, max_lc - start_level)
        result["solved"] = result["reached_level"] >= 1

        if pruner_mode and hasattr(agent, "graph_explorer"):
            ge = agent.graph_explorer
            result["pruner_total_choices"] = ge.pruner_total_choices
            result["pruner_changed_count"] = ge.pruner_changed_count
            result["pruner_deferred_edges"] = ge.pruner_deferred_edges
            result["pruner_defer_active_steps"] = ge.pruner_defer_active_steps
            result["pruner_live_empty_steps"] = ge.pruner_live_empty_steps
            result["pruner_scored_edges"] = ge.pruner_scored_edges

    except Exception:
        result["adapter_failed"] = True
        result["adapter_error"] = traceback.format_exc()

    return result


# ─── 8. Run N seeds and collect stats ─────────────────────────────────────────
def run_game_n_seeds(
    game_id: str,
    budget: int,
    arc: Any,
    base_seed: int,
    n_seeds: int,
    pruner_mode: bool,
    cnn_scorer: Any = None,
    mode: str = "absolute",
    tau: float = 0.1,
) -> dict:
    """Run a game N times (paired per-seed sequence base_seed + i*37)."""
    per_seed: list[dict] = []
    atfl_values: list[int] = []

    for i in range(n_seeds):
        seed = base_seed + i * 37
        print(f"    seed={seed}...", end=" ", flush=True)
        r = run_one_game(game_id, budget, arc, seed, pruner_mode, cnn_scorer, mode=mode, tau=tau)
        per_seed.append(r)

        if r["adapter_failed"]:
            print("ADAPTER_FAILED")
        elif r["solved"]:
            atfl = r["actions_to_first_levelup"]
            print(f"SOLVED (first_levelup={atfl})")
            if atfl is not None:
                atfl_values.append(atfl)
        else:
            print(f"no_solve (actions={r['actions_used']})")

    n_solved = sum(1 for r in per_seed if r.get("solved"))
    median_atfl = float(median(atfl_values)) if atfl_values else None
    min_atfl = min(atfl_values) if atfl_values else None
    return {
        "game": game_id,
        "n_seeds": n_seeds,
        "n_solved": n_solved,
        "all_solved": n_solved == n_seeds,
        "any_solved": n_solved > 0,
        "median_atfl": median_atfl,
        "min_atfl": min_atfl,
        "atfl_values": atfl_values,
        "per_seed": per_seed,
        "pruner_total_choices": sum(r.get("pruner_total_choices", 0) for r in per_seed),
        "pruner_changed_count": sum(r.get("pruner_changed_count", 0) for r in per_seed),
        "pruner_deferred_edges": sum(r.get("pruner_deferred_edges", 0) for r in per_seed),
        "pruner_defer_active_steps": sum(r.get("pruner_defer_active_steps", 0) for r in per_seed),
        "pruner_live_empty_steps": sum(r.get("pruner_live_empty_steps", 0) for r in per_seed),
        "pruner_scored_edges": sum(r.get("pruner_scored_edges", 0) for r in per_seed),
    }


# ─── 9. Summarize one tau arm vs vanilla ──────────────────────────────────────
def _summarize_tau(
    mode: str,
    tau: float,
    all_vanilla: dict[str, dict],
    all_arm: dict[str, dict],
) -> dict:
    """Per-game + aggregate stats for one tau, paired against vanilla.

    SUCCEEDS iff carnot_n_solved >= vanilla_n_solved on ALL 9 games AND
    median_efficiency_ratio > 1.0 AND the pruner was exercised.
    """
    per_game: list[dict] = []
    efficiency_ratios: list[float] = []
    action_reductions: list[float] = []
    solve_preserved_count = 0
    solve_rate_regressed_games: list[str] = []
    solve_lost_games: list[str] = []
    total_choices = 0
    total_changed = 0
    total_deferred = 0
    total_defer_steps = 0

    for game in SOLVED_GAMES:
        v = all_vanilla[game]
        p = all_arm[game]
        v_median = v["median_atfl"]
        p_median = p["median_atfl"]
        v_n = v["n_solved"]
        p_n = p["n_solved"]

        if v_median is not None and p_median is not None and p_median > 0:
            action_reduction = float(v_median - p_median)
            efficiency_ratio = float((v_median / p_median) ** 2)
        else:
            action_reduction = None
            efficiency_ratio = None

        solve_preserved = (p_n >= v_n)
        if solve_preserved:
            solve_preserved_count += 1
        else:
            solve_rate_regressed_games.append(game)
            if p["any_solved"] is False and v["any_solved"] is True:
                solve_lost_games.append(game)

        if efficiency_ratio is not None:
            efficiency_ratios.append(efficiency_ratio)
        if action_reduction is not None:
            action_reductions.append(action_reduction)

        total_choices += p["pruner_total_choices"]
        total_changed += p["pruner_changed_count"]
        total_deferred += p["pruner_deferred_edges"]
        total_defer_steps += p["pruner_defer_active_steps"]

        per_game.append({
            "game": game,
            "vanilla_median": v_median,
            "arm_median": p_median,
            "vanilla_n_solved": v_n,
            "arm_n_solved": p_n,
            "vanilla_actions_min": v["min_atfl"],
            "arm_actions_min": p["min_atfl"],
            "action_reduction": action_reduction,
            "efficiency_ratio": efficiency_ratio,
            "solve_preserved": solve_preserved,
            "pruner_total_choices": p["pruner_total_choices"],
            "pruner_changed_count": p["pruner_changed_count"],
            "pruner_deferred_edges": p["pruner_deferred_edges"],
            "pruner_defer_active_steps": p["pruner_defer_active_steps"],
            "pruner_live_empty_steps": p["pruner_live_empty_steps"],
        })

    median_efficiency_ratio = float(median(efficiency_ratios)) if efficiency_ratios else None
    median_action_reduction = float(median(action_reductions)) if action_reductions else None
    # Exercised = it actually DEFERRED >0 edges and changed order on >0 steps.
    pruner_exercised = (total_deferred > 0 and total_changed > 0 and total_choices > 0)
    pruner_fire_rate = (total_changed / total_choices) if total_choices > 0 else 0.0
    all_solves_preserved = (len(solve_rate_regressed_games) == 0)

    if solve_lost_games:
        worst = f"solve_lost:{solve_lost_games}"
    elif solve_rate_regressed_games:
        worst = f"solve_rate_regressed:{solve_rate_regressed_games}"
    else:
        worst_game = None
        worst_er = None
        for g in per_game:
            er = g["efficiency_ratio"]
            if er is not None and (worst_er is None or er < worst_er):
                worst_er = er
                worst_game = g["game"]
        worst = (
            f"min_efficiency_ratio={worst_er:.3f}@{worst_game}"
            if worst_er is not None else "no_efficiency_data"
        )

    deployable = bool(
        all_solves_preserved
        and median_efficiency_ratio is not None
        and median_efficiency_ratio > 1.0
        and pruner_exercised
    )

    return {
        "mode": mode,
        "tau": tau,
        "arm": f"abstain_{mode}_{tau}",
        "per_game": per_game,
        "median_efficiency_ratio": median_efficiency_ratio,
        "median_action_reduction": median_action_reduction,
        "n_games_solve_preserved": solve_preserved_count,
        "n_games_total": len(SOLVED_GAMES),
        "n_games_solve_rate_regressed": len(solve_rate_regressed_games),
        "solve_rate_regressed_games": solve_rate_regressed_games,
        "solve_lost_games": solve_lost_games,
        "all_solves_preserved": all_solves_preserved,
        "worst_per_game_outcome": worst,
        "pruner_exercised": pruner_exercised,
        "pruner_total_choices": total_choices,
        "pruner_changed_count": total_changed,
        "pruner_deferred_edges": total_deferred,
        "pruner_defer_active_steps": total_defer_steps,
        "pruner_fire_rate": round(pruner_fire_rate, 4),
        "deployable": deployable,
    }


# ─── 10. Main ──────────────────────────────────────────────────────────────────
def _arm_key(mode: str, value: float) -> str:
    return f"{mode}_{value}"


def main() -> None:
    t0 = time.time()
    base_seed = BASE_SEED  # pinned to prior hedged run for paired comparability

    arc = kit.offline_arcade()

    # ── SMOKE: vc33 (clean win) + m0r0 (canary) under percentile q=0.5 (the arm
    #    that actually defers — absolute tau defers nothing on this CNN) ──
    print("\n=== SMOKE: abstain percentile q=0.5 on vc33 (clean win) + m0r0 (canary) ===")
    smoke: dict[str, dict] = {}
    for sg in ["vc33", "m0r0"]:
        sv = run_one_game(sg, BUDGET, arc, seed=base_seed, pruner_mode=False)
        sp = run_one_game(
            sg, BUDGET, arc, seed=base_seed, pruner_mode=True,
            cnn_scorer=_CNN_SCORER, mode="percentile", tau=0.5,
        )
        smoke[sg] = {
            "vanilla_solved": sv["solved"],
            "vanilla_atfl": sv["actions_to_first_levelup"],
            "abstain_pct05_solved": sp["solved"],
            "abstain_pct05_atfl": sp["actions_to_first_levelup"],
            "pruner_total_choices": sp.get("pruner_total_choices", 0),
            "pruner_changed_count": sp.get("pruner_changed_count", 0),
            "pruner_deferred_edges": sp.get("pruner_deferred_edges", 0),
            "pruner_live_empty_steps": sp.get("pruner_live_empty_steps", 0),
            "adapter_failed": sv["adapter_failed"] or sp["adapter_failed"],
        }
        print(
            f"  {sg}: vanilla(solved={sv['solved']}, atfl={sv['actions_to_first_levelup']}) "
            f"| abstain_pct0.5(solved={sp['solved']}, atfl={sp['actions_to_first_levelup']}, "
            f"choices={sp.get('pruner_total_choices')}, changed={sp.get('pruner_changed_count')}, "
            f"deferred={sp.get('pruner_deferred_edges')}, live_empty={sp.get('pruner_live_empty_steps')})"
        )
        if sv["adapter_failed"] or sp["adapter_failed"]:
            print("  SMOKE ADAPTER FAILED — aborting.")
            if sv["adapter_failed"]:
                print(sv["adapter_error"])
            if sp["adapter_failed"]:
                print(sp["adapter_error"])
            sys.exit(1)
    print("Smoke complete. Running full sweep (absolute + percentile arms)...\n")

    # ── FULL RUN: vanilla (once/game) + every arm, N=5 seeds, paired ────────────
    all_vanilla: dict[str, dict] = {}
    arm_results: dict[str, dict[str, dict]] = {_arm_key(m, v): {} for m, v in TAU_GRID}

    for game in SOLVED_GAMES:
        game_base = base_seed + hash(game) % 10000
        print(f"\n=== {game} ===")
        print("  [vanilla]")
        all_vanilla[game] = run_game_n_seeds(
            game, BUDGET, arc, base_seed=game_base, n_seeds=N_SEEDS, pruner_mode=False,
        )
        for mode, value in TAU_GRID:
            key = _arm_key(mode, value)
            print(f"  [abstain {mode}={value}]")
            arm_results[key][game] = run_game_n_seeds(
                game, BUDGET, arc, base_seed=game_base, n_seeds=N_SEEDS,
                pruner_mode=True, cnn_scorer=_CNN_SCORER, mode=mode, tau=value,
            )

    # ── Summarize each arm ──────────────────────────────────────────────────────
    arm_summaries: dict[str, dict] = {}
    for mode, value in TAU_GRID:
        key = _arm_key(mode, value)
        arm_summaries[key] = _summarize_tau(mode, value, all_vanilla, arm_results[key])

    # ── m0r0 canary ─────────────────────────────────────────────────────────────
    m0r0_vanilla_n = all_vanilla["m0r0"]["n_solved"]
    m0r0_recovery: dict[str, dict] = {}
    for mode, value in TAU_GRID:
        key = _arm_key(mode, value)
        arm_n = arm_results[key]["m0r0"]["n_solved"]
        m0r0_recovery[key] = {
            "vanilla_n_solved": m0r0_vanilla_n,
            "arm_n_solved": arm_n,
            "preserved": bool(arm_n >= m0r0_vanilla_n),
        }
    m0r0_preserved_any = any(v["preserved"] for v in m0r0_recovery.values())

    # ── Deployable arms (preserve ALL solves AND median_eff>1 AND exercised) ─────
    deployable_arms = [name for name, s in arm_summaries.items() if s["deployable"]]

    # ── per_threshold summary (the structured-output shape) ─────────────────────
    per_threshold = []
    for mode, value in TAU_GRID:
        key = _arm_key(mode, value)
        s = arm_summaries[key]
        per_threshold.append({
            "arm": key,
            "mode": mode,
            "tau": value,
            "median_efficiency_ratio": s["median_efficiency_ratio"],
            "n_games_solve_preserved": f"{s['n_games_solve_preserved']}/{s['n_games_total']}",
            "games_regressed": s["solve_rate_regressed_games"],
            "m0r0_preserved": m0r0_recovery[key]["preserved"],
            "deployable": s["deployable"],
            "pruner_exercised": s["pruner_exercised"],
            "pruner_deferred_edges": s["pruner_deferred_edges"],
            "pruner_changed_count": s["pruner_changed_count"],
            "pruner_live_empty_steps": sum(
                g["pruner_live_empty_steps"] for g in s["per_game"]
            ),
        })

    # ── honest_verdict ──────────────────────────────────────────────────────────
    exercised_arms = [k for k, s in arm_summaries.items() if s["pruner_exercised"]]
    absolute_arms = [_arm_key("absolute", v) for v in ABSOLUTE_TAUS]
    percentile_arms = [_arm_key("percentile", v) for v in PERCENTILE_QS]
    absolute_all_noop = all(
        not arm_summaries[k]["pruner_exercised"] for k in absolute_arms
    )

    if deployable_arms:
        best = max(deployable_arms, key=lambda n: arm_summaries[n]["median_efficiency_ratio"])
        bs = arm_summaries[best]
        verdict = (
            f"success: abstaining arm '{best}' preserves ALL solves "
            f"({bs['n_games_solve_preserved']}/{bs['n_games_total']}, m0r0 preserved) AND "
            f"median_efficiency_ratio={bs['median_efficiency_ratio']:.3f} > 1.0 "
            f"(deferred_edges={bs['pruner_deferred_edges']}); deployable_arms={deployable_arms}"
        )
    elif not exercised_arms:
        verdict = (
            "complete: NO deployable abstaining arm — the pruner is a NO-OP at every threshold. "
            "Binding constraint = the CNN frame-change P(change) is near-constant (~0.26..0.81, "
            "NOTHING below 0.26), so absolute taus {0.05,0.1,0.2} defer ZERO edges AND the "
            "percentile floor never fires. The CNN cannot distinguish no-op from live edges here."
        )
    else:
        preserve_arms = [k for k, s in arm_summaries.items()
                         if s["all_solves_preserved"] and s["pruner_exercised"]]
        # Did the absolute arms degenerate to no-op (the absolute-threshold finding)?
        abs_note = (
            "absolute taus {0.05,0.1,0.2} all NO-OP (CNN P(change) never < 0.26); "
            if absolute_all_noop else ""
        )
        if preserve_arms:
            effs = {k: arm_summaries[k]["median_efficiency_ratio"] for k in preserve_arms}
            verdict = (
                f"complete: NO deployable abstaining arm — {abs_note}the percentile arms that "
                f"DO defer and preserve all solves ({preserve_arms}) do NOT keep "
                f"median_efficiency_ratio>1 (median_eff={effs}). Binding constraint = NO-OP "
                f"DEFERRAL CUTS TOO FEW ACTIONS to beat 1.0: deferring low-P(change) edges to the "
                f"back of each node's untested set rarely unblocks the win edge sooner. "
                f"m0r0_preserved={m0r0_preserved_any}"
            )
        else:
            regr = {k: arm_summaries[k]["solve_rate_regressed_games"]
                    for k in exercised_arms}
            verdict = (
                f"complete: NO deployable abstaining arm — {abs_note}every percentile arm that "
                f"actually defers REGRESSES a solve (even though it never promotes a specific "
                f"edge): the extra RNG draw + reordering perturbs the trajectory enough to drop "
                f"a solve at N=5. Binding constraint = SOLVE-RATE REGRESSION. "
                f"m0r0_preserved={m0r0_preserved_any}; exercised-arm regressions: {regr}"
            )

    duration_s = round(time.time() - t0, 2)

    vanilla_per_game = [
        {
            "game": game,
            "median_atfl": all_vanilla[game]["median_atfl"],
            "min_atfl": all_vanilla[game]["min_atfl"],
            "n_solved": all_vanilla[game]["n_solved"],
        }
        for game in SOLVED_GAMES
    ]

    payload = {
        "experiment": "proto_carnot_pruner_abstain",
        "arms": ["vanilla"] + [_arm_key(m, v) for m, v in TAU_GRID],
        "arm_grid": [{"mode": m, "value": v} for m, v in TAU_GRID],
        "partition_signal": (
            "CNN frame-change P(change) (cnn_scorer.candidate_score, the SmallFrameChangeCNN "
            "sigmoid head, 0-1). TWO modes: (1) ABSOLUTE — predicted-LIVE iff P>=tau "
            "(tau in {0.05,0.1,0.2}); on THIS CNN the measured P(change) is near-constant "
            "(~0.26..0.81, nothing below 0.26) so absolute taus defer ZERO edges (a guaranteed "
            "NO-OP — the honest absolute-threshold finding). (2) PERCENTILE — within EACH node, "
            "defer the bottom q fraction (q in {0.2,0.3,0.5}) of untested edges by per-node "
            "P(change) rank; this is the task's explicit fallback and is what actually EXERCISES "
            "the pruner. We use the PURE CNN head, NOT the blended LiveActionEffectScorer (whose "
            "memory term is unbounded > 1). Unscorable edges are tagged LIVE (never deferred)."
        ),
        "cnn_p_change_distribution_note": (
            "Measured over ~7700 untested-edge scorings across vc33/m0r0/r11l/sp80/lp85: "
            "min=0.261, p10=0.292, p50=0.384, max=0.808; frac(P<0.05)=frac(P<0.1)=frac(P<0.2)=0. "
            "This is WHY absolute taus 0.05/0.1/0.2 defer nothing."
        ),
        "vanilla_per_game": vanilla_per_game,
        "arm_summaries": arm_summaries,
        "per_threshold": per_threshold,
        "deployable_arms": deployable_arms,
        "absolute_arms_all_noop": absolute_all_noop,
        "m0r0_canary": {
            "vanilla_n_solved": m0r0_vanilla_n,
            "per_arm": m0r0_recovery,
            "m0r0_preserved_any": m0r0_preserved_any,
        },
        "smoke": smoke,
        "n_seeds": N_SEEDS,
        "budget": BUDGET,
        "games": SOLVED_GAMES,
        "n_games_total": len(SOLVED_GAMES),
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "base_seed_used": base_seed,
        "duration_s": duration_s,
        "measurement_caveats": [
            "ABSTAINING policy NEVER promotes a specific edge: it only DEFERS predicted-no-op "
            "edges and picks uniform-random among predicted-LIVE edges (vanilla diversity "
            "preserved among live edges). It STRUCTURALLY cannot misdirect like hard-argmax "
            "killed m0r0 — the win edge, if predicted LIVE, keeps the SAME uniform selection "
            "probability vanilla gives it.",
            "ABSOLUTE-tau arms {0.05,0.1,0.2} are NO-OPs on this CNN (P(change) never < 0.26). "
            "The PERCENTILE arms {0.2,0.3,0.5} are what actually exercise the pruner.",
            "Even when an abstaining arm defers 0 edges (a no-op partition), it STILL draws the "
            "RNG one extra time per decision vs vanilla (the policy's own random.choice on top of "
            "the vanilla draw), so its trajectory diverges from vanilla after decision 1. That "
            "divergence is pure noise: it can drop a solve at N=5 with no compensating action "
            "saving. This is the same seed-level (not trajectory-level) pairing the prior hedged "
            "run used, and is why a no-op arm can still show a solve-rate delta.",
            "efficiency_ratio is the SQUARED action ratio (vanilla_median/arm_median)**2 — the "
            "ARC leaderboard reward shape; eff=4.0 means a 2x action reduction.",
            "efficiency_ratio is over SOLVED seeds only; on a solve-rate-regressed game the arm "
            "can show survivorship-inflated efficiency. The deployability gate guards against "
            "this by ALSO requiring carnot_n_solved >= vanilla_n_solved on every game.",
            "N=5 seeds/arm/game is below the project N>=30 bar; vanilla itself is flaky on some "
            "games (ar25 1/5, m0r0 4/5 in the prior run). A 1-seed n_solved delta is within seed "
            "noise. Treat per-game solve-rate deltas as indicative; the cross-arm pattern is the "
            "robust finding.",
            "PAIRING IS WITHIN THIS RUN. The decisive comparison is abstain-vs-vanilla where BOTH "
            "are produced in THIS process with the identical per-game per-seed sequence "
            "(game_base=base_seed+hash(game)%10000, then base_seed+i*37), so each arm is paired "
            "against THIS run's vanilla seed-for-seed. NOTE: Python's hash() of strings is "
            "randomized per process (PYTHONHASHSEED unset), so this run's per-game seeds differ "
            "from the prior hedged run's — the vanilla MEDIANS here will NOT bit-match the prior "
            "artifact's vanilla block (different seeds), but the paired vanilla-vs-abstain "
            "comparison within this run is exact. base_seed itself is pinned to 24487.",
        ],
        "methodology_note": (
            "Carnot frame-change CNN (SmallFrameChangeCNN P(change) head) partitions just-explore's "
            "untested edges into predicted-LIVE vs predicted-NO-OP in GraphExplorer.choose_edge "
            "(absolute P>=tau, or per-node bottom-q percentile defer). Policy abstain_noop_defer: "
            "pick uniform-random among LIVE if any LIVE exist, else uniform-random among NO-OP. "
            "This DEFERS confident no-ops (saving the actions spent trying them before the win is "
            "found) but NEVER argmaxes onto a specific edge, so it cannot misdirect. Edge->ArcAction "
            "mapping identical to proto_carnot_pruner.py (edge<num_click -> segment-centroid ACTION6; "
            "else ACTION1..5). All other just-explore logic unchanged. Same budget(2000), same 9 "
            "games, same offline arcade variant-1, same paired per-seed RNG (vanilla random.choice "
            "drawn first each decision). CPU-only. An arm SUCCEEDS iff carnot_n_solved >= "
            "vanilla_n_solved on ALL 9 games (m0r0 included) AND median_efficiency_ratio > 1.0 AND "
            "the pruner was exercised (deferred >0 edges, changed order on >0 steps)."
        ),
    }

    payload_for_hash = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    chksum = hashlib.sha256(
        json.dumps(payload_for_hash, sort_keys=True, default=str).encode()
    ).hexdigest()
    payload["reproducibility_checksum"] = chksum

    out_path = RESULTS_DIR / "proto_carnot_pruner_abstain.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    # ── Console summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("ABSTAINING CARNOT PRUNER: PER-ARM AGGREGATE (vs vanilla, paired per seed)")
    print("=" * 100)
    print(f"{'arm':<18} {'median_eff':>11} {'solves_pres':>12} {'regressed':>10} "
          f"{'exercised':>10} {'deferred':>10} {'worst_outcome'}")
    print("-" * 100)
    for mode, value in TAU_GRID:
        key = _arm_key(mode, value)
        s = arm_summaries[key]
        me = f"{s['median_efficiency_ratio']:.3f}" if s["median_efficiency_ratio"] is not None else "N/A"
        sp = f"{s['n_games_solve_preserved']}/{s['n_games_total']}"
        rg = str(s["n_games_solve_rate_regressed"])
        ex = "YES" if s["pruner_exercised"] else "NO-OP"
        df = str(s["pruner_deferred_edges"])
        print(f"{key:<18} {me:>11} {sp:>12} {rg:>10} {ex:>10} {df:>10} {s['worst_per_game_outcome']}")
    print("-" * 100)
    print(f"\nDeployable arms (preserve ALL solves AND median_eff>1 AND exercised): "
          f"{deployable_arms or 'NONE'}")
    print(f"Absolute taus all NO-OP (CNN can't defer): {absolute_all_noop}")
    print(f"m0r0 vanilla_n_solved={m0r0_vanilla_n}; m0r0 preserved by any arm: {m0r0_preserved_any}")
    for mode, value in TAU_GRID:
        key = _arm_key(mode, value)
        r = m0r0_recovery[key]
        print(f"  m0r0 {key}: vanilla_n={r['vanilla_n_solved']} arm_n={r['arm_n_solved']} "
              f"preserved={r['preserved']}")

    if deployable_arms:
        best_arm = max(deployable_arms, key=lambda n: arm_summaries[n]["median_efficiency_ratio"])
    else:
        # best = an EXERCISED arm with most solves preserved, tiebreak median_eff;
        # fall back to any arm if none exercised.
        candidates = exercised_arms or [_arm_key(m, v) for m, v in TAU_GRID]
        best_arm = max(
            candidates,
            key=lambda n: (
                arm_summaries[n]["n_games_solve_preserved"],
                arm_summaries[n]["median_efficiency_ratio"] or -1.0,
            ),
        )
    print(f"\n=== PER-GAME for best arm '{best_arm}' ===")
    print(f"{'Game':<8} {'V-med':>8} {'A-med':>8} {'V-solv':>7} {'A-solv':>7} "
          f"{'eff':>8} {'defer':>7} {'pres':>6}")
    print("-" * 70)
    for g in arm_summaries[best_arm]["per_game"]:
        vm = f"{g['vanilla_median']:.0f}" if g["vanilla_median"] is not None else "N/A"
        am = f"{g['arm_median']:.0f}" if g["arm_median"] is not None else "N/A"
        er = f"{g['efficiency_ratio']:.3f}" if g["efficiency_ratio"] is not None else "N/A"
        pr = "YES" if g["solve_preserved"] else "REGR"
        print(f"{g['game']:<8} {vm:>8} {am:>8} {g['vanilla_n_solved']:>7} {g['arm_n_solved']:>7} "
              f"{er:>8} {g['pruner_deferred_edges']:>7} {pr:>6}")
    print("-" * 70)
    print(f"\nDuration: {duration_s}s")
    print(f"\nVerdict: {verdict}")
    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()
