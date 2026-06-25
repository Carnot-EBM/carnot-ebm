"""proto_carnot_pruner.py — Carnot frame-change verifier as pruner on just-explore.

Measures whether reordering just-explore's untested edges by the Carnot
frame-change-verifier score (LiveActionEffectScorer) cuts the action count to
the first level-up vs vanilla random.choice — on the 9 games just-explore
already solves at budget 2000.

Two arms, same budget=2000, same offline arcade:
  - vanilla: random.choice(untested_edges), N=5 seeds
  - carnot_pruned: argmax(untested_edges, key=carnot_score), N=5 seeds (with
    random tiebreak so seeds still matter)

Artifact: results/proto_carnot_pruner.json
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

RANDOM_SEED = 4731
N_SEEDS = 5
BUDGET = 2000
SOLVED_GAMES = ["ar25", "cd82", "ft09", "lp85", "m0r0", "r11l", "s5i5", "sp80", "vc33"]


# ─── 1. Load just-explore modules WITHOUT their broken __init__.py ─────────────
def _load_je_modules() -> dict[str, Any]:
    """Load just-explore structs, tracing, recorder, agent, graph_explorer, heuristic_agent.

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


# ─── 3. Load the live action-effect scorer ─────────────────────────────────────
print("Loading Carnot LiveActionEffectScorer...", flush=True)
_CARNOT_SCORER = load_live_action_effect_scorer(REPO_ROOT)
if _CARNOT_SCORER is None:
    print("ERROR: Carnot scorer failed to load (no transition corpus / no CNN checkpoint)")
    sys.exit(1)
print(f"  Scorer loaded: memory={_CARNOT_SCORER.memory is not None}, cnn={_CARNOT_SCORER.cnn_scorer is not None}")


# ─── 4. Pruned GraphExplorer: injects Carnot scorer into choose_edge ───────────
class CarnotPrunedGraphExplorer(OrigGraphExplorer):
    """Subclass of GraphExplorer that replaces random.choice(untested_edges)
    with argmax over Carnot frame-change scores.

    WHY: The only change to just-explore logic is the selection of WHICH
    untested edge to try next. All graph building, transition recording, and
    BFS frontier logic is left completely intact.

    Threading current frame info through: the HeuristicAgent sets
    `self.carnot_current_frame` and `self.carnot_edge_to_action` on the
    pruned agent before calling choose_edge, so the pruner can score each edge.
    """

    def __init__(self, *args: Any, carnot_scorer: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.carnot_scorer = carnot_scorer
        # Will be set by the agent before each choose_edge call:
        self._current_frame: Any = None
        self._edge_to_action_fn: Any = None  # callable(edge_idx) -> ArcAction
        # Diagnostics: track when we actually changed the order
        self.pruner_total_choices = 0
        self.pruner_changed_count = 0

    def choose_edge(self, node: Any, return_reasoning: bool = False) -> Any:
        """Override: score untested edges with Carnot, take argmax.

        WHY: edge_idx < num_click_actions → segment click (ACTION6 at centroid)
             edge_idx >= num_click_actions → directional action (ACTION1..5)
        We convert each edge_idx to an ArcAction, score it, and pick the max.
        Tiebreaks are random (so seeds still matter and vary the run).
        """
        node_info = self._nodes[node]
        if node_info.has_open_group(self.active_group):
            untested_edges = []
            for group_id in range(self.active_group + 1):
                untested_edges.extend(node_info.group2remaining_candidate_ids[group_id])
            if not untested_edges:
                raise ValueError("No untested edges in the current group while the group is open")

            self.pruner_total_choices += 1
            random_choice = random.choice(untested_edges)  # what vanilla would pick

            if (
                self.carnot_scorer is not None
                and self._current_frame is not None
                and self._edge_to_action_fn is not None
                and len(untested_edges) > 1
            ):
                # Score each untested edge
                scores = []
                for e in untested_edges:
                    arc_action = self._edge_to_action_fn(e)
                    if arc_action is not None:
                        s = float(self.carnot_scorer.candidate_score(self._current_frame, arc_action))
                    else:
                        s = 0.0
                    scores.append((s, e))

                # Check if scores vary (pruner actually fires)
                score_vals = [s for s, _ in scores]
                scores_vary = len(set(round(s, 8) for s in score_vals)) > 1
                max_score = max(score_vals)

                # argmax with random tiebreak among max-score edges
                max_edges = [e for s, e in scores if abs(s - max_score) < 1e-9]
                carnot_choice = random.choice(max_edges)

                if scores_vary and carnot_choice != random_choice:
                    self.pruner_changed_count += 1

                edge_idx = carnot_choice
                reasoning = (
                    f"Carnot-pruned: chose edge {edge_idx} "
                    f"(score={max_score:.4f}) from {len(untested_edges)} untested edges "
                    f"(scores_varied={scores_vary})\n"
                )
            else:
                # Fallback to vanilla random if scorer not ready
                edge_idx = random_choice
                reasoning = f"Vanilla fallback: chose edge {edge_idx} (scorer not ready)\n"
        else:
            # Lowest-distance traversal path (not scoring this path - it uses tested edges)
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
        else:
            return edge_idx


# ─── 5. Pruned HeuristicAgent: injects Carnot scorer and edge->action mapping ──
class CarnotPrunedHeuristicAgent(OrigHeuristicAgent):
    """HeuristicAgent subclass that uses CarnotPrunedGraphExplorer.

    WHY: The only changes are:
    1. graph_explorer is replaced with CarnotPrunedGraphExplorer
    2. Before each choose_edge call (inside choose_action), we set the
       current frame and edge_to_action mapping on the pruned explorer.

    All other logic (segmentation, graph building, BFS, transition recording)
    is completely unchanged from the original.
    """

    def __init__(self, *args: Any, carnot_scorer: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Replace the graph explorer with the pruned version
        self.graph_explorer = CarnotPrunedGraphExplorer(
            verbose_level=self.verbose_level,
            n_groups=self.N_GROUPS,
            carnot_scorer=carnot_scorer,
        )
        self._carnot_scorer = carnot_scorer
        # Will be set during choose_action before the explorer chooses
        self._last_segmented_frame: Any = None
        self._last_frame_segments: list[Any] = []
        self._last_arrow_actions: list[Any] = []
        self._last_num_click_actions: int = 0
        self._last_je_frame: Any = None  # JE FrameData for the current step

    def _edge_to_arc_action(self, edge_idx: int) -> ArcAction | None:
        """Convert a just-explore edge_idx to a Carnot ArcAction for scoring.

        WHY: edge_idx maps to either:
          - A segment click: edge_idx < num_click_actions
            → centroid of segment edge_idx in segmented_frame
          - A directional action: edge_idx >= num_click_actions
            → arrow_actions[edge_idx - num_click_actions] (ACTION1..5)

        We use the segment centroid (median of pixel positions) for click scoring
        because that's the "expected" click location for the edge. The actual click
        uses a random point within the segment, but the centroid best represents
        where the scorer should evaluate.
        """
        if edge_idx < self._last_num_click_actions:
            # Click action: find centroid of segment edge_idx
            if self._last_segmented_frame is None:
                return None
            seg_mask = self._last_segmented_frame == edge_idx
            seg_points = np.argwhere(seg_mask)
            if len(seg_points) == 0:
                return None
            # centroid: mean row/col
            ys = seg_points[:, 0]
            xs = seg_points[:, 1]
            cy = int(np.median(ys))
            cx = int(np.median(xs))
            return ArcAction(action_id=6, data={"x": cx, "y": cy}, source="segment_centroid")
        else:
            # Arrow action
            arrow_idx = edge_idx - self._last_num_click_actions
            if arrow_idx >= len(self._last_arrow_actions):
                return None
            je_arrow = self._last_arrow_actions[arrow_idx]
            # je_arrow is a JEGameAction enum member like GameAction.ACTION1
            aid = je_arrow.value  # int 1..5
            if 1 <= aid <= 5:
                return ArcAction(action_id=aid, data=None, source="directional_action")
            return None

    def choose_action(self, frames: Any, latest_frame: Any) -> Any:
        """Wrap choose_action to inject Carnot scoring context before choose_edge.

        WHY: choose_action computes segmented_frame, frame_segments, arrow_actions,
        num_click_actions BEFORE calling self.graph_explorer.choose_edge. We need
        those values to map edge_idx -> ArcAction for scoring. We intercept by
        wrapping the graph_explorer.choose_edge call to capture context first.

        Strategy: subclass choose_edge call to receive scoring context by
        monkey-patching the graph_explorer's context attributes before the
        parent's choose_action calls choose_edge.

        We override by wrapping the graph_explorer's choose_edge to intercept the
        'hashed_frame' call, but that's complex. Instead, we override
        choose_action's internals by intercepting at the agent level:
        We use a different approach - we set context on the explorer just
        BEFORE the parent class calls choose_edge, by hooking into the
        explorer's choose_edge via its context attributes.

        Implementation: We monkey-patch the explorer's context attributes
        at key points by overriding only the graph_explorer interaction.
        Since we can't easily intercept mid-function, we instead capture
        the frame data that was available in the PREVIOUS step for scoring
        (the frame is stable within a step until choose_edge is called).

        ACTUAL approach: We track the frame state ourselves in parallel
        with the parent's internal state. When choose_action calls into
        self.graph_explorer.choose_edge, our pruned explorer's choose_edge
        will read self.graph_explorer._current_frame (set by us beforehand).

        Since the parent's choose_action computes segmented_frame in a
        specific order, we intercept via a wrapper on self.frame_processor.segment_frame
        to capture the output, and similarly capture arrow_actions.
        """
        # Reset context before each choose_action
        self._last_segmented_frame = None
        self._last_frame_segments = []
        self._last_arrow_actions = []
        self._last_num_click_actions = 0
        self._last_je_frame = latest_frame

        # Wrap frame_processor.segment_frame to capture segmented_frame + segments
        original_segment_frame = self.frame_processor.segment_frame

        def capturing_segment_frame(frame_np: Any) -> Any:
            result = original_segment_frame(frame_np)
            # Parent calls this twice: once for status bars (level_up),
            # once for the main segmentation. We capture the second call.
            # We distinguish by checking if status_bar_mask is being set:
            # The parent calls segment_frame for status bars ONLY when self.level_up is True.
            # We always capture both but the second call's result is what matters.
            seg_frame, seg_list = result
            # Store latest (will be overwritten if called again = fine, last one is main)
            self._last_segmented_frame = seg_frame
            self._last_frame_segments = seg_list
            self._last_num_click_actions = len(seg_list) if len(seg_list) > 0 else 0
            return result

        self.frame_processor.segment_frame = capturing_segment_frame

        # Wrap the parent's choose_action and capture arrow_actions.
        # Since arrow_actions is a local variable in choose_action, we can't
        # intercept it directly. Instead, we set the graph_explorer context
        # right before the graph_explorer.choose_edge call.
        #
        # The parent calls self.graph_explorer.choose_edge(hashed_frame, return_reasoning=True)
        # We intercept by wrapping the explorer's choose_edge so it gets the
        # frame context we've captured:

        original_choose_edge = self.graph_explorer.choose_edge

        def wrapped_choose_edge(node: Any, return_reasoning: bool = False) -> Any:
            # At this point, segment_frame has been called (we captured the result)
            # We need arrow_actions but it's local to choose_action. We reconstruct:
            # The parent sets arrow_actions by iterating available_actions.
            # We read from latest_frame.available_actions to reconstruct arrow_actions.
            avail = getattr(latest_frame, "available_actions", []) or []
            SIMPLE_ACTION_ID2GAME_ACTION = OrigHeuristicAgent.SIMPLE_ACTION_ID2GAME_ACTION
            arrow_actions = []
            for action_id in avail:
                if action_id in SIMPLE_ACTION_ID2GAME_ACTION:
                    arrow_actions.append(SIMPLE_ACTION_ID2GAME_ACTION[action_id])
            self._last_arrow_actions = arrow_actions

            # Set the context on the pruned graph explorer
            self.graph_explorer._current_frame = latest_frame
            self.graph_explorer._edge_to_action_fn = self._edge_to_arc_action

            return original_choose_edge(node, return_reasoning=return_reasoning)

        self.graph_explorer.choose_edge = wrapped_choose_edge

        try:
            result = super().choose_action(frames, latest_frame)
        finally:
            # Restore originals to avoid leaks
            self.frame_processor.segment_frame = original_segment_frame
            self.graph_explorer.choose_edge = original_choose_edge

        return result


# ─── 6. Frame conversion helpers (identical to proto_just_explore_diag.py) ─────
def _our_raw_to_je_fd(raw: Any, game_id: str, start_level: int) -> Any:
    """Convert Carnot FrameDataRaw to just-explore FrameData."""
    grid = grid_of(raw)  # (64, 64) int16
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
    """Convert JE GameAction to our (label_str, OurGameAction, data_or_None)."""
    aid = je_action.value  # int 0..6

    if aid == 0:
        return "RESET", OurGameAction.RESET, None

    if aid == 6:
        ad = je_action.action_data
        x, y = int(ad.x), int(ad.y)
        return json.dumps({"action": 6, "x": x, "y": y}), OurGameAction.ACTION6, {"x": x, "y": y}

    our_ga = getattr(OurGameAction, f"ACTION{aid}")
    return json.dumps({"action": aid}), our_ga, None


# ─── 7. Run one game at one budget (generic: accepts agent factory) ────────────
def run_one_game(
    game_id: str,
    budget: int,
    arc: Any,
    seed: int,
    pruner_mode: bool,
    carnot_scorer: Any = None,
) -> dict:
    """Run one game with either vanilla or Carnot-pruned agent.

    WHY pruner_mode:
    - False: vanilla HeuristicAgent (random.choice untested edges)
    - True: CarnotPrunedHeuristicAgent (argmax Carnot score over untested edges)

    seed controls random.seed for this run. The same seed produces the same
    result for vanilla; for Carnot-pruned, the seed only affects tiebreaks.
    """
    # Seed random for reproducibility
    random.seed(seed)
    np.random.seed(seed % (2**31))

    result: dict = {
        "game": game_id,
        "budget": budget,
        "seed": seed,
        "pruner_mode": pruner_mode,
        "reached_level": 0,
        "solved": False,
        "actions_used": 0,
        "actions_to_first_levelup": None,
        "adapter_failed": False,
        "adapter_error": None,
        "pruner_total_choices": 0,
        "pruner_changed_count": 0,
    }

    try:
        sc = arc.open_scorecard()
        base_env = arc.make(game_id, scorecard_id=sc)
        env = VariantEnv(base_env, game_id, 1)
        raw = env.reset()
        start_level = _levels_completed(raw)

        if pruner_mode:
            agent = CarnotPrunedHeuristicAgent(
                card_id="pruner_card",
                game_id=game_id,
                agent_name="carnot_pruned",
                ROOT_URL="http://localhost:0",
                record=False,
                carnot_scorer=carnot_scorer,
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
        agent.minimal_step_time = 0.0  # suppress API rate-limit sleep

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

        # Capture pruner diagnostics from pruned explorer
        if pruner_mode and hasattr(agent, "graph_explorer"):
            result["pruner_total_choices"] = agent.graph_explorer.pruner_total_choices
            result["pruner_changed_count"] = agent.graph_explorer.pruner_changed_count

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
    carnot_scorer: Any = None,
) -> dict:
    """Run a game N times with different seeds, return per-seed results + aggregates."""
    per_seed: list[dict] = []
    atfl_values: list[int] = []

    for i in range(n_seeds):
        seed = base_seed + i * 37  # spread seeds
        print(f"    seed={seed}...", end=" ", flush=True)
        r = run_one_game(game_id, budget, arc, seed, pruner_mode, carnot_scorer)
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
    total_changed = sum(r.get("pruner_changed_count", 0) for r in per_seed)
    total_choices = sum(r.get("pruner_total_choices", 0) for r in per_seed)

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
        "pruner_total_choices": total_choices,
        "pruner_changed_count": total_changed,
    }


# ─── 9. Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    rng = random.Random(RANDOM_SEED)
    base_seed = rng.randint(10000, 99999)

    arc = kit.offline_arcade()

    # ── SMOKE TEST: vc33 (cheapest game, ~9 actions vanilla) ──────────────────
    print("\n=== SMOKE TEST: vc33 ===")
    smoke_vanilla = run_one_game("vc33", BUDGET, arc, seed=base_seed, pruner_mode=False, carnot_scorer=None)
    smoke_pruned = run_one_game("vc33", BUDGET, arc, seed=base_seed, pruner_mode=True, carnot_scorer=_CARNOT_SCORER)

    print(f"  vanilla:  solved={smoke_vanilla['solved']} atfl={smoke_vanilla['actions_to_first_levelup']}")
    print(f"  pruned:   solved={smoke_pruned['solved']} atfl={smoke_pruned['actions_to_first_levelup']}")
    print(f"  pruner fired: total_choices={smoke_pruned.get('pruner_total_choices')} changed={smoke_pruned.get('pruner_changed_count')}")

    if smoke_vanilla["adapter_failed"] or smoke_pruned["adapter_failed"]:
        print("SMOKE FAILED — aborting.")
        if smoke_vanilla["adapter_failed"]:
            print("Vanilla error:", smoke_vanilla["adapter_error"])
        if smoke_pruned["adapter_failed"]:
            print("Pruned error:", smoke_pruned["adapter_error"])
        sys.exit(1)

    if not smoke_vanilla["solved"]:
        print("WARNING: vanilla vc33 did not solve in smoke test! (may be stochastic — continuing)")
    print("Smoke complete. Running full 9-game comparison...\n")

    # ── FULL RUN: vanilla + pruned, N=5 seeds each ─────────────────────────────
    all_vanilla: dict[str, dict] = {}
    all_pruned: dict[str, dict] = {}

    for game in SOLVED_GAMES:
        print(f"\n=== {game} ===")
        print(f"  [vanilla]")
        vanilla_result = run_game_n_seeds(
            game, BUDGET, arc, base_seed=base_seed + hash(game) % 10000,
            n_seeds=N_SEEDS, pruner_mode=False, carnot_scorer=None
        )
        all_vanilla[game] = vanilla_result

        print(f"  [carnot_pruned]")
        pruned_result = run_game_n_seeds(
            game, BUDGET, arc, base_seed=base_seed + hash(game) % 10000,
            n_seeds=N_SEEDS, pruner_mode=True, carnot_scorer=_CARNOT_SCORER
        )
        all_pruned[game] = pruned_result

        v_atfl = vanilla_result["median_atfl"]
        p_atfl = pruned_result["median_atfl"]
        print(f"  vanilla median_atfl={v_atfl}  pruned median_atfl={p_atfl}")
        if v_atfl is not None and p_atfl is not None:
            ratio = (v_atfl / p_atfl) ** 2 if p_atfl > 0 else None
            print(f"  efficiency_ratio={(ratio):.3f}" if ratio else "  efficiency_ratio=N/A")

    # ── Per-game results ───────────────────────────────────────────────────────
    per_game: list[dict] = []
    action_reductions: list[float] = []
    efficiency_ratios: list[float] = []
    solve_preserved_count = 0
    solve_lost_games: list[str] = []
    total_pruner_choices = 0
    total_pruner_changed = 0

    for game in SOLVED_GAMES:
        v = all_vanilla[game]
        p = all_pruned[game]

        v_median = v["median_atfl"]
        p_median = p["median_atfl"]

        if v_median is not None and p_median is not None and p_median > 0:
            action_reduction = float(v_median - p_median)
            efficiency_ratio = float((v_median / p_median) ** 2)
        elif v_median is not None and p_median is None:
            # Pruned lost the solve entirely
            action_reduction = None  # type: ignore
            efficiency_ratio = None  # type: ignore
        else:
            action_reduction = None  # type: ignore
            efficiency_ratio = None  # type: ignore

        # Solve preserved: pruned arm must still solve (any seed)
        solve_preserved = p["any_solved"]
        if solve_preserved:
            solve_preserved_count += 1
        elif v["any_solved"]:
            # vanilla solved but pruned did not
            solve_lost_games.append(game)

        if action_reduction is not None:
            action_reductions.append(action_reduction)
        if efficiency_ratio is not None:
            efficiency_ratios.append(efficiency_ratio)

        total_pruner_choices += p["pruner_total_choices"]
        total_pruner_changed += p["pruner_changed_count"]

        per_game.append({
            "game": game,
            "vanilla_actions_median": v_median,
            "vanilla_actions_min": v["min_atfl"],
            "vanilla_n_solved": v["n_solved"],
            "carnot_actions_median": p_median,
            "carnot_actions_min": p["min_atfl"],
            "carnot_n_solved": p["n_solved"],
            "action_reduction": action_reduction,
            "efficiency_ratio": efficiency_ratio,
            "solve_preserved": solve_preserved,
            "pruner_total_choices": p["pruner_total_choices"],
            "pruner_changed_count": p["pruner_changed_count"],
        })

    # ── Aggregates ─────────────────────────────────────────────────────────────
    median_action_reduction = float(median(action_reductions)) if action_reductions else None
    median_efficiency_ratio = float(median(efficiency_ratios)) if efficiency_ratios else None

    # pruner_exercised: scores varied AND changed the order on >0 steps
    pruner_exercised = (total_pruner_changed > 0 and total_pruner_choices > 0)
    pruner_fire_rate = (total_pruner_changed / total_pruner_choices) if total_pruner_choices > 0 else 0.0

    # Smoke verification values
    smoke_pruner_fired = (
        smoke_pruned.get("pruner_changed_count", 0) > 0
        and smoke_pruned.get("pruner_total_choices", 0) > 0
    )

    # ── honest_verdict ─────────────────────────────────────────────────────────
    all_solves_preserved = (len(solve_lost_games) == 0)
    genuine_efficiency_win = (
        median_efficiency_ratio is not None
        and median_efficiency_ratio > 1.0
        and pruner_exercised
        and all_solves_preserved
    )

    if genuine_efficiency_win:
        verdict = (
            f"success: Carnot pruner reduces actions "
            f"(median_action_reduction={median_action_reduction:.1f}, "
            f"median_efficiency_ratio={median_efficiency_ratio:.3f}), "
            f"{solve_preserved_count}/{len(SOLVED_GAMES)} solves preserved, "
            f"pruner exercised ({total_pruner_changed}/{total_pruner_choices} choices changed)"
        )
    elif not pruner_exercised:
        verdict = (
            f"complete: pruner NOT exercised "
            f"(total_choices={total_pruner_choices}, changed={total_pruner_changed}) — "
            f"scorer returned uniform scores or edge mapping failed; result is a false null"
        )
    elif not all_solves_preserved:
        verdict = (
            f"complete: pruner LOST solves on games={solve_lost_games}; "
            f"median_efficiency_ratio={median_efficiency_ratio}; "
            f"solve regression is a real finding — pruner is not a drop-in replacement"
        )
    elif median_efficiency_ratio is not None and median_efficiency_ratio <= 1.0:
        verdict = (
            f"complete: pruner exercised but NO action reduction "
            f"(median_efficiency_ratio={median_efficiency_ratio:.3f} ≤ 1.0); "
            f"Carnot frame-change scoring does not help just-explore on these games at this budget"
        )
    else:
        verdict = (
            f"complete: pruner ran but results inconclusive "
            f"(median_efficiency_ratio={median_efficiency_ratio}, "
            f"solve_preserved={solve_preserved_count}/{len(SOLVED_GAMES)})"
        )

    duration_s = round(time.time() - t0, 2)

    # ── Artifact ───────────────────────────────────────────────────────────────
    payload = {
        "per_game": per_game,
        "median_action_reduction": median_action_reduction,
        "median_efficiency_ratio": median_efficiency_ratio,
        "n_games_solve_preserved": solve_preserved_count,
        "n_games_total": len(SOLVED_GAMES),
        "solve_lost_games": solve_lost_games,
        "pruner_exercised": pruner_exercised,
        "pruner_total_choices": total_pruner_choices,
        "pruner_changed_count": total_pruner_changed,
        "pruner_fire_rate": round(pruner_fire_rate, 4),
        "smoke_vanilla_atfl": smoke_vanilla.get("actions_to_first_levelup"),
        "smoke_pruned_atfl": smoke_pruned.get("actions_to_first_levelup"),
        "smoke_pruner_fired": smoke_pruner_fired,
        "smoke_pruner_choices": smoke_pruned.get("pruner_total_choices", 0),
        "smoke_pruner_changed": smoke_pruned.get("pruner_changed_count", 0),
        "n_seeds": N_SEEDS,
        "budget": BUDGET,
        "games": SOLVED_GAMES,
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "base_seed_used": base_seed,
        "duration_s": duration_s,
        "methodology_note": (
            "Carnot frame-change verifier (LiveActionEffectScorer: PersistentAEM + SmallFrameChangeCNN) "
            "scores each untested edge in just-explore's GraphExplorer.choose_edge. "
            "Edge->ArcAction mapping: edge_idx < num_click_actions -> segment centroid click (ACTION6 x,y); "
            "edge_idx >= num_click_actions -> directional arrow action (ACTION1..5). "
            "Pruner replaces random.choice with argmax, random tiebreak for equal scores. "
            "All other just-explore logic (segmentation, graph, BFS, transition recording) unchanged. "
            "Same budget, same games, same offline arcade variant-1 for both arms. CPU-only."
        ),
    }

    payload_for_hash = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    chksum = hashlib.sha256(
        json.dumps(payload_for_hash, sort_keys=True, default=str).encode()
    ).hexdigest()
    payload["reproducibility_checksum"] = chksum

    out_path = RESULTS_DIR / "proto_carnot_pruner.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CARNOT PRUNER vs VANILLA: RESULTS")
    print("=" * 70)
    print(f"{'Game':<8} {'V-median':>10} {'P-median':>10} {'Reduction':>10} {'Eff-ratio':>10} {'Preserved':>10}")
    print("-" * 70)
    for g in per_game:
        vm = f"{g['vanilla_actions_median']:.0f}" if g["vanilla_actions_median"] is not None else "N/A"
        pm = f"{g['carnot_actions_median']:.0f}" if g["carnot_actions_median"] is not None else "N/A"
        ar = f"{g['action_reduction']:.1f}" if g["action_reduction"] is not None else "N/A"
        er = f"{g['efficiency_ratio']:.3f}" if g["efficiency_ratio"] is not None else "N/A"
        sp = "YES" if g["solve_preserved"] else "LOST"
        print(f"{g['game']:<8} {vm:>10} {pm:>10} {ar:>10} {er:>10} {sp:>10}")
    print("-" * 70)
    print(f"{'MEDIAN':<8} {'':<10} {'':<10} {median_action_reduction if median_action_reduction is not None else 'N/A':>10.1f} {median_efficiency_ratio if median_efficiency_ratio is not None else 'N/A':>10.3f}")
    print()
    print(f"Solves preserved: {solve_preserved_count}/{len(SOLVED_GAMES)}")
    print(f"Solve lost games: {solve_lost_games}")
    print(f"Pruner exercised: {pruner_exercised} ({total_pruner_changed}/{total_pruner_choices} choices changed, fire_rate={pruner_fire_rate:.3f})")
    print(f"Smoke: vanilla_atfl={smoke_vanilla.get('actions_to_first_levelup')} pruned_atfl={smoke_pruned.get('actions_to_first_levelup')}")
    print(f"Duration: {duration_s}s")
    print(f"\nVerdict: {verdict}")
    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()
