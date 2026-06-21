"""CarnotAgent — an ARC Prize 2026 / ARC-AGI-3 competition agent (the Kaggle
submission shape). The competition runs an Agent subclass step-wise and OFFLINE:
each turn the harness calls `choose_action(frames, latest_frame) -> GameAction` and
`is_done(frames, latest_frame) -> bool`. No internet at eval time.

This module is FRAMEWORK-AGNOSTIC so it is testable offline without the
ARC-AGI-3-Agents package: the decision logic lives in `CarnotAgentPolicy`, and
`make_carnot_agent(Agent)` adapts it onto the real `Agent` base class at submission
time (a thin subclass). The validation harness `scripts/arc_competition_validate.py`
drives the policy through our offline sims (environment_files), mimicking the
competition loop, to confirm the agent scores BEFORE any submission.

Policy (v1 — recognize-and-replay): the harness gives the agent its `game_id` at
construction. For a game we have an OFFLINE-REPRODUCED solution (the 13-level
registry), the agent replays that banked action sequence (Mode-1; no search, no
internet — ideal for the offline eval IF the eval games == the public 25). For an
UNKNOWN game (hidden eval), it falls back to a step-wise systematic explorer
(navigate-by-RESET-replay, take untested salient actions) — the online form of
graph_explore_solve_v2. The replay path is the validated v1; the explore fallback is
the generalizing path for held-out games.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import carnot.agentic.arc_strategy_router as arc_strategy_router
from carnot.agentic.arc_world_model_dsl import ObjectDeltaModel
from carnot.agentic.arc_llm_reinduction import (
    MAX_REFINEMENT_ROUNDS,
    execute_bounded_llm_reinduction,
)
from carnot.agentic.arc_world_model_trust_energy import (
    HIDDEN_STATE_GAME_IDS,
    WorldModelCandidate,
    select_trusted_world_model,
)

REPO = Path(__file__).resolve().parents[3]

# the 11 reproduced games and their target (offline-reproduced) level
CLAIMED = {
    "r11l": 1,
    "lp85": 3,
    "ls20": 1,
    "wa30": 1,
    "cd82": 1,
    "sp80": 1,
    "su15": 1,
    "tu93": 1,
    "cn04": 1,
    "m0r0": 1,
    "sk48": 1,
}
MAX_ACTIONS = 200
# value_weight=0.0 (reverted from 5.0, 2026-06-20): the v3 cross-game value head (LOO-AUROC 0.674,
# loaded by load_cross_game_value_head) IS wired and used as a frontier TIEBREAKER. But weight>0 makes
# the search pay the (now richer/more-expensive v3) value eval on EVERY node -> measured REGRESSION:
# value_weight=5 was slower than bare BFS and solved fewer games in bounded time (the 25-game sim timed
# out 20/25; A1's own benchmark delta=0.0; bridge w5 regressed 8->6). The head's offline LOO of 0.674 is
# NOT yet shown to help LIVE routing enough to justify the per-node cost. .416 re-measures the submitted-
# default solve-rate with the v3 head at weight>0 (+ a possible lazy/cheap eval); raise value_weight ONLY
# if it beats bare-BFS on solve-rate AND finishes in budget. Until then: v3 head loaded, weight 0 (cheap).
SUBMITTED_VALUE_WEIGHT = 0.0
# Exp 4524 measured the fixed 8-game local gate and found the run-to-completion control banks only L1 on
# every CORE solve; target 1 stops at the scored gate target instead of burning the post-level-up tail.
SUBMITTED_TARGET_LEVELS = 1
# Smart grace-period early-stop: stop this many moves after the LAST level-up if no new level appears
# (cuts the fruitless post-solve tail, WITHOUT capping the configured scored target). None = disabled.
SUBMITTED_EARLY_STOP_GRACE: Optional[int] = None
SUBMITTED_SEARCH_MODE = "depth_first_ride"
SUBMITTED_GRAPH_EXPLORE_BUDGET = 80
SUBMITTED_ROUTED_EXPLORE_BUDGET = 24
SUBMITTED_LAZY_VALUE_TOP_K = 4
SUBMITTED_FRONTIER_BATCH_SIZE: int | str = 1
SUBMITTED_NAVIGATION_COST_TIEBREAK = False
_DEFAULT_VALUE_HEAD = object()


def load_solutions() -> dict[str, list[dict]]:
    """game-short -> [{"action": int, "data": {x,y}|None}] for every banked solution,
    via the metaharness's loader (single source of truth for the trajectories)."""
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py")
    )
    mh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mh)  # type: ignore
    sols: dict[str, list[dict]] = {}
    for short in CLAIMED:
        src = mh.RESOLVED_ARTIFACTS.get(short, mh.GAME_ARTIFACTS.get(short))
        if not src:
            continue
        steps = []
        for a in mh.load_actions(src):
            aid, data = mh.normalize(a)
            if aid is not None:
                steps.append({"action": int(aid), "data": data})
        sols[short] = steps
    return sols


def _level_of(frame: Any) -> int:
    if frame is None:
        return 0
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed

    try:
        return _levels_completed(frame)
    except Exception:
        return 0


def _action_key(action_id: int, data: Any) -> tuple:
    if int(action_id) == 6 and isinstance(data, dict) and "x" in data and "y" in data:
        return (6, int(data["x"]), int(data["y"]))
    return (int(action_id),)


def _frame_only_mechanic_hint(frame: Any) -> Optional[str]:
    """Cheap frame-only mechanic hint used when the live route has no registry class.

    The full probe-based detector lives at the I/O edge (`scripts/arc3_frame_induction.py`).
    The submitted policy cannot touch `env._game`, so this hint only inspects the rendered
    grid and returns a positive class when a dense lower-board editor palette is visible.
    Unknown frames return None and keep the registry/default route.
    """
    if frame is None:
        return None
    try:
        import numpy as np
        from carnot.agentic.arc_agi3_world_model import grid_of

        grid = grid_of(frame)
        if getattr(grid, "shape", None) is None or grid.ndim != 2:
            return None
        h, w = grid.shape
        lower = grid[int(h * 0.55) :, :]
        if lower.size == 0:
            return None
        vals, counts = np.unique(lower, return_counts=True)
        bg = int(vals[counts.argmax()])
        active = lower != bg
        density = float(active.mean())
        columns = int(np.count_nonzero(active.any(axis=0)))
        rows = int(np.count_nonzero(active.any(axis=1)))
        if density >= 0.08 and columns >= max(12, w // 5) and rows >= 5:
            return "program_editor"
    except Exception:
        return None
    return None


def _route_explore_budget(route: dict[str, Any]) -> int:
    if route.get("uses_goal_distance_heuristic") is False:
        return SUBMITTED_ROUTED_EXPLORE_BUDGET
    return SUBMITTED_GRAPH_EXPLORE_BUDGET


class StepwiseExplorer:
    """Generic, game-AGNOSTIC step-wise solver — the competition asset (eval games are
    UNSEEN, so replay is useless; this is the only thing that scores). It is
    `graph_explore_solve_v2` turned inside-out into an EVENT LOOP the harness drives one
    action per turn: maintain a state-transition graph, expand the current state's
    untested SALIENT actions DEPTH-first (no navigation cost), and when the current
    state is exhausted or dead-ends, navigate to the shallowest frontier state with
    untested actions by RESET + replaying its path (deepcopy/jump is impossible live;
    RESET-replay is the only navigation). Action-efficient (scoring rewards fewer
    actions: min(human/agent,1)^2). Stops at the first level-up (+1 incremental unit) or
    when fully explored."""

    def __init__(
        self,
        target_levels: int = 1,
        max_depth: int = 45,
        hud_mask=None,
        value_head=None,
        value_weight: float = 0.0,
        search_mode: str = "depth_first_ride",
        online_discriminative: bool = True,
        discriminative_featurizer=None,
        discriminative_min_positives: int = 3,
        discriminative_min_negatives: int = 3,
        discriminative_fit_iters: int = 120,
        discriminative_refit_every: int = 4,
        discriminative_prune_threshold: float = 0.12,
        frame_change_scorer: Any | None = None,
        frame_change_prune_threshold: float | None = None,
        action_prior: Any | None = None,
        action_prior_prune_quantile: float | None = None,
        adaptive_budget_threshold: float | None = None,
        adaptive_budget_value_head: Any | None = None,
        adaptive_budget_noop_threshold: float = 0.5,
        lazy_value_top_k: int = SUBMITTED_LAZY_VALUE_TOP_K,
        early_stop_grace: Optional[int] = None,
        frontier_batch_size: int | str | None = SUBMITTED_FRONTIER_BATCH_SIZE,
        navigation_cost_tiebreak: bool = SUBMITTED_NAVIGATION_COST_TIEBREAK,
    ) -> None:
        self.hud_mask = hud_mask  # E1: mask step-counter cells out of node identity
        # BRIDGE: a frame-only cross-game value head (frame -> predicted steps-to-next-level-up, LOWER =
        # closer). Trained offline on ALL banked solves (the offline->live distillation). The frontier is
        # ordered A*-style: priority = depth + value_weight*value. value_weight=0 (default) -> pure BFS
        # (value only breaks ties; can't regress). value_weight>0 -> the value head NUDGES the search
        # toward predicted-closer states (the A* routing that unlocked cn04 in graph_explore_solve_v2).
        self.value_head = value_head
        self.value_weight = float(value_weight)
        self.lazy_value_top_k = max(1, int(lazy_value_top_k))
        self._value_cache: dict[str, float] = {}
        self._value_head_evals = 0
        self._value_cache_hits = 0
        self.frame_change_scorer = frame_change_scorer
        self.frame_change_prune_threshold = frame_change_prune_threshold
        self.action_prior = action_prior
        self.action_prior_prune_quantile = action_prior_prune_quantile
        self.adaptive_budget_threshold = adaptive_budget_threshold
        self.adaptive_budget_value_head = adaptive_budget_value_head
        self.adaptive_budget_noop_threshold = float(adaptive_budget_noop_threshold)
        self._adaptive_budget_history: list[dict[str, Any]] = []
        self._adaptive_budget_commit_count = 0
        self._adaptive_budget_expanded_count = 0
        self._adaptive_budget_candidates_skipped = 0
        self.online_discriminative = bool(online_discriminative)
        self.discriminative_featurizer = discriminative_featurizer
        self.discriminative_min_positives = max(1, int(discriminative_min_positives))
        self.discriminative_min_negatives = max(1, int(discriminative_min_negatives))
        self.discriminative_fit_iters = max(1, int(discriminative_fit_iters))
        self.discriminative_refit_every = max(1, int(discriminative_refit_every))
        self.discriminative_prune_threshold = float(discriminative_prune_threshold)
        self.online_discriminator = None
        self._disc_X: list[list[float]] = []
        self._disc_y: list[float] = []
        self._disc_seen: set[tuple[int, str]] = set()
        self._disc_samples_since_fit = 0
        self._disc_negative_sources: dict[str, int] = {}
        self._disc_fit_count = 0
        self._disc_frontier_pruned = 0
        # SEARCH MODE: "depth_first_ride" (default, proven) rides the current branch depth-first (action-
        # efficient; load-bearing for the deep wins lp85/sp80). "best_first" ALWAYS expands the globally-
        # best A*-value frontier (depth + value_weight*value) -- this is the graph_explore_solve_v2 search
        # form where the cross-game value head's routing actually helped (it unlocked cn04). best_first
        # only beats the ride when the value head is good enough to route the deep wins itself; measure it.
        self.search_mode = search_mode
        self.graph: dict[
            str, dict
        ] = {}  # hash -> {"path": [...], "untested": [...], "value": float}
        self.root: Optional[str] = None
        self.cur: Optional[str] = None
        self.start_level: Optional[int] = None
        self.best_level = 0
        self.target_levels = target_levels  # stop after this many levels beyond start
        self.max_depth = max_depth  # cap DFS branch length -> forces backtrack
        self.pending: list[dict] = []  # queued nav/probe actions
        self.awaiting: Optional[dict] = None  # last probe, to attribute its result
        self.explored_out = False
        self.frontier_batch_size = self._normalize_frontier_batch_size(frontier_batch_size)
        self.navigation_cost_tiebreak = bool(navigation_cost_tiebreak)
        # Smart grace-period early-stop (does NOT cap levels). After reaching >=1 level, keep searching
        # for the next; stop only if no NEW level within `early_stop_grace` moves of the last level-up.
        # Consecutive level-ups reset the window, so multi-level games are NOT capped -- only the fruitless
        # tail after the last findable level is cut (that tail destroys the (human/actions)^2 score: e.g.
        # lp85 reaches L1 at action 20 but ran to 7792 hunting unreachable deeper levels). None = disabled.
        self.early_stop_grace = early_stop_grace
        self._early_stop_level_mark = 0
        self._early_stop_frame_mark = 0
        self.early_stopped = False
        self.adj: dict[str, list] = {}  # known forward edges: hash -> [(action_dict, next_hash)]
        self._nav_attempts = 0
        self._nav_exact_hits = 0
        self._nav_partial_hits = 0
        self._nav_reset_fallbacks = 0
        self._nav_edges_recorded = 0
        self._nav_forward_steps = 0
        self._nav_reset_replay_steps = 0
        self.goal_bias = None
        self.goal_bias_label = ""
        self.goal_bias_lower_is_better = False
        self._goal_bias_scored = 0
        self._goal_bias_errors = 0

    @staticmethod
    def _normalize_frontier_batch_size(value: int | str | None) -> int | None:
        """REQ-ARC-FCP-4523: normalize k for the opt-in frontier batch sweep."""

        if value is None:
            return 1
        if isinstance(value, str) and value.lower() == "all":
            return None
        return max(1, int(value))

    def _disc_features(self, frame) -> Optional[list[float]]:
        if not self.online_discriminative:
            return None
        try:
            if self.discriminative_featurizer is None:
                from carnot.agentic.arc_value_learner import cross_game_features_v2

                self.discriminative_featurizer = cross_game_features_v2
            return [float(v) for v in self.discriminative_featurizer(frame)]
        except Exception:
            return None

    def _record_discriminative_sample(
        self,
        frame,
        *,
        label: int,
        source: str,
        node_hash: Optional[str] = None,
    ) -> Optional[list[float]]:
        features = self._disc_features(frame)
        if features is None:
            return None
        self._record_discriminative_features(
            features,
            label=label,
            source=source,
            node_hash=node_hash or self._hash(frame),
        )
        return features

    def _record_discriminative_features(
        self,
        features: Sequence[float] | None,
        *,
        label: int,
        source: str,
        node_hash: str,
    ) -> None:
        if features is None:
            return
        key = (int(label), node_hash)
        if key in self._disc_seen:
            return
        self._disc_seen.add(key)
        self._disc_X.append([float(v) for v in features])
        self._disc_y.append(float(label))
        self._disc_samples_since_fit += 1
        if int(label) == 0:
            self._disc_negative_sources[source] = self._disc_negative_sources.get(source, 0) + 1
        self._maybe_fit_discriminator()

    def _maybe_fit_discriminator(self) -> None:
        if not self.online_discriminative or not self._disc_y:
            return
        positives = int(sum(1 for y in self._disc_y if y >= 0.5))
        negatives = len(self._disc_y) - positives
        if positives < self.discriminative_min_positives:
            return
        if negatives < self.discriminative_min_negatives:
            return
        if (
            self.online_discriminator is not None
            and self._disc_samples_since_fit < self.discriminative_refit_every
        ):
            return
        try:
            from carnot.agentic.arc_value_learner import DiscriminativeVerifier

            verifier = DiscriminativeVerifier(lambda frame: self._disc_features(frame) or [])
            verifier.fit(
                self._disc_X,
                self._disc_y,
                iters=self.discriminative_fit_iters,
                lr=0.3,
                l2=1e-3,
            )
            self.online_discriminator = verifier
            self._disc_fit_count += 1
            self._disc_samples_since_fit = 0
        except Exception:
            return

    def _node_on_path_proba(self, node: dict) -> float:
        if self.online_discriminator is None:
            return 1.0
        features = node.get("discriminative_features")
        if features is None:
            return 1.0
        try:
            return float(self.online_discriminator.proba_features(features))
        except Exception:
            return 1.0

    def online_discriminator_diagnostics(self) -> dict[str, Any]:
        positives = int(sum(1 for y in self._disc_y if y >= 0.5))
        negatives = len(self._disc_y) - positives
        return {
            "enabled": bool(self.online_discriminative),
            "trained": self.online_discriminator is not None,
            "positive_samples": positives,
            "negative_samples": negatives,
            "negative_sources": dict(self._disc_negative_sources),
            "fit_count": int(self._disc_fit_count),
            "frontier_pruned": int(self._disc_frontier_pruned),
            "prune_threshold": float(self.discriminative_prune_threshold),
        }

    def adaptive_budget_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.adaptive_budget_threshold is not None,
            "threshold": self.adaptive_budget_threshold,
            "commit_count": int(self._adaptive_budget_commit_count),
            "expanded_count": int(self._adaptive_budget_expanded_count),
            "candidates_skipped": int(self._adaptive_budget_candidates_skipped),
            "history": list(self._adaptive_budget_history[-64:]),
        }

    def lazy_value_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.value_head is not None and self.value_weight != 0.0,
            "lazy_top_k": int(self.lazy_value_top_k),
            "cache_by_frame_hash": True,
            "value_head_evals": int(self._value_head_evals),
            "cache_hits": int(self._value_cache_hits),
            "cached_frame_hashes": int(len(self._value_cache)),
        }

    def navigation_diagnostics(self) -> dict[str, Any]:
        """SCENARIO-ARC-FCP-4516: expose whether frontier navigation avoids RESET replay."""

        hits = int(self._nav_exact_hits + self._nav_partial_hits)
        attempts = int(self._nav_attempts)
        return {
            "navigation_attempts": attempts,
            "exact_shortest_path_hits": int(self._nav_exact_hits),
            "partial_forward_walk_hits": int(self._nav_partial_hits),
            "forward_walk_hits": hits,
            "reset_replay_fallbacks": int(self._nav_reset_fallbacks),
            "forward_edges_recorded": int(self._nav_edges_recorded),
            "forward_navigation_steps": int(self._nav_forward_steps),
            "reset_replay_steps": int(self._nav_reset_replay_steps),
            "forward_walk_hit_rate": float(hits / attempts) if attempts else 0.0,
        }

    def set_goal_bias(self, goal_bias, *, label: str = "", lower_is_better: bool = False) -> None:
        """REQ-ARC-WMTE-4533/4534: install a depth-preserving goal or energy bias."""

        self.goal_bias = goal_bias
        self.goal_bias_label = str(label or "")
        self.goal_bias_lower_is_better = bool(lower_is_better and goal_bias is not None)

    def goal_bias_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.goal_bias is not None,
            "label": self.goal_bias_label,
            "lower_is_better": bool(self.goal_bias_lower_is_better),
            "nodes_scored": int(self._goal_bias_scored),
            "errors": int(self._goal_bias_errors),
        }

    def _goal_bias_score(self, node: Mapping[str, Any]) -> float:
        if self.goal_bias is None:
            return 0.0
        frame = node.get("frame")
        if frame is None:
            return 0.0
        try:
            score = float(self.goal_bias(frame))
            self._goal_bias_scored += 1
            return score
        except Exception:
            self._goal_bias_errors += 1
            return 0.0

    def _goal_bias_key(self, score: float) -> float:
        if self.goal_bias is None:
            return 0.0
        if self.goal_bias_lower_is_better:
            return float(score)
        return -float(score)

    def _hash(self, frame) -> str:
        from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash

        g = grid_of(frame)
        if self.hud_mask is not None and getattr(self.hud_mask, "shape", None) == g.shape:
            g = g.copy()
            g[self.hud_mask] = 0  # collapse counter/timer cells so equal game states dedup
        return frame_hash(g)

    def _candidates(self, frame, path: Sequence[dict] | None = None) -> list[dict]:
        from carnot.agentic.arc_graph_explore import rich_action_candidates

        action_prior = self.action_prior
        if action_prior is not None and hasattr(action_prior, "for_path"):
            action_prior = action_prior.for_path(path or [])
        candidates = rich_action_candidates(
            frame,
            frame_change_scorer=self.frame_change_scorer,
            frame_change_prune_threshold=self.frame_change_prune_threshold,
            action_prior=action_prior,
            action_prior_prune_quantile=self.action_prior_prune_quantile,
        )
        if self.adaptive_budget_threshold is not None and candidates:
            from carnot.agentic.arc_adaptive_budget import apply_adaptive_budget

            try:
                frame_hash = self._hash(frame)
                frame_is_novel = frame_hash not in self.graph
            except Exception:
                frame_is_novel = True
            gated, decision = apply_adaptive_budget(
                frame,
                candidates,
                threshold=self.adaptive_budget_threshold,
                value_head=self.adaptive_budget_value_head or self.value_head,
                frame_change_scorer=self.frame_change_scorer,
                frame_is_novel=frame_is_novel,
                change_threshold=self.adaptive_budget_noop_threshold,
            )
            if decision.committed_single_candidate:
                self._adaptive_budget_commit_count += 1
                self._adaptive_budget_candidates_skipped += max(
                    0,
                    int(decision.normal_width) - int(decision.budget),
                )
            else:
                self._adaptive_budget_expanded_count += 1
            self._adaptive_budget_history.append(decision.as_dict())
            candidates = gated
        return [{"action": int(c.action_id), "data": c.data} for c in candidates]

    @staticmethod
    def _game_over(frame) -> bool:
        from carnot.agentic.arc_agi3_live_adapter import _game_over

        try:
            return bool(_game_over(frame))
        except Exception:
            return False

    def _value(self, frame, node_hash: str | None = None) -> float:
        """Frame-only learned progress score (predicted steps-to-next-level-up; LOWER == closer).
        0.0 when no value head -> the frontier falls back to shallowest-first. Never crashes the loop.
        Also 0.0 when value_weight==0: at weight 0 the value term is multiplied by 0 in the frontier
        priority (ordering is depth-primary + on-path tiebreak, identical with or without the value), so
        computing the expensive v3 featurizer per node would be pure dead cost (the 2026-06-20 regression:
        it made the weight-0 submitted default slower than bare BFS for ZERO routing benefit). The v3 head
        stays fully wired and fires unchanged whenever value_weight>0.

        Positive weights use a frame-hash cache so repeated frontier visits do
        not re-run the expensive v3 featurizer.
        """
        if self.value_head is None or self.value_weight == 0.0:
            return 0.0
        if node_hash is not None and node_hash in self._value_cache:
            self._value_cache_hits += 1
            return self._value_cache[node_hash]
        try:
            value = float(self.value_head(frame))
        except Exception:
            value = 0.0
        self._value_head_evals += 1
        if node_hash is not None:
            self._value_cache[node_hash] = value
        return value

    def _initial_value(self, frame) -> tuple[float | None, Any | None]:
        if self.value_head is None or self.value_weight == 0.0:
            return 0.0, None
        return None, frame

    @staticmethod
    def _same_path_step(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return int(left.get("action")) == int(right.get("action")) and left.get("data") == right.get("data")

    @classmethod
    def _path_is_prefix(cls, prefix: Sequence[Mapping[str, Any]], path: Sequence[Mapping[str, Any]]) -> bool:
        if len(prefix) > len(path):
            return False
        return all(cls._same_path_step(left, right) for left, right in zip(prefix, path))

    def _record_forward_edge(self, origin: Optional[str], action: Mapping[str, Any], next_hash: str) -> None:
        if origin is None or next_hash == origin:
            return
        act = {"action": int(action["action"]), "data": action.get("data")}
        edges = self.adj.setdefault(origin, [])
        if any(existing == act and nxt == next_hash for existing, nxt in edges):
            return
        edges.append((act, next_hash))
        self._nav_edges_recorded += 1

    def _ingest(self, latest) -> None:
        if latest is None:
            return
        h = self._hash(latest)
        lvl = _level_of(latest)
        over = self._game_over(latest)
        if self.start_level is None:
            self.start_level = lvl
        self.best_level = max(self.best_level, lvl)
        features = None
        if self.awaiting is not None:
            o = self.awaiting
            self.awaiting = None
            if over:
                self._record_discriminative_sample(
                    latest,
                    label=0,
                    source="game_over",
                    node_hash=h,
                )
            else:
                act = {"action": o["action"], "data": o["data"]}
                # record the forward edge for frontier-distance navigation (only if the
                # action actually CHANGED state — a no-op self-edge is useless to navigate)
                self._record_forward_edge(o["origin"], act, h)
                if h not in self.graph:
                    opath = self.graph.get(o["origin"], {}).get("path", [])
                    features = self._record_discriminative_sample(
                        latest,
                        label=1,
                        source="alive_frontier",
                        node_hash=h,
                    )
                    new_path = opath + [act]
                    value, frame_for_value = self._initial_value(latest)
                    self.graph[h] = {
                        "path": new_path,
                        "untested": self._candidates(latest, path=new_path),
                        "value": value,
                        "frame": latest if self.goal_bias is not None else frame_for_value,
                        "discriminative_features": features,
                    }
        self.cur = h
        if self.root is None and not over:
            self.root = h
            features = self._record_discriminative_sample(
                latest,
                label=1,
                source="root",
                node_hash=h,
            )
            value, frame_for_value = self._initial_value(latest)
            self.graph.setdefault(
                h,
                {
                    "path": [],
                    "untested": self._candidates(latest, path=[]),
                    "value": value,
                    "frame": latest if self.goal_bias is not None else frame_for_value,
                    "discriminative_features": features,
                },
            )

    def _frontier(self) -> Optional[str]:
        # BRIDGE: A*-style frontier order -- priority = depth + value_weight*value. value_weight=0 is
        # depth-primary (pure BFS; value only breaks ties -> provably cannot regress). value_weight>0
        # lets the value head NUDGE toward predicted-closer states (the routing that unlocked cn04 in
        # graph_explore at weight 5). A full value-OVERRIDE (ignoring depth) measurably REGRESSED the
        # baseline (the weak head misroutes from shallow wins), so the blend keeps depth load-bearing.
        use_value = self.value_head is not None and self.value_weight != 0.0
        w = self.value_weight
        eligible: list[tuple[str, dict, int, float]] = []
        for h, node in self.graph.items():
            if not node["untested"]:
                self._record_discriminative_features(
                    node.get("discriminative_features"),
                    label=0,
                    source="frontier_exhausted",
                    node_hash=h,
                )
                continue
            on_path = self._node_on_path_proba(node)
            node["on_path_proba"] = on_path
            if (
                self.online_discriminator is not None
                and on_path < self.discriminative_prune_threshold
            ):
                if node.get("discriminative_pruned") is not True:
                    self._disc_frontier_pruned += 1
                node["discriminative_pruned"] = True
                continue
            node["discriminative_pruned"] = False
            depth = len(node["path"])
            eligible.append((h, node, depth, on_path, self._goal_bias_score(node)))

        if use_value and eligible:
            cheap_ranked = sorted(
                eligible,
                key=lambda item: (
                    item[2],
                    -item[3],
                    item[0],
                ),
            )[: self.lazy_value_top_k]
            for h, node, _depth, _on_path, _goal_bias in cheap_ranked:
                if node.get("value") is None:
                    node["value"] = self._value(node.get("frame"), node_hash=h)
                    if self.goal_bias is None:
                        node["frame"] = None

        best = None
        best_key = None
        for h, node, depth, on_path, goal_bias in eligible:
            nav_key = self._frontier_navigation_cost_key(h) if self.navigation_cost_tiebreak else ()
            if self.navigation_cost_tiebreak and use_value:
                value = node.get("value", 0.0)
                if value is None:
                    value = 0.0
                key = (depth, w * float(value), self._goal_bias_key(goal_bias), *nav_key, -on_path)
            elif use_value:
                value = node.get("value", 0.0)
                if value is None:
                    value = 0.0
                key = (depth + w * float(value), depth, self._goal_bias_key(goal_bias), -on_path)
            elif self.navigation_cost_tiebreak:
                key = (depth, self._goal_bias_key(goal_bias), *nav_key, -on_path)
            else:
                key = (depth, self._goal_bias_key(goal_bias), -on_path)
            if best is None or key < best_key:
                best, best_key = h, key
        return best

    def _frontier_navigation_cost_key(self, node_hash: str) -> tuple[int, int]:
        """SCENARIO-ARC-FCP-4523: prefer cheap navigation only within equal depth."""

        fwd = self._shortest_path(self.cur, node_hash)
        if fwd is not None:
            return (0, len(fwd))
        return (1, len(self.graph.get(node_hash, {}).get("path", [])))

    def _shortest_path(self, src: Optional[str], dst: str) -> Optional[list]:
        """Frontier-distance navigation: BFS over the KNOWN forward edges from src to dst.
        Returns the action sequence to walk there WITHOUT a RESET (cheaper than replay-
        from-root), or None if dst isn't forward-reachable from src in the known graph."""
        from collections import deque

        if src is None or src == dst:
            return [] if src == dst else None
        seen = {src}
        q = deque([(src, [])])
        while q:
            node, path = q.popleft()
            for act, nxt in self.adj.get(node, []):
                if nxt in seen:
                    continue
                npath = path + [act]
                if nxt == dst:
                    return npath
                seen.add(nxt)
                q.append((nxt, npath))
        return None

    def _partial_forward_path(self, src: Optional[str], dst: str) -> Optional[list]:
        """Walk to the deepest reachable ancestor of dst, then replay only the suffix."""

        if src is None or dst not in self.graph:
            return None
        target_path = self.graph.get(dst, {}).get("path", [])
        best_depth = -1
        best_plan = None
        for ancestor, node in self.graph.items():
            if ancestor == dst:
                continue
            ancestor_path = node.get("path", [])
            if not self._path_is_prefix(ancestor_path, target_path):
                continue
            forward = self._shortest_path(src, ancestor)
            if forward is None:
                continue
            depth = len(ancestor_path)
            if depth <= best_depth:
                continue
            suffix = [
                {"action": int(step["action"]), "data": step.get("data")}
                for step in target_path[depth:]
            ]
            best_depth = depth
            best_plan = list(forward) + suffix
        return best_plan

    def _serve(self) -> tuple:
        item = self.pending.pop(0)
        if item["kind"] == "RESET":
            self.awaiting = None  # RESET has no forward edge to attribute
            return ("RESET", None)
        if item.get("probe"):
            origin = item.get("origin", self.cur)
            if "origin" not in item:
                self._drop_queued_action_from_current_frontier(origin, item)
            self.awaiting = {"origin": origin, "action": item["kind"], "data": item["data"]}
        else:
            # nav / RESET-replay step (probe:False): attribute its forward edge from the CURRENT state so
            # adj FILLS IN the replayed path. Previously only probe steps recorded edges, so replayed paths
            # were never learned -> _shortest_path returned None -> every backtrack RESET-replayed from root,
            # burning actions (the 2026-06-20 regression: lp85 7792 actions vs bare BFS's 21). Recording
            # these edges lets future navigation use _shortest_path (forward-walk) instead of RESET-replay.
            self.awaiting = {"origin": self.cur, "action": item["kind"], "data": item["data"]}
        return (item["kind"], item["data"])

    def _drop_queued_action_from_current_frontier(self, origin: Optional[str], item: Mapping[str, Any]) -> None:
        if origin is None:
            return
        node = self.graph.get(origin)
        if not node:
            return
        act = {"action": int(item["kind"]), "data": item.get("data")}
        for idx, candidate in enumerate(list(node.get("untested") or [])):
            if self._same_path_step(candidate, act):
                del node["untested"][idx]
                return

    def _pop_frontier_batch(self, node: dict) -> list[dict]:
        limit = len(node["untested"]) if self.frontier_batch_size is None else self.frontier_batch_size
        count = min(int(limit), len(node["untested"]))
        actions = node["untested"][:count]
        del node["untested"][:count]
        return actions

    def next_move(self, frames, latest) -> tuple:
        if self.root is None and latest is None:  # bootstrap: RESET to get the first frame
            return ("RESET", None)
        self._ingest(latest)
        if self.pending:
            return self._serve()
        over = latest is not None and self._game_over(latest)
        cur_node = self.graph.get(self.cur) if not over else None
        # 1) DEPTH-first ride (search_mode="depth_first_ride", default): expand the current state's
        #    untested SALIENT actions while under the depth cap (no nav cost; reaches the deep wins
        #    lp85/sp80 need — BFS-order regressed those). best_first SKIPS this and always expands the
        #    globally-best A*-value frontier (step 2) so the value head drives the search order.
        if (
            self.search_mode == "depth_first_ride"
            and cur_node
            and cur_node["untested"]
            and len(cur_node["path"]) < self.max_depth
        ):
            a = cur_node["untested"].pop(0)
            self.awaiting = {"origin": self.cur, "action": a["action"], "data": a["data"]}
            return (a["action"], a["data"])
        # 2) Expand the best frontier (A*-value order). In best_first this is the primary step; in
        #    depth_first_ride it fires when the current node is exhausted / dead-end / depth-capped.
        th = self._frontier()
        if th is None:
            self.explored_out = True
            return (None, None)
        node = self.graph[th]
        if (
            th == self.cur and not over
        ):  # best frontier IS the current state -> expand in place (no nav)
            a = node["untested"].pop(0)
            self.awaiting = {"origin": self.cur, "action": a["action"], "data": a["data"]}
            return (a["action"], a["data"])
        batch = self._pop_frontier_batch(node)
        self._nav_attempts += 1
        fwd = self._shortest_path(self.cur, th) if not over else None
        if fwd is not None:
            self._nav_exact_hits += 1
            self._nav_forward_steps += len(fwd)
            self.pending = [{"kind": s["action"], "data": s["data"], "probe": False} for s in fwd]
        else:
            partial = self._partial_forward_path(self.cur, th) if not over else None
            if partial is not None:
                self._nav_partial_hits += 1
                self._nav_forward_steps += len(partial)
                self.pending = [
                    {"kind": s["action"], "data": s["data"], "probe": False} for s in partial
                ]
            else:
                self._nav_reset_fallbacks += 1
                self._nav_reset_replay_steps += len(node["path"])
                self.pending = [{"kind": "RESET", "data": None, "probe": False}]
                self.pending += [
                    {"kind": s["action"], "data": s["data"], "probe": False} for s in node["path"]
                ]
        for idx, action in enumerate(batch):
            item = {"kind": action["action"], "data": action["data"], "probe": True}
            if idx == 0:
                item["origin"] = th
            self.pending.append(item)
        return self._serve()

    def is_done(self, frames, latest) -> bool:
        if latest is not None:
            lvl = _level_of(latest)
            if self.start_level is None:
                self.start_level = lvl
            self.best_level = max(self.best_level, lvl)
        if (
            self.start_level is not None
            and self.best_level >= self.start_level + self.target_levels
        ):
            return True
        # Smart grace-period early-stop: cut the fruitless post-solve tail WITHOUT capping levels. Once at
        # least one level is reached, a new level-up resets the window; if no new level appears within
        # `early_stop_grace` moves, stop (the agent is churning on unreachable deeper levels, and every
        # extra action quadratically erodes the (human/agent_actions)^2 efficiency score). Riding
        # consecutive level-ups keeps the window alive, so reachable deeper levels are still solved.
        if (
            self.early_stop_grace is not None
            and self.start_level is not None
            and self.best_level > self.start_level
        ):
            if self.best_level > self._early_stop_level_mark:
                self._early_stop_level_mark = self.best_level
                self._early_stop_frame_mark = len(frames)
            elif (len(frames) - self._early_stop_frame_mark) > self.early_stop_grace:
                self.early_stopped = True
                return True
        return self.explored_out


def load_cross_game_value_head():
    """BRIDGE loader: the frame-only cross-game value head (frame -> predicted steps-to-next-level-up),
    trained offline on ALL banked solves by scripts/arc_cross_game_verifier_train.py. Returns a callable
    the StepwiseExplorer routes its frontier with on an UNSEEN game, or None if not yet trained. This is
    the offline->live distillation: continued offline solves retrain it and the live agent inherits them."""
    from pathlib import Path

    models = Path(__file__).resolve().parents[3] / "models"
    try:
        from carnot.agentic.arc_value_learner import (
            LearnedVerifier,
            cross_game_features,
            cross_game_features_v2,
            cross_game_features_v3,
        )

        v3 = models / "arc_verifier_cross_game_v3.json"
        if v3.exists():
            v = LearnedVerifier.load(v3, cross_game_features_v3)
            return lambda frame: v(frame)
        # prefer the RICHER v2 head (spatial occupancy; it routed cn04 where v1's 5 scalars could not)
        v2 = models / "arc_verifier_cross_game_v2.json"
        if v2.exists():
            v = LearnedVerifier.load(v2, cross_game_features_v2)
            return lambda frame: v(frame)
        v1 = models / "arc_verifier_cross_game.json"
        if v1.exists():
            v = LearnedVerifier.load(v1, cross_game_features)
            return lambda frame: v(frame)
    except Exception:
        return None
    return None


class CarnotAgentPolicy:
    """Framework-agnostic decision logic. `next_move` yields ("RESET",None) once, then
    the banked plan one step at a time, then (None,None) when exhausted. `is_done`
    stops when the target level is reached or the plan is spent."""

    def __init__(
        self,
        game_id: str,
        solutions: Optional[dict] = None,
        target_level: Optional[int] = None,
        force_explore: bool = False,
        hud_mask=None,
        value_head=None,
        value_weight: float = 0.0,
        search_mode: str = "depth_first_ride",
        frame_change_scorer: Any | None = None,
        frame_change_prune_threshold: float | None = None,
        action_prior: Any | None = None,
        action_prior_prune_quantile: float | None = None,
        adaptive_budget_threshold: float | None = None,
        adaptive_budget_value_head: Any | None = None,
        adaptive_budget_noop_threshold: float = 0.5,
        lazy_value_top_k: int = SUBMITTED_LAZY_VALUE_TOP_K,
        frontier_batch_size: int | str | None = SUBMITTED_FRONTIER_BATCH_SIZE,
        navigation_cost_tiebreak: bool = SUBMITTED_NAVIGATION_COST_TIEBREAK,
    ) -> None:
        self.short = str(game_id).split("-", 1)[0]
        sols = solutions if solutions is not None else load_solutions()
        self.plan = [] if force_explore else sols.get(self.short, [])
        self.i = 0
        self.reset_sent = False
        self.target = target_level if target_level is not None else CLAIMED.get(self.short, 1)
        self.has_plan = bool(self.plan)
        # eval games are UNSEEN -> no banked plan -> the generic step-wise explorer runs (value_head +
        # value_weight A*-route its frontier when provided -- the offline->live bridge).
        self.explorer: Optional[StepwiseExplorer] = (
            None
            if self.has_plan
            else StepwiseExplorer(
                hud_mask=hud_mask,
                value_head=value_head,
                value_weight=value_weight,
                search_mode=search_mode,
                frame_change_scorer=frame_change_scorer,
                frame_change_prune_threshold=frame_change_prune_threshold,
                action_prior=action_prior,
                action_prior_prune_quantile=action_prior_prune_quantile,
                adaptive_budget_threshold=adaptive_budget_threshold,
                adaptive_budget_value_head=adaptive_budget_value_head,
                adaptive_budget_noop_threshold=adaptive_budget_noop_threshold,
                lazy_value_top_k=lazy_value_top_k,
                frontier_batch_size=frontier_batch_size,
                navigation_cost_tiebreak=navigation_cost_tiebreak,
            )
        )

    def next_move(self, frames, latest_frame) -> tuple:
        """-> ("RESET", None) | (action_id:int, data:dict|None) | (None, None)."""
        if self.explorer is not None:  # unknown game: generic solver
            return self.explorer.next_move(frames, latest_frame)
        if not self.reset_sent:  # known game: replay banked solution
            self.reset_sent = True
            return ("RESET", None)
        if self.i < len(self.plan):
            s = self.plan[self.i]
            self.i += 1
            return (int(s["action"]), s.get("data"))
        return (None, None)

    def is_done(self, frames, latest_frame) -> bool:
        if self.explorer is not None:
            return self.explorer.is_done(frames, latest_frame)
        if _level_of(latest_frame) >= self.target:
            return True
        return self.reset_sent and self.i >= len(self.plan)


class E3AgentPolicy:
    """E3-mode agent: the STRONG choose_action. Phase machine driven step-wise by the
    harness — EXPLORE (collect transitions from its own play) -> INDUCE (an OFFLINE
    local proposer writes a world model from those transitions) -> VERIFY (Carnot
    WorldModelVerifier grounds it) -> PLAN (search to a win INSIDE the verified model)
    -> EXECUTE (replay the plan; on divergence, back to EXPLORE). The proposer is
    INJECTED and defaults to the offline-legal local one, never a closed online API, so
    this is competition-legal (no internet at eval) AND decentralized.

    The induce/verify/plan quality (esp. from a small LOCAL model) is the open milestone
    the focused loop measures next; the EXPLORE + collect + verify wiring is exercised
    here, and EXECUTE re-uses the same env interface as the explorer."""

    def __init__(
        self,
        game_id: str,
        proposer=None,
        explore_budget: Optional[int] = None,
        target_levels: int = SUBMITTED_TARGET_LEVELS,
        value_head: Any = _DEFAULT_VALUE_HEAD,
        value_weight: float = SUBMITTED_VALUE_WEIGHT,
        search_mode: str = SUBMITTED_SEARCH_MODE,
        mechanic_detector=None,
        frame_change_scorer: Any | None = None,
        frame_change_prune_threshold: float | None = None,
        action_prior: Any | None = None,
        action_prior_prune_quantile: float | None = None,
        adaptive_budget_threshold: float | None = None,
        adaptive_budget_value_head: Any | None = None,
        adaptive_budget_noop_threshold: float = 0.5,
        lazy_value_top_k: int = SUBMITTED_LAZY_VALUE_TOP_K,
        frontier_batch_size: int | str | None = SUBMITTED_FRONTIER_BATCH_SIZE,
        navigation_cost_tiebreak: bool = SUBMITTED_NAVIGATION_COST_TIEBREAK,
    ) -> None:
        self.short = str(game_id).split("-", 1)[0]
        self.target_levels = int(target_levels)
        if value_head is _DEFAULT_VALUE_HEAD:
            value_head = load_cross_game_value_head()
        self.strategy_route = arc_strategy_router.route_for_game(self.short)
        self._route_from_frame_checked = False
        self.mechanic_detector = mechanic_detector or _frame_only_mechanic_hint
        self.dsl_model = ObjectDeltaModel(self.short)
        self.dsl_energy: Optional[dict[str, Any]] = None
        self._dsl_transitions: list[tuple[Any, tuple, Any]] = []
        self.explorer = StepwiseExplorer(
            target_levels=target_levels,
            value_head=value_head,
            value_weight=value_weight,
            search_mode=search_mode,
            frame_change_scorer=frame_change_scorer,
            frame_change_prune_threshold=frame_change_prune_threshold,
            action_prior=action_prior,
            action_prior_prune_quantile=action_prior_prune_quantile,
            adaptive_budget_threshold=adaptive_budget_threshold,
            adaptive_budget_value_head=adaptive_budget_value_head,
            adaptive_budget_noop_threshold=adaptive_budget_noop_threshold,
            lazy_value_top_k=lazy_value_top_k,
            frontier_batch_size=frontier_batch_size,
            navigation_cost_tiebreak=navigation_cost_tiebreak,
        )
        self.transitions: list = []  # (grid_before, action, data, grid_after) self-collected
        self.explore_budget = (
            int(explore_budget)
            if explore_budget is not None
            else _route_explore_budget(self.strategy_route)
        )
        self.proposer = proposer  # default set lazily to LocalGGUFProposer
        self.phase = "explore"
        self.plan: list = []
        self.pi = 0
        self._prev = None  # last (grid, action_id, data) for transition pairing
        self.cell = 1
        self.induced = False
        self.root_grid = None  # the reset-state logical grid; plan_in_model starts here
        self.world_model_trust_selection = None
        self._observed_level: Optional[int] = None
        self._start_level: Optional[int] = None
        self._current_goal_level: Optional[int] = None
        self._episode_transition_start = 0
        self._episode_dsl_transition_start = 0
        self._level_reinduction_pending = False
        self._pending_induction_reason: Optional[str] = None
        self._execute_plan_from_current = False
        self.level_induction_events: list[dict[str, Any]] = []
        self.induction_attempts: list[dict[str, Any]] = []

    def _maybe_route_from_frame(self, latest: Any) -> None:
        if self._route_from_frame_checked or latest is None:
            return
        self._route_from_frame_checked = True
        try:
            mechanic = self.mechanic_detector(latest) if self.mechanic_detector else None
        except Exception:
            mechanic = None
        if isinstance(mechanic, dict):
            mechanic = mechanic.get("mechanic")
        if not mechanic or mechanic == "unknown":
            return
        routed = arc_strategy_router.route_for_game(self.short, mechanic=str(mechanic))
        if routed.get("name") != self.strategy_route.get("name"):
            self.strategy_route = routed
            self.explore_budget = min(self.explore_budget, _route_explore_budget(routed))

    def _fit_dsl_model(self) -> None:
        active = self._active_dsl_transitions()
        if not active:
            return
        try:
            self.dsl_model = ObjectDeltaModel(self.short).fit(active)
            self.dsl_energy = self.dsl_model.consistency_energy(active)
        except Exception:
            self.dsl_energy = {"energy": None, "n_heldout": len(active)}

    def _active_transitions(self) -> list:
        return list(self.transitions[self._episode_transition_start :])

    def _active_dsl_transitions(self) -> list[tuple[Any, tuple, Any]]:
        return list(self._dsl_transitions[self._episode_dsl_transition_start :])

    def _observe_level_boundary(self, latest: Any, *, frames_seen: int) -> list[dict[str, Any]]:
        """SCENARIO-ARC-WMTE-4533: level-up starts a new goal-acquisition episode."""

        if latest is None:
            return []
        level = _level_of(latest)
        if self._start_level is None:
            self._start_level = level
        if self._observed_level is None:
            self._observed_level = level
            self._current_goal_level = level + 1
            return []
        if level <= self._observed_level:
            return []
        events: list[dict[str, Any]] = []
        start = int(self._start_level or 0)
        for new_level in range(self._observed_level + 1, level + 1):
            relative = new_level - start
            if relative >= self.target_levels:
                continue
            event = self._begin_level_goal_episode(new_level, frames_seen=frames_seen)
            events.append(event)
        self._observed_level = level
        return events

    def _begin_level_goal_episode(self, completed_level: int, *, frames_seen: int) -> dict[str, Any]:
        next_goal = int(completed_level) + 1
        self._current_goal_level = next_goal
        self._episode_transition_start = len(self.transitions)
        self._episode_dsl_transition_start = len(self._dsl_transitions)
        self.induced = False
        self.plan = []
        self.pi = 0
        self._level_reinduction_pending = True
        self._execute_plan_from_current = True
        self.world_model_trust_selection = None
        self.dsl_energy = None
        self.explorer.set_goal_bias(None, label="")
        event = {
            "trigger": "level_up",
            "completed_level": int(completed_level),
            "next_goal_level": next_goal,
            "transition_start": int(self._episode_transition_start),
            "dsl_transition_start": int(self._episode_dsl_transition_start),
            "frames_seen": int(frames_seen),
        }
        self.level_induction_events.append(event)
        return event

    def _current_goal_reached(self) -> bool:
        if self._current_goal_level is None:
            return self.explorer.best_level > (self.explorer.start_level or 0)
        return self.explorer.best_level >= self._current_goal_level

    def _should_enter_induction(self, *, stalled: bool, won: bool) -> tuple[bool, Optional[str]]:
        if (
            self._level_reinduction_pending
            and not self.induced
            and len(self.transitions) > self._episode_transition_start
        ):
            return True, "level_up_reinduction"
        if stalled and not won and not self.induced:
            return True, "stall"
        return False, None

    def _install_goal_bias(self, is_done) -> None:
        if not callable(is_done):
            return

        def _bias(frame: Any) -> float:
            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import to_logical

            grid = to_logical(grid_of(frame), self.cell)
            return 1.0 if is_done(grid) else 0.0

        label = f"L{self._current_goal_level or '?'}_induced_goal_predicate"
        self.explorer.set_goal_bias(_bias, label=label)

    def _next_plan_move(self) -> tuple:
        step = self.plan[self.pi]
        self.pi += 1
        return (step["action"], step.get("data"))

    def _proposer(self):
        if self.proposer is None:
            import os
            from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

            # Live-submission generator (validated 2026-06-19): Qwen3.5-9B-MTP + MTP + 8-bit KV + /no_think,
            # n_predict>=2048. 5.9GB Q4 fits 16GB; 62.5% Layer-B grounding (DeepSeek-Flash 25%, gemma verbose).
            # Kaggle deploy: set CARNOT_ARC_GGUF_PATH to the bundled /kaggle/input/.../Qwen3.5-9B-Q4_K_M.gguf;
            # CARNOT_ARC_MTP=0 disables MTP if a tight-VRAM box needs the ~4GB the self-draft costs.
            self.proposer = LocalGGUFProposer(
                repo_substr="Qwen3.5-9B-MTP",
                model_path=os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
                mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0"),
                kv_quant="q8_0",
                no_think_prefix="/no_think\n",
                max_tokens=2560,
            )
        return self.proposer

    def _world_model_candidates(self, engine, is_done) -> list[WorldModelCandidate]:
        candidates = [WorldModelCandidate("loaded_world_model.py", engine, is_done)]
        provider = getattr(self.proposer, "world_model_candidates", None)
        if provider is None:
            provider = getattr(self.proposer, "candidate_engines", None)
        if not callable(provider):
            return candidates
        for i, row in enumerate(provider(self.short)):
            if isinstance(row, WorldModelCandidate):
                candidates.append(row)
            elif isinstance(row, dict):
                candidates.append(
                    WorldModelCandidate(
                        str(row.get("name") or f"candidate_{i}"),
                        row["engine"],
                        row.get("is_level_complete"),
                    )
                )
            else:
                name, candidate_engine, *rest = row
                candidates.append(
                    WorldModelCandidate(
                        str(name),
                        candidate_engine,
                        rest[0] if rest else None,
                    )
                )
        return candidates

    def next_move(self, frames, latest):
        from carnot.agentic.arc_executable_world_model import to_logical, detect_cell

        self._maybe_route_from_frame(latest)
        # collect a transition from the last action's outcome
        if self._prev is not None and latest is not None:
            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import Transition

            g0, aid, data = self._prev
            g1 = to_logical(grid_of(latest), self.cell)
            self.transitions.append(Transition(g0, aid, data, g1, 0, _level_of(latest)))
            self._dsl_transitions.append((g0, _action_key(aid, data), g1))
        boundary_events = self._observe_level_boundary(latest, frames_seen=len(frames))
        if boundary_events and latest is not None:
            try:
                from carnot.agentic.arc_agi3_world_model import grid_of

                self.cell = detect_cell(grid_of(latest))
                self.root_grid = to_logical(grid_of(latest), self.cell)
                self._prev = None
            except Exception:
                pass
        if self.phase == "explore":
            should_induce, reason = self._should_enter_induction(
                stalled=False,
                won=self._current_goal_reached(),
            )
            if should_induce:
                self.phase = "induce"
                self._pending_induction_reason = reason
                if reason != "level_up_reinduction":
                    self._execute_plan_from_current = False
            else:
                mv = self.explorer.next_move(frames, latest)
                if latest is not None:
                    from carnot.agentic.arc_agi3_world_model import grid_of

                    self.cell = detect_cell(grid_of(latest))
                    if self.root_grid is None and self.explorer.root is not None:
                        self.root_grid = to_logical(grid_of(latest), self.cell)
                    if mv[0] not in ("RESET", None):
                        self._prev = (to_logical(grid_of(latest), self.cell), int(mv[0]), mv[1])
                    else:
                        self._prev = None
                # VERIFIER-ROUTED CASCADE escalation: hand off to the tier-3 induction path on
                # a genuine stall, and also after a level-up once post-boundary evidence exists.
                won = self._current_goal_reached()
                stalled = len(self.transitions) >= self.explore_budget or self.explorer.explored_out
                should_induce, reason = self._should_enter_induction(stalled=stalled, won=won)
                if should_induce:
                    self.phase = "induce"
                    self._pending_induction_reason = reason
                    if reason != "level_up_reinduction":
                        self._execute_plan_from_current = False
                return mv
        if self.phase == "induce" and not self.induced:
            self.induced = True
            self._induce_and_plan()
            self._level_reinduction_pending = False
            self.phase = "execute" if self.plan else "explore"
            self._prev = None
            if self.plan:
                if self._execute_plan_from_current:
                    return self._next_plan_move()
                return ("RESET", None)
            return self.explorer.next_move(frames, latest)
        if self.phase == "execute" and self.pi < len(self.plan):
            return self._next_plan_move()
        # plan exhausted / no model -> keep exploring
        self.phase = "explore"
        return self.explorer.next_move(frames, latest)

    def _induce_and_plan(self):
        import os

        from carnot.agentic import arc_executable_world_model as e3

        active_transitions = self._active_transitions()
        attempt = {
            "reason": self._pending_induction_reason or "stall",
            "goal_level": self._current_goal_level,
            "transition_count": len(active_transitions),
            "dsl_transition_count": len(self._active_dsl_transitions()),
            "model_specs": "offline_dsl_induction_no_llm",
            "planned": False,
            "skipped": "",
        }
        self.induction_attempts.append(attempt)

        # Production-safe escape hatch (default OFF): when CARNOT_ARC_DISABLE_INDUCTION=1, skip the LLM
        # world-model induction tier entirely and stay in the (fast) tier-1 explorer. The local submission
        # GATE sets this so it measures the explorer's SEARCH/efficiency cleanly + fast, without paying the
        # ~30s+ llama-server spawn (a one-time, acceptable cost under the real 12h eval, but it dominates a
        # bounded local gate run and is irrelevant to detecting a SEARCH regression). Unset in production
        # (Kaggle) -> induction runs exactly as before.
        if os.environ.get("CARNOT_ARC_DISABLE_INDUCTION") == "1":
            attempt["skipped"] = "disabled_by_env"
            return
        if not active_transitions:
            attempt["skipped"] = "no_active_transitions"
            return

        try:
            if attempt["reason"] == "level_up_reinduction":
                self._fit_dsl_model()
                outcome = execute_bounded_llm_reinduction(
                    game=self.short,
                    transitions=active_transitions,
                    cell=self.cell,
                    root_grid=self.root_grid,
                    proposer=self._proposer(),
                    candidate_provider=self._world_model_candidates,
                    load_engine=e3.load_engine,
                    plan_in_model=e3.plan_in_model,
                    max_rounds=MAX_REFINEMENT_ROUNDS,
                )
                attempt.update(
                    {
                        "model_specs": outcome.model_specs,
                        "planned": bool(outcome.planned),
                        "skipped": outcome.skipped,
                        "plan_length": len(outcome.plan),
                        "selected_candidate_name": outcome.selected_candidate_name,
                        "goal_candidate_names": list(outcome.goal_candidate_names),
                        "dynamics_candidate_names": list(outcome.dynamics_candidate_names),
                        "refinement_rounds_used": int(outcome.refinement_rounds_used),
                        "refinement_rounds": list(outcome.rounds),
                        "counterexamples": list(outcome.counterexamples),
                        "verifier_is_oracle": bool(outcome.verifier_is_oracle),
                    }
                )
                if outcome.goal_predicate is not None:
                    self._install_goal_bias(outcome.goal_predicate)
                if outcome.planned:
                    self.plan = list(outcome.plan)
                return
            self._fit_dsl_model()
            ok, _ = self._proposer().induce(self.short, active_transitions, self.cell)
            if not ok or self.root_grid is None:
                attempt["skipped"] = "proposer_failed_or_missing_root"
                return
            engine, is_done = e3.load_engine(self.short)
            if self.short in HIDDEN_STATE_GAME_IDS:
                self.world_model_trust_selection = select_trusted_world_model(
                    active_transitions,
                    self._world_model_candidates(engine, is_done),
                    hidden_state=True,
                )
                if self.world_model_trust_selection.selected_score.heldout_accuracy < 0.5:
                    attempt["skipped"] = "hidden_state_trust_below_threshold"
                    return
                engine = self.world_model_trust_selection.selected.engine
                is_done = self.world_model_trust_selection.selected.is_level_complete or is_done
            else:
                vr = e3.WorldModelVerifier(active_transitions).score(engine)
                if vr.accuracy < 0.5:  # too weak to trust for execution-grounded planning
                    attempt["skipped"] = "world_model_accuracy_below_threshold"
                    return
            self._install_goal_bias(is_done)
            # plan ENTIRELY in the model (zero real actions); execute phase RESETs then
            # replays this plan in the real env, halting on divergence.
            plan = e3.plan_in_model(engine, is_done, self.root_grid)
            if plan:
                self.plan = plan
                attempt["planned"] = True
                attempt["plan_length"] = len(plan)
        except Exception:
            attempt["skipped"] = "exception"
            return

    def is_done(self, frames, latest):
        return self.explorer.is_done(frames, latest) and self.phase == "explore"


# ============================================================================================
# SINGLE SOURCE OF TRUTH for WHAT SHIPS. The 0.08 incident (2026-06-19): the offline eval measured
# STRONGER opt-in configs (explorer_bf unlocked cn04) while the SUBMITTED default shipped bare BFS,
# and nobody caught it because "better" was opt-in-only and the headline metric was banked-replay
# levels, not the submitted path. RULE (enforced by test_arc_submitted_agent_parity.py): the
# STRONGEST measured config MUST be this declared submitted config -- improvements go HERE, never to
# an opt-in-only eval flag. The eval's submission-baseline + the parity test both read this dict, so
# the shipped agent and the measured baseline can never silently diverge again.
SUBMITTED_AGENT_CONFIG = {
    "policy": "E3AgentPolicy",  # the verifier-routed cascade (NOT cascade=False banked-replay)
    "cascade": True,
    # explorer config the live agent actually runs.
    "value_weight": SUBMITTED_VALUE_WEIGHT,
    "target_levels": SUBMITTED_TARGET_LEVELS,
    "search_mode": SUBMITTED_SEARCH_MODE,
    "graph_explore_budget": SUBMITTED_GRAPH_EXPLORE_BUDGET,
    "routed_explore_budget": SUBMITTED_ROUTED_EXPLORE_BUDGET,
    "lazy_value_top_k": SUBMITTED_LAZY_VALUE_TOP_K,
    "frontier_batch_size": SUBMITTED_FRONTIER_BATCH_SIZE,
    "navigation_cost_tiebreak": SUBMITTED_NAVIGATION_COST_TIEBREAK,
    "router_wired": True,
    "world_model_dsl_wired": True,
    "online_discriminative": True,
}


def make_carnot_agent(base_cls, cascade: bool = True, proposer=None):
    """Adapt the Carnot policy onto the real ARC-AGI-3-Agents `Agent` base class.
    Submission: `from agents.agent import Agent; CarnotAgent = make_carnot_agent(Agent)`.

    cascade=True (DEFAULT, the competition path): the VERIFIER-ROUTED CASCADE
    (E3AgentPolicy) — tier-1 training-free explorer; on STALL escalate to tier-3 E3
    induction with the bundled open proposer (the verifier routes + grounds). This is the
    unified choose_action the hard eval needs. cascade=False: pure recognize-and-replay
    (dev/known games only — useless on the hidden eval).

    The exact shipped config is declared in SUBMITTED_AGENT_CONFIG (single source of truth);
    test_arc_submitted_agent_parity.py asserts this function's default policy matches it, so a
    silent divergence between what we measure and what we ship cannot recur (the 0.08 incident)."""

    class CarnotAgent(base_cls):  # type: ignore
        # The framework's Agent.MAX_ACTIONS default is 80 ("avoid looping forever"),
        # which is far too low for the eval: our deepest banked replay (lp85 -> L5)
        # alone needs well over 80 actions, and the held-out-game explore fallback
        # needs room to probe + solve several levels. 80 would truncate even our best
        # known game. The real bound is the eval's wall-clock budget (<=12h across all
        # games), not this per-game loop guard; Playback overrides it to 1e6 for the
        # same reason, so it is an intended override point. 400 comfortably covers our
        # multi-level replays + explore while staying well inside the time budget.
        MAX_ACTIONS = 400

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            gid = getattr(self, "game_id", "")
            self._policy = (
                E3AgentPolicy(
                    gid,
                    proposer=proposer,
                    target_levels=int(SUBMITTED_AGENT_CONFIG["target_levels"]),
                    value_weight=float(SUBMITTED_AGENT_CONFIG["value_weight"]),
                    search_mode=str(SUBMITTED_AGENT_CONFIG["search_mode"]),
                    lazy_value_top_k=int(SUBMITTED_AGENT_CONFIG["lazy_value_top_k"]),
                )
                if cascade
                else CarnotAgentPolicy(gid, load_solutions())
            )

        def is_done(self, frames, latest_frame) -> bool:
            return self._policy.is_done(frames, latest_frame)

        def choose_action(self, frames, latest_frame):
            from arcengine import GameAction

            kind, data = self._policy.next_move(frames, latest_frame)
            if kind == "RESET" or kind is None:
                return GameAction.RESET
            act = getattr(GameAction, f"ACTION{kind}")
            if data:
                # CRITICAL: GameAction.set_data() RETURNS the inner ComplexAction, NOT
                # the enum member. The framework's do_action_request reads
                # `action.action_data` off the object choose_action returns, so we must
                # mutate the enum in place and return the ENUM. Returning set_data()'s
                # result (a ComplexAction, which has no .action_data) crashes every
                # coordinate/click action against the real harness. game_id is a required
                # ComplexAction field (Playback injects it too), so carry it through.
                act.set_data({"game_id": getattr(self, "game_id", ""), **data})
            return act

    return CarnotAgent
