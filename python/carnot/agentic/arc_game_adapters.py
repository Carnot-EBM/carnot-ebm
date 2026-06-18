"""Per-game ADAPTERS for the standing ARC learning loop — the output of each
game's (irreducible) per-game reverse-engineering, captured as a reusable plug-in
for arc_solver_kit.OfflineSolver. A new game, once its win/action/state delta is
RE'd, registers an adapter here and is then solvable + reproducible + learnable by
the standing loop (scripts/arc_loop_solve.py) with no further bespoke code.

An adapter provides the four game-specific callables the kit needs:
  action_labels(env) -> [str]   : env-discovered action vocabulary
  apply(env, label, frame)      : execute one action, return the new frame
  state_key(game)               : the dedup key (every load-bearing piece of state)
  featurize(game) -> [float]    : features for the LEARNED verifier (optional)
plus optional warmup_label and a hand verifier (goal-distance) for cold start.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from arcengine import GameAction


@dataclass
class GameAdapter:
    game: str
    action_labels: Callable[[Any], Sequence[str]]
    apply: Callable[[Any, str, Any], Any]
    state_key: Callable[[Any], Any]
    featurize: Optional[Callable[[Any], Sequence[float]]] = None
    hand_verifier: Optional[Callable[[Any], float]] = None
    warmup_label: Optional[str] = None
    depth_caps: dict = field(default_factory=lambda: {})
    # how the OfflineSolver navigates between search nodes: "replay" (default; replay-from-reset) or
    # "deepcopy" (snapshot/restore env._game per node). Use "deepcopy" only for a game whose env is
    # deepcopy-injectable AND whose replay-from-reset doesn't faithfully reproduce the searched state.
    branch_mode: str = "replay"


# ---------------- lp85 (reference adapter; click-only rotation puzzle) ----------------
def _lp85():
    from carnot.experiment_4179_arc_incremental_progress import (
        discover_click_buttons, _goal_key, _target_goal_key,
    )

    def action_labels(env):
        return [json.dumps({"x": int(b["x"]), "y": int(b["y"])}) for b in discover_click_buttons(env)]

    def apply(env, label, frame):
        a = json.loads(label)
        return env.step(GameAction.ACTION6, data={"x": a["x"], "y": a["y"]})

    def _dists(game):
        actual = _goal_key(game)
        target = _target_goal_key(game)
        by_type = defaultdict(list)
        for t, x, y in actual:
            by_type[t].append((x, y))
        return [min((abs(tx - x) + abs(ty - y)) for x, y in by_type.get(t, [])) if by_type.get(t) else 1000.0
                for t, tx, ty in target]

    def featurize(game):
        ds = _dists(game)
        n = len(ds) or 1
        return [sum(ds), float(sum(1 for d in ds if d > 0)), sum(ds) / n, float(max(ds) if ds else 0), float(n)]

    return GameAdapter(
        game="lp85", action_labels=action_labels, apply=apply, state_key=_goal_key,
        featurize=featurize, hand_verifier=lambda g: float(sum(_dists(g))),
        depth_caps={1: 20, 2: 70, 3: 90},
    )


# ---------------- tu93 (4-direction keyboard maze; frame-based RE) ----------------
def _tu93():
    """tu93 -- a 4-direction keyboard maze (ACTION1-4). FRAME-BASED RE (no internal-state read): the
    PLAYER is the moving colour-9 sprite, the GOAL is the static colour-14 marker (RE'd 2026-06-17 by
    motion: only colour-9 + the colour-4 key drift across moves; colour-14 is static). The
    hand_verifier is the player->goal Manhattan distance (goal-distance-routed best-first search).

    branch_mode='fresh_env' is LOAD-BEARING here (gotcha #7): tu93's env.reset() is NON-IDEMPOTENT --
    it leaves a parity-toggling hidden state (same path, 6 reset+replays -> levels [1,2,1,2,1,2]). The
    default reuse-one-env 'replay' search therefore detects parity-CONTINGENT 'wins' that fail the
    fresh-env reproduction gate (the gate correctly rejects them -- no false claim), so 'replay' only
    reproduces L1. Evaluating EVERY candidate on a brand-new env (fresh_env mode) makes each see the
    same pristine parity-0 the gate uses, so found paths reproduce. (deepcopy mode does NOT work for
    tu93 -- its env._game is not deepcopy-injectable, gotcha #3, like sc25.)

    VALIDATED 2026-06-17: with branch_mode='fresh_env' the adapter DEEP-SOLVES to L3 reproducibly
    (47 moves, offline_reproduced=True), vs L1 under replay. featurize is None (the learned verifier
    is fed env._game internals via collect_trajectory_data, which this frame-based RE doesn't read)."""
    import numpy as np
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    PLAYER, GOAL = 9, 14

    def _grid2d(frame):
        g = grid_of(frame)
        if g.ndim == 1:                                # some stepped frames flatten -> reshape square
            s = int(round(g.size ** 0.5))
            if s * s == g.size:
                g = g.reshape(s, s)
        return g

    def _centroid(g, col):
        ys, xs = np.where(g == col)
        return (float(xs.mean()), float(ys.mean())) if len(xs) else None

    def action_labels(env, frame=None, path=None):
        av = list(getattr(frame, "available_actions", []) or []) if frame is not None else []
        moves = [a for a in av if a in (1, 2, 3, 4)] or [1, 2, 3, 4]
        return [json.dumps({"action": int(a)}) for a in moves]

    def apply(env, label, frame):
        return env.step(_game_action(GameAction, json.loads(label)["action"]))

    def state_key(game, frame=None):
        # frame-based dedup: the FULL grid (player position + maze + key + the blocked-move counter).
        # Do NOT mask any region -- the corner counter is LOAD-BEARING (masking it collapses distinct
        # states and yields a non-reproducing path; the full-grid hash is what graph_explore reproduces
        # tu93 with).
        if frame is None:
            return None
        return _grid2d(frame).tobytes()

    def hand_verifier(game, frame=None):
        if frame is None:
            return 1000.0
        g = _grid2d(frame)
        p, t = _centroid(g, PLAYER), _centroid(g, GOAL)
        if p is None or t is None:
            return 1000.0
        return abs(p[0] - t[0]) + abs(p[1] - t[1])     # lower == closer to the goal

    return GameAdapter(
        game="tu93", action_labels=action_labels, apply=apply, state_key=state_key,
        featurize=None, hand_verifier=hand_verifier, warmup_label=None,
        depth_caps={1: 40, 2: 60, 3: 80, 4: 90, 5: 90},
        branch_mode="fresh_env",   # gotcha #7: tu93 reset is non-idempotent -> fresh env per node
    )


# ---------------- tr87 (glyph-substitution configuration puzzle; RE'd 2026-06-17) ----------------
def _tr87():
    """tr87 -- a GLYPH-SUBSTITUTION configuration puzzle (RE'd 2026-06-17, frame + internal-state).

    Mechanic: a row of 5 EDITABLE glyphs (sprite series 'B', each a value 1-7) must be set to match a
    TARGET row (series 'A', values 1-7) THROUGH a substitution rule. The visible top reference grid IS
    the rule -- it pairs A-values with B-values (e.g. A4<->B3). Win = for every position i,
    value(editable_i) == rule_map[value(target_i)]. ACTION1/ACTION2 cycle the SELECTED glyph's value
    (-1/+1 mod 7); ACTION3/ACTION4 move the selector among the 5 glyphs. A move budget (128) decrements
    per action; running out loses. (Later levels add alter_rules / tree_translation / double_translation
    twists; L1-L5 of the base mechanic are handled here.)

    The visible reference grid is a REWRITE rule: each target glyph expands to a SEQUENCE of editable
    glyphs (L1: 1-to-1, e.g. A4->[B3]; L2: 1-to-many, e.g. B3->[C1,C5,C1]). The win = the editable
    sequence equals the concat of the rule expansion over the target sequence. The hand_verifier reads
    the game's internal config -- rule map (cifzvbcuwqe) + target (zvojhrjxxm) + current (ztgmtnnufb) --
    and returns the count of editable positions NOT yet at their rule-expanded value (0 == win); the
    SAME internal-state-reading pattern as the lp85 adapter (_goal_key). It routes the best-first search
    to set each glyph to its target.

    VALIDATED: solves L1 (15 moves) AND L2 (1-to-many expansion) reproducibly, offline_reproduced=True.
    L3+ add a tree_translation / double_translation twist where the EDITABLE glyphs also expand (editable
    n != expansion n), which the L1/L2 formula leaves a residual on, so the search stops at L2 rather
    than false-claiming the unmodelled twist -- a clean honest boundary. Frame-only perception
    (classifying glyph bitmaps + decoding the rule grid from pixels) is a future upgrade; the solve is
    reproduction-gated regardless (the gate replays ACTIONS, not internal reads). branch_mode='fresh_env'
    (gotcha #7): the WIN ANIMATION leaves residual state (yfetxjexviz) that a reuse-one-env replay search
    sees but a fresh replay does not, so the reuse-one-env search finds animation-contingent 'wins' that
    FAIL the reproduction gate (it reproduced L1 by luck but not L2). Evaluating each candidate on a fresh
    env makes the search's win-detection match the gate. state_key is the full-grid hash so the
    win-animation frames stay distinct (the search reaches the level-up)."""
    from itertools import product as _product

    from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    _parse_cache: dict = {}                             # alter_rules parse-search memo (fixed per level)

    def _val(s):
        return int(s.name[-1])                          # glyph value = trailing digit of the sprite name

    def _cyc(a, b):
        return min((b - a) % 7, (a - b) % 7)            # cyclic distance over the 7-value wheel

    def _level_flag(game, name):
        try:
            return bool(game.current_level.get_data(name))
        except Exception:
            return False

    def _required_editable(game):
        # GREEDY multi-glyph-LHS rewrite matcher -- mirrors the game's bsqsshqpox win predicate. The
        # visible reference grid is a set of rewrite rules LHS->RHS (LHS over the TARGET series, RHS over
        # the next series). One PASS scans a sequence left-to-right and at each position takes the FIRST
        # rule whose LHS is a prefix, emits its RHS, and advances. The required editable sequence is the
        # rewrite of the target applied PASSES times:
        #   L1-L3 (base):                       1 pass  -- 1-to-1 (A4->[B3]), 1-to-many (B3->[C1,C5,C1]),
        #                                                  many-to-many ([C3,C3]->[A6,A1]).
        #   L4 (double_translation / tree_translation): 2 passes -- a two-level chain A->B->C, so the
        #                                                  editable matches the target rewritten twice.
        # alter_rules (deeper) is a DIFFERENT mechanic (editing the rules, not the glyphs) and is NOT
        # modelled here -- a pass will fail to match and the verifier returns large (search stops, no
        # false claim). Returns None if any pass cannot match a position.
        rules = [([s.name for s in lhs], [s.name for s in rhs]) for lhs, rhs in game.cifzvbcuwqe]

        def _rewrite(seq):
            out, pos = [], 0
            while pos < len(seq):
                for lhs_names, rhs_names in rules:
                    if seq[pos:pos + len(lhs_names)] == lhs_names:
                        out.extend(rhs_names)
                        pos += len(lhs_names)
                        break
                else:
                    return None
            return out

        seq = [s.name for s in game.zvojhrjxxm]
        passes = 2 if (_level_flag(game, "tree_translation") or _level_flag(game, "double_translation")) else 1
        for _ in range(passes):
            seq = _rewrite(seq)
            if seq is None:
                return None
        return [int(n[-1]) for n in seq]

    def _solve_rule_parse(structs, target, editable):
        # ALTER_RULES inverse puzzle: the RULES are editable, target+editable are FIXED. Find rule values
        # so the greedy rewrite of target == editable. Rule STRUCTURE (lhs_len, rhs_len) is fixed; only
        # values change, and all glyphs in a side share one value. RHS is DETERMINED once the LHS values
        # fix the greedy parse, so search only the LHS values (7^nrules) and read RHS off the editable
        # segments. Returns (lhs_vals, {rule_idx: rhs_val}) or None. Cached -- fixed per level.
        key = (structs, target, editable)
        if key in _parse_cache:
            return _parse_cache[key]
        result = None
        for lhs_vals in _product(range(1, 8), repeat=len(structs)):
            pos, parse, ok = 0, [], True
            while pos < len(target):                       # forced greedy parse for this LHS assignment
                for ri, (ll, _rl) in enumerate(structs):
                    if pos + ll <= len(target) and all(target[pos + k] == lhs_vals[ri] for k in range(ll)):
                        parse.append(ri)
                        pos += ll
                        break
                else:
                    ok = False
                    break
            if not ok or pos != len(target):
                continue
            ep, rhs, good = 0, {}, True
            for ri in parse:                               # read RHS off the editable segments
                rl = structs[ri][1]
                seg = editable[ep:ep + rl]
                if len(seg) < rl or len(set(seg)) != 1 or (ri in rhs and rhs[ri] != seg[0]):
                    good = False
                    break
                rhs[ri] = seg[0]
                ep += rl
            if good and ep == len(editable):
                result = (lhs_vals, rhs)
                break
        _parse_cache[key] = result
        return result

    def _rule_sides(game):
        # the 2*nrules editable rule-SIDES in the selector's cycle order: [r0.LHS, r0.RHS, r1.LHS, ...]
        cur: list = []
        for lhs, rhs in game.cifzvbcuwqe:
            cur.append(int(lhs[0].name[-1]))
            cur.append(int(rhs[0].name[-1]))
        return cur

    def _rule_distance(game):
        structs = tuple((len(lhs), len(rhs)) for lhs, rhs in game.cifzvbcuwqe)
        target = tuple(int(s.name[-1]) for s in game.zvojhrjxxm)
        editable = tuple(int(s.name[-1]) for s in game.ztgmtnnufb)
        res = _solve_rule_parse(structs, target, editable)
        if res is None:
            return 1000.0                                  # no valid rule config found -> search stops
        lhs_vals, rhs_assign = res
        cur = _rule_sides(game)
        req: list = []
        for i in range(len(structs)):
            req.append(lhs_vals[i])
            req.append(rhs_assign.get(i, cur[2 * i + 1]))  # unparsed rule's RHS: leave at current (no-op)
        return float(sum(_cyc(c, r) for c, r in zip(cur, req)))

    def _distance(game):
        # alter_rules INVERTS the puzzle: the RULES are editable (selector cycles rule-sides, ACTION1/2
        # edits a rule), the target+editable are FIXED. Route by the cyclic distance of each rule-side to
        # a winning rule config found by _solve_rule_parse.
        if _level_flag(game, "alter_rules"):
            return _rule_distance(game)
        # base (L1-L4): the EDITABLE glyphs are editable; route to the N-pass rewrite of the target.
        req = _required_editable(game)
        if req is None:
            return 1000.0                                  # unmodelled twist -> search stops, no false win
        cur = [_val(s) for s in game.ztgmtnnufb]
        n = min(len(cur), len(req))
        # SUM of per-glyph cyclic distance (NOT a bare mismatch count): gives the best-first search a
        # smooth gradient -- every ACTION1/2 toward target drops the score by 1, so the search walks
        # straight to the win (mismatch-count gave no gradient and exploded at L2's 7 glyphs). The 7x
        # length-gap term bounds the unmodelled-twist case so the search stops with no false claim.
        return float(sum(_cyc(cur[i], req[i]) for i in range(n)) + 7 * abs(len(cur) - len(req)))

    def action_labels(env, frame=None, path=None):
        return [json.dumps({"action": a}) for a in (1, 2, 3, 4)]

    def apply(env, label, frame):
        return env.step(_game_action(GameAction, json.loads(label)["action"]))

    def state_key(game, frame=None):
        # full-grid hash: distinguishes every (selector, glyph-values) config AND the win-animation
        # frames (where the config is frozen but the grid changes), so the search reaches the level-up.
        return frame_hash(grid_of(frame)) if frame is not None else None

    def hand_verifier(game, frame=None):
        # goal-distance = summed cyclic distance of each editable glyph to its rule-expanded target
        # (0 == win). Internal-state read (the lp85 pattern); routes the best-first search. Guarded so a
        # malformed level never crashes.
        try:
            return _distance(game)
        except Exception:
            return 1000.0

    return GameAdapter(
        game="tr87", action_labels=action_labels, apply=apply, state_key=state_key,
        featurize=None, hand_verifier=hand_verifier, warmup_label=None,
        depth_caps={1: 40, 2: 90, 3: 90, 4: 90, 5: 90}, branch_mode="fresh_env",
    )


_BUILDERS = {"lp85": _lp85, "tu93": _tu93, "tr87": _tr87}


def get_adapter(game: str) -> Optional[GameAdapter]:
    """Return the adapter for `game`, or None if it hasn't been RE'd/registered yet."""
    b = _BUILDERS.get(game)
    return b() if b else None


def adaptered_games() -> list[str]:
    return sorted(_BUILDERS)
