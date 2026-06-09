"""M2-v2: object-level delta-DSL world-model inducer for ARC-AGI-3.

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2). M2-v1a
(results/arc3_m2_world_model.json) ruled out the naive per-pixel template inducer: it could not
generalize to unseen clicks even on deterministic games (generalization AUROC 0.49, energy ~0.95).
The diagnosis: a RELATIVE-PIXEL template can't capture position-independent OBJECT mechanics. This
module induces rules at the OBJECT level, which generalize across positions:

  - keyboard action a -> ('translate', color, dy, dx): all cells of `color` move by (dy, dx)
    (the classic agent/cursor move), or ('recolor_all', c_from, c_to), or ('noop',).
  - click on a cell of color C -> ('recolor_clicked', c_to): the clicked connected component
    recolors to c_to (selection / toggle / paint), or ('noop',). Keyed by the CLICKED color so it
    generalizes to unseen click locations.

Induction is MDL-flavored: for each action family, enumerate candidate rules from the observed deltas
and keep the one with the highest grid-grounded dynamics accuracy on the training transitions (the
simplest rule that best reproduces reality). The SAME grade_predictions() verifier from
arc_world_model_synth scores held-out generalization, so M2-v2 is measured against the M2-v1a naive
baseline on the identical metric. Generator (this inducer) proposes the program; the consistency
energy (verifier) prunes/certifies it. Deterministic numpy; no LLM, no GPU, no oracle.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from .arc_agi3_world_model import frame_hash
from .arc_world_model_synth import grade_predictions, _click_xy


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _connected_component(grid: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    """Cells of the same color 4-connected to (y, x)."""
    h, w = grid.shape
    if not (0 <= y < h and 0 <= x < w):
        return []
    color = grid[y, x]
    seen = {(y, x)}
    stack = [(y, x)]
    out = []
    while stack:
        cy, cx = stack.pop()
        out.append((cy, cx))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == color and (ny, nx) not in seen:
                seen.add((ny, nx)); stack.append((ny, nx))
    return out


def _detect_translation(s: np.ndarray, s2: np.ndarray, color: int) -> Optional[tuple[int, int]]:
    """If every cell of `color` in s moved by a single consistent (dy, dx) to form s2's `color` set,
    return (dy, dx); else None. (0, 0) is returned as None (no move)."""
    cs = list(zip(*np.where(s == color)))
    cs2 = set(zip(*np.where(s2 == color)))
    if not cs or len(cs) != len(cs2):
        return None
    miny_s = min(y for y, _ in cs); minx_s = min(x for _, x in cs)
    miny_2 = min(y for y, _ in cs2); minx_2 = min(x for _, x in cs2)
    dy, dx = miny_2 - miny_s, minx_2 - minx_s
    if dy == 0 and dx == 0:
        return None
    if {(y + dy, x + dx) for y, x in cs} == cs2:
        return (int(dy), int(dx))
    return None


class ObjectDeltaModel:
    """Object-level delta-DSL world-model: induces per-action translate/recolor rules + an exact table
    for memorized transitions, with the shared grid-grounded consistency-energy verifier."""

    def __init__(self, game_id: str = "?"):
        self.game_id = game_id
        self._exact: dict[tuple, Counter] = defaultdict(Counter)
        self._exact_grid: dict[bytes, np.ndarray] = {}
        self.kbd_rules: dict[int, tuple] = {}        # action_int -> rule
        self.click_rules: dict[int, tuple] = {}      # clicked_color -> rule
        self.bg = 0

    # ---- rule application ----
    def _apply(self, s: np.ndarray, rule: tuple) -> np.ndarray:
        kind = rule[0]
        if kind == "translate":
            _, color, dy, dx = rule
            out = s.copy()
            mask = (s == color)
            out[mask] = self.bg
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                ny, nx = y + dy, x + dx
                if 0 <= ny < s.shape[0] and 0 <= nx < s.shape[1]:
                    out[ny, nx] = color
            return out
        if kind == "recolor_all":
            _, c_from, c_to = rule
            out = s.copy(); out[s == c_from] = c_to
            return out
        return s.copy()                               # noop

    def _apply_click(self, s: np.ndarray, x: int, y: int, rule: tuple) -> np.ndarray:
        if rule[0] == "recolor_clicked":
            out = s.copy()
            for (cy, cx) in _connected_component(s, y, x):
                out[cy, cx] = rule[1]
            return out
        return s.copy()

    def _dyn_acc(self, pred: np.ndarray, s: np.ndarray, s2: np.ndarray) -> float:
        real = (s != s2)
        if not real.any():
            return 1.0 if np.array_equal(pred, s2) else 0.0
        union = real | (s != pred)
        return float(((pred == s2) & union).sum() / union.sum())

    # ---- induction ----
    def fit(self, transitions) -> "ObjectDeltaModel":
        kbd: dict[int, list] = defaultdict(list)
        clk: list = []
        bgs = Counter()
        for s, akey, s2 in transitions:
            s = np.asarray(s, dtype=np.int16); s2 = np.asarray(s2, dtype=np.int16)
            akey = tuple(akey)
            bgs[_background(s)] += 1
            fh = frame_hash(s)
            b2 = s2.astype(np.uint8).tobytes()
            self._exact[(fh, akey)][b2] += 1
            self._exact_grid[b2] = s2
            if _click_xy(akey) is not None:
                clk.append((s, akey, s2))
            else:
                kbd[akey[0]].append((s, s2))
        self.bg = bgs.most_common(1)[0][0] if bgs else 0

        # keyboard rules: pick the candidate maximizing mean dynamics accuracy over that action
        for a, pairs in kbd.items():
            cands = {("noop",)}
            for s, s2 in pairs:
                for color in np.unique(s):
                    if int(color) == self.bg:
                        continue
                    t = _detect_translation(s, s2, int(color))
                    if t:
                        cands.add(("translate", int(color), t[0], t[1]))
                # a global recolor candidate (c_from -> c_to) if exactly one color remapped
                diff_colors = set(zip(s[s != s2].tolist(), s2[s != s2].tolist()))
                if len(diff_colors) == 1:
                    cf, ct = next(iter(diff_colors))
                    cands.add(("recolor_all", int(cf), int(ct)))
            best, best_acc = ("noop",), -1.0
            for r in cands:
                acc = np.mean([self._dyn_acc(self._apply(s, r), s, s2) for s, s2 in pairs])
                # tie-break toward simpler rules (noop < recolor < translate already by acc); prefer higher acc
                if acc > best_acc:
                    best_acc, best = acc, r
            self.kbd_rules[a] = best

        # click rules: per clicked color, modal recolor target of the clicked component
        by_color: dict[int, Counter] = defaultdict(Counter)
        for s, akey, s2 in clk:
            x, y = akey[1], akey[2]
            if not (0 <= y < s.shape[0] and 0 <= x < s.shape[1]):
                continue
            c = int(s[y, x])
            comp = _connected_component(s, y, x)
            targets = {int(s2[cy, cx]) for cy, cx in comp}
            cand = ("recolor_clicked", targets.pop()) if len(targets) == 1 else ("noop",)
            by_color[c][cand] += 1
        for c, counter in by_color.items():
            self.click_rules[c] = counter.most_common(1)[0][0]
        return self

    def predict(self, s_grid, akey: tuple) -> np.ndarray:
        s = np.asarray(s_grid, dtype=np.int16)
        akey = tuple(akey)
        ex = self._exact.get((frame_hash(s), akey))
        if ex:                                        # memorized seen transition
            return self._exact_grid[ex.most_common(1)[0][0]].copy()
        xy = _click_xy(akey)
        if xy is not None:
            x, y = xy
            if 0 <= y < s.shape[0] and 0 <= x < s.shape[1]:
                rule = self.click_rules.get(int(s[y, x]), ("noop",))
                return self._apply_click(s, x, y, rule)
            return s.copy()
        return self._apply(s, self.kbd_rules.get(akey[0], ("noop",)))

    def consistency_energy(self, held_out) -> dict:
        return grade_predictions(self.predict, held_out)
