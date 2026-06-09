"""M2: induced world-model + GRID-GROUNDED consistency-energy verifier for ARC-AGI-3.

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2, Family-B). The determinism probe
(results/arc3_determinism_probe.json) split the 25 offline games: 14 are grid-Markov (visible grid +
action determines the next grid) and 11 are hidden-state. This module is the first place the CARNOT
ENERGY VERIFIER does real, non-tautological work on real ARC games:

  - InducedWorldModel learns a transition function predict(grid, action) -> grid from observed
    (state, action, next_state) transitions: an EXACT table for seen (frame_hash, action) plus a
    learned per-action delta TEMPLATE (keyed by the clicked cell's color) that GENERALIZES to unseen
    clicks. No oracle, no ground-truth goal — only what the agent observed.

  - consistency_energy(model, held_out) = the model's MISPREDICTION RATE on HELD-OUT real transitions
    it did not train on. 0 = the model perfectly reproduces reality (trustworthy -> safe to plan on);
    high = the model cannot predict the dynamics (untrustworthy -> the Meta-EBM cascade router must
    escalate to latent-state modeling or a soft energy, NOT plan on a wrong model).

This is the honest contrast with the flagged exp3929 tautology: there the verifier "scored" a planted
arithmetic-contradiction string the encoder wrote from oracle ground truth. HERE the verifier predicts
a grid and is graded against the OBSERVED next grid it never saw — a wrong prediction is fully
possible, so the energy is doing real work. The verifier's load-bearing claim is testable: its
consistency_energy should be LOW on the 14 Markov games and HIGH on the 11 hidden-state games,
matching the determinism-probe ground truth.

Perception/induction is deterministic numpy; the only learning is frequency-counting delta templates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from .arc_agi3_world_model import frame_hash


def _click_xy(akey: tuple) -> Optional[tuple[int, int]]:
    return (akey[1], akey[2]) if len(akey) == 3 and akey[0] == 6 else None


def _relative_template(s: np.ndarray, s2: np.ndarray, x: int, y: int) -> tuple:
    """Express the (s -> s2) change as offsets RELATIVE to the click (x, y): a sorted tuple of
    (dy, dx, new_color). This is the click's local effect, position-independent, so it can be applied
    at a new click location to predict an unseen click."""
    diff = np.argwhere(s != s2)
    rel = []
    for (cy, cx) in diff:
        rel.append((int(cy) - y, int(cx) - x, int(s2[cy, cx])))
    return tuple(sorted(rel))


class InducedWorldModel:
    """Deterministic-delta world-model induced from observed transitions, with a grid-grounded
    consistency-energy self-assessment. The 'energy' is held-out misprediction rate (0 = perfect)."""

    def __init__(self, game_id: str = "?"):
        self.game_id = game_id
        # exact memory: (frame_hash, akey) -> Counter of next-grid-bytes (multiplicity tracks determinism)
        self._exact: dict[tuple, Counter] = defaultdict(Counter)
        self._exact_grid: dict[bytes, np.ndarray] = {}            # next-grid-bytes -> grid (for reconstruct)
        # generalizer: click template keyed by clicked-cell color -> modal relative delta
        self._click_tpl: dict[int, Counter] = defaultdict(Counter)
        # generalizer: keyboard action -> modal absolute delta (sorted (cy,cx,new)) if consistent
        self._kbd_tpl: dict[int, Counter] = defaultdict(Counter)
        self._shape: Optional[tuple[int, int]] = None
        self.n_train = 0

    def fit(self, transitions) -> "InducedWorldModel":
        """transitions: iterable of (s_grid, akey, s2_grid) numpy/int grids."""
        for s, akey, s2 in transitions:
            s = np.asarray(s, dtype=np.int16); s2 = np.asarray(s2, dtype=np.int16)
            if self._shape is None:
                self._shape = s.shape
            akey = tuple(akey)
            fh = frame_hash(s)
            b2 = s2.astype(np.uint8).tobytes()
            self._exact[(fh, akey)][b2] += 1
            self._exact_grid[b2] = s2
            xy = _click_xy(akey)
            if xy is not None:
                x, y = xy
                clicked = int(s[y, x]) if (0 <= y < s.shape[0] and 0 <= x < s.shape[1]) else -1
                self._click_tpl[clicked][_relative_template(s, s2, x, y)] += 1
            elif akey[0] != 6:
                diff = np.argwhere(s != s2)
                abs_delta = tuple(sorted((int(cy), int(cx), int(s2[cy, cx])) for cy, cx in diff))
                self._kbd_tpl[akey[0]][abs_delta] += 1
            self.n_train += 1
        return self

    def predict(self, s_grid, akey: tuple) -> np.ndarray:
        """Predict the next grid for (s_grid, akey). Exact table first (modal next state if it was
        ever multivalued), then the learned generalizer, then no-op fallback."""
        s = np.asarray(s_grid, dtype=np.int16)
        akey = tuple(akey)
        fh = frame_hash(s)
        ex = self._exact.get((fh, akey))
        if ex:                                          # seen this exact (state, action) before
            b2 = ex.most_common(1)[0][0]
            return self._exact_grid[b2].copy()
        xy = _click_xy(akey)
        if xy is not None:                              # generalize an unseen click via its color-template
            x, y = xy
            if 0 <= y < s.shape[0] and 0 <= x < s.shape[1]:
                clicked = int(s[y, x])
                tpl = self._click_tpl.get(clicked)
                if tpl:
                    rel = tpl.most_common(1)[0][0]
                    out = s.copy()
                    for (dy, dx, new) in rel:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < s.shape[0] and 0 <= nx < s.shape[1]:
                            out[ny, nx] = new
                    return out
            return s.copy()                             # unknown clicked color -> predict no-op
        kb = self._kbd_tpl.get(akey[0])                 # generalize a keyboard action via its modal delta
        if kb:
            abs_delta = kb.most_common(1)[0][0]
            out = s.copy()
            for (cy, cx, new) in abs_delta:
                if 0 <= cy < s.shape[0] and 0 <= cx < s.shape[1]:
                    out[cy, cx] = new
            return out
        return s.copy()                                 # nothing learned -> no-op

    def consistency_energy(self, held_out) -> dict:
        """GRID-GROUNDED verifier: predict each held-out transition, grade against the OBSERVED next
        grid. The static background dominates a 64x64 grid, so whole-grid exact-match is too brittle
        and cell-accuracy is trivially ~1.0; the load-bearing signal is over the CHANGED region (the
        dynamics). We report:
          - energy_exact   : 1 - whole-grid exact-match rate (reference; brittle)
          - cell_accuracy  : mean per-cell match (reference; background-dominated)
          - dynamics_accuracy : on transitions that ACTUALLY changed in reality, the fraction of the
                changed-region (reality's delta UNION the model's predicted delta) the model gets right.
                A model that mispredicts the change, or predicts no-op when reality changed, scores low.
          - energy = 1 - dynamics_accuracy  : THE headline. 0 = the model captures the dynamics
                (trustworthy -> plan on it); high = it cannot (untrustworthy -> escalate)."""
        n = 0
        exact_hit = 0
        cell_acc_sum = 0.0
        dyn_acc_sum = 0.0
        n_changed = 0
        by_path = Counter()
        for s, akey, s2 in held_out:
            s = np.asarray(s, dtype=np.int16); s2 = np.asarray(s2, dtype=np.int16)
            akey = tuple(akey)
            fh = frame_hash(s)
            if self._exact.get((fh, akey)):
                by_path["exact"] += 1
            elif _click_xy(akey) is not None and self._click_tpl.get(
                    int(s[akey[2], akey[1]]) if (0 <= akey[2] < s.shape[0] and 0 <= akey[1] < s.shape[1]) else -1):
                by_path["click_template"] += 1
            elif akey[0] != 6 and self._kbd_tpl.get(akey[0]):
                by_path["kbd_template"] += 1
            else:
                by_path["noop_fallback"] += 1
            pred = self.predict(s, akey)
            n += 1
            if pred.shape != s2.shape:
                n_changed += 1  # shape mismatch = a real change the model failed to predict
                continue
            exact_hit += int(np.array_equal(pred, s2))
            cell_acc_sum += float((pred == s2).mean())
            real_changed = (s != s2)
            if real_changed.any():                       # only score transitions that actually changed
                n_changed += 1
                pred_changed = (s != pred)
                union = real_changed | pred_changed       # cells either side thinks changed
                dyn_acc_sum += float(((pred == s2) & union).sum() / union.sum())
        if n == 0:
            return {"energy": None, "n_heldout": 0}
        dynamics_accuracy = round(dyn_acc_sum / n_changed, 4) if n_changed else None
        return {
            "energy": round(1.0 - dynamics_accuracy, 4) if dynamics_accuracy is not None else None,
            "dynamics_accuracy": dynamics_accuracy,
            "n_changed_transitions": n_changed,
            "energy_exact": round(1.0 - exact_hit / n, 4),
            "transition_exact_rate": round(exact_hit / n, 4),
            "cell_accuracy": round(cell_acc_sum / n, 4),
            "n_heldout": n,
            "prediction_paths": dict(by_path),
        }

    def is_trustworthy(self, held_out, energy_threshold: float = 0.2) -> bool:
        """The Meta-EBM cascade gate: trust the model for planning only if held-out (dynamics) energy
        is low. High energy -> escalate to latent-state modeling / soft energy, do NOT plan."""
        e = self.consistency_energy(held_out)
        return e["energy"] is not None and e["energy"] <= energy_threshold
