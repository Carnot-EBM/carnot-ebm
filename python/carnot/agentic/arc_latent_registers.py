import numpy as np
from collections import defaultdict, Counter
from typing import Optional

from carnot.agentic.arc_world_model_synth import InducedWorldModel, _click_xy, _relative_template, frame_hash

def compute_latent_registers(trajectory):
    """
    Given a trajectory of (grid, akey, next_grid), computes a list of latent state tuples.
    Returns: list of latents of length len(trajectory) + 1.
    Each latent is (step_counter, frozenset(colors_clicked)).
    """
    latents = []
    step = 0
    clicked = set()
    latents.append((step, frozenset(clicked)))
    for grid, akey, next_grid in trajectory:
        step += 1
        if akey[0] == 6:  # click
            x, y = akey[1], akey[2]
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                clicked.add(int(grid[y, x]))
        latents.append((step, frozenset(clicked)))
    return latents


def grade_predictions_augmented(predict_fn, held_out) -> dict:
    n = 0
    exact_hit = 0
    dyn_acc_sum = 0.0
    n_changed = 0
    for s, latent, akey, s2, latent2 in held_out:
        s = np.asarray(s, dtype=np.int16)
        s2 = np.asarray(s2, dtype=np.int16)
        pred = np.asarray(predict_fn(s, latent, tuple(akey)), dtype=np.int16)
        n += 1
        if pred.shape != s2.shape:
            n_changed += 1
            continue
        exact_hit += int(np.array_equal(pred, s2))
        real_changed = (s != s2)
        if real_changed.any():
            n_changed += 1
            union = real_changed | (s != pred)
            dyn_acc_sum += float(((pred == s2) & union).sum() / union.sum())
    if n == 0:
        return {"energy": None, "n_heldout": 0}
    dynamics_accuracy = round(dyn_acc_sum / n_changed, 4) if n_changed else None
    return {
        "energy": round(1.0 - dynamics_accuracy, 4) if dynamics_accuracy is not None else None,
        "dynamics_accuracy": dynamics_accuracy,
        "n_changed_transitions": n_changed,
        "transition_exact_rate": round(exact_hit / n, 4),
        "n_heldout": n,
    }


class AugmentedInducedWorldModel(InducedWorldModel):
    def fit_augmented(self, transitions):
        for s, latent, akey, s2, latent2 in transitions:
            s = np.asarray(s, dtype=np.int16)
            s2 = np.asarray(s2, dtype=np.int16)
            if self._shape is None:
                self._shape = s.shape
            akey = tuple(akey)
            fh = (frame_hash(s), latent)
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

    def predict_augmented(self, s_grid, latent, akey: tuple) -> np.ndarray:
        s = np.asarray(s_grid, dtype=np.int16)
        akey = tuple(akey)
        fh = (frame_hash(s), latent)
        ex = self._exact.get((fh, akey))
        if ex:
            b2 = ex.most_common(1)[0][0]
            return self._exact_grid[b2].copy()
        
        xy = _click_xy(akey)
        if xy is not None:
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
            return s.copy()
        kb = self._kbd_tpl.get(akey[0])
        if kb:
            abs_delta = kb.most_common(1)[0][0]
            out = s.copy()
            for (cy, cx, new) in abs_delta:
                if 0 <= cy < s.shape[0] and 0 <= cx < s.shape[1]:
                    out[cy, cx] = new
            return out
        return s.copy()

    def consistency_energy_augmented(self, held_out) -> dict:
        return grade_predictions_augmented(self.predict_augmented, held_out)
