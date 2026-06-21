"""Object-displacement + RLE encoding for ARC world-model induction prompts.

Motivation (lp85 = conveyor-ring rotation): per-cell (row,col,from,to) deltas (capped OR full)
hide the permutation structure and drive the model to memorize cells (416-line, non-completing
engines; DSL held-out plateaus ~0.17). This encoder shows each transition as:
  1. RLE delta  -- LOSSLESS ground truth (horizontal runs), ~0.48x the tokens of raw tuples.
  2. OBJECT hint -- per-color connected-component centroids before->after, revealing the SHIFT.
Plus it raises coverage from 6 -> K_CHANGING shown changing transitions (held-out climbs with k).
RLE is the ground truth; the object hint is advisory (a wrong segmentation can't corrupt the
lossless RLE). Drop-in for _transitions_block via monkeypatch.
"""
import numpy as np
import scipy.ndimage as ndi


def _rle_delta(g0, g1):
    """Lossless: maximal horizontal runs of changed cells. Each run = (row, col0, [new_vals])."""
    g0 = np.asarray(g0); g1 = np.asarray(g1)
    if g0.shape != g1.shape:
        return []
    diff = g0 != g1
    H, W = g0.shape
    runs = []
    for r in range(H):
        c = 0
        while c < W:
            if diff[r, c]:
                c0 = c
                while c < W and diff[r, c]:
                    c += 1
                runs.append((r, c0, [int(v) for v in g1[r, c0:c]]))
            else:
                c += 1
    return runs


def _rle_apply(g0, runs):
    """Reconstruct g1 from g0 + RLE runs (proves losslessness)."""
    g = np.asarray(g0).copy()
    for r, c0, vals in runs:
        g[r, c0:c0 + len(vals)] = vals
    return g


def _rle_str(runs, cap_runs=120):
    parts = [f"r{r}c{c0}:{''.join(str(v) for v in vals)}" for r, c0, vals in runs[:cap_runs]]
    extra = f" (+{len(runs)-cap_runs} more runs)" if len(runs) > cap_runs else ""
    return " ".join(parts) + extra


def _bg(g):
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


def _object_hint(g0, g1):
    """Per non-background color, connected-component centroids BEFORE -> AFTER (only colors whose
    layout changed). Reveals 'pieces shifted by one slot' for a rotation. Advisory hint."""
    g0 = np.asarray(g0); g1 = np.asarray(g1)
    bg = _bg(g0)
    colors = sorted(set(np.unique(g0).tolist()) | set(np.unique(g1).tolist()))
    out = []
    for color in colors:
        if color == bg:
            continue
        def cents(g):
            lbl, n = ndi.label(g == color)
            if n == 0:
                return []
            cs = ndi.center_of_mass(g == color, lbl, range(1, n + 1))
            return sorted((int(round(y)), int(round(x))) for y, x in cs)
        b, a = cents(g0), cents(g1)
        if b != a:
            # compact: show up to 12 centroids each side
            bs = ",".join(f"({y},{x})" for y, x in b[:12]) + ("..." if len(b) > 12 else "")
            as_ = ",".join(f"({y},{x})" for y, x in a[:12]) + ("..." if len(a) > 12 else "")
            out.append(f"  color {color}: [{bs}] -> [{as_}]")
    return out


def make_transitions_block(to_ascii, K_CHANGING=20):
    """Return a _transitions_block(trans, k=...) drop-in using RLE + object hint + higher coverage."""
    def _block(trans, k=8):
        changed = [t for t in trans if not np.array_equal(t.grid, t.next_grid)]
        noop = [t for t in trans if np.array_equal(t.grid, t.next_grid)]
        sample = changed[:K_CHANGING] + noop[:2]
        out = []
        if sample:
            out.append("INITIAL GRID (one full example of the state layout; all grids are this shape):\n"
                       + to_ascii(sample[0].grid))
            out.append("\nTransitions are shown as: (1) RLE = lossless changed cells as horizontal runs "
                       "'r<row>c<col>:<new_values>'; (2) OBJECTS = per-color connected-component centroids "
                       "BEFORE->AFTER (watch how pieces move/shift). Induce the GENERAL rule from how the "
                       "objects move, then verify it reproduces the RLE.")
        for t in sample:
            click = f" data={t.data}" if t.data else ""
            if np.array_equal(t.grid, t.next_grid):
                out.append(f"--- ACTION{t.action}{click} (level {t.level_before}->{t.level_after}): NO-OP (grid unchanged)")
                continue
            runs = _rle_delta(t.grid, t.next_grid)
            hint = _object_hint(t.grid, t.next_grid)
            out.append(f"--- ACTION{t.action}{click} (level {t.level_before}->{t.level_after}):")
            out.append(f"  RLE: {_rle_str(runs)}")
            if hint:
                out.append("  OBJECTS (centroids before->after):")
                out.extend(hint)
        win = next((t for t in trans if t.level_after > t.level_before), None)
        if win is not None:
            out.append("WIN STATE (full grid of a level-complete state — is_level_complete must return True here):\n"
                       + to_ascii(win.next_grid))
        return "\n".join(out)
    return _block
