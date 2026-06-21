"""Button-discovery sweep for lp85 (generalizes to any click-UI game): from the deterministic
reset state, click EVERY grid position, record which positions change the grid and the effect
signature, then cluster positions into distinct buttons and characterize what each moves.

Turns the salience-clicker's 2/80 coverage into systematic coverage. CPU-only, no LLM, no GPU.
Reusable primitive -> belongs in arc_solver_kit.discover_actuators().
"""
import json, hashlib
import numpy as np
import scipy.ndimage as ndi
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_over, _game_action
from carnot.agentic.arc_executable_world_model import to_logical, detect_cell

GAME = "lp85"
arc = kit.offline_arcade(); env = arc.make(GAME, scorecard_id=arc.open_scorecard())
f0 = env.reset(); cell = detect_cell(grid_of(f0)); BASE = to_logical(grid_of(f0), cell)
H, W = BASE.shape

def sig(g):
    """signature of the change vs BASE = sorted (r,c,newval) of changed cells, hashed."""
    d = np.argwhere(BASE != g)
    if len(d) == 0: return None
    return hashlib.md5(str([(int(r), int(c), int(g[r, c])) for r, c in d]).encode()).hexdigest()[:10]

# ---- SWEEP: reset+click every (x,y), record effective positions ----
eff = {}          # (x,y) -> signature
sig_changes = {}  # signature -> (n_changed, example grid)
print(f"sweeping {H*W} positions...", flush=True)
for y in range(H):
    for x in range(W):
        env.reset()
        nf = env.step(_game_action(GameAction, 6), data={'x': x, 'y': y})
        if nf is None: continue
        g = to_logical(grid_of(nf), cell)
        if not np.array_equal(BASE, g):
            s = sig(g)
            eff[(x, y)] = s
            if s not in sig_changes:
                sig_changes[s] = (int((BASE != g).sum()), g.copy())
print(f"effective click positions: {len(eff)}", flush=True)
print(f"DISTINCT effects (= distinct buttons reachable from initial state): {len(sig_changes)}", flush=True)

# ---- cluster effective positions spatially into button regions ----
mask = np.zeros((H, W), bool)
for (x, y) in eff: mask[y, x] = True
lbl, n_regions = ndi.label(mask)
print(f"spatially-connected click regions (button hitboxes): {n_regions}", flush=True)

# ---- characterize each distinct effect: which piece moved (object-displacement) ----
def obj_disp(g):
    bg = int(np.bincount(BASE.flatten()).argmax())
    moves = []
    colors = sorted(set(np.unique(BASE).tolist()) | set(np.unique(g).tolist()))
    for color in colors:
        if color == bg: continue
        def cents(grid):
            l, k = ndi.label(grid == color)
            if k == 0: return []
            return sorted((int(round(yy)), int(round(xx))) for yy, xx in ndi.center_of_mass(grid == color, l, range(1, k+1)))
        b, a = cents(BASE), cents(g)
        if b != a: moves.append((color, len(b), len(a)))
    return moves

# distinct piece-effects
print("\nsample of distinct buttons (effect signature -> #cells changed, region centroid, what moved):", flush=True)
region_centroids = ndi.center_of_mass(mask, lbl, range(1, n_regions+1))
shown = 0
for s, (ncell, g) in list(sig_changes.items())[:12]:
    md = obj_disp(g)
    print(f"  effect {s}: {ncell} cells changed; colors-moved={md[:4]}", flush=True)
    shown += 1

# ---- summary + save reusable button map ----
btn_map = {f"{x},{y}": eff[(x, y)] for (x, y) in eff}
out = {
    "game": GAME, "grid": [H, W],
    "positions_swept": H*W,
    "effective_positions": len(eff),
    "distinct_buttons_from_initial": len(sig_changes),
    "button_hitbox_regions": int(n_regions),
    "source_total_buttons": 80,
    "salience_clicker_found": 2,
    "button_position_to_effect": btn_map,
}
json.dump(out, open("/tmp/lp85_button_map.json", "w"), indent=1)
print(f"\n=== SUMMARY ===", flush=True)
print(f"  source has 80 buttons; salience-clicker found 2; SYSTEMATIC SWEEP found "
      f"{n_regions} hitbox regions / {len(sig_changes)} distinct effects from the initial state", flush=True)
print(f"  ({len(eff)} effective pixels of {H*W} swept). Button map saved to /tmp/lp85_button_map.json", flush=True)
