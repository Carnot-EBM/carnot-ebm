import numpy as np
from typing import Callable, List, Optional
from carnot.agentic.arc_agi3_world_model import objects

def induce_goal_predicate(win_grids: List[np.ndarray], non_win_grids: List[np.ndarray]) -> Optional[Callable[[np.ndarray], bool]]:
    """
    Induce a grid-grounded goal predicate from positive (win) and negative (non-win) examples.
    Requires at least 2 win examples to avoid single-example mis-induction.
    """
    if len(win_grids) < 2:
        return None
        
    win_obj_counts = [len(objects(g)) for g in win_grids]
    non_win_obj_counts = [len(objects(g)) for g in non_win_grids]
    
    max_win_objs = max(win_obj_counts)
    min_non_win_objs = min(non_win_obj_counts) if non_win_grids else float('inf')
    
    # Hypothesis 1: The goal is to reduce the number of objects (e.g. coincidence/merging)
    if max_win_objs < min_non_win_objs:
        return lambda g: len(objects(g)) <= max_win_objs

    # Hypothesis 2: The goal is exactly a specific number of objects
    if len(set(win_obj_counts)) == 1 and (not non_win_grids or win_obj_counts[0] not in non_win_obj_counts):
        target_objs = win_obj_counts[0]
        return lambda g: len(objects(g)) == target_objs
        
    win_colors = [len(np.unique(g)) for g in win_grids]
    non_win_colors = [len(np.unique(g)) for g in non_win_grids]
    
    max_win_colors = max(win_colors)
    min_non_win_colors = min(non_win_colors) if non_win_grids else float('inf')

    # Hypothesis 3: The goal is to reduce the number of unique colors (e.g. covering all targets)
    if max_win_colors < min_non_win_colors:
        return lambda g: len(np.unique(g)) <= max_win_colors
        
    # Hypothesis 4: Specific colors disappear (e.g. targets get covered)
    non_win_color_sets = [set(np.unique(g)) for g in non_win_grids]
    win_color_sets = [set(np.unique(g)) for g in win_grids]
    
    if non_win_color_sets and win_color_sets:
        # Colors present in EVERY non-win grid
        common_non_win_colors = set.intersection(*non_win_color_sets)
        # Colors present in ANY win grid
        all_win_colors = set.union(*win_color_sets)
        
        # Colors that disappear in ALL win grids
        disappearing_colors = common_non_win_colors - all_win_colors
        
        if disappearing_colors:
            return lambda g: len(disappearing_colors.intersection(set(np.unique(g)))) == 0

    # Hypothesis 5: The goal is a strict reduction in objects compared to the STARTING state.
    # But since we just get individual grids, we can just say "objects <= max_win_objs"
    # Even if it overlaps with SOME non_wins, it might still have high precision/recall!
    # Let's return this as a fallback.
    return lambda g: len(objects(g)) <= max_win_objs


def induce_goal_energy(
    win_grids: List[np.ndarray], non_win_grids: List[np.ndarray]
) -> Optional[Callable[[np.ndarray], float]]:
    """GRADED counterpart of ``induce_goal_predicate`` (2026-06-23, wires exp4020 is_goal as a goal-ENERGY,
    closes GAP-ARCH-GOAL-NOT-VERIFIED). Returns ``energy(grid) -> float >= 0`` where 0 means the induced
    goal is SATISFIED and a higher value is the VIOLATION MAGNITUDE (how far the grid is from the goal),
    using the SAME object-count / colour hypotheses as the binary predicate. A graded energy lets a planner
    DESCEND toward the win (best-first), where the binary predicate only marks the terminal state.

    SAFETY (per the .426 directive): the exp4020 frozen r11l predicate is precision-1.0 but n=6 on ONE
    game; wiring it as universal is silent-failure-prone. The intended use is PER-GAME induction from the
    agent's OWN observed level-up (win) states + non-win states -- self-discovered, not a frozen transfer.
    Returns None when there are <2 win examples (cannot induce). Callers MUST run an ablation control and
    fall back to plain BFS if the energy mis-orders (does not reach the win faster than navigation-only)."""
    if len(win_grids) < 2:
        return None
    win_obj = [len(objects(g)) for g in win_grids]
    non_win_obj = [len(objects(g)) for g in non_win_grids]
    max_win_objs = max(win_obj)
    min_non_win_objs = min(non_win_obj) if non_win_grids else float("inf")
    # H1: goal = reduce objects (energy = objects above the win ceiling)
    if max_win_objs < min_non_win_objs:
        return lambda g: float(max(0, len(objects(g)) - max_win_objs))
    # H2: goal = exactly N objects (energy = |distance| to that count)
    if len(set(win_obj)) == 1 and (not non_win_grids or win_obj[0] not in non_win_obj):
        target = win_obj[0]
        return lambda g: float(abs(len(objects(g)) - target))
    win_colors = [len(np.unique(g)) for g in win_grids]
    non_win_colors = [len(np.unique(g)) for g in non_win_grids]
    max_win_colors = max(win_colors)
    min_non_win_colors = min(non_win_colors) if non_win_grids else float("inf")
    # H3: goal = reduce unique colours (energy = colours above the win ceiling)
    if max_win_colors < min_non_win_colors:
        return lambda g: float(max(0, len(np.unique(g)) - max_win_colors))
    # H4: goal = specific colours disappear (energy = count of disappearing colours still present)
    if non_win_grids and win_grids:
        common_non_win = set.intersection(*[set(np.unique(g)) for g in non_win_grids])
        all_win = set.union(*[set(np.unique(g)) for g in win_grids])
        disappearing = common_non_win - all_win
        if disappearing:
            return lambda g: float(len(disappearing.intersection(set(np.unique(g).tolist()))))
    # H5 fallback (same as the predicate's fallback, graded)
    return lambda g: float(max(0, len(objects(g)) - max_win_objs))


def induce_goal_energy_single_positive(
    win_grid: Optional[np.ndarray], non_win_grids: List[np.ndarray]
) -> Optional[Callable[[np.ndarray], float]]:
    """Single-WIN-exemplar goal-energy (GAP-4890, 2026-06-27 — the within-game L2->L3 deepening unblock).

    WHY: ``induce_goal_energy`` returns None with <2 win grids, but at a game's solved frontier the agent
    has only ONE level-completion exemplar (verified empirically on cd82:
    results/arc_within_game_l3_self_induction_cd82_stage1.json). The >=2 floor was a conservatism guard
    against single-example mis-induction, NOT a mathematical necessity — H1/H3/H4 already use only the win's
    feature value plus the NON-WIN distribution. This relaxes the floor to ONE win while KEEPING the
    anti-mis-induction guard a different way: a hypothesis fires ONLY when the lone win is STRICTLY
    separated from EVERY negative on that feature (so a single accidental win cannot mis-induce a goal that
    the negatives also satisfy). Returns ``energy(grid) -> float >= 0`` (0 == goal satisfied), or None when
    no feature separates the win from the negatives — an HONEST "cannot induce", not a fabricated goal.

    Contract matches ``induce_goal_energy`` so it drops into the graph_explore_solve_v2 ``goal_energy`` hook.
    Per the goal-induction doctrine the CALLER MUST still run the BFS-only ablation: the energy only counts
    if it reaches the win faster than navigation-only."""
    if win_grid is None or not non_win_grids:
        return None
    w_obj = len(objects(win_grid))
    n_obj = [len(objects(g)) for g in non_win_grids]
    # H1: win has strictly FEWER objects than every negative -> goal = reduce objects to the win's count.
    if w_obj < min(n_obj):
        return lambda g: float(max(0, len(objects(g)) - w_obj))
    # H2: win has strictly MORE objects than every negative -> goal = grow objects to the win's count.
    if w_obj > max(n_obj):
        return lambda g: float(max(0, w_obj - len(objects(g))))
    w_col = len(np.unique(win_grid))
    n_col = [len(np.unique(g)) for g in non_win_grids]
    # H3: win has strictly FEWER unique colours than every negative -> reduce colours to the win's count.
    if w_col < min(n_col):
        return lambda g: float(max(0, len(np.unique(g)) - w_col))
    # H4: colours present in EVERY negative but ABSENT from the win -> goal = make them disappear.
    common_non_win = set.intersection(*[set(np.unique(g).tolist()) for g in non_win_grids])
    win_cols = set(np.unique(win_grid).tolist())
    disappearing = common_non_win - win_cols
    if disappearing:
        return lambda g: float(len(disappearing.intersection(set(np.unique(g).tolist()))))
    # H5: exact object-count match, but ONLY if the win's count never appears among the negatives
    # (otherwise the energy would read 0 on a negative -> mis-induction).
    if w_obj not in n_obj:
        return lambda g: float(abs(len(objects(g)) - w_obj))
    # No feature strictly separates the single win from the negatives -> cannot induce honestly.
    return None


# --- GAP-4891: richer goal-feature family (value / fill / spatial), beyond object/colour COUNTS ----

def _goal_feature_value(grid: np.ndarray, feat: str) -> float:
    """One scalar goal-relevant feature of a grid (GAP-4891). Beyond the count features
    (objects / unique-colours) that GAP-4890 showed cannot separate the win from non-wins on
    spatial/value/order goals (cd82 region-fill, sk48 reorder, sp80 placement, cn04 alignment),
    these capture FILL/VALUE/SPATIAL changes that leave the counts unchanged."""
    arr = np.asarray(grid)
    vals, counts = np.unique(arr, return_counts=True)
    if feat == "n_objects":
        return float(len(objects(arr)))
    if feat == "n_unique_colors":
        return float(len(vals))
    if feat == "nonbg_cells":  # cells not the dominant (background) colour -- FILL extent
        return float(arr.size - int(counts.max()))
    if feat == "max_color_count":  # extent of the most-common colour (fills shrink/grow it)
        return float(int(counts.max()))
    if feat == "nonbg_bbox_area":  # spatial extent (bounding box) of non-background -- placement/alignment
        bg = vals[int(counts.argmax())]
        ys, xs = np.where(arr != bg)
        return float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) if ys.size else 0.0
    if feat == "color_entropy":  # distributional spread of colours (reorder/redistribute goals)
        p = counts / counts.sum()
        return float(-(p * np.log(p + 1e-12)).sum())
    raise ValueError(f"unknown goal feature {feat}")


_RICHER_GOAL_FEATURES = (
    "n_objects",
    "n_unique_colors",
    "nonbg_cells",
    "max_color_count",
    "nonbg_bbox_area",
    "color_entropy",
)


def induce_goal_energy_richer(
    win_grid: Optional[np.ndarray],
    non_win_grids: List[np.ndarray],
    features: tuple[str, ...] = _RICHER_GOAL_FEATURES,
) -> Optional[Callable[[np.ndarray], float]]:
    """GAP-4891 goal-energy over a RICHER scalar-feature family (the within-game L2->L3 deepening
    unblock, after GAP-4890's single-positive operator cleared the win-exemplar floor).

    WHY: GAP-4890 proved that with the floor cleared, all 4 grid-based stalled games (cd82/sk48/sp80/
    cn04) still returned None because object/colour-COUNT features cannot SEPARATE the lone win from the
    non-wins -- their goals are spatial/value/order (region-fill, reorder, placement, alignment), which
    leave counts unchanged (results/arc_within_game_l3_self_induction_*_stage1.json). This adds FILL/
    VALUE/SPATIAL scalar features (non-background cell count, dominant-colour extent, non-bg bounding-box
    area, colour entropy) and applies the SAME strict-separation anti-mis-induction guard from GAP-4890:
    a feature fires ONLY when the lone win is strictly above/below EVERY negative on it. Returns
    energy(grid)->float>=0 (0 at the win's value), or None if no feature separates -- an honest
    'cannot induce'. Same contract as induce_goal_energy -> drops into graph_explore_solve_v2's
    goal_energy hook. The caller MUST still run the BFS-only ablation (energy only counts if it reaches
    the win faster than navigation-only)."""
    if win_grid is None or not non_win_grids:
        return None
    for feat in features:
        try:
            wv = _goal_feature_value(win_grid, feat)
            nv = [_goal_feature_value(g, feat) for g in non_win_grids]
        except Exception:
            continue
        if not nv:
            continue
        if wv < min(nv):  # win strictly LOWER -> goal = reduce this feature to the win's value
            return lambda g, _f=feat, _w=wv: float(max(0.0, _goal_feature_value(g, _f) - _w))
        if wv > max(nv):  # win strictly HIGHER -> goal = increase this feature to the win's value
            return lambda g, _f=feat, _w=wv: float(max(0.0, _w - _goal_feature_value(g, _f)))
    # No richer feature strictly separates the single win from the negatives -> cannot induce honestly.
    return None


# --- GAP-4891 RELATIONAL: within-frame target-match (canvas == target-shown-at-offset) ------------

def _overlap_pair(arr: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Aligned (a, b) over the in-bounds overlap such that a[i,j]=arr[y,x] and b[i,j]=arr[y+dy,x+dx]."""
    h, w = arr.shape
    a = arr[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
    b = arr[max(0, dy) : h - max(0, -dy), max(0, dx) : w - max(0, -dx)]
    return a, b


def induce_goal_energy_relational(
    win_grid: Optional[np.ndarray],
    non_win_grids: List[np.ndarray],
    *,
    min_mask: int = 6,
) -> Optional[Callable[[np.ndarray], float]]:
    """GAP-4891 RELATIONAL goal-energy (2026-06-27, after the count [GAP-4890] and richer-SCALAR
    [induce_goal_energy_richer] ladders were empirically refuted on cd82/sk48/sp80/cn04).

    WHY scalars failed: the negatives include NEAR-WIN frames (penultimate differs from the win by ~one
    cell), so no global statistic separates them. The goal is RELATIONAL: at the win a CANVAS region
    matches a TARGET shown elsewhere in the SAME frame; at every non-win it does not (yet). This models
    that as TRANSLATIONAL self-similarity with an induced mask:
      - Search offsets (dy,dx). For each, the win's NON-BACKGROUND self-match set M = {cells where
        win==win-shifted-by-(dy,dx) AND win!=background} is the induced canvas/target overlap (background
        excluded so trivial background self-matches don't dominate).
      - A hypothesis fires only if |M|>=min_mask AND EVERY negative has >=1 cell in M that does NOT match
        at the same offset (strict separation -- the canvas hasn't matched the target yet).
      - energy(g) = count of M-cells where g != g-shifted-by-(dy,dx): 0 at the win (matches over all M),
        >0 at near-win negatives (the wrong cell is in M). Pick the largest separating M.
    Generalises across levels: (dy,dx) is the constant screen layout offset; the target is re-read from
    each level's frame. Same contract -> drops into graph_explore_solve_v2's goal_energy hook. Returns
    None if no offset separates -- honest 'the relational structure is not a simple translate' (e.g. a
    learned mask/scale), which would itself be the next finding."""
    if win_grid is None or not non_win_grids:
        return None
    arr = np.asarray(win_grid)
    if arr.ndim != 2:
        return None
    h, w = arr.shape
    vals, counts = np.unique(arr, return_counts=True)
    bg = vals[int(counts.argmax())]
    max_off = max(1, min(h, w) // 2)
    # rank candidate offsets by the win's non-bg self-match-set size (cheap, win-only), largest first
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for dy in range(-max_off, max_off + 1):
        for dx in range(-max_off, max_off + 1):
            if dy == 0 and dx == 0:
                continue
            a, b = _overlap_pair(arr, dy, dx)
            if a.size < min_mask:
                continue
            mask = (a == b) & (a != bg)  # non-background canvas cells that match the target-at-offset
            ms = int(mask.sum())
            if ms >= min_mask:
                candidates.append((ms, dy, dx, mask))
    candidates.sort(key=lambda t: t[0], reverse=True)

    def _energy_at(g: np.ndarray, dy: int, dx: int, mask: np.ndarray) -> float:
        ga, gb = _overlap_pair(np.asarray(g), dy, dx)
        if ga.shape != mask.shape:
            return float(int(mask.sum()))  # shape mismatch (different-size frame) -> max violation
        return float((((ga != gb) & mask)).sum())

    for ms, dy, dx, mask in candidates:
        # strict separation: every negative must violate (>0) over the win's match-mask at this offset
        if all(_energy_at(g, dy, dx, mask) > 0.0 for g in non_win_grids):
            return lambda g, _dy=dy, _dx=dx, _m=mask: _energy_at(g, _dy, _dx, _m)
    return None
