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
