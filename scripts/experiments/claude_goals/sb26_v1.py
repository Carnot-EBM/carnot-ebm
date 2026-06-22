def goal_progress(grid):
    # sb26, first-contact induction by Claude (2026-06-22). Evidence: ACTION5 paints the full-width
    # horizontal BAR (row ~53, color 2) to color 3, one cell at a time (observed col 63->62->61).
    # Most-supported hypothesis: the win is FILLING that progress bar (all its 2s -> 3). Progress =
    # count of color-2 cells in any row that is MOSTLY color 2 (the bar identified by structure, not a
    # hardcoded row, so it survives layout shifts); drops by 1 per ACTION5, reaches 0 when filled. An
    # A* descending this commits to the ~64-deep sequential ACTION5 line that breadth-first BFS misses.
    g = np.asarray(grid)
    twos_per_row = (g == 2).sum(axis=1)
    bar_mask = twos_per_row > (g.shape[1] // 2)
    return float(twos_per_row[bar_mask].sum())
