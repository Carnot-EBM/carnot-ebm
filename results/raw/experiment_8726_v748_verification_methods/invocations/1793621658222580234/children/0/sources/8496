def goal_progress(grid):
    # lp85, HINDSIGHT goal induced by Claude from the OBSERVED (L1-start, L1-win) PAIR (2026-06-22).
    # The top bar (rows 0-3, mostly row 1) is a sequence of 5555 "slot" blocks; the L1 win turned the
    # FIRST 5555 slot -> 4444 (a click matched a middle block to fill the next slot). Level-invariant
    # goal: FILL the top-bar slots. Progress = count of color-5 cells in the top-bar region; drops by 4
    # each time a slot fills, 0 when all filled. The A* descends this -> finds the slot-filling click ->
    # levels up; grounded on an OBSERVED win, not static structure (the chicken-and-egg breaker).
    g = np.asarray(grid)
    return float((g[0:4] == 5).sum())
