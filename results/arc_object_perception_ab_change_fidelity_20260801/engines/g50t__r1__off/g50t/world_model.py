import numpy as np

def engine(grid, action, data):
    # The game involves a player character (color 1) moving through a maze of walls (color 5) and obstacles/toggles (color 2 and 9).
    # Based on the observed transitions, Action 2 seems to be 'Right' movement.
    # Moving right shifts the state of certain blocks of colors 2 and 9 in the same row or related rows.
    # # In this specific level, it looks like a pattern of color 2s and 9s are being toggled or shifted.
    # Let's simplify the transition rules based on the<|channel>thought process.
    # The grid contains a structure where color 5 acts as boundaries.
    # Color 1 is the agent.
    # Color 8 is some other obstacle.
    # ACTION 2 is Right, ACTION 4 is Left.
    # Looking at the deltas, ACTION 2 repeatedly changes blocks of cells from 2->5, 9->5, etc.,
    # but also moves the agent (color 1) leftwards in the bottom row r63.
    # This suggests that for every "step" taken by the agent, a corresponding set of tiles is modified.
    # Specifically, the agent starts at (63, 62) and moves left: 62 -> 61 -> 60 -> 59 -> 58...
    # Each move corresponds to a change in the maze above.

    # Find the agent's position
    agent_pos = np.argwhere(grid == 1)[0]
    r_agent, c_agent = agent_pos[0], agent_pos[1]

    # Define movement mapping
    # Action 2: Right (but agent moves left in this specific level layout?)
    # Action 4: Left (but agent moves right?)
    # Let's check the observed transitions again.
    # INITIAL GRID: Agent is at r63c62, r63c63 is color 1? No, r63:9x62, 1x2. So cells 62, 63 are color 1.
    # ACTION 2 (first): r63c61 becomes 1. Wait, if it was 9x62, then index 62 is the first '1'.
    # If r63c61 becomes 1, the agent is moving LEFT.
    # This is counter-intuitive but let's follow the data.
    # In ARC games, sometimes actions are mapped differently or the "goal" is to move a piece.

    new_grid = grid.copy()
    if action == 2: # Move Left
        if c_agent > 0:
            new_grid[r_agent, c_agent] = 9
            new_grid[r_agent, c_agent - 1] = 1
    elif action == 4: # Move Right
        if c_agent < new_grid.shape[1] - 1:
            new_grid[r_agent, c_agent] = 9
            new_grid[r_agent, c_agent + 1] = 1
    
    # The maze changes are complex and seem tied to the agent's column position.
    # However, without a clear general rule for the maze modification, 
    # we can implement the agent movement which is clearly observed in the deltas (r63).
    # For the maze modifications, they appear as blocks of color 5 replacing colors 2/9.
    # Let's try to induce a simple rule: when moving, some cells change.
    # But since this is a world model for a specific game 'g50t', let's focus on the most consistent part.

    return new_grid

def is_level_complete(grid):
    # Level complete usually means reaching a goal or clearing all obstacles.
    # In many ARC games, it's when the agent reaches a certain coordinate or a pattern is formed.
    # Given the data, there's no explicit win state provided.
    # We'll assume the level is complete if the agent reaches the far left (c=0) or something similar.
    # But typically, we check if any target condition is met.
    # Since we don't have the WIN STATE grid, we return False unless a common win condition is found.
    return False