def engine(grid, action, data):
    """
    Simulates the agent's movement and interaction with the grid.
    The grid is a 2D array where:
    - 0: Empty space
    - 63: Wall
    - 6: Agent
    - 9: Target
    - 4: Special target (possibly a different type)
    - 15-23: Other entities or obstacles

    Actions:
    - 0: Up
    - 1: Down
    - 2: Left
    - 3: Right
    - 4: Collect (or similar interaction)

    The function updates the grid based on the action taken by the agent.
    """
    # Find the agent's current position
    agent_pos = None
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 6:
                agent_pos = (i, j)
                break
        if agent_pos:
            break

    if not agent_pos:
        return grid

    agent_row, agent_col = agent_pos

    # Define movement directions
    directions = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
        4: (0, 0)    # Collect (no movement)
    }

    dr, dc = directions.get(action, (0, 0))
    new_row, new_col = agent_row + dr, agent_col + dc

    # Check if the new position is within bounds
    if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
        # Check if the new position is a wall
        if grid[new_row][new_col] == 63:
            return grid  # Cannot move into a wall

        # Check if the new position is a target (9 or 4)
        if grid[new_row][new_col] in [9, 4]:
            # Collect the target
            grid[agent_row][agent_col] = 0  # Agent moves to the target
            grid[new_row][new_col] = 6     # Target becomes agent
            return grid

        # Move to the new position
        grid[agent_row][agent_col] = 0
        grid[new_row][new_col] = 6
        return grid

    # If the new position is out of bounds, stay in place
    return grid


def is_level_complete(grid):
    """
    Checks if the level is complete.
    The level is complete if there are no targets (9 or 4) left on the grid.
    """
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] in [9, 4]:
                return False
    return True