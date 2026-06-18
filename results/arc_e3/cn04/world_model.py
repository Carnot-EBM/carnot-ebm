import numpy as np

def engine(grid, action, data):
    """
    Simulates the game logic for ARC-AGI-3 'cn04'.
    
    The game involves a grid with a 'wall' structure (color 4) and 'blocks' (color 10).
    The player manipulates the blocks to fill a specific target area defined by the wall.
    
    Key Observations:
    1. The grid contains a static wall structure (color 4) that defines a target region.
    2. There are blocks (color 10) that can be moved.
    3. Action 1 (likely 'Move Down' or 'Shift') causes blocks to fall or shift.
    4. Action 2 (likely 'Move Up' or 'Shift') causes blocks to rise or shift.
    5. Action 6 is a click action with pixel coordinates.
    6. The goal is to fill the target region (defined by the wall) with blocks (color 10).
    
    Logic:
    - The wall (color 4) is static and defines the boundaries of the target area.
    - Blocks (color 10) move according to the action.
    - Action 1: Blocks move down. If a block is above an empty space (0), it falls.
    - Action 2: Blocks move up. If a block is below an empty space (0), it rises.
    - Action 6: Clicking on a block or area might select or move it, but based on the data, 
      it seems to be used to place blocks or interact with the wall.
    
    However, looking at the transitions:
    - ACTION1 causes blocks to move down and fill the target area.
    - ACTION2 causes blocks to move up and clear the target area.
    - The target area is defined by the wall (color 4).
    
    Refined Logic:
    - The wall (color 4) forms a container.
    - Blocks (color 10) are gravity-driven within this container.
    - Action 1: Apply gravity (blocks fall down).
    - Action 2: Apply anti-gravity (blocks rise up).
    - Action 6: Click to place a block or toggle a cell.
    
    Let's implement a simple gravity model for Action 1 and 2.
    """
    grid = grid.copy()
    H, W = grid.shape
    
    if action == 1:
        # Move blocks down (gravity)
        # For each column, move all 10s to the bottom of the container defined by the wall.
        # The wall (4) acts as a barrier.
        for c in range(W):
            # Extract the column
            col = grid[:, c]
            # Identify the container boundaries (top and bottom walls)
            # Find the indices of the wall (4) in the column
            wall_indices = np.where(col == 4)[0]
            if len(wall_indices) == 0:
                continue
            
            # The container is between the top-most and bottom-most wall
            top_wall = wall_indices[0]
            bottom_wall = wall_indices[-1]
            
            # Extract the region inside the container
            # The region is from top_wall + 1 to bottom_wall - 1
            if top_wall + 1 > bottom_wall - 1:
                continue
                
            region = col[top_wall+1 : bottom_wall]
            # Count the number of blocks (10) in the region
            num_blocks = np.sum(region == 10)
            # Create a new region with blocks at the bottom
            new_region = np.zeros_like(region)
            new_region[-num_blocks:] = 10
            # Update the grid
            grid[top_wall+1 : bottom_wall, c] = new_region
            
    elif action == 2:
        # Move blocks up (anti-gravity)
        for c in range(W):
            col = grid[:, c]
            wall_indices = np.where(col == 4)[0]
            if len(wall_indices) == 0:
                continue
            
            top_wall = wall_indices[0]
            bottom_wall = wall_indices[-1]
            
            if top_wall + 1 > bottom_wall - 1:
                continue
                
            region = col[top_wall+1 : bottom_wall]
            num_blocks = np.sum(region == 10)
            # Create a new region with blocks at the top
            new_region = np.zeros_like(region)
            new_region[:num_blocks] = 10
            # Update the grid
            grid[top_wall+1 : bottom_wall, c] = new_region
            
    elif action == 6:
        # Click action
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            # Convert pixel to logical coordinates
            # Assuming pixel = logical * 1, so direct mapping
            r, c = py, px
            if 0 <= r < H and 0 <= c < W:
                # Toggle the cell or place a block
                # Based on the transitions, clicking seems to place a block (10) if empty,
                # or remove it if it's a block.
                if grid[r, c] == 0:
                    grid[r, c] = 10
                elif grid[r, c] == 10:
                    grid[r, c] = 0
                # Note: The transitions show that clicking can also affect the wall (4),
                # but in the provided data, the wall (4) is only removed, not added.
                # However, the wall seems to be static in the initial grid and transitions.
                # Let's assume the wall is static and clicking only affects 0 and 10.
                
    return grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    The level is complete if the target area (defined by the wall) is filled with blocks (10).
    """
    H, W = grid.shape
    # Find the wall (4) to determine the target area
    wall_indices = np.argwhere(grid == 4)
    if len(wall_indices) == 0:
        return False
    
    # Determine the bounding box of the wall
    min_r, min_c = np.min(wall_indices, axis=0)
    max_r, max_c = np.max(wall_indices, axis=0)
    
    # The target area is inside the wall
    # Check if the area inside the wall is filled with blocks (10)
    # The area is from min_r+1 to max_r-1 and min_c+1 to max_c-1
    target_area = grid[min_r+1 : max_r, min_c+1 : max_c]
    
    # Check if all cells in the target area are 10
    if np.all(target_area == 10):
        return True
    else:
        return False