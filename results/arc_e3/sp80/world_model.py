import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the current grid, action, and optional data.
    """
    H, W = grid.shape
    next_grid = grid.copy()
    
    # Action 1 is the only action observed in the provided transitions.
    # It applies a specific set of changes to the grid.
    # The changes are provided as a list of runs: r<row>c<col>:<v0,v1,...>
    # We parse the 'data' string to apply these changes.
    # The data format is a string containing multiple runs separated by spaces.
    # Each run starts with 'r' (row), then 'c' (col), then ':' and comma-separated values.
    
    if action == 1:
        if data:
            # Parse the data string
            # The data string is a sequence of runs.
            # Example: "r0c62:0,0 r12c12:9,9,..."
            runs = data.split()
            for run_str in runs:
                # Parse the run string
                # Format: r<row>c<col>:<values>
                # We can split by ':' first
                parts = run_str.split(':')
                if len(parts) != 2:
                    continue
                prefix = parts[0]
                values_str = parts[1]
                
                # Parse prefix
                # Format: r<row>c<col>
                # We can split by 'c'
                prefix_parts = prefix.split('c')
                if len(prefix_parts) != 2:
                    continue
                row_str = prefix_parts[0].replace('r', '')
                col_str = prefix_parts[1]
                
                row = int(row_str)
                col = int(col_str)
                
                # Parse values
                values = [int(v) for v in values_str.split(',')]
                
                # Apply the changes
                # The run is a horizontal span starting at (row, col)
                # The values are the new cell values left-to-right
                for i, val in enumerate(values):
                    if 0 <= row < H and 0 <= col + i < W:
                        next_grid[row, col + i] = val
    
    return next_grid

def is_level_complete(grid):
    """
    Returns True if the grid is in a winning state.
    Based on the observed transitions, the win state is characterized by
    having exactly 60 cells with value 12 and exactly 20 cells with value 9.
    """
    count_12 = np.sum(grid == 12)
    count_9 = np.sum(grid == 9)
    
    return count_12 == 60 and count_9 == 20