def engine(grid, action, data):
    # grid is 64x64 (64 rows, 64 cols)
    # action: 0=Up, 1=Down, 2=Left, 3=Right, 4=RotateCW, 5=RotateCCW, 6=Jump, 7=Shoot
    # data: {"player": (r, c), "entities": [(r, c, type)], "level": int}
    # types: 0=Empty, 1=Wall, 2=Enemy, 3=PowerUp, 4=Goal, 5=Bullet
    
    # Helper to check bounds
    def in_bounds(r, c):
        return 0 <= r < 64 and 0 <= c < 64
    
    # Helper to get entity type at position
    def get_entity_type(r, c):
        for er, ec, etype in data["entities"]:
            if er == r and ec == c:
                return etype
        return 0  # Empty
    
    # Helper to move player
    def move_player(r, c, action):
        if action == 0:  # Up
            return r - 1, c
        elif action == 1:  # Down
            return r + 1, c
        elif action == 2:  # Left
            return r, c - 1
        elif action == 3:  # Right
            return r, c + 1
        elif action == 6:  # Jump
            return r, c
        elif action == 7:  # Shoot
            return r, c
        return r, c
    
    # Helper to rotate player
    def rotate_player(r, c, action):
        return r, c
    
    # Helper to handle shooting
    def shoot(r, c, action):
        # Create bullet in direction of movement
        bullet_r, bullet_c = r, c
        if action == 0:  # Up
            bullet_r -= 1
        elif action == 1:  # Down
            bullet_r += 1
        elif action == 2:  # Left
            bullet_c -= 1
        elif action == 3:  # Right
            bullet_c += 1
        elif action == 7:  # Shoot
            bullet_r, bullet_c = r, c
        return bullet_r, bullet_c
    
    # Helper to handle jumping
    def jump(r, c, action):
        # Jump over obstacles
        return r, c
    
    # Get current player position
    player_r, player_c = data["player"]
    
    # Get action
    action = data.get("action", 0)
    
    # Move player
    new_r, new_c = move_player(player_r, player_c, action)
    
    # Check if new position is valid
    if not in_bounds(new_r, new_c):
        return grid, data
    
    # Check if new position is a wall
    if get_entity_type(new_r, new_c) == 1:
        return grid, data
    
    # Check if new position is an enemy
    if get_entity_type(new_r, new_c) == 2:
        # Kill enemy
        data["entities"] = [(er, ec, etype) for er, ec, etype in data["entities"] if not (er == new_r and ec == new_c)]
        return grid, data
    
    # Check if new position is a power-up
    if get_entity_type(new_r, new_c) == 3:
        # Collect power-up
        data["entities"] = [(er, ec, etype) for er, ec, etype in data["entities"] if not (er == new_r and ec == new_c)]
        return grid, data
    
    # Check if new position is a goal
    if get_entity_type(new_r, new_c) == 4:
        # Win level
        return grid, data
    
    # Update grid with new player position
    grid[new_r][new_c] = 1  # Player is now at new position
    
    # Handle shooting
    if action == 7:  # Shoot
        # Create bullet
        bullet_r, bullet_c = shoot(player_r, player_c, action)
        if in_bounds(bullet_r, bullet_c):
            # Check if bullet hits enemy
            if get_entity_type(bullet_r, bullet_c) == 2:
                # Kill enemy
                data["entities"] = [(er, ec, etype) for er, ec, etype in data["entities"] if not (er == bullet_r and ec == bullet_c)]
    
    # Handle jumping
    if action == 6:  # Jump
        # Jump over obstacles
        pass
    
    # Handle rotation
    if action == 4:  # RotateCW
        # Rotate player
        pass
    
    # Handle rotation
    if action == 5:  # RotateCCW
        # Rotate player
        pass
    
    return grid, data

def is_level_complete(grid):
    # Check if all enemies are dead
    for er, ec, etype in data["entities"]:
        if etype == 2:
            return False
    return True