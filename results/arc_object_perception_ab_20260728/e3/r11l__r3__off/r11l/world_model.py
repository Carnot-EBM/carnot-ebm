import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        new_grid[py, px] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    for i in range(h):
        row_str = ""
        for j in range(w):
            val = grid[i, j]
            if val == 0:
                row_str += "0x1,"
            elif val == 1:
                row_str += "1x1,"
            elif val == 2:
                row_str += "2x1,"
            elif val == 5:
                row_str += "5x1,"
            elif val == 6:
                row_str += "6x1,"
            elif val == 15:
                row_str += "15x1,"
        if row_str != "0x1,2x7,5x2,2x11,5x13,2x7,5x8,2x15,0x1,2x7,5x2,2x11,5x31,2x12,0x1,2x7,5x46,2x10,0x1,2x7,5x49,2x7,0x1,2x6,5x10,0x1,5x40,2x6,0x1,2x6,5x9,0x3,5x39,2x6,0x1,2x5,5x9,0x2,12x1,0x2,5x44,0x1,2x1,5x14,0x3,5x30,3x1,5x14,0x1,2x1,5x15,0x1,5x1,1x1,5x28,3x3,5x13,0x1,2x1,5x18,1x2,5x23,1x2,3x2,12x1,3x2,5x12,0x1,2x1,5x20,1x1,12x3,5x11,1x8,5x3,3x3,5x13,0x1,2x1,5x20,12x5,5x2,1x8,5x12,3x1,5x14,0x1,2x1,5x20,12x2,6x1,12x2,1x2,5x35,0x1,2x1,5x20,12x5,5x37,0x1,5x19,1x2,5x1,12x3,5x38,0x1,5x17,1x2,5x36,15x1,5x1,15x1,5x5,0x1,5x16,1x1,5x37,15x1,5x3,15x1,5x4,0x1,5x14,1x2,5x37,15x1,5x5,15x1,5x3,0x1,5x12,1x2,5x49,0x1,5x7,3x1,5x2,1x2,5x41,15x1,5x5,15x1,5x3,0x1,5x6,3x3,1x1,5x44,15x1,5x3,15x1,5x4,0x1,5x5,3x2,12x1,3x2,5x45,15x1,5x1,15x1,5x5,0x1,5x6,3x3,5x15,10x10,5x29,0x1,5x7,3x1,5x15,10x14,5x26,0x1,5x20,10x17,5x26,0x1,5x18,10x20,5x25,0x1,5x16,10x19,5x28,0x1,5x16,10x14,5x33,0x1,5x16,10x14,5x33,0x1,5x16,10x14,5x24,10x4,5x5,0x1,5x15,10x15,5x23,10x5,5x5,0x1,5x13,10x17,5x22,10x11,0x1,10x1,5x12,10x10,5x28,10x12,0x1,10x1,5x43,3x1,5x6,10x12,0x1,10x2,5x41,3x3,5x5,10x12,0x1,10x4,5x38,3x2,15x1,3x2,5x6,10x10,0x1,10x5,5x38,3x3,5x8,10x9,0x1,10x6,5x38,3x1,1x1,5x8,10x9,0x1,10x6,5x12,10x5,5x23,1x1,5x9,10x7,0x1,10x6,5x9,10x10,5x22,15x3,5x7,10x6,0x1,10x6,5x9,10x12,5x19,15x5,5x6,10x6,0x1,10x6,5x9,10x13,5x18,15x2,6x1,15x2,5x6,10x6,0x1,10x28,5x18,15x5,5x7,10x5,0x1,10x28,5x19,15x3,5x8,10x5,0x1,2x3,10x25,5x22,1x1,5x7,10x5,0x1,2x3,10x19,5x29,1x1,5x6,10x5,0x1,2x3,10x19,5x30,1x1,3x1,5x4,10x5,0x1,2x3,10x19,5x30,3x3,5x4,10x4,0x1,2x3,10x15,5x20,12x1,5x1,12x1,5x10,3x2,15x1,3x2,5x3,2x4,0x1,2x3,10x15,5x19,12x1,5x3,12x1,5x10,3x3,5x4,2x4,0x1,2x6,10x12,5x18,12x1,5x5,12x1,5x10,3x1,5x5,2x4,0x1,2x6,10x3,2x7,10x2,5x41,2x4,0x1,2x6,10x2,2x8,10x2,5x18,12x1,5x5,12x1,5x16,2x4,0x1,2x6,10x2,2x8,10x2,5x19,12x1,5x3,12x1,5x17,2x4,0x1,2x6,10x2,2x8,10x2,5x20,12x1,5x1,12x1,5x18,2x4,0x1,2x16,5x42,2x5,0x1,2x21,5x32,2x10,0x1,2x21,5x32,2x10,0x1,2x21,5x32,2x10,0x1,2x21,5x32,2x10,0x1,2x21,5x29,2x13,0x1,2x21,5x25,2x17,0x1,2x21,5x25,2x17,0x1,2x15,5x31,2x17":
            return False
    return True