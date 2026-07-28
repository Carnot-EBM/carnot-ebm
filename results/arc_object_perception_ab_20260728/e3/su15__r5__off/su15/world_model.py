import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 5
            if py > 0:
                grid[py-1, px] = 5
                if py > 1:
                    grid[py-2, px] = 5
                    if py > 2:
                        grid[py-3, px] = 5
                        if py > 3:
                            grid[py-4, px] = 5
                            if py > 4:
                                grid[py-5, px] = 5
                                if py > 5:
                                    grid[py-6, px] = 5
                                    if py > 6:
                                        grid[py-7, px] = 5
                                        if py > 7:
                                            grid[py-8, px] = 5
                                            if py > 8:
                                                grid[py-9, px] = 5
                                                if py > 9:
                                                    grid[py-10, px] = 5
                                                    if py > 10:
                                                        grid[py-11, px] = 5
                                                        if py > 11:
                                                            grid[py-12, px] = 5
                                                            if py > 12:
                                                                grid[py-13, px] = 5
                                                                if py > 13:
                                                                    grid[py-14, px] = 5
                                                                    if py > 14:
                                                                        grid[py-15, px] = 5
                                                                        if py > 15:
                                                                            grid[py-16, px] = 5
                                                                            if py > 16:
                                                                                grid[py-17, px] = 5
                                                                                if py > 17:
                                                                                    grid[py-18, px] = 5
                                                                                    if py > 18:
                                                                                        grid[py-19, px] = 5
                                                                                        if py > 19:
                                                                                            grid[py-20, px] = 5
                                                                                            if py > 20:
                                                                                                grid[py-21, px] = 5
                                                                                                if py > 21:
                                                                                                    grid[py-22, px] = 5
                                                                                                    if py > 22:
                                                                                                        grid[py-23, px] = 5
                                                                                                        if py > 23:
                                                                                                            grid[py-24, px] = 5
                                                                                                            if py > 24:
                                                                                                                grid[py-25, px] = 5
                                                                                                                if py > 25:
                                                                                                                    grid[py-26, px] = 5
                                                                                                                    if py > 26:
                                                                                                                        grid[py-27, px] = 5
                                                                                                                        if py > 27:
                                                                                                                            grid[py-28, px] = 5
                                                                                                                            if py > 28:
                                                                                                                                grid[py-29, px] = 5
                                                                                                                                if py > 29:
                                                                                                                                    grid[py-30, px] = 5
                                                                                                                                    if py > 30:
                                                                                                                                        grid[py-31, px] = 5
                                                                                                                                        if py > 31:
                                                                                                                                            grid[py-32, px] = 5
                                                                                                                                            if py > 32:
                                                                                                                                                grid[py-33, px] = 5
                                                                                                                                                if py > 33:
                                                                                                                                                    grid[py-34, px] = 5
                                                                                                                                                    if py > 34:
                                                                                                                                                        grid[py-35, px] = 5
                                                                                                                                                        if py > 35:
                                                                                                                                                            grid[py-36, px] = 5
                                                                                                                                                            if py > 36:
                                                                                                                                                                grid[py-37, px] = 5
                                                                                                                                                                if py > 37:
                                                                                                                                                                    grid[py-38, px] = 5
                                                                                                                                                                    if py > 38:
                                                                                                                                                                        grid[py-39, px] = 5
                                                                                                                                                                        if py > 39:
                                                                                                                                                                            grid[py-40, px] = 5
                                                                                                                                                                            if py > 40:
                                                                                                                                                                                grid[py-41, px] = 5
                                                                                                                                                                                if py > 41:
                                                                                                                                                                                    grid[py-42, px] = 5
                                                                                                                                                                                    if py > 42:
                                                                                                                                                                                        grid[py-43, px] = 5
                                                                                                                                                                                        if py > 43:
                                                                                                                                                                                            grid[py-44, px] = 5
                                                                                                                                                                                            if py > 44:
                                                                                                                                                                                                grid[py-45, px] = 5
                                                                                                                                                                                                if py > 45:
                                                                                                                                                                                                    grid[py-46, px] = 5
                                                                                                                                                                                                    if py > 46:
                                                                                                                                                                                                        grid[py-47, px] = 5
                                                                                                                                                                                                        if py > 47:
                                                                                                                                                                                                            grid[py-48, px] = 5
                                                                                                                                                                                                            if py > 48:
                                                                                                                                                                                                                grid[py-49, px] = 5
                                                                                                                                                                                                                if py > 49:
                                                                                                                                                                                                                    grid[py-50, px] = 5
                                                                                                                                                                                                                    if py > 50:
                                                                                                                                                                                                                        grid[py-51, px] = 5
                                                                                                                                                                                                                        if py > 51:
                                                                                                                                                                                                                            grid[py-52, px] = 5
                                                                                                                                                                                                                            if py > 52:
                                                                                                                                                                                                                                grid[py-53, px] = 5
                                                                                                                                                                                                                                if py > 53:
                                                                                                                                                                                                                                    grid[py-54, px] = 5
                                                                                                                                                                                                                                    if py > 54:
                                                                                                                                                                                                                                        grid[py-55, px] = 5
                                                                                                                                                                                                                                        if py > 55:
                                                                                                                                                                                                                                            grid[py-56, px] = 5
                                                                                                                                                                                                                                            if py > 56:
                                                                                                                                                                                                                                                grid[py-57, px] = 5
                                                                                                                                                                                                                                                if py > 57:
                                                                                                                                                                                                                                                    grid[py-58, px] = 5
                                                                                                                                                                                                                                                    if py > 58:
                                                                                                                                                                                                                                                        grid[py-59, px] = 5
                                                                                                                                                                                                                                                        if py > 59:
                                                                                                                                                                                                                                                            grid[py-60, px] = 5
                                                                                                                                                                                                                                                            if py > 60:
                                                                                                                                                                                                                                                                grid[py-61, px] = 5
                                                                                                                                                                                                                                                                if py > 61:
                                                                                                                                                                                                                                                                    grid[py-62, px] = 5
                                                                                                                                                                                                                                                                    if py > 62:
                                                                                                                                                                                                                                                                        grid[py-63, px] = 5
    return grid

def is_level_complete(grid):
    return False

import numpy as np

def is_level_complete(grid):
    if grid.shape != (15, 15):
        return False
    if grid.dtype != np.int8:
        return False
    if not np.all(grid == 0):
        return False
    return True
