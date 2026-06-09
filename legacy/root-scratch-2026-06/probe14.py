import sys
sys.path.insert(0, 'python')
from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction
from carnot.agentic.arc_agi3_world_model import grid_of
import numpy as np

def _background(grid): return 5

def _components(grid, colors):
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    comps = []
    target = np.isin(grid, list(colors))
    for i in range(h):
        for j in range(w):
            if target[i, j] and not seen[i, j]:
                col = int(grid[i, j]); stack = [(i, j)]; seen[i, j] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and target[ny, nx] and not seen[ny, nx] \
                                and grid[ny, nx] == col:
                            seen[ny, nx] = True; stack.append((ny, nx))
                ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
                comps.append({"color": col, "cells": cells, "area": len(cells),
                              "centroid": (sum(ys) // len(cells), sum(xs) // len(cells)),
                              "bbox": (max(ys) - min(ys) + 1, max(xs) - min(xs) + 1)})
    return comps

arc = Arcade(arc_api_key='', operation_mode=OperationMode.OFFLINE, environments_dir='environment_files')
env = arc.make('r11l-495a7899')

f = env.reset()
grid = grid_of(f)

pieces = [c for c in _components(grid, {0, 3, 4}) if c["area"] >= 2 and c["bbox"] != (63, 1)]
targets = [c for c in _components(grid, {6, 7, 8, 9, 11, 12, 13, 14, 15}) if c["area"] >= 2]

print("Pieces:", len(pieces), "Targets:", len(targets))
if len(targets) == 1:
    tx, ty = targets[0]["centroid"][1], targets[0]["centroid"][0]
    offsets = [(-6, 0), (6, 0)]
    for i, p in enumerate(pieces):
        px, py = p["centroid"][1], p["centroid"][0]
        f = env.step(GameAction.ACTION6, data={"x": px, "y": py})
        f = env.step(GameAction.ACTION6, data={"x": tx + offsets[i][0], "y": ty + offsets[i][1]})
        while env._game.yfbjozweime:
            f = env.step(GameAction.ACTION6, data={"x": -1, "y": -1})
    print("Level 0 completed?", getattr(f, 'levels_completed', 0))

