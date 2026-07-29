import sys
from pathlib import Path
import numpy as np

# Resolved from this file rather than hardcoded so a fresh clone or a
# worktree writes into ITS OWN tree. Inlined (not carnot.paths.repo_root)
# because the next line is what makes ``carnot`` importable -- importing
# the resolver here would be circular. Same rule, same answer.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
from carnot.agentic.arc_agi3_world_model import objects, grid_of

env = Arcade(
    arc_api_key="",
    operation_mode=OperationMode.OFFLINE,
    environments_dir=str(REPO / "environment_files"),
).make("sc25-635fd71a")
f = env.reset()
g = grid_of(f)

for cy, cx in objects(g):
    mask = g == g[cy, cx]
    ys, xs = np.where(mask)
    print(
        f"Object at {cy}, {cx} (color {g[cy, cx]}): y=[{ys.min()}..{ys.max()}], x=[{xs.min()}..{xs.max()}]"
    )
