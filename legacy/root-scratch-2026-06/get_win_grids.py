import sys
import numpy as np
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_agi3_world_model import grid_of, objects
from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction, GameState
import arc3_graph_explore as gx

arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()
by_id = {a.value: a for a in GameAction}
rng = random.Random(42)

for ep in range(10):
    f = env.reset()
    for step in range(50):
        av = getattr(f, "available_actions", [])
        if not av: break
        grid = grid_of(f)
        cands = gx._candidate_akeys(grid, av)
        if not cands: break
        akey = rng.choice(cands)
        a_int = akey[0]
        data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
        f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
        
        lc = int(getattr(f, 'levels_completed', 0) or 0)
        st = getattr(f, 'state', None)
        
        if lc > 0:
            print(f"Ep {ep} Step {step} LEVEL UP! LC={lc}")
            frames = f.frame if hasattr(f, "frame") else [f]
            if isinstance(frames, list) and len(frames) >= 2:
                print("Frame -2 objects:", len(objects(np.array(frames[-2]))))
                print("Frame -1 objects:", len(objects(np.array(frames[-1]))))
                
                print("Frame -2 unique colors:", len(np.unique(frames[-2])))
                print("Frame -1 unique colors:", len(np.unique(frames[-1])))
            elif isinstance(frames, np.ndarray) and frames.ndim == 3 and frames.shape[0] >= 2:
                print("Frame -2 objects:", len(objects(frames[-2])))
                print("Frame -1 objects:", len(objects(frames[-1])))
            break
        if st in (GameState.WIN, GameState.GAME_OVER):
            break
