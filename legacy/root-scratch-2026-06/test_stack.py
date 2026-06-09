import sys, numpy as np
from pathlib import Path
REPO = Path().resolve()
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction
import random

arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
f = env.reset()
f = env.step(GameAction.ACTION6, data={"x": 36, "y": 7})
f = env.step(GameAction.ACTION6, data={"x": 21, "y": 39})
lc = getattr(f, "levels_completed", 0)
frames = f.frame if hasattr(f, "frame") else [f]
print(f"Level up! LC={lc}. Frame stack len={len(frames) if isinstance(frames, list) else np.array(frames).shape[0]}")
if isinstance(frames, list) and len(frames) >= 2:
    print("Frame -2 colors:", np.unique(frames[-2], return_counts=True))
    print("Frame -1 colors:", np.unique(frames[-1], return_counts=True))
elif isinstance(frames, np.ndarray) and frames.ndim == 3 and frames.shape[0] >= 2:
    print("Frame -2 colors:", np.unique(frames[-2], return_counts=True))
    print("Frame -1 colors:", np.unique(frames[-1], return_counts=True))
