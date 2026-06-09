import json
import random
from arc_agi import Arcade
from arc_agi.base import OperationMode
from scripts.experiments.arc3_offline_eval import ENVDIR, _objects, _grid_dims

def main():
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    game = "lp85-305b61c3"
    env = arc.make(game)
    f = env.reset()
    h, w = _grid_dims(f)
    print(f"Grid dims: {h}x{w}")
    objs = _objects(f)
    print(f"Objects: {len(objs)}")

if __name__ == "__main__":
    main()
