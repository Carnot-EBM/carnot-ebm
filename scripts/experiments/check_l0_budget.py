import sys
from pathlib import Path

# Resolved from this file rather than hardcoded so a fresh clone or a
# worktree writes into ITS OWN tree. Inlined (not carnot.paths.repo_root)
# because the next line is what makes ``carnot`` importable -- importing
# the resolver here would be circular. Same rule, same answer.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arc_agi import Arcade
from arc_agi.base import OperationMode


def run():
    arc = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    for gid in ["sc25-635fd71a", "tn36-ef4dde99", "su15-1944f8ab", "dc22-fdcac232"]:
        try:
            env = arc.make(gid)
            f = env.reset()
            import numpy as np

            arr = np.array(f.frame)
            if arr.ndim == 3:
                arr = arr[-1]
            from carnot.agentic.arc_agi3_world_model import objects

            objs = objects(arr)
            print(
                f"{gid}: L0 budget={f.available_actions}, budget_count={len(f.available_actions) if f.available_actions else 'N/A'}, objects={len(objs)}"
            )
        except Exception as e:
            print(f"{gid} failed: {e}")


if __name__ == "__main__":
    run()
