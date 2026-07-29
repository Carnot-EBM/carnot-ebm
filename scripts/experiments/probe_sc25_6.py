import sys
from pathlib import Path
import copy
from collections import deque
import numpy as np

# Resolved from this file rather than hardcoded so a fresh clone or a
# worktree writes into ITS OWN tree. Inlined (not carnot.paths.repo_root)
# because the next line is what makes ``carnot`` importable -- importing
# the resolver here would be circular. Same rule, same answer.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arc_agi import Arcade
from arc_agi.base import OperationMode
from carnot.agentic.arc_agi3_world_model import objects, compute_grid_delta, grid_of
from arcengine.enums import GameAction


def attempt_sc25():
    env = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    ).make("sc25-635fd71a")
    f = env.reset()
    original_game = copy.deepcopy(env._game)

    click_coords = [
        (25, 50),
        (30, 50),
        (35, 50),
        (25, 55),
        (30, 55),
        (35, 55),
        (25, 60),
        (30, 60),
        (35, 60),
    ]

    for combo in range(512):
        env._game = copy.deepcopy(original_game)
        last_f = f
        # Apply clicks
        for i in range(9):
            if (combo >> i) & 1:
                last_f = env.step(
                    GameAction.ACTION6, data={"x": click_coords[i][0], "y": click_coords[i][1]}
                )

        # Now BFS walk to exit
        q = deque([(copy.deepcopy(env._game), last_f, 0)])
        seen = {grid_of(last_f).tobytes()}

        while q:
            curr_game, curr_f, depth = q.popleft()
            if curr_f.levels_completed and curr_f.levels_completed > 0:
                print(f"WON with combo {combo}!")
                return combo

            if depth > 15:
                continue

            for action in [
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
            ]:
                env._game = copy.deepcopy(curr_game)
                new_f = env.step(action)
                if new_f.levels_completed and new_f.levels_completed > 0:
                    print(f"WON with combo {combo} at depth {depth + 1}!")
                    return combo

                g2 = grid_of(new_f)
                state_bytes = g2.tobytes()
                if state_bytes not in seen:
                    seen.add(state_bytes)
                    q.append((copy.deepcopy(env._game), new_f, depth + 1))

        if combo % 64 == 0:
            print(f"Tested {combo}/512...")

    print("Did not win.")


if __name__ == "__main__":
    attempt_sc25()
