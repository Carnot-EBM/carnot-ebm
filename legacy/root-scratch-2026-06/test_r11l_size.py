import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "python"))
from arc_agi import Arcade
from arc_agi.base import OperationMode
arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
env = arc.make("r11l-495a7899")
env.reset()
for cpyyshywyc, data in env._game.kacotwgjcyq.items():
    pieces = data["lecfirgqbwunn"]
    for p in pieces:
        print(f"Piece size: {p.height}x{p.width}")
