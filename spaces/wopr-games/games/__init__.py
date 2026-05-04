"""WOPR Games cartridges.

Each cartridge is a self-contained energy-based reasoning demo
wrapped in the WOPR aesthetic. See `_base.WOPRGame` for the
interface every cartridge must satisfy.
"""

from games._base import WOPRGame
from games.hashi import HashiGame
from games.lights_out import LightsOutGame
from games.masyu import MasyuGame
from games.nqueens import NQueensGame
from games.slitherlink import SlitherlinkGame
from games.sudoku import SudokuGame
from games.thermonuclear_war import ThermonuclearWarGame
from games.tictactoe import TicTacToeGame

ALL_GAMES: list[WOPRGame] = [
    SudokuGame(),
    TicTacToeGame(),
    LightsOutGame(),
    ThermonuclearWarGame(),
    NQueensGame(),
    HashiGame(),
    SlitherlinkGame(),
    MasyuGame(),
]

__all__ = [
    "WOPRGame",
    "SudokuGame",
    "TicTacToeGame",
    "LightsOutGame",
    "ThermonuclearWarGame",
    "NQueensGame",
    "HashiGame",
    "SlitherlinkGame",
    "MasyuGame",
    "ALL_GAMES",
]
