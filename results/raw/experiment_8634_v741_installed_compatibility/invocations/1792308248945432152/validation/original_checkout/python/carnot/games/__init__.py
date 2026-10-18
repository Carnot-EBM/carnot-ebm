"""Game cartridges for Carnot WOPR-style energy demonstrations."""

from carnot.games.connect_four import ConnectFourIsingCartridge
from carnot.games.futoshiki import FutoshikiIsingEBM, FutoshikiPuzzle, FutoshikiSolver
from carnot.games.hex import GibbsEnergyPlayer, GreedyEnergyPlayer, HexBoard, HexGame, RandomPlayer
from carnot.games.nonogram import NonogramIsingEBM, NonogramPuzzle, NonogramSolver

__all__ = [
    "ConnectFourIsingCartridge",
    "FutoshikiIsingEBM",
    "FutoshikiPuzzle",
    "FutoshikiSolver",
    "GibbsEnergyPlayer",
    "GreedyEnergyPlayer",
    "HexBoard",
    "HexGame",
    "NonogramIsingEBM",
    "NonogramPuzzle",
    "NonogramSolver",
    "RandomPlayer",
]
