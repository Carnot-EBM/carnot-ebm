module RKANSpec
import Mathlib.Data.Rat.Basic

def input_dim : Nat := 2

def edge_splines : List (Nat × Nat × List Rat) := [
  (0, 1, [(1/2 : Rat), (3/4 : Rat), (1 : Rat)])
]

def bias_splines : List (List Rat) := [
  [(0 : Rat), (1/2 : Rat), (1 : Rat)],
  [(-1 : Rat), (0 : Rat), (1 : Rat)]
]
