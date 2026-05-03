# WOPR Games Capability Specification

**Capability:** wopr-games
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines WOPR gallery cartridges that expose classic game and puzzle rules as
energy-based constraint systems.

## Requirements

### REQ-CONNECT4-001: Connect Four Occupancy Energy

The Connect Four cartridge MUST expose a 6x7 board as 42 binary Ising spins,
where +1 means occupied and -1 means empty. Its energy MUST be zero only when
all occupied cells satisfy gravity and the occupied-cell count equals the
configured initial piece count.

**Acceptance criteria:**
- `n_spins` is 42 for the 6x7 board.
- Empty and gravity-valid boards with the configured piece count have energy 0.
- A floating occupied cell above an empty lower cell has positive energy.
- A board with the wrong occupied-cell count has positive energy.

### REQ-CONNECT4-002: Connect Four Rule Helpers

The Connect Four cartridge MUST provide board validation and winner detection
helpers for WOPR gallery scenarios. Winner detection MUST report horizontal,
vertical, and diagonal four-in-a-row outcomes for RED and YELLOW, DRAW for a
full board with no winner, and ONGOING otherwise.

**Acceptance criteria:**
- `is_valid(board)` checks gravity plus configured piece count.
- `check_winner(board)` returns `"RED"`, `"YELLOW"`, `"DRAW"`, or `"ONGOING"`.

### REQ-CONNECT4-003: Connect Four Sampler

The Connect Four cartridge sampler MUST return a gravity-valid board with the
configured piece count and zero energy. For a gravity-violated input state, the
sampler MUST be able to repair the occupancy pattern to a zero-energy board
with the same number of occupied cells.

**Acceptance criteria:**
- `sample(n_steps=1000, beta=2.0)` returns a `(6, 7)` board.
- The sampled board is valid under `is_valid`.
- The sampled board has zero energy.

### REQ-HEX-001: Hex Board and Move Rules

The Hex cartridge MUST expose an `n x n` board where 0 means empty, 1 means
Black, and 2 means White. `HexGame.reset()` MUST return an empty board,
`legal_actions(board)` MUST return every empty `(row, col)` action in row-major
order, and `step(board, action, player)` MUST return a copied board with the
move applied plus terminal status.

**Acceptance criteria:**
- `HexBoard(n)` creates an `n x n` zero board and rejects non-positive sizes.
- `legal_actions(board)` returns exactly the empty cells as `(row, col)` tuples.
- `step(board, action, player)` rejects occupied cells and invalid players.

### REQ-HEX-002: Hex Connectivity Winner Detection

The Hex cartridge MUST detect Black wins as top-to-bottom connectivity and
White wins as left-to-right connectivity. Winner detection MUST use union-find
style connectivity over the six hex-neighbor directions and MUST return
`None`, `1`, or `2`.

**Acceptance criteria:**
- A connected Black chain from the top edge to the bottom edge returns `1`.
- A connected White chain from the left edge to the right edge returns `2`.
- Complete legal Hex games terminate with a winner rather than a draw.

### REQ-HEX-003: Hex Energy Players and Round Robin

The Hex cartridge MUST expose a position energy for the current player based on
the negative longest connected path strength toward that player's goal. It MUST
provide random, greedy energy, and blocked-Gibbs energy players, and the Hex
experiment MUST report 7x7 round-robin win rates for Random, Greedy, and Gibbs.

**Acceptance criteria:**
- `RandomPlayer` samples uniformly from legal actions.
- `GreedyEnergyPlayer` selects a legal action with minimal post-move energy.
- `GibbsEnergyPlayer` uses `Phase4Sampler` blocked Gibbs sampling to minimize a
  k=5 AND-composed free energy over candidate moves.
- The experiment artifact reports at least 30 total 7x7 games.

## Scenarios

### SCENARIO-CONNECT4-001: Empty Board Ground State

**Given** a Connect Four cartridge configured with zero initial pieces
**When** the empty board is scored
**Then** the energy is 0.0 and the sampler returns an all-empty valid board.

**Spec traces:** REQ-CONNECT4-001, REQ-CONNECT4-003

### SCENARIO-CONNECT4-002: Floating Piece Repair

**Given** a board containing an occupied cell above an empty lower cell
**When** the cartridge scores the board
**Then** the energy is positive
**And** sampling repairs the board to a gravity-valid zero-energy board with
the same occupied-cell count.

**Spec traces:** REQ-CONNECT4-001, REQ-CONNECT4-003

### SCENARIO-CONNECT4-003: Winner Detection

**Given** Connect Four boards containing horizontal, vertical, or diagonal
four-in-a-row patterns
**When** winner detection runs
**Then** it returns the matching `"RED"` or `"YELLOW"` winner.

**Spec traces:** REQ-CONNECT4-002

### SCENARIO-HEX-001: Legal Move Enumeration

**Given** a partially occupied Hex board
**When** legal move enumeration runs
**Then** only empty cells are returned as row-major `(row, col)` actions.

**Spec traces:** REQ-HEX-001

### SCENARIO-HEX-002: Connectivity Winner Detection

**Given** a Hex board with a connected edge-to-edge chain for one player
**When** winner detection runs
**Then** it returns that player's integer id.

**Spec traces:** REQ-HEX-002

### SCENARIO-HEX-003: Complete Games Have Winners

**Given** repeated legal Hex games on finite boards
**When** players continue until no terminal move remains
**Then** every completed game reports Black or White as the winner.

**Spec traces:** REQ-HEX-002, REQ-HEX-003

## Implementation Status

| Requirement | Status | Experiment |
|-------------|--------|------------|
| REQ-CONNECT4-001 | Implemented | Exp 1175 |
| REQ-CONNECT4-002 | Implemented | Exp 1175 |
| REQ-CONNECT4-003 | Implemented | Exp 1175 |
| REQ-HEX-001 | Implemented | Exp 1188 |
| REQ-HEX-002 | Implemented | Exp 1188 |
| REQ-HEX-003 | Implemented | Exp 1188 |
| SCENARIO-CONNECT4-001 | Implemented | Exp 1175 |
| SCENARIO-CONNECT4-002 | Implemented | Exp 1175 |
| SCENARIO-CONNECT4-003 | Implemented | Exp 1175 |
| SCENARIO-HEX-001 | Implemented | Exp 1188 |
| SCENARIO-HEX-002 | Implemented | Exp 1188 |
| SCENARIO-HEX-003 | Implemented | Exp 1188 |
