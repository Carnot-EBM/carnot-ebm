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

## Implementation Status

| Requirement | Status | Experiment |
|-------------|--------|------------|
| REQ-CONNECT4-001 | Implemented | Exp 1175 |
| REQ-CONNECT4-002 | Implemented | Exp 1175 |
| REQ-CONNECT4-003 | Implemented | Exp 1175 |
| SCENARIO-CONNECT4-001 | Implemented | Exp 1175 |
| SCENARIO-CONNECT4-002 | Implemented | Exp 1175 |
| SCENARIO-CONNECT4-003 | Implemented | Exp 1175 |
