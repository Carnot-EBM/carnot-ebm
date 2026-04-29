# WOPR Games Gallery — Extending the Sudoku Demo

**Status:** Draft change proposal, follow-up to
`huggingface-spaces-sudoku-demo.md`. Ships incrementally after the
Sudoku demo is live.

**Origin:** 2026-04-29. Once the Sudoku-with-WOPR-aesthetic demo is
live, the natural extension is a *gallery* of energy-minimisation
games under the same WOPR shell. Each new game is a separate energy
formulation; the WOPR terminal just selects which one to "play."

This makes Carnot's "general energy-based reasoning" claim concrete
across multiple problem domains, all visually accessible from the
same Spaces page.

**Target window:** ships incrementally over 4-8 weeks after the
v1 Sudoku Spaces demo lands. Each game is a 1-3 day increment.

**Priority:** **Medium-high.** The gallery extension is what
converts the Sudoku demo from "neat one-off" to "this is a paradigm,
not a trick."

**Depends on:** the v1 Sudoku Spaces demo (per
`huggingface-spaces-sudoku-demo.md`) being live.

## Strategic rationale

A single Sudoku demo demonstrates Carnot's energy framework on one
problem class (constraint satisfaction). A *gallery* of games
demonstrates the framework's *generality*:

| Game | Problem class | Why include it |
|------|---------------|----------------|
| Sudoku | Constraint satisfaction | Already shipped (v1) |
| Tic-tac-toe | Adversarial game tree (small) | The literal WarGames game; iconic; trivial energy |
| Connect Four | Adversarial game tree (medium) | Tractable solved game; visual |
| Checkers | Adversarial game tree (medium-large) | Iconic; ~$10^{20}$ states; tractable for EBM |
| Reversi/Othello | Adversarial game tree (medium-large) | Endgame is constraint-satisfaction; clean energy formulation |
| N-Queens | Constraint satisfaction (variable size) | Classic; visual; scales |
| Graph coloring | Constraint satisfaction (real-world) | Map colouring, register allocation |
| Chess | Adversarial game tree (large) | Iconic; ~$10^{120}$ states; demonstrates limits |

The gallery answers a question visitors have after Sudoku: *"OK,
but does this generalise?"* The answer is unambiguous when they can
click between Sudoku, checkers, and chess from the same page.

## Game shipping order (recommended)

Order chosen by ascending complexity of the energy formulation. Each
game adds one new "WOPR game cartridge" to the Spaces app.

### Increment 1 — Tic-Tac-Toe (1 day)

The literal WarGames game. Tiny state space (~$10^4$ positions).
Energy formulation:
- Win-loss-draw evaluator from any position (lookup table).
- Adversarial: WOPR plays optimally; user plays as challenger.
- Demonstrates the canonical WarGames lesson on-camera: WOPR
  recognises tic-tac-toe is a forced draw between optimal players.

**Deliverable:** `spaces/wopr-games/games/tictactoe.py` + selector
entry in the Spaces UI.

**Iconic moment:** WOPR explicitly displays
`A STRANGE GAME. THE ONLY WINNING MOVE IS NOT TO PLAY.` after
playing itself to a draw.

### Increment 2 — N-Queens (1 day)

Classic constraint-satisfaction problem. Energy formulation already
established (similar to Sudoku's uniqueness constraints applied to
diagonals + rows + columns). Scales from N=4 to N=20 with continuous
visual feedback.

**Deliverable:** `spaces/wopr-games/games/nqueens.py` with selectable
board size N.

### Increment 3 — Connect Four (2 days)

Adversarial game with known solved-game status. Energy:
- Heuristic from board pattern (winning lines, blocking
  opportunities).
- Optional: pretrained value network as energy.
- WOPR plays via energy-minimised move selection.

**Deliverable:** `spaces/wopr-games/games/connect_four.py` with
adjustable difficulty.

### Increment 4 — Checkers (2-3 days)

Larger state space than Connect Four; iconic American-style
checkers (8x8). Energy formulation:
- Material count + king bonus (basic).
- Mobility evaluation.
- Center-control bonus.
- WOPR plays at adjustable depth via energy-bounded MCTS or pure
  energy minimisation per move.

**Deliverable:** `spaces/wopr-games/games/checkers.py` with WOPR
playing at user-selected difficulty.

**Iconic moment:** include a "ladders" preset where Carnot can
show solving forced wins.

### Increment 5 — Reversi/Othello (2 days)

Endgame is pure constraint-satisfaction (counting). Mid-game uses
heuristic energy. Visual aesthetic is excellent (board flips
cascade nicely on the WOPR terminal).

**Deliverable:** `spaces/wopr-games/games/reversi.py`.

### Increment 6 — Graph Coloring (1-2 days)

Real-world constraint satisfaction (map colouring, register
allocation). Demonstrates that the energy framework solves problems
beyond games.

**Deliverable:** `spaces/wopr-games/games/graph_coloring.py` with
preset graphs (USA states, simple graph theory examples).

### Increment 7 (optional, ambitious) — Chess (1-2 weeks)

Chess is hard. State space is $10^{120}$; modern Stockfish is far
stronger than any pure-EBM approach. **Carnot will lose to
Stockfish** — that's the point.

The honest framing:
- WOPR plays via Carnot's energy formulation against Stockfish at
  ELO ~1200 (consumer-grade baseline).
- Visualisation shows energy descent across move evaluation.
- Educational frame: *"Why doesn't this win? Because chess's energy
  landscape is too high-dimensional for the unstructured Boolean
  rotation defence; foundation-model-scale training is required.
  See the position paper for the architectural fix."*

**Deliverable:** `spaces/wopr-games/games/chess.py` with explicit
"educational defeat" framing and a link to the position paper.

This increment is OPTIONAL — the gallery is complete and compelling
without it. Chess is included only if it strengthens the
"foundation-model self-distillation" narrative for the position
paper.

## Architecture notes

The gallery uses a single Spaces app with a game selector. Each
game module exposes:

```python
class WOPRGame:
    name: str
    description: str
    energy_function: Callable[[State], float]
    available_actions: Callable[[State], list[Action]]
    apply_action: Callable[[State, Action], State]
    visualize: Callable[[State, Energy], HTMLBlob]  # WOPR-styled

    def carnot_play(self, state: State) -> Action:
        """Use Carnot sampler to select energy-minimising action."""
```

This keeps each game small and isolated; the WOPR shell handles
common UI (terminal aesthetic, typewriter streaming, energy bar,
flavour text). Adding a new game is a single new file.

## WOPR aesthetic across games

Every game shares the WOPR shell from the v1 Sudoku demo:
- Green-on-black CRT terminal
- Typewriter character streaming
- Periodic flavour text:
  - On boot: `GREETINGS PROFESSOR FALKEN.`
  - On game selection: `LOADING [GAME NAME]. STAND BY.`
  - On move: `EVALUATING [N] CANDIDATE MOVES. ENERGY GRADIENT...`
  - On loss: `INTERESTING. LET ME RECONSIDER.`
  - On win: `EXIT STATE: VICTORIOUS.`
  - On draw: `A STRANGE GAME. NEITHER PLAYER PREVAILS.`

**Easter eggs (across all games):**
- Typing `LIST GAMES` in any game's input echoes the gallery list.
- Typing `GLOBAL THERMONUCLEAR WAR` triggers the WarGames quote
  and redirects to `tictactoe.py`.
- Typing `HOW ABOUT A NICE GAME OF CHESS` skips to chess (if
  shipped).

## Forward links beyond the gallery

- **Hardware demo extension:** when KV260 Hybrid Coprocessor lands,
  add a "WOPR ON FPGA" mode showing 100× speedup on checkers/chess
  move-evaluation.
- **Adversarial demo extension:** when the multi-verifier rotation
  architecture (v0.4) ships, add a "WOPR vs Adversarial WOPR" mode
  showing the rotation-defence in action — one WOPR plays optimally;
  the other tries to game-the-energy.
- **Educational blog post:** "Eight Games, One Energy Function:
  How Carnot Generalises" — pairs with the gallery for a Twitter/HN
  launch when each game lands.

## Estimated effort

| Increment | Effort | Cumulative |
|-----------|--------|-----------|
| 1. Tic-Tac-Toe | 1 day | 1 day |
| 2. N-Queens | 1 day | 2 days |
| 3. Connect Four | 2 days | 4 days |
| 4. Checkers | 2-3 days | 6-7 days |
| 5. Reversi | 2 days | 8-9 days |
| 6. Graph Coloring | 1-2 days | 9-11 days |
| 7. Chess (optional) | 1-2 weeks | 16-25 days |

The base gallery (Increments 1-6) ships in ~2 weeks of part-time
work spread across other deliverables. Chess adds another 1-2 weeks
if pursued.

## Acceptance criteria

1. Each game ships as an independent module under
   `spaces/wopr-games/games/`.
2. WOPR shell stays consistent across games — same aesthetic, same
   typewriter speed, same flavour-text cadence.
3. Each game runs on Spaces' free CPU tier in <60s per move (or
   <60s total for solved-game increments).
4. Each game has at least one preset that produces a memorable
   WarGames-style moment (forced draw on tic-tac-toe, ladder on
   checkers, etc.).
5. Gallery selector page lists all games with brief description and
   "PLAY" button styled as WOPR command.
6. Every WarGames easter egg works as documented.

## Why this proposal is filed alongside the Sudoku one

The base Sudoku demo and the games-gallery extension are
*architecturally* the same project (one Spaces app, one WOPR shell)
but *strategically* separable: Sudoku ships first in 3-5 days; games
gallery extends that base over the following weeks.

Filing as separate proposals lets the .82 / .83 planner pick them up
independently. Sudoku-first is a hard precondition; gallery is a
modular extension where each game is its own ship-it artifact.
