---
title: WOPR Games
emoji: "\U0001F4BE"
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: "Carnot EBM reasoning demos in a WarGames CRT terminal"
hardware: cpu-basic
---

# WOPR Games — Carnot Energy-Based Reasoning Demos

> *"GREETINGS PROFESSOR FALKEN."*
> *"SHALL WE PLAY A GAME?"*

A HuggingFace Space showcasing Carnot's energy-based verification on
classic puzzle and game problems, all wrapped in the iconic WOPR
terminal aesthetic from *WarGames* (1983).

## What this is

Carnot is a verifier-filtered self-distillation framework — its core
contribution is a closed-form Phase-3 → Phase-8 architecture for
energy-based output verification (see the position paper). This
Space turns the abstract energy-minimization machinery into
*clickable demos* across multiple problem classes.

Each game is a separate **cartridge** with the same WOPR shell.
You can:

- Watch Carnot's MCMC sampler descend the energy landscape live
  (Sudoku grid fills in cell-by-cell; Lights Out cells cascade off;
  Hex board paint paths)
- Type WarGames-iconic commands: `LIST GAMES`,
  `GLOBAL THERMONUCLEAR WAR`, `HOW ABOUT A NICE GAME OF CHESS`,
  `GREETINGS PROFESSOR FALKEN`
- Watch the energy bar collapse to zero as the puzzle solves

## Cartridges shipped

- **Sudoku v1** — the canonical constraint-satisfaction demo
- **Tic-Tac-Toe** — the literal WarGames game (forced draw)
- **Global Thermonuclear War** — *"A STRANGE GAME. THE ONLY WINNING MOVE IS NOT TO PLAY."*
- **Lights Out** — XOR-based grid puzzle (perfect Carnot-Ising fit)
- **N-Queens** — classical combinatorial CSP; Ising spins encode queen placement
- **Hashi** — bridge-count CSP; planar graph connectivity via spin variables (exp1124)

## Cartridges planned

See `openspec/change-proposals/wopr-games-gallery-extension.md` in
the main repo for the full 16-cartridge plan: Connect Four,
Nonograms, Conway's Life-reverse, Slitherlink, Hex, Sokoban,
Cryptarithmetic, Mastermind, optionally Chess.

## Benchmark results (live GPU, 2026-05-01)

All results from GPU-verified experiments on the Carnot FoVer evaluation corpus.

| Metric | Value | Source |
|--------|-------|--------|
| ThinkPRM v2 AUROC (eval 500 FoVer) | 0.9946 | exp1111 |
| ThinkPRM v1 AUROC baseline | 0.9885 | exp1111 |
| AUROC improvement v1 -> v2 | +0.0061 | exp1111 |
| k=5 AND-composition max pairwise r | 0.462 | exp1108/exp1121 |
| k=5 AND-composition production deployed | yes | exp1121 |
| Energy verifier retrain AUROC (post-inversion fix) | 0.9774 | exp1120 |
| Energy inversion fixed after retrain | yes | exp1120 |
| LLM failure exemplar corpus size | 36 exemplars | exp1112 |
| Mathematical-objective category TP rate | 100% (arithmetic+code) | exp1112 |
| Cascade TP rate (all 12 categories) | 80.6% | exp1112 |

The k=5 AND-composition empirically validates the k_max~7-8 architectural
assumption: 14 of 15 pairwise correlations fall below the r=0.5 threshold,
confirming that cross-mechanism diversity (Z3 + AST + Semantic) drives
ensemble decorrelation.

## Architecture

```
spaces/wopr-games/
├── app.py                   # Gradio entry point
├── wopr_shell.py            # CRT terminal aesthetic + easter eggs
├── games/
│   ├── __init__.py
│   ├── _base.py             # WOPRGame interface
│   ├── sudoku.py            # Sudoku cartridge
│   ├── tictactoe.py         # Tic-Tac-Toe cartridge
│   ├── lights_out.py        # Lights Out cartridge
│   ├── thermonuclear_war.py # The non-game game
│   ├── nqueens.py           # N-Queens CSP cartridge
│   └── hashi.py             # Hashi bridge-count cartridge (exp1124)
├── requirements.txt
└── README.md
```

Each cartridge implements:

```python
class WOPRGame:
    name: str
    description: str
    initial_state: Callable[[], State]
    energy_function: Callable[[State], float]
    available_actions: Callable[[State], list[Action]]
    apply_action: Callable[[State, Action], State]
    visualize: Callable[[State, float], str]   # WOPR-styled HTML
    carnot_play: Callable[[State], Action]      # use Carnot sampler
```

## Local development

```bash
cd spaces/wopr-games
pip install -r requirements.txt
python app.py
# Opens local Gradio server at http://localhost:7860
```

## Deployment

Push to a HuggingFace Space (`Carnot-EBM/wopr-games`):

```bash
git remote add hf https://huggingface.co/spaces/Carnot-EBM/wopr-games
git subtree push --prefix spaces/wopr-games hf main
```

## Easter eggs

| Command | Behavior |
|---|---|
| `LIST GAMES` | Echo the gallery list |
| `GLOBAL THERMONUCLEAR WAR` | Redirect to thermonuclear-war cartridge |
| `HOW ABOUT A NICE GAME OF CHESS` | Print the chess-coming-soon message + redirect to tic-tac-toe |
| `GREETINGS PROFESSOR FALKEN` | Print boot sequence |
| `JOSHUA` | Reveal the secret password |
| `LEARN` | Display "I'M LEARNING. PLEASE WAIT." with energy descent animation |

## Context

This Space is the empirical companion to the Carnot position paper
(*"Carnot: A Provably-Bounded Architecture for Verifier-Filtered
Self-Distillation Under Concept Drift"*). The paper's Phase-3 →
Phase-8 architecture is theoretical; this Space lets you *play with*
the energy-minimization mechanics that make it work.

The architecture chain is documented at
`docs/research-notes/*-deep-think-results.md` in the main repo.
