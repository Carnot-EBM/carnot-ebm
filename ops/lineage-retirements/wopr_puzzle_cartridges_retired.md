# WOPR Puzzle Cartridge Lineage Retirement

Run date: `20260507`
Artifact: `results/experiment_1457_wopr_puzzle_cartridge_retirement.json`

## Experiments Reviewed

| experiment | cartridge | outcome | evidence | scope reason |
|---|---|---|---|---|
| exp1059 | Sudoku Space | code_complete_deploy_pending | HuggingFace Space demo scaffold created | useful WOPR demo substrate, not active verify-repair research |
| exp1069 | Sudoku HF deploy | deployed_live | Sudoku Space deploy proved the gallery path | deployment artifact remains demo-only |
| exp1070 | Global Thermonuclear War | cartridge_shipped | cultural-anchor cartridge shipped with final energy 0.0 | thematic demo, not thesis-critical |
| exp1071 | Lights Out | cartridge_shipped | classic Ising ground-state demo shipped with final energy 0.0 | early demo utility exhausted |
| exp1097 | N-Queens | cartridge_shipped | N-Queens cartridge shipped after an earlier gate-blocked attempt | another CSP demo, not a new research direction |
| exp1102 | N-Queens gallery update | gallery_updated_n_queens_live | gallery update path proved live deployment once | gallery mechanics should not drive new research tasks |
| exp1124 | Hashi | e0_achieved | Hashi bridge-counting cartridge achieved E=0 | CSP encoding lesson retained as demo |
| exp1125 | Hashi gallery update | deployed_live | HF Spaces gallery update deployed Hashi | deployment preserved, not an active line |
| exp1136 | Slitherlink | blocked_gate_check_failed | pre-gate blocked because prior_failures metadata was missing | precursor block shows gallery churn cost |
| exp1141 | Slitherlink rescue | e0_achieved | canonical puzzle reached E=0.0 in 1 iteration with 24 spins | successful rescue remains a demo asset |
| exp1175 | Connect Four | cartridge_shipped_e0_at_convergence | 42-spin gravity-valid cartridge shipped | board-game gallery baseline rather than verifier improvement |
| exp1188 | Hex | hex_operational_energy_player_wins | 7x7 Hex cartridge operational with energy-player wins | game-playing demo without a Carnot repair-thesis link |
| exp1201 | Nonogram precursor | blocked_gate_check_failed | pre-gate blocked by incomplete prior-failure metadata | another gallery-block iteration |
| exp1214 | Nonogram | nonogram_shipped_e0_at_solution | run-length solution energy E=0 | classic CSP cartridge, not current research trajectory |
| exp1227 | Futoshiki | futoshiki_shipped_e0_at_solution | valid solution E=0 and violations score positive | inequality puzzle demo with no direct verifier lift |
| exp1240 | Kakuro precursor | blocked_gate_check_failed | pre-gate blocked by incomplete prior-failure metadata | repeated gallery-gate friction |
| exp1243 | Kakuro v2 skeleton | in_progress | stale in-progress skeleton before later minimal shipment | skeleton churn does not add research signal |
| exp1253 | Masyu precursor | blocked_gate_check_failed | pre-gate blocked by incomplete prior-failure metadata | gallery line kept recurring without thesis lift |
| exp1261 | Kakuro v3 skeleton | in_progress | stale in-progress skeleton before v4 minimal shipment | skeleton churn should not be repeated |
| exp1262 | Masyu v2 skeleton | in_progress | known-issues named Masyu line before v3 minimal shipment | not enough to justify further variants |
| exp1279 | Kakuro v4 minimal | shipped | valid E=0.0 and deterministic invalid E=17.0 | minimal gallery cartridge after repeated gate blocks |
| exp1280 | Masyu v3 minimal | shipped_minimal_masyu_cartridge | valid E=0.0 and invalid E=3.0 | minimal loop-puzzle demo after repeated gate blocks |

## Known-Issues ID Notes

- ops/known-issues.md names exp1198 as Connect Four in the scope-C sketch, but research-complete.yaml identifies exp1198 as FoVer v7; the actual Connect Four cartridge artifact is exp1175.
- ops/known-issues.md names exp1262 for Masyu; the terminal shipped Masyu artifact is exp1280 after exp1253/exp1262 precursor blocks and skeletons.

## Preserved Demo Assets

- `python/carnot/games/connect_four.py`
- `python/carnot/games/hex.py`
- `python/carnot/games/nonogram.py`
- `python/carnot/games/futoshiki.py`
- `tests/python/games/test_connect_four.py`
- `tests/python/games/test_hex.py`
- `tests/python/test_nonogram_cartridge.py`
- `tests/python/test_futoshiki_cartridge.py`
- `spaces/wopr-games/app.py`
- `spaces/wopr-games/wopr_shell.py`
- `spaces/wopr-games/README.md`
- `spaces/wopr-games/games/hashi.py`
- `spaces/wopr-games/games/kakuro.py`
- `spaces/wopr-games/games/masyu.py`
- `spaces/wopr-games/games/slitherlink.py`
- `spaces/wopr-games/games/sudoku.py`
- `spaces/wopr-games/games/lights_out.py`
- `spaces/wopr-games/games/nqueens.py`
- `spaces/wopr-games/tests/test_kakuro.py`
- `spaces/wopr-games/tests/test_masyu.py`
- `spaces/wopr-games/tests/test_slitherlink.py`
- `spaces/wopr_games/games/slitherlink.py`

## Future Reopen Conditions

- An operator explicitly reopens gallery work and names why the new puzzle cartridge is research-critical rather than a demo expansion.
- The proposal states a direct verify-repair LLM thesis link, such as a measured reduction in verifier false accepts or repair failures on an LLM-output corpus.
- The proposal states a direct Phase-3 substrate link, such as a reusable hardware-acceleratable EBM primitive not already demonstrated by existing WOPR cartridges.
- The proposal includes a falsifiable acceptance gate and a retirement rule if it again produces only gallery/demo evidence.

## Research Scope Decision

The WOPR puzzle-cartridge lineage is retired as active research scope. The existing code, tests, docs, and HuggingFace Spaces assets remain usable as demos, but new puzzle-cartridge or gallery-update research tasks are blocked unless an operator explicitly reopens the gallery under the conditions above.
