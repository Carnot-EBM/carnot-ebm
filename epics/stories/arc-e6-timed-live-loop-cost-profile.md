# ARC E6 timed live-loop cost profile

**Status:** Complete
**Requirements:** REQ-ARC-WMTE-7491, REQ-ARC-WMTE-7492
**Experiments:** 7491, 7492

## Goal

Measure exclusive E3 loop cost. Separate verifier, planner, and environment
time. Add enough complete current-model episodes for the E6 gate.

## Stage 1

The observer imports Experiment 7471. It is off by default. It records nested
monotonic spans and joins backend tokens by request ID. Recorder failures are
caught and counted. Tests prove on/off parity and interval reconciliation.

The frozen panel has 12 non-E4 public games and three seeds per game. The
schedule has 36 units. The reducer imports Experiment 7490 and fails closed on
sample, control, reconciliation, token, separation, attribution, and kill
gates.

## Stage 2

Experiment 7491 ran on physical GPU 1 after the corrected pre-initialization
idle check passed. Its owned server occupied 18,030 MiB and all 36 units
completed with zero recorder errors. Experiment 7492 then combined the 36
fully timed rows with 26 earlier compatible rows. Every support,
positive-control, reconciliation, token-join, separation, attribution, and
kill-rule gate passed, so numeric shares and Amdahl ceilings were published.
Both terminal artifacts pass adversarial verification.

No hidden-game efficacy claim is in scope.
