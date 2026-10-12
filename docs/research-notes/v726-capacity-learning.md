# V726 bounded continuous learning

Exp8434 measures a numeric learner on released public human judgments.
Its primary artifact is `results/experiment_8434_v726_capacity_online_learning.json`.
Primitive evidence remains under `results/raw/experiment_8434_v726_capacity_online_learning/`.
The thin CLI accepts `--date 20261012`. It loads no language model.
Cached GPT-5-mini judgments and generator answers remain historical provenance.
They are not current Qwen inference.

The frozen Exp8431 spline head starts each arm. Exp8433 supplies the qualified
nine-input, 109-coefficient store. The original five ordered blocks contain
20 questions and 60 sibling answers each. Seeds are 17, 29, 43, 71 and 101.
Capacity 8 is primary. Capacity 16 is secondary. Frozen and FIFO admit the
first C questions. Uniform samples C questions without replacement, before
reading targets. Shuffled feedback uses the same uniform admission IDs.
Its separate saved RNG permutes only the current block's released labels.

Each block has two durable transactions. The first seals all predictions and
admissions. A separate evaluator then releases only admitted labels.
The updater visits questions in original order and siblings in release order.
It takes one sparse log-loss step per valid answer, rate 0.01 and L2 0.001.
The fitted scaler and temperature stay fixed. Intermediate steps are tentative.
The second transaction commits the block update and cursor together.
The learner retains at most C pending clusters or 3*C answer feature records.
Unused targets stay in evaluator storage. Audit storage is separate.

This is a block-admission ablation without importance weighting.
It differs from the cited delayed-OCO scheduler and carries no regret guarantee.
All 12000 intended answer slots and 4000 question units stay in the result.
Only zero-based blocks 1 through 4 enter later-loss summaries.
A separate read-only process scores 50 tune questions after each boundary.
Tune labels never enter the learner. Seed 17/capacity 8 also undergoes a real
exit after zero-based block 2 and exact restart comparison in every arm.
Recovery equality, parameter changes and learning benefit are separate results.

Readiness requires qualified causal measurements and every required check.
A favorable loss contrast is not required. Exp8435 owns the benefit audit.
Seeds and sibling answers do not multiply the 100 test questions.
Prior public exposure and unknown historical overlap prevent independent
factuality, current-Qwen skill or hidden-game generalization claims.
Both generalization scores remain zero. V717 H1 and H2 stay closed.

Validation uses scoped serial pytest, genuine Coverage.py subprocess/exit
shards, private disk scratch, Ruff, strict mypy and spec tracing.
Private E2E-018, E2E-020 and E2E-021 remain required.
Cold replay checks primitive histories against independently pinned operands.
Missing inputs, deliberate errors and rehashed tampering are negative controls.
Unchanged adversarial and row validators retain every finding.
Failed owned checks disqualify readiness. Missing external inputs block it.
Atomic publication and both existing readers bind the terminal bytes.
The conductor handles ops and BMAD reconciliation.

The measured invocation completed all 40 cells in 1240.93 seconds.
It retained 12000 answer rows, 4000 question units and 30000 retention rows.
There were 5400 nonzero committed answer updates. Frozen made no parameter writes.
Pending high water was 24 answer records at capacity 8 and 48 at capacity 16.
Lost feedback was zero. Every real exit returned 73. All four restart states
matched their uninterrupted states exactly.

Capacity-8 later losses average the five seeds over blocks 1 through 4:

| Arm | Brier | Log loss | Decision loss |
|---|---:|---:|---:|
| Frozen | 0.1377642743 | 0.4404018732 | 0.1600000000 |
| FIFO updates | 0.1371089265 | 0.4376736735 | 0.1525000000 |
| Uniform updates | 0.1379488457 | 0.4413217464 | 0.1570000000 |
| Shuffled released feedback | 0.1381316620 | 0.4425992228 | 0.1583333333 |

These are point summaries of exposed histories. Uniform admission did not
improve Brier over FIFO or frozen in these measured histories. Exp8435 owns
uncertainty and retention acceptance. Recovery does not establish learning benefit.
