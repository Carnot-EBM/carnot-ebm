# V725 methods, sealed 2026-10-11

The protocol is `openspec/change-proposals/v725-methods-protocol.json`.
It was written before analysis. Its source snapshots retain actual activation,
design, release, references, exclusion manifest and predecessor bytes.
The active roadmap contains only Exp8416–8419. The independent design has no
exact task table, embedded complete task contract or canonical task digest.
Exp8420–8428 complete objects are absent. Contract admission remains blocked;
this note does not invent them or activate a different roadmap.

The released study is retrospective. SHA256 of `v725:` plus the full stored
question hash orders all 300 questions: 150 fit, 50 tune and 100 test. The protocol
lists every question ID and all five ordered 20-question feedback blocks.
Sibling answers and four prompt conditions stay with the whole question.
Usable support floors are 128/32/80, with eight questions containing each class
in tune/test. No reseeding follows outcome inspection. Independent evaluation
still requires 80 unexposed clusters and eight per class. Public relabeling cannot
meet this requirement; older overlap remains unknown.

[The Labeling Problem](https://arxiv.org/abs/2610.08026) maps to Exp8417–8419:
use human factuality labels and keep reference support a separate criterion.
This does not establish new current-Qwen skill.
[The Detectability Gap](https://arxiv.org/html/2609.35860v1) maps to Exp8417/8419:
fix dataset/generator strata without detector scores and retain paired questions.
The paper's alias-based labels differ from independent adjudication.
[Online KAN](https://arxiv.org/abs/2602.02056) maps to Exp8418/8420:
reuse the shipped cubic basis and sparse updates, then measure writes and retention.
The published FPGA result does not prove local KV260 support or speed.
[Capacity-constrained delayed OCO](https://arxiv.org/html/2606.11711v1) maps to
Exp8420–8421: compare FIFO and uniform admission at finite capacity, recording
inclusion probability C/20. This empirical adaptive learner inherits no convex
regret theorem. The dated scan's access limitations remain preserved.

Nine inputs comprise p2 hallucination probability and eight public lexical
features. Generator identity is excluded. Preserve abstentions and source bytes.
Five arms use seeds 17/29/43/71/101, 200 steps, rate 0.01 and l2 0.001.
Temperature 0.5/1/2 is selected on tune log loss. Freeze fitted bytes before test.
Accept p<=0.2, reject p>=0.8, otherwise escalate; costs are 0/1/0.2.
Brier(logistic)-Brier(spline) uses 10,000 question bootstrap draws, seed 7258419,
after seed/answer averaging. The exploratory lower95 must be positive without
increased decision loss. All controls, A2 sensitivity, missing bounds and strata
remain required.

Feedback capacities are 8 and 16. Uniform draws C question IDs without replacement;
FIFO takes the first C. Predict three siblings and seal the full block before
releasing selected labels. Frozen, FIFO, uniform and paired shuffled feedback
keep equal admission accounting; frozen writes no coefficients. Audit storage
is separate. There are 4,000 planned question/arm/seed/capacity units and 12,000
answer rows. Evaluate later blocks 1–4 and all 50 tune retention questions.
Seed17/capacity8 exits after block2 and must resume the complete causal checkpoint.
The 109-coefficient checkpoint needs its own qualification.
The audit contrast Brier(FIFO)-Brier(uniform) uses seed7258421, 10,000 question
bootstrap draws, positive conditional gain versus frozen, no increased decision
loss and retention upper95 Brier increase<=0.01. Histories are adaptive and exposed.

Seven old service cells preserve V724 head, policy, work, delay and durability:
natural (1,1), (1,8), (1,64), (8,8), (32,8); ordinary (1,8); boundary (1,8).
One warmup and five paired repeats give 70 summaries. Each repeat contains eight
cycles, 8*r prediction attempts and eight designated feedback attempts, with
eight logical ticks of delay and a timed deterministic tail. Copies, gathering,
updates, serialization, file/directory fsync, publication and response remain
included. Measurement ends within 3000 seconds with 1200 reserved for validation.
Retain censored cells. A 10x claim requires paired lower95>10 at natural (1,8).
This five-input service does not qualify the new nine-input learner for native use.

Read only a supplied causal CUDA receipt within 30 seconds. No device command,
model load or generation runs here. Absence zeros the optional runtime gate.
V717/V721/V724 protocols and the preserved V724 design stay unchanged.
V717 H1=-0.00390625 and H2=0 remain closed. The old .111 directive was completed;
GRPO, puzzle, generator and external text reranking tracks remain retired.
Every generalization and generalized-learning-benefit score remains zero.
