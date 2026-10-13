# V728 complete durable journal service cost

Exp8466 compares the sealed Exp8464 natural tape using the unchanged qualified
Exp8465 snapshot and journal stores. This is a cached feature local verifier
service. It performs no model acquisition, model load or current LLM call.
Generator weights stay frozen. Public labels and model outputs remain retrospective.

The primary cell uses batch1/reuse8. The secondary cell uses batch8/reuse8.
Each cell has one warmup pair and five measured pairs, alternating snapshot first
and journal first. Every arm runs in a fresh process on the same disk filesystem.
Every arm restores the same initial head. The complete clock includes cold
numerical prefix reconstruction, durable initialization, all decision and delayed
feedback requests, final reads and the journal's final checkpoint. The qualified
journal retains segments and checkpoints; it performs no reclamation or compaction.
No pending maintenance or feedback work is omitted from the clock.

Cold reconstruction is reported separately and divided by actual attempted
decisions. Startup excludes that separately reported reconstruction. Restart
clears numerical caches and reads the committed store outside the complete run
clock. It is reported as an additional deployment cost. Actual fsync calls are
counted. Linux process write counters include overwritten metadata; the metric
subtracts flushed progress text. Kernel write bytes are retained separately.
Memory is the isolated process peak RSS through service completion. All store
files grow or persist, so the retained final bytes are the peak store disk use.

Each primitive keeps exact predictions, feedback responses, acknowledgment IDs,
logical state, monotonic request clocks and durable store file hashes. Replay
reconstructs the entire tape from independently sealed initial operands. It also
reads private copies of the retained stores. Rehashed aggregate, source, state,
clock, acknowledgment and store mutations are negative controls.

The speed estimate divides pooled snapshot complete wall time by pooled journal
complete wall time. The interval resamples paired repeat indices10000 times with
seed7278451. The p95 ratio divides mean journal repeat p95 by mean snapshot repeat
p95. Every batch member receives its whole transaction latency. Primary benefit
requires lower95>1, p95 ratio<=1.05 and all exact accounting and qualification
checks. The descriptive tenfold target separately requires lower95>=10.
Incomplete measurement reports missing arms and no partial speed ratio.

Five same-host repeats establish conditional engineering evidence only. Oracle
engineering positives use circular_positive. Both generalization scores stay0.
Even a tenfold host-storage benefit leaves NFR-01 open: it requires matched
Rust versus Python service measurements. Process-exit recovery and actual fsync
do not certify physical power loss.

V717 H1=-0.00390625 and H2=0 remain closed. V726's admission-learning audit remains
null. Its arithmetic and persistence shares remain0.093 percent and96.44 percent.
V727 methods readiness remains0, with no executed scalar-source, Brier-objective
or journal-cost hypothesis. The earlier acceleration ceiling stays closed.

The dated command is:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=python:. PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/experiments/experiment_8466_v728_journal_service_cost.py --date 20261013
```

Scoped unit and real private CLI checks, genuine unit/child coverage shards,
private E2E-018 and E2E-021, Ruff, strict mypy and spec references qualify owned
work. Unchanged adversarial and strict row consumers check private candidates.
The unchanged primary publisher exposes only checked atomic terminal bytes.
The conductor owns ops/status/changelog and BMAD reconciliation.

## Measured costs on 20261013

All24 intended arms completed, including both warmup pairs. Every arm attempted
64 decisions and completed64 decisions. Every arm processed8 delayed feedback
events, applied7 applicable updates and left no pending feedback. Exact states,
predictions, feedback and acknowledgment ledgers agree between stores.

| Cell | Mean snapshot complete wall | Mean journal complete wall | Pooled speedup | Paired95 percent interval | Journal/snapshot p95 |
|---|---:|---:|---:|---:|---:|
| Primary batch1/reuse8 | 15.112337s | 14.464565s | 1.044783 | [0.994235,1.109590] | 0.500556 |
| Secondary batch8/reuse8 | 3.585822s | 3.935408s | 0.911169 | [0.802486,1.018855] | 0.712882 |

The primary lower bound is below1. The primary speed-benefit gate fails despite
the lower point estimate for journal cost. The journal is slower in the secondary
cell. Neither cell establishes the descriptive tenfold target. These are complete
paired estimates; no slow arm was removed. The engineering benefit result is null.

| Cell and arm | Mean cold reconstruction | Mean startup | Mean restart replay | Actual write bytes/run | Fsyncs/run | Retained store bytes/run |
|---|---:|---:|---:|---:|---:|---:|
| Batch1 snapshot | 0.189559s | 0.282743s | 0.089036s | 1532938 | 292 | 1519906 |
| Batch1 journal | 0.190672s | 0.667357s | 0.093326s | 389253 | 243 | 388828 |
| Batch8 snapshot | 0.046796s | 0.278314s | 0.042929s | 398928 | 68 | 396032 |
| Batch8 journal | 0.053552s | 0.661446s | 0.046583s | 155739 | 63 | 155598 |

Cold reconstruction is already charged inside each complete wall. Each primitive
also divides that cost by64 actual attempted decisions. Restart is an additional
cost. Every isolated arm reports peak RSS640606208 bytes through service completion.
The JSON retains per-repeat request p50/p95 and all complete primitive clocks.

The result is [the Exp8466 artifact](../../results/experiment_8466_v728_journal_service_cost.json).
Primitive evidence is under
`results/raw/experiment_8466_v728_journal_service_cost/invocations/1791850339210658455/`.
Required validation determines readiness independently of this null benefit.

The first validation attempt passed317/317 statement coverage and every scoped
unit, E2E, lint, type and spec command. Its runner then stopped while constructing
the deliberate rehashed primitive control: the builder rejected the altered
state before the expected cold child could run. The reporting correction clears
readiness for invalid primitive candidates and allows that negative replay.
Validation resumes from the same sealed24 arms; it does not repeat measurements.
Original owned source bytes and the failure record remain in the original
invocation's `original_owned_sources/` and `owned_validation_abort.json`.
The terminal artifact binds both source versions and the prior work reference.

The final scoped recheck passes28 tests and344/344 newly owned statements.
Private E2E-018 and E2E-021, Ruff check/format, strict mypy and spec tracing pass.
All validation commands retain actual exits, durations and hashed logs below the
terminal invocation. Historical publications and validation consumers stay unchanged.
