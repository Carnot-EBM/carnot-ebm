# V728 persistence localization

Exp8464 measures the complete original five-input Exp8407 service on a real
disk-backed filesystem. Exp8457's sealed service root supplies the frozen head,
source slots and delayed labels. Historical admission reconstructs the original
producer authority. The blocked Exp8449 producer is not an admission gate.

The primary cell is natural batch1/reuse8. The secondary cell is natural
batch8/reuse8. Each has one warmup pair and five measured pairs. Every arm
attempts64 predictions and8 designated feedback releases. Labels remain delayed
by eight event ticks. Feedback boundaries flush batches. The final tail drains
before the outer clock stops. Startup and initial durable state are included.
The full27-cell grid is not repeated.

Each pair alternates the original qualified store and the instrumented adapter.
The adapter preserves snapshot and pointer bytes, the2MiB state byte limit, and
the original order: snapshot write, file fsync, snapshot rename, directory fsync,
pointer encoding and write, pointer file fsync, pointer rename, directory fsync,
acknowledgment. Extra monotonic clocks and bookkeeping change elapsed time.
Whole-clock overhead is reported for each pair, including negative host noise.
No overhead is subtracted from any stage or complete clock.

Disjoint persistence leaves separate serialization and hashing, the original
byte-budget validation, buffered writes and flushes, file fsyncs, renames,
directory fsyncs, and pointer operations. The pointer encoder's individual
writes are timed separately from its CPU work. The store does not copy the whole
state during commit. Its state-copy count is therefore zero within persistence;
validation time is still measured. Service startup, input conversion, snapshot
reads, scoring, scheduling, response encoding, measurement bookkeeping and gaps
remain explicit remaining orchestration. That residual is not disk latency.

The original Exp8407 reducer reconstructs all request and feedback semantics and
reads every saved snapshot. An additional reducer checks persistence nesting,
operation order, four fsyncs per commit and response hashes. Paired final states
and response encodings must match exactly. Primitive files preserve request and
transaction IDs, monotonic clocks, byte counts, file hashes and stage leaves.
The frozen workload also records exact event order and expected state/response
hashes for journal qualification and complete cost comparison.

Readiness requires authentic complete pairs, unchanged causal semantics, exact
nesting,100 percent owned statement coverage and required validation. It does
not require a favorable bottleneck. Incomplete arms retain their intended
slots and block cost comparison. Missing external operands block; owned failures
disqualify. Cold processes reject missing inputs, deliberate errors, rehashed
aggregate changes and rehashed primitive changes. Unchanged publication,
adversarial and strict row consumers check terminal candidate bytes.

This task loads no model and invokes no language model. Numerical heads use
cached candidates. Reader work aggregates upstream artifacts. Imported models
remain historical provenance. Both generalization scores are zero. V727's
methods score0 and unexecuted science remain unchanged. V726's null admission
audit and historical kernel0.093 percent/persistence96.44 percent remain intact.
V717 H1=-0.00390625 and H2=0 stay closed. Generator weights remain frozen.

The experiment reserves900 seconds for validation within a4800-second task cap
and stops measurement at1800 seconds. It uses unbuffered phase and child status
output. Scoped serial tests, genuine Coverage.py unit/child shards, private
E2E-018/021, Ruff, strict mypy and owned spec coverage form the validation plan.
The conductor owns subsequent ops and BMAD reconciliation.

Run from the repository with:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=python:. PYTHONUNBUFFERED=1 \
  .venv/bin/python -u scripts/experiments/experiment_8464_v728_persistence_localization.py \
  --date 20261012
```

The terminal artifact is
`results/experiment_8464_v728_persistence_localization.json`. Preserved primitive
evidence, frozen replay operands and validation logs are under
`results/raw/experiment_8464_v728_persistence_localization/`.

The20261012 measurement completed all24 arms: four warmup arms and20 measured
arms, forming ten measured pairs. Admission gates and owned measurement
failures were empty. Both arms used XFS on `/dev/mapper/vg-root`, mount `/`,
device253:2. The table aggregates the five measured repeats in each cell.

| Complete measured work | Primary batch1/reuse8 | Secondary batch8/reuse8 |
|---|---:|---:|
| Instrumented whole-clock total |74.976168s |17.775043s |
| Original baseline whole-clock total |76.415690s |17.980181s |
| Instrumented mean complete repeat |14.995234s |3.555009s |
| Snapshot directory fsync total |11.955301s |2.393367s |
| Pointer directory fsync total |60.839450s |14.823027s |
| State serialization/hash total |0.084028s |0.020628s |
| Pointer serialization total |0.024803s |0.005530s |
| Remaining orchestration total |0.663691s |0.270901s |
| Paired whole-clock difference / baseline |−1.884% |−1.141% |

Directory fsync accounts for about97 percent of complete instrumented service
time in both cells. State and pointer serialization together account for
about0.15 percent. These observations isolate synchronization on this host;
they do not support a Python serialization accelerator. Remaining orchestration
is explicit and includes nonpersistence work and unclassified gaps. The negative
paired differences retain host timing variation and adapter cost. They are not
treated as a speed benefit, and no time is subtracted from the profiles.

Each pair preserves exact state and response hashes. The frozen event tape
contains72 ordered events per repeat and preserves request-level expected
outputs. Candidate journal work units contain72 service transactions for the
primary and16 for the secondary, with eight designated feedback releases in
each. Startup commits remain in the complete clocks and primitive files.
