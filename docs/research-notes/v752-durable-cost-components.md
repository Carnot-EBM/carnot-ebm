# V752 ordinary durable component costs

Exp8784 (`exp8784-durable-cost-components`) measures one constructed numerical
workload. The fixed plan has three ordinary starting offsets (0, 2032, 4064),
two repeats, Python/JSON-native/resident-native arms, and paired plain and
instrumented conditions: 36 complete windows and 1,152 acknowledged updates.
Arm order and observer order are randomized from seed7528784 before measurement.

Every32-update window uses two fresh child processes. The second imports the
first process's durable checkpoint after16 updates. Both checkpoint import and
post-restart state verification are included in the complete parent cost.
Each update retains the shipped Store's sealed and acknowledged transactions,
including real file and directory fsync. Arithmetic-only prefix reconstruction
is timed separately and cannot be extrapolated to whole-service throughput.

The opt-in observer compiles private copies of the shipped composite functions
with direct clock reads. It does not wrap integer primitives. Nested storage,
feature, update, serialization and acknowledgment spans are reduced to exclusive
intervals. Parent clocks cover startup and shutdown. Gaps remain unassigned.
The native boundary combines native computation, binding transit and telemetry;
Python clocks cannot identify a pure transfer cost. This limitation applies even
if attribution passes its overhead gates.

Attribution requires every instrumented/plain same-work process pair to have
at most5 percent complete-cost overhead, and clock reconciliation error to be
at most2 percent. Failure retains raw complete timings and zeros
`component_cost_ready_score`. Complete admitted timing rows independently set
`durable_cost_rows_ready_score`; neither score requires an observed advantage.
Six process pairs describe one workload and do not establish population speedup
or natural dense-event frequency. There are no model loads, generation calls,
or numerical training claims. `MODEL_SPECS=[]` and the substrate is `no_model_load`.

Private tests construct their own input-only arithmetic fixtures, use a smaller
control roster, and cannot release readiness. They do not read production
primaries. Scoped validation preserves exact nodes, JUnit, child SQLite, argv,
normal exits and log hashes. The one bounded full Python-suite diagnostic is
repository health, separate from scoped acceptance. Terminal publication runs
unchanged adversarial and strict row validators and a fresh cold replay from
independently sealed measurement/validation operands. Ops and BMAD reconciliation
remain conductor-owned.

The raw `serialization` clock is a coarse durability-envelope block. It includes
file open/write/flush/replace work outside the separately timed sync barriers;
it is not a pure JSON-encoder microbenchmark. Raw component labels describe
these observed blocks and cannot support narrower operation-level fractions.

The sealed production measurement completed all36 windows and1,152 updates in
1,066.854614157 seconds, with zero failed or censored windows. There are18
instrumented/plain arm pairs. Their worst observed complete-cost overhead is
18.6708706221 percent, above the5 percent gate, so component attribution is
unadmitted. Raw spans and complete timings are retained. This observed ratio
includes process and storage variability and is not a population estimate of
causal observer overhead.

| Plain arm | Observed complete-window range (seconds) |
|---|---:|
| Python | 24.8953–28.3319 |
| JSON-native | 23.9131–27.9346 |
| Resident-native | 26.1679–36.4811 |

Each range contains six bounded windows on one workload. These values establish
no population speedup. Final cost admission and validation receipts are carried
by `results/experiment_8784_v752_durable_cost_components.json`, with primitives
under `results/raw/experiment_8784_v752_durable_cost_components/`.

Scoped validation passed all32 nodes with no skips:18 owned tests and14
consumer/private E2E-018/021 nodes. Fresh unit/CLI-child coverage is372/372 owned
statements (100 percent), with zero excluded statements. Ruff check/format,
strict mypy and explicit spec tracing passed. The full Python-suite health
receipt is separate and does not establish a repository-wide pass.
