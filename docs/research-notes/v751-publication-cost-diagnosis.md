# V751 publication cost diagnosis — Exp8764

Date: 2026-11-03. This task measures CPU publication work. It loads no model,
uses `MODEL_SPECS=[]`, and makes zero LLM calls. Numerical reconstruction is
validation work, not a new learning experiment.

The governing requirements are REQ-REPORT-V751-PLAN, REQ-REPORT-8764,
SCENARIO-REPORT-8764-NULL, REQ-VERIFY-8764 and
SCENARIO-VERIFY-8764-REPLAY. The final primary and its invocation-bound raw
receipts determine the measured outcome. This note describes the frozen method.

Exp8750 remains disqualified. Its unprofiled supervisor raised `KeyError: event`
before cleanup. Its first case later completed beyond the420-second budget.
The timer arm also exceeded its budget. Retained stdout includes20 cold-replay
cell messages and100 retention-boundary messages in the unprofiled parent.
Those messages count emitted progress, not function calls or independent cases.
Parent snapshots and timer stacks locate publication waits but do not establish
why the old timeout occurred. Late output cannot qualify the original controls.

The source path is:

```text
deadline_edge_8603.deadline_case
  causal_brier_runner_8582.run(private=True, measurement=...)
    copy four derived row files; bind the invocation clock
    build preliminary artifact; scan it in process
    build final candidate
    primary_publication.publish_primary
      write locked candidate
      causal_brier_runner_8582.terminal
        v709_execution.child -> unchanged adversarial scanner
        v709_execution.child -> unchanged strict row lint
        v709_execution.child -> fresh CLI -> stream.replay
          authenticate inputs; reconstruct every numerical cell
          checked_step -> check_step; retention; reductions; build
      hash checked bytes; write sidecar; replace primary atomically
    existing primary readers
after run returns, deadline_case calls stream.replay independently again
```

The preliminary scan and terminal scan guard different publication boundaries.
The child and parent cold checks are independent validation. Their repeated
numerical work cannot simply be skipped. Repeated source reads, row reductions
or builds inside a process are candidates for investigation; call counts alone
are insufficient to establish a safe reusable computation.

One retained original measurement supplies the numerical fixture. Its historical
runner hash differs from the current source. The established Exp8724 fixture-copy
helper preserves those historical identities and binds current source hashes.
Old execution receipts remain ineligible. The diagnostic freezes the consumed
numeric operands, authenticated upstream envelope bytes, current transitive code
closure, interpreter, dependency versions, exact collected test nodes and coverage
configuration. Historical validation histories inside upstream envelopes are not
current executable operands. No historical verdict is reused as current success.

Each original stage runs in its own supervised process: preconditions, fixture
copy, materialization, adversarial scan, strict row lint and cold numerical replay.
Stages run serially with disjoint wall spans. Each has a300-second limit; the
aggregate diagnostic ceiling is1800 seconds. The first unexplained failure or
overrun ends measurement and leaves later stages explicit and censored. This
never launches the original3599/3600/3601 deadline benchmark.

The wrapper observes function boundaries for reads, hashes, builds, counters,
reductions, step reconstruction and retention. Cumulative process-local snapshots
are retained during long calls. Boundary durations are inclusive and may nest;
they must not be added to the disjoint child wall spans. Every child starts with
fresh numerical replay state. No shared validator or process-wide clock changes.
The parent emits flushed progress and30-second heartbeats. Timeout cleanup kills
the owned group and retained detached descendants, then waits for its direct child.

No correction is selected before evidence. A correction needs demonstrated repeat
work with full content/configuration identity, mutation rejection and three paired
original/corrected repeats. Without that qualification, the terminal outcome is a
diagnostic null and `publication_path_ready_score=0`. Paired rows and a complete
original-union projection are absent; neither may be inferred from partial timing.
Only Exp8765 can qualify original deadline semantics and E2E-020.

Owned validation includes requirement-linked tests, real CLI children, actual
statement coverage without exclusions, unchanged publication/clock consumers,
private E2E-018/021, Ruff check and format, strict mypy and explicit spec tracing.
One bounded invocation of the complete Python suite is retained as repository
health. Its outcome does not replace scoped acceptance or imply a full-suite pass.
Atomic publication uses unchanged adversarial and strict row scanners and a fresh
process that rebuilds this task's exact terminal fields from authenticated evidence.
The conductor owns ops/status, changelog and BMAD traceability reconciliation.

## Owned observer correction during implementation

The first diagnostic parent incorrectly treated the original scanner's warning
exit1 as failure. The scanner finished in50.06 seconds with nine warnings and
no critical findings; no stage exceeded its budget. The parent was interrupted
before publication. Its raw directory retains the killed process tree, source
snapshots and the unqualified observations. Those stage rows are not reused as
current successful evidence.

The corrected supervisor preserves actual exit codes and accepts only the
original expected exit, exact frozen findings and no critical findings. New
regressions require both warning acceptance and changed-finding rejection. This
is a diagnostic observer correction, not a numerical optimization or a cached
verdict. The single global health command had already begun during the first
attempt. Its interruption, unknown wait status and log hashes are retained;
it is not run again or represented as a full-suite pass.

## Measured terminal stages

Invocation `1793711285202145301` completed all six original stages. No stage
exceeded its budget. These are disjoint inclusive child wall times:

| Stage | Seconds | Actual exit | Accepted |
|---|---:|---:|---|
| precondition | 11.547652 | 0 | True |
| copy | 1.891109 | 0 | True |
| materialize | 7.178182 | 0 | True |
| scan | 50.864632 | 1 | True |
| row_lint | 0.474714 | 0 | True |
| cold_replay | 14.058980 | 0 | True |

Materialization made one publication build, two row reductions and260
source-read calls. There were120 source-read keys visited twice. Fresh cold
replay made100 step reconstructions,20 retention replays, three row reductions,
two counter materializations, one publication build,778 source-read calls and
2412 source-hash calls. Nested boundary durations are not additive.

The repeated reads and reductions identify candidate local work; no cache or
input-materialization correction was qualified. Independent parent/child replay
requirements remain intact. No paired corrected repeats or complete original-union
projection exist. The diagnostic therefore supports a terminal null with readiness
zero, and does not explain the historical timeout causally. Scoped validation and
final publication receipts are recorded in the primary and its bound sidecar.

The second parent's six original stages and all scoped commands completed, but
its final coverage merge raised `NoDataError`: the output basename was used to
search for differently named parallel shards. Genuine SQLite shards remain in
that raw directory. A failing real-SQLite regression reproduced the error, and
the combiner now receives explicit shard paths. This owned validation failure
prevented publication. The final invocation freezes the repaired sources and
measures its own stages; earlier successful stage observations remain historical.

## Final invocation evidence

The repaired source/configuration freeze belongs to invocation `1793711938729949984`.
All six original stages completed normally with successful cleanup. These rows
supersede earlier attempts for current measurement; earlier raw bytes remain.

| Stage | Seconds | Actual exit | Expected exit |
|---|---:|---:|---:|
| precondition | 11.345483 | 0 | 0 |
| copy | 1.888748 | 0 | 0 |
| materialize | 7.244527 | 0 | 0 |
| scan | 49.763194 | 1 | 1 |
| row_lint | 0.477330 | 0 | 0 |
| cold_replay | 14.102722 | 0 | 0 |

The21 owned tests pass. Final scoped acceptance and genuine statement coverage
are determined by this invocation's retained receipts and coverage SQLite. The
primary and bound terminal sidecar record publication acceptance. The original
deadline panel remains unrun, and publication readiness remains zero.
