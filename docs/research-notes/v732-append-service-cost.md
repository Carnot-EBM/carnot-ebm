# V732 complete append service cost

Exp8520 measures fresh local append and snapshot service after exact Exp8519
qualification. The [primary result](../../results/experiment_8520_v732_append_service_cost.json)
holds the observations, acceptance gates and validation receipts. Its raw
invocation retains the original operands, each child log, primitive clocks,
synced store files and terminal publication sidecar.

The experiment keeps the sealed V730 append protocol and original five-input
Exp8407 event tapes. V731 has no scientific append producer. Exp8494 retains its
original 20261013 date and disqualified timing status. Those clocks do not enter
the fresh speed estimate. Imported model output appears only in historical
provenance. This experiment loads no model and makes zero current LLM calls.

Both natural cells use 64 decisions and eight predictions per update. Batch1 is
primary; batch8 is secondary. Each cell has one warmup and five measured pairs.
Snapshot/append order alternates with append/snapshot. Every arm starts in a
fresh child and restores identical logical state on the same disk filesystem.
Both arms retain syscall observer overhead and all durability barriers.

Complete service clocks include numerical cold reconstruction, store startup,
feature gathering, prediction, causal feedback, serialization, file and directory
sync, checkpoint/rotation, responses and final drain. The separate restart clock
clears the numerical prefix cache and reads the retained store in the same arm
process. It includes disk replay and reconstruction; it excludes interpreter
startup. The result exposes this scope rather than calling it a fresh-process
startup measurement.

| Gate | Rule |
| --- | --- |
| Service readiness | Complete equal work, qualified durability and all required validation |
| Primary speed benefit | Lower95>1 and append/snapshot p95 ratio<=1.05 |
| Host tenfold target | Lower95>=10; separate from speed benefit |
| Secondary | Descriptive; cannot rescue primary |
| Incomplete pairs | Censored; no speed qualification |
| Generalization | Both scores remain zero; engineering positives are circular_positive |

The point estimate divides summed snapshot complete walls by summed append
complete walls. The interval resamples five whole paired repeat indices 10000
times with seed7298480. Requests and repeated local runs do not count as
independent hosts. Rust/Python NFR-01 remains open.

Per-repeat metrics retain latency, bytes, syncs, namespace operations, peak
memory/disk and pending/applied feedback counts. Complete-cost shares split cold
reconstruction, startup, transaction/response service and final drain. The
transaction span contains numerical and durable work together; it does not
isolate a hardware kernel. These shares support future measurement choices and
make no claim of achieved hardware acceleration.

Unchanged replay checks reconstruct exact responses, acknowledged prefixes and
final disk states. The imported recovery qualification covers Linux process
exit, idempotent retries and synced namespace barriers. Physical power loss is
untested. Private reconstruction prevents replay from overwriting historical
primaries. Missing inputs, intentional failure and rehashed tampering are
separate controls. Unchanged adversarial and strict row checks validate terminal
bytes before publication, followed by the actual downstream consumer.

Validation collects original test IDs before exact disjoint serial shards.
E2E-018 and E2E-021 keep private roots and original dates. Coverage applies only
to the three newly owned implementation files and includes real child processes.
The conductor owns operational and traceability reconciliation.

An unpublished owned invocation exposed implicit imports when strict typing
included both the adapter and shipped files. The adapter now imports those
modules and Store directly. A regression runs that exact seven-file type check.
The stopped invocation remains disqualified under raw
`invocations/1791997210671696852/aborted_owned_invocation.json`; its clocks do not
enter the final result. The final invocation measures every arm again.
