# V730 publication cost model — Exp8492

This task reconstructs qualified V728 clocks and freezes a stable append
protocol. It does not measure a new append store. The terminal qualification
and exact invocation receipts are in
[Exp8492](../../results/experiment_8492_v730_publication_cost_model.json).
Primitive operands and validation receipts remain under
`results/raw/experiment_8492_v730_publication_cost_model/`.

The changed condition is complete activated V730 authority. V729 did not
measure this mechanism after its truncated contract blocked admission.
Exp8485 supplies separate methods and service-input readiness. Exp8464,
Exp8465 and Exp8466 retain their pinned original V728 authority, primitive
bytes and unchanged historical replay consumers. No old benchmark is rerun.
The inference substrate is `aggregation_from_upstream_artifacts`, its class
is `no_model_load`, `MODEL_SPECS=[]`, and current LLM calls are zero.
Optional CUDA and boards are not CPU prerequisites.

Across the five measured localization repeats, directory synchronization
occupies 97.0905% of batch-1 complete time and 96.8571% of batch-8 complete
time. Serialization occupies 0.1452% and 0.1472%; file synchronization
occupies 1.7329% and 1.3578%. Pointer rename occupies 0.0201% and 0.0173%.
All remaining work is retained as residual: 1.0114% and 1.6206%.
The artifact retains each repeat and its warmup separately. These fractions
describe instrumented snapshot publication, not measured journal syscall
durations. Instrumentation overhead is not subtracted.

The shipped snapshot commit writes and syncs its immutable state file,
renames it, syncs the directory, then writes and syncs a temporary pointer,
replaces the pointer, and syncs the directory again. For `n` transactions,
including initialization, its count is `4*(n+1)` synchronization calls.
The shipped journal already appends to a stable event file. Each transaction
also creates a durable commit certificate through a temporary file and rename,
then syncs the directory. Checkpoint creation, checkpoint pointer replacement
and checkpoint marker creation each require file and directory synchronization.
Initialization plus periodic and final checkpoints yield
`9 + 3*n + 6*(floor(n/32)+1)` calls. These independently derived counts match
all original Exp8466 rows. There are no measured directory syncs without a
namespace change. The new mechanism removes transaction certificate namespace
publication, rather than deleting a durability obligation from the old store.

The conditional ceiling charges each original journal complete clock, then
subtracts transaction-only snapshot pointer serialization, write, file sync,
rename and directory sync. Initialization and checkpoint costs remain charged.
The pooled ceilings are 6.14230x for batch1 and 3.61748x for batch8 under three
explicit assumptions: retained work is unchanged, the new protocol adds zero
overhead, and snapshot pointer marginal costs equal journal certificate
publication marginal costs. Journal primitives do not contain disjoint
syscall durations, so this transfer is an assumption. Each row reports its
own denominator and conditional ceiling; a nonpositive retained denominator
invalidates the bound. `measured_append_speedup` remains null. Exp8494 must
measure the complete append service before any observed speedup claim.

The frozen protocol uses one writer and the unchanged five-input service
head and event tape. A framed record binds framing version, sequence, payload
length and checksum, previous digest, transition inputs, original code/config
hashes and a stable acknowledgment ID. The reference fixture uses a four-byte
big-endian payload length followed by canonical JSON. Recovery checks that
length and all bytes against independent sealed inputs. It replays validated
complete records after the checkpoint and discards only an incomplete final
frame. Interior corruption and conflicting identities fail closed. Complete
unacknowledged transactions may replay; a retry returns the saved identical
response without a second logical transition.

Precreate the journal and sync its file and parent directory. Append a complete
record, handling short writes, and sync the file before acknowledgment.
Publish checkpoints only every32 committed transactions and at final drain.
Every create, rename, unlink, rotation and checkpoint namespace change still
requires directory sync. A checkpoint names its durable prefix before any
reclamation; pinned readers retain their operands. Overflow blocks without
dropping feedback. Preserve MAX_EVENTS256, MAX_BYTES2MiB and journal16MiB.
Exp8493 owns implementation and real process-exit crash qualification; these
reference fixtures do not certify a durable store or physical power loss.

Independent transition fixtures apply the unchanged arithmetic sequentially
and compare full state, replies and stable identities against the sealed
oracle. Ordering and negative recovery fixtures test synchronization before
acknowledgment, durable checkpoint before reclamation, interior length
corruption, complete corruption, conflicting IDs and repaired input hashes.
All24 original localization primitives and24 original service arms authenticate;
the36 reported diagnostics comprise12 instrumented spans and24 service arms.
Repeats, arms and warmups are engineering diagnostics with independent
generalization count0. Both generalization scores remain0.

V728 qualified journal equivalence remains intact. Its complete-service null
is speedup1.0447833367347368, CI95[0.9942346767648512,1.1095895447657544].
V726 exposed-history results are not reopened. V717 H1=-0.00390625 and H2=0
remain closed. This cost model makes no learning-benefit or Rust/Python NFR01
claim. Generator weights stay frozen and nothing is published externally.

Validation collects original node IDs and proves bounded historical shard
unions. It runs scoped serial pytest, genuine Coverage.py unit and child
shards with100 percent newly owned statement coverage, Ruff, strict mypy,
spec tracing, E2E-018 authority and E2E-021 frozen replay. Fresh bounded
children reject missing inputs, deliberate errors and self-consistently
rehashed tampering. Unchanged primary publication, adversarial verification
and strict row validation guard terminal bytes. Exact receipts and final
coverage in the artifact determine readiness; unresolved findings block it.
The conductor owns subsequent ops/status/changelog/traceability reconciliation.
