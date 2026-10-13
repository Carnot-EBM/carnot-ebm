# V730 durable append qualification — Exp8493

This experiment qualifies a separate stable-file journal against Exp8492's sealed
append protocol and original V728 service tapes. The production snapshot and
journal stores are unchanged. The result belongs to
`exp8493-append-store-qualification`, milestone `2026.10.730`, invoked with
`--date 20261013`.

The writer keeps `events.frames` at the same inode throughout the bounded run.
Each length-prefixed canonical record carries framing version, sequence, previous
digest, payload length and checksum, frozen code/config hashes and acknowledgment
ID. The unchanged five-input transition function computes responses and delayed
feedback. Independently reconstructed snapshot prefixes check their exact bytes.

Initialization syncs the journal and parent namespace before accepting work.
Append handles short writes and file sync precedes acknowledgment. Recovery
verifies every complete record against the independently sealed tape. Complete
unacknowledged records replay once; retries return saved responses. Only an
incomplete final frame is trimmed. Complete corruption, interior length errors,
conflicting IDs and reordered or self-consistently rehashed transitions reject.

A checkpoint is published every 32 transactions and at final drain. The new
checkpoint and prefix pointer are file-synced, replaced and directory-synced
before old checkpoints are unlinked. Reclamation receives another directory
sync. A failed namespace sync remains an outstanding obligation on retry.
Readers hold complete immutable state copies, so reclamation cannot alter an
already pinned view. The bounded journal retains its event data; it does not
rotate or reclaim event segments. Overflow blocks at MAX_EVENTS=256,
MAX_BYTES=2 MiB and the 16 MiB journal ceiling.

Primitive evidence is retained under
`results/raw/experiment_8493_v730_append_store_qualification/`. It includes two
service-cell equivalence fixtures and fourteen actual exit boundaries: before
and after append, file sync, acknowledgment, checkpoint file sync, checkpoint
replacement, directory sync and reclamation. Caller acknowledgment files,
crashed store copies and resumed states are separate operands. Cold replay copies
those disk prefixes into fresh private scratch and reconstructs the state,
responses, release cursor and feedback. It also verifies immutable source and
code references and rejects repaired aggregate and primitive tampering.

The validation plan collects required test nodes before execution, preserves
nonoverlapping historical shard unions, and runs E2E-018 authority and E2E-021
historical replay checks. Genuine Coverage.py unit and subprocess shards include
`_exit`; readiness requires all newly owned statements. Ruff check/format, strict
mypy, spec tracing, cold controls, unchanged adversarial verification, strict row
consistency and bound primary publication remain required. Every child retains
its command, exit, duration and log hashes. Missing external authority blocks;
owned failures disqualify. Optional CUDA and boards are irrelevant to CPU work.

No model is loaded: MODEL_SPECS is empty and current LLM calls are zero. Cached
model outputs remain historical provenance. Oracle fixture qualification uses
`circular_positive`; both generalization scores remain zero. Real process exits
and actual fsync calls do not certify physical power-loss durability. This task
makes no speed claim. The final readiness score and exact validation receipts
are in `results/experiment_8493_v730_append_store_qualification.json`.

V728 qualified journal equivalence remains preserved. Its complete-service
speedup is the null result 1.0447833367, CI95 [0.9942346768, 1.1095895448]. V726
exposed-history results are retained, and V717 H1=-0.00390625 / H2=0 remain closed.
Exp8494 is a future cost measurement, not an observation in this qualification.
The conductor owns subsequent ops/status/changelog/traceability reconciliation.
