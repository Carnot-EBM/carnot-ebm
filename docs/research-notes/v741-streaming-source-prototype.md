# V741 streaming source prototype

Exp8629 qualifies an opt-in lexical extractor on deterministic fixtures. The
result and retained commands are in
`results/experiment_8629_v741_streaming_source_prototype.json`. No language model,
energy head, fit, threshold selection or natural release census runs here.

The extractor emits sentence windows followed by consecutive three-sentence
windows. Full UTF-8 byte intervals include whitespace. SQLite uses exact stripped
byte keys for deduplication; hashes never decide equality. Each original window,
including duplicate occurrences, has a disk record with its interval, group and
all contributing answer-unit IDs. Chunk sizes1,32 and128 control batches only.

The four lexical measurements match the old132-feature pair computation. Four
scalars per unique window remain in SQLite. Ordered cursor reductions use the
same Python `sum()` as the old extractor. This matters because replacing that
operation with incremental floating-point addition changed low-order bits during
the initial failing parity tests. Eight final values and their feature hashes
must agree exactly. The implementation retains no full pair-feature tensor.

The registered bounds are512KiB source,64KiB answer,16 answer sentences,30 seconds
per response and256MiB additional working memory. The prototype also caps its
deduplication database and window log at32MiB. A conservative working reservation,
a1MiB SQLite cache, file-backed temporary storage, disk checks and elapsed-time
checks constrain the new path. Traced Python allocation peaks and real latency
are descriptive costs of these fixtures, with instrumentation overhead. They are
not a deployment memory benchmark. Resource exhaustion returns an explicit
abstention and marks its window log incomplete.

Qualification compares eager and streaming enumeration below and above128
windows. Above two sentences, the original window total is even. The boundary
fixtures therefore include128 and130 original windows and129 unique windows
formed by one duplicate. Tests also cover multibyte text, punctuation, empty
inputs, whitespace, malformed UTF-8, byte caps, answer caps, hash collisions and
resource exits. Fresh processes reject missing chunks, changed offsets, changed
features, label-bearing keys and repaired derived manifests against original
code-defined fixtures. Private E2E-018/021 and both publication readers remain
part of validation. Historical readers and `source_alignment.prepare` are unchanged.

The readiness score qualifies implementation behavior only. The verdict remains
`null` for scientific benefit. Complete lexical windows are not entailed claims,
calibrated confidence or independent generalization. Exp8632 owns natural coverage
and exact comparison on the exposed190-response/128-question roster. This work
does not change V740's55 source-window abstentions, lost-shard disqualification,
blocked learning, unmet NFR-01, or the horizon observer's limited qualification.

Operations, status and traceability reconciliation belong to the conductor's
separate step for this invocation.

The initial invocation was disqualified because its new retention check ran
after Coverage.py's report command. In the installed version, that command
implicitly combines with `keep=False`, deleting original shards even after an
explicit `combine --keep`. The original disqualification and134 evidence files
are preserved under `attempts/initial_disqualified/`. The corrected invocation
archives original shards before the first combine; a regression test deletes
the scratch originals after combine and requires the archive to survive.
