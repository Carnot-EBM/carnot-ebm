# V743 complete-source comparison

Exp8659 uses the fixed exposed Trivia+ roster: 190 responses in 128 question
clusters. Exp8653 supplies the separately authenticated input gate and original
protocol. Exp8656 supplies the qualified streaming extractor gate. Neither gate
changes the historical Exp8642 disqualification or Exp8645 blocked comparison.

The runner loads no language model or numerical energy head. Historical response
generators remain separate provenance. The numerical head checkpoint is checked
by its own bytes and is not evaluated. No detector is fitted, p2 confidence is
not imputed, and no feature enters the existing nine-input learner.

Each response has three paired cost repeats. The middle repeat reverses the two
arms; response order stays fixed. Every extraction runs in a fresh process forked after module imports. The worker
opens only its public content descriptor. No previous response extraction runs
in the parent, so response caches are not inherited.
Chunk32 is the measurement arm, with chunk1 and chunk128 invariance controls.
The original 128-window cap remains in the capped arm. Streaming retains the
512 KiB source, 64 KiB answer, 16 answer-unit, 30 second, 256 MiB extra-memory
and 32 MiB deduplication limits. Resource abstentions retain their full roster
position. Measurement stops at 3600 seconds and retains pending positions.

Raw response JSON files retain feature hashes, actual extraction time, kernel
logical/physical I/O counters, source/answer bytes, window counts and fresh-process
peak extra memory. Peak extra memory is the process high-water increase over the
post-import, fresh-child baseline; it can be observed zero. Ordered window JSONL files retain
UTF-8 byte offsets, exact-byte deduplication groups and first-occurrence records.
Content-addressed copies preserve primitives. Private release text stays in the
authenticated disk-backed cache outside the repository.

Features are sealed before human votes are joined. All four original fixed
lexical scores are reported separately for original common support and newly
recovered support. Question bootstrap intervals use 10000 draws and seed7418632.
Siblings remain together. Unanimous and disputed annotations remain distinct.
Changed-denominator AUROC cannot establish detector benefit.

The engineering gate requires exact common-domain feature hashes, chunk
invariance, complete ordered window custody, resource bounds, all three paired
repeats and at least one recovered response. Source comparison readiness and
coverage benefit are separate numeric gates. Neither establishes semantic
entailment, calibration, factuality or independent generalization.

The invocation retains exact collected test IDs, separate producer/consumer
coverage, raw child SQLite and JUnit evidence before combining copies. A fresh
bounded replay reconstructs numerical features and window custody from original
private bytes. The unchanged primary publisher, adversarial verifier and strict
row consumer check final bytes before atomic publication. The conductor owns
ops/status, changelog and traceability reconciliation.

An initial CLI-worker invocation was interrupted because repeated module imports
cost about two seconds per extraction while extraction itself took milliseconds.
Its raw files and interruption receipt remain under invocation1792388454922953259.
It is disqualified and supplies no primitives to the fresh forked-worker run.

The next invocation1792388716231834551 retained a complete measurement but was
interrupted before publication. Its declared terminal sidecar was not persisted
and its primary did not embed the actual scanner report. A new failing private
publication assertion exposed that gap. Both issues were corrected before the
final fresh invocation; none of that attempt's primitives were reused.

Measured result, invocation1792389156779020790: all190 streaming responses are
usable. All135 original usable responses retain exact common-domain feature
hashes; all55 previous window-limit abstentions are recovered. Coverage moves
from135/190 (71.05%) to190/190 (100%). The recovered fraction is28.95 percentage
points, with a descriptive question-cluster95% interval of20.72–38.27 percentage
points (10000 draws, seed7418632). Common and recovered support have90 and38
question groups respectively. All128 remain exposed units.

All chunk1/32/128 controls agree, and33422 ordered source-window records pass
byte-offset and exact-byte deduplication checks. All1140 cost rows are retained:
570 capped and570 streaming. Median extraction time is6.234 ms capped and
9.191 ms streaming. The capped arm includes55 abstentions. Maximum observed
times are132.196 ms and195.251 ms respectively. Maximum extra memory is
6766592 bytes capped and8491008 bytes streaming. Kernel counters retain
both logical/physical I/O bytes and named system-call counts. Preloaded imports
are excluded from extraction costs; actual parent/child clocks remain in
receipts. The final measurement phase took128.996734032 seconds.

The four common-support AUROCs are0.6024,0.4894,0.7096 and0.7095, in registered
score order. Recovered-support values are0.4338,0.5122,0.5505 and0.5505. Their
support differs. These descriptive numbers do not establish detector benefit,
and no aggregate changed-denominator AUROC is used for promotion.

All54 scoped test nodes passed:17 newly owned,18 private E2E-018,12 primary
publication and7 private E2E-021 nodes. All401 newly owned executable statements
are covered (100%), with consumer coverage kept separate. Ruff/format, strict
mypy and spec tracing passed. The primary retains the exact validation evidence. Its actual verdict and
readiness gates govern qualification. Final cold replay and unchanged scanner
reports are bound to the published bytes in the declared terminal sidecar and
the primary publisher's hash-specific sidecar. Historical nulls, quarantines and
both interrupted owned attempts remain unchanged.
