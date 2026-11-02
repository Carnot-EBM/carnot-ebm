# V748 collection contract — 2026-11-02

Exp8723 repairs collection ownership and binds administrative execution
eligibility for Exp8723–Exp8736. The primary is
`results/experiment_8723_v748_collection_contract.json`; invocation primitives,
completed child evidence and terminal receipts live below
`results/raw/experiment_8723_v748_collection_contract/`.

The original collector created `private/0` before discovering a missing suite.
The retained Exp8709 child log records FileExistsError at that path. New tests
first reproduced sentinel collision, partial writes, and acceptance of malformed
or foreign nodes. Those failing runs are retained in `development/`.

The shared collector now validates the entire inventory before creating child
scratch or coverage configuration. Valid calls use distinct invocation-owned
directories. Explicit collisions reject without overwriting sentinel bytes.
Repeated incomplete calls retain `missing_collection_nodes`; no failed call
can consume old coverage. Collection logs remain evidence, rather than child
execution state. Tests also invoke the original full-path assertions unchanged
with V748 bindings, including their deliberate failed-coverage control.

The V748 version binding reuses the existing bounded runner and report schema
in separate module namespaces. Complete objects include prompts, routing,
deliverables, gates and scientific method instructions. The visible table and
canonical digest are separate checks. The canonical full-task SHA-256 is
`88a6e9a1c4822285e86c9b089047d4383f6e34d63430bdaff72c33beb979a07a`.
Two new branches intentionally have no prior-failure field. That exact absence
is bound by the digest; no historical failure is invented. Historical contracts
retain their original prior requirements. The original pinned roadmap helper
and publication validators remain byte-identical.

Retained Exp8695 authority is authenticated through its original pinned primary,
sealed operands, exact commands, source identities, 77 nodes and completed
SQLite/JUnit evidence. Its historical producers are not executed. Original V744
objectives and the 39+25 learning partition remain separate method operands.
Exp8724 owns current learning-core execution; independent readers, verifier and
native branches require only their own operands. A skip is scheduling evidence.

This invocation loads no model and performs no generation. Contract positives
are constructed administrative controls, so they are circular_positive and do
not establish independent generalization, useful learning, or device speed.
Preserve Exp8712's constructed lexical-collision boundary, Exp8715's unequal
probe costs, Exp8717's 1.1382x batching and 1.0054x same-batch ratios, and
Exp8719's exact signed32/24 parity without a speed claim. CUDA repair is absent;
no new ARC outcomes or physical acceleration are established.

The primary retains scoped test and private E2E-018/021 receipts, new statement
coverage, exact child exits/node IDs, source/configuration hashes, SQLite,
JUnit, current clocks, cold replay, adversarial verification and strict row
checking. Its terminal sidecar identifies the validated publication bytes.
Ops status, changelog and BMAD traceability reconciliation are conductor-owned.

Activation repair: the conductor escapes literal JSON braces in the Qwen task
prompt for Python formatting. The authority reader accepts that exact encoding
in staged or active YAML and compares the complete task with the canonical
design object. The pinned digest and raw file hashes remain authoritative.
Escaped placeholders, extra escaping, altered prompt text and other task-field
changes reject. Regression tests preserve source bytes across repeated checks
and run the original V747 rejection controls under the current V748 bindings.

Repair validation: 171 distinct scoped tests pass, including the final 43-test
focused run, private E2E-018/021 and ten fix-gate controls. Unit/subprocess
coverage reaches 373/373 affected statements with zero exclusions. Scoped Ruff,
strict mypy and spec tracing pass. Receipts are in
`/var/tmp/carnot-fix-8723-v8b24069/`; the full repository suite was not run.
Global reconciliation retains the existing 1,142 missing spec references and
stale architecture date. The separate repair records in ops and BMAD document
this fix without replacing conductor-owned scientific reconciliation.
