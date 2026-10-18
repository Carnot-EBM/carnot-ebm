# V741 installed compatibility — Exp8634

This study qualifies deployment compatibility. It does not establish learning,
independent calibration, generalization, a speed gain or NFR-01 compliance.
It loads no language model. `MODEL_SPECS` is empty and language-model calls are zero.

The input is the published Exp8633 wheel. Its primary artifact, bound publication
sidecar and wheel hash are authenticated before installation. Installation uses
an offline private environment in `/var/tmp`. Worker processes run outside the
checkout with `PYTHONPATH` removed. Module files must resolve inside that environment.
The native extension is copied into that environment and its actual loaded path
and hash must match the immutable configuration.

The numerical test keeps the original fixed 109-parameter spline head. It compares
339 original/control requests and six threshold requests in each arithmetic arm.
The private worker provides the execution baseline. Cold replay also recomputes
arithmetic. Numeric tolerance remains `1e-12 * (1 + abs(reference))`. Native fallback
flags keep their original arm-specific values. Cached labels and question metadata
do not enter the worker. No research head is fitted or changed.

Public checks use existing contracts. They cover explicit imports, import-star,
repeated lazy access, absent optional JAX and propagated model import errors.
The Python code verifier, CLI verification command and MCP health/verification
tools are called through their existing modules. The existing HTTP server exposes
`POST /v1/score-candidates`; it has no resident-inference or GET health route.
Its smoke check starts the real server and tests valid and invalid POSTs.
That existing API fits a small detector on 80 synthetic fixture examples.
This fixture behavior gives no calibration or scientific-benefit credit.

Worker controls retain the original rejection families. Additional checks cover
invalid IDs, unknown operations, oversized frames and fragmented delivery. Each
case records actual responses, recovery, exit, cleanup, command and log hash.
Intentional worker death is a negative control. Unexpected forced cleanup fails
qualification.

Validation preserves the historical 122-node union. Historical authority tests
use private copies with original scheduling bytes. Assertions and thresholds stay
unchanged. Current tests add installed public-surface checks. Tests run in bounded,
disjoint children with `-n 0 -o addopts= --no-cov`. Real child coverage files remain
available before `coverage combine --keep`. New statements and affected initializer
and worker statements require 100 percent coverage. Ruff, strict mypy, spec tracing,
private E2E-018/021 and publication readers are required.

The primary result is `results/experiment_8634_v741_installed_compatibility.json`.
Its invocation directory retains primitive evidence, exact nodes, test receipts,
coverage shards and the terminal validation report. An external authority hash
binds cold replay. Missing inputs and rehashed aggregate tampering must fail.
Publication checks final bytes with `primary_publication`, adversarial verification
and strict verdict-row lint before atomic replacement.

Historical Exp8609 methods blockage remains historical evidence. Exp8616 shard
loss and all earlier nulls and quarantines remain unchanged. Exp8635 and other
same-roadmap outputs are future dependencies. This branch does not consume them.
The conductor owns status, changelog and traceability reconciliation.

The first historical qualification child correctly rejected the changed protocol
path. Exp8608 binds its earlier protocol hash. The exact bytes were recovered from
commit `ae7e7115c6` and authenticated against that binding. A private mount namespace
restored those bytes for historical children. It exposed `/dev` and `/proc` so Python
could start. This changed no checkout source file. All 27 affected historical tests then
passed. Their old protocol statements were omitted from current-code coverage.
Failed setup and source-binding attempts remain in the invocation's raw evidence.
Successful test receipts are reused only while their test-file hashes still match.
