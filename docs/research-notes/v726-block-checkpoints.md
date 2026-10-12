# Bounded feedback checkpoints

This task qualifies an isolated nine-input, 109-coefficient checkpoint. It uses
synthetic labels to test recovery. It makes no claim about factuality, current
Qwen skill, human-label benefit or hidden-game generalization.

Each block contains 20 questions and three sibling answers per question. FIFO
or uniform admission selects 8 or 16 clusters before labels become visible.
The learner stores at most C clusters and 3*C answer feature records. Audit
files remain separate from prediction and update arguments.

The first transaction seals every sibling prediction, selected cluster and
both random states. The second transaction applies admitted feedback in block
order. It commits all updates with the next cursor. Intermediate updates remain
tentative. An updated checkpoint advances without applying those labels again.
Stable acknowledgment identities survive a crash after commit and before delivery.

Immutable snapshots contain the head, decay scale, scaler/temperature seal,
prediction identities and hashes, cursor, admission, pending records, applied
feedback identities and both random states. Pointer publication orders temporary
write, file fsync, atomic replace, directory fsync and acknowledgment. Readers
hold one complete immutable version. Repeated requests preserve saved effects.

Qualification uses five blocks, both capacities and both admission policies.
Real children exit at four publication boundaries in each transaction and after
block 2. Complete resumed states include future admissions and predictions.
A shuffled-feedback control consumes the second random stream and checks its
exact restart. Separate frozen controls preserve all coefficient bytes and make
zero parameter updates. Independent SciPy basis arithmetic checks a nonzero
update at tolerance 1e-12. Rehashed coefficient, cursor, release and random-state
changes must fail replay.

Private E2E-018/020/021 retain original assertions. Genuine Coverage.py unit
and child shards measure only newly owned statements. Ruff, strict mypy, spec
tracing, cold replay and unchanged publication consumers qualify terminal bytes.
Unknown or unsupported findings prevent readiness. The result artifact records
the actual verdict and retains all findings. Readiness never implies science.

Run `PYTHONPATH=python:. PYTHONUNBUFFERED=1 .venv/bin/python -u
scripts/experiments/experiment_8433_v726_block_checkpoint_qualification.py
--date 20261012` from the repository. Primitive rows and exact child receipts
remain under the task's raw invocation. Historical protocols remain unchanged.
The conductor owns ops and BMAD reconciliation after this task exits.
