# V732 append import qualification (2026-10-14)

Exp8519 qualifies the current append replay reader after its explicit import
repair. The historical strict-mypy failure was at line309: the numeric module
did not explicitly export its imported `append` module. The consumer now imports
`Store` directly from `carnot.verify.append_store_8493`. No arithmetic, durable
barrier, checkpoint rule or benchmark threshold changes.

The original reader source and the reproduced failing strict-mypy receipt are
under `results/raw/experiment_8519_v732_append_import_qualification/historical/`.
The repair diff has its own reference. The result hashes the measurement kernel,
current report reader and validation repair separately.

Cold replay authenticates the original V730 authority and producer hashes,
checks the sealed source snapshots, and reduces all16 crash/equivalence operands
and all24 historical complete-clock arms in a fresh bounded process. Actual
retained store bytes must agree with the independent logical oracle. Missing
inputs, changed primitive bytes and rehashed aggregate tampering reject.
Exp8493 keeps its original 20261013 date and qualified recovery scope. Exp8494
keeps its original 20261013 date, bytes and disqualified status. Its clocks are
historical evidence; this import repair provides no speed observation.

The current runner authenticates the complete activated V732 task and the
Exp8512 methods/service gates before replay. It checks tools, available memory,
disk space and disk-backed private scratch. CUDA and boards are optional. It
loads no model and makes zero current LLM calls. Imported model provenance stays
in `historical_model_provenance`.

Validation collects the original append and E2E-018/021 nodes before running
bounded disjoint serial shards. Historical CLI dates and private roots remain
unchanged. Unit and real CLI child coverage applies only to newly owned code.
The current reader also passes the original strict typing command. Validation
fixtures exercise the original clocks, but produce no new scientific timing
claim. Publication uses the unchanged atomic publisher, adversarial verifier,
strict row lint and actual downstream consumers, with receipts bound to exact
terminal bytes. Required owned failures disqualify; external absence blocks.
Readiness is an engineering fixture result with zero generalization scores.

Run:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=python:. PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/experiments/experiment_8519_v732_append_import_qualification.py --date 20261014
```

The primary is `results/experiment_8519_v732_append_import_qualification.json`.
The conductor owns ops, status and traceability reconciliation. No active
roadmap, conductor, external service or generator weight is changed.

An initial validation attempt placed pytest temporary roots under `results/`.
The repository write guard redirected child fixture writes, so the crash tests
failed to find their acknowledgment files. That owned attempt is retained as
disqualified in raw evidence; its terminated child cannot pass validation. The
corrected runner creates fixtures, collections and coverage data under `/var/tmp`
and copies their evidence into raw storage only after validation. A fresh
qualification uses the unchanged original assertions and isolated roots.

The original store test fixture also requires activated V730 authority. A later
attempt exposed that it was reading the active V732 roadmap. That failed attempt
is retained as disqualified. The historical pytest adapter now uses the shipped
`frozen_inputs` reader with independently pinned V730 design and active snapshots.
It changes only private reads, restores them after the session, and keeps the
original tests, task checks, dates and live roadmap bytes unchanged.
