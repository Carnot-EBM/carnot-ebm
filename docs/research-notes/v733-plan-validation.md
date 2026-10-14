# V733 planning validation — 2026-10-14

This note validates the staged plan. It does not activate or execute experiments.
The design and YAML contain14 complete tasks, Exp8526–Exp8539, in four phases.

## Contract and activation checks

- Pydantic `Roadmap.model_validate`: pass.
- Existing `roadmap_contract.parse_design` and `compare_contract`: pass using
  the staged queue and an independent private activation copy.
- Full task dictionary equality and canonical digest: pass.
- Twelve rejected mutations: count,order,ID,title,phase,deliverable,model,
  substrate,gate,prompt,retirement flag and missing end marker.
- Fourteen structured gate conditions: producers precede consumers and declare
  exact fields in REQUIRED ARTIFACT FIELDS.
- Twenty-one prior-failure entries: exact artifact-backed verdicts and all four
  required fields,including `retire_if_same_verdict: true`.
-217 existing read-first path checks: pass. Future deliverables are identified as
  future task dependencies,not existing files.
- Every prompt contains numbered progress,loop-heartbeat and bounded-write steps,
  explicit substrate,principle-annotated fields and the required command footer.
- Exclusion-manifest lint,harness-fit lint,overdue-priority lint: pass.
- `audit_roadmap_gates.py`:14 tasks,14 gate checks,zero failures,no Luna tasks.
- ARC generalization-floor lint: pass. Its lexical count includes generic context;
  manual review identifies the two substantive ARC tasks,Exp8536/8537.

Canonical task digest:
`2ef10dc73811b0b01d16722133a0ba5b5d36ee703bbc72a521f513396582ade3`.

## Existing unit and private end-to-end tests

```bash
PYTHONPATH=python:. .venv/bin/python -m pytest -n 0 -o addopts= --no-cov -q \
  tests/python/test_roadmap_schema.py \
  tests/python/test_conductor_gates.py \
  tests/python/test_audit_roadmap_gates.py
```

75 passed in8.95 seconds.

```bash
PYTHONPATH=python:. .venv/bin/python -m pytest -n 0 -o addopts= --no-cov -q \
  tests/python/test_experiment_7891_v685_authority_lifecycle.py \
  tests/python/test_restricted_decision_audit_8210.py
```

25 passed in52.75 seconds,with one existing `ast.Num` deprecation warning.
These private CLI fixtures cover applicable E2E-018 and E2E-021 paths,including
valid publication and rejected mutated evidence. Historical production CLIs
were not rerun against mutable current authority. No source code changed,so no
new implementation coverage or numerical/hardware E2E result is claimed.

## Lint and repository limits

Scoped Ruff check passed for the three schema/gate scripts and five test files
above. Scoped spec-reference checks passed for all five test files.
Ruff format check reports one unchanged file:
`scripts/audit_roadmap_gates.py`. Seven other checked files are formatted.
Repository-wide spec tracing reports1,142 existing missing requirement references.
The plan does not change these scripts or tests or claim repository-wide health.

`git diff --check` passes on owned tracked changes. The active
`research-roadmap.yaml`, `scripts/research_conductor.py` and
`ops/exclusion_manifest.yaml` match their task-start SHA256 hashes.
The preserved V732 design has SHA256
`1dc51cf5035a2b7635362d47f17db4ee5d5e34a777a088d24f6e7286d395f5aa`.

Machine contract results and global reconciliation logs are under
`/var/tmp/carnot-v733-plan-validation/`. Scratch planning generators and mutation
checks were written outside the repository root.
