# V716 planning validation — 2026-10-08

This validates a staged plan, not experimental outcomes or activation.

- Milestone: `2026.10.716`.
- Exactly 14 ordered tasks: Exp8290–Exp8303, four phases, 15 structured gates.
- Design: `openspec/change-proposals/research-roadmap-vNEXT.md`.
- Execution: `research-roadmap-next.yaml`.
- Complete JSON task objects equal the YAML objects, including every prompt.
- Canonical tasks SHA-256:
  `1631010541a7298fa4908b9fcbe21576b4b333b81248b52c8643b0aaf44b71ed`.
- Four model tasks: Exp8292, Exp8293, Exp8294, Exp8296. Every one declares
  `unsloth/Qwen3.8-27B-GGUF`, Q4_K_M and `model_bounded_generation`.
- All gate fields exist in the upstream REQUIRED ARTIFACT FIELDS blocks;
  upstreams occur earlier in this same roadmap. Every failure-lineage entry
  contains all four required fields and `retire_if_same_verdict: true`.
- All prompts contain the requested headings, placeholders, concrete primary
  and runner paths, numbered progress/bounded-writing requirements and exact
  no-push/no-conductor-edit ending. All comparative tasks request per-unit rows.
- Existing-code read lists contain no missing historical paths and no
  self/forward producer references. Future execution-contract creation is
  explicit in Exp8290.

## Passed checks

1. `scripts.roadmap_schema.Roadmap.model_validate` on the emitted YAML.
2. Existing `carnot.reporting.roadmap_contract.compare_contract` with explicit
   milestone, first_id=8290 and count=14; full-object equality and independent
   digest recomputation. Both supplied comparison slots used staged bytes:
   this checks staged agreement and deliberately does not claim activation.
   Mutating a title or removing a task was rejected by the existing reader.
3. `scripts/audit_roadmap_gates.py research-roadmap-next.yaml`: all checks pass,
   15 gate-field references, 14 failure-lineage checks, no model/backend gaps.
4. `scripts/exclusion_manifest_lint.py research-roadmap-next.yaml`: clean.
5. `scripts/harness_fit_lint.py research-roadmap-next.yaml`: no risks.
6. `scripts/arc_levelup_guarantee_lint.py research-roadmap-next.yaml`: public
   solve requirement retired by existing operator policy; two generalization
   floor matches detected. Exp8300 is the explicit outcome-audit task.
7. `scripts/overdue_priority_lint.py`: exit 0.
8. Scoped Ruff on roadmap schema, gate/exclusion/execution-fit/ARC validators
   and the existing roadmap contract reader: pass.
9. Scoped `scripts/check_spec_coverage.py --files` on the five test files below:
   pass. No implementation code or tests were added by planning.
10. `git diff --check`: pass.

Unit tests and applicable private E2E-018 authority-lifecycle checks:

```bash
.venv/bin/python -m pytest -n 0 -o addopts= --no-cov -q \
  tests/python/test_roadmap_schema.py \
  tests/python/test_audit_roadmap_gates.py \
  tests/python/test_exclusion_manifest_lint.py \
  tests/python/test_arc_levelup_guarantee_lint.py \
  tests/python/test_experiment_7891_v685_authority_lifecycle.py
```

Result: **105 passed in 19.03s**. The private CLI checks cover matching
activation authority and negative/mutated replay. The historical publishing
command was not rerun against the current staged milestone. Experiment-specific
live inference, board and scientific E2Es remain future execution work.

## Repository-health limitation

`bash scripts/validate-reconciliation.sh` exited 1 with one issue: the global
spec-reference audit reports **1,142 tests missing traceability**. Documentation
freshness passed. The same count is recorded in existing October 1 changelog
entries for Exp7975/Exp7992; no implementation/test files were changed by this
plan. This is not a global verification pass. Scoped planning checks passed.

## Preserved authorities

| File | SHA-256 before and after planning |
|---|---|
| `research-roadmap.yaml` | `8f249ac539ee3463285baa7b53070c1db603b79243aa3564dd6a8c3e49beb1f1` |
| `scripts/research_conductor.py` | `4d9dffebf1249bd858f9af7747182509611819ca816d460fcfacb68895f7a556` |
| `openspec/change-proposals/v713-evidence-intervention-protocol.json` | `f4c06e17a9f8eb72f1be80b265ab7e3507cbc5f6f3b918df6421fe49dae02018` |

V715's previous design is preserved verbatim at
`openspec/change-proposals/research-roadmap-v715-preserved-20261008.md`, SHA-256
`fb6cb87fa2529a34044907f71046bd1b3c36b1c6d2cdae5b33e4fb7da6754d74`.
No experiment, model or board run was activated. No push or external
publication occurred. The literature scan and its access limits were recorded
in `research-references.md` before the task design.
