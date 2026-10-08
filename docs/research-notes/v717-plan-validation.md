# V717 planning validation — 2026-10-08

This validates a staged research plan. No experiment or activation was run.

- Milestone `2026.10.717`: exactly 14 tasks, Exp8304–Exp8317, four phases.
- Design: `openspec/change-proposals/research-roadmap-vNEXT.md`.
- Execution: `research-roadmap-next.yaml`.
- All complete JSON task objects equal the YAML, including every prompt;
  the visible table agrees independently.
- Canonical tasks SHA-256:
  `2f67c35e346dc59dfb333560706f79981378f516ba8a8e056a001b3968d2c9fd`.
- All 12 gates reference earlier producers and identically spelled REQUIRED
  ARTIFACT FIELDS. All 25 prior-failure entries contain all four required fields
  with `retire_if_same_verdict: true`.
- All 206 existing-code path references resolve. Every task has a deliverable,
  concrete runner command, required prompt headings/placeholders, numbered
  progress and bounded-writing instructions, per-unit rows and the exact
  no-push/no-conductor-edit ending.
- Exp8313 is the only live LLM task: `unsloth/Qwen3.8-27B-GGUF`, Q4_K_M,
  `model_bounded_generation`. Other tasks explicitly separate cached historical
  inference from current no-model work.

## Passed checks

1. `scripts.roadmap_schema.Roadmap.model_validate` on the emitted YAML.
2. `carnot.reporting.roadmap_contract.compare_contract`, explicitly using
   milestone `2026.10.717`, first_id=8304, count=14; independent full-object
   equality and SHA-256 recomputation. Both comparison inputs use staged bytes;
   this does not check or imply activation. Mutating a title and deleting a task
   each fail the existing reader as required by SCENARIO-REPORT-V717-PLAN.
3. `scripts/audit_roadmap_gates.py research-roadmap-next.yaml`: all checks pass,
   12 gate references, 14 task-level failure-lineage checks, zero backend/model
   coherence gaps and zero Luna tasks.
4. `scripts/exclusion_manifest_lint.py research-roadmap-next.yaml`: clean.
5. `scripts/harness_fit_lint.py research-roadmap-next.yaml`: no risks.
6. `scripts/arc_levelup_guarantee_lint.py research-roadmap-next.yaml`: existing
   public-solve requirement retired; two generalization-floor matches found.
   Exp8314 is the explicit current supervisor-outcome/coverage task.
7. `scripts/overdue_priority_lint.py --strict`: exit 0. This CLI reads the staged
   path itself; an initial unsupported positional argument exited 2, then the
   documented invocation passed without changing any validator.
8. Scoped Ruff on roadmap_schema, audit_roadmap_gates, exclusion_manifest_lint,
   harness_fit_lint, arc_levelup_guarantee_lint and the existing roadmap_contract
   reader: all checks pass.
9. `scripts/check_spec_coverage.py --files` on the five test files below: pass.
10. `git diff --check`: pass.

Focused unit and applicable private E2E-018 authority-lifecycle checks:

```bash
.venv/bin/python -m pytest -n 0 -o addopts= --no-cov -q \
  tests/python/test_roadmap_schema.py \
  tests/python/test_audit_roadmap_gates.py \
  tests/python/test_exclusion_manifest_lint.py \
  tests/python/test_arc_levelup_guarantee_lint.py \
  tests/python/test_experiment_7891_v685_authority_lifecycle.py
```

Result: **105 passed in 22.64s**. The private cases exercise authority matching,
real CLI lifecycle and negative/mutated replay. Historical publishing commands
were not rerun against current state. Live inference, board and experiment-specific
scientific E2Es are future execution work. No implementation/test file was added.

## Scientific feasibility review

A read-only adversarial reviewer examined the draft and revised plan. Corrections
include a retention floor compatible with the fixed 23 usable sources (9/14 by
label), separate transport/feature denominators, a statically learned holistic
slope to handle saturated logits, exact-budget optimizer/action controls, and
sealing all static/learning/retention predictions before evaluator target access.
The 104 labeled fit sources contain 62 distinct local feature vectors, making
32 distinct RBF centers feasible. These censuses inspect existing evidence;
they are not new experiment results. The final review found no remaining blocking
scientific infeasibility. Two minor wording inconsistencies were then corrected
and full contract equality/digest checks rerun successfully.

The cached cohort is exposed development. Descriptive intervals and the small
retention panel cannot establish independent generalization. An update-budget
failure cannot be promoted into a null about all sentence representations.

## Repository-health limitation

Global `scripts/check_spec_coverage.py` exits 1 with **1,142 tests missing spec
traceability**, matching the existing V716 validation record. Scoped planning
checks pass; this is not a global repository verification pass. The reconciliation
script's final outcome is recorded below after completion.

## Preserved authorities

| File | SHA-256 before and after planning |
|---|---|
| `research-roadmap.yaml` | `b9f6a1ea738c0b55d3bec332169c3d9608eacc316602e535cb74d0f1305a31b7` |
| `scripts/research_conductor.py` | `4d9dffebf1249bd858f9af7747182509611819ca816d460fcfacb68895f7a556` |
| `openspec/change-proposals/v713-evidence-intervention-protocol.json` | `f4c06e17a9f8eb72f1be80b265ab7e3507cbc5f6f3b918df6421fe49dae02018` |

V716's previous design is preserved byte-for-byte at
`openspec/change-proposals/research-roadmap-v716-preserved-20261008.md`, SHA-256
`4927d920746871d2c73b3b11f50abc52169ca76247e115c560f228143c9c729e`.
The dated references were recorded before task design. No experiment, model,
board run, external publication or push was performed by this planning work.
