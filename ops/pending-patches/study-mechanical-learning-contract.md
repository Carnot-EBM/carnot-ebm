# Pending Patch: Study Mechanical Learning Contract

Status: pending
Prepared: 2026-05-06
Apply when: conductor is paused or between tasks, with no active conductor-owned
dirty files in `scripts/research_conductor.py`, `research-roadmap*.yaml`,
`openspec/capabilities/research-harnesses/spec.md`, or the target test files.

## Problem

The conductor planning prompt currently asks the study phase to add promising
research findings before designing the next milestone. That keeps the roadmap
current, but it does not mechanically prove that the study phase changed
dispatch behavior. A study sweep can therefore succeed as prose while producing
no new gates, retirement rules, tests, artifact fields, or rerun prevention.

## Goal

Make a study/planning phase count as useful only when it emits a mechanical
learning contract. The contract must either:

- produce at least one enforceable control-loop change; or
- explicitly justify why no mechanical change was warranted.

This keeps useful source sweeps, but prevents advisory-only research churn from
passing as loop learning.

## Target Files

- `openspec/capabilities/research-harnesses/spec.md`
- `scripts/check_study_mechanical_learning.py`
- `tests/python/test_check_study_mechanical_learning.py`
- `scripts/research_conductor.py`
- `_bmad/traceability.md`
- `ops/changelog.md`
- `ops/status.md`

## Spec Delta

Add `REQ-HARNESS-013: Study Mechanical Learning Contract`.

Required behavior:

- Every planner-produced milestone roadmap must include a top-level
  `mechanical_learning` mapping.
- `mechanical_learning.operator_intervention_required` must be a boolean.
- If the study phase adds sources, every source must be mapped to at least one
  planned task or explicitly deferred with a reason.
- At least one of these fields must be non-empty unless
  `no_mechanical_change_justification` is present:
  `new_gate_checks`, `new_retirement_rules`, `dispatch_changes`,
  `tests_or_checks_added`, `new_artifact_fields`, `reruns_prevented`.
- Referenced task ids must exist in the same roadmap.
- A roadmap with only references/prose and no mechanical delta must fail
  planner-output validation before activation.

Add scenarios:

- `SCENARIO-HARNESS-008`: advisory-only study output is rejected.
- `SCENARIO-HARNESS-009`: a source mapped to a gate and a task passes.
- `SCENARIO-HARNESS-010`: no-change study output passes only with explicit
  justification and operator-intervention field.

## YAML Contract

```yaml
mechanical_learning:
  study_sources_added:
    - source: arXiv:2603.03305
      finding: draft-conditioned constrained decoding fits repair hints
      mapped_to_tasks:
        - exp1428-dccd-schema-constrained-repair-v2
  new_gate_checks:
    - task_id: exp1431-fullscale-pipeline-v4-micro-gated
      rule: exp1431 requires exp1428.repaired_case_success_rate > 0.0
  new_retirement_rules:
    - retire exact exp1419-style 200-case rerun without nonzero repair evidence
  reruns_prevented:
    - exp1419-fullscale-pipeline-v3-repair-executor
  dispatch_changes:
    - exp1431 changed from 200-case scale run to micro-gated 50-case run
  tests_or_checks_added:
    - scripts/check_study_mechanical_learning.py
  new_artifact_fields:
    - mechanical_learning
  operator_intervention_required: false
```

Valid no-change form:

```yaml
mechanical_learning:
  study_sources_added: []
  new_gate_checks: []
  new_retirement_rules: []
  dispatch_changes: []
  tests_or_checks_added: []
  new_artifact_fields: []
  reruns_prevented: []
  no_mechanical_change_justification: >-
    Current milestone is a pure carry-forward execution milestone; study found
    no new source or failure pattern that should alter dispatch.
  operator_intervention_required: false
```

## Validator Shape

Add `scripts/check_study_mechanical_learning.py`.

CLI:

```bash
python scripts/check_study_mechanical_learning.py \
  --roadmap research-roadmap-next.yaml \
  --doc openspec/change-proposals/research-roadmap-vNEXT.md
```

Validation rules:

- Load YAML with `yaml.safe_load`.
- Require top-level mapping.
- Require `tasks` list and collect task ids.
- Require top-level `mechanical_learning` mapping.
- Require `operator_intervention_required` to be a boolean.
- Require at least one concrete mechanical field to be non-empty, unless
  `no_mechanical_change_justification` is a non-empty string.
- For each `study_sources_added` item, require `source` and `finding`, and
  either:
  - non-empty `mapped_to_tasks` whose ids all exist in `tasks`; or
  - non-empty `deferred_reason`.
- For each `new_gate_checks` dict with `task_id`, require the id to exist.
- Print a compact JSON summary with `ok`, `errors`, and
  `mechanical_change_count`.
- Exit 0 on pass, 1 on fail.

## Conductor Integration

Patch `_plan_next_milestone()` in `scripts/research_conductor.py` after the
planner creates `research-roadmap-next.yaml` and before it commits planned
roadmap files:

```python
study_contract = run_cmd(
    [
        sys.executable,
        "scripts/check_study_mechanical_learning.py",
        "--roadmap",
        str(NEXT_ROADMAP_FILE),
        "--doc",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
    ],
    timeout=120,
    check=False,
)
if study_contract.returncode != 0:
    logger.warning("Planner output failed study mechanical-learning contract")
    log_step("Plan next milestone", "FAIL", "Study mechanical-learning contract failed")
    return False
```

Also update the planner prompt to require:

- a `## Mechanical Learning Contract` section in the roadmap doc; and
- the top-level YAML `mechanical_learning` mapping.

## Tests First

Add `tests/python/test_check_study_mechanical_learning.py` with:

- `# REQ-HARNESS-013`: fails when the roadmap has sources but no mechanical
  fields or no no-change justification.
- `# SCENARIO-HARNESS-009`: passes when a source maps to an existing task and a
  gate check exists.
- `# SCENARIO-HARNESS-010`: passes when no mechanical change exists but the
  no-change justification and `operator_intervention_required` are present.
- Fails when `mapped_to_tasks` references an unknown task id.
- Fails when `operator_intervention_required` is missing or non-boolean.

Targeted verification:

```bash
pytest tests/python/test_check_study_mechanical_learning.py -q --no-cov -p no:cacheprovider -n 0
ruff check scripts/check_study_mechanical_learning.py tests/python/test_check_study_mechanical_learning.py
ruff format --check scripts/check_study_mechanical_learning.py tests/python/test_check_study_mechanical_learning.py
python scripts/check_spec_coverage.py tests/python/test_check_study_mechanical_learning.py
```

Full verification should wait until the active conductor task has landed, then
run the relevant harness/conductor tests and the repository reconciliation
checks from `CODEX.md`.

## Acceptance

The patch is complete only when:

- advisory-only study output fails validation;
- mechanically mapped study output passes validation;
- no-change study output passes only with explicit justification;
- `_plan_next_milestone()` refuses to activate or commit invalid planner output;
- specs, traceability, changelog, and status mention `REQ-HARNESS-013`; and
- no current conductor-owned experiment files are overwritten.
