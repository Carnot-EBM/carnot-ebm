# Canonical Eval Metrics + Self-Heal Production-Bug Detector

**Status:** Draft change proposal. **REQUESTED FOR MILESTONE 2026.04.80
(MANDATORY)** — pinned in `ops/known-issues.md → MANDATORY-NEXT-MILESTONE
PRIORITIES`.
**Origin:** 2026-04-28 inverted-AUROC discovery — `_roc_auc_score` in
`exp995` and `exp1003` returned `1 − AUROC` for ~24h before being caught.
The conductor self-heal Sonnet *cycled trying to fix the test rather than
the production code* and would have masked the bug indefinitely. Ad-hoc
fixes shipped 2026-04-28: canonical `python/carnot/eval/metrics.py`
(15 tests + 50-trial sklearn cross-validation), AUROC fixes inlined into
both experiment scripts, audit script
`scripts/audit_metric_provenance.py`, commit watchdog
`scripts/conductor_commit_watchdog.sh`. This proposal scopes the *follow-on
systemic work* — full migration of existing per-experiment metric helpers,
the production-bug detector for self-heal, and the metrics-provenance
discipline.

**Target milestone:** **2026.04.80** (mandatory pickup per the .79 retro
or the known-issues.md mandatory section).

**Priority:** **High.** The 2026-04-28 audit found 2 affected files; the
broader pattern (158 files with their own `def *auc/roc/auroc`) means N
more bugs of this shape are likely waiting. The conductor self-heal
failure mode (mask-production-by-fixing-test) is also ongoing — every
bugged test that the conductor "fixes" before a human notices is a
polluted milestone.

**Depends on:** nothing. The canonical metrics module already ships
(2026-04-28). This proposal scopes the migration + self-heal upgrades.

## Summary

Three layered fixes:

1. **Canonical metrics module migration.** `python/carnot/eval/metrics.py`
   ships canonical `auroc`, `precision_recall`, `f1_score`. Migrate all
   per-experiment metric helpers in `scripts/experiment_*.py` and
   `python/carnot/**/*.py` to import from it. Forbid future re-implementation
   in CLAUDE.md.
2. **Conductor self-heal production-bug detector.** Cross-checks failing
   tests against reference implementations (sklearn for ML metrics, scipy
   for stats) before the heal-Sonnet modifies production code. Adds the
   hard rule "self-heal must never modify tests, only production code."
3. **Metrics provenance tagging.** Every deliverable JSON carries
   `metrics_provenance: {"auroc": "carnot.eval.metrics.auroc:v1.0", …}`.
   The `experiment_template.py:build_result()` flushes it
   automatically. The audit script `scripts/audit_metric_provenance.py`
   walks `results/` and lists deliverables tagged with a known-buggy
   version when a bug is found.

## What this proposal IS NOT

- **Not a sklearn-everywhere migration.** The bare-venv runtime
  intentionally minimizes dependencies. sklearn stays a *test-time-only*
  reference; production code uses `carnot.eval.metrics`.
- **Not a refactor of existing experiment scripts beyond the import swap.**
  Per-experiment scripts can keep their experiment-specific logic. Only
  the metric *computation* moves.
- **Not a forced-Opus self-heal upgrade.** The bug-detector approach
  fixes the same problem at lower cost. Tiered Sonnet→Opus escalation
  is the right cost trade-off.

## Proposed experiments

### Exp A — Migrate all per-experiment metric helpers

**Deliverable:** edits to all `scripts/experiment_*.py` and
`python/carnot/**/*.py` that define their own `_roc_auc_score`,
`_f1_score`, `_precision_recall` etc. + `results/experiment_<N>_canonical_metrics_migration.json`.

**What it does:**

1. Audit list: `grep -rln "def _roc_auc_score\|def _auroc\|def _f1\|def _precision_recall" scripts/ python/carnot/`.
2. For each match, swap the per-file definition for `from carnot.eval.metrics import auroc, f1_score, precision_recall` and replace call sites.
3. Run the full pytest suite. Verify each migrated file's existing tests still pass.
4. List every helper found bugged by the migration in the artifact;
   each becomes a candidate for retrospective verdict re-evaluation.

**Acceptance:** all per-experiment metric helpers in `scripts/`
removed or re-pointed to `carnot.eval.metrics`. Full suite passes.

### Exp B — Conductor self-heal production-bug detector

**Deliverable:** edits to `scripts/research_conductor.py` (the self-heal
call site) + new `python/carnot/conductor/test_failure_classifier.py` +
`tests/python/test_self_heal_production_bug_detector.py`.

**What it does:**

1. **Failure classifier.** Parses pytest's failure output, extracts
   `(file, test_name, comparison_op, actual, expected, inputs)`. For
   common metrics in `carnot.eval.metrics`, looks up the reference
   implementation (sklearn) and computes the reference value on the
   same inputs. Three-way classification:
   - `reference_value == expected` → production code is buggy.
     Self-heal Sonnet is told "the test's expected value matches the
     sklearn reference; the production code under test is the bug."
   - `reference_value == actual` → test is buggy. Self-heal Sonnet is
     told "the production code matches the sklearn reference; the test
     is wrong." (And the conductor outer logic *declines* to allow
     auto-modification of tests — see hard rule below.)
   - Indeterminate → fall through to existing self-heal flow.
2. **Hard rule: self-heal never modifies test files.** Any heal diff
   that touches `tests/python/*.py` is rejected at the conductor level.
3. **Tiered self-heal:** first attempt 30-turn Sonnet with augmented
   prompt; second attempt 50-turn Opus on indeterminate cases only.

**Acceptance:** synthetic test scenarios cover (a) production-buggy
classifier, (b) test-buggy classifier, (c) tests-modify-rejection
guard. The 2026-04-28 inverted-AUROC scenario is the load-bearing
regression test.

### Exp C — Metrics provenance tagging in deliverables

**Deliverable:** edits to `scripts/experiment_template.py:build_result()`
(*ALREADY SHIPPED 2026-04-28* — `metrics_used` parameter added) +
`python/carnot/eval/__init__.py:__version__` (*ALREADY SHIPPED*) +
`scripts/audit_metric_provenance.py` (*ALREADY SHIPPED*) +
`tests/python/test_metric_provenance.py` + retroactive backfill of
`research-complete.yaml` with notes on exp995, exp1003.

**What it does (post-2026-04-28 portion):**

1. Migrate experiment scripts to call `build_result(...,
   metrics_used=["auroc"])` for every metric they publish.
2. Run `audit_metric_provenance.py` against `results/`; backfill
   `research-complete.yaml` entries for exp995 and exp1003 with
   `auroc_inverted_pre_2026_04_28: true`.

**Acceptance:** new experiments emit `metrics_provenance` field
automatically. Audit script run against `results/` lists all
deliverables tagged with a known-buggy version on demand.

### Exp D — CLAUDE.md update enforcing the new discipline

**Deliverable:** edits to root `CLAUDE.md` adding three rules.

**What it does:**

Add to "Operational Principles" section:

```
## Metric Implementation Discipline (MANDATORY)

1. No per-experiment metric helpers. All AUROC, F1, precision/recall,
   and similar metric computations must import from
   `carnot.eval.metrics`. Per-experiment helpers are forbidden — the
   2026-04-28 inverted-AUROC bug shipped because of the antipattern.

2. Self-heal never modifies tests. When the conductor's pre-flight
   self-heal Sonnet fixes a failing test, it must modify *production
   code* only. If the test is genuinely bugged, the failure-ledger
   discipline (skip after 3 attempts) handles it and surfaces the test
   for human review. Modifying a test to match buggy production code
   is the primary failure mode the 2026-04-28 retroactive correction
   guards against.

3. Provenance tagging. Every deliverable JSON whose metrics were
   computed by `carnot.eval.metrics` must carry a top-level
   `metrics_provenance` field. The `experiment_template.py:build_result()`
   flushes this automatically when `metrics_used=[...]` is passed;
   experiment scripts must not strip it out.
```

**Acceptance:** CLAUDE.md updated. Planner Sonnet sees the rules
during milestone planning and applies them when designing new
experiments.

## Decentralization implications

- **Rule 1 (local-first):** unaffected. `carnot.eval.metrics` has zero
  external dependencies in its runtime path; sklearn is test-only.
- **Rule 7 (no vendor abstractions):** unaffected. All metric
  definitions are pure numpy in the core package.

## Risks

- **Migration creates churn.** ~30+ experiment scripts will need an
  import swap. Mitigation: scripted migration + existing tests must
  pass post-migration.
- **Self-heal classifier false positives.** Sklearn might disagree
  with carnot for non-standard edge cases. Mitigation: the classifier
  only runs on functions explicitly in `carnot.eval.metrics`.
- **Tiered self-heal cost.** Opus on second attempt costs more than
  Sonnet. Mitigation: production-bug detector should *reduce* total
  Sonnet wall-clock by guiding attempts more accurately.

## Acceptance criteria

1. ✅ `python/carnot/eval/metrics.py` exists, all 15 tests pass
   (sklearn cross-validation included). **Already shipped 2026-04-28.**
2. ✅ `scripts/audit_metric_provenance.py` exists. **Already shipped 2026-04-28.**
3. ✅ `experiment_template.py:build_result()` accepts `metrics_used` param
   and flushes `metrics_provenance` field. **Already shipped 2026-04-28.**
4. All `scripts/experiment_*.py` and `python/carnot/**/*.py`
   per-experiment metric helpers either deleted or replaced with
   imports from `carnot.eval.metrics`.
5. Conductor self-heal cross-checks failing tests against
   `carnot.eval.metrics` reference implementations.
6. Conductor self-heal rejects diffs that touch `tests/python/*.py`.
7. Tiered self-heal: 30-turn Sonnet first, 50-turn Opus on second
   attempt only.
8. CLAUDE.md updated with the three new rules.
9. `research-complete.yaml` retroactively backfilled with the
   `auroc_inverted_pre_2026_04_28` notes on exp995 + exp1003.

## Why this is in change-proposals, not just a code change

The discipline around metric implementation and self-heal touches the
planner's milestone-design behavior and the conductor's operational
core. Both must respect the rules at design time, not just at code
time. CLAUDE.md is the planner's required reading and the conductor's
operational reference; the rules belong there with explicit rationale
that links back to the 2026-04-28 inverted-AUROC incident as the
load-bearing precedent.
