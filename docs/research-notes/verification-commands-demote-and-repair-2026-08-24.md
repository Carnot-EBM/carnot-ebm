# verification_commands: demote and repair (2026-08-24)

What changed, why, and what it costs. The diagnosis this implements is
`flagship-source-shard-precondition-diagnosis-2026-08-24.md`.

Operator directive: demote and repair, do not delete.

## Blast radius, corrected

The handover said two files carry the check. It is **one**.
`experiment_6582_gemma4_31b_flagship_source_shard.py:27` imports
`experiment_6581_qwen36_flagship_source_shard as shared` and delegates
`_collect_preconditions` to it, so the `checks` dict is defined once at
`6581:1513`. Both files did define their own `FULL_PYTEST_COMMAND` and their own
`_checkpoint_tests` command tuple, so both needed editing — but there is a
single decision point, not two.

## What changed

**1. Demote — the repository-wide suite is out of the precondition set.**
`FULL_PYTEST_COMMAND` is removed from `_checkpoint_tests` in both modules, and
the constant is deleted. The set is now six focused commands: the module's own
test file, its coverage run and report, ruff check, ruff format, and spec
coverage. All six ran and exited 0 in both blocked artifacts; total cost about
25 seconds against the 900-2400 seconds the full suite consumed.

Rationale in the spec, not just here: a pre-launch precondition exists to prove
the resources a run needs are present. Repo-wide suite health is not such a
resource. Gating a live model load on it blocks the run for a reason unrelated
to the measurement, which is exactly what happened.

**2. Repair — fail-closed, in a tested seam.** The inline expression became a
named function so it can be tested at all. The old check lived inside
`_collect_preconditions`, which is `# pragma: no cover`, so it was never under
test.

```python
def focused_verification_ok(tests_run: Sequence[Mapping[str, Any]]) -> bool:
    rows = list(tests_run)
    if not rows:
        return False
    return all(row.get("exit_code") == 0 for row in rows)
```

**3. Regression guards.** Two new tests in the exp6581 suite and one rewritten
test in the exp6582 suite assert no command targets the whole `tests/python`
tree, and that every pytest command carries `--no-cov`.

**4. Spec.** `REQ-REPORT-6581-VERIFY-SCOPE` and
`SCENARIO-REPORT-6581-VERIFY-FAIL-CLOSED` added to
`openspec/capabilities/research-reporting/spec.md`.

**5. A contradicting requirement superseded.** The working tree already held an
uncommitted `REQ-REPORT-6582-VERIFICATION-BUDGET` requiring the task to "run the
exact full Python suite once with a bounded timeout of at least 4,801 seconds" —
the third timeout escalation, written into the spec. It was preserve-committed
first (`3b9cbf77d3`), then superseded in place rather than deleted, per
never-prune: the original text is quoted, its first sentence is marked
superseded by `REQ-REPORT-6581-VERIFY-SCOPE`, and its second and third sentences
(verification must not consume the 4,200-second family budget; the family
deadline starts after verification) are explicitly kept in force. Its test was
rewritten to assert the opposite of what it used to assert — see mutation 3.

## Fail direction: closed, deliberately

**A verification that could not run blocks the run.** Empty set is False. A row
with no `exit_code`, or `exit_code: None`, is False. Only a non-empty, all-zero
set passes.

Why closed: the six remaining commands are all scoped to this experiment's own
module and test file. If they cannot run, the experiment's own code is not
known-good, and a measurement produced by code that failed its own tests is not
evidence. This is the narrow, resource-shaped question a precondition should
ask. The cost of a false block is one cheap re-run; the cost of a false pass is
a fabricated-looking artifact.

The repo contains the opposite pattern, and it should not be copied:
`experiment_6457_independent_verifier_bounded_csl_audit.py:1246-1247` has

```python
def _tests_passed_or_pending(tests): return all(row.get("exit_code") in (0, None) for row in tests)
```

which treats "never ran" as a pass. That is the trusted-and-silent state
CLAUDE.md calls the worst state for a guard. Mutation 2 below is exactly this
pattern applied to the new helper, and a test refuses it.

## Mutation proofs

Each mutation applied, suite run, then restored. All at
`pytest ... -q --no-cov -n 0`.

| # | Mutation | Result |
|---|---|---|
| 1 | Delete the `if not rows: return False` guard | **RED** — `test_focused_verification_fails_closed`, `assert True is False` |
| 2 | `== 0` becomes `in (0, None)` (the exp6457 fail-open pattern) | **RED** — same test |
| 3 | Re-add `FULL_PYTEST_COMMAND` to both command tuples | **RED** ×2 — `test_verification_scope_excludes_repository_suite` (`assert not True`) and exp6582's `test_verification_is_bounded_without_spending_family_runtime` (`assert 7 == 6`) |
| — | Restored | **GREEN** — 26 passed |

Also green after the change: 100% coverage on both modules under their own
`--fail-under=100` command (381 and 199 statements), ruff check, ruff format,
spec coverage.

## What this costs — the red suite

**This change removes the last thing that reported the red suite.** Stated
plainly because it is the part worth arguing with.

The suite is genuinely red: 87, 68, 470, 433 failures across artifacts over
twelve days. The check was reporting a true fact in the wrong place.

Nothing else in the loop runs it. `scripts/research_conductor.py:1467`
`run_tests(full: bool = False)` has a correct full-suite branch at `:1525-1538`
(`--no-cov -o addopts=`), but **every call site passes `full=False`** —
`:6125`, `:6181`, `:6671`, `:6798`, two commented "full suite hangs serially".
The `full=True` branch is dead code.

So the signal is filed in `ops/known-issues.md` under **NEW 2026-08-24**, with
the measurements, the caveat that coverage and xdist collateral inflate the
counts, and three ordered steps: get one `--no-cov` baseline, triage against
`--cov-fail-under=99`, then give the check a real home (revive the conductor's
dead `full=True` branch on a low cadence, or a periodic job).

That entry, not this fix, is where the red suite now lives. If it is closed
without a baseline, the signal is lost for real.

## What could break

- **Weaker per-experiment regression coverage.** These two experiments no longer
  notice a repo-wide break. Accepted: they never noticed it usefully — the check
  returned non-zero on every measured run, so it could not distinguish a new
  break from the standing one, and it cost 15-40 minutes and the model load each
  time.
- **Nothing else.** Structured gates, negative fixtures, attack rows,
  protected-file hashes and the six focused commands are untouched and pass.
- **Not addressed on purpose:** `idle_supported_gpu`, which exp6582 also failed.
  It correctly detected a busy GPU (GPU 1 held at 20686 MB by the seed sweep).
  Left alone per operator instruction. exp6582 therefore still will not run
  while the box is contended, harness fix or not.
- **Retirement unchanged.** exp6582 stays retired; exp6581 stays un-retired.
  Not my call, and not touched.

## Related defect, not fixed here

`experiment_6580_v572_source_and_joint_method_protocol.py:320` records
`{"command": FULL_PYTEST_COMMAND, "exit_code": 0, "duration_s": 0.1}` — a
declared full-suite pass in 0.1 seconds. The corpus scan behind the diagnosis
found 92 of 97 full-suite rows carry only `{command, exit_code, summary}` with
no duration, and 57 of those declare exit 0. Those declared passes should not be
read as the suite being green. Out of scope for this fix; worth its own pass.

## Files

- `python/carnot/experiment_6581_qwen36_flagship_source_shard.py` — helper, check, command set
- `python/carnot/experiment_6582_gemma4_31b_flagship_source_shard.py` — command set, override map
- `tests/python/test_experiment_6581_qwen36_flagship_source_shard.py` — two new tests
- `tests/python/test_experiment_6582_gemma4_31b_flagship_source_shard.py` — rewritten scope test
- `openspec/capabilities/research-reporting/spec.md` — REQ and SCENARIO
- `ops/known-issues.md` — the red-suite entry
