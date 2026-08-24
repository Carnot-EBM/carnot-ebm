# Independent review: verification_commands demote-and-repair (commit 696cb32224)

**Date:** 2026-08-24
**Reviewer:** independent outer-loop agent. Did not read the builder's notes
before forming a view. Findings come from the code, the artifacts, and the
live box.
**Subject:** commit `696cb32224`, "Demote and repair verification_commands;
file the red suite". Seven files.

## Verdict

The change is sound. Ship it.

One label is wrong. For the repository-wide suite the outcome is REMOVAL, not
relocation. Point 6 covers this. The builder states the cost openly, so the
label is imprecise rather than misleading.

## 1. Does the red suite still get reported? No. The claim is TRUE.

Nothing standing reports it. I confirmed the builder's central claim
independently, because that claim decides whether this change traded a visible
block for an invisible problem.

**`run_tests(full=True)` is dead code — CONFIRMED.**
`scripts/research_conductor.py:1467` defines `run_tests(full: bool = False)`.
There are four call sites. Every one of them runs with `full=False`:

| Call site | Form |
|---|---|
| `scripts/research_conductor.py:6125` | `run_tests()` — default |
| `scripts/research_conductor.py:6181` | `run_tests()` — default |
| `scripts/research_conductor.py:6671` | `run_tests(full=False)`, commented "full suite hangs serially" |
| `scripts/research_conductor.py:6798` | `run_tests(full=False)` |

The `full=True` branch at `:1523-1538` is correctly written. It uses
`--no-cov -o addopts= -n 0`. It is never reached.

**CI does not report it either.** `.github/workflows/ci.yml:101` gives the
`Python Tests (100% coverage)` job `needs: python-lint`. The `python-lint` job
has failed on every one of the last 60 runs I sampled. So the tests job
reports `skipped` on every run. Checked on five individual runs.

**The conductor pretest does not report it.** It runs in subset mode. The
current `ops/.pretest-cache.json` records `81 passed, 1 warning in 9.24s`.

So the repository has had no standing full-suite reporter. A misplaced
precondition was the only thing incidentally surfacing the failures.

### Correction to the premise

The two blocked runs did NOT observe a red suite. Both recorded
`exit_code 124`, a `TimeoutExpired`, at 900 s and 2400 s, with no pytest
summary produced at all. Read the receipts in
`results/experiment_6581_qwen36_flagship_source_shard.json` and
`results/experiment_6582_gemma4_31b_flagship_source_shard.json`.

The redness is separately real. I verified it from independent artifacts. The
87-failure figure reproduces exactly:
`git show b40a131bfd:results/experiment_6582_gemma4_31b_flagship_source_shard.json`
contains `87 failed, 10326 passed, 9 skipped, 112 warnings in 1093.25s`.

So the check was failing for a THIRD reason on top of the two known ones. It
inherited `--cov`, the suite was red, AND the timeout budget could not cover
it. That strengthens the case for the change.

### A related integrity finding, outside this diff

The only full-suite rows now entering NEW artifacts are fabricated.
`python/carnot/experiment_6580_v572_source_and_joint_method_protocol.py:320`
hardcodes `{"command": ".venv/bin/pytest tests/python -q", "exit_code": 0,
"duration_s": 0.1}`. `experiment_6572_content_derived_gguf_metadata_resolver.py`
does the same. exp6580's row lands verbatim inside exp6581's artifact at
`.source_protocol.tests_run[4]`.

A declared 0.1-second pass of a 1093-second suite must never become the answer
to "is the suite green". Now that the honest runner is gone, it is the only
full-suite evidence reaching the record.

## 2. Fail direction: closed, in every case I could construct

`focused_verification_ok` fails closed. So does `_run_command` beneath it:
`experiment_6581_qwen36_flagship_source_shard.py:1253-1259` maps `OSError` to
127 and `TimeoutExpired` to 124.

### MISSED INPUT

I could not construct an input that passes when it could not run. Each case
below was executed end to end through the real `_run_named_test`, not reasoned
about.

| Constructed case | exit | gate |
|---|---|---|
| missing binary (wrong venv path) | 127 | BLOCK |
| missing target test file | 4 | BLOCK |
| import error in target | 1 | BLOCK |
| timeout (0.01 s budget) | 124 | BLOCK |
| zero tests collected (`-k` matches nothing) | 5 | BLOCK |
| no rows at all | n/a | BLOCK |

The fail-open pattern the builder named is real and IS live, in exactly one
place: `python/carnot/experiment_6457_independent_verifier_bounded_csl_audit.py:1247`,
`return all(row.get("exit_code") in (0, None) for row in tests)`. It is
fail-open twice. An empty list passes vacuously, and a row that never reported
passes. It feeds `verification_commands_pass` in that module's readiness gate.
A repository-wide sweep for the pattern returns this one hit and no others. The
builder was right not to copy it. It deserves its own ticket.

## 3. Mutation proofs bite — re-run independently

I re-ran them in-process, so I never touched the builder's files. Each mutation
was applied and then restored.

| Mutation | Result |
|---|---|
| A — `focused_verification_ok` fails open on an empty set | RED |
| B — restore `FULL_PYTEST_COMMAND` | RED |
| C — drop `--no-cov` from the focused pytest command | RED |
| D — treat a missing `exit_code` as a pass | RED |
| restored | GREEN |

Committed state: 26 passed, ruff clean, `check_spec_coverage.py` OK.

## 4. Nothing else was weakened

The committed diff to the 11-key precondition dict is exactly one line:

```
-        "verification_commands": bool(tests_run)
+        "verification_commands": focused_verification_ok(tests_run),
```

All 11 keys remain. `idle_supported_gpu`, `llama_cpp_cuda_build`,
`gpu_telemetry` and `fresh_qwen_process` are untouched. `idle_supported_gpu`
was NOT relaxed. It correctly reported the busy GPU.

## 5. Scope: one decision point, not two — CONFIRMED

The delegation is real.
`experiment_6582_gemma4_31b_flagship_source_shard.py:455` reads
`preconditions, server, initial = shared._collect_preconditions(*args, **kwargs)`,
where `shared` is `experiment_6581_qwen36_flagship_source_shard` (imported at
`:27`). So exp6582 inherited the gate without carrying the string, which is why
a grep for the key found one definition.

Wider sweep, for the question "is the fix narrower than the problem": 664
modules carry a bare repo-wide `pytest tests/python` with no `--no-cov`. Only
four of those also carry precondition machinery — exp6457, exp6572, exp6573,
exp6575. I checked each. None gates a model load on the full suite. The fix
covers the whole problem.

## 6. Placement: this is a removal, not a relocation

There is no post-work verification. `FULL_PYTEST_COMMAND` is deleted from both
modules, and nothing runs the suite afterward. The spec codifies the removal:
`REQ-REPORT-6581-VERIFY-SCOPE` states "SHALL NOT run the repository-wide test
suite."

What was genuinely demoted and repaired is the CHECK FUNCTION. An inline
expression inside a `# pragma: no cover` function became
`focused_verification_ok`, a named and tested seam that fails closed over six
focused commands. That part is real and is an improvement.

A reader of the commit TITLE alone would infer a post-work run exists. It does
not. The commit BODY names the cost in its own paragraph, and
`ops/known-issues.md` carries the signal with measurements, the coverage and
xdist caveat, and three ordered steps. So this is an imprecise label, not a
hidden deletion.

The supersede of `REQ-REPORT-6582-VERIFICATION-BUDGET` is handled well. The
original text is preserved verbatim per never-prune, the reason is stated, and
sentences two and three stay in force. They also stay under test:
`test_verification_is_bounded_without_spending_family_runtime` still asserts
`family_task_deadline(...) == 5200.0`.

## Residual risks

1. **Fabricated full-suite receipts** — see the finding under point 1.
2. **`run_tests(full=True)` has `timeout=600`**, but the one completed measured
   run took 1093 s with coverage. Without coverage it should be faster, but that
   is unmeasured. Measure before relying on it for step 3 of the known-issues
   plan.
3. **Step 6, cheaply.** The pre-commit hook `arc-precondition-nocov-lint`
   already enforces this exact `--no-cov` rule. Its `files:` regex matches only
   ARC-shaped experiment names, so it never saw exp6581 or exp6582. Widening
   that hook's file scope to all `python/carnot/experiment_*.py` is smaller than
   writing a new check, and it makes the lesson fire.

## What I could not determine

- The suite's true failure count under a correct invocation
  (`--no-cov -o addopts= -n 0`). No such receipt exists in the corpus. The
  known-issues entry says the same. I deliberately did not run it. It takes
  18+ minutes, it rewrites `results/**`, and many agents are active in this
  session. That is the concurrent-destruction hazard CLAUDE.md documents.
- Whether the 87 to 470 failures are product breakage or coverage and xdist
  collateral. Both artifacts I traced ended in worker crashes: an
  `xdist worker MemoryError`, and "Interrupted after 1:03:03 at 45%". The
  builder flags the same uncertainty rather than claiming a number.

## Working-tree note

I introduced no dirt. Four files show modified — `ops/.audit_findings_ledger_state.json`,
`ops/.run_sentinel_state.json`, `ops/.stop_authority_state.json`, and
`results/experiment_1822_rtl_synth.log`. All four were already modified before
my first command. They are live conductor and sentinel state. I left them
alone. A blanket restore over another workflow's live state is the documented
data-loss hazard. I ran only the two focused test files, never the full suite,
and touched no GPU.

## Cross-references

- commit `696cb32224` — the change under review
- `git show b40a131bfd` — the 87-failure receipt that reproduces the claim
- `docs/research-notes/flagship-source-shard-precondition-diagnosis-2026-08-24.md`
- `ops/known-issues.md` 2026-08-24 entry — where the red-suite signal now lives
- CLAUDE.md "Pre-Launch Preconditions Discipline", "QA-Layer Authenticity
  Discipline", "Test-Run Record Integrity Discipline"
