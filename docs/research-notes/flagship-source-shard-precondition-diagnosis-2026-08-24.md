# Flagship source-shard precondition: diagnosis (2026-08-24)

Scope: why `exp6581` (Qwen3.6-35B-A3B) and `exp6582` (Gemma-4-31B) both end at
`blocked_precondition_failed_without_quality_claim`.

Evidence is read-only. No experiment was re-run. No GPU work was launched. No
pytest was run — see "What I could not determine" for why that limit was
deliberate.

This note was reviewed by an independent adversarial reviewer before commit.
The review refuted four claims in the first draft. Those corrections are folded
in below and listed in "Corrections to the first draft", per the Error
Lifecycle rule.

## Summary

Two things are true at once, and the first draft of this note got the second
one wrong.

1. **The check is wrongly placed and wrongly formed.** It runs the whole
   project test suite as a *precondition*, so it aborts the run before any
   model loads. It runs that suite with a command form this repo diagnosed and
   fixed in 2026-06, and the fix never reached these scripts.
2. **The check is nevertheless reporting a true fact.** The full suite is
   genuinely red, repo-wide, and has been for at least twelve days — 68 to 470
   failures depending on the run. It is not a broken check crying wolf.

So the fix is **not** to delete the check. It is to move it off the critical
path, fix its command form, and escalate the red suite as its own finding.

**Neither model was ever loaded**, so neither artifact is a negative result
about Qwen3.6 or Gemma-4-31B. The work remains unmeasured.

## 1. The check, by file and line

`python/carnot/experiment_6581_qwen36_flagship_source_shard.py:1513`

```python
"verification_commands": bool(tests_run)
and all(row.get("exit_code") == 0 for row in tests_run),
```

`tests_run` comes from `_checkpoint_tests()` (`:1455`), which runs seven
commands. The fourth is the full suite:

```python
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"   # :169
```

`exp6582` does not define its own check. It delegates to
`shared._collect_preconditions`; the check exists once, at `6581:1513`. Its
command tuple is at `6582:441-450`, its constant at `6582:100`.

The check feeds `all_required_preconditions_available = all(checks.values())`,
which guards the live run:

- `experiment_6582_...py:494` — `if preconditions["all_required_preconditions_available"]:` wraps `_run_live_shard(...)`
- `experiment_6581_...py:1926` — the same guard

When False, the script skips the model, then sets
`reason = "precondition_failed"` (`6582:510` / `6581:1942`) and writes
`f"blocked_{reason}_without_quality_claim"` (`6582:337` / `6581:1162`).

## 2. Why it fails, and why not after a load

**It does not fail after a load. No model is ever loaded.** The brief's premise
is incorrect here, and so are two entries already in the repo (§5). The
artifacts say so directly:

| Field | exp6581 | exp6582 |
|---|---|---|
| `preconditions_checked.model_process_started` | `false` | `false` |
| `rows` | 0 | 0 |
| `process_and_gpu_receipts` | 0 | 0 |
| `raw_response_receipts` | 0 | 0 |
| `duration_s` | 927.03 | 2411.02 |
| sum of `tests_run` durations | 925.00 | 2409.08 |

Verification commands are 99.8% of each run. The "15-40 minutes of GPU work"
is the test suite. There is no GPU work in either artifact.

**The two artifacts do not fail for the same reason set.**

```
exp6581: failed_preconditions = ["verification_commands"]
exp6582: failed_preconditions = ["idle_supported_gpu", "verification_commands"]
```

exp6582 also had no free GPU: `selected_gpu: null`, `selection_reason: "no
eligible idle GPU"` — GPU 0 held by a `.venv/bin/python` at 256 MB, GPU 1 by a
`llama-server` at 20686 MB. **Fixing the pytest gate alone does not unblock
exp6582.** It would still block on box contention.

Six of seven commands exit 0 in every run. The failing one is the full suite:

| Run | Exit | Duration | What the output says |
|---|---|---|---|
| exp6582 first (`b40a131bfd`) | **3** | 1097.27s | `87 failed, 10326 passed, 9 skipped` then an xdist `INTERNALERROR ... assert not crashitem` after a worker `MemoryError` |
| exp6581 | **124** | 900.05s | timeout (124 is synthesized by `_run_command`, `6581:1257`, on `TimeoutExpired`) |
| exp6582 re-run (`a00ecfa1bd`) | **124** | 2400.04s | timeout |

## 3. Is the check wrong, or the work?

**Both answers are partly right, and that is the finding.**

### The check is wrongly placed and wrongly formed

**a. The command form is known-bad here.** `pyproject.toml:215-233` sets
`addopts = ["--cov=python/carnot", "--cov-report=term-missing",
"--cov-fail-under=99", ...]` plus `"-v"` and `"-n", "4"` at `:231-232`. A bare
`pytest tests/python -q` inherits all of it. `tests/python/conftest.py:97-105`
names this exact string as a past incident:

> PRETEST_COMMAND=`pytest tests/python -q`, WITH pyproject coverage addopts)
> ran codex past its 4801s cap. ROOT CAUSE ... COVERAGE INSTRUMENTATION made
> them ~15x slower (6s each with --no-cov vs 90s+ with coverage) ... the
> conductor's own full-suite gate already uses `--no-cov -o addopts=`

The canonical form is `scripts/research_conductor.py:1525-1538`:

```python
[venv_pytest, "tests/python", "-q", "--no-header", "-n", "0",
 "--no-cov", "-o", "addopts="]
timeout=600
```

Note `-n 0` there. The bare form keeps `-n 4`, and the observed exit-3 crash is
an **xdist worker `MemoryError`** under coverage — four coverage-instrumented
workers on a contended box. The same scripts get this right for the focused
test (`pytest <own-test> -q --no-cov -n 0`) and omit it only on the full-suite
line.

**b. The task prompt never asked for it.** In `research-roadmap.yaml`
(milestone 2026.08.572), step 0 PRECONDITIONS lists gates, hashes, GGUF
metadata, tokenizer, CPU, RAM, disk, CUDA support, per-GPU memory, seed,
budget, timeout, expected unload. No test suite. Step 9 says "Run focused
tests, lint, spec coverage..." — **focused**, and **after** "Write one terminal
artifact atomically". The script promoted a post-work focused check into a
pre-work full-suite gate.

**c. It gates the wrong thing.** Two model families reach a byte-identical
verdict because the check reads the repo's test suite, not the model.

**d. No measured run of it has ever passed.** Across the last 250 artifacts,
97 rows carry the exact string `.venv/bin/pytest tests/python -q`. Only **4**
have a recorded duration over 60s — the only plausibly-measured full runs — and
their exits are 3, 2, 124, 124. **None exited 0.** The other 92 rows carry only
`{command, exit_code, summary}` with no duration; 57 of those declare exit 0
without evidence of a run. Those declared passes should not be read as the
suite being green.

### But the check is reporting something true

The suite is genuinely red, and independent artifacts using the same command
say so in their own words:

| Artifact | Reported |
|---|---|
| exp6582 first run | `87 failed, 10326 passed, 9 skipped` |
| exp6510 | "repository-wide run stopped after 68 failed, 9638 passed ... and an xdist worker MemoryError" |
| exp6506 | "global suite interrupted after unrelated repository-wide failures, missing optional ONNX deps, **tracked result mutations**, and JAX worker aborts" |

The first draft of this note claimed the check "distinguished nothing". That
was wrong. It distinguished a suite carrying dozens to hundreds of failures.
Deleting it would remove a signal that is firing truthfully — the mirror image
of the `SILENT_NON_FIRING` hazard, and the same class of mistake.

### Is the work wrong?

**Unknown, and that is the point.** Zero rows, zero receipts, no model load, no
quality claim. These artifacts contain no evidence about source-shard
capability in either direction. Retiring them as failed experiments records a
negative that was never measured.

## 4. Proposed fix

**Do not delete `FULL_PYTEST_COMMAND`.** Three changes instead:

1. **Take it out of `checks`.** Keep running it; stop letting it set
   `all_required_preconditions_available`. Record its result as an advisory
   receipt. The prompt asks for focused tests as a gate, and those already pass.
2. **Fix the command form** to the canonical one: add `--no-cov -o addopts=`
   and `-n 0`. This removes the ~15x coverage slowdown, the `-n 4` worker OOM,
   and the unrelated `--cov-fail-under=99` gate. Only then does its exit code
   mean "tests failed".
3. **Move it after the artifact write**, per prompt step 9, so a repo-health
   check never costs a GPU slot.

**Separately, and more important than either experiment:** the red suite is a
real finding that needs its own owner. 68-470 failures over twelve days, with
`--cov-fail-under=99` in the default `addopts`, means the project's documented
`pytest` command does not pass today. That belongs in `ops/known-issues.md`,
not buried in a blocked experiment's `tests_run` array.

**What the fix would break.** Little, if the check is kept and demoted rather
than deleted. It does not weaken the structured gates, negative fixtures,
attack rows, protected-file hashes, or the focused/coverage/lint/spec checks —
all of those run and pass today. The real risk is the opposite one: demoting
the check without filing the red-suite finding would quietly discard a true
signal. Step 2 above is what keeps it honest; the escalation is what keeps it
visible.

**It does not unblock exp6582 on its own** (§2): that run also failed
`idle_supported_gpu` on a contended box. A reopen of exp6582 needs a free GPU,
not just a harness fix.

**A secondary hazard, not manifested here.** Running the full suite inside an
experiment risks the Test-Run Record Integrity Discipline: a test run rewrites
`results/**`. exp6506's summary reports exactly that ("tracked result
mutations"). For these two runs specifically, `git show --stat` on
`b40a131bfd` and `a00ecfa1bd` shows no `results/**` collateral beyond each
experiment's own artifact, so the hazard is real in general but did not fire
here. Note also that `protected_files_unchanged` covers only
`research-roadmap.yaml` and `scripts/research_conductor.py`; its
`all_unchanged: true` says nothing about `results/**`.

## 5. Two existing records in the repo are wrong

Both were written the same day and both should be corrected:

- `ops/exclusion_manifest.yaml:1059` — "Both runs loaded a model and did real
  work first". False. `model_process_started: false` in both, zero rows, zero
  GPU receipts. This is likely the source of the premise that the failure comes
  after a successful load.
- `scripts/conductor_exclusion_manifest.json:174` — "`preconditions_checked`
  absent. No precondition failed." False on both counts.
  `preconditions_checked` is present in both artifacts, and
  `failed_preconditions` is populated in both.

Retirement state as of this note: `exp6581` was **un-retired** by commit
`aaf899eca4`; `exp6582` remains retired.

## 6. Corrections to the first draft

Recorded rather than silently patched, per the Error Lifecycle rule.

1. **Claimed `verification_commands` was the only failed precondition in both.**
   Wrong for exp6582, which also failed `idle_supported_gpu`. I read exp6581's
   field and generalized without checking exp6582's.
2. **Claimed the check "carries no information".** Wrong, and the most
   consequential error: the suite is genuinely red. The conclusion flipped from
   "delete it" to "demote and fix it".
3. **Attributed exit 3 to the JAX/absl coverage SIGABRT** recorded in memory.
   The actual traceback is an xdist `assert not crashitem` after a worker
   `MemoryError`. Related family, different mechanism.
4. **Claimed a 7200s pytest would be "killed by the task budget"** of 4200s.
   Wrong: `family_task_deadline` (`6582:424-428`) is computed at `:493`, after
   `_checkpoint_tests`, and its docstring says the budget starts "after
   verification and resource checks".
5. **Claimed only two scripts use the bare command.** It appears in 97 rows
   across the last 250 artifacts. My first scan filtered on list-form commands
   of length <= 4 and missed the rest.
6. **Stated exp6581 was retired.** It had been un-retired by `aaf899eca4`.

## 7. What I could not determine

**Whether `pytest tests/python -q --no-cov -o addopts= -n 0` passes, and how
long it takes.** No such receipt exists anywhere in the recent corpus — nobody
has a green baseline. I did not produce one, deliberately:

- It rewrites `results/**`, the evidence I was asked to read and not write.
- It takes over 18 minutes at minimum; the pre-crash run reached 1097s.
- `tests/python/conftest.py` runs a GPU-zombie sweep at import. A live seed
  sweep holds GPU 1 and the conductor holds GPU 0.

This matters more now than it did in the first draft: with the suite known red,
the open question is how much of the 87-470 failure count is real product
breakage and how much is coverage/xdist collateral. That needs a measured
`--no-cov` baseline, run by someone who can safely take the box.

**Whether the conductor's `STALL_TIMEOUT` (1800s, `research_conductor.py:646`)
would kill a longer pytest.** The 2411s run survived it, so something kept
output flowing. Unresolved.

**Whether `exp6567`'s `admitted=[]` shares this root cause.** Not examined.

## Recommendation

Reopening is the operator's call. The harness defect is named and real, so a
reopen satisfies the Failed-Experiment Rerun Discipline's "name what is
different" condition, and the retirement should not stand as a *measured*
negative about either model — nothing was measured. But a reopen of exp6582
also needs a free GPU, and the red suite is a bigger finding than either
experiment.

## 8. Addendum (same day): the zero-terminal-rows question

A follow-up asked why a live run produces zero authentic terminal rows, on the
premise that `preconditions_checked` is absent from the artifacts and that no
precondition failed. **That premise is false, and it traces to the incorrect
manifest entry named in §5.** Verified with `json.load` on both artifacts:

```
'preconditions_checked' in artifact  : True   (both; 29 keys each)
all_required_preconditions_available : False  (both)
failed_preconditions                 : exp6581 ['verification_commands']
                                       exp6582 ['idle_supported_gpu','verification_commands']
```

The aggregate's own checks dict agrees: `checks.preconditions` is **False** in
both — it reads `preconditions_checked.all_required_preconditions_available`
(`6581:969-972`).

### Q1 — what makes a row "authentic terminal"

`_row_is_authentic()` at `6581:787`. Applied per expected source unit at
`6581:868-870`, producing the `authentic` list. Two consumers:

- `"authentic_terminal_rows": exact_coverage and all(authentic)` (`6581:949`)
- `"authentic_terminal_row_count": sum(authentic)` (`6581:979`)

Every field in `aggregate_row_recomputation` is derived from `rows`
(`6581:977-991`): `terminal_row_count = len(rows)`, and the token, latency and
`charged_cost` totals are sums over `rows`.

### Q2 — none produced, not produced-then-rejected

Decisive, because `terminal_row_count` and `authentic_terminal_row_count` are
separate counters:

| Counter | exp6581 | exp6582 |
|---|---|---|
| `expected_unit_count` | 4 | 4 |
| `terminal_row_count` | **0** | **0** |
| `authentic_terminal_row_count` | 0 | 0 |
| `claim_bearing_row_count` / `failure_row_count` | 0 / 0 | 0 / 0 |
| `prompt` / `response` / `total_token_count` | 0 / 0 / 0 | 0 / 0 / 0 |
| `latency_s` / `charged_cost` | 0 / 0 | 0 / 0 |

Produced-then-rejected would show `terminal_row_count > 0` with
`authentic_terminal_row_count == 0`. Both are 0. Zero tokens and zero latency
confirm no generation ever ran. `charged_cost: 0` is not a hint of rejection —
it is a sum over an empty list.

The cause is the guard in §1: `rows` is initialized empty, `_run_live_shard` is
never called, and `model_process_started` stays `false`. **No GPU work
occurred in either run.** The 927s / 2411s is the test suite (§2).

### Q3 — the verdict is not lying

"All structured gates passed" and "precondition_failed" are not in conflict.
They are the two arms of one branch (`6581:1941-1942`, `6582:509-510`):

```python
structured_failed = any(row.get("passed") is not True for row in gates)
reason = "structured_gate_failed" if structured_failed else "precondition_failed"
```

`gates` is only the two upstream artifact field checks (exp6579, exp6580).
Both passed, so `structured_failed` is False, so the reason is
`precondition_failed` — which is exactly what `failed_preconditions` records.
The verdict string is accurate. What misleads is that `gate_check_summary`
covers upstream gates only, while the precondition set lives in a different
field; reading the first and not the second suggests nothing failed.

**Cheap fix worth making:** have the blocked report name the failed
preconditions in the verdict or a top-level field, so a reader does not have
to open `preconditions_checked` to learn which of eleven checks failed.

### New finding: several aggregate checks pass vacuously on zero rows

`raw_receipts`, `checkpoints`, `diagnostics`, `costs_recomputed`,
`failures_retained` are `all(...)` / `len(x) == len(rows)` over empty lists
(`6581:912-934`). With `rows == []` they are all True. So the artifact reports
ten to twelve passing checks for a run that did nothing. `ready_score` is still
0.0 because the conjunction includes the checks that do fail, but the per-check
detail reads far healthier than the run was. This is the QA-layer
"green because there was nothing to check" pattern and is worth a guard.

### Corrections carried in from the follow-up

- The conductor poisoned and re-ran **twice**, both exp6582, not ten times.
- exp6581 is **un-retired** (`aaf899eca4`); only exp6582 remains retired. This
  note already said so in §5 and does not treat exp6581 as a dead direction.

## Cross-references

- `results/experiment_6581_qwen36_flagship_source_shard.json`
- `results/experiment_6582_gemma4_31b_flagship_source_shard.json` (and `b40a131bfd`, the exit-3 run)
- `python/carnot/experiment_6581_qwen36_flagship_source_shard.py:169,1257,1455,1513,1926,1942`
- `python/carnot/experiment_6582_gemma4_31b_flagship_source_shard.py:100,104,424,441,494,510`
- `scripts/research_conductor.py:1525-1538` — the canonical full-suite form
- `tests/python/conftest.py:97-105` — the 2026-06 incident this repeats
- `pyproject.toml:215-233` — the inherited coverage addopts
- `ops/exclusion_manifest.yaml:1052-1096`, `scripts/conductor_exclusion_manifest.json:172-174` — the two incorrect entries
- CLAUDE.md "Pre-Launch Preconditions Discipline", "Failed-Experiment Rerun
  Discipline", "Test-Run Record Integrity Discipline", "QA-Layer Authenticity
  Discipline" (the `SILENT_NON_FIRING` mirror image), "The Error Lifecycle"
