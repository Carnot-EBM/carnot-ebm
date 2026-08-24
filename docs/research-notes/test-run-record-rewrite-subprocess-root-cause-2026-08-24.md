# The test that rewrites the research record: named, proven, and fixed

Date: 2026-08-24. Author: outer-loop session.
Related: CLAUDE.md "Test-Run Record Integrity Discipline" · REQ-REPORT-6143 · REQ-REPORT-6157 ·
`docs/research-notes/test-suite-rewrites-the-record-survey-2026-07-29.md`.

## Summary

Seven pytest runs on 2026-08-24 each rewrote tracked files under `results/`, stripping
`flagged_adversarial`, `corrigendum_note` and `corrigendum_pending` and adding nothing.

The writers are named below. The root cause is one sentence:

> The project's `results/` guard works, and both of its mechanisms are per-interpreter, so
> every write that still lands comes from a **child process** the guard cannot see.

This was fixed by carrying the redirect across the subprocess boundary. Measured cost after
optimisation: **-0.1 ms per child spawn**, which is below noise.

## 1. Which test writes `results/experiment_1736_kanele_synth.json`

`tests/python/test_experiment_1736_kanele_synth.py:14`

```python
result = subprocess.run(["python", script_path], capture_output=True, text=True)
```

Call path:

| Step | Location |
|---|---|
| test spawns a child, cwd = repo root (pytest's cwd) | `tests/python/test_experiment_1736_kanele_synth.py:14` |
| child builds a hardcoded dict | `scripts/experiment_1736_kanele_synth.py:39-50` |
| child writes the repo-relative path | `scripts/experiment_1736_kanele_synth.py:52-54` |

The same shape drives the three archive/activate artifacts. Each of those test files has
**two** tests, and only the second one does damage:

| Artifact | In-process test (clean) | Subprocess test (damaging) |
|---|---|---|
| `experiment_3361_archive_v309_activate_v310.json` | `test_..._3361_module:7` | `test_..._3361_script:14` |
| `experiment_3377_archive_v310_activate_v311.json` | `test_..._3377_module:12` | `test_..._3377_script:22` |
| `experiment_3392_archive_v311_activate_v312.json` | `test_..._3392_module:12` | `test_..._3392_script:22` |

Each writes through `python/carnot/reporting/archive_v*_activate_v*_*.py:25-27`
(`Path("results/...").write_text(...)`, CWD-relative).

**The observed damage set is the subprocess-launching test set.** Sixteen test files spawn
`subprocess.run(["python", ...])` or `[sys.executable, ...]`; twelve of the fifteen files the
monitoring saw modified map onto that list exactly.

## 2. Why exactly three keys vanish

The writers rebuild the artifact from a hardcoded template and `json.dump` it. Nothing merges
the file that is already on disk, so every key added *after* the original run is dropped. The
three that vanish are precisely the ones the conductor's `adversarial_verify --backfill`
stamps on later.

`scripts/experiment_1736_kanele_synth.py` also branches on whether Vivado is on `PATH`. The
conductor's systemd `PATH` (`.../.venv/bin:/usr/local/bin:/usr/bin`) has no Vivado, so it
always takes the "simulating success" branch — which is why the damage signature is stable.

Reproduced byte-exactly in a scratch copy, under the conductor's own `PATH`:

```
BEFORE: [bitfile_generated, corrigendum_note, corrigendum_pending, experiment,
         flagged_adversarial, honest_verdict, status, utilization, vivado_available, wns]
AFTER : [bitfile_generated, experiment, honest_verdict, status, utilization,
         vivado_available, wns]
```

**This writer should not call `scripts/artifact_merge_preserve.py`.** That helper exists for
analyzer *rebuilds*, which are legitimate regenerations of a live artifact. These writers are
not rebuilding anything — they are a test re-running an old experiment against the committed
record. The correct outcome is that the write never reaches `results/` at all, which is what
CLAUDE.md rule 2 already says. Merge-preserve would have hidden the symptom and left the
rule broken.

## 3. Why the existing guard did not fire

`python/carnot/testing/tracked_results_guard.py` is wired into `tests/python/conftest.py` and
has two mechanisms:

* `install()` — a PEP 578 audit hook, which belongs to the interpreter that added it;
* `install_legacy_results_write_compat()` — monkeypatches `builtins.open`/`io.open`/`os.open`/
  `os.rename`/`os.replace`, which belongs to that interpreter's memory.

Neither crosses `fork`+`exec`. `operator_curated_doc_guard.py`'s own docstring already states
this hole for its case; nothing had connected it to `results/`.

`test_experiment_3361_*` isolates the two halves on one writer, so the dichotomy is directly
measurable:

```
pytest ...::test_experiment_3361_module   -> 1 passed;  git status results/  CLEAN
pytest ...::test_experiment_3361_script   -> 1 passed;  git status results/  M ...3361...json
```

Running the exp1736 test alone reproduces the same thing, and the run's own mutation observer
reported the file as **NOT ATTRIBUTED** — it could not see its own run's write, because the
write happened in a child. That is a textbook `SILENT_NON_FIRING`: green guard, damaged
record, and positive evidence that nothing was wrong.

## 4. The fix

`python/carnot/testing/child_results_guard.py` (new), installed from `conftest.py`
`pytest_configure`. It wraps `subprocess.Popen` so every child Python interpreter starts with
a generated stdlib-only `sitecustomize.py` that reinstalls the same redirect. Writes aimed at
`<repo>/results/...` land under the session's temp artifact root; reads are untouched, so a
child that writes and reads its own path back still sees its own bytes.

Stdlib-only is load-bearing: the children here are launched as bare `python`, which on this
box is `/usr/bin/python` with no venv and no `carnot` package.

**Redirect, not refuse.** This matches what the in-process layer already does, so it adds no
new policy: the offending tests stay green and stop touching the record. Refusing is stricter
and would fail them outright; that belongs to whoever owns REQ-REPORT-6157, not to a
side effect of closing this gap.

### Verification

| Check | Result |
|---|---|
| 6 new guard tests | pass |
| exp1736 + 3361 + 3377 + 3392 + 3391 + 3403 (11 tests) | pass, all four artifacts byte-identical, `results/` clean |
| sibling guard suites (`test_operator_curated_doc_guard`, `test_experiment_artifact_isolation`) | 63 passed total, no regressions |
| per-spawn overhead, interleaved A/B, n=25 | +27.3 ms before optimisation, **-0.1 ms after** |

The first overhead measurement was real and was caused by `importlib.util.find_spec` running
its full finder protocol at the start of every child. Replacing it with a plain `isfile()`
scan removed the entire cost.

## 5. What this fix does NOT catch

Stated plainly, because a guard believed to be total is worse than one known to be partial.

* **Non-Python children.** `sh -c 'echo x > results/y.json'`, a compiled binary, or a vendor
  tool such as Vivado. `PYTHONPATH` means nothing to them.
* **`os.system`, `os.exec*`, `os.posix_spawn`.** Only `subprocess.Popen` is wrapped.
* **`python -S` / `python -E`.** Both skip `sitecustomize`; `-E` also drops the injected
  `PYTHONPATH`.
* **A child that overwrites `PYTHONPATH` for its own grandchildren.** Ordinary inheritance
  does reach grandchildren; deliberate replacement does not.
* **Everything outside `<repo>/results/`.** `openspec/**`, `ops/**` and `output/**` are just
  as tracked and are not covered.
* **The tests themselves are still wrong.** They still re-run production experiment scripts
  against the repo root. CLAUDE.md rule 2 says a test must never write tracked state; this
  guard makes the write harmless, it does not make the test correct. Pointing all sixteen at
  `tmp_path` remains the right follow-up, and is the one fix that would survive someone
  removing this guard.

A clean run under this guard is therefore not proof that nothing wrote to the record.
`scripts/test_suite_mutation_check.py` and `git status` remain the backstop.

## 6. Could not determine

* **Why the blast radius varied run to run** (some runs hit only exp1736). The most likely
  explanation is that different invocations selected different test subsets, and that a file
  already stripped and not yet re-stamped produces no diff on a second strip. Not proven.
* **Three of the fifteen observed files** — `experiment_2510_ensemble_v7.json`,
  `experiment_2514_kv260_pynq_flash.json`, `experiment_1822_rtl_synth.log` — are not written
  by any of the sixteen subprocess tests. They carry no `flagged_adversarial` stamp, so they
  lose nothing, but their writer was not identified.
* **Full-suite blast radius of the guard.** Verified against the offender set and the sibling
  guard suites, not against all ~2,305 test files. A test that spawns a child, writes
  `results/`, and then asserts the *real* file changed would now fail. No such test was found
  among those exercised.

## 7. Housekeeping

Diagnostic runs during this investigation dirtied `results/experiment_1736_kanele_synth.json`
and `results/experiment_3361_archive_v309_activate_v310.json`. Both were restored with
`git checkout --` and verified byte-identical by md5 before any commit. The pending
mutation marker retired itself once the tree showed the damage undone.

`results/experiment_1736_kanele_synth.json` was also observed going dirty with the same
signature at a moment when this session had not run its test — a concurrent run reproducing
the bug live. `output/kanele_synth/post_synth.dcp` was modified inside this session's window
by a Vivado run this session did not perform; it was left alone rather than reverted, per the
standing rule against blanket-reverting another agent's in-flight work.
