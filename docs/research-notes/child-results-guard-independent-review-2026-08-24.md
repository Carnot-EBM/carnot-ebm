# Independent review of the child-results guard (commit 9b43df2c82)

Date: 2026-08-24. Reviewer: an outer-loop agent that did not write the fix and
did not read the builder's note before forming its own view.

Subject: commit `9b43df2c82`, "Stop test subprocesses rewriting the tracked
results record", and the diagnosis behind it.

## Verdict

The diagnosis is correct. The fix works for the writers it targets. One live
counterexample survives the fix, inside a blind spot the commit declares.

## 1. The named writer is necessary and sufficient

Both halves were tested with the guard deliberately disabled. The disable used
a throwaway pytest plugin that restores `subprocess.Popen.__init__` from
`child_results_guard._ORIGINAL_POPEN_INIT`. No tracked file was edited to run
the test.

- Sufficient. `tests/python/test_experiment_1736_kanele_synth.py` run alone
  changed the artifact from `4b2f773a...` to `8dd4681d...`. It dropped exactly
  `flagged_adversarial`, `corrigendum_note` and `corrigendum_pending`. The test
  reported PASSED while doing it.
- Necessary. The other 15 affected test files, guard disabled, left the same
  artifact byte-identical. The hash was `4b2f773a...` before and after.
- With the guard active, the same test leaves the tree clean.

The `test_experiment_3361_*` isolation claim also holds. Its `_module` half
writes in process and leaves the tree clean. Its `_script` half writes through a
subprocess, rewrites the committed artifact, and still reports green.

## 2. A trap for anyone who re-verifies this

`scripts/experiment_1736_kanele_synth.py:11` branches on `which vivado`.

- With Vivado on `PATH`, the script runs Vivado, fails a DRC check, exits
  non-zero, and writes NOTHING under `results/`. It does dirty the tracked file
  `output/kanele_synth/post_synth.dcp`.
- The conductor's systemd unit has no Vivado on `PATH`. It always takes the
  writing branch.

An interactive shell on this box has Vivado at
`/tools/Xilinx/2025.2.1/Vivado/bin/vivado`. So a reviewer who reproduces from a
normal shell sees the test write nothing, and concludes wrongly. Reproduce under
the conductor's `PATH`.

## 3. Live counterexample: a non-Python child still rewrites tracked evidence

`results/experiment_1822_rtl_synth.log` is a tracked file under `results/`. It
still changes with the fix installed.

Reproduced deliberately. Guard active, committed tree, no disable plugin:

    pytest tests/python/test_experiment_1822_rtl_synth.py -q --no-cov

The file went from `3df4a636...` to `24b4897e...`. It was restored
byte-identically afterwards.

Mechanism. `tests/python/test_experiment_1822_rtl_synth.py:12` calls
`subprocess.run(["make", "synth-constraints"])`. `make` and `yosys` are compiled
binaries. They never import `sitecustomize`, so the injected `PYTHONPATH` has no
effect on them. The guard redirects rather than refuses, so a child it cannot
reach writes to the real path with no signal.

This is the first blind spot the commit lists, so the fix behaves as documented.
The gap is between the commit's title and its scope. Non-Python children still
rewrite the tracked results record.

Why nobody noticed. The file is not JSON and holds no determination keys, so a
determination-loss check never flags it. The repo already recorded it once:
`python/carnot/experiment_6211_v538_post_marker_source_scope_prereg.py:259`
names this same log in an earlier suite-run rewrite set.

Size of the exposed class: 14 test files spawn a non-Python child
(`git`, `bash`, `make`, `which`, `echo`, `ls`, `gitleaks`). Only the `make` case
is a demonstrated writer into `results/` today.

## 4. Two more blind spots the commit does not list

The child shim wraps `open`, `io.open`, `os.open`, `os.rename` and `os.replace`.
It wraps no deletion or truncation call. The in-process sibling does cover them
(`tracked_results_guard.py:244` handles `os.remove`, `os.unlink`, `os.truncate`).

Measured in an isolated fake repository. Real evidence was never touched.

| child vector | evidence survives | listed in commit |
|---|---|---|
| `os.remove` | no, file deleted | no |
| `os.truncate` | no, file emptied | no |
| `os.system` | no | yes |
| `python -E` | no | yes |
| `shutil.copyfile` | yes | not applicable |
| plain `open` (control) | yes | not applicable |

This is not theoretical. `tests/python/test_experiment_3395_energy_based_replay.py:66`
calls `os.remove` on a tracked results path today. It is caught only because it
runs in process. Move that line into the script under test and it becomes silent
evidence deletion. `os.link` and `os.symlink` share the gap.

## 5. Two behaviour bugs inside the covered path

Both reproduce against the committed module.

- Write-then-read-back fails. Writes are redirected and reads are not, so a child
  that reads back its own output raises `FileNotFoundError`. This contradicts the
  module docstring at `child_results_guard.py:34`, which states that such a child
  "still sees its own bytes".
- Atomic publish fails. `_wrap_move` redirects `dst` but not `src`, so the common
  `write tmp` then `os.replace(tmp, final)` pattern raises
  `OSError: [Errno 18] Invalid cross-device link`. The in-process sibling
  `_compat_os_rename` redirects both. The redirect root is on tmpfs, so the
  devices differ.

## 6. False positives: none measured

- 16 affected test files, 251 tests, guard active: 250 passed, 1 failed.
- The same 16 files with the guard disabled: the same test still fails.

So the failure is not caused by the fix. It is the pre-existing
`tracked_results_guard.py:249` refusing an in-process `os.remove`. With the guard
active, `results/`, `output/` and `openspec/` all stayed clean.

The guard cannot fire on conductor work. `install()` returns False unless
`CARNOT_CHILD_GUARD_REDIRECT_ROOT` is set, and only `conftest.py` sets it. A
concurrent non-test pytest ran throughout this review under the committed guard
without incident.

One side observation: with the guard disabled,
`test_determination_preservation_lint` drops from 74 passed to 73 passed and 1
failed. The guard is load-bearing for a test in the suite. The failing test was
not isolated by name.

## 7. Overhead: measured independently

Median over 40 spawns each, warm, same interpreter:

- no shim: 18.04 ms, minimum 14.23 ms
- with shim: 19.04 ms, minimum 14.19 ms
- delta: +1.00 ms per spawn

The commit reports -0.1 ms. A negative overhead is noise, not a speed-up, so that
figure should read "within noise". The substantive claim holds. The cost is about
0 ms to 1 ms, not the roughly 27 ms of the `importlib.find_spec` version.

## 8. Attribution is not repaired, only made to matter less

Probed from inside a child under the committed guard:

- `CARNOT_MUTATION_WRITE_LOG` is unset.
- `scripts/_mutation_observer` is absent from `sys.path`.
- The only `sitecustomize` loaded is the guard's own shim.

So the mutation observer never runs in a child. Every marker armed during this
review recorded `attributed_to_run: []` for files a test had just written.

The fix stops the write, so for the `results/` JSON class there is nothing left
to attribute. For every uncovered vector, a child-caused mutation is still
reported as NOT ATTRIBUTED, with the advice that it "may be a concurrent agent's
in-flight work" and must not be reverted. That advice is backwards for real test
damage. It tells the operator to leave it in place.

A cheap repair exists. The guard already injects a `PYTHONPATH` entry into every
child. Adding `scripts/_mutation_observer` to that same entry, and exporting
`CARNOT_MUTATION_WRITE_LOG`, reuses machinery that already exists. The shim's own
chain loop would then load it.

## 9. Recommended follow-ups, highest value first

1. Wrap `os.remove`, `os.unlink` and `os.truncate` in the child shim, to match
   the in-process sibling. Silent deletion is worse than a stale rewrite, and the
   pattern is in the repo now.
2. Fix `tests/python/test_experiment_1822_rtl_synth.py` at the test, not the
   guard. No `PYTHONPATH` trick can reach `make`. Point the Makefile target at an
   output directory, or mark the log as generated.
3. Redirect `src` in `_wrap_move`, and place the redirect root on the repository's
   filesystem so `os.replace` does not cross devices.
4. Redirect reads for paths already redirected, or delete the claim at line 34.
5. Add `scripts/_mutation_observer` to the child `PYTHONPATH` to restore
   attribution.
6. Add a test asserting `child_results_guard.is_installed()` under a live
   session. Both sibling guards have one
   (`test_operator_curated_doc_guard.py:54`,
   `test_experiment_artifact_isolation.py:348`). This guard does not, and every
   one of its failure paths is fail-open.

## 10. Corrections to the commit record

- The commit attributes `output/kanele_synth/post_synth.dcp` to concurrent-agent
  work. It was not. The Vivado branch of the exp1736 diagnostic run wrote it, and
  this reviewer restored it with `git checkout` minutes later. The three
  `ops/.*_state.json` files WERE concurrent work, and leaving them alone was
  correct.
- The overhead figure "-0.1 ms" should read "within noise". See section 7.

## 11. What this review did not determine

- Which tests each of the seven originally observed runs selected.
- When the `test_experiment_3395` failure began. It is independent of this fix.
- Which test in `test_determination_preservation_lint` needs the guard.
- Whether the read-back, atomic-replace and deletion classes occur among children
  spawned in a FULL suite run. Only 16 test files were run, not the whole suite.

## 12. Tree hygiene for this review

Dirtied and restored, each verified byte-identical by sha256:
`experiment_1736_kanele_synth.json`, `experiment_3361_archive_v309_activate_v310.json`,
`experiment_3377_archive_v310_activate_v311.json`,
`experiment_3392_archive_v311_activate_v312.json`,
`experiment_833_constraint_delta_root_cause.json`,
`experiment_1822_rtl_synth.log`, and `output/kanele_synth/post_synth.dcp`.

`scripts/test_suite_mutation_check.py --gate` returns OK. No results file was
left modified by this review.
