# Running the test suite rewrites the research record — the survey

**Date:** 2026-07-29
**Author:** outer loop (unattended session; operator asleep)
**Status:** SURVEY COMPLETE for the stated scope. **NO TEST WAS CHANGED.** The repair is a design
call left to the operator — options and a recommendation are in §6.
**Origin:** commit `b3e31d341`, "[outer-loop] HAZARD: running the test suite silently rewrites the
research record" (diagnosed, not fixed).

---

## 1. The hazard, restated

Running

```
pytest tests/python/test_arc_*.py tests/python/test_experiment_*.py
```

left **39 tracked files modified** that were clean before the run: 36 `results/*.json` / `.log`
artifacts, plus `openspec/papers/paper-v6/section-6-limitations.md` and
`output/kanele_synth/post_synth.dcp`.

The rewrites are never-prune violations, not cosmetic:

| artifact | what the run did to it |
|---|---|
| `results/experiment_3946_r11l_first_solve.json` | lost `inference_substrate_correction_note`, `inference_substrate_original_invalid_value`, `solve_provenance`, `solve_provenance_note` — a hand-written 2026-07-27 corrigendum, deleted |
| `results/experiment_307_jepa_real_training.json` | `inference_mode` flipped `live_gpu` → `cpu_training` |
| `results/experiment_1035_dualgpu_rocm_v3.json` | `run_date` / `started_at` / `finished_at` rewritten to today |

These artifacts are the input to the fabrication gate (`scripts/adversarial_verify.py`), to every
capstone aggregation, and to the paper. A green test run that deletes a corrigendum or flips an
`inference_mode` makes the record disagree with what was measured, silently. Anyone who then commits
with `git add -A` publishes the rewrite.

The hazard commit named a suspected mechanism — `runpy.run_path` on a real experiment script — and
said explicitly that **the artifact-writing set was NOT enumerated**. This document enumerates it.

---

## 2. Method

The point of the method is that it does not guess which tests "look like" they write.

**Isolation.** Three throwaway `git worktree --detach` checkouts of `HEAD` on tmpfs. The canonical
repo was never the test target. `environment_files/` (5.2 MB, gitignored) was copied into each
worktree — without it the ARC tests abort early with `Game <id> not found in scanned environments`
and the survey would under-report.

**The editable-install trap — and where this survey did not fully avoid it.** The venv installs
`carnot` in editable mode via `__editable__.carnot_ebm-0.1.0b1.pth`, which points at the
**canonical** `.../ianblenke/carnot/python`. A worktree run therefore imports the real repo's
package unless `sys.path` is reordered, and any module computing a path from `carnot.__file__`
writes into the real repo. The whole-suite run and the per-batch attribution runs set
`PYTHONPATH=<worktree>/python:<worktree>/scripts/experiments` to put the worktree first. **The two
per-file subset harnesses (§3.1, §3.2) did not** — an oversight found while writing this up. Their
results are nevertheless sound, because they rest on a stronger check than isolation: the canonical
repo's 31,564 tracked files were sha256-hashed before and after those runs and showed only this
session's deliberate edits. Isolation was the intended mechanism; the hash comparison is the one
that actually carries the conclusion.

**Isolation leaked three times anyway, which is itself a result.** Over the whole session, three
artifacts appeared as modified in the CANONICAL repo while every test run was pointed at a `/tmp`
worktree: `results/experiment_3734_…json`, `results/experiment_2427_kv260_yosys_v4.json`, and
`results/experiment_3351_gatemate_latency_benchmark.json`. All three were restored and verified
byte-identical against `HEAD`. Two are explained by §3.4 (a hardcoded absolute path in the script).
The third looked inexplicable at first — `scripts/experiment_3351_gatemate_latency_benchmark.py:9`
resolves `REPO_ROOT = Path(__file__).resolve().parents[1]`, which is correct and repo-relative — and
the explanation turns out to matter more than the leak:

**One hardcoded `sys.path` insert poisons an entire xdist worker.** Three test files insert the
HARDCODED canonical path directly onto `sys.path`:

```python
# tests/python/test_arc3_gap3_stage2_ebm.py:14 (also …gap4_rule_exec, …gap3_stage2v2_ebm)
sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/scripts/experiments")
```

and ten scripts under `scripts/` do the equivalent with a hardcoded `PROJECT_ROOT`. An xdist worker
is a long-lived process running hundreds of test files in sequence. Once ANY of those runs, the
canonical tree is on that worker's `sys.path` for the rest of its life — so every later
`import experiment_NNNN` in that worker resolves to the CANONICAL module, its `__file__` is the
canonical path, and a perfectly correct `Path(__file__).resolve().parents[1]` then points at the
operator's checkout. A repo-relative script writes canonically through no fault of its own.

**Consequence: worktree isolation is structurally defeated, per worker, by a single line in an
unrelated test.** Treat "the tests were run in a worktree" as a mitigation, never as containment.
The load-bearing check is the before/after hash of the canonical tree.

**The oracle is git, not inference.** After each unit of work, `git status --porcelain -uno` names
every tracked file that moved. Untracked files are deliberately excluded: an experiment script
writing a *brand new* artifact is normal work; the harm is an *existing, committed* file changing
underneath you.

**Per-test attribution inside a single run.** A pytest plugin installs a CPython audit hook
(`sys.addaudithook`) that records every `open`-for-write / `os.rename` / `os.remove` whose path is
inside the repo, attributed to the pytest nodeid executing at the time. This fires inside a
`runpy.run_path` of an experiment script, which is precisely where the writes happen. Output is
sharded per xdist worker and merged.

**Restoration.** Every worktree was `git checkout -- .`'d between units, and the canonical repo was
hashed before and after the whole session (§5).

---

## 3. What the survey found

### 3.1 The `runpy` → experiment-script class is NOT the main source

The hazard commit's hypothesis was that the damage comes from tests that `runpy.run_path` a real
experiment script. That class was enumerated exactly and **run one file at a time**, each in a clean
worktree:

* **20 candidate test files** call `runpy.run_path` on a path naming an experiment script.
  (Only **2** name a path literally under `scripts/experiments/`:
  `test_experiment_3946_r11l_first_solve.py` and `test_experiment_3967_m3_honest_efficiency.py`.
  The hazard commit's "11 target `scripts/experiments/`" was a looser grep.)
* Of those 20, **exactly one moved a tracked file**:

  | test file | tracked file it rewrote |
  |---|---|
  | `tests/python/test_experiment_3946_r11l_first_solve.py` | `results/experiment_3946_r11l_first_solve.json` |

  The rewrite loses all four records listed in §1 and changes `duration_s`. Reproduced twice,
  independently (once under the audit-hook plugin, once by plain `git status`).
* The other 19 moved nothing — no tracked modification and no untracked artifact.

**So the `runpy` hypothesis explains 1 of the 39 files, not the bulk.** That is a real correction to
the hazard commit, and it matters: a repair aimed only at the `runpy` call sites would have left
~38 rewrites in place while looking like a fix. §3.3 identifies what the bulk actually is.

### 3.2 All 125 `test_arc_*.py` files, individually: zero tracked-file movement

Every `tests/python/test_arc_*.py` was run alone in a clean worktree. **None moved a tracked file.**
Six aborted early in the worktree (`test_arc_early_stop_sweep`, `test_arc_kaggle_machine_shape`,
`test_arc_object_history_salience_live_wiring`, `test_arc_ptrm_stage1_generator`,
`test_arc_structured_memory_causal_audit`, `test_arc_world_model_trust_energy`) — a test that errors
before its write would be a false negative, so all six were re-run **in the canonical repo** under
`scripts/test_suite_mutation_check.py --restore --run`. Result: 67 passed, 2 failed for unrelated
content-drift reasons, and **no tracked file was modified**.

### 3.3 The bulk is `import the module and call main()`, not `runpy`

The 3,366 `tests/python/test_experiment_*.py` files were run in 14 batches of 250 in a clean
worktree, `git status` after each; every batch that moved a tracked file was then re-run under the
audit-hook plugin for exact per-test attribution.

Batch 00 (the alphabetically-first 250 files) moved **6** tracked artifacts, and the plugin named the
responsible test for every one:

| test | artifact rewritten |
|---|---|
| `test_experiment_1035_dualgpu_rocm_v3.py::test_main_writes_well_formed_artifact` | `results/experiment_1035_dualgpu_rocm_v3.json` |
| `test_experiment_1038_milestone_retro_80.py::TestMainEntryPoint::test_main_writes_artifact` | `results/experiment_1038_milestone_retro_80.json` |
| `test_experiment_1081_fpga_scale_benchmark.py::test_run_experiment_writes_artifact_with_required_fields` (and `::test_run_experiment_marks_board_unreachable_when_ping_fails`) | `results/experiment_1081_fpga_scale_benchmark.json` |
| `test_experiment_1089_milestone_retro_84.py::TestMainEntryPoint::test_main_writes_artifact` | `results/experiment_1089_milestone_retro_84.json` |
| `test_experiment_1103_milestone_retro_85.py::TestMainEntryPoint::test_main_writes_artifact` | `results/experiment_1103_milestone_retro_85.json` |
| `test_experiment_1593_cdg_repair.py::test_main` | `results/experiment_1593_cdg_repair.json` |

**None of these uses `runpy`.** Every one does `sys.path.insert(0, ".../scripts")`, `import
experiment_NNNN_… as expNNNN`, then calls the module's `main()` / `run_experiment()` and asserts the
artifact exists. `runpy.run_path` was a red herring: it is one way to execute the script, and the
overwhelmingly more common way is a plain import-and-call.

**The candidate class is therefore three orders of magnitude larger than the hazard commit's
estimate.** Static count over `tests/python/`:

| class | files |
|---|---|
| import an `experiment_*` module | **1,696** |
| …and call `main()` / `run_experiment()` | **704** |
| call `runpy.run_path` (any target) | 125 |
| …with an experiment-script target | 20 |

Not all 704 write — the test name is the tell (`test_main_writes_artifact`,
`test_run_experiment_writes_artifact_with_required_fields`), and many only exercise helpers. The
batch survey is what separates the two.

**Per-batch results (250 files each, alphabetical; the run was still in progress when this document
was written — stated rather than rounded up):**

| batch | test files | tracked files moved |
|---|---:|---:|
| `expbatch_00` | 250 | 6 |
| `expbatch_01` | 250 | 16 |
| `expbatch_02` | 250 | 3 |
| **so far** | **750** | **25** |

**Batch 01, attributed per test** (the audit-hook plugin, merged across xdist workers):

| test node | file rewritten |
|---|---|
| `test_experiment_209_cleanup.py::test_run_cleanup_rewrites_public_docs_with_provenance_labels` (+3 siblings, and `test_experiment_1931_hf_publisher.py::test_success_upload`, `test_experiment_1750.py::{test_experiment_1750_blocked,_success}`) | **`README.md`** |
| `test_experiment_1911.py::test_experiment_1911_schema` | `openspec/papers/paper-v6/section-6-limitations.md`, `results/experiment_1911_phase4_canonical_decision.json` |
| `test_experiment_1747_diagnostic.py::test_main` | `results/experiment_1747_ebt_mode_collapse_check.json` |
| `test_experiment_1750.py::{test_experiment_1750_blocked,test_experiment_1750_success}` | `results/experiment_1750_huggingface_retry.json` |
| `test_experiment_1938_nrgpt_loss_probe.py::test_experiment_1938_nrgpt_loss_probe` | `results/experiment_1938_nrgpt_loss_probe.json` |
| `test_experiment_1970_domino_fast_constraints.py::test_domino_fast_constraints` | `results/experiment_1970_domino_fast_constraints.json` |
| `test_experiment_2085_pem_sudoku_eval.py::test_run_experiment` | `results/experiment_2085_pem_sudoku_eval.json` |
| `test_experiment_2090_crane_humaneval.py::test_main` | `results/experiment_2090_crane_humaneval.json` |
| `test_experiment_2097.py::test_experiment_2097_evaluates_eqm_vs_pem` | `results/experiment_2097_eqm_eval.json` |
| `test_experiment_2521_ensemble_v7.py::test_run_experiment_creates_deliverable` | `results/experiment_2521_ensemble_v7.json` |
| `test_experiment_2538_kv260_sd_flash.py::test_experiment_2538_produces_valid_artifact` | `results/experiment_2538_kv260_sd_flash.json` |
| *(unattributed — see below)* | `output/kanele_synth/post_synth.dcp`, `results/experiment_1822_rtl_synth.log`, `results/experiment_2510_ensemble_v7.json` |

**`README.md` is rewritten by a test that says so in its own name.**
`test_run_cleanup_rewrites_public_docs_with_provenance_labels` runs `experiment_209_cleanup`'s
public-docs rewriter against the live `README.md`. CLAUDE.md's Public Documentation Discipline lists
`README.md` as operator-curated and forbids the autonomous loop from editing it; a passing test does
it on every suite run. The hazard commit's list of 39 did not include it.

**Three files are unattributed, and that is a stated limit of the instrument, not an oversight.** A
CPython audit hook only sees writes made by the Python process it is installed in. A script that
shells out — Vivado for `output/kanele_synth/post_synth.dcp`, yosys for
`results/experiment_1822_rtl_synth.log` — writes from a CHILD process the hook never observes. Those
three are visible to `git status` (which is why they appear in the batch total) but not to per-test
attribution. Anyone extending this survey should treat subprocess writers as a separate class
requiring a different instrument (e.g. an `strace`/`fanotify` layer, or wrapping `subprocess.run`).

Batch 01's 16 include the two NON-`results/` files from the original hazard —
`openspec/papers/paper-v6/section-6-limitations.md` and `output/kanele_synth/post_synth.dcp` — and
one the hazard commit did not list at all: **`README.md`**, which CLAUDE.md's Public Documentation
Discipline marks operator-curated. Two of the 16 (`experiment_1938_nrgpt_loss_probe.json`,
`experiment_2085_pem_sudoku_eval.json`) are artifacts from the 2026-07-27 incident — see §3.5.

### 3.4 The worse finding: 139 scripts write to a HARDCODED absolute path, so nothing can isolate them

Midway through the survey the canonical repo went dirty on its own —
`results/experiment_3734_fix_harness_and_bounded_train_chunk1.json` lost `flagged_adversarial: true`,
its `corrigendum_pending` / `corrigendum_note`, AND the hand-written
`flagged_adversarial_restoration_note` recording that this same artifact had already been stripped
once on 2026-07-27 and restored. Nothing in this session had touched it, and the survey was running
in a `/tmp` worktree.

The cause is line 11 of `scripts/experiment_3734_fix_harness_and_bounded_train_chunk1.py`:

```python
PROJECT_ROOT = "/home/ianblenke/github.com/ianblenke/carnot"
```

The script does not resolve its output path relative to anything. It writes to the operator's
canonical checkout **by absolute path**, from whatever directory it is executed in. Worktree
isolation cannot contain it. `tmp_path` cannot contain it. A `CARNOT_RESULTS_DIR` env var cannot
contain it, because the script never asks. A CI runner executing it writes to a path that does not
exist on that machine — or, on this machine, to the live research record.

**Scale, counted mechanically:**

| | files |
|---|---:|
| `scripts/**/*.py` containing a literal absolute repo path | **150** |
| …of those, that also open-for-write / `write_text` / `json.dump` | **139** |
| `python/carnot/**/*.py` containing a literal absolute repo path | **99** |

Two spellings are in use and both must be counted: `/home/ianblenke/github.com/ianblenke/carnot`
(104 files) and `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm` (46 files), the latter being a
symlink to the former. An earlier count in this document said 104/94/36 because it grepped only the
first spelling — `scripts/experiment_2427_kv260_yosys_v4.py`, caught leaking live during this
survey, uses the second.

**Nothing was watching for this.** `scripts/canonical_url_lint.py` explicitly and correctly PERMITS
the local filesystem path — its rule is about the project's canonical *GitHub URL*, and it lists
`/home/ianblenke/github.com/ianblenke/carnot/...` under "Permitted (these are NOT canonical-URL
violations — leave alone)". That exemption is right for URLs and leaves the path completely
unguarded as a *write target*.

**Three consequences, in increasing order of importance:**

1. It partially invalidates this survey's own negatives. A hardcoded-path script writing during a
   worktree run shows up in the CANONICAL repo, not in the worktree's `git status` — so
   "this test moved nothing" is a false negative for that class. The §3.1 and §3.2 results are
   unaffected (the canonical repo was hashed immediately after both and showed only this session's
   deliberate edits), but §3.3's per-batch negatives carry this caveat.
2. It defeats repair options (A) and (E) for this class. Both work by making the output path
   configurable; a script that never reads a path cannot be redirected. These 94 need the literal
   replaced with a repo-relative resolution before any path-based repair means anything.
3. **It is a reproducibility defect independent of the test suite.** These scripts only work on one
   machine, in one directory, for one user. That bears directly on G2 — the independent-reproducer
   gate — and on CLAUDE.md's decentralization constraints: an experiment nobody else can run is not
   an experiment anyone else can check.

**The guard caught it.** The widened lint refused the commit that would have published the exp3734
strip, naming all four lost fields including the two restoration records that the pre-widening form
had no pattern for. The artifact was restored and verified byte-identical against `HEAD`
(`bb8b6b1ae8d324b4ea16b5de5e20bb8d20d32b92d80d5008a2a4eccd2ca62d02`). That is the first live catch,
on a real incident, of exactly the class the widening was written for — and it happened during the
session that shipped it.

### 3.5 The 2026-07-27 "conductor re-run" was a TEST RUN — 5 of its 7 artifacts reproduced on demand

REQ-ARC-WMTE-5995 records that on 2026-07-27 something stripped `flagged_adversarial: True` from
seven artifacts (exp1861, exp1938, exp2085, exp3734, exp4162, exp4170, exp696), six of which also
lost their corrigendum records. The diagnosis written at the time was "a conductor re-run overwrote
the artifact in place."

That diagnosis appears to be wrong. Running the tests reproduces it on demand:

| command | artifacts stripped | result |
|---|---|---|
| `pytest tests/python/test_experiment_1938_nrgpt_loss_probe.py tests/python/test_experiment_2085_pem_sudoku_eval.py` | exp1938, exp2085 | **4 tests PASSED** |
| `pytest tests/python/test_experiment_4162_… tests/python/test_experiment_4170_…` | exp4162, exp4170 | passed |
| (leaked into the canonical repo from a `/tmp` worktree via the hardcoded path in §3.4) | exp3734 | — |

**Five of the seven, reproduced.** Each loses exactly the same four things: `flagged_adversarial:
True`, `corrigendum_pending`, `corrigendum_note`, and — the tell — the hand-written
`flagged_adversarial_restoration_note` that the 2026-07-27 repair itself wrote into the artifact to
explain the restoration. The suite is re-breaking the repair. The remaining two (exp1861, exp696)
have no test file matching their experiment number and could not be attributed this way.

**Every one of these runs was GREEN.** Nothing failed, nothing warned; the quarantine was lifted on
five artifacts by a passing test suite.

**What this changes.** REQ-ARC-WMTE-5995's *fix* was right — a commit-time content lint catches the
class regardless of who wrote the file — but its *reasoning* ("the conductor does not write these
files; the experiment SCRIPT does, and there are thousands of them") was right for a reason it did
not name. The writer is not the conductor and not usually a script the conductor invoked: it is the
test suite, running on a developer's or an agent's machine, in the ordinary course of checking that
nothing is broken. Any repair that targets the conductor's write path would have missed all five.
All five artifacts were restored and verified byte-identical against `HEAD`.

### 3.6 A note on why `--restore` is opt-in, learned the hard way during this session

To measure the §3.4 class empirically, this session ran a background collector that polled the
canonical repo for leaks and called `test_suite_mutation_check.py --check --restore` on whatever it
found. It worked — and it also reverted an in-progress edit to *this document*, because a file the
author is editing is indistinguishable, at the git layer, from a file a test rewrote. The edit was
reconstructed and the collector was killed by explicit PID.

That is not a bug in the tool; it is the reason the tool is shaped the way it is. `--check` alone is
strictly read-only, `--restore` must be asked for by name, and there is a test
(`test_the_detector_never_edits_anything_unless_asked`) that pins it. The lesson for anyone wiring
this into automation: **never run `--restore` on a schedule against a tree someone is working in.**
Snapshot, run the thing, restore once, and take the baseline immediately before the run — not
minutes earlier while edits are still landing.

---

## 4. Coverage — what this survey does and does not cover

**Covered.**

* All 20 test files that `runpy.run_path` an experiment script — individually, exact attribution.
* All 125 `tests/python/test_arc_*.py` — individually, exact attribution.
* All 3,366 `tests/python/test_experiment_*.py` — see §3.3 for the granularity achieved.

**Not covered, stated plainly.**

* `tests/integration/`, `tests/archive/`, `tests/quarantine/`, and the ~986 `tests/python/test_*.py`
  files matching neither `test_arc_*` nor `test_experiment_*` were **not** surveyed. The hazard
  command did not include them; a future run that does could surface more.
* **Ordering and cross-test interaction.** The per-file runs use `-n0`; the hazard run used the
  configured `-n 4`. A rewrite that only occurs when test A runs before test B in the same worker
  would be missed by per-file attribution and caught only by the whole-suite run.
* **Byte-identical rewrites are invisible to this method.** A script that overwrites an artifact
  with exactly its current content leaves no git diff. That is harmless today but is a latent
  landmine: the same call site becomes destructive the moment the script's output changes.
* Six ARC tests and two experiment tests fail in the worktree for environment reasons. Failures
  short-circuit writes, so each is a potential false negative; the six ARC ones were re-checked in
  the canonical repo (§3.2), the experiment-side ones were not individually re-checked.
* **Subprocess writers are invisible to per-test attribution.** The audit hook sees only the Python
  process it is installed in, so a script shelling out to Vivado or yosys is unattributable. Three
  of batch 01's 16 are in that class.
* **At least one leak route into the canonical repo is unidentified** (§2, exp3351). Worktree
  isolation is a mitigation here, not containment; the load-bearing check is the before/after hash
  of the canonical tree.

---

## 5. The record is intact

The canonical repo was hashed with `sha256sum` over all 31,564 tracked files at session start and
again at the end. The **only** tracked files that differ are the four this session deliberately
edited (`.gitignore`, `.pre-commit-config.yaml`, `scripts/determination_preservation_lint.py`,
`tests/python/test_determination_preservation_lint.py`) plus the two new files it added. No
`results/**`, no `openspec/**`, no `output/**` file changed. The protected evidence directories
(`results/arc_e3`, `results/arc_logo_snapshot`, `results/arc_e3_origin_fixtures`) were read only.

One test *was* deliberately run against the canonical repo — `test_experiment_3946_r11l_first_solve.py`,
to dogfood the new detector end to end. It rewrote its artifact, the detector caught it, `--restore`
put it back, and `sha256` is byte-identical before and after
(`6834b56cb72129f044a65a980b62ad3f26d02c21f3471bcda0648205750142ac`).

---

## 6. The repair — options, and a recommendation the operator can overrule

**This session shipped no test change.** The hazard commit was right that the repair is a design call
with a visible tradeoff. The options, with what each costs:

**(A) Redirect the artifact path under test.** Have the experiment script honour an output-directory
environment variable (e.g. `CARNOT_RESULTS_DIR`), and have the test point it at `tmp_path`.
*Keeps* full execution of the real script. *Costs* a change to every script that must participate,
and the scripts are written by many agents over many months with no shared convention — a script
that ignores the variable silently reverts to writing the real path, and nothing tells you.

**(B) Snapshot-and-restore around the call.** A fixture reads the artifact, runs the script, writes
the original bytes back. *Keeps* full execution and needs no change to any script. *Costs*
correctness under `-n 4`: two workers touching the same artifact interleave, and the restore races.
It also leaves a window in which the record on disk is wrong, which is exactly the window a
concurrent `git add -A` would capture.

**(C) Stop re-running the script; assert against the committed artifact.** *Costs* the most: the
test stops exercising the script at all and becomes a schema check on a static file. That is a real
loss of coverage, and it is the option the hazard commit flagged as the one to think hardest about.

**(D) Do nothing to the tests; rely on the guards shipped alongside this survey.** *Costs* nothing
and loses no coverage, but it is detection rather than prevention: the record still gets rewritten
on every suite run, and the guards' job is to stop the rewrite from being *committed*.

**(E) Make the artifact PATH the thing under test, once, centrally.** Every writing script resolves
its output through one shared helper (`scripts/experiment_template.py` already exists and most
scripts already import it) that consults a `CARNOT_RESULTS_DIR` env var; a repo-level autouse
fixture points that var at `tmp_path` for the whole suite. One helper change plus one fixture, and
scripts that bypass the helper are findable mechanically (grep for a literal `results/` open) rather
than one at a time.

**(F) Replace the 139 hardcoded absolute paths with repo-relative resolution.** Independent of, and
prerequisite to, (A)/(E) for that class — see §3.4. Also fixes a reproducibility defect that has
nothing to do with tests: these scripts run correctly on exactly one machine.

**Recommendation: (D) now, then (F), then (E) — NOT (A).**

**This recommendation was revised mid-survey, and the revision is the point.** The first draft of
this document recommended (A) on the strength of §3.1: the `runpy` class is exactly one test and one
artifact, so a targeted per-script fix looked cheap and precise. §3.3 then found that `runpy` is a
red herring. The real mechanism is `import the experiment module and call main()`, the candidate
class is **704 test files**, and every artifact-writing test found so far is in it. (A) does not
survive that: a per-script `CARNOT_RESULTS_DIR` opt-in across hundreds of scripts written by many
agents with no shared convention is precisely the "a script that ignores the variable silently
reverts to writing the real path, and nothing tells you" failure it was already noted to carry —
tolerable for one script, not for hundreds.

(E) is (A) done once instead of hundreds of times, and it converts the residual risk from *silent*
to *mechanically findable*.

(B) should be rejected outright regardless: the suite runs `-n 4` by default, so a restore-based fix
is racy by construction, and a racy guard on the research record is worse than an honest gap.

(F) comes before (E) because it is the harder constraint and it changes what (E) can even reach: a
script that writes to a hardcoded absolute path is unreachable by any configuration mechanism, so
building (E) first would produce a fix that silently does nothing for 139 scripts. (F) is also the
only one of the six that is worth doing even if the test-suite question is abandoned entirely — an
experiment that only runs on one machine in one directory cannot be independently reproduced, which
is the G2 gate.

(D) is the right first move under any of these because it is already shipped, costs nothing, and
loses no coverage — and because it stays correct while (F) and (E) are being built and after they
land, for whatever scripts bypass the helper.

---

## 7. What did ship: two guards, because a guard that did not fire is worse than no guard

`scripts/determination_preservation_lint.py` already refused any commit that drops
`flagged_adversarial` or a `corrigendum*` record. **It sat directly in the path of the exp3946
deletion and stayed silent**, because `inference_substrate_correction_note` is a corrigendum in
substance but not in name. That is the failure mode a trusted guard must never have.

**Widened** (`scripts/determination_preservation_lint.py`), from 2 rules to 4:

* Rule 3 — no *marker* field may be dropped: any top-level key whose name marks it as a correction,
  provenance declaration, disclosure, acknowledgment, retraction, or review note. The pattern list
  was derived by censusing every top-level key in all 15,331 `results/**/*.json` (31,510 distinct
  names) and keeping the shapes that mark a **review output**, rejecting those that mark a
  **measurement** — `correct` is deliberately *not* a pattern (601 artifacts carry `energy_correct` /
  `n_correct`, which a re-run must be free to change), only the prose-shaped `correction` is.
* Rule 4 — a substrate/mode declaration may not be **weakened in place** (`live_gpu` → `cpu_training`,
  `real_model` → `synthetic_runner`, anything → `blocked`) without a note beside it. Values are
  principle-unwrapped first (`{"principle":…, "value":…}`, 162 artifacts), and an unrecognised
  substrate string ranks as *unknown*, never as *weak*.

**Calibrated against 1,200 commits of real history** (3,232 modified artifact pairs):

| rule | violations | commits refused |
|---|---:|---:|
| R1 stamp dropped (original) | 83 | 38 |
| R2 corrigendum dropped (original) | 93 | 41 |
| R3 marker dropped (new) | 10 | 9 |
| R4 substrate weakened (new) | 8 | 8 |

Every R4 hit is a genuine instance of the hazard class, and one of them was previously unknown:
`results/experiment_911_drift_probe_tier0i.json` took `real_model` → `synthetic_runner` **five
times**, and `experiment_307` took `live_gpu` → `cpu_training` **twice**, all landed in `main`, with
nothing in the repo watching. Across the 8 commits since the lint shipped: zero violations from
either new rule.

**New** (`scripts/test_suite_mutation_check.py`): the broad, shallow half. It answers "did this run
modify tracked files?" and names them, with no opinion about content — so it covers `openspec/` and
`output/`, which the lint does not. `--run` wraps a command; `--restore` puts everything back;
`--gate` is wired as a pre-commit hook that refuses while
`ops/.test_suite_mutation_pending.json` exists, so an unattended agent that runs the suite, sees the
record move, and then tries `git add -A && git commit` is stopped rather than publishing.

Both are pinned by tests that replay the three confirmed incidents as fixtures, including an explicit
test that the timestamp class (incident 3) is *deliberately* not the lint's job — that boundary is
what keeps the lint from refusing ordinary fail-forward re-runs, which is how a guard gets disabled.

---

## 8. Cross-references

* commit `b3e31d341` — the hazard as first diagnosed
* `scripts/determination_preservation_lint.py` — the content-level guard (widened here)
* `scripts/test_suite_mutation_check.py` — the file-level detector (new here)
* `tests/python/test_determination_preservation_lint.py`, `tests/python/test_test_suite_mutation_check.py`
* `ops/known-issues.md` 2026-07-29 — the operator-facing entry
* CLAUDE.md "Documentation Update Rules" (never-prune), "Adversarial Artifact Verification",
  "Inference-Substrate Declaration Discipline", "ARC Live-Path Reachability Discipline"
  (`solve_provenance`), "QA-Layer Authenticity Discipline" (principle-wrapped fields)
