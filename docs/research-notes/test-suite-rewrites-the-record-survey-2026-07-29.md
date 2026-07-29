# Running the test suite rewrites the research record — the survey

**Date:** 2026-07-29
**Author:** outer loop (unattended session; operator asleep)
**Status:** SURVEY COMPLETE for the stated scope — the whole-suite union is 41 tracked files, 28 of
them attributed to an exact test file (§3.3b). **NO TEST WAS CHANGED.** The repair is a design call
left to the operator — six options, costed, with a recommendation in §6.
**Revision 2 (2026-07-29):** an adversarial review found ten defects in the guards and this
document, two serious — rule 4 was refusing the project's own documented substrate convention, and
the pre-commit interlock was disarmed by the exact `pytest` invocation that caused both incidents.
All ten fixed; see **§7c**. Numbers, coverage claims and attribution labels corrected in place.
**Record integrity:** verified byte-identical, §5.
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

* **20 candidate test files** call `runpy.run_path` on a *literal* path naming an experiment script.
  (Only **2** name a path literally under `scripts/experiments/`:
  `test_experiment_3946_r11l_first_solve.py` and `test_experiment_3967_m3_honest_efficiency.py`.
  The hazard commit's "11 target `scripts/experiments/`" was a looser grep.)

  **"20" is what a literal-path grep can see, not the true total.** The enumeration matches a path
  string, so it is blind to a target built at runtime. At least three such sites exist:
  `tests/python/test_live_trace_memory.py:498`, `test_self_learning_replay.py:649` and
  `test_self_learning_replay_v2.py:688` all call
  `runpy.run_path(str(Path(<module>.__file__)), run_name="__main__")` on the real
  `scripts/experiment_222/223/241`, each of which writes tracked `results/*.json`. They are
  **empirically safe** — every one monkeypatches `CARNOT_REPO_ROOT` to a `tmp_path` repo first, and
  the scripts honour it (§6, option E) — so no finding in this document changes. What is wrong is
  the *exhaustiveness* claim: closing this class properly needs an AST or audit-hook enumeration,
  not a grep. The "~986 not surveyed" bullet in §4 covers these only at class level.
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

**Per-batch results — SUPERSEDED BY §3.3b, DO NOT READ AS COVERAGE.** The batch sweep was a
progressive attribution run that reached batch 02 of 13 and was then **abandoned in favour of the
whole-suite union run** in §3.3b, which covers all 3,366 files at once and is the evidence behind
§4's "Covered: all 3,366" line. Batches 03–13 were never run. The table is kept for the per-test
attribution it produced (below), not as a count of anything.

| batch | test files | tracked files moved |
|---|---:|---:|
| `expbatch_00` | 250 | 6 |
| `expbatch_01` | 250 | 16 |
| `expbatch_02` | 250 | 3 |
| **batches 00–02 only** | **750** | **25** |
| *batches 03–13* | *never run — superseded by §3.3b* | — |

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

### 3.3b THE MAPPING — the whole-suite union, 41 files, 28 attributed to an exact test

Running the hazard command in full (`pytest tests/python/test_arc_*.py
tests/python/test_experiment_*.py`, the configured `-n 4`) in a clean worktree leaves **41 tracked
files modified** — the hazard commit counted 39, and the composition matches. This is the answer to
"which tests rewrite which tracked files":

| tracked file rewritten | responsible test file(s) |
|---|---|
| `openspec/papers/paper-v6/section-6-limitations.md` | `tests/python/test_experiment_1911.py` |
| `output/kanele_synth/post_synth.dcp` | *unattributed — build-tool output, no experiment script in-tree* |
| `results/experiment_1035_dualgpu_rocm_v3.json` | `tests/python/test_experiment_1035_dualgpu_rocm_v3.py` |
| `results/experiment_1038_milestone_retro_80.json` | `tests/python/test_experiment_1038_milestone_retro_80.py` |
| `results/experiment_1081_fpga_scale_benchmark.json` | `tests/python/test_experiment_1081_fpga_scale_benchmark.py` |
| `results/experiment_1089_milestone_retro_84.json` | `tests/python/test_experiment_1089_milestone_retro_84.py` |
| `results/experiment_1103_milestone_retro_85.json` | `tests/python/test_experiment_1103_milestone_retro_85.py` |
| `results/experiment_1593_cdg_repair.json` | `tests/python/test_experiment_1593_cdg_repair.py` |
| `results/experiment_1747_ebt_mode_collapse_check.json` | `tests/python/test_experiment_1747_diagnostic.py` |
| `results/experiment_1750_huggingface_retry.json` | `tests/python/test_experiment_1750.py` |
| `results/experiment_1822_rtl_synth.log` | *unattributed — build-tool output, no experiment script in-tree* |
| `results/experiment_1911_phase4_canonical_decision.json` | `tests/python/test_experiment_1911.py` |
| `results/experiment_1938_nrgpt_loss_probe.json` | `tests/python/test_experiment_1938_nrgpt_loss_probe.py` |
| `results/experiment_1970_domino_fast_constraints.json` | `tests/python/test_experiment_1970_domino_fast_constraints.py` |
| `results/experiment_2085_pem_sudoku_eval.json` | `tests/python/test_experiment_2085_pem_sudoku_eval.py` |
| `results/experiment_2090_crane_humaneval.json` | `tests/python/test_experiment_2090_crane_humaneval.py` |
| `results/experiment_2097_eqm_eval.json` | `tests/python/test_experiment_2097.py` |
| `results/experiment_2510_ensemble_v7.json` | *unattributed — worker shard not flushed; **no subprocess in its script**, re-runnable with the existing plugin* |
| `results/experiment_2521_ensemble_v7.json` | `tests/python/test_experiment_2521_ensemble_v7.py` |
| `results/experiment_2538_kv260_sd_flash.json` | `tests/python/test_experiment_2538_kv260_sd_flash.py` |
| `results/experiment_2721_paper_v6_theory_update_v2.json` | `tests/python/test_experiment_2721.py` |
| `results/experiment_2758_weak_strong_policy_fix_v2.json` | `tests/python/test_experiment_2758.py` |
| `results/experiment_2824_cross_corpus_verifier_matrix.json` | *unattributed — worker shard not flushed; **no subprocess in its script**, re-runnable with the existing plugin* |
| `results/experiment_307_jepa_real_training.json` | `tests/python/test_experiment_307_jepa_real_training.py` |
| `results/experiment_3343_verifier_diversity_reaudit_after_axis_v3.json` | *unattributed — worker shard not flushed; **no subprocess in its script**, re-runnable with the existing plugin* |
| `results/experiment_3351_gatemate_latency_benchmark.json` | `tests/python/test_experiment_3351_gatemate_latency_benchmark.py` |
| `results/experiment_3386_fr11_nonforgetting.json` | *unattributed — worker shard not flushed; **no subprocess in its script**, re-runnable with the existing plugin* |
| `results/experiment_3394_kona_global_opt.json` | *unattributed — worker shard not flushed; **no subprocess in its script**, re-runnable with the existing plugin* |
| `results/experiment_3395_energy_based_replay.json` | *unattributed — worker shard not flushed; **no subprocess in its script**, re-runnable with the existing plugin* |
| `results/experiment_339_session_startup.json` | *unattributed — worker shard not flushed; **no subprocess in its script**, re-runnable with the existing plugin* |
| `results/experiment_3408_kona_global_opt.json` | *unattributed — worker shard not flushed; **no subprocess in its script**, re-runnable with the existing plugin* |
| `results/experiment_3420_kv260_terminal_latency_transcript_v1.json` | *unattributed — script shells out (`subprocess` x5), audit hook cannot see a child* |
| `results/experiment_3420_kv260_terminal_transcript.log` | *unattributed — script shells out (`subprocess` x5), audit hook cannot see a child* |
| `results/experiment_3843.json` | `tests/python/test_experiment_3843.py` |
| `results/experiment_3946_r11l_first_solve.json` | `tests/python/test_experiment_3946_r11l_first_solve.py` |
| `results/experiment_410_precision_live.json` | `tests/python/test_experiment_431_eorm_jepa_retrain.py` |
| `results/experiment_4162_sota_ingestion_verifier_moat_guidance.json` | `tests/python/test_experiment_4162_sota_ingestion_verifier_moat_guidance.py` |
| `results/experiment_4170_sota_ingestion_verifier_moat_guidance.json` | `tests/python/test_experiment_4170_sota_ingestion_verifier_moat_guidance.py` |
| `results/experiment_5794_hardware_terminal_action_receipt.json` | `tests/python/test_experiment_5794_hardware_terminal_action_receipt.py` |
| `results/experiment_5861_attached_board_state_receipts.json` | `tests/python/test_experiment_5861_attached_board_state_receipts.py` |
| `results/experiment_833_constraint_delta_root_cause.json` | *unattributed — worker shard not flushed; **no subprocess in its script**, re-runnable with the existing plugin* |

**28 of 41 carry an exact test file.** The 13 unattributed are the subprocess class (a CPython audit
hook cannot see a child process) plus a tail the last xdist worker had not yet flushed when the
survey was stopped. Attribution is merged from the whole-suite run's per-worker shards and the
per-batch attribution runs.

**Two entries deserve individual attention:**

* `results/experiment_410_precision_live.json` is written by
  `tests/python/test_experiment_431_eorm_jepa_retrain.py` — a test for a *different* experiment.
  Any repair or audit that pairs `test_experiment_N` with `experiment_N` by name will miss it.
* `openspec/papers/paper-v6/section-6-limitations.md` — a PAPER section — is written by
  `tests/python/test_experiment_1911.py`.

---

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

* All 20 test files that `runpy.run_path` an experiment script **via a literal path string** —
  individually, exact attribution. Sites that build the target at runtime are NOT in that 20; at
  least three exist and are enumerated in §3.1.
* All 125 `tests/python/test_arc_*.py` — individually, exact attribution.
* All 3,366 `tests/python/test_experiment_*.py` — via the **whole-suite union run of §3.3b**, which
  is the evidence behind this line. (§3.3's per-batch table covers only batches 00–02 and is
  superseded; see the note there.)

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
again after every survey process had been stopped by explicit PID. **Twelve tracked files differ, and
all twelve are this session's deliberate edits:**

```
.gitignore                                            scripts/determination_preservation_lint.py
.pre-commit-config.yaml                               scripts/pytest_write_audit_plugin.py
docs/research-notes/…-survey-2026-07-29.md            scripts/test_suite_mutation_check.py
openspec/capabilities/arc-world-model-trust-energy/   tests/python/test_determination_preservation_lint.py
ops/changelog.md                                      tests/python/test_pytest_write_audit_plugin.py
ops/known-issues.md                                   tests/python/test_test_suite_mutation_check.py
```

**No `results/`, `output/`, or operator-curated public doc changed.** The protected evidence
directories (`results/arc_e3`, `results/arc_logo_snapshot`, `results/arc_e3_origin_fixtures`) were
read only. `determination_preservation_lint.py --all` is clean across every tracked
`results/**/*.json`.

Nine artifacts were deliberately or accidentally rewritten during the investigation and every one was
restored and proven byte-identical against `HEAD` by sha256: exp3946 (dogfooding the detector),
exp1938 + exp2085 + exp4162 + exp4170 (reproducing §3.5), exp2427 + exp3351 + exp3734 ×2 (leaks
per §2/§3.4).

**A note on the verification itself.** The first attempt at the comparison above printed
"NONE — the research record is byte-identical", and it was wrong: the `diff | grep '^>' | awk` chain
read `$2` (the sha256 hash) where the path is `$3`, because the `> ` prefix shifts the fields. It
therefore grepped hashes for `^results/`, matched nothing, and reported all-clear on a check that had
verified nothing. It was caught only because a file this session *had* edited
(`openspec/…/spec.md`) was missing from the output. This is the same defect class the whole session
is about — a check that runs, passes, and inspects the wrong thing — and it is recorded here rather
than quietly fixed, because "the verification passed" is exactly the claim a reader has no way to
audit.

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
scripts already import it); a repo-level autouse fixture redirects it for the whole suite. One
helper change plus one fixture, and scripts that bypass the helper are findable mechanically (grep
for a literal `results/` open) rather than one at a time.

**Use the convention that already exists — do NOT invent a second one.** An earlier draft of this
option proposed a new `CARNOT_RESULTS_DIR` env var. That name occurs **0 times** in `scripts/`,
`tests/` or `python/`. The convention it describes is already in the tree and already works:
**`CARNOT_REPO_ROOT`**, read by a `get_repo_root()` helper, appears in **52 scripts and 56 test
files**. It is exactly why the three variable-target `runpy` sites in §3.1 rewrite nothing —
`tests/python/test_live_trace_memory.py:464` does `monkeypatch.setenv("CARNOT_REPO_ROOT",
str(repo))` and the script it runs honours it, so a full `runpy.run_path` of a real experiment
script writes into `tmp_path` instead of the record. That is a working, in-tree precedent for (E)
covering ~108 files, and proposing a competing variable name would fragment it. Read (E) as
*"extend `CARNOT_REPO_ROOT` to the scripts that do not yet honour it, and make the fixture
repo-wide rather than per-test."*

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

**Calibrated against real history. The window is stated explicitly because the first draft's was
not, which made its numbers unreproducible** — "1,200 commits" is ambiguous between *the last 1,200
commits* (3,232 modified artifact pairs) and *the last 1,200 commits that touch `results/`* (7,024
pairs). The R1/R2 rows below are from the first window; the R3/R4 rows were re-run over the second
(larger, more informative) window **after** the 2026-07-29 rule-4 repair described in §7c:

| rule | violations | commits refused | window |
|---|---:|---:|---|
| R1 stamp dropped (original) | 83 | 38 | 1,200 commits overall / 3,232 pairs |
| R2 corrigendum dropped (original) | 93 | 41 | 1,200 commits overall / 3,232 pairs |
| R3 marker dropped (new) | 10 | 9 | 1,200 touching `results/` / 7,024 pairs |
| R4 substrate weakened (new) | 12 | 12 | 1,200 touching `results/` / 7,024 pairs |

The same sweep run **before** the §7c repair reported **15** R4 hits. The three that no longer fire
are exactly the honest-substrate-correction class §7c fixed (exp5178, exp5161, exp5240) —
so the count going *down* is the guard getting more correct, not less protective.

**Per-hit adjudication** (replacing the first draft's blanket "every R4 hit is a real incident",
which was true only inside the narrower window):

* R4, 12/12 genuine — `experiment_911_drift_probe_tier0i.json` `real_model` → `synthetic_runner`
  (10×, previously unknown, nothing was watching), `experiment_307_jepa_real_training.json`
  `live_gpu` → `cpu_training` (2×).
* R3, 10/10 genuine — one dropped `flagged_adversarial: false` (rule 1 owns only the `true` case),
  and nine review notes (`duration_note`, `random_seed_note`, `action_sequence_note`,
  `step_accounting_note`, `registry_note`, `l6_partial_progress_note`) lost when fixed-path ARC
  round-probes (`results/outer_loop_fable5_*`) overwrote the previous round's record. That is the
  origin incident's own mechanism — a fixed output path plus a re-run — not a false positive.

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

## 7b. Validating the guards — the false-fire proof, and four checks that were not being tested

Section 7 describes what shipped. This section is the evidence that it works, and it is separate
because **three of the validations below failed on their first attempt in a way that looked like
success** — the same failure mode as the guard that did not fire.

### 7b.1 The must-not-fire proof, against the real analyser rebuild

The rule that matters most for adoption is not "does it catch the incident" but "does it stay quiet
during honest work". Re-running an analyser is the most common write to `results/` in this project;
a lint that refuses it gets switched off within a day.

Tested against the real thing rather than a fixture: commit `8441055c0` rebuilt **12 analyser
artifacts** (9 `outer_loop_*`, plus the card-ground-truth and reset-attribution passes). Their diff
moves exactly four things — `build_timestamp_utc`, `duration_s`, `provenance.git_head`, and the
`provenance.code[].sha256` / `.bytes` dependency hashes — and nothing else. The widened lint,
pointed at that historical pair:

```
files examined: 28  | analysers: 12  (both sides parsed on all 12)
WIDENED lint vs the real 12-artifact analyser rebuild -> OK -- NO false fire
```

**The first run of this check was worthless and looked fine.** It was executed from a worktree
checked out *at* `8441055c0` — which predates the widening — so it exercised the OLD two-rule lint
and reported a clean pass that proved nothing about the new rules. The tell was an `AttributeError`
on `_unwrap_principle` in a follow-up probe; without that accident the invalid result would have
been recorded as the proof. The rerun above imports the lint from the working tree and repoints
`REPO` at the historical worktree, which is what the module's own tests do.

Because a clean pass is also what blindness looks like, the same run injects violations into the
same real artifact and requires each to be refused:

| injected into `outer_loop_arc_max_actions_answer_20260726.json` | result |
|---|---|
| drop top-level `provenance` | **FIRES** — lost provenance declaration |
| drop `inference_substrate_principle` | **FIRES** — lost inference-substrate declaration |
| drop `verifier_is_oracle` | **FIRES** — lost circularity declaration |
| weaken `inference_substrate` → `sota_gguf_mock` | **FIRES** — WEAKENED, `aggregation…` → mock |
| *control:* change `duration_s` | silent (correct — fail-forward) |
| *control:* bump `git_head` + `build_timestamp_utc` | silent (correct — that IS the rebuild) |

Both halves are pinned (`test_a_legitimate_analyser_rebuild_is_not_refused` and its companion
`test_the_analyser_rebuild_fixture_is_not_silently_unprotected`) — the second exists precisely
because the first would keep passing if the lint stopped seeing the artifact at all.

### 7b.2 Mutation testing found two lint checks that no test was exercising

Nineteen mutations, each disabling exactly one behaviour of one new check, each requiring some test
to go red. First pass: **17 of 19 killed, 2 survivors** — both genuine gaps, not harness artifacts:

* **Deleting the `correction` marker pattern left the suite green.** The incident-1 test names
  `inference_substrate_correction_note` and `inference_substrate_original_invalid_value`, but both
  are *also* matched by `^inference_substrate`, and its `solve_provenance*` fields by `provenance`.
  Every field in the test that the `correction` pattern is named for was double-covered, so the
  pattern itself was never under test. A bare `correction_note` / `data_correction` was unprotected.
* **Blanking half the rule-1/rule-2 dedup left the suite green.** The existing
  `test_one_deletion_produces_one_refusal_line` pins the dedup for `flagged_adversarial` only, so
  the half that stops rule 3 re-reporting the *corrigendum* family was unpinned; a corrigendum
  deletion would have produced two refusal lines naming the same fields.

Both were confirmed by hand, then closed with a test each. Both gaps are the *same shape* as the
original bug — coverage that exists on paper and does not bite. Neither would have been found by
reading the tests: both tests genuinely pass and genuinely assert something, they just were not
asserting the thing their names implied.

### 7b.3 The detector ate this document, twice — and the fix is recoverability, not attribution

While validating the above, a wrapped full-suite run (`--restore --run`) reported 6 tracked files
modified and restored all 6. One of them was **this file**. The §7b section had been written
*after* the snapshot was taken, so `mutations()` correctly classified it as "changed since the
baseline", attributed it to the run, and `git checkout --`'d it away. The section had to be
rewritten from scratch.

That is the **second** occurrence — §3.6 records the first, also this document, also mid-edit. Twice
is a design defect, not user error. The root cause is not fixable by better attribution: *the
detector cannot tell a test's write from a human's concurrent edit*, because at the file level they
are identical events. Anything cleverer would be a heuristic guessing at authorship.

So the fix makes the mistake **survivable** instead of trying to prevent it. `restore()` now copies
every file's pre-revert content to `ops/.test_suite_mutation_backup/<path>` (gitignored) before
reverting, prints where they went, and warns if any could not be saved. `git checkout --` is
unrecoverable for uncommitted content; a `cp` first turns a destroyed afternoon into a restored
file. Pinned by `test_restore_backs_a_file_up_before_reverting_it`, which asserts the backup holds
the **pre-revert** content — the committed version was never at risk, it is already in git.

The operational rule, now stated in the module docstring: **do not edit tracked files while a
`--restore` run is in flight.**

That run is also the detector's first real proof in anger: it caught 6 rewrites mid-suite —
including `output/kanele_synth/post_synth.dcp`, one of the original 39 — restored all 6, and the
four in-flight files in the snapshot baseline were verified byte-identical afterwards.

### 7b.4 Final state

**22 of 22 mutations killed** across both guards (13 lint, 9 detector), including three for the new
backup behaviour. Guard test suites: **41 passed**, and the run leaves no `ops/.test_suite_mutation_backup/`
behind in the real tree — the fixture repoints `BACKUP` at the throwaway repo, because a test suite
that litters the tree it is checking is the exact hazard this module exists to detect.

---

## 7c. Review repairs — the guards were refusing honest work, and the interlock was disarmed

An adversarial review of §7's guards found ten defects. Two were serious enough to invert a guard's
purpose. All are fixed; each is pinned by a regression test named for its incident.

### 7c.1 SERIOUS — rule 4 refused the project's own documented substrate convention

`adversarial_verify.py` documents `<canonical value><separator><human note>` and strips the note
before matching. `_strength_rank` did not: it scanned the **whole** string, including the prose, and
took the minimum band across everything it found. **233+ live declarations use that form.**

The failure is the negation-blindness class CLAUDE.md's QA-Layer Authenticity Discipline names — a
checker confusing "did X" with "explicitly did NOT do X". exp5178 declares:

> `live_llm_embedding_extraction; Substrate corrected 2026-07-03: … no iterative token-by-token
> generation. The original declaration ('live_llm_inference') implied full generative inference
> (60s floor) … verifier_ensemble_against_cached_candidates also does not fit (its definition
> explicitly requires the LLM NOT be loaded, and this task did load it, for embeddings).`

The leading token is band 3 — a real GGUF load. But the note contains `cached`, so the whole-string
scan ranked it band 2 and the lint refused the commit **while asserting the opposite of what the
artifact says**. exp5161 is the same shape. Both are CLAUDE.md's own named exemplars for their
substrates, so the rule was refusing the convention the project documents.

Fixed by delegating the match to `adversarial_verify._match_declared_substrate` rather than
re-deriving separator handling — a drifted copy of a matcher is exactly how this bug arose.
`_has_change_note` now also accepts a rationale carried **inline** in the value, which is where the
most carefully documented corrections in the corpus put it.

**Verified corpus-wide, not just on the two exemplars:** of all 2,673 `inference_substrate`
declarations in the tree, **0** now rank differently from their bare leading token.

### 7c.2 SERIOUS — the pre-commit interlock was disarmed by the exact invocation that caused both incidents

`--gate` refuses a commit while `ops/.test_suite_mutation_pending.json` exists — and **only `--run`
ever wrote that marker**. So the interlock protected the wrapper nobody uses and was silent for:

```
pytest tests/python/test_arc_*.py tests/python/test_experiment_*.py
```

Demonstrated on the real tree: after simulating that rewrite class, `README.md` (operator-curated,
one of the original 39) and `openspec/papers/paper-v6/section-6-limitations.md` were both modified
while `--gate` exited 0 **and** the determination lint printed OK.

Fixed at the source rather than at the wrapper: `tests/python/conftest.py` takes a `git status`
baseline in `pytest_configure` and calls `test_suite_mutation_check.arm_from_pytest()` in
`pytest_sessionfinish`, so the marker is armed **however pytest was invoked**. Proven end-to-end: a
green single-test run that rewrites `README.md` now emits a `PytestWarning`, writes the marker, and
`--gate` exits 1. The tree was restored and verified byte-identical.

**Stated limitation, observed while validating it:** a run that is *killed* never arms.
`pytest_sessionfinish` does not run under SIGTERM/SIGKILL, so a suite that times out or is
interrupted can leave rewrites on disk with no marker — a 2-minute timeout killing a full
`test_arc_*.py` run produced exactly that (it had moved nothing, but the gap is real). `--run` does
not share the gap, since it checks after the child exits however it exited. This is a reason to keep
`--run` for long jobs, not a reason to distrust the hook: the hook covers the *bare pytest* case
that previously had no coverage at all.

Three properties are pinned because each could quietly undo it: it skips xdist **workers** (only the
controller may arm, or four partial views race); it **disarms** on a clean run (or one bad run blocks
commits forever); and it **never restores** (auto-reverting has already destroyed in-flight work
twice — see §7b.3). Wiring `--check` into pre-commit instead was rejected: with no baseline it cannot
distinguish a test's rewrite from a human's edit, and would refuse every commit on a dirty tree.

### 7c.3 The other eight

* **Rule 4 invented a phantom LIVE band from a non-substrate name.** exp5240 declared
  `arc_live_path_patch_synthesis` — the ARC live *code* path, never a legal enum value. A token scan
  reads its `live` token as LIVE/HARDWARE, so an honest later correction to
  `aggregation_from_upstream_artifacts` was refused as a downgrade. Fixed structurally: for the
  enum-governed field, an unrecognised name may rank **weak** but never **strong**. A claim of
  strength is cheap to make by accident; an admission of weakness (`mock`, `blocked`) is not, so
  admissions are trusted from any string and claims only from the documented vocabulary. That
  asymmetry matters — a first attempt returned `None` for *all* non-enum values and silently
  un-protected `sota_gguf_mock`, CLAUDE.md's own fabrication exemplar. Two pre-existing tests caught
  the regression, which is what they are for.
* **The calibration did not reproduce, and its strongest claim did not generalise.** "1,200 commits"
  was ambiguous between two populations. Both are now named, the sweep re-run, and the blanket
  "every R4 hit is a real incident" replaced with per-hit adjudication — see §7's table.
* **Four review-output markers were uncovered:** `n_samples_justification` (56 artifacts — the
  sample-size disclosure CLAUDE.md *requires* for a distributional claim), `false_negative_risk_checked`
  (63), `paper_v6_forbidden_claims` (22), `adversarial_verify_flags` (22). Added. `honest_verdict`
  (5,245) remains deliberately excluded and is now pinned by a test, because a re-run legitimately
  produces a new one.
* **A supporting census figure was wrong:** the `correct`-exclusion rationale cited 601 artifacts;
  the stated predicate yields **465**. The design decision is unaffected — those are measurements
  either way — but an unreproducible number is not evidence.
* **Three doc claims overstated their evidence** and are corrected in place: the 13 unattributed rows
  in §3.3b were all labelled "subprocess writer" when only **2 of 13** contain a subprocess (§3.3b);
  §4's "All 20 `runpy` files" is what a *literal-path grep* can see, and misses at least 3
  variable-target sites (§3.1); §3.3's batch table stopped at batch 02 of 13 and is marked superseded.
* **Option (E) proposed a variable that does not exist.** `CARNOT_RESULTS_DIR` occurs **0** times in
  the tree; the convention it describes already exists as `CARNOT_REPO_ROOT` (52 scripts, 56 test
  files) and already works. §6 now cites the working precedent instead of inventing a competitor.

### 7c.4 Mutation testing, again — and again it found an unprotected check

10 mutations of the new logic: **9 killed, 1 survivor.** Reverting token anchoring to a bare
substring test left the suite green — the anchoring this document claims was entirely untested. It
matters because the rank is a *minimum*: `uncached_live_gguf_inference` unanchored matches `cached`
and ranks band 2; `unblocked_live_gpu_run` matches `blocked` and ranks band 0. Both are the same
negation-blindness class as §7c.1, now pinned. Final: **10/10 killed, 56 tests passing** across the
two guard suites.

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
