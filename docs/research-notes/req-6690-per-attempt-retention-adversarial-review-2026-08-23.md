# Adversarial review: per-attempt world-model retention (REQ-ARC-WMTE-6690)

Date: 2026-08-23. Independent reviewer, no exposure to the builder's or the
verifier's reports. Reviewed commits `8e2f938a0f`, `580aa9f74f`, and the bulk of
the change, which landed inside conductor commit `99fdf2797e`.

## Verdict in one line

The fix is sound and the tests are real, not decorative. Five core mutations each
turn the suite RED. Four gaps are recorded below; none of them makes the shipped
behaviour worse than before the fix, and two are cheap to close.

## Method

Mutations ran in a hard-link sandbox (`cp -al`) with the two mutated files
unlinked from the repo, because another agent was editing
`arc_executable_world_model.py` during the review. The real working tree was
never modified. Confirmed after the review: no stray `attempts/` directory in
`results/arc_e3`, and no file dirtied by this work.

Mutations were re-verified against HEAD after that agent's commits landed, since
the file changed mid-review.

## What holds

Every mutation below turned the suite RED, so these assertions bite:

| Mutation | Result |
|---|---|
| drop the archive call in `_gen_to_file` | 4 failed |
| drop the archive call in `_write_world_model` | 5 failed |
| `dedup` forced False | 1 failed |
| kill-switch early-return removed | 1 failed |
| `_guard_engine_write(adir)` removed | 1 failed |
| manifest append removed | 7 failed |
| archive only when `note` is empty | 2 failed |

Producer-seam coverage is complete. Every path that writes the canonical
`world_model.py` reaches an archive call: `_gen_to_file` and `_write_world_model`
directly, `LocalGGUFProposer.induce`/`refactor` and the tool loop
(`arc_induction_tool_loop.py:472`) through those two, and `CodexProposer` post-hoc.

## Finding 1 (medium-high): per-game keying is not under test

Replace `adir = Path(E3_DIR) / game / "attempts"` with a hardcoded
`Path(E3_DIR) / "gme" / "attempts"` — the archive ignores its `game` argument
entirely — and all 16 tests still pass. Every test in the file uses the single
game name `gme`.

Why it matters if it regresses: all games' attempts land in one directory, and
content-hash dedup then applies across games. Two games that induce the same
trivial engine — and degenerate engines are a large share of the population, which
is the whole reason `arc_induction_quality.py` exists — would dedup against each
other. The second game's manifest line records `file: None`, so its attempt has no
retrievable file. That is the loss this REQ exists to prevent, shipping green.

Fix: one test that archives two different games and asserts the files land in
separate directories.

## Finding 2 (medium): the concept is narrower than the REQ title

The REQ is titled "Every Induced World Model Is Archived At Write Time". The body
is narrower and accurate: "nothing that **reaches the store** may be destroyed."
The implementation matches the body, not the title.

Measured missed input. In `LocalGGUFProposer.induce`'s split-induce fallback, the
engine half can succeed and the goal half then fail
(`arc_executable_world_model.py:8282-8283`). The engine source is discarded: never
written canonically, never archived. This is not a regression — it was lost before
too — but it is an induced model the retention does not retain.

Frequency, measured on the real corpus rather than assumed: 31 of 180 cells (17%)
in `results/arc_goal_defect_reask_ab_20260801/out/cells/` record
`split induce: goal failed`; 37 result files across `results/` mention it.

This also bounds commit `580aa9f74f`'s claim that "the unbiased population exists
on disk." The archived population is *engines that passed extraction and were
written*, not *everything the generator produced*. For measuring the degenerate
rate that distinction is harmless — degenerate engines do pass extraction — but the
population should be described as what it is.

## Finding 3 (medium): dedup keys on filename, so a crash leaves an unrepairable archive

`dedup = bool(next(iter(adir.glob(f"wm_*__{sha}.py")), None))` matches on name
only, and `write_text` is not atomic (no temp-plus-rename). A kill during the
archive write leaves a truncated file under the correct hash-named path.

Demonstrated: after truncating an archived file to 40 of 106 bytes, a second
identical induction reports `{'archived': True, 'deduplicated': True}` and does not
rewrite it. The manifest records `sha256_16: bd26175d3061f2d4`; the file on disk
hashes to `cc331052f5ef5e1c`. The engine is unrecoverable and the archive says it
succeeded.

Detectable after the fact — the manifest carries the hash — but nothing checks it.
Fix: `os.replace` from a temp file, or verify the hash on a dedup hit.

Realistic here: this environment produces `[conductor] Checkpoint: preserve
uncommitted work from interrupted run` commits regularly, so mid-write kills happen.
Disk-full is *not* this case — that raises `OSError`, which is caught and reported.

## Finding 4 (medium): the fail-open's visibility is nominal

The direction is right and clearly stated: fail closed on the test guard, fail open
after it. But the spec (line 27185) says the failure "SHALL be visible: recorded on
the proposer as `last_attempt_archive` **and counted**."

- No counter exists. `grep` for `n_archive|archive_failures` in the module returns 0.
- Nothing in `python/` or `scripts/` **reads** `last_attempt_archive`. Only tests do.

So on the scored path, an archive that fails on every single call is
indistinguishable from one that works. CLAUDE.md's own words: a guard that is
trusted and silent is the worst state in this system. The dev-side lever harness
does record `induction_archive` deltas per row, so a total failure is visible
*there* — but `E3AgentPolicy`, the scored path, has no such signal.

Minor, same area: `last_attempt_archive` is never initialised in `__init__`, so any
future consumer must use `getattr` with a default or it raises `AttributeError`
before the first induction.

## Finding 5 (low-medium): unbounded growth in a tracked directory, no pruning

`results/arc_e3/**/attempts/` is **not** gitignored — verified with
`git check-ignore`. The conductor commits with `git add -A`, so every archived
attempt becomes a permanent tracked file.

Scale, from measured inputs (mean `world_model.py` = 2,239 bytes; 40 attempts per
25-game run): about 87 KiB and 40 new files per run. At 20 runs/day that is roughly
620 MiB and ~290,000 new tracked files per year, into a tree where
`results/arc_e3` currently holds 54 tracked files and `results/` holds 23,232.

Byte growth is tolerable. File-count growth is the real cost, and there is no cap
or prune policy. This also sits against CLAUDE.md's "`results/arc_e3` is EVIDENCE —
read them, never write them": the live agent legitimately writes there, so the
archive is consistent with practice, but it mixes high-churn machine output into a
tree the project protects as curated evidence. Worth an explicit decision:
gitignore the `attempts/` subtree, cap it, or accept the churn on purpose.

## The zero-behaviour-change claim

**Substantially proven at the producer seam; not proven end-to-end on the scored
path.** Being precise about which:

Proven. `test_archive_on_off_equivalence` asserts identical `(ok, msg, canonical
bytes)` with archiving on and off. Reading the code supports it structurally: the
canonical `write_text` happens *before* the archive; the archive returns a value no
caller consumes; every exception after the guard is swallowed. The one call outside
the `try` — `_guard_engine_write(adir)` — can only raise when `PYTEST_CURRENT_TEST`
is set, which is never true on the scored path.

Not proven. No end-to-end `E3AgentPolicy` A/B with the archive on versus off. The
equivalence test covers `_gen_to_file` only, not the `_write_world_model`
split-induce seam. Residual real-world deltas are wall-clock (a mkdir, a glob, a
write, an append per induction) and the growing directory the glob scans.

Given the structure, I judge the residual scored-path risk small — but "small by
inspection" is a weaker claim than the commit message's flat assertion, and the bar
set elsewhere on this project is proof at the seam by mutation.

## Could not determine

- Whether the live Kaggle submission filesystem is writable at `E3_DIR`. If it is
  read-only, every archive fails silently and, per finding 4, nothing reports it.
- The real rate of mid-write kills behind finding 3. The mechanism is demonstrated;
  its frequency is not measured.
- No live GPU A/B was run — a verification run was in flight on GPU 1 and
  displacing it would have destroyed a measurement.
