# Story: REQ-ARC-WMTE-6642 amendment - The eval-run consumer lint works from a worktree

**Status:** Completed 2026-09-05
**Spec:** `openspec/capabilities/arc-world-model-trust-energy/spec.md`,
SCENARIO-ARC-WMTE-6642-WORKTREE-CORPUS and SCENARIO-ARC-WMTE-6642-CORPUS-ABSENT-SKIP.
**Goal:** The pre-commit hook `eval-run-consumer-field-lint` must not refuse a
`scripts/*.py` commit only because the checkout is a git worktree.
**Rationale:** The corpus `results/arc_leaderboard_eval_runs/` is gitignored. A
worktree has none. The lint failed closed on the missing directory, so every
agent in a worktree was refused. A refused commit leaves work staged in a shared
index, where the next `git add -A` sweeps it under an unrelated message.

## Stories
- [x] Reproduce the refusal in a worktree, directly and through `pre-commit run`
- [x] Add the two scenarios and the fail-closed amendment to the spec
- [x] Write the tests first: a real `git worktree add` fixture, the skip-loudly
  case, the superset property, the two new fail-closed conditions, the OK line
- [x] Implement `main_checkout_root`, `resolve_runs_dir`, the loud skip
- [x] Prove every rule by mutation at the call site: RED, byte-identical restore, GREEN
- [x] Verify end to end: the hook passes in this worktree and names the main corpus

## Decision
Fail-closed stays for everything that means "could not look". The artifact
corpus is a pass-widener, so its absence is a loud skip, not a failure. See the
lint docstring for the one-sentence proof and the test that pins it.
