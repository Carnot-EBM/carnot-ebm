# Preserved worktree state (2026-09-19 stale-branch cleanup)

During a stale-branch/worktree cleanup (24 abandoned branches, reclaiming
gitea/github storage bloat from an unrelated large-blob history rewrite the
same day), 20 branches were investigated. 17 were confirmed to have zero
content not already reflected on `main` (verified via `git cherry`, not just
commit-message inspection -- two branches initially flagged as "has real
unmerged work" by an earlier pass turned out, on closer inspection, to have
their only interesting-looking commits already patch-matched on `main`; their
"unique" commits by `git cherry` were a shared block of old 2026-06-01 to
2026-09-04 conductor task-prompt commits, not branch-specific content). Those
17 were deleted (local + both remotes).

Three branches had **real uncommitted work** sitting in their worktrees --
never committed anywhere, so `git cherry` could not see it at all. That work
is exported here before the branches/worktrees were deleted, so it is not
lost. None of it has been reviewed or integrated into `main` -- it is
preserved in its original, unreviewed, experimental state.

## outer-loop-a2-deeper-levels-*

From branch `outer-loop/a2-deeper-levels` (worktree `~/carnot-wt-a2`, last
active 2026-06-21). Uncommitted diff to `python/carnot/agentic/
arc_executable_world_model.py` and `results/arc_e3/lp85/world_model.py`
(`outer-loop-a2-deeper-levels-uncommitted.diff`, `-status.txt` for the raw
`git status --short`). Untracked experiment dirs `results/arc_e3/
lp85_fix_coder_16384/` and `_32768/` copied whole into
`outer-loop-a2-deeper-levels-untracked/` (their own `.pyc` cache stripped).
An untracked `environment_files/` dir was present but empty (the ARC SDK's
own gitignored cache) -- not preserved, nothing in it.

## outer-loop-bp35-diag-*

From branch `outer-loop/bp35-diag` (worktree `~/carnot-wt-diag`, last active
2026-06-24). The branch's own tip commit already records this as a dead end:
"KILLED — progress/value signals do NOT carry L1->L2". Uncommitted diff to
`results/arc_e3/lp85/world_model.py` plus five untracked
`proto_multilevel_diag*.json` result files, copied to
`outer-loop-bp35-diag-untracked/`. Low priority given the recorded negative
finding, exported for completeness only.

## outer-loop-multiwin-goal-*

From branch `outer-loop/multiwin-goal` (worktree `~/carnot-wt-multiwin`, last
active 2026-06-24). Branch tip: "Lever #4 align-offset template + GATE A:
built; grid-recoloring obstacle found". Uncommitted diff to `results/arc_e3/
lp85/world_model.py` and `scripts/experiments/proto_multiwin_goal.py`, plus
one untracked `results/proto_multiwin_goal.json`, copied to
`outer-loop-multiwin-goal-untracked/`.

## Why these weren't just committed instead

All three are unreviewed experimental state from abandoned sessions, months
old, with no test coverage or verification run against them. Committing
untested diffs to a live-ARC-path file (`arc_executable_world_model.py`, on
the scored live-agent's import path) without review would violate this
project's own discipline on that file. Exporting preserves the option to
review and selectively integrate later without losing the material now.

## Branches deleted in this pass (confirmed zero unique content on `main`)

`grammar-27b-trial2`, `worktree-agent-ababac7cb1a3e50b8`, `gemini-worktree`,
`outer-loop/astra-state-persistence`, `outer-loop/kv-persistence-confirm`,
`worktree-agent-a092d79bdf0704db3`, `worktree-agent-a3741d26a3e5591bd`,
`worktree-agent-a4a99d96ace0bec57`, `worktree-agent-a78d7102eb43d0feb`,
`worktree-agent-a89647aea78a76a58`, `worktree-agent-a9699853b6e22301b`,
`worktree-agent-ad2b64c85092d00d3`, `worktree-agent-afd77ea2887b9da58`,
`worktree-agent-aff05cc9b49c41da6`, plus the three above whose uncommitted
diffs are preserved here (`outer-loop/a2-deeper-levels`, `outer-loop/bp35-diag`,
`outer-loop/multiwin-goal`).

## Branches NOT touched (live)

Three worktrees remain locked and untouched: `worktree-agent-a385887308776466e`,
`worktree-agent-a83acaaa5293c1b8f`, `worktree-agent-ac986fe334205e8c2` -- all
held by a running `claude --resume` session (PID 3102595, started
2026-09-04), confirmed alive via `/proc/3102595` at cleanup time.
