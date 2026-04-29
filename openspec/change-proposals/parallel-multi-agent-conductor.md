# Parallel Multi-Agent Conductor — Cross-Backend Worktree Isolation

**Status:** Draft change proposal, ready for .82 mandatory pickup.
Companion to `multi-agent-routing.md` (per-task agent_type) and the
`huggingface-spaces-sudoku-demo.md` / `wopr-games-gallery-extension.md`
proposals (which together specify the WOPR-cartridge workload most
in need of parallel execution).

**Origin:** 2026-04-29 evening. Today's milestone .80 took ~24 hours
end-to-end with single-stream serial execution; the .81 milestone is
following the same pattern. With Claude Max 20×, Codex Max 20×
(gpt-5.5), and Gemini Ultra subscriptions running on independent
quota pools, single-stream execution leaves 2/3 of the available
agent throughput idle.

**Priority:** **HIGH** for .82. Without this, the WOPR-games-gallery
cartridge sprint stretches from ~1 week to ~3 weeks of wall-clock
time. The Sudoku v1 demo + WarGames + Lights Out MVP target of
2026-05-08 + position paper preprint target of 2026-05-15 *both
depend* on the cartridge timeline that this proposal compresses.

## Problem statement

The conductor's `research_step()` runs **one** subprocess per
iteration via `run_agent()`. Even with today's per-task routing
(`agent_type: claude|codex|gemini` from `multi-agent-routing.md`),
two tasks targeting different agent backends still execute
*serially*. This wastes available throughput across three
independent quota pools:

- **Claude Max 20×** — own rate limit
- **Codex Max 20×** (gpt-5.5) — own rate limit
- **Gemini Ultra** — own rate limit

Empirical evidence from .80 + .81:
- Each research task: 5–15 min Sonnet/Opus subprocess
- Each pre-test self-heal: 5–50 min
- Single-stream cumulative: 12 tasks × ~15 min/avg ≈ 3 hours of
  active agent time, but spans 24 hours wall-clock due to retry
  loops + planning + retro overhead
- Three-way parallel target: same work in ~8 hours

The .82 mandate already specifies WOPR-games-gallery cartridges
that route to Codex while architecture/research work routes to
Claude. **Today, those would run serially.** This proposal makes
them concurrent.

## Solution: three-tier rollout

### Tier A — Dual-conductor (Claude + Codex)

The minimum viable parallel execution: two conductor instances, each
in its own git worktree, each driving its own AGENT_TYPE.

**Worktree layout:**

```
/home/ianblenke/github.com/ianblenke/carnot/                 # main worktree
/home/ianblenke/github.com/ianblenke/carnot-codex/           # codex worktree (NEW)
```

Each worktree is a real `git worktree add` checkout sharing the same
`.git/` directory. Disjoint working trees → no file-edit races during
agent runs. `git fetch` + `git rebase` synchronizes after each task.

**Task partitioning** (planner-driven via existing `agent_type`):

```yaml
# In research-roadmap.yaml — task partitioning by agent_type:
- id: expNNNN-zenil-alpha-fr11
  agent_type: claude       # → main worktree, claude conductor
  worktree: main           # NEW field (defaults to main)

- id: expNNNN-wopr-games-lights-out-cartridge
  agent_type: codex
  worktree: codex          # NEW field — runs in codex worktree
  model: gpt-5.5
```

**Two `systemctl --user` instances:**

```bash
# Claude conductor (existing, unchanged):
systemd-run --user --unit=carnot-conductor \
  --working-directory=/home/ianblenke/github.com/ianblenke/carnot \
  --setenv=AGENT_TYPE=claude \
  --setenv=PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/research_conductor.py \
  --loop --interval 10 --in-process-docs --async-doc-recon \
  --worktree main

# Codex conductor (NEW):
systemd-run --user --unit=carnot-conductor-codex \
  --working-directory=/home/ianblenke/github.com/ianblenke/carnot-codex \
  --setenv=AGENT_TYPE=codex \
  --setenv=AGENT_MODEL=gpt-5.5 \
  --setenv=PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/research_conductor.py \
  --loop --interval 10 --in-process-docs --async-doc-recon \
  --worktree codex
```

**Per-worktree state isolation:**

The conductor's state files (`ops/conductor-heartbeat.json`,
`ops/conductor-state.json`, `ops/conductor-log.md`) live in the
worktree. A new `--worktree NAME` CLI flag suffixes them:

- Claude: `ops/conductor-heartbeat-main.json` (default = `main`)
- Codex: `ops/conductor-heartbeat-codex.json`

Backwards-compatible default: when `--worktree` is unspecified or
`main`, the existing filename is used.

**Per-worktree pre-test cache:**

Pre-test results cached per-worktree so the codex conductor doesn't
re-run pre-tests the claude conductor just passed. Cache invalidated
on commit/merge events affecting the test files.

**Merge-back protocol:**

After each task completes (deliverable written + commit landed in
worktree branch), the conductor runs:

```bash
git fetch origin main
git rebase origin/main      # rebase task commit onto current main
git push origin <branch>    # push to a remote branch
# main worktree's conductor watches for these and fast-forwards
```

A periodic synchronizer in the main worktree pulls task-branch
commits and fast-forwards `main` when conflicts are absent. On
conflict (rare for disjoint task scopes), surface to operator
via `ops/parallel-merge-conflicts.log`.

### Tier B — Gemini worktree added

Once Tier A proves out (~1 week empirical), add a Gemini worktree
for long-context audits:

```
/home/ianblenke/github.com/ianblenke/carnot-gemini/   # gemini worktree
```

Used for: failure-ledger pattern detection, architecture coherence
audits, multi-paper literature synthesis. These tasks run rarely
(weekly?) but benefit from 1M-context analysis.

### Tier C — Within-backend parallelism (LATER)

If Tier A+B prove the model, extend to running multiple tasks within
the *same* worktree concurrently when their file scopes don't
overlap. Requires:

- Per-task `modifies:` YAML field declaring which files each task
  edits
- Pre-flight scope-disjointness check: only schedule task pair (A, B)
  concurrently if `A.modifies ∩ B.modifies = ∅`
- Per-task lock files via flock on each declared path

Higher complexity; defer until empirical evidence shows Tier A+B
isn't enough.

## Schema extension

Add a `worktree` field to `ResearchTask` (orthogonal to `agent_type`):

```python
class ResearchTask(BaseModel):
    # ...existing fields including agent_type from multi-agent-routing...
    worktree: Literal["main", "codex", "gemini"] | None = None
```

Default `None` → routes to the conductor's startup `--worktree`
arg (typically `main`). Explicit `worktree: codex` → runs only on
the codex conductor instance.

**Cross-field validator:** `worktree` should be consistent with
`agent_type`. Set at planner time; schema validator warns on
mismatch (codex worktree + claude agent_type is permitted but
unusual).

## Test scaffolding

Tests for parallel-conductor behavior:

```python
def test_worktree_field_default_is_none():
    """Task without worktree falls through to startup --worktree arg."""

def test_worktree_field_accepts_main_codex_gemini():
    """Three valid worktree values for Tier A+B."""

def test_worktree_field_rejects_unknown():
    """Typos and unsupported worktree names fail validation."""

def test_worktree_yaml_round_trip():
    """worktree=codex + agent_type=codex YAML survives round-trip."""
```

## Cost-benefit

### Engineering investment

- **Tier A (week 1 of .82):** ~2-3 days
  - Schema field + tests
  - `--worktree` CLI flag
  - Per-worktree state file suffixing
  - Merge-back protocol scaffolding
  - Two systemd-run unit launchers
  - First worktree creation: `git worktree add ../carnot-codex codex-worktree`

- **Tier B (week 2-3):** ~1-2 days
  - Third worktree
  - Schema Literal extension to include "gemini"
  - Per-worktree pre-test cache invalidation refinements

- **Tier C (later):** ~3-5 days
  - Per-task `modifies:` field
  - Scope-disjointness checker
  - Per-path flock infrastructure

### Benefit (concrete)

For .82 specifically:

- **WOPR-games-gallery cartridges (~10 cartridges):**
  - Serial: ~2-3 weeks (1-2 days/cartridge, single-stream)
  - Tier A parallel (Codex worktree dedicated): ~0.5-1 week
  - **2× speedup** on cartridge shipping

- **Position paper drafting + cartridge shipping:**
  - Serial: cartridges block paper writing (same operator attention)
  - Parallel: paper drafted on Claude main while Codex worktree ships
    cartridges
  - **Both lands within ~1.5 weeks** instead of ~3-4 weeks serial

For ongoing milestones (.82+):
- Heterogeneous milestones (mix of Claude research + Codex code +
  Gemini audits) compress 24-hour wall-clocks to ~8 hours
- **3× ongoing speedup** for typical research-heavy milestones

### Real risks

1. **Merge conflicts.** Rare for disjoint task scopes (cartridges
   modify only `spaces/wopr-games/games/*.py`; architecture work
   modifies `python/carnot/`). Will occasionally happen on shared
   files (`ops/changelog.md`, `_bmad/architecture.md`). Mitigation:
   merge-conflict log surfaces these; operator resolves manually.

2. **State-file race conditions.** Per-worktree state isolation
   solves this for *separate* conductors but *not* for the C+E
   escalation pattern, which writes to a shared log. Mitigation:
   adopt fcntl-based append for `ops/conductor-log.md` (or use a
   per-worktree log that the synchronizer concatenates).

3. **Today's run_agent timeout bug** (the 49-min Sonnet hang we
   killed at 16:23Z). Bug exists in single-stream conductor; doesn't
   get worse with parallelism but doesn't get fixed either. Should
   address separately.

4. **Conductor self-protection guard** (RETRO-067 reverter on
   `scripts/research_conductor.py` edits) interacts poorly with
   parallel worktrees writing to the conductor itself. Mitigation:
   only the main worktree edits `scripts/research_conductor.py`;
   subordinate worktrees treat it as read-only.

## prior_failures

```yaml
prior_failures: []  # Genuinely new feature; no prior failed attempts.
                    # Builds on multi-agent-routing.md (today's
                    # commit aa3c2707) which formalized the per-task
                    # agent_type field.
```

## Acceptance criteria

1. Schema validator (`scripts/roadmap_schema.py`) declares
   `worktree: Literal["main", "codex", "gemini"] | None = None`.
2. Test coverage in `tests/python/test_roadmap_schema.py`:
   - Default None case
   - Each of the three valid worktree values
   - Rejection of unknown names
   - YAML round-trip
3. Conductor (`scripts/research_conductor.py`):
   - `--worktree NAME` CLI argument added
   - Per-worktree state file suffixing
   - `pick_next_task()` filters by `task.get("worktree", "main")`
     matching the conductor's current `--worktree`
   - Merge-back hook after each successful task
4. Two systemd-run unit launchers documented in `_bmad/architecture.md`:
   - `carnot-conductor` (main, claude default)
   - `carnot-conductor-codex` (codex worktree, AGENT_TYPE=codex)
5. Operator-facing documentation in `ops/parallel-multi-agent-conductor.md`:
   - How to launch the second conductor
   - How to monitor both
   - How to resolve merge conflicts
6. .82 milestone planner output uses `worktree: codex` on at least 3
   WOPR-cartridge tasks (test-driven validation of the routing).

## Strategic alignment

This proposal directly enables the .82 milestone's coordinated
launch:

- **2026-05-08 target:** WOPR Sudoku demo + WarGames parody +
  Lights Out MVP go live on HuggingFace Spaces
- **2026-05-15 target:** position paper preprint on arXiv
- **2026-05-22 target:** Twitter/HN announcement combining theory
  + demo + first 5 cartridges

Without parallel-multi-agent-conductor, the cartridge timeline
slips to ~3 weeks and pushes both targets back. With it, the
launch lands as planned.

It's the **sixth** operator-attention-reduction infrastructure
proposal in the recent series:

1. `conductor-supervisor.md` (.81 mandatory — landed)
2. `roadmap-schema-validation.md` (.81 mandatory — landed)
3. `conductor-fastpath-bootstrap-skip.md` (.81 mandatory — landed)
4. `differential-agent-routing.md` (.81 mandatory — landed)
5. `multi-agent-routing.md` (today — landed `aa3c2707`)
6. **`parallel-multi-agent-conductor.md`** (this proposal — .82 mandatory)

## Out of scope

- **Multi-host execution.** Running worktrees on different machines
  is a future-work concern. This proposal targets single-host
  parallelism.
- **Adversarial-aware merge resolution.** Conflicts are assumed
  cooperative (planner partitions tasks correctly). Adversarial
  scenarios (planner generates conflicting task pairs) are out of
  scope.
- **Real-time scheduling optimization.** Static planner-time
  partitioning is sufficient. Dynamic load-balancing across
  worktrees is Tier C+ future work.
- **Within-backend parallelism.** Documented as Tier C; deferred
  until empirical evidence shows Tier A+B is insufficient.

## Estimated effort

- Tier A (week 1 of .82): 2-3 days
- Tier B (week 2-3): 1-2 days
- Tier C (later): 3-5 days

Total for Tier A+B (covering all near-term needs): ~5 days of
focused engineering, recoupable inside the first compressed
milestone.
