# Pending fixes — written, verified, and NOT landed

A patch here is finished work that a mechanical gate refuses for a reason unrelated to its own
correctness. It is committed rather than held in a session scratchpad so it survives the session,
and so the reason it did not land is auditable next to the change itself.

**This is not a TODO list.** Nothing belongs here that has not been written and tested. A patch
here should apply cleanly (`git apply --check`) and carry its own regression test.

## 2026-07-31-induce-budget-single-source.patch

**What it fixes.** `_INDUCE_DEFAULT_MAX_TOKENS` (`arc_executable_world_model.py`) documents itself
as the single source for the induce/refactor completion budget across four sites, and
`_default_induce_n_ctx()`'s docstring claims "both halves now come from the same arithmetic".
Neither was structurally true: the constant had exactly ONE reader in the tree (the context-pool
derivation) and the other three sites were independent hardcoded `4096` literals
(`LocalGGUFProposer.max_tokens`, and the `CARNOT_ARC_INDUCE_MAX_TOKENS` fallback at each of the two
construction sites in `arc_competition_agent.py`). The claims held only via the ENV path: exporting
the env var did move both halves, while editing the constant in source grew the llama-server
context pool and left the completion budget alone — allocating KV cells for tokens nothing would
ever request, silently. Same drift class as REQ-ARC-FCP-5699-35's recorded incident, one level
down.

**What it contains.** The four sites reading one name (value UNCHANGED at 4096 — this is plumbing,
not a budget raise), a mutation-verified regression test
(`tests/python/test_arc_induce_budget_single_source_2026_07_31.py`; 2 of its cases fail against the
pre-fix code, checked by reinstating the literals), and `REQ-ARC-FCP-5699-40` with four scenarios.
`ruff`, `ruff format` and `mypy` clean; the 126 generator-adjacent tests pass with it applied.

**Why it did not land.** `artifact-freshness-lint` refuses any commit that leaves a registered
analyser-produced artifact stale, and editing either agentic module marks **12** artifacts stale.
Discharging that needs 8 rebuilds plus 4 `provenance.freshness_acknowledgements`. One of the 8 is
`arc_per_level_reset_attribution_capture.py --games ... --budget 400` — a LIVE capture, not an
analysis, whose re-run would replace published numbers with fresh nondeterministic ones. Three more
are GPU experiment re-runs (`experiment_6011/6012/6013`). Paying that price to land a
comment-and-constant change, unattended, is not proportionate, and replacing published measurements
as a side effect of a plumbing fix would be a correction owed rather than a formality.

**To land it.**

```
git apply ops/pending-fixes/2026-07-31-induce-budget-single-source.patch
cp ops/pending-fixes/test_arc_induce_budget_single_source_2026_07_31.py.deferred \
   tests/python/test_arc_induce_budget_single_source_2026_07_31.py   # if the patch's new-file hunk is skipped
python3 scripts/artifact_freshness_lint.py      # lists the 12 and their rebuild commands
```

Then either rebuild the 8 and diff each (the lint's own instruction: "report exactly which numbers
moved"), or — if you accept that a comment-and-constant change cannot move a number — add a
`freshness_acknowledgement` to each of the 12 pinning `sha256_now`/`sha256_was`/`reason`/`evidence`.
The acknowledgement path is sanctioned but deliberately expensive to abuse, and the lint's own
docstring says to prefer rebuilding where rebuilding is possible.

**Context:** `ops/known-issues.md` 2026-07-31 ·
`docs/research-notes/arc-induce-completion-budget-2026-07-31.md` ·
`results/outer_loop_arc_induce_budget_phase1_20260731.json`
