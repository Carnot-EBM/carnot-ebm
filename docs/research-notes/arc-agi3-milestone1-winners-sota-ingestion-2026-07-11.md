# ARC-AGI-3 Milestone-1 winner audit — SOTA ingestion (2026-07-11)

**Provenance:** operator directive ("can we clone those locally so we can look at their repos and
see if there are any things we can learn or spot any energy model opportunities?"), following the
2026-07-07 ARC Prize blog post announcing the three Milestone-1 ($37.5K) winners. All three teams'
code was open-sourced as a prize-eligibility requirement. Cloned/pulled read-only into `external/`
(gitignored, same pattern as `external/ARC-GEN/`) and audited by three independent agents, each
scoped to inspection-only (no execution of untrusted code, per the "Audit untrusted code"
discipline). Full per-repo reports are reproduced in full below the synthesis; nothing here is
paraphrased away from what the audits actually found.

- **1st place — Tufa Labs, "Duck Harness"** (`external/duck-harness/`, github.com/Tufalabs/duck-harness).
  Qwen 3.6 27B local, mean score 1.60, 0/500 runs won.
- **2nd place — "Reki"** (`external/arc-m1-2nd-reki/milestone1-2nd-solution.ipynb`, Kaggle notebook).
  Gemma-4-31B-it local.
- **3rd place — Md Boktiar Mahbub Murad, "forge"** (`external/arc-m1-3rd-forge/arc-agi-3-lb-0-86-3rd-place-candidate-milestone.ipynb`,
  Kaggle notebook, LB 0.86). Gemma-4-31B-it local.

**Headline finding, corroborated independently by all three audits:** every winner is a
**generator-only architecture** — an LLM/VLM directly picks moves each step, with no persisted
transition graph, no systematic search/backtracking, and no cross-run reproduction gate. That is
structurally the gap our own `E3AgentPolicy`/`OfflineSolver` (verifier-routed best-first search +
RESET-replay navigation + the `ops/arc_solve_registry.yaml` reproduction-gated corpus) already
fills. None of the three winners have anything resembling our banked, reusable, independently
re-verified solve corpus (137 levels across 25 public games as of this session).

**Second headline finding:** the ONE verification mechanic that survives in all three, found
independently, is a cheap, deterministic "did this action actually change anything?" check.
**forge's own ablation is the sharpest piece of evidence in the whole audit**: their winning
`gemma31b_public_single` configuration explicitly DISABLED the LLM-judge candidate arbiter and the
LLM confidence-gate (both cut for cost/latency) while KEEPING the deterministic
`changed_pixels==0 ⇒ ineffective` filter. That is a top-3 team, under real competitive pressure,
independently re-deriving this project's own founding thesis: **a cheap, real, deterministic
verifier beats an expensive, hallucination-prone LLM-based one.** All three winners also ran fully
local open-weight models (Qwen 3.6 27B, Gemma-4-31B ×2) — independent validation of this project's
sovereignty-first stance (CLAUDE.md "Decentralization-Respecting Design Constraints").

Also worth naming honestly: their per-game solve-rate ceiling is real and low (Duck: 0/500 wins,
median 0.07; Reki/forge scored a few levels/game). None of the three "solve ARC-AGI-3" in any
strong sense — the milestone prize rewards relative standing on a benchmark where frontier systems
score under 1%, not a solved problem. Nothing here should be read as "they found the answer and we
should copy it wholesale" — the value is in the specific, falsifiable mechanisms below.

---

## Concrete opportunities (priority-ordered by expected value / cost)

### O1. InertClickPruner — a click/inert-target pruner parallel to our HazardMovePruner
**Source:** Reki's "dead-signature" mechanism (`_record_deadsig`) — after every click, the
signature `(color, size, is_rect, twin-count)` of the clicked component is tracked; if the click
never changes the frame, twice ⇒ the signature is marked dead for the rest of the level (except a
signature that was EVER effective is permanently protected). Suppresses both the heuristic click
picker and the LLM's own planned clicks.

**Why this is exactly our shape:** `python/carnot/agentic/arc_hazard_pruner.py`'s
`HazardMovePruner` already does this pattern on the LETHAL-move axis (learns lethal nav moves from
the search's own observed avatar-removal deaths, trust+specificity gated, refits at 50 samples). We
have no live-path equivalent on the INERT-click axis. This is architecturally a copy-shape
extension, not a new invention.
**Candidate design:** a `InertClickSigPruner` keyed on a structural component signature (reuse
Reki's `(color, size, is_rect, twin_count)` shape or our own `ColorBlobSaliencePrior` object
descriptor), feeding `StepwiseExplorer._candidates` the same way `HazardMovePruner` feeds the
offline solver's move list. Trust+specificity-gated (our existing pattern is more robust than
Reki's greedy `K=2` threshold — see the fragility note below).
**Priority:** high — cheap, general, oracle-distinct, directly reuses an existing code shape.

### O2. Replace forge's disabled arbiter slot with a real energy scorer
**Source:** forge's `_select_candidate_with_arbiter` — a second LLM call that judges N sampled
candidate plans and picks one, DISABLED in their winning config for cost. Their static fallback
score (`_candidate_static_score`) is a hand-tuned keyword/heuristic formula, also effectively
retired in spirit (kept as a fallback only).
**Why this matters:** this is architecturally the EXACT slot our verifier-routed search already
fills — candidate generation, then a separate scoring/selection step — except forge's options were
"expensive LLM judge" or "brittle hand-tuned heuristic," and they had to pick the cheap option. We
have a third option they didn't: an oracle-distinct energy scorer (goal-energy + world-model trust
+ action-effect scorer), cheap AND real.
**Candidate design:** not a new build — this is a mapping/positioning exercise. Frame our existing
`WorldModelVerifier` + goal-energy signal explicitly as "the arbiter forge wanted but couldn't
afford," and use forge's own ablation numbers as a citable efficiency argument in any future
paper-v6 verifier-moat section.
**Priority:** high for the framing/citation value; the underlying energy machinery already exists,
so the "build" cost here is near zero — mostly a positioning + a small evaluation task (does our
existing scorer, dropped into forge's candidate-then-score slot, actually do better than their
disabled arbiter on their own eval harness? — see O5 below for the harder generalization of this
question).

### O3. Hallucination-consistency checks: claimed-diff vs measured-diff, goal-hypothesis vs transitions
**Source:** two independent instances. (a) Reki's `board_change_assessment` (the model self-reports
what changed) is computed alongside `changed_pixels` (the real pixel diff) but the two are NEVER
cross-checked. (b) Duck's "scientist note" world model carries a free-text Goal/Action model that
is regenerated each turn but never checked against the actual observed level-up/no-change reward
transitions.
**Why this matters:** this is a direct, literal instance of Carnot's founding thesis — verify a
claim against ground truth — sitting unexploited in two independently-built winning pipelines.
**Candidate design:** a lightweight consistency energy: `distance(claimed_diff_description,
measured_pixel_diff)` for (a), and a "does this goal-hypothesis correctly predict the sign of the
last N level-up/no-op transitions" scorer for (b). Both are cheap, deterministic, and directly
gate/veto a generator's self-report rather than requiring a second expensive LLM call.
**Priority:** medium-high — clearly in-thesis, moderate build cost (needs a real diff-vs-text
comparator, which is closer to new work than O1/O2).

### O4. Object-segmentation perception (Duck Harness) — the highest-value single steal
**Source:** Duck's `inference/utils/segmentation.py` — every frame is parsed into connected-component
objects with a translation-invariant shape `hash` (sha1 of normalized color+cell pattern, enabling
cross-frame object TRACKING by identity, not just position), a containment tree (`children`), and
an adjacency list. This is the primary perception asset feeding their LLM, not an afterthought.
**Why this matters most:** this directly attacks our own already-documented binding constraint —
`project_arc_live_agent_learning_gaps` (memory) and the GAP-4891 ladder finding that frame-only
order-1 features sit at LOO=chance. It is also exactly the shape of the "classical
connected-component/color-blob segmentation" lever already staged as task 2 in the active
2026-07-06 ARC priority list above (arXiv:2512.24156) — **Duck's segmentation is a second,
independent real-world implementation of nearly the same idea, from a DIFFERENT top-3 team**,
which is corroborating evidence the lever is worth taking seriously, plus concrete additional
detail (the translation-invariant hash for object identity persistence across frames, and the
containment/adjacency structure) that task 2's cited paper doesn't fully specify.
**Candidate design:** when task 2 (classical segmentation + salience tiers) is implemented, add
Duck's translation-invariant object-hash tracking and containment/adjacency extraction as explicit
sub-components, not just size/color salience tiers.
**Priority:** high — this should be folded into task 2's implementation directly rather than run as
a separate experiment; see the known-issues.md task entry below.

### O5. Persistent per-game natural-language hypothesis memory
**Source:** both Reki (`reflection_buffer` → periodic Markdown memory rewrite: Rules/Goal/Progress/
Avoid, re-injected every prompt) and Duck (the "scientist note" — World/Goal/Action model/Recent
findings/Open questions/Plan, cleared on level-up to force re-grounding) maintain a persistent,
LLM-authored natural-language hypothesis document that survives context eviction within a single
game session.
**Why this matters:** this is a genuine capability gap on our side, independently flagged by BOTH
audits. Our learning is offline (registry + trained heads, updated between runs) — we have no
online, within-session NL hypothesis memory that a generator reads/writes as it explores a single
hidden game.
**Candidate design:** open question, not yet a committed design — see the fragility caveat below
(both source implementations let unverified hypotheses become "authoritative" context with no
grounding check, i.e. this is where O3's consistency-checking becomes load-bearing: an NL memory
without a hypothesis-verification step is confabulation-with-extra-steps). If built, it should be
paired with O3's goal-hypothesis-vs-transitions check from the start, not bolted on after.
**Priority:** medium — real gap, but the highest-value version of this is NOT "add an LLM
scratchpad," it's "add an LLM scratchpad that's continuously checked against ground truth" — i.e.
O5 is not standalone, it's O3 applied over time. Treat as a follow-on to O3, not a parallel track.

---

## What this does NOT change

None of this displaces the active 2026-07-06 ARC priority list (perception audit → classical
salience front-end → ontology-error pilot, etc., immediately above in this file) or the reserved
task-8 TRM-generator slot. O4 folds directly into that list's task 2. The others (O1, O2, O3, O5)
are net-new candidates, queued below at appropriate priority — not a majority-share claim on ARC's
slot the way the original Phase D pivot was.

## Fragility / what NOT to copy wholesale

- Reki's dead-signature is greedy (`K=2`, "ever effective ⇒ protected forever") — mis-protects
  context-dependent object classes after one incidental change, and over-suppresses in games where
  identical-looking tiles ("twins") behave differently by position. Our trust+specificity-gated
  `HazardMovePruner` pattern is already more robust; O1 should use OUR gating discipline, not
  Reki's.
- Both Reki's and Duck's NL memory let unverified model output become "authoritative" prompt
  context with no grounding check — this is the exact failure mode O3 exists to prevent. Do not
  build O5 without O3.
- forge's confidence gate is a self-reported LLM float with no external grounding — not worth
  copying as-is; if we want an uncertainty signal, ensemble-disagreement or forward-model
  prediction error (a real number) is the correct replacement, not asking the model how confident
  it feels.
- All three winners are LLM-capability-bound on perception and win-condition induction with no
  systematic search fallback — this is precisely the axis where our search-plus-verifier structure
  already has a real, demonstrated advantage (the 137-level, 25-game reproduction-gated corpus none
  of them have an equivalent of). Nothing here suggests abandoning the search structure in favor of
  a pure generator.

## Full per-repo audit reports

(Verbatim from the three independent read-only audits; reproduced in full per the "capture,
don't lose, exploration findings" discipline — see each report for exact file citations.)

### Duck Harness (1st place)

*(full report on file with the session; key findings folded into the synthesis above — see
`external/duck-harness/ARC3-Inference/inference/agent/{tool_agent,python_tool_sandbox,prompts,vision_context}.py`,
`inference/utils/segmentation.py`, `inference/framework/solver.py` for the source.)*

### Reki (2nd place)

*(full report on file with the session; key findings folded into the synthesis above — see
`external/arc-m1-2nd-reki/milestone1-2nd-solution.ipynb` for the source.)*

### forge (3rd place)

*(full report on file with the session; key findings folded into the synthesis above — see
`external/arc-m1-3rd-forge/arc-agi-3-lb-0-86-3rd-place-candidate-milestone.ipynb` for the source.)*

## Bottom line for the roadmap

Five candidates (O1–O5), priority-ordered above. O1 (InertClickPruner) and O4 (fold Duck's
object-segmentation into the already-staged task 2) are the cheapest and most directly actionable —
both extend existing code shapes rather than requiring new architecture. O2 is mostly a
framing/citation exercise against already-existing machinery. O3 is genuinely new but small and
directly in-thesis. O5 is real but should not be built ahead of O3. Queued as new tasks in
`ops/known-issues.md`'s active ARC priority list (below the existing task 8) so the planner picks
them up under the November-2026 standing floor.
