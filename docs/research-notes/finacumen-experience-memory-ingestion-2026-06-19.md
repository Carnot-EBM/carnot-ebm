# SOTA ingestion: FinAcumen self-evolving experience memory (arXiv:2606.17642) → ARC generic-solver

**Date:** 2026-06-19 · **Trigger:** operator handed the paper for ingestion.
**Source (VERIFIED):** "FinAcumen: Financial Multimodal Reasoning via Self-Evolving Experience Memory
Harness," Guo, Zhou, Jian, Chen (BUPT / Queen Mary University of London), arXiv:2606.17642v1,
https://arxiv.org/html/2606.17642v1 (fetched 2026-06-19).

## What it is (and why it maps onto us)

FinAcumen is a **financial** multimodal-reasoning agent, NOT an ARC/program-synthesis paper — but it
is a near-exact **structural mirror** of the Carnot ARC live-solver bet, so the value is architectural
analogy + concrete refinements, not a drop-in method.

| FinAcumen | Carnot ARC live agent |
|---|---|
| Frozen **8B** Qwen3-VL backbone | Frozen **9B** Qwen3.5-9B-MTP ([[project_arc_live_generator]]) |
| Self-evolving **experience memory** (Findings + failure Cautions) | example corpus (`ops/arc_solve_registry.yaml`: rules, world-models, gotchas) |
| **Semantic retrieval**, cosine **τ=0.65**, dedup + rank, **k_max=5** | `recommend_approach` transfer-routing |
| **Answer Consolidation Gate** (source-traceability/consistency) | Carnot `WorldModelVerifier` grounding |
| **Fallback to tool-only** when retrieval is uncertain | `graph_explore` model-free fallback |

**Headline corroboration:** a frozen 8B + experience memory **rivals/beats GPT-4o and a 72B** on
several splits (BizBench 27.67%→68.65%, +41 pts; FinMMR-Easy 59.17%→81.67%). Empirical support for our
exact design: *the lever is corpus/retrieval quality, not model size* — do novel invention in dev with
the frontier model, transfer + verify at runtime with the frozen local model.

## The decisive lesson: selective activation (precision > recall)

FinAcumen's load-bearing finding is that an **irrelevant retrieved example actively DEGRADES**
reasoning — so it only injects memory when similarity ≥ τ, and falls back cleanly otherwise. For us:
routing an unseen game to a *dissimilar* recipe MISLEADS the small proposer — worse than no example.
This is the highest-value borrow and directly de-risks the held-out (leaderboard) path.

## What was IMPLEMENTED this ingestion (committed 2026-06-19)

`python/carnot/agentic/arc_solve_learning.py:recommend_approach` (additive, backward-compatible;
5 tests in `tests/python/test_arc_solve_learning_confidence.py`):
- **`confident_transfer` / `routing_confidence` / `top_similarity`** — only few-shot the top recipe
  when the match clears the bar (`_CONFIDENT_TRANSFER_MIN_SIM=3.0`, i.e. ≥ an action-type match);
  below it, induce COLD (strategy solver / graph-explore). The **unseen LIVE game** path correctly
  returns `confident_transfer=False` (verified: `zz99_unseen` → False/0.0; `tr87` → True/1.0).
- **`cautions`** — aggregate the top matches' failure dead-ends + general gotchas (FinAcumen's
  *Cautions*, the complement of the success *Findings*), deduped + capped at 8, for the induction
  prompt to be told what NOT to do.

## SOTA → experiment mapping flagged for `.411 (the planner reads this)

1. **[STRONGEST] Wire `confident_transfer` + `cautions` into the runtime induction prompt** of the
   example-conditioned inducers (`.410 A2/A3 successors, `arc_executable_world_model` induce/refactor):
   when `confident_transfer` is True, few-shot the recipe **and** the cautions; when False, suppress
   the (misleading) recipe and induce cold with cautions-only guardrails. *Takes:* thread the
   `recommend_approach` confidence/cautions into the induce prompt builder. *Pitfall:* the threshold is
   on a hand-built similarity scale (not cosine) — calibrate it against the `.410 LOO benchmark (does
   gating below the bar reduce false transfers without dropping true ones?).
2. **Dedup + rank + cap the few-shot examples (FinAcumen k_max=5).** The inducers currently pass an
   ad-hoc example set; cap to the top-k most-similar deduplicated recipes. *Takes:* small change in the
   prompt builder. *Pitfall:* none material.
3. **Systematic corpus distillation.** FinAcumen scores multiple trajectories and a summary agent
   synthesizes clean Findings+Cautions. Our registry updates ad-hoc per solve; a distillation pass that
   scores attempts and writes clean reusable Findings + Cautions would raise transfer quality.
   *Takes:* a registry-hygiene sub-step. *Pitfall:* don't over-compress away game-specific gotchas.

## Honest caveats (do NOT overclaim)

- Domain mismatch: financial QA (retrieval-heavy text strategies), not program/world-model induction.
  Their memory is prompt-prefix text; ours includes executable `world_model.py` (richer, harder to
  retrieve/rank) — the analogy is architectural.
- It sharpens **transfer**, not **invention**: it does nothing for the genuinely-novel-mechanic tail
  (their tasks are in-distribution). It raises the "rhymes-with-corpus" hit rate, not the novel ceiling
  — consistent with the `.410 capstone's `generic_solver_gap_state: partial`.
- The threshold is the win (FinAcumen's +41 pts came largely from *selective* activation), but our
  τ-equivalent must be calibrated on ARC data, not adopted at 0.65.

**flagged_for_v411:** item (1) — confidence-gated + cautions-injected runtime induction — is the single
strongest method to carry into `.411 (precision-over-recall transfer on held-out unseen games).

## Cross-refs
- `feedback_sota_ingestion_cycle` (memory) · CLAUDE.md "SOTA-Ingestion Cycle Discipline"
- `python/carnot/agentic/arc_solve_learning.py` (implemented) · `arc_executable_world_model.py` (the induce/refactor target)
- `ops/known-issues.md` 2026-06-19 ARC artifact-discipline entry (sibling .410/.411 routing)
- [[project_arc_agi3_north_star]] · [[project_arc_live_generator]]
