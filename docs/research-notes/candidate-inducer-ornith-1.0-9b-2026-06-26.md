# Candidate alternative inducer LLM: Ornith-1.0-9B (FUTURE — gated behind the energy program)

**Date:** 2026-06-26 · **Author:** outer-loop, operator-directed ("make note of it as another
alternative LLM to consider in the future; do not let it get in the way of our energy model
attempts: we want to try S2/S3/S4 first").
**Status:** NOTE ONLY — not queued as an experiment. **Explicitly deprioritized behind the
oracle-distinct structural-energy program (S2 → S3 → S4).** Do NOT spend an ARC slot on this until
the energy stages have run (or the operator re-prioritizes).

## What it is
`deepreinforce-ai/Ornith-1.0-9B-GGUF` (https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF):
- Dense **9B, Qwen-3.5 variant**, **MIT-licensed**, GGUF (Q4_K_M **5.63 GB**, Q8_0 9.53 GB, BF16 17.9 GB).
- RL-trained for **agentic coding**; notably trained to "generate not only solution rollouts, but
  also **the scaffold that drives those rollouts**" (a self-improving search framework).
- Reported: **SWE-bench Verified 69.4%**, Terminal-Bench 2.1 43.1%, SWE-bench Pro 42.9%,
  NL2Repo 27.2%, Claw-eval avg 63.1%. "SOTA among open-source models of comparable size on coding."
- Strong tool-calling / agentic-coding; OpenAI-compatible function calling.

## Why it could matter to Carnot (the one real hook)
The binding ARC wall (levers ledger + 2026-06-25/26 sessions) is **induction quality** — the
free-form LLM dynamics engine is **0.12-accurate on lp85** because the current generator writes
rambling placeholder `engine()` code; the SOTA ARC winner (executable-world-model, arXiv:2605.05138)
wins by using a **strong coding-agent inducer**. Writing the executable world-model
`engine()`/`is_level_complete()` IS a coding task. Ornith is a **same-envelope** (Qwen-3.5-9B,
~5.6 GB Q4, MIT, GGUF — fits Kaggle ~16 GB) but **markedly stronger agentic-coder** than the current
`Qwen3.5-9B-MTP` ([[project_arc_live_generator]]). So it is the strongest same-footprint candidate to
test against the engine wall, in the **offline engine-induction** role (the live submission stack is
frozen + MTP; the offline induction may use a different model).

## What it does NOT change
- Carnot's thesis (the energy VERIFIER) and the structural-energy program are downstream of whichever
  generator induces — Ornith is a generation-side input, fits the hybrid architecture
  ([[feedback_hybrid_pragmatic_architecture]]; north-star §0 generator=commodity, energy=verifier).
- NOT a path to revive energy-as-RFT-teacher (RETIRED) — Ornith's RL self-improvement is
  generator-side, a different mechanism.

## Honest caveats (why this is a hypothesis, not a given)
1. **SWE-bench ≠ ARC** — agentic-coding strength may not transfer to ARC dynamics-induction; the hook
   is specifically the code-induction sub-task, not the whole agent.
2. **The wall may be deeper than model strength** — our own structured-engine attempt (exp4749)
   also nulled, so a stronger inducer might help but is not guaranteed to clear the wall.
3. **No MTP** — the current pick is Qwen3.5-9B-**MTP** (chosen for live throughput); Ornith is
   standard dense, so it fits the OFFLINE induction role, not a live-stack swap.
4. **Sprint freeze** — the live submission stack is frozen through 2026-06-30; this is a
   post-deadline / offline candidate.

## If/when evaluated (the scoped experiment, FUTURE)
A scoped **offline A/B**: Ornith vs Qwen3.5-9B-MTP on **held-out engine-induction accuracy** (the 0.12
lp85 baseline + a couple more games), reproduction-gated. PRECONDITIONS step 0: cache check (not yet
in `~/.cache/huggingface`). Gate: Ornith materially raises held-out engine accuracy over the 0.12
baseline → candidate inducer for the post-deadline live stack; else cheaply rules out "the wall is
just a weak coder." **Gated behind S2/S3/S4** per the operator directive.

## Cross-references
- [[project_arc_live_generator]] (the frozen Qwen3.5-9B-MTP it would be compared against)
- [[feedback_hybrid_pragmatic_architecture]] (generator=commodity, energy=verifier)
- `docs/research-notes/oracle-distinct-structural-energy-program-2026-06-26.md` (the S2/S3/S4 program this is gated behind)
- levers ledger `docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md` (the induction-quality wall)
