# ARC-AGI-3 Live Submission — Open-Weights Model Selection (2026-06-17)

Operator note (2026-06-17): closed foundation models (Claude / codex / gemini) are fine
for DEV-TIME work — authoring the TRM / BFS / DFS / routing / proposer code. But the LIVE
offline submission must bundle an OPEN-WEIGHTS model (no internet at eval; closed APIs
forbidden). This note identifies the candidate + the empirical selection method.

## The two-tier architecture (locked)

| Tier | Model | Role |
|---|---|---|
| DEV (internet OK) | Claude / codex (gpt-5.5) / gemini | author the harness code; validate the E3 loop; the verifier-accuracy CEILING baseline |
| LIVE (offline sandbox) | an OPEN-WEIGHTS model, bundled as a Kaggle dataset, mounted at /kaggle/input, loaded from disk | the E3 world-model proposer (or a trained TRM-class dynamics model) |

The verifier grounds whatever the live model proposes, so a weaker open model just earns
a lower verifier score — the loop measures it honestly.

## Sandbox constraints that rule out most "SOTA"

The offline Kaggle sandbox is a single T4 (~16GB) or 2×T4 (~32GB), ~12h. So the
top open-weights coders — **DeepSeek V4-Pro** (LiveCodeBench 93.5), **MiniMax M3**
(SWE-Bench Pro 59%), **Kimi K2.6**, **GLM-4.7/5.1**, **Qwen3.7 Max** — are OUT: they're
100B+ / huge-MoE and don't fit. The bundled model must be SMALL-but-strong.

## Sandbox-fittable open-weights candidates (the real shortlist)

| Model | Fit | License | Notes |
|---|---|---|---|
| **Qwen3.6-35B-A3B** (MoE, ~3B active) | Q4 ~20GB → 2×T4 | **Apache-2.0** (cleanest for prize eligibility) | flagship MoE, ~3B-active = fast inference; already cached; strong code |
| **gemma-4-26B-A4B-it** (MoE, ~4B active) | Q4 ~14–18GB → single T4/2×T4 | Gemma license (permissive, has a use policy — verify acceptable) | **Kaggle HOSTS Gemma 4** → trivial to attach as a Kaggle Model (no 30GB self-upload) |
| **gemma-4-12B-it** (dense) | Q4 ~7GB → single T4 easily | Gemma license | the lightweight SOTA: small/fast for MANY induction calls; CLAUDE.md's flagged light pick |
| **gemma-4-31B-it** (dense) | Q4 ~18GB → 2×T4 | Gemma license | strongest dense; beats Qwen3.5-32B on AIME/LiveCodeBench |
| **gemma-4-E4B** (~8B total/4.5B eff) | <8GB → any | Gemma license | the budget option for a single small GPU |

License steer: **Qwen3.6-35B-A3B (Apache-2.0)** is the cleanest as a 3rd-party dependency
(the ARC rule allows Apache-2.0 for non-submitter code). Gemma-4 is permissive too and
Kaggle-hosts it (operational convenience) but carries a use policy — confirm it qualifies
as "an open source license which allows public sharing" before relying on it for prizes.

## Selection METHOD — measure, don't guess (the focused loop applied to models)

Pick empirically: run the E3 induction with each candidate as the LocalGGUFProposer on the
SAME transition sets, and score with the Carnot WorldModelVerifier + the downstream
plan-in-model solve. Metrics per candidate: (1) verifier reproduction accuracy, (2)
does plan_in_model find a solving plan, (3) induction wall-time (must fit the per-game
eval budget), (4) VRAM. The codex run (dev) is the ceiling baseline. Driver already
exists: `scripts/arc_e3_agent_solve.py <game> --local` (swap the cached repo via
`LocalGGUFProposer(repo_substr=...)`).

## The ALTERNATIVE engine — a trained TRM-class model (not an off-the-shelf LLM)

Instead of bundling a 12–35B LLM that writes world-model CODE at eval time (heavy: load +
inference per game), train a SMALL recurrent TRM-class model OFFLINE on ARC-AGI-3 game
transitions to predict next-frame / good-action directly. Bundled weights are tiny (~5M
params), forward-pass-only at eval (cheap, fast, fits any GPU), and it's the most
sandbox-friendly + decentralization-pure option. Our explorer already GENERATES the
transition corpus to train it. This is the "use TRM, not a foundation model" path — a
real candidate, distinct from the LLM-proposer route; the two can be benchmarked head to
head by the same verifier-accuracy + solve loop.

## Recommendation

1. **Default open proposer for first live attempts: Qwen3.6-35B-A3B (Apache-2.0, ~3B
   active, cached)** — cleanest license + fast MoE inference. Fall back to
   **gemma-4-12B** if the sandbox is single-T4 (fits easily, Kaggle-hosted).
2. **Benchmark both (+ gemma-4-26B-A4B) empirically** via `arc_e3_agent_solve --local`
   once the codex dev-ceiling lands, ranking by verifier accuracy + plan-solve + fit.
3. **In parallel, prototype the trained TRM-class dynamics model** as the lighter,
   decentralization-pure alternative; compare on the same loop.

Cross-refs: `docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md`
(offline + bundled-weights mechanics, license scope), `...focused-loop-and-engine...md`
(E3 = the efficiency lever), CLAUDE.md "SOTA Local Models" + "Decentralization-Respecting
Design Constraints".
