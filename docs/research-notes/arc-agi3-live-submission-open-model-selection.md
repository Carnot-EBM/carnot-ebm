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

## The open SOTA LLM is a REQUIRED ESCALATION TIER, not an alternative (operator, 2026-06-17)

"There will undoubtedly be cases even with TRM where an open SOTA model will be required
to try and help solve a game." So the live stack is NOT "TRM vs LLM" — it is a
**verifier-routed CASCADE**, and the open SOTA LLM is a load-bearing tier that MUST be
bundled regardless of whether TRM is also bundled:

| Tier | Engine | Handles | Cost |
|---|---|---|---|
| 1 | training-free explorer (BFS/DFS + frontier-distance) | shallow/medium games (8/11 of the public set) | cheap, no model |
| 2 | trained TRM-class dynamics/verifier | efficiency + learned routing/pruning where it generalizes | tiny (~5M), forward-pass only |
| 3 | **open SOTA LLM (bundled, offline)** — E3 world-model induction + goal reasoning | the HARD tail: novel mechanics, goal induction, anything tiers 1–2 stall on | heavy (load+infer), invoked SELECTIVELY |

**The Carnot VERIFIER is the router AND the ground:** it detects when a cheap tier has
stalled (escalate) and it grounds the LLM's induced model before trusting its plan (the
cd82 divergence-halt is this working). The LLM is the top of the cascade, not a swap-in.

**Key budget implication:** because tier 3 fires SELECTIVELY (only the hard games the
explorer/TRM can't crack — a minority), its per-invocation load+inference cost is
AMORTIZED across the 8h budget. That RELAXES the "must be tiny" constraint for the LLM:
we can afford a **stronger** open model (gemma-4-26B-A4B / Qwen3.6-35B-A3B) for the
escalation tier precisely because it isn't run on every game. Tier 2 (TRM) stays tiny for
the common case; tier 3 (LLM) can be bigger for the rare hard case.

**Integration step:** unify the agent's current explorer-mode and E3-mode into ONE
verifier-routed `choose_action`: run the explorer; when the verifier flags a stall (no
progress within a budget window), escalate to E3 induction with the bundled open LLM;
grounded by the verifier; plan_in_model; execute with divergence-halt. This is the
Meta-EBM-Cascade-Router thesis made concrete on ARC-AGI-3.

## The TRM-class engine — the cheap/common tier (tier 2, alongside the LLM tier)

Train a SMALL recurrent TRM-class model OFFLINE on ARC-AGI-3 game transitions to predict
next-frame / good-action directly. Bundled weights are tiny (~5M params), forward-pass-
only at eval (cheap, fast, fits any GPU), the most sandbox-friendly + decentralization-
pure option. Our explorer already GENERATES the transition corpus to train it. This is
tier 2 of the cascade — it makes the COMMON case efficient (learned dynamics/routing where
it generalizes); it does NOT replace the tier-3 LLM, which handles the hard tail TRM
can't. Both are bundled; both benchmark on the same verifier-accuracy + solve loop.

## Recommendation

1. **Default open proposer for first live attempts: Qwen3.6-35B-A3B (Apache-2.0, ~3B
   active, cached)** — cleanest license + fast MoE inference. Fall back to
   **gemma-4-12B** if the sandbox is single-T4 (fits easily, Kaggle-hosted).
2. **Benchmark both (+ gemma-4-26B-A4B) empirically** via `arc_e3_agent_solve --local`
   once the codex dev-ceiling lands, ranking by verifier accuracy + plan-solve + fit.
3. **Bundle BOTH tiers** (the cascade, not an either/or): the tiny TRM-class model as
   tier 2 (cheap/common) AND a stronger open LLM (tier 3) for the hard tail. Selective
   tier-3 escalation amortizes its cost, so the LLM can be the bigger 26–35B.
4. **Integration:** unify explorer-mode + E3-mode into one verifier-routed `choose_action`
   (escalate to the bundled LLM when the verifier flags a stall).

Cross-refs: `docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md`
(offline + bundled-weights mechanics, license scope), `...focused-loop-and-engine...md`
(E3 = the efficiency lever), CLAUDE.md "SOTA Local Models" + "Decentralization-Respecting
Design Constraints".
