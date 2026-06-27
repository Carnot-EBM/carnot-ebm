# SOTA ingestion: "Thinking to Recall" (arXiv:2603.09906) → verifier-gated reasoning

**Date:** 2026-06-27 · outer-loop (interactive, operator-requested) · SOTA-Ingestion Cycle Discipline.
**Source:** Google Research blog, "Thinking to recall: how reasoning unlocks parametric knowledge in
LLMs" (research.google/blog/thinking-to-recall-...), underlying paper **arXiv:2603.09906**.
**Channel:** reliable low-concurrency WebFetch (1 pass, 2026-06-27); NOT /deep-research (banned from the
autonomous loop). Quotes below are single-pass WebFetch and therefore **[FLAGGED — re-check against live
HTML before any paper-v6 use]** per the corroboration-bank verification discipline
(`anthropic-rsi-and-w2s-citations.md`).

---

## 1. What the paper establishes

Thesis: **a reasoning trace unlocks facts already in the model's weights ("parametric knowledge") even on
SIMPLE single-hop questions** — not just complex multi-step ones. Reasoning acts as a *retrieval /
elicitation* mechanism over latent knowledge, via two mechanisms:

1. **Computational buffer** — reasoning tokens are extra forward passes for latent processing.
   Demonstrated with *dummy-trace injection*: even meaningless repeated "Let me think" tokens "substantially
   improve recall vs. reasoning turned off" — but "never fully match the model's natural reasoning traces."
2. **Factual priming** — the model emits topically-related facts that "build a contextual bridge" to the
   answer (spreading activation). Conditioning on the *extracted intermediate facts alone* "recovers most
   of reasoning's gains."

**The load-bearing finding (this is the Carnot hook):**
> "If a reasoning trace contains even a single hallucinated intermediate fact, the model is significantly
> less likely to arrive at the correct final answer."

**The stated, UNFIXED fragility:**
> Factual priming "introduces a fundamental risk" because "the model generates these intermediate facts
> itself, [so] they might be hallucinated."

They audit this with "a search-enabled verifier to independently check the correctness of every single
intermediate fact" — i.e., they reach for an external verifier to *measure* the fragility, but do not ship
a verifier *in the loop* to fix it.

Benchmarks: Gemini-2.5 Flash/Pro, Qwen3-32B on **SimpleQA Verified** and **EntityQuestions**; pass@k with
reasoning on vs off. Limits: the buffer effect has diminishing returns at longer dummy lengths; priming is
fragile precisely because of self-generated hallucinated facts.

Cited: arXiv:2603.09906 (primary); arXiv:2504.13837 (capability boundary); arXiv:2107.03374 (pass@k);
arXiv:2310.02226 (token generation as computation).

---

## 2. Paper-v6 corroboration quote bank (FLAGGED — re-verify before publication)

For the verification / limitations / "verifier as the reliability layer" sections of paper-v6, alongside
the Anthropic RSI+W2S and LeCun banks. All single-pass WebFetch 2026-06-27 — diff against live HTML first.

- **[FLAGGED]** "If a reasoning trace contains even a single hallucinated intermediate fact, the model is
  significantly less likely to arrive at the correct final answer." — the cleanest external statement that
  *intermediate-fact correctness gates the final answer*, i.e. the exact failure a process verifier
  removes.
- **[FLAGGED]** Factual priming "introduces a fundamental risk" because "the model generates these
  intermediate facts itself, [so] they might be hallucinated." — a frontier lab naming the gap **without a
  fix**; Carnot's verify-and-repair core is that fix.
- **[FLAGGED]** they use "a search-enabled verifier to independently check the correctness of every single
  intermediate fact" — independent convergence on "reasoning needs a verifier on its intermediate facts."

**Why this strengthens paper-v6.** It is a Google-Research-independent witness that reasoning's gains are
*gated by intermediate-fact correctness* and that self-generated facts are an unsolved reliability risk.
That is Carnot's thesis (escape hallucination via verification) stated from the outside, and it is the
**process-verification** regime where Carnot's verifier *measurably wins* (FoVer step-verification AUROC
0.913), not the ARC generation regime where it does not.

---

## 3. SOTA → Carnot mapping (honest buckets)

**(A) Strong corroboration of the verifier core — cite, don't build.** Section 2 above. This is the
highest-value takeaway: a clean external corroboration for the paper-v6 verifier-reliability claim.

**(B) One genuinely-actionable lever — VERIFIER-GATED REASONING (scoped in §4).** The paper shows factual
priming is fragile because the model's own intermediate facts may be hallucinated, and that one bad fact
poisons the answer. The fix is a *process verifier in the loop* that filters/repairs hallucinated
intermediate facts **before** they prime the answer — converting the paper's fragile priming into a
reliable one. This lands in Carnot's win domain (process/step verification, FoVer 0.913) and uses the
verify-and-repair pipeline (`python/carnot/pipeline/verify_repair.py`) exactly as designed. **Distinct
from, and independent of, the ARC generation wall.**

**(C) Honest NON-application — does NOT reopen the ARC line.** The mechanism is *recall of latent
parametric knowledge*. ARC-AGI-3 hidden games are *novel / OOD by design* — there is no parametric
knowledge to recall, so reasoning-to-recall cannot make a winning prefix appear (the generation wall is
unmoved; cf. `arc-008-wall-root-cause`, the `.449` INDUCER_CEILING corroboration exp4871). It is also not
the local-induction fix: ARC induction is *novel rule inference*, not recall, and a frontier model *with*
full reasoning still produced ~0 accuracy world-models (exp4871 tu93 control 0.0). A tempting micro-bridge
— our `CARNOT_ARC_CODEONLY_INDUCE=1` directive *suppresses* reasoning for speed, which by this paper could
cost the buffer/priming effect — is capped in EV by the codex-with-reasoning null; worth at most a cheap
A/B, never a headline bet.

**(D) Bonus corroboration of "answer present but mis-surfaced."** "The answer is in the model but not
surfaced directly" is the same shape as Carnot's TRM finding (pass@1000 ≫ pass@2;
`reference_trm_tta_mcgovern`) and GAP-4 ("answer in the pool, vote misses it";
`project_gap3_verifier_program`). One more independent witness that the gap is *surfacing/selection* — the
verifier's job — not raw capability.

---

## 4. Experiment scope — VERIFIER-GATED REASONING (GPU 1, outer-loop)

**Hypothesis.** A process verifier that filters/repairs hallucinated intermediate facts during a reasoning
trace makes factual priming RELIABLE — recovering reasoning's recall gain (the paper's effect) while
removing the hallucination-poisoning the paper identifies as unfixed. Tests Carnot's verifier in its
*home* regime (process verification with real headroom), NOT the ARC generation wall.

**Hardware/decentralization.** Local open model on **GPU 1** (the outer-loop's dedicated 3090 per the
2026-06-27 allocation); no frontier API; offline-legal.

**Setup.**
- **Reasoner (generator):** a local open LLM (Qwen3.5-9B on GPU 1) producing single-hop factual-QA answers
  with a reasoning trace that emits intermediate facts.
- **Corpus:** a held-out single-hop factual-QA set whose intermediate facts are *checkable* (the paper used
  SimpleQA-Verified / EntityQuestions; for an offline/decentralized run, a local factual-QA subset + a
  local knowledge source for fact-checking). The corpus MUST have real headroom (reasoning > no-reasoning,
  i.e. the paper's effect reproduces) — else there is nothing for the verifier to protect.
- **Verifier (oracle-DISTINCT):** Carnot's process/step verifier (FoVer-class) + the verify-and-repair
  pipeline (`verify_repair.py`) scoring each *intermediate fact* and pruning/repairing the ones it flags as
  likely-hallucinated, BEFORE they condition the final answer. `verifier_is_oracle=false` — it scores
  intermediate facts; the final answer is checked separately by the QA oracle.

**Arms (matched, same corpus + seeds).**
- **A. No-reasoning baseline** (direct answer) — the paper's lower bound.
- **B. Raw reasoning** (full unfiltered trace) — the paper's gain, but poisoned by hallucinated facts.
- **C. Verifier-gated reasoning** (Carnot process verifier filters/repairs intermediate facts) — the test.

**Decisive gate.** C > B on final-answer pass@1 with a paired-bootstrap **CI95 excluding 0**, AND the gain
attributable to removed/repaired hallucinated facts (report the hallucinated-intermediate-fact rate per
arm — the paper's mechanism). **Positive control:** on traces with NO hallucinated facts, C ≈ B (the
verifier must not degrade clean traces). **Matched control + FALSE_NEGATIVE guard:** if C ≈ B overall,
confirm the corpus actually contained hallucinated-fact traces for the verifier to catch (else it's a
no-headroom null, not a method null).

**Guards / artifact discipline.** `inference_substrate=live_llm_inference` (60s floor; the reasoner runs
live on GPU 1); `verifier_is_oracle=false`; `random_seed` + `reproducibility_checksum`;
`preconditions_checked` (local model cached on GPU 1, verifier importable, corpus present);
adversarial-verify clean (no circularity: the intermediate-fact verifier must be distinct from the
final-answer oracle).

**Honest risks.**
1. **Domain transfer of the verifier — DE-RISKED POSITIVE (2026-06-27).** FoVer's 0.913 is on math/code
   *step* verification; *factual* intermediate-fact verification is a different domain. The concern was
   that Carnot's model-native signals may not detect factual hallucinations at all — the paper used a
   *retrieval-grounded* ("search-enabled") verifier. **Resolved by `verifier_gated_reasoning_derisk`**
   (`results/verifier_gated_reasoning_derisk_hardened.json`, n=500, 45 correct / 455 hallucinated on
   SimpleQA-Verified, Qwen3.5-9B on GPU 1): three *no-retrieval* model-native signals all discriminate
   correct-vs-hallucinated FINAL answers above chance, strongest being **self-consistency AUROC 0.759,
   paired-bootstrap CI95 [0.698, 0.818]** (lower bound strictly > 0.5; label-shuffle control 0.536).
   So a no-retrieval Carnot-side verifier CAN discriminate this model's own factual hallucinations →
   the full experiment does NOT need retrieval grounding for the discrimination step; self-consistency
   is the lever. **Residual (carry into the full experiment):** the de-risk measured discrimination of
   the *final answer*; the full experiment gates on discriminating *intermediate facts* in a trace
   (reasonable extrapolation — intermediate facts are also self-generated factual claims — but unproven
   until the A/B/C run). The de-risk did NOT establish headroom (that gating improves final accuracy);
   that is risk #2 and the full experiment's C>B gate.
2. **Headroom dependence.** If the corpus has few hallucinated-fact traces, there's nothing to gate (the
   FALSE_NEGATIVE guard catches this).
3. **Cost.** Live local reasoning on GPU 1; scope to a few hundred QA items + a few seeds (sub-hour).

**Why it's worth running (vs the ARC energy line).** Unlike every ARC energy lever (selection on a
generation wall), this puts the verifier where it *already wins* (process verification) on a task with
*real, externally-corroborated* headroom (the paper's hallucination-poisoning effect). It directly
produces a paper-v6 result: "Carnot's process verifier converts fragile reasoning-to-recall into reliable
recall," with the Google paper as the motivating corroboration.

---

## 4a. UPDATE 2026-06-27 — risk #2 (headroom) checked: PREMISE DOES NOT HOLD for this model/corpus

Before building the full A/B/C, the headroom premise (risk #2) was validated cheaply
(`scripts/experiments/verifier_gated_reasoning_headroom.py`,
`results/verifier_gated_reasoning_headroom.json`, n=208, GPU 1). Two-stage factual priming (generate
intermediate facts → answer conditioned on them; the paper's mechanism) vs direct answer, graded by a
local LLM-judge applied identically to both arms (the crude substring matcher had a length bias
favoring the verbose priming arm — fixed):

- **acc(priming) = 0.101, acc(direct) = 0.077, delta = +2.4pp, paired-bootstrap CI95 [−0.024, +0.072]
  — STRADDLES 0.** `reasoning_headroom_confirmed: false`.

**Conclusion:** the arXiv:2603.09906 recall effect (measured by the authors on Gemini-2.5 / Qwen3-32B)
does **not** significantly reproduce on **Qwen3.5-9B / SimpleQA-Verified**. With no significant
reasoning-recall gain, a process verifier protecting intermediate facts has nothing to improve
(C ≈ B by construction) — so **the full A/B/C build is moot for this exact model+corpus** and is NOT
built. This is the cheap-validation discipline working as intended: the #1 risk was resolved positive
(the verifier *can* discriminate, §1) but the #2 premise failed the gate, saving the large build.

**Revival conditions (any one would re-open the build):** (a) a larger local model where the recall
effect is significant — the authors' positive results were on a 32B-class model, so a Qwen3-32B-class
local run on GPU 0/1 is the natural next test; (b) a higher-headroom corpus (multi-hop / EntityQuestions,
where priming helps more than on single-hop SimpleQA); (c) restricting to the items the paper's effect
is strongest on (`multi_step`/`requires_reasoning` flags exist in the SimpleQA-Verified schema).
Discordant pairs were ON-right/OFF-wrong=15 vs OFF-right/ON-wrong=10 — a real but small, noisy tilt
toward priming, consistent with "effect exists but is sub-threshold at 9B."

## 5. Flagged for the next roadmap

- **Verifier-gated reasoning (§4): PAUSED — premise failed at 9B (§4a).** Do NOT build the full A/B/C
  on Qwen3.5-9B/SimpleQA-Verified. Re-open only under a §4a revival condition (larger model, or a
  higher-headroom corpus). The #1-risk de-risk (model-native verifier discriminates, §1) remains a
  standalone positive result usable elsewhere.
- **Cite in paper-v6:** the §2 quote bank, alongside the Anthropic + LeCun banks, in the
  verification/limitations sections.
- Marked ingested in `research-studying.md`.

Cross-refs: `anthropic-rsi-and-w2s-citations.md`, `lecun-world-models-citations.md`,
`feedback_quote_anthropic_corroboration` (memory), `project_gap3_verifier_program`,
`reference_trm_tta_mcgovern`, `python/carnot/pipeline/verify_repair.py`,
`python/carnot/verify/fover_semantic_calibration.py`.
