# Research Roadmap v331 — Finish the De-Contamination: Does Verifier Value Generalize Once the Test Is Genuinely Fair?

**Milestone:** 2026.06.331 (June calendar-prefix rollover; today UTC = 2026-06-01)
**Status:** Pre-staged by outer-loop (Claude Opus 4.8), 2026-06-01
**Predecessor:** 2026.05.330 (Verifier Cross-Domain Value, DE-CONTAMINATED)

---

## 1. What `.330` Was Supposed to Prove — and Why It Didn't

`.329` concluded the verifier ensemble is "math-only, domain-bound" (exp3576). `.330` correctly
diagnosed that null as **contaminated** (degenerate corpora + inert/wrong verifier sets) and set out to
re-run the test fairly. **`.330` did not finish the job.** Reading the `.330` artifacts via
`scripts/summarize_artifact.py`:

### 1a. The gate cascade broke the centerpiece

| Task | Intended | Actual `.330` outcome |
|---|---|---|
| exp3585 build realistic factual corpus | corpus with confidence headroom | `complete` — corpus built (n=200, confidence AUROC 0.4573) |
| exp3586 score factual-applicable verifiers | per-verifier AUROC table | **`blocked_gate_check_failed`** |
| exp3587 retrieval/NLI grounding verifier | honest factual signal | `complete` — but **AUROC = 1.0 (IMPLAUSIBLE_PERFECT)** |
| exp3588 corrected cross-domain re-measurement | **the milestone centerpiece** | **NO ARTIFACT — never ran** |
| exp3589 additivity / McNemar | second-pair-of-eyes value | **`blocked_gate_check_failed`** (missing exp3588) |

**Root cause (the load-bearing operational finding):** the `gated_on` mechanism in
`scripts/conductor_gates.py:_eval_op` compared exp3585's `corpus_is_realistic` field — emitted as a
**principle-annotated dict** `{value: true, principle: "..."}` — against the bare expected `true`. The
`_coerce_gate_value` helper only normalizes `bool/str/int/float`; it does **not** unwrap `{value: ...}`.
So `{value: true} == true` evaluated False, exp3586 blocked, and the cascade took out exp3588 (the
centerpiece) and exp3589.

> **Forward rule (`.331` design constraint):** ANY field referenced in a downstream `gated_on` block MUST
> be emitted as a **BARE top-level JSON value** in the upstream artifact (`"corpus_v2_is_realistic": true`),
> never a `{value, principle}` dict. The `principle:` annotation lives in the task-prompt spec, not in the
> gated field's stored value. Every `.331` gated field below obeys this. We do NOT modify the conductor.

### 1b. The one factual number that landed is contaminated

exp3587 reported `grounding_verifier_auroc = 1.0` and `ensemble_with_grounding_auroc = 1.0`. But the
corpus `data/realistic_factual_corpus_v1.jsonl` carries only `{question, answer, is_hallucination,
model_confidence}` — **there is no independent evidence/reference passage to ground against.** A
retrieval/NLI grounding verifier cannot legitimately reach 1.0 on a corpus with no evidence column; it is
almost certainly **label-leaking** (or the "retrieval" returned the gold answer). `nli_substrate` was a
`disclosed_text_statistical_proxy` — no real NLI model was loaded. AUROC = 1.0 on n=200 is exactly the
`IMPLAUSIBLE_PERFECT` pattern the Adversarial Artifact Verification rule exists to catch.

### 1c. The verdict was synthesized from broken partial data

The `.330` capstone (exp3596) and synthesis (exp3591) emitted "329_null_was_artifact / verifier value
math_only_earned" — but the centerpiece (exp3588) never ran and the only factual number was a
contaminated 1.0. **The de-contamination question is STILL OPEN.** `paper_ready` remains true (G1-G4 met;
the FoVer 0.9131 headline is independent of all of this).

---

## 2. The `.331` Thesis

> **Finish the de-contamination honestly.** Fix the gate cascade (bare gated fields). Adversarially
> prove or disprove the exp3587 AUROC = 1.0 leak. Build a factual corpus that carries **independent
> evidence passages** (the HaluEval QA `knowledge` field) with genuine confidence headroom. Build a
> **REAL NLI-model** atomic-claim grounding verifier (the HalluSearch / VeriScore SOTA recipe) scored
> against **held-out** evidence (never the gold label). Then run the centerpiece corrected cross-domain
> re-measurement that `.330` skipped — math (0.9131, frozen) | code (execution-applicable verifiers) |
> facts (real grounding verifier) — plus the literature-grounded **math→code PRM-transfer positive
> control** (arXiv:2506.00027, which measured +4–8 pts and predicts the `.330` code AUROC of 0.44 was a
> wiring bug, not a real limit), against the **discriminative-fragility falsifier** (ThinkPRM,
> arXiv:2504.16828).

**Either outcome is genuine learning:**
- Verifiers fire on a fair test → the "math-only" verdict was a contamination artifact; verifier value
  generalizes → a broader, stronger paper claim (scoped to measured corpora).
- Verifiers still don't fire on a genuinely fair test (real NLI, held-out evidence, headroom corpus,
  applicable verifiers) → "math-only" is **EARNED** — a precise, honest, defensible limitation, with the
  ThinkPRM discriminative-fragility mechanism as the explanation.

**Invariants (do NOT regress):** P0.1 stays honest-negative (the Depth-Over-Breadth forcing function is
retired; do NOT re-test Route-1/Route-2). `paper_ready` stays true (G1∧G2∧G3∧G4). No emojis in public
docs. Verifier Authenticity Discipline binds every new verifier (implementation must match docstring; no
np.random features, no sleep-padding, no score-capping; honest name).

---

## 3. Architecture of the Milestone

```
 Phase 0  -- ops transition --------------------------------------------------
   exp3597  archive .330 (record cascade + contaminated 1.0) -> activate .331

 Phase 1  -- diagnose & audit (no new science asserted yet) -----------------
   exp3598  DIAGNOSIS: gate-cascade root cause + adversarial audit of the
            exp3587 AUROC=1.0 leak + corpus-evidence-gap finding +
            per-domain applicable-verifier enumeration + bare-gate convention

 Phase 2  -- build the fair factual test apparatus --------------------------
   exp3599  build factual corpus v2 WITH independent evidence passages
            (HaluEval QA `knowledge` field) + confidence headroom in (0.5,0.95)
            -> emits BARE corpus_v2_is_realistic / corpus_v2_has_evidence
   exp3600  REAL NLI-model atomic-claim grounding verifier (HalluSearch recipe)
            scored vs HELD-OUT evidence  [gated_on exp3599.corpus_v2_has_evidence]

 Phase 3  -- the fair re-test (the science .330 skipped) --------------------
   exp3601  CENTERPIECE corrected cross-domain re-measurement:
            math (0.9131 frozen) | code (exec-applicable verifiers, exp1999) |
            facts (real grounding verifier, corpus v2)  vs strong confidence
            baseline, >=3 seeds, CI  [gated_on exp3599.corpus_v2_is_realistic]
            -> emits BARE positive_control_valid
   exp3602  math->code PRM-transfer positive control (arXiv:2506.00027) --
            does the math-trained signal survive the modality jump?
   exp3603  additivity / McNemar second-pair-of-eyes (fused detector vs
            confidence)  [gated_on exp3601.positive_control_valid]

 Phase 4  -- self-learning + synthesis --------------------------------------
   exp3604  FR-11 continuous self-learning v6 -- calibrate the REAL grounding
            verifier threshold online; conservative-default prevents collapse
   exp3605  cross-domain synthesis v3 -- the corrected, de-contaminated verdict
   exp3606  G1-G4 gate-status synthesis v331 (paper_ready preserved)

 Phase 5  -- hardware continuity (mandatory, cheap) -------------------------
   exp3607  KV260 SSH continuity   exp3608  PolarFire continuity
   exp3609  GateMate audit-only

 Phase 6  -- capstone -------------------------------------------------------
   exp3610  Capstone v331 -- was '.329 math-only' a contamination artifact, or
            EARNED against a genuinely fair test?
```

### Dependency graph (gated_on edges use BARE fields only)

```
exp3597 -> exp3598 -> exp3599 --(corpus_v2_has_evidence)--> exp3600
                         |                                      |
                         +--(corpus_v2_is_realistic)--> exp3601 + (also reads exp3600)
                                                            |
                            exp3602 (independent)           +--(positive_control_valid)--> exp3603
                                                            |
   exp3600/3601/3602/3603 -> exp3604, exp3605 -> exp3606 -> exp3610
   exp3607 / exp3608 / exp3609 (independent hardware)
```

**Cascade-robustness:** unlike `.330`, gated_on is used only on **bare** boolean fields, and every
downstream task ALSO carries a `PRECONDITIONS` step that reads the upstream artifact and blocks honestly
(`blocked_*`) if it is missing — so a single broken edge degrades one row rather than wiping the
centerpiece. The math row of exp3601 is the frozen exp2837 0.9131 headline (no dependency, always lands).

---

## 4. Phase Descriptions

**Phase 1 (diagnose & audit).** Per the FALSE_NEGATIVE_RISK / Reading-Results disciplines, no fair re-test
can be trusted until the `.330` contamination is characterized. exp3598 confirms the gate-cascade root
cause, proves the exp3587 1.0 is a leak (the corpus has no evidence column), names the corpus-evidence
gap, and enumerates the per-domain structurally-applicable verifier sets. Its outputs are the spec every
Phase-2/3 task consumes.

**Phase 2 (build the apparatus).** The fair test needs two things `.330` lacked: a corpus with
**independent evidence** (so a grounding verifier has something legitimate to entail against — the
HaluEval QA `knowledge` passage, held out from the label) and a **real NLI verifier** (a small
DeBERTa/MiniLM-NLI checkpoint per the HalluSearch / VeriScore recipe, or an honestly-disclosed
text-statistical proxy if no checkpoint is fetchable — never a fabricated 1.0). exp3600's evidence MUST be
held out from the gold answer; a verifier that can see the label is not a verifier.

**Phase 3 (the fair re-test).** exp3601 is the centerpiece exp3588 skipped: the math|code|facts
generalization table, each domain with its applicable verifiers, on headroom-bearing corpora, vs a strong
confidence baseline, >=3 seeds, bootstrap CI. exp3602 runs the literature's exact positive control
(math->code PRM transfer) to distinguish "the `.330` code 0.44 was a wiring bug" from "code is genuinely
hard for Carnot." exp3603 re-derives the additive "second-pair-of-eyes" value with a paired McNemar test.

**Phase 4 (self-learning + synthesis).** exp3604 is the mandatory continuous-self-learning experiment with
a stated forward difference (online-calibrate the new grounding verifier via the conservative-default
rule). exp3605/exp3606 synthesize the corrected verdict and confirm the gate status.

**Phase 5 (hardware continuity).** One task per attached board per the Hardware-Task Continuity Discipline
(KV260 SSH-reachability, PolarFire reachability, GateMate audit-only — the flash/smoke host-IO path hangs
per known-issues). All audit-cheap.

**Phase 6 (capstone).** The honest milestone headline: was '.329 math-only' a contamination artifact or an
earned limitation, measured against a genuinely fair test?

---

## 5. Hardware Requirements

- **exp3599 / exp3600:** RTX 3090 (GPU) for SOTA-GGUF corpus generation/scoring and the NLI checkpoint.
  Mandated SOTA GGUF (>=1 of `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`,
  `unsloth/gemma-4-26B-A4B-it-GGUF`) via `cached_sota_pair()` when live generation is used; PRECONDITIONS
  block falls back to a fetched public dataset / disclosed proxy rather than fabricating.
- **exp3601 / exp3602 / exp3603 / exp3604:** CPU — `verifier_ensemble_against_cached_candidates`
  (score verifiers against cached labeled corpora; no LLM load).
- **exp3597 / exp3598 / exp3605 / exp3606 / exp3610:** CPU — `aggregation_from_upstream_artifacts`.
- **exp3607 / exp3608 / exp3609:** SSH-attached boards (`hardware_smoke`). KV260 precondition is SSH
  reachability ONLY (host `/dev/mmcblk*` checks are permanently retired).

---

## 6. Acceptance / Success Criteria

The milestone succeeds (regardless of the scientific direction) when:
1. The gate cascade does not recur (every gated field is bare; no `blocked_gate_check_failed` from a
   dict/bare mismatch).
2. The exp3587 AUROC = 1.0 is adversarially adjudicated (leak proven or refuted).
3. A factual corpus with independent held-out evidence + confidence headroom exists and is validated.
4. The centerpiece corrected cross-domain table (exp3601) actually lands with a `positive_control_valid`
   verdict.
5. The math->code positive control (exp3602) returns a verdict that explains the `.330` code 0.44.
6. The capstone states whether '.329 math-only' was an artifact or earned, with `paper_ready` still true.

---

## 7. Cross-References

- `ops/north-star.md` §1 (headline claim; every milestone advances it or is noise), §2 (G1-G4 gate)
- `research-references.md` 2026-06-01 sweep (the 9-finding literature scan + the `.330` cascade diagnosis)
- `scripts/conductor_gates.py:_eval_op` / `_coerce_gate_value` (the dict-vs-bare gate bug)
- exp3585/3586/3587/3588/3589/3591/3596 (the `.330` artifacts this milestone repairs)
- exp2837 (FoVer 0.9131 frozen math headline) · exp1999 (HumanEval code corpus)
- arXiv:2506.00027 (math->code positive control) · 2504.16828 (discriminative-fragility falsifier) ·
  2504.10168 / 2406.19276 (HalluSearch / VeriScore factual-grounding recipe)
- CLAUDE.md: Verifier Authenticity · Adversarial Artifact Verification (IMPLAUSIBLE_PERFECT) ·
  FALSE_NEGATIVE_RISK · Reading-Results · Pre-Launch Preconditions · Hardware-Task Continuity ·
  Gemini-Default · Principle-Annotated Artifact Fields
