# Claim-Scoping Proposal — narrow "hallucination" framing to "reasoning/constraint error" verification

**Status:** PROPOSAL for operator review. NOT applied (docs/index.html, README,
docs/technical-report.md prose are operator-curated per CLAUDE.md "Public
Documentation Discipline"). Drafted 2026-06-01 by the outer-loop after the .329
verifier cross-domain experiment.

## Why

Two independent lines of evidence say Carnot's verifier is a **structural
reasoning/constraint-error** detector, NOT a **factual-hallucination** detector:

1. **Pre-existing, already in the technical report (§3.2):** "activation EBMs
   detect model confidence, not factual correctness... Confident hallucinations
   are indistinguishable from confident correct answers... the answer was
   definitively no." The report already pivoted the project to *structural*
   verification.
2. **New (.329, preliminary; .331 de-contaminating + confirming):** the
   verifier ensemble's verifiers go to **AUROC ~0.50 (random)** on code bugs
   (exp3573) and factual claims (exp3574) — vs **0.9131 on FoVer math
   step-errors** (G1, G2-reproduced). Synthesis (exp3576):
   `verifier_value_generalizes_math_only_domain_bound`. On code, the ensemble
   (0.50) is a *worse* signal than the model's own confidence (0.90).

So the framing words **"hallucination" / "makes up" / "hallucination-resistant"**
imply factual-fabrication detection the product does not do. The examples on the
landing page are already honest (arithmetic errors); only the framing overclaims.

**This terminology scoping is warranted NOW by the existing report §3.2 alone.**
The .329-.331 cross-domain AUROC is confirming detail; wait for .331 to land
before citing the specific code/factual numbers (the first factual corpus was
contaminated — confidence AUROC 1.0, FLAGGED — and is being rebuilt).

## What is NOT a problem (leave alone)

- The **shipped code** (`carnot-ebm`): unaffected; it's a domain-appropriate
  verifier + repair pipeline. PyPI description "verifying and repairing LLM
  outputs" is broad but true.
- **"works with any LLM"**: fine — that's model-agnosticism (upstream Qwen/GPT/
  Gemma), not error-type generality.
- The **"second pair of eyes" card**: its example is `47 + 28 = 76` (a reasoning
  error) — already correctly scoped. Keep.
- The **0.9131 AUROC**: the result-card prose already says "on FoVer (5-seed
  dual-condition)". Only the top-of-page stat *label* is unscoped (see below).
- The **technical report** is already honest; this only *adds* a one-paragraph
  explicit scope statement, it does not retract anything.

## Proposed edits (operator applies)

### docs/index.html

| Loc | Current | Proposed |
|---|---|---|
| L7 meta | "...combining Energy-Based Models with LLMs **to reduce hallucinations**." | "...combining Energy-Based Models with LLMs **to catch reasoning and constraint errors in LLM output**." |
| Hero h1 | "Catch the mistakes your LLM **confidently makes up**." | "Catch the **reasoning errors your LLM states with total confidence**." |
| L308 stat label | "Verifier AUROC (5-seed dual-condition)" | "Verifier AUROC — FoVer math step-errors (5-seed)" |
| L663 CTA | "...a **hallucination-resistant** verify-and-repair function..." | "...a **reasoning-error-catching** verify-and-repair function..." |

### docs/technical-report.md — add an explicit SCOPE statement near §1.1 / the verifier section

> **Scope of the verifier.** Carnot performs *structural* verification: it
> catches reasoning, arithmetic, and constraint-satisfaction errors that have
> checkable structure (the FoVer math step-error setting, where the verifier
> ensemble reaches 0.9131 AUROC, independently reproduced — publication gate
> G2). It is **not** a factual-hallucination detector: §3.2 established that
> activation-based factual-hallucination detection fails, and cross-domain
> testing (milestones .329–.331) finds the structural verifier ensemble is
> domain-bound — its verifiers are near-random (AUROC ≈ 0.5) on code bugs and
> on factual claims, where a model-confidence baseline is the stronger signal.
> The honest product claim is a **reasoning-error verifier for structured
> domains**, not a general fact-checker.

*(Cite the specific code/factual AUROCs only after .331 confirms them on the
de-contaminated corpus.)*

## Apply conditions

1. The **terminology** edits (hallucination → reasoning/constraint error) can be
   applied now — they're justified by the existing report §3.2.
2. The **specific cross-domain AUROC numbers** in the report scope paragraph:
   wait for .331's de-contaminated re-measurement (real factual corpus v2 + NLI
   grounding verifier + math→code positive control) to land, then pin the
   numbers.
3. Run `python3 scripts/pages_fever_dream_lint.py` before committing any
   docs/index.html change.
