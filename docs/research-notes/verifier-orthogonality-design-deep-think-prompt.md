# Deep Think Q11 — Principled Verifier Design for Joint-Kernel Orthogonality

**Status:** PROMPT DRAFT — for paste into Gemini chat UI Deep Think
**Drafted:** 2026-05-04 ~01:15Z
**Strategic role:** Provide principled methodology for exp1233 verifier redesign BEFORE it runs (.96 milestone)
**Predecessors:** Q9 (in-situ adversarial robustness), Q10 (NRGPT non-monotonicity), DR-2 (multi-verifier ensemble defenses)
**Empirical anchor:** exp1224 Phase-5-C adversarial probe found pairwise P(V_i|V_j) = 1.000 — the k=3 ensemble's effective independence collapsed to k=1 via decoder-induced verifier correlation

---

## Paste boundaries

```
START:  "## The Deep Think question" (line ~16)
END:    end of "## Output format requested" section
SKIP:   this header, "## Why this question now"
```

---

## The Deep Think question

I am building Carnot, an open-source energy-based-model framework that
defends against reward-hacking in self-improving agents (Sakana
Darwin-Gödel-Machine threat model). Defense relies on a k-AND-composed
ensemble of formally distinct verifiers — a candidate output is accepted
only if ALL k verifiers grade it as valid. The architectural intuition
is that intersection of acceptance regions across formally distinct
disciplines is exponentially smaller than any single verifier's
acceptance region.

I need formal guidance on how to *design* the verifier ensemble such
that joint null spaces are provably bounded — not how to *detect* them
post-hoc, which Spera (2026, arXiv:2603.15973) proves is coNP-complete.

### The empirical observation that motivates this question

Phase-5-C of my in-situ training derisking experiment ran a hostile-
reviewer probe against a k=3 ensemble: V0 (in_bounds: action targets a
valid grid cell), V1 (changes_grid: action modifies at least one cell),
V2 (no_duplicate_cells: action sequence doesn't target the same cell
twice). The decoder used a `snap_to_action` quadrant-anchor mechanism
that snapped continuous latents z ∈ [-1,1]^d to discrete action tokens.

The probe's pairwise correlation attack succeeded with **P(V_i | V_j) =
1.000 across all pairs**. Diagnosis: the snap_to_action quadrant
anchors structurally guaranteed V0 (in_bounds) for ALL inputs and V2
(no_duplicate_cells) for MOST. Effective independence collapsed:
k_effective = 1 (only V1 changes_grid provided genuinely independent
signal). The verifiers were topologically distinct (different formal
mechanisms — bounds-checking, change-detection, set-uniqueness) but
their joint null space was vacuous because the decoder geometry made
two of three checks trivially satisfiable.

This empirically realizes Spera Theorem 9.2 (Safety Non-Compositionality,
arXiv:2603.15973): topological distinctness of verifiers does NOT
guarantee disjoint acceptance regions; AND-composition CREATES emergent
conjunctive blind spots when decoder geometry pre-satisfies subsets of
verifiers.

### The core question

Given:

1. A specified decoder D mapping continuous latents z ∈ Z (a bounded
   manifold, e.g., [-1,1]^d) to discrete actions a ∈ A.
2. A specified threat model T (e.g., adversarial gradient-based attack
   on z under a budget B, attempting to find z* such that D(z*) is
   accepted by V_full but violates ground truth oracle O).
3. A budget k of verifiers we can run at decision time.

Derive a constructive procedure that selects k verifiers V₁, ..., V_k
from a verifier-template library (or constructs them de novo) such that:

- Each V_i is sound for its individual specification (V_i(D(z)) = 1
  implies a partial correctness property holds).
- The pairwise joint null spaces |Null(V_i ∩ V_j)| are bounded above by
  some ε (small).
- The k-fold joint null space |∩ᵢ Null(V_i)| is bounded above by ε^c
  for some c ≥ 1 (composition gain).
- The bound is verifiable at design time — i.e., the procedure must
  yield witnesses or proofs of joint-null-space dimension that don't
  require post-hoc coNP-hard search.

### Specific sub-questions to engage

**Q11.1 — Can joint-null-space orthogonality be made compositional via
input-space partition?** If verifiers operate on disjoint input-space
projections (V_i depends only on coordinate subset S_i ⊆ {1,...,d},
with S_i ∩ S_j = ∅), is their joint null space provably the product of
individual null spaces? Or does the decoder D's coupling re-introduce
correlation even under disjoint projections?

**Q11.2 — Decoder-aware verifier synthesis.** Given decoder D's
structural guarantees (e.g., snap_to_action makes V0 vacuous), can we
formally characterize "decoder-induced trivial verifiers" — verifiers
whose acceptance region ⊇ Image(D) — and exclude them from the
ensemble at design time? What is the algorithmic procedure?

**Q11.3 — Threat-model-conditional design.** Given threat model T's
attack budget B, is there a budget-relative bound: "to defend against
attacks at budget B, k = O(log(1/ε)) formally distinct verifiers
suffice if pairwise joint null space < ε"? Or is the relationship
non-asymptotic? Derive the bound or a counterexample.

**Q11.4 — Continuous-latent → discrete-action transpilation.** Carnot's
Phase-3 DBAE-EBM substrate maps continuous z ∈ [-1,1]^d → sign(z) →
Ising → discrete actions. The decoder is *not* an arbitrary function
— it's a sign-discretization. Does this specific structure admit
stronger verifier-independence guarantees than the general case?
Specifically: are there verifier templates (operating on sign(z) directly,
on z's magnitude, on z's gradient, etc.) whose joint null spaces are
provably disjoint by the discretization structure?

**Q11.5 — Synthesis vs detection complexity gap.** Spera 2026 proves
detection is coNP-complete (calculating minimal unsafe-set membership
in AND-composed ensembles is intractable). Is synthesis (constructing
ensembles with bounded joint null space) easier? Provide a complexity
result: either an efficient construction algorithm, or a hardness
result showing synthesis is also coNP-hard / Σ₂-hard / etc.

### Output format requested

Please structure as:

1. **Executive summary (3-5 paragraphs).** State whether
   joint-kernel-orthogonality verifier design is solvable in general,
   under what restrictions, and what the cleanest result is.

2. **Per sub-question (Q11.1 – Q11.5).** For each: a derivation or proof
   sketch, with explicit assumptions and the resulting bound or
   counterexample.

3. **Constructive procedure (if one exists).** Algorithm pseudocode
   for: input (D, T, k, library of verifier templates), output (k
   verifiers with bounded joint null space + witness/proof). Include
   computational complexity of the procedure.

4. **Application to Carnot's k=6 production ensemble.** Given the
   actual production verifiers — Z3 SMT solver, AST structural check,
   semantic embedding probe, ThinkPRM process reward model, JSON
   schema validator, SC-Energy set-consistency network — apply the
   procedure (or its restricted form) and predict which pairs are most
   likely to share large joint null space. Spell out the structural
   reasoning, e.g., "Z3 + ThinkPRM share null on logically-consistent-
   but-semantically-empty outputs because both check internal
   consistency without grounding."

5. **Honest framing.** Where does the analysis break down? Identify the
   weakest assumption or step. If joint-orthogonality construction is
   provably hard in the general case, say so explicitly and identify
   the specific Carnot-relevant restriction that makes it tractable.

6. **Open questions for empirical follow-up.** What measurements should
   exp1232 (orthogonality audit) report to confirm or refute the
   theoretical predictions in (4)?

### Format constraints

- Use formal notation where helpful (set-theoretic null space, Hilbert
  space inner products on indicator functions, ensemble cardinality
  bounds).
- Cite related work explicitly: Spera 2026 (arXiv:2603.15973), CAF
  (AAAI 2026), SentinelAgent (arXiv:2604.02767), BEAVER-lite
  (arXiv:2512.05439), Lean Atlas (arXiv:2604.16347).
- For controversial or speculative steps, flag with "Speculative:"
  and provide alternative interpretations.
- For results that depend on unverified assumptions about Carnot's
  decoder/substrate (which you may not know in detail), state the
  assumption explicitly and how the conclusion would change if false.

---

## Why this question now (decision-leverage)

exp1224 (Phase-5-C) just empirically confirmed Spera Theorem 9.2 on
Carnot's k=3 ensemble. exp1232 (orthogonality audit, .96 mandatory)
will measure the production k=6 ensemble's pairwise correlation
matrix. exp1233 (verifier redesign, .96) will replace correlated
pairs.

Currently exp1233's design space is "pattern-match V3/V4 candidates
from exp1224." Without principled methodology, the redesign is
calibration-by-intuition — same failure mode that gave us a vacuously
correlated k=3 originally.

A Deep Think round on principled construction BEFORE exp1233 fires
upgrades it from pattern-match to formal-procedure. Cost is ~10
minutes of operator paste time + Deep Think compute (Gemini Ultra,
not on Carnot's quota); benefit is exp1233 ships a redesign with
provable bounds, and the methodology generalizes to all future
verifier additions.

The cost asymmetry: ~30 minutes of theoretical work now vs the
embarrassment of paper-v6 reviewers asking "you measured k=6
correlation; what's your principled procedure for choosing the
replacements?" and having no answer.

## Cross-references

- exp1224 artifact: `results/experiment_1224_phase5c_adversarial_probe.json`
- Spera Theorem 9.2 memory: `memory/reference_spera_theorem_92.md`
- DR-2 synthesis: `docs/research-notes/multi-verifier-ensemble-defense-deep-research-results.md`
- Phase-5 derisking proposal: `openspec/change-proposals/in-situ-training-phase5-derisking.md`
- Continuous-Ising-Rank Theorem (Phase-3 substrate foundation): `memory/project_continuous_ising_rank.md`
- Pathological joint null space (exp1108 precedent): `memory/project_pathological_joint_null_space.md`
