# Deep Think Prompt — AND-Composition Feasibility Ceiling

**Status:** Ready to send. Asks Deep Think to derive the realistic
upper bound on k for AND-composition over Carnot-style energy
verifiers, given the 2026-05-01 empirical r-correlation finding.
**Date drafted:** 2026-05-01
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project's Phase-3 architecture rests on a load-bearing
assumption: AND-composition over k=15+ energy verifiers
exponentially shrinks the joint null space `∩_i ker(E_i)` such that
no input fools all k verifiers simultaneously. The k=15 number
emerged from the original 9-round Zenil derivation chain.

**Today (2026-05-01) experiment 1093 measured the first empirical
data point:**

- 3 deployed Tier-0 text probes (NUP / SpilledEnergy / KAN-based
  SOSKANEnergyV3) on a fixed 200-pair text corpus
- Joint null-space fraction: **0.0** (no input fools all 3; good)
- Maximum pairwise r-correlation between verifier outputs: **0.66**
  (between SpilledEnergy and NUP)
- Acceptance threshold for AND-composition robustness was **r < 0.5**
- Verdict: `verifiers_correlated_diversity_needed` — the 3 probes
  share enough numerical-analysis signal that AND-composition
  effectively reduces to OR over a smaller equivalent set

The architectural assumption was that adding more verifiers
mechanically reduces the joint null space; the empirical finding
shows that adding more *of the same kind* doesn't.

### The question

**Given pairwise r-correlation ≥ 0.66 across 3 verifiers built from
the same broad family (text/numerical-feature probes), what is the
realistic upper bound on k for AND-composition where pairwise
r-correlation stays below 0.5?**

This is an information-theoretic / signal-processing question. The
answer determines whether the architecture's k=15 assumption is
achievable, whether a smaller k=5-7 with higher diversity would
give better guarantees, or whether AND-composition is fundamentally
bottlenecked at much smaller k by the structure of the input
distribution itself.

### Sub-questions to address

#### SQ1. Information-theoretic ceiling

Given an input distribution `p(x)` over LLM outputs (token sequences,
program code, theorem-prover output), how many *information-
theoretically independent* verifier mechanisms can extract signal
from the same input? Specifically:

- Treat each verifier as a function `f_i: X → R` mapping input to
  scalar energy. Pairwise r-correlation across verifiers measures
  the linear part of mutual information.
- The Friedrichs angle between verifier null spaces measures the
  geometric independence; r-correlation low does NOT imply
  Friedrichs angle high (r is one-dimensional projection, Friedrichs
  is geometric).
- Is there a theoretical bound of the form "no more than k* verifier
  mechanisms can achieve pairwise r < 0.5 on natural-language
  inputs"? If so, derive it.

#### SQ2. Topological diversity catalog

Enumerate the verifier mechanism families currently or plausibly
deployable against LLM outputs, ranked by mechanistic distinctness.
Candidates to evaluate:

| Family | Mechanism | Example |
|--------|-----------|---------|
| Numerical-feature probes | Hand-crafted features over token logits | NUP (next-token uncertainty), SpilledEnergy |
| Learned numerical probes | Trained classifier on hidden states | KAN-based SOSKANEnergyV3 |
| Formal-verification | Symbolic theorem prover | Z3-AST extractor over generated program |
| Step-level reasoning | Process-reward model | ThinkPRMProbe |
| Semantic embedding | Cosine similarity vs reference | sentence-transformer embedding |
| Output-format | Structural / regex match | JSON schema, extraction format |
| Runtime execution | Generated code actually runs | gVisor sandbox + functional test |
| Combinatorial encoding | Ising/Potts ground state on output graph | exp1098 Potts, exp1097 N-Queens |
| Self-consistency | k-shot agreement | majority vote over k samples |
| Mechanistic interpretability | Feature/circuit activation pattern | Goodfire-Silico-style neuron probe |

For each family, estimate (a) typical pairwise r-correlation against
the 3 existing text probes, (b) compute cost per verification, (c)
which input distributions it discriminates well (vs zero gradient).
Identify the subset of these families that empirically achieves
mutual r-correlation < 0.5.

#### SQ3. The k=5 vs k=15 tradeoff

Derive the joint-null-space-fraction guarantee at:

- **k=5** with worst-case r-correlation 0.3 across all pairs (high
  diversity, smaller composition)
- **k=15** with worst-case r-correlation 0.55 across all pairs
  (lower diversity, larger composition)
- **k=10** with worst-case r-correlation 0.4 across all pairs
  (intermediate)

Which gives the strongest guarantee? Show the math.

The intuition is that AND-composition's null-space-shrink rate is
exponential in k but degraded by correlation; specifically the
effective independent verifier count goes like `k * (1 - r)`. At
k=5 r=0.3 the effective count is 3.5; at k=15 r=0.55 it's 6.75; at
k=10 r=0.4 it's 6.0. But this naive linearization may be wrong;
verify or correct.

#### SQ4. Adaptive Verifier Synthesis fit

The Phase-4 architecture proposes Dynamic Verifier Synthesis (DVS)
that injects a new verifier `E_{k+1}` trained on accepted-but-
corrupt samples with quality threshold `Λ* = Z_{k+1}`. Given the
empirical r=0.66 finding, can DVS reliably find verifiers with
r-correlation < 0.5 against the existing suite? Or does DVS
plateau at the same correlation ceiling that produced r=0.66 in
the first place (the underlying input distribution doesn't have
that many independent feature axes)?

#### SQ5. Empirical validation path

Specify the empirical experiment that would resolve the k-ceiling
question with the strongest evidence. Should be runnable as a
single milestone-scoped experiment. Include:

- Hypothesis (e.g., "k_max with r < 0.5 is at most 8")
- Test corpus (e.g., 1000 LLM outputs spanning {natural-language,
  code, math reasoning, multi-step plans})
- Verifier suite candidates (top 8 from SQ2's family ranking)
- Metric: maximum k-element subset with pairwise r < 0.5
- Acceptance: empirical k_max compared to architectural k=15

### What I am NOT asking

- I am NOT asking for new defense theory beyond what Phase-3 already
  has. The question is empirically grounding the existing k=15
  assumption, not extending the architecture.
- I am NOT asking for sample efficiency analysis of the verifier
  TRAINING data (that's a separate concern).
- I am NOT asking to invalidate the Phase-3 architecture if the
  ceiling is below k=15. Recommend pivots and tradeoffs.

### Output format

1. **Executive summary** — 1 paragraph naming the realistic k-ceiling
2. **SQ1** — information-theoretic bound derivation
3. **SQ2** — verifier family ranking + which subset achieves r<0.5
4. **SQ3** — k=5 vs k=15 vs k=10 quantitative comparison with math
5. **SQ4** — DVS feasibility under correlation ceiling
6. **SQ5** — empirical experiment specification (the next .86 task)
7. **Recommended Phase-3 architecture amendment** — if k=15 is
   infeasible, what's the minimum viable k and what theoretical
   guarantees does it preserve?

### Honesty requirement

If the answer is "k=15 is infeasible without a fundamentally new
verifier mechanism family", say so. The architectural pivot to
"smaller k with higher Friedrichs angle" is acceptable; the
empirical reality must dictate.
