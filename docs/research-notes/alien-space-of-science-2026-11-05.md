# Alien Space of Science: reading note (2026-11-05)

Status: reading note. No experiment was run. Nothing here is a Carnot result.

Source: arXiv 2603.01092 v3 (2026-10-21), "Alien Space of Science". Read from the PDF text: abstract, method, results and discussion. The appendices were not read. Code release and replications were not checked.

## What the method is

- The authors extract "idea atoms" from 16,068 LLM papers. An LLM pulls out method statements. HDBSCAN (a clustering method) groups them into 273 atoms.
- A coherence model scores how likely a set of atoms is to appear in one paper. It is an autoregressive Transformer with 42.8M parameters. The score is a length-normalized log-likelihood.
- An availability model scores how close an existing author community is to a set of atoms. It is a dual encoder trained with InfoNCE on 39,791 authors. Availability A(S) is the median of the top-m author similarities. Unavailability U is -A.
- All 3,353,896 atom triples are ranked by F_beta = (1 - beta) * z_C + beta * z_U. The sampler uses beta = 0.7. The top 300 triples go to an LLM, which writes research ideas.

## Reported results

| Item | Result |
|---|---|
| Temporal test, frozen scorers (pre-2025 data only) | Top 0.875% of 1,143,135 triples recovers 31.4% of 2025 target combinations (35.9x chance) |
| Strictly novel targets, beta = 0 | 2.35x chance |
| Strictly novel targets, beta = 0.4 | 3.24x chance, 40 hits |
| Effective distinct ideas | 225.2 for this method, 56.5 for independent Claude sampling |
| Human idea quality | 4.03, tied with Claude (10 raters, 40 reviews) |

## Limits

- The authors say ideas only recombine existing concepts.
- The authors say availability is a lower bound derived from publications.
- The authors say the human study is small and there is no longitudinal validation.
- The outer loop saw no popularity or co-occurrence baseline in the main text. The "35.9x chance" figure is against a uniform random ranking, which is a weak null. The appendix may hold a stronger baseline. This is not checked.
- The novel-combination result is small: 40 hits.
- One domain only (LLM papers).
- Per-idea quality only ties Claude. Most quality evidence comes from LLM judges.

## What it means for Carnot

### Usable ideas

1. **Frozen-scorer temporal test.** Train and tune every scorer on data before a cutoff. Test only on data after it. This is a clean leakage-safe design. It fits the calibration pilot in the JEPA consensus plan (`jepa-two-model-consensus-plan-2026-11-05.md`, step 4).
2. **Planner diversity audit.** The paper finds LLM ideation clusters in dense regions. Our planner shows carry-forward bias. A cheap CPU check: count effective distinct tasks per roadmap, and compare roadmaps before and after the planner moved to Astra. This is a monitoring idea. It links to `covert-underperformance-safety-tasks-2026-11-05.md`. Print the denominator before drawing any conclusion.
3. **Score shape.** Two learned scores, one for plausibility and one for availability, combine in a weighted sum. It looks like a composite energy. It is not an energy-based model: the coherence score is a normalized autoregressive likelihood.

### Not applicable

- It is ideation evidence, not verification evidence. It makes no claim about verifier value.
- It does not touch the calibrated-decision floor or `GAP-ORACLE-DISTINCT`.
- It does not change any step in the JEPA consensus plan.

## Rule checks

- Circularity / Oracle-Distinctness: no verifier-value claim is made. The rule does not apply.
- ARC stays suspended. No ARC task follows.
- No weight update to the mandated generator.

## Verdict

Low value for the energy-based goals. One small follow-up is worth considering: the planner diversity audit, only if the operator wants planner monitoring. No experiment follows from this note.
