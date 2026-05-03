# In-Situ Training — Phase-5 Architectural Derisking

**Status:** UPDATED 2026-05-03 ~19:35Z post Deep Think Q9 review
**Target milestones:** .94/.95 (small-scale) + .96/.97 (intermediate-scale)
**Estimated scope:** 4 small-scale + 1 intermediate-scale experiment, ~3-6 weeks total
**Strategic role:** Bridge Phase-4 (sampler operational) → Phase-5 (online learning)

## Why this exists

Carnot's architecture is uniquely suited to in-situ training: model updates DURING inference using the verifier ensemble as supervision. This addresses Sakana DGM's open problem (reward hacking) directly because the verifier doesn't co-train.

Full Phase-5 deployment is ~6-12 months out (requires Phase-3 substrate trained at 1B+ scale). Small-scale + intermediate-scale validation derisks early. Cost asymmetry: ~3-6 weeks now to catch architectural failure modes vs 100-500 GPU-hours of substrate training that may need to be redone.

## What Q9 Deep Think changed

The original 4-experiment plan targeted ALL architectural commitments. Q9 review identified that **3 of 8 failure modes are STRUCTURALLY UNDETECTABLE at toy scale (50K params, d~16)**:

- Mode collapse (overparameterized degenerate attractors) — requires capacity
- MCMC mixing paralysis (PCD distillation degradation) — requires high-d landscape
- Substrate shift (measure concentration in [-1,1]^d) — geometric phase transition

These three only manifest at production scale (1B+ params, d=256-1024). Toy-scale validation is necessary but insufficient. Plan now has TWO TIERS:

1. **Small-scale (.94/.95)** detects 5 of 8 failure modes at toy scale
2. **Intermediate-scale (.96/.97)** catches the 3 production-scale-only modes before full deployment

## Q9 failure mode catalog (Deep Think classification)

| # | Mode | Q9 detectability at toy | Severity |
|---|---|---|---|
| 1 | Mode collapse | NO | Critical |
| 2 | MCMC mixing paralysis | NO | Critical |
| 3 | Substrate shift (measure concentration) | NO | Critical |
| 4 | Gradient-driven null space excavation | YES | High |
| 5 | Correlated evaluator blind spots | YES | High |
| 6 | Unbounded representation drift | YES | High |
| 7 | Autocatalytic acceptance spiral | YES | High |
| 8 | Capability overwriting | YES | High |

Three honest unresolvables at any scale (Q9-flagged):
- Worst-case joint null space size (no a-priori bound exists)
- NP-hard threshold for crafting attacks (continuous EBM circumvents discrete bounds)
- Diverse degeneracy vs diverse correctness (unresolvable from internal metrics; requires frozen oracle)

## Required infrastructure (across all experiments)

Before any experiment runs, ship:

- **Ground-truth oracle** as continuous evaluator. Frozen ARC-AGI training set (or task-domain equivalent) used as the ultimate correctness signal. Every metric pivots on comparison to this oracle.
- **Held-out task buffer** for retention measurement. Periodic eval (every 50 queries) on broad-coverage benchmark hidden from the in-situ loop.
- **Vacuous-anchor library** for null-space-excavation detection. Curated set of known shared-null-space examples (cf. exp1108 vacuous dead code). Latent distance to anchors tracked per query.

## Phase 1 — Small-Scale Mechanism Validation (.94 or .95)

### exp_NEXT_A — Minimal in-situ substrate prototype

**Scope:** small EBM substrate end-to-end at toy scale.

- Encoder: small CNN, 5×5 grid → z ∈ [-1,1]^16
- Energy network: ~10K params, MLP with bounded outputs
- Decoder: snap to action sequence (Q8 Option A)
- Total: ~50K params

**Q9 additions:**
- Vacuous-anchor distance tracking instrumented from start
- Verifier conditional acceptance probability matrix instrumented from start

**Acceptance:** prototype runs end-to-end on 100 random 5×5 puzzles, produces valid action sequences for ≥50%, anchor-distance + conditional-acceptance matrices recorded.

**Cost:** 4-6 GPU-hours, 1 week.

### exp_NEXT_B — In-situ training loop with verifier-ensemble grounding

**Scope:** PCD update step using k=3 verifier subset (Z3-AST + AST structural + ThinkPRM).

**Update mechanism** (per query):
1. Encode → sample → verify (k=3 AND-compose) → return best
2. IF verifier-AND PASSES: PCD update (η=1e-5)
3. Persist updated weights

**Q9 additions:**
- Ground-truth oracle eval every 50 queries
- Encoder weight spectral norm tracking
- Acceptance-rate first-derivative monitoring (detects autocatalytic spiral)
- Vacuous-anchor distance per query

**Acceptance:** 1000-query trajectory with ALL Q9-detectable failure mode signatures within bounds:
- E(z_accepted) decreases ≥30% (mechanism works)
- Encoder spectral norm growth rate < threshold (no representation drift)
- Acceptance-rate first derivative is sub-linear (no autocatalytic spiral)
- Vacuous-anchor distance stays >threshold (no null-space excavation)
- Held-out oracle accuracy retained within 5pp (no catastrophic forgetting)

**Cost:** 8-12 GPU-hours, 1-1.5 weeks.

## Phase 2 — Adversarial Probe (.95)

### exp_NEXT_C — Hostile-reviewer probe (Q9-aware)

**Q9 added attack class:** correlated evaluator blind spots. The original probe targeted single-verifier gaming. Q9-revised probe explicitly tests:

- Single-verifier gaming (original threat model)
- **Pairwise verifier correlation exploitation** (new — verifier ablation maps conditional acceptance probabilities; attacker targets P(V_i|V_j) > P(V_i) regions)
- **Joint null space attack** (new — gradient search for inputs that land in shared null space of all k=3 verifiers simultaneously)

**Acceptance:** EITHER no gaming found within budget B (architecture validated for these 3 attack classes at toy scale)
OR gaming found and the failure mode + which architectural revision blocks it documented as research deliverable.

**Cost:** 6-8 GPU-hours, 1 week.

## Phase 3 — Intermediate-Scale Validation (.96 or .97)

### exp_NEXT_E — Intermediate-scale derisking (~100-300M params)

**Why required:** Q9 identified 3 failure modes that ONLY manifest at production scale. Going from 50K params directly to 1B+ deployment skips a regime where these modes might appear and could be cheaply caught.

**Scope:**
- Substrate: ~100-300M params (between toy 50K and production 1B+)
- Domain: real ARC-AGI-1 or ARC-AGI-2 puzzles (real distribution, not synthetic 5×5)
- Latent dim: d=128-256 (between toy d=16 and production d=256-1024)
- Verifier ensemble: k=5+ (full production set, not k=3 subset)
- Duration: 10K queries

**Q9 acceptance gates** (all 5 detectable + 3 production-scale modes):

```
Detectable-at-toy modes (5):
  4. gradient-driven null space    : vacuous-anchor distance stays >threshold
  5. correlated evaluator blind    : conditional acceptance independence holds
  6. unbounded representation drift: encoder spectral norm bounded
  7. autocatalytic acceptance spiral: acceptance-rate derivative sub-linear
  8. capability overwriting        : held-out oracle accuracy within 5pp

Production-scale modes (3):
  1. mode collapse                 : conditional output entropy stays >threshold,
                                      latent variance preserved across diverse inputs
  2. MCMC mixing paralysis         : Gibbs autocorrelation stays <threshold,
                                      L2(positive_z, negative_z) stays >threshold
  3. substrate shift               : L∞(z) doesn't saturate at 1.0,
                                      dimensional histograms stay multimodal
```

**Cost:** 30-60 GPU-hours, 2-3 weeks.

**Comparator-set benchmarks (NEW 2026-05-03 21:55Z post Deep Research):**

Per Deep Research's "Architectures to Explicitly Compare Against" guidance, exp_NEXT_E must include head-to-head benchmark comparisons against the open-source non-AR comparator set:

```
Comparator         | Citation                  | Comparison axis          | Why it matters
NRGPT              | arXiv:2512.16762          | per-token energy + AUROC | Carnot's external grounding differs from NRGPT's purely internal integration
LLaDA              | arXiv:2502.09992 (8B)     | text-task AUROC          | Gold standard for open-source non-AR; Carnot must explain how it differs from masked diffusion
Coconut            | arXiv:2412.06769          | latent-reasoning AUROC + | Demonstrates training curriculum efficiency vs Coconut's multi-stage unrolling
                                                training compute
```

If exp_NEXT_E exists without these comparator benchmarks, the paper-v6 reviewers (per Deep Research's strategic-positioning guidance) will flag the omission. Including them costs ~5-10 additional GPU-hours but is mandatory for paper-v6 publication readiness.

## Acceptance criteria for "Phase-5 derisked"

```
SMALL-SCALE (after exp_NEXT_A-C):
  5/5 detectable failure modes absent at toy scale
  → Phase-5 architecture viable at toy scale; INTERMEDIATE-SCALE NEXT

INTERMEDIATE-SCALE (after exp_NEXT_E):
  8/8 failure modes absent at intermediate scale
  → Phase-5 architecture validated for production scale-up
  → Substrate training at 1B+ proceeds with confidence

DEFERRED:
  3 honest unresolvables remain after both stages.
  Worst-case joint null space size, NP-hard crafting threshold, and
  diverse-degeneracy-vs-correctness distinction require continuous
  ground-truth-oracle monitoring at production deployment.
```

## Critical scope-limit declaration

**This proposal does NOT claim the architecture is validated for production after the small-scale phase.** Per Q9, 3 critical failure modes only manifest at production scale. Even after exp_NEXT_A-C all pass, the only valid claim is "5 of 8 modes detected absent at toy scale; 3 production-scale modes pending intermediate-scale validation."

This is the most important correction Q9 surfaced. Earlier framing risked overclaiming.

## Strategic alignment

- **Bridges Phase-4 → Phase-5** with two staged validation tiers
- **Empirically grounded** for paper-v6 eAI section's Sakana-defense claim
- **Aligned with CLAUDE.md MANDATORY** (Phase Prototype + Empirical Validation + Adversarial Check)
- **Q9-honest** — explicit about what toy-scale CANNOT validate
- **Decentralization-respecting** — operates locally; per-user model copies preserve sovereignty

## Cross-references

- Q9 prompt: `docs/research-notes/in-situ-training-adversarial-robustness-deep-think-prompt.md`
- Q9 results: `docs/research-notes/in-situ-training-adversarial-robustness-deep-think-results.md`
- Sakana DGM threat model: `memory/reference_sakana_dgm.md`
- PNAS Breeder Scenario: `memory/reference_pnas_evolvable_ai.md`
- Paper-v6 eAI section draft: `docs/research-notes/paper-v5-decentralization-section-draft.md`
