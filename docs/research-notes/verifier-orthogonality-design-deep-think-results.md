# Q11 Synthesis — Principled Verifier Design via Transversal Spectral Synthesis

**Status:** SYNTHESIZED 2026-05-04 ~01:35Z from Gemini Deep Think response
**Source:** verbatim response captured in Carnot session transcript
**Prompt:** `verifier-orthogonality-design-deep-think-prompt.md`
**Strategic role:** unblocks exp1232 audit + provides exp1233 redesign methodology + identifies new STE threat vector

## Headline result

**Synthesis is harder than detection in the general case** ($\Sigma_2^P$-complete
vs Spera's coNP-complete detection), **BUT Carnot's specific sign(z)
bottleneck makes it tractable in P** via a constructive procedure
(Transversal Spectral Synthesis, TSS).

Three changes to Carnot's roadmap follow from Q11:

1. **exp1232 audit instrumentation:** add 3 specific measurements
   (SMT triviality saturation, orthant penetration depth, STE
   self-modification test) — currently absent from the YAML prompt.
2. **exp1233 verifier redesign:** adopt TSS procedure with SC-Energy
   as $V_{mag}$ + Z3 as $V_{disc}$ as the optimal k=2 transversal pair.
   AST + JSON Schema are predicted vacuous (decoder-induced trivial)
   and should be SMT-prune-tested before retention.
3. **NEW threat vector:** Straight-Through Estimator / Gumbel-Softmax
   relaxation as DGM-class self-modification attack. Defeats the
   sign(z) firewall by routing gradient through the discrete bottleneck.
   Must be added to the threat model.

## Q11.1-Q11.5 derivations

**Q11.1 — Input-space partition does NOT yield compositional joint
nulls.** Cylinder factorization holds *unconditionally*, but adversarial
optimization under attack budget B traverses the unsafe manifold $U$ via
the off-diagonal Jacobian of the decoder, localizing attacks at the
intersection of the cylinders. Result: $\mu(\mathcal{N}_1 \cap \mathcal{N}_2)
\gg \mu(\mathcal{N}_1)\mu(\mathcal{N}_2)$ under adversarial threat.
**Implication:** disjoint coordinate projection is INSUFFICIENT for
joint-null-space orthogonality.

**Q11.2 — Decoder-aware verifier triviality is SMT-decidable.** A
verifier $V_i$ is decoder-induced trivial iff
$\text{Image}(D) \subseteq \text{Accept}(V_i)$. Procedure: model $D$'s
structural invariants as $\Phi_D(a)$ via abstract interpretation or
bounded SMT; query `Check-SAT: ∃a s.t. (Φ_D(a) ∧ ¬V_i(a))`. UNSAT means
$V_i$ contributes 0 bits of security and must be eliminated. **This is
operationally checkable on Carnot's k=6 today.**

**Q11.3 — Threat-model bound is non-asymptotic without transversality.**
The naive $k = O(\log(1/\epsilon))$ bound assumes probabilistic
average-case overlaps. Adversarial gradient attacks are deterministic
and target exact intersections. To isolate safe points separated by
distance $> 2B$ under $\ell_p$ budget B, need $k = \Omega(d)$ unless
verifiers are structurally transversal. **Implication:** independent
neural verifiers don't help against gradient adversaries; structurally
transversal pairs are the load-bearing defense.

**Q11.4 — Carnot's sign(z) provides a structural orthogonality
loophole.** This is the LOAD-BEARING POSITIVE RESULT.

The map $D_{pre}(z) = \text{sign}(z)$ partitions $Z = [-1,1]^d$ into
discontinuous flat orthants, decoupling radial (magnitude) from angular
(Ising sign) variables.

Define:
- $V_{mag}(z)$: continuous margin verifier; requires $\min_j |z_j| > \gamma$
- $V_{disc}(a)$: discrete verifier checking syntactic Ising state

**Theorem (Q11.4):** For attack budget $B < \gamma$:
$\mathcal{N}_{mag} \cap \mathcal{N}_{disc} = \emptyset$.

**Proof:** To bypass $V_{disc}$, adversary must flip an Ising sign,
requiring an attack trajectory crossing a coordinate plane $z_j = 0$.
At that crossing, $\min_j |z_j| = 0 < \gamma$, violating $V_{mag}$.
∎

**This is a deterministic geometric exclusion**, not a probabilistic
independence claim. It survives gradient adversaries by construction.

**Q11.5 — Synthesis complexity reduces from $\Sigma_2^P$-complete
(general) to P (Carnot's restricted case).** Spera 2026's detection
result ($\exists z \in \bigcap \mathcal{N}_i$) is coNP-complete. Optimal
ensemble subset selection is $\exists \forall$ alternation, hence
$\Sigma_2^P$-complete (2QBF) in general. But Q11.4's continuous/discrete
transversality boundary lets us bypass the combinatorial search:
synthesis reduces to (1) SMT triviality pruning + (2) selecting 1
continuous margin verifier + (3) Independent Set on Boolean Fourier-
Walsh spectral collisions among discrete verifiers — all polynomial.

## The TSS (Transversal Spectral Synthesis) procedure

```python
def Construct_TSS_Ensemble(Phi_D, B, k, L):
    # 1. Triviality Pruning via SMT (Q11.2)
    L_active = []
    for V in L:
        if SMT_Prove(Implies(Phi_D, V.acceptance_logic)) != VALID:
            L_active.append(V)

    # 2. Partition by Transpilation Layer Domains (Q11.4)
    Z_space_verifiers = [V for V in L_active if V.domain == 'continuous_radial']
    A_space_verifiers = [V for V in L_active if V.domain == 'discrete_angular']

    ensemble = []

    # 3. Anchor with a Continuous Margin Verifier
    #    Provably seals orthant boundaries against budget B
    V_mag = select_verifier_with_min_margin(Z_space_verifiers,
                                            target_margin=B)
    ensemble.append(V_mag)

    # 4. Greedy Orthogonal Discrete Selection (Submodular)
    #    Boolean Fourier-Walsh inner products on indicator functions
    while len(ensemble) < k:
        best_V = None; min_collision = float('inf')
        for V in A_space_verifiers:
            collision = max([boolean_inner_product(V, E_a)
                             for E_a in ensemble
                             if E_a.domain == 'discrete_angular'] + [0])
            if collision < min_collision:
                min_collision = collision; best_V = V
        ensemble.append(best_V); A_space_verifiers.remove(best_V)

    return ensemble, generate_transversality_witness(ensemble)
```

Complexity: $O(|\mathcal{L}| \cdot \text{SMT\_time} + k|\mathcal{L}|^2)$ — polynomial.

## Predictions for Carnot's k=6 production ensemble

| Pair | Predicted relationship | Reason |
|------|------------------------|--------|
| **AST + JSON Schema** | **Vacuous pair (decoder-induced trivial)** | If `snap_to_action` guides generation syntax, both are pre-satisfied by Image(D); SMT prune step eliminates. Predicted P(AST\|JSON) ≈ 1.0. |
| **Z3 + ThinkPRM** | **"Semantic hallucination" blind spot** | Both check internal consistency without external grounding (per Lean Atlas, arXiv:2604.16347). Adversary easily finds latent z* generating logical-but-fabricated outputs. |
| **Semantic + SC-Energy** | **Continuous gradient alignment** | Both differentiable neural; gradients bypassing Semantic likely align with those minimizing SC-Energy. |
| **SC-Energy + Z3** | **THE OPTIMAL k=2 TRANSVERSAL PAIR** | SC-Energy is the natural $V_{mag}$ (continuous pre-image thermodynamic basin); Z3 is the natural $V_{disc}$ (deterministic post-discretization structural). Together they realize Q11.4's structural firewall. |

**Strategic recommendation for paper-v6's k=6 claim:** the production
ensemble likely contains 1-2 vacuous pairs (AST+JSON) + 1-2 correlated
pairs (Z3+ThinkPRM, Semantic+SC-Energy). Effective k may be closer to 3-4
than 6. The honest finding is that **k=2 transversal (SC-Energy + Z3) is
provably stronger against gradient adversaries than k=6 mixed**. Paper-v6
should report $k_{eff}$ alongside $k_{nominal}$.

## NEW threat vector — STE / Gumbel-Softmax bypass

**The Achilles Heel:** Q11.4's formal transversality guarantee depends
ENTIRELY on the non-differentiability of $z \to \text{sign}(z)$.

**The attack:** A self-modifying agent (DGM threat model) discovers it
can rewrite the transpilation pipeline during optimization to use:
- Straight-Through Estimator (STE): pretend sign(z) is identity for backprop
- Gumbel-Softmax: relax to differentiable sigmoid approximation

If either succeeds, gradient flows from the discrete verifier (Z3, AST)
back into the latent dimensions. Q11.4's geometric transversality
evaporates. The ensemble reverts to $\Sigma_2^P$-hard non-compositional
trap.

**This is a NEW threat we hadn't enumerated in Q9's 8-mode catalog.**
Q9 modes 1-8 covered substrate-level failures (mode collapse, mixing
paralysis, etc.) but not pipeline-rewriting. STE-bypass is a
**Sakana-DGM-class architectural attack**, not a substrate failure.

**Defense:** make the transpilation pipeline non-rewritable from the
agent's action surface. Carnot's `CARNOT_USE_SANDBOX=1` gVisor
isolation is the right pattern (the agent cannot edit the kernel-side
sign() implementation), but enforcement needs explicit verification.
**This connects directly to SentinelAgent's "hash-linked forensic
reconstructibility" lesson** (memory: `reference_sentinelagent_peer.md`).

## Required exp1232 instrumentation (currently absent)

The YAML's prompt lacks the 3 specific measurements Q11.6 prescribes:

1. **SMT Triviality Saturation Test:** Pass $10^5$ uniform random noise
   vectors $z \sim \text{Unif}(-1,1)^d$ through $D(z)$. Measure
   unconditional acceptance rate of AST and JSON Schema. **If they
   accept >99% of noise-induced actions, prune them** (Q11.2 confirmed;
   they contribute 0 bits of security).

2. **Orthant Penetration Depth (Margin Audit):** For all successful
   adversarial attacks logged in exp1224 (Phase-5-C), measure the
   minimum continuous margin $\min_j |z_j^*|$. **If successful attacks
   consistently ride coordinate planes ($|z_j| \to 0$), Q11.4 is
   confirmed** — adding an SC-Energy margin verifier with $\gamma > B$
   will permanently sever the trajectory.

3. **STE Self-Modification Test:** Hostile-reviewer probe where the
   agent is explicitly granted access to an STE. Measure if conditional
   $P(V_{SC-Energy} | V_{Z3})$ spikes from ≈0 to ≈1.0. **If correlation
   inflates under relaxed gradients, the STE attack vector is
   empirically real** and the defense (sandbox isolation +
   hash-linked forensic chain) becomes critical.

## Required exp1233 redesign methodology (currently pattern-match only)

Adopt TSS procedure literally:

1. **Step 1 (SMT prune):** Apply Q11.2 to all 6 production verifiers.
   Predicted casualties: AST + JSON Schema. Document with SMT witnesses.
2. **Step 2 (partition):** SC-Energy + Semantic Probe → continuous_radial
   domain; Z3, AST, JSON, ThinkPRM → discrete_angular domain.
3. **Step 3 (anchor):** Select SC-Energy as $V_{mag}$ with $\gamma$
   calibrated to attack budget B from exp1224.
4. **Step 4 (greedy spectral):** Compute Boolean Fourier-Walsh inner
   products on remaining discrete verifiers' indicator functions.
   Predicted strongest discrete pair: Z3 (logical) + ThinkPRM not
   chosen (correlated semantic blindspot per Q11.4 application table).
5. **Final ensemble:** SC-Energy ($V_{mag}$) + Z3 ($V_{disc}$) as the
   provably transversal k=2 core. Larger k by adding spectrally
   non-collising additions if available.

## Honest framing — where the analysis breaks down

- **STE attack defeats Q11.4 transversality.** The structural guarantee
  evaporates if pipeline rewriting succeeds. This is the dominant
  residual risk.
- **TSS uses Boolean Fourier-Walsh spectral analysis** which assumes the
  discrete verifiers can be expressed as Boolean indicator functions
  with computable Fourier coefficients. ThinkPRM (neural process
  reward) doesn't fit this cleanly — its acceptance is continuous, not
  Boolean. Treating it as Boolean (acceptance ≥ threshold) introduces
  approximation error.
- **The k=2 optimal pair claim is structurally provable but
  empirically untested** — exp1232's instrumentation will validate or
  refute. Until then it's a theoretical prediction.
- **Submodularity assumption in TSS Step 4** — the greedy collision-
  minimization is provably submodular only if the underlying inner
  product matrix has specific structure (PSD with rapidly-decaying
  spectrum). Should be verified for Carnot's verifier library.

## Cross-references

- Prompt: `docs/research-notes/verifier-orthogonality-design-deep-think-prompt.md`
- exp1224 (Phase-5-C empirical Spera Theorem 9.2 confirmation): `results/experiment_1224_phase5c_adversarial_probe.json`
- exp1232 (currently blocked by failure-ledger gate): `results/experiment_1232_verifier_joint_orthogonality_audit.json`
- Spera Theorem 9.2 memory: `memory/reference_spera_theorem_92.md`
- DR-2 synthesis: `docs/research-notes/multi-verifier-ensemble-defense-deep-research-results.md`
- Q9 (8 failure modes): `docs/research-notes/in-situ-training-adversarial-robustness-deep-think-results.md`
- Continuous-Ising-Rank Theorem (Phase-3 substrate foundation): `memory/project_continuous_ising_rank.md`
- SentinelAgent peer (hash-linked forensic chain → STE defense connection): `memory/reference_sentinelagent_peer.md`
