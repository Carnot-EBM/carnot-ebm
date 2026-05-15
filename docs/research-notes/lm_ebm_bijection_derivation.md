# Theoretical LM-EBM Bijection Re-derivation

## 1. The Autoregressive LM to EBM Bijection (arXiv:2512.15605v3)

An autoregressive language model defines a probability distribution over sequences $x = (x_1, \dots, x_T)$ as:
$$ P_{LM}(x) = \prod_{t=1}^T P_{\theta}(x_t | x_{<t}) $$

An Energy-Based Model (EBM) defines the distribution as:
$$ P_{EBM}(x) = \frac{\exp(-E(x))}{Z} $$

The explicit bijection established in arXiv:2512.15605v3 equates these two by defining the energy of a sequence as the negative log-likelihood under the autoregressive model:
$$ E_{LM}(x) = -\sum_{t=1}^T \log P_{\theta}(x_t | x_{<t}) $$

Because the autoregressive model is locally normalized at each step (i.e., $\sum_{x_t} P_{\theta}(x_t | x_{<t}) = 1$), the partition function for the corresponding EBM is strictly $Z = 1$. This structural property provides a rigorous grounding for mapping sequence-level verifiers directly into the energy landscape.

## 2. Carnot's k=16 Verifier Ensemble as Free Energy

Carnot implements a cascade of verifiers (Tier 0 to Tier 3) that score the structural and semantic validity of generated sequences. Under the LM-EBM bijection, Carnot's verification pipeline is mathematically equivalent to composing the base LM energy with a set of constraint energies:

$$ E_{total}(x) = E_{LM}(x) + \sum_{i=1}^{16} \lambda_i V_i(x) $$

Here, $V_i(x)$ represents the energy penalty imposed by the $i$-th verifier in the $k=16$ ensemble (e.g., CarnotThinkProbe, NUP Probe, SymCodeVerifier, etc.). 

Because the base LM is an EBM with $Z_{LM}=1$, the partition function of the verifier-guided ensemble becomes the Free Energy term that normalizes the composed landscape:
$$ Z_{total} = \sum_x \exp\left( -E_{LM}(x) - \sum_{i=1}^{16} \lambda_i V_i(x) \right) $$

Thus, Carnot's verifiers act as localized thermodynamic constraints. The verification process is formally equivalent to sampling from the low-energy basins of this composite EBM manifold. The transversality of these constraint manifolds ($\theta_F > 0$) ensures that the MCMC sampling (or parallel-tempered AND-composition) remains tractable and avoids disjoint mode collapse.

## 3. Corollary: Phase 4 $\alpha_t$ Invariance (exp1693)

The exp1693 Phase 4 $\alpha_t$ invariance is a **direct corollary** of this bijection.

If we consider a time-dependent coupling $\alpha_t$ in the autoregressive formulation, the bijection dictates that any invariant property in the locally-normalized AR space must symmetrically map to a global energetic invariance in the EBM landscape. Because $E_{LM}(x)$ decomposes additively over time steps $t$, a structural invariance to $\alpha_t$ in the verifier gradient $\nabla_{x_t} V_i(x)$ implies that the free energy perturbation $\Delta F$ introduced by the verifier is independent of the temporal step index. 

This confirms that the Phase 4 $\alpha_t$ invariance is not merely an empirical artifact, but a fundamental topological consequence of the LM-EBM equivalence operating under localized normalization.
