# Research-Informed Roadmap: Five Proposals from ArXiv Papers

**Created:** 2026-04-04
**Status:** Planned
**Priority order:** P1 → P2 → P3 → P4 → P5

These proposals integrate ideas from recent EBM research into Carnot. Each is self-contained and independently implementable. They build on the existing infrastructure (verify-and-repair pipeline, learned verifiers, autoresearch, LLM solver) and move Carnot closer to the end goal of LLM+EBM anti-hallucination.

---

## P1: Semantic Energy Hallucination Detector

**Source:** [Semantic Energy: Detecting LLM Hallucination Beyond Entropy](https://arxiv.org/abs/2508.14496) (Aug 2025)
**Effort:** Low (1 session)
**Impact:** High — first-line hallucination detection before constraint checking

### Key Idea

Energy = negative logit from the LLM's output layer. High energy = low confidence = likely hallucination. The paper showed 4-5% AUROC improvement over entropy-based methods, and 13% in edge cases where all responses are semantically identical.

Formula: `U(x) = (1/T) Σ_t -z_θ(x_t)` (average negative logit across tokens)

### What to Build

**New file: `python/carnot/inference/semantic_energy.py` (~150 lines)**

```python
def compute_semantic_energy(logprobs: list[float]) -> float:
    """Compute semantic energy from LLM token logprobs.
    
    Energy = mean(-logprob) across tokens. High = likely hallucination.
    Uses logprobs from OpenAI API (logprobs=True parameter).
    """

def classify_hallucination(energy: float, threshold: float = 2.0) -> bool:
    """True if energy exceeds threshold (likely hallucination)."""

class SemanticEnergyConstraint(BaseConstraint):
    """Wraps semantic energy as a constraint for ComposedEnergy.
    
    Can be composed with hand-coded and learned constraints for
    multi-signal verification: LLM confidence + structural checks + learned patterns.
    """

def solve_with_logprobs(config: LLMSolverConfig, prompt: str) -> tuple[str, list[float]]:
    """Call LLM with logprobs=True, return (response, token_logprobs)."""
```

**Modify: `python/carnot/inference/llm_solver.py`**
- Add `logprobs=True` to API calls
- Extract token logprobs from response
- Compute semantic energy alongside structural verification
- Add `semantic_energy` field to `VerifyRepairResult`

**New spec: `openspec/capabilities/llm-ebm-inference/spec.md`**
- REQ-INFER-008: Semantic energy from LLM logprobs
- SCENARIO-INFER-009: High-energy responses flagged as hallucinations

### Integration Point

The semantic energy becomes a "pre-filter" in the verify-and-repair pipeline:
```
LLM response + logprobs
    → semantic_energy(logprobs)  ← NEW: first-line confidence check
    → parse assignment
    → hand-coded constraint verification
    → learned model verification
    → gradient repair if needed
    → certificate
```

### Test Strategy

- Mock LLM responses with varying logprob distributions
- Verify high-confidence (low energy) vs low-confidence (high energy) classification
- Integration test with `verify_and_repair()` pipeline
- ~10 tests, all referencing REQ-INFER-008

---

## P2: Multi-Start Self-Verification

**Source:** [Energy-Based Transformers are Scalable Learners and Thinkers](https://arxiv.org/abs/2507.02092) (July 2025)
**Effort:** Low (1 session)
**Impact:** Medium — improves repair success rate

### Key Idea

Instead of repairing from a single starting point, generate M candidates (random perturbations of the LLM's answer), optimize each via gradient descent, and select the one with minimum energy. The EBT paper showed 10-14% improvement from self-verification.

### What to Build

**New file: `python/carnot/inference/multi_start.py` (~120 lines)**

```python
def multi_start_repair(
    assignment: jax.Array,
    energy: ComposedEnergy,
    n_starts: int = 5,
    perturbation_scale: float = 0.1,
    step_size: float = 0.1,
    max_repair_steps: int = 200,
    round_fn: Callable | None = None,
    key: jax.Array | None = None,
) -> VerifyRepairResult:
    """Run repair from N random perturbations, return best result.
    
    Algorithm (from EBT paper, Algorithm 2):
    1. Generate N perturbations: x_i = assignment + noise_i * perturbation_scale
    2. Run repair() on each x_i independently
    3. Round each to discrete domain
    4. Select the result with minimum energy after rounding
    5. Return that result with all trajectories
    """

@dataclass
class MultiStartResult(VerifyRepairResult):
    """Extends VerifyRepairResult with multi-start metadata."""
    n_starts: int = 0
    all_final_energies: list[float] = field(default_factory=list)
    best_start_index: int = 0
```

**Modify: `python/carnot/inference/verify_and_repair.py`**
- Add `n_starts` parameter to `verify_and_repair()` — when > 1, delegates to `multi_start_repair()`
- Backward compatible: default `n_starts=1` preserves current behavior

**New spec additions:**
- REQ-INFER-009: Multi-start self-verification
- SCENARIO-INFER-010: Multi-start achieves lower energy than single-start on SAT

### Test Strategy

- Verify multi-start finds lower energy than single-start on a tricky SAT instance
- Verify backward compat (n_starts=1 = current behavior)
- Verify best result is selected correctly
- ~8 tests

---

## P3: Self-Normalised Likelihood (SNL) Training Loss

**Source:** [Learning EBMs by Self-Normalising the Likelihood](https://arxiv.org/abs/2503.07021) (March 2025)
**Effort:** Medium (1 session)
**Impact:** Medium — better training for learned verifiers

### Key Idea

The partition function Z is intractable for EBMs. NCE sidesteps it via classification. SNL takes a different approach: treat log(Z) as a single learnable parameter and optimize a lower bound of log-likelihood. The bound is tight at the optimum and concave for exponential families.

Formula: `L_SNL(θ, c) = E_data[-E_θ(x)] - c - E_proposal[exp(-E_θ(x) - c)]`

where c ≈ log(Z) is a learnable scalar.

### What to Build

**New file: `python/carnot/training/snl.py` (~120 lines)**

```python
def snl_loss(
    energy_fn: HasEnergy | callable,
    data_batch: jax.Array,
    proposal_batch: jax.Array,
    log_z: jax.Array,  # learnable scalar (the normalization estimate)
) -> jax.Array:
    """Self-Normalised Likelihood loss.
    
    L = -mean(E(data)) - log_z - mean(exp(-E(proposal) - log_z))
    
    Advantages over NCE:
    - Lower bound of true log-likelihood (tighter than NCE)
    - Concave in θ for exponential families
    - No MCMC needed (like NCE) but better gradient signal
    """

def snl_loss_stochastic(
    energy_fn, data_batch, noise_scale, log_z, key
) -> jax.Array:
    """SNL with freshly sampled proposal noise."""
```

**Modify: `python/carnot/training/__init__.py`**
- Export `snl_loss`, `snl_loss_stochastic`

**Modify: `python/carnot/inference/learned_verifier.py`**
- Add option to train with SNL instead of NCE: `training_method="nce"` or `"snl"` in `LearnedVerifierConfig`

**New spec:**
- REQ-TRAIN-004: Self-normalised likelihood training
- SCENARIO-TRAIN-004: SNL converges to lower loss than NCE on same data

### Test Strategy

- Verify SNL loss is finite and decreases during training
- Verify log_z converges to approximate true log(Z) for simple distributions
- Compare SNL vs NCE training on learned SAT verifier (accuracy comparison)
- ~10 tests

---

## P4: ARM↔EBM Bijection — Extract Energy from LLM Logits

**Source:** [Autoregressive Language Models are Secretly Energy-Based Models](https://arxiv.org/abs/2512.15605) (Dec 2025)
**Effort:** Medium-High (1-2 sessions)
**Impact:** High — unifies LLM and EBM into single model

### Key Idea

Every autoregressive LLM is an EBM in disguise. The bijection:
- `r(s, y) = q(s, y) - V_q(s ⊕ y)` — immediate reward (energy contribution per token)
- `q(s, y)` = LLM logit for token y given context s
- `V_q(s)` = log-sum-exp of logits = soft value function

This means we can extract a per-token energy signal from any LLM without training a separate EBM. The "energy" of a complete sequence is the sum of per-token rewards: `E(y|x) = -Σ_t r(s_t, y_t)`.

### What to Build

**New file: `python/carnot/inference/arm_ebm_bridge.py` (~200 lines)**

```python
def extract_token_rewards(
    logits: list[list[float]],  # per-position logit vectors
    token_ids: list[int],       # chosen tokens
) -> list[float]:
    """Extract per-token immediate reward via ARM-EBM bijection.
    
    r(s_t, y_t) = logit(y_t) - logsumexp(logits_t)
    
    This is the log-probability — but framed as energy, it reveals
    which tokens the model is "confident" about (low |r|) vs 
    "uncertain" about (high |r|).
    """

def sequence_energy(token_rewards: list[float]) -> float:
    """Total sequence energy = -sum(rewards). Lower = more confident."""

def identify_hallucination_tokens(
    token_rewards: list[float],
    threshold: float = -2.0,
) -> list[int]:
    """Find token positions where reward is below threshold (uncertain).
    
    These are the specific tokens most likely to be hallucinated.
    Enables surgical repair: only fix the uncertain tokens.
    """

class ARMEBMConstraint(BaseConstraint):
    """Wrap ARM-derived energy as a constraint for the pipeline.
    
    Requires access to full logit vectors (not just logprobs).
    Works with any model that exposes logits via API.
    """
```

**Challenge:** The OpenAI API only exposes `logprobs` (log-probabilities of chosen tokens), not the full logit vectors needed for `V_q = logsumexp(logits)`. Options:
1. Use `logprobs` as an approximation (P1 already does this)
2. Use a local model (vLLM, Ollama) that exposes full logits
3. Derive V_q from top-k logprobs (approximate logsumexp)

### Depends On

- P1 (semantic energy) provides the simpler logprobs-based version
- This extends P1 with the full theoretical framework

### Test Strategy

- Verify bijection: r(s,y) + V_q(s⊕y) = q(s,y) on synthetic logit data
- Verify sequence energy correlates with correctness
- Verify hallucination token identification on known-bad sequences
- ~12 tests

---

## P5: Training Through Optimization (Hessian-Vector Products)

**Source:** [Energy-Based Transformers](https://arxiv.org/abs/2507.02092) (July 2025)
**Effort:** High (2-3 sessions)
**Impact:** High — learns energy landscapes optimized for repair, not just classification

### Key Idea

Standard EBM training (NCE, DSM) learns to discriminate correct from incorrect. But that doesn't guarantee the energy landscape is *easy to repair* — the gradient might point away from the solution, or there might be local minima that trap the repair process.

EBT training backpropagates through the gradient descent optimization itself:
1. Start from random prediction ŷ₀
2. Run N gradient descent steps: `ŷ_{i+1} = ŷ_i - α∇E(x, ŷ_i)`
3. Compute loss on the final prediction ŷ_N (cross-entropy or MSE)
4. Backpropagate through ALL N steps → requires second-order derivatives (Hessian-vector products)

This trains the energy landscape to have the property: "gradient descent from random starts converges to the correct answer." This is exactly what we want for verify-and-repair.

### What to Build

**New file: `python/carnot/training/optimization_training.py` (~250 lines)**

```python
def optimization_training_loss(
    energy_fn: EnergyFunction,
    data_batch: jax.Array,        # correct answers
    n_optimization_steps: int = 10,
    step_size: float = 0.1,
    key: jax.Array | None = None,
) -> jax.Array:
    """Loss that backpropagates through gradient descent on the energy.
    
    Algorithm (from EBT paper, Algorithm 1):
    1. Sample random initial predictions: ŷ₀ ~ N(0, I)
    2. Run N steps of gradient descent: ŷ_{i+1} = ŷ_i - α∇_ŷ E(x, ŷ_i)
    3. Compute loss = MSE(ŷ_N, y_true)  [or cross-entropy for discrete]
    4. Backpropagate through the entire trajectory
    
    This requires Hessian-vector products: ∂loss/∂θ flows through each
    gradient step, requiring ∂(∇_ŷ E)/∂θ = Hessian of E w.r.t. (ŷ, θ).
    JAX handles this automatically via jax.grad of jax.grad.
    """

def _unrolled_optimization(
    energy_fn, x_init, n_steps, step_size
) -> tuple[jax.Array, list[jax.Array]]:
    """Run gradient descent and return (final_state, trajectory).
    
    Uses jax.lax.scan for the loop (JIT-compilable).
    The entire computation graph is retained for backprop.
    """
```

**Key JAX technique:** `jax.grad(jax.grad(f))` gives Hessian-vector products automatically. The unrolled optimization is just a `jax.lax.scan` where each step applies `x - α * jax.grad(E)(x)`. JAX traces through the entire scan, making backprop through optimization "free" (no manual Hessian code needed).

**Modify: `python/carnot/inference/learned_verifier.py`**
- Add `training_method="optimization"` option to `LearnedVerifierConfig`
- Train with optimization loss instead of NCE: learns landscapes that repair well

### Why This Matters

NCE trains: "correct has lower energy than incorrect" ✓
Optimization training trains: "gradient descent from random starts finds the correct answer" ✓✓

The second property is strictly stronger. A model trained this way produces energy landscapes where `repair()` is guaranteed to work (by construction), not just hoped to work.

### Depends On

- None (can be implemented independently)
- Benefits from P3 (SNL) as a comparison baseline

### Test Strategy

- Verify loss decreases during training
- Verify trained model's energy landscape has correct minima
- Compare repair success: NCE-trained vs optimization-trained on same SAT instances
- Verify Hessian-vector products are computed correctly (JAX autograd)
- ~12 tests

---

## Implementation Schedule

| Phase | Proposals | Sessions | Dependencies |
|-------|-----------|----------|-------------|
| **Phase 1** | P1 + P2 | 1 session (parallel) | None |
| **Phase 2** | P3 | 1 session | None |
| **Phase 3** | P4 | 1-2 sessions | P1 |
| **Phase 4** | P5 | 2-3 sessions | None (but benefits from P3) |

Total: ~5-7 sessions to implement all five proposals.

## Success Criteria

After all five proposals are implemented:

1. **P1**: Hallucination detection AUROC > 70% on SAT (LLM gives wrong answer → semantic energy is high)
2. **P2**: Multi-start repair success rate > single-start (measured on SAT benchmark)
3. **P3**: SNL-trained verifier matches or beats NCE-trained verifier accuracy
4. **P4**: Per-token energy from LLM logits correlates with per-token correctness
5. **P5**: Optimization-trained verifier achieves higher repair success than NCE-trained on same instances

## Relationship to End Goal

```
P1 (semantic energy)     → first-line LLM confidence check
P2 (multi-start)         → more robust repair
P3 (SNL training)        → better learned verifiers
P4 (ARM↔EBM bijection)  → LLM IS the EBM (theoretical unification)
P5 (train through opt)   → energy landscapes designed for repair

Together: LLM generates → its own energy detects uncertainty (P1/P4)
          → multi-start repair fixes it (P2)
          → using an energy landscape trained to be repairable (P3/P5)
          → autoresearch improves everything autonomously
```
