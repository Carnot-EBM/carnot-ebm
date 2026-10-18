# Zenil-Grounded Self-Distillation: Deployable Stack

**Status:** **Major revision 2026-04-29** after the 9-round
Carnot/Gemini/Deep Think derivation chain converged on a complete
deployable Phase 3 architecture. Supersedes all prior drafts of this
file.

**Target milestone:** .82 or .83 (scope expanded from 4 to 8
experiments — .81 unrealistic).

**Priority:** **High.** This proposal scopes the implementation of
Carnot's Phase 3 self-distillation architecture as derived in
`docs/research-notes/zenil-deep-think-round{6,7,8,9}-results.md`.
The mathematical foundation is complete; this is the code-side
follow-up.

**Origin:** 2026-04-28 deep collaboration between Carnot, Gemini, and
Deep Think on the verifier-filtered self-distillation framework. The
chain produced six publishable contributions and a complete deployable
4-step architecture. See `memory/project_phase3_architecture_complete.md`
for the converged recipe.

**Depends on:** nothing. All math derivations have landed (commits
`f25c697e`, `2641fc44`, `8b2fd55a`, `cc50c904`, `f743ff5a`, `b89b12bc`,
`ea58be84`).

## The 4-step Phase 3 architecture (Deep Think Round-9 convergence)

The deployable architecture for verifier-filtered self-distillation
security:

1. **Spanning suite ($m \geq 6$):** 2× Z3 + 2× fuzzers (disjoint
   distributions) + 2× LLM/Mutant. Pairwise Friedrichs angle
   $\theta_F \geq \pi/4$.
2. **Recursive hardening ($k=15$):** AND-compose pairs →
   $\binom{6}{2}=15$ effective rotators. Bypass rate suppression
   $\varepsilon \to \varepsilon^2 \cos\theta_F$.
3. **Pre-flight audit:** $M^* \geq 192{,}000$ red-team adversarial
   samples via joint-null-maximizer RL policy. Assert
   $\lambda_{\min}(\widehat\Sigma_{15\times 15}) > 0.1$.
4. **In-loop tripwire:** EMA of $\lambda_{\min}(\widehat\Sigma_t)$ from
   native false-positives (zero extra FLOPs). On threshold breach:
   halt + auto-generate new Active Liveness Mutant + resume.

This proposal scopes 8 experiments (Exps A–H) implementing this
architecture plus the supporting Φ measurement, annealing schedule,
and Gray-code transpiler theorem from earlier rounds.

## What this proposal IS

A complete, deployable Phase 3 self-distillation stack:

- **Math foundation modules:** `phi_test.py`, `anneal.py`,
  `gray_code_factor.py` (Exps A, B, D — derived in Rounds 6-7).
- **Architecture modules:** 6 base verifiers + AND-composition
  machinery + red-team agent + audit/tripwire infra
  (Exps E1-E4 — derived in Rounds 8-9).
- **Spec contracts:** REQ-PHASE2-006 (Gray code) + REQ-PHASE3-001-004
  (the architectural axioms).

## What this proposal IS NOT

- **Not a refactor of the existing transpiler.** The Phase 2
  modules (`gray_code.py`, `distill.py`, `diagnostics.py`) shipped
  2026-04-27 and are correct.
- **Not a Phase 3 deployment.** This proposal builds the *infrastructure*;
  the actual Phase 3 self-distillation training runs are downstream
  experiments using this infrastructure.
- **Not a position paper.** The math derivations are publishable
  (six contributions across Rounds 6-9) but the writeup is a separate
  artifact. This proposal scopes only code/spec work.

## Proposed experiments

### Exp A — Φ > 0 measurement module

**Deliverable:** `python/carnot/eval/phi_test.py` +
`tests/python/test_phi_test.py` +
`results/experiment_<N>_phi_test_module.json`.

**What it does:**

1. Implement `phi_test(Q_samples, log_mu_P_fn, E_fn, delta_t) -> PhiTestResult`
   returning $\widehat\Phi$ via the corrected covariance formula:
   $$\widehat\Phi = \frac{1}{\delta_t} \widehat{\mathrm{Cov}}_{Q_t}\!\left(\frac{Q_t}{\mu_P}, E\right)$$
2. Bootstrap CI (1000 resamples) and hypothesis test ($\alpha=0.05$).
3. Orthogonality stall detection: when $\widehat\Phi < \varepsilon$,
   return `stalled=True`.
4. 15+ unit tests covering aligned / orthogonal / anti-aligned cases.

**Acceptance:** module imports, tests pass, artifact reports
`phi_test_module_complete`.

### Exp B — Constant-T schedule module

**Deliverable:** `python/carnot/training/anneal.py` +
`tests/python/test_anneal_schedule.py` +
`results/experiment_<N>_annealing_module.json`.

**What it does:**

1. `constant_t_schedule(t, T_floor, sigma2, tau_int, Phi, delta_t)`
   returning $(T_t, N_t)$ with:
   $$T_t = T_{\text{floor}}, \quad N_t = \left\lceil \frac{\sigma^2 T_{\text{floor}} \tau_{\text{int}}}{2 \Phi \delta_t}\right\rceil$$
2. `T_floor` calibration helper (estimates $T_{\text{crit}}$ from
   energy landscape).
3. Hardware-tier mode (CPU / KV260 / Extropic / photonic with
   samples-per-second caps).
4. 12+ unit tests.

**Acceptance:** module callable from FR-11 experiments, tests pass,
hardware-tier table matches `_bmad/architecture.md`.

### Exp C — DROPPED

Original PT-acceptance hyperparameter scope refuted by Deep Think
Round-7 (the 0.35 number was a Gemini hallucination conflating MALA
with PT swaps). Keep PT swap acceptance at standard 0.234 with
explicit citation. No code change required beyond a
`PT_TARGET_ACCEPTANCE = 0.234` constant in `distill.py` for
documentation.

### Exp D — REQ-PHASE2-006 Gray-code factor measurement

**Deliverable:** `python/carnot/hardware/transpiler/measurements.py` +
`scripts/experiment_<N>_gray_code_factor.py` +
`tests/python/test_gray_code_factor.py` + new spec entry
REQ-PHASE2-006.

**What it does:**

1. Run identical Hamiltonian under standard binary and Gray-code
   encoding at 8-bit width.
2. Measure $\widehat\tau_{\text{int}}$ for each via diagnostics module.
3. Compare to corrected theoretical
   $\Gamma = \exp(\kappa(2^{k-1} - 1)/T)$.
4. Pass condition: empirical $\widehat\Gamma \geq \Gamma_{\text{theory}} / 10$.

**REQ-PHASE2-006:** "The continuous-to-Ising transpiler's Gray-code
visible-spin encoder reduces integrated autocorrelation time by factor
$\Gamma \approx \exp(\kappa(2^{k-1} - 1)/T)$ vs standard binary encoding
at bit width $k$, where $\kappa$ is the underlying Hamiltonian's
coupling strength."

**Acceptance:** experiment artifact reports `gray_code_factor_confirmed`
or publishable `gray_code_factor_below_theoretical` with diagnostic.

### Exp E1 — Six-base verifier suite implementation

**Deliverable:** `python/carnot/verifier_suite/__init__.py` +
`python/carnot/verifier_suite/base.py` +
`python/carnot/verifier_suite/{z3_smt.py, fuzz_a.py, fuzz_b.py, llm_judge.py, mutant_a.py, mutant_b.py}` +
`tests/python/test_verifier_suite_base.py` +
`results/experiment_<N>_six_base_verifiers.json`.

**What it does:**

1. Define `VerifierBase` protocol enforcing the three Q1 axioms via
   type signatures and runtime checks:
   - **Axiom 1 (Semantic Grounding):** `requires_input: bool` flag;
     at least one verifier in any suite must have `requires_input=True`.
   - **Axiom 2 (Topological Transversality):** `kernel_class: str`
     enum naming the verifier's blind-spot category.
     Suite-validator refuses any pair with same `kernel_class`.
   - **Axiom 3 (Information-Theoretic Diversity):** `expected_mutual_info_bound: float`
     measured at audit time.

2. Implement six concrete verifiers per the Round-9 recipe:
   - **`z3_smt.py`:** auto-compile types + post-conditions to bounded
     Z3 (kernel: structural-soundness).
   - **`fuzz_a.py`, `fuzz_b.py`:** Hypothesis-style property fuzzing
     on disjoint input distributions (kernel: empirical-execution).
   - **`llm_judge.py`:** intent-grounded judge using a separate model
     (kernel: semantic-alignment).
   - **`mutant_a.py`, `mutant_b.py`:** Active Liveness Mutants —
     auto-generate semantically-altered anti-specs from $\sigma$;
     code must FAIL anti-specs while passing original (kernel:
     anti-vacuity).

3. Suite-validator runs the **Pathological Joint Null Theorem
   checklist**: for each candidate transformation $T$ in a library
   of known invariant transforms (dead-code padding, dummy variable
   wraps, tautological gates, etc.), check whether all 6 verifiers
   are invariant under $T$. If any $T$ passes all 6 → suite is broken.

**Acceptance:** all 6 verifiers callable, suite-validator rejects
known-broken designs (e.g., {Z3, AST density, liveness} from .80
exp1027), accepts the Round-9 recipe.

### Exp E2 — AND-composition machinery

**Deliverable:** `python/carnot/verifier_suite/composition.py` +
`tests/python/test_composition_algebra.py` +
`results/experiment_<N>_and_composition.json`.

**What it does:**

1. Implement `AndComposed(verifier_a, verifier_b) -> VerifierBase`
   that returns a verifier whose acceptance is the *product* of
   sub-verifier acceptances (intersection of valid manifolds).
2. Implement `compose_pairs(base_suite: list[VerifierBase]) -> list[VerifierBase]`
   generating all $\binom{m}{2}$ pairs from $m$ base verifiers.
   For $m=6$, returns 15 composed verifiers.
3. Implement bypass-rate measurement on synthetic adversarial outputs:
   verify that single $E_i$ has bypass $\varepsilon$, AND-composition
   $E_i \wedge E_j$ has bypass $\approx \varepsilon^2 \cos\theta_F$.
4. Reject `OrComposed` and `SumComposed` explicitly with type-error
   messages citing Round-9 result (sum/OR composition grows kernels
   and destroys security).
5. 15+ unit tests covering von Neumann projection algebra, bypass
   rate scaling, rejection of unsafe composition modes.

**Acceptance:** AND-composition empirically confirms
$\varepsilon^2 \cos\theta_F$ bypass scaling within 20%; OR/sum
composition explicitly rejected.

### Exp E3 — Joint-null-maximizer red-team RL agent

**Deliverable:** `python/carnot/verifier_suite/red_team.py` +
`scripts/experiment_<N>_red_team_audit.py` +
`tests/python/test_red_team.py` +
`results/experiment_<N>_red_team_agent.json`.

**What it does:**

1. Implement RL policy $\pi_{\text{red}}$ that generates outputs
   $x$ maximizing $\sum_{i=1}^{k} E_i(x) \cdot \mathbf{1}[x \notin \mathcal{T}]$.
   Reward: $+1$ for each verifier fooled (false-positive), $-\infty$
   for landing in truth manifold $\mathcal{T}$ (anti-reward).
2. Use a small base model (Qwen3.5-0.8B or Gemma3-E2B from Carnot's
   SOTA list) fine-tuned via REINFORCE/PPO on the cumulative-bypass
   reward. Train for $T_{\text{red}} \in [10^4, 10^5]$ steps until
   bypass rate plateaus.
3. Generate $M^* = 192{,}000$ adversarial samples for the pre-flight
   audit batch (Exp E4).
4. 10+ unit tests verifying reward shape, policy convergence on
   synthetic targets, and that the red-team agent strictly improves
   bypass rate over a random-baseline.

**Acceptance:** trained $\pi_{\text{red}}$ achieves $\geq 5\times$
bypass rate over random baseline on the 6-base-verifier suite;
sample throughput $\geq 100$ samples/min on local ROCm GPU.

### Exp E4 — Pre-flight audit + in-loop tripwire infrastructure

**Deliverable:** `python/carnot/verifier_suite/audit.py` +
`python/carnot/verifier_suite/tripwire.py` +
`tests/python/test_audit_tripwire.py` +
`results/experiment_<N>_audit_tripwire.json`.

**What it does:**

1. **`audit.py`:** runs the pre-flight audit per Exp E3's red-team
   batch:
   - Compute $\widehat\Sigma_{k\times k}$ binary cross-covariance.
   - Estimate $\widehat\Theta_k$ via Matrix Bernstein:
     $M^* \geq (2k^2\varepsilon + 2k\eta/3)/\eta^2 \cdot \log(2k/\beta)$.
   - Assert $\lambda_{\min}(\widehat\Sigma) > 0.1$.
   - Return `AuditResult(passed: bool, lambda_min: float, suite_safe_for_phase3: bool)`.

2. **`tripwire.py`:** in-loop tripwire for self-distillation training:
   - Maintain EMA of $\widehat\Sigma_t$ from native false-positives
     (zero extra FLOPs, just outer-product update on FP samples).
   - On each round, compute $\lambda_{\min}(\widehat\Sigma_t)$.
   - If $\lambda_{\min} < \tau$ (configurable safety threshold,
     default $0.1$): emit `TripwireBreach` event.
   - Tripwire breach handler: halt training, invoke
     `MutantGenerator(spec)` from Exp E1's mutant verifier to
     auto-generate new Active Liveness Mutant; add to suite;
     re-audit with red-team batch; resume training if audit passes.

3. **Spec contract REQ-PHASE3-002:** "Phase 3 self-distillation training
   MUST instrument an in-loop tripwire on $\lambda_{\min}(\widehat\Sigma_t)$;
   training MUST halt on tripwire breach pending mutant suite
   regeneration."

4. 12+ unit tests covering audit pass/fail decisions, tripwire
   detection on synthetic stall trajectories, mutant regeneration on
   breach.

**Acceptance:** audit detects known-broken suite (Z3+AST+liveness)
as $\lambda_{\min} < 0.05$; accepts Round-9 recipe; tripwire fires
on synthetic adversarial trajectory within 100 rounds of training.

### Exp F — Empirical Friedrichs angle measurement on Carnot suites

**Deliverable:** `scripts/experiment_<N>_friedrichs_angle_audit.py` +
`results/experiment_<N>_friedrichs_angle_<suite>.json` for several
candidate suites.

**What it does:**

1. Generate 192,000 red-team samples via Exp E3's $\pi_{\text{red}}$.
2. For each candidate Phase 3 suite design (Round-9 recipe + 2
   variants), compute $\lambda_{\min}(\widehat\Sigma)$.
3. Report which suite designs pass the audit ($\lambda_{\min} > 0.1$)
   and which fail.
4. Specifically include a regression test: the .80 exp1027-derived
   suite (Z3 + AST density + liveness) MUST fail the audit per the
   Round-8 prediction.

**Acceptance:** Round-9 recipe passes; Z3+AST+liveness fails;
audit takes $\leq 30$ min wall-clock on local ROCm.

## New spec requirements

- **REQ-PHASE2-006:** Gray-code visible-spin encoder reduces
  $\tau_{\text{int}}$ by factor $\Gamma \approx \exp(\kappa(2^{k-1}-1)/T)$
  vs binary at bit width $k$.

- **REQ-PHASE3-001:** Phase 3 self-distillation MUST include $k \geq 15$
  semantically orthogonal verifiers via the Round-9 6-base + AND-
  composition recipe. Single-verifier or ensemble-combine architectures
  are explicitly rejected.

- **REQ-PHASE3-002:** Phase 3 self-distillation training MUST instrument
  an in-loop tripwire on $\lambda_{\min}(\widehat\Sigma_t)$; training
  MUST halt on tripwire breach pending mutant suite regeneration.

- **REQ-PHASE3-003:** AND-composition is the unique permitted verifier
  composition mode. OR / sum composition is explicitly rejected
  (kernel growth + glassy MCMC).

- **REQ-PHASE3-004 (Pathological Joint Null Audit):** any candidate
  Phase 3 verifier suite MUST pass the Pathological Joint Null Theorem
  checklist before deployment: no continuous non-injective $T$ with
  $E_i \circ T = E_i$ ∀$i$ AND $D_{KL}(\mu_P \| T_\# \mu_P) \gg 0$.
  Operational test: enumerate known invariant transforms; reject the
  suite if any apply to all verifiers.

## Decentralization implications

**Rule 1 (local-first):** unaffected. All artifacts run on CPU/ROCm.
LLM-judge verifier is the only one that *might* call closed-weight
models — but the architecture explicitly tolerates open-weight judges
(Qwen, Gemma) at the cost of weaker $\theta_F$ on that verifier.

**Rule 5 (hardware portability as political requirement):** strengthened.
The Q2 throughput target of $k \cdot N_t = 15 \cdot 10^6$ samples/round
mandates Phase 2 hardware acceleration (KV260, Extropic XTR-0,
photonic). Sovereign access to high-throughput verifier suite execution
is part of Phase 3 sovereignty.

**Rule 7 (no vendor abstractions):** strengthened. Z3 is open-source.
The fuzzers (Hypothesis-derived) are open-source. The mutant generators
are pure Python. The LLM-judge can use any model. **The architecture
is sovereign-deployable end-to-end.**

**New rule implication:** sovereign access to *suite design* — not
just verifier components — is now part of Phase 3 sovereignty. The
Pathological Joint Null Theorem audit is the operational test that
ensures suite design hasn't been compromised by an adversarial
verifier author.

## Risks

1. **Φ test estimation variance.** Small held-out sets give wide CI;
   may fail to reject $H_0$ at small $\alpha$. Mitigation: report CI;
   experiments choose own significance.

2. **Annealing schedule constants.** $T_0, N_0$ are problem-specific;
   `calibrate()` helper uses early-iteration $\delta$ measurements.

3. **REQ-PHASE2-006 may falsify.** Gray-code factor 127 at $k=8$ is
   bold; sub-leading corrections may dominate. Mitigation: acceptance
   threshold is generous (theoretical/10).

4. **Red-team agent may be too weak.** A small base model trained on
   $T_{\text{red}}$ steps might not find the worst-case attacks.
   Mitigation: comparison against random baseline; if $5\times$ improvement
   not achieved, fall back to manual adversarial example library.

5. **AND-composition combinatorial explosion.** $k=15$ at $m=6$;
   $k=21$ at $m=7$; $k=28$ at $m=8$. Throughput grows quadratically.
   Mitigation: rotation policy uses only top-$k'$ composites by recent
   $\widehat\Phi_i$.

6. **Mutant generator may produce unsatisfiable specs.** If the
   anti-spec generator inverts a property that's structurally
   necessary, the mutant becomes unsatisfiable. Mitigation: require
   the original spec satisfiable witness; reject mutants that don't
   admit some witness.

7. **The architecture is for *iterated* self-distillation.** Carnot's
   current Phase 1 product (single-pass verify-repair) doesn't iterate,
   so the orthogonality stall isn't a current threat. This proposal's
   Phase 3 infrastructure ships ahead of the actual Phase 3 training
   runs. Risk: infrastructure rots before being used.

## Acceptance criteria

1. ✅ All math derivations landed (Rounds 6-9 results files).
2. `phi_test.py` exists with corrected covariance formula and bootstrap CI.
3. `anneal.py` exists with constant-T schedule and hardware-tier
   mode.
4. `gray_code_factor.py` confirms Gray-code factor within /10 of
   theoretical at $k=8$.
5. Six base verifiers implemented per Round-9 recipe; suite-validator
   rejects known-broken designs.
6. AND-composition machinery confirms $\varepsilon^2 \cos\theta_F$
   bypass scaling; OR/sum composition explicitly rejected.
7. Red-team agent achieves $\geq 5\times$ bypass over random baseline;
   generates 192,000-sample audit batch.
8. Audit detects known-broken Z3+AST+liveness suite as $\lambda_{\min} < 0.05$;
   accepts Round-9 recipe at $\lambda_{\min} > 0.1$.
9. Tripwire fires on synthetic adversarial trajectory within 100 rounds.
10. New spec entries REQ-PHASE2-006, REQ-PHASE3-001-004 added to
    `openspec/capabilities/*/spec.md`.
11. Cross-references to `docs/research-notes/zenil-*.md` in all
    Phase 3 modules.

## Why this is in change-proposals, not split into milestones

The 8 experiments cluster around a single mathematical result (the
9-round Zenil derivation) with deeply coupled dependencies (E2 needs
E1, E3 needs E1, E4 needs E2+E3, F needs E1+E2+E3). Splitting across
milestones risks one half shipping without the other and stalling
deployment.

Bundling as one proposal preserves coherence; the 8 experiments are
sized to fit in $\sim 2$ milestones (.82 + .83) given the $\sim 30$ min
wall-clock per experiment with current infrastructure.

## Estimated implementation effort

- **Exp A, B, D:** $\sim 2$ hours each (math is fully specified).
- **Exp E1:** $\sim 8$ hours (6 verifiers + axiom enforcement + suite
  validator).
- **Exp E2:** $\sim 4$ hours (composition algebra + tests).
- **Exp E3:** $\sim 12$ hours (RL agent training is non-trivial).
- **Exp E4:** $\sim 6$ hours (audit + tripwire + mutant regeneration).
- **Exp F:** $\sim 4$ hours (audit script wrapping E1-E4).

**Total:** $\sim 40$ hours of implementation work, fitting in 2
milestones at typical Carnot conductor cadence.

## Publication targets

The 9-round chain produces six publishable results worth ~1 position
paper:

1. Closed-form orthogonality plateau (Round-7).
2. Ensemble vs rotation: rotation strictly dominates (Round-7).
3. Null-space mimicry attack on Boolean verifiers (Round-7).
4. Pathological joint null space + iff theorem (Round-8 + 9).
5. AND-composition kernel-shrinkage theorem (Round-9).
6. Complete 4-step deployable Phase 3 architecture (Round-9).

Working title: **"Sovereign Foundation Models Require Multi-Verifier
Pre-emptive Rotation: A Mathematical Foundation for Phase 3
Self-Distillation."**

The position paper is downstream of this proposal — write after the
empirical validation lands in Exp F.
