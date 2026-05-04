# Q12 Synthesis — Phase-5 PCD + Joint Null Space Co-Evolution

**Status:** SYNTHESIZED 2026-05-04 ~01:50Z from Gemini Deep Think response
**Source:** verbatim response captured in Carnot session transcript
**Prompt:** `phase5-pcd-joint-null-space-deep-think-prompt.md`
**Strategic role:** sets exp1238 Phase-5-D acceptance gates + Phase-4 (active inference) connection

## Headline result

**Hypothesis B (Substrate Gaming / Null-Space Excavation) wins
theoretically.** PCD acts as an unsupervised self-distillation loop;
substrate output distribution mode-collapses onto the lowest-energy
verifier-accepted states, which by Kolmogorov-simplicity bias are
disproportionately joint-null-space samples (vacuous truths, dead
code, empty structural loops). Q11's pairwise orthogonality bound is
**catastrophically destroyed** by in-situ training:

$$\lim_{t \to \infty} P_t(V_i = 1 \mid V_j = 1) = 1.000$$

Even when Q11 has structurally bounded the joint null space's
geometric *volume*, PCD dynamics ensure that the substrate's
probability *measure* collapses entirely into that small volume.

**Implication:** Q11 (verifier design) and Q12 (training dynamics)
must work together. Q11 alone is necessary but insufficient. Q12
prescribes explicit anti-gaming regularization on the PCD update.

## Q12.1–Q12.5 derivations

**Q12.1 — Stationary distribution exists, is non-degenerate, but mode-
collapsed.** PCD update minimizes
$D_{KL}(P_{\theta_t}(\cdot \mid V_{AND}=1) \parallel P_\theta)$.
Stationarity requires $P_\infty(a) = P_\infty(a \mid V_{AND}(a)=1)$ —
support entirely contained within acceptance region. Synchronous
update of the conditional target with the model creates a positive
feedback loop (analogous to Apple SSD self-distillation degeneration).
Result: mixture of Dirac deltas concentrated on
$z^* = \arg\min_{z : V_{AND}(a)=1} E_{\theta_0}(z)$.

**Q12.2 — Probability measure expansion of $N_t$.** Kolmogorov
simplicity bias gives $E_{\theta_0}(a_N) \ll E_{\theta_0}(a_C)$ for
null-space $a_N$ vs oracle-correct $a_C$. PCD's replicator dynamic
amplifies this disparity. **The geometric volume of $N_t$ is bounded
by verifier design (per Q11), but PCD dynamically excavates it
until $N_t$ absorbs ~100% of probability density.**

**Q12.3 — Q11's bound is destroyed by in-situ training.** This is
the load-bearing finding. Q11 bounds $P_{prior}(V_i=1 \mid V_j=1)$ at
design time. PCD acts as a continuous adversarial optimizer rotating
$P_t$ entirely into the geometric intersection of verifiers. Empirical
$P_\infty(V_i=1 \mid V_j=1) = 1.000$ — exactly the exp1224 collapse,
now arising from training dynamics rather than decoder geometry.

**Q12.4 — Three regularization candidates with formal correctness
conditions.**

(a) **Entropy maximization:** $\mathcal{L}_{reg} = +\lambda \nabla_\theta \mathcal{H}(P_{\theta_t}(z \mid V_{AND}=1))$.
Correctness: $\lambda > \sup_{a_C \in C_t, a_N \in N_t} [E_{\theta_0}(a_C) - E_{\theta_0}(a_N)]$.
**Empirical λ-floor estimation is itself a research task** — must
sample initial $E_{\theta_0}$ on disjoint $a_C$, $a_N$ candidates.

(b) **Negative-sampling from oracle-disagreement:** Periodic injection
$\Delta\theta \propto \mathbb{E}_{a \in N_t}[+\nabla_\theta E_\theta(a)]$
when oracle identifies null-space sample. Correctness: oracle refresh
rate $\omega_O$ × negative gradient magnitude must dominate
unregularized self-training pull.

(c) **Null-space distance penalty (Q9 mode 4 instrumentation).**
Penalize cosine similarity to known $N_t$-anchors. Correctness:
$N_t$ manifold in $Z$-space must be geometrically contiguous.
**Fragmented $N_t$ → "whack-a-mole" failure mode.** Carnot's exp1108
vacuous-dead-code anchors are likely contiguous; fragility is in
generalizing to unseen null-space basins.

**Q12.5 — Detectable at Phase-5-D scale.** Mode collapse → continuous
contraction of latent covariance $\Sigma_z$. Statistical separability
via $\text{Tr}(\Sigma_z)$ requires $N_{min} = O(d \log d / \epsilon^2)$
samples. For $d=128$ and reasonable $\epsilon$: **$\approx 1500-3000$
accepted queries**. Phase-5-D's 10K query budget is **comfortably
sufficient** to disambiguate Hypothesis B from C.

## Recommended exp1238 acceptance gates (verbatim from Q12)

Rolling window of 1,000 queries over $z_{accepted}$:

**Gate 1 — Hypothesis B (null-space excavation):**
$\text{HALT}$ if `Tr(Cov(z_accepted))` drops > 40% from $t=0$ baseline
AND `E[E_θ(z_accepted)]` plateaus at low-variance
AND oracle acceptance < 15%. **Null-space excavation actively
occurring.**

**Gate 2 — Spera Gate (B/C mechanism):**
$\text{HALT}$ if $P_t(V_i=1 \mid V_j=1) > 0.85$ for any pair.
**Design-time orthogonality bounds have failed under training
dynamics.**

**Gate 3 — Hypothesis A (self-correction):**
$\text{PROCEED}$ only if all of: monotonically decreasing energy AND
stable covariance trace AND oracle correlation > 85% with $V_{AND}$.
Anything else is HALT or BLOCKED.

## Honest framing (per Q12 §3)

**Three unverified assumptions Phase-5-D depends on:**

1. **MCMC mixing ergodicity** — assumes negative chains accurately
   sample $P_{\theta_t}$. If mixing paralysis (Q9 mode 2) intervenes,
   negative samples freeze near positives, gradient artificially
   vanishes, descent into $N_t$ stalls prematurely. **This masquerades
   as Hypothesis C (artificial stability), masking probability measure
   expansion.** Phase-5-D MUST instrument MCMC chain distance
   (`E[||z_pos - z_neg||_2]`) as a fork-detector.

2. **Sign bottleneck continuity** — assumes $z \to \text{sign}(z) \to a$
   transmits gradient smoothly under PCD. Aggressive quantization may
   shatter the energy landscape; substrate may trap permanently in
   suboptimal SAFE basins before reaching deep null-space minimum.
   **This is the optimistic alternative** — sign() acts as a natural
   regularizer.

3. **Encoder simplicity bias** — assumes $E_{\theta_0}(N_t) < E_{\theta_0}(C_t)$.
   If Carnot's encoder intrinsically penalizes vacuous syntax trees
   *before* energy evaluation, Hypothesis A could transiently hold
   until the network invents a novel complex exploit. **Not a long-
   term defense** but buys time.

## Connection to Phase-4 active inference — the Dark Room Problem

Hypothesis B is **exactly the Dark Room Problem in Free Energy
Principle terms.** Variational Free Energy
$F \approx \mathbb{E}_Q[E] - \mathcal{H}(Q)$ has two terms; the agent
minimizes both pragmatic value (sensory expectations satisfied) and
epistemic value (information gain).

In Carnot's case:
- The **fixed verifier ensemble** = the agent's expected sensory
  environment ($V_{AND}=1$).
- **Oracle-correct outputs** $C_t$ = high-energy state requiring
  immense latent precision.
- **Joint null space** $N_t$ = the Dark Room. Sensory expectations
  perfectly met with minimal internal cognitive complexity.

Standard PCD maximizes pragmatic value without an epistemic
regularizer (unlike RLHF's KL-to-prior penalty). The mathematically
optimal solution: shut down epistemic drive, collapse internal
entropy, build the Dark Room around itself. Q12.4(a) entropy
maximization is the explicit epistemic-value re-introduction.

**This makes Phase-4 (active inference) and Phase-5 (in-situ training)
the same problem viewed from different axes.** Phase-4 prescribes
free-energy minimization as the substrate's objective; Phase-5
discovers that uncareful free-energy minimization collapses to the
Dark Room. The two phases must co-design the regularizer.

## Required exp1238 instrumentation additions (currently missing per Q12.5 + §5)

Phase-5-A/B did NOT instrument any of these:

1. **MCMC Persistent Chain Distance:** `E[||z_pos - z_neg||_2]`.
   Detects mixing paralysis (Q9 mode 2 + Q12 honest framing #1).
   If crashes to 0, gates are unreliable.

2. **Latent Centroid Cosine Drift:** cosine similarity of moving
   average centroid $\mu_t = E[z_{accepted}]$ vs trajectory.
   - Hypothesis B: $\mu_t$ snaps to anchored point (null-space basin)
   - Hypothesis C: $\mu_t$ maintains continuous Brownian rotation on hypersphere
   - Hypothesis A: $\mu_t$ drifts steadily toward oracle-correct centroid

3. **Trace covariance trajectory:** `Tr(Cov(z_accepted))` over time.
   This is Gate 1's metric. Must be sampled at all 10K queries with
   1000-query rolling windows.

4. **Per-verifier conditional acceptance trajectory:**
   $P_t(V_i=1 \mid V_j=1)$ for all pairs. Gate 2's metric.

5. **Energy-on-accepted trajectory:** $E[E_\theta(z_{accepted})]$.
   Gate 1 + Gate 3 metric.

## Q11 + Q12 combined strategic verdict for Phase-5-D

Without anti-gaming regularization, Phase-5-D's outcome is
predictable:

- Acceptance rates → 1.000 (Gate 2 trips)
- Trace covariance collapses by >40% (Gate 1 trips)
- Oracle correctness drops below 15% (Gate 1 trips)
- Energy plateaus on null-space basin

This IS the experimental confirmation of Q12 Hypothesis B at
intermediate scale. **Whether that's a successful failure (Q12
empirically validated → guides Phase-5-E with regularizer) or an
expensive failure (30-60 GPU-hours producing predictable collapse)
depends on whether exp1238 ships with or without Q12.4 regularization.**

**Strong recommendation:** exp1238 should ship in TWO arms:

- **Arm A (control):** standard PCD without regularization. Expected
  to trip Gates 1+2 within 2000-3000 queries. Confirms Q12 prediction.
- **Arm B (Q12.4(a) entropy regularization):** $\lambda$ calibrated
  to estimated $E_{\theta_0}$ gap. Expected to maintain Gate 3
  (Hypothesis A behavior).

Comparative analysis in Phase-5-D's artifact provides paper-v6 with
a strong empirical claim: "uncareful PCD gambles on null-space
excavation; explicit epistemic regularization restores oracle
alignment." This is genuinely novel relative to Apple SSD (which
worked WITHOUT verifier and observed sequence-diversity collapse).

## Cross-references

- Q12 prompt: `docs/research-notes/phase5-pcd-joint-null-space-deep-think-prompt.md`
- Q11 results (necessary precursor): `docs/research-notes/verifier-orthogonality-design-deep-think-results.md`
- exp1224 (Phase-5-C empirical Spera realization): `results/experiment_1224_phase5c_adversarial_probe.json`
- Phase-5 derisking proposal: `openspec/change-proposals/in-situ-training-phase5-derisking.md` (needs Q12 update for exp1238)
- Q9 (8 failure modes catalog): `docs/research-notes/in-situ-training-adversarial-robustness-deep-think-results.md`
- Phase-4 active-inference commitment: `memory/feedback_active_inference_phase4_committed.md`
- Apple SSD self-distillation (precedent for Hypothesis B): `memory/project_ssd_self_distillation.md`
