# Distributional-energy verifier on MuSR — first real result (2026-06-29, outer-loop)

Operator-directed ("tackle the off-ARC headroom", then "re-launch on a 3090 so it actually finishes").
First REAL execution of the post-6/30 distributional-energy-verifier pivot (arXiv:2605.18871) on a
non-saturated, no-cheap-oracle domain. `scripts/experiments/exp_distributional_energy_verifier_musr.py`;
artifact `results/distributional_energy_verifier_musr.json`. Generator = Qwen3.5-9B-MTP on GPU 1 (CUDA),
N=50 MuSR/murder_mysteries (binary MCQ), K=8 candidates, M=3 quality-ensemble. All selection methods
ORACLE-DISTINCT (none sees gold; verifier_is_oracle=False).

## Result: NULL for the cheap energy verifier, but the domain has real headroom (judge>SC)

| method | accuracy (/50) |
|---|---|
| self-consistency (baseline) | **0.58** (29) |
| distributional-energy (with-abstain) | 0.52 (26) |
| distributional-energy (pure min-energy) | 0.52 (26) |
| **LLM-judge** | **0.64** (32) |

- **Energy − SC = -0.06, CI95 [-0.18, +0.06]** (incl 0) -> the cheap **prompted process-reward
  decomposed-energy verifier does NOT beat self-consistency**. abstain_rate 0.26 (genuinely exercised).
- **But the LLM-judge beats SC (0.64 vs 0.58)** and SC is NOT saturated -> **MuSR HAS verifier headroom,
  reachable by a holistic verifier; the decomposed process-reward energy formulation underperformed it.**
- TAUTOLOGY flag (both energy variants = 26/50) inspected + documented as a small-n coincidence (abstain
  flipped only 2/13, netting even), NOT a bug. Informative null (methods differ on 9 q), oracle-distinct.

## Honest read + next step
The cheap prompted stand-in for the arXiv:2605.18871 learned-quality-scorer LoRA-ensemble is a NULL. The
judge>SC signal means the moat headroom is REAL on MuSR -- so the null is about the FORMULATION (cheap
process-reward step-validity + analytical penalty), not the domain. Do NOT conclude "energy can't help
off-ARC" from this. Candidate next experiments (ranked): (1) make the energy verifier a holistic learned
quality scorer (train the real LoRA-EBM ensemble on a MuSR-quality signal -- the judge already shows a
holistic scorer beats SC); (2) larger N (n=50 CIs are wide -- the judge's +0.06 is also only suggestive);
(3) a stronger generator for more diverse candidates. CAVEAT: preliminary; not headline until replicated.

## Cross-ref to ARC (the side-benefit path)
Per the 2026-06-29 analysis: a verifier formulation that DOES beat SC here (the judge, or a trained scorer)
is the "new energy signal" candidate to later test on ARC's SELECTION sub-problem (active-probe posterior),
where static state-energy scored 0.4958. This first result says: the holistic-judge direction, not the
cheap decomposed process-reward, is the one to carry forward.
