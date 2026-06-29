# Distributional-energy verifier on MuSR — first real result (2026-06-29, outer-loop)

Operator-directed ("tackle the off-ARC headroom", then "re-launch on a 3090 so it actually finishes").
First REAL execution of the post-6/30 distributional-energy-verifier pivot (arXiv:2605.18871) on a
non-saturated, no-cheap-oracle domain. `scripts/experiments/exp_distributional_energy_verifier_musr.py`;
artifact `results/distributional_energy_verifier_musr.json`. Generator = Qwen3.5-9B-MTP on GPU 1 (CUDA),
N=50 MuSR/murder_mysteries (binary MCQ), K=8 candidates, M=3 quality-ensemble. All selection methods
ORACLE-DISTINCT (none sees gold; verifier_is_oracle=False).

## FINAL RESULT (n=200): clean NEGATIVE — no method beats self-consistency

| method | accuracy (/200) | vs SC (McNemar p) |
|---|---|---|
| self-consistency (baseline) | **0.560** (112) | — |
| distributional-energy (pure min) | 0.535 (107) | −0.025 |
| distributional-energy (with-abstain) | 0.515 (103) | −0.045, CI [−0.105,+0.015], **p=0.188** |
| LLM-judge | 0.545 (109) | **p=0.736** |

**At n=200, NO method beats self-consistency.** The cheap prompted decomposed-energy verifier **trails SC**
(−0.045, p=0.188 — directionally worse, not significant); the LLM-judge is tied/below SC (p=0.736). The
n=50 "judge>SC headroom" was **confirmed noise**. This is a clean NEGATIVE for the cheap prompted energy
verifier off ARC.

### (Superseded n=50 snapshot, kept for the record)
n=50 read was SC 0.58 / energy 0.52 / judge 0.64 — the judge's apparent +0.06 evaporated at n=200 (it was
McNemar p≈0.51 noise, flagged by adversarial review at the time).

- **Energy − SC = -0.06, CI95 [-0.18, +0.06]** (incl 0) -> the cheap **prompted process-reward
  decomposed-energy verifier does NOT beat self-consistency**. abstain_rate 0.26 (genuinely exercised).
- **CORRECTION (adversarial review):** the LLM-judge is numerically above SC (0.64 vs 0.58) **but NOT
  significant** -- McNemar exact p≈0.51 (judge-only 6 vs sc-only 3), bootstrap CI [-0.06,+0.18] incl 0.
  At n=50, **NO method significantly beats SC** ("headroom is REAL" was an over-claim on a noise-level
  result; defensible statement = "judge ≥ SC but indistinguishable from noise at n=50").
- TAUTOLOGY flag (both energy variants = 26/50) inspected + documented as a small-n coincidence (abstain
  flipped only 2/13, netting even), NOT a bug. Informative null (energy overrides SC on 11 q, net -3),
  oracle-distinct (verifier never sees gold; traced clean).

## Honest read + next step
At n=50 this is INCONCLUSIVE: the cheap prompted decomposed-energy verifier shows no signal (clean null),
and the holistic judge is marginally higher but within sampling noise (McNemar p≈0.51). Do NOT conclude
either "energy can't help off-ARC" OR "headroom is real" -- both are unsupported at this n. Candidate next
experiments (ranked): (1) **replicate at n>=200** -- the decisive lever, since every effect here is
noise-level and the CIs are wide; (2) IF a larger run shows the judge (or a trained scorer) genuinely
beats SC, THEN train the real arXiv:2605.18871 LoRA-EBM holistic quality scorer (the cheap prompted
process-reward is only a stand-in); (3) a stronger generator for more diverse candidates. CAVEAT:
preliminary; nothing here is headline-eligible.

## Cross-ref to ARC (the side-benefit path)
Per the 2026-06-29 analysis: IF a verifier formulation is later shown to genuinely beat SC here (at n>=200
-- not yet established), that would be the "new energy signal" candidate to test on ARC's SELECTION
sub-problem (active-probe posterior), where static state-energy scored 0.4958. This first n=50 result is
inconclusive, so it does NOT yet license carrying any specific formulation forward to ARC -- the decisive
move is the n>=200 MuSR replication first.
