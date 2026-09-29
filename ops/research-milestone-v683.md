# V683 research milestone — 2026-09-29

Verdict: `complete_disqualified_required_v683_validation`. Science complete: 0. Measured benefit: 0.

## Current evidence

| Task | Status | Qualified |
| --- | --- | --- |
| exp7865-contract-methods | disqualified | False |
| exp7866-source-boundary | disqualified | False |
| exp7867-natural-runtime | eligible | True |
| exp7868-intervention-protocol | disqualified | False |
| exp7869-energy-fit | missing_producer | False |
| exp7870-decision-abstention | missing_producer | False |
| exp7871-qwen-sufficiency | missing_producer | False |
| exp7872-causal-acquisition | missing_producer | False |
| exp7873-feedback-scheduling | missing_producer | False |
| exp7874-arc-supervisor-delta | disqualified | False |
| exp7875-service-cost | missing_producer | False |
| exp7876-hardware-evidence | disqualified | False |
| exp7877-independent-audit | disqualified | False |
| exp7878-capstone | own_reconciliation | False |

Six required scientific producers are absent. A skip receipt is administrative. Exposed natural labels and deterministic fixtures do not establish hidden generalization.

## PRD gaps

- FR-12: blocked. No qualified current energy fit and calibrated natural-label decision evidence. Next: Can a fitted head improve Brier and paired decision cost on independent source families?
- FR-11: blocked. No qualified prediction-before-feedback causal update or persistent retention producer. Next: Does one admitted constraint persist and improve later held-out decisions after feedback?
- FR-05/FR-08/ARC: blocked. Service producer absent; ARC and board continuity artifacts fail required checks. Next: Can a paired live-route run and new board receipt establish usable cost and custody?

GAP-ORACLE-DISTINCT remains open. The September 28 retractions of Exp4245, Exp5160 and Exp5171 remain in force. DiffusionGemma is not promoted.

## Stable publication gates

- G1: True — FoVer dual-condition AUROC artifact present (experiment_2850_fover_dual_condition_integrity_v4.json); CI reported=True
- G2: True — reproducer confirmed: GitHub Actions workflow 'FoVer Headline Independent Reproducer' (.github/workflows/reproduce-fover-headline.yml), run 26725185125 on Carnot-EBM/carnot-ebm, 2026-05-31, conclusion=success. Clean ubuntu-latest runner, fresh CPU-only install from a clean checkout (non-operator environment), recomputed the FoVer headline AUROC via scripts/reproduce_fover_headline.py and asserted the published CIs: condition_A (production) AUROC in [0.9027, 0.9235] and learning_contribution in [0.0125, 0.0245] — both PASS. This is a CI-run independent reproducer per the Phase-1 ship gate ('could be a teammate, a CI run, or an external user'). Internally further de-risked by exp3430 (fresh git worktree + venv) and the local run on 2026-05-31 (AUROC 0.9131 within CI). NOTE: G2 attests the FoVer verifier-ensemble headline reproduces; it does NOT attest the broader energy-descent existential claim, which remains an honest-negative (Route-1/Route-2 bounded). Run: https://github.com/Carnot-EBM/carnot-ebm/actions/runs/26725185125
- G3: True — no retracted phrasings asserted as live claims
- G4: True — experiment_2850_fover_dual_condition_integrity_v4.json: random_seed/seeds=True, reproducibility_checksum=True

Scientific readiness is separate from release authorization. No publication or production activation occurred.

## Retirement and continuation


## Literature and hardware triggers

- source_sufficiency: continue only with independent source-family labels.
- causal_feedback: continue only after delayed-feedback control qualification.
- diffusiongemma: defer until independent scorer and local control pass.
- Require a paired live route, independent ARC firings, and authenticated new board execution before deployment claims.
