# V690 capstone

Verdict: complete_blocked_missing_science.

- exp7953-contract-methods: disqualified ([artifact](../../results/experiment_7953_v690_contract_methods.json)).
- exp7954-training-coverage: disqualified ([artifact](../../results/experiment_7954_v690_training_coverage.json)).
- exp7955-response-targets: null ([artifact](../../results/experiment_7955_v690_response_targets.json)).
- exp7956-energy-fit: skipped ([artifact](../../results/experiment_7956_v690_energy_fit.json)).
- exp7957-decision-abstention: absent ([artifact](../../results/experiment_7957_v690_decision_abstention.json)).
- exp7958-qwen-response-risk: null ([artifact](../../results/experiment_7958_v690_qwen_response_risk.json)).
- exp7959-evidence-fragility: absent ([artifact](../../results/experiment_7959_v690_evidence_fragility.json)).
- exp7960-causal-acquisition: absent ([artifact](../../results/experiment_7960_v690_causal_acquisition.json)).
- exp7961-delayed-calibration: absent ([artifact](../../results/experiment_7961_v690_delayed_calibration.json)).
- exp7962-arc-supervisor-delta: null ([artifact](../../results/experiment_7962_v690_arc_supervisor_delta.json)).
- exp7963-service-cost: absent ([artifact](../../results/experiment_7963_v690_service_cost.json)).
- exp7964-hardware-evidence: null ([artifact](../../results/experiment_7964_v690_hardware_evidence.json)).
- exp7965-capstone: self_administrative ([artifact](../../results/experiment_7965_v690_capstone.json)).

- FR-05/FR-08/FR-09/FR-10/NFR-01: blocked. Continue: Changed qualified inputs meet registered cost, retention and whole-service bounds. Retire: Retire unchanged scope when the same verdict recurs without changed inputs or qualification.
- FR-11: blocked. Continue: Changed qualified inputs meet registered cost, retention and whole-service bounds. Retire: Retire unchanged scope when the same verdict recurs without changed inputs or qualification.
- FR-12/FR-06: blocked. Continue: Changed qualified inputs meet registered cost, retention and whole-service bounds. Retire: Retire unchanged scope when the same verdict recurs without changed inputs or qualification.

Audit readiness is separate from scientific benefit and FoVer publication readiness. GAP-ORACLE-DISTINCT remains open after the September 28 correction. DiffusionGemma remains pending. Historical required failures remain preserved. Historical E2E-016 uses 20260929 on both routes; current execution uses 20261001. No model was loaded. Ops and traceability reconciliation belongs to the conductor.

Current primary: [Exp7965](../../results/experiment_7965_v690_capstone.json).
The [registered V690 methods](../../openspec/change-proposals/research-roadmap-vNEXT.md)
bind each comparison. The primary names the immutable authority snapshots,
validation command manifest, primitive rows, terminal reports and reader receipt.

Continue FR-12/FR-06 only with changed prerequisites and current qualified
Exp7956, Exp7957 and Exp7959 inputs. Exp7954 must first pass its exact
training_coverage_ready_score=1 and runtime_ready_score=1 gates, with a
qualified verdict and flagged_adversarial=false. Decision benefit needs cost
gain >=.02, positive paired cost and Brier lower intervals against every
registered control, Holm-adjusted p<.05 and automated coverage >=.20.
Equal-coverage contrasts fail when realized coverage differs by more than .05.
Qwen needs at least 32 complete cluster pairs and eight examples per class.
Its cost and Brier gains must each reach .02 with positive lower intervals,
Holm-adjusted p<.05, coverage >=.20 and no added false accepts. The current
Qwen null remains valid; response annotations are exposed development data.

Continue FR-11 only after current eligible fitted heads and a sealed Exp7960
trajectory exist. Future cost gain must reach .02 against both complete-static
and no-write, with positive lower intervals and Holm-adjusted p<.05.
Require nonworse Brier, retention cost-increase upper interval <=.01, no added
false accepts and an earlier committed write that changes a later decision
before feedback. Exp7961 cannot change the learned trajectory or use future
labels. A complete valid null can establish measurement readiness.

Continue FR-05/FR-08/FR-09/FR-10/NFR-01 only with qualified current producers,
complete service spans and matched coverage. A measured speedup needs no
added false accepts and a positive paired saved-time lower interval.
Modeled kernel gains and historical board custody are separate evidence.
Unknown board capabilities remain unknown; no power or board-speed claim follows.

Retire each exact repeated verdict identified in retirement_decisions.
Require a changed mechanism or authenticated prerequisite for failed research.
An ARC no-event scan may advance to a new authenticated receipt frontier.
It is not permission to repeat historical game solves or an unchanged study.
The September 28 oracle-distinct correction remains in force. DiffusionGemma
stays pending. Paper readiness from G1-G4 does not close these three PRD gaps.
