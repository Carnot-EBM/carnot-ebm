# Paper v6 Anchored Claims Matrix

Run date: `20260507`
Paper source path: `docs/arxiv-paper/main.tex`

## Anchored Claims

| Claim | Paper Section | Empirical Artifacts | Theoretical Support | Boundary |
|---|---|---|---|---|
| CLAIM-1: Verifier composition is bounded by measured correlation | Anchored Claims / Theoretical Bounds | results/experiment_1093_phase1c_verifier_joint_null_space_measurement.json, results/experiment_1224_phase5c_adversarial_probe.json, results/experiment_1256_verifier_orthogonality_audit_v3.json | sqrt(det(Sigma)) joint-volume correction, Welch/Rankin Simplex bound on verifier packing, Spera non-composability boundary for shared verifier null spaces | Does not claim k=6, k=15, or exponential AND-composition gains beyond the measured evidence. |
| CLAIM-2: Exact sampling requires sparse fast-path plus CPU fallback | Anchored Claims / Hardware Acceleration | results/experiment_1068_kv260_smoke_test_v9.json, results/experiment_1094_phase2a_sampler_correctness_audit.json, results/experiment_1451_discrete_sb_rtl_lint_sim_rerun.json, results/experiment_1460_hardware_portfolio_narrowing.json | single-site Gibbs detailed balance, chromatic-Glauber batching under graph-coloring constraints, scope-reduction hardware portfolio gate | Does not claim same-basis CPU-vs-FPGA speedup, KV260 board execution for new bitstreams, Extropic execution, NPU acceleration, or photonic execution. |
| CLAIM-3: Energy verifier calibration is distribution-bound | Anchored Claims / Empirical Realities | results/experiment_1072_sos_kan_v3_neural_gram.json, results/experiment_1100_cascade_validation_sota_outputs.json, results/experiment_1120_energy_verifier_retrain_sota.json, results/experiment_1265_diffutruth_vs_carnot_baseline.json | Goodhart/reward-hacking interpretation of verifier-shaped optimization, energy-based calibration under distribution shift, OOD boundary between FoVer and optimized SOTA-output corpora | Does not claim universal verifier dominance, cross-corpus DiffuTruth dominance, or future SOTA-family generalization. |
| CLAIM-4: Self-learning is a narrow verified-memory-growth claim | Anchored Claims / Self-Learning Scope | results/experiment_1374_continuous_self_learning_v3_verifier_selected_or_csp_fallback.json, results/experiment_1433_fr11_self_learning_v6_dvi_v3_gated.json, results/experiment_1447_fr11_v7_memory_policy_growth.json, results/experiment_1459_self_learning_nonheadline_lineage_decision.json | Zenil exogenous-grounding condition for self-distillation, primary semantic-verifier acceptance path, non-forgetting gate for persisted memory promotion | Does not claim replay-only learning, adapter-only learning, broad autonomous self-improvement, or completed DVI training. |

## Unsupported Territory Moved

| Topic | Destination | Reason | Supporting Input |
|---|---|---|---|
| same-basis CPU-vs-FPGA speedup | future_work | The KV260 latency point is measured, but same-N CPU-vs-FPGA timing has not been measured on the same per-sample basis. | results/experiment_1460_hardware_portfolio_narrowing.json |
| Extropic Z1/XTR-0, NPU, photonic, D-Wave, and large-FPGA execution | future_work | The 20260507 hardware narrowing keeps these tracks deferred until authenticated local execution or concrete reopen gates exist. | docs/research-notes/hardware_portfolio_narrowing.md |
| k=6, k=15, and arbitrary verifier-composition scaling | appendix | Measured k_eff and joint-null-space evidence support only the bounded k=5 framing plus theory-backed future cross-mechanism work. | results/experiment_1256_verifier_orthogonality_audit_v3.json |
| broad self-learning improves everything wording | future_work | Exp 1459 permits only the Exp 1447 verified-memory-growth pivot; replay-only and adapter-only claims stay non-headline. | results/experiment_1459_self_learning_nonheadline_lineage_decision.json |
| LARQL, Skillify, GStack, and ontology-governance comparator territory | future_work | Exp 1461 classifies these as watchlist or retired rather than active paper-v6 claims. | results/experiment_1461_comparator_integration_cite_retire_audit.json |
| full ARC-AGI-3 or Seed IQ parity | future_work | The local Phase-4 result is a toy proxy bridge; no full ARC-AGI-3 leaderboard or Seed IQ reproduction claim is locally anchored. | results/experiment_1165_phase4_active_inference_pilot_v1.json |

## Source Inputs Reviewed

- `ops/experiment_signal_noise_summary.md`: exists=yes; summary=13403
- `ops/mandatory_priority_audit.md`: exists=yes; summary=5753
- `docs/research-notes/self_learning_lineage_decision.md`: exists=yes; summary=2818
- `docs/research-notes/hardware_portfolio_narrowing.md`: exists=yes; summary=3799
- `docs/research-notes/comparator_cite_retire_audit.md`: exists=yes; summary=2977
- `results/experiment_1454_experiment_artifact_signal_noise_classifier.json`: exists=yes; summary=complete_exp1454_signal_noise_ledger_written_1132_artifacts_547_signal_138_noise_447_ambiguous
- `results/experiment_1455_known_issues_mandatory_priority_audit.json`: exists=yes; summary=complete_exp1455_known_issues_priorities_trimmed_from_24_to_7
- `results/experiment_1459_self_learning_nonheadline_lineage_decision.json`: exists=yes; summary=self_learning_headline_pivot_selected_exp1447_verified_growth_only
- `results/experiment_1460_hardware_portfolio_narrowing.json`: exists=yes; summary=active_tracks_narrowed_to_3_with_no KV260 board, Extropic, NPU, or photonic execution claim
- `results/experiment_1461_comparator_integration_cite_retire_audit.json`: exists=yes; summary=comparator_scope_narrowed_6_cite_1_retire_3_watchlist
