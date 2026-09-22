# V657 method map

Date: 2026-09-22. Scope: advisory method accounting.
Paper results are not local Carnot results.

| Method | Primary source and section | Adaptation | Counterexample | Named task |
|---|---|---|---|---|
| binary_scoring_token_expectation | [arXiv:2607.05391v2 (2026-07-07)](https://arxiv.org/html/2607.05391), Section 3.2, equation 3.1 | Use the expectation over the two existing semantic option logits. | A sharp binary score can still be miscalibrated against source labels. | exp7504-evidence-interface, exp7507-static-evaluation |
| equal_detector_access | [arXiv:2606.06959v1 (2026-06-08)](https://arxiv.org/html/2606.06959), Sections 2.1-2.2 and detector implementation appendix | Ledger equal source, window, option-order, label, and metric access. | Equal prompts do not help if one detector receives hidden labels. | exp7504-evidence-interface, exp7508-static-audit |
| aligned_vs_shuffled_information | [arXiv:2606.26476v1 (2026-06-25)](https://arxiv.org/html/2606.26476), Section 4.4, aligned-versus-shuffled five-arm suite | Shuffle only labels available within the same release batch. | A cross-batch shuffle can import future labels and fake causality. | exp7506-causal-prototype, exp7509-causal-online |
| kan_retention | [arXiv:2511.12828v1 (2025-11-17)](https://arxiv.org/html/2511.12828), Bounded Retention, Lemma 1 and Theorem 1 | Measure local-support overlap and a frozen held-out retention set. | Local splines still forget when later supports overlap earlier supports. | exp7506-causal-prototype, exp7509-causal-online |
| whole_service_hardware_accounting | [arXiv:2602.15985v2 (2026-09-04)](https://arxiv.org/html/2602.15985), Sections 2.3, 4.2, and 5.3.2 end-to-end TTS | Count conversion, state, transfer, update, fsync, and acknowledgement. | A fast kernel can leave the complete service slower after orchestration. | exp7513-placement-continuity, exp7514-service-trace |

The primary HTML sections were read with bounded sequential requests.
No paper benchmark number is treated as a local measurement.
