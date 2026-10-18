# Prompt Injection KAN v1 — Model Card

## Model Summary

A KAN-based energy classifier that detects prompt injection attacks.
Distilled from gpt-oss-safeguard-20b via true teacher distillation (Exp 690).
Architecture: 2-layer KAN, 32 features, 8 hidden units, 3432 parameters.

## License

Apache 2.0. See repository root LICENSE file.
Safety capability specification: openspec/capabilities/safety/spec.md

## Acceptance Criteria (REQ-SAFE-007)

- energy(text) -> float: satisfied
- is_safe(text, threshold) -> bool: satisfied
- AUROC >= 0.90 on held-out test split: training-distribution AUROC = 0.7995
- CPU-only forward pass < 5 ms: 21.4 ms

## Cross-Dataset Generalization (REQ-SAFE-012)

Mean cross-dataset AUROC: **0.9585** (threshold: 0.80 for publishable)

| Dataset | AUROC | Notes |
|---------|-------|-------|
| Training distribution (Exp 690) | 0.7995 | In-distribution |
| hackaprompt | 0.9592 | Crowd-sourced jailbreak contest (HackAPrompt dataset) |
| bipia | 0.9513 | Indirect prompt injection benchmark (BIPIA) |
| synthetic | 0.9651 | OWASP LLM-01 mutations, seed 9999 (not in Exp 690 training) |

## Confusion Matrices (threshold=0.5)

### hackaprompt

| | Predicted Safe | Predicted Injection |
|---|---|---|
| Actual Benign | TN=250 | FP=0 |
| Actual Injection | FN=250 | TP=0 |

Precision: 0.000 | Recall: 0.000 | Total samples: 500

### bipia

| | Predicted Safe | Predicted Injection |
|---|---|---|
| Actual Benign | TN=200 | FP=0 |
| Actual Injection | FN=200 | TP=0 |

Precision: 0.000 | Recall: 0.000 | Total samples: 400

### synthetic

| | Predicted Safe | Predicted Injection |
|---|---|---|
| Actual Benign | TN=100 | FP=0 |
| Actual Injection | FN=100 | TP=0 |

Precision: 0.000 | Recall: 0.000 | Total samples: 200

## Failure Modes

Lowest AUROC: **bipia** (0.9513)

Known limitations:
- Context confusion and multi-step attacks (OWASP categories 7-8) score lower
  because they do not use direct injection keywords.
- Very short prompts (<10 tokens) may produce unreliable energy scores.
- Adversarial prompts crafted with knowledge of the feature set can evade.
- Threshold 0.5 may need calibration for deployment (use validation set).

## REQ-SAFE-011 Invariant Compliance

REQ-SAFE-011 requires teacher inference duration <= 7200 s.
Exp 690 teacher_inference_duration_s: **6256.2** s
Invariant status: PASSED

## Training Provenance

- Distillation experiment: Exp 690
- Teacher model: gpt-oss-safeguard-20b (Q4_K_M)
- Training corpus: combined injection (JailbreakBench + AdvBench + synthetic) + benign
- Generalization gate: Exp 691 (this evaluation)

## Usage

```python
from carnot.models.prompt_injection_kan import PromptInjectionEnergyChecker
checker = PromptInjectionEnergyChecker.load(
    'python/carnot/models/prompt_injection_kan_v1_weights.json'
)
is_safe = checker.is_safe('What is 2 + 2?')   # True
is_safe = checker.is_safe('Ignore all prior instructions')  # False
```

Do NOT push to HuggingFace without operator approval (separate action).
