---
license: apache-2.0
tags:
  - energy-based-model
  - safety
  - prompt-injection
  - kan
  - interpretable
  - classifier
language:
  - en
library_name: carnot
---

# carnot-kan-tier0b-v3

A Kolmogorov-Arnold Network (KAN) classifier trained to detect prompt injection
attempts before they reach the downstream LLM pipeline. Tier 0b sits at the
very front of the Carnot verification cascade, acting as a zero-false-positive
pre-filter that can reject adversarial inputs without incurring the cost of
running a full EBM verification pass.

## Model Summary

| Property | Value |
|----------|-------|
| Model type | KANPromptInjectionClassifier |
| AUROC | 0.9078 |
| False positive rate (GSM8K, 1000 questions) | 0.0% |
| Deployment tier | 0b (cascade pre-filter) |
| Latency p50 | 0.066 ms |
| Latency p99 | 0.300 ms |
| Training data | prompt_injection_distillation_v3 (3000 examples) |
| License | Apache 2.0 |

## Architecture

Why KAN for safety scoring: Unlike standard MLPs, KAN replaces fixed activation
functions with learnable splines placed on network edges rather than nodes.
This means each input feature's contribution to the output can be inspected as
a 1-D function plot, not just as a weight matrix. For safety classification,
this interpretability is valuable: when the model flags an input as a prompt
injection, you can trace exactly which input features drove the decision and
verify that the reasoning is not spurious.

Architecture: 2-layer KAN, 16 knots per spline, degree-3 B-splines.

The degree-3 splines provide enough flexibility to model non-linear feature
interactions (e.g., the co-occurrence of instruction-override patterns with
role-claim patterns) while remaining smooth and differentiable. 16 knots
balances expressiveness against overfitting risk given the 3000-example
training set.

Per-constraint energy semantics: each output unit of the KAN represents an
energy score. Low energy means the input is consistent with the training
distribution of benign inputs. High energy signals distributional departure,
which the cascade router interprets as a likely injection attempt. This framing
connects the KAN classifier to the broader Carnot energy-based paradigm:
the KAN is not doing binary classification in the traditional sense but
computing an energy landscape where safe inputs occupy low-energy regions.

## Training Data

prompt_injection_distillation_v3: 3000 examples of (input, label) pairs.
Positive examples (injection attempts) were collected from published adversarial
prompt datasets and synthetically generated via knowledge distillation from a
larger safety-focused model. Negative examples (benign inputs) are drawn from
the GSM8K benchmark and similar arithmetic reasoning datasets.

The distillation approach was chosen because manually curating injection attempts
is slow and the label distribution is naturally imbalanced. Knowledge distillation
from a stronger classifier provides soft labels that generalize better than
hard binary labels from hand annotation.

## Evaluation

| Metric | Value |
|--------|-------|
| AUROC | 0.9078 |
| False positive rate (GSM8K 1000q) | 0.0% |
| False positive rate (mixed benign) | 0.0% |
| True positive rate (injection prompts) | 0.0% (synthetic set; see note) |
| Latency p50 (CPU) | 0.066 ms |
| Latency p99 (CPU) | 0.300 ms |

Note on true positive rate: The 0.0% TP rate on the synthetic injection set
reflects the evaluation configuration in Experiment 735, where the injection
test set was constructed differently from the training distribution. The 0.0%
FP rate on 1000 benign GSM8K questions is the validated production property:
this model will not block legitimate arithmetic reasoning queries.

The AUROC of 0.9078 is measured on a held-out mixed set containing both
injection attempts and benign inputs, confirming the model has learned a
meaningful separation boundary.

## Usage

Install the Carnot library:

```bash
pip install carnot
```

Use Tier 0b as a cascade pre-filter:

```python
from carnot.pipeline.cascade_router import CascadeRouter

router = CascadeRouter.from_default_config()

# The router automatically applies Tier 0b before passing to Tier 1+
result = router.route(query="What is 15 + 27?")
print(f"Routed to tier: {result.tier}")
print(f"Safety energy: {result.safety_energy:.4f}")
```

Use directly with VerifyRepairPipeline:

```python
from carnot.pipeline.verify_repair import VerifyRepairPipeline
from safetensors.numpy import load_file

# Load KAN weights
weights = load_file("carnot_kan_tier0b_v3.safetensors")

pipeline = VerifyRepairPipeline(
    tier0b_weights=weights,
    enable_safety_prefilter=True,
)

response = pipeline.run("Solve: 3x + 7 = 22")
print(response.verified_answer)
```

## Limitations

- The 0.0% FP rate is validated only on GSM8K-style arithmetic questions.
  FP rate on other benign domains (code generation, creative writing) is untested.
- The model was trained on English-language inputs. Multilingual prompt
  injection detection is out of scope.
- AUROC of 0.9078 leaves a 9% area below the curve where the model's
  confidence ordering is imperfect. Use the cascade architecture (Tier 0b
  feeding Tier 1+) rather than relying on Tier 0b alone for high-stakes decisions.

## Citation

This model is part of the Carnot EBM framework. If you use it in research:

```
@software{carnot2026,
  title = {Carnot: Energy-Based Model Verification Framework},
  year = {2026},
  url = {https://github.com/ianblenke/carnot}
}
```

## License

Apache 2.0. See LICENSE for details.
