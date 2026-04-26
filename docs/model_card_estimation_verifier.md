# Model Card: EstimationVerifier

## Model Description

EstimationVerifier is Carnot's Tier 1 fast constraint checker. It verifies
arithmetic reasoning traces by checking whether the extracted numerical answer
falls within a plausible range computed from the question's operands and
operation type.

**Model ID:** Carnot-EBM/estimation-verifier-v1
**Architecture:** Rule-based range estimator (no learned weights)
**Framework:** Python (no JAX/PyTorch dependency)
**Experiment:** Exp 908 (EstimationVerifier SVAMP AUC vs FoVer baseline)

## Architecture Details

EstimationVerifier is not a neural network but a deterministic constraint
checker. It:

1. **Extracts the numerical answer** from the LLM response using regex.
2. **Identifies the operation type** from the question text (add, subtract,
   multiply, divide, or unknown) based on keyword matching.
3. **Computes a plausible range** from the operands found in the question:
   - Addition: [min_operand, sum_of_all]
   - Subtraction: [0, max_operand]
   - Multiplication: [min_operand, product_of_all]
   - Division: [1, max_operand]
   - Unknown: [0, 3 * max_operand]
4. **Returns a violation_prob** of 0.0 if the answer is in range, 1.0 if not.

This approach is intentionally simple: it trades accuracy for speed and
interpretability. The range estimator catches gross hallucinations (answers
off by orders of magnitude) cheaply, leaving subtle errors for Tier 2 (VJEPA v2).

## Evaluation

| Metric | Value |
|--------|-------|
| SVAMP AUC (EstimationVerifier) | 0.90 |
| SVAMP AUC (FoVer baseline) | 0.125 |
| Signed improvement over baseline | 0.775 |
| Evaluation set size | 20 question-answer pairs |

The FoVer baseline (Exp 872) collapsed because SVAMP problems do not have
chain-of-thought labels that FoVer's labeling pipeline could parse. By replacing
FoVer with a label-free range estimator, Exp 908 recovered from AUC=0.125 to
AUC=0.90 on the same SVAMP evaluation set.

## Limitations

- **Rule-based range:** The plausible range is a heuristic, not a learned
  distribution. Problems with unusual operand relationships (percentages, unit
  conversions, multi-step chains) may produce incorrect ranges.
- **Keyword operation detection:** Operation type is detected by keyword matching
  ("how many more" -> subtract), which fails on paraphrase-heavy problem text.
- **Binary violation_prob:** The output is 0.0 or 1.0 (in-range or out-of-range),
  not a calibrated probability. This limits its use as a soft ranking signal.
- **No learned weights:** The model cannot improve from training data without
  adding a learned range-estimation component.
- **SVAMP-focused evaluation:** The 20-question evaluation set is SVAMP word
  problems. Performance on other arithmetic domains is not measured.

## Intended Use

EstimationVerifier is intended as a fast Tier 1 gate in Carnot's verification
cascade. It catches gross arithmetic hallucinations before more expensive
VJEPA v2 or energy-based verification. It is not intended for:

- Detecting subtle reasoning errors that produce in-range wrong answers
- Non-arithmetic domains
- Production use without cascade integration

## Decentralization

Per Carnot's decentralization policy (CLAUDE.md rule 3), this model is published
to at least two independent distribution channels:

- HuggingFace Hub: https://huggingface.co/Carnot-EBM/estimation-verifier-v1
- Gitea mirror: ssh://git@gitea.noblehunt.org:2222/ianblenke/carnot.git

The implementation is pure Python with no vendor-specific dependencies.

## Specifications

- REQ-VER-085 (EstimationVerifier SVAMP constraint checking)
- SCENARIO-VER-085a (SVAMP AUC vs FoVer baseline)
