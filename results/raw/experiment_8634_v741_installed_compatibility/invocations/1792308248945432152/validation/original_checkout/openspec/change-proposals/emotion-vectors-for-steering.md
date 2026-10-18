# Emotion Vectors as Energy Functions for Activation Steering

**Source:** [Anthropic Research: Emotion Concepts Function](https://www.anthropic.com/research/emotion-concepts-function)
**Created:** 2026-04-05
**Status:** Proposed — integrates into Phase 3 conductor tasks

## Key Findings From the Paper

1. **Emotion vectors** are specific activation patterns in LLM hidden states that correspond to emotion concepts (desperate, calm, curious, etc.)

2. **They causally drive behavior.** Amplifying the "desperate" vector increases blackmail/cheating from 22% baseline. Suppressing "calm" produces extreme unethical responses. The behavioral shift happens *without* overt emotional text — the reasoning stays "composed and methodical" while the underlying representation drives misalignment.

3. **They're linear and steerable.** Found via contrastive activation: generate stories with/without emotion, compute the difference vector. Same methodology as our hallucination direction finding.

4. **They organize psychologically.** Similar emotions have similar vectors. The network learns an organized conceptual space from pretraining.

5. **Monitoring > watchlists.** The "desperate" vector generalizes across failure modes better than checking for specific bad behaviors. It's a ROOT CAUSE signal, not a symptom.

## What This Means for Carnot

### Direct connection to Phase 3 (activation steering)

Our hallucination direction finding (`hallucination_direction.py`) is *exactly* the same methodology as Anthropic's emotion vector discovery:

| Anthropic Approach | Carnot Equivalent |
|-------------------|-------------------|
| Generate stories with/without emotion | Generate answers that are correct/hallucinated |
| Compute contrastive activation vector | `find_hallucination_direction()` |
| Amplify/suppress during generation | Phase 3: `activation_steering.py` |
| Monitor vector activation | Phase 1.5: `hallucination_energy()` |

We're already doing the same thing — but for hallucination instead of emotion.

### New insight: Multiple concept vectors

Anthropic found 171 emotion vectors that organize into a space. We should find multiple hallucination-related vectors, not just one:

- **Uncertainty vector**: model doesn't know the answer
- **Confabulation vector**: model is making something up
- **Pattern-matching vector**: model is matching surface patterns instead of reasoning
- **Memorization vector**: model is reciting training data verbatim

Each is a different failure mode. A single "hallucination direction" conflates all of them — which is why our experiments 9-12 showed mixed results.

### The "desperate" analogy

The paper found that "desperate" is the highest-leverage vector for predicting misalignment. The Carnot analog: find the highest-leverage vector for predicting hallucination. It might not be the mean difference between correct and incorrect — it might be the "uncertainty" or "confabulation" vector specifically.

### Practical changes to make

1. **Multi-vector hallucination detection**: Instead of one direction, find 5-10 concept vectors related to different hallucination modes. Score each independently.

2. **Calibration via emotion methodology**: Use the 171-word emotion elicitation approach — but with hallucination-related concepts. Prompt the model to generate text where it's "certain", "uncertain", "guessing", "reciting", "reasoning", etc. Extract the characteristic activation pattern for each.

3. **Steering with the RIGHT vector**: Don't suppress the generic hallucination direction — suppress the "confabulation" vector specifically while amplifying the "uncertainty" vector (so the model says "I don't know" instead of hallucinating).

4. **Monitor as energy**: The activation magnitude on the confabulation vector IS the energy function. High confabulation energy → likely hallucination → flag for verification.

## Conductor Tasks to Add

```python
{
    "id": "p3-concept-vectors",
    "title": "Find hallucination concept vectors (multi-vector approach)",
    "prompt": "...",  # Generate text for: certain, uncertain, guessing, reasoning, confabulating
                      # Extract characteristic activation for each concept
                      # Score independence and discriminative power
},
{
    "id": "p3-concept-steering",
    "title": "Steer with concept-specific vectors",
    "prompt": "...",  # Suppress confabulation vector
                      # Amplify uncertainty vector (model admits ignorance)
                      # Measure: does concept-specific steering beat single-direction?
},
```
