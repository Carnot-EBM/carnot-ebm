# Result: Negative

Experiment 37 proved this approach doesn't work. General-purpose sentence
encoders embed topic similarity, not factual correctness. "Fortune cookies
from China" and "Fortune cookies from San Francisco" have 0.828 cosine
similarity — the EBM can't learn to distinguish them.

The NCE loss never decreased (stuck at ln(2) = 1.386 for all 500 epochs).
EBM discrimination: 57.5%. Repair: 0/50 improved.

**Superseded by:** Roadmap v6 constraint-based reasoning via Ising/thrml.
