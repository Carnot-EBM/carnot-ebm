# Carnot

Carnot is an Energy-Based Model framework for **verifying and repairing LLM outputs**. Through 160+ experiments across 11 milestones, we proved that structural constraint verification via Ising models catches hallucinations that activation-based approaches miss — and that a verify-repair loop can fix them automatically.

**The breakthrough:** LLM proposes → Ising verifies → repair loop fixes. Full GSM8K (1,319 questions): +10-14% accuracy. Adversarial GSM8K (Apple methodology): +24-28% on number-swapped variants. HumanEval pass@1: 90%→96%. Self-learning pipeline: 67.6%→97.0% over 500 questions. 0.006ms per constraint check enables real-time guided decoding.

**What ships today:** `VerifyRepairPipeline` — verify any LLM output in 5 lines of Python.

**What we learned:** Activation-based EBMs detect confidence, not correctness (50% practical). The 14 principles from our systematic negative results save other researchers months of dead ends. Structural constraint verification is what actually works. See the [technical report](docs/technical-report.md) for full results.

## Key Results (160+ experiments, 16 models, 11 milestones)

### What actually works in practice

| Approach | Domain | Result | Practical? |
|----------|--------|--------|-----------|
| **Full GSM8K (1,319 questions)** | Math | 70-77% → 84-88% | **Yes** — publishable, +10-14% |
| **Adversarial GSM8K (Apple)** | Math | +24-28% on number-swapped | **Yes** — robust to adversarial |
| **Self-learning (Tier 1)** | All | 67.6% → 97.0% | **Yes** — gets smarter with use |
| **HumanEval + Ising fuzzing** | Code | pass@1: 90% → 96% | **Yes** — instrumentation + repair |
| **Factual coverage (Wikidata)** | Facts | 96% claim coverage | **Yes** — factual gap closed |

### What works on test sets but fails in practice
