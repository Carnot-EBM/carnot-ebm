# Theory Additions (v256)

## §2 Theoretical Background
Recent work has established a formal bijection between autoregressive language models and Energy-Based Models (EBMs), along with corresponding distillation error bounds (blondel2025arm). This ARM-EBM bijection provides the crucial theoretical scaffolding for phase 3 of our verify-repair architecture.

## §3 Architecture
Our LLM-verifier pipeline can be modeled as an absorbing Markov chain. Theoretical bounds guarantee that such pipelines terminate in $E[n] \le 4/\delta$ iterations (dantas2025four), providing a formal convergence guarantee for Carnot's repair loop.

Furthermore, Carnot's verify-repair design is validated by recent developments in dual-timescale LLM architectures. Fast-Slow Training (FST), which employs slow frozen weights alongside fast context-updated weights, mirrors our architecture and demonstrates a 3x sample efficiency improvement compared to RL-only approaches (hashimoto2026fst).

## Citations
- blondel2025arm: arXiv:2512.15605
- dantas2025four: arXiv:2512.02080
- hashimoto2026fst: arXiv:2605.12484
