# Carnot — Known Issues

**Last Updated:** 2026-04-03

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 1 | PyO3 0.24 doesn't support Python 3.14 natively | Low | Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` |
| ~~2~~ | ~~Gibbs/Boltzmann grad_energy uses numerical finite differences~~ | ~~Resolved~~ | Analytical backprop implemented |
| ~~3~~ | ~~Python test suite not yet written~~ | ~~Resolved~~ | 48 tests, 100% coverage |
| 4 | Ackley and GaussianMixture benchmarks use numerical gradients | Low | Analytical gradients are complex; numerical is acceptable for benchmarks |
| 5 | RETRO-028: llama.cpp#21516 tokenizer bug causes Gemma4-E4B-it to emit only `<unused8>` tokens (token_id=14), producing 0.0% accuracy (false negative in Exp 439) | High — Fix implemented | Use GemmaTransformersLoader (python/carnot/pipeline/gemma_loader.py) instead of any llama.cpp backend. GPU verification pending (Exp 450). |
