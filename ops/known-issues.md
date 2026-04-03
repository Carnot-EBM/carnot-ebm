# Carnot — Known Issues

**Last Updated:** 2026-04-03

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 1 | PyO3 0.24 doesn't support Python 3.14 natively | Low | Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` |
| ~~2~~ | ~~Gibbs/Boltzmann grad_energy uses numerical finite differences~~ | ~~Resolved~~ | Analytical backprop implemented |
| ~~3~~ | ~~Python test suite not yet written~~ | ~~Resolved~~ | 48 tests, 100% coverage |
| 4 | Ackley and GaussianMixture benchmarks use numerical gradients | Low | Analytical gradients are complex; numerical is acceptable for benchmarks |
