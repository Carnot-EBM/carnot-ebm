# Carnot — Known Issues

**Last Updated:** 2026-04-03

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 1 | PyO3 0.24 doesn't support Python 3.14 natively | Low | Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` |
| 2 | Gibbs/Boltzmann grad_energy uses numerical finite differences | Medium | Implement analytical backprop |
| 3 | Python test suite not yet written | High | Rust tests exist, Python tests pending |
