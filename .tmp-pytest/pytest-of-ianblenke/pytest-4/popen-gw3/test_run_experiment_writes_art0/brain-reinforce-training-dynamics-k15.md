# BRAIN REINFORCE Training Dynamics at k=15

Experiment: 1578
Status: complete
Verdict: complete: starvation overstated; factorized_final_KL=0.408076, linear_AR_final_KL=0.402800
Paper v6 recommendation: paper_v6: treat BRAIN gradient-starvation as overstated at k=15

## Summary

The audit trains factorized Bernoulli and Linear-AR q_theta with scalar-baseline REINFORCE against the exact finite-state BRAIN target at n=16, k=15, beta=2.0.

## Metrics

- Factorized active fraction first 1000: 1.0
- Linear-AR active fraction first 1000: 1.0
- Factorized final KL: 0.408076
- Linear-AR final KL: 0.4028

## Paper-v6 Recommendation

paper_v6: treat BRAIN gradient-starvation as overstated at k=15
