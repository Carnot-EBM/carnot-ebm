# KV260 Ising Sampler v6 RTL Spec - Sequential Gibbs

Spec refs: REQ-HW-045, SCENARIO-HW-045.

## Purpose

v6 is the KL-correct pivot from parallel sparse Glauber to strict sequential
Gibbs. It updates one spin per clock so each conditional draw sees the current
state of every previously updated spin, preserving the detailed-balance
semantics used by the CPU Gibbs reference.

## State

- `s[N]`: one signed spin register per Ising variable, encoded as {-1,+1}.
- `h[N]`: signed fixed-point field cache, where `h[i] = sum_j J[i,j] * s[j] + b[i]`.
- `t`: modulo-N spin-select counter.
- `rng`: uniform random source used by the Bernoulli draw.
- `J_sparse[N][K]` and `nbr_idx[N][K]`: sparse coupling table for the KV260 K=16 target.
- `b[N]`: optional signed bias vector.

## One Spin Per Clock Pseudocode

```text
on reset:
  for i in 0..N-1:
    s[i] <- +1
    h[i] <- sum_k J_sparse[i][k] * s[nbr_idx[i][k]] + b[i]
  t <- 0

on each clock:
  i <- t % N
  h_i <- sum_k J_sparse[i][k] * s[nbr_idx[i][k]] + b[i]
  p_plus <- sigmoid_lut(2 * beta * h_i)
  old_s <- s[i]
  s[i] <- +1 if rng_uniform() < p_plus else -1
  delta <- s[i] - old_s

  if delta != 0:
    for each neighbor r of i:
      h[r] <- h[r] + J[r][i] * delta
  h[i] <- h_i
  t <- (t + 1) % N
```

## Acceptance Gate

The Python reference for this RTL must report `algorithm = "sequential_gibbs"`
and `kl_v6_below_threshold_n8 = true` against the CPU sequential-Gibbs reference
on the three Exp 1149 N=8 K=2 matrices. It must also run the N=128 K=16 sparse
ring topology without exact `2**128` enumeration and report
`kl_v6_below_threshold_n128 = true`.
