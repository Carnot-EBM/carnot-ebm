# Threshold error bound, 2026-10-10

The unchanged 65-point signed-int16 table uses 520 bytes and eleven fractional bits.
Cubic second derivatives are piecewise linear. Exact binary-rational recursion gives their extrema at split-piece endpoints.
Each whole-cell bound is h² sup|f″|/8 plus 1/4096 rounding and an outward basic-arithmetic allowance.
The basis error recurrence is e[k] <= 12 e[k-1] + 64u, e[0]=0, u=2^-53. Widths exceed .19.
Three stages give 10048u. Both endpoint and direct dot arithmetic are included. SciPy PPoly checks extrema independently.
Four cell intervals add before unchanged slope, intercept and temperature arithmetic. Monotone sigmoid maps the empirical logit endpoints.
No all-input error contract was authenticated for unchanged SciPy expit and platform exp. The 1e-12 sigmoid allowance is empirical.
This prevents complete numerical certification. The deployed guard always returns the original direct float64 action.
Strict regions are accept below .25, reject above .75 and escalate on [.25,.75]. Threshold intersections fall back.
Verdict: complete_null_empirical_threshold_guard. Original flips: 3.
Observed interval escapes: 0; guarded mismatches: 0.
Guard readiness: 0; usefulness: 0; actual fast-path fraction: 0.0.
Empirical candidate fraction: 1.0. This fraction grants no deployment permission.
Every vector, interval, action, fault and sparse refresh remains in byte-bound raw primitives.
Cached heads are exposed development. Current model calls and both generalization scores are zero.
Next evidence: authenticate a universal error contract for the unchanged numeric sigmoid before enabling any fast result.
