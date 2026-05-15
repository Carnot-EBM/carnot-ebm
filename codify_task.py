import json

limitations_text = """### Near-Critical Sampler Limits
Carnot and THRML samplers fail to reach the analytic Curie-Weiss equilibrium near the critical beta threshold, as evidenced by exp1692 and exp1709. While the samplers operate as expected in the deep symmetry-broken regime, they exhibit severe critical fluctuations closer to the phase transition, leading to substantial gaps between empirical means and the analytic ground truth.

In the deep symmetry-broken regime (beta = 1.50), the default 500-step burn-in successfully recovers the ground-truth state, leaving a minimal gap of delta_m=0.006 (exact delta_m: 0.005814120984296567) and no bimodal distribution is observed. In the intermediate regime (beta = 1.20), closing the gap to delta_m=0.019 (exact delta_m: 0.01933614741784284) requires a 50,000-step burn-in, which is a two orders of magnitude increase in warmup time. However, near the critical threshold (beta = 1.05), no intervention in the 54-cell ablation grid closes the gap, with `smallest_intervention_closing_gap["1.05"] = null`. A bimodal distribution is observed at both beta = 1.05 and beta = 1.20, confirming the symmetry-breaking failure mode.

These limitations pose significant implications for the Z1 hardware mapping and Phase 4 downstream planning, as the Z1 inherits whichever Carnot sampler primitive is wired through. Feasible mitigations include providing a longer burn-in budget where possible, or employing an explicit symmetry-breaking field if the underlying hardware supports it. Further mitigation for regimes where beta < 1.10 is deferred to future work.
"""

with open("openspec/papers/paper-v6/section-6-limitations.md", "w") as f:
    f.write(limitations_text)

refs_append = """
### Analytic Curie-Weiss Mean-Field Reference and Critical Fluctuations
- **Reference:** Stanley, H. E. (1971). Introduction to Phase Transitions and Critical Phenomena. Oxford University Press. (Provides the standard reference for the m = tanh(beta * m) self-consistency equation).
- **Source Artifacts:** `results/experiment_1709_thrml_critical_fluctuation.json`, `results/experiment_1692_thrml_curie_weiss_ground_truth.json`
- **Relevance to Carnot:** Codifies the baseline expectations for the analytic Curie-Weiss mean-field behavior. Experiment 1709 demonstrates that at beta=1.50, delta_m=0.006 is achieved with 500-step burn-in (exact `delta_m`: 0.005814120984296567). At beta=1.20, delta_m=0.019 is achieved with 50,000-step burn-in (exact `delta_m`: 0.01933614741784284). However, at beta=1.05, `smallest_intervention_closing_gap["1.05"] = null`, with bimodality observed at both 1.05 and 1.20.
"""

with open("research-references.md", "a") as f:
    f.write(refs_append)

known_issues_append = """
### exp1709 Near-Critical Sampler Limit (.176+ MANDATORY Z1 + Phase 4 downstream)

**Origin:** Findings from 54-cell ablation in `results/experiment_1709_thrml_critical_fluctuation.json` and ground-truth comparison in `results/experiment_1692_thrml_curie_weiss_ground_truth.json`.

**What:** Carnot and THRML samplers miss the analytic Curie-Weiss equilibrium near the critical beta. The ablation grid shows:
- At beta=1.50 (deep symmetry-broken): default 500-step burn-in recovers ground-truth within delta_m=0.006. (Exact `smallest_intervention_closing_gap["1.5"]` shows `delta_m`: 0.005814120984296567). `bimodal_distribution_observed["1.5"]` is false.
- At beta=1.20 (intermediate): closing the gap to delta_m=0.019 requires a 50,000-step burn-in. (Exact `smallest_intervention_closing_gap["1.2"]` shows `delta_m`: 0.01933614741784284). `bimodal_distribution_observed["1.2"]` is true.
- At beta=1.05 (near critical beta_c=1.0): NO intervention in the 54-cell ablation closes the gap (`smallest_intervention_closing_gap["1.05"] = null`). `bimodal_distribution_observed["1.05"]` is true.

**Relevance to Carnot:** This is a ship-eligible finding for paper-v6 §6 (limitations) and has direct Z1 hardware-mapping implications because Z1 inherits the Carnot sampler primitive. Downstream planning must account for longer burn-in budgets at beta=1.20 and fundamental limits at beta=1.05. Explicit symmetry-breaking fields may be required if hardware supports them.
"""

with open("ops/known-issues.md", "a") as f:
    f.write(known_issues_append)
