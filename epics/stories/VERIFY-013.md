# Epic: VERIFY-013 - CoT Monitorability Audit and Fallback Policy

**Status:** Completed
**Goal:** Publish a live Exp 213 audit at `results/experiment_213_results.json`
plus a machine-readable fallback policy at
`results/monitorability_policy_213.json` so Carnot can decide when to request
structured reasoning, when terse output is enough, and when free-form traces
should be distrusted.
**Rationale:** Exp 203 / 206 / 207 showed that wrong answers can remain hidden
behind plausible prose, while Exp 210's monitorability scan argued that CoT
should be optional evidence gated by measurement. Exp 213 turns that warning
into an audited policy grounded in Qwen3.5-0.8B and Gemma4-E4B-it behavior on
the Exp 211 benchmark slices Carnot actually cares about next.

## Stories
- [x] Add `REQ-VERIFY-013`, `REQ-VERIFY-014`, `SCENARIO-VERIFY-013`, and `SCENARIO-VERIFY-014` to the verifiable-reasoning spec
- [x] Write tests first for the Exp 213 subset selection, scoring, policy generation, and artifact-writing workflow
- [x] Implement `scripts/experiment_213_monitorability_audit.py`
- [x] Generate `results/experiment_213_results.json` and `results/monitorability_policy_213.json`
- [x] Run unit tests, the full Python suite, targeted 100% coverage on the new script, and the applicable E2E/integration checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`, and `ops/metrics.md`
