# Experiment claim-refutation audit

One question per artifact: what would REFUTE the headline claim, and was that
checked? Fabrication is out of scope (adversarial_verify covers it); this audit
targets claims that are true by construction, circular, in-sample, baseline-weak,
or contradicted by their own rows.

This audit never edits an artifact and never blocks anything. It surfaces; the
operator decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity
guard rest on evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CLAIM_SUPPORTED | 1 |
| NO_CLAIM | 1 |
| CANNOT_DETERMINE | 6 |

## experiment_7983_reserved_decisions.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7984_v692_evidence_ablation.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7985_delayed_acquisition.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7987_issued_confidence.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7988_v692_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_7989_v692_service_cost.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Evaluating verifier arms against cached candidates incurs measurable CPU engineering overhead over the baseline without providing fresh inference speedup, resulting in an honest null service cost finding.

## WHAT WOULD REFUTE IT
Any verifier arm demonstrating lower latency or higher throughput than the scalar baseline (i.e., a statistically significant negative paired overhead with confidence intervals bounded below zero), or any observation supporting a fresh inference speedup.

## WAS THAT CHECKED
Yes; checked in `paired_cpu_comparisons`, `complete_service_cost`, and `service_summary` across 3,840 execution rows comprising 32 independent source groups measured over 10 randomized timing repeats across arms and storage configurations.

## EVIDENCE
- `"honest_verdict": "complete_null_service_cost"`
- `"verdict_class": "null"`
- `"efficiency": "descriptive_only"`
- `"claim": "engineering_overhead_only"`
- `"fresh_inference_speedup_claim": false`
- `"arm": "scalar"`
- `"arm": "gibbs"`
- `"mean_overhead_s": 0.038992330025`
- `0.028194533257421255`
- `0.04979012679257874`
- `"p50_s": 1.4769137699553279`
- `"throughput_requests_s": 0.6867034645276905`
- `"p50_s": 1.527494572955328`
- `"throughput_requests_s": 0.6670046344545669`

## RECOMMENDATION
KEEP

## experiment_7990_v692_hardware_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An observation of actual measured hardware acceleration, positive speedup, or completed device-side execution over host baselines would refute the premise that no performance or capability claim is made.

## WAS THAT CHECKED
No. The artifact is a read-only host aggregation receipt; device workloads were not run and device execution was not measured.

## EVIDENCE
- `hardware_speedup_claimed`: `false`
- `hardware_advantage`: `unmeasured`
- `acceleration`: `no measured device execution`
- `current_device_execution_count`: `0`
- `inference_substrate`: `aggregation_from_upstream_artifacts`
- `honest_verdict`: `complete_blocked_required_custody_or_service`
- `arm`: `historical_accounting`
- `status`: `historical_read_only`

## RECOMMENDATION
KEEP

## experiment_7991_v692_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
