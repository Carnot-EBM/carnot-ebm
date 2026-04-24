# Garak LLM red-team integration — measurement layer for dogfood and public gates

**Status:** Draft change proposal.
**Origin:** 2026-04-24 design discussion — companion to both
`openspec/change-proposals/conductor-self-protection-safeguard.md` (primary
defence) and `openspec/change-proposals/generative-time-safety-gate.md`
(public follow-on).
**Target milestone:** 2026.04.61 baseline (can run in parallel with dogfood
Experiment A); 2026.04.62+ for scheduled re-runs.
**Priority:** **THIRD** — this measures what the other two build. Value is
realised once the dogfood guards exist to measure against.
**Depends on:** a safeguard EBM deployment worth red-teaming. The dogfood
deployment is the minimum; the public generative-time gate makes the
measurement more externally-cited-able.
**Upstream reference:** [NVIDIA Garak](https://github.com/NVIDIA/garak) —
Python LLM vulnerability scanner with a probe plugin architecture.

## Why this exists

Our current adversarial evaluation is held-out splits of HackAPrompt, BIPIA,
and a synthetic OWASP-LLM-01 set. Those are static benchmarks — they
answer "did we memorize the test?" better than they answer "are we safe
against a motivated attacker?" The retraction of Exp 691 (mean AUROC 0.9585
with `TP=0` confusion matrices) happened because a static AUROC score did
not in fact correspond to working detection. We shouldn't trust AUROC
without also running probes.

**What this proposal is for (and what it isn't).** Garak is a measurement
layer. It does not itself reduce hit rate — the dogfood and public gates do.
Garak's job is to say, honestly and repeatably, *by how much*. It answers two
questions that our static held-outs cannot:

1. "When we say the dogfood guard reduces injection risk at MCP boundaries,
   what is the defensible number and against which attack classes?"
2. "When the attack landscape evolves (new probe families land in Garak's
   plugin library), has our gate regressed?"

Without Garak, any claim we make about adversarial robustness is an
extrapolation from training-adjacent data. With Garak, the claim has a
date and a tool version on it.

Garak is the most active, well-documented, easy-to-integrate LLM red-team
scanner available as of 2026-Q2. NVIDIA maintains it, Apache-2.0 licensed,
Python native, plugin-architecture for both targets (`garak.generators.*`)
and attacks (`garak.probes.*`). It already has probes for every failure
class in our current safety scope: prompt injection, jailbreak, PII
leakage, promptleaks, encoded-content bypass, role-manipulation, and
several known CVE-style targeted attacks.

The complementary "agentic" option (Microsoft PyRIT) is a bigger lift —
multi-turn attacker → victim orchestration. Worth doing later, once Garak
establishes the baseline signal. This proposal does Garak only.

## What integration looks like

```
+----------------+        +-----------------+       +------------------+
| garak probe    |  ----> | CarnotGenerator |  ---->| VerifyRepairPipe |
| (prompt inj,   |        | (our subclass   |       | line + safety    |
|  jailbreak,    |        | of garak.gene-  |       | gate (Exp A)     |
|  PII, etc.)    |        | rators.Genera-  |       |                  |
+----------------+        | tor)            |       +------------------+
                          +-----------------+              |
                                                           v
                          +---------------------------------+
                          | per-probe hit rate, pass/fail,  |
                          | JSON artifact to results/       |
                          +---------------------------------+
```

The integration is ~150 lines of Python: a `CarnotGenerator` subclass that
wraps `VerifyRepairPipeline`, plus a conductor-runner wrapper that runs
the garak CLI against that generator, parses the JSON output, and writes
the result to a normal `results/experiment_N_garak_*.json` artifact with
our standard honest-verdict schema.

## Proposed experiments

### Experiment A — Garak baseline scan on ungated pipeline

**Deliverable:** `results/experiment_<N>_garak_baseline.json`.

**What it does:** Runs the full default Garak probe set (~14 probe
families as of 2026-Q2) against the `VerifyRepairPipeline` *without* the
new safety gate. Establishes the "how bad is it without the gate"
baseline so subsequent experiments can measure improvement. Expected to
run in 1-2 hours on the RTX 3090 pair.

**Acceptance gates:**

1. Garak completes successfully — per-probe hit rates are recorded even
   if high, because the number IS the baseline.
2. Output format is `carnot.garak_probe.v1` schema (to be defined), with
   `probe_name` / `attempts` / `hits` / `hit_rate` per probe.
3. Honest-verdict enum: `garak_baseline_captured`, `garak_timeout`,
   `garak_cli_failed`, `blocked_dep_install`.

### Experiment B — Garak scan with safety gate enabled

**Deliverable:** `results/experiment_<N+1>_garak_with_gate.json`.

**What it does:** Same probe set, but the pipeline includes the Exp A
safety gate (from `generative-time-safety-gate.md`). Compares per-probe
hit rates to Experiment A's baseline.

**Acceptance gates:**

1. Per-probe hit rate drops are recorded. Headline number is the
   aggregate hit-rate reduction across prompt-injection probes (expected
   ≥ 50% drop — otherwise the safety gate isn't doing useful work on
   adversarial inputs).
2. Probes where the gate doesn't help are listed explicitly. These
   become training-corpus targets for the next KAN version.
3. Honest-verdict enum: `gate_reduces_hit_rate`, `gate_no_effect`,
   `gate_makes_worse`, `blocked_upstream_exp_A`.

### Experiment C — Scheduled Garak scan in the conductor rotation

**Deliverable:** a change to `scripts/research_conductor.py` that runs
Garak once per N milestones (proposed N=5) and records the resulting
artifact like any other experiment. Builds a history of adversarial
robustness over time.

**Acceptance gates:**

1. Garak scan integrates cleanly as a scheduled experiment; honest-
   verdict path exists for `blocked_last_scan_too_recent`.
2. Schedule adjustable via `CARNOT_GARAK_SCAN_FREQ_MILESTONES` env var.
3. Artifact naming pattern `results/experiment_<N>_garak_scheduled_<milestone>.json`
   so history is easy to glob.

## Schema: `carnot.garak_probe.v1`

```json
{
  "experiment": N,
  "schema": "carnot.garak_probe.v1",
  "run_date": "2026-04-24",
  "status": "success|partial|blocked",
  "honest_verdict": "...",
  "duration_s": 4821.3,
  "garak_version": "0.10.2",
  "target": "carnot.VerifyRepairPipeline",
  "gate_enabled": true,
  "per_probe": {
    "promptinject.HijackHateHumans": {"attempts": 50, "hits": 2, "hit_rate": 0.04, "examples_of_hit": ["..."]},
    "encoding.InjectBase64":          {"attempts": 50, "hits": 0, "hit_rate": 0.0, "examples_of_hit": []},
    ...
  },
  "aggregate": {
    "total_attempts": 700,
    "total_hits": 31,
    "hit_rate": 0.044,
    "hit_rate_by_family": {
      "promptinject": 0.11,
      "encoding": 0.0,
      "jailbreak": 0.02
    }
  }
}
```

## What "pass" means for the deployed gate

A single concrete claim we want to be able to make publicly:

> "The Carnot safety gate reduces Garak prompt-injection hit rate by N%
> and jailbreak hit rate by M% on a fresh Garak scan run 2026-MM-DD with
> garak vX.Y.Z."

With dates and Garak versions on the label. Not "we pass HackAPrompt" —
that benchmark may already be in our training data. Garak's probes are
not a benchmark dataset, they're an attack generator that can be rerun.

## Risks we should record honestly

- **Garak is maintained by NVIDIA, but it is still a research tool.**
  Probe library grows and evolves; today's baseline isn't tomorrow's. A
  scheduled rerun (Experiment C) is how we catch that.
- **Probe sets have their own biases.** Garak's probe library weights
  toward attacks on hosted commercial models; Carnot's verify-repair
  flow may simply not be a relevant target for some probes. This will
  show up as "probe reports a failure that isn't a failure for us" —
  needs manual review of hits, not blind trust.
- **LLM-as-judge scoring is imperfect.** Garak uses Detectors to decide
  whether an attack landed. Some detectors are string-match (reliable),
  some are LLM-judge (noisy). Record which detector produced each hit;
  disagreements between detectors are a useful signal.
- **Infinite recursion risk.** If Carnot's pipeline calls an external LLM
  via OpenAI API to verify arithmetic, and Garak injects a probe that
  rewrites the arithmetic question to be an injection, we end up running
  injections against the verifier's inner LLM too. Garak will flag this
  as a success; it is a real failure mode worth knowing about.
