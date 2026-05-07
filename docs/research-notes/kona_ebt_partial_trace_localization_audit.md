# Kona/EBT Partial-Trace Localization Audit

Run date: 20260507

## Scope

This note documents Exp 1490, a bounded local audit of partial-trace failure
localization. The audit uses existing Carnot telemetry from Exp 1480 and the
Exp 1450 EBT/NRGPT smoke-audit boundary. It does not use Kona internals, does
not call a Kona service, and does not generate new headline LLM samples.

## Method

The audit selected clean, format-valid Exp 1480 telemetry rows with token
logprobs, top-k alternatives, expected answers, and deterministic adversarial
wrong answers. For each row, it injected the wrong answer into the final answer
span and ranked that injected span against clean spans.

The local score is deliberately simple:

- Clean spans use observed token surprisal from the existing trace.
- The injected answer span uses the top-k energy of the wrong alternative.
- A verifier mismatch penalty is added because the injected answer no longer
  matches the known expected answer.

The audit reports top-1 and top-3 localization rates, a random baseline derived
from the number of candidate spans per trace, and a superficial span-length
baseline.

## Boundary

This is an injected-failure diagnostic, not a decoded-quality result. A high
localization rate means the available local trace features can point at a known
synthetic bad span in this bounded setup. It does not show that Carnot can infer
natural model reasoning failures in arbitrary traces, does not establish Kona
parity, and does not justify a decoded answer quality claim.

The result artifact sets `decoded_quality_claim_allowed=false` and
`kona_dependency_used=false` for that reason.
