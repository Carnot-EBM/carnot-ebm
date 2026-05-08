# Extropic Z1 readiness packet - .121 THRML alignment

Spec refs: REQ-REPORT-065, SCENARIO-REPORT-065.

## Status

- Current status: simulator-only readiness update.
- No Z1, XTR, or TSU hardware access is claimed.
- No Extropic device latency, sample-quality, SDK, firmware, or transcript evidence is claimed.
- Prior packet: `ops/extropic_z1_readiness_packet.md`.

## THRML vendoring alignment

- THRML 0.1.3 is vendored under Apache-2.0.
- Carnot uses the vendored THRML block-Gibbs transition as the simulator reference.
- Vendoring complete: True.
- KL to THRML after vendoring: 0.0.
- This is a software/simulator alignment fact, not a hardware-execution result.

## Candidate warm-start API requirement

- The required policy is candidate warm-start for every verifier request.
- Future THRML or SDK-backed Z1 evaluation must accept the current verifier payload as `{prompt, candidate}`.
- The sampler initialization state must be `bits(candidate)`, not uniform cold-start and not cached state from another prompt.
- Exp 1566 deployment policy: candidate_warm_start.
- Cold-start accuracy drop at K=100: 51.052632 percent.

## Soft-Gibbs Residual relevance

- Soft-Gibbs Residual remains relevant because hard residual rejection can have an empty operational intersection.
- Soft residual implemented: True.
- Soft BRS decay confirmed: True.
- Hard BRS acceptance rate on the contradictory fixture: 0.0.
- A future hardware packet should keep residual conditioning separate from any hardware sampling evidence.

## pre-silicon correction prerequisites

- Detailed-balance drift correction is required before any Z1 claim.
- The explicit prerequisite name is detailed-balance drift correction.
- The correction must account for analog beta drift across die, temperature, and voltage before Carnot compares Z1 samples to the THRML simulator reference.
- The prerequisite is software correction plus validation on synthetic drift before any authenticated Z1 packet can move from readiness to hardware evidence.

## Claim boundary

- Simulator-only evidence can support API readiness, benchmark manifests, and transcript requirements.
- It cannot support Z1, XTR, TSU, device-latency, device-throughput, or sample-quality hardware claims.
- The next unblocker is an authenticated device transcript plus the detailed-balance drift-correction prerequisite above.
