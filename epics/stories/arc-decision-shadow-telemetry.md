# ARC decision shadow telemetry

**Status:** Complete
**Requirement:** REQ-ARC-WMTE-7465

## Goal

Record live ARC decision inputs and cost. Keep the policy unchanged.

## Scope

- Record candidate actions, induction timing, world-model gates, and supervisor arms.
- Bound every episode and run.
- Keep telemetry off by default.
- Add a pure offline summary reader.
- Prove action and provenance parity with a scripted fake environment.

## Exclusions

- Do not change the scored submission kernel.
- Do not run a real game or model.
- Do not record source, hidden state, future frames, or adapter data.

## Verification

Run the focused telemetry tests and all import-affected ARC tests. Run Ruff,
format, mypy, the ARC orphan lint, and E2E-011. Do not run a real game or model.
