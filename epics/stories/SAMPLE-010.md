# Epic: SAMPLE-010 - KV260 Hardware Round-Trip Validation Artifact

**Status:** Complete
**Goal:** Add a spec-backed Exp 242 bring-up script that attempts a real KV260
overlay/MMIO round trip, records measured upload/trigger/readback latency when
hardware is present, and emits a blocker artifact with exact missing
dependencies when it is not.
**Rationale:** Exp 228 proved the software-model register contract, but the
next step is to replace "software-model ready" with real board evidence or an
honest blocker artifact. The script must preserve the existing CPU fallback
behavior while making it impossible for blocked hardware to look like a
successful live run.

## Stories
- [x] Add `REQ-SAMPLE-007` and `SCENARIO-SAMPLE-012` through
  `SCENARIO-SAMPLE-014` to the `training-inference` spec before
  implementation changes
- [x] Write tests first for Exp 242 hardware, software-model, and blocked
  branching plus artifact labeling and CLI output
- [x] Implement `scripts/experiment_242_kv260_roundtrip.py`
- [x] Run targeted 100% coverage for the new script plus the required Python
  suite, spec-coverage, lint, type-check, and applicable E2E/integration
  checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
