# AGENTS.md

Codex automatically reads this file for repository instructions.

## Required Startup

- Read `CODEX.md` before non-trivial changes or any `.harness/` work.
- This repo uses spec-anchored development (BMAD + OpenSpec): spec first, tests first, implement, verify, then reconcile specs and ops docs.
- Before editing implementation code, ensure the relevant `openspec/capabilities/*/spec.md` exists and contains `REQ-*` requirements.
- Before reporting done, run the relevant unit/lint/spec-coverage commands and the applicable end-to-end checks from `ops/e2e-test-plan.md`.
- Keep `openspec/`, `_bmad/traceability.md`, `ops/status.md`, and `ops/changelog.md` aligned with the code you changed.
