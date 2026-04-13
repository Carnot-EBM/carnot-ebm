# Epic: VERIFY-031 - Package Code Verification For End Users

**Status:** Complete
**Goal:** Package Carnot's strongest live code-verification path as a usable
CLI command, MCP tool, and Python API with end-user docs and an end-to-end
generate-verify-repair demonstration.
**Rationale:** The Hypothesis-backed verifier is effective in research and live
benchmark code paths, but end users still have to know which internal module to
call. This story packages that capability without touching
`scripts/research_conductor.py`.

## Stories
- [x] Add `REQ-CODE-019` through `REQ-CODE-022` and
  `SCENARIO-CODE-016` through `SCENARIO-CODE-019` to the
  `code-verification` spec before implementation changes
- [x] Write tests first for the Python API export, the `verify-code` CLI
  command, the `verify_code_with_pbt` MCP tool, the end-to-end
  generate-verify-repair workflow, and the docs examples
- [x] Implement the packaged verification surfaces and docs
- [x] Run the required Python suite, 100% coverage checks for new code,
  spec-coverage, lint/type checks, and the applicable end-to-end validation
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
