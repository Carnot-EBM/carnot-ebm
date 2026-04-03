Run the spec reconciler agent to sync all specs with the current codebase state.

Read .claude/agents/spec-reconciler.md for instructions, then:

1. For each openspec/capabilities/*/spec.md, grep codebase for each REQ-* to determine actual status
2. Update Implementation Status tables to match reality
3. Update _bmad/traceability.md to match
4. Update ops/status.md if features changed
5. Run python scripts/check_spec_coverage.py to verify
6. Report what was changed
