Run the spec validator agent to check code-spec alignment.

Read .claude/agents/spec-validator.md for instructions, then:

1. Run `python scripts/check_spec_coverage.py`
2. Read each `openspec/capabilities/*/spec.md` Implementation Status table
3. Verify claimed status matches actual code/test state
4. Check `_bmad/traceability.md` accuracy
5. Report in the format specified in the agent definition
