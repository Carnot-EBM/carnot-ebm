Run the evaluator agent to assess the quality of recent changes.

Read .claude/agents/evaluator.md for instructions, then:

1. Check `git diff HEAD~1..HEAD` to see what changed
2. For each changed file, evaluate against the checklist:
   - Spec conformance
   - Test quality
   - Code quality
   - Cross-language consistency
   - Architectural alignment
3. Report verdict: APPROVE / REQUEST_CHANGES / REJECT
