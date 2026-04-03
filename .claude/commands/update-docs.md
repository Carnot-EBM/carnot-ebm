Run the documentation keeper agent to sync all documentation with current codebase state.

Read .claude/agents/docs-keeper.md for instructions, then:

1. Check what changed: `git diff HEAD~1..HEAD --stat` (or `git diff --cached --stat` if uncommitted changes exist)
2. For each affected area, read the source and update the corresponding docs
3. Update README.md, CLAUDE.md, _bmad/, openspec/ specs, and ops/ as needed
4. Report what was updated
