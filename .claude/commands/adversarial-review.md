Run the adversarial reviewer agent to find gaps between what was requested and what was delivered.

Read .claude/agents/adversarial-reviewer.md for instructions, then:

1. Identify the user's original request for the most recent change
2. Read `git diff HEAD~1..HEAD` to see what was actually delivered
3. Verify every CLAUDE.md mandatory step was followed (especially E2E testing)
4. Check every applicable spec constraint was satisfied
5. Look for silent omissions, stubs, buried caveats, or scope reductions
6. Verify numerical correctness for any math/science code
7. Report verdict: APPROVE / GAPS_FOUND / REJECT
