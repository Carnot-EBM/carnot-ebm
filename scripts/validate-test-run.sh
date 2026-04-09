#!/bin/bash
# PostToolUse:Bash hook — validate test coverage after test runs.
# Only checks coverage when the Bash command was actually a pytest run.
# For all other commands, exits silently with success.

# The hook receives the tool output via environment or stdin.
# We check if the output contains pytest coverage results.
# If it does and coverage is below 100%, we warn.

# For now, this is a no-op passthrough — the coverage threshold is
# enforced by pytest's --cov-fail-under=100 flag in the test command.
exit 0
