Run the lint checker agent on the entire codebase.

Read .claude/agents/lint-checker.md for instructions, then:

1. Run `cargo fmt --all -- --check`
2. Run `cargo clippy --workspace --exclude carnot-python -- -D warnings`
3. Run `ruff check python/ tests/`
4. Run `ruff format --check python/ tests/`
5. Run `mypy python/carnot` (if mypy is installed)
6. Report results in the table format specified in the agent definition
