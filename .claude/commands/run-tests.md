Run the test runner agent to execute all test suites.

Read .claude/agents/test-runner.md for instructions, then:

1. Run Rust tests: `cargo test --workspace --exclude carnot-python`
2. Run Python tests: `source .venv/bin/activate && PYTHONPATH=python:$PYTHONPATH pytest tests/python -v --cov=python/carnot --cov-report=term-missing --cov-fail-under=100`
3. Run spec coverage: `python scripts/check_spec_coverage.py`
4. Report results in the table format specified in the agent definition
