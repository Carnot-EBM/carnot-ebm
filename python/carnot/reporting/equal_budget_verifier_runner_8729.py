"""Bind the shipped lifecycle once so historical validation controllers stay unchanged."""

from carnot.reporting import equal_budget_verifier_8729 as e
from carnot.reporting import verification_methods_runner_8726 as historical

runner = e.binding.namespace(historical, "equal_budget_runner")
runner.e = e
runner.shared.e = e
runner.shared.CONSUMERS = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py::test_cli_private_success_failure_and_replay",
    "tests/python/test_restricted_decision_audit_8210.py::test_private_cli_and_rehashed_mutations",
]
main, run, replay, shared = runner.main, runner.run, runner.replay, runner.shared
