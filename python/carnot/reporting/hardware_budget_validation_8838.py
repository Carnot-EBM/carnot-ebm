"""REQ-VERIFY-8838: reuse the scoped validator without changing historical code."""

from carnot.reporting import hardware_operation_budget_8838 as m
from carnot.reporting import hardware_operation_runner_8825 as shared
from carnot.reporting.v748_collection_contract import namespace

# The isolated function globals keep the old producer's scope unchanged.
checker = namespace(shared, "hardware8838_validation")
checker.m = m
qualify = checker.validate
terminal = checker.terminal
