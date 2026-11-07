"""REQ-REPORT-8876: bind unchanged unit, child, lint and terminal validators."""

from carnot.reporting import hardware_operation_budget_8876 as m
from carnot.reporting import hardware_operation_runner_8825 as shared
from carnot.reporting.v748_collection_contract import namespace

checker = namespace(shared, "hardware8876_validation")
checker.m = m
qualify = checker.validate
terminal = checker.terminal
