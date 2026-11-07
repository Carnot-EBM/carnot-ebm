"""REQ-REPORT-8851: run existing checks with isolated current ownership."""

from carnot.reporting import hardware_operation_budget_8851 as m
from carnot.reporting import hardware_operation_runner_8825 as shared
from carnot.reporting.v748_collection_contract import namespace

checker = namespace(shared, "hardware8851_validation")
checker.m = m
qualify = checker.validate
terminal = checker.terminal
