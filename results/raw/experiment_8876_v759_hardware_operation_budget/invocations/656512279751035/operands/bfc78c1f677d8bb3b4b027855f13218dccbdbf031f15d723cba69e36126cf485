"""REQ-REPORT-8864: bind existing checks without changing historical validators."""

from carnot.reporting import hardware_budget_8864 as m
from carnot.reporting import hardware_operation_runner_8825 as shared
from carnot.reporting.v748_collection_contract import namespace

checker = namespace(shared, "hardware8864_validation")
checker.m = m
qualify = checker.validate
terminal = checker.terminal
