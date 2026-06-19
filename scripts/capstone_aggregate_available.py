"""Compatibility import for the capstone aggregate-available helper.

Spec refs: REQ-REPORT-4449, SCENARIO-REPORT-4449.
"""

from carnot.reporting.capstone_aggregate_available import AxisSpec, aggregate_available_report_gaps

__all__ = ["AxisSpec", "aggregate_available_report_gaps"]
