"""REQ-REPORT-8300: reuse the authenticated Exp8272 frontier without new parsing.

Exp8286's execution failed validation. Its small adapter still authenticates
Exp8272 itself; no Exp8286 result is used as frontier authority.
"""

from types import FunctionType
from pathlib import Path
from typing import Any, cast
from carnot.reporting import arc_outcome_frontier_8286 as original_source

ROOT = original_source.ROOT
FRONTIER = original_source.FRONTIER
PINNED_FRONTIER = original_source.PINNED_FRONTIER
TASK_ID = "exp8300-arc-outcome-frontier"
QUALIFIED_CODE = original_source.QUALIFIED_CODE
qualified = original_source.qualified
authority = original_source.authority
signature = original_source.signature
summarize = original_source.summarize


def inspect(locator: Path, frontier: Path, adapter: Path) -> dict[str, Any]:
    """Retain authenticated parsing and bind current configuration on every call."""
    adapted = FunctionType(original_source.inspect.__code__, vars(original_source) | globals())
    return cast(dict[str, Any], adapted(locator, frontier, adapter))
