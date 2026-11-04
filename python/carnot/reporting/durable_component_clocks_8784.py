"""REQ-REPORT-8784: direct coarse clocks retain the shipped numerical/store bodies.

Private compiled copies add two clock reads per composite block. No Python
arithmetic primitive or fsync is replaced by an observer wrapper. Nested blocks
are reduced to exclusive spans; native computation and binding transit remain
an explicitly combined boundary cost because Python cannot distinguish them.
"""

from __future__ import annotations

import ast
from collections import defaultdict
import inspect
import textwrap
import time
from typing import Any
from unittest.mock import patch
from contextlib import ExitStack

from carnot.verify import resident_update_cost_8732 as numeric

Json = dict[str, Any]
__all__ = ["numeric", "instrument", "install", "partition"]


def instrument(function: Any, label: str | None, spans: list[Any]) -> Any:
    """Compile a private copy so source arithmetic and durable barriers stay intact."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition = tree.body[0]
    assert isinstance(definition, ast.FunctionDef)
    definition.decorator_list = []
    counter = 0

    def record(name: str, start: str) -> list[ast.stmt]:
        """Inline append avoids a Python callback at every observed boundary."""
        return ast.parse(f"_spans.append(({name!r}, {start}, _clock()))").body

    class Blocks(ast.NodeTransformer):
        """Only storage barriers and returns need extra boundaries within a block."""

        def visit_Return(self, node: ast.Return) -> Any:
            if label is None:
                return node
            return [
                ast.Assign(
                    targets=[ast.Name(id="_result", ctx=ast.Store())],
                    value=node.value if node.value is not None else ast.Constant(None),
                ),
                *record(label, "_block_start"),
                ast.Return(value=ast.Name(id="_result", ctx=ast.Load())),
            ]

        def visit_Assign(self, node: ast.Assign) -> Any:
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"semantic", "checkpoint"}
            ):
                return [
                    *ast.parse("_ack_start = _clock()").body,
                    node,
                    *record("acknowledgment", "_ack_start"),
                ]
            return node

        def visit_Expr(self, node: ast.Expr) -> Any:
            nonlocal counter
            source = ast.unparse(node)
            kind = (
                "file_sync"
                if source.startswith("os.fsync(")
                else "directory_sync"
                if source.startswith("sync_directory(")
                else "acknowledgment"
                if source == "hook('acknowledgment')"
                else None
            )
            if kind is None:
                return node
            counter += 1
            start = f"_barrier_{counter}"
            return [*ast.parse(f"{start} = _clock()").body, node, *record(kind, start)]

    definition = Blocks().visit(definition)
    if label is not None:
        definition.body.insert(1, ast.parse("_block_start = _clock()").body[0])
        if not isinstance(definition.body[-1], ast.Return):
            definition.body.extend(record(label, "_block_start"))
    ast.fix_missing_locations(tree)
    scope = dict(function.__globals__, _spans=spans, _clock=time.perf_counter_ns)
    exec(compile(tree, inspect.getsourcefile(function) or "<coarse-clock>", "exec"), scope)
    return scope[function.__name__]


def install(stack: ExitStack, arm: str, spans: list[Any]) -> None:
    """Patch private function copies for this child only, retaining nested globals."""
    resident = numeric.resident
    owners = [
        (numeric.store_module.Store, "commit", "serialization"),
        (numeric.store_module, "atomic_json", "serialization"),
    ]
    arithmetic = (
        numeric.numeric
        if arm == "python"
        else (resident.original if arm == "json_native" else resident.engine)
    )
    owners += [
        (arithmetic, "step", "update_arithmetic"),
        (arithmetic, "probability", "feature_constraint"),
    ]
    if arm == "resident_native":
        owners += [
            (resident, "probability", "feature_constraint"),
            (resident.ResidentFixed, "call", "boundary_transfer"),
        ]
    elif arm == "json_native":
        owners += [(resident.original.NativeFixed, "call", "boundary_transfer")]
    owners += [
        (numeric.numeric if arm == "python" else resident.original, "serialize", "serialization")
    ]
    # Compile after dependencies are patched so each copy sees the same nested bodies.
    for owner, name, label in reversed(owners):
        stack.enter_context(
            patch.object(owner, name, instrument(getattr(owner, name), label, spans))
        )
    stack.enter_context(patch.object(numeric, "drive", instrument(numeric.drive, None, spans)))


def partition(spans: list[Any], begin: int, end: int) -> Json:
    """Subtract enclosed leaves and retain measured gaps instead of inventing costs."""
    if end <= begin or any(a < begin or b > end or b < a for _, a, b in spans):
        raise ValueError("clock_span")
    ordered = sorted(spans, key=lambda s: (s[1], -s[2]))
    active: list[Any] = []
    costs: dict[str, int] = defaultdict(int)
    covered = 0
    for label, start, stop in ordered:
        while active and start >= active[-1][2]:
            active.pop()
        if active:
            if stop > active[-1][2]:
                raise ValueError("clock_overlap")
            costs[active[-1][0]] -= stop - start
        else:
            covered += stop - start
        costs[label] += stop - start
        active.append((label, start, stop))
    return dict(
        costs_ns=dict(costs),
        unassigned_ns=end - begin - covered,
        independently_observed_covered_ns=covered,
        reconciliation_error_fraction=abs(sum(costs.values()) - covered) / (end - begin),
    )
