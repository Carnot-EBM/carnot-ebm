"""REQ-REPORT-8921: reuse original boundaries without event observer callbacks."""

import ast
import inspect
from pathlib import Path
import runpy
import sys
import time
import tracemalloc
from types import FunctionType
from typing import Any, Callable, cast

old = runpy.run_path(str(Path(__file__).with_name("buffered_clock_worker_8894.py")))
Json = dict[str, Any]


class InlineStores(ast.NodeTransformer):
    """Keep the original phase injector and replace only its observer calls."""

    def visit_Expr(self, node: ast.Expr) -> Any:
        call = node.value
        if isinstance(call, ast.Call) and ast.unparse(call.func) in ("_phase", "timer.mark"):
            phase = ast.unparse(call.args[0])
            site = ast.unparse(call.args[1]) if len(call.args) > 1 else "0"
            return ast.parse(
                "_i = _inline.cursor\n"
                "_inline.buffer[_i] = _inline.clock()\n"
                f"_inline.buffer[_i + 1] = {phase}\n"
                f"_inline.buffer[_i + 2] = {site}\n"
                "_inline.cursor = _i + 3\n"
            ).body
        return self.generic_visit(node)


def inline_compile(tree: ast.Module, filename: str, mode: str) -> Any:
    """Compile once outside requests; no event dictionaries enter the measured path."""
    return compile(ast.fix_missing_locations(InlineStores().visit(tree)), filename, mode)


def marked(source: str, scope: Json, timer: Any, site: int) -> Callable[..., Any]:
    """Preserve every phase rule in the authenticated injector."""
    fn = FunctionType(
        old["marked"].__code__, dict(old["marked"].__globals__, compile=inline_compile)
    )
    return cast(Callable[..., Any], fn(source, dict(scope, _inline=timer), timer, site))


def evaluator(source: str, arm: str, cap: int, mode: Any, timer: Any) -> Callable[[Any], Json]:
    """Instrumentation defaults off; inline changes only private function bindings."""
    mode = "buffered" if mode is True else "off" if mode is False else mode
    if mode not in ("off", "buffered", "inline"):
        raise ValueError("mode")
    timer.clock = time.perf_counter_ns
    factory = (
        FunctionType(old["evaluator"].__code__, dict(old["evaluator"].__globals__, marked=marked))
        if mode == "inline"
        else old["evaluator"]
    )
    invoke = factory(source, arm, cap, mode != "off", timer)
    if mode != "inline":
        return cast(Callable[[Any], Json], invoke)
    tree = ast.parse(inspect.getsource(old["evaluator"]))
    owner = tree.body[0]
    assert isinstance(owner, ast.FunctionDef)
    body = next(n for n in owner.body if isinstance(n, ast.FunctionDef))
    scope = dict(invoke.__globals__, **inspect.getclosurevars(invoke).nonlocals, _inline=timer)
    exec(
        inline_compile(ast.Module(body=[body], type_ignores=[]), "<inline-request>", "exec"), scope
    )
    return cast(Callable[[Any], Json], scope["invoke"])


def allocation_control() -> Json:
    """Measure retained allocations, while disclosing temporary boxed clock integers."""
    timer = old["BufferedClock"](128)
    scope = dict(_inline=timer)
    exec(
        inline_compile(ast.parse("def store():\n    _phase(0)\n"), "<allocation-control>", "exec"),
        scope,
    )
    capacity = len(timer.buffer)
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(32):
        scope["store"]()
    after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    diffs = after.compare_to(before, "lineno")
    return dict(
        event_count=32,
        event_container_growth=len(timer.buffer) - capacity,
        retained_block_delta=sum(d.count_diff for d in diffs),
        retained_byte_delta=sum(d.size_diff for d in diffs),
        traced_current_bytes=current,
        traced_peak_bytes=peak,
        transient_clock_integers="perf_counter_ns returns boxed Python integers; no total-allocation-free claim",
        observer_callbacks_per_event=0,
        scope="allocation_control_outside_elapsed_qualification",
    )


def main() -> int:
    """Use the original worker protocol; each parent supplies exactly one request."""
    fn = FunctionType(old["main"].__code__, dict(old["main"].__globals__, evaluator=evaluator))
    return int(fn())


if __name__ == "__main__":
    raise SystemExit(main())
