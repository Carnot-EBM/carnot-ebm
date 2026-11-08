"""REQ-REPORT-8882: research-only exclusive clocks around the retained installed consumer."""

import time

ENTRY_NS = time.monotonic_ns()

import ast
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Callable, Iterator
from unittest.mock import patch

from carnot.verify import _indexed_witness_core as core
from carnot.verify import _indexed_witness_meter as meter
from carnot.verify import size_aware_witness as size_aware

HISTORICAL = runpy.run_path(str(Path(__file__).with_name("size_aware_cost_worker_8874.py")))
Json = dict[str, Any]


class ExclusiveClock:
    """Pause the enclosing category while a nested category owns elapsed time."""

    def __init__(self, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.clock = clock
        self.stack: list[str] = []
        self.totals: dict[str, int] = {}
        self.intervals: list[Json] = []
        self.last = 0

    def tick(self) -> None:
        now = self.clock()
        if self.stack:
            key = self.stack[-1]
            self.totals[key] = self.totals.get(key, 0) + now - self.last
            self.intervals.append(dict(component=key, start_ns=self.last, end_ns=now))
        self.last = now

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        self.tick()
        self.stack.append(name)
        try:
            yield
        finally:
            self.tick()
            self.stack.pop()

    def wrap(self, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        def measured(*args: Any, **kwargs: Any) -> Any:
            with self.span(name):
                return fn(*args, **kwargs)

        return measured


def compile_baseline(source: str) -> Callable[..., Any]:
    """Keep original syntax available for a separate instrumented function binding."""
    baseline: Callable[..., Any] = HISTORICAL["compile_baseline"](source)
    baseline.__globals__["source8882"] = source
    return baseline


def profiled_baseline(baseline: Callable[..., Any], clock: ExclusiveClock) -> Callable[..., Any]:
    """Bracket scan preparation without changing any original expression or charge."""
    tree = ast.parse(baseline.__globals__["source8882"])
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    boundary = next(
        i
        for i, node in enumerate(fn.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "possible"
    )
    bracket = ast.parse("with clock8882.span('preparation'):\n    pass").body[0]
    assert isinstance(bracket, ast.With)
    bracket.body = fn.body[2:boundary]
    fn.body[2:boundary] = [bracket]
    scope = dict(baseline.__globals__, clock8882=clock)
    exec(
        compile(ast.fix_missing_locations(tree), "<instrumented-authenticated-scan>", "exec"), scope
    )
    return clock.wrap(scope["strong"], "exact_joins")


def evaluate(
    encoded: str, arm: str, baseline: Callable[..., Any], cap: int, instrumented: bool
) -> Json:
    """Keep observers opt-in and preserve cap exhaustion and strict input failures."""
    timer = ExclusiveClock()
    with ExitStack() as stack:
        if instrumented:
            for owner, attr, category in [
                (meter, "parse", "parse_validate"),
                (baseline.__globals__["original"], "parse", "parse_validate"),
                (core, "build_index", "preparation"),
                (core, "solve", "exact_joins"),
                (json, "loads", "parse_validate"),
                (json, "dumps", "response_encode"),
            ]:
                stack.enter_context(
                    patch.object(owner, attr, timer.wrap(getattr(owner, attr), category))
                )
            direct = size_aware._DIRECT_SOLVE
            stack.enter_context(
                patch.dict(
                    direct.__globals__,
                    build_index=timer.wrap(size_aware.build_direct, "preparation"),
                )
            )
            stack.enter_context(
                patch.object(size_aware, "_DIRECT_SOLVE", timer.wrap(direct, "exact_joins"))
            )
            baseline = profiled_baseline(baseline, timer)
        start = time.monotonic_ns()
        result: Json = HISTORICAL["attempt"](encoded, arm, baseline, cap)
        response = json.dumps(
            dict(decision=result["decision"], witness=result["witness"]), sort_keys=True
        )
        response_bytes = len(response.encode())
        del response
        end = time.monotonic_ns()
    return dict(
        result,
        components_ns=timer.totals,
        component_intervals=timer.intervals,
        request_elapsed_ns=end - start,
        response_bytes=response_bytes,
    )


def main() -> int:
    """Execute frozen installed requests; the parent retains spawn-through-exit clocks."""
    baseline = compile_baseline(Path(sys.argv[1]).read_text())
    request = json.load(sys.stdin)
    startup = time.monotonic_ns() - ENTRY_NS
    noop = []
    for _ in range(100):
        start = time.monotonic_ns()
        noop.append(time.monotonic_ns() - start)
    rows = []
    for world in request["worlds"]:
        for repeat in request["repeats"]:
            for arm in world["arm_order"]:
                for unit in world["requests"]:
                    start = time.monotonic_ns()
                    encoded = json.dumps(unit["input"], sort_keys=True)
                    result = evaluate(encoded, arm, baseline, request["cap"], world["instrumented"])
                    del encoded
                    end = time.monotonic_ns()
                    rows.append(
                        dict(
                            result,
                            unit_id=unit["unit_id"],
                            cluster_id=world["cluster_id"],
                            arm=arm,
                            repeat=repeat,
                            size=world["size"],
                            population=world["population"],
                            seed=request["seed"],
                            denominator=1,
                            metric=int(result["decision"] != "insufficient"),
                            instrumented=world["instrumented"],
                            started_monotonic_ns=start,
                            ended_monotonic_ns=end,
                            complete_wall_time_s=(end - start) / 1e9,
                            startup_import_ns=0,
                        )
                    )
        print(
            f"[exp8882-worker] completed={len(rows)} pending=0 cluster={world['cluster_id']}",
            file=sys.stderr,
            flush=True,
        )
    for row in rows:
        row["startup_import_ns"] = startup / len(rows)
    rows[0]["noop_clock_ns"] = noop
    imports = []
    for name in (
        "_indexed_witness_core",
        "_indexed_witness_meter",
        "indexed_witness",
        "size_aware_witness",
    ):
        path = Path(sys.modules["carnot.verify." + name].__file__ or "")
        imports.append(
            dict(path=str(path), sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
        )
    json.dump(
        dict(rows=rows, installed_imports=imports, noop_clock_ns=noop, startup_import_ns=startup),
        sys.stdout,
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
