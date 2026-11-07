"""REQ-REPORT-8894: opt-in flat clocks around unchanged installed expressions."""

from array import array
import ast
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import time
from types import FunctionType, SimpleNamespace
from typing import Any, Callable, cast

from carnot.verify import _indexed_witness_core as core
from carnot.verify import indexed_witness as indexed
from carnot.verify import size_aware_witness as size_aware

OLD = runpy.run_path(str(Path(__file__).with_name("size_aware_cost_worker_8874.py")))
Json = dict[str, Any]


class BufferedClock:
    """Store integer endpoints in reserved memory; reduction happens after the block."""

    def __init__(self, capacity: int, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.buffer = array("q", [0]) * (capacity * 3)
        self.cursor = 0
        self.clock = clock

    def mark(self, phase: int, site: int = 0) -> None:
        """Change the active exclusive phase without logging or allocating a record."""
        i = self.cursor
        if i + 3 > len(self.buffer):
            raise ValueError("buffer_capacity")
        self.buffer[i] = self.clock()
        self.buffer[i + 1] = phase
        self.buffer[i + 2] = site
        self.cursor = i + 3

    def events(self, start: int, end: int) -> list[list[int]]:
        """Materialize raw triples after the timed block has ended."""
        return [list(self.buffer[i : i + 3]) for i in range(start, end, 3)]


def marked(source: str, scope: Json, timer: BufferedClock, site: int) -> Callable[..., Any]:
    """Insert flat phase transitions once, before resident requests execute."""
    tree = ast.parse(source)
    rules = {
        "original.parse(": 1,
        "signs:": 2,
        "possible = False": 3,
        "meter.parse(": 1,
        "allocate('index_allocation')": 2,
        "allocate('binding_copy')": 3,
        "meter.charge('result_serialization')": 4,
        "counter = meter.Counter(": 0,
        "direct = ": 0,
        "for _ in encoded.encode(": 1,
        "payload = json.loads(": 1,
        "counter.charge('request_dispatch')": 0,
        "validation_counts = ": 1,
        "for _ in response.encode(": 4,
        "del index, signs": 0,
    }

    def statements(nodes: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for node in nodes:
            text = ast.unparse(node)
            phase = next((v for k, v in rules.items() if text.startswith(k)), None)
            serialization = (
                isinstance(node, (ast.Assign, ast.Expr))
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "json.dumps"
            )
            if serialization:
                result.extend(ast.parse(f"_phase(4, {site})").body)
            elif phase is not None:
                result.extend(ast.parse(f"_phase({phase})").body)
            if isinstance(node, ast.Try):
                node.body = statements(node.body)
                node.finalbody = ast.parse("_phase(0)").body + statements(node.finalbody)
                for handler in node.handlers:
                    handler.body = ast.parse("_phase(0)").body + statements(handler.body)
            elif isinstance(node, ast.If):
                node.body, node.orelse = statements(node.body), statements(node.orelse)
            result.append(node)
            if serialization:
                result.extend(ast.parse(f"_phase(0, {-site})").body)
        return result

    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    fn.body = statements(fn.body)
    environment = dict(scope, _phase=timer.mark)
    exec(compile(ast.fix_missing_locations(tree), "<buffered-installed-copy>", "exec"), environment)
    return cast(Callable[..., Any], environment[fn.name])


def evaluator(
    source: str, arm: str, cap: int, observed: bool, timer: BufferedClock
) -> Callable[[Any], Json]:
    """Reuse the original runner, with private compiled bindings only when requested."""
    if arm not in ("scan", "indexed", "size_aware"):
        raise ValueError("arm")
    baseline = OLD["compile_baseline"](source)
    attempt = OLD["attempt"]
    if observed:
        solve_source = inspect.getsource(core.solve)
        solve = marked(solve_source, vars(core), timer, 3)
        direct = marked(solve_source, size_aware._DIRECT_SOLVE.__globals__, timer, 3)
        proxy = SimpleNamespace(**dict(vars(core), solve=solve))
        owner = size_aware if arm == "size_aware" else indexed
        verify = (
            size_aware.verify_size_aware_witness
            if arm == "size_aware"
            else indexed.verify_indexed_witness
        )
        public = marked(
            inspect.getsource(verify), dict(vars(owner), core=proxy, _DIRECT_SOLVE=direct), timer, 0
        )
        baseline = marked(source, baseline.__globals__, timer, 0)
        original = OLD["HISTORICAL"]["attempt"]
        measured = marked(
            inspect.getsource(original),
            dict(
                original.__globals__,
                ARMS=("scan", arm if arm != "scan" else "indexed"),
                verify_indexed_witness=public,
            ),
            timer,
            2,
        )
        scope = dict(
            OLD["attempt"].__globals__,
            HISTORICAL=dict(OLD["HISTORICAL"], attempt=measured),
            verify_size_aware_witness=public,
        )
        attempt = FunctionType(OLD["attempt"].__code__, scope)

    def invoke(payload: Any) -> Json:
        if observed:
            timer.mark(4, 1)
        encoded = json.dumps(payload, sort_keys=True)
        if observed:
            timer.mark(0, -1)
        result: Json = attempt(encoded, arm, baseline, cap)
        if observed:
            timer.mark(4, 4)
        response = json.dumps(
            dict(decision=result["decision"], witness=result["witness"]), sort_keys=True
        )
        response_bytes = len(response.encode())
        del response, encoded
        if observed:
            timer.mark(0, -4)
            timer.mark(5)
        return dict(result, response_bytes=response_bytes)

    return invoke


def evaluate(payload: Any, arm: str, source: str, cap: int, observed: bool) -> Json:
    """Expose a small real-clock control without including compilation in phase sums."""
    timer = BufferedClock(128)
    invoke = evaluator(source, arm, cap, observed, timer)
    start = time.monotonic_ns()
    result = invoke(payload)
    end = time.monotonic_ns()
    events = timer.events(0, timer.cursor)
    return dict(
        result,
        endpoints=events,
        start_ns=events[0][0] if events else start,
        end_ns=events[-1][0] if events else end,
    )


def main() -> int:
    """Buffer an entire request block and emit output only after its end clock."""
    source = Path(sys.argv[1]).read_text()
    request = json.load(sys.stdin)
    rows = []
    for world in request["worlds"]:
        count = len(world["requests"]) * len(world["arm_order"]) * len(request["repeats"])
        timer = BufferedClock(128 * (count + 1))
        functions = {
            arm: evaluator(source, arm, request["cap"], world["instrumented"], timer)
            for arm in world["arm_order"]
        }
        print(
            "[exp8894-worker] phase=before_warmup completed=0 pending=" + str(count),
            file=sys.stderr,
            flush=True,
        )
        for arm in world["arm_order"] if world.get("warmup", True) else []:
            functions[arm](world["requests"][0]["input"])
            timer.cursor = 0
        print(
            "[exp8894-worker] phase=before_block completed=0 pending=" + str(count),
            file=sys.stderr,
            flush=True,
        )
        buffered: list[Any] = [None] * count
        i = 0
        block_start = time.monotonic_ns()
        for repeat in request["repeats"]:
            for arm in world["arm_order"]:
                for unit in world["requests"]:
                    offset, start = timer.cursor, time.monotonic_ns()
                    result = functions[arm](unit["input"])
                    end = time.monotonic_ns()
                    buffered[i] = (
                        unit["unit_id"],
                        arm,
                        repeat,
                        result,
                        start,
                        end,
                        offset,
                        timer.cursor,
                    )
                    i += 1
        block_end = time.monotonic_ns()
        print(
            f"[exp8894-worker] phase=after_block completed={count} pending=0",
            file=sys.stderr,
            flush=True,
        )
        for unit_id, arm, repeat, result, start, end, offset, stop in buffered:
            events = timer.events(offset, stop)
            rows.append(
                dict(
                    result,
                    unit_id=unit_id,
                    cluster_id=world["cluster_id"],
                    arm=arm,
                    repeat=repeat,
                    size=world["size"],
                    population=world["population"],
                    seed=request["seed"],
                    metric=int(result["decision"] != "insufficient"),
                    denominator=1,
                    instrumented=world["instrumented"],
                    endpoints=events,
                    start_ns=events[0][0] if events else start,
                    end_ns=events[-1][0] if events else end,
                    block_start_ns=block_start,
                    block_end_ns=block_end,
                    block_elapsed_ns=block_end - block_start,
                    complete_wall_time_s=(end - start) / 1e9,
                )
            )
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
    json.dump(dict(rows=rows, installed_imports=imports), sys.stdout)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
