"""REQ-VERIFY-8818: charge explicit work without replacing existing solver rules."""

import ast
from contextvars import ContextVar
from functools import lru_cache
import inspect
import time
from types import SimpleNamespace
from typing import Any, Callable

from carnot.reporting import evidence_intervention_methods_8791 as old
from carnot.reporting import evidence_set_methods_8768 as original
from carnot.reporting import evidence_qualification_8777 as qualification
from carnot.reporting import v749_verifier_protocol as numerical
from carnot.reporting.verifier_family_surface_8755 import verifier as historical_verifier

Json = dict[str, Any]
KINDS = (
    "fact_comparisons",
    "candidate_bindings",
    "join_attempts",
    "record_visits",
    "query_visits",
    "other_iterations",
    "lookup_attempts",
    "other_comparisons",
)
CURRENT: ContextVar["Counter"] = ContextVar("exp8818_counter")


class Exhausted(Exception):
    """Stop before unpaid work so a partial proof cannot become a decision."""


class Counter:
    """The same explicit operations cost one unit in every unchanged method."""

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.counts = dict.fromkeys(KINDS, 0)
        self.total = 0

    def charge(self, kind: str) -> None:
        """Check the ceiling before a primitive executes, including inside comprehensions."""
        if self.limit is not None and self.total >= self.limit:
            raise Exhausted
        self.total += 1
        self.counts[kind] += 1


def charge(kind: str) -> bool:
    """Return true so a comprehension charge precedes its original filters."""
    CURRENT.get().charge(kind)
    return True


def charged(node: ast.expr, kind: str) -> ast.expr:
    """Tuple evaluation adds a charge while retaining the expression's exact value."""
    call = ast.Call(
        func=ast.Name(id="_charge", ctx=ast.Load()), args=[ast.Constant(kind)], keywords=[]
    )
    return ast.Subscript(
        value=ast.Tuple(elts=[call, node], ctx=ast.Load()), slice=ast.Constant(1), ctx=ast.Load()
    )


class Instrument(ast.NodeTransformer):
    """Instrument operations, never source lines or labels, in frozen function syntax."""

    @staticmethod
    def iteration(target: ast.expr, iterable: ast.expr) -> str:
        """Name assignment and binding work separately from input and query visits."""
        name, source = ast.unparse(target), ast.unparse(iterable)
        if name in {"assignment", "values"}:
            return "candidate_bindings"
        if name == "binding":
            return "join_attempts"
        if "facts" in source or name in {"fact", "f", "(s, r, o, sign)"}:
            return "record_visits"
        if "query" in source or name in {"q", "predicate"}:
            return "query_visits"
        return "other_iterations"

    def visit_For(self, node: ast.For) -> ast.AST:
        """For loops pay before entering their original body."""
        self.generic_visit(node)
        node.body.insert(
            0,
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="_charge", ctx=ast.Load()),
                    args=[ast.Constant(self.iteration(node.target, node.iter))],
                    keywords=[],
                )
            ),
        )
        return node

    def visit_comprehension(self, node: ast.comprehension) -> ast.AST:
        """Rejected candidates also cost work, so charge before all original filters."""
        self.generic_visit(node)
        node.ifs.insert(
            0,
            ast.Call(
                func=ast.Name(id="_charge", ctx=ast.Load()),
                args=[ast.Constant(self.iteration(node.target, node.iter))],
                keywords=[],
            ),
        )
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """Fact and binding comparisons include sign checks; protocol guards stay separate."""
        self.generic_visit(node)
        names = {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}
        kind = (
            "fact_comparisons"
            if names & {"f", "fact", "term", "actual", "v", "values"}
            else "other_comparisons"
        )
        return ast.copy_location(charged(node, kind), node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Charge hash lookups too, so indexed reference work is never free."""
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"get", "setdefault"}:
            return ast.copy_location(charged(node, "lookup_attempts"), node)
        return node


def instrument(function: Callable[..., Any], **overrides: Any) -> Callable[..., Any]:
    """Compile only the original function with added charges and isolated dependencies."""
    tree = Instrument().visit(ast.parse(inspect.getsource(function)))
    ast.fix_missing_locations(tree)
    scope = dict(function.__globals__, _charge=charge, **overrides)
    exec(compile(tree, "<exp8818-metered-" + function.__name__ + ">", "exec"), scope)
    result: Callable[..., Any] = scope[function.__name__]
    return result


@lru_cache(maxsize=1)
def freeze() -> Callable[..., Any]:
    """Clone dependencies into private namespaces so historical consumers stay unchanged."""
    truth = instrument(numerical.truth)
    verifier = instrument(
        historical_verifier,
        numerical=SimpleNamespace(VARIANTS=numerical.VARIANTS, truth=truth),
    )
    compare = instrument(
        original.compare,
        decide=instrument(original.decide),
        historical=SimpleNamespace(truth=truth),
        historical_verifier=verifier,
    )
    bounded = instrument(
        qualification.bounded, original=SimpleNamespace(parse=original.parse, compare=compare)
    )
    return instrument(
        old.evaluate, strong=instrument(old.strong), qualification=SimpleNamespace(bounded=bounded)
    )


def run(arm: str, payload: Json, limit: int | None = 200000) -> Json:
    """An exhausted attempt abstains and retains its actual paid work and elapsed cost."""
    original.parse(payload)
    counter = Counter(limit)
    token = CURRENT.set(counter)
    start, cpu = time.monotonic_ns(), time.process_time_ns()
    exhausted = False
    try:
        decision, calls = freeze()(arm, payload)
    except Exhausted:
        decision, calls, exhausted = "insufficient", None, True
    finally:
        CURRENT.reset(token)
    end = time.monotonic_ns()
    return dict(
        decision=decision,
        evaluator_invocations=calls,
        primitive_limit=limit,
        primitive_operations=counter.total,
        operation_counts=counter.counts,
        budget_exhausted=exhausted,
        wall_time_s=(end - start) / 1e9,
        cpu_time_s=(time.process_time_ns() - cpu) / 1e9,
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
