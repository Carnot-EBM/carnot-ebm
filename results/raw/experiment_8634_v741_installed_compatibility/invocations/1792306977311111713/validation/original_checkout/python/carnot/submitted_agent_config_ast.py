"""Read ``SUBMITTED_AGENT_CONFIG`` out of ``arc_competition_agent.py`` WITHOUT importing it.

WHY THIS EXISTS AT ALL (read this before "simplifying" it into a plain import).
The integration-gate experiments need to know exactly which configuration the *submitted*
ARC agent ships with -- which flags are on, what the budgets are -- so they can assert the
gate they measured corresponds to the agent that would actually run. The obvious way to get
that is ``from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG``.

That obvious way is too expensive here. ``arc_competition_agent`` is the live scored agent's
entrypoint: importing it drags in the whole agentic stack (the world-model inducer, the
generator-proposer plumbing, and transitively the optional ``llama_cpp`` binding). A gate
experiment that only wants to read ~90 booleans and ints should not pay that, and should not
fail merely because an optional inference dependency is absent on the machine. So instead we
parse the module's source with ``ast`` and evaluate just the constant assignments. See
``experiment_4754_submitted_agent_config.py`` for the other tradeoff -- it *does* do the real
import, and marks that function an explicit "import boundary".

WHAT WENT WRONG, AND WHY THE FIX IS SHAPED THIS WAY (the bug this module was extracted to fix).
The original reader lived duplicated inside two experiment modules and evaluated each
module-level assignment with ``ast.literal_eval``. ``literal_eval`` only accepts *literals*.
So the moment a constant was defined by referring to something else --

    from carnot.agentic.arc_frontier_discipline import TIER_COUNT as FRONTIER_TIER_COUNT
    ...
    SUBMITTED_FRONTIER_TIER_COUNT = FRONTIER_TIER_COUNT     # <-- a Name, not a literal

-- ``literal_eval`` raised, the reader's ``except: continue`` swallowed it, and the constant
was silently never recorded. Nothing complained at that point. The failure surfaced much later
and much less legibly, as a bare ``KeyError: 'SUBMITTED_FRONTIER_TIER_COUNT'`` thrown from the
line that looked the value up while building the config dict.

Two properties of that failure are worth keeping in mind, because they are what this module is
designed to prevent recurring:

1. It is SILENT AT THE POINT OF CAUSE and LOUD SOMEWHERE ELSE. The swallowed exception is the
   real defect; the KeyError is just where the consequence became visible. Anyone debugging
   from the traceback starts in the wrong place.
2. It is LATENT PER-NAME. Refactoring any ``SUBMITTED_*`` constant from a literal into a
   reference -- an entirely reasonable, invisible-looking edit in the agent module -- breaks
   this reader. At the time of the fix there were already TWO such names
   (``SUBMITTED_FRONTIER_TIER_COUNT`` and ``GOAL_ENERGY_SOURCE``); the KeyError only ever
   named the first, so fixing just the name in the message would have left the second waiting.

Hence: resolve references properly rather than skipping them, and when a reference genuinely
cannot be resolved, say so in terms of the name and the reason instead of failing later.

WHAT WE RESOLVE:
  * plain literals                     -- ``ast.literal_eval``, as before.
  * a reference to an earlier constant -- ``A = 5`` then ``B = A``: chased through the
    constants we have already evaluated, in source order (which is also Python's own order,
    so a forward reference is correctly *not* resolvable).
  * a reference to a ``from X import Y [as Z]`` alias -- resolved by importing X and reading Y
    off it. This is what makes ``SUBMITTED_FRONTIER_TIER_COUNT = FRONTIER_TIER_COUNT`` work.
    Note this keeps the cost property we came here for: the modules that supply these pins
    (``arc_frontier_discipline``, ``arc_goal_energy_live``, ``arc_executable_world_model``) are
    leaf constant-holders, so importing one of *them* does not pull in the agent stack we were
    avoiding.

  * containers of any of the above, at any depth -- ``"frozen_generator"`` is a nested dict
    whose values are themselves ``ARC_LIVE_GENERATOR_*`` aliases.
  * a bounded set of PURE expressions over resolved values: comparisons (``X != "0"``),
    ``and``/``or``/``not``, and the conditional (``"draft-mtp" if X != "0" else None``). The
    config genuinely contains these -- flags derived from a pinned string rather than written
    as a bare bool.

WHAT WE DELIBERATELY DO NOT RESOLVE: anything that would require *running* code -- a call, an
attribute lookup, arithmetic, a subscript, a comprehension, an f-string. Evaluating those means
executing module code, the very cost this module exists to avoid. Such a value raises
:class:`UnresolvableConfigValue` naming the key, so it is a visible, actionable failure rather
than a silently-dropped entry.

ON THE SIZE OF THAT BOUNDED SET: it was chosen by *measuring* the config, not by guessing. The
complete inventory of AST node types appearing in ``SUBMITTED_AGENT_CONFIG``'s values at the
time of writing was: ``Constant``, ``Name``, ``List``, ``Dict``, ``Compare`` (``NotEq`` only),
and one ``IfExp``. Everything supported here is side-effect-free by construction -- there is no
generic ``eval`` anywhere in this module, and the comparison operators are an explicit table --
so widening the set slightly (all comparison operators, ``and``/``or``/``not``) costs nothing in
risk while removing the most likely next false failure. Anything genuinely computed still
refuses loudly. If you find yourself tempted to add call or attribute support, that is the
signal to switch this consumer to a real import instead (see
``experiment_4754_submitted_agent_config.py``), not to keep growing an interpreter.
"""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from typing import Any

__all__ = ["UnresolvableConfigValue", "parse_submitted_agent_config"]


class UnresolvableConfigValue(RuntimeError):
    """A ``SUBMITTED_AGENT_CONFIG`` entry could not be resolved without executing code.

    Raised instead of letting a bare ``KeyError`` escape from a dictionary lookup. The whole
    point is that the message names *which* config key failed and *why*, so the reader of the
    traceback lands on the actual cause rather than on the lookup that happened to trip over
    it.
    """


def _import_alias_tables(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Map each ``from X import Y as Z`` local name to the ``(module, attribute)`` it came from.

    Only ``from``-imports with an explicit module are recorded. A bare ``import X`` binds the
    module object itself, which is never a valid config *value* here, and a relative import
    (``from . import Y``) has no absolute module path to hand to ``import_module`` from this
    context -- both are skipped rather than guessed at.
    """
    aliases: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module or node.level:
            continue
        for alias in node.names:
            if alias.name == "*":
                # A star-import gives us no names to bind, and expanding it would require
                # importing the module -- the cost we are avoiding. Skipped knowingly.
                continue
            aliases[alias.asname or alias.name] = (node.module, alias.name)
    return aliases


def _resolve_name(
    name: str,
    constants: dict[str, Any],
    aliases: dict[str, tuple[str, str]],
) -> Any:
    """Resolve a bare name to a value, or raise :class:`UnresolvableConfigValue`.

    Order matters and mirrors Python's own: a module-level assignment seen earlier in the file
    shadows an import of the same name, because that is what would actually be in scope by the
    time ``SUBMITTED_AGENT_CONFIG`` is constructed.
    """
    if name in constants:
        return constants[name]
    if name in aliases:
        module_name, attribute = aliases[name]
        try:
            module = import_module(module_name)
        except ImportError as exc:  # pragma: no cover - only if a pin's module is unimportable.
            raise UnresolvableConfigValue(
                f"{name!r} is imported from {module_name!r}, which could not be imported "
                f"({exc}). The submitted-agent config cannot be read without it."
            ) from exc
        try:
            return getattr(module, attribute)
        except AttributeError as exc:  # pragma: no cover - only on a renamed/removed pin.
            raise UnresolvableConfigValue(
                f"{name!r} is imported as {module_name}.{attribute}, but that module has no "
                f"such attribute. The pin was probably renamed or removed."
            ) from exc
    raise UnresolvableConfigValue(
        f"{name!r} is neither a module-level constant assigned earlier in "
        f"arc_competition_agent.py nor a name imported into it. It may be defined inside a "
        f"function, assigned conditionally, or declared after SUBMITTED_AGENT_CONFIG."
    )


def _evaluate(
    value_node: ast.expr,
    constants: dict[str, Any],
    aliases: dict[str, tuple[str, str]],
) -> Any:
    """Evaluate one right-hand side: a literal, a resolvable name, or a container of those.

    Strategy, in order:

    1. A bare name is resolved via :func:`_resolve_name`.
    2. Otherwise ``ast.literal_eval`` is tried, because it covers the overwhelming majority in
       one cheap call (numbers, strings, ``None``, bools, and containers built purely of them).
    3. If that fails and the node is a *container*, we recurse element-wise. This is the case
       ``literal_eval`` cannot do and that a flat implementation gets wrong: a dict or list
       whose *elements* are names rather than literals. ``SUBMITTED_AGENT_CONFIG`` really does
       contain one -- ``"frozen_generator"`` is a nested dict of ``ARC_LIVE_GENERATOR_*``
       import aliases -- so this is load-bearing, not defensive. Note that recursion, not a
       second flat pass, is what makes arbitrary nesting depth work.
    4. Anything else is a computed expression (a call, an f-string, arithmetic, an attribute
       lookup) and is refused rather than executed.
    """
    if isinstance(value_node, ast.Name):
        return _resolve_name(value_node.id, constants, aliases)

    try:
        return ast.literal_eval(value_node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as literal_exc:
        if isinstance(value_node, ast.Dict):
            resolved: dict[Any, Any] = {}
            for key_node, item_node in zip(value_node.keys, value_node.values):
                if key_node is None:
                    raise UnresolvableConfigValue(
                        "nested dict contains a ** spread, which cannot be expanded without "
                        "importing the module."
                    ) from literal_exc
                resolved[ast.literal_eval(key_node)] = _evaluate(item_node, constants, aliases)
            return resolved
        if isinstance(value_node, (ast.List, ast.Tuple, ast.Set)):
            items = [_evaluate(item, constants, aliases) for item in value_node.elts]
            if isinstance(value_node, ast.Tuple):
                return tuple(items)
            if isinstance(value_node, ast.Set):
                return set(items)
            return items
        if isinstance(value_node, ast.Compare):
            return _evaluate_compare(value_node, constants, aliases)
        if isinstance(value_node, ast.IfExp):
            branch = (
                value_node.body
                if _evaluate(value_node.test, constants, aliases)
                else value_node.orelse
            )
            return _evaluate(branch, constants, aliases)
        if isinstance(value_node, ast.BoolOp):
            values = [_evaluate(v, constants, aliases) for v in value_node.values]
            if isinstance(value_node.op, ast.And):
                result: Any = True
                for candidate in values:
                    result = candidate
                    if not candidate:
                        break
                return result
            result = False
            for candidate in values:
                result = candidate
                if candidate:
                    break
            return result
        if isinstance(value_node, ast.UnaryOp) and isinstance(value_node.op, ast.Not):
            return not _evaluate(value_node.operand, constants, aliases)
        raise UnresolvableConfigValue(
            "value is a computed expression "
            f"({type(value_node).__name__}), which is outside the bounded set this reader "
            "evaluates (literals, names, containers, comparisons, and/or/not, and a "
            "conditional). Evaluating it would mean running module code, which is the cost "
            f"this reader exists to avoid ({literal_exc})."
        ) from literal_exc


# The comparison operators we evaluate. Deliberately an explicit table rather than a generic
# `eval`: every entry is a pure function of already-resolved plain values, so nothing here can
# have a side effect, import anything, or call into the module being read.
_COMPARISONS: dict[type[ast.cmpop], Any] = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _evaluate_compare(
    node: ast.Compare,
    constants: dict[str, Any],
    aliases: dict[str, tuple[str, str]],
) -> bool:
    """Evaluate a comparison chain such as ``X != "0"`` or ``0 < n <= 5``.

    The config uses this for flags derived from a pinned string -- e.g.
    ``"mtp": ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT != "0"``. Chained comparisons short-circuit
    the same way Python's do, which is why this walks the operator/comparator pairs rather than
    evaluating them all up front.
    """
    left = _evaluate(node.left, constants, aliases)
    for operator, comparator_node in zip(node.ops, node.comparators):
        handler = _COMPARISONS.get(type(operator))
        if handler is None:  # pragma: no cover - every cmpop is covered by the table above.
            raise UnresolvableConfigValue(
                f"comparison operator {type(operator).__name__} is not supported."
            )
        right = _evaluate(comparator_node, constants, aliases)
        if not handler(left, right):
            return False
        left = right
    return True


def parse_submitted_agent_config(agent_source_path: Path) -> dict[str, Any]:
    """Return ``SUBMITTED_AGENT_CONFIG`` as a plain dict, read from source without importing.

    Args:
        agent_source_path: path to ``python/carnot/agentic/arc_competition_agent.py``.

    Returns:
        The config dict, with every value resolved to a real Python object.

    Raises:
        UnresolvableConfigValue: if any entry cannot be resolved. Deliberately a hard failure:
            a config silently missing the one flag a gate is about to assert on is far worse
            than a loud error, because the gate would then measure the wrong agent and report
            success.
    """
    tree = ast.parse(agent_source_path.read_text(encoding="utf-8"))
    aliases = _import_alias_tables(tree)
    constants: dict[str, Any] = {}

    for node in tree.body:
        target: str | None = None
        value: ast.expr | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
            value = node.value
        if target is None or value is None:
            continue

        if target == "SUBMITTED_AGENT_CONFIG" and isinstance(value, ast.Dict):
            config: dict[str, Any] = {}
            for key_node, value_node in zip(value.keys, value.values):
                if key_node is None:  # pragma: no cover - ``{**spread}`` inside the config.
                    raise UnresolvableConfigValue(
                        "SUBMITTED_AGENT_CONFIG contains a ** spread, which this reader cannot "
                        "expand without importing the module."
                    )
                key = ast.literal_eval(key_node)
                try:
                    config[key] = _evaluate(value_node, constants, aliases)
                except UnresolvableConfigValue as exc:
                    raise UnresolvableConfigValue(
                        f"SUBMITTED_AGENT_CONFIG[{key!r}] could not be resolved: {exc}"
                    ) from exc
            return config

        # A module-level constant. Unlike the original reader we do NOT silently skip a name we
        # cannot resolve here: plenty of module-level assignments in the agent are genuinely
        # computed and irrelevant to the config, so skipping *those* is correct. The difference
        # is that a name needed by the config now fails loudly at the point of use above,
        # naming the key, rather than vanishing here and resurfacing as a KeyError.
        try:
            constants[target] = _evaluate(value, constants, aliases)
        except UnresolvableConfigValue:
            continue

    return {}
