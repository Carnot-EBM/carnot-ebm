"""Bounded ASP-to-energy compiler for exact semantic checks.

Spec refs: REQ-CONSTRAINT-6274,
SCENARIO-CONSTRAINT-6274-SOLVER-PARITY,
SCENARIO-CONSTRAINT-6274-FAIL-CLOSED,
SCENARIO-CONSTRAINT-6274-LOCAL-RECEIPTS.

The compiler accepts a small propositional ASP subset. It turns each rule into
an inspectable non-negative energy term. A state has zero energy exactly when
it is a stable model for the supported subset. Clingo remains the independent
oracle used by the experiment harness.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
import re
from typing import Any


ATOM_RE = re.compile(r"^[a-z][a-z0-9_]*$")
CARDINALITY_RE = re.compile(r"^(?P<lower>\d+)\s*\{\s*(?P<atoms>[^{}]*)\s*\}\s*(?P<upper>\d+)$")


class UnsupportedASPSyntax(ValueError):
    """Raised when source text leaves the explicit Exp6274 ASP subset.

    The exception records that no energy was built. Callers can use that bit to
    prove fail-closed behavior for malformed or richer ASP programs.
    """

    def __init__(self, syntax_class: str, statement: str) -> None:
        self.syntax_class = syntax_class
        self.statement = statement
        self.energy_constructed = False
        super().__init__(f"unsupported_asp_syntax:{syntax_class}:{statement}")


@dataclass(frozen=True)
class NormalRule:
    """A grounded normal rule with optional default-negated body atoms."""

    rule_id: str
    head: str
    positive: tuple[str, ...]
    default_negated: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class IntegrityConstraint:
    """A headless rule that forbids its body from becoming true."""

    rule_id: str
    positive: tuple[str, ...]
    default_negated: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class CardinalityRule:
    """A standalone bounded choice over a finite list of propositional atoms."""

    rule_id: str
    lower: int
    atoms: tuple[str, ...]
    upper: int
    source: str


@dataclass(frozen=True)
class ASPProgram:
    """Parsed program in the accepted Exp6274 propositional subset."""

    program_id: str
    source: str
    facts: tuple[tuple[str, str, str], ...]
    rules: tuple[NormalRule, ...]
    constraints: tuple[IntegrityConstraint, ...]
    cardinalities: tuple[CardinalityRule, ...]
    atoms: tuple[str, ...]

    def to_clingo(self) -> str:
        """Render the parsed program back to clingo-compatible ASP text."""

        lines: list[str] = []
        for _, atom, _ in self.facts:
            lines.append(f"{atom}.")
        for card in self.cardinalities:
            lines.append(f"{card.lower} {{{'; '.join(card.atoms)}}} {card.upper}.")
        for rule in self.rules:
            lines.append(f"{rule.head} :- {_body_text(rule.positive, rule.default_negated)}.")
        for constraint in self.constraints:
            lines.append(f":- {_body_text(constraint.positive, constraint.default_negated)}.")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ASPEnergyTerm:
    """One local, inspectable non-negative energy contribution."""

    rule_id: str
    kind: str
    source: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class CompiledASPProgram:
    """Executable energy view over a parsed bounded ASP program."""

    program: ASPProgram
    energy_terms: tuple[ASPEnergyTerm, ...]

    @property
    def exact_state_count(self) -> int:
        """Return the full Boolean state count for the program atoms."""

        return 2 ** len(self.program.atoms)

    def enumerate_states(self) -> list[list[str]]:
        """Enumerate every bounded state as a sorted atom list."""

        atoms = self.program.atoms
        return [list(state) for state in _powerset(atoms)]

    def energy(self, state: Iterable[str]) -> int:
        """Return total non-negative semantic energy for one state."""

        receipt = self.decompose_state(state)
        return int(receipt["total_energy"])

    def zero_energy_states(self) -> list[list[str]]:
        """Return every enumerated state with zero semantic energy."""

        return sorted(state for state in self.enumerate_states() if self.energy(state) == 0)

    def decompose_state(self, state: Iterable[str]) -> dict[str, Any]:
        """Return total energy plus per-rule violation receipts for a state."""

        canonical = tuple(canonical_state(state))
        state_set = set(canonical)
        rows = [self._evaluate_term(term, state_set) for term in self.energy_terms]
        return {
            "state": list(canonical),
            "total_energy": sum(int(row["energy"]) for row in rows),
            "terms": rows,
        }

    def _evaluate_term(self, term: ASPEnergyTerm, state: set[str]) -> dict[str, Any]:
        if term.kind == "fact":
            atom = str(term.payload["atom"])
            energy = 0 if atom in state else 1
            violation = "satisfied" if energy == 0 else f"missing_fact:{atom}"
        elif term.kind == "normal_rule":
            body_true = _body_true(term.payload["positive"], term.payload["default_negated"], state)
            head = str(term.payload["head"])
            energy = 1 if body_true and head not in state else 0
            violation = "satisfied" if energy == 0 else "body_true_head_false"
        elif term.kind == "integrity":
            body_true = _body_true(term.payload["positive"], term.payload["default_negated"], state)
            energy = 1 if body_true else 0
            violation = "satisfied" if energy == 0 else "forbidden_body_true"
        elif term.kind == "cardinality":
            atoms = tuple(str(atom) for atom in term.payload["atoms"])
            count = sum(1 for atom in atoms if atom in state)
            lower = int(term.payload["lower"])
            upper = int(term.payload["upper"])
            if count < lower:
                energy = lower - count
                violation = f"cardinality_below_lower:{count}<{lower}"
            elif count > upper:
                energy = count - upper
                violation = f"cardinality_above_upper:{count}>{upper}"
            else:
                energy = 0
                violation = "satisfied"
        else:
            least = _least_reduct_model(self.program, state)
            missing = sorted(least - state)
            unsupported = sorted(state - least)
            energy = len(missing) + len(unsupported)
            pieces = []
            if missing:
                pieces.append(f"missing_atoms:{','.join(missing)}")
            if unsupported:
                pieces.append(f"unsupported_atoms:{','.join(unsupported)}")
            violation = ";".join(pieces) if pieces else "satisfied"
        return {
            "rule_id": term.rule_id,
            "kind": term.kind,
            "energy": int(energy),
            "violation": violation,
        }


def compile_program(source: str, *, program_id: str = "asp_program") -> CompiledASPProgram:
    """Parse supported ASP text and build inspectable energy terms."""

    program = parse_program(source, program_id=program_id)
    terms = _build_energy_terms(program)
    return CompiledASPProgram(program=program, energy_terms=tuple(terms))


def parse_program(source: str, *, program_id: str = "asp_program") -> ASPProgram:
    """Parse the explicit Exp6274 ASP subset and reject richer syntax."""

    statements = _statements(source)
    facts: list[tuple[str, str, str]] = []
    rules: list[NormalRule] = []
    constraints: list[IntegrityConstraint] = []
    cardinalities: list[CardinalityRule] = []
    for statement in statements:
        _reject_global_unsupported(statement)
        card_match = CARDINALITY_RE.fullmatch(statement)
        if card_match is not None:
            cardinalities.append(_parse_cardinality(card_match, statement, len(cardinalities) + 1))
        elif "{" in statement or "}" in statement:
            raise UnsupportedASPSyntax("malformed_cardinality", statement)
        elif ":-" in statement:
            head_text, body_text = (part.strip() for part in statement.split(":-", 1))
            positive, default_negated = _parse_body(body_text, statement)
            if head_text:
                head = _parse_atom(head_text, statement)
                rules.append(
                    NormalRule(
                        f"R{len(rules) + 1:03d}",
                        head,
                        positive,
                        default_negated,
                        statement,
                    )
                )
            else:
                constraints.append(
                    IntegrityConstraint(
                        f"I{len(constraints) + 1:03d}",
                        positive,
                        default_negated,
                        statement,
                    )
                )
        else:
            atom = _parse_atom(statement, statement)
            facts.append((f"F{len(facts) + 1:03d}", atom, statement))
    atoms = _program_atoms(facts, rules, constraints, cardinalities)
    return ASPProgram(
        program_id=program_id,
        source=source,
        facts=tuple(facts),
        rules=tuple(rules),
        constraints=tuple(constraints),
        cardinalities=tuple(cardinalities),
        atoms=tuple(sorted(atoms)),
    )


def solve_with_clingo(program: ASPProgram) -> list[list[str]]:
    """Return answer sets from clingo for an already parsed supported program."""

    try:
        import clingo
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("independent_solver_missing:clingo") from exc

    control = clingo.Control(["0", "--warn=none"])
    control.add("base", [], program.to_clingo())
    control.ground([("base", [])])
    answer_sets: list[list[str]] = []
    with control.solve(yield_=True) as handle:
        for model in handle:
            answer_sets.append(canonical_state(str(symbol) for symbol in model.symbols(shown=True)))
    return sorted(answer_sets)


def canonical_state(state: Iterable[str]) -> list[str]:
    """Return a sorted, duplicate-free atom list after validating atoms."""

    atoms = sorted(set(str(atom) for atom in state))
    for atom in atoms:
        _parse_atom(atom, atom)
    return atoms


def solver_name_version() -> str:
    """Return the independent solver name and version used by parity checks."""

    try:
        import clingo
    except ImportError:  # pragma: no cover
        return "clingo:missing"
    return f"clingo {clingo.__version__}"


def _build_energy_terms(program: ASPProgram) -> list[ASPEnergyTerm]:
    terms: list[ASPEnergyTerm] = []
    for rule_id, atom, source in program.facts:
        terms.append(ASPEnergyTerm(rule_id, "fact", source, {"atom": atom}))
    for card in program.cardinalities:
        terms.append(
            ASPEnergyTerm(
                card.rule_id,
                "cardinality",
                card.source,
                {"lower": card.lower, "atoms": card.atoms, "upper": card.upper},
            )
        )
    for rule in program.rules:
        terms.append(
            ASPEnergyTerm(
                rule.rule_id,
                "normal_rule",
                rule.source,
                {
                    "head": rule.head,
                    "positive": rule.positive,
                    "default_negated": rule.default_negated,
                },
            )
        )
    for constraint in program.constraints:
        terms.append(
            ASPEnergyTerm(
                constraint.rule_id,
                "integrity",
                constraint.source,
                {
                    "positive": constraint.positive,
                    "default_negated": constraint.default_negated,
                },
            )
        )
    terms.append(ASPEnergyTerm("STABLE_SUPPORT", "stable_support", "stable model reduct", {}))
    return terms


def _least_reduct_model(program: ASPProgram, candidate: set[str]) -> set[str]:
    open_atoms = {atom for card in program.cardinalities for atom in card.atoms}
    model = {atom for _, atom, _ in program.facts} | (candidate & open_atoms)
    changed = True
    while changed:
        changed = False
        for rule in program.rules:
            if set(rule.default_negated) & candidate:
                continue
            if set(rule.positive).issubset(model) and rule.head not in model:
                model.add(rule.head)
                changed = True
    return model


def _body_true(
    positive: Sequence[str] | Any,
    default_negated: Sequence[str] | Any,
    state: set[str],
) -> bool:
    return set(positive).issubset(state) and not (set(default_negated) & state)


def _statements(source: str) -> list[str]:
    stripped_lines = [line.split("%", 1)[0].strip() for line in source.splitlines()]
    cleaned = "\n".join(line for line in stripped_lines if line)
    if not cleaned:
        raise UnsupportedASPSyntax("empty_program", "")
    if not cleaned.rstrip().endswith("."):
        raise UnsupportedASPSyntax("missing_period", cleaned)
    statements = [part.strip() for part in cleaned.split(".") if part.strip()]
    if not statements:
        raise UnsupportedASPSyntax("empty_program", cleaned)
    return statements


def _reject_global_unsupported(statement: str) -> None:
    if statement.startswith("#"):
        raise UnsupportedASPSyntax("directive_or_optimization", statement)
    if "|" in statement:
        raise UnsupportedASPSyntax("disjunction", statement)
    if "{" in statement and "}" in statement:
        inner = statement[statement.index("{") + 1 : statement.rindex("}")]
        if ":" in inner:
            raise UnsupportedASPSyntax("conditional_cardinality", statement)
    scan = statement.replace(":-", " ")
    if any(token in scan for token in ("+", "*", "/", "=", "<", ">")):
        raise UnsupportedASPSyntax("arithmetic_or_comparison", statement)


def _parse_cardinality(
    match: re.Match[str],
    statement: str,
    ordinal: int,
) -> CardinalityRule:
    lower = int(match.group("lower"))
    upper = int(match.group("upper"))
    atom_text = match.group("atoms").strip()
    if not atom_text:
        raise UnsupportedASPSyntax("empty_cardinality", statement)
    atoms = tuple(_parse_atom(part.strip(), statement) for part in atom_text.split(";"))
    if len(set(atoms)) != len(atoms):
        raise UnsupportedASPSyntax("duplicate_cardinality_atom", statement)
    if lower > upper or upper > len(atoms):
        raise UnsupportedASPSyntax("invalid_cardinality_bounds", statement)
    return CardinalityRule(f"C{ordinal:03d}", lower, atoms, upper, statement)


def _parse_body(body_text: str, statement: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not body_text:
        raise UnsupportedASPSyntax("malformed_body", statement)
    positive: list[str] = []
    default_negated: list[str] = []
    for raw in body_text.split(","):
        literal = raw.strip()
        if not literal:
            raise UnsupportedASPSyntax("malformed_literal", statement)
        if literal.startswith("not "):
            atom = literal[4:].strip()
            default_negated.append(_parse_atom(atom, statement))
        elif literal == "not":
            raise UnsupportedASPSyntax("malformed_literal", statement)
        else:
            positive.append(_parse_atom(literal, statement))
    return tuple(positive), tuple(default_negated)


def _parse_atom(atom_text: str, statement: str) -> str:
    if re.search(r"[A-Z]", atom_text):
        raise UnsupportedASPSyntax("variables", statement)
    if "(" in atom_text or ")" in atom_text:
        raise UnsupportedASPSyntax("function_or_predicate_terms", statement)
    if not ATOM_RE.fullmatch(atom_text):
        raise UnsupportedASPSyntax("malformed_atom", statement)
    return atom_text


def _program_atoms(
    facts: Sequence[tuple[str, str, str]],
    rules: Sequence[NormalRule],
    constraints: Sequence[IntegrityConstraint],
    cardinalities: Sequence[CardinalityRule],
) -> set[str]:
    atoms = {atom for _, atom, _ in facts}
    for card in cardinalities:
        atoms.update(card.atoms)
    for rule in rules:
        atoms.add(rule.head)
        atoms.update(rule.positive)
        atoms.update(rule.default_negated)
    for constraint in constraints:
        atoms.update(constraint.positive)
        atoms.update(constraint.default_negated)
    return atoms


def _powerset(atoms: Sequence[str]) -> list[tuple[str, ...]]:
    return [subset for size in range(len(atoms) + 1) for subset in combinations(atoms, size)]


def _body_text(positive: Sequence[str], default_negated: Sequence[str]) -> str:
    body = [*positive, *(f"not {atom}" for atom in default_negated)]
    return ", ".join(body)
