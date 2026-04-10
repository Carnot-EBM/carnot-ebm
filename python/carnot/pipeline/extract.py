"""Pluggable constraint extraction from text, code, and natural language.

**Researcher summary:**
    Provides a Protocol-based extractor architecture where each domain
    (arithmetic, code, logic, NL) has its own extractor class. AutoExtractor
    combines them all and auto-detects which domains apply to a given input.

**Detailed explanation for engineers:**
    This module consolidates constraint extraction logic that was previously
    scattered across experiment scripts (Exp 47, 48, 49) into a clean,
    importable library. Each extractor parses a specific type of content and
    returns a list of ConstraintResult objects that can optionally carry an
    energy term for Ising-model verification.

    The architecture is pluggable: add a new extractor class implementing the
    ConstraintExtractor Protocol, register it with AutoExtractor, and the
    pipeline automatically picks it up.

    Key classes:
    - ConstraintResult: dataclass holding one extracted constraint with its
      type, description, optional energy term, and domain-specific metadata.
    - ConstraintExtractor: Protocol defining the extract() interface.
    - ArithmeticExtractor: Parses "X + Y = Z" style claims (from Exp 47).
    - CodeExtractor: Parses Python code via ast (from Exp 48).
    - LogicExtractor: Parses "If P then Q" logical claims (from Exp 47).
    - NLExtractor: Parses factual claims via regex patterns (from Exp 49).
    - AutoExtractor: Combines all extractors and merges results.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from carnot.verify.constraint import ConstraintTerm


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class ConstraintResult:
    """A single extracted constraint ready for verification.

    **Detailed explanation for engineers:**
        Each constraint extracted from text or code is wrapped in this
        dataclass. The ``energy_term`` field is optional -- extractors that
        know how to build an Ising/EBM energy term for the constraint will
        populate it; otherwise downstream pipeline stages can construct one
        from the metadata.

    Attributes:
        constraint_type: Category tag, e.g. "arithmetic", "type_check",
            "bound", "implication", "factual".
        description: Human-readable summary of what the constraint checks.
        energy_term: Optional ConstraintTerm for Ising verification. None
            means the constraint was extracted but not yet encoded as energy.
        metadata: Domain-specific details. For arithmetic: keys ``a``, ``b``,
            ``claimed_result``, ``correct_result``, ``satisfied``. For code:
            keys ``kind``, ``function``, ``variable``, etc.

    Spec: REQ-VERIFY-001
    """

    constraint_type: str
    description: str
    energy_term: ConstraintTerm | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extractor Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ConstraintExtractor(Protocol):
    """Protocol for domain-specific constraint extractors.

    **Detailed explanation for engineers:**
        Any class that implements ``extract()`` and ``supported_domains``
        can be used as a pluggable extractor in the pipeline. The Protocol
        is runtime-checkable so you can verify at registration time that
        a class conforms.

    Spec: REQ-VERIFY-001
    """

    @property
    def supported_domains(self) -> list[str]:
        """List of domain tags this extractor handles (e.g. ["arithmetic"])."""
        ...

    def extract(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
        """Extract constraints from *text*.

        Args:
            text: Input text (may contain prose, code, or mixed content).
            domain: Optional domain hint. If provided, the extractor may
                skip processing if the domain is not in supported_domains.

        Returns:
            List of extracted ConstraintResult objects (may be empty).
        """
        ...


# ---------------------------------------------------------------------------
# ArithmeticExtractor — from Experiment 47
# ---------------------------------------------------------------------------


class ArithmeticExtractor:
    """Extract and verify "X + Y = Z" style arithmetic claims from text.

    **Detailed explanation for engineers:**
        Scans the input text for patterns like "47 + 28 = 75" using regex.
        Supports addition and subtraction with positive and negative integers.
        Each match is verified by direct computation and returned as a
        ConstraintResult with ``satisfied`` in metadata.

        Reuses the verification logic from Experiment 47's
        ``verify_arithmetic_constraint`` but as a clean, stateless method.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
    """

    @property
    def supported_domains(self) -> list[str]:
        return ["arithmetic"]

    def extract(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
        if domain is not None and domain not in self.supported_domains:
            return []

        results: list[ConstraintResult] = []
        # Match patterns like "47 + 28 = 75", "-3 + 10 = 7", "15 - 7 = 8"
        pattern = r"(-?\d+)\s*([+\-])\s*(-?\d+)\s*=\s*(-?\d+)"
        for match in re.finditer(pattern, text):
            a = int(match.group(1))
            op = match.group(2)
            b_raw = int(match.group(3))
            claimed = int(match.group(4))

            # Apply the operator to get the effective operand.
            b = b_raw if op == "+" else -b_raw
            correct = a + b
            satisfied = claimed == correct

            results.append(
                ConstraintResult(
                    constraint_type="arithmetic",
                    description=f"{a} {op} {b_raw} = {claimed}"
                    + ("" if satisfied else f" (correct: {correct})"),
                    metadata={
                        "a": a,
                        "b": b,
                        "operator": op,
                        "claimed_result": claimed,
                        "correct_result": correct,
                        "satisfied": satisfied,
                    },
                )
            )
        return results


# ---------------------------------------------------------------------------
# CodeExtractor — from Experiment 48
# ---------------------------------------------------------------------------


class CodeExtractor:
    """Extract verifiable constraints from Python code blocks in text.

    **Detailed explanation for engineers:**
        Finds fenced Python code blocks (triple-backtick or indented) in the
        input text, parses each via ``ast``, and extracts four kinds of
        constraints per function definition:

        1. **Type constraints** — parameter type annotations.
        2. **Return-type constraints** — declared return types and whether
           literal return values match.
        3. **Loop-bound constraints** — ``for i in range(...)`` implies
           ``0 <= i < bound``.
        4. **Initialization constraints** — variables used but never assigned.

        Consolidates the logic from Experiment 48's ``code_to_constraints``
        into a clean Protocol-conforming class.

    Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
    """

    @property
    def supported_domains(self) -> list[str]:
        return ["code"]

    def extract(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
        if domain is not None and domain not in self.supported_domains:
            return []

        # Extract code blocks from text (fenced or treat whole text as code).
        code_blocks = self._extract_code_blocks(text)
        results: list[ConstraintResult] = []
        for code in code_blocks:
            results.extend(self._extract_from_code(code))
        return results

    def _extract_code_blocks(self, text: str) -> list[str]:
        """Find fenced Python code blocks or treat raw text as code.

        **Detailed explanation for engineers:**
            Looks for ```python ... ``` or ``` ... ``` fenced blocks first.
            If none found, tries to parse the entire text as Python. If that
            also fails, returns an empty list (no code to analyze).
        """
        blocks: list[str] = []
        # Fenced code blocks: ```python ... ``` or ``` ... ```
        pattern = r"```(?:python)?\s*\n(.*?)```"
        for match in re.finditer(pattern, text, re.DOTALL):
            blocks.append(match.group(1))

        if not blocks:
            # Try parsing the whole text as code.
            try:
                ast.parse(text)
                blocks.append(text)
            except SyntaxError:
                pass
        return blocks

    def _extract_from_code(self, code: str) -> list[ConstraintResult]:
        """Parse a single code string and extract constraints.

        **Detailed explanation for engineers:**
            Walks the AST looking for FunctionDef nodes. For each function,
            extracts type annotations, return types, loop bounds, and
            initialization checks. This is the core logic from Exp 48's
            ``code_to_constraints``, adapted to return ConstraintResult objects.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        results: list[ConstraintResult] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            func_name = node.name
            results.extend(self._extract_type_constraints(node, func_name))
            results.extend(self._extract_return_constraints(node, func_name))
            results.extend(self._extract_loop_bounds(node, func_name))
            results.extend(self._extract_initialization(node, func_name))

        return results

    def _extract_type_constraints(
        self, node: ast.FunctionDef, func_name: str
    ) -> list[ConstraintResult]:
        """Extract parameter type annotation constraints."""
        results: list[ConstraintResult] = []
        for arg in node.args.args:
            if arg.annotation:
                tname = self._annotation_to_type_name(arg.annotation)
                if tname:
                    results.append(
                        ConstraintResult(
                            constraint_type="type_check",
                            description=(
                                f"{func_name}(): parameter '{arg.arg}'"
                                f" annotated as {tname}"
                            ),
                            metadata={
                                "kind": "type",
                                "function": func_name,
                                "variable": arg.arg,
                                "expected_type": tname,
                            },
                        )
                    )
        return results

    def _extract_return_constraints(
        self, node: ast.FunctionDef, func_name: str
    ) -> list[ConstraintResult]:
        """Extract return type annotation and literal return type checks."""
        results: list[ConstraintResult] = []
        return_type: str | None = None

        if node.returns:
            return_type = self._annotation_to_type_name(node.returns)
            if return_type:
                results.append(
                    ConstraintResult(
                        constraint_type="return_type",
                        description=(
                            f"{func_name}() annotated to return {return_type}"
                        ),
                        metadata={
                            "kind": "return_type",
                            "function": func_name,
                            "expected_type": return_type,
                        },
                    )
                )

        # Check literal return values against declared type.
        if return_type:
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Return)
                    and child.value is not None
                    and isinstance(child.value, ast.Constant)
                ):
                    actual = type(child.value.value).__name__
                    compatible = self._types_compatible(actual, return_type)
                    results.append(
                        ConstraintResult(
                            constraint_type="return_value_type",
                            description=(
                                f"{func_name}() returns literal of type"
                                f" {actual}, expected {return_type}"
                                + ("" if compatible else " — MISMATCH")
                            ),
                            metadata={
                                "kind": "return_value_type",
                                "function": func_name,
                                "expected_type": return_type,
                                "actual_type": actual,
                                "satisfied": compatible,
                            },
                        )
                    )
        return results

    def _extract_loop_bounds(
        self, node: ast.FunctionDef, func_name: str
    ) -> list[ConstraintResult]:
        """Extract loop variable bound constraints from range() calls."""
        results: list[ConstraintResult] = []
        for child in ast.walk(node):
            if not (
                isinstance(child, ast.For)
                and isinstance(child.target, ast.Name)
            ):
                continue
            loop_var = child.target.id
            if not (
                isinstance(child.iter, ast.Call)
                and isinstance(child.iter.func, ast.Name)
                and child.iter.func.id == "range"
            ):
                continue
            args = child.iter.args
            if len(args) == 1:
                bound = self._expr_source(args[0])
                results.append(
                    ConstraintResult(
                        constraint_type="bound",
                        description=(
                            f"{func_name}(): loop var '{loop_var}'"
                            f" bounded by 0 <= {loop_var} < {bound}"
                        ),
                        metadata={
                            "kind": "loop_bound",
                            "function": func_name,
                            "variable": loop_var,
                            "lower": 0,
                            "upper_expr": bound,
                            "satisfied": True,
                        },
                    )
                )
            elif len(args) >= 2:
                lo = self._expr_source(args[0])
                hi = self._expr_source(args[1])
                results.append(
                    ConstraintResult(
                        constraint_type="bound",
                        description=(
                            f"{func_name}(): loop var '{loop_var}'"
                            f" bounded by {lo} <= {loop_var} < {hi}"
                        ),
                        metadata={
                            "kind": "loop_bound",
                            "function": func_name,
                            "variable": loop_var,
                            "lower_expr": lo,
                            "upper_expr": hi,
                            "satisfied": True,
                        },
                    )
                )
        return results

    def _extract_initialization(
        self, node: ast.FunctionDef, func_name: str
    ) -> list[ConstraintResult]:
        """Check that all used variables are assigned or passed as params."""
        results: list[ConstraintResult] = []
        builtins_set = {
            "print", "len", "range", "int", "str", "float", "bool", "list",
            "dict", "set", "tuple", "type", "isinstance", "enumerate", "zip",
            "map", "filter", "sorted", "reversed", "sum", "min", "max", "abs",
            "True", "False", "None", "ValueError", "TypeError", "IndexError",
            "KeyError", "Exception", "super", "property", "staticmethod",
            "classmethod", "open", "input", "any", "all", "iter", "next",
            "hasattr", "getattr", "setattr",
        }
        param_names = {arg.arg for arg in node.args.args}
        assigned = param_names | self._collect_assigned_names(node.body)
        used = self._collect_used_names(node)
        uninitialized = used - assigned - builtins_set - {node.name, "self"}

        for uname in sorted(uninitialized):
            results.append(
                ConstraintResult(
                    constraint_type="initialization",
                    description=(
                        f"{func_name}(): variable '{uname}' used but"
                        " never assigned or passed as parameter"
                    ),
                    metadata={
                        "kind": "initialization",
                        "function": func_name,
                        "variable": uname,
                        "satisfied": False,
                    },
                )
            )
        return results

    # --- AST helpers (from Exp 48) ---

    @staticmethod
    def _annotation_to_type_name(node: ast.expr) -> str | None:
        """Extract simple type name from an AST annotation node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    @staticmethod
    def _types_compatible(actual: str, expected: str) -> bool:
        """Check if actual Python type is compatible with expected annotation.

        Conservative check: int->float promotion OK, bool->int OK.
        """
        if actual == expected:
            return True
        if actual == "int" and expected == "float":
            return True
        if actual == "bool" and expected in ("int", "bool"):
            return True
        return False

    @staticmethod
    def _expr_source(node: ast.expr) -> str:
        """Best-effort source text for an AST expression.

        **Detailed explanation for engineers:**
            Uses ast.unparse (available since Python 3.9) to reconstruct
            source code from an AST node. Since this project targets
            Python 3.11+, ast.unparse is always available.
        """
        return ast.unparse(node)

    @staticmethod
    def _collect_assigned_names(body: list[ast.stmt]) -> set[str]:
        """Walk statements and return all names that receive an assignment."""
        assigned: set[str] = set()
        for stmt in body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = (
                    stmt.targets
                    if isinstance(stmt, ast.Assign)
                    else [stmt.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        assigned.add(target.id)
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name):
                    assigned.add(stmt.target.id)
                assigned |= CodeExtractor._collect_assigned_names(stmt.body)
            elif isinstance(stmt, (ast.If, ast.While)):
                assigned |= CodeExtractor._collect_assigned_names(stmt.body)
                assigned |= CodeExtractor._collect_assigned_names(stmt.orelse)
            elif isinstance(stmt, ast.With):
                for item in stmt.items:
                    if item.optional_vars and isinstance(
                        item.optional_vars, ast.Name
                    ):
                        assigned.add(item.optional_vars.id)
                assigned |= CodeExtractor._collect_assigned_names(stmt.body)
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    assigned.add(stmt.target.id)
        return assigned

    @staticmethod
    def _collect_used_names(node: ast.AST) -> set[str]:
        """Return all Name ids in load context (read, not written)."""
        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(
                child.ctx, ast.Load
            ):
                names.add(child.id)
        return names


# ---------------------------------------------------------------------------
# LogicExtractor — from Experiment 47
# ---------------------------------------------------------------------------


class LogicExtractor:
    """Extract "If P then Q" and logical relationship claims from text.

    **Detailed explanation for engineers:**
        Scans text for conditional patterns ("if ... then ...", "if ..., ...")
        and mutual exclusion patterns ("X but not Y", "either X or Y").
        Returns ConstraintResult objects tagged as "implication", "exclusion",
        or "disjunction". These can be encoded as Ising couplings for
        consistency checking.

        Consolidates the logical constraint parsing from Experiment 47's
        scenario format and Experiment 49's ``extract_implication`` and
        ``extract_exclusion`` into a single extractor.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
    """

    @property
    def supported_domains(self) -> list[str]:
        return ["logic"]

    def extract(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
        if domain is not None and domain not in self.supported_domains:
            return []

        results: list[ConstraintResult] = []
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            results.extend(self._extract_implication(sent))
            results.extend(self._extract_exclusion(sent))
            results.extend(self._extract_disjunction(sent))
            results.extend(self._extract_negation(sent))
            results.extend(self._extract_universal(sent))
        return results

    def _extract_implication(self, sentence: str) -> list[ConstraintResult]:
        """Parse "if X then Y" or "if X, Y" patterns."""
        s = self._normalize(sentence)
        m = re.match(r"^if\s+(.+?)(?:,\s*(?:then\s+)?|\s+then\s+)(.+?)\.?$", s)
        if m:
            ante = m.group(1).strip()
            cons = m.group(2).strip()
            return [
                ConstraintResult(
                    constraint_type="implication",
                    description=f"If {ante}, then {cons}",
                    metadata={
                        "antecedent": ante,
                        "consequent": cons,
                        "raw": sentence.strip(),
                    },
                )
            ]
        return []

    def _extract_exclusion(self, sentence: str) -> list[ConstraintResult]:
        """Parse "X but not Y" exclusion patterns."""
        s = self._normalize(sentence)
        m = re.match(r"^(.+?)\s+but\s+not\s+(.+?)\.?$", s)
        if m:
            return [
                ConstraintResult(
                    constraint_type="exclusion",
                    description=(
                        f"{m.group(1).strip()} but not {m.group(2).strip()}"
                    ),
                    metadata={
                        "positive": m.group(1).strip(),
                        "negative": m.group(2).strip(),
                        "raw": sentence.strip(),
                    },
                )
            ]
        return []

    def _extract_disjunction(self, sentence: str) -> list[ConstraintResult]:
        """Parse "X or Y" disjunction patterns."""
        s = self._normalize(sentence)
        m = re.match(r"^(?:either\s+)?(.+?)\s+or\s+(.+?)\.?$", s)
        if m:
            return [
                ConstraintResult(
                    constraint_type="disjunction",
                    description=(
                        f"{m.group(1).strip()} or {m.group(2).strip()}"
                    ),
                    metadata={
                        "left": m.group(1).strip(),
                        "right": m.group(2).strip(),
                        "raw": sentence.strip(),
                    },
                )
            ]
        return []

    def _extract_negation(self, sentence: str) -> list[ConstraintResult]:
        """Parse "X cannot/can't/does not Y" negation patterns."""
        s = self._normalize(sentence)
        m = re.match(
            r"^(.+?)\s+(?:cannot|can't|can not|do not|does not|don't|doesn't)"
            r"\s+(.+?)\.?$",
            s,
        )
        if m:
            return [
                ConstraintResult(
                    constraint_type="negation",
                    description=(
                        f"{m.group(1).strip()} cannot {m.group(2).strip()}"
                    ),
                    metadata={
                        "subject": m.group(1).strip(),
                        "predicate": m.group(2).strip(),
                        "raw": sentence.strip(),
                    },
                )
            ]
        return []

    def _extract_universal(self, sentence: str) -> list[ConstraintResult]:
        """Parse "All X are Y" universal quantifier patterns."""
        s = self._normalize(sentence)
        m = re.match(r"^all\s+(.+?)\s+(?:are|is)\s+(.+?)\.?$", s)
        if m:
            return [
                ConstraintResult(
                    constraint_type="universal",
                    description=(
                        f"All {m.group(1).strip()} are {m.group(2).strip()}"
                    ),
                    metadata={
                        "category": m.group(1).strip(),
                        "property": m.group(2).strip(),
                        "raw": sentence.strip(),
                    },
                )
            ]
        return []

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and strip extra whitespace for pattern matching."""
        return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# NLExtractor — from Experiment 49
# ---------------------------------------------------------------------------


class NLExtractor:
    """Extract factual claims (named entities, quantities) from natural language.

    **Detailed explanation for engineers:**
        Scans text for "X is Y", "X is the Y of Z" factual assertions using
        regex patterns. Also extracts quantity claims like "there are N items".
        Returns ConstraintResult objects tagged as "factual" or
        "factual_relation".

        Consolidates the pattern-based extraction from Experiment 49's
        ``extract_factual_is`` and related functions into a single extractor.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
    """

    @property
    def supported_domains(self) -> list[str]:
        return ["nl"]

    def extract(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
        if domain is not None and domain not in self.supported_domains:
            return []

        results: list[ConstraintResult] = []
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            results.extend(self._extract_factual_relation(sent))
            results.extend(self._extract_factual_is(sent))
            results.extend(self._extract_quantity(sent))
        return results

    def _extract_factual_relation(
        self, sentence: str
    ) -> list[ConstraintResult]:
        """Parse "X is the Y of Z" factual relation claims."""
        s = self._normalize(sentence)
        m = re.match(r"^(.+?)\s+is\s+the\s+(.+?)\s+of\s+(.+?)\.?$", s)
        if m:
            return [
                ConstraintResult(
                    constraint_type="factual_relation",
                    description=(
                        f"{m.group(1).strip()} is the {m.group(2).strip()}"
                        f" of {m.group(3).strip()}"
                    ),
                    metadata={
                        "subject": m.group(1).strip(),
                        "relation": m.group(2).strip(),
                        "object": m.group(3).strip(),
                        "raw": sentence.strip(),
                    },
                )
            ]
        return []

    def _extract_factual_is(self, sentence: str) -> list[ConstraintResult]:
        """Parse "X is/are Y" factual claims (excluding relation patterns)."""
        s = self._normalize(sentence)
        # Skip if already matched by factual_relation pattern.
        if re.match(r"^(.+?)\s+is\s+the\s+(.+?)\s+of\s+(.+?)\.?$", s):
            return []
        m = re.match(r"^(.+?)\s+(?:is|are)\s+(.+?)\.?$", s)
        if m:
            return [
                ConstraintResult(
                    constraint_type="factual",
                    description=(
                        f"{m.group(1).strip()} is/are {m.group(2).strip()}"
                    ),
                    metadata={
                        "subject": m.group(1).strip(),
                        "predicate": m.group(2).strip(),
                        "raw": sentence.strip(),
                    },
                )
            ]
        return []

    def _extract_quantity(self, sentence: str) -> list[ConstraintResult]:
        """Parse "there are N items" or "X has N Y" quantity claims."""
        s = self._normalize(sentence)
        results: list[ConstraintResult] = []

        # "there are N X"
        m = re.match(r"^there\s+(?:are|is)\s+(\d+)\s+(.+?)\.?$", s)
        if m:
            results.append(
                ConstraintResult(
                    constraint_type="quantity",
                    description=f"There are {m.group(1)} {m.group(2).strip()}",
                    metadata={
                        "quantity": int(m.group(1)),
                        "subject": m.group(2).strip(),
                        "raw": sentence.strip(),
                    },
                )
            )
            return results

        # "X has/have N Y"
        m = re.match(
            r"^(.+?)\s+(?:has|have)\s+(\d+)\s+(.+?)\.?$", s
        )
        if m:
            results.append(
                ConstraintResult(
                    constraint_type="quantity",
                    description=(
                        f"{m.group(1).strip()} has {m.group(2)}"
                        f" {m.group(3).strip()}"
                    ),
                    metadata={
                        "owner": m.group(1).strip(),
                        "quantity": int(m.group(2)),
                        "subject": m.group(3).strip(),
                        "raw": sentence.strip(),
                    },
                )
            )
        return results

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and strip extra whitespace."""
        return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# AutoExtractor — combines all extractors
# ---------------------------------------------------------------------------


class AutoExtractor:
    """Combines all domain extractors and auto-detects applicable domains.

    **Detailed explanation for engineers:**
        Holds a registry of ConstraintExtractor instances. When ``extract()``
        is called without a domain hint, it runs ALL extractors and merges
        their results (deduplicating by description). When a domain is
        specified, only extractors that support that domain are invoked.

        The default constructor registers ArithmeticExtractor, CodeExtractor,
        LogicExtractor, NLExtractor, and FactualKBExtractor (Exp 113).
        Additional extractors can be added via ``add_extractor()``.

    Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
    """

    def __init__(self) -> None:
        # Import here to avoid circular imports (knowledge_base imports from
        # extract, so we defer the import to instantiation time).
        from carnot.pipeline.knowledge_base import FactualKBExtractor  # noqa: PLC0415

        self._extractors: list[ConstraintExtractor] = [
            ArithmeticExtractor(),
            CodeExtractor(),
            LogicExtractor(),
            NLExtractor(),
            FactualKBExtractor(),
        ]

    @property
    def supported_domains(self) -> list[str]:
        """Union of all registered extractors' supported domains."""
        domains: list[str] = []
        for ext in self._extractors:
            for d in ext.supported_domains:
                if d not in domains:
                    domains.append(d)
        return domains

    def add_extractor(self, extractor: ConstraintExtractor) -> None:
        """Register an additional domain extractor."""
        self._extractors.append(extractor)

    def extract(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
        """Run all applicable extractors and merge results.

        **Detailed explanation for engineers:**
            If domain is None, runs every extractor. If domain is specified,
            only runs extractors whose supported_domains include it.
            Deduplicates results by description to avoid reporting the same
            constraint from multiple extractors.
        """
        results: list[ConstraintResult] = []
        seen_descriptions: set[str] = set()

        for ext in self._extractors:
            if domain is not None and domain not in ext.supported_domains:
                continue
            for result in ext.extract(text, domain):
                if result.description not in seen_descriptions:
                    seen_descriptions.add(result.description)
                    results.append(result)
        return results
