"""Tests for carnot.pipeline.extract — constraint extraction pipeline.

Each test references REQ-VERIFY-* or SCENARIO-VERIFY-* per spec-anchored
development requirements.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import pytest

from carnot.pipeline.extract import (
    ArithmeticExtractor,
    AutoExtractor,
    CodeExtractor,
    ConstraintExtractor,
    ConstraintResult,
    LogicExtractor,
    NLExtractor,
)


# ---------------------------------------------------------------------------
# ArithmeticExtractor tests — REQ-VERIFY-001, SCENARIO-VERIFY-002
# ---------------------------------------------------------------------------


class TestArithmeticExtractor:
    """Tests for arithmetic claim extraction and verification."""

    def setup_method(self) -> None:
        self.ext = ArithmeticExtractor()

    def test_correct_addition(self) -> None:
        """REQ-VERIFY-001: Correct arithmetic claim is marked satisfied."""
        results = self.ext.extract("47 + 28 = 75")
        assert len(results) == 1
        assert results[0].constraint_type == "arithmetic"
        assert results[0].metadata["satisfied"] is True
        assert results[0].metadata["correct_result"] == 75

    def test_wrong_addition(self) -> None:
        """REQ-VERIFY-001: Incorrect arithmetic claim is marked unsatisfied."""
        results = self.ext.extract("47 + 28 = 76")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is False
        assert results[0].metadata["correct_result"] == 75

    def test_subtraction(self) -> None:
        """REQ-VERIFY-001: Subtraction claims are correctly verified."""
        results = self.ext.extract("15 - 7 = 8")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True

    def test_wrong_subtraction(self) -> None:
        """REQ-VERIFY-001: Wrong subtraction is detected."""
        results = self.ext.extract("15 - 7 = 9")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is False

    def test_negative_operand(self) -> None:
        """REQ-VERIFY-001: Negative operands are handled."""
        results = self.ext.extract("-3 + 10 = 7")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True

    def test_multiple_claims(self) -> None:
        """REQ-VERIFY-001: Multiple arithmetic claims in one text."""
        text = "First: 2 + 3 = 5. Then: 10 - 4 = 6."
        results = self.ext.extract(text)
        assert len(results) == 2
        assert all(r.metadata["satisfied"] for r in results)

    def test_no_arithmetic(self) -> None:
        """REQ-VERIFY-001: Text with no arithmetic returns empty list."""
        results = self.ext.extract("The sky is blue.")
        assert results == []

    def test_empty_text(self) -> None:
        """REQ-VERIFY-001: Empty string returns empty list."""
        results = self.ext.extract("")
        assert results == []

    def test_domain_filter(self) -> None:
        """SCENARIO-VERIFY-002: Domain filter skips non-matching domains."""
        results = self.ext.extract("2 + 3 = 5", domain="code")
        assert results == []

    def test_domain_match(self) -> None:
        """SCENARIO-VERIFY-002: Domain filter passes matching domain."""
        results = self.ext.extract("2 + 3 = 5", domain="arithmetic")
        assert len(results) == 1

    def test_supported_domains(self) -> None:
        """REQ-VERIFY-001: supported_domains includes arithmetic."""
        assert "arithmetic" in self.ext.supported_domains


# ---------------------------------------------------------------------------
# CodeExtractor tests — REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
# ---------------------------------------------------------------------------


class TestCodeExtractor:
    """Tests for Python code constraint extraction."""

    def setup_method(self) -> None:
        self.ext = CodeExtractor()

    def test_type_annotation(self) -> None:
        """REQ-VERIFY-001: Parameter type annotations are extracted."""
        code = "def add(x: int, y: int) -> int:\n    return x + y"
        results = self.ext.extract(code)
        type_results = [r for r in results if r.constraint_type == "type_check"]
        assert len(type_results) == 2
        assert type_results[0].metadata["expected_type"] == "int"

    def test_return_type(self) -> None:
        """REQ-VERIFY-001: Return type annotations are extracted."""
        code = "def get_count() -> int:\n    return 42"
        results = self.ext.extract(code)
        ret_results = [r for r in results if r.constraint_type == "return_type"]
        assert len(ret_results) == 1
        assert ret_results[0].metadata["expected_type"] == "int"

    def test_return_value_type_match(self) -> None:
        """REQ-VERIFY-002: Literal return value type check passes when matching."""
        code = "def get_count() -> int:\n    return 42"
        results = self.ext.extract(code)
        rv_results = [
            r for r in results if r.constraint_type == "return_value_type"
        ]
        assert len(rv_results) == 1
        assert rv_results[0].metadata["satisfied"] is True

    def test_return_value_type_mismatch(self) -> None:
        """REQ-VERIFY-002: Literal return value type mismatch is detected."""
        code = 'def get_id() -> int:\n    return "not_a_number"'
        results = self.ext.extract(code)
        rv_results = [
            r for r in results if r.constraint_type == "return_value_type"
        ]
        assert len(rv_results) == 1
        assert rv_results[0].metadata["satisfied"] is False

    def test_loop_bound(self) -> None:
        """REQ-VERIFY-001: Loop bounds from range() are extracted."""
        code = (
            "def sum_list(arr: list) -> int:\n"
            "    total = 0\n"
            "    for i in range(len(arr)):\n"
            "        total += arr[i]\n"
            "    return total"
        )
        results = self.ext.extract(code)
        bound_results = [r for r in results if r.constraint_type == "bound"]
        assert len(bound_results) == 1
        assert bound_results[0].metadata["lower"] == 0

    def test_loop_bound_two_args(self) -> None:
        """REQ-VERIFY-001: range(start, end) bounds are extracted."""
        code = (
            "def partial(arr: list, start: int, end: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(start, end):\n"
            "        total += arr[i]\n"
            "    return total"
        )
        results = self.ext.extract(code)
        bound_results = [r for r in results if r.constraint_type == "bound"]
        assert len(bound_results) == 1
        assert "lower_expr" in bound_results[0].metadata

    def test_uninitialized_variable(self) -> None:
        """REQ-VERIFY-002: Uninitialized variable usage is detected."""
        code = "def compute(x: int) -> int:\n    return x + uninitialized_var"
        results = self.ext.extract(code)
        init_results = [
            r for r in results if r.constraint_type == "initialization"
        ]
        assert len(init_results) == 1
        assert init_results[0].metadata["variable"] == "uninitialized_var"
        assert init_results[0].metadata["satisfied"] is False

    def test_all_initialized(self) -> None:
        """REQ-VERIFY-002: No initialization issues for well-formed code."""
        code = "def add(x: int, y: int) -> int:\n    return x + y"
        results = self.ext.extract(code)
        init_results = [
            r for r in results if r.constraint_type == "initialization"
        ]
        assert len(init_results) == 0

    def test_fenced_code_block(self) -> None:
        """SCENARIO-VERIFY-002: Code inside fenced blocks is parsed."""
        text = (
            "Here is some code:\n"
            "```python\n"
            "def greet(name: str) -> str:\n"
            '    return "hello"\n'
            "```\n"
        )
        results = self.ext.extract(text)
        assert len(results) > 0
        assert any(r.constraint_type == "type_check" for r in results)

    def test_no_code(self) -> None:
        """REQ-VERIFY-001: Non-code text returns empty list."""
        results = self.ext.extract("Just a regular sentence.")
        assert results == []

    def test_syntax_error_code(self) -> None:
        """REQ-VERIFY-001: Malformed code returns empty list, no crash."""
        results = self.ext.extract("def broken(:\n    pass")
        assert results == []

    def test_empty_text(self) -> None:
        """REQ-VERIFY-001: Empty string returns empty list."""
        results = self.ext.extract("")
        assert results == []

    def test_domain_filter(self) -> None:
        """SCENARIO-VERIFY-002: Domain filter skips non-matching domains."""
        code = "def add(x: int) -> int:\n    return x"
        results = self.ext.extract(code, domain="arithmetic")
        assert results == []

    def test_supported_domains(self) -> None:
        """REQ-VERIFY-001: supported_domains includes code."""
        assert "code" in self.ext.supported_domains

    def test_syntax_error_in_fenced_block(self) -> None:
        """REQ-VERIFY-001: SyntaxError inside a fenced block is handled gracefully."""
        text = (
            "```python\n"
            "def broken(:\n"
            "    pass\n"
            "```\n"
        )
        results = self.ext.extract(text)
        assert results == []

    def test_for_loop_without_range(self) -> None:
        """REQ-VERIFY-001: For loops not using range() are skipped for bounds."""
        code = (
            "def process(items: list) -> int:\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        total += item\n"
            "    return total"
        )
        results = self.ext.extract(code)
        bound_results = [r for r in results if r.constraint_type == "bound"]
        assert len(bound_results) == 0

    def test_string_annotation(self) -> None:
        """REQ-VERIFY-001: String-style type annotations are extracted."""
        code = 'def f(x: "int") -> "str":\n    return "hello"'
        results = self.ext.extract(code)
        type_results = [r for r in results if r.constraint_type == "type_check"]
        assert len(type_results) == 1
        assert type_results[0].metadata["expected_type"] == "int"

    def test_attribute_annotation(self) -> None:
        """REQ-VERIFY-001: Dotted attribute type annotations are extracted."""
        code = "def f(x: typing.List) -> typing.Dict:\n    return {}"
        results = self.ext.extract(code)
        type_results = [r for r in results if r.constraint_type == "type_check"]
        assert len(type_results) == 1
        assert type_results[0].metadata["expected_type"] == "List"

    def test_int_to_float_compatible(self) -> None:
        """REQ-VERIFY-002: int literal returning for float annotation is compatible."""
        code = "def get_val() -> float:\n    return 42"
        results = self.ext.extract(code)
        rv_results = [
            r for r in results if r.constraint_type == "return_value_type"
        ]
        assert len(rv_results) == 1
        assert rv_results[0].metadata["satisfied"] is True

    def test_bool_to_int_compatible(self) -> None:
        """REQ-VERIFY-002: bool literal returning for int annotation is compatible."""
        code = "def get_flag() -> int:\n    return True"
        results = self.ext.extract(code)
        rv_results = [
            r for r in results if r.constraint_type == "return_value_type"
        ]
        assert len(rv_results) == 1
        assert rv_results[0].metadata["satisfied"] is True

    def test_ann_assign_in_body(self) -> None:
        """REQ-VERIFY-002: Annotated assignments in function body are tracked."""
        code = (
            "def f() -> int:\n"
            "    x: int = 5\n"
            "    return x"
        )
        results = self.ext.extract(code)
        init_results = [
            r for r in results if r.constraint_type == "initialization"
        ]
        assert len(init_results) == 0

    def test_with_statement_assignment(self) -> None:
        """REQ-VERIFY-002: With-statement optional_vars are tracked as assigned."""
        code = (
            "def read_file(path: str) -> str:\n"
            "    with open(path) as f:\n"
            "        data = f.read()\n"
            "    return data"
        )
        results = self.ext.extract(code)
        init_results = [
            r for r in results if r.constraint_type == "initialization"
        ]
        # f, data, path are all assigned; open is a builtin
        uninitialized_vars = [r.metadata["variable"] for r in init_results]
        assert "f" not in uninitialized_vars
        assert "data" not in uninitialized_vars

    def test_aug_assign_in_body(self) -> None:
        """REQ-VERIFY-002: Augmented assignments (+=) are tracked as assigned."""
        code = (
            "def accumulate(items: list) -> int:\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        total += item\n"
            "    return total"
        )
        results = self.ext.extract(code)
        init_results = [
            r for r in results if r.constraint_type == "initialization"
        ]
        uninitialized_vars = [r.metadata["variable"] for r in init_results]
        assert "total" not in uninitialized_vars

    def test_complex_annotation_returns_none(self) -> None:
        """REQ-VERIFY-001: Complex annotations (subscripts) yield no type constraint."""
        code = "def f(x: list[int]) -> int:\n    return 0"
        results = self.ext.extract(code)
        type_results = [r for r in results if r.constraint_type == "type_check"]
        # list[int] is a Subscript node, not Name/Constant/Attribute
        assert len(type_results) == 0

    def test_if_statement_assignment_tracked(self) -> None:
        """REQ-VERIFY-002: Variables assigned inside if/else are tracked."""
        code = (
            "def f(x: int) -> int:\n"
            "    if x > 0:\n"
            "        result = x\n"
            "    else:\n"
            "        result = 0\n"
            "    return result"
        )
        results = self.ext.extract(code)
        init_results = [
            r for r in results if r.constraint_type == "initialization"
        ]
        uninitialized_vars = [r.metadata["variable"] for r in init_results]
        assert "result" not in uninitialized_vars


# ---------------------------------------------------------------------------
# LogicExtractor tests — REQ-VERIFY-001, SCENARIO-VERIFY-002
# ---------------------------------------------------------------------------


class TestLogicExtractor:
    """Tests for logical claim extraction."""

    def setup_method(self) -> None:
        self.ext = LogicExtractor()

    def test_if_then(self) -> None:
        """REQ-VERIFY-001: 'If P then Q' implication is extracted."""
        results = self.ext.extract("If it rains, then the ground is wet.")
        implications = [
            r for r in results if r.constraint_type == "implication"
        ]
        assert len(implications) == 1
        assert implications[0].metadata["antecedent"] == "it rains"
        assert implications[0].metadata["consequent"] == "the ground is wet"

    def test_if_comma(self) -> None:
        """REQ-VERIFY-001: 'If P, Q' pattern (comma separator) is extracted."""
        results = self.ext.extract("If it rains, the ground is wet.")
        implications = [
            r for r in results if r.constraint_type == "implication"
        ]
        assert len(implications) == 1

    def test_exclusion(self) -> None:
        """REQ-VERIFY-001: 'X but not Y' exclusion is extracted."""
        results = self.ext.extract("Mammals but not reptiles.")
        exclusions = [r for r in results if r.constraint_type == "exclusion"]
        assert len(exclusions) == 1
        assert exclusions[0].metadata["positive"] == "mammals"
        assert exclusions[0].metadata["negative"] == "reptiles"

    def test_disjunction(self) -> None:
        """REQ-VERIFY-001: 'X or Y' disjunction is extracted."""
        results = self.ext.extract("Either cats or dogs.")
        disj = [r for r in results if r.constraint_type == "disjunction"]
        assert len(disj) == 1

    def test_negation(self) -> None:
        """REQ-VERIFY-001: 'X cannot Y' negation is extracted."""
        results = self.ext.extract("Penguins cannot fly.")
        negs = [r for r in results if r.constraint_type == "negation"]
        assert len(negs) == 1
        assert negs[0].metadata["subject"] == "penguins"
        assert negs[0].metadata["predicate"] == "fly"

    def test_universal(self) -> None:
        """REQ-VERIFY-001: 'All X are Y' universal quantifier is extracted."""
        results = self.ext.extract("All mammals are warm-blooded.")
        univs = [r for r in results if r.constraint_type == "universal"]
        assert len(univs) == 1
        assert univs[0].metadata["category"] == "mammals"
        assert univs[0].metadata["property"] == "warm-blooded"

    def test_no_logic(self) -> None:
        """REQ-VERIFY-001: Text with no logical patterns returns empty."""
        results = self.ext.extract("The sky is blue.")
        assert results == []

    def test_empty_text(self) -> None:
        """REQ-VERIFY-001: Empty string returns empty list."""
        results = self.ext.extract("")
        assert results == []

    def test_multiple_sentences(self) -> None:
        """REQ-VERIFY-001: Multiple logical claims in one text."""
        text = "If A then B. If B then C."
        results = self.ext.extract(text)
        implications = [
            r for r in results if r.constraint_type == "implication"
        ]
        assert len(implications) == 2

    def test_domain_filter(self) -> None:
        """SCENARIO-VERIFY-002: Domain filter skips non-matching domains."""
        results = self.ext.extract("If A then B.", domain="code")
        assert results == []

    def test_supported_domains(self) -> None:
        """REQ-VERIFY-001: supported_domains includes logic."""
        assert "logic" in self.ext.supported_domains


# ---------------------------------------------------------------------------
# NLExtractor tests — REQ-VERIFY-001, SCENARIO-VERIFY-002
# ---------------------------------------------------------------------------


class TestNLExtractor:
    """Tests for natural language factual claim extraction."""

    def setup_method(self) -> None:
        self.ext = NLExtractor()

    def test_factual_relation(self) -> None:
        """REQ-VERIFY-001: 'X is the Y of Z' relation is extracted."""
        results = self.ext.extract("Paris is the capital of France.")
        rels = [r for r in results if r.constraint_type == "factual_relation"]
        assert len(rels) == 1
        assert rels[0].metadata["subject"] == "paris"
        assert rels[0].metadata["relation"] == "capital"
        assert rels[0].metadata["object"] == "france"

    def test_factual_is(self) -> None:
        """REQ-VERIFY-001: 'X is/are Y' factual claim is extracted."""
        results = self.ext.extract("Whales are mammals.")
        facts = [r for r in results if r.constraint_type == "factual"]
        assert len(facts) == 1
        assert facts[0].metadata["subject"] == "whales"
        assert facts[0].metadata["predicate"] == "mammals"

    def test_factual_relation_not_duplicated(self) -> None:
        """REQ-VERIFY-002: 'X is the Y of Z' does not also produce 'X is ...'."""
        results = self.ext.extract("Paris is the capital of France.")
        facts = [r for r in results if r.constraint_type == "factual"]
        assert len(facts) == 0  # Only factual_relation, not also factual

    def test_quantity_there_are(self) -> None:
        """REQ-VERIFY-001: 'There are N X' quantity claims are extracted."""
        results = self.ext.extract("There are 5 apples.")
        quants = [r for r in results if r.constraint_type == "quantity"]
        assert len(quants) == 1
        assert quants[0].metadata["quantity"] == 5

    def test_quantity_has(self) -> None:
        """REQ-VERIFY-001: 'X has N Y' quantity claims are extracted."""
        results = self.ext.extract("The basket has 10 oranges.")
        quants = [r for r in results if r.constraint_type == "quantity"]
        assert len(quants) == 1
        assert quants[0].metadata["quantity"] == 10

    def test_no_factual_claims(self) -> None:
        """REQ-VERIFY-001: Text with no factual patterns returns empty."""
        results = self.ext.extract("Hello world!")
        assert results == []

    def test_empty_text(self) -> None:
        """REQ-VERIFY-001: Empty string returns empty list."""
        results = self.ext.extract("")
        assert results == []

    def test_multiple_claims(self) -> None:
        """REQ-VERIFY-001: Multiple factual claims in one text."""
        text = "Paris is the capital of France. Berlin is the capital of Germany."
        results = self.ext.extract(text)
        rels = [r for r in results if r.constraint_type == "factual_relation"]
        assert len(rels) == 2

    def test_domain_filter(self) -> None:
        """SCENARIO-VERIFY-002: Domain filter skips non-matching domains."""
        results = self.ext.extract("Paris is the capital of France.", domain="code")
        assert results == []

    def test_supported_domains(self) -> None:
        """REQ-VERIFY-001: supported_domains includes nl."""
        assert "nl" in self.ext.supported_domains


# ---------------------------------------------------------------------------
# AutoExtractor tests — REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
# ---------------------------------------------------------------------------


class TestAutoExtractor:
    """Tests for the combined auto-detecting extractor."""

    def setup_method(self) -> None:
        self.ext = AutoExtractor()

    def test_combines_arithmetic_and_nl(self) -> None:
        """REQ-VERIFY-002: AutoExtractor combines results from multiple extractors."""
        text = "2 + 3 = 5. Paris is the capital of France."
        results = self.ext.extract(text)
        types = {r.constraint_type for r in results}
        assert "arithmetic" in types
        assert "factual_relation" in types

    def test_combines_logic_and_nl(self) -> None:
        """REQ-VERIFY-002: Logic and NL extractors both contribute."""
        text = "If it rains, the ground is wet. Whales are mammals."
        results = self.ext.extract(text)
        types = {r.constraint_type for r in results}
        assert "implication" in types
        assert "factual" in types

    def test_code_extraction(self) -> None:
        """SCENARIO-VERIFY-002: Code blocks in mixed text are parsed."""
        text = (
            "Here is code:\n"
            "```python\n"
            "def add(x: int) -> int:\n"
            "    return x\n"
            "```\n"
            "And 2 + 3 = 5."
        )
        results = self.ext.extract(text)
        types = {r.constraint_type for r in results}
        assert "type_check" in types
        assert "arithmetic" in types

    def test_domain_filter(self) -> None:
        """SCENARIO-VERIFY-002: Domain filter restricts to specific extractor."""
        text = "2 + 3 = 5. Paris is the capital of France."
        arith_only = self.ext.extract(text, domain="arithmetic")
        assert all(r.constraint_type == "arithmetic" for r in arith_only)

    def test_empty_text(self) -> None:
        """REQ-VERIFY-001: Empty text returns empty from all extractors."""
        results = self.ext.extract("")
        assert results == []

    def test_no_constraints(self) -> None:
        """REQ-VERIFY-001: Text with no recognizable patterns returns empty."""
        results = self.ext.extract("Hello world!")
        assert results == []

    def test_deduplication(self) -> None:
        """REQ-VERIFY-002: Same constraint from multiple extractors is deduplicated."""
        text = "All mammals are warm-blooded."
        results = self.ext.extract(text)
        descriptions = [r.description for r in results]
        assert len(descriptions) == len(set(descriptions))

    def test_supported_domains_union(self) -> None:
        """REQ-VERIFY-001: supported_domains is the union of all extractors."""
        domains = self.ext.supported_domains
        assert "arithmetic" in domains
        assert "code" in domains
        assert "logic" in domains
        assert "nl" in domains

    def test_add_extractor(self) -> None:
        """REQ-VERIFY-002: Custom extractors can be registered."""
        class CustomExtractor:
            @property
            def supported_domains(self) -> list[str]:
                return ["custom"]

            def extract(
                self, text: str, domain: str | None = None
            ) -> list[ConstraintResult]:
                if domain is not None and domain != "custom":
                    return []
                return [
                    ConstraintResult(
                        constraint_type="custom",
                        description="custom constraint",
                        metadata={"text": text},
                    )
                ]

        self.ext.add_extractor(CustomExtractor())
        assert "custom" in self.ext.supported_domains
        results = self.ext.extract("anything", domain="custom")
        assert len(results) == 1
        assert results[0].constraint_type == "custom"

    def test_malformed_input(self) -> None:
        """REQ-VERIFY-001: Malformed input does not crash any extractor."""
        # Mix of garbage, partial code, partial arithmetic.
        text = "def (\n 2 + = \n !@#$%"
        results = self.ext.extract(text)
        # Should return whatever it can parse without crashing.
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Protocol conformance tests — REQ-VERIFY-001
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify all extractor classes satisfy the ConstraintExtractor Protocol."""

    @pytest.mark.parametrize(
        "cls",
        [ArithmeticExtractor, CodeExtractor, LogicExtractor, NLExtractor, AutoExtractor],
    )
    def test_is_constraint_extractor(self, cls: type) -> None:
        """REQ-VERIFY-001: All extractors satisfy ConstraintExtractor Protocol."""
        instance = cls()
        assert isinstance(instance, ConstraintExtractor)


# ---------------------------------------------------------------------------
# ConstraintResult tests — REQ-VERIFY-001
# ---------------------------------------------------------------------------


class TestConstraintResult:
    """Tests for the ConstraintResult dataclass."""

    def test_default_fields(self) -> None:
        """REQ-VERIFY-001: Default energy_term is None, metadata is empty dict."""
        result = ConstraintResult(
            constraint_type="test", description="a test"
        )
        assert result.energy_term is None
        assert result.metadata == {}

    def test_with_metadata(self) -> None:
        """REQ-VERIFY-001: Metadata is stored correctly."""
        result = ConstraintResult(
            constraint_type="arithmetic",
            description="2 + 3 = 5",
            metadata={"a": 2, "b": 3, "satisfied": True},
        )
        assert result.metadata["a"] == 2
        assert result.metadata["satisfied"] is True
