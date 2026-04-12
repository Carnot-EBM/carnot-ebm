"""Tests for carnot.pipeline.z3_extractor.

Each test references REQ-VERIFY-* or SCENARIO-VERIFY-* per spec-anchored
development requirements.

Spec: REQ-VERIFY-001, REQ-VERIFY-003, REQ-VERIFY-009, SCENARIO-VERIFY-009
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from z3 import Sqrt

from carnot.pipeline.verify_repair import VerifyRepairPipeline
from carnot.pipeline.z3_extractor import Z3ArithmeticExtractor


def _load_exp203_results() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    results_path = repo_root / "results" / "experiment_203_results.json"
    return json.loads(results_path.read_text())


class TestZ3ArithmeticExtractor:
    """Tests for the SMT-backed arithmetic extractor."""

    def setup_method(self) -> None:
        self.ext = Z3ArithmeticExtractor()

    def test_supported_domains(self) -> None:
        """REQ-VERIFY-001: supported_domains includes arithmetic."""
        assert "arithmetic" in self.ext.supported_domains

    def test_domain_filter_skips_non_matching_domain(self) -> None:
        """SCENARIO-VERIFY-009: Non-arithmetic domain hints are ignored."""
        assert self.ext.extract("2 + 3 = 5", domain="code") == []

    def test_exact_equation_is_satisfiable(self) -> None:
        """REQ-VERIFY-009: Exact arithmetic equations are solver-checked."""
        results = self.ext.extract("47 + 28 = 75")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True
        assert results[0].metadata["solver"] == "z3"
        assert results[0].metadata["correct_result"] == 75

    def test_exact_equation_unsat_reports_correction(self) -> None:
        """REQ-VERIFY-009: Wrong arithmetic returns solver-derived correction."""
        results = self.ext.extract("47 + 28 = 76")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is False
        assert results[0].metadata["correct_result"] == 75
        assert results[0].metadata["claimed_result"] == 76
        assert results[0].metadata["step_text"] == "47 + 28 = 76"

    def test_implicit_half_statement_is_supported(self) -> None:
        """REQ-VERIFY-009: Verbal arithmetic like 'half of 48' is encoded."""
        results = self.ext.extract("Half of 48 is 24.")
        assert len(results) == 1
        assert results[0].metadata["operator"] == "half_of"
        assert results[0].metadata["satisfied"] is True
        assert results[0].metadata["correct_result"] == 24

    def test_wrong_implicit_half_statement_is_unsat(self) -> None:
        """REQ-VERIFY-009: Wrong verbal arithmetic becomes UNSAT."""
        results = self.ext.extract("Half of 48 is 20.")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is False
        assert results[0].metadata["correct_result"] == 24

    def test_compound_expression_and_latex_fraction_are_supported(self) -> None:
        """REQ-VERIFY-009: Compound arithmetic and LaTeX fractions are parsed."""
        text = (
            "Number of cards with A = $80 \\times \\frac{2}{5} = 32$ cards.\n"
            "Then $\\frac{160}{5} = 32$."
        )
        results = self.ext.extract(text)
        assert len(results) == 2
        assert all(result.metadata["satisfied"] is True for result in results)

    def test_approximate_statement_uses_bounded_range(self) -> None:
        """REQ-VERIFY-009: Approximate claims are checked with tolerance bounds."""
        results = self.ext.extract("49 + 99 is about 150.")
        assert len(results) == 1
        assert results[0].metadata["approximate"] is True
        assert results[0].metadata["tolerance"] == 5.0
        assert results[0].metadata["satisfied"] is True
        assert results[0].metadata["correct_result"] == 148

    def test_approximate_statement_can_still_be_unsat(self) -> None:
        """REQ-VERIFY-009: Approximate bounds still reject distant claims."""
        results = self.ext.extract("49 + 99 is about 160.")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is False
        assert results[0].metadata["correct_result"] == 148

    def test_multiple_steps_return_multiple_results(self) -> None:
        """SCENARIO-VERIFY-009: Multi-step chains emit one result per step."""
        text = "First: 2 + 3 = 5.\nThen: 5 * 4 = 20.\nFinally: about 20 + 1 is 21."
        results = self.ext.extract(text)
        assert len(results) == 3
        assert [result.metadata["step_index"] for result in results] == [1, 2, 3]

    def test_unsupported_expression_is_ignored(self) -> None:
        """REQ-VERIFY-009: Unsupported symbolic forms do not crash extraction."""
        assert self.ext.extract("sqrt(9) = 3") == []

    def test_empty_text_returns_no_constraints(self) -> None:
        """REQ-VERIFY-003: Empty input yields no constraints."""
        assert self.ext.extract("") == []

    def test_text_that_normalizes_to_empty_returns_no_constraints(self) -> None:
        """REQ-VERIFY-003: Formatting-only input is discarded after normalization."""
        assert self.ext.extract("$$$") == []

    def test_verify_repair_pipeline_accepts_custom_z3_extractor(self) -> None:
        """REQ-VERIFY-003: VerifyRepairPipeline consumes metadata-backed results."""
        pipeline = VerifyRepairPipeline(extractor=self.ext)
        result = pipeline.verify(question="What is 47 + 28?", response="47 + 28 = 76")
        assert result.verified is False
        assert len(result.violations) == 1
        assert result.violations[0].metadata["correct_result"] == 75

    def test_exp203_correct_examples_have_zero_false_positives(self) -> None:
        """SCENARIO-VERIFY-009: Correct Exp 203 contrast cases stay SAT."""
        data = _load_exp203_results()
        showcase = data["correct_answer_examples"][:3]

        for case in showcase:
            results = self.ext.extract(case["response"])
            assert results
            assert all(result.metadata["satisfied"] for result in results)

    def test_exp203_wrong_cases_remain_response_internal_when_sat(self) -> None:
        """REQ-VERIFY-009: Exp 203 wrong cases still extract internal arithmetic facts."""
        data = _load_exp203_results()
        cases = {case["dataset_idx"]: case for case in data["wrong_answer_autopsies"]}

        lana_results = self.ext.extract(cases[923]["response"])
        arcade_results = self.ext.extract(cases[814]["response"])
        cds_results = self.ext.extract(cases[943]["response"])

        assert len(lana_results) >= 3
        assert any(result.metadata["operator"] == "/" for result in lana_results)
        assert len(arcade_results) >= 6
        assert any(result.metadata["operator"] == "*" for result in arcade_results)
        assert len(cds_results) >= 4
        assert any(result.metadata["operator"] == "-" for result in cds_results)

    def test_unsupported_power_operator_is_ignored(self) -> None:
        """REQ-VERIFY-009: Unsupported AST operators are skipped cleanly."""
        assert self.ext.extract("2 ** 3 = 8") == []

    def test_parse_expression_syntax_error_returns_none(self) -> None:
        """REQ-VERIFY-009: Invalid math syntax is rejected before solving."""
        assert self.ext._parse_expression("2 +") is None

    def test_ast_to_z3_rejects_non_numeric_constants(self) -> None:
        """REQ-VERIFY-009: Non-numeric AST constants are not converted to Z3."""
        assert self.ext._ast_to_z3(ast.parse("True", mode="eval").body) is None
        assert self.ext._ast_to_z3(ast.parse("'hello'", mode="eval").body) is None

    def test_ast_to_z3_rejects_unknown_unary_and_binary_nodes(self) -> None:
        """REQ-VERIFY-009: Symbolic AST nodes without numeric grounding return None."""
        assert self.ext._ast_to_z3(ast.parse("-flag", mode="eval").body) is None
        assert self.ext._ast_to_z3(ast.parse("1 + flag", mode="eval").body) is None
        negative_three = self.ext._ast_to_z3(ast.parse("-3", mode="eval").body)
        assert negative_three is not None
        assert str(negative_three) == "-3"

    def test_helper_formatting_and_algebraic_conversion(self) -> None:
        """REQ-VERIFY-009: Helper formatting handles compound and algebraic values."""
        assert self.ext._infer_operator("42") == "compound"
        assert self.ext._parse_number("42") == 42
        assert self.ext._format_number(4.0) == "4"
        assert self.ext._format_number(4.25) == "4.25"
        sqrt_two = self.ext._z3_value_to_number(Sqrt(2))
        assert isinstance(sqrt_two, float)
        assert 1.41 < sqrt_two < 1.42
