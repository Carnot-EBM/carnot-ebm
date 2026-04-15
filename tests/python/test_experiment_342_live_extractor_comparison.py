"""Tests for scripts/experiment_342_live_extractor_comparison.py.

Covers 100% of the experiment script and the extractor_comparison library:

Library (carnot.pipeline.extractor_comparison):
  - ExtractorResult dataclass: construction, field types
  - _estimate_fp_categories: empty, zero violations, no prior, zero prior total
  - _compute_estimated_precision: zero violations, nonzero, penalty clamping
  - _run_extractor_fn: normal, exception-safe, empty-list extractor
  - compare_extractors: zero responses, single extractor, multiple extractors, fp_prior default
  - build_comparison_artifact: empty results, single, tie-break, best_precision

Script (experiment_342):
  - _load_exp340_responses: missing file, partial file, valid list, dict items, empty list
  - _synthetic_responses: length, cycle repeats, all strings
  - _load_fp_prior: missing file, valid file, malformed, empty dist
  - _build_extractors: returns 4 pairs, each callable
  - build_agreement_matrix: zero responses, self-agreement=1.0, both-flag, one-flags
  - main(): simulated mode produces valid artifact on disk

All tests run under CARNOT_FORCE_LIVE=0 (no GPU, no real LLM calls).

Spec: REQ-EXTRACT-017, SCENARIO-EXTRACT-036, SCENARIO-EXTRACT-037
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "experiment_342_live_extractor_comparison.py"

for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("CARNOT_FORCE_LIVE", "0")


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_script() -> Any:
    """Load experiment_342 as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("experiment_342", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_342"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()

_load_exp340_responses = _mod._load_exp340_responses
_synthetic_responses = _mod._synthetic_responses
_load_fp_prior = _mod._load_fp_prior
_build_extractors = _mod._build_extractors
build_agreement_matrix = _mod.build_agreement_matrix
main = _mod.main

# Library imports
from carnot.pipeline.extractor_comparison import (  # noqa: E402
    ExtractorResult,
    _compute_estimated_precision,
    _estimate_fp_categories,
    _run_extractor_fn,
    build_comparison_artifact,
    compare_extractors,
)


# ===========================================================================
# Library: ExtractorResult
# ===========================================================================


class TestExtractorResult:
    def test_construction(self):
        r = ExtractorResult(
            extractor_name="ArithmeticExtractor",
            n_responses_checked=50,
            n_violations_found=10,
            violation_rate=0.2,
            estimated_precision=0.8,
            fp_categories={"VALID_INTERMEDIATE": 2},
        )
        assert r.extractor_name == "ArithmeticExtractor"
        assert r.n_responses_checked == 50
        assert r.n_violations_found == 10
        assert r.violation_rate == 0.2
        assert r.estimated_precision == 0.8
        assert r.fp_categories == {"VALID_INTERMEDIATE": 2}

    def test_default_fp_categories(self):
        r = ExtractorResult(
            extractor_name="X",
            n_responses_checked=5,
            n_violations_found=0,
            violation_rate=0.0,
            estimated_precision=1.0,
        )
        assert r.fp_categories == {}

    def test_field_types(self):
        r = ExtractorResult(
            extractor_name="X",
            n_responses_checked=10,
            n_violations_found=3,
            violation_rate=0.3,
            estimated_precision=0.7,
        )
        assert isinstance(r.extractor_name, str)
        assert isinstance(r.n_responses_checked, int)
        assert isinstance(r.n_violations_found, int)
        assert isinstance(r.violation_rate, float)
        assert isinstance(r.estimated_precision, float)
        assert isinstance(r.fp_categories, dict)


# ===========================================================================
# Library: _estimate_fp_categories
# ===========================================================================


class TestEstimateFpCategories:
    def test_zero_violations(self):
        assert _estimate_fp_categories(0, {"VALID_INTERMEDIATE": 2}) == {}

    def test_no_prior_returns_uncategorized(self):
        result = _estimate_fp_categories(5, {})
        assert result == {"UNCATEGORIZED": 5}

    def test_zero_prior_total_returns_uncategorized(self):
        result = _estimate_fp_categories(3, {"VALID_INTERMEDIATE": 0})
        assert result == {"UNCATEGORIZED": 3}

    def test_single_category_all_violations(self):
        result = _estimate_fp_categories(4, {"VALID_INTERMEDIATE": 1})
        assert result.get("VALID_INTERMEDIATE") == 4

    def test_multi_category_sums_to_n_violations(self):
        prior = {"VALID_INTERMEDIATE": 2, "PRECISION_LIMIT": 1, "REGEX_ARTIFACT": 1}
        n = 8
        result = _estimate_fp_categories(n, prior)
        assert sum(result.values()) == n

    def test_standard_prior_sums_correctly(self):
        prior = {
            "VALID_INTERMEDIATE": 2,
            "PRECISION_LIMIT": 1,
            "REGEX_ARTIFACT": 1,
            "REPAIR_DEGRADATION": 2,
            "UNCATEGORIZED": 0,
        }
        n = 10
        result = _estimate_fp_categories(n, prior)
        assert sum(result.values()) == n

    def test_one_violation_standard_prior(self):
        prior = {"VALID_INTERMEDIATE": 2, "PRECISION_LIMIT": 1}
        result = _estimate_fp_categories(1, prior)
        assert sum(result.values()) == 1


# ===========================================================================
# Library: _compute_estimated_precision
# ===========================================================================


class TestComputeEstimatedPrecision:
    def test_zero_violations_returns_one(self):
        assert _compute_estimated_precision({}, 0) == 1.0

    def test_all_high_fp_category(self):
        # REGEX_ARTIFACT has penalty 0.95 → estimated_precision ≈ 0.05
        fp_cats = {"REGEX_ARTIFACT": 10}
        prec = _compute_estimated_precision(fp_cats, 10)
        assert prec == pytest.approx(0.05, abs=0.01)

    def test_all_low_fp_category(self):
        # UNCATEGORIZED has penalty 0.30 → estimated_precision ≈ 0.70
        fp_cats = {"UNCATEGORIZED": 5}
        prec = _compute_estimated_precision(fp_cats, 5)
        assert prec == pytest.approx(0.70, abs=0.01)

    def test_precision_clamped_to_zero(self):
        # Penalty > 1.0 would give negative precision → clamp to 0.0
        fp_cats = {"REGEX_ARTIFACT": 100, "VALID_INTERMEDIATE": 100}
        prec = _compute_estimated_precision(fp_cats, 10)
        assert prec == 0.0

    def test_unknown_category_uses_default_penalty(self):
        # Unknown category uses 0.30 penalty
        fp_cats = {"MYSTERY_CATEGORY": 10}
        prec = _compute_estimated_precision(fp_cats, 10)
        assert prec == pytest.approx(0.70, abs=0.01)

    def test_mixed_categories(self):
        fp_cats = {"VALID_INTERMEDIATE": 5, "UNCATEGORIZED": 5}
        prec = _compute_estimated_precision(fp_cats, 10)
        # penalty = 5*0.90 + 5*0.30 = 4.5 + 1.5 = 6.0 / 10 = 0.60 fp_rate → 0.40 precision
        assert prec == pytest.approx(0.40, abs=0.01)


# ===========================================================================
# Library: _run_extractor_fn
# ===========================================================================


class TestRunExtractorFn:
    def test_normal_returns_list(self):
        fn = lambda resp: ["violation"]  # noqa: E731
        result = _run_extractor_fn(fn, "some text")
        assert result == ["violation"]

    def test_exception_returns_empty(self):
        def bad_fn(resp):
            raise RuntimeError("boom")
        result = _run_extractor_fn(bad_fn, "text")
        assert result == []

    def test_returns_none_coerced_to_empty(self):
        fn = lambda resp: None  # noqa: E731
        result = _run_extractor_fn(fn, "text")
        assert result == []

    def test_empty_list_returns_empty(self):
        fn = lambda resp: []  # noqa: E731
        result = _run_extractor_fn(fn, "text")
        assert result == []


# ===========================================================================
# Library: compare_extractors
# ===========================================================================


class TestCompareExtractors:
    def test_zero_responses_returns_zero_rates(self):
        fn = lambda r: ["v"]  # noqa: E731
        results = compare_extractors([], [("X", fn)])
        assert len(results) == 1
        assert results[0].violation_rate == 0.0
        assert results[0].n_responses_checked == 0

    def test_single_extractor_all_violations(self):
        fn = lambda r: ["v"]  # noqa: E731
        responses = ["r1", "r2", "r3"]
        results = compare_extractors(responses, [("X", fn)])
        assert results[0].n_violations_found == 3
        assert results[0].violation_rate == 1.0
        assert results[0].n_responses_checked == 3

    def test_single_extractor_no_violations(self):
        fn = lambda r: []  # noqa: E731
        results = compare_extractors(["r1", "r2"], [("X", fn)])
        assert results[0].n_violations_found == 0
        assert results[0].violation_rate == 0.0
        assert results[0].estimated_precision == 1.0

    def test_multiple_extractors_returned_in_order(self):
        fn_a = lambda r: ["v"]  # noqa: E731
        fn_b = lambda r: []  # noqa: E731
        results = compare_extractors(["r1"], [("A", fn_a), ("B", fn_b)])
        assert results[0].extractor_name == "A"
        assert results[1].extractor_name == "B"
        assert results[0].n_violations_found == 1
        assert results[1].n_violations_found == 0

    def test_default_fp_prior_used_when_none(self):
        fn = lambda r: ["v"]  # noqa: E731
        results = compare_extractors(["r1", "r2"], [("X", fn)], fp_prior=None)
        assert results[0].estimated_precision >= 0.0

    def test_custom_fp_prior_used(self):
        fn = lambda r: ["v"]  # noqa: E731
        prior = {"REGEX_ARTIFACT": 10}  # high FP penalty
        results = compare_extractors(["r1"], [("X", fn)], fp_prior=prior)
        # REGEX_ARTIFACT has penalty 0.95 → precision ≈ 0.05
        assert results[0].estimated_precision < 0.20

    def test_violation_rate_fraction(self):
        calls = [0]
        def fn(r):
            calls[0] += 1
            return ["v"] if calls[0] % 2 == 1 else []
        results = compare_extractors(["r1", "r2", "r3", "r4"], [("X", fn)])
        # 4 calls: 1st, 3rd flagged → 2 violations
        assert results[0].n_violations_found == 2
        assert results[0].violation_rate == pytest.approx(0.5, abs=0.01)


# ===========================================================================
# Library: build_comparison_artifact
# ===========================================================================


class TestBuildComparisonArtifact:
    def test_empty_results(self):
        art = build_comparison_artifact([])
        assert art["comparison_schema"] == "carnot.extractor_comparison.v1"
        assert art["best_precision"] == 0.0
        assert art["recommended_extractor"] == ""
        assert art["extractor_results"] == []

    def test_single_result(self):
        r = ExtractorResult(
            extractor_name="ArithmeticExtractor",
            n_responses_checked=10,
            n_violations_found=2,
            violation_rate=0.2,
            estimated_precision=0.7,
        )
        art = build_comparison_artifact([r])
        assert art["recommended_extractor"] == "ArithmeticExtractor"
        assert art["best_precision"] == 0.7
        assert art["comparison_schema"] == "carnot.extractor_comparison.v1"

    def test_highest_precision_selected(self):
        r_low = ExtractorResult(
            extractor_name="Low",
            n_responses_checked=10,
            n_violations_found=5,
            violation_rate=0.5,
            estimated_precision=0.3,
        )
        r_high = ExtractorResult(
            extractor_name="High",
            n_responses_checked=10,
            n_violations_found=2,
            violation_rate=0.2,
            estimated_precision=0.9,
        )
        art = build_comparison_artifact([r_low, r_high])
        assert art["recommended_extractor"] == "High"
        assert art["best_precision"] == 0.9

    def test_tie_break_first_wins(self):
        r_a = ExtractorResult(
            extractor_name="First",
            n_responses_checked=10,
            n_violations_found=0,
            violation_rate=0.0,
            estimated_precision=1.0,
        )
        r_b = ExtractorResult(
            extractor_name="Second",
            n_responses_checked=10,
            n_violations_found=0,
            violation_rate=0.0,
            estimated_precision=1.0,
        )
        art = build_comparison_artifact([r_a, r_b])
        assert art["recommended_extractor"] == "First"

    def test_extractor_results_list_length(self):
        results = [
            ExtractorResult("A", 10, 1, 0.1, 0.9),
            ExtractorResult("B", 10, 2, 0.2, 0.8),
            ExtractorResult("C", 10, 0, 0.0, 1.0),
        ]
        art = build_comparison_artifact(results)
        assert len(art["extractor_results"]) == 3

    def test_extractor_results_fields(self):
        r = ExtractorResult(
            extractor_name="X",
            n_responses_checked=5,
            n_violations_found=1,
            violation_rate=0.2,
            estimated_precision=0.85,
            fp_categories={"VALID_INTERMEDIATE": 1},
        )
        art = build_comparison_artifact([r])
        entry = art["extractor_results"][0]
        assert entry["extractor_name"] == "X"
        assert entry["n_responses_checked"] == 5
        assert entry["n_violations_found"] == 1
        assert entry["violation_rate"] == 0.2
        assert entry["estimated_precision"] == 0.85
        assert entry["fp_categories"] == {"VALID_INTERMEDIATE": 1}


# ===========================================================================
# Script: _load_exp340_responses
# ===========================================================================


class TestLoadExp340Responses:
    def test_missing_file_returns_none(self, tmp_path):
        result = _load_exp340_responses(tmp_path / "nonexistent.json", 50)
        assert result is None

    def test_invalid_json_returns_none(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json")
        result = _load_exp340_responses(f, 50)
        assert result is None

    def test_no_responses_key_returns_none(self, tmp_path):
        f = tmp_path / "partial.json"
        f.write_text(json.dumps({"status": "partial", "finding": "blocked"}))
        result = _load_exp340_responses(f, 50)
        assert result is None

    def test_empty_responses_list_returns_none(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text(json.dumps({"responses": []}))
        result = _load_exp340_responses(f, 50)
        assert result is None

    def test_string_responses_returned(self, tmp_path):
        f = tmp_path / "valid.json"
        f.write_text(json.dumps({"responses": ["resp1", "resp2", "resp3"]}))
        result = _load_exp340_responses(f, 50)
        assert result == ["resp1", "resp2", "resp3"]

    def test_dict_responses_extracted(self, tmp_path):
        f = tmp_path / "dict.json"
        f.write_text(json.dumps({"responses": [{"response": "text_a"}, {"response": "text_b"}]}))
        result = _load_exp340_responses(f, 50)
        assert result == ["text_a", "text_b"]

    def test_dict_text_key_fallback(self, tmp_path):
        f = tmp_path / "text.json"
        f.write_text(json.dumps({"responses": [{"text": "t1"}]}))
        result = _load_exp340_responses(f, 50)
        assert result == ["t1"]

    def test_dict_answer_key_fallback(self, tmp_path):
        f = tmp_path / "answer.json"
        f.write_text(json.dumps({"responses": [{"answer": "a1"}]}))
        result = _load_exp340_responses(f, 50)
        assert result == ["a1"]

    def test_dict_no_known_key_returns_empty_string(self, tmp_path):
        f = tmp_path / "unknown.json"
        f.write_text(json.dumps({"responses": [{"unknown_key": "ignored"}]}))
        result = _load_exp340_responses(f, 50)
        # Falls back to str("") for unknown keys
        assert result == [""]

    def test_truncated_to_n(self, tmp_path):
        f = tmp_path / "big.json"
        f.write_text(json.dumps({"responses": [f"r{i}" for i in range(100)]}))
        result = _load_exp340_responses(f, 10)
        assert len(result) == 10


# ===========================================================================
# Script: _synthetic_responses
# ===========================================================================


class TestSyntheticResponses:
    def test_length(self):
        assert len(_synthetic_responses(50)) == 50

    def test_all_strings(self):
        for r in _synthetic_responses(20):
            assert isinstance(r, str)
            assert len(r) > 0

    def test_cycle_repeats(self):
        r20 = _synthetic_responses(20)
        r40 = _synthetic_responses(40)
        # Second cycle should match first cycle.
        assert r40[20] == r40[0]

    def test_zero_returns_empty(self):
        assert _synthetic_responses(0) == []

    def test_single_response(self):
        r = _synthetic_responses(1)
        assert len(r) == 1


# ===========================================================================
# Script: _load_fp_prior
# ===========================================================================


class TestLoadFpPrior:
    def test_missing_file_returns_default(self, tmp_path):
        result = _load_fp_prior(tmp_path / "missing.json")
        assert "VALID_INTERMEDIATE" in result
        assert isinstance(result["VALID_INTERMEDIATE"], int)

    def test_valid_file_loaded(self, tmp_path):
        f = tmp_path / "fp.json"
        f.write_text(json.dumps({
            "category_distribution": {
                "VALID_INTERMEDIATE": 5,
                "REGEX_ARTIFACT": 3,
            }
        }))
        result = _load_fp_prior(f)
        assert result["VALID_INTERMEDIATE"] == 5
        assert result["REGEX_ARTIFACT"] == 3

    def test_malformed_json_returns_default(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid")
        result = _load_fp_prior(f)
        assert "VALID_INTERMEDIATE" in result

    def test_no_category_distribution_key_returns_default(self, tmp_path):
        f = tmp_path / "no_key.json"
        f.write_text(json.dumps({"status": "ok"}))
        result = _load_fp_prior(f)
        assert "VALID_INTERMEDIATE" in result

    def test_empty_distribution_returns_default(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text(json.dumps({"category_distribution": {}}))
        result = _load_fp_prior(f)
        assert "VALID_INTERMEDIATE" in result


# ===========================================================================
# Script: _build_extractors
# ===========================================================================


class TestBuildExtractors:
    def test_returns_four_pairs(self):
        extractors = _build_extractors(force_live=False)
        assert len(extractors) == 4

    def test_all_named(self):
        extractors = _build_extractors(force_live=False)
        names = [name for name, _ in extractors]
        assert "ArithmeticExtractor" in names
        assert "NL2Z3Extractor" in names
        assert "VergeRefiner" in names
        assert "CoTCircuitVerifier" in names

    def test_all_callable(self):
        extractors = _build_extractors(force_live=False)
        for name, fn in extractors:
            assert callable(fn), f"{name} extractor is not callable"

    def test_arithmetic_extractor_finds_wrong_arithmetic(self):
        extractors = _build_extractors(force_live=False)
        arith_fn = next(fn for name, fn in extractors if name == "ArithmeticExtractor")
        # "47 + 28 = 76" is wrong (correct is 75)
        result = arith_fn("The total is 47 + 28 = 76.")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_arithmetic_extractor_passes_correct(self):
        extractors = _build_extractors(force_live=False)
        arith_fn = next(fn for name, fn in extractors if name == "ArithmeticExtractor")
        # "10 + 5 = 15" is correct → no violations
        result = arith_fn("We have 10 + 5 = 15 items.")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_crv_extractor_returns_list(self):
        extractors = _build_extractors(force_live=False)
        crv_fn = next(fn for name, fn in extractors if name == "CoTCircuitVerifier")
        result = crv_fn("Step 1: we have 10. Step 2: from step 1, we get 10.")
        assert isinstance(result, list)

    def test_nl2z3_returns_list_in_ci_mode(self):
        extractors = _build_extractors(force_live=False)
        nl2z3_fn = next(fn for name, fn in extractors if name == "NL2Z3Extractor")
        result = nl2z3_fn("x + y = z")
        assert isinstance(result, list)

    def test_verge_returns_list_in_ci_mode(self):
        extractors = _build_extractors(force_live=False)
        verge_fn = next(fn for name, fn in extractors if name == "VergeRefiner")
        result = verge_fn("some response")
        assert isinstance(result, list)


# ===========================================================================
# Script: build_agreement_matrix
# ===========================================================================


class TestBuildAgreementMatrix:
    def test_empty_responses(self):
        fn = lambda r: ["v"]  # noqa: E731
        result = build_agreement_matrix([], [("A", fn)])
        assert result["extractor_names"] == ["A"]
        assert result["agreement_matrix"]["A"]["A"] == 1.0

    def test_self_agreement_always_one(self):
        fn = lambda r: ["v"]  # noqa: E731
        result = build_agreement_matrix(["r1", "r2", "r3"], [("A", fn)])
        assert result["agreement_matrix"]["A"]["A"] == 1.0

    def test_both_flag_same_responses(self):
        fn_a = lambda r: ["v"]  # noqa: E731
        fn_b = lambda r: ["v"]  # noqa: E731
        result = build_agreement_matrix(["r1", "r2"], [("A", fn_a), ("B", fn_b)])
        # Both flag all responses → 100% agreement
        assert result["agreement_matrix"]["A"]["B"] == 1.0

    def test_one_flags_none_flags(self):
        fn_a = lambda r: ["v"]  # noqa: E731
        fn_b = lambda r: []  # noqa: E731
        result = build_agreement_matrix(["r1", "r2", "r3"], [("A", fn_a), ("B", fn_b)])
        # A flags all 3, B flags none → 0% agreement
        assert result["agreement_matrix"]["A"]["B"] == 0.0

    def test_partial_agreement(self):
        # A flags r1, r2. B flags r2, r3.  Agree on r2 (both flag) + neither flags r0 ≠ 3
        # 3 responses: r1, r2, r3
        calls_a = {"n": 0}
        calls_b = {"n": 0}

        def fn_a(r):
            calls_a["n"] += 1
            return ["v"] if calls_a["n"] <= 2 else []

        def fn_b(r):
            calls_b["n"] += 1
            return ["v"] if calls_b["n"] >= 2 else []

        result = build_agreement_matrix(["r1", "r2", "r3"], [("A", fn_a), ("B", fn_b)])
        # A: [T, T, F], B: [F, T, T]
        # r1: A=T, B=F → disagree; r2: A=T, B=T → agree; r3: A=F, B=T → disagree
        assert result["agreement_matrix"]["A"]["B"] == pytest.approx(1 / 3, abs=0.01)

    def test_matrix_is_symmetric(self):
        fn_a = lambda r: ["v"] if "flag" in r else []  # noqa: E731
        fn_b = lambda r: ["v"] if "flag" in r else []  # noqa: E731
        responses = ["flag_a", "other", "flag_a"]
        result = build_agreement_matrix(responses, [("A", fn_a), ("B", fn_b)])
        assert result["agreement_matrix"]["A"]["B"] == result["agreement_matrix"]["B"]["A"]

    def test_returns_extractor_names(self):
        fn = lambda r: []  # noqa: E731
        result = build_agreement_matrix(["r1"], [("X", fn), ("Y", fn)])
        assert set(result["extractor_names"]) == {"X", "Y"}


# ===========================================================================
# Script: main()
# ===========================================================================


def _safe_extractors(force_live: bool) -> list[tuple[str, Any]]:
    """Return CI-safe (no LLM) extractors for main() tests."""
    from carnot.pipeline.extract import ArithmeticExtractor  # noqa: PLC0415
    from carnot.pipeline.cot_circuit_verifier import CoTCircuitVerifier  # noqa: PLC0415

    arith = ArithmeticExtractor()
    crv = CoTCircuitVerifier()

    return [
        ("ArithmeticExtractor",
         lambda r: [v for v in arith.extract(r, "arithmetic") if not v.metadata.get("satisfied", True)]),
        ("NL2Z3Extractor", lambda r: []),  # CI-safe no-op
        ("VergeRefiner", lambda r: []),    # CI-safe no-op
        ("CoTCircuitVerifier", lambda r: crv.extract("", r, "reasoning")),
    ]


class TestMain:
    def test_main_simulated_mode_writes_artifact(self, tmp_path):
        """Main runs in CI mode (CARNOT_FORCE_LIVE=0), writes valid artifact."""
        output_path = tmp_path / "results" / "experiment_342_live_extractor_comparison.json"

        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}),
            patch.object(_mod, "_DELIVERABLE", str(output_path.relative_to(tmp_path))),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_EXP_340_RESULTS", tmp_path / "exp340.json"),
            patch.object(_mod, "_EXP_331_RESULTS", tmp_path / "exp331.json"),
            patch.object(_mod, "_build_extractors", _safe_extractors),
        ):
            # Write a minimal exp331 file so _load_fp_prior succeeds.
            (tmp_path).mkdir(exist_ok=True)
            (tmp_path / "exp331.json").write_text(json.dumps({
                "category_distribution": {"VALID_INTERMEDIATE": 2, "REGEX_ARTIFACT": 1}
            }))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            main()

        assert output_path.exists()
        with open(output_path) as f:
            art = json.load(f)

        assert art["comparison_schema"] == "carnot.extractor_comparison.v1"
        assert "recommended_extractor" in art
        assert "best_precision" in art
        assert "inference_mode" in art
        assert art["inference_mode"] == "simulated"
        assert "agreement" in art
        assert "extractor_results" in art
        assert len(art["extractor_results"]) == 4

    def test_main_artifact_has_exp311_comparison(self, tmp_path):
        output_path = tmp_path / "results" / "experiment_342_live_extractor_comparison.json"

        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}),
            patch.object(_mod, "_DELIVERABLE", str(output_path.relative_to(tmp_path))),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_EXP_340_RESULTS", tmp_path / "exp340.json"),
            patch.object(_mod, "_EXP_331_RESULTS", tmp_path / "exp331.json"),
            patch.object(_mod, "_build_extractors", _safe_extractors),
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            main()

        with open(output_path) as f:
            art = json.load(f)

        assert "exp311_comparison" in art
        assert art["exp311_comparison"]["exp311_winner"] == "ArithmeticExtractor"

    def test_main_uses_live_exp340_when_force_live(self, tmp_path):
        """When CARNOT_FORCE_LIVE=1 and exp340 has responses, use them."""
        output_path = tmp_path / "results" / "experiment_342_live_extractor_comparison.json"
        exp340_path = tmp_path / "exp340.json"
        exp340_path.write_text(json.dumps({
            "responses": [f"response {i}" for i in range(50)]
        }))

        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(_mod, "_DELIVERABLE", str(output_path.relative_to(tmp_path))),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_EXP_340_RESULTS", exp340_path),
            patch.object(_mod, "_EXP_331_RESULTS", tmp_path / "exp331.json"),
            patch.object(_mod, "_build_extractors", _safe_extractors),
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            main()

        with open(output_path) as f:
            art = json.load(f)

        assert art["inference_mode"] == "live_exp340"
        assert art["n_responses"] == 50

    def test_main_falls_back_to_synthetic_when_exp340_missing(self, tmp_path):
        """When CARNOT_FORCE_LIVE=1 but exp340 is missing, fall back to synthetic."""
        output_path = tmp_path / "results" / "experiment_342_live_extractor_comparison.json"

        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(_mod, "_DELIVERABLE", str(output_path.relative_to(tmp_path))),
            patch.object(_mod, "_REPO_ROOT", tmp_path),
            patch.object(_mod, "_EXP_340_RESULTS", tmp_path / "missing_exp340.json"),
            patch.object(_mod, "_EXP_331_RESULTS", tmp_path / "missing_exp331.json"),
            patch.object(_mod, "_build_extractors", _safe_extractors),
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            main()

        with open(output_path) as f:
            art = json.load(f)

        assert art["inference_mode"] == "simulated"
