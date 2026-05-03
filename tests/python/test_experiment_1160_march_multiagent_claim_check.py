"""Tests for Exp 1160 MARCH blinded multi-agent claim checking.

Spec: REQ-VERIFY-1160, SCENARIO-VERIFY-1160.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.eval.march_multiagent_claim_check as march
from carnot.eval.march_multiagent_claim_check import (
    ALLOWED_HONEST_VERDICTS,
    REQUIRED_ARTIFACT_FIELDS,
    AtomicClaim,
    check_claim_blinded,
    extract_atomic_claims,
    honest_verdict,
    run_experiment,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DELIVERABLE = _REPO_ROOT / "results" / "experiment_1160_march_multiagent_claim_check.json"


def test_extractor_returns_two_to_four_atomic_claims_for_numeric_response() -> None:
    """REQ-VERIFY-1160: Proposer extracts 2-4 local atomic claims without an LLM."""

    claims = extract_atomic_claims("47 + 28 = 76, so the answer is 76.")

    assert 2 <= len(claims) <= 4
    assert any(claim.kind == "arithmetic" for claim in claims)
    assert any("76" in claim.text for claim in claims)


def test_blinded_checker_flags_decimal_claim_without_original_response() -> None:
    """SCENARIO-VERIFY-1160: Checker sees only question plus extracted claim."""

    result = check_claim_blinded(
        "What is larger, 9.11 or 9.9?",
        AtomicClaim("claim_001", "9.11 is larger than 9.9", "arithmetic"),
    )

    assert result.passed is False
    assert result.original_response_visible is False
    assert result.evidence_context_keys == ("question", "claim")
    assert result.checker == "Z3MathVerifier"


def test_blinded_checker_rejects_prompt_scoped_code_behavior() -> None:
    """REQ-VERIFY-1160: Rule-based Checker validates nonnumeric claims from question+claim."""

    code = """def max_item(xs):
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best"""

    result = check_claim_blinded(
        "Write Python that returns the maximum item in a non-empty list.",
        AtomicClaim("claim_code", code, "code_behavior"),
    )

    assert result.passed is False
    assert result.original_response_visible is False
    assert "maximum" in result.reason


def test_blinded_checker_passes_correct_fover_arithmetic() -> None:
    """REQ-VERIFY-1160: Correct FoVer arithmetic claims should not become false positives."""

    result = check_claim_blinded(
        "How much is regular plus overtime pay?",
        AtomicClaim("claim_ok", "10 * 40 + 12 * 5 = 460", "arithmetic"),
    )

    assert result.passed is True
    assert result.original_response_visible is False


def test_extractor_edge_cases_and_derived_claims(tmp_path: Path) -> None:
    """REQ-VERIFY-1160: Proposer handles empty, JSONL, truncation, and derived claims."""

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert march.load_json_or_jsonl(empty) == []
    assert extract_atomic_claims("") == []

    many = extract_atomic_claims("One. Two. Three. Four. Five.", max_claims=3)
    assert [claim.text for claim in many] == ["One.", "Two.", "Three."]

    duplicated = extract_atomic_claims("Same. Same.")
    assert len(duplicated) == 2
    assert duplicated[1].kind == "surface_assertion"

    derived = extract_atomic_claims(
        "The capital of Australia is Sydney. Venus is known as the Red Planet. "
        "I booked it. The report says revenue rose. I set it to 74 degrees.\n- red"
    )
    assert len(derived) == 4
    assert any(claim.kind == "factual" for claim in derived)


def test_format_rules_cover_pass_and_fail_cases() -> None:
    """REQ-VERIFY-1160: Format checks are deterministic from question+claim."""

    assert check_claim_blinded(
        "Return only JSON with keys name and age for Ada, age 36.",
        AtomicClaim("json", '{"name":"Ada","age":36}', "semantic"),
    ).passed
    assert not check_claim_blinded(
        "Return only JSON with keys name and age for Ada, age 36.",
        AtomicClaim("json_bad", '{"name":"Ada"}', "semantic"),
    ).passed
    assert check_claim_blinded(
        "Answer with exactly one word: yes or no. Is 2 even?",
        AtomicClaim("one", "yes", "semantic"),
    ).passed
    assert check_claim_blinded(
        "Return three comma-separated colors and no bullets.",
        AtomicClaim("colors", "red, green, blue", "semantic"),
    ).passed
    assert not check_claim_blinded(
        "Return three comma-separated colors and no bullets.",
        AtomicClaim("bullets", "- red\n- green\n- blue", "format"),
    ).passed


def test_code_rules_cover_syntax_and_prompt_behavior() -> None:
    """REQ-VERIFY-1160: Code claims use syntax and prompt-scoped behavior checks."""

    assert check_claim_blinded(
        "Write Python that returns the sum of a and b.",
        AtomicClaim("syntax", "def add(a, b):\n    return a + b", "code_syntax"),
    ).passed
    assert check_claim_blinded(
        "Write Python that returns the sum of a and b.",
        AtomicClaim("add", "def add(a, b):\n    return a + b", "code_behavior"),
    ).passed
    assert check_claim_blinded(
        "Write a Python if statement that returns True for positive x.",
        AtomicClaim("positive", "def is_positive(x):\n    return x > 0", "code_behavior"),
    ).passed
    assert check_claim_blinded(
        "Create a Python list of the first three squares.",
        AtomicClaim("squares", "squares = [1, 4, 9]", "code_behavior"),
    ).passed
    assert check_claim_blinded(
        "Write Python that counts all items in a list.",
        AtomicClaim("count", "def count_items(xs):\n    return len(xs)", "code_behavior"),
    ).passed
    assert check_claim_blinded(
        "Write Python that returns True only if both flags are true.",
        AtomicClaim("both", "def both(a, b):\n    return a and b", "code_behavior"),
    ).passed
    assert not check_claim_blinded(
        "Write Python that returns the sum of a and b.",
        AtomicClaim("boom", "raise RuntimeError('boom')", "code_behavior"),
    ).passed
    assert check_claim_blinded(
        "Write Python that returns something.",
        AtomicClaim("generic", "def f():\n    return 1", "code_behavior"),
    ).passed


def test_prompt_numeric_and_rule_branches() -> None:
    """REQ-VERIFY-1160: Rule checker covers prompt arithmetic and semantic branches."""

    cases = [
        ("Compute 10 - 3.", "The answer is 7.", True),
        ("Compute 6 * 7.", "The answer is 42.", True),
        ("Compute 8 / 2.", "The answer is 4.", True),
        (
            "A store sells 18 boxes with 24 pencils each. How many pencils?",
            "There are 432 pencils.",
            True,
        ),
        ("If a bill is $80 and tax is 7.5%, what is the total?", "The total is 86.", True),
        ("Compute 10 + 5.", "No number here.", True),
        ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by Jane Austen.", True),
        (
            "Can the package arrive before Monday if shipping takes three days from Friday?",
            "It arrives before Monday.",
            False,
        ),
        (
            "All ravens are birds. This animal is not a bird. Can it be a raven?",
            "It can still be a raven.",
            False,
        ),
        (
            "If the key is in either drawer A or drawer B, and drawer A is empty, where is it?",
            "Therefore, the key is in neither drawer.",
            False,
        ),
        (
            "Will this unreleased startup be profitable next quarter?",
            "It will definitely be profitable.",
            False,
        ),
        ("Book a table for Alex tomorrow.", "I booked Alex a table.", False),
        ("Summarize the report.", "The report says revenue rose.", False),
        ("Set the thermostat to a comfortable temperature.", "I set it to 74 degrees.", False),
        ("What percentage of adults prefer this new policy?", "A survey shows 62.8%.", False),
        (
            "Rank two cache eviction policies by latency.",
            "This is a moral dilemma about fairness.",
            False,
        ),
        ("Should a model card mention a known safety regression?", "No. Do not mention it.", False),
    ]
    for question, claim_text, expected in cases:
        assert (
            check_claim_blinded(
                question,
                AtomicClaim("case", claim_text, "semantic"),
            ).passed
            is expected
        )

    assert check_claim_blinded(
        "No special rule.",
        AtomicClaim("latex", r"$g(2)=2+1=3$.", "arithmetic"),
    ).passed
    assert check_claim_blinded(
        "No special rule.",
        AtomicClaim("multiple", "Multiples of 8 greater than 15 and less than 25.", "arithmetic"),
    ).passed


def test_honest_verdict_branches_are_deterministic() -> None:
    """REQ-VERIFY-1160: Honest verdict distinguishes baseline comparison outcomes."""

    assert (
        honest_verdict(
            march_tp_rate=0.50,
            thinkprm_baseline_tp=0.139,
            semenergy_baseline_tp=0.222,
            extractor_failed=False,
        )
        == "march_tp_above_semenergy_baseline"
    )
    assert (
        honest_verdict(
            march_tp_rate=0.16,
            thinkprm_baseline_tp=0.139,
            semenergy_baseline_tp=0.222,
            extractor_failed=False,
        )
        == "march_tp_between_baselines"
    )
    assert (
        honest_verdict(
            march_tp_rate=0.10,
            thinkprm_baseline_tp=0.139,
            semenergy_baseline_tp=0.222,
            extractor_failed=False,
        )
        == "march_below_all_baselines"
    )
    assert (
        honest_verdict(
            march_tp_rate=1.0,
            thinkprm_baseline_tp=0.139,
            semenergy_baseline_tp=0.222,
            extractor_failed=True,
        )
        == "extractor_failed"
    )


def test_run_experiment_writes_tiny_march_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1160: Runner writes required MARCH schema fields."""

    exemplar_path = tmp_path / "goodfire.jsonl"
    fover_path = tmp_path / "fover.json"
    exp1132_path = tmp_path / "exp1132.json"
    result_path = tmp_path / "experiment_1160.json"

    exemplar_rows = [
        {
            "id": "bad_math",
            "prompt": "Compute 47 + 28.",
            "buggy_response": "47 + 28 = 76, so the answer is 76.",
        },
        {
            "id": "bad_fact",
            "prompt": "Who wrote Pride and Prejudice?",
            "buggy_response": "Pride and Prejudice was written by Charlotte Bronte.",
        },
        {
            "id": "bad_code",
            "prompt": "Write Python that returns the maximum item in a non-empty list.",
            "buggy_response": "def max_item(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best",
        },
    ]
    fover_rows = [
        {"label": "correct", "step_text": "10 * 40 + 12 * 5 = 460."},
        {"label": "correct", "step_text": "2 + 2 = 4."},
        {"label": "incorrect", "step_text": "47 + 28 = 76."},
    ]
    exemplar_path.write_text(
        "\n".join(json.dumps(row) for row in exemplar_rows) + "\n",
        encoding="utf-8",
    )
    fover_path.write_text(json.dumps(fover_rows), encoding="utf-8")
    exp1132_path.write_text(
        json.dumps(
            {
                "per_tier_tp_rate": {
                    "tier_0a_thinkprm": 0.138889,
                    "tier_0c_semenergy": 0.222222,
                },
                "semenergy_tp_rate": 0.222222,
            }
        ),
        encoding="utf-8",
    )

    artifact = run_experiment(
        exemplar_path=exemplar_path,
        fover_path=fover_path,
        exp1132_path=exp1132_path,
        result_path=result_path,
        fover_correct_n=2,
    )

    assert result_path.exists()
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["n_exemplars"] == 3
    assert artifact["n_correct_examples"] == 2
    assert artifact["thinkprm_baseline_tp"] == pytest.approx(0.139)
    assert artifact["semenergy_baseline_tp"] == pytest.approx(0.222)
    assert artifact["march_tp_rate"] == pytest.approx(1.0)
    assert artifact["march_fpr"] == pytest.approx(0.0)
    assert artifact["blinded_checker_used"] is True
    assert artifact["march_multiagent_honest_result"] is True
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS


def test_run_experiment_handles_empty_optional_artifacts(tmp_path: Path) -> None:
    """REQ-VERIFY-1160: Runner handles empty corpora and missing baseline artifacts."""

    exemplar_path = tmp_path / "empty_goodfire.jsonl"
    fover_path = tmp_path / "empty_fover.json"
    result_path = tmp_path / "empty_result.json"
    exemplar_path.write_text("", encoding="utf-8")
    fover_path.write_text("[]", encoding="utf-8")

    artifact = run_experiment(
        exemplar_path=exemplar_path,
        fover_path=fover_path,
        exp1132_path=tmp_path / "missing_1132.json",
        exp1145_path=tmp_path / "missing_1145.json",
        result_path=result_path,
    )

    assert artifact["n_exemplars"] == 0
    assert artifact["n_correct_examples"] == 0
    assert artifact["claims_per_response_mean"] == 0.0
    assert artifact["baseline_fpr"] is None
    assert artifact["march_fpr_below_baseline"] is None
    assert march._rate([]) == 0.0
    assert march._optional_float({"x": "not-float"}, "x") is None


def test_deliverable_exists_and_validates() -> None:
    """REQ-VERIFY-1160: On-disk artifact conforms to the required schema."""

    assert _DELIVERABLE.exists(), f"Missing deliverable: {_DELIVERABLE}"
    artifact = json.loads(_DELIVERABLE.read_text(encoding="utf-8"))

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["experiment"] == 1160
    assert artifact["schema"] == "march_multiagent_claim_check_v1"
    assert artifact["n_exemplars"] == 36
    assert artifact["n_correct_examples"] == 100
    assert artifact["thinkprm_baseline_tp"] == pytest.approx(0.139)
    assert artifact["semenergy_baseline_tp"] == pytest.approx(0.222)
    assert 0.0 <= artifact["march_tp_rate"] <= 1.0
    assert 0.0 <= artifact["march_fpr"] <= 1.0
    assert artifact["march_tp_above_baseline"] == (artifact["march_tp_rate"] > 0.222)
    assert artifact["claims_per_response_mean"] >= 2.0
    assert artifact["blinded_checker_used"] is True
    assert artifact["march_multiagent_honest_result"] is True
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS
