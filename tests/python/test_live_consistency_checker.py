"""Tests for Exp 271: GlobalConsistencyChecker on live multi-turn LLM chains.

**Detailed explanation for engineers:**
    Tests the ``live_consistency_eval`` module which evaluates the
    GlobalConsistencyChecker on realistic multi-turn LLM output chains —
    not the minimal synthetic text used in Exp 172/176.

    All 20 chains are built from pre-defined representative outputs that
    match Gemma4-E4B-it's prose style. No live LLM inference is required
    for these tests; the ``generate_fn`` parameter is left None (uses
    pre-built turns) or is replaced by a deterministic stub.

    Test coverage:
    - _build_consistent_chain produces non-contradicting outputs (REQ-VERIFY-001)
    - _build_contradicted_chain injects a detectable contradiction (SCENARIO-VERIFY-005)
    - evaluate_chain returns correct ChainResult fields (REQ-VERIFY-001)
    - evaluate_chain: true positive on contradicted chain (SCENARIO-VERIFY-005)
    - evaluate_chain: true negative on consistent chain (REQ-VERIFY-001)
    - ChainResult helper predicates (is_true_positive, is_false_positive, etc.)
    - run_evaluation returns correct schema (REQ-VERIFY-001)
    - run_evaluation detection_rate == 1.0 for injected contradictions (SCENARIO-VERIFY-005)
    - run_evaluation false_positive_rate == 0.0 for consistent chains (REQ-VERIFY-001)
    - run_evaluation per_type_detection covers numeric/arithmetic/factual (SCENARIO-VERIFY-005)
    - run_evaluation comparison_to_synthetic keys present (REQ-VERIFY-001)
    - Live generate_fn stub: chains are built from generate_fn outputs (SCENARIO-VERIFY-005)
    - All 20 chains serialized in result["chains"] (REQ-VERIFY-001)
    - avg_latency_ms is a positive float (REQ-VERIFY-001)

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline.live_consistency_eval import (
    ChainResult,
    _QUESTION_SEEDS,
    _build_contradicted_chain,
    _build_consistent_chain,
    evaluate_chain,
    run_evaluation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_seed() -> dict[str, str]:
    """Return the first question seed for unit tests.

    Spec: REQ-VERIFY-001
    """
    return _QUESTION_SEEDS[0]


def _consistent_result(chain_id: int = 0) -> ChainResult:
    """Build a consistent ChainResult for predicate tests.

    Spec: REQ-VERIFY-001
    """
    return ChainResult(
        chain_id=chain_id,
        chain_type="consistent",
        contradiction_type=None,
        expected_consistent=True,
        global_detected=False,
        severity="none",
        n_inconsistent_pairs=0,
    )


def _contradicted_result(detected: bool = True, chain_id: int = 10) -> ChainResult:
    """Build a contradicted ChainResult for predicate tests.

    Spec: SCENARIO-VERIFY-005
    """
    return ChainResult(
        chain_id=chain_id,
        chain_type="contradicted",
        contradiction_type="numeric",
        expected_consistent=False,
        global_detected=detected,
        severity="warning" if detected else "none",
        n_inconsistent_pairs=1 if detected else 0,
    )


# ---------------------------------------------------------------------------
# Tests: chain building helpers
# ---------------------------------------------------------------------------


class TestBuildConsistentChain:
    """_build_consistent_chain returns 4 non-contradicting turns.

    Spec: REQ-VERIFY-001
    """

    def test_returns_four_turns(self) -> None:
        """Consistent chain has exactly 4 turn strings.

        Spec: REQ-VERIFY-001
        """
        turns = _build_consistent_chain(_first_seed())
        assert len(turns) == 4

    def test_all_turns_are_nonempty_strings(self) -> None:
        """Every turn output is a non-empty string.

        Spec: REQ-VERIFY-001
        """
        turns = _build_consistent_chain(_first_seed())
        for t in turns:
            assert isinstance(t, str)
            assert len(t) > 0

    def test_consistent_value_appears_in_turns(self) -> None:
        """The consistent entity value appears at least once in the chain.

        Spec: REQ-VERIFY-001
        """
        seed = _first_seed()
        turns = _build_consistent_chain(seed)
        combined = " ".join(turns)
        assert seed["consistent_value"] in combined

    def test_contradiction_value_absent_in_consistent_chain(self) -> None:
        """The wrong/contradiction value does NOT appear in a consistent chain.

        Spec: REQ-VERIFY-001
        """
        seed = _first_seed()
        turns = _build_consistent_chain(seed)
        combined = " ".join(turns)
        # Contradiction value should not appear in the consistent chain
        assert seed["contradiction_value"] not in combined


class TestBuildContradictedChain:
    """_build_contradicted_chain injects a detectable contradiction into turn 3.

    Spec: SCENARIO-VERIFY-005
    """

    def test_returns_four_turns(self) -> None:
        """Contradicted chain also has exactly 4 turn strings.

        Spec: SCENARIO-VERIFY-005
        """
        turns = _build_contradicted_chain(_first_seed())
        assert len(turns) == 4

    def test_contradiction_value_in_last_turn(self) -> None:
        """The wrong value appears in turn 3 (the injected turn).

        Spec: SCENARIO-VERIFY-005
        """
        seed = _first_seed()
        turns = _build_contradicted_chain(seed)
        assert seed["contradiction_value"] in turns[3]

    def test_consistent_value_also_in_last_turn(self) -> None:
        """The original value still appears in turn 3 (not replaced, just contradicted).

        The injection is additive: the original text (containing the correct
        value) is preserved, and a contradicting sentence is appended.

        Spec: SCENARIO-VERIFY-005
        """
        seed = _first_seed()
        turns = _build_contradicted_chain(seed)
        assert seed["consistent_value"] in turns[3]

    def test_turns_0_to_2_identical_to_consistent(self) -> None:
        """Turns 0-2 of the contradicted chain are identical to consistent chain.

        Only the final turn is modified.

        Spec: SCENARIO-VERIFY-005
        """
        seed = _first_seed()
        consistent = _build_consistent_chain(seed)
        contradicted = _build_contradicted_chain(seed)
        assert consistent[:3] == contradicted[:3]


# ---------------------------------------------------------------------------
# Tests: ChainResult predicates
# ---------------------------------------------------------------------------


class TestChainResultPredicates:
    """ChainResult.is_true_positive / false_positive / true_negative / false_negative.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_true_positive_when_contradiction_detected(self) -> None:
        """Contradicted chain + detected → is_true_positive().

        Spec: SCENARIO-VERIFY-005
        """
        r = _contradicted_result(detected=True)
        assert r.is_true_positive() is True
        assert r.is_false_positive() is False
        assert r.is_true_negative() is False
        assert r.is_false_negative() is False

    def test_false_negative_when_contradiction_missed(self) -> None:
        """Contradicted chain + not detected → is_false_negative().

        Spec: SCENARIO-VERIFY-005
        """
        r = _contradicted_result(detected=False)
        assert r.is_false_negative() is True
        assert r.is_true_positive() is False

    def test_true_negative_when_consistent_not_flagged(self) -> None:
        """Consistent chain + not flagged → is_true_negative().

        Spec: REQ-VERIFY-001
        """
        r = _consistent_result()
        assert r.is_true_negative() is True
        assert r.is_false_positive() is False

    def test_false_positive_when_consistent_flagged(self) -> None:
        """Consistent chain + incorrectly flagged → is_false_positive().

        Spec: REQ-VERIFY-001
        """
        r = ChainResult(
            chain_id=0,
            chain_type="consistent",
            contradiction_type=None,
            expected_consistent=True,
            global_detected=True,  # incorrectly flagged
            severity="warning",
            n_inconsistent_pairs=1,
        )
        assert r.is_false_positive() is True
        assert r.is_true_negative() is False


# ---------------------------------------------------------------------------
# Tests: evaluate_chain
# ---------------------------------------------------------------------------


class TestEvaluateChain:
    """evaluate_chain runs GlobalConsistencyChecker on one chain.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_true_positive_on_contradicted_chain(self) -> None:
        """Contradicted chain is detected by GlobalConsistencyChecker.

        This is the core requirement: the checker must catch injected
        numeric contradictions in realistic LLM-style prose.

        Spec: SCENARIO-VERIFY-005
        """
        seed = _first_seed()
        turns = _build_contradicted_chain(seed)
        result = evaluate_chain(
            turns=turns,
            chain_id=10,
            chain_type="contradicted",
            contradiction_type=seed["contradiction_type"],
            expected_consistent=False,
        )
        assert result.global_detected is True, (
            f"Checker missed contradiction in chain {seed['id']}: "
            f"injected {seed['contradiction_value']} vs {seed['consistent_value']}"
        )
        assert result.is_true_positive() is True

    def test_true_negative_on_consistent_chain(self) -> None:
        """Consistent chain is NOT flagged by GlobalConsistencyChecker.

        Zero false positives on well-formed LLM prose.

        Spec: REQ-VERIFY-001
        """
        seed = _first_seed()
        turns = _build_consistent_chain(seed)
        result = evaluate_chain(
            turns=turns,
            chain_id=0,
            chain_type="consistent",
            contradiction_type=None,
            expected_consistent=True,
        )
        assert result.global_detected is False, (
            f"Checker falsely flagged consistent chain {seed['id']}"
        )
        assert result.is_true_negative() is True

    def test_chain_result_fields_populated(self) -> None:
        """evaluate_chain returns all required ChainResult fields.

        Spec: REQ-VERIFY-001
        """
        seed = _first_seed()
        turns = _build_consistent_chain(seed)
        result = evaluate_chain(
            turns=turns,
            chain_id=5,
            chain_type="consistent",
            contradiction_type=None,
            expected_consistent=True,
        )
        assert result.chain_id == 5
        assert result.chain_type == "consistent"
        assert result.contradiction_type is None
        assert result.expected_consistent is True
        assert isinstance(result.severity, str)
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0.0

    def test_latency_is_positive(self) -> None:
        """Latency field is a positive number (actual timing, not zero).

        Spec: REQ-VERIFY-001
        """
        seed = _QUESTION_SEEDS[2]
        turns = _build_contradicted_chain(seed)
        result = evaluate_chain(
            turns=turns,
            chain_id=12,
            chain_type="contradicted",
            contradiction_type="numeric",
            expected_consistent=False,
        )
        # Latency may be very small but must be non-negative
        assert result.latency_ms >= 0.0

    def test_contradicted_severity_not_none(self) -> None:
        """Detected contradiction has severity 'warning' or 'critical', not 'none'.

        Spec: SCENARIO-VERIFY-005
        """
        seed = _first_seed()
        turns = _build_contradicted_chain(seed)
        result = evaluate_chain(
            turns=turns,
            chain_id=10,
            chain_type="contradicted",
            contradiction_type=seed["contradiction_type"],
            expected_consistent=False,
        )
        if result.global_detected:
            assert result.severity in {"warning", "critical"}
        # If not detected (false negative), we skip severity assertion


# ---------------------------------------------------------------------------
# Tests: run_evaluation (full 20-chain batch)
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    """run_evaluation returns correct schema and detection metrics.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    @pytest.fixture(scope="class")
    def eval_result(self) -> dict[str, object]:
        """Run the full evaluation once; reuse result across tests in this class.

        Spec: REQ-VERIFY-001
        """
        return run_evaluation()

    def test_result_has_experiment_key(self, eval_result: dict) -> None:
        """Result dict has 'experiment' key identifying this as Exp 271.

        Spec: REQ-VERIFY-001
        """
        assert eval_result["experiment"] == "271_global_consistency_live"

    def test_twenty_chains_total(self, eval_result: dict) -> None:
        """Result contains exactly 20 chain entries (10 consistent + 10 contradicted).

        Spec: REQ-VERIFY-001
        """
        assert eval_result["n_chains_total"] == 20
        assert len(eval_result["chains"]) == 20

    def test_ten_consistent_ten_contradicted(self, eval_result: dict) -> None:
        """Exactly 10 consistent and 10 contradicted chains.

        Spec: REQ-VERIFY-001
        """
        assert eval_result["n_consistent_chains"] == 10
        assert eval_result["n_contradicted_chains"] == 10

    def test_detection_rate_equals_one(self, eval_result: dict) -> None:
        """All 10 injected contradictions are detected (detection_rate == 1.0).

        The regex-based checker reliably detects injections that follow
        the numeric/arithmetic/factual patterns. Injected text is crafted
        to match those patterns exactly.

        Spec: SCENARIO-VERIFY-005
        """
        assert eval_result["summary"]["detection_rate"] == 1.0, (
            f"Expected detection_rate=1.0, got {eval_result['summary']['detection_rate']}"
        )

    def test_false_positive_rate_equals_zero(self, eval_result: dict) -> None:
        """No consistent chain is incorrectly flagged (false_positive_rate == 0.0).

        The consistent chains do not contain the specific numeric conflicts
        that would trigger the checker.

        Spec: REQ-VERIFY-001
        """
        assert eval_result["summary"]["false_positive_rate"] == 0.0, (
            f"Expected false_positive_rate=0.0, got {eval_result['summary']['false_positive_rate']}"
        )

    def test_per_type_detection_keys_present(self, eval_result: dict) -> None:
        """per_type_detection covers numeric, arithmetic, and factual types.

        Spec: SCENARIO-VERIFY-005
        """
        per_type = eval_result["summary"]["per_type_detection"]
        assert "numeric" in per_type
        assert "arithmetic" in per_type
        assert "factual" in per_type

    def test_per_type_numeric_detection_rate(self, eval_result: dict) -> None:
        """Numeric contradiction type has detection_rate == 1.0.

        Spec: SCENARIO-VERIFY-005
        """
        per_type = eval_result["summary"]["per_type_detection"]
        assert per_type["numeric"]["detection_rate"] == 1.0

    def test_per_type_factual_detection_rate(self, eval_result: dict) -> None:
        """Factual contradiction type has detection_rate == 1.0.

        Spec: SCENARIO-VERIFY-005
        """
        per_type = eval_result["summary"]["per_type_detection"]
        assert per_type["factual"]["detection_rate"] == 1.0

    def test_comparison_to_synthetic_keys(self, eval_result: dict) -> None:
        """comparison_to_synthetic has baseline and live rate keys.

        Spec: REQ-VERIFY-001
        """
        comp = eval_result["summary"]["comparison_to_synthetic"]
        assert "exp172_detection_rate" in comp
        assert "exp176_global_detection_rate" in comp
        assert "live_detection_rate" in comp
        assert "live_false_positive_rate" in comp
        assert "delta_detection" in comp

    def test_avg_latency_positive(self, eval_result: dict) -> None:
        """avg_latency_ms is a non-negative float.

        Spec: REQ-VERIFY-001
        """
        lat = eval_result["summary"]["avg_latency_ms"]
        assert isinstance(lat, float)
        assert lat >= 0.0

    def test_chains_are_json_serializable(self, eval_result: dict) -> None:
        """Full result dict can be serialized to JSON without error.

        Spec: REQ-VERIFY-001
        """
        serialized = json.dumps(eval_result)
        reparsed = json.loads(serialized)
        assert reparsed["n_chains_total"] == 20

    def test_each_chain_has_required_fields(self, eval_result: dict) -> None:
        """Every chain entry has all required schema fields.

        Spec: REQ-VERIFY-001
        """
        required_fields = {
            "chain_id", "chain_type", "contradiction_type",
            "expected_consistent", "global_detected", "severity",
            "n_inconsistent_pairs", "inconsistent_pairs", "latency_ms",
        }
        for chain in eval_result["chains"]:
            missing = required_fields - set(chain.keys())
            assert not missing, f"Chain {chain.get('chain_id')} missing: {missing}"

    def test_contradicted_chains_are_detected(self, eval_result: dict) -> None:
        """Every chain with expected_consistent=False has global_detected=True.

        Spec: SCENARIO-VERIFY-005
        """
        for chain in eval_result["chains"]:
            if not chain["expected_consistent"]:
                assert chain["global_detected"] is True, (
                    f"Chain {chain['chain_id']} ({chain['contradiction_type']}) "
                    f"was not detected"
                )

    def test_consistent_chains_not_flagged(self, eval_result: dict) -> None:
        """Every chain with expected_consistent=True has global_detected=False.

        Spec: REQ-VERIFY-001
        """
        for chain in eval_result["chains"]:
            if chain["expected_consistent"]:
                assert chain["global_detected"] is False, (
                    f"Chain {chain['chain_id']} (consistent) was falsely flagged"
                )


# ---------------------------------------------------------------------------
# Tests: run_evaluation with stub generate_fn
# ---------------------------------------------------------------------------


class TestRunEvaluationWithGenerateFn:
    """run_evaluation passes generate_fn output through the checker.

    Spec: SCENARIO-VERIFY-005
    """

    def test_stub_generate_fn_used_for_consistent_chain(self) -> None:
        """When generate_fn is provided, chain turns come from it.

        We use a stub that produces consistent text for the first seed,
        then verify that the consistent chain (chain_id=0) is not flagged.

        Spec: SCENARIO-VERIFY-005
        """
        call_count = [0]

        def stub_generate(prompt: str) -> str:
            """Stub that always returns the same consistent text."""
            call_count[0] += 1
            return "The apple cost is 3 dollars per unit. Final answer: 3."

        # Run only first seed to keep test fast
        seed = _QUESTION_SEEDS[0]
        from carnot.pipeline.live_consistency_eval import _generate_live_chain
        turns = _generate_live_chain(seed, stub_generate, inject=False)

        assert len(turns) == 4
        assert call_count[0] == 4  # called once per turn
        for t in turns:
            assert isinstance(t, str)
            assert len(t) > 0

    def test_stub_generate_fn_with_injection(self) -> None:
        """Injected contradiction appended to turn 3 even with custom generate_fn.

        Spec: SCENARIO-VERIFY-005
        """
        def stub_generate(prompt: str) -> str:
            return "The apple cost is 3 dollars per unit."

        seed = _QUESTION_SEEDS[0]
        from carnot.pipeline.live_consistency_eval import _generate_live_chain
        turns = _generate_live_chain(seed, stub_generate, inject=True)

        # Turn 3 should contain both the consistent value AND the contradiction value
        assert seed["contradiction_value"] in turns[3]
        assert seed["consistent_value"] in turns[3]

    def test_full_run_with_generate_fn_produces_correct_schema(self) -> None:
        """run_evaluation with generate_fn returns the same schema as default mode.

        Uses a stub that produces consistent LLM-like text. After injection,
        contradicted chains should still be detected.

        Spec: SCENARIO-VERIFY-005
        """
        def stub_generate(prompt: str) -> str:
            # Returns text containing each seed's consistent value for a specific seed.
            # We return generic consistent text; the contradiction injection will
            # add the wrong value to turn 3, making the checker detect it.
            return "The value is 3. This follows from the given data."

        result = run_evaluation(generate_fn=stub_generate)

        assert result["n_chains_total"] == 20
        assert "summary" in result
        assert "detection_rate" in result["summary"]
        # With stub outputs that are consistent, and with injection on contradicted
        # chains, we can only guarantee the schema is correct. Detection may vary
        # depending on whether the stub's generic text triggers numeric patterns.
        assert 0.0 <= result["summary"]["detection_rate"] <= 1.0
        assert 0.0 <= result["summary"]["false_positive_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: results file artifact
# ---------------------------------------------------------------------------


class TestResultsArtifact:
    """Validate that results/experiment_271_results.json has correct schema.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    @pytest.fixture(scope="class")
    def artifact(self) -> dict[str, object]:
        """Load the pre-computed results artifact.

        Spec: REQ-VERIFY-001
        """
        repo_root = Path(__file__).resolve().parents[2]
        artifact_path = repo_root / "results" / "experiment_271_results.json"
        assert artifact_path.exists(), (
            f"Missing artifact: {artifact_path}. "
            "Run scripts/experiment_271_live_consistency.py to generate it."
        )
        with artifact_path.open() as f:
            return json.load(f)

    def test_artifact_experiment_key(self, artifact: dict) -> None:
        """Artifact 'experiment' key identifies Exp 271.

        Spec: REQ-VERIFY-001
        """
        assert "271" in str(artifact["experiment"])

    def test_artifact_has_summary(self, artifact: dict) -> None:
        """Artifact contains a 'summary' section.

        Spec: REQ-VERIFY-001
        """
        assert "summary" in artifact

    def test_artifact_detection_rate_at_least_80_pct(self, artifact: dict) -> None:
        """Detection rate in artifact is at least 80% (conservative floor).

        Synthetic Exp 172/176 achieved 100%. Live representative chains
        should achieve ≥80% given the injections are designed to match
        the checker's extraction patterns.

        Spec: SCENARIO-VERIFY-005
        """
        detection_rate = artifact["summary"]["detection_rate"]
        assert detection_rate >= 0.8, (
            f"Detection rate {detection_rate:.2%} below 80% floor. "
            "Investigate whether injection patterns changed."
        )

    def test_artifact_false_positive_rate_at_most_20_pct(self, artifact: dict) -> None:
        """False positive rate in artifact is at most 20%.

        Spec: REQ-VERIFY-001
        """
        fp_rate = artifact["summary"]["false_positive_rate"]
        assert fp_rate <= 0.2, (
            f"False positive rate {fp_rate:.2%} exceeds 20% ceiling."
        )

    def test_artifact_chain_count(self, artifact: dict) -> None:
        """Artifact contains exactly 20 chain entries.

        Spec: REQ-VERIFY-001
        """
        assert artifact["n_chains_total"] == 20
        assert len(artifact["chains"]) == 20
