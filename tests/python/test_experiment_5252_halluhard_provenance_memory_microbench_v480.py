"""Tests for Exp 5252 HalluHard-style provenance-memory microbench.

Spec refs: REQ-VERIFY-5252, SCENARIO-VERIFY-5252.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from carnot.verify import halluhard_provenance_memory_microbench_v480 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


class FakeBatchGenerator:
    """Deterministic stand-in for REQ-VERIFY-5252 live GGUF batch calls."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_tokens: int, seed: int, tag: str) -> mod.GenerationReceipt:
        del max_tokens
        self.prompts.append(prompt)
        return mod.GenerationReceipt(
            tag=tag,
            prompt=prompt,
            text=self.outputs.pop(0),
            seed=seed,
            command=("fake-gguf", tag),
            duration_s=0.01,
            returncode=0,
            stderr_tail="",
            stdout_tail="",
        )


def _preconditions() -> mod.PreconditionReport:
    return mod.PreconditionReport(
        ok=True,
        checks=[
            {"resource": "cuda_gpu", "available": True},
            {"resource": "local_gguf_runtime", "available": True},
            {"resource": "mandated_sota_gguf", "available": True},
        ],
        selected_model={
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "model_path": "/tmp/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "quantization": "UD-Q4_K_M",
        },
        runtime_command=("fake-gguf", "-m", "/tmp/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
    )


def _case_lines(answer_by_id: dict[str, tuple[str, str]]) -> str:
    return "\n".join(
        f"CASE {fixture_id}: ANSWER: {answer}; CITATIONS: {citation}"
        for fixture_id, (answer, citation) in answer_by_id.items()
    )


def test_req_verify_5252_spec_declares_contract() -> None:
    """REQ-VERIFY-5252: OpenSpec anchors the benchmark before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5252") : spec.index("### REQ-VERIFY-2976")]

    for marker in (
        "REQ-VERIFY-5252",
        "SCENARIO-VERIFY-5252",
        mod.RESULT_RELATIVE_PATH,
        "no_memory",
        "raw_conversation_memory",
        "typed_provenance_memory",
        "live_llm_inference_local_gguf_sota",
        "unsupported_claim_rate_no_memory",
        "unsupported_claim_rate_typed_memory",
        "repeated_error_delta",
        "citation_support_delta",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle in section


def test_req_verify_5252_fixture_labels_are_local_and_deterministic() -> None:
    """REQ-VERIFY-5252: curated fixtures carry evidence, turns, and support labels."""

    fixtures = mod.default_fixtures()

    assert 10 <= len(fixtures) <= 20
    assert any(fixture.expected_answer is None for fixture in fixtures)
    assert any(fixture.expected_answer is not None for fixture in fixtures)
    for fixture in fixtures:
        evidence_text = " ".join(snippet.text for snippet in fixture.evidence).lower()
        assert fixture.turn1_question
        assert fixture.turn2_question
        assert fixture.expected_supported_claims or fixture.expected_answer is None
        assert fixture.expected_unsupported_claims
        if fixture.expected_answer is not None:
            assert fixture.expected_answer.lower() in evidence_text
            assert fixture.expected_answer.lower() not in fixture.turn1_question.lower()

    leakage = mod.leakage_checks(fixtures, [])
    assert leakage["passed"] is True
    assert leakage["fixture_count_in_range"]["passed"] is True
    assert leakage["gold_answers_in_evidence"]["passed"] is True
    assert leakage["gold_answers_not_in_questions"]["passed"] is True


def test_req_verify_5252_support_checker_labels_citations_and_refusals() -> None:
    """REQ-VERIFY-5252: deterministic support checks distinguish support errors."""

    answerable = mod.MicrobenchFixture(
        fixture_id="unit-answerable",
        evidence=(mod.EvidenceSnippet("E1", "Unit result: the dax meter read 17 lumens."),),
        turn1_question="What did the dax meter read?",
        turn2_question="Repeat the reading with a citation.",
        expected_answer="17 lumens",
        expected_citation="E1",
        answer_aliases=("17 lumens",),
        expected_supported_claims=("17 lumens",),
        expected_unsupported_claims=("19 lumens",),
        missing_field="",
    )
    missing = mod.MicrobenchFixture(
        fixture_id="unit-missing",
        evidence=(mod.EvidenceSnippet("E1", "Unit note: the dax meter was calibrated Tuesday."),),
        turn1_question="Who built the dax meter?",
        turn2_question="Repeat the builder with a citation.",
        expected_answer=None,
        expected_citation=None,
        answer_aliases=(),
        expected_supported_claims=(),
        expected_unsupported_claims=("Orion Labs",),
        missing_field="builder",
    )

    supported = mod.score_case_response(answerable, "ANSWER: 17 lumens. CITATIONS: [E1]")
    wrong_cite = mod.score_case_response(answerable, "ANSWER: 17 lumens. CITATIONS: [E2]")
    wrong_value = mod.score_case_response(answerable, "ANSWER: 19 lumens. CITATIONS: [E1]")
    refusal = mod.score_case_response(answerable, "ANSWER: INSUFFICIENT_EVIDENCE. CITATIONS: []")
    proper_missing = mod.score_case_response(missing, "ANSWER: INSUFFICIENT_EVIDENCE. CITATIONS: []")
    fabricated_missing = mod.score_case_response(missing, "ANSWER: Orion Labs. CITATIONS: [E1]")

    assert supported.unsupported_claim is False
    assert supported.citation_supported is True
    assert wrong_cite.unsupported_claim is True
    assert wrong_value.unsupported_claim is True
    assert refusal.over_refusal is True
    assert refusal.missed_answer is True
    assert proper_missing.unsupported_claim is False
    assert proper_missing.citation_supported is True
    assert fabricated_missing.unsupported_claim is True


def test_scenario_verify_5252_metric_math_reports_typed_memory_delta() -> None:
    """SCENARIO-VERIFY-5252: repeated-error and citation deltas use typed minus baseline."""

    fixtures = mod.default_fixtures()[:10]
    good = {fixture.fixture_id: (fixture.expected_answer or "INSUFFICIENT_EVIDENCE", f"[{fixture.expected_citation}]") for fixture in fixtures}
    bad = {
        fixture.fixture_id: (
            fixture.expected_unsupported_claims[0],
            f"[{fixture.expected_citation or fixture.evidence[0].evidence_id}]",
        )
        for fixture in fixtures
    }
    outputs = [
        _case_lines(bad),
        _case_lines(bad),
        _case_lines(bad),
        _case_lines(bad),
        _case_lines(bad),
        _case_lines(good),
    ]

    artifact = mod.run_microbench(
        repo_root=REPO,
        generator=FakeBatchGenerator(outputs),
        fixtures=fixtures,
        preconditions=_preconditions(),
        write=False,
    )

    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "reduced" in artifact["honest_verdict"]["value"]
    assert artifact["fixture_count"]["value"] == 10
    assert artifact["unsupported_claim_rate_no_memory"]["value"] == 1.0
    assert artifact["unsupported_claim_rate_typed_memory"]["value"] == 0.5
    assert artifact["repeated_error_delta"]["value"] == 1.0
    assert artifact["citation_support_delta"]["value"] == 0.5
    assert artifact["no_network_at_benchmark_time"]["value"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5252_blocked_artifact_has_neutral_metrics() -> None:
    """REQ-VERIFY-5252: missing runtime writes blocked artifact without tiny headline."""

    artifact = mod.run_microbench(
        repo_root=REPO,
        generator=FakeBatchGenerator([]),
        fixtures=mod.default_fixtures()[:10],
        preconditions=mod.PreconditionReport(
            ok=False,
            checks=[{"resource": "cuda_gpu", "available": False}],
            selected_model=None,
            runtime_command=(),
            blocked_reason="blocked_precondition_cuda_gpu",
        ),
        write=False,
    )

    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["fixture_count"]["value"] == 0
    assert artifact["unsupported_claim_rate_no_memory"]["value"] == 0.0
    assert artifact["unsupported_claim_rate_typed_memory"]["value"] == 0.0
    assert artifact["model_specs"]["value"]["headline_model"] is None
    assert "tiny" not in str(artifact["model_specs"]).lower()
    assert artifact["no_network_at_benchmark_time"]["value"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5252_schema_errors_cover_required_failures() -> None:
    """REQ-VERIFY-5252: schema guard catches malformed required artifact fields."""

    artifact = mod.build_complete_artifact(
        arm_results={
            arm: mod.empty_arm_result(arm, mod.default_fixtures()[:10]) for arm in mod.ARM_NAMES
        },
        preconditions=_preconditions(),
        started_at="2026-07-05T00:00:00Z",
        finished_at="2026-07-05T00:00:01Z",
        duration_s=1.0,
        leakage=mod.leakage_checks(mod.default_fixtures()[:10], []),
    )

    missing_object = copy.deepcopy(artifact)
    missing_object["fixture_count"] = 10
    assert "missing_object_field:fixture_count" in mod.artifact_schema_errors(missing_object)

    invalid = copy.deepcopy(artifact)
    invalid["honest_verdict"]["value"] = "pending"
    invalid["inference_substrate"]["value"] = "cached"
    invalid["fixture_count"]["value"] = 2
    invalid["model_specs"]["value"]["headline_model"] = "tiny/model"
    errors = mod.artifact_schema_errors(invalid)
    assert "honest_verdict_prefix" in errors
    assert "inference_substrate" in errors
    assert "complete_fixture_count_out_of_bounds" in errors
    assert "headline_model_not_mandated_sota" in errors


def test_req_verify_5252_repository_artifact_matches_schema() -> None:
    """REQ-VERIFY-5252: checked-in artifact preserves the terminal schema."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["honest_verdict"]["value"].startswith(("complete:", "blocked_"))
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["no_network_at_benchmark_time"]["value"] is True
    for field in mod.REQUIRED_OBJECT_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]
    assert mod.artifact_schema_errors(artifact) == []
