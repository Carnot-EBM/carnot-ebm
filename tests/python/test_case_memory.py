"""Spec: REQ-VERIFY-050, REQ-VERIFY-051,
SCENARIO-VERIFY-052, SCENARIO-VERIFY-053, SCENARIO-VERIFY-054,
SCENARIO-VERIFY-055.
"""

from __future__ import annotations

import importlib

import pytest


def load_case_module():
    return importlib.import_module("carnot.pipeline.case_memory")


def load_replay_module():
    return importlib.import_module("carnot.pipeline.self_learning_replay")


def test_case_normalization_keeps_semantic_and_code_specific_keys():
    """SCENARIO-VERIFY-052: semantic and code traces normalize into distinct case keys."""
    module = load_case_module()

    semantic = module.CaseRecord.normalize(
        benchmark="gsm8k_semantic",
        benchmark_slice="gsm8k_semantic/live_gsm8k_semantic_failure",
        model_name="Qwen3.5-0.8B",
        case_id="gsm8k-1",
        violation_types=(
            "question_grounding_failures:answer_target_mismatch",
            "semantic:final_answer_binding",
        ),
        prompt_text="How many more calories did Sue consume than her sister?",
        description_texts=(
            "The response does not compute the requested comparison quantity.",
            "semantic:final_answer_binding",
        ),
        baseline_success=False,
        repair_success=True,
        confidence=0.93,
        source_experiment=219,
        source_artifact="results/experiment_219_results.json",
        response_mode="structured_json",
        verifier_path="semantic_grounding",
    )
    code = module.CaseRecord.normalize(
        benchmark="humaneval_property",
        benchmark_slice="humaneval_property/code_typed_properties",
        model_name="Qwen3.5-0.8B",
        case_id="humaneval-1",
        violation_types=("official_test_failure",),
        prompt_text="",
        description_texts=(
            "sorted_output (prompt_intent) failed for input=([2, 1],): expected [1, 2]",
            "input_immutability (prompt_intent) failed because the input list was mutated",
        ),
        baseline_success=False,
        repair_success=True,
        confidence=0.99,
        source_experiment=220,
        source_artifact="results/experiment_220_results.json",
        response_mode="answer_only_terse",
        verifier_path="execution_plus_property",
    )

    assert semantic.case_kind == "semantic_verification"
    assert semantic.violation_families == (
        "question_grounding_failures",
        "semantic",
    )
    assert semantic.property_names == ()
    assert semantic.prompt_sketch.startswith("how")
    assert semantic.repair_outcome == "improved"
    assert semantic.key.benchmark_slice == "gsm8k_semantic/live_gsm8k_semantic_failure"
    assert semantic.provenance.source_artifact == "results/experiment_219_results.json"

    assert code.case_kind == "code_verification"
    assert code.violation_families == ("official_test_failure",)
    assert code.property_names == ("input_immutability", "sorted_output")
    assert code.prompt_sketch != "generic"
    assert code.repair_outcome == "improved"
    assert code.key.prompt_sketch == code.prompt_sketch
    assert code.key != semantic.key


def test_retrieval_ranks_exact_model_prompt_and_property_matches_first():
    """SCENARIO-VERIFY-053: specific prompt and property matches outrank coarse slice matches."""
    module = load_case_module()
    memory = module.CaseMemory()

    qwen_exact = module.CaseRecord.normalize(
        benchmark="humaneval_property",
        benchmark_slice="humaneval_property/code_typed_properties",
        model_name="Qwen3.5-0.8B",
        case_id="qwen-exact",
        violation_types=("official_test_failure",),
        prompt_text="Return a sorted copy of the input list without mutating it.",
        description_texts=(
            "sorted_output (prompt_intent) failed for input=([3, 1],): expected [1, 3]",
            "input_immutability (prompt_intent) failed because the input list was mutated",
        ),
        baseline_success=False,
        repair_success=True,
        confidence=0.96,
        source_experiment=220,
    )
    gemma_cross_model = module.CaseRecord.normalize(
        benchmark="humaneval_property",
        benchmark_slice="humaneval_property/code_typed_properties",
        model_name="Gemma4-E4B-it",
        case_id="gemma-cross",
        violation_types=("official_test_failure",),
        prompt_text="Return a sorted copy of the input list without mutating it.",
        description_texts=(
            "sorted_output (prompt_intent) failed for input=([5, 2],): expected [2, 5]",
            "input_immutability (prompt_intent) failed because the input list was mutated",
        ),
        baseline_success=False,
        repair_success=True,
        confidence=0.91,
        source_experiment=220,
    )
    qwen_coarse = module.CaseRecord.normalize(
        benchmark="humaneval_property",
        benchmark_slice="humaneval_property/code_typed_properties",
        model_name="Qwen3.5-0.8B",
        case_id="qwen-coarse",
        violation_types=("official_test_failure",),
        prompt_text="Reverse the string and preserve upper-case letters.",
        description_texts=(
            "reverse_output (prompt_intent) failed for input=('Ab',): expected 'bA'",
        ),
        baseline_success=False,
        repair_success=True,
        confidence=0.9,
        source_experiment=220,
    )
    for record in (qwen_exact, gemma_cross_model, qwen_coarse):
        memory.record(record)

    query_record = module.CaseRecord.normalize(
        benchmark="humaneval_property",
        benchmark_slice="humaneval_property/code_typed_properties",
        model_name="Qwen3.5-0.8B",
        case_id="heldout",
        violation_types=("property_violation",),
        prompt_text="Return a sorted copy of the input list without mutating it.",
        description_texts=(
            "sorted_output (prompt_intent) failed for input=([9, 1],): expected [1, 9]",
            "input_immutability (prompt_intent) failed because the input list was mutated",
        ),
        baseline_success=None,
        repair_success=None,
        confidence=0.5,
        source_experiment=223,
    )
    query = module.CaseQuery.from_record(query_record, preferred_repair_outcome="improved")
    matches = memory.retrieve(query, limit=3)

    assert [match.entry.provenance[0].case_id for match in matches] == [
        "qwen-exact",
        "gemma-cross",
        "qwen-coarse",
    ]
    assert "model_name" in matches[0].matched_fields
    assert "property_names" in matches[0].matched_fields
    assert "prompt_sketch" in matches[0].matched_fields
    assert matches[0].score > matches[1].score > matches[2].score


def test_case_memory_serialization_round_trips_support_confidence_and_provenance(tmp_path):
    """SCENARIO-VERIFY-054: case memory serialization stays deterministic across save/load."""
    module = load_case_module()
    memory = module.CaseMemory()

    record_a = module.CaseRecord.normalize(
        benchmark="gsm8k_semantic",
        benchmark_slice="gsm8k_semantic/live_gsm8k_semantic_failure",
        model_name="Qwen3.5-0.8B",
        case_id="gsm8k-a",
        violation_types=("question_grounding_failures:answer_target_mismatch",),
        prompt_text="How many more apples does the second child have than the first child?",
        description_texts=("The response computes the wrong target quantity.",),
        baseline_success=False,
        repair_success=True,
        confidence=0.8,
        source_experiment=219,
        source_artifact="results/experiment_219_results.json",
    )
    record_b = module.CaseRecord.normalize(
        benchmark="gsm8k_semantic",
        benchmark_slice="gsm8k_semantic/live_gsm8k_semantic_failure",
        model_name="Qwen3.5-0.8B",
        case_id="gsm8k-b",
        violation_types=("question_grounding_failures:answer_target_mismatch",),
        prompt_text="How many more apples does the second child have than the first child?",
        description_texts=("The response computes the wrong target quantity.",),
        baseline_success=False,
        repair_success=True,
        confidence=1.0,
        source_experiment=219,
        source_artifact="results/experiment_219_results.json",
    )

    memory.record(record_a)
    memory.record(record_b)
    path = tmp_path / "case_memory.json"
    memory.save(path)
    restored = module.CaseMemory.load(path)

    assert restored.to_dict() == memory.to_dict()
    assert len(restored.entries()) == 1
    entry = restored.entries()[0]
    assert entry.support == 2
    assert entry.confidence == pytest.approx(0.9)
    assert [item.case_id for item in entry.provenance] == ["gsm8k-a", "gsm8k-b"]

    query = module.CaseQuery.from_record(record_a, preferred_repair_outcome="improved")
    matches = restored.retrieve(query, limit=1)
    assert matches[0].entry.key == entry.key


def test_case_memory_helper_branches_cover_generic_queries_and_error_paths(tmp_path):
    """REQ-VERIFY-050 and REQ-VERIFY-051: helper branches stay deterministic."""
    module = load_case_module()

    semantic_record = module.CaseRecord.normalize(
        benchmark="custom_eval",
        benchmark_slice="custom_eval/general_slice",
        model_name="Model-A",
        case_id="semantic-generic",
        violation_types=("semantic:unsupported_reference",),
        prompt_text="",
        description_texts=(),
        baseline_success=True,
        repair_success=True,
        confidence=0.4,
        source_experiment=999,
    )
    assert semantic_record.prompt_sketch == "generic"
    assert semantic_record.repair_outcome == "unchanged_success"
    assert semantic_record.case_kind == "semantic_verification"
    assert module.CaseRecord.from_dict(semantic_record.to_dict()) == semantic_record

    memory = module.CaseMemory()
    memory.record(semantic_record)
    assert len(memory) == 1

    token_record = module.CaseRecord.normalize(
        benchmark="humaneval_property",
        benchmark_slice="humaneval_property/code_typed_properties",
        model_name="Model-A",
        case_id="token-match",
        violation_types=("property_violation",),
        prompt_text="Return a sorted copy of the input list without mutating it.",
        description_texts=("sorted_output (prompt_intent) failed for input=([3, 1])",),
        baseline_success=False,
        repair_success=True,
        confidence=0.8,
        source_experiment=220,
    )
    memory.record(token_record)

    token_query = module.CaseQuery.from_record(
        module.CaseRecord.normalize(
            benchmark="humaneval_property",
            benchmark_slice="humaneval_property/code_typed_properties",
            model_name="Model-A",
            case_id="token-query",
            violation_types=("property_violation",),
            prompt_text="Sort the input list without mutating it and return a copy.",
            description_texts=("sorted_output (prompt_intent) failed for input=([9, 1])",),
            baseline_success=None,
            repair_success=None,
            confidence=0.1,
            source_experiment=223,
        ),
        preferred_repair_outcome="improved",
    )
    token_matches = memory.retrieve(token_query, limit=1)
    assert "prompt_tokens" in token_matches[0].matched_fields
    assert token_matches[0].to_dict()["matched_fields"][-1] in {
        "repair_outcome",
        "prompt_tokens",
    }

    mismatch_query = module.CaseQuery.from_record(
        module.CaseRecord.normalize(
            benchmark="other_benchmark",
            benchmark_slice="other_benchmark/other_slice",
            model_name="Model-Z",
            case_id="mismatch",
            violation_types=("other",),
            prompt_text="",
            description_texts=(),
            baseline_success=None,
            repair_success=None,
            confidence=0.0,
            source_experiment=1,
        ),
        preferred_repair_outcome="improved",
    )
    assert memory.retrieve(mismatch_query) == []
    assert (
        memory.retrieve(
            module.CaseQuery.from_record(
                module.CaseRecord.normalize(
                    benchmark="humaneval_property",
                    benchmark_slice="humaneval_property/code_typed_properties",
                    model_name="Model-Z",
                    case_id="same-slice-no-overlap",
                    violation_types=("other",),
                    prompt_text="completely unrelated prompt",
                    description_texts=("unrelated_property failed",),
                    baseline_success=None,
                    repair_success=None,
                    confidence=0.0,
                    source_experiment=1,
                ),
                preferred_repair_outcome="improved",
            )
        )
        == []
    )

    payload = memory.to_dict()
    restored = module.CaseMemory.from_dict(
        {"version": module.VERSION, "entries": [payload["entries"][0], "bad-entry"]}
    )
    assert len(restored) == 1

    with pytest.raises(ValueError, match="version"):
        module.CaseMemory.from_dict({"version": 99, "entries": []})

    invalid_path = tmp_path / "invalid_case_memory.json"
    invalid_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        module.CaseMemory.load(invalid_path)


def test_replay_adds_case_memory_without_breaking_existing_pattern_memory_flow():
    """SCENARIO-VERIFY-055: replay can reuse additive case memory on coarse-label drift."""
    case_module = load_case_module()
    replay = load_replay_module()
    del case_module

    cases = [
        replay.ReplayCase(
            source_experiment=220,
            benchmark="humaneval_property",
            metric_name="pass_rate",
            domain="code_typed_properties",
            model_name="Qwen3.5-0.8B",
            case_id="learn-1",
            sample_position=1,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("official_test_failure",),
            descriptions=(
                "sorted_output (prompt_intent) failed for input=([2, 1],): expected [1, 2]",
                "input_immutability (prompt_intent) failed because the input list was mutated",
            ),
            baseline_success=False,
            repair_success=True,
        ),
        replay.ReplayCase(
            source_experiment=220,
            benchmark="humaneval_property",
            metric_name="pass_rate",
            domain="code_typed_properties",
            model_name="Qwen3.5-0.8B",
            case_id="learn-2",
            sample_position=2,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("official_test_failure",),
            descriptions=(
                "sorted_output (prompt_intent) failed for input=([5, 4],): expected [4, 5]",
                "input_immutability (prompt_intent) failed because the input list was mutated",
            ),
            baseline_success=False,
            repair_success=True,
        ),
        replay.ReplayCase(
            source_experiment=220,
            benchmark="humaneval_property",
            metric_name="pass_rate",
            domain="code_typed_properties",
            model_name="Qwen3.5-0.8B",
            case_id="learn-3",
            sample_position=3,
            held_out=False,
            actual_error=True,
            detected=True,
            error_types=("official_test_failure",),
            descriptions=(
                "sorted_output (prompt_intent) failed for input=([9, 6],): expected [6, 9]",
                "input_immutability (prompt_intent) failed because the input list was mutated",
            ),
            baseline_success=False,
            repair_success=True,
        ),
        replay.ReplayCase(
            source_experiment=220,
            benchmark="humaneval_property",
            metric_name="pass_rate",
            domain="code_typed_properties",
            model_name="Gemma4-E4B-it",
            case_id="heldout-case-memory",
            sample_position=4,
            held_out=True,
            actual_error=True,
            detected=True,
            error_types=("property_violation",),
            descriptions=(
                "sorted_output (prompt_intent) failed for input=([8, 1],): expected [1, 8]",
                "input_immutability (prompt_intent) failed because the input list was mutated",
            ),
            baseline_success=False,
            repair_success=True,
        ),
    ]

    payload = replay.run_replay_cases(
        cases,
        tracker_min_support=5,
        tracker_min_precision=0.99,
        memory_min_support=3,
    )

    decision = payload["held_out_decisions"][0]["strategies"]["tracker_plus_memory"]
    assert decision["use_repair"] is True
    assert decision["reason"] == "case_memory_reuse"
    assert decision["candidate_error_types"] == ["official_test_failure"]
    assert decision["matched_error_types"] == []
    assert len(decision["candidate_case_keys"]) >= 1
    assert len(decision["matched_case_keys"]) >= 1
    assert decision["support_models"] == ["Qwen3.5-0.8B"]
    assert (
        payload["transfer_effects"]["tracker_plus_memory"]["Gemma4-E4B-it"][
            "cross_model_helpful_events"
        ]
        == 1
    )
