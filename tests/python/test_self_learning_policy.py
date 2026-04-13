"""Spec: REQ-VERIFY-052, REQ-VERIFY-053,
SCENARIO-VERIFY-056, SCENARIO-VERIFY-057, SCENARIO-VERIFY-058,
SCENARIO-VERIFY-059.
"""

from __future__ import annotations

import importlib
import json

import pytest


def load_policy_module():
    return importlib.import_module("carnot.pipeline.self_learning_policy")


def load_case_module():
    return importlib.import_module("carnot.pipeline.case_memory")


def load_tracker_module():
    return importlib.import_module("carnot.pipeline.tracker")


def make_case_memory():
    case_module = load_case_module()
    memory = case_module.CaseMemory()

    semantic_a = case_module.CaseRecord.normalize(
        benchmark="gsm8k_semantic",
        benchmark_slice="gsm8k_semantic/live_gsm8k_semantic_failure",
        model_name="Qwen3.5-0.8B",
        case_id="gsm8k-1",
        violation_types=("semantic:answer_target_mismatch",),
        prompt_text="How many more apples does the second child have?",
        description_texts=("The response computes the wrong target quantity.",),
        baseline_success=False,
        repair_success=True,
        confidence=0.96,
        source_experiment=219,
        source_artifact="results/experiment_219_results.json",
        response_mode="structured_json",
        verifier_path="semantic_grounding",
    )
    semantic_b = case_module.CaseRecord.normalize(
        benchmark="gsm8k_semantic",
        benchmark_slice="gsm8k_semantic/live_gsm8k_semantic_failure",
        model_name="Qwen3.5-0.8B",
        case_id="gsm8k-2",
        violation_types=("semantic:answer_target_mismatch",),
        prompt_text="How many more apples does the second child have?",
        description_texts=("The response computes the wrong target quantity.",),
        baseline_success=False,
        repair_success=True,
        confidence=0.94,
        source_experiment=219,
        source_artifact="results/experiment_219_results.json",
        response_mode="structured_json",
        verifier_path="semantic_grounding",
    )
    code_a = case_module.CaseRecord.normalize(
        benchmark="humaneval_property",
        benchmark_slice="humaneval_property/code_typed_properties",
        model_name="Qwen3.5-0.8B",
        case_id="humaneval-1",
        violation_types=("official_test_failure",),
        prompt_text="Return a sorted copy of the input list without mutating it.",
        description_texts=(
            "sorted_output (prompt_intent) failed for input=([2, 1],): expected [1, 2]",
            "input_immutability (prompt_intent) failed because the input list was mutated",
        ),
        baseline_success=False,
        repair_success=True,
        confidence=0.98,
        source_experiment=220,
        source_artifact="results/experiment_220_results.json",
        response_mode="answer_only_terse",
        verifier_path="execution_plus_property",
    )
    code_b = case_module.CaseRecord.normalize(
        benchmark="humaneval_property",
        benchmark_slice="humaneval_property/code_typed_properties",
        model_name="Qwen3.5-0.8B",
        case_id="humaneval-2",
        violation_types=("official_test_failure",),
        prompt_text="Return a sorted copy of the input list without mutating it.",
        description_texts=(
            "sorted_output (prompt_intent) failed for input=([5, 2],): expected [2, 5]",
            "input_immutability (prompt_intent) failed because the input list was mutated",
        ),
        baseline_success=False,
        repair_success=True,
        confidence=0.95,
        source_experiment=220,
        source_artifact="results/experiment_220_results.json",
        response_mode="answer_only_terse",
        verifier_path="execution_plus_property",
    )

    for record in (semantic_a, semantic_b, code_a, code_b):
        memory.record(record)

    return memory, semantic_a, code_a


def make_tracker():
    tracker_module = load_tracker_module()
    tracker = tracker_module.ConstraintTracker()

    for _ in range(6):
        tracker.record("semantic", fired=True, caught_error=True, any_error_in_batch=True)
    for _ in range(4):
        tracker.record(
            "official_test_failure",
            fired=True,
            caught_error=True,
            any_error_in_batch=True,
        )
    tracker.record("official_test_failure", fired=True, caught_error=False, any_error_in_batch=True)

    return tracker


def accepted_repairs():
    return (
        {
            "snippet_id": "humaneval_property:official_test_failure",
            "benchmark": "humaneval_property",
            "domain": "code_typed_properties",
            "model_names": ["Qwen3.5-0.8B"],
            "trigger_error_type": "official_test_failure",
            "template": (
                "Keep the function signature unchanged.\n"
                "Preserve input immutability and return the corrected body only."
            ),
            "support": 3,
            "successful_cases": 2,
            "failed_cases": 1,
            "provenance": [
                {
                    "source_experiment": 220,
                    "model_name": "Qwen3.5-0.8B",
                    "case_id": "humaneval-1",
                    "iteration": 1,
                },
                {
                    "source_experiment": 220,
                    "model_name": "Qwen3.5-0.8B",
                    "case_id": "humaneval-2",
                    "iteration": 1,
                },
            ],
        },
    )


def test_compile_policy_from_high_precision_cases_emits_threshold_budget_and_routing_updates():
    """SCENARIO-VERIFY-056: high-precision cases compile into deterministic policy updates."""
    module = load_policy_module()
    memory, _, _ = make_case_memory()
    tracker = make_tracker()

    compiler = module.SelfLearningPolicyCompiler(min_case_support=2, min_case_confidence=0.9)
    policy = compiler.compile(case_memory=memory, accepted_repairs=(), tracker=tracker)

    assert [item.update_id for item in policy.threshold_overrides] == [
        "threshold:qwen3.5-0.8b:gsm8k_semantic/live_gsm8k_semantic_failure:semantic",
        "threshold:qwen3.5-0.8b:humaneval_property/code_typed_properties:official_test_failure",
    ]
    semantic_override = policy.threshold_overrides[0]
    assert semantic_override.verifier_name == "semantic_verifier_v2"
    assert semantic_override.threshold_name == "repair_trigger_threshold"
    assert semantic_override.threshold_value < semantic_override.baseline_value
    assert semantic_override.support == 2
    assert semantic_override.provenance[0].source_type == "case_memory"
    assert semantic_override.provenance[0].case_id == "gsm8k-1"

    assert len(policy.property_budget_updates) == 1
    budget = policy.property_budget_updates[0]
    assert budget.update_id == (
        "property_budget:qwen3.5-0.8b:humaneval_property/code_typed_properties"
    )
    assert budget.budget == 2
    assert budget.property_names == ("input_immutability", "sorted_output")
    assert budget.provenance[0].source_artifact == "results/experiment_220_results.json"

    assert len(policy.routing_hints) == 2
    assert [hint.route_to for hint in policy.routing_hints] == [
        "case_memory_then_repair",
        "case_memory_then_repair",
    ]


def test_compile_policy_from_accepted_repairs_emits_prompt_patches_with_provenance():
    """SCENARIO-VERIFY-057: accepted repairs become reusable prompt patches."""
    module = load_policy_module()
    memory, _, _ = make_case_memory()

    policy = module.SelfLearningPolicyCompiler().compile(
        case_memory=memory,
        accepted_repairs=accepted_repairs(),
        tracker=None,
    )

    assert len(policy.repair_prompt_patches) == 1
    patch = policy.repair_prompt_patches[0]
    assert patch.update_id == (
        "repair_patch:qwen3.5-0.8b:humaneval_property/code_typed_properties:official_test_failure"
    )
    assert patch.prompt_patch.startswith("Keep the function signature unchanged.")
    assert patch.success_rate == pytest.approx(2 / 3)
    assert patch.support == 3
    assert patch.provenance[0].source_type == "repair_snippet"
    assert patch.provenance[0].case_id == "humaneval-1"


def test_policy_serialization_round_trips_deterministically(tmp_path):
    """SCENARIO-VERIFY-058: policy artifact serialization round-trips deterministically."""
    module = load_policy_module()
    memory, semantic_record, _ = make_case_memory()
    tracker = make_tracker()
    policy = module.SelfLearningPolicyCompiler().compile(
        case_memory=memory,
        accepted_repairs=accepted_repairs(),
        tracker=tracker,
    )

    path = tmp_path / "self_learning_policy.json"
    policy.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    restored = module.SelfLearningPolicy.load(path)
    from_json = module.SelfLearningPolicy.from_json(policy.to_json())

    assert payload["run_date"] == "20260413"
    assert payload["summary"] == {
        "n_threshold_overrides": 2,
        "n_property_budget_updates": 1,
        "n_repair_prompt_patches": 1,
        "n_routing_hints": 2,
    }
    assert restored.to_dict() == policy.to_dict() == from_json.to_dict()

    query = module.PolicyQuery.from_record(semantic_record, preferred_repair_outcome="improved")
    assert (
        restored.runtime_context(query).threshold_overrides
        == policy.runtime_context(query).threshold_overrides
    )


def test_runtime_policy_context_stays_additive_to_tracker_and_case_memory():
    """SCENARIO-VERIFY-059: runtime context stays additive to tracker and memory."""
    module = load_policy_module()
    memory, _, code_record = make_case_memory()
    tracker = make_tracker()
    policy = module.SelfLearningPolicyCompiler().compile(
        case_memory=memory,
        accepted_repairs=accepted_repairs(),
        tracker=tracker,
    )

    query = module.PolicyQuery.from_record(code_record, preferred_repair_outcome="improved")
    context = policy.runtime_context(query, tracker=tracker, case_memory=memory)

    assert context.tracker_stats["official_test_failure"]["fired"] == 5
    assert context.tracker_stats["official_test_failure"]["precision"] == pytest.approx(0.8)
    assert len(context.case_matches) == 1
    assert context.case_matches[0].entry.provenance[0].case_id == "humaneval-1"
    assert context.threshold_overrides[0].update_id.startswith("threshold:qwen3.5-0.8b")
    assert context.property_budget_updates[0].budget == 2
    assert context.repair_prompt_patches[0].success_rate == pytest.approx(2 / 3)
    assert context.routing_hints[0].route_to == "case_memory_then_repair"


def test_helper_paths_handle_empty_inputs_dict_rows_and_generic_runtime_context(tmp_path):
    """REQ-VERIFY-052 and REQ-VERIFY-053: helper branches stay deterministic on sparse input."""
    module = load_policy_module()
    case_module = load_case_module()

    generic_record = case_module.CaseRecord.normalize(
        benchmark="custom_eval",
        benchmark_slice="custom_eval/general_slice",
        model_name="Model-A",
        case_id="generic-1",
        violation_types=("semantic:unsupported_reference",),
        prompt_text="",
        description_texts=(),
        baseline_success=None,
        repair_success=None,
        confidence=0.2,
        source_experiment=None,
    )
    memory = case_module.CaseMemory()
    memory.record(generic_record)

    policy = module.SelfLearningPolicyCompiler(min_case_support=2, min_patch_support=2).compile(
        case_memory=memory,
        accepted_repairs=(
            {"snippet_id": "", "support": 0},
            {
                "snippet_id": "missing-models",
                "benchmark": "custom_eval",
                "domain": "general_slice",
                "template": "Patch text",
                "support": 2,
                "model_names": "Model-A",
            },
        ),
        tracker=None,
    )
    query = module.PolicyQuery.from_record(generic_record)
    context = policy.runtime_context(query, tracker=None, case_memory=memory)

    regressed_record = case_module.CaseRecord.normalize(
        benchmark="custom_eval",
        benchmark_slice="custom_eval/general_slice",
        model_name="Model-A",
        case_id="regressed-1",
        violation_types=("unsupported_reference",),
        prompt_text="",
        description_texts=("unsupported reference",),
        baseline_success=True,
        repair_success=False,
        confidence=0.95,
        source_experiment=1,
    )
    regressed_entry = case_module.CaseEntry(
        key=regressed_record.key,
        case_kind=regressed_record.case_kind,
        benchmark=regressed_record.benchmark,
        violation_types=regressed_record.violation_types,
        prompt_tokens=regressed_record.prompt_tokens,
        support=2,
        confidence=regressed_record.confidence,
        provenance=(regressed_record.provenance,),
    )
    no_family_entry = case_module.CaseEntry(
        key=case_module.CaseKey(
            model_name="Model-A",
            benchmark_slice="custom_eval/general_slice",
            violation_families=(),
            prompt_sketch="generic",
            property_names=(),
            repair_outcome="improved",
        ),
        case_kind="constraint_verification",
        benchmark="custom_eval",
        violation_types=("bare_violation",),
        prompt_tokens=(),
        support=2,
        confidence=0.95,
        provenance=(generic_record.provenance,),
    )
    empty_signal_entry = case_module.CaseEntry(
        key=case_module.CaseKey(
            model_name="Model-A",
            benchmark_slice="custom_eval/general_slice",
            violation_families=(),
            prompt_sketch="generic",
            property_names=(),
            repair_outcome="unknown",
        ),
        case_kind="constraint_verification",
        benchmark="custom_eval",
        violation_types=(),
        prompt_tokens=(),
        support=2,
        confidence=0.95,
        provenance=(generic_record.provenance,),
    )

    compiler = module.SelfLearningPolicyCompiler()
    regressed_override = compiler._compile_threshold_overrides(
        (regressed_entry, empty_signal_entry),
        {},
    )[0]
    regressed_routing = compiler._compile_routing_hints(
        (
            regressed_entry,
            empty_signal_entry,
        )
    )[0]

    assert policy.threshold_overrides == ()
    assert policy.property_budget_updates == ()
    assert policy.repair_prompt_patches == ()
    assert policy.routing_hints == ()
    assert context.tracker_stats == {}
    assert context.case_matches[0].entry.key.prompt_sketch == "generic"
    assert regressed_override.threshold_value > regressed_override.baseline_value
    assert regressed_routing.route_to == "verify_only"
    assert module._as_strings("Model-A") == ()
    assert module._weighted_mean(()) == 0.0
    assert module._primary_signal(no_family_entry) == "bare_violation"
    assert module._primary_signal(empty_signal_entry) == "generic"
    assert (
        module.SelfLearningPolicyCompiler._provenance_from_repair_row(
            {"provenance": "bad"},
            support=1,
            confidence=0.5,
        )
        == ()
    )
    assert (
        module.SelfLearningPolicyCompiler._provenance_from_repair_row(
            {"provenance": [None]},
            support=1,
            confidence=0.5,
        )
        == ()
    )
    assert module.SelfLearningPolicy.from_dict(policy.to_dict()) == policy
    with pytest.raises(ValueError, match="Unsupported self-learning policy format"):
        module.SelfLearningPolicy.from_dict({"version": 999})
    with pytest.raises(ValueError, match="JSON object"):
        module.SelfLearningPolicy.from_json("[]")
    bad_policy_path = tmp_path / "bad_policy.json"
    bad_policy_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        module.SelfLearningPolicy.load(bad_policy_path)
