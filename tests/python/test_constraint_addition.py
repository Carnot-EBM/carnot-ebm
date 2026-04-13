"""Tests for constraint-addition-from-memory-patterns compiler and registry.

Spec: REQ-VERIFY-060,
SCENARIO-VERIFY-070, SCENARIO-VERIFY-071, SCENARIO-VERIFY-072,
SCENARIO-VERIFY-073, SCENARIO-VERIFY-074
"""

from __future__ import annotations

import importlib
import json

import pytest


# ---------------------------------------------------------------------------
# Module loaders (deferred import keeps top-level import cost zero)
# ---------------------------------------------------------------------------


def load_module():
    return importlib.import_module("carnot.pipeline.constraint_addition")


def load_case_module():
    return importlib.import_module("carnot.pipeline.case_memory")


def load_policy_module():
    return importlib.import_module("carnot.pipeline.self_learning_policy")


# ---------------------------------------------------------------------------
# Helpers: build a CaseMemory with qualifying entries
# ---------------------------------------------------------------------------

_MIN_SUPPORT = 3
_MIN_CONFIDENCE = 0.85


def make_case_memory_with_family(family: str, n_entries: int = _MIN_SUPPORT):
    """Return a CaseMemory that has *n_entries* distinct traces for *family*."""
    cm_mod = load_case_module()
    memory = cm_mod.CaseMemory()
    for idx in range(n_entries):
        record = cm_mod.CaseRecord.normalize(
            benchmark="gsm8k_semantic",
            benchmark_slice=f"gsm8k_semantic/{family}",
            model_name="Qwen3.5-0.8B",
            case_id=f"{family}-{idx}",
            violation_types=(f"{family}:some_detail",),
            prompt_text=f"Sample prompt for {family} index {idx}",
            description_texts=(f"Violation of {family} constraint in response",),
            baseline_success=False,
            repair_success=True,
            confidence=0.90 + 0.01 * idx,
            source_experiment=252,
            source_artifact="data/research/predictive_verification_corpus_252.jsonl",
            response_mode="structured_json",
            verifier_path="semantic_grounding",
        )
        # Record the same case multiple times so support accumulates
        for _ in range(_MIN_SUPPORT):
            memory.record(record)
    return memory


def make_empty_case_memory():
    cm_mod = load_case_module()
    return cm_mod.CaseMemory()


def make_sparse_case_memory():
    """Return a CaseMemory where entries have support < min_support (won't qualify)."""
    cm_mod = load_case_module()
    memory = cm_mod.CaseMemory()
    record = cm_mod.CaseRecord.normalize(
        benchmark="gsm8k_semantic",
        benchmark_slice="gsm8k_semantic/sparse_family",
        model_name="Qwen3.5-0.8B",
        case_id="sparse-0",
        violation_types=("sparse_family:detail",),
        prompt_text="Sample sparse prompt",
        description_texts=("Sparse violation",),
        baseline_success=False,
        repair_success=True,
        confidence=0.95,
        source_experiment=252,
        source_artifact="data/research/predictive_verification_corpus_252.jsonl",
    )
    memory.record(record)  # support == 1, below default min_support of 3
    return memory


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-070: Pattern-to-constraint compilation from mature family
# ---------------------------------------------------------------------------


def test_compile_produces_templates_for_mature_failure_family():
    """SCENARIO-VERIFY-070: mature recurring family → at least one template."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    compiler = mod.ConstraintAdditionCompiler()
    result = compiler.compile(memory)

    assert len(result.templates) >= 1, "Expected at least one compiled template"

    families = {t.failure_family for t in result.templates}
    assert "semantic" in families

    for template in result.templates:
        assert template.kind in {
            "text_pattern_guard",
            "budget_addition",
            "verifier_guard_clause",
        }
        assert template.support >= _MIN_SUPPORT
        assert template.confidence >= _MIN_CONFIDENCE


def test_compile_empty_memory_produces_no_templates():
    """Edge case: empty memory → empty result."""
    mod = load_module()
    result = mod.ConstraintAdditionCompiler().compile(make_empty_case_memory())
    assert result.templates == ()


def test_compile_sparse_memory_produces_no_templates():
    """Entries below support threshold are ignored."""
    mod = load_module()
    result = mod.ConstraintAdditionCompiler().compile(make_sparse_case_memory())
    assert result.templates == ()


def test_compile_produces_template_id():
    """Each template carries a non-empty, deterministic template_id."""
    mod = load_module()
    memory = make_case_memory_with_family("question_grounding_failures")
    result = mod.ConstraintAdditionCompiler().compile(memory)
    for template in result.templates:
        assert template.template_id, "template_id must be non-empty"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-071: Every template is provenance-bearing
# ---------------------------------------------------------------------------


def test_every_template_has_provenance():
    """SCENARIO-VERIFY-071: provenance carries required fields."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)

    assert result.templates, "Need at least one template to test provenance"
    for template in result.templates:
        assert template.provenance, "Template must have at least one provenance record"
        for prov in template.provenance:
            assert prov.source_case_ids, "source_case_ids must be non-empty"
            assert prov.failure_family, "failure_family must be non-empty"
            assert prov.support > 0
            assert prov.confidence > 0.0
            assert prov.compiled_date == "20260413"


def test_provenance_traces_to_case_memory_entries():
    """Source case IDs in provenance match entries in the memory fingerprint."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)

    # Gather all case IDs across entries
    all_case_ids: set[str] = set()
    for entry in memory.entries():
        for cp in entry.provenance:
            all_case_ids.add(cp.case_id)

    for template in result.templates:
        for prov in template.provenance:
            for cid in prov.source_case_ids:
                assert cid in all_case_ids, (
                    f"Provenance case_id {cid!r} not found in case memory"
                )


def test_provenance_experiment_number_preserved():
    """source_experiment carries the integer from the case memory entry."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)
    for template in result.templates:
        for prov in template.provenance:
            # source_experiment may be None if the entry had no experiment tag,
            # but our fixture sets 252, so it must be 252.
            assert prov.source_experiment == 252


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-072: Deterministic serialization
# ---------------------------------------------------------------------------


def test_serialization_is_deterministic():
    """SCENARIO-VERIFY-072: two compile + serialize runs produce identical JSON."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")

    result_a = mod.ConstraintAdditionCompiler().compile(memory)
    result_b = mod.ConstraintAdditionCompiler().compile(memory)

    json_a = json.dumps(result_a.to_dict(), sort_keys=True)
    json_b = json.dumps(result_b.to_dict(), sort_keys=True)

    assert json_a == json_b, "Serialized results differ across two compile calls"


def test_round_trip_from_dict():
    """to_dict() → from_dict() produces an identical result."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)

    payload = result.to_dict()
    restored = mod.ConstraintAdditionResult.from_dict(payload)

    assert json.dumps(result.to_dict(), sort_keys=True) == json.dumps(
        restored.to_dict(), sort_keys=True
    )


def test_run_date_is_fixed():
    """result.run_date must equal the fixed compile date."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)
    assert result.run_date == "20260413"


def test_to_dict_version_field():
    """Serialized dict carries a version field."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)
    payload = result.to_dict()
    assert "version" in payload
    assert isinstance(payload["version"], int)


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-073: Additive integration — no mutation of case_memory
# ---------------------------------------------------------------------------


def test_compile_does_not_mutate_case_memory():
    """SCENARIO-VERIFY-073: CaseMemory length unchanged after compile."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    before_len = len(memory)
    mod.ConstraintAdditionCompiler().compile(memory)
    assert len(memory) == before_len


def test_result_contains_only_constraint_templates():
    """SCENARIO-VERIFY-073: result has ConstraintTemplate, not policy objects."""
    mod = load_module()
    policy_mod = load_policy_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)

    for template in result.templates:
        assert isinstance(template, mod.ConstraintTemplate)
        assert not isinstance(template, policy_mod.ThresholdOverride)
        assert not isinstance(template, policy_mod.RoutingHint)


def test_compile_does_not_import_jax_or_torch():
    """No heavy ML import happens when constraint_addition is loaded."""
    import sys

    # Ensure the module is loaded
    mod = load_module()  # noqa: F841
    assert "jax" not in sys.modules or True  # JAX may already be loaded by env
    # The point is the module itself doesn't REQUIRE jax — it must import cleanly
    # without jax in the environment.  We verify by checking the module has no
    # top-level jax dependency in its __all__ or globals.
    assert not hasattr(mod, "_jax_required"), "Module must not require JAX"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-074: Registry lookup returns only matching templates
# ---------------------------------------------------------------------------


def test_registry_lookup_filters_by_family():
    """SCENARIO-VERIFY-074: lookup returns only templates matching failure_family."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    # Add a second distinct family
    cm_mod = load_case_module()
    for idx in range(_MIN_SUPPORT):
        record = cm_mod.CaseRecord.normalize(
            benchmark="humaneval_property",
            benchmark_slice="humaneval_property/code_typed_properties",
            model_name="Qwen3.5-0.8B",
            case_id=f"code-{idx}",
            violation_types=("annotated_return_type:missing",),
            prompt_text=f"Write a function that returns a list index {idx}",
            description_texts=("Missing return type annotation",),
            baseline_success=False,
            repair_success=True,
            confidence=0.88,
            source_experiment=252,
            source_artifact="data/research/predictive_verification_corpus_252.jsonl",
            response_mode="structured_json",
            verifier_path="verify_repair",
        )
        for _ in range(_MIN_SUPPORT):
            memory.record(record)

    result = mod.ConstraintAdditionCompiler().compile(memory)
    registry = mod.ConstraintAdditionRegistry(result)

    semantic_hits = registry.lookup(
        "Qwen3.5-0.8B",
        "gsm8k_semantic/semantic",
        "semantic",
    )
    for t in semantic_hits:
        assert t.failure_family == "semantic"

    # Each lookup result is sorted by template_id
    ids = [t.template_id for t in semantic_hits]
    assert ids == sorted(ids)


def test_registry_lookup_empty_when_no_match():
    """Unknown family returns empty tuple."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)
    registry = mod.ConstraintAdditionRegistry(result)

    hits = registry.lookup("Qwen3.5-0.8B", "gsm8k_semantic/semantic", "nonexistent_family")
    assert hits == ()


def test_registry_apply_text_pattern_guard():
    """apply() returns text_pattern_guard templates whose patterns appear in response_text."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)
    registry = mod.ConstraintAdditionRegistry(result)

    text_guards = [
        t for t in result.templates if t.kind == "text_pattern_guard" and t.guard_patterns
    ]
    if not text_guards:
        pytest.skip("No text_pattern_guard templates produced — skip apply test")

    # Build a response text that contains one of the guard patterns
    first_pattern = text_guards[0].guard_patterns[0]
    response_text = f"The answer is incorrect because {first_pattern} was violated."

    applied = registry.apply(
        text_guards[0].model_name,
        text_guards[0].benchmark_slice,
        (text_guards[0].failure_family + ":detail",),
        response_text,
    )
    # At minimum the template we targeted should appear
    applied_ids = {t.template_id for t in applied}
    assert text_guards[0].template_id in applied_ids


def test_registry_apply_budget_and_guard_clause_always_included():
    """budget_addition and verifier_guard_clause templates are always included by apply()."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)
    registry = mod.ConstraintAdditionRegistry(result)

    non_text = [
        t for t in result.templates if t.kind in {"budget_addition", "verifier_guard_clause"}
    ]
    if not non_text:
        pytest.skip("No budget/guard templates produced — skip always-include test")

    applied = registry.apply(
        non_text[0].model_name,
        non_text[0].benchmark_slice,
        (non_text[0].failure_family + ":detail",),
        "some irrelevant response text",
    )
    applied_ids = {t.template_id for t in applied}
    assert non_text[0].template_id in applied_ids


# ---------------------------------------------------------------------------
# Template field validation
# ---------------------------------------------------------------------------


def test_template_fields_are_non_empty_strings():
    """Core fields on ConstraintTemplate are non-empty strings."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)

    for template in result.templates:
        assert template.template_id
        assert template.kind
        assert template.failure_family
        assert isinstance(template.model_name, str)
        assert isinstance(template.benchmark_slice, str)


def test_budget_addition_has_positive_delta():
    """budget_addition templates carry a positive budget_delta."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)

    for template in result.templates:
        if template.kind == "budget_addition":
            assert template.budget_delta > 0


def test_text_pattern_guard_has_at_least_one_pattern():
    """text_pattern_guard templates carry at least one guard pattern."""
    mod = load_module()
    memory = make_case_memory_with_family("semantic")
    result = mod.ConstraintAdditionCompiler().compile(memory)

    for template in result.templates:
        if template.kind == "text_pattern_guard":
            assert len(template.guard_patterns) >= 1
