#!/usr/bin/env python3
"""Experiment 211: constraint IR benchmark for semantic grounding.

This workflow writes:

- ``data/research/constraint_ir_benchmark_211.jsonl``
- ``results/experiment_211_results.json``

The benchmark is deterministic and self-contained. It mixes:

- live GSM8K semantic or question-grounding failures from Exp 203 / 206 / 207
- multi-constraint instruction-following prompts inspired by recent
  verifier-style instruction benchmarks
- code prompts whose requirements can be represented as typed properties

Spec: REQ-VERIFY-011, REQ-VERIFY-012, SCENARIO-VERIFY-011,
SCENARIO-VERIFY-012
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

RUN_DATE = "20260412"
EXPERIMENT_LABEL = "Exp 211"

SURFACE_ONLY_RATIONALE = "Surface-form checks are primary; hidden reasoning is unnecessary."
SCHEMA_ONLY_RATIONALE = "The answer schema is fully specified at the surface level."
GROUNDED_EVIDENCE_RATIONALE = (
    "All required evidence is in the prompt, so explicit reasoning can be checked."
)
REWRITE_RATIONALE = "The task is judged by transformed output, not by hidden rationale."
DECISION_RATIONALE = "Decision evidence can be matched directly back to prompt records."
PLAN_RATIONALE = "The plan structure is explicit and does not require trusting hidden reasoning."
DERIVED_VALUE_RATIONALE = "The prompt provides the source facts needed to verify the derived field."
CODE_CONTRACT_RATIONALE = (
    "Verification should rely on typed contracts and execution, not free-form code rationale."
)


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


REPO_ROOT = get_repo_root()
BENCHMARK_PATH = REPO_ROOT / "data" / "research" / "constraint_ir_benchmark_211.jsonl"
RESULTS_PATH = REPO_ROOT / "results" / "experiment_211_results.json"


def make_constraint(
    constraint_id: str,
    constraint_type: str,
    target: str,
    relation: str,
    value: Any,
    *,
    unit: str | None = None,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "constraint_id": constraint_id,
        "type": constraint_type,
        "target": target,
        "relation": relation,
        "value": value,
    }
    if unit is not None:
        item["unit"] = unit
    if depends_on is not None:
        item["depends_on"] = depends_on
    return item


def make_example(
    *,
    example_id: str,
    source_family: str,
    source_refs: list[str],
    prompt: str,
    gold_atomic_constraints: list[dict[str, Any]],
    constraint_types: list[str],
    expected_verifier_path: str,
    expected_answer_schema: dict[str, Any],
    free_form_reasoning_monitorable: bool,
    monitorability_rationale: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "source_family": source_family,
        "source_refs": source_refs,
        "prompt": prompt,
        "gold_atomic_constraints": gold_atomic_constraints,
        "constraint_types": constraint_types,
        "expected_verifier_path": expected_verifier_path,
        "expected_answer_schema": expected_answer_schema,
        "free_form_reasoning_monitorable": free_form_reasoning_monitorable,
        "monitorability_rationale": monitorability_rationale,
        "notes": notes,
    }


def number_schema(numeric_type: str = "integer") -> dict[str, str]:
    return {"type": "number", "numeric_type": numeric_type}


def bullet_list_schema(count: int) -> dict[str, Any]:
    return {"type": "bullet_list", "count": count}


def json_object_schema(required_keys: list[str]) -> dict[str, Any]:
    return {"type": "json_object", "required_keys": required_keys}


def markdown_sections_schema(required_sections: list[str]) -> dict[str, Any]:
    return {"type": "markdown_sections", "required_sections": required_sections}


def yaml_object_schema(required_keys: list[str]) -> dict[str, Any]:
    return {"type": "yaml_object", "required_keys": required_keys}


def text_schema(text_type: str) -> dict[str, str]:
    return {"type": text_type}


def python_function_schema(name: str, signature: str) -> dict[str, str]:
    return {
        "type": "python_function",
        "name": name,
        "signature": signature,
    }


def make_surface_bullet_example(
    *,
    example_id: str,
    prompt: str,
    required_tokens: list[str],
    forbidden_tokens: list[str],
    notes: str,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "count_exact", "bullet_count", "equals", 3),
        make_constraint("c2", "word_count_range", "bullet_word_count", "between", [4, 7]),
    ]
    constraints.extend(
        make_constraint(
            f"c{index}",
            "must_include_token",
            "answer_surface",
            "contains",
            token,
        )
        for index, token in enumerate(required_tokens, start=3)
    )
    constraints.extend(
        make_constraint(
            f"c{index}",
            "forbidden_token",
            "answer_surface",
            "not_contains",
            token,
        )
        for index, token in enumerate(
            forbidden_tokens,
            start=3 + len(required_tokens),
        )
    )
    return make_example(
        example_id=example_id,
        source_family="instruction_following",
        source_refs=["vifbench-inspired", "cfbench-inspired"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["literal"],
        expected_verifier_path="prompt_ir.surface_schema",
        expected_answer_schema=bullet_list_schema(3),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=SURFACE_ONLY_RATIONALE,
        notes=notes,
    )


def make_json_object_example(
    *,
    example_id: str,
    prompt: str,
    required_keys: list[str],
    enum_constraints: dict[str, list[str]],
    notes: str,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "json_exact_keys", "json_keys", "equals", required_keys),
        make_constraint("c2", "no_extra_keys", "json_keys", "subset_equals", required_keys),
    ]
    constraints.extend(
        make_constraint(
            f"c{index}",
            "enum_membership",
            key,
            "in",
            allowed_values,
        )
        for index, (key, allowed_values) in enumerate(enum_constraints.items(), start=3)
    )
    return make_example(
        example_id=example_id,
        source_family="instruction_following",
        source_refs=["constraintbench-inspired", "cfbench-inspired"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["literal", "compositional"],
        expected_verifier_path="prompt_ir.surface_json_schema",
        expected_answer_schema=json_object_schema(required_keys),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=SCHEMA_ONLY_RATIONALE,
        notes=notes,
    )


def make_markdown_sections_example(
    *,
    example_id: str,
    prompt: str,
    sections: list[str],
    required_tokens: list[str],
    notes: str,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "section_order", "markdown_sections", "equals", sections),
        make_constraint("c2", "sentence_count_per_section", "section_sentence_count", "equals", 1),
    ]
    constraints.extend(
        make_constraint(
            f"c{index}",
            "must_include_token",
            section.lower(),
            "contains",
            token,
        )
        for index, (section, token) in enumerate(
            zip(sections, required_tokens, strict=True), start=3
        )
    )
    return make_example(
        example_id=example_id,
        source_family="instruction_following",
        source_refs=["followbench-inspired", "realinstruct-inspired"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["literal", "compositional"],
        expected_verifier_path="prompt_ir.ordered_sections",
        expected_answer_schema=markdown_sections_schema(sections),
        free_form_reasoning_monitorable=False,
        monitorability_rationale="Section order and content are checked directly on the answer.",
        notes=notes,
    )


def make_grounded_selection_example(
    *,
    example_id: str,
    prompt: str,
    selected_ids: list[str],
    ordering: str,
    answer_type: str,
    notes: str,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "count_exact", "selection_count", "equals", len(selected_ids)),
        make_constraint("c2", "grounded_selection", "selected_ids", "equals", selected_ids),
        make_constraint("c3", "ordering", "selected_ids", "ordered_by", ordering),
    ]
    return make_example(
        example_id=example_id,
        source_family="instruction_following",
        source_refs=["vifbench-inspired", "constraintbench-inspired"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["compositional", "semantic_grounding"],
        expected_verifier_path="prompt_ir.grounded_selection",
        expected_answer_schema=text_schema(answer_type),
        free_form_reasoning_monitorable=True,
        monitorability_rationale=GROUNDED_EVIDENCE_RATIONALE,
        notes=notes,
    )


def make_rewrite_example(
    *,
    example_id: str,
    prompt: str,
    required_phrases: list[str],
    forbidden_phrases: list[str],
    word_range: list[int],
    notes: str,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "word_count_range", "answer_word_count", "between", word_range),
        make_constraint("c2", "tone", "answer_style", "equals", "professional_and_calm"),
    ]
    constraints.extend(
        make_constraint(
            f"c{index}",
            "must_include_phrase",
            "answer_surface",
            "contains",
            phrase,
        )
        for index, phrase in enumerate(required_phrases, start=3)
    )
    constraints.extend(
        make_constraint(
            f"c{index}",
            "forbidden_phrase",
            "answer_surface",
            "not_contains",
            phrase,
        )
        for index, phrase in enumerate(forbidden_phrases, start=3 + len(required_phrases))
    )
    return make_example(
        example_id=example_id,
        source_family="instruction_following",
        source_refs=["realinstruct-inspired", "cfbench-inspired"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["literal", "compositional"],
        expected_verifier_path="prompt_ir.rewrite_constraints",
        expected_answer_schema=text_schema("paragraph"),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=REWRITE_RATIONALE,
        notes=notes,
    )


def make_decision_json_example(
    *,
    example_id: str,
    prompt: str,
    choice: str,
    evidence_ids: list[str],
    notes: str,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "json_exact_keys", "json_keys", "equals", ["choice", "evidence"]),
        make_constraint("c2", "grounded_selection", "choice", "equals", choice),
        make_constraint("c3", "grounded_evidence_ids", "evidence", "equals", evidence_ids),
    ]
    return make_example(
        example_id=example_id,
        source_family="instruction_following",
        source_refs=["vifbench-inspired", "cfbench-inspired"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["literal", "semantic_grounding"],
        expected_verifier_path="prompt_ir.grounded_decision_json",
        expected_answer_schema=json_object_schema(["choice", "evidence"]),
        free_form_reasoning_monitorable=True,
        monitorability_rationale=DECISION_RATIONALE,
        notes=notes,
    )


def make_plan_example(
    *,
    example_id: str,
    prompt: str,
    required_step_roles: list[str],
    forbidden_tokens: list[str],
    notes: str,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "step_count", "plan_steps", "equals", 4),
        make_constraint("c2", "step_roles", "plan_roles", "equals", required_step_roles),
    ]
    constraints.extend(
        make_constraint(
            f"c{index}",
            "forbidden_token",
            "answer_surface",
            "not_contains",
            token,
        )
        for index, token in enumerate(forbidden_tokens, start=3)
    )
    return make_example(
        example_id=example_id,
        source_family="instruction_following",
        source_refs=["constraintbench-inspired", "realinstruct-inspired"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["literal", "compositional"],
        expected_verifier_path="prompt_ir.plan_schema",
        expected_answer_schema=text_schema("numbered_list"),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=PLAN_RATIONALE,
        notes=notes,
    )


def make_yaml_extract_example(
    *,
    example_id: str,
    prompt: str,
    required_keys: list[str],
    derived_value: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "yaml_exact_keys", "yaml_keys", "equals", required_keys),
        make_constraint(
            "c2", "derived_value", derived_value["target"], "equals", derived_value["value"]
        ),
    ]
    return make_example(
        example_id=example_id,
        source_family="instruction_following",
        source_refs=["followbench-inspired", "vifbench-inspired"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["literal", "semantic_grounding"],
        expected_verifier_path="prompt_ir.grounded_yaml_extraction",
        expected_answer_schema=yaml_object_schema(required_keys),
        free_form_reasoning_monitorable=True,
        monitorability_rationale=DERIVED_VALUE_RATIONALE,
        notes=notes,
    )


def make_negation_example(
    *,
    example_id: str,
    prompt: str,
    selected_items: list[str],
    notes: str,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "count_exact", "selection_count", "equals", len(selected_items)),
        make_constraint("c2", "grounded_selection", "selected_items", "equals", selected_items),
        make_constraint(
            "c3",
            "negation_scope",
            "selection_rule",
            "equals",
            "exclude_items_matching_forbidden_property",
        ),
    ]
    return make_example(
        example_id=example_id,
        source_family="instruction_following",
        source_refs=["constraintbench-inspired", "cfbench-inspired"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["compositional", "semantic_grounding"],
        expected_verifier_path="prompt_ir.negation_scope_grounding",
        expected_answer_schema=text_schema("comma_separated_list"),
        free_form_reasoning_monitorable=True,
        monitorability_rationale="The prompt enumerates all candidates and exclusions explicitly.",
        notes=notes,
    )


def make_two_sentence_example(
    *,
    example_id: str,
    prompt: str,
    required_token: str,
    required_uncertainty: str,
    notes: str,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "sentence_count", "answer_sentence_count", "equals", 2),
        make_constraint("c2", "must_include_token", "sentence_1", "contains", required_token),
        make_constraint(
            "c3", "must_include_phrase", "sentence_2", "contains", required_uncertainty
        ),
        make_constraint("c4", "forbidden_token", "answer_surface", "not_contains", "I"),
    ]
    return make_example(
        example_id=example_id,
        source_family="instruction_following",
        source_refs=["realinstruct-inspired", "followbench-inspired"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["literal", "compositional"],
        expected_verifier_path="prompt_ir.answer_plus_uncertainty",
        expected_answer_schema=text_schema("two_sentences"),
        free_form_reasoning_monitorable=False,
        monitorability_rationale="Only the explicit two-sentence output format matters here.",
        notes=notes,
    )


def make_python_function_example(
    *,
    example_id: str,
    prompt: str,
    name: str,
    signature: str,
    return_type: str,
    semantic_constraints: list[dict[str, Any]],
    notes: str,
    forbidden_apis: list[str] | None = None,
    raises: str | None = None,
    complexity: str | None = None,
    preserves_input: bool = False,
) -> dict[str, Any]:
    constraints = [
        make_constraint("c1", "function_name", "function_name", "equals", name),
        make_constraint("c2", "signature", "signature", "equals", signature),
        make_constraint("c3", "return_type", "return_type", "equals", return_type),
    ]
    if complexity is not None:
        constraints.append(
            make_constraint("c4", "time_complexity", "time_complexity", "equals", complexity)
        )
        next_constraint_id = 5
    else:
        next_constraint_id = 4
    if preserves_input:
        constraints.append(
            make_constraint(
                f"c{next_constraint_id}",
                "input_immutability",
                "input_object",
                "not_mutated",
                True,
            )
        )
        next_constraint_id += 1
    if raises is not None:
        constraints.append(
            make_constraint(
                f"c{next_constraint_id}",
                "error_policy",
                "raised_exception",
                "equals",
                raises,
            )
        )
        next_constraint_id += 1
    if forbidden_apis is not None:
        constraints.append(
            make_constraint(
                f"c{next_constraint_id}",
                "forbidden_api",
                "implementation",
                "not_contains",
                forbidden_apis,
            )
        )
        next_constraint_id += 1
    for offset, semantic_constraint in enumerate(semantic_constraints, start=next_constraint_id):
        semantic_copy = dict(semantic_constraint)
        semantic_copy["constraint_id"] = f"c{offset}"
        constraints.append(semantic_copy)
    return make_example(
        example_id=example_id,
        source_family="code_typed_properties",
        source_refs=["formalbench-inspired", "code-properties-curated"],
        prompt=prompt,
        gold_atomic_constraints=constraints,
        constraint_types=["typed_property", "compositional"],
        expected_verifier_path="code_ir.typed_contracts_plus_execution",
        expected_answer_schema=python_function_schema(name, signature),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=CODE_CONTRACT_RATIONALE,
        notes=notes,
    )


LIVE_GSM8K_EXAMPLES: list[dict[str, Any]] = [
    make_example(
        example_id="exp211-live-gsm8k-923",
        source_family="live_gsm8k_semantic_failure",
        source_refs=["exp203:923", "exp206:923", "exp207:923"],
        prompt=(
            "Lana is brewing cups of tea for her friends. She has 27 cups, and she "
            "divides these into 3 rows. In each row, she creates equal amounts of "
            "chamomile and mint tea cups. She then uses the remaining cups to brew "
            "a total of 15 cups of cinnamon tea. How many cups of mint tea are in "
            "each row?"
        ),
        gold_atomic_constraints=[
            make_constraint(
                "c1",
                "derived_quantity",
                "chamomile_and_mint_total",
                "equals",
                "27 - 15",
                unit="cups",
            ),
            make_constraint(
                "c2",
                "equal_partition",
                "cups_per_row",
                "equals",
                "chamomile_and_mint_total / 3",
                unit="cups",
                depends_on=["c1"],
            ),
            make_constraint(
                "c3",
                "equal_split",
                "mint_per_row",
                "equals",
                "cups_per_row / 2",
                unit="cups",
                depends_on=["c2"],
            ),
            make_constraint(
                "c4",
                "final_answer_binding",
                "answer",
                "equals",
                "mint_per_row",
                depends_on=["c3"],
            ),
        ],
        constraint_types=["compositional", "semantic_grounding"],
        expected_verifier_path="question_grounding.quantity_graph",
        expected_answer_schema=number_schema(),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=(
            "Prompt-grounded quantities matter more than trusting free-form chain-of-thought."
        ),
        notes=(
            "Exp 203 autopsy: the model stopped at a row total instead of "
            "the mint-per-row quantity."
        ),
    ),
    make_example(
        example_id="exp211-live-gsm8k-814",
        source_family="live_gsm8k_semantic_failure",
        source_refs=["exp203:814", "exp206:814", "exp207:814"],
        prompt=(
            "A new arcade opens up and Jack decides to play with his 3 friends. "
            "Jack can play a game with 1 quarter for 20 minutes. Two of his friends "
            "are significantly worse than him and can only play half as long. One "
            "of them is significantly better and can play for 1.5 times as long. "
            "They play for 4 hours. How much money is used?"
        ),
        gold_atomic_constraints=[
            make_constraint(
                "c1",
                "base_rate_binding",
                "jack_minutes_per_quarter",
                "equals",
                20,
                unit="minutes",
            ),
            make_constraint(
                "c2",
                "semantic_modifier",
                "worse_friend_minutes_per_quarter",
                "equals",
                "jack_minutes_per_quarter / 2",
                unit="minutes",
                depends_on=["c1"],
            ),
            make_constraint(
                "c3",
                "semantic_modifier",
                "better_friend_minutes_per_quarter",
                "equals",
                "jack_minutes_per_quarter * 1.5",
                unit="minutes",
                depends_on=["c1"],
            ),
            make_constraint(
                "c4",
                "discrete_counting",
                "cost_model",
                "equals",
                "money_is_quarter_count_times_0.25",
            ),
        ],
        constraint_types=["semantic_grounding", "compositional"],
        expected_verifier_path="question_grounding.unit_reasoner",
        expected_answer_schema=number_schema(),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=(
            "The key failure is semantic binding of 'half as long' and "
            "'1.5 times as long,' not the free-form trace."
        ),
        notes=(
            "Exp 203 autopsy: the model turned duration-per-quarter "
            "semantics into a continuous per-minute rate."
        ),
    ),
    make_example(
        example_id="exp211-live-gsm8k-943",
        source_family="live_gsm8k_semantic_failure",
        source_refs=["exp203:943", "exp206:943", "exp207:943"],
        prompt=(
            "James gets 10 new CDs. Each CD cost $15. He gets them for 40% off. "
            "He decides he doesn't like 5 of them and sells them for 40. How much "
            "money was he out?"
        ),
        gold_atomic_constraints=[
            make_constraint("c1", "discounted_total", "purchase_total", "equals", "10 * 15 * 0.60"),
            make_constraint("c2", "scope_alignment", "resale_total", "equals", 40, unit="dollars"),
            make_constraint(
                "c3",
                "net_difference",
                "loss",
                "equals",
                "purchase_total - resale_total",
                unit="dollars",
                depends_on=["c1", "c2"],
            ),
            make_constraint(
                "c4",
                "final_answer_binding",
                "answer",
                "equals",
                "loss",
                depends_on=["c3"],
            ),
        ],
        constraint_types=["semantic_grounding", "compositional"],
        expected_verifier_path="question_grounding.scope_alignment",
        expected_answer_schema=number_schema(),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=(
            "The important signal is that 'for 40' scopes to the whole resale event, not each CD."
        ),
        notes="Exp 203 autopsy: the model treated the resale amount as per-item instead of total.",
    ),
    make_example(
        example_id="exp211-live-gsm8k-506",
        source_family="live_gsm8k_semantic_failure",
        source_refs=["exp206:506", "exp207:506"],
        prompt=(
            "Mario needs to buy snowshoes for his 6 sled dogs. Assuming his dogs "
            "each has four legs and each pair of snowshoes costs $12.00, how much "
            "will it cost him to buy snowshoes for all of his dogs?"
        ),
        gold_atomic_constraints=[
            make_constraint("c1", "count_binding", "legs_per_dog", "equals", 4),
            make_constraint(
                "c2",
                "unit_conversion",
                "pairs_per_dog",
                "equals",
                "legs_per_dog / 2",
                depends_on=["c1"],
            ),
            make_constraint(
                "c3",
                "total_pairs",
                "snowshoe_pairs_needed",
                "equals",
                "6 * pairs_per_dog",
                depends_on=["c2"],
            ),
            make_constraint(
                "c4",
                "final_cost",
                "answer",
                "equals",
                "snowshoe_pairs_needed * 12",
                unit="dollars",
                depends_on=["c3"],
            ),
        ],
        constraint_types=["semantic_grounding", "compositional"],
        expected_verifier_path="question_grounding.count_expansion",
        expected_answer_schema=number_schema(),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=(
            "The verifier should expand the 'four legs' fact into a count "
            "of snowshoe pairs directly."
        ),
        notes=(
            "Live cohort case: the wrong answer collapsed four legs into "
            "one pair instead of two pairs per dog."
        ),
    ),
    make_example(
        example_id="exp211-live-gsm8k-1019",
        source_family="live_gsm8k_semantic_failure",
        source_refs=["exp206:1019", "exp207:1019"],
        prompt=(
            "On a busy Saturday morning, a hotel was completely booked with 100 "
            "guests. 24 guests elected an early checkout and 15 elected for a late "
            "checkout. In the afternoon twice as many people checked in as those "
            "who opted for a late checkout. 7 more people checked in after dinner "
            "was served. How many guests does the hotel now have?"
        ),
        gold_atomic_constraints=[
            make_constraint("c1", "initial_quantity", "starting_guests", "equals", 100),
            make_constraint("c2", "event_effect", "checkout_guests", "equals", "24 + 15"),
            make_constraint("c3", "derived_quantity", "afternoon_checkins", "equals", "2 * 15"),
            make_constraint("c4", "event_effect", "after_dinner_checkins", "equals", 7),
            make_constraint(
                "c5",
                "timeline_accumulation",
                "answer",
                "equals",
                "starting_guests - checkout_guests + afternoon_checkins + after_dinner_checkins",
                depends_on=["c1", "c2", "c3", "c4"],
            ),
        ],
        constraint_types=["semantic_grounding", "compositional"],
        expected_verifier_path="question_grounding.event_timeline",
        expected_answer_schema=number_schema(),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=(
            "The key error mode is event-timeline bookkeeping, "
            "not whether the prose sounds coherent."
        ),
        notes=(
            "Live cohort case: 'late checkout' still describes a checkout "
            "event that must be accounted for by the end state."
        ),
    ),
    make_example(
        example_id="exp211-live-gsm8k-515",
        source_family="live_gsm8k_semantic_failure",
        source_refs=["exp206:515", "exp207:515"],
        prompt=(
            "Tara bought 8 packs of 5 canvas bags for $4 each. She painted them and "
            "sold them at a craft fair for $8 each. How much profit did she earn on "
            "her bags?"
        ),
        gold_atomic_constraints=[
            make_constraint("c1", "count_binding", "bags_per_pack", "equals", 5),
            make_constraint(
                "c2", "scope_alignment", "purchase_price_scope", "equals", "price_applies_per_bag"
            ),
            make_constraint(
                "c3",
                "total_cost",
                "cost_total",
                "equals",
                "8 * bags_per_pack * 4",
                unit="dollars",
                depends_on=["c1", "c2"],
            ),
            make_constraint(
                "c4",
                "total_revenue",
                "revenue_total",
                "equals",
                "8 * bags_per_pack * 8",
                unit="dollars",
                depends_on=["c1"],
            ),
            make_constraint(
                "c5",
                "final_answer_binding",
                "answer",
                "equals",
                "revenue_total - cost_total",
                unit="dollars",
                depends_on=["c3", "c4"],
            ),
        ],
        constraint_types=["semantic_grounding", "compositional"],
        expected_verifier_path="question_grounding.scope_alignment",
        expected_answer_schema=number_schema(),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=(
            "The verifier needs to ground what 'for $4 each' modifies before any arithmetic chain."
        ),
        notes=(
            "Live cohort case: the wrong answer treated $4 as a per-pack "
            "price instead of a per-bag price."
        ),
    ),
    make_example(
        example_id="exp211-live-gsm8k-1077",
        source_family="live_gsm8k_semantic_failure",
        source_refs=["exp206:1077", "exp207:1077"],
        prompt=(
            "Paul is at a train station and is waiting for his train. He isn't sure "
            "how long he needs to wait, but he knows that the fourth train "
            "scheduled to arrive at the station is the one he needs to get on. The "
            "first train is scheduled to arrive in 10 minutes, and this train will "
            "stay in the station for 20 minutes. The second train is to arrive half "
            "an hour after the first train leaves the station, and this second "
            "train will stay in the station for a quarter of the amount of time "
            "that the first train stayed in the station. The third train is to "
            "arrive an hour after the second train leaves the station, and this "
            "third train is to leave the station immediately after it arrives. The "
            "fourth train will arrive 20 minutes after the third train leaves, and "
            "this is the train Paul will board. In total, how long, in minutes, "
            "will Paul wait for his train?"
        ),
        gold_atomic_constraints=[
            make_constraint(
                "c1", "arrival_time", "first_leave", "equals", "10 + 20", unit="minutes"
            ),
            make_constraint(
                "c2",
                "arrival_time",
                "second_arrive",
                "equals",
                "first_leave + 30",
                unit="minutes",
                depends_on=["c1"],
            ),
            make_constraint(
                "c3",
                "arrival_time",
                "second_leave",
                "equals",
                "second_arrive + 5",
                unit="minutes",
                depends_on=["c2"],
            ),
            make_constraint(
                "c4",
                "arrival_time",
                "third_arrive",
                "equals",
                "second_leave + 60",
                unit="minutes",
                depends_on=["c3"],
            ),
            make_constraint(
                "c5",
                "final_answer_binding",
                "answer",
                "equals",
                "third_arrive + 20",
                unit="minutes",
                depends_on=["c4"],
            ),
        ],
        constraint_types=["semantic_grounding", "compositional"],
        expected_verifier_path="question_grounding.temporal_schedule",
        expected_answer_schema=number_schema(),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=(
            "A prompt-grounded event graph is more reliable than trusting "
            "chain-of-thought narration."
        ),
        notes=(
            "Live cohort case: the wrong answer dropped one interval in the arrival/leave schedule."
        ),
    ),
    make_example(
        example_id="exp211-live-gsm8k-796",
        source_family="live_gsm8k_semantic_failure",
        source_refs=["exp206:796", "exp207:796"],
        prompt=(
            "A company's HR hires 20 new employees every month to add to its total "
            "workforce. If the company's initial employee number is 200, and each "
            "employee is paid a $4000 salary per month, calculate the total amount "
            "of money the company pays to its employees after three months?"
        ),
        gold_atomic_constraints=[
            make_constraint("c1", "monthly_state", "month_1_employees", "equals", 220),
            make_constraint("c2", "monthly_state", "month_2_employees", "equals", 240),
            make_constraint("c3", "monthly_state", "month_3_employees", "equals", 260),
            make_constraint(
                "c4",
                "aggregation",
                "total_payroll",
                "equals",
                "(month_1_employees + month_2_employees + month_3_employees) * 4000",
                unit="dollars",
                depends_on=["c1", "c2", "c3"],
            ),
            make_constraint(
                "c5", "final_answer_binding", "answer", "equals", "total_payroll", depends_on=["c4"]
            ),
        ],
        constraint_types=["semantic_grounding", "compositional"],
        expected_verifier_path="question_grounding.repeated_event_accumulator",
        expected_answer_schema=number_schema(),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=(
            "The prompt defines a month-by-month accumulator that should be modeled directly."
        ),
        notes=(
            "Live cohort case: the wrong answer failed to sum payroll over "
            "all three monthly workforce states."
        ),
    ),
    make_example(
        example_id="exp211-live-gsm8k-1309",
        source_family="live_gsm8k_semantic_failure",
        source_refs=["exp206:1309", "exp207:1309"],
        prompt=(
            "The girls are trying to raise money for a carnival. Kim raises $320 "
            "more than Alexandra, who raises $430, and Maryam raises $400 more "
            "than Sarah, who raises $300. How much money, in dollars, did they all "
            "raise in total?"
        ),
        gold_atomic_constraints=[
            make_constraint("c1", "grounded_value", "alexandra", "equals", 430, unit="dollars"),
            make_constraint(
                "c2",
                "grounded_value",
                "kim",
                "equals",
                "430 + 320",
                unit="dollars",
                depends_on=["c1"],
            ),
            make_constraint("c3", "grounded_value", "sarah", "equals", 300, unit="dollars"),
            make_constraint(
                "c4",
                "grounded_value",
                "maryam",
                "equals",
                "300 + 400",
                unit="dollars",
                depends_on=["c3"],
            ),
            make_constraint(
                "c5",
                "aggregation",
                "answer",
                "equals",
                "alexandra + kim + sarah + maryam",
                unit="dollars",
                depends_on=["c1", "c2", "c3", "c4"],
            ),
        ],
        constraint_types=["semantic_grounding", "compositional"],
        expected_verifier_path="question_grounding.annotation_review",
        expected_answer_schema=number_schema(),
        free_form_reasoning_monitorable=False,
        monitorability_rationale=(
            "Prompt-grounded arithmetic should be reconstructed directly, "
            "even when benchmark labels warrant review."
        ),
        notes=(
            "Live cohort anomaly: prompt-grounded arithmetic sums to 2180, "
            "so this case is retained as a label-conflict review example."
        ),
    ),
]


def build_live_gsm8k_examples() -> list[dict[str, Any]]:
    return list(LIVE_GSM8K_EXAMPLES)


def build_instruction_examples() -> list[dict[str, Any]]:
    examples = [
        make_surface_bullet_example(
            example_id="exp211-instruction-bullets-1",
            prompt=(
                "Summarize the release handoff in exactly 3 bullet points. Each "
                "bullet must be 4-7 words. Mention the tokens risk, owner, and "
                "deadline somewhere in the answer. Do not use the word urgent."
            ),
            required_tokens=["risk", "owner", "deadline"],
            forbidden_tokens=["urgent"],
            notes="Literal bullet-count and token-presence control.",
        ),
        make_surface_bullet_example(
            example_id="exp211-instruction-bullets-2",
            prompt=(
                "Give exactly 3 bullet points for the incident update. Each bullet "
                "must be 4-7 words. Include the tokens status, blocker, and next "
                "step. Do not use the word apology."
            ),
            required_tokens=["status", "blocker", "next step"],
            forbidden_tokens=["apology"],
            notes="Literal bullet-count and forbidden-token control.",
        ),
        make_surface_bullet_example(
            example_id="exp211-instruction-bullets-3",
            prompt=(
                "Produce exactly 3 bullet points for the hiring brief. Each bullet "
                "must be 4-7 words. Include role, interviewer, and timeline. Do "
                "not use the word flexible."
            ),
            required_tokens=["role", "interviewer", "timeline"],
            forbidden_tokens=["flexible"],
            notes="Literal summary-format constraint bundle.",
        ),
        make_json_object_example(
            example_id="exp211-instruction-json-1",
            prompt=(
                "Return JSON only with keys action, reason, and confidence. action "
                "must be one of approve, hold, reject. confidence must be one of "
                "low, medium, high."
            ),
            required_keys=["action", "reason", "confidence"],
            enum_constraints={
                "action": ["approve", "hold", "reject"],
                "confidence": ["low", "medium", "high"],
            },
            notes="Schema-only JSON control.",
        ),
        make_json_object_example(
            example_id="exp211-instruction-json-2",
            prompt=(
                "Return JSON only with keys verdict, owner, and priority. verdict "
                "must be pass or fail. priority must be p0, p1, or p2."
            ),
            required_keys=["verdict", "owner", "priority"],
            enum_constraints={
                "verdict": ["pass", "fail"],
                "priority": ["p0", "p1", "p2"],
            },
            notes="Schema-plus-enum constraint bundle.",
        ),
        make_json_object_example(
            example_id="exp211-instruction-json-3",
            prompt=(
                "Return JSON only with keys status, channel, and escalation. "
                "status must be green, yellow, or red. escalation must be yes or no."
            ),
            required_keys=["status", "channel", "escalation"],
            enum_constraints={
                "status": ["green", "yellow", "red"],
                "escalation": ["yes", "no"],
            },
            notes="Enum-valued incident-routing schema.",
        ),
        make_markdown_sections_example(
            example_id="exp211-instruction-sections-1",
            prompt=(
                "Write an answer with markdown sections Overview, Risks, and Next "
                "Step in that order. Each section must contain exactly one sentence. "
                "Mention api freeze in Overview, staffing gap in Risks, and beta "
                "date in Next Step."
            ),
            sections=["Overview", "Risks", "Next Step"],
            required_tokens=["api freeze", "staffing gap", "beta date"],
            notes="Ordered-section format case.",
        ),
        make_markdown_sections_example(
            example_id="exp211-instruction-sections-2",
            prompt=(
                "Write markdown sections Context, Constraint, and Action in that "
                "order. Each section must contain exactly one sentence. Mention "
                "migration window in Context, no downtime in Constraint, and dry run "
                "in Action."
            ),
            sections=["Context", "Constraint", "Action"],
            required_tokens=["migration window", "no downtime", "dry run"],
            notes="Ordered-section plan case.",
        ),
        make_markdown_sections_example(
            example_id="exp211-instruction-sections-3",
            prompt=(
                "Write markdown sections Signal, Risk, and Mitigation in that "
                "order. Each section must contain exactly one sentence. Mention "
                "error spike in Signal, cache miss in Risk, and rollback switch in "
                "Mitigation."
            ),
            sections=["Signal", "Risk", "Mitigation"],
            required_tokens=["error spike", "cache miss", "rollback switch"],
            notes="Ordered incident summary case.",
        ),
        make_grounded_selection_example(
            example_id="exp211-instruction-grounded-1",
            prompt=(
                "Choose exactly two projects that are both under $50k and have a "
                "delivery date before June.\n"
                "P1 | $45k | May 20\n"
                "P2 | $52k | May 18\n"
                "P3 | $31k | April 30\n"
                "P4 | $48k | June 02\n"
                "Return the chosen IDs as a comma-separated list sorted by cost "
                "ascending."
            ),
            selected_ids=["P3", "P1"],
            ordering="cost_ascending",
            answer_type="comma_separated_list",
            notes="Grounded selection with explicit sort key.",
        ),
        make_grounded_selection_example(
            example_id="exp211-instruction-grounded-2",
            prompt=(
                "Select exactly two books that were published before 1950 and are "
                "longer than 200 pages.\n"
                "B1 | 1948 | 220 pages\n"
                "B2 | 1955 | 180 pages\n"
                "B3 | 1932 | 410 pages\n"
                "B4 | 1949 | 190 pages\n"
                "Return the chosen IDs separated by commas, ordered by page count "
                "descending."
            ),
            selected_ids=["B3", "B1"],
            ordering="page_count_descending",
            answer_type="comma_separated_list",
            notes="Grounded document selection case.",
        ),
        make_grounded_selection_example(
            example_id="exp211-instruction-grounded-3",
            prompt=(
                "Pick exactly two sensors that are active, indoors, and have latency "
                "below 12 ms.\n"
                "S1 | active | indoors | 10 ms\n"
                "S2 | active | outdoors | 7 ms\n"
                "S3 | paused | indoors | 8 ms\n"
                "S4 | active | indoors | 11 ms\n"
                "Return the chosen IDs as a comma-separated list ordered by latency "
                "ascending."
            ),
            selected_ids=["S1", "S4"],
            ordering="latency_ascending",
            answer_type="comma_separated_list",
            notes="Grounded filter over structured prompt data.",
        ),
        make_grounded_selection_example(
            example_id="exp211-instruction-schedule-1",
            prompt=(
                "Choose the earliest meeting slot that starts after 10:00, includes "
                "Priya, and lasts at least 30 minutes.\n"
                "M1 | 09:45 | 30m | Priya, Alex\n"
                "M2 | 10:15 | 25m | Priya, Sam\n"
                "M3 | 10:30 | 45m | Priya, Alex\n"
                "M4 | 11:00 | 30m | Sam, Alex\n"
                "Return only the meeting ID."
            ),
            selected_ids=["M3"],
            ordering="single_best_match",
            answer_type="identifier",
            notes="Temporal grounded-selection case.",
        ),
        make_grounded_selection_example(
            example_id="exp211-instruction-schedule-2",
            prompt=(
                "Pick the earliest support window after 14:00 that has an on-call "
                "engineer and at least 2 free seats.\n"
                "W1 | 13:30 | 3 seats | on-call\n"
                "W2 | 14:10 | 1 seat | on-call\n"
                "W3 | 14:20 | 2 seats | on-call\n"
                "W4 | 14:05 | 4 seats | no on-call\n"
                "Return only the window ID."
            ),
            selected_ids=["W3"],
            ordering="single_best_match",
            answer_type="identifier",
            notes="Grounded earliest-valid-window case.",
        ),
        make_grounded_selection_example(
            example_id="exp211-instruction-schedule-3",
            prompt=(
                "Choose the earliest train that arrives after 08:00, stops at Oak, "
                "and has fewer than 3 delays this week.\n"
                "T1 | 07:55 | Oak | 0 delays\n"
                "T2 | 08:05 | Pine | 1 delay\n"
                "T3 | 08:12 | Oak | 4 delays\n"
                "T4 | 08:20 | Oak | 2 delays\n"
                "Return only the train ID."
            ),
            selected_ids=["T4"],
            ordering="single_best_match",
            answer_type="identifier",
            notes="Grounded route-selection case.",
        ),
        make_grounded_selection_example(
            example_id="exp211-instruction-filter-1",
            prompt=(
                "Pick exactly two menu items that are vegan, nut-free, and keep the "
                "total price at or below $20.\n"
                "A | vegan | nut-free | $7\n"
                "B | vegan | contains nuts | $6\n"
                "C | vegetarian | nut-free | $8\n"
                "D | vegan | nut-free | $9\n"
                "Return the chosen IDs separated by commas ordered by price ascending."
            ),
            selected_ids=["A", "D"],
            ordering="price_ascending",
            answer_type="comma_separated_list",
            notes="Grounded inventory selection under budget.",
        ),
        make_grounded_selection_example(
            example_id="exp211-instruction-filter-2",
            prompt=(
                "Select exactly two parts that fit rack-2, are in stock, and weigh "
                "under 5 kg.\n"
                "P1 | rack-2 | in stock | 4 kg\n"
                "P2 | rack-2 | backorder | 3 kg\n"
                "P3 | rack-1 | in stock | 2 kg\n"
                "P4 | rack-2 | in stock | 3 kg\n"
                "Return the chosen IDs separated by commas ordered by weight ascending."
            ),
            selected_ids=["P4", "P1"],
            ordering="weight_ascending",
            answer_type="comma_separated_list",
            notes="Grounded hardware filter case.",
        ),
        make_grounded_selection_example(
            example_id="exp211-instruction-filter-3",
            prompt=(
                "Choose exactly two speakers who are available Tuesday, remote, and "
                "cost less than $900.\n"
                "S1 | Tuesday | remote | $850\n"
                "S2 | Tuesday | onsite | $600\n"
                "S3 | Wednesday | remote | $700\n"
                "S4 | Tuesday | remote | $500\n"
                "Return the chosen IDs separated by commas ordered by fee ascending."
            ),
            selected_ids=["S4", "S1"],
            ordering="fee_ascending",
            answer_type="comma_separated_list",
            notes="Grounded speaker shortlist case.",
        ),
        make_rewrite_example(
            example_id="exp211-instruction-rewrite-1",
            prompt=(
                "Rewrite this note in a calm professional tone using 25-32 words. "
                "Keep the phrases API freeze and Friday deploy. Omit the phrase bad "
                "surprise.\nOriginal: This was a bad surprise, but the API freeze is "
                "still on and the Friday deploy is unchanged."
            ),
            required_phrases=["API freeze", "Friday deploy"],
            forbidden_phrases=["bad surprise"],
            word_range=[25, 32],
            notes="Constrained rewrite with preserve-and-omit rules.",
        ),
        make_rewrite_example(
            example_id="exp211-instruction-rewrite-2",
            prompt=(
                "Rewrite this message in a calm professional tone using 25-32 words. "
                "Keep the phrases vendor delay and backup supplier. Omit the phrase "
                "total mess.\nOriginal: The vendor delay is a total mess, but we have "
                "a backup supplier ready."
            ),
            required_phrases=["vendor delay", "backup supplier"],
            forbidden_phrases=["total mess"],
            word_range=[25, 32],
            notes="Constrained rewrite with omission rule.",
        ),
        make_rewrite_example(
            example_id="exp211-instruction-rewrite-3",
            prompt=(
                "Rewrite this update in a calm professional tone using 25-32 words. "
                "Keep the phrases staging fix and customer pilot. Omit the phrase "
                "panic mode.\nOriginal: We were in panic mode, but the staging fix is "
                "done and the customer pilot can continue."
            ),
            required_phrases=["staging fix", "customer pilot"],
            forbidden_phrases=["panic mode"],
            word_range=[25, 32],
            notes="Constrained rewrite with length control.",
        ),
        make_decision_json_example(
            example_id="exp211-instruction-decision-1",
            prompt=(
                "Choose the strongest launch option and return JSON with keys choice "
                "and evidence.\n"
                "O1 | reach 7 | cost 4 | risk high\n"
                "O2 | reach 6 | cost 3 | risk medium\n"
                "O3 | reach 5 | cost 2 | risk low\n"
                "Prefer lower risk, then higher reach. evidence must list the chosen "
                "row ID and the risk label."
            ),
            choice="O3",
            evidence_ids=["O3", "risk low"],
            notes="Grounded choice with explicit evidence IDs.",
        ),
        make_decision_json_example(
            example_id="exp211-instruction-decision-2",
            prompt=(
                "Pick the best backup host and return JSON with keys choice and "
                "evidence.\n"
                "H1 | latency 40 | capacity 60 | health yellow\n"
                "H2 | latency 55 | capacity 90 | health green\n"
                "H3 | latency 35 | capacity 50 | health green\n"
                "Prefer green health, then lower latency. evidence must list the "
                "chosen host ID and the health label."
            ),
            choice="H3",
            evidence_ids=["H3", "health green"],
            notes="Grounded backup-host decision case.",
        ),
        make_decision_json_example(
            example_id="exp211-instruction-decision-3",
            prompt=(
                "Choose the best reviewer and return JSON with keys choice and "
                "evidence.\n"
                "R1 | domain fit high | load 5 | timezone UTC+1\n"
                "R2 | domain fit medium | load 1 | timezone UTC-5\n"
                "R3 | domain fit high | load 2 | timezone UTC+0\n"
                "Prefer higher domain fit, then lower load. evidence must list the "
                "chosen reviewer ID and the domain-fit label."
            ),
            choice="R3",
            evidence_ids=["R3", "domain fit high"],
            notes="Grounded reviewer-assignment decision.",
        ),
        make_plan_example(
            example_id="exp211-instruction-plan-1",
            prompt=(
                "Write a numbered 4-step rollout plan. Step 1 must gather baseline "
                "metrics. Step 2 must stage the change. Step 3 must validate the "
                "result. Step 4 must include rollback. Do not mention feature flags."
            ),
            required_step_roles=["baseline_metrics", "stage_change", "validate_result", "rollback"],
            forbidden_tokens=["feature flags"],
            notes="Compositional four-step plan.",
        ),
        make_plan_example(
            example_id="exp211-instruction-plan-2",
            prompt=(
                "Write a numbered 4-step migration plan. Step 1 must inventory "
                "dependencies. Step 2 must create a dry run. Step 3 must execute the "
                "change. Step 4 must define rollback. Do not mention overtime."
            ),
            required_step_roles=["inventory_dependencies", "dry_run", "execute_change", "rollback"],
            forbidden_tokens=["overtime"],
            notes="Migration-plan composition case.",
        ),
        make_plan_example(
            example_id="exp211-instruction-plan-3",
            prompt=(
                "Write a numbered 4-step support plan. Step 1 must gather logs. "
                "Step 2 must isolate scope. Step 3 must apply a fix. Step 4 must "
                "document rollback. Do not mention blame."
            ),
            required_step_roles=["gather_logs", "isolate_scope", "apply_fix", "rollback"],
            forbidden_tokens=["blame"],
            notes="Support-response plan composition case.",
        ),
        make_yaml_extract_example(
            example_id="exp211-instruction-yaml-1",
            prompt=(
                "Convert this record to YAML with keys owner, region, and "
                "days_until_due.\nRecord: owner=Jules, region=west, due=2026-05-04, "
                "today=2026-05-01."
            ),
            required_keys=["owner", "region", "days_until_due"],
            derived_value={"target": "days_until_due", "value": 3},
            notes="Grounded extraction with derived day count.",
        ),
        make_yaml_extract_example(
            example_id="exp211-instruction-yaml-2",
            prompt=(
                "Convert this record to YAML with keys service, team, and "
                "minutes_remaining.\nRecord: service=search, team=infra, "
                "window_end=14:20, now=13:50."
            ),
            required_keys=["service", "team", "minutes_remaining"],
            derived_value={"target": "minutes_remaining", "value": 30},
            notes="Grounded extraction with derived minutes.",
        ),
        make_yaml_extract_example(
            example_id="exp211-instruction-yaml-3",
            prompt=(
                "Convert this record to YAML with keys project, budget, and "
                "budget_left.\nRecord: project=atlas, budget=90, spent=55."
            ),
            required_keys=["project", "budget", "budget_left"],
            derived_value={"target": "budget_left", "value": 35},
            notes="Grounded extraction with derived subtraction.",
        ),
        make_negation_example(
            example_id="exp211-instruction-negation-1",
            prompt=(
                "List the servers that are NOT in maintenance and NOT in the eu-west "
                "region.\n"
                "srv-a | active | us-east\n"
                "srv-b | maintenance | us-east\n"
                "srv-c | active | eu-west\n"
                "srv-d | active | ap-south\n"
                "Return the matching server IDs separated by commas."
            ),
            selected_items=["srv-a", "srv-d"],
            notes="Negation-scope grounding case over server inventory.",
        ),
        make_negation_example(
            example_id="exp211-instruction-negation-2",
            prompt=(
                "List the tasks that are NOT blocked and NOT assigned to team-red.\n"
                "t1 | ready | team-blue\n"
                "t2 | blocked | team-blue\n"
                "t3 | ready | team-red\n"
                "t4 | ready | team-green\n"
                "Return the matching task IDs separated by commas."
            ),
            selected_items=["t1", "t4"],
            notes="Negation-scope grounding case over task roster.",
        ),
        make_negation_example(
            example_id="exp211-instruction-negation-3",
            prompt=(
                "List the vendors that are NOT provisional and NOT missing security "
                "review.\n"
                "v1 | approved | review done\n"
                "v2 | provisional | review done\n"
                "v3 | approved | review missing\n"
                "v4 | approved | review done\n"
                "Return the matching vendor IDs separated by commas."
            ),
            selected_items=["v1", "v4"],
            notes="Negation-scope grounding case over vendor list.",
        ),
        make_two_sentence_example(
            example_id="exp211-instruction-two-sentence-1",
            prompt=(
                "Answer in exactly two sentences. The first sentence must name the "
                "chosen region east-2. The second sentence must mention one "
                "uncertainty using the phrase remaining risk. Do not use first-person "
                "pronouns."
            ),
            required_token="east-2",
            required_uncertainty="remaining risk",
            notes="Answer-plus-uncertainty format case.",
        ),
        make_two_sentence_example(
            example_id="exp211-instruction-two-sentence-2",
            prompt=(
                "Answer in exactly two sentences. The first sentence must name the "
                "selected owner Priya. The second sentence must mention one "
                "uncertainty using the phrase open question. Do not use first-person "
                "pronouns."
            ),
            required_token="Priya",
            required_uncertainty="open question",
            notes="Two-sentence owner-selection case.",
        ),
        make_two_sentence_example(
            example_id="exp211-instruction-two-sentence-3",
            prompt=(
                "Answer in exactly two sentences. The first sentence must name the "
                "release train r7. The second sentence must mention one uncertainty "
                "using the phrase unresolved dependency. Do not use first-person "
                "pronouns."
            ),
            required_token="r7",
            required_uncertainty="unresolved dependency",
            notes="Two-sentence release-summary case.",
        ),
    ]
    return examples


def build_code_examples() -> list[dict[str, Any]]:
    examples = [
        make_python_function_example(
            example_id="exp211-code-dedupe-1",
            prompt=(
                "Write a Python function `dedupe_preserve_order(items: list[str]) -> "
                "list[str]` that returns the first occurrence of each string in input "
                "order. Do not mutate the input list."
            ),
            name="dedupe_preserve_order",
            signature="dedupe_preserve_order(items: list[str]) -> list[str]",
            return_type="list[str]",
            preserves_input=True,
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "output", "equals", "first_occurrence_order_preserved"
                ),
            ],
            notes="Typed-property function contract for stable deduplication.",
        ),
        make_python_function_example(
            example_id="exp211-code-dedupe-2",
            prompt=(
                "Write a Python function `dedupe_casefold(items: list[str]) -> "
                "list[str]` that removes duplicates case-insensitively while "
                "preserving the first original spelling and input order."
            ),
            name="dedupe_casefold",
            signature="dedupe_casefold(items: list[str]) -> list[str]",
            return_type="list[str]",
            semantic_constraints=[
                make_constraint(
                    "x",
                    "semantic_property",
                    "dedupe_rule",
                    "equals",
                    "casefold_compare_keep_first_surface",
                ),
            ],
            notes="Case-insensitive stable dedupe.",
        ),
        make_python_function_example(
            example_id="exp211-code-dedupe-3",
            prompt=(
                "Write a Python function `dedupe_tuples(items: list[tuple[int, int]]) "
                "-> list[tuple[int, int]]` that removes repeated tuples while keeping "
                "the first occurrence of each tuple in order."
            ),
            name="dedupe_tuples",
            signature="dedupe_tuples(items: list[tuple[int, int]]) -> list[tuple[int, int]]",
            return_type="list[tuple[int, int]]",
            semantic_constraints=[
                make_constraint(
                    "x",
                    "semantic_property",
                    "output",
                    "equals",
                    "first_occurrence_tuple_order_preserved",
                ),
            ],
            notes="Tuple-valued stable dedupe.",
        ),
        make_python_function_example(
            example_id="exp211-code-slugify-1",
            prompt=(
                "Write `slugify(text: str) -> str` that lowercases ASCII letters, "
                "replaces whitespace with single dashes, removes punctuation, and "
                "collapses repeated dashes."
            ),
            name="slugify",
            signature="slugify(text: str) -> str",
            return_type="str",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "normalization", "equals", "ascii_lower_dash_slug"
                ),
            ],
            notes="Classic surface-normalization contract.",
            forbidden_apis=["re.sub with Unicode classes"],
        ),
        make_python_function_example(
            example_id="exp211-code-slugify-2",
            prompt=(
                "Write `slugify_filename(name: str) -> str` that lowercases ASCII "
                "letters, keeps digits, converts spaces to dashes, strips leading "
                "and trailing dashes, and drops other punctuation."
            ),
            name="slugify_filename",
            signature="slugify_filename(name: str) -> str",
            return_type="str",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "normalization", "equals", "filename_safe_ascii_slug"
                ),
            ],
            notes="Filename-safe slug contract.",
        ),
        make_python_function_example(
            example_id="exp211-code-slugify-3",
            prompt=(
                "Write `slugify_tag(label: str) -> str` that returns a lowercase "
                "ASCII slug joined with underscores instead of dashes."
            ),
            name="slugify_tag",
            signature="slugify_tag(label: str) -> str",
            return_type="str",
            semantic_constraints=[
                make_constraint("x", "semantic_property", "separator", "equals", "underscore"),
            ],
            notes="Slug variant with underscore separator.",
        ),
        make_python_function_example(
            example_id="exp211-code-intervals-1",
            prompt=(
                "Write `merge_intervals(intervals: list[tuple[int, int]]) -> "
                "list[tuple[int, int]]` that merges overlapping closed intervals and "
                "returns them sorted by start."
            ),
            name="merge_intervals",
            signature="merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]",
            return_type="list[tuple[int, int]]",
            preserves_input=True,
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "merge_rule", "equals", "overlap_closed_intervals"
                ),
            ],
            notes="Interval-merging contract.",
        ),
        make_python_function_example(
            example_id="exp211-code-intervals-2",
            prompt=(
                "Write `merge_touching_intervals(intervals: list[tuple[int, int]]) "
                "-> list[tuple[int, int]]` that merges intervals when they overlap "
                "or touch at a boundary."
            ),
            name="merge_touching_intervals",
            signature=(
                "merge_touching_intervals(intervals: list[tuple[int, int]]) "
                "-> list[tuple[int, int]]"
            ),
            return_type="list[tuple[int, int]]",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "merge_rule", "equals", "overlap_or_touching"
                ),
            ],
            notes="Touching-boundary interval merge.",
        ),
        make_python_function_example(
            example_id="exp211-code-intervals-3",
            prompt=(
                "Write `insert_interval(intervals: list[tuple[int, int]], new_interval: "
                "tuple[int, int]) -> list[tuple[int, int]]` that inserts one closed "
                "interval into a sorted non-overlapping list and merges as needed."
            ),
            name="insert_interval",
            signature=(
                "insert_interval(intervals: list[tuple[int, int]], "
                "new_interval: tuple[int, int]) -> list[tuple[int, int]]"
            ),
            return_type="list[tuple[int, int]]",
            semantic_constraints=[
                make_constraint(
                    "x",
                    "semantic_property",
                    "merge_rule",
                    "equals",
                    "insert_and_merge_single_interval",
                ),
            ],
            notes="Single-insert interval merge.",
        ),
        make_python_function_example(
            example_id="exp211-code-toposort-1",
            prompt=(
                "Write `topo_sort(edges: list[tuple[str, str]]) -> list[str]` that "
                "returns a valid topological order for a directed acyclic graph and "
                "raises ValueError on cycles."
            ),
            name="topo_sort",
            signature="topo_sort(edges: list[tuple[str, str]]) -> list[str]",
            return_type="list[str]",
            raises="ValueError",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "ordering_rule", "equals", "u_before_v_for_each_edge"
                ),
            ],
            notes="Topological-order contract with cycle error.",
        ),
        make_python_function_example(
            example_id="exp211-code-toposort-2",
            prompt=(
                "Write `topo_sort_nodes(nodes: list[str], edges: list[tuple[str, str]]) "
                "-> list[str]` that includes isolated nodes and raises ValueError on "
                "cycles."
            ),
            name="topo_sort_nodes",
            signature=(
                "topo_sort_nodes(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]"
            ),
            return_type="list[str]",
            raises="ValueError",
            semantic_constraints=[
                make_constraint(
                    "x",
                    "semantic_property",
                    "node_coverage",
                    "equals",
                    "all_nodes_including_isolated",
                ),
            ],
            notes="Topo-sort with explicit node universe.",
        ),
        make_python_function_example(
            example_id="exp211-code-toposort-3",
            prompt=(
                "Write `course_order(prereqs: list[tuple[str, str]]) -> list[str]` "
                "that returns a valid course order and raises ValueError if the "
                "prerequisite graph is cyclic."
            ),
            name="course_order",
            signature="course_order(prereqs: list[tuple[str, str]]) -> list[str]",
            return_type="list[str]",
            raises="ValueError",
            semantic_constraints=[
                make_constraint(
                    "x",
                    "semantic_property",
                    "ordering_rule",
                    "equals",
                    "prerequisite_before_course",
                ),
            ],
            notes="Domain-flavored topo-sort prompt.",
        ),
        make_python_function_example(
            example_id="exp211-code-phone-1",
            prompt=(
                "Write `normalize_us_phone(text: str) -> str` that returns a phone "
                "number in the form 555-123-4567 and raises ValueError if the input "
                "does not contain exactly 10 digits."
            ),
            name="normalize_us_phone",
            signature="normalize_us_phone(text: str) -> str",
            return_type="str",
            raises="ValueError",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "output_format", "equals", "ddd-ddd-dddd"
                ),
            ],
            notes="String-format normalization with validation.",
        ),
        make_python_function_example(
            example_id="exp211-code-phone-2",
            prompt=(
                "Write `normalize_ext_phone(text: str) -> tuple[str, str | None]` "
                "that returns the normalized 10-digit phone number plus an optional "
                "extension if present."
            ),
            name="normalize_ext_phone",
            signature="normalize_ext_phone(text: str) -> tuple[str, str | None]",
            return_type="tuple[str, str | None]",
            semantic_constraints=[
                make_constraint(
                    "x",
                    "semantic_property",
                    "extension_rule",
                    "equals",
                    "preserve_numeric_extension_if_present",
                ),
            ],
            notes="Phone normalization with extension extraction.",
        ),
        make_python_function_example(
            example_id="exp211-code-phone-3",
            prompt=(
                "Write `normalize_digits_only(text: str) -> str` that strips all "
                "non-digit characters and returns the last 10 digits, raising "
                "ValueError if fewer than 10 digits are present."
            ),
            name="normalize_digits_only",
            signature="normalize_digits_only(text: str) -> str",
            return_type="str",
            raises="ValueError",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "digit_rule", "equals", "take_last_10_digits"
                ),
            ],
            notes="Phone-like digit normalization variant.",
        ),
        make_python_function_example(
            example_id="exp211-code-csv-1",
            prompt=(
                "Write `parse_user_row(row: str) -> dict[str, str | None]` that "
                "parses a comma-separated row with fields id,name,email, strips "
                "whitespace, and returns None for empty fields."
            ),
            name="parse_user_row",
            signature="parse_user_row(row: str) -> dict[str, str | None]",
            return_type="dict[str, str | None]",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "field_order", "equals", ["id", "name", "email"]
                ),
            ],
            notes="CSV row parser with fixed key order.",
        ),
        make_python_function_example(
            example_id="exp211-code-csv-2",
            prompt=(
                "Write `parse_metric_row(row: str) -> dict[str, int | None]` that "
                "parses a comma-separated row with keys day,errors,latency_ms and "
                "casts numeric fields to integers when present."
            ),
            name="parse_metric_row",
            signature="parse_metric_row(row: str) -> dict[str, int | None]",
            return_type="dict[str, int | None]",
            semantic_constraints=[
                make_constraint(
                    "x",
                    "semantic_property",
                    "numeric_cast_fields",
                    "equals",
                    ["errors", "latency_ms"],
                ),
            ],
            notes="CSV row parser with numeric casting.",
        ),
        make_python_function_example(
            example_id="exp211-code-csv-3",
            prompt=(
                "Write `parse_flag_row(row: str) -> dict[str, bool | str]` that "
                "parses key order name,enabled,owner and converts enabled to a bool "
                "from the strings true or false."
            ),
            name="parse_flag_row",
            signature="parse_flag_row(row: str) -> dict[str, bool | str]",
            return_type="dict[str, bool | str]",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "boolean_cast_field", "equals", "enabled"
                ),
            ],
            notes="CSV row parser with boolean field conversion.",
        ),
        make_python_function_example(
            example_id="exp211-code-rolling-1",
            prompt=(
                "Write `rolling_average(values: list[float], window: int) -> "
                "list[float]` that returns the average of every full sliding window "
                "in O(n) time."
            ),
            name="rolling_average",
            signature="rolling_average(values: list[float], window: int) -> list[float]",
            return_type="list[float]",
            complexity="O(n)",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "window_rule", "equals", "full_windows_only"
                ),
            ],
            notes="Sliding-window numeric contract.",
        ),
        make_python_function_example(
            example_id="exp211-code-rolling-2",
            prompt=(
                "Write `rolling_sum(values: list[int], window: int) -> list[int]` "
                "that returns every full sliding-window sum in O(n) time."
            ),
            name="rolling_sum",
            signature="rolling_sum(values: list[int], window: int) -> list[int]",
            return_type="list[int]",
            complexity="O(n)",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "window_rule", "equals", "full_windows_only"
                ),
            ],
            notes="Sliding-window sum contract.",
        ),
        make_python_function_example(
            example_id="exp211-code-rolling-3",
            prompt=(
                "Write `rolling_max(values: list[int], window: int) -> list[int]` "
                "that returns the maximum of every full window in O(n) time."
            ),
            name="rolling_max",
            signature="rolling_max(values: list[int], window: int) -> list[int]",
            return_type="list[int]",
            complexity="O(n)",
            semantic_constraints=[
                make_constraint("x", "semantic_property", "aggregate_rule", "equals", "window_max"),
            ],
            notes="Sliding-window max contract.",
        ),
        make_python_function_example(
            example_id="exp211-code-config-1",
            prompt=(
                "Write `load_config(env: dict[str, str]) -> dict[str, int | str]` "
                "that requires HOST and PORT, casts PORT to int, and defaults MODE "
                "to safe when missing."
            ),
            name="load_config",
            signature="load_config(env: dict[str, str]) -> dict[str, int | str]",
            return_type="dict[str, int | str]",
            raises="ValueError",
            semantic_constraints=[
                make_constraint("x", "required_keys", "env_keys", "equals", ["HOST", "PORT"]),
                make_constraint("y", "default_value", "MODE", "equals", "safe"),
            ],
            notes="Env-config loader with defaults and type coercion.",
        ),
        make_python_function_example(
            example_id="exp211-code-config-2",
            prompt=(
                "Write `load_retry_config(env: dict[str, str]) -> dict[str, int]` "
                "that requires RETRIES, casts it to int, and defaults TIMEOUT to 30 "
                "when missing."
            ),
            name="load_retry_config",
            signature="load_retry_config(env: dict[str, str]) -> dict[str, int]",
            return_type="dict[str, int]",
            raises="ValueError",
            semantic_constraints=[
                make_constraint("x", "required_keys", "env_keys", "equals", ["RETRIES"]),
                make_constraint("y", "default_value", "TIMEOUT", "equals", 30),
            ],
            notes="Integer-heavy env-config loader.",
        ),
        make_python_function_example(
            example_id="exp211-code-config-3",
            prompt=(
                "Write `load_feature_config(env: dict[str, str]) -> dict[str, bool | str]` "
                "that requires OWNER, defaults CHANNEL to general, and parses ENABLED "
                "to bool from true or false."
            ),
            name="load_feature_config",
            signature="load_feature_config(env: dict[str, str]) -> dict[str, bool | str]",
            return_type="dict[str, bool | str]",
            raises="ValueError",
            semantic_constraints=[
                make_constraint("x", "required_keys", "env_keys", "equals", ["OWNER"]),
                make_constraint("y", "default_value", "CHANNEL", "equals", "general"),
                make_constraint("z", "boolean_cast_field", "ENABLED", "equals", True),
            ],
            notes="Feature-toggle config loader.",
        ),
        make_python_function_example(
            example_id="exp211-code-groupby-1",
            prompt=(
                "Write `group_by_team(rows: list[dict[str, str]]) -> dict[str, "
                "list[dict[str, str]]]` that groups rows by the team key while "
                "preserving input order within each group."
            ),
            name="group_by_team",
            signature=(
                "group_by_team(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]"
            ),
            return_type="dict[str, list[dict[str, str]]]",
            semantic_constraints=[
                make_constraint("x", "semantic_property", "group_key", "equals", "team"),
            ],
            notes="Group-by preserving intra-group order.",
        ),
        make_python_function_example(
            example_id="exp211-code-groupby-2",
            prompt=(
                "Write `group_by_priority(rows: list[dict[str, str]]) -> dict[str, "
                "list[str]]` that groups task IDs by priority and preserves the task "
                "ID order inside each priority bucket."
            ),
            name="group_by_priority",
            signature="group_by_priority(rows: list[dict[str, str]]) -> dict[str, list[str]]",
            return_type="dict[str, list[str]]",
            semantic_constraints=[
                make_constraint("x", "semantic_property", "group_key", "equals", "priority"),
            ],
            notes="Group-by projection over task IDs.",
        ),
        make_python_function_example(
            example_id="exp211-code-groupby-3",
            prompt=(
                "Write `group_words_by_initial(words: list[str]) -> dict[str, "
                "list[str]]` that groups words by first lowercase letter and "
                "preserves the original order within each list."
            ),
            name="group_words_by_initial",
            signature="group_words_by_initial(words: list[str]) -> dict[str, list[str]]",
            return_type="dict[str, list[str]]",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "group_key", "equals", "first_lowercase_letter"
                ),
            ],
            notes="Character-based grouping contract.",
        ),
        make_python_function_example(
            example_id="exp211-code-roman-1",
            prompt=(
                "Write `to_roman(value: int) -> str` that converts integers from 1 to "
                "3999 to Roman numerals and raises ValueError outside that range."
            ),
            name="to_roman",
            signature="to_roman(value: int) -> str",
            return_type="str",
            raises="ValueError",
            semantic_constraints=[
                make_constraint("x", "value_range", "accepted_input", "equals", [1, 3999]),
            ],
            notes="Classical Roman-numeral encoder.",
        ),
        make_python_function_example(
            example_id="exp211-code-roman-2",
            prompt=(
                "Write `from_roman(text: str) -> int` that converts uppercase Roman "
                "numerals to integers and raises ValueError on invalid numerals."
            ),
            name="from_roman",
            signature="from_roman(text: str) -> int",
            return_type="int",
            raises="ValueError",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "accepted_format", "equals", "uppercase_roman"
                ),
            ],
            notes="Roman-numeral decoder.",
        ),
        make_python_function_example(
            example_id="exp211-code-roman-3",
            prompt=(
                "Write `is_valid_roman(text: str) -> bool` that returns True only for "
                "uppercase Roman numerals in canonical subtractive notation."
            ),
            name="is_valid_roman",
            signature="is_valid_roman(text: str) -> bool",
            return_type="bool",
            semantic_constraints=[
                make_constraint(
                    "x",
                    "semantic_property",
                    "validation_rule",
                    "equals",
                    "canonical_subtractive_notation",
                ),
            ],
            notes="Roman-numeral validator.",
        ),
        make_python_function_example(
            example_id="exp211-code-chunks-1",
            prompt=(
                "Write `chunk_list(items: list[int], size: int) -> list[list[int]]` "
                "that splits a list into consecutive chunks of at most size."
            ),
            name="chunk_list",
            signature="chunk_list(items: list[int], size: int) -> list[list[int]]",
            return_type="list[list[int]]",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "chunk_rule", "equals", "consecutive_max_size_chunks"
                ),
            ],
            notes="Chunking contract.",
        ),
        make_python_function_example(
            example_id="exp211-code-chunks-2",
            prompt=(
                "Write `chunk_text(text: str, size: int) -> list[str]` that splits a "
                "string into consecutive substrings of at most size characters."
            ),
            name="chunk_text",
            signature="chunk_text(text: str, size: int) -> list[str]",
            return_type="list[str]",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "chunk_rule", "equals", "consecutive_substrings"
                ),
            ],
            notes="String chunking contract.",
        ),
        make_python_function_example(
            example_id="exp211-code-chunks-3",
            prompt=(
                "Write `chunk_pairs(items: list[tuple[int, int]], size: int) -> "
                "list[list[tuple[int, int]]]` that chunks tuple rows into consecutive "
                "groups of at most size."
            ),
            name="chunk_pairs",
            signature=(
                "chunk_pairs(items: list[tuple[int, int]], size: int) -> "
                "list[list[tuple[int, int]]]"
            ),
            return_type="list[list[tuple[int, int]]]",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "chunk_rule", "equals", "consecutive_tuple_groups"
                ),
            ],
            notes="Tuple-list chunking contract.",
        ),
        make_python_function_example(
            example_id="exp211-code-score-1",
            prompt=(
                "Write `score_keywords(text: str, weights: dict[str, int]) -> int` "
                "that returns the sum of weights for each keyword appearing at least "
                "once in the text."
            ),
            name="score_keywords",
            signature="score_keywords(text: str, weights: dict[str, int]) -> int",
            return_type="int",
            semantic_constraints=[
                make_constraint(
                    "x",
                    "semantic_property",
                    "scoring_rule",
                    "equals",
                    "sum_weight_per_present_keyword",
                ),
            ],
            notes="Keyword scoring without repeated-count inflation.",
        ),
        make_python_function_example(
            example_id="exp211-code-score-2",
            prompt=(
                "Write `score_casefold_keywords(text: str, weights: dict[str, int]) -> "
                "int` that performs the same keyword scoring case-insensitively."
            ),
            name="score_casefold_keywords",
            signature="score_casefold_keywords(text: str, weights: dict[str, int]) -> int",
            return_type="int",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "scoring_rule", "equals", "casefold_keyword_presence"
                ),
            ],
            notes="Case-insensitive keyword scoring.",
        ),
        make_python_function_example(
            example_id="exp211-code-score-3",
            prompt=(
                "Write `score_tag_overlap(tags: list[str], weights: dict[str, int]) "
                "-> int` that returns the sum of weights for each distinct tag that "
                "appears in the list."
            ),
            name="score_tag_overlap",
            signature="score_tag_overlap(tags: list[str], weights: dict[str, int]) -> int",
            return_type="int",
            semantic_constraints=[
                make_constraint(
                    "x", "semantic_property", "scoring_rule", "equals", "distinct_tag_weight_sum"
                ),
            ],
            notes="Distinct-tag overlap scoring.",
        ),
    ]
    return examples


def build_benchmark() -> list[dict[str, Any]]:
    examples = []
    examples.extend(build_live_gsm8k_examples())
    examples.extend(build_instruction_examples())
    examples.extend(build_code_examples())
    return examples


def build_results(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_source_family = Counter(example["source_family"] for example in examples)
    by_verifier_path = Counter(example["expected_verifier_path"] for example in examples)
    by_answer_schema_type = Counter(
        example["expected_answer_schema"]["type"] for example in examples
    )
    by_constraint_type: Counter[str] = Counter()
    for example in examples:
        for constraint_type in set(example["constraint_types"]):
            by_constraint_type[constraint_type] += 1
    by_monitorability = {
        "true": sum(1 for example in examples if example["free_form_reasoning_monitorable"]),
        "false": sum(1 for example in examples if not example["free_form_reasoning_monitorable"]),
    }

    coverage_checks = {
        "example_count_in_range": 80 <= len(examples) <= 120,
        "has_live_gsm8k_examples": by_source_family["live_gsm8k_semantic_failure"] > 0,
        "has_instruction_examples": by_source_family["instruction_following"] > 0,
        "has_code_examples": by_source_family["code_typed_properties"] > 0,
        "has_literal_constraints": by_constraint_type["literal"] > 0,
        "has_compositional_constraints": by_constraint_type["compositional"] > 0,
        "has_semantic_grounding_constraints": by_constraint_type["semantic_grounding"] > 0,
        "has_typed_property_constraints": by_constraint_type["typed_property"] > 0,
        "has_monitorable_examples": by_monitorability["true"] > 0,
    }

    return {
        "experiment": EXPERIMENT_LABEL,
        "title": "Constraint IR benchmark for semantic grounding and typed prompt constraints",
        "run_date": RUN_DATE,
        "benchmark_path": "data/research/constraint_ir_benchmark_211.jsonl",
        "summary": {
            "n_examples": len(examples),
            "by_source_family": dict(by_source_family),
            "by_constraint_type": dict(by_constraint_type),
            "by_verifier_path": dict(by_verifier_path),
            "by_answer_schema_type": dict(by_answer_schema_type),
            "by_monitorability": by_monitorability,
            "coverage_checks": coverage_checks,
        },
        "benchmark_preview": [example["example_id"] for example in examples[:10]],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    examples = build_benchmark()
    results = build_results(examples)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(BENCHMARK_PATH, examples)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
