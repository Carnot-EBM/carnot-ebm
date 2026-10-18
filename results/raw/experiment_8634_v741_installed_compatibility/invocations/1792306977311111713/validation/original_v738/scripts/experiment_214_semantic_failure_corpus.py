#!/usr/bin/env python3
"""Experiment 214: semantic failure corpus for verifier training.

This workflow writes:

- ``data/research/semantic_failure_corpus_214.jsonl``
- ``results/experiment_214_results.json``

The corpus is deterministic and unit-test friendly. It combines:

- curated live semantic failures from Exp 203 / 206 / 207
- targeted follow-up prompts that isolate the semantic failure categories
- Exp 208-informed code follow-ups that keep code-oracle misses explicit

Spec: REQ-VERIFY-018, REQ-VERIFY-019, SCENARIO-VERIFY-018,
SCENARIO-VERIFY-019
"""

from __future__ import annotations

import json
import os
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

RUN_DATE = "20260412"
EXPERIMENT_LABEL = "Exp 214"

TAXONOMY_LABELS = [
    "question_grounding_failures",
    "omitted_premises",
    "entity_quantity_binding_errors",
    "unit_aggregation_errors",
    "genuine_arithmetic_slips",
    "code_specific_oracle_property_misses",
]


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


REPO_ROOT = get_repo_root()
CORPUS_PATH = REPO_ROOT / "data" / "research" / "semantic_failure_corpus_214.jsonl"
RESULTS_PATH = REPO_ROOT / "results" / "experiment_214_results.json"


def clean_block(text: str) -> str:
    return textwrap.dedent(text).strip()


def make_record(
    *,
    example_id: str,
    source_type: str,
    source_artifact: str,
    source_refs: list[str],
    domain: str,
    prompt: str,
    response: str,
    taxonomy_label: str,
    failure_mechanism: str,
    expected_outcome: str,
    observed_outcome: str,
    verifier_path: str,
    verifier_signal: str,
    structured_reasoning_helpful: bool,
    structured_reasoning_why: str,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "source_type": source_type,
        "source_artifact": source_artifact,
        "source_refs": source_refs,
        "domain": domain,
        "prompt": clean_block(prompt),
        "response": clean_block(response),
        "gold_diagnosis": {
            "taxonomy_label": taxonomy_label,
            "failure_mechanism": failure_mechanism,
            "expected_outcome": expected_outcome,
            "observed_outcome": observed_outcome,
        },
        "expected_verifier_signal": {
            "verifier_path": verifier_path,
            "signal_summary": verifier_signal,
            "should_flag": True,
        },
        "structured_reasoning_helpful": structured_reasoning_helpful,
        "structured_reasoning_why": structured_reasoning_why,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


LIVE_TRACE_SPECS: list[dict[str, Any]] = [
    {
        "dataset_idx": 923,
        "example_id": "exp214-live-923",
        "source_artifact": "exp203_live",
        "source_refs": ["exp203:923", "exp206:923", "exp207:923"],
        "taxonomy_label": "omitted_premises",
        "failure_mechanism": (
            "The response derives the cups-per-row quantity but never applies "
            "the final premise that each row is split equally between "
            "chamomile and mint."
        ),
        "expected_outcome": "2 cups of mint tea per row",
        "observed_outcome": "4 cups after stopping at the row-level quantity",
        "verifier_path": "question_grounding.quantity_graph",
        "verifier_signal": (
            "The quantity graph should require the final row-level split into "
            "mint and chamomile before binding the answer."
        ),
        "structured_reasoning_helpful": True,
        "structured_reasoning_why": (
            "A scaffold with explicit fields for remaining cups, cups per row, "
            "and mint-per-row would expose the missing final step."
        ),
    },
    {
        "dataset_idx": 814,
        "example_id": "exp214-live-814",
        "source_artifact": "exp203_live",
        "source_refs": ["exp203:814", "exp206:814", "exp207:814"],
        "taxonomy_label": "unit_aggregation_errors",
        "failure_mechanism": (
            "The response converts quarter duration into a continuous "
            "per-minute rate and binds 'half as long' to hours played instead "
            "of minutes per quarter."
        ),
        "expected_outcome": "$11 total",
        "observed_outcome": "$10 after a unit-mismatched continuous-rate model",
        "verifier_path": "question_grounding.unit_reasoner",
        "verifier_signal": (
            "The verifier should preserve the token-duration unit and derive "
            "discrete quarter counts rather than a free-floating minute rate."
        ),
        "structured_reasoning_helpful": True,
        "structured_reasoning_why": (
            "Structured fields for minutes-per-quarter and quarters-needed per "
            "player would make the unit mismatch obvious."
        ),
    },
    {
        "dataset_idx": 943,
        "example_id": "exp214-live-943",
        "source_artifact": "exp203_live",
        "source_refs": ["exp203:943", "exp206:943", "exp207:943"],
        "taxonomy_label": "entity_quantity_binding_errors",
        "failure_mechanism": (
            "The response binds the $40 resale amount to each CD instead of "
            "to the whole resale event."
        ),
        "expected_outcome": "$50 net loss",
        "observed_outcome": (
            "Treats the $40 as per-CD revenue, implying a $200 resale and a negative loss."
        ),
        "verifier_path": "question_grounding.scope_alignment",
        "verifier_signal": (
            "A scope-alignment check should attach the sale amount to the full "
            "event, not to each item."
        ),
        "structured_reasoning_helpful": True,
        "structured_reasoning_why": (
            "Explicit structured fields for purchase total, resale total, and "
            "net loss would surface the scope error before arithmetic checks."
        ),
    },
    {
        "dataset_idx": 506,
        "example_id": "exp214-live-506",
        "source_artifact": "exp206_live",
        "source_refs": ["exp206:506", "exp207:506"],
        "taxonomy_label": "entity_quantity_binding_errors",
        "failure_mechanism": (
            "The response binds one pair of snowshoes to each dog rather than to each pair of legs."
        ),
        "expected_outcome": "$144 total cost",
        "observed_outcome": "$72 after halving the required number of pairs",
        "verifier_path": "question_grounding.count_expansion",
        "verifier_signal": (
            "The verifier should expand four legs into two snowshoe pairs per dog before pricing."
        ),
        "structured_reasoning_helpful": False,
        "structured_reasoning_why": (
            "The key signal is prompt-grounded count expansion; more prose does "
            "not help if the entity binding is wrong."
        ),
    },
    {
        "dataset_idx": 1019,
        "example_id": "exp214-live-1019",
        "source_artifact": "exp206_live",
        "source_refs": ["exp206:1019", "exp207:1019"],
        "taxonomy_label": "question_grounding_failures",
        "failure_mechanism": (
            "The response treats late checkouts as still present in the final "
            "state instead of removing them by the end-of-day answer target."
        ),
        "expected_outcome": "98 guests",
        "observed_outcome": "113 guests after mis-grounding the hotel end state",
        "verifier_path": "question_grounding.event_timeline",
        "verifier_signal": (
            "An event-timeline state tracker should remove both early and late "
            "checkouts before applying later check-ins."
        ),
        "structured_reasoning_helpful": True,
        "structured_reasoning_why": (
            "A structured timeline with explicit guest counts after each event "
            "would reveal the incorrect end-state semantics."
        ),
    },
    {
        "dataset_idx": 515,
        "example_id": "exp214-live-515",
        "source_artifact": "exp206_live",
        "source_refs": ["exp206:515", "exp207:515"],
        "taxonomy_label": "entity_quantity_binding_errors",
        "failure_mechanism": (
            "The response binds '$4 each' to packs instead of to individual canvas bags."
        ),
        "expected_outcome": "$160 profit",
        "observed_outcome": "$288 profit after undercounting the bag cost",
        "verifier_path": "question_grounding.scope_alignment",
        "verifier_signal": (
            "A modifier-scope check should attach the purchase price to each "
            "bag while keeping the pack count as a separate multiplicity."
        ),
        "structured_reasoning_helpful": True,
        "structured_reasoning_why": (
            "Separating units into packs, bags-per-pack, cost-per-bag, and "
            "sale-price-per-bag would expose the scope mistake."
        ),
    },
    {
        "dataset_idx": 1077,
        "example_id": "exp214-live-1077",
        "source_artifact": "exp206_live",
        "source_refs": ["exp206:1077", "exp207:1077"],
        "taxonomy_label": "omitted_premises",
        "failure_mechanism": (
            "The response drops the initial 10-minute wait before the first "
            "train when accumulating the total wait time."
        ),
        "expected_outcome": "145 minutes",
        "observed_outcome": "135 minutes after omitting one interval",
        "verifier_path": "question_grounding.temporal_schedule",
        "verifier_signal": (
            "A temporal schedule graph should require every arrival and dwell "
            "interval in the prompt before binding the final wait time."
        ),
        "structured_reasoning_helpful": True,
        "structured_reasoning_why": (
            "A structured schedule with one field per train arrival and "
            "departure would make the missing 10-minute interval obvious."
        ),
    },
    {
        "dataset_idx": 796,
        "example_id": "exp214-live-796",
        "source_artifact": "exp206_live",
        "source_refs": ["exp206:796", "exp207:796"],
        "taxonomy_label": "unit_aggregation_errors",
        "failure_mechanism": (
            "The response uses only the final workforce state instead of "
            "aggregating payroll across all three monthly employee counts."
        ),
        "expected_outcome": "$2,880,000 total payroll",
        "observed_outcome": "$2,640,000 after collapsing the monthly accumulator",
        "verifier_path": "question_grounding.repeated_event_accumulator",
        "verifier_signal": (
            "A repeated-event accumulator should sum payroll over month 1, "
            "month 2, and month 3 rather than pricing only the final state."
        ),
        "structured_reasoning_helpful": True,
        "structured_reasoning_why": (
            "A table of monthly headcounts and monthly payroll values would "
            "make the aggregation requirement visible."
        ),
    },
]


def build_live_trace_examples() -> list[dict[str, Any]]:
    exp203 = load_json(REPO_ROOT / "results" / "experiment_203_results.json")
    exp206 = load_json(REPO_ROOT / "results" / "experiment_206_results.json")

    exp203_wrong = {
        int(case["dataset_idx"]): case for case in exp203["all_cases"] if not bool(case["correct"])
    }
    exp206_wrong = {
        int(row["dataset_idx"]): row
        for row in exp206["results"]
        if not bool(row["baseline"]["correct"])
    }

    examples: list[dict[str, Any]] = []
    for spec in LIVE_TRACE_SPECS:
        dataset_idx = int(spec["dataset_idx"])
        if dataset_idx in exp203_wrong:
            row = exp203_wrong[dataset_idx]
            prompt = str(row["question"])
            response = str(row["response"])
            observed_answer = row["extracted_answer"]
        else:
            row = exp206_wrong[dataset_idx]
            prompt = str(row["question"])
            response = str(row["baseline"]["response"])
            observed_answer = row["baseline"]["extracted_answer"]

        observed_outcome = str(spec["observed_outcome"])
        if observed_answer is not None:
            observed_outcome = f"{observed_outcome} (extracted answer: {observed_answer})"

        examples.append(
            make_record(
                example_id=str(spec["example_id"]),
                source_type="live_trace",
                source_artifact=str(spec["source_artifact"]),
                source_refs=list(spec["source_refs"]),
                domain="word_problem",
                prompt=prompt,
                response=response,
                taxonomy_label=str(spec["taxonomy_label"]),
                failure_mechanism=str(spec["failure_mechanism"]),
                expected_outcome=str(spec["expected_outcome"]),
                observed_outcome=observed_outcome,
                verifier_path=str(spec["verifier_path"]),
                verifier_signal=str(spec["verifier_signal"]),
                structured_reasoning_helpful=bool(spec["structured_reasoning_helpful"]),
                structured_reasoning_why=str(spec["structured_reasoning_why"]),
            )
        )
    return examples


def word_followup(
    *,
    example_id: str,
    taxonomy_label: str,
    prompt: str,
    response: str,
    expected_outcome: str,
    observed_outcome: str,
    failure_mechanism: str,
    verifier_path: str,
    verifier_signal: str,
    source_refs: list[str],
    structured_reasoning_helpful: bool,
    structured_reasoning_why: str,
) -> dict[str, Any]:
    return make_record(
        example_id=example_id,
        source_type="targeted_follow_up",
        source_artifact="exp214_followup",
        source_refs=source_refs,
        domain="word_problem",
        prompt=prompt,
        response=response,
        taxonomy_label=taxonomy_label,
        failure_mechanism=failure_mechanism,
        expected_outcome=expected_outcome,
        observed_outcome=observed_outcome,
        verifier_path=verifier_path,
        verifier_signal=verifier_signal,
        structured_reasoning_helpful=structured_reasoning_helpful,
        structured_reasoning_why=structured_reasoning_why,
    )


def code_followup(
    *,
    example_id: str,
    prompt: str,
    response: str,
    expected_outcome: str,
    observed_outcome: str,
    failure_mechanism: str,
    verifier_signal: str,
    source_refs: list[str],
) -> dict[str, Any]:
    return make_record(
        example_id=example_id,
        source_type="targeted_follow_up",
        source_artifact="exp208_followup",
        source_refs=source_refs,
        domain="code",
        prompt=prompt,
        response=response,
        taxonomy_label="code_specific_oracle_property_misses",
        failure_mechanism=failure_mechanism,
        expected_outcome=expected_outcome,
        observed_outcome=observed_outcome,
        verifier_path="code_ir.typed_contracts_plus_execution",
        verifier_signal=verifier_signal,
        structured_reasoning_helpful=False,
        structured_reasoning_why=(
            "The decisive signal should come from typed properties and probe "
            "execution, not from free-form reasoning about the code."
        ),
    )


def arithmetic_followup(
    *,
    example_id: str,
    prompt: str,
    response: str,
    expected_outcome: str,
    observed_outcome: str,
    failure_mechanism: str,
) -> dict[str, Any]:
    return word_followup(
        example_id=example_id,
        taxonomy_label="genuine_arithmetic_slips",
        prompt=prompt,
        response=response,
        expected_outcome=expected_outcome,
        observed_outcome=observed_outcome,
        failure_mechanism=failure_mechanism,
        verifier_path="arithmetic_chain.exact_check",
        verifier_signal=(
            "A deterministic arithmetic checker should flag the incorrect "
            "numeric step even though the prompt grounding is otherwise sound."
        ),
        source_refs=["exp214:arithmetic_control"],
        structured_reasoning_helpful=False,
        structured_reasoning_why=(
            "More reasoning text is unnecessary when a direct arithmetic check can catch the slip."
        ),
    )


def build_question_grounding_followups() -> list[dict[str, Any]]:
    return [
        word_followup(
            example_id="exp214-followup-question-grounding-1",
            taxonomy_label="question_grounding_failures",
            prompt="""
            A hotel starts the day with 80 guests. 12 guests check out before lunch
            and 10 choose a late checkout. In the afternoon 18 guests check in, and
            4 more check in after dinner. How many guests are in the hotel now?
            """,
            response="""
            Start with 80 guests.
            After 12 early checkouts, 68 remain.
            The 10 late checkouts are still there for now, so keep 68.
            Add 18 afternoon check-ins and 4 dinner check-ins: 68 + 18 + 4 = 90.
            Answer: 90
            """,
            expected_outcome="80 guests",
            observed_outcome="90 guests",
            failure_mechanism=(
                "The answer targets the end-of-day guest count but the response "
                "never removes the late-checkout group from the final state."
            ),
            verifier_path="question_grounding.event_timeline",
            verifier_signal=(
                "The verifier should track the hotel's state at the answer time, "
                "not only during the midday intermediate state."
            ),
            source_refs=["exp206:1019", "exp207:1019"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured timeline with explicit state after each event would "
                "show that late checkout is still a checkout by the end."
            ),
        ),
        word_followup(
            example_id="exp214-followup-question-grounding-2",
            taxonomy_label="question_grounding_failures",
            prompt="""
            Nora spends $24 on wax and sells 8 candles for $5 each. How much
            profit does she make?
            """,
            response="""
            Revenue is 8 x 5 = 40.
            Answer: 40
            """,
            expected_outcome="$16 profit",
            observed_outcome="$40 revenue",
            failure_mechanism=(
                "The response answers a different question from the one asked: "
                "it returns revenue instead of profit."
            ),
            verifier_path="question_grounding.answer_target_alignment",
            verifier_signal=(
                "The verifier should align the answer target with the prompt's "
                "requested quantity, which is profit rather than gross revenue."
            ),
            source_refs=["exp203:943", "exp206:943", "exp207:943"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A scaffold with separate revenue, cost, and profit fields would "
                "force the answer to bind to the requested target."
            ),
        ),
        word_followup(
            example_id="exp214-followup-question-grounding-3",
            taxonomy_label="question_grounding_failures",
            prompt="""
            The Tigers score 27 points and the Hawks score 19. By how many
            points did the Tigers win?
            """,
            response="""
            The Tigers scored 27 points.
            Answer: 27
            """,
            expected_outcome="8 points",
            observed_outcome="27 points",
            failure_mechanism=(
                "The response reports one team's score instead of the winning "
                "margin asked for in the prompt."
            ),
            verifier_path="question_grounding.compare_target_alignment",
            verifier_signal=(
                "A comparison-target check should require the difference between "
                "the two grounded values, not either raw score."
            ),
            source_refs=["exp214:compare-target"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=(
                "The missing signal is the answer target itself; extra prose is "
                "less useful than directly checking the requested relation."
            ),
        ),
        word_followup(
            example_id="exp214-followup-question-grounding-4",
            taxonomy_label="question_grounding_failures",
            prompt="""
            Route A is 18 miles. Route B is 5 miles shorter, but then a 2-mile
            detour is added to Route B. How many miles longer is Route A than
            Route B now?
            """,
            response="""
            Route B is 18 - 5 + 2 = 15 miles long.
            Answer: 15
            """,
            expected_outcome="3 miles",
            observed_outcome="15 miles",
            failure_mechanism=(
                "The response computes Route B correctly but binds the answer to "
                "Route B's length instead of to the requested difference."
            ),
            verifier_path="question_grounding.compare_target_alignment",
            verifier_signal=(
                "The verifier should require the final answer to be the margin "
                "between the grounded route lengths."
            ),
            source_refs=["exp214:compare-target"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "Explicit fields for route_a, route_b, and difference would make "
                "the answer target mismatch visible."
            ),
        ),
        word_followup(
            example_id="exp214-followup-question-grounding-5",
            taxonomy_label="question_grounding_failures",
            prompt="""
            Eli has $50. He buys a $17 book and a $9 notebook. How much money
            does he have left?
            """,
            response="""
            He spends 17 + 9 = 26.
            Answer: 26
            """,
            expected_outcome="$24 left",
            observed_outcome="$26 spent",
            failure_mechanism=(
                "The response returns the amount spent, not the amount remaining "
                "requested by the question."
            ),
            verifier_path="question_grounding.answer_target_alignment",
            verifier_signal=(
                "A ledger-style check should distinguish spent money from the "
                "remaining balance before binding the final answer."
            ),
            source_refs=["exp214:ledger-target"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured ledger with starting balance, expenses, and final "
                "balance would expose the wrong target quantity."
            ),
        ),
        word_followup(
            example_id="exp214-followup-question-grounding-6",
            taxonomy_label="question_grounding_failures",
            prompt="""
            There are 3 crates with 12 oranges each. 5 oranges are bruised and
            thrown out. How many good oranges are left?
            """,
            response="""
            There are 3 x 12 = 36 oranges in total.
            Answer: 36
            """,
            expected_outcome="31 oranges",
            observed_outcome="36 oranges",
            failure_mechanism=(
                "The response answers with the pre-loss total rather than the "
                "grounded post-loss quantity requested."
            ),
            verifier_path="question_grounding.answer_target_alignment",
            verifier_signal=(
                "The verifier should bind the answer to the surviving orange "
                "count after the bruised fruit is removed."
            ),
            source_refs=["exp214:remaining-total"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=(
                "A direct answer-target check is more valuable than extra prose "
                "when the missing step is simply not applied."
            ),
        ),
        word_followup(
            example_id="exp214-followup-question-grounding-7",
            taxonomy_label="question_grounding_failures",
            prompt="""
            A library shelf has 14 novels. Another shelf has 9 novels and then
            receives 3 more. How many fewer novels does the second shelf have
            now?
            """,
            response="""
            The second shelf has 9 + 3 = 12 novels.
            Answer: 12
            """,
            expected_outcome="2 fewer novels",
            observed_outcome="12 novels",
            failure_mechanism=(
                "The response reports the second shelf's count instead of the "
                "difference requested by the question."
            ),
            verifier_path="question_grounding.compare_target_alignment",
            verifier_signal=(
                "A comparison-target check should require the final answer to be "
                "the remaining gap between the shelves."
            ),
            source_refs=["exp214:compare-target"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured comparison table would force the model to include "
                "the comparison target rather than a raw count."
            ),
        ),
        word_followup(
            example_id="exp214-followup-question-grounding-8",
            taxonomy_label="question_grounding_failures",
            prompt="""
            A hotel starts with 70 guests. 8 guests leave in the morning,
            11 guests check in during the afternoon, and 6 guests check in after
            dinner. How many guests checked in during the afternoon?
            """,
            response="""
            Start with 70.
            After 8 leave, 62 remain.
            Then 11 plus 6 check in, so the hotel ends with 79 guests.
            Answer: 79
            """,
            expected_outcome="11 guests",
            observed_outcome="79 guests",
            failure_mechanism=(
                "The response answers with the final occupancy instead of the "
                "specific afternoon check-in quantity asked for."
            ),
            verifier_path="question_grounding.answer_target_alignment",
            verifier_signal=(
                "The verifier should tie the answer to the named event span "
                "('during the afternoon') rather than to the overall end state."
            ),
            source_refs=["exp206:1019", "exp207:1019"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured event list with event-specific quantities would "
                "make the target span explicit."
            ),
        ),
        word_followup(
            example_id="exp214-followup-question-grounding-9",
            taxonomy_label="question_grounding_failures",
            prompt="""
            A farmer harvests 24 red apples and 18 green apples. He sells
            10 red apples. How many green apples does he still have?
            """,
            response="""
            He has 24 - 10 = 14 apples left after the sale.
            Answer: 14
            """,
            expected_outcome="18 green apples",
            observed_outcome="14 apples from the red-apple ledger",
            failure_mechanism=(
                "The response grounds the subtraction against the sold red apples "
                "instead of against the green-apple quantity the question asks for."
            ),
            verifier_path="question_grounding.entity_target_alignment",
            verifier_signal=(
                "An entity-target check should require the answer to bind to the "
                "green-apple count, which never changed."
            ),
            source_refs=["exp214:entity-target"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=(
                "The missing signal is entity alignment rather than hidden reasoning detail."
            ),
        ),
    ]


def build_omitted_premise_followups() -> list[dict[str, Any]]:
    return [
        word_followup(
            example_id="exp214-followup-omitted-premise-1",
            taxonomy_label="omitted_premises",
            prompt="""
            A baker has 48 muffins. She sets 12 aside for display. The rest are
            split equally into 3 baskets, and half of each basket is blueberry.
            How many blueberry muffins are in each basket?
            """,
            response="""
            After setting aside 12 muffins, 48 - 12 = 36 remain.
            Split across 3 baskets gives 36 / 3 = 12 muffins per basket.
            Answer: 12
            """,
            expected_outcome="6 blueberry muffins per basket",
            observed_outcome="12 muffins per basket",
            failure_mechanism=(
                "The response stops after the equal-basket step and omits the "
                "explicit premise that only half of each basket is blueberry."
            ),
            verifier_path="question_grounding.quantity_graph",
            verifier_signal=(
                "The quantity graph should require the final half-of-basket "
                "premise before binding the answer."
            ),
            source_refs=["exp203:923", "exp206:923", "exp207:923"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A scaffold with one field per transformation would expose the "
                "missing final premise."
            ),
        ),
        word_followup(
            example_id="exp214-followup-omitted-premise-2",
            taxonomy_label="omitted_premises",
            prompt="""
            A jacket costs $80. A 25% coupon applies, and then a $6 shipping
            fee is added. How much does the jacket cost altogether?
            """,
            response="""
            A 25% discount on $80 is $20, so the discounted price is $60.
            Answer: 60
            """,
            expected_outcome="$66 total",
            observed_outcome="$60",
            failure_mechanism=(
                "The response omits the explicit shipping-fee premise after "
                "correctly applying the discount."
            ),
            verifier_path="question_grounding.post_adjustment_chain",
            verifier_signal=(
                "The verifier should require every post-discount adjustment in "
                "the order given by the prompt."
            ),
            source_refs=["exp214:post-adjustment"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "Separate structured fields for discounted subtotal and final "
                "charge would reveal the omitted fee."
            ),
        ),
        word_followup(
            example_id="exp214-followup-omitted-premise-3",
            taxonomy_label="omitted_premises",
            prompt="""
            A painting crew finishes 6 rooms per full day. They work two full
            days and a half day on the third day. How many rooms do they paint?
            """,
            response="""
            They paint 6 rooms per day for 3 days, so 6 x 3 = 18 rooms.
            Answer: 18
            """,
            expected_outcome="15 rooms",
            observed_outcome="18 rooms",
            failure_mechanism=(
                "The response omits the half-day premise and treats the third "
                "day as a full workday."
            ),
            verifier_path="question_grounding.partial_interval",
            verifier_signal=(
                "The verifier should encode the half-day modifier as a partial "
                "interval rather than as another full day."
            ),
            source_refs=["exp214:partial-interval"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured day-by-day table would expose the shortened third day."
            ),
        ),
        word_followup(
            example_id="exp214-followup-omitted-premise-4",
            taxonomy_label="omitted_premises",
            prompt="""
            A courier drives 12 miles to town, 5 miles to a warehouse, and
            7 miles back home. How many miles does she drive total?
            """,
            response="""
            She drives 12 + 5 = 17 miles.
            Answer: 17
            """,
            expected_outcome="24 miles",
            observed_outcome="17 miles",
            failure_mechanism=("The response omits the final 7-mile trip back home."),
            verifier_path="question_grounding.event_timeline",
            verifier_signal=(
                "A route accumulator should include every named leg before the answer is bound."
            ),
            source_refs=["exp214:route-accumulator"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=(
                "A direct path accumulator can catch the omission without more free-form reasoning."
            ),
        ),
        word_followup(
            example_id="exp214-followup-omitted-premise-5",
            taxonomy_label="omitted_premises",
            prompt="""
            Three notebooks cost $4 each. After a $2 discount, a $3 service fee
            is added. How much is the final charge?
            """,
            response="""
            The notebooks cost 3 x 4 = 12 dollars.
            After the discount, 12 - 2 = 10.
            Answer: 10
            """,
            expected_outcome="$13 final charge",
            observed_outcome="$10",
            failure_mechanism=(
                "The response omits the explicit service fee after computing the "
                "discounted subtotal."
            ),
            verifier_path="question_grounding.post_adjustment_chain",
            verifier_signal=(
                "The verifier should require the service-fee step after the discount is applied."
            ),
            source_refs=["exp214:post-adjustment"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "Separating subtotal, discount, and fee fields would expose the "
                "missing last adjustment."
            ),
        ),
        word_followup(
            example_id="exp214-followup-omitted-premise-6",
            taxonomy_label="omitted_premises",
            prompt="""
            The first bus arrives in 8 minutes and waits 4 minutes. The second
            bus arrives 10 minutes after the first bus leaves and leaves
            immediately. The third bus arrives 6 minutes after the second bus
            leaves, and that is Mina's bus. How long does Mina wait in total?
            """,
            response="""
            Mina waits 8 minutes for the first bus, then 4 minutes while it is
            there, then 10 more minutes for the second bus: 8 + 4 + 10 = 22.
            Answer: 22
            """,
            expected_outcome="28 minutes",
            observed_outcome="22 minutes",
            failure_mechanism=(
                "The response omits the final 6-minute gap between the second bus and Mina's bus."
            ),
            verifier_path="question_grounding.temporal_schedule",
            verifier_signal=(
                "A schedule graph should require each bus arrival gap, including "
                "the final one leading to the target bus."
            ),
            source_refs=["exp206:1077", "exp207:1077"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured schedule with one row per bus would make the "
                "missing final gap explicit."
            ),
        ),
        word_followup(
            example_id="exp214-followup-omitted-premise-7",
            taxonomy_label="omitted_premises",
            prompt="""
            A gardener picks 60 tulips. 12 are donated. The rest are divided
            equally into 4 bouquets, and one quarter of each bouquet is red.
            How many red tulips are in each bouquet?
            """,
            response="""
            After donating 12 tulips, 48 remain.
            48 divided by 4 bouquets is 12 per bouquet.
            Answer: 12
            """,
            expected_outcome="3 red tulips per bouquet",
            observed_outcome="12 tulips per bouquet",
            failure_mechanism=(
                "The response omits the one-quarter-red premise after correctly "
                "computing the bouquet size."
            ),
            verifier_path="question_grounding.quantity_graph",
            verifier_signal=(
                "The verifier should require the final color-fraction step "
                "before the answer is bound."
            ),
            source_refs=["exp203:923", "exp206:923", "exp207:923"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured quantity graph would reveal the missing fraction step."
            ),
        ),
        word_followup(
            example_id="exp214-followup-omitted-premise-8",
            taxonomy_label="omitted_premises",
            prompt="""
            A museum charges $9 per ticket. Lena buys 4 tickets, but one ticket
            is free with the family promotion. How much does she pay?
            """,
            response="""
            Four tickets at $9 each cost 4 x 9 = 36.
            Answer: 36
            """,
            expected_outcome="$27",
            observed_outcome="$36",
            failure_mechanism=(
                "The response omits the free-ticket promotion even though it is "
                "an explicit premise."
            ),
            verifier_path="question_grounding.promotion_adjustment",
            verifier_signal=(
                "The verifier should apply the free-ticket promotion before pricing the basket."
            ),
            source_refs=["exp214:promotion-adjustment"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=("A direct promotion rule is more important than more prose."),
        ),
    ]


def build_entity_binding_followups() -> list[dict[str, Any]]:
    return [
        word_followup(
            example_id="exp214-followup-entity-binding-1",
            taxonomy_label="entity_quantity_binding_errors",
            prompt="""
            Rina buys 6 shirts at $10 each and gets 20% off the total. She later
            sells 2 shirts for $18 total. How much money is she out?
            """,
            response="""
            The original total is 6 x 10 = 60.
            After 20% off, she pays 48.
            Selling 2 shirts for $18 each gives 36 back.
            She is out 48 - 36 = 12.
            Answer: 12
            """,
            expected_outcome="$30",
            observed_outcome="$12",
            failure_mechanism=(
                "The response binds the $18 resale amount to each shirt instead "
                "of to the total resale event."
            ),
            verifier_path="question_grounding.scope_alignment",
            verifier_signal=(
                "A scope-alignment check should attach the resale amount to the full sale event."
            ),
            source_refs=["exp203:943", "exp206:943", "exp207:943"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "Explicit event-level fields for purchase total and resale total "
                "would make the binding scope visible."
            ),
        ),
        word_followup(
            example_id="exp214-followup-entity-binding-2",
            taxonomy_label="entity_quantity_binding_errors",
            prompt="""
            Five horses each need shoes on all four hooves. Shoes are sold in
            pairs for $9. How much will all the horses' shoes cost?
            """,
            response="""
            Each horse needs one pair of shoes, so 5 horses need 5 pairs.
            5 x 9 = 45.
            Answer: 45
            """,
            expected_outcome="$90",
            observed_outcome="$45",
            failure_mechanism=(
                "The response binds one pair to each horse instead of one pair "
                "to each pair of hooves."
            ),
            verifier_path="question_grounding.count_expansion",
            verifier_signal=(
                "A count-expansion check should turn four hooves into two pairs per horse."
            ),
            source_refs=["exp206:506", "exp207:506"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=(
                "The core issue is entity binding, which a direct count rule can "
                "catch without more hidden reasoning."
            ),
        ),
        word_followup(
            example_id="exp214-followup-entity-binding-3",
            taxonomy_label="entity_quantity_binding_errors",
            prompt="""
            Tara buys 9 packs of 4 candles for $3 each. She sells each candle
            for $5. What profit does she make?
            """,
            response="""
            There are 9 x 4 = 36 candles.
            The packs cost 9 x 3 = 27.
            Revenue is 36 x 5 = 180.
            Profit is 180 - 27 = 153.
            Answer: 153
            """,
            expected_outcome="$72",
            observed_outcome="$153",
            failure_mechanism=("The response binds '$3 each' to packs rather than to candles."),
            verifier_path="question_grounding.scope_alignment",
            verifier_signal=(
                "A modifier-scope check should attach the purchase price to each "
                "candle while keeping pack count and candles-per-pack separate."
            ),
            source_refs=["exp206:515", "exp207:515"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A schema with pack_count, items_per_pack, cost_per_item, and "
                "sale_price_per_item would expose the wrong binding."
            ),
        ),
        word_followup(
            example_id="exp214-followup-entity-binding-4",
            taxonomy_label="entity_quantity_binding_errors",
            prompt="""
            Three boxes hold 8 markers each. Two markers in each box are blue.
            How many blue markers are there?
            """,
            response="""
            There are 2 blue markers.
            Answer: 2
            """,
            expected_outcome="6 blue markers",
            observed_outcome="2 blue markers",
            failure_mechanism=(
                "The response binds the per-box blue count to the whole set of "
                "boxes instead of multiplying by the number of boxes."
            ),
            verifier_path="question_grounding.entity_binding",
            verifier_signal=(
                "An entity-binding check should multiply the per-box count by the number of boxes."
            ),
            source_refs=["exp214:per-box-binding"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=(
                "The missing signal is simply the scope of the per-box count."
            ),
        ),
        word_followup(
            example_id="exp214-followup-entity-binding-5",
            taxonomy_label="entity_quantity_binding_errors",
            prompt="""
            Four shelves hold 6 jars each. Each jar contains 3 coins. How many
            coins are there?
            """,
            response="""
            There are 4 shelves and each jar has 3 coins, so 4 x 3 = 12.
            Answer: 12
            """,
            expected_outcome="72 coins",
            observed_outcome="12 coins",
            failure_mechanism=(
                "The response skips the jar-per-shelf multiplicity and binds the "
                "coins-per-jar count directly to shelves."
            ),
            verifier_path="question_grounding.entity_binding",
            verifier_signal=(
                "The verifier should preserve all multiplicities in the nesting "
                "structure: shelves, jars per shelf, and coins per jar."
            ),
            source_refs=["exp214:nested-counts"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured multiplicity table would show which entity each quantity belongs to."
            ),
        ),
        word_followup(
            example_id="exp214-followup-entity-binding-6",
            taxonomy_label="entity_quantity_binding_errors",
            prompt="""
            A bakery sells 5 trays with 8 rolls on each tray for $2 each. How
            much revenue comes from the rolls?
            """,
            response="""
            Five trays at $2 each make 10 dollars.
            Answer: 10
            """,
            expected_outcome="$80",
            observed_outcome="$10",
            failure_mechanism=(
                "The response binds the unit price to trays instead of to the individual rolls."
            ),
            verifier_path="question_grounding.scope_alignment",
            verifier_signal=(
                "A price-scope check should attach the '$2 each' modifier to rolls, not to trays."
            ),
            source_refs=["exp206:515", "exp207:515"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=(
                "This is a direct modifier-scope error rather than a hidden "
                "reasoning visibility issue."
            ),
        ),
        word_followup(
            example_id="exp214-followup-entity-binding-7",
            taxonomy_label="entity_quantity_binding_errors",
            prompt="""
            Lina decorates 7 gift boxes with 3 ribbons on each box. Each ribbon
            costs $2. How much does she spend on ribbons?
            """,
            response="""
            Three ribbons cost 3 x 2 = 6.
            Answer: 6
            """,
            expected_outcome="$42",
            observed_outcome="$6",
            failure_mechanism=(
                "The response binds the per-box ribbon count to the whole task "
                "instead of to each of the 7 boxes."
            ),
            verifier_path="question_grounding.entity_binding",
            verifier_signal=(
                "An entity-binding check should expand the ribbon count across "
                "all boxes before pricing."
            ),
            source_refs=["exp214:per-box-binding"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=("A direct multiplicity check is enough to catch the error."),
        ),
    ]


def build_unit_aggregation_followups() -> list[dict[str, Any]]:
    return [
        word_followup(
            example_id="exp214-followup-unit-aggregation-1",
            taxonomy_label="unit_aggregation_errors",
            prompt="""
            A company starts with 200 employees and hires 20 each month. Each
            employee earns $4000 per month. How much payroll is paid over the
            first 3 months?
            """,
            response="""
            After 3 months the company has 260 employees.
            Payroll is 260 x 4000 = 1040000.
            Answer: 1040000
            """,
            expected_outcome="$2,880,000",
            observed_outcome="$1,040,000",
            failure_mechanism=(
                "The response prices only the final headcount instead of "
                "aggregating payroll across the month-1, month-2, and month-3 states."
            ),
            verifier_path="question_grounding.repeated_event_accumulator",
            verifier_signal=(
                "A repeated-event accumulator should sum each month's payroll "
                "rather than only the final state."
            ),
            source_refs=["exp206:796", "exp207:796"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A month-by-month table would make the aggregation requirement explicit."
            ),
        ),
        word_followup(
            example_id="exp214-followup-unit-aggregation-2",
            taxonomy_label="unit_aggregation_errors",
            prompt="""
            A cyclist rides 2 hours at 12 mph and then 30 minutes at 8 mph.
            How far does she go?
            """,
            response="""
            Her speed changes from 12 mph to 8 mph, so together that is 20 mph.
            She rides for 2.5 hours, so 20 x 2.5 = 50 miles.
            Answer: 50
            """,
            expected_outcome="28 miles",
            observed_outcome="50 miles",
            failure_mechanism=(
                "The response aggregates speeds directly instead of converting "
                "each segment into a distance before summing."
            ),
            verifier_path="question_grounding.segment_unit_reasoner",
            verifier_signal=(
                "The verifier should compute per-segment distances and only then aggregate them."
            ),
            source_refs=["exp214:segment-units"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "Structured fields for segment duration, segment rate, and "
                "segment distance would expose the unit error."
            ),
        ),
        word_followup(
            example_id="exp214-followup-unit-aggregation-3",
            taxonomy_label="unit_aggregation_errors",
            prompt="""
            A recipe uses 750 milliliters of broth per pot. Omar has 4.5 liters
            of broth. How many full pots can he make?
            """,
            response="""
            4.5 liters is 450 milliliters.
            450 / 750 = 0 full pots.
            Answer: 0
            """,
            expected_outcome="6 full pots",
            observed_outcome="0 full pots",
            failure_mechanism=(
                "The response mis-converts liters to milliliters before dividing "
                "by the per-pot requirement."
            ),
            verifier_path="question_grounding.unit_reasoner",
            verifier_signal=(
                "A unit check should normalize liters and milliliters before "
                "computing the quotient."
            ),
            source_refs=["exp214:unit-conversion"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=("The decisive signal is the conversion rule itself."),
        ),
        word_followup(
            example_id="exp214-followup-unit-aggregation-4",
            taxonomy_label="unit_aggregation_errors",
            prompt="""
            A bakery makes 120 loaves on Monday, 140 on Tuesday, and 160 on
            Wednesday. Each loaf sells for $3. How much revenue is earned across
            the three days?
            """,
            response="""
            On Wednesday they make 160 loaves.
            160 x 3 = 480.
            Answer: 480
            """,
            expected_outcome="$1260",
            observed_outcome="$480",
            failure_mechanism=(
                "The response prices only the final day's production instead of "
                "aggregating all three daily totals."
            ),
            verifier_path="question_grounding.aggregation",
            verifier_signal=(
                "An aggregation check should sum the daily loaf counts before pricing the total."
            ),
            source_refs=["exp214:aggregation"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured daily ledger would make the missing aggregation visible."
            ),
        ),
        word_followup(
            example_id="exp214-followup-unit-aggregation-5",
            taxonomy_label="unit_aggregation_errors",
            prompt="""
            A class reads 8 pages per student per day. There are 25 students and
            the class reads for 2 days. How many pages are read in total?
            """,
            response="""
            8 pages per student for 25 students is 8 x 25 = 200 pages.
            Answer: 200
            """,
            expected_outcome="400 pages",
            observed_outcome="200 pages",
            failure_mechanism=("The response omits the day dimension in the total aggregation."),
            verifier_path="question_grounding.aggregation",
            verifier_signal=(
                "The verifier should preserve all axes in the total: pages per "
                "student, students, and days."
            ),
            source_refs=["exp214:aggregation"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured total formula would reveal the missing day factor."
            ),
        ),
        word_followup(
            example_id="exp214-followup-unit-aggregation-6",
            taxonomy_label="unit_aggregation_errors",
            prompt="""
            A gym charges $20 per month plus a one-time $30 sign-up fee. What
            does 3 months cost?
            """,
            response="""
            Each month costs 20 + 30 = 50 dollars.
            Over 3 months that is 50 x 3 = 150.
            Answer: 150
            """,
            expected_outcome="$90",
            observed_outcome="$150",
            failure_mechanism=(
                "The response aggregates the one-time fee as if it recurred each month."
            ),
            verifier_path="question_grounding.recurring_vs_one_time",
            verifier_signal=(
                "The verifier should keep recurring and one-time charges in "
                "separate buckets before summing them."
            ),
            source_refs=["exp214:recurring-vs-onetime"],
            structured_reasoning_helpful=True,
            structured_reasoning_why=(
                "A structured breakdown by fee type would expose the one-time "
                "versus recurring distinction."
            ),
        ),
        word_followup(
            example_id="exp214-followup-unit-aggregation-7",
            taxonomy_label="unit_aggregation_errors",
            prompt="""
            Each gift basket weighs 2 kilograms, and the wrapping adds
            250 grams. What is the total weight of 4 wrapped baskets in grams?
            """,
            response="""
            Each basket is 2 + 250 = 252 grams.
            Four baskets weigh 252 x 4 = 1008 grams.
            Answer: 1008
            """,
            expected_outcome="9000 grams",
            observed_outcome="1008 grams",
            failure_mechanism=(
                "The response adds kilograms and grams directly without unit normalization."
            ),
            verifier_path="question_grounding.unit_reasoner",
            verifier_signal=(
                "A unit-normalization check should convert kilograms to grams before addition."
            ),
            source_refs=["exp214:unit-conversion"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=(
                "The relevant signal is the unit conversion rule, not more text."
            ),
        ),
        word_followup(
            example_id="exp214-followup-unit-aggregation-8",
            taxonomy_label="unit_aggregation_errors",
            prompt="""
            A train travels 90 minutes at 40 mph. How far does it go?
            """,
            response="""
            Distance is 90 x 40 = 3600.
            Answer: 3600
            """,
            expected_outcome="60 miles",
            observed_outcome="3600 miles",
            failure_mechanism=(
                "The response multiplies miles-per-hour by minutes without "
                "converting time into hours."
            ),
            verifier_path="question_grounding.unit_reasoner",
            verifier_signal=(
                "A unit check should normalize 90 minutes to 1.5 hours before "
                "multiplying by the speed."
            ),
            source_refs=["exp214:unit-conversion"],
            structured_reasoning_helpful=False,
            structured_reasoning_why=("A direct unit-normalization rule is sufficient here."),
        ),
    ]


def build_arithmetic_followups() -> list[dict[str, Any]]:
    return [
        arithmetic_followup(
            example_id="exp214-followup-arithmetic-1",
            prompt="""
            A classroom has 18 desks in each of 7 rows. Four desks are removed.
            How many desks remain?
            """,
            response="""
            There are 18 x 7 = 124 desks at first.
            After removing 4 desks, 124 - 4 = 120 remain.
            Answer: 120
            """,
            expected_outcome="122 desks",
            observed_outcome="120 desks",
            failure_mechanism=("The grounding is correct, but the multiplication step is wrong."),
        ),
        arithmetic_followup(
            example_id="exp214-followup-arithmetic-2",
            prompt="""
            Maya reads 37 pages on Monday and 28 pages on Tuesday. How many
            pages does she read in total?
            """,
            response="""
            37 + 28 = 55.
            Answer: 55
            """,
            expected_outcome="65 pages",
            observed_outcome="55 pages",
            failure_mechanism=("The only failure is an addition slip in the final sum."),
        ),
        arithmetic_followup(
            example_id="exp214-followup-arithmetic-3",
            prompt="""
            A store has $150 in the register and pays a $47 bill. How much
            money remains?
            """,
            response="""
            150 - 47 = 97.
            Answer: 97
            """,
            expected_outcome="$103",
            observed_outcome="$97",
            failure_mechanism=("The subtraction is incorrect even though the setup is right."),
        ),
        arithmetic_followup(
            example_id="exp214-followup-arithmetic-4",
            prompt="""
            Ninety-six cookies are packed equally into 8 boxes. How many
            cookies go in each box?
            """,
            response="""
            96 / 8 = 11.
            Answer: 11
            """,
            expected_outcome="12 cookies",
            observed_outcome="11 cookies",
            failure_mechanism=("The response chooses the right operation but divides incorrectly."),
        ),
        arithmetic_followup(
            example_id="exp214-followup-arithmetic-5",
            prompt="""
            A $80 jacket is discounted by 25%. How many dollars is the discount?
            """,
            response="""
            25% of 80 is 18.
            Answer: 18
            """,
            expected_outcome="$20",
            observed_outcome="$18",
            failure_mechanism=("The grounding is correct, but the percentage arithmetic is wrong."),
        ),
        arithmetic_followup(
            example_id="exp214-followup-arithmetic-6",
            prompt="""
            Fifty-four brownies are shared equally between 2 trays. How many
            brownies go on each tray?
            """,
            response="""
            54 / 2 = 22.
            Answer: 22
            """,
            expected_outcome="27 brownies",
            observed_outcome="22",
            failure_mechanism=("The response applies the intended arithmetic step incorrectly."),
        ),
        arithmetic_followup(
            example_id="exp214-followup-arithmetic-7",
            prompt="""
            A farmer fills 12 crates with 5 melons each and then adds 9 extra
            melons. How many melons are there altogether?
            """,
            response="""
            12 x 5 = 60.
            60 + 9 = 65.
            Answer: 65
            """,
            expected_outcome="69 melons",
            observed_outcome="65 melons",
            failure_mechanism=("The multiplication is correct, but the final addition is wrong."),
        ),
        arithmetic_followup(
            example_id="exp214-followup-arithmetic-8",
            prompt="""
            A recipe uses 3.5 cups of flour and then 2.75 more cups. How much
            flour is used in total?
            """,
            response="""
            3.5 + 2.75 = 6.0.
            Answer: 6.0
            """,
            expected_outcome="6.25 cups",
            observed_outcome="6.0 cups",
            failure_mechanism=("The decimal addition is incorrect despite correct grounding."),
        ),
        arithmetic_followup(
            example_id="exp214-followup-arithmetic-9",
            prompt="""
            A rectangle is 14 feet long and 9 feet wide. What is its area?
            """,
            response="""
            Area = 14 x 9 = 117.
            Answer: 117
            """,
            expected_outcome="126 square feet",
            observed_outcome="117 square feet",
            failure_mechanism=("The response selects the correct formula but multiplies wrongly."),
        ),
        arithmetic_followup(
            example_id="exp214-followup-arithmetic-10",
            prompt="""
            A game awards 7 points in each of 8 rounds, then subtracts a
            6-point penalty. What is the final score?
            """,
            response="""
            7 x 8 = 56.
            56 - 6 = 48.
            Answer: 48
            """,
            expected_outcome="50 points",
            observed_outcome="48 points",
            failure_mechanism=("The setup is right, but the final subtraction is wrong."),
        ),
    ]


def build_code_followups() -> list[dict[str, Any]]:
    return [
        code_followup(
            example_id="exp214-followup-code-dedupe-1",
            prompt="""
            Write `def dedupe_keep_order(items: list[str]) -> list[str]` that
            returns the first occurrence of each string in its original order.
            Do not mutate `items`.
            """,
            response="""
            def dedupe_keep_order(items: list[str]) -> list[str]:
                return sorted(set(items))
            """,
            expected_outcome="Preserve first-occurrence order without mutating the input list",
            observed_outcome="Sorts alphabetically and loses order-of-first-appearance semantics",
            failure_mechanism=(
                "The function violates the order-preserving property even though "
                "it removes duplicates."
            ),
            verifier_signal=(
                "A typed contract plus probe execution should catch that duplicate "
                "removal alone is not enough; the original order is part of the oracle."
            ),
            source_refs=["exp208:HumanEval/26"],
        ),
        code_followup(
            example_id="exp214-followup-code-chunks-1",
            prompt="""
            Write `def chunk(items: list[int], size: int) -> list[list[int]]`
            that returns consecutive chunks and keeps a final shorter chunk when
            the length is not divisible by `size`.
            """,
            response="""
            def chunk(items: list[int], size: int) -> list[list[int]]:
                return [items[i:i + size] for i in range(0, len(items) - size, size)]
            """,
            expected_outcome="Return all chunks including the final partial chunk",
            observed_outcome="Drops the tail chunk because the range stops too early",
            failure_mechanism=(
                "The implementation misses the final partial chunk, so the code "
                "fails the prompt's output-shape property."
            ),
            verifier_signal=(
                "Probe cases with non-divisible lengths should trigger a "
                "typed-property failure on the missing tail chunk."
            ),
            source_refs=["exp208:HumanEval/106"],
        ),
        code_followup(
            example_id="exp214-followup-code-score-1",
            prompt="""
            Write `def score_colors(text: str, weights: dict[str, int]) -> int`
            that adds each distinct color word at most once, in any order.
            Repeated color words should not be double-counted.
            """,
            response="""
            def score_colors(text: str, weights: dict[str, int]) -> int:
                total = 0
                for token in text.split():
                    total += weights.get(token, 0)
                return total
            """,
            expected_outcome="Count each distinct color at most once",
            observed_outcome="Counts duplicate color tokens repeatedly",
            failure_mechanism=(
                "The implementation satisfies the type signature but violates the "
                "distinct-token scoring oracle."
            ),
            verifier_signal=(
                "Execution probes with repeated colors should surface the missing "
                "deduplication property."
            ),
            source_refs=["exp208:HumanEval/79"],
        ),
        code_followup(
            example_id="exp214-followup-code-normalize-1",
            prompt="""
            Write `def normalize_names(names: list[str]) -> list[str]` that
            returns a new alphabetically sorted list of stripped lowercase names
            without mutating `names`.
            """,
            response="""
            def normalize_names(names: list[str]) -> list[str]:
                for index, name in enumerate(names):
                    names[index] = name.strip().lower()
                names.sort()
                return names
            """,
            expected_outcome="Return a normalized new list and leave the input list unchanged",
            observed_outcome="Mutates the input list in place before returning it",
            failure_mechanism=(
                "The implementation violates the no-mutation property even though "
                "the returned values look plausible."
            ),
            verifier_signal=(
                "Probe cases that inspect the input list after the call should "
                "flag the mutation oracle miss."
            ),
            source_refs=["exp208:HumanEval/90"],
        ),
        code_followup(
            example_id="exp214-followup-code-merge-1",
            prompt="""
            Write `def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]`
            that merges overlapping or touching closed intervals.
            """,
            response="""
            def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
                ordered = sorted(intervals)
                merged: list[tuple[int, int]] = []
                for start, end in ordered:
                    if not merged or start > merged[-1][1]:
                        merged.append((start, end))
                    else:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                return merged
            """,
            expected_outcome="Merge touching intervals such as (1, 3) and (4, 5)",
            observed_outcome=(
                "Leaves touching intervals separate because it only merges strict overlap"
            ),
            failure_mechanism=(
                "The implementation misses the prompt's closed-interval oracle for touching ranges."
            ),
            verifier_signal=(
                "A probe with touching intervals should fail the typed property "
                "that touching closed intervals must merge."
            ),
            source_refs=["exp208:HumanEval/127"],
        ),
        code_followup(
            example_id="exp214-followup-code-palindrome-1",
            prompt="""
            Write `def is_loose_palindrome(text: str) -> bool` that returns
            `True` when `text` is a palindrome after removing spaces and ignoring
            letter case.
            """,
            response="""
            def is_loose_palindrome(text: str) -> bool:
                return text == text[::-1]
            """,
            expected_outcome="Ignore spaces and case when checking palindromes",
            observed_outcome="Uses the raw string and fails on inputs like 'Never odd or even'",
            failure_mechanism=("The implementation misses the prompt's normalization oracle."),
            verifier_signal=(
                "Probe strings with mixed case and spaces should trigger a property failure."
            ),
            source_refs=["exp208:HumanEval/1"],
        ),
        code_followup(
            example_id="exp214-followup-code-moving-average-1",
            prompt="""
            Write `def prefix_moving_average(values: list[float], window: int) -> list[float]`
            that returns a list the same length as `values`, using shorter
            prefix windows until the full window is available.
            """,
            response="""
            def prefix_moving_average(values: list[float], window: int) -> list[float]:
                result: list[float] = []
                for index in range(window - 1, len(values)):
                    chunk = values[index - window + 1:index + 1]
                    result.append(sum(chunk) / window)
                return result
            """,
            expected_outcome="Return a same-length list with prefix-aware averages",
            observed_outcome="Drops the prefix positions and only returns full-window averages",
            failure_mechanism=(
                "The implementation satisfies the numeric type but violates the "
                "same-length output oracle."
            ),
            verifier_signal=(
                "Probe cases that check output length and prefix behavior should "
                "flag the missing prefix averages."
            ),
            source_refs=["exp208:HumanEval/64"],
        ),
        code_followup(
            example_id="exp214-followup-code-clamp-1",
            prompt="""
            Write `def clamp_values(values: list[int], low: int, high: int) -> list[int]`
            that returns a list the same length as `values`, replacing any value
            below `low` with `low` and any value above `high` with `high`.
            """,
            response="""
            def clamp_values(values: list[int], low: int, high: int) -> list[int]:
                return [value for value in values if low <= value <= high]
            """,
            expected_outcome="Clamp every value while preserving the original list length",
            observed_outcome="Filters out-of-range values and shortens the output",
            failure_mechanism=(
                "The implementation violates the prompt's same-length clamp oracle."
            ),
            verifier_signal=(
                "Probe cases with out-of-range values should catch that the "
                "function filters instead of clamping."
            ),
            source_refs=["exp208:HumanEval/92"],
        ),
        code_followup(
            example_id="exp214-followup-code-topk-1",
            prompt="""
            Write `def top_k_words(words: list[str], k: int) -> list[str]` that
            returns the `k` most frequent words. Break ties alphabetically.
            """,
            response="""
            def top_k_words(words: list[str], k: int) -> list[str]:
                counts: dict[str, int] = {}
                for word in words:
                    counts[word] = counts.get(word, 0) + 1
                ranked = sorted(counts, key=counts.get, reverse=True)
                return ranked[:k]
            """,
            expected_outcome="Use alphabetical order when frequencies tie",
            observed_outcome="Tie ordering depends on dict order rather than alphabetical order",
            failure_mechanism=(
                "The function misses the tie-break oracle even though the counts "
                "themselves are correct."
            ),
            verifier_signal=(
                "Probe cases with tied frequencies should surface the missing "
                "alphabetical tie-break property."
            ),
            source_refs=["exp208:HumanEval/138"],
        ),
        code_followup(
            example_id="exp214-followup-code-rotate-1",
            prompt="""
            Write `def rotate_clockwise(matrix: list[list[int]]) -> list[list[int]]`
            that returns a new matrix rotated 90 degrees clockwise without
            mutating the input matrix.
            """,
            response="""
            def rotate_clockwise(matrix: list[list[int]]) -> list[list[int]]:
                return [list(row) for row in zip(*matrix)]
            """,
            expected_outcome="Return a clockwise rotation as a new matrix",
            observed_outcome="Returns only the transpose, which is not a clockwise rotation",
            failure_mechanism=(
                "The implementation preserves types but misses the directional rotation oracle."
            ),
            verifier_signal=(
                "Execution probes on a non-symmetric matrix should catch that the "
                "transpose is not the requested clockwise rotation."
            ),
            source_refs=["exp208:HumanEval/150"],
        ),
    ]


def build_targeted_follow_up_examples() -> list[dict[str, Any]]:
    return (
        build_question_grounding_followups()
        + build_omitted_premise_followups()
        + build_entity_binding_followups()
        + build_unit_aggregation_followups()
        + build_arithmetic_followups()
        + build_code_followups()
    )


def build_corpus() -> list[dict[str, Any]]:
    return build_live_trace_examples() + build_targeted_follow_up_examples()


def build_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source_type = Counter(record["source_type"] for record in records)
    by_source_artifact = Counter(record["source_artifact"] for record in records)
    by_domain = Counter(record["domain"] for record in records)
    by_taxonomy = Counter(record["gold_diagnosis"]["taxonomy_label"] for record in records)
    by_verifier_path = Counter(
        record["expected_verifier_signal"]["verifier_path"] for record in records
    )
    by_structured_reasoning_helpful = Counter(
        "true" if record["structured_reasoning_helpful"] else "false" for record in records
    )

    coverage_checks = {
        "min_example_count_met": len(records) >= 60,
        "has_live_trace_examples": by_source_type["live_trace"] > 0,
        "has_targeted_follow_up_examples": by_source_type["targeted_follow_up"] > 0,
        "covers_required_taxonomy": all(by_taxonomy[label] > 0 for label in TAXONOMY_LABELS),
        "has_exp208_informed_code_followups": by_source_artifact["exp208_followup"] > 0,
    }

    return {
        "experiment": EXPERIMENT_LABEL,
        "run_date": RUN_DATE,
        "title": "Semantic failure corpus for verifier training and semantic diagnosis",
        "artifacts": {
            "corpus_path": "data/research/semantic_failure_corpus_214.jsonl",
            "results_path": "results/experiment_214_results.json",
        },
        "summary": {
            "n_examples": len(records),
            "by_source_type": dict(by_source_type),
            "by_source_artifact": dict(by_source_artifact),
            "by_domain": dict(by_domain),
            "by_taxonomy": dict(by_taxonomy),
            "by_expected_verifier_signal": dict(by_verifier_path),
            "by_structured_reasoning_helpful": dict(by_structured_reasoning_helpful),
            "coverage_checks": coverage_checks,
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(record, sort_keys=False) for record in records) + "\n"
    path.write_text(body, encoding="utf-8")


def main() -> int:
    records = build_corpus()
    results = build_results(records)
    write_jsonl(CORPUS_PATH, records)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
