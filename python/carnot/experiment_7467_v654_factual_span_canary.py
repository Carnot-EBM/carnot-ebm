"""Run the factual development canary for the sealed claim-span comparison.

Exp7451 proved that the runtime was healthy but used three paragraphs with no
factual proposition. This module changes only the development panel and the
gate reduction. It keeps the model, prompts, evaluation roster, and lifecycle.

Spec refs: REQ-VERIFY-7467 and SCENARIO-VERIFY-7467-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any

from carnot import experiment_7442_v652_span_capture as engine
from carnot import experiment_7451_v653_span_capture as predecessor
from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.reporting.experiment_7303_validation_scope import CommandSpec


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.654"
PHASE = 2
EXPERIMENT_ID = "exp7467-v654-factual-span-canary"
TASK_ID = "experiment_7467_v654_factual_span_canary"
SCHEMA = "carnot.exp7467.v654.factual_span_canary.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

RESULT_PATH = Path("results/experiment_7467_v654_factual_span_canary.json")
RAW_DIR = Path("results/raw/experiment_7467_v654_factual_span_canary")
MODULE_PATH = Path("python/carnot/experiment_7467_v654_factual_span_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7467_v654_factual_span_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7467_v654_factual_span_canary.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
BROAD_SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
PREDECESSOR_PATH = Path("results/experiment_7451_v653_span_capture.json")
AUDIT_PATH = Path("results/experiment_7456_v653_extraction_audit.json")
CORPUS_PATH = Path("data/ragtruth/response.jsonl")
SEALED_SCHEDULE_PATH = Path("results/raw/experiment_7437_v652_span_protocol/sealed_schedule.json")
EVALUATOR_PATH = Path("results/raw/experiment_7437_v652_span_protocol/evaluator_annotations.json")

EXPECTED_PREDECESSOR_SHA256 = (
    "sha256:1f7bce1cccac24b3efc4dfac0d562d4373c1eb70e83426f934d1f65495d6047a"
)
EXPECTED_AUDIT_SHA256 = "sha256:03e22487655e4096aadbda495c8e4f34caca57f03f6600ff27438917c9795f06"
EXPECTED_CORPUS_SHA256 = "sha256:e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073"
EXPECTED_SCHEDULE_SHA256 = "sha256:df06ff0c4fe9a785238b97f7a9235aa16c8acc2778d54ac70af3a5b4a2a03007"

MODEL_SPECS = ["unsloth/Qwen3.8-27B-GGUF"]
INFERENCE_SUBSTRATE = "owned_native_cuda_llama_cpp_factual_span_generation"
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
EXECUTION_VENUE = "host"
RANDOM_SEED = predecessor.RANDOM_SEED
MAX_NEW_TOKENS = 256
DEVELOPMENT_PARAGRAPHS = 6
DEVELOPMENT_CALLS = 12
FACTUAL_PARAGRAPHS = 4
NONFACTUAL_PARAGRAPHS = 2
FACTUAL_USABLE_MINIMUM = 3
NONFACTUAL_EMPTY_MINIMUM = 2
EVALUATION_CALLS = 96
MAX_GENERATION_CALLS = 108
BOOTSTRAP_DRAWS = engine.BOOTSTRAP_DRAWS
EXPECTED_FACTUAL_RESPONSE_IDS = ("0", "7", "12", "18")
OLD_DEVELOPMENT_RESPONSE_IDS = frozenset({"14067", "5590", "12911", "7722"})
OLD_DEVELOPMENT_GROUP_IDS = frozenset({"15264", "13587", "14469", "14159"})

NONFACTUAL_CONTROLS: tuple[JsonDict, ...] = (
    {
        "response_id": "constructed-control-00",
        "group_id": "constructed-nonfactual",
        "paragraph": "Hello, and thank you for sharing the material.",
    },
    {
        "response_id": "constructed-control-01",
        "group_id": "constructed-nonfactual",
        "paragraph": "Please return the requested result in the specified format.",
    },
)

AFFECTED_CHECK_NAMES = engine.AFFECTED_CHECK_NAMES
TERMINAL_CHECK_NAMES = engine.TERMINAL_CHECK_NAMES
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

FIELD_PRINCIPLES = deepcopy(predecessor.FIELD_PRINCIPLES)
FIELD_PRINCIPLES.update(
    {
        "schema": "Use the exact V654 schema, experiment identity, milestone, phase, and terminal status.",
        "run_date": "Use 20260920 with measured UTC and monotonic boundaries tied to the current host boot.",
        "preconditions_checked": "Authenticate historical artifacts, flags, corpus bytes, selection, device identity, and ownership before model work.",
        "duration_s": "Measure current work and separate model, reduction, validation, and cold replay without padding.",
        "sample_size_budget": "Account for twelve development and 96 evaluation calls, including explicit unstarted dispositions.",
        "span_capture_complete_score": "Use bare zero or one for complete planned dispositions, even when development closes evaluation.",
        "span_value_score": "Use bare zero or one and require both registered paired confidence bounds plus qualifier nonloss.",
        "development_rows": "Retain all twelve factual and nonfactual canary calls with separate credit fields.",
        "development_gate": "Keep the compatibility alias for the complete factual development gate.",
        "factual_development_gate": "Record separate factual and nonfactual denominators and exact observed failures.",
        "paragraph_selection": "Bind deterministic annotation rules and exact pre-inference paragraph text without consulting model success.",
        "raw_reply_shards": "Expose every failed or empty-claim reply through content-addressed raw evidence for Exp7470.",
        "paired_output_token_ci95": "Use the paired bootstrap upper bound, not only the mean, for the token-cost benefit gate.",
        "span_value_gate_results": "Keep completion, qualifier, and token-cost benefit operands independently auditable.",
        "qualifier_retention_report": "Separate constructed exact qualifier authority from human source-support annotations.",
        "retirement": "Retire only a repeated factual-canary failure until another diagnosed cause changes.",
    }
)
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

SEMANTIC_SCOPE = {
    "real_paragraphs": "human_source_support_annotations_do_not_label_extraction_semantics",
    "constructed_pairs": "synthetic_exact_qualifier_authority",
    "natural_language_truth_established": False,
    "complete_semantic_coverage_established": False,
}

artifact_checksum = engine.artifact_checksum
sha256_file = engine.sha256_file
atomic_json = engine.atomic_json
load_object = predecessor.load_object

_ENGINE_BUILD_DEVELOPMENT_SCHEDULE = engine.build_development_schedule
_ENGINE_BUILD_CAPTURE_ROW = engine.build_capture_row
_ENGINE_REDUCE_DEVELOPMENT_GATE = engine.reduce_development_gate
_ENGINE_REDUCE_EVALUATION = engine.reduce_evaluation
_ENGINE_BASE_ARTIFACT = engine._base_artifact
_ENGINE_BUILD_BLOCKED_ARTIFACT = engine.build_blocked_artifact
_ENGINE_BUILD_FIXTURE_ARTIFACT = engine.build_fixture_artifact
_ENGINE_FIXTURE_RESPONSE = engine._fixture_response
_ENGINE_RUN_EXPERIMENT = engine.run_experiment
_PREDECESSOR_BUILD_CAPTURE_ROW = predecessor.build_capture_row
_PREDECESSOR_REDUCE_DEVELOPMENT_GATE = predecessor.reduce_development_gate
_PREDECESSOR_BASE_ARTIFACT = predecessor.base_artifact
_PREDECESSOR_MEASURED_ARTIFACT = predecessor.measured_artifact
_PREDECESSOR_VALIDATE_ARTIFACT = predecessor.validate_artifact


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print each phase boundary so the conductor can distinguish work from a stall."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7467] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _sha256_text(value: str) -> str:
    """Hash exact text because character offsets depend on unchanged bytes."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _first_sentence(value: str) -> str:
    """Take the first punctuated sentence with a fixed, dependency-free rule."""

    return re.split(r"(?<=[.!?])\s+", value.strip(), maxsplit=1)[0]


def select_development_panel(
    root: Path = REPO_ROOT,
    *,
    sealed_schedule: Mapping[str, Any] | None = None,
    expected_response_ids: Sequence[str] = EXPECTED_FACTUAL_RESPONSE_IDS,
) -> JsonDict:
    """Select the first four annotated factual sentences without reading model output.

    Empty RAGTruth labels mean that human annotators marked no unsupported span.
    That status supports corpus selection only. It is not a truth certificate for
    the new extractor or for facts outside the supplied source text.
    """

    schedule = dict(sealed_schedule or load_object(root / SEALED_SCHEDULE_PATH))
    evaluation = [dict(row) for row in schedule.get("rows") or []]
    evaluation_ids = {str(row.get("response_id")) for row in evaluation}
    evaluation_groups = {str(row.get("group_id")) for row in evaluation}
    evaluation_hashes = {str(row.get("paragraph_sha256")) for row in evaluation}

    factual: list[JsonDict] = []
    seen_groups: set[str] = set()
    with (root / CORPUS_PATH).open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            response_id = str(source.get("id"))
            group_id = str(source.get("source_id"))
            sentence = _first_sentence(str(source.get("response") or ""))
            eligible = bool(
                source.get("split") == "train"
                and source.get("quality") == "good"
                and source.get("labels") == []
                and response_id not in evaluation_ids
                and group_id not in evaluation_groups
                and response_id not in OLD_DEVELOPMENT_RESPONSE_IDS
                and group_id not in OLD_DEVELOPMENT_GROUP_IDS
                and group_id not in seen_groups
                and 60 <= len(sentence) <= 300
                and not re.match(r"(?i)(sure|based on|here)", sentence)
            )
            if not eligible:
                continue
            seen_groups.add(group_id)
            factual.append(
                {
                    "unit_id": f"factual-{len(factual):02d}",
                    "response_id": response_id,
                    "group_id": group_id,
                    "paragraph": sentence,
                    "paragraph_sha256": _sha256_text(sentence),
                    "canary_kind": "factual",
                    "human_annotations": deepcopy(source["labels"]),
                    "human_annotation_status": "no_unsupported_span_annotated",
                    "annotation_scope": "unchanged_response_source_support_only",
                    "split": source["split"],
                    "quality": source["quality"],
                    "clipped": False,
                    "complete_response_coverage_eligible": True,
                }
            )
            if len(factual) == FACTUAL_PARAGRAPHS:
                break

    observed_ids = tuple(row["response_id"] for row in factual)
    if observed_ids != tuple(expected_response_ids):
        raise ValueError(
            f"annotated_factual_selection_mismatch:{observed_ids!r}!={tuple(expected_response_ids)!r}"
        )

    controls = []
    for index, value in enumerate(NONFACTUAL_CONTROLS):
        row = deepcopy(value)
        paragraph = str(row["paragraph"])
        row.update(
            {
                "unit_id": f"nonfactual-{index:02d}",
                "paragraph_sha256": _sha256_text(paragraph),
                "canary_kind": "nonfactual_control",
                "human_annotations": [],
                "human_annotation_status": "constructed_empty_output_control",
                "annotation_scope": "constructed_nonfactual_control",
                "split": None,
                "quality": None,
                "clipped": False,
                "complete_response_coverage_eligible": True,
            }
        )
        controls.append(row)

    paragraphs = [*factual, *controls]
    overlap = {
        "group_ids": sorted({str(row["group_id"]) for row in paragraphs} & evaluation_groups),
        "paragraph_sha256": sorted(
            {str(row["paragraph_sha256"]) for row in paragraphs} & evaluation_hashes
        ),
        "response_ids": sorted({str(row["response_id"]) for row in paragraphs} & evaluation_ids),
    }
    if any(overlap.values()):
        raise ValueError(f"development_evaluation_overlap:{overlap}")
    return {
        "schema": "carnot.exp7467.development_panel.v1",
        "selected_before_inference": True,
        "selection_model_outputs_consulted": False,
        "factual_count": len(factual),
        "nonfactual_count": len(controls),
        "selection_rules": {
            "source_order": "response.jsonl byte order",
            "split": "train",
            "quality": "good",
            "human_annotation_rule": "labels is the empty list",
            "sentence_rule": "first whitespace-delimited punctuated sentence",
            "length_chars": [60, 300],
            "unique_source_groups": True,
            "exclude_old_development": True,
            "exclude_sealed_evaluation": True,
            "model_success_filter": False,
        },
        "source": {
            "path": CORPUS_PATH.as_posix(),
            "sha256": sha256_file(root / CORPUS_PATH),
            "human_annotation_scope": "RAGTruth unsupported-span annotations",
        },
        "evaluation_roster": {
            "path": SEALED_SCHEDULE_PATH.as_posix(),
            "sha256": sha256_file(root / SEALED_SCHEDULE_PATH),
            "call_count_after_pair_expansion": EVALUATION_CALLS,
        },
        "evaluation_overlap": overlap,
        "paragraphs": paragraphs,
    }


def build_development_schedule(development: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Render six frozen paragraphs through both unchanged extraction arms."""

    values = [deepcopy(dict(row)) for row in development]
    if len(values) != DEVELOPMENT_PARAGRAPHS:
        values = list(select_development_panel(REPO_ROOT)["paragraphs"])
    rows: list[JsonDict] = []
    for index, source in enumerate(values):
        source.update(
            {
                "case_index": index,
                "condition": "annotated_factual_canary"
                if source["canary_kind"] == "factual"
                else "nonfactual_empty_control",
                "capture_phase": "development",
            }
        )
        first = "span" if index % 2 == 0 else "verbatim"
        arms = (first, "verbatim" if first == "span" else "span")
        pair_id = f"development-{index:02d}"
        for arm_order, arm in enumerate(arms):
            row = engine._schedule_row(
                source,
                arm=arm,
                arm_order=arm_order,
                pair_id=pair_id,
                call_id=f"{pair_id}-{arm}",
            )
            row.update(
                {
                    key: deepcopy(source[key])
                    for key in (
                        "canary_kind",
                        "human_annotations",
                        "human_annotation_status",
                        "annotation_scope",
                    )
                }
            )
            rows.append(row)
    return rows


def build_capture_row(schedule: Mapping[str, Any], response: Mapping[str, Any]) -> JsonDict:
    """Add factual and empty-control credit without changing raw parsing."""

    row = _PREDECESSOR_BUILD_CAPTURE_ROW(schedule, response)
    if row.get("capture_phase") != "development":
        return row
    factual = row.get("canary_kind") == "factual"
    disposition = row.get("development_disposition")
    row.update(
        {
            "factual_recall_success": disposition == "usable_nonempty" if factual else None,
            "correct_empty_control": disposition == "correct_empty" if not factual else False,
            "correct_empty_counts_as_factual_success": False,
        }
    )
    return row


def _failure_rows(rows: Sequence[Mapping[str, Any]], success: str) -> JsonDict:
    """Expose exact failed calls by arm instead of reducing them to one count."""

    return {
        arm: [
            {
                "call_id": row.get("call_id"),
                "response_id": row.get("response_id"),
                "disposition": row.get("development_disposition"),
            }
            for row in rows
            if row.get("arm") == arm and row.get("development_disposition") != success
        ]
        for arm in engine.protocol.ARMS
    }


def reduce_development_gate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require factual recall and correct empty controls as separate operands."""

    values = [deepcopy(dict(row)) for row in rows]
    factual = [row for row in values if row.get("canary_kind") == "factual"]
    controls = [row for row in values if row.get("canary_kind") == "nonfactual_control"]
    usable = {
        arm: sum(
            row.get("development_disposition") == "usable_nonempty"
            for row in factual
            if row.get("arm") == arm
        )
        for arm in engine.protocol.ARMS
    }
    correct_empty = {
        arm: sum(
            row.get("development_disposition") == "correct_empty"
            for row in controls
            if row.get("arm") == arm
        )
        for arm in engine.protocol.ARMS
    }
    factual_shape = len(factual) == FACTUAL_PARAGRAPHS * 2 and all(
        sum(row.get("arm") == arm for row in factual) == FACTUAL_PARAGRAPHS
        for arm in engine.protocol.ARMS
    )
    control_shape = len(controls) == NONFACTUAL_PARAGRAPHS * 2 and all(
        sum(row.get("arm") == arm for row in controls) == NONFACTUAL_PARAGRAPHS
        for arm in engine.protocol.ARMS
    )
    factual_pass = factual_shape and all(
        usable[arm] >= FACTUAL_USABLE_MINIMUM for arm in engine.protocol.ARMS
    )
    control_pass = control_shape and all(
        correct_empty[arm] == NONFACTUAL_EMPTY_MINIMUM for arm in engine.protocol.ARMS
    )
    opened = bool(len(values) == DEVELOPMENT_CALLS and factual_pass and control_pass)
    return {
        "planned": DEVELOPMENT_CALLS,
        "attempted": sum(row.get("attempted") is True for row in values),
        "completed": sum(row.get("terminal_state") == "response" for row in values),
        "factual": {
            "paragraphs": FACTUAL_PARAGRAPHS,
            "planned_calls": FACTUAL_PARAGRAPHS * 2,
            "planned_per_arm": FACTUAL_PARAGRAPHS,
            "required_usable_per_arm": FACTUAL_USABLE_MINIMUM,
            "usable_by_arm": usable,
            "failures": _failure_rows(factual, "usable_nonempty"),
            "passed": factual_pass,
        },
        "nonfactual": {
            "paragraphs": NONFACTUAL_PARAGRAPHS,
            "planned_calls": NONFACTUAL_PARAGRAPHS * 2,
            "planned_per_arm": NONFACTUAL_PARAGRAPHS,
            "required_correct_empty_per_arm": NONFACTUAL_EMPTY_MINIMUM,
            "correct_empty_by_arm": correct_empty,
            "failures": _failure_rows(controls, "correct_empty"),
            "passed": control_pass,
        },
        "usable_by_arm": usable,
        "required_usable_per_arm": FACTUAL_USABLE_MINIMUM,
        "disposition_counts": dict(
            sorted(Counter(str(row.get("development_disposition")) for row in values).items())
        ),
        "correct_empty_counts_as_factual_success": False,
        "capture_open": opened,
        "terminal_class": "ready" if opened else "null",
        "repeated_factual_canary_failure_retires_construction": not factual_pass,
    }


def paired_interval(differences: Sequence[float]) -> JsonDict:
    """Compute the registered paired percentile interval with one frozen seed."""

    values = [float(value) for value in differences]
    if not values:
        return {
            "ci95_high": None,
            "ci95_low": None,
            "draws": BOOTSTRAP_DRAWS,
            "estimate": None,
            "pairs": 0,
            "seed": RANDOM_SEED,
        }
    rng = random.Random(RANDOM_SEED)
    draws = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(BOOTSTRAP_DRAWS)
    )
    return {
        "ci95_high": draws[int(0.975 * (BOOTSTRAP_DRAWS - 1))],
        "ci95_low": draws[int(0.025 * (BOOTSTRAP_DRAWS - 1))],
        "draws": BOOTSTRAP_DRAWS,
        "estimate": sum(values) / len(values),
        "pairs": len(values),
        "seed": RANDOM_SEED,
    }


def span_value_gates(
    completion: Mapping[str, Any], token_cost: Mapping[str, Any], qualifier_delta: float | None
) -> list[JsonDict]:
    """Keep every scientific benefit operand visible and independently testable."""

    declarations = (
        (
            "paired_completion_ci95_lower",
            ">",
            0.0,
            completion.get("ci95_low"),
            completion.get("ci95_low") is not None and float(completion["ci95_low"]) > 0.0,
            "Span completion must improve on verbatim with a positive paired lower bound.",
        ),
        (
            "qualifier_nonloss",
            ">=",
            0.0,
            qualifier_delta,
            qualifier_delta is not None and qualifier_delta >= 0.0,
            "Constructed exact cases must show no increase in qualifier loss.",
        ),
        (
            "paired_token_cost_ci95_upper",
            "<",
            0.0,
            token_cost.get("ci95_high"),
            token_cost.get("ci95_high") is not None and float(token_cost["ci95_high"]) < 0.0,
            "Span output-token cost must have a negative paired upper bound.",
        ),
    )
    return [
        {
            "check": check,
            "category": "benefit",
            "operator": operator,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "principle": principle,
        }
        for check, operator, expected, observed, passed, principle in declarations
    ]


def reduce_span_value(
    completion: Mapping[str, Any], token_cost: Mapping[str, Any], qualifier_delta: float | None
) -> int:
    """Return one only when all registered paired benefit gates pass."""

    return int(
        all(
            row["passed"] is True
            for row in span_value_gates(completion, token_cost, qualifier_delta)
        )
    )


def qualifier_retention_report(semantic_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Separate exact constructed checks from human source-support annotations."""

    evaluator = load_object(REPO_ROOT / EVALUATOR_PATH)
    annotation_rows = [dict(row) for row in evaluator.get("rows") or []]
    span_retained = sum(row.get("span_qualifier_retained") is True for row in semantic_rows)
    verbatim_retained = sum(row.get("verbatim_qualifier_retained") is True for row in semantic_rows)
    return {
        "constructed_exact_cases": {
            "authority": "constructed_exact_string",
            "pairs": len(semantic_rows),
            "span_retained": span_retained,
            "verbatim_retained": verbatim_retained,
            "paired_delta": (
                (span_retained - verbatim_retained) / len(semantic_rows) if semantic_rows else None
            ),
        },
        "human_natural_language_annotations": {
            "authority": evaluator.get("annotation_scope"),
            "paragraphs": len(annotation_rows),
            "paragraphs_with_unsupported_span": sum(
                int(row.get("human_annotation_count", 0) or 0) > 0 for row in annotation_rows
            ),
            "explicit_qualifier_labels_available": False,
            "qualifier_retention_rate": None,
            "factual_truth_established": False,
            "complete_semantic_coverage_established": False,
        },
    }


def reduce_evaluation(
    rows: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Add the registered token interval to the unchanged paired reduction."""

    reduced = _ENGINE_REDUCE_EVALUATION(rows, controls)
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in reduced["extraction_rows"]:
        pairs[str(row.get("pair_id"))].append(row)
    token_differences: list[float] = []
    for pair in pairs.values():
        arms = {str(row.get("arm")): row for row in pair}
        if set(arms) == set(engine.protocol.ARMS) and all(
            arms[arm].get("attempted") is True for arm in engine.protocol.ARMS
        ):
            token_differences.append(
                float(arms["span"].get("completion_tokens", 0) or 0)
                - float(arms["verbatim"].get("completion_tokens", 0) or 0)
            )
    token_ci = paired_interval(token_differences)
    completion = dict(reduced["paired_completion_advantage"])
    qualifier_delta = reduced.get("constructed_qualifier_delta")
    reduced.update(
        {
            "paired_output_token_ci95": token_ci,
            "span_value_gate_results": span_value_gates(completion, token_ci, qualifier_delta),
            "span_value_score": reduce_span_value(completion, token_ci, qualifier_delta),
            "qualifier_retention_report": qualifier_retention_report(reduced["semantic_pair_rows"]),
        }
    )
    return reduced


def build_raw_reply_shards(
    rows: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Expose every failed or empty-claim reply through its raw byte receipt."""

    if len(rows) != len(references):
        raise ValueError("reply_shard_count_mismatch")
    exposed: list[JsonDict] = []
    for row, reference in zip(rows, references, strict=True):
        failed_or_empty = bool(
            not list(row.get("claims") or [])
            or row.get("disposition") in {"failed", "malformed", "truncated", "cancelled"}
            or not str(row.get("raw_reply") or "").strip()
        )
        if not failed_or_empty:
            continue
        path = Path(str(reference.get("path") or ""))
        if not path.is_absolute():
            path = RAW_DIR / "owned_runtime" / path
        exposed.append(
            {
                **deepcopy(dict(reference)),
                "path": path.as_posix(),
                "call_id": row.get("call_id"),
                "capture_phase": row.get("capture_phase"),
                "arm": row.get("arm"),
                "disposition": row.get("development_disposition") or row.get("disposition"),
                "raw_reply_empty": not bool(str(row.get("raw_reply") or "").strip()),
                "correct_empty_control": row.get("correct_empty_control") is True,
                "factual_recall_success": row.get("factual_recall_success"),
            }
        )
    return exposed


class FactualShardRecorder(engine._ShardRecorder):  # pragma: no cover - live callback boundary.
    """Stop only after all twelve development calls have a durable disposition."""

    def __call__(self, phase: str, event: str, **details: Any) -> None:
        progress(self.started, phase, event, **details)
        if (phase, event) == ("load", "before_model_load"):
            self._append("model-load", "model_load", "attempted")
        elif (phase, event) == ("load", "after_model_load") and details.get("healthy") is True:
            self._append("model-load", "model_load", "completed")
        elif (phase, event) == ("generation", "before_call"):
            self._append(f"generation-{int(details['completed'])}", "generation", "attempted")
        elif (phase, event) == ("generation", "after_call"):
            state = "completed" if details.get("terminal_state") == "response" else "failed"
            self._append(f"generation-{int(details['completed']) - 1}", "generation", state)
            if details.get("completed") == DEVELOPMENT_CALLS:
                gate = reduce_development_gate(self.development_rows)
                if gate["capture_open"] is not True:
                    raise engine._DevelopmentGateClosed(
                        "factual_development_gate:"
                        + json.dumps(
                            {
                                "factual": gate["factual"],
                                "nonfactual": gate["nonfactual"],
                            },
                            sort_keys=True,
                        )
                    )


def _fixture_response(row: Mapping[str, Any], *, tokens: int) -> JsonDict:
    """Make constructed nonfactual fixture calls return the required empty list."""

    if row.get("canary_kind") != "nonfactual_control":
        return _ENGINE_FIXTURE_RESPONSE(row, tokens=tokens)
    reply = json.dumps({"claims": []})
    return {
        "raw_request": {
            "messages": [{"role": "user", "content": row["prompt"]}],
            "temperature": engine.TEMPERATURE,
            "max_tokens": MAX_NEW_TOKENS,
        },
        "raw_response": {
            "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": tokens},
        },
        "raw_reply": reply,
        "attempted": True,
        "terminal_state": "response",
        "finish_reason": "stop",
        "prompt_tokens": 20,
        "completion_tokens": tokens,
        "latency_s": 0.1,
        "runtime_identity_receipt": {"pid": 1234, "start_time_ticks": 5678},
    }


@contextmanager
def _historical_engine_contract() -> Iterator[None]:
    """Use the exact V653 identity only while authenticating historical inputs."""

    values = {
        "MILESTONE": predecessor.MILESTONE,
        "EXPERIMENT_ID": predecessor.EXPERIMENT_ID,
        "TASK_ID": predecessor.TASK_ID,
        "SCHEMA": predecessor.SCHEMA,
        "RESULT_PATH": predecessor.RESULT_PATH,
        "RAW_DIR": predecessor.RAW_DIR,
        "MODULE_PATH": predecessor.MODULE_PATH,
        "WRAPPER_PATH": predecessor.WRAPPER_PATH,
        "TEST_PATH": predecessor.TEST_PATH,
        "SPEC_PATH": predecessor.SPEC_PATH,
        "MODEL_SPECS": predecessor.MODEL_SPECS,
        "INFERENCE_SUBSTRATE": predecessor.INFERENCE_SUBSTRATE,
        "INFERENCE_SUBSTRATE_CLASS": predecessor.INFERENCE_SUBSTRATE_CLASS,
        "EXECUTION_VENUE": predecessor.EXECUTION_VENUE,
        "RANDOM_SEED": predecessor.RANDOM_SEED,
        "DEVELOPMENT_PARAGRAPHS": 4,
        "DEVELOPMENT_CALLS": 8,
        "DEVELOPMENT_USABLE_MINIMUM": 3,
        "MAX_GENERATION_CALLS": 104,
        "VALIDATION_MANIFEST": predecessor.VALIDATION_MANIFEST,
        "FIELD_PRINCIPLES": predecessor.FIELD_PRINCIPLES,
        "REQUIRED_FIELDS": predecessor.REQUIRED_FIELDS,
        "build_development_schedule": _ENGINE_BUILD_DEVELOPMENT_SCHEDULE,
        "build_capture_row": _PREDECESSOR_BUILD_CAPTURE_ROW,
        "reduce_development_gate": _PREDECESSOR_REDUCE_DEVELOPMENT_GATE,
        "_base_artifact": _PREDECESSOR_BASE_ARTIFACT,
        "progress": predecessor.progress,
    }
    previous = {name: getattr(engine, name) for name in values}
    try:
        for name, value in values.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)


def _gate(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    path: Path,
    field: str,
    operator: str = "==",
    principle: str = "Only exact authenticated historical and selection evidence can authorize current model work.",
) -> JsonDict:
    """Create one exact V654 precondition without relying on old default paths."""

    return engine._gate(
        check,
        "precondition",
        operator,
        expected,
        observed,
        passed,
        principle,
        upstream=path.as_posix(),
        path=path.as_posix(),
        field=field,
    )


def collect_preconditions(root: Path = REPO_ROOT) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate history, annotations, selection, and the unchanged evaluation roster."""

    with _historical_engine_contract():
        checks, context = predecessor._collect_preconditions(root)
    checks = [deepcopy(dict(row)) for row in checks]
    context = deepcopy(dict(context))
    predecessor_value = load_object(root / PREDECESSOR_PATH)
    audit_value = load_object(root / AUDIT_PATH)
    historical = (
        (
            "exp7451_artifact_hash",
            PREDECESSOR_PATH,
            EXPECTED_PREDECESSOR_SHA256,
            sha256_file(root / PREDECESSOR_PATH) if (root / PREDECESSOR_PATH).is_file() else None,
        ),
        (
            "exp7456_audit_hash",
            AUDIT_PATH,
            EXPECTED_AUDIT_SHA256,
            sha256_file(root / AUDIT_PATH) if (root / AUDIT_PATH).is_file() else None,
        ),
        (
            "annotated_corpus_hash",
            CORPUS_PATH,
            EXPECTED_CORPUS_SHA256,
            sha256_file(root / CORPUS_PATH) if (root / CORPUS_PATH).is_file() else None,
        ),
        (
            "sealed_evaluation_schedule_hash",
            SEALED_SCHEDULE_PATH,
            EXPECTED_SCHEDULE_SHA256,
            sha256_file(root / SEALED_SCHEDULE_PATH)
            if (root / SEALED_SCHEDULE_PATH).is_file()
            else None,
        ),
    )
    checks.extend(
        _gate(name, expected, observed, observed == expected, path=path, field="sha256")
        for name, path, expected, observed in historical
    )
    declarations = (
        (
            "exp7451_identity",
            PREDECESSOR_PATH,
            "experiment_id",
            "exp7451-v653-span-capture",
            predecessor_value.get("experiment_id"),
        ),
        (
            "exp7451_null_verdict",
            PREDECESSOR_PATH,
            "verdict_class",
            "null",
            predecessor_value.get("verdict_class"),
        ),
        (
            "exp7451_unflagged",
            PREDECESSOR_PATH,
            "flagged_adversarial",
            False,
            predecessor_value.get("flagged_adversarial"),
        ),
        (
            "exp7451_gate_closed",
            PREDECESSOR_PATH,
            "development_gate.capture_open",
            False,
            dict(predecessor_value.get("development_gate") or {}).get("capture_open"),
        ),
        (
            "exp7456_identity",
            AUDIT_PATH,
            "experiment_id",
            "exp7456-v653-extraction-audit",
            audit_value.get("experiment_id"),
        ),
        (
            "exp7456_null_verdict",
            AUDIT_PATH,
            "verdict_class",
            "null",
            audit_value.get("verdict_class"),
        ),
        (
            "exp7456_unflagged",
            AUDIT_PATH,
            "flagged_adversarial",
            False,
            audit_value.get("flagged_adversarial"),
        ),
    )
    checks.extend(
        _gate(name, expected, observed, observed == expected, path=path, field=field)
        for name, path, field, expected, observed in declarations
    )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        _gate(
            "driving_requirement_v654",
            "REQ-VERIFY-7467",
            "REQ-VERIFY-7467" if "REQ-VERIFY-7467" in spec else None,
            "REQ-VERIFY-7467" in spec,
            path=SPEC_PATH,
            field="REQ-*",
            principle="The V654 requirement must exist before current model work.",
        )
    )
    broad_spec = root / BROAD_SPEC_PATH
    broad_observed = (
        "readable_nonempty_bytes" if broad_spec.is_file() and broad_spec.stat().st_size else None
    )
    checks.append(
        _gate(
            "source_bytes:openspec/capabilities/verification/spec.md",
            "readable_nonempty_bytes",
            broad_observed,
            broad_observed == "readable_nonempty_bytes",
            path=BROAD_SPEC_PATH,
            field="bytes",
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7467" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        _gate(
            "current_task_not_quarantined_v654",
            False,
            excluded,
            not excluded,
            path=Path("ops/exclusion_manifest.yaml"),
            field=EXPERIMENT_ID,
        )
    )
    try:
        selection = select_development_panel(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        selection = {}
        selection_error: str | None = f"{type(exc).__name__}:{exc}"
        development_schedule: list[JsonDict] = []
    else:
        selection_error = None
        development_schedule = build_development_schedule(selection["paragraphs"])
    checks.append(
        _gate(
            "frozen_factual_development_panel",
            {"error": None, "paragraphs": 6, "calls": 12, "overlap": False},
            {
                "error": selection_error,
                "paragraphs": len(selection.get("paragraphs") or []),
                "calls": len(development_schedule),
                "overlap": any(dict(selection.get("evaluation_overlap") or {}).values()),
            },
            selection_error is None
            and len(selection.get("paragraphs") or []) == DEVELOPMENT_PARAGRAPHS
            and len(development_schedule) == DEVELOPMENT_CALLS
            and not any(dict(selection.get("evaluation_overlap") or {}).values()),
            path=CORPUS_PATH,
            field="selection_rules",
        )
    )
    evaluation_schedule = list(context.get("evaluation_schedule") or [])
    predecessor_schedule = dict(predecessor_value.get("protocol_receipt") or {}).get(
        "schedule_sha256"
    )
    current_schedule = (
        evaluation_schedule[0].get("evaluation_schedule_sha256") if evaluation_schedule else None
    )
    checks.append(
        _gate(
            "sealed_96_call_roster_unchanged",
            predecessor_schedule,
            current_schedule,
            len(evaluation_schedule) == EVALUATION_CALLS
            and current_schedule == predecessor_schedule,
            path=PREDECESSOR_PATH,
            field="protocol_receipt.schedule_sha256",
        )
    )
    context.update(
        {
            "development_schedule": development_schedule,
            "paragraph_selection": selection,
        }
    )
    source_hashes = deepcopy(dict(context.get("source_hashes") or {}))
    for path, value, source_class in (
        (PREDECESSOR_PATH, predecessor_value, "historical_model_producer_artifact"),
        (AUDIT_PATH, audit_value, "historical_audit_artifact"),
        (CORPUS_PATH, {}, "human_annotated_selection_corpus"),
        (SEALED_SCHEDULE_PATH, {}, "sealed_evaluation_roster"),
        (EVALUATOR_PATH, {}, "human_annotation_authority"),
        (BROAD_SPEC_PATH, {}, "required_read_source"),
        (MODULE_PATH, {}, "current_experiment_code"),
        (WRAPPER_PATH, {}, "current_experiment_entrypoint"),
        (TEST_PATH, {}, "current_experiment_tests"),
    ):
        resolved = root / path
        if resolved.is_file():
            source_hashes[path.as_posix()] = {
                "path": path.as_posix(),
                "sha256": sha256_file(resolved),
                "source_receipt_class": source_class,
                **(
                    {
                        "original_verdict_class": value.get("verdict_class"),
                        "original_flagged_adversarial": value.get("flagged_adversarial"),
                    }
                    if value
                    else {}
                ),
            }
    context["source_hashes"] = source_hashes
    return checks, context


def base_artifact() -> JsonDict:
    """Extend the shared terminal shape with V654 selection and evidence fields."""

    value = _ENGINE_BASE_ARTIFACT()
    value.update(
        {
            "phase": PHASE,
            "semantic_scope": deepcopy(SEMANTIC_SCOPE),
            "paragraph_selection": {},
            "factual_development_gate": {},
            "raw_reply_shards": [],
            "paired_output_token_ci95": {},
            "span_value_gate_results": [],
            "qualifier_retention_report": qualifier_retention_report([]),
            "retirement": {
                "retire_if_repeated_factual_canary_failure": True,
                "retired_by_current_result": False,
                "reopen_requires": "another diagnosed cause changes",
                "parameter_sweep_authorized": False,
            },
            "field_principles": deepcopy(FIELD_PRINCIPLES),
        }
    )
    value["sample_size_budget"].update(
        {
            "planned": MAX_GENERATION_CALLS,
            "evaluation_planned": EVALUATION_CALLS,
            "development_planned": DEVELOPMENT_CALLS,
            "independent_paired_evaluation_units": engine.SEALED_EVALUATION_UNITS,
            "maximum_generation_calls": MAX_GENERATION_CALLS,
            "stop_rule": "twelve fixed development calls; require three of four factual and two of two empty controls per arm; then preserve all 96 evaluation dispositions without retry",
        }
    )
    return value


def build_blocked_artifact(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build a V654 block with planned model identity but no invented work."""

    with factual_span_contract():
        return _ENGINE_BUILD_BLOCKED_ARTIFACT(checks)


def development_terminal_outcome(gate: Mapping[str, Any]) -> JsonDict:
    """Name the two closed-gate causes without hiding a successful open gate."""

    if gate.get("capture_open") is True:
        return {}
    if gate.get("repeated_factual_canary_failure_retires_construction") is True:
        verdict = "complete_null_factual_span_canary_development_gate_closed"
    else:
        verdict = "complete_null_factual_span_empty_control_gate_closed"
    return {
        "honest_verdict": verdict,
        "status": "complete_development_gate_closed",
        "verdict_class": "null",
    }


def measured_artifact(**kwargs: Any) -> JsonDict:
    """Add factual gate, reply exposure, and CI gates to shared measured evidence."""

    value = _PREDECESSOR_MEASURED_ARTIFACT(**kwargs)
    capture = dict(kwargs["capture"])
    reduced = dict(kwargs["reduced"])
    context = dict(kwargs["context"])
    gate = reduce_development_gate(capture.get("development_rows") or [])
    attempted_rows = [
        row
        for row in [
            *list(capture.get("development_rows") or []),
            *list(capture.get("rows") or []),
        ]
        if row.get("attempted") is True
    ]
    references = list(capture.get("response_shards") or [])
    reply_shards = build_raw_reply_shards(attempted_rows, references) if references else []
    span_gates = deepcopy(list(reduced.get("span_value_gate_results") or []))
    existing_checks = {str(row.get("check")) for row in value["acceptance_gate_results"]}
    value["acceptance_gate_results"].extend(
        {
            **row,
            "upstream": EXPERIMENT_ID,
            "path": RESULT_PATH.as_posix(),
            "field": row["check"],
        }
        for row in span_gates
        if row["check"] not in existing_checks
    )
    retired = gate.get("repeated_factual_canary_failure_retires_construction") is True
    value.update(development_terminal_outcome(gate))
    value.update(
        {
            "factual_development_gate": deepcopy(gate),
            "development_gate": deepcopy(gate),
            "paragraph_selection": deepcopy(
                context.get("paragraph_selection") or select_development_panel(REPO_ROOT)
            ),
            "raw_reply_shards": reply_shards,
            "paired_output_token_ci95": deepcopy(reduced.get("paired_output_token_ci95") or {}),
            "span_value_gate_results": span_gates,
            "qualifier_retention_report": deepcopy(
                reduced.get("qualifier_retention_report")
                or qualifier_retention_report(value.get("semantic_pair_rows") or [])
            ),
            "semantic_scope": deepcopy(SEMANTIC_SCOPE),
            "retirement": {
                "retire_if_repeated_factual_canary_failure": True,
                "retired_by_current_result": retired,
                "reopen_requires": "another diagnosed cause changes",
                "parameter_sweep_authorized": False,
            },
            "field_principles": deepcopy(FIELD_PRINCIPLES),
        }
    )
    value["sample_size_budget"].update(
        {
            "planned": MAX_GENERATION_CALLS,
            "evaluation_planned": EVALUATION_CALLS,
            "development_planned": DEVELOPMENT_CALLS,
            "independent_paired_evaluation_units": engine.SEALED_EVALUATION_UNITS,
            "maximum_generation_calls": MAX_GENERATION_CALLS,
            "stop_rule": "twelve fixed development calls; require three of four factual and two of two empty controls per arm; then preserve all 96 evaluation dispositions without retry",
        }
    )
    value["gate_check_summary"] = engine._gate_summary(value["acceptance_gate_results"])
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Build fresh V654 replay, independent reduction, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7467_v654_factual_span_canary import independent_reduce_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=independent_reduce_artifact(v,root=pathlib.Path.cwd(),require_terminal=False);"
        "print(json.dumps({'errors':e},sort_keys=True),flush=True);raise SystemExit(bool(e))"
    )
    return [
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "measured_candidate",
        ),
        CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_candidate",
        ),
    ]


@contextmanager
def factual_span_contract() -> Iterator[None]:
    """Apply V654 behavior to reused modules and restore every changed global."""

    engine_values = {
        "MILESTONE": MILESTONE,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "TASK_ID": TASK_ID,
        "SCHEMA": SCHEMA,
        "RESULT_PATH": RESULT_PATH,
        "RAW_DIR": RAW_DIR,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "SPEC_PATH": SPEC_PATH,
        "MODEL_SPECS": MODEL_SPECS,
        "INFERENCE_SUBSTRATE": INFERENCE_SUBSTRATE,
        "INFERENCE_SUBSTRATE_CLASS": INFERENCE_SUBSTRATE_CLASS,
        "EXECUTION_VENUE": EXECUTION_VENUE,
        "RANDOM_SEED": RANDOM_SEED,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "DEVELOPMENT_PARAGRAPHS": DEVELOPMENT_PARAGRAPHS,
        "DEVELOPMENT_CALLS": DEVELOPMENT_CALLS,
        "DEVELOPMENT_USABLE_MINIMUM": FACTUAL_USABLE_MINIMUM,
        "MAX_GENERATION_CALLS": MAX_GENERATION_CALLS,
        "VALIDATION_MANIFEST": VALIDATION_MANIFEST,
        "FIELD_PRINCIPLES": FIELD_PRINCIPLES,
        "REQUIRED_FIELDS": REQUIRED_FIELDS,
        "progress": progress,
        "build_development_schedule": build_development_schedule,
        "build_capture_row": build_capture_row,
        "reduce_development_gate": reduce_development_gate,
        "reduce_evaluation": reduce_evaluation,
        "_base_artifact": base_artifact,
        "_measured_artifact": measured_artifact,
        "collect_preconditions": collect_preconditions,
        "_terminal_commands": terminal_commands,
        "validate_artifact": validate_artifact,
        "independent_reduce_artifact": independent_reduce_artifact,
        "_ShardRecorder": FactualShardRecorder,
        "_fixture_response": _fixture_response,
    }
    predecessor_values = {
        "MILESTONE": MILESTONE,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "TASK_ID": TASK_ID,
        "SCHEMA": SCHEMA,
        "RESULT_PATH": RESULT_PATH,
        "RAW_DIR": RAW_DIR,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "SPEC_PATH": SPEC_PATH,
        "MODEL_SPECS": MODEL_SPECS,
        "INFERENCE_SUBSTRATE": INFERENCE_SUBSTRATE,
        "INFERENCE_SUBSTRATE_CLASS": INFERENCE_SUBSTRATE_CLASS,
        "EXECUTION_VENUE": EXECUTION_VENUE,
        "RANDOM_SEED": RANDOM_SEED,
        "VALIDATION_MANIFEST": VALIDATION_MANIFEST,
        "FIELD_PRINCIPLES": FIELD_PRINCIPLES,
        "REQUIRED_FIELDS": REQUIRED_FIELDS,
        "SEMANTIC_SCOPE": SEMANTIC_SCOPE,
        "progress": progress,
        "build_capture_row": build_capture_row,
        "reduce_development_gate": reduce_development_gate,
        "base_artifact": base_artifact,
        "measured_artifact": measured_artifact,
        "collect_preconditions": collect_preconditions,
        "terminal_commands": terminal_commands,
        "validate_artifact": validate_artifact,
        "independent_reduce_artifact": independent_reduce_artifact,
    }
    engine_previous = {name: getattr(engine, name) for name in engine_values}
    predecessor_previous = {name: getattr(predecessor, name) for name in predecessor_values}
    try:
        for name, value in engine_values.items():
            setattr(engine, name, value)
        for name, value in predecessor_values.items():
            setattr(predecessor, name, value)
        yield
    finally:
        for name, value in predecessor_previous.items():
            setattr(predecessor, name, value)
        for name, value in engine_previous.items():
            setattr(engine, name, value)


def build_fixture_artifact() -> JsonDict:
    """Build a deterministic V654 artifact without loading a model."""

    with factual_span_contract():
        return _ENGINE_BUILD_FIXTURE_ARTIFACT()


def validate_artifact(
    value: object, *, require_terminal: bool, root: Path = REPO_ROOT
) -> list[str]:
    """Cold-check shared evidence plus V654 selection, gates, and confidence bounds."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    normalized = deepcopy(dict(value))
    normalized_budget = deepcopy(dict(normalized.get("sample_size_budget") or {}))
    normalized_budget["planned"] = EVALUATION_CALLS
    normalized["sample_size_budget"] = normalized_budget
    normalized["reproducibility_checksum"] = artifact_checksum(normalized)
    with factual_span_contract():
        errors = _PREDECESSOR_VALIDATE_ARTIFACT(
            normalized, require_terminal=require_terminal, root=root
        )
    blocked = value.get("verdict_class") == "blocked"
    if blocked:
        return list(dict.fromkeys(errors))
    development = list(value.get("development_rows") or [])
    expected_gate = reduce_development_gate(development) if development else {}
    if value.get("factual_development_gate") != expected_gate:
        errors.append("factual_development_gate_mismatch")
    if value.get("development_gate") != expected_gate:
        errors.append("development_gate_mismatch")
    try:
        expected_selection = select_development_panel(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"paragraph_selection_replay_failed:{type(exc).__name__}:{exc}")
    else:
        if value.get("paragraph_selection") != expected_selection:
            errors.append("paragraph_selection_mismatch")
    rows = list(value.get("extraction_rows") or [])
    if len(rows) == EVALUATION_CALLS:
        try:
            reduced = reduce_evaluation(rows, engine.protocol.parser_control_rows())
        except (TypeError, ValueError) as exc:
            errors.append(f"v654_reduction_failed:{type(exc).__name__}:{exc}")
        else:
            for field in (
                "paired_output_token_ci95",
                "span_value_gate_results",
                "qualifier_retention_report",
                "span_value_score",
            ):
                if value.get(field) != reduced.get(field):
                    errors.append(f"{field}_mismatch")
    references = list(
        dict(value.get("source_artifact_hashes") or {}).get("current_response_shards") or []
    )
    if references:
        attempted = [
            row
            for row in [*development, *rows]
            if isinstance(row, Mapping) and row.get("attempted") is True
        ]
        try:
            expected_replies = build_raw_reply_shards(attempted, references)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if value.get("raw_reply_shards") != expected_replies:
                errors.append("raw_reply_shards_mismatch")
    budget = dict(value.get("sample_size_budget") or {})
    if budget.get("planned") != MAX_GENERATION_CALLS:
        errors.append("sample_size_budget_planned_mismatch")
    if budget.get("development_planned") != DEVELOPMENT_CALLS:
        errors.append("sample_size_budget_development_mismatch")
    if value.get("semantic_scope") != SEMANTIC_SCOPE:
        errors.append("semantic_scope_mismatch")
    if set(value.get("field_principles") or {}) != REQUIRED_FIELDS:
        errors.append("field_principles_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def independent_reduce_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Recompute the V654 artifact without trusting its summary fields."""

    return validate_artifact(value, root=root, require_terminal=require_terminal)


def date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V654 protocol."""

    if value != RUN_DATE:
        raise ValueError(f"date must be {RUN_DATE}")
    return value


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - bounded live capability E2E.
    """Run the qualified shared engine once under the isolated V654 contract."""

    with factual_span_contract():
        return _ENGINE_RUN_EXPERIMENT(root=root, run_date=run_date, output_path=output_path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run live capture or cold-replay one unpublished terminal candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "entrypoint", "start", date=args.date)
    if args.validate is not None:
        value = load_object(args.validate)
        names = {
            row.get("name")
            for row in value.get("validation_receipts") or []
            if isinstance(row, Mapping)
        }
        errors = independent_reduce_artifact(
            value,
            root=REPO_ROOT,
            require_terminal=set(TERMINAL_CHECK_NAMES).issubset(names),
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "honest_verdict": result["honest_verdict"],
                "span_capture_complete_score": result["span_capture_complete_score"],
                "span_value_score": result["span_value_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
