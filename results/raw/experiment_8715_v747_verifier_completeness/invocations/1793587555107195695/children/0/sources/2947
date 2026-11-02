"""Seal a lossless claim-span extraction protocol without loading a model.

The protocol changes representation after archived relation triples exhausted a
64-token development budget. It keeps that transport failure separate from the
archived producer's missing passing safety receipt.

Spec refs: REQ-VERIFY-7437 and SCENARIO-VERIFY-7437-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
    write_immutable_sidecar,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "exp7437-v652-span-protocol"
SCHEMA = "carnot.exp7437.v652.span_protocol.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

RESULT_PATH = Path("results/experiment_7437_v652_span_protocol.json")
RAW_DIR = Path("results/raw/experiment_7437_v652_span_protocol")
MODULE_PATH = Path("python/carnot/experiment_7437_v652_span_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7437_v652_span_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7437_v652_span_protocol.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
ARCHIVE_PATH = Path("results/experiment_7429_v651_anchored_capture.json")
RAGTRUTH_SOURCE_PATH = Path("data/ragtruth/source_info.jsonl")
RAGTRUTH_RESPONSE_PATH = Path("data/ragtruth/response.jsonl")

EXPECTED_SOURCE_SHA256 = "sha256:0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b"
EXPECTED_RESPONSE_SHA256 = "sha256:e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073"
EXPECTED_ARCHIVE_SHA256 = "sha256:9b98d59057a19eb36317c9b6340882b44ac2063dd86852b2ffc98ac3a657a65a"
RANDOM_SEED = 6_527_437
MAX_PARAGRAPH_CHARS = 512
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
DEVELOPMENT_GROUPS = 4
EVALUATION_GROUPS = 24
ARMS = ("span", "verbatim")
INFERENCE_SUBSTRATE = "host_cpu_protocol_reduction_no_model_load"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"

AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7416_v650_anchored_extraction.py"),
    Path("python/carnot/experiment_7422_v651_runtime_ownership.py"),
    Path("python/carnot/experiment_7423_v651_annotated_protocol.py"),
    Path("python/carnot/experiment_7429_v651_anchored_capture.py"),
    Path("python/carnot/inference/sota_models.py"),
    SPEC_PATH,
    ARCHIVE_PATH,
    RAGTRUTH_SOURCE_PATH,
    RAGTRUTH_RESPONSE_PATH,
)

FIELD_PRINCIPLES = {
    "schema": "Use a versioned plain top-level schema with experiment identity, milestone, and terminal status.",
    "run_date": "Use 20260920 and record actual UTC and monotonic boundaries.",
    "preconditions_checked": "Name each observed resource, path, identity, and prerequisite before dependent work.",
    "MODEL_SPECS": "Use an empty list because this experiment invokes no current LLM.",
    "model_invoked": "Keep current attempted model work separate from archived and scripted evidence.",
    "invocation_counts": "Reconcile every current load and generation disposition from owned events.",
    "inference_substrate": "Use a truthful string and keep device facts in the detail object.",
    "inference_substrate_class": "Declare no_model_load because no current model operation is attempted.",
    "execution_venue": "Use host and record CPU, CUDA, and external-device identity separately.",
    "duration_s": "Measure current protocol, computation, cold replay, and validation work without padding.",
    "phase_spans": "Bind phase time, progress boundaries, completed units, and checkpoints.",
    "random_seed": "Freeze label-blind panel selection and arm ordering before evaluation.",
    "reproducibility_checksum": "Bind code, protocol, inputs, sidecars, rows, and exact validation scope.",
    "source_artifact_hashes": "Preserve source identities and original flags through byte hashes.",
    "rows": "Keep per-unit arm, group, condition, seed, metrics, and unstarted disposition.",
    "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
    "acceptance_gate_results": "Keep check, category, operator, operands, result, and principle separate.",
    "gate_check_summary": "Name the exact first failed upstream, path, field, expected value, and observation.",
    "verifier_is_oracle": "Mark exact constructed checks as oracle-defined and do not call them semantic benefit.",
    "honest_verdict": "Use complete_ for finished work and blocked_ only for unavailable external inputs.",
    "verdict_class": "Use the closed positive, circular_positive, null, blocked, disqualified, or partial class.",
    "flagged_adversarial": "Preserve critical findings because flagged evidence cannot supply readiness.",
    "validation_receipts": "Record scoped argv, environment, exits, durations, and hashed logs, including adversarial verification.",
    "field_principles": "Explain field intent separately while gate scalars remain plain values.",
    "promotion_score": "Remain zero because this milestone authorizes no rollout, publication, or weight update.",
    "span_protocol_ready_score": "Require a lossless parser, sealed panel, provenance, and complete validation contract.",
    "archive_diagnosis_rows": "Preserve archived truncation and missing-reader causes as separate diagnoses.",
    "span_protocol_manifest": "Hash immutable inputs, arms, budget, unit counts, and semantic endpoints.",
    "parser_control_rows": "Expose malformed, ambiguous, Unicode, empty, modifier, and partial-output behavior.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)
_FENCE = re.compile(r"^\s*```", re.MULTILINE)
_PARAGRAPH_BREAK = re.compile(r"\r?\n[ \t]*\r?\n")
_EVALUATOR_FIELDS = frozenset(
    {"labels", "annotations", "quality", "model", "temperature", "primary_label"}
)


def utc_now() -> str:  # pragma: no cover - real wall-clock boundary.
    """Return one real UTC boundary while elapsed work uses a monotonic clock."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(
    started: float, phase: str, event: str, *, completed_units: int = 0, **details: Any
) -> None:  # pragma: no cover - required live progress.
    """Flush one truthful boundary or heartbeat for a potentially long phase."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7437] phase={phase} event={event} "
        f"completed_units={completed_units} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash the complete artifact except for its self-referential checksum field."""

    copied = deepcopy(dict(value))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def _sha256_text(value: str) -> str:
    """Hash exact UTF-8 text so Unicode character offsets retain a byte identity."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object when external bytes are unusable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _base_parse(disposition: str = "malformed_json") -> JsonDict:
    """Create the closed parser result without implying a repair or retry."""

    return {
        "disposition": disposition,
        "syntax_valid": False,
        "parse_valid": False,
        "claims": [],
        "claim_spans": [],
        "literal_span_reconstruction": False,
        "qualifier_retention": None,
        "missing_modifiers": [],
        "retry_count": 0,
        "repair_attempted": False,
    }


def parse_claim_output(
    raw_reply: str,
    *,
    arm: str,
    paragraph: str,
    required_modifiers: Sequence[str] = (),
) -> JsonDict:
    """Parse one fixed schema and never repair missing bytes or modifiers."""

    if arm not in ARMS:
        raise ValueError(f"unsupported_arm:{arm}")
    result = _base_parse()
    if _FENCE.search(raw_reply):
        result["disposition"] = "markdown_fence_forbidden"
        return result
    try:
        decoded = json.loads(raw_reply)
    except json.JSONDecodeError:
        stripped = raw_reply.strip()
        result["disposition"] = (
            "partial_json_object"
            if stripped.startswith("{") and not stripped.endswith("}")
            else "malformed_json"
        )
        return result
    result["syntax_valid"] = True
    if not isinstance(decoded, dict) or set(decoded) != {"claims"}:
        result["disposition"] = "top_level_schema"
        return result
    claims_value = decoded["claims"]
    if not isinstance(claims_value, list):
        result["disposition"] = "claims_not_list"
        return result
    if not claims_value:
        result.update(
            {
                "disposition": "completed_valid_empty",
                "parse_valid": True,
                "literal_span_reconstruction": True,
                "qualifier_retention": not required_modifiers,
                "missing_modifiers": list(required_modifiers),
            }
        )
        return result

    claims: list[str] = []
    spans: list[list[int]] = []
    if arm == "span":
        for span in claims_value:
            if (
                not isinstance(span, list)
                or len(span) != 2
                or not all(isinstance(item, int) and not isinstance(item, bool) for item in span)
            ):
                result["disposition"] = "span_type"
                return result
            start, end = span
            if not 0 <= start < end <= len(paragraph):
                result["disposition"] = "span_bounds"
                return result
            spans.append([start, end])
            claims.append(paragraph[start:end])
    else:
        for claim in claims_value:
            if not isinstance(claim, str):
                result["disposition"] = "verbatim_claim_type"
                return result
            if not claim:
                result["disposition"] = "verbatim_claim_empty"
                return result
            occurrences = paragraph.count(claim)
            if occurrences == 0:
                result["disposition"] = "claim_not_found"
                return result
            if occurrences > 1:
                result["disposition"] = "duplicate_text_ambiguous"
                return result
            start = paragraph.find(claim)
            spans.append([start, start + len(claim)])
            claims.append(claim)

    missing = [
        modifier for modifier in required_modifiers if not any(modifier in x for x in claims)
    ]
    reconstructed = [paragraph[start:end] for start, end in spans]
    result.update(
        {
            "disposition": "completed_valid",
            "parse_valid": True,
            "claims": claims,
            "claim_spans": spans,
            "literal_span_reconstruction": reconstructed == claims,
            "qualifier_retention": not missing,
            "missing_modifiers": missing,
        }
    )
    return result


def seal_response_paragraph(response: str) -> JsonDict:
    """Select one exact response substring and mark all incomplete-response cases."""

    if not isinstance(response, str) or not response:
        raise ValueError("response_text_required")
    start = 0
    segment = response
    for part in _PARAGRAPH_BREAK.split(response):
        position = response.find(part, start)
        if part.strip():
            start = position
            segment = part
            break
        start = position + len(part)
    clipped = len(segment) > MAX_PARAGRAPH_CHARS
    paragraph = segment[:MAX_PARAGRAPH_CHARS]
    end = start + len(paragraph)
    return {
        "paragraph": paragraph,
        "paragraph_start": start,
        "paragraph_end": end,
        "paragraph_sha256": _sha256_text(paragraph),
        "response_sha256": _sha256_text(response),
        "response_char_count": len(response),
        "clipped": clipped,
        "complete_response_coverage_eligible": start == 0 and end == len(response) and not clipped,
    }


def _rank(scope: str, group_id: str, response_id: str = "") -> str:
    """Rank only stable predictor identity fields with the frozen selection seed."""

    return canonical_hash(
        {"seed": RANDOM_SEED, "scope": scope, "group_id": group_id, "response_id": response_id}
    )


def _select_partition(
    rows: Sequence[Mapping[str, Any]], *, split: str, count: int, scope: str
) -> list[JsonDict]:
    """Select one response per source group without reading evaluator fields."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == split:
            grouped[str(row["group_id"])].append(row)
    representatives: list[JsonDict] = []
    for group_id, siblings in grouped.items():
        winner = min(
            siblings,
            key=lambda row: _rank(scope, group_id, str(row["response_id"])),
        )
        representatives.append(deepcopy(dict(winner)))
    representatives.sort(key=lambda row: _rank(scope, str(row["group_id"])))
    if len(representatives) < count:
        raise ValueError(f"insufficient_distinct_groups:{scope}:{len(representatives)}<{count}")
    selected: list[JsonDict] = []
    for row in representatives[:count]:
        sealed = seal_response_paragraph(str(row["response_text"]))
        selected.append(
            {
                "group_id": str(row["group_id"]),
                "response_id": str(row["response_id"]),
                "split": split,
                "task_type": str(row.get("task_type") or ""),
                "source_text": str(row.get("source_text") or ""),
                "source_sha256": _sha256_text(str(row.get("source_text") or "")),
                "response_text": str(row["response_text"]),
                **sealed,
            }
        )
    return selected


def seal_panel(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Seal four development and 24 evaluation paragraphs before labels are joined."""

    for row in rows:
        forbidden = sorted(_EVALUATOR_FIELDS.intersection(row))
        if forbidden:
            raise ValueError(f"predictor_contains_evaluator_field:{forbidden[0]}")
        required = {"group_id", "response_id", "split", "source_text", "response_text"}
        if not required.issubset(row):
            raise ValueError("predictor_fields_missing")
    development = _select_partition(
        rows, split="train", count=DEVELOPMENT_GROUPS, scope="development"
    )
    evaluation = _select_partition(rows, split="test", count=EVALUATION_GROUPS, scope="evaluation")
    if {row["group_id"] for row in development}.intersection(row["group_id"] for row in evaluation):
        raise ValueError("development_evaluation_group_overlap")
    return {
        "schema": "carnot.exp7437.panel.v1",
        "selection_seed": RANDOM_SEED,
        "selection_fields": ["group_id", "response_id", "split"],
        "denied_selection_fields": sorted(_EVALUATOR_FIELDS),
        "development": development,
        "evaluation": evaluation,
    }


def _prompt(arm: str, paragraph: str) -> str:
    """Render one claim-only prompt without source labels or implied facts."""

    common = (
        "Extract whole factual propositions from RESPONSE. Include every explicit modifier "
        "inside each selected proposition. Do not create triples, implied arguments, or new "
        'facts. Return one JSON object and no markdown. Use {"claims":[]} when there are '
        "no factual propositions.\nRESPONSE:\n" + paragraph + "\n"
    )
    if arm == "span":
        return common + (
            'Schema: {"claims":[[start,end]]}. Offsets are zero-based half-open Unicode '
            "character offsets into RESPONSE."
        )
    if arm == "verbatim":
        return common + '{"claims":["exact RESPONSE substring"]}. Copy each claim exactly once.'
    raise ValueError(f"unsupported_arm:{arm}")


def build_evaluation_schedule(evaluation: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build 48 paired future calls with one fixed budget and no retries."""

    if len(evaluation) != EVALUATION_GROUPS:
        raise ValueError("evaluation_group_count")
    schedule: list[JsonDict] = []
    for case_index, value in enumerate(evaluation):
        row = dict(value)
        paragraph = str(row["paragraph"])
        first = "span" if int(_rank("arm", str(row["group_id"]))[-1], 16) % 2 == 0 else "verbatim"
        ordered_arms = (first, "verbatim" if first == "span" else "span")
        for arm_order, arm in enumerate(ordered_arms):
            prompt = _prompt(arm, paragraph)
            schedule.append(
                {
                    "unit_id": f"eval-{case_index:02d}-{arm}",
                    "case_index": case_index,
                    "group_id": row["group_id"],
                    "response_id": row["response_id"],
                    "condition": "ragtruth_unchanged_response",
                    "arm": arm,
                    "arm_order": arm_order,
                    "seed": RANDOM_SEED,
                    "paragraph": paragraph,
                    "paragraph_sha256": row["paragraph_sha256"],
                    "clipped": row["clipped"],
                    "complete_response_coverage_eligible": row[
                        "complete_response_coverage_eligible"
                    ],
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": TEMPERATURE,
                    "generation_count": 1,
                    "grammar_mask": False,
                    "parser_retry_count": 0,
                    "prompt": prompt,
                    "prompt_sha256": _sha256_text(prompt),
                }
            )
    return schedule


def _qualifier_rows() -> tuple[tuple[str, str, str, str], ...]:
    """Keep exact constructed contrasts small, visible, and independent."""

    return (
        ("negation", "The trial met its endpoint.", "The trial did not meet its endpoint.", "not"),
        (
            "time",
            "The permit is valid.",
            "The permit is valid only until June 2027.",
            "only until June 2027",
        ),
        (
            "count",
            "The panel approved cases.",
            "The panel approved exactly three cases.",
            "exactly three",
        ),
        ("unit", "The dose increased.", "The dose increased by 5 mg.", "5 mg"),
        (
            "comparison",
            "Model A was faster.",
            "Model A was 12% faster than Model B.",
            "12% faster than Model B",
        ),
        (
            "condition",
            "The alarm activates.",
            "The alarm activates only when both sensors fail.",
            "only when both sensors fail",
        ),
        (
            "location",
            "The ban applies.",
            "The ban applies only inside the northern district.",
            "only inside the northern district",
        ),
        (
            "frequency",
            "The audit runs.",
            "The audit runs at most twice per year.",
            "at most twice per year",
        ),
        (
            "uncertainty",
            "The comet will return.",
            "The comet will probably return after 2030.",
            "probably",
        ),
        (
            "exception",
            "All records were released.",
            "All records except sealed exhibits were released.",
            "except sealed exhibits",
        ),
        (
            "range",
            "The temperature stayed stable.",
            "The temperature stayed between 18°C and 21°C.",
            "between 18°C and 21°C",
        ),
        (
            "attribution",
            "Revenue increased.",
            "According to the audited filing, revenue increased.",
            "According to the audited filing",
        ),
    )


def constructed_qualifier_pairs() -> list[JsonDict]:
    """Return twelve exact pairs that do not use human corpus annotations."""

    return [
        {
            "pair_id": f"qualifier-{index:02d}-{family}",
            "family": family,
            "base_paragraph": base,
            "qualified_paragraph": qualified,
            "required_modifiers": [modifier],
            "authority": "constructed_exact_string",
        }
        for index, (family, base, qualified, modifier) in enumerate(_qualifier_rows(), 1)
    ]


def reduce_constructed_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Run both parser arms while keeping exact checks outside corpus diagnostics."""

    controls: list[JsonDict] = []
    for pair in pairs:
        paragraph = str(pair["qualified_paragraph"])
        required = tuple(str(item) for item in pair["required_modifiers"])
        for arm in ARMS:
            payload: object = [[0, len(paragraph)]] if arm == "span" else [paragraph]
            parsed = parse_claim_output(
                json.dumps({"claims": payload}, ensure_ascii=False),
                arm=arm,
                paragraph=paragraph,
                required_modifiers=required,
            )
            controls.append(
                {
                    "pair_id": pair["pair_id"],
                    "family": pair["family"],
                    "arm": arm,
                    "scope": "constructed_exact_check",
                    **parsed,
                }
            )
    return controls


def parser_control_rows() -> list[JsonDict]:
    """Expose the required parser boundary dispositions as frozen controls."""

    paragraph = "Café rose by €5 only in 2026. It repeated. It repeated."
    controls = (
        ("unicode", "span", json.dumps({"claims": [[0, 31]]}), ("only in 2026",)),
        ("empty", "span", '{"claims":[]}', ()),
        ("ambiguous", "verbatim", '{"claims":["It repeated."]}', ()),
        ("malformed", "span", "not-json", ()),
        ("partial", "span", '{"claims":[[0,4]]', ()),
        ("bad_bounds", "span", '{"claims":[[0,999]]}', ()),
        ("missing_modifier", "verbatim", '{"claims":["Café rose by €5"]}', ("only in 2026",)),
    )
    return [
        {
            "control": name,
            "arm": arm,
            **parse_claim_output(reply, arm=arm, paragraph=paragraph, required_modifiers=mods),
        }
        for name, arm, reply, mods in controls
    ]


def reduce_archived_capture(artifact: Mapping[str, Any]) -> JsonDict:
    """Independently reduce four raw development replies and the receipt defect."""

    model_specs = artifact.get("model_specs") or artifact.get("MODEL_SPECS") or []
    tokenizer = None
    if model_specs and isinstance(model_specs[0], Mapping):
        tokenizer = model_specs[0].get("native_tokenizer")
    rows = artifact.get("development_rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("archive_development_row_count")
    diagnoses: list[JsonDict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("archive_development_row_shape")
        raw_reply = str(row.get("raw_reply") or "")
        try:
            json.loads(raw_reply)
        except json.JSONDecodeError:
            parse_status = (
                "invalid_truncated_json"
                if row.get("finish_reason") == "length"
                else "invalid_malformed_json"
            )
        else:
            parse_status = "valid_json"
        request = row.get("raw_request") if isinstance(row.get("raw_request"), Mapping) else {}
        response = row.get("raw_response") if isinstance(row.get("raw_response"), Mapping) else {}
        diagnoses.append(
            {
                "call_id": row.get("call_id"),
                "finish_reason": row.get("finish_reason"),
                "actual_token_ceiling": request.get("max_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "tokenizer_identity": tokenizer,
                "parse_status": parse_status,
                "raw_request_sha256": row.get("raw_request_sha256") or canonical_hash(request),
                "raw_response_sha256": row.get("raw_response_sha256") or canonical_hash(response),
                "raw_reply_sha256": row.get("raw_reply_sha256") or _sha256_text(raw_reply),
            }
        )
    adversarial = [
        row
        for row in artifact.get("validation_receipts") or []
        if isinstance(row, Mapping) and row.get("name") == "adversarial_verify"
    ]
    passing = [
        row for row in adversarial if row.get("passed") is True and row.get("exit_code") == 0
    ]
    return {
        "archive_diagnosis_rows": diagnoses,
        "producer_validation_diagnosis": {
            "missing_receipt": "adversarial_verify" if not passing else None,
            "observed_named_receipt_count": len(adversarial),
            "observed_passing_receipt_count": len(passing),
            "cause": "missing_passing_terminal_receipt" if not passing else None,
            "separate_from_output_truncation": True,
        },
    }


def validate_terminal_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require exactly one passing receipt for every fixed terminal command."""

    errors: list[str] = []
    for name in TERMINAL_CHECK_NAMES:
        matching = [row for row in receipts if row.get("name") == name]
        if not matching:
            errors.append(f"missing_terminal_receipt:{name}")
        elif len(matching) > 1:
            errors.append(f"duplicate_terminal_receipt:{name}")
        elif not (
            matching[0].get("passed") is True
            and matching[0].get("exit_code") == 0
            and matching[0].get("timed_out") is not True
        ):
            errors.append(f"failed_terminal_receipt:{name}")
    return errors


def _affected_receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing scoped receipt for each Exp7303 affected check."""

    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
            for row in receipts
        )
        == 1
        for name in AFFECTED_CHECK_NAMES
    )


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    upstream: str = EXPERIMENT_ID,
    path: str = RESULT_PATH.as_posix(),
    field: str | None = None,
) -> JsonDict:
    """Keep a gate's operands and reason explicit for independent reduction."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
        "principle": principle,
        "upstream": upstream,
        "path": path,
        "field": field or check,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose the first exact failure without hiding later failed gates."""

    failed = [dict(row) for row in gates if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [row["check"] for row in failed],
        "upstream": first["upstream"] if first else EXPERIMENT_ID,
        "path": first["path"] if first else RESULT_PATH.as_posix(),
        "check": first["check"] if first else "all_required_checks",
        "field": first["field"] if first else "gate_check_summary",
        "operator": first["operator"] if first else "==",
        "expected": deepcopy(first["expected"]) if first else True,
        "observed": deepcopy(first["observed"]) if first else True,
        "passed": not failed,
    }


def _row_from_schedule(row: Mapping[str, Any]) -> JsonDict:
    """Project one future call into an accountable unstarted metric row."""

    return {
        "unit_id": row["unit_id"],
        "case_index": row["case_index"],
        "group_id": row["group_id"],
        "response_id": row["response_id"],
        "condition": row["condition"],
        "arm": row["arm"],
        "arm_order": row["arm_order"],
        "seed": row["seed"],
        "paragraph_sha256": row["paragraph_sha256"],
        "clipped": row["clipped"],
        "complete_response_coverage_eligible": row["complete_response_coverage_eligible"],
        "max_new_tokens": row["max_new_tokens"],
        "temperature": row["temperature"],
        "generation_count": row["generation_count"],
        "status": "unstarted",
        "completed_valid_output": None,
        "whole_proposition_coverage": None,
        "qualifier_retention": None,
        "literal_span_reconstruction": None,
        "prompt_tokens": None,
        "output_tokens": None,
        "latency_s": None,
    }


def _reference(path: Path, *, root: Path) -> JsonDict:
    """Bind one immutable sidecar using a stable in-repository path when possible."""

    resolved = path.resolve()
    try:
        label = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _resolve_reference(root: Path, reference: Mapping[str, Any]) -> Path:
    """Resolve one bound path without changing the path recorded in the artifact."""

    path = Path(str(reference.get("path") or ""))
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value: Mapping[str, Any]) -> JsonDict:
    """Write deterministic JSON and return a byte reference for later replay."""

    atomic_json(path, value)
    return dict(value)


def _protocol_manifest(
    panel_ref: Mapping[str, Any],
    schedule_ref: Mapping[str, Any],
    evaluator_ref: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Freeze input identities, arm settings, unit counts, and endpoint semantics."""

    manifest: JsonDict = {
        "schema": "carnot.exp7437.span_protocol_manifest.v1",
        "panel": deepcopy(dict(panel_ref)),
        "schedule": deepcopy(dict(schedule_ref)),
        "evaluator": deepcopy(dict(evaluator_ref)),
        "development_paragraph_count": DEVELOPMENT_GROUPS,
        "evaluation_paragraph_count": EVALUATION_GROUPS,
        "evaluation_unit_count": EVALUATION_GROUPS * len(ARMS),
        "arms": list(ARMS),
        "max_paragraph_chars": MAX_PARAGRAPH_CHARS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "generations_per_unit": 1,
        "grammar_mask": False,
        "parser_retry_count": 0,
        "schedule_identity": canonical_hash(schedule),
        "endpoints": [
            "completed_valid_output",
            "whole_proposition_coverage",
            "qualifier_retention",
            "literal_span_reconstruction",
            "prompt_tokens",
            "output_tokens",
            "latency_s",
        ],
        "human_annotation_scope": "source_support_of_unchanged_response_only",
        "extraction_semantic_claim": "precursor_not_entailment_verifier",
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    return manifest


def _build_artifact(
    *,
    root: Path,
    started_ns: int,
    ended_ns: int,
    started_at: str,
    completed_at: str,
    phase_spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    receipt_sidecars: Sequence[Mapping[str, Any]],
    protocol_manifest: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    archive_reduction: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    terminal_control_rows: Sequence[Mapping[str, Any]],
    fixture_artifact: bool,
) -> JsonDict:
    """Build one ordinary-field artifact from immutable raw protocol evidence."""

    current = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}:{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "work": "host CPU hashing, JSON parsing, and protocol sealing",
            "cpu_used": True,
            "cuda_device_used": None,
            "external_device_used": None,
            "current_model_loaded": False,
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        sidecar_references=receipt_sidecars,
        phase_spans=phase_spans,
        small_ebm_training={"performed": False, "receipts": []},
    )
    current["monotonic_start_timestamp_ns"] = current.pop("started_monotonic_ns")
    current["monotonic_end_timestamp_ns"] = current.pop("ended_monotonic_ns")
    affected_ok = _affected_receipts_pass(validation_receipts)
    terminal_errors = validate_terminal_receipts(validation_receipts)
    terminal_ok = not terminal_errors
    controls_ok = all(
        row.get("parse_valid") is True and row.get("qualifier_retention") is True
        for row in controls
        if row.get("scope") == "constructed_exact_check"
    )
    parser_boundaries_ok = {
        row.get("control"): row.get("disposition") for row in controls if "control" in row
    } == {
        "unicode": "completed_valid",
        "empty": "completed_valid_empty",
        "ambiguous": "duplicate_text_ambiguous",
        "malformed": "malformed_json",
        "partial": "partial_json_object",
        "bad_bounds": "span_bounds",
        "missing_modifier": "completed_valid",
    }
    archive_rows = archive_reduction.get("archive_diagnosis_rows") or []
    archive_ok = len(archive_rows) == 4 and all(
        row.get("finish_reason") == "length"
        and row.get("actual_token_ceiling") == 64
        and row.get("tokenizer_identity") == "embedded_gguf"
        and row.get("parse_status") == "invalid_truncated_json"
        for row in archive_rows
    )
    panel_ok = len(schedule) == 48 and len({row["group_id"] for row in schedule}) == 24
    receipt_controls_ok = len(terminal_control_rows) == 2 and all(
        row.get("passed") is True for row in terminal_control_rows
    )
    ready = int(
        affected_ok
        and terminal_ok
        and controls_ok
        and parser_boundaries_ok
        and archive_ok
        and panel_ok
        and receipt_controls_ok
    )
    gates = [
        _gate(
            "archived_failure_diagnosis",
            "provenance",
            "==",
            True,
            archive_ok,
            archive_ok,
            "Four transport truncations stay separate from the producer receipt defect.",
            upstream=ARCHIVE_PATH.as_posix(),
            path=ARCHIVE_PATH.as_posix(),
        ),
        _gate(
            "lossless_parser_controls",
            "validity",
            "==",
            True,
            controls_ok and parser_boundaries_ok,
            controls_ok and parser_boundaries_ok,
            "Exact reconstruction and fail-closed dispositions are required.",
        ),
        _gate(
            "sealed_distinct_group_panel",
            "validity",
            "==",
            {"groups": 24, "units": 48},
            {"groups": len({row["group_id"] for row in schedule}), "units": len(schedule)},
            panel_ok,
            "Both arms must share 24 label-blind RAGTruth paragraph inputs.",
        ),
        _gate(
            "terminal_receipt_controls",
            "validation",
            "==",
            True,
            receipt_controls_ok,
            receipt_controls_ok,
            "A missing adversarial receipt must fail before the complete set passes.",
        ),
        _gate(
            "affected_validation",
            "validation",
            "==",
            True,
            affected_ok,
            affected_ok,
            "Every frozen affected command must pass in its narrow scope.",
        ),
        _gate(
            "terminal_validation",
            "validation",
            "==",
            True,
            terminal_ok,
            terminal_ok,
            "Fresh replay, independent reduction, adversarial verification, and strict row reading must pass.",
        ),
        _gate(
            "scientific_benefit",
            "benefit",
            "==",
            True,
            None,
            False,
            "No model evaluation ran, so the protocol supplies no extraction or entailment benefit claim.",
        ),
    ]
    rows = [_row_from_schedule(row) for row in schedule]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_span_protocol_ready" if ready else "complete_span_protocol_candidate",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **current,
        "duration_s": (ended_ns - started_ns) / 1_000_000_000,
        "computation_duration_s": sum(
            float(row.get("duration_s") or 0.0)
            for row in phase_spans
            if row.get("phase") in {"archive_reduction", "panel_seal", "parser_controls"}
        ),
        "model_duration_s": 0.0,
        "validation_duration_s": sum(
            float(row.get("duration_s") or 0.0)
            for row in validation_receipts
            if row.get("name") in {*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES}
        ),
        "cold_start_duration_s": sum(
            float(row.get("duration_s") or 0.0)
            for row in validation_receipts
            if row.get("name") in {"declared_entrypoint_cold_replay", "independent_cold_reducer"}
        ),
        "random_seed": {
            "panel_selection": RANDOM_SEED,
            "arm_order": RANDOM_SEED,
            "fitting": None,
            "sampling": None,
            "resampling": None,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": rows,
        "sample_size_budget": {
            "planned": 48,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 48,
            "independent_groups": 24,
            "arms_per_group": 2,
            "development_paragraphs": 4,
            "constructed_qualifier_pairs": 12,
            "maximum_new_tokens_per_call": MAX_NEW_TOKENS,
            "stop_rule": "protocol-only seal; future evaluation makes one generation per paired unit without retry",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": "complete_null_span_protocol_ready_no_model_evaluation",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "span_protocol_ready_score": ready,
        "archive_diagnosis_rows": deepcopy(list(archive_rows)),
        "producer_validation_diagnosis": deepcopy(
            dict(archive_reduction.get("producer_validation_diagnosis") or {})
        ),
        "span_protocol_manifest": deepcopy(dict(protocol_manifest)),
        "parser_control_rows": [deepcopy(dict(row)) for row in controls],
        "terminal_receipt_control_rows": [deepcopy(dict(row)) for row in terminal_control_rows],
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "affected_checks": list(AFFECTED_CHECK_NAMES),
            "terminal_checks": list(TERMINAL_CHECK_NAMES),
        },
        "human_annotation_scope": "source_support_of_unchanged_response_only",
        "human_annotations_certify_extraction": False,
        "constructed_checks_scope": "exact_string_controls_separate_from_corpus_diagnostics",
        "claim_span_is_entailment_verifier": False,
        "fixture_artifact": fixture_artifact,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _read_bound_json(root: Path, reference: Mapping[str, Any]) -> JsonDict:
    """Read one sidecar only when its exact bytes match the sealed reference."""

    path = _resolve_reference(root, reference)
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError(f"sidecar_hash_mismatch:{reference.get('path')}")
    value = _load_object(path)
    if not value:
        raise ValueError(f"sidecar_invalid:{reference.get('path')}")
    return value


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, require_terminal: bool
) -> list[str]:
    """Cold-check identity, raw reductions, current counts, rows, and receipts."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "promotion_score": 0,
        "verifier_is_oracle": True,
        "human_annotations_certify_extraction": False,
        "claim_span_is_entailment_verifier": False,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(
                "current_model_provenance_invalid"
                if field in {"MODEL_SPECS", "model_invoked", "invocation_counts"}
                else f"declaration_mismatch:{field}"
            )
    if set(value.get("field_principles") or {}) != REQUIRED_FIELDS:
        errors.append("field_principles_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    current_view = dict(value)
    current_view["started_monotonic_ns"] = value.get("monotonic_start_timestamp_ns")
    current_view["ended_monotonic_ns"] = value.get("monotonic_end_timestamp_ns")
    errors.extend(validate_current_work_receipt(current_view, root=root))
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if value.get("verdict_class") == "blocked":
        if not str(value.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("blocked_verdict_prefix_invalid")
        if value.get("span_protocol_ready_score") != 0:
            errors.append("blocked_readiness_invalid")
        return list(dict.fromkeys(errors))
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_verdict_prefix_invalid")

    manifest = value.get("span_protocol_manifest")
    schedule: list[JsonDict] = []
    if not isinstance(manifest, Mapping):
        errors.append("span_protocol_manifest_invalid")
    else:
        copied = deepcopy(dict(manifest))
        observed_hash = copied.pop("manifest_hash", None)
        if observed_hash != canonical_hash(copied):
            errors.append("span_protocol_manifest_hash_mismatch")
        try:
            panel = _read_bound_json(root, dict(manifest.get("panel") or {}))
            schedule_value = _read_bound_json(root, dict(manifest.get("schedule") or {}))
            evaluator = _read_bound_json(root, dict(manifest.get("evaluator") or {}))
        except ValueError as exc:
            errors.append(str(exc))
        else:
            schedule_raw = schedule_value.get("rows")
            schedule = [dict(row) for row in schedule_raw] if isinstance(schedule_raw, list) else []
            if len(panel.get("development") or []) != 4 or len(panel.get("evaluation") or []) != 24:
                errors.append("panel_counts_invalid")
            if len({row.get("group_id") for row in panel.get("evaluation") or []}) != 24:
                errors.append("panel_groups_not_distinct")
            if len(schedule) != 48 or manifest.get("schedule_identity") != canonical_hash(schedule):
                errors.append("schedule_identity_mismatch")
            if evaluator.get("annotation_scope") != "unchanged_response_source_support_only":
                errors.append("evaluator_scope_invalid")
            if any("annotation" in str(row.get("prompt") or "").lower() for row in schedule):
                errors.append("evaluator_annotation_prompt_leak")
    if schedule and value.get("rows") != [_row_from_schedule(row) for row in schedule]:
        errors.append("evaluation_rows_mismatch")

    controls = value.get("parser_control_rows") or []
    constructed = [row for row in controls if row.get("scope") == "constructed_exact_check"]
    if len(constructed) != 24 or not all(
        row.get("parse_valid") is True
        and row.get("literal_span_reconstruction") is True
        and row.get("qualifier_retention") is True
        for row in constructed
    ):
        errors.append("constructed_parser_controls_invalid")
    boundary = {row.get("control"): row.get("disposition") for row in controls if "control" in row}
    if boundary != {
        "unicode": "completed_valid",
        "empty": "completed_valid_empty",
        "ambiguous": "duplicate_text_ambiguous",
        "malformed": "malformed_json",
        "partial": "partial_json_object",
        "bad_bounds": "span_bounds",
        "missing_modifier": "completed_valid",
    }:
        errors.append("parser_boundary_controls_invalid")
    archive_rows = value.get("archive_diagnosis_rows") or []
    if len(archive_rows) != 4 or not all(
        row.get("finish_reason") == "length"
        and row.get("actual_token_ceiling") == 64
        and row.get("tokenizer_identity") == "embedded_gguf"
        and row.get("parse_status") == "invalid_truncated_json"
        for row in archive_rows
    ):
        errors.append("archive_diagnosis_invalid")
    if (value.get("producer_validation_diagnosis") or {}).get("cause") != (
        "missing_passing_terminal_receipt"
    ):
        errors.append("archive_producer_diagnosis_invalid")

    receipts = value.get("validation_receipts") or []
    affected_ok = _affected_receipts_pass(receipts)
    terminal_errors = validate_terminal_receipts(receipts) if require_terminal else []
    errors.extend(terminal_errors)
    receipt_controls = value.get("terminal_receipt_control_rows") or []
    receipt_controls_ok = len(receipt_controls) == 2 and all(
        row.get("passed") is True for row in receipt_controls
    )
    expected_ready = int(
        require_terminal
        and affected_ok
        and not terminal_errors
        and not any(error for error in errors if error not in {"reproducibility_checksum_mismatch"})
        and receipt_controls_ok
        and value.get("flagged_adversarial") is False
    )
    if value.get("span_protocol_ready_score") != expected_ready:
        errors.append("span_protocol_ready_score_mismatch")
    return list(dict.fromkeys(errors))


def independent_reduce_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Recompute the protocol from bound sidecars without trusting summary gates."""

    errors = validate_artifact(value, root=root, require_terminal=require_terminal)
    manifest = value.get("span_protocol_manifest")
    if not isinstance(manifest, Mapping):
        return errors
    try:
        schedule_value = _read_bound_json(root, dict(manifest.get("schedule") or {}))
    except ValueError as exc:
        errors.append(str(exc))
    else:
        schedule = schedule_value.get("rows") or []
        if value.get("rows") != [_row_from_schedule(row) for row in schedule]:
            errors.append("evaluation_rows_mismatch")
    return list(dict.fromkeys(errors))


def _source_hashes(root: Path, paths: Sequence[Path]) -> JsonDict:
    """Bind exact source bytes and preserve original artifact flags."""

    hashes: JsonDict = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        original_flag = None
        if relative.suffix == ".json":
            original_flag = _load_object(path).get("flagged_adversarial")
        hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "original_flagged_adversarial": original_flag,
        }
    return hashes


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate all branch-local evidence before labels or archived rows are read."""

    checks: list[JsonDict] = []
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "bytes",
                "operator": "==",
                "expected": "readable_nonempty_bytes",
                "observed": observed,
                "passed": observed == "readable_nonempty_bytes",
            }
        )
    expected_hashes = {
        RAGTRUTH_SOURCE_PATH: EXPECTED_SOURCE_SHA256,
        RAGTRUTH_RESPONSE_PATH: EXPECTED_RESPONSE_SHA256,
        ARCHIVE_PATH: EXPECTED_ARCHIVE_SHA256,
    }
    for relative, expected in expected_hashes.items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "check": f"source_hash:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "sha256",
                "operator": "==",
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        {
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "path": SPEC_PATH.as_posix(),
            "field": "REQ-*",
            "operator": "==",
            "expected": "REQ-VERIFY-7437",
            "observed": "REQ-VERIFY-7437" if "REQ-VERIFY-7437" in spec_text else None,
            "passed": "REQ-VERIFY-7437" in spec_text,
        }
    )
    archive = _load_object(root / ARCHIVE_PATH)
    checks.extend(
        [
            {
                "check": "archive_identity",
                "upstream": ARCHIVE_PATH.as_posix(),
                "path": ARCHIVE_PATH.as_posix(),
                "field": "experiment_id",
                "operator": "==",
                "expected": "exp7429-v651-anchored-capture",
                "observed": archive.get("experiment_id"),
                "passed": archive.get("experiment_id") == "exp7429-v651-anchored-capture",
            },
            {
                "check": "archive_original_flag",
                "upstream": ARCHIVE_PATH.as_posix(),
                "path": ARCHIVE_PATH.as_posix(),
                "field": "flagged_adversarial",
                "operator": "==",
                "expected": True,
                "observed": archive.get("flagged_adversarial"),
                "passed": archive.get("flagged_adversarial") is True,
            },
        ]
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    excluded = "experiment_id: 7437" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        {
            "check": "current_task_not_quarantined",
            "upstream": "ops/exclusion_manifest.yaml",
            "path": "ops/exclusion_manifest.yaml",
            "field": EXPERIMENT_ID,
            "operator": "==",
            "expected": False,
            "observed": excluded,
            "passed": not excluded,
        }
    )
    return checks, _source_hashes(root, INPUT_PATHS)


def _jsonl(path: Path, *, started: float | None = None) -> list[JsonDict]:  # pragma: no cover
    """Read immutable JSONL while reporting completed rows during long decodes."""

    rows: list[JsonDict] = []
    last = time.monotonic()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_row_not_object:{path}:{line_number}")
            rows.append(value)
            now = time.monotonic()
            if started is not None and now - last >= 60.0:
                progress(started, "corpus_decode", "heartbeat", completed_units=len(rows))
                last = now
    return rows


def _serialize_source(value: Any) -> str:
    """Preserve string source text or canonically serialize structured source content."""

    return (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )


def _load_ragtruth(
    root: Path, *, started: float | None = None
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Project label-free predictors before joining evaluator annotations by response ID."""

    sources = _jsonl(root / RAGTRUTH_SOURCE_PATH, started=started)
    responses = _jsonl(root / RAGTRUTH_RESPONSE_PATH, started=started)
    source_by_id = {str(row["source_id"]): row for row in sources}
    predictors: list[JsonDict] = []
    evaluators: JsonDict = {}
    for row in responses:
        source_id = str(row.get("source_id") or "")
        source = source_by_id.get(source_id)
        if source is None or not isinstance(row.get("response"), str) or not row["response"]:
            continue
        response_id = str(row.get("id") or "")
        predictors.append(
            {
                "group_id": source_id,
                "response_id": response_id,
                "split": str(row.get("split") or ""),
                "task_type": str(source.get("task_type") or ""),
                "source_text": _serialize_source(source.get("source_info")),
                "response_text": row["response"],
            }
        )
        evaluators[response_id] = {
            "response_id": response_id,
            "group_id": source_id,
            "labels": deepcopy(row.get("labels") or []),
            "quality": row.get("quality"),
            "model": row.get("model"),
            "temperature": row.get("temperature"),
        }
    return predictors, evaluators


def _evaluator_view(panel: Mapping[str, Any], evaluators: Mapping[str, Any]) -> JsonDict:
    """Join annotations only after selection and keep them out of every model prompt."""

    rows: list[JsonDict] = []
    for selected in panel.get("evaluation") or []:
        response_id = str(selected["response_id"])
        evaluator = evaluators.get(response_id)
        if not isinstance(evaluator, Mapping):
            raise ValueError(f"evaluator_missing:{response_id}")
        labels = deepcopy(list(evaluator.get("labels") or []))
        start = int(selected["paragraph_start"])
        end = int(selected["paragraph_end"])
        rows.append(
            {
                "response_id": response_id,
                "group_id": selected["group_id"],
                "unchanged_response_sha256": selected["response_sha256"],
                "human_annotation_count": len(labels),
                "human_source_support_label": "unsupported_span_present"
                if labels
                else "no_unsupported_span_annotated",
                "paragraph_overlapping_annotation_count": sum(
                    isinstance(label, Mapping)
                    and isinstance(label.get("start"), int)
                    and isinstance(label.get("end"), int)
                    and int(label["start"]) < end
                    and int(label["end"]) > start
                    for label in labels
                ),
                "annotations": labels,
            }
        )
    return {
        "schema": "carnot.exp7437.evaluator_sidecar.v1",
        "annotation_scope": "unchanged_response_source_support_only",
        "certifies_new_extraction": False,
        "rows": rows,
    }


def _receipt_controls(
    base: Mapping[str, Any], *, root: Path, affected: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Exercise a private incomplete candidate before the complete receipt set."""

    synthetic = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in TERMINAL_CHECK_NAMES
    ]
    missing = [row for row in synthetic if row["name"] != "adversarial_verify"]
    missing_errors = validate_terminal_receipts(missing)
    complete_errors = validate_terminal_receipts(synthetic)
    private_path = Path(tempfile.mkdtemp(prefix="exp7437-receipt-control-", dir="/tmp"))
    for name, receipts in (("missing-adversarial", missing), ("complete", synthetic)):
        candidate = deepcopy(dict(base))
        candidate["validation_receipts"] = [*deepcopy(list(affected)), *deepcopy(receipts)]
        candidate["span_protocol_ready_score"] = int(name == "complete")
        candidate["reproducibility_checksum"] = artifact_checksum(candidate)
        atomic_json(private_path / f"{name}.json", candidate)
    return [
        {
            "control": "missing_adversarial_verify",
            "expected_errors": ["missing_terminal_receipt:adversarial_verify"],
            "observed_errors": missing_errors,
            "passed": missing_errors == ["missing_terminal_receipt:adversarial_verify"],
        },
        {
            "control": "complete_terminal_receipts",
            "expected_errors": [],
            "observed_errors": complete_errors,
            "passed": not complete_errors,
        },
    ]


def _span(
    phase: str, phase_started: float, run_started: float, completed_units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    """Close one measured phase with its monotonic boundary and checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "checkpoint": checkpoint,
        "checkpoint_at_utc": utc_now(),
    }


def _terminal_commands(
    root: Path, candidate: Path
) -> list[validation_scope.CommandSpec]:  # pragma: no cover
    """Build fresh replay, independent reduction, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7437_v652_span_protocol import independent_reduce_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=independent_reduce_artifact(v,root=pathlib.Path.cwd(),require_terminal=False);"
        "print(json.dumps({'errors':e},sort_keys=True),flush=True);raise SystemExit(bool(e))"
    )
    return [
        validation_scope.CommandSpec(
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
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "measured_candidate",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "measured_candidate",
        ),
    ]


def _fixture_predictors() -> list[JsonDict]:
    """Create enough label-free groups for deterministic unit-test artifacts."""

    return [
        {
            "group_id": f"source-{index:02d}",
            "response_id": f"response-{index:02d}",
            "split": "train" if index < 6 else "test",
            "task_type": "Summary",
            "source_text": f"Source {index}",
            "response_text": f"Group {index} reported exactly {index + 1} cases before 2026.",
        }
        for index in range(32)
    ]


def build_fixture_artifact(root: Path) -> JsonDict:
    """Build a fully bound no-model artifact without network or child processes."""

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    panel = seal_panel(_fixture_predictors())
    schedule = build_evaluation_schedule(panel["evaluation"])
    evaluator = {
        "schema": "carnot.exp7437.evaluator_sidecar.v1",
        "annotation_scope": "unchanged_response_source_support_only",
        "certifies_new_extraction": False,
        "rows": [],
    }
    panel_path = raw / "panel.json"
    schedule_path = raw / "schedule.json"
    evaluator_path = raw / "evaluator.json"
    _write_json(panel_path, panel)
    _write_json(schedule_path, {"schema": "carnot.exp7437.schedule.v1", "rows": schedule})
    _write_json(evaluator_path, evaluator)
    manifest = _protocol_manifest(
        _reference(panel_path, root=root),
        _reference(schedule_path, root=root),
        _reference(evaluator_path, root=root),
        schedule,
    )
    archive_source = {
        "model_specs": [{"native_tokenizer": "embedded_gguf"}],
        "development_rows": [
            {
                "call_id": f"archived-{index}",
                "finish_reason": "length",
                "completion_tokens": 64,
                "raw_reply": '{"claims":[',
                "raw_request": {"max_tokens": 64},
                "raw_response": {"finish_reason": "length"},
            }
            for index in range(4)
        ],
        "validation_receipts": [{"name": "adversarial_verify", "passed": False, "exit_code": 1}],
    }
    archive = reduce_archived_capture(archive_source)
    historical = write_immutable_sidecar(
        raw / "archive.json",
        scope="historical_model_receipts",
        payload={"source_artifact": archive_source},
        root=root,
    )
    controls = [*parser_control_rows(), *reduce_constructed_pairs(constructed_qualifier_pairs())]
    scripted = write_immutable_sidecar(
        raw / "scripted.json",
        scope="simulated_transport_events",
        payload={"parser_control_rows": controls},
        root=root,
    )
    receipts = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]
    terminal_controls = [
        {
            "control": "missing_adversarial_verify",
            "expected_errors": ["missing_terminal_receipt:adversarial_verify"],
            "observed_errors": ["missing_terminal_receipt:adversarial_verify"],
            "passed": True,
        },
        {
            "control": "complete_terminal_receipts",
            "expected_errors": [],
            "observed_errors": [],
            "passed": True,
        },
    ]
    return _build_artifact(
        root=root,
        started_ns=0,
        ended_ns=0,
        started_at="2026-09-20T00:00:00Z",
        completed_at="2026-09-20T00:00:00Z",
        phase_spans=[],
        preconditions=[{"check": "fixture", "passed": True}],
        source_hashes={},
        receipt_sidecars=[historical, scripted],
        protocol_manifest=manifest,
        schedule=schedule,
        archive_reduction=archive,
        controls=controls,
        validation_receipts=receipts,
        terminal_control_rows=terminal_controls,
        fixture_artifact=True,
    )


def _blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Publish exact external absence without inventing dependent model work."""

    current = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}:blocked",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={"work": "precondition checks only"},
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
    )
    current["monotonic_start_timestamp_ns"] = current.pop("started_monotonic_ns")
    current["monotonic_end_timestamp_ns"] = current.pop("ended_monotonic_ns")
    gate = _gate(
        str(failed["check"]),
        "precondition",
        str(failed["operator"]),
        failed["expected"],
        failed["observed"],
        False,
        "Unavailable branch-local evidence cannot be fabricated.",
        upstream=str(failed["upstream"]),
        path=str(failed["path"]),
        field=str(failed["field"]),
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked_precondition",
        "run_date": RUN_DATE,
        "started_at_utc": utc_now(),
        "completed_at_utc": utc_now(),
        "preconditions_checked": [deepcopy(dict(failed))],
        **current,
        "random_seed": {"panel_selection": RANDOM_SEED},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": 48,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 48,
            "stop_rule": "external prerequisite unavailable before dependent work",
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": _gate_summary([gate]),
        "verifier_is_oracle": True,
        "honest_verdict": f"blocked_{failed['check']}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "span_protocol_ready_score": 0,
        "archive_diagnosis_rows": [],
        "span_protocol_manifest": {},
        "parser_control_rows": [],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - declared capability E2E.
    """Authenticate, seal, validate, and atomically publish the no-model protocol."""

    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    checks, source_hashes = collect_preconditions(root)
    checks.insert(
        0,
        {
            "check": "run_date",
            "upstream": "execution_contract",
            "path": "execution_contract",
            "field": "run_date",
            "operator": "==",
            "expected": RUN_DATE,
            "observed": run_date,
            "passed": run_date == RUN_DATE,
        },
    )
    spans.append(
        _span("preconditions", phase_started, started, len(checks), "inputs_authenticated")
    )
    progress(
        started,
        "preconditions",
        "end",
        completed_units=len(checks),
        passed=all(row["passed"] for row in checks),
    )
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        blocked = _blocked_artifact(failed)
        progress(started, "write", "before_atomic_publish")
        atomic_json(output, blocked)
        progress(started, "write", "after_atomic_publish", completed_units=1)
        return blocked

    phase_started = time.monotonic()
    archive_source = _load_object(root / ARCHIVE_PATH)
    archive = reduce_archived_capture(archive_source)
    archive_sidecar = write_immutable_sidecar(
        raw_dir / "archived_exp7429_transport.json",
        scope="historical_model_receipts",
        payload={
            "source_path": ARCHIVE_PATH.as_posix(),
            "source_sha256": sha256_file(root / ARCHIVE_PATH),
            "original_flagged_adversarial": archive_source.get("flagged_adversarial"),
            "source_artifact": {
                "model_specs": deepcopy(archive_source.get("model_specs") or []),
                "development_rows": deepcopy(archive_source.get("development_rows") or []),
                "validation_receipts": deepcopy(archive_source.get("validation_receipts") or []),
            },
        },
        root=root,
    )
    spans.append(_span("archive_reduction", phase_started, started, 4, archive_sidecar["path"]))
    progress(started, "archive_reduction", "checkpoint", completed_units=4)

    phase_started = time.monotonic()
    progress(started, "corpus_decode", "before_decode")
    predictors, evaluators = _load_ragtruth(root, started=started)
    progress(started, "corpus_decode", "after_decode", completed_units=len(predictors))
    panel = seal_panel(predictors)
    schedule = build_evaluation_schedule(panel["evaluation"])
    evaluator = _evaluator_view(panel, evaluators)
    panel_path = raw_dir / "sealed_panel.json"
    schedule_path = raw_dir / "sealed_schedule.json"
    evaluator_path = raw_dir / "evaluator_annotations.json"
    _write_json(panel_path, panel)
    _write_json(schedule_path, {"schema": "carnot.exp7437.schedule.v1", "rows": schedule})
    _write_json(evaluator_path, evaluator)
    manifest = _protocol_manifest(
        _reference(panel_path, root=root),
        _reference(schedule_path, root=root),
        _reference(evaluator_path, root=root),
        schedule,
    )
    manifest_path = raw_dir / "span_protocol_manifest.json"
    _write_json(manifest_path, manifest)
    source_hashes["sealed_panel"] = _reference(panel_path, root=root)
    source_hashes["sealed_schedule"] = _reference(schedule_path, root=root)
    source_hashes["evaluator_annotations"] = _reference(evaluator_path, root=root)
    spans.append(_span("panel_seal", phase_started, started, 28, panel_path.as_posix()))
    progress(started, "panel_seal", "checkpoint", completed_units=28)

    phase_started = time.monotonic()
    controls = [*parser_control_rows(), *reduce_constructed_pairs(constructed_qualifier_pairs())]
    controls_sidecar = write_immutable_sidecar(
        raw_dir / "scripted_parser_controls.json",
        scope="simulated_transport_events",
        payload={"parser_control_rows": controls},
        root=root,
    )
    spans.append(
        _span("parser_controls", phase_started, started, len(controls), controls_sidecar["path"])
    )
    progress(started, "parser_controls", "checkpoint", completed_units=len(controls))

    phase_started = time.monotonic()
    private_root = Path(tempfile.mkdtemp(prefix="exp7437-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(started, "affected_validation", "before_subprocesses", planned=len(commands))
    affected_receipts = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    affected = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected_receipts)
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected_receipts),
        passed=affected["passed"],
    )
    spans.append(
        _span(
            "affected_validation",
            phase_started,
            started,
            len(affected_receipts),
            "affected_receipts_complete",
        )
    )

    provisional_controls = [
        {
            "control": "missing_adversarial_verify",
            "expected_errors": ["missing_terminal_receipt:adversarial_verify"],
            "observed_errors": ["missing_terminal_receipt:adversarial_verify"],
            "passed": True,
        },
        {
            "control": "complete_terminal_receipts",
            "expected_errors": [],
            "observed_errors": [],
            "passed": True,
        },
    ]
    candidate = _build_artifact(
        root=root,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        started_at=started_at,
        completed_at=utc_now(),
        phase_spans=spans,
        preconditions=checks,
        source_hashes=source_hashes,
        receipt_sidecars=[archive_sidecar, controls_sidecar],
        protocol_manifest=manifest,
        schedule=schedule,
        archive_reduction=archive,
        controls=controls,
        validation_receipts=affected_receipts,
        terminal_control_rows=provisional_controls,
        fixture_artifact=False,
    )
    receipt_controls = _receipt_controls(candidate, root=root, affected=affected_receipts)
    candidate["terminal_receipt_control_rows"] = receipt_controls
    candidate["reproducibility_checksum"] = artifact_checksum(candidate)
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    terminal_specs = _terminal_commands(root, candidate_path)
    if tuple(spec.name for spec in terminal_specs) != TERMINAL_CHECK_NAMES:
        raise RuntimeError("terminal_command_name_drift")
    progress(started, "terminal_validation", "before_subprocesses", planned=len(terminal_specs))
    terminal_receipts = run_categorized_commands(
        root,
        [
            PlannedCommand(
                spec,
                "safety" if spec.name == "adversarial_verify" else "completion",
                True,
            )
            for spec in terminal_specs
        ],
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=not validate_terminal_receipts(terminal_receipts),
    )
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            started,
            len(terminal_receipts),
            "terminal_receipts_complete",
        )
    )

    final = _build_artifact(
        root=root,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        started_at=started_at,
        completed_at=utc_now(),
        phase_spans=spans,
        preconditions=checks,
        source_hashes=source_hashes,
        receipt_sidecars=[archive_sidecar, controls_sidecar],
        protocol_manifest=manifest,
        schedule=schedule,
        archive_reduction=archive,
        controls=controls,
        validation_receipts=[*affected_receipts, *terminal_receipts],
        terminal_control_rows=receipt_controls,
        fixture_artifact=False,
    )
    errors = independent_reduce_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "write", "before_atomic_publish", path=output)
    atomic_json(output, final)
    progress(
        started, "write", "after_atomic_publish", completed_units=1, verdict=final["honest_verdict"]
    )
    return final


def date_argument(value: str) -> str:
    """Reject an execution date outside the fixed V652 protocol boundary."""

    if value != RUN_DATE:
        raise ValueError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the protocol or cold-replay one candidate in a fresh process."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        value = _load_object(args.validate)
        names = {
            row.get("name")
            for row in value.get("validation_receipts") or []
            if isinstance(row, Mapping)
        }
        require_terminal = bool(names.intersection(TERMINAL_CHECK_NAMES))
        errors = independent_reduce_artifact(
            value, root=REPO_ROOT, require_terminal=require_terminal
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "honest_verdict": result["honest_verdict"],
                "span_protocol_ready_score": result["span_protocol_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
