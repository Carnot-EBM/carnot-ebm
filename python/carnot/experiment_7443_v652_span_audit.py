"""Audit V652 claim-span capture bytes without producer parsing code.

The audit treats the capture as external evidence. It reads raw JSON, repeats
literal extraction checks, and keeps unstarted work distinct from observed
zeroes. No producer parser or verdict reducer is imported.

Spec refs: REQ-REPORT-7443 and SCENARIO-REPORT-7443-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import random
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
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "exp7443-v652-span-audit"
SCHEMA = "carnot.exp7443.v652.span_audit.v1"
RANDOM_SEED = 6_527_443
BOOTSTRAP_DRAWS = 10_000
ARMS = ("span", "verbatim")
DEVELOPMENT_UNITS = 4
EVALUATION_UNITS = 48

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7443_v652_span_audit.json")
RAW_DIR = Path("results/raw/experiment_7443_v652_span_audit")
MODULE_PATH = Path("python/carnot/experiment_7443_v652_span_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7443_v652_span_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7443_v652_span_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXP7437_PATH = Path("results/experiment_7437_v652_span_protocol.json")
EXP7442_PATH = Path("results/experiment_7442_v652_span_capture.json")
EXP7442_RAW = Path("results/raw/experiment_7442_v652_span_capture/owned_runtime")

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
    SPEC_PATH,
    Path("python/carnot/experiment_7429_v651_anchored_capture.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("results/experiment_7429_v651_anchored_capture.json"),
    ROADMAP_PATH,
)

REQUIRED_MUTATIONS = (
    "spans",
    "offsets",
    "finish_reasons",
    "canary_open_state",
    "per_arm_count",
    "current_historical_event_tags",
)

# Each marker is fixed by the sealed protocol. The audit repeats the literal
# check and does not ask a model to judge another model's extraction.
CONSTRUCTED_QUALIFIERS: dict[str, tuple[str, str]] = {
    "negation": ("The trial did not meet its endpoint.", "not"),
    "time": ("The permit is valid only until June 2027.", "only until June 2027"),
    "count": ("The panel approved exactly three cases.", "exactly three"),
    "unit": ("The dose increased by 5 mg.", "5 mg"),
    "comparison": ("Model A was 12% faster than Model B.", "12% faster than Model B"),
    "condition": (
        "The alarm activates only when both sensors fail.",
        "only when both sensors fail",
    ),
    "location": ("The ban applies only inside the northern district.", "northern district"),
    "frequency": ("The audit runs at most twice per year.", "at most twice per year"),
    "uncertainty": ("The comet will probably return after 2030.", "probably"),
    "exception": (
        "All records except sealed exhibits were released.",
        "except sealed exhibits",
    ),
    "range": ("The temperature stayed between 18°C and 21°C.", "between 18°C and 21°C"),
    "attribution": (
        "According to the audited filing, revenue increased.",
        "According to the audited filing",
    ),
}

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_details",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "model_duration_s",
    "computation_duration_s",
    "cold_start_duration_s",
    "validation_duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "promotion_score",
    "extraction_audit_complete_score",
    "audited_cohort",
    "independent_endpoint_rows",
    "semantic_scope",
)

FIELD_PRINCIPLES = {
    "schema": "Use a versioned plain top-level schema with experiment identity and terminal status.",
    "run_date": "Use the fixed execution date and actual UTC and monotonic boundaries.",
    "preconditions_checked": "Name each resource and observed prerequisite before dependent work.",
    "MODEL_SPECS": "Use an empty list because this audit performs no current LLM work.",
    "model_invoked": "Keep current attempted model work separate from producer evidence.",
    "invocation_counts": "Reconcile every current load and generation state; all are zero here.",
    "inference_substrate": "Name the host aggregation substrate without implying model inference.",
    "inference_substrate_class": "Declare aggregation because this run only reduces existing bytes.",
    "execution_venue": "Use host and report CPU, CUDA, and external-device identity separately.",
    "duration_s": "Measure current work and separate model, computation, cold-start, and validation time.",
    "phase_spans": "Bind phase boundaries, progress events, completed units, and checkpoints.",
    "random_seed": "Freeze the paired bootstrap seed; no sampling or fitting seed is needed.",
    "reproducibility_checksum": "Bind code, protocol, sources, rows, controls, and validation scope.",
    "source_artifact_hashes": "Preserve source bytes, original classes, flags, and typed sidecars.",
    "rows": "Keep one independently decoded row for every development and evaluation call.",
    "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
    "acceptance_gate_results": "Separate evidence validity, audit completion, and scientific benefit.",
    "gate_check_summary": "Name the exact failed path and field without hiding later failures.",
    "verifier_is_oracle": "The deployed verifier is not the scoring authority for this audit.",
    "honest_verdict": "Complete available auditing even when the evaluation panel did not run.",
    "verdict_class": "Use null for a valid completed audit with no treatment estimate.",
    "flagged_adversarial": "Preserve the producer's critical runtime finding.",
    "validation_receipts": "Record exact scoped commands, exits, durations, and hashed logs.",
    "field_principles": "Explain field intent separately while keeping scalars plain.",
    "promotion_score": "Keep zero because this milestone authorizes no rollout or weight change.",
    "extraction_audit_complete_score": "One means every available raw outcome passed independent checks.",
    "audited_cohort": "State whether evidence is development-only or the sealed evaluation panel.",
    "independent_endpoint_rows": "Expose recomputed per-call status and costs instead of aggregate-only claims.",
    "semantic_scope": "Separate literal checks, constructed exact authority, and unknown real semantics.",
}

V652_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and return an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    path: str,
    field: str,
    operator: str = "==",
    principle: str = "Authenticate the exact field before dependent work.",
) -> JsonDict:
    """Build one plain gate row with a stable failure shape."""

    passed = observed == expected if operator == "==" else observed in expected
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
        "field": field,
    }


def _source_reference(
    root: Path,
    relative: Path,
    receipt_class: str,
    **metadata: Any,
) -> JsonDict:
    """Bind exact source bytes and retain their evidence class."""

    path = root / relative
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "source_receipt_class": receipt_class,
        **deepcopy(metadata),
    }


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, JsonDict], dict[str, JsonDict]]:
    """Authenticate required files, roadmap declarations, and producer identity."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _gate(
                f"source_bytes:{relative.as_posix()}",
                "precondition",
                "readable_nonempty_bytes",
                observed,
                upstream=relative.as_posix(),
                path=relative.as_posix(),
                field="bytes",
            )
        )
        if observed is not None:
            sources[relative.as_posix()] = _source_reference(root, relative, "current_audit_input")

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _gate(
            "driving_requirement",
            "precondition",
            "REQ-REPORT-7443",
            "REQ-REPORT-7443" if "REQ-REPORT-7443" in spec_text else None,
            upstream=SPEC_PATH.as_posix(),
            path=SPEC_PATH.as_posix(),
            field="REQ-*",
        )
    )
    roadmap = (
        (root / ROADMAP_PATH).read_text(encoding="utf-8") if (root / ROADMAP_PATH).is_file() else ""
    )
    for task, deliverable in (
        ("exp7437-span-protocol", EXP7437_PATH),
        ("exp7442-span-capture", EXP7442_PATH),
    ):
        observed = task if task in roadmap and deliverable.as_posix() in roadmap else None
        checks.append(
            _gate(
                f"roadmap_declaration:{task}",
                "precondition",
                task,
                observed,
                upstream="research-roadmap.yaml",
                path=ROADMAP_PATH.as_posix(),
                field=task,
            )
        )

    artifacts: dict[str, JsonDict] = {}
    for label, relative, experiment_id in (
        ("exp7437", EXP7437_PATH, "exp7437-v652-span-protocol"),
        ("exp7442", EXP7442_PATH, "exp7442-v652-span-capture"),
    ):
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _gate(
                f"{label}_artifact_bytes",
                "precondition",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                upstream=experiment_id,
                path=relative.as_posix(),
                field="bytes",
            )
        )
        artifact = _load_object(path) if available else {}
        if artifact:
            artifacts[label] = artifact
            sources[relative.as_posix()] = _source_reference(
                root,
                relative,
                "historical_producer_artifact",
                original_verdict_class=artifact.get("verdict_class"),
                original_flagged_adversarial=artifact.get("flagged_adversarial"),
                original_status=artifact.get("status"),
            )
        for field, expected in (("experiment_id", experiment_id), ("milestone", MILESTONE)):
            checks.append(
                _gate(
                    f"{label}_{field}",
                    "precondition",
                    expected,
                    artifact.get(field),
                    upstream=experiment_id,
                    path=relative.as_posix(),
                    field=field,
                )
            )
        checks.append(
            _gate(
                f"{label}_flag_type",
                "precondition",
                True,
                isinstance(artifact.get("flagged_adversarial"), bool),
                upstream=experiment_id,
                path=relative.as_posix(),
                field="flagged_adversarial",
            )
        )

    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    quarantined = "experiment_id: 7443" in exclusion or "exp7443-v652-span-audit" in exclusion
    checks.append(
        _gate(
            "current_task_not_quarantined",
            "precondition",
            False,
            quarantined,
            upstream="ops/exclusion_manifest.yaml",
            path="ops/exclusion_manifest.yaml",
            field=EXPERIMENT_ID,
        )
    )
    return checks, sources, artifacts


def _choice_transport(row: Mapping[str, Any]) -> tuple[str, str | None, JsonDict]:
    response = row.get("raw_response")
    if not isinstance(response, Mapping):
        return "", None, {}
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return "", None, dict(response)
    choice = choices[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else ""
    return str(content or ""), choice.get("finish_reason"), dict(response)


def decode_raw_outcome(row: Mapping[str, Any]) -> JsonDict:
    """Decode one transport row without trusting stored parser fields."""

    errors: list[str] = []
    raw_request = row.get("raw_request") if isinstance(row.get("raw_request"), Mapping) else {}
    raw_response = row.get("raw_response") if isinstance(row.get("raw_response"), Mapping) else {}
    raw_reply = str(row.get("raw_reply") or "")
    for field, value in (
        ("raw_request", raw_request),
        ("raw_response", raw_response),
        ("raw_reply", raw_reply),
    ):
        if row.get(f"{field}_sha256") != canonical_hash(value):
            errors.append(f"{field}_hash_mismatch")

    attempted = row.get("attempted") is True
    terminal_state = str(row.get("terminal_state") or "")
    response_reply, response_finish, response_object = _choice_transport(row)
    if terminal_state == "response" and response_reply != raw_reply:
        errors.append("raw_reply_response_mismatch")
    if terminal_state == "response" and response_finish != row.get("finish_reason"):
        errors.append("finish_reason_mismatch")

    request_ceiling = raw_request.get("max_tokens") if attempted else None
    if attempted and request_ceiling != row.get("max_new_tokens"):
        errors.append("request_token_ceiling_mismatch")
    usage = (
        response_object.get("usage") if isinstance(response_object.get("usage"), Mapping) else {}
    )
    timings = (
        response_object.get("timings")
        if isinstance(response_object.get("timings"), Mapping)
        else {}
    )
    completion_tokens = usage.get("completion_tokens")
    predicted_tokens = timings.get("predicted_n")
    if (
        isinstance(completion_tokens, int)
        and isinstance(predicted_tokens, int)
        and completion_tokens != predicted_tokens
    ):
        errors.append("server_token_receipt_mismatch")

    claims: list[str] = []
    spans: list[list[int]] = []
    parse_status = "not_attempted"
    paragraph = str(row.get("paragraph") or "")
    disposition = "unstarted"
    if attempted and terminal_state == "cancelled":
        disposition = "cancelled"
    elif attempted and (terminal_state == "failed" or row.get("error")):
        disposition = "failed"
    elif attempted and row.get("finish_reason") in {"length", "max_tokens"}:
        disposition = "truncated"
    elif attempted and not raw_reply.strip():
        disposition = "empty"
        parse_status = "empty_transport"
    elif attempted:
        try:
            payload = json.loads(raw_reply)
        except json.JSONDecodeError:
            disposition = "malformed"
            parse_status = "malformed_json"
        else:
            parse_status = "valid_json"
            claim_values = payload.get("claims") if isinstance(payload, dict) else None
            if (
                not isinstance(payload, dict)
                or set(payload) != {"claims"}
                or not isinstance(claim_values, list)
            ):
                errors.append("claims_schema")
                disposition = "malformed"
            elif str(row.get("arm")) == "span":
                valid = True
                for value in claim_values:
                    if (
                        not isinstance(value, list)
                        or len(value) != 2
                        or any(
                            not isinstance(item, int) or isinstance(item, bool) for item in value
                        )
                    ):
                        errors.append("span_shape")
                        valid = False
                        continue
                    start, end = value
                    if start < 0 or end <= start or end > len(paragraph):
                        errors.append("span_bounds")
                        valid = False
                        continue
                    spans.append([start, end])
                    claims.append(paragraph[start:end])
                disposition = (
                    "empty" if valid and not claims else ("complete" if valid else "malformed")
                )
            elif str(row.get("arm")) == "verbatim":
                valid = True
                seen: set[str] = set()
                for value in claim_values:
                    if not isinstance(value, str) or not value:
                        errors.append("verbatim_shape")
                        valid = False
                        continue
                    occurrences = paragraph.count(value)
                    if occurrences == 0:
                        errors.append("verbatim_not_literal")
                        valid = False
                        continue
                    if occurrences > 1:
                        errors.append("verbatim_ambiguous")
                        valid = False
                        continue
                    if value in seen:
                        errors.append("verbatim_duplicate")
                        valid = False
                        continue
                    seen.add(value)
                    start = paragraph.index(value)
                    spans.append([start, start + len(value)])
                    claims.append(value)
                disposition = (
                    "empty" if valid and not claims else ("complete" if valid else "malformed")
                )
            else:
                errors.append("arm_invalid")
                disposition = "malformed"

    output_tokens = completion_tokens if attempted and terminal_state == "response" else None
    reduced: JsonDict = {
        "call_id": row.get("call_id"),
        "unit_id": row.get("unit_id"),
        "capture_phase": row.get("capture_phase"),
        "arm": row.get("arm"),
        "condition": row.get("condition"),
        "source_group": row.get("group_id"),
        "source_text": paragraph,
        "selected_propositions": claims,
        "claim_spans": spans,
        "attempted": attempted,
        "disposition": disposition,
        "finish_reason": row.get("finish_reason") if attempted else None,
        "parse_status": parse_status,
        "literal_span_reconstruction": disposition in {"complete", "empty"},
        "request_token_ceiling": request_ceiling,
        "declared_token_ceiling": row.get("max_new_tokens"),
        "server_prompt_tokens": usage.get("prompt_tokens") if attempted else None,
        "server_completion_tokens": completion_tokens if attempted else None,
        "server_predicted_tokens": predicted_tokens if attempted else None,
        "output_tokens": output_tokens,
        "semantic_quality": None,
        "errors": list(dict.fromkeys(errors)),
    }
    reduced["row_checksum"] = canonical_hash(reduced)
    return reduced


def _bootstrap_interval(values: Sequence[float], *, seed: int, draws: int) -> JsonDict:
    if not values:
        return {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "pairs": 0,
            "draws": draws,
            "seed": seed,
        }
    rng = random.Random(seed)
    estimate = sum(values) / len(values)
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws))
    return {
        "estimate": estimate,
        "ci95_low": samples[int(0.025 * (draws - 1))],
        "ci95_high": samples[int(0.975 * (draws - 1))],
        "pairs": len(values),
        "draws": draws,
        "seed": seed,
    }


def paired_effects(
    evaluation_rows: Sequence[Mapping[str, Any]], *, seed: int, draws: int
) -> JsonDict:
    """Reduce paired completion and token cost only from observed evaluation calls."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in evaluation_rows:
        grouped[str(row.get("unit_id"))][str(row.get("arm"))] = row
    completion: list[float] = []
    token_cost: list[float] = []
    any_attempted = any(row.get("attempted") is True for row in evaluation_rows)
    if any_attempted:
        for arms in grouped.values():
            if set(arms) != set(ARMS):
                continue
            span, verbatim = arms["span"], arms["verbatim"]
            if span.get("attempted") is True and verbatim.get("attempted") is True:
                completion.append(
                    float(span.get("disposition") == "complete")
                    - float(verbatim.get("disposition") == "complete")
                )
                if isinstance(span.get("output_tokens"), int) and isinstance(
                    verbatim.get("output_tokens"), int
                ):
                    token_cost.append(float(span["output_tokens"] - verbatim["output_tokens"]))
    return {
        "cohort": "sealed_evaluation_panel",
        "cluster_unit": "evaluation_unit",
        "paired_completion_effect": _bootstrap_interval(completion, seed=seed, draws=draws),
        "paired_token_cost_effect": _bootstrap_interval(token_cost, seed=seed + 1, draws=draws),
    }


def reduce_capture(
    development_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    draws: int,
) -> JsonDict:
    """Independently decode all planned calls and preserve every disposition."""

    rows = [decode_raw_outcome(row) for row in [*development_rows, *evaluation_rows]]
    counts = Counter(str(row["disposition"]) for row in rows)
    ordered_counts = {
        disposition: counts[disposition]
        for disposition in (
            "complete",
            "truncated",
            "malformed",
            "empty",
            "failed",
            "cancelled",
            "unstarted",
        )
    }
    evaluated = rows[len(development_rows) :]
    effects = paired_effects(evaluated, seed=seed, draws=draws)
    attempted_evaluation = sum(row["attempted"] is True for row in evaluated)
    return {
        "rows": rows,
        "raw_disposition_counts": ordered_counts,
        "evaluation_coverage": attempted_evaluation / len(evaluated) if evaluated else 0.0,
        **effects,
    }


def audit_constructed_controls(
    controls: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[str]]:
    """Check all sealed qualifier markers without assigning real-text semantics."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    errors: list[str] = []
    for row in controls:
        if row.get("scope") != "constructed_exact_check":
            continue
        grouped[str(row.get("pair_id"))][str(row.get("arm"))] = row
    result: list[JsonDict] = []
    for pair_id in sorted(grouped):
        arm_rows = grouped[pair_id]
        if set(arm_rows) != set(ARMS):
            errors.append(f"constructed_arm_count:{pair_id}")
            continue
        family = str(arm_rows["span"].get("family"))
        expected = CONSTRUCTED_QUALIFIERS.get(family)
        if expected is None:
            errors.append(f"constructed_family_unknown:{pair_id}")
            continue
        proposition, marker = expected
        selected: dict[str, list[str]] = {}
        retained = True
        for arm in ARMS:
            row = arm_rows[arm]
            claims = [str(value) for value in row.get("claims") or []]
            spans = row.get("claim_spans") or []
            selected[arm] = claims
            valid = (
                claims == [proposition]
                and spans == [[0, len(proposition)]]
                and marker in proposition
            )
            if not valid:
                errors.append(f"constructed_marker_missing:{pair_id}:{arm}")
                retained = False
        result.append(
            {
                "pair_id": pair_id,
                "family": family,
                "authority": "constructed_exact_string",
                "source_text": proposition,
                "required_marker": marker,
                "selected_propositions": selected,
                "required_marker_retained": retained,
                "real_semantic_authority": False,
            }
        )
    if len(result) != len(CONSTRUCTED_QUALIFIERS):
        errors.append("constructed_pair_count_mismatch")
    return result, list(dict.fromkeys(errors))


def capture_integrity_errors(fixture: Mapping[str, Any]) -> list[str]:
    """Reject raw-row, canary, count, and producer-event provenance drift."""

    development = list(fixture.get("development_rows") or [])
    evaluation = list(fixture.get("evaluation_rows") or [])
    errors: list[str] = []
    decoded_development = [decode_raw_outcome(row) for row in development]
    decoded_evaluation = [decode_raw_outcome(row) for row in evaluation]
    for row in [*decoded_development, *decoded_evaluation]:
        errors.extend(str(error) for error in row["errors"])
    development_counts = Counter(row.get("arm") for row in development)
    if any(development_counts[arm] != DEVELOPMENT_UNITS for arm in ARMS):
        errors.append("development_per_arm_count_mismatch")
    evaluation_counts = Counter(row.get("arm") for row in evaluation)
    if any(evaluation_counts[arm] != EVALUATION_UNITS for arm in ARMS):
        errors.append("evaluation_per_arm_count_mismatch")
    usable = Counter(
        row["arm"]
        for row in decoded_development
        if row["disposition"] == "complete" and row["errors"] == []
    )
    expected_open = all(usable[arm] >= 3 for arm in ARMS)
    if fixture.get("canary_open") is not expected_open:
        errors.append("canary_open_state_mismatch")
    if fixture.get("producer_event_receipt_class") != "historical_producer_model_events":
        errors.append("producer_event_class_invalid")
    for event in fixture.get("producer_events") or []:
        if not isinstance(event, Mapping) or event.get("scope") != "current":
            errors.append("producer_original_scope_changed")
    return list(dict.fromkeys(errors))


def _replace_reply(row: JsonDict, reply: str) -> None:
    row["raw_reply"] = reply
    row["raw_reply_sha256"] = canonical_hash(reply)
    row["raw_response"]["choices"][0]["message"]["content"] = reply
    row["raw_response_sha256"] = canonical_hash(row["raw_response"])


def _control_fixture() -> JsonDict:
    development: list[JsonDict] = []
    for index in range(DEVELOPMENT_UNITS):
        paragraph = "Café confirms the bounded fixture."
        for arm in ARMS:
            reply = (
                json.dumps({"claims": [[0, len(paragraph)]]})
                if arm == "span"
                else json.dumps({"claims": [paragraph]}, ensure_ascii=False)
            )
            request = {"max_tokens": 256}
            response = {
                "choices": [{"finish_reason": "stop", "message": {"content": reply}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4},
                "timings": {"predicted_n": 4},
            }
            development.append(
                {
                    "arm": arm,
                    "attempted": True,
                    "call_id": f"development-{index:02d}-{arm}",
                    "capture_phase": "development",
                    "condition": "sealed_development_paragraph",
                    "error": None,
                    "finish_reason": "stop",
                    "max_new_tokens": 256,
                    "paragraph": paragraph,
                    "raw_reply": reply,
                    "raw_reply_sha256": canonical_hash(reply),
                    "raw_request": request,
                    "raw_request_sha256": canonical_hash(request),
                    "raw_response": response,
                    "raw_response_sha256": canonical_hash(response),
                    "terminal_state": "response",
                    "unit_id": f"development-{index:02d}",
                }
            )
    evaluation: list[JsonDict] = []
    for index in range(EVALUATION_UNITS):
        for arm in ARMS:
            evaluation.append(_unstarted_fixture(index, arm))
    return {
        "development_rows": development,
        "evaluation_rows": evaluation,
        "canary_open": True,
        "producer_event_receipt_class": "historical_producer_model_events",
        "producer_events": [
            {
                "call_id": "generation-0",
                "operation": "generation",
                "scope": "current",
                "state": "attempted",
            },
            {
                "call_id": "generation-0",
                "operation": "generation",
                "scope": "current",
                "state": "completed",
            },
        ],
    }


def _unstarted_fixture(index: int, arm: str) -> JsonDict:
    empty: JsonDict = {}
    return {
        "arm": arm,
        "attempted": False,
        "call_id": f"evaluation-{index:02d}-{arm}",
        "capture_phase": "evaluation",
        "condition": "ragtruth_unchanged_response",
        "error": None,
        "finish_reason": None,
        "max_new_tokens": 256,
        "paragraph": "Unstarted source paragraph.",
        "raw_reply": "",
        "raw_reply_sha256": canonical_hash(""),
        "raw_request": empty,
        "raw_request_sha256": canonical_hash(empty),
        "raw_response": empty,
        "raw_response_sha256": canonical_hash(empty),
        "terminal_state": "unstarted",
        "unit_id": f"evaluation-{index:02d}",
    }


def run_mutation_controls() -> list[JsonDict]:
    """Plant every registered corruption and require a named rejection."""

    fixture = _control_fixture()
    mutations: list[tuple[str, JsonDict, str]] = []

    changed = deepcopy(fixture)
    _replace_reply(changed["development_rows"][0], '{"claims":[["0",4]]}')
    mutations.append(("spans", changed, "span_shape"))
    changed = deepcopy(fixture)
    _replace_reply(changed["development_rows"][0], '{"claims":[[0,999]]}')
    mutations.append(("offsets", changed, "span_bounds"))
    changed = deepcopy(fixture)
    changed["development_rows"][0]["finish_reason"] = "length"
    mutations.append(("finish_reasons", changed, "finish_reason_mismatch"))
    changed = deepcopy(fixture)
    changed["canary_open"] = False
    mutations.append(("canary_open_state", changed, "canary_open_state_mismatch"))
    changed = deepcopy(fixture)
    changed["evaluation_rows"].pop()
    mutations.append(("per_arm_count", changed, "evaluation_per_arm_count_mismatch"))
    changed = deepcopy(fixture)
    changed["producer_event_receipt_class"] = "current_audit_model_events"
    mutations.append(("current_historical_event_tags", changed, "producer_event_class_invalid"))

    rows: list[JsonDict] = []
    for attack, candidate, expected in mutations:
        observed = capture_integrity_errors(candidate)
        rows.append(
            {
                "attack": attack,
                "expected_error": expected,
                "observed_errors": observed,
                "passed": expected in observed,
            }
        )
    return rows


def _fixture_controls() -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, (family, (proposition, _marker)) in enumerate(CONSTRUCTED_QUALIFIERS.items(), 1):
        for arm in ARMS:
            rows.append(
                {
                    "pair_id": f"qualifier-{index:02d}-{family}",
                    "family": family,
                    "arm": arm,
                    "scope": "constructed_exact_check",
                    "claims": [proposition],
                    "claim_spans": [[0, len(proposition)]],
                }
            )
    return rows


def _load_sidecar_rows(
    root: Path,
    references: Sequence[Mapping[str, Any]],
    *,
    base: Path,
    receipt_class: str,
    sources: dict[str, JsonDict],
) -> tuple[list[JsonDict], list[str]]:
    rows: list[JsonDict] = []
    errors: list[str] = []
    for index, reference in enumerate(references):
        relative = base / str(reference.get("path") or "")
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        if observed != reference.get("sha256"):
            errors.append(f"sidecar_hash_mismatch:{relative.as_posix()}")
            continue
        value = _load_object(path)
        if not value:
            errors.append(f"sidecar_json_invalid:{relative.as_posix()}")
            continue
        label = f"{receipt_class}:{index:03d}"
        sources[label] = _source_reference(root, relative, receipt_class)
        rows.append(value)
    return rows, errors


def _audit_sources(
    root: Path,
) -> tuple[list[JsonDict], dict[str, JsonDict], JsonDict]:
    checks, sources, artifacts = collect_preconditions(root)
    protocol = artifacts.get("exp7437") or {}
    capture = artifacts.get("exp7442") or {}
    if not capture:
        return checks, sources, {"available": False, "audit_errors": ["exp7442_missing"]}

    errors: list[str] = []
    hashes = capture.get("source_artifact_hashes")
    hashes = dict(hashes) if isinstance(hashes, Mapping) else {}
    native_ref = hashes.get("current_native_call")
    if not isinstance(native_ref, Mapping):
        errors.append("native_call_reference_missing")
        native = {}
    else:
        native_path = Path(str(native_ref.get("path") or ""))
        observed = sha256_file(root / native_path) if (root / native_path).is_file() else None
        if observed != native_ref.get("sha256"):
            errors.append("native_call_hash_mismatch")
            native = {}
        else:
            native = _load_object(root / native_path)
            sources["exp7442_current_native_call"] = _source_reference(
                root, native_path, "historical_producer_native_call"
            )

    response_rows, response_errors = _load_sidecar_rows(
        root,
        hashes.get("current_response_shards") or [],
        base=EXP7442_RAW,
        receipt_class="historical_producer_model_response",
        sources=sources,
    )
    event_rows, event_errors = _load_sidecar_rows(
        root,
        hashes.get("current_event_shards") or [],
        base=EXP7442_RAW,
        receipt_class="historical_producer_model_event",
        sources=sources,
    )
    evaluation_rows, evaluation_errors = _load_sidecar_rows(
        root,
        hashes.get("current_evaluation_shards") or [],
        base=EXP7442_RAW,
        receipt_class="producer_evaluation_row",
        sources=sources,
    )
    errors.extend([*response_errors, *event_errors, *evaluation_errors])
    if len(response_rows) != 1:
        errors.append("response_sidecar_count_mismatch")
    elif native and any(
        response_rows[0].get(field) != native.get(field)
        for field in ("call_id", "raw_reply", "raw_request", "raw_response", "finish_reason")
    ):
        errors.append("native_response_sidecar_mismatch")

    development_rows = list(capture.get("development_rows") or [])
    if native and development_rows:
        for field in ("call_id", "raw_reply_sha256", "raw_request_sha256", "raw_response_sha256"):
            if development_rows[0].get(field) != native.get(field):
                errors.append(f"native_development_mismatch:{field}")
    if len(evaluation_rows) != len(capture.get("extraction_rows") or []):
        errors.append("evaluation_sidecar_count_mismatch")

    controls_ref = next(
        (
            row
            for row in protocol.get("receipt_sidecars") or []
            if isinstance(row, Mapping) and row.get("scope") == "simulated_transport_events"
        ),
        None,
    )
    controls: list[JsonDict] = []
    if not isinstance(controls_ref, Mapping):
        errors.append("constructed_control_reference_missing")
    else:
        control_path = Path(str(controls_ref.get("path") or ""))
        observed = sha256_file(root / control_path) if (root / control_path).is_file() else None
        if observed != controls_ref.get("sha256"):
            errors.append("constructed_control_hash_mismatch")
        else:
            control_object = _load_object(root / control_path)
            payload = control_object.get("payload")
            if isinstance(payload, Mapping):
                controls = list(payload.get("parser_control_rows") or [])
            sources["exp7437_scripted_parser_controls"] = _source_reference(
                root, control_path, "scripted_constructed_controls"
            )

    constructed_rows, constructed_errors = audit_constructed_controls(controls)
    errors.extend(constructed_errors)
    fixture = {
        "development_rows": development_rows,
        "evaluation_rows": evaluation_rows,
        "canary_open": (capture.get("development_gate") or {}).get("capture_open"),
        "producer_event_receipt_class": "historical_producer_model_events",
        "producer_events": event_rows,
    }
    errors.extend(capture_integrity_errors(fixture))
    runner = capture.get("runner_receipt") or {}
    for row in development_rows:
        if row.get("attempted") is True:
            request = row.get("raw_request") or {}
            declared = runner.get("decoding") or {}
            if request.get("max_tokens") != declared.get("max_new_tokens"):
                errors.append("runner_request_token_ceiling_mismatch")

    reduced = reduce_capture(
        development_rows,
        evaluation_rows,
        seed=RANDOM_SEED,
        draws=BOOTSTRAP_DRAWS,
    )
    return (
        checks,
        sources,
        {
            "available": True,
            "audit_errors": list(dict.fromkeys(errors)),
            "capture": reduced,
            "constructed_pair_audit_rows": constructed_rows,
            "producer": {
                "status": capture.get("status"),
                "honest_verdict": capture.get("honest_verdict"),
                "verdict_class": capture.get("verdict_class"),
                "flagged_adversarial": capture.get("flagged_adversarial"),
                "gate_check_summary": deepcopy(capture.get("gate_check_summary")),
                "model_invoked": capture.get("model_invoked"),
                "invocation_counts": deepcopy(capture.get("invocation_counts")),
            },
            "archived_model_event_sidecars": [
                deepcopy(reference)
                for label, reference in sources.items()
                if "producer_model_event" in str(reference.get("source_receipt_class"))
                or "producer_model_response" in str(reference.get("source_receipt_class"))
                or "producer_native_call" in str(reference.get("source_receipt_class"))
            ],
        },
    )


def _required_validation_passed(receipts: Sequence[Mapping[str, Any]], candidate: bool) -> bool:
    affected = set(validation_scope.REQUIRED_CHECK_NAMES)
    terminal = {
        "declared_entrypoint_cold_replay",
        "independent_cold_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    required = affected if candidate else affected | terminal
    rows = {str(row.get("name")): row for row in receipts}
    return all(
        name in rows
        and rows[name].get("passed") is True
        and rows[name].get("exit_code") == 0
        and rows[name].get("timed_out") is not True
        for name in required
    )


def _row_checksum_valid(row: Mapping[str, Any]) -> bool:
    value = dict(row)
    checksum = value.pop("row_checksum", None)
    return checksum == canonical_hash(value)


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable audit evidence while allowing real timing receipts to vary."""

    return canonical_hash(
        {
            "schema": value.get("schema"),
            "experiment_id": value.get("experiment_id"),
            "source_artifact_hashes": value.get("source_artifact_hashes"),
            "rows": value.get("rows"),
            "constructed_pair_audit_rows": value.get("constructed_pair_audit_rows"),
            "audit_mutation_rows": value.get("audit_mutation_rows"),
            "sample_size_budget": value.get("sample_size_budget"),
            "semantic_scope": value.get("semantic_scope"),
            "validation_manifest": value.get("validation_manifest"),
            "upstream_producer_disposition": value.get("upstream_producer_disposition"),
        }
    )


def _summary_from_gate(gate: Mapping[str, Any]) -> JsonDict:
    return {
        "all_passed": False,
        "failed_count": 1,
        "failed_checks": [gate.get("check")],
        **{
            key: deepcopy(gate.get(key))
            for key in (
                "check",
                "upstream",
                "path",
                "field",
                "operator",
                "expected",
                "observed",
                "passed",
            )
        },
    }


def _sample_budget(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    attempted = sum(row.get("attempted") is True for row in rows)
    completed = sum(row.get("disposition") in {"complete", "empty"} for row in rows)
    failed = sum(
        row.get("disposition") in {"failed", "cancelled", "malformed", "truncated"} for row in rows
    )
    unstarted = sum(row.get("disposition") == "unstarted" for row in rows)
    return {
        "planned_calls": DEVELOPMENT_UNITS * len(ARMS) + EVALUATION_UNITS * len(ARMS),
        "planned_independent_evaluation_units": EVALUATION_UNITS,
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "censored": 0,
        "unstarted": unstarted,
        "stopping_rule": "Audit every available producer row; do not open or rerun model work.",
    }


def _build_artifact(
    evidence: Mapping[str, Any],
    *,
    checks: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
    candidate: bool,
    fixture: bool = False,
) -> JsonDict:
    available = evidence.get("available") is True
    errors = list(evidence.get("audit_errors") or [])
    capture = dict(evidence.get("capture") or {})
    rows = deepcopy(capture.get("rows") or [])
    validation_passed = _required_validation_passed(validation_receipts, candidate) or fixture
    preconditions_passed = all(row.get("passed") is True for row in checks)
    audit_complete = available and not errors and preconditions_passed and validation_passed
    producer = dict(evidence.get("producer") or {})
    evaluation_ran = any(
        row.get("attempted") is True and row.get("capture_phase") == "evaluation" for row in rows
    )
    audited_cohort = "sealed_evaluation_panel" if evaluation_ran else "development_only"

    gates = [deepcopy(dict(row)) for row in checks]
    gates.extend(
        [
            _gate(
                "independent_raw_integrity",
                "validity",
                [],
                errors,
                upstream="exp7442-v652-span-capture",
                path=EXP7442_PATH.as_posix(),
                field="raw_sidecars",
                principle="Every available raw outcome must pass independent byte and shape checks.",
            ),
            _gate(
                "producer_runtime_integrity",
                "scientific_context",
                True,
                producer.get("verdict_class") != "disqualified",
                upstream="exp7442-v652-span-capture",
                path=EXP7442_PATH.as_posix(),
                field="verdict_class",
                principle="Preserve producer invalidity without suppressing the independent audit.",
            ),
            _gate(
                "sealed_evaluation_observed",
                "benefit",
                True,
                evaluation_ran,
                upstream="exp7442-v652-span-capture",
                path=EXP7442_PATH.as_posix(),
                field="extraction_rows.attempted",
                principle="No treatment effect follows from development-only evidence.",
            ),
            _gate(
                "required_validation",
                "completion",
                True,
                validation_passed,
                upstream=EXPERIMENT_ID,
                path=RESULT_PATH.as_posix(),
                field="validation_receipts",
                principle="Every declared affected and terminal reader must pass.",
            ),
        ]
    )

    missing_exp7442 = next(
        (
            row
            for row in checks
            if row.get("check") == "exp7442_artifact_bytes" and not row.get("passed")
        ),
        None,
    )
    if missing_exp7442 is not None:
        status = "blocked_exp7442_artifact_missing"
        verdict = "blocked_exp7442_artifact_missing"
        verdict_class = "blocked"
        summary = _summary_from_gate(missing_exp7442)
    elif errors or not preconditions_passed or not validation_passed:
        status = "complete_span_audit_disqualified"
        verdict = "complete_disqualified_span_audit_integrity_failed"
        verdict_class = "disqualified"
        first = next(row for row in gates if row.get("passed") is not True)
        summary = _summary_from_gate(first)
    else:
        status = (
            "complete_development_only_span_audit" if not evaluation_ran else "complete_span_audit"
        )
        verdict = (
            "complete_null_development_only_producer_runtime_failure"
            if not evaluation_ran
            else "complete_null_span_audit_no_registered_benefit"
        )
        verdict_class = "null"
        producer_summary = producer.get("gate_check_summary")
        if isinstance(producer_summary, Mapping):
            summary = deepcopy(dict(producer_summary))
        else:
            benefit = next(row for row in gates if row.get("check") == "sealed_evaluation_observed")
            summary = _summary_from_gate(benefit)

    completion = capture.get("paired_completion_effect") or {
        "estimate": None,
        "ci95_low": None,
        "ci95_high": None,
        "pairs": 0,
        "draws": BOOTSTRAP_DRAWS,
        "seed": RANDOM_SEED,
    }
    token_cost = capture.get("paired_token_cost_effect") or {
        "estimate": None,
        "ci95_low": None,
        "ci95_high": None,
        "pairs": 0,
        "draws": BOOTSTRAP_DRAWS,
        "seed": RANDOM_SEED + 1,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": deepcopy(list(checks)),
        **deepcopy(dict(current_receipt)),
        "model_duration_s": 0.0,
        "computation_duration_s": sum(
            float(row.get("duration_s", 0.0))
            for row in current_receipt.get("phase_spans") or []
            if row.get("phase") == "source_audit"
        ),
        "cold_start_duration_s": 0.0,
        "validation_duration_s": sum(
            float(row.get("duration_s", 0.0)) for row in validation_receipts
        ),
        "random_seed": {"paired_bootstrap": RANDOM_SEED, "sampling": None, "fitting": None},
        "source_artifact_hashes": deepcopy(dict(sources)),
        "rows": rows,
        "independent_endpoint_rows": deepcopy(rows),
        "raw_disposition_counts": deepcopy(capture.get("raw_disposition_counts") or {}),
        "evaluation_coverage": float(capture.get("evaluation_coverage", 0.0)),
        "paired_completion_effect": deepcopy(completion),
        "paired_token_cost_effect": deepcopy(token_cost),
        "sample_size_budget": _sample_budget(rows),
        "acceptance_gate_results": gates,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(producer.get("flagged_adversarial")),
        "validation_receipts": deepcopy(list(validation_receipts)),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "extraction_audit_complete_score": int(audit_complete),
        "audited_cohort": audited_cohort,
        "semantic_scope": {
            "constructed": "exact_string_qualifier_controls_only",
            "literal_reconstruction": "half_open_unicode_offsets_and_exact_substrings",
            "real_paragraphs": "unknown_mechanical_omission_inspection_only",
            "ragtruth_response_labels_certify_extraction": False,
        },
        "constructed_pair_audit_rows": deepcopy(evidence.get("constructed_pair_audit_rows") or []),
        "audit_integrity_errors": errors,
        "audit_mutation_rows": run_mutation_controls(),
        "upstream_producer_disposition": producer,
        "archived_model_event_sidecars": deepcopy(
            evidence.get("archived_model_event_sidecars") or []
        ),
        "continuation_decision": {
            "decision": "repair_runtime_ownership_then_run_changed_capture",
            "actual_cause": (producer.get("gate_check_summary") or {}).get("check"),
            "retire_unchanged_representation": False,
            "reason": "The representation received no sealed evaluation and did not repeat its registered null.",
        },
        "validation_manifest": {
            "test_paths": list(V652_MANIFEST.test_paths),
            "changed_modules": list(V652_MANIFEST.changed_modules),
            "static_paths": list(V652_MANIFEST.static_paths),
            "affected_checks": list(validation_scope.REQUIRED_CHECK_NAMES),
            "terminal_checks": [
                "declared_entrypoint_cold_replay",
                "independent_cold_reducer",
                "adversarial_verify",
                "verdict_row_consistency_strict",
            ],
        },
        "candidate_artifact": candidate,
        "fixture_artifact": fixture,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test() -> JsonDict:
    """Build one complete development-only fixture through production reducers."""

    fixture = _control_fixture()
    first = fixture["development_rows"][0]
    _replace_reply(first, '{"claims":[]}')
    fixture["development_rows"] = [first] + [
        {
            **_unstarted_fixture(index, arm),
            "call_id": f"development-{index:02d}-{arm}",
            "capture_phase": "development",
            "condition": "sealed_development_paragraph",
            "unit_id": f"development-{index:02d}",
        }
        for index, arm in (
            (0, "verbatim"),
            (1, "span"),
            (1, "verbatim"),
            (2, "span"),
            (2, "verbatim"),
            (3, "span"),
            (3, "verbatim"),
        )
    ]
    fixture["canary_open"] = False
    capture = reduce_capture(
        fixture["development_rows"], fixture["evaluation_rows"], seed=RANDOM_SEED, draws=20
    )
    controls, errors = audit_constructed_controls(_fixture_controls())
    receipt = build_current_work_receipt(
        run_id="exp7443-fixture",
        owner_pid=1,
        events=[],
        inference_substrate="host_cpu_json_aggregation",
        inference_substrate_details={"cpu": "fixture", "cuda": None, "external_device": None},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=1_000_000_000,
        phase_spans=[
            {
                "phase": "audit",
                "start_s": 0.0,
                "end_s": 1.0,
                "duration_s": 1.0,
                "completed_units": 104,
                "checkpoint": "fixture",
            }
        ],
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False, "duration_s": 0.0}
        for name in validation_scope.REQUIRED_CHECK_NAMES
    ]
    producer = {
        "status": "complete_producer_runtime_failed",
        "honest_verdict": "complete_disqualified_span_capture_producer_runtime_error",
        "verdict_class": "disqualified",
        "flagged_adversarial": True,
        "gate_check_summary": {
            "check": "recovered_owned_process_identity",
            "upstream": "gpu_lease_journal",
            "path": "/tmp/carnot-gpu-leases",
            "field": "owner",
            "operator": "==",
            "expected": "owned_identity",
            "observed": None,
            "passed": False,
        },
    }
    return _build_artifact(
        {
            "available": True,
            "audit_errors": errors,
            "capture": capture,
            "constructed_pair_audit_rows": controls,
            "producer": producer,
        },
        checks=[],
        sources={},
        validation_receipts=receipts,
        current_receipt=receipt,
        started_at_utc="2026-09-20T00:00:00+00:00",
        completed_at_utc="2026-09-20T00:00:01+00:00",
        candidate=True,
        fixture=True,
    )


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    """Publish exact external absence without inventing capture rows."""

    receipt = build_current_work_receipt(
        run_id="exp7443-blocked",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_cpu_json_aggregation",
        inference_substrate_details={
            "cpu": platform.processor() or "host_cpu",
            "cuda": None,
            "external_device": None,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    return _build_artifact(
        {"available": False, "audit_errors": ["exp7442_missing"], "producer": {}},
        checks=checks,
        sources=sources,
        validation_receipts=[],
        current_receipt=receipt,
        started_at_utc=datetime.now(UTC).isoformat(),
        completed_at_utc=datetime.now(UTC).isoformat(),
        candidate=True,
    )


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_source_bytes: bool = True
) -> list[str]:
    """Cold-check identity, independent rows, provenance, controls, and checksum."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in value:
            errors.append(f"required_field_missing:{field}")
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_invalid")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("artifact_schedule_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current_model_declaration_invalid")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_invalid")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class_invalid")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_invalid")
    rows = value.get("rows") or []
    independent = value.get("independent_endpoint_rows") or []
    if canonical_hash(rows) != canonical_hash(independent) or any(
        not isinstance(row, Mapping) or not _row_checksum_valid(row) for row in rows
    ):
        errors.append("independent_rows_mismatch")
    counts = Counter(row.get("disposition") for row in rows if isinstance(row, Mapping))
    expected_counts = {
        name: counts[name]
        for name in (
            "complete",
            "truncated",
            "malformed",
            "empty",
            "failed",
            "cancelled",
            "unstarted",
        )
    }
    if value.get("raw_disposition_counts") != expected_counts and rows:
        errors.append("disposition_counts_mismatch")
    if value.get("audited_cohort") == "development_only":
        if value.get("evaluation_coverage") != 0.0:
            errors.append("development_only_coverage_invalid")
        for field in ("paired_completion_effect", "paired_token_cost_effect"):
            if (value.get(field) or {}).get("estimate") is not None:
                errors.append("development_effect_not_unknown")
    attacks = value.get("audit_mutation_rows") or []
    if {row.get("attack") for row in attacks if isinstance(row, Mapping)} != set(
        REQUIRED_MUTATIONS
    ) or any(row.get("passed") is not True for row in attacks if isinstance(row, Mapping)):
        errors.append("mutation_controls_invalid")
    semantic = value.get("semantic_scope") or {}
    if (
        semantic.get("real_paragraphs") != "unknown_mechanical_omission_inspection_only"
        or semantic.get("ragtruth_response_labels_certify_extraction") is not False
    ):
        errors.append("semantic_scope_invalid")
    if value.get("extraction_audit_complete_score") not in {0, 1}:
        errors.append("audit_complete_score_invalid")
    if value.get("fixture_artifact") is not True:
        errors.extend(validate_current_work_receipt(value, root=root))
    if verify_source_bytes:
        for label, reference in (value.get("source_artifact_hashes") or {}).items():
            if not isinstance(reference, Mapping):
                errors.append(f"source_reference_invalid:{label}")
                continue
            path = Path(str(reference.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            observed = sha256_file(resolved) if resolved.is_file() else None
            if observed != reference.get("sha256"):
                errors.append(f"source_hash_mismatch:{label}")
    candidate = value.get("candidate_artifact") is True
    if value.get("fixture_artifact") is not True and not _required_validation_passed(
        value.get("validation_receipts") or [], candidate
    ):
        errors.append("required_validation_failed")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(
    path: Path, *, root: Path = REPO_ROOT, verify_source_bytes: bool = True
) -> list[str]:
    value = _load_object(path)
    return (
        validate_artifact(value, root=root, verify_source_bytes=verify_source_bytes)
        if value
        else ["candidate_artifact_unreadable"]
    )


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Reload producer sidecars and compare the complete scientific projection."""

    value = _load_object(path)
    errors = validate_artifact(value, root=root)
    _checks, _sources, evidence = _audit_sources(root)
    if evidence.get("available"):
        capture = evidence.get("capture") or {}
        comparisons = {
            "rows": capture.get("rows"),
            "raw_disposition_counts": capture.get("raw_disposition_counts"),
            "evaluation_coverage": capture.get("evaluation_coverage"),
            "paired_completion_effect": capture.get("paired_completion_effect"),
            "paired_token_cost_effect": capture.get("paired_token_cost_effect"),
            "constructed_pair_audit_rows": evidence.get("constructed_pair_audit_rows"),
        }
        for field, observed in comparisons.items():
            if canonical_hash(value.get(field)) != canonical_hash(observed):
                errors.append(f"independent_reduction_mismatch:{field}")
    return list(dict.fromkeys(errors))


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    commands = build_command_plan(root, V652_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, V652_MANIFEST, commands)
    if plan_errors:
        raise ValueError("validation_plan_invalid:" + ",".join(plan_errors))
    return commands


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:
    values = (
        (
            "declared_entrypoint_cold_replay",
            (".venv/bin/python", "-u", WRAPPER_PATH.as_posix(), "--cold-replay", str(candidate)),
            "completion",
        ),
        (
            "independent_cold_reducer",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--independent-reduce",
                str(candidate),
            ),
            "completion",
        ),
        (
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "completion",
        ),
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate", timeout_s=1200.0),
            category,
            True,
        )
        for name, argv, category in values
    ]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7443] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint": _utc_now(),
    }


def _current_receipt(
    started_ns: int,
    ended_ns: int,
    spans: Sequence[Mapping[str, Any]],
    sidecars: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return build_current_work_receipt(
        run_id=f"exp7443-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_cpu_json_and_numeric_aggregation",
        inference_substrate_details={
            "cpu": platform.processor() or "host_cpu",
            "cuda": None,
            "external_device": None,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=ended_ns - started_ns,
        sidecar_references=sidecars,
        phase_spans=spans,
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )


def _receipt_sidecars(evidence: Mapping[str, Any]) -> list[JsonDict]:
    rows = []
    for reference in evidence.get("archived_model_event_sidecars") or []:
        rows.append(
            {
                "path": reference.get("path"),
                "sha256": reference.get("sha256"),
                "scope": "historical_model_receipts",
            }
        )
    return rows


def run_experiment(
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - executed through the declared entrypoint E2E.
    """Authenticate, reduce, validate, replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    _progress(started, "startup", "start")

    phase_started = time.monotonic()
    _progress(started, "source_audit", "before_benchmark")
    checks, sources, evidence = _audit_sources(root)
    for relative, receipt_class in (
        (MODULE_PATH, "current_audit_code"),
        (WRAPPER_PATH, "current_audit_entrypoint"),
        (TEST_PATH, "current_audit_tests"),
        (SPEC_PATH, "current_audit_protocol"),
    ):
        sources[relative.as_posix()] = _source_reference(root, relative, receipt_class)
    units = len((evidence.get("capture") or {}).get("rows") or [])
    spans.append(_span("source_audit", phase_started, started, units))
    _progress(started, "source_audit", "after_benchmark", completed_units=units)

    private = Path(tempfile.mkdtemp(prefix="exp7443-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private)
    phase_started = time.monotonic()
    _progress(started, "affected_validation", "before_subprocesses", units=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
    )
    affected_reduction = reduce_affected_receipts(root, V652_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    _progress(
        started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"],
    )

    receipt = _current_receipt(started_ns, time.monotonic_ns(), spans, _receipt_sidecars(evidence))
    candidate = _build_artifact(
        evidence,
        checks=checks,
        sources=sources,
        validation_receipts=affected,
        current_receipt=receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        candidate=True,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    _progress(started, "candidate_write", "before_atomic", path=candidate_path)
    atomic_json(candidate_path, candidate)
    _progress(started, "candidate_write", "after_atomic", bytes=candidate_path.stat().st_size)

    phase_started = time.monotonic()
    terminal_plan = _terminal_commands(candidate_path)
    _progress(started, "terminal_validation", "before_subprocesses", units=len(terminal_plan))
    terminal = run_categorized_commands(
        root,
        terminal_plan,
        log_dir=root / RAW_DIR / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=all(row.get("passed") is True for row in terminal),
    )

    final_receipt = _current_receipt(
        started_ns, time.monotonic_ns(), spans, _receipt_sidecars(evidence)
    )
    final = _build_artifact(
        evidence,
        checks=checks,
        sources=sources,
        validation_receipts=[*affected, *terminal],
        current_receipt=final_receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        candidate=False,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "terminal_write", "before_atomic", path=output)
    atomic_json(candidate_path, final)
    atomic_json(output, final)
    _progress(started, "terminal_write", "after_atomic", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--skip-source-bytes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, verify_source_bytes=not args.skip_source_bytes)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = (
            cold_replay(args.independent_reduce, verify_source_bytes=False)
            if args.skip_source_bytes
            else independent_replay(args.independent_reduce)
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
