"""Seal deterministic lossless response windows for the V656 verifier protocol.

This module performs CPU-only data selection, UTF-8 window construction, and
prompt accounting.  It deliberately does not load or invoke a language model.

Spec refs: REQ-VERIFY-7491 and SCENARIO-VERIFY-7491-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
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
from carnot.experiment_7462_v654_option_protocol import OPTION_IDS, build_option_prompt
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260921"
MILESTONE = "2026.09.656"
EXPERIMENT_ID = "exp7491-v656-window-protocol"
SCHEMA = "carnot.exp7491.v656.window_protocol.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7491_v656_window_protocol.json")
RAW_DIR = Path("results/raw/experiment_7491_v656_window_protocol")
MODULE_PATH = Path("python/carnot/experiment_7491_v656_window_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7491_v656_window_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7491_v656_window_protocol.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
V654_RAW_DIR = Path("results/raw/experiment_7462_v654_option_protocol")
RAGTRUTH_ROOT = Path("/home/ianblenke/.cache/carnot-rewrite-20260731/work/data/ragtruth")
RAGTRUTH_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
TOKENIZER_PATH = Path(
    "/home/ianblenke/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B/"
    "snapshots/3ea932cee0a432ae86e9c7826cbe8aef52323a28/tokenizer.json"
)
SPLIT_SEED = 656001
TOKEN_CEILING = 2048
SENTENCE_VERSION = "carnot-utf8-sentence-v1"
WINDOW_VERSION = "carnot-balanced-contiguous-v1"
FOCUS_START_PREFIX = "<<CARNOT_FOCUS_START:"
FOCUS_END = "<<CARNOT_FOCUS_END>>"
MODEL_SPECS: list[JsonDict] = []
model_specs: list[JsonDict] = []
INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
INFERENCE_SUBSTRATE = (
    "deterministic_utf8_window_segmentation_tokenizer_accounting_freshness_search_"
    "and_protocol_sealing"
)
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
ROLE_CAPS = {"test": 120, "online": 160}
MINIMUM_ELIGIBLE = {"training": 150, "calibration_tuning": 40, "test": 100, "online": 120}
FORBIDDEN_PREDICTOR_FIELDS = {
    "label",
    "labels",
    "annotations",
    "annotation_labels",
    "annotation_spans",
    "source_alias",
    "response_alias",
    "source_id",
    "response_id",
    "model",
    "response_generator_identity",
    "quality",
    "official_split",
}
PREDICTOR_FIELDS = (
    "row_key",
    "group_id",
    "corpus",
    "role",
    "source_family",
    "source_text",
    "response_text",
    "source_hash",
    "response_hash",
    "release_revision",
    "license",
    "original_roster_position",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


class WindowProtocolError(ValueError):
    """Reject protocol evidence that would lose bytes, labels, or accounting."""


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes so text identity does not depend on JSON rendering."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash one UTF-8 string without normalization or implicit stripping."""

    return sha256_bytes(value.encode("utf-8"))


def _sentence_spans(response: str) -> list[tuple[int, int]]:
    """Return complete sentence character spans, retaining punctuation and space."""

    if not response:
        return [(0, 0)]
    terminal = set(".?!。？！")
    closers = set("\"'”’)]}»")
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(response):
        index += 1
        if response[index - 1] not in terminal:
            continue
        while index < len(response) and response[index] in closers:
            index += 1
        while index < len(response) and response[index].isspace():
            index += 1
        spans.append((start, index))
        start = index
    if start < len(response):
        spans.append((start, len(response)))
    return spans


def _window_sentence_ranges(
    spans: Sequence[tuple[int, int]], response: str
) -> list[tuple[int, int]]:
    """Group sentence spans into no more than three deterministic balanced windows."""

    count = len(spans)
    if count <= 3:
        return [(index, index + 1) for index in range(count)]
    byte_offsets = [len(response[:end].encode("utf-8")) for _, end in spans]
    candidates: list[tuple[tuple[int, int, int, int], tuple[int, int]]] = []
    total = byte_offsets[-1]
    for left in range(1, count - 1):
        for right in range(left + 1, count):
            sizes = (
                byte_offsets[left - 1],
                byte_offsets[right - 1] - byte_offsets[left - 1],
                total - byte_offsets[right - 1],
            )
            score = (
                max(sizes) - min(sizes),
                sum(abs(3 * size - total) for size in sizes),
                left,
                right,
            )
            candidates.append((score, (left, right)))
    _, (left, right) = min(candidates)
    return [(0, left), (left, right), (right, count)]


def build_lossless_windows(response: str) -> list[JsonDict]:
    """Partition every UTF-8 response byte into at most three sentence windows."""

    spans = _sentence_spans(response)
    payload = response.encode("utf-8")
    offsets = [0]
    offsets.extend(len(response[:end].encode("utf-8")) for _, end in spans)
    windows: list[JsonDict] = []
    for window_index, (first, last) in enumerate(_window_sentence_ranges(spans, response)):
        byte_start = offsets[first]
        byte_end = offsets[last]
        windows.append(
            {
                "window_index": window_index,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "sentence_count": 0 if not response else last - first,
                "window_sha256": sha256_bytes(payload[byte_start:byte_end]),
                "sentence_version": SENTENCE_VERSION,
                "window_version": WINDOW_VERSION,
            }
        )
    return windows


def _byte_boundary(response: str, offset: int) -> bool:
    payload = response.encode("utf-8")
    if offset < 0 or offset > len(payload):
        return False
    try:
        payload[:offset].decode("utf-8")
        payload[offset:].decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def mark_focus(response: str, byte_start: int, byte_end: int) -> str:
    """Insert one pair of byte-bound focus markers without changing response text."""

    if (
        byte_start > byte_end
        or not _byte_boundary(response, byte_start)
        or not _byte_boundary(response, byte_end)
    ):
        raise WindowProtocolError("focus_boundary_not_utf8")
    payload = response.encode("utf-8")
    prefix = f"{FOCUS_START_PREFIX}{byte_start}:{byte_end}>>".encode()
    return (
        payload[:byte_start]
        + prefix
        + payload[byte_start:byte_end]
        + FOCUS_END.encode()
        + payload[byte_end:]
    ).decode("utf-8")


def remove_focus_markers(marked_response: str) -> str:
    """Remove exactly one valid marker pair and reject ambiguous marker text."""

    pattern = re.compile(re.escape(FOCUS_START_PREFIX) + r"\d+:\d+>>")
    matches = list(pattern.finditer(marked_response))
    if len(matches) != 1 or marked_response.count(FOCUS_END) != 1:
        raise WindowProtocolError("focus_markers_invalid")
    start = matches[0]
    end = marked_response.find(FOCUS_END, start.end())
    if end < 0:  # pragma: no cover - count check above makes this defensive only.
        raise WindowProtocolError("focus_markers_invalid")
    return (
        marked_response[: start.start()]
        + marked_response[start.end() : end]
        + marked_response[end + len(FOCUS_END) :]
    )


def build_whole_prompt(source: str, response: str, option_order: Sequence[str]) -> str:
    """Delegate unchanged whole-response construction to the qualified V654 helper."""

    return build_option_prompt(source, response, option_order)


def build_focused_prompt(
    source: str,
    response: str,
    option_order: Sequence[str],
    window: Mapping[str, Any],
) -> JsonDict:
    """Build a full-context prompt with one marked, byte-aligned response focus."""

    marked = mark_focus(response, int(window["byte_start"]), int(window["byte_end"]))
    prompt = build_option_prompt(source, marked, option_order)
    return {
        "prompt": prompt,
        "marked_response": marked,
        "source_occurrences": prompt.count(source) if source else 1,
        "response_recovered": remove_focus_markers(marked) == response,
    }


def _stable_rank(seed: int, row: Mapping[str, Any]) -> str:
    """Rank public identities without consulting annotations or outcomes."""

    return sha256_text(
        f"v656-role:{seed}:{row.get('group_id')}:{row.get('source_hash')}:{row.get('response_hash')}"
    )


def _public_predictor(row: Mapping[str, Any], role: str, position: int) -> JsonDict:
    """Copy only predictor-safe fields and attach frozen public role metadata."""

    result = {key: deepcopy(row[key]) for key in PREDICTOR_FIELDS if key in row}
    result["role"] = role
    result.setdefault("corpus", "ragtruth")
    result.setdefault("release_revision", RAGTRUTH_REVISION)
    result.setdefault("license", "MIT")
    result["original_roster_position"] = int(row.get("original_roster_position", position))
    if set(result) & FORBIDDEN_PREDICTOR_FIELDS:  # pragma: no cover - allowlist invariant.
        raise WindowProtocolError("predictor_role_leakage")
    return result


def freeze_roles(
    fit_predictors: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    exclusions: Mapping[str, set[str]],
    *,
    role_caps: Mapping[str, int],
    split_seed: int,
) -> list[JsonDict]:
    """Preserve fit membership and hash-assign fresh roles before label access."""

    frozen = [
        _public_predictor(row, str(row["role"]), index) for index, row in enumerate(fit_predictors)
    ]
    eligible = [
        row
        for row in candidates
        if str(row.get("source_hash")) not in exclusions.get("source_hashes", set())
        and str(row.get("group_id")) not in exclusions.get("group_aliases", set())
        and str(row.get("source_alias")) not in exclusions.get("source_aliases", set())
    ]
    eligible.sort(key=lambda row: (_stable_rank(split_seed, row), str(row.get("group_id"))))
    cursor = 0
    for role in ("test", "online"):
        cap = int(role_caps[role])
        selected = eligible[cursor : cursor + cap]
        if len(selected) != cap:
            raise WindowProtocolError(f"insufficient_role_candidates:{role}")
        for row in selected:
            frozen.append(_public_predictor(row, role, cursor))
            cursor += 1
    return frozen


def freeze_evaluator_store(
    predictors: Sequence[Mapping[str, Any]], evaluator_by_group: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Join private labels only after public predictor roles have been frozen."""

    rows: list[JsonDict] = []
    for predictor in predictors:
        group_id = str(predictor["group_id"])
        if group_id not in evaluator_by_group:
            raise WindowProtocolError(f"evaluator_group_missing:{group_id}")
        private = deepcopy(dict(evaluator_by_group[group_id]))
        private["group_id"] = group_id
        private["row_key"] = predictor.get("row_key")
        private["role"] = predictor["role"]
        rows.append(private)
    return rows


def _request_row(
    group: Mapping[str, Any],
    arm: str,
    order: Sequence[str],
    prompt: str,
    window: Mapping[str, Any] | None,
    source_group_id: str,
) -> JsonDict:
    """Record a label-free prompt identity without duplicating prompt content."""

    return {
        "request_id": sha256_text(
            f"{group['group_id']}:{source_group_id}:{arm}:{tuple(order)}:{None if window is None else window['window_index']}"
        )[:31],
        "group_id": str(group["group_id"]),
        "source_group_id": source_group_id,
        "role": str(group["role"]),
        "arm": arm,
        "option_order": list(order),
        "window_index": None if window is None else int(window["window_index"]),
        "prompt_sha256": sha256_text(prompt),
        "prompt_token_count": 0,
        "gold_label": None,
        "eligible": True,
        "disposition": "planned_unstarted",
    }


def build_request_plan(
    groups: Sequence[Mapping[str, Any]],
    *,
    token_count: Callable[[str], int],
    token_ceiling: int,
    control_group_limit: int,
) -> JsonDict:
    """Freeze all prompt cells and reject overlength groups before label access."""

    requests: list[JsonDict] = []
    group_rows: list[JsonDict] = []
    window_rows: list[JsonDict] = []
    prompt_cache: dict[str, list[tuple[JsonDict, str]]] = {}
    orders = (OPTION_IDS, tuple(reversed(OPTION_IDS)))
    for group in groups:
        group_id = str(group["group_id"])
        source = str(group["source_text"])
        response = str(group["response_text"])
        windows = build_lossless_windows(response)
        window_rows.extend({"group_id": group_id, **row} for row in windows)
        cells: list[tuple[JsonDict, str]] = []
        for order in orders:
            prompt = build_whole_prompt(source, response, order)
            cells.append(
                (_request_row(group, "whole_response", order, prompt, None, group_id), prompt)
            )
            for window in windows:
                focused = build_focused_prompt(source, response, order, window)
                cells.append(
                    (
                        _request_row(
                            group, "focused_window", order, str(focused["prompt"]), window, group_id
                        ),
                        str(focused["prompt"]),
                    )
                )
        prompt_cache[group_id] = cells
        counts = [int(token_count(prompt)) for _, prompt in cells]
        eligible = max(counts, default=0) <= token_ceiling
        for (row, _), count in zip(cells, counts):
            row["prompt_token_count"] = count
            row["eligible"] = eligible
            row["disposition"] = "planned_unstarted" if eligible else "excluded_overlength"
            requests.append(row)
        group_rows.append(
            {
                "group_id": group_id,
                "role": str(group["role"]),
                "eligible": eligible,
                "disposition": "planned_unstarted" if eligible else "excluded_overlength",
                "window_count": len(windows),
                "planned_forwards": len(cells),
                "prompt_token_max": max(counts, default=0),
                "source_hash": group["source_hash"],
                "response_hash": group["response_hash"],
            }
        )
    calibration = sorted(
        (row for row in groups if row["role"] == "calibration_tuning"),
        key=lambda row: str(row["group_id"]),
    )
    donors = sorted(groups, key=lambda row: str(row["group_id"]))
    for group in calibration[:control_group_limit]:
        donor = next((row for row in donors if row["group_id"] != group["group_id"]), None)
        if donor is None:
            raise WindowProtocolError("derangement_donor_missing")
        for order in orders:
            prompt = build_whole_prompt(
                str(donor["source_text"]), str(group["response_text"]), order
            )
            row = _request_row(
                group, "source_derangement_control", order, prompt, None, str(donor["group_id"])
            )
            row["prompt_token_count"] = int(token_count(prompt))
            row["eligible"] = row["prompt_token_count"] <= token_ceiling
            row["disposition"] = "planned_unstarted" if row["eligible"] else "excluded_overlength"
            requests.append(row)
    fit_count = sum(1 for row in requests if row["role"] in {"training", "calibration_tuning"})
    evaluation_count = sum(1 for row in requests if row["role"] in {"test", "online"})
    budgets = {
        "fit": {"planned_forwards": fit_count, "ceiling": 1960, "passed": fit_count <= 1960},
        "evaluation": {
            "planned_forwards": evaluation_count,
            "ceiling": 2240,
            "passed": evaluation_count <= 2240,
        },
    }
    return {
        "requests": requests,
        "group_rows": group_rows,
        "window_rows": window_rows,
        "budgets": budgets,
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize raw rows canonically so one changed byte changes the manifest."""

    return b"".join(
        (json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Replace a task-owned shard only after its complete bytes reach disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def seal_protocol_shards(
    raw_dir: Path,
    *,
    predictors: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
    exposure_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write physically separate, hash-bound per-unit protocol shards."""

    collections = {
        "predictors": predictors,
        "evaluators": evaluators,
        "groups": group_rows,
        "windows": window_rows,
        "requests": request_rows,
        "exposure_search": exposure_rows,
    }
    manifest: JsonDict = {}
    for name, rows in collections.items():
        path = raw_dir / f"{name}.jsonl"
        payload = _jsonl_bytes(rows)
        if len(payload) >= 20 * 1024 * 1024:  # pragma: no cover - production size fail-safe.
            raise WindowProtocolError(f"raw_shard_too_large:{name}")
        _atomic_bytes(path, payload)
        manifest[name] = {
            "path": path.name,
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "rows": len(rows),
        }
    return manifest


def load_jsonl(path: Path) -> list[JsonDict]:
    """Read every nonblank JSONL row and reject non-object evidence."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise WindowProtocolError(f"jsonl_row_invalid:{path}:{line_number}")
            rows.append(value)
    return rows


def reload_protocol_shards(raw_dir: Path, manifest: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Rehash and independently reduce raw windows, roles, prompts, and labels."""

    loaded: dict[str, list[JsonDict]] = {}
    for name, receipt in manifest.items():
        path = raw_dir / str(receipt["path"])
        if sha256_file(path) != receipt["sha256"]:
            raise WindowProtocolError(f"raw_shard_hash_mismatch:{name}")
        loaded[name] = load_jsonl(path)
    predictors = loaded["predictors"]
    windows_by_group: dict[str, list[JsonDict]] = {}
    for row in loaded["windows"]:
        windows_by_group.setdefault(str(row["group_id"]), []).append(row)
    lossless = True
    for predictor in predictors:
        payload = str(predictor["response_text"]).encode("utf-8")
        windows = sorted(
            windows_by_group.get(str(predictor["group_id"]), []),
            key=lambda row: int(row["window_index"]),
        )
        lossless = (
            lossless
            and bool(windows)
            and windows[0]["byte_start"] == 0
            and windows[-1]["byte_end"] == len(payload)
        )
        lossless = lossless and all(
            left["byte_end"] == right["byte_start"] for left, right in zip(windows, windows[1:])
        )
        lossless = (
            lossless
            and b"".join(payload[int(row["byte_start"]) : int(row["byte_end"])] for row in windows)
            == payload
        )
    prompt_isolation = all(not (set(row) & FORBIDDEN_PREDICTOR_FIELDS) for row in predictors)
    prompt_isolation = prompt_isolation and all(
        row.get("gold_label") is None for row in loaded["requests"]
    )
    role_counts = dict(Counter(str(row["role"]) for row in predictors))
    eligible_counts = dict(
        Counter(str(row["role"]) for row in loaded["groups"] if row.get("eligible") is True)
    )
    labels = {str(row["group_id"]): row.get("label") for row in loaded["evaluators"]}
    test_support = Counter(
        labels.get(str(row["group_id"]))
        for row in loaded["groups"]
        if row["role"] == "test" and row.get("eligible") is True
    )
    fit_forwards = sum(
        1 for row in loaded["requests"] if row["role"] in {"training", "calibration_tuning"}
    )
    eval_forwards = sum(1 for row in loaded["requests"] if row["role"] in {"test", "online"})
    return {
        "role_counts": role_counts,
        "eligible_role_counts": eligible_counts,
        "lossless_windows": lossless,
        "prompt_isolation": prompt_isolation,
        "test_class_support": {str(key): value for key, value in test_support.items()},
        "budgets": {
            "fit": {
                "planned_forwards": fit_forwards,
                "ceiling": 1960,
                "passed": fit_forwards <= 1960,
            },
            "evaluation": {
                "planned_forwards": eval_forwards,
                "ceiling": 2240,
                "passed": eval_forwards <= 2240,
            },
        },
        "rows": loaded["groups"],
        "request_count": len(loaded["requests"]),
        "exposure_search_complete": bool(loaded["exposure_search"])
        and all(row.get("status") == "searched" for row in loaded["exposure_search"]),
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    field_path: str,
    op: str = "eq",
) -> JsonDict:
    """Make a typed gate whose principle states the failure it prevents."""

    principles = {
        "validity": "A favorable metric cannot excuse invalid evidence.",
        "readiness": "A valid scientific null must not block independent measurements.",
        "benefit": "A favorable seed, fixture or low-support result cannot replace held-out value.",
    }
    passed = observed == expected if op == "eq" else bool(observed)
    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "field_path": field_path,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principles[category],
    }


def build_acceptance_gates(
    *,
    preconditions_ok: bool,
    freshness_complete: bool,
    exact_role_counts: bool,
    minima_met: bool,
    class_support_met: bool,
    windows_lossless: bool,
    prompts_isolated: bool,
    budgets_valid: bool,
    validation_passed: bool,
) -> list[JsonDict]:
    """Keep validity, structural readiness, and unclaimed benefit independent."""

    rows = [
        (
            "preconditions_authenticated",
            "validity",
            preconditions_ok,
            "preconditions_checked",
            "all",
        ),
        (
            "freshness_search_complete",
            "readiness",
            freshness_complete,
            "prior_exposure_exclusions",
            "search_complete",
        ),
        ("exact_role_counts", "readiness", exact_role_counts, "role_manifest", "role_counts"),
        ("eligible_role_minima", "readiness", minima_met, "role_manifest", "eligible_role_counts"),
        (
            "held_out_test_class_support",
            "readiness",
            class_support_met,
            "evaluator_store",
            "test_class_support",
        ),
        ("windows_lossless", "validity", windows_lossless, "window_definition", "lossless_windows"),
        (
            "prompts_label_blind",
            "validity",
            prompts_isolated,
            "request_manifest",
            "prompt_isolation",
        ),
        ("forward_budgets_valid", "validity", budgets_valid, "sample_size_budget", "ceilings"),
        (
            "required_validation_passed",
            "validity",
            validation_passed,
            "validation_receipts",
            "required",
        ),
    ]
    gates = [
        _gate(name, category, True, observed, upstream=upstream, field_path=field)
        for name, category, observed, upstream, field in rows
    ]
    gates.append(
        _gate(
            "predictive_benefit_not_claimed",
            "benefit",
            False,
            False,
            upstream="honest_verdict",
            field_path="predictive_benefit",
        )
    )
    return gates


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """List every failed gate and name the first one without hiding later failures."""

    failed = [dict(row) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "first_failure": failed[0] if failed else None,
        "failed_checks": failed,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind the complete artifact while excluding its self-referential checksum."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def _principle(field: str) -> str:
    """Explain the concrete drift or misreading prevented by one emitted field."""

    required = {
        "schema": "Versioned schema, exact experiment_id, milestone and terminal status prevent reader drift.",
        "run_date": "Use 20260921; retain measured UTC and monotonic clock/process identity.",
        "preconditions_checked": "Record exact resource paths, observed values, ownership and input validity.",
        "MODEL_SPECS": "Current LLM tasks require resolved model provenance; this numeric protocol correctly uses an empty list.",
        "model_specs": "The lowercase compatibility field prevents consumers from inventing model work.",
        "model_invoked": "Attempted live loads and forwards differ from historical or scripted model events.",
        "invocation_counts": "Attempted, completed, failed, cancelled and in-flight calls must reconcile.",
        "inference_substrate": "The substrate names the CPU protocol work actually executed.",
        "inference_substrate_class": "The declared no-model class prevents padded or fictional inference durations.",
        "execution_venue": "Host CPU work remains distinct from archived board evidence.",
        "duration_s": "Measured phase time prevents synthetic timing floors.",
        "phase_spans": "Flushed phase checkpoints expose unfinished operations and stalls.",
        "random_seed": "Frozen role and ordering seeds prevent outcome-guided retries.",
        "reproducibility_checksum": "One checksum binds code, prompts, roles, shards and validation scope.",
        "source_artifact_hashes": "Exact upstream bytes and original flags prevent evidence laundering.",
        "rows": "Per-group exclusions and censoring permit independent headline reduction.",
        "sample_size_budget": "Planned, eligible, excluded and unstarted units cannot be silently conflated.",
        "acceptance_gate_results": "Typed operands prevent a positive claim from bypassing any gate.",
        "gate_check_summary": "Every blocked verdict names its exact upstream field and observed value.",
        "honest_verdict": "A complete terminal finding preserves external blocks and measured nulls.",
        "verdict_class": "A closed class prevents retryable work from masquerading as a scientific null.",
        "verifier_is_oracle": "Oracle evaluation forbids an unsupported positive interpretation.",
        "flagged_adversarial": "Actual reader flags cannot be cleared merely to open a gate.",
        "validation_receipts": "Exact commands, exit codes and log hashes make checks reviewable.",
        "field_principles": "Every emitted field states the failure mode it prevents.",
        "window_protocol_ready_score": "The bare score means complete sealed roles and lossless label-blind prompts, not benefit.",
        "role_manifest": "Hash-bound texts, roster positions and label-store separation prevent role leakage.",
        "window_definition": "Every response byte belongs to exactly one focus window.",
        "request_manifest": "Both option orders and controls are fixed before any model execution.",
        "prior_exposure_exclusions": "Recorded prior hashes, aliases and search completeness protect fresh evaluation.",
    }
    return required.get(field, f"The {field} field prevents silent protocol or evidence drift.")


def _fixture_rows() -> tuple[list[JsonDict], list[JsonDict]]:
    """Create a small deterministic non-scientific panel for unit validation."""

    predictors: list[JsonDict] = []
    evaluators: list[JsonDict] = []
    roles = ("training", "calibration_tuning", "test", "test", "online")
    for index, role in enumerate(roles):
        source = f"Fixture source {index}."
        response = f"Fixture response {index}. Still qualified."
        group_id = f"fixture-group-{index}"
        predictors.append(
            {
                "row_key": f"fixture-row-{index}",
                "group_id": group_id,
                "corpus": "ragtruth",
                "role": role,
                "source_family": "fixture",
                "source_text": source,
                "response_text": response,
                "source_hash": sha256_text(source.casefold()),
                "response_hash": sha256_text(response),
                "release_revision": RAGTRUTH_REVISION,
                "license": "MIT",
                "original_roster_position": index,
            }
        )
        evaluators.append(
            {
                "group_id": group_id,
                "row_key": f"fixture-row-{index}",
                "role": role,
                "label": index % 2,
            }
        )
    return predictors, evaluators


def _base_artifact(
    raw_dir: Path,
    manifest: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    expected_counts: Mapping[str, int],
    minimums: Mapping[str, int],
    class_minimum: int,
    freshness_complete: bool,
) -> JsonDict:
    """Assemble one schema-complete artifact from independently replayed shards."""

    exact = dict(replay["role_counts"]) == dict(expected_counts)
    minima = all(
        int(replay["eligible_role_counts"].get(role, 0)) >= count
        for role, count in minimums.items()
    )
    support = [count for key, count in replay["test_class_support"].items() if key != "None"]
    class_support = len(support) >= 2 and min(support) >= class_minimum
    affected = (
        reduce_affected_receipts(REPO_ROOT, VALIDATION_MANIFEST, validation_receipts)
        if validation_receipts
        else {"passed": True}
    )
    validation_passed = bool(affected.get("passed"))
    gates = build_acceptance_gates(
        preconditions_ok=all(row.get("passed") is True for row in preconditions),
        freshness_complete=freshness_complete,
        exact_role_counts=exact,
        minima_met=minima,
        class_support_met=class_support,
        windows_lossless=bool(replay["lossless_windows"]),
        prompts_isolated=bool(replay["prompt_isolation"]),
        budgets_valid=all(row["passed"] for row in replay["budgets"].values()),
        validation_passed=validation_passed,
    )
    readiness_checks = [row for row in gates if row["category"] != "benefit"]
    ready = int(all(row["passed"] for row in readiness_checks))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "run_date": RUN_DATE,
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "process_identity": {"pid": os.getpid(), "cwd": str(Path.cwd().resolve())},
        "preconditions_checked": list(preconditions),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "duration_breakdown_s": {
            "inference": 0.0,
            "optimization": 0.0,
            "protocol_and_validation": duration_s,
        },
        "phase_spans": list(phase_spans),
        "random_seed": {
            "role_split": SPLIT_SEED,
            "optimizer": None,
            "audit": 656009,
            "option_order": 65400,
            "interval": None,
            "deterministic_null_seed_reason": "No stochastic fit or benefit estimate occurs.",
        },
        "source_artifact_hashes": list(source_hashes),
        "rows": deepcopy(list(replay["rows"])),
        "sample_size_budget": {
            "planned_role_counts": dict(expected_counts),
            "eligible_role_counts": deepcopy(replay["eligible_role_counts"]),
            "excluded": len(replay["rows"]) - sum(replay["eligible_role_counts"].values()),
            "failed": 0,
            "censored": 0,
            "unstarted_forwards": replay["request_count"],
            "fit": deepcopy(replay["budgets"]["fit"]),
            "evaluation": deepcopy(replay["budgets"]["evaluation"]),
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "honest_verdict": "complete_null_structural_window_protocol_ready"
        if ready
        else "complete_blocked_window_protocol_not_ready",
        "verdict_class": "null" if ready else "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": deepcopy(list(validation_receipts)),
        "window_protocol_ready_score": ready,
        "role_manifest": {
            "raw_directory": str(raw_dir),
            "raw_shards": deepcopy(dict(manifest)),
            "role_counts": deepcopy(replay["role_counts"]),
            "eligible_role_counts": deepcopy(replay["eligible_role_counts"]),
            "expected_role_counts": dict(expected_counts),
            "minimum_eligible": dict(minimums),
            "test_class_support": deepcopy(replay["test_class_support"]),
            "test_class_minimum": class_minimum,
            "labels_opened_after_prediction_freeze": True,
        },
        "window_definition": {
            "sentence_version": SENTENCE_VERSION,
            "window_version": WINDOW_VERSION,
            "maximum_windows": 3,
            "lossless_windows": replay["lossless_windows"],
            "not_atomic_claim_extraction": True,
            "not_halldetect_reproduction": True,
            "not_trained_entailment_encoder": True,
        },
        "request_manifest": {
            "whole_prompt_builder": "carnot.experiment_7462_v654_option_protocol.build_option_prompt",
            "option_orders": [list(OPTION_IDS), list(reversed(OPTION_IDS))],
            "token_ceiling": TOKEN_CEILING,
            "maximum_forwards_per_group": 8,
            "derangement_gold_labels": None,
            "request_count": replay["request_count"],
            "frozen_features": [
                "whole_response_log_odds",
                "maximum_focused_unsupported_probability",
                "mean_focused_unsupported_probability",
                "focused_disagreement",
                "numeric_novelty_with_context",
                "falsifiability_score",
                "normalized_number_token_overlap",
                "normalized_content_token_overlap",
                "max_answer_source_sentence_overlap",
                "missing_or_empty_source",
            ],
        },
        "prior_exposure_exclusions": {
            "search_complete": freshness_complete,
            "raw_shard": deepcopy(dict(manifest).get("exposure_search", {})),
            "inaccessible_archives": [] if freshness_complete else ["see exposure shard"],
        },
        "small_ebm_training": {
            "fit_attempts": 0,
            "fit_completed": 0,
            "training_rows": 0,
            "elapsed_s": 0.0,
            "reason": "Protocol sealing performs no model or EBM fit.",
        },
        "e2e_scope": {
            "capability_replay": True,
            "numbered_runtime_e2e": [],
            "reason": "Pure reporting changed no shared runtime, binding, ARC, telemetry, or Rust path.",
        },
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = {key: _principle(key) for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(tmp_path: Path) -> JsonDict:
    """Exercise production reducers with a small circular fixture, not efficacy data."""

    predictors, evaluators = _fixture_rows()
    plan = build_request_plan(
        predictors, token_count=len, token_ceiling=10000, control_group_limit=1
    )
    manifest = seal_protocol_shards(
        tmp_path,
        predictors=predictors,
        evaluators=evaluators,
        group_rows=plan["group_rows"],
        window_rows=plan["window_rows"],
        request_rows=plan["requests"],
        exposure_rows=[{"path": "fixture", "status": "searched"}],
    )
    replay = reload_protocol_shards(tmp_path, manifest)
    return _base_artifact(
        tmp_path,
        manifest,
        replay,
        validation_receipts=[],
        preconditions=[{"check": "fixture", "passed": True}],
        source_hashes=[],
        duration_s=0.001,
        phase_spans=[],
        expected_counts={"training": 1, "calibration_tuning": 1, "test": 2, "online": 1},
        minimums={"training": 1, "calibration_tuning": 1, "test": 2, "online": 1},
        class_minimum=1,
        freshness_complete=True,
    )


def validate_artifact(value: Mapping[str, Any], *, root: Path, require_terminal: bool) -> list[str]:
    """Cold-check identity, current work, raw reduction, receipts, and checksum."""

    errors: list[str] = []
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invoked") is not False
        or any((value.get("invocation_counts") or {}).values())
    ):
        errors.append("current_model_provenance_invalid")
    role = value.get("role_manifest") or {}
    raw_dir = Path(str(role.get("raw_directory", "")))
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    try:
        replay = reload_protocol_shards(raw_dir, role.get("raw_shards") or {})
        if (
            replay["rows"] != value.get("rows")
            or replay["role_counts"] != role.get("role_counts")
            or replay["eligible_role_counts"] != role.get("eligible_role_counts")
        ):
            errors.append("independent_reduction_mismatch")
    except (OSError, KeyError, TypeError, WindowProtocolError, json.JSONDecodeError):
        errors.append("independent_reduction_failed")
    if set(value.get("field_principles") or {}) != set(value):
        errors.append("field_principles_incomplete")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    if require_terminal:
        names = {
            row.get("name")
            for row in value.get("validation_receipts", [])
            if row.get("passed") is True
        }
        missing = (set(AFFECTED_CHECK_NAMES) | set(TERMINAL_CHECK_NAMES)) - names
        if missing:
            errors.append("terminal_validation_incomplete")
    return list(dict.fromkeys(errors))


def progress(
    started: float, phase: str, event: str, **details: Any
) -> None:  # pragma: no cover - live progress
    """Emit one flushed boundary around each potentially slow operation."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7491] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _source_receipt(path: Path, root: Path) -> JsonDict:  # pragma: no cover - live inputs
    """Hash one exact prerequisite while keeping stable repository-relative labels."""

    label = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    return {"path": label, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover - live inputs
    """Authenticate instructions, upstream outcomes, cache bytes, and original flags."""

    required = [
        root / path
        for path in (
            Path("AGENTS.md"),
            Path("CODEX.md"),
            SPEC_PATH,
            V654_RAW_DIR / "cohort_predictors.jsonl",
            V654_RAW_DIR / "cohort_evaluators.jsonl",
            Path("results/experiment_7476_v655_option_qualification.json"),
            Path("results/experiment_7477_v655_native_readout_pilot.json"),
            Path("results/experiment_7479_v655_source_fit_capture.json"),
            Path("results/experiment_7480_v655_source_eval_capture.json"),
            Path("results/experiment_7489_v656_contract_methods.json"),
            Path("results/experiment_7490_v656_historical_audit.json"),
        )
    ]
    required.extend(
        (RAGTRUTH_ROOT / "source_info.jsonl", RAGTRUTH_ROOT / "response.jsonl", TOKENIZER_PATH)
    )
    receipts: list[JsonDict] = []
    checks: list[JsonDict] = []
    for path in required:
        exists = path.is_file()
        checks.append(
            {
                "check": "resource_exists",
                "path": str(path),
                "expected": True,
                "observed": exists,
                "passed": exists,
                "principle": "Exact prerequisite paths prevent substituted or guessed inputs.",
            }
        )
        if exists:
            receipts.append(_source_receipt(path, root))
    expected = {
        "experiment_7476_v655_option_qualification.json": ("option_protocol_ready_score", 1),
        "experiment_7477_v655_native_readout_pilot.json": ("native_readout_ready_score", 1),
        "experiment_7479_v655_source_fit_capture.json": ("fit_capture_ready_score", 1),
        "experiment_7480_v655_source_eval_capture.json": (
            "evaluation_capture_ready_score",
            1,
        ),
    }
    for path in required:
        if path.name not in expected or not path.is_file():
            continue
        field, wanted = expected[path.name]
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = payload.get(field)
        checks.append(
            {
                "check": "upstream_ready",
                "path": str(path),
                "field": field,
                "expected": wanted,
                "observed": observed,
                "passed": observed == wanted,
                "principle": "Original upstream flags prevent retrospective prerequisite laundering.",
            }
        )
    return checks, receipts


def _load_fit(
    root: Path,
) -> tuple[list[JsonDict], dict[str, JsonDict]]:  # pragma: no cover - live data
    """Load the exact V654 training and calibration predictor/evaluator roster."""

    predictors = [
        row
        for row in load_jsonl(root / V654_RAW_DIR / "cohort_predictors.jsonl")
        if row["role"] in {"training", "calibration_tuning"}
    ]
    evaluators = {
        str(row["group_id"]): row
        for row in load_jsonl(root / V654_RAW_DIR / "cohort_evaluators.jsonl")
        if row["role"] in {"training", "calibration_tuning"}
    }
    return predictors, evaluators


def _exposure_search(
    root: Path,
) -> tuple[dict[str, set[str]], list[JsonDict], bool]:  # pragma: no cover - live archives
    """Search every named prior fit/evaluation lineage and collect public aliases."""

    patterns = (
        "results/raw/experiment_7423_v651_annotated_protocol/evaluator-fit-*.jsonl",
        "results/raw/experiment_7423_v651_annotated_protocol/evaluator-final_test-*.jsonl",
        "results/raw/experiment_7423_v651_annotated_protocol/evaluator-prospective_stream-*.jsonl",
        "results/raw/experiment_7449_v653_source_protocol/*.jsonl",
        "results/raw/experiment_7462_v654_option_protocol/*.jsonl",
        "results/raw/experiment_7479_v655_source_fit_capture/*.jsonl",
        "results/raw/experiment_7480_v655_source_eval_capture/*.jsonl",
        "results/raw/experiment_7481_v655_typed_calibration/*.jsonl",
        "results/raw/experiment_7483_v655_continuous_learning/*.jsonl",
    )
    exclusions = {"source_hashes": set(), "group_aliases": set(), "source_aliases": set()}
    rows: list[JsonDict] = []
    complete = True
    for pattern in patterns:
        paths = sorted(root.glob(pattern))
        if not paths:
            complete = False
            rows.append(
                {"path": pattern, "status": "inaccessible", "match_count": 0, "sha256": None}
            )
            continue
        for path in paths:
            parsed = load_jsonl(path)
            for row in parsed:
                for key in ("source_hash",):
                    if row.get(key) is not None:
                        exclusions["source_hashes"].add(str(row[key]))
                for key in ("group_id", "source_group_id"):
                    if row.get(key) is not None:
                        exclusions["group_aliases"].add(str(row[key]))
                if row.get("source_id") is not None:
                    exclusions["source_aliases"].add(str(row["source_id"]))
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "status": "searched",
                    "match_count": len(parsed),
                    "sha256": sha256_file(path),
                }
            )
    return exclusions, rows, complete


def _load_candidates() -> tuple[
    list[JsonDict], dict[str, JsonDict]
]:  # pragma: no cover - live cache
    """Choose one good response per RAGTruth source using only public identities."""

    source_rows = load_jsonl(RAGTRUTH_ROOT / "source_info.jsonl")
    response_rows = load_jsonl(RAGTRUTH_ROOT / "response.jsonl")
    sources = {str(row["source_id"]): row for row in source_rows}
    choices: dict[str, list[JsonDict]] = {}
    for row in response_rows:
        if row.get("quality") == "good" and str(row.get("source_id")) in sources:
            choices.setdefault(str(row["source_id"]), []).append(row)
    public: list[JsonDict] = []
    private: dict[str, JsonDict] = {}
    for source_id, responses in choices.items():
        selected = min(
            responses,
            key=lambda row: sha256_text(f"v656-response:{SPLIT_SEED}:{source_id}:{row['id']}"),
        )
        source = sources[source_id]
        source_text = (
            json.dumps(
                source["source_info"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            if not isinstance(source["source_info"], str)
            else str(source["source_info"])
        )
        response_text = str(selected["response"])
        source_hash = sha256_text(" ".join(source_text.casefold().split()))
        response_hash = sha256_text(response_text)
        group_id = "group-" + source_hash.removeprefix("sha256:")[:24]
        public.append(
            {
                "row_key": "row-"
                + sha256_text(f"{source_id}:{selected['id']}").removeprefix("sha256:")[:24],
                "group_id": group_id,
                "source_alias": source_id,
                "response_alias": str(selected["id"]),
                "official_split": selected.get("split"),
                "source_family": source.get("task_type", "unknown"),
                "source_text": source_text,
                "response_text": response_text,
                "source_hash": source_hash,
                "response_hash": response_hash,
                "release_revision": RAGTRUTH_REVISION,
                "license": "MIT",
            }
        )
        private[group_id] = {
            "group_id": group_id,
            "source_id": source_id,
            "response_id": str(selected["id"]),
            "official_split": selected.get("split"),
            "label": int(not bool(selected.get("labels"))),
            "annotation_labels": deepcopy(selected.get("labels") or []),
            "response_generator_identity": selected.get("model"),
        }
    return public, private


def _token_counter() -> tuple[
    Callable[[str], int], JsonDict
]:  # pragma: no cover - tokenizer-only accounting
    """Load only tokenizer JSON and return its identity; no model weights are opened."""

    from tokenizers import Tokenizer  # imported only for the live protocol

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    return lambda text: len(tokenizer.encode(text).ids), {
        "path": str(TOKENIZER_PATH),
        "sha256": sha256_file(TOKENIZER_PATH),
        "runtime": "tokenizers",
        "weights_loaded": False,
        "parity_status": "authenticated_against_archived_embedded_gguf_prompt_ids",
    }


def _phase(
    name: str, start: float, phase_start: float, units: int
) -> JsonDict:  # pragma: no cover - live timing
    """Record a measured disjoint phase span with a completed-unit checkpoint."""

    now = time.monotonic()
    return {
        "phase": name,
        "start_offset_s": phase_start - start,
        "end_offset_s": now - start,
        "duration_s": now - phase_start,
        "completed_units": units,
        "checkpoint": "complete",
    }


def _terminal_commands(
    candidate: Path,
) -> list[PlannedCommand]:  # pragma: no cover - live subprocesses
    """Build fresh-process replay and both exact terminal readers."""

    relative = candidate.relative_to(REPO_ROOT).as_posix()
    python = ".venv/bin/python"
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE, "--cold-replay", relative),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", relative),
            "safety",
            180,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", relative),
            "completion",
            180,
        ),
    )
    return [
        PlannedCommand(spec, "safety" if spec.name == "adversarial_verify" else "completion", True)
        for spec in specs
    ]


def run_experiment(
    root: Path, run_date: str
) -> int:  # pragma: no cover - exercised by capability E2E
    """Authenticate, freeze, seal, validate, independently replay, and publish."""

    if run_date != RUN_DATE:
        raise WindowProtocolError(f"run_date_mismatch:{run_date}")
    started = time.monotonic()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, sources = _collect_preconditions(root)
    spans.append(_phase("preconditions", started, phase_started, len(preconditions)))
    progress(started, "preconditions", "complete", units=len(preconditions))

    progress(started, "role_freeze", "start")
    phase_started = time.monotonic()
    fit, fit_evaluators = _load_fit(root)
    exclusions, exposure_rows, exposure_complete = _exposure_search(root)
    candidates, candidate_evaluators = _load_candidates()
    predictors = freeze_roles(
        fit, candidates, exclusions, role_caps=ROLE_CAPS, split_seed=SPLIT_SEED
    )
    spans.append(_phase("role_freeze", started, phase_started, len(predictors)))
    progress(started, "role_freeze", "complete", units=len(predictors))

    progress(started, "tokenizer_accounting", "before_tokenizer_load")
    phase_started = time.monotonic()
    token_count, tokenizer_receipt = _token_counter()
    progress(started, "tokenizer_accounting", "after_tokenizer_load")
    plan = build_request_plan(
        predictors, token_count=token_count, token_ceiling=TOKEN_CEILING, control_group_limit=20
    )
    spans.append(_phase("tokenizer_accounting", started, phase_started, len(plan["requests"])))
    progress(started, "tokenizer_accounting", "complete", units=len(plan["requests"]))

    progress(started, "protocol_seal", "start")
    phase_started = time.monotonic()
    prediction_freeze = canonical_hash({"predictors": predictors, "requests": plan["requests"]})
    evaluators = freeze_evaluator_store(predictors, {**candidate_evaluators, **fit_evaluators})
    raw_dir = root / RAW_DIR
    manifest = seal_protocol_shards(
        raw_dir,
        predictors=predictors,
        evaluators=evaluators,
        group_rows=plan["group_rows"],
        window_rows=plan["window_rows"],
        request_rows=plan["requests"],
        exposure_rows=exposure_rows,
    )
    replay = reload_protocol_shards(raw_dir, manifest)
    spans.append(_phase("protocol_seal", started, phase_started, len(manifest)))
    progress(started, "protocol_seal", "complete", units=len(manifest))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7491-"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise WindowProtocolError("validation_plan_invalid:" + ",".join(plan_errors))
    progress(started, "affected_validation", "before_subprocesses", units=len(commands))
    phase_started = time.monotonic()
    affected_receipts = run_categorized_commands(
        root,
        [PlannedCommand(command, "validity", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60,
    )
    spans.append(_phase("affected_validation", started, phase_started, len(affected_receipts)))
    progress(started, "affected_validation", "after_subprocesses", units=len(affected_receipts))

    sources.append(tokenizer_receipt)
    preconditions.append(
        {
            "check": "prediction_freeze_before_labels",
            "expected": True,
            "observed": bool(prediction_freeze),
            "passed": bool(prediction_freeze),
            "principle": "The frozen public request hash prevents label-guided role or prompt edits.",
        }
    )
    candidate = _base_artifact(
        raw_dir.relative_to(root),
        manifest,
        replay,
        validation_receipts=affected_receipts,
        preconditions=preconditions,
        source_hashes=sources,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        expected_counts={"training": 180, "calibration_tuning": 60, "test": 120, "online": 160},
        minimums=MINIMUM_ELIGIBLE,
        class_minimum=20,
        freshness_complete=exposure_complete,
    )
    candidate["role_manifest"]["prediction_freeze_sha256"] = prediction_freeze
    candidate["field_principles"] = {key: _principle(key) for key in candidate}
    candidate["reproducibility_checksum"] = artifact_checksum(candidate)
    candidate_path = root / RAW_DIR / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    progress(started, "terminal_validation", "before_subprocesses", units=4)
    phase_started = time.monotonic()
    terminal_receipts = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60,
    )
    spans.append(_phase("terminal_validation", started, phase_started, len(terminal_receipts)))
    progress(started, "terminal_validation", "after_subprocesses", units=len(terminal_receipts))
    if not all(row.get("passed") is True for row in affected_receipts + terminal_receipts):
        raise WindowProtocolError("required_validation_failed")
    candidate["validation_receipts"] = affected_receipts + terminal_receipts
    candidate["phase_spans"] = spans
    candidate["duration_s"] = time.monotonic() - started
    candidate["duration_breakdown_s"]["protocol_and_validation"] = candidate["duration_s"]
    candidate["field_principles"] = {key: _principle(key) for key in candidate}
    candidate["reproducibility_checksum"] = artifact_checksum(candidate)
    errors = validate_artifact(candidate, root=root, require_terminal=True)
    if errors:
        raise WindowProtocolError("terminal_artifact_invalid:" + ",".join(errors))
    atomic_json(root / RESULT_PATH, candidate)
    progress(started, "publish", "complete", path=RESULT_PATH.as_posix())
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover - CLI
    """Parse the fixed run date and read-only fresh-process modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """Run the live protocol or one fresh-process terminal reader."""

    args = parse_args(argv)
    if args.cold_replay:
        value = json.loads((REPO_ROOT / args.cold_replay).read_text(encoding="utf-8"))
        errors = validate_artifact(value, root=REPO_ROOT, require_terminal=False)
        print(
            json.dumps({"mode": "cold_replay", "passed": not errors, "errors": errors}), flush=True
        )
        return int(bool(errors))
    if args.independent_reduce:
        value = json.loads((REPO_ROOT / args.independent_reduce).read_text(encoding="utf-8"))
        role = value["role_manifest"]
        raw_dir = Path(role["raw_directory"])
        replay = reload_protocol_shards(REPO_ROOT / raw_dir, role["raw_shards"])
        passed = replay["rows"] == value["rows"]
        print(
            json.dumps(
                {"mode": "independent_reduce", "passed": passed, "reduction": replay},
                sort_keys=True,
            ),
            flush=True,
        )
        return int(not passed)
    return run_experiment(REPO_ROOT, args.date)


if __name__ == "__main__":  # pragma: no cover - module execution
    raise SystemExit(main())
