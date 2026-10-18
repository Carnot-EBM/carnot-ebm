"""Replay the immutable relation corpus without invoking a model.

The source acquisition is evidence, including its failed attempts and its
quarantine flag. This reducer checks those bytes and derives new summaries. It
does not repair the source file or treat a clean reducer as proof of semantics.

Spec refs: REQ-REPORT-6912 and SCENARIO-REPORT-6912-*.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6900_authentic_anchored_relation_corpus as producer


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6912_alias_safe_relation_corpus_reducer.json")
SOURCE_PATHS = {
    "exp6899": Path("results/experiment_6899_live_relation_acquisition_canary.json"),
    "exp6900": Path("results/experiment_6900_authentic_anchored_relation_corpus.json"),
    "producer_module": Path("python/carnot/experiment_6900_authentic_anchored_relation_corpus.py"),
    "producer_entrypoint": Path(
        "scripts/experiments/experiment_6900_authentic_anchored_relation_corpus.py"
    ),
}
EXPECTED_SOURCE_HASHES = {
    "exp6899": "sha256:7c24282585cf771a56af627d0ed42e31f081d4a014c3ecc55a4f33d5f58dbb82",
    "exp6900": "sha256:beb442dfa3743bc3271150eb88d35cf0e31ac8b657e00664ed143611ed7d0c0c",
    "producer_module": "sha256:45ee78083f54c2ac8ddbebc29bb87586d8e86f204f75572b928cfd7ac05c73cd",
    "producer_entrypoint": "sha256:fbd41372783c35594620902255a836f2472ceaff752adc4a19afa7502be6f76a",
}
EXPECTED_SOURCE_VERDICT = "complete_positive_authentic_anchored_relation_corpus"
EXPECTED_TAUTOLOGY_FINDING = {
    "kind": "TAUTOLOGY",
    "severity": "critical",
    "detail": (
        "duration_s=665.794477 and live_duration_s=665.794477 agree to >5 sig figs. "
        "Two distinct metrics matching this precisely is more likely a bug than a finding."
    ),
}
REQUIRED_MODELS = tuple(producer.MODEL_SPECS)
CONTROL_ARMS = (producer.ENOKI_ARM, producer.RULE_ARM)
REQUIRED_SEEDS = tuple(producer.SEEDS)
EXPECTED_CELL_COUNT = 1_400
RANDOM_SEED = 6912
SCHEMA = "carnot.exp6912.alias_safe_relation_corpus_reducer.v1"
INFERENCE_SUBSTRATE = "deterministic_cpu_immutable_relation_corpus_reducer_no_llm"
BLOCKED_VERDICT = "complete_blocked_alias_safe_relation_corpus_reducer"
READY_VERDICT = "complete_positive_alias_safe_relation_corpus_reducer"

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_live_duration_s",
    "source_live_duration_derivation",
    "duration_alias_detected",
    "source_artifact_hashes",
    "source_flag_preservation",
    "rows",
    "cell_identity_rows",
    "raw_hash_rows",
    "lifecycle_rows",
    "timing_rows",
    "arm_rows",
    "model_rows",
    "seed_rows",
    "source_group_rows",
    "parser_rows",
    "stop_reason_rows",
    "timeout_rows",
    "truncation_rows",
    "reported_vs_recomputed_metrics",
    "source_cell_count",
    "replayed_cell_count",
    "duplicate_cell_count",
    "source_mutation_count",
    "model_inference_call_count",
    "fresh_adversarial_rows",
    "random_seed",
    "reproducibility_checksum",
    "clean_relation_corpus_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required value states why it is needed for later audits.",
    "preconditions_checked": "Exact source checks prevent reduction of changed evidence.",
    "inference_substrate": "The declaration proves this process performs no model inference.",
    "duration_s": "A local monotonic interval proves the reducer itself ran.",
    "source_live_duration_s": "A separate source-derived interval prevents duration aliasing.",
    "source_live_duration_derivation": "The formula makes every source-time contribution replayable.",
    "duration_alias_detected": "The original equality remains visible instead of being erased.",
    "source_artifact_hashes": "Content hashes bind this receipt to immutable source bytes.",
    "source_flag_preservation": "Source quarantine and verdict evidence must survive reduction.",
    "rows": "Per-cell and per-check rows prevent favorable outcomes from hiding failures.",
    "cell_identity_rows": "Exact identities reveal missing, duplicate, or substituted cells.",
    "raw_hash_rows": "Raw request and response hashes bind parser evidence to bytes.",
    "lifecycle_rows": "Lifecycle boundaries define source live time without a copied aggregate.",
    "timing_rows": "Cell intervals expose reversed, negative, or overlapping timing evidence.",
    "arm_rows": "Arm counts keep each acquisition method in its original denominator.",
    "model_rows": "Model counts expose family substitution or omission.",
    "seed_rows": "Seed counts expose random-identity drift.",
    "source_group_rows": "Source-group counts preserve the balanced acquisition design.",
    "parser_rows": "Parser outcomes retain malformed and unsuccessful cells.",
    "stop_reason_rows": "Stop reasons preserve transport failures and normal completion.",
    "timeout_rows": "Timeout counts cannot disappear from a favorable headline.",
    "truncation_rows": "Truncation counts cannot disappear from a favorable headline.",
    "reported_vs_recomputed_metrics": "Side-by-side values expose stale source aggregates.",
    "source_cell_count": "The source count fixes the replay denominator.",
    "replayed_cell_count": "The replay count proves every source cell was processed.",
    "duplicate_cell_count": "Zero duplicates prevent one cell from receiving repeated credit.",
    "source_mutation_count": "Zero mutations are required before evidence can be admitted.",
    "model_inference_call_count": "Zero confirms that the immutable corpus was not reacquired.",
    "fresh_adversarial_rows": "Current verifier findings remain evidence, not hidden status.",
    "random_seed": "A fixed seed identifies this deterministic reducer contract.",
    "reproducibility_checksum": "A stable digest proves replay evidence agrees across processes.",
    "clean_relation_corpus_ready_score": "Only complete, alias-safe, clean replay opens semantics.",
    "gate_check_summary": "Failed checks retain their expected and observed values.",
    "verifier_is_oracle": "False prevents a reducer check from becoming semantic authority.",
    "verdict_class": "The closed enum gives downstream tools one terminal classification.",
    "honest_verdict": "A terminal prefix prevents partial or blocked work from looking complete.",
}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable ordering so hashes are process-independent."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Return the repository's prefixed SHA-256 representation."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash one file without loading a large artifact into a second memory copy."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return "missing"
    return f"sha256:{digest.hexdigest()}"


def sha256_json(value: Any) -> str:
    """Hash canonical JSON evidence rather than platform-specific formatting."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _check(name: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Build one check row that always keeps both sides of the comparison."""

    return {
        "check": name,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed if passed is None else bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep the first failure convenient while retaining every failed check."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = [row for row in copied if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "passed": not failed,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else True,
        "observed": first.get("observed") if first else True,
        "failed_checks": failed,
        "checks": copied,
    }


def check_exact_hashes(observed: Mapping[str, str], expected: Mapping[str, str]) -> list[JsonDict]:
    """Compare frozen sources in declared order so the first blocker is stable."""

    return [
        _check(f"source_hash:{name}", digest, observed.get(name))
        for name, digest in expected.items()
    ]


def _decode_embedded(value: Any) -> tuple[bytes | None, str | None]:
    """Decode strict base64 so malformed evidence cannot silently become empty bytes."""

    if not isinstance(value, str):
        return None, "missing_embedded_bytes"
    try:
        return base64.b64decode(value.encode("ascii"), validate=True), None
    except (UnicodeEncodeError, ValueError):
        return None, "invalid_base64"


def _raw_component(row: Mapping[str, Any], prefix: str, count_field: str, root: Path) -> JsonDict:
    """Hash an embedded or file-backed raw component with one common receipt shape."""

    path_value = row.get(f"raw_{prefix}_path")
    if isinstance(path_value, str) and path_value:
        path = Path(path_value)
        path = path if path.is_absolute() else root / path
        try:
            payload = path.read_bytes()
            error = None
        except OSError as exc:
            payload = None
            error = f"unreadable:{type(exc).__name__}"
        source_kind = "file"
    else:
        payload, error = _decode_embedded(row.get(f"raw_{prefix}_b64"))
        path = None
        source_kind = "embedded"
    declared_hash = row.get(f"raw_{prefix}_sha256")
    declared_count = row.get(count_field)
    observed_hash = sha256_bytes(payload) if payload is not None else None
    observed_count = len(payload) if payload is not None else None
    passed = error is None and declared_hash == observed_hash and declared_count == observed_count
    return {
        "source_kind": source_kind,
        "path": str(path) if path is not None else None,
        "declared_sha256": declared_hash,
        "observed_sha256": observed_hash,
        "declared_byte_count": declared_count,
        "observed_byte_count": observed_count,
        "error": error,
        "passed": passed,
    }


def _reference_identity(row: Mapping[str, Any]) -> str:
    """Use attempt identity when present so retries never overwrite final cells."""

    return str(row.get("attempt_identity") or row.get("cell_identity") or "")


def replay_raw_references(source: Mapping[str, Any], root: Path) -> JsonDict:
    """Hash all final and prior raw manifests and cross-check final cell copies."""

    request_rows = [
        row for row in source.get("raw_request_manifest", []) if isinstance(row, Mapping)
    ]
    output_rows = [row for row in source.get("raw_output_manifest", []) if isinstance(row, Mapping)]
    request_by_id: dict[str, Mapping[str, Any]] = {}
    output_by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_refs: list[str] = []
    ordered_ids: list[str] = []
    for target, manifest in ((request_by_id, request_rows), (output_by_id, output_rows)):
        for row in manifest:
            identity = _reference_identity(row)
            if identity in target:
                duplicate_refs.append(identity)
            else:
                target[identity] = row
                if identity not in ordered_ids:
                    ordered_ids.append(identity)
    final_cells = {
        str(row.get("cell_identity")): row
        for row in source.get("cell_manifest", [])
        if isinstance(row, Mapping)
    }
    raw_hash_rows: list[JsonDict] = []
    for identity in ordered_ids:
        request = request_by_id.get(identity, {})
        output = output_by_id.get(identity, {})
        request_result = _raw_component(request, "request", "request_byte_count", root)
        output_result = _raw_component(output, "output", "output_byte_count", root)
        http_result = _raw_component(output, "http_response", "http_response_byte_count", root)
        cell = final_cells.get(identity)
        cell_matches = True
        if cell is not None:
            compared_fields = (
                "raw_request_sha256",
                "request_byte_count",
                "raw_output_sha256",
                "output_byte_count",
                "raw_http_response_sha256",
                "http_response_byte_count",
            )
            combined = {**request, **output}
            cell_matches = all(cell.get(field) == combined.get(field) for field in compared_fields)
        errors = []
        if identity in duplicate_refs:
            errors.append("duplicate_raw_reference")
        if identity not in request_by_id:
            errors.append("missing_request_reference")
        if identity not in output_by_id:
            errors.append("missing_output_reference")
        if not cell_matches:
            errors.append("cell_manifest_raw_reference_drift")
        passed = (
            not errors
            and request_result["passed"]
            and output_result["passed"]
            and http_result["passed"]
        )
        raw_hash_rows.append(
            {
                "reference_identity": identity,
                "cell_identity": request.get("cell_identity") or output.get("cell_identity"),
                "prior_attempt": "attempt_identity" in request or "attempt_identity" in output,
                "request": request_result,
                "output": output_result,
                "http_response": http_result,
                "cell_manifest_matches": cell_matches,
                "errors": errors,
                "passed": passed,
            }
        )
    missing_ids = sorted(set(request_by_id) ^ set(output_by_id))
    return {
        "passed": bool(raw_hash_rows)
        and not missing_ids
        and not duplicate_refs
        and all(row["passed"] for row in raw_hash_rows),
        "raw_hash_rows": raw_hash_rows,
        "duplicate_reference_identities": sorted(set(duplicate_refs)),
        "unpaired_reference_identities": missing_ids,
        "referenced_raw_file_count": sum(
            component["source_kind"] == "file"
            for row in raw_hash_rows
            for component in (row["request"], row["output"], row["http_response"])
        ),
    }


def _cell_raw_bytes(cell: Mapping[str, Any], prefix: str) -> bytes:
    """Return embedded cell bytes, leaving malformed evidence for a separate error."""

    payload, _ = _decode_embedded(cell.get(f"raw_{prefix}_b64"))
    return payload or b""


def _expected_identity(arm: str, hf_id: Any, seed: Any, fixture_id: str) -> str | None:
    """Derive identity from explicit fields instead of trusting the stored identity."""

    if arm.startswith("gguf:") and isinstance(hf_id, str) and isinstance(seed, int):
        return producer.gguf_cell_identity(hf_id, seed, fixture_id)
    if arm in CONTROL_ARMS and hf_id is None and seed is None:
        return producer.control_cell_identity(arm, fixture_id)
    return None


def _transport_state(cell: Mapping[str, Any]) -> JsonDict:
    """Derive terminal transport state from raw HTTP evidence and fixed control rules."""

    arm = str(cell.get("arm") or "")
    if cell.get("timed_out") is True and cell.get("stop_reason") == "timeout":
        stop_reason = "timeout"
    elif arm == producer.ENOKI_ARM:
        stop_reason = "encoder_complete"
    elif arm == producer.RULE_ARM:
        stop_reason = "rule_complete"
    else:
        stop_reason = None
        raw_http = _cell_raw_bytes(cell, "http_response")
        try:
            payload = json.loads(raw_http.decode("utf-8"))
            stop_reason = payload["choices"][0]["finish_reason"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError):
            stop_reason = cell.get("stop_reason")
    timed_out = stop_reason == "timeout"
    truncated = stop_reason == "length"
    return {
        "stop_reason": stop_reason,
        "timed_out": timed_out,
        "truncated": truncated,
        "terminal": bool(stop_reason),
    }


def _cell_replay_row(cell: Mapping[str, Any], source_record: Mapping[str, Any] | None) -> JsonDict:
    """Replay identity, parser output, and terminal state for one final cell."""

    arm = str(cell.get("arm") or "")
    hf_id = cell.get("hf_id")
    seed = cell.get("seed")
    fixture_id = str(cell.get("fixture_id") or "")
    expected_identity = _expected_identity(arm, hf_id, seed, fixture_id)
    errors: list[str] = []
    if arm.startswith("gguf:"):
        if hf_id not in REQUIRED_MODELS or arm != f"gguf:{hf_id}":
            errors.append("model_identity")
        identity_parts = str(cell.get("cell_identity") or "").split("::")
        identity_seed = identity_parts[1] if len(identity_parts) == 3 else None
        if identity_seed != str(seed):
            errors.append("seed_identity")
    if cell.get("cell_identity") != expected_identity:
        errors.append("cell_identity")
    if source_record is None:
        errors.append("missing_source_record")
        replayed_parse_rows: list[JsonDict] = []
    else:
        for field in ("group_id", "family", "split", "source_text_hash", "source_order"):
            if cell.get(field) != source_record.get(field):
                errors.append("source_group_identity")
                break
        output = _cell_raw_bytes(cell, "output").decode("utf-8", errors="replace")
        if arm == producer.ENOKI_ARM:
            try:
                enoki_result = json.loads(output)
            except json.JSONDecodeError:
                enoki_result = {}
            replayed_parse_rows = producer.base.parse_enoki_result(enoki_result, source_record)
        else:
            replayed_parse_rows = producer.parse_relation_output(output, source_record)
        if (
            cell.get("parse_rows") != replayed_parse_rows
            or cell.get("parser_attempted") is not True
        ):
            errors.append("parser_outcome")
    transport = _transport_state(cell)
    if any(cell.get(field) != transport[field] for field in transport):
        errors.append("terminal_state")
    source_cell = {
        key: deepcopy(cell.get(key))
        for key in (
            "cell_identity",
            "arm",
            "hf_id",
            "model_family",
            "seed",
            "fixture_id",
            "group_id",
            "family",
            "split",
            "source_order",
            "source_text_hash",
            "raw_request_sha256",
            "raw_http_response_sha256",
            "raw_output_sha256",
            "parser_input_sha256",
            "parse_rows",
            "stop_reason",
            "timed_out",
            "truncated",
            "terminal",
        )
    }
    return {
        "row_type": "source_cell_replay",
        "cell_identity": cell.get("cell_identity"),
        "source_cell": source_cell,
        "replay_result": {
            "expected_identity": expected_identity,
            "parser_rows": replayed_parse_rows,
            "parser_outcome_sha256": sha256_json(replayed_parse_rows),
            **transport,
            "errors": errors,
            "passed": not errors,
        },
    }


def _count_rows(cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build every required grouping directly from final cell rows."""

    arm_counts = Counter(str(cell.get("arm")) for cell in cells)
    model_counts = Counter((cell.get("hf_id"), str(cell.get("model_family"))) for cell in cells)
    seed_counts = Counter(
        str(cell.get("seed")) if cell.get("seed") is not None else "deterministic" for cell in cells
    )
    group_fixtures: dict[str, set[str]] = defaultdict(set)
    group_cells = Counter()
    parser_counts = Counter()
    stop_counts = Counter(str(cell.get("stop_reason")) for cell in cells)
    timeout_counts = Counter(bool(cell.get("timed_out")) for cell in cells)
    truncation_counts = Counter(bool(cell.get("truncated")) for cell in cells)
    arm_family: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for cell in cells:
        group = str(cell.get("group_id"))
        fixture = str(cell.get("fixture_id"))
        group_fixtures[group].add(fixture)
        group_cells[group] += 1
        arm_family[str(cell.get("arm"))][str(cell.get("family"))].add(fixture)
        for parse_row in cell.get("parse_rows", []):
            status = parse_row.get("status") if isinstance(parse_row, Mapping) else "invalid_row"
            parser_counts[str(status)] += 1
    return {
        "arm_rows": [{"arm": key, "cell_count": arm_counts[key]} for key in sorted(arm_counts)],
        "model_rows": [
            {"hf_id": key[0], "model_family": key[1], "cell_count": count}
            for key, count in sorted(model_counts.items(), key=lambda item: str(item[0]))
        ],
        "seed_rows": [{"seed": key, "cell_count": seed_counts[key]} for key in sorted(seed_counts)],
        "source_group_rows": [
            {
                "group_id": key,
                "fixture_count": len(group_fixtures[key]),
                "cell_count": group_cells[key],
            }
            for key in sorted(group_cells)
        ],
        "parser_rows": [
            {"parser_status": key, "row_count": parser_counts[key]} for key in sorted(parser_counts)
        ],
        "stop_reason_rows": [
            {"stop_reason": key, "cell_count": stop_counts[key]} for key in sorted(stop_counts)
        ],
        "timeout_rows": [
            {"timed_out": key, "cell_count": timeout_counts[key]} for key in (False, True)
        ],
        "truncation_rows": [
            {"truncated": key, "cell_count": truncation_counts[key]} for key in (False, True)
        ],
        "per_arm_family_counts": {
            arm: {family: len(fixtures) for family, fixtures in sorted(families.items())}
            for arm, families in sorted(arm_family.items())
        },
    }


def _aggregate_comparisons(
    source: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Keep producer aggregates next to values recomputed from raw cell rows."""

    prior = [row for row in source.get("prior_attempt_cells", []) if isinstance(row, Mapping)]
    flattened = producer.flatten_terminal_rows(cells) + producer._prior_terminal_rows(prior)
    counts = _count_rows(cells)
    comparisons = [
        ("source_cell_count", len(source.get("cell_manifest", [])), len(cells)),
        (
            "source_record_count",
            source.get("prompt_manifest", {}).get("source_count"),
            len(source_records),
        ),
        (
            "per_arm_family_counts",
            source.get("per_arm_family_counts"),
            counts["per_arm_family_counts"],
        ),
        (
            "raw_request_manifest_count",
            len(source.get("raw_request_manifest", [])),
            len(cells) + len(prior),
        ),
        (
            "raw_output_manifest_count",
            len(source.get("raw_output_manifest", [])),
            len(cells) + len(prior),
        ),
        ("flattened_rows_sha256", sha256_json(source.get("rows", [])), sha256_json(flattened)),
    ]
    for field, statuses in (
        ("parse_failure_rows", {"malformed", "unsupported", "invalid_span", "duplicate"}),
        ("empty_rows", {"empty"}),
        ("timeout_rows", {"timeout"}),
        ("truncation_rows", {"truncated"}),
        ("abstention_rows", {"abstention"}),
    ):
        recomputed = [row for row in flattened if row.get("status") in statuses]
        comparisons.append(
            (f"{field}_sha256", sha256_json(source.get(field, [])), sha256_json(recomputed))
        )
    return [
        {
            "metric": name,
            "reported": reported,
            "recomputed": recomputed,
            "passed": reported == recomputed,
        }
        for name, reported, recomputed in comparisons
    ]


def replay_cells(
    source: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    *,
    required_models: Sequence[str] = REQUIRED_MODELS,
    required_seeds: Sequence[int] = REQUIRED_SEEDS,
    expected_cell_count: int = EXPECTED_CELL_COUNT,
) -> JsonDict:
    """Replay the complete final matrix and recompute all requested aggregates."""

    cells = [row for row in source.get("cell_manifest", []) if isinstance(row, Mapping)]
    records_by_fixture = {str(row.get("fixture_id")): row for row in source_records}
    observed_ids = [str(row.get("cell_identity")) for row in cells]
    identity_counts = Counter(observed_ids)
    duplicate_count = sum(count - 1 for count in identity_counts.values() if count > 1)
    expected_ids = {
        producer.gguf_cell_identity(model, seed, str(record["fixture_id"]))
        for model in required_models
        for seed in required_seeds
        for record in source_records
    }
    expected_ids.update(
        producer.control_cell_identity(arm, str(record["fixture_id"]))
        for arm in CONTROL_ARMS
        for record in source_records
    )
    rows = [
        _cell_replay_row(cell, records_by_fixture.get(str(cell.get("fixture_id"))))
        for cell in cells
    ]
    counts = _count_rows(cells)
    comparisons = _aggregate_comparisons(source, cells, source_records)
    checks = [
        _check("exact_cell_identity_set", sorted(expected_ids), sorted(set(observed_ids))),
        _check("exact_cell_count", expected_cell_count, len(cells)),
        _check("duplicate_cell_count", 0, duplicate_count),
        _check(
            "cell_replay_errors",
            {},
            {
                str(row.get("cell_identity")): row["replay_result"]["errors"]
                for row in rows
                if row["replay_result"]["errors"]
            },
        ),
        _check(
            "reported_vs_recomputed_metrics",
            True,
            all(row["passed"] for row in comparisons),
        ),
    ]
    summary = gate_summary(checks)
    return {
        "passed": summary["passed"],
        "rows": rows,
        "cell_identity_rows": [
            {
                "cell_identity": identity,
                "occurrence_count": identity_counts[identity],
                "expected": identity in expected_ids,
            }
            for identity in sorted(identity_counts)
        ],
        **{key: value for key, value in counts.items() if key != "per_arm_family_counts"},
        "reported_vs_recomputed_metrics": comparisons,
        "source_cell_count": len(cells),
        "replayed_cell_count": len(rows),
        "duplicate_cell_count": duplicate_count,
        "gate_check_summary": summary,
    }


def derive_source_live_duration(
    cells: Sequence[Mapping[str, Any]], source_lifecycle_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Derive source live time once from explicit cell and lifecycle evidence."""

    timing_rows: list[JsonDict] = []
    by_model: dict[str, list[JsonDict]] = defaultdict(list)
    enoki_durations: list[float] = []
    rule_duration = 0.0
    for cell in cells:
        arm = str(cell.get("arm") or "")
        try:
            elapsed = float(cell.get("wall_time_s"))
        except (TypeError, ValueError):
            elapsed = math.nan
        errors: list[str] = []
        if not math.isfinite(elapsed):
            errors.append("non_finite_elapsed")
        elif elapsed < 0:
            errors.append("negative_elapsed")
        receipt_ns = cell.get("runtime_receipt", {}).get("receipt_monotonic_ns")
        start_ns: int | None = None
        end_ns: int | None = None
        if arm.startswith("gguf:"):
            if not isinstance(receipt_ns, int) or receipt_ns <= 0:
                errors.append("missing_receipt_monotonic_ns")
            elif math.isfinite(elapsed):
                start_ns = receipt_ns
                end_ns = receipt_ns + round(elapsed * 1_000_000_000)
                if start_ns > end_ns:
                    errors.append("reversed_interval")
        elif arm == producer.ENOKI_ARM:
            try:
                enoki_durations.append(float(cell.get("runtime_receipt", {}).get("duration_s")))
            except (TypeError, ValueError):
                errors.append("missing_enoki_batch_duration")
        elif arm == producer.RULE_ARM and math.isfinite(elapsed) and elapsed >= 0:
            rule_duration += elapsed
        row = {
            "cell_identity": cell.get("cell_identity"),
            "arm": arm,
            "wall_time_s": elapsed,
            "receipt_monotonic_ns": receipt_ns,
            "derived_start_monotonic_ns": start_ns,
            "derived_end_monotonic_ns": end_ns,
            "errors": errors,
            "passed": not errors,
        }
        timing_rows.append(row)
        if arm.startswith("gguf:"):
            by_model[str(cell.get("hf_id"))].append(row)

    lifecycle_by_model = {
        str(row.get("hf_id")): row for row in source_lifecycle_rows if isinstance(row, Mapping)
    }
    lifecycle_rows: list[JsonDict] = []
    for model in REQUIRED_MODELS:
        model_rows = by_model.get(model, [])
        errors: list[str] = []
        previous_end: int | None = None
        for row in model_rows:
            start_ns = row["derived_start_monotonic_ns"]
            end_ns = row["derived_end_monotonic_ns"]
            if not isinstance(start_ns, int) or not isinstance(end_ns, int):
                continue
            if previous_end is not None and end_ns < previous_end:
                row["errors"].append("non_monotonic_receipt")
                row["passed"] = False
                errors.append("non_monotonic_receipt")
            if previous_end is not None and start_ns < previous_end:
                row["errors"].append("overlapping_intervals")
                row["passed"] = False
                errors.append("overlapping_intervals")
            previous_end = end_ns
        valid = [
            row
            for row in model_rows
            if isinstance(row["derived_start_monotonic_ns"], int)
            and isinstance(row["derived_end_monotonic_ns"], int)
        ]
        source_lifecycle = lifecycle_by_model.get(model)
        if not valid:
            errors.append("missing_model_timing_rows")
            start_ns = end_ns = None
            duration = 0.0
        else:
            start_ns = valid[0]["derived_start_monotonic_ns"]
            end_ns = valid[-1]["derived_end_monotonic_ns"]
            duration = (end_ns - start_ns) / 1_000_000_000
        if source_lifecycle is None:
            errors.append("missing_source_lifecycle")
        else:
            first_receipt = next(
                (cell.get("runtime_receipt", {}) for cell in cells if cell.get("hf_id") == model),
                {},
            )
            if source_lifecycle.get("pid") != first_receipt.get("server_pid"):
                errors.append("lifecycle_pid_mismatch")
            if source_lifecycle.get("process_identity_match") is not True:
                errors.append("lifecycle_identity_unverified")
        lifecycle_rows.append(
            {
                "lifecycle_kind": "gguf_process",
                "hf_id": model,
                "first_cell_identity": valid[0]["cell_identity"] if valid else None,
                "last_cell_identity": valid[-1]["cell_identity"] if valid else None,
                "start_monotonic_ns": start_ns,
                "end_monotonic_ns": end_ns,
                "derived_duration_s": round(duration, 9),
                "source_lifecycle": deepcopy(dict(source_lifecycle or {})),
                "errors": sorted(set(errors)),
                "passed": not errors,
            }
        )

    enoki_errors: list[str] = []
    finite_enoki = [value for value in enoki_durations if math.isfinite(value) and value >= 0]
    unique_enoki = sorted({round(value, 9) for value in finite_enoki})
    if len(finite_enoki) != len(enoki_durations) or len(unique_enoki) != 1:
        enoki_errors.append("inconsistent_enoki_batch_duration")
    enoki_duration = unique_enoki[0] if len(unique_enoki) == 1 else 0.0
    lifecycle_rows.append(
        {
            "lifecycle_kind": "enoki_batch",
            "hf_id": None,
            "source_cell_count": len(enoki_durations),
            "contribution_rule": "one shared runtime_receipt.duration_s value",
            "derived_duration_s": enoki_duration,
            "errors": enoki_errors,
            "passed": not enoki_errors,
        }
    )
    lifecycle_rows.append(
        {
            "lifecycle_kind": "lexical_cells",
            "hf_id": None,
            "source_cell_count": sum(cell.get("arm") == producer.RULE_ARM for cell in cells),
            "contribution_rule": "sum per-cell wall_time_s",
            "derived_duration_s": round(rule_duration, 9),
            "errors": [],
            "passed": True,
        }
    )
    gguf_lifecycles = [row for row in lifecycle_rows if row["lifecycle_kind"] == "gguf_process"]
    ordered_intervals = sorted(
        (
            (row["start_monotonic_ns"], row["end_monotonic_ns"], row)
            for row in gguf_lifecycles
            if isinstance(row["start_monotonic_ns"], int)
            and isinstance(row["end_monotonic_ns"], int)
        ),
        key=lambda item: item[0],
    )
    for previous, current in zip(ordered_intervals, ordered_intervals[1:]):
        if current[0] < previous[1]:
            current[2]["errors"].append("overlapping_process_lifecycles")
            current[2]["passed"] = False
    source_live = round(sum(float(row["derived_duration_s"]) for row in lifecycle_rows), 6)
    passed = all(row["passed"] for row in timing_rows) and all(
        row["passed"] for row in lifecycle_rows
    )
    return {
        "passed": passed,
        "source_live_duration_s": source_live,
        "source_live_duration_derivation": {
            "formula": (
                "sum(gguf earliest_pre_request_receipt_to_latest_cell_end) + "
                "one_enoki_batch_duration + sum(lexical_cell_wall_time_s)"
            ),
            "contributing_lifecycle_rows": [
                {
                    "lifecycle_kind": row["lifecycle_kind"],
                    "hf_id": row.get("hf_id"),
                    "derived_duration_s": row["derived_duration_s"],
                }
                for row in lifecycle_rows
            ],
        },
        "timing_rows": timing_rows,
        "lifecycle_rows": lifecycle_rows,
    }


def duration_independence_check(duration_s: float, source_live_duration_s: float) -> JsonDict:
    """Reject reducer/source durations that agree closely enough to be copied aliases."""

    independent = not math.isclose(
        float(duration_s), float(source_live_duration_s), rel_tol=1e-9, abs_tol=1e-9
    )
    return _check(
        "independent_duration_definitions",
        True,
        independent,
        independent,
    )


def check_source_flag_preservation(source: Mapping[str, Any]) -> JsonDict:
    """Require the original alias, quarantine, finding, and verdict without rewriting them."""

    findings = source.get("corrigendum_pending", [])
    finding_present = isinstance(findings, list) and EXPECTED_TAUTOLOGY_FINDING in findings
    try:
        old_alias = float(source.get("duration_s")) == float(source.get("live_duration_s"))
    except (TypeError, ValueError):
        old_alias = False
    checks = [
        _check("source_flagged_adversarial", True, source.get("flagged_adversarial")),
        _check("source_tautology_finding", True, finding_present),
        _check("source_honest_verdict", EXPECTED_SOURCE_VERDICT, source.get("honest_verdict")),
        _check("source_verdict_class", "positive", source.get("verdict_class")),
        _check("source_duration_alias", True, old_alias),
    ]
    summary = gate_summary(checks)
    return {
        "passed": summary["passed"],
        "duration_alias_detected": old_alias,
        "flagged_adversarial": source.get("flagged_adversarial"),
        "honest_verdict": source.get("honest_verdict"),
        "verdict_class": source.get("verdict_class"),
        "corrigendum_pending": deepcopy(findings),
        "gate_check_summary": summary,
    }


def _attach_principles(artifact: JsonDict) -> None:
    """Explain every top-level field so later auditors do not infer its purpose."""

    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves deterministic reducer evidence.")
        for key in artifact
    }
    for key in REQUIRED_ARTIFACT_FIELDS:
        artifact["field_principles"][key] = FIELD_PRINCIPLES[key]


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic evidence while excluding the local process interval and self hash."""

    included = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "duration_s",
            "field_principles",
            "fresh_adversarial_rows",
            "gate_check_summary",
            "honest_verdict",
            "reproducibility_checksum",
            "rows",
            "status",
            "verdict_class",
        }
    }
    return sha256_json(included)


def _verify_candidate(
    artifact: Mapping[str, Any], verify_fn: Callable[[str], Mapping[str, Any]]
) -> JsonDict:
    """Run the current verifier on a temporary receipt before the final file is admitted."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
        json.dump(artifact, handle, sort_keys=True)
        handle.flush()
        return deepcopy(dict(verify_fn(handle.name)))


def reduce_corpus(
    *,
    source: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    root: Path,
    date: str,
    duration_s: float | Callable[[], float],
    source_artifact_hashes: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    required_models: Sequence[str] = REQUIRED_MODELS,
    required_seeds: Sequence[int] = REQUIRED_SEEDS,
    expected_cell_count: int = EXPECTED_CELL_COUNT,
    verify_fn: Callable[[str], Mapping[str, Any]],
) -> JsonDict:
    """Reduce all source evidence and admit only a clean, independent receipt."""

    raw = replay_raw_references(source, root)
    replay = replay_cells(
        source,
        source_records,
        required_models=required_models,
        required_seeds=required_seeds,
        expected_cell_count=expected_cell_count,
    )
    cells = [row for row in source.get("cell_manifest", []) if isinstance(row, Mapping)]
    timing = derive_source_live_duration(cells, source.get("server_lifecycle_rows", []))
    flags = check_source_flag_preservation(source)
    measured_duration = float(duration_s() if callable(duration_s) else duration_s)
    duration_check = duration_independence_check(
        measured_duration, timing["source_live_duration_s"]
    )
    precondition_checks = [deepcopy(dict(row)) for row in preconditions_checked.get("checks", [])]
    checks = [
        *precondition_checks,
        _check("preconditions_passed", True, preconditions_checked.get("passed") is True),
        *replay["gate_check_summary"]["checks"],
        _check("raw_reference_replay", True, raw["passed"]),
        _check("source_timing_replay", True, timing["passed"]),
        *flags["gate_check_summary"]["checks"],
        duration_check,
        _check("model_inference_call_count", 0, 0),
    ]
    provisional_summary = gate_summary(checks)
    source_mutation_count = sum(
        row.get("passed") is not True
        for row in checks
        if str(row.get("check", "")).startswith(("source_hash:", "raw_"))
    )
    provisional_ready = int(provisional_summary["passed"] and source_mutation_count == 0)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6912,
        "run_date": date,
        "status": "complete" if provisional_ready else "blocked",
        "field_principles": {},
        "preconditions_checked": deepcopy(dict(preconditions_checked)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(measured_duration, 6),
        "source_live_duration_s": timing["source_live_duration_s"],
        "source_live_duration_derivation": timing["source_live_duration_derivation"],
        "duration_alias_detected": flags["duration_alias_detected"],
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "source_flag_preservation": flags,
        "rows": replay["rows"],
        "cell_identity_rows": replay["cell_identity_rows"],
        "raw_hash_rows": raw["raw_hash_rows"],
        "lifecycle_rows": timing["lifecycle_rows"],
        "timing_rows": timing["timing_rows"],
        "arm_rows": replay["arm_rows"],
        "model_rows": replay["model_rows"],
        "seed_rows": replay["seed_rows"],
        "source_group_rows": replay["source_group_rows"],
        "parser_rows": replay["parser_rows"],
        "stop_reason_rows": replay["stop_reason_rows"],
        "timeout_rows": replay["timeout_rows"],
        "truncation_rows": replay["truncation_rows"],
        "reported_vs_recomputed_metrics": replay["reported_vs_recomputed_metrics"],
        "prior_attempt_rows": [
            {
                "attempt_index": index,
                "cell_identity": row.get("cell_identity"),
                "stop_reason": row.get("stop_reason"),
                "parser_outcomes": [
                    item.get("status")
                    for item in row.get("parse_rows", [])
                    if isinstance(item, Mapping)
                ],
                "raw_request_sha256": row.get("raw_request_sha256"),
                "raw_output_sha256": row.get("raw_output_sha256"),
                "terminal": row.get("terminal"),
            }
            for index, row in enumerate(source.get("prior_attempt_cells", []))
            if isinstance(row, Mapping)
        ],
        "source_cell_count": replay["source_cell_count"],
        "replayed_cell_count": replay["replayed_cell_count"],
        "duplicate_cell_count": replay["duplicate_cell_count"],
        "source_mutation_count": source_mutation_count,
        "model_inference_call_count": 0,
        "fresh_adversarial_rows": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "clean_relation_corpus_ready_score": provisional_ready,
        "gate_check_summary": provisional_summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if provisional_ready else "blocked",
        "honest_verdict": READY_VERDICT if provisional_ready else BLOCKED_VERDICT,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    report = _verify_candidate(artifact, verify_fn)
    report_flags = [dict(row) for row in report.get("flags", []) if isinstance(row, Mapping)]
    critical = [row for row in report_flags if str(row.get("severity", "")).lower() == "critical"]
    verifier_rows = [
        {
            "row_type": "adversarial_verifier_summary",
            "loaded": report.get("loaded") is True,
            "gate_version": report.get("gate_version"),
            "critical_count": len(critical),
            "flag_count": len(report_flags),
        },
        *[{"row_type": "adversarial_finding", **row} for row in report_flags],
    ]
    final_checks = [
        *checks,
        _check("fresh_adversarial_verifier_loaded", True, report.get("loaded") is True),
        _check("fresh_adversarial_critical_count", 0, len(critical)),
    ]
    final_summary = gate_summary(final_checks)
    ready = int(final_summary["passed"] and source_mutation_count == 0)
    artifact.update(
        {
            "status": "complete" if ready else "blocked",
            "rows": [
                *replay["rows"],
                *[{"row_type": "reducer_check", **row} for row in final_checks],
            ],
            "fresh_adversarial_rows": verifier_rows,
            "clean_relation_corpus_ready_score": ready,
            "gate_check_summary": final_summary,
            "verdict_class": "positive" if ready else "blocked",
            "honest_verdict": READY_VERDICT if ready else BLOCKED_VERDICT,
        }
    )
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def _load_current_verifier(path: str) -> Mapping[str, Any]:
    """Load the verifier lazily so importing this reducer performs no audit work."""

    verifier_path = REPO_ROOT / "scripts/adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_exp6912_adversarial", verifier_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("adversarial_verifier_import_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def _read_json(path: Path) -> JsonDict:
    """Read one source artifact and fail closed on non-object JSON."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source_artifact_must_be_object")
    return value


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace only the requested receipt after a complete JSON serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    expected_hashes: Mapping[str, str] = EXPECTED_SOURCE_HASHES,
    source_records: Sequence[Mapping[str, Any]] | None = None,
    required_models: Sequence[str] = REQUIRED_MODELS,
    required_seeds: Sequence[int] = REQUIRED_SEEDS,
    expected_cell_count: int = EXPECTED_CELL_COUNT,
    verify_fn: Callable[[str], Mapping[str, Any]] = _load_current_verifier,
    clock: Callable[[], float] = time.monotonic,
) -> JsonDict:
    """Run the reducer, always writing a terminal ready or blocked receipt."""

    started = clock()
    observed_hashes = {
        name: sha256_file(root / relative) for name, relative in SOURCE_PATHS.items()
    }
    source_hash_rows = check_exact_hashes(observed_hashes, expected_hashes)
    source_path = root / SOURCE_PATHS["exp6900"]
    source_error: str | None = None
    try:
        source = _read_json(source_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        source = {}
        source_error = f"{type(exc).__name__}: {exc}"
    if source_records is None:
        source_records = producer.reconstruct_source_records()
    raw = replay_raw_references(source, root)
    flags = check_source_flag_preservation(source)
    declared_exp6899 = source.get("source_artifact_hashes", {}).get("exp6899", {}).get("sha256")
    precondition_checks = [
        *source_hash_rows,
        _check("source_artifact_readable", None, source_error),
        _check(
            "source_declared_exp6899_hash",
            expected_hashes.get("exp6899"),
            declared_exp6899,
        ),
        _check("raw_reference_precondition", True, raw["passed"]),
        _check("source_flag_precondition", True, flags["passed"]),
    ]
    preconditions = gate_summary(precondition_checks)
    source_hash_artifact = {
        name: {
            "path": SOURCE_PATHS[name].as_posix(),
            "expected_sha256": expected_hashes.get(name),
            "observed_sha256": observed_hashes.get(name),
        }
        for name in SOURCE_PATHS
    }
    artifact = reduce_corpus(
        source=source,
        source_records=source_records,
        root=root,
        date=date,
        duration_s=lambda: clock() - started,
        source_artifact_hashes=source_hash_artifact,
        preconditions_checked=preconditions,
        required_models=required_models,
        required_seeds=required_seeds,
        expected_cell_count=expected_cell_count,
        verify_fn=verify_fn,
    )
    destination = output_path or root / RESULT_PATH
    _write_json(destination, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the required date and write one terminal reducer receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run(date=args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - the required wrapper calls main.
    raise SystemExit(main())
