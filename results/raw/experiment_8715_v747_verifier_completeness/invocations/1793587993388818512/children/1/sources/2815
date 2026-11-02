"""Audit raw source semantics against matched direct self-consistency.

This module invokes no model. It authenticates Exp7278 bytes, parses retained
responses with a second implementation, and groups uncertainty by source base.

Spec refs: REQ-VERIFY-7279 and SCENARIO-VERIFY-7279-*.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import threading
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260913"
MILESTONE = "2026.09.640"
EXPERIMENT_ID = "exp7279-source-audit"
SCHEMA = "carnot.exp7279.v640_source_audit.v1"
RANDOM_SEED = 727_920_260_913
BOOTSTRAP_SEED = 727_920_261_000
BOOTSTRAP_DRAWS = 10_000
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "usable_answers": 0,
}

PLANNED_BASE_GROUPS = 16
PLANNED_UNITS = 64
PLANNED_CALLS = 256
ARMS = (
    "verifier",
    "direct_self_consistency",
    "direct_one_shot",
    "source_shuffle_control",
)
PRIMARY_ARMS = ("verifier", "direct_self_consistency")
DIRECT_DECISIONS = {"supported", "contradicted", "unknown"}
AUTHORITY_FIELDS = {
    "authority_parser",
    "condition",
    "expected_decision",
    "game_source",
    "gold_claim_completion",
    "gold_source_completion",
}

UPSTREAM_PATH = Path("results/experiment_7278_v640_source_measurement.json")
UPSTREAM_RAW_DIR = Path("results/raw/experiment_7278")
RESULT_PATH = Path("results/experiment_7279_v640_source_audit.json")
RAW_DIR = Path("results/raw/experiment_7279")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7279_v640_source_audit.json")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
MODULE_PATH = Path("python/carnot/experiment_7279_v640_source_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7279_v640_source_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7279_v640_source_audit.py")

PINNED_INPUT_HASHES = {
    "upstream": "sha256:f744a63c1e9c6bcb39c1c5f40bc966fcffd6ca24c52b029160136e61c669d67c",
    "public": "sha256:c090b18fd9f7d7a22912735ffa3b2a673de2f9f4378c27ceeb88bad45ee9a2d4",
    "authority": "sha256:b621ad6749baf95344957719fb48bef4784201a0679d1123addf662918bfedc0",
    "schedule": "sha256:d1ef361383428209e9725d7e5f45edce2a20308d548b5902918b5224f131531e",
    "raw_manifest": "sha256:cd6a775483b56aab8544aeaeb24ef70935e93239d3d36262280e6a45998306fb",
}

REQUIRED_VALIDATION_NAMES = (
    "focused_pytest",
    "affected_suites",
    "scoped_coverage",
    "scoped_coverage_report",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec_coverage",
    "independent_raw_replay",
    "adversarial_verify",
    "verdict_row_consistency",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in separate checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical identities in hashed sidecars.",
    "model_invoked": "Derive from actual calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for actual computation, not an invented task label.",
    "inference_substrate_class": "Use the correct no-LLM class for CPU replay and tests; never pad duration.",
    "execution_venue": "Host orchestration is host; identify real device execution separately.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, input manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, and censored units and the stopping rule.",
    "acceptance_gate_results": "Each criterion records expected, observed, passed, and principle; separate completeness and value.",
    "gate_check_summary": "For blocked work name upstream, exact field or check, observed value, and expected value.",
    "verifier_is_oracle": "Expose shared verifier and evaluator authority; exact conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_; external absence starts blocked_; retain the measured finding.",
    "verdict_class": "Use the closed verdict set. Oracle evidence forbids positive, and failed efficacy gates forbid positive.",
    "validation_receipts": "Retain command, exit code, timing, and log hash; do not hide failures.",
    "source_audit_complete_score": "One means every fixed row and independent check has a disposition, including negative findings.",
    "source_promotion_score": "One only when all primary scientific and independence criteria pass.",
    "paired_comparisons": "Use base-group confidence intervals and retain recomputable arm numerators.",
    "causal_control_rows": "Retain authority, renaming, replacement, deletion, and shuffled-label interventions.",
    "claim_boundary": "A 16-base synthetic source pilot cannot support model-wide or GSM8K claims.",
    "arm_summaries": "Show every arm's raw numerators and denominators before a pooled comparison.",
    "matched_coverage_risk": "Report paired selective risk without choosing a threshold from private labels.",
    "raw_replay_rows": "Retain the second parser's disposition for every frozen raw call.",
    "replay_mismatches": "Keep missing, corrupt, or inconsistent raw evidence visible.",
    "source_leakage_errors": "Name any private authority field that entered a generation request.",
    "authority_boundary": "Keep public inputs, private labels, and the second parser in separate roles.",
    "historical_model_receipts": "Refer to historical model evidence only through authenticated sidecar hashes.",
    "timestamps": "Record actual UTC observations for task start and completion.",
    "phase_spans": "Record disjoint monotonic work spans for each phase.",
    "positive_control_results": "Use matched direct self-consistency to expose headroom without calling it independent truth.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

GATE_PRINCIPLES = {
    "fixed_rows_accounted": "All four audit arms must retain every one of the 64 source units.",
    "fixed_base_groups": "The uncertainty unit is the 16 independent source bases, not 64 correlated conditions.",
    "causal_controls_disposed": "Every fixed mutation must report an observed result even when it fails.",
    "accuracy_ci95_lower_above_zero": "Promotion needs a paired accuracy gain that stays positive under base-group resampling.",
    "false_accept_ci95_upper_at_most_zero": "A verifier gain cannot come from more confident wrong accepts.",
    "no_source_leakage": "Private authority fields must not enter any retained request.",
    "independent_replay_match": "A second parser must reconstruct all source decisions from raw bytes.",
    "beats_fixed_source_shuffle": "Real source use must beat the frozen source-replacement control.",
    "independence_controls_pass": "Authority, replacement, labels, naming, and deletion checks must all behave as expected.",
    "focused_validation": "Every focused command must pass before terminal publication.",
}


def canonical_json(value: Any) -> str:
    """Use one stable Unicode JSON spelling for hashes and byte comparisons."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes so parsing cannot hide a changed source."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so large evidence stays cheap to inspect."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding process-local clock observations."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str | None,
    artifact_field: str,
) -> JsonDict:
    """Keep both sides of one gate so a terminal block stays actionable."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project the first failure without discarding its exact expected value."""

    failure = next((row for row in checks if row.get("passed") is not True), None)
    if failure is None:
        return {
            "passed": True,
            "failed_check": None,
            "upstream": None,
            "artifact_field": None,
            "expected_value": "all_required_checks_pass",
            "observed_value": "all_required_checks_pass",
        }
    return {
        "passed": False,
        "failed_check": failure.get("check"),
        "upstream": failure.get("upstream"),
        "artifact_field": failure.get("artifact_field"),
        "expected_value": deepcopy(failure.get("expected_value")),
        "observed_value": deepcopy(failure.get("observed_value")),
    }


def _utc_now() -> str:
    """Record an actual UTC observation for process provenance."""

    return datetime.now(UTC).isoformat()


def _progress(phase: int, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase and long-operation boundary for the outer conductor."""

    print(
        canonical_json({"experiment": 7279, "phase": phase, "event": event, **details}), flush=True
    )


def _read_json(path: Path) -> tuple[bytes, JsonDict]:
    """Read exact JSON bytes and require a mapping at the document root."""

    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"mapping_required:{path}")
    return raw, value


def load_upstream_bundle(
    root: Path,
    *,
    upstream_path: Path | None = None,
    raw_dir: Path | None = None,
) -> JsonDict:
    """Load Exp7278 and every manifest or call file needed for raw replay."""

    upstream_file = upstream_path or root / UPSTREAM_PATH
    source_dir = raw_dir or root / UPSTREAM_RAW_DIR
    upstream_bytes, upstream = _read_json(upstream_file)
    public_bytes, public = _read_json(source_dir / "public_manifest.json")
    authority_bytes, authority = _read_json(source_dir / "private_authority_manifest.json")
    schedule_bytes, schedule_document = _read_json(source_dir / "schedule.json")
    manifest_bytes, raw_manifest = _read_json(source_dir / "raw_call_manifest.json")
    schedule = schedule_document.get("schedule")
    if not isinstance(schedule, list):
        raise ValueError("schedule_list")
    call_files: dict[int, JsonDict] = {}
    for index in range(len(schedule)):
        path = source_dir / f"call_{index:02d}.json"
        if path.is_file():
            raw, value = _read_json(path)
            call_files[index] = {"path": path, "bytes": raw, "value": value}
    exclusion_path = root / EXCLUSION_PATH
    exclusion_manifest = (
        yaml.safe_load(exclusion_path.read_text(encoding="utf-8"))
        if exclusion_path.is_file()
        else {}
    )
    return {
        "root": root,
        "paths": {
            "upstream": upstream_file,
            "public": source_dir / "public_manifest.json",
            "authority": source_dir / "private_authority_manifest.json",
            "schedule": source_dir / "schedule.json",
            "raw_manifest": source_dir / "raw_call_manifest.json",
        },
        "upstream_bytes": upstream_bytes,
        "public_bytes": public_bytes,
        "authority_bytes": authority_bytes,
        "schedule_bytes": schedule_bytes,
        "raw_manifest_bytes": manifest_bytes,
        "upstream": upstream,
        "public": public,
        "authority": authority,
        "schedule": schedule,
        "raw_manifest": raw_manifest,
        "call_files": call_files,
        "exclusion_manifest": exclusion_manifest,
    }


def _manifest_lists_experiment(value: Any, wanted: set[str]) -> bool:
    """Search identifier fields only so ordinary prose cannot cause retirement."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"experiment_id", "experiment_ids", "id", "scope_key"}:
                values = item if isinstance(item, list) else [item]
                if any(str(candidate) in wanted for candidate in values):
                    return True
            if _manifest_lists_experiment(item, wanted):
                return True
    elif isinstance(value, list):
        return any(_manifest_lists_experiment(item, wanted) for item in value)
    return False


def _mapping_has_true(value: Mapping[str, Any], fields: Sequence[str]) -> bool:
    """Recognize explicit quarantine markers without matching narrative prose."""

    return any(value.get(field) is True for field in fields)


def authenticate_bundle(bundle: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate upstream identity, sidecars, call bytes, and authority separation."""

    upstream = bundle["upstream"]
    selection = dict(upstream.get("selection_receipt") or {})
    hashes = {
        name: sha256_bytes(bundle[f"{name}_bytes"])
        for name in ("upstream", "public", "authority", "schedule", "raw_manifest")
    }
    checks = [
        gate_row(
            "upstream_measurement_identity",
            PINNED_INPUT_HASHES["upstream"],
            hashes["upstream"],
            hashes["upstream"] == PINNED_INPUT_HASHES["upstream"],
            upstream="exp7278-source-measurement",
            artifact_field="sha256",
        ),
        gate_row(
            "upstream_terminal_contract",
            {"status": "complete", "source_capture_complete_score": 1, "checksum_valid": True},
            {
                "status": upstream.get("status"),
                "source_capture_complete_score": upstream.get("source_capture_complete_score"),
                "checksum_valid": upstream.get("reproducibility_checksum")
                == artifact_checksum(upstream),
            },
            upstream.get("status") == "complete"
            and upstream.get("source_capture_complete_score") == 1
            and upstream.get("reproducibility_checksum") == artifact_checksum(upstream),
            upstream="exp7278-source-measurement",
            artifact_field="status|source_capture_complete_score|reproducibility_checksum",
        ),
    ]
    decoded_identities = {
        "public": sha256_bytes(canonical_json(bundle["public"]).encode("utf-8")),
        "authority": sha256_bytes(canonical_json(bundle["authority"]).encode("utf-8")),
    }
    checks.extend(
        [
            gate_row(
                "public_manifest_identity",
                {
                    "file_sha256": PINNED_INPUT_HASHES["public"],
                    "content_sha256": selection.get("public_manifest_sha256"),
                },
                {
                    "file_sha256": hashes["public"],
                    "content_sha256": decoded_identities["public"],
                },
                hashes["public"] == PINNED_INPUT_HASHES["public"]
                and decoded_identities["public"] == selection.get("public_manifest_sha256"),
                upstream="exp7278-public-manifest",
                artifact_field="file_and_canonical_sha256",
            ),
            gate_row(
                "private_authority_identity",
                {
                    "file_sha256": PINNED_INPUT_HASHES["authority"],
                    "content_sha256": selection.get("private_authority_sha256"),
                },
                {
                    "file_sha256": hashes["authority"],
                    "content_sha256": decoded_identities["authority"],
                },
                hashes["authority"] == PINNED_INPUT_HASHES["authority"]
                and decoded_identities["authority"] == selection.get("private_authority_sha256"),
                upstream="exp7278-private-authority",
                artifact_field="file_and_canonical_sha256",
            ),
            gate_row(
                "schedule_identity",
                {
                    "file_sha256": PINNED_INPUT_HASHES["schedule"],
                    "content_sha256": selection.get("schedule_sha256"),
                },
                {
                    "file_sha256": hashes["schedule"],
                    "content_sha256": sha256_bytes(
                        canonical_json(bundle["schedule"]).encode("utf-8")
                    ),
                },
                hashes["schedule"] == PINNED_INPUT_HASHES["schedule"]
                and sha256_bytes(canonical_json(bundle["schedule"]).encode("utf-8"))
                == selection.get("schedule_sha256"),
                upstream="exp7278-schedule",
                artifact_field="file_and_canonical_sha256",
            ),
            gate_row(
                "raw_manifest_identity",
                PINNED_INPUT_HASHES["raw_manifest"],
                hashes["raw_manifest"],
                hashes["raw_manifest"] == PINNED_INPUT_HASHES["raw_manifest"],
                upstream="exp7278-raw-manifest",
                artifact_field="sha256",
            ),
        ]
    )
    upstream_quarantined = _mapping_has_true(
        upstream, ("quarantined", "flagged_adversarial", "fabricated")
    )
    retired = _manifest_lists_experiment(
        bundle.get("exclusion_manifest"),
        {"7278", "7279", "exp7278-source-measurement", EXPERIMENT_ID},
    )
    checks.extend(
        [
            gate_row(
                "structured_quarantine",
                False,
                upstream_quarantined,
                not upstream_quarantined,
                upstream="exp7278-source-measurement",
                artifact_field="quarantined|flagged_adversarial|fabricated",
            ),
            gate_row(
                "retirement_manifest",
                False,
                retired,
                not retired,
                upstream="ops/exclusion_manifest.yaml",
                artifact_field="experiment_id",
            ),
        ]
    )
    manifest = bundle["raw_manifest"]
    calls = list(manifest.get("calls") or [])
    call_errors: list[str] = []
    call_files = bundle["call_files"]
    for index in range(PLANNED_CALLS):
        entry = call_files.get(index)
        declared = calls[index] if index < len(calls) else {}
        if entry is None:
            call_errors.append(f"call_{index}:missing_file")
        elif sha256_bytes(entry["bytes"]) != declared.get("sha256"):
            call_errors.append(f"call_{index}:sha256")
    manifest_contract = {
        "status": manifest.get("status"),
        "raw_call_count": manifest.get("raw_call_count"),
        "planned_call_count": manifest.get("planned_call_count"),
        "call_entries": len(calls),
        "call_errors": call_errors,
    }
    checks.append(
        gate_row(
            "raw_call_file_set",
            {
                "status": "complete",
                "raw_call_count": PLANNED_CALLS,
                "planned_call_count": PLANNED_CALLS,
                "call_entries": PLANNED_CALLS,
                "call_errors": [],
            },
            manifest_contract,
            manifest_contract
            == {
                "status": "complete",
                "raw_call_count": PLANNED_CALLS,
                "planned_call_count": PLANNED_CALLS,
                "call_entries": PLANNED_CALLS,
                "call_errors": [],
            },
            upstream="exp7278-raw-calls",
            artifact_field="manifest_and_call_sha256",
        )
    )
    leakage = source_leakage_errors(bundle["schedule"], manifest)
    checks.append(
        gate_row(
            "authority_separation",
            [],
            leakage,
            not leakage,
            upstream="exp7278-generation-requests",
            artifact_field="private_authority_fields",
        )
    )
    return checks


def collect_preconditions(
    root: Path,
    run_date: str = RUN_DATE,
    *,
    upstream_path: Path | None = None,
    raw_dir: Path | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Read external prerequisites or return one exact terminal block chain."""

    upstream_file = upstream_path or root / UPSTREAM_PATH
    source_dir = raw_dir or root / UPSTREAM_RAW_DIR
    checks = [
        gate_row(
            "run_date",
            RUN_DATE,
            run_date,
            run_date == RUN_DATE,
            upstream=None,
            artifact_field="run_date",
        ),
        gate_row(
            "required_upstream_measurement",
            "readable_file",
            str(upstream_file) if upstream_file.is_file() else "missing_or_unreadable",
            upstream_file.is_file() and os.access(upstream_file, os.R_OK),
            upstream="exp7278-source-measurement",
            artifact_field="results/experiment_7278_v640_source_measurement.json",
        ),
    ]
    if any(row["passed"] is False for row in checks):
        return checks, {}
    required_names = (
        "public_manifest.json",
        "private_authority_manifest.json",
        "schedule.json",
        "raw_call_manifest.json",
    )
    for name in required_names:
        path = source_dir / name
        checks.append(
            gate_row(
                "required_raw_input",
                "readable_file",
                str(path) if path.is_file() else "missing_or_unreadable",
                path.is_file() and os.access(path, os.R_OK),
                upstream="exp7278-source-measurement",
                artifact_field=name,
            )
        )
    if any(row["passed"] is False for row in checks):
        return checks, {}
    try:
        bundle = load_upstream_bundle(root, upstream_path=upstream_file, raw_dir=source_dir)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        checks.append(
            gate_row(
                "upstream_parse",
                "valid_json_and_yaml",
                f"{type(exc).__name__}:{exc}",
                False,
                upstream="exp7278-source-measurement",
                artifact_field="input_bytes",
            )
        )
        return checks, {}
    checks.extend(authenticate_bundle(bundle))
    spec_path = root / SPEC_PATH
    spec_ready = spec_path.is_file() and "REQ-VERIFY-7279" in spec_path.read_text(encoding="utf-8")
    checks.append(
        gate_row(
            "driving_capability",
            "REQ-VERIFY-7279",
            "REQ-VERIFY-7279" if spec_ready else "missing",
            spec_ready,
            upstream="openspec/capabilities/verification/spec.md",
            artifact_field="REQ-VERIFY-7279",
        )
    )
    return checks, bundle


def _iter_keys(value: Any) -> list[str]:
    """Collect nested field names without treating label vocabulary as leakage."""

    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_iter_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_iter_keys(item))
    return keys


def source_leakage_errors(
    schedule: Sequence[Mapping[str, Any]], raw_manifest: Mapping[str, Any]
) -> list[str]:
    """Detect private fields in requests and any declared authority-file access."""

    errors: list[str] = []
    for index, row in enumerate(schedule):
        leaked = sorted(set(_iter_keys(row)) & AUTHORITY_FIELDS)
        if leaked:
            errors.append(f"call_{index}:private_fields:{','.join(leaked)}")
    if raw_manifest.get("authority_path_opened_by_model_worker") is not False:
        errors.append("authority_path_opened_by_model_worker")
    return errors


def _decode_b64(value: Any) -> bytes:
    """Decode transport bytes strictly so corrupt evidence cannot be repaired."""

    if not isinstance(value, str):
        raise ValueError("base64_type")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("base64_invalid") from exc


def _parse_pointer(content: str, document: Mapping[str, Any]) -> tuple[JsonDict | None, list[str]]:
    """Parse the pointer grammar without using the producer's parser or compiler."""

    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None, ["pointer_json_parse"]
    if not isinstance(value, dict) or set(value) != {"outcome", "relations"}:
        return None, ["pointer_shape"]
    outcome = value.get("outcome")
    relations = value.get("relations")
    if outcome == "unknown" and relations == []:
        return {"outcome": "unknown", "relations": []}, []
    if outcome != "known" or not isinstance(relations, list) or not relations:
        return None, ["pointer_outcome"]
    pointers = {
        str(row.get("mention_id"))
        for row in document.get("mentions", [])
        if isinstance(row, Mapping)
    }
    parsed: list[JsonDict] = []
    required = {"object_pointer", "polarity", "predicate", "subject_pointer"}
    for relation in relations:
        if not isinstance(relation, dict) or set(relation) != required:
            return None, ["pointer_relation_shape"]
        if (
            relation.get("subject_pointer") not in pointers
            or relation.get("object_pointer") not in pointers
        ):
            return None, ["pointer_unknown_mention"]
        if relation.get("polarity") not in {"positive", "negative"}:
            return None, ["pointer_polarity"]
        if not isinstance(relation.get("predicate"), str) or not relation["predicate"]:
            return None, ["pointer_predicate"]
        parsed.append({key: relation[key] for key in sorted(required)})
    return {"outcome": "known", "relations": parsed}, []


def _parse_direct(content: str) -> tuple[str | None, list[str]]:
    """Parse the matched direct decision as strict JSON, including unknown."""

    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None, ["direct_json_parse"]
    if not isinstance(value, dict) or set(value) != {"decision"}:
        return None, ["direct_shape"]
    decision = value.get("decision")
    if decision not in DIRECT_DECISIONS:
        return None, ["direct_decision"]
    return str(decision), []


def _missing_replay_row(sealed: Mapping[str, Any]) -> JsonDict:
    """Represent absent raw evidence as a censored parser row, never a deletion."""

    return {
        "call_order": sealed.get("call_order"),
        "call_id": sealed.get("call_id"),
        "unit_id": sealed.get("unit_id"),
        "base_group_id": sealed.get("base_group_id"),
        "arm": sealed.get("arm"),
        "call_type": sealed.get("call_type"),
        "draw_index": sealed.get("draw_index"),
        "seed": sealed.get("seed"),
        "parse_valid": False,
        "parsed": None,
        "decision": None,
        "explicit_unknown": False,
        "abstention": True,
        "censored": True,
        "errors": ["missing_file"],
        "request_bytes_sha256": None,
        "response_bytes_sha256": None,
    }


def replay_raw_calls(bundle: Mapping[str, Any]) -> tuple[list[JsonDict], list[str]]:
    """Rebuild all parser outcomes from raw request and response bytes."""

    schedule = list(bundle["schedule"])
    call_files = bundle["call_files"]
    declared_calls = list(bundle["raw_manifest"].get("calls") or [])
    rows: list[JsonDict] = []
    mismatches: list[str] = []
    if len(schedule) != PLANNED_CALLS:
        mismatches.append("schedule_denominator")
    for index, sealed in enumerate(schedule):
        entry = call_files.get(index)
        if entry is None:
            mismatches.append(f"call_{index}:missing_file")
            rows.append(_missing_replay_row(sealed))
            continue
        document = entry["value"]
        stored_schedule = document.get("schedule")
        completion = document.get("completion")
        if stored_schedule != sealed:
            mismatches.append(f"call_{index}:schedule")
        if not isinstance(completion, Mapping):
            mismatches.append(f"call_{index}:completion")
            rows.append(_missing_replay_row(sealed))
            continue
        declared = declared_calls[index] if index < len(declared_calls) else {}
        if sha256_bytes(entry["bytes"]) != declared.get("sha256"):
            mismatches.append(f"call_{index}:file_sha256")
        errors: list[str] = []
        try:
            request_bytes = _decode_b64(completion.get("raw_request_bytes_b64"))
            response_bytes = _decode_b64(completion.get("raw_response_bytes_b64"))
            request = json.loads(request_bytes)
            response = json.loads(response_bytes)
            content = response["choices"][0]["message"]["content"]
            if not isinstance(request, dict) or not isinstance(content, str):
                raise ValueError("transport_shape")
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            mismatches.append(f"call_{index}:transport:{type(exc).__name__}:{exc}")
            rows.append(_missing_replay_row(sealed))
            continue
        request_sha = sha256_bytes(request_bytes)
        response_sha = sha256_bytes(response_bytes)
        if request_sha != completion.get("request_bytes_sha256") or request_sha != declared.get(
            "request_bytes_sha256"
        ):
            mismatches.append(f"call_{index}:request_sha256")
        if response_sha != completion.get("response_bytes_sha256") or response_sha != declared.get(
            "response_bytes_sha256"
        ):
            mismatches.append(f"call_{index}:response_sha256")
        if request != completion.get("actual_parameters"):
            mismatches.append(f"call_{index}:request_payload")
        messages = request.get("messages") or []
        request_contract = bool(
            messages
            and isinstance(messages[-1], Mapping)
            and messages[-1].get("content") == sealed.get("prompt")
            and request.get("grammar") == sealed.get("grammar")
            and request.get("max_tokens") == sealed.get("output_token_budget")
            and request.get("seed") == sealed.get("seed")
        )
        if not request_contract:
            mismatches.append(f"call_{index}:request_schedule_join")
        if content != completion.get("raw_completion"):
            mismatches.append(f"call_{index}:response_content")
        if sha256_bytes(content.encode("utf-8")) != completion.get("raw_completion_sha256"):
            mismatches.append(f"call_{index}:completion_sha256")
        if sealed.get("arm") == "mention_pointer":
            parsed, errors = _parse_pointer(content, sealed.get("document") or {})
            decision = None
            explicit_unknown = bool(parsed and parsed.get("outcome") == "unknown")
        else:
            decision, errors = _parse_direct(content)
            parsed = {"decision": decision} if decision is not None else None
            explicit_unknown = decision == "unknown"
        rows.append(
            {
                "call_order": index,
                "call_id": sealed.get("call_id"),
                "unit_id": sealed.get("unit_id"),
                "base_group_id": sealed.get("base_group_id"),
                "arm": sealed.get("arm"),
                "call_type": sealed.get("call_type"),
                "draw_index": sealed.get("draw_index"),
                "seed": sealed.get("seed"),
                "parse_valid": parsed is not None,
                "parsed": parsed,
                "decision": decision,
                "explicit_unknown": explicit_unknown,
                "abstention": explicit_unknown or parsed is None,
                "censored": completion.get("censored") is True,
                "errors": errors,
                "request_bytes_sha256": request_sha,
                "response_bytes_sha256": response_sha,
            }
        )
    return rows, list(dict.fromkeys(mismatches))


def _relation_surfaces(
    parsed: Mapping[str, Any], document: Mapping[str, Any]
) -> list[tuple[str, str, str, str]]:
    """Map public pointers to text surfaces before comparing two documents."""

    surfaces = {
        str(row.get("mention_id")): str(row.get("surface_text"))
        for row in document.get("mentions", [])
        if isinstance(row, Mapping)
    }
    return [
        (
            surfaces[str(relation["subject_pointer"])],
            str(relation["predicate"]),
            surfaces[str(relation["object_pointer"])],
            str(relation["polarity"]),
        )
        for relation in parsed.get("relations", [])
        if isinstance(relation, Mapping)
    ]


def _decide_pointer(
    source_document: Mapping[str, Any],
    claim_document: Mapping[str, Any],
    source_parsed: Mapping[str, Any] | None,
    claim_parsed: Mapping[str, Any] | None,
) -> tuple[str, list[str]]:
    """Execute source relations with a small authority independent of Exp7278."""

    if source_parsed is None or claim_parsed is None:
        return "unknown", ["missing_or_invalid_extraction"]
    if source_parsed.get("outcome") == "unknown" or claim_parsed.get("outcome") == "unknown":
        return "unknown", ["explicit_unknown"]
    source_relations = _relation_surfaces(source_parsed, source_document)
    claim_relations = _relation_surfaces(claim_parsed, claim_document)
    source_entities = {item for relation in source_relations for item in (relation[0], relation[2])}
    claim_entities = {item for relation in claim_relations for item in (relation[0], relation[2])}
    if not claim_entities.issubset(source_entities):
        return "unknown", ["missing_entity_mapping"]
    inverse = {
        "precedes": "follows",
        "follows": "precedes",
        "starts before": "starts after",
        "starts after": "starts before",
        "ends before": "ends after",
        "ends after": "ends before",
        "occurs before": "occurs after",
        "occurs after": "occurs before",
    }
    for claim in claim_relations:
        if claim in source_relations:
            return "supported", []
        reversed_same = (claim[2], claim[1], claim[0], claim[3])
        inverse_same = (claim[2], inverse.get(claim[1], ""), claim[0], claim[3])
        if reversed_same in source_relations or inverse_same in source_relations:
            return "contradicted", []
    return "unknown", ["relation_not_resolved"]


def _direct_vote(draws: Sequence[Mapping[str, Any]]) -> tuple[str, str, bool, list[str]]:
    """Apply the frozen two-draw rule while retaining the first draw separately."""

    values = [
        str(row.get("decision"))
        if row.get("parse_valid") is True and row.get("decision") in DIRECT_DECISIONS
        else "unknown"
        for row in draws[:2]
    ]
    while len(values) < 2:
        values.append("unknown")
    tied = values[0] != values[1]
    errors = [error for row in draws[:2] for error in row.get("errors", [])]
    return ("unknown" if tied else values[0], values[0], tied, errors)


def _source_shuffle_permutation(unit_ids: Sequence[str]) -> dict[str, str]:
    """Rebuild the frozen derangement without reading the producer summary."""

    shuffled = list(unit_ids)
    random.Random(727_820_260_914).shuffle(shuffled)
    return dict(zip(shuffled, shuffled[1:] + shuffled[:1], strict=True))


def _outcome_row(
    *,
    unit_id: str,
    base_group_id: str,
    condition: str,
    arm: str,
    seed: list[Any],
    prediction: str,
    expected: str,
    parse_valid: bool,
    source_exact: bool | None,
    claim_exact: bool | None,
    errors: Sequence[str],
    censored: bool,
    secondary: bool,
) -> JsonDict:
    """Create one unconditional unit-arm row with no hidden filtering."""

    abstention = prediction == "unknown"
    return {
        "unit_id": unit_id,
        "base_group_id": base_group_id,
        "condition": condition,
        "arm": arm,
        "seed": seed,
        "prediction": prediction,
        "expected_decision": expected,
        "decision_correct": prediction == expected,
        "false_accept": prediction != "unknown" and prediction != expected,
        "coverage": not abstention,
        "abstention": abstention,
        "parse_valid": parse_valid,
        "parse_error": not parse_valid,
        "source_exact": source_exact,
        "claim_exact": claim_exact,
        "errors": list(errors),
        "censored": censored,
        "secondary": secondary,
        "additional_model_calls": 0,
    }


def _ratio(numerator: int, denominator: int) -> JsonDict:
    """Keep count evidence beside its derived rate."""

    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def summarize_arms(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report each arm's numerators before any pooled comparison."""

    summaries: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        denominator = len(arm_rows)
        source_rows = [row for row in arm_rows if row.get("source_exact") is not None]
        claim_rows = [row for row in arm_rows if row.get("claim_exact") is not None]
        summaries[arm] = {
            "units": denominator,
            "accuracy": _ratio(
                sum(row.get("decision_correct") is True for row in arm_rows), denominator
            ),
            "false_accepts": _ratio(
                sum(row.get("false_accept") is True for row in arm_rows), denominator
            ),
            "coverage": _ratio(sum(row.get("coverage") is True for row in arm_rows), denominator),
            "parse_errors": _ratio(
                sum(row.get("parse_error") is True for row in arm_rows), denominator
            ),
            "abstentions": _ratio(
                sum(row.get("abstention") is True for row in arm_rows), denominator
            ),
            "censored": _ratio(sum(row.get("censored") is True for row in arm_rows), denominator),
            "source_exactness": _ratio(
                sum(row.get("source_exact") is True for row in source_rows), len(source_rows)
            ),
            "claim_exactness": _ratio(
                sum(row.get("claim_exact") is True for row in claim_rows), len(claim_rows)
            ),
            "secondary": arm == "direct_one_shot",
            "additional_model_calls": 0,
        }
    return summaries


def _matched_coverage_risk(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compare error only where both primary arms made a non-unknown decision."""

    by_identity = {(str(row["unit_id"]), str(row["arm"])): row for row in rows}
    units = sorted({str(row["unit_id"]) for row in rows})
    matched = [
        unit
        for unit in units
        if by_identity[(unit, "verifier")]["coverage"] is True
        and by_identity[(unit, "direct_self_consistency")]["coverage"] is True
    ]
    verifier_errors = sum(
        by_identity[(unit, "verifier")]["decision_correct"] is not True for unit in matched
    )
    direct_errors = sum(
        by_identity[(unit, "direct_self_consistency")]["decision_correct"] is not True
        for unit in matched
    )
    denominator = len(matched)
    return {
        "coverage_rule": "both_predictions_are_not_unknown",
        "threshold_selected_from_labels": False,
        "matched_units": denominator,
        "verifier_risk": _ratio(verifier_errors, denominator),
        "direct_self_consistency_risk": _ratio(direct_errors, denominator),
        "risk_difference": (
            (verifier_errors - direct_errors) / denominator if denominator else None
        ),
    }


def independent_reduce(bundle: Mapping[str, Any]) -> JsonDict:
    """Reduce raw bytes with the second parser and private authority."""

    replay_rows, mismatches = replay_raw_calls(bundle)
    public_rows = list(bundle["public"].get("rows") or [])
    authority_rows = list(bundle["authority"].get("rows") or [])
    authority_by_id = {str(row["unit_id"]): row for row in authority_rows}
    public_by_id = {str(row["unit_id"]): row for row in public_rows}
    calls_by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        calls_by_unit[str(row.get("unit_id"))].append(row)
    primary_rows: dict[tuple[str, str], JsonDict] = {}
    pointer_calls: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    rows: list[JsonDict] = []
    for public in public_rows:
        unit_id = str(public["unit_id"])
        authority = authority_by_id[unit_id]
        calls = calls_by_unit[unit_id]
        source = next(
            (
                row
                for row in calls
                if row.get("arm") == "mention_pointer" and row.get("call_type") == "source"
            ),
            _missing_replay_row({"unit_id": unit_id, "base_group_id": authority["base_group_id"]}),
        )
        claim = next(
            (
                row
                for row in calls
                if row.get("arm") == "mention_pointer" and row.get("call_type") == "claim"
            ),
            _missing_replay_row({"unit_id": unit_id, "base_group_id": authority["base_group_id"]}),
        )
        pointer_calls[unit_id] = (source, claim)
        source_parsed = source.get("parsed") if isinstance(source.get("parsed"), Mapping) else None
        claim_parsed = claim.get("parsed") if isinstance(claim.get("parsed"), Mapping) else None
        prediction, errors = _decide_pointer(
            public["source"], public["claim"], source_parsed, claim_parsed
        )
        verifier = _outcome_row(
            unit_id=unit_id,
            base_group_id=str(authority["base_group_id"]),
            condition=str(authority["condition"]),
            arm="verifier",
            seed=[source.get("seed"), claim.get("seed")],
            prediction=prediction,
            expected=str(authority["expected_decision"]),
            parse_valid=source.get("parse_valid") is True and claim.get("parse_valid") is True,
            source_exact=source_parsed == authority.get("gold_source_completion"),
            claim_exact=claim_parsed == authority.get("gold_claim_completion"),
            errors=[*source.get("errors", []), *claim.get("errors", []), *errors],
            censored=source.get("censored") is True or claim.get("censored") is True,
            secondary=False,
        )
        rows.append(verifier)
        primary_rows[(unit_id, "verifier")] = verifier
        direct_draws = sorted(
            (row for row in calls if row.get("arm") == "direct_self_consistency"),
            key=lambda row: int(row.get("draw_index") or 0),
        )
        direct, one_shot, tied, direct_errors = _direct_vote(direct_draws)
        direct_row = _outcome_row(
            unit_id=unit_id,
            base_group_id=str(authority["base_group_id"]),
            condition=str(authority["condition"]),
            arm="direct_self_consistency",
            seed=[row.get("seed") for row in direct_draws],
            prediction=direct,
            expected=str(authority["expected_decision"]),
            parse_valid=len(direct_draws) == 2
            and all(row.get("parse_valid") is True for row in direct_draws),
            source_exact=None,
            claim_exact=None,
            errors=[*direct_errors, *(["self_consistency_tie"] if tied else [])],
            censored=any(row.get("censored") is True for row in direct_draws),
            secondary=False,
        )
        rows.append(direct_row)
        primary_rows[(unit_id, "direct_self_consistency")] = direct_row
        first = direct_draws[0] if direct_draws else {}
        rows.append(
            _outcome_row(
                unit_id=unit_id,
                base_group_id=str(authority["base_group_id"]),
                condition=str(authority["condition"]),
                arm="direct_one_shot",
                seed=[first.get("seed")],
                prediction=one_shot,
                expected=str(authority["expected_decision"]),
                parse_valid=first.get("parse_valid") is True,
                source_exact=None,
                claim_exact=None,
                errors=first.get("errors", []),
                censored=first.get("censored") is True,
                secondary=True,
            )
        )
    unit_ids = [str(row["unit_id"]) for row in public_rows]
    permutation = _source_shuffle_permutation(unit_ids)
    for target_id in unit_ids:
        source_id = permutation[target_id]
        target_public = public_by_id[target_id]
        source_public = public_by_id[source_id]
        authority = authority_by_id[target_id]
        source, _ = pointer_calls[source_id]
        _, claim = pointer_calls[target_id]
        source_parsed = source.get("parsed") if isinstance(source.get("parsed"), Mapping) else None
        claim_parsed = claim.get("parsed") if isinstance(claim.get("parsed"), Mapping) else None
        prediction, errors = _decide_pointer(
            source_public["source"], target_public["claim"], source_parsed, claim_parsed
        )
        source_authority = authority_by_id[source_id]
        rows.append(
            _outcome_row(
                unit_id=target_id,
                base_group_id=str(authority["base_group_id"]),
                condition=str(authority["condition"]),
                arm="source_shuffle_control",
                seed=[source.get("seed"), claim.get("seed")],
                prediction=prediction,
                expected=str(authority["expected_decision"]),
                parse_valid=source.get("parse_valid") is True and claim.get("parse_valid") is True,
                source_exact=source_parsed == source_authority.get("gold_source_completion"),
                claim_exact=claim_parsed == authority.get("gold_claim_completion"),
                errors=errors,
                censored=source.get("censored") is True or claim.get("censored") is True,
                secondary=False,
            )
        )
    summaries = summarize_arms(rows)
    return {
        "raw_replay_rows": replay_rows,
        "replay_mismatches": mismatches,
        "rows": rows,
        "arm_summaries": summaries,
        "source_shuffle_permutation": permutation,
        "source_leakage_errors": source_leakage_errors(bundle["schedule"], bundle["raw_manifest"]),
        "matched_coverage_risk": _matched_coverage_risk(rows),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    """Select a deterministic empirical percentile from frozen draws."""

    ordered = sorted(values)
    return ordered[int(probability * (len(ordered) - 1))]


def paired_bootstrap(
    rows: Sequence[Mapping[str, Any]], seed: int, draws: int = BOOTSTRAP_DRAWS
) -> list[JsonDict]:
    """Resample 16 bases while retaining four correlated conditions per base."""

    if draws <= 0:
        raise ValueError("bootstrap_draws")
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("arm") in ARMS:
            grouped[str(row["base_group_id"])][str(row["arm"])].append(row)
    groups = sorted(grouped)
    if len(groups) != PLANNED_BASE_GROUPS or any(
        len(grouped[group][arm]) != 4 for group in groups for arm in ARMS
    ):
        raise ValueError("paired_base_groups")
    definitions = (
        (
            "verifier_minus_direct_self_consistency",
            "verifier",
            "direct_self_consistency",
            "accuracy_difference",
            "decision_correct",
        ),
        (
            "verifier_minus_direct_self_consistency",
            "verifier",
            "direct_self_consistency",
            "false_accept_difference",
            "false_accept",
        ),
        (
            "verifier_minus_direct_self_consistency",
            "verifier",
            "direct_self_consistency",
            "coverage_difference",
            "coverage",
        ),
        (
            "verifier_minus_source_shuffle_control",
            "verifier",
            "source_shuffle_control",
            "accuracy_difference",
            "decision_correct",
        ),
    )
    generator = random.Random(seed)
    sampled_groups = [
        [groups[generator.randrange(len(groups))] for _ in groups] for _ in range(draws)
    ]
    result: list[JsonDict] = []
    for comparison, treatment, control, metric, field in definitions:
        treatment_rows = [row for row in rows if row.get("arm") == treatment]
        control_rows = [row for row in rows if row.get("arm") == control]
        treatment_numerator = sum(row.get(field) is True for row in treatment_rows)
        control_numerator = sum(row.get(field) is True for row in control_rows)
        point = treatment_numerator / len(treatment_rows) - control_numerator / len(control_rows)
        values = []
        for sample in sampled_groups:
            differences = [
                float(row.get(field) is True)
                for group in sample
                for row in grouped[group][treatment]
            ]
            control_values = [
                float(row.get(field) is True) for group in sample for row in grouped[group][control]
            ]
            values.append(
                sum(differences) / len(differences) - sum(control_values) / len(control_values)
            )
        result.append(
            {
                "comparison": comparison,
                "metric": metric,
                "treatment_arm": treatment,
                "control_arm": control,
                "treatment": {
                    "numerator": treatment_numerator,
                    "denominator": len(treatment_rows),
                },
                "control": {
                    "numerator": control_numerator,
                    "denominator": len(control_rows),
                },
                "estimate": point,
                "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
                "bootstrap_draws": draws,
                "independent_base_groups": len(groups),
                "conditions_per_base_group": 4,
                "resampling_unit": "base_group",
            }
        )
    return result


def causal_controls(bundle: Mapping[str, Any], reduced: Mapping[str, Any]) -> list[JsonDict]:
    """Run fixed authority, source, label, naming, and deletion interventions."""

    selection = dict(bundle["upstream"].get("selection_receipt") or {})
    authority_changed = deepcopy(bundle["authority"])
    first_label = authority_changed["rows"][0]["expected_decision"]
    authority_changed["rows"][0]["expected_decision"] = (
        "unknown" if first_label != "unknown" else "supported"
    )
    authority_hash = sha256_bytes(canonical_json(authority_changed).encode("utf-8"))
    public_changed = deepcopy(bundle["public"])
    public_changed["rows"][0]["source"]["text"] = "source replacement control"
    public_hash = sha256_bytes(canonical_json(public_changed).encode("utf-8"))
    shuffled = deepcopy(bundle["authority"])
    labels = [row["expected_decision"] for row in shuffled["rows"]]
    rotated = labels[1:] + labels[:1]
    for row, label in zip(shuffled["rows"], rotated, strict=True):
        row["expected_decision"] = label
    shuffled_hash = sha256_bytes(canonical_json(shuffled).encode("utf-8"))
    by_group: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in reduced["rows"]:
        if row["arm"] in PRIMARY_ARMS:
            by_group[str(row["base_group_id"])][str(row["arm"])][str(row["condition"])] = row
    renaming_checks = [
        conditions["supported"]["prediction"]
        == conditions["consistent_entity_renaming"]["prediction"]
        for arms in by_group.values()
        for conditions in arms.values()
    ]
    deleted = dict(bundle)
    deleted["call_files"] = dict(bundle["call_files"])
    deleted["call_files"].pop(0, None)
    deleted_rows, deleted_errors = replay_raw_calls(deleted)
    deleted_detected = (
        "call_0:missing_file" in deleted_errors and deleted_rows[0]["censored"] is True
    )
    return [
        {
            "control": "wrong_authority",
            "expected": "changed authority hash is rejected",
            "observed": {
                "changed_sha256": authority_hash,
                "authenticated_sha256": selection.get("private_authority_sha256"),
            },
            "passed": authority_hash != selection.get("private_authority_sha256"),
            "principle": "An alternate label authority must not silently replace the sealed evaluator.",
        },
        {
            "control": "source_replacement",
            "expected": "changed public source hash is rejected",
            "observed": {
                "changed_sha256": public_hash,
                "authenticated_sha256": selection.get("public_manifest_sha256"),
            },
            "passed": public_hash != selection.get("public_manifest_sha256"),
            "principle": "A source edit must change authenticated input identity.",
        },
        {
            "control": "shuffled_labels",
            "expected": "shuffled authority hash is rejected",
            "observed": {
                "changed_sha256": shuffled_hash,
                "authenticated_sha256": selection.get("private_authority_sha256"),
                "changed_label_count": sum(
                    left != right for left, right in zip(labels, rotated, strict=True)
                ),
            },
            "passed": shuffled_hash != selection.get("private_authority_sha256"),
            "principle": "Label order must remain bound to each public unit before scoring.",
        },
        {
            "control": "consistent_renaming",
            "expected": {"invariant_pairs": len(renaming_checks)},
            "observed": {
                "invariant_pairs": sum(renaming_checks),
                "total_pairs": len(renaming_checks),
            },
            "passed": bool(renaming_checks) and all(renaming_checks),
            "principle": "A proof-preserving entity rename should not change either primary decision.",
        },
        {
            "control": "deleted_evidence",
            "expected": "missing call is a mismatch and censored row",
            "observed": {
                "missing_call_error": "call_0:missing_file" in deleted_errors,
                "censored_row": bool(deleted_rows and deleted_rows[0].get("censored") is True),
            },
            "passed": deleted_detected,
            "principle": "Deleted raw evidence must become a visible failure, never a smaller denominator.",
        },
    ]


def _comparison(
    comparisons: Sequence[Mapping[str, Any]], comparison: str, metric: str
) -> Mapping[str, Any]:
    """Select one preregistered interval and reject duplicates."""

    matches = [
        row
        for row in comparisons
        if row.get("comparison") == comparison and row.get("metric") == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"comparison:{comparison}:{metric}")
    return matches[0]


def acceptance_results(
    reduced: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Apply frozen completeness, scientific-value, and independence criteria."""

    accuracy = _comparison(
        comparisons, "verifier_minus_direct_self_consistency", "accuracy_difference"
    )
    false_accept = _comparison(
        comparisons, "verifier_minus_direct_self_consistency", "false_accept_difference"
    )
    shuffle = _comparison(
        comparisons, "verifier_minus_source_shuffle_control", "accuracy_difference"
    )
    arm_counts = {arm: sum(row.get("arm") == arm for row in reduced["rows"]) for arm in ARMS}
    groups = {row["base_group_id"] for row in reduced["rows"]}
    values = [
        (
            "fixed_rows_accounted",
            "completeness",
            {arm: PLANNED_UNITS for arm in ARMS},
            arm_counts,
            arm_counts == {arm: PLANNED_UNITS for arm in ARMS},
        ),
        (
            "fixed_base_groups",
            "completeness",
            PLANNED_BASE_GROUPS,
            len(groups),
            len(groups) == PLANNED_BASE_GROUPS,
        ),
        (
            "causal_controls_disposed",
            "completeness",
            5,
            len(controls),
            len(controls) == 5 and all("passed" in row for row in controls),
        ),
        (
            "accuracy_ci95_lower_above_zero",
            "value",
            ">0",
            accuracy["ci95"][0],
            accuracy["ci95"][0] > 0,
        ),
        (
            "false_accept_ci95_upper_at_most_zero",
            "value",
            "<=0",
            false_accept["ci95"][1],
            false_accept["ci95"][1] <= 0,
        ),
        (
            "no_source_leakage",
            "independence",
            [],
            reduced["source_leakage_errors"],
            not reduced["source_leakage_errors"],
        ),
        (
            "independent_replay_match",
            "independence",
            [],
            reduced["replay_mismatches"],
            not reduced["replay_mismatches"],
        ),
        (
            "beats_fixed_source_shuffle",
            "value",
            ">0",
            shuffle["estimate"],
            shuffle["estimate"] > 0,
        ),
        (
            "independence_controls_pass",
            "independence",
            True,
            all(row.get("passed") is True for row in controls),
            all(row.get("passed") is True for row in controls),
        ),
    ]
    return [
        {
            "criterion": name,
            "category": category,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
            "principle": GATE_PRINCIPLES[name],
        }
        for name, category, expected, observed, passed in values
    ]


def classify_result(criteria: Sequence[Mapping[str, Any]], *, complete: bool) -> JsonDict:
    """Keep complete negative science distinct from unfinished task work."""

    completeness_ok = complete and all(
        row.get("passed") is True for row in criteria if row.get("category") == "completeness"
    )
    if not completeness_ok:
        return {
            "status": "running",
            "source_audit_complete_score": 0,
            "source_promotion_score": 0,
            "verdict_class": "partial",
            "honest_verdict": "partial_source_audit_unfinished_checkpoint_only",
        }
    promotion = int(
        all(
            row.get("passed") is True
            for row in criteria
            if row.get("category") in {"value", "independence"}
        )
    )
    if promotion:
        return {
            "status": "complete",
            "source_audit_complete_score": 1,
            "source_promotion_score": 1,
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_circular_positive_source_pilot_all_fixed_criteria_passed",
        }
    return {
        "status": "complete",
        "source_audit_complete_score": 1,
        "source_promotion_score": 0,
        "verdict_class": "null",
        "honest_verdict": "complete_null_verifier_did_not_beat_matched_direct_self_consistency",
    }


def _validations_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require each focused command once and require every command to pass."""

    names = [row.get("name") for row in receipts]
    return sorted(names) == sorted(REQUIRED_VALIDATION_NAMES) and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in receipts
    )


def _source_hashes(root: Path, bundle: Mapping[str, Any]) -> JsonDict:
    """Hash code, configuration, manifests, and the full raw call set."""

    paths = {
        "agents": root / "AGENTS.md",
        "claude": root / "CLAUDE.md",
        "codex": root / "CODEX.md",
        "research_program": root / "research-program.md",
        "exclusion_manifest": root / EXCLUSION_PATH,
        "e2e_test_plan": root / "ops/e2e-test-plan.md",
        "verification_spec": root / SPEC_PATH,
        "upstream_measurement": bundle["paths"]["upstream"],
        "public_manifest": bundle["paths"]["public"],
        "private_authority_manifest": bundle["paths"]["authority"],
        "schedule": bundle["paths"]["schedule"],
        "raw_call_manifest": bundle["paths"]["raw_manifest"],
        "module": root / MODULE_PATH,
        "entrypoint": root / WRAPPER_PATH,
        "focused_tests": root / TEST_PATH,
    }
    result = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else "missing",
            "retired": False,
            "quarantined": False,
        }
        for name, path in paths.items()
    }
    call_hashes = [
        {
            "call_order": index,
            "sha256": sha256_bytes(entry["bytes"]),
        }
        for index, entry in sorted(bundle["call_files"].items())
    ]
    result["raw_call_files"] = {
        "count": len(call_hashes),
        "combined_sha256": sha256_bytes(canonical_json(call_hashes).encode("utf-8")),
        "retired": False,
        "quarantined": False,
    }
    return result


def base_artifact(run_date: str) -> JsonDict:
    """Create every required field before a fallible source read."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "running",
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "random_seed": {"audit": RANDOM_SEED, "bootstrap": BOOTSTRAP_SEED},
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_base_groups": PLANNED_BASE_GROUPS,
            "planned_units": PLANNED_UNITS,
            "planned_calls": PLANNED_CALLS,
            "planned_audit_rows": PLANNED_UNITS * len(ARMS),
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": PLANNED_UNITS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "stopping_rule": "authenticate and reduce every frozen raw call once; never stop on accuracy",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary([]),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_source_audit_unfinished_checkpoint_only",
        "verdict_class": "partial",
        "validation_receipts": [],
        "source_audit_complete_score": 0,
        "source_promotion_score": 0,
        "paired_comparisons": [],
        "causal_control_rows": [],
        "claim_boundary": {
            "population": "16-base synthetic source pilot",
            "broad_hallucination_claim": False,
            "gsm8k_claim": False,
            "model_wide_superiority_claim": False,
        },
        "arm_summaries": {},
        "matched_coverage_risk": {},
        "raw_replay_rows": [],
        "replay_mismatches": [],
        "source_leakage_errors": [],
        "authority_boundary": {},
        "historical_model_receipts": {},
        "timestamps": {"started_at_utc": _utc_now(), "completed_at_utc": None},
        "phase_spans": [],
        "positive_control_results": {},
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Finish an external failure as blocked, never as partial measurement."""

    artifact["status"] = "blocked"
    artifact["preconditions_checked"] = deepcopy(list(checks))
    artifact["gate_check_summary"] = gate_summary(checks)
    artifact["MODEL_SPECS"] = []
    artifact["model_invoked"] = False
    artifact["invocation_counts"] = deepcopy(ZERO_INVOCATION_COUNTS)
    artifact["inference_substrate"] = "blocked_before_qualifying_computation"
    artifact["inference_substrate_class"] = "blocked_no_run"
    artifact["source_audit_complete_score"] = 0
    artifact["source_promotion_score"] = 0
    artifact["verdict_class"] = "blocked"
    failure = artifact["gate_check_summary"].get("failed_check") or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_source_audit_{failure}"
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = _utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def assemble_measured_artifact(
    root: Path,
    bundle: Mapping[str, Any],
    reduced: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Assemble one measured artifact without consulting producer summaries."""

    criteria = acceptance_results(reduced, comparisons, controls)
    validation_ok = _validations_complete(validation_receipts)
    criteria.append(
        {
            "criterion": "focused_validation",
            "category": "publication",
            "expected": True,
            "observed": validation_ok,
            "passed": validation_ok,
            "principle": GATE_PRINCIPLES["focused_validation"],
        }
    )
    outcome = classify_result(criteria, complete=True)
    artifact = base_artifact(RUN_DATE)
    artifact.update(outcome)
    artifact.update(
        {
            "preconditions_checked": authenticate_bundle(bundle),
            "inference_substrate": "cpu_exact_solver_or_simulator",
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "duration_s": duration_s,
            "source_artifact_hashes": _source_hashes(root, bundle),
            "rows": deepcopy(list(reduced["rows"])),
            "sample_size_budget": {
                "planned_base_groups": PLANNED_BASE_GROUPS,
                "planned_units": PLANNED_UNITS,
                "planned_calls": PLANNED_CALLS,
                "planned_audit_rows": PLANNED_UNITS * len(ARMS),
                "attempted_units": PLANNED_UNITS,
                "completed_units": PLANNED_UNITS,
                "censored_units": sum(
                    any(
                        row.get("unit_id") == unit and row.get("censored") is True
                        for row in reduced["rows"]
                    )
                    for unit in {row["unit_id"] for row in reduced["rows"]}
                ),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "stopping_rule": "authenticate and reduce every frozen raw call once; never stop on accuracy",
            },
            "acceptance_gate_results": criteria,
            "gate_check_summary": gate_summary(authenticate_bundle(bundle)),
            "validation_receipts": deepcopy(list(validation_receipts)),
            "paired_comparisons": deepcopy(list(comparisons)),
            "causal_control_rows": deepcopy(list(controls)),
            "arm_summaries": deepcopy(dict(reduced["arm_summaries"])),
            "matched_coverage_risk": deepcopy(dict(reduced["matched_coverage_risk"])),
            "raw_replay_rows": deepcopy(list(reduced["raw_replay_rows"])),
            "replay_mismatches": list(reduced["replay_mismatches"]),
            "source_leakage_errors": list(reduced["source_leakage_errors"]),
            "authority_boundary": {
                "public_input": {
                    "sha256": sha256_bytes(bundle["public_bytes"]),
                    "model_visible": True,
                },
                "private_authority": {
                    "sha256": sha256_bytes(bundle["authority_bytes"]),
                    "model_visible": False,
                    "joined_after_raw_parsing": True,
                },
                "second_authority_implementation": MODULE_PATH.as_posix(),
                "producer_summary_helpers_invoked": False,
            },
            "historical_model_receipts": {
                "upstream_measurement": {
                    "path": UPSTREAM_PATH.as_posix(),
                    "sha256": sha256_bytes(bundle["upstream_bytes"]),
                    "current_invocation": False,
                },
                "raw_call_manifest": {
                    "path": (UPSTREAM_RAW_DIR / "raw_call_manifest.json").as_posix(),
                    "sha256": sha256_bytes(bundle["raw_manifest_bytes"]),
                    "current_invocation": False,
                },
            },
            "timestamps": {
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
            },
            "phase_spans": deepcopy(list(phase_spans)),
            "positive_control_results": {
                "arm": "direct_self_consistency",
                "accuracy": deepcopy(
                    reduced["arm_summaries"]["direct_self_consistency"]["accuracy"]
                ),
                "passed": reduced["arm_summaries"]["direct_self_consistency"]["accuracy"][
                    "numerator"
                ]
                > 0,
                "circular_control_only": True,
            },
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object, *, require_validations: bool) -> list[str]:
    """Cold-check terminal schema, fixed rows, derived metrics, and checksums."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        value.get("schema"),
        value.get("experiment_id"),
        value.get("milestone"),
        value.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity")
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract")
    if value.get("execution_venue") != "host" or not value.get("execution_host"):
        errors.append("execution_identity")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    if value.get("status") == "blocked":
        if (
            value.get("verdict_class") != "blocked"
            or not str(value.get("honest_verdict", "")).startswith("blocked_")
            or value.get("source_audit_complete_score") != 0
            or value.get("source_promotion_score") != 0
            or value.get("rows") != []
            or dict(value.get("gate_check_summary") or {}).get("passed") is not False
            or value.get("inference_substrate_class") != "blocked_no_run"
        ):
            errors.append("blocked_terminal_state")
        return list(dict.fromkeys(errors))
    if value.get("status") != "complete":
        errors.append("status")
        return list(dict.fromkeys(errors))
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != PLANNED_UNITS * len(ARMS):
        errors.append("row_denominator")
        return list(dict.fromkeys(errors))
    arm_counts = {arm: sum(row.get("arm") == arm for row in rows) for arm in ARMS}
    if arm_counts != {arm: PLANNED_UNITS for arm in ARMS}:
        errors.append("arm_denominators")
    if value.get("arm_summaries") != summarize_arms(rows):
        errors.append("arm_summaries")
    try:
        comparisons = paired_bootstrap(rows, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    except ValueError as exc:
        errors.append(str(exc))
        comparisons = []
    if value.get("paired_comparisons") != comparisons:
        errors.append("paired_comparisons")
    controls = value.get("causal_control_rows")
    if not isinstance(controls, list) or {row.get("control") for row in controls} != {
        "wrong_authority",
        "source_replacement",
        "shuffled_labels",
        "consistent_renaming",
        "deleted_evidence",
    }:
        errors.append("causal_control_rows")
    if comparisons and isinstance(controls, list):
        scientific = acceptance_results(value, comparisons, controls)
        observed_scientific = [
            row
            for row in value.get("acceptance_gate_results", [])
            if row.get("category") != "publication"
        ]
        if observed_scientific != scientific:
            errors.append("acceptance_gate_results")
        outcome = classify_result(scientific, complete=True)
        for field in (
            "status",
            "source_audit_complete_score",
            "source_promotion_score",
            "verdict_class",
            "honest_verdict",
        ):
            if value.get(field) != outcome[field]:
                errors.append("terminal_classification")
                break
    if (
        value.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or value.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
    ):
        errors.append("inference_substrate")
    if require_validations and not _validations_complete(value.get("validation_receipts") or []):
        errors.append("validation_receipts")
    boundary = value.get("claim_boundary") or {}
    if boundary.get("population") != "16-base synthetic source pilot" or any(
        boundary.get(field) is not False
        for field in (
            "broad_hallucination_claim",
            "gsm8k_claim",
            "model_wide_superiority_claim",
        )
    ):
        errors.append("claim_boundary")
    return list(dict.fromkeys(errors))


def write_terminal_artifact(artifact: Mapping[str, Any], path: Path) -> None:
    """Publish only a cold-valid terminal artifact with complete validations."""

    errors = validate_artifact(artifact, require_validations=True)
    if errors:
        raise ValueError(f"invalid Exp7279 artifact: {errors}")
    atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)


def _pending_receipts() -> list[JsonDict]:  # pragma: no cover - runtime candidate only.
    """Keep validation incomplete until every subprocess returns an exit code."""

    return [
        {
            "name": name,
            "command": "pending",
            "exit_code": None,
            "passed": False,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"results/raw/experiment_7279/validation/{name}.log",
            "log_sha256": "pending",
        }
        for name in REQUIRED_VALIDATION_NAMES
    ]


def validation_commands(
    root: Path, candidate: Path
) -> list[tuple[str, list[str]]]:  # pragma: no cover
    """Return focused validation commands and never a repository-wide test run."""

    python = str(root / ".venv/bin/python")
    coverage_file = "/tmp/.coverage-exp7279-v640"
    changed = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), TEST_PATH.as_posix()]
    affected = [
        "tests/python/test_experiment_7239_v637_semantic_audit.py",
        "tests/python/test_experiment_7278_v640_source_measurement.py",
    ]
    return [
        (
            "focused_pytest",
            [
                python,
                "-u",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7279-focused",
                TEST_PATH.as_posix(),
                "-q",
            ],
        ),
        (
            "affected_suites",
            [
                python,
                "-u",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7279-affected",
                *affected,
                "-q",
            ],
        ),
        (
            "scoped_coverage",
            [
                python,
                "-u",
                "-m",
                "coverage",
                "run",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7279-coverage",
                TEST_PATH.as_posix(),
                "-q",
            ],
        ),
        (
            "scoped_coverage_report",
            [
                python,
                "-u",
                "-m",
                "coverage",
                "report",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        ("ruff_check", [python, "-u", "-m", "ruff", "check", *changed]),
        ("ruff_format", [python, "-u", "-m", "ruff", "format", "--check", *changed]),
        ("mypy", [python, "-u", "-m", "mypy", MODULE_PATH.as_posix()]),
        (
            "scoped_spec_coverage",
            [python, "-u", "scripts/check_spec_coverage.py", TEST_PATH.as_posix(), *affected],
        ),
        (
            "independent_raw_replay",
            [
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--replay-raw",
                str(root / UPSTREAM_RAW_DIR),
            ],
        ),
        ("adversarial_verify", [python, "-u", "scripts/adversarial_verify.py", str(candidate)]),
        (
            "verdict_row_consistency",
            [python, "-u", "scripts/verdict_row_consistency_lint.py", str(candidate)],
        ),
    ]


def _run_validations(root: Path, candidate: Path) -> list[JsonDict]:  # pragma: no cover
    """Stream each checker and emit a truthful heartbeat while it blocks."""

    validation_dir = candidate.parent / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    receipts: list[JsonDict] = []
    commands = validation_commands(root, candidate)
    for index, (name, command) in enumerate(commands, start=1):
        _progress(7, "subprocess_start", operation=name, completed=index - 1, total=len(commands))
        started = time.monotonic()
        process = subprocess.Popen(  # noqa: S603 - fixed local command list.
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(45.0):
                _progress(
                    7,
                    "subprocess_heartbeat",
                    operation=name,
                    elapsed_s=round(time.monotonic() - started, 3),
                    completed=index - 1,
                    total=len(commands),
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            print(f"[exp7279:{name}] {line.rstrip()}", flush=True)
        return_code = process.wait()
        stop.set()
        thread.join(timeout=1.0)
        log_path = validation_dir / f"{name}.log"
        log_path.write_text("".join(lines), encoding="utf-8")
        receipt = {
            "name": name,
            "command": shlex.join(command),
            "exit_code": return_code,
            "passed": return_code == 0,
            "timed_out": False,
            "duration_s": time.monotonic() - started,
            "log_path": log_path.relative_to(root).as_posix(),
            "log_sha256": sha256_file(log_path),
        }
        receipts.append(receipt)
        _progress(
            7,
            "subprocess_end",
            operation=name,
            exit_code=return_code,
            completed=index,
            total=len(commands),
        )
    return receipts


def _write_checkpoint(path: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover
    """Write unfinished state only below the task checkpoint directory."""

    value = deepcopy(dict(artifact))
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=False, sort_keys=True)


def run_experiment(
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    output_root: Path | None = None,
    validation_runner: Callable[[Path, Path], list[JsonDict]] = _run_validations,
) -> JsonDict:  # pragma: no cover - terminal filesystem orchestration.
    """Build, validate, and atomically publish the fixed-date source audit."""

    started = time.monotonic()
    started_utc = _utc_now()
    repository = root or find_repo_root(start=__file__)
    destination = output_root or repository
    checkpoint = destination / CHECKPOINT_PATH
    candidate = destination / RAW_CANDIDATE_PATH
    result_path = destination / RESULT_PATH
    spans: list[JsonDict] = []
    _progress(0, "phase_start", name="checkpoint_and_paths")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    artifact = base_artifact(run_date)
    _write_checkpoint(checkpoint, artifact)
    _progress(0, "phase_end", name="checkpoint_and_paths")

    phase_started = time.monotonic()
    _progress(1, "phase_start", name="authenticate_inputs")
    checks, bundle = collect_preconditions(repository, run_date)
    output_state = {
        "result_absent": not result_path.exists(),
        "result_parent_writable": os.access(result_path.parent, os.W_OK),
        "raw_parent_writable": os.access(candidate.parent, os.W_OK),
    }
    checks.append(
        gate_row(
            "authenticated_output_paths",
            {key: True for key in output_state},
            output_state,
            all(output_state.values()),
            upstream="host_filesystem",
            artifact_field="task_owned_output_paths",
        )
    )
    spans.append(
        {"phase": 1, "name": "authenticate_inputs", "duration_s": time.monotonic() - phase_started}
    )
    failure = next((row for row in checks if row.get("passed") is not True), None)
    _progress(
        1,
        "phase_end",
        name="authenticate_inputs",
        failed_check=failure.get("check") if failure else None,
    )
    if failure is not None:
        artifact = finalize_blocked_artifact(
            artifact, checks, duration_s=time.monotonic() - started
        )
        artifact["phase_spans"] = spans
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        errors = validate_artifact(artifact, require_validations=False)
        if errors:
            raise ValueError(f"invalid blocked Exp7279 artifact: {errors}")
        _progress(8, "write_start", path=str(result_path))
        atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
        _progress(8, "write_end", path=str(result_path))
        return artifact

    phase_started = time.monotonic()
    _progress(2, "benchmark_start", name="independent_raw_replay", total_calls=PLANNED_CALLS)
    reduced = independent_reduce(bundle)
    spans.append(
        {
            "phase": 2,
            "name": "independent_raw_replay",
            "duration_s": time.monotonic() - phase_started,
        }
    )
    _progress(
        2,
        "benchmark_end",
        name="independent_raw_replay",
        completed_calls=len(reduced["raw_replay_rows"]),
    )

    phase_started = time.monotonic()
    _progress(3, "benchmark_start", name="base_group_bootstrap", draws=BOOTSTRAP_DRAWS)
    comparisons = paired_bootstrap(reduced["rows"], BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    spans.append(
        {"phase": 3, "name": "base_group_bootstrap", "duration_s": time.monotonic() - phase_started}
    )
    _progress(3, "benchmark_end", name="base_group_bootstrap", interval_rows=len(comparisons))

    phase_started = time.monotonic()
    _progress(4, "benchmark_start", name="causal_controls", total_controls=5)
    controls = causal_controls(bundle, reduced)
    spans.append(
        {"phase": 4, "name": "causal_controls", "duration_s": time.monotonic() - phase_started}
    )
    _progress(4, "benchmark_end", name="causal_controls", completed_controls=len(controls))

    _progress(5, "phase_start", name="raw_terminal_candidate")
    candidate_artifact = assemble_measured_artifact(
        repository,
        bundle,
        reduced,
        comparisons,
        controls,
        validation_receipts=_pending_receipts(),
        started_at_utc=started_utc,
        completed_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    atomic_write_json(candidate, candidate_artifact, allow_override=False, sort_keys=True)
    _progress(5, "phase_end", name="raw_terminal_candidate", path=str(candidate))

    _progress(6, "phase_start", name="focused_validation")
    receipts = validation_runner(repository, candidate)
    _progress(6, "phase_end", name="focused_validation", passed=_validations_complete(receipts))
    artifact = assemble_measured_artifact(
        repository,
        bundle,
        reduced,
        comparisons,
        controls,
        validation_receipts=receipts,
        started_at_utc=started_utc,
        completed_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    errors = validate_artifact(artifact, require_validations=True)
    if errors:
        failure_path = candidate.parent / "validation-failure-receipt.json"
        atomic_write_json(
            failure_path,
            {"errors": errors, "validation_receipts": receipts, "observed_at_utc": _utc_now()},
            allow_override=False,
            sort_keys=True,
        )
        raise ValueError(f"invalid Exp7279 terminal artifact: {errors}")
    _write_checkpoint(checkpoint, artifact)
    _progress(8, "write_start", path=str(result_path))
    write_terminal_artifact(artifact, result_path)
    _progress(8, "write_end", path=str(result_path))
    return artifact


def replay_terminal_artifact(root: Path, raw_dir: Path) -> list[str]:  # pragma: no cover
    """Run the second parser directly from Exp7278 raw evidence."""

    checks, bundle = collect_preconditions(root, raw_dir=raw_dir)
    failures = [str(row["check"]) for row in checks if row.get("passed") is not True]
    if failures:
        return failures
    reduced = independent_reduce(bundle)
    try:
        paired_bootstrap(reduced["rows"], BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    except ValueError as exc:
        return [str(exc)]
    return list(reduced["replay_mismatches"])


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the task contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the source audit or independently replay its raw prerequisite."""

    print("[exp7279] startup", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--replay-raw", type=Path)
    args = parser.parse_args(argv)
    root = find_repo_root(start=__file__)
    if args.replay_raw is not None:
        _progress(9, "benchmark_start", name="independent_raw_replay_cli")
        errors = replay_terminal_artifact(root, args.replay_raw)
        _progress(9, "benchmark_end", name="independent_raw_replay_cli", errors=errors)
        return int(bool(errors))
    artifact = run_experiment(root, args.date)
    print(
        f"[exp7279] terminal verdict={artifact['honest_verdict']} "
        f"complete={artifact['source_audit_complete_score']} "
        f"promotion={artifact['source_promotion_score']}",
        flush=True,
    )
    return 0
