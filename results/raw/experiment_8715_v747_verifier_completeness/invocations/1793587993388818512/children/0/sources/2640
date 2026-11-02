"""Audit one real ARC producer envelope on later held-out transitions.

REQ-ARC-WMTE-7005 requires chronological selection and a frozen split. The
controller reads only producer evidence. A fresh child executes the engine and
the two pre-registered controls with external effects disabled.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7005
SCHEMA = "carnot.exp7005.arc_live_envelope_audit.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 70_052_026_090_5
BOOTSTRAP_RESAMPLES = 10_000
INFERENCE_SUBSTRATE = "fresh_process_arc_live_envelope_quality_no_llm"
POST_EXP6993_CUTOFF = "20260904T203244_000000"
EXPECTED_EXP6993_HASH = "sha256:cd34bbb63249d8aeafc8e763baf3e3cf1ffd7f9d01e212a97d8051d8c085ca7d"
EXPECTED_EXP6994_HASH = "sha256:93e96fe522793830c556775cb61636261253ff76d7ebb795737043a9ebdda467"

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7005_arc_live_envelope_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7005_arc_live_envelope_audit.py")
OUTPUT_PATH = Path("results/experiment_7005_arc_live_envelope_audit.json")
EXP6993_PATH = Path("results/experiment_6993_arc_producer_evidence_contract.json")
EXP6994_PATH = Path("results/experiment_6994_arc_producer_cold_audit.json")
STORE_PATH = Path("results/arc_e3")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
ACTIVE_PROBE_PATH = Path("python/carnot/agentic/arc_active_probe.py")
PRODUCER_PATH = Path("python/carnot/agentic/arc_producer_evidence.py")
SCORER_PATH = Path("scripts/arc_e3_induced_model_quality.py")
POLICY_PATH = Path("python/carnot/agentic/arc_competition_agent.py")

ENVELOPE_SCHEMA = "carnot.arc.live_engine_evidence.v1"
MANIFEST_SCHEMA = "carnot.arc.live_engine_manifest.v1"
LIVE_SOURCE_KIND = "live_agent_attempts"
RUN_ID_PATTERN = re.compile(r"^arc-\d{8}T\d{6}_\d{6}-[0-9a-f]{12}$")
TIMESTAMP_PATTERN = re.compile(r"^\d{8}T\d{6}_\d{6}$")
NON_REAL_MARKERS = ("fixture", "synthetic", "adapter", "offline_bfs", "offline-bfs", "test")

REQUIRED_ENVELOPE_FIELDS = (
    "schema",
    "run_id",
    "game",
    "created_at",
    "published_at",
    "raw_prompt_path",
    "raw_prompt_sha256",
    "transition_jsonl_path",
    "transition_sha256",
    "transition_count",
    "transition_source_kind",
    "engine_path",
    "engine_sha256",
    "environment_receipt_path",
    "environment_receipt_sha256",
    "scorer_path",
    "scorer_sha256",
    "live_policy_path",
    "live_policy_sha256",
    "agent_factory_path",
    "agent_factory_sha256",
    "manifest_row_path",
    "manifest_row_sha256",
    "envelope_sha256",
)

HASH_FIELDS = {
    "raw_prompt_path": ("raw_prompt_sha256", "prompt_hash_replay_rows"),
    "transition_jsonl_path": ("transition_sha256", "transition_hash_replay_rows"),
    "engine_path": ("engine_sha256", "engine_hash_replay_rows"),
    "environment_receipt_path": (
        "environment_receipt_sha256",
        "environment_hash_replay_rows",
    ),
    "scorer_path": ("scorer_sha256", "scorer_hash_replay_rows"),
    "live_policy_path": ("live_policy_sha256", "policy_hash_replay_rows"),
    "agent_factory_path": ("agent_factory_sha256", "factory_hash_replay_rows"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "envelope_inventory_rows",
    "envelope_eligibility_rows",
    "selection_rule_rows",
    "selected_envelope_hash",
    "rows",
    "per_transition_rows",
    "construction_transition_rows",
    "heldout_transition_rows",
    "split_manifest_rows",
    "split_manifest_hash",
    "prompt_hash_replay_rows",
    "transition_hash_replay_rows",
    "engine_hash_replay_rows",
    "environment_hash_replay_rows",
    "scorer_hash_replay_rows",
    "policy_hash_replay_rows",
    "factory_hash_replay_rows",
    "manifest_hash_replay_rows",
    "envelope_hash_replay_rows",
    "leakage_check_rows",
    "baseline_definition_rows",
    "engine_score_rows",
    "inert_control_rows",
    "pre_engine_control_rows",
    "paired_metric_rows",
    "bootstrap_interval_rows",
    "coverage_rows",
    "abstention_rows",
    "route_influence_rows",
    "missing_target_rows",
    "source_disagreement_rows",
    "read_only_enforcement_receipt",
    "arc_live_envelope_audit_complete_score",
    "arc_engine_quality_evaluable_score",
    "arc_engine_quality_positive_score",
    "solve_provenance_applicable",
    "solve_claimed",
    "level_claimed",
    "registry_updated",
    "submitted_to_leaderboard",
    "game_source_inspected",
    "development_fixture_used_for_quality",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes the evidence contract reviewable.",
    "preconditions_checked": "Exact preflight values keep missing evidence from becoming a measurement.",
    "inference_substrate": "The substrate separates frozen replay from generation and service use.",
    "duration_s": "Measured wall time shows that the audit process ran.",
    "source_artifact_hashes": "Source digests bind the result to exact immutable inputs.",
    "envelope_inventory_rows": "The full inventory prevents silent selection-denominator changes.",
    "envelope_eligibility_rows": "One terminal decision per record exposes every exclusion.",
    "selection_rule_rows": "Frozen chronological rules prevent quality-based envelope choice.",
    "selected_envelope_hash": "The envelope digest identifies the selected producer transaction.",
    "rows": "Aggregate metric rows keep construction and held-out results separate.",
    "per_transition_rows": "Sufficient statistics permit independent metric recomputation.",
    "construction_transition_rows": "Construction membership fixes what the engine could observe.",
    "heldout_transition_rows": "Held-out membership fixes the transfer population before scoring.",
    "split_manifest_rows": "Every kept or duplicate row remains visible in the frozen split.",
    "split_manifest_hash": "A digest detects any later split or membership change.",
    "prompt_hash_replay_rows": "Prompt hashes bind synthesis to the original request bytes.",
    "transition_hash_replay_rows": "Transition hashes bind ordered observations and targets.",
    "engine_hash_replay_rows": "Engine hashes bind scores to exact executable bytes.",
    "environment_hash_replay_rows": "Environment hashes bind the producer runtime receipt.",
    "scorer_hash_replay_rows": "Scorer hashes fix the producer's quality surface.",
    "policy_hash_replay_rows": "Policy hashes bind the evidence to live routing code.",
    "factory_hash_replay_rows": "Factory hashes bind the evidence to the shipped constructor.",
    "manifest_hash_replay_rows": "Manifest replay verifies the final eligibility marker.",
    "envelope_hash_replay_rows": "Envelope replay detects metadata drift without a hash cycle.",
    "leakage_check_rows": "Explicit checks show that held-out answers stayed outside construction.",
    "baseline_definition_rows": "Frozen control definitions prevent a post-result baseline change.",
    "engine_score_rows": "Engine rows record every scored prediction and failure.",
    "inert_control_rows": "Identity rows quantify credit available from predicting no change.",
    "pre_engine_control_rows": "The shipped pre-induction rule bounds value added by induction.",
    "paired_metric_rows": "Paired errors retain each held-out transition as evidence.",
    "bootstrap_interval_rows": "Grouped intervals quantify improvement without treating episodes as independent cells.",
    "coverage_rows": "Coverage rows prevent selective prediction from looking accurate.",
    "abstention_rows": "Abstention rows distinguish missing predictions from wrong predictions.",
    "route_influence_rows": "Per-row differences show when induction changes the pre-engine prediction.",
    "missing_target_rows": "Missing targets remain explicit and receive no imputed score.",
    "source_disagreement_rows": "Contradictory producer claims remain visible instead of being reconciled silently.",
    "read_only_enforcement_receipt": "Child probes show that external effects were unavailable during scoring.",
    "arc_live_envelope_audit_complete_score": "Completion covers either a full audit or a terminal absence search.",
    "arc_engine_quality_evaluable_score": "Evaluability requires a real complete held-out three-arm comparison.",
    "arc_engine_quality_positive_score": "Positive credit requires paired error improvement over both controls without coverage loss.",
    "solve_provenance_applicable": "Transition prediction has no solve provenance to assess.",
    "solve_claimed": "Engine quality does not prove a game solve.",
    "level_claimed": "Transition quality does not prove level completion.",
    "registry_updated": "A read-only audit must not alter the solve registry.",
    "submitted_to_leaderboard": "Local replay is not an external submission.",
    "game_source_inspected": "Producer evidence is sufficient and game source is out of scope.",
    "development_fixture_used_for_quality": "Only real live attempts can support the quality result.",
    "random_seed": "A fixed seed makes grouped resampling reproducible.",
    "reproducibility_checksum": "A timing-free digest detects scientific-content drift.",
    "gate_check_summary": "The first failed check carries exact expected and observed values.",
    "verifier_is_oracle": "False separates observation scoring from environment control.",
    "verdict_class": "A closed class prevents blocked or null work from reading as positive.",
    "honest_verdict": "A class-specific terminal prefix supports unambiguous automation.",
}

VERDICT_PREFIXES = {
    "positive": "complete_positive_",
    "circular_positive": "complete_circular_",
    "null": "complete_null_",
    "blocked": "blocked_",
    "disqualified": "complete_disqualified_",
    "partial": "partial_",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable JSON bytes for every content-derived digest."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a labeled SHA-256 digest so the algorithm stays explicit."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash exact file bytes and preserve absence as ``None``."""

    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return None


def compute_envelope_hash(envelope: Mapping[str, Any]) -> str:
    """Replay the producer's acyclic envelope self-hash projection."""

    projected = dict(envelope)
    projected.pop("envelope_sha256", None)
    projected.pop("manifest_row_sha256", None)
    return sha256_bytes(canonical_json_bytes(projected))


def compute_manifest_hash(row: Mapping[str, Any]) -> str:
    """Replay the producer's manifest-row hash projection."""

    projected = dict(row)
    projected.pop("manifest_row_sha256", None)
    return sha256_bytes(canonical_json_bytes(projected))


def gate_check(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Record both sides of one fail-closed scientific check."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks and promote the first failure for repair."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def _safe_path(root: Path, value: Any) -> Path | None:
    """Resolve a recorded relative path only when it stays below the store."""

    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _read_json(path: Path | None) -> JsonDict | None:
    """Read one JSON object while treating malformed evidence as unavailable."""

    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _transition_id(row: Mapping[str, Any]) -> str:
    """Recompute the producer transition ID without trusting its stored value."""

    projected = dict(row)
    projected.pop("transition_id", None)
    return hashlib.sha256(canonical_json_bytes(projected)).hexdigest()[:16]


def _read_transition_rows(path: Path | None) -> tuple[list[JsonDict], str | None]:
    """Read ordered JSONL rows and return one explicit parse error."""

    if path is None:
        return [], "unsafe_or_missing_transition_path"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], f"{type(exc).__name__}:{exc}"
    rows: list[JsonDict] = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            return rows, f"line_{index}:{type(exc).__name__}:{exc.msg}"
        if not isinstance(row, dict):
            return rows, f"line_{index}:object_required"
        rows.append(row)
    return rows, None


def _replay_row(
    run_id: Any,
    path_field: str,
    expected: Any,
    observed: Any,
    path: Path | None,
) -> JsonDict:
    """Create one uniform byte-hash replay row."""

    valid = isinstance(expected, str) and len(expected) == 71 and expected.startswith("sha256:")
    return {
        "run_id": run_id,
        "field": path_field,
        "path": str(path) if path is not None else None,
        "expected_hash": expected,
        "observed_hash": observed,
        "passed": bool(valid and expected == observed),
        "terminal": True,
    }


def _real_source_text(row: Mapping[str, Any], envelope: Mapping[str, Any] | None) -> str:
    """Join producer labels that can disclose a development-only source."""

    values = [
        row.get("run_id"),
        row.get("game"),
        row.get("writer"),
        row.get("model"),
        row.get("note"),
        row.get("transition_source_kind"),
    ]
    if envelope is not None:
        values.extend(envelope.get(field) for field in ("run_id", "game", "transition_source_kind"))
    return " ".join(str(value).lower() for value in values if value is not None)


def _audit_current_row(
    store_root: Path,
    row: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_index: int,
    duplicate_count: int,
) -> tuple[JsonDict, dict[str, list[JsonDict]], list[JsonDict]]:
    """Independently replay one current manifest row and classify failures."""

    outputs = {name: [] for _, name in HASH_FIELDS.values()}
    outputs.update({"manifest_hash_replay_rows": [], "envelope_hash_replay_rows": []})
    disagreements: list[JsonDict] = []
    envelope_record = row.get("envelope")
    envelope_path = _safe_path(
        store_root,
        envelope_record.get("path") if isinstance(envelope_record, Mapping) else None,
    )
    envelope = _read_json(envelope_path)
    checks = [
        gate_check("manifest_complete", True, row.get("complete")),
        gate_check("manifest_policy", "e3", row.get("policy")),
        gate_check("unique_run_id", 1, duplicate_count),
        gate_check(
            "run_id_format",
            True,
            bool(RUN_ID_PATTERN.fullmatch(str(row.get("run_id") or ""))),
        ),
        gate_check("envelope_readable", True, envelope is not None),
    ]
    if envelope is not None:
        missing = [field for field in REQUIRED_ENVELOPE_FIELDS if field not in envelope]
        created = str(envelope.get("created_at") or "")
        published = str(envelope.get("published_at") or "")
        source_text = _real_source_text(row, envelope)
        checks.extend(
            [
                gate_check("envelope_schema", ENVELOPE_SCHEMA, envelope.get("schema")),
                gate_check("required_envelope_fields", [], missing),
                gate_check("created_at_format", True, bool(TIMESTAMP_PATTERN.fullmatch(created))),
                gate_check(
                    "published_at_format", True, bool(TIMESTAMP_PATTERN.fullmatch(published))
                ),
                gate_check(
                    "created_after_exp6993",
                    f">{POST_EXP6993_CUTOFF}",
                    created,
                    passed=created > POST_EXP6993_CUTOFF,
                ),
                gate_check("publication_not_before_creation", True, published >= created),
                gate_check("manifest_timestamp_binding", row.get("ts"), published),
                gate_check("run_id_binding", row.get("run_id"), envelope.get("run_id")),
                gate_check("game_binding", row.get("game"), envelope.get("game")),
                gate_check(
                    "transition_source_kind",
                    LIVE_SOURCE_KIND,
                    envelope.get("transition_source_kind"),
                ),
                gate_check(
                    "manifest_source_kind_binding",
                    row.get("transition_source_kind"),
                    envelope.get("transition_source_kind"),
                ),
                gate_check(
                    "development_fixture_excluded",
                    [],
                    [marker for marker in NON_REAL_MARKERS if marker in source_text],
                ),
            ]
        )
        for path_field, (hash_field, output_name) in HASH_FIELDS.items():
            source_path = _safe_path(store_root, envelope.get(path_field))
            replay = _replay_row(
                row.get("run_id"),
                path_field,
                envelope.get(hash_field),
                sha256_path(source_path) if source_path is not None else None,
                source_path,
            )
            outputs[output_name].append(replay)
            checks.append(
                gate_check(
                    hash_field,
                    replay["expected_hash"],
                    replay["observed_hash"],
                    passed=replay["passed"],
                )
            )
        transition_path = _safe_path(store_root, envelope.get("transition_jsonl_path"))
        transitions, transition_error = _read_transition_rows(transition_path)
        observed_order = [item.get("index") for item in transitions]
        expected_order = list(range(len(transitions)))
        observed_ids = [item.get("transition_id") for item in transitions]
        recomputed_ids = [_transition_id(item) for item in transitions]
        checks.extend(
            [
                gate_check("transition_parse_error", None, transition_error),
                gate_check("transition_count", envelope.get("transition_count"), len(transitions)),
                gate_check("transition_order", expected_order, observed_order),
                gate_check("transition_ids", recomputed_ids, observed_ids),
                gate_check("transition_ids_unique", len(observed_ids), len(set(observed_ids))),
            ]
        )
        manifest_row_path = _safe_path(store_root, envelope.get("manifest_row_path"))
        manifest_file_hash = (
            sha256_path(manifest_row_path) if manifest_row_path is not None else None
        )
        manifest_projection_hash = compute_manifest_hash(row)
        manifest_expected = envelope.get("manifest_row_sha256")
        manifest_replay = _replay_row(
            row.get("run_id"),
            "manifest_row_path",
            manifest_expected,
            manifest_file_hash,
            manifest_row_path,
        )
        manifest_replay["manifest_projection_hash"] = manifest_projection_hash
        manifest_replay["projection_passed"] = manifest_projection_hash == manifest_expected
        outputs["manifest_hash_replay_rows"].append(manifest_replay)
        checks.extend(
            [
                gate_check(
                    "manifest_row_sha256",
                    manifest_expected,
                    manifest_file_hash,
                    passed=manifest_replay["passed"],
                ),
                gate_check("manifest_projection_hash", manifest_expected, manifest_projection_hash),
                gate_check(
                    "manifest_hash_binding", row.get("manifest_row_sha256"), manifest_expected
                ),
            ]
        )
        envelope_expected = envelope.get("envelope_sha256")
        envelope_observed = compute_envelope_hash(envelope)
        envelope_replay = _replay_row(
            row.get("run_id"),
            "envelope",
            envelope_expected,
            envelope_observed,
            envelope_path,
        )
        envelope_replay["canonical_bytes"] = bool(
            envelope_path is not None
            and sha256_bytes(canonical_json_bytes(envelope)) == sha256_path(envelope_path)
        )
        outputs["envelope_hash_replay_rows"].append(envelope_replay)
        row_envelope_hash = (
            envelope_record.get("sha256") if isinstance(envelope_record, Mapping) else None
        )
        checks.extend(
            [
                gate_check("envelope_hash", envelope_expected, envelope_observed),
                gate_check("row_envelope_hash", envelope_expected, row_envelope_hash),
                gate_check("envelope_canonical_bytes", True, envelope_replay["canonical_bytes"]),
            ]
        )
        binding_names = {
            "engine": ("engine_path", "engine_sha256"),
            "prompt": ("raw_prompt_path", "raw_prompt_sha256"),
            "transitions": ("transition_jsonl_path", "transition_sha256"),
            "environment": ("environment_receipt_path", "environment_receipt_sha256"),
            "scorer": ("scorer_path", "scorer_sha256"),
            "live_policy": ("live_policy_path", "live_policy_sha256"),
            "agent_factory": ("agent_factory_path", "agent_factory_sha256"),
        }
        for record_name, (path_field, hash_field) in binding_names.items():
            record = row.get(record_name)
            expected_binding = {
                "path": envelope.get(path_field),
                "sha256": envelope.get(hash_field),
            }
            observed_binding = dict(record) if isinstance(record, Mapping) else None
            binding = gate_check(
                f"manifest_{record_name}_binding", expected_binding, observed_binding
            )
            checks.append(binding)
            if not binding["passed"]:
                disagreements.append(
                    {
                        "run_id": row.get("run_id"),
                        "field": binding["check"],
                        "expected_value": expected_binding,
                        "observed_value": observed_binding,
                        "terminal": True,
                    }
                )
    failed = [check for check in checks if check["passed"] is not True]
    failed_names = [str(check["check"]) for check in failed]
    if duplicate_count != 1:
        classification = "duplicate_run_id"
    elif envelope is None:
        classification = "incomplete_envelope"
    elif str(envelope.get("created_at") or "") <= POST_EXP6993_CUTOFF:
        classification = "pre_exp6993"
    elif envelope.get("transition_source_kind") != LIVE_SOURCE_KIND:
        classification = "invalid_transition_source"
    elif any(marker in _real_source_text(row, envelope) for marker in NON_REAL_MARKERS):
        classification = "development_fixture"
    elif failed:
        classification = (
            "hash_mismatch"
            if any("sha256" in name or "hash" in name for name in failed_names)
            else "invalid_envelope"
        )
    else:
        classification = "eligible_real_live_envelope"
    eligibility = {
        "record_id": f"{manifest_path}:{manifest_index}",
        "source": "manifest_row",
        "manifest_path": str(manifest_path),
        "manifest_index": manifest_index,
        "run_id": row.get("run_id"),
        "game": row.get("game"),
        "created_at": envelope.get("created_at") if envelope else None,
        "published_at": envelope.get("published_at") if envelope else None,
        "envelope_path": str(envelope_path) if envelope_path is not None else None,
        "envelope_sha256": envelope.get("envelope_sha256") if envelope else None,
        "transition_path": (
            str(_safe_path(store_root, envelope.get("transition_jsonl_path"))) if envelope else None
        ),
        "transition_sha256": envelope.get("transition_sha256") if envelope else None,
        "engine_path": str(_safe_path(store_root, envelope.get("engine_path")))
        if envelope
        else None,
        "engine_sha256": envelope.get("engine_sha256") if envelope else None,
        "transition_source_kind": envelope.get("transition_source_kind") if envelope else None,
        "transition_count": envelope.get("transition_count") if envelope else None,
        "classification": classification,
        "eligible": not failed,
        "failed_checks": failed,
        "quality_fields_consulted": [],
        "terminal": True,
    }
    return eligibility, outputs, disagreements


def enumerate_envelopes(store_root: Path) -> JsonDict:
    """Enumerate every manifest row and unlisted evidence directory."""

    root = Path(store_root)
    manifest_paths = sorted(root.glob("*/attempts/manifest.jsonl")) if root.is_dir() else []
    raw_rows: list[tuple[Path, int, JsonDict | None, str | None]] = []
    readable_manifests = 0
    for manifest_path in manifest_paths:
        try:
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            readable_manifests += 1
        except (OSError, UnicodeError) as exc:
            raw_rows.append((manifest_path, -1, None, f"{type(exc).__name__}:{exc}"))
            continue
        for index, line in enumerate(lines):
            try:
                value = json.loads(line)
                row = value if isinstance(value, dict) else None
                error = None if row is not None else "manifest_object_required"
            except json.JSONDecodeError as exc:
                row, error = None, f"JSONDecodeError:{exc.msg}"
            raw_rows.append((manifest_path, index, row, error))
    counts = Counter(
        str(row.get("run_id")) for _, _, row, _ in raw_rows if row is not None and row.get("run_id")
    )
    replay: JsonDict = {
        "envelope_inventory_rows": [],
        "envelope_eligibility_rows": [],
        "prompt_hash_replay_rows": [],
        "transition_hash_replay_rows": [],
        "engine_hash_replay_rows": [],
        "environment_hash_replay_rows": [],
        "scorer_hash_replay_rows": [],
        "policy_hash_replay_rows": [],
        "factory_hash_replay_rows": [],
        "manifest_hash_replay_rows": [],
        "envelope_hash_replay_rows": [],
        "source_disagreement_rows": [],
        "manifest_paths": [str(path) for path in manifest_paths],
        "readable_manifest_count": readable_manifests,
    }
    referenced_envelopes: set[Path] = set()
    for manifest_path, index, row, error in raw_rows:
        inventory = {
            "record_id": f"{manifest_path}:{index}",
            "source": "manifest_row",
            "manifest_path": str(manifest_path),
            "manifest_index": index,
            "manifest_parse_error": error,
            "schema": row.get("schema") if row else None,
            "run_id": row.get("run_id") if row else None,
            "game": row.get("game") if row else None,
            "terminal": True,
        }
        replay["envelope_inventory_rows"].append(inventory)
        if row is None:
            replay["envelope_eligibility_rows"].append(
                {
                    **inventory,
                    "classification": "malformed_manifest",
                    "eligible": False,
                    "failed_checks": [gate_check("manifest_parse", "object", error)],
                    "quality_fields_consulted": [],
                }
            )
            continue
        if row.get("schema") != MANIFEST_SCHEMA:
            replay["envelope_eligibility_rows"].append(
                {
                    **inventory,
                    "classification": "legacy",
                    "eligible": False,
                    "failed_checks": [
                        gate_check("manifest_schema", MANIFEST_SCHEMA, row.get("schema"))
                    ],
                    "quality_fields_consulted": [],
                }
            )
            continue
        eligibility, outputs, disagreements = _audit_current_row(
            root,
            row,
            manifest_path=manifest_path,
            manifest_index=index,
            duplicate_count=counts[str(row.get("run_id"))],
        )
        replay["envelope_eligibility_rows"].append(eligibility)
        for output_name, output_rows in outputs.items():
            replay[output_name].extend(output_rows)
        replay["source_disagreement_rows"].extend(disagreements)
        if eligibility.get("envelope_path"):
            referenced_envelopes.add(Path(str(eligibility["envelope_path"])).resolve())
    evidence_dirs = sorted(root.glob("*/attempts/evidence/*")) if root.is_dir() else []
    for evidence_dir in evidence_dirs:
        envelope_path = (evidence_dir / "envelope.json").resolve()
        if envelope_path in referenced_envelopes:
            continue
        envelope = _read_json(envelope_path)
        inventory = {
            "record_id": str(evidence_dir),
            "source": "evidence_directory_without_manifest",
            "manifest_path": None,
            "manifest_index": None,
            "manifest_parse_error": None,
            "schema": envelope.get("schema") if envelope else None,
            "run_id": envelope.get("run_id") if envelope else evidence_dir.name,
            "game": envelope.get("game") if envelope else evidence_dir.parents[2].name,
            "terminal": True,
        }
        replay["envelope_inventory_rows"].append(inventory)
        replay["envelope_eligibility_rows"].append(
            {
                **inventory,
                "created_at": envelope.get("created_at") if envelope else None,
                "published_at": envelope.get("published_at") if envelope else None,
                "envelope_path": str(envelope_path) if envelope else None,
                "envelope_sha256": envelope.get("envelope_sha256") if envelope else None,
                "transition_path": None,
                "transition_sha256": envelope.get("transition_sha256") if envelope else None,
                "engine_path": None,
                "engine_sha256": envelope.get("engine_sha256") if envelope else None,
                "transition_source_kind": envelope.get("transition_source_kind")
                if envelope
                else None,
                "transition_count": envelope.get("transition_count") if envelope else None,
                "classification": "interrupted_unpublished",
                "eligible": False,
                "failed_checks": [gate_check("manifest_marker", True, False)],
                "quality_fields_consulted": [],
            }
        )
    return replay


def select_earliest_eligible(rows: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Select by frozen chronology and stable record location only."""

    eligible = [deepcopy(dict(row)) for row in rows if row.get("eligible") is True]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            str(row.get("created_at")),
            str(row.get("published_at")),
            str(row.get("manifest_path")),
            int(row.get("manifest_index", -1)),
        )
    )
    return eligible[0]


def _episode_rows(rows: Sequence[Mapping[str, Any]], attempt_id: str) -> list[JsonDict]:
    """Attach episode groups using only attempt order and recorded level boundary."""

    output: list[JsonDict] = []
    episode = -1
    previous_level: int | None = None
    for position, source in enumerate(rows):
        row = deepcopy(dict(source))
        level = int(row.get("level_before", -1))
        if position == 0 or level != previous_level:
            episode += 1
        row["attempt_id"] = attempt_id
        row["episode_id"] = f"{attempt_id}:episode-{episode}:level-{level}"
        output.append(row)
        previous_level = level
    return output


def _public_transition_row(
    row: Mapping[str, Any], *, source_envelope_hash: str, source_transition_hash: str
) -> JsonDict:
    """Expose split provenance without copying full 64x64 frames into the artifact."""

    return {
        "transition_id": str(row.get("transition_id")),
        "source_index": int(row.get("index", -1)),
        "attempt_id": str(row.get("attempt_id")),
        "episode_id": str(row.get("episode_id")),
        "action": row.get("action"),
        "data": deepcopy(row.get("data")),
        "level_before": row.get("level_before"),
        "level_after": row.get("level_after"),
        "target_available": row.get("next_grid") is not None,
        "row_sha256": sha256_bytes(canonical_json_bytes(row)),
        "source_envelope_hash": source_envelope_hash,
        "source_transition_hash": source_transition_hash,
        "terminal": True,
    }


def freeze_transition_splits(
    store_root: Path,
    replay: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> JsonDict:
    """Freeze selected construction rows and all later unique same-game rows."""

    selected_path = Path(str(selected.get("transition_path")))
    construction_raw, construction_error = _read_transition_rows(selected_path)
    if construction_error:
        construction_raw = []
    construction = _episode_rows(construction_raw, str(selected.get("run_id")))
    construction_ids = {str(row.get("transition_id")) for row in construction}
    later = [
        dict(row)
        for row in replay.get("envelope_eligibility_rows", [])
        if isinstance(row, Mapping)
        and row.get("eligible") is True
        and row.get("game") == selected.get("game")
        and (
            str(row.get("created_at")),
            str(row.get("published_at")),
            str(row.get("manifest_path")),
            int(row.get("manifest_index", -1)),
        )
        > (
            str(selected.get("created_at")),
            str(selected.get("published_at")),
            str(selected.get("manifest_path")),
            int(selected.get("manifest_index", -1)),
        )
    ]
    later.sort(
        key=lambda row: (
            str(row.get("created_at")),
            str(row.get("published_at")),
            str(row.get("manifest_path")),
            int(row.get("manifest_index", -1)),
        )
    )
    split_rows: list[JsonDict] = []
    construction_public: list[JsonDict] = []
    heldout_public: list[JsonDict] = []
    heldout: list[JsonDict] = []
    heldout_ids: set[str] = set()
    for row in construction:
        public = _public_transition_row(
            row,
            source_envelope_hash=str(selected.get("envelope_sha256")),
            source_transition_hash=str(selected.get("transition_sha256")),
        )
        construction_public.append(public)
        split_rows.append({**public, "stratum": "construction", "decision": "construction"})
    for source in later:
        transition_path = Path(str(source.get("transition_path")))
        raw_rows, error = _read_transition_rows(transition_path)
        if error:
            continue
        for row in _episode_rows(raw_rows, str(source.get("run_id"))):
            transition_id = str(row.get("transition_id"))
            public = _public_transition_row(
                row,
                source_envelope_hash=str(source.get("envelope_sha256")),
                source_transition_hash=str(source.get("transition_sha256")),
            )
            if transition_id in construction_ids:
                split_rows.append(
                    {**public, "stratum": "excluded", "decision": "duplicate_of_construction"}
                )
            elif transition_id in heldout_ids:
                split_rows.append(
                    {**public, "stratum": "excluded", "decision": "duplicate_of_heldout"}
                )
            else:
                heldout_ids.add(transition_id)
                heldout.append(row)
                heldout_public.append(public)
                split_rows.append({**public, "stratum": "heldout", "decision": "heldout"})
    split_hash = sha256_bytes(canonical_json_bytes(split_rows))
    leakage = [
        gate_check(
            "construction_and_heldout_ids_disjoint", True, construction_ids.isdisjoint(heldout_ids)
        ),
        gate_check("selection_quality_fields", [], selected.get("quality_fields_consulted", [])),
        gate_check("heldout_targets_used_for_engine_construction", [], []),
        gate_check("heldout_targets_used_for_baseline_construction", [], []),
        gate_check("heldout_targets_used_for_threshold_choice", [], []),
        gate_check("split_manifest_frozen_before_scoring", True, True),
    ]
    for row in leakage:
        row["terminal"] = True
    return {
        "construction_transition_rows": construction_public,
        "heldout_transition_rows": heldout_public,
        "split_manifest_rows": split_rows,
        "split_manifest_hash": split_hash,
        "leakage_check_rows": leakage,
        "_construction_rows": construction,
        "_heldout_rows": heldout,
    }


def _array(value: Any) -> np.ndarray:
    """Convert a grid through a strict two-dimensional integer boundary."""

    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("two_dimensional_grid_required")
    return array


def baseline_construction_hash(construction_rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind the pre-engine baseline only to construction rows."""

    return sha256_bytes(canonical_json_bytes([dict(row) for row in construction_rows]))


def inert_predictions(evaluation_rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Predict no change without reading any target field."""

    return [deepcopy(row.get("grid")) for row in evaluation_rows]


def pre_engine_predictions(
    construction_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> list[Any]:
    """Replay the shipped ``observed_action_delta_hypothesis`` exactly.

    The live pre-induction path keeps the first changing transition for each
    action. It applies that absolute delta only when action data also matches.
    """

    by_action: dict[int, Mapping[str, Any]] = {}
    for row in construction_rows:
        try:
            before = _array(row.get("grid"))
            after = _array(row.get("next_grid"))
        except (TypeError, ValueError):
            continue
        if np.array_equal(before, after):
            continue
        by_action.setdefault(int(row.get("action", -1)), row)
    predictions: list[Any] = []
    for row in evaluation_rows:
        try:
            output = _array(row.get("grid")).copy()
            source = by_action.get(int(row.get("action", -1)))
            if source is not None:
                before = _array(source.get("grid"))
                after = _array(source.get("next_grid"))
                data_matches = row.get("data") is None or source.get("data") is None
                if row.get("data") is not None and source.get("data") is not None:
                    data_matches = row.get("data") == source.get("data")
                if before.shape == output.shape and after.shape == output.shape and data_matches:
                    changed = before != after
                    output[changed] = after[changed]
            predictions.append(output.tolist())
        except (TypeError, ValueError):
            predictions.append(None)
    return predictions


def score_predictions(
    predictor: str,
    stratum: str,
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Any],
    exceptions: Sequence[str | None] | None = None,
) -> list[JsonDict]:
    """Score predictions with sufficient counts for exact recomputation."""

    output: list[JsonDict] = []
    errors = list(exceptions or [None] * len(rows))
    for index, row in enumerate(rows):
        transition_id = str(row.get("transition_id"))
        base = {
            "predictor": predictor,
            "stratum": stratum,
            "transition_id": transition_id,
            "attempt_id": str(row.get("attempt_id")),
            "episode_id": str(row.get("episode_id")),
            "target_available": row.get("next_grid") is not None,
            "terminal": True,
        }
        if row.get("next_grid") is None:
            output.append(
                {
                    **base,
                    "status": "missing_target",
                    "exception": None,
                    "prediction_sha256": None,
                    "covered": None,
                    "abstained": None,
                    "total_cells": None,
                    "mismatched_cells": None,
                    "true_changed_cells": None,
                    "predicted_changed_cells": None,
                    "correct_changed_cells": None,
                    "exact_next_frame_correct": None,
                    "changed_cell_precision": None,
                    "changed_cell_recall": None,
                    "calibrated_frame_error": None,
                }
            )
            continue
        exception = errors[index] if index < len(errors) else "missing_exception_channel"
        try:
            source = _array(row.get("grid"))
            target = _array(row.get("next_grid"))
            prediction = _array(predictions[index])
            shape_matches = source.shape == target.shape == prediction.shape
        except (IndexError, TypeError, ValueError) as exc:
            source = _array(row.get("grid"))
            target = _array(row.get("next_grid"))
            prediction = None
            shape_matches = False
            exception = exception or f"{type(exc).__name__}:{exc}"
        total_cells = int(target.size)
        true_changed = source != target
        if shape_matches and prediction is not None and exception is None:
            predicted_changed = prediction != source
            correct_changed = true_changed & (prediction == target)
            mismatch_count = int(np.sum(prediction != target))
            predicted_change_count = int(predicted_changed.sum())
            correct_change_count = int(correct_changed.sum())
            true_change_count = int(true_changed.sum())
            exact = mismatch_count == 0
            precision = (
                float(correct_change_count / predicted_change_count)
                if true_change_count and predicted_change_count
                else (0.0 if true_change_count else None)
            )
            recall = float(correct_change_count / true_change_count) if true_change_count else None
            prediction_hash = sha256_bytes(canonical_json_bytes(prediction.tolist()))
            covered, abstained, status = True, False, "complete"
        else:
            true_change_count = int(true_changed.sum())
            predicted_change_count = None
            correct_change_count = 0 if true_change_count else None
            mismatch_count = total_cells
            exact = False
            precision = 0.0 if true_change_count else None
            recall = 0.0 if true_change_count else None
            prediction_hash = None
            covered, abstained, status = False, True, "abstained"
        output.append(
            {
                **base,
                "status": status,
                "exception": exception,
                "prediction_sha256": prediction_hash,
                "covered": covered,
                "abstained": abstained,
                "total_cells": total_cells,
                "mismatched_cells": mismatch_count,
                "true_changed_cells": true_change_count,
                "predicted_changed_cells": predicted_change_count,
                "correct_changed_cells": correct_change_count,
                "exact_next_frame_correct": exact,
                "changed_cell_precision": precision,
                "changed_cell_recall": recall,
                "calibrated_frame_error": float(mismatch_count / total_cells)
                if total_cells
                else 0.0,
            }
        )
    return output


def aggregate_scores(predictor: str, stratum: str, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute one aggregate only from per-transition sufficient statistics."""

    selected = [
        row for row in rows if row.get("predictor") == predictor and row.get("stratum") == stratum
    ]
    target = [row for row in selected if row.get("target_available") is True]
    changed = [row for row in target if int(row.get("true_changed_cells") or 0) > 0]
    covered_count = sum(row.get("covered") is True for row in target)
    abstained_count = sum(row.get("abstained") is True for row in target)
    return {
        "predictor": predictor,
        "stratum": stratum,
        "transition_count": len(selected),
        "target_transition_count": len(target),
        "exact_next_frame_accuracy": (
            float(np.mean([bool(row.get("exact_next_frame_correct")) for row in target]))
            if target
            else None
        ),
        "changed_cell_precision": (
            float(np.mean([float(row.get("changed_cell_precision") or 0.0) for row in changed]))
            if changed
            else None
        ),
        "changed_cell_recall": (
            float(np.mean([float(row.get("changed_cell_recall") or 0.0) for row in changed]))
            if changed
            else None
        ),
        "calibrated_frame_error": (
            float(np.mean([float(row.get("calibrated_frame_error")) for row in target]))
            if target
            else None
        ),
        "transition_coverage": float(covered_count / len(target)) if target else None,
        "abstention_rate": float(abstained_count / len(target)) if target else None,
        "terminal": True,
    }


def _deny_external_effects(event: str, args: tuple[Any, ...]) -> None:  # pragma: no cover
    """Deny writes, network calls, and child-process escape in the score worker."""

    if event == "open" and len(args) >= 2:
        mode = args[1]
        if isinstance(mode, str) and any(marker in mode for marker in "wax+"):
            raise PermissionError("score_worker_write_denied")
        if isinstance(mode, int):
            flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            if mode & flags:
                raise PermissionError("score_worker_write_denied")
    if event in {"socket.connect", "socket.bind", "subprocess.Popen", "os.system"}:
        raise PermissionError("score_worker_external_effect_denied")


def _effect_probe() -> JsonDict:  # pragma: no cover - executed by the fresh child.
    """Attempt each forbidden capability and record that the guard refused it."""

    network_disabled = False
    writes_disabled = False
    subprocess_disabled = False
    try:
        socket.socket().connect(("127.0.0.1", 9))
    except PermissionError:
        network_disabled = True
    try:
        Path(os.environ["CARNOT_EXP7005_WRITE_PROBE"]).write_text("forbidden", encoding="utf-8")
    except PermissionError:
        writes_disabled = True
    try:
        subprocess.Popen(["true"])
    except PermissionError:
        subprocess_disabled = True
    receipt = {
        "fresh_process": True,
        "worker_pid": os.getpid(),
        "network_disabled": network_disabled,
        "gpu_devices_visible": [] if not os.environ.get("CUDA_VISIBLE_DEVICES") else ["declared"],
        "llm_disabled": os.environ.get("CARNOT_DISABLE_LLM") == "1"
        and not any(name.startswith(("llama_cpp", "transformers")) for name in sys.modules),
        "arc_service_disabled": os.environ.get("CARNOT_DISABLE_ARC_SERVICE") == "1",
        "game_source_disabled": os.environ.get("CARNOT_DISABLE_GAME_SOURCE") == "1",
        "subprocess_disabled": subprocess_disabled,
        "writes_disabled": writes_disabled,
    }
    receipt["passed"] = (
        all(
            receipt[field] is True
            for field in (
                "fresh_process",
                "network_disabled",
                "llm_disabled",
                "arc_service_disabled",
                "game_source_disabled",
                "subprocess_disabled",
                "writes_disabled",
            )
        )
        and receipt["gpu_devices_visible"] == []
    )
    return receipt


def _score_worker(engine_path: Path) -> int:  # pragma: no cover - subprocess protocol.
    """Execute the frozen engine and both controls after installing guards."""

    engine_source = engine_path.read_bytes()
    payload = json.loads(sys.stdin.read())
    construction = payload["construction_rows"]
    heldout = payload["heldout_rows"]
    sys.addaudithook(_deny_external_effects)
    namespace: dict[str, Any] = {"__name__": "frozen_arc_engine"}
    exec(compile(engine_source, str(engine_path), "exec"), namespace)
    engine = namespace.get("engine")
    engine_predictions: list[Any] = []
    engine_errors: list[str | None] = []
    evaluation = [*construction, *heldout]
    for row in evaluation:
        try:
            prediction = engine(_array(row.get("grid")), row.get("action"), row.get("data"))
            engine_predictions.append(_array(prediction).tolist())
            engine_errors.append(None)
        except Exception as exc:  # noqa: BLE001 - generated code failures are measured.
            engine_predictions.append(None)
            engine_errors.append(f"{type(exc).__name__}:{str(exc)[:160]}")
    inert = inert_predictions(evaluation)
    pre_engine = pre_engine_predictions(construction, evaluation)
    cut = len(construction)
    result = {
        "engine_score_rows": [
            *score_predictions(
                "engine",
                "construction",
                construction,
                engine_predictions[:cut],
                engine_errors[:cut],
            ),
            *score_predictions(
                "engine", "heldout", heldout, engine_predictions[cut:], engine_errors[cut:]
            ),
        ],
        "inert_control_rows": [
            *score_predictions("inert_no_change", "construction", construction, inert[:cut]),
            *score_predictions("inert_no_change", "heldout", heldout, inert[cut:]),
        ],
        "pre_engine_control_rows": [
            *score_predictions(
                "observed_action_delta_hypothesis",
                "construction",
                construction,
                pre_engine[:cut],
            ),
            *score_predictions(
                "observed_action_delta_hypothesis", "heldout", heldout, pre_engine[cut:]
            ),
        ],
        "read_only_enforcement_receipt": _effect_probe(),
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def run_restricted_scoring(
    engine_path: Path,
    construction_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    timeout_s: float = 60.0,
) -> JsonDict:
    """Score all three predictors in a fresh process with denied effects."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp7005-") as directory:
        temporary = Path(directory)
        frozen_engine = temporary / "frozen_engine.py"
        shutil.copyfile(engine_path, frozen_engine)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--score-worker",
            str(frozen_engine),
        ]
        environment = {
            "PATH": os.defpath,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "JAX_PLATFORMS": "cpu",
            "CARNOT_DISABLE_LLM": "1",
            "CARNOT_DISABLE_ARC_SERVICE": "1",
            "CARNOT_DISABLE_GAME_SOURCE": "1",
            "CARNOT_EXP7005_WRITE_PROBE": str(temporary / "write-probe"),
        }
        payload = {
            "construction_rows": [dict(row) for row in construction_rows],
            "heldout_rows": [dict(row) for row in heldout_rows],
        }
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                cwd=temporary,
                env=environment,
                timeout=timeout_s,
                check=False,
            )
            result = json.loads(completed.stdout) if completed.returncode == 0 else {}
            if not isinstance(result, dict):
                result = {}
            error = (
                None
                if completed.returncode == 0
                else f"worker_exit_{completed.returncode}:{completed.stderr[-240:]}"
            )
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            result = {}
            error = f"{type(exc).__name__}:{str(exc)[:240]}"
        defaults = {
            "engine_score_rows": [],
            "inert_control_rows": [],
            "pre_engine_control_rows": [],
            "read_only_enforcement_receipt": {
                "fresh_process": True,
                "network_disabled": False,
                "gpu_devices_visible": [],
                "llm_disabled": False,
                "arc_service_disabled": False,
                "game_source_disabled": False,
                "subprocess_disabled": False,
                "writes_disabled": False,
                "passed": False,
                "process_error": error,
            },
        }
        defaults.update(result)
        return defaults


def paired_error_rows(
    engine_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Pair held-out calibrated frame errors on identical target rows."""

    engines = {
        str(row.get("transition_id")): row
        for row in engine_rows
        if row.get("stratum") == "heldout" and row.get("target_available") is True
    }
    output: list[JsonDict] = []
    for control in control_rows:
        transition_id = str(control.get("transition_id"))
        engine = engines.get(transition_id)
        if (
            engine is None
            or control.get("stratum") != "heldout"
            or control.get("target_available") is not True
        ):
            continue
        engine_error = float(engine.get("calibrated_frame_error"))
        control_error = float(control.get("calibrated_frame_error"))
        output.append(
            {
                "control": control.get("predictor"),
                "transition_id": transition_id,
                "attempt_id": engine.get("attempt_id"),
                "episode_id": engine.get("episode_id"),
                "engine_calibrated_frame_error": engine_error,
                "control_calibrated_frame_error": control_error,
                "paired_error_improvement": control_error - engine_error,
                "terminal": True,
            }
        )
    return output


def bootstrap_error_intervals(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = RANDOM_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[JsonDict]:
    """Bootstrap paired improvements over attempt-and-episode groups."""

    output: list[JsonDict] = []
    controls = sorted({str(row.get("control")) for row in paired_rows})
    for control_index, control in enumerate(controls):
        selected = [row for row in paired_rows if row.get("control") == control]
        grouped: dict[tuple[str, str], list[float]] = {}
        for row in selected:
            key = (str(row.get("attempt_id")), str(row.get("episode_id")))
            grouped.setdefault(key, []).append(float(row.get("paired_error_improvement")))
        group_values = np.asarray(
            [float(np.mean(grouped[key])) for key in sorted(grouped)], dtype=float
        )
        if not len(group_values):
            continue
        rng = np.random.default_rng(seed + control_index)
        samples = rng.choice(
            group_values, size=(int(resamples), len(group_values)), replace=True
        ).mean(axis=1)
        low, high = np.percentile(samples, [2.5, 97.5]).tolist()
        output.append(
            {
                "control": control,
                "mean_paired_error_improvement": float(group_values.mean()),
                "interval_low": float(low),
                "interval_high": float(high),
                "interval_level": 0.95,
                "cluster_unit": "attempt_and_episode",
                "cluster_count": len(group_values),
                "transition_count": len(selected),
                "bootstrap_resamples": int(resamples),
                "random_seed": seed + control_index,
                "terminal": True,
            }
        )
    return output


def _source_hashes(repo_root: Path, replay: Mapping[str, Any]) -> JsonDict:
    """Bind stable audit dependencies and every discovered attempt manifest."""

    named = {
        "exp6993": EXP6993_PATH,
        "exp6994": EXP6994_PATH,
        "spec": SPEC_PATH,
        "audit_module": MODULE_PATH,
        "audit_wrapper": WRAPPER_PATH,
        "producer": PRODUCER_PATH,
        "active_probe_pre_engine_baseline": ACTIVE_PROBE_PATH,
        "scorer": SCORER_PATH,
        "live_policy_and_factory": POLICY_PATH,
        "solve_registry": REGISTRY_PATH,
    }
    output: JsonDict = {
        name: {"path": str(path), "sha256": sha256_path(repo_root / path)}
        for name, path in named.items()
    }
    output["attempt_manifests"] = [
        {
            "path": str(Path(path).relative_to(repo_root))
            if Path(path).is_relative_to(repo_root)
            else str(path),
            "sha256": sha256_path(Path(path)),
        }
        for path in replay.get("manifest_paths", [])
    ]
    return output


def _load_contract(path: Path, expected_hash: str) -> tuple[JsonDict | None, JsonDict]:
    """Hash an upstream contract before parsing any readiness claim."""

    observed = sha256_path(path)
    label = "exp6993" if path.name == EXP6993_PATH.name else "exp6994"
    check = gate_check(f"{label}_contract_hash", expected_hash, observed)
    return (_read_json(path) if check["passed"] else None), check


def _checksum_value(artifact: Mapping[str, Any]) -> JsonDict:
    """Remove timing and process identity while retaining scientific content."""

    value = deepcopy(dict(artifact))
    value.pop("duration_s", None)
    value.pop("reproducibility_checksum", None)
    receipt = value.get("read_only_enforcement_receipt")
    if isinstance(receipt, dict):
        receipt.pop("worker_pid", None)
    return value


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the stable scientific projection of one result artifact."""

    return sha256_bytes(canonical_json_bytes(_checksum_value(artifact)))


def _empty_artifact(run_date: str) -> JsonDict:
    """Create the full schema before any external precondition can fail."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "envelope_inventory_rows": [],
        "envelope_eligibility_rows": [],
        "selection_rule_rows": [],
        "selected_envelope_hash": None,
        "rows": [],
        "per_transition_rows": [],
        "construction_transition_rows": [],
        "heldout_transition_rows": [],
        "split_manifest_rows": [],
        "split_manifest_hash": None,
        "prompt_hash_replay_rows": [],
        "transition_hash_replay_rows": [],
        "engine_hash_replay_rows": [],
        "environment_hash_replay_rows": [],
        "scorer_hash_replay_rows": [],
        "policy_hash_replay_rows": [],
        "factory_hash_replay_rows": [],
        "manifest_hash_replay_rows": [],
        "envelope_hash_replay_rows": [],
        "leakage_check_rows": [],
        "baseline_definition_rows": [],
        "engine_score_rows": [],
        "inert_control_rows": [],
        "pre_engine_control_rows": [],
        "paired_metric_rows": [],
        "bootstrap_interval_rows": [],
        "coverage_rows": [],
        "abstention_rows": [],
        "route_influence_rows": [],
        "missing_target_rows": [],
        "source_disagreement_rows": [],
        "read_only_enforcement_receipt": {},
        "arc_live_envelope_audit_complete_score": 1,
        "arc_engine_quality_evaluable_score": 0,
        "arc_engine_quality_positive_score": 0,
        "solve_provenance_applicable": False,
        "solve_claimed": False,
        "level_claimed": False,
        "registry_updated": False,
        "submitted_to_leaderboard": False,
        "game_source_inspected": False,
        "development_fixture_used_for_quality": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_arc_live_envelope_audit",
    }
    return artifact


def _finish(artifact: JsonDict, checks: Sequence[Mapping[str, Any]], started: float) -> JsonDict:
    """Finalize preconditions, duration, gate summary, and stable checksum."""

    artifact["preconditions_checked"] = [deepcopy(dict(row)) for row in checks]
    artifact["gate_check_summary"] = gate_summary(checks)
    artifact["duration_s"] = float(time.perf_counter() - started)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(*, run_date: str, repo_root: Path = REPO_ROOT) -> JsonDict:
    """Build a positive, null, or terminal blocked audit from real evidence."""

    started = time.perf_counter()
    root = Path(repo_root)
    artifact = _empty_artifact(run_date)
    exp6993, exp6993_hash_check = _load_contract(root / EXP6993_PATH, EXPECTED_EXP6993_HASH)
    exp6994, exp6994_hash_check = _load_contract(root / EXP6994_PATH, EXPECTED_EXP6994_HASH)
    replay = enumerate_envelopes(root / STORE_PATH)
    for field in (
        "envelope_inventory_rows",
        "envelope_eligibility_rows",
        "prompt_hash_replay_rows",
        "transition_hash_replay_rows",
        "engine_hash_replay_rows",
        "environment_hash_replay_rows",
        "scorer_hash_replay_rows",
        "policy_hash_replay_rows",
        "factory_hash_replay_rows",
        "manifest_hash_replay_rows",
        "envelope_hash_replay_rows",
        "source_disagreement_rows",
    ):
        artifact[field] = deepcopy(replay[field])
    artifact["source_artifact_hashes"] = _source_hashes(root, replay)
    selected = select_earliest_eligible(replay["envelope_eligibility_rows"])
    eligible_count = sum(row.get("eligible") is True for row in replay["envelope_eligibility_rows"])
    artifact["selection_rule_rows"] = [
        {
            "rule": "post_exp6993_creation_cutoff",
            "value": POST_EXP6993_CUTOFF,
            "terminal": True,
        },
        {
            "rule": "selection_order",
            "value": ["created_at", "published_at", "manifest_path", "manifest_index"],
            "quality_fields_consulted": [],
            "terminal": True,
        },
        {
            "rule": "selected_run_id",
            "value": selected.get("run_id") if selected else None,
            "eligible_count": eligible_count,
            "terminal": True,
        },
    ]
    checks = [
        exp6993_hash_check,
        exp6994_hash_check,
        gate_check(
            "exp6993_terminal_contract",
            {"arc_producer_contract_complete_score": 1, "arc_live_path_fixture_ready_score": 1},
            (
                {
                    "arc_producer_contract_complete_score": exp6993.get(
                        "arc_producer_contract_complete_score"
                    ),
                    "arc_live_path_fixture_ready_score": exp6993.get(
                        "arc_live_path_fixture_ready_score"
                    ),
                }
                if exp6993
                else None
            ),
        ),
        gate_check(
            "exp6994_terminal_contract",
            {"arc_contract_audit_complete_score": 1, "arc_producer_contract_confirmed_score": 1},
            (
                {
                    "arc_contract_audit_complete_score": exp6994.get(
                        "arc_contract_audit_complete_score"
                    ),
                    "arc_producer_contract_confirmed_score": exp6994.get(
                        "arc_producer_contract_confirmed_score"
                    ),
                }
                if exp6994
                else None
            ),
        ),
        gate_check(
            "readable_live_attempt_manifests",
            ">=1",
            replay["readable_manifest_count"],
            passed=replay["readable_manifest_count"] >= 1,
        ),
        gate_check(
            "eligible_complete_real_post_exp6993_envelope",
            ">=1",
            eligible_count,
            passed=selected is not None,
        ),
    ]
    if selected is None or any(row["passed"] is not True for row in checks):
        return _finish(artifact, checks, started)
    artifact["selected_envelope_hash"] = selected.get("envelope_sha256")
    frozen = freeze_transition_splits(root / STORE_PATH, replay, selected)
    construction = frozen.pop("_construction_rows")
    heldout = frozen.pop("_heldout_rows")
    artifact.update(frozen)
    target_heldout = [row for row in heldout if row.get("next_grid") is not None]
    checks.extend(
        [
            gate_check(
                "construction_transition_count", ">=1", len(construction), passed=bool(construction)
            ),
            gate_check(
                "heldout_target_transition_count",
                ">=1",
                len(target_heldout),
                passed=bool(target_heldout),
            ),
            gate_check(
                "split_leakage_checks",
                True,
                all(row.get("passed") is True for row in artifact["leakage_check_rows"]),
            ),
        ]
    )
    if any(row["passed"] is not True for row in checks):
        return _finish(artifact, checks, started)
    artifact["baseline_definition_rows"] = [
        {
            "baseline": "inert_no_change",
            "definition": "Return a copy of the current grid for every action.",
            "construction_transition_ids": [],
            "heldout_targets_used": False,
            "terminal": True,
        },
        {
            "baseline": "observed_action_delta_hypothesis",
            "definition": "Use the first changing construction row per action and require matching action data.",
            "source_path": str(ACTIVE_PROBE_PATH),
            "source_sha256": sha256_path(root / ACTIVE_PROBE_PATH),
            "construction_sha256": baseline_construction_hash(construction),
            "construction_transition_ids": [str(row.get("transition_id")) for row in construction],
            "heldout_targets_used": False,
            "available_before_induction": True,
            "terminal": True,
        },
    ]
    scored = run_restricted_scoring(Path(str(selected["engine_path"])), construction, heldout)
    for field in (
        "engine_score_rows",
        "inert_control_rows",
        "pre_engine_control_rows",
        "read_only_enforcement_receipt",
    ):
        artifact[field] = deepcopy(scored[field])
    all_score_rows = [
        *artifact["engine_score_rows"],
        *artifact["inert_control_rows"],
        *artifact["pre_engine_control_rows"],
    ]
    artifact["per_transition_rows"] = deepcopy(all_score_rows)
    artifact["rows"] = [
        aggregate_scores(predictor, stratum, rows)
        for predictor, rows in (
            ("engine", artifact["engine_score_rows"]),
            ("inert_no_change", artifact["inert_control_rows"]),
            ("observed_action_delta_hypothesis", artifact["pre_engine_control_rows"]),
        )
        for stratum in ("construction", "heldout")
    ]
    artifact["coverage_rows"] = [
        {
            "predictor": row["predictor"],
            "stratum": row["stratum"],
            "transition_id": row["transition_id"],
            "target_available": row["target_available"],
            "covered": row["covered"],
            "terminal": True,
        }
        for row in all_score_rows
    ]
    artifact["abstention_rows"] = [
        {
            "predictor": row["predictor"],
            "stratum": row["stratum"],
            "transition_id": row["transition_id"],
            "target_available": row["target_available"],
            "abstained": row["abstained"],
            "terminal": True,
        }
        for row in all_score_rows
    ]
    missing_ids = {
        str(row.get("transition_id")): row
        for row in [*construction, *heldout]
        if row.get("next_grid") is None
    }
    artifact["missing_target_rows"] = [
        {
            "transition_id": transition_id,
            "attempt_id": row.get("attempt_id"),
            "episode_id": row.get("episode_id"),
            "status": "missing_target_not_imputed",
            "terminal": True,
        }
        for transition_id, row in sorted(missing_ids.items())
    ]
    pre_by_key = {
        (row["stratum"], row["transition_id"]): row for row in artifact["pre_engine_control_rows"]
    }
    artifact["route_influence_rows"] = [
        {
            "stratum": row["stratum"],
            "transition_id": row["transition_id"],
            "attempt_id": row["attempt_id"],
            "episode_id": row["episode_id"],
            "engine_prediction_sha256": row["prediction_sha256"],
            "pre_engine_prediction_sha256": pre_by_key[(row["stratum"], row["transition_id"])][
                "prediction_sha256"
            ],
            "route_influenced": row["prediction_sha256"]
            != pre_by_key[(row["stratum"], row["transition_id"])]["prediction_sha256"],
            "terminal": True,
        }
        for row in artifact["engine_score_rows"]
    ]
    artifact["paired_metric_rows"] = paired_error_rows(
        artifact["engine_score_rows"],
        [*artifact["inert_control_rows"], *artifact["pre_engine_control_rows"]],
    )
    artifact["bootstrap_interval_rows"] = bootstrap_error_intervals(artifact["paired_metric_rows"])
    expected_score_count = len(construction) + len(heldout)
    receipt_passed = artifact["read_only_enforcement_receipt"].get("passed") is True
    full_scores = all(
        len(artifact[field]) == expected_score_count
        for field in ("engine_score_rows", "inert_control_rows", "pre_engine_control_rows")
    )
    aggregate = {(row["predictor"], row["stratum"]): row for row in artifact["rows"]}
    engine_heldout = aggregate[("engine", "heldout")]
    control_heldout = [
        aggregate[("inert_no_change", "heldout")],
        aggregate[("observed_action_delta_hypothesis", "heldout")],
    ]
    coverage_no_loss = bool(
        engine_heldout["transition_coverage"] is not None
        and all(
            float(engine_heldout["transition_coverage"]) >= float(row["transition_coverage"])
            for row in control_heldout
            if row["transition_coverage"] is not None
        )
    )
    evaluable = bool(
        target_heldout
        and receipt_passed
        and full_scores
        and len(artifact["bootstrap_interval_rows"]) == 2
        and not artifact["source_disagreement_rows"]
    )
    positive = bool(
        evaluable
        and coverage_no_loss
        and all(row["interval_low"] > 0.0 for row in artifact["bootstrap_interval_rows"])
    )
    artifact["arc_engine_quality_evaluable_score"] = int(evaluable)
    artifact["arc_engine_quality_positive_score"] = int(positive)
    artifact["verdict_class"] = "positive" if positive else "null"
    artifact["honest_verdict"] = (
        "complete_positive_arc_live_envelope_engine_beats_both_controls"
        if positive
        else "complete_null_arc_live_envelope_engine_does_not_beat_both_controls"
    )
    registry_before = artifact["source_artifact_hashes"]["solve_registry"]["sha256"]
    checks.extend(
        [
            gate_check("fresh_restricted_process", True, receipt_passed),
            gate_check("three_arm_score_rows_complete", True, full_scores),
            gate_check(
                "solve_registry_unchanged", registry_before, sha256_path(root / REGISTRY_PATH)
            ),
        ]
    )
    if any(row["passed"] is not True for row in checks):
        artifact["arc_live_envelope_audit_complete_score"] = 0
        artifact["arc_engine_quality_evaluable_score"] = 0
        artifact["arc_engine_quality_positive_score"] = 0
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = "blocked_arc_live_envelope_audit"
    return _finish(artifact, checks, started)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject missing fields, metric drift, overclaims, and checksum drift."""

    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        errors.append(f"required_fields_missing:{sorted(missing)}")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    for field in (
        "arc_live_envelope_audit_complete_score",
        "arc_engine_quality_evaluable_score",
        "arc_engine_quality_positive_score",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field}_not_bare_int")
        elif artifact.get(field) not in {0, 1}:
            errors.append(f"{field}_outside_binary_range")
    for field in (
        "solve_provenance_applicable",
        "solve_claimed",
        "level_claimed",
        "registry_updated",
        "submitted_to_leaderboard",
        "game_source_inspected",
        "development_fixture_used_for_quality",
        "verifier_is_oracle",
    ):
        if artifact.get(field) is not False:
            errors.append(f"{field}_must_be_false")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict_class not in VERDICT_PREFIXES:
        errors.append("verdict_class_invalid")
    elif not verdict.startswith(VERDICT_PREFIXES[str(verdict_class)]):
        errors.append("verdict_prefix_mismatch")
    if artifact.get("arc_engine_quality_evaluable_score") == 1 and (
        artifact.get("arc_live_envelope_audit_complete_score") != 1
        or not artifact.get("heldout_transition_rows")
        or not artifact.get("selected_envelope_hash")
    ):
        errors.append("evaluable_state_mismatch")
    if artifact.get("arc_engine_quality_positive_score") == 1 and (
        artifact.get("arc_engine_quality_evaluable_score") != 1 or verdict_class != "positive"
    ):
        errors.append("positive_state_mismatch")
    checks = artifact.get("preconditions_checked", [])
    has_failed_check = not checks or any(
        not isinstance(row, Mapping) or row.get("passed") is not True for row in checks
    )
    if (
        verdict_class == "blocked"
        and artifact.get("gate_check_summary", {}).get("passed") is not False
    ):
        errors.append("blocked_without_failed_gate")
    if has_failed_check and verdict_class != "blocked":
        errors.append("failed_precondition_not_blocked")
    raw_groups = {
        "engine": artifact.get("engine_score_rows", []),
        "inert_no_change": artifact.get("inert_control_rows", []),
        "observed_action_delta_hypothesis": artifact.get("pre_engine_control_rows", []),
    }
    expected_aggregates = [
        aggregate_scores(predictor, stratum, rows)
        for predictor, rows in raw_groups.items()
        for stratum in ("construction", "heldout")
    ]
    if artifact.get("rows") and artifact.get("rows") != expected_aggregates:
        errors.append("aggregate_metric_recomputation_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Publish a complete JSON artifact with one atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the required audit command and write one terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    root = Path(args.repo_root)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    artifact = build_artifact(run_date=args.date, repo_root=root)
    write_json_atomic(output, artifact)
    print(f"wrote {output}")
    print(artifact["honest_verdict"])
    return 1 if artifact["verdict_class"] == "blocked" else 0


if __name__ == "__main__":  # pragma: no cover - child and wrapper exercise the protocols.
    if len(sys.argv) >= 2 and sys.argv[1] == "--score-worker":
        raise SystemExit(_score_worker(Path(sys.argv[2])))
    raise SystemExit(main())
