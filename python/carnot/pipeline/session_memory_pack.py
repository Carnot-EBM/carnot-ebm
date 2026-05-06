"""Portable JSON pack import/export for SessionMemory.

The native ``SessionMemory`` file is intentionally local and implementation-facing.
This module wraps that state in a stable, schema-versioned pack format so learned
cases, template observations, and FP calibration can be shared across installations.

Spec: REQ-LEARN-060, REQ-LEARN-061, REQ-LEARN-062,
      SCENARIO-LEARN-104, SCENARIO-LEARN-105, SCENARIO-LEARN-106
"""

from __future__ import annotations

import json
import pathlib
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from os import PathLike
from typing import Any

from carnot._version import __version__ as CARNOT_VERSION
from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
from carnot.pipeline.case_memory import CaseEntry, CaseMemory
from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
from carnot.pipeline.session_memory import SessionMemory, _escape_model_id

PACK_SCHEMA = "carnot.session_memory_pack.v1"
PACK_SCHEMA_VERSION = "1.0.0"
PACK_SCHEMA_URL = "https://carnot-ebm.org/schemas/session_memory_v1.json"
DEFAULT_LICENSE = "Apache-2.0"

JsonDict = dict[str, Any]
PackInput = JsonDict | str | pathlib.Path | PathLike[str]


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_clone(payload: JsonDict) -> JsonDict:
    return json.loads(json.dumps(payload, sort_keys=True))


def _write_json(path: str | pathlib.Path | PathLike[str], payload: JsonDict) -> None:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_raw_state(session: SessionMemory) -> JsonDict | None:
    try:
        raw = json.loads(session._state_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _empty_components() -> tuple[CaseMemory, ConstraintTemplateLibrary, PerModelFPTracker]:
    library = ConstraintTemplateLibrary()
    library.register_builtin_templates()
    return CaseMemory(), library, PerModelFPTracker()


def _load_components(
    storage_dir: str | pathlib.Path | PathLike[str],
    model_id: str,
) -> tuple[CaseMemory, ConstraintTemplateLibrary, PerModelFPTracker, str | None]:
    session = SessionMemory(str(storage_dir), model_id)
    raw_state = _load_raw_state(session)
    loaded = session.load()
    if loaded is None:
        case_memory, template_library, fp_tracker = _empty_components()
    else:
        case_memory, template_library, fp_tracker = loaded
        template_library.register_builtin_templates()
    saved_at = str(raw_state.get("saved_at")) if raw_state and raw_state.get("saved_at") else None
    return case_memory, template_library, fp_tracker, saved_at


def _merge_unique(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in (*left, *right):
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _merge_provenance(left: tuple[Any, ...], right: tuple[Any, ...]) -> tuple[Any, ...]:
    result: list[Any] = list(left)
    for item in right:
        if item not in result:
            result.append(item)
    return tuple(result)


def _merge_case_entry(existing: CaseEntry, incoming: CaseEntry) -> CaseEntry:
    support = existing.support + incoming.support
    confidence = 0.0
    if support > 0:
        confidence = (
            (existing.confidence * existing.support) + (incoming.confidence * incoming.support)
        ) / support
    return CaseEntry(
        key=existing.key,
        case_kind=existing.case_kind or incoming.case_kind,
        benchmark=existing.benchmark or incoming.benchmark,
        violation_types=_merge_unique(existing.violation_types, incoming.violation_types),
        prompt_tokens=_merge_unique(existing.prompt_tokens, incoming.prompt_tokens),
        support=support,
        confidence=confidence,
        provenance=_merge_provenance(existing.provenance, incoming.provenance),
    )


def _redact_case_payload(payload: JsonDict) -> JsonDict:
    redacted = deepcopy(payload)
    for entry in redacted.get("entries", []):
        if not isinstance(entry, dict):
            continue
        for provenance in entry.get("provenance", []):
            if not isinstance(provenance, dict):
                continue
            provenance["case_id"] = "REDACTED"
            provenance["source_artifact"] = None
            provenance["verifier_path"] = ""
    return redacted


def _case_portable_entries(case_memory: CaseMemory, exported_at: str) -> list[JsonDict]:
    portable: list[JsonDict] = []
    for entry in case_memory.entries():
        canonical_form = entry.key.fingerprint
        portable.append(
            {
                "question_hash": sha256(canonical_form.encode("utf-8")).hexdigest()[:32],
                "canonical_form": canonical_form,
                "observed_precision": entry.confidence,
                "n_observations": entry.support,
                "last_seen": exported_at,
                "case_kind": entry.case_kind,
                "benchmark": entry.benchmark,
                "violation_types": list(entry.violation_types),
            }
        )
    return portable


def _target_claim_type(pattern_key: str) -> str:
    if pattern_key in {"carry_check", "sign_check", "comparison_direction"}:
        return "arithmetic"
    if pattern_key == "unit_consistency":
        return "unit"
    if pattern_key == "manipulable_signal_dependency":
        return "source_trust"
    return "custom"


def _template_summaries(template_library: ConstraintTemplateLibrary) -> list[JsonDict]:
    if not template_library._templates:
        template_library.register_builtin_templates()
    summaries: list[JsonDict] = []
    for pattern_key in sorted(template_library._templates):
        template = template_library._templates[pattern_key]
        summaries.append(
            {
                "id": template.pattern_key,
                "target_claim_type": _target_claim_type(template.pattern_key),
                "description": template.description,
                "patterns": {"regex": [], "ast": []},
                "energy_weight": 1.0,
                "min_frequency": template.min_frequency,
                "is_active": template.is_active,
                "activation_count": template.activation_count,
            }
        )
    return summaries


def _session_state_summary(session: SessionMemory) -> JsonDict:
    violations_by_type = getattr(session, "_violations_by_type", {})
    violations_by_domain = getattr(session, "_violations_by_domain", {})
    return {
        "violations_by_type": [
            {"violation_type": key, "count": value}
            for key, value in sorted(violations_by_type.items())
        ],
        "violations_by_domain": [
            {"domain": key, "count": value} for key, value in sorted(violations_by_domain.items())
        ],
    }


def _metadata_payload(metadata: dict[str, Any] | None, exported_at: str) -> JsonDict:
    payload: JsonDict = {
        "created_at": exported_at,
        "source": "local-session",
        "license": DEFAULT_LICENSE,
        "carnot_version": CARNOT_VERSION,
        "backwards_compatibility": "v1 readers reject unknown major versions",
    }
    if metadata:
        payload.update(metadata)
    if not payload.get("source"):
        payload["source"] = "local-session"
    if not payload.get("license"):
        payload["license"] = DEFAULT_LICENSE
    return payload


def export_session_memory(
    storage_dir: str | pathlib.Path | PathLike[str],
    model_id: str,
    output_path: str | pathlib.Path | PathLike[str] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    redact_provenance: bool = False,
) -> JsonDict:
    """Export one model's SessionMemory state as a portable pack.

    Missing local state exports as a valid empty pack so starter packs and fresh
    installations use the same schema path as populated installations.

    Spec: REQ-LEARN-060-2
    """
    exported_at = _utc_now()
    case_memory, template_library, fp_tracker, saved_at = _load_components(storage_dir, model_id)
    case_payload = case_memory.to_dict()
    if redact_provenance:
        case_payload = _redact_case_payload(case_payload)
    case_payload["portable_entries"] = _case_portable_entries(case_memory, exported_at)
    session = SessionMemory(str(storage_dir), model_id)
    model_payload: JsonDict = {
        "model_id": model_id,
        "safe_model_id": _escape_model_id(model_id),
        "saved_at": saved_at,
        "exported_at": exported_at,
        "case_memory": case_payload,
        "constraint_templates": _template_summaries(template_library),
        "template_library": template_library.to_dict(),
        "fp_tracker": fp_tracker.to_dict(),
        "session_state": _session_state_summary(session),
    }
    pack: JsonDict = {
        "$schema": PACK_SCHEMA_URL,
        "schema": PACK_SCHEMA,
        "schema_version": PACK_SCHEMA_VERSION,
        "metadata": _metadata_payload(metadata, exported_at),
        "models": [model_payload],
    }
    validate_session_memory_pack(pack)
    if output_path is not None:
        _write_json(output_path, pack)
    return pack


def load_session_memory_pack(path: str | pathlib.Path | PathLike[str]) -> JsonDict:
    """Load and validate a portable SessionMemory pack from disk.

    Spec: REQ-LEARN-060-3
    """
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SessionMemory pack must be a JSON object")
    validate_session_memory_pack(payload)
    return payload


def _coerce_pack(pack: PackInput) -> JsonDict:
    if isinstance(pack, dict):
        payload = _json_clone(pack)
        validate_session_memory_pack(payload)
        return payload
    return load_session_memory_pack(pack)


def validate_session_memory_pack(payload: JsonDict) -> None:
    """Validate the minimal portable pack contract without extra dependencies.

    The checked-in JSON Schema is the public contract; this function is the
    lightweight runtime guard used by CLI and tests.

    Spec: REQ-LEARN-060-3
    """
    if not isinstance(payload, dict):
        raise ValueError("SessionMemory pack must be a JSON object")
    if payload.get("schema") != PACK_SCHEMA:
        raise ValueError(f"schema must be {PACK_SCHEMA!r}")
    version = payload.get("schema_version")
    if not isinstance(version, str) or not version.startswith("1."):
        raise ValueError("schema_version must be a supported v1 version")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be present")
    if not isinstance(metadata.get("source"), str) or not metadata.get("source"):
        raise ValueError("metadata.source must be present")
    if not isinstance(metadata.get("license"), str) or not metadata.get("license"):
        raise ValueError("metadata.license must be present")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    seen_model_ids: set[str] = set()
    required_sections = {
        "model_id",
        "safe_model_id",
        "case_memory",
        "constraint_templates",
        "template_library",
        "fp_tracker",
        "session_state",
    }
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("models entries must be JSON objects")
        missing = sorted(required_sections - set(model))
        if missing:
            raise ValueError(f"model {model.get('model_id', '<unknown>')} missing {missing[0]}")
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be present")
        if model_id in seen_model_ids:
            raise ValueError(f"duplicate model_id: {model_id}")
        seen_model_ids.add(model_id)
        case_memory = model.get("case_memory")
        if not isinstance(case_memory, dict) or not isinstance(case_memory.get("entries"), list):
            raise ValueError("case_memory.entries must be present")
        if case_memory.get("version") != 1:
            raise ValueError("case_memory.version must be 1")
        if not isinstance(case_memory.get("portable_entries", []), list):
            raise ValueError("case_memory.portable_entries must be a list")
        if not isinstance(model.get("constraint_templates"), list):
            raise ValueError("constraint_templates must be a list")
        template_library = model.get("template_library")
        if not isinstance(template_library, dict) or not isinstance(
            template_library.get("observations", []), list
        ):
            raise ValueError("template_library.observations must be present")
        fp_tracker = model.get("fp_tracker")
        if not isinstance(fp_tracker, dict) or not isinstance(fp_tracker.get("stats", []), list):
            raise ValueError("fp_tracker.stats must be present")
        session_state = model.get("session_state")
        if not isinstance(session_state, dict):
            raise ValueError("session_state must be present")


def _select_model(payload: JsonDict, model_id: str | None) -> JsonDict:
    models = payload["models"]
    if model_id is None:
        return models[0]
    for model in models:
        if model["model_id"] == model_id:
            return model
    if len(models) == 1:
        replacement = deepcopy(models[0])
        replacement["model_id"] = model_id
        replacement["safe_model_id"] = _escape_model_id(model_id)
        return replacement
    raise ValueError(f"model_id {model_id!r} not found in pack")


def _components_from_model(model: JsonDict) -> tuple[CaseMemory, ConstraintTemplateLibrary, PerModelFPTracker]:
    case_memory = CaseMemory.from_dict(model["case_memory"])
    template_library = ConstraintTemplateLibrary.from_dict(model["template_library"])
    template_library.register_builtin_templates()
    fp_tracker = PerModelFPTracker.from_dict(model["fp_tracker"])
    return case_memory, template_library, fp_tracker


def _count_components(
    case_memory: CaseMemory,
    template_library: ConstraintTemplateLibrary,
    fp_tracker: PerModelFPTracker,
) -> JsonDict:
    return {
        "case_entries": len(case_memory.entries()),
        "template_observations": len(template_library._observations),
        "fp_stats": len(fp_tracker._stats),
    }


def _merge_components(
    local: tuple[CaseMemory, ConstraintTemplateLibrary, PerModelFPTracker],
    incoming: tuple[CaseMemory, ConstraintTemplateLibrary, PerModelFPTracker],
) -> tuple[CaseMemory, ConstraintTemplateLibrary, PerModelFPTracker, JsonDict]:
    local_case, local_templates, local_fp = local
    incoming_case, incoming_templates, incoming_fp = incoming

    case_entries_added = 0
    case_entries_merged = 0
    for entry in incoming_case.entries():
        existing = local_case._entries.get(entry.key)
        if existing is None:
            local_case._entries[entry.key] = entry
            case_entries_added += 1
        else:
            local_case._entries[entry.key] = _merge_case_entry(existing, entry)
            case_entries_merged += 1

    template_observations_added = 0
    template_observations_merged = 0
    for key, count in incoming_templates._observations.items():
        if key in local_templates._observations:
            local_templates._observations[key] += count
            template_observations_merged += 1
        else:
            local_templates._observations[key] = count
            template_observations_added += 1
    local_templates.register_builtin_templates()

    fp_stats_added = 0
    fp_stats_merged = 0
    for key, incoming_counts in incoming_fp._stats.items():
        if key not in local_fp._stats:
            local_fp._stats[key] = dict(incoming_counts)
            fp_stats_added += 1
            continue
        local_counts = local_fp._stats[key]
        for field in ("fp_count", "tp_count", "n_observations"):
            local_counts[field] = local_counts.get(field, 0) + incoming_counts.get(field, 0)
        fp_stats_merged += 1

    details = {
        "case_entries_added": case_entries_added,
        "case_entries_merged": case_entries_merged,
        "template_observations_added": template_observations_added,
        "template_observations_merged": template_observations_merged,
        "fp_stats_added": fp_stats_added,
        "fp_stats_merged": fp_stats_merged,
    }
    return local_case, local_templates, local_fp, details


def import_session_memory(
    pack: PackInput,
    storage_dir: str | pathlib.Path | PathLike[str],
    *,
    model_id: str | None = None,
    merge: bool = False,
    replace: bool = False,
    dry_run: bool = False,
) -> JsonDict:
    """Import a portable pack into local SessionMemory storage.

    The safe default is merge mode when neither merge nor replace is specified.

    Spec: REQ-LEARN-061
    """
    if merge and replace:
        raise ValueError("merge and replace are mutually exclusive")
    if not merge and not replace:
        merge = True

    payload = _coerce_pack(pack)
    selected_model = _select_model(payload, model_id)
    target_model_id = str(selected_model["model_id"])
    incoming = _components_from_model(selected_model)
    session = SessionMemory(str(storage_dir), target_model_id)
    loaded = session.load()
    if loaded is None:
        local = _empty_components()
    else:
        local_case, local_templates, local_fp = loaded
        local_templates.register_builtin_templates()
        local = (local_case, local_templates, local_fp)

    before = _count_components(*local)
    if replace:
        final_case, final_templates, final_fp = incoming
        mode_details = {
            "case_entries_added": len(final_case.entries()),
            "case_entries_merged": 0,
            "template_observations_added": len(final_templates._observations),
            "template_observations_merged": 0,
            "fp_stats_added": len(final_fp._stats),
            "fp_stats_merged": 0,
        }
        mode = "replace"
    else:
        final_case, final_templates, final_fp, mode_details = _merge_components(local, incoming)
        mode = "merge"
    after = _count_components(final_case, final_templates, final_fp)

    if not dry_run:
        session.save(final_case, final_templates, final_fp)

    return {
        "schema": "carnot.session_memory_import_report.v1",
        "mode": mode,
        "model_id": target_model_id,
        "dry_run": dry_run,
        "written": not dry_run,
        "before": before,
        "after": after,
        **mode_details,
    }


def _semantic_model(model: JsonDict) -> JsonDict:
    case_memory = model["case_memory"]
    return {
        "model_id": model["model_id"],
        "case_memory": {
            "version": case_memory["version"],
            "entries": case_memory["entries"],
        },
        "template_library": model["template_library"],
        "fp_tracker": model["fp_tracker"],
        "session_state": model["session_state"],
    }


def _model_index(payload: JsonDict) -> dict[str, JsonDict]:
    return {str(model["model_id"]): model for model in payload["models"]}


def _stable_hash(payload: JsonDict) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def diff_session_memory_packs(left: PackInput, right: PackInput) -> JsonDict:
    """Return a semantic diff between two portable SessionMemory packs.

    Export timestamps and other pack metadata are ignored. Learning state sections
    are compared exactly.

    Spec: REQ-LEARN-061-1
    """
    left_payload = _coerce_pack(left)
    right_payload = _coerce_pack(right)
    left_models = _model_index(left_payload)
    right_models = _model_index(right_payload)
    left_ids = set(left_models)
    right_ids = set(right_models)
    added = sorted(right_ids - left_ids)
    removed = sorted(left_ids - right_ids)
    changed: list[JsonDict] = []
    for model_id in sorted(left_ids & right_ids):
        left_semantic = _semantic_model(left_models[model_id])
        right_semantic = _semantic_model(right_models[model_id])
        if _stable_hash(left_semantic) == _stable_hash(right_semantic):
            continue
        changed.append(
            {
                "model_id": model_id,
                "left_hash": _stable_hash(left_semantic),
                "right_hash": _stable_hash(right_semantic),
            }
        )
    return {
        "schema": "carnot.session_memory_diff.v1",
        "is_empty": not added and not removed and not changed,
        "models_added": added,
        "models_removed": removed,
        "models_changed": changed,
    }
