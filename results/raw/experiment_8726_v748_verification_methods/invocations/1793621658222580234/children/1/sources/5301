"""Exp 1514 trace2skill portable skill/provenance pack.

Exp 1513 decides which verifier-feedback policy updates survive rollback.
This module turns only those rollback-passing rows into a compact local-first
distribution manifest.  The manifest keeps enough provenance for a future
importer to resolve the source row, inspect the verifier evidence, and know
that rejected rows were not silently promoted.

Spec: REQ-LEARN-1514, SCENARIO-LEARN-1516, SCENARIO-LEARN-1517.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OPS_DIR = REPO_ROOT / "ops"

OUTPUT_FILE = "experiment_1514_trace2skill_portable_skill_pack_v2.json"
PACK_MANIFEST_FILE = "trace2skill_portable_skill_pack_manifest_1514.json"
ROLLBACK_MANIFEST_FILE = "fr11_policy_rollback_replay_1513.jsonl"
OPS_NOTE_FILE = "ops/trace2skill_portable_skill_pack_1514.md"

DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_PACK_MANIFEST_PATH = DEFAULT_RESULTS_DIR / PACK_MANIFEST_FILE
DEFAULT_OPS_NOTE_PATH = REPO_ROOT / OPS_NOTE_FILE
DEFAULT_ROLLBACK_ARTIFACT_PATH = (
    DEFAULT_RESULTS_DIR / "experiment_1513_fr11_policy_rollback_replay_audit.json"
)
DEFAULT_REACHABILITY_ARTIFACT_PATH = (
    DEFAULT_RESULTS_DIR / "experiment_1498_trace2skill_artifact_reachability_audit.json"
)
DEFAULT_ROLLBACK_MANIFEST_PATH = DEFAULT_RESULTS_DIR / ROLLBACK_MANIFEST_FILE

RUN_DATE = "20260508"
ARTIFACT_SCHEMA = "trace2skill_portable_skill_pack_artifact_v2"
PACK_SCHEMA = "trace2skill_portable_skill_pack_v1"

PASSED_VERDICT = "complete: trace2skill_portable_skill_pack_ready"
EMPTY_VERDICT = "complete: trace2skill_portable_skill_pack_zero_eligible_no_false_promotion"
GATED_VERDICT = "complete: trace2skill_portable_skill_pack_gate_blocked"

TERMINAL_VERDICT_PREFIXES: tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "portable_skill_pack_ready",
    "gated_inputs_present",
    "rollback_passing_entries",
    "packaged_skill_entries",
    "rejected_skill_entries",
    "provenance_fields_present",
    "resolver_keys_present",
    "pack_manifest_path",
    "ops_note_path",
    "blockers",
    "honest_verdict",
)

REQUIRED_ENTRY_FIELDS: tuple[str, ...] = (
    "skill_id",
    "source_artifact",
    "verifier_evidence",
    "resolver_key",
    "created_date",
    "promotion_status",
)

JsonDict = dict[str, Any]


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _display_path(path: Path | str, *, project_root: Path | str = REPO_ROOT) -> str:
    target = Path(path)
    try:
        return target.relative_to(Path(project_root)).as_posix()
    except ValueError:
        return target.name


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    destination.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return serializable


def _load_json(path: Path | str) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON artifact must be an object: {path}")
    return payload


def _load_jsonl(path: Path | str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise AssertionError(f"JSONL row must be an object: {path}")
        rows.append(row)
    return rows


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    pack_manifest_path: Path | str = DEFAULT_PACK_MANIFEST_PATH,
    ops_note_path: Path | str = DEFAULT_OPS_NOTE_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1514-1/5: write the bootstrap artifact before source loading."""

    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA,
        "spec": ["REQ-LEARN-1514", "SCENARIO-LEARN-1516", "SCENARIO-LEARN-1517"],
        "run_date": run_date,
        "started_at": _timestamp(),
        "status": "in_progress",
        "portable_skill_pack_ready": False,
        "gated_inputs_present": False,
        "rollback_passing_entries": 0,
        "packaged_skill_entries": 0,
        "rejected_skill_entries": 0,
        "provenance_fields_present": False,
        "resolver_keys_present": False,
        "pack_manifest_path": _display_path(pack_manifest_path, project_root=project_root),
        "ops_note_path": _display_path(ops_note_path, project_root=project_root),
        "blockers": ["pack_build_in_progress"],
        "honest_verdict": "complete: trace2skill_portable_skill_pack_in_progress",
    }
    return _write_json(output_path, artifact)


def _reachable_source_artifacts(reachability_artifact: Mapping[str, Any]) -> list[str]:
    audit = reachability_artifact.get("source_artifact_audit")
    if not isinstance(audit, Sequence) or isinstance(audit, (str, bytes)):
        return []
    paths: list[str] = []
    for record in audit:
        if isinstance(record, Mapping) and record.get("status") == "reachable":
            path = str(record.get("path") or record.get("referenced_as") or "")
            if path:
                paths.append(path)
    return sorted(dict.fromkeys(paths))


def _resolver_checks(reachability_artifact: Mapping[str, Any]) -> list[str]:
    raw = reachability_artifact.get("resolver_keys")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return list(dict.fromkeys(str(value) for value in raw if value))


def _verifier_evidence(row: Mapping[str, Any]) -> JsonDict:
    return {
        "rollback_decision": str(row.get("decision") or ""),
        "policy_action": str(row.get("policy_action") or ""),
        "false_accept_delta": int(row.get("false_accept_delta", 0)),
        "soundness_mistakes": int(row.get("soundness_mistakes", 0)),
        "deterministic_validator_supported": bool(row.get("deterministic_validator_supported")),
        "source_evidence_reachable": bool(row.get("source_evidence_reachable")),
        "source_evidence_stale": bool(row.get("source_evidence_stale")),
        "utility_delta": int(row.get("utility_delta", 0)),
    }


def _rejection_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not row.get("skill_id"):
        reasons.append("missing_skill_id")
    if row.get("decision") != "keep":
        reasons.append("rollback_decision_not_keep")
    if row.get("source_evidence_reachable") is not True:
        reasons.append("source_evidence_unreachable")
    if bool(row.get("source_evidence_stale")):
        reasons.append("source_evidence_stale")
    if row.get("deterministic_validator_supported") is not True:
        reasons.append("missing_deterministic_validator_support")
    if int(row.get("soundness_mistakes", 0)) > 0:
        reasons.append("soundness_mistake")
    if int(row.get("false_accept_delta", 0)) > 0:
        reasons.append("false_accept_delta_positive")
    rollback_reasons = row.get("rollback_reasons")
    if isinstance(rollback_reasons, Sequence) and not isinstance(rollback_reasons, (str, bytes)):
        reasons.extend(f"rollback_reason:{reason}" for reason in rollback_reasons if reason)
    if row.get("exp1512_quarantined") is True:
        reasons.append("exp1512_quarantined")
    return sorted(dict.fromkeys(reasons))


def _base_entry(
    row: Mapping[str, Any],
    *,
    rollback_manifest_path: Path | str,
    project_root: Path | str,
    run_date: str,
) -> JsonDict:
    resolver_key = str(row.get("source_event_id") or row.get("skill_id") or "")
    return {
        "skill_id": str(row.get("skill_id") or ""),
        "source_event_id": str(row.get("source_event_id") or ""),
        "source_case_id": str(row.get("source_case_id") or ""),
        "source_kind": str(row.get("source_kind") or ""),
        "source_artifact": _display_path(rollback_manifest_path, project_root=project_root),
        "verifier_evidence": _verifier_evidence(row),
        "resolver_key": resolver_key,
        "created_date": run_date,
    }


def build_pack_manifest(
    rollback_rows: Sequence[Mapping[str, Any]],
    *,
    reachability_artifact: Mapping[str, Any],
    rollback_manifest_path: Path | str = DEFAULT_ROLLBACK_MANIFEST_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1514-3/4: package eligible rows and reject the rest."""

    resolver_checks = _resolver_checks(reachability_artifact)
    reachable_sources = _reachable_source_artifacts(reachability_artifact)
    entries: list[JsonDict] = []
    rejected_entries: list[JsonDict] = []
    for row in rollback_rows:
        reasons = _rejection_reasons(row)
        base = _base_entry(
            row,
            rollback_manifest_path=rollback_manifest_path,
            project_root=project_root,
            run_date=run_date,
        )
        if reasons:
            rejected_entries.append(
                {
                    **base,
                    "promotion_status": "rejected_not_promoted",
                    "rejection_reasons": reasons,
                }
            )
            continue
        entries.append(
            {
                **base,
                "reachable_source_artifacts": list(reachable_sources),
                "resolver_checks": list(resolver_checks),
                "promotion_status": "packaged_rollback_passed",
            }
        )
    manifest: JsonDict = {
        "schema": PACK_SCHEMA,
        "spec": ["REQ-LEARN-1514", "SCENARIO-LEARN-1516", "SCENARIO-LEARN-1517"],
        "run_date": run_date,
        "created_at": _timestamp(),
        "source_artifacts": [
            _display_path(rollback_manifest_path, project_root=project_root),
            *reachable_sources,
        ],
        "resolver_keys": resolver_checks,
        "packaged_entry_count": len(entries),
        "rejected_entry_count": len(rejected_entries),
        "entries": entries,
        "rejected_entries": rejected_entries,
    }
    validate_pack_manifest(manifest)
    return manifest


def _entry_has_provenance(entry: Mapping[str, Any]) -> bool:
    return all(field in entry and entry[field] not in ("", None) for field in REQUIRED_ENTRY_FIELDS)


def _artifact_blockers(
    *,
    gated_inputs_present: bool,
    manifest_exists: bool,
    gate_blockers: Sequence[str],
) -> list[str]:
    blockers = list(gate_blockers)
    if not gated_inputs_present:
        blockers.append("gated_inputs_missing")
    if gated_inputs_present and not manifest_exists:
        blockers.append("pack_manifest_not_written")
    return sorted(dict.fromkeys(blockers))


def build_artifact(
    *,
    rollback_artifact: Mapping[str, Any],
    reachability_artifact: Mapping[str, Any],
    pack_manifest: Mapping[str, Any],
    pack_manifest_path: Path | str,
    ops_note_path: Path | str,
    manifest_exists: bool,
    gated_inputs_present: bool,
    gate_blockers: Sequence[str],
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-LEARN-1514-5: summarize the terminal portable-pack contract."""

    entries = pack_manifest.get("entries", [])
    rejected_entries = pack_manifest.get("rejected_entries", [])
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        entries = []
    if not isinstance(rejected_entries, Sequence) or isinstance(rejected_entries, (str, bytes)):
        rejected_entries = []
    packaged_count = len(entries)
    rejected_count = len(rejected_entries)
    provenance_fields_present = all(
        isinstance(entry, Mapping) and _entry_has_provenance(entry) for entry in entries
    )
    resolver_keys_present = bool(pack_manifest.get("resolver_keys")) and all(
        isinstance(entry, Mapping) and bool(entry.get("resolver_key")) for entry in entries
    )
    blockers = _artifact_blockers(
        gated_inputs_present=gated_inputs_present,
        manifest_exists=manifest_exists,
        gate_blockers=gate_blockers,
    )
    portable_ready = bool(
        gated_inputs_present
        and not blockers
        and (packaged_count > 0 or rejected_count >= 0)
        and provenance_fields_present
    )
    if packaged_count > 0 and not resolver_keys_present:
        portable_ready = False
        blockers.append("resolver_keys_missing")
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA,
        "spec": ["REQ-LEARN-1514", "SCENARIO-LEARN-1516", "SCENARIO-LEARN-1517"],
        "run_date": run_date,
        "finished_at": _timestamp(),
        "status": "complete" if portable_ready else "blocked",
        "portable_skill_pack_ready": portable_ready,
        "gated_inputs_present": gated_inputs_present,
        "rollback_passing_entries": packaged_count,
        "packaged_skill_entries": packaged_count,
        "rejected_skill_entries": rejected_count,
        "provenance_fields_present": provenance_fields_present,
        "resolver_keys_present": resolver_keys_present,
        "pack_manifest_path": _display_path(pack_manifest_path, project_root=project_root),
        "ops_note_path": _display_path(ops_note_path, project_root=project_root),
        "blockers": sorted(dict.fromkeys(blockers)),
        "honest_verdict": (
            PASSED_VERDICT
            if portable_ready and packaged_count > 0
            else EMPTY_VERDICT
            if portable_ready
            else GATED_VERDICT
        ),
        "source_summary": {
            "rollback_audit_passed": bool(rollback_artifact.get("rollback_audit_passed")),
            "artifact_reachability_audit_complete": bool(
                reachability_artifact.get("artifact_reachability_audit_complete")
            ),
        },
        "tests_run": list(tests_run or []),
    }
    validate_artifact(artifact)
    return artifact


def validate_pack_manifest(manifest: Mapping[str, Any]) -> None:
    """REQ-LEARN-1514-4: ensure packaged and rejected rows cannot be confused."""

    if manifest.get("schema") != PACK_SCHEMA:
        raise AssertionError("unsupported pack manifest schema")
    entries = manifest.get("entries")
    rejected_entries = manifest.get("rejected_entries")
    if not isinstance(entries, list):
        raise AssertionError("entries must be a list")
    if not isinstance(rejected_entries, list):
        raise AssertionError("rejected_entries must be a list")
    if int(manifest.get("packaged_entry_count", -1)) != len(entries):
        raise AssertionError("packaged_entry_count must match entries")
    if int(manifest.get("rejected_entry_count", -1)) != len(rejected_entries):
        raise AssertionError("rejected_entry_count must match rejected_entries")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise AssertionError("packaged entries must be objects")
        missing = [field for field in REQUIRED_ENTRY_FIELDS if field not in entry]
        if missing:
            raise AssertionError(f"packaged entry missing fields: {missing}")
        if entry["promotion_status"] != "packaged_rollback_passed":
            raise AssertionError("packaged entry has invalid promotion status")
        if not entry.get("resolver_key"):
            raise AssertionError("packaged entry missing resolver key")
    for entry in rejected_entries:
        if not isinstance(entry, Mapping):
            raise AssertionError("rejected entries must be objects")
        if entry.get("promotion_status") != "rejected_not_promoted":
            raise AssertionError("rejected entry must not be promoted")
        reasons = entry.get("rejection_reasons")
        if not isinstance(reasons, list) or not reasons:
            raise AssertionError("rejected entry requires rejection reasons")


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    pack_manifest_path: Path | str | None = None,
    ops_note_path: Path | str | None = None,
) -> None:
    """REQ-LEARN-1514-5: enforce required fields and no-false-promotion counts."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    if artifact["status"] == "in_progress":
        return
    packaged = int(artifact["packaged_skill_entries"])
    rejected = int(artifact["rejected_skill_entries"])
    rollback_passing = int(artifact["rollback_passing_entries"])
    if packaged < 0 or rejected < 0 or rollback_passing < 0:
        raise AssertionError("entry counts must be non-negative")
    if packaged != rollback_passing:
        raise AssertionError("packaged entries must equal rollback-passing entries")
    if artifact["portable_skill_pack_ready"]:
        if artifact["blockers"]:
            raise AssertionError("ready pack cannot have blockers")
        if packaged > 0 and artifact["provenance_fields_present"] is not True:
            raise AssertionError("packaged entries require provenance fields")
        if packaged > 0 and artifact["resolver_keys_present"] is not True:
            raise AssertionError("packaged entries require resolver keys")
        if pack_manifest_path is not None and not Path(pack_manifest_path).exists():
            raise AssertionError("ready pack requires a manifest path")
        if ops_note_path is not None and not Path(ops_note_path).exists():
            raise AssertionError("ready pack requires an ops note path")
    elif not artifact["blockers"]:
        raise AssertionError("not-ready artifact must explain blockers")
    if (
        artifact["gated_inputs_present"] is True
        and packaged == 0
        and rejected == 0
        and artifact["portable_skill_pack_ready"] is False
    ):
        raise AssertionError("zero-entry artifacts must be ready or explicitly gated")


def write_ops_note(
    pack_manifest: Mapping[str, Any],
    artifact: Mapping[str, Any],
    note_path: Path | str = DEFAULT_OPS_NOTE_PATH,
) -> str:
    """REQ-LEARN-1514-4: summarize packaged and rejected entries for operators."""

    packaged = int(pack_manifest.get("packaged_entry_count", 0))
    rejected = int(pack_manifest.get("rejected_entry_count", 0))
    lines = [
        "# Trace2Skill Portable Skill Pack 1514",
        "",
        f"Run date: {artifact.get('run_date', RUN_DATE)}",
        f"Packaged entries: {packaged}",
        f"Rejected entries: {rejected}",
        f"Pack manifest: {artifact.get('pack_manifest_path')}",
        "",
        "Packaged entries include only rollback-passing rows with reachable evidence and deterministic validator support.",
    ]
    if rejected:
        lines.extend(["", "Rejected rows remain unpromoted and are listed in the pack manifest."])
    text = "\n".join(lines) + "\n"
    destination = Path(note_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return text


def _empty_pack_manifest(run_date: str) -> JsonDict:
    return {
        "schema": PACK_SCHEMA,
        "spec": ["REQ-LEARN-1514", "SCENARIO-LEARN-1517"],
        "run_date": run_date,
        "created_at": _timestamp(),
        "source_artifacts": [],
        "resolver_keys": [],
        "packaged_entry_count": 0,
        "rejected_entry_count": 0,
        "entries": [],
        "rejected_entries": [],
    }


def _reachability_clean(artifact: Mapping[str, Any]) -> bool:
    return bool(
        artifact.get("artifact_reachability_audit_complete") is True
        and artifact.get("gated_inputs_present") is True
        and int(artifact.get("unreachable_artifact_count", 0)) == 0
        and int(artifact.get("stale_artifact_count", 0)) == 0
        and int(artifact.get("ambiguous_resolver_count", 0)) == 0
        and not artifact.get("blockers")
    )


def _gate_inputs(
    *,
    rollback_artifact_path: Path | str,
    reachability_artifact_path: Path | str,
    rollback_manifest_path: Path | str,
) -> tuple[bool, list[str], JsonDict, JsonDict]:
    blockers: list[str] = []
    rollback_artifact: JsonDict = {}
    reachability_artifact: JsonDict = {}
    rollback_path = Path(rollback_artifact_path)
    reachability_path = Path(reachability_artifact_path)
    manifest_path = Path(rollback_manifest_path)
    if not rollback_path.exists():
        blockers.append("missing_exp1513_rollback_artifact")
    else:
        rollback_artifact = _load_json(rollback_path)
        if rollback_artifact.get("rollback_audit_passed") is not True:
            blockers.append("exp1513_rollback_audit_not_passed")
    if not reachability_path.exists():
        blockers.append("missing_exp1498_reachability_artifact")
    else:
        reachability_artifact = _load_json(reachability_path)
        if not _reachability_clean(reachability_artifact):
            blockers.append("exp1498_reachability_not_clean")
    if not manifest_path.exists():
        blockers.append("missing_rollback_manifest")
    return not blockers, sorted(dict.fromkeys(blockers)), rollback_artifact, reachability_artifact


def run(
    *,
    rollback_artifact_path: Path | str = DEFAULT_ROLLBACK_ARTIFACT_PATH,
    reachability_artifact_path: Path | str = DEFAULT_REACHABILITY_ARTIFACT_PATH,
    rollback_manifest_path: Path | str = DEFAULT_ROLLBACK_MANIFEST_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    pack_manifest_path: Path | str = DEFAULT_PACK_MANIFEST_PATH,
    ops_note_path: Path | str = DEFAULT_OPS_NOTE_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run Exp 1514 and write the pack manifest, ops note, and terminal artifact."""

    write_in_progress_artifact(
        output_path,
        pack_manifest_path=pack_manifest_path,
        ops_note_path=ops_note_path,
        project_root=project_root,
        run_date=run_date,
    )
    gated, blockers, rollback_artifact, reachability_artifact = _gate_inputs(
        rollback_artifact_path=rollback_artifact_path,
        reachability_artifact_path=reachability_artifact_path,
        rollback_manifest_path=rollback_manifest_path,
    )
    if not gated:
        artifact = build_artifact(
            rollback_artifact=rollback_artifact,
            reachability_artifact=reachability_artifact,
            pack_manifest=_empty_pack_manifest(run_date),
            pack_manifest_path=pack_manifest_path,
            ops_note_path=ops_note_path,
            manifest_exists=False,
            gated_inputs_present=False,
            gate_blockers=blockers,
            project_root=project_root,
            run_date=run_date,
            tests_run=tests_run,
        )
        _write_json(output_path, artifact)
        return artifact

    rollback_rows = _load_jsonl(rollback_manifest_path)
    pack_manifest = build_pack_manifest(
        rollback_rows,
        reachability_artifact=reachability_artifact,
        rollback_manifest_path=rollback_manifest_path,
        project_root=project_root,
        run_date=run_date,
    )
    _write_json(pack_manifest_path, pack_manifest)
    artifact = build_artifact(
        rollback_artifact=rollback_artifact,
        reachability_artifact=reachability_artifact,
        pack_manifest=pack_manifest,
        pack_manifest_path=pack_manifest_path,
        ops_note_path=ops_note_path,
        manifest_exists=Path(pack_manifest_path).exists(),
        gated_inputs_present=True,
        gate_blockers=[],
        project_root=project_root,
        run_date=run_date,
        tests_run=tests_run,
    )
    write_ops_note(pack_manifest, artifact, ops_note_path)
    validate_artifact(
        artifact,
        pack_manifest_path=pack_manifest_path,
        ops_note_path=ops_note_path,
    )
    _write_json(output_path, artifact)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
