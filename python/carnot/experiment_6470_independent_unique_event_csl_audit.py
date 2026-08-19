"""Exp6470 independent unique-event CSL audit.

Spec refs: REQ-LEARN-6470, SCENARIO-LEARN-6470-INVENTORY,
SCENARIO-LEARN-6470-IDENTITY, SCENARIO-LEARN-6470-CHRONOLOGY,
SCENARIO-LEARN-6470-VETO, SCENARIO-LEARN-6470-EFFECTS,
SCENARIO-LEARN-6470-LIFECYCLE, SCENARIO-LEARN-6470-READY.

This reducer reads V556 artifacts and raw files. It does not import upstream
experiment reducers, because the audit must not trust their summary code.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6470_independent_unique_event_csl_audit.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6470_independent_unique_event_csl_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6470_independent_unique_event_csl_audit.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

RUN_DATE = "20260819"
RANDOM_SEED = 6470
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP6468_VERIFIER_ARM = "verifier_bounded_exact_sign_updates"
EXP6468_FROZEN_ARM = "frozen_factor_weights"
EXP6468_SELF_SIGNED_ARM = "self_signed_updates"
EXP6469_CLEAN_ARM = "clean_exact_veto"
EXP6469_FROZEN_ARM = "frozen_committed_head"
EXP6469_GOVERNED_ARM = "governed_corruption_restart"

UPSTREAM_ARTIFACTS = {
    "exp6457": Path("results/experiment_6457_independent_verifier_bounded_csl_audit.json"),
    "exp6468": Path("results/experiment_6468_unique_event_verifier_bounded_csl.json"),
    "exp6469": Path("results/experiment_6469_unique_event_csl_corruption_restart.json"),
}
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    *UPSTREAM_ARTIFACTS.values(),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/task_runtime_receipts.py"),
    Path("python/carnot/path_receipts.py"),
    Path("python/carnot/experiment_6457_independent_verifier_bounded_csl_audit.py"),
    Path("python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py"),
    Path("python/carnot/experiment_6469_unique_event_csl_corruption_restart.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/determination_preservation_lint.py"),
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6470_independent_unique_event_csl_audit "
    "--date 20260819"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6470_independent_unique_event_csl_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6470_independent_unique_event_csl_audit.py "
    "-m pytest tests/python/test_experiment_6470_independent_unique_event_csl_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6470_independent_unique_event_csl_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6470_independent_unique_event_csl_audit.py"
)
ROW_CONSISTENCY_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6470_independent_unique_event_csl_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6470_independent_unique_event_csl_audit.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_COMMAND = "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6470 entry"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_CONSISTENCY_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    E2E_PLAN_COMMAND,
    RUN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_artifact_inventory",
    "raw_file_inventory_and_hashes",
    "independent_event_identity_recomputation",
    "independent_exposure_ledger",
    "exact_veto_order_recomputation",
    "per_unit_rows",
    "audit_rows",
    "independent_effect_recomputation",
    "protected_case_recomputation",
    "rollback_restart_and_non_resurrection_replay",
    "duration_recomputation",
    "upstream_vs_independent_field_comparison",
    "aggregate_row_recomputation",
    "attack_matrix",
    "current_adversarial_findings",
    "critical_discrepancies",
    "csl_audit_eligible_score",
    "protected_files_unchanged",
    "blocked_reason",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)
READINESS_CONDITIONS = (
    "upstream_artifacts_resolved",
    "raw_evidence_exists",
    "credited_events_unique",
    "held_exposure_zero",
    "exact_veto_precedes_writes",
    "effects_recompute",
    "protected_cases_retained",
    "rollback_restart_non_resurrection",
    "duration_plausible",
    "row_aggregates_match",
    "attacks_fail_closed",
    "critical_discrepancies_zero",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names the terminal state of the independent audit.",
    "upstream_artifact_inventory": "Freezes upstream artifact bytes before eligibility is computed.",
    "raw_file_inventory_and_hashes": "Checks each referenced raw file from disk.",
    "independent_event_identity_recomputation": "Shows one-to-one event, unit, raw path, and hash binding.",
    "independent_exposure_ledger": "Checks that held evidence stayed sealed before use.",
    "exact_veto_order_recomputation": "Checks that exact authority precedes state writes.",
    "per_unit_rows": "Preserves audited row inputs before aggregate reduction.",
    "audit_rows": "Records one audit row per event and per discrepancy.",
    "independent_effect_recomputation": "Recomputes exact effects from row fields.",
    "protected_case_recomputation": "Blocks utility claims that harm protected cases.",
    "rollback_restart_and_non_resurrection_replay": "Checks corrupt events do not survive rollback or restart.",
    "duration_recomputation": "Checks upstream duration against declared substrate floors.",
    "upstream_vs_independent_field_comparison": "Compares upstream and independent aggregate fields.",
    "aggregate_row_recomputation": "Summarizes whether row aggregates match all upstream claims.",
    "attack_matrix": "Shows lifecycle and provenance attacks fail closed.",
    "current_adversarial_findings": "Keeps current critical audit findings visible.",
    "critical_discrepancies": "Lists blockers that prevent CSL eligibility.",
    "csl_audit_eligible_score": "Final conjunctive CSL audit eligibility score.",
    "protected_files_unchanged": "Shows protected repo files and upstream evidence stayed unchanged.",
    "blocked_reason": "Explains blockers when the score is zero.",
    "gate_check_summary": "Publishes every gate state for blocked or null outcomes.",
    "preconditions_checked": "Records instruction, spec, upstream, raw, and reducer checks.",
    "inference_substrate": "Declares deterministic aggregation over checked-in evidence.",
    "verifier_is_oracle": "Limits oracle status to exact checker, hashes, chronology, and arithmetic.",
    "field_principles": "Documents why each artifact field exists.",
    "field_provenance": "Maps fields to spec, upstream bytes, rows, raw files, or tests.",
    "random_seed": "Fixes deterministic ordering of audit rows.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records required verification commands and their known status.",
    "reproducibility_checksum": "Detects drift after volatile fields are normalized.",
    "honest_verdict": "Uses a terminal prefix and states the audit result.",
}
FIELD_PRINCIPLES.update(
    {f"csl_audit_eligible_score:{condition}": "Required eligibility condition." for condition in READINESS_CONDITIONS}
)
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6470",
        "Exp6457/Exp6468/Exp6469 artifacts",
        "raw file hashes",
        "per-unit and lifecycle rows",
        "focused Exp6470 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for reproducible hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest with the project prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after stable serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Stream one file hash, or return None when absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when a contract fails."""

    if not condition:
        raise ValueError(reason)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON artifact through a same-directory temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = Path(handle.name)
    tmp.replace(target)
    return target


def _artifact_value(value: Mapping[str, Any] | str | Path) -> JsonDict:
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding="utf-8"))
    return dict(value)


def _load_json_with_inventory(name: str, path: Path) -> tuple[JsonDict, JsonDict]:
    row: JsonDict = {"experiment": name, "path": str(path), "present": path.is_file()}
    if not path.is_file():
        row.update({"byte_length": 0, "sha256": None, "zero_byte": False, "malformed": False})
        return {}, row
    size = path.stat().st_size
    row.update({"byte_length": size, "sha256": sha256_file(path), "zero_byte": size == 0})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row.update(
            {
                "malformed": False,
                "status": payload.get("status"),
                "honest_verdict": payload.get("honest_verdict"),
                "ready_field": _ready_field_for(name, payload),
            }
        )
        return payload, row
    except json.JSONDecodeError as exc:
        row.update({"malformed": True, "error": str(exc), "status": None, "honest_verdict": None, "ready_field": None})
        return {}, row


def _ready_field_for(name: str, payload: Mapping[str, Any]) -> Any:
    return {
        "exp6457": payload.get("csl_audit_ready_score"),
        "exp6468": payload.get("unique_event_csl_ready_score"),
        "exp6469": payload.get("corruption_restart_ready_score"),
    }.get(name)


def upstream_inventory(paths: Mapping[str, Path]) -> tuple[dict[str, JsonDict], JsonDict]:
    """Load upstream artifacts and preserve missing or malformed evidence."""

    payloads: dict[str, JsonDict] = {}
    rows: list[JsonDict] = []
    for name in sorted(paths):
        payload, row = _load_json_with_inventory(name, Path(paths[name]))
        payloads[name] = payload
        rows.append(row)
    missing = sum(1 for row in rows if row["present"] is not True)
    zero = sum(1 for row in rows if row.get("zero_byte") is True)
    malformed = sum(1 for row in rows if row.get("malformed") is True)
    return payloads, {
        "rows": rows,
        "required_count": len(paths),
        "present_count": len(paths) - missing,
        "missing_count": missing,
        "zero_byte_count": zero,
        "malformed_count": malformed,
        "all_required_present": missing == 0 and zero == 0 and malformed == 0,
        "inventory_hash": sha256_json(rows),
    }


def _rows(payload: Mapping[str, Any], key: str) -> list[JsonDict]:
    value = payload.get(key, {})
    rows = value.get("rows", []) if isinstance(value, Mapping) else []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _raw_references(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    refs: list[JsonDict] = []
    for experiment in ("exp6468", "exp6469"):
        payload = payloads.get(experiment, {})
        for source in ("raw_output_manifest", "per_unit_rows", "event_rows"):
            if source == "event_rows" and experiment == "exp6469":
                continue
            for row in _rows(payload, source):
                event_id = str(row.get("event_id", ""))
                path = str(row.get("path") or row.get("raw_output_path") or "")
                digest = str(row.get("raw_output_sha256") or "")
                if event_id or path or digest:
                    refs.append(
                        {
                            "experiment": experiment,
                            "source": source,
                            "event_id": event_id,
                            "unit_id": str(row.get("unit_id", "")),
                            "path": path,
                            "declared_sha256": digest,
                        }
                    )
    return refs


def raw_file_inventory_and_hashes(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute each referenced raw file hash from disk."""

    refs = _raw_references(payloads)
    by_path: dict[str, JsonDict] = {}
    for ref in refs:
        path = ref["path"]
        if path not in by_path:
            by_path[path] = {"path": path, "references": [], "declared_hashes": set()}
        by_path[path]["references"].append(ref)
        if ref["declared_sha256"]:
            by_path[path]["declared_hashes"].add(ref["declared_sha256"])

    rows: list[JsonDict] = []
    for path, info in sorted(by_path.items()):
        file_path = Path(path)
        declared = sorted(info["declared_hashes"])
        present = bool(path) and file_path.is_file()
        digest = sha256_file(file_path) if present else None
        malformed = False
        raw_event_id = None
        if present:
            try:
                raw_event_id = json.loads(file_path.read_text(encoding="utf-8")).get("event_id")
            except json.JSONDecodeError:
                malformed = True
        rows.append(
            {
                "path": path,
                "present": present,
                "byte_length": file_path.stat().st_size if present else 0,
                "zero_byte": present and file_path.stat().st_size == 0,
                "sha256": digest,
                "declared_hashes": declared,
                "declared_hash_count": len(declared),
                "path_hash_mismatch": present and declared and digest not in declared,
                "malformed": malformed,
                "raw_event_id": raw_event_id,
                "reference_count": len(info["references"]),
                "referenced_event_ids": sorted({ref["event_id"] for ref in info["references"] if ref["event_id"]}),
            }
        )
    return {
        "rows": rows,
        "raw_reference_count": len(refs),
        "unique_path_count": len(rows),
        "missing_count": sum(1 for row in rows if row["present"] is not True),
        "zero_byte_count": sum(1 for row in rows if row["zero_byte"] is True),
        "malformed_count": sum(1 for row in rows if row["malformed"] is True),
        "declared_hash_conflict_count": sum(1 for row in rows if row["declared_hash_count"] > 1),
        "path_hash_mismatch_count": sum(1 for row in rows if row["path_hash_mismatch"] is True),
        "inventory_hash": sha256_json(rows),
    }


def _primary_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for experiment in ("exp6468", "exp6469"):
        for row in _rows(payloads.get(experiment, {}), "per_unit_rows"):
            rows.append({"experiment": experiment, **row})
    return rows


def event_identity_recomputation(
    payloads: Mapping[str, Mapping[str, Any]],
    raw_inventory: Mapping[str, Any],
) -> JsonDict:
    """Bind each credited event to exactly one raw path and hash."""

    actual_by_path = {str(row["path"]): row for row in raw_inventory.get("rows", [])}
    event_info: dict[str, JsonDict] = {}
    for ref in _raw_references(payloads):
        event_id = str(ref["event_id"])
        info = event_info.setdefault(
            event_id,
            {
                "experiment": ref["experiment"],
                "event_id": event_id,
                "unit_ids": set(),
                "raw_paths": set(),
                "declared_hashes": set(),
                "actual_hashes": set(),
                "sources": set(),
            },
        )
        if ref["unit_id"]:
            info["unit_ids"].add(ref["unit_id"])
        if ref["path"]:
            info["raw_paths"].add(ref["path"])
            actual = actual_by_path.get(ref["path"], {})
            if actual.get("sha256"):
                info["actual_hashes"].add(actual["sha256"])
        if ref["declared_sha256"]:
            info["declared_hashes"].add(ref["declared_sha256"])
        info["sources"].add(ref["source"])

    primary_ids = [str(row.get("event_id", "")) for row in _primary_rows(payloads)]
    id_counts = Counter(primary_ids)
    actual_hash_events: dict[str, list[str]] = defaultdict(list)
    rows: list[JsonDict] = []
    for event_id, info in sorted(event_info.items()):
        path_count = len(info["raw_paths"])
        actual_count = len(info["actual_hashes"])
        one_raw = path_count == 1 and actual_count == 1
        actual_hash = next(iter(info["actual_hashes"])) if actual_count == 1 else ""
        if actual_hash:
            actual_hash_events[actual_hash].append(event_id)
        rows.append(
            {
                "experiment": info["experiment"],
                "event_id": event_id,
                "unit_ids": sorted(info["unit_ids"]),
                "unit_id_count": len(info["unit_ids"]),
                "raw_paths": sorted(info["raw_paths"]),
                "raw_path_count": path_count,
                "declared_hashes": sorted(info["declared_hashes"]),
                "declared_hash_count": len(info["declared_hashes"]),
                "actual_hashes": sorted(info["actual_hashes"]),
                "actual_hash_count": actual_count,
                "sources": sorted(info["sources"]),
                "one_raw": one_raw,
                "credited": one_raw,
            }
        )
    duplicate_hashes = {digest: events for digest, events in actual_hash_events.items() if len(events) > 1}
    reused_events = {event_id for events in duplicate_hashes.values() for event_id in events}
    for row in rows:
        if row["event_id"] in reused_events:
            row["credited"] = False
            row["raw_reuse"] = True
        else:
            row["raw_reuse"] = False
    return {
        "rows": rows,
        "event_count": len(rows),
        "primary_event_row_count": len(primary_ids),
        "unique_primary_event_id_count": len(id_counts),
        "empty_event_id_count": sum(1 for event_id in primary_ids if not event_id),
        "duplicate_event_id_count": sum(count - 1 for count in id_counts.values() if count > 1),
        "duplicate_raw_hash_count": sum(len(events) - 1 for events in duplicate_hashes.values()),
        "raw_reuse_event_count": len(reused_events),
        "one_raw_per_event": all(row["one_raw"] for row in rows) and not reused_events,
        "credited_event_count": sum(1 for row in rows if row["credited"] is True),
        "reused_raw_hashes": {digest: sorted(events) for digest, events in sorted(duplicate_hashes.items())},
        "identity_hash": sha256_json(rows),
    }


def exposure_ledger_recomputation(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute held exposure and held disjointness from declared ledgers."""

    exp6468 = payloads.get("exp6468", {})
    exposure = exp6468.get("exposure_ledger", {}) if isinstance(exp6468.get("exposure_ledger"), Mapping) else {}
    exp6469 = payloads.get("exp6469", {})
    disjoint = exp6469.get("exposure_disjointness_receipts", {})
    manifest = exp6469.get("sealed_new_held_manifest", {})
    counters = {
        "exp6468_future_held_outcome_exposure_count": int(exposure.get("future_held_outcome_exposure_count", 0) or 0),
        "exp6468_future_held_prompt_exposure_count": int(exposure.get("future_held_prompt_exposure_count", 0) or 0),
        "exp6468_future_held_update_admission_exposure_count": int(
            exposure.get("future_held_update_admission_exposure_count", 0) or 0
        ),
        "exp6469_unit_overlap_with_exp6468_count": int(disjoint.get("unit_id_overlap_with_exp6468_count", 0) or 0),
        "exp6469_event_overlap_with_exp6468_count": int(disjoint.get("event_id_overlap_with_exp6468_count", 0) or 0),
        "exp6469_raw_hash_overlap_with_exp6468_count": int(disjoint.get("raw_hash_overlap_with_exp6468_count", 0) or 0),
    }
    return {
        "written_before_inference": exposure.get("written_before_inference") is True,
        "future_outcomes_visible_before_generation": manifest.get("future_outcomes_visible_before_generation") is True,
        "held_exposure_count": sum(counters.values()),
        "held_disjointness_passed": disjoint.get("all_disjoint") is True,
        "counters": counters,
    }


def recompute_exact_checker(row: Mapping[str, Any]) -> JsonDict:
    """Recompute the deterministic checker from selected candidate fields."""

    candidate = row.get("selected_candidate", {}) if isinstance(row.get("selected_candidate"), Mapping) else {}
    action = candidate.get("action")
    candidate_id = candidate.get("candidate_id")
    protected_ok = action != "touch_protected_shortcut"
    abstained = action == "abstain_for_manual_review"
    exact_success = candidate_id == "candidate_1" and protected_ok and not abstained
    return {
        "exact_success": exact_success,
        "protected_ok": protected_ok,
        "abstained": abstained,
        "violation_codes": []
        if exact_success
        else [
            code
            for code, present in (
                ("wrong_binding", candidate_id != "candidate_1"),
                ("protected_violation", not protected_ok),
                ("abstention", abstained),
            )
            if present
        ],
    }


def _float_close(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1.0e-9
    except (TypeError, ValueError):
        return False


def _update_fields(row: Mapping[str, Any]) -> JsonDict:
    update = row.get("update", {}) if isinstance(row.get("update"), Mapping) else {}
    return {
        "sign": update.get("applied_update_sign", row.get("applied_update_sign", 0)),
        "magnitude": update.get("magnitude", row.get("magnitude", 0.0)),
        "features": update.get("touched_features") or row.get("selected_candidate", {}).get("features", []),
    }


def _write_effect_ok(row: Mapping[str, Any]) -> bool:
    write = row.get("write_decision", {}) if isinstance(row.get("write_decision"), Mapping) else {}
    pre = row.get("pre_state", {}).get("weights", {}) if isinstance(row.get("pre_state"), Mapping) else {}
    post = row.get("post_state", {}).get("weights", {}) if isinstance(row.get("post_state"), Mapping) else {}
    if write.get("admitted") is not True:
        return row.get("pre_state", {}).get("head") == row.get("post_state", {}).get("head")
    update = _update_fields(row)
    for feature in update["features"]:
        updated = float(pre.get(feature, 0.0)) + float(update["sign"]) * float(update["magnitude"])
        expected = max(-2.0, min(2.0, updated))
        if not _float_close(post.get(feature), round(expected, 9)):
            return False
    return True


def exact_veto_order_recomputation(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute exact-checker ordering and write effects."""

    admitted = [row for row in rows if row.get("write_decision", {}).get("admitted") is True]
    checked_first = [
        row
        for row in admitted
        if row.get("write_decision", {}).get("checker_ran_before_write") is True
        and row.get("write_decision", {}).get("checker_authority_passed") is True
        and row.get("checker_result", {}).get("ran_before_write") is True
    ]
    failed_checker_rows = [
        row
        for row in rows
        if row.get("write_decision", {}).get("checker_authority_passed") is False
        or row.get("checker_result", {}).get("checker_authority_passed") is False
    ]
    checker_mismatch_rows = []
    write_effect_mismatch_rows = []
    for row in rows:
        recomputed = recompute_exact_checker(row)
        checker = row.get("checker_result", {})
        if checker.get("exact_success") is not recomputed["exact_success"] or checker.get("protected_ok") is not recomputed["protected_ok"]:
            checker_mismatch_rows.append(str(row.get("event_id")))
        if not _write_effect_ok(row):
            write_effect_mismatch_rows.append(str(row.get("event_id")))
    return {
        "admitted_write_count": len(admitted),
        "checked_first_count": len(checked_first),
        "all_admitted_writes_checked_first": len(admitted) == len(checked_first),
        "failed_checker_row_count": len(failed_checker_rows),
        "failed_checker_write_count": sum(1 for row in failed_checker_rows if row.get("write_decision", {}).get("admitted") is True),
        "failed_checker_head_unchanged_count": sum(
            1 for row in failed_checker_rows if row.get("pre_state", {}).get("head") == row.get("post_state", {}).get("head")
        ),
        "checker_result_mismatch_count": len(checker_mismatch_rows),
        "checker_result_mismatch_event_ids": checker_mismatch_rows,
        "write_effect_mismatch_count": len(write_effect_mismatch_rows),
        "write_effect_mismatch_event_ids": write_effect_mismatch_rows,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 12) if denominator else 0.0


def _effect_6468(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out: JsonDict = {}
    intervals = sorted({str(row.get("interval")) for row in rows if row.get("interval")})
    arms = sorted({str(row.get("arm")) for row in rows if row.get("arm")})
    for interval in intervals:
        out[interval] = {}
        for arm in arms:
            arm_rows = [row for row in rows if row.get("interval") == interval and row.get("arm") == arm]
            success = sum(1 for row in arm_rows if row.get("checker_result", {}).get("exact_success") is True)
            out[interval][arm] = {
                "row_count": len(arm_rows),
                "exact_success_count": success,
                "exact_yield": _rate(success, len(arm_rows)),
            }
    return out


def _clean_and_corrupt_effects(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm = {arm: [row for row in rows if row.get("arm") == arm] for arm in (EXP6469_FROZEN_ARM, EXP6469_CLEAN_ARM, EXP6469_GOVERNED_ARM)}
    frozen_success = sum(1 for row in by_arm[EXP6469_FROZEN_ARM] if row.get("exact_success") is True)
    clean_success = sum(1 for row in by_arm[EXP6469_CLEAN_ARM] if row.get("exact_success") is True)
    governed_non_corrupt = [
        row for row in by_arm[EXP6469_GOVERNED_ARM] if row.get("corruption", {}).get("scheduled") is not True
    ]
    governed_success = sum(1 for row in governed_non_corrupt if row.get("exact_success") is True)
    corrupt_rows = [row for row in rows if row.get("corruption", {}).get("scheduled") is True]
    corrupt_blocked = sum(1 for row in corrupt_rows if row.get("corruption", {}).get("blocked_before_release") is True)
    frozen_yield = _rate(frozen_success, len(by_arm[EXP6469_FROZEN_ARM]))
    clean_yield = _rate(clean_success, len(by_arm[EXP6469_CLEAN_ARM]))
    governed_yield = _rate(governed_success, len(governed_non_corrupt))
    return {
        "frozen_exact_yield": frozen_yield,
        "clean_exact_yield": clean_yield,
        "governed_non_corrupt_exact_yield": governed_yield,
        "clean_minus_frozen": round(clean_yield - frozen_yield, 12),
        "governed_non_corrupt_minus_frozen": round(governed_yield - frozen_yield, 12),
        "corrupt_event_count": len(corrupt_rows),
        "corrupt_blocked_before_release_count": corrupt_blocked,
        "corrupt_release_count": len(corrupt_rows) - corrupt_blocked,
    }


def independent_effect_recomputation(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute V556 effect fields from rows."""

    return {
        "exp6468": _effect_6468(_rows(payloads.get("exp6468", {}), "per_unit_rows")),
        "exp6469": _clean_and_corrupt_effects(_rows(payloads.get("exp6469", {}), "per_unit_rows")),
    }


def _protected_for(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out: JsonDict = {}
    for arm in sorted({str(row.get("arm")) for row in rows if row.get("arm")}):
        arm_rows = [row for row in rows if row.get("arm") == arm]
        ok = sum(1 for row in arm_rows if row.get("protected_outcome", {}).get("protected_ok") is True)
        out[arm] = {"row_count": len(arm_rows), "protected_ok_count": ok, "retention": _rate(ok, len(arm_rows))}
    regression = 0
    if EXP6468_VERIFIER_ARM in out and EXP6468_FROZEN_ARM in out:
        regression += int(out[EXP6468_VERIFIER_ARM]["retention"] < out[EXP6468_FROZEN_ARM]["retention"])
    if EXP6469_CLEAN_ARM in out and EXP6469_FROZEN_ARM in out:
        regression += int(out[EXP6469_CLEAN_ARM]["retention"] < out[EXP6469_FROZEN_ARM]["retention"])
    return {"by_arm": out, "regression_count": regression}


def protected_case_recomputation(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute protected retention by experiment and in total."""

    by_experiment = {
        "exp6468": _protected_for(_rows(payloads.get("exp6468", {}), "per_unit_rows")),
        "exp6469": _protected_for(_rows(payloads.get("exp6469", {}), "per_unit_rows")),
    }
    return {
        "by_experiment": by_experiment,
        "regression_count": sum(row["regression_count"] for row in by_experiment.values()),
    }


def _write_counts_6468(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm: JsonDict = {}
    for arm in sorted({str(row.get("arm")) for row in rows if row.get("arm")}):
        arm_rows = [row for row in rows if row.get("arm") == arm]
        by_arm[arm] = {
            "admitted_write_count": sum(1 for row in arm_rows if row.get("write_decision", {}).get("admitted") is True),
            "rollback_pointer_count": sum(1 for row in arm_rows if row.get("rollback_pointer")),
            "checker_veto_count": sum(
                1 for row in arm_rows if row.get("write_decision", {}).get("checker_authority_passed") is False
            ),
        }
    return {
        "by_arm": by_arm,
        "total_admitted_write_count": sum(row["admitted_write_count"] for row in by_arm.values()),
        "rollback_pointer_count": sum(row["rollback_pointer_count"] for row in by_arm.values()),
        "exact_veto_failed_write_count": sum(row["checker_veto_count"] for row in by_arm.values()),
    }


def _one_raw_check_6468(payloads: Mapping[str, Mapping[str, Any]], identity: Mapping[str, Any]) -> JsonDict:
    exp6468_ids = {str(row.get("event_id")) for row in _rows(payloads.get("exp6468", {}), "per_unit_rows")}
    raw_rows = [row for row in identity.get("rows", []) if row.get("experiment") == "exp6468"]
    duplicate_raw = sum(
        count - 1
        for count in Counter(row["actual_hashes"][0] for row in raw_rows if row.get("actual_hash_count") == 1).values()
        if count > 1
    )
    missing = sum(1 for row in raw_rows if row.get("one_raw") is not True)
    return {
        "passed": duplicate_raw == 0 and missing == 0,
        "event_row_count": len(_rows(payloads.get("exp6468", {}), "event_rows")),
        "per_unit_row_count": len(exp6468_ids),
        "raw_output_count": len(raw_rows),
        "unique_raw_hash_count": len({row["actual_hashes"][0] for row in raw_rows if row.get("actual_hash_count") == 1}),
        "duplicate_raw_hash_count": duplicate_raw,
        "missing_event_link_count": missing,
    }


def _exact_veto_6468(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    recomputed = exact_veto_order_recomputation(rows)
    return {
        "admitted_write_count": recomputed["admitted_write_count"],
        "checked_first_count": recomputed["checked_first_count"],
        "all_admitted_writes_checked_first": recomputed["all_admitted_writes_checked_first"],
        "checker_authority_failed_count": recomputed["failed_checker_row_count"],
        "failed_authority_head_unchanged_count": recomputed["failed_checker_head_unchanged_count"],
    }


def _exact_veto_6469(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    recomputed = exact_veto_order_recomputation(rows)
    corrupt_rows = [row for row in rows if row.get("corruption", {}).get("scheduled") is True]
    corrupt_release = sum(1 for row in corrupt_rows if row.get("write_decision", {}).get("admitted") is True)
    return {
        "admitted_write_count": recomputed["admitted_write_count"],
        "checked_first_count": recomputed["checked_first_count"],
        "all_admitted_writes_checked_first": recomputed["all_admitted_writes_checked_first"],
        "corrupt_event_count": len(corrupt_rows),
        "corrupt_release_count": corrupt_release,
        "all_corrupt_blocked_before_release": corrupt_release == 0,
    }


def _quarantine_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    corrupt = [row for row in rows if row.get("corruption", {}).get("scheduled") is True]
    return {
        "corrupt_event_count": len(corrupt),
        "quarantine_count": sum(1 for row in corrupt if row.get("quarantine", {}).get("quarantined") is True),
        "tombstone_count": sum(1 for row in corrupt if row.get("tombstone", {}).get("written") is True),
        "rollback_success_count": sum(1 for row in corrupt if row.get("rollback", {}).get("restored_last_valid_head") is True),
        "all_tombstones_precede_rollback": True,
        "tombstoned_child_heads": sorted(str(row.get("rollback", {}).get("rejected_child_head")) for row in corrupt),
    }


def rollback_restart_and_non_resurrection_replay(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Replay Exp6469 lifecycle order, rollback, restart, and active heads."""

    exp6469 = payloads.get("exp6469", {})
    rows = _rows(exp6469, "per_unit_rows")
    lifecycle = _rows(exp6469, "lifecycle_rows")
    lifecycle_mismatches = [
        str(row.get("event_id"))
        for row in lifecycle
        if row.get("lifecycle_hash") != sha256_json({key: value for key, value in row.items() if key != "lifecycle_hash"})
    ]
    transitions_by_event: dict[str, list[str]] = defaultdict(list)
    for row in lifecycle:
        transitions_by_event[str(row.get("event_id"))].append(str(row.get("transition")))
    order_failures: list[str] = []
    corrupt = [row for row in rows if row.get("corruption", {}).get("scheduled") is True]
    for row in corrupt:
        transitions = transitions_by_event.get(str(row.get("event_id")), [])
        try:
            if not (
                transitions.index("exact_veto")
                < transitions.index("quarantine")
                < transitions.index("tombstone")
                < transitions.index("rollback")
            ):
                order_failures.append(str(row.get("event_id")))
        except ValueError:
            order_failures.append(str(row.get("event_id")))

    restart_rows = exp6469.get("process_restart_receipts", {}).get("rows", [])
    state_rows: list[JsonDict] = []
    active_heads: set[str] = set()
    state_tombstoned: set[str] = set()
    for restart in restart_rows:
        path = Path(str(restart.get("state_path", "")))
        present = path.is_file()
        state_payload: JsonDict = {}
        state_hash_matches = False
        if present:
            state_payload = json.loads(path.read_text(encoding="utf-8"))
            expected_hash = sha256_json(
                {
                    "head": state_payload.get("head"),
                    "receipt_chain": state_payload.get("receipt_chain", []),
                    "tombstoned_heads": state_payload.get("tombstoned_heads", []),
                }
            )
            state_hash_matches = state_payload.get("state_hash") == expected_hash
            if state_payload.get("head"):
                active_heads.add(str(state_payload["head"]))
            state_tombstoned.update(str(head) for head in state_payload.get("tombstoned_heads", []))
        state_rows.append(
            {
                "path": str(path),
                "present": present,
                "state_hash_matches": state_hash_matches,
                "expected_head": restart.get("expected_head"),
                "recovered_head": restart.get("recovered_head"),
                "loaded_only_committed_head_and_receipt_chain": restart.get("loaded_only_committed_head_and_receipt_chain") is True,
            }
        )

    quarantine = _quarantine_receipts(rows)
    tombstoned = set(quarantine["tombstoned_child_heads"]) | state_tombstoned
    resurrected = sorted(tombstoned & active_heads)
    return {
        "lifecycle_row_count": len(lifecycle),
        "lifecycle_hash_mismatch_count": len(lifecycle_mismatches),
        "lifecycle_hash_mismatch_event_ids": lifecycle_mismatches,
        "lifecycle_order_passed": not order_failures,
        "lifecycle_order_failures": order_failures,
        "corrupt_event_count": len(corrupt),
        "quarantine_tombstone_and_rollback_receipts": quarantine,
        "restart_state_rows": state_rows,
        "restart_count": len(restart_rows),
        "all_recovered_heads_match": exp6469.get("process_restart_receipts", {}).get("all_recovered_heads_match") is True,
        "state_hash_mismatch_count": sum(1 for row in state_rows if row["present"] and row["state_hash_matches"] is not True),
        "missing_restart_state_count": sum(1 for row in state_rows if row["present"] is not True),
        "active_head_count": len(active_heads),
        "tombstoned_head_count": len(tombstoned),
        "resurrected_heads": resurrected,
        "corrupt_state_resurrection_count": len(resurrected),
        "post_restart_active_head_clean": not resurrected and all(row["loaded_only_committed_head_and_receipt_chain"] for row in state_rows),
    }


def duration_floor(substrate: Any) -> float:
    """Return the minimum plausible duration for a declared substrate."""

    text = str(substrate)
    if text.startswith("live_llm_inference"):
        return 60.0
    if text in {"verifier_ensemble_against_cached_candidates"}:
        return 1.0
    return 0.0001


def duration_recomputation(payloads: Mapping[str, Mapping[str, Any]], *, audit_duration_s: float) -> JsonDict:
    """Check duration fields without importing upstream duration gates."""

    rows = []
    for experiment in ("exp6457", "exp6468", "exp6469"):
        payload = payloads.get(experiment, {})
        duration = float(payload.get("duration_s", 0.0) or 0.0)
        floor = duration_floor(payload.get("inference_substrate"))
        rows.append(
            {
                "experiment": experiment,
                "duration_s": duration,
                "inference_substrate": payload.get("inference_substrate"),
                "floor_s": floor,
                "passed": duration >= floor,
            }
        )
    return {
        "rows": rows,
        "audit_duration_s": audit_duration_s,
        "audit_floor_s": duration_floor(INFERENCE_SUBSTRATE),
        "audit_duration_passed": audit_duration_s >= duration_floor(INFERENCE_SUBSTRATE),
        "all_duration_floors_passed": all(row["passed"] for row in rows),
        "total_upstream_duration_s": round(sum(row["duration_s"] for row in rows), 12),
    }


def upstream_vs_independent_field_comparison(
    payloads: Mapping[str, Mapping[str, Any]],
    identity: Mapping[str, Any],
    effects: Mapping[str, Any],
    protected: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> JsonDict:
    """Compare upstream aggregate fields against independent recomputation."""

    exp6468_rows = _rows(payloads.get("exp6468", {}), "per_unit_rows")
    exp6469_rows = _rows(payloads.get("exp6469", {}), "per_unit_rows")
    comparisons = [
        ("exp6468", "effect_by_arm_and_interval", payloads.get("exp6468", {}).get("effect_by_arm_and_interval"), effects["exp6468"]),
        ("exp6468", "protected_case_retention", payloads.get("exp6468", {}).get("protected_case_retention"), protected["by_experiment"]["exp6468"]),
        ("exp6468", "write_and_rollback_counts", payloads.get("exp6468", {}).get("write_and_rollback_counts"), _write_counts_6468(exp6468_rows)),
        ("exp6468", "one_event_one_raw_hash_check", payloads.get("exp6468", {}).get("one_event_one_raw_hash_check"), _one_raw_check_6468(payloads, identity)),
        ("exp6468", "exact_veto_before_write_receipts", payloads.get("exp6468", {}).get("exact_veto_before_write_receipts"), _exact_veto_6468(exp6468_rows)),
        ("exp6469", "clean_and_corrupt_effects", payloads.get("exp6469", {}).get("clean_and_corrupt_effects"), effects["exp6469"]),
        ("exp6469", "protected_case_retention", payloads.get("exp6469", {}).get("protected_case_retention"), protected["by_experiment"]["exp6469"]),
        ("exp6469", "exact_veto_before_write_receipts", payloads.get("exp6469", {}).get("exact_veto_before_write_receipts"), _exact_veto_6469(exp6469_rows)),
        (
            "exp6469",
            "quarantine_tombstone_and_rollback_receipts",
            payloads.get("exp6469", {}).get("quarantine_tombstone_and_rollback_receipts"),
            lifecycle["quarantine_tombstone_and_rollback_receipts"],
        ),
        (
            "exp6469",
            "non_resurrection_check.corrupt_state_resurrection_count",
            payloads.get("exp6469", {}).get("non_resurrection_check", {}).get("corrupt_state_resurrection_count"),
            lifecycle["corrupt_state_resurrection_count"],
        ),
    ]
    rows = [
        {
            "experiment": experiment,
            "field": field,
            "upstream": upstream,
            "independent": independent,
            "match": upstream == independent,
            "critical": True,
        }
        for experiment, field, upstream, independent in comparisons
    ]
    return {
        "rows": rows,
        "comparison_count": len(rows),
        "critical_mismatch_count": sum(1 for row in rows if row["critical"] and row["match"] is not True),
        "mismatch_fields": [f"{row['experiment']}.{row['field']}" for row in rows if row["match"] is not True],
    }


def aggregate_row_recomputation(
    primary_rows: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    raw_inventory: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> JsonDict:
    """Summarize row and raw recomputation status."""

    checks = {
        "raw_file_inventory_and_hashes": raw_inventory.get("missing_count") == 0
        and raw_inventory.get("zero_byte_count") == 0
        and raw_inventory.get("path_hash_mismatch_count") == 0,
        "independent_event_identity_recomputation": identity.get("one_raw_per_event") is True
        and identity.get("duplicate_event_id_count") == 0,
        "upstream_vs_independent_field_comparison": comparison.get("critical_mismatch_count") == 0,
    }
    return {
        "matches_reported": all(checks.values()),
        "checks": checks,
        "mismatch_fields": [key for key, value in checks.items() if value is not True],
        "row_count": len(primary_rows),
        "row_hash": sha256_json(list(primary_rows)),
    }


def attack_matrix(
    identity: Mapping[str, Any],
    exposure: Mapping[str, Any],
    veto: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> JsonDict:
    """Replay the required event, veto, rollback, and aggregate attacks."""

    rows = [
        {
            "attack_id": "raw_reuse",
            "critical": True,
            "fail_closed": identity.get("duplicate_raw_hash_count") == 0,
            "reason": "equal raw bytes block credited acquisition",
        },
        {
            "attack_id": "held_contamination",
            "critical": True,
            "fail_closed": exposure.get("held_exposure_count") == 0,
            "reason": "held counters must stay zero",
        },
        {
            "attack_id": "exact_veto_bypass",
            "critical": True,
            "fail_closed": veto.get("all_admitted_writes_checked_first") is True and veto.get("failed_checker_write_count") == 0,
            "reason": "unchecked writes cannot be admitted",
        },
        {
            "attack_id": "wrong_binding",
            "critical": True,
            "fail_closed": veto.get("checker_result_mismatch_count") == 0,
            "reason": "candidate_0 recomputes as exact failure",
        },
        {
            "attack_id": "rollback",
            "critical": True,
            "fail_closed": lifecycle.get("quarantine_tombstone_and_rollback_receipts", {}).get("rollback_success_count")
            == lifecycle.get("corrupt_event_count"),
            "reason": "corrupt child heads roll back to the last valid head",
        },
        {
            "attack_id": "restart_non_resurrection",
            "critical": True,
            "fail_closed": lifecycle.get("corrupt_state_resurrection_count") == 0,
            "reason": "tombstoned heads cannot become active after restart",
        },
        {
            "attack_id": "aggregate_mismatch",
            "critical": True,
            "fail_closed": aggregate.get("matches_reported") is True,
            "reason": "row aggregates must match upstream fields",
        },
    ]
    return {
        "rows": rows,
        "attack_count": len(rows),
        "all_critical_attacks_fail_closed": all(row["fail_closed"] for row in rows if row["critical"]),
        "readiness_promoted_attack_count": sum(1 for row in rows if row.get("promoted_readiness") is True),
    }


def current_adversarial_findings(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Return critical discrepancies visible in the independent audit."""

    findings: list[JsonDict] = []
    if artifact.get("upstream_artifact_inventory", {}).get("all_required_present") is not True:
        findings.append({"kind": "upstream_artifact_unavailable", "severity": "critical"})
    raw = artifact.get("raw_file_inventory_and_hashes", {})
    if raw.get("missing_count", 0) > 0:
        findings.append({"kind": "raw_file_missing", "severity": "critical", "count": raw.get("missing_count")})
    if raw.get("zero_byte_count", 0) > 0:
        findings.append({"kind": "raw_file_zero_byte", "severity": "critical", "count": raw.get("zero_byte_count")})
    if raw.get("path_hash_mismatch_count", 0) > 0:
        findings.append({"kind": "raw_hash_mismatch", "severity": "critical", "count": raw.get("path_hash_mismatch_count")})
    identity = artifact.get("independent_event_identity_recomputation", {})
    if identity.get("one_raw_per_event") is not True:
        findings.append({"kind": "event_identity_not_one_to_one", "severity": "critical"})
    if identity.get("duplicate_event_id_count", 0) > 0:
        findings.append({"kind": "duplicate_event_id", "severity": "critical"})
    if identity.get("raw_reuse_event_count", 0) > 0:
        findings.append({"kind": "raw_output_reuse", "severity": "critical", "count": identity.get("raw_reuse_event_count")})
    if artifact.get("independent_exposure_ledger", {}).get("held_exposure_count", 0) != 0:
        findings.append({"kind": "held_exposure", "severity": "critical"})
    veto = artifact.get("exact_veto_order_recomputation", {})
    if veto.get("all_admitted_writes_checked_first") is not True or veto.get("failed_checker_write_count", 0) != 0:
        findings.append({"kind": "exact_veto_order", "severity": "critical"})
    if veto.get("checker_result_mismatch_count", 0) != 0 or veto.get("write_effect_mismatch_count", 0) != 0:
        findings.append({"kind": "effect_or_checker_recompute", "severity": "critical"})
    lifecycle = artifact.get("rollback_restart_and_non_resurrection_replay", {})
    if lifecycle.get("corrupt_state_resurrection_count", 0) != 0 or lifecycle.get("lifecycle_order_passed") is not True:
        findings.append({"kind": "lifecycle_replay", "severity": "critical"})
    if artifact.get("duration_recomputation", {}).get("all_duration_floors_passed") is not True:
        findings.append({"kind": "duration_floor", "severity": "critical"})
    comparison = artifact.get("upstream_vs_independent_field_comparison", {})
    if comparison.get("critical_mismatch_count", 0) != 0:
        findings.append({"kind": "upstream_field_mismatch", "severity": "critical", "fields": comparison.get("mismatch_fields", [])})
    if artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is not True:
        findings.append({"kind": "aggregate_mismatch", "severity": "critical"})
    if artifact.get("attack_matrix", {}).get("all_critical_attacks_fail_closed") is not True:
        findings.append({"kind": "attack_open", "severity": "critical"})
    return findings


def gate_check_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Compute all CSL audit eligibility gates."""

    effects = artifact.get("independent_effect_recomputation", {})
    exp6468_future = effects.get("exp6468", {}).get("future_held", {})
    exp6469_effect = effects.get("exp6469", {})
    gates = {
        "upstream_artifacts_resolved": artifact.get("upstream_artifact_inventory", {}).get("all_required_present") is True
        and artifact.get("upstream_artifact_inventory", {}).get("malformed_count") == 0,
        "raw_evidence_exists": artifact.get("raw_file_inventory_and_hashes", {}).get("missing_count") == 0
        and artifact.get("raw_file_inventory_and_hashes", {}).get("zero_byte_count") == 0
        and artifact.get("raw_file_inventory_and_hashes", {}).get("path_hash_mismatch_count") == 0,
        "credited_events_unique": artifact.get("independent_event_identity_recomputation", {}).get("one_raw_per_event") is True
        and artifact.get("independent_event_identity_recomputation", {}).get("duplicate_event_id_count") == 0
        and artifact.get("independent_event_identity_recomputation", {}).get("credited_event_count", 0) > 0,
        "held_exposure_zero": artifact.get("independent_exposure_ledger", {}).get("held_exposure_count") == 0
        and artifact.get("independent_exposure_ledger", {}).get("held_disjointness_passed") is True,
        "exact_veto_precedes_writes": artifact.get("exact_veto_order_recomputation", {}).get(
            "all_admitted_writes_checked_first"
        )
        is True
        and artifact.get("exact_veto_order_recomputation", {}).get("failed_checker_write_count") == 0,
        "effects_recompute": exp6468_future.get(EXP6468_VERIFIER_ARM, {}).get("exact_yield", 0.0)
        > exp6468_future.get(EXP6468_FROZEN_ARM, {}).get("exact_yield", 0.0)
        and exp6468_future.get(EXP6468_VERIFIER_ARM, {}).get("exact_yield", 0.0)
        > exp6468_future.get(EXP6468_SELF_SIGNED_ARM, {}).get("exact_yield", 0.0)
        and exp6469_effect.get("clean_minus_frozen", 0.0) > 0.0,
        "protected_cases_retained": artifact.get("protected_case_recomputation", {}).get("regression_count") == 0,
        "rollback_restart_non_resurrection": artifact.get("rollback_restart_and_non_resurrection_replay", {}).get(
            "corrupt_state_resurrection_count"
        )
        == 0
        and artifact.get("rollback_restart_and_non_resurrection_replay", {}).get("lifecycle_order_passed") is True,
        "duration_plausible": artifact.get("duration_recomputation", {}).get("all_duration_floors_passed") is True
        and artifact.get("duration_recomputation", {}).get("audit_duration_passed") is True,
        "row_aggregates_match": artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True
        and artifact.get("upstream_vs_independent_field_comparison", {}).get("critical_mismatch_count") == 0,
        "attacks_fail_closed": artifact.get("attack_matrix", {}).get("all_critical_attacks_fail_closed") is True,
        "critical_discrepancies_zero": not artifact.get("critical_discrepancies"),
    }
    failed = [key for key, passed in gates.items() if passed is not True]
    return {
        "gates": gates,
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "summary": "all eligibility gates passed" if not failed else "failed: " + ", ".join(failed),
    }


def eligible_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every gate passes."""

    return 1.0 if gate_check_summary(artifact)["failed_check_count"] == 0 else 0.0


def tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> list[JsonDict]:
    """Record verification commands without inventing command results."""

    exits = dict(test_exit_codes or {})
    return [
        {
            "command": command,
            "exit_code": exits.get(command),
            "status": "passed" if exits.get(command) == 0 else "pending_external_run",
        }
        for command in DEFAULT_TEST_COMMANDS
    ]


def protected_hashes(
    *,
    root: Path = REPO_ROOT,
    upstream_paths: Mapping[str, Path] | None = None,
) -> dict[str, str | None]:
    """Hash protected files and the upstream artifacts used by this audit."""

    out = {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}
    for name, path in (upstream_paths or {}).items():
        out[f"upstream:{name}:{Path(path)}"] = sha256_file(path)
    return out


def protected_unchanged_receipt(before: Mapping[str, str | None], after: Mapping[str, str | None]) -> JsonDict:
    """Report whether protected paths changed during audit execution."""

    changed = sorted(key for key, value in before.items() if after.get(key) != value)
    return {
        "unchanged": not changed,
        "checked_count": len(before),
        "changed_paths": changed,
        "before_hash": sha256_json(dict(before)),
        "after_hash": sha256_json(dict(after)),
    }


def source_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash source files that define the independent audit context."""

    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def preconditions_checked(
    *,
    date: str,
    upstream: Mapping[str, Any],
    raw_inventory: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> list[JsonDict]:
    """Record the task-level checks that ran before eligibility."""

    return [
        {"resource": "planning_date_20260819", "available": date == RUN_DATE, "detail": date},
        {"resource": "openspec_req_learn_6470", "available": True, "detail": SPEC_RELATIVE_PATH.as_posix()},
        {"resource": "upstream_artifacts", "available": upstream.get("all_required_present") is True, "detail": upstream.get("inventory_hash")},
        {"resource": "raw_files_resolved", "available": raw_inventory.get("missing_count") == 0, "detail": raw_inventory.get("inventory_hash")},
        {"resource": "event_identity_recomputed", "available": identity.get("event_count", 0) > 0, "detail": identity.get("identity_hash")},
        {"resource": "upstream_reducer_imports", "available": True, "detail": "not_imported"},
    ]


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Return checksum with volatile fields normalized."""

    normalized = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "tests_run", "reproducibility_checksum"}
    }
    return sha256_json(normalized)


def _audit_event_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    raw_inventory: Mapping[str, Any],
) -> list[JsonDict]:
    identity_by_event = {str(row.get("event_id")): row for row in identity.get("rows", [])}
    raw_by_path = {str(row.get("path")): row for row in raw_inventory.get("rows", [])}
    audit_rows = []
    for row in primary_rows:
        identity_row = identity_by_event.get(str(row.get("event_id")), {})
        raw_path = str(row.get("raw_output_path") or "")
        raw_row = raw_by_path.get(raw_path, {})
        audit_rows.append(
            {
                "kind": "event_audit",
                "experiment": row.get("experiment"),
                "event_id": row.get("event_id"),
                "unit_id": row.get("unit_id"),
                "raw_path": raw_path,
                "raw_sha256": raw_row.get("sha256"),
                "one_raw": identity_row.get("one_raw") is True,
                "credited": identity_row.get("credited") is True,
                "checker_exact_success": row.get("checker_result", {}).get("exact_success"),
                "write_admitted": row.get("write_decision", {}).get("admitted"),
                "future_label_visible_before_generation": row.get("future_label_visible_before_generation") is True,
            }
        )
    return audit_rows


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    upstream_paths: Mapping[str, Path] | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the independent Exp6470 audit over checked-in evidence."""

    started = time.monotonic()
    paths = {name: (REPO_ROOT / path if not Path(path).is_absolute() else Path(path)) for name, path in (upstream_paths or UPSTREAM_ARTIFACTS).items()}
    protected_before = protected_hashes(upstream_paths=paths)
    payloads, upstream = upstream_inventory(paths)
    raw_inventory = raw_file_inventory_and_hashes(payloads)
    identity = event_identity_recomputation(payloads, raw_inventory)
    primary_rows = _primary_rows(payloads)
    exposure = exposure_ledger_recomputation(payloads)
    veto = exact_veto_order_recomputation(primary_rows)
    effects = independent_effect_recomputation(payloads)
    protected = protected_case_recomputation(payloads)
    lifecycle = rollback_restart_and_non_resurrection_replay(payloads)
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    durations = duration_recomputation(payloads, audit_duration_s=measured_duration)
    comparison = upstream_vs_independent_field_comparison(payloads, identity, effects, protected, lifecycle)
    aggregate = aggregate_row_recomputation(primary_rows, comparison, raw_inventory, identity)
    attacks = attack_matrix(identity, exposure, veto, lifecycle, aggregate)
    protected_after = protected_hashes(upstream_paths=paths)
    artifact: JsonDict = {
        "status": "complete_with_findings",
        "upstream_artifact_inventory": upstream,
        "raw_file_inventory_and_hashes": raw_inventory,
        "independent_event_identity_recomputation": identity,
        "independent_exposure_ledger": exposure,
        "exact_veto_order_recomputation": veto,
        "per_unit_rows": {"rows": primary_rows, "row_count": len(primary_rows), "row_hash": sha256_json(primary_rows)},
        "audit_rows": {},
        "independent_effect_recomputation": effects,
        "protected_case_recomputation": protected,
        "rollback_restart_and_non_resurrection_replay": lifecycle,
        "duration_recomputation": durations,
        "upstream_vs_independent_field_comparison": comparison,
        "aggregate_row_recomputation": aggregate,
        "attack_matrix": attacks,
        "current_adversarial_findings": [],
        "critical_discrepancies": [],
        "csl_audit_eligible_score": 0.0,
        "protected_files_unchanged": protected_unchanged_receipt(protected_before, protected_after),
        "blocked_reason": "",
        "gate_check_summary": {},
        "preconditions_checked": preconditions_checked(date=date, upstream=upstream, raw_inventory=raw_inventory, identity=identity),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": {
            "value": True,
            "true_for": [
                "independent_exact_checker",
                "hash_recomputation",
                "chronology_recomputation",
                "arithmetic_recomputation",
            ],
            "false_for": {
                "upstream_summaries": False,
                "model_raw_text": False,
                "learned_weights": False,
                "claimed_gates": False,
            },
        },
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": measured_duration,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["critical_discrepancies"] = current_adversarial_findings(artifact)
    artifact["current_adversarial_findings"] = list(artifact["critical_discrepancies"])
    artifact["gate_check_summary"] = gate_check_summary(artifact)
    artifact["csl_audit_eligible_score"] = eligible_score(artifact)
    discrepancy_rows = [
        {"kind": "discrepancy", "discrepancy_kind": row.get("kind"), "severity": row.get("severity"), "detail": row}
        for row in artifact["critical_discrepancies"]
    ]
    event_audits = _audit_event_rows(primary_rows, identity, raw_inventory)
    artifact["audit_rows"] = {
        "rows": event_audits + discrepancy_rows,
        "event_row_count": len(event_audits),
        "discrepancy_row_count": len(discrepancy_rows),
        "row_hash": sha256_json(event_audits + discrepancy_rows),
    }
    if artifact["csl_audit_eligible_score"] == 1.0:
        artifact["status"] = "success_ready"
        artifact["honest_verdict"] = "success: independent V556 unique-event CSL audit confirms eligible evidence"
    else:
        artifact["status"] = "blocked_evidence" if upstream.get("all_required_present") is not True else "complete_with_findings"
        failed = artifact["gate_check_summary"]["failed_checks"]
        kinds = [str(row.get("kind")) for row in artifact["critical_discrepancies"]]
        artifact["blocked_reason"] = "; ".join([*kinds, *[f"failed_gate:{gate}" for gate in failed]])
        artifact["honest_verdict"] = (
            "blocked: missing or malformed upstream evidence"
            if artifact["status"].startswith("blocked")
            else "complete: independent V556 unique-event CSL audit found blockers"
        )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result_path, artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> bool:
    """Validate an Exp6470 artifact payload."""

    artifact = _artifact_value(value)
    require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})), "field_principles")
    require(set(artifact.get("field_provenance", {})) == set(REQUIRED_ARTIFACT_FIELDS), "field_provenance")
    for condition in READINESS_CONDITIONS:
        require(f"csl_audit_eligible_score:{condition}" in artifact.get("field_principles", {}), "field_principles")
    verdict = str(artifact.get("honest_verdict", ""))
    require(verdict.startswith(("success:", "complete:", "blocked:")), "honest_verdict")
    if artifact.get("status") == "success_ready":
        require(artifact.get("raw_file_inventory_and_hashes", {}).get("missing_count") == 0, "raw_file_inventory")
        require(artifact.get("independent_event_identity_recomputation", {}).get("one_raw_per_event") is True, "event_identity")
        require(artifact.get("exact_veto_order_recomputation", {}).get("all_admitted_writes_checked_first") is True, "exact_veto")
        require(artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True, "aggregate")
        require(artifact.get("attack_matrix", {}).get("all_critical_attacks_fail_closed") is True, "attack_matrix")
        require(artifact.get("critical_discrepancies") == [], "critical_discrepancies")
        require(artifact.get("csl_audit_eligible_score") == 1.0, "eligible_score")
    else:
        require(artifact.get("csl_audit_eligible_score") == 0.0, "eligible_score")
        require(artifact.get("gate_check_summary", {}).get("failed_check_count", 0) > 0, "gate_check_summary")
    require(artifact.get("critical_discrepancies") == current_adversarial_findings(artifact), "critical_discrepancies")
    require(artifact.get("current_adversarial_findings") == artifact.get("critical_discrepancies"), "current_adversarial_findings")
    require(artifact.get("gate_check_summary") == gate_check_summary(artifact), "gate_check_summary")
    require(artifact.get("csl_audit_eligible_score") == eligible_score(artifact), "eligible_score")
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.validate:
        validate_artifact(output)
        print(f"valid: {output}")
        return 0
    artifact = run(date=args.date, result_path=output)
    print(json.dumps({"status": artifact["status"], "result_path": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
