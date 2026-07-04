"""Exp 5236: reclassify GAP-4 after artifact-QA calibration.

Spec refs: REQ-REPORT-5236, SCENARIO-REPORT-5236-CLEAN-NULL,
SCENARIO-REPORT-5236-BLOCKED-RECHECK.

This module is deliberately a reader, not a generator. It takes the frozen
Exp 5224 pool and Exp 5225 validation as evidence, rechecks their receipts,
and emits the narrow status decision downstream capstones need.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot import experiment_5224_gap4_canonical_pool_builder_v478 as exp5224
from carnot import experiment_5225_gap4_clean_scale_validation_gated_v478 as exp5225
from scripts import adversarial_verify as av


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5236_gap4_clean_status_after_qa_calibration_v479"
EXPERIMENT_ID = 5236
MILESTONE = "2026.07.479"
RUN_DATE = "2026-07-04"
SCHEMA = "carnot.experiment_5236.gap4_clean_status_after_qa_calibration.v479"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5236_gap4_clean_status_after_qa_calibration_v479.json"
)
EXP5235_RELATIVE_PATH = Path(
    "results/experiment_5235_adversarial_qa_null_tautology_calibration_v479.json"
)
EXP5224_RELATIVE_PATH = Path(exp5224.RESULT_RELATIVE_PATH)
EXP5225_RELATIVE_PATH = Path(exp5225.RESULT_RELATIVE_PATH)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5236_gap4_clean_status_after_qa_calibration_v479.py"
)
INFERENCE_SUBSTRATE = "frozen_gap4_artifact_reclassification"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
DECISIONS = {
    "clean_null",
    "clean_positive",
    "blocked_flagged",
    "blocked_missing_receipts",
}
SPEC_REFS = [
    "REQ-REPORT-5236",
    "SCENARIO-REPORT-5236-CLEAN-NULL",
    "SCENARIO-REPORT-5236-BLOCKED-RECHECK",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "gap4_status_decision": (
        "BARE top-level string, exactly one of clean_null, clean_positive, "
        "blocked_flagged, blocked_missing_receipts."
    ),
    "gap4_headline_eligible": (
        "BARE top-level boolean. True only for a clean-positive GAP-4 decision "
        "that crosses the unchanged min-six rule."
    ),
    "canonical_pool_n": (
        "BARE top-level integer copied from the frozen Exp 5224 canonical pool; "
        "must remain 120 for the .478 reclassification."
    ),
    "wins": (
        "BARE top-level integer copied from frozen Exp 5225 validation wins; "
        "no new scoring pass may alter it."
    ),
    "losses": (
        "BARE top-level integer copied from frozen Exp 5225 validation losses; "
        "no new scoring pass may alter it."
    ),
    "ties": (
        "BARE top-level integer copied from frozen Exp 5225 validation ties; "
        "no new scoring pass may alter it."
    ),
    "qa_recheck_commands": (
        "List of commands or deterministic checks and pass/fail outcomes used "
        "to recheck Exp 5224 and Exp 5225."
    ),
    "pool_regenerated": (
        "Must be false; Exp 5236 is a reclassification over the frozen .478 "
        "pool, not candidate generation."
    ),
    "ops_docs_updated": (
        "Bare boolean recording whether this task updated "
        "ops/status/changelog/traceability docs; false is valid when a "
        "conductor stop rule delegates reconciliation."
    ),
    "inference_substrate": "Must be frozen_gap4_artifact_reclassification.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state "
        "whether GAP-4 is clean-null, clean-positive, or still blocked."
    ),
}

REQUIRED_SCHEMA_FIELDS = {
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "duration_s",
    "field_principles",
    "source_artifacts",
    "qa_calibration_artifact_path",
    "qa_calibration_passed",
    "recheck_reports",
    "remaining_blocker",
    "status_rationale",
    "tests_added_or_updated",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES,
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Return a checksum that changes when any emitted evidence field changes."""

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_stable_json(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def load_source_artifacts(root: Path | str = REPO_ROOT) -> tuple[JsonDict, JsonDict, JsonDict]:
    """Read the three frozen evidence artifacts from a root directory."""

    root_path = Path(root)
    return (
        _read_json(root_path / EXP5235_RELATIVE_PATH),
        _read_json(root_path / EXP5224_RELATIVE_PATH),
        _read_json(root_path / EXP5225_RELATIVE_PATH),
    )


def default_schema_recheck(
    root: Path | str,
    pool_artifact: Mapping[str, Any],
    validation_artifact: Mapping[str, Any],
) -> list[JsonDict]:
    """Run the same deterministic artifact-schema checks the source modules use."""

    del root
    pool_rows = pool_artifact.get("candidate_rows")
    pool_errors = (
        exp5224.artifact_schema_errors(pool_artifact, pool_rows)
        if isinstance(pool_rows, list)
        else ["candidate_rows_missing"]
    )
    validation_errors = (
        exp5225.artifact_schema_errors(validation_artifact)
        if validation_artifact
        else ["exp5225_artifact_missing"]
    )
    return [
        {
            "name": "exp5224_artifact_schema_errors",
            "path": str(EXP5224_RELATIVE_PATH),
            "passed": not pool_errors,
            "errors": list(pool_errors),
        },
        {
            "name": "exp5225_artifact_schema_errors",
            "path": str(EXP5225_RELATIVE_PATH),
            "passed": not validation_errors,
            "errors": list(validation_errors),
        },
    ]


def default_adversarial_recheck(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    """Run the real artifact-QA verifier over the frozen Exp 5224/5225 files."""

    root_path = Path(root)
    reports: list[JsonDict] = []
    for relative in (EXP5224_RELATIVE_PATH, EXP5225_RELATIVE_PATH):
        report = av.verify_artifact(root_path / relative)
        flags = report.get("flags", [])
        flag_count = _as_int(report.get("flag_count"))
        if flag_count is None:
            flag_count = len(flags) if isinstance(flags, list) else 0
        reports.append(
            {
                "name": "adversarial_verify",
                "path": str(relative),
                "passed": report.get("loaded") is True and flag_count == 0,
                "loaded": report.get("loaded") is True,
                "flag_count": flag_count,
                "flags": flags if isinstance(flags, list) else [],
                "error": report.get("error"),
            }
        )
    return reports


def _schema_command_summaries(schema_reports: Sequence[Mapping[str, Any]]) -> list[str]:
    commands: list[str] = []
    for report in schema_reports:
        name = str(report.get("name") or "schema_recheck")
        errors = report.get("errors")
        error_count = len(errors) if isinstance(errors, list) else 0
        status = "PASS" if report.get("passed") is True else "FAIL"
        commands.append(f"{name}: {status} ({error_count} errors)")
    return commands


def _adversarial_command_summary(reports: Sequence[Mapping[str, Any]]) -> str:
    flagged = sum(int(report.get("flag_count") or 0) for report in reports)
    status = "PASS" if flagged == 0 and all(report.get("passed") is True for report in reports) else "FAIL"
    return (
        ".venv/bin/python scripts/adversarial_verify.py "
        f"{EXP5224_RELATIVE_PATH} {EXP5225_RELATIVE_PATH} --json: "
        f"{status} (flagged_count={flagged})"
    )


def _flag_blockers(adversarial_reports: Sequence[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for report in adversarial_reports:
        flags = report.get("flags")
        if not isinstance(flags, list) or not flags:
            continue
        kinds = sorted({str(flag.get("kind") or "unknown") for flag in flags if isinstance(flag, Mapping)})
        blockers.append(f"{report.get('path')}: adversarial_flags={','.join(kinds)}")
    return blockers


def _receipt_blockers(
    *,
    qa_artifact: Mapping[str, Any],
    pool_artifact: Mapping[str, Any],
    validation_artifact: Mapping[str, Any],
    schema_reports: Sequence[Mapping[str, Any]],
    adversarial_reports: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if qa_artifact.get("qa_calibration_passed") is not True:
        blockers.append("qa_calibration_not_passed")
    pool_n = _as_int(pool_artifact.get("canonical_pool_n"))
    if pool_artifact.get("gap4_canonical_pool_usable") is not True:
        blockers.append("exp5224_pool_not_usable")
    if pool_n != exp5224.CANONICAL_POOL_TARGET_N:
        blockers.append("canonical_pool_n_not_120")
    if validation_artifact.get("gap4_clean_validation_complete") is not True:
        blockers.append("exp5225_clean_validation_not_complete")
    if validation_artifact.get("precondition_errors"):
        blockers.append("exp5225_precondition_errors_present")
    validation_pool_n = _as_int(validation_artifact.get("canonical_pool_n"))
    if validation_pool_n is not None and pool_n is not None and validation_pool_n != pool_n:
        blockers.append("exp5224_exp5225_pool_n_mismatch")
    if _as_int(validation_artifact.get("n_scored")) is None:
        blockers.append("n_scored_missing")
    for field in ("wins", "losses", "ties"):
        if _as_int(validation_artifact.get(field)) is None:
            blockers.append(f"{field}_missing")
    for report in schema_reports:
        if report.get("passed") is True:
            continue
        errors = report.get("errors")
        detail = ",".join(str(item) for item in errors) if isinstance(errors, list) else "failed"
        blockers.append(f"{report.get('name')}: {detail}")
    for report in adversarial_reports:
        if report.get("passed") is True or int(report.get("flag_count") or 0) > 0:
            continue
        blockers.append(f"{report.get('path')}: adversarial_recheck_missing_or_failed")
    return sorted(dict.fromkeys(blockers))


def _decision(
    validation_artifact: Mapping[str, Any],
    flag_blockers: Sequence[str],
    receipt_blockers: Sequence[str],
) -> str:
    if flag_blockers:
        return "blocked_flagged"
    if receipt_blockers:
        return "blocked_missing_receipts"
    if validation_artifact.get("exact_test_passes_min6_rule") is True:
        return "clean_positive"
    return "clean_null"


def _honest_verdict(decision: str, n: int, wins: int, losses: int, ties: int) -> str:
    if decision == "clean_positive":
        return (
            f"success: GAP-4 is clean-positive after QA calibration; frozen n={n} "
            f"validation has wins={wins}, losses={losses}, ties={ties} and crosses "
            "the unchanged min-six rule."
        )
    if decision == "clean_null":
        return (
            f"complete: GAP-4 is clean-null after QA calibration; frozen n={n} "
            f"validation has wins={wins}, losses={losses}, ties={ties} and does "
            "not cross the unchanged min-six rule."
        )
    return (
        f"complete: GAP-4 is still blocked after QA calibration with decision "
        f"{decision}; frozen n={n} validation currently has wins={wins}, "
        f"losses={losses}, ties={ties}."
    )


def _source_artifact_summaries(root: Path | str) -> list[JsonDict]:
    root_path = Path(root)
    summaries: list[JsonDict] = []
    for relative in (EXP5235_RELATIVE_PATH, EXP5224_RELATIVE_PATH, EXP5225_RELATIVE_PATH):
        path = root_path / relative
        payload = _read_json(path)
        summaries.append(
            {
                "path": str(relative),
                "exists": path.exists(),
                "sha256": _file_sha256(path),
                "experiment": payload.get("experiment"),
                "experiment_id": payload.get("experiment_id"),
            }
        )
    return summaries


def _status_rationale(decision: str, blocker: str | None) -> str:
    if decision == "clean_positive":
        return "Frozen GAP-4 validation is clean and crosses the unchanged min-six floor."
    if decision == "clean_null":
        return "Frozen GAP-4 validation is clean but does not cross the unchanged min-six floor."
    return str(blocker or "blocked without a more specific blocker")


def build_artifact(
    *,
    qa_artifact: Mapping[str, Any],
    pool_artifact: Mapping[str, Any],
    validation_artifact: Mapping[str, Any],
    schema_reports: Sequence[Mapping[str, Any]],
    adversarial_reports: Sequence[Mapping[str, Any]],
    qa_recheck_commands: Sequence[str] = (),
    duration_s: float = 0.0,
    ops_docs_updated: bool = False,
    source_artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the terminal status artifact from already-frozen evidence.

    The key safety rule is that the status can improve from "blocked by QA" to
    "clean null" only through reclassification. No candidate rows or thresholds
    are recomputed here.
    """

    flags = _flag_blockers(adversarial_reports)
    receipts = _receipt_blockers(
        qa_artifact=qa_artifact,
        pool_artifact=pool_artifact,
        validation_artifact=validation_artifact,
        schema_reports=schema_reports,
        adversarial_reports=adversarial_reports,
    )
    decision = _decision(validation_artifact, flags, receipts)
    blockers = flags if flags else receipts
    remaining_blocker = "; ".join(blockers) if blockers else None
    n = _as_int(pool_artifact.get("canonical_pool_n")) or 0
    wins = _as_int(validation_artifact.get("wins")) or 0
    losses = _as_int(validation_artifact.get("losses")) or 0
    ties = _as_int(validation_artifact.get("ties")) or 0
    commands = [
        (
            f"read {EXP5235_RELATIVE_PATH}: "
            f"{'PASS' if qa_artifact.get('qa_calibration_passed') is True else 'FAIL'} "
            f"(qa_calibration_passed={qa_artifact.get('qa_calibration_passed')!r})"
        ),
        *_schema_command_summaries(schema_reports),
        _adversarial_command_summary(adversarial_reports),
        *list(qa_recheck_commands),
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts": [dict(item) for item in source_artifacts or ()],
        "qa_calibration_artifact_path": str(EXP5235_RELATIVE_PATH),
        "qa_calibration_passed": qa_artifact.get("qa_calibration_passed") is True,
        "recheck_reports": {
            "schema": [dict(report) for report in schema_reports],
            "adversarial": [dict(report) for report in adversarial_reports],
        },
        "remaining_blocker": remaining_blocker,
        "status_rationale": _status_rationale(decision, remaining_blocker),
        "tests_added_or_updated": [str(TEST_RELATIVE_PATH)],
        "gap4_status_decision": decision,
        "gap4_headline_eligible": decision == "clean_positive",
        "canonical_pool_n": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "qa_recheck_commands": commands,
        "pool_regenerated": False,
        "ops_docs_updated": bool(ops_docs_updated),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(decision, n, wins, losses, ties),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _require_bare_bool(artifact: Mapping[str, Any], field: str) -> None:
    if not isinstance(artifact.get(field), bool):
        raise ValueError(f"{field}_bare_bool")


def _require_bare_int(artifact: Mapping[str, Any], field: str) -> None:
    if _as_int(artifact.get(field)) is None:
        raise ValueError(f"{field}_bare_int")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 5236 schema before the JSON is written."""

    missing = REQUIRED_SCHEMA_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles")
    decision = artifact.get("gap4_status_decision")
    if decision not in DECISIONS:
        raise ValueError("gap4_status_decision")
    for field in ("gap4_headline_eligible", "pool_regenerated", "ops_docs_updated"):
        _require_bare_bool(artifact, field)
    for field in ("canonical_pool_n", "wins", "losses", "ties"):
        _require_bare_int(artifact, field)
    if artifact["gap4_headline_eligible"] is not (decision == "clean_positive"):
        raise ValueError("gap4_headline_eligible")
    if artifact["pool_regenerated"] is not False:
        raise ValueError("pool_regenerated")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    commands = artifact.get("qa_recheck_commands")
    if not isinstance(commands, list) or not all(isinstance(item, str) for item in commands):
        raise ValueError("qa_recheck_commands")
    if decision in {"blocked_flagged", "blocked_missing_receipts"}:
        if not isinstance(artifact.get("remaining_blocker"), str) or not artifact["remaining_blocker"]:
            raise ValueError("remaining_blocker")
    elif artifact.get("remaining_blocker") is not None:
        raise ValueError("remaining_blocker")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict")
    if decision == "clean_null" and "clean-null" not in verdict:
        raise ValueError("honest_verdict")
    if decision == "clean_positive" and "clean-positive" not in verdict:
        raise ValueError("honest_verdict")
    if str(decision).startswith("blocked") and "still blocked" not in verdict:
        raise ValueError("honest_verdict")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")


SchemaRechecker = Callable[[Path | str, Mapping[str, Any], Mapping[str, Any]], list[JsonDict]]
AdversarialRechecker = Callable[[Path | str], list[JsonDict]]


def write_outputs(
    *,
    root: Path | str = REPO_ROOT,
    schema_rechecker: SchemaRechecker = default_schema_recheck,
    adversarial_rechecker: AdversarialRechecker = default_adversarial_recheck,
    qa_recheck_commands: Sequence[str] = (),
    duration_s: float = 0.0,
    ops_docs_updated: bool = False,
) -> JsonDict:
    """Read frozen GAP-4 evidence, recheck it, and write the Exp 5236 artifact."""

    root_path = Path(root)
    qa_artifact, pool_artifact, validation_artifact = load_source_artifacts(root_path)
    schema_reports = schema_rechecker(root_path, pool_artifact, validation_artifact)
    adversarial_reports = adversarial_rechecker(root_path)
    artifact = build_artifact(
        qa_artifact=qa_artifact,
        pool_artifact=pool_artifact,
        validation_artifact=validation_artifact,
        schema_reports=schema_reports,
        adversarial_reports=adversarial_reports,
        qa_recheck_commands=qa_recheck_commands,
        duration_s=duration_s,
        ops_docs_updated=ops_docs_updated,
        source_artifacts=_source_artifact_summaries(root_path),
    )
    validate_artifact(artifact)
    _write_json_atomic(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--qa-command", action="append", default=[])
    parser.add_argument("--ops-docs-updated", action="store_true")
    args = parser.parse_args(argv)
    artifact = write_outputs(
        root=args.root,
        duration_s=args.duration_s,
        qa_recheck_commands=args.qa_command,
        ops_docs_updated=args.ops_docs_updated,
    )
    print(_stable_json(artifact))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
