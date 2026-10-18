"""Exp 1498 trace2skill artifact reachability and resolver freshness audit.

Exp 1497 promotes or retains trace-derived skills only as far as the daily
manifest can still resolve its evidence. This module performs the next bounded
check: every source artifact path is tested for existence and structured
parseability, manifest resolver checks are inspected for stale or ambiguous
signals, and the terminal artifact records repair or retirement decisions
without deleting any learned skill.

Spec: REQ-LEARN-1498, SCENARIO-LEARN-1498-A, SCENARIO-LEARN-1498-B.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

OUTPUT_FILE = "experiment_1498_trace2skill_artifact_reachability_audit.json"
DEFAULT_EXP1497_PATH = DEFAULT_RESULTS_DIR / "experiment_1497_fr11_trace2skill_daily_eval_v10.json"
DEFAULT_MANIFEST_PATH = DEFAULT_RESULTS_DIR / "fr11_trace2skill_daily_eval_manifest_1497.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1498_trace2skill_artifact_reachability_audit"
SCHEMA = "trace2skill_artifact_reachability_audit_v1"
RUN_DATE = "20260507"
PASSED_VERDICT = "complete: trace2skill_artifact_reachability_audit_passed"
DECISIONS_VERDICT = "complete: trace2skill_artifact_reachability_decisions_recorded"
BLOCKED_VERDICT = "complete: blocked_missing_or_malformed_exp1497_inputs"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "artifact_reachability_audit_complete",
    "gated_inputs_present",
    "skills_checked",
    "source_artifacts_checked",
    "reachable_artifact_count",
    "unreachable_artifact_count",
    "stale_artifact_count",
    "ambiguous_resolver_count",
    "repair_decisions",
    "retirement_decisions",
    "blockers",
    "honest_verdict",
)
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
SOURCE_REQUIRED_FIELDS: tuple[str, ...] = ("status", "schema", "spec", "honest_verdict")
MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "skill_id",
    "case_id",
    "source_artifacts",
    "expected_resolver_checks",
)
RESOLVER_REQUIRED_FIELDS: tuple[str, ...] = ("name", "expected", "observed")


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _display_path(path: Path | str, *, project_root: str | Path = REPO_ROOT) -> str:
    target = Path(path)
    try:
        return target.relative_to(Path(project_root)).as_posix()
    except ValueError:
        return target.name


def _resolve_path(path: Path | str, *, project_root: str | Path = REPO_ROOT) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(project_root) / candidate
    return candidate


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1498-1/7: persist a bootstrap artifact before gate loading."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "spec": [
                "REQ-LEARN-1498",
                "SCENARIO-LEARN-1498-A",
                "SCENARIO-LEARN-1498-B",
            ],
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "started_at": _timestamp(),
            "status": "in_progress",
            "artifact_reachability_audit_complete": False,
            "gated_inputs_present": False,
            "skills_checked": 0,
            "source_artifacts_checked": 0,
            "reachable_artifact_count": 0,
            "unreachable_artifact_count": 0,
            "stale_artifact_count": 0,
            "ambiguous_resolver_count": 0,
            "repair_decisions": [],
            "retirement_decisions": [],
            "blockers": [],
            "honest_verdict": "in_progress",
        },
    )


def parse_structured_file(path: Path | str) -> dict[str, Any]:
    """REQ-LEARN-1498-4: parse JSON, JSONL, or YAML evidence without raising."""

    target = Path(path)
    if not target.exists():
        return {"path": str(target), "parse_status": "missing", "error": "path does not exist"}
    try:
        raw = target.read_text(encoding="utf-8")
        suffix = target.suffix.lower()
        if suffix == ".jsonl":
            payload = [json.loads(line) for line in raw.splitlines() if line.strip()]
        elif suffix in {".yaml", ".yml"}:
            import yaml

            payload = yaml.safe_load(raw)
        else:
            payload = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - audit records parse failure as data.
        return {"path": str(target), "parse_status": "error", "error": str(exc)}
    return {"path": str(target), "parse_status": "parsed", "payload": payload}


def load_manifest_rows(path: Path | str) -> list[Mapping[str, Any]]:
    """REQ-LEARN-1498-3: load manifest rows as JSON objects."""

    parsed = parse_structured_file(path)
    if parsed["parse_status"] != "parsed":
        raise AssertionError(f"manifest failed to parse: {parsed.get('error')}")
    payload = parsed["payload"]
    if not isinstance(payload, list):
        raise AssertionError("manifest must parse to a list of rows")
    if not all(isinstance(row, Mapping) for row in payload):
        raise AssertionError("manifest rows must be JSON objects")
    return list(payload)


def _unique(values: Sequence[str]) -> list[str]:
    return sorted({value for value in values if value})


def _row_skill_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("skill_id")
    return str(value) if value else f"missing-skill-id-{index}"


def _source_artifact_refs(manifest_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    refs: list[str] = []
    for row in manifest_rows:
        source_artifacts = row.get("source_artifacts", [])
        if isinstance(source_artifacts, Sequence) and not isinstance(source_artifacts, (str, bytes)):
            refs.extend(str(path) for path in source_artifacts)
    return list(dict.fromkeys(refs))


def _model_references(exp1497_artifact: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    model_specs = exp1497_artifact.get("model_specs", [])
    if isinstance(model_specs, Sequence) and not isinstance(model_specs, (str, bytes)):
        refs.extend(str(model) for model in model_specs)
    models_used = exp1497_artifact.get("models_used", [])
    if isinstance(models_used, Sequence) and not isinstance(models_used, (str, bytes)):
        for model in models_used:
            if isinstance(model, Mapping):
                refs.extend(str(model[key]) for key in ("hf_id", "model_id", "id") if key in model)
            elif model:
                refs.append(str(model))
    return _unique(refs)


def _verifier_dependencies(manifest_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    refs: list[str] = []
    for row in manifest_rows:
        for outcome_key in ("baseline_outcome", "memory_assisted_outcome"):
            outcome = row.get(outcome_key)
            if isinstance(outcome, Mapping) and outcome.get("verifier_signal"):
                refs.append(str(outcome["verifier_signal"]))
    return _unique(refs)


def _audit_source_artifact(
    raw_path: str,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    resolved = _resolve_path(raw_path, project_root=project_root)
    parsed = parse_structured_file(resolved)
    record: dict[str, Any] = {
        "referenced_as": raw_path,
        "path": _display_path(resolved, project_root=project_root),
        "parse_status": parsed["parse_status"],
    }
    if parsed["parse_status"] != "parsed":
        record.update({"status": "unreachable", "reason": parsed.get("error", "unparseable")})
        return record

    payload = parsed["payload"]
    if not isinstance(payload, Mapping):
        record.update({"status": "stale", "reason": "source artifact is not a JSON object"})
        return record

    missing = [field for field in SOURCE_REQUIRED_FIELDS if field not in payload]
    if missing:
        record.update(
            {
                "status": "stale",
                "reason": "missing expected source fields",
                "missing_fields": missing,
            }
        )
        return record
    if payload.get("status") != "complete":
        record.update(
            {
                "status": "stale",
                "reason": f"source status is {payload.get('status')!r}",
            }
        )
        return record

    record.update({"status": "reachable", "source_schema": payload.get("schema")})
    return record


def _resolver_audit(manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    resolver_keys: list[str] = []
    ambiguous_count = 0
    mismatch_count = 0
    affected_skills: set[str] = set()

    for index, row in enumerate(manifest_rows):
        seen_in_row: set[str] = set()
        checks = row.get("expected_resolver_checks", [])
        if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
            ambiguous_count += 1
            affected_skills.add(_row_skill_id(row, index))
            continue
        for check in checks:
            if not isinstance(check, Mapping) or any(field not in check for field in RESOLVER_REQUIRED_FIELDS):
                ambiguous_count += 1
                affected_skills.add(_row_skill_id(row, index))
                continue
            name = str(check["name"])
            resolver_keys.append(name)
            if name in seen_in_row:
                ambiguous_count += 1
                affected_skills.add(_row_skill_id(row, index))
            seen_in_row.add(name)
            if check["observed"] != check["expected"]:
                mismatch_count += 1
                affected_skills.add(_row_skill_id(row, index))

    return {
        "resolver_keys": _unique(resolver_keys),
        "ambiguous_count": ambiguous_count,
        "mismatch_count": mismatch_count,
        "affected_skills": sorted(affected_skills),
    }


def _row_contract_blockers(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    run_date: str,
) -> list[str]:
    blockers: list[str] = []
    if any(any(field not in row for field in MANIFEST_REQUIRED_FIELDS) for row in manifest_rows):
        blockers.append("manifest_row_missing_required_fields")
    if any(str(row.get("run_date") or "") != run_date for row in manifest_rows):
        blockers.append("manifest_run_date_stale")
    return blockers


def _repair_decisions(
    *,
    source_audit: Sequence[Mapping[str, Any]],
    resolver_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for record in source_audit:
        status = record["status"]
        if status == "unreachable":
            decisions.append(
                {
                    "issue": "unreachable_source_artifact",
                    "artifact_path": record["path"],
                    "decision": "restore_or_regenerate_source_artifact_before_promotion",
                }
            )
        if status == "stale":
            decisions.append(
                {
                    "issue": "stale_source_artifact",
                    "artifact_path": record["path"],
                    "decision": "rerun_source_or_refresh_manifest_reference",
                }
            )
    if int(resolver_audit["ambiguous_count"]) > 0:
        decisions.append(
            {
                "issue": "ambiguous_resolver",
                "affected_skills": list(resolver_audit["affected_skills"]),
                "decision": "deduplicate_or_name_resolver_checks_before_promotion",
            }
        )
    if int(resolver_audit["mismatch_count"]) > 0:
        decisions.append(
            {
                "issue": "resolver_observation_mismatch",
                "affected_skills": list(resolver_audit["affected_skills"]),
                "decision": "refresh_daily_eval_manifest_from_live_resolver_checks",
            }
        )
    return decisions


def _retirement_decisions(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    source_audit: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if not any(record["status"] in {"unreachable", "stale"} for record in source_audit):
        return []
    return [
        {
            "skill_id": _row_skill_id(row, index),
            "decision": "retire_if_unrepaired",
            "reason": "source evidence is unreachable or stale",
        }
        for index, row in enumerate(manifest_rows)
    ]


def _blocked_artifact(
    *,
    blockers: Sequence[str],
    project_root: str | Path,
    run_date: str,
    started_at: str,
    duration_s: float,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1498", "SCENARIO-LEARN-1498-B"],
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "status": "blocked",
        "artifact_reachability_audit_complete": False,
        "gated_inputs_present": False,
        "skills_checked": 0,
        "source_artifacts_checked": 0,
        "reachable_artifact_count": 0,
        "unreachable_artifact_count": 0,
        "stale_artifact_count": 0,
        "ambiguous_resolver_count": 0,
        "repair_decisions": [],
        "retirement_decisions": [],
        "blockers": list(blockers),
        "honest_verdict": BLOCKED_VERDICT,
        "tests_run": list(commands_run or []),
    }
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    exp1497_artifact: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    exp1497_path: Path | str,
    manifest_path: Path | str,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1498-3/4/5/6/7: build the terminal audit artifact."""

    source_refs = _source_artifact_refs(manifest_rows)
    source_audit = [
        _audit_source_artifact(path, project_root=project_root) for path in source_refs
    ]
    resolver_audit = _resolver_audit(manifest_rows)
    blockers = _row_contract_blockers(manifest_rows, run_date=run_date)

    unreachable = sum(1 for record in source_audit if record["status"] == "unreachable")
    stale = sum(1 for record in source_audit if record["status"] == "stale")
    reachable = sum(1 for record in source_audit if record["status"] == "reachable")
    if unreachable:
        blockers.append("unreachable_source_artifact")
    if stale:
        blockers.append("stale_source_artifact")
    if int(resolver_audit["ambiguous_count"]) > 0:
        blockers.append("ambiguous_resolver")
    if int(resolver_audit["mismatch_count"]) > 0:
        blockers.append("resolver_observation_mismatch")

    repairs = _repair_decisions(source_audit=source_audit, resolver_audit=resolver_audit)
    retirements = _retirement_decisions(manifest_rows, source_audit=source_audit)
    skill_ids = [_row_skill_id(row, index) for index, row in enumerate(manifest_rows)]
    honest_verdict = PASSED_VERDICT if not repairs and not retirements and not blockers else DECISIONS_VERDICT
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": [
            "REQ-LEARN-1498",
            "SCENARIO-LEARN-1498-A",
            "SCENARIO-LEARN-1498-B",
        ],
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or _timestamp(),
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete",
        "artifact_reachability_audit_complete": True,
        "gated_inputs_present": True,
        "skills_checked": len(set(skill_ids)),
        "source_artifacts_checked": len(source_refs),
        "reachable_artifact_count": reachable,
        "unreachable_artifact_count": unreachable,
        "stale_artifact_count": stale,
        "ambiguous_resolver_count": int(resolver_audit["ambiguous_count"]),
        "repair_decisions": repairs,
        "retirement_decisions": retirements,
        "blockers": blockers,
        "honest_verdict": honest_verdict,
        "source_artifact_audit": source_audit,
        "resolver_keys": list(resolver_audit["resolver_keys"]),
        "model_references": _model_references(exp1497_artifact),
        "verifier_dependencies": _verifier_dependencies(manifest_rows),
        "gated_input_paths": {
            "exp1497_artifact": _display_path(exp1497_path, project_root=project_root),
            "daily_eval_manifest": _display_path(manifest_path, project_root=project_root),
        },
        "tests_run": list(commands_run or []),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1498-7: enforce required fields and terminal verdict discipline."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] == "in_progress":
        return
    if artifact["status"] not in {"complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    honest_verdict = str(artifact["honest_verdict"])
    if not honest_verdict.startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    for field in (
        "skills_checked",
        "source_artifacts_checked",
        "reachable_artifact_count",
        "unreachable_artifact_count",
        "stale_artifact_count",
        "ambiguous_resolver_count",
    ):
        if int(artifact[field]) < 0:
            raise AssertionError("counts must be non-negative")


def run(
    *,
    exp1497_path: Path | str = DEFAULT_EXP1497_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the Exp 1498 audit and write a terminal artifact."""

    started_at = _timestamp()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)

    blockers: list[str] = []
    exp1497_target = Path(exp1497_path)
    manifest_target = Path(manifest_path)
    if not exp1497_target.exists():
        blockers.append("missing_exp1497_daily_eval_artifact")
    if not manifest_target.exists():
        blockers.append("missing_exp1497_daily_eval_manifest")
    if blockers:
        artifact = _blocked_artifact(
            blockers=blockers,
            project_root=project_root,
            run_date=run_date,
            started_at=started_at,
            duration_s=time.perf_counter() - t0,
            commands_run=commands_run,
        )
        return _write_json(out_path, artifact)

    exp1497_parsed = parse_structured_file(exp1497_target)
    try:
        manifest_rows = load_manifest_rows(manifest_target)
    except AssertionError as exc:
        blockers.append(f"malformed_exp1497_daily_eval_manifest: {exc}")
        manifest_rows = []
    exp1497_payload = exp1497_parsed.get("payload")
    if exp1497_parsed["parse_status"] != "parsed" or not isinstance(exp1497_payload, Mapping):
        blockers.append("malformed_exp1497_daily_eval_artifact")

    if blockers:
        artifact = _blocked_artifact(
            blockers=blockers,
            project_root=project_root,
            run_date=run_date,
            started_at=started_at,
            duration_s=time.perf_counter() - t0,
            commands_run=commands_run,
        )
        return _write_json(out_path, artifact)

    artifact = build_artifact(
        exp1497_artifact=exp1497_payload,
        manifest_rows=manifest_rows,
        exp1497_path=exp1497_target,
        manifest_path=manifest_target,
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        commands_run=commands_run,
    )
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
