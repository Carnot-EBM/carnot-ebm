"""Audit the Exp 3084 exact fixture bank for `.289` live-evaluation reuse.

Spec refs: REQ-VERIFY-3097, SCENARIO-VERIFY-3097.

The audit is intentionally offline. It reads checked-in artifacts, validates
fixture rows against their exact local metadata, and writes a derived manifest
that downstream live model runs can sample without treating tiny panels as
headline evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.eval import resyn_exact_fixture_bank_generator_v1 as fixture_bank


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3097_exact_fixture_eval_protocol_audit_v1"
SCHEMA = "carnot.exact_fixture_eval_protocol_audit.v1"
OUTPUT_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
STRATIFIED_MANIFEST_REL_PATH = Path(
    "results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl"
)
FIXTURE_MANIFEST_REL_PATH = Path("results/resyn_exact_fixture_bank_3084/fixture_manifest.jsonl")
EXP3084_REL_PATH = Path("results/experiment_3084_resyn_exact_fixture_bank_generator_v1.json")
EXP3085_REL_PATH = Path("results/experiment_3085_icalm_task_abstention_sota_panel_v2.json")
EXP3085_ROWS_REL_PATH = Path("results/icalm_task_abstention_sota_panel_3085/rows.jsonl")
EXP3086_REL_PATH = Path("results/experiment_3086_dafny_z3_formal_feedback_pilot_v1.json")
EXP3094_REL_PATH = Path("results/experiment_3094_capstone_v288.json")
MINIMUM_LIVE_EVAL_COUNT = 48
MINIMUM_FORMAL_FEEDBACK_REPAIR_COUNT = 18
MINIMUM_REPAIR_GATING_COUNT = 24
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = (
    "eval_protocol_ready",
    "usable_fixture_count",
    "rejected_fixture_count",
    "stratified_eval_manifest_path",
    "minimum_live_eval_count",
    "fixture_family_counts",
    "downstream_usage",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
REQUIRED_MANIFEST_FIELDS = (
    "task_family",
    "expected_answer",
    "solver_label",
    "perturbation_type",
    "verifier_target",
    "repair_target",
)
SOURCE_REL_PATHS: tuple[tuple[str, Path, str], ...] = (
    ("codex", Path("CODEX.md"), "repo spec-first workflow"),
    ("claude", Path("CLAUDE.md"), "sample-size and artifact-authenticity discipline"),
    ("experiment_template", Path("scripts/experiment_template.py"), "experiment schema context"),
    ("exp3084_artifact", EXP3084_REL_PATH, "exact fixture-bank terminal artifact"),
    ("exp3084_manifest", FIXTURE_MANIFEST_REL_PATH, "checked-in exact fixture rows"),
    ("exp3085_artifact", EXP3085_REL_PATH, "tiny abstention-panel evidence"),
    ("exp3085_rows", EXP3085_ROWS_REL_PATH, "row transcript used to diagnose the 9-case panel"),
    ("exp3086_artifact", EXP3086_REL_PATH, "formal-feedback pilot evidence"),
    ("exp3094_capstone", EXP3094_REL_PATH, ".288 capstone and .289 recommendation"),
)


@dataclass(frozen=True)
class AuditConfig:
    """Runtime paths and thresholds for the offline audit."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    fixture_manifest_path: Path | None = None
    stratified_manifest_path: Path | None = None
    minimum_live_eval_count: int = MINIMUM_LIVE_EVAL_COUNT
    started_s: float | None = None
    clock: ClockFn = time.perf_counter

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def source_manifest_path(self) -> Path:
        return self.fixture_manifest_path or self.repo_root / FIXTURE_MANIFEST_REL_PATH

    def protocol_manifest_path(self) -> Path:
        return self.stratified_manifest_path or self.repo_root / STRATIFIED_MANIFEST_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_s is None else float(self.started_s)


@dataclass(frozen=True)
class FixtureAudit:
    """Validated usable rows plus rejected-row diagnostics."""

    usable_rows: list[JsonDict]
    rejected_rows: list[JsonDict]


def write_artifact(config: AuditConfig | None = None) -> JsonDict:
    """Build and write the terminal audit artifact and derived manifest."""

    active = config or AuditConfig()
    started_s = active.start_time()
    artifact = build_artifact(active, started_s=started_s)
    _write_json(active.artifact_path(), artifact)
    return artifact


def build_artifact(
    config: AuditConfig | None = None, *, started_s: float | None = None
) -> JsonDict:
    """Return the Exp 3097 protocol artifact without invoking live inference."""

    active = config or AuditConfig()
    started = active.start_time() if started_s is None else started_s
    raw_rows = _safe_load_jsonl(active.source_manifest_path())
    fixture_audit = audit_fixture_rows(raw_rows)
    manifest_rows = build_stratified_manifest_rows(fixture_audit.usable_rows)
    manifest_path = active.protocol_manifest_path()
    if manifest_rows:
        _write_jsonl(manifest_path, manifest_rows)
    exp3085 = _safe_load_json(active.repo_root / EXP3085_REL_PATH)
    exp3085_rows = _safe_load_jsonl(active.repo_root / EXP3085_ROWS_REL_PATH)
    family_counts = dict(sorted(Counter(str(row["task_family"]) for row in manifest_rows).items()))
    perturbation_counts = dict(
        sorted(Counter(str(row["perturbation_type"]) for row in manifest_rows).items())
    )
    ready = (
        len(manifest_rows) >= int(active.minimum_live_eval_count)
        and len(manifest_rows) == len(fixture_audit.usable_rows)
        and _manifest_rows_have_required_fields(manifest_rows)
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "eval_protocol_ready": ready,
        "usable_fixture_count": len(fixture_audit.usable_rows),
        "rejected_fixture_count": len(fixture_audit.rejected_rows),
        "rejected_fixtures": fixture_audit.rejected_rows,
        "stratified_eval_manifest_path": _relative_path(active.repo_root, manifest_path),
        "stratified_eval_manifest_sha256": sha256_file(manifest_path)
        if manifest_rows and manifest_path.is_file()
        else None,
        "minimum_live_eval_count": int(active.minimum_live_eval_count),
        "fixture_family_counts": family_counts,
        "fixture_perturbation_counts": perturbation_counts,
        "manifest_required_fields": list(REQUIRED_MANIFEST_FIELDS),
        "exp3085_tiny_panel_diagnosis": diagnose_exp3085_tiny_panel(
            exp3085=exp3085,
            panel_rows=exp3085_rows,
            usable_fixture_count=len(fixture_audit.usable_rows),
        ),
        "downstream_usage": downstream_usage(
            usable_rows=manifest_rows,
            minimum_live_eval_count=int(active.minimum_live_eval_count),
        ),
        "source_artifacts": source_artifacts(active.repo_root),
        "inference_substrate": inference_substrate(),
        "duration_s": active.clock() - started,
        "honest_verdict": _honest_verdict(
            ready=ready,
            usable_fixture_count=len(fixture_audit.usable_rows),
            rejected_fixture_count=len(fixture_audit.rejected_rows),
            minimum_live_eval_count=int(active.minimum_live_eval_count),
        ),
    }
    validate_artifact(artifact)
    return artifact


def audit_fixture_rows(rows: Sequence[Mapping[str, Any]]) -> FixtureAudit:
    """Classify fixture rows as usable or rejected with visible reasons."""

    usable: list[JsonDict] = []
    rejected: list[JsonDict] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        fixture_id = str(row.get("fixture_id", f"row_{index}"))
        prompt_hash = str(row.get("prompt_payload_sha256", ""))
        missing = sorted(set(fixture_bank.REQUIRED_FIXTURE_FIELDS) - set(row))
        if missing:
            rejected.append(
                {
                    "fixture_id": fixture_id,
                    "row_index": index,
                    "reason": "missing_required_fields",
                    "detail": missing,
                }
            )
            continue
        if fixture_id in seen_ids:
            rejected.append(
                {
                    "fixture_id": fixture_id,
                    "row_index": index,
                    "reason": "duplicate_fixture_id",
                    "detail": "fixture_id already accepted earlier in manifest",
                }
            )
            continue
        if prompt_hash in seen_hashes:
            rejected.append(
                {
                    "fixture_id": fixture_id,
                    "row_index": index,
                    "reason": "duplicate_prompt_payload_sha256",
                    "detail": "prompt payload hash already accepted earlier in manifest",
                }
            )
            continue
        try:
            fixture_bank.validate_fixture_rows([row])
        except Exception as exc:  # noqa: BLE001 - diagnostics must preserve any validator failure.
            rejected.append(
                {
                    "fixture_id": fixture_id,
                    "row_index": index,
                    "reason": "exact_authority_validation_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        seen_ids.add(fixture_id)
        seen_hashes.add(prompt_hash)
        usable.append(row)
    return FixtureAudit(usable_rows=usable, rejected_rows=rejected)


def build_stratified_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Convert exact fixture rows into the `.289` downstream protocol manifest."""

    output = []
    for row in rows:
        targets = expected_targets(row)
        output.append(
            {
                "schema": "carnot.exact_fixture_eval_manifest.v1",
                "source_fixture_id": row["fixture_id"],
                "task_family": row["family"],
                "task_axis": row["task_axis"],
                "expected_answer": targets["expected_answer"],
                "solver_label": targets["solver_label"],
                "perturbation_type": row["perturbation_family"],
                "verifier_target": targets["verifier_target"],
                "repair_target": targets["repair_target"],
                "label_source": row["label_source"],
                "exact_label_kind": row["exact_label"]["kind"],
                "source_prompt_payload_sha256": row["prompt_payload_sha256"],
                "leakage_safe_prompt_payload": row["leakage_safe_prompt_payload"],
                "evaluation_tasks": _evaluation_tasks(row),
                "stratum_key": "|".join(
                    [
                        str(row["family"]),
                        str(row["task_axis"]),
                        str(row["perturbation_family"]),
                        str(targets["expected_answer"]),
                    ]
                ),
            }
        )
    return sorted(output, key=lambda item: str(item["source_fixture_id"]))


def expected_targets(row: Mapping[str, Any]) -> JsonDict:
    """Derive exact post-generation targets from one fixture row."""

    label = row["exact_label"]
    family = str(row["family"])
    if family == "smt_constraints":
        expected_answer = "SAT" if bool(label.get("is_satisfiable")) else "UNSAT"
        solver_label = str(label.get("solver_status"))
        expected_action = "accept" if expected_answer == "SAT" else "reject"
        repair_target = {"applicable": False, "reason": "not_a_repair_fixture"}
    elif family == "arithmetic_code_assertions":
        assertion_passes = bool(label.get("assertion_passes"))
        expected_answer = "VALID" if assertion_passes else "INVALID"
        solver_label = "assertion_passes" if assertion_passes else "assertion_fails"
        expected_action = "accept" if assertion_passes else "reject"
        repair_target = {"applicable": False, "reason": "not_a_repair_fixture"}
    elif family == "repairable_invalid_candidates":
        candidate_valid = bool(label.get("candidate_valid"))
        repairable = bool(label.get("repairable"))
        if candidate_valid:
            expected_answer = "VALID"
            expected_action = "accept"
        elif repairable:
            expected_answer = "REPAIRABLE"
            expected_action = "reject"
        else:
            expected_answer = "UNREPAIRABLE"
            expected_action = "reject"
        solver_label = (
            "candidate_valid" if candidate_valid else "repairable" if repairable else "unrepairable"
        )
        repair_target = {
            "applicable": True,
            "candidate_valid": candidate_valid,
            "repairable": repairable,
            "failure_kind": label.get("failure_kind"),
            "repair_validation": label.get("repair_validation"),
        }
    else:
        raise ValueError(f"unknown fixture family: {family}")
    return {
        "expected_answer": expected_answer,
        "solver_label": solver_label,
        "verifier_target": {
            "expected_action": expected_action,
            "expected_reject": expected_action == "reject",
        },
        "repair_target": repair_target,
    }


def diagnose_exp3085_tiny_panel(
    *,
    exp3085: Mapping[str, Any],
    panel_rows: Sequence[Mapping[str, Any]],
    usable_fixture_count: int,
) -> JsonDict:
    """Explain the checked-in 9-case abstention panel using row-level evidence."""

    fixture_ids = sorted(
        {str(row.get("fixture_id")) for row in panel_rows if row.get("fixture_id")}
    )
    task_rows = [row for row in panel_rows if row.get("policy") == "task_abstention"]
    selected_family_counts = dict(
        sorted(Counter(str(row.get("family")) for row in task_rows).items())
    )
    prompt_rows_per_fixture = round(len(panel_rows) / len(fixture_ids), 6) if fixture_ids else 0.0
    unique_count = len(fixture_ids) or int(exp3085.get("exact_ground_truth_count") or 0)
    per_family_values = sorted(set(selected_family_counts.values()))
    if per_family_values == [3] and len(selected_family_counts) == 3:
        reason = (
            "Exp 3085 used 3 fixtures per family and two prompt-policy rows per fixture; "
            "the remaining exact fixtures were not selected, rather than rejected by the manifest."
        )
    elif unique_count:
        reason = (
            "Exp 3085 row evidence shows a bounded subset smaller than the checked-in fixture bank; "
            "the artifact does not record the sampling knob, so the exact cause is inferred from rows."
        )
    else:
        reason = (
            "Exp 3085 row evidence is unavailable; tiny-panel cause cannot be proven from rows."
        )
    return {
        "exact_ground_truth_count_reported": int(exp3085.get("exact_ground_truth_count") or 0),
        "panel_row_count_reported": int(exp3085.get("panel_row_count") or 0),
        "baseline_row_count_reported": int(exp3085.get("baseline_row_count") or 0),
        "task_abstention_row_count_reported": int(exp3085.get("task_abstention_row_count") or 0),
        "unique_exact_fixtures_in_transcript": unique_count,
        "prompt_policy_rows_per_fixture": prompt_rows_per_fixture,
        "selected_fixture_family_counts": selected_family_counts,
        "remaining_usable_fixture_count": max(0, usable_fixture_count - unique_count),
        "why_only_9_exact_cases": reason,
        "artifact_sampling_knob_recorded": "sample_per_family" in exp3085,
    }


def downstream_usage(
    *,
    usable_rows: Sequence[Mapping[str, Any]],
    minimum_live_eval_count: int,
) -> JsonDict:
    """Return concrete skip gates for `.289` live tasks."""

    total = len(usable_rows)
    repair_count = sum(1 for row in usable_rows if row.get("repair_target", {}).get("applicable"))
    actions = Counter(
        str(row.get("verifier_target", {}).get("expected_action")) for row in usable_rows
    )
    has_accept_reject_repair = (
        actions.get("accept", 0) > 0 and actions.get("reject", 0) > 0 and repair_count > 0
    )
    headline_ready = total >= minimum_live_eval_count
    formal_ready = repair_count >= MINIMUM_FORMAL_FEEDBACK_REPAIR_COUNT
    repair_ready = repair_count >= MINIMUM_REPAIR_GATING_COUNT
    return {
        "abstention_sota_panel_v3": {
            "consume": STRATIFIED_MANIFEST_REL_PATH.as_posix(),
            "minimum_unique_fixtures": minimum_live_eval_count,
            "recommended_unique_fixtures": total,
            "ready_for_headline": headline_ready,
            "honest_skip_when": (
                f"skip if fewer than {minimum_live_eval_count} unique exact fixtures are selected"
            ),
        },
        "verifier_calibration_v3": {
            "consume": STRATIFIED_MANIFEST_REL_PATH.as_posix(),
            "minimum_unique_fixtures": minimum_live_eval_count,
            "ready_for_headline": headline_ready,
            "honest_skip_when": (
                f"skip if fewer than {minimum_live_eval_count} unique exact fixtures are selected"
            ),
        },
        "formal_feedback_v2": {
            "consume": STRATIFIED_MANIFEST_REL_PATH.as_posix(),
            "minimum_unique_repair_fixtures": MINIMUM_FORMAL_FEEDBACK_REPAIR_COUNT,
            "available_repair_fixtures": repair_count,
            "ready_for_headline": formal_ready,
            "honest_skip_when": "skip if fewer than 18 repair-target fixtures are selected",
        },
        "repair_gating_v2": {
            "consume": STRATIFIED_MANIFEST_REL_PATH.as_posix(),
            "minimum_unique_repair_fixtures": MINIMUM_REPAIR_GATING_COUNT,
            "available_repair_fixtures": repair_count,
            "ready_for_headline": repair_ready,
            "honest_skip_when": "skip if fewer than 24 repair-target fixtures are selected",
        },
        "fr11_stress_v2": {
            "consume": STRATIFIED_MANIFEST_REL_PATH.as_posix(),
            "minimum_unique_fixtures": minimum_live_eval_count,
            "requires_accept_reject_repair_targets": True,
            "ready_for_headline": headline_ready and has_accept_reject_repair,
            "honest_skip_when": (
                "skip if fewer than 48 unique exact fixtures span accept, reject, and repair targets"
            ),
        },
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the terminal protocol artifact overstates readiness."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or substrate.get("no_live_llm_inference") is not True:
        raise ValueError("inference_substrate must declare no live model inference")
    if substrate.get("executes_models") is not False:
        raise ValueError("protocol audit must not execute models")
    if artifact.get("eval_protocol_ready") is True:
        if int(artifact.get("usable_fixture_count") or 0) < int(
            artifact.get("minimum_live_eval_count") or 0
        ):
            raise ValueError("ready protocol requires usable fixtures above minimum")
        if int(artifact.get("rejected_fixture_count") or 0) != len(
            artifact.get("rejected_fixtures", [])
        ):
            raise ValueError("rejected_fixture_count must match rejected_fixtures")
        if not str(artifact.get("honest_verdict", "")).startswith(SUCCESS_PREFIXES):
            raise ValueError("ready protocol honest_verdict must start with a success prefix")
    else:
        if not str(artifact.get("honest_verdict", "")).startswith(
            "blocked_exact_fixture_protocol_precondition_failed"
        ):
            raise ValueError("blocked protocol must disclose exact fixture precondition failure")


def source_artifacts(repo_root: Path) -> list[JsonDict]:
    """Return existence and checksum evidence for every input artifact."""

    rows = []
    for source_id, rel_path, role in SOURCE_REL_PATHS:
        path = repo_root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "role": role,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return rows


def inference_substrate() -> JsonDict:
    """Declare the audit substrate explicitly so downstream claims stay bounded."""

    return {
        "kind": "offline_checked_in_fixture_audit",
        "executes_models": False,
        "live_llm_calls": 0,
        "no_live_llm_inference": True,
        "executes_z3_for_fixture_validation": True,
        "executes_python_runtime_for_fixture_validation": True,
        "executes_json_parser_for_fixture_validation": True,
        "uses_checked_in_artifacts_only": True,
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_tasks(row: Mapping[str, Any]) -> list[str]:
    family = str(row["family"])
    tasks = ["abstention_sota_panel_v3", "verifier_calibration_v3", "fr11_stress_v2"]
    if family == "repairable_invalid_candidates":
        tasks.extend(["formal_feedback_v2", "repair_gating_v2"])
    return tasks


def _manifest_rows_have_required_fields(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(set(REQUIRED_MANIFEST_FIELDS) <= set(row) for row in rows)


def _honest_verdict(
    *,
    ready: bool,
    usable_fixture_count: int,
    rejected_fixture_count: int,
    minimum_live_eval_count: int,
) -> str:
    if ready:
        return (
            "complete: eval_protocol_ready=true; "
            f"usable_fixture_count={usable_fixture_count}; "
            f"rejected_fixture_count={rejected_fixture_count}; "
            f"minimum_live_eval_count={minimum_live_eval_count}"
        )
    return (
        "blocked_exact_fixture_protocol_precondition_failed: "
        f"usable_fixture_count={usable_fixture_count} below minimum_live_eval_count="
        f"{minimum_live_eval_count}"
    )


def _safe_load_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _safe_load_jsonl(path: Path) -> list[JsonDict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"malformed_json_line": line})
            continue
        rows.append(dict(value) if isinstance(value, Mapping) else {"non_object_row": value})
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:  # pragma: no cover - thin manual entrypoint.
    write_artifact()


if __name__ == "__main__":  # pragma: no cover
    main()
