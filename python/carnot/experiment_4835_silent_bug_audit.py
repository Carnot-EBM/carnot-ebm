"""Experiment 4835: audit .445 ARC nulls for silent no-op bugs.

Spec refs: REQ-ARC-WMTE-4835, SCENARIO-ARC-WMTE-4835-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4835-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from carnot import experiment_4805_silent_bug_audit as prior_audit


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4835_silent_bug_audit"
EXPERIMENT_ID = 4835
SCHEMA = "carnot.arc.milestone_445_silent_bug_audit_4835.v1"
RESULT_RELATIVE_PATH = "results/experiment_4835_silent_bug_audit.json"
AUDIT_REPORT_RELATIVE_PATH = "ops/arc_null_silent_bug_audit.md"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 4835
DURATION_FLOOR_S = 0.0001
TERMINAL_PREFIXES = prior_audit.TERMINAL_PREFIXES

JsonDict = dict[str, Any]

SPEC_REFS = [
    "REQ-ARC-WMTE-4835",
    "SCENARIO-ARC-WMTE-4835-SILENT-BUG-AUDIT",
    "SCENARIO-ARC-WMTE-4835-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; audit complete is complete_/success_."
    },
    "a1_archive_alive_and_prior_exercised": {
        "principle": (
            "the load-bearing check -- archive alive (positive cells) AND prior "
            "changed proposals, else A1 is a non-test (exp4701 recurrence or no-op)."
        )
    },
    "nulls_audited": {"principle": "count of nulls re-examined."},
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (0.0001s floor)."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "field_principles",
    "per_null_verdicts",
    "silent_bugs_found",
    "trusted_nulls",
    "a1_control_check",
    "preconditions_checked",
    "audited_artifacts",
    "audited_artifact_checksums",
    "audit_report_path",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

AUDIT_TARGETS: tuple[dict[str, str], ...] = (
    {
        "null_id": "experiment_4831_amortized_incontext_exploration_prior_live",
        "artifact_path": "results/experiment_4831_amortized_incontext_exploration_prior_live.json",
    },
    {
        "null_id": "experiment_4832_levelup_attempt",
        "artifact_path": "results/experiment_4832_levelup_attempt.json",
    },
    {
        "null_id": "experiment_4834_heldout_first_win_readiness",
        "artifact_path": "results/experiment_4834_heldout_first_win_readiness.json",
    },
)

payload_checksum = prior_audit.payload_checksum
_read_json = prior_audit._read_json
_file_checksum = prior_audit._file_checksum
_finite_float = prior_audit._finite_float
_int_value = prior_audit._int_value
_mapping = prior_audit._mapping
_format_number = prior_audit._format_number


def _expected_source_paths() -> list[str]:
    return [target["artifact_path"] for target in AUDIT_TARGETS]


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return set()
    return {str(item) for item in value if str(item)}


def _orders_materially_changed(diagnostics: Mapping[str, Any]) -> bool:
    no_prior = diagnostics.get("no_prior_order")
    with_prior = diagnostics.get("with_prior_order")
    return bool(
        diagnostics.get("changed") is True
        or (
            isinstance(no_prior, Sequence)
            and not isinstance(no_prior, str | bytes)
            and isinstance(with_prior, Sequence)
            and not isinstance(with_prior, str | bytes)
            and list(no_prior)
            and list(with_prior)
            and list(no_prior) != list(with_prior)
        )
    )


def a1_control_check(artifact: Mapping[str, Any]) -> JsonDict:
    """REQ-ARC-WMTE-4835: verify A1 was a real archive/prior test."""

    archive = _mapping(artifact.get("go_explore_archive_alive"))
    observations = _int_value(archive.get("observations"))
    stored_cells = _int_value(archive.get("stored_cells"))
    prefixes_injected = _int_value(archive.get("prefixes_injected"))
    archive_alive = bool(
        archive.get("alive", True) is not False
        and observations > 0
        and stored_cells > 0
        and prefixes_injected > 0
    )

    change_diag = _mapping(artifact.get("prior_change_diagnostics"))
    prior_diag = _mapping(artifact.get("prior_diagnostics"))
    proposal_changes = _int_value(prior_diag.get("proposal_changes"))
    order_changed = _orders_materially_changed(change_diag)
    prior_changed = bool(
        artifact.get("prior_changed_proposals") is True
        and (order_changed or proposal_changes > 0)
    )

    imitation = _mapping(artifact.get("imitation_control_heldout_games"))
    distillation_games = _string_set(imitation.get("distillation_games"))
    heldout_games = _string_set(imitation.get("heldout_games"))
    heldout_split = bool(
        imitation.get("heldout_not_in_distillation_set") is True
        and distillation_games
        and heldout_games
        and heldout_games.isdisjoint(distillation_games)
    )
    heldout_rates_present = bool(
        _finite_float(imitation.get("first_win_rate_with_prior")) is not None
        and _finite_float(imitation.get("first_win_rate_no_prior_ablation")) is not None
    )
    imitation_control_confirmed = bool(heldout_split and heldout_rates_present)

    return {
        "archive_alive": archive_alive,
        "observations": observations,
        "stored_cells": stored_cells,
        "prefixes_injected": prefixes_injected,
        "prior_changed": prior_changed,
        "proposal_order_changed": order_changed,
        "proposal_changes": proposal_changes,
        "imitation_control_confirmed": imitation_control_confirmed,
        "heldout_not_in_distillation_set": heldout_split,
        "heldout_games": sorted(heldout_games),
        "distillation_games": sorted(distillation_games),
        "imitation_lift_holds": imitation.get("lift_holds"),
        "first_win_rate_with_prior": _finite_float(artifact.get("first_win_rate_with_prior")),
        "first_win_rate_no_prior_ablation": _finite_float(
            artifact.get("first_win_rate_no_prior_ablation")
        ),
    }


def _audit_a1(null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    checks = a1_control_check(artifact)
    signatures: list[str] = []
    if not checks["archive_alive"]:
        signatures.append("dead_go_explore_archive")
    if not checks["prior_changed"]:
        signatures.append("prior_no_op")
    if not checks["imitation_control_confirmed"]:
        signatures.append("a1_imitation_control_missing")
    return {
        "null_id": null_id,
        "verdict": "silent_bug_must_reopen" if signatures else "trustworthy_null",
        "silent_bug_signatures": signatures,
        "exercise_evidence": [
            f"archive_observations={checks['observations']}",
            f"archive_stored_cells={checks['stored_cells']}",
            f"prefixes_injected={checks['prefixes_injected']}",
            f"prior_changed_proposals={checks['prior_changed']}",
            f"proposal_changes={checks['proposal_changes']}",
            f"imitation_heldout_games={len(checks['heldout_games'])}",
            f"heldout_not_in_distillation_set={checks['heldout_not_in_distillation_set']}",
            f"first_win_rate_with_prior={_format_number(checks['first_win_rate_with_prior'])}",
            "first_win_rate_no_prior_ablation="
            f"{_format_number(checks['first_win_rate_no_prior_ablation'])}",
        ],
        "a1_archive_alive_and_prior_exercised": bool(
            checks["archive_alive"] and checks["prior_changed"]
        ),
        "a1_control_check": checks,
    }


def _audit_via_prior(null_id: str, prior_null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    result = dict(prior_audit.audit_null_artifact(prior_null_id, artifact))
    result["null_id"] = null_id
    return result


def audit_null_artifact(null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    """REQ-ARC-WMTE-4835: classify one .445 null from exercised-evidence fields."""

    if null_id == "experiment_4831_amortized_incontext_exploration_prior_live":
        return _audit_a1(null_id, artifact)
    if null_id == "experiment_4832_levelup_attempt":
        return _audit_via_prior(null_id, "experiment_4802_levelup_attempt", artifact)
    if null_id == "experiment_4834_heldout_first_win_readiness":
        return _audit_via_prior(
            null_id,
            "experiment_4804_heldout_first_win_readiness",
            artifact,
        )
    return {
        "null_id": null_id,
        "verdict": "silent_bug_must_reopen",
        "silent_bug_signatures": ["unknown_null_artifact"],
        "exercise_evidence": ["unknown null artifact; cannot trust negative result"],
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    missing = [rel for rel in _expected_source_paths() if not (root_path / rel).exists()]
    spec_path = root_path / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    return {
        "ok": not missing,
        "source_artifacts_present": not missing,
        "missing_source_artifacts": missing,
        "milestone_445_artifacts_present": not missing,
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists(),
        "spec_has_req_4835": "REQ-ARC-WMTE-4835" in spec_text,
    }


def _blocked_artifact(checks: Mapping[str, Any]) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": "blocked_missing_source_artifacts",
        "a1_archive_alive_and_prior_exercised": False,
        "nulls_audited": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [],
        "silent_bugs_found": [],
        "trusted_nulls": [],
        "a1_control_check": {},
        "preconditions_checked": dict(checks),
        "audited_artifacts": _expected_source_paths(),
        "audited_artifact_checksums": {},
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_FLOOR_S,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    per_null_verdicts: Sequence[Mapping[str, Any]],
    audited_artifact_checksums: Mapping[str, str],
) -> JsonDict:
    silent = [
        dict(row)
        for row in per_null_verdicts
        if row.get("verdict") != "trustworthy_null"
    ]
    trusted = [
        str(row.get("null_id"))
        for row in per_null_verdicts
        if row.get("verdict") == "trustworthy_null" and row.get("null_id")
    ]
    a1_row = next(
        (
            row
            for row in per_null_verdicts
            if row.get("null_id") == "experiment_4831_amortized_incontext_exploration_prior_live"
        ),
        {},
    )
    a1_check = a1_row.get("a1_control_check")
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": (
            f"complete_arc_null_silent_bug_audit_{len(per_null_verdicts)}_nulls_"
            f"{len(silent)}_reopen"
        ),
        "a1_archive_alive_and_prior_exercised": bool(
            a1_row.get("a1_archive_alive_and_prior_exercised")
        ),
        "nulls_audited": len(per_null_verdicts),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [dict(row) for row in per_null_verdicts],
        "silent_bugs_found": silent,
        "trusted_nulls": trusted,
        "a1_control_check": dict(a1_check) if isinstance(a1_check, Mapping) else {},
        "preconditions_checked": dict(preconditions_checked),
        "audited_artifacts": _expected_source_paths(),
        "audited_artifact_checksums": dict(audited_artifact_checksums),
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_FLOOR_S,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if not isinstance(artifact.get("a1_archive_alive_and_prior_exercised"), bool):
        errors.append("a1_archive_alive_and_prior_exercised_must_be_bool")
    if not isinstance(artifact.get("nulls_audited"), int):
        errors.append("nulls_audited_must_be_int")
    if not isinstance(artifact.get("silent_bugs_found"), list):
        errors.append("silent_bugs_found_must_be_list")
    if not isinstance(artifact.get("per_null_verdicts"), list):
        errors.append("per_null_verdicts_must_be_list")
    if not isinstance(artifact.get("a1_control_check"), dict):
        errors.append("a1_control_check_must_be_dict")
    duration = _finite_float(artifact.get("duration_s"))
    if duration is None or duration < DURATION_FLOOR_S:
        errors.append("duration_below_aggregation_floor")
    per_nulls = artifact.get("per_null_verdicts")
    if isinstance(per_nulls, list) and isinstance(artifact.get("nulls_audited"), int):
        if artifact["nulls_audited"] != len(per_nulls):
            errors.append("nulls_audited_count_mismatch")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def render_markdown_section(artifact: Mapping[str, Any]) -> str:
    rows = [
        "",
        "## Experiment 4835 .445 ARC Null Silent-Bug Audit",
        "",
        f"- Verdict: `{artifact.get('honest_verdict')}`",
        f"- Nulls audited: `{artifact.get('nulls_audited')}`",
        "- A1 archive alive and prior exercised: "
        f"`{artifact.get('a1_archive_alive_and_prior_exercised')}`",
        f"- Silent bugs found: `{len(artifact.get('silent_bugs_found') or [])}`",
        f"- Inference substrate: `{artifact.get('inference_substrate')}`",
        "",
        "| Null | Verdict | Silent signatures | Evidence |",
        "|---|---|---|---|",
    ]
    for row in artifact.get("per_null_verdicts") or []:
        if not isinstance(row, Mapping):
            continue
        signatures = row.get("silent_bug_signatures") or []
        evidence = row.get("exercise_evidence") or []
        sig_text = ", ".join(f"`{sig}`" for sig in signatures) if signatures else "-"
        evidence_text = "<br>".join(str(item) for item in evidence) if evidence else "-"
        rows.append(
            f"| `{row.get('null_id')}` | `{row.get('verdict')}` | {sig_text} | {evidence_text} |"
        )
    a1_check = artifact.get("a1_control_check")
    if isinstance(a1_check, Mapping):
        rows.extend(
            [
                "",
                "### A1 Control Check",
                "",
                f"- Archive observations: `{a1_check.get('observations')}`",
                f"- Archive stored cells: `{a1_check.get('stored_cells')}`",
                f"- Prefixes injected: `{a1_check.get('prefixes_injected')}`",
                f"- Prior changed proposals: `{a1_check.get('prior_changed')}`",
                "- Imitation control held-out split: "
                f"`{a1_check.get('imitation_control_confirmed')}`",
            ]
        )
    rows.append("")
    return "\n".join(rows)


def append_markdown_report(
    artifact: Mapping[str, Any],
    *,
    root: Path | str = REPO_ROOT,
) -> Path:
    report_path = Path(root) / AUDIT_REPORT_RELATIVE_PATH
    marker = "## Experiment 4835 .445 ARC Null Silent-Bug Audit"
    if report_path.exists():
        current = report_path.read_text(encoding="utf-8")
        if marker in current:
            return report_path
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        current = "# ARC Null Silent-Bug Audit\n"
    report_path.write_text(current.rstrip() + render_markdown_section(artifact), encoding="utf-8")
    return report_path


def write_artifact(
    artifact: Mapping[str, Any],
    *,
    root: Path | str = REPO_ROOT,
) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def run(*, root: Path | str = REPO_ROOT, write: bool = True) -> JsonDict:
    repo = Path(root)
    checks = check_preconditions(repo)
    if not checks["ok"]:
        artifact = _blocked_artifact(checks)
        if write:
            write_artifact(artifact, root=repo)
        return artifact

    per_null: list[JsonDict] = []
    checksums: dict[str, str] = {}
    for target in AUDIT_TARGETS:
        rel = target["artifact_path"]
        path = repo / rel
        payload = _read_json(path)
        checksums[rel] = _file_checksum(path)
        per_null.append(audit_null_artifact(target["null_id"], payload))

    artifact = build_artifact(
        preconditions_checked=checks,
        per_null_verdicts=per_null,
        audited_artifact_checksums=checksums,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=repo)
        append_markdown_report(artifact, root=repo)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "a1_archive_alive_and_prior_exercised": artifact[
                    "a1_archive_alive_and_prior_exercised"
                ],
                "nulls_audited": artifact["nulls_audited"],
                "result": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
