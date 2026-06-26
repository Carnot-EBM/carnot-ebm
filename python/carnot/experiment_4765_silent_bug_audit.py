"""Experiment 4765: audit .438 ARC nulls for silent no-op bugs.

Spec refs: REQ-ARC-WMTE-4765, SCENARIO-ARC-WMTE-4765-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4765-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

EXPERIMENT = "experiment_4765_silent_bug_audit"
SCHEMA = "carnot.arc.milestone_438_silent_bug_audit_4765.v1"
RESULT_RELATIVE_PATH = "results/experiment_4765_silent_bug_audit.json"
AUDIT_REPORT_RELATIVE_PATH = "ops/arc_null_silent_bug_audit.md"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 4765
DURATION_FLOOR_S = 0.0001
TERMINAL_PREFIXES = ("complete_", "complete:", "success_", "success:", "blocked_", "blocked:")

JsonDict = dict[str, Any]

SPEC_REFS = [
    "REQ-ARC-WMTE-4765",
    "SCENARIO-ARC-WMTE-4765-SILENT-BUG-AUDIT",
    "SCENARIO-ARC-WMTE-4765-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; audit complete is complete_/success_."
    },
    "nulls_audited": {
        "principle": (
            "count of nulls re-examined -- a null that tested dead code is not a "
            "trustworthy null."
        )
    },
    "silent_bugs_found": {
        "principle": (
            "the load-bearing output -- which nulls must be reopened vs trusted."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts; 0.0001s floor."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "spec_refs",
    "field_principles",
    "per_null_verdicts",
    "trusted_nulls",
    "preconditions_checked",
    "audited_artifacts",
    "audited_artifact_checksums",
    "prior_audit_context",
    "audit_report_path",
    "verifier_is_oracle",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

AUDIT_TARGETS: tuple[dict[str, str], ...] = (
    {
        "null_id": "experiment_4761_structural_energy_s0_core_bet_probe",
        "artifact_path": "results/experiment_4761_structural_energy_s0_core_bet_probe.json",
    },
    {
        "null_id": "experiment_4762_levelup_attempt",
        "artifact_path": "results/experiment_4762_levelup_attempt.json",
    },
    {
        "null_id": "experiment_4764_heldout_first_win_readiness",
        "artifact_path": "results/experiment_4764_heldout_first_win_readiness.json",
    },
)

PRIOR_AUDIT_PATHS = (
    "results/experiment_4725_silent_bug_audit.json",
    "results/experiment_4755_silent_bug_audit.json",
)

ZERO_CELL_KEYS = {
    "engine_cell_changes",
    "engine_changed_cells",
    "cells_changed_by_engine",
    "candidate_changed_cells",
    "transition_changed_cells",
}

ARM_LABEL_KEYS = {"arm", "name", "label", "variant"}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_stable_json(value).encode("utf-8"))


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _sha256_json(payload)


def _read_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _file_checksum(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_value(value: Any) -> int:
    parsed = _finite_float(value)
    return int(parsed) if parsed is not None else 0


def _format_number(value: float | None) -> str:
    return "None" if value is None else f"{value:g}"


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _append_unique(rows: list[str], text: str) -> None:
    if text and text not in rows:
        rows.append(text)


def _contains_zero_cell_engine(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, raw in value.items():
            if str(key) in ZERO_CELL_KEYS and _finite_float(raw) == 0.0:
                return True
        return any(_contains_zero_cell_engine(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_zero_cell_engine(child) for child in value)
    return False


def _normalised_arm(row: Mapping[str, Any]) -> JsonDict:
    return {str(key): value for key, value in row.items() if str(key) not in ARM_LABEL_KEYS}


def _has_byte_identical_arms(value: Any) -> bool:
    if isinstance(value, Mapping):
        arms = _list_of_mappings(value.get("arms"))
        if len(arms) >= 2 and _normalised_arm(arms[0]) == _normalised_arm(arms[1]):
            return True
        return any(_has_byte_identical_arms(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_byte_identical_arms(child) for child in value)
    return False


def _classification(signatures: Sequence[str]) -> str:
    return "silent_bug_must_reopen" if signatures else "trustworthy_null"


def _with_generic_signatures(artifact: Mapping[str, Any], signatures: list[str]) -> None:
    if _contains_zero_cell_engine(artifact):
        _append_unique(signatures, "dead_identity_engine_zero_cell_changes")
    if _has_byte_identical_arms(artifact):
        _append_unique(signatures, "byte_identical_ab_arms")


def _audit_s0(null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    signatures: list[str] = []
    evidence: list[str] = []
    checks = artifact.get("preconditions_checked")
    checked = checks if isinstance(checks, Mapping) else {}
    candidate_rows = _int_value(artifact.get("n_candidate_rows") or checked.get("candidate_rows"))
    origin_rows = _int_value(artifact.get("n_origin_probe_rows") or checked.get("origin_probe_rows"))
    origin_probe = _finite_float(artifact.get("origin_probe_auroc"))
    near_miss = _finite_float(artifact.get("near_miss_negative_fraction"))
    in_sample = _finite_float(artifact.get("in_sample_auroc"))

    _append_unique(evidence, f"s0_candidate_rows={candidate_rows}")
    _append_unique(evidence, f"origin_probe_rows={origin_rows}")
    _append_unique(evidence, f"origin_probe_auroc={_format_number(origin_probe)}")
    _append_unique(evidence, f"near_miss_negative_fraction={_format_number(near_miss)}")
    _append_unique(evidence, f"in_sample_auroc={_format_number(in_sample)}")

    if candidate_rows <= 0:
        _append_unique(signatures, "s0_candidate_rows_missing")
    if origin_rows <= 0 or origin_probe is None:
        _append_unique(signatures, "s0_origin_probe_not_run")
    elif origin_probe >= 0.6:
        _append_unique(signatures, "s0_origin_probe_leak")
    if near_miss is None or near_miss <= 0.0:
        _append_unique(signatures, "s0_near_miss_negatives_missing")
    if in_sample is None or in_sample <= 0.6:
        _append_unique(signatures, "s0_positive_control_not_exercised")
    _with_generic_signatures(artifact, signatures)

    return {
        "null_id": null_id,
        "verdict": _classification(signatures),
        "silent_bug_signatures": signatures,
        "exercise_evidence": evidence,
    }


def _is_timed_no_gate(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("residual_cause") or "") == "time_budget_no_terminal_gate"
        and (_finite_float(row.get("elapsed_s")) or 0.0) > 0.0
    )


def _audit_levelup(null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    signatures: list[str] = []
    attempts = _list_of_mappings(artifact.get("attempted_games"))
    if not attempts:
        _append_unique(signatures, "levelup_attempts_missing")

    exercised = 0
    reproduced_existing = 0
    timed_no_gate = 0
    for row in attempts:
        labels = row.get("solution_labels")
        label_count = len(labels) if isinstance(labels, list) else 0
        gate = row.get("reproduction_gate")
        gate_present = isinstance(gate, Mapping) and bool(gate)
        timed = _is_timed_no_gate(row)
        if label_count > 0 or gate_present or timed:
            exercised += 1
        if row.get("offline_reproduced_existing_depth") is True and gate_present:
            reproduced_existing += 1
        if row.get("offline_reproduced_existing_depth") is True and not gate_present:
            _append_unique(signatures, "reproduction_gate_missing")
        if timed:
            timed_no_gate += 1

    if attempts and exercised == 0:
        _append_unique(signatures, "levelup_mechanism_not_exercised")
    _with_generic_signatures(artifact, signatures)

    return {
        "null_id": null_id,
        "verdict": _classification(signatures),
        "silent_bug_signatures": signatures,
        "exercise_evidence": [
            f"levelup_attempts={len(attempts)}",
            f"exercised_attempts={exercised}",
            f"reproduced_existing={reproduced_existing}",
            f"timed_no_gate={timed_no_gate}",
        ],
    }


def _audit_heldout(null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    signatures: list[str] = []
    rate = _finite_float(artifact.get("heldout_first_win_rate"))
    baseline = _finite_float(artifact.get("first_win_baseline"))
    attempts = _int_value(artifact.get("heldout_variant_attempts"))
    positive_control = artifact.get("positive_control_passed") is True
    parity_green = artifact.get("parity_test_green") is True
    note = str(artifact.get("null_delta_methodology_note") or "")
    flat_004 = rate is not None and baseline is not None and abs(rate - baseline) <= 1e-12 and rate == 0.04

    if attempts < 100:
        _append_unique(signatures, "heldout_attempt_floor_not_met")
    if flat_004 and not note:
        _append_unique(signatures, "first_win_0_04_tautology_unannotated")
    if flat_004 and not positive_control:
        _append_unique(signatures, "first_win_positive_control_missing")
    if not parity_green:
        _append_unique(signatures, "parity_test_not_green")
    _with_generic_signatures(artifact, signatures)

    return {
        "null_id": null_id,
        "verdict": _classification(signatures),
        "silent_bug_signatures": signatures,
        "exercise_evidence": [
            f"heldout_first_win_rate={_format_number(rate)}",
            f"first_win_baseline={_format_number(baseline)}",
            f"heldout_attempts={attempts}",
            f"positive_control_passed={positive_control}",
            f"parity_test_green={parity_green}",
            f"null_delta_methodology_note_present={bool(note)}",
        ],
    }


def audit_null_artifact(null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    """REQ-ARC-WMTE-4765: classify one .438 null from exercised-evidence fields."""

    if null_id == "experiment_4761_structural_energy_s0_core_bet_probe":
        result = _audit_s0(null_id, artifact)
    elif null_id == "experiment_4762_levelup_attempt":
        result = _audit_levelup(null_id, artifact)
    elif null_id == "experiment_4764_heldout_first_win_readiness":
        result = _audit_heldout(null_id, artifact)
    else:
        result = {
            "null_id": null_id,
            "verdict": "silent_bug_must_reopen",
            "silent_bug_signatures": ["unknown_null_artifact"],
            "exercise_evidence": ["unknown null artifact; cannot trust negative result"],
        }
    return result


def _expected_source_paths() -> list[str]:
    return [target["artifact_path"] for target in AUDIT_TARGETS] + list(PRIOR_AUDIT_PATHS)


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    milestone_paths = [target["artifact_path"] for target in AUDIT_TARGETS]
    prior_paths = list(PRIOR_AUDIT_PATHS)
    missing = [rel for rel in milestone_paths + prior_paths if not (root_path / rel).exists()]
    return {
        "ok": not missing,
        "source_artifacts_present": not missing,
        "missing_source_artifacts": missing,
        "milestone_438_artifacts_present": not any(rel in missing for rel in milestone_paths),
        "prior_audits_present": not any(rel in missing for rel in prior_paths),
    }


def _floored_duration(duration_s: float) -> float:
    return round(max(DURATION_FLOOR_S, float(duration_s)), 6)


def _prior_audit_context(prior_4725: Mapping[str, Any], prior_4755: Mapping[str, Any]) -> JsonDict:
    silent_4725 = prior_4725.get("silent_bug_nulls")
    reopen_4755 = prior_4755.get("must_reopen")
    return {
        "experiment_4725": {
            "nulls_audited": _int_value(prior_4725.get("nulls_audited")),
            "silent_bug_null_count": len(silent_4725) if isinstance(silent_4725, list) else 0,
            "go_explore_fix_confirmed": prior_4725.get("go_explore_fix_confirmed") is True,
        },
        "experiment_4755": {
            "levers_audited": len(prior_4755.get("levers_audited") or []),
            "must_reopen": list(reopen_4755) if isinstance(reopen_4755, list) else [],
        },
    }


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": "blocked_missing_source_artifacts",
        "nulls_audited": 0,
        "silent_bugs_found": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [],
        "trusted_nulls": [],
        "preconditions_checked": dict(checks),
        "audited_artifacts": _expected_source_paths(),
        "audited_artifact_checksums": {},
        "prior_audit_context": {},
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "duration_s": _floored_duration(duration_s),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    per_null_verdicts: Sequence[Mapping[str, Any]],
    audited_artifact_checksums: Mapping[str, str],
    prior_audit_context: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    silent = [
        dict(row)
        for row in per_null_verdicts
        if row.get("verdict") == "silent_bug_must_reopen"
    ]
    trusted = [
        str(row.get("null_id"))
        for row in per_null_verdicts
        if row.get("verdict") == "trustworthy_null" and row.get("null_id")
    ]
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": (
            f"complete_arc_null_silent_bug_audit_{len(per_null_verdicts)}_nulls_"
            f"{len(silent)}_reopen"
        ),
        "nulls_audited": len(per_null_verdicts),
        "silent_bugs_found": silent,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [dict(row) for row in per_null_verdicts],
        "trusted_nulls": trusted,
        "preconditions_checked": dict(preconditions_checked),
        "audited_artifacts": _expected_source_paths(),
        "audited_artifact_checksums": dict(audited_artifact_checksums),
        "prior_audit_context": dict(prior_audit_context),
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "duration_s": _floored_duration(duration_s),
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
    if not isinstance(artifact.get("nulls_audited"), int):
        errors.append("nulls_audited_must_be_int")
    if not isinstance(artifact.get("silent_bugs_found"), list):
        errors.append("silent_bugs_found_must_be_list")
    if not isinstance(artifact.get("per_null_verdicts"), list):
        errors.append("per_null_verdicts_must_be_list")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    duration = _finite_float(artifact.get("duration_s"))
    if duration is None or duration < DURATION_FLOOR_S:
        errors.append("duration_below_aggregation_floor")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if verdict.startswith(("complete_", "success_")) and isinstance(artifact.get("per_null_verdicts"), list):
        if artifact.get("nulls_audited") != len(artifact["per_null_verdicts"]):
            errors.append("nulls_audited_does_not_match_verdicts")
    return errors


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_markdown_section(artifact: Mapping[str, Any]) -> str:
    rows = [
        "",
        "## Experiment 4765 .438 ARC Null Silent-Bug Audit",
        "",
        f"- Verdict: `{artifact.get('honest_verdict')}`",
        f"- Nulls audited: `{artifact.get('nulls_audited')}`",
        f"- Silent bugs found: `{len(artifact.get('silent_bugs_found') or [])}`",
        f"- Inference substrate: `{artifact.get('inference_substrate')}`",
        "",
        "| Null | Verdict | Silent signatures | Evidence |",
        "|---|---|---|---|",
    ]
    for verdict in artifact.get("per_null_verdicts") or []:
        if not isinstance(verdict, Mapping):
            continue
        signatures = verdict.get("silent_bug_signatures") or []
        evidence = verdict.get("exercise_evidence") or []
        sig_text = ", ".join(f"`{item}`" for item in signatures) if signatures else "-"
        evidence_text = "<br>".join(str(item) for item in evidence) if evidence else "-"
        rows.append(
            f"| `{verdict.get('null_id')}` | `{verdict.get('verdict')}` | {sig_text} | {evidence_text} |"
        )
    rows.append("")
    return "\n".join(rows)


def append_markdown_report(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / AUDIT_REPORT_RELATIVE_PATH
    existing = path.read_text(encoding="utf-8") if path.exists() else "# ARC Null Silent-Bug Audit\n"
    marker = "## Experiment 4765 .438 ARC Null Silent-Bug Audit"
    if marker in existing:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing.rstrip() + "\n" + render_markdown_section(artifact), encoding="utf-8")
    return path


def run(root: Path | str = REPO_ROOT, *, write: bool = True) -> JsonDict:
    started = time.monotonic()
    root_path = Path(root)
    checks = check_preconditions(root_path)
    if not checks.get("ok"):
        artifact = _blocked_artifact(checks, time.monotonic() - started)
        if write:
            write_artifact(artifact, root=root_path)
        return artifact

    checksums: dict[str, str] = {}
    verdicts: list[JsonDict] = []
    for target in AUDIT_TARGETS:
        rel = target["artifact_path"]
        path = root_path / rel
        payload = _read_json(path)
        checksums[rel] = _file_checksum(path)
        verdict = audit_null_artifact(target["null_id"], payload)
        verdict["artifact_path"] = rel
        verdicts.append(verdict)

    prior_4725 = _read_json(root_path / PRIOR_AUDIT_PATHS[0])
    prior_4755 = _read_json(root_path / PRIOR_AUDIT_PATHS[1])
    for rel in PRIOR_AUDIT_PATHS:
        checksums[rel] = _file_checksum(root_path / rel)

    artifact = build_artifact(
        preconditions_checked=checks,
        per_null_verdicts=verdicts,
        audited_artifact_checksums=checksums,
        prior_audit_context=_prior_audit_context(prior_4725, prior_4755),
        duration_s=time.monotonic() - started,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=root_path)
        append_markdown_report(artifact, root=root_path)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not artifact_schema_errors(artifact) else 1


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
