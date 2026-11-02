"""Experiment 4775: audit .439 ARC nulls for silent no-op bugs.

Spec refs: REQ-ARC-WMTE-4775, SCENARIO-ARC-WMTE-4775-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4775-BLOCKED-PRECONDITION.
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

EXPERIMENT = "experiment_4775_silent_bug_audit"
SCHEMA = "carnot.arc.milestone_439_silent_bug_audit_4775.v1"
RESULT_RELATIVE_PATH = "results/experiment_4775_silent_bug_audit.json"
AUDIT_REPORT_RELATIVE_PATH = "ops/arc_null_silent_bug_audit.md"
S0PRIME_MODULE_RELATIVE_PATH = (
    "python/carnot/experiment_4771_structural_energy_s0prime_origin_matched.py"
)
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 4775
DURATION_FLOOR_S = 0.0001
TERMINAL_PREFIXES = ("complete_", "complete:", "success_", "success:", "blocked_", "blocked:")
S0_ORIGIN_LEAK_AUROC = 0.732793

JsonDict = dict[str, Any]

SPEC_REFS = [
    "REQ-ARC-WMTE-4775",
    "SCENARIO-ARC-WMTE-4775-SILENT-BUG-AUDIT",
    "SCENARIO-ARC-WMTE-4775-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; audit complete is complete_/success_."
    },
    "nulls_audited": {
        "principle": (
            "count of nulls re-examined -- a null that tested dead code is not trustworthy."
        )
    },
    "s0prime_leak_controls_fired": {
        "principle": (
            "the load-bearing check -- the origin-probe + shuffled-label controls must "
            "have genuinely executed, else S0''s reopen/retire verdict is uninterpretable."
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
    "silent_bugs_found",
    "trusted_nulls",
    "s0prime_leak_control_checks",
    "preconditions_checked",
    "audited_artifacts",
    "audited_artifact_checksums",
    "audit_report_path",
    "verifier_is_oracle",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

AUDIT_TARGETS: tuple[dict[str, str], ...] = (
    {
        "null_id": "experiment_4771_structural_energy_s0prime_origin_matched",
        "artifact_path": "results/experiment_4771_structural_energy_s0prime_origin_matched.json",
    },
    {
        "null_id": "experiment_4772_levelup_attempt",
        "artifact_path": "results/experiment_4772_levelup_attempt.json",
    },
    {
        "null_id": "experiment_4774_heldout_first_win_readiness",
        "artifact_path": "results/experiment_4774_heldout_first_win_readiness.json",
    },
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


def _with_generic_signatures(artifact: Mapping[str, Any], signatures: list[str]) -> None:
    if _contains_zero_cell_engine(artifact):
        _append_unique(signatures, "dead_identity_engine_zero_cell_changes")
    if _has_byte_identical_arms(artifact):
        _append_unique(signatures, "byte_identical_ab_arms")


def _classification(signatures: Sequence[str]) -> str:
    return "silent_bug_must_reopen" if signatures else "trustworthy_null"


def _per_game_loo_structural(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    per_game_loo = artifact.get("per_game_loo")
    if not isinstance(per_game_loo, Mapping):
        return {}
    structural = per_game_loo.get("structural")
    return structural if isinstance(structural, Mapping) else {}


def _s0prime_class_balance_checks(artifact: Mapping[str, Any]) -> JsonDict:
    balance = artifact.get("per_game_class_balance")
    if not isinstance(balance, Mapping) or not balance:
        return {
            "class_balance_non_degenerate": False,
            "contributing_games_with_both_classes": [],
            "contributing_games_missing_a_class": [],
            "single_class_games_skipped": [],
            "single_class_games_skipped_correctly": False,
        }

    loo_structural = _per_game_loo_structural(artifact)
    contributing: list[str] = []
    bad_contributing: list[str] = []
    skipped_single_class: list[str] = []
    bad_skips: list[str] = []
    for game, raw in sorted(balance.items()):
        if not isinstance(raw, Mapping):
            bad_contributing.append(str(game))
            continue
        correct = _int_value(raw.get("correct"))
        wrong = _int_value(raw.get("wrong"))
        contributes = raw.get("contributes_to_loo") is True
        has_both = correct > 0 and wrong > 0
        if contributes and has_both:
            contributing.append(str(game))
        elif contributes:
            bad_contributing.append(str(game))
        elif not has_both:
            skipped_single_class.append(str(game))
            loo_entry = loo_structural.get(game)
            skipped = isinstance(loo_entry, Mapping) and loo_entry.get("skipped") is True
            if not skipped:
                bad_skips.append(str(game))

    global_ok = _int_value(artifact.get("n_pos")) > 0 and _int_value(artifact.get("n_neg")) > 0
    skip_ok = not bad_skips
    return {
        "class_balance_non_degenerate": bool(global_ok and contributing and not bad_contributing),
        "contributing_games_with_both_classes": contributing,
        "contributing_games_missing_a_class": bad_contributing,
        "single_class_games_skipped": skipped_single_class,
        "single_class_games_not_skipped": bad_skips,
        "single_class_games_skipped_correctly": skip_ok,
    }


def s0prime_leak_control_checks(
    artifact: Mapping[str, Any],
    *,
    module_source: str = "",
) -> JsonDict:
    """REQ-ARC-WMTE-4775: decide if S0' leak controls genuinely fired."""

    n_rows = _int_value(artifact.get("n_candidate_rows"))
    origin_probe = artifact.get("origin_probe")
    origin_probe_map = origin_probe if isinstance(origin_probe, Mapping) else {}
    origin_counts = origin_probe_map.get("origin_counts")
    origin_counts_map = origin_counts if isinstance(origin_counts, Mapping) else {}
    origin_status = str(origin_probe_map.get("status") or "")
    origin_auroc = _finite_float(artifact.get("origin_probe_auroc"))
    induced_count = _int_value(origin_counts_map.get("induced"))
    dataset = artifact.get("dataset_diagnostics")
    dataset_map = dataset if isinstance(dataset, Mapping) else {}
    origin_matched_real = bool(
        dataset_map.get("origin_matched") is True
        and n_rows > 0
        and induced_count == n_rows
        and set(origin_counts_map) == {"induced"}
    )
    explicit_refit = origin_probe_map.get("refit_on_origin_matched_data") is True
    status_refit = "refit" in origin_status and "single_origin" not in origin_status
    origin_probe_refit = bool(origin_matched_real and (explicit_refit or status_refit))
    origin_probe_not_stale_s0 = bool(
        origin_auroc is not None
        and abs(origin_auroc - S0_ORIGIN_LEAK_AUROC) > 1e-6
        and origin_matched_real
    )

    shuffled_auroc = _finite_float(artifact.get("shuffled_label_control_auroc"))
    controls = artifact.get("controls")
    controls_map = controls if isinstance(controls, Mapping) else {}
    shuffled_resamples = _int_value(controls_map.get("shuffled_label_resamples"))
    module_has_permutation_loo = (
        "permutation(" in module_source and "_loo_metrics_candidate" in module_source
    )
    shuffled_reran = bool(
        shuffled_resamples > 0
        and shuffled_auroc is not None
        and abs(shuffled_auroc - 0.5) > 1e-9
        and module_has_permutation_loo
    )
    class_checks = _s0prime_class_balance_checks(artifact)
    controls_fired = bool(
        class_checks["class_balance_non_degenerate"]
        and class_checks["single_class_games_skipped_correctly"]
        and origin_matched_real
        and origin_probe_refit
        and origin_probe_not_stale_s0
        and shuffled_reran
    )
    return {
        **class_checks,
        "origin_matching_real": origin_matched_real,
        "origin_probe_status": origin_status,
        "origin_probe_refit_on_origin_matched_data": origin_probe_refit,
        "origin_probe_not_stale_s0_value": origin_probe_not_stale_s0,
        "origin_probe_auroc": origin_auroc,
        "shuffled_label_resamples": shuffled_resamples,
        "shuffled_label_control_auroc": shuffled_auroc,
        "shuffled_label_module_permutation_loo": module_has_permutation_loo,
        "shuffled_label_permuted_and_reran_loo": shuffled_reran,
        "all_controls_fired": controls_fired,
    }


def _audit_s0prime(
    null_id: str,
    artifact: Mapping[str, Any],
    *,
    module_source: str = "",
) -> JsonDict:
    checks = s0prime_leak_control_checks(artifact, module_source=module_source)
    signatures: list[str] = []
    if not checks["class_balance_non_degenerate"]:
        _append_unique(signatures, "s0prime_class_balance_degenerate")
    if not checks["single_class_games_skipped_correctly"]:
        _append_unique(signatures, "s0prime_single_class_games_not_skipped")
    if not checks["origin_matching_real"]:
        _append_unique(signatures, "s0prime_origin_matching_not_real")
    if not checks["origin_probe_refit_on_origin_matched_data"]:
        _append_unique(signatures, "s0prime_origin_probe_not_refit")
    if not checks["shuffled_label_permuted_and_reran_loo"]:
        _append_unique(signatures, "s0prime_shuffled_label_control_not_permuted_loo")
    pending = artifact.get("corrigendum_pending")
    if isinstance(pending, list) and pending:
        _append_unique(signatures, "s0prime_corrigendum_pending")
    _with_generic_signatures(artifact, signatures)

    evidence = [
        f"s0prime_candidate_rows={_int_value(artifact.get('n_candidate_rows'))}",
        f"s0prime_n_pos={_int_value(artifact.get('n_pos'))}",
        f"s0prime_n_neg={_int_value(artifact.get('n_neg'))}",
        "s0prime_contributing_games_with_both_classes="
        f"{len(checks['contributing_games_with_both_classes'])}",
        "s0prime_single_class_games_skipped="
        f"{len(checks['single_class_games_skipped'])}",
        f"origin_matched={checks['origin_matching_real']}",
        f"origin_probe_status={checks['origin_probe_status']}",
        "origin_probe_refit_on_origin_matched_data="
        f"{checks['origin_probe_refit_on_origin_matched_data']}",
        f"origin_probe_auroc={_format_number(checks['origin_probe_auroc'])}",
        "shuffled_label_control_auroc="
        f"{_format_number(checks['shuffled_label_control_auroc'])}",
        f"shuffled_label_resamples={checks['shuffled_label_resamples']}",
        "shuffled_label_module_permutation_loo="
        f"{checks['shuffled_label_module_permutation_loo']}",
    ]
    return {
        "null_id": null_id,
        "verdict": _classification(signatures),
        "silent_bug_signatures": signatures,
        "exercise_evidence": evidence,
        "s0prime_leak_controls_fired": checks["all_controls_fired"],
        "s0prime_leak_control_checks": checks,
    }


def _audit_levelup(null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    signatures: list[str] = []
    attempts = _list_of_mappings(artifact.get("attempted_games"))
    if not attempts:
        _append_unique(signatures, "levelup_attempts_missing")

    exercised = 0
    reproduced_existing = 0
    label_rows = 0
    for row in attempts:
        labels = row.get("solution_labels")
        label_count = len(labels) if isinstance(labels, list) else 0
        label_rows += label_count
        gate = row.get("reproduction_gate")
        gate_present = isinstance(gate, Mapping) and bool(gate)
        if label_count > 0 or gate_present:
            exercised += 1
        if row.get("offline_reproduced_existing_depth") is True and gate_present:
            reproduced_existing += 1
        if row.get("offline_reproduced_existing_depth") is True and not gate_present:
            _append_unique(signatures, "reproduction_gate_missing")

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
            f"solution_label_count={label_rows}",
            f"reproduced_existing={reproduced_existing}",
            f"new_levels_banked={_int_value(artifact.get('new_levels_banked'))}",
            f"reproduced_levels={_int_value(artifact.get('reproduced_levels'))}",
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
    flat_004 = (
        rate is not None
        and baseline is not None
        and abs(rate - baseline) <= 1e-12
        and abs(rate - 0.04) <= 1e-12
    )

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


def audit_null_artifact(
    null_id: str,
    artifact: Mapping[str, Any],
    *,
    module_source: str = "",
) -> JsonDict:
    """REQ-ARC-WMTE-4775: classify one .439 null from exercised-evidence fields."""

    if null_id == "experiment_4771_structural_energy_s0prime_origin_matched":
        return _audit_s0prime(null_id, artifact, module_source=module_source)
    if null_id == "experiment_4772_levelup_attempt":
        return _audit_levelup(null_id, artifact)
    if null_id == "experiment_4774_heldout_first_win_readiness":
        return _audit_heldout(null_id, artifact)
    return {
        "null_id": null_id,
        "verdict": "silent_bug_must_reopen",
        "silent_bug_signatures": ["unknown_null_artifact"],
        "exercise_evidence": ["unknown null artifact; cannot trust negative result"],
    }


def _expected_source_paths() -> list[str]:
    return [target["artifact_path"] for target in AUDIT_TARGETS]


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    missing = [rel for rel in _expected_source_paths() if not (root_path / rel).exists()]
    module_present = (root_path / S0PRIME_MODULE_RELATIVE_PATH).exists()
    return {
        "ok": not missing,
        "source_artifacts_present": not missing,
        "missing_source_artifacts": missing,
        "milestone_439_artifacts_present": not missing,
        "s0prime_module_present": module_present,
        "s0prime_module_path": S0PRIME_MODULE_RELATIVE_PATH,
    }


def _floored_duration(duration_s: float) -> float:
    return round(max(DURATION_FLOOR_S, float(duration_s)), 6)


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": "blocked_missing_source_artifacts",
        "nulls_audited": 0,
        "s0prime_leak_controls_fired": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [],
        "silent_bugs_found": [],
        "trusted_nulls": [],
        "s0prime_leak_control_checks": {},
        "preconditions_checked": dict(checks),
        "audited_artifacts": _expected_source_paths(),
        "audited_artifact_checksums": {},
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
    s0prime_row = next(
        (
            row
            for row in per_null_verdicts
            if row.get("null_id") == "experiment_4771_structural_energy_s0prime_origin_matched"
        ),
        {},
    )
    s0prime_checks = s0prime_row.get("s0prime_leak_control_checks")
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
        "s0prime_leak_controls_fired": bool(
            s0prime_row.get("s0prime_leak_controls_fired")
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [dict(row) for row in per_null_verdicts],
        "silent_bugs_found": silent,
        "trusted_nulls": trusted,
        "s0prime_leak_control_checks": (
            dict(s0prime_checks) if isinstance(s0prime_checks, Mapping) else {}
        ),
        "preconditions_checked": dict(preconditions_checked),
        "audited_artifacts": _expected_source_paths(),
        "audited_artifact_checksums": dict(audited_artifact_checksums),
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
    if not isinstance(artifact.get("s0prime_leak_controls_fired"), bool):
        errors.append("s0prime_leak_controls_fired_must_be_bool")
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
    if verdict.startswith(("complete_", "success_")) and isinstance(
        artifact.get("per_null_verdicts"), list
    ):
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
        "## Experiment 4775 .439 ARC Null Silent-Bug Audit",
        "",
        f"- Verdict: `{artifact.get('honest_verdict')}`",
        f"- Nulls audited: `{artifact.get('nulls_audited')}`",
        f"- S0' leak controls fired: `{artifact.get('s0prime_leak_controls_fired')}`",
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
            f"| `{verdict.get('null_id')}` | `{verdict.get('verdict')}` | "
            f"{sig_text} | {evidence_text} |"
        )
    rows.append("")
    return "\n".join(rows)


def append_markdown_report(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / AUDIT_REPORT_RELATIVE_PATH
    existing = path.read_text(encoding="utf-8") if path.exists() else "# ARC Null Silent-Bug Audit\n"
    marker = "## Experiment 4775 .439 ARC Null Silent-Bug Audit"
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

    module_path = root_path / S0PRIME_MODULE_RELATIVE_PATH
    module_source = module_path.read_text(encoding="utf-8") if module_path.exists() else ""
    checksums: dict[str, str] = {}
    verdicts: list[JsonDict] = []
    for target in AUDIT_TARGETS:
        rel = target["artifact_path"]
        path = root_path / rel
        payload = _read_json(path)
        checksums[rel] = _file_checksum(path)
        verdict = audit_null_artifact(
            target["null_id"],
            payload,
            module_source=module_source,
        )
        verdict["artifact_path"] = rel
        verdicts.append(verdict)
    if module_path.exists():
        checksums[S0PRIME_MODULE_RELATIVE_PATH] = _file_checksum(module_path)

    artifact = build_artifact(
        preconditions_checked=checks,
        per_null_verdicts=verdicts,
        audited_artifact_checksums=checksums,
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
