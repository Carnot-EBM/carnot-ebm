"""Experiment 4795: audit .441 ARC nulls for silent no-op bugs.

Spec refs: REQ-ARC-WMTE-4795, SCENARIO-ARC-WMTE-4795-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4795-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

EXPERIMENT = "experiment_4795_silent_bug_audit"
EXPERIMENT_ID = 4795
SCHEMA = "carnot.arc.milestone_441_silent_bug_audit_4795.v1"
RESULT_RELATIVE_PATH = "results/experiment_4795_silent_bug_audit.json"
AUDIT_REPORT_RELATIVE_PATH = "ops/arc_null_silent_bug_audit.md"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 4795
DURATION_FLOOR_S = 0.0001
EXPECTED_BINARY_SELECTION_RULE = (
    "first_candidate_with_WorldModelVerifier_exact_accuracy_ge_0.5_else_first"
)
TERMINAL_PREFIXES = (
    "complete_",
    "complete:",
    "success_",
    "success:",
    "passed_",
    "passed:",
    "shipped_",
    "shipped:",
    "blocked_",
    "blocked:",
)

JsonDict = dict[str, Any]
LintRunner = Callable[[Path], Mapping[str, Any]]

SPEC_REFS = [
    "REQ-ARC-WMTE-4795",
    "SCENARIO-ARC-WMTE-4795-SILENT-BUG-AUDIT",
    "SCENARIO-ARC-WMTE-4795-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; audit complete is complete_/success_."
    },
    "nulls_audited": {"principle": "count of nulls re-examined."},
    "s2_live_path_reachable_confirmed": {
        "principle": (
            "the load-bearing check -- S2's energy gate must be reachable from "
            "E3AgentPolicy, else its 'live trust gate' claim is hollow."
        )
    },
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
    "s2_control_check",
    "arc_orphan_solver_lint",
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
        "null_id": "experiment_4791_structural_energy_s2_offpath_trust_gate",
        "artifact_path": "results/experiment_4791_structural_energy_s2_offpath_trust_gate.json",
    },
    {
        "null_id": "experiment_4792_levelup_attempt",
        "artifact_path": "results/experiment_4792_levelup_attempt.json",
    },
    {
        "null_id": "experiment_4794_heldout_first_win_readiness",
        "artifact_path": "results/experiment_4794_heldout_first_win_readiness.json",
    },
)


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _append_unique(rows: list[str], text: str) -> None:
    if text and text not in rows:
        rows.append(text)


def _format_number(value: float | None) -> str:
    return "None" if value is None else f"{value:g}"


def _expected_source_paths() -> list[str]:
    return [target["artifact_path"] for target in AUDIT_TARGETS]


def _tail(text: str, *, limit: int = 2000) -> str:
    return text[-limit:] if len(text) > limit else text


def run_arc_orphan_solver_lint(root: Path) -> JsonDict:
    """REQ-ARC-WMTE-4795: run the live-path orphan lint for the S2 graft."""

    command = [sys.executable, "scripts/arc_orphan_solver_lint.py"]
    proc = subprocess.run(
        command,
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return {
        "command": " ".join(command),
        "passed": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }


def _e3_policy_imports_trust_energy(root: Path) -> bool:
    path = root / "python/carnot/agentic/arc_competition_agent.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "arc_world_model_trust_energy" in text and "select_trusted_world_model" in text


def _binary_control_real(artifact: Mapping[str, Any]) -> bool:
    control = _mapping(artifact.get("binary_accuracy_control"))
    threshold = _finite_float(control.get("threshold"))
    return bool(
        control.get("verifier_is_oracle") is True
        and control.get("selection_rule") == EXPECTED_BINARY_SELECTION_RULE
        and threshold is not None
        and abs(threshold - 0.5) <= 1e-12
    )


def _candidate_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("candidate_name")) for row in rows if row.get("candidate_name")]


def s2_control_check(
    artifact: Mapping[str, Any],
    *,
    lint_result: Mapping[str, Any],
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    """REQ-ARC-WMTE-4795: reconstruct S2's live path and binary gate control."""

    repo = Path(root)
    candidate_sets = _mapping(artifact.get("candidate_sets"))
    game_results = _list_of_mappings(artifact.get("game_results"))
    threshold = _finite_float(_mapping(artifact.get("binary_accuracy_control")).get("threshold"))
    threshold = 0.5 if threshold is None else threshold

    identical_sets = bool(game_results)
    selection_reconstructed = bool(game_results)
    rows_present = bool(game_results)
    finite_energy_scores = bool(game_results)
    checked_games = 0
    reconstructed: list[JsonDict] = []
    mismatches: list[str] = []

    for game_row in game_results:
        game = str(game_row.get("game") or "")
        rows = _list_of_mappings(game_row.get("candidate_rows"))
        if not rows:
            rows_present = False
            selection_reconstructed = False
            mismatches.append(f"{game}:candidate_rows_missing")
            continue
        checked_games += 1
        row_names = _candidate_names(rows)
        declared = candidate_sets.get(game)
        if not isinstance(declared, list) or [str(name) for name in declared] != row_names:
            identical_sets = False
            mismatches.append(f"{game}:candidate_set_mismatch")

        passing = [row for row in rows if row.get("binary_gate_pass") is True]
        expected = passing[0] if passing else rows[0]
        expected_name = str(expected.get("candidate_name") or "")
        actual_name = str(game_row.get("accuracy_gate_selected_candidate") or "")
        exact_accuracy = _finite_float(game_row.get("accuracy_gate_exact_accuracy"))
        exact_passed = bool(exact_accuracy is not None and exact_accuracy >= threshold)
        if actual_name != expected_name:
            selection_reconstructed = False
            mismatches.append(f"{game}:selected={actual_name}:expected={expected_name}")
        if game_row.get("accuracy_gate_passed") is not exact_passed:
            selection_reconstructed = False
            mismatches.append(f"{game}:accuracy_gate_passed_mismatch")
        energy_name = str(game_row.get("energy_selected_candidate") or "")
        if energy_name not in row_names:
            selection_reconstructed = False
            mismatches.append(f"{game}:energy_selected_not_in_candidate_set")
        if any(_finite_float(row.get("offpath_structural_energy")) is None for row in rows):
            finite_energy_scores = False
            mismatches.append(f"{game}:missing_offpath_structural_energy")
        reconstructed.append(
            {
                "game": game,
                "declared_candidate_count": len(declared) if isinstance(declared, list) else 0,
                "row_candidate_count": len(row_names),
                "expected_binary_selection": expected_name,
                "actual_binary_selection": actual_name,
                "binary_any_candidate_passed": bool(passing),
            }
        )

    min_games = _int_value(artifact.get("min_heldout_games")) or 5
    n_heldout = _int_value(artifact.get("n_heldout_games"))
    lint_passed = lint_result.get("passed") is True
    artifact_live = artifact.get("live_path_reachable") is True
    preconditions = _mapping(artifact.get("preconditions_checked"))
    artifact_lint_passed = preconditions.get("arc_orphan_solver_lint_passed") is True
    e3_import = _e3_policy_imports_trust_energy(repo)
    live_confirmed = bool(lint_passed and artifact_live and artifact_lint_passed and e3_import)
    real_control = _binary_control_real(artifact)

    return {
        "s2_live_path_reachable_confirmed": live_confirmed,
        "lint_passed": lint_passed,
        "artifact_live_path_reachable": artifact_live,
        "artifact_lint_precondition_passed": artifact_lint_passed,
        "e3_policy_imports_trust_energy": e3_import,
        "real_binary_accuracy_control": real_control,
        "identical_candidate_sets": identical_sets,
        "binary_selection_reconstructed": selection_reconstructed,
        "candidate_rows_present": rows_present,
        "finite_offpath_energy_scores": finite_energy_scores,
        "checked_games": checked_games,
        "n_heldout_games": n_heldout,
        "min_heldout_games": min_games,
        "heldout_game_floor_met": n_heldout >= min_games and checked_games >= min_games,
        "reconstructed_binary_selections": reconstructed,
        "mismatches": mismatches,
    }


def _audit_s2(
    null_id: str,
    artifact: Mapping[str, Any],
    *,
    lint_result: Mapping[str, Any],
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    checks = s2_control_check(artifact, lint_result=lint_result, root=root)
    signatures: list[str] = []
    if not checks["s2_live_path_reachable_confirmed"]:
        _append_unique(signatures, "s2_energy_gate_not_live_path_reachable")
    if not checks["real_binary_accuracy_control"]:
        _append_unique(signatures, "s2_binary_control_not_real_incumbent")
    if not checks["identical_candidate_sets"]:
        _append_unique(signatures, "s2_candidate_sets_not_identical")
    if not checks["binary_selection_reconstructed"]:
        _append_unique(signatures, "s2_binary_selection_not_reconstructed")
    if not checks["candidate_rows_present"]:
        _append_unique(signatures, "s2_candidate_rows_missing")
    if not checks["finite_offpath_energy_scores"]:
        _append_unique(signatures, "s2_energy_scores_missing")
    if not checks["heldout_game_floor_met"]:
        _append_unique(signatures, "s2_heldout_game_floor_not_met")
    return {
        "null_id": null_id,
        "verdict": "silent_bug_must_reopen" if signatures else "trustworthy_null",
        "silent_bug_signatures": signatures,
        "exercise_evidence": [
            f"lint_passed={checks['lint_passed']}",
            f"artifact_live_path_reachable={checks['artifact_live_path_reachable']}",
            "artifact_lint_precondition_passed="
            f"{checks['artifact_lint_precondition_passed']}",
            f"e3_policy_imports_trust_energy={checks['e3_policy_imports_trust_energy']}",
            f"real_binary_accuracy_control={checks['real_binary_accuracy_control']}",
            f"identical_candidate_sets={checks['identical_candidate_sets']}",
            f"binary_selection_reconstructed={checks['binary_selection_reconstructed']}",
            f"checked_games={checks['checked_games']}",
        ],
        "s2_live_path_reachable_confirmed": checks["s2_live_path_reachable_confirmed"],
        "s2_control_check": checks,
    }


def _audit_levelup(null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    signatures: list[str] = []
    attempts = _list_of_mappings(artifact.get("attempted_games"))
    if not attempts:
        _append_unique(signatures, "levelup_attempts_missing")

    labels = 0
    gates = 0
    existing_depth = 0
    new_depth_claims = 0
    accounting_consistent = True
    for row in attempts:
        label_count = len(row.get("solution_labels")) if isinstance(row.get("solution_labels"), list) else 0
        labels += label_count
        gate = row.get("reproduction_gate")
        gate_ok = isinstance(gate, Mapping) and gate.get("reproduced") is True
        gates += int(gate_ok)
        existing_depth += int(row.get("offline_reproduced_existing_depth") is True and gate_ok)
        new_depth_claims += int(row.get("offline_reproduced_new_depth") is True)
        prior = _int_value(row.get("prior_level"))
        reached = _int_value(row.get("reached_level"))
        new_levels = _int_value(row.get("new_levels_banked"))
        if row.get("offline_reproduced_new_depth") is True and not (reached > prior and new_levels > 0):
            accounting_consistent = False
        if row.get("offline_reproduced_existing_depth") is True and reached > prior:
            accounting_consistent = False

    if attempts and labels <= 0:
        _append_unique(signatures, "levelup_solution_labels_missing")
    if attempts and gates <= 0:
        _append_unique(signatures, "levelup_reproduction_gate_missing")
    if attempts and existing_depth <= 0 and _int_value(artifact.get("new_levels_banked")) <= 0:
        _append_unique(signatures, "levelup_same_depth_evidence_missing")
    if not accounting_consistent:
        _append_unique(signatures, "levelup_depth_accounting_inconsistent")
    if artifact.get("offline_reproduced") is True and _int_value(artifact.get("new_levels_banked")) <= 0:
        _append_unique(signatures, "levelup_offline_reproduced_true_without_new_bank")
    if artifact.get("schema_errors"):
        _append_unique(signatures, "levelup_schema_errors_present")
    preconditions = _mapping(artifact.get("preconditions_checked"))
    if _mapping(preconditions.get("offline_arcade")).get("ok") is False:
        _append_unique(signatures, "levelup_offline_arcade_failed")
    return {
        "null_id": null_id,
        "verdict": "silent_bug_must_reopen" if signatures else "trustworthy_null",
        "silent_bug_signatures": signatures,
        "exercise_evidence": [
            f"levelup_attempts={len(attempts)}",
            f"solution_label_count={labels}",
            f"reproduction_gates={gates}",
            f"existing_depth_reproduced={existing_depth}",
            f"new_depth_claims={new_depth_claims}",
            f"new_levels_banked={_int_value(artifact.get('new_levels_banked'))}",
            f"offline_reproduced={artifact.get('offline_reproduced') is True}",
        ],
    }


def _audit_firstwin(null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    signatures: list[str] = []
    rate = _finite_float(artifact.get("heldout_first_win_rate"))
    baseline = _finite_float(artifact.get("first_win_baseline"))
    attempts = _int_value(artifact.get("heldout_variant_attempts"))
    positive = artifact.get("positive_control_passed") is True
    parity_green = artifact.get("parity_test_green") is True
    note = str(artifact.get("null_delta_methodology_note") or "")
    substrate = str(artifact.get("inference_substrate") or "")
    proxy = _mapping(artifact.get("heldout_proxy_summary"))
    flat_004 = (
        rate is not None
        and baseline is not None
        and abs(rate - baseline) <= 1e-12
        and abs(rate - 0.04) <= 1e-12
    )

    if attempts < 100:
        _append_unique(signatures, "heldout_attempt_floor_not_met")
    if flat_004 and not note:
        _append_unique(signatures, "firstwin_0_04_tautology_unannotated")
    if flat_004 and not positive:
        _append_unique(signatures, "firstwin_positive_control_missing")
    if not parity_green:
        _append_unique(signatures, "parity_test_not_green")
    if substrate == INFERENCE_SUBSTRATE and artifact.get("live_agent_ran") is True:
        _append_unique(signatures, "firstwin_substrate_declares_aggregation_but_live_ran")
    if substrate == INFERENCE_SUBSTRATE and not proxy.get("proxy_cache_used"):
        _append_unique(signatures, "firstwin_aggregation_cache_evidence_missing")
    return {
        "null_id": null_id,
        "verdict": "silent_bug_must_reopen" if signatures else "trustworthy_null",
        "silent_bug_signatures": signatures,
        "exercise_evidence": [
            f"heldout_first_win_rate={_format_number(rate)}",
            f"first_win_baseline={_format_number(baseline)}",
            f"heldout_attempts={attempts}",
            f"positive_control_passed={positive}",
            f"parity_test_green={parity_green}",
            f"null_delta_methodology_note_present={bool(note)}",
            f"inference_substrate={substrate}",
            f"proxy_cache_used={proxy.get('proxy_cache_used') is True}",
        ],
    }


def audit_null_artifact(
    null_id: str,
    artifact: Mapping[str, Any],
    *,
    lint_result: Mapping[str, Any] | None = None,
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    """REQ-ARC-WMTE-4795: classify one .441 null from exercised-evidence fields."""

    if null_id == "experiment_4791_structural_energy_s2_offpath_trust_gate":
        return _audit_s2(
            null_id,
            artifact,
            lint_result=lint_result or {},
            root=root,
        )
    if null_id == "experiment_4792_levelup_attempt":
        return _audit_levelup(null_id, artifact)
    if null_id == "experiment_4794_heldout_first_win_readiness":
        return _audit_firstwin(null_id, artifact)
    return {
        "null_id": null_id,
        "verdict": "silent_bug_must_reopen",
        "silent_bug_signatures": ["unknown_null_artifact"],
        "exercise_evidence": ["unknown null artifact; cannot trust negative result"],
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    missing = [rel for rel in _expected_source_paths() if not (root_path / rel).exists()]
    return {
        "ok": not missing,
        "source_artifacts_present": not missing,
        "missing_source_artifacts": missing,
        "milestone_441_artifacts_present": not missing,
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists(),
        "checkpoint_441_present": (root_path / "results/checkpoints/experiment_441").exists(),
        "batch_ckpt_exp441_present": (root_path / "results/batch_ckpt/exp441").exists(),
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
        "nulls_audited": 0,
        "s2_live_path_reachable_confirmed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [],
        "silent_bugs_found": [],
        "trusted_nulls": [],
        "s2_control_check": {},
        "arc_orphan_solver_lint": {},
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
    lint_result: Mapping[str, Any],
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
    s2_row = next(
        (
            row
            for row in per_null_verdicts
            if row.get("null_id") == "experiment_4791_structural_energy_s2_offpath_trust_gate"
        ),
        {},
    )
    s2_check = s2_row.get("s2_control_check")
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
        "nulls_audited": len(per_null_verdicts),
        "s2_live_path_reachable_confirmed": bool(
            s2_row.get("s2_live_path_reachable_confirmed")
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [dict(row) for row in per_null_verdicts],
        "silent_bugs_found": silent,
        "trusted_nulls": trusted,
        "s2_control_check": dict(s2_check) if isinstance(s2_check, Mapping) else {},
        "arc_orphan_solver_lint": dict(lint_result),
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
    if not isinstance(artifact.get("nulls_audited"), int):
        errors.append("nulls_audited_must_be_int")
    if not isinstance(artifact.get("s2_live_path_reachable_confirmed"), bool):
        errors.append("s2_live_path_reachable_confirmed_must_be_bool")
    if not isinstance(artifact.get("silent_bugs_found"), list):
        errors.append("silent_bugs_found_must_be_list")
    if not isinstance(artifact.get("per_null_verdicts"), list):
        errors.append("per_null_verdicts_must_be_list")
    if not isinstance(artifact.get("s2_control_check"), dict):
        errors.append("s2_control_check_must_be_dict")
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
        "## Experiment 4795 .441 ARC Null Silent-Bug Audit",
        "",
        f"- Verdict: `{artifact.get('honest_verdict')}`",
        f"- Nulls audited: `{artifact.get('nulls_audited')}`",
        f"- S2 live path reachable confirmed: `{artifact.get('s2_live_path_reachable_confirmed')}`",
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
    s2_check = artifact.get("s2_control_check")
    if isinstance(s2_check, Mapping):
        rows.extend(
            [
                "",
                "### S2 Control Check",
                "",
                f"- Real binary control: `{s2_check.get('real_binary_accuracy_control')}`",
                f"- Identical candidate sets: `{s2_check.get('identical_candidate_sets')}`",
                f"- Binary selection reconstructed: `{s2_check.get('binary_selection_reconstructed')}`",
                f"- Checked games: `{s2_check.get('checked_games')}`",
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
    marker = "## Experiment 4795 .441 ARC Null Silent-Bug Audit"
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


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    lint_runner: LintRunner | None = None,
) -> JsonDict:
    repo = Path(root)
    checks = check_preconditions(repo)
    if not checks["ok"]:
        artifact = _blocked_artifact(checks)
        if write:
            write_artifact(artifact, root=repo)
        return artifact

    runner = lint_runner or run_arc_orphan_solver_lint
    lint_result = dict(runner(repo))
    per_null: list[JsonDict] = []
    checksums: dict[str, str] = {}
    for target in AUDIT_TARGETS:
        rel = target["artifact_path"]
        path = repo / rel
        payload = _read_json(path)
        checksums[rel] = _file_checksum(path)
        per_null.append(
            audit_null_artifact(
                target["null_id"],
                payload,
                lint_result=lint_result,
                root=repo,
            )
        )

    artifact = build_artifact(
        preconditions_checked=checks,
        per_null_verdicts=per_null,
        audited_artifact_checksums=checksums,
        lint_result=lint_result,
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
                "nulls_audited": artifact["nulls_audited"],
                "s2_live_path_reachable_confirmed": artifact[
                    "s2_live_path_reachable_confirmed"
                ],
                "result": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
