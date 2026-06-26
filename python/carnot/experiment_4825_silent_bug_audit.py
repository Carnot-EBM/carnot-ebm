"""Experiment 4825: audit .444 ARC nulls for silent no-op bugs.

Spec refs: REQ-ARC-WMTE-4825, SCENARIO-ARC-WMTE-4825-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4825-BLOCKED-PRECONDITION.
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

from carnot import experiment_4805_silent_bug_audit as prior_audit


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4825_silent_bug_audit"
EXPERIMENT_ID = 4825
SCHEMA = "carnot.arc.milestone_444_silent_bug_audit_4825.v1"
RESULT_RELATIVE_PATH = "results/experiment_4825_silent_bug_audit.json"
AUDIT_REPORT_RELATIVE_PATH = "ops/arc_null_silent_bug_audit.md"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 4825
DURATION_FLOOR_S = 0.0001
TERMINAL_PREFIXES = prior_audit.TERMINAL_PREFIXES

JsonDict = dict[str, Any]
LintRunner = Callable[[Path], Mapping[str, Any]]

SPEC_REFS = [
    "REQ-ARC-WMTE-4825",
    "SCENARIO-ARC-WMTE-4825-SILENT-BUG-AUDIT",
    "SCENARIO-ARC-WMTE-4825-BLOCKED-PRECONDITION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; audit complete is complete_/success_."
    },
    "s3_controls_verified": {
        "principle": (
            "the load-bearing check -- matched lambda=0, NEW-levels-not-re-ranking, "
            "reachable-winner positive control; else S3's verdict is uninterpretable."
        )
    },
    "s3_guidance_exercised": {
        "principle": (
            "true only if lambda=1 produced different candidate proposals/search than "
            "lambda=0 on >=1 headroom game -- else the S3 0.0 delta is a no-op, "
            "not a genuine generation null."
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
    "s3_control_check",
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
        "null_id": "experiment_4821_structural_energy_s3_generation_lift",
        "artifact_path": "results/experiment_4821_structural_energy_s3_generation_lift.json",
    },
    {
        "null_id": "experiment_4822_levelup_attempt",
        "artifact_path": "results/experiment_4822_levelup_attempt.json",
    },
    {
        "null_id": "experiment_4824_heldout_first_win_readiness",
        "artifact_path": "results/experiment_4824_heldout_first_win_readiness.json",
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
    """REQ-ARC-WMTE-4825: run the live-path lint for the S3 goal-energy graft."""

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


def _truthy_passed(value: Any) -> bool:
    if isinstance(value, Mapping):
        return value.get("passed") is True or value.get("ok") is True
    return value is True


def _headroom_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("banked_by_bare") is False]


def _lambda0_control_matched(artifact: Mapping[str, Any]) -> bool:
    lambda0 = _mapping(artifact.get("lambda0_control"))
    return bool(
        (_finite_float(artifact.get("lambda_guidance")) or 0.0) > 0.0
        and (_finite_float(lambda0.get("lambda")) == 0.0)
        and lambda0.get("matched_control") is True
    )


def _per_game_budget_matched(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        bare_attempts = _int_value(row.get("lambda0_attempts"))
        guided_attempts = _int_value(row.get("e_guided_attempts"))
        if bare_attempts <= 0 or bare_attempts != guided_attempts:
            return False
    return True


def _rows_by_game(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        game = str(row.get("game") or "")
        if game and row.get("attempted", True) is True:
            grouped.setdefault(game, []).append(row)
    return grouped


def _variant_signatures(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        signature = row.get("variant_signature")
        if signature is not None:
            out.add(str(signature))
        else:
            out.add(_stable_json([row.get("game"), row.get("variant")]))
    return out


def _same_source_games_and_budget(
    source: Mapping[str, Any] | None,
    *,
    headroom_games: set[str],
) -> JsonDict:
    if not source:
        return {
            "source_present": False,
            "baseline_headroom_games": 0,
            "guided_headroom_games": 0,
            "per_game_variant_counts_match": False,
            "variant_signatures_match": False,
            "single_random_seed_recorded": False,
            "same_source_games_and_budget": False,
        }
    baseline = _list_of_mappings(_mapping(source.get("baseline_measurement")).get("variant_attempts"))
    guided = _list_of_mappings(_mapping(source.get("goal_energy_measurement")).get("variant_attempts"))
    baseline_by_game = _rows_by_game(baseline)
    guided_by_game = _rows_by_game(guided)
    shared_games = set(baseline_by_game) & set(guided_by_game) & headroom_games
    counts_match = bool(shared_games) and all(
        len(baseline_by_game[game]) == len(guided_by_game[game])
        for game in shared_games
    )
    baseline_signatures = _variant_signatures(
        [row for game in shared_games for row in baseline_by_game[game]]
    )
    guided_signatures = _variant_signatures(
        [row for game in shared_games for row in guided_by_game[game]]
    )
    signatures_match = bool(baseline_signatures) and baseline_signatures == guided_signatures
    seed_recorded = source.get("random_seed") is not None
    return {
        "source_present": True,
        "baseline_headroom_games": len(set(baseline_by_game) & headroom_games),
        "guided_headroom_games": len(set(guided_by_game) & headroom_games),
        "per_game_variant_counts_match": bool(counts_match),
        "variant_signatures_match": bool(signatures_match),
        "single_random_seed_recorded": bool(seed_recorded),
        "same_source_games_and_budget": bool(counts_match and signatures_match),
    }


def _positive_float(*values: Any) -> bool:
    return any((_finite_float(value) or 0.0) > 0.0 for value in values)


def _guidance_exercise_check(
    source: Mapping[str, Any] | None,
    *,
    headroom_games: set[str],
) -> JsonDict:
    if not source:
        return {
            "source_present": False,
            "candidate_pool_differs_from_baseline": False,
            "candidate_states_scored": 0,
            "goal_energy_score_variance": 0.0,
            "real_candidate_state_evidence": False,
            "differing_headroom_games": [],
            "fingerprint_differences": [],
            "guidance_exercised": False,
        }

    nondegeneracy = _mapping(source.get("nondegeneracy"))
    diagnostics = _mapping(nondegeneracy.get("diagnostics"))
    candidate_pool_differs = bool(
        source.get("candidate_pool_differs_from_baseline") is True
        or nondegeneracy.get("candidate_pool_differs_from_baseline") is True
        or diagnostics.get("candidate_pool_differs_from_baseline") is True
    )
    candidate_states_scored = max(
        _int_value(nondegeneracy.get("candidate_states_scored")),
        _int_value(diagnostics.get("candidate_states_scored")),
        _int_value(diagnostics.get("candidate_states_scored_total")),
    )
    variance = max(
        _finite_float(source.get("goal_energy_score_variance")) or 0.0,
        _finite_float(nondegeneracy.get("goal_energy_score_variance")) or 0.0,
        _finite_float(diagnostics.get("goal_energy_score_variance")) or 0.0,
    )
    real_candidate_state = bool(
        source.get("arms_non_degenerate") is True
        or nondegeneracy.get("arms_non_degenerate") is True
        or diagnostics.get("real_candidate_state_evidence") is True
    )

    baseline = _list_of_mappings(_mapping(source.get("baseline_measurement")).get("variant_attempts"))
    guided = _list_of_mappings(_mapping(source.get("goal_energy_measurement")).get("variant_attempts"))
    baseline_by_key = {
        (str(row.get("game") or ""), str(row.get("variant_signature") or row.get("variant") or "")): row
        for row in baseline
    }
    differing_games: set[str] = set()
    fingerprint_differences: list[str] = []
    fingerprint_fields = (
        "proposal_fingerprint",
        "search_fingerprint",
        "explored_nodes",
        "candidate_proposals",
        "state_coverage",
        "distinct_win_relevant_states",
    )
    for row in guided:
        game = str(row.get("game") or "")
        if game not in headroom_games:
            continue
        row_diag = _mapping(row.get("goal_candidate_guidance_diagnostics"))
        if (
            row_diag.get("candidate_pool_differs_from_baseline") is True
            and _int_value(row_diag.get("candidate_states_scored")) > 0
            and _positive_float(row_diag.get("goal_energy_score_variance"))
            and row_diag.get("real_candidate_state_evidence") is True
        ):
            differing_games.add(game)
        key = (game, str(row.get("variant_signature") or row.get("variant") or ""))
        baseline_row = baseline_by_key.get(key)
        if baseline_row is not None:
            for field in fingerprint_fields:
                if field in row and row.get(field) != baseline_row.get(field):
                    differing_games.add(game)
                    _append_unique(fingerprint_differences, f"{game}:{field}")

    exercised = bool(
        differing_games
        and candidate_pool_differs
        and candidate_states_scored > 0
        and variance > 0.0
        and real_candidate_state
    )
    return {
        "source_present": True,
        "candidate_pool_differs_from_baseline": bool(candidate_pool_differs),
        "candidate_states_scored": int(candidate_states_scored),
        "goal_energy_score_variance": float(variance),
        "real_candidate_state_evidence": bool(real_candidate_state),
        "differing_headroom_games": sorted(differing_games),
        "fingerprint_differences": fingerprint_differences,
        "guidance_exercised": bool(exercised),
    }


def _new_levels_not_reranking(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if row.get("banked_by_E") is True:
            if row.get("was_already_in_bare_pool") is True or row.get("banked_by_bare") is True:
                return False
    return True


def s3_control_check(
    artifact: Mapping[str, Any],
    *,
    source_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    lint_result: Mapping[str, Any] | None = None,
) -> JsonDict:
    """REQ-ARC-WMTE-4825: verify S3's matched control and no-op guard."""

    rows = _list_of_mappings(artifact.get("game_results"))
    headroom_candidates = _headroom_candidate_rows(rows)
    headroom_games = {str(row.get("game")) for row in headroom_candidates if row.get("game")}
    source = _mapping((source_artifacts or {}).get("matched_generation_measurement"))
    source_match = _same_source_games_and_budget(source, headroom_games=headroom_games)
    guidance = _guidance_exercise_check(source, headroom_games=headroom_games)
    preconditions = _mapping(artifact.get("preconditions_checked"))
    artifact_lint = _truthy_passed(_mapping(preconditions.get("arc_orphan_solver_lint")))
    lint_passed = _truthy_passed(lint_result)
    live_path = bool(artifact.get("live_path_reachable") is True and artifact_lint and lint_passed)
    min_headroom = _int_value(artifact.get("min_headroom_games"))
    n_headroom = len(headroom_candidates)
    positive_all = bool(
        artifact.get("positive_control_passed") is True
        and n_headroom >= min_headroom > 0
        and all(row.get("positive_control_reachable") is True for row in headroom_candidates)
    )
    source_budget_ok = bool(
        source_match["same_source_games_and_budget"]
        if source_match["source_present"]
        else True
    )
    same_budget = bool(_per_game_budget_matched(headroom_candidates) and source_budget_ok)
    controls_verified = bool(
        _lambda0_control_matched(artifact)
        and same_budget
        and positive_all
        and _new_levels_not_reranking(rows)
        and live_path
    )
    return {
        "matched_lambda0_control": bool(_lambda0_control_matched(artifact)),
        "same_games_seeds_budget": bool(same_budget),
        "source_match_check": source_match,
        "n_headroom_games": n_headroom,
        "min_headroom_games": min_headroom,
        "positive_control_passed": bool(positive_all),
        "new_levels_not_in_bare_pool": bool(_new_levels_not_reranking(rows)),
        "live_path_reachable": bool(live_path),
        "artifact_live_path_reachable": artifact.get("live_path_reachable") is True,
        "artifact_arc_orphan_solver_lint_passed": bool(artifact_lint),
        "current_arc_orphan_solver_lint_passed": bool(lint_passed),
        "guidance_exercise_check": guidance,
        "s3_controls_verified": bool(controls_verified),
        "s3_guidance_exercised": bool(guidance["guidance_exercised"]),
    }


def _audit_s3(
    null_id: str,
    artifact: Mapping[str, Any],
    *,
    source_artifacts: Mapping[str, Mapping[str, Any]] | None,
    lint_result: Mapping[str, Any] | None,
) -> JsonDict:
    checks = s3_control_check(
        artifact,
        source_artifacts=source_artifacts,
        lint_result=lint_result,
    )
    signatures: list[str] = []
    if not checks["matched_lambda0_control"] or not checks["same_games_seeds_budget"]:
        _append_unique(signatures, "s3_lambda0_control_not_matched")
    if not checks["positive_control_passed"]:
        _append_unique(signatures, "s3_positive_control_missing_for_headroom")
    if not checks["new_levels_not_in_bare_pool"]:
        _append_unique(signatures, "s3_new_level_is_reranking")
    if not checks["live_path_reachable"]:
        _append_unique(signatures, "s3_goal_energy_not_live_path_reachable")
    guidance = _mapping(checks.get("guidance_exercise_check"))
    if not checks["s3_guidance_exercised"]:
        if not guidance.get("source_present"):
            _append_unique(signatures, "s3_guidance_evidence_missing")
        else:
            _append_unique(signatures, "s3_guidance_no_op")

    if not checks["s3_controls_verified"]:
        verdict = "silent_bug_must_reopen"
    elif not checks["s3_guidance_exercised"]:
        verdict = "inconclusive_guidance_no_op"
    else:
        verdict = "trustworthy_null"
    return {
        "null_id": null_id,
        "verdict": verdict,
        "silent_bug_signatures": signatures,
        "exercise_evidence": [
            f"matched_lambda0_control={checks['matched_lambda0_control']}",
            f"same_games_seeds_budget={checks['same_games_seeds_budget']}",
            f"n_headroom_games={checks['n_headroom_games']}",
            f"positive_control_passed={checks['positive_control_passed']}",
            f"new_levels_not_in_bare_pool={checks['new_levels_not_in_bare_pool']}",
            f"live_path_reachable={checks['live_path_reachable']}",
            f"s3_guidance_exercised={checks['s3_guidance_exercised']}",
            "differing_headroom_games="
            f"{len(_mapping(checks.get('guidance_exercise_check')).get('differing_headroom_games') or [])}",
        ],
        "s3_controls_verified": bool(checks["s3_controls_verified"]),
        "s3_guidance_exercised": bool(checks["s3_guidance_exercised"]),
        "s3_control_check": checks,
    }


def _audit_via_prior(null_id: str, prior_null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    result = dict(prior_audit.audit_null_artifact(prior_null_id, artifact))
    result["null_id"] = null_id
    return result


def audit_null_artifact(
    null_id: str,
    artifact: Mapping[str, Any],
    *,
    source_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    lint_result: Mapping[str, Any] | None = None,
) -> JsonDict:
    """REQ-ARC-WMTE-4825: classify one .444 null from exercised-evidence fields."""

    if null_id == "experiment_4821_structural_energy_s3_generation_lift":
        return _audit_s3(
            null_id,
            artifact,
            source_artifacts=source_artifacts,
            lint_result=lint_result,
        )
    if null_id == "experiment_4822_levelup_attempt":
        return _audit_via_prior(null_id, "experiment_4802_levelup_attempt", artifact)
    if null_id == "experiment_4824_heldout_first_win_readiness":
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
    return {
        "ok": not missing,
        "source_artifacts_present": not missing,
        "missing_source_artifacts": missing,
        "milestone_444_artifacts_present": not missing,
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists(),
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
        "s3_controls_verified": False,
        "s3_guidance_exercised": False,
        "nulls_audited": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [],
        "silent_bugs_found": [],
        "trusted_nulls": [],
        "s3_control_check": {},
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
    not_trusted = [
        dict(row)
        for row in per_null_verdicts
        if row.get("verdict") != "trustworthy_null"
    ]
    trusted = [
        str(row.get("null_id"))
        for row in per_null_verdicts
        if row.get("verdict") == "trustworthy_null" and row.get("null_id")
    ]
    s3_row = next(
        (
            row
            for row in per_null_verdicts
            if row.get("null_id") == "experiment_4821_structural_energy_s3_generation_lift"
        ),
        {},
    )
    s3_check = s3_row.get("s3_control_check")
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": (
            f"complete_arc_null_silent_bug_audit_{len(per_null_verdicts)}_nulls_"
            f"{len(not_trusted)}_reopen"
        ),
        "s3_controls_verified": bool(s3_row.get("s3_controls_verified")),
        "s3_guidance_exercised": bool(s3_row.get("s3_guidance_exercised")),
        "nulls_audited": len(per_null_verdicts),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_null_verdicts": [dict(row) for row in per_null_verdicts],
        "silent_bugs_found": not_trusted,
        "trusted_nulls": trusted,
        "s3_control_check": dict(s3_check) if isinstance(s3_check, Mapping) else {},
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
    if not isinstance(artifact.get("s3_controls_verified"), bool):
        errors.append("s3_controls_verified_must_be_bool")
    if not isinstance(artifact.get("s3_guidance_exercised"), bool):
        errors.append("s3_guidance_exercised_must_be_bool")
    if not isinstance(artifact.get("nulls_audited"), int):
        errors.append("nulls_audited_must_be_int")
    if not isinstance(artifact.get("silent_bugs_found"), list):
        errors.append("silent_bugs_found_must_be_list")
    if not isinstance(artifact.get("per_null_verdicts"), list):
        errors.append("per_null_verdicts_must_be_list")
    if not isinstance(artifact.get("s3_control_check"), dict):
        errors.append("s3_control_check_must_be_dict")
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
        "## Experiment 4825 .444 ARC Null Silent-Bug Audit",
        "",
        f"- Verdict: `{artifact.get('honest_verdict')}`",
        f"- Nulls audited: `{artifact.get('nulls_audited')}`",
        f"- S3 controls verified: `{artifact.get('s3_controls_verified')}`",
        f"- S3 guidance exercised: `{artifact.get('s3_guidance_exercised')}`",
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
    s3_check = artifact.get("s3_control_check")
    if isinstance(s3_check, Mapping):
        guidance = _mapping(s3_check.get("guidance_exercise_check"))
        rows.extend(
            [
                "",
                "### S3 Control Check",
                "",
                f"- Matched lambda-zero control: `{s3_check.get('matched_lambda0_control')}`",
                f"- Same games/seeds/budget: `{s3_check.get('same_games_seeds_budget')}`",
                f"- Headroom games: `{s3_check.get('n_headroom_games')}`",
                f"- Reachable-winner positive control: `{s3_check.get('positive_control_passed')}`",
                f"- New levels not in bare pool: `{s3_check.get('new_levels_not_in_bare_pool')}`",
                f"- Live-path reachable: `{s3_check.get('live_path_reachable')}`",
                f"- Guidance exercised: `{s3_check.get('s3_guidance_exercised')}`",
                "- Differing headroom games: "
                f"`{len(guidance.get('differing_headroom_games') or [])}`",
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
    marker = "## Experiment 4825 .444 ARC Null Silent-Bug Audit"
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


def _load_s3_sources(
    *,
    root: Path,
    s3_artifact: Mapping[str, Any],
    checksums: dict[str, str],
) -> dict[str, Mapping[str, Any]]:
    sources: dict[str, Mapping[str, Any]] = {}
    source_paths = _mapping(s3_artifact.get("source_artifacts"))
    rel = source_paths.get("matched_generation_measurement")
    if isinstance(rel, str):
        path = root / rel
        if path.exists():
            sources["matched_generation_measurement"] = _read_json(path)
            checksums[rel] = _file_checksum(path)
    return sources


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
        source_artifacts: dict[str, Mapping[str, Any]] = {}
        if target["null_id"] == "experiment_4821_structural_energy_s3_generation_lift":
            source_artifacts = _load_s3_sources(
                root=repo,
                s3_artifact=payload,
                checksums=checksums,
            )
        per_null.append(
            audit_null_artifact(
                target["null_id"],
                payload,
                source_artifacts=source_artifacts,
                lint_result=lint_result,
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
                "s3_controls_verified": artifact["s3_controls_verified"],
                "s3_guidance_exercised": artifact["s3_guidance_exercised"],
                "result": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
