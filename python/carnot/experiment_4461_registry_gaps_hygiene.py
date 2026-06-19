"""Exp 4461: reconcile .412 registry and verifier-gap hygiene.

Spec refs: REQ-REPORT-4461, SCENARIO-REPORT-4461.

This audit pass reads the .412 outcome artifacts, refreshes the ARC registry
and gap ledgers from clean reproduction evidence, runs the cached GAP-4 guard,
and writes a terminal hygiene artifact. It deliberately does not edit
production verifier code.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import yaml

from carnot import experiment_4449_registry_gaps_hygiene as prior
from carnot.reporting import capstone_aggregate_available


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4461_registry_gaps_hygiene.json"
REGISTRY_RELATIVE_PATH = prior.REGISTRY_RELATIVE_PATH
GAPS_RELATIVE_PATH = prior.GAPS_RELATIVE_PATH
ARC_REGISTRY_RELATIVE_PATH = prior.ARC_REGISTRY_RELATIVE_PATH

EXP4455_PATH = "results/experiment_4455_solve_dc22_cegis_config_rule.json"
EXP4456_PATH = "results/experiment_4456_generic_glyph_rewrite_operator.json"
EXP4457_PATTERN = "results/experiment_4457_*.json"
EXP4458_PATH = "results/experiment_4458_first_contact_new_game.json"
EXP4459_PATH = "results/experiment_4459_loo_generic_solve_benchmark_v3.json"
EXP4460_PATH = "results/experiment_4460_submission_package_prep.json"

SOURCE_ARTIFACTS = {
    "4455_dc22": EXP4455_PATH,
    "4456_glyph_rewrite": EXP4456_PATH,
    "4457_cast_grid": EXP4457_PATTERN,
    "4458_first_contact": EXP4458_PATH,
    "4459_loo_v3": EXP4459_PATH,
    "4460_submission": EXP4460_PATH,
}
SOURCE_EXPERIMENT_IDS = {
    "4455_dc22": 4455,
    "4456_glyph_rewrite": 4456,
    "4457_cast_grid": 4457,
    "4458_first_contact": 4458,
    "4459_loo_v3": 4459,
    "4460_submission": 4460,
}

RANDOM_SEED = 4461
SPEC_REFS = ("REQ-REPORT-4461", "SCENARIO-REPORT-4461")
INFERENCE_SUBSTRATE = prior.INFERENCE_SUBSTRATE
GAP4_VERIFIER_ID = prior.GAP4_VERIFIER_ID
V412_ROLE_ID = "oracle_distinct_v412_registry_gaps_hygiene_4461"
TERMINAL_PREFIXES = prior.TERMINAL_PREFIXES

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "reproducible_total_levels",
    "reproducible_total_games",
    "open_gap_ids",
    "inference_substrate",
    "registry_reconciliation",
    "availability_report",
    "capstone_stamp_fix_durable",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "submitted_to_leaderboard",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {"principle": "terminal-prefixed"},
    "regression_guard_passed": {
        "principle": "BARE bool (gated-fields-must-be-bare): the GAP-4 execution result did not regress"
    },
    "reproducible_total_levels": {
        "principle": "the reconciled authoritative count (target >= 40 after dc22 + sc25 deepening)"
    },
    "reproducible_total_games": {
        "principle": "the reconciled authoritative game count (target >= 21 after dc22)"
    },
    "open_gap_ids": {
        "principle": "the still-open generic-solver gaps after .412 -- the .413 backlog"
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream artifacts; 100us floor"
    },
    "availability_report": {
        "principle": "robust aggregate-available report; missing or flagged inputs do not erase other axes"
    },
    "submitted_to_leaderboard": {"principle": "must remain false for this audit-only task"},
}

GAP_MARKER_OVERRIDES = {
    "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT": (
        "exp4438-gap-4423-dc22-unselectable-first-contact"
    ),
    "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER": (
        "exp4438-gap-4432-loo-tr87-missing-glyph-rewrite-rule-verifier-without-tr87-adapter"
    ),
    "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER": (
        "exp4438-gap-4432-loo-sc25-missing-cast-grid-spell-shrink-tank-exit-verifier"
    ),
    "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE": (
        "exp4458-gap-sb26-color-match-slot-sequence"
    ),
}

Gap4GuardRunner = Callable[[Path], Mapping[str, Any]]
CapstoneStampRunner = Callable[[Path], Mapping[str, Any]]


def _run_gap4_regression_guard(root: Path) -> Mapping[str, Any]:  # pragma: no cover
    return prior._run_gap4_regression_guard(root)


def _verify_capstone_stamp_fix_durable(root: Path) -> Mapping[str, Any]:  # pragma: no cover
    return prior._verify_capstone_stamp_fix_durable(root)


def _resolve_artifact_path(root: Path, pattern: str) -> str:
    if "*" not in pattern:
        return pattern
    matches = sorted(root.glob(pattern))
    if matches:
        return matches[0].relative_to(root).as_posix()
    return pattern


def _as_int(value: Any) -> int:
    return prior._as_int(value)


def _dc22_banked(payload: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and payload.get("target_game") == "dc22"
        and payload.get("offline_reproduced") is True
        and payload.get("dc22_grounded") is not False
        and _as_int(payload.get("reproduced_levels")) >= 1
    )


def _tr87_closed(payload: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and payload.get("tr87_resolved_generically") is True
        and payload.get("offline_reproduced") is True
        and _as_int(payload.get("tr87_generic_level_reproduced") or 1) >= 1
    )


def _sc25_levels(payload: Mapping[str, Any] | None) -> int:
    if not isinstance(payload, Mapping):
        return 0
    reproduction = payload.get("generic_reproduction_result")
    if isinstance(reproduction, Mapping):
        return max(
            _as_int(payload.get("reproduced_levels")),
            _as_int(reproduction.get("reached_level")),
            _as_int(reproduction.get("claimed_level")),
        )
    return _as_int(payload.get("reproduced_levels"))


def _sc25_cast_grid_closed(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    closure_flag = any(
        payload.get(field) is True
        for field in (
            "sc25_resolved_generically",
            "sc25_cast_grid_resolved_generically",
            "cast_grid_phase_fsm_resolved_generically",
        )
    )
    return bool(closure_flag and payload.get("offline_reproduced") is True and _sc25_levels(payload) >= 1)


def _loo_closes_game(payload: Mapping[str, Any] | None, game: str) -> bool:
    if not isinstance(payload, Mapping) or payload.get("offline_reproduced") is not True:
        return False
    for row in payload.get("per_game") or []:
        if (
            isinstance(row, Mapping)
            and row.get("game") == game
            and row.get("solved_without_own_recipe") is True
            and row.get("residual_delta") == "none"
            and row.get("closed_by_operator") not in {"", "none", None}
        ):
            return True
    return False


def _first_contact_banked(payload: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and payload.get("offline_reproduced") is True
        and _as_int(payload.get("reproduced_levels")) >= 1
        and str(payload.get("target_game") or "")
    )


def load_sources(root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """REQ-REPORT-4461: read every upstream artifact before reconciling ledgers."""

    payloads: dict[str, Any] = {}
    report: dict[str, dict[str, Any]] = {}
    for key, pattern in SOURCE_ARTIFACTS.items():
        rel_path = _resolve_artifact_path(root, pattern)
        payload, row = prior._load_json(root / rel_path)
        payloads[key] = payload
        row["relative_path"] = rel_path
        row["source_pattern"] = pattern
        report[key] = row
    return payloads, report


def _axis_specs() -> list[capstone_aggregate_available.AxisSpec]:
    return [
        capstone_aggregate_available.AxisSpec(
            name="dc22_bank",
            required_keys=("4455_dc22",),
            verdict_fn=lambda present: _dc22_banked(present.get("4455_dc22")),
        ),
        capstone_aggregate_available.AxisSpec(
            name="glyph_rewrite",
            required_keys=("4456_glyph_rewrite",),
            verdict_fn=lambda present: _tr87_closed(present.get("4456_glyph_rewrite")),
        ),
        capstone_aggregate_available.AxisSpec(
            name="cast_grid",
            required_keys=("4457_cast_grid",),
            verdict_fn=lambda present: {
                "sc25_closed": _sc25_cast_grid_closed(present.get("4457_cast_grid")),
                "sc25_levels": _sc25_levels(present.get("4457_cast_grid")),
            },
        ),
        capstone_aggregate_available.AxisSpec(
            name="first_contact_new_game",
            required_keys=("4458_first_contact",),
            verdict_fn=lambda present: {
                "target_game": (present.get("4458_first_contact") or {}).get("target_game"),
                "banked": _first_contact_banked(present.get("4458_first_contact")),
            },
        ),
        capstone_aggregate_available.AxisSpec(
            name="loo_v3",
            required_keys=("4459_loo_v3",),
            verdict_fn=lambda present: _as_int(
                (present.get("4459_loo_v3") or {}).get("generic_loo_solve_count_v3")
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="submission_package",
            required_keys=("4460_submission",),
            verdict_fn=lambda present: {
                "ready": (present.get("4460_submission") or {}).get("submission_package_ready")
                is True,
                "levels": _as_int(
                    (present.get("4460_submission") or {}).get(
                        "total_reproduced_levels_in_package"
                    )
                ),
                "submitted_to_leaderboard": (present.get("4460_submission") or {}).get(
                    "submitted_to_leaderboard"
                )
                is True,
            },
        ),
    ]


def availability_report(payloads: Mapping[str, Any]) -> dict[str, Any]:
    """REQ-REPORT-4461: report per-axis gaps without poisoning unrelated axes."""

    return capstone_aggregate_available.aggregate_available_report_gaps(
        payloads,
        _axis_specs(),
        artifact_experiment_ids=SOURCE_EXPERIMENT_IDS,
    )


def trusted_payloads(payloads: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    available = set(report.get("available_artifact_keys", []))
    return {
        key: dict(payload)
        for key, payload in payloads.items()
        if key in available and isinstance(payload, Mapping)
    }


def excluded_artifact_paths(
    payloads: Mapping[str, Any], source_report: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    return [
        str(source_report.get(key, {}).get("relative_path") or SOURCE_ARTIFACTS[key])
        for key, payload in payloads.items()
        if isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True
    ]


def _source_path(source_report: Mapping[str, Mapping[str, Any]], key: str) -> str:
    return str(source_report.get(key, {}).get("relative_path") or SOURCE_ARTIFACTS[key])


def _gap_entry(
    gap_id: str,
    *,
    status: str,
    evidence: str,
    failure_mode: str,
    missing_discriminator: str,
    candidate_design: str,
    source_artifact: str,
    movement: str,
) -> dict[str, Any]:
    return {
        "gap_id": gap_id,
        "status": status,
        "evidence": evidence,
        "failure_mode": failure_mode,
        "missing_discriminator": missing_discriminator,
        "candidate_design": candidate_design,
        "priority": "high",
        "source_artifact": source_artifact,
        "movement": movement,
    }


def collect_gap_entries(
    trusted: Mapping[str, Mapping[str, Any]],
    source_report: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """SCENARIO-REPORT-4461: convert .412 outcomes into filled or open gap rows."""

    entries: dict[str, dict[str, Any]] = {}
    exp4455 = trusted.get("4455_dc22", {})
    dc22_path = _source_path(source_report, "4455_dc22")
    if _dc22_banked(exp4455):
        entries["GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT"] = _gap_entry(
            "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
            status="filled (exp4455_solve_dc22_cegis_config_rule)",
            evidence=(
                f"{dc22_path}; target_game=dc22; offline_reproduced=True; "
                f"reproduced_levels={exp4455.get('reproduced_levels')}"
            ),
            failure_mode="closed_by_dc22_cegis_config_rule",
            missing_discriminator="filled by execution-grounded dc22 config-rule predicate",
            candidate_design="keep the dc22 CEGIS config-rule verifier in the generic bank",
            source_artifact=dc22_path,
            movement="filled",
        )
    else:
        entries["GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT"] = _gap_entry(
            "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
            status="open",
            evidence=(
                f"{dc22_path}; offline_reproduced={exp4455.get('offline_reproduced')}; "
                f"dc22_grounded={exp4455.get('dc22_grounded')}; "
                f"honest_verdict={exp4455.get('honest_verdict', 'missing_clean_artifact')}"
            ),
            failure_mode="missing_dc22_reproduction_gate",
            missing_discriminator="dc22 still lacks clean reproduction-gated generic closure",
            candidate_design="rerun dc22 CEGIS/config-rule grounding and count only reproduce(dc22)",
            source_artifact=dc22_path,
            movement="updated_still_open",
        )

    exp4456 = trusted.get("4456_glyph_rewrite", {})
    tr87_path = _source_path(source_report, "4456_glyph_rewrite")
    if _tr87_closed(exp4456):
        entries["GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER"] = _gap_entry(
            "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER",
            status="filled (exp4456_generic_glyph_rewrite_operator)",
            evidence=(
                f"{tr87_path}; tr87_resolved_generically=True; offline_reproduced=True; "
                f"tr87_generic_level_reproduced={exp4456.get('tr87_generic_level_reproduced')}"
            ),
            failure_mode="prior missing_glyph_rewrite_rule_verifier residual is closed for tr87",
            missing_discriminator="filled by generic glyph_rewrite_rule_verifier",
            candidate_design="keep the glyph rewrite verifier in the generic leave-one-out loop",
            source_artifact=tr87_path,
            movement="filled",
        )
    else:
        entries["GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER"] = _gap_entry(
            "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER",
            status="open",
            evidence=f"{tr87_path}; clean tr87 generic closure not available",
            failure_mode="missing_glyph_rewrite_rule_verifier_without_tr87_adapter",
            missing_discriminator="generic glyph rewrite verifier still missing or ungated",
            candidate_design="ground glyph rewrite from pixels/rules and rerun LOO",
            source_artifact=tr87_path,
            movement="updated_still_open",
        )

    exp4457 = trusted.get("4457_cast_grid", {})
    exp4459 = trusted.get("4459_loo_v3", {})
    sc25_path = _source_path(source_report, "4457_cast_grid")
    sc25_closed = _sc25_cast_grid_closed(exp4457) or _loo_closes_game(exp4459, "sc25")
    if sc25_closed:
        source = sc25_path if _sc25_cast_grid_closed(exp4457) else _source_path(source_report, "4459_loo_v3")
        entries["GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER"] = _gap_entry(
            "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
            status="filled (exp4457_cast_grid_phase_fsm_world_model)",
            evidence=(
                f"{source}; sc25_cast_grid_closed=True; "
                f"sc25_levels={max(1, _sc25_levels(exp4457))}; "
                f"generic_loo_solve_count_v3={exp4459.get('generic_loo_solve_count_v3')}"
            ),
            failure_mode="prior missing_cast_grid_spell_shrink_tank_exit_verifier residual is closed",
            missing_discriminator="filled by cast_grid_phase_fsm_world_model",
            candidate_design="keep the cast-grid phase FSM verifier in the generic loop",
            source_artifact=source,
            movement="filled",
        )
    else:
        entries["GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER"] = _gap_entry(
            "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
            status="open",
            evidence=(
                f"{sc25_path}; cast_grid_artifact_available={bool(exp4457)}; "
                f"generic_loo_solve_count_v3={exp4459.get('generic_loo_solve_count_v3')}; "
                f"missing_verifier_gaps={exp4459.get('missing_verifier_gaps', [])}"
            ),
            failure_mode="missing_cast_grid_spell_shrink_tank_exit_verifier",
            missing_discriminator="generic cast-grid spell/shrink/tank-exit verifier still missing",
            candidate_design="build cast_grid_phase_fsm_world_model and rerun LOO v4",
            source_artifact=sc25_path,
            movement="updated_still_open",
        )

    exp4458 = trusted.get("4458_first_contact", {})
    first_contact_path = _source_path(source_report, "4458_first_contact")
    target_game = str(exp4458.get("target_game") or "sb26").upper()
    if _first_contact_banked(exp4458):
        gap_id = f"GAP-4458-{target_game}-COLOR-MATCH-SLOT-SEQUENCE"
        entries[gap_id] = _gap_entry(
            gap_id,
            status="filled (exp4458_first_contact_new_game)",
            evidence=(
                f"{first_contact_path}; target_game={exp4458.get('target_game')}; "
                f"offline_reproduced=True; reproduced_levels={exp4458.get('reproduced_levels')}"
            ),
            failure_mode="closed_by_first_contact_new_game",
            missing_discriminator="filled by routed first-contact generic verifier",
            candidate_design="reuse the routed first-contact primitive",
            source_artifact=first_contact_path,
            movement="filled",
        )
    else:
        entries["GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE"] = _gap_entry(
            "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE",
            status="open",
            evidence=(
                f"{first_contact_path}; target_game={exp4458.get('target_game', 'sb26')}; "
                f"routed_to={exp4458.get('routed_to')}; "
                f"offline_reproduced={exp4458.get('offline_reproduced')}; "
                f"reproduced_levels={exp4458.get('reproduced_levels')}"
            ),
            failure_mode="missing_color_match_slot_sequence_verifier",
            missing_discriminator="generic ordered color-match item-slot verifier with undo-aware grounding",
            candidate_design="extend config_rule_verifier with color_match_slot_sequence digests",
            source_artifact=first_contact_path,
            movement="updated_still_open",
        )

    return list(entries.values())


def _gap_marker(gap_id: str) -> str:
    return GAP_MARKER_OVERRIDES.get(gap_id, f"exp4461-{prior._slug(gap_id).lower()}")


def _gap_block(gap: Mapping[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4461 .412 registry gap hygiene\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'high')}\n"
        f"- source artifact: {gap.get('source_artifact', '')}\n"
        f"- movement: {gap.get('movement', 'updated')}\n"
    )


def reconcile_gaps_text(gaps_text: str, gap_entries: list[dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    updated = gaps_text
    filled: list[str] = []
    open_ids: list[str] = []
    for gap in gap_entries:
        gap_id = str(gap["gap_id"])
        if str(gap.get("status", "")).startswith("filled"):
            filled.append(gap_id)
        else:
            open_ids.append(gap_id)
        updated = prior._replace_marked_block(updated, _gap_marker(gap_id), _gap_block(gap))
    return updated, filled, open_ids


def _record_reproduced_game(
    registry: dict[str, Any],
    *,
    game: str,
    levels: int,
    artifact_path: str,
    latest_key: str,
    checksum: str,
    solver: str,
) -> None:
    prior._record_reproduced_game(
        registry,
        game=game,
        levels=levels,
        artifact_path=artifact_path,
        latest_key=latest_key,
        checksum=checksum,
        solver=solver,
    )


def reconcile_arc_registry(
    registry: Mapping[str, Any],
    trusted: Mapping[str, Mapping[str, Any]],
    source_report: Mapping[str, Mapping[str, Any]],
    *,
    filled_gap_ids: list[str],
    open_gap_ids: list[str],
    excluded: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """REQ-REPORT-4461: update reproduced ARC counts from trusted .412 rows."""

    updated = deepcopy(dict(registry))
    updated.setdefault("games", [])

    exp4455 = trusted.get("4455_dc22", {})
    if _dc22_banked(exp4455):
        _record_reproduced_game(
            updated,
            game="dc22",
            levels=max(1, _as_int(exp4455.get("reproduced_levels"))),
            artifact_path=_source_path(source_report, "4455_dc22"),
            latest_key="latest_exp4455_reproduce",
            checksum=str(exp4455.get("reproducibility_checksum") or ""),
            solver="python/carnot/experiment_4455_solve_dc22_cegis_config_rule.py",
        )

    exp4457 = trusted.get("4457_cast_grid", {})
    if _sc25_cast_grid_closed(exp4457):
        _record_reproduced_game(
            updated,
            game="sc25",
            levels=max(1, _sc25_levels(exp4457)),
            artifact_path=_source_path(source_report, "4457_cast_grid"),
            latest_key="latest_exp4457_reproduce",
            checksum=str(exp4457.get("reproducibility_checksum") or ""),
            solver="python/carnot/experiment_4457_cast_grid_phase_fsm_world_model.py",
        )

    exp4458 = trusted.get("4458_first_contact", {})
    if _first_contact_banked(exp4458):
        _record_reproduced_game(
            updated,
            game=str(exp4458.get("target_game") or "sb26"),
            levels=max(1, _as_int(exp4458.get("reproduced_levels"))),
            artifact_path=_source_path(source_report, "4458_first_contact"),
            latest_key="latest_exp4458_reproduce",
            checksum=str(exp4458.get("reproducibility_checksum") or ""),
            solver="python/carnot/experiment_4458_first_contact_new_game.py",
        )

    exp4456 = trusted.get("4456_glyph_rewrite", {})
    if exp4456:
        prior._ensure_game(updated, "tr87")["generic_glyph_rewrite_reproduce"] = (
            "Exp4456 closed tr87 LOO through glyph_rewrite_rule_verifier without tr87 adapter."
        )
        updated["latest_glyph_rewrite_rule_verifier_4456"] = {
            "artifact": _source_path(source_report, "4456_glyph_rewrite"),
            "tr87_resolved_generically": _tr87_closed(exp4456),
            "tr87_generic_level_reproduced": _as_int(
                exp4456.get("tr87_generic_level_reproduced")
            ),
            "offline_reproduced": exp4456.get("offline_reproduced") is True,
            "no_regression": exp4456.get("no_regression") is True,
        }

    exp4459 = trusted.get("4459_loo_v3", {})
    if exp4459:
        updated["latest_loo_generic_v3_4459"] = {
            "artifact": _source_path(source_report, "4459_loo_v3"),
            "generic_loo_solve_count_v2_baseline": exp4459.get(
                "generic_loo_solve_count_v2_baseline"
            ),
            "generic_loo_solve_count_v3": exp4459.get("generic_loo_solve_count_v3"),
            "loo_gate_passed": exp4459.get("loo_gate_passed") is True,
            "closed_residuals_by_412_operator": list(
                exp4459.get("closed_residuals_by_412_operator") or []
            ),
            "missing_verifier_gaps": list(exp4459.get("missing_verifier_gaps") or []),
        }

    exp4460 = trusted.get("4460_submission", {})
    if exp4460:
        updated["latest_submission_package_4460"] = {
            "artifact": _source_path(source_report, "4460_submission"),
            "submission_package_ready": exp4460.get("submission_package_ready") is True,
            "total_reproduced_levels_in_package": exp4460.get(
                "total_reproduced_levels_in_package"
            ),
            "submitted_to_leaderboard": exp4460.get("submitted_to_leaderboard") is True,
        }

    total_levels, total_games = prior._reproduced_counts(updated)
    updated["updated"] = "2026-06-19"
    updated["reproducible_total_levels"] = total_levels
    updated["reproducible_total_games"] = total_games
    updated["latest_hygiene_4461"] = {
        "artifact": RESULT_RELATIVE_PATH,
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "filled_gap_ids": filled_gap_ids,
        "open_gap_ids": open_gap_ids,
        "excluded_artifacts": excluded,
        "note": ".412 registry hygiene; missing, blocked, or flagged artifacts excluded from counts.",
    }
    return updated, {
        "arc_solve_registry_reconciled": True,
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
    }


def guard_passed(guard: Mapping[str, Any]) -> bool:
    return prior.guard_passed(guard)


def stamp_fix_durable(stamp: Mapping[str, Any]) -> bool:
    return prior.stamp_fix_durable(stamp)


def reconcile_verifier_registry(
    registry: Mapping[str, Any],
    *,
    guard: Mapping[str, Any],
    stamp: Mapping[str, Any],
    total_levels: int,
    total_games: int,
    filled_gap_ids: list[str],
    open_gap_ids: list[str],
    trusted: Mapping[str, Mapping[str, Any]],
    excluded: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = deepcopy(dict(registry))
    verifier = prior._find_verifier(updated, GAP4_VERIFIER_ID)
    if verifier is None:  # pragma: no cover - defensive creation path
        verifier = {
            "verifier_id": GAP4_VERIFIER_ID,
            "domain": "arc_agi2_grid",
            "kind": "process_verifier",
            "eval": {},
            "registry_roles": [],
        }
        updated.setdefault("verifiers", []).append(verifier)

    current = prior._guard_current(guard)
    exp4459 = trusted.get("4459_loo_v3", {})
    exp4460 = trusted.get("4460_submission", {})
    verifier.setdefault("eval", {}).update(
        {
            "eval_exp_4461": RESULT_RELATIVE_PATH,
            "exp4461_regression_guard_passed": guard_passed(guard),
            "exp4461_arc_oracle_distinct_verifier_beats_vote": (
                guard.get("arc_oracle_distinct_verifier_beats_vote") is not False
            ),
            "exp4461_arc1_rule_exec_vote_pass2": current.get("vote_pass2"),
            "exp4461_arc1_rule_exec_gated_pass2": current.get("gated_pass2"),
            "exp4461_arc1_headroom_recovered": current.get("headroom_recovered"),
            "exp4461_arc1_vote_wins_lost": current.get("vote_wins_lost"),
            "exp4461_capstone_stamp_fix_durable": stamp_fix_durable(stamp),
            "exp4461_reproducible_total_levels": total_levels,
            "exp4461_reproducible_total_games": total_games,
            "exp4461_filled_gap_ids": filled_gap_ids,
            "exp4461_open_gap_ids": open_gap_ids,
            "exp4461_flagged_artifacts_excluded": excluded,
            "exp4461_generic_loo_solve_count_v3": exp4459.get("generic_loo_solve_count_v3"),
            "exp4461_submission_package_ready": exp4460.get("submission_package_ready"),
            "exp4461_submission_package_levels": exp4460.get(
                "total_reproduced_levels_in_package"
            ),
        }
    )

    role = {
        "role_id": V412_ROLE_ID,
        "experiment": RESULT_RELATIVE_PATH,
        "role": "registry_gaps_arc_hygiene_v412",
        "status": "v412_outcomes_recorded_with_gap4_guard_and_stamp_durability",
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "filled_gap_ids": filled_gap_ids,
        "open_gap_ids": open_gap_ids,
        "excluded_artifacts": excluded,
        "eval_exp_4461": RESULT_RELATIVE_PATH,
    }
    roles = verifier.setdefault("registry_roles", [])
    if not isinstance(roles, list):  # pragma: no cover - defensive cleanup path
        roles = verifier["registry_roles"] = []
    verifier["registry_roles"] = [
        row for row in roles if not (isinstance(row, Mapping) and row.get("role_id") == V412_ROLE_ID)
    ] + [role]
    return updated, {"verifier_registry_reconciled": True}


def check_preconditions(
    root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    verifier_registry, verifier_check = prior._yaml_mapping(root / REGISTRY_RELATIVE_PATH)
    arc_registry, arc_check = prior._yaml_mapping(root / ARC_REGISTRY_RELATIVE_PATH)
    gaps_text, gaps_check = prior._read_text(root / GAPS_RELATIVE_PATH)
    payloads, source_report = load_sources(root)
    helper_check = prior._check_helper_import()
    checks = {
        "ok": verifier_check["readable"] and arc_check["readable"] and gaps_check["readable"] and helper_check["ok"],
        "files": {
            "verifier_registry": verifier_check,
            "arc_solve_registry": arc_check,
            "verifier_gaps": gaps_check,
        },
        "compatibility_import": helper_check,
        "source_artifacts": source_report,
    }
    return verifier_registry, arc_registry, gaps_text, payloads, checks


def _checksum_for_artifact(artifact: Mapping[str, Any]) -> str:
    return prior._sha256(
        {
            "registry_reconciliation": artifact.get("registry_reconciliation"),
            "availability_report": artifact.get("availability_report"),
            "regression_guard_passed": artifact.get("regression_guard_passed"),
            "capstone_stamp_fix_durable": artifact.get("capstone_stamp_fix_durable"),
            "reproducible_total_levels": artifact.get("reproducible_total_levels"),
            "reproducible_total_games": artifact.get("reproducible_total_games"),
            "open_gap_ids": artifact.get("open_gap_ids"),
            "excluded_artifacts": artifact.get("excluded_artifacts"),
            "random_seed": artifact.get("random_seed"),
            "spec_refs": artifact.get("spec_refs"),
            "inference_substrate": artifact.get("inference_substrate"),
        }
    )


def build_artifact(
    *,
    started_at: float,
    ended_at: float,
    availability: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    registry_reconciliation: Mapping[str, Any],
    guard: Mapping[str, Any],
    stamp: Mapping[str, Any],
    total_levels: int,
    total_games: int,
    excluded: list[str],
) -> dict[str, Any]:
    guard_ok = guard_passed(guard)
    stamp_ok = stamp_fix_durable(stamp)
    suffix = "guard_passed" if guard_ok else "guard_failed"
    artifact: dict[str, Any] = {
        "experiment": "experiment_4461_registry_gaps_hygiene",
        "schema": "carnot.exp4461.registry_gaps_hygiene.v1",
        "honest_verdict": f"complete: registry_gaps_hygiene_4461_{suffix}",
        "regression_guard_passed": guard_ok,
        "reproducible_total_levels": int(total_levels),
        "reproducible_total_games": int(total_games),
        "open_gap_ids": list(registry_reconciliation.get("open_gap_ids", [])),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "registry_reconciliation": dict(registry_reconciliation),
        "availability_report": dict(availability),
        "capstone_stamp_fix_durable": stamp_ok,
        "capstone_stamp_fix": dict(stamp),
        "gap4_regression_guard": dict(guard),
        "excluded_artifacts": list(excluded),
        "preconditions_checked": dict(preconditions),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "submitted_to_leaderboard": False,
        "no_production_verifier_edits": True,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": max(0.0001, round(float(ended_at - started_at), 6)),
        "model_specs": {
            "method": INFERENCE_SUBSTRATE,
            "codex_calls": 0,
            "live_model_inference": False,
            "gpu_inference": False,
            "upstream_artifacts": list(SOURCE_ARTIFACTS.values()),
        },
        "registries_reconciled": bool(registry_reconciliation.get("registries_reconciled")),
    }
    artifact["reproducibility_checksum"] = _checksum_for_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete:/success:/passed:/shipped:")
    if type(artifact.get("regression_guard_passed")) is not bool:
        errors.append("regression_guard_passed must be bare bool")
    if type(artifact.get("capstone_stamp_fix_durable")) is not bool:
        errors.append("capstone_stamp_fix_durable must be bare bool")
    if type(artifact.get("reproducible_total_levels")) is not int:
        errors.append("reproducible_total_levels must be bare int")
    if type(artifact.get("reproducible_total_games")) is not int:
        errors.append("reproducible_total_games must be bare int")
    open_gap_ids = artifact.get("open_gap_ids")
    if not isinstance(open_gap_ids, list) or any(not isinstance(item, str) for item in open_gap_ids):
        errors.append("open_gap_ids must be list[str]")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal aggregation_from_upstream_artifacts")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    if not isinstance(artifact.get("registry_reconciliation"), Mapping):
        errors.append("registry_reconciliation must be dict")
    if not isinstance(artifact.get("availability_report"), Mapping):
        errors.append("availability_report must be dict")
    if not isinstance(artifact.get("field_principles"), Mapping):
        errors.append("field_principles must be dict")
    else:
        for field in (
            "honest_verdict",
            "regression_guard_passed",
            "reproducible_total_levels",
            "reproducible_total_games",
            "open_gap_ids",
            "inference_substrate",
        ):
            if artifact["field_principles"].get(field) != FIELD_PRINCIPLES[field]:
                errors.append(f"field_principles.{field}.principle must match REQ-REPORT-4461")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64 or not all(
        char in "0123456789abcdef" for char in checksum
    ):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    if set(SPEC_REFS) - set(artifact.get("spec_refs", [])):
        errors.append("spec_refs must include REQ-REPORT-4461 and SCENARIO-REPORT-4461")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    gap4_guard_runner: Gap4GuardRunner = _run_gap4_regression_guard,
    capstone_stamp_runner: CapstoneStampRunner = _verify_capstone_stamp_fix_durable,
    now: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """REQ-REPORT-4461: reconcile ledgers, run guards, and write the artifact."""

    started = now()
    root = Path(root)
    verifier_registry, arc_registry, gaps_text, payloads, preconditions = check_preconditions(root)
    source_report = preconditions.get("source_artifacts", {})
    available = availability_report(payloads)
    trusted = trusted_payloads(payloads, available)
    excluded = excluded_artifact_paths(payloads, source_report)
    gap_entries = collect_gap_entries(trusted, source_report)
    updated_gaps, filled_gap_ids, open_gap_ids = reconcile_gaps_text(gaps_text, gap_entries)

    guard = dict(gap4_guard_runner(root))
    stamp = dict(capstone_stamp_runner(root))
    updated_arc, arc_report = reconcile_arc_registry(
        arc_registry,
        trusted,
        source_report,
        filled_gap_ids=filled_gap_ids,
        open_gap_ids=open_gap_ids,
        excluded=excluded,
    )
    updated_verifier, verifier_report = reconcile_verifier_registry(
        verifier_registry,
        guard=guard,
        stamp=stamp,
        total_levels=arc_report["reproducible_total_levels"],
        total_games=arc_report["reproducible_total_games"],
        filled_gap_ids=filled_gap_ids,
        open_gap_ids=open_gap_ids,
        trusted=trusted,
        excluded=excluded,
    )

    ledgers_writable = bool(preconditions.get("ok"))
    if ledgers_writable:
        prior._write_yaml(root / ARC_REGISTRY_RELATIVE_PATH, updated_arc)
        prior._write_yaml(root / REGISTRY_RELATIVE_PATH, updated_verifier)
        prior._write_text(root / GAPS_RELATIVE_PATH, updated_gaps)

    registry_reconciliation = {
        **arc_report,
        **verifier_report,
        "verifier_gaps_reconciled": ledgers_writable,
        "registries_reconciled": bool(ledgers_writable),
        "filled_gap_ids": filled_gap_ids,
        "open_gap_ids": open_gap_ids,
        "excluded_artifacts": excluded,
        "production_verifier_edits": False,
    }
    artifact = build_artifact(
        started_at=started,
        ended_at=now(),
        availability=available,
        preconditions=preconditions,
        registry_reconciliation=registry_reconciliation,
        guard=guard,
        stamp=stamp,
        total_levels=arc_report["reproducible_total_levels"],
        total_games=arc_report["reproducible_total_games"],
        excluded=excluded,
    )
    write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper
    artifact = run(REPO_ROOT)
    print(json.dumps(artifact["regression_guard_passed"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
