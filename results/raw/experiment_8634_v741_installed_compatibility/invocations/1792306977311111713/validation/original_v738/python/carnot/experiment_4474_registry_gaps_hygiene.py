"""Exp 4474: reconcile .413 registry and verifier-gap hygiene.

Spec refs: REQ-REPORT-4474, SCENARIO-REPORT-4474.

This audit reads the .413 outcome artifacts, updates the ARC and verifier
registries, reconciles the gap ledger, runs the cached GAP-4 guard, and writes
the terminal hygiene artifact. It does not edit production verifier code.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import yaml

from carnot import experiment_4449_registry_gaps_hygiene as base
from carnot import experiment_4461_registry_gaps_hygiene as prior
from carnot.reporting import capstone_aggregate_available


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4474_registry_gaps_hygiene.json"
REGISTRY_RELATIVE_PATH = prior.REGISTRY_RELATIVE_PATH
GAPS_RELATIVE_PATH = prior.GAPS_RELATIVE_PATH
ARC_REGISTRY_RELATIVE_PATH = prior.ARC_REGISTRY_RELATIVE_PATH

EXP4467_PATH = "results/experiment_4467_solve_dc22_cegis_nocov.json"
EXP4468_PATH = "results/experiment_4468_bank_sc25_provisional_levels.json"
EXP4469_PATH = "results/experiment_4469_generic_cast_grid_fsm_operator.json"
EXP4470_PATH = "results/experiment_4470_color_match_slot_operator_solve_sb26.json"
EXP4471_PATH = "results/experiment_4471_first_contact_rotated_new_game.json"
EXP4472_PATH = "results/experiment_4472_variant_generic_transfer_benchmark_v4.json"
EXP4473_PATH = "results/experiment_4473_submission_package_prep_refresh.json"

SOURCE_ARTIFACTS = {
    "4467_dc22": EXP4467_PATH,
    "4468_sc25_deep": EXP4468_PATH,
    "4469_sc25_operator": EXP4469_PATH,
    "4470_sb26": EXP4470_PATH,
    "4471_first_contact": EXP4471_PATH,
    "4472_variant_loo_v4": EXP4472_PATH,
    "4473_submission": EXP4473_PATH,
}
SOURCE_EXPERIMENT_IDS = {
    "4467_dc22": 4467,
    "4468_sc25_deep": 4468,
    "4469_sc25_operator": 4469,
    "4470_sb26": 4470,
    "4471_first_contact": 4471,
    "4472_variant_loo_v4": 4472,
    "4473_submission": 4473,
}

RANDOM_SEED = 4474
SPEC_REFS = ("REQ-REPORT-4474", "SCENARIO-REPORT-4474")
INFERENCE_SUBSTRATE = prior.INFERENCE_SUBSTRATE
GAP4_VERIFIER_ID = prior.GAP4_VERIFIER_ID
V413_ROLE_ID = "oracle_distinct_v413_registry_gaps_hygiene_4474"
TERMINAL_PREFIXES = prior.TERMINAL_PREFIXES

DC22_GAP_ID = "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT"
SC25_GAP_ID = "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER"
SB26_GAP_ID = "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE"
RE86_GAP_ID = "GAP-4471-RE86-MISSING-PATTERN-MATCH-SPRITE-RESIZE-VERIFIER"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "reproducible_total_levels",
    "reproducible_total_games",
    "provisional_total_levels",
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
        "principle": "the reconciled authoritative count (target > 39 after dc22 + sc25 deepening)"
    },
    "reproducible_total_games": {
        "principle": "the reconciled authoritative game count (target >= 21 after dc22)"
    },
    "provisional_total_levels": {
        "principle": "the reconciled provisional count (target < 5 after sc25 deeper levels move provisional -> reproduced)"
    },
    "open_gap_ids": {
        "principle": "the still-open generic-solver gaps after .413 -- the .414 backlog"
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
    DC22_GAP_ID: "exp4438-gap-4423-dc22-unselectable-first-contact",
    SC25_GAP_ID: "exp4438-gap-4432-loo-sc25-missing-cast-grid-spell-shrink-tank-exit-verifier",
    SB26_GAP_ID: "exp4458-gap-sb26-color-match-slot-sequence",
    RE86_GAP_ID: "exp4471-gap-re86-pattern-match-sprite-resize",
}

Gap4GuardRunner = Callable[[Path], Mapping[str, Any]]
CapstoneStampRunner = Callable[[Path], Mapping[str, Any]]


def _run_gap4_regression_guard(root: Path) -> Mapping[str, Any]:  # pragma: no cover
    return prior._run_gap4_regression_guard(root)


def _verify_capstone_stamp_fix_durable(root: Path) -> Mapping[str, Any]:  # pragma: no cover
    return prior._verify_capstone_stamp_fix_durable(root)


def _as_int(value: Any) -> int:
    return prior._as_int(value)


def _source_path(source_report: Mapping[str, Mapping[str, Any]], key: str) -> str:
    return str(source_report.get(key, {}).get("relative_path") or SOURCE_ARTIFACTS[key])


def _dc22_banked(payload: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and payload.get("target_game") == "dc22"
        and payload.get("offline_reproduced") is True
        and payload.get("dc22_grounded") is not False
        and _as_int(payload.get("reproduced_levels")) >= 1
    )


def _sc25_deeper_levels(payload: Mapping[str, Any] | None) -> tuple[int, int]:
    if not isinstance(payload, Mapping):
        return 0, 0
    new_levels = max(
        _as_int(payload.get("new_sc25_levels_reproduced")),
        _as_int(payload.get("reproduced_levels")),
    )
    total_levels = max(
        _as_int(payload.get("sc25_levels_reproduced_total")),
        new_levels + 1 if new_levels else 0,
    )
    return new_levels, total_levels


def _sc25_deep_banked(payload: Mapping[str, Any] | None) -> bool:
    new_levels, total_levels = _sc25_deeper_levels(payload)
    return bool(
        isinstance(payload, Mapping)
        and payload.get("target_game") == "sc25"
        and payload.get("offline_reproduced") is True
        and new_levels > 0
        and total_levels >= new_levels
    )


def _sc25_operator_closed(payload: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and payload.get("target_game") == "sc25"
        and payload.get("offline_reproduced") is True
        and payload.get("sc25_resolved_generically") is True
        and _as_int(payload.get("sc25_generic_level_reproduced")) >= 1
    )


def _sb26_banked(payload: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and payload.get("target_game") == "sb26"
        and payload.get("offline_reproduced") is True
        and payload.get("color_match_operator_built") is True
        and _as_int(payload.get("reproduced_levels")) >= 1
    )


def _first_contact_banked(payload: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and payload.get("offline_reproduced") is True
        and _as_int(payload.get("reproduced_levels")) >= 1
        and str(payload.get("target_game") or "")
    )


def _first_contact_gap_id(payload: Mapping[str, Any] | None) -> str:
    if isinstance(payload, Mapping):
        gaps = payload.get("missing_verifier_gaps")
        if isinstance(gaps, list):
            for row in gaps:
                if isinstance(row, Mapping) and isinstance(row.get("gap_id"), str):
                    return str(row["gap_id"])
    return RE86_GAP_ID


def _variant_verdict(payload: Mapping[str, Any] | None) -> dict[str, int]:
    source = payload if isinstance(payload, Mapping) else {}
    return {
        "generic_loo_solve_count_v4": _as_int(source.get("generic_loo_solve_count_v4")),
        "variants_attempted": _as_int(source.get("variants_attempted")),
        "variants_solved": _as_int(source.get("variants_solved")),
    }


def _submission_verdict(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, Mapping) else {}
    return {
        "ready": source.get("submission_package_ready") is True,
        "levels": _as_int(source.get("total_reproduced_levels_in_package")),
        "submitted_to_leaderboard": source.get("submitted_to_leaderboard") is True,
    }


def load_sources(root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """REQ-REPORT-4474: read every upstream artifact before reconciling ledgers."""

    payloads: dict[str, Any] = {}
    report: dict[str, dict[str, Any]] = {}
    for key, rel_path in SOURCE_ARTIFACTS.items():
        payload, row = base._load_json(root / rel_path)
        payloads[key] = payload
        row["relative_path"] = rel_path
        row["source_pattern"] = rel_path
        report[key] = row
    return payloads, report


def _axis_specs() -> list[capstone_aggregate_available.AxisSpec]:
    return [
        capstone_aggregate_available.AxisSpec(
            name="dc22_bank",
            required_keys=("4467_dc22",),
            verdict_fn=lambda present: _dc22_banked(present.get("4467_dc22")),
        ),
        capstone_aggregate_available.AxisSpec(
            name="sc25_deeper_bank",
            required_keys=("4468_sc25_deep",),
            verdict_fn=lambda present: {
                "new_levels": _sc25_deeper_levels(present.get("4468_sc25_deep"))[0],
                "total_levels": _sc25_deeper_levels(present.get("4468_sc25_deep"))[1],
                "moved_from_provisional": _sc25_deep_banked(present.get("4468_sc25_deep")),
            },
        ),
        capstone_aggregate_available.AxisSpec(
            name="sc25_generic_operator",
            required_keys=("4469_sc25_operator",),
            verdict_fn=lambda present: _sc25_operator_closed(present.get("4469_sc25_operator")),
        ),
        capstone_aggregate_available.AxisSpec(
            name="sb26_bank",
            required_keys=("4470_sb26",),
            verdict_fn=lambda present: _sb26_banked(present.get("4470_sb26")),
        ),
        capstone_aggregate_available.AxisSpec(
            name="first_contact_new_game",
            required_keys=("4471_first_contact",),
            verdict_fn=lambda present: {
                "target_game": (present.get("4471_first_contact") or {}).get("target_game"),
                "banked": _first_contact_banked(present.get("4471_first_contact")),
                "gap_id": _first_contact_gap_id(present.get("4471_first_contact")),
            },
        ),
        capstone_aggregate_available.AxisSpec(
            name="variant_transfer_loo_v4",
            required_keys=("4472_variant_loo_v4",),
            verdict_fn=lambda present: _variant_verdict(present.get("4472_variant_loo_v4")),
        ),
        capstone_aggregate_available.AxisSpec(
            name="submission_package",
            required_keys=("4473_submission",),
            verdict_fn=lambda present: _submission_verdict(present.get("4473_submission")),
        ),
    ]


def availability_report(payloads: Mapping[str, Any]) -> dict[str, Any]:
    """REQ-REPORT-4474: report per-axis gaps without poisoning unrelated axes."""

    return capstone_aggregate_available.aggregate_available_report_gaps(
        payloads,
        _axis_specs(),
        artifact_experiment_ids=SOURCE_EXPERIMENT_IDS,
    )


def trusted_payloads(
    payloads: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
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
    """SCENARIO-REPORT-4474: convert .413 outcomes into filled or open gap rows."""

    entries: dict[str, dict[str, Any]] = {}
    exp4467 = trusted.get("4467_dc22", {})
    dc22_path = _source_path(source_report, "4467_dc22")
    if _dc22_banked(exp4467):
        entries[DC22_GAP_ID] = _gap_entry(
            DC22_GAP_ID,
            status="filled (experiment_4467_solve_dc22_cegis_nocov)",
            evidence=(
                f"{dc22_path}; target_game=dc22; offline_reproduced=True; "
                f"dc22_grounded={exp4467.get('dc22_grounded')}; "
                f"reproduced_levels={exp4467.get('reproduced_levels')}"
            ),
            failure_mode="closed_by_dc22_cegis_config_rule",
            missing_discriminator="filled by execution-grounded buezna toggle plus jfva->goknoi navigation predicate",
            candidate_design="keep dc22_toggle_navigation in config_rule_verifier and the dc22 GameAdapter",
            source_artifact=dc22_path,
            movement="filled",
        )
    else:
        entries[DC22_GAP_ID] = _gap_entry(
            DC22_GAP_ID,
            status="open",
            evidence=f"{dc22_path}; clean dc22 offline reproduction not available",
            failure_mode="missing_dc22_reproduction_gate",
            missing_discriminator="dc22 still lacks clean reproduction-gated generic closure",
            candidate_design="rerun dc22 CEGIS/config-rule grounding and count only reproduce(dc22)",
            source_artifact=dc22_path,
            movement="updated_still_open",
        )

    exp4469 = trusted.get("4469_sc25_operator", {})
    sc25_path = _source_path(source_report, "4469_sc25_operator")
    if _sc25_operator_closed(exp4469):
        entries[SC25_GAP_ID] = _gap_entry(
            SC25_GAP_ID,
            status="filled (experiment_4469_generic_cast_grid_fsm_operator)",
            evidence=(
                f"{sc25_path}; sc25_resolved_generically=True; "
                f"sc25_generic_level_reproduced={exp4469.get('sc25_generic_level_reproduced')}; "
                "offline_reproduced=True"
            ),
            failure_mode="closed_by_cast_grid_phase_fsm_world_model",
            missing_discriminator="filled by execution-grounded cast_grid_phase_fsm_world_model",
            candidate_design="reuse two-phase cast/config toggle then navigation FSMs for future cast-grid games",
            source_artifact=sc25_path,
            movement="filled",
        )
    else:
        entries[SC25_GAP_ID] = _gap_entry(
            SC25_GAP_ID,
            status="open",
            evidence=f"{sc25_path}; clean sc25 generic cast-grid closure not available",
            failure_mode="missing_cast_grid_spell_shrink_tank_exit_verifier",
            missing_discriminator="generic cast-grid spell/shrink/tank-exit verifier still missing",
            candidate_design="build cast_grid_phase_fsm_world_model and rerun LOO v4",
            source_artifact=sc25_path,
            movement="updated_still_open",
        )

    exp4470 = trusted.get("4470_sb26", {})
    sb26_path = _source_path(source_report, "4470_sb26")
    if _sb26_banked(exp4470):
        entries[SB26_GAP_ID] = _gap_entry(
            SB26_GAP_ID,
            status="filled (experiment_4470_color_match_slot_operator_solve_sb26)",
            evidence=(
                f"{sb26_path}; color_match_operator_built=True; offline_reproduced=True; "
                f"reproduced_levels={exp4470.get('reproduced_levels')}; "
                f"counterexample_rounds={exp4470.get('counterexample_rounds')}"
            ),
            failure_mode="closed_by_color_match_slot_sequence_verifier",
            missing_discriminator="filled by execution-grounded ordered color-match item-slot verifier with undo-aware grounding",
            candidate_design="reuse color_match_slot_sequence_verifier for ordered item-slot color puzzles",
            source_artifact=sb26_path,
            movement="filled",
        )
    else:
        entries[SB26_GAP_ID] = _gap_entry(
            SB26_GAP_ID,
            status="open",
            evidence=f"{sb26_path}; clean sb26 color-match offline reproduction not available",
            failure_mode="missing_color_match_slot_sequence_verifier",
            missing_discriminator="generic ordered color-match item-slot verifier with undo-aware grounding",
            candidate_design="extend config_rule_verifier with color_match_slot_sequence digests",
            source_artifact=sb26_path,
            movement="updated_still_open",
        )

    exp4471 = trusted.get("4471_first_contact", {})
    first_path = _source_path(source_report, "4471_first_contact")
    first_gap_id = _first_contact_gap_id(exp4471)
    target = str(exp4471.get("target_game") or "re86")
    if _first_contact_banked(exp4471):
        entries[first_gap_id] = _gap_entry(
            first_gap_id,
            status="filled (experiment_4471_first_contact_rotated_new_game)",
            evidence=(
                f"{first_path}; target_game={target}; offline_reproduced=True; "
                f"reproduced_levels={exp4471.get('reproduced_levels')}"
            ),
            failure_mode="closed_by_rotated_first_contact",
            missing_discriminator="filled by routed rotated-game generic verifier",
            candidate_design="reuse the routed first-contact primitive",
            source_artifact=first_path,
            movement="filled",
        )
    else:
        entries[first_gap_id] = _gap_entry(
            first_gap_id,
            status="open",
            evidence=(
                f"{first_path}; target_game={target}; routed_to={exp4471.get('routed_to')}; "
                f"offline_reproduced={exp4471.get('offline_reproduced')}; "
                f"reproduced_levels={exp4471.get('reproduced_levels')}"
            ),
            failure_mode="missing_pattern_match_sprite_resize_verifier",
            missing_discriminator="generic sprite-overlay pattern-match and resize/transformation verifier",
            candidate_design="extend graph/object operators with exact overlay and ACTION5 resize grounding",
            source_artifact=first_path,
            movement="updated_still_open",
        )

    return list(entries.values())


def _gap_marker(gap_id: str) -> str:
    return GAP_MARKER_OVERRIDES.get(gap_id, f"exp4474-{base._slug(gap_id).lower()}")


def _gap_block(gap: Mapping[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4474 .413 registry gap hygiene\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'high')}\n"
        f"- source artifact: {gap.get('source_artifact', '')}\n"
        f"- movement: {gap.get('movement', 'updated')}\n"
    )


def reconcile_gaps_text(
    gaps_text: str, gap_entries: list[dict[str, Any]]
) -> tuple[str, list[str], list[str]]:
    updated = gaps_text
    filled: list[str] = []
    open_ids: list[str] = []
    for gap in gap_entries:
        gap_id = str(gap["gap_id"])
        if str(gap.get("status", "")).startswith("filled"):
            filled.append(gap_id)
        else:
            open_ids.append(gap_id)
        updated = base._replace_marked_block(updated, _gap_marker(gap_id), _gap_block(gap))
    return updated, filled, open_ids


def _game_levels(registry: Mapping[str, Any], game: str) -> int:
    games = registry.get("games")
    if not isinstance(games, list):
        return 0
    for row in games:
        if isinstance(row, Mapping) and row.get("game") == game:
            return _as_int(row.get("levels_reproduced"))
    return 0


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
    base._record_reproduced_game(
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
    """REQ-REPORT-4474: update reproduced and provisional ARC counts."""

    updated = deepcopy(dict(registry))
    updated.setdefault("games", [])

    exp4467 = trusted.get("4467_dc22", {})
    if _dc22_banked(exp4467):
        _record_reproduced_game(
            updated,
            game="dc22",
            levels=max(1, _as_int(exp4467.get("reproduced_levels"))),
            artifact_path=_source_path(source_report, "4467_dc22"),
            latest_key="latest_exp4467_reproduce",
            checksum=str(exp4467.get("reproducibility_checksum") or ""),
            solver="python/carnot/experiment_4467_solve_dc22_cegis_nocov.py",
        )

    exp4468 = trusted.get("4468_sc25_deep", {})
    sc25_prior_levels = _game_levels(updated, "sc25")
    if _sc25_deep_banked(exp4468):
        new_levels, total_levels = _sc25_deeper_levels(exp4468)
        _record_reproduced_game(
            updated,
            game="sc25",
            levels=max(1, total_levels),
            artifact_path=_source_path(source_report, "4468_sc25_deep"),
            latest_key="latest_exp4468_reproduce",
            checksum=str(exp4468.get("reproducibility_checksum") or ""),
            solver="python/carnot/experiment_4468_bank_sc25_provisional_levels.py",
        )
        provisional_delta = max(0, total_levels - sc25_prior_levels)
        updated["provisional_total_levels"] = max(
            0,
            _as_int(updated.get("provisional_total_levels")) - provisional_delta,
        )
        base._ensure_game(updated, "sc25")["latest_exp4468_reproduce"].update(
            {
                "new_sc25_levels_reproduced": new_levels,
                "sc25_levels_reproduced_total": total_levels,
            }
        )
    else:
        updated["provisional_total_levels"] = _as_int(updated.get("provisional_total_levels"))

    exp4470 = trusted.get("4470_sb26", {})
    if _sb26_banked(exp4470):
        _record_reproduced_game(
            updated,
            game="sb26",
            levels=max(1, _as_int(exp4470.get("reproduced_levels"))),
            artifact_path=_source_path(source_report, "4470_sb26"),
            latest_key="latest_exp4470_color_match",
            checksum=str(exp4470.get("reproducibility_checksum") or ""),
            solver="python/carnot/experiment_4470_color_match_slot_operator_solve_sb26.py",
        )
        base._ensure_game(updated, "sb26")["latest_exp4470_color_match"]["operator"] = (
            "color_match_slot_sequence_verifier"
        )

    exp4471 = trusted.get("4471_first_contact", {})
    if _first_contact_banked(exp4471):
        _record_reproduced_game(
            updated,
            game=str(exp4471.get("target_game") or "re86"),
            levels=max(1, _as_int(exp4471.get("reproduced_levels"))),
            artifact_path=_source_path(source_report, "4471_first_contact"),
            latest_key="latest_exp4471_first_contact",
            checksum=str(exp4471.get("reproducibility_checksum") or ""),
            solver="python/carnot/experiment_4471_first_contact_rotated_new_game.py",
        )
    elif exp4471:
        row = base._ensure_game(updated, str(exp4471.get("target_game") or "re86"))
        row["reproducibility"] = row.get("reproducibility") or "unsolved"
        row["latest_exp4471_first_contact"] = {
            "artifact": _source_path(source_report, "4471_first_contact"),
            "offline_reproduced": exp4471.get("offline_reproduced") is True,
            "reproduced_levels": _as_int(exp4471.get("reproduced_levels")),
            "gap_id": _first_contact_gap_id(exp4471),
        }

    exp4469 = trusted.get("4469_sc25_operator", {})
    if exp4469:
        row = base._ensure_game(updated, "sc25")
        row["latest_exp4469_generic_cast_grid"] = {
            "artifact": _source_path(source_report, "4469_sc25_operator"),
            "operator": "cast_grid_phase_fsm_world_model",
            "sc25_resolved_generically": _sc25_operator_closed(exp4469),
            "sc25_generic_level_reproduced": _as_int(exp4469.get("sc25_generic_level_reproduced")),
            "offline_reproduced": exp4469.get("offline_reproduced") is True,
            "reproducibility_checksum": str(exp4469.get("reproducibility_checksum") or ""),
        }

    exp4472 = trusted.get("4472_variant_loo_v4", {})
    if exp4472:
        updated["latest_variant_generic_transfer_v4_4472"] = {
            "artifact": _source_path(source_report, "4472_variant_loo_v4"),
            **_variant_verdict(exp4472),
            "generic_loo_solve_count_v3_baseline": exp4472.get(
                "generic_loo_solve_count_v3_baseline"
            ),
            "honest_verdict": exp4472.get("honest_verdict"),
        }

    exp4473 = trusted.get("4473_submission", {})
    if exp4473:
        updated["latest_submission_package_4473"] = {
            "artifact": _source_path(source_report, "4473_submission"),
            **_submission_verdict(exp4473),
        }

    total_levels, total_games = base._reproduced_counts(updated)
    updated["updated"] = "2026-06-20"
    updated["reproducible_total_levels"] = total_levels
    updated["reproducible_total_games"] = total_games
    provisional_total = _as_int(updated.get("provisional_total_levels"))
    updated["provisional_total_levels"] = provisional_total
    updated["latest_hygiene_4474"] = {
        "artifact": RESULT_RELATIVE_PATH,
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "provisional_total_levels": provisional_total,
        "filled_gap_ids": filled_gap_ids,
        "open_gap_ids": open_gap_ids,
        "excluded_artifacts": excluded,
        "note": ".413 registry hygiene; missing, blocked, or flagged artifacts excluded from counts.",
    }
    return updated, {
        "arc_solve_registry_reconciled": True,
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "provisional_total_levels": provisional_total,
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
    provisional_total: int,
    filled_gap_ids: list[str],
    open_gap_ids: list[str],
    trusted: Mapping[str, Mapping[str, Any]],
    excluded: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = deepcopy(dict(registry))
    verifier = base._find_verifier(updated, GAP4_VERIFIER_ID)
    if verifier is None:  # pragma: no cover - defensive creation path
        verifier = {
            "verifier_id": GAP4_VERIFIER_ID,
            "domain": "arc_agi2_grid",
            "kind": "process_verifier",
            "eval": {},
            "registry_roles": [],
        }
        updated.setdefault("verifiers", []).append(verifier)

    current = base._guard_current(guard)
    exp4468 = trusted.get("4468_sc25_deep", {})
    exp4472 = trusted.get("4472_variant_loo_v4", {})
    exp4473 = trusted.get("4473_submission", {})
    verifier.setdefault("eval", {}).update(
        {
            "eval_exp_4474": RESULT_RELATIVE_PATH,
            "exp4474_regression_guard_passed": guard_passed(guard),
            "exp4474_arc_oracle_distinct_verifier_beats_vote": (
                guard.get("arc_oracle_distinct_verifier_beats_vote") is not False
            ),
            "exp4474_arc1_rule_exec_vote_pass2": current.get("vote_pass2"),
            "exp4474_arc1_rule_exec_gated_pass2": current.get("gated_pass2"),
            "exp4474_arc1_headroom_recovered": current.get("headroom_recovered"),
            "exp4474_arc1_vote_wins_lost": current.get("vote_wins_lost"),
            "exp4474_capstone_stamp_fix_durable": stamp_fix_durable(stamp),
            "exp4474_reproducible_total_levels": total_levels,
            "exp4474_reproducible_total_games": total_games,
            "exp4474_provisional_total_levels": provisional_total,
            "exp4474_filled_gap_ids": filled_gap_ids,
            "exp4474_open_gap_ids": open_gap_ids,
            "exp4474_flagged_artifacts_excluded": excluded,
            "exp4474_sc25_new_levels_banked": exp4468.get("new_sc25_levels_reproduced"),
            "exp4474_generic_loo_solve_count_v4": exp4472.get("generic_loo_solve_count_v4"),
            "exp4474_submission_package_ready": exp4473.get("submission_package_ready"),
            "exp4474_submission_package_levels": exp4473.get("total_reproduced_levels_in_package"),
        }
    )

    role = {
        "role_id": V413_ROLE_ID,
        "experiment": RESULT_RELATIVE_PATH,
        "role": "registry_gaps_arc_hygiene_v413",
        "status": "v413_outcomes_recorded_with_gap4_guard_and_stamp_durability",
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "provisional_total_levels": provisional_total,
        "filled_gap_ids": filled_gap_ids,
        "open_gap_ids": open_gap_ids,
        "excluded_artifacts": excluded,
        "eval_exp_4474": RESULT_RELATIVE_PATH,
    }
    roles = verifier.setdefault("registry_roles", [])
    if not isinstance(roles, list):  # pragma: no cover - defensive cleanup path
        roles = verifier["registry_roles"] = []
    verifier["registry_roles"] = [
        row
        for row in roles
        if not (isinstance(row, Mapping) and row.get("role_id") == V413_ROLE_ID)
    ] + [role]
    return updated, {"verifier_registry_reconciled": True}


def check_preconditions(
    root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    verifier_registry, verifier_check = base._yaml_mapping(root / REGISTRY_RELATIVE_PATH)
    arc_registry, arc_check = base._yaml_mapping(root / ARC_REGISTRY_RELATIVE_PATH)
    gaps_text, gaps_check = base._read_text(root / GAPS_RELATIVE_PATH)
    payloads, source_report = load_sources(root)
    helper_check = base._check_helper_import()
    checks = {
        "ok": verifier_check["readable"]
        and arc_check["readable"]
        and gaps_check["readable"]
        and helper_check["ok"],
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
    return base._sha256(
        {
            "registry_reconciliation": artifact.get("registry_reconciliation"),
            "availability_report": artifact.get("availability_report"),
            "regression_guard_passed": artifact.get("regression_guard_passed"),
            "capstone_stamp_fix_durable": artifact.get("capstone_stamp_fix_durable"),
            "reproducible_total_levels": artifact.get("reproducible_total_levels"),
            "reproducible_total_games": artifact.get("reproducible_total_games"),
            "provisional_total_levels": artifact.get("provisional_total_levels"),
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
    provisional_total: int,
    excluded: list[str],
) -> dict[str, Any]:
    guard_ok = guard_passed(guard)
    stamp_ok = stamp_fix_durable(stamp)
    suffix = "guard_passed" if guard_ok else "guard_failed"
    artifact: dict[str, Any] = {
        "experiment": "experiment_4474_registry_gaps_hygiene",
        "schema": "carnot.exp4474.registry_gaps_hygiene.v1",
        "honest_verdict": f"complete: registry_gaps_hygiene_4474_{suffix}",
        "regression_guard_passed": guard_ok,
        "reproducible_total_levels": int(total_levels),
        "reproducible_total_games": int(total_games),
        "provisional_total_levels": int(provisional_total),
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
    if type(artifact.get("provisional_total_levels")) is not int:
        errors.append("provisional_total_levels must be bare int")
    open_gap_ids = artifact.get("open_gap_ids")
    if not isinstance(open_gap_ids, list) or any(
        not isinstance(item, str) for item in open_gap_ids
    ):
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
            "provisional_total_levels",
            "open_gap_ids",
            "inference_substrate",
        ):
            if artifact["field_principles"].get(field) != FIELD_PRINCIPLES[field]:
                errors.append(f"field_principles.{field}.principle must match REQ-REPORT-4474")
    checksum = artifact.get("reproducibility_checksum")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or not all(char in "0123456789abcdef" for char in checksum)
    ):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    if set(SPEC_REFS) - set(artifact.get("spec_refs", [])):
        errors.append("spec_refs must include REQ-REPORT-4474 and SCENARIO-REPORT-4474")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    gap4_guard_runner: Gap4GuardRunner = _run_gap4_regression_guard,
    capstone_stamp_runner: CapstoneStampRunner = _verify_capstone_stamp_fix_durable,
    now: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """REQ-REPORT-4474: reconcile ledgers, run guards, and write the artifact."""

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
        provisional_total=arc_report["provisional_total_levels"],
        filled_gap_ids=filled_gap_ids,
        open_gap_ids=open_gap_ids,
        trusted=trusted,
        excluded=excluded,
    )

    ledgers_writable = bool(preconditions.get("ok"))
    if ledgers_writable:
        base._write_yaml(root / ARC_REGISTRY_RELATIVE_PATH, updated_arc)
        base._write_yaml(root / REGISTRY_RELATIVE_PATH, updated_verifier)
        base._write_text(root / GAPS_RELATIVE_PATH, updated_gaps)

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
        provisional_total=arc_report["provisional_total_levels"],
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
