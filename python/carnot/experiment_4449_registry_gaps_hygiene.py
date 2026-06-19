"""Exp 4449: reconcile .411 registry and verifier-gap hygiene.

Spec refs: REQ-REPORT-4449, SCENARIO-REPORT-4449.

This audit pass reads the .411 outcome artifacts, refreshes the registry and
gap ledgers, runs the cached GAP-4 execution guard, and writes a terminal
hygiene artifact. It deliberately does not edit production verifier code.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping

import yaml

from carnot.reporting import capstone_aggregate_available


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4449_registry_gaps_hygiene.json"
REGISTRY_RELATIVE_PATH = "ops/verifier_registry.yaml"
GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
ARC_REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"

EXP4443_PATH = "results/experiment_4443_bank_g50t_example_conditioned_win.json"
EXP4444_PATH = "results/experiment_4444_generic_config_rule_verifier_operator.json"
EXP4445_PATH = "results/experiment_4445_generic_object_motion_world_model_operator.json"
EXP4446_PATH = "results/experiment_4446_drive_generic_first_contact_bank.json"
EXP4447_PATH = "results/experiment_4447_lilo_documented_primitive_library.json"
EXP4448_PATH = "results/experiment_4448_loo_generic_solve_benchmark_v2.json"

SOURCE_ARTIFACTS = {
    "4443_g50t_bank": EXP4443_PATH,
    "4444_config_rule": EXP4444_PATH,
    "4445_object_motion": EXP4445_PATH,
    "4446_first_contact": EXP4446_PATH,
    "4447_primitive_library": EXP4447_PATH,
    "4448_loo_v2": EXP4448_PATH,
}
SOURCE_EXPERIMENT_IDS = {
    "4443_g50t_bank": 4443,
    "4444_config_rule": 4444,
    "4445_object_motion": 4445,
    "4446_first_contact": 4446,
    "4447_primitive_library": 4447,
    "4448_loo_v2": 4448,
}

RANDOM_SEED = 4449
SPEC_REFS = ("REQ-REPORT-4449", "SCENARIO-REPORT-4449")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
GAP4_VERIFIER_ID = "gap4_program_induction_stack"
V411_ROLE_ID = "oracle_distinct_v411_registry_gaps_hygiene_4449"
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "reproducible_total_levels",
    "reproducible_total_games",
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
        "principle": "the reconciled authoritative count (target >= 38 after the g50t bank)"
    },
    "reproducible_total_games": {"principle": "the reconciled authoritative game count (target >= 19)"},
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts -- reads upstream artifacts; 100us floor"
    },
    "availability_report": {
        "principle": "robust aggregate-available report; missing or flagged inputs do not erase other axes"
    },
    "submitted_to_leaderboard": {"principle": "must remain false for this audit-only task"},
}

GAP_MARKER_OVERRIDES = {
    "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT": "exp4427-gap-4423-g50t-unselectable-first-contact",
    "GAP-4432-LOO-KA59-MISSING-PUSH-BLOCK-WORLD-MODEL-AND-DYNAMIC-SELECTION": (
        "exp4438-gap-4432-loo-ka59-missing-push-block-world-model-and-dynamic-selection"
    ),
    "GAP-4432-LOO-AR25-MISSING-REFLECTION-WORLD-MODEL-AND-OBJECT-MOTION-PLAN": (
        "exp4438-gap-4432-loo-ar25-missing-reflection-world-model-and-object-motion-plan"
    ),
    "GAP-4432-LOO-FT09-MISSING-LOCAL-CONSTRAINT-COLOR-CYCLE-VERIFIER": (
        "exp4438-gap-4432-loo-ft09-missing-local-constraint-color-cycle-verifier"
    ),
    "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER": (
        "exp4438-gap-4432-loo-tr87-missing-glyph-rewrite-rule-verifier-without-tr87-adapter"
    ),
    "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER": (
        "exp4438-gap-4432-loo-sc25-missing-cast-grid-spell-shrink-tank-exit-verifier"
    ),
    "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT": "exp4438-gap-4423-dc22-unselectable-first-contact",
    "GAP-4423-VC33-UNSELECTABLE-FIRST-CONTACT": "exp4446-gap-4423-vc33-unselectable-first-contact",
}

Gap4GuardRunner = Callable[[Path], Mapping[str, Any]]
CapstoneStampRunner = Callable[[Path], Mapping[str, Any]]


def _run_gap4_regression_guard(root: Path) -> Mapping[str, Any]:  # pragma: no cover - live guard boundary
    from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4410 as exp4410

    return exp4410.run_gap4_regression_guard(root)


def _verify_capstone_stamp_fix_durable(root: Path) -> Mapping[str, Any]:  # pragma: no cover
    from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4410 as exp4410

    return exp4410.verify_capstone_stamp_fix_durable(root)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _slug(value: Any) -> str:
    text = str(value or "").strip().upper()
    return re.sub(r"[^A-Z0-9]+", "-", text).strip("-")


def _yaml_mapping(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, {
            "path": str(path),
            "readable": False,
            "yaml_safe_load": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(loaded, dict):
        return {}, {
            "path": str(path),
            "readable": False,
            "yaml_safe_load": True,
            "error": "top-level YAML is not a mapping",
        }
    return loaded, {"path": str(path), "readable": True, "yaml_safe_load": True, "error": ""}


def _read_text(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return "", {"path": str(path), "readable": False, "error": f"{type(exc).__name__}: {exc}"}
    return text, {"path": str(path), "readable": True, "error": ""}


def _load_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, {
            "path": str(path),
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(loaded, dict):
        return None, {"path": str(path), "available": False, "error": "top-level JSON is not an object"}
    return loaded, {
        "path": str(path),
        "available": True,
        "error": "",
        "flagged_adversarial": loaded.get("flagged_adversarial") is True,
        "honest_verdict": str(loaded.get("honest_verdict") or ""),
        "reproducibility_checksum": str(loaded.get("reproducibility_checksum") or ""),
    }


def load_sources(root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """REQ-REPORT-4449: read every upstream artifact before reconciling ledgers."""

    payloads: dict[str, Any] = {}
    report: dict[str, dict[str, Any]] = {}
    for key, rel_path in SOURCE_ARTIFACTS.items():
        payload, row = _load_json(root / rel_path)
        payloads[key] = payload
        row["relative_path"] = rel_path
        report[key] = row
    return payloads, report


def _axis_specs() -> list[capstone_aggregate_available.AxisSpec]:
    return [
        capstone_aggregate_available.AxisSpec(
            name="g50t_bank",
            required_keys=("4443_g50t_bank",),
            verdict_fn=lambda present: (
                present["4443_g50t_bank"].get("offline_reproduced") is True
                and _as_int(present["4443_g50t_bank"].get("reproduced_levels")) >= 1
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="config_rule",
            required_keys=("4444_config_rule",),
            verdict_fn=lambda present: {
                "ft09_resolved_generically": present["4444_config_rule"].get(
                    "ft09_resolved_generically"
                )
                is True,
                "dc22_state": present["4444_config_rule"].get("dc22_state"),
            },
        ),
        capstone_aggregate_available.AxisSpec(
            name="object_motion",
            required_keys=("4445_object_motion",),
            verdict_fn=lambda present: list(
                present["4445_object_motion"].get("residuals_closed_generically") or []
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="first_contact",
            required_keys=("4446_first_contact",),
            verdict_fn=lambda present: (
                present["4446_first_contact"].get("offline_reproduced") is True
                and _as_int(present["4446_first_contact"].get("reproduced_levels")) >= 1
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="primitive_library",
            required_keys=("4447_primitive_library",),
            verdict_fn=lambda present: _as_float(
                present["4447_primitive_library"].get("library_coverage")
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="loo_v2",
            required_keys=("4448_loo_v2",),
            verdict_fn=lambda present: _as_int(
                present["4448_loo_v2"].get("generic_loo_solve_count_v2")
            ),
        ),
    ]


def availability_report(payloads: Mapping[str, Any]) -> dict[str, Any]:
    """REQ-REPORT-4449: report per-axis gaps without poisoning unrelated axes."""

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


def excluded_artifact_paths(payloads: Mapping[str, Any]) -> list[str]:
    return [
        SOURCE_ARTIFACTS[key]
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
    priority: str = "high",
    movement: str = "updated",
) -> dict[str, Any]:
    return {
        "gap_id": gap_id,
        "status": status,
        "evidence": evidence,
        "failure_mode": failure_mode,
        "missing_discriminator": missing_discriminator,
        "candidate_design": candidate_design,
        "priority": priority,
        "source_artifact": source_artifact,
        "movement": movement,
    }


def _loo_gap_id(game: str, residual: str) -> str:
    return f"GAP-4432-LOO-{_slug(game)}-{_slug(residual)}"


def _add_gap(entries: dict[str, dict[str, Any]], gap: dict[str, Any]) -> None:
    entries.setdefault(gap["gap_id"], gap)


def collect_gap_entries(trusted: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """SCENARIO-REPORT-4449: convert .411 outcomes into filled or open gap rows."""

    entries: dict[str, dict[str, Any]] = {}
    exp4443 = trusted.get("4443_g50t_bank", {})
    if exp4443.get("offline_reproduced") is True and _as_int(exp4443.get("reproduced_levels")) >= 1:
        _add_gap(
            entries,
            _gap_entry(
                "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT",
                status="filled (exp4443_bank_g50t_example_conditioned_win)",
                evidence=(
                    f"{EXP4443_PATH}; target_game=g50t; offline_reproduced=True; "
                    f"reproduced_levels={exp4443.get('reproduced_levels')}; "
                    f"reproducible_total_levels={exp4443.get('reproducible_total_levels')}"
                ),
                failure_mode="prior first-contact route could not select the winning target-offset predicate",
                missing_discriminator="filled by execution-grounded target-offset verifier",
                candidate_design="keep the target-offset config-rule verifier in the generic bank",
                source_artifact=EXP4443_PATH,
                movement="filled",
            ),
        )

    exp4444 = trusted.get("4444_config_rule", {})
    if exp4444.get("ft09_resolved_generically") is True:
        _add_gap(
            entries,
            _gap_entry(
                _loo_gap_id("ft09", "missing_local_constraint_color_cycle_verifier"),
                status="filled (exp4444_generic_config_rule_verifier_operator)",
                evidence=(
                    f"{EXP4444_PATH}; ft09_resolved_generically=True; offline_reproduced="
                    f"{exp4444.get('offline_reproduced')}; reproduced_levels={exp4444.get('reproduced_levels')}; "
                    "operator=config_rule_verifier; target_recipe_withheld=ft09"
                ),
                failure_mode="prior missing_local_constraint_color_cycle_verifier residual is closed for ft09 L1",
                missing_discriminator=(
                    "filled by generic execution-grounded local_constraint_color_cycle verifier"
                ),
                candidate_design="reuse config_rule_verifier for future local-constraint/toggle digests",
                source_artifact=EXP4444_PATH,
                movement="filled",
            ),
        )
    if exp4444:
        _add_gap(
            entries,
            _gap_entry(
                "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                status="open",
                evidence=(
                    f"{EXP4444_PATH}; dc22_state={exp4444.get('dc22_state')}; "
                    f"honest_verdict={exp4444.get('honest_verdict')}; "
                    "offline_reproduced=True for ft09 gate only; dc22 reproduced_levels=0"
                ),
                failure_mode="missing_config_rule_verifier_grounding",
                missing_discriminator=(
                    "dc22 exposes movement/object interaction win logic rather than a grounded "
                    "local-constraint, marker-coverage, or target-offset toggle digest"
                ),
                candidate_design=(
                    "build a dc22 movement/object-interaction verifier or richer digest extractor; "
                    "do not count a level until reproduce(dc22) passes"
                ),
                source_artifact=EXP4444_PATH,
                movement="updated_still_open",
            ),
        )

    exp4445 = trusted.get("4445_object_motion", {})
    closed_by_operator = set(exp4445.get("residuals_closed_generically") or [])
    if exp4445.get("offline_reproduced") is True:
        if "ar25" in closed_by_operator:
            _add_gap(
                entries,
                _gap_entry(
                    _loo_gap_id("ar25", "missing_reflection_world_model_and_object_motion_plan"),
                    status="filled (exp4445_generic_object_motion_world_model_operator)",
                    evidence=(
                        f"{EXP4445_PATH}; residuals_closed_generically includes ar25; "
                        f"offline_reproduced=True; reproduced_levels={exp4445.get('reproduced_levels')}; "
                        "operator=object_motion_world_model; target_recipe_withheld=ar25; "
                        f"world_model_accuracy_with_examples={exp4445.get('world_model_accuracy_with_examples')}; "
                        f"world_model_accuracy_cold={exp4445.get('world_model_accuracy_cold')}"
                    ),
                    failure_mode="missing_reflection_world_model_and_object_motion_plan",
                    missing_discriminator="filled by generic object-slot translate/reflect transition model",
                    candidate_design="keep the generic operator in the standing loop",
                    source_artifact=EXP4445_PATH,
                    movement="filled",
                ),
            )
        if "ka59" in closed_by_operator:
            _add_gap(
                entries,
                _gap_entry(
                    _loo_gap_id("ka59", "missing_push_block_world_model_and_dynamic_selection"),
                    status="filled (exp4445_generic_object_motion_world_model_operator)",
                    evidence=(
                        f"{EXP4445_PATH}; residuals_closed_generically includes ka59; "
                        f"offline_reproduced=True; reproduced_levels={exp4445.get('reproduced_levels')}; "
                        "operator=object_motion_world_model; target_recipe_withheld=ka59; "
                        f"world_model_accuracy_with_examples={exp4445.get('world_model_accuracy_with_examples')}; "
                        f"world_model_accuracy_cold={exp4445.get('world_model_accuracy_cold')}"
                    ),
                    failure_mode="missing_push_block_world_model_and_dynamic_selection",
                    missing_discriminator=(
                        "filled by generic object-slot translate/push transition model with dynamic selection"
                    ),
                    candidate_design="keep the generic operator in the standing loop",
                    source_artifact=EXP4445_PATH,
                    movement="filled",
                ),
            )

    exp4446 = trusted.get("4446_first_contact", {})
    if exp4446.get("offline_reproduced") is True and _as_int(exp4446.get("reproduced_levels")) >= 1:
        target_game = str(exp4446.get("target_game") or "vc33")
        _add_gap(
            entries,
            _gap_entry(
                f"GAP-4423-{_slug(target_game)}-UNSELECTABLE-FIRST-CONTACT",
                status="filled (exp4446_drive_generic_first_contact_bank)",
                evidence=(
                    f"{EXP4446_PATH}; target_game={target_game}; routed_to={exp4446.get('routed_to')}; "
                    f"offline_reproduced=True; reproduced_levels={exp4446.get('reproduced_levels')}"
                ),
                failure_mode="closed_by_support_clearance_config_rule",
                missing_discriminator="filled by generic config_rule_verifier support-clearance digest",
                candidate_design="reuse routed config-rule support-clearance predicates",
                source_artifact=EXP4446_PATH,
                movement="filled",
            ),
        )

    exp4448 = trusted.get("4448_loo_v2", {})
    if exp4448:
        for row in exp4448.get("closed_residuals_by_new_operator", []):
            if not isinstance(row, Mapping):
                continue
            operator = str(row.get("closed_by_operator") or "")
            if operator == "object_motion_world_model" and "4445_object_motion" not in trusted:
                continue
            if operator == "config_rule_verifier" and "4444_config_rule" not in trusted:
                continue
            game = str(row.get("game") or "")
            residual = str(row.get("v1_residual_delta") or "")
            if game and residual:
                status = f"filled (exp4448_loo_v2_{operator})"
                _add_gap(
                    entries,
                    _gap_entry(
                        _loo_gap_id(game, residual),
                        status=status,
                        evidence=(
                            f"{EXP4448_PATH}; game={game}; closed_by_operator={operator}; "
                            f"generic_loo_solve_count_v2={exp4448.get('generic_loo_solve_count_v2')}"
                        ),
                        failure_mode=f"prior {residual} residual is closed in LOO v2",
                        missing_discriminator=f"filled by {operator}",
                        candidate_design="keep the .411 operator in the generic leave-one-out loop",
                        source_artifact=EXP4448_PATH,
                        movement="filled",
                    ),
                )
        for row in exp4448.get("missing_verifier_gaps", []):
            if not isinstance(row, Mapping):
                continue
            game = str(row.get("game") or "")
            residual = str(row.get("residual_delta") or "")
            if not game or not residual:
                continue
            _add_gap(
                entries,
                _gap_entry(
                    _loo_gap_id(game, residual),
                    status="open",
                    evidence=(
                        f"{EXP4448_PATH}; game={game}; residual_delta={residual}; "
                        f"retrieved_operator={row.get('retrieved_operator')}; "
                        f"generic_loo_solve_count_v2={exp4448.get('generic_loo_solve_count_v2')}"
                    ),
                    failure_mode=residual,
                    missing_discriminator=f"generic primitive/verifier still missing for {residual}",
                    candidate_design=(
                        "promote the residual into a reusable primitive or verifier, then rerun LOO v3"
                    ),
                    source_artifact=EXP4448_PATH,
                    movement="updated_still_open",
                ),
            )
    return list(entries.values())


def _gap_marker(gap_id: str) -> str:
    return GAP_MARKER_OVERRIDES.get(gap_id, f"exp4449-{_slug(gap_id).lower()}")


def _gap_block(gap: Mapping[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4449 .411 registry gap hygiene\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'high')}\n"
        f"- source artifact: {gap.get('source_artifact', '')}\n"
        f"- movement: {gap.get('movement', 'updated')}\n"
    )


def _replace_marked_block(text: str, marker: str, block: str) -> str:
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    replacement = f"{start}\n{block.rstrip()}\n{end}"
    if start in text and end in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        return f"{prefix}{replacement}{suffix}"
    return text.rstrip() + "\n\n" + replacement + "\n"


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
        updated = _replace_marked_block(updated, _gap_marker(gap_id), _gap_block(gap))
    return updated, filled, open_ids


def _find_game(registry: dict[str, Any], game: str) -> dict[str, Any] | None:
    games = registry.get("games")
    if not isinstance(games, list):
        return None
    for row in games:
        if isinstance(row, dict) and row.get("game") == game:
            return row
    return None


def _ensure_game(registry: dict[str, Any], game: str) -> dict[str, Any]:
    row = _find_game(registry, game)
    if row is not None:
        return row
    row = {"game": game, "reproducibility": "reproduced", "levels_reproduced": 0, "gotchas": []}
    registry.setdefault("games", []).append(row)
    return row


def _reproduced_counts(registry: Mapping[str, Any]) -> tuple[int, int]:
    levels = 0
    games_count = 0
    games = registry.get("games")
    if not isinstance(games, list):
        return 0, 0
    for row in games:
        if not isinstance(row, Mapping):
            continue
        level_count = _as_int(row.get("levels_reproduced"))
        if row.get("reproducibility") == "reproduced" and level_count > 0:
            games_count += 1
            levels += level_count
    return levels, games_count


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
    row = _ensure_game(registry, game)
    row["reproducibility"] = "reproduced"
    row["levels_reproduced"] = max(_as_int(row.get("levels_reproduced")), levels)
    row.setdefault("gotchas", [])
    row["solver"] = solver
    row[latest_key] = {
        "artifact": artifact_path,
        "offline_reproduced": True,
        "reproduced_levels": levels,
        "reproducibility_checksum": checksum,
    }


def _library_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("primitives_documented")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def reconcile_arc_registry(
    registry: Mapping[str, Any],
    trusted: Mapping[str, Mapping[str, Any]],
    *,
    filled_gap_ids: list[str],
    open_gap_ids: list[str],
    excluded: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """REQ-REPORT-4449: update reproduced ARC counts from trusted .411 rows."""

    updated = deepcopy(dict(registry))
    updated.setdefault("games", [])

    exp4443 = trusted.get("4443_g50t_bank", {})
    if exp4443.get("offline_reproduced") is True:
        _record_reproduced_game(
            updated,
            game=str(exp4443.get("target_game") or "g50t"),
            levels=max(1, _as_int(exp4443.get("reproduced_levels"))),
            artifact_path=EXP4443_PATH,
            latest_key="latest_exp4443_reproduce",
            checksum=str(exp4443.get("reproducibility_checksum") or ""),
            solver=(
                "python/carnot/experiment_4443_bank_g50t_example_conditioned_win.py "
                "execution-grounds the target-offset predicate and replays OfflineSolver."
            ),
        )

    exp4446 = trusted.get("4446_first_contact", {})
    if exp4446.get("offline_reproduced") is True:
        _record_reproduced_game(
            updated,
            game=str(exp4446.get("target_game") or "vc33"),
            levels=max(1, _as_int(exp4446.get("reproduced_levels"))),
            artifact_path=EXP4446_PATH,
            latest_key="latest_exp4446_reproduce",
            checksum=str(exp4446.get("reproducibility_checksum") or ""),
            solver=(
                "python/carnot/experiment_4446_drive_generic_first_contact_bank.py "
                "routes first contact through a generic config-rule verifier."
            ),
        )

    exp4444 = trusted.get("4444_config_rule", {})
    if exp4444:
        row = _ensure_game(updated, "ft09")
        row["generic_verifier_reproduce"] = (
            f"Exp4444 {EXP4444_PATH} ft09_resolved_generically="
            f"{exp4444.get('ft09_resolved_generically')} via config_rule_verifier."
        )
        updated["latest_config_rule_verifier_4444"] = {
            "artifact": EXP4444_PATH,
            "operator": "config_rule_verifier",
            "ft09_resolved_generically": exp4444.get("ft09_resolved_generically") is True,
            "dc22_state": exp4444.get("dc22_state"),
            "reproduced_levels": _as_int(exp4444.get("reproduced_levels")),
            "offline_reproduced": exp4444.get("offline_reproduced") is True,
            "no_regression": exp4444.get("no_regression") is True,
            "filled_gap_ids": [
                _loo_gap_id("ft09", "missing_local_constraint_color_cycle_verifier")
            ]
            if exp4444.get("ft09_resolved_generically") is True
            else [],
            "remaining_gap_ids": ["GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT"],
        }

    exp4445 = trusted.get("4445_object_motion", {})
    if exp4445:
        for game in exp4445.get("residuals_closed_generically", []) or []:
            row = _ensure_game(updated, str(game))
            row["generic_object_motion_reproduce"] = (
                f"Exp4445 object_motion_world_model re-solved {game} L1 without "
                f"{game}'s hand world-model recipe."
            )
        updated["latest_object_motion_world_model_4445"] = {
            "artifact": EXP4445_PATH,
            "operator": "object_motion_world_model",
            "residuals_closed_generically": list(exp4445.get("residuals_closed_generically") or []),
            "world_model_accuracy_with_examples": exp4445.get("world_model_accuracy_with_examples"),
            "world_model_accuracy_cold": exp4445.get("world_model_accuracy_cold"),
            "reproduced_levels": _as_int(exp4445.get("reproduced_levels")),
            "offline_reproduced": exp4445.get("offline_reproduced") is True,
            "no_regression": exp4445.get("no_regression") is True,
        }

    exp4447 = trusted.get("4447_primitive_library", {})
    if exp4447:
        updated["latest_documented_primitive_library_4447"] = {
            "artifact": EXP4447_PATH,
            "inference_substrate": exp4447.get("inference_substrate", INFERENCE_SUBSTRATE),
            "library_coverage": exp4447.get("library_coverage"),
            "retrieval_precision_at_1": exp4447.get("retrieval_precision_at_1"),
            "constant_leak_violations": list(exp4447.get("constant_leak_violations") or []),
            "documented_primitives": _library_rows(exp4447),
        }

    exp4448 = trusted.get("4448_loo_v2", {})
    if exp4448:
        updated["latest_loo_generic_v2_4448"] = {
            "artifact": EXP4448_PATH,
            "generic_loo_solve_count_v1_baseline": exp4448.get(
                "generic_loo_solve_count_v1_baseline"
            ),
            "generic_loo_solve_count_v2": exp4448.get("generic_loo_solve_count_v2"),
            "loo_gate_passed": exp4448.get("loo_gate_passed") is True,
            "closed_residuals_by_new_operator": list(
                exp4448.get("closed_residuals_by_new_operator") or []
            ),
            "missing_verifier_gaps": list(exp4448.get("missing_verifier_gaps") or []),
        }

    total_levels, total_games = _reproduced_counts(updated)
    updated["updated"] = "2026-06-19"
    updated["reproducible_total_levels"] = total_levels
    updated["reproducible_total_games"] = total_games
    updated["latest_hygiene_4449"] = {
        "artifact": RESULT_RELATIVE_PATH,
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "filled_gap_ids": filled_gap_ids,
        "open_gap_ids": open_gap_ids,
        "excluded_artifacts": excluded,
        "note": ".411 registry hygiene; flagged_adversarial artifacts excluded from counts.",
    }
    return updated, {
        "arc_solve_registry_reconciled": True,
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
    }


def _find_verifier(registry: dict[str, Any], verifier_id: str) -> dict[str, Any] | None:
    verifiers = registry.get("verifiers")
    if not isinstance(verifiers, list):
        return None
    for row in verifiers:
        if isinstance(row, dict) and row.get("verifier_id") == verifier_id:
            return row
    return None


def _guard_current(guard: Mapping[str, Any]) -> Mapping[str, Any]:
    current = guard.get("current")
    if isinstance(current, Mapping):
        return current
    replayed = guard.get("replayed_arc1_rule_exec")
    if isinstance(replayed, Mapping):
        return replayed
    return {}


def guard_passed(guard: Mapping[str, Any]) -> bool:
    current = _guard_current(guard)
    gated = _as_float(current.get("gated_pass2"))
    vote = _as_float(current.get("vote_pass2"))
    lost = _as_int(current.get("vote_wins_lost"))
    explicit = guard.get("regression_guard_passed") is True or guard.get("gap4_execution_guard_passed") is True
    beats_vote = guard.get("arc_oracle_distinct_verifier_beats_vote")
    return bool(explicit and gated >= vote and lost == 0 and beats_vote is not False)


def stamp_fix_durable(stamp: Mapping[str, Any]) -> bool:
    return bool(
        stamp.get("capstone_stamp_fix_durable") is True
        or stamp.get("capstone_stamp_fix_verified") is True
    ) and stamp.get("circular_moat_overclaim_fired") is not True


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
    verifier = _find_verifier(updated, GAP4_VERIFIER_ID)
    if verifier is None:
        verifier = {
            "verifier_id": GAP4_VERIFIER_ID,
            "domain": "arc_agi2_grid",
            "kind": "process_verifier",
            "eval": {},
            "registry_roles": [],
        }
        updated.setdefault("verifiers", []).append(verifier)

    current = _guard_current(guard)
    exp4447 = trusted.get("4447_primitive_library", {})
    exp4448 = trusted.get("4448_loo_v2", {})
    verifier.setdefault("eval", {}).update(
        {
            "eval_exp_4449": RESULT_RELATIVE_PATH,
            "exp4449_regression_guard_passed": guard_passed(guard),
            "exp4449_arc_oracle_distinct_verifier_beats_vote": (
                guard.get("arc_oracle_distinct_verifier_beats_vote") is not False
            ),
            "exp4449_arc1_rule_exec_vote_pass2": current.get("vote_pass2"),
            "exp4449_arc1_rule_exec_gated_pass2": current.get("gated_pass2"),
            "exp4449_arc1_headroom_recovered": current.get("headroom_recovered"),
            "exp4449_arc1_vote_wins_lost": current.get("vote_wins_lost"),
            "exp4449_capstone_stamp_fix_durable": stamp_fix_durable(stamp),
            "exp4449_reproducible_total_levels": total_levels,
            "exp4449_reproducible_total_games": total_games,
            "exp4449_filled_gap_ids": filled_gap_ids,
            "exp4449_open_gap_ids": open_gap_ids,
            "exp4449_flagged_artifacts_excluded": excluded,
            "exp4449_generic_loo_solve_count_v2": exp4448.get("generic_loo_solve_count_v2"),
            "exp4449_primitive_library_coverage": exp4447.get("library_coverage"),
            "exp4449_retrieval_precision_at_1": exp4447.get("retrieval_precision_at_1"),
        }
    )

    role = {
        "role_id": V411_ROLE_ID,
        "experiment": RESULT_RELATIVE_PATH,
        "role": "registry_gaps_arc_hygiene_v411",
        "status": "v411_outcomes_recorded_with_gap4_guard_and_stamp_durability",
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "filled_gap_ids": filled_gap_ids,
        "open_gap_ids": open_gap_ids,
        "excluded_artifacts": excluded,
        "eval_exp_4449": RESULT_RELATIVE_PATH,
    }
    roles = verifier.setdefault("registry_roles", [])
    if not isinstance(roles, list):
        roles = verifier["registry_roles"] = []
    verifier["registry_roles"] = [
        row for row in roles if not (isinstance(row, Mapping) and row.get("role_id") == V411_ROLE_ID)
    ] + [role]
    return updated, {"verifier_registry_reconciled": True}


def _check_helper_import() -> dict[str, Any]:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        module = importlib.import_module("scripts.capstone_aggregate_available")
    except Exception as exc:  # pragma: no cover - exception type depends on import machinery
        return {"ok": False, "module": "scripts.capstone_aggregate_available", "error": str(exc)}
    return {
        "ok": hasattr(module, "aggregate_available_report_gaps") and hasattr(module, "AxisSpec"),
        "module": "scripts.capstone_aggregate_available",
        "error": "",
    }


def check_preconditions(
    root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    verifier_registry, verifier_check = _yaml_mapping(root / REGISTRY_RELATIVE_PATH)
    arc_registry, arc_check = _yaml_mapping(root / ARC_REGISTRY_RELATIVE_PATH)
    gaps_text, gaps_check = _read_text(root / GAPS_RELATIVE_PATH)
    payloads, source_report = load_sources(root)
    helper_check = _check_helper_import()
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


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(payload), sort_keys=False)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _checksum_for_artifact(artifact: Mapping[str, Any]) -> str:
    return _sha256(
        {
            "registry_reconciliation": artifact.get("registry_reconciliation"),
            "availability_report": artifact.get("availability_report"),
            "regression_guard_passed": artifact.get("regression_guard_passed"),
            "capstone_stamp_fix_durable": artifact.get("capstone_stamp_fix_durable"),
            "reproducible_total_levels": artifact.get("reproducible_total_levels"),
            "reproducible_total_games": artifact.get("reproducible_total_games"),
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
        "experiment": "experiment_4449_registry_gaps_hygiene",
        "schema": "carnot.exp4449.registry_gaps_hygiene.v1",
        "honest_verdict": f"complete: registry_gaps_hygiene_4449_{suffix}",
        "regression_guard_passed": guard_ok,
        "reproducible_total_levels": int(total_levels),
        "reproducible_total_games": int(total_games),
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
            "inference_substrate",
        ):
            if artifact["field_principles"].get(field) != FIELD_PRINCIPLES[field]:
                errors.append(f"field_principles.{field}.principle must match REQ-REPORT-4449")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64 or not all(
        char in "0123456789abcdef" for char in checksum
    ):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    if set(SPEC_REFS) - set(artifact.get("spec_refs", [])):
        errors.append("spec_refs must include REQ-REPORT-4449 and SCENARIO-REPORT-4449")
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
    """REQ-REPORT-4449: reconcile ledgers, run guards, and write the artifact."""

    started = now()
    root = Path(root)
    verifier_registry, arc_registry, gaps_text, payloads, preconditions = check_preconditions(root)
    available = availability_report(payloads)
    trusted = trusted_payloads(payloads, available)
    excluded = excluded_artifact_paths(payloads)
    gap_entries = collect_gap_entries(trusted)
    updated_gaps, filled_gap_ids, open_gap_ids = reconcile_gaps_text(gaps_text, gap_entries)

    guard = dict(gap4_guard_runner(root))
    stamp = dict(capstone_stamp_runner(root))
    updated_arc, arc_report = reconcile_arc_registry(
        arc_registry,
        trusted,
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
        _write_yaml(root / ARC_REGISTRY_RELATIVE_PATH, updated_arc)
        _write_yaml(root / REGISTRY_RELATIVE_PATH, updated_verifier)
        _write_text(root / GAPS_RELATIVE_PATH, updated_gaps)

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
