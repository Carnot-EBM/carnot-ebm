"""Exp 4438: reconcile .410 registry and verifier-gap hygiene.

Spec refs: REQ-REPORT-4438, SCENARIO-REPORT-4438.

This module is intentionally an audit and bookkeeping pass. It reads outcome
artifacts, updates the ops ledgers that explain what is reproducible, and runs
the cached GAP-4/stamp guards. It does not touch production verifier code
because verifier behavior must only change in dedicated implementation tasks.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

import yaml

from carnot.reporting import capstone_aggregate_available


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4438_registry_gaps_hygiene.json"
REGISTRY_RELATIVE_PATH = "ops/verifier_registry.yaml"
GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
ARC_REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"

EXP4432_PATH = "results/experiment_4432_loo_generic_solve_benchmark.json"
EXP4433_PATH = "results/experiment_4433_example_conditioned_win_induction.json"
EXP4434_PATH = "results/experiment_4434_example_conditioned_action_model.json"
EXP4435_PATH = "results/experiment_4435_generic_first_contact_fixed.json"
EXP4436_PATH = "results/experiment_4436_deepen_plus_primitive_consolidation.json"

SOURCE_ARTIFACTS = {
    "4432_loo_generic": EXP4432_PATH,
    "4433_win_induction": EXP4433_PATH,
    "4434_action_model": EXP4434_PATH,
    "4435_first_contact": EXP4435_PATH,
    "4436_primitives": EXP4436_PATH,
}
SOURCE_EXPERIMENT_IDS = {
    "4432_loo_generic": 4432,
    "4433_win_induction": 4433,
    "4434_action_model": 4434,
    "4435_first_contact": 4435,
    "4436_primitives": 4436,
}

RANDOM_SEED = 4438
SPEC_REFS = ("REQ-REPORT-4438", "SCENARIO-REPORT-4438")
INFERENCE_SUBSTRATE = "cached_registry_reconciliation_gap4_guard_and_stamp_audit"
GAP4_VERIFIER_ID = "gap4_program_induction_stack"
V410_ROLE_ID = "oracle_distinct_v410_registry_gaps_hygiene_4438"

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "reproducible_total_levels",
    "reproducible_total_games",
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
        "principle": "BARE bool (gated-fields-must-be-bare): the GAP-4 result did not regress"
    },
    "reproducible_total_levels": {"principle": "the reconciled authoritative count"},
    "reproducible_total_games": {
        "principle": "the reconciled authoritative reproduced-game count"
    },
    "availability_report": {
        "principle": "robust aggregate-available report; missing or flagged inputs do not erase other axes"
    },
    "submitted_to_leaderboard": {"principle": "must remain false for this audit-only task"},
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
    """REQ-REPORT-4438: read every upstream artifact without trusting flagged rows."""

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
            name="loo_generic",
            required_keys=("4432_loo_generic",),
            verdict_fn=lambda present: int(
                present["4432_loo_generic"].get("generic_loo_solve_count") or 0
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="win_induction",
            required_keys=("4433_win_induction",),
            verdict_fn=lambda present: (
                present["4433_win_induction"].get("offline_reproduced") is True
                and _as_int(present["4433_win_induction"].get("reproduced_levels")) >= 1
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="action_model",
            required_keys=("4434_action_model",),
            verdict_fn=lambda present: (
                present["4434_action_model"].get("offline_reproduced") is True
                or _as_float(
                    present["4434_action_model"].get("world_model_accuracy_with_examples")
                )
                > _as_float(present["4434_action_model"].get("world_model_accuracy_cold"))
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="first_contact",
            required_keys=("4435_first_contact",),
            verdict_fn=lambda present: bool(
                present["4435_first_contact"].get("verdict_contract_fixed")
                or str(present["4435_first_contact"].get("honest_verdict", "")).startswith(
                    TERMINAL_PREFIXES
                )
            ),
        ),
        capstone_aggregate_available.AxisSpec(
            name="primitives",
            required_keys=("4436_primitives",),
            verdict_fn=lambda present: (
                present["4436_primitives"].get("no_regression") is True
                and isinstance(present["4436_primitives"].get("primitives_consolidated"), list)
                and bool(present["4436_primitives"].get("primitives_consolidated"))
            ),
        ),
    ]


def availability_report(payloads: Mapping[str, Any]) -> dict[str, Any]:
    """REQ-REPORT-4438: use the robust helper so one bad input does not poison all axes."""

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
    priority: str = "high",
    source_artifact: str,
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
    }


def _collect_loo_gaps(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    raw_gaps = payload.get("missing_verifier_gaps")
    rows = raw_gaps if isinstance(raw_gaps, list) else payload.get("per_game", [])
    if not isinstance(rows, list):
        return gaps
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        residual = str(row.get("residual_delta") or "")
        game = str(row.get("game") or "unknown")
        if not residual or residual == "none":
            continue
        gap_id = f"GAP-4432-LOO-{_slug(game)}-{_slug(residual)}"
        gaps.append(
            _gap_entry(
                gap_id,
                status="open",
                evidence=(
                    f"{EXP4432_PATH}; game={game}; routed_to={row.get('routed_to')}; "
                    f"residual_delta={residual}; "
                    f"generic_loo_solve_count={payload.get('generic_loo_solve_count')}"
                ),
                failure_mode=residual,
                missing_discriminator=f"generic primitive/verifier for {residual}",
                candidate_design=(
                    "promote the residual into a reusable primitive or verifier, then rerun "
                    "the leave-one-out fold without the target game's own recipe"
                ),
                source_artifact=EXP4432_PATH,
            )
        )
    return gaps


def _collect_first_contact_gaps(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_gaps = payload.get("missing_verifier_gaps")
    if not isinstance(raw_gaps, list):
        return []
    gaps: list[dict[str, Any]] = []
    for raw in raw_gaps:
        if not isinstance(raw, Mapping):
            continue
        game = str(raw.get("game") or payload.get("target_game") or "unknown")
        gap_id = str(raw.get("gap_id") or f"GAP-4435-FIRST-CONTACT-{_slug(game)}")
        gaps.append(
            _gap_entry(
                gap_id,
                status=str(raw.get("status") or "open"),
                evidence=(
                    f"{EXP4435_PATH}; target_game={game}; "
                    f"honest_verdict={payload.get('honest_verdict')}; "
                    f"offline_reproduced={payload.get('offline_reproduced')}; "
                    f"reproduced_levels={payload.get('reproduced_levels')}"
                ),
                failure_mode=str(raw.get("failure_mode") or "routed first-contact no-new-level"),
                missing_discriminator=str(
                    raw.get("missing_discriminator")
                    or "selectable verifier for the first-contact winning delta"
                ),
                candidate_design=str(
                    raw.get("candidate_design")
                    or "convert the routed recipe into a reproduction-gated verifier"
                ),
                source_artifact=EXP4435_PATH,
            )
        )
    return gaps


def _collect_action_model_gaps(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_gaps = payload.get("missing_verifier_gaps")
    if not isinstance(raw_gaps, list):
        return []
    gaps: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_gaps):
        if isinstance(raw, Mapping):
            residual = str(raw.get("residual_delta") or raw.get("failure_mode") or "action_model_gap")
            game = str(raw.get("game") or payload.get("target_game") or "unknown")
        else:
            residual = str(raw)
            game = str(payload.get("target_game") or "unknown")
        if not residual:
            continue
        gaps.append(
            _gap_entry(
                f"GAP-4434-ACTION-MODEL-{_slug(game)}-{_slug(residual or index)}",
                status="open",
                evidence=f"{EXP4434_PATH}; game={game}; residual={residual}",
                failure_mode=residual,
                missing_discriminator=f"{game} action-model verifier for {residual}",
                candidate_design="turn the measured action-model residual into an executable gate",
                priority="medium",
                source_artifact=EXP4434_PATH,
            )
        )
    return gaps


def collect_gap_entries(trusted: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """SCENARIO-REPORT-4438: convert non-adversarial residuals into ledger entries."""

    entries: dict[str, dict[str, Any]] = {}
    if "4432_loo_generic" in trusted:
        for gap in _collect_loo_gaps(trusted["4432_loo_generic"]):
            entries[gap["gap_id"]] = gap
    if "4434_action_model" in trusted:
        for gap in _collect_action_model_gaps(trusted["4434_action_model"]):
            entries[gap["gap_id"]] = gap
    if "4435_first_contact" in trusted:
        for gap in _collect_first_contact_gaps(trusted["4435_first_contact"]):
            entries[gap["gap_id"]] = gap
    return list(entries.values())


def _gap_marker(gap_id: str) -> str:
    return f"exp4438-{_slug(gap_id).lower()}"


def _gap_block(gap: Mapping[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4438 .410 registry gap hygiene\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'high')}\n"
        f"- source artifact: {gap.get('source_artifact', '')}\n"
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


def reconcile_gaps_text(gaps_text: str, gap_entries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    updated = gaps_text
    ids: list[str] = []
    for gap in gap_entries:
        ids.append(str(gap["gap_id"]))
        updated = _replace_marked_block(updated, _gap_marker(str(gap["gap_id"])), _gap_block(gap))
    return updated, ids


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


def _ensure_general_gotcha(registry: dict[str, Any], primitive: Mapping[str, Any]) -> None:
    operator = str(primitive.get("operator") or "")
    if not operator:
        return
    gotchas = registry.setdefault("general_gotchas", [])
    if not isinstance(gotchas, list):
        registry["general_gotchas"] = gotchas = []
    gotcha_id = f"primitive_{operator}"
    row = next((item for item in gotchas if isinstance(item, dict) and item.get("id") == gotcha_id), None)
    payload = {
        "id": gotcha_id,
        "operator": operator,
        "derived_from_games": [str(game) for game in primitive.get("derived_from_games", [])],
        "note": f"Exp 4438 records consolidated generic primitive {operator} from Exp 4436.",
    }
    if row is None:
        gotchas.append(payload)
    else:
        row.update(payload)


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


def reconcile_arc_registry(
    registry: Mapping[str, Any],
    trusted: Mapping[str, Mapping[str, Any]],
    *,
    gap_ids: list[str],
    excluded: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """REQ-REPORT-4438: update reproduced ARC counts from trusted, reproduction-gated rows."""

    updated = deepcopy(dict(registry))
    updated.setdefault("games", [])

    exp4436 = trusted.get("4436_primitives", {})
    if (
        exp4436.get("offline_reproduced") is True
        and exp4436.get("no_regression") is True
        and _as_int(exp4436.get("reproduced_levels")) > 0
    ):
        game = str(exp4436.get("deepened_game") or "tu93")
        row = _ensure_game(updated, game)
        row["reproducibility"] = "reproduced"
        row["levels_reproduced"] = max(
            _as_int(row.get("levels_reproduced")),
            _as_int(exp4436.get("reproduced_levels")),
        )
        row["latest_exp4436_reproduce"] = {
            "artifact": EXP4436_PATH,
            "offline_reproduced": True,
            "new_levels_reproduced": _as_int(exp4436.get("new_levels_reproduced")),
            "reproducibility_checksum": str(exp4436.get("reproducibility_checksum") or ""),
        }
        for primitive in exp4436.get("primitives_consolidated", []):
            if isinstance(primitive, Mapping):
                _ensure_general_gotcha(updated, primitive)

    for key, artifact_path in (("4433_win_induction", EXP4433_PATH), ("4435_first_contact", EXP4435_PATH)):
        payload = trusted.get(key)
        if not payload or payload.get("offline_reproduced") is not True:
            continue
        levels = _as_int(payload.get("reproduced_levels"))
        if levels <= 0:
            continue
        game = str(payload.get("target_game") or "")
        if not game:
            continue
        row = _ensure_game(updated, game)
        row["reproducibility"] = "reproduced"
        row["levels_reproduced"] = max(_as_int(row.get("levels_reproduced")), levels)
        row[f"latest_exp{SOURCE_EXPERIMENT_IDS[key]}_reproduce"] = {
            "artifact": artifact_path,
            "offline_reproduced": True,
            "reproducibility_checksum": str(payload.get("reproducibility_checksum") or ""),
        }

    total_levels, total_games = _reproduced_counts(updated)
    updated["updated"] = "2026-06-19"
    updated["reproducible_total_levels"] = total_levels
    updated["reproducible_total_games"] = total_games
    updated["latest_hygiene_4438"] = {
        "artifact": RESULT_RELATIVE_PATH,
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "gap_ids_logged": gap_ids,
        "excluded_artifacts": excluded,
        "note": ".410 registry hygiene; flagged_adversarial artifacts excluded from counts.",
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
    gap_ids: list[str],
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
    exp4434 = trusted.get("4434_action_model", {})
    exp4436 = trusted.get("4436_primitives", {})
    verifier.setdefault("eval", {}).update(
        {
            "eval_exp_4438": RESULT_RELATIVE_PATH,
            "exp4438_regression_guard_passed": guard_passed(guard),
            "exp4438_arc_oracle_distinct_verifier_beats_vote": (
                guard.get("arc_oracle_distinct_verifier_beats_vote") is not False
            ),
            "exp4438_arc1_rule_exec_vote_pass2": current.get("vote_pass2"),
            "exp4438_arc1_rule_exec_gated_pass2": current.get("gated_pass2"),
            "exp4438_arc1_headroom_recovered": current.get("headroom_recovered"),
            "exp4438_arc1_vote_wins_lost": current.get("vote_wins_lost"),
            "exp4438_capstone_stamp_fix_durable": stamp_fix_durable(stamp),
            "exp4438_reproducible_total_levels": total_levels,
            "exp4438_reproducible_total_games": total_games,
            "exp4438_gap_ids_logged": gap_ids,
            "exp4438_flagged_artifacts_excluded": excluded,
            "exp4438_loo_generic_solve_count": _as_int(
                trusted.get("4432_loo_generic", {}).get("generic_loo_solve_count")
            ),
            "exp4438_action_model_accuracy_delta": round(
                _as_float(exp4434.get("world_model_accuracy_with_examples"))
                - _as_float(exp4434.get("world_model_accuracy_cold")),
                6,
            )
            if exp4434
            else None,
            "exp4438_primitives_consolidated": [
                row.get("operator")
                for row in exp4436.get("primitives_consolidated", [])
                if isinstance(row, Mapping)
            ],
        }
    )

    role = {
        "role_id": V410_ROLE_ID,
        "experiment": RESULT_RELATIVE_PATH,
        "role": "registry_gaps_arc_hygiene_v410",
        "status": "v410_outcomes_recorded_with_gap4_guard_and_stamp_durability",
        "reproducible_total_levels": total_levels,
        "reproducible_total_games": total_games,
        "gap_ids_logged": gap_ids,
        "excluded_artifacts": excluded,
        "eval_exp_4438": RESULT_RELATIVE_PATH,
    }
    roles = verifier.setdefault("registry_roles", [])
    if not isinstance(roles, list):
        roles = verifier["registry_roles"] = []
    verifier["registry_roles"] = [
        row for row in roles if not (isinstance(row, Mapping) and row.get("role_id") == V410_ROLE_ID)
    ] + [role]
    return updated, {"verifier_registry_reconciled": True}


def check_preconditions(root: Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    verifier_registry, verifier_check = _yaml_mapping(root / REGISTRY_RELATIVE_PATH)
    arc_registry, arc_check = _yaml_mapping(root / ARC_REGISTRY_RELATIVE_PATH)
    gaps_text, gaps_check = _read_text(root / GAPS_RELATIVE_PATH)
    payloads, source_report = load_sources(root)
    checks = {
        "ok": verifier_check["readable"] and arc_check["readable"] and gaps_check["readable"],
        "files": {
            "verifier_registry": verifier_check,
            "arc_solve_registry": arc_check,
            "verifier_gaps": gaps_check,
        },
        "source_artifacts": source_report,
    }
    return verifier_registry, arc_registry, gaps_text, payloads, checks


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    reconciled = bool(registry_reconciliation.get("registries_reconciled"))
    suffix = "guard_passed" if guard_ok else "guard_failed"
    artifact: dict[str, Any] = {
        "experiment": "experiment_4438_registry_gaps_hygiene",
        "schema": "carnot.exp4438.registry_gaps_hygiene.v1",
        "honest_verdict": f"complete: registry_gaps_hygiene_4438_{suffix}",
        "regression_guard_passed": guard_ok,
        "reproducible_total_levels": int(total_levels),
        "reproducible_total_games": int(total_games),
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
        "inference_substrate": INFERENCE_SUBSTRATE,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": max(0.0, round(float(ended_at - started_at), 6)),
        "model_specs": {
            "method": INFERENCE_SUBSTRATE,
            "codex_calls": 0,
            "live_model_inference": False,
            "gpu_inference": False,
            "upstream_artifacts": list(SOURCE_ARTIFACTS.values()),
        },
        "registries_reconciled": reconciled,
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
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    if not isinstance(artifact.get("registry_reconciliation"), Mapping):
        errors.append("registry_reconciliation must be dict")
    if not isinstance(artifact.get("availability_report"), Mapping):
        errors.append("availability_report must be dict")
    if not isinstance(artifact.get("field_principles"), Mapping):
        errors.append("field_principles must be dict")
    else:
        for field in ("honest_verdict", "regression_guard_passed", "reproducible_total_levels"):
            if artifact["field_principles"].get(field) != FIELD_PRINCIPLES[field]:
                errors.append(f"field_principles.{field}.principle must match REQ-REPORT-4438")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64 or not all(
        char in "0123456789abcdef" for char in checksum
    ):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    if set(SPEC_REFS) - set(artifact.get("spec_refs", [])):
        errors.append("spec_refs must include REQ-REPORT-4438 and SCENARIO-REPORT-4438")
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
    """REQ-REPORT-4438: reconcile ledgers, run guards, and write the terminal artifact."""

    started = now()
    root = Path(root)
    verifier_registry, arc_registry, gaps_text, payloads, preconditions = check_preconditions(root)
    available = availability_report(payloads)
    trusted = trusted_payloads(payloads, available)
    excluded = excluded_artifact_paths(payloads)
    gaps = collect_gap_entries(trusted)
    updated_gaps, gap_ids = reconcile_gaps_text(gaps_text, gaps)

    guard = dict(gap4_guard_runner(root))
    stamp = dict(capstone_stamp_runner(root))
    updated_arc, arc_report = reconcile_arc_registry(
        arc_registry,
        trusted,
        gap_ids=gap_ids,
        excluded=excluded,
    )
    updated_verifier, verifier_report = reconcile_verifier_registry(
        verifier_registry,
        guard=guard,
        stamp=stamp,
        total_levels=arc_report["reproducible_total_levels"],
        total_games=arc_report["reproducible_total_games"],
        gap_ids=gap_ids,
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
        "gap_ids_logged": gap_ids,
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
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    print(f"regression_guard_passed={artifact['regression_guard_passed']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
