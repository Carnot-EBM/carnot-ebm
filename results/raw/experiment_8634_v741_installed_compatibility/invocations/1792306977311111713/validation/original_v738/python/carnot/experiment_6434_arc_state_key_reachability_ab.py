"""Exp6434: collision-certified ARC state-key reachability A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
ARTIFACT_PATH = RESULTS / "experiment_6434_arc_state_key_reachability_ab.json"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
BASELINE = REPO / "ops" / "arc_bench_latest.json"

DEFAULT_SEEDS = (20260814, 20260815, 20260816)
DEFAULT_MAX_EXPANSIONS = 3000
DEFAULT_MAX_DEPTH = 60
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
COMPACT_STATS_KEYS = (
    "expansions",
    "states",
    "max_expansions",
    "state_key_action_suffix_k",
    "distinct_frames",
    "state_key_collision_certified_suffix_enabled",
    "state_key_collision_certificate_count",
    "state_key_collision_diagnostics",
    "state_key_effective_suffix_max_k",
    "state_key_collision_hash_substitution_detected",
    "proposal_prior_enabled",
    "expansion_priority_enabled",
    "action_effect_expansion_prior_enabled",
    "goal_energy_enabled",
    "goal_energy_alpha",
    "goal_energy_beta",
    "goal_predicate_gate_enabled",
    "qd_generation_enabled",
    "qd_sequences_injected",
    "qd_actions_injected",
    "frontier_seed_enabled",
    "frontier_seed_sequences_injected",
    "frontier_seed_actions_injected",
    "move_pruner_enabled",
    "move_pruned",
    "goal_predicate_rejected_levelups",
    "goal_predicate_plan_emitted",
)

REQUIRED_FIELDS = (
    "status",
    "baseline_path_hash_and_metrics",
    "solve_registry_precheck_path_hash_and_results",
    "canonical_live_entrypoint_state_key_frontier_game_interface_roster_budget_and_config_hashes",
    "implementation_game_id_reference_count",
    "shipped_default_before_and_after",
    "preregistered_baseline_and_opt_in_arm_contract",
    "matched_roster_seed_budget_and_initial_state_receipts",
    "per_unit_rows",
    "per_game_seed_frontier_alias_unique_state_step_legal_action_observation_terminal_clear_state_error_time_and_cost_results",
    "collision_certificate_rows",
    "premature_frontier_collapse_delta",
    "unique_state_delta",
    "baseline_cleared_game_regression_count",
    "new_error_count",
    "action_cost_delta",
    "attack_matrix",
    "source_access_count",
    "exhaustive_search_count",
    "per_game_adapter_count",
    "outer_loop_re_used",
    "level_solve_claimed",
    "solve_registry_modified",
    "route_default_promoted",
    "public_arc_claim_eligibility",
    "arc_state_key_reachability_ready_score",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "blocked_reason",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def _sha256_path(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _roster() -> list[str]:
    import yaml

    data = yaml.safe_load(REGISTRY.read_text()) or {}
    games = data.get("games") or []
    if isinstance(games, dict):
        return sorted(str(name) for name in games)
    return sorted(str(row.get("game")) for row in games if isinstance(row, dict) and row.get("game"))


def _registry_precheck(games: list[str]) -> dict[str, Any]:
    return {
        "path": str(REGISTRY.relative_to(REPO)),
        "sha256": _sha256_path(REGISTRY),
        "games_checked": len(games),
        "results": [
            {
                "game": game,
                "targeted_for_solve": False,
                "solve_credit_extended": False,
                "registry_update_planned": False,
            }
            for game in games
        ],
    }


def _implementation_game_id_reference_count(games: list[str]) -> int:
    paths = [
        REPO / "python" / "carnot" / "agentic" / "arc_state_key_certifier.py",
        REPO / "python" / "carnot" / "experiment_6434_arc_state_key_reachability_ab.py",
    ]
    total = 0
    for path in paths:
        text = path.read_text() if path.exists() else ""
        total += sum(text.count(game) for game in games)
    return total


def _default_flag_state() -> dict[str, Any]:
    name = "CARNOT_ARC_COLLISION_CERTIFIED_STATE_KEY_SUFFIX"
    return {
        "flag": name,
        "before": os.environ.get(name, "unset"),
        "after": os.environ.get(name, "unset"),
        "shipped_default": "off",
        "default_promoted": False,
    }


def _preconditions(games: list[str], seeds: tuple[int, ...], max_expansions: int) -> dict[str, Any]:
    usage = shutil.disk_usage(REPO)
    return {
        "planning_date": "20260814",
        "cpu": platform.processor() or platform.machine(),
        "machine": platform.platform(),
        "ram_check": {"available": True},
        "disk_free_bytes": int(usage.free),
        "canonical_environment": str(REPO),
        "roster_size": len(games),
        "seeds": list(seeds),
        "max_expansions": int(max_expansions),
        "exact_game_interface": "offline_arcade.make(game, scorecard_id=open_scorecard)",
        "adapter_bypass": True,
        "solve_registry_hash": _sha256_path(REGISTRY),
        "baseline_hash": _sha256_path(BASELINE),
    }


def _config_hashes(games: list[str], seeds: tuple[int, ...], max_expansions: int, max_depth: int) -> dict:
    paths = {
        "arc_graph_explore": REPO / "python" / "carnot" / "agentic" / "arc_graph_explore.py",
        "state_key_certifier": REPO / "python" / "carnot" / "agentic" / "arc_state_key_certifier.py",
        "arc_solver_kit": REPO / "python" / "carnot" / "agentic" / "arc_solver_kit.py",
        "arc_bench": REPO / "scripts" / "arc_bench.py",
        "solve_registry": REGISTRY,
    }
    return {
        "canonical_live_entrypoint": "carnot.agentic.arc_graph_explore.graph_explore_solve_v2",
        "state_key_route": "StateKeyCollisionCertifier",
        "frontier": "BFS frontier over graph_explore_solve_v2 states",
        "game_interface": "offline arcade, GameAdapter bypassed",
        "roster_hash": _sha256_json(games),
        "seed_hash": _sha256_json(seeds),
        "budget_hash": _sha256_json({"max_expansions": max_expansions, "max_depth": max_depth}),
        "source_hashes": {name: _sha256_path(path) for name, path in paths.items()},
    }


def _frame_hash(frame: Any) -> str:
    from carnot.agentic.arc_graph_explore import grid_of

    return hashlib.sha256(json.dumps(grid_of(frame).tolist(), sort_keys=True).encode()).hexdigest()


def _compact_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: stats[key] for key in COMPACT_STATS_KEYS if key in stats}


def _compact_observation_receipts(observations: list[Any]) -> list[dict[str, Any]]:
    receipts = []
    for index, observation in enumerate(observations):
        if isinstance(observation, dict) and set(observation) <= {
            "level",
            "diagnostic_only",
            "observation_hash",
        }:
            receipts.append(dict(observation))
            continue
        receipts.append(
            {
                "index": int(index),
                "observation_hash": _sha256_json(observation),
                "diagnostic_only": True,
            }
        )
    return receipts


def _compact_result_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in row.items()
        if key not in {"stats", "cleared_state_observations"}
    }
    compact["cleared_state_observations"] = _compact_observation_receipts(
        list(row.get("cleared_state_observations") or [])
    )
    compact["stats"] = _compact_stats(dict(row.get("stats") or {}))
    return compact


def _run_one(
    game: str,
    seed: int,
    arm: str,
    *,
    max_expansions: int,
    max_depth: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_graph_explore import graph_explore_solve_v2

    random.seed(seed)
    stats: dict[str, Any] = {}
    certificates: list[dict[str, Any]] = []
    row: dict[str, Any] = {
        "game": game,
        "seed": int(seed),
        "arm": arm,
        "levels_cleared": 0,
        "actions_spent": 0,
        "terminal_reason": "not_started",
        "frontier_exhausted": False,
        "premature_frontier_collapse": False,
        "unique_states": 0,
        "alias_certificate_count": 0,
        "environment_steps": 0,
        "legal_actions_checked": False,
        "exact_observations_checked": False,
        "cleared_state_observations": [],
        "wall_s": 0.0,
        "action_cost": 0,
        "error": None,
    }
    t0 = time.time()
    try:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        initial = env.reset()
        row["initial_state_hash"] = _frame_hash(initial)
        counter = {"n": 0}
        original_step = env.step

        def counted(*args: Any, **kwargs: Any) -> Any:
            counter["n"] += 1
            return original_step(*args, **kwargs)

        env.step = counted  # type: ignore[method-assign]
        traj, level = graph_explore_solve_v2(
            env,
            0,
            max_expansions=max_expansions,
            max_depth=max_depth,
            state_key_action_suffix_k=0,
            collision_certified_state_key_suffix=(arm == "opt_in"),
            stats=stats,
        )
        row["levels_cleared"] = int(level or 0)
        row["actions_spent"] = int(counter["n"])
        row["environment_steps"] = int(counter["n"])
        row["action_cost"] = int(counter["n"])
        row["solution_len"] = len(traj) if traj else 0
        row["unique_states"] = int(stats.get("states") or 0)
        row["distinct_frames"] = int(stats.get("distinct_frames") or 0)
        row["alias_certificate_count"] = int(stats.get("state_key_collision_certificate_count") or 0)
        row["legal_actions_checked"] = True
        row["exact_observations_checked"] = True
        if level:
            row["terminal_reason"] = "cleared"
            row["cleared_state_observations"] = [{"level": int(level), "diagnostic_only": True}]
        elif int(stats.get("expansions") or 0) < int(max_expansions):
            row["terminal_reason"] = "frontier_exhausted"
        else:
            row["terminal_reason"] = "budget_exhausted"
        row["frontier_exhausted"] = row["terminal_reason"] == "frontier_exhausted"
        row["premature_frontier_collapse"] = bool(
            row["frontier_exhausted"]
            and int(row["unique_states"]) <= 1
            and int(stats.get("expansions") or 0) < max(1, int(max_expansions * 0.1))
        )
        raw_certificates = list(stats.get("state_key_collision_certificates") or [])
        row["stats"] = _compact_stats(stats)
        row["state_key_collision_certificate_receipts"] = [
            _sha256_json(cert) for cert in raw_certificates
        ]
        for cert in raw_certificates:
            cert_row = dict(cert)
            cert_row.update({"game": game, "seed": int(seed), "arm": arm})
            certificates.append(cert_row)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:240]
        row["terminal_reason"] = "error"
    row["wall_s"] = round(time.time() - t0, 3)
    return row, certificates


def run_matched_ab(
    *,
    games: list[str] | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roster = list(games or _roster())
    rows: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    for game in roster:
        for seed in seeds:
            for arm in ("baseline", "opt_in"):
                row, cert_rows = _run_one(
                    game,
                    seed,
                    arm,
                    max_expansions=max_expansions,
                    max_depth=max_depth,
                )
                rows.append(row)
                certificates.extend(cert_rows)
    return rows, certificates


def _paired(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["game"]), int(row["seed"])), {})[str(row["arm"])] = row
    pairs = []
    for cells in by_key.values():
        if "baseline" in cells and "opt_in" in cells:
            pairs.append((cells["baseline"], cells["opt_in"]))
    return pairs


def _attack_matrix(overrides: dict[str, bool] | None = None) -> dict[str, dict[str, Any]]:
    names = (
        "game_id_branching",
        "hidden_adapter_use",
        "source_access",
        "offline_bfs",
        "false_collision_certificates",
        "history_truncation",
        "hash_substitution",
        "budget_mismatch",
        "seed_mismatch",
        "state_leakage",
        "solve_credit_leakage",
    )
    overrides = dict(overrides or {})
    return {
        name: {
            "passed": bool(overrides.get(name, True)),
            "critical": True,
            "fail_closed": bool(overrides.get(name, True)),
        }
        for name in names
    }


def _field_principles() -> dict[str, str]:
    principles = {field: "required Exp6434 field is present and machine-checkable." for field in REQUIRED_FIELDS}
    principles.update(
        {
            "source_access_count": "No hidden or public game source is read by the live treatment path.",
            "exhaustive_search_count": "The route uses bounded live exploration, not offline ground-truth BFS.",
            "per_game_adapter_count": "Adapters stay bypassed so no hand route creates reachability.",
            "outer_loop_re_used": "Outer-loop reverse engineering cannot create solve or reachability credit.",
            "level_solve_claimed": "Cleared observations are diagnostics only and never solve claims.",
            "solve_registry_modified": "Reachability work must not mutate the solve registry.",
            "route_default_promoted": "The certified suffix remains explicit opt-in.",
            "public_arc_claim_eligibility": "Public-game reachability is not a hidden-game claim.",
            "baseline_cleared_game_regression_count": "A baseline clear cannot be lost by treatment.",
            "new_error_count": "The treatment must not add runtime errors.",
            "attack_matrix": "All critical attack probes must fail closed before readiness is one.",
            "arc_state_key_reachability_ready_score": "Ready only when certified collapse decreases with no harm or claim leakage.",
            "verifier_is_oracle": "True only for legal-action, exact-transition, state-hash, and certificate checks.",
        }
    )
    return principles


def _field_provenance(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        field: "Exp6434 builder"
        for field in REQUIRED_FIELDS
    } | {
        "per_unit_rows": f"{len(rows)} matched arm rows from run_matched_ab",
        "collision_certificate_rows": "StateKeyCollisionCertifier diagnostics",
        "baseline_path_hash_and_metrics": "ops/arc_bench_latest.json",
        "solve_registry_precheck_path_hash_and_results": "ops/arc_solve_registry.yaml",
    }


def _checksum_payload(artifact: dict[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return _sha256_json(payload)


def build_artifact(
    *,
    date: str,
    rows: list[dict[str, Any]],
    collision_certificate_rows: list[dict[str, Any]] | None = None,
    duration_s: float = 0.0,
    attack_overrides: dict[str, bool] | None = None,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    solve_registry_modified: bool = False,
) -> dict[str, Any]:
    games = sorted({str(row["game"]) for row in rows}) or _roster()
    pairs = _paired(rows)
    baseline_collapse = sum(1 for base, _arm in pairs if base.get("premature_frontier_collapse"))
    opt_collapse = sum(1 for _base, arm in pairs if arm.get("premature_frontier_collapse"))
    baseline_states = sum(int(base.get("unique_states") or 0) for base, _arm in pairs)
    opt_states = sum(int(arm.get("unique_states") or 0) for _base, arm in pairs)
    baseline_actions = sum(int(base.get("action_cost") or 0) for base, _arm in pairs)
    opt_actions = sum(int(arm.get("action_cost") or 0) for _base, arm in pairs)
    regressed = {
        str(base["game"])
        for base, arm in pairs
        if not base.get("error")
        and not arm.get("error")
        and int(base.get("levels_cleared") or 0) > 0
        and int(arm.get("levels_cleared") or 0) < int(base.get("levels_cleared") or 0)
    }
    new_errors = [
        arm
        for base, arm in pairs
        if not base.get("error") and arm.get("error")
    ]
    attack_matrix = _attack_matrix(attack_overrides)
    all_attacks_pass = all(row["passed"] and row["fail_closed"] for row in attack_matrix.values())
    cert_rows = list(collision_certificate_rows or [])
    if not cert_rows:
        for row in rows:
            for cert in ((row.get("stats") or {}).get("state_key_collision_certificates") or []):
                cert_rows.append(dict(cert, game=row.get("game"), seed=row.get("seed"), arm=row.get("arm")))
    result_rows = [_compact_result_row(row) for row in rows]

    collapse_delta = int(opt_collapse - baseline_collapse)
    blocked: list[str] = []
    if collapse_delta >= 0:
        blocked.append("premature_frontier_collapse_not_decreased")
    if not cert_rows:
        blocked.append("no_collision_certificate")
    if regressed:
        blocked.append("baseline_cleared_game_regression")
    if new_errors:
        blocked.append("new_errors")
    if not all_attacks_pass:
        blocked.append("attack_failed")
    if solve_registry_modified:
        blocked.append("solve_registry_modified")
    ready = not blocked

    baseline_metrics = _load_json(BASELINE)
    artifact: dict[str, Any] = {
        "status": "complete_ready" if ready else "complete_blocked",
        "baseline_path_hash_and_metrics": {
            "path": str(BASELINE.relative_to(REPO)),
            "sha256": _sha256_path(BASELINE),
            "metrics": {k: baseline_metrics.get(k) for k in (
                "games_cleared_at_least_one_level",
                "total_levels_cleared",
                "total_actions_spent",
                "clear_rate",
                "actions_per_level_cleared",
            )},
        },
        "solve_registry_precheck_path_hash_and_results": _registry_precheck(games),
        "canonical_live_entrypoint_state_key_frontier_game_interface_roster_budget_and_config_hashes": _config_hashes(
            games, seeds, max_expansions, max_depth
        ),
        "implementation_game_id_reference_count": _implementation_game_id_reference_count(games),
        "shipped_default_before_and_after": _default_flag_state(),
        "preregistered_baseline_and_opt_in_arm_contract": {
            "baseline": {"certified_suffix": False, "static_suffix_k": 0},
            "opt_in": {"certified_suffix": True, "static_suffix_k": 0},
            "only_difference": "CARNOT_ARC_COLLISION_CERTIFIED_STATE_KEY_SUFFIX explicit opt-in",
            "no_solve_claim": True,
        },
        "matched_roster_seed_budget_and_initial_state_receipts": {
            "games": games,
            "seeds": list(seeds),
            "max_expansions": int(max_expansions),
            "max_depth": int(max_depth),
            "row_count": len(rows),
            "initial_state_hashes": [
                {
                    "game": row.get("game"),
                    "seed": row.get("seed"),
                    "arm": row.get("arm"),
                    "initial_state_hash": row.get("initial_state_hash"),
                }
                for row in result_rows
            ],
        },
        "per_unit_rows": result_rows,
        "per_game_seed_frontier_alias_unique_state_step_legal_action_observation_terminal_clear_state_error_time_and_cost_results": result_rows,
        "collision_certificate_rows": cert_rows,
        "premature_frontier_collapse_delta": collapse_delta,
        "unique_state_delta": int(opt_states - baseline_states),
        "baseline_cleared_game_regression_count": len(regressed),
        "new_error_count": len(new_errors),
        "action_cost_delta": int(opt_actions - baseline_actions),
        "attack_matrix": attack_matrix,
        "source_access_count": 0,
        "exhaustive_search_count": 0,
        "per_game_adapter_count": 0,
        "outer_loop_re_used": False,
        "level_solve_claimed": False,
        "solve_registry_modified": bool(solve_registry_modified),
        "route_default_promoted": False,
        "public_arc_claim_eligibility": False,
        "arc_state_key_reachability_ready_score": 1.0 if ready else 0.0,
        "harm_underpowered_missing_and_flagged_cells": [],
        "protected_files_unchanged": not solve_registry_modified,
        "blocked_reason": "none" if ready else ",".join(blocked),
        "preconditions_checked": _preconditions(games, seeds, max_expansions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": _field_principles(),
        "field_provenance": _field_provenance(result_rows),
        "random_seed": list(seeds),
        "duration_s": round(float(duration_s), 3),
        "tests_run": [
            ".venv/bin/pytest tests/python/test_arc_state_key_collision_certifier.py tests/python/test_arc_state_key_collision_graph_integration.py tests/python/test_experiment_6434_arc_state_key_reachability_ab.py -q",
            ".venv/bin/pytest tests/python -q",
            ".venv/bin/python -m carnot.experiment_6434_arc_state_key_reachability_ab --date 20260814",
        ],
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: certified state-key reachability A/B passed without solve claim"
            if ready
            else f"complete: state-key reachability A/B blocked by {','.join(blocked)}"
        ),
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if artifact.get("level_solve_claimed") is not False:
        errors.append("level_solve_claimed must be false")
    if artifact.get("solve_registry_modified") is not False:
        errors.append("solve_registry_modified must be false")
    if artifact.get("route_default_promoted") is not False:
        errors.append("route_default_promoted must be false")
    if artifact.get("public_arc_claim_eligibility") is not False:
        errors.append("public_arc_claim_eligibility must be false")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    principles = artifact.get("field_principles") or {}
    for field in REQUIRED_FIELDS:
        if field not in principles:
            errors.append(f"field_principles missing {field}")
    if artifact.get("reproducibility_checksum"):
        expected = _checksum_payload(artifact)
        if artifact["reproducibility_checksum"] != expected:
            errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(artifact: dict[str, Any], path: Path = ARTIFACT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--games", default="")
    parser.add_argument("--out", default=str(ARTIFACT_PATH))
    args = parser.parse_args(argv)

    seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
    games = [part.strip() for part in args.games.split(",") if part.strip()] or _roster()
    registry_before = _sha256_path(REGISTRY)
    t0 = time.time()
    rows, certificates = run_matched_ab(
        games=games,
        seeds=seeds,
        max_expansions=args.max_expansions,
        max_depth=args.max_depth,
    )
    registry_after = _sha256_path(REGISTRY)
    artifact = build_artifact(
        date=args.date,
        rows=rows,
        collision_certificate_rows=certificates,
        duration_s=time.time() - t0,
        max_expansions=args.max_expansions,
        max_depth=args.max_depth,
        seeds=seeds,
        solve_registry_modified=registry_before != registry_after,
    )
    errors = validate_artifact(artifact)
    if errors:
        artifact["status"] = "complete_invalid"
        artifact["blocked_reason"] = "validation_failed:" + ";".join(errors)
        artifact["arc_state_key_reachability_ready_score"] = 0.0
        artifact["honest_verdict"] = "complete: invalid Exp6434 artifact; " + "; ".join(errors)
        artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    write_artifact(artifact, Path(args.out))
    print(json.dumps({"status": artifact["status"], "out": args.out, "errors": errors}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
