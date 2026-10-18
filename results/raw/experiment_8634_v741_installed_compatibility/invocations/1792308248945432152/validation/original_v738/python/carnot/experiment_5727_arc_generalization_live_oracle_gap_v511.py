"""Experiment 5727: full-registry ARC live-vs-oracle generalization gap.

This module compiles the fresh ``scripts/arc_leaderboard_eval.py --games oracle
--policy e3 --budget 400`` result into the terminal artifact required by
REQ-ARC-WMTE-5727. It does not run a solver, read game source, claim registry
credit, or attempt to close any gap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_5727_arc_generalization_live_oracle_gap_v511.json"
LIVE_GAP_RELATIVE_PATH = "results/arc_live_oracle_gap.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
VERIFIER_GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
EXPERIMENT_ID = "experiment_5727_arc_generalization_live_oracle_gap_v511"
SCHEMA = "carnot.exp5727.arc_generalization_live_oracle_gap.v1"
RANDOM_SEED = 20260719
EXPECTED_POLICY = "e3"
EXPECTED_BUDGET = 400
EXPECTED_HARNESS = "scripts/arc_leaderboard_eval.py"
EXPECTED_REGISTRY_GAMES = 25

REQUIRED_ARTIFACT_FIELDS = (
    "harness_used",
    "policy_kind",
    "budget_per_game",
    "oracle_source_registry_hash",
    "games_measured",
    "live_levels_total",
    "oracle_levels_total",
    "gap_total",
    "per_game_gap",
    "worst_gap_games",
    "verifier_gaps_entries_added",
    "any_new_level_found",
    "inference_substrate",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PROVENANCE = {
    "harness_used": {
        "principle": "exact reproducibility of the measurement; the harness path is part of the claim."
    },
    "policy_kind": {
        "principle": "e3 means the real submitted E3AgentPolicy cascade, not a banked replay or adapter path."
    },
    "budget_per_game": {
        "principle": "400 is the submission-faithful action cap rather than an optimistic unbounded exploration budget."
    },
    "oracle_source_registry_hash": {
        "principle": "content-addressed registry baseline prevents measuring against a stale oracle ceiling."
    },
    "games_measured": {
        "principle": "the no-silent-caps ethos requires 25 games or an explicit skipped-game list."
    },
    "live_levels_total": {"principle": "the live side of the north-star generalization metric."},
    "oracle_levels_total": {
        "principle": "the offline-dev oracle ceiling the live path is measured against."
    },
    "gap_total": {"principle": "the total live-vs-oracle headroom this floor exists to shrink."},
    "per_game_gap": {
        "principle": "per-game gaps and oracle-win provenance make the aggregate actionable."
    },
    "worst_gap_games": {
        "principle": "gap numbers are only useful when the stall class is grounded in run evidence."
    },
    "verifier_gaps_entries_added": {
        "principle": "connects new missing-discriminator findings to the verifier backlog."
    },
    "any_new_level_found": {
        "principle": "new level credit requires honest frame.levels_completed evidence and a separate reproduction gate."
    },
    "inference_substrate": {
        "principle": "distinguishes offline frame-only simulation from per-game real Qwen3.5-9B-MTP escalation."
    },
    "preconditions_checked": {
        "principle": "records the GGUF, registry, GPU pinning, and non-default port gates checked before trusting the run."
    },
    "random_seed": {"principle": "determinism precondition for replaying the measurement."},
    "reproducibility_checksum": {
        "principle": "content-addressed result payload catches silent drift."
    },
    "honest_verdict": {
        "principle": "terminal-prefixed complete:/blocked: verdict preserves blocked preconditions and honest nulls."
    },
}

VALID_PROVENANCE = {
    "development_proxy",
    "live_agent_self_discovery",
    "outer_loop_re",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    return _sha256_bytes(_stable_json(payload).encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _registry_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _normalise_provenance(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    for token in VALID_PROVENANCE:
        if re.search(rf"\b{re.escape(token)}\b", value):
            return token
    return None


def _collect_provenance_values(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "solve_provenance":
                token = _normalise_provenance(child)
                if token:
                    found.append(token)
            found.extend(_collect_provenance_values(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_provenance_values(child))
    else:
        token = _normalise_provenance(value)
        if token:
            found.append(token)
    return found


def oracle_win_solve_provenance(registry_row: Mapping[str, Any] | None) -> str:
    """Return the most conservative structured provenance label available."""

    if registry_row is None:
        return "unknown_registry_missing"
    values = _collect_provenance_values(registry_row)
    if "development_proxy" in values:
        return "development_proxy"
    if "live_agent_self_discovery" in values:
        return "live_agent_self_discovery"
    if "outer_loop_re" in values:
        return "outer_loop_re"
    return "unknown_registry_unstructured"


def _registry_by_game(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }


def _int_field(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def per_game_gap_rows(
    live_result: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    registry_rows = _registry_by_game(registry)
    rows: list[dict[str, Any]] = []
    for row in live_result.get("per_game", []):
        if not isinstance(row, Mapping):
            continue
        game = str(row.get("game") or "")
        if not game:
            continue
        reg_row = registry_rows.get(game)
        live_levels = _int_field(row, "levels")
        oracle_levels = _int_field(
            row,
            "oracle_levels",
            _int_field(reg_row or {}, "levels_reproduced"),
        )
        rows.append(
            {
                "game": game,
                "live_levels": live_levels,
                "oracle_levels": oracle_levels,
                "gap": max(0, oracle_levels - live_levels),
                "oracle_win_solve_provenance": oracle_win_solve_provenance(reg_row),
            }
        )
    return rows


def _provenance_tiebreak(provenance: str) -> int:
    if provenance == "development_proxy":
        return 0
    if provenance == "live_agent_self_discovery":
        return 1
    return 2


def select_worst_gap_rows(per_game_gap: Sequence[Mapping[str, Any]], limit: int = 3) -> list[dict]:
    ranked = sorted(
        per_game_gap,
        key=lambda row: (
            -_int_field(row, "gap"),
            _provenance_tiebreak(str(row.get("oracle_win_solve_provenance") or "")),
            str(row.get("game") or ""),
        ),
    )
    return [dict(row) for row in ranked[:limit] if _int_field(row, "gap") > 0]


def _last_induction_attempt(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    diagnostics = row.get("policy_diagnostics")
    attempts = diagnostics.get("induction_attempts") if isinstance(diagnostics, Mapping) else None
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], Mapping):
        return attempts[-1]
    return None


def stall_class_for_row(row: Mapping[str, Any], budget: int) -> str:
    attempt = _last_induction_attempt(row)
    if attempt is not None and not bool(attempt.get("planned")):
        skipped = str(attempt.get("skipped") or "")
        if (
            skipped
            in {
                "proposer_failed_or_missing_root",
                # REQ-ARC-WMTE-6610 (2026-08-21) split the conflated label above into the three
                # below; artifacts recorded before that date carry the old one, later reruns emit
                # the new ones. Both generations must classify identically.
                "proposer_failed",
                "missing_plan_start_grid",
                "proposer_failed_and_missing_plan_start_grid",
                "world_model_accuracy_below_threshold",
                "hidden_state_trust_below_threshold",
                "degenerate_goal_predicate",
            }
            or attempt.get("refinement_rounds_used") is not None
        ):
            return "INDUCTION QUALITY"
    actions = _int_field(row, "actions")
    if actions >= int(budget):
        return "SEARCH/BUDGET"
    if _int_field(row, "levels") == 0:
        return "PERCEPTION"
    return "OTHER: partial live transfer short of oracle depth"


def _frame_evidence(row: Mapping[str, Any]) -> str:
    frames = row.get("frame_sequence")
    if isinstance(frames, list) and frames:
        first = frames[0] if isinstance(frames[0], Mapping) else {}
        last = frames[-1] if isinstance(frames[-1], Mapping) else {}
        return (
            "frame_sequence evidence: "
            f"frame[{first.get('frame_index')}].levels_completed="
            f"{first.get('levels_completed')} hash={first.get('grid_hash')}; "
            f"frame[{last.get('frame_index')}].levels_completed="
            f"{last.get('levels_completed')} hash={last.get('grid_hash')}"
        )
    return "frame_sequence evidence unavailable in harness row"


def grounded_evidence_for_row(row: Mapping[str, Any], budget: int) -> list[str]:
    evidence = [
        (
            "results/arc_live_oracle_gap.json per_game row: "
            f"game={row.get('game')} live_levels={row.get('levels')} "
            f"oracle_levels={row.get('oracle_levels')} gap_vs_oracle={row.get('gap_vs_oracle')} "
            f"actions={row.get('actions')}/{budget} "
            f"actions_to_first_levelup={row.get('actions_to_first_levelup')}"
        ),
        _frame_evidence(row),
    ]
    nav = row.get("navigation_diagnostics")
    if isinstance(nav, Mapping):
        evidence.append(
            "navigation_diagnostics: "
            f"reset_replay_steps={nav.get('reset_replay_steps')} "
            f"forward_walk_hit_rate={nav.get('forward_walk_hit_rate')}"
        )
    attempt = _last_induction_attempt(row)
    if attempt is not None:
        evidence.append(
            "policy_diagnostics.induction_attempts[-1]: "
            f"reason={attempt.get('reason')} transition_count={attempt.get('transition_count')} "
            f"planned={attempt.get('planned')} skipped={attempt.get('skipped')} "
            f"verify_accuracy={attempt.get('verify_accuracy')} "
            f"refinement_rounds_used={attempt.get('refinement_rounds_used')}"
        )
    else:
        evidence.append("policy_diagnostics.induction_attempts: []")
    return evidence


def verifier_gap_reference(stall_class: str, verifier_gaps_text: str) -> str:
    if stall_class == "INDUCTION QUALITY" and "GAP-3" in verifier_gaps_text:
        return "GAP-3"
    if stall_class == "SEARCH/BUDGET":
        return "not_a_new_missing_discriminator_search_or_budget_gap"
    if stall_class == "PERCEPTION" and "perception" in verifier_gaps_text.lower():
        return "existing_perception_gap"
    return "no_new_missing_discriminator_logged"


def _tier3_escalated(row: Mapping[str, Any]) -> bool:
    diagnostics = row.get("policy_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    proposer = diagnostics.get("proposer")
    if isinstance(proposer, Mapping) and proposer.get("instantiated"):
        return True
    attempts = diagnostics.get("induction_attempts")
    return bool(attempts)


def inference_substrate_by_game(
    live_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in live_rows:
        game = str(row.get("game") or "")
        diagnostics = row.get("policy_diagnostics")
        proposer = diagnostics.get("proposer") if isinstance(diagnostics, Mapping) else {}
        proposer = proposer if isinstance(proposer, Mapping) else {}
        escalated = _tier3_escalated(row)
        out[game] = {
            "base": "offline_sim_no_quota_frame_only_live_agent",
            "tier3_qwen35_mtp_escalated": escalated,
            "tier3_substrate": (
                "real_qwen35_9b_mtp_local_gguf_inference" if escalated else "not_used"
            ),
            "proposer_repo_substr": proposer.get("repo_substr"),
            "proposer_port": proposer.get("port"),
            "proposer_mtp": proposer.get("mtp"),
        }
    return out


def _preconditions_with_registry(
    preconditions_checked: Mapping[str, Any] | None,
    registry: Mapping[str, Any],
    registry_hash: str,
) -> dict[str, Any]:
    preconditions = dict(preconditions_checked or {})
    preconditions.setdefault(
        "registry_reproducible_total_games", registry.get("reproducible_total_games")
    )
    preconditions.setdefault(
        "registry_premise_ok",
        registry.get("reproducible_total_games") == EXPECTED_REGISTRY_GAMES,
    )
    preconditions["oracle_source_registry_hash"] = registry_hash
    return preconditions


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    live_gap_path: Path | None = None,
    registry_path: Path | None = None,
    verifier_gaps_path: Path | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    live_path = live_gap_path or root / LIVE_GAP_RELATIVE_PATH
    reg_path = registry_path or root / REGISTRY_RELATIVE_PATH
    gaps_path = verifier_gaps_path or root / VERIFIER_GAPS_RELATIVE_PATH
    live_result = _read_json(live_path)
    registry = _read_yaml(reg_path)
    verifier_gaps_text = gaps_path.read_text(encoding="utf-8") if gaps_path.exists() else ""
    reg_hash = _registry_hash(reg_path)
    per_game_gap = per_game_gap_rows(live_result, registry)
    live_rows = [row for row in live_result.get("per_game", []) if isinstance(row, Mapping)]
    live_by_game = {str(row.get("game")): row for row in live_rows}
    worst_base = select_worst_gap_rows(per_game_gap, limit=3)
    worst = []
    for gap_row in worst_base:
        live_row = live_by_game.get(gap_row["game"], {})
        stall_class = stall_class_for_row(live_row, EXPECTED_BUDGET)
        worst.append(
            {
                **gap_row,
                "stall_class": stall_class,
                "grounded_evidence": grounded_evidence_for_row(live_row, EXPECTED_BUDGET),
                "verifier_gap_reference": verifier_gap_reference(stall_class, verifier_gaps_text),
                "not_attempted_to_fix": (
                    "No per-game fix, hand solve, source read, registry edit, or new credit was attempted."
                ),
            }
        )
    measured_games = {str(row.get("game")) for row in live_rows if row.get("game")}
    registry_games = {
        str(row.get("game"))
        for row in registry.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    new_level_rows = [
        row for row in per_game_gap if int(row["live_levels"]) > int(row["oracle_levels"])
    ]
    live_levels_total = sum(int(row["live_levels"]) for row in per_game_gap)
    oracle_levels_total = sum(int(row["oracle_levels"]) for row in per_game_gap)
    gap_total = sum(int(row["gap"]) for row in per_game_gap)
    games_measured = len(per_game_gap)
    honest_verdict = (
        f"complete: arc_generalization_live_oracle_gap_{live_levels_total}_of_"
        f"{oracle_levels_total}_levels_gap_{gap_total}"
    )
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "field_provenance": FIELD_PROVENANCE,
        "harness_used": EXPECTED_HARNESS,
        "policy_kind": str(live_result.get("policy") or ""),
        "budget_per_game": _int_field(live_result, "budget"),
        "oracle_source_registry_hash": reg_hash,
        "games_measured": games_measured,
        "expected_registry_games": registry.get("reproducible_total_games"),
        "skipped_games": sorted(registry_games - measured_games),
        "live_levels_total": live_levels_total,
        "oracle_levels_total": oracle_levels_total,
        "gap_total": gap_total,
        "per_game_gap": per_game_gap,
        "worst_gap_games": worst,
        "verifier_gaps_entries_added": [],
        "any_new_level_found": bool(new_level_rows),
        "new_level_evidence": new_level_rows,
        "target_model": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "model_specs": {
            "name": "Qwen3.5-9B-MTP",
            "model_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
            "repo_substr": "Qwen3.5-9B-MTP",
            "gguf_filename": "Qwen3.5-9B-Q4_K_M.gguf",
            "mtp": True,
            "spec_type": "draft-mtp",
            "kv_quant": "q8_0",
            "no_think_prefix": "/no_think\n",
            "proposer_port_env": "CARNOT_ARC_PROPOSER_PORT",
            "cuda_gpu_env": "CARNOT_ARC_GENERATOR_CUDA_GPU",
        },
        "inference_substrate": (
            "offline_sim_no_quota_frame_only_live_agent_with_per_game_qwen35_mtp_notes"
        ),
        "inference_substrate_by_game": inference_substrate_by_game(live_rows),
        "preconditions_checked": _preconditions_with_registry(
            preconditions_checked,
            registry,
            reg_hash,
        ),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    return artifact


def blocked_artifact(
    honest_verdict: str,
    *,
    preconditions_checked: Mapping[str, Any],
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "field_provenance": FIELD_PROVENANCE,
        "harness_used": EXPECTED_HARNESS,
        "policy_kind": EXPECTED_POLICY,
        "budget_per_game": EXPECTED_BUDGET,
        "oracle_source_registry_hash": "",
        "games_measured": 0,
        "expected_registry_games": EXPECTED_REGISTRY_GAMES,
        "skipped_games": [],
        "live_levels_total": 0,
        "oracle_levels_total": 0,
        "gap_total": 0,
        "per_game_gap": [],
        "worst_gap_games": [],
        "verifier_gaps_entries_added": [],
        "any_new_level_found": False,
        "new_level_evidence": [],
        "target_model": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "model_specs": {
            "name": "Qwen3.5-9B-MTP",
            "model_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
            "repo_substr": "Qwen3.5-9B-MTP",
            "gguf_filename": "Qwen3.5-9B-Q4_K_M.gguf",
            "mtp": True,
            "spec_type": "draft-mtp",
            "kv_quant": "q8_0",
            "no_think_prefix": "/no_think\n",
            "proposer_port_env": "CARNOT_ARC_PROPOSER_PORT",
            "cuda_gpu_env": "CARNOT_ARC_GENERATOR_CUDA_GPU",
        },
        "inference_substrate": "not_run_precondition_blocked",
        "inference_substrate_by_game": {},
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance missing")
    else:
        for field, expected in FIELD_PROVENANCE.items():
            if provenance.get(field) != expected:
                errors.append(f"field_provenance mismatch: {field}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (
        verdict.startswith("complete:")
        or verdict.startswith("blocked_")
        or verdict.startswith("blocked:")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if artifact.get("reproducibility_checksum") != _checksum_payload(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("policy_kind") and artifact.get("policy_kind") != EXPECTED_POLICY:
        errors.append("policy_kind must be e3")
    if artifact.get("budget_per_game") and artifact.get("budget_per_game") != EXPECTED_BUDGET:
        errors.append("budget_per_game must be 400")
    if artifact.get("worst_gap_games") and len(artifact["worst_gap_games"]) != 3:
        errors.append("worst_gap_games must contain exactly three games")
    return errors


def _default_preconditions(root: Path) -> dict[str, Any]:
    gguf = root.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3.5-9B-MTP-GGUF"
    hits = sorted(gguf.glob("snapshots/*/Qwen3.5-9B-Q4_K_M.gguf"))
    port = os.environ.get("CARNOT_ARC_PROPOSER_PORT")
    return {
        "qwen35_9b_gguf_cached": bool(hits and hits[0].exists()),
        "qwen35_9b_gguf_path": str(hits[0]) if hits else "",
        "cuda_gpu_env": os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU", ""),
        "proposer_port": int(port) if port and port.isdigit() else None,
        "proposer_port_non_default": bool(port and port.isdigit() and int(port) != 8919),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = list(argv if argv is not None else sys.argv[1:])
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    if "--out" in args:
        out_path = Path(args[args.index("--out") + 1])
    artifact = build_artifact(
        root=REPO_ROOT,
        preconditions_checked=_default_preconditions(REPO_ROOT),
        random_seed=RANDOM_SEED,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise SystemExit("; ".join(errors))
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
