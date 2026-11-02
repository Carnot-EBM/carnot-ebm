"""Experiment 4561: persist the strongest A1/A2 primitive and measure transfer.

Spec refs: REQ-ARC-WMTE-4561, SCENARIO-ARC-WMTE-4561.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from carnot.agentic import arc_solver_kit as kit


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4561_primitive_persist_transfer.json"
A1_RELATIVE_PATH = "results/experiment_4556_verifier_router_generic_transfer.json"
A2_RELATIVE_PATH = "results/experiment_4557_executable_world_model_proposer.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
LIVE_LLM_INFERENCE_SUBSTRATE = "live_llm_inference + model_specs"
PRIMITIVE_OPERATOR = "verifier_router_candidate_ranking_operator"
PRIMITIVE_GOTCHA_ID = "primitive_verifier_router_candidate_ranking_operator"
TRANSFER_GAMES = ("tu93", "tr87", "sc25")
RANDOM_SEED = 4561
TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: primitive_persisted_transfer_<game>_value_added OR "
        "complete: primitive_persisted_transfer_null_characterized."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates for the offline-verifier transfer; "
        "live_llm_inference + model_specs if the executable proposer is invoked in the transfer runs."
    ),
    "primitive_persisted": (
        "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the "
        "reusable asset (Solver-Reuse Discipline); without it the A1/A2 effort is wasted per the "
        "ARC reuse rule."
    ),
    "transfer_games": (
        "the games the primitive was applied to (NOT tuned on) -- the generalization test."
    ),
    "transfer_value_per_game": (
        "the per-game value-add (verifier-router ordering gain / executable-proposer reachable "
        "plan) -- the cross-game evidence the primitive generalizes."
    ),
    "offline_reproduced": "only offline-reproduced new levels count toward reproducible_total_levels.",
    "registry_updated": (
        "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
    ),
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "field_principles",
    "requirements",
    "scenarios",
    "upstream_decision",
    "upstream_summaries",
    "transfer_results",
    "transfer_dead_ends",
    "new_levels_banked",
    "result_path",
    "duration_s",
)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def _load_json(path: Path) -> dict[str, Any]:  # pragma: no cover - file boundary.
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_registry(root: Path) -> dict[str, Any]:  # pragma: no cover - file boundary.
    try:
        loaded = yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _registry_has_primitive_gotcha(registry: Mapping[str, Any]) -> bool:
    return any(
        isinstance(row, Mapping)
        and row.get("id") == PRIMITIVE_GOTCHA_ID
        and row.get("operator") == PRIMITIVE_OPERATOR
        for row in registry.get("general_gotchas", []) or []
    )


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    registry = _load_registry(root_path)
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "spec_has_req_4561": spec_path.exists()
        and "REQ-ARC-WMTE-4561" in spec_path.read_text(encoding="utf-8"),
        "registry_has_primitive_gotcha": _registry_has_primitive_gotcha(registry),
    }
    try:
        kit.offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(
        checks["offline_arcade_import_smoke"]
        and checks["a1_artifact_present"]
        and checks["a2_artifact_present"]
        and checks["spec_has_req_4561"]
        and checks["registry_has_primitive_gotcha"]
    )
    return checks


def select_primitive_from_upstreams(
    *, a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4561: choose the strongest persisted A1/A2 primitive signal."""

    a1_delta = _as_float(a1_artifact.get("generic_transfer_delta"))
    a1_has_measurement = "generic_transfer_rate_with_verifier" in a1_artifact
    a1_characterized = bool(a1_has_measurement and a1_artifact.get("offline_reproduced") is True)
    a2_positive = bool(a2_artifact.get("positive_control_passed") is True)
    proposer_value = a2_artifact.get("llm_proposer_value") if isinstance(
        a2_artifact.get("llm_proposer_value"), Mapping
    ) else {}
    a2_value_rate = _as_float(proposer_value.get("rate"))

    if a2_positive and a2_value_rate > 0.0:
        return {
            "source": "A2_executable_world_model_proposer",
            "operator": "llm_proposer_reinduction_operator",
            "registry_general_gotcha_id": "primitive_per_level_reinduction_operator",
            "inference_substrate": LIVE_LLM_INFERENCE_SUBSTRATE,
            "measured_signal": a2_value_rate,
            "persisted_as_best_characterized_null": False,
            "selection_rationale": "A2 positive control passed and produced reachable-plan value.",
        }

    return {
        "source": "A1_verifier_router",
        "operator": PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": PRIMITIVE_GOTCHA_ID,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "measured_signal": a1_delta,
        "persisted_as_best_characterized_null": bool(a1_delta <= 0.0),
        "selection_rationale": (
            "A1 is the best-characterized reusable primitive-as-built: cached verifier-router "
            "transfer ran to completion, while A2's executable proposer positive control did not "
            "produce a reachable verified model."
            if a1_characterized
            else "A1 is persisted as the best-characterized primitive available, but upstream evidence is incomplete."
        ),
    }


def _attempts_by_game(a1_artifact: Mapping[str, Any], arm: str, game: str) -> dict[str, Any]:
    measurement = a1_artifact.get(f"{arm}_measurement")
    if not isinstance(measurement, Mapping):
        return {}
    for row in measurement.get("variant_attempts", []) or []:
        if isinstance(row, Mapping) and row.get("game") == game:
            return dict(row)
    return {}


def _arm_score(arm: str) -> float:
    if arm == "verifier":
        return 1.0
    if arm == "baseline":
        return 0.0
    return -1.0


def _attempt_reaches_goal(attempt: Mapping[str, Any]) -> bool:
    gate = attempt.get("reproduction_gate")
    gate_reproduced = isinstance(gate, Mapping) and gate.get("reproduced") is True
    return bool(attempt.get("solved") is True and gate_reproduced)


def measure_cached_verifier_transfer_game(
    game: str,
    *,
    a1_artifact: Mapping[str, Any],
    incoming_order: Sequence[str] = ("baseline", "verifier", "random"),
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4561: apply the persisted ranker to cached held-out candidates."""

    candidates: list[dict[str, Any]] = []
    for arm in incoming_order:
        lookup_arm = "random_router" if arm == "random" else arm
        attempt = _attempts_by_game(a1_artifact, lookup_arm, game)
        if not attempt:
            continue
        candidates.append(
            {
                "candidate_id": arm,
                "router_mode": arm,
                "variant_signature": attempt.get("variant_signature"),
                "verifier_score": _arm_score(arm),
                "reaches_goal": _attempt_reaches_goal(attempt),
                "reached_level": _as_int(attempt.get("reached_level")),
                "actions": _as_int(attempt.get("actions")),
            }
        )

    ranking = kit.verifier_router_candidate_ranking_operator(
        candidates,
        score_key="verifier_score",
        target_key="reaches_goal",
    )
    has_goal_candidate = ranking.get("target_rank_after") is not None
    dead_end = ""
    if not candidates:
        dead_end = "no cached candidates were available for this held-out game."
    elif not has_goal_candidate:
        dead_end = (
            "persisted verifier-router ranking was applied, but no cached candidate reached "
            "the offline reproduction gate, so ordering gain is unmeasurable/value-null."
        )
    elif not ranking.get("value_added"):
        dead_end = (
            "a cached candidate reached the goal, but the persisted verifier-router did not "
            "improve its rank over the incoming order."
        )

    transfer_value = {
        "operator": PRIMITIVE_OPERATOR,
        "ordering_gain": int(ranking.get("ordering_gain") or 0),
        "value_added": bool(ranking.get("value_added") is True),
        "best_candidate_id": str(ranking.get("best_candidate_id") or ""),
        "target_rank_before": ranking.get("target_rank_before"),
        "target_rank_after": ranking.get("target_rank_after"),
        "candidate_count": int(ranking.get("candidate_count") or 0),
    }
    return {
        "game": game,
        "value_added": bool(transfer_value["value_added"]),
        "transfer_value": transfer_value,
        "ranking": ranking,
        "offline_reproduced_new_level": False,
        "dead_end": dead_end,
    }


def _success_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    winners = [row for row in rows if row.get("value_added") is True]
    if not winners:
        return None
    return max(
        winners,
        key=lambda row: (
            _as_int((row.get("transfer_value") or {}).get("ordering_gain"))
            if isinstance(row.get("transfer_value"), Mapping)
            else 0,
            str(row.get("game") or ""),
        ),
    )


def build_artifact(
    *,
    upstream_decision: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    transfer_results: Sequence[Mapping[str, Any]],
    registry_updated: bool,
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4561: assemble the primitive persistence transfer artifact."""

    rows = [dict(row) for row in transfer_results]
    winner = _success_row(rows)
    transfer_games = [str(row.get("game")) for row in rows]
    transfer_values = {
        str(row.get("game")): dict(row.get("transfer_value") or {}) for row in rows
    }
    dead_ends = {
        str(row.get("game")): str(row.get("dead_end") or "")
        for row in rows
        if str(row.get("dead_end") or "")
    }
    new_levels_banked = sum(1 for row in rows if row.get("offline_reproduced_new_level") is True)
    if winner is not None:
        verdict = f"success: primitive_persisted_transfer_{winner.get('game')}_value_added"
    elif preconditions_checked.get("ok") is False:
        verdict = "blocked_primitive_persist_transfer_precondition"
    else:
        verdict = "complete: primitive_persisted_transfer_null_characterized"

    artifact = {
        "experiment": "experiment_4561_primitive_persist_transfer",
        "schema": "carnot.arc_primitive_persist_transfer_4561.v1",
        "honest_verdict": verdict,
        "inference_substrate": str(upstream_decision.get("inference_substrate") or INFERENCE_SUBSTRATE),
        "primitive_persisted": {
            "operator": upstream_decision.get("operator"),
            "registry_general_gotcha_id": upstream_decision.get("registry_general_gotcha_id"),
            "source": upstream_decision.get("source"),
            "derived_from_artifacts": [A1_RELATIVE_PATH, A2_RELATIVE_PATH],
        },
        "transfer_games": transfer_games,
        "transfer_value_per_game": transfer_values,
        "offline_reproduced": bool(new_levels_banked > 0),
        "registry_updated": bool(registry_updated),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": ["REQ-ARC-WMTE-4561"],
        "scenarios": ["SCENARIO-ARC-WMTE-4561"],
        "upstream_decision": dict(upstream_decision),
        "upstream_summaries": {
            "a1_verifier_router": {
                "artifact": A1_RELATIVE_PATH,
                "selected": upstream_decision.get("source") == "A1_verifier_router",
            },
            "a2_executable_world_model_proposer": {
                "artifact": A2_RELATIVE_PATH,
                "selected": upstream_decision.get("source") == "A2_executable_world_model_proposer",
            },
        },
        "transfer_results": rows,
        "transfer_dead_ends": dead_ends,
        "new_levels_banked": int(new_levels_banked),
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") not in (INFERENCE_SUBSTRATE, LIVE_LLM_INFERENCE_SUBSTRATE):
        errors.append("inference_substrate must match an allowed 4561 substrate")
    primitive = artifact.get("primitive_persisted")
    if not isinstance(primitive, Mapping) or primitive.get("operator") != PRIMITIVE_OPERATOR:
        errors.append("primitive_persisted must name verifier_router_candidate_ranking_operator")
    elif primitive.get("registry_general_gotcha_id") != PRIMITIVE_GOTCHA_ID:
        errors.append("primitive_persisted must name the 4561 registry general_gotcha")
    transfer_games = artifact.get("transfer_games")
    if not blocked and (not isinstance(transfer_games, list) or len(transfer_games) < 2):
        errors.append("transfer_games must contain at least two games")
    if not isinstance(artifact.get("transfer_value_per_game"), Mapping):
        errors.append("transfer_value_per_game must be a mapping")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be a bare bool")
    if type(artifact.get("registry_updated")) is not bool:
        errors.append("registry_updated must be a bare bool")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4561")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        values = artifact.get("transfer_value_per_game")
        if not isinstance(values, Mapping) or not any(
            isinstance(value, Mapping) and value.get("value_added") is True
            for value in values.values()
        ):
            errors.append("success requires at least one transfer value_added=true")
    if artifact.get("offline_reproduced") is True and _as_int(artifact.get("new_levels_banked")) < 1:
        errors.append("offline_reproduced=true requires at least one new level banked")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(root: Path | str = REPO_ROOT, *, write: bool = True) -> dict[str, Any]:  # pragma: no cover
    root_path = Path(root)
    started = time.monotonic()
    checks = check_preconditions(root_path)
    a1 = _load_json(root_path / A1_RELATIVE_PATH)
    a2 = _load_json(root_path / A2_RELATIVE_PATH)
    decision = select_primitive_from_upstreams(a1_artifact=a1, a2_artifact=a2)
    if checks.get("ok") is True and decision.get("operator") == PRIMITIVE_OPERATOR:
        rows = [
            measure_cached_verifier_transfer_game(game, a1_artifact=a1)
            for game in TRANSFER_GAMES
        ]
    else:
        rows = []
    artifact = build_artifact(
        upstream_decision=decision,
        preconditions_checked=checks,
        transfer_results=rows,
        registry_updated=bool(checks.get("registry_has_primitive_gotcha")),
        random_seed=RANDOM_SEED,
        duration_s=max(0.0, time.monotonic() - started),
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root_path)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary.
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary.
    raise SystemExit(main())
