"""Experiment 4762: ARC rotation level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4762,
SCENARIO-ARC-WMTE-4762-ROTATION-PRECHECK,
SCENARIO-ARC-WMTE-4762-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4762-STABLE-ARTIFACT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ARTIFACT = RESULTS / "experiment_4762_levelup_attempt.json"

EXPERIMENT = "experiment_4762_levelup_attempt"
SCHEMA = "carnot.exp4762.levelup_attempt.v1"
RESULT_RELATIVE_PATH = "results/experiment_4762_levelup_attempt.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4762
PUBLIC_ROTATION_TARGETS = ("re86", "sb26", "bp35", "lf52")
ADAPTERED_FALLBACKS = (
    "ka59",
    "dc22",
    "vc33",
    "ls20",
    "sk48",
    "bp35",
    "re86",
    "sb26",
    "lf52",
)
EXTRA_PROBED_TARGETS = ("dc22",)
TIMED_NO_GATE_PROBES = {
    "dc22": {
        "elapsed_s": 115.0,
        "loop_result_path": "results/arc_loop_solve_dc22.json",
    },
}
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = "adapter_search_only_no_induction"

SPEC_REFS = [
    "REQ-ARC-WMTE-4762",
    "SCENARIO-ARC-WMTE-4762-ROTATION-PRECHECK",
    "SCENARIO-ARC-WMTE-4762-REPRODUCTION-GATE",
    "SCENARIO-ARC-WMTE-4762-STABLE-ARTIFACT",
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; a banked level is success_, a no-bank is "
        "complete_<game>_no_new_level_residual_<cause>."
    ),
    "solve_provenance": (
        "live_agent_self_discovery -- the agent advanced via its OWN attempts + runtime RE; "
        "NOT outer_loop_re and NOT a bare development_proxy adapter for a headline."
    ),
    "offline_reproduced": (
        "only reproduced levels count toward reproducible_total_levels -- a live-recorded "
        "trajectory alone is provisional."
    ),
    "reproduced_levels": "the new reproducible depth; the monotonic ARC progress metric.",
    "inference_substrate": (
        "live_llm_inference if induction runs; otherwise adapter_search_only_no_induction "
        "for standing-loop adapter/search-only runs."
    ),
    "verifier_is_oracle": (
        "the live solver's reproduction gate is execution-grounded (true); this is a SOLVE "
        "task, not a moat claim."
    ),
    "preconditions_checked": (
        "records arcade/env/generator checks so a missing-resource run emits blocked_, "
        "never a fabricated solve."
    ),
}

REQUIRED_FIELDS = (
    "experiment",
    "schema",
    "spec_refs",
    "field_principles",
    "honest_verdict",
    "solve_provenance",
    "offline_reproduced",
    "reproduced_levels",
    "new_levels_banked",
    "inference_substrate",
    "verifier_is_oracle",
    "preconditions_checked",
    "target_game",
    "attempted_games",
    "dead_ends",
    "registry_update",
    "reproducibility_checksum",
    "schema_errors",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"reproducibility_checksum", "schema_errors"}
    }


def stable_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(_checksum_payload(payload)).encode("utf-8")).hexdigest()


def _checksum_is_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def load_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = REGISTRY if path is None else Path(path)
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def registry_levels(registry: dict[str, Any]) -> dict[str, int]:
    levels: dict[str, int] = {}
    for row in registry.get("games", []):
        if not isinstance(row, dict):
            continue
        game = row.get("game")
        if not game:
            continue
        try:
            levels[str(game)] = int(row.get("levels_reproduced") or 0)
        except (TypeError, ValueError):
            levels[str(game)] = 0
    return levels


def registry_total_levels(registry: dict[str, Any]) -> int:
    try:
        return int(registry.get("reproducible_total_levels") or 0)
    except (TypeError, ValueError):
        return 0


def select_rotation_attempts(registry: dict[str, Any]) -> list[dict[str, Any]]:
    levels = registry_levels(registry)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for game in PUBLIC_ROTATION_TARGETS:
        prior = int(levels.get(game, 0))
        selected.append(
            {
                "game": game,
                "prior_level": prior,
                "target_level": prior + 1 if prior > 0 else 1,
                "reason": (
                    "preferred_public_first_contact"
                    if prior < 1
                    else "preferred_public_already_reproduced_deepen"
                ),
            }
        )
        seen.add(game)

    fallback_candidates = [
        (int(levels.get(game, 0)), index, game)
        for index, game in enumerate(ADAPTERED_FALLBACKS)
        if game not in seen and int(levels.get(game, 0)) > 0
    ]
    if fallback_candidates:
        prior, _index, game = min(fallback_candidates)
        selected.append(
            {
                "game": game,
                "prior_level": prior,
                "target_level": prior + 1,
                "reason": "shallowest_adaptered_fallback",
            }
        )
        seen.add(game)

    for game in EXTRA_PROBED_TARGETS:
        if game in seen or int(levels.get(game, 0)) < 1:
            continue
        prior = int(levels[game])
        selected.append(
            {
                "game": game,
                "prior_level": prior,
                "target_level": prior + 1,
                "reason": "post_preferred_deepen_probe",
            }
        )
        seen.add(game)
    return selected


def _gate(loop_result: dict[str, Any]) -> dict[str, Any]:
    gate = loop_result.get("reproduction_gate")
    return dict(gate) if isinstance(gate, dict) else {}


def _reached_level(loop_result: dict[str, Any]) -> int:
    gate = _gate(loop_result)
    try:
        return int(gate.get("reached_level") or loop_result.get("reached_level") or 0)
    except (TypeError, ValueError):
        return 0


def _gate_reproduced(loop_result: dict[str, Any]) -> bool:
    gate = _gate(loop_result)
    return bool(loop_result.get("offline_reproduced") and gate.get("reproduced", True))


def summarize_loop_attempt(
    *,
    game: str,
    prior_level: int,
    target_level: int,
    loop_result: dict[str, Any],
    loop_result_path: str,
) -> dict[str, Any]:
    reached = _reached_level(loop_result)
    gate_reproduced = _gate_reproduced(loop_result)
    new_levels = max(0, reached - int(prior_level)) if gate_reproduced else 0
    if not gate_reproduced:
        residual = "offline_reproduction_failed"
    elif new_levels < 1:
        residual = "reproduced_existing_or_lower_level"
    else:
        residual = "banked_new_level"
    return {
        "game": game,
        "prior_level": int(prior_level),
        "target_level": int(target_level),
        "reached_level": reached,
        "loop_result_path": loop_result_path,
        "reproduction_gate": _gate(loop_result),
        "offline_reproduced_existing_depth": bool(gate_reproduced and new_levels < 1),
        "offline_reproduced_new_depth": bool(gate_reproduced and new_levels > 0),
        "new_levels_banked": int(new_levels),
        "residual_cause": residual,
        "loop_solve_provenance": loop_result.get("solve_provenance"),
        "learned_verifier_checkpoint": loop_result.get("learned_verifier_checkpoint"),
        "solution_labels": list(loop_result.get("solution_labels") or []),
        "dead_end": _dead_end_for_attempt(game, prior_level, reached, residual),
    }


def summarize_timed_no_gate(
    *,
    game: str,
    prior_level: int,
    target_level: int,
    elapsed_s: float,
    loop_result_path: str,
) -> dict[str, Any]:
    return {
        "game": game,
        "prior_level": int(prior_level),
        "target_level": int(target_level),
        "reached_level": 0,
        "loop_result_path": loop_result_path,
        "reproduction_gate": {},
        "offline_reproduced_existing_depth": False,
        "offline_reproduced_new_depth": False,
        "new_levels_banked": 0,
        "residual_cause": "time_budget_no_terminal_gate",
        "elapsed_s": float(elapsed_s),
        "loop_solve_provenance": None,
        "learned_verifier_checkpoint": None,
        "solution_labels": [],
        "dead_end": f"{game}: timed no-gate residual after {elapsed_s:.1f}s; no new reproduced level banked",
    }


def _dead_end_for_attempt(game: str, prior_level: int, reached: int, residual: str) -> str:
    if residual == "banked_new_level":
        return f"{game}: banked L{reached} over prior L{prior_level}"
    if residual == "reproduced_existing_or_lower_level":
        return (
            f"{game}: same-depth reproduction reached L{reached} against registry prior "
            f"L{prior_level}; next run needs a new RE angle"
        )
    return f"{game}: {residual}; no new reproduced level banked"


def _best_success(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    for attempt in attempts:
        if int(attempt.get("new_levels_banked") or 0) > 0 and attempt.get(
            "offline_reproduced_new_depth"
        ):
            return attempt
    return None


def _no_bank_cause(attempts: list[dict[str, Any]]) -> str:
    if any(attempt.get("residual_cause") == "reproduced_existing_or_lower_level" for attempt in attempts):
        return "existing_depth"
    if attempts:
        return str(attempts[0].get("residual_cause") or "unknown")
    return "no_attempts"


def build_artifact(
    *,
    registry: dict[str, Any],
    attempts: list[dict[str, Any]],
    preconditions_checked: dict[str, Any],
) -> dict[str, Any]:
    success = _best_success(attempts)
    total_before = registry_total_levels(registry)
    if success is not None:
        target_game = str(success["game"])
        reached_level = int(success["reached_level"])
        new_levels = int(success["new_levels_banked"])
        verdict = f"success_{target_game}_L{reached_level}_offline_reproduced"
        offline_reproduced = True
        reproduced_levels = reached_level
        total_after = total_before + new_levels
        registry_updated = True
    else:
        target_game = str(attempts[0]["game"] if attempts else "none")
        cause = _no_bank_cause(attempts)
        verdict = f"complete_{target_game}_no_new_level_residual_{cause}"
        offline_reproduced = False
        reproduced_levels = 0
        new_levels = 0
        total_after = total_before
        registry_updated = False

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": verdict,
        "solve_provenance": SOLVE_PROVENANCE,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": int(reproduced_levels),
        "new_levels_banked": int(new_levels),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "preconditions_checked": dict(preconditions_checked),
        "target_game": target_game,
        "attempted_games": list(attempts),
        "dead_ends": [str(attempt.get("dead_end")) for attempt in attempts if attempt.get("dead_end")],
        "registry_update": {
            "updated": registry_updated,
            "path": REGISTRY_RELATIVE_PATH,
            "reproducible_total_levels_before": int(total_before),
            "reproducible_total_levels_after": int(total_after),
            "reason": "banked_new_level" if registry_updated else "no_new_level_banked",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "schema_errors": [],
    }
    artifact["reproducibility_checksum"] = stable_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def artifact_schema_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(principles, dict) or principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")
    verdict = str(payload.get("honest_verdict") or "")
    if not verdict.startswith(("success_", "complete_", "blocked_")):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance_mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_must_be_true")
    checksum = payload.get("reproducibility_checksum")
    if not _checksum_is_hex(checksum):
        errors.append("invalid_reproducibility_checksum")
    elif checksum != stable_checksum(dict(payload)):
        errors.append("checksum_mismatch")
    if int(payload.get("new_levels_banked") or 0) > 0 and payload.get("offline_reproduced") is not True:
        errors.append("bank_without_offline_reproduction")
    if int(payload.get("new_levels_banked") or 0) == 0 and payload.get("offline_reproduced") is True:
        errors.append("offline_reproduced_true_without_new_bank")
    return errors


def write_artifact(payload: dict[str, Any], path: Path | None = None) -> Path:
    output = ARTIFACT if path is None else Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def check_preconditions() -> dict[str, Any]:  # pragma: no cover - live environment boundary.
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return {
        "AGENTS.md": (REPO / "AGENTS.md").exists(),
        "CODEX.md": (REPO / "CODEX.md").exists(),
        "offline_arcade": {"ok": True, "check": "arc_solver_kit.offline_arcade()"},
        "registry_loadable": {"ok": REGISTRY.exists(), "path": REGISTRY_RELATIVE_PATH},
        "induction_needed": False,
        "qwen_igpu": {"needed": False, "ok": None},
    }


def _read_loop_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def collect_attempts(registry: dict[str, Any]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for selection in select_rotation_attempts(registry):
        game = str(selection["game"])
        result_path = RESULTS / f"arc_loop_solve_{game}.json"
        relative = f"results/arc_loop_solve_{game}.json"
        timed_probe = TIMED_NO_GATE_PROBES.get(game)
        if timed_probe is not None:
            attempts.append(
                summarize_timed_no_gate(
                    game=game,
                    prior_level=int(selection["prior_level"]),
                    target_level=int(selection["target_level"]),
                    elapsed_s=float(timed_probe["elapsed_s"]),
                    loop_result_path=str(timed_probe.get("loop_result_path") or relative),
                )
            )
            continue

        result = _read_loop_result(result_path)
        if result is not None:
            attempts.append(
                summarize_loop_attempt(
                    game=game,
                    prior_level=int(selection["prior_level"]),
                    target_level=int(selection["target_level"]),
                    loop_result=result,
                    loop_result_path=relative,
                )
            )
            continue

        attempts.append(
            summarize_timed_no_gate(
                game=game,
                prior_level=int(selection["prior_level"]),
                target_level=int(selection["target_level"]),
                elapsed_s=0.0,
                loop_result_path=relative,
            )
        )
    return attempts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)

    preconditions = check_preconditions()
    registry = load_registry(REGISTRY)
    attempts = collect_attempts(registry)
    artifact = build_artifact(
        registry=registry,
        attempts=attempts,
        preconditions_checked=preconditions,
    )
    write_artifact(artifact, ARTIFACT)
    print(f"honest_verdict={artifact['honest_verdict']}")
    print(f"new_levels_banked={artifact['new_levels_banked']}")
    print(f"schema_errors={artifact['schema_errors']}")
    return 0 if not artifact["schema_errors"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
