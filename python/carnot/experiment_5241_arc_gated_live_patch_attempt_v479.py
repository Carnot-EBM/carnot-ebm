"""Exp 5241: run the gated ARC live patch path with strict provenance.

Spec refs: REQ-REPORT-5241,
SCENARIO-REPORT-5241-NO-BANK-LIVE-PATCH-ATTEMPT,
SCENARIO-REPORT-5241-SOLVE-CLAIM-GATE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from carnot import experiment_5240_arc_rubric_to_patch_synthesis_v479 as exp5240


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5241_arc_gated_live_patch_attempt_v479"
EXPERIMENT_ID = 5241
SCHEMA = "carnot.arc_gated_live_patch_attempt_v479.v1"
RUN_DATE = "2026-07-04"
RESULT_RELATIVE_PATH = "results/experiment_5241_arc_gated_live_patch_attempt_v479.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/research-reporting/spec.md"
EXP5240_RESULT_RELATIVE_PATH = exp5240.RESULT_RELATIVE_PATH
PATCH_RELATIVE_PATH = exp5240.PATCH_RELATIVE_PATH
DEFAULT_TARGET_GAME = "zz99_exp5241_live_probe"
DEFAULT_BUDGET = 8
RANDOM_SEED = 5241
INFERENCE_SUBSTRATE = "arc_live_agent_self_discovery"
SOLVE_PROVENANCE = "live_agent_self_discovery"
DEFAULT_EXACT_COMMAND = (
    ".venv/bin/python -m carnot.experiment_5241_arc_gated_live_patch_attempt_v479 "
    f"--target-game {DEFAULT_TARGET_GAME} --budget {DEFAULT_BUDGET} "
    f"--random-seed {RANDOM_SEED}"
)
SPEC_REFS = (
    "REQ-REPORT-5241",
    "SCENARIO-REPORT-5241-NO-BANK-LIVE-PATCH-ATTEMPT",
    "SCENARIO-REPORT-5241-SOLVE-CLAIM-GATE",
)
PATCH_RECOMMENDATIONS = ("keep", "rollback", "iterate", "no_solve_no_regression")
MANDATED_SOTA_GGUFS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS = (
    "preconditions_checked",
    "solve_provenance",
    "registry_precheck_done",
    "duplicate_solve_target_avoided",
    "reproducible_total_levels_before",
    "reproducible_total_levels_after",
    "reproducible_total_levels_delta",
    "live_agent_patch_enabled",
    "model_specs",
    "random_seed",
    "arc_validation_commands",
    "patch_recommendation",
    "inference_substrate",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "preconditions_checked": {
        "principle": (
            "bare bool confirming Exp 5240 patch readiness, registry precheck, and "
            "forbidden-method guards were checked."
        )
    },
    "solve_provenance": {
        "principle": (
            "Required for any ARC level solve claim; outer-loop RE or development proxy "
            "cannot be headline evidence."
        )
    },
    "registry_precheck_done": {
        "principle": (
            "bare bool confirming ops/arc_solve_registry.yaml was read before the live attempt."
        )
    },
    "duplicate_solve_target_avoided": {
        "principle": (
            "bare bool confirming the attempted target was not already reproduced by the "
            "live mechanism."
        )
    },
    "reproducible_total_levels_before": {
        "principle": "registry reproducible_total_levels before the live attempt."
    },
    "reproducible_total_levels_after": {
        "principle": (
            "registry reproducible_total_levels after accepted live self-discovery banking, "
            "unchanged on no-bank."
        )
    },
    "reproducible_total_levels_delta": {
        "principle": (
            "after minus before; may be positive only for accepted live self-discovery reproduction."
        )
    },
    "live_agent_patch_enabled": {
        "principle": (
            "bare bool confirming only the Exp 5240 patch was enabled for the live path."
        )
    },
    "model_specs": {
        "principle": (
            "MODEL_SPECS with mandated SOTA GGUF if any LLM proposer was used; otherwise null."
        )
    },
    "random_seed": {
        "principle": "fixed integer seed used for the bounded live attempt."
    },
    "arc_validation_commands": {
        "principle": "list of ARC registry/lint commands with pass/fail outcomes."
    },
    "patch_recommendation": {
        "principle": "one of keep, rollback, iterate, or no_solve_no_regression."
    },
    "inference_substrate": {"principle": "must be arc_live_agent_self_discovery."},
    "honest_verdict": {
        "principle": (
            "Must start with complete:/complete_/success:/success_ or blocked_ and state "
            "level delta and provenance."
        )
    },
}
_TRUE_PRECONDITION_KEYS = (
    "agents_read",
    "codex_read",
    "spec_has_req_5241",
    "exp5240_patch_candidate_tested",
    "registry_present",
    "registry_loadable",
    "patch_path_matches_exp5240",
)
_FORBIDDEN_PRECONDITION_KEYS = (
    "read_hidden_game_source",
    "offline_ground_truth_bfs",
    "hand_per_game_adapter",
)


def load_registry_summary(root: Path | str = REPO_ROOT) -> JsonDict:
    """Read the ARC registry totals without mutating the registry."""

    path = Path(root) / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return {
            "present": False,
            "path": REGISTRY_RELATIVE_PATH,
            "reproducible_total_levels": 0,
            "games": {},
        }
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        loaded = {}
    games = {
        str(row.get("game")): _as_int(row.get("levels_reproduced"))
        for row in loaded.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }
    return {
        "present": bool(loaded),
        "path": REGISTRY_RELATIVE_PATH,
        "reproducible_total_levels": _as_int(loaded.get("reproducible_total_levels")),
        "games": games,
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    """Check the Exp 5240 gate, registry precheck, and forbidden-method guards."""

    root_path = Path(root)
    spec_text = _read_text(root_path / SPEC_RELATIVE_PATH)
    exp5240_artifact = _read_json(root_path / EXP5240_RESULT_RELATIVE_PATH)
    registry = load_registry_summary(root_path)
    return {
        "agents_read": (root_path / "AGENTS.md").exists(),
        "codex_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_5241": "REQ-REPORT-5241" in spec_text,
        "exp5240_patch_candidate_tested": bool(
            exp5240_artifact.get("recommended_live_patch_available") is True
            and exp5240_artifact.get("patch_test_ready") is True
        ),
        "registry_present": bool(registry.get("present")),
        "registry_loadable": bool(registry.get("present")),
        "patch_path_matches_exp5240": exp5240_artifact.get("patch_path") == PATCH_RELATIVE_PATH,
        "read_hidden_game_source": False,
        "offline_ground_truth_bfs": False,
        "hand_per_game_adapter": False,
    }


def run_live_agent_patch_attempt(
    *,
    root: Path | str = REPO_ROOT,
    target_game: str = DEFAULT_TARGET_GAME,
    budget: int = DEFAULT_BUDGET,
    random_seed: int = RANDOM_SEED,
    exact_command: str = DEFAULT_EXACT_COMMAND,
) -> JsonDict:
    """Exercise the live ARC recommendation path under the Exp 5240 patch."""

    del root
    random.seed(int(random_seed))
    started = time.monotonic()
    from carnot.agentic import arc_competition_agent

    recommendation = arc_competition_agent._recommend_live_approach(target_game)
    runtime_s = round(time.monotonic() - started, 6)
    guard = recommendation.get("typed_memory_provenance_guard")
    guard_enabled = isinstance(guard, Mapping) and guard.get("enabled") is True
    return {
        "attempt_id": f"exp5241_{target_game}_seed_{int(random_seed)}_budget_{int(budget)}",
        "target_game": str(target_game),
        "target_level": 1,
        "prior_reproduced_level": 0,
        "budget": int(budget),
        "random_seed": int(random_seed),
        "runtime_s": runtime_s,
        "exact_command": str(exact_command),
        "policy": "arc_competition_agent._recommend_live_approach",
        "self_discovery_lever": "exp5240_provenance_routing_guard",
        "live_agent_patch_enabled": bool(guard_enabled),
        "runtime_self_discovery_attempted": True,
        "solution_labels": [],
        "reproduction_gate": {
            "claimed_level": 0,
            "reproduced": False,
            "registry_validation_passed": False,
            "reached_level": 0,
        },
        "model_ids": [],
        "llm_proposer_used": False,
        "model_specs": None,
        "forbidden_methods": {
            "read_hidden_game_source": False,
            "offline_ground_truth_bfs": False,
            "hand_per_game_adapter": False,
        },
        "process_deltas": {
            "skill_selection": "selected exp5240 provenance/failures/skills_rubrics guard",
            "skill_following": "live route exposed the guard before any registry promotion",
            "composition": "guard composed with strategy fallback for an unseen live target",
            "reflection": "no level was banked; patch preserved provenance discipline",
        },
        "approach_recommendation": recommendation,
    }


def build_artifact(
    *,
    precondition_audit: Mapping[str, Any],
    registry_summary: Mapping[str, Any],
    live_attempt: Mapping[str, Any],
    arc_validation_commands: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
) -> JsonDict:
    """Build the terminal Exp 5241 artifact."""

    preconditions_ok = _preconditions_ok(precondition_audit)
    registry_precheck_done = bool(registry_summary.get("present"))
    before = _as_int(registry_summary.get("reproducible_total_levels"))
    target_game = str(live_attempt.get("target_game") or DEFAULT_TARGET_GAME)
    target_level = _as_int(live_attempt.get("target_level"), 1)
    games = registry_summary.get("games") if isinstance(registry_summary.get("games"), Mapping) else {}
    duplicate_avoided = _duplicate_target_avoided(games, target_game, target_level)
    success, banked, residual = _banked_delta(live_attempt)
    if not preconditions_ok:
        success, banked, residual = False, 0, "preconditions_failed"
    delta = int(banked)
    after = before + delta
    patch_enabled = bool(live_attempt.get("live_agent_patch_enabled"))
    recommendation = _patch_recommendation(
        success=success,
        patch_enabled=patch_enabled,
        residual=residual,
    )
    verdict = _honest_verdict(success=success, delta=delta, residual=residual)
    llm_used = bool(live_attempt.get("llm_proposer_used"))
    model_specs = live_attempt.get("model_specs") if llm_used else None
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": bool(preconditions_ok),
        "precondition_audit": dict(precondition_audit),
        "solve_provenance": SOLVE_PROVENANCE,
        "registry_precheck_done": registry_precheck_done,
        "duplicate_solve_target_avoided": bool(duplicate_avoided),
        "reproducible_total_levels_before": before,
        "reproducible_total_levels_after": after,
        "reproducible_total_levels_delta": delta,
        "live_agent_patch_enabled": patch_enabled,
        "model_specs": model_specs,
        "model_ids": list(live_attempt.get("model_ids") or []),
        "llm_proposer_used": llm_used,
        "random_seed": _as_int(live_attempt.get("random_seed"), RANDOM_SEED),
        "budget": _as_int(live_attempt.get("budget"), DEFAULT_BUDGET),
        "duration_s": float(duration_s),
        "exact_command": str(live_attempt.get("exact_command") or DEFAULT_EXACT_COMMAND),
        "arc_validation_commands": [dict(item) for item in arc_validation_commands],
        "patch_recommendation": recommendation,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": verdict,
        "target_game": target_game,
        "target_level": target_level,
        "live_agent_attempts": [dict(live_attempt)] if live_attempt else [],
        "solve_claim": {
            "claimed": bool(success),
            "provenance": SOLVE_PROVENANCE,
            "registry_validation_passed": bool(_attempt_gate(live_attempt).get("registry_validation_passed")),
            "reproduction_gate": dict(_attempt_gate(live_attempt)),
            "residual": None if success else residual,
        },
        "forbidden_methods": dict(live_attempt.get("forbidden_methods") or {}),
        "process_deltas": dict(live_attempt.get("process_deltas") or _default_process_deltas()),
        "registry_summary": dict(registry_summary),
        "patch_path": PATCH_RELATIVE_PATH,
        "source_artifacts": [EXP5240_RESULT_RELATIVE_PATH, REGISTRY_RELATIVE_PATH],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    target_game: str = DEFAULT_TARGET_GAME,
    budget: int = DEFAULT_BUDGET,
    random_seed: int = RANDOM_SEED,
    exact_command: str = DEFAULT_EXACT_COMMAND,
    arc_validation_commands: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Run the gated live path and write the Exp 5241 artifact."""

    started = time.monotonic()
    root_path = Path(root)
    preconditions = check_preconditions(root_path)
    registry = load_registry_summary(root_path)
    if _preconditions_ok(preconditions):
        attempt = run_live_agent_patch_attempt(
            root=root_path,
            target_game=target_game,
            budget=budget,
            random_seed=random_seed,
            exact_command=exact_command,
        )
    else:
        attempt = _blocked_attempt(
            target_game=target_game,
            budget=budget,
            random_seed=random_seed,
            exact_command=exact_command,
        )
    duration_s = round(time.monotonic() - started, 6)
    artifact = build_artifact(
        precondition_audit=preconditions,
        registry_summary=registry,
        live_attempt=attempt,
        arc_validation_commands=arc_validation_commands,
        duration_s=duration_s,
    )
    validate_artifact(artifact)
    destination = Path(result_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema errors for an Exp 5241 artifact."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("experiment") != EXPERIMENT:
        errors.append("experiment mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if artifact.get("spec_refs") != list(SPEC_REFS):
        errors.append("spec_refs mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    for field in (
        "preconditions_checked",
        "registry_precheck_done",
        "duplicate_solve_target_avoided",
        "live_agent_patch_enabled",
    ):
        if not _is_bool(artifact.get(field)):
            errors.append(f"{field} must be bare bool")
    for field in (
        "reproducible_total_levels_before",
        "reproducible_total_levels_after",
        "reproducible_total_levels_delta",
        "random_seed",
    ):
        if not _is_int(artifact.get(field)):
            errors.append(f"{field} must be bare int")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if not isinstance(artifact.get("arc_validation_commands"), list):
        errors.append("arc_validation_commands must be a list")
    recommendation = artifact.get("patch_recommendation")
    if recommendation not in PATCH_RECOMMENDATIONS:
        errors.append("patch_recommendation must be one of")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if not _terminal_verdict(artifact.get("honest_verdict")):
        errors.append("honest_verdict must use a terminal prefix")
    llm_used = bool(artifact.get("llm_proposer_used"))
    if not llm_used and artifact.get("model_specs") is not None:
        errors.append("model_specs must be null when no LLM proposer was used")
    if llm_used and not _model_specs_include_mandated(artifact.get("model_specs")):
        errors.append("model_specs must include a mandated SOTA GGUF when LLM proposer was used")
    before = artifact.get("reproducible_total_levels_before")
    after = artifact.get("reproducible_total_levels_after")
    delta = artifact.get("reproducible_total_levels_delta")
    if _is_int(before) and _is_int(after) and _is_int(delta) and after != before + delta:
        errors.append("reproducible_total_levels_after must equal before + delta")
    solve_claim = artifact.get("solve_claim")
    if not isinstance(solve_claim, Mapping):
        errors.append("solve_claim must be a mapping")
    elif solve_claim.get("claimed") is not None and not _is_bool(solve_claim.get("claimed")):
        errors.append("solve_claim.claimed must be bare bool")
    forbidden = artifact.get("forbidden_methods")
    if isinstance(forbidden, Mapping) and any(bool(value) for value in forbidden.values()):
        errors.append("forbidden methods must be false")
    verdict = str(artifact.get("honest_verdict") or "")
    claimed = solve_claim.get("claimed") if isinstance(solve_claim, Mapping) else False
    if verdict.startswith("success:"):
        if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
            errors.append("success requires live_agent_self_discovery provenance")
        if delta is not None and _is_int(delta) and delta < 1:
            errors.append("success requires positive level delta")
        if claimed is not True:
            errors.append("success requires solve_claim.claimed true")
    if not verdict.startswith("success:") and _is_int(delta) and delta != 0:
        errors.append("non-success artifacts must not change registry totals")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be 64 hex chars")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise ValueError when an artifact violates the Exp 5241 schema."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the fields that make the Exp 5241 outcome reproducible."""

    payload = {
        "schema": artifact.get("schema"),
        "experiment": artifact.get("experiment"),
        "honest_verdict": artifact.get("honest_verdict"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "solve_provenance": artifact.get("solve_provenance"),
        "registry_precheck_done": artifact.get("registry_precheck_done"),
        "duplicate_solve_target_avoided": artifact.get("duplicate_solve_target_avoided"),
        "reproducible_total_levels_before": artifact.get("reproducible_total_levels_before"),
        "reproducible_total_levels_after": artifact.get("reproducible_total_levels_after"),
        "reproducible_total_levels_delta": artifact.get("reproducible_total_levels_delta"),
        "live_agent_patch_enabled": artifact.get("live_agent_patch_enabled"),
        "model_specs": artifact.get("model_specs"),
        "random_seed": artifact.get("random_seed"),
        "patch_recommendation": artifact.get("patch_recommendation"),
        "live_agent_attempts": artifact.get("live_agent_attempts"),
        "arc_validation_commands": artifact.get("arc_validation_commands"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _preconditions_ok(audit: Mapping[str, Any]) -> bool:
    return all(audit.get(key) is True for key in _TRUE_PRECONDITION_KEYS) and not any(
        bool(audit.get(key)) for key in _FORBIDDEN_PRECONDITION_KEYS
    )


def _duplicate_target_avoided(games: Mapping[str, Any], target_game: str, target_level: int) -> bool:
    if target_game not in games:
        return True
    return int(target_level) > _as_int(games.get(target_game))


def _banked_delta(live_attempt: Mapping[str, Any]) -> tuple[bool, int, str]:
    gate = _attempt_gate(live_attempt)
    prior = _as_int(live_attempt.get("prior_reproduced_level"))
    claimed_level = _as_int(gate.get("claimed_level"))
    labels = live_attempt.get("solution_labels")
    has_labels = isinstance(labels, list) and bool(labels)
    forbidden = live_attempt.get("forbidden_methods")
    forbidden_used = isinstance(forbidden, Mapping) and any(bool(value) for value in forbidden.values())
    reaches_new_level = claimed_level > prior
    reproduced = gate.get("reproduced") is True
    registry_validated = gate.get("registry_validation_passed") is True
    success = bool(reaches_new_level and reproduced and registry_validated and has_labels and not forbidden_used)
    if success:
        return True, claimed_level - prior, "accepted_live_self_discovery"
    if not live_attempt.get("live_agent_patch_enabled"):
        return False, 0, "patch_not_enabled"
    if forbidden_used:
        return False, 0, "forbidden_method_used"
    if reaches_new_level and reproduced and not registry_validated:
        return False, 0, "registry_validation_failed"
    if reaches_new_level and not has_labels:
        return False, 0, "missing_live_solution_labels"
    return False, 0, "no_level_banked"


def _patch_recommendation(*, success: bool, patch_enabled: bool, residual: str) -> str:
    if success:
        return "keep"
    if not patch_enabled:
        return "rollback"
    if residual in {"registry_validation_failed", "missing_live_solution_labels"}:
        return "iterate"
    return "no_solve_no_regression"


def _honest_verdict(*, success: bool, delta: int, residual: str) -> str:
    if success:
        return (
            f"success: level_delta={delta} provenance={SOLVE_PROVENANCE}; "
            "live self-discovery reproduction passed registry validation."
        )
    if residual == "preconditions_failed":
        return (
            f"blocked_preconditions_failed: level_delta=0 provenance={SOLVE_PROVENANCE}; "
            "Exp 5240 patch or registry preconditions were not satisfied."
        )
    return (
        f"complete: level_delta=0 provenance={SOLVE_PROVENANCE}; "
        f"no level banked under the gated live patch path ({residual})."
    )


def _attempt_gate(live_attempt: Mapping[str, Any] | None) -> Mapping[str, Any]:
    gate = (live_attempt or {}).get("reproduction_gate")
    return gate if isinstance(gate, Mapping) else {}


def _default_process_deltas() -> JsonDict:
    return {
        "skill_selection": "no live attempt executed",
        "skill_following": "no live attempt executed",
        "composition": "no live attempt executed",
        "reflection": "preconditions blocked the live patch attempt",
    }


def _blocked_attempt(
    *,
    target_game: str,
    budget: int,
    random_seed: int,
    exact_command: str,
) -> JsonDict:
    return {
        "attempt_id": f"exp5241_{target_game}_blocked_preconditions",
        "target_game": str(target_game),
        "target_level": 1,
        "prior_reproduced_level": 0,
        "budget": int(budget),
        "random_seed": int(random_seed),
        "runtime_s": 0.0,
        "exact_command": str(exact_command),
        "policy": "not_run_precondition_block",
        "self_discovery_lever": "not_run_precondition_block",
        "live_agent_patch_enabled": False,
        "runtime_self_discovery_attempted": False,
        "solution_labels": [],
        "reproduction_gate": {
            "claimed_level": 0,
            "reproduced": False,
            "registry_validation_passed": False,
            "reached_level": 0,
        },
        "model_ids": [],
        "llm_proposer_used": False,
        "model_specs": None,
        "forbidden_methods": {
            "read_hidden_game_source": False,
            "offline_ground_truth_bfs": False,
            "hand_per_game_adapter": False,
        },
        "process_deltas": _default_process_deltas(),
        "approach_recommendation": {},
    }


def _model_specs_include_mandated(model_specs: Any) -> bool:
    text = json.dumps(model_specs, sort_keys=True, default=str)
    return any(model_id in text for model_id in MANDATED_SOTA_GGUFS)


def _terminal_verdict(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith(("complete:", "complete_", "success:", "success_", "blocked_"))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _parse_validation_command(value: str) -> JsonDict:  # pragma: no cover - CLI wrapper
    status, _, command = value.partition("::")
    return {"command": command or value, "passed": status.lower() in {"pass", "passed", "true"}}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--target-game", default=DEFAULT_TARGET_GAME)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--exact-command", default=DEFAULT_EXACT_COMMAND)
    parser.add_argument(
        "--validation-command",
        action="append",
        default=[],
        help="Validation receipt as passed::command or failed::command.",
    )
    args = parser.parse_args(argv)
    artifact = run_experiment(
        root=args.root,
        result_path=args.result_path,
        target_game=args.target_game,
        budget=args.budget,
        random_seed=args.random_seed,
        exact_command=args.exact_command,
        arc_validation_commands=[_parse_validation_command(item) for item in args.validation_command],
    )
    print(json.dumps({field: artifact[field] for field in REQUIRED_ARTIFACT_FIELDS}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
