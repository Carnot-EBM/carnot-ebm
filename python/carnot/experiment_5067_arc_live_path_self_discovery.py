"""Experiment 5067: ARC live-path self-discovery artifact.

Spec refs: REQ-ARC-WMTE-5067,
SCENARIO-ARC-WMTE-5067-REGISTRY-PRIOR-PRECHECK,
SCENARIO-ARC-WMTE-5067-PROVENANCE-GATE,
SCENARIO-ARC-WMTE-5067-STABLE-ARTIFACT.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO / "python"))

from carnot import experiment_5054_arc_live_path_self_discovery as exp5054


EXPERIMENT = "experiment_5067_arc_live_path_self_discovery"
EXPERIMENT_ID = 5067
SCHEMA = "carnot.arc_live_path_self_discovery_5067.v1"
RESULT_RELATIVE_PATH = "results/experiment_5067_arc_live_path_self_discovery.json"
REGISTRY_RELATIVE_PATH = exp5054.REGISTRY_RELATIVE_PATH
SPEC_RELATIVE_PATH = exp5054.SPEC_RELATIVE_PATH
RANDOM_SEED = 5067
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = exp5054.INFERENCE_SUBSTRATE
CURRENT_TARGET = ("lp85", 6)
DEFAULT_CANDIDATE_GAMES = (
    "lp85",
    "tu93",
    "s5i5",
    "bp35",
    "re86",
    "sb26",
    "lf52",
    "g50t",
    "cn04",
    "sc25",
)
HIDDEN_STATE_TARGETS = ("ka59", "wa30", "ar25")
DEFAULT_BUDGET = 36
PRIOR_LIVE_PATH_ARTIFACTS = ("results/experiment_5054_arc_live_path_self_discovery.json",)
MODEL_SPECS: dict[str, Any] = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
    "live_reasoning_invoked": False,
    "reasoning_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
}
LEGACY_MODELS_SMOKE_ONLY = True
SPEC_REFS = [
    "REQ-ARC-WMTE-5067",
    "SCENARIO-ARC-WMTE-5067-REGISTRY-PRIOR-PRECHECK",
    "SCENARIO-ARC-WMTE-5067-PROVENANCE-GATE",
    "SCENARIO-ARC-WMTE-5067-STABLE-ARTIFACT",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success_<game>_levelup_banked only for a strict "
            "live-agent reproduced level, complete_<game>_no_new_level_residual_* "
            "for honest no-bank, blocked_* for failed preconditions."
        )
    },
    "registry_precheck_passed": {
        "principle": (
            "bare bool confirming the registry/prior-artifact duplicate guard ran before "
            "target selection."
        )
    },
    "target_game": {
        "principle": (
            "selected unsolved next-level target after registry, prior-artifact, duplicate, "
            "and dead-end precheck."
        )
    },
    "target_level": {"principle": "the next level attempted for the selected target game."},
    "prior_reproduced_level": {"principle": "the registry depth before this live-path attempt."},
    "new_levels_banked": {
        "principle": (
            "bare int; increments only for strict reproduction-gated progress beyond registry depth."
        )
    },
    "duplicate_solve_avoided": {
        "principle": (
            "bare bool; true when duplicate, prior no-bank, or same-depth outcomes are not credited."
        )
    },
    "solve_claim": {
        "principle": (
            "bare claim object; claimed=true only when the live-agent trace reproduces beyond "
            "registry depth."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery for live-path attempts; never outer_loop_re for a headline bank."
        )
    },
    "provenance_evidence": {
        "principle": (
            "auditable runtime path showing no hidden source reading, no offline ground-truth BFS, "
            "no hand-built adapter, and whether an LLM was invoked."
        )
    },
    "reproducible_total_levels_before": {
        "principle": "registry reproducible_total_levels before the attempt."
    },
    "reproducible_total_levels_after": {
        "principle": "before + new_levels_banked; unchanged for no-bank and blocked artifacts."
    },
    "model_specs": {
        "principle": (
            "records the mandated flagship_moe, flagship_dense, and middle_moe GGUF specs; if no "
            "LLM is invoked, records that fact explicitly."
        )
    },
    "legacy_models_smoke_only": {
        "principle": (
            "bare bool; legacy small models are smoke-only and cannot provide live reasoning credit."
        )
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of target selection, provenance, live attempts, model specs, and bank summary."
        )
    },
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "field_principles",
    "live_agent_attempts",
    "offline_reproduced",
    "inference_substrate",
    "preconditions_checked",
    "candidate_selection",
    "random_seed",
    "duration_s",
)


def _load_registry_text(root: Path) -> str:
    return (root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")


def _load_registry(registry_text: str) -> dict[str, Any]:
    loaded = yaml.safe_load(registry_text)
    return loaded if isinstance(loaded, dict) else {}


def _game_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("game")): row
        for row in registry.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }


def _registry_total(registry: Mapping[str, Any]) -> int:
    return int(registry.get("reproducible_total_levels") or 0)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - malformed optional evidence
        return default


def _prior_live_path_attempts(
    root: Path,
    artifact_relpaths: Sequence[str] = PRIOR_LIVE_PATH_ARTIFACTS,
) -> dict[tuple[str, int], dict[str, Any]]:
    attempts: dict[tuple[str, int], dict[str, Any]] = {}
    for relpath in artifact_relpaths:
        path = root / relpath
        if not path.exists():  # pragma: no cover - optional historical evidence
            continue
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - optional historical evidence
            continue
        if not isinstance(artifact, Mapping):
            continue
        game = str(artifact.get("target_game") or "")
        target_level = _as_int(artifact.get("target_level"))
        if not game or target_level <= 0:
            continue
        live_attempts = artifact.get("live_agent_attempts")
        max_reached = 0
        if isinstance(live_attempts, list):
            max_reached = max(
                (
                    _as_int(item.get("max_level_reached"))
                    for item in live_attempts
                    if isinstance(item, Mapping)
                ),
                default=0,
            )
        new_banked = _as_int(artifact.get("new_levels_banked"))
        if new_banked > 0 and max_reached < target_level:
            continue
        attempts[(game, target_level)] = {
            "artifact": relpath,
            "honest_verdict": artifact.get("honest_verdict"),
            "max_level_reached": max_reached,
            "new_levels_banked": new_banked,
            "status": (
                "prior_live_path_reached_target"
                if max_reached >= target_level
                else "prior_live_path_no_bank"
            ),
        }
    return attempts


def _candidate_audit_row(
    game: str,
    row: Mapping[str, Any] | None,
    *,
    status: str,
    reason: str,
    target_level: int = 0,
    prior_attempt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = exp5054._candidate_audit_row(
        game,
        row,
        status=status,
        reason=reason,
        target_level=target_level,
    )
    if prior_attempt is not None:
        audit["prior_live_path_attempt"] = dict(prior_attempt)
    return audit


def select_target(
    registry: Mapping[str, Any],
    *,
    root: Path = REPO,
    current_target: tuple[str, int] = CURRENT_TARGET,
    candidate_games: Sequence[str] = DEFAULT_CANDIDATE_GAMES,
    hidden_state_targets: Sequence[str] = HIDDEN_STATE_TARGETS,
    prior_artifact_relpaths: Sequence[str] = PRIOR_LIVE_PATH_ARTIFACTS,
) -> dict[str, Any]:
    rows = _game_rows(registry)
    current_game, current_level = str(current_target[0]), int(current_target[1])
    hidden = {str(game) for game in hidden_state_targets}
    prior_attempts = _prior_live_path_attempts(Path(root), prior_artifact_relpaths)
    audit: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    duplicate_current: dict[str, Any] | None = None

    for raw_game in candidate_games:
        game = str(raw_game)
        row = rows.get(game)
        if row is None:
            audit.append(
                _candidate_audit_row(
                    game,
                    None,
                    status="skip_missing_registry_row",
                    reason="missing_registry_row",
                )
            )
            continue
        prior = _as_int(row.get("levels_reproduced"))
        target_level = prior + 1
        if game in hidden:
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status="skip_hidden_state_target",
                    reason="hidden_state_bound_target",
                    target_level=target_level,
                )
            )
            continue
        if game == current_game and current_level <= prior:
            duplicate_current = {
                "game": game,
                "prior_reproduced_level": prior,
                "target_level": current_level,
            }
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status="skip_duplicate_current_target",
                    reason="current_target_already_reproduced",
                    target_level=current_level,
                )
            )
            continue
        if exp5054._has_next_level_dry_dead_end(row, game, target_level):
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status="skip_recorded_dry_next_level",
                    reason=f"recorded_l{target_level}_dry_dead_end",
                    target_level=target_level,
                )
            )
            continue
        prior_attempt = prior_attempts.get((game, target_level))
        if prior_attempt is not None:
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status=str(prior_attempt["status"]).replace(
                        "prior_live_path", "skip_prior_live_path"
                    ),
                    reason=str(prior_attempt["status"]),
                    target_level=target_level,
                    prior_attempt=prior_attempt,
                )
            )
            continue
        if selected is None:
            selected = {
                "game": game,
                "prior_reproduced_level": prior,
                "target_level": target_level,
                "status": "selected",
                "reason": "unsolved_next_level_live_path_target",
            }
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status="candidate_selected",
                    reason="unsolved_next_level_live_path_target",
                    target_level=target_level,
                )
            )
        else:
            audit.append(
                _candidate_audit_row(
                    game,
                    row,
                    status="alternate_not_selected",
                    reason="higher_priority_candidate_selected",
                    target_level=target_level,
                )
            )

    base = {
        "candidate_audit": audit,
        "candidate_order": [str(game) for game in candidate_games],
        "current_target": {"game": current_game, "target_level": current_level},
        "hidden_state_targets_avoided": sorted(hidden),
        "prior_live_path_artifacts_consulted": [str(path) for path in prior_artifact_relpaths],
        "duplicate_solve_avoided": True,
    }
    if selected is not None:
        return {**selected, "registry_precheck_passed": True, **base}
    if duplicate_current is not None:
        return {
            **duplicate_current,
            "status": "blocked_duplicate_target",
            "reason": "current_target_already_reproduced_and_no_rotation_available",
            "registry_precheck_passed": False,
            **base,
        }
    return {
        "game": "none",
        "prior_reproduced_level": 0,
        "target_level": 0,
        "status": "blocked_no_unsolved_target",
        "reason": "no_unsolved_live_path_target_available",
        "registry_precheck_passed": False,
        **base,
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        "schema": artifact.get("schema"),
        "experiment": artifact.get("experiment"),
        "honest_verdict": artifact.get("honest_verdict"),
        "registry_precheck_passed": artifact.get("registry_precheck_passed"),
        "target_game": artifact.get("target_game"),
        "target_level": artifact.get("target_level"),
        "prior_reproduced_level": artifact.get("prior_reproduced_level"),
        "new_levels_banked": artifact.get("new_levels_banked"),
        "duplicate_solve_avoided": artifact.get("duplicate_solve_avoided"),
        "solve_claim": artifact.get("solve_claim"),
        "solve_provenance": artifact.get("solve_provenance"),
        "provenance_evidence": artifact.get("provenance_evidence"),
        "reproducible_total_levels_before": artifact.get("reproducible_total_levels_before"),
        "reproducible_total_levels_after": artifact.get("reproducible_total_levels_after"),
        "model_specs": artifact.get("model_specs"),
        "legacy_models_smoke_only": artifact.get("legacy_models_smoke_only"),
        "live_agent_attempts": artifact.get("live_agent_attempts"),
        "candidate_selection": artifact.get("candidate_selection"),
        "random_seed": artifact.get("random_seed"),
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _attempt_gate(live_attempt: Mapping[str, Any] | None) -> Mapping[str, Any]:
    gate = (live_attempt or {}).get("reproduction_gate")
    return gate if isinstance(gate, Mapping) else {}


def _bank_summary(
    *,
    prior_level: int,
    live_attempt: Mapping[str, Any],
) -> tuple[bool, int, str]:
    return exp5054._bank_summary(prior_level=prior_level, live_attempt=live_attempt)


def _provenance_evidence(
    *,
    success: bool,
    live_attempt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attempt = dict(live_attempt or {})
    return {
        "attempted_path": "E3AgentPolicy/bounded_live_policy"
        if attempt
        else "not_run_precondition_block",
        "runtime_self_discovery": bool(attempt.get("runtime_self_discovery")),
        "policy": attempt.get("policy"),
        "self_discovery_lever": attempt.get("self_discovery_lever"),
        "solution_labels_from_live_run": bool(attempt.get("solution_labels")) and success,
        "offline_source_reading_used": bool(attempt.get("offline_source_reading_used")),
        "offline_ground_truth_bfs_used": bool(
            attempt.get("offline_ground_truth_bfs_used") or attempt.get("per_game_bfs_used")
        ),
        "hand_built_adapter_used": bool(attempt.get("hand_built_adapter_used")),
        "llm_reasoning_invoked": bool(attempt.get("llm_reasoning_invoked")),
        "model_specs_obeyed": dict(attempt.get("model_specs") or MODEL_SPECS) == MODEL_SPECS,
        "legacy_models_smoke_only": attempt.get(
            "legacy_models_smoke_only", LEGACY_MODELS_SMOKE_ONLY
        )
        is True,
    }


def _solve_claim(
    *,
    success: bool,
    residual: str,
    live_attempt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "claimed": bool(success),
        "provenance": SOLVE_PROVENANCE,
        "attempted_path": "live_agent_self_discovery",
        "residual": None if success else residual,
        "reproduction_gate": dict(_attempt_gate(live_attempt)),
    }


def build_artifact(
    *,
    selection: Mapping[str, Any],
    registry_total: int,
    live_attempt: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    game = str(selection.get("game") or "none")
    prior_level = _as_int(selection.get("prior_reproduced_level"))
    target_level = _as_int(selection.get("target_level"), prior_level + 1)
    artifact_attempt = dict(live_attempt)
    go_diagnostics = artifact_attempt.pop("go_explore_archive", None)
    if isinstance(go_diagnostics, Mapping):
        actions_injected = _as_int(go_diagnostics.get("actions_injected"))
        prefixes_injected = _as_int(go_diagnostics.get("prefixes_injected"))
        artifact_attempt["live_path_diagnostics"] = {
            "policy_observations": _as_int(go_diagnostics.get("observations")),
            "policy_stored_cells": _as_int(go_diagnostics.get("stored_cells")),
            "injection_exercised": actions_injected > 0 or prefixes_injected > 0,
        }
        if artifact_attempt["live_path_diagnostics"]["injection_exercised"] is False:
            artifact_attempt["self_discovery_lever"] = "bounded_e3_policy_no_archive_injection"
    success, banked, residual = _bank_summary(
        prior_level=prior_level,
        live_attempt=artifact_attempt,
    )
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": (
            f"success_{game}_levelup_banked"
            if success
            else f"complete_{game}_no_new_level_residual_{residual}"
        ),
        "registry_precheck_passed": bool(selection.get("registry_precheck_passed")),
        "target_game": game,
        "target_level": target_level,
        "prior_reproduced_level": prior_level,
        "new_levels_banked": int(banked),
        "duplicate_solve_avoided": not success,
        "solve_claim": _solve_claim(
            success=success,
            residual=residual,
            live_attempt=artifact_attempt,
        ),
        "solve_provenance": SOLVE_PROVENANCE,
        "provenance_evidence": _provenance_evidence(
            success=success,
            live_attempt=artifact_attempt,
        ),
        "reproducible_total_levels_before": int(registry_total),
        "reproducible_total_levels_after": int(registry_total) + int(banked),
        "model_specs": dict(MODEL_SPECS),
        "legacy_models_smoke_only": LEGACY_MODELS_SMOKE_ONLY,
        "live_agent_attempts": [artifact_attempt],
        "offline_reproduced": bool(success),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "candidate_selection": dict(selection),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def blocked_artifact(
    *,
    reason: str,
    selection: Mapping[str, Any],
    registry_total: int,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    game = str(selection.get("game") or "none")
    prior_level = _as_int(selection.get("prior_reproduced_level"))
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": f"blocked_{reason}",
        "registry_precheck_passed": False,
        "target_game": game,
        "target_level": _as_int(selection.get("target_level")),
        "prior_reproduced_level": prior_level,
        "new_levels_banked": 0,
        "duplicate_solve_avoided": True,
        "solve_claim": _solve_claim(success=False, residual=reason, live_attempt=None),
        "solve_provenance": SOLVE_PROVENANCE,
        "provenance_evidence": _provenance_evidence(success=False, live_attempt=None),
        "reproducible_total_levels_before": int(registry_total),
        "reproducible_total_levels_after": int(registry_total),
        "model_specs": dict(MODEL_SPECS),
        "legacy_models_smoke_only": LEGACY_MODELS_SMOKE_ONLY,
        "live_agent_attempts": [],
        "offline_reproduced": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "candidate_selection": dict(selection),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("experiment") != EXPERIMENT:
        errors.append("experiment mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs mismatch")
    if artifact.get("result_path") != RESULT_RELATIVE_PATH:
        errors.append("result_path mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    for field in (
        "registry_precheck_passed",
        "duplicate_solve_avoided",
        "offline_reproduced",
        "legacy_models_smoke_only",
    ):
        if not _is_bool(artifact.get(field)):
            errors.append(f"{field} must be bare bool")
    for field in (
        "target_level",
        "prior_reproduced_level",
        "new_levels_banked",
        "reproducible_total_levels_before",
        "reproducible_total_levels_after",
    ):
        if not _is_int(artifact.get(field)):
            errors.append(f"{field} must be bare int")
    if not isinstance(artifact.get("solve_claim"), Mapping):
        errors.append("solve_claim must be a mapping")
    if not isinstance(artifact.get("provenance_evidence"), Mapping):
        errors.append("provenance_evidence must be a mapping")
    if artifact.get("model_specs") != MODEL_SPECS:
        errors.append("model_specs mismatch")
    if artifact.get("legacy_models_smoke_only") is not True:
        errors.append("legacy_models_smoke_only must be true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if not isinstance(artifact.get("candidate_selection"), Mapping):
        errors.append("candidate_selection must be a mapping")
    if not isinstance(artifact.get("live_agent_attempts"), list):
        errors.append("live_agent_attempts must be a list")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (
        verdict.startswith("success_")
        or verdict.startswith("complete_")
        or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict must use a terminal prefix")
    before = artifact.get("reproducible_total_levels_before")
    after = artifact.get("reproducible_total_levels_after")
    banked = artifact.get("new_levels_banked")
    if _is_int(before) and _is_int(after) and _is_int(banked) and after != before + banked:
        errors.append("reproducible_total_levels_after must equal before + new_levels_banked")

    claim = artifact.get("solve_claim")
    evidence = artifact.get("provenance_evidence")
    claimed = claim.get("claimed") if isinstance(claim, Mapping) else None
    if claimed is not None and not _is_bool(claimed):
        errors.append("solve_claim.claimed must be bare bool")
    forbidden = False
    if isinstance(evidence, Mapping):
        forbidden = bool(
            evidence.get("offline_source_reading_used")
            or evidence.get("offline_ground_truth_bfs_used")
            or evidence.get("hand_built_adapter_used")
        )
    if verdict.startswith("success_"):
        if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
            errors.append("success requires live_agent_self_discovery provenance")
        if artifact.get("offline_reproduced") is not True:
            errors.append("success requires offline_reproduced true")
        if _is_int(banked) and banked < 1:
            errors.append("success requires new_levels_banked >= 1")
        if claimed is not True:
            errors.append("success requires solve_claim.claimed true")
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("solution_labels_from_live_run") is not True
        ):
            errors.append("success requires live-agent solution label evidence")
        if forbidden:
            errors.append("success cannot use hidden source, offline BFS, or hand adapter")
    elif artifact.get("offline_reproduced") is True:
        errors.append("non-success cannot set offline_reproduced true")
    if _is_int(banked) and banked == 0 and artifact.get("duplicate_solve_avoided") is not True:
        errors.append("no-bank artifacts must set duplicate_solve_avoided true")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        errors.append("reproducibility_checksum must be 64 hex chars")
    elif checksum != reproducibility_checksum(artifact):
        errors.append("checksum mismatch")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def offline_arcade_available() -> bool:  # pragma: no cover - environment probe
    return exp5054.offline_arcade_available()


def run_live_agent_attempt(
    *,
    root: Path,
    selection: Mapping[str, Any],
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:  # pragma: no cover - ARC runtime boundary
    attempt = dict(exp5054.run_live_agent_attempt(root=root, selection=selection, budget=budget))
    attempt["model_specs"] = dict(MODEL_SPECS)
    attempt["legacy_models_smoke_only"] = LEGACY_MODELS_SMOKE_ONLY
    attempt["offline_ground_truth_bfs_used"] = bool(attempt.get("per_game_bfs_used"))
    return attempt


def _write_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocked_and_write(
    *,
    root: Path,
    reason: str,
    selection: Mapping[str, Any],
    registry_total: int,
    preconditions: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    artifact = blocked_artifact(
        reason=reason,
        selection=selection,
        registry_total=registry_total,
        preconditions_checked=preconditions,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact)
    _write_artifact(root, artifact)
    return artifact


def run_experiment(
    *,
    root: Path = REPO,
    current_target: tuple[str, int] = CURRENT_TARGET,
    candidate_games: Sequence[str] = DEFAULT_CANDIDATE_GAMES,
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root)
    spec_path = root / SPEC_RELATIVE_PATH
    registry_path = root / REGISTRY_RELATIVE_PATH
    empty_selection = {
        "game": "none",
        "prior_reproduced_level": 0,
        "target_level": 0,
        "registry_precheck_passed": False,
        "candidate_audit": [],
    }
    preconditions: dict[str, Any] = {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "arc_world_model_trust_energy_spec_has_req_5067": (
            "REQ-ARC-WMTE-5067" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
        "registry_present": registry_path.exists(),
        "registry_loadable": False,
        "prior_live_path_artifacts_consulted": list(PRIOR_LIVE_PATH_ARTIFACTS),
        "offline_arcade_available": False,
        "llm_reasoning_invoked": False,
        "offline_source_reading_used": False,
        "offline_ground_truth_bfs_used": False,
        "per_game_bfs_used": False,
        "hand_built_adapter_used": False,
    }
    if not preconditions["arc_world_model_trust_energy_spec_has_req_5067"]:
        return _blocked_and_write(
            root=root,
            reason="spec_missing",
            selection=empty_selection,
            registry_total=0,
            preconditions=preconditions,
            started=started,
        )
    try:
        registry_text = _load_registry_text(root)
        registry = _load_registry(registry_text)
        preconditions["registry_loadable"] = bool(registry)
    except (OSError, yaml.YAMLError):
        return _blocked_and_write(
            root=root,
            reason="arc_solve_registry_unreadable",
            selection=empty_selection,
            registry_total=0,
            preconditions=preconditions,
            started=started,
        )
    if not preconditions["registry_loadable"]:
        return _blocked_and_write(
            root=root,
            reason="arc_solve_registry_unreadable",
            selection=empty_selection,
            registry_total=0,
            preconditions=preconditions,
            started=started,
        )

    registry_total = _registry_total(registry)
    selection = select_target(
        registry,
        root=root,
        current_target=current_target,
        candidate_games=candidate_games,
    )
    if str(selection.get("status") or "").startswith("blocked_"):
        reason = str(selection.get("status")).removeprefix("blocked_")
        return _blocked_and_write(
            root=root,
            reason=reason,
            selection=selection,
            registry_total=registry_total,
            preconditions=preconditions,
            started=started,
        )
    preconditions["offline_arcade_available"] = offline_arcade_available()
    if not preconditions["offline_arcade_available"]:
        return _blocked_and_write(
            root=root,
            reason="offline_arcade_missing",
            selection=selection,
            registry_total=registry_total,
            preconditions=preconditions,
            started=started,
        )
    live_attempt = run_live_agent_attempt(root=root, selection=selection, budget=budget)
    artifact = build_artifact(
        selection=selection,
        registry_total=registry_total,
        live_attempt=live_attempt,
        preconditions_checked=preconditions,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact)
    _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
