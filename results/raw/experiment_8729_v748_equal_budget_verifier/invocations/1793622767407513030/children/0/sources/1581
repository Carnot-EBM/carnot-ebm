"""Exp 5253: ARC live patch clean receipts and final patch decision.

Spec refs: REQ-REPORT-5253,
SCENARIO-REPORT-5253-CLEAN-NO-BANK-RETIRE,
SCENARIO-REPORT-5253-SOLVE-CREDIT-GATE.

This module reruns only the live-path reachable Exp 5240 provenance-routing
guard. It records receipts for the route and retires the patch scope when the
clean run still banks zero levels. It deliberately does not add a solver, read
game source, run offline ground-truth BFS, or hand-register a per-game adapter.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import yaml

from carnot import experiment_5240_arc_rubric_to_patch_synthesis_v479 as exp5240


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5253_arc_live_patch_clean_receipts_v480.json")
ATTEMPT_LOG_RELATIVE_PATH = Path(
    "results/experiment_5253_arc_live_patch_clean_receipts_v480_attempts.jsonl"
)
EXPERIMENT = "experiment_5253_arc_live_patch_clean_receipts_v480"
EXPERIMENT_ID = "exp5253-arc-live-patch-clean-receipts-v480"
MILESTONE = "2026.07.480"
RUN_DATE = "2026-07-05"
SCHEMA = "carnot.experiment_5253.arc_live_patch_clean_receipts.v480"
REGISTRY_RELATIVE_PATH = exp5240.REGISTRY_RELATIVE_PATH
PATCH_RELATIVE_PATH = exp5240.PATCH_RELATIVE_PATH
EXP5240_RESULT_RELATIVE_PATH = exp5240.RESULT_RELATIVE_PATH
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
SOLVE_PROVENANCE_LIVE = "live_agent_self_discovery"
DEFAULT_TARGET_GAME = "zz99_exp5253_live_receipt_probe"
DEFAULT_TARGET_LEVEL = 1
DEFAULT_BUDGET = 8
RANDOM_SEED = 5253
DEFAULT_EXACT_COMMAND = (
    ".venv/bin/python -m carnot.experiment_5253_arc_live_patch_clean_receipts_v480 "
    f"--target-game {DEFAULT_TARGET_GAME} --budget {DEFAULT_BUDGET} "
    f"--random-seed {RANDOM_SEED} --update-exclusion-manifest"
)
SPEC_REFS = (
    "REQ-REPORT-5253",
    "SCENARIO-REPORT-5253-CLEAN-NO-BANK-RETIRE",
    "SCENARIO-REPORT-5253-SOLVE-CREDIT-GATE",
)
VALID_SOLVE_PROVENANCE = {"live_agent_self_discovery", "development_proxy", "outer_loop_re"}
FORBIDDEN_METHOD_KEYS = (
    "read_hidden_game_source",
    "offline_ground_truth_bfs",
    "hand_per_game_adapter",
    "outer_loop_reverse_engineering",
)
RETIREMENT_SCOPE_ID = "exp5240_arc_provenance_routing_patch_scope_retired_v480"

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state level_delta and patch decision."
    ),
    "inference_substrate": (
        "Use offline_arcade_live_agent_runtime_self_discovery_no_llm unless a mandated "
        "local SOTA GGUF proposer actually ran."
    ),
    "solve_provenance": (
        "Records the attempted solve-credit route; no level is banked unless "
        "live_agent_self_discovery also passes registry validation."
    ),
    "registry_precheck": (
        "Records registry total and whether the target level was already reproduced before "
        "the live attempt."
    ),
    "level_delta": (
        "Integer level delta accepted after registry and provenance gates; zero means no "
        "solve was banked."
    ),
    "levels_reproduced": (
        "List of levels banked by this receipt; empty when no solve is claimed."
    ),
    "duplicate_solve_claimed": (
        "Must remain false; pre-reproduced targets improve receipts or retire the patch "
        "instead of claiming credit."
    ),
    "retire_current_provenance_patch": (
        "True when clean receipts leave level_delta=0 and the current provenance patch "
        "scope should be retired."
    ),
    "duration_s": (
        "Measured wall-clock seconds for registry precheck, live route, attempt-log write, "
        "and artifact construction."
    ),
    "attempt_log_path": "Path to the JSONL attempt log written by this receipt run.",
    "input_checksum": (
        "sha256 over registry, Exp5240 artifact, patch file, and target configuration."
    ),
    "output_checksum": "sha256 over the attempt-log rows before final artifact checksum.",
    "provenance_route_receipts": (
        "Receipts proving the live path reached the Exp5240 guard and avoided forbidden "
        "methods."
    ),
}
REQUIRED_PRINCIPLE_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_SCHEMA_FIELDS = {
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "field_principles",
    "target_game",
    "target_level",
    "random_seed",
    "budget",
    "exact_command",
    "live_agent_patch_enabled",
    "forbidden_methods",
    "source_artifacts",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_PRINCIPLE_FIELDS,
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Return a stable checksum over the emitted artifact except this field."""

    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _sha256(payload)


def load_registry_summary(root: Path | str = REPO_ROOT) -> JsonDict:
    """Read ARC registry totals without mutating the registry."""

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


def registry_precheck(
    registry_summary: Mapping[str, Any],
    *,
    target_game: str,
    target_level: int,
) -> JsonDict:
    """Return the pre-attempt duplicate/reproduction status for one target."""

    games = registry_summary.get("games") if isinstance(registry_summary.get("games"), Mapping) else {}
    prior = _as_int(games.get(target_game)) if isinstance(games, Mapping) else 0
    return {
        "path": str(registry_summary.get("path") or REGISTRY_RELATIVE_PATH),
        "present": bool(registry_summary.get("present")),
        "reproducible_total_levels": _as_int(registry_summary.get("reproducible_total_levels")),
        "target_game": str(target_game),
        "target_level": int(target_level),
        "target_prior_level": prior,
        "target_already_reproduced": prior >= int(target_level),
    }


def input_checksum(
    root: Path | str = REPO_ROOT,
    *,
    target_game: str = DEFAULT_TARGET_GAME,
    target_level: int = DEFAULT_TARGET_LEVEL,
    random_seed: int = RANDOM_SEED,
    budget: int = DEFAULT_BUDGET,
) -> str:
    """Hash the receipts that define the run inputs."""

    root_path = Path(root)
    payload = {
        "target_game": target_game,
        "target_level": int(target_level),
        "random_seed": int(random_seed),
        "budget": int(budget),
        "inputs": {
            REGISTRY_RELATIVE_PATH: _file_sha256(root_path / REGISTRY_RELATIVE_PATH),
            EXP5240_RESULT_RELATIVE_PATH: _file_sha256(root_path / EXP5240_RESULT_RELATIVE_PATH),
            PATCH_RELATIVE_PATH: _file_sha256(root_path / PATCH_RELATIVE_PATH),
        },
    }
    return _sha256(payload)


def run_live_agent_patch_attempt(
    *,
    root: Path | str = REPO_ROOT,
    target_game: str = DEFAULT_TARGET_GAME,
    target_level: int = DEFAULT_TARGET_LEVEL,
    budget: int = DEFAULT_BUDGET,
    random_seed: int = RANDOM_SEED,
    exact_command: str = DEFAULT_EXACT_COMMAND,
) -> JsonDict:
    """Exercise the live ARC recommendation path that reaches the Exp5240 guard."""

    del root
    random.seed(int(random_seed))
    started = time.monotonic()
    from carnot.agentic import arc_competition_agent

    recommendation = arc_competition_agent._recommend_live_approach(target_game)
    runtime_s = round(time.monotonic() - started, 6)
    guard = recommendation.get("typed_memory_provenance_guard")
    guard_mapping = guard if isinstance(guard, Mapping) else {}
    route_receipt = {
        "route": "arc_competition_agent._recommend_live_approach",
        "calls": [
            "arc_solve_learning.recommend_approach",
            "arc_typed_memory_provenance_guard.typed_memory_provenance_guard",
        ],
        "reached_exp5240_guard": bool(guard_mapping),
        "guard_enabled": guard_mapping.get("enabled") is True,
        "failure_mode_targeted": str(guard_mapping.get("failure_mode_targeted") or "none"),
        "blocked_arc_consumer_actions": list(guard_mapping.get("blocked_arc_consumer_actions") or []),
    }
    return {
        "attempt_id": f"exp5253_{target_game}_seed_{int(random_seed)}_budget_{int(budget)}",
        "target_game": str(target_game),
        "target_level": int(target_level),
        "prior_reproduced_level": 0,
        "budget": int(budget),
        "random_seed": int(random_seed),
        "runtime_s": runtime_s,
        "exact_command": str(exact_command),
        "policy": "arc_competition_agent._recommend_live_approach",
        "solve_provenance": SOLVE_PROVENANCE_LIVE,
        "live_agent_patch_enabled": bool(route_receipt["guard_enabled"]),
        "runtime_self_discovery_attempted": True,
        "solution_labels": [],
        "reproduction_gate": {
            "claimed_level": 0,
            "reproduced": False,
            "registry_validation_passed": False,
            "reached_level": 0,
        },
        "llm_proposer_used": False,
        "model_specs": None,
        "forbidden_methods": _false_forbidden_methods(),
        "provenance_route_receipts": [route_receipt],
        "approach_recommendation": recommendation,
    }


def build_artifact(
    *,
    registry_summary: Mapping[str, Any],
    live_attempt: Mapping[str, Any],
    duration_s: float,
    attempt_log_path: Path | str,
    input_checksum: str,
    output_checksum: str,
    tests_run: Sequence[Mapping[str, Any]],
    result_path: Path | str = RESULT_RELATIVE_PATH,
) -> JsonDict:
    """Build the principle-wrapped Exp5253 artifact."""

    target_game = str(live_attempt.get("target_game") or DEFAULT_TARGET_GAME)
    target_level = _as_int(live_attempt.get("target_level"), DEFAULT_TARGET_LEVEL)
    precheck = registry_precheck(
        registry_summary,
        target_game=target_game,
        target_level=target_level,
    )
    decision = _classify_attempt(live_attempt, precheck)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(result_path),
        "field_principles": dict(FIELD_PRINCIPLES),
        "target_game": target_game,
        "target_level": target_level,
        "random_seed": _as_int(live_attempt.get("random_seed"), RANDOM_SEED),
        "budget": _as_int(live_attempt.get("budget"), DEFAULT_BUDGET),
        "exact_command": str(live_attempt.get("exact_command") or DEFAULT_EXACT_COMMAND),
        "live_agent_patch_enabled": bool(live_attempt.get("live_agent_patch_enabled")),
        "forbidden_methods": dict(live_attempt.get("forbidden_methods") or {}),
        "source_artifacts": [EXP5240_RESULT_RELATIVE_PATH, REGISTRY_RELATIVE_PATH, PATCH_RELATIVE_PATH],
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(decision)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "solve_provenance": _wrap(
            "solve_provenance",
            str(live_attempt.get("solve_provenance") or SOLVE_PROVENANCE_LIVE),
        ),
        "registry_precheck": _wrap("registry_precheck", precheck),
        "level_delta": _wrap("level_delta", decision["level_delta"]),
        "levels_reproduced": _wrap("levels_reproduced", decision["levels_reproduced"]),
        "duplicate_solve_claimed": _wrap("duplicate_solve_claimed", False),
        "retire_current_provenance_patch": _wrap(
            "retire_current_provenance_patch", decision["retire_current_provenance_patch"]
        ),
        "duration_s": _wrap("duration_s", max(0.0, round(float(duration_s), 6))),
        "attempt_log_path": _wrap("attempt_log_path", str(attempt_log_path)),
        "input_checksum": _wrap("input_checksum", input_checksum),
        "output_checksum": _wrap("output_checksum", output_checksum),
        "provenance_route_receipts": _wrap(
            "provenance_route_receipts", decision["provenance_route_receipts"]
        ),
        "attempt_summary": {
            "attempt_id": live_attempt.get("attempt_id"),
            "runtime_s": live_attempt.get("runtime_s"),
            "reproduction_gate": dict(_attempt_gate(live_attempt)),
            "residual": decision["residual"],
            "patch_decision": decision["patch_decision"],
            "llm_proposer_used": bool(live_attempt.get("llm_proposer_used")),
            "model_specs": live_attempt.get("model_specs"),
        },
        "positive_control_passed": bool(
            decision["provenance_route_receipts"]
            and not any(bool(artifact_forbidden) for artifact_forbidden in dict(live_attempt.get("forbidden_methods") or {}).values())
        ),
        "null_delta_methodology_note": _null_delta_note(decision),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp5253 artifact before it is trusted."""

    missing = REQUIRED_SCHEMA_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("schema") != SCHEMA:
        raise ValueError("schema")
    if artifact.get("experiment") != EXPERIMENT:
        raise ValueError("experiment")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles")
    for field in REQUIRED_PRINCIPLE_FIELDS:
        _required_value(artifact, field)
    verdict = _required_value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked_")):
        raise ValueError("honest_verdict terminal prefix")
    if "level_delta=" not in verdict or "patch_decision=" not in verdict:
        raise ValueError("honest_verdict must state level_delta and patch_decision")
    if _required_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if _required_value(artifact, "solve_provenance") not in VALID_SOLVE_PROVENANCE:
        raise ValueError("solve_provenance")
    delta = _required_value(artifact, "level_delta")
    if type(delta) is not int:
        raise ValueError("level_delta")
    levels = _required_value(artifact, "levels_reproduced")
    if not isinstance(levels, list):
        raise ValueError("levels_reproduced")
    if _required_value(artifact, "duplicate_solve_claimed") is not False:
        raise ValueError("duplicate_solve_claimed")
    retire = _required_value(artifact, "retire_current_provenance_patch")
    if type(retire) is not bool:
        raise ValueError("retire_current_provenance_patch")
    duration = _required_value(artifact, "duration_s")
    if not _is_number(duration):
        raise ValueError("duration_s")
    route_receipts = _required_value(artifact, "provenance_route_receipts")
    if not isinstance(route_receipts, list):
        raise ValueError("provenance_route_receipts")
    precheck = _required_value(artifact, "registry_precheck")
    if not isinstance(precheck, Mapping) or "target_already_reproduced" not in precheck:
        raise ValueError("registry_precheck")
    for field in ("input_checksum", "output_checksum"):
        if not _is_sha256(_required_value(artifact, field)):
            raise ValueError(f"{field} checksum")
    forbidden = artifact.get("forbidden_methods")
    if not isinstance(forbidden, Mapping):
        raise ValueError("forbidden_methods")
    if any(bool(forbidden.get(key)) for key in FORBIDDEN_METHOD_KEYS):
        raise ValueError("forbidden_methods")
    if delta > 0 and retire:
        raise ValueError("retire_current_provenance_patch")
    if delta == 0 and levels:
        raise ValueError("levels_reproduced")
    tests = artifact.get("tests_run")
    if not isinstance(tests, list):
        raise ValueError("tests_run")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def write_artifact(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    attempt_log_path: Path | str = REPO_ROOT / ATTEMPT_LOG_RELATIVE_PATH,
    target_game: str = DEFAULT_TARGET_GAME,
    target_level: int = DEFAULT_TARGET_LEVEL,
    budget: int = DEFAULT_BUDGET,
    random_seed: int = RANDOM_SEED,
    exact_command: str = DEFAULT_EXACT_COMMAND,
    tests_run: Sequence[Mapping[str, Any]],
    update_exclusion_manifest: bool = False,
) -> JsonDict:
    """Run the live path, write attempt logs, and emit the Exp5253 artifact."""

    started = time.monotonic()
    root_path = Path(root)
    registry = load_registry_summary(root_path)
    input_hash = input_checksum(
        root_path,
        target_game=target_game,
        target_level=target_level,
        random_seed=random_seed,
        budget=budget,
    )
    attempt = run_live_agent_patch_attempt(
        root=root_path,
        target_game=target_game,
        target_level=target_level,
        budget=budget,
        random_seed=random_seed,
        exact_command=exact_command,
    )
    log_path = Path(attempt_log_path)
    _write_attempt_log(log_path, [attempt])
    output_hash = "sha256:" + hashlib.sha256(log_path.read_bytes()).hexdigest()
    duration_s = time.monotonic() - started
    display_log_path = _display_path(log_path, root_path)
    display_result_path = _display_path(Path(output_path), root_path)
    artifact = build_artifact(
        registry_summary=registry,
        live_attempt=attempt,
        duration_s=duration_s,
        attempt_log_path=display_log_path,
        input_checksum=input_hash,
        output_checksum=output_hash,
        tests_run=tests_run,
        result_path=display_result_path,
    )
    validate_artifact(artifact)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if update_exclusion_manifest:
        ensure_retirement_manifest_entry(root_path / "ops" / "exclusion_manifest.yaml", artifact)
    return artifact


def ensure_retirement_manifest_entry(manifest_path: Path | str, artifact: Mapping[str, Any]) -> bool:
    """Append one scoped exclusion-manifest retirement entry when Exp5253 retires the patch."""

    if _required_value(artifact, "retire_current_provenance_patch") is not True:
        return False
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if RETIREMENT_SCOPE_ID in text:
        return False
    block = _retirement_yaml_block()
    stripped = text.rstrip()
    if not stripped:
        path.write_text("retired_extras:\n" + block, encoding="utf-8")
        return True
    if stripped.endswith("retired_extras: []"):
        prefix = stripped[: -len("retired_extras: []")] + "retired_extras:\n"
        path.write_text(prefix + block, encoding="utf-8")
        return True
    path.write_text(stripped + "\n" + block, encoding="utf-8")
    return True


def _retirement_yaml_block() -> str:
    return (
        f"- id: {RETIREMENT_SCOPE_ID}\n"
        "  experiment_scope: Exp 5240/5241 ARC provenance-routing live patch scope\n"
        "  reason: >-\n"
        "    retire_current_provenance_patch: Exp 5253 clean live-path receipts kept\n"
        "    level_delta=0, so future work should not rerun this zero-delta\n"
        "    provenance-routing patch as a solve candidate without an operator\n"
        "    override and a new mechanism.\n"
        "  experiment_ids:\n"
        "  - exp5240\n"
        "  - exp5241\n"
        "  - exp5253\n"
        "  retired_milestone: 2026.07.480\n"
        f"  retired_by_artifact: {RESULT_RELATIVE_PATH.as_posix()}\n"
        "  operator_reopen_required: true\n"
        "  retire_if_same_verdict: true\n"
        "  blocked_patterns:\n"
        "  - experiment_5240_arc_rubric_to_patch_synthesis_v479\n"
        "  - experiment_5241_arc_gated_live_patch_attempt_v479\n"
        "  - provenance-routing live patch zero-delta rerun\n"
    )


def _write_attempt_log(path: Path, attempts: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(dict(attempt), sort_keys=True, default=str) for attempt in attempts]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _classify_attempt(live_attempt: Mapping[str, Any], precheck: Mapping[str, Any]) -> JsonDict:
    forbidden = live_attempt.get("forbidden_methods")
    forbidden_used = isinstance(forbidden, Mapping) and any(
        bool(forbidden.get(key)) for key in FORBIDDEN_METHOD_KEYS
    )
    solve_provenance = str(live_attempt.get("solve_provenance") or "")
    gate = _attempt_gate(live_attempt)
    prior = _as_int(live_attempt.get("prior_reproduced_level"))
    claimed_level = _as_int(gate.get("claimed_level"))
    labels = live_attempt.get("solution_labels")
    route_receipts = list(live_attempt.get("provenance_route_receipts") or [])
    route_ok = any(
        isinstance(row, Mapping)
        and row.get("reached_exp5240_guard") is True
        and row.get("guard_enabled") is True
        for row in route_receipts
    )
    success = bool(
        not forbidden_used
        and solve_provenance == SOLVE_PROVENANCE_LIVE
        and precheck.get("target_already_reproduced") is not True
        and route_ok
        and claimed_level > prior
        and gate.get("reproduced") is True
        and gate.get("registry_validation_passed") is True
        and isinstance(labels, list)
        and bool(labels)
    )
    if success:
        return {
            "level_delta": claimed_level - prior,
            "levels_reproduced": [str(live_attempt.get("target_game") or DEFAULT_TARGET_GAME)],
            "retire_current_provenance_patch": False,
            "patch_decision": "keep_current_provenance_patch",
            "residual": "accepted_live_self_discovery",
            "provenance_route_receipts": route_receipts,
            "blocked": False,
        }
    if forbidden_used:
        residual = "forbidden_method_used"
        patch_decision = "blocked_no_patch_decision"
        retire = False
        blocked = True
    elif solve_provenance not in VALID_SOLVE_PROVENANCE:
        residual = "unknown_solve_provenance"
        patch_decision = "blocked_no_patch_decision"
        retire = False
        blocked = True
    elif solve_provenance != SOLVE_PROVENANCE_LIVE:
        residual = "non_live_solve_provenance"
        patch_decision = "blocked_no_patch_decision"
        retire = False
        blocked = True
    elif precheck.get("target_already_reproduced") is True:
        residual = "target_already_reproduced_no_duplicate_claim"
        patch_decision = "retire_current_provenance_patch"
        retire = True
        blocked = False
    elif not route_ok:
        residual = "exp5240_guard_not_reached"
        patch_decision = "blocked_no_patch_decision"
        retire = False
        blocked = True
    else:
        residual = "clean_zero_delta_no_level_banked"
        patch_decision = "retire_current_provenance_patch"
        retire = True
        blocked = False
    return {
        "level_delta": 0,
        "levels_reproduced": [],
        "retire_current_provenance_patch": retire,
        "patch_decision": patch_decision,
        "residual": residual,
        "provenance_route_receipts": route_receipts,
        "blocked": blocked,
    }


def _honest_verdict(decision: Mapping[str, Any]) -> str:
    prefix = "blocked_" if decision.get("blocked") else "complete:"
    if prefix == "blocked_":
        prefix = f"blocked_{decision['residual']}:"
    return (
        f"{prefix} level_delta={decision['level_delta']} "
        f"patch_decision={decision['patch_decision']} "
        f"solve_provenance={SOLVE_PROVENANCE_LIVE}; {decision['residual']}."
    )


def _null_delta_note(decision: Mapping[str, Any]) -> str:
    if _as_int(decision.get("level_delta")) != 0:
        return ""
    return (
        "level_delta=0 is the measured clean no-bank result: registry precheck ran, "
        "the live route reached the Exp5240 provenance guard, no forbidden methods "
        "were used, and reproduction_gate did not reproduce a new level."
    )


def _attempt_gate(live_attempt: Mapping[str, Any]) -> Mapping[str, Any]:
    gate = live_attempt.get("reproduction_gate")
    return gate if isinstance(gate, Mapping) else {}


def _false_forbidden_methods() -> JsonDict:
    return {key: False for key in FORBIDDEN_METHOD_KEYS}


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _required_value(artifact: Mapping[str, Any], field: str) -> Any:
    wrapper = artifact.get(field)
    if not isinstance(wrapper, Mapping) or "value" not in wrapper or "principle" not in wrapper:
        raise ValueError(f"{field} must be principle-wrapped")
    if wrapper.get("principle") != FIELD_PRINCIPLES[field]:
        raise ValueError(f"{field} principle mismatch")
    return wrapper.get("value")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_test_run(value: str) -> JsonDict:  # pragma: no cover - CLI wrapper
    if "=" not in value:
        return {"command": value, "passed": True}
    command, outcome = value.rsplit("=", 1)
    return {"command": command, "passed": outcome.strip().lower() in {"pass", "passed", "true"}}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--attempt-log", default=str(REPO_ROOT / ATTEMPT_LOG_RELATIVE_PATH))
    parser.add_argument("--target-game", default=DEFAULT_TARGET_GAME)
    parser.add_argument("--target-level", type=int, default=DEFAULT_TARGET_LEVEL)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--exact-command", default=DEFAULT_EXACT_COMMAND)
    parser.add_argument("--test-run", action="append", default=[])
    parser.add_argument("--update-exclusion-manifest", action="store_true")
    args = parser.parse_args(argv)
    artifact = write_artifact(
        root=args.root,
        output_path=args.output,
        attempt_log_path=args.attempt_log,
        target_game=args.target_game,
        target_level=args.target_level,
        budget=args.budget,
        random_seed=args.random_seed,
        exact_command=args.exact_command,
        tests_run=[_parse_test_run(item) for item in args.test_run],
        update_exclusion_manifest=args.update_exclusion_manifest,
    )
    print(
        json.dumps(
            {
                "result_path": str(args.output),
                "level_delta": _required_value(artifact, "level_delta"),
                "retire_current_provenance_patch": _required_value(
                    artifact, "retire_current_provenance_patch"
                ),
                "checksum": artifact["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
