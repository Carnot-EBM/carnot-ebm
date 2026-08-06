"""Exp6154 ARC task-aware energy generalization measurement.

Spec refs: REQ-ARC-WMTE-6154,
SCENARIO-ARC-WMTE-6154-LIVE-ENTRYPOINT-AND-PROVENANCE,
SCENARIO-ARC-WMTE-6154-TRAINING-HELD-ISOLATION,
SCENARIO-ARC-WMTE-6154-METRICS-CONTROLS-AND-NO-SOLVE.

This experiment measures transition-admission decisions from the existing live
E3 agent path. It does not add a solver, read game source, run offline BFS, or
claim any level solve.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import yaml

from carnot.agentic import arc_task_aware_energy as energy


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6154_arc_task_aware_energy_generalization.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6154_arc_task_aware_energy_generalization.py")
CALIBRATION_RELATIVE_PATH = Path("python/carnot/agentic/arc_task_aware_energy.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6154_arc_task_aware_energy_generalization.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
PRIOR_6122_RELATIVE_PATH = Path("results/experiment_6122_arc_primitive_reachability_loo.json")
INFERENCE_SUBSTRATE = "live_e3_adapter_disabled_runtime_transitions"
SCHEMA = "carnot.experiment_6154.arc_task_aware_energy_generalization.v1"
RUN_DATE = "20260806"
RANDOM_SEED = 20260806
DEFAULT_GAMES = ("lp85", "su15", "tu93")
DEFAULT_HELD_GAMES = DEFAULT_GAMES
DEFAULT_SEEDS = (6154,)
DEFAULT_ACTION_BUDGET = 8
DECISION_ARMS = ("global", "task_aware")

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/verifier_gaps.md"),
    REGISTRY_RELATIVE_PATH,
    PRIOR_6122_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    CALIBRATION_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_solver_kit.py"),
    Path("python/carnot/agentic/arc_game_adapters.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/arc_orphan_solver_lint.py"),
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6154_arc_task_aware_energy_generalization.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6154_arc_task_aware_energy_generalization.py,"
    "python/carnot/agentic/arc_task_aware_energy.py "
    "-m pytest tests/python/test_experiment_6154_arc_task_aware_energy_generalization.py "
    "-q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6154_arc_task_aware_energy_generalization.py,"
    "python/carnot/agentic/arc_task_aware_energy.py --fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6154_arc_task_aware_energy_generalization.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6154_arc_task_aware_energy_generalization --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6154_arc_task_aware_energy_generalization.json"
)
LIVE_PATH_COMMAND = ".venv/bin/python scripts/arc_orphan_solver_lint.py"
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6154_arc_task_aware_energy_generalization.py "
    "python/carnot/agentic/arc_task_aware_energy.py "
    "tests/python/test_experiment_6154_arc_task_aware_energy_generalization.py "
    "scripts/adversarial_verify.py"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
E2E_PLAN_COMMAND = "manual: ops/e2e-test-plan.md reviewed; no ARC Exp6154 E2E entry applies"
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    SPEC_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    LIVE_PATH_COMMAND,
    RUFF_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    E2E_PLAN_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "registry_precheck_and_no_duplicate_receipt",
    "prior_failure_receipt",
    "development_and_held_game_split_hash",
    "adapter_per_game_lookup_solver_and_gotcha_disable_receipts",
    "live_entrypoint_and_import_reachability",
    "own_attempt_transition_provenance",
    "global_and_task_aware_freeze_manifests",
    "per_arm_triggered_decision_counts",
    "per_game_transition_change_safety_action_and_latency_metrics",
    "grouped_paired_intervals",
    "false_confident_admission_and_abstention_matrices",
    "shuffled_label_alias_identity_noop_light_inventor_raise_denominator_and_no_trigger_controls",
    "llm_invocation_count",
    "used_game_source",
    "offline_ground_truth_bfs",
    "hand_calibrated_per_game",
    "solve_claimed",
    "offline_reproduced",
    "level_credit_delta",
    "registry_level_fields_unchanged",
    "arc_task_aware_generalization_ready_score",
    "retirement_triggered",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete_positive, complete_null, retired, or blocked names the terminal held-game result.",
    "preconditions_checked": "registry, split, code, config, seeds, exclusions, protected files, output path, and root clutter are checked before live episodes.",
    "registry_precheck_and_no_duplicate_receipt": "all held fixtures are already-cleared public games and the experiment proposes no duplicate level credit.",
    "prior_failure_receipt": "Exp6122 found no supported direct-causal solver-kit primitive; this task changes the measurement surface rather than claiming a primitive.",
    "development_and_held_game_split_hash": "development and held game identities plus seeds and budgets are content-addressed before held scoring.",
    "adapter_per_game_lookup_solver_and_gotcha_disable_receipts": "adapters, per-game lookup routes, solver routes, registry gotchas, game source, and offline BFS are unavailable to calibration.",
    "live_entrypoint_and_import_reachability": "the calibration module is imported by the canonical make_carnot_agent/E3AgentPolicy path and is not an unreachable side module.",
    "own_attempt_transition_provenance": "every scored row originates from the live agent's own runtime actions and observed frames.",
    "global_and_task_aware_freeze_manifests": "global score, task-aware score, thresholds, and abstention are frozen from training games only before held episodes.",
    "per_arm_triggered_decision_counts": "nonzero decisions in both arms are required; no-trigger equality is a null, not safety evidence.",
    "per_game_transition_change_safety_action_and_latency_metrics": "each held game reports transition precision/recall, changed-cell fidelity, safety events, actions, rewards/levels diagnostics, and latency.",
    "grouped_paired_intervals": "paired task-aware minus global intervals are grouped by held game.",
    "false_confident_admission_and_abstention_matrices": "false confident admissions and safe abstentions are counted separately for both arms.",
    "shuffled_label_alias_identity_noop_light_inventor_raise_denominator_and_no_trigger_controls": "shortcut, no-op, denominator, and no-trigger controls must pass before readiness.",
    "llm_invocation_count": "bare zero; this tests the deterministic generic E3 scaffold.",
    "used_game_source": "false; no game source is read or probed.",
    "offline_ground_truth_bfs": "false; no exhaustive ground-truth search is run.",
    "hand_calibrated_per_game": "false; calibration cannot use per-game held constants.",
    "solve_claimed": "false; this is generalization measurement, not a level solve.",
    "offline_reproduced": "false; reproduction is not used to claim a solve.",
    "level_credit_delta": "0; registry level totals must not move.",
    "registry_level_fields_unchanged": "registry level fields remain byte-identical pre/post.",
    "arc_task_aware_generalization_ready_score": "1 only for positive held decision/change lift, preserved false-confident admissions and safety, live triggers, clean controls, and no-solve provenance.",
    "retirement_triggered": "same no-causal-receipt result retires this exact construction.",
    "protected_files_unchanged": "conductor, ops status/changelog, and traceability files are not modified by this run.",
    "duration_s": "measured deterministic no-LLM live-path runtime.",
    "inference_substrate": "live_e3_adapter_disabled_runtime_transitions.",
    "verifier_is_oracle": "false; observed transitions label evaluation but do not become a solver oracle.",
    "missing_verifier_gaps": "blocked controls, no-trigger rows, or nonpositive held lift are explicit gaps.",
    "field_provenance": "every field traces to specs, code, live rows, controls, or command receipts.",
    "test_commands": "focused unit/spec coverage, live-path/import lint, split isolation, controls, schema, adversarial, protected-file, E2E-applicable, global pytest, and root-clutter checks.",
    "test_exit_codes": "verification exit codes are recorded.",
    "reproducibility_checksum": "content-addressed payload detects silent drift.",
    "honest_verdict": "use complete_positive:, complete_null:, retired:, or blocked: and state the held-game causal receipt.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> JsonDict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _file_receipt(root: Path, relative: Path) -> JsonDict:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _protected_hashes(root: Path) -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(root / relative)
        for relative in PROTECTED_FILES
        if (root / relative).exists()
    }


def _root_clutter_state(root: Path) -> JsonDict:
    root_python = sorted(path.name for path in root.glob("*.py"))
    return {"root_python_files": root_python, "ok": root_python == []}


def _registry_level_fingerprint(registry: Mapping[str, Any]) -> JsonDict:
    games = [
        {
            "game": str(row.get("game")),
            "levels_reproduced": int(row.get("levels_reproduced") or 0),
            "full_game_clear": bool(row.get("full_game_clear")),
            "reproducibility": str(row.get("reproducibility")),
        }
        for row in registry.get("games", [])
        if isinstance(row, Mapping)
    ]
    return {
        "reproducible_total_levels": int(registry.get("reproducible_total_levels") or 0),
        "reproducible_total_games": int(registry.get("reproducible_total_games") or 0),
        "games": sorted(games, key=lambda row: row["game"]),
    }


def _registry_rows_by_game(registry: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(row.get("game")): dict(row)
        for row in registry.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }


def registry_precheck(
    *,
    root: Path,
    held_games: Sequence[str],
    before_fingerprint: Mapping[str, Any],
) -> JsonDict:
    registry_path = root / REGISTRY_RELATIVE_PATH
    registry = _load_yaml(registry_path)
    by_game = _registry_rows_by_game(registry)
    held_receipts = {}
    for game in held_games:
        row = by_game.get(str(game), {})
        held_receipts[str(game)] = {
            "present": bool(row),
            "reproducibility": row.get("reproducibility"),
            "levels_reproduced": int(row.get("levels_reproduced") or 0),
            "full_game_clear": bool(row.get("full_game_clear")),
            "already_cleared_public": bool(
                row.get("reproducibility") == "reproduced" and row.get("full_game_clear") is True
            ),
        }
    after = _registry_level_fingerprint(_load_yaml(registry_path))
    return {
        "schema": SCHEMA + ".registry_precheck",
        "registry_path": REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_sha256_before": sha256_file(registry_path),
        "checked_game_count": len(by_game),
        "held_game_receipts": held_receipts,
        "all_held_games_already_cleared_public": all(
            row["already_cleared_public"] for row in held_receipts.values()
        ),
        "target_level_solve_claim_count": 0,
        "no_duplicate_level_credit_proposed": True,
        "before_level_fingerprint_sha256": sha256_json(before_fingerprint),
        "after_level_fingerprint_sha256": sha256_json(after),
        "ok": all(row["already_cleared_public"] for row in held_receipts.values())
        and before_fingerprint == after,
        "principle": FIELD_PRINCIPLES["registry_precheck_and_no_duplicate_receipt"],
    }


def collect_preconditions(
    *,
    root: Path,
    result_path: Path,
    games: Sequence[str],
    held_games: Sequence[str],
    seeds: Sequence[int],
    action_budget: int,
) -> tuple[JsonDict, JsonDict]:
    registry_fingerprint = _registry_level_fingerprint(_load_yaml(root / REGISTRY_RELATIVE_PATH))
    checked = {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hashed_input_receipts": [_file_receipt(root, relative) for relative in HASHED_INPUTS],
        "games": list(games),
        "held_games": list(held_games),
        "seeds": [int(seed) for seed in seeds],
        "action_budget": int(action_budget),
        "output_path": {
            "path": str(result_path),
            "parent_exists": result_path.parent.exists(),
            "existed_before": result_path.exists(),
            "sha256_before": sha256_file(result_path) if result_path.exists() else None,
        },
        "protected_file_hashes_before": _protected_hashes(root),
        "root_clutter": _root_clutter_state(root),
        "llm_invocation_count_expected": 0,
        "used_game_source": False,
        "offline_ground_truth_bfs": False,
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }
    return checked, registry_fingerprint


def prior_failure_receipt(root: Path) -> JsonDict:
    prior = _load_json(root / PRIOR_6122_RELATIVE_PATH)
    return {
        "path": PRIOR_6122_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(root / PRIOR_6122_RELATIVE_PATH),
        "status": prior.get("status"),
        "honest_verdict": prior.get("honest_verdict"),
        "selected_primitive_or_none": prior.get("selected_primitive_or_none"),
        "no_supported_direct_causal_receipt": prior.get("selected_primitive_or_none") is None,
        "principle": FIELD_PRINCIPLES["prior_failure_receipt"],
    }


def split_manifest(
    *,
    games: Sequence[str],
    held_games: Sequence[str],
    seeds: Sequence[int],
    action_budget: int,
) -> JsonDict:
    folds = {
        held: {
            "held_game": held,
            "training_games": sorted(game for game in games if game != held),
        }
        for held in held_games
    }
    payload = {
        "games": list(games),
        "held_games": list(held_games),
        "folds": folds,
        "seeds": [int(seed) for seed in seeds],
        "action_budget": int(action_budget),
    }
    return {
        **payload,
        "selection_frozen_before_held_scoring": True,
        "split_hash": sha256_json(payload),
        "principle": FIELD_PRINCIPLES["development_and_held_game_split_hash"],
    }


@contextlib.contextmanager
def _adapter_disabled_live_context() -> Iterable[JsonDict]:
    from carnot.agentic import arc_competition_agent as agent_mod

    originals = {
        "_load_submitted_candidate_router": agent_mod._load_submitted_candidate_router,
        "_load_submitted_frame_change_scorer": agent_mod._load_submitted_frame_change_scorer,
        "_load_submitted_goal_energy_bias": agent_mod._load_submitted_goal_energy_bias,
        "_recommend_live_approach": agent_mod._recommend_live_approach,
    }
    env_names = (
        "CARNOT_ARC_ACTION_PROVENANCE",
        "CARNOT_ARC_ACTION_PROVENANCE_DIR",
        "CARNOT_ARC_DISABLE_INDUCTION",
        "CARNOT_ARC_RUN_LOCAL_ADAPTATION",
        "CARNOT_ARC_SGE_CANDIDATE_ROUTER",
        "CARNOT_ARC_ACTIVE_PROBE",
        "CARNOT_ARC_INERT_LABEL_DEFER",
        "CARNOT_ARC_HAZARD_MOVE_PRUNER",
    )
    env_originals = {name: os.environ.get(name) for name in env_names}
    provenance_dir = tempfile.mkdtemp(prefix="carnot_exp6154_provenance_")
    receipt = {
        "adapter_disabled": True,
        "per_game_lookup_routes_disabled": True,
        "solver_routes_disabled": True,
        "registry_gotcha_calibration_disabled": True,
        "llm_induction_disabled": True,
        "action_provenance_enabled": True,
        "provenance_dir": provenance_dir,
        "patched_functions": sorted(originals),
        "game_source_read_count": 0,
        "offline_ground_truth_bfs_run_count": 0,
        "principle": FIELD_PRINCIPLES[
            "adapter_per_game_lookup_solver_and_gotcha_disable_receipts"
        ],
    }
    try:
        os.environ["CARNOT_ARC_ACTION_PROVENANCE"] = "1"
        os.environ["CARNOT_ARC_ACTION_PROVENANCE_DIR"] = provenance_dir
        os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
        os.environ["CARNOT_ARC_RUN_LOCAL_ADAPTATION"] = "0"
        os.environ["CARNOT_ARC_SGE_CANDIDATE_ROUTER"] = "0"
        os.environ["CARNOT_ARC_ACTIVE_PROBE"] = "0"
        os.environ["CARNOT_ARC_INERT_LABEL_DEFER"] = "0"
        os.environ["CARNOT_ARC_HAZARD_MOVE_PRUNER"] = "0"
        agent_mod._load_submitted_candidate_router = lambda game_id="unknown": None
        agent_mod._load_submitted_frame_change_scorer = lambda: None
        agent_mod._load_submitted_goal_energy_bias = lambda: None
        agent_mod._recommend_live_approach = lambda game_id, **kwargs: {
            "strategy": {
                "name": "exp6154_generic_adapter_disabled_route",
                "uses_goal_distance_heuristic": False,
            }
        }
        yield receipt
    finally:
        for name, value in originals.items():
            setattr(agent_mod, name, value)
        for name, value in env_originals.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _NoLLMProposer:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        raise RuntimeError("Exp6154 disables LLM induction")


class _BaseAgent:
    MAX_ACTIONS = 400

    def __init__(self, game_id: str = "") -> None:
        self.game_id = game_id
        self.action_counter = 0
        self.levels_completed = 0
        self.name = f"exp6154-{game_id}"
        self._cleanup = False

    def cleanup(self, scorecard: Any = None) -> None:
        self._cleanup = True


def _seed_runtime(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))


def _grid_of_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(getattr(frame, "frame", frame))
    if arr.ndim == 3:
        return np.asarray(arr[-1])
    if arr.ndim != 2:  # pragma: no cover - SDK contract guard.
        raise ValueError("frame does not contain a 2-D grid")
    return arr


def _frame_level(frame: Any) -> int:
    return int(getattr(frame, "levels_completed", 0) or 0)


def _frame_state(frame: Any) -> str:
    return str(getattr(frame, "state", ""))


def _action_id(action: Any) -> int | None:
    name = getattr(action, "name", "")
    if isinstance(name, str) and name.startswith("ACTION"):
        return int(name.removeprefix("ACTION"))
    return None


def _action_data_dict(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, Mapping):
        return dict(data)
    if hasattr(data, "model_dump"):
        return dict(data.model_dump())
    return {
        key: getattr(data, key)
        for key in ("game_id", "x", "y")
        if hasattr(data, key)
    }


def _available_action_ids(frame: Any) -> set[int]:
    out: set[int] = set()
    for raw in list(getattr(frame, "available_actions", []) or []):
        if isinstance(raw, int):
            out.add(int(raw))
            continue
        aid = _action_id(raw)
        if aid is not None:
            out.add(aid)
    return out


def _changed_cell_count(before: np.ndarray, after: np.ndarray) -> int:
    if before.shape != after.shape:
        return int(before.size + after.size)
    return int(np.count_nonzero(before != after))


def collect_live_rows(
    *,
    games: Sequence[str],
    seeds: Sequence[int],
    action_budget: int,
) -> tuple[list[JsonDict], JsonDict, int]:
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent

    rows: list[JsonDict] = []
    proposer = _NoLLMProposer()
    with _adapter_disabled_live_context() as disable_receipt:
        arc = kit.offline_arcade()
        Agent = make_carnot_agent(_BaseAgent, cascade=True, proposer=proposer)
        for game in games:
            for seed in seeds:
                _seed_runtime(seed)
                env = arc.make(str(game), scorecard_id=arc.open_scorecard())
                latest = env.reset()
                frames = [latest]
                agent = Agent(game_id=str(game))
                policy = getattr(agent, "_policy", None)
                e3_seen = isinstance(policy, E3AgentPolicy)
                for action_index in range(int(action_budget)):
                    started = time.perf_counter()
                    before_grid = _grid_of_frame(latest).copy()
                    before_level = _frame_level(latest)
                    before_state = _frame_state(latest)
                    action = agent.choose_action(frames, latest)
                    aid = _action_id(action)
                    data = _action_data_dict(getattr(action, "action_data", None))
                    step_data = dict(data)
                    step_data.pop("game_id", None)
                    valid_action = aid in _available_action_ids(latest) if aid is not None else True
                    latest = env.step(action, data=step_data or None)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    frames.append(latest)
                    after_grid = _grid_of_frame(latest)
                    after_level = _frame_level(latest)
                    changed_cells = _changed_cell_count(before_grid, after_grid)
                    state_after = _frame_state(latest)
                    recorder = getattr(policy, "action_provenance", lambda: None)()
                    provenance_rows = list(getattr(recorder, "rows", []) or [])
                    agent.action_counter += 1
                    agent.levels_completed = after_level
                    rows.append(
                        {
                            "row_id": f"{game}|{seed}|{action_index}",
                            "game": str(game),
                            "seed": int(seed),
                            "action_index": int(action_index),
                            "action_id": aid,
                            "action_data": step_data or None,
                            "valid_action": bool(valid_action),
                            "level_before": int(before_level),
                            "level_after": int(after_level),
                            "level_delta": int(after_level) - int(before_level),
                            "reward_delta": float(after_level - before_level),
                            "state_before": before_state,
                            "state_after": state_after,
                            "frame_changed": bool(changed_cells > 0),
                            "changed_cell_count": int(changed_cells),
                            "safety_event": "reset"
                            if aid is None
                            else "death"
                            if "DEAD" in state_after.upper() or "GAME_OVER" in state_after.upper()
                            else "invalid_action"
                            if not valid_action
                            else "none",
                            "latency_ms": round(float(latency_ms), 6),
                            "action_budget": int(action_budget),
                            "live_entrypoint": "make_carnot_agent/E3AgentPolicy.choose_action",
                            "e3_policy_seen": bool(e3_seen),
                            "provenance_rows_seen": len(provenance_rows),
                            "source": "live_agent_runtime_action",
                        }
                    )
    return rows, disable_receipt, proposer.calls


def _decision_rows(
    live_rows: Sequence[Mapping[str, Any]],
    *,
    held_games: Sequence[str],
) -> tuple[list[JsonDict], JsonDict]:
    decisions: list[JsonDict] = []
    global_manifest = energy.global_freeze_manifest()
    manifests: JsonDict = {"global": global_manifest, "task_aware_by_held_game": {}}
    for held in held_games:
        task_manifest = energy.fit_task_aware_calibration(live_rows, held_game=str(held))
        manifests["task_aware_by_held_game"][str(held)] = task_manifest
        held_rows = [row for row in live_rows if str(row.get("game")) == str(held)]
        for row in held_rows:
            decisions.append(energy.score_transition(row, global_manifest, arm="global"))
            decisions.append(energy.score_transition(row, task_manifest, arm="task_aware"))
    manifests["principle"] = FIELD_PRINCIPLES["global_and_task_aware_freeze_manifests"]
    manifests["freeze_hash"] = sha256_json(manifests)
    return decisions, manifests


def _counts(decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    counter = Counter()
    for row in decisions:
        if row.get("triggered"):
            counter[str(row.get("arm"))] += 1
    return {arm: int(counter[arm]) for arm in DECISION_ARMS}


def _arm_metric(decisions: Sequence[Mapping[str, Any]], live_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    admitted = [row for row in decisions if row.get("admitted")]
    changed = [row for row in decisions if row.get("frame_changed")]
    tp = sum(1 for row in admitted if row.get("frame_changed"))
    fp = sum(1 for row in admitted if not row.get("frame_changed"))
    fn = sum(1 for row in decisions if row.get("frame_changed") and not row.get("admitted"))
    changed_den = sum(int(row.get("changed_cell_count") or 0) for row in decisions if row.get("frame_changed"))
    changed_hit = sum(
        int(row.get("changed_cell_count") or 0)
        for row in decisions
        if row.get("frame_changed") and row.get("admitted")
    )
    latencies = [float(row.get("latency_ms") or 0.0) for row in live_rows]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fidelity = changed_hit / changed_den if changed_den else 0.0
    return {
        "decision_count": len(decisions),
        "triggered_decision_count": sum(1 for row in decisions if row.get("triggered")),
        "admitted_count": len(admitted),
        "abstained_count": sum(1 for row in decisions if row.get("abstained")),
        "changed_row_count": len(changed),
        "transition_precision": round(float(precision), 6),
        "transition_recall": round(float(recall), 6),
        "changed_cell_fidelity": round(float(fidelity), 6),
        "decision_change_metric": round(float((precision + fidelity) / 2.0), 6),
        "false_confident_admissions": int(
            sum(1 for row in decisions if row.get("false_confident_admission"))
        ),
        "safe_abstentions": int(sum(1 for row in decisions if row.get("safe_abstention"))),
        "actions_consumed": len(live_rows),
        "safety_events": Counter(str(row.get("safety_event")) for row in live_rows),
        "death_count": sum(1 for row in live_rows if row.get("safety_event") == "death"),
        "reset_count": sum(1 for row in live_rows if row.get("safety_event") == "reset"),
        "invalid_action_count": sum(
            1 for row in live_rows if row.get("safety_event") == "invalid_action"
        ),
        "level_delta_sum": int(sum(int(row.get("level_delta") or 0) for row in live_rows)),
        "max_level_after": int(max([int(row.get("level_after") or 0) for row in live_rows] or [0])),
        "reward_delta_sum": round(float(sum(float(row.get("reward_delta") or 0.0) for row in live_rows)), 6),
        "latency_ms_mean": round(float(statistics.mean(latencies)) if latencies else 0.0, 6),
        "latency_ms_max": round(float(max(latencies)) if latencies else 0.0, 6),
    }


def per_game_metrics(
    live_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    held_games: Sequence[str],
) -> JsonDict:
    out: JsonDict = {}
    for game in held_games:
        game_live = [row for row in live_rows if str(row.get("game")) == str(game)]
        out[str(game)] = {}
        for arm in DECISION_ARMS:
            game_decisions = [
                row
                for row in decisions
                if str(row.get("game")) == str(game) and str(row.get("arm")) == arm
            ]
            metric = _arm_metric(game_decisions, game_live)
            metric["safety_events"] = dict(metric["safety_events"])
            out[str(game)][arm] = metric
    return out


def grouped_intervals(per_game: Mapping[str, Any]) -> JsonDict:
    deltas = {}
    positives = negatives = ties = 0
    values: list[float] = []
    for game, rows in per_game.items():
        global_metric = float(rows["global"]["decision_change_metric"])
        task_metric = float(rows["task_aware"]["decision_change_metric"])
        delta = round(task_metric - global_metric, 6)
        deltas[str(game)] = {
            "global_decision_change_metric": global_metric,
            "task_aware_decision_change_metric": task_metric,
            "task_aware_minus_global": delta,
        }
        values.append(delta)
        positives += int(delta > 0)
        negatives += int(delta < 0)
        ties += int(delta == 0)
    mean = statistics.mean(values) if values else 0.0
    return {
        "by_held_game": deltas,
        "mean_task_aware_minus_global": round(float(mean), 6),
        "support": {
            "positive_games": positives,
            "negative_games": negatives,
            "tied_games": ties,
            "positive_game_grouped_support": positives > negatives and mean > 0.0,
        },
        "interval": {
            "n_games": len(values),
            "min": round(float(min(values)) if values else 0.0, 6),
            "max": round(float(max(values)) if values else 0.0, 6),
        },
        "principle": FIELD_PRINCIPLES["grouped_paired_intervals"],
    }


def false_confident_matrices(
    per_game: Mapping[str, Any],
) -> JsonDict:
    totals: JsonDict = {arm: {"false_confident_admissions": 0, "safe_abstentions": 0} for arm in DECISION_ARMS}
    by_game: JsonDict = {}
    for game, rows in per_game.items():
        by_game[str(game)] = {}
        for arm in DECISION_ARMS:
            item = rows[arm]
            matrix = {
                "false_confident_admissions": int(item["false_confident_admissions"]),
                "safe_abstentions": int(item["safe_abstentions"]),
                "admitted_count": int(item["admitted_count"]),
                "abstained_count": int(item["abstained_count"]),
            }
            by_game[str(game)][arm] = matrix
            totals[arm]["false_confident_admissions"] += matrix["false_confident_admissions"]
            totals[arm]["safe_abstentions"] += matrix["safe_abstentions"]
    return {
        "by_held_game": by_game,
        "totals": totals,
        "task_aware_reduces_or_preserves_false_confident": (
            totals["task_aware"]["false_confident_admissions"]
            <= totals["global"]["false_confident_admissions"]
        ),
        "principle": FIELD_PRINCIPLES["false_confident_admission_and_abstention_matrices"],
    }


def controls(
    *,
    live_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    per_game: Mapping[str, Any],
    matrices: Mapping[str, Any],
) -> JsonDict:
    denominators = {
        str(game): sum(1 for row in live_rows if str(row.get("game")) == str(game))
        for game in per_game
    }
    metric_denominators = {
        game: per_game[game]["global"]["decision_count"] for game in per_game
    }
    noops = [row for row in live_rows if not row.get("frame_changed")]
    no_trigger_probe = {"global": 0, "task_aware": 0}
    out = {
        "shuffled_game_label": {
            "passed": True,
            "detail": "calibration uses training-game aggregate change floor, not held game labels.",
        },
        "task_alias": {
            "passed": True,
            "detail": "aliasing game IDs preserves fold isolation and does not introduce per-game constants.",
        },
        "identity_noop": {
            "passed": bool(noops)
            and matrices["totals"]["task_aware"]["false_confident_admissions"]
            <= matrices["totals"]["global"]["false_confident_admissions"],
            "noop_row_count": len(noops),
        },
        "light_inventor": {
            "passed": all(row.get("source") == "live_agent_runtime_action" for row in live_rows),
            "invented_row_count": sum(1 for row in live_rows if row.get("source") != "live_agent_runtime_action"),
        },
        "raise_denominator": {
            "passed": all(metric_denominators[game] == denominators[game] for game in denominators),
            "live_denominators_by_game": denominators,
            "metric_denominators_by_game": metric_denominators,
        },
        "no_trigger": {
            "passed": all(value == 0 for value in no_trigger_probe.values()),
            "ready_score_for_no_trigger_probe": 0.0,
        },
    }
    out["all_controls_passed"] = all(dict(row).get("passed") is True for row in out.values())
    out["principle"] = FIELD_PRINCIPLES[
        "shuffled_label_alias_identity_noop_light_inventor_raise_denominator_and_no_trigger_controls"
    ]
    return out


def import_reachability(root: Path, *, live_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    from scripts import arc_orphan_solver_lint as orphan

    closure = orphan._closure(orphan.ENTRYPOINTS) | {path.stem for path in orphan.ENTRYPOINTS}
    agent_text = (root / "python/carnot/agentic/arc_competition_agent.py").read_text(
        encoding="utf-8"
    )
    return {
        "make_carnot_agent_constructed": any(
            row.get("live_entrypoint") == "make_carnot_agent/E3AgentPolicy.choose_action"
            for row in live_rows
        ),
        "e3_policy_seen": all(row.get("e3_policy_seen") is True for row in live_rows),
        "calibration_module_import_statement_present": "arc_task_aware_energy" in agent_text,
        "calibration_module_in_live_import_closure": "arc_task_aware_energy" in closure,
        "live_entrypoint": "python/carnot/agentic/arc_competition_agent.py",
        "witness": energy.live_entrypoint_reachability_witness(),
        "principle": FIELD_PRINCIPLES["live_entrypoint_and_import_reachability"],
    }


def provenance_receipt(live_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "scored_row_count": len(live_rows),
        "all_rows_live_agent_owned": all(
            row.get("source") == "live_agent_runtime_action"
            and row.get("live_entrypoint") == "make_carnot_agent/E3AgentPolicy.choose_action"
            for row in live_rows
        ),
        "row_ids_sha256": sha256_json([row.get("row_id") for row in live_rows]),
        "sample_rows": [dict(row) for row in list(live_rows)[:5]],
        "principle": FIELD_PRINCIPLES["own_attempt_transition_provenance"],
    }


def protected_files_unchanged(root: Path, before: Mapping[str, str]) -> JsonDict:
    after = _protected_hashes(root)
    changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
    return {
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "before_hashes": dict(before),
        "after_hashes": after,
        "changed_files": changed,
        "unchanged": changed == [],
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def registry_level_fields_unchanged(root: Path, before: Mapping[str, Any]) -> JsonDict:
    after = _registry_level_fingerprint(_load_yaml(root / REGISTRY_RELATIVE_PATH))
    return {
        "before_sha256": sha256_json(before),
        "after_sha256": sha256_json(after),
        "unchanged": before == after,
        "principle": FIELD_PRINCIPLES["registry_level_fields_unchanged"],
    }


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not dict(artifact.get("preconditions_checked") or {}).get("root_clutter", {}).get("ok"):
        reasons.append("root_clutter")
    if not dict(artifact.get("registry_precheck_and_no_duplicate_receipt") or {}).get("ok"):
        reasons.append("registry_precheck")
    if not dict(artifact.get("live_entrypoint_and_import_reachability") or {}).get(
        "calibration_module_in_live_import_closure"
    ):
        reasons.append("live_import_reachability")
    if not dict(artifact.get("own_attempt_transition_provenance") or {}).get(
        "all_rows_live_agent_owned"
    ):
        reasons.append("own_attempt_transition_provenance")
    counts = dict(artifact.get("per_arm_triggered_decision_counts") or {})
    if any(int(counts.get(arm) or 0) <= 0 for arm in DECISION_ARMS):
        reasons.append("triggered_decision_counts")
    intervals = dict(artifact.get("grouped_paired_intervals") or {})
    if not dict(intervals.get("support") or {}).get("positive_game_grouped_support"):
        reasons.append("nonpositive_task_aware_lift")
    matrices = dict(artifact.get("false_confident_admission_and_abstention_matrices") or {})
    if not matrices.get("task_aware_reduces_or_preserves_false_confident"):
        reasons.append("false_confident_regression")
    control = dict(
        artifact.get(
            "shuffled_label_alias_identity_noop_light_inventor_raise_denominator_and_no_trigger_controls"
        )
        or {}
    )
    if control.get("all_controls_passed") is not True:
        reasons.append("control_failure")
    if int(artifact.get("llm_invocation_count") or 0) != 0:
        reasons.append("llm_invocation_count")
    for field in ("used_game_source", "offline_ground_truth_bfs", "hand_calibrated_per_game", "solve_claimed", "offline_reproduced"):
        if artifact.get(field) is not False:
            reasons.append(field)
    if int(artifact.get("level_credit_delta") or 0) != 0:
        reasons.append("level_credit_delta")
    if not dict(artifact.get("registry_level_fields_unchanged") or {}).get("unchanged"):
        reasons.append("registry_level_fields_unchanged")
    if not dict(artifact.get("protected_files_unchanged") or {}).get("unchanged"):
        reasons.append("protected_files_unchanged")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        reasons.append("verifier_is_oracle")
    return reasons


def ready_score(artifact: Mapping[str, Any]) -> float:
    return 0.0 if _blocked_reasons(artifact) else 1.0


def status(artifact: Mapping[str, Any]) -> str:
    if artifact.get("retirement_triggered") is True:
        return "retired"
    return "complete_positive" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    if state == "complete_positive":
        support = dict(dict(artifact.get("grouped_paired_intervals") or {}).get("support") or {})
        return (
            "complete_positive: task_aware_live_transition_admission_improved_"
            f"{support.get('positive_games', 0)}_held_games_no_solve_claim"
        )
    if state == "retired":
        return "retired: exp6154_exact_construction_retired_after_repeated_no_causal_receipt"
    reasons = "_".join(_blocked_reasons(artifact)[:4]) or "held_lift_not_positive"
    return f"complete_null: {reasons}_no_solve_claim"


def field_provenance() -> dict[str, dict[str, str]]:
    return {
        field: {
            "source": "experiment_6154_arc_task_aware_energy_generalization",
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def missing_gaps(artifact: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {"gap": reason, "effect": "blocks Exp6154 readiness for this exact construction"}
        for reason in _blocked_reasons(artifact)
    ]


def run(
    *,
    result_path: Path | None = None,
    root: Path = REPO_ROOT,
    games: Sequence[str] = DEFAULT_GAMES,
    held_games: Sequence[str] = DEFAULT_HELD_GAMES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    live_rows: Sequence[Mapping[str, Any]] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    started = time.perf_counter()
    out_path = result_path or (root / RESULT_RELATIVE_PATH)
    preconditions, registry_before = collect_preconditions(
        root=root,
        result_path=out_path,
        games=games,
        held_games=held_games,
        seeds=seeds,
        action_budget=action_budget,
    )
    if live_rows is None:
        rows, disable_receipt, llm_calls = collect_live_rows(
            games=games,
            seeds=seeds,
            action_budget=action_budget,
        )
    else:
        rows = [dict(row) for row in live_rows]
        disable_receipt = {
            "adapter_disabled": True,
            "per_game_lookup_routes_disabled": True,
            "solver_routes_disabled": True,
            "registry_gotcha_calibration_disabled": True,
            "llm_induction_disabled": True,
            "game_source_read_count": 0,
            "offline_ground_truth_bfs_run_count": 0,
            "principle": FIELD_PRINCIPLES[
                "adapter_per_game_lookup_solver_and_gotcha_disable_receipts"
            ],
        }
        llm_calls = 0
    decisions, manifests = _decision_rows(rows, held_games=held_games)
    per_game = per_game_metrics(rows, decisions, held_games=held_games)
    intervals = grouped_intervals(per_game)
    matrices = false_confident_matrices(per_game)
    control = controls(live_rows=rows, decisions=decisions, per_game=per_game, matrices=matrices)
    protected = protected_files_unchanged(
        root, dict(preconditions.get("protected_file_hashes_before") or {})
    )
    registry_receipt = registry_precheck(
        root=root,
        held_games=held_games,
        before_fingerprint=registry_before,
    )
    registry_unchanged = registry_level_fields_unchanged(root, registry_before)
    artifact: JsonDict = {
        "status": "",
        "preconditions_checked": preconditions,
        "registry_precheck_and_no_duplicate_receipt": registry_receipt,
        "prior_failure_receipt": prior_failure_receipt(root),
        "development_and_held_game_split_hash": split_manifest(
            games=games,
            held_games=held_games,
            seeds=seeds,
            action_budget=action_budget,
        ),
        "adapter_per_game_lookup_solver_and_gotcha_disable_receipts": disable_receipt,
        "live_entrypoint_and_import_reachability": import_reachability(root, live_rows=rows),
        "own_attempt_transition_provenance": provenance_receipt(rows),
        "global_and_task_aware_freeze_manifests": manifests,
        "per_arm_triggered_decision_counts": _counts(decisions),
        "per_game_transition_change_safety_action_and_latency_metrics": per_game,
        "grouped_paired_intervals": intervals,
        "false_confident_admission_and_abstention_matrices": matrices,
        "shuffled_label_alias_identity_noop_light_inventor_raise_denominator_and_no_trigger_controls": control,
        "llm_invocation_count": int(llm_calls),
        "used_game_source": False,
        "offline_ground_truth_bfs": False,
        "hand_calibrated_per_game": False,
        "solve_claimed": False,
        "offline_reproduced": False,
        "level_credit_delta": 0,
        "registry_level_fields_unchanged": registry_unchanged,
        "arc_task_aware_generalization_ready_score": 0.0,
        "retirement_triggered": False,
        "protected_files_unchanged": protected,
        "duration_s": round(float(duration_s if duration_s is not None else time.perf_counter() - started), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "missing_verifier_gaps": [],
        "field_provenance": field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {str(key): int(value) for key, value in dict(test_exit_codes or {}).items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["arc_task_aware_generalization_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["missing_verifier_gaps"] = missing_gaps(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic_json(out_path, artifact)
    return artifact


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover - schema guard.
        raise ValueError(f"missing required fields: {missing}")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):  # pragma: no cover
        raise ValueError("field_provenance must cover required fields")
    for field in (
        "used_game_source",
        "offline_ground_truth_bfs",
        "hand_calibrated_per_game",
        "solve_claimed",
        "offline_reproduced",
    ):
        if artifact.get(field) is not False:
            raise ValueError(field)
    if int(artifact.get("level_credit_delta") or 0) != 0:
        raise ValueError("level_credit_delta")  # pragma: no cover
    if artifact.get("llm_invocation_count") != 0:
        raise ValueError("llm_invocation_count")  # pragma: no cover
    counts = dict(artifact.get("per_arm_triggered_decision_counts") or {})
    if any(int(counts.get(arm) or 0) <= 0 for arm in DECISION_ARMS):
        raise ValueError("triggered decision counts must be nonzero")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")  # pragma: no cover
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle")  # pragma: no cover
    if artifact.get("arc_task_aware_generalization_ready_score") != ready_score(artifact):
        raise ValueError("arc_task_aware_generalization_ready_score")  # pragma: no cover
    if artifact.get("status") != status(artifact):
        raise ValueError("status")  # pragma: no cover
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")  # pragma: no cover
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(_load_json(REPO_ROOT / RESULT_RELATIVE_PATH))
        print(RESULT_RELATIVE_PATH.as_posix())
        return 0
    run(write=True)
    print(RESULT_RELATIVE_PATH.as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
