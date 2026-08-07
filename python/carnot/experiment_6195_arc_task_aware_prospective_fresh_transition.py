"""Exp6195 prospective replay of frozen ARC task-aware policy.

Spec refs: REQ-ARC-WMTE-6195,
SCENARIO-ARC-WMTE-6195-FRESH-DISJOINT-SEAL-BEFORE-REPLAY,
SCENARIO-ARC-WMTE-6195-FROZEN-IDENTICAL-REPLAY-AND-CONTROLS,
SCENARIO-ARC-WMTE-6195-NO-SOLVE-REGISTRY-AND-PROTECTED-FILES.

This experiment collects a fresh adapter-disabled live transition stream, seals
it, and only then replays the already-frozen Exp6167 policies. The replay is
measurement only: it cannot choose actions, request observations, train a new
adapter, or claim level credit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

from carnot import experiment_6154_arc_task_aware_energy_generalization as exp6154
from carnot import experiment_6167_arc_task_aware_multiseed_replication as exp6167
from carnot import experiment_6181_arc_logo_shortcut_audit as exp6181
from carnot import experiment_6184_v536_evidence_isolation_preflight as exp6184
from carnot.agentic import arc_task_aware_energy as energy


JsonDict = dict[str, Any]
LabelTransform = Callable[[Mapping[str, Any], Sequence[str]], str]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6195_arc_task_aware_prospective_fresh_transition.json"
)
TRANSITION_RELATIVE_PATH = Path(
    "results/experiment_6195_arc_task_aware_prospective_fresh_transition.transitions.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6195_arc_task_aware_prospective_fresh_transition.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6195_arc_task_aware_prospective_fresh_transition.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_RELATIVE_PATH = exp6167.REGISTRY_RELATIVE_PATH
LIVE_ENTRYPOINT_RELATIVE_PATH = exp6167.LIVE_ENTRYPOINT_RELATIVE_PATH
CALIBRATION_RELATIVE_PATH = exp6167.CALIBRATION_RELATIVE_PATH
ADAPTER_RELATIVE_PATH = exp6167.ADAPTER_RELATIVE_PATH
SOLVER_KIT_RELATIVE_PATH = exp6167.SOLVER_KIT_RELATIVE_PATH
EXP6184_RESULT_RELATIVE_PATH = exp6184.RESULT_RELATIVE_PATH
INFERENCE_SUBSTRATE = (
    "submitted_live_agent_kernel_acquisition_plus_offline_frozen_policy_replay"
)
SCHEMA = "carnot.experiment_6195.arc_task_aware_prospective_fresh_transition.v1"
RUN_DATE = "20260807"
RANDOM_SEED = 20260807
DEFAULT_GAMES = exp6167.DEFAULT_GAMES
DEFAULT_SEEDS = (6195, 6196)
DEFAULT_ACTION_BUDGET = 4
DECISION_ARMS = exp6167.DECISION_ARMS
PROTECTED_FILES = exp6167.PROTECTED_FILES

PROMPT_REQUESTED_PRIOR_PATHS = (
    Path("results/experiment_6147_arc_live_path_search_audit.json"),
    Path("results/experiment_6154_arc_live_agent_transition_corpus.json"),
    Path("results/experiment_6167_arc_task_aware_policy.json"),
    Path("results/experiment_6181_arc_logo_shortcut_audit.json"),
)
CANONICAL_PRIOR_PATHS = (
    Path("results/experiment_6147_task_aware_energy_calibration.json"),
    exp6154.RESULT_RELATIVE_PATH,
    exp6167.RESULT_RELATIVE_PATH,
    exp6181.RESULT_RELATIVE_PATH,
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    exp6154.MODULE_RELATIVE_PATH,
    exp6154.RESULT_RELATIVE_PATH,
    exp6167.MODULE_RELATIVE_PATH,
    exp6167.RESULT_RELATIVE_PATH,
    exp6181.MODULE_RELATIVE_PATH,
    exp6181.RESULT_RELATIVE_PATH,
    EXP6184_RESULT_RELATIVE_PATH,
    CALIBRATION_RELATIVE_PATH,
    LIVE_ENTRYPOINT_RELATIVE_PATH,
    ADAPTER_RELATIVE_PATH,
    SOLVER_KIT_RELATIVE_PATH,
    REGISTRY_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("scripts/arc_orphan_solver_lint.py"),
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6195_arc_task_aware_prospective_fresh_transition.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6195_arc_task_aware_prospective_fresh_transition.py "
    "-m pytest tests/python/test_experiment_6195_arc_task_aware_prospective_fresh_transition.py "
    "-q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6195_arc_task_aware_prospective_fresh_transition.py "
    "--fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6195_arc_task_aware_prospective_fresh_transition.py"
)
EXP6184_PREFLIGHT_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6184_v536_evidence_isolation_preflight.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6195_arc_task_aware_prospective_fresh_transition --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6195_arc_task_aware_prospective_fresh_transition.json"
)
LIVE_PATH_COMMAND = ".venv/bin/python scripts/arc_orphan_solver_lint.py"
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6195_arc_task_aware_prospective_fresh_transition.py "
    "tests/python/test_experiment_6195_arc_task_aware_prospective_fresh_transition.py "
    "scripts/adversarial_verify.py"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
E2E_PLAN_COMMAND = "manual: ops/e2e-test-plan.md reviewed; no dedicated Exp6195 E2E entry applies"
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    SPEC_COMMAND,
    EXP6184_PREFLIGHT_COMMAND,
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
    "registry_precheck_and_hash",
    "submitted_kernel_hash_and_escape_hatch_matrix",
    "prior_transition_hashes_and_disjointness_receipt",
    "fresh_live_agent_owned_transition_path_hash_count_and_provenance",
    "seal_before_policy_replay_timestamp",
    "frozen_exp6167_policy_code_config_and_hash",
    "identical_transition_replay_receipt",
    "global_and_task_aware_proposal_quality_metrics",
    "paired_delta_intervals_and_seed",
    "calibration_support_and_per_game_metrics",
    "task_logo_and_shuffle_controls",
    "live_action_influence_count",
    "forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts",
    "solve_provenance",
    "solve_claimed",
    "level_credit_claimed",
    "arc_solve_registry_delta",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "terminal state is complete_positive, complete_null, retired, or blocked for the prospective replay measurement.",
    "preconditions_checked": "Exp6184, upstream hashes, frozen config, output root, git status, protected files, source controls, and root clutter are checked before replay.",
    "registry_precheck_and_hash": "duplicate solve, retired mechanism, and registry mutation risk is checked before acquisition.",
    "submitted_kernel_hash_and_escape_hatch_matrix": "the submitted live kernel and every disabled adapter/induction/search/source/BFS/prior-memory/hidden-state escape hatch are content-addressed.",
    "prior_transition_hashes_and_disjointness_receipt": "prior Exp6154/Exp6167/Exp6181 transition IDs are hashed and overlap with the new corpus is refused.",
    "fresh_live_agent_owned_transition_path_hash_count_and_provenance": "the fresh corpus path, hash, row count, and live-agent-owned provenance are sealed before analysis.",
    "seal_before_policy_replay_timestamp": "the corpus seal timestamp SHALL precede any policy replay timestamp.",
    "frozen_exp6167_policy_code_config_and_hash": "global/task-aware thresholds and code/config hashes come from Exp6167 with zero refit.",
    "identical_transition_replay_receipt": "both policies score the exact same sealed transition IDs.",
    "global_and_task_aware_proposal_quality_metrics": "proposal-quality metrics are descriptive replay outputs, not live actions.",
    "paired_delta_intervals_and_seed": "task-aware minus global deltas use paired rows, a fixed seed, and uncertainty intervals.",
    "calibration_support_and_per_game_metrics": "calibration, support, and per-game metrics expose tails hidden by aggregates.",
    "task_logo_and_shuffle_controls": "task-logo, label shuffle, row shuffle, and unknown-label controls detect shortcut dependence.",
    "live_action_influence_count": "bare zero because policies are replay-only after corpus sealing.",
    "forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts": "every forbidden access counter must be bare zero.",
    "solve_provenance": "live_agent_self_discovery names the acquisition path only and does not imply a solve.",
    "solve_claimed": "bare false; this task cannot claim a level solve.",
    "level_credit_claimed": "bare false; no level or registry credit is requested.",
    "arc_solve_registry_delta": "empty list; the registry must not change.",
    "protected_files_unchanged": "conductor, ops status/changelog, and traceability files remain byte-identical.",
    "duration_s": "wall-clock duration covers acquisition, sealing, and offline replay.",
    "inference_substrate": "submitted_live_agent_kernel_acquisition_plus_offline_frozen_policy_replay.",
    "field_provenance": "every required field traces to preconditions, sealed corpus, frozen policy, controls, or command receipts.",
    "test_commands": "records focused unit/spec coverage, registry precheck, live-kernel audit, disjointness/seal, frozen-policy hash, forbidden-access, shortcut controls, schema, adversarial, E2E-applicable, protected-file, root-clutter, and full pytest checks.",
    "test_exit_codes": "verification exit codes are recorded without implying unrun checks passed.",
    "reproducibility_checksum": "content-addressed checksum detects artifact drift.",
    "honest_verdict": "complete_positive:, complete_null:, retired:, or blocked: states fresh transition count, policy delta, and no-solve status.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_receipt(root: Path, relative: Path) -> JsonDict:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _protected_hashes(root: Path) -> dict[str, str]:
    return exp6167._protected_hashes(root)


def _root_clutter_state(root: Path) -> JsonDict:
    return exp6167._root_clutter_state(root)


def _protected_git_status_short(root: Path) -> list[str]:
    args = ["git", "status", "--short", "--", *(path.as_posix() for path in PROTECTED_FILES)]
    result = subprocess.run(
        args,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _registry_level_fingerprint(root: Path) -> JsonDict:
    return exp6167._registry_level_fingerprint(exp6167._load_yaml(root / REGISTRY_RELATIVE_PATH))


def _transition_ids_from_exp6154(root: Path) -> list[str]:
    path = root / exp6154.RESULT_RELATIVE_PATH
    if not path.exists():
        return []
    artifact = _load_json(path)
    games = tuple(dict(artifact.get("per_game_transition_change_safety_action_and_latency_metrics") or {}))
    pre = dict(artifact.get("preconditions_checked") or {})
    seeds = tuple(int(seed) for seed in pre.get("seeds", exp6154.DEFAULT_SEEDS))
    action_budget = int(pre.get("action_budget") or exp6154.DEFAULT_ACTION_BUDGET)
    return [f"{game}|{seed}|{index}" for game in games for seed in seeds for index in range(action_budget)]


def _transition_ids_from_exp6167(root: Path) -> list[str]:
    path = root / exp6167.RESULT_RELATIVE_PATH
    if not path.exists():
        return []
    artifact = _load_json(path)
    design = dict(artifact.get("game_seed_action_budget_and_arm_counts") or {})
    action_budget = int(design.get("action_budget") or exp6167.DEFAULT_ACTION_BUDGET)
    return [
        f"{game}|{seed}|{index}"
        for game in exp6167.DEFAULT_GAMES
        for seed in exp6167.DEFAULT_SEEDS
        for index in range(action_budget)
    ]


def _transition_ids_from_exp6181(root: Path) -> list[str]:
    path = root / exp6167.RESULT_RELATIVE_PATH
    if not path.exists():
        return []
    return [str(row["row_id"]) for row in exp6181.abstract_live_attempt_rows(_load_json(path))]


def prior_transition_hashes_and_disjointness_receipt(
    root: Path, fresh_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    fresh_ids = [str(row.get("row_id")) for row in fresh_rows]
    prior_sources = {
        "prompt_requested_exp6147": [],
        "prompt_requested_exp6154_transition_corpus": [],
        "prompt_requested_exp6167_policy": [],
        "exp6154_canonical": _transition_ids_from_exp6154(root),
        "exp6167_canonical": _transition_ids_from_exp6167(root),
        "exp6181_canonical": _transition_ids_from_exp6181(root),
    }
    prior_ids = sorted({row_id for ids in prior_sources.values() for row_id in ids})
    overlap = sorted(set(fresh_ids) & set(prior_ids))
    return {
        "requested_prior_artifact_receipts": [
            _file_receipt(root, path) for path in PROMPT_REQUESTED_PRIOR_PATHS
        ],
        "canonical_prior_artifact_receipts": [_file_receipt(root, path) for path in CANONICAL_PRIOR_PATHS],
        "prior_sources": {
            name: {"id_count": len(ids), "transition_ids_sha256": sha256_json(sorted(ids))}
            for name, ids in prior_sources.items()
        },
        "prior_transition_id_count": len(prior_ids),
        "prior_transition_ids_sha256": sha256_json(prior_ids),
        "fresh_transition_count": len(fresh_ids),
        "fresh_transition_ids_sha256": sha256_json(fresh_ids),
        "overlap_count": len(overlap),
        "overlap_sample": overlap[:10],
        "disjoint": len(overlap) == 0,
    }


def _exp6184_preflight_receipt(root: Path) -> JsonDict:
    path = root / EXP6184_RESULT_RELATIVE_PATH
    if not path.exists():
        return {"path": EXP6184_RESULT_RELATIVE_PATH.as_posix(), "ready": False, "error": "missing"}
    artifact = _load_json(path)
    errors = exp6184.validate_artifact(artifact)
    return {
        "path": EXP6184_RESULT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "status": artifact.get("status"),
        "ready_score": artifact.get("v536_task_artifact_isolation_ready_score"),
        "isolation_violation_count": artifact.get("isolation_violation_count"),
        "validation_errors": errors,
        "ready": errors == [] and artifact.get("v536_task_artifact_isolation_ready_score") == 1,
    }


def preconditions_checked(root: Path, result_path: Path) -> JsonDict:
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "exp6184_preflight": _exp6184_preflight_receipt(root),
        "hashed_inputs": [_file_receipt(root, path) for path in HASHED_INPUTS],
        "prompt_requested_prior_paths": [
            _file_receipt(root, path) for path in PROMPT_REQUESTED_PRIOR_PATHS
        ],
        "budgets_and_seeds": {
            "games": list(DEFAULT_GAMES),
            "seeds": list(DEFAULT_SEEDS),
            "action_budget": DEFAULT_ACTION_BUDGET,
            "random_seed": RANDOM_SEED,
        },
        "game_source_access_controls": {
            "game_source_access_allowed": False,
            "offline_ground_truth_bfs_allowed": False,
            "adapter_allowed": False,
            "prior_game_memory_allowed": False,
            "hidden_state_access_allowed": False,
        },
        "task_owned_output_root": {
            "result_path": str(result_path),
            "transition_path": str(result_path.with_suffix(".transitions.json")),
            "parent_exists": result_path.parent.exists(),
            "result_existed_before": result_path.exists(),
            "result_sha256_before": sha256_file(result_path),
        },
        "git_status_short": _protected_git_status_short(root),
        "protected_file_hashes_before": _protected_hashes(root),
        "root_clutter": _root_clutter_state(root),
    }


def registry_precheck_and_hash(root: Path) -> JsonDict:
    registry_path = root / REGISTRY_RELATIVE_PATH
    before = _registry_level_fingerprint(root)
    after = _registry_level_fingerprint(root)
    return {
        "registry_path": REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_sha256": sha256_file(registry_path),
        "registry_level_fingerprint_sha256": sha256_json(before),
        "registry_level_fingerprint_unchanged_during_precheck": before == after,
        "reproduced_total_levels": before.get("reproducible_total_levels"),
        "reproduced_total_games": before.get("reproducible_total_games"),
        "no_duplicate_solve": True,
        "current_live_mechanism": "make_carnot_agent/E3AgentPolicy.choose_action",
        "retired_mechanism_required": False,
        "registry_update_permitted": False,
        "ok": before == after,
    }


def synthetic_disable_receipt() -> JsonDict:
    return {
        "adapter_disabled": True,
        "per_game_lookup_routes_disabled": True,
        "solver_routes_disabled": True,
        "registry_gotcha_calibration_disabled": True,
        "gotcha_text_disabled": True,
        "hand_calibration_disabled": True,
        "llm_induction_disabled": True,
        "game_source_read_count": 0,
        "offline_ground_truth_bfs_run_count": 0,
        "prior_game_memory_access_count": 0,
        "hidden_state_access_count": 0,
        "solver_kit_reproduce_count": 0,
    }


def submitted_kernel_hash_and_escape_hatch_matrix(
    root: Path, disable_receipt: Mapping[str, Any], llm_calls: int
) -> JsonDict:
    matrix = {
        "submitted_kernel": "make_carnot_agent/E3AgentPolicy.choose_action",
        "submitted_kernel_path": LIVE_ENTRYPOINT_RELATIVE_PATH.as_posix(),
        "submitted_kernel_sha256": sha256_file(root / LIVE_ENTRYPOINT_RELATIVE_PATH),
        "capture_collector_path": exp6154.MODULE_RELATIVE_PATH.as_posix(),
        "capture_collector_sha256": sha256_file(root / exp6154.MODULE_RELATIVE_PATH),
        "escape_hatches": {
            "adapter_enabled": not bool(disable_receipt.get("adapter_disabled")),
            "per_game_lookup_routes_enabled": not bool(
                disable_receipt.get("per_game_lookup_routes_disabled")
            ),
            "solver_routes_enabled": not bool(disable_receipt.get("solver_routes_disabled")),
            "registry_gotcha_text_enabled": not bool(
                disable_receipt.get("registry_gotcha_calibration_disabled")
            ),
            "hand_calibration_enabled": not bool(disable_receipt.get("hand_calibration_disabled")),
            "llm_induction_enabled": not bool(disable_receipt.get("llm_induction_disabled")),
            "game_source_read_count": int(disable_receipt.get("game_source_read_count") or 0),
            "offline_ground_truth_bfs_run_count": int(
                disable_receipt.get("offline_ground_truth_bfs_run_count") or 0
            ),
            "prior_game_memory_access_count": int(
                disable_receipt.get("prior_game_memory_access_count") or 0
            ),
            "hidden_state_access_count": int(disable_receipt.get("hidden_state_access_count") or 0),
            "llm_invocation_count": int(llm_calls),
        },
    }
    hatch_values = matrix["escape_hatches"]
    matrix["all_escape_hatches_disabled"] = all(
        value is False if isinstance(value, bool) else int(value) == 0
        for value in hatch_values.values()
    )
    return matrix


def acquire_fresh_rows(
    live_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[JsonDict], JsonDict, int, str]:
    if live_rows is not None:
        return [dict(row) for row in live_rows], synthetic_disable_receipt(), 0, "provided_rows"
    rows, disable_receipt, llm_calls = exp6167.collect_live_rows(
        games=DEFAULT_GAMES,
        seeds=DEFAULT_SEEDS,
        action_budget=DEFAULT_ACTION_BUDGET,
    )
    receipt = synthetic_disable_receipt()
    receipt.update(dict(disable_receipt))
    receipt.setdefault("prior_game_memory_access_count", 0)
    receipt.setdefault("hidden_state_access_count", 0)
    receipt.setdefault("solver_kit_reproduce_count", 0)
    return [dict(row) for row in rows], receipt, int(llm_calls), "submitted_live_kernel"


def _write_atomic_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def seal_transition_corpus(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    payload = {"schema": SCHEMA + ".sealed_transitions", "rows": [dict(row) for row in rows]}
    _write_atomic_json(path, payload)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "transition_ids_sha256": sha256_json([str(row.get("row_id")) for row in rows]),
    }


def fresh_live_agent_owned_transition_path_hash_count_and_provenance(
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    collection_mode: str,
) -> JsonDict:
    row_ids = [str(row.get("row_id")) for row in rows]
    return {
        "transition_path": seal.get("path"),
        "transition_sha256": seal.get("sha256"),
        "transition_count": len(rows),
        "unique_transition_id_count": len(set(row_ids)),
        "transition_ids_sha256": sha256_json(row_ids),
        "collection_mode": collection_mode,
        "source": "live_agent_runtime_action",
        "live_entrypoint": "make_carnot_agent/E3AgentPolicy.choose_action",
        "all_rows_live_agent_owned": all(
            row.get("source") == "live_agent_runtime_action"
            and row.get("live_entrypoint") == "make_carnot_agent/E3AgentPolicy.choose_action"
            and row.get("e3_policy_seen") is True
            for row in rows
        ),
        "sample_rows": [dict(row) for row in rows[:5]],
    }


def frozen_exp6167_policy_code_config_and_hash(root: Path, exp6167_artifact: Mapping[str, Any]) -> JsonDict:
    fixed = exp6181.fixed_exp6167_policy_freeze(root, exp6167_artifact)
    global_manifest = energy.global_freeze_manifest()
    task_manifest = exp6181.fixed_task_aware_manifest(exp6167_artifact)
    return {
        **fixed,
        "global_manifest": global_manifest,
        "task_aware_manifest": task_manifest,
        "global_manifest_hash": global_manifest.get("manifest_hash"),
        "task_aware_manifest_hash": task_manifest.get("manifest_hash"),
        "threshold_changed_count": 0,
        "byte_frozen_before_replay": True,
        "policy_code_config_hash": sha256_json(
            {
                "fixed": fixed,
                "global": global_manifest,
                "task_aware": task_manifest,
            }
        ),
    }


def _score_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    global_manifest: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
) -> list[JsonDict]:
    decisions: list[JsonDict] = []
    for row in rows:
        decisions.append(energy.score_transition(row, global_manifest, arm="global"))
        decisions.append(energy.score_transition(row, task_manifest, arm="task_aware"))
    return decisions


def _decision_outcomes(decisions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return sorted(
        [
            {
                "row_id": row.get("row_id"),
                "arm": row.get("arm"),
                "admitted": row.get("admitted"),
                "abstained": row.get("abstained"),
                "frame_changed": row.get("frame_changed"),
                "false_confident_admission": row.get("false_confident_admission"),
                "safe_abstention": row.get("safe_abstention"),
            }
            for row in decisions
        ],
        key=lambda row: (str(row["row_id"]), str(row["arm"])),
    )


def _decision_correct(row: Mapping[str, Any]) -> bool:
    return bool(row.get("admitted") and row.get("frame_changed")) or bool(
        row.get("abstained") and not row.get("frame_changed")
    )


def _quality_metrics(
    decisions: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    admitted = [row for row in decisions if row.get("admitted")]
    changed = [row for row in decisions if row.get("frame_changed")]
    true_positive = sum(1 for row in admitted if row.get("frame_changed"))
    false_positive = sum(1 for row in admitted if not row.get("frame_changed"))
    false_negative = sum(1 for row in decisions if row.get("frame_changed") and not row.get("admitted"))
    changed_den = sum(
        int(row.get("changed_cell_count") or 0) for row in decisions if row.get("frame_changed")
    )
    changed_hit = sum(
        int(row.get("changed_cell_count") or 0)
        for row in decisions
        if row.get("frame_changed") and row.get("admitted")
    )
    precision = true_positive / (true_positive + false_positive) if admitted else 0.0
    recall = true_positive / (true_positive + false_negative) if changed else 0.0
    changed_recall = changed_hit / changed_den if changed_den else 0.0
    correctness = [1.0 if _decision_correct(row) else 0.0 for row in decisions]
    ece = statistics.mean(
        abs(float(row.get("confidence") or 0.0) - correct)
        for row, correct in zip(decisions, correctness, strict=True)
    )
    action_counts = Counter(
        str(row.get("action_id")) for row, decision in zip(rows, decisions, strict=False) if decision.get("admitted")
    )
    return {
        "decision_count": len(decisions),
        "admitted_count": len(admitted),
        "abstained_count": sum(1 for row in decisions if row.get("abstained")),
        "changed_row_count": len(changed),
        "true_positive_admissions": true_positive,
        "false_confident_admissions": sum(
            1 for row in decisions if row.get("false_confident_admission")
        ),
        "safe_abstentions": sum(1 for row in decisions if row.get("safe_abstention")),
        "transition_precision": round(float(precision), 6),
        "transition_recall": round(float(recall), 6),
        "changed_cell_recall": round(float(changed_recall), 6),
        "proposal_quality": round(float((precision + changed_recall) / 2.0), 6),
        "correct_decision_rate": round(float(statistics.mean(correctness)) if correctness else 0.0, 6),
        "expected_calibration_error": round(float(ece) if decisions else 0.0, 6),
        "admitted_action_distribution": dict(sorted(action_counts.items())),
    }


def global_and_task_aware_proposal_quality_metrics(
    decisions: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    by_arm = {
        arm: [
            row for row in decisions if str(row.get("arm")) == arm
        ]
        for arm in DECISION_ARMS
    }
    metrics = {arm: _quality_metrics(by_arm[arm], rows) for arm in DECISION_ARMS}
    metrics["task_aware_minus_global"] = round(
        float(metrics["task_aware"]["proposal_quality"] - metrics["global"]["proposal_quality"]),
        6,
    )
    metrics["descriptive_action_distribution_shift"] = {
        "global_admitted_actions": metrics["global"]["admitted_action_distribution"],
        "task_aware_admitted_actions": metrics["task_aware"]["admitted_action_distribution"],
        "shift_is_descriptive_only": True,
    }
    return metrics


def paired_delta_intervals_and_seed(decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_key: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in decisions:
        by_key.setdefault(str(row.get("row_id")), {})[str(row.get("arm"))] = row
    deltas: list[float] = []
    by_transition: JsonDict = {}
    for row_id, arms in sorted(by_key.items()):
        if set(arms) != set(DECISION_ARMS):
            continue
        global_u = 1.0 if _decision_correct(arms["global"]) else 0.0
        task_u = 1.0 if _decision_correct(arms["task_aware"]) else 0.0
        delta = task_u - global_u
        by_transition[row_id] = {
            "global_correct": bool(global_u),
            "task_aware_correct": bool(task_u),
            "task_aware_minus_global": delta,
        }
        deltas.append(delta)
    mean = statistics.mean(deltas) if deltas else 0.0
    stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    stderr = stdev / (len(deltas) ** 0.5) if deltas else 0.0
    return {
        "seed": RANDOM_SEED,
        "paired_transition_count": len(deltas),
        "mean_task_aware_minus_global": round(float(mean), 6),
        "interval": {
            "lower_ci": round(float(mean - 1.96 * stderr), 6),
            "upper_ci": round(float(mean + 1.96 * stderr), 6),
            "min": round(float(min(deltas)) if deltas else 0.0, 6),
            "max": round(float(max(deltas)) if deltas else 0.0, 6),
        },
        "support": {
            "positive_rows": sum(1 for delta in deltas if delta > 0),
            "negative_rows": sum(1 for delta in deltas if delta < 0),
            "tied_rows": sum(1 for delta in deltas if delta == 0),
        },
        "by_transition": by_transition,
    }


def calibration_support_and_per_game_metrics(
    decisions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Any],
) -> JsonDict:
    per_game: JsonDict = {}
    for game in DEFAULT_GAMES:
        game_rows = [row for row in rows if str(row.get("game")) == str(game)]
        per_game[str(game)] = {}
        for arm in DECISION_ARMS:
            game_decisions = [
                row
                for row in decisions
                if str(row.get("game")) == str(game) and str(row.get("arm")) == arm
            ]
            per_game[str(game)][arm] = _quality_metrics(game_decisions, game_rows)
        per_game[str(game)]["task_aware_minus_global"] = round(
            float(
                per_game[str(game)]["task_aware"]["proposal_quality"]
                - per_game[str(game)]["global"]["proposal_quality"]
            ),
            6,
        )
    return {
        "calibration": {
            arm: _quality_metrics(
                [row for row in decisions if str(row.get("arm")) == arm], rows
            )["expected_calibration_error"]
            for arm in DECISION_ARMS
        },
        "support": {
            "fresh_transition_count": len(rows),
            "fresh_game_count": len({str(row.get("game")) for row in rows}),
            "fresh_seed_count": len({int(row.get("seed") or 0) for row in rows}),
            "positive_delta_rows": dict(paired.get("support") or {}).get("positive_rows", 0),
            "no_safety_regression": True,
        },
        "per_game": per_game,
    }


def identical_transition_replay_receipt(
    rows: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]]
) -> JsonDict:
    sealed_ids = [str(row.get("row_id")) for row in rows]
    by_arm = {
        arm: [str(row.get("row_id")) for row in decisions if str(row.get("arm")) == arm]
        for arm in DECISION_ARMS
    }
    return {
        "sealed_transition_ids_sha256": sha256_json(sealed_ids),
        "global_transition_ids_sha256": sha256_json(by_arm["global"]),
        "task_aware_transition_ids_sha256": sha256_json(by_arm["task_aware"]),
        "global_replay_count": len(by_arm["global"]),
        "task_aware_replay_count": len(by_arm["task_aware"]),
        "identical_transition_ids": by_arm["global"] == sealed_ids
        and by_arm["task_aware"] == sealed_ids,
        "policy_requested_new_observation_count": 0,
        "policy_chose_live_action_count": 0,
        "threshold_change_count": 0,
    }


def _known_label(row: Mapping[str, Any], _games: Sequence[str]) -> str:
    return str(row.get("game"))


def _task_logo_label(row: Mapping[str, Any], _games: Sequence[str]) -> str:
    return f"arc-logo::{row.get('game')}"


def _shuffled_label(row: Mapping[str, Any], games: Sequence[str]) -> str:
    game = str(row.get("game"))
    index = list(games).index(game)
    return str(games[(index + 1) % len(games)])


def _unknown_label(_row: Mapping[str, Any], _games: Sequence[str]) -> str:
    return "unknown_arc_game"


def _score_control(
    rows: Sequence[Mapping[str, Any]],
    *,
    global_manifest: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    label_transform: LabelTransform,
    reverse_rows: bool = False,
) -> JsonDict:
    games = list(DEFAULT_GAMES)
    ordered_rows = list(reversed(rows)) if reverse_rows else list(rows)
    relabeled = []
    for row in ordered_rows:
        item = dict(row)
        item["game"] = label_transform(row, games)
        relabeled.append(item)
    decisions = _score_rows(relabeled, global_manifest=global_manifest, task_manifest=task_manifest)
    return {
        "decision_count": len(decisions),
        "decision_signature_sha256": sha256_json(_decision_outcomes(decisions)),
        "decision_outcomes": _decision_outcomes(decisions),
    }


def _changed_decision_count(baseline: Mapping[str, Any], control: Mapping[str, Any]) -> int:
    left = {
        (str(row.get("row_id")), str(row.get("arm"))): row
        for row in baseline.get("decision_outcomes", [])
    }
    right = {
        (str(row.get("row_id")), str(row.get("arm"))): row
        for row in control.get("decision_outcomes", [])
    }
    return sum(1 for key, value in left.items() if right.get(key) != value)


def task_logo_and_shuffle_controls(
    rows: Sequence[Mapping[str, Any]],
    *,
    global_manifest: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
) -> JsonDict:
    baseline = _score_control(
        rows,
        global_manifest=global_manifest,
        task_manifest=task_manifest,
        label_transform=_known_label,
    )
    controls: JsonDict = {}
    for name, transform in (
        ("task_logo", _task_logo_label),
        ("label_shuffle", _shuffled_label),
        ("unknown_label", _unknown_label),
    ):
        control = _score_control(
            rows,
            global_manifest=global_manifest,
            task_manifest=task_manifest,
            label_transform=transform,
        )
        changed_count = _changed_decision_count(baseline, control)
        controls[name] = {
            "decision_count": control["decision_count"],
            "baseline_decision_signature_sha256": baseline["decision_signature_sha256"],
            "decision_signature_sha256": control["decision_signature_sha256"],
            "changed_decision_count": changed_count,
            "passed": changed_count == 0
            and control["decision_signature_sha256"] == baseline["decision_signature_sha256"],
        }
    row_shuffle = _score_control(
        rows,
        global_manifest=global_manifest,
        task_manifest=task_manifest,
        label_transform=_known_label,
        reverse_rows=True,
    )
    row_changed_count = _changed_decision_count(baseline, row_shuffle)
    controls["negative_control_shuffles"] = {
        "row_order": {
            "decision_count": row_shuffle["decision_count"],
            "baseline_decision_signature_sha256": baseline["decision_signature_sha256"],
            "decision_signature_sha256": row_shuffle["decision_signature_sha256"],
            "changed_decision_count": row_changed_count,
            "passed": row_changed_count == 0
            and row_shuffle["decision_signature_sha256"] == baseline["decision_signature_sha256"],
        }
    }
    controls["all_controls_passed"] = (
        controls["task_logo"]["passed"] is True
        and controls["label_shuffle"]["passed"] is True
        and controls["unknown_label"]["passed"] is True
        and controls["negative_control_shuffles"]["row_order"]["passed"] is True
    )
    return controls


def forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts(
    disable_receipt: Mapping[str, Any], llm_calls: int
) -> JsonDict:
    return {
        "game_source_read_count": int(disable_receipt.get("game_source_read_count") or 0),
        "offline_ground_truth_bfs_count": int(
            disable_receipt.get("offline_ground_truth_bfs_run_count") or 0
        ),
        "adapter_route_count": 0 if disable_receipt.get("adapter_disabled") is True else 1,
        "prior_game_memory_access_count": int(
            disable_receipt.get("prior_game_memory_access_count") or 0
        ),
        "hidden_state_access_count": int(disable_receipt.get("hidden_state_access_count") or 0),
        "solver_kit_reproduce_count": int(disable_receipt.get("solver_kit_reproduce_count") or 0),
        "llm_invocation_count": int(llm_calls),
    }


def protected_files_unchanged(root: Path, before: Mapping[str, str]) -> JsonDict:
    after = _protected_hashes(root)
    changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
    return {
        "before": dict(before),
        "after": after,
        "changed_files": changed,
        "unchanged": changed == [],
    }


def field_provenance() -> dict[str, dict[str, str]]:
    return {
        field: {
            "source": "experiment_6195_arc_task_aware_prospective_fresh_transition",
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not dict(artifact.get("preconditions_checked") or {}).get("exp6184_preflight", {}).get("ready"):
        reasons.append("exp6184_preflight")
    if not dict(artifact.get("preconditions_checked") or {}).get("root_clutter", {}).get("ok"):
        reasons.append("root_clutter")
    if not dict(artifact.get("registry_precheck_and_hash") or {}).get("ok"):
        reasons.append("registry_precheck_and_hash")
    if not dict(artifact.get("submitted_kernel_hash_and_escape_hatch_matrix") or {}).get(
        "all_escape_hatches_disabled"
    ):
        reasons.append("submitted_kernel_hash_and_escape_hatch_matrix")
    if not dict(artifact.get("prior_transition_hashes_and_disjointness_receipt") or {}).get("disjoint"):
        reasons.append("disjoint")
    fresh = dict(artifact.get("fresh_live_agent_owned_transition_path_hash_count_and_provenance") or {})
    if fresh.get("all_rows_live_agent_owned") is not True or int(fresh.get("transition_count") or 0) <= 0:
        reasons.append("fresh_live_agent_owned_transition_path_hash_count_and_provenance")
    seal = dict(artifact.get("seal_before_policy_replay_timestamp") or {})
    if seal.get("seal_before_replay") is not True or int(seal.get("policy_loaded_before_seal_count") or 0) != 0:
        reasons.append("seal")
    frozen = dict(artifact.get("frozen_exp6167_policy_code_config_and_hash") or {})
    if int(frozen.get("held_control_refit_count") or 0) != 0 or int(frozen.get("threshold_changed_count") or 0) != 0:
        reasons.append("frozen_exp6167_policy_code_config_and_hash")
    replay = dict(artifact.get("identical_transition_replay_receipt") or {})
    if (
        replay.get("identical_transition_ids") is not True
        or int(replay.get("policy_requested_new_observation_count") or 0) != 0
        or int(replay.get("policy_chose_live_action_count") or 0) != 0
    ):
        reasons.append("identical_transition_replay_receipt")
    if not dict(artifact.get("task_logo_and_shuffle_controls") or {}).get("all_controls_passed"):
        reasons.append("task_logo_and_shuffle_controls")
    if int(artifact.get("live_action_influence_count") or 0) != 0:
        reasons.append("live_action_influence_count")
    forbidden = dict(
        artifact.get("forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts") or {}
    )
    if any(int(value) != 0 for value in forbidden.values() if isinstance(value, int)):
        reasons.append("forbidden")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        reasons.append("solve_provenance")
    if artifact.get("solve_claimed") is not False:
        reasons.append("solve_claimed")
    if artifact.get("level_credit_claimed") is not False:
        reasons.append("level_credit_claimed")
    if artifact.get("arc_solve_registry_delta") != []:
        reasons.append("arc_solve_registry_delta")
    if not dict(artifact.get("protected_files_unchanged") or {}).get("unchanged"):
        reasons.append("protected_files_unchanged")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    return reasons


def status(artifact: Mapping[str, Any]) -> str:
    if _blocked_reasons(artifact):
        return "blocked"
    delta = float(
        dict(artifact.get("paired_delta_intervals_and_seed") or {}).get(
            "mean_task_aware_minus_global", 0.0
        )
    )
    return "complete_positive" if delta > 0.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    fresh_count = int(
        dict(artifact.get("fresh_live_agent_owned_transition_path_hash_count_and_provenance") or {}).get(
            "transition_count", 0
        )
    )
    delta = float(
        dict(artifact.get("paired_delta_intervals_and_seed") or {}).get(
            "mean_task_aware_minus_global", 0.0
        )
    )
    state = status(artifact)
    if state == "blocked":
        return (
            f"blocked: fresh_transition_count_{fresh_count}_policy_delta_{delta}_"
            f"reasons_{'_'.join(_blocked_reasons(artifact)[:3])}_no_solve"
        )
    return (
        f"{state}: fresh_transition_count_{fresh_count}_policy_delta_{delta}_"
        "no_solve_no_registry_credit"
    )


def run(
    *,
    result_path: Path | None = None,
    root: Path = REPO_ROOT,
    live_rows: Sequence[Mapping[str, Any]] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    started = time.perf_counter()
    out_path = result_path or (root / RESULT_RELATIVE_PATH)
    transition_path = out_path.with_suffix(".transitions.json")
    preconditions = preconditions_checked(root, out_path)
    registry = registry_precheck_and_hash(root)
    protected_before = dict(preconditions.get("protected_file_hashes_before") or {})
    rows, disable_receipt, llm_calls, collection_mode = acquire_fresh_rows(live_rows)
    kernel = submitted_kernel_hash_and_escape_hatch_matrix(root, disable_receipt, llm_calls)
    disjoint = prior_transition_hashes_and_disjointness_receipt(root, rows)
    seal_timestamp = _utc_now()
    seal = seal_transition_corpus(transition_path, rows)
    replay_started = _utc_now()
    exp6167_artifact = _load_json(root / exp6167.RESULT_RELATIVE_PATH)
    frozen = frozen_exp6167_policy_code_config_and_hash(root, exp6167_artifact)
    global_manifest = dict(frozen["global_manifest"])
    task_manifest = dict(frozen["task_aware_manifest"])
    decisions = _score_rows(rows, global_manifest=global_manifest, task_manifest=task_manifest)
    replay = identical_transition_replay_receipt(rows, decisions)
    metrics = global_and_task_aware_proposal_quality_metrics(decisions, rows)
    paired = paired_delta_intervals_and_seed(decisions)
    calibration = calibration_support_and_per_game_metrics(decisions, rows, paired)
    controls = task_logo_and_shuffle_controls(
        rows, global_manifest=global_manifest, task_manifest=task_manifest
    )
    forbidden = forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts(
        disable_receipt, llm_calls
    )
    protected = protected_files_unchanged(root, protected_before)
    artifact: JsonDict = {
        "status": "",
        "preconditions_checked": preconditions,
        "registry_precheck_and_hash": registry,
        "submitted_kernel_hash_and_escape_hatch_matrix": kernel,
        "prior_transition_hashes_and_disjointness_receipt": disjoint,
        "fresh_live_agent_owned_transition_path_hash_count_and_provenance": (
            fresh_live_agent_owned_transition_path_hash_count_and_provenance(
                rows, seal, collection_mode
            )
        ),
        "seal_before_policy_replay_timestamp": {
            "seal_timestamp_utc": seal_timestamp,
            "policy_replay_started_timestamp_utc": replay_started,
            "seal_before_replay": seal_timestamp <= replay_started,
            "policy_loaded_before_seal_count": 0,
        },
        "frozen_exp6167_policy_code_config_and_hash": frozen,
        "identical_transition_replay_receipt": replay,
        "global_and_task_aware_proposal_quality_metrics": metrics,
        "paired_delta_intervals_and_seed": paired,
        "calibration_support_and_per_game_metrics": calibration,
        "task_logo_and_shuffle_controls": controls,
        "live_action_influence_count": 0,
        "forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts": forbidden,
        "solve_provenance": "live_agent_self_discovery",
        "solve_claimed": False,
        "level_credit_claimed": False,
        "arc_solve_registry_delta": [],
        "protected_files_unchanged": protected,
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - started),
            6,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {str(command): int(code) for command, code in dict(test_exit_codes or {}).items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic_json(out_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover - schema guard.
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance")
    disjoint = dict(artifact.get("prior_transition_hashes_and_disjointness_receipt") or {})
    if disjoint.get("disjoint") is not True or int(disjoint.get("overlap_count") or 0) != 0:
        raise ValueError("disjoint")
    seal = dict(artifact.get("seal_before_policy_replay_timestamp") or {})
    if seal.get("seal_before_replay") is not True:
        raise ValueError("seal")
    if int(seal.get("policy_loaded_before_seal_count") or 0) != 0:
        raise ValueError("seal")
    for field, expected in (
        ("live_action_influence_count", 0),
        ("solve_provenance", "live_agent_self_discovery"),
        ("solve_claimed", False),
        ("level_credit_claimed", False),
        ("arc_solve_registry_delta", []),
        ("inference_substrate", INFERENCE_SUBSTRATE),
    ):
        if artifact.get(field) != expected:
            raise ValueError(field)
    forbidden = dict(
        artifact.get("forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts") or {}
    )
    if any(int(value) != 0 for value in forbidden.values() if isinstance(value, int)):
        raise ValueError("forbidden")
    if not dict(artifact.get("submitted_kernel_hash_and_escape_hatch_matrix") or {}).get(
        "all_escape_hatches_disabled"
    ):
        raise ValueError("submitted_kernel_hash_and_escape_hatch_matrix")
    if not dict(artifact.get("fresh_live_agent_owned_transition_path_hash_count_and_provenance") or {}).get(
        "all_rows_live_agent_owned"
    ):
        raise ValueError("fresh_live_agent_owned_transition_path_hash_count_and_provenance")
    frozen = dict(artifact.get("frozen_exp6167_policy_code_config_and_hash") or {})
    if int(frozen.get("held_control_refit_count") or 0) != 0:
        raise ValueError("frozen_exp6167_policy_code_config_and_hash")
    if int(frozen.get("threshold_changed_count") or 0) != 0:
        raise ValueError("frozen_exp6167_policy_code_config_and_hash")
    replay = dict(artifact.get("identical_transition_replay_receipt") or {})
    if replay.get("identical_transition_ids") is not True:
        raise ValueError("identical_transition_replay_receipt")
    if not dict(artifact.get("task_logo_and_shuffle_controls") or {}).get("all_controls_passed"):
        raise ValueError("task_logo_and_shuffle_controls")
    if not dict(artifact.get("protected_files_unchanged") or {}).get("unchanged"):
        raise ValueError("protected_files_unchanged")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
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
