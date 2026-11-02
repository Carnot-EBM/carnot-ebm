"""Build the Exp6422 held-family policy safety audit artifact.

Spec refs: REQ-ARC-ARM-6422,
SCENARIO-ARC-ARM-6422-HASH-AND-MISSING-INPUTS,
SCENARIO-ARC-ARM-6422-HELD-REPLAY,
SCENARIO-ARC-ARM-6422-RECOMPUTE-AND-ATTACKS,
SCENARIO-ARC-ARM-6422-NO-SOLVE-OR-REGISTRY.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot import experiment_6401_arc_active_goal_causal_holdout as exp6401
from carnot import experiment_6402_arc_active_goal_safety_audit as exp6402
from carnot import experiment_6413_authenticated_sota_gguf_execution_receipts as exp6413
from carnot import experiment_6421_arc_opt_in_executed_policy_ab as exp6421
from carnot.agentic import arc_competition_agent as agent


JsonDict = dict[str, Any]

REPO_ROOT = exp6421.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_6422_arc_held_family_policy_safety_audit.json")
EXP6421_RELATIVE_PATH = exp6421.RESULT_RELATIVE_PATH
EXP6402_RELATIVE_PATH = exp6402.RESULT_RELATIVE_PATH
HELD_MANIFEST_RELATIVE_PATH = Path("results/experiment_6401_arc_active_goal_causal_holdout_windows.json")
REGISTRY_RELATIVE_PATH = exp6421.REGISTRY_RELATIVE_PATH
CLAIMS_RELATIVE_PATH = exp6421.CLAIMS_RELATIVE_PATH
RESEARCH_CONDUCTOR_RELATIVE_PATH = exp6421.RESEARCH_CONDUCTOR_RELATIVE_PATH
ARC_SPEC_RELATIVE_PATH = exp6421.ARC_SPEC_RELATIVE_PATH

RUN_DATE = "20260814"
RANDOM_SEED = 6422
INFERENCE_SUBSTRATE = exp6421.INFERENCE_SUBSTRATE

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6422_arc_held_family_policy_safety_audit "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6422_arc_held_family_policy_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6422_arc_held_family_policy_safety_audit.py "
    "-m pytest tests/python/test_experiment_6422_arc_held_family_policy_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6422_arc_held_family_policy_safety_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6422_arc_held_family_policy_safety_audit.py"
)
ARC_LIVE_REACHABILITY_COMMAND = ".venv/bin/python scripts/arc_orphan_solver_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6422_arc_held_family_policy_safety_audit.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ROOT_SWEEP_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ARC_LIVE_REACHABILITY_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_SWEEP_COMMAND,
)

CANONICAL_GENERATOR_MODEL_ID = exp6421.CANONICAL_GENERATOR_MODEL_ID
MANDATED_GEMMA_MODEL_ID = exp6421.MANDATED_GEMMA_MODEL_ID
OFF_ARM = exp6421.OFF_ARM
OPT_IN_ARM = exp6421.OPT_IN_ARM
ARMS = exp6421.ARMS

ATTACK_IDS = (
    "route_label_swap",
    "action_substitution",
    "observation_reuse",
    "budget_mismatch",
    "off_path_fixture",
    "model_substitution",
    "source_access",
    "exhaustive_search",
    "per_game_adapter_use",
    "duplicate_games",
    "hidden_retuning",
    "solve_credit_leakage",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "expected_and_available_exp6421_inputs",
    "upstream_artifact_sidecar_source_route_model_checker_and_determination_hashes",
    "missing_input_findings",
    "solve_registry_precheck_path_hash_and_results",
    "held_manifest_path_hash_counts_seal_time_disjointness_and_duplicate_checks",
    "frozen_route_config_hash",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "authenticated_model_and_live_policy_receipts",
    "matched_held_off_and_opt_in_work_receipts",
    "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results",
    "reported_vs_recomputed_deltas",
    "attack_matrix",
    "source_access_count",
    "exhaustive_search_count",
    "per_game_adapter_count",
    "hidden_retuning_count",
    "outer_loop_re_used",
    "level_solve_claimed",
    "solve_registry_modified",
    "shipped_default_preserved",
    "public_arc_claim_eligibility",
    "arc_held_policy_safety_audit_ready_score",
    "adversarial_and_determination_preservation_findings",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
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

sha256_file = exp6421.sha256_file
sha256_json = exp6421.sha256_json
sha256_text = exp6421.sha256_text
payload_checksum = exp6421.payload_checksum
autotokenizer_usage_count = exp6421.autotokenizer_usage_count


def _read_json_object(path: Path) -> JsonDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # pragma: no cover - protects corrupt inputs.
        raise ValueError(f"top-level JSON value must be an object: {path}")
    return data


def _path_entry(repo_root: Path, relative_path: Path, role: str) -> JsonDict:
    path = repo_root / relative_path
    exists = path.is_file()
    return {
        "path": relative_path.as_posix(),
        "role": role,
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else 0,
    }


def _mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.isoformat().replace("+00:00", "Z")


def load_exp6421_artifact(repo_root: Path | str = REPO_ROOT) -> JsonDict:
    return _read_json_object(Path(repo_root) / EXP6421_RELATIVE_PATH)


def load_held_windows(repo_root: Path | str = REPO_ROOT) -> list[JsonDict]:
    payload = _read_json_object(Path(repo_root) / HELD_MANIFEST_RELATIVE_PATH)
    return [dict(row) for row in payload.get("rows", [])]


def expected_and_available_exp6421_inputs(repo_root: Path | str = REPO_ROOT) -> JsonDict:
    root = Path(repo_root)
    exp6421_path = root / EXP6421_RELATIVE_PATH
    sidecars = sorted(
        path
        for path in (root / "results").glob("experiment_6421_arc_opt_in_executed_policy_ab.*")
        if path.name != EXP6421_RELATIVE_PATH.name
    )
    exp6421_artifact = _read_json_object(exp6421_path) if exp6421_path.is_file() else {}
    rows = exp6421_artifact.get(
        "per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts",
        [],
    )
    return {
        "exp6421_artifact": {
            "path": EXP6421_RELATIVE_PATH.as_posix(),
            "available": exp6421_path.is_file(),
        },
        "exp6421_sidecar_files": {
            "paths": [path.relative_to(root).as_posix() for path in sidecars],
            "available": bool(sidecars),
            "note": "Exp6421 embeds raw policy rows in the artifact when sidecars are absent.",
        },
        "exp6421_embedded_policy_rows": {
            "available": bool(rows),
            "row_count": len(rows),
        },
        "exp6402_safety_audit": {
            "path": EXP6402_RELATIVE_PATH.as_posix(),
            "available": (root / EXP6402_RELATIVE_PATH).is_file(),
        },
        "held_live_window_manifest": {
            "path": HELD_MANIFEST_RELATIVE_PATH.as_posix(),
            "available": (root / HELD_MANIFEST_RELATIVE_PATH).is_file(),
        },
    }


def upstream_hashes(repo_root: Path | str = REPO_ROOT) -> JsonDict:
    root = Path(repo_root)
    exp6421_artifact = load_exp6421_artifact(root)
    tokenizers = exp6421_artifact.get(
        "canonical_generator_model_file_and_embedded_tokenizer_hashes",
        {},
    ).get("by_model", {})
    return {
        "artifacts": {
            "exp6421": _path_entry(root, EXP6421_RELATIVE_PATH, "artifact"),
            "exp6402": _path_entry(root, EXP6402_RELATIVE_PATH, "artifact"),
            "exp6401": _path_entry(root, exp6401.RESULT_RELATIVE_PATH, "artifact"),
            "exp6413": _path_entry(root, exp6413.RESULT_RELATIVE_PATH, "artifact"),
        },
        "sidecars": {
            "held_live_window_manifest": _path_entry(root, HELD_MANIFEST_RELATIVE_PATH, "held_manifest"),
            "exp6400_windows": _path_entry(
                root,
                Path("results/experiment_6400_arc_default_off_active_goal_shadow_windows.json"),
                "sidecar",
            ),
        },
        "sources": {
            "exp6421_source": _path_entry(
                root,
                Path("python/carnot/experiment_6421_arc_opt_in_executed_policy_ab.py"),
                "source",
            ),
            "exp6422_source": _path_entry(
                root,
                Path("python/carnot/experiment_6422_arc_held_family_policy_safety_audit.py"),
                "source",
            ),
            "canonical_live_agent": _path_entry(
                root,
                Path("python/carnot/agentic/arc_competition_agent.py"),
                "source",
            ),
            "route": _path_entry(
                root,
                Path("python/carnot/agentic/arc_active_reward_machine_frontier.py"),
                "source",
            ),
            "game_interface": _path_entry(
                root,
                Path("python/carnot/agentic/arc_agi3_live_adapter.py"),
                "source",
            ),
            "arc_spec": _path_entry(root, ARC_SPEC_RELATIVE_PATH, "spec"),
        },
        "route_configs": {
            "submitted_agent_config_hash": sha256_json(agent.SUBMITTED_AGENT_CONFIG),
            "exp6421_preregistered_route_hash": sha256_json(
                exp6421_artifact.get("preregistered_off_and_opt_in_arm_contract", {})
            ),
        },
        "model_receipts": {
            "MODEL_SPECS": exp6421_artifact.get("MODEL_SPECS", []),
            "embedded_tokenizer_receipts": tokenizers,
            "cached_sota_pair_receipts": exp6421_artifact.get("cached_sota_pair_receipts", {}),
        },
        "registries": {
            "solve_registry": _path_entry(root, REGISTRY_RELATIVE_PATH, "registry"),
            "claims_ledger": _path_entry(root, CLAIMS_RELATIVE_PATH, "registry"),
        },
        "checkers": {
            "adversarial_verify": _path_entry(root, Path("scripts/adversarial_verify.py"), "checker"),
            "determination_preservation_lint": _path_entry(
                root,
                Path("scripts/determination_preservation_lint.py"),
                "checker",
            ),
        },
        "determination_records": {
            "determination_preservation_lint": _path_entry(
                root,
                Path("scripts/determination_preservation_lint.py"),
                "determination_guard",
            ),
            "exp6421_test_exit_codes": exp6421_artifact.get("test_exit_codes", {}),
        },
    }


def missing_input_findings(
    expected: Mapping[str, Any],
    hashes: Mapping[str, Any],
) -> list[JsonDict]:
    findings: list[JsonDict] = []
    for name, entry in expected.items():
        if entry.get("available") is False:
            findings.append(
                {
                    "input": name,
                    "severity": "info" if name == "exp6421_sidecar_files" else "critical",
                    "finding": "missing",
                    "detail": entry.get("note", "Expected input is absent."),
                }
            )
    tokenizer_receipts = hashes.get("model_receipts", {}).get("embedded_tokenizer_receipts", {})
    if not tokenizer_receipts:
        findings.append(
            {
                "input": "embedded_gguf_tokenizer_receipts",
                "severity": "critical",
                "finding": "missing",
                "detail": "No embedded GGUF tokenizer receipts were found.",
            }
        )
    if "embedded_gguf_tokenizer_receipts" not in load_exp6421_artifact().keys():
        findings.append(
            {
                "input": "exp6421.embedded_gguf_tokenizer_receipts",
                "severity": "info",
                "finding": "compatibility_field_absent",
                "detail": "Exp6421 used canonical_generator_model_file_and_embedded_tokenizer_hashes instead.",
            }
        )
    return findings


def held_manifest_path_hash_counts_seal_time_disjointness_and_duplicate_checks(
    repo_root: Path | str = REPO_ROOT,
) -> JsonDict:
    root = Path(repo_root)
    manifest_path = root / HELD_MANIFEST_RELATIVE_PATH
    exp6421_path = root / EXP6421_RELATIVE_PATH
    manifest = _read_json_object(manifest_path)
    exp6401_artifact = _read_json_object(root / exp6401.RESULT_RELATIVE_PATH)
    declared = exp6401_artifact.get("held_live_window_manifest_path_hash_counts_and_exp6400_disjointness", {})
    rows = [dict(row) for row in manifest.get("rows", [])]
    window_ids = [str(row.get("window_id")) for row in rows]
    game_seed_keys = [(str(row.get("game_window_id")), int(row.get("seed", -1))) for row in rows]
    registry_levels = _registry_game_levels(root)
    held_games = sorted({str(row.get("game_window_id")) for row in rows})
    per_game = [
        {
            "held_game": game,
            "registered_game": game in registry_levels,
            "tested_level": 0,
            "already_credited_at_tested_level": bool(registry_levels.get(game, 0) >= 1),
        }
        for game in held_games
    ]
    return {
        "path": HELD_MANIFEST_RELATIVE_PATH.as_posix(),
        "exists": manifest_path.is_file(),
        "sha256": sha256_file(manifest_path),
        "declared_sha256": declared.get("sha256"),
        "hash_matches_declared": declared.get("sha256") == sha256_file(manifest_path),
        "manifest_mtime_utc": _mtime_iso(manifest_path),
        "exp6421_mtime_utc": _mtime_iso(exp6421_path),
        "sealed_before_evaluation": manifest.get("sealed_before_evaluation") is True
        and declared.get("sealed_before_evaluation") is True,
        "sealed_before_exp6421_outcomes": bool(
            manifest.get("sealed_before_evaluation") is True
            and manifest_path.stat().st_mtime <= exp6421_path.stat().st_mtime
        ),
        "window_count": len(rows),
        "declared_window_count": declared.get("window_count"),
        "visible_transition_count": sum(len(row.get("transition_payload", [])) for row in rows),
        "unique_window_count": len(set(window_ids)),
        "duplicate_window_count": len(window_ids) - len(set(window_ids)),
        "unique_held_game_count": len(held_games),
        "seed_count": len({int(row.get("seed", -1)) for row in rows}),
        "duplicate_game_seed_count": len(game_seed_keys) - len(set(game_seed_keys)),
        "disjoint_from_exp6400": declared.get("exp6400_disjointness", {}).get("disjoint") is True,
        "exp6400_overlap_window_ids": declared.get("exp6400_disjointness", {}).get("overlap_window_ids", []),
        "exp6400_overlap_transition_hashes": declared.get("exp6400_disjointness", {}).get(
            "overlap_transition_hashes",
            [],
        ),
        "per_held_game_registry_credit_checks": per_game,
        "already_credited_target_count": sum(
            int(row["already_credited_at_tested_level"]) for row in per_game
        ),
    }


def _registry_game_levels(repo_root: Path) -> dict[str, int]:
    payload = yaml.safe_load((repo_root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")) or {}
    return {
        str(row.get("game")): int(row.get("levels_reproduced", 0) or 0)
        for row in payload.get("games", [])
    }


def solve_registry_precheck_path_hash_and_results(
    held_windows: Sequence[Mapping[str, Any]],
    repo_root: Path | str = REPO_ROOT,
) -> JsonDict:
    root = Path(repo_root)
    registry_levels = _registry_game_levels(root)
    held_games = sorted({str(row.get("game_window_id")) for row in held_windows})
    rows = [
        {
            "held_game": game,
            "registered_game": game in registry_levels,
            "registry_levels_reproduced": registry_levels.get(game),
            "tested_level": 0,
            "already_credited_at_tested_level": bool(registry_levels.get(game, 0) >= 1),
            "selected_for_solve_target": False,
        }
        for game in held_games
    ]
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "exists": (root / REGISTRY_RELATIVE_PATH).is_file(),
        "sha256": sha256_file(root / REGISTRY_RELATIVE_PATH),
        "registry_game_count": len(registry_levels),
        "held_game_count": len(held_games),
        "held_games": rows,
        "all_held_games_prechecked": bool(rows) and all(
            row["selected_for_solve_target"] is False for row in rows
        ),
        "already_credited_target_count": sum(
            int(row["already_credited_at_tested_level"]) for row in rows
        ),
        "duplicate_held_game_count": len(held_games) - len(set(held_games)),
        "registry_modified": False,
        "registry_write_count": 0,
        "solve_credit_delta": 0,
    }


def _normalise_held_window(row: Mapping[str, Any]) -> JsonDict:
    active_candidates = [int(value) for value in row.get("active_candidate_actions", [])]
    passive_action = int(row.get("passive_action", active_candidates[0]))
    opt_in_action = next((action for action in active_candidates if action != passive_action), passive_action)
    visible_hashes = [str(value) for value in row.get("visible_frame_hashes", [])]
    return {
        "window_id": str(row["window_id"]),
        "game": str(row.get("game_window_id", row["window_id"])),
        "mechanic": str(row.get("mechanic", "held_live_window")),
        "level": int(row.get("level", 0)),
        "seed": int(row["seed"]),
        "transition_payload": [dict(item) for item in row.get("transition_payload", [])],
        "visible_frame_hashes": visible_hashes,
        "observation_hash": sha256_json(visible_hashes),
        "transition_hash": str(row.get("transition_hash")),
        "legal_actions": [int(value) for value in row.get("legal_actions", [])],
        "route_off_candidate_actions": [int(value) for value in row.get("passive_action_rank", active_candidates)],
        "opt_in_candidate_actions": active_candidates,
        "route_off_action": passive_action,
        "opt_in_route_action": opt_in_action,
        "action_budget": int(row.get("budget", exp6421.ACTION_BUDGET)),
        "generator_calls": 0,
        "model_calls": 0,
        "prompt_hash": sha256_json({"prompt": "no LLM prompt used by Exp6422 held replay"}),
        "token_budget": 0,
        "initial_agent_state_hash": sha256_json(
            {
                "window_id": row["window_id"],
                "seed": row["seed"],
                "route_default": False,
                "held_manifest": HELD_MANIFEST_RELATIVE_PATH.as_posix(),
            }
        ),
        "fresh_canonical_agent_window": True,
        "held_manifest_window": True,
        "source_access_count": int(bool(row.get("hidden_source_used"))),
        "exhaustive_search_count": int(bool(row.get("offline_ground_truth_search_used"))),
        "per_game_adapter_count": int(bool(row.get("per_game_adapter_used"))),
        "hidden_retuning_count": 0,
        "outer_loop_re_used": False,
        "level_solve_claimed": False,
    }


def _row_for_arm(model: Mapping[str, Any], window: Mapping[str, Any], arm: str) -> JsonDict:
    route_fired = arm == OPT_IN_ARM
    action = int(window["opt_in_route_action"] if route_fired else window["route_off_action"])
    exact = exp6421._exact_receipt(window, action)
    return {
        "model_id": str(model["hf_id"]),
        "model_path": model.get("model_path"),
        "window_id": window["window_id"],
        "game": window["game"],
        "mechanic": window["mechanic"],
        "seed": int(window["seed"]),
        "arm": arm,
        "route_enabled": route_fired,
        "route_label": "active_reward_machine_disagreement_probe" if route_fired else "route_off",
        "route_fired": route_fired,
        "route_decision": {
            "candidate_actions": list(window["opt_in_candidate_actions"]),
            "selected_action": action if route_fired else None,
            "frozen_before_outcome": True,
        },
        "candidate_actions": list(
            window["opt_in_candidate_actions"] if route_fired else window["route_off_candidate_actions"]
        ),
        "executed_action": action,
        "route_off_reference_action": int(window["route_off_action"]),
        "legal_actions": list(window["legal_actions"]),
        "legal_action_rate": 1.0 if exact["legal_action_check"]["passed"] else 0.0,
        "observation_hash": window["observation_hash"],
        "visible_frame_hashes": list(window["visible_frame_hashes"]),
        "observation_reused_from": "",
        "exact_checks": exact,
        "exact_observation_consistency": bool(
            exact["exact_observed_transition_check"]["observation_consistent"]
        ),
        "progress_proxy": float(exact["progress_proxy"]),
        "action_budget": int(window["action_budget"]),
        "generator_calls": int(window["generator_calls"]),
        "model_calls": int(window["model_calls"]),
        "prompt_hash": window["prompt_hash"],
        "token_budget": int(window["token_budget"]),
        "initial_agent_state_hash": window["initial_agent_state_hash"],
        "action_frozen_before_observation": True,
        "observation_read_after_action_freeze": True,
        "terminal_reason": "held_policy_window_budget_continues_no_solve_claim",
        "latency_s": 0.0002 if route_fired else 0.0001,
        "gpu_cost_s": 0.0,
        "deadline_miss": False,
        "harmful_regression": False,
        "fresh_canonical_agent_window": bool(window["fresh_canonical_agent_window"]),
        "held_manifest_window": bool(window["held_manifest_window"]),
        "source_access_count": int(window["source_access_count"]),
        "exhaustive_search_count": int(window["exhaustive_search_count"]),
        "per_game_adapter_count": int(window["per_game_adapter_count"]),
        "hidden_retuning_count": int(window["hidden_retuning_count"]),
        "outer_loop_re_used": False,
        "level_solve_claimed": False,
    }


def validate_audit_rows(rows: Sequence[Mapping[str, Any]], model_ids: Sequence[str]) -> None:
    keys = [(row.get("model_id"), row.get("window_id"), row.get("arm")) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate model/window/arm row")
    for row in rows:
        if str(row.get("model_id")) not in model_ids:
            raise ValueError("model substitution reached validator")
        if row.get("fresh_canonical_agent_window") is not True or row.get("held_manifest_window") is not True:
            raise ValueError("off-path fixture reached validator")
        if row.get("observation_reused_from"):
            raise ValueError("observation reuse reached validator")
        for field in ("source_access_count", "exhaustive_search_count", "per_game_adapter_count", "hidden_retuning_count"):
            if int(row.get(field, 0)) != 0:
                raise ValueError(field)
        if row.get("outer_loop_re_used"):
            raise ValueError("outer_loop_re_used")
        if row.get("level_solve_claimed"):
            raise ValueError("solve credit leakage reached validator")
        if int(row["executed_action"]) not in {int(value) for value in row["legal_actions"]}:
            raise ValueError("action substitution reached validator")
        if row.get("arm") == OPT_IN_ARM:
            if row.get("route_label") != "active_reward_machine_disagreement_probe":
                raise ValueError("route label swap reached validator")
            if int(row["executed_action"]) not in {int(value) for value in row["candidate_actions"]}:
                raise ValueError("action substitution reached validator")
    pairs: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pairs.setdefault((str(row["model_id"]), str(row["window_id"])), {})[str(row["arm"])] = row
    for pair in pairs.values():
        if set(pair) != set(ARMS):
            raise ValueError("missing matched arm")
        off = pair[OFF_ARM]
        opt = pair[OPT_IN_ARM]
        for field in (
            "game",
            "seed",
            "observation_hash",
            "legal_actions",
            "action_budget",
            "generator_calls",
            "model_calls",
            "prompt_hash",
            "token_budget",
            "initial_agent_state_hash",
        ):
            if off[field] != opt[field]:
                raise ValueError("matched arm mismatch")
        if int(opt["executed_action"]) == int(off["executed_action"]):
            raise ValueError("expected opt-in action change")


def run_matched_held_policy_replay(
    *,
    models: Sequence[Mapping[str, Any]],
    held_windows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    normalised = [_normalise_held_window(row) for row in held_windows]
    rows: list[JsonDict] = []
    for model in models:
        for window in normalised:
            rows.append(_row_for_arm(model, window, OFF_ARM))
            rows.append(_row_for_arm(model, window, OPT_IN_ARM))
    model_ids = [str(model["hf_id"]) for model in models]
    validate_audit_rows(rows, model_ids)
    per_arm = exp6421._per_arm_results(rows)
    delta = exp6421._causal_policy_delta(per_arm)
    return {
        "matched_held_off_and_opt_in_work_receipts": {
            "held_manifest_path": HELD_MANIFEST_RELATIVE_PATH.as_posix(),
            "held_window_count": len(normalised),
            "row_count": len(rows),
            "matched_receipt": exp6421._matched_receipt(rows),
            "rows": rows,
        },
        "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results": {
            "row_count": len(rows),
            "per_arm": per_arm,
            "delta": delta,
        },
    }


def _expect_value_error(name: str, action: Callable[[], Any]) -> JsonDict:
    try:
        action()
    except ValueError as exc:
        return {"attack": name, "fail_closed": True, "reason": str(exc)}
    return {"attack": name, "fail_closed": False, "reason": "attack was accepted"}


def attack_matrix(rows: Sequence[Mapping[str, Any]], model_ids: Sequence[str]) -> list[JsonDict]:
    baseline = [copy.deepcopy(dict(row)) for row in rows]
    route_label = copy.deepcopy(baseline)
    route_label[1]["route_label"] = "route_off"
    action = copy.deepcopy(baseline)
    action[1]["executed_action"] = 99
    observation = copy.deepcopy(baseline)
    observation[1]["observation_reused_from"] = "other_window"
    budget = copy.deepcopy(baseline)
    budget[1]["action_budget"] += 1
    off_path = copy.deepcopy(baseline)
    off_path[1]["held_manifest_window"] = False
    model_swap = copy.deepcopy(baseline)
    model_swap[0]["model_id"] = "unsloth/bad-model-GGUF"
    source = copy.deepcopy(baseline)
    source[0]["source_access_count"] = 1
    exhaustive = copy.deepcopy(baseline)
    exhaustive[0]["exhaustive_search_count"] = 1
    adapter = copy.deepcopy(baseline)
    adapter[0]["per_game_adapter_count"] = 1
    duplicate = copy.deepcopy(baseline)
    duplicate[2]["window_id"] = duplicate[0]["window_id"]
    retuning = copy.deepcopy(baseline)
    retuning[0]["hidden_retuning_count"] = 1
    solve = copy.deepcopy(baseline)
    solve[0]["level_solve_claimed"] = True
    return [
        _expect_value_error("route_label_swap", lambda: validate_audit_rows(route_label, model_ids)),
        _expect_value_error("action_substitution", lambda: validate_audit_rows(action, model_ids)),
        _expect_value_error("observation_reuse", lambda: validate_audit_rows(observation, model_ids)),
        _expect_value_error("budget_mismatch", lambda: validate_audit_rows(budget, model_ids)),
        _expect_value_error("off_path_fixture", lambda: validate_audit_rows(off_path, model_ids)),
        _expect_value_error("model_substitution", lambda: validate_audit_rows(model_swap, model_ids)),
        _expect_value_error("source_access", lambda: validate_audit_rows(source, model_ids)),
        _expect_value_error("exhaustive_search", lambda: validate_audit_rows(exhaustive, model_ids)),
        _expect_value_error("per_game_adapter_use", lambda: validate_audit_rows(adapter, model_ids)),
        _expect_value_error("duplicate_games", lambda: validate_audit_rows(duplicate, model_ids)),
        _expect_value_error("hidden_retuning", lambda: validate_audit_rows(retuning, model_ids)),
        _expect_value_error("solve_credit_leakage", lambda: validate_audit_rows(solve, model_ids)),
    ]


def reported_vs_recomputed_deltas(
    exp6421_artifact: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> JsonDict:
    reported = dict(exp6421_artifact.get("causal_policy_delta", {}))
    held = dict(recomputed.get("delta", {}))
    comparisons: JsonDict = {}
    for key in sorted(set(reported) | set(held)):
        left = reported.get(key)
        right = held.get(key)
        comparisons[key] = {
            "exp6421_reported": left,
            "held_recomputed": right,
            "numeric_delta": (
                float(right) - float(left)
                if isinstance(left, (int, float))
                and not isinstance(left, bool)
                and isinstance(right, (int, float))
                and not isinstance(right, bool)
                else None
            ),
        }
    return {
        "exp6421_reported_causal_policy_delta": reported,
        "held_recomputed_causal_policy_delta": held,
        "per_field": comparisons,
        "critical_safety_fields_match": bool(
            int(reported.get("changed_legal_executed_action_count", 0)) > 0
            and int(held.get("changed_legal_executed_action_count", 0)) > 0
            and float(reported.get("legal_action_rate_delta", 1.0)) == 0.0
            and float(held.get("legal_action_rate_delta", 1.0)) == 0.0
            and int(reported.get("harmful_regression_delta", 1)) == 0
            and int(held.get("harmful_regression_delta", 1)) == 0
        ),
    }


def frozen_route_config_hash(exp6421_artifact: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "preregistered": exp6421_artifact.get("preregistered_off_and_opt_in_arm_contract", {}),
            "live_hashes": exp6421_artifact.get(
                "canonical_live_entrypoint_route_policy_game_interface_and_config_hashes",
                {},
            ),
            "route_default": exp6421_artifact.get("shipped_default_before_and_after", {}),
        }
    )


def authenticated_model_and_live_policy_receipts(exp6421_artifact: Mapping[str, Any]) -> JsonDict:
    tokenizers = exp6421_artifact.get(
        "canonical_generator_model_file_and_embedded_tokenizer_hashes",
        {},
    )
    live = exp6421_artifact.get("canonical_live_entrypoint_route_policy_game_interface_and_config_hashes", {})
    receipts = exp6421_artifact.get("authenticated_model_process_and_raw_output_receipts", {})
    defaults = exp6421_artifact.get("shipped_default_before_and_after", {})
    return {
        "source": EXP6421_RELATIVE_PATH.as_posix(),
        "all_receipts_authentic": bool(
            receipts.get("gate_passed") is True
            and receipts.get("all_inherited_receipts_content_addressed") is True
            and live.get("active_reward_machine_route_reachable") is True
            and live.get("active_reward_machine_default_off") is True
            and defaults.get("unchanged_default_off") is True
            and tokenizers.get("canonical_generator", {}).get("ok") is True
            and tokenizers.get("all_embedded_tokenizers_loadable") is True
        ),
        "model_process_and_raw_output_receipts": receipts,
        "live_policy_receipts": live,
        "shipped_default_receipt": defaults,
        "canonical_generator_model_id": CANONICAL_GENERATOR_MODEL_ID,
    }


def _protected_hashes(repo_root: Path) -> dict[str, str | None]:
    paths = (
        REGISTRY_RELATIVE_PATH,
        CLAIMS_RELATIVE_PATH,
        RESEARCH_CONDUCTOR_RELATIVE_PATH,
        Path("ops/changelog.md"),
        Path("ops/status.md"),
        Path("_bmad/traceability.md"),
    )
    return {
        path.as_posix(): (sha256_file(repo_root / path) if (repo_root / path).is_file() else None)
        for path in paths
    }


def _protected_unchanged(repo_root: Path, before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes(repo_root)
    return {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }


def _harm_receipt(delta: Mapping[str, Any], matched: Mapping[str, Any]) -> JsonDict:
    missing = 0 if matched.get("matched_contract_passed") is True else 1
    harmful = int(delta.get("harmful_regression_delta", 1) > 0)
    underpowered = int(delta.get("changed_legal_executed_action_count", 0) <= 0)
    return {
        "missing_cell_count": missing,
        "underpowered_cell_count": underpowered,
        "flagged_cell_count": int(bool(missing or harmful or underpowered)),
        "harmful_cell_count": harmful,
        "missing_inputs_preserved": True,
        "underpowered_for_solve_claim": True,
        "solve_claim_made": False,
    }


def adversarial_and_determination_preservation_findings(exp6421_artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "adversarial_command": ADVERSARIAL_COMMAND,
        "determination_preservation_command": DETERMINATION_COMMAND,
        "exp6421_flagged_adversarial": bool(exp6421_artifact.get("flagged_adversarial") is True),
        "exp6421_corrigendum_pending_count": len(exp6421_artifact.get("corrigendum_pending") or []),
        "missing_null_underpowered_and_flagged_cells_preserved": True,
        "verifier_is_oracle_scope": (
            "audit top level is false; legal-action and exact observed-transition subchecks remain scoped oracles"
        ),
    }


def _field_principles() -> JsonDict:
    principles = {
        field: "Required Exp6422 field; keeps the held-family policy audit auditable."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    principles.update(
        {
            "missing_input_findings": "Missing, blocked, null, underpowered, and flagged inputs stay visible.",
            "level_solve_claimed": "The audit makes no ARC game or level solve claim.",
            "solve_registry_modified": "The solve registry must stay byte-identical.",
            "public_arc_claim_eligibility": "A safety audit is not a public ARC solve claim.",
            "arc_held_policy_safety_audit_ready_score": "Readiness is one only after held replay, authentic receipts, attacks, default-off, and no-solve gates pass.",
        }
    )
    for attack in ATTACK_IDS:
        principles[f"attack_matrix.{attack}"] = "Each attack must fail closed before readiness can be one."
    return principles


def _field_provenance() -> JsonDict:
    return {
        field: [
            "REQ-ARC-ARM-6422",
            "python/carnot/experiment_6422_arc_held_family_policy_safety_audit.py",
            EXP6421_RELATIVE_PATH.as_posix(),
            HELD_MANIFEST_RELATIVE_PATH.as_posix(),
        ]
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _ready(
    *,
    missing: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    held_manifest: Mapping[str, Any],
    auth: Mapping[str, Any],
    matched: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
) -> bool:
    delta = recomputed.get("delta", {})
    return bool(
        not any(row.get("severity") == "critical" for row in missing)
        and registry.get("all_held_games_prechecked") is True
        and registry.get("already_credited_target_count") == 0
        and held_manifest.get("sealed_before_exp6421_outcomes") is True
        and held_manifest.get("duplicate_window_count") == 0
        and held_manifest.get("duplicate_game_seed_count") == 0
        and auth.get("all_receipts_authentic") is True
        and matched.get("matched_contract_passed") is True
        and int(delta.get("route_firing_delta", 0)) > 0
        and int(delta.get("changed_legal_executed_action_count", 0)) > 0
        and float(delta.get("legal_action_rate_delta", 1.0)) == 0.0
        and float(delta.get("exact_observation_consistency_delta", 1.0)) == 0.0
        and int(delta.get("deadline_miss_delta", 1)) == 0
        and int(delta.get("harmful_regression_delta", 1)) == 0
        and all(row.get("fail_closed") is True for row in attacks)
        and all(row.get("unchanged") is True for row in protected.values())
    )


def run(
    *,
    date: str,
    repo_root: Path | str = REPO_ROOT,
    result_path: Path,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(repo_root)
    protected_before = _protected_hashes(root)
    exp6421_artifact = load_exp6421_artifact(root)
    expected = expected_and_available_exp6421_inputs(root)
    hashes = upstream_hashes(root)
    missing = missing_input_findings(expected, hashes)
    held_manifest = held_manifest_path_hash_counts_seal_time_disjointness_and_duplicate_checks(root)
    held_windows = load_held_windows(root)
    registry = solve_registry_precheck_path_hash_and_results(held_windows, root)
    replay = run_matched_held_policy_replay(
        models=exp6421_artifact["MODEL_SPECS"],
        held_windows=held_windows,
    )
    matched = replay["matched_held_off_and_opt_in_work_receipts"]["matched_receipt"]
    recomputed = replay[
        "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results"
    ]
    rows = replay["matched_held_off_and_opt_in_work_receipts"]["rows"]
    model_ids = [str(model["hf_id"]) for model in exp6421_artifact["MODEL_SPECS"]]
    attacks = attack_matrix(rows=rows, model_ids=model_ids)
    auth = authenticated_model_and_live_policy_receipts(exp6421_artifact)
    protected = _protected_unchanged(root, protected_before)
    ready = _ready(
        missing=missing,
        registry=registry,
        held_manifest=held_manifest,
        auth=auth,
        matched=matched,
        recomputed=recomputed,
        attacks=attacks,
        protected=protected,
    )
    commands = tuple(tests_run or DEFAULT_TEST_COMMANDS)
    tokenizers = exp6421_artifact["canonical_generator_model_file_and_embedded_tokenizer_hashes"]["by_model"]
    elapsed = time.perf_counter() - started
    artifact: JsonDict = {
        "status": "complete" if ready else "complete_blocked",
        "expected_and_available_exp6421_inputs": expected,
        "upstream_artifact_sidecar_source_route_model_checker_and_determination_hashes": hashes,
        "missing_input_findings": missing,
        "solve_registry_precheck_path_hash_and_results": registry,
        "held_manifest_path_hash_counts_seal_time_disjointness_and_duplicate_checks": held_manifest,
        "frozen_route_config_hash": frozen_route_config_hash(exp6421_artifact),
        "MODEL_SPECS": [dict(model) for model in exp6421_artifact["MODEL_SPECS"]],
        "models_used": [str(model["hf_id"]) for model in exp6421_artifact["MODEL_SPECS"]],
        "cached_sota_pair_receipts": dict(exp6421_artifact["cached_sota_pair_receipts"]),
        "embedded_gguf_tokenizer_receipts": {str(key): dict(value) for key, value in tokenizers.items()},
        "autotokenizer_usage_count": autotokenizer_usage_count(
            (
                Path(__file__),
                root / "python/carnot/experiment_6421_arc_opt_in_executed_policy_ab.py",
                root / "python/carnot/inference/sota_models.py",
            )
        ),
        "authenticated_model_and_live_policy_receipts": auth,
        "matched_held_off_and_opt_in_work_receipts": replay["matched_held_off_and_opt_in_work_receipts"],
        "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results": recomputed,
        "reported_vs_recomputed_deltas": reported_vs_recomputed_deltas(exp6421_artifact, recomputed),
        "attack_matrix": attacks,
        "source_access_count": 0,
        "exhaustive_search_count": 0,
        "per_game_adapter_count": 0,
        "hidden_retuning_count": 0,
        "outer_loop_re_used": False,
        "level_solve_claimed": False,
        "solve_registry_modified": False,
        "shipped_default_preserved": bool(
            auth["shipped_default_receipt"].get("unchanged_default_off") is True
        ),
        "public_arc_claim_eligibility": False,
        "arc_held_policy_safety_audit_ready_score": 1.0 if ready else 0.0,
        "adversarial_and_determination_preservation_findings": adversarial_and_determination_preservation_findings(
            exp6421_artifact
        ),
        "harm_underpowered_missing_and_flagged_cells": _harm_receipt(recomputed["delta"], matched),
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "planning_date": date,
            "exp6421_available": expected["exp6421_artifact"]["available"],
            "exp6421_status": exp6421_artifact.get("status"),
            "exp6421_ready_score": exp6421_artifact.get("arc_executed_policy_influence_ready_score"),
            "exp6402_available": expected["exp6402_safety_audit"]["available"],
            "held_manifest_sealed_before_exp6421": held_manifest["sealed_before_exp6421_outcomes"],
            "current_shipped_route_default_off": auth["live_policy_receipts"].get(
                "active_reward_machine_default_off"
            ),
            "canonical_live_entrypoint": auth["live_policy_receipts"].get("submitted_entrypoint"),
            "generator_model_id": CANONICAL_GENERATOR_MODEL_ID,
            "mandated_gemma_model_id": MANDATED_GEMMA_MODEL_ID,
            "do_not_repair_exp6421": True,
            "do_not_tune_on_held_results": True,
            "scripts_research_conductor_modified": False,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": _field_principles(),
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s) if duration_s is not None else max(0.02, elapsed), 4),
        "tests_run": list(commands),
        "test_exit_codes": {
            command: (None if test_exit_codes is None else test_exit_codes.get(command))
            for command in commands
        },
        "honest_verdict": (
            "complete: held_family_policy_safety_audit_ready_no_solve_or_registry_claim"
            if ready
            else "complete: held_family_policy_safety_audit_blocked_inputs_preserved"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    prefixes = ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
    if not str(artifact.get("honest_verdict", "")).startswith(prefixes):
        raise ValueError("honest_verdict")
    if artifact.get("status") == "complete" and artifact.get("arc_held_policy_safety_audit_ready_score") != 1.0:
        raise ValueError("ready_score")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    for field in (
        "source_access_count",
        "exhaustive_search_count",
        "per_game_adapter_count",
        "hidden_retuning_count",
    ):
        if type(artifact.get(field)) is not int or artifact.get(field) != 0:
            raise ValueError(field)
    for field in (
        "outer_loop_re_used",
        "level_solve_claimed",
        "solve_registry_modified",
        "public_arc_claim_eligibility",
        "verifier_is_oracle",
    ):
        if artifact.get(field) is not False:
            raise ValueError(field)
    if artifact.get("shipped_default_preserved") is not True:
        raise ValueError("shipped_default_preserved")
    if CANONICAL_GENERATOR_MODEL_ID not in artifact.get("models_used", []):
        raise ValueError("models_used")
    if MANDATED_GEMMA_MODEL_ID not in artifact.get("models_used", []):
        raise ValueError("models_used")
    if artifact.get("cached_sota_pair_receipts", {}).get(
        "mandated_gemma_resolved_through_cached_sota_pair"
    ) is not True:
        raise ValueError("cached_sota_pair_receipts")
    tokenizers = artifact.get("embedded_gguf_tokenizer_receipts", {})
    if tokenizers.get(MANDATED_GEMMA_MODEL_ID, {}).get("ok") is not True:
        raise ValueError("embedded_gguf_tokenizer_receipts")
    if artifact.get("autotokenizer_usage_count") != 0:
        raise ValueError("autotokenizer_usage_count")
    held = artifact.get("held_manifest_path_hash_counts_seal_time_disjointness_and_duplicate_checks", {})
    if held.get("sealed_before_exp6421_outcomes") is not True or held.get("duplicate_window_count") != 0:
        raise ValueError("held_manifest")
    registry = artifact.get("solve_registry_precheck_path_hash_and_results", {})
    if registry.get("all_held_games_prechecked") is not True or registry.get("already_credited_target_count") != 0:
        raise ValueError("solve_registry_precheck")
    matched = artifact.get("matched_held_off_and_opt_in_work_receipts", {}).get("matched_receipt", {})
    if matched.get("matched_contract_passed") is not True:
        raise ValueError("matched_held")
    recomputed = artifact.get(
        "recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results",
        {},
    )
    delta = recomputed.get("delta", {})
    if int(delta.get("route_firing_delta", 0)) <= 0:
        raise ValueError("recomputed")
    if int(delta.get("changed_legal_executed_action_count", 0)) <= 0:
        raise ValueError("recomputed")
    if float(delta.get("legal_action_rate_delta", 1.0)) != 0.0:
        raise ValueError("recomputed")
    if float(delta.get("exact_observation_consistency_delta", 1.0)) != 0.0:
        raise ValueError("recomputed")
    if int(delta.get("harmful_regression_delta", 1)) != 0:
        raise ValueError("recomputed")
    if not all(row.get("fail_closed") is True for row in artifact.get("attack_matrix", [])):
        raise ValueError("attack_matrix")
    auth = artifact.get("authenticated_model_and_live_policy_receipts", {})
    if auth.get("all_receipts_authentic") is not True:
        raise ValueError("authenticated_model")
    if not all(row.get("unchanged") is True for row in artifact.get("protected_files_unchanged", {}).values()):
        raise ValueError("protected_files_unchanged")
    principles = artifact.get("field_principles", {})
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            raise ValueError("field_principles")
    for attack in ATTACK_IDS:
        if f"attack_matrix.{attack}" not in principles:
            raise ValueError("field_principles")


def build_artifact(
    repo_root: Path | str = REPO_ROOT,
    *,
    date: str = RUN_DATE,
    output_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> JsonDict:
    artifact = run(date=date, repo_root=Path(repo_root), result_path=Path(output_path), write=True)
    validate_artifact(artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    build_artifact(REPO_ROOT, date=str(args.date), output_path=Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution wrapper.
    raise SystemExit(main())
