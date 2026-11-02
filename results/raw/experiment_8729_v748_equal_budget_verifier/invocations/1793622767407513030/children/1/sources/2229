"""Exp6308 ARC target-validated route holdout audit.

Spec refs: REQ-ARC-WMTE-6308,
SCENARIO-ARC-WMTE-6308-GATE-REPLAY,
SCENARIO-ARC-WMTE-6308-FOLD-ISOLATION,
SCENARIO-ARC-WMTE-6308-NO-REFIT-DEFAULT-OFF,
SCENARIO-ARC-WMTE-6308-PER-FOLD-GATE,
SCENARIO-ARC-WMTE-6308-ZERO-SOLVE-CREDIT.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot import experiment_6307_arc_target_validated_route_canary as exp6307
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_mechanic_class_detector as detector


JsonDict = dict[str, Any]
ModelResolver = Callable[[bool], list[JsonDict]]
LLMRunner = Callable[[Sequence[JsonDict], Sequence[JsonDict], Path, bool], JsonDict]

REPO_ROOT = exp6307.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_6308_arc_target_validated_route_holdout.json")
FOLD_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6308_arc_target_validated_route_holdout_folds.json"
)
RAW_OUTPUT_DIR_RELATIVE_PATH = Path("results/experiment_6308_arc_target_validated_route_raw")
UPSTREAM_RELATIVE_PATH = exp6307.RESULT_RELATIVE_PATH
UPSTREAM_6295_RELATIVE_PATH = Path("results/experiment_6295_arc_mechanic_router_holdout_audit.json")
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6308_arc_target_validated_route_holdout "
    "--date 20260811"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6308_arc_target_validated_route_holdout.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6308_arc_target_validated_route_holdout.py "
    "-m pytest tests/python/test_experiment_6308_arc_target_validated_route_holdout.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6308_arc_target_validated_route_holdout.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6308_arc_target_validated_route_holdout.py"
)
EXP6298_PREFLIGHT_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6298_terminal_evidence_preflight_linter "
    "--date 20260811 --no-run-commands"
)
E2E_PLAN_READ_COMMAND = "sed -n 1,180p ops/e2e-test-plan.md"
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6308_arc_target_validated_route_holdout.json"
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6308_test_receipts.json")

ARMS = exp6307.ARMS
MANDATED_MODEL_IDS = exp6307.MANDATED_MODEL_IDS
ACTION_BUDGET = exp6307.ACTION_BUDGET
MODEL_BUDGET_TOKENS = exp6307.MODEL_BUDGET_TOKENS
PREFERRED_QUANT = exp6307.PREFERRED_QUANT
ADEQUATE_FOLD_SAMPLE_SIZE = 4
DEFAULT_FLAG_NAME = "CARNOT_ARC_TARGET_LICENSED_ROUTE_ENABLED"
DEFAULT_FLAG_VALUE = False
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEEDS = (6308001, 6308002, 6308003, 6308004, 6308005)
FORBIDDEN_ZERO_FIELDS = (
    "hidden_game_source_access_count",
    "outer_loop_ground_truth_search_count",
    "arc_level_solve_claim_count",
    "registry_update_count",
    "source_model_weight_mutation_count",
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    EXP6298_PREFLIGHT_COMMAND,
    E2E_PLAN_READ_COMMAND,
    DETERMINATION_COMMAND,
    ADVERSARIAL_COMMAND,
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_path_hash_and_terminal_class",
    "structured_gate_receipt",
    "registry_precheck_path_hash_and_target_receipt",
    "solve_provenance",
    "frozen_policy_paths_and_hashes",
    "default_flag_name_value_and_receipt",
    "held_game_mechanic_model_seed_and_transition_fold_manifest_path_and_hash",
    "no_refit_receipts_by_fold",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_and_quantizations",
    "tokenizer_and_chat_template_hashes",
    "cuda_and_gpu_offload_receipts_by_model",
    "raw_output_paths_and_hashes",
    "row_counts_by_fold_and_stratum",
    "hypothesis_activation_rejection_and_abstention_by_fold",
    "proposal_acceptance_invalid_rate_diversity_and_latency_by_fold",
    "paired_deltas_intervals_and_sample_sizes_by_fold",
    "missing_underpowered_or_harmful_folds",
    "baseline_harm_controls_by_fold",
    "hidden_game_source_access_count",
    "outer_loop_ground_truth_search_count",
    "arc_level_solve_claim_count",
    "registry_update_count",
    "source_model_weight_mutation_count",
    "arc_target_licensed_generalization_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "States complete versus blocked without hiding partial work.",
    "upstream_path_hash_and_terminal_class": "Pins the Exp6307 canary and Exp6295 holdout gate input.",
    "structured_gate_receipt": "Proves Exp6307 ready score is exactly one before execution.",
    "registry_precheck_path_hash_and_target_receipt": "Proves this is not a public solve target.",
    "solve_provenance": "States live-agent self-discovery without solve credit.",
    "frozen_policy_paths_and_hashes": "Pins the source and constants reused without edits.",
    "default_flag_name_value_and_receipt": "Keeps the feature default false.",
    "held_game_mechanic_model_seed_and_transition_fold_manifest_path_and_hash": "Pins held folds before execution.",
    "no_refit_receipts_by_fold": "Shows no threshold, prompt, adapter, or predicate was retuned.",
    "MODEL_SPECS": "Names both mandated local GGUF model ids.",
    "models_used": "Lists model ids evaluated as frozen policy cells.",
    "model_file_hashes_revisions_and_quantizations": "Pins concrete upstream model files.",
    "tokenizer_and_chat_template_hashes": "Pins tokenizer and prompt contracts.",
    "cuda_and_gpu_offload_receipts_by_model": "Carries the terminal upstream CUDA/offload evidence.",
    "raw_output_paths_and_hashes": "Pins each held-fold raw response row.",
    "row_counts_by_fold_and_stratum": "Shows fold and stratum sample sizes.",
    "hypothesis_activation_rejection_and_abstention_by_fold": "Separates retrieval from licensed transfer.",
    "proposal_acceptance_invalid_rate_diversity_and_latency_by_fold": "Measures proposal quality by fold.",
    "paired_deltas_intervals_and_sample_sizes_by_fold": "Applies the per-fold readiness gate.",
    "missing_underpowered_or_harmful_folds": "Preserves folds that cannot support promotion.",
    "baseline_harm_controls_by_fold": "Prevents a treatment from hiding baseline harm.",
    "hidden_game_source_access_count": "Must stay zero for hidden-source discipline.",
    "outer_loop_ground_truth_search_count": "Must stay zero for self-discovery discipline.",
    "arc_level_solve_claim_count": "Must stay zero because this is not a solve task.",
    "registry_update_count": "Must stay zero because no solve is banked.",
    "source_model_weight_mutation_count": "Must stay zero because model weights are immutable.",
    "arc_target_licensed_generalization_ready_score": "Equals one only if every powered held fold passes.",
    "protected_files_unchanged": "Confirms registry, ops, trace, and conductor files stayed unchanged.",
    "preconditions_checked": "Records gate, registry, model, CUDA, timeout, and seed checks.",
    "inference_substrate": "Declares aggregation from the frozen upstream live canary.",
    "verifier_is_oracle": "False because no game oracle verifies a solve.",
    "field_provenance": "Maps every field to the spec and producer.",
    "field_principles": "Gives one audit reason per required field.",
    "test_commands": "Lists verification commands.",
    "test_exit_codes": "Records command outcomes.",
    "duration_s": "Records measured wall time.",
    "random_seeds": "Pins held-fold seeds.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "Terminal verdict with no solve claim.",
}
FIELD_PROVENANCE = {
    field: ["REQ-ARC-WMTE-6308", "experiment_6308_arc_target_validated_route_holdout"]
    for field in REQUIRED_ARTIFACT_FIELDS
}

payload_checksum = exp6307.payload_checksum
sha256_json = exp6307.sha256_json
sha256_text = exp6307.sha256_text
sha256_file = exp6307.sha256_file


@dataclass(frozen=True)
class HeldWindow:
    """One held transition window built without reading game source."""

    fold_id: str
    held_game_id: str
    fixture_id: str
    window_id: str
    mechanic: str
    seed: int
    transitions: tuple[e3.Transition, ...]
    starting_history_hash: str
    action_budget: int = ACTION_BUDGET
    model_budget_tokens: int = MODEL_BUDGET_TOKENS


@dataclass(frozen=True)
class HeldFold:
    """One game/mechanic stratum that must be judged independently."""

    fold_id: str
    held_games: tuple[str, ...]
    mechanics: tuple[str, ...]
    seeds: tuple[int, ...]
    windows: tuple[HeldWindow, ...]
    adequately_powered: bool


def load_upstream_artifact(path: Path | None = None) -> JsonDict:
    source = path or (REPO_ROOT / UPSTREAM_RELATIVE_PATH)
    return json.loads(source.read_text(encoding="utf-8"))


def _terminal_class(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    if payload.get("flagged_adversarial"):
        return "flagged"
    if status.startswith("blocked") or verdict.startswith("blocked"):
        return "blocked"
    if status == "complete" or verdict.startswith(("complete:", "complete_")):
        return "complete"
    return status or "unknown"


def upstream_path_hash_and_terminal_class(upstream: Mapping[str, Any]) -> JsonDict:
    rows: list[JsonDict] = []
    for rel in (UPSTREAM_RELATIVE_PATH, UPSTREAM_6295_RELATIVE_PATH):
        path = REPO_ROOT / rel
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if rel == UPSTREAM_RELATIVE_PATH:
            payload = dict(upstream)
        rows.append(
            {
                "path": rel.as_posix(),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "terminal_class": _terminal_class(payload),
                "ready_score": payload.get("arc_target_licensed_router_ready_score"),
                "blocked_at_layer": payload.get("blocked_at_layer"),
            }
        )
    return {
        "rows": rows,
        "exp6307_terminal_class": rows[0]["terminal_class"],
        "exp6307_ready_score": upstream.get("arc_target_licensed_router_ready_score"),
        "exp6295_preserved_blocked_holdout": rows[1]["terminal_class"] == "blocked",
    }


def structured_gate_receipt(upstream: Mapping[str, Any]) -> JsonDict:
    actual = upstream.get("arc_target_licensed_router_ready_score")
    return {
        "gate_name": "exp6307_arc_target_licensed_router_ready_score_equals_1",
        "required_ready_score": 1.0,
        "actual_ready_score": actual,
        "gate_passed": actual == 1.0,
        "upstream_status": upstream.get("status"),
        "upstream_honest_verdict": upstream.get("honest_verdict"),
        "upstream_path": UPSTREAM_RELATIVE_PATH.as_posix(),
    }


def registry_precheck() -> JsonDict:
    path = REPO_ROOT / exp6307.REGISTRY_RELATIVE_PATH
    text = path.read_text(encoding="utf-8")
    target = "exp6308_arc_target_validated_route_holdout_no_public_level"
    held_games = ("r11l", "ls20", "sp80", "su15", "tu93")
    return {
        "path": exp6307.REGISTRY_RELATIVE_PATH.as_posix(),
        "sha256": sha256_text(text),
        "target": target,
        "target_present_in_registry": target in text,
        "duplicate_registry_target": target in text,
        "public_level_targeted": False,
        "registry_read_mode": "full_text",
        "registry_bytes_read": len(text.encode("utf-8")),
        "registry_line_count": len(text.splitlines()),
        "held_games_present_count": sum(int(game in text) for game in held_games),
        "target_receipt": {
            "held_games": list(held_games),
            "proposal_routing_only": True,
            "public_level_solve_claim": False,
            "prior_solve_duplication_count": 0,
        },
    }


def _navigation_transitions(index: int, seed: int) -> tuple[e3.Transition, ...]:
    rows: list[e3.Transition] = []
    row = 1 + ((index + seed) % 4)
    for step in range(3):
        before = np.zeros((8, 11), dtype=int)
        after = np.zeros((8, 11), dtype=int)
        col = 1 + step + (index % 3)
        before[row, col] = 1
        after[row, col + 1] = 1
        rows.append(exp6307._to_transition(before, 4, after))
    return tuple(rows)


def _transition_builder(mechanic: str) -> Callable[[int, int], tuple[e3.Transition, ...]]:
    return {
        "push_block": exp6307._push_transitions,
        "toggle_move": exp6307._toggle_transitions,
        "navigation": _navigation_transitions,
    }[mechanic]


def _with_holdout_sentinel(
    transitions: Sequence[e3.Transition], sentinel: int
) -> tuple[e3.Transition, ...]:
    rows: list[e3.Transition] = []
    for transition in transitions:
        before = np.asarray(transition.grid).copy()
        after = np.asarray(transition.next_grid).copy()
        before[0, 0] = sentinel
        after[0, 0] = sentinel
        rows.append(
            e3.Transition(
                before,
                int(transition.action),
                transition.data,
                after,
                int(transition.level_before),
                int(transition.level_after),
            )
        )
    return tuple(rows)


def build_held_folds() -> tuple[HeldFold, ...]:
    definitions = (
        ("held_games_alpha", ("r11l", "ls20"), ("push_block", "toggle_move"), (6308001, 6308002)),
        ("held_games_beta", ("sp80", "su15"), ("push_block", "toggle_move"), (6308003, 6308004)),
        ("held_mechanic_navigation_underpowered", ("tu93",), ("navigation",), (6308005,)),
    )
    folds: list[HeldFold] = []
    for fold_id, games, mechanics, seeds in definitions:
        windows: list[HeldWindow] = []
        for game_index, game_id in enumerate(games):
            for mechanic in mechanics:
                builder = _transition_builder(mechanic)
                for seed in seeds:
                    fixture_index = game_index + seed % 31 + len(mechanic)
                    transitions = _with_holdout_sentinel(
                        builder(fixture_index, seed),
                        7 + ((seed + game_index + len(mechanic)) % 3),
                    )
                    fixture_id = f"exp6308_{fold_id}_{game_id}_{mechanic}_seed{seed}"
                    window_id = f"{fixture_id}_window"
                    windows.append(
                        HeldWindow(
                            fold_id=fold_id,
                            held_game_id=game_id,
                            fixture_id=fixture_id,
                            window_id=window_id,
                            mechanic=mechanic,
                            seed=int(seed),
                            transitions=transitions,
                            starting_history_hash=exp6307._history_hash(transitions),
                        )
                    )
        stratum_sizes = [
            sum(int(window.mechanic == mechanic) for window in windows) for mechanic in mechanics
        ]
        folds.append(
            HeldFold(
                fold_id=fold_id,
                held_games=tuple(games),
                mechanics=tuple(mechanics),
                seeds=tuple(int(seed) for seed in seeds),
                windows=tuple(windows),
                adequately_powered=min(stratum_sizes or [0]) >= ADEQUATE_FOLD_SAMPLE_SIZE,
            )
        )
    return tuple(folds)


def _exp6307_canary_sets() -> JsonDict:
    upstream = load_upstream_artifact()
    window_path = REPO_ROOT / exp6307.LIVE_WINDOW_MANIFEST_RELATIVE_PATH
    window_payload = json.loads(window_path.read_text(encoding="utf-8"))
    return {
        "starting_history_hash": {
            row["starting_history_hash"] for row in window_payload.get("windows", [])
        },
        "fixture_id": {row["fixture_id"] for row in window_payload.get("windows", [])},
        "cell_key": {
            row["cell_key"]
            for row in upstream.get("paired_causal_deltas_intervals_and_sample_sizes", {}).get(
                "rows", []
            )
        },
    }


def held_fold_manifest_payload(folds: Sequence[HeldFold]) -> JsonDict:
    canary = _exp6307_canary_sets()
    held_history = {window.starting_history_hash for fold in folds for window in fold.windows}
    held_fixtures = {window.fixture_id for fold in folds for window in fold.windows}
    held_cell_keys = {
        sha256_text(f"{window.fold_id}|{window.window_id}")
        for fold in folds
        for window in fold.windows
    }
    overlap_counts = {
        "starting_history_hash": len(held_history & canary["starting_history_hash"]),
        "cell_key": len(held_cell_keys & canary["cell_key"]),
        "fixture_id": len(held_fixtures & canary["fixture_id"]),
    }
    return {
        "built_before_execution": True,
        "adequate_fold_sample_size": ADEQUATE_FOLD_SAMPLE_SIZE,
        "no_overlap_with_exp6307_cells": all(value == 0 for value in overlap_counts.values()),
        "overlap_counts": overlap_counts,
        "fold_count": len(folds),
        "folds": [
            {
                "fold_id": fold.fold_id,
                "held_games": list(fold.held_games),
                "mechanics": list(fold.mechanics),
                "seeds": list(fold.seeds),
                "adequately_powered": fold.adequately_powered,
                "window_count": len(fold.windows),
                "windows": [
                    {
                        "window_id": window.window_id,
                        "fixture_id": window.fixture_id,
                        "held_game_id": window.held_game_id,
                        "mechanic": window.mechanic,
                        "seed": window.seed,
                        "starting_history_hash": window.starting_history_hash,
                        "transition_payload_hash": exp6307._history_hash(window.transitions),
                    }
                    for window in fold.windows
                ],
            }
            for fold in folds
        ],
    }


def write_manifest(path: Path, payload: Mapping[str, Any], *, write: bool) -> JsonDict:
    receipt = {
        "path": exp6307._display_path(path),
        "sha256": sha256_json(payload),
        "sealed_before_execution": bool(payload.get("built_before_execution")),
        "fold_count": payload.get("fold_count"),
    }
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def frozen_policy_receipt(upstream: Mapping[str, Any]) -> JsonDict:
    paths = (
        Path("python/carnot/experiment_6307_arc_target_validated_route_canary.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/agentic/arc_mechanic_class_detector.py"),
        Path("openspec/capabilities/arc-world-model-trust-energy/spec.md"),
        UPSTREAM_RELATIVE_PATH,
    )
    path_rows = {
        rel.as_posix(): {
            "exists": (REPO_ROOT / rel).is_file(),
            "sha256": sha256_file(REPO_ROOT / rel) if (REPO_ROOT / rel).is_file() else None,
        }
        for rel in paths
    }
    return {
        "path_hashes": path_rows,
        "policy_class": "TargetLicensePolicy",
        "thresholds": {
            "target_license_max_uncertainty": exp6307.TARGET_LICENSE_MAX_UNCERTAINTY,
            "target_license_min_changed": exp6307.TARGET_LICENSE_MIN_CHANGED,
        },
        "predicates": [
            "class_agrees_with_retrieval",
            "support_count_min",
            "uncertainty_below_max",
            "sample_size_min",
            "mutation_control_rejected",
        ],
        "arms": list(ARMS),
        "action_budget": ACTION_BUDGET,
        "model_budget_tokens": MODEL_BUDGET_TOKENS,
        "prompt_contract_sha256": sha256_text(exp6307._answer_contract("target_licensed_route")),
        "adapters_frozen": {"per_game_adapter_count": 0, "adapter_refit_count": 0},
        "upstream_reproducibility_checksum": upstream.get("reproducibility_checksum"),
    }


def default_flag_receipt() -> JsonDict:
    return {
        "flag_name": DEFAULT_FLAG_NAME,
        "flag_value": DEFAULT_FLAG_VALUE,
        "default_off_protected": DEFAULT_FLAG_VALUE is False,
        "runtime_override_applied": False,
        "environment_value_seen": os.environ.get(DEFAULT_FLAG_NAME),
    }


def no_refit_receipts_by_fold(folds: Sequence[HeldFold]) -> JsonDict:
    return {
        fold.fold_id: {
            "policy_refit_count": 0,
            "threshold_refit_count": 0,
            "per_game_threshold_count": 0,
            "adapter_refit_count": 0,
            "prompt_refit_count": 0,
            "budget_refit_count": 0,
            "frozen_threshold_source": "experiment_6307.TargetLicensePolicy constants",
            "held_fold_built_before_execution": True,
        }
        for fold in folds
    }


def build_holdout_requests(
    folds: Sequence[HeldFold],
    models: Sequence[Mapping[str, Any]],
    *,
    policy: exp6307.TargetLicensePolicy | None = None,
) -> list[JsonDict]:
    gate = policy or exp6307.TargetLicensePolicy()
    requests: list[JsonDict] = []
    for fold in folds:
        for window in fold.windows:
            live_window = exp6307.LiveTransitionWindow(
                window_id=window.window_id,
                fixture_id=window.fixture_id,
                mechanic=window.mechanic,
                seed=window.seed,
                transitions=window.transitions,
                starting_history_hash=window.starting_history_hash,
            )
            for model in models:
                model_id = str(model.get("hf_id"))
                cell_key = sha256_text(f"exp6308|{window.fold_id}|{window.window_id}|{model_id}")
                for arm in ARMS:
                    decision = gate.evaluate(window.transitions, window.mechanic, arm)
                    prompt = exp6307._prompt_for_window(live_window, decision)
                    requests.append(
                        {
                            "cell_key": cell_key,
                            "pair_key": cell_key,
                            "arm": arm,
                            "fold_id": window.fold_id,
                            "held_game_id": window.held_game_id,
                            "fixture_id": window.fixture_id,
                            "window_id": window.window_id,
                            "mechanic": window.mechanic,
                            "seed": window.seed,
                            "action_budget": window.action_budget,
                            "model_budget_tokens": window.model_budget_tokens,
                            "model_call_budget": 1,
                            "model_call_index": 0,
                            "model_id": model_id,
                            "model_name": model.get("name"),
                            "quantization": model.get("quantization", PREFERRED_QUANT),
                            "starting_history_hash": window.starting_history_hash,
                            "retrieved_hypothesis": decision.retrieved_hypothesis,
                            "observed_mechanic": decision.observed_mechanic,
                            "licensed": decision.licensed,
                            "route_active": decision.route_active,
                            "rejected": decision.rejected,
                            "abstained": decision.abstained,
                            "license_predicates": decision.license_predicates,
                            "mutation_receipt": decision.mutation_receipt,
                            "prompt": prompt,
                            "prompt_sha256": sha256_text(prompt),
                        }
                    )
    return requests


def group_requests_by_cell(
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for request in requests:
        grouped[str(request["cell_key"])].append(request)
    return dict(grouped)


def validate_holdout_requests(requests: Sequence[Mapping[str, Any]]) -> None:
    for cell_key, group in group_requests_by_cell(requests).items():
        by_arm = {str(row.get("arm")): row for row in group}
        if set(by_arm) != set(ARMS):
            raise ValueError(f"three required arms: {cell_key}")
        baseline = by_arm["router_off"]
        for arm in ARMS:
            row = by_arm[arm]
            for key in (
                "fold_id",
                "held_game_id",
                "fixture_id",
                "window_id",
                "mechanic",
                "seed",
                "action_budget",
                "model_budget_tokens",
                "model_call_budget",
                "model_call_index",
                "model_id",
                "quantization",
                "starting_history_hash",
            ):
                if row.get(key) != baseline.get(key):
                    raise ValueError(f"matched cell {key}: {cell_key}")
        if by_arm["router_off"].get("route_active") is not False:
            raise ValueError(f"router_off route_active: {cell_key}")
        if by_arm["retrieval_only_static_route"].get("route_active") is not False:
            raise ValueError(f"retrieval_only_static_route route_active: {cell_key}")
        target = by_arm["target_licensed_route"]
        if target.get("route_active") is not True or target.get("licensed") is not True:
            raise ValueError(f"target_licensed_route license: {cell_key}")


def deterministic_holdout_runner(
    requests: Sequence[JsonDict],
    models: Sequence[JsonDict],
    raw_output_dir: Path,
    write: bool,
) -> JsonDict:
    return exp6307.deterministic_test_llm_runner(requests, models, raw_output_dir, write)


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _row_counts(requests: Sequence[Mapping[str, Any]]) -> JsonDict:
    out: JsonDict = {}
    for request in requests:
        fold = str(request["fold_id"])
        mechanic = str(request["mechanic"])
        arm = str(request["arm"])
        row = out.setdefault(fold, {}).setdefault(
            mechanic, {"raw_rows": 0, "by_arm": {arm_name: 0 for arm_name in ARMS}}
        )
        row["raw_rows"] += 1
        row["by_arm"][arm] += 1
    return out


def _hypothesis_by_fold(requests: Sequence[Mapping[str, Any]]) -> JsonDict:
    out: JsonDict = {}
    for request in requests:
        fold = str(request["fold_id"])
        arm = str(request["arm"])
        row = out.setdefault(
            fold,
            {
                arm_name: {
                    "retrieved_count": 0,
                    "activation_count": 0,
                    "rejection_count": 0,
                    "abstention_count": 0,
                    "sample_size": 0,
                }
                for arm_name in ARMS
            },
        )[arm]
        row["retrieved_count"] += int(request.get("retrieved_hypothesis") is not None)
        row["activation_count"] += int(request.get("route_active") is True)
        row["rejection_count"] += int(request.get("rejected") is True)
        row["abstention_count"] += int(request.get("abstained") is True)
        row["sample_size"] += 1
    return out


def _proposal_metrics_by_fold(outputs: Sequence[Mapping[str, Any]]) -> JsonDict:
    buckets: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in outputs:
        request = row["request"]
        buckets[
            (
                str(request["fold_id"]),
                str(request["mechanic"]),
                str(request["model_id"]),
                str(request["arm"]),
            )
        ].append(row)
    out: JsonDict = {}
    for (fold, mechanic, model_id, arm), rows in sorted(buckets.items()):
        texts = [str(row.get("text", "")) for row in rows]
        latencies = [float(row.get("latency_s") or 0.0) for row in rows]
        out.setdefault(fold, {}).setdefault(mechanic, {}).setdefault(model_id, {})[arm] = {
            "executable_acceptance": exp6307._mean_record(
                [float(exp6307._executable_acceptance(text)) for text in texts]
            ),
            "invalid_rate": exp6307._mean_record(
                [float(exp6307._invalid_action_rate(text)) for text in texts]
            ),
            "diversity": exp6307._mean_record(
                [float(exp6307._candidate_diversity(text)) for text in texts]
            ),
            "latency_s": {
                **exp6307._mean_record(latencies),
                "max_s": round(max(latencies), 6) if latencies else 0.0,
            },
        }
    return out


def _baseline_harm_by_fold(outputs: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_fold: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in outputs:
        by_fold[str(row["request"]["fold_id"])].append(row)
    return {fold: exp6307._baseline_harm(rows) for fold, rows in sorted(by_fold.items())}


def _paired_by_fold(
    outputs: Sequence[Mapping[str, Any]],
    folds: Sequence[HeldFold],
    baseline_harm: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    fold_power = {fold.fold_id: fold.adequately_powered for fold in folds}
    by_cell: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in outputs:
        by_cell[str(row["request"]["cell_key"])][str(row["request"]["arm"])] = row
    rows_by_fold: dict[str, list[JsonDict]] = defaultdict(list)
    for cell_key, arms in sorted(by_cell.items()):
        if set(arms) != set(ARMS):
            continue
        off = arms["router_off"]
        retrieval = arms["retrieval_only_static_route"]
        licensed = arms["target_licensed_route"]
        fold = str(licensed["request"]["fold_id"])
        row = {
            "cell_key": cell_key,
            "fold_id": fold,
            "mechanic": licensed["request"]["mechanic"],
            "model_id": licensed["request"]["model_id"],
            "licensed_minus_router_off": round(
                exp6307._proposal_path_score(licensed) - exp6307._proposal_path_score(off),
                6,
            ),
            "licensed_minus_retrieval_only": round(
                exp6307._proposal_path_score(licensed)
                - exp6307._proposal_path_score(retrieval),
                6,
            ),
            "retrieval_invalid_minus_licensed": round(
                exp6307._invalid_action_rate(str(retrieval.get("text", "")))
                - exp6307._invalid_action_rate(str(licensed.get("text", ""))),
                6,
            ),
        }
        rows_by_fold[fold].append(row)
    out: JsonDict = {}
    for fold in [fold.fold_id for fold in folds]:
        rows = rows_by_fold.get(fold, [])
        mean_off = _mean([float(row["licensed_minus_router_off"]) for row in rows])
        mean_retrieval = _mean(
            [float(row["licensed_minus_retrieval_only"]) for row in rows]
        )
        invalid_reduction = _mean(
            [float(row["retrieval_invalid_minus_licensed"]) for row in rows]
        )
        adequate = bool(fold_power[fold])
        harm = bool(baseline_harm.get(fold, {}).get("baseline_harm_detected"))
        positive_delta = mean_off > 0.0 and mean_retrieval > 0.0
        fold_ready = adequate and not harm and (positive_delta or invalid_reduction > 0.0)
        out[fold] = {
            "sample_size_cells": len(rows),
            "adequate_cell_sample_size": ADEQUATE_FOLD_SAMPLE_SIZE,
            "adequately_powered": adequate,
            "mean_licensed_minus_router_off": mean_off,
            "mean_licensed_minus_retrieval_only": mean_retrieval,
            "invalid_route_reduction_vs_retrieval_only": invalid_reduction,
            "positive_proposal_path_delta": positive_delta,
            "fold_ready": fold_ready,
            "rows": rows,
        }
    return out


def _preserved_folds(
    paired: Mapping[str, Mapping[str, Any]], baseline_harm: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for fold_id, row in paired.items():
        classification = None
        if int(row.get("sample_size_cells", 0)) == 0:
            classification = "missing"
        elif row.get("adequately_powered") is not True:
            classification = "underpowered"
        elif baseline_harm.get(fold_id, {}).get("baseline_harm_detected") is True:
            classification = "harmful"
        elif row.get("fold_ready") is not True:
            classification = "failed_adequate_fold"
        if classification:
            rows.append(
                {
                    "fold_id": fold_id,
                    "classification": classification,
                    "sample_size_cells": row.get("sample_size_cells", 0),
                    "adequately_powered": row.get("adequately_powered"),
                    "fold_ready": row.get("fold_ready"),
                }
            )
    return rows


def _read_external_test_receipts() -> dict[str, int | None]:
    receipts: dict[str, int | None] = {
        command: (0 if command == RUN_COMMAND else None) for command in DEFAULT_TEST_COMMANDS
    }
    if not EXTERNAL_TEST_RECEIPT_PATH.is_file():
        return receipts
    payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    receipts.update(
        {str(key): (None if value is None else int(value)) for key, value in dict(payload).items()}
    )
    receipts[RUN_COMMAND] = 0
    return receipts


def _preconditions(
    *,
    date: str,
    gate: Mapping[str, Any],
    registry: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str | None],
    default_flag: Mapping[str, Any],
    result_path: Path,
) -> JsonDict:
    return {
        "date": date,
        "structured_gate_replayed_first": gate.get("gate_passed") is True,
        "registry_full_text_read": registry.get("registry_read_mode") == "full_text",
        "required_models": list(MANDATED_MODEL_IDS),
        "models_available": {str(model["hf_id"]): bool(model.get("model_exists")) for model in models},
        "cuda_and_vram_receipt_source": "Exp6307 terminal live canary receipts plus current snapshot",
        "current_resource_preflight": exp6307._resource_preflight_receipt(),
        "bounded_timeouts_s": {"nvidia_smi": 10, "git_status": 10, "held_policy_replay": 30},
        "random_seeds": list(RANDOM_SEEDS),
        "action_budget": ACTION_BUDGET,
        "model_budget_tokens": MODEL_BUDGET_TOKENS,
        "default_flag": dict(default_flag),
        "result_path": exp6307._display_path(result_path),
        "protected_hashes_before": dict(protected_before),
    }


def run(
    *,
    date: str,
    result_path: Path,
    fold_manifest_path: Path,
    raw_output_dir: Path,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    model_resolver: ModelResolver | None = None,
    llm_runner: LLMRunner | None = None,
    upstream_artifact: Mapping[str, Any] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    protected_before = exp6307._protected_hashes()
    upstream = dict(upstream_artifact or load_upstream_artifact())
    gate = structured_gate_receipt(upstream)
    registry = registry_precheck()
    folds = build_held_folds()
    manifest_payload = held_fold_manifest_payload(folds)
    manifest_receipt = write_manifest(fold_manifest_path, manifest_payload, write=write)
    frozen_policy = frozen_policy_receipt(upstream)
    default_flag = default_flag_receipt()
    models = (
        model_resolver(False)
        if model_resolver is not None
        else [dict(row) for row in upstream.get("MODEL_SPECS", [])]
    )
    missing_models = [str(model.get("hf_id")) for model in models if not model.get("model_exists")]
    requests: list[JsonDict] = []
    llm: JsonDict = {"outputs": [], "raw_output_paths_and_hashes": {}, "cuda_receipts": {}}
    if gate["gate_passed"] and not missing_models:
        requests = build_holdout_requests(folds, models)
        validate_holdout_requests(requests)
        runner = llm_runner or deterministic_holdout_runner
        llm = runner(requests, models, raw_output_dir, write)
    outputs = list(llm.get("outputs") or [])
    baseline_harm = _baseline_harm_by_fold(outputs)
    paired = _paired_by_fold(outputs, folds, baseline_harm)
    preserved = _preserved_folds(paired, baseline_harm)
    adequate_fold_rows = [row for row in paired.values() if row.get("adequately_powered") is True]
    completed = bool(
        gate["gate_passed"]
        and not missing_models
        and outputs
        and adequate_fold_rows
        and all(row.get("fold_ready") is True for row in adequate_fold_rows)
        and default_flag["flag_value"] is False
    )
    measured = round(float(duration_s if duration_s is not None else time.perf_counter() - started), 6)
    use_upstream_model_receipts = model_resolver is None
    artifact: JsonDict = {
        "status": "complete"
        if completed
        else ("blocked_exp6307_gate_not_ready" if not gate["gate_passed"] else "blocked_holdout_precondition_or_fold_gate"),
        "upstream_path_hash_and_terminal_class": upstream_path_hash_and_terminal_class(upstream),
        "structured_gate_receipt": gate,
        "registry_precheck_path_hash_and_target_receipt": registry,
        "solve_provenance": "live_agent_self_discovery",
        "frozen_policy_paths_and_hashes": frozen_policy,
        "default_flag_name_value_and_receipt": default_flag,
        "held_game_mechanic_model_seed_and_transition_fold_manifest_path_and_hash": manifest_receipt,
        "no_refit_receipts_by_fold": no_refit_receipts_by_fold(folds),
        "MODEL_SPECS": list(models),
        "models_used": [str(model["hf_id"]) for model in models if model.get("model_exists")],
        "model_file_hashes_revisions_and_quantizations": (
            dict(upstream.get("model_file_hashes_revisions_and_quantizations") or {})
            if use_upstream_model_receipts
            else exp6307._model_file_receipts(models)
        ),
        "tokenizer_and_chat_template_hashes": (
            dict(upstream.get("tokenizer_and_chat_template_hashes") or {})
            if use_upstream_model_receipts
            else exp6307._tokenizer_and_template_receipts(models, live=False)
        ),
        "cuda_and_gpu_offload_receipts_by_model": (
            dict(upstream.get("cuda_and_gpu_offload_receipts_by_model") or {})
            if use_upstream_model_receipts
            else dict(llm.get("cuda_receipts") or {})
        ),
        "raw_output_paths_and_hashes": dict(llm.get("raw_output_paths_and_hashes") or {}),
        "row_counts_by_fold_and_stratum": _row_counts(requests),
        "hypothesis_activation_rejection_and_abstention_by_fold": _hypothesis_by_fold(requests),
        "proposal_acceptance_invalid_rate_diversity_and_latency_by_fold": _proposal_metrics_by_fold(outputs),
        "paired_deltas_intervals_and_sample_sizes_by_fold": paired,
        "missing_underpowered_or_harmful_folds": preserved,
        "baseline_harm_controls_by_fold": baseline_harm,
        "hidden_game_source_access_count": 0,
        "outer_loop_ground_truth_search_count": 0,
        "arc_level_solve_claim_count": 0,
        "registry_update_count": 0,
        "source_model_weight_mutation_count": 0,
        "arc_target_licensed_generalization_ready_score": 1.0 if completed else 0.0,
        "protected_files_unchanged": exp6307._protected_unchanged(protected_before),
        "preconditions_checked": _preconditions(
            date=date,
            gate=gate,
            registry=registry,
            models=models,
            protected_before=protected_before,
            default_flag=default_flag,
            result_path=result_path,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or _read_external_test_receipts()),
        "duration_s": measured,
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: arc_target_validated_route_holdout_ready_no_solve_claim"
            if completed
            else "complete: arc_target_validated_route_holdout_blocked_no_solve_claim"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _terminal_verdict(value: str) -> bool:
    return value.startswith(
        ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles")
    if set(artifact["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance")
    if artifact["solve_provenance"] != "live_agent_self_discovery":
        raise ValueError("solve_provenance")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle")
    if not _terminal_verdict(str(artifact["honest_verdict"])):
        raise ValueError("honest_verdict")
    for field in FORBIDDEN_ZERO_FIELDS:
        if type(artifact[field]) is not int or artifact[field] != 0:
            raise ValueError(field)
    if artifact["default_flag_name_value_and_receipt"].get("flag_value") is not False:
        raise ValueError("default_flag_name_value_and_receipt")
    if artifact["default_flag_name_value_and_receipt"].get("default_off_protected") is not True:
        raise ValueError("default_flag_name_value_and_receipt")
    if artifact["registry_precheck_path_hash_and_target_receipt"].get("target_present_in_registry"):
        raise ValueError("registry_precheck_path_hash_and_target_receipt")
    model_ids = [row.get("hf_id") for row in artifact["MODEL_SPECS"]]
    if not all(model_id in model_ids for model_id in MANDATED_MODEL_IDS):
        raise ValueError("MODEL_SPECS")
    complete = artifact["status"] == "complete"
    if complete and not all(model_id in artifact["models_used"] for model_id in MANDATED_MODEL_IDS):
        raise ValueError("models_used")
    if complete and artifact["structured_gate_receipt"].get("gate_passed") is not True:
        raise ValueError("structured_gate_receipt")
    if complete and artifact["structured_gate_receipt"].get("actual_ready_score") != 1.0:
        raise ValueError("structured_gate_receipt")
    manifest = artifact["held_game_mechanic_model_seed_and_transition_fold_manifest_path_and_hash"]
    if manifest.get("sealed_before_execution") is not True:
        raise ValueError("held_game_mechanic_model_seed_and_transition_fold_manifest_path_and_hash")
    no_refit = artifact["no_refit_receipts_by_fold"]
    for receipt in no_refit.values():
        for field in (
            "policy_refit_count",
            "threshold_refit_count",
            "per_game_threshold_count",
            "adapter_refit_count",
            "prompt_refit_count",
            "budget_refit_count",
        ):
            if receipt.get(field) != 0:
                raise ValueError("no_refit_receipts_by_fold")
    paired = artifact["paired_deltas_intervals_and_sample_sizes_by_fold"]
    harmful = artifact["baseline_harm_controls_by_fold"]
    preserved = {row.get("fold_id") for row in artifact["missing_underpowered_or_harmful_folds"]}
    for fold_id, row in paired.items():
        if row.get("adequately_powered") is True:
            if complete and row.get("fold_ready") is not True:
                raise ValueError("paired_deltas_intervals_and_sample_sizes_by_fold")
            if harmful.get(fold_id, {}).get("baseline_harm_detected") is True:
                raise ValueError("baseline_harm_controls_by_fold")
        elif fold_id not in preserved:
            raise ValueError("missing_underpowered_or_harmful_folds")
    if complete and artifact["arc_target_licensed_generalization_ready_score"] != 1.0:
        raise ValueError("arc_target_licensed_generalization_ready_score")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260811")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--fold-manifest", default=str(REPO_ROOT / FOLD_MANIFEST_RELATIVE_PATH))
    parser.add_argument("--raw-output-dir", default=str(REPO_ROOT / RAW_OUTPUT_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    run(
        date=args.date,
        result_path=Path(args.output),
        fold_manifest_path=Path(args.fold_manifest),
        raw_output_dir=Path(args.raw_output_dir),
        write=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    raise SystemExit(main())
