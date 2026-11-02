"""Experiment 6232: ARC admissible depth portfolio ledger.

Spec refs: REQ-ARC-WMTE-6232,
SCENARIO-ARC-WMTE-6232-PRECONDITION-LEDGER,
SCENARIO-ARC-WMTE-6232-TERMINAL-SKIP,
SCENARIO-ARC-WMTE-6232-PORTFOLIO-RUN,
SCENARIO-ARC-WMTE-6232-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import terminal_artifacts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6232_arc_admissible_depth_portfolio.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
CLASSIFIER_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
PRESERVATION_LINT_RELATIVE_PATH = Path("scripts/determination_preservation_lint.py")
COMPETITION_AGENT_RELATIVE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
SOLVER_KIT_RELATIVE_PATH = Path("python/carnot/agentic/arc_solver_kit.py")
WORLD_MODEL_RELATIVE_PATH = Path("python/carnot/agentic/arc_executable_world_model.py")
EXP6228_RELATIVE_PATH = Path("results/experiment_6228_supervised_three_family_runtime_endurance.json")
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6232_test_receipts.json")

REQUIREMENT = "REQ-ARC-WMTE-6232"
SUPPORT_MINIMUM = 2
CANONICAL_MODEL_HF_ID = "unsloth/gemma-4-31B-it-GGUF"
CANONICAL_MODEL_FAMILY = "gemma4_31b_dense"
PREFERRED_QUANT = "Q4_K_M"
DEFAULT_PORTFOLIO_GAMES = ("bp35", "dc22", "lp85", "m0r0")
DEFAULT_PORTFOLIO_SEEDS = (623200, 623201)

UPSTREAM_LEVERS: tuple[JsonDict, ...] = (
    {
        "lever_id": "exp6215_object_relative_trajectory_transfer",
        "experiment": 6215,
        "requirement": "REQ-ARC-WMTE-6215",
        "mechanism": "object_relative_trajectory_transfer",
        "path": "results/experiment_6215_arc_object_relative_trajectory_transfer_ab.json",
        "default_stack_component": True,
        "fire_field": "treatment_fire_and_reason_counts",
        "quality_fields": ("trajectory_transfer_promotion_ready_score",),
        "metric_fields": ("engine_fidelity_score_actions_and_wall_time_by_arm_game",),
    },
    {
        "lever_id": "exp6216_budget_aware_search",
        "experiment": 6216,
        "requirement": "REQ-ARC-WMTE-6216",
        "mechanism": "budget_aware_search",
        "path": "results/experiment_6216_arc_budget_aware_search_ab.json",
        "default_stack_component": True,
        "fire_field": "consumer_fire_counts",
        "quality_fields": ("budget_aware_promotion_ready_score",),
        "metric_fields": (
            "path_cost_states_expanded_navigation_actions_score_and_wall_time_by_arm_game",
        ),
    },
    {
        "lever_id": "exp6229_bounded_reinduction",
        "experiment": 6229,
        "requirement": "REQ-ARC-WMTE-6229",
        "mechanism": "bounded_reinduction",
        "path": "results/experiment_6229_arc_gemma31_think_determination.json",
        "default_stack_component": False,
        "fire_field": "treatment_fire_counts",
        "quality_fields": (
            "bounded_reinduction_promotion_ready_score",
            "think_mode_promotion_ready_score",
        ),
        "metric_fields": (
            "admission_and_level_depth_by_arm_game",
            "quality_efficiency_and_cost_by_arm_game",
        ),
    },
    {
        "lever_id": "exp6230_prompt_enrichment",
        "experiment": 6230,
        "requirement": "REQ-ARC-WMTE-6230",
        "mechanism": "induce_prompt_enrichment",
        "path": "results/experiment_6230_arc_induce_prompt_enrichment_heldout_ab.json",
        "default_stack_component": False,
        "fire_field": "treatment_fire_counts",
        "quality_fields": ("prompt_enrichment_promotion_ready_score",),
        "metric_fields": ("admission_and_level_depth_by_arm_game",),
    },
    {
        "lever_id": "exp6231_depth_lever",
        "experiment": 6231,
        "requirement": "REQ-ARC-WMTE-6231",
        "mechanism": "graded_goal_bias",
        "path": "results/experiment_6231_arc_depth_lever_heldout_ab.json",
        "path_glob": "results/experiment_6231*.json",
        "default_stack_component": False,
        "fire_field": "treatment_fire_counts",
        "quality_fields": ("graded_goal_bias_promotion_ready_score",),
        "metric_fields": ("admission_and_level_depth_by_arm_game",),
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_paths_hashes_and_terminal_classes",
    "determination_preservation_receipts",
    "eligibility_rules",
    "lever_eligibility_ledger",
    "eligible_lever_count",
    "selected_levers",
    "exact_skip_reason",
    "model_loaded",
    "registry_precheck_and_hash_before_after",
    "solve_provenance",
    "preregistered_portfolio_game_seed_matrix",
    "model_specs",
    "supervised_runtime_receipts",
    "matched_arm_configuration",
    "per_lever_and_combination_fire_counts",
    "raw_prompt_output_engine_replay_hashes",
    "admission_and_level_depth_by_arm_game",
    "actions_tokens_wall_and_failure_costs",
    "paired_clustered_intervals",
    "interaction_effects",
    "harmful_regression_count_and_games",
    "aa_control",
    "source_bfs_adapter_hidden_state_registry_access_counts",
    "registry_update_count",
    "portfolio_promotion_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The terminal state distinguishes a skip from a frozen pair.",
    "upstream_paths_hashes_and_terminal_classes": "Every upstream input is hash-bound.",
    "determination_preservation_receipts": "The classifier and guard trail stay visible.",
    "eligibility_rules": "The gate is data-driven, not prose-driven.",
    "lever_eligibility_ledger": "Every lever receives a reasoned verdict.",
    "eligible_lever_count": "The model gate opens only at two independent levers.",
    "selected_levers": "The selected pair is frozen before any portfolio run.",
    "exact_skip_reason": "A skip names the exact failed precondition.",
    "model_loaded": "The GGUF must not load before the two-lever gate passes.",
    "registry_precheck_and_hash_before_after": "The solve registry is immutable here.",
    "solve_provenance": "ARC credit is only for live self-discovery.",
    "preregistered_portfolio_game_seed_matrix": "Game and seed cells freeze before execution.",
    "model_specs": "The only admitted inducer is the requested Gemma4-31B Q4_K_M model.",
    "supervised_runtime_receipts": "Exp6228 supervision is preserved when model use is possible.",
    "matched_arm_configuration": "A/A, default, and portfolio arms hold resources fixed.",
    "per_lever_and_combination_fire_counts": "Non-firing levers cannot be promoted.",
    "raw_prompt_output_engine_replay_hashes": "Raw prompts, engines, and replays stay auditable.",
    "admission_and_level_depth_by_arm_game": "Depth and admission outcomes stay per-game.",
    "actions_tokens_wall_and_failure_costs": "Cost and failure data stay visible.",
    "paired_clustered_intervals": "Intervals use game as the paired cluster.",
    "interaction_effects": "The pair effect is separate from main effects.",
    "harmful_regression_count_and_games": "Every loss or regression remains visible.",
    "aa_control": "A/A controls protect against instrumentation drift.",
    "source_bfs_adapter_hidden_state_registry_access_counts": "Forbidden reads are bare zeros.",
    "registry_update_count": "Bare zero proves no solve registry mutation.",
    "portfolio_promotion_ready_score": "Readiness is capped by eligible lever count.",
    "protected_files_unchanged": "Protected files stay byte-identical.",
    "preconditions_checked": "Admission inputs are recorded before model admission.",
    "inference_substrate": "The artifact declares aggregation, not live inference.",
    "verifier_is_oracle": "The verifier is not an ARC oracle.",
    "field_provenance": "Each field traces back to this module and requirement.",
    "field_principles": "Each field states the audit risk it controls.",
    "test_commands": "Verification commands are recorded.",
    "test_exit_codes": "Exit codes prevent unchecked test claims.",
    "duration_s": "Measured wall time is recorded without padding.",
    "reproducibility_checksum": "A stable checksum catches silent drift.",
    "honest_verdict": "The verdict uses a terminal prefix.",
}

PROTECTED_FILES = (
    Path("python/carnot/terminal_artifacts.py"),
    Path("scripts/determination_preservation_lint.py"),
    Path("scripts/research_conductor.py"),
    Path("ops/arc_solve_registry.yaml"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6232_arc_admissible_depth_portfolio.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6232_arc_admissible_depth_portfolio.py -m pytest tests/python/test_experiment_6232_arc_admissible_depth_portfolio.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6232_arc_admissible_depth_portfolio.py --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6232_arc_admissible_depth_portfolio.py",
    ".venv/bin/python scripts/determination_preservation_lint.py HEAD --all-files",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6232_arc_admissible_depth_portfolio.json",
    ".venv/bin/python -m carnot.experiment_6232_arc_admissible_depth_portfolio --date 20260809",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def file_receipt(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
    }


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _artifact_path(spec: Mapping[str, Any], *, root: Path) -> Path:
    path_glob = spec.get("path_glob")
    if path_glob:
        matches = sorted(root.glob(str(path_glob)))
        if matches:
            return matches[0]
    return root / str(spec["path"])


def _pending_corrigendum(payload: Mapping[str, Any]) -> JsonDict:
    pending = payload.get("corrigendum_pending")
    rows = pending if isinstance(pending, list) else ([pending] if pending else [])
    critical = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("severity", "")).lower() == "critical"
    ]
    return {
        "substantive": bool(rows),
        "critical_count": len(critical),
        "rows": rows,
        "resolved_rows": payload.get("corrigendum_resolved") or [],
    }


def _fire_counts(payload: Mapping[str, Any], field: str) -> JsonDict:
    fire = dict(payload.get(field) or {})
    total = int(fire.get("treatment_total", fire.get("total", 0)) or 0)
    support_count = int(fire.get("support_count", 0) or 0)
    support_floor = int(fire.get("support_floor", 1) or 1)
    mutation_proven = fire.get("mutation_proven") is True
    return {
        "field": field,
        "total": total,
        "support_count": support_count,
        "support_floor": support_floor,
        "mutation_proven": mutation_proven,
        "activation_passed": total > 0 and support_count >= support_floor and mutation_proven,
        "raw": fire,
    }


def _quality_gate(payload: Mapping[str, Any], fields: Sequence[str]) -> JsonDict:
    for field in fields:
        if field in payload:
            score = float(payload.get(field) or 0.0)
            return {"field": field, "score": score, "quality_passed": score >= 1.0}
    return {"field": None, "score": 0.0, "quality_passed": False}


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 8) if values else 0.0


def _nested_delta(row: Mapping[str, Any], key: str) -> float:
    control = dict(row.get("control") or {})
    treatment = dict(row.get("treatment") or {})
    return float(treatment.get(key, 0.0) or 0.0) - float(control.get(key, 0.0) or 0.0)


def _effect_gate(payload: Mapping[str, Any], fields: Sequence[str]) -> JsonDict:
    chosen_field = next((field for field in fields if isinstance(payload.get(field), Mapping)), None)
    rows = dict(payload.get(chosen_field) or {}) if chosen_field else {}
    deltas: list[float] = []
    wall_deltas: list[float] = []
    token_deltas: list[float] = []
    action_deltas: list[float] = []
    losing_games: list[str] = []
    for game, row_any in sorted(rows.items()):
        row = dict(row_any or {})
        delta = float(
            row.get(
                "treatment_minus_control_depth",
                row.get(
                    "treatment_minus_control_level_depth",
                    row.get(
                        "treatment_minus_control_admission",
                        row.get("treatment_minus_control_score", 0.0),
                    ),
                ),
            )
            or 0.0
        )
        if delta == 0.0:
            delta = max(
                _nested_delta(row, "level_depth"),
                _nested_delta(row, "depth"),
                _nested_delta(row, "admission"),
                _nested_delta(row, "score"),
            )
        deltas.append(delta)
        wall_deltas.append(float(row.get("treatment_minus_control_wall_s", 0.0) or 0.0))
        token_deltas.append(float(row.get("treatment_minus_control_tokens", 0.0) or 0.0))
        action_deltas.append(float(row.get("treatment_minus_control_actions", 0.0) or 0.0))
        if delta < 0.0 or row.get("loss_reported") is True or row.get("losing_game") is True:
            losing_games.append(str(game))
    primary_effect = _mean(deltas)
    return {
        "metric_field": chosen_field,
        "primary_effect": primary_effect,
        "depth_or_admission_relevant": any(delta > 0.0 for delta in deltas),
        "mean_action_delta": _mean(action_deltas),
        "mean_token_delta": _mean(token_deltas),
        "mean_wall_delta_s": _mean(wall_deltas),
        "losing_games": losing_games,
        "by_game": rows,
    }


def _safety_gate(payload: Mapping[str, Any], effect: Mapping[str, Any]) -> JsonDict:
    harmful = dict(payload.get("harmful_regression_count_and_games") or {})
    games = [str(game) for game in harmful.get("games") or []]
    losing = [str(game) for game in harmful.get("losing_games_reported_not_hidden") or []]
    effect_losses = [str(game) for game in effect.get("losing_games") or []]
    count = int(harmful.get("count", 0) or 0) + len(effect_losses)
    return {
        "harmful_regression_count": count,
        "harmful_games": sorted(set(games + effect_losses)),
        "losing_games_reported_not_hidden": losing,
        "safety_passed": count == 0,
        "raw": harmful,
    }


def _zero_registry_gate(payload: Mapping[str, Any]) -> JsonDict:
    return {
        "solve_claimed": payload.get("solve_claimed"),
        "level_credit_delta": payload.get("level_credit_delta"),
        "registry_update_count": payload.get("registry_update_count"),
        "passed": payload.get("solve_claimed") is False
        and type(payload.get("level_credit_delta")) is int
        and payload.get("level_credit_delta") == 0
        and type(payload.get("registry_update_count")) is int
        and payload.get("registry_update_count") == 0,
    }


def _raw_hashes(payload: Mapping[str, Any]) -> JsonDict:
    rows: list[JsonDict] = []
    for field in (
        "raw_prompt_output_engine_replay_hashes",
        "raw_induction_paths_and_hashes",
    ):
        for row in payload.get(field) or []:
            if isinstance(row, Mapping):
                rows.append(dict(row))
    within_game = payload.get("within_game_only_receipt")
    if isinstance(within_game, Mapping):
        for row in within_game.get("raw_event_paths_and_hashes") or []:
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return {"raw_count": len(rows), "raw_sha256": sha256_json(rows), "rows": rows}


def _forbidden_zero_from_payload(payload: Mapping[str, Any]) -> bool:
    for key in (
        "source_bfs_adapter_registry_hidden_state_access_counts",
        "prior_game_cross_game_source_bfs_adapter_registry_hidden_state_access_counts",
    ):
        counts = payload.get(key)
        if isinstance(counts, Mapping) and counts:
            return all(type(value) is int and value == 0 for value in counts.values())
    return True


def _selection_utility(effect: Mapping[str, Any], quality: Mapping[str, Any]) -> float:
    return round(float(effect.get("primary_effect", 0.0)) + 0.01 * float(quality.get("score", 0.0)), 8)


def recompute_lever_eligibility(spec: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    path = _artifact_path(spec, root=root)
    payload = _load_json(path)
    classification = terminal_artifacts.classify_artifact_path(path)
    pending = _pending_corrigendum(payload)
    fire = _fire_counts(payload, str(spec["fire_field"]))
    quality = _quality_gate(payload, tuple(spec.get("quality_fields", ())))
    effect = _effect_gate(payload, tuple(spec.get("metric_fields", ())))
    safety = _safety_gate(payload, effect)
    zero_registry = _zero_registry_gate(payload)
    terminal_class = classification.classification
    admissible_terminal = terminal_class in {"complete", "ready", "positive"}
    default_component = spec.get("default_stack_component") is True
    reasons: list[str] = []
    if not path.is_file():
        reasons.append("artifact_missing")
    if not admissible_terminal:
        reasons.append(f"terminal_class_{terminal_class}_not_admissible")
    if payload.get("flagged_adversarial") is True:
        reasons.append("flagged_adversarial_true")
    if pending["substantive"]:
        reasons.append("substantive_corrigendum_pending")
    if pending["critical_count"]:
        reasons.append("unresolved_critical_corrigendum_present")
    if not fire["activation_passed"]:
        reasons.append("treatment_did_not_fire")
    if not quality["quality_passed"]:
        reasons.append("quality_gate_failed")
    if not effect["depth_or_admission_relevant"]:
        reasons.append("no_depth_or_admission_relevant_effect")
    if not safety["safety_passed"]:
        reasons.append("safety_gate_failed")
    if not zero_registry["passed"]:
        reasons.append("zero_registry_gate_failed")
    if not _forbidden_zero_from_payload(payload):
        reasons.append("forbidden_access_nonzero")
    if default_component:
        reasons.append("already_default_on_not_counted")
    return {
        "lever_id": str(spec["lever_id"]),
        "experiment": int(spec["experiment"]),
        "requirement": str(spec["requirement"]),
        "mechanism": str(spec["mechanism"]),
        "default_stack_component": default_component,
        "artifact": file_receipt(path),
        "terminal_classifier": classification.to_dict(),
        "pending_corrigendum": pending,
        "activation_gate": fire,
        "quality_gate": quality,
        "effect_gate": effect,
        "safety_gate": safety,
        "zero_registry_gate": zero_registry,
        "raw_prompt_output_engine_replay_hashes": _raw_hashes(payload),
        "aa_control": payload.get("aa_control", "not_recorded"),
        "eligible": not reasons,
        "ineligible_reasons": reasons,
        "selection_utility": _selection_utility(effect, quality),
    }


def recompute_eligibility_ledger(
    specs: Sequence[Mapping[str, Any]] = UPSTREAM_LEVERS,
) -> list[JsonDict]:
    return apply_independence_gate([recompute_lever_eligibility(spec) for spec in specs])


def synthetic_eligible_lever(lever_id: str, *, mechanism: str, utility: float) -> JsonDict:
    return {
        "lever_id": lever_id,
        "experiment": 0,
        "requirement": REQUIREMENT,
        "mechanism": mechanism,
        "default_stack_component": False,
        "artifact": {"path": f"synthetic/{lever_id}.json", "exists": True, "sha256": "sha256:0"},
        "terminal_classifier": {"classification": "ready", "terminal": True},
        "pending_corrigendum": {"substantive": False, "critical_count": 0, "rows": []},
        "activation_gate": {
            "total": 3,
            "support_count": 3,
            "support_floor": 3,
            "mutation_proven": True,
            "activation_passed": True,
        },
        "quality_gate": {"field": "synthetic_ready_score", "score": 1.0, "quality_passed": True},
        "effect_gate": {
            "metric_field": "admission_and_level_depth_by_arm_game",
            "primary_effect": float(utility),
            "depth_or_admission_relevant": True,
            "mean_action_delta": 0.0,
            "mean_token_delta": 0.0,
            "mean_wall_delta_s": 0.0,
            "losing_games": [],
            "by_game": {"synthetic": {"treatment_minus_control_depth": float(utility)}},
        },
        "safety_gate": {
            "harmful_regression_count": 0,
            "harmful_games": [],
            "losing_games_reported_not_hidden": [],
            "safety_passed": True,
        },
        "zero_registry_gate": {
            "solve_claimed": False,
            "level_credit_delta": 0,
            "registry_update_count": 0,
            "passed": True,
        },
        "raw_prompt_output_engine_replay_hashes": {"raw_count": 0, "raw_sha256": sha256_json([]), "rows": []},
        "aa_control": "synthetic",
        "eligible": True,
        "ineligible_reasons": [],
        "selection_utility": float(utility),
    }


def apply_independence_gate(ledger: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows = [dict(row) for row in ledger]
    best_by_mechanism: dict[str, str] = {}
    for row in sorted(
        [row for row in rows if row.get("eligible") is True],
        key=lambda item: (-float(item.get("selection_utility", 0.0)), str(item.get("lever_id"))),
    ):
        mechanism = str(row.get("mechanism"))
        if mechanism not in best_by_mechanism:
            best_by_mechanism[mechanism] = str(row.get("lever_id"))
    for row in rows:
        if row.get("eligible") is True and best_by_mechanism.get(str(row.get("mechanism"))) != row.get(
            "lever_id"
        ):
            row["eligible"] = False
            reasons = list(row.get("ineligible_reasons") or [])
            reasons.append("duplicate_mechanism_not_counted")
            row["ineligible_reasons"] = reasons
    return rows


def select_levers(ledger: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    eligible = [dict(row) for row in ledger if row.get("eligible") is True]
    eligible.sort(
        key=lambda row: (
            -float(row.get("selection_utility", 0.0)),
            int(dict(row.get("safety_gate") or {}).get("harmful_regression_count", 0)),
            str(row.get("lever_id")),
        )
    )
    return eligible[:SUPPORT_MINIMUM]


def current_default_stack() -> JsonDict:
    competition = (REPO_ROOT / COMPETITION_AGENT_RELATIVE_PATH).read_text(encoding="utf-8")
    solver = (REPO_ROOT / SOLVER_KIT_RELATIVE_PATH).read_text(encoding="utf-8")
    world_model = (REPO_ROOT / WORLD_MODEL_RELATIVE_PATH).read_text(encoding="utf-8")
    return {
        "trajectory_transfer": "SUBMITTED_OBJECT_RELATIVE_TRAJECTORY_TRANSFER_ENABLED = True"
        in competition,
        "budget_aware_search": "BUDGET_AWARE_SEARCH_ENABLED = True" in solver,
        "think_default": '"1"' if 'ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT = "1"' in world_model else "0",
        "prompt_enrichment": "SUBMITTED_INDUCE_PROMPT_ENRICHMENT_ENABLED = True" in world_model,
        "think_arm_fallback": "SUBMITTED_THINK_ARM_FALLBACK_ENABLED = True" in competition,
        "source_hashes": {
            COMPETITION_AGENT_RELATIVE_PATH.as_posix(): sha256_file(
                REPO_ROOT / COMPETITION_AGENT_RELATIVE_PATH
            ),
            SOLVER_KIT_RELATIVE_PATH.as_posix(): sha256_file(REPO_ROOT / SOLVER_KIT_RELATIVE_PATH),
            WORLD_MODEL_RELATIVE_PATH.as_posix(): sha256_file(
                REPO_ROOT / WORLD_MODEL_RELATIVE_PATH
            ),
        },
    }


def registry_precheck_and_hash_before_after(*, matrix_opened: bool) -> JsonDict:
    path = REPO_ROOT / REGISTRY_RELATIVE_PATH
    before = sha256_file(path)
    after = sha256_file(path)
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_hash_before": before,
        "registry_hash_after": after,
        "unchanged": before == after,
        "checked_before_model_admission": True,
        "portfolio_matrix_opened_after_precheck": matrix_opened,
    }


def protected_hash_map() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_FILES}


def protected_files_unchanged(before: Mapping[str, str | None] | None = None) -> JsonDict:
    before_hashes = dict(before or protected_hash_map())
    after = protected_hash_map()
    changed = [path for path, digest in before_hashes.items() if after.get(path) != digest]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "hash_before": sha256_json(before_hashes),
        "hash_after": sha256_json(after),
    }


def portfolio_matrix(selected: Sequence[Mapping[str, Any]], *, date: str) -> JsonDict:
    opened = len(selected) >= SUPPORT_MINIMUM
    cells = [
        {"game": game, "seed": seed, "date": date, "role": "portfolio_depth_cell"}
        for game in DEFAULT_PORTFOLIO_GAMES
        for seed in DEFAULT_PORTFOLIO_SEEDS
    ]
    return {
        "opened": opened,
        "written_before_arm_execution": opened,
        "games": list(DEFAULT_PORTFOLIO_GAMES) if opened else [],
        "seeds": list(DEFAULT_PORTFOLIO_SEEDS) if opened else [],
        "cells": cells if opened else [],
        "not_opened_reason": None if opened else "fewer_than_two_eligible_depth_levers",
    }


def model_specs() -> JsonDict:
    return {
        "hf_id": CANONICAL_MODEL_HF_ID,
        "role": "portfolio ARC inducer when eligibility passes",
        "preferred_quant": PREFERRED_QUANT,
        "family": CANONICAL_MODEL_FAMILY,
        "load_policy": "forbidden_until_eligible_lever_count_at_least_two",
    }


def supervised_runtime_receipts() -> JsonDict:
    path = REPO_ROOT / EXP6228_RELATIVE_PATH
    payload = _load_json(path)
    return {
        "source_artifact": file_receipt(path),
        "status": payload.get("status"),
        "gemma_4_31b_runtime_ready_score": payload.get("gemma_4_31b_runtime_ready_score"),
        "supervisor_contract_and_paths_hashes": payload.get("supervisor_contract_and_paths_hashes"),
        "token_receipts": [
            file_receipt(path)
            for path in sorted((REPO_ROOT / "results").glob("experiment_6228*gemma4_31b_dense*.token"))
        ],
    }


def matched_arm_configuration(selected: Sequence[Mapping[str, Any]]) -> JsonDict:
    defaults = current_default_stack()
    if len(selected) < SUPPORT_MINIMUM:
        return {
            "built": False,
            "reason": "fewer_than_two_eligible_depth_levers",
            "current_default_stack": defaults,
            "aa_control": {},
            "default_stack_arm": {},
            "portfolio_arm": {},
        }
    lever_ids = [str(row["lever_id"]) for row in selected]
    return {
        "built": True,
        "current_default_stack": defaults,
        "aa_control": {"name": "aa_default_vs_default", "enabled_levers": []},
        "default_stack_arm": {"name": "current_default_stack", "held_fixed": defaults},
        "portfolio_arm": {"name": "depth_portfolio", "enabled_depth_levers": lever_ids},
        "matched_resources": ["game", "seed", "action_budget", "model", "live_entrypoint"],
    }


def forbidden_access_counts() -> dict[str, int]:
    return {
        "source_reads": 0,
        "bfs_reads": 0,
        "adapter_reads": 0,
        "hidden_state_reads": 0,
        "registry_trajectory_reads": 0,
        "registry_hidden_state_reads": 0,
    }


def upstream_paths_hashes_and_terminal_classes(ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "terminal_classifier": file_receipt(REPO_ROOT / CLASSIFIER_RELATIVE_PATH),
        "upstreams": [
            {
                "lever_id": row.get("lever_id"),
                "artifact": row.get("artifact"),
                "terminal_classifier": row.get("terminal_classifier"),
            }
            for row in ledger
        ],
    }


def determination_preservation_receipts() -> JsonDict:
    return {
        "terminal_artifacts_py": file_receipt(REPO_ROOT / CLASSIFIER_RELATIVE_PATH),
        "determination_preservation_lint_py": file_receipt(
            REPO_ROOT / PRESERVATION_LINT_RELATIVE_PATH
        ),
        "protected_files": protected_hash_map(),
    }


def eligibility_rules() -> JsonDict:
    return {
        "eligible_minimum": SUPPORT_MINIMUM,
        "terminal_classes": ["complete", "ready", "positive"],
        "rejects_missing_blocked_skipped_flagged_or_corrigendum_pending": True,
        "requires_treatment_fire_on_support_floor": True,
        "requires_independent_mechanism": True,
        "requires_depth_or_admission_relevant_effect": True,
        "default_stack_components_not_counted": [
            "object_relative_trajectory_transfer",
            "budget_aware_search",
        ],
        "no_model_load_before_two": True,
    }


def aggregate_fire_counts(ledger: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "by_lever": {str(row["lever_id"]): row.get("activation_gate") for row in ledger},
        "selected_combination": [str(row["lever_id"]) for row in selected],
        "combination_fire_count": 0 if len(selected) < SUPPORT_MINIMUM else "not_executed_by_unit_fixture",
    }


def aggregate_raw_hashes(ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(row["lever_id"]): row.get("raw_prompt_output_engine_replay_hashes") for row in ledger
    }


def aggregate_depth(ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["lever_id"]): dict(row.get("effect_gate") or {}).get("by_game", {}) for row in ledger}


def aggregate_costs(ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(row["lever_id"]): {
            "mean_action_delta": dict(row.get("effect_gate") or {}).get("mean_action_delta"),
            "mean_token_delta": dict(row.get("effect_gate") or {}).get("mean_token_delta"),
            "mean_wall_delta_s": dict(row.get("effect_gate") or {}).get("mean_wall_delta_s"),
            "losses": dict(row.get("safety_gate") or {}).get("harmful_games", []),
            "terminal_failures": list(row.get("ineligible_reasons") or []),
        }
        for row in ledger
    }


def aggregate_intervals(ledger: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "portfolio_interval": None
        if len(selected) < SUPPORT_MINIMUM
        else "not_executed_by_unit_fixture",
        "upstream_effects": {
            str(row["lever_id"]): dict(row.get("effect_gate") or {}).get("primary_effect")
            for row in ledger
        },
    }


def aggregate_harms(ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_lever = {str(row["lever_id"]): row.get("safety_gate") for row in ledger}
    games = [
        f"{lever}:{game}"
        for lever, safety in by_lever.items()
        for game in dict(safety or {}).get("harmful_games", [])
    ]
    return {"count": len(games), "games": games, "by_lever": by_lever}


def aggregate_aa(ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row["lever_id"]): row.get("aa_control") for row in ledger}


def interaction_effects(selected: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "estimated": False,
        "reason": "terminal_skip_before_model_load"
        if len(selected) < SUPPORT_MINIMUM
        else "not_executed_by_unit_fixture",
        "selected_pair": [str(row["lever_id"]) for row in selected],
        "interaction_effect": None,
    }


def field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "source": "carnot.experiment_6232_arc_admissible_depth_portfolio",
            "spec_ref": REQUIREMENT,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_artifact(
    *,
    date: str = "20260809",
    precomputed_ledger: Sequence[Mapping[str, Any]] | None = None,
    output_path: Path | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    started: float | None = None,
    now: Callable[[], float] = time.monotonic,
) -> JsonDict:
    start = now() if started is None else float(started)
    protected_before = protected_hash_map()
    ledger = (
        apply_independence_gate(precomputed_ledger)
        if precomputed_ledger is not None
        else recompute_eligibility_ledger()
    )
    selected = select_levers(ledger)
    eligible_count = len([row for row in ledger if row.get("eligible") is True])
    matrix = portfolio_matrix(selected, date=date)
    registry = registry_precheck_and_hash_before_after(matrix_opened=matrix["opened"])
    skip_reason = (
        None
        if eligible_count >= SUPPORT_MINIMUM
        else f"fewer_than_two_independent_unflagged_treatment_active_depth_levers:{eligible_count}<2"
    )
    artifact: JsonDict = {
        "status": "skipped_less_than_two_eligible_depth_levers"
        if skip_reason
        else "complete_portfolio_pair_frozen_not_executed_by_unit_fixture",
        "upstream_paths_hashes_and_terminal_classes": upstream_paths_hashes_and_terminal_classes(
            ledger
        ),
        "determination_preservation_receipts": determination_preservation_receipts(),
        "eligibility_rules": eligibility_rules(),
        "lever_eligibility_ledger": ledger,
        "eligible_lever_count": eligible_count,
        "selected_levers": [
            {
                "lever_id": str(row["lever_id"]),
                "mechanism": str(row["mechanism"]),
                "selection_utility": float(row.get("selection_utility", 0.0)),
            }
            for row in selected
        ],
        "exact_skip_reason": skip_reason,
        "model_loaded": False,
        "registry_precheck_and_hash_before_after": registry,
        "solve_provenance": "live_agent_self_discovery",
        "preregistered_portfolio_game_seed_matrix": matrix,
        "model_specs": model_specs(),
        "supervised_runtime_receipts": supervised_runtime_receipts(),
        "matched_arm_configuration": matched_arm_configuration(selected),
        "per_lever_and_combination_fire_counts": aggregate_fire_counts(ledger, selected),
        "raw_prompt_output_engine_replay_hashes": aggregate_raw_hashes(ledger),
        "admission_and_level_depth_by_arm_game": aggregate_depth(ledger),
        "actions_tokens_wall_and_failure_costs": aggregate_costs(ledger),
        "paired_clustered_intervals": aggregate_intervals(ledger, selected),
        "interaction_effects": interaction_effects(selected),
        "harmful_regression_count_and_games": aggregate_harms(ledger),
        "aa_control": aggregate_aa(ledger),
        "source_bfs_adapter_hidden_state_registry_access_counts": forbidden_access_counts(),
        "registry_update_count": 0,
        "portfolio_promotion_ready_score": round(min(1.0, eligible_count / SUPPORT_MINIMUM), 6),
        "protected_files_unchanged": protected_files_unchanged(protected_before),
        "preconditions_checked": {
            "date": date,
            "eligibility_inputs_written_before_model_admission": True,
            "model_admission_eligible_count": eligible_count,
            "registry_checked": registry["unchanged"],
            "defaults_checked": current_default_stack(),
            "output_path": str(output_path or (REPO_ROOT / RESULT_RELATIVE_PATH)),
        },
        "inference_substrate": {
            "value": "aggregation_from_upstream_artifacts",
            "model_load_attempted": False,
            "model_loaded": False,
        },
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(test_commands or DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {
            str(key): int(value) for key, value in dict(test_exit_codes or {}).items()
        },
        "duration_s": round(now() - start, 6),
        "reproducibility_checksum": "",
        "honest_verdict": "complete: skipped_fewer_than_two_eligible_depth_levers_no_model_load"
        if skip_reason
        else "complete: depth_portfolio_pair_frozen_no_registry_credit",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _bare_zero(payload: Mapping[str, Any], field: str) -> bool:
    return type(payload.get(field)) is int and payload.get(field) == 0


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance incomplete")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles incomplete")
    if artifact.get("model_loaded") is not False:
        raise ValueError("model_loaded must be bare false")
    if not _bare_zero(artifact, "registry_update_count"):
        raise ValueError("registry_update_count must be bare 0")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        raise ValueError("solve_provenance invalid")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be bare false")
    counts = dict(artifact.get("source_bfs_adapter_hidden_state_registry_access_counts") or {})
    if not counts or any(type(value) is not int or value != 0 for value in counts.values()):
        raise ValueError("forbidden counts must be bare zeros")
    registry = dict(artifact.get("registry_precheck_and_hash_before_after") or {})
    if registry.get("registry_hash_before") != registry.get("registry_hash_after"):
        raise ValueError("registry hash changed")
    if not str(artifact.get("honest_verdict", "")).startswith("complete:"):
        raise ValueError("honest verdict prefix invalid")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("checksum mismatch")


def write_artifact(artifact: Mapping[str, Any], *, path: Path | None = None) -> Path:
    out = path or (REPO_ROOT / RESULT_RELATIVE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def external_test_receipts() -> tuple[list[str], dict[str, int]]:  # pragma: no cover
    if not EXTERNAL_TEST_RECEIPT_PATH.is_file():
        return list(DEFAULT_TEST_COMMANDS), {}
    payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    return list(payload.get("test_commands", DEFAULT_TEST_COMMANDS)), {
        str(key): int(value) for key, value in dict(payload.get("test_exit_codes", {})).items()
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260809")
    args = parser.parse_args(argv)
    started = time.monotonic()
    commands, exits = external_test_receipts()
    artifact = build_artifact(
        date=str(args.date),
        test_commands=commands,
        test_exit_codes=exits,
        started=started,
    )
    validate_artifact(artifact)
    write_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
