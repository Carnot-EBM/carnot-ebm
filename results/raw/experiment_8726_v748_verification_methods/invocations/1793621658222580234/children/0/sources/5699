"""Exp 4299 registry/gaps/manifest hygiene for .397 verifier outcomes.

Spec refs: REQ-VERIFY-4299, SCENARIO-VERIFY-4299.

This runner is an offline truth-ledger reconciler. It refuses to mutate the
registry if required .397 artifacts are missing, because a hygiene run that
fills ledger rows from expectations instead of landed evidence would make the
verifier backlog less trustworthy.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import time
from typing import Any, Callable

import yaml

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4252 as exp4252
from carnot.reporting import verifier_registry_gaps_hygiene_4266 as exp4266
from carnot.reporting import verifier_registry_gaps_hygiene_4277 as exp4277
from carnot.reporting import verifier_registry_gaps_hygiene_4287 as exp4287


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4299
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4299_ARTIFACT_PATH = "results/experiment_4299_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
EXCLUSION_MANIFEST_PATH = exp4277.EXCLUSION_MANIFEST_PATH
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
ARC1_POOL_PATH = exp4266.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4266.ARC1_PROGRAMS_PATH

EXP4287_PATH = exp4287.EXP4287_ARTIFACT_PATH
EXP4291_PATH = "results/experiment_4291_arcgen_cross_generator_nondegenerate.json"
EXP4292_PATH = "results/experiment_4292_partial_state_diffusion_scorer_build.json"
EXP4293_PATH = "results/experiment_4293_diffusiongemma_energy_guided_run_partial_state.json"
EXP4294_PATH = "results/experiment_4294_verifier_efficiency_harden_strong_judge.json"
EXP4295_PATH = "results/experiment_4295_self_learning_tier2_fixed_retrieval.json"
EXP4296_PATH = "results/experiment_4296_arc_incremental_progress_new_game.json"

REQUIRED_UPSTREAM_PATHS = [
    EXP4287_PATH,
    EXP4291_PATH,
    EXP4292_PATH,
    EXP4294_PATH,
    EXP4295_PATH,
    EXP4296_PATH,
]
OPTIONAL_UPSTREAM_PATHS = [EXP4293_PATH]
REQUIRED_COPY_PATHS = [
    *REQUIRED_UPSTREAM_PATHS,
    ARC1_POOL_PATH,
    ARC1_PROGRAMS_PATH,
]
OPTIONAL_COPY_PATHS = list(OPTIONAL_UPSTREAM_PATHS)

GAP_CROSS_GENERATOR_SELECTION = "GAP-ARC-CROSS-GENERATOR-SELECTION-4299"
GAP_LEAK_FREE_PARTIAL_STATE = "GAP-DIFFUSIONGEMMA-LEAK-FREE-PARTIAL-STATE-4299"
V397_ROLE_ID = "oracle_distinct_v397_registry_gaps_hygiene_4299"
V397_HARDENED_STATE = (
    "cross_generator_holds__partial_state_leak_free__strong_judge_efficiency_recorded__"
    "self_learning_improves__arc_plus_one"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_logged",
    "registry_reconciled",
    "manifest_reconciled",
    "v397_outcomes",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records the registry/gaps reconciled + regression guard result."
    ),
    "regression_guard_passed": (
        "BARE bool: the GAP-4 execution numbers did not regress vs .396 -- "
        "the standing-capability guard."
    ),
    "gaps_logged": (
        "List of new missing-verifier gap entries (failure mode + missing discriminator "
        "+ candidate design + priority) -- the verifier build backlog."
    ),
    "reproducibility_checksum": (
        "Hash of the reconciled registry + gaps + manifest; catches silent drift."
    ),
}

GAP_ENTRY_REQUIRED_FIELDS = (
    "gap_id",
    "failure_mode",
    "missing_discriminator",
    "candidate_design",
    "priority",
)


def _load_manifest(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        loaded.setdefault("retired", [])
        loaded.setdefault("retired_experiments", [])
        loaded.setdefault("retired_extras", [])
        return loaded
    return {"retired": [], "retired_experiments": [], "retired_extras": []}


def _check_resource(
    repo_root: Path,
    resource: str,
    path: str,
    loader: Callable[[Path], Any],
) -> dict[str, Any]:
    full_path = repo_root / path
    try:
        loader(full_path)
    except Exception as exc:
        return {
            "resource": resource,
            "path": path,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"resource": resource, "path": path, "available": True, "error": ""}


def _load_registry_for_check(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("registry must parse as a mapping")
    return loaded


def _load_gaps_for_check(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("gaps ledger must not be empty")
    return text


def _load_manifest_for_check(path: Path) -> dict[str, Any]:
    return _load_manifest(path)


def _load_json_for_check(path: Path) -> dict[str, Any]:
    loaded = base._load_json(path)
    if not isinstance(loaded, dict):
        raise ValueError("artifact must parse as an object")
    return loaded


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4299: ledgers parse and required .397 artifacts exist."""
    checks = [
        _check_resource(repo_root, "verifier_registry", REGISTRY_PATH, _load_registry_for_check),
        _check_resource(repo_root, "verifier_gaps", GAPS_PATH, _load_gaps_for_check),
        _check_resource(
            repo_root,
            "exclusion_manifest",
            EXCLUSION_MANIFEST_PATH,
            _load_manifest_for_check,
        ),
        _check_resource(
            repo_root,
            "gap4_arc1_candidate_pool",
            ARC1_POOL_PATH,
            exp4266.exp4227._load_gzip_json,
        ),
        _check_resource(repo_root, "gap4_arc1_programs", ARC1_PROGRAMS_PATH, base._load_json),
    ]
    checks.extend(
        _check_resource(repo_root, Path(path).stem, path, _load_json_for_check)
        for path in REQUIRED_UPSTREAM_PATHS
    )
    for path in OPTIONAL_UPSTREAM_PATHS:
        full_path = repo_root / path
        if full_path.exists():
            checks.append(_check_resource(repo_root, Path(path).stem, path, _load_json_for_check))
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4299: compare cached replay with .396 recorded GAP-4 numbers."""
    prior = base._load_json(repo_root / EXP4287_PATH)
    recorded = dict(
        prior.get("regression_guard", {}).get("replayed_arc1_rule_exec")
        or prior.get("regression_guard", {}).get("recorded_arc1_rule_exec", {})
    )
    replay = exp4252.replay_gap4_arc1(repo_root)
    replayed = dict(replay.get("arc1_rule_exec", {}))
    passed = (
        replayed.get("n") == recorded.get("n")
        and replayed.get("vote_pass2") == recorded.get("vote_pass2")
        and replayed.get("gated_pass2", 0.0) >= recorded.get("gated_pass2", 0.0)
        and replayed.get("headroom_recovered", 0) >= recorded.get("headroom_recovered", 0)
        and replayed.get("vote_wins_lost", 999999) <= recorded.get("vote_wins_lost", 999999)
    )
    return {
        "regression_guard_passed": bool(passed),
        "prior_artifact_path": EXP4287_PATH,
        "recorded_arc1_rule_exec": recorded,
        "replayed_arc1_rule_exec": replayed,
        "offline_replay": replay,
    }


def load_v397_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4299: read .397 outcomes without manufacturing fields."""
    cross_generator = base._load_json(repo_root / EXP4291_PATH)
    partial_state = base._load_json(repo_root / EXP4292_PATH)
    efficiency = base._load_json(repo_root / EXP4294_PATH)
    self_learning = base._load_json(repo_root / EXP4295_PATH)
    arc = base._load_json(repo_root / EXP4296_PATH)
    in_generation_path = repo_root / EXP4293_PATH
    in_generation = base._load_json(in_generation_path) if in_generation_path.exists() else {}

    return {
        "cross_generator": {
            "artifact_path": EXP4291_PATH,
            "honest_verdict": str(cross_generator.get("honest_verdict", "")),
            "cross_generator_holds": cross_generator.get("cross_generator_holds") is True,
            "cross_generator_delta": cross_generator.get("cross_generator_delta"),
            "cross_generator_ci95": cross_generator.get("cross_generator_ci95"),
            "held_out_generator_n": cross_generator.get("held_out_generator_n"),
            "held_out_task_n": cross_generator.get("held_out_task_n"),
            "vote_at_1": cross_generator.get("vote_at_1"),
            "oracle_at_k": cross_generator.get("oracle_at_k"),
            "matched_control_delta": cross_generator.get("matched_control_delta"),
            "per_substrate_delta": dict(cross_generator.get("per_substrate_delta", {})),
            "non_degenerate_guards_pass": cross_generator.get("non_degenerate_guards_pass") is True,
            "verifier_is_oracle": cross_generator.get("verifier_is_oracle") is True,
        },
        "partial_state": {
            "artifact_path": EXP4292_PATH,
            "honest_verdict": str(partial_state.get("honest_verdict", "")),
            "partial_state_scorer_built": partial_state.get("partial_state_scorer_built")
            is True,
            "partial_state_leak_free": partial_state.get("partial_state_leak_free") is True,
            "partial_state_auroc": partial_state.get("partial_state_auroc"),
            "leak_ablation_auroc": partial_state.get("leak_ablation_auroc"),
            "heldout_task_n": partial_state.get("heldout_task_n"),
            "scorer_loadable": partial_state.get("scorer_loadable") is True,
            "verifier_is_oracle": partial_state.get("verifier_is_oracle") is True,
        },
        "in_generation": {
            "artifact_path": EXP4293_PATH,
            "ran": bool(in_generation),
            "honest_verdict": str(in_generation.get("honest_verdict", "")),
            "diffusiongemma_guidance_moat": in_generation.get("diffusiongemma_guidance_moat")
            is True,
            "carnot_minus_rfg_delta": in_generation.get("carnot_minus_rfg_delta"),
            "carnot_minus_unguided_delta": in_generation.get("carnot_minus_unguided_delta"),
            "guidance_moat_ci95": in_generation.get("guidance_moat_ci95"),
            "guidance_changes_selection": in_generation.get("guidance_changes_selection") is True,
            "condition_accuracy": dict(in_generation.get("condition_accuracy", {})),
            "flagged_adversarial": in_generation.get("flagged_adversarial") is True,
            "corrigendum_pending": list(in_generation.get("corrigendum_pending", [])),
            "verifier_is_oracle": in_generation.get("verifier_is_oracle") is True,
        },
        "efficiency": {
            "artifact_path": EXP4294_PATH,
            "honest_verdict": str(efficiency.get("honest_verdict", "")),
            "efficiency_pareto_holds": efficiency.get("efficiency_pareto_holds") is True,
            "accuracy_energy_verifier": efficiency.get("accuracy_energy_verifier"),
            "accuracy_best_judge": efficiency.get("accuracy_best_judge"),
            "accuracy_delta_ci95": efficiency.get("accuracy_delta_ci95"),
            "cost_ratio": efficiency.get("cost_ratio"),
            "judge_metrics": dict(efficiency.get("judge_metrics", {})),
            "verifier_is_oracle": efficiency.get("verifier_is_oracle") is True,
        },
        "self_learning": {
            "artifact_path": EXP4295_PATH,
            "honest_verdict": str(self_learning.get("honest_verdict", "")),
            "online_adaptation_helps": self_learning.get("online_adaptation_helps") is True,
            "static_cross_family_delta": self_learning.get("static_cross_family_delta"),
            "online_cross_family_delta": self_learning.get("online_cross_family_delta"),
            "tier2_retrieval_cross_family_delta": self_learning.get(
                "tier2_retrieval_cross_family_delta"
            ),
            "tier2_memory_cross_family_delta": self_learning.get(
                "tier2_memory_cross_family_delta"
            ),
            "adaptive_minus_static_ci95": dict(
                self_learning.get("adaptive_minus_static_ci95", {})
            ),
            "held_out_family_n": self_learning.get("held_out_family_n"),
            "held_out_task_n": self_learning.get("held_out_task_n"),
            "tier2_not_noop": self_learning.get("tier2_not_noop") is True,
            "verifier_is_oracle": self_learning.get("verifier_is_oracle") is True,
        },
        "arc_progress": {
            "artifact_path": EXP4296_PATH,
            "honest_verdict": str(arc.get("honest_verdict", "")),
            "acceptance_gate_passed": arc.get("acceptance_gate_passed") is True,
            "game_advanced": str(arc.get("game_advanced", "")),
            "target_game": str(arc.get("target_game", "")),
            "target_level": arc.get("target_level"),
            "prior_level": arc.get("prior_level"),
            "levels_completed": arc.get("levels_completed"),
            "new_levels_solved_this_task": arc.get("new_levels_solved_this_task"),
            "prior_total_levels_solved": arc.get("prior_total_levels_solved"),
            "total_levels": arc.get("total_levels"),
            "total_levels_solved": arc.get("total_levels_solved"),
            "real_env_confirmed": arc.get("real_env_confirmed") is True,
            "verifier_validated": arc.get("verifier_validated") is True,
            "selection_mode": str(arc.get("selection_mode", "")),
        },
    }


def build_gap_entries(outcomes: dict[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4299: log Missing-Verifier gaps exposed by failed .397 axes."""
    gaps: list[dict[str, Any]] = []
    cross_generator = outcomes["cross_generator"]
    if cross_generator["cross_generator_holds"] is not True:
        gaps.append(
            {
                "gap_id": GAP_CROSS_GENERATOR_SELECTION,
                "status": "open",
                "evidence": (
                    f"{EXP4291_PATH}; honest_verdict={cross_generator['honest_verdict']}; "
                    f"cross_generator_holds={cross_generator['cross_generator_holds']}; "
                    f"cross_generator_delta={cross_generator['cross_generator_delta']}"
                ),
                "failure_mode": (
                    "The learned ARC selector did not preserve its held-out-family win "
                    "on construction-disjoint ARC-GEN generators."
                ),
                "missing_discriminator": (
                    "generator-family-invariant ARC candidate correctness signal that "
                    "does not key on one procedural generator's artifacts."
                ),
                "candidate_design": (
                    "Train and calibrate the set encoder with generator-disjoint folds, "
                    "explicit substrate labels, and an adversarial generator-family "
                    "invariance penalty before accepting cross-generator moat claims."
                ),
                "priority": "high",
            }
        )

    partial_state = outcomes["partial_state"]
    if (
        partial_state["partial_state_scorer_built"] is not True
        or partial_state["partial_state_leak_free"] is not True
    ):
        gaps.append(
            {
                "gap_id": GAP_LEAK_FREE_PARTIAL_STATE,
                "status": "open",
                "evidence": (
                    f"{EXP4292_PATH}; honest_verdict={partial_state['honest_verdict']}; "
                    f"partial_state_scorer_built={partial_state['partial_state_scorer_built']}; "
                    f"partial_state_leak_free={partial_state['partial_state_leak_free']}"
                ),
                "failure_mode": (
                    "The partial-state diffusion scorer either was not built or failed "
                    "the answer-masked leak audit, so in-generation guidance would be "
                    "circular rather than verifier-distinct."
                ),
                "missing_discriminator": (
                    "leak-free partial-state diffusion energy that survives masking "
                    "answer-bearing canvas cells."
                ),
                "candidate_design": (
                    "Rebuild the partial-state value head on task-disjoint canvases with "
                    "answer-span masking as a training-time invariant and require masked "
                    "AUROC above the pre-registered floor before using it for guidance."
                ),
                "priority": "high",
            }
        )
    return gaps


def ensure_ledgers_record_v397(
    registry: dict[str, Any],
    gaps_text: str,
    exclusion_manifest: dict[str, Any],
    regression_guard: dict[str, Any],
    outcomes: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return registry, gap text, and manifest with the .397 truth represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcomes, gaps_logged)
    _ensure_v397_role(updated_registry, outcomes, gaps_logged)

    updated_gaps = gaps_text
    for gap in gaps_logged:
        updated_gaps = base._replace_marked_block(
            updated_gaps,
            f"exp4299-{gap['gap_id'].lower()}",
            _gap_entry_block(gap),
        )

    updated_manifest = deepcopy(exclusion_manifest)
    gap_ids = [gap["gap_id"] for gap in gaps_logged]
    return (
        updated_registry,
        updated_gaps,
        updated_manifest,
        {
            "registry_reconciled": registry_contains_v397(updated_registry),
            "manifest_reconciled": isinstance(updated_manifest, dict),
            "gaps_logged_ids": [gap_id for gap_id in gap_ids if gap_id in updated_gaps],
        },
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    regression_guard: dict[str, Any],
    outcomes: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = regression_guard.get("replayed_arc1_rule_exec", {})
    cross_generator = outcomes["cross_generator"]
    partial_state = outcomes["partial_state"]
    in_generation = outcomes["in_generation"]
    efficiency = outcomes["efficiency"]
    self_learning = outcomes["self_learning"]
    arc = outcomes["arc_progress"]
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4299": EXP4299_ARTIFACT_PATH,
            "exp4299_regression_guard_passed": bool(
                regression_guard.get("regression_guard_passed")
            ),
            "exp4299_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4299_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4299_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4299_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4299_v397_hardened_state": V397_HARDENED_STATE,
            "exp4299_cross_generator_artifact": EXP4291_PATH,
            "exp4299_cross_generator_holds": cross_generator["cross_generator_holds"],
            "exp4299_cross_generator_delta": cross_generator["cross_generator_delta"],
            "exp4299_cross_generator_ci95": cross_generator["cross_generator_ci95"],
            "exp4299_cross_generator_held_out_task_n": cross_generator["held_out_task_n"],
            "exp4299_partial_state_artifact": EXP4292_PATH,
            "exp4299_partial_state_scorer_built": partial_state[
                "partial_state_scorer_built"
            ],
            "exp4299_partial_state_leak_free": partial_state["partial_state_leak_free"],
            "exp4299_partial_state_auroc": partial_state["partial_state_auroc"],
            "exp4299_leak_ablation_auroc": partial_state["leak_ablation_auroc"],
            "exp4299_in_generation_artifact": EXP4293_PATH if in_generation["ran"] else None,
            "exp4299_diffusiongemma_guidance_moat": in_generation[
                "diffusiongemma_guidance_moat"
            ],
            "exp4299_diffusiongemma_flagged_adversarial": in_generation[
                "flagged_adversarial"
            ],
            "exp4299_diffusiongemma_carnot_minus_rfg_delta": in_generation[
                "carnot_minus_rfg_delta"
            ],
            "exp4299_efficiency_artifact": EXP4294_PATH,
            "exp4299_efficiency_pareto_holds": efficiency["efficiency_pareto_holds"],
            "exp4299_efficiency_cost_ratio": efficiency["cost_ratio"],
            "exp4299_efficiency_accuracy_energy_verifier": efficiency[
                "accuracy_energy_verifier"
            ],
            "exp4299_efficiency_accuracy_best_judge": efficiency["accuracy_best_judge"],
            "exp4299_efficiency_accuracy_delta_ci95": efficiency["accuracy_delta_ci95"],
            "exp4299_self_learning_artifact": EXP4295_PATH,
            "exp4299_online_adaptation_helps": self_learning["online_adaptation_helps"],
            "exp4299_static_cross_family_delta": self_learning[
                "static_cross_family_delta"
            ],
            "exp4299_online_cross_family_delta": self_learning[
                "online_cross_family_delta"
            ],
            "exp4299_tier2_retrieval_cross_family_delta": self_learning[
                "tier2_retrieval_cross_family_delta"
            ],
            "exp4299_arc_progress_artifact": EXP4296_PATH,
            "exp4299_arc_total_levels": arc["total_levels"],
            "exp4299_arc_total_levels_solved": arc["total_levels_solved"],
            "exp4299_arc_new_levels_solved": arc["new_levels_solved_this_task"],
            "exp4299_arc_game_advanced": arc["game_advanced"],
            "exp4299_gaps_logged": [gap["gap_id"] for gap in gaps_logged],
        }
    )


def _ensure_v397_role(
    registry: dict[str, Any],
    outcomes: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    cross_generator = outcomes["cross_generator"]
    partial_state = outcomes["partial_state"]
    in_generation = outcomes["in_generation"]
    efficiency = outcomes["efficiency"]
    self_learning = outcomes["self_learning"]
    arc = outcomes["arc_progress"]
    role = {
        "role_id": V397_ROLE_ID,
        "experiment": EXP4299_ARTIFACT_PATH,
        "role": "registry_gap_manifest_hygiene_v397",
        "status": "v397_outcomes_recorded",
        "v397_hardened_state": V397_HARDENED_STATE,
        "cross_generator_artifact": EXP4291_PATH,
        "cross_generator_holds": cross_generator["cross_generator_holds"],
        "cross_generator_delta": cross_generator["cross_generator_delta"],
        "partial_state_artifact": EXP4292_PATH,
        "partial_state_scorer_built": partial_state["partial_state_scorer_built"],
        "partial_state_leak_free": partial_state["partial_state_leak_free"],
        "in_generation_artifact": EXP4293_PATH if in_generation["ran"] else None,
        "diffusiongemma_guidance_moat": in_generation["diffusiongemma_guidance_moat"],
        "diffusiongemma_flagged_adversarial": in_generation["flagged_adversarial"],
        "efficiency_artifact": EXP4294_PATH,
        "efficiency_pareto_holds": efficiency["efficiency_pareto_holds"],
        "efficiency_cost_ratio": efficiency["cost_ratio"],
        "self_learning_artifact": EXP4295_PATH,
        "online_adaptation_helps": self_learning["online_adaptation_helps"],
        "online_cross_family_delta": self_learning["online_cross_family_delta"],
        "arc_progress_artifact": EXP4296_PATH,
        "arc_total_levels": arc["total_levels"],
        "arc_new_levels_solved": arc["new_levels_solved_this_task"],
        "arc_game_advanced": arc["game_advanced"],
        "gap_ids_logged": [gap["gap_id"] for gap in gaps_logged],
        "eval_exp_4299": EXP4299_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V397_ROLE_ID] + [
        role
    ]


def _gap_entry_block(gap: dict[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4299 .397 missing-verifier gap\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap['failure_mode']}\n"
        f"- missing discriminator: {gap['missing_discriminator']}\n"
        f"- candidate design: {gap['candidate_design']}\n"
        f"- priority: {gap['priority']}\n"
    )


def registry_contains_v397(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4299") == EXP4299_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4299_v397_hardened_state") == V397_HARDENED_STATE
        and any(role.get("role_id") == V397_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def ledger_checksum(registry_path: Path, gaps_path: Path, manifest_path: Path) -> str:
    """REQ-VERIFY-4299: hash reconciled ledgers to catch silent drift."""
    digest = hashlib.sha256()
    for label, path in (
        ("registry", registry_path),
        ("gaps", gaps_path),
        ("manifest", manifest_path),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def model_specs_for_reconciliation() -> dict[str, Any]:
    return {
        "method": "cached_v397_ledger_reconciliation",
        "gap4_candidate_set": ARC1_POOL_PATH,
        "gap4_program_outputs": ARC1_PROGRAMS_PATH,
        "required_upstream_artifacts": list(REQUIRED_UPSTREAM_PATHS),
        "optional_upstream_artifacts": list(OPTIONAL_UPSTREAM_PATHS),
        "codex_calls": 0,
        "live_model_inference": False,
        "gguf_inference": False,
        "gpu_inference": False,
        "trm_training_touched": False,
        "stable_checkpoint_write": False,
    }


def build_artifact(
    *,
    regression_guard: dict[str, Any],
    v397_outcomes: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
    registry_reconciled: bool,
    manifest_reconciled: bool,
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4299 terminal JSON payload."""
    guard_ok = bool(regression_guard.get("regression_guard_passed"))
    complete = guard_ok and registry_reconciled and manifest_reconciled
    prefix = "complete:" if complete else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4299_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4299_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_manifest_reconciled_to_v397_truth_"
            f"regression_guard_passed_{guard_ok}_gaps_logged_{len(gaps_logged)}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_logged": list(gaps_logged),
        "registry_reconciled": bool(registry_reconciled),
        "manifest_reconciled": bool(manifest_reconciled),
        "v397_outcomes": v397_outcomes,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum,
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4299", "SCENARIO-VERIFY-4299"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "regression_guard": regression_guard,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "exclusion_manifest_path": EXCLUSION_MANIFEST_PATH,
        "cited_upstream_artifacts": list(REQUIRED_UPSTREAM_PATHS + OPTIONAL_UPSTREAM_PATHS),
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4299_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4299_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": "blocked_v397_artifacts_missing",
        "regression_guard_passed": False,
        "gaps_logged": [],
        "registry_reconciled": False,
        "manifest_reconciled": False,
        "v397_outcomes": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "blocked:v397_artifacts_missing",
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4299", "SCENARIO-VERIFY-4299"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions": preflight,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4299 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if type(artifact["regression_guard_passed"]) is not bool:
        raise ValueError("regression_guard_passed must be a BARE bool")
    if type(artifact["registry_reconciled"]) is not bool:
        raise ValueError("registry_reconciled must be a bare bool")
    if type(artifact["manifest_reconciled"]) is not bool:
        raise ValueError("manifest_reconciled must be a bare bool")
    if not isinstance(artifact["gaps_logged"], list):
        raise ValueError("gaps_logged must be a list")
    for gap in artifact["gaps_logged"]:
        if not isinstance(gap, dict) or not all(field in gap for field in GAP_ENTRY_REQUIRED_FIELDS):
            raise ValueError("gaps_logged gap entry is missing required fields")
    if isinstance(artifact["random_seed"], bool) or not isinstance(artifact["random_seed"], int):
        raise ValueError("random_seed must be a bare int")
    if (
        not isinstance(artifact["reproducibility_checksum"], str)
        or not artifact["reproducibility_checksum"]
    ):
        raise ValueError("reproducibility_checksum must be a non-empty string")
    if not isinstance(artifact["model_specs"], dict) or not artifact["model_specs"]:
        raise ValueError("model_specs must be a non-empty object")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4299 principles")
    if artifact["spec_refs"] != ["REQ-VERIFY-4299", "SCENARIO-VERIFY-4299"]:
        raise ValueError("spec_refs must cite REQ-VERIFY-4299 and SCENARIO-VERIFY-4299")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4299 and write the terminal artifact plus reconciled ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4299_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    manifest_path = repo_root / EXCLUSION_MANIFEST_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    manifest = _load_manifest(manifest_path)
    regression_guard = run_gap4_regression_guard(repo_root)
    outcomes = load_v397_outcomes(repo_root)
    gaps_logged = build_gap_entries(outcomes)
    registry, gaps_text, _manifest, ledger_summary = ensure_ledgers_record_v397(
        registry,
        gaps_text,
        manifest,
        regression_guard,
        outcomes,
        gaps_logged,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    checksum = ledger_checksum(registry_path, gaps_path, manifest_path)
    artifact = build_artifact(
        regression_guard=regression_guard,
        v397_outcomes=outcomes,
        gaps_logged=gaps_logged,
        registry_reconciled=bool(ledger_summary["registry_reconciled"]),
        manifest_reconciled=bool(ledger_summary["manifest_reconciled"]),
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through results entrypoint.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4299_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
