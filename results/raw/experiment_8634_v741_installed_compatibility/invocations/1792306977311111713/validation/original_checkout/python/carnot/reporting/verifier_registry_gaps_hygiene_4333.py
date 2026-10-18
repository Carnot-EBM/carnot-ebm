"""Exp 4333 registry/gaps/manifest hygiene for .400 verifier outcomes.

Spec refs: REQ-VERIFY-4333, SCENARIO-VERIFY-4333.

This runner is the .400 continuation of Exp 4321. It treats missing .400
artifacts as axis-local availability gaps through the robust aggregate helper,
then reconciles the verifier truth ledgers to the decision-grade evidence that
is present.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Mapping

import yaml

from carnot.reporting import capstone_aggregate_available as aggregate
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4252 as exp4252
from carnot.reporting import verifier_registry_gaps_hygiene_4310 as exp4310
from carnot.reporting import verifier_registry_gaps_hygiene_4321 as exp4321


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4333
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4333_ARTIFACT_PATH = "results/experiment_4333_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
EXCLUSION_MANIFEST_PATH = exp4321.EXCLUSION_MANIFEST_PATH
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
ARC1_POOL_PATH = exp4321.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4321.ARC1_PROGRAMS_PATH

EXP4321_PATH = exp4321.EXP4321_ARTIFACT_PATH
EXP4308_PATH = exp4321.EXP4308_PATH
EXP4314_PATH = exp4321.EXP4314_PATH
EXP4325_PATH = "results/experiment_4325_in_generation_moat_replicate_second_corpus.json"
EXP4326_PATH = "results/experiment_4326_adaptive_guided_generation_scaleup.json"
EXP4327_PATH = "results/experiment_4327_e3_executable_world_model_ar25.json"
EXP4328_PATH = "results/experiment_4328_e3_executable_world_model_ka59.json"
EXP4329_PATH = "results/experiment_4329_e3_executable_world_model_tr87_ft09.json"
EXP4330_PATH = "results/experiment_4330_arc_adapter_free_discovery_sweep_shallow_tail.json"
EXP4331_PATH = "results/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.json"

OUTCOME_ARTIFACT_PATHS = [
    EXP4325_PATH,
    EXP4326_PATH,
    EXP4327_PATH,
    EXP4328_PATH,
    EXP4329_PATH,
    EXP4330_PATH,
    EXP4331_PATH,
]
REQUIRED_COPY_PATHS = [
    EXP4321_PATH,
    EXP4308_PATH,
    *OUTCOME_ARTIFACT_PATHS,
    ARC1_POOL_PATH,
    ARC1_PROGRAMS_PATH,
]

GAP_DIFFUSIONGEMMA_SECOND_CORPUS_SCORER = (
    "GAP-DIFFUSIONGEMMA-SECOND-CORPUS-LEAK-FREE-SCORER-4325"
)
GAP_ARC_GRID_GENERATION_SCORER = "GAP-ARC-GRID-GENERATION-SCORER-4326"
GAP_E3_WORLD_MODEL_RULE_AR25 = "GAP-E3-WORLD-MODEL-RULE-AR25-4327"
GAP_E3_WORLD_MODEL_RULE_KA59 = "GAP-E3-WORLD-MODEL-RULE-KA59-4328"
GAP_E3_WORLD_MODEL_RULE_TR87 = "GAP-E3-WORLD-MODEL-RULE-TR87-4329"
GAP_E3_WORLD_MODEL_RULE_FT09 = "GAP-E3-WORLD-MODEL-RULE-FT09-4329"
GAP_GAME_INVARIANT_ARC_VALUE_4331 = "GAP-4331"
CROSS_DOMAIN_RETIREMENT_ID = "cross_domain_selection_retired_exp4314_v399"

V400_ROLE_ID = "oracle_distinct_v400_registry_gaps_hygiene_4333"
V400_STATE = (
    "second_corpus_scorer_leaky__adaptive_guidance_null__"
    "e3_world_models_partial__arc_13_shallow_no_advance__"
    "learned_encoder_transfer_null"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_logged",
    "registry_reconciled",
    "manifest_reconciled",
    "v400_outcomes",
    "availability_report",
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
        "Terminal-prefixed. Records the registry/gaps reconciled + regression guard "
        "result (using the robust aggregator, NOT a hard-block)."
    ),
    "regression_guard_passed": (
        "BARE bool: the GAP-4 execution numbers did not regress vs .399 -- "
        "the standing-capability guard."
    ),
    "gaps_logged": (
        "List of new/updated missing-verifier gap entries (failure mode + missing "
        "discriminator + candidate design + priority) -- the verifier build backlog."
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

ARTIFACT_KEYS = {
    "4325_in_generation_replication": EXP4325_PATH,
    "4326_adaptive_scaleup": EXP4326_PATH,
    "4327_e3_ar25": EXP4327_PATH,
    "4328_e3_ka59": EXP4328_PATH,
    "4329_e3_tr87_ft09": EXP4329_PATH,
    "4330_shallow_tail_sweep": EXP4330_PATH,
    "4331_learned_encoder_transfer": EXP4331_PATH,
}
ARTIFACT_EXPERIMENT_IDS = {
    "4325_in_generation_replication": 4325,
    "4326_adaptive_scaleup": 4326,
    "4327_e3_ar25": 4327,
    "4328_e3_ka59": 4328,
    "4329_e3_tr87_ft09": 4329,
    "4330_shallow_tail_sweep": 4330,
    "4331_learned_encoder_transfer": 4331,
}

check_preconditions = exp4321.check_preconditions
ledger_checksum = exp4321.ledger_checksum
robust_aggregator_ok = exp4321.robust_aggregator_ok
_load_optional_json = exp4321._load_optional_json
_load_manifest = exp4321._load_manifest


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4333: compare cached replay with .399 recorded GAP-4 numbers."""
    recorded: dict[str, Any] = {}
    prior_path = EXP4321_PATH
    try:
        prior_artifact = base._load_json(repo_root / EXP4321_PATH)
        recorded = exp4310._recorded_arc1_rule_exec(prior_artifact)
    except Exception:  # pragma: no cover - defensive fallback for damaged copies.
        fallback = exp4321.run_gap4_regression_guard(repo_root)
        recorded = dict(fallback.get("replayed_arc1_rule_exec", {}))
        prior_path = fallback.get("prior_artifact_path", EXP4321_PATH)

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
        "prior_artifact_path": prior_path,
        "recorded_arc1_rule_exec": recorded,
        "replayed_arc1_rule_exec": replayed,
        "offline_replay": replay,
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="in_generation_replication",
            required_keys=("4325_in_generation_replication",),
            verdict_fn=lambda present: (
                present["4325_in_generation_replication"].get(
                    "in_generation_moat_replicates"
                )
                is True
                and present["4325_in_generation_replication"].get(
                    "scorer_leak_recheck_passed"
                )
                is True
                and present["4325_in_generation_replication"].get(
                    "controls_differentiated"
                )
                is True
            ),
        ),
        aggregate.AxisSpec(
            name="adaptive_scaleup",
            required_keys=("4326_adaptive_scaleup",),
            verdict_fn=lambda present: (
                present["4326_adaptive_scaleup"].get("adaptive_guidance_beats_control")
                is True
                and present["4326_adaptive_scaleup"].get("scorer_leak_recheck_passed")
                is True
                and present["4326_adaptive_scaleup"].get("controls_differentiated")
                is True
            ),
        ),
        aggregate.AxisSpec(
            name="e3_deep_tail",
            required_keys=("4327_e3_ar25", "4328_e3_ka59", "4329_e3_tr87_ft09"),
            verdict_fn=lambda present: (
                present["4327_e3_ar25"].get("offline_reproduced") is True
                or present["4328_e3_ka59"].get("offline_reproduced") is True
                or int(present["4329_e3_tr87_ft09"].get("reproduced_levels_total", 0))
                > 0
            ),
        ),
        aggregate.AxisSpec(
            name="shallow_tail_sweep",
            required_keys=("4330_shallow_tail_sweep",),
            verdict_fn=lambda present: (
                present["4330_shallow_tail_sweep"].get("offline_reproduced") is True
                and int(present["4330_shallow_tail_sweep"].get("reproducible_total_levels", 0))
                > int(present["4330_shallow_tail_sweep"].get("prior_reproducible_total_levels", 13))
            ),
        ),
        aggregate.AxisSpec(
            name="learned_encoder_transfer",
            required_keys=("4331_learned_encoder_transfer",),
            verdict_fn=lambda present: present["4331_learned_encoder_transfer"].get(
                "learned_encoder_transfer_helps"
            )
            is True,
        ),
    ]


def load_v400_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4333: read available .400 outcomes through robust availability."""
    raw_artifacts: dict[str, Any] = {}
    artifact_errors: dict[str, str] = {}
    for key, rel_path in ARTIFACT_KEYS.items():
        payload, error = _load_optional_json(repo_root, rel_path)
        raw_artifacts[key] = payload
        if error:
            artifact_errors[key] = error

    availability_report = aggregate.aggregate_available_report_gaps(
        raw_artifacts,
        _axis_specs(),
        artifact_experiment_ids=ARTIFACT_EXPERIMENT_IDS,
    )
    robust_payload, robust_error = _load_optional_json(repo_root, EXP4308_PATH)

    return {
        "v400_outcomes": {
            "in_generation_replication": _read_in_generation_replication(
                raw_artifacts.get("4325_in_generation_replication")
            ),
            "adaptive_scaleup": _read_adaptive_scaleup(
                raw_artifacts.get("4326_adaptive_scaleup")
            ),
            "e3_deep_tail": _read_e3_deep_tail(raw_artifacts),
            "shallow_tail_sweep": _read_shallow_tail_sweep(
                raw_artifacts.get("4330_shallow_tail_sweep")
            ),
            "learned_encoder_transfer": _read_learned_encoder_transfer(
                raw_artifacts.get("4331_learned_encoder_transfer")
            ),
            "robust_aggregator": exp4310._read_robust_aggregator_evidence(
                robust_payload,
                robust_error,
            ),
        },
        "availability_report": availability_report,
        "artifact_errors": artifact_errors,
    }


def _read_in_generation_replication(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4325_PATH, "available": False}
    return {
        "artifact_path": EXP4325_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "in_generation_moat_replicates": payload.get("in_generation_moat_replicates")
        is True,
        "controls_differentiated": payload.get("controls_differentiated") is True,
        "scorer_leak_recheck_passed": payload.get("scorer_leak_recheck_passed") is True,
        "benchmark_n": payload.get("benchmark_n"),
        "benchmark_n_per_seed": payload.get("benchmark_n_per_seed"),
        "carnot_minus_best_control_delta": payload.get("carnot_minus_best_control_delta"),
        "carnot_minus_self_reward_smc_delta": payload.get(
            "carnot_minus_self_reward_smc_delta"
        ),
        "carnot_minus_unguided_delta": payload.get("carnot_minus_unguided_delta"),
        "replication_ci95": payload.get("replication_ci95"),
        "independent_leak_recheck": dict(payload.get("independent_leak_recheck", {})),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_adaptive_scaleup(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4326_PATH, "available": False}
    return {
        "artifact_path": EXP4326_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "adaptive_guidance_beats_control": payload.get("adaptive_guidance_beats_control")
        is True,
        "domain_used": str(payload.get("domain_used", "")),
        "controls_differentiated": payload.get("controls_differentiated") is True,
        "scorer_leak_recheck_passed": payload.get("scorer_leak_recheck_passed") is True,
        "benchmark_n": payload.get("benchmark_n"),
        "best_control": str(payload.get("best_control", "")),
        "carnot_minus_best_control_delta": payload.get("carnot_minus_best_control_delta"),
        "adaptive_ci95": payload.get("adaptive_ci95"),
        "condition_accuracy": dict(payload.get("condition_accuracy", {})),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_e3_deep_tail(raw_artifacts: Mapping[str, Any]) -> dict[str, Any]:
    games = {
        "ar25": _read_e3_single(raw_artifacts.get("4327_e3_ar25"), EXP4327_PATH, "ar25"),
        "ka59": _read_e3_single(raw_artifacts.get("4328_e3_ka59"), EXP4328_PATH, "ka59"),
    }
    games.update(_read_e3_multi(raw_artifacts.get("4329_e3_tr87_ft09")))
    total = sum(
        int(game.get("reproduced_levels") or 0)
        for game in games.values()
        if isinstance(game, Mapping)
    )
    return {
        "artifact_paths": [EXP4327_PATH, EXP4328_PATH, EXP4329_PATH],
        "available": any(game.get("available") is True for game in games.values()),
        "offline_reproduced_any": any(
            game.get("offline_reproduced") is True for game in games.values()
        ),
        "reproduced_levels_total": total,
        "games": games,
    }


def _read_e3_single(payload: Any, artifact_path: str, game: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": artifact_path, "game": game, "available": False}
    return {
        "artifact_path": artifact_path,
        "game": str(payload.get("game", game)),
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "offline_reproduced": payload.get("offline_reproduced") is True,
        "plan_executed": payload.get("plan_executed") is True,
        "reproduced_levels": payload.get("reproduced_levels"),
        "residual_mismatch_class": str(payload.get("residual_mismatch_class", "")),
        "verifier_best_accuracy": payload.get("verifier_best_accuracy"),
        "verifier_accuracy_per_round": list(payload.get("verifier_accuracy_per_round", [])),
        "world_model_path": str(payload.get("world_model_path", "")),
        "world_model_sha256": str(payload.get("world_model_sha256", "")),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_e3_multi(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return {
            "tr87": {"artifact_path": EXP4329_PATH, "game": "tr87", "available": False},
            "ft09": {"artifact_path": EXP4329_PATH, "game": "ft09", "available": False},
        }
    scorecard = payload.get("per_game_scorecard")
    if not isinstance(scorecard, Mapping):
        scorecard = {}
    games: dict[str, dict[str, Any]] = {}
    for game in ("tr87", "ft09"):
        row = scorecard.get(game)
        if not isinstance(row, Mapping):
            games[game] = {"artifact_path": EXP4329_PATH, "game": game, "available": False}
            continue
        games[game] = {
            "artifact_path": EXP4329_PATH,
            "game": game,
            "available": True,
            "honest_verdict": str(row.get("status", payload.get("honest_verdict", ""))),
            "offline_reproduced": row.get("offline_reproduced") is True,
            "plan_executed": row.get("plan_executed") is True,
            "reproduced_levels": row.get("reproduced_levels"),
            "residual_mismatch_class": str(row.get("residual_mismatch_class", "")),
            "verifier_best_accuracy": row.get("best_verifier_accuracy"),
            "verifier_accuracy_per_round": list(row.get("verifier_accuracy_per_round", [])),
            "world_model_path": str(row.get("world_model_path", "")),
            "world_model_sha256": str(row.get("world_model_sha256", "")),
            "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
        }
    return games


def _read_shallow_tail_sweep(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4330_PATH, "available": False}
    return {
        "artifact_path": EXP4330_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "offline_reproduced": payload.get("offline_reproduced") is True,
        "reproducible_total_levels": payload.get("reproducible_total_levels"),
        "prior_reproducible_total_levels": payload.get("prior_reproducible_total_levels"),
        "games_advanced": list(payload.get("games_advanced", [])),
        "swept_games": list(payload.get("swept_games", [])),
        "excluded_games": list(payload.get("excluded_games", [])),
        "tn36_schema_finding": dict(payload.get("tn36_schema_finding", {})),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_learned_encoder_transfer(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {
            "artifact_path": EXP4331_PATH,
            "available": False,
            "missing_verifier_gaps": [],
        }
    return {
        "artifact_path": EXP4331_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "acceptance_gate_passed": payload.get("acceptance_gate_passed") is True,
        "learned_encoder_transfer_helps": payload.get("learned_encoder_transfer_helps")
        is True,
        "baseline_solves_held_out": payload.get("baseline_solves_held_out") is True,
        "cross_game_state_reduction": payload.get("cross_game_state_reduction"),
        "cross_game_state_reduction_ci95": payload.get("cross_game_state_reduction_ci95"),
        "n_held_out_levels": payload.get("n_held_out_levels"),
        "missing_verifier_gaps": list(payload.get("missing_verifier_gaps", [])),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def build_gap_entries(outcome_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4333: log Missing-Verifier gaps exposed by .400 axes."""
    outcomes = outcome_bundle["v400_outcomes"]
    gaps: list[dict[str, Any]] = []

    in_generation = outcomes["in_generation_replication"]
    if (
        in_generation.get("available") is True
        and (
            in_generation.get("in_generation_moat_replicates") is not True
            or in_generation.get("scorer_leak_recheck_passed") is not True
        )
    ):
        gaps.append(_diffusiongemma_second_corpus_gap(in_generation))

    adaptive = outcomes["adaptive_scaleup"]
    if (
        adaptive.get("available") is True
        and adaptive.get("adaptive_guidance_beats_control") is not True
    ):
        gaps.append(_arc_grid_generation_gap(adaptive))

    e3_games = outcomes["e3_deep_tail"]["games"]
    for game in ("ar25", "ka59", "tr87", "ft09"):
        row = e3_games.get(game, {})
        if (
            isinstance(row, Mapping)
            and row.get("available") is True
            and row.get("offline_reproduced") is not True
            and str(row.get("residual_mismatch_class", "")).startswith(
                "missing_world_model_rule_gap"
            )
        ):
            gaps.append(_e3_world_model_gap(game, row))

    learned = outcomes["learned_encoder_transfer"]
    if (
        learned.get("available") is True
        and learned.get("learned_encoder_transfer_helps") is not True
    ):
        gaps.extend(
            _upstream_or_fallback(
                learned.get("missing_verifier_gaps", []),
                EXP4331_PATH,
                _learned_encoder_gap(learned),
            )
        )

    return _dedupe_gap_entries(gaps)


def _upstream_or_fallback(
    upstream_gaps: Any,
    evidence_path: str,
    fallback: dict[str, Any],
) -> list[dict[str, Any]]:
    valid = [
        _normalize_upstream_gap(upstream, evidence_path)
        for upstream in upstream_gaps
        if isinstance(upstream, Mapping)
    ]
    return valid or [fallback]


def _normalize_upstream_gap(upstream: Mapping[str, Any], evidence_path: str) -> dict[str, Any]:
    priority = str(upstream.get("priority", "medium"))
    return {
        "gap_id": str(upstream.get("gap_id", GAP_GAME_INVARIANT_ARC_VALUE_4331)),
        "status": str(upstream.get("status", "open_small_encoder_insufficient")),
        "evidence": f"{evidence_path}; upstream_missing_verifier_gap=true",
        "failure_mode": str(upstream.get("failure_mode", "")),
        "missing_discriminator": str(upstream.get("missing_discriminator", "")),
        "candidate_design": str(upstream.get("candidate_design", "")),
        "priority": priority,
    }


def _diffusiongemma_second_corpus_gap(in_generation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gap_id": GAP_DIFFUSIONGEMMA_SECOND_CORPUS_SCORER,
        "status": "open",
        "evidence": (
            f"{EXP4325_PATH}; in_generation_moat_replicates="
            f"{in_generation.get('in_generation_moat_replicates')}; "
            f"scorer_leak_recheck_passed={in_generation.get('scorer_leak_recheck_passed')}; "
            f"benchmark_n={in_generation.get('benchmark_n')}; "
            f"answer_masked_auroc="
            f"{in_generation.get('independent_leak_recheck', {}).get('answer_masked_auroc')}"
        ),
        "failure_mode": (
            "the second-corpus in-generation replication did not run decision-grade "
            "rows because the partial-state scorer failed the independent leak recheck"
        ),
        "missing_discriminator": (
            "corpus-general leak-free partial-state scorer that remains predictive "
            "after answer-bearing cells are masked"
        ),
        "candidate_design": (
            "retrain and calibrate the scorer on multiple oracle-distinct corpora with "
            "fresh masked held-out AUROC gates before accepting another guided-generation moat"
        ),
        "priority": "high",
    }


def _arc_grid_generation_gap(adaptive: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gap_id": GAP_ARC_GRID_GENERATION_SCORER,
        "status": "open",
        "evidence": (
            f"{EXP4326_PATH}; adaptive_guidance_beats_control="
            f"{adaptive.get('adaptive_guidance_beats_control')}; "
            f"domain_used={adaptive.get('domain_used')}; "
            f"adaptive_ci95={adaptive.get('adaptive_ci95')}; "
            f"carnot_minus_best_control_delta="
            f"{adaptive.get('carnot_minus_best_control_delta')}"
        ),
        "failure_mode": (
            "adaptive DiffusionGemma guidance stayed bounded to a reasoning-corpus null "
            "and did not establish an ARC-grid generation scorer"
        ),
        "missing_discriminator": (
            "oracle-distinct ARC-grid partial-state generation scorer that can rank "
            "candidate grid states during denoising rather than post-hoc reasoning choices"
        ),
        "candidate_design": (
            "build a grid-native canvas scorer with masked-cell leak checks, no-adaptation "
            "controls, and ARC environment reproduction gates before rerunning adaptive guidance"
        ),
        "priority": "high",
    }


def _e3_world_model_gap(game: str, row: Mapping[str, Any]) -> dict[str, Any]:
    gap_id = {
        "ar25": GAP_E3_WORLD_MODEL_RULE_AR25,
        "ka59": GAP_E3_WORLD_MODEL_RULE_KA59,
        "tr87": GAP_E3_WORLD_MODEL_RULE_TR87,
        "ft09": GAP_E3_WORLD_MODEL_RULE_FT09,
    }[game]
    return {
        "gap_id": gap_id,
        "status": "open",
        "evidence": (
            f"{row.get('artifact_path')}; game={game}; offline_reproduced="
            f"{row.get('offline_reproduced')}; reproduced_levels="
            f"{row.get('reproduced_levels')}; verifier_best_accuracy="
            f"{row.get('verifier_best_accuracy')}; residual_mismatch_class="
            f"{row.get('residual_mismatch_class')}"
        ),
        "failure_mode": (
            f"E3 induced world model for {game} remained partial and could not execute "
            "a reproduced level through the real offline environment"
        ),
        "missing_discriminator": (
            f"{game} executable world-model rule coverage for "
            f"{row.get('residual_mismatch_class')}"
        ),
        "candidate_design": (
            "mine the divergent transition traces, add the missing action/rule cases "
            "to the executable model, and keep halt-on-divergence plus reproduce() as the gate"
        ),
        "priority": "high",
    }


def _learned_encoder_gap(learned: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gap_id": GAP_GAME_INVARIANT_ARC_VALUE_4331,
        "status": "open_small_encoder_insufficient",
        "evidence": (
            f"{EXP4331_PATH}; learned_encoder_transfer_helps="
            f"{learned.get('learned_encoder_transfer_helps')}; "
            f"cross_game_state_reduction={learned.get('cross_game_state_reduction')}; "
            f"cross_game_state_reduction_ci95="
            f"{learned.get('cross_game_state_reduction_ci95')}; "
            f"n_held_out_levels={learned.get('n_held_out_levels')}"
        ),
        "failure_mode": (
            "small learned frame encoder over the current solved set did not produce "
            "a decision-grade held-out OfflineSolver state reduction"
        ),
        "missing_discriminator": "game-invariant ARC value representation",
        "candidate_design": (
            "larger learned frame encoder, more reproduced solved traces, or "
            "adapter-conditioned value head with a hardware-portable path"
        ),
        "priority": "medium",
    }


def _dedupe_gap_entries(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        deduped[gap["gap_id"]] = gap
    return list(deduped.values())


def ensure_ledgers_record_v400(
    registry: dict[str, Any],
    gaps_text: str,
    exclusion_manifest: dict[str, Any],
    regression_guard: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return registry, gap text, and manifest with the .400 truth represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcome_bundle, gaps_logged)
    _ensure_v400_role(updated_registry, outcome_bundle, gaps_logged)

    updated_gaps = gaps_text
    for gap in gaps_logged:
        updated_gaps = base._replace_marked_block(
            updated_gaps,
            f"exp4333-{gap['gap_id'].lower()}",
            _gap_entry_block(gap),
        )

    updated_manifest = deepcopy(exclusion_manifest)
    _ensure_cross_domain_retirement(updated_manifest)
    gap_ids = [gap["gap_id"] for gap in gaps_logged]
    return (
        updated_registry,
        updated_gaps,
        updated_manifest,
        {
            "registry_reconciled": registry_contains_v400(updated_registry),
            "manifest_reconciled": manifest_contains_cross_domain_retirement(
                updated_manifest
            ),
            "gaps_logged_ids": [gap_id for gap_id in gap_ids if gap_id in updated_gaps],
        },
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    regression_guard: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    outcomes = outcome_bundle["v400_outcomes"]
    availability = outcome_bundle["availability_report"]
    arc1 = regression_guard.get("replayed_arc1_rule_exec", {})
    robust = outcomes["robust_aggregator"]
    in_generation = outcomes["in_generation_replication"]
    adaptive = outcomes["adaptive_scaleup"]
    e3 = outcomes["e3_deep_tail"]
    e3_games = e3["games"]
    shallow = outcomes["shallow_tail_sweep"]
    learned = outcomes["learned_encoder_transfer"]
    tn36 = shallow.get("tn36_schema_finding", {})
    eval_update = {
        "eval_exp_4333": EXP4333_ARTIFACT_PATH,
        "exp4333_regression_guard_passed": bool(
            regression_guard.get("regression_guard_passed")
        ),
        "exp4333_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
        "exp4333_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
        "exp4333_arc1_headroom_recovered": arc1.get("headroom_recovered"),
        "exp4333_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        "exp4333_v400_state": V400_STATE,
        "exp4333_robust_aggregator_used": robust_aggregator_ok(robust),
        "exp4333_available_artifact_keys": list(
            availability.get("available_artifact_keys", [])
        ),
        "exp4333_missing_upstream_artifacts": list(
            availability.get("missing_upstream_artifacts", [])
        ),
        "exp4333_flagged_artifacts_excluded": list(
            availability.get("flagged_artifacts_excluded", [])
        ),
        "exp4333_in_generation_artifact": EXP4325_PATH,
        "exp4333_in_generation_moat_replicates": in_generation.get(
            "in_generation_moat_replicates"
        ),
        "exp4333_in_generation_controls_differentiated": in_generation.get(
            "controls_differentiated"
        ),
        "exp4333_scorer_leak_recheck_passed": in_generation.get(
            "scorer_leak_recheck_passed"
        ),
        "exp4333_in_generation_benchmark_n": in_generation.get("benchmark_n"),
        "exp4333_in_generation_answer_masked_auroc": in_generation.get(
            "independent_leak_recheck", {}
        ).get("answer_masked_auroc"),
        "exp4333_adaptive_artifact": EXP4326_PATH,
        "exp4333_adaptive_guidance_beats_control": adaptive.get(
            "adaptive_guidance_beats_control"
        ),
        "exp4333_adaptive_domain_used": adaptive.get("domain_used"),
        "exp4333_adaptive_controls_differentiated": adaptive.get(
            "controls_differentiated"
        ),
        "exp4333_adaptive_scorer_leak_recheck_passed": adaptive.get(
            "scorer_leak_recheck_passed"
        ),
        "exp4333_adaptive_carnot_minus_best_control_delta": adaptive.get(
            "carnot_minus_best_control_delta"
        ),
        "exp4333_adaptive_ci95": adaptive.get("adaptive_ci95"),
        "exp4333_e3_reproduced_levels_total": e3.get("reproduced_levels_total"),
        "exp4333_e3_offline_reproduced_any": e3.get("offline_reproduced_any"),
        "exp4333_shallow_artifact": EXP4330_PATH,
        "exp4333_shallow_reproducible_total_levels": shallow.get(
            "reproducible_total_levels"
        ),
        "exp4333_shallow_games_advanced": shallow.get("games_advanced"),
        "exp4333_tn36_schema": tn36.get("schema"),
        "exp4333_tn36_normalizer": tn36.get("normalizer"),
        "exp4333_learned_encoder_artifact": EXP4331_PATH,
        "exp4333_learned_encoder_transfer_helps": learned.get(
            "learned_encoder_transfer_helps"
        ),
        "exp4333_cross_game_state_reduction": learned.get("cross_game_state_reduction"),
        "exp4333_cross_game_state_reduction_ci95": learned.get(
            "cross_game_state_reduction_ci95"
        ),
        "exp4333_cross_game_n_held_out_levels": learned.get("n_held_out_levels"),
        "exp4333_gaps_logged": [gap["gap_id"] for gap in gaps_logged],
    }
    for game, row in e3_games.items():
        eval_update[f"exp4333_e3_{game}_offline_reproduced"] = row.get(
            "offline_reproduced"
        )
        eval_update[f"exp4333_e3_{game}_reproduced_levels"] = row.get(
            "reproduced_levels"
        )
        eval_update[f"exp4333_e3_{game}_residual_mismatch_class"] = row.get(
            "residual_mismatch_class"
        )
        eval_update[f"exp4333_e3_{game}_verifier_best_accuracy"] = row.get(
            "verifier_best_accuracy"
        )
    entry.setdefault("eval", {}).update(eval_update)


def _ensure_v400_role(
    registry: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    outcomes = outcome_bundle["v400_outcomes"]
    shallow = outcomes["shallow_tail_sweep"]
    role = {
        "role_id": V400_ROLE_ID,
        "experiment": EXP4333_ARTIFACT_PATH,
        "role": "registry_gap_manifest_hygiene_v400",
        "status": "v400_outcomes_recorded_with_robust_availability",
        "v400_state": V400_STATE,
        "robust_aggregator_used": robust_aggregator_ok(outcomes["robust_aggregator"]),
        "in_generation_moat_replicates": outcomes["in_generation_replication"].get(
            "in_generation_moat_replicates"
        ),
        "adaptive_guidance_beats_control": outcomes["adaptive_scaleup"].get(
            "adaptive_guidance_beats_control"
        ),
        "e3_reproduced_levels_total": outcomes["e3_deep_tail"].get(
            "reproduced_levels_total"
        ),
        "shallow_reproducible_total_levels": shallow.get("reproducible_total_levels"),
        "tn36_schema": shallow.get("tn36_schema_finding", {}).get("schema"),
        "learned_encoder_transfer_helps": outcomes["learned_encoder_transfer"].get(
            "learned_encoder_transfer_helps"
        ),
        "cross_game_state_reduction": outcomes["learned_encoder_transfer"].get(
            "cross_game_state_reduction"
        ),
        "gap_ids_logged": [gap["gap_id"] for gap in gaps_logged],
        "eval_exp_4333": EXP4333_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V400_ROLE_ID] + [
        role
    ]


def _ensure_cross_domain_retirement(manifest: dict[str, Any]) -> None:
    if manifest_contains_cross_domain_retirement(manifest):
        return
    manifest.setdefault("retired_extras", []).append(
        {
            "id": CROSS_DOMAIN_RETIREMENT_ID,
            "experiment_scope": (
                "cross-domain verifier selection axis after repeated domain-bound "
                "IR3DE/CASCAL/ContextPRM selector verdicts"
            ),
            "reason": (
                "retire_if_same_verdict: Exp 4314 repeated Exp 4305's domain-bound "
                "cross-domain selection verdict; future cross-domain selector reruns "
                "need a new root cause, a different discriminator, and operator authorization."
            ),
            "experiment_ids": ["exp4305", "exp4314"],
            "retired_milestone": "2026.06.400",
            "retired_by_artifact": EXP4314_PATH,
            "recorded_by_artifact": EXP4333_ARTIFACT_PATH,
            "operator_reopen_required": True,
            "retire_if_same_verdict": True,
            "blocked_patterns": [
                "cross-domain verifier selection",
                "cross_domain_selector_ir3de_cascal",
                "family-invariant cross-domain selection rerun",
            ],
        }
    )


def manifest_contains_cross_domain_retirement(manifest: Mapping[str, Any]) -> bool:
    for entry in manifest.get("retired_extras", []):
        if isinstance(entry, Mapping) and entry.get("id") == CROSS_DOMAIN_RETIREMENT_ID:
            return True
    return False


def _gap_entry_block(gap: dict[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4333 .400 verifier gap update\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap['failure_mode']}\n"
        f"- missing discriminator: {gap['missing_discriminator']}\n"
        f"- candidate design: {gap['candidate_design']}\n"
        f"- priority: {gap['priority']}\n"
    )


def registry_contains_v400(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4333") == EXP4333_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4333_v400_state") == V400_STATE
        and any(role.get("role_id") == V400_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def model_specs_for_reconciliation() -> dict[str, Any]:
    return {
        "method": "cached_v400_ledger_reconciliation",
        "gap4_candidate_set": ARC1_POOL_PATH,
        "gap4_program_outputs": ARC1_PROGRAMS_PATH,
        "prior_hygiene_artifact": EXP4321_PATH,
        "upstream_artifacts": list(OUTCOME_ARTIFACT_PATHS),
        "robust_aggregator_artifact": EXP4308_PATH,
        "robust_aggregator_helper": "carnot.reporting.capstone_aggregate_available",
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
    outcome_bundle: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
    registry_reconciled: bool,
    manifest_reconciled: bool,
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4333 terminal JSON payload."""
    guard_ok = bool(regression_guard.get("regression_guard_passed"))
    complete = guard_ok and registry_reconciled and manifest_reconciled
    prefix = "complete:" if complete else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4333_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4333_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_manifest_reconciled_to_v400_truth_"
            f"regression_guard_passed_{guard_ok}_gaps_logged_{len(gaps_logged)}_"
            "robust_aggregator_used"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_logged": list(gaps_logged),
        "registry_reconciled": bool(registry_reconciled),
        "manifest_reconciled": bool(manifest_reconciled),
        "v400_outcomes": outcome_bundle["v400_outcomes"],
        "availability_report": outcome_bundle["availability_report"],
        "artifact_errors": outcome_bundle.get("artifact_errors", {}),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum,
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4333", "SCENARIO-VERIFY-4333"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "regression_guard": regression_guard,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "exclusion_manifest_path": EXCLUSION_MANIFEST_PATH,
        "cited_upstream_artifacts": list(OUTCOME_ARTIFACT_PATHS + [EXP4308_PATH]),
    }
    validate_artifact(artifact)
    return artifact


def _blocked_ledgers_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4333_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4333_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": "blocked_ledgers_unparseable",
        "regression_guard_passed": False,
        "gaps_logged": [],
        "registry_reconciled": False,
        "manifest_reconciled": False,
        "v400_outcomes": {},
        "availability_report": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "blocked:ledgers_unparseable",
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4333", "SCENARIO-VERIFY-4333"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions": preflight,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4333 fields before writing the artifact."""
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
    if not isinstance(artifact["v400_outcomes"], dict):
        raise ValueError("v400_outcomes must be an object")
    if not isinstance(artifact["availability_report"], dict):
        raise ValueError("availability_report must be an object")
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
        raise ValueError("field_principles must match the required Exp 4333 principles")
    if artifact["spec_refs"] != ["REQ-VERIFY-4333", "SCENARIO-VERIFY-4333"]:
        raise ValueError("spec_refs must cite REQ-VERIFY-4333 and SCENARIO-VERIFY-4333")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4333 and write the terminal artifact plus reconciled ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4333_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_ledgers_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    manifest_path = repo_root / EXCLUSION_MANIFEST_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    manifest = _load_manifest(manifest_path)
    regression_guard = run_gap4_regression_guard(repo_root)
    outcome_bundle = load_v400_outcomes(repo_root)
    gaps_logged = build_gap_entries(outcome_bundle)
    registry, gaps_text, manifest, ledger_summary = ensure_ledgers_record_v400(
        registry,
        gaps_text,
        manifest,
        regression_guard,
        outcome_bundle,
        gaps_logged,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    checksum = ledger_checksum(registry_path, gaps_path, manifest_path)
    artifact = build_artifact(
        regression_guard=regression_guard,
        outcome_bundle=outcome_bundle,
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
    print(f"Wrote {REPO_ROOT / EXP4333_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
