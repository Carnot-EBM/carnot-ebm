"""Exp 4266 registry/gaps hygiene for .394 verifier outcomes.

Spec refs: REQ-VERIFY-4266, SCENARIO-VERIFY-4266.

This runner reconciles the verifier registry and missing-verifier gap ledger to
the .394 fork truth. It does not run live inference or training; it only replays
the standing GAP-4 guard from cached artifacts and records the already-written
A1-A4, B1, C1, and C2 outcome artifacts.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
from typing import Any, Callable

import yaml

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4227 as exp4227
from carnot.reporting import verifier_registry_gaps_hygiene_4252 as exp4252


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4266
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4266_ARTIFACT_PATH = "results/experiment_4266_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
ARC1_POOL_PATH = exp4252.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4252.ARC1_PROGRAMS_PATH

EXP4252_PATH = exp4252.EXP4252_ARTIFACT_PATH
EXP4256_PATH = "results/experiment_4256_arc_oracle_distinct_leak_audit.json"
EXP4257_PATH = "results/experiment_4257_arc_oracle_distinct_multiseed_replication.json"
EXP4258_PATH = "results/experiment_4258_arc_oracle_distinct_cross_game_transfer.json"
EXP4259_PATH = "results/experiment_4259_arc_agglm_grid_synthesis.json"
EXP4260_PATH = "results/experiment_4260_diffusiongemma_energy_guided_preflight.json"
EXP4263_PATH = "results/experiment_4263_verifier_as_reward_out_of_band_or_retire.json"
EXP4264_PATH = "results/experiment_4264_code_oracle_distinct_replication_retry.json"

REQUIRED_UPSTREAM_PATHS = [
    EXP4252_PATH,
    EXP4256_PATH,
    EXP4257_PATH,
    EXP4258_PATH,
    EXP4259_PATH,
    EXP4260_PATH,
    EXP4263_PATH,
    EXP4264_PATH,
]

GAP_CROSS_GAME_ARC_SELECTION = "GAP-ARC-CROSS-GAME-SELECTION-4266"
GAP_SUPRA_ORACLE_SYNTHESIS = "GAP-ARC-SUPRA-ORACLE-K-SYNTHESIS-4266"
GAP_DIFFUSIONGEMMA_PREFLIGHT = "GAP-DIFFUSIONGEMMA-LOADER-GUIDANCE-4266"
GAP_CODE_ORACLE_DISTINCT_ROBUSTNESS = "GAP-CODE-ORACLE-DISTINCT-ROBUSTNESS-4266"
V394_ROLE_ID = "oracle_distinct_v394_hardened_hygiene_4266"
V394_HARDENED_STATE = (
    "within_pool_hardened_provenance_blind_multiseed__cross_game_blocked__"
    "synthesis_underperforms_selection"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_logged",
    "registry_reconciled",
    "v394_outcomes",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
    "adversarial_verify",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records the registry/gaps reconciled + regression guard result."
    ),
    "regression_guard_passed": (
        "BARE bool: the GAP-4 execution numbers did not regress vs .393 -- "
        "the standing-capability guard."
    ),
    "gaps_logged": (
        "List of new missing-verifier gap entries (failure mode + missing discriminator + "
        "candidate design + priority) -- the verifier build backlog (Missing-Verifier Gap Logging)."
    ),
    "reproducibility_checksum": "Hash of the reconciled registry + gaps; catches silent drift.",
}

GAP_ENTRY_REQUIRED_FIELDS = (
    "gap_id",
    "failure_mode",
    "missing_discriminator",
    "candidate_design",
    "priority",
)


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


def _check_resource(
    repo_root: Path,
    resource: str,
    path: str,
    loader: Callable[[Path], Any],
) -> dict[str, Any]:
    full_path = repo_root / path
    try:
        loader(full_path)
    except Exception as exc:  # pragma: no cover - exact parser exceptions vary.
        return {
            "resource": resource,
            "path": path,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"resource": resource, "path": path, "available": True, "error": ""}


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4266: ledgers parse and every .394 artifact exists before writes."""
    checks = [
        _check_resource(repo_root, "verifier_registry", REGISTRY_PATH, _load_registry_for_check),
        _check_resource(repo_root, "verifier_gaps", GAPS_PATH, _load_gaps_for_check),
        _check_resource(repo_root, "gap4_arc1_candidate_pool", ARC1_POOL_PATH, exp4227._load_gzip_json),
        _check_resource(repo_root, "gap4_arc1_programs", ARC1_PROGRAMS_PATH, base._load_json),
    ]
    checks.extend(
        _check_resource(repo_root, Path(path).stem, path, base._load_json)
        for path in REQUIRED_UPSTREAM_PATHS
    )
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def load_v394_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4266: read A1-A4, B1, C1, and C2 without fabricating fields."""
    a1 = base._load_json(repo_root / EXP4256_PATH)
    a2 = base._load_json(repo_root / EXP4257_PATH)
    a3 = base._load_json(repo_root / EXP4258_PATH)
    a4 = base._load_json(repo_root / EXP4259_PATH)
    b1 = base._load_json(repo_root / EXP4260_PATH)
    c1 = base._load_json(repo_root / EXP4263_PATH)
    c2 = base._load_json(repo_root / EXP4264_PATH)
    return {
        "a1_provenance_blind": {
            "artifact_path": EXP4256_PATH,
            "honest_verdict": str(a1.get("honest_verdict", "")),
            "headline_outcome": str(a1.get("headline_outcome", "")),
            "win_survives_provenance_blind": a1.get("win_survives_provenance_blind") is True,
            "provenance_blind_delta": a1.get("provenance_blind_delta"),
            "provenance_blind_ci95": a1.get("provenance_blind_ci95"),
            "origin_probe_auroc": a1.get("origin_probe_auroc"),
            "verifier_is_oracle": a1.get("verifier_is_oracle") is True,
        },
        "a2_multiseed": {
            "artifact_path": EXP4257_PATH,
            "honest_verdict": str(a2.get("honest_verdict", "")),
            "headline_outcome": str(a2.get("headline_outcome", "")),
            "oracle_distinct_win_replicates": a2.get("oracle_distinct_win_replicates") is True,
            "mean_delta": a2.get("mean_delta"),
            "cross_seed_ci95": a2.get("cross_seed_ci95"),
            "cross_seed_ci95_excludes_zero": a2.get("cross_seed_ci95_excludes_zero") is True,
            "n_seeds": a2.get("n_seeds"),
            "verifier_is_oracle": a2.get("verifier_is_oracle") is True,
        },
        "a3_cross_game": {
            "artifact_path": EXP4258_PATH,
            "honest_verdict": str(a3.get("honest_verdict", "")),
            "headline_outcome": str(a3.get("headline_outcome", "")),
            "honest_read": str(a3.get("honest_read", "")),
            "cross_game_transfer_claim": bool(
                a3.get("cross_game_delta") is not None and a3.get("ci95_excludes_zero") is True
            ),
            "cross_game_delta": a3.get("cross_game_delta"),
            "cross_game_ci95": a3.get("cross_game_ci95"),
            "held_out_game_n": a3.get("held_out_game_n"),
            "held_out_task_n": a3.get("held_out_task_n"),
            "verifier_is_oracle": a3.get("verifier_is_oracle") is True,
        },
        "a4_synthesis": {
            "artifact_path": EXP4259_PATH,
            "honest_verdict": str(a4.get("honest_verdict", "")),
            "headline_outcome": str(a4.get("headline_outcome", "")),
            "synthesis_beats_selection": a4.get("synthesis_beats_selection") is True,
            "synthesis_breaks_oracle_ceiling": a4.get("synthesis_breaks_oracle_ceiling") is True,
            "synthesis_minus_oracle_delta": a4.get("synthesis_minus_oracle_delta"),
            "synthesis_minus_selection_ci95": a4.get("synthesis_minus_selection_ci95"),
            "synthesis_minus_vote_delta": a4.get("synthesis_minus_vote_delta"),
            "oracle_at_k": a4.get("oracle_at_k"),
            "flagged_adversarial": a4.get("flagged_adversarial") is True,
            "verifier_is_oracle": a4.get("verifier_is_oracle") is True,
        },
        "b1_diffusiongemma": {
            "artifact_path": EXP4260_PATH,
            "honest_verdict": str(b1.get("honest_verdict", "")),
            "preflight_go": b1.get("preflight_go") is True,
            "guidance_changes_selection": b1.get("guidance_changes_selection") is True,
            "guidance_selection_change_count": b1.get("guidance_selection_change_count"),
            "verifier_is_oracle": b1.get("verifier_is_oracle") is True,
        },
        "c1_verifier_reward": {
            "artifact_path": EXP4263_PATH,
            "honest_verdict": str(c1.get("honest_verdict", "")),
            "ready_for_out_of_band": c1.get("ready_for_out_of_band") is True,
            "verifier_as_reward_retired": c1.get("verifier_as_reward_retired") is True,
            "out_of_band_runner_path": str(c1.get("out_of_band_runner_path", "")),
            "verifier_is_oracle": c1.get("verifier_is_oracle") is True,
        },
        "c2_code_replication": {
            "artifact_path": EXP4264_PATH,
            "honest_verdict": str(c2.get("honest_verdict", "")),
            "replication_read": str(c2.get("replication_read", "")),
            "code_replication_beats_vote": c2.get("code_replication_beats_vote") is True,
            "code_predictor_minus_vote_delta": c2.get("code_predictor_minus_vote_delta"),
            "code_predictor_minus_vote_ci95": c2.get("code_predictor_minus_vote_ci95"),
            "oracle_at_k": c2.get("oracle_at_k"),
            "off_fold_auroc": c2.get("off_fold_auroc"),
            "verifier_is_oracle": c2.get("verifier_is_oracle") is True,
        },
    }


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4266: compare cached replay with .393 recorded GAP-4 numbers."""
    prior = base._load_json(repo_root / EXP4252_PATH)
    recorded = dict(prior.get("offline_replay", {}).get("arc1_rule_exec", {}))
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
        "prior_artifact_path": EXP4252_PATH,
        "recorded_arc1_rule_exec": recorded,
        "replayed_arc1_rule_exec": replayed,
        "offline_replay": replay,
    }


def build_gap_entries(outcomes: dict[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4266: build Missing-Verifier Gap Logging entries for .394."""
    a3 = outcomes["a3_cross_game"]
    a4 = outcomes["a4_synthesis"]
    b1 = outcomes["b1_diffusiongemma"]
    c2 = outcomes["c2_code_replication"]
    return [
        {
            "gap_id": GAP_CROSS_GAME_ARC_SELECTION,
            "status": "open",
            "evidence": (
                f"{EXP4258_PATH}; honest_verdict={a3['honest_verdict']}; "
                f"held_out_game_n={a3['held_out_game_n']}; held_out_task_n={a3['held_out_task_n']}"
            ),
            "failure_mode": (
                "The ARC oracle-distinct selector is hardened within-pool, but cross-game "
                "transfer could not be measured because game/family ids were unrecoverable."
            ),
            "missing_discriminator": (
                "game/family provenance for every ARC candidate row so selection can be "
                "evaluated on family-disjoint held-out games."
            ),
            "candidate_design": (
                "Persist source-kind, generator family, game id, fold id, and target hash in "
                "the ARC candidate manifest, then rerun the set-encoder against vote on "
                "held-out families."
            ),
            "priority": "high",
        },
        {
            "gap_id": GAP_SUPRA_ORACLE_SYNTHESIS,
            "status": "open",
            "evidence": (
                f"{EXP4259_PATH}; synthesis_beats_selection={a4['synthesis_beats_selection']}; "
                f"synthesis_breaks_oracle_ceiling={a4['synthesis_breaks_oracle_ceiling']}; "
                f"synthesis_minus_oracle_delta={a4['synthesis_minus_oracle_delta']}"
            ),
            "failure_mode": (
                "Score-weighted grid synthesis underperformed selection and did not solve "
                "tasks beyond oracle@K candidate availability."
            ),
            "missing_discriminator": (
                "a supra-oracle@K verifier signal that can infer missing output cells or "
                "shapes when no cached candidate is correct."
            ),
            "candidate_design": (
                "Add rule-consistency or latent task-family constraints to propose cells "
                "outside the selected candidate family, with exact-match and selector-only controls."
            ),
            "priority": "high",
        },
        {
            "gap_id": GAP_DIFFUSIONGEMMA_PREFLIGHT,
            "status": "open",
            "evidence": (
                f"{EXP4260_PATH}; honest_verdict={b1['honest_verdict']}; "
                f"preflight_go={b1['preflight_go']}; "
                f"guidance_changes_selection={b1['guidance_changes_selection']}"
            ),
            "failure_mode": (
                "DiffusionGemma energy guidance remained blocked before the guidance hook could "
                "demonstrate verifier-shaped token selection."
            ),
            "missing_discriminator": (
                "a loader-validated diffusion guidance path that proves verifier energy changes "
                "denoising selections before a full run is scheduled."
            ),
            "candidate_design": (
                "Repair the GGUF vocab loader path, then run a tiny deterministic guidance smoke "
                "that records changed selections and exact verifier controls."
            ),
            "priority": "medium",
        },
        {
            "gap_id": GAP_CODE_ORACLE_DISTINCT_ROBUSTNESS,
            "status": "open",
            "evidence": (
                f"{EXP4264_PATH}; replication_read={c2['replication_read']}; "
                f"code_replication_beats_vote={c2['code_replication_beats_vote']}; "
                f"code_predictor_minus_vote_delta={c2['code_predictor_minus_vote_delta']}"
            ),
            "failure_mode": (
                "The code oracle-distinct read is corpus-specific; the second corpus did not "
                "replicate a vote-beating predictor despite oracle headroom."
            ),
            "missing_discriminator": (
                "source-robust code candidate features that distinguish real hidden-test pass "
                "signal from corpus-specific lexical or vote-signature artifacts."
            ),
            "candidate_design": (
                "Evaluate source-disjoint code pools with normalized-code, AST, agreement, and "
                "self-consistency features, plus per-source ablations before any robustness claim."
            ),
            "priority": "medium",
        },
    ]


def ensure_ledgers_record_v394(
    registry: dict[str, Any],
    gaps_text: str,
    regression_guard: dict[str, Any],
    outcomes: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gap text with the .394 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcomes, gaps_logged)
    _ensure_v394_role(updated_registry, outcomes, gaps_logged)

    updated_gaps = gaps_text
    for gap in gaps_logged:
        updated_gaps = base._replace_marked_block(
            updated_gaps,
            f"exp4266-{gap['gap_id'].lower()}",
            _gap_entry_block(gap),
        )
    gap_ids = [gap["gap_id"] for gap in gaps_logged]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_reconciled": registry_contains_v394(updated_registry),
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
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4266": EXP4266_ARTIFACT_PATH,
            "exp4266_regression_guard_passed": bool(
                regression_guard.get("regression_guard_passed")
            ),
            "exp4266_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4266_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4266_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4266_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4266_v394_hardened_state": V394_HARDENED_STATE,
            "exp4266_a1_provenance_blind_survives": bool(
                outcomes["a1_provenance_blind"]["win_survives_provenance_blind"]
            ),
            "exp4266_a1_provenance_blind_delta": outcomes["a1_provenance_blind"][
                "provenance_blind_delta"
            ],
            "exp4266_a2_multiseed_replicates": bool(
                outcomes["a2_multiseed"]["oracle_distinct_win_replicates"]
            ),
            "exp4266_a2_mean_delta": outcomes["a2_multiseed"]["mean_delta"],
            "exp4266_a3_cross_game_status": outcomes["a3_cross_game"]["honest_verdict"],
            "exp4266_a3_cross_game_transfer_claim": bool(
                outcomes["a3_cross_game"]["cross_game_transfer_claim"]
            ),
            "exp4266_a4_synthesis_beats_selection": bool(
                outcomes["a4_synthesis"]["synthesis_beats_selection"]
            ),
            "exp4266_a4_synthesis_breaks_oracle_ceiling": bool(
                outcomes["a4_synthesis"]["synthesis_breaks_oracle_ceiling"]
            ),
            "exp4266_a4_synthesis_minus_oracle_delta": outcomes["a4_synthesis"][
                "synthesis_minus_oracle_delta"
            ],
            "exp4266_b1_preflight_go": bool(outcomes["b1_diffusiongemma"]["preflight_go"]),
            "exp4266_b1_honest_verdict": outcomes["b1_diffusiongemma"]["honest_verdict"],
            "exp4266_c1_ready_for_out_of_band": bool(
                outcomes["c1_verifier_reward"]["ready_for_out_of_band"]
            ),
            "exp4266_c1_verifier_as_reward_retired": bool(
                outcomes["c1_verifier_reward"]["verifier_as_reward_retired"]
            ),
            "exp4266_c2_replication_read": outcomes["c2_code_replication"]["replication_read"],
            "exp4266_c2_code_replication_beats_vote": bool(
                outcomes["c2_code_replication"]["code_replication_beats_vote"]
            ),
            "exp4266_c2_code_predictor_minus_vote_delta": outcomes["c2_code_replication"][
                "code_predictor_minus_vote_delta"
            ],
            "exp4266_gaps_logged": [gap["gap_id"] for gap in gaps_logged],
        }
    )


def _ensure_v394_role(
    registry: dict[str, Any],
    outcomes: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    role = {
        "role_id": V394_ROLE_ID,
        "experiment": EXP4266_ARTIFACT_PATH,
        "role": "registry_gap_ledger_hygiene_v394",
        "status": "v394_hardened_outcomes_recorded_missing_verifier_gaps_logged",
        "v394_hardened_state": V394_HARDENED_STATE,
        "a1_artifact": EXP4256_PATH,
        "a1_win_survives_provenance_blind": outcomes["a1_provenance_blind"][
            "win_survives_provenance_blind"
        ],
        "a1_provenance_blind_delta": outcomes["a1_provenance_blind"][
            "provenance_blind_delta"
        ],
        "a2_artifact": EXP4257_PATH,
        "a2_oracle_distinct_win_replicates": outcomes["a2_multiseed"][
            "oracle_distinct_win_replicates"
        ],
        "a2_mean_delta": outcomes["a2_multiseed"]["mean_delta"],
        "a3_artifact": EXP4258_PATH,
        "a3_cross_game_status": outcomes["a3_cross_game"]["honest_verdict"],
        "a3_cross_game_transfer_claim": outcomes["a3_cross_game"]["cross_game_transfer_claim"],
        "a4_artifact": EXP4259_PATH,
        "a4_synthesis_beats_selection": outcomes["a4_synthesis"]["synthesis_beats_selection"],
        "a4_synthesis_breaks_oracle_ceiling": outcomes["a4_synthesis"][
            "synthesis_breaks_oracle_ceiling"
        ],
        "a4_synthesis_minus_oracle_delta": outcomes["a4_synthesis"][
            "synthesis_minus_oracle_delta"
        ],
        "b1_artifact": EXP4260_PATH,
        "b1_preflight_go": outcomes["b1_diffusiongemma"]["preflight_go"],
        "b1_honest_verdict": outcomes["b1_diffusiongemma"]["honest_verdict"],
        "c1_artifact": EXP4263_PATH,
        "c1_ready_for_out_of_band": outcomes["c1_verifier_reward"]["ready_for_out_of_band"],
        "c1_verifier_as_reward_retired": outcomes["c1_verifier_reward"][
            "verifier_as_reward_retired"
        ],
        "c2_artifact": EXP4264_PATH,
        "c2_replication_read": outcomes["c2_code_replication"]["replication_read"],
        "c2_code_replication_beats_vote": outcomes["c2_code_replication"][
            "code_replication_beats_vote"
        ],
        "c2_code_predictor_minus_vote_delta": outcomes["c2_code_replication"][
            "code_predictor_minus_vote_delta"
        ],
        "gap_ids_logged": [gap["gap_id"] for gap in gaps_logged],
        "eval_exp_4266": EXP4266_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V394_ROLE_ID] + [
        role
    ]


def _gap_entry_block(gap: dict[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4266 .394 missing-verifier gap\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap['failure_mode']}\n"
        f"- missing discriminator: {gap['missing_discriminator']}\n"
        f"- candidate design: {gap['candidate_design']}\n"
        f"- priority: {gap['priority']}\n"
    )


def registry_contains_v394(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4266") == EXP4266_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4266_v394_hardened_state") == V394_HARDENED_STATE
        and any(role.get("role_id") == V394_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def ledger_checksum(registry_path: Path, gaps_path: Path) -> str:
    """REQ-VERIFY-4266: hash reconciled registry + gaps to catch silent drift."""
    digest = hashlib.sha256()
    for label, path in (("registry", registry_path), ("gaps", gaps_path)):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def model_specs_for_reconciliation() -> dict[str, Any]:
    return {
        "method": "cached_v394_ledger_reconciliation",
        "gap4_candidate_set": ARC1_POOL_PATH,
        "gap4_program_outputs": ARC1_PROGRAMS_PATH,
        "upstream_artifacts": list(REQUIRED_UPSTREAM_PATHS),
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
    v394_outcomes: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
    registry_reconciled: bool,
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4266 terminal JSON payload."""
    guard_ok = bool(regression_guard.get("regression_guard_passed"))
    complete = guard_ok and registry_reconciled and bool(gaps_logged)
    prefix = "complete:" if complete else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4266_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4266_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v394_truth_"
            f"regression_guard_passed_{guard_ok}_gaps_logged_{len(gaps_logged)}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_logged": list(gaps_logged),
        "registry_reconciled": bool(registry_reconciled),
        "v394_outcomes": v394_outcomes,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum,
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4266", "SCENARIO-VERIFY-4266"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": {"status": "pending"},
        "regression_guard": regression_guard,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "cited_upstream_artifacts": list(REQUIRED_UPSTREAM_PATHS) + [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
        ],
        "v394_hardened_state": V394_HARDENED_STATE,
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4266_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4266_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": "blocked_v394_artifacts_missing",
        "regression_guard_passed": False,
        "gaps_logged": [],
        "registry_reconciled": False,
        "v394_outcomes": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "blocked:v394_artifacts_missing",
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4266", "SCENARIO-VERIFY-4266"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": {"status": "pending"},
        "preconditions": preflight,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4266 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["regression_guard_passed"], bool):
        raise ValueError("regression_guard_passed must be a BARE bool")
    if not isinstance(artifact["registry_reconciled"], bool):
        raise ValueError("registry_reconciled must be a bare bool")
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
        raise ValueError("field_principles must match the required Exp 4266 principles")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def _run_adversarial_verify(
    repo_root: Path, artifact_path: Path
) -> dict[str, Any]:  # pragma: no cover
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "adversarial_verify.py"),
            "--json",
            str(artifact_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def _clean_adversarial_report(report: dict[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    for item in report.get("reports", []):
        if isinstance(item, dict):
            flags.extend(flag for flag in item.get("flags", []) if isinstance(flag, dict))
    methodology_missing_clean = not any(flag.get("kind") == "METHODOLOGY_MISSING" for flag in flags)
    return {
        "status": "clean" if not flags else "flagged",
        "methodology_missing_clean": methodology_missing_clean,
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Exp 4266 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4266_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    regression_guard = run_gap4_regression_guard(repo_root)
    outcomes = load_v394_outcomes(repo_root)
    gaps_logged = build_gap_entries(outcomes)
    registry, gaps_text, ledger_summary = ensure_ledgers_record_v394(
        registry,
        gaps_text,
        regression_guard,
        outcomes,
        gaps_logged,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    checksum = ledger_checksum(registry_path, gaps_path)
    artifact = build_artifact(
        regression_guard=regression_guard,
        v394_outcomes=outcomes,
        gaps_logged=gaps_logged,
        registry_reconciled=bool(ledger_summary["registry_reconciled"]),
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    raw_report = (
        adversarial_runner(out_path)
        if adversarial_runner is not None
        else _run_adversarial_verify(repo_root, out_path)
    )
    artifact["adversarial_verify"] = _clean_adversarial_report(raw_report)
    validate_artifact(artifact)
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4266_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
