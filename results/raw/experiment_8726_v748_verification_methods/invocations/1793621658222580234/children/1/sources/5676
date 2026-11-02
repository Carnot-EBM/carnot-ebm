"""Exp 4051 registry and gaps hygiene for the GAP-4 verifier line.

Spec refs: REQ-VERIFY-4051, SCENARIO-VERIFY-4051.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP4051_ARTIFACT_PATH = "results/experiment_4051_verifier_registry_and_gaps_hygiene.json"
REGISTRY_PATH = "ops/verifier_registry.yaml"
GAPS_PATH = "ops/verifier_gaps.md"

ARC1_RULE_EXEC_PATH = "results/arc3_gap4_rule_exec_verifier.json"
ARC2_RULE_EXEC_PATH = "results/arc3_gap4_arc2_rule_exec_verifier.json"
ARC2_CHAIN_PATH = "results/arc3_gap4_arc2_chain_ensemble.json"
G1_OFF_ARC_PATH = "results/experiment_4045_offarc_transfer_power.json"
G2_CLOSED_LOOP_PATH = "results/experiment_4046_closed_loop_replan_over_vc33_wm.json"

GAP4_VERIFIER_ID = "gap4_program_induction_stack"
G1_DEMOFIT_VERIFIER_ID = "gap4_code_demo_fit_execution_transfer_4045"
G2_CLOSED_LOOP_VERIFIER_ID = "arc3_vc33_verified_wm_closed_loop_planner_4046"
CODE_GAP_ID = "GAP-CODE-EXEC-DEMOFIT"
SIM2REAL_GAP_ID = "GAP-ARC3-VC33-SIM2REAL-CEILING"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "offline_reeval_bitexact",
    "g1_off_arc_outcome_recorded",
    "g2_closed_loop_outcome_recorded",
    "registry_updated",
    "gaps_updated",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix summary of GAP-4 replay and .374 G1/G2 ledger routing.",
    "offline_reeval_bitexact": (
        "A reusable verifier that does not reproduce its headline numbers is not the same verifier."
    ),
    "g1_off_arc_outcome_recorded": (
        "The off-ARC power measurement must land in the ledger, not just an artifact."
    ),
    "g2_closed_loop_outcome_recorded": (
        "The planning outcome becomes either a registry capability, a new gap, or a pending record."
    ),
    "registry_updated": "Bare bool; registry reflects the verifier coverage after Exp 4051.",
    "gaps_updated": "Bare bool; gaps ledger reflects the missing-verifier state after Exp 4051.",
    "inference_substrate": "aggregation_from_upstream_artifacts; no Codex, GGUF, or live inference.",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_registry(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        loaded.setdefault("verifiers", [])
        return loaded
    return {"verifiers": []}


def _write_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")


def _find_verifier(registry: dict[str, Any], verifier_id: str) -> dict[str, Any] | None:
    for entry in registry.get("verifiers", []):
        if entry.get("verifier_id") == verifier_id:
            return entry
    return None


def _upsert_verifier(registry: dict[str, Any], entry: dict[str, Any]) -> bool:
    existing = _find_verifier(registry, entry["verifier_id"])
    if existing is None:
        registry.setdefault("verifiers", []).append(entry)
        return True
    if existing != entry:
        existing.clear()
        existing.update(entry)
        return True
    return False


def _round4(value: float) -> float:
    return round(float(value), 4)


def replay_gap4_headlines(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Replay GAP-4 ARC headline numbers from cached artifacts only."""
    arc1 = _load_json(repo_root / ARC1_RULE_EXEC_PATH)
    arc2 = _load_json(repo_root / ARC2_RULE_EXEC_PATH)
    arc2_chain = _load_json(repo_root / ARC2_CHAIN_PATH)

    arc1_summary = {
        "n": int(arc1["n_tasks"]),
        "vote_pass2": _round4(arc1["rankers"]["TRM_VOTE"]["pass@2"]),
        "gated_pass2": _round4(arc1["rankers"]["GAP4_GATED"]["pass@2"]),
        "headroom_recovered": int(arc1["gates"]["headroom_recovered"]),
        "vote_wins_lost": int(arc1["gates"]["vote_wins_lost"]),
    }
    arc2_summary = {
        "n": int(arc2["n_tasks"]),
        "vote_pass2": _round4(arc2["rankers"]["TRM_VOTE"]["pass@2"]),
        "gated_pass2": _round4(arc2["rankers"]["GAP4_GATED"]["pass@2"]),
        "headroom_recovered": int(arc2["gates"]["headroom_recovered"]),
        "vote_wins_lost": int(arc2["gates"]["vote_wins_lost"]),
    }
    fresh = arc2_chain["per_arm_gold_given_perfect"]["fresh"]
    chain_summary = {
        "gold": int(fresh["gold"]),
        "n": 31,
        "pass_at_1": _round4(int(fresh["gold"]) / 31),
    }
    expected = {
        "arc1_rule_exec": {
            "n": 31,
            "vote_pass2": 0.4516,
            "gated_pass2": 0.5806,
            "headroom_recovered": 4,
            "vote_wins_lost": 0,
        },
        "arc2_rule_exec": {
            "n": 31,
            "vote_pass2": 0.0645,
            "gated_pass2": 0.0645,
            "headroom_recovered": 0,
            "vote_wins_lost": 0,
        },
        "arc2_registered_chain": {
            "gold": 19,
            "n": 31,
            "pass_at_1": 0.6129,
        },
    }
    observed = {
        "arc1_rule_exec": arc1_summary,
        "arc2_rule_exec": arc2_summary,
        "arc2_registered_chain": chain_summary,
    }
    return {
        "offline_reeval_bitexact": observed == expected,
        **observed,
        "expected": expected,
    }


def classify_g1_off_arc_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify Exp 4045 into registry transfer, stronger closure, open gap, or pending."""
    rel_path = G1_OFF_ARC_PATH
    path = repo_root / rel_path
    if not path.exists():
        return _pending_g1("missing_exp4045_artifact", rel_path)

    artifact = _load_json(path)
    common = _g1_common(rel_path, artifact)
    n_tasks = int(artifact.get("n_tasks", 0))
    powered_floor = int(artifact.get("powered_task_floor", 160))
    verdict = str(artifact.get("honest_verdict", ""))
    if (
        n_tasks < powered_floor
        or "partial" in verdict
        or artifact.get("raw_artifact_present") is False
    ):
        return _pending_g1("partial_or_incomplete_exp4045", rel_path, artifact)

    if bool(artifact.get("demofit_ci_excludes_zero")):
        return {
            "g1_off_arc_outcome_recorded": "g1_demofit_ci_excludes_zero",
            "status": "transfer",
            **common,
        }

    if bool(artifact.get("best_arm_ci_excludes_zero")):
        best_arm = str(artifact.get("best_arm", "stronger_arm"))
        return {
            "g1_off_arc_outcome_recorded": f"g1_stronger_{best_arm}_ci_excludes_zero",
            "status": "stronger_discriminator_registered",
            **common,
        }

    return {
        "g1_off_arc_outcome_recorded": "g1_all_arms_touch_zero_gap_open",
        "status": "gap_open",
        **common,
    }


def _pending_g1(
    reason: str,
    artifact_path: str,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "g1_off_arc_outcome_recorded": "g1_off_arc_power_pending",
        "status": "pending",
        "reason": reason,
        "artifact_path": artifact_path,
    }
    if artifact is not None:
        payload.update(_g1_common(artifact_path, artifact))
    return payload


def _g1_common(artifact_path: str, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": artifact_path,
        "n_tasks": int(artifact.get("n_tasks", 0)),
        "powered_task_floor": int(artifact.get("powered_task_floor", 160)),
        "demofit_delta_pp": float(artifact.get("demofit_delta_pp", 0.0)),
        "demofit_bootstrap_ci95": list(artifact.get("demofit_bootstrap_ci95", [0.0, 0.0])),
        "best_arm": str(artifact.get("best_arm", "")),
        "best_arm_delta_pp": float(artifact.get("best_arm_delta_pp", 0.0)),
        "best_arm_ci95": list(artifact.get("best_arm_ci95", [0.0, 0.0])),
        "oracle_passrate": float(artifact.get("oracle_passrate", 0.0)),
        "oracle_headroom": bool(artifact.get("oracle_headroom", False)),
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
    }


def classify_g2_closed_loop_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify Exp 4046 as closed-loop capability, sim2real ceiling gap, or pending."""
    rel_path = G2_CLOSED_LOOP_PATH
    path = repo_root / rel_path
    if not path.exists():
        return _pending_g2("missing_exp4046_artifact", rel_path)

    artifact = _load_json(path)
    common = _g2_common(rel_path, artifact)
    if bool(artifact.get("closed_loop_broke_wall")):
        return {
            "g2_closed_loop_outcome_recorded": "g2_closed_loop_capability_registered",
            "status": "capability_registered",
            **common,
        }

    return {
        "g2_closed_loop_outcome_recorded": "g2_sim2real_ceiling_gap_logged",
        "status": "gap_logged",
        **common,
    }


def _pending_g2(reason: str, artifact_path: str) -> dict[str, Any]:
    return {
        "g2_closed_loop_outcome_recorded": "g2_closed_loop_pending",
        "status": "pending",
        "reason": reason,
        "artifact_path": artifact_path,
    }


def _g2_common(artifact_path: str, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": artifact_path,
        "closed_loop_broke_wall": bool(artifact.get("closed_loop_broke_wall", False)),
        "new_levels_solved_this_task": int(artifact.get("new_levels_solved_this_task", 0)),
        "per_step_wm_real_divergence_rate": float(
            artifact.get("per_step_wm_real_divergence_rate", 0.0)
        ),
        "divergence_gate_fired_count": int(artifact.get("divergence_gate_fired_count", 0)),
        "real_env_confirmed": bool(artifact.get("real_env_confirmed", False)),
        "goal_predicate_heldout_precision": float(
            artifact.get("goal_predicate_heldout_precision", 0.0)
        ),
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    g1_outcome: dict[str, Any],
    g2_outcome: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, bool]]:
    """Return registry and gaps text with Exp 4051 outcomes represented idempotently."""
    updated_registry = deepcopy(registry)
    updated_gaps = gaps_text
    registry_changed = _ensure_gap4_eval(updated_registry, offline_replay)

    g1_recorded = g1_outcome["g1_off_arc_outcome_recorded"]
    if g1_recorded == "g1_demofit_ci_excludes_zero":
        registry_changed = (
            _upsert_verifier(updated_registry, _g1_demofit_entry(g1_outcome)) or registry_changed
        )
    elif g1_outcome["status"] == "stronger_discriminator_registered":
        registry_changed = (
            _upsert_verifier(updated_registry, _g1_stronger_entry(g1_outcome)) or registry_changed
        )
    else:
        updated_gaps = _replace_marked_block(updated_gaps, "exp4051-g1", _g1_gap_block(g1_outcome))

    g2_recorded = g2_outcome["g2_closed_loop_outcome_recorded"]
    if g2_recorded == "g2_closed_loop_capability_registered":
        registry_changed = (
            _upsert_verifier(updated_registry, _g2_capability_entry(g2_outcome)) or registry_changed
        )
    else:
        updated_gaps = _replace_marked_block(updated_gaps, "exp4051-g2", _g2_gap_block(g2_outcome))

    registry_ok = _registry_contains_outcomes(updated_registry, g1_outcome, g2_outcome)
    gaps_ok = _gaps_contain_outcomes(updated_gaps, g1_outcome, g2_outcome)
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": registry_ok or registry_changed,
            "gaps_updated": gaps_ok,
        },
    )


def _ensure_gap4_eval(registry: dict[str, Any], offline_replay: dict[str, Any]) -> bool:
    entry = _find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - defensive for malformed local ledgers
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    eval_block = entry.setdefault("eval", {})
    required = {
        "eval_exp_4051": EXP4051_ARTIFACT_PATH,
        "arc1_rule_exec_vote_pass2": offline_replay.get("arc1_rule_exec", {}).get("vote_pass2"),
        "arc1_rule_exec_gated_pass2": offline_replay.get("arc1_rule_exec", {}).get("gated_pass2"),
        "arc2_rule_exec_vote_pass2": offline_replay.get("arc2_rule_exec", {}).get("vote_pass2"),
        "arc2_rule_exec_gated_pass2": offline_replay.get("arc2_rule_exec", {}).get("gated_pass2"),
        "exp4051_offline_reeval_bitexact": bool(offline_replay.get("offline_reeval_bitexact")),
    }
    changed = False
    for key, value in required.items():
        if eval_block.get(key) != value:
            eval_block[key] = value
            changed = True
    return changed


def _g1_demofit_entry(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "verifier_id": G1_DEMOFIT_VERIFIER_ID,
        "domain": "code",
        "version": 1,
        "kind": "process_verifier",
        "code_commit": "HEAD",
        "code_path": "scripts/experiments/exp4045_offarc_transfer_power_collect.py",
        "weights_hf": None,
        "weights_cid": None,
        "weights_sha256": None,
        "training_data_ref": outcome["artifact_path"],
        "label_source": "hidden_tests",
        "eval": {
            "metric": "hidden_pass_at_1_delta_pp",
            "delta_pp": outcome["demofit_delta_pp"],
            "bootstrap_ci95": outcome["demofit_bootstrap_ci95"],
            "n": outcome["n_tasks"],
            "eval_artifact": outcome["artifact_path"],
        },
        "lineage": {
            "from": GAP4_VERIFIER_ID,
            "change": "Exp 4051 recorded Exp 4045 demo-fit CI excluding zero.",
        },
        "status": "candidate",
        "notes": "G1 off-ARC power outcome recorded by Exp 4051.",
    }


def _g1_stronger_entry(outcome: dict[str, Any]) -> dict[str, Any]:
    best_arm = outcome["best_arm"] or "stronger_arm"
    return {
        "verifier_id": f"gap4_code_{best_arm}_transfer_4045",
        "domain": "code",
        "version": 1,
        "kind": "process_verifier",
        "code_commit": "HEAD",
        "code_path": "scripts/experiments/exp4045_offarc_transfer_power_collect.py",
        "weights_hf": None,
        "weights_cid": None,
        "weights_sha256": None,
        "training_data_ref": outcome["artifact_path"],
        "label_source": "hidden_tests",
        "eval": {
            "metric": "hidden_pass_at_1_delta_pp",
            "delta_pp": outcome["best_arm_delta_pp"],
            "bootstrap_ci95": outcome["best_arm_ci95"],
            "n": outcome["n_tasks"],
            "eval_artifact": outcome["artifact_path"],
        },
        "lineage": {
            "from": GAP4_VERIFIER_ID,
            "change": f"Exp 4051 recorded stronger code-domain discriminator {best_arm}.",
        },
        "status": "candidate",
        "notes": "G1 stronger-arm off-ARC outcome recorded by Exp 4051.",
    }


def _g2_capability_entry(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "verifier_id": G2_CLOSED_LOOP_VERIFIER_ID,
        "domain": "arc_agi3_game",
        "version": 1,
        "kind": "closed_loop_planner",
        "code_commit": "HEAD",
        "code_path": "python/carnot/agentic/arc_vc33_closed_loop_replan.py",
        "weights_hf": None,
        "weights_cid": None,
        "weights_sha256": None,
        "training_data_ref": outcome["artifact_path"],
        "label_source": "real_env",
        "eval": {
            "metric": "new_levels_solved_this_task",
            "value": int(outcome.get("new_levels_solved_this_task", 0)),
            "wm_real_divergence_rate": outcome["per_step_wm_real_divergence_rate"],
            "eval_artifact": outcome["artifact_path"],
        },
        "lineage": {
            "from": "arc3_vc33_verified_world_model",
            "change": "Exp 4051 recorded verified-WM closed-loop planner as a capability.",
        },
        "status": "candidate",
        "notes": "G2 closed-loop outcome recorded by Exp 4051.",
    }


def _g1_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"#### Exp 4051 G1 off-ARC power update for {CODE_GAP_ID}\n"
        f"- status: {outcome['g1_off_arc_outcome_recorded']}\n"
        f"- evidence: `{outcome.get('artifact_path', G1_OFF_ARC_PATH)}`; "
        f"n_tasks={outcome.get('n_tasks', 0)}; powered_task_floor="
        f"{outcome.get('powered_task_floor', 160)}; demo_fit_CI95="
        f"{outcome.get('demofit_bootstrap_ci95')}; best_arm={outcome.get('best_arm')}; "
        f"best_arm_CI95={outcome.get('best_arm_ci95')}.\n"
        "- failure mode: visible/demo-fit code tests are not yet a powered hidden-semantic discriminator.\n"
        "- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics.\n"
        "- candidate design: finish the powered code run or add hidden-property, symbolic, or formal/runtime oracles.\n"
        "- priority: high\n"
    )


def _g2_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {SIM2REAL_GAP_ID}: vc33 verified-WM closed-loop sim2real ceiling\n"
        f"- status: {outcome['g2_closed_loop_outcome_recorded']}\n"
        f"- evidence: `{outcome.get('artifact_path', G2_CLOSED_LOOP_PATH)}`; "
        f"per_step_wm_real_divergence_rate="
        f"{outcome.get('per_step_wm_real_divergence_rate')}; "
        f"divergence_gate_fired_count={outcome.get('divergence_gate_fired_count')}.\n"
        "- failure mode: bounded WM search produced a plan whose predicted next state diverged from the real environment.\n"
        "- missing discriminator: per-step WM-to-real transition trust signal strong enough to plan past vc33's wall.\n"
        "- candidate design: improve the verified world model or add a conservative real-env grounding/replan guard.\n"
        "- priority: high\n"
    )


def _replace_marked_block(text: str, marker: str, block: str) -> str:
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    replacement = f"{start}\n{block.rstrip()}\n{end}"
    if start in text and end in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        return f"{prefix}{replacement}{suffix}"
    return text.rstrip() + "\n\n" + replacement + "\n"


def _registry_contains_outcomes(
    registry: dict[str, Any],
    g1_outcome: dict[str, Any],
    g2_outcome: dict[str, Any],
) -> bool:
    gap4 = _find_verifier(registry, GAP4_VERIFIER_ID)
    if gap4 is None:
        return False
    if gap4.get("eval", {}).get("eval_exp_4051") != EXP4051_ARTIFACT_PATH:
        return False
    ok = True
    if g1_outcome["g1_off_arc_outcome_recorded"] == "g1_demofit_ci_excludes_zero":
        ok = ok and _find_verifier(registry, G1_DEMOFIT_VERIFIER_ID) is not None
    if g2_outcome["g2_closed_loop_outcome_recorded"] == "g2_closed_loop_capability_registered":
        ok = ok and _find_verifier(registry, G2_CLOSED_LOOP_VERIFIER_ID) is not None
    return ok


def _gaps_contain_outcomes(
    gaps_text: str,
    g1_outcome: dict[str, Any],
    g2_outcome: dict[str, Any],
) -> bool:
    needs_g1_gap = g1_outcome["g1_off_arc_outcome_recorded"] not in {
        "g1_demofit_ci_excludes_zero",
        "g1_stronger_armC_symbolic_ci_excludes_zero",
        "g1_stronger_armApp_aces_ci_excludes_zero",
    }
    needs_g2_gap = (
        g2_outcome["g2_closed_loop_outcome_recorded"] != "g2_closed_loop_capability_registered"
    )
    g1_ok = (not needs_g1_gap) or (
        CODE_GAP_ID in gaps_text and g1_outcome["g1_off_arc_outcome_recorded"] in gaps_text
    )
    g2_ok = (not needs_g2_gap) or (
        SIM2REAL_GAP_ID in gaps_text and g2_outcome["g2_closed_loop_outcome_recorded"] in gaps_text
    )
    return g1_ok and g2_ok and (needs_g1_gap or needs_g2_gap)


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    g1_outcome: dict[str, Any],
    g2_outcome: dict[str, Any],
    registry_updated: bool,
    gaps_updated: bool,
    duration_s: float,
) -> dict[str, Any]:
    bit_label = "bitexact" if offline_replay.get("offline_reeval_bitexact") else "mismatch"
    g1_label = g1_outcome["g1_off_arc_outcome_recorded"]
    g2_label = g2_outcome["g2_closed_loop_outcome_recorded"]
    artifact = {
        "experiment": "experiment_4051_verifier_registry_and_gaps_hygiene",
        "schema": "carnot.experiment_4051_verifier_registry_and_gaps_hygiene.v1",
        "honest_verdict": f"complete: gap4_reeval_{bit_label}_g1_{g1_label}_g2_{g2_label}_recorded",
        "offline_reeval_bitexact": bool(offline_replay.get("offline_reeval_bitexact")),
        "g1_off_arc_outcome_recorded": g1_label,
        "g2_closed_loop_outcome_recorded": g2_label,
        "registry_updated": bool(registry_updated),
        "gaps_updated": bool(gaps_updated),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 3),
        "offline_replay": offline_replay,
        "g1_off_arc_outcome": g1_outcome,
        "g2_closed_loop_outcome": g2_outcome,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_RULE_EXEC_PATH,
            ARC2_RULE_EXEC_PATH,
            ARC2_CHAIN_PATH,
            G1_OFF_ARC_PATH,
            G2_CLOSED_LOOP_PATH,
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")  # pragma: no cover
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")  # pragma: no cover
    for field in ("offline_reeval_bitexact", "registry_updated", "gaps_updated"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")  # pragma: no cover
    if not isinstance(artifact["g1_off_arc_outcome_recorded"], str):
        raise ValueError("g1_off_arc_outcome_recorded must be a string")  # pragma: no cover
    if not isinstance(artifact["g2_closed_loop_outcome_recorded"], str):
        raise ValueError("g2_closed_loop_outcome_recorded must be a string")  # pragma: no cover
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")  # pragma: no cover


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    started = time.time()
    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH

    registry = _load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_headlines(repo_root)
    g1_outcome = classify_g1_off_arc_outcome(repo_root)
    g2_outcome = classify_g2_closed_loop_outcome(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        g1_outcome,
        g2_outcome,
    )
    _write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        g1_outcome=g1_outcome,
        g2_outcome=g2_outcome,
        registry_updated=ledger_summary["registry_updated"],
        gaps_updated=ledger_summary["gaps_updated"],
        duration_s=time.time() - started,
    )
    _write_json(repo_root / EXP4051_ARTIFACT_PATH, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised by the experiment command
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4051_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
