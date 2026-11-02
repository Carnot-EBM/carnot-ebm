"""Exp 4063 registry and gaps hygiene for the GAP-4 verifier line.

Spec refs: REQ-VERIFY-4063, SCENARIO-VERIFY-4063.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP4063_ARTIFACT_PATH = "results/experiment_4063_verifier_registry_and_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_RULE_EXEC_PATH = base.ARC1_RULE_EXEC_PATH
ARC2_RULE_EXEC_PATH = base.ARC2_RULE_EXEC_PATH
ARC2_CHAIN_PATH = base.ARC2_CHAIN_PATH
G1_EVALPLUS_PATH = "results/experiment_4057_offarc_power_evalplus.json"
G3_DECENTRALIZATION_PATH = "results/experiment_4059_decentralization_moe_resume.json"
G2_CLOSED_LOOP_PATH = base.G2_CLOSED_LOOP_PATH

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
CODE_GAP_ID = base.CODE_GAP_ID
DECENTRALIZATION_GAP_ID = "GAP-DECENTRALIZATION-MOE-BASE-4048"
SIM2REAL_GAP_ID = base.SIM2REAL_GAP_ID
G1_DEMOFIT_VERIFIER_ID = "gap4_code_evalplus_demo_fit_transfer_4057"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "offline_reeval_bitexact",
    "g1_off_arc_outcome_recorded",
    "g3_decentralization_outcome_recorded",
    "g2_vc33_ceiling_logged",
    "registry_updated",
    "gaps_updated",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix summary of GAP-4 replay and .375 G1/G3/G2 ledger routing.",
    "offline_reeval_bitexact": (
        "A reusable verifier that does not reproduce its headline numbers is not the same verifier."
    ),
    "g1_off_arc_outcome_recorded": (
        "The EvalPlus off-ARC power measurement must land in the ledger, not just an artifact."
    ),
    "g3_decentralization_outcome_recorded": (
        "The MoE-base accumulated coverage result must be recorded as latent, absent, accumulating, or pending."
    ),
    "g2_vc33_ceiling_logged": "Bare bool; the banked vc33 sim2real ceiling is visible in the gaps ledger.",
    "registry_updated": "Bare bool; registry reflects the replayed GAP-4 line after Exp 4063.",
    "gaps_updated": "Bare bool; gaps ledger reflects the missing-verifier state after Exp 4063.",
    "inference_substrate": "aggregation_from_upstream_artifacts; no Codex, GGUF, or live inference.",
}


def replay_gap4_headlines(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Replay GAP-4 ARC headline numbers from cached artifacts only."""
    return base.replay_gap4_headlines(repo_root)


def classify_g1_evalplus_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify Exp 4057 into EvalPlus transfer, stronger closure, open gap, or accumulating."""
    rel_path = G1_EVALPLUS_PATH
    path = repo_root / rel_path
    if not path.exists():
        return _pending_g1("missing_exp4057_artifact", rel_path)

    artifact = base._load_json(path)
    common = _g1_common(rel_path, artifact)
    accumulated_n = int(artifact.get("accumulated_n_tasks", 0))
    powered_floor = int(artifact.get("powered_task_floor", 160))
    verdict = str(artifact.get("honest_verdict", ""))
    if (
        accumulated_n < powered_floor
        or verdict.startswith("blocked_")
        or artifact.get("raw_artifact_present") is False
        or not bool(artifact.get("oracle_headroom_present", False))
    ):
        return {
            "g1_off_arc_outcome_recorded": "g1_evalplus_accumulating",
            "status": "accumulating",
            **common,
        }

    if bool(artifact.get("demofit_ci_excludes_zero")):
        return {
            "g1_off_arc_outcome_recorded": "g1_evalplus_demofit_ci_excludes_zero",
            "status": "transfer",
            **common,
        }

    if bool(artifact.get("best_arm_ci_excludes_zero")):
        best_arm = str(artifact.get("best_arm", "stronger_arm"))
        return {
            "g1_off_arc_outcome_recorded": f"g1_evalplus_stronger_{best_arm}_ci_excludes_zero",
            "status": "stronger_discriminator_registered",
            **common,
        }

    return {
        "g1_off_arc_outcome_recorded": "g1_evalplus_all_arms_touch_zero_gap_open",
        "status": "gap_open",
        **common,
    }


def _pending_g1(reason: str, artifact_path: str) -> dict[str, Any]:
    return {
        "g1_off_arc_outcome_recorded": "g1_evalplus_pending",
        "status": "pending",
        "reason": reason,
        "artifact_path": artifact_path,
    }


def _g1_common(artifact_path: str, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": artifact_path,
        "accumulated_n_tasks": int(artifact.get("accumulated_n_tasks", 0)),
        "powered_task_floor": int(artifact.get("powered_task_floor", 160)),
        "oracle_headroom_present": bool(artifact.get("oracle_headroom_present", False)),
        "oracle_passrate": float(artifact.get("oracle_passrate", 0.0)),
        "demofit_delta_pp": float(artifact.get("demofit_delta_pp", 0.0)),
        "demofit_bootstrap_ci95": list(artifact.get("demofit_bootstrap_ci95", [0.0, 0.0])),
        "best_arm": str(artifact.get("best_arm", "")),
        "best_arm_delta_pp": float(artifact.get("best_arm_delta_pp", 0.0)),
        "best_arm_ci95": list(artifact.get("best_arm_ci95", [0.0, 0.0])),
        "missing_verifier_gaps": [str(gap) for gap in artifact.get("missing_verifier_gaps", [])],
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
    }


def classify_g3_decentralization_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify Exp 4059 accumulated MoE-base coverage as latent, absent, accumulating, or pending."""
    rel_path = G3_DECENTRALIZATION_PATH
    path = repo_root / rel_path
    if not path.exists():
        return {
            "g3_decentralization_outcome_recorded": "g3_decentralization_moe_base_4048_pending",
            "status": "pending",
            "reason": "missing_exp4059_artifact",
            "artifact_path": rel_path,
            "accumulated_coverage": 0.0,
            "coverage_delta_vs_12b": 0.0,
            "bootstrap_ci95": [0.0, 0.0],
            "accumulated_n": 0,
            "n_tasks_scored": 0,
            "local_support_diagnosis": "pending",
            "raw_complete": False,
            "missing_verifier_gaps": [],
        }

    artifact = base._load_json(path)
    common = _g3_common(rel_path, artifact)
    diagnosis = str(artifact.get("local_support_diagnosis", "uninformative"))
    complete = bool(artifact.get("raw_complete", False))
    if complete and diagnosis in {"latent", "absent"}:
        return {
            "g3_decentralization_outcome_recorded": (
                f"g3_decentralization_{diagnosis}_coverage_{_fmt(common['accumulated_coverage'])}"
            ),
            "status": diagnosis,
            **common,
        }
    return {
        "g3_decentralization_outcome_recorded": (
            f"g3_decentralization_accumulating_coverage_{_fmt(common['accumulated_coverage'])}"
        ),
        "status": "accumulating",
        **common,
    }


def _g3_common(artifact_path: str, artifact: dict[str, Any]) -> dict[str, Any]:
    coverage = float(artifact.get("moe_base_demo_perfect_coverage", 0.0))
    return {
        "artifact_path": artifact_path,
        "accumulated_coverage": round(coverage, 4),
        "coverage_delta_vs_12b": float(artifact.get("coverage_delta_vs_12b", 0.0)),
        "bootstrap_ci95": list(artifact.get("bootstrap_ci95", [0.0, 0.0])),
        "accumulated_n": int(artifact.get("ACCUMULATED-N", artifact.get("n_tasks_scored", 0))),
        "n_tasks_scored": int(artifact.get("n_tasks_scored", 0)),
        "local_support_diagnosis": str(artifact.get("local_support_diagnosis", "uninformative")),
        "raw_complete": bool(artifact.get("raw_complete", False)),
        "missing_verifier_gaps": [str(gap) for gap in artifact.get("missing_verifier_gaps", [])],
    }


def classify_g2_vc33_ceiling(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify the banked Exp 4046 vc33 closed-loop result as a sim2real ceiling record."""
    rel_path = G2_CLOSED_LOOP_PATH
    path = repo_root / rel_path
    if not path.exists():
        return {
            "g2_vc33_ceiling_logged": False,
            "g2_vc33_ceiling_outcome_recorded": "g2_vc33_ceiling_pending",
            "status": "pending",
            "reason": "missing_exp4046_artifact",
            "artifact_path": rel_path,
        }

    artifact = base._load_json(path)
    common = base._g2_common(rel_path, artifact)
    if bool(artifact.get("closed_loop_broke_wall")):
        return {
            "g2_vc33_ceiling_logged": False,
            "g2_vc33_ceiling_outcome_recorded": "g2_vc33_closed_loop_capability_registered",
            "status": "capability_registered",
            **common,
        }
    return {
        "g2_vc33_ceiling_logged": True,
        "g2_vc33_ceiling_outcome_recorded": "g2_vc33_sim2real_ceiling_logged",
        "status": "gap_logged",
        **common,
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    g1_outcome: dict[str, Any],
    g3_outcome: dict[str, Any],
    g2_outcome: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, bool]]:
    """Return registry and gaps text with Exp 4063 outcomes represented idempotently."""
    updated_registry = deepcopy(registry)
    updated_gaps = gaps_text
    registry_changed = _ensure_gap4_eval(updated_registry, offline_replay)

    if g1_outcome["g1_off_arc_outcome_recorded"] == "g1_evalplus_demofit_ci_excludes_zero":
        registry_changed = (
            base._upsert_verifier(updated_registry, _g1_demofit_entry(g1_outcome))
            or registry_changed
        )
    else:
        updated_gaps = base._replace_marked_block(
            updated_gaps, "exp4063-g1", _g1_gap_block(g1_outcome)
        )

    updated_gaps = base._replace_marked_block(
        updated_gaps, "exp4063-g3", _g3_gap_block(g3_outcome)
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps, "exp4063-g2", _g2_gap_block(g2_outcome)
    )

    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": _gaps_contain_outcomes(updated_gaps, g1_outcome, g3_outcome, g2_outcome),
        },
    )


def _ensure_gap4_eval(registry: dict[str, Any], offline_replay: dict[str, Any]) -> bool:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    eval_block = entry.setdefault("eval", {})
    required = {
        "eval_exp_4063": EXP4063_ARTIFACT_PATH,
        "arc1_rule_exec_vote_pass2": offline_replay.get("arc1_rule_exec", {}).get("vote_pass2"),
        "arc1_rule_exec_gated_pass2": offline_replay.get("arc1_rule_exec", {}).get("gated_pass2"),
        "arc2_rule_exec_vote_pass2": offline_replay.get("arc2_rule_exec", {}).get("vote_pass2"),
        "arc2_rule_exec_gated_pass2": offline_replay.get("arc2_rule_exec", {}).get("gated_pass2"),
        "exp4063_offline_reeval_bitexact": bool(offline_replay.get("offline_reeval_bitexact")),
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
        "code_path": "scripts/experiments/exp4057_offarc_power_evalplus_collect.py",
        "weights_hf": None,
        "weights_cid": None,
        "weights_sha256": None,
        "training_data_ref": outcome["artifact_path"],
        "label_source": "evalplus_hidden_tests",
        "eval": {
            "metric": "evalplus_hidden_pass_at_1_delta_pp",
            "delta_pp": outcome["demofit_delta_pp"],
            "bootstrap_ci95": outcome["demofit_bootstrap_ci95"],
            "n": outcome["accumulated_n_tasks"],
            "eval_artifact": outcome["artifact_path"],
        },
        "lineage": {
            "from": GAP4_VERIFIER_ID,
            "change": "Exp 4063 recorded Exp 4057 EvalPlus demo-fit CI excluding zero.",
        },
        "status": "candidate",
        "notes": "G1 EvalPlus off-ARC outcome recorded by Exp 4063.",
    }


def _g1_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"#### Exp 4063 G1 EvalPlus update for {CODE_GAP_ID}\n"
        f"- status: {outcome['g1_off_arc_outcome_recorded']}\n"
        f"- evidence: `{outcome.get('artifact_path', G1_EVALPLUS_PATH)}`; "
        f"accumulated_n_tasks={outcome.get('accumulated_n_tasks', 0)}; "
        f"powered_task_floor={outcome.get('powered_task_floor', 160)}; "
        f"oracle_headroom_present={outcome.get('oracle_headroom_present')}; "
        f"demo_fit_CI95={outcome.get('demofit_bootstrap_ci95')}; "
        f"best_arm={outcome.get('best_arm')}; best_arm_CI95={outcome.get('best_arm_ci95')}.\n"
        "- failure mode: visible/demo-fit code tests are not yet a powered EvalPlus hidden-semantic discriminator.\n"
        "- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics.\n"
        "- candidate design: continue accumulation or add hidden-property, symbolic, formal, or runtime oracles.\n"
        "- priority: high\n"
    )


def _g3_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"#### Exp 4063 G3 accumulated update for {DECENTRALIZATION_GAP_ID}\n"
        f"- status: {outcome['g3_decentralization_outcome_recorded']}\n"
        f"- evidence: `{outcome.get('artifact_path', G3_DECENTRALIZATION_PATH)}`; "
        f"accumulated_coverage={outcome.get('accumulated_coverage')}; "
        f"n_tasks_scored={outcome.get('n_tasks_scored', 0)}; "
        f"ACCUMULATED-N={outcome.get('accumulated_n', 0)}; "
        f"diagnosis={outcome.get('local_support_diagnosis', outcome.get('status'))}; "
        f"bootstrap_CI95={outcome.get('bootstrap_ci95')}.\n"
        "- failure mode: local MoE best-of-N support has not established a sovereign GAP-4 replacement.\n"
        "- missing discriminator: verifier-side signal or stronger local base that surfaces demo-perfect programs without Codex.\n"
        "- candidate design: continue accumulation, use a stronger local base, or add verifier-guided generation before distillation.\n"
        "- priority: high\n"
    )


def _g2_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {SIM2REAL_GAP_ID}: vc33 verified-WM closed-loop sim2real ceiling\n"
        f"- status: {outcome['g2_vc33_ceiling_outcome_recorded']}\n"
        f"- evidence: `{outcome.get('artifact_path', G2_CLOSED_LOOP_PATH)}`; "
        f"per_step_wm_real_divergence_rate="
        f"{outcome.get('per_step_wm_real_divergence_rate')}; "
        f"divergence_gate_fired_count={outcome.get('divergence_gate_fired_count')}.\n"
        "- failure mode: bounded WM search produced a plan whose predicted next state diverged from the real environment.\n"
        "- missing discriminator: per-step WM-to-real transition trust signal strong enough to plan past vc33's wall.\n"
        "- candidate design: improve the verified world model or add a conservative real-env grounding/replan guard.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return gap4 is not None and gap4.get("eval", {}).get("eval_exp_4063") == EXP4063_ARTIFACT_PATH


def _gaps_contain_outcomes(
    gaps_text: str,
    g1_outcome: dict[str, Any],
    g3_outcome: dict[str, Any],
    g2_outcome: dict[str, Any],
) -> bool:
    return (
        CODE_GAP_ID in gaps_text
        and g1_outcome["g1_off_arc_outcome_recorded"] in gaps_text
        and DECENTRALIZATION_GAP_ID in gaps_text
        and g3_outcome["g3_decentralization_outcome_recorded"] in gaps_text
        and SIM2REAL_GAP_ID in gaps_text
        and g2_outcome["g2_vc33_ceiling_outcome_recorded"] in gaps_text
    )


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    g1_outcome: dict[str, Any],
    g3_outcome: dict[str, Any],
    g2_outcome: dict[str, Any],
    registry_updated: bool,
    gaps_updated: bool,
    duration_s: float,
) -> dict[str, Any]:
    bit_label = "bitexact" if offline_replay.get("offline_reeval_bitexact") else "mismatch"
    g1_label = g1_outcome["g1_off_arc_outcome_recorded"]
    g3_label = g3_outcome["g3_decentralization_outcome_recorded"]
    g2_label = "ceiling_recorded" if g2_outcome.get("g2_vc33_ceiling_logged") else "ceiling_pending"
    artifact = {
        "experiment": "experiment_4063_verifier_registry_and_gaps_hygiene",
        "schema": "carnot.experiment_4063_verifier_registry_and_gaps_hygiene.v1",
        "honest_verdict": f"complete: gap4_reeval_{bit_label}_g1_{g1_label}_g3_{g3_label}_g2_{g2_label}",
        "offline_reeval_bitexact": bool(offline_replay.get("offline_reeval_bitexact")),
        "g1_off_arc_outcome_recorded": g1_label,
        "g3_decentralization_outcome_recorded": g3_label,
        "g2_vc33_ceiling_logged": bool(g2_outcome.get("g2_vc33_ceiling_logged")),
        "registry_updated": bool(registry_updated),
        "gaps_updated": bool(gaps_updated),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 3),
        "offline_replay": offline_replay,
        "g1_off_arc_outcome": g1_outcome,
        "g3_decentralization_outcome": g3_outcome,
        "g2_vc33_ceiling_outcome": g2_outcome,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_RULE_EXEC_PATH,
            ARC2_RULE_EXEC_PATH,
            ARC2_CHAIN_PATH,
            G1_EVALPLUS_PATH,
            G3_DECENTRALIZATION_PATH,
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
    for field in (
        "offline_reeval_bitexact",
        "g2_vc33_ceiling_logged",
        "registry_updated",
        "gaps_updated",
    ):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")  # pragma: no cover
    for field in ("g1_off_arc_outcome_recorded", "g3_decentralization_outcome_recorded"):
        if not isinstance(artifact[field], str):
            raise ValueError(f"{field} must be a string")  # pragma: no cover
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")  # pragma: no cover


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4063 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH

    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_headlines(repo_root)
    g1_outcome = classify_g1_evalplus_outcome(repo_root)
    g3_outcome = classify_g3_decentralization_outcome(repo_root)
    g2_outcome = classify_g2_vc33_ceiling(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        g1_outcome,
        g3_outcome,
        g2_outcome,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        g1_outcome=g1_outcome,
        g3_outcome=g3_outcome,
        g2_outcome=g2_outcome,
        registry_updated=ledger_summary["registry_updated"],
        gaps_updated=ledger_summary["gaps_updated"],
        duration_s=time.time() - started,
    )
    base._write_json(repo_root / EXP4063_ARTIFACT_PATH, artifact)
    return artifact


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def main() -> None:  # pragma: no cover - exercised by the experiment command
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4063_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
