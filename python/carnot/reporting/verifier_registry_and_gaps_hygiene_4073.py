"""Exp 4073 registry and gaps hygiene for the GAP-4 verifier line.

Spec refs: REQ-VERIFY-4073, SCENARIO-VERIFY-4073.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_and_gaps_hygiene_4063 as exp4063


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP4073_ARTIFACT_PATH = "results/experiment_4073_verifier_registry_and_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_RULE_EXEC_PATH = base.ARC1_RULE_EXEC_PATH
ARC2_RULE_EXEC_PATH = base.ARC2_RULE_EXEC_PATH
ARC2_CHAIN_PATH = base.ARC2_CHAIN_PATH
G1_CORPUS_ROUTED_PATH = "results/experiment_4068_offarc_transfer_power_sync.json"
G3_DECENTRALIZATION_PATH = "results/experiment_4069_decentralization_moe_sync.json"
G2_CLOSED_LOOP_PATH = base.G2_CLOSED_LOOP_PATH

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
CODE_GAP_ID = base.CODE_GAP_ID
DECENTRALIZATION_GAP_ID = "GAP-DECENTRALIZATION-MOE-BASE-4048"
SIM2REAL_GAP_ID = base.SIM2REAL_GAP_ID
G1_DEMOFIT_VERIFIER_ID = "gap4_code_corpus_routed_demo_fit_transfer_4068"

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
    "honest_verdict": "Terminal-prefix summary of GAP-4 replay and .376 G1/G3/G2 ledger routing.",
    "offline_reeval_bitexact": (
        "A reusable verifier that does not reproduce its headline numbers is not the same verifier."
    ),
    "g1_off_arc_outcome_recorded": (
        "The corpus-routed off-ARC power measurement must land in the ledger, not just an artifact."
    ),
    "g3_decentralization_outcome_recorded": (
        "The Exp 4069 MoE-base accumulated coverage result must be recorded as latent, absent, "
        "accumulating, or pending."
    ),
    "g2_vc33_ceiling_logged": "Bare bool; the banked vc33 sim2real ceiling is visible in the gaps ledger.",
    "registry_updated": "Bare bool; registry reflects the replayed GAP-4 line after Exp 4073.",
    "gaps_updated": "Bare bool; gaps ledger reflects the missing-verifier state after Exp 4073.",
    "inference_substrate": "aggregation_from_upstream_artifacts; no Codex, GGUF, or live inference.",
}


def replay_gap4_headlines(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Replay GAP-4 ARC headline numbers from cached artifacts only."""
    return base.replay_gap4_headlines(repo_root)


def classify_g1_corpus_routed_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify Exp 4068 into corpus-routed transfer, stronger closure, open, or accumulating."""
    rel_path = G1_CORPUS_ROUTED_PATH
    path = repo_root / rel_path
    if not path.exists():
        return _pending_g1("missing_exp4068_artifact", rel_path)
    return classify_g1_corpus_routed_outcome_fixture(base._load_json(path), rel_path)


def classify_g1_corpus_routed_outcome_fixture(
    artifact: dict[str, Any],
    artifact_path: str = G1_CORPUS_ROUTED_PATH,
) -> dict[str, Any]:
    """Classify an already-loaded Exp 4068-shaped artifact."""
    common = _g1_common(artifact_path, artifact)
    label = common["corpus"]
    accumulated_n = common["accumulated_n_tasks"]
    powered_floor = common["powered_task_floor"]
    verdict = str(artifact.get("honest_verdict", ""))
    if (
        accumulated_n < powered_floor
        or verdict.startswith("blocked_")
        or "partial" in verdict
        or artifact.get("raw_artifact_present") is False
        or not common["oracle_headroom_present"]
    ):
        return {
            "g1_off_arc_outcome_recorded": f"g1_{label}_accumulating",
            "status": "accumulating",
            **common,
        }

    if bool(artifact.get("demofit_ci_excludes_zero")):
        return {
            "g1_off_arc_outcome_recorded": f"g1_{label}_demofit_ci_excludes_zero",
            "status": "transfer",
            **common,
        }

    if bool(artifact.get("best_arm_ci_excludes_zero")):
        best_arm = common["best_arm"] or "stronger_arm"
        return {
            "g1_off_arc_outcome_recorded": f"g1_{label}_stronger_{best_arm}_ci_excludes_zero",
            "status": "stronger_discriminator_registered",
            **common,
        }

    return {
        "g1_off_arc_outcome_recorded": f"g1_{label}_all_arms_touch_zero_gap_open",
        "status": "gap_open",
        **common,
    }


def _pending_g1(reason: str, artifact_path: str) -> dict[str, Any]:
    return {
        "g1_off_arc_outcome_recorded": "g1_corpus_routed_pending",
        "status": "pending",
        "reason": reason,
        "artifact_path": artifact_path,
    }


def _g1_common(artifact_path: str, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": artifact_path,
        "accumulated_n_tasks": int(artifact.get("accumulated_n_tasks", 0)),
        "powered_task_floor": int(artifact.get("powered_task_floor", 160)),
        "corpus": _corpus_label(str(artifact.get("corpus", "corpus_routed"))),
        "evaluation_corpus": str(artifact.get("evaluation_corpus", "")),
        "corpus_routed_reason": str(artifact.get("corpus_routed_reason", "")),
        "oracle_headroom_present": bool(artifact.get("oracle_headroom_present", False)),
        "oracle_passrate": float(artifact.get("oracle_passrate", 0.0)),
        "armA_vote_passrate": float(artifact.get("armA_vote_passrate", 0.0)),
        "armB_demofit_passrate": float(artifact.get("armB_demofit_passrate", 0.0)),
        "armApp_aces_passrate": float(artifact.get("armApp_aces_passrate", 0.0)),
        "armC_symbolic_passrate": float(artifact.get("armC_symbolic_passrate", 0.0)),
        "demofit_delta_pp": float(artifact.get("demofit_delta_pp", 0.0)),
        "demofit_bootstrap_ci95": list(artifact.get("demofit_bootstrap_ci95", [0.0, 0.0])),
        "best_arm": str(artifact.get("best_arm", "")),
        "best_arm_delta_pp": float(artifact.get("best_arm_delta_pp", 0.0)),
        "best_arm_ci95": list(artifact.get("best_arm_ci95", [0.0, 0.0])),
        "missing_verifier_gaps": [str(gap) for gap in artifact.get("missing_verifier_gaps", [])],
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
    }


def classify_g3_decentralization_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify Exp 4069 accumulated MoE-base coverage as latent, absent, accumulating, or pending."""
    rel_path = G3_DECENTRALIZATION_PATH
    path = repo_root / rel_path
    if not path.exists():
        return {
            "g3_decentralization_outcome_recorded": "g3_decentralization_moe_sync_4069_pending",
            "status": "pending",
            "reason": "missing_exp4069_artifact",
            "artifact_path": rel_path,
            "accumulated_coverage": 0.0,
            "coverage_delta_vs_12b": 0.0,
            "bootstrap_ci95": [0.0, 0.0],
            "accumulated_n_tasks": 0,
            "n_demo_perfect_tasks": 0,
            "local_support_diagnosis": "pending",
            "raw_complete": False,
            "missing_verifier_gaps": [],
        }
    return classify_g3_decentralization_outcome_fixture(base._load_json(path), rel_path)


def classify_g3_decentralization_outcome_fixture(
    artifact: dict[str, Any],
    artifact_path: str = G3_DECENTRALIZATION_PATH,
) -> dict[str, Any]:
    """Classify an already-loaded Exp 4069-shaped artifact."""
    common = _g3_common(artifact_path, artifact)
    diagnosis = common["local_support_diagnosis"]
    if common["raw_complete"] and diagnosis in {"latent", "absent"}:
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
    accumulated_n = int(
        artifact.get("accumulated_n_tasks", artifact.get("ACCUMULATED-N", artifact.get("n_tasks_scored", 0)))
    )
    target_n = int(artifact.get("target_n_tasks", accumulated_n))
    verdict = str(artifact.get("honest_verdict", ""))
    raw_complete = bool(artifact.get("raw_complete", artifact.get("summarize_artifact", True)))
    complete = raw_complete and verdict.startswith("complete:") and accumulated_n >= target_n
    return {
        "artifact_path": artifact_path,
        "accumulated_coverage": round(float(artifact.get("moe_base_demo_perfect_coverage", 0.0)), 4),
        "coverage_delta_vs_12b": float(artifact.get("coverage_delta_vs_12b", 0.0)),
        "bootstrap_ci95": list(artifact.get("bootstrap_ci95", [0.0, 0.0])),
        "accumulated_n_tasks": accumulated_n,
        "target_n_tasks": target_n,
        "new_tasks_processed": int(artifact.get("new_tasks_processed", 0)),
        "n_demo_perfect_tasks": int(artifact.get("n_demo_perfect_tasks", 0)),
        "local_support_diagnosis": str(artifact.get("local_support_diagnosis", "uninformative")),
        "raw_complete": complete,
        "oracle_coverage": float(artifact.get("oracle_coverage", 0.0)),
        "missing_verifier_gaps": [str(gap) for gap in artifact.get("missing_verifier_gaps", [])],
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
    }


def classify_g2_vc33_ceiling(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify the banked Exp 4046 vc33 closed-loop result as a sim2real ceiling record."""
    return exp4063.classify_g2_vc33_ceiling(repo_root)


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    g1_outcome: dict[str, Any],
    g3_outcome: dict[str, Any],
    g2_outcome: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, bool]]:
    """Return registry and gaps text with Exp 4073 outcomes represented idempotently."""
    updated_registry = deepcopy(registry)
    updated_gaps = gaps_text
    registry_changed = _ensure_gap4_eval(updated_registry, offline_replay)

    if g1_outcome["status"] == "transfer":
        registry_changed = (
            base._upsert_verifier(updated_registry, _g1_demofit_entry(g1_outcome))
            or registry_changed
        )
    elif g1_outcome["status"] == "stronger_discriminator_registered":
        registry_changed = (
            base._upsert_verifier(updated_registry, _g1_stronger_entry(g1_outcome))
            or registry_changed
        )
    else:
        updated_gaps = base._replace_marked_block(
            updated_gaps, "exp4073-g1", _g1_gap_block(g1_outcome)
        )

    updated_gaps = base._replace_marked_block(
        updated_gaps, "exp4073-g3", _g3_gap_block(g3_outcome)
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps, "exp4073-g2", _g2_gap_block(g2_outcome)
    )

    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry, g1_outcome)
            or registry_changed,
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
        "eval_exp_4073": EXP4073_ARTIFACT_PATH,
        "arc1_rule_exec_vote_pass2": offline_replay.get("arc1_rule_exec", {}).get("vote_pass2"),
        "arc1_rule_exec_gated_pass2": offline_replay.get("arc1_rule_exec", {}).get("gated_pass2"),
        "arc2_rule_exec_vote_pass2": offline_replay.get("arc2_rule_exec", {}).get("vote_pass2"),
        "arc2_rule_exec_gated_pass2": offline_replay.get("arc2_rule_exec", {}).get("gated_pass2"),
        "exp4073_offline_reeval_bitexact": bool(offline_replay.get("offline_reeval_bitexact")),
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
        "code_path": "scripts/experiments/exp4068_offarc_transfer_power_sync.py",
        "weights_hf": None,
        "weights_cid": None,
        "weights_sha256": None,
        "training_data_ref": outcome["artifact_path"],
        "label_source": f"{outcome['evaluation_corpus']}_hidden_tests",
        "eval": {
            "metric": "corpus_routed_hidden_pass_at_1_delta_pp",
            "delta_pp": outcome["demofit_delta_pp"],
            "bootstrap_ci95": outcome["demofit_bootstrap_ci95"],
            "n": outcome["accumulated_n_tasks"],
            "eval_artifact": outcome["artifact_path"],
        },
        "lineage": {
            "from": GAP4_VERIFIER_ID,
            "change": "Exp 4073 recorded Exp 4068 corpus-routed demo-fit CI excluding zero.",
        },
        "status": "candidate",
        "notes": "G1 corpus-routed off-ARC outcome recorded by Exp 4073.",
    }


def _g1_stronger_entry(outcome: dict[str, Any]) -> dict[str, Any]:
    best_arm = outcome["best_arm"] or "stronger_arm"
    return {
        "verifier_id": f"gap4_code_{best_arm}_transfer_4068",
        "domain": "code",
        "version": 1,
        "kind": "process_verifier",
        "code_commit": "HEAD",
        "code_path": "scripts/experiments/exp4068_offarc_transfer_power_sync.py",
        "weights_hf": None,
        "weights_cid": None,
        "weights_sha256": None,
        "training_data_ref": outcome["artifact_path"],
        "label_source": f"{outcome['evaluation_corpus']}_hidden_tests",
        "eval": {
            "metric": "corpus_routed_hidden_pass_at_1_delta_pp",
            "delta_pp": outcome["best_arm_delta_pp"],
            "bootstrap_ci95": outcome["best_arm_ci95"],
            "n": outcome["accumulated_n_tasks"],
            "eval_artifact": outcome["artifact_path"],
        },
        "lineage": {
            "from": GAP4_VERIFIER_ID,
            "change": f"Exp 4073 recorded stronger code-domain discriminator {best_arm}.",
        },
        "status": "candidate",
        "notes": "G1 stronger-arm corpus-routed outcome recorded by Exp 4073.",
    }


def _g1_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"#### Exp 4073 G1 corpus-routed update for {CODE_GAP_ID}\n"
        f"- status: {outcome['g1_off_arc_outcome_recorded']}\n"
        f"- evidence: `{outcome.get('artifact_path', G1_CORPUS_ROUTED_PATH)}`; "
        f"corpus={outcome.get('corpus')}; "
        f"accumulated_n_tasks={outcome.get('accumulated_n_tasks', 0)}; "
        f"powered_task_floor={outcome.get('powered_task_floor', 160)}; "
        f"oracle_headroom_present={outcome.get('oracle_headroom_present')}; "
        f"oracle_passrate={outcome.get('oracle_passrate')}; "
        f"demo_fit_CI95={outcome.get('demofit_bootstrap_ci95')}; "
        f"best_arm={outcome.get('best_arm')}; best_arm_CI95={outcome.get('best_arm_ci95')}; "
        f"route={outcome.get('corpus_routed_reason')}.\n"
        "- failure mode: visible/demo-fit code tests are not yet a powered hidden-semantic discriminator.\n"
        "- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics.\n"
        "- candidate design: continue accumulation on a corpus with oracle headroom or add hidden-property, symbolic, formal, or runtime oracles.\n"
        "- priority: high\n"
    )


def _g3_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"#### Exp 4073 G3 synchronous update for {DECENTRALIZATION_GAP_ID}\n"
        f"- status: {outcome['g3_decentralization_outcome_recorded']}\n"
        f"- evidence: `{outcome.get('artifact_path', G3_DECENTRALIZATION_PATH)}`; "
        f"accumulated_coverage={outcome.get('accumulated_coverage')}; "
        f"ACCUMULATED-N={outcome.get('accumulated_n_tasks', 0)}; "
        f"n_demo_perfect_tasks={outcome.get('n_demo_perfect_tasks', 0)}; "
        f"diagnosis={outcome.get('local_support_diagnosis', outcome.get('status'))}; "
        f"bootstrap_CI95={outcome.get('bootstrap_ci95')}; "
        f"missing_verifier_gaps={outcome.get('missing_verifier_gaps', [])}.\n"
        "- failure mode: local MoE best-of-N support has not established a sovereign GAP-4 replacement.\n"
        "- missing discriminator: verifier-side signal or stronger local base that surfaces demo-perfect programs without Codex.\n"
        "- candidate design: continue accumulation, use a stronger local base, or add verifier-guided generation before distillation.\n"
        "- priority: high\n"
    )


def _g2_gap_block(outcome: dict[str, Any]) -> str:
    return exp4063._g2_gap_block(outcome)


def _registry_contains_outcomes(registry: dict[str, Any], g1_outcome: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if gap4 is None:
        return False
    if gap4.get("eval", {}).get("eval_exp_4073") != EXP4073_ARTIFACT_PATH:
        return False
    if g1_outcome["status"] == "transfer":
        return base._find_verifier(registry, G1_DEMOFIT_VERIFIER_ID) is not None
    if g1_outcome["status"] == "stronger_discriminator_registered":
        return base._find_verifier(
            registry,
            f"gap4_code_{g1_outcome.get('best_arm') or 'stronger_arm'}_transfer_4068",
        ) is not None
    return True


def _gaps_contain_outcomes(
    gaps_text: str,
    g1_outcome: dict[str, Any],
    g3_outcome: dict[str, Any],
    g2_outcome: dict[str, Any],
) -> bool:
    needs_g1_gap = g1_outcome["status"] not in {"transfer", "stronger_discriminator_registered"}
    g1_ok = (not needs_g1_gap) or (
        CODE_GAP_ID in gaps_text and g1_outcome["g1_off_arc_outcome_recorded"] in gaps_text
    )
    return (
        g1_ok
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
        "experiment": "experiment_4073_verifier_registry_and_gaps_hygiene",
        "schema": "carnot.experiment_4073_verifier_registry_and_gaps_hygiene.v1",
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
            G1_CORPUS_ROUTED_PATH,
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
    """Run Exp 4073 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH

    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_headlines(repo_root)
    g1_outcome = classify_g1_corpus_routed_outcome(repo_root)
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
    base._write_json(repo_root / EXP4073_ARTIFACT_PATH, artifact)
    return artifact


def _corpus_label(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "corpus_routed"


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def main() -> None:  # pragma: no cover - exercised by the experiment command
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4073_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
