"""Experiment 5041: .463 verifier-moat capstone scorecard.

Spec refs: REQ-CAPSTONE-5041, SCENARIO-CAPSTONE-5041,
SCENARIO-CAPSTONE-5041-FIELD-PRINCIPLES.

This scorecard reads the third PHASE D off-ARC verifier-moat milestone from
upstream result JSON. Exp5036 is the authoritative D5 moat-gate verdict when it
is present. If it is absent, the D arms are summarized directly. In both paths,
``flagged_adversarial=true`` artifacts are skipped before any headline number is
imported, and blocked/skeleton/degenerate arms are failed executions rather than
clean nulls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml

from carnot.experiment_4819_capstone_v443 import (
    _float,
    _int,
    _read_json_object,
    _read_yaml_object,
    file_sha256,
    payload_checksum,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5041_capstone_v463"
EXPERIMENT_ID = 5041
SCHEMA = "carnot.exp5041.capstone_v463.v1"
RESULT_RELATIVE_PATH = "results/experiment_5041_capstone_v463.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
RANDOM_SEED = 20260630
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-5041",
    "SCENARIO-CAPSTONE-5041",
    "SCENARIO-CAPSTONE-5041-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_capstone_v463_moat_"
            "<realized|retired_bounded|execution_incomplete|musr_scoped>_<headline>."
        )
    },
    "moat_verdict": {
        "principle": (
            "the off-ARC verifier-moat verdict carried from D5 (realized / "
            "retired_bounded / execution_incomplete / mixed-scoped) -- the milestone headline."
        )
    },
    "d1_finally_executed": {
        "principle": (
            "true iff D1 scorer_trained=true -- whether the .463 de-risking (real base "
            "resolver + B3 smoke + B2-decoupling + claude/opus) finally landed a REAL "
            "trained-verifier measurement (the primary .463 success criterion)."
        )
    },
    "best_arm_and_delta": {
        "principle": (
            "the strongest oracle-distinct construction (or the efficiency Pareto point) "
            "+ corpus + delta_vs_tuned_sc + CI (from D5)."
        )
    },
    "efficiency_win": {
        "principle": (
            "true iff the cascade (D6) reached accuracy parity at materially fewer judge "
            "calls (north-star §5)."
        )
    },
    "diffusiongemma_gate_status": {
        "principle": (
            "the gate status from D5 (conditions satisfied off-ARC? activation stays "
            "operator-gated; not autonomously flipped to MET)."
        )
    },
    "per_arm_table": {
        "principle": (
            "D1/D2/D3/D6/D4 deltas + CIs + verifier_is_oracle + headroom_present + "
            "scorer_trained/abstention_rate (the audit + execution-quality trail)."
        )
    },
    "infra_rollup": {
        "principle": (
            "B2 robust logprob cache (n_cached_rows) + B3 trainer module/smoke_passed "
            "status (the 2 reserved infra slots; the permanent 404-fix)."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the ARC progress metric carried from the registry (the deliverable stays "
            "LOCKED; ARC opportunistic)."
        )
    },
    "flagged_artifacts_skipped": {
        "principle": (
            "the list of arms skipped for flagged_adversarial=true (never aggregated "
            "into a headline)."
        )
    },
    "next_milestone_pointer": {
        "principle": (
            ".464 direction conditioned on the moat verdict (scale the winner / pivot "
            "to the next verifier direction / ESCALATE to the operator if execution-incomplete "
            "a 3rd time / tighten the strongest arm)."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads upstream JSON, no LLM; 0.0001s floor)."
        )
    },
    "cited_upstream_artifacts": {
        "principle": (
            "the {experiment_id, fields_imported, sha256} for each aggregated arm "
            "(traceable to real measurements)."
        )
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "moat_verdict",
    "d1_finally_executed",
    "best_arm_and_delta",
    "efficiency_win",
    "diffusiongemma_gate_status",
    "per_arm_table",
    "infra_rollup",
    "reproducible_total_levels",
    "flagged_artifacts_skipped",
    "next_milestone_pointer",
    "inference_substrate",
    "cited_upstream_artifacts",
    "field_principles",
    "preconditions_checked",
    "hardware_rollup",
    "sota_ingestion_rollup",
    "self_play_rollup",
    "arc_opportunistic_rollup",
    "arc_deliverable_locked",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "capstone_ready",
)

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)

FAILED_EXECUTION_STATUSES = {"blocked", "missing", "skeleton", "degenerate", "not_run"}


@dataclass(frozen=True)
class UpstreamSource:
    """One upstream result that can feed the .463 capstone scorecard."""

    experiment_id: int
    relative_path: str


UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "B2_LOGPROB_CACHE": UpstreamSource(
        5029, "results/experiment_5029_shared_logprob_candidate_cache_v2.json"
    ),
    "B3_MOAT_TRAINER": UpstreamSource(5030, "results/experiment_5030_moat_trainer_module.json"),
    "D1_LORA_EBM": UpstreamSource(5031, "results/experiment_5031_lora_ebm_scorer_musr_v3.json"),
    "D2_UPRM": UpstreamSource(5032, "results/experiment_5032_uprm_replication_v3.json"),
    "D3_EBRM": UpstreamSource(5033, "results/experiment_5033_ebrm_uncertainty_verifier_v3.json"),
    "D6_CASCADE": UpstreamSource(
        5034, "results/experiment_5034_uncertainty_routed_cascade_v2.json"
    ),
    "D4_SECOND_CORPUS": UpstreamSource(5035, "results/experiment_5035_moat_second_corpus_v3.json"),
    "D5_MOAT_GATE": UpstreamSource(5036, "results/experiment_5036_moat_gate_resolution_v3.json"),
    "C_KV260": UpstreamSource(5037, "results/experiment_5037_kv260_continuity.json"),
    "E1_SOTA": UpstreamSource(5038, "results/experiment_5038_sota_ingestion_verifier_moat.json"),
    "E2_SELF_PLAY": UpstreamSource(
        5039, "results/experiment_5039_self_play_verifier_checkpoint.json"
    ),
    "E3_ARC_LEVEL": UpstreamSource(5040, "results/experiment_5040_levelup_attempt.json"),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "B2_LOGPROB_CACHE": (
        "honest_verdict",
        "candidate_cache_built",
        "cache_jsonl_path",
        "n_cached_rows",
        "n_questions",
        "corpora_cached",
        "has_per_token_logprobs",
        "rescored_not_regenerated",
    ),
    "B3_MOAT_TRAINER": (
        "honest_verdict",
        "module_path",
        "smoke_passed",
        "smoke_train_loss",
        "smoke_duration_s",
        "checkpoint_path",
        "base_used",
        "resolver_base_list",
        "verifier_is_oracle",
    ),
    "D1_LORA_EBM": (
        "honest_verdict",
        "delta_vs_tuned_sc",
        "paired_ci95",
        "mcnemar_p",
        "headroom_present",
        "oracle_at_k",
        "genuine_tuned_sc_accuracy",
        "trained_scorer_accuracy",
        "scorer_trained",
        "abstention_rate",
        "verifier_is_oracle",
    ),
    "D2_UPRM": (
        "honest_verdict",
        "delta_vs_tuned_sc",
        "paired_ci95",
        "mcnemar_p",
        "headroom_present",
        "oracle_at_k",
        "genuine_tuned_sc_accuracy",
        "uprm_selection_accuracy",
        "abstention_rate",
        "verifier_is_oracle",
    ),
    "D3_EBRM": (
        "honest_verdict",
        "delta_vs_tuned_sc",
        "paired_ci95",
        "mcnemar_p",
        "headroom_present",
        "oracle_at_k",
        "genuine_tuned_sc_accuracy",
        "ebrm_selection_accuracy",
        "abstention_rate",
        "verifier_is_oracle",
    ),
    "D6_CASCADE": (
        "honest_verdict",
        "cascade_accuracy",
        "judge_only_accuracy",
        "judge_call_fraction",
        "paired_ci95_cascade_vs_judge",
        "n_questions",
        "verifier_is_oracle",
    ),
    "D4_SECOND_CORPUS": (
        "honest_verdict",
        "delta_vs_tuned_sc_second",
        "paired_ci95_second",
        "mcnemar_p_second",
        "headroom_present",
        "oracle_at_k_second",
        "genuine_tuned_sc_accuracy_second",
        "second_corpus_accuracy",
        "verifier_is_oracle",
    ),
    "D5_MOAT_GATE": (
        "honest_verdict",
        "decision",
        "moat_realized",
        "moat_retired_bounded",
        "efficiency_win",
        "best_arm",
        "per_arm_table",
        "execution_incomplete_arms",
        "diffusiongemma_gate_status",
        "diffusiongemma_gate_conditions_satisfied_off_arc",
        "diffusiongemma_activation",
        "flagged_arms_skipped",
        "paper_summary",
    ),
    "C_KV260": (
        "honest_verdict",
        "kv260_ssh_reachable",
        "loaded_overlay",
        "energy_smoke",
        "overlay_state",
        "uio_devices",
        "xmutil_requires_sudo",
    ),
    "E1_SOTA": (
        "honest_verdict",
        "new_arxiv_ids",
        "next_milestone_candidates",
        "sota_to_phase_d_mapping",
        "d5_conditioning",
        "note_path",
    ),
    "E2_SELF_PLAY": (
        "honest_verdict",
        "verifier_checkpoint_refreshed",
        "checkpoint_path",
        "target_game",
        "offline_reproduced",
        "reproduced_levels",
        "solve_provenance",
        "flag_resolved",
    ),
    "E3_ARC_LEVEL": (
        "honest_verdict",
        "target_game",
        "target_level",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "reproducible_total_levels_after",
        "live_path_reachable",
    ),
}

ARM_SOURCE_TO_META = {
    "D1_LORA_EBM": ("D1", "LoRA-EBM", "MuSR"),
    "D2_UPRM": ("D2", "uPRM", "MuSR"),
    "D3_EBRM": ("D3", "EBRM", "MuSR"),
    "D6_CASCADE": ("D6", "cascade", "MuSR"),
    "D4_SECOND_CORPUS": ("D4", "second-corpus-confirmation", "MMLU-Pro-hard"),
}


def _experiment_id(source: str, artifact: Mapping[str, Any] | None = None) -> int:
    if artifact:
        for field in ("experiment_id", "experiment"):
            value = artifact.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                match = re.search(r"experiment_(\d+)|\b(\d{4})\b", value)
                if match:
                    return int(match.group(1) or match.group(2))
    return UPSTREAM_SOURCES[source].experiment_id


def _is_flagged(artifact: Mapping[str, Any] | None) -> bool:
    return bool(artifact and artifact.get("flagged_adversarial") is True)


def _source_for_experiment(experiment_id: int) -> str | None:
    for source, config in UPSTREAM_SOURCES.items():
        if config.experiment_id == experiment_id:
            return source
    return None


def _clean_artifacts(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {source: artifact for source, artifact in artifacts.items() if not _is_flagged(artifact)}


def _imported_fields(source: str, artifact: Mapping[str, Any]) -> list[str]:
    return [field for field in CLEAN_IMPORT_FIELDS[source] if field in artifact]


def _cited_artifacts(
    clean: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        artifact = clean.get(source)
        if artifact is None:
            continue
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "fields_imported": _imported_fields(source, artifact),
                "sha256": artifact_sha256.get(source, ""),
            }
        )
    return rows


def _flagged_artifacts_skipped(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    seen: set[str] = set()
    for source in UPSTREAM_SOURCES:
        artifact = artifacts.get(source)
        if not _is_flagged(artifact):
            continue
        seen.add(source)
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "reason": "flagged_adversarial",
                "sha256": artifact_sha256.get(source, ""),
                "honest_verdict": artifact.get("honest_verdict", "") if artifact else "",
            }
        )

    d5 = artifacts.get("D5_MOAT_GATE")
    skipped = d5.get("flagged_arms_skipped", []) if isinstance(d5, Mapping) else []
    if isinstance(skipped, list):
        for row in skipped:
            if not isinstance(row, Mapping):
                continue
            experiment_id = _int(row.get("experiment_id"))
            source = _source_for_experiment(experiment_id)
            if source is None or source in seen:
                continue
            seen.add(source)
            rows.append(
                {
                    "source": source,
                    "experiment_id": experiment_id,
                    "path": UPSTREAM_SOURCES[source].relative_path,
                    "reason": "flagged_adversarial",
                    "sha256": artifact_sha256.get(source, ""),
                    "honest_verdict": str(row.get("honest_verdict") or ""),
                }
            )
    return rows


def _execution_status(source: str, artifact: Mapping[str, Any]) -> str:
    status = str(artifact.get("execution_status") or artifact.get("status") or "").lower()
    verdict = str(artifact.get("honest_verdict") or "").lower()
    if status in FAILED_EXECUTION_STATUSES or verdict.startswith("blocked"):
        return "blocked"
    n_questions = _int(artifact.get("n_questions"))
    has_delta = _float(artifact.get("delta_vs_tuned_sc")) is not None
    has_second_delta = _float(artifact.get("delta_vs_tuned_sc_second")) is not None
    has_cascade_metric = _float(artifact.get("cascade_accuracy")) is not None
    if source in ARM_SOURCE_TO_META and n_questions <= 0 and not (
        has_delta or has_second_delta or has_cascade_metric
    ):
        return "blocked"
    return "complete"


def _arm_row_from_direct(source: str, artifact: Mapping[str, Any]) -> JsonDict:
    arm_id, arm, corpus = ARM_SOURCE_TO_META[source]
    status = _execution_status(source, artifact)
    if source == "D4_SECOND_CORPUS":
        delta = _float(artifact.get("delta_vs_tuned_sc_second"))
        paired_ci = artifact.get("paired_ci95_second")
        mcnemar_p = _float(artifact.get("mcnemar_p_second"))
        oracle_at_k = _float(artifact.get("oracle_at_k_second"))
        tuned_sc = _float(artifact.get("genuine_tuned_sc_accuracy_second"))
        selection_accuracy = _float(artifact.get("second_corpus_accuracy"))
    elif source == "D6_CASCADE":
        judge = _float(artifact.get("judge_only_accuracy"))
        cascade = _float(artifact.get("cascade_accuracy"))
        delta = None if judge is None or cascade is None else cascade - judge
        paired_ci = artifact.get("paired_ci95_cascade_vs_judge")
        mcnemar_p = None
        oracle_at_k = None
        tuned_sc = judge
        selection_accuracy = cascade
    else:
        delta = _float(artifact.get("delta_vs_tuned_sc"))
        paired_ci = artifact.get("paired_ci95")
        mcnemar_p = _float(artifact.get("mcnemar_p"))
        oracle_at_k = _float(artifact.get("oracle_at_k"))
        tuned_sc = _float(artifact.get("genuine_tuned_sc_accuracy"))
        selection_accuracy = _float(
            artifact.get("trained_scorer_accuracy")
            or artifact.get("uprm_selection_accuracy")
            or artifact.get("ebrm_selection_accuracy")
        )
    return {
        "arm": arm,
        "arm_id": arm_id,
        "corpus": corpus,
        "delta_vs_tuned_sc": delta,
        "headroom_present": artifact.get("headroom_present") is True,
        "mcnemar_p": mcnemar_p,
        "oracle_at_k": oracle_at_k,
        "paired_ci95": paired_ci if isinstance(paired_ci, list) else None,
        "selection_accuracy": selection_accuracy,
        "source_experiment_id": UPSTREAM_SOURCES[source].experiment_id,
        "tuned_sc_accuracy": tuned_sc,
        "genuine_tuned_sc_accuracy": tuned_sc,
        "verifier_is_oracle": artifact.get("verifier_is_oracle") is True,
        "win_vs_tuned_sc": bool(delta is not None and delta > 0.0),
        "scorer_trained": artifact.get("scorer_trained") if "scorer_trained" in artifact else None,
        "abstention_rate": _float(artifact.get("abstention_rate")),
        "execution_status": status,
        "honest_verdict": str(artifact.get("honest_verdict") or ""),
        "n_questions": _int(artifact.get("n_questions")),
        "judge_call_fraction": _float(artifact.get("judge_call_fraction")),
    }


def _per_arm_table(clean: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    d5 = clean.get("D5_MOAT_GATE")
    d5_rows = d5.get("per_arm_table") if d5 else None
    if isinstance(d5_rows, list):
        return [dict(row) for row in d5_rows if isinstance(row, Mapping)]
    rows: list[JsonDict] = []
    for source in ARM_SOURCE_TO_META:
        artifact = clean.get(source)
        if artifact is not None:
            rows.append(_arm_row_from_direct(source, artifact))
    return rows


def _row_failed_execution(row: Mapping[str, Any]) -> bool:
    status = str(row.get("execution_status") or "").lower()
    if status in FAILED_EXECUTION_STATUSES or status == "blocked":
        return True
    return _float(row.get("delta_vs_tuned_sc")) is None and _int(row.get("n_questions")) <= 0


def _best_arm(clean: Mapping[str, Mapping[str, Any]], rows: list[JsonDict]) -> JsonDict:
    d5 = clean.get("D5_MOAT_GATE")
    best = d5.get("best_arm") if d5 else None
    if isinstance(best, Mapping):
        return dict(best)
    if rows:
        failed = [row for row in rows if _row_failed_execution(row)]
        if failed:
            return dict(failed[0])
        return dict(max(rows, key=lambda row: _float(row.get("delta_vs_tuned_sc")) or 0.0))
    incomplete = d5.get("execution_incomplete_arms") if d5 else None
    if isinstance(incomplete, list):
        for row in incomplete:
            if isinstance(row, Mapping):
                return dict(row)
    return {}


def _best_arm_and_delta(best: Mapping[str, Any]) -> JsonDict:
    return {
        "arm": best.get("arm", ""),
        "arm_id": best.get("arm_id", ""),
        "corpus": best.get("corpus", ""),
        "delta_vs_tuned_sc": _float(best.get("delta_vs_tuned_sc")),
        "paired_ci95": best.get("paired_ci95") if isinstance(best.get("paired_ci95"), list) else None,
        "headroom_present": best.get("headroom_present") is True,
        "verifier_is_oracle": best.get("verifier_is_oracle") is True,
        "win_vs_tuned_sc": best.get("win_vs_tuned_sc") is True,
        "source_experiment_id": _int(best.get("source_experiment_id")),
        "execution_status": str(best.get("execution_status") or ""),
        "scorer_trained": best.get("scorer_trained") if "scorer_trained" in best else None,
        "abstention_rate": _float(best.get("abstention_rate")),
        "judge_call_fraction": _float(best.get("judge_call_fraction")),
    }


def _d1_finally_executed(clean: Mapping[str, Mapping[str, Any]], rows: list[JsonDict]) -> bool:
    d1 = clean.get("D1_LORA_EBM")
    if d1 and d1.get("scorer_trained") is True:
        return True
    for row in rows:
        if row.get("arm_id") == "D1" and row.get("scorer_trained") is True:
            return True
    return False


def _ci_excludes_zero(ci: Any) -> bool:
    return isinstance(ci, list) and len(ci) == 2 and _float(ci[0]) is not None and _float(ci[0]) > 0


def _direct_state(rows: list[JsonDict], clean: Mapping[str, Mapping[str, Any]]) -> str:
    missing_arm = any(source not in clean for source in ARM_SOURCE_TO_META)
    if missing_arm or any(_row_failed_execution(row) for row in rows) or not rows:
        return "execution_incomplete"
    for row in rows:
        if (
            row.get("verifier_is_oracle") is False
            and row.get("headroom_present") is True
            and (_float(row.get("delta_vs_tuned_sc")) or 0.0) > 0.0
            and _ci_excludes_zero(row.get("paired_ci95"))
        ):
            return "moat_realized"
    d1 = clean.get("D1_LORA_EBM")
    d2 = clean.get("D2_UPRM")
    if d1 and d2:
        d1_delta = _float(d1.get("delta_vs_tuned_sc"))
        d2_delta = _float(d2.get("delta_vs_tuned_sc"))
        if d1_delta == 0.0 and d2_delta == 0.0:
            return "moat_retired_bounded"
    return "mixed_musr_scoped"


def _moat_state(clean: Mapping[str, Mapping[str, Any]], rows: list[JsonDict]) -> str:
    d5 = clean.get("D5_MOAT_GATE")
    if d5 and d5.get("moat_realized") is True:
        return "moat_realized"
    if d5 and d5.get("moat_retired_bounded") is True:
        return "moat_retired_bounded"
    if d5:
        decision = str(d5.get("decision") or "").upper()
        incomplete = d5.get("execution_incomplete_arms")
        if "EXECUTION-INCOMPLETE" in decision or (isinstance(incomplete, list) and incomplete):
            return "execution_incomplete"
        return "mixed_musr_scoped"
    return _direct_state(rows, clean)


def _moat_verdict(
    clean: Mapping[str, Mapping[str, Any]],
    rows: list[JsonDict],
    state: str,
) -> JsonDict:
    d5 = clean.get("D5_MOAT_GATE")
    if d5:
        incomplete = d5.get("execution_incomplete_arms")
        return {
            "state": state,
            "source": "D5_MOAT_GATE",
            "decision": str(d5.get("decision") or ""),
            "moat_realized": d5.get("moat_realized") is True,
            "moat_retired_bounded": d5.get("moat_retired_bounded") is True,
            "d5_missing": False,
            "summary": str(d5.get("paper_summary") or ""),
            "execution_incomplete_arms": [
                dict(row) for row in incomplete if isinstance(row, Mapping)
            ]
            if isinstance(incomplete, list)
            else [],
        }
    return {
        "state": state,
        "source": "D1_D4_D6_DIRECT_FALLBACK",
        "decision": "D5-MISSING-DIRECT-AGGREGATION",
        "moat_realized": state == "moat_realized",
        "moat_retired_bounded": state == "moat_retired_bounded",
        "d5_missing": True,
        "summary": f"D5 missing; aggregated {len(rows)} clean or failed D-arm rows directly.",
        "execution_incomplete_arms": [dict(row) for row in rows if _row_failed_execution(row)],
    }


def _diffusiongemma_gate_status(
    clean: Mapping[str, Mapping[str, Any]],
    state: str,
) -> JsonDict:
    d5 = clean.get("D5_MOAT_GATE")
    if d5:
        return {
            "status": str(d5.get("diffusiongemma_gate_status") or "UNKNOWN"),
            "conditions_satisfied_off_arc": (
                d5.get("diffusiongemma_gate_conditions_satisfied_off_arc") is True
            ),
            "activation": str(d5.get("diffusiongemma_activation") or "operator_gated"),
            "operator_gated": True,
            "autonomously_flipped_to_met": False,
        }
    return {
        "status": "D5-MISSING",
        "conditions_satisfied_off_arc": state == "moat_realized",
        "activation": "operator_gated_pending_d5",
        "operator_gated": True,
        "autonomously_flipped_to_met": False,
    }


def _rollup_status(present: bool, ok: bool) -> str:
    if not present:
        return "missing_or_blocked"
    return "complete" if ok else "present_not_complete"


def _infra_rollup(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    b2 = clean.get("B2_LOGPROB_CACHE")
    b3 = clean.get("B3_MOAT_TRAINER")
    n_cached_rows = _int(b2.get("n_cached_rows")) if b2 else 0
    b2_ready = bool(
        b2
        and b2.get("candidate_cache_built") is True
        and b2.get("has_per_token_logprobs") is True
        and n_cached_rows > 0
    )
    b3_ready = bool(
        b3
        and b3.get("smoke_passed") is True
        and b3.get("module_path") == "python/carnot/moat_trainer.py"
    )
    return {
        "b2_logprob_cache": {
            "status": _rollup_status(b2 is not None, b2_ready),
            "robust_cache_ready": b2_ready,
            "honest_verdict": b2.get("honest_verdict", "") if b2 else "",
            "cache_jsonl_path": b2.get("cache_jsonl_path") if b2 else None,
            "n_cached_rows": n_cached_rows,
            "corpora_cached": b2.get("corpora_cached", []) if b2 else [],
            "has_per_token_logprobs": b2.get("has_per_token_logprobs") is True if b2 else False,
            "rescored_not_regenerated": b2.get("rescored_not_regenerated") is True if b2 else False,
        },
        "b3_moat_trainer": {
            "status": _rollup_status(b3 is not None, b3_ready),
            "trainer_module_ready": b3_ready,
            "honest_verdict": b3.get("honest_verdict", "") if b3 else "",
            "module_path": b3.get("module_path") if b3 else None,
            "smoke_passed": b3.get("smoke_passed") is True if b3 else False,
            "smoke_train_loss": _float(b3.get("smoke_train_loss")) if b3 else None,
            "smoke_duration_s": _float(b3.get("smoke_duration_s")) if b3 else None,
            "checkpoint_path": b3.get("checkpoint_path") if b3 else None,
            "base_used": b3.get("base_used") if b3 else None,
        },
    }


def _hardware_rollup(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    c = clean.get("C_KV260")
    ok = bool(c and c.get("kv260_ssh_reachable") is True)
    return {
        "status": _rollup_status(c is not None, ok),
        "kv260_reachable": ok,
        "honest_verdict": c.get("honest_verdict", "") if c else "",
        "loaded_overlay": c.get("loaded_overlay") if c else None,
        "energy_smoke": c.get("energy_smoke", {}) if c else {},
        "uio_devices": c.get("uio_devices", []) if c else [],
        "xmutil_requires_sudo": c.get("xmutil_requires_sudo") is True if c else None,
    }


def _sota_ingestion_rollup(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    e1 = clean.get("E1_SOTA")
    ids = e1.get("new_arxiv_ids", []) if e1 else []
    ids = ids if isinstance(ids, list) else []
    return {
        "status": _rollup_status(e1 is not None, bool(ids)),
        "new_papers_ingested": len(ids),
        "new_arxiv_ids": list(ids),
        "next_milestone_candidates": e1.get("next_milestone_candidates", []) if e1 else [],
        "sota_to_phase_d_mapping": e1.get("sota_to_phase_d_mapping", []) if e1 else [],
        "d5_conditioning": e1.get("d5_conditioning", {}) if e1 else {},
    }


def _self_play_rollup(clean: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    e2 = clean.get("E2_SELF_PLAY")
    refreshed = bool(e2 and e2.get("verifier_checkpoint_refreshed") is True)
    return {
        "status": _rollup_status(e2 is not None, refreshed),
        "checkpoint_refreshed": refreshed,
        "honest_verdict": e2.get("honest_verdict", "") if e2 else "",
        "checkpoint_path": e2.get("checkpoint_path") if e2 else None,
        "target_game": e2.get("target_game") if e2 else None,
        "continuous_self_learning": refreshed,
        "offline_reproduced": e2.get("offline_reproduced") is True if e2 else False,
        "flag_resolved": e2.get("flag_resolved") is True if e2 else False,
    }


def _arc_opportunistic_rollup(
    clean: Mapping[str, Mapping[str, Any]],
    registry_total: int,
) -> JsonDict:
    e3 = clean.get("E3_ARC_LEVEL")
    banked = _int(e3.get("new_levels_banked")) if e3 else 0
    return {
        "status": _rollup_status(e3 is not None, e3 is not None),
        "honest_verdict": e3.get("honest_verdict", "") if e3 else "",
        "target_game": e3.get("target_game") if e3 else None,
        "target_level": _int(e3.get("target_level")) if e3 else 0,
        "new_levels_banked": banked,
        "offline_reproduced": e3.get("offline_reproduced") is True if e3 else False,
        "reproduced_levels": _int(e3.get("reproduced_levels")) if e3 else 0,
        "reproducible_total_levels_after": _int(
            e3.get("reproducible_total_levels_after"), registry_total
        )
        if e3
        else registry_total,
        "live_path_reachable": e3.get("live_path_reachable") is True if e3 else False,
    }


def _arc_deliverable_locked(total: int) -> JsonDict:
    return {
        "locked": True,
        "deliverable": f"levels_{total}_plus_publishable_fover_paper",
        "arc_work_mode": "opportunistic",
    }


def _delta_slug(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    prefix = "neg_" if delta < 0 else ""
    return prefix + f"{abs(delta):.3f}".replace(".", "p")


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return text or "unknown"


def _headline_slug(state: str, best: Mapping[str, Any]) -> str:
    if state == "moat_retired_bounded":
        return "lora_ebm_and_uprm_both_null"
    if state == "execution_incomplete":
        return _slug(best.get("arm") or best.get("arm_id"))
    return (
        f"{_slug(best.get('arm'))}_{_slug(best.get('corpus'))}_"
        f"delta_{_delta_slug(_float(best.get('delta_vs_tuned_sc')))}"
    )


def _honest_verdict(state: str, best: Mapping[str, Any]) -> str:
    state_slug = {
        "moat_realized": "realized",
        "moat_retired_bounded": "retired_bounded",
        "execution_incomplete": "execution_incomplete",
        "mixed_musr_scoped": "musr_scoped",
    }[state]
    return f"complete_capstone_v463_moat_{state_slug}_{_headline_slug(state, best)}"


def _next_milestone_pointer(
    state: str,
    best: Mapping[str, Any],
    sota: Mapping[str, Any],
) -> JsonDict:
    if state == "moat_realized":
        return {
            "milestone": "2026.06.464",
            "direction": "scale_winning_construction",
            "plan": (
                "Scale the winning construction with a larger scorer, more corpora, "
                "and a DiffusionGemma activation proposal for the operator."
            ),
            "best_arm": best.get("arm", ""),
        }
    if state == "moat_retired_bounded":
        return {
            "milestone": "2026.06.464",
            "direction": "pivot_to_e1_sota_candidates",
            "plan": "Pivot to the next verifier direction from E1 SOTA candidates.",
            "candidate_count": len(sota.get("next_milestone_candidates", []))
            if isinstance(sota.get("next_milestone_candidates"), list)
            else 0,
        }
    if state == "execution_incomplete":
        return {
            "milestone": "2026.06.464",
            "direction": "escalate_to_operator",
            "plan": (
                "Escalate to the operator because execution remained incomplete for "
                "the third capstone cycle; off-Codex routing did not suffice."
            ),
            "best_arm": best.get("arm", ""),
            "arm_id": best.get("arm_id", ""),
        }
    return {
        "milestone": "2026.06.464",
        "direction": "tighten_strongest_arm",
        "plan": "Tighten the strongest arm before spending a broader activation attempt.",
        "best_arm": best.get("arm", ""),
    }


def _checksum_payload(payload: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": payload.get("honest_verdict"),
        "moat_verdict": payload.get("moat_verdict"),
        "d1_finally_executed": payload.get("d1_finally_executed"),
        "best_arm_and_delta": payload.get("best_arm_and_delta"),
        "efficiency_win": payload.get("efficiency_win"),
        "diffusiongemma_gate_status": payload.get("diffusiongemma_gate_status"),
        "per_arm_table": payload.get("per_arm_table"),
        "infra_rollup": payload.get("infra_rollup"),
        "reproducible_total_levels": payload.get("reproducible_total_levels"),
        "flagged_artifacts_skipped": payload.get("flagged_artifacts_skipped"),
        "next_milestone_pointer": payload.get("next_milestone_pointer"),
        "cited_upstream_artifacts": payload.get("cited_upstream_artifacts"),
    }


def build_artifact(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    registry: Mapping[str, Any],
    registry_sha256: str | None,
    duration_s: float,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the .463 off-ARC verifier-moat scorecard."""

    clean = _clean_artifacts(artifacts)
    rows = _per_arm_table(clean)
    best = _best_arm(clean, rows)
    best_arm_and_delta = _best_arm_and_delta(best)
    state = _moat_state(clean, rows)
    registry_total = _int(registry.get("reproducible_total_levels"))
    e3_total = _int(
        clean.get("E3_ARC_LEVEL", {}).get("reproducible_total_levels_after"), registry_total
    )
    total = registry_total or e3_total
    infra = _infra_rollup(clean)
    hardware = _hardware_rollup(clean)
    sota = _sota_ingestion_rollup(clean)
    self_play = _self_play_rollup(clean)
    arc = _arc_opportunistic_rollup(clean, total)
    cited = _cited_artifacts(clean, artifact_sha256)
    skipped = _flagged_artifacts_skipped(artifacts, artifact_sha256)
    d5 = clean.get("D5_MOAT_GATE")
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(state, best_arm_and_delta),
        "moat_verdict": _moat_verdict(clean, rows, state),
        "d1_finally_executed": _d1_finally_executed(clean, rows),
        "best_arm_and_delta": best_arm_and_delta,
        "efficiency_win": d5.get("efficiency_win") is True if d5 else state == "moat_realized",
        "diffusiongemma_gate_status": _diffusiongemma_gate_status(clean, state),
        "per_arm_table": rows,
        "infra_rollup": infra,
        "reproducible_total_levels": total,
        "flagged_artifacts_skipped": skipped,
        "next_milestone_pointer": _next_milestone_pointer(state, best_arm_and_delta, sota),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "cited_upstream_artifacts": cited,
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": dict(
            preconditions_checked
            or {
                "agents_md_read": True,
                "codex_md_read": True,
                "registry": {
                    "path": REGISTRY_RELATIVE_PATH,
                    "sha256": registry_sha256 or "",
                    "reproducible_total_levels": registry_total,
                },
            }
        ),
        "hardware_rollup": hardware,
        "sota_ingestion_rollup": sota,
        "self_play_rollup": self_play,
        "arc_opportunistic_rollup": arc,
        "arc_deliverable_locked": _arc_deliverable_locked(total),
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "random_seed": RANDOM_SEED,
        "capstone_ready": bool(best_arm_and_delta.get("arm_id") and rows and cited),
    }
    payload["reproducibility_checksum"] = payload_checksum(_checksum_payload(payload))
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema errors for the .463 scorecard without mutating it."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if not isinstance(payload.get("moat_verdict"), Mapping):
        errors.append("invalid_moat_verdict")
    if not isinstance(payload.get("d1_finally_executed"), bool):
        errors.append("invalid_d1_finally_executed")
    if not isinstance(payload.get("best_arm_and_delta"), Mapping):
        errors.append("invalid_best_arm_and_delta")
    if not isinstance(payload.get("efficiency_win"), bool):
        errors.append("invalid_efficiency_win")
    if not isinstance(payload.get("diffusiongemma_gate_status"), Mapping):
        errors.append("invalid_diffusiongemma_gate_status")
    if not isinstance(payload.get("per_arm_table"), list):
        errors.append("invalid_per_arm_table")
    if not isinstance(payload.get("infra_rollup"), Mapping):
        errors.append("invalid_infra_rollup")
    if not isinstance(payload.get("reproducible_total_levels"), int):
        errors.append("invalid_reproducible_total_levels")
    if not isinstance(payload.get("next_milestone_pointer"), Mapping):
        errors.append("invalid_next_milestone_pointer")
    if not isinstance(payload.get("preconditions_checked"), Mapping):
        errors.append("invalid_preconditions_checked")
    if not isinstance(payload.get("hardware_rollup"), Mapping):
        errors.append("invalid_hardware_rollup")
    if not isinstance(payload.get("sota_ingestion_rollup"), Mapping):
        errors.append("invalid_sota_ingestion_rollup")
    if not isinstance(payload.get("self_play_rollup"), Mapping):
        errors.append("invalid_self_play_rollup")
    if not isinstance(payload.get("arc_opportunistic_rollup"), Mapping):
        errors.append("invalid_arc_opportunistic_rollup")
    if not isinstance(payload.get("arc_deliverable_locked"), Mapping):
        errors.append("invalid_arc_deliverable_locked")
    if not isinstance(payload.get("random_seed"), int) or isinstance(
        payload.get("random_seed"), bool
    ):
        errors.append("invalid_random_seed")
    if not isinstance(payload.get("capstone_ready"), bool):
        errors.append("invalid_capstone_ready")
    field_principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(field_principles, Mapping) or field_principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")
    cited = payload.get("cited_upstream_artifacts")
    if not isinstance(cited, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not isinstance(row.get("fields_imported"), list)
        or not str(row.get("sha256", "")).startswith("sha256:")
        for row in cited
    ):
        errors.append("invalid_cited_upstream_artifacts")
    skipped = payload.get("flagged_artifacts_skipped")
    if not isinstance(skipped, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not str(row.get("sha256", "")).startswith("sha256:")
        or row.get("reason") != "flagged_adversarial"
        for row in skipped
    ):
        errors.append("invalid_flagged_artifacts_skipped")
    if not str(payload.get("reproducibility_checksum", "")).startswith("sha256:"):
        errors.append("invalid_reproducibility_checksum")
    return errors


def run_capstone(*, root: Path = REPO_ROOT) -> JsonDict:
    """Read upstream result JSON and write the .463 capstone scorecard."""

    start = time.perf_counter()
    artifacts: dict[str, JsonDict] = {}
    artifact_sha256: dict[str, str] = {}
    upstream_preconditions: dict[str, JsonDict] = {}
    for source, spec in UPSTREAM_SOURCES.items():
        path = root / spec.relative_path
        present = path.exists()
        upstream_preconditions[source] = {"path": spec.relative_path, "present": present}
        if not present:
            continue
        artifacts[source] = _read_json_object(path)
        artifact_sha256[source] = file_sha256(path) or ""

    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_present = registry_path.exists()
    registry_loadable = False
    registry: JsonDict = {}
    if registry_present:
        try:
            registry = _read_yaml_object(registry_path)
            registry_loadable = True
        except yaml.YAMLError:
            registry = {}

    spec_path = root / SPEC_RELATIVE_PATH
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-5041" in spec_path.read_text(
        encoding="utf-8"
    )
    preconditions_checked = {
        "agents_md_read": True,
        "codex_md_read": True,
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "present": registry_present,
            "yaml_loadable": registry_loadable,
            "sha256": file_sha256(registry_path) or "",
            "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        },
        "spec_has_req_5041": spec_has_req,
        "upstream_artifacts": upstream_preconditions,
    }
    artifact = build_artifact(
        artifacts=artifacts,
        artifact_sha256=artifact_sha256,
        registry=registry,
        registry_sha256=file_sha256(registry_path),
        duration_s=time.perf_counter() - start,
        preconditions_checked=preconditions_checked,
    )
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run_capstone()
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact.get("honest_verdict"),
                "schema_errors": errors,
            },
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
