"""Build the Exp 2884 milestone .272 capstone artifact.

Spec refs: REQ-REPORT-2884, SCENARIO-REPORT-2884.

This module is deliberately a synthesis layer. It reads already-written
milestone artifacts, classifies their claim status, and writes the paper-v6
claim boundary. It does not run model inference, modify the active roadmap, or
touch the conductor.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.milestone_capstone.v272"
MILESTONE = "2026.05.272"
RUN_DATE = "20260522"
OUTPUT_REL_PATH = Path("results/experiment_2884_capstone_v272.json")
PRIOR_CAPSTONE_REL_PATH = Path("results/experiment_2872_capstone_v271.json")

EXPECTED_ARTIFACTS: dict[str, Path] = {
    "exp2873": Path("results/experiment_2873_archive_v271_activate_v272.json"),
    "exp2874": Path("results/experiment_2874_sota_runtime_clean_corrigendum_v4.json"),
    "exp2875": Path(
        "results/experiment_2875_sota_energy_micro_panel_logprob_corrigendum_v2.json"
    ),
    "exp2876": Path("results/experiment_2876_kan_pwa_milp_corrigendum_v2.json"),
    "exp2877": Path(
        "results/experiment_2877_exact_frontier_expansion_halueval_fever_v2.json"
    ),
    "exp2878": Path("results/experiment_2878_halueval_fever_error_verifiability_v1.json"),
    "exp2879": Path("results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json"),
    "exp2880": Path("results/experiment_2880_cross_corpus_matrix_v6.json"),
    "exp2881": Path("results/experiment_2881_fr11_recmem_recurrence_trigger_v1.json"),
    "exp2882": Path("results/experiment_2882_fr11_recmem_replay_scaleup_v1.json"),
    "exp2883": Path("results/experiment_2883_thrml_sampler_portability_smoke_v2.json"),
}

REQUIRED_SUCCESS_FIELDS: dict[str, tuple[str, ...]] = {
    "exp2873": ("archive_already_present",),
    "exp2874": ("sota_runtime_clean", "sota_runtime_ready_v4"),
    "exp2875": ("micro_panel_clean",),
    "exp2876": ("kan_corrigendum_ready", "tautology_flag_cleared"),
    "exp2877": ("frontier_expansion_ready",),
    "exp2878": ("error_verifiability_ready",),
    "exp2879": ("code_manifest_pilot_ready",),
    "exp2880": ("cross_corpus_matrix_built",),
    "exp2881": ("continuous_self_learning_task", "recmem_trigger_ready"),
    "exp2882": ("continuous_self_learning_task", "recmem_replay_scaleup_ready"),
    "exp2883": ("thrml_portability_ready",),
}

FIELD_PRINCIPLES = {
    "paper_ready": (
        "True only when clean matrix evidence contains FoVer plus at least one"
        " clean non-FoVer headline row; flagged side artifacts cannot create"
        " paper readiness."
    ),
    "clean_artifacts": "Expected .272 deliverables with complete verdicts, required booleans, and no flags.",
    "flagged_artifacts": "Artifacts with adversarial/corrigendum flags, even if their own ready boolean is true.",
    "blocked_artifacts": "Artifacts that honestly report dependency or gate blocks.",
    "missing_artifacts": "Expected .272 deliverables that are absent or malformed.",
    "pilot_only_artifacts": "Artifacts that intentionally provide pilot evidence without headline metrics.",
    "corrected_271_flags": "The three .271 flagged branches judged only from their .272 source status.",
    "continuous_self_learning_result": (
        "FR-11 trigger and scale-up evidence kept separate so a flagged scale-up"
        " cannot validate trigger-only evidence."
    ),
    "thrml_sampler_status": "Software sampler status only; no hardware claim is inferred.",
    "paper_v6_safe_claims": "Claims allowed from clean artifacts and explicit pilot boundaries.",
    "paper_v6_forbidden_claims": "Claims excluded because they are flagged, blocked, missing, or pilot-only.",
    "duration_s": "Measured wall-clock duration for synthesis; never sleep-padded.",
}


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object from ``path``, or ``{}`` when it cannot be trusted."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _terminal_success(verdict: object) -> bool:
    if not isinstance(verdict, str):
        return False
    return verdict.strip().startswith(
        ("complete:", "complete_", "success:", "success_", "micro_panel_clean")
    )


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _has_flags(payload: dict[str, Any]) -> bool:
    if payload.get("flagged_adversarial") is True:
        return True
    pending = payload.get("corrigendum_pending")
    if isinstance(pending, list) and pending:
        return True
    flags = payload.get("adversarial_verify_flags")
    if isinstance(flags, list) and flags:
        return True
    summary = payload.get("adversarial_verify_summary")
    if isinstance(summary, dict) and (_number_or_none(summary.get("flag_count")) or 0.0) > 0.0:
        return True
    return payload.get("adversarial_verify_passed") is False


def _required_booleans_pass(exp_id: str, payload: dict[str, Any]) -> bool:
    fields = REQUIRED_SUCCESS_FIELDS.get(exp_id, ())
    return all(payload.get(field) is True for field in fields)


def _is_pilot_only(exp_id: str, payload: dict[str, Any]) -> bool:
    return (
        exp_id == "exp2879"
        and payload.get("code_manifest_pilot_ready") is True
        and payload.get("headline_metric_claim_made") is False
    )


def classify_artifact(exp_id: str, payload: dict[str, Any], present: bool) -> str:
    """REQ-REPORT-2884: classify one source artifact's claim status."""

    if not present or not payload:
        return "missing"
    if _has_flags(payload):
        return "flagged"
    if _is_pilot_only(exp_id, payload):
        return "pilot-only"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    blocked_reason = payload.get("blocked_reason")
    if isinstance(blocked_reason, str) and blocked_reason.startswith("blocked"):
        return "blocked"
    if _terminal_success(payload.get("honest_verdict")) and _required_booleans_pass(
        exp_id, payload
    ):
        return "clean"
    if _terminal_success(payload.get("honest_verdict")):
        return "blocked"
    return "missing"


def _load_expected(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    payloads: dict[str, dict[str, Any]] = {}
    present: dict[str, bool] = {}
    for exp_id, rel_path in EXPECTED_ARTIFACTS.items():
        path = root / rel_path
        present[exp_id] = path.is_file()
        payloads[exp_id] = read_json(path) if present[exp_id] else {}
    return payloads, present


def _classify_all(
    payloads: dict[str, dict[str, Any]], present: dict[str, bool]
) -> dict[str, str]:
    return {
        exp_id: classify_artifact(exp_id, payloads[exp_id], present[exp_id])
        for exp_id in EXPECTED_ARTIFACTS
    }


def _ids_with_status(statuses: dict[str, str], wanted: str) -> list[str]:
    return [exp_id for exp_id in EXPECTED_ARTIFACTS if statuses.get(exp_id) == wanted]


def _prior_v5_headline_rows(root: Path) -> list[str]:
    prior = read_json(root / PRIOR_CAPSTONE_REL_PATH)
    rows = prior.get("headline_eligible_rows")
    if isinstance(rows, list) and all(isinstance(row, str) for row in rows):
        return list(rows)
    return ["FoVer", "HaluEval/FEVER"]


def _headline_rows(statuses: dict[str, str], matrix_payload: dict[str, Any]) -> list[str]:
    if statuses.get("exp2880") != "clean":
        return []
    rows = matrix_payload.get("headline_eligible_rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, str)]


def _pilot_rows(matrix_payload: dict[str, Any]) -> list[str]:
    rows = matrix_payload.get("pilot_only_rows")
    return [row for row in rows if isinstance(row, str)] if isinstance(rows, list) else []


def _paper_ready(statuses: dict[str, str], matrix_payload: dict[str, Any]) -> bool:
    rows = _headline_rows(statuses, matrix_payload)
    return (
        statuses.get("exp2880") == "clean"
        and matrix_payload.get("cross_corpus_matrix_built") is True
        and "FoVer" in rows
        and any(row != "FoVer" for row in rows)
    )


def _corrected_271_flags(
    payloads: dict[str, dict[str, Any]], statuses: dict[str, str]
) -> dict[str, dict[str, Any]]:
    return {
        "runtime": {
            "prior_flagged_artifact": "exp2862",
            "correcting_artifact": "exp2874",
            "status": statuses["exp2874"],
            "corrected": statuses["exp2874"] == "clean"
            and payloads["exp2874"].get("sota_runtime_clean") is True,
            "two_model_cached_pair_ready": bool(
                payloads["exp2874"].get("cached_sota_pair_returned_two_loadable_specs")
            ),
        },
        "micro_panel": {
            "prior_flagged_artifact": "exp2870",
            "correcting_artifact": "exp2875",
            "status": statuses["exp2875"],
            "source_reported_micro_panel_clean": bool(
                payloads["exp2875"].get("micro_panel_clean")
            ),
            "corrected": statuses["exp2875"] == "clean"
            and payloads["exp2875"].get("micro_panel_clean") is True,
        },
        "kan_pwa_milp": {
            "prior_flagged_artifact": "exp2871",
            "correcting_artifact": "exp2876",
            "status": statuses["exp2876"],
            "corrected": statuses["exp2876"] == "clean"
            and payloads["exp2876"].get("tautology_flag_cleared") is True,
            "solver_status": payloads["exp2876"].get("solver_status"),
        },
    }


def _matrix_v6_comparison(root: Path, matrix_payload: dict[str, Any]) -> dict[str, Any]:
    v5_rows = _prior_v5_headline_rows(root)
    v6_headline = [
        row for row in matrix_payload.get("headline_eligible_rows", []) if isinstance(row, str)
    ]
    v6_pilot = _pilot_rows(matrix_payload)
    v6_total = int(matrix_payload.get("clean_row_count") or 0)
    v5_total = len(v5_rows)
    return {
        "v5_headline_eligible_rows": v5_rows,
        "v6_headline_eligible_rows": v6_headline,
        "v6_pilot_only_rows": v6_pilot,
        "v5_clean_or_headline_row_count": v5_total,
        "v6_clean_or_pilot_row_count": v6_total,
        "v6_has_more_total_clean_or_pilot_evidence_than_v5": v6_total > v5_total,
        "new_headline_eligible_rows_vs_v5": [row for row in v6_headline if row not in v5_rows],
        "new_pilot_only_rows_vs_v5": [row for row in v6_pilot if row not in v5_rows],
        "truthfulqa_status": (
            "missing" if "TruthfulQA" in dict(matrix_payload.get("missing_rows") or {}) else "present"
        ),
    }


def _continuous_self_learning_result(
    payloads: dict[str, dict[str, Any]], statuses: dict[str, str]
) -> dict[str, Any]:
    trigger = payloads["exp2881"]
    scaleup = payloads["exp2882"]
    trigger_ready = statuses["exp2881"] == "clean" and trigger.get("recmem_trigger_ready") is True
    scaleup_clean = (
        statuses["exp2882"] == "clean" and scaleup.get("recmem_replay_scaleup_ready") is True
    )
    forgetting = scaleup.get("forgetting_regression_count")
    if scaleup_clean and forgetting == 0:
        non_forgetting_status = "clean_scaleup_reports_zero_forgetting"
    elif statuses["exp2882"] == "flagged" and forgetting == 0:
        non_forgetting_status = "flagged_scaleup_reports_zero_forgetting"
    else:
        non_forgetting_status = "not_established"
    return {
        "continuous_self_learning_task": bool(
            trigger.get("continuous_self_learning_task")
            or scaleup.get("continuous_self_learning_task")
        ),
        "recurrence_trigger_status": statuses["exp2881"],
        "recurrence_trigger_ready": trigger_ready,
        "n_events_ingested": trigger.get("n_events_ingested"),
        "n_recurrence_clusters": trigger.get("n_recurrence_clusters"),
        "n_consolidations_triggered": trigger.get("n_consolidations_triggered"),
        "eager_consolidations_avoided": trigger.get("eager_consolidations_avoided"),
        "token_reduction_proxy_pct": _number_or_none(trigger.get("token_reduction_proxy_pct")),
        "trigger_contradiction_rate": _number_or_none(trigger.get("contradiction_rate")),
        "trigger_duplicate_rate": _number_or_none(trigger.get("duplicate_rate")),
        "trigger_forgetting_regression_count": trigger.get("forgetting_regression_count"),
        "replay_scaleup_status": statuses["exp2882"],
        "scaleup_reported_ready": bool(scaleup.get("recmem_replay_scaleup_ready")),
        "scaleup_claim_clean": scaleup_clean,
        "n_examples": scaleup.get("n_examples"),
        "target_examples_met": bool(scaleup.get("target_examples_met")),
        "energy_delta_mean": _number_or_none(scaleup.get("energy_delta_mean")),
        "correctness_delta": _number_or_none(scaleup.get("correctness_delta")),
        "auroc_delta": _number_or_none(scaleup.get("auroc_delta")),
        "scaleup_token_reduction_pct": _number_or_none(scaleup.get("token_reduction_pct")),
        "memory_drift_score": _number_or_none(scaleup.get("memory_drift_score")),
        "forgetting_regression_count": forgetting,
        "model_weights_mutated": scaleup.get("model_weights_mutated"),
        "live_llm_called": bool(trigger.get("live_llm_called") or scaleup.get("live_llm_called")),
        "non_forgetting_status": non_forgetting_status,
        "safe_fr11_claim": (
            "recurrence_trigger_only" if trigger_ready and not scaleup_clean else "trigger_and_scaleup"
        )
        if trigger_ready
        else "none",
    }


def _thrml_status(payload: dict[str, Any], status: str) -> str:
    if status == "missing":
        return "missing"
    if status == "flagged":
        return "flagged"
    if payload.get("hardware_claim_made") is True:
        return "invalid_hardware_claim_made"
    if status == "clean":
        return "thrml_portability_ready_no_hardware_claim"
    if payload.get("blocked_reason") == "blocked_thrml_unavailable":
        if payload.get("local_fallback_ran") is True:
            return "blocked_thrml_unavailable_local_fallback_ran_no_hardware_claim"
        return "blocked_thrml_unavailable_no_hardware_claim"
    return status


def _safe_claims(
    *,
    paper_ready: bool,
    headline_rows: list[str],
    statuses: dict[str, str],
    payloads: dict[str, dict[str, Any]],
    fr11: dict[str, Any],
    matrix_comparison: dict[str, Any],
) -> list[str]:
    claims: list[str] = []
    if paper_ready:
        claims.append(
            "FoVer and HaluEval/FEVER remain the only headline-eligible paper-v6 "
            "matrix rows from clean evidence."
        )
    if statuses["exp2874"] == "clean":
        claims.append(
            "A single mandated SOTA GGUF runtime path is clean; two-model cached-pair "
            f"readiness remains {payloads['exp2874'].get('cached_sota_pair_returned_two_loadable_specs') is True}."
        )
    if statuses["exp2876"] == "clean":
        claims.append(
            "KAN PWA/MILP corrigendum cleared the .271 tautology with distinct local "
            "and global bounds in a tiny z3-backed case."
        )
    if fr11["recurrence_trigger_ready"]:
        claims.append(
            "FR-11 RecMem recurrence-triggered consolidation is ready as a clean "
            "trigger prototype with token-cost reduction and no live LLM call."
        )
    pilot_rows = matrix_comparison["v6_pilot_only_rows"]
    if pilot_rows:
        claims.append(
            "MBPP and HumanEval may be described only as manifest execution pilot "
            f"rows, not headline metrics: {', '.join(pilot_rows)}."
        )
    return claims or ["No paper-v6 claim is safe from the available clean evidence."]


def _forbidden_claims(
    *,
    statuses: dict[str, str],
    payloads: dict[str, dict[str, Any]],
    fr11: dict[str, Any],
    matrix_comparison: dict[str, Any],
    thrml_status: str,
) -> list[str]:
    claims = [
        "Do not cite MBPP or HumanEval as headline benchmark rows; Exp 2879/2880 mark them pilot-only.",
        "Do not cite TruthfulQA metrics; matrix v6 still marks TruthfulQA missing.",
        "Do not claim THRML, TSU, or hardware acceleration from the sampler branch; "
        f"status is {thrml_status}.",
    ]
    if statuses["exp2875"] == "flagged" or payloads["exp2875"].get("benchmark_claim_made") is False:
        claims.append(
            "Do not cite Exp 2875 as a SOTA energy/logprob benchmark; it is flagged "
            "or explicitly makes no benchmark claim."
        )
    if not payloads["exp2874"].get("cached_sota_pair_returned_two_loadable_specs"):
        claims.append("Do not claim two-model cached SOTA-pair readiness.")
    if statuses["exp2882"] == "flagged" or not fr11["scaleup_claim_clean"]:
        claims.append(
            "Do not cite Exp 2882 RecMem scale-up correctness, AUROC, energy-vs-eager, "
            "or non-forgetting as clean evidence."
        )
    if matrix_comparison["new_headline_eligible_rows_vs_v5"] == []:
        claims.append("Do not claim matrix v6 added new headline-eligible rows beyond v5.")
    return claims


def _top_3_next_actions(
    *,
    statuses: dict[str, str],
    paper_ready: bool,
    matrix_comparison: dict[str, Any],
    fr11: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    if statuses["exp2875"] == "flagged":
        actions.append(
            "Re-run Exp 2875 with adversarial-clean duration and reproducibility checksum, "
            "or downgrade it permanently to a non-benchmark telemetry note."
        )
    if statuses["exp2882"] == "flagged":
        actions.append(
            "Repair Exp 2882 with non-tautological eager-vs-RecMem metrics before claiming "
            "FR-11 scale-up correctness, AUROC, energy, drift, or non-forgetting."
        )
    if not paper_ready:
        actions.append("Restore a clean matrix v6 before carrying paper-v6 readiness forward.")
    if matrix_comparison["new_pilot_only_rows_vs_v5"] or matrix_comparison["truthfulqa_status"] == "missing":
        actions.append(
            "Promote MBPP/HumanEval from pilot-only and materialize TruthfulQA only when "
            "clean generated-code or label evidence exists."
        )
    if fr11["recurrence_trigger_ready"] and not fr11["scaleup_claim_clean"]:
        actions.append(
            "Keep the clean RecMem trigger, but gate any self-learning headline on a clean "
            "scale-up rerun."
        )
    if len(actions) < 3:
        actions.append("Use matrix v6 safe claims only, and leave flagged branches out of paper-v6.")
    return actions[:3]


def _compose_verdict(
    *,
    paper_ready: bool,
    clean_count: int,
    flagged_count: int,
    blocked_count: int,
    missing_count: int,
    pilot_count: int,
) -> str:
    return (
        "complete: .272 capstone synthesized; "
        f"paper_ready={str(paper_ready).lower()}; "
        f"clean_artifacts={clean_count}; flagged_artifacts={flagged_count}; "
        f"blocked_artifacts={blocked_count}; missing_artifacts={missing_count}; "
        f"pilot_only_artifacts={pilot_count}"
    )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2884: synthesize the milestone .272 paper claim boundary."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    payloads, present = _load_expected(root_path)
    statuses = _classify_all(payloads, present)

    headline_rows = _headline_rows(statuses, payloads["exp2880"])
    paper_ready = _paper_ready(statuses, payloads["exp2880"])
    matrix_comparison = _matrix_v6_comparison(root_path, payloads["exp2880"])
    fr11 = _continuous_self_learning_result(payloads, statuses)
    sampler_status = _thrml_status(payloads["exp2883"], statuses["exp2883"])

    clean_artifacts = _ids_with_status(statuses, "clean")
    flagged_artifacts = _ids_with_status(statuses, "flagged")
    blocked_artifacts = _ids_with_status(statuses, "blocked")
    missing_artifacts = _ids_with_status(statuses, "missing")
    pilot_only_artifacts = _ids_with_status(statuses, "pilot-only")

    safe_claims = _safe_claims(
        paper_ready=paper_ready,
        headline_rows=headline_rows,
        statuses=statuses,
        payloads=payloads,
        fr11=fr11,
        matrix_comparison=matrix_comparison,
    )
    forbidden_claims = _forbidden_claims(
        statuses=statuses,
        payloads=payloads,
        fr11=fr11,
        matrix_comparison=matrix_comparison,
        thrml_status=sampler_status,
    )
    top_3 = _top_3_next_actions(
        statuses=statuses,
        paper_ready=paper_ready,
        matrix_comparison=matrix_comparison,
        fr11=fr11,
    )
    end = time.perf_counter() if now_s is None else now_s

    return {
        "schema": SCHEMA,
        "artifact": "experiment_2884_capstone_v272",
        "honest_verdict": _compose_verdict(
            paper_ready=paper_ready,
            clean_count=len(clean_artifacts),
            flagged_count=len(flagged_artifacts),
            blocked_count=len(blocked_artifacts),
            missing_count=len(missing_artifacts),
            pilot_count=len(pilot_only_artifacts),
        ),
        "milestone": MILESTONE,
        "paper_ready": paper_ready,
        "clean_artifacts": clean_artifacts,
        "flagged_artifacts": flagged_artifacts,
        "blocked_artifacts": blocked_artifacts,
        "missing_artifacts": missing_artifacts,
        "pilot_only_artifacts": pilot_only_artifacts,
        "corrected_271_flags": _corrected_271_flags(payloads, statuses),
        "sota_runtime_clean": statuses["exp2874"] == "clean"
        and payloads["exp2874"].get("sota_runtime_clean") is True,
        "micro_panel_clean": statuses["exp2875"] == "clean"
        and payloads["exp2875"].get("micro_panel_clean") is True,
        "kan_tautology_cleared": statuses["exp2876"] == "clean"
        and payloads["exp2876"].get("tautology_flag_cleared") is True,
        "cross_corpus_matrix_built": statuses["exp2880"] == "clean"
        and payloads["exp2880"].get("cross_corpus_matrix_built") is True,
        "headline_eligible_rows": headline_rows,
        "continuous_self_learning_result": fr11,
        "thrml_sampler_status": sampler_status,
        "paper_v6_safe_claims": safe_claims,
        "paper_v6_forbidden_claims": forbidden_claims,
        "top_3_next_actions": top_3,
        "matrix_v6_comparison": matrix_comparison,
        "source_artifact_status": {
            exp_id: {
                "path": str(EXPECTED_ARTIFACTS[exp_id]),
                "status": statuses[exp_id],
                "present": present[exp_id],
                "honest_verdict": payloads[exp_id].get("honest_verdict"),
            }
            for exp_id in EXPECTED_ARTIFACTS
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "files_not_modified": ["research-roadmap.yaml", "scripts/research_conductor.py"],
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, end - start), 6),
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2884 capstone JSON deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())
