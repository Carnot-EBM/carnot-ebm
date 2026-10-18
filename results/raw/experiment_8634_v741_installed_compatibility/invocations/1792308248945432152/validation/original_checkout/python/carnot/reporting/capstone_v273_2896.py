"""Build the Exp 2896 milestone .273 capstone artifact.

Spec refs: REQ-REPORT-2896, SCENARIO-REPORT-2896.

This module is a pure synthesis layer for the 2026.05.273 milestone close:
it reads the already-written .273 deliverables, classifies each one against
strict clean/flagged/blocked/missing/pilot-only/taxonomy-only buckets, and
emits the paper-v6 claim boundary. It does not invoke models, modify the
roadmap, or touch the conductor. The honest_verdict and paper_ready flags
are derived only from clean matrix evidence; flagged side artifacts cannot
create paper readiness.

Layout follows the .272 capstone (capstone_v272_2884) so a future reader
sees the diff between milestones at a glance.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.milestone_capstone.v273"
MILESTONE = "2026.05.273"
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2896_capstone_v273.json")
PRIOR_CAPSTONE_REL_PATH = Path("results/experiment_2884_capstone_v272.json")

EXPECTED_ARTIFACTS: dict[str, Path] = {
    "exp2885": Path("results/experiment_2885_archive_v272_activate_v273.json"),
    "exp2886": Path("results/experiment_2886_sota_micro_panel_clean_telemetry_v3.json"),
    "exp2887": Path("results/experiment_2887_fr11_fast_slow_memory_corrigendum_v2.json"),
    "exp2888": Path("results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json"),
    "exp2889": Path("results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json"),
    "exp2890": Path("results/experiment_2890_code_structural_dependency_verifier_v1.json"),
    "exp2891": Path("results/experiment_2891_cctu_executable_constraint_validator_pilot_v1.json"),
    "exp2892": Path("results/experiment_2892_vericot_exact_frontier_expansion_v1.json"),
    "exp2893": Path("results/experiment_2893_kan_hardware_complexity_accounting_v1.json"),
    "exp2894": Path("results/experiment_2894_cross_corpus_matrix_v7.json"),
    "exp2895": Path("results/experiment_2895_paper_v6_evidence_table_v4.json"),
}

# Booleans the source artifact must report True for the classifier to even
# consider classifying the artifact "clean". A failure here downgrades to
# "blocked" because the artifact's own contract isn't satisfied.
REQUIRED_SUCCESS_FIELDS: dict[str, tuple[str, ...]] = {
    "exp2885": ("paper_ready_from_capstone",),
    "exp2886": ("micro_panel_clean",),
    "exp2887": ("fr11_scaleup_clean", "continuous_self_learning_task"),
    "exp2888": ("truthfulqa_taxonomy_ready",),
    "exp2889": ("manifest_contract_ready",),
    "exp2890": ("structural_dependency_verifier_ready",),
    "exp2891": ("cctu_validator_ready",),
    "exp2892": ("vericot_frontier_ready",),
    "exp2893": ("kan_complexity_accounting_ready",),
    "exp2894": ("cross_corpus_matrix_built",),
    "exp2895": ("paper_evidence_table_ready",),
}

# Pilot-only and taxonomy-only artifacts make manifest-style evidence but
# explicitly do not claim a headline benchmark metric. They are NOT flagged
# and they are NOT blocked — they're a distinct evidence tier.
PILOT_ONLY_IDS: tuple[str, ...] = ("exp2891",)
TAXONOMY_ONLY_IDS: tuple[str, ...] = ("exp2888",)

FIELD_PRINCIPLES = {
    "paper_ready": (
        "True only when matrix v7 is clean and contains FoVer plus at least"
        " one other clean headline row; flagged side artifacts cannot create"
        " paper readiness."
    ),
    "clean_artifacts": "Expected .273 deliverables with complete verdicts, required booleans, and no flags.",
    "flagged_artifacts": "Artifacts with adversarial/corrigendum flags, even if their own ready boolean is true.",
    "blocked_artifacts": "Artifacts that honestly report dependency or gate blocks.",
    "missing_artifacts": "Expected .273 deliverables that are absent or malformed.",
    "pilot_only_artifacts": "Artifacts that intentionally provide pilot evidence without headline metrics (CCTU).",
    "taxonomy_only_artifacts": "Artifacts that provide error-taxonomy or manifest-only rows with no generated-answer metrics (TruthfulQA).",
    "corrected_272_flags": "The two .272 flagged branches (micro_panel, RecMem scaleup) judged only from their .273 source status.",
    "micro_panel_clean": "Exp 2886 micro-panel clean only if its own boolean is True AND it is not flagged.",
    "fr11_scaleup_clean": "Exp 2887 fr11_scaleup_clean only if its own boolean is True AND it is not flagged.",
    "cross_corpus_matrix_built": "Exp 2894 matrix v7 built only if its own boolean is True AND it is not flagged.",
    "headline_eligible_rows": "Headline rows reported by matrix v7 when v7 is clean; otherwise empty.",
    "continuous_self_learning_result": (
        "FR-11 evidence from Exp 2887 fast/slow memory comparator with eager, RecMem,"
        " and fast/slow policies separated so a tautological policy cannot mask"
        " another."
    ),
    "constraint_benchmark_status": (
        "Status of the .273 constraint-benchmark expansion: CCTU pilot, VeriCoT formal"
        " frontier, structural dependency verifier."
    ),
    "kan_complexity_status": (
        "Exp 2893 KAN PWA/MILP complexity accounting status; ready iff the artifact is"
        " clean and makes no hardware or analog execution claim."
    ),
    "paper_v6_safe_claims": "Claims allowed from clean artifacts and explicit pilot/taxonomy boundaries.",
    "paper_v6_forbidden_claims": "Claims excluded because they are flagged, blocked, missing, pilot-only, or taxonomy-only.",
    "top_3_next_actions": "Three operator-actionable next steps that close the largest .273 gaps.",
    "duration_s": "Measured wall-clock duration for synthesis; never sleep-padded.",
    "run_date": "Pinned conductor run date.",
    "inference_substrate": (
        "Forward-only declaration that this artifact is pure aggregation. The"
        " adversarial-verify linter therefore applies the aggregation duration"
        " floor (1ms), not the live-LLM 60s floor."
    ),
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
        ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
    )


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _has_flags(payload: dict[str, Any]) -> bool:
    """Return True if any adversarial flag mechanism fires on the payload."""

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


def classify_artifact(exp_id: str, payload: dict[str, Any], present: bool) -> str:
    """REQ-REPORT-2896: classify one source artifact's claim status.

    Order matters: flags beat everything (an adversarially-flagged artifact
    cannot also be "clean"), then pilot-only and taxonomy-only buckets (a
    deliberately pilot-only artifact is not a regression), then blocked
    verdicts, then clean iff terminal + required-booleans pass.
    """

    if not present or not payload:
        return "missing"
    if _has_flags(payload):
        return "flagged"
    if exp_id in PILOT_ONLY_IDS and _terminal_success(payload.get("honest_verdict")):
        if payload.get("headline_metric_claim_made") is False:
            return "pilot-only"
    if exp_id in TAXONOMY_ONLY_IDS and _terminal_success(payload.get("honest_verdict")):
        if payload.get("headline_metric_claim_made") is False:
            return "taxonomy-only"
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


def _headline_rows(statuses: dict[str, str], matrix_payload: dict[str, Any]) -> list[str]:
    if statuses.get("exp2894") != "clean":
        return []
    rows = matrix_payload.get("headline_eligible_rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, str)]


def _paper_ready(statuses: dict[str, str], matrix_payload: dict[str, Any]) -> bool:
    rows = _headline_rows(statuses, matrix_payload)
    return (
        statuses.get("exp2894") == "clean"
        and matrix_payload.get("cross_corpus_matrix_built") is True
        and "FoVer" in rows
        and any(row != "FoVer" for row in rows)
    )


def _corrected_272_flags(
    payloads: dict[str, dict[str, Any]], statuses: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Report whether each .272 flagged branch was corrected by a .273 task."""

    return {
        "micro_panel": {
            "prior_flagged_artifact": "exp2875",
            "correcting_artifact": "exp2886",
            "status": statuses["exp2886"],
            "source_reported_micro_panel_clean": bool(
                payloads["exp2886"].get("micro_panel_clean")
            ),
            "benchmark_claim_made": bool(payloads["exp2886"].get("benchmark_claim_made")),
            "corrected": statuses["exp2886"] == "clean"
            and payloads["exp2886"].get("micro_panel_clean") is True,
        },
        "fr11_scaleup": {
            "prior_flagged_artifact": "exp2882",
            "correcting_artifact": "exp2887",
            "status": statuses["exp2887"],
            "best_policy": payloads["exp2887"].get("best_policy"),
            "exp2882_root_cause_recorded": bool(
                (payloads["exp2887"].get("exp2882_flag_diagnosis") or {}).get("root_cause")
            ),
            "corrected": statuses["exp2887"] == "clean"
            and payloads["exp2887"].get("fr11_scaleup_clean") is True,
        },
    }


def _continuous_self_learning_result(
    payloads: dict[str, dict[str, Any]], statuses: dict[str, str]
) -> dict[str, Any]:
    """Build the FR-11 fast/slow vs eager vs RecMem summary from Exp 2887."""

    payload = payloads["exp2887"]
    scaleup_clean = (
        statuses["exp2887"] == "clean" and payload.get("fr11_scaleup_clean") is True
    )
    policies = payload.get("policies_compared")
    if not isinstance(policies, list):
        policies = []
    energy = payload.get("energy_delta_by_policy")
    if not isinstance(energy, dict):
        energy = {}
    correctness = payload.get("correctness_delta_by_policy")
    if not isinstance(correctness, dict):
        correctness = {}
    auroc = payload.get("auroc_delta_by_policy")
    if not isinstance(auroc, dict):
        auroc = {}
    contradiction = payload.get("contradiction_rate_by_policy")
    if not isinstance(contradiction, dict):
        contradiction = {}
    duplicate = payload.get("duplicate_rate_by_policy")
    if not isinstance(duplicate, dict):
        duplicate = {}
    drift = payload.get("memory_drift_by_policy")
    if not isinstance(drift, dict):
        drift = {}
    forgetting = payload.get("forgetting_regression_count_by_policy")
    if not isinstance(forgetting, dict):
        forgetting = {}
    policy_metrics = payload.get("policy_metrics")
    if not isinstance(policy_metrics, dict):
        policy_metrics = {}

    def _per_policy_token_pct() -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for name in policies:
            metrics = policy_metrics.get(name) if isinstance(policy_metrics, dict) else None
            if isinstance(metrics, dict):
                result[name] = _number_or_none(metrics.get("token_reduction_pct"))
            else:
                result[name] = None
        return result

    return {
        "continuous_self_learning_task": bool(payload.get("continuous_self_learning_task")),
        "policies_compared": policies,
        "best_policy": payload.get("best_policy"),
        "fr11_scaleup_clean": scaleup_clean,
        "n_examples": payload.get("n_examples"),
        "energy_delta_by_policy": {k: _number_or_none(v) for k, v in energy.items()},
        "correctness_delta_by_policy": {k: _number_or_none(v) for k, v in correctness.items()},
        "auroc_delta_by_policy": {k: _number_or_none(v) for k, v in auroc.items()},
        "contradiction_rate_by_policy": {k: _number_or_none(v) for k, v in contradiction.items()},
        "duplicate_rate_by_policy": {k: _number_or_none(v) for k, v in duplicate.items()},
        "memory_drift_by_policy": {k: _number_or_none(v) for k, v in drift.items()},
        "forgetting_regression_count_by_policy": dict(forgetting),
        "token_reduction_pct_by_policy": _per_policy_token_pct(),
        "live_llm_called": bool(payload.get("live_llm_called")),
        "model_weights_mutated": bool(payload.get("model_weights_mutated")),
        "exp2882_diagnosis_recorded": bool(payload.get("exp2882_flag_diagnosis")),
        "non_tautological_policy_energy": bool(
            (payload.get("adversarial_clean_checks") or {}).get("non_tautological_policy_energy")
        ),
        "fast_slow_separates_from_recmem": bool(
            (payload.get("adversarial_clean_checks") or {}).get("fast_slow_separates_from_recmem")
        ),
        "safe_fr11_claim": (
            "fast_slow_vs_recmem_vs_eager_separated"
            if scaleup_clean
            else "recurrence_trigger_only"
        ),
    }


def _constraint_benchmark_status(
    payloads: dict[str, dict[str, Any]], statuses: dict[str, str]
) -> dict[str, Any]:
    """Summarize the .273 constraint-benchmark expansion (CCTU, VeriCoT, structural)."""

    cctu = payloads["exp2891"]
    vericot = payloads["exp2892"]
    structural = payloads["exp2890"]

    return {
        "cctu_pilot": {
            "status": statuses["exp2891"],
            "n_cases": cctu.get("n_cases"),
            "category_coverage": cctu.get("category_coverage"),
            "executable_validation_used": bool(cctu.get("executable_validation_used")),
            "headline_metric_claim_made": cctu.get("headline_metric_claim_made"),
            "ready": statuses["exp2891"] in ("clean", "pilot-only")
            and cctu.get("cctu_validator_ready") is True,
        },
        "vericot_frontier": {
            "status": statuses["exp2892"],
            "n_candidate_rows": vericot.get("n_candidate_rows"),
            "n_vericot_supported_rows": vericot.get("n_vericot_supported_rows"),
            "n_unsupported_rows": vericot.get("n_unsupported_rows"),
            "solver_backend": vericot.get("solver_backend"),
            "autoformalization_llm_called": vericot.get("autoformalization_llm_called"),
            "ready": statuses["exp2892"] == "clean"
            and vericot.get("vericot_frontier_ready") is True,
        },
        "structural_verifier": {
            "status": statuses["exp2890"],
            "n_contracts_built": structural.get("n_contracts_built"),
            "n_rows_verified": structural.get("n_rows_verified"),
            "violation_types": structural.get("violation_types"),
            "contract_schema_errors": structural.get("contract_schema_errors"),
            "ready": statuses["exp2890"] == "clean"
            and structural.get("structural_dependency_verifier_ready") is True,
        },
    }


def _kan_complexity_status(payload: dict[str, Any], status: str) -> str:
    """Translate Exp 2893's hardware-claim-boundary booleans into a status string."""

    if status == "missing":
        return "missing"
    if status == "flagged":
        return "flagged"
    if payload.get("analog_kan_claim_made") is True or payload.get(
        "hardware_execution_claim_made"
    ) is True:
        return "invalid_hardware_or_analog_claim_made"
    if status == "clean":
        return "complexity_accounting_ready_no_hardware_or_analog_claim"
    if status == "blocked":
        return "blocked"
    return status


def _matrix_v7_comparison(
    root: Path, matrix_payload: dict[str, Any], statuses: dict[str, str]
) -> dict[str, Any]:
    """Compare v7 against the prior .272 capstone's v6 headline/pilot set."""

    prior = read_json(root / PRIOR_CAPSTONE_REL_PATH)
    prior_headline = prior.get("headline_eligible_rows")
    if not (isinstance(prior_headline, list) and all(isinstance(r, str) for r in prior_headline)):
        prior_headline = ["FoVer", "HaluEval/FEVER"]
    v7_headline = _headline_rows(statuses, matrix_payload)
    v7_pilot = matrix_payload.get("pilot_only_rows")
    v7_pilot = [r for r in v7_pilot if isinstance(r, str)] if isinstance(v7_pilot, list) else []
    v7_taxonomy = matrix_payload.get("taxonomy_only_rows")
    v7_taxonomy = (
        [r for r in v7_taxonomy if isinstance(r, str)] if isinstance(v7_taxonomy, list) else []
    )
    v7_clean_count = int(matrix_payload.get("clean_row_count") or 0)
    return {
        "v6_headline_eligible_rows": list(prior_headline),
        "v7_headline_eligible_rows": v7_headline,
        "v7_pilot_only_rows": v7_pilot,
        "v7_taxonomy_only_rows": v7_taxonomy,
        "v7_clean_or_pilot_or_taxonomy_row_count": v7_clean_count,
        "new_headline_eligible_rows_vs_v6": [
            row for row in v7_headline if row not in prior_headline
        ],
        "new_pilot_only_rows_vs_v6": v7_pilot,
        "new_taxonomy_only_rows_vs_v6": v7_taxonomy,
        "matrix_v7_adds_headline_evidence_beyond_v6": any(
            row not in prior_headline for row in v7_headline
        ),
    }


def _safe_claims(
    *,
    paper_ready: bool,
    statuses: dict[str, str],
    payloads: dict[str, dict[str, Any]],
    fr11: dict[str, Any],
    constraint_status: dict[str, Any],
    kan_status: str,
    matrix_comparison: dict[str, Any],
) -> list[str]:
    claims: list[str] = []
    if paper_ready:
        claims.append(
            "FoVer and HaluEval/FEVER remain the only headline-eligible paper-v6 matrix rows from clean evidence."
        )
    if statuses["exp2886"] == "clean":
        claims.append(
            "Exp 2886 micro-panel telemetry is clean (logprobs available, fixed seed, reproducibility checksum) but explicitly makes no benchmark claim."
        )
    if fr11["fr11_scaleup_clean"]:
        claims.append(
            "FR-11 fast/slow memory cleanly separates from RecMem-causal and eager replay with non-tautological energy/duplicate metrics on n="
            f"{fr11.get('n_examples')} examples."
        )
    if constraint_status["vericot_frontier"]["ready"]:
        claims.append(
            "VeriCoT supports {} of {} candidate rows via deterministic z3-backed checks; cite only as bounded formal support, not broad exact verification.".format(
                constraint_status["vericot_frontier"].get("n_vericot_supported_rows"),
                constraint_status["vericot_frontier"].get("n_candidate_rows"),
            )
        )
    if constraint_status["structural_verifier"]["ready"]:
        claims.append(
            "Code structural-dependency contracts are ready as MBPP/HumanEval support metadata, not as pass@k claims."
        )
    if kan_status == "complexity_accounting_ready_no_hardware_or_analog_claim":
        claims.append(
            "KAN PWA/MILP complexity accounting (BoP, RM, NABS counts) is ready as a software-only proxy; no hardware or analog execution is claimed."
        )
    if statuses["exp2891"] == "pilot-only":
        claims.append(
            "CCTU constraint validation is pilot-only: 5 cases across 5 categories, no headline metric claim."
        )
    if statuses["exp2888"] == "taxonomy-only":
        claims.append(
            "TruthfulQA materializes as a local 100/200 error-taxonomy manifest; cite only the manifest, never accuracy or AUROC."
        )
    if matrix_comparison["new_pilot_only_rows_vs_v6"]:
        claims.append(
            "Matrix v7 adds MBPP and HumanEval as pilot-only rows beyond v6; do not cite them as headline benchmark rows."
        )
    return claims or ["No paper-v6 claim is safe from the available clean evidence."]


def _forbidden_claims(
    *,
    statuses: dict[str, str],
    matrix_comparison: dict[str, Any],
    kan_status: str,
) -> list[str]:
    claims = [
        "Do not cite MBPP or HumanEval as headline benchmark rows; matrix v7 marks them pilot_only.",
        "Do not cite TruthfulQA accuracy, AUROC, or generated-answer metrics; matrix v7 marks it taxonomy_only.",
        "Do not claim THRML, TSU, or any hardware acceleration from .273; the THRML branch remained blocked into the .274 backlog.",
    ]
    if statuses["exp2889"] == "flagged":
        claims.append(
            "Do not cite Exp 2889 MBPP/HumanEval generated-code outputs as evidence; the artifact is flagged DURATION_TOO_SHORT and yielded zero candidate passes."
        )
    if statuses["exp2885"] == "flagged":
        claims.append(
            "Do not cite Exp 2885 archive/activation artifact as compute evidence; it is flagged for adversarial-verify DURATION_TOO_SHORT and METHODOLOGY_MISSING despite being a pure aggregation task."
        )
    if not matrix_comparison["matrix_v7_adds_headline_evidence_beyond_v6"]:
        claims.append(
            "Do not claim matrix v7 added new headline-eligible rows beyond v6; v7 only added pilot-only and taxonomy-only support rows."
        )
    if kan_status != "complexity_accounting_ready_no_hardware_or_analog_claim":
        claims.append(
            "Do not cite KAN hardware or analog execution from .273; Exp 2893 explicitly makes no such claim."
        )
    return claims


def _top_3_next_actions(
    *,
    statuses: dict[str, str],
    paper_ready: bool,
    matrix_comparison: dict[str, Any],
    constraint_status: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    if statuses["exp2889"] == "flagged":
        actions.append(
            "Re-run Exp 2889 MBPP/HumanEval generated-code row under an adversarial-clean live SOTA GGUF budget (k>=8 candidates, duration_s>=60, reproducibility_checksum), or promote pilot-only rows to clean by another route."
        )
    if not matrix_comparison["matrix_v7_adds_headline_evidence_beyond_v6"]:
        actions.append(
            "Add at least one new headline-eligible row to matrix v8 (TruthfulQA generated-answer with InFi-Check labels, or a clean MBPP/HumanEval generated-code row) before claiming new paper-v6 headline lift."
        )
    if statuses["exp2885"] == "flagged":
        actions.append(
            "Set inference_substrate=aggregation_from_upstream_artifacts on the .274 archive task so adversarial-verify stops false-positive flagging the admin step."
        )
    if not paper_ready:
        actions.append(
            "Restore a clean matrix v7 with FoVer plus at least one other headline row before carrying paper-v6 readiness into .274."
        )
    if constraint_status["vericot_frontier"]["ready"]:
        vericot = constraint_status["vericot_frontier"]
        n_supported = vericot.get("n_vericot_supported_rows") or 0
        n_candidate = vericot.get("n_candidate_rows") or 0
        if n_candidate and n_supported / max(n_candidate, 1) < 0.1:
            actions.append(
                "Expand VeriCoT formal-support coverage beyond {}/{} rows before citing it as broad formal verification.".format(
                    n_supported, n_candidate
                )
            )
    if len(actions) < 3:
        actions.append(
            "Use matrix v7 safe claims only and keep flagged/pilot/taxonomy branches outside paper-v6 headline evidence."
        )
    return actions[:3]


def _compose_verdict(
    *,
    paper_ready: bool,
    clean_count: int,
    flagged_count: int,
    blocked_count: int,
    missing_count: int,
    pilot_count: int,
    taxonomy_count: int,
) -> str:
    return (
        "complete: .273 capstone synthesized; "
        f"paper_ready={str(paper_ready).lower()}; "
        f"clean_artifacts={clean_count}; flagged_artifacts={flagged_count}; "
        f"blocked_artifacts={blocked_count}; missing_artifacts={missing_count}; "
        f"pilot_only_artifacts={pilot_count}; taxonomy_only_artifacts={taxonomy_count}"
    )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2896: synthesize the milestone .273 paper claim boundary."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    payloads, present = _load_expected(root_path)
    statuses = _classify_all(payloads, present)

    matrix_payload = payloads["exp2894"]
    headline_rows = _headline_rows(statuses, matrix_payload)
    paper_ready = _paper_ready(statuses, matrix_payload)
    matrix_comparison = _matrix_v7_comparison(root_path, matrix_payload, statuses)
    fr11 = _continuous_self_learning_result(payloads, statuses)
    constraint_status = _constraint_benchmark_status(payloads, statuses)
    kan_status = _kan_complexity_status(payloads["exp2893"], statuses["exp2893"])

    clean_artifacts = _ids_with_status(statuses, "clean")
    flagged_artifacts = _ids_with_status(statuses, "flagged")
    blocked_artifacts = _ids_with_status(statuses, "blocked")
    missing_artifacts = _ids_with_status(statuses, "missing")
    pilot_only_artifacts = _ids_with_status(statuses, "pilot-only")
    taxonomy_only_artifacts = _ids_with_status(statuses, "taxonomy-only")

    safe_claims = _safe_claims(
        paper_ready=paper_ready,
        statuses=statuses,
        payloads=payloads,
        fr11=fr11,
        constraint_status=constraint_status,
        kan_status=kan_status,
        matrix_comparison=matrix_comparison,
    )
    forbidden_claims = _forbidden_claims(
        statuses=statuses,
        matrix_comparison=matrix_comparison,
        kan_status=kan_status,
    )
    top_3 = _top_3_next_actions(
        statuses=statuses,
        paper_ready=paper_ready,
        matrix_comparison=matrix_comparison,
        constraint_status=constraint_status,
    )
    end = time.perf_counter() if now_s is None else now_s

    return {
        "schema": SCHEMA,
        "artifact": "experiment_2896_capstone_v273",
        "honest_verdict": _compose_verdict(
            paper_ready=paper_ready,
            clean_count=len(clean_artifacts),
            flagged_count=len(flagged_artifacts),
            blocked_count=len(blocked_artifacts),
            missing_count=len(missing_artifacts),
            pilot_count=len(pilot_only_artifacts),
            taxonomy_count=len(taxonomy_only_artifacts),
        ),
        "milestone": MILESTONE,
        "paper_ready": paper_ready,
        "clean_artifacts": clean_artifacts,
        "flagged_artifacts": flagged_artifacts,
        "blocked_artifacts": blocked_artifacts,
        "missing_artifacts": missing_artifacts,
        "pilot_only_artifacts": pilot_only_artifacts,
        "taxonomy_only_artifacts": taxonomy_only_artifacts,
        "corrected_272_flags": _corrected_272_flags(payloads, statuses),
        "micro_panel_clean": statuses["exp2886"] == "clean"
        and payloads["exp2886"].get("micro_panel_clean") is True,
        "fr11_scaleup_clean": statuses["exp2887"] == "clean"
        and payloads["exp2887"].get("fr11_scaleup_clean") is True,
        "cross_corpus_matrix_built": statuses["exp2894"] == "clean"
        and matrix_payload.get("cross_corpus_matrix_built") is True,
        "headline_eligible_rows": headline_rows,
        "continuous_self_learning_result": fr11,
        "constraint_benchmark_status": constraint_status,
        "kan_complexity_status": kan_status,
        "paper_v6_safe_claims": safe_claims,
        "paper_v6_forbidden_claims": forbidden_claims,
        "top_3_next_actions": top_3,
        "matrix_v7_comparison": matrix_comparison,
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
        "inference_substrate": INFERENCE_SUBSTRATE,
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
    """Build and persist the Exp 2896 capstone JSON deliverable."""

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
