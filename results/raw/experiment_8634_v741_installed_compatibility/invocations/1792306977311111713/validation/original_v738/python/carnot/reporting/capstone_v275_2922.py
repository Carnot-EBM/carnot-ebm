"""Build the Exp 2922 milestone .275 capstone artifact.

Spec refs: REQ-REPORT-2922, SCENARIO-REPORT-2922.

This module is a pure closeout layer for milestone 2026.05.275. It reads the
already-written upstream artifacts, classifies each one by its own verdict,
required readiness fields, and unresolved flags, then emits the milestone-level
claim boundary the next roadmap needs.

Why this is aggregation-only: a capstone is not the place to rerun a model,
hardware board, sampler, or verifier. If an upstream branch is blocked or
flagged, the honest capstone behavior is to preserve that boundary explicitly
and keep the derived claim booleans scoped to the clean artifacts that actually
support them.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.milestone_capstone.v275"
MILESTONE = "2026.05.275"
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2922_capstone_v275.json")


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    required_fields: tuple[str, ...]


EXPECTED_ARTIFACTS: dict[str, SourceSpec] = {
    "exp2909": SourceSpec(
        "exp2909",
        Path("results/experiment_2909_archive_v274_activate_v275.json"),
        ("archive_ready",),
    ),
    "exp2910": SourceSpec(
        "exp2910",
        Path("results/experiment_2910_sota_code_generation_corrigendum_v2.json"),
        ("codegen_corrigendum_ready", "candidate_generation_clean"),
    ),
    "exp2911": SourceSpec(
        "exp2911",
        Path("results/experiment_2911_code_hallucination_taxonomy_verifier_v1.json"),
        ("code_hallucination_verifier_ready",),
    ),
    "exp2912": SourceSpec(
        "exp2912",
        Path("results/experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1.json"),
        ("same_basis_cpu_baseline_ready",),
    ),
    "exp2913": SourceSpec(
        "exp2913",
        Path("results/experiment_2913_kv260_hardware_cpu_claim_boundary_v1.json"),
        ("kv260_claim_boundary_ready", "same_basis_verified"),
    ),
    "exp2914": SourceSpec(
        "exp2914",
        Path("results/experiment_2914_gatemate_toolchain_preflight_v2.json"),
        ("gatemate_toolchain_ready",),
    ),
    "exp2915": SourceSpec(
        "exp2915",
        Path("results/experiment_2915_gatemate_n16_ising_tile_bitstream_build_v2.json"),
        ("gatemate_bitstream_built",),
    ),
    "exp2916": SourceSpec(
        "exp2916",
        Path("results/experiment_2916_thrml_kv260_sampler_parity_v1.json"),
        ("thrml_kv260_parity_ready", "no_tsu_hardware_claim"),
    ),
    "exp2917": SourceSpec(
        "exp2917",
        Path("results/experiment_2917_spilled_energy_logit_detector_micro_panel_v1.json"),
        ("spilled_energy_micro_panel_ready",),
    ),
    "exp2918": SourceSpec(
        "exp2918",
        Path("results/experiment_2918_fr11_verifiable_process_rewards_self_learning_v1.json"),
        ("online_self_learning_ready",),
    ),
    "exp2919": SourceSpec(
        "exp2919",
        Path("results/experiment_2919_constraintbench_mini_direct_optimization_v1.json"),
        ("constraintbench_mini_ready",),
    ),
    "exp2920": SourceSpec(
        "exp2920",
        Path("results/experiment_2920_opencomputer_style_state_verifier_harness_v1.json"),
        ("state_verifier_harness_ready",),
    ),
    "exp2921": SourceSpec(
        "exp2921",
        Path("results/experiment_2921_cross_corpus_matrix_v9_paper_boundary_v1.json"),
        ("cross_corpus_matrix_v9_built", "paper_claim_boundary_ready"),
    ),
}


HEADLINE_ROW_SOURCE: dict[str, tuple[str, ...]] = {
    "exp2910_sota_codegen": ("exp2910",),
    "exp2913_kv260_claim_boundary": ("exp2912", "exp2913"),
    "exp2918_fr11_process_rewards": ("exp2918",),
    "exp2920_state_verifier_harness": ("exp2920",),
}


def read_json(path: Path) -> dict[str, Any]:
    """Return a trusted JSON object or `{}` when the file is absent or malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def classify_artifact(exp_id: str, payload: dict[str, Any], present: bool) -> str:
    """REQ-REPORT-2922: classify one .275 source artifact.

    Flags take precedence because a ready boolean is not enough to make a row
    headline-usable when adversarial verification or a corrigendum has already
    identified a methodology gap. Diagnostic-only rows are separated from clean
    rows so simulator parity and micro-panels remain visible without becoming
    paper claims.
    """

    if not present or not payload:
        return "missing"
    if _has_flags(payload):
        return "flagged"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if payload.get("pilot_only") is True:
        return "pilot_only"
    if _is_diagnostic_only(exp_id, payload):
        return "diagnostic_only"
    if _is_clean(exp_id, payload):
        return "clean"
    return "blocked"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2922: synthesize the milestone .275 capstone."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    payloads, present = _load_expected(root_path)
    statuses = _classify_all(payloads, present)

    clean_artifacts = _ids_with_status(statuses, "clean")
    flagged_artifacts = _ids_with_status(statuses, "flagged")
    blocked_artifacts = _ids_with_status(statuses, "blocked")
    missing_artifacts = _ids_with_status(statuses, "missing")
    pilot_only_artifacts = _ids_with_status(statuses, "pilot_only")
    diagnostic_only_artifacts = _ids_with_status(statuses, "diagnostic_only")

    exp2910 = payloads["exp2910"]
    exp2911 = payloads["exp2911"]
    exp2912 = payloads["exp2912"]
    exp2913 = payloads["exp2913"]
    exp2914 = payloads["exp2914"]
    exp2915 = payloads["exp2915"]
    exp2916 = payloads["exp2916"]
    exp2917 = payloads["exp2917"]
    exp2918 = payloads["exp2918"]
    exp2919 = payloads["exp2919"]
    exp2920 = payloads["exp2920"]
    exp2921 = payloads["exp2921"]

    headline_rows = _headline_rows(exp2921)
    paper_ready = _paper_ready(exp2921, headline_rows, statuses)
    hardware_baselines_ready = statuses["exp2912"] == "clean" and statuses["exp2913"] == "clean"
    hardware_speedup_eligible = (
        hardware_baselines_ready and exp2913.get("hardware_speedup_claim_eligible") is True
    )
    sota_code_row_repaired = statuses["exp2910"] == "clean"
    fr11_self_learning_clean = statuses["exp2918"] == "clean"

    hardware_boundary = _hardware_claim_boundary(
        exp2912=exp2912,
        exp2913=exp2913,
        exp2914=exp2914,
        exp2915=exp2915,
        exp2916=exp2916,
        hardware_baselines_ready=hardware_baselines_ready,
        hardware_speedup_eligible=hardware_speedup_eligible,
    )
    codegen_boundary = _codegen_claim_boundary(
        exp2910=exp2910,
        exp2911=exp2911,
        exp2919=exp2919,
        sota_code_row_repaired=sota_code_row_repaired,
    )
    fr11_boundary = _fr11_claim_boundary(
        exp2918=exp2918,
        fr11_self_learning_clean=fr11_self_learning_clean,
    )

    end = time.perf_counter() if now_s is None else now_s
    artifact = {
        "schema": SCHEMA,
        "artifact": "experiment_2922_capstone_v275",
        "honest_verdict": _compose_verdict(
            paper_ready=paper_ready,
            hardware_speedup_eligible=hardware_speedup_eligible,
            sota_code_row_repaired=sota_code_row_repaired,
            fr11_self_learning_clean=fr11_self_learning_clean,
            clean_count=len(clean_artifacts),
            flagged_count=len(flagged_artifacts),
            blocked_count=len(blocked_artifacts),
            missing_count=len(missing_artifacts),
            diagnostic_count=len(diagnostic_only_artifacts),
        ),
        "milestone": MILESTONE,
        "paper_ready": paper_ready,
        "hardware_baselines_ready": hardware_baselines_ready,
        "hardware_speedup_claim_eligible": hardware_speedup_eligible,
        "sota_code_row_repaired": sota_code_row_repaired,
        "fr11_self_learning_clean": fr11_self_learning_clean,
        "clean_artifacts": clean_artifacts,
        "flagged_artifacts": flagged_artifacts,
        "blocked_artifacts": blocked_artifacts,
        "missing_artifacts": missing_artifacts,
        "pilot_only_artifacts": pilot_only_artifacts,
        "diagnostic_only_artifacts": diagnostic_only_artifacts,
        "headline_eligible_rows": headline_rows,
        "hardware_claim_boundary": hardware_boundary,
        "codegen_claim_boundary": codegen_boundary,
        "fr11_claim_boundary": fr11_boundary,
        "top_3_next_actions": _top_3_next_actions(statuses),
        "codegen_corrigendum_ready": exp2910.get("codegen_corrigendum_ready") is True,
        "code_hallucination_verifier_ready": exp2911.get("code_hallucination_verifier_ready")
        is True,
        "same_basis_cpu_baseline_ready": exp2912.get("same_basis_cpu_baseline_ready") is True,
        "kv260_claim_boundary_ready": exp2913.get("kv260_claim_boundary_ready") is True,
        "gatemate_toolchain_ready": exp2914.get("gatemate_toolchain_ready") is True,
        "gatemate_bitstream_built": exp2915.get("gatemate_bitstream_built") is True,
        "thrml_kv260_parity_ready": exp2916.get("thrml_kv260_parity_ready") is True,
        "no_tsu_hardware_claim": exp2916.get("no_tsu_hardware_claim") is True,
        "spilled_energy_micro_panel_ready": exp2917.get("spilled_energy_micro_panel_ready") is True,
        "online_self_learning_ready": exp2918.get("online_self_learning_ready") is True,
        "constraintbench_mini_ready": exp2919.get("constraintbench_mini_ready") is True,
        "state_verifier_harness_ready": exp2920.get("state_verifier_harness_ready") is True,
        "cross_corpus_matrix_v9_built": exp2921.get("cross_corpus_matrix_v9_built") is True,
        "paper_claim_boundary_ready": exp2921.get("paper_claim_boundary_ready") is True,
        "source_artifact_status": _source_status(payloads, present, statuses),
        "cited_upstream_artifacts": _cited_upstream_artifacts(root_path, present),
        "files_not_modified": [
            "scripts/research_conductor.py",
            "ops/status.md",
            "ops/changelog.md",
            "_bmad/traceability.md",
        ],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, end - start), 6),
        "run_date": RUN_DATE,
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2922 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def _load_expected(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    payloads: dict[str, dict[str, Any]] = {}
    present: dict[str, bool] = {}
    for exp_id, spec in EXPECTED_ARTIFACTS.items():
        path = root / spec.path
        present[exp_id] = path.is_file()
        payloads[exp_id] = read_json(path) if present[exp_id] else {}
    return payloads, present


def _classify_all(payloads: dict[str, dict[str, Any]], present: dict[str, bool]) -> dict[str, str]:
    return {
        exp_id: classify_artifact(exp_id, payloads[exp_id], present[exp_id])
        for exp_id in EXPECTED_ARTIFACTS
    }


def _ids_with_status(statuses: dict[str, str], wanted: str) -> list[str]:
    return [exp_id for exp_id in EXPECTED_ARTIFACTS if statuses.get(exp_id) == wanted]


def _has_flags(payload: dict[str, Any]) -> bool:
    if payload.get("flagged_adversarial") is True:
        return True
    if payload.get("adversarial_verify_passed") is False:
        return True
    for key in ("corrigendum_pending", "adversarial_verify_flags"):
        value = payload.get(key)
        if isinstance(value, list) and bool(value):
            return True
    summary = payload.get("adversarial_verify_summary")
    return isinstance(summary, dict) and int(summary.get("flag_count") or 0) > 0


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _terminal_success(verdict: object) -> bool:
    if not isinstance(verdict, str):
        return False
    return verdict.strip().startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    )


def _is_clean(exp_id: str, payload: dict[str, Any]) -> bool:
    if not _terminal_success(payload.get("honest_verdict")):
        return False
    if not all(payload.get(field) is True for field in EXPECTED_ARTIFACTS[exp_id].required_fields):
        return False
    if exp_id == "exp2910":
        return (
            payload.get("legacy_smoke_only") is not True
            and _is_number(payload.get("aggregate_pass_at_1"))
            and _is_number(payload.get("aggregate_pass_at_k"))
            and payload.get("pass_at_k_exceeds_pass_at_1") is True
        )
    if exp_id == "exp2918":
        update_happened = (
            payload.get("online_update_performed") is True
            or payload.get("replay_scheduler_updated") is True
        )
        return (
            update_happened
            and payload.get("model_weights_mutated") is False
            and _is_number(payload.get("forgetting_rate"))
        )
    if exp_id == "exp2920":
        return payload.get("llm_judge_used") is False
    return True


def _is_diagnostic_only(exp_id: str, payload: dict[str, Any]) -> bool:
    if not _terminal_success(payload.get("honest_verdict")):
        return False
    if exp_id == "exp2916":
        return (
            payload.get("thrml_kv260_parity_ready") is True
            and payload.get("no_tsu_hardware_claim") is True
            and payload.get("inference_substrate") == "simulator_parity"
        )
    if exp_id == "exp2917":
        return (
            payload.get("spilled_energy_micro_panel_ready") is True
            and payload.get("benchmark_claim_made") is False
            and payload.get("claim_boundary") == "diagnostic_only_no_benchmark_claim"
        )
    return False


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _headline_rows(matrix_payload: dict[str, Any]) -> list[str]:
    rows = matrix_payload.get("headline_eligible_rows")
    return [row for row in rows if isinstance(row, str)] if isinstance(rows, list) else []


def _paper_ready(
    matrix_payload: dict[str, Any],
    headline_rows: list[str],
    statuses: dict[str, str],
) -> bool:
    if matrix_payload.get("paper_claim_boundary_ready") is not True:
        return False
    if matrix_payload.get("cross_corpus_matrix_v9_built") is not True:
        return False
    for row_id in headline_rows:
        for exp_id in HEADLINE_ROW_SOURCE.get(row_id, ()):
            if statuses.get(exp_id) != "clean":
                return False
    return bool(headline_rows)


def _hardware_claim_boundary(
    *,
    exp2912: dict[str, Any],
    exp2913: dict[str, Any],
    exp2914: dict[str, Any],
    exp2915: dict[str, Any],
    exp2916: dict[str, Any],
    hardware_baselines_ready: bool,
    hardware_speedup_eligible: bool,
) -> dict[str, Any]:
    return {
        "same_basis_cpu_baseline_ready": exp2912.get("same_basis_cpu_baseline_ready") is True,
        "kv260_claim_boundary_ready": exp2913.get("kv260_claim_boundary_ready") is True,
        "same_basis_verified": exp2913.get("same_basis_verified") is True,
        "hardware_baselines_ready": hardware_baselines_ready,
        "hardware_speedup_claim_eligible": hardware_speedup_eligible,
        "speedup_claim_made": hardware_speedup_eligible
        and exp2913.get("speedup_claim_made") is True,
        "speedup_ratio_median_by_sample_count": exp2913.get(
            "speedup_ratio_median_by_sample_count",
            {},
        ),
        "paper_claim_boundary": exp2913.get("paper_claim_boundary"),
        "gatemate_toolchain_ready": exp2914.get("gatemate_toolchain_ready") is True,
        "gatemate_bitstream_built": exp2915.get("gatemate_bitstream_built") is True,
        "thrml_kv260_parity_ready": exp2916.get("thrml_kv260_parity_ready") is True,
        "no_tsu_hardware_claim": exp2916.get("no_tsu_hardware_claim") is True,
        "claim_boundary": (
            "A numeric KV260/CPU speedup claim is eligible only for the matched"
            " n=64 sparse Ising workload. GateMate remains non-claimable until"
            " a bitstream artifact exists, and THRML remains simulator-only with"
            " no TSU hardware claim."
        ),
    }


def _codegen_claim_boundary(
    *,
    exp2910: dict[str, Any],
    exp2911: dict[str, Any],
    exp2919: dict[str, Any],
    sota_code_row_repaired: bool,
) -> dict[str, Any]:
    return {
        "codegen_corrigendum_ready": exp2910.get("codegen_corrigendum_ready") is True,
        "sota_code_row_repaired": sota_code_row_repaired,
        "candidate_generation_clean": exp2910.get("candidate_generation_clean") is True,
        "aggregate_pass_at_1": exp2910.get("aggregate_pass_at_1"),
        "aggregate_pass_at_k": exp2910.get("aggregate_pass_at_k"),
        "pass_at_k_exceeds_pass_at_1": exp2910.get("pass_at_k_exceeds_pass_at_1") is True,
        "code_hallucination_verifier_ready": exp2911.get("code_hallucination_verifier_ready")
        is True,
        "constraintbench_mini_ready": exp2919.get("constraintbench_mini_ready") is True,
        "remaining_methodology_risks": _methodology_risks(
            {
                "exp2910": exp2910,
                "exp2911": exp2911,
                "exp2919": exp2919,
            }
        ),
        "claim_boundary": (
            "The SOTA code row is repaired only for Exp 2910's bounded pass@1"
            " and pass@k metrics. Flagged taxonomy or ConstraintBench evidence"
            " remains non-headline until its own methodology flags are cleared."
        ),
    }


def _fr11_claim_boundary(
    *,
    exp2918: dict[str, Any],
    fr11_self_learning_clean: bool,
) -> dict[str, Any]:
    return {
        "online_self_learning_ready": exp2918.get("online_self_learning_ready") is True,
        "fr11_self_learning_clean": fr11_self_learning_clean,
        "online_update_performed": exp2918.get("online_update_performed") is True,
        "replay_scheduler_updated": exp2918.get("replay_scheduler_updated") is True,
        "model_weights_mutated": exp2918.get("model_weights_mutated") is True,
        "delta_overall": exp2918.get("delta_overall"),
        "delta_energy_proxy": exp2918.get("delta_energy_proxy"),
        "forgetting_rate": exp2918.get("forgetting_rate"),
        "hardware_replay_used": exp2918.get("hardware_replay_used") is True,
        "claim_boundary": (
            "FR-11 has a clean continuous self-learning row when verifier"
            " process rewards update the online replay/scheduler state, report"
            " delta and forgetting metrics, and do not mutate model weights."
        ),
    }


def _methodology_risks(payloads: dict[str, dict[str, Any]]) -> list[str]:
    risks: list[str] = []
    for exp_id, payload in payloads.items():
        for item in payload.get("corrigendum_pending") or []:
            if isinstance(item, dict):
                risks.append(
                    f"{exp_id}:{item.get('kind', 'unknown')}:{item.get('severity', 'unknown')}"
                )
        if payload.get("adversarial_verify_passed") is False:
            risks.append(f"{exp_id}:adversarial_verify_passed:false")
        if payload.get("flagged_adversarial") is True and not payload.get("corrigendum_pending"):
            risks.append(f"{exp_id}:flagged_adversarial:true")
    return risks


def _top_3_next_actions(statuses: dict[str, str]) -> list[str]:
    actions: list[str] = []
    if statuses.get("exp2921") == "flagged":
        actions.append(
            "Repair matrix v9 aggregation-only adversarial-verify metadata so"
            " aggregation_from_upstream_artifacts is honored before relying on"
            " the matrix v9 artifact as an unflagged closeout source."
        )
    if statuses.get("exp2911") == "flagged":
        actions.append(
            "Clear the code hallucination taxonomy verifier methodology flags"
            " or rerun it with explicit deterministic-verifier provenance so"
            " its ready boolean can become headline-supporting evidence."
        )
    if statuses.get("exp2919") == "flagged":
        actions.append(
            "Rerun ConstraintBench mini with reproducibility checksum,"
            " non-tautological feasibility/syntax metrics, and live-inference"
            " duration/provenance that passes adversarial verification."
        )
    if statuses.get("exp2914") == "blocked" or statuses.get("exp2915") == "missing":
        actions.append(
            "Provision the GateMate toolchain, rerun the n=16 Ising tile build,"
            " and produce a bitstream SHA before any flash or hardware claim."
        )
    while len(actions) < 3:
        actions.append(
            "Keep .276 claim boundaries forward-only: promote only artifacts"
            " that are clean, unflagged, and directly support a bounded row."
        )
    return actions[:3]


def _source_status(
    payloads: dict[str, dict[str, Any]],
    present: dict[str, bool],
    statuses: dict[str, str],
) -> dict[str, dict[str, Any]]:
    return {
        exp_id: {
            "path": str(spec.path),
            "present": present[exp_id],
            "status": statuses[exp_id],
            "honest_verdict": payloads[exp_id].get("honest_verdict"),
        }
        for exp_id, spec in EXPECTED_ARTIFACTS.items()
    }


def _cited_upstream_artifacts(root: Path, present: dict[str, bool]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for exp_id, spec in EXPECTED_ARTIFACTS.items():
        path = root / spec.path
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if present[exp_id] else None
        citations.append(
            {
                "experiment_id": exp_id,
                "artifact_path": str(spec.path),
                "sha256": sha256,
                "present": present[exp_id],
            }
        )
    return citations


def _compose_verdict(
    *,
    paper_ready: bool,
    hardware_speedup_eligible: bool,
    sota_code_row_repaired: bool,
    fr11_self_learning_clean: bool,
    clean_count: int,
    flagged_count: int,
    blocked_count: int,
    missing_count: int,
    diagnostic_count: int,
) -> str:
    return (
        "complete: .275 capstone synthesized; "
        f"paper_ready={str(paper_ready).lower()}; "
        f"hardware_speedup_claim_eligible={str(hardware_speedup_eligible).lower()}; "
        f"sota_code_row_repaired={str(sota_code_row_repaired).lower()}; "
        f"fr11_self_learning_clean={str(fr11_self_learning_clean).lower()}; "
        f"clean_artifacts={clean_count}; flagged_artifacts={flagged_count}; "
        f"blocked_artifacts={blocked_count}; missing_artifacts={missing_count}; "
        f"diagnostic_only_artifacts={diagnostic_count}"
    )


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())
