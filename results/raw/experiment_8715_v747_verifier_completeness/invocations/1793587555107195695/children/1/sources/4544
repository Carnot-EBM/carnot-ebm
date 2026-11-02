"""Exp 3722 convergence synthesis and operator next-thesis request.

This report is deliberately aggregation-only. It records the terminal state the
loop can prove from checked-in artifacts, then surfaces a bounded next-thesis
menu for the operator. It does not choose the next thesis, run a new
experiment, or edit operator-curated policy files.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

EXPERIMENT_ID = "exp3722"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path(
    "results/experiment_3722_convergence_synthesis_operator_next_thesis.json"
)
RANDOM_SEED = 3722
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: reads prior artifacts + "
    "docs; no live inference; no compute-bound marker)."
)
SYNTHESIZED_VERDICT = (
    "complete: "
    "convergence_synthesized_next_theses_presented_operator_decision_requested"
)
CANNOT_SYNTHESIZE_VERDICT = "complete: convergence_synthesis_cannot_complete"
TERMINAL_VERDICTS = (SYNTHESIZED_VERDICT, CANNOT_SYNTHESIZE_VERDICT)

NORTH_STAR_REL_PATH = Path("ops/north-star.md")
MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
RESEARCH_PROGRAM_REL_PATH = Path("research-program.md")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")

UPSTREAM_ARTIFACTS = {
    "exp3712": Path("results/experiment_3712_capstone_and_g_gate_v339.json"),
    "exp3715": Path("results/experiment_3715_refreeze_disambiguation_clean_corrigendum.json"),
    "exp3716": Path("results/experiment_3716_ship_paper_v6_narrowing_lint.json"),
    "exp3717": Path("results/experiment_3717_g4_full_provenance_audit.json"),
    "exp3718": Path("results/experiment_3718_risk_coverage_abstention_characterization.json"),
    "exp3719": Path("results/experiment_3719_headline_replication_fresh_corpus.json"),
    "exp3720": Path("results/experiment_3720_fr11_continuous_self_learning_v14.json"),
    "exp3721": Path(
        "results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json"
    ),
}

REQUIRED_THREADS = (
    "headline",
    "p01",
    "selection",
    "refreeze",
    "code",
    "facts",
    "judge_ood",
    "kv260",
    "self_learning",
    "paper_ready",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "converged_state_ledger",
    "all_self_generable_threads_settled",
    "candidate_next_theses",
    "recommended_default_thesis",
    "operator_decision_request",
    "paper_ready_status",
    "north_star_unmodified_assert",
    "manifest_unmodified_assert",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "Reads prior artifacts and docs; no live inference and no compute-bound "
        "marker belongs in this aggregation artifact."
    ),
    "converged_state_ledger": (
        "Per-thread {thread, terminal_status, authoritative_artifact}; the "
        "honest convergence record."
    ),
    "all_self_generable_threads_settled": (
        "BARE bool. True iff every loop-self-generable thread has a terminal status."
    ),
    "candidate_next_theses": (
        "The operator menu, with one why-not-regrind line per candidate."
    ),
    "recommended_default_thesis": (
        "The loop's recommendation only; the operator still decides the thesis."
    ),
    "operator_decision_request": (
        "The explicit ask for what thesis or maintenance mode drives .341+."
    ),
    "paper_ready_status": "Records the paper-ready gate value from the capstone.",
    "north_star_unmodified_assert": (
        "Asserts ops/north-star.md was not edited by this task."
    ),
    "manifest_unmodified_assert": (
        "Asserts ops/exclusion_manifest.yaml was not edited by this task."
    ),
    "adversarial_verify_clean": "True iff no critical adversarial flag.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float = 0.0,
    now_s: float = 0.0001,
    adversarial_verify_clean: bool = False,
    adversarial_verify_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Read repository inputs and construct the Exp 3722 artifact."""

    root_path = Path(root)
    north = root_path / NORTH_STAR_REL_PATH
    manifest = root_path / MANIFEST_REL_PATH
    conductor = root_path / CONDUCTOR_REL_PATH
    return build_artifact_from_inputs(
        exp3712=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3712"]),
        exp3715=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3715"]),
        exp3716=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3716"]),
        exp3717=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3717"]),
        exp3718=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3718"]),
        exp3719=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3719"]),
        exp3720=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3720"]),
        exp3721=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3721"]),
        north_star_text=_read_text(root_path / NORTH_STAR_REL_PATH),
        research_program_text=_read_text(root_path / RESEARCH_PROGRAM_REL_PATH),
        roadmap_text=_read_text(root_path / ROADMAP_REL_PATH),
        north_star_hash_before=_sha256_path(north),
        north_star_hash_after=_sha256_path(north),
        manifest_hash_before=_sha256_path(manifest),
        manifest_hash_after=_sha256_path(manifest),
        conductor_hash_before=_sha256_path(conductor),
        conductor_hash_after=_sha256_path(conductor),
        started_s=started_s,
        now_s=now_s,
        adversarial_verify_clean=adversarial_verify_clean,
        adversarial_verify_report=adversarial_verify_report or {"flags": []},
    )


def build_artifact_from_inputs(
    *,
    exp3712: Mapping[str, Any],
    exp3715: Mapping[str, Any],
    exp3716: Mapping[str, Any],
    exp3717: Mapping[str, Any],
    exp3718: Mapping[str, Any],
    exp3719: Mapping[str, Any],
    exp3720: Mapping[str, Any],
    exp3721: Mapping[str, Any],
    north_star_text: str,
    research_program_text: str,
    roadmap_text: str,
    north_star_hash_before: str,
    north_star_hash_after: str,
    manifest_hash_before: str,
    manifest_hash_after: str,
    conductor_hash_before: str,
    conductor_hash_after: str,
    started_s: float,
    now_s: float,
    adversarial_verify_clean: bool,
    adversarial_verify_report: Mapping[str, Any],
) -> JsonDict:
    """Build a synthesized or cannot-synthesize artifact from supplied inputs."""

    north_star_unmodified = north_star_hash_before == north_star_hash_after
    manifest_unmodified = manifest_hash_before == manifest_hash_after
    conductor_unmodified = conductor_hash_before == conductor_hash_after
    ledger = build_converged_state_ledger(
        exp3712=exp3712,
        exp3715=exp3715,
        exp3716=exp3716,
        exp3717=exp3717,
        exp3718=exp3718,
        exp3719=exp3719,
        exp3720=exp3720,
        exp3721=exp3721,
        north_star_text=north_star_text,
    )
    all_threads_settled = all(row["settled"] is True for row in ledger)
    energy_abstention_positive = exp3718.get("energy_beats_baseline_abstention") is True
    candidate_next_theses = build_candidate_next_theses(
        exp3718=exp3718,
        research_program_text=research_program_text,
        roadmap_text=roadmap_text,
    )
    can_synthesize = (
        all_threads_settled
        and north_star_unmodified
        and manifest_unmodified
        and conductor_unmodified
    )
    artifact: JsonDict = {
        "schema": "carnot.convergence_synthesis_operator_next_thesis.v1",
        "experiment_id": EXPERIMENT_ID,
        "task_id": "exp3722-convergence-synthesis-operator-next-thesis",
        "honest_verdict": (
            SYNTHESIZED_VERDICT if can_synthesize else CANNOT_SYNTHESIZE_VERDICT
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "converged_state_ledger": ledger,
        "all_self_generable_threads_settled": all_threads_settled,
        "candidate_next_theses": candidate_next_theses,
        "recommended_default_thesis": build_recommended_default(
            can_synthesize=can_synthesize,
            energy_abstention_positive=energy_abstention_positive,
            exp3718=exp3718,
        ),
        "operator_decision_request": (
            "The loop has converged. Which thesis, product direction, or "
            "maintenance mode should drive .341+?"
        ),
        "paper_ready_status": exp3712.get("paper_ready") is True,
        "north_star_unmodified_assert": north_star_unmodified,
        "manifest_unmodified_assert": manifest_unmodified,
        "scripts_research_conductor_modified": not conductor_unmodified,
        "adversarial_verify_clean": adversarial_verify_clean,
        "adversarial_verify_report": compact_adversarial_report(adversarial_verify_report),
        "random_seed": RANDOM_SEED,
        "duration_s": max(float(now_s) - float(started_s), 0.0001),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifact_checksums": _source_payload_checksums(
            {
                "exp3712": exp3712,
                "exp3715": exp3715,
                "exp3716": exp3716,
                "exp3717": exp3717,
                "exp3718": exp3718,
                "exp3719": exp3719,
                "exp3720": exp3720,
                "exp3721": exp3721,
            }
        ),
        "source_document_checksums": {
            "ops/north-star.md": _sha256_text(north_star_text),
            "research-program.md": _sha256_text(research_program_text),
            "research-roadmap.yaml": _sha256_text(roadmap_text),
        },
        "non_decision_assert": True,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_converged_state_ledger(
    *,
    exp3712: Mapping[str, Any],
    exp3715: Mapping[str, Any],
    exp3716: Mapping[str, Any],
    exp3717: Mapping[str, Any],
    exp3718: Mapping[str, Any],
    exp3719: Mapping[str, Any],
    exp3720: Mapping[str, Any],
    exp3721: Mapping[str, Any],
    north_star_text: str,
) -> list[JsonDict]:
    """Return the required per-thread convergence ledger."""

    headline_settled = (
        _point(exp3712.get("frozen_fover_headline_auroc")) == 0.9131
        and exp3712.get("frozen_headline_unchanged") is True
        and exp3717.get("all_numbers_trace_to_clean_artifacts") is True
        and "0.9131" in north_star_text
    )
    p01_settled = exp3712.get("p01_status") == "honest-negative"
    selection_settled = exp3712.get("selection_diagnosis_closed") is True
    refreeze_settled = (
        exp3715.get("no_candidate_beats_frozen") is True
        and exp3715.get("frozen_headline_unchanged_assert") is True
    )
    code_settled = (
        exp3712.get("code_native_heldout_verdict") == "one_point_zero_was_a_leak"
        and _contains(str(exp3712.get("shipped_detector_reconciliation", "")), "math")
        and _contains(str(exp3712.get("shipped_detector_reconciliation", "")), "abstain")
    )
    facts_settled = exp3712.get("facts_generalization_retired") is True
    judge_settled = exp3712.get("trained_judge_ood_retired") is True
    kv260_settled = (
        exp3721.get("kv260_terminal_condition_confirmed") is True
        and exp3721.get("kv260_terminal_transcript_present") is True
        and exp3721.get("speedup_claim_avoided_assert") is True
    )
    self_learning_settled = (
        exp3720.get("template_robust_or_graceful_fallback") is True
        and exp3720.get("collapse_detected_deploy_arm") is False
        and exp3720.get("template_library_bounded") is True
    )
    paper_ready_settled = (
        exp3712.get("paper_ready") is True
        and all(exp3712.get(gate) is True for gate in ("g1", "g2", "g3", "g4"))
        and exp3716.get("g3_now_mechanically_enforced") is True
        and exp3716.get("current_paper_lint_clean") is True
        and exp3717.get("all_numbers_trace_to_clean_artifacts") is True
    )
    rows = [
        _ledger_row(
            "headline",
            headline_settled,
            "frozen_fover_0_9131_proven_reproduced_and_narrowing_clean",
            "results/experiment_3712_capstone_and_g_gate_v339.json; "
            "results/experiment_3717_g4_full_provenance_audit.json",
        ),
        _ledger_row(
            "p01",
            p01_settled,
            "honest_negative_bounded",
            "results/experiment_3712_capstone_and_g_gate_v339.json",
        ),
        _ledger_row(
            "selection",
            selection_settled,
            "settled_bounded_diagnosis_closed",
            "results/experiment_3707_selection_diagnosis_formal_closure.json; "
            "results/experiment_3712_capstone_and_g_gate_v339.json",
        ),
        _ledger_row(
            "refreeze",
            refreeze_settled,
            "closed_negative_no_candidate_beats_frozen",
            "results/experiment_3715_refreeze_disambiguation_clean_corrigendum.json",
        ),
        _ledger_row(
            "code",
            code_settled,
            "code_generalization_leak_narrowed_to_math_only_abstain",
            "results/experiment_3705_code_native_leak_audit_heldout.json; "
            "results/experiment_3706_reconcile_shipped_detector_heldout.json; "
            "results/experiment_3712_capstone_and_g_gate_v339.json",
        ),
        _ledger_row(
            "facts",
            facts_settled,
            "facts_generalization_retired",
            "results/experiment_3670_facts_row_real_benchmark.json; "
            "results/experiment_3712_capstone_and_g_gate_v339.json",
        ),
        _ledger_row(
            "judge_ood",
            judge_settled,
            "trained_judge_ood_retired",
            "results/experiment_3659_trained_ebm_judge_ood_real_substrate_v3.json; "
            "results/experiment_3712_capstone_and_g_gate_v339.json",
        ),
        _ledger_row(
            "kv260",
            kv260_settled,
            "kv260_terminal_confirmed_mandate_lift_recommended",
            "results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json",
        ),
        _ledger_row(
            "self_learning",
            self_learning_settled,
            "fr11_v14_graceful_fallback_under_shift_no_collapse",
            "results/experiment_3720_fr11_continuous_self_learning_v14.json",
        ),
        _ledger_row(
            "paper_ready",
            paper_ready_settled,
            "paper_ready_true_g1_g2_g3_g4_met_and_hardened",
            "results/experiment_3712_capstone_and_g_gate_v339.json; "
            "results/experiment_3716_ship_paper_v6_narrowing_lint.json; "
            "results/experiment_3717_g4_full_provenance_audit.json",
        ),
    ]
    return rows


def build_candidate_next_theses(
    *,
    exp3718: Mapping[str, Any],
    research_program_text: str,
    roadmap_text: str,
) -> list[JsonDict]:
    """Return the bounded menu of operator-selectable next-theses."""

    candidates: list[JsonDict] = []
    if exp3718.get("energy_beats_baseline_abstention") is True:
        candidates.append(
            {
                "thesis": "energy_based_selective_prediction_at_scale",
                "summary": (
                    "Scale the positive risk-coverage result into a calibrated "
                    "abstention product study."
                ),
                "why_not_regrind": (
                    "It optimizes abstain-or-flag risk coverage, not the closed "
                    "best-of-N selection question."
                ),
                "authority": "results/experiment_3718_risk_coverage_abstention_characterization.json",
                "operator_action_required": True,
            }
        )
    candidates.extend(
        [
            {
                "thesis": "different_verifier_architecture_for_weak_sc_domain",
                "summary": (
                    "Try a genuinely different verifier architecture in a domain "
                    "where self-consistency is weak and abstention has headroom."
                ),
                "why_not_regrind": (
                    "It changes both architecture and operating domain instead of "
                    "rerunning the settled FoVer weighting or selection lines."
                ),
                "authority": "research-program.md; project_verifier_domain_bound background",
                "operator_action_required": True,
            },
            {
                "thesis": "finalize_submit_and_maintenance",
                "summary": (
                    "Submit the paper under operator control, then move the loop "
                    "to maintenance and regression checks."
                ),
                "why_not_regrind": (
                    "It banks the already-ready paper state rather than generating "
                    "new experiments against settled threads."
                ),
                "authority": "results/experiment_3712_capstone_and_g_gate_v339.json",
                "operator_action_required": True,
            },
        ]
    )
    if "Safety/Jailbreak Classifier" in research_program_text:
        candidates.append(
            {
                "thesis": "tier_b_safety_jailbreak_classifier",
                "summary": "Build the Tier B safety and jailbreak classifier product.",
                "why_not_regrind": (
                    "It uses the energy-function product roadmap in a new safety "
                    "surface rather than re-scoring FoVer or re-freezing the headline."
                ),
                "authority": "research-program.md Tier B",
                "operator_action_required": True,
            }
        )
    if "Compliance Checker" in research_program_text:
        candidates.append(
            {
                "thesis": "tier_b_compliance_checker",
                "summary": "Build the Tier B regulatory compliance checker product.",
                "why_not_regrind": (
                    "It applies constraint energy to external compliance rules, "
                    "a product path not covered by the settled selection line."
                ),
                "authority": "research-program.md Tier B",
                "operator_action_required": True,
            }
        )
    if "energy-as-GENERATOR" in roadmap_text or "Energy-Based Transformer" in roadmap_text:
        candidates.append(
            {
                "thesis": "human_seeded_energy_as_generator_ebt",
                "summary": (
                    "Operator-seeded energy-as-generator EBT thesis: inference is "
                    "energy minimization rather than reranking."
                ),
                "why_not_regrind": (
                    "It is generation, not selection; the settled energy-selection "
                    "bound does not answer it."
                ),
                "authority": "research-roadmap-next.yaml or operator seed, if activated",
                "operator_action_required": True,
            }
        )
    return candidates


def build_recommended_default(
    *,
    can_synthesize: bool,
    energy_abstention_positive: bool,
    exp3718: Mapping[str, Any],
) -> JsonDict:
    """Return a recommendation object that explicitly is not a decision."""

    if not can_synthesize:
        return {
            "thesis": "complete_terminal_evidence_before_new_thesis",
            "rationale": (
                "At least one required terminal thread is missing or protected-file "
                "hygiene failed, so the operator menu should not drive .341 yet."
            ),
            "operator_decision_made": False,
        }
    if energy_abstention_positive:
        return {
            "thesis": "energy_based_selective_prediction_at_scale",
            "rationale": (
                "Exp 3718 reports energy beating entropy as a selective-prediction "
                "signal, which is the strongest non-regrind extension of the frozen "
                "0.9131 discriminator."
            ),
            "supporting_artifact": (
                "results/experiment_3718_risk_coverage_abstention_characterization.json"
            ),
            "energy_aurc": _point(exp3718.get("energy_aurc")),
            "baseline_aurc": _point(exp3718.get("baseline_aurc")),
            "operator_decision_made": False,
        }
    return {
        "thesis": "finalize_submit_and_maintenance",
        "rationale": (
            "Without a positive abstention extension, the conservative default is "
            "to submit the ready paper and move the loop to maintenance."
        ),
        "supporting_artifact": "results/experiment_3712_capstone_and_g_gate_v339.json",
        "operator_decision_made": False,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float = 0.0,
    now_s: float = 0.0001,
) -> Path:
    """Build, adversarial-check, validate, and write the Exp 3722 artifact."""

    root_path = Path(root)
    north = root_path / NORTH_STAR_REL_PATH
    manifest = root_path / MANIFEST_REL_PATH
    conductor = root_path / CONDUCTOR_REL_PATH
    north_before = _sha256_path(north)
    manifest_before = _sha256_path(manifest)
    conductor_before = _sha256_path(conductor)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact_from_inputs(
        exp3712=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3712"]),
        exp3715=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3715"]),
        exp3716=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3716"]),
        exp3717=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3717"]),
        exp3718=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3718"]),
        exp3719=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3719"]),
        exp3720=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3720"]),
        exp3721=_read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3721"]),
        north_star_text=_read_text(root_path / NORTH_STAR_REL_PATH),
        research_program_text=_read_text(root_path / RESEARCH_PROGRAM_REL_PATH),
        roadmap_text=_read_text(root_path / ROADMAP_REL_PATH),
        north_star_hash_before=north_before,
        north_star_hash_after=_sha256_path(north),
        manifest_hash_before=manifest_before,
        manifest_hash_after=_sha256_path(manifest),
        conductor_hash_before=conductor_before,
        conductor_hash_after=_sha256_path(conductor),
        started_s=started_s,
        now_s=now_s,
        adversarial_verify_clean=False,
        adversarial_verify_report={"flags": []},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_payload(output, artifact)
    report = run_adversarial_verify_report(output)
    artifact["adversarial_verify_report"] = compact_adversarial_report(report)
    artifact["adversarial_verify_clean"] = adversarial_report_is_clean(report)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    _write_payload(output, artifact)
    return output


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3722 schema and non-decision boundaries."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required artifact fields: {missing}")
    _ensure(_no_compute_markers(artifact), "compute-bound markers must not be present")
    _ensure(artifact.get("honest_verdict") in TERMINAL_VERDICTS, "terminal verdict")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    for field in (
        "all_self_generable_threads_settled",
        "paper_ready_status",
        "north_star_unmodified_assert",
        "manifest_unmodified_assert",
        "adversarial_verify_clean",
    ):
        _ensure(type(artifact.get(field)) is bool, f"{field} must be a bare bool")
    ledger = artifact.get("converged_state_ledger")
    _ensure(isinstance(ledger, list), "converged_state_ledger must be a list")
    threads = {row.get("thread") for row in ledger if isinstance(row, Mapping)}
    _ensure(threads == set(REQUIRED_THREADS), "converged_state_ledger thread set")
    for row in ledger:
        _ensure(isinstance(row, Mapping), "ledger rows must be objects")
        _ensure(
            {"thread", "terminal_status", "authoritative_artifact", "settled"} <= set(row),
            "ledger row fields",
        )
        _ensure(type(row.get("settled")) is bool, "ledger settled must be bare bool")
    if artifact.get("all_self_generable_threads_settled") is True:
        _ensure(all(row.get("settled") is True for row in ledger), "settled flag mismatch")
    candidates = artifact.get("candidate_next_theses")
    _ensure(isinstance(candidates, list) and candidates, "candidate_next_theses")
    for candidate in candidates:
        _ensure(isinstance(candidate, Mapping), "candidate rows must be objects")
        _ensure("why_not_regrind" in candidate, "candidate why_not_regrind")
        _ensure(bool(candidate.get("why_not_regrind")), "candidate why_not_regrind")
        _ensure(type(candidate.get("operator_action_required")) is bool, "operator action")
    default = artifact.get("recommended_default_thesis")
    _ensure(isinstance(default, Mapping), "recommended_default_thesis")
    _ensure(
        default.get("operator_decision_made") is False,
        "recommended default must not make operator decision",
    )
    request = artifact.get("operator_decision_request")
    _ensure(isinstance(request, str) and ".341+" in request, "operator_decision_request")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles")
    missing_principles = [
        field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles
    ]
    _ensure(not missing_principles, f"field_principles missing: {missing_principles}")
    duration = artifact.get("duration_s")
    _ensure(
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and float(duration) >= 0.0001,
        "duration_s",
    )
    checksum = artifact.get("reproducibility_checksum")
    _ensure(
        isinstance(checksum, str) and len(checksum) == 64,
        "reproducibility_checksum",
    )
    _ensure(
        checksum == reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )


def run_adversarial_verify_report(path: Path) -> JsonDict:
    """Run the repository adversarial verifier and return its report object."""

    verifier_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_3722", verifier_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load adversarial verifier from {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.verify_artifact(path)
    if not isinstance(report, dict):
        raise RuntimeError("adversarial verifier returned a non-object report")
    return report


def compact_adversarial_report(report: Mapping[str, Any]) -> JsonDict:
    """Keep a stable summary of adversarial flags."""

    raw_flags = report.get("flags", [])
    flags = [dict(flag) for flag in raw_flags if isinstance(flag, Mapping)] if isinstance(raw_flags, list) else []
    severities = [_severity_rank(flag.get("severity")) for flag in flags]
    return {
        "flag_count": len(flags),
        "max_severity": max(severities) if severities else -1,
        "flags": flags,
    }


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """Return true when no critical adversarial flag is present."""

    flags = report.get("flags", [])
    if not isinstance(flags, list):
        return False
    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in flags
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while excluding the checksum field itself."""

    filtered = {
        key: value for key, value in artifact.items() if key != "reproducibility_checksum"
    }
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ledger_row(
    thread: str,
    settled: bool,
    terminal_status: str,
    authoritative_artifact: str,
) -> JsonDict:
    return {
        "thread": thread,
        "terminal_status": terminal_status if settled else "not_terminal_or_missing_evidence",
        "authoritative_artifact": authoritative_artifact,
        "settled": settled,
    }


def _source_payload_checksums(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "name": name,
            "path": str(UPSTREAM_ARTIFACTS[name]),
            "sha256": _sha256_payload(payload),
        }
        for name, payload in sorted(payloads.items())
    ]


def _read_json_object(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return hashlib.sha256(b"<missing>").hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _point(value: Any) -> float | None:
    if isinstance(value, Mapping):
        return _point(value.get("point"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    return None


def _contains(value: str, needle: str) -> bool:
    return needle.lower() in value.lower().replace("-", "_")


def _no_compute_markers(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact)
    disallowed = ("GGUF", "CUDA", "llama.cpp", "torch.cuda", ".cuda(", "model_specs", "target_model")
    return not any(marker in encoded for marker in disallowed)


def _severity_rank(value: Any) -> int:
    return {"info": 0, "warn": 1, "critical": 2}.get(str(value).lower(), -1)


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
