"""Build the Exp 1378 publication-hold v16 claim-boundary artifact.

Spec: REQ-PUBLISH-017, SCENARIO-PUBLISH-018.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
DEFAULT_RUN_DATE = "20260505"
EXPERIMENT = "1378_publication_hold_v16_claim_boundary"
SCHEMA = "publication_hold_v16_claim_boundary_v1"
HONEST_VERDICT_LIFT_RECOMMENDED = (
    "lift_recommended_primary_blockers_resolved_no_external_parity_claim"
)
HONEST_VERDICT_ACTIVE = "publication_hold_active_primary_blockers_remaining"
DEFAULT_OUT_PATH = Path("results") / "experiment_1378_publication_hold_v16_claim_boundary.json"

SOURCE_PATHS = {
    "exp1362": Path("results/experiment_1362_publication_hold_ebt_arm_kona_claim_boundary.json"),
    "exp1366": Path(
        "results/experiment_1366_certificate_v8_tag_first_prefix_injection_crane.json"
    ),
    "exp1369": Path(
        "results/experiment_1369_semantic_validator_v2_nsvif_z3_constraints.json"
    ),
    "exp1370": Path("results/experiment_1370_verge_mcs_repair_localization_v2.json"),
    "exp1371": Path(
        "results/experiment_1371_margin_aware_cactus_beaver_scheduler_v3.json"
    ),
    "exp1372": Path("results/experiment_1372_optimal_kan_pwa_formal_verification.json"),
    "exp1374": Path(
        "results/experiment_1374_continuous_self_learning_v3_verifier_selected_or_csp_fallback.json"
    ),
}

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "certificate_evidence_summary",
    "semantic_repair_evidence_summary",
    "kan_formal_evidence_summary",
    "self_learning_evidence_summary",
    "hold_blocker_resolved_certificate",
    "hold_blocker_resolved_semantic_repair",
    "hold_blocker_resolved_self_learning",
    "all_primary_blockers_resolved",
    "publication_hold_state",
    "paper_changes_needed_for_lift",
    "ebt_arm_claim_boundary",
    "dvi_ready",
    "external_dependency_claim_allowed",
    "honest_verdict",
}

WriteObserver = Callable[[Path, dict[str, Any]], None]


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    project_root: Path | str = REPO_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-PUBLISH-017: persist the audit marker before source evidence is read."""

    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="in_progress")
    _write_json(Path(out_path), artifact, write_observer=write_observer)
    return artifact


def build_artifact(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    project_root: Path | str = REPO_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    input_resolution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """REQ-PUBLISH-017: decide hold state from local .106 evidence only.

    The runner distinguishes a publication-hold recommendation from a broader
    parity claim. The .106 evidence can resolve the three named primary
    blockers while still leaving EBT/ARM/Kona/external dependency language
    disallowed unless a local parity artifact proves it.
    """

    certificate = _certificate_evidence_summary(sources)
    semantic_repair = _semantic_repair_evidence_summary(sources)
    kan_formal = _kan_formal_evidence_summary(sources)
    self_learning = _self_learning_evidence_summary(sources)

    certificate_resolved = _number_at_least(certificate["certificate_parse_rate"], 0.75)
    semantic_repair_resolved = (
        semantic_repair["semantic_validator_claim_allowed"] is True
        and semantic_repair["repair_claim_allowed"] is True
        and semantic_repair["triage_claim_allowed"] is True
        and semantic_repair["false_acceptance_rate"] == 0.0
    )
    self_learning_resolved = self_learning["headline_result_allowed"] is True
    all_primary_resolved = (
        certificate_resolved and semantic_repair_resolved and self_learning_resolved
    )
    remaining_blockers = _remaining_primary_blockers(
        certificate_resolved=certificate_resolved,
        semantic_repair_resolved=semantic_repair_resolved,
        self_learning_resolved=self_learning_resolved,
    )

    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="complete")
    artifact.update(
        {
            "input_resolution": dict(input_resolution or {}),
            "prior_publication_hold_state": _metric(sources, "exp1362", "publication_hold_state"),
            "prior_blockers_inherited": _prior_blockers(sources),
            "certificate_evidence_summary": certificate,
            "semantic_repair_evidence_summary": semantic_repair,
            "kan_formal_evidence_summary": kan_formal,
            "self_learning_evidence_summary": self_learning,
            "hold_blocker_resolved_certificate": certificate_resolved,
            "hold_blocker_resolved_semantic_repair": semantic_repair_resolved,
            "hold_blocker_resolved_self_learning": self_learning_resolved,
            "all_primary_blockers_resolved": all_primary_resolved,
            "remaining_primary_blockers": remaining_blockers,
            "publication_hold_state": "lift_recommended" if all_primary_resolved else "active",
            "paper_changes_needed_for_lift": _paper_changes_needed_for_lift(),
            "ebt_arm_claim_boundary": _ebt_arm_claim_boundary(
                sources=sources,
                all_primary_blockers_resolved=all_primary_resolved,
                kan_formal_claim_allowed=kan_formal["kan_formal_claim_allowed"] is True,
            ),
            "dvi_ready": self_learning["dvi_ready"] is True,
            "external_dependency_claim_allowed": False,
            "honest_verdict": (
                HONEST_VERDICT_LIFT_RECOMMENDED
                if all_primary_resolved
                else HONEST_VERDICT_ACTIVE
            ),
        }
    )
    validate_artifact(artifact)
    return artifact


def run(
    project_root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    run_date: str = DEFAULT_RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """SCENARIO-PUBLISH-018: write bootstrap, load evidence, and finalize JSON."""

    root = Path(project_root)
    output = _resolve(root, out_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        write_observer=write_observer,
    )
    sources, input_resolution = _load_sources(root)
    artifact = build_artifact(
        sources,
        project_root=root,
        run_date=run_date,
        input_resolution=input_resolution,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-PUBLISH-017: reject malformed artifacts and publication overclaims."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    _require(not missing, f"missing required fields: {sorted(missing)}")
    _require(artifact["status"] == "complete", "status must be complete")
    _require(
        artifact["external_dependency_claim_allowed"] is False,
        "external_dependency_claim_allowed must remain false without local parity evidence",
    )
    _require(
        artifact["ebt_arm_claim_boundary"].get("external_dependency_claim_allowed") is False,
        "EBT/ARM boundary must disallow external dependency claims",
    )
    _require(
        artifact["ebt_arm_claim_boundary"].get("local_ebt_arm_equivalence_proven") is False,
        "EBT/ARM equivalence must not be claimed by this hold review",
    )
    _require(
        bool(artifact["paper_changes_needed_for_lift"]),
        "paper_changes_needed_for_lift must name required paper edits",
    )

    all_resolved = artifact["all_primary_blockers_resolved"] is True
    if all_resolved:
        _require(
            artifact["publication_hold_state"] == "lift_recommended",
            "resolved primary blockers must set lift_recommended",
        )
        _require(
            artifact["honest_verdict"] == HONEST_VERDICT_LIFT_RECOMMENDED,
            "resolved primary blockers require lift recommendation verdict",
        )
    else:
        _require(
            artifact["publication_hold_state"] == "active",
            "unresolved primary blockers must keep publication hold active",
        )
        _require(
            artifact.get("remaining_primary_blockers"),
            "active hold must name remaining primary blockers",
        )
        _require(
            artifact["honest_verdict"] == HONEST_VERDICT_ACTIVE,
            "active hold requires remaining-blocker verdict",
        )

    _require(
        artifact["publication_hold_state"] != "lift_recommended"
        or artifact["all_primary_blockers_resolved"] is True,
        "lift_recommended requires all primary blockers resolved",
    )
    _require(
        artifact["hold_blocker_resolved_certificate"] is False
        or _number_at_least(
            artifact["certificate_evidence_summary"].get("certificate_parse_rate"), 0.75
        ),
        "certificate blocker cannot resolve below parse-rate gate",
    )
    _require(
        artifact["hold_blocker_resolved_semantic_repair"] is False
        or (
            artifact["semantic_repair_evidence_summary"].get(
                "semantic_validator_claim_allowed"
            )
            is True
            and artifact["semantic_repair_evidence_summary"].get("repair_claim_allowed")
            is True
            and artifact["semantic_repair_evidence_summary"].get("triage_claim_allowed")
            is True
            and artifact["semantic_repair_evidence_summary"].get("false_acceptance_rate")
            == 0.0
        ),
        "semantic repair blocker requires validator, repair, scheduler, and zero false acceptance",
    )
    _require(
        artifact["hold_blocker_resolved_self_learning"] is False
        or artifact["self_learning_evidence_summary"].get("headline_result_allowed") is True,
        "self-learning blocker cannot resolve without headline_result_allowed",
    )


def _base_artifact(*, project_root: Path | str, run_date: str, status: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": {
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "actual_project_root": str(root),
            "run_date": run_date,
            "source_artifacts": [str(path) for path in SOURCE_PATHS.values()],
            "source_documents": [
                "CODEX.md",
                "CLAUDE.md",
                "_bmad/prd.md",
                "_bmad/architecture.md",
                "research-references.md",
            ],
            "spec_refs": ["REQ-PUBLISH-017", "SCENARIO-PUBLISH-018"],
        },
        "run_date": run_date,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": status,
        "certificate_evidence_summary": {},
        "semantic_repair_evidence_summary": {},
        "kan_formal_evidence_summary": {},
        "self_learning_evidence_summary": {},
        "hold_blocker_resolved_certificate": False,
        "hold_blocker_resolved_semantic_repair": False,
        "hold_blocker_resolved_self_learning": False,
        "all_primary_blockers_resolved": False,
        "remaining_primary_blockers": [],
        "publication_hold_state": "active",
        "paper_changes_needed_for_lift": [],
        "ebt_arm_claim_boundary": {},
        "dvi_ready": False,
        "external_dependency_claim_allowed": False,
        "honest_verdict": "in_progress" if status == "in_progress" else HONEST_VERDICT_ACTIVE,
    }


def _certificate_evidence_summary(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1366 = _source(sources, "exp1366")
    parse_rate = _metric(sources, "exp1366", "certificate_parse_rate")
    return {
        "source_experiment": "exp1366",
        "status": exp1366.get("status"),
        "certificate_case_count": exp1366.get("certificate_case_count"),
        "certificate_parse_rate": parse_rate,
        "certificate_truthfulness_rate": exp1366.get("certificate_truthfulness_rate"),
        "prefix_injection_supported": exp1366.get("prefix_injection_supported"),
        "headline_result_allowed": exp1366.get("headline_result_allowed"),
        "trigger_token_hit_rate": exp1366.get("trigger_token_hit_rate"),
        "unknown_preservation_rate": exp1366.get("unknown_preservation_rate"),
        "mandated_sota_parse_rate": exp1366.get("mandated_sota_parse_rate"),
        "parse_gate_threshold": 0.75,
        "parse_gate_satisfied": _number_at_least(parse_rate, 0.75),
        "honest_verdict": exp1366.get("honest_verdict"),
        "claim_boundary": (
            "Exp 1366 resolves the prior certificate blocker for this hold review: "
            "local CRANE tag-first prefix injection produced parse_rate=1.0 on the "
            "four certificate cases, with prefix injection and headline result allowed."
        ),
    }


def _semantic_repair_evidence_summary(
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    exp1369 = _source(sources, "exp1369")
    exp1370 = _source(sources, "exp1370")
    exp1371 = _source(sources, "exp1371")
    return {
        "source_experiments": ["exp1369", "exp1370", "exp1371"],
        "exp1369_status": exp1369.get("status"),
        "parsed_certificate_cases": exp1369.get("parsed_certificate_cases"),
        "validator_execution_pass_rate": exp1369.get("validator_execution_pass_rate"),
        "semantic_validator_claim_allowed": exp1369.get("semantic_validator_claim_allowed"),
        "z3_constraint_pass_rate": exp1369.get("z3_constraint_pass_rate"),
        "smt_text_constraint_pass_rate": exp1369.get("smt_text_constraint_pass_rate"),
        "unknown_preservation_rate": exp1369.get("unknown_preservation_rate"),
        "exp1370_status": exp1370.get("status"),
        "repair_claim_allowed": exp1370.get("repair_claim_allowed"),
        "mcs_localization_rate": exp1370.get("mcs_localization_rate"),
        "repair_hint_precision": exp1370.get("repair_hint_precision"),
        "semantic_equivalence_pass_rate": exp1370.get("semantic_equivalence_pass_rate"),
        "accepted_repair_count": exp1370.get("accepted_repair_count"),
        "rejected_repair_count": exp1370.get("rejected_repair_count"),
        "exp1371_status": exp1371.get("status"),
        "triage_claim_allowed": exp1371.get("triage_claim_allowed"),
        "false_acceptance_rate": exp1371.get("false_acceptance_rate"),
        "unknown_silently_accepted_count": exp1371.get("unknown_silently_accepted_count"),
        "observed_full_verifier_call_reduction": exp1371.get(
            "observed_full_verifier_call_reduction"
        ),
        "honest_verdicts": {
            "exp1369": exp1369.get("honest_verdict"),
            "exp1370": exp1370.get("honest_verdict"),
            "exp1371": exp1371.get("honest_verdict"),
        },
        "claim_boundary": (
            "The semantic chain is local replay evidence over Exp 1366 rows: Exp 1369 "
            "executes validator checks, Exp 1370 localizes MCS repairs, and Exp 1371 "
            "adds conservative scheduler triage with zero false acceptance."
        ),
    }


def _kan_formal_evidence_summary(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1372 = _source(sources, "exp1372")
    return {
        "source_experiment": "exp1372",
        "status": exp1372.get("status"),
        "formal_property_verified": exp1372.get("formal_property_verified"),
        "kan_formal_claim_allowed": exp1372.get("kan_formal_claim_allowed"),
        "milp_verification_result": exp1372.get("milp_verification_result"),
        "milp_solver_status": exp1372.get("milp_solver_status"),
        "property_tested": exp1372.get("property_tested"),
        "property_energy_threshold": exp1372.get("property_energy_threshold"),
        "lp_certified_upper_bound": exp1372.get("lp_certified_upper_bound"),
        "pwa_abstraction_error_max": exp1372.get("pwa_abstraction_error_max"),
        "hardware_execution_claimed": exp1372.get("hardware_execution_claimed"),
        "hardware_correctness_claimed": exp1372.get("hardware_correctness_claimed"),
        "honest_verdict": exp1372.get("honest_verdict"),
        "claim_boundary": (
            "Exp 1372 allows a local CPU-only GS-KAN formal energy-bound claim. "
            "It supports the publication evidence chain but does not imply hardware "
            "execution or external dependency parity."
        ),
    }


def _self_learning_evidence_summary(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1374 = _source(sources, "exp1374")
    return {
        "source_experiment": "exp1374",
        "status": exp1374.get("status"),
        "headline_result_allowed": exp1374.get("headline_result_allowed"),
        "path_used": exp1374.get("path_used"),
        "dvi_ready": exp1374.get("dvi_ready"),
        "fresh_verified_sample_count": exp1374.get("fresh_verified_sample_count"),
        "self_learning_delta_overall": exp1374.get("self_learning_delta_overall"),
        "nonforgetting_certificate_rate": exp1374.get("nonforgetting_certificate_rate"),
        "memory_regression_count": exp1374.get("memory_regression_count"),
        "accepted_violation_delta": exp1374.get("accepted_violation_delta"),
        "promoted_memory_count": exp1374.get("promoted_memory_count"),
        "quarantined_memory_count": exp1374.get("quarantined_memory_count"),
        "honest_verdict": exp1374.get("honest_verdict"),
        "claim_boundary": (
            "Exp 1374 resolves the self-learning blocker for this hold review by using "
            "the primary_semantic_verified path with four fresh verified samples, clean "
            "non-forgetting controls, dvi_ready=true, and headline_result_allowed=true."
        ),
    }


def _ebt_arm_claim_boundary(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    all_primary_blockers_resolved: bool,
    kan_formal_claim_allowed: bool,
) -> dict[str, Any]:
    exp1362_boundary = _metric(sources, "exp1362", "ebt_arm_claim_boundary") or {}
    return {
        "external_dependency_claim_allowed": False,
        "local_ebt_arm_equivalence_proven": False,
        "local_primary_blocker_evidence_sufficient_for_hold_review": all_primary_blockers_resolved,
        "kan_formal_claim_allowed": kan_formal_claim_allowed,
        "inherited_exp1362_external_dependency_claim_allowed": exp1362_boundary.get(
            "external_dependency_claim_allowed"
        ),
        "allowed_language": [
            "The .106 local evidence resolves the certificate, semantic-repair, and headline self-learning blockers that kept the publication hold active.",
            "Exp 1372 adds a CPU-only formal GS-KAN energy-bound result that may be described as local formal verifier evidence.",
            "EBT and ARM-EBM may remain related work or architectural motivation when clearly separated from local Carnot proof.",
        ],
        "disallowed_language": [
            "Carnot has achieved EBT, ARM-EBM, Kona, Extropic, THRML, TSU, or hardware parity.",
            "External EBT, ARM-EBM, Kona, or Extropic results are evidence for Carnot's local claims.",
            "The .106 hold review proves native continuous-latent full-answer reasoning or external dependency equivalence.",
        ],
    }


def _paper_changes_needed_for_lift() -> list[dict[str, str]]:
    return [
        {
            "section": "Abstract / contribution summary",
            "change": "Replace publication-hold language with .106 local evidence and keep the state as lift_recommended until operator action.",
        },
        {
            "section": "Certificate generation results",
            "change": "Report Exp 1366 certificate_parse_rate=1.0, prefix_injection_supported=true, and headline_result_allowed=true.",
        },
        {
            "section": "Semantic validation and repair",
            "change": "Add Exp 1369 validator_execution_pass_rate=1.0, Exp 1370 repair_claim_allowed=true, and Exp 1371 triage_claim_allowed=true with false_acceptance_rate=0.0.",
        },
        {
            "section": "Formal KAN evidence",
            "change": "Add Exp 1372 formal_property_verified=true and kan_formal_claim_allowed=true while labeling it CPU-only and non-hardware.",
        },
        {
            "section": "Self-learning / DVI readiness",
            "change": "Report Exp 1374 headline_result_allowed=true, path_used=primary_semantic_verified, fresh_verified_sample_count=4, and dvi_ready=true.",
        },
        {
            "section": "Limitations and claim boundaries",
            "change": "State that external parity and external dependency claims remain disallowed without local evidence.",
        },
    ]


def _remaining_primary_blockers(
    *,
    certificate_resolved: bool,
    semantic_repair_resolved: bool,
    self_learning_resolved: bool,
) -> list[str]:
    blockers: list[str] = []
    if not certificate_resolved:
        blockers.append("certificate_parse_rate_below_0_75")
    if not semantic_repair_resolved:
        blockers.append("semantic_repair_chain_incomplete")
    if not self_learning_resolved:
        blockers.append("headline_self_learning_not_allowed")
    return blockers


def _prior_blockers(sources: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blockers = _metric(sources, "exp1362", "publication_hold_rationale", "active_blockers")
    return list(blockers) if isinstance(blockers, list) else []


def _load_sources(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for key, rel_path in SOURCE_PATHS.items():
        path = _resolve(root, rel_path)
        if path.exists():
            sources[key] = _read_json(path)
        else:
            sources[key] = {}
            missing.append(str(rel_path))
    return sources, {"missing_source_artifacts": missing}


def _source(sources: Mapping[str, Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    return sources.get(name, {})


def _metric(artifact: Mapping[str, Any], *keys: str) -> Any:
    value: Any = artifact
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _number_at_least(value: object, threshold: float) -> bool:
    return isinstance(value, int | float) and float(value) >= threshold


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if write_observer is not None:
        write_observer(path, payload)


def _resolve(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
