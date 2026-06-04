"""Build the Exp 3796 v347 post-convergence capstone artifact.

Spec refs: REQ-REPORT-3796, SCENARIO-REPORT-3796,
SCENARIO-REPORT-3796-P1-GUARD, SCENARIO-REPORT-3796-FUNDAMENTAL-GUARD,
SCENARIO-REPORT-3796-MISSING-BLOCKED, SCENARIO-REPORT-3796-FLAGGED.

This module is an aggregation capstone. It reads already-written experiment
artifacts, records what landed, and keeps missing or blocked upstream tasks out
of the research-negative column. It does not rerun live models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct import guard
    sys.path.insert(0, str(REPO_ROOT))

from scripts import adversarial_verify  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_3796_capstone_v347.json")
RANDOM_SEED = 3796
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a capstone reads upstream JSON, "
    "runs no live model)."
)
UPSTREAM_IDS = tuple(range(3786, 3796))

DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    3786: Path("results/experiment_3786_archive_v346_activate_v347.json"),
    3787: Path("results/experiment_3787_p1_discrete_search_adjudication_v3_retry.json"),
    3788: Path("results/experiment_3788_fr11_self_learning_v19_tier3_predictive.json"),
    3789: Path("results/experiment_3789_abstention_cli_batch_surface.json"),
    3790: Path("results/experiment_3790_verifier_gaming_resistance_characterization.json"),
    3791: Path("results/experiment_3791_anomaly_escalation_classifier_validation.json"),
    3792: Path("results/experiment_3792_product_headline_provenance_confirmation_g4.json"),
    3793: Path("results/experiment_3793_edlm_no_train_preflight_readiness.json"),
    3794: Path("results/experiment_3794_external_research_refresh.json"),
    3795: Path("results/experiment_3795_kv260_opportunistic_continuity_audit.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "p1_adjudication",
    "p1_positive_control_passed",
    "energy_as_generator_still_bounded",
    "verifier_product_hardened",
    "product_headline_restorable",
    "fr11_v19_tier3_self_learning",
    "anomaly_escalation_validated",
    "edlm_seed_preflighted",
    "paper_ready_preserved",
    "frozen_headline_unchanged",
    "next_thesis_remains_operator_surface",
    "flagged_artifacts_excluded",
    "not_landed_or_blocked_recorded_honestly",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the milestone's one-line outcome.",
    "inference_substrate": (
        "aggregation_from_upstream_artifacts (principle: a capstone reads upstream "
        "JSON, runs no live model)."
    ),
    "p1_adjudication": (
        "The exp3787 verdict {decode_artifact_bounded | fundamental | inconclusive | "
        "blocked_no_free_gpu} -- the milestone-defining mechanistic result (or honest block)."
    ),
    "p1_positive_control_passed": (
        "BARE bool -- whether the AR positive control reached >=0.3; a null "
        "'FUNDAMENTAL' is invalid without it (the v1/v2 lesson)."
    ),
    "energy_as_generator_still_bounded": (
        "BARE bool, true -- the part-b strategic bound stands regardless of the P1 "
        "mechanism verdict."
    ),
    "verifier_product_hardened": (
        "Records the abstention CLI+batch surface (exp3789) + gaming-resistance "
        "curve (exp3790) + product-headline provenance (exp3792) -- the "
        "deployable-product advances."
    ),
    "product_headline_restorable": (
        "Records exp3792's recommendation {restorable | restorable_with_caveat | "
        "not_yet_eligible} for the operator."
    ),
    "fr11_v19_tier3_self_learning": (
        "Records the Tier-3 predictive self-learning (exp3788, headline ensemble "
        "unchanged, memory contribution preserved) -- the self-learning mandate."
    ),
    "anomaly_escalation_validated": (
        "BARE bool -- the .346 classifier was validated against the historical corpus "
        "(exp3791, recommend-only, conductor unmodified)."
    ),
    "edlm_seed_preflighted": (
        "Records the EDLM go/no-go readiness + one-command seed (exp3793) -- the loop "
        "does NOT commit."
    ),
    "paper_ready_preserved": (
        "BARE bool -- G1-G4 stay met; the milestone must not regress the banked "
        "verifier product."
    ),
    "frozen_headline_unchanged": (
        "BARE bool -- frozen FoVer 0.9131 stays frozen; .347 uses but never moves it."
    ),
    "next_thesis_remains_operator_surface": (
        "BARE bool -- the next-Phase-3 decision remains an operator-seeding surface; "
        "the loop does not self-commit."
    ),
    "flagged_artifacts_excluded": (
        "Lists any flagged_adversarial artifact excluded from aggregation "
        "(fabrication gate)."
    ),
    "not_landed_or_blocked_recorded_honestly": (
        "Lists any upstream task that did not land or blocked (e.g. exp3787 if no "
        "free GPU) -- recorded as not-run/blocked, NOT as a research negative "
        "(the .344 capstone-confusion guard)."
    ),
    "cited_upstream_artifacts": "Provenance trail from the capstone summary to the real artifacts.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

P1_ALLOWED = {
    "decode_artifact_bounded",
    "fundamental_causal_inductive_bias_gap",
    "inconclusive_positive_control_failed",
    "blocked_no_free_gpu",
}

PRODUCT_HEADLINE_ALLOWED = {"restorable", "restorable_with_caveat", "not_yet_eligible"}

SUMMARY_FIELDS: Mapping[int, tuple[str, ...]] = {
    3786: (
        "honest_verdict",
        "both_energy_routes_still_bounded",
        "paper_ready_preserved",
        "paper_ready_evidence.paper_ready",
        "paper_ready_evidence.g1",
        "paper_ready_evidence.g2",
        "paper_ready_evidence.g3",
        "paper_ready_evidence.g4",
        "paper_ready_evidence.frozen_headline_unchanged",
        "paper_ready_evidence.frozen_headline_auroc",
        "v347_focus_recorded",
    ),
    3787: (
        "honest_verdict",
        "adjudication",
        "p1_adjudication",
        "positive_control_passed",
        "p1_positive_control_passed",
        "energy_as_generator_still_bounded",
        "handoff_to_operator",
        "ar_best",
        "ebt_best",
    ),
    3788: (
        "honest_verdict",
        "continuous_self_learning_task",
        "is_tier3_not_tier1_or_tier2",
        "predictive_auroc",
        "headline_ensemble_unchanged",
        "frozen_headline_ensemble_auroc",
        "memory_contribution_preserved",
        "tracker_state_persisted",
        "acceptance_gate.passed",
    ),
    3789: (
        "honest_verdict",
        "cli_abstention_surface_added",
        "batch_path_works",
        "default_off_preserves_prior_behavior",
        "e2e_cli_abstention_passed",
        "tests_assert_real_behavior",
        "scripts_research_conductor_modified",
    ),
    3790: (
        "honest_verdict",
        "gaming_degradation_curve",
        "headline_unchanged",
        "n_samples",
        "not_a_moat_reopen",
        "perturbations_tested",
        "verifier_degrades_where",
    ),
    3791: (
        "honest_verdict",
        "false_escalation_rate",
        "frame_violating_recall",
        "never_relaxes_verification",
        "conductor_unmodified",
        "tests_assert_real_behavior",
        "supports_wiring_in",
    ),
    3792: (
        "honest_verdict",
        "product_headline_restorable",
        "exp1999_g4_pass",
        "exp2090_g4_pass",
        "operator_curated_doc_unedited",
        "provenance_table",
    ),
    3793: (
        "honest_verdict",
        "readiness_verdict",
        "reference_impl_fetchable",
        "minimal_kill_gate_sound",
        "operator_seed_command",
        "loop_does_not_commit",
    ),
    3794: (
        "honest_verdict",
        "references_added",
        "n_references_added",
        "section_appended_not_replaced",
        "numbers_are_as_reported",
    ),
    3795: (
        "honest_verdict",
        "terminal_state_holds",
        "kv260_ssh_reachable",
        "kv260_overlay_loadable",
        "speedup_claim_made",
    ),
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    adversarial_reports: Mapping[int, Mapping[str, Any]] | None = None,
    publication_gate_data: Mapping[str, Any] | None = None,
    capstone_adversarial_verify_clean: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the v347 capstone from upstream JSON artifacts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = {experiment_id: resolve_upstream_path(root_path, experiment_id) for experiment_id in UPSTREAM_IDS}
    upstreams = {
        experiment_id: read_json_object(path) if path.exists() else None
        for experiment_id, path in paths.items()
    }
    reports = (
        {experiment_id: dict(report) for experiment_id, report in adversarial_reports.items()}
        if adversarial_reports is not None
        else verify_upstreams(paths)
    )
    flagged_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True
    }
    clean_upstreams = {
        experiment_id: payload
        for experiment_id, payload in upstreams.items()
        if payload is not None and experiment_id not in flagged_ids
    }
    headline_upstreams = {
        experiment_id: payload
        for experiment_id, payload in clean_upstreams.items()
        if not is_blocked_payload(payload)
    }
    summaries = compact_summary_records(
        summary_records if summary_records is not None else run_summarize_artifacts(root_path, paths)
    )
    gate_data = dict(publication_gate_data) if publication_gate_data is not None else load_publication_gate(root_path)

    exp3786 = headline_upstreams.get(3786, {})
    exp3787_clean = clean_upstreams.get(3787)
    exp3788 = headline_upstreams.get(3788, {})
    exp3789 = headline_upstreams.get(3789, {})
    exp3790 = headline_upstreams.get(3790, {})
    exp3791 = headline_upstreams.get(3791, {})
    exp3792 = headline_upstreams.get(3792, {})
    exp3793 = headline_upstreams.get(3793, {})
    exp3794 = headline_upstreams.get(3794, {})
    exp3795 = headline_upstreams.get(3795, {})

    p1 = p1_state(exp3787_clean, missing=upstreams.get(3787) is None, flagged=3787 in flagged_ids)
    energy_bounded = True
    verifier = verifier_product_hardening(exp3789, exp3790, exp3792)
    product_headline = normalize_product_headline_status(exp3792.get("product_headline_restorable"))
    fr11 = fr11_v19_tier3_self_learning(exp3788)
    anomaly_validation = anomaly_escalation_validation(exp3791)
    anomaly = bool(anomaly_validation["validated"])
    edlm = edlm_seed_preflight(exp3793)
    paper_ready = paper_ready_from_gate(gate_data)
    frozen = frozen_headline_unchanged(exp3786)
    next_operator = bool(edlm["preflighted"] and edlm["loop_does_not_commit"])
    references = references_refreshed(exp3794)
    kv260 = kv260_terminal_confirmed(exp3795)
    duration_s = round(max(0.0001, (time.perf_counter() if now_s is None else float(now_s)) - start), 6)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v347_post_convergence_3796.v1",
        "experiment_id": "exp3796",
        "honest_verdict": terminal_verdict(
            p1_adjudication=p1["adjudication"],
            energy_bounded=energy_bounded,
            verifier_hardened=bool(verifier["hardened"]),
            fr11=bool(fr11["validated"]),
            anomaly=anomaly,
            edlm=bool(edlm["preflighted"]),
            paper_ready=paper_ready,
            frozen=frozen,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "p1_adjudication": p1["adjudication"],
        "p1_positive_control_passed": p1["positive_control_passed"],
        "p1_mechanism_status": p1["mechanism_status"],
        "p1_handoff_to_operator": p1["handoff_to_operator"],
        "energy_as_generator_still_bounded": energy_bounded,
        "energy_as_selector_status": "honest-negative-bounded",
        "energy_as_generator_status": "honest-negative-bounded",
        "verifier_product_hardened": verifier,
        "product_headline_restorable": product_headline,
        "fr11_v19_tier3_self_learning": fr11,
        "anomaly_escalation_validated": anomaly,
        "anomaly_escalation_validation": anomaly_validation,
        "edlm_seed_preflighted": edlm,
        "paper_ready_preserved": paper_ready,
        "publication_gate_state": publication_gate_state(gate_data),
        "frozen_headline_unchanged": frozen,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "next_thesis_remains_operator_surface": next_operator,
        "references_refreshed": references,
        "kv260_terminal_confirmed": kv260,
        "milestone_outcome_plain": milestone_outcome(p1["adjudication"], p1["mechanism_status"]),
        "no_new_existential_claim": True,
        "regrinds_nothing_already_bounded": True,
        "headline_aggregation_experiment_ids": sorted(headline_upstreams),
        "flagged_artifacts_excluded": flagged_artifacts(paths, flagged_ids),
        "not_landed_or_blocked_recorded_honestly": not_landed_or_blocked(paths, upstreams, flagged_ids),
        "cited_upstream_artifacts": cited_upstream_artifacts(root_path, paths, clean_upstreams),
        "summarized_upstream_artifacts": summaries,
        "upstream_adversarial_critical_flags": critical_adversarial_flags(reports),
        "adversarial_verify_clean": capstone_adversarial_verify_clean,
        "adversarial_verify_report": {"flags": []},
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    *,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    adversarial_reports: Mapping[int, Mapping[str, Any]] | None = None,
    publication_gate_data: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write, adversarial-verify, and rewrite the Exp 3796 artifact."""

    root_path = Path(root)
    out_path = root_path / OUTPUT_REL_PATH
    artifact = build_artifact(
        root_path,
        summary_records=summary_records,
        adversarial_reports=adversarial_reports,
        publication_gate_data=publication_gate_data,
        capstone_adversarial_verify_clean=True,
        started_s=started_s,
        now_s=now_s,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = adversarial_verify.verify_artifact(out_path)
    artifact["adversarial_verify_report"] = report
    artifact["adversarial_verify_clean"] = report_is_clean(report)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and honesty errors for the Exp 3796 capstone."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required artifact fields: {', '.join(missing)}")
    if not str(artifact.get("honest_verdict") or "").startswith("complete: capstone_v347_p1_"):
        errors.append("honest_verdict must be a terminal Exp 3796 verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare the v347 aggregation-only substrate")
    p1_adjudication = artifact.get("p1_adjudication")
    if not valid_p1_adjudication(p1_adjudication):
        errors.append("p1_adjudication must be a valid adjudication or blocked_* value")
    if not isinstance(artifact.get("p1_positive_control_passed"), bool):
        errors.append("p1_positive_control_passed must be a bare bool")
    if (
        p1_adjudication == "fundamental_causal_inductive_bias_gap"
        and artifact.get("p1_positive_control_passed") is not True
    ):
        errors.append("fundamental P1 adjudication requires positive control")
    if artifact.get("energy_as_generator_still_bounded") is not True:
        errors.append("energy_as_generator_still_bounded must be true")
    validate_verifier_product(artifact.get("verifier_product_hardened"), errors)
    if artifact.get("product_headline_restorable") not in PRODUCT_HEADLINE_ALLOWED:
        errors.append("product_headline_restorable must be restorable, restorable_with_caveat, or not_yet_eligible")
    validate_tier3(artifact.get("fr11_v19_tier3_self_learning"), errors)
    if not isinstance(artifact.get("anomaly_escalation_validated"), bool):
        errors.append("anomaly_escalation_validated must be a bare bool")
    validate_edlm(artifact.get("edlm_seed_preflighted"), errors)
    if artifact.get("paper_ready_preserved") is not True:
        errors.append("paper_ready_preserved must be true")
    if artifact.get("frozen_headline_unchanged") is not True:
        errors.append("frozen_headline_unchanged must be true")
    if artifact.get("next_thesis_remains_operator_surface") is not True:
        errors.append("next_thesis_remains_operator_surface must be true")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        errors.append("flagged_artifacts_excluded must be a list")
    if not isinstance(artifact.get("not_landed_or_blocked_recorded_honestly"), list):
        errors.append("not_landed_or_blocked_recorded_honestly must be a list")
    validate_citations(artifact.get("cited_upstream_artifacts"), errors)
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed must equal 3796")
    duration_s = artifact.get("duration_s")
    if (
        not isinstance(duration_s, int | float)
        or isinstance(duration_s, bool)
        or float(duration_s) < 0.0001
    ):
        errors.append("duration_s must be numeric with the aggregation plausibility floor")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles must cover all required artifact fields")
    if has_live_model_markers(artifact):
        errors.append("artifact must not copy live-model substrate markers")
    if not report_is_clean(artifact.get("adversarial_verify_report", {"flags": []})):
        errors.append("adversarial verifier must report no critical flag")
    checksum = artifact.get("reproducibility_checksum")
    if not is_sha256(checksum):
        errors.append("reproducibility_checksum must be a sha256 hex string")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum does not match artifact content")
    return errors


def validate_verifier_product(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("hardened"), bool):
        errors.append("verifier_product_hardened must record hardening components")


def validate_tier3(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("validated"), bool):
        errors.append("fr11_v19_tier3_self_learning must record validation state")


def validate_edlm(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("preflighted"), bool):
        errors.append("edlm_seed_preflighted must record preflight state")


def terminal_verdict(
    *,
    p1_adjudication: str,
    energy_bounded: bool,
    verifier_hardened: bool,
    fr11: bool,
    anomaly: bool,
    edlm: bool,
    paper_ready: bool,
    frozen: bool,
) -> str:
    """Return the terminal verdict string using classified milestone states."""

    return (
        f"complete: capstone_v347_p1_{p1_adjudication}_"
        f"{'energy_as_generator_still_bounded' if energy_bounded else 'energy_as_generator_not_preserved'}_"
        f"{'verifier_product_hardened' if verifier_hardened else 'verifier_product_not_hardened'}_"
        f"{'fr11_v19_tier3' if fr11 else 'fr11_v19_tier3_not_validated'}_"
        f"{'anomaly_validated' if anomaly else 'anomaly_not_validated'}_"
        f"{'edlm_preflighted' if edlm else 'edlm_not_preflighted'}_"
        f"paper_ready_{str(paper_ready).lower()}_"
        f"{'frozen_headline_unchanged' if frozen else 'frozen_headline_changed'}"
    )


def resolve_upstream_path(root: Path, experiment_id: int) -> Path:
    """Return the default path or the first same-ID artifact in results."""

    default = root / DEFAULT_UPSTREAM_PATHS[experiment_id]
    if default.exists():
        return default
    matches = sorted((root / "results").glob(f"experiment_{experiment_id}_*.json"))
    return matches[0] if matches else default


def read_json_object(path: Path) -> JsonDict:
    """Read an upstream JSON object; array artifacts are invalid provenance."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def run_summarize_artifacts(
    root: Path,
    paths: Mapping[int, Path],
) -> list[JsonDict]:  # pragma: no cover - subprocess boundary
    records: list[JsonDict] = []
    for experiment_id in UPSTREAM_IDS:
        path = paths[experiment_id]
        arg = str(path) if path.exists() else str(experiment_id)
        completed = subprocess.run(
            [sys.executable, "scripts/summarize_artifact.py", arg],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        records.append(
            {
                "experiment_id": experiment_id,
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
            }
        )
    return records


def verify_upstreams(paths: Mapping[int, Path]) -> dict[int, JsonDict]:  # pragma: no cover
    return {
        experiment_id: adversarial_verify.verify_artifact(path) if path.exists() else {"flags": []}
        for experiment_id, path in paths.items()
    }


def load_publication_gate(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary
    completed = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("publication_gate.py --json must return a JSON object")
    return payload


def compact_summary_records(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep deterministic proof that the artifact summarizer was run."""

    return [
        {
            "experiment_id": record.get("experiment_id", record.get("exp")),
            "returncode": record.get("returncode"),
            "stdout_sha256": record.get("stdout_sha256"),
            "stderr_sha256": record.get("stderr_sha256"),
        }
        for record in records
    ]


def p1_state(
    payload: Mapping[str, Any] | None,
    *,
    missing: bool,
    flagged: bool,
) -> JsonDict:
    """Return the P1 adjudication without converting absence into a negative."""

    if flagged:
        adjudication = "blocked_flagged_adversarial"
        positive = False
    elif missing or not isinstance(payload, Mapping):
        adjudication = "blocked_missing_upstream_artifact"
        positive = False
    else:
        default = "blocked_upstream_artifact" if is_blocked_payload(payload) else "blocked_unknown_p1_adjudication"
        adjudication = normalize_p1_adjudication(payload, default=default)
        positive = bool(payload.get("positive_control_passed") or payload.get("p1_positive_control_passed"))
    mechanism_open = adjudication.startswith("blocked_") or adjudication == "inconclusive_positive_control_failed"
    return {
        "adjudication": adjudication,
        "positive_control_passed": positive,
        "mechanism_status": (
            f"open_for_operator: {adjudication}"
            if mechanism_open
            else f"settled_or_sharpened: {adjudication}"
        ),
        "handoff_to_operator": mechanism_open,
    }


def normalize_p1_adjudication(payload: Mapping[str, Any], *, default: str) -> str:
    verdict = str(payload.get("honest_verdict") or "").strip().lower()
    if verdict == "blocked_no_free_gpu" or "blocked_no_free_gpu" in verdict or "no_free_gpu" in verdict:
        return "blocked_no_free_gpu"
    if verdict.startswith("blocked:") and "no free gpu" in verdict:
        return "blocked_no_free_gpu"
    raw = str(payload.get("p1_adjudication") or payload.get("adjudication") or "").strip()
    lowered = raw.lower()
    if lowered in P1_ALLOWED or lowered.startswith("blocked_"):
        return lowered
    if lowered == "fundamental":
        return "fundamental_causal_inductive_bias_gap"
    if lowered == "inconclusive":
        return "inconclusive_positive_control_failed"
    if "decode_artifact_bounded" in verdict:
        return "decode_artifact_bounded"
    if "fundamental" in verdict:
        return "fundamental_causal_inductive_bias_gap"
    if "inconclusive" in verdict or "positive_control_failed" in verdict:
        return "inconclusive_positive_control_failed"
    if verdict.startswith("blocked:"):
        suffix = verdict.split(":", 1)[1].strip().replace("-", "_").replace(" ", "_")
        return f"blocked_{suffix}" if suffix else default
    if verdict.startswith("blocked_"):
        return verdict
    return default


def valid_p1_adjudication(value: Any) -> bool:
    return isinstance(value, str) and (value in P1_ALLOWED or value.startswith("blocked_"))


def is_blocked_payload(payload: Mapping[str, Any]) -> bool:
    verdict = str(payload.get("honest_verdict") or "").lower()
    return verdict.startswith("blocked:") or verdict.startswith("blocked_")


def verifier_product_hardening(
    exp3789: Mapping[str, Any],
    exp3790: Mapping[str, Any],
    exp3792: Mapping[str, Any],
) -> JsonDict:
    abstention = bool(
        exp3789
        and truthy(exp3789.get("cli_abstention_surface_added"))
        and truthy(exp3789.get("batch_path_works"))
        and truthy(exp3789.get("default_off_preserves_prior_behavior"))
        and truthy(exp3789.get("e2e_cli_abstention_passed"))
        and truthy(exp3789.get("tests_assert_real_behavior"))
        and exp3789.get("scripts_research_conductor_modified") is not True
    )
    gaming = bool(
        exp3790
        and isinstance(exp3790.get("gaming_degradation_curve"), Mapping)
        and numeric(exp3790.get("n_samples")) is not None
        and float(exp3790.get("n_samples")) >= 240
        and truthy(exp3790.get("not_a_moat_reopen"))
        and truthy(exp3790.get("headline_unchanged"))
    )
    provenance = bool(
        exp3792
        and normalize_product_headline_status(exp3792.get("product_headline_restorable")) in PRODUCT_HEADLINE_ALLOWED
        and truthy(exp3792.get("operator_curated_doc_unedited"))
    )
    return {
        "hardened": abstention and gaming and provenance,
        "abstention_cli_batch_surface": abstention,
        "default_off": bool(exp3789.get("default_off_preserves_prior_behavior") is True),
        "e2e_cli_abstention_passed": bool(exp3789.get("e2e_cli_abstention_passed") is True),
        "batch_path_works": bool(exp3789.get("batch_path_works") is True),
        "gaming_resistance_curve": gaming,
        "not_a_moat_reopen": bool(exp3790.get("not_a_moat_reopen") is True),
        "product_headline_provenance_confirmed": provenance,
    }


def normalize_product_headline_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "not_yet_headline_eligible":
        return "not_yet_eligible"
    if raw in PRODUCT_HEADLINE_ALLOWED:
        return raw
    return "not_yet_eligible"


def fr11_v19_tier3_self_learning(payload: Mapping[str, Any]) -> JsonDict:
    predictive_auroc = numeric(payload.get("predictive_auroc"))
    validated = bool(
        payload
        and truthy(payload.get("continuous_self_learning_task"))
        and truthy(payload.get("is_tier3_not_tier1_or_tier2"))
        and predictive_auroc is not None
        and truthy(payload.get("headline_ensemble_unchanged"))
        and truthy(payload.get("memory_contribution_preserved"))
        and truthy(payload.get("tracker_state_persisted"))
        and truthy(nested_get(payload, "acceptance_gate.passed"))
    )
    return {
        "validated": validated,
        "predictive_auroc": predictive_auroc,
        "headline_ensemble_unchanged": bool(payload.get("headline_ensemble_unchanged") is True),
        "frozen_headline_ensemble_auroc": numeric(payload.get("frozen_headline_ensemble_auroc")),
        "memory_contribution_preserved": bool(payload.get("memory_contribution_preserved") is True),
        "tracker_state_persisted": bool(payload.get("tracker_state_persisted") is True),
    }


def anomaly_escalation_validation(payload: Mapping[str, Any]) -> JsonDict:
    false_rate = numeric(payload.get("false_escalation_rate"))
    recall = numeric(payload.get("frame_violating_recall"))
    never_relaxes = bool(payload.get("never_relaxes_verification") is True)
    conductor_unmodified = bool(payload.get("conductor_unmodified") is True)
    validated = bool(
        payload
        and false_rate is not None
        and recall is not None
        and never_relaxes
        and conductor_unmodified
        and truthy(payload.get("tests_assert_real_behavior"))
    )
    return {
        "validated": validated,
        "false_escalation_rate": false_rate,
        "frame_violating_recall": recall,
        "never_relaxes_verification": never_relaxes,
        "conductor_unmodified": conductor_unmodified,
        "supports_wiring_in": bool(payload.get("supports_wiring_in") is True),
    }


def edlm_seed_preflight(payload: Mapping[str, Any]) -> JsonDict:
    command = payload.get("operator_seed_command")
    preflighted = bool(
        payload
        and payload.get("readiness_verdict") == "go"
        and truthy(payload.get("reference_impl_fetchable"))
        and truthy(payload.get("minimal_kill_gate_sound"))
        and isinstance(command, str)
        and bool(command.strip())
        and truthy(payload.get("loop_does_not_commit"))
    )
    return {
        "preflighted": preflighted,
        "readiness_verdict": payload.get("readiness_verdict"),
        "operator_seed_command": command,
        "minimal_kill_gate_sound": bool(payload.get("minimal_kill_gate_sound") is True),
        "reference_impl_fetchable": bool(payload.get("reference_impl_fetchable") is True),
        "loop_does_not_commit": bool(payload.get("loop_does_not_commit") is True),
    }


def references_refreshed(payload: Mapping[str, Any]) -> bool:
    n_references = payload.get("n_references_added")
    return bool(
        payload
        and isinstance(n_references, int)
        and 4 <= n_references <= 6
        and truthy(payload.get("section_appended_not_replaced"))
        and truthy(payload.get("numbers_are_as_reported"))
    )


def kv260_terminal_confirmed(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("terminal_state_holds"))
        and truthy(payload.get("kv260_ssh_reachable"))
        and truthy(payload.get("kv260_overlay_loadable"))
        and payload.get("speedup_claim_made") is not True
    )


def paper_ready_from_gate(gate_data: Mapping[str, Any]) -> bool:
    gates = gate_data.get("gates")
    return bool(
        gate_data.get("paper_ready") is True
        and isinstance(gates, Mapping)
        and all(nested_get(gates, f"{name}.pass") is True for name in ("G1", "G2", "G3", "G4"))
    )


def publication_gate_state(gate_data: Mapping[str, Any]) -> JsonDict:
    gates = gate_data.get("gates") if isinstance(gate_data.get("gates"), Mapping) else {}
    return {
        "paper_ready": gate_data.get("paper_ready") is True,
        "g1": nested_get(gates, "G1.pass") is True,
        "g2": nested_get(gates, "G2.pass") is True,
        "g3": nested_get(gates, "G3.pass") is True,
        "g4": nested_get(gates, "G4.pass") is True,
        "unmet_gates": gate_data.get("unmet_gates") if isinstance(gate_data.get("unmet_gates"), list) else [],
    }


def frozen_headline_unchanged(exp3786: Mapping[str, Any]) -> bool:
    evidence = exp3786.get("paper_ready_evidence")
    return bool(
        isinstance(evidence, Mapping)
        and evidence.get("frozen_headline_unchanged") is True
        and numeric(evidence.get("frozen_headline_auroc")) == FROZEN_FOVER_AUROC
    )


def milestone_outcome(p1_adjudication: str, mechanism_status: str) -> str:
    if p1_adjudication.startswith("blocked_") or p1_adjudication == "inconclusive_positive_control_failed":
        return (
            ".347 is a LEAN POST-CONVERGENCE milestone: it hardens the banked "
            "verifier product toward deployability, advances self-learning to "
            "Tier-3, validates the endorsed process upgrade, lowers the operator's "
            "EDLM seed cost, and leaves the P1 mechanism open for operator handoff. "
            "It makes NO new existential claim and re-grinds nothing bounded."
        )
    return (
        f".347 is a LEAN POST-CONVERGENCE milestone: it {mechanism_status}, "
        "hardens the banked verifier product toward deployability, advances "
        "self-learning to Tier-3, validates the endorsed process upgrade, and "
        "lowers the operator's EDLM seed cost. It makes NO new existential claim "
        "and re-grinds nothing bounded."
    )


def not_landed_or_blocked(
    paths: Mapping[int, Path],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    flagged_ids: set[int],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for experiment_id in UPSTREAM_IDS:
        if experiment_id in flagged_ids:
            continue
        payload = upstreams.get(experiment_id)
        if payload is None:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "path": str(paths[experiment_id]),
                    "status": "not-landed",
                    "reason": "artifact_missing",
                }
            )
        elif is_blocked_payload(payload):
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "path": str(paths[experiment_id]),
                    "status": "blocked",
                    "reason": str(payload.get("honest_verdict") or "blocked"),
                }
            )
    return rows


def flagged_artifacts(paths: Mapping[int, Path], flagged_ids: set[int]) -> list[JsonDict]:
    return [
        {"experiment_id": experiment_id, "path": str(paths[experiment_id]), "reason": "flagged_adversarial=true"}
        for experiment_id in sorted(flagged_ids)
    ]


def cited_upstream_artifacts(
    root: Path,
    paths: Mapping[int, Path],
    clean_upstreams: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    citations: list[JsonDict] = []
    for experiment_id in sorted(clean_upstreams):
        path = paths[experiment_id]
        payload = clean_upstreams[experiment_id]
        citations.append(
            {
                "experiment_id": experiment_id,
                "path": str(path),
                "fields_imported": [
                    field for field in SUMMARY_FIELDS[experiment_id] if nested_get(payload, field) is not None
                ],
                "honest_verdict": payload.get("honest_verdict"),
                "status": "blocked" if is_blocked_payload(payload) else "landed",
                "sha256": sha256_file(path if path.is_absolute() else root / path),
            }
        )
    return citations


def critical_adversarial_flags(reports: Mapping[int, Mapping[str, Any]]) -> list[JsonDict]:
    critical: list[JsonDict] = []
    for experiment_id, report in sorted(reports.items()):
        for flag in report.get("flags") or []:
            if isinstance(flag, Mapping) and str(flag.get("severity") or "").lower() == "critical":
                critical.append(
                    {
                        "experiment_id": experiment_id,
                        "kind": flag.get("kind"),
                        "detail": flag.get("detail"),
                    }
                )
    return critical


def validate_citations(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("cited_upstream_artifacts must be a list")
        return
    for item in value:
        if not isinstance(item, Mapping):
            errors.append("each citation must be an object")
            continue
        if not isinstance(item.get("fields_imported"), list):
            errors.append("each citation must include fields_imported")
        if not is_sha256(item.get("sha256")):
            errors.append("each citation must include a sha256 hex string")


def has_live_model_markers(payload: Mapping[str, Any]) -> bool:
    blob = json.dumps(payload, sort_keys=True).lower()
    return any(
        marker in blob
        for marker in ("live_llm_inference", "target_model", "model_specs", "live_model_invoked", "gguf", "cuda")
    )


def report_is_clean(report: Any) -> bool:
    if not isinstance(report, Mapping):
        return True
    for flag in report.get("flags") or []:
        if isinstance(flag, Mapping) and str(flag.get("severity") or "").lower() == "critical":
            return False
    return True


def nested_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def truthy(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() in {"true", "pass", "passed", "shipped"})


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return round(float(value), 6)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = dict(payload)
    filtered["reproducibility_checksum"] = ""
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:  # pragma: no cover - CLI wrapper
    out_path = run(REPO_ROOT)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
