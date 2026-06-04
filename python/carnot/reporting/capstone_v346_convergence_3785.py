"""Build the Exp 3785 v346 convergence capstone artifact.

Spec refs: REQ-REPORT-3785, SCENARIO-REPORT-3785,
SCENARIO-REPORT-3785-P1-GUARD, SCENARIO-REPORT-3785-MISSING-BLOCKED,
SCENARIO-REPORT-3785-FLAGGED.

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


OUTPUT_REL_PATH = Path("results/experiment_3785_capstone_v346.json")
RANDOM_SEED = 3785
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a capstone reads upstream JSON, "
    "runs no live model)."
)
UPSTREAM_IDS = tuple(range(3776, 3785))

DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    3776: Path("results/experiment_3776_archive_v345_activate_v346.json"),
    3777: Path("results/experiment_3777_p1_discrete_search_adjudication_v3.json"),
    3778: Path("results/experiment_3778_fr11_self_learning_v18_tier2_constraint_memory.json"),
    3779: Path("results/experiment_3779_abstention_operating_point_product_wiring.json"),
    3780: Path("results/experiment_3780_anomaly_escalation_classifier_prototype.json"),
    3781: Path("results/experiment_3781_edlm_next_thesis_feasibility_scoping.json"),
    3782: Path("results/experiment_3782_technical_report_g4_correction_prep.json"),
    3783: Path("results/experiment_3783_external_research_refresh.json"),
    3784: Path("results/experiment_3784_kv260_opportunistic_continuity_audit.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "p1_adjudication",
    "p1_positive_control_passed",
    "energy_as_generator_still_bounded",
    "verifier_product_banked",
    "anomaly_escalation_prototyped",
    "edlm_seed_scaffolded",
    "fr11_v18_self_learning",
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
        "The exp3777 verdict {decode_artifact_bounded | fundamental | "
        "inconclusive | blocked} -- the milestone-defining mechanistic result."
    ),
    "p1_positive_control_passed": (
        "BARE bool -- whether the AR positive control reached >=0.3; a null "
        "'FUNDAMENTAL' is invalid without it (the v1/v2 lesson)."
    ),
    "energy_as_generator_still_bounded": (
        "BARE bool, true -- the part-b strategic bound stands regardless of the "
        "P1 mechanism verdict."
    ),
    "verifier_product_banked": (
        "Records the abstention mode wired into the verify API (exp3779) -- the "
        "deployable product surface."
    ),
    "anomaly_escalation_prototyped": (
        "BARE bool -- the P3 classifier prototype + change-proposal landed "
        "(recommend-only, conductor unmodified)."
    ),
    "edlm_seed_scaffolded": (
        "Records the EDLM feasibility brief + minimal kill-gate handed to the "
        "operator (exp3781) -- the loop does NOT commit."
    ),
    "fr11_v18_self_learning": (
        "Records the Tier-2 constraint-memory consolidation (exp3778, memory "
        "contribution preserved) -- the self-learning mandate."
    ),
    "paper_ready_preserved": (
        "BARE bool -- G1-G4 stay met; the milestone must not regress the banked "
        "verifier product."
    ),
    "frozen_headline_unchanged": (
        "BARE bool -- frozen FoVer 0.9131 stays frozen; .346 uses but never moves it."
    ),
    "next_thesis_remains_operator_surface": (
        "BARE bool -- the next-Phase-3 decision remains an operator-seeding "
        "surface; the loop does not self-commit."
    ),
    "flagged_artifacts_excluded": (
        "Lists any flagged_adversarial artifact excluded from aggregation "
        "(fabrication gate)."
    ),
    "not_landed_or_blocked_recorded_honestly": (
        "Lists any upstream task that did not land or blocked -- recorded as "
        "not-run/blocked, NOT as a research negative."
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
}

SUMMARY_FIELDS: Mapping[int, tuple[str, ...]] = {
    3776: (
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
    ),
    3777: (
        "honest_verdict",
        "adjudication",
        "p1_adjudication",
        "positive_control_passed",
        "p1_positive_control_passed",
        "energy_as_generator_still_bounded",
        "ar_best",
        "ebt_best",
    ),
    3778: (
        "honest_verdict",
        "continuous_self_learning_task",
        "is_tier2_not_tier1",
        "memory_contribution_preserved",
        "tracker_state_persisted",
        "auroc_within_frozen_ci",
        "acceptance_gate.passed",
    ),
    3779: (
        "honest_verdict",
        "abstention_mode_wired",
        "default_off_preserves_prior_behavior",
        "e2e_abstention_passed",
        "mcp_surface_confirmed",
        "tests_assert_real_behavior",
    ),
    3780: (
        "honest_verdict",
        "classifier_shipped",
        "classifier_only_recommends",
        "never_relaxes_verification",
        "change_proposal_written",
    ),
    3781: (
        "honest_verdict",
        "minimal_kill_gate_design",
        "operator_decision_framing",
        "loop_does_not_commit",
    ),
    3782: (
        "honest_verdict",
        "proposed_correction_written",
        "operator_curated_doc_unedited",
        "unsupported_numbers_identified",
        "real_numbers_confirmed",
    ),
    3783: (
        "honest_verdict",
        "references_added",
        "n_references_added",
        "section_appended_not_replaced",
        "numbers_are_as_reported",
    ),
    3784: (
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
    """Build the v346 capstone from upstream JSON artifacts."""

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

    exp3776 = headline_upstreams.get(3776, {})
    exp3777_clean = clean_upstreams.get(3777)
    exp3778 = headline_upstreams.get(3778, {})
    exp3779 = headline_upstreams.get(3779, {})
    exp3780 = headline_upstreams.get(3780, {})
    exp3781 = headline_upstreams.get(3781, {})
    exp3782 = headline_upstreams.get(3782, {})
    exp3783 = headline_upstreams.get(3783, {})
    exp3784 = headline_upstreams.get(3784, {})

    p1 = p1_state(exp3777_clean, missing=upstreams.get(3777) is None, flagged=3777 in flagged_ids)
    energy_bounded = True
    verifier_product = verifier_product_banked(exp3779)
    anomaly = anomaly_escalation_prototyped(exp3780)
    edlm = edlm_seed_scaffolded(exp3781)
    fr11 = fr11_v18_self_learning(exp3778)
    paper_ready = paper_ready_from_gate(gate_data)
    frozen = frozen_headline_unchanged(exp3776)
    next_operator = edlm and truthy(exp3781.get("loop_does_not_commit"))
    g4 = g4_correction_prepped(exp3782)
    references = references_refreshed(exp3783)
    kv260 = kv260_terminal_confirmed(exp3784)
    duration_s = round(max(0.0001, (time.perf_counter() if now_s is None else float(now_s)) - start), 6)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v346_convergence_3785.v1",
        "experiment_id": "exp3785",
        "honest_verdict": terminal_verdict(
            p1_adjudication=p1["adjudication"],
            energy_bounded=energy_bounded,
            verifier_product=verifier_product,
            anomaly=anomaly,
            edlm=edlm,
            fr11=fr11,
            paper_ready=paper_ready,
            frozen=frozen,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "p1_adjudication": p1["adjudication"],
        "p1_positive_control_passed": p1["positive_control_passed"],
        "p1_mechanism_status": p1["mechanism_status"],
        "energy_as_generator_still_bounded": energy_bounded,
        "energy_as_selector_status": "honest-negative-bounded",
        "energy_as_generator_status": "honest-negative-bounded",
        "verifier_product_banked": verifier_product,
        "anomaly_escalation_prototyped": anomaly,
        "edlm_seed_scaffolded": edlm,
        "fr11_v18_self_learning": fr11,
        "g4_correction_prepped": g4,
        "references_refreshed": references,
        "kv260_terminal_confirmed": kv260,
        "paper_ready_preserved": paper_ready,
        "publication_gate_state": publication_gate_state(gate_data),
        "frozen_headline_unchanged": frozen,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "next_thesis_remains_operator_surface": next_operator,
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
    """Write, adversarial-verify, and rewrite the Exp 3785 artifact."""

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
    """Return schema and honesty errors for the Exp 3785 capstone."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required artifact fields: {', '.join(missing)}")
    if not str(artifact.get("honest_verdict") or "").startswith("complete: capstone_v346_p1_"):
        errors.append("honest_verdict must be a terminal Exp 3785 verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare the v346 aggregation-only substrate")
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
    for field in (
        "verifier_product_banked",
        "anomaly_escalation_prototyped",
        "edlm_seed_scaffolded",
        "fr11_v18_self_learning",
        "next_thesis_remains_operator_surface",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare bool")
    if artifact.get("paper_ready_preserved") is not True:
        errors.append("paper_ready_preserved must be true")
    if artifact.get("frozen_headline_unchanged") is not True:
        errors.append("frozen_headline_unchanged must be true")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        errors.append("flagged_artifacts_excluded must be a list")
    if not isinstance(artifact.get("not_landed_or_blocked_recorded_honestly"), list):
        errors.append("not_landed_or_blocked_recorded_honestly must be a list")
    validate_citations(artifact.get("cited_upstream_artifacts"), errors)
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed must equal 3785")
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


def terminal_verdict(
    *,
    p1_adjudication: str,
    energy_bounded: bool,
    verifier_product: bool,
    anomaly: bool,
    edlm: bool,
    fr11: bool,
    paper_ready: bool,
    frozen: bool,
) -> str:
    """Return the terminal verdict string using classified milestone states."""

    return (
        f"complete: capstone_v346_p1_{p1_adjudication}_"
        f"{'energy_as_generator_still_bounded' if energy_bounded else 'energy_as_generator_not_preserved'}_"
        f"{'verifier_product_banked' if verifier_product else 'verifier_product_not_banked'}_"
        f"{'anomaly_escalation_prototyped' if anomaly else 'anomaly_escalation_not_landed'}_"
        f"{'edlm_scaffolded' if edlm else 'edlm_not_scaffolded'}_"
        f"{'fr11_v18' if fr11 else 'fr11_v18_not_preserved'}_"
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
    elif is_blocked_payload(payload):
        adjudication = normalize_p1_adjudication(payload, default="blocked_upstream_artifact")
        positive = bool(payload.get("positive_control_passed") or payload.get("p1_positive_control_passed"))
    else:
        adjudication = normalize_p1_adjudication(payload, default="blocked_unknown_p1_adjudication")
        positive = bool(payload.get("positive_control_passed") or payload.get("p1_positive_control_passed"))
    mechanism = (
        f"open_for_operator: {adjudication}"
        if adjudication.startswith("blocked_") or adjudication == "inconclusive_positive_control_failed"
        else f"settled_or_sharpened: {adjudication}"
    )
    return {
        "adjudication": adjudication,
        "positive_control_passed": positive,
        "mechanism_status": mechanism,
    }


def normalize_p1_adjudication(payload: Mapping[str, Any], *, default: str) -> str:
    raw = str(payload.get("p1_adjudication") or payload.get("adjudication") or "").strip()
    lowered = raw.lower()
    if lowered in P1_ALLOWED or lowered.startswith("blocked_"):
        return lowered
    if lowered == "fundamental":
        return "fundamental_causal_inductive_bias_gap"
    if lowered == "inconclusive":
        return "inconclusive_positive_control_failed"
    verdict = str(payload.get("honest_verdict") or "").lower()
    if "decode_artifact_bounded" in verdict:
        return "decode_artifact_bounded"
    if "fundamental" in verdict:
        return "fundamental_causal_inductive_bias_gap"
    if "inconclusive" in verdict or "positive_control_failed" in verdict:
        return "inconclusive_positive_control_failed"
    if verdict.startswith("blocked:"):
        suffix = verdict.split(":", 1)[1].strip().replace("-", "_").replace(" ", "_")
        return f"blocked_{suffix}" if suffix else default
    return default


def valid_p1_adjudication(value: Any) -> bool:
    return isinstance(value, str) and (value in P1_ALLOWED or value.startswith("blocked_"))


def is_blocked_payload(payload: Mapping[str, Any]) -> bool:
    verdict = str(payload.get("honest_verdict") or "").lower()
    return verdict.startswith("blocked:") or verdict.startswith("blocked_")


def verifier_product_banked(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("abstention_mode_wired"))
        and truthy(payload.get("default_off_preserves_prior_behavior"))
        and truthy(payload.get("e2e_abstention_passed"))
        and truthy(payload.get("mcp_surface_confirmed"))
        and truthy(payload.get("tests_assert_real_behavior"))
    )


def anomaly_escalation_prototyped(payload: Mapping[str, Any]) -> bool:
    verdict = str(payload.get("honest_verdict") or "").lower()
    return bool(
        payload
        and truthy(payload.get("classifier_shipped"))
        and truthy(payload.get("classifier_only_recommends"))
        and truthy(payload.get("never_relaxes_verification"))
        and truthy(payload.get("change_proposal_written"))
        and "conductor_unmodified" in verdict
    )


def edlm_seed_scaffolded(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and payload.get("minimal_kill_gate_design")
        and payload.get("operator_decision_framing")
        and truthy(payload.get("loop_does_not_commit"))
    )


def fr11_v18_self_learning(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("continuous_self_learning_task"))
        and truthy(payload.get("is_tier2_not_tier1"))
        and truthy(payload.get("memory_contribution_preserved"))
        and truthy(payload.get("tracker_state_persisted"))
        and truthy(payload.get("auroc_within_frozen_ci"))
        and truthy(nested_get(payload, "acceptance_gate.passed"))
    )


def g4_correction_prepped(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("proposed_correction_written"))
        and truthy(payload.get("operator_curated_doc_unedited"))
        and payload.get("unsupported_numbers_identified")
        and payload.get("real_numbers_confirmed")
    )


def references_refreshed(payload: Mapping[str, Any]) -> bool:
    references = payload.get("references_added")
    return bool(
        payload
        and isinstance(references, list)
        and len(references) == 5
        and payload.get("n_references_added") == 5
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


def frozen_headline_unchanged(exp3776: Mapping[str, Any]) -> bool:
    evidence = exp3776.get("paper_ready_evidence")
    return bool(
        isinstance(evidence, Mapping)
        and evidence.get("frozen_headline_unchanged") is True
        and numeric(evidence.get("frozen_headline_auroc")) == FROZEN_FOVER_AUROC
    )


def milestone_outcome(p1_adjudication: str, mechanism_status: str) -> str:
    if p1_adjudication.startswith("blocked_") or p1_adjudication == "inconclusive_positive_control_failed":
        return (
            ".346 is a CONVERGENCE milestone: it banks the verifier product, "
            "builds the endorsed process upgrade, scaffolds the operator's EDLM "
            "seed, and leaves the P1 mechanism open for operator follow-up. It "
            "makes NO new existential claim and re-grinds nothing already bounded."
        )
    return (
        f".346 is a CONVERGENCE milestone: it {mechanism_status}, banks the "
        "verifier product into a deployable surface, builds the endorsed process "
        "upgrade, and scaffolds the operator's EDLM seed. It makes NO new "
        "existential claim and re-grinds nothing already bounded."
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
    return round(float(value), 4)


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
