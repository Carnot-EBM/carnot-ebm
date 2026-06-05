"""Build the Exp 3868 v356 moat-durability capstone artifact.

Spec refs: REQ-CAPSTONE-3868, SCENARIO-CAPSTONE-3868.

This is an aggregation artifact. It reads upstream experiment outputs, excludes
fabrication-gated evidence, and states the moat verdict only after conditioning
the exp3859 scissor result on the exp3860 independence audit. The module does
not run a model and does not move the frozen FoVer 0.9131 headline.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script import guard.
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_REL_PATH = Path("results/experiment_3868_capstone_v356.json")
RANDOM_SEED = 3868
FROZEN_FOVER_AUROC = 0.9131
RESIDUAL_CATCH_HIGH_LOWER_BOUND = 0.50
RESIDUAL_CATCH_SUBSUMED_UPPER_BOUND = 0.30
LOW_ERROR_CORRELATION_MAX = 0.20
LDT_TARGET_MARGIN = 0.010
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: capstone reads upstream "
    "JSON plus summarize_artifact status; no live model or hardware action)."
)

UPSTREAM_IDS = tuple(range(3858, 3868))
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    3858: Path("data/step_error_balanced_v2.json"),
    3859: Path("results/experiment_3859_moat_scissor_at_scale_v3.json"),
    3860: Path("results/experiment_3860_verifier_reasoner_independence_audit.json"),
    3861: Path("results/experiment_3861_thinkprm_complementarity.json"),
    3862: Path("results/experiment_3862_graph_grounding_fact_verifier_prototype_v2.json"),
    3863: Path("results/experiment_3863_graph_verifier_facts_complementarity_v2.json"),
    3864: Path("results/experiment_3864_fr11_self_learning_v23_independence_reweighting.json"),
    3865: Path("results/experiment_3865_ldt_lattice_margin_sharpening_v2.json"),
    3866: Path("results/experiment_3866_gatemate_ising_tile_flash_v2.json"),
    3867: Path("results/experiment_3867_polarfire_soc_smoke_v4.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "moat_durability_verdict",
    "moat_is_real_independence",
    "thinkprm_complementarity_summary",
    "facts_new_architecture_outcome",
    "self_learning_v23_outcome",
    "ldt_margin_outcome",
    "hardware_board_states",
    "paper_ready",
    "frozen_fover_auroc_unchanged",
    "artifacts_skipped_flagged",
    "operator_forward_recommendation",
    "preconditions_checked",
    "cited_upstream_artifacts",
    "inference_substrate",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefix milestone result matching the required capstone_v356 "
        "acceptance shape."
    ),
    "moat_durability_verdict": (
        "THE milestone headline -- the scissor verdict CONDITIONED on the "
        "independence audit; the single sentence the operator needs to decide "
        "product-forward strategy."
    ),
    "moat_is_real_independence": (
        "Bare bool -- true only if residual_catch CI lower bound is high AND "
        "reasoner_carnot_error_correlation is low (exp3860); guards against "
        "citing a fake moat (2604.07650)."
    ),
    "thinkprm_complementarity_summary": (
        "Does the cheap ensemble add catch over a strong generative PRM."
    ),
    "facts_new_architecture_outcome": (
        "Did graph-grounding open the earned-negative facts domain (a forward "
        "path off the domain-bound ceiling)?"
    ),
    "self_learning_v23_outcome": (
        "Self-learning invariant held and whether the v23 independence state "
        "preserved the frozen headline CI."
    ),
    "ldt_margin_outcome": (
        "Self-learning invariant held + whether the LDT lattice edge is real "
        "or marginal."
    ),
    "hardware_board_states": (
        "GateMate + PolarFire terminal/partial states -- hardware-continuity "
        "compliance."
    ),
    "paper_ready": "Bare bool -- G1-G4 converged invariant; MUST stay true.",
    "frozen_fover_auroc_unchanged": (
        "Bare bool -- the 0.9131 headline did not move."
    ),
    "artifacts_skipped_flagged": (
        "Which upstream artifacts were excluded for flagged_adversarial==true "
        "(Fabrication Gate transparency)."
    ),
    "operator_forward_recommendation": (
        "The loop scaffolds, the operator decides (Verification Trap P3) -- "
        "the concrete next-paradigm or freeze recommendation given the moat "
        "verdict."
    ),
    "preconditions_checked": (
        "Aggregation substrate (0.0001s floor); records upstream existence, "
        "summarize_artifact return codes, Fabrication Gate status, and "
        "publication-gate evaluation."
    ),
    "cited_upstream_artifacts": (
        "Aggregation substrate (0.0001s floor); full provenance of the "
        "capstone numbers with sha256 for every aggregated upstream artifact."
    ),
    "inference_substrate": (
        "Aggregation substrate (0.0001s floor); no live model or hardware "
        "execution is performed by the capstone."
    ),
    "duration_s": (
        "Aggregation substrate (0.0001s floor); wall-clock duration with a "
        "floor so fast aggregation is still explicit."
    ),
}


def is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase/uppercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def resolve_upstream_path(root: Path | str, experiment_id: int) -> Path:
    """Resolve an upstream experiment id to the path this capstone expects."""

    return Path(root) / DEFAULT_UPSTREAM_PATHS[experiment_id]


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and fail if the artifact is not a mapping."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover - defensive.
    return payload


def sha256_file(path: Path) -> str:
    """Hash an upstream artifact so the capstone can cite exact provenance."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(value: object) -> float | None:
    """Convert JSON scalar metrics to float while rejecting booleans."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ci_low(ci95: object) -> float | None:
    """Extract a lower CI bound from the dict/list shapes used by experiments."""

    if isinstance(ci95, Mapping):
        for key in ("low", "lower", "lower_bound", "ci95_low"):
            found = numeric(ci95.get(key))
            if found is not None:
                return found
    if isinstance(ci95, list | tuple) and ci95:
        return numeric(ci95[0])
    return None


def has_live_critical(report: Mapping[str, Any] | None) -> bool:
    """Return whether adversarial_verify found a live critical flag."""

    if not isinstance(report, Mapping):
        return False
    for flag in report.get("flags", []):
        if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical":
            return True
    return False


def conditioned_moat_verdict(
    scissor: Mapping[str, Any] | None,
    independence_audit: Mapping[str, Any] | None,
) -> JsonDict:
    """Condition exp3859's residual catch on exp3860's error-independence audit."""

    if not isinstance(scissor, Mapping) or not isinstance(independence_audit, Mapping):
        return {
            "verdict": "INCONCLUSIVE",
            "moat_is_real_independence": False,
            "rationale": (
                "exp3859 scissor or exp3860 independence audit is missing, so "
                "residual_catch cannot be cited as a moat."
            ),
            "residual_catch_ci95_low": None,
            "reasoner_carnot_error_correlation": None,
        }

    scissor_verdict = str(scissor.get("honest_verdict", ""))
    audit_verdict = str(independence_audit.get("honest_verdict", ""))
    if scissor_verdict.startswith("blocked") or audit_verdict.startswith("blocked"):
        return {
            "verdict": "INCONCLUSIVE",
            "moat_is_real_independence": False,
            "rationale": (
                "at least one required upstream gate blocked, so the scissor "
                "cannot support a durable-moat conclusion."
            ),
            "residual_catch_ci95_low": ci_low(scissor.get("residual_catch_ci95")),
            "reasoner_carnot_error_correlation": numeric(
                independence_audit.get("reasoner_carnot_error_correlation")
            ),
        }

    residual_low = ci_low(scissor.get("residual_catch_ci95"))
    correlation = numeric(independence_audit.get("reasoner_carnot_error_correlation"))
    independence_is_real = independence_audit.get("independence_is_real") is True
    if residual_low is None or correlation is None:
        return {
            "verdict": "INCONCLUSIVE",
            "moat_is_real_independence": False,
            "rationale": (
                "residual_catch_ci95 or reasoner_carnot_error_correlation is "
                "absent, so the conditioned moat gate cannot fire."
            ),
            "residual_catch_ci95_low": residual_low,
            "reasoner_carnot_error_correlation": correlation,
        }

    low_corr = correlation <= LOW_ERROR_CORRELATION_MAX
    high_residual = residual_low >= RESIDUAL_CATCH_HIGH_LOWER_BOUND
    if high_residual and low_corr and independence_is_real:
        verdict = "MOAT DURABLE"
        rationale = (
            f"residual CI lower bound {residual_low:.3f} is high and error "
            f"correlation {correlation:.3f} is low."
        )
        moat_real = True
    elif high_residual:
        verdict = "MOAT FAKE-INDEPENDENCE"
        rationale = (
            f"residual CI lower bound {residual_low:.3f} is high, but error "
            f"correlation {correlation:.3f} or independence_is_real={independence_is_real} "
            "makes it a fake moat."
        )
        moat_real = False
    elif residual_low <= RESIDUAL_CATCH_SUBSUMED_UPPER_BOUND:
        verdict = "MOAT SUBSUMED"
        rationale = (
            f"residual CI lower bound {residual_low:.3f} is too low to show "
            "durable catch beyond the reasoner."
        )
        moat_real = False
    else:
        verdict = "INCONCLUSIVE"
        rationale = (
            f"residual CI lower bound {residual_low:.3f} sits between the "
            "subsumed and durable thresholds."
        )
        moat_real = False

    return {
        "verdict": verdict,
        "moat_is_real_independence": moat_real,
        "rationale": rationale,
        "residual_catch_ci95_low": residual_low,
        "reasoner_carnot_error_correlation": correlation,
        "residual_catch_rate": numeric(scissor.get("residual_catch_rate")),
        "overlap": scissor.get("overlap"),
    }


def summarize_thinkprm(payload: Mapping[str, Any] | None) -> JsonDict:
    """Summarize exp3861's cheap-ensemble lift over ThinkPRM."""

    if not isinstance(payload, Mapping):
        return {"outcome": "missing_exp3861", "cheap_ensemble_adds_catch": False}
    lift = numeric(payload.get("union_lift_over_thinkprm"))
    adds = payload.get("cheap_ensemble_adds_catch_over_thinkprm") is True or (lift is not None and lift > 0.0)
    return {
        "outcome": "adds_catch_over_thinkprm" if adds else "no_lift_over_thinkprm",
        "cheap_ensemble_adds_catch": adds,
        "union_lift_over_thinkprm": lift,
        "thinkprm_catch_rate": numeric(payload.get("thinkprm_catch_rate")),
        "union_catch_rate": numeric(payload.get("union_catch_rate")),
        "source_verdict": payload.get("honest_verdict"),
    }


def summarize_facts_domain(
    clean_upstreams: Mapping[int, Mapping[str, Any]],
    *,
    any_facts_excluded: bool,
) -> JsonDict:
    """Summarize whether graph-grounding opened the facts-domain ceiling."""

    payload = clean_upstreams.get(3863) or clean_upstreams.get(3862)
    if not isinstance(payload, Mapping):
        outcome = "excluded_flagged_or_live_critical" if any_facts_excluded else "missing_facts_artifact"
        return {"outcome": outcome, "new_architecture_opened": False}
    lift = numeric(payload.get("union_lift_over_math_ensemble"))
    delta = numeric(payload.get("facts_catch_delta"))
    opened = payload.get("extended_ensemble_recommended") is True or (lift is not None and lift > 0.0) or (delta is not None and delta > 0.0)
    return {
        "outcome": "new_architecture_opened" if opened else "domain_ceiling_not_opened",
        "new_architecture_opened": opened,
        "union_lift_over_math_ensemble": lift,
        "facts_catch_delta": delta,
        "graph_facts_catch_rate": numeric(payload.get("graph_facts_catch_rate")),
        "math_ensemble_facts_catch_rate": numeric(payload.get("math_ensemble_facts_catch_rate")),
        "source_verdict": payload.get("honest_verdict"),
    }


def summarize_self_learning(payload: Mapping[str, Any] | None) -> JsonDict:
    """Summarize exp3864's FR-11 v23 invariant."""

    if not isinstance(payload, Mapping):
        return {"outcome": "missing_exp3864", "auroc_in_frozen_ci": False}
    held = payload.get("auroc_in_frozen_ci") is True and payload.get("memory_ablation_contribution_preserved") is True
    return {
        "outcome": "self_learning_v23_invariant_held" if held else "self_learning_v23_regressed",
        "auroc_in_frozen_ci": payload.get("auroc_in_frozen_ci") is True,
        "memory_ablation_contribution_preserved": payload.get("memory_ablation_contribution_preserved") is True,
        "reweighted_ensemble_auroc": numeric(payload.get("reweighted_ensemble_auroc")),
        "frozen_headline_ensemble_auroc": numeric(payload.get("frozen_headline_ensemble_auroc")),
        "state_persisted_path": payload.get("state_persisted_path"),
        "source_verdict": payload.get("honest_verdict"),
    }


def summarize_ldt_margin(payload: Mapping[str, Any] | None) -> JsonDict:
    """Summarize exp3865 and explicitly test whether the 0.010 margin held."""

    if not isinstance(payload, Mapping):
        return {"outcome": "missing_exp3865", "threshold_0_010_held": False}
    margin = numeric(payload.get("ensemble_vs_score_matched_margin"))
    lower = ci_low(payload.get("margin_ci95"))
    threshold_held = margin is not None and margin >= LDT_TARGET_MARGIN and lower is not None and lower > 0.0
    real_edge = lower is not None and lower > 0.0
    if threshold_held:
        edge = "real_at_or_above_0_010"
    elif real_edge:
        edge = "real_but_below_0_010"
    else:
        edge = "marginal_or_unproven"
    return {
        "outcome": edge,
        "edge": edge,
        "threshold_0_010_held": threshold_held,
        "ensemble_vs_score_matched_margin": margin,
        "margin_ci95": payload.get("margin_ci95"),
        "frozen_fover_auroc_unchanged": payload.get("frozen_fover_auroc_unchanged") is True,
        "source_verdict": payload.get("honest_verdict"),
    }


def summarize_hardware(
    clean_upstreams: Mapping[int, Mapping[str, Any]],
    flagged_ids: set[int],
) -> JsonDict:
    """Summarize GateMate and PolarFire terminal/partial states."""

    gatemate = clean_upstreams.get(3866)
    polarfire = clean_upstreams.get(3867)
    if 3866 in flagged_ids:
        gatemate_state = {"state": "excluded_flagged"}
    elif isinstance(gatemate, Mapping):
        flashed = gatemate.get("gatemate_bitstream_flashed") is True or str(gatemate.get("honest_verdict", "")).startswith("success")
        gatemate_state = {"state": "terminal_flashed" if flashed else "partial_or_unproven", "source_verdict": gatemate.get("honest_verdict")}
    else:
        gatemate_state = {"state": "missing_exp3866"}

    if isinstance(polarfire, Mapping):
        terminal = polarfire.get("polarfire_workload_validated") is True and polarfire.get("result_hash_match") is True
        polarfire_state = {
            "state": "terminal_hash_verified" if terminal else "partial_or_unproven",
            "no_fpga_fabric_claim": polarfire.get("no_fpga_fabric_claim") is True,
            "source_verdict": polarfire.get("honest_verdict"),
        }
    else:
        polarfire_state = {"state": "missing_exp3867"}
    return {"gatemate": gatemate_state, "polarfire": polarfire_state}


def operator_recommendation(moat_verdict: str) -> str:
    """Return the operator-facing next action under Verification Trap P3."""

    if moat_verdict == "MOAT DURABLE":
        return (
            "Treat the verifier moat as product-forward evidence, but keep "
            "the loop operator-seeded: freeze FoVer 0.9131 and let the "
            "operator choose the next paradigm."
        )
    if moat_verdict == "MOAT FAKE-INDEPENDENCE":
        return (
            "Do not product-forward the moat; cite the frozen FoVer 0.9131 "
            "headline only, and have the operator choose either a new "
            "architecture path or an explicit freeze."
        )
    if moat_verdict == "MOAT SUBSUMED":
        return (
            "Treat the moat as subsumed by the reasoner; freeze the current "
            "paper-ready headline unless the operator seeds a new architecture."
        )
    return (
        "Do not self-seed a product claim; rerun the missing scissor/audit "
        "chain and let the operator decide the next paradigm after the "
        "conditioned verdict lands."
    )


def slug_for_moat(verdict: str) -> str:
    """Map the moat enum to the required honest_verdict slug."""

    return {
        "MOAT DURABLE": "durable",
        "MOAT SUBSUMED": "subsumed",
        "MOAT FAKE-INDEPENDENCE": "fake_independence",
        "INCONCLUSIVE": "inconclusive",
    }.get(verdict, "inconclusive")


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    adversarial_reports: Mapping[int, Mapping[str, Any]] | None = None,
    publication_gate_data: Mapping[str, Any] | None = None,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the v356 capstone from existing upstream artifacts."""

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
    summaries = (
        {experiment_id: dict(status) for experiment_id, status in summary_statuses.items()}
        if summary_statuses is not None
        else run_summarize_statuses(root_path, paths)
    )
    publication_gate = dict(publication_gate_data) if publication_gate_data is not None else publication_gate_state()

    flagged_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True
    }
    live_critical_ids = {
        experiment_id
        for experiment_id, report in reports.items()
        if has_live_critical(report)
    } - flagged_ids
    clean_upstreams = {
        experiment_id: payload
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping)
        and experiment_id not in flagged_ids
        and experiment_id not in live_critical_ids
    }

    moat = conditioned_moat_verdict(clean_upstreams.get(3859), clean_upstreams.get(3860))
    facts_excluded = bool(({3862, 3863} & flagged_ids) or ({3862, 3863} & live_critical_ids))
    facts = summarize_facts_domain(clean_upstreams, any_facts_excluded=facts_excluded)
    thinkprm = summarize_thinkprm(clean_upstreams.get(3861))
    self_learning = summarize_self_learning(clean_upstreams.get(3864))
    ldt = summarize_ldt_margin(clean_upstreams.get(3865))
    hardware = summarize_hardware(clean_upstreams, flagged_ids)
    paper_ready = publication_gate.get("paper_ready") is True
    frozen_fover_auroc_unchanged = (
        paper_ready
        and (
            self_learning.get("frozen_headline_ensemble_auroc") == FROZEN_FOVER_AUROC
            or ldt.get("frozen_fover_auroc_unchanged") is True
        )
    )

    cited = [
        {
            "experiment_id": experiment_id,
            "path": str(paths[experiment_id].relative_to(root_path)),
            "sha256": sha256_file(paths[experiment_id]),
            "honest_verdict": payload.get("honest_verdict"),
            "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
        }
        for experiment_id, payload in sorted(clean_upstreams.items())
    ]
    skipped_flagged = [
        {
            "experiment_id": experiment_id,
            "path": str(paths[experiment_id].relative_to(root_path)),
            "reason": "flagged_adversarial==true",
            "honest_verdict": upstreams[experiment_id].get("honest_verdict") if isinstance(upstreams[experiment_id], Mapping) else None,
        }
        for experiment_id in sorted(flagged_ids)
    ]
    skipped_live_critical = [
        {
            "experiment_id": experiment_id,
            "path": str(paths[experiment_id].relative_to(root_path)),
            "reason": "live_adversarial_verify_critical",
            "honest_verdict": upstreams[experiment_id].get("honest_verdict") if isinstance(upstreams[experiment_id], Mapping) else None,
        }
        for experiment_id in sorted(live_critical_ids)
    ]
    preconditions = {
        "upstream_artifacts": {
            experiment_id: {
                "path": str(path.relative_to(root_path)),
                "exists": upstreams[experiment_id] is not None,
                "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
                "flagged_adversarial": experiment_id in flagged_ids,
                "live_critical": experiment_id in live_critical_ids,
            }
            for experiment_id, path in paths.items()
        },
        "publication_gate_checked": True,
        "publication_gate_unmet": publication_gate.get("unmet_gates", []),
    }

    moat_slug = slug_for_moat(moat["verdict"])
    independence_slug = "real" if moat["moat_is_real_independence"] else ("fake" if moat["verdict"] == "MOAT FAKE-INDEPENDENCE" else "mixed")
    facts_slug = {
        "new_architecture_opened": "new_architecture_opened",
        "domain_ceiling_not_opened": "not_opened",
        "excluded_flagged_or_live_critical": "excluded",
    }.get(str(facts.get("outcome")), "missing")
    honest_verdict = (
        f"complete: capstone_v356_moat_{moat_slug}_independence_{independence_slug}_"
        f"facts_{facts_slug}_paper_ready_{str(paper_ready).lower()}_"
        f"frozen_headline_{'unchanged' if frozen_fover_auroc_unchanged else 'changed'}"
    )

    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = max(0.0001, end - start)
    artifact: JsonDict = {
        "honest_verdict": honest_verdict,
        "moat_durability_verdict": f"{moat['verdict']} - {moat['rationale']}",
        "moat_is_real_independence": moat["moat_is_real_independence"],
        "moat_conditioning_details": moat,
        "thinkprm_complementarity_summary": thinkprm,
        "facts_new_architecture_outcome": facts,
        "self_learning_v23_outcome": self_learning,
        "ldt_margin_outcome": ldt,
        "hardware_board_states": hardware,
        "paper_ready": paper_ready,
        "publication_gate": publication_gate,
        "frozen_fover_auroc_unchanged": frozen_fover_auroc_unchanged,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "artifacts_skipped_flagged": skipped_flagged,
        "artifacts_skipped_live_critical": skipped_live_critical,
        "operator_forward_recommendation": operator_recommendation(moat["verdict"]),
        "preconditions_checked": preconditions,
        "cited_upstream_artifacts": cited,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact payload while excluding its self-referential checksum."""

    normalized = dict(payload)
    normalized.pop("reproducibility_checksum", None)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the schema constraints that keep this capstone honest."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")  # pragma: no cover - defensive.
    field_principles = artifact.get("field_principles")
    if not isinstance(field_principles, Mapping):
        raise ValueError("field_principles must be a mapping")  # pragma: no cover - defensive.
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in field_principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")  # pragma: no cover - defensive.
    if not str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:", "success:", "failure:")):
        raise ValueError("honest_verdict must start with a terminal prefix")  # pragma: no cover - defensive.
    if not isinstance(artifact.get("moat_is_real_independence"), bool):
        raise ValueError("moat_is_real_independence must be a bare bool")  # pragma: no cover - defensive.
    if not isinstance(artifact.get("paper_ready"), bool):
        raise ValueError("paper_ready must be a bare bool")  # pragma: no cover - defensive.
    if not isinstance(artifact.get("frozen_fover_auroc_unchanged"), bool):
        raise ValueError("frozen_fover_auroc_unchanged must be a bare bool")  # pragma: no cover - defensive.
    if numeric(artifact.get("duration_s")) is None or numeric(artifact.get("duration_s")) < 0.0001:
        raise ValueError("duration_s must respect the 0.0001s floor")  # pragma: no cover - defensive.
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")  # pragma: no cover - defensive.
    for citation in artifact.get("cited_upstream_artifacts", []):
        if not is_sha256(citation.get("sha256")):
            raise ValueError("cited_upstream_artifacts entries need sha256")  # pragma: no cover - defensive.


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: str | Path = OUTPUT_REL_PATH,
    adversarial_reports: Mapping[int, Mapping[str, Any]] | None = None,
    publication_gate_data: Mapping[str, Any] | None = None,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 3868 capstone artifact."""

    root_path = Path(root)
    artifact = build_artifact(
        root_path,
        adversarial_reports=adversarial_reports,
        publication_gate_data=publication_gate_data,
        summary_statuses=summary_statuses,
        started_s=started_s,
        now_s=now_s,
    )
    validate_artifact(artifact)
    output = root_path / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def verify_upstreams(paths: Mapping[int, Path]) -> dict[int, JsonDict]:  # pragma: no cover - host verifier wrapper.
    """Run the live adversarial verifier over upstream artifacts."""

    from scripts import adversarial_verify

    reports: dict[int, JsonDict] = {}
    for experiment_id, path in paths.items():
        reports[experiment_id] = adversarial_verify.verify_artifact(path) if path.exists() else {"flags": []}
    return reports


def publication_gate_state() -> JsonDict:  # pragma: no cover - thin wrapper around existing gate.
    """Evaluate the stable G1-G4 publication gate."""

    from scripts import publication_gate

    return publication_gate.evaluate()


def run_summarize_statuses(root: Path, paths: Mapping[int, Path]) -> dict[int, JsonDict]:  # pragma: no cover - subprocess IO.
    """Run summarize_artifact.py for each available upstream path."""

    statuses: dict[int, JsonDict] = {}
    summarizer = root / "scripts" / "summarize_artifact.py"
    for experiment_id, path in paths.items():
        if not path.exists():
            statuses[experiment_id] = {"returncode": 1, "missing": True}
            continue
        result = subprocess.run(
            [sys.executable, str(summarizer), str(path.relative_to(root))],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        statuses[experiment_id] = {
            "returncode": result.returncode,
            "summary_excerpt": result.stdout[:1000],
            "stderr_excerpt": result.stderr[:1000],
        }
    return statuses


def main() -> int:  # pragma: no cover - CLI wrapper.
    output = write_artifact(REPO_ROOT)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
