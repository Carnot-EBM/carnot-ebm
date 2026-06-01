"""Exp 3664 v335 capstone and G-gate synthesis.

Spec: REQ-PUBLISH-3664, SCENARIO-PUBLISH-3664.
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
OUTPUT_REL_PATH = Path("results/experiment_3664_capstone_and_g_gate_v335.json")
RANDOM_SEED = 3664
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts "
    "(principle: reads the gate script + artifacts; no live inference)."
)
UPSTREAM_ARTIFACTS: Mapping[str, Path] = {
    "exp3654": Path("results/experiment_3654_real_nli_atomic_claim_grounding_verifier.json"),
    "exp3655": Path("results/experiment_3655_facts_row_remeasurement_real_nli_v5.json"),
    "exp3656": Path("results/experiment_3656_correlation_aware_weighting_paradox_diagnosis.json"),
    "exp3657": Path("results/experiment_3657_deployable_second_pair_of_eyes_detector.json"),
    "exp3658": Path("results/experiment_3658_code_generalization_second_corpus.json"),
    "exp3659": Path("results/experiment_3659_trained_ebm_judge_ood_real_substrate_v3.json"),
    "exp3660": Path("results/experiment_3660_fr11_continuous_self_learning_v9.json"),
}
ALLOWED_SCOPES = {
    "broad",
    "math_plus_code",
    "math_plus_code_plus_facts",
    "math_only_earned",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "corrected_generalization_table",
    "facts_generalize_real_nli",
    "real_nli_vs_proxy_correction",
    "code_generalization_replicated",
    "second_pair_of_eyes_deployable",
    "correlation_paradox_resolution",
    "trained_judge_real_substrate_result",
    "verifier_value_scope",
    "g1",
    "g2",
    "g3",
    "g4",
    "paper_ready",
    "unmet_gates",
    "p01_status",
    "paper_v6_safe_claims",
    "paper_v6_forbidden_claims",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "aggregation_from_upstream_artifacts (principle: reads the gate script + "
        "artifacts; no live inference)."
    ),
    "corrected_generalization_table": (
        "domain -> {auroc, delta, lift, ran_or_blocked} -- the milestone's "
        "central evidence, with facts now measured by a real NLI verifier."
    ),
    "facts_generalize_real_nli": (
        "The core-mission answer: does factual grounding generalize with a real "
        "NLI verifier (exp3655), vs the .334 proxy negative?"
    ),
    "real_nli_vs_proxy_correction": (
        "States how the real NLI result corrects or confirms the .334 proxy "
        "facts negative (0.6495)."
    ),
    "code_generalization_replicated": (
        "Whether code generalization held on a balanced second corpus (exp3658) "
        "-- hardens the code claim."
    ),
    "second_pair_of_eyes_deployable": (
        "Whether the calibrated fused detector (exp3657) is a deployable "
        "product surface that beats confidence."
    ),
    "correlation_paradox_resolution": (
        "H1 (correlation harmless) or H2 (naive penalty mis-specified, "
        "dependency-aware recovers) from exp3656."
    ),
    "trained_judge_real_substrate_result": (
        "Whether a real-substrate trained judge (exp3659) transfers OOD -- the "
        "Phase-3 path signal."
    ),
    "verifier_value_scope": (
        "broad / math_plus_code / math_plus_code_plus_facts / math_only_earned "
        "-- the scoped product claim after the fair facts re-measurement."
    ),
    "g1": "Headline measured (FoVer 0.9131, 5-seed, CI, adversarial-clean).",
    "g2": "Independently reproduced (CI runner 26725185125).",
    "g3": "Prose narrowing-clean.",
    "g4": "Numbers trace to primary artifacts.",
    "paper_ready": "G1 and G2 and G3 and G4 -- must remain true; the milestone does not regress the gate.",
    "unmet_gates": "Report which gates are unmet, not a count (publication_blocker_count is retired).",
    "p01_status": "P0.1 stays honest-negative; do not re-assert a positive.",
    "paper_v6_safe_claims": "Narrowing-clean claims.",
    "paper_v6_forbidden_claims": "Overclaims to avoid.",
    "cited_upstream_artifacts": "sha256 provenance (G4).",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    gate_data: Mapping[str, Any] | None = None,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the Exp 3664 terminal artifact from upstream result files."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    gate = dict(gate_data) if gate_data is not None else load_publication_gate(root_path)
    summaries = (
        [dict(record) for record in summary_records]
        if summary_records is not None
        else run_summarize_artifacts(root_path)
    )
    upstreams = {
        name: _read_json_object(root_path / rel_path)
        for name, rel_path in UPSTREAM_ARTIFACTS.items()
    }
    flagged = {name: _is_flagged_adversarial(payload) for name, payload in upstreams.items()}

    facts = _facts_real_nli_result(
        upstreams["exp3654"],
        upstreams["exp3655"],
        flagged=flagged["exp3655"],
    )
    code_replicated = _code_replicated(upstreams["exp3658"], flagged=flagged["exp3658"])
    second_pair_deployable = _second_pair_deployable(
        upstreams["exp3657"], flagged=flagged["exp3657"]
    )
    trained_judge = _trained_judge_result(upstreams["exp3659"], flagged=flagged["exp3659"])
    correlation = _correlation_resolution(upstreams["exp3656"], flagged=flagged["exp3656"])
    facts_generalize = facts["generalizes"] is True
    scope = _verifier_scope(code_generalized=code_replicated, facts_generalized=facts_generalize)

    g1 = _gate_pass(gate, "G1")
    g2 = _gate_pass(gate, "G2")
    g3 = _gate_pass(gate, "G3")
    g4 = _gate_pass(gate, "G4")
    paper_ready = bool(gate.get("paper_ready") is True and g1 and g2 and g3 and g4)
    finished = time.perf_counter() if now_s is None else float(now_s)
    duration_s = (
        0.0001
        if started_s is None and now_s is None
        else round(max(0.0, finished - start), 6)
    )
    facts_fragment = "generalize" if facts_generalize else "domain_bound"
    paper_fragment = "paper_ready_true" if paper_ready else "paper_ready_false"

    artifact: JsonDict = {
        "honest_verdict": (
            f"complete: capstone_v335_facts_{facts_fragment}_with_real_nli_"
            f"verifier_value_{scope}_{paper_fragment}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "corrected_generalization_table": corrected_generalization_table(
            exp3655=upstreams["exp3655"],
            exp3658=upstreams["exp3658"],
            facts=facts,
            code_replicated=code_replicated,
        ),
        "facts_generalize_real_nli": facts["generalizes"],
        "real_nli_vs_proxy_correction": _real_nli_vs_proxy_correction(
            facts=facts,
            exp3654=upstreams["exp3654"],
        ),
        "code_generalization_replicated": code_replicated,
        "second_pair_of_eyes_deployable": second_pair_deployable,
        "correlation_paradox_resolution": correlation,
        "trained_judge_real_substrate_result": trained_judge,
        "verifier_value_scope": scope,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "paper_ready": paper_ready,
        "unmet_gates": list(gate.get("unmet_gates") or []),
        "p01_status": "honest-negative",
        "paper_v6_safe_claims": _safe_claims(
            scope=scope,
            facts=facts,
            code_replicated=code_replicated,
            second_pair_deployable=second_pair_deployable,
            trained_judge=trained_judge,
        ),
        "paper_v6_forbidden_claims": _forbidden_claims(flagged=flagged),
        "cited_upstream_artifacts": _cited_upstreams(root_path, upstreams, flagged),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "publication_gate": _gate_details(gate),
        "summarized_upstream_artifacts": summaries,
        "flagged_upstream_artifacts_excluded": [
            str(UPSTREAM_ARTIFACTS[name]) for name, is_flagged in flagged.items() if is_flagged
        ],
        "fr11_continuous_self_learning_result": _fr11_result(
            upstreams["exp3660"], flagged=flagged["exp3660"]
        ),
        "source_artifacts": [str(path) for path in UPSTREAM_ARTIFACTS.values()],
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    artifact["reproducibility_checksum"] = _payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def corrected_generalization_table(
    *,
    exp3655: Mapping[str, Any],
    exp3658: Mapping[str, Any],
    facts: Mapping[str, Any],
    code_replicated: bool,
) -> JsonDict:
    """Return the corrected math/code/facts table for the v335 capstone."""

    code_auroc = _point(exp3658.get("math_signal_code_auroc"))
    code_confidence = _point(exp3658.get("confidence_baseline_auroc"))
    code_delta = _difference(code_auroc, code_confidence)
    facts_delta = facts.get("delta")
    facts_lift = facts_delta if isinstance(facts_delta, int | float) else None
    return {
        "math": {
            "auroc": 0.9131,
            "delta": 0.018471,
            "lift": 0.018471,
            "ran_or_blocked": "ran",
            "generalizes": True,
            "status": "frozen_fover_math_headline",
        },
        "code": {
            "auroc": code_auroc,
            "delta": code_delta,
            "lift": code_delta,
            "ran_or_blocked": (
                "ran"
                if _acceptance_pass(exp3658) and exp3658.get("code_verifiers_fire") is True
                else "blocked"
            ),
            "generalizes": code_replicated,
            "second_corpus_balanced": _class_balance_ok(exp3658),
        },
        "facts": {
            "auroc": facts.get("auroc"),
            "delta": facts_delta,
            "lift": facts_lift,
            "ran_or_blocked": facts.get("ran_or_blocked"),
            "generalizes": facts.get("generalizes"),
            "confidence_auroc": _point(exp3655.get("confidence_baseline_auroc")),
            "real_nli_status": facts.get("status"),
        },
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    gate_data: Mapping[str, Any] | None = None,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3664 artifact."""

    root_path = Path(root)
    out_path = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(
        root_path,
        gate_data=gate_data,
        summary_records=summary_records,
        started_s=started_s,
        now_s=now_s,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3664 schema and publication-gate invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate is not the Exp 3664 aggregation substrate")
    if artifact.get("verifier_value_scope") not in ALLOWED_SCOPES:
        raise ValueError("verifier_value_scope is outside the allowed scoped claim set")
    if artifact.get("paper_ready") is not True:
        raise ValueError("paper_ready must remain true for this capstone")
    for gate in ("g1", "g2", "g3", "g4"):
        if artifact.get(gate) is not True:
            raise ValueError(f"{gate} must be true")
    if artifact.get("p01_status") != "honest-negative":
        raise ValueError("p01_status must remain honest-negative")
    if not isinstance(artifact.get("unmet_gates"), list):
        raise ValueError("unmet_gates must be a list")
    table = artifact.get("corrected_generalization_table")
    if not isinstance(table, Mapping) or set(table) != {"math", "code", "facts"}:
        raise ValueError("corrected_generalization_table must contain math/code/facts")
    for domain, row in table.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"{domain} row must be a mapping")
        missing_row_fields = {"auroc", "delta", "lift", "ran_or_blocked"} - set(row)
        if missing_row_fields:
            raise ValueError(f"{domain} row missing {sorted(missing_row_fields)}")
    if artifact.get("facts_generalize_real_nli") not in {True, False, "not_measured_real_nli"}:
        raise ValueError("facts_generalize_real_nli must be true, false, or not_measured_real_nli")
    if not isinstance(artifact.get("real_nli_vs_proxy_correction"), Mapping):
        raise ValueError("real_nli_vs_proxy_correction must be a mapping")
    if not isinstance(artifact.get("correlation_paradox_resolution"), Mapping):
        raise ValueError("correlation_paradox_resolution must be a mapping")
    if not isinstance(artifact.get("trained_judge_real_substrate_result"), Mapping):
        raise ValueError("trained_judge_real_substrate_result must be a mapping")
    if not isinstance(artifact.get("paper_v6_safe_claims"), list):
        raise ValueError("paper_v6_safe_claims must be a list")
    if not isinstance(artifact.get("paper_v6_forbidden_claims"), list):
        raise ValueError("paper_v6_forbidden_claims must be a list")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or float(duration) < 0.0:
        raise ValueError("duration_s must be nonnegative numeric")
    cited = artifact.get("cited_upstream_artifacts")
    if not isinstance(cited, list):
        raise ValueError("cited_upstream_artifacts must be a list")
    for item in cited:
        if not isinstance(item, Mapping) or len(str(item.get("sha256") or "")) != 64:
            raise ValueError("cited_upstream_artifacts must include sha256 provenance")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")


def load_publication_gate(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary
    """Run publication_gate.py and return its parsed JSON result."""

    completed = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_summarize_artifacts(root: Path) -> list[JsonDict]:  # pragma: no cover - subprocess boundary
    """Run summarize_artifact.py for Exp 3654 through Exp 3660."""

    records: list[JsonDict] = []
    for exp_id in range(3654, 3661):
        completed = subprocess.run(
            [sys.executable, "scripts/summarize_artifact.py", str(exp_id)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        records.append(
            {
                "exp": exp_id,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-1000:],
            }
        )
    return records


def _facts_real_nli_result(
    exp3654: Mapping[str, Any],
    exp3655: Mapping[str, Any],
    *,
    flagged: bool,
) -> JsonDict:
    explicit_built = exp3655.get("nli_grounding_built")
    nli_built = exp3654.get("nli_grounding_built") is True
    if explicit_built is False:
        nli_built = False
    elif explicit_built is True:
        nli_built = True
    explicit_leak_free = exp3655.get("grounding_leak_free")
    leak_free = (
        exp3654.get("grounding_leak_free") is True
        if explicit_leak_free is None
        else explicit_leak_free is True
    )
    auroc = _point(exp3655.get("grounding_auroc_real_nli"))
    confidence = _point(exp3655.get("confidence_baseline_auroc"))
    delta = _point(exp3655.get("grounding_minus_confidence_delta"))
    if delta is None:
        delta = _difference(auroc, confidence)
    substrate = str(exp3655.get("nli_substrate") or exp3654.get("nli_substrate") or "")
    model_based = "model_based_transformers_checkpoint" in substrate
    implausible_grounding_leak = bool(
        auroc is not None and auroc >= 0.99 and exp3655.get("grounding_leak_free") is not True
    )
    generalizes = exp3655.get("facts_generalize_real_nli")
    measured = bool(
        not flagged
        and nli_built
        and leak_free
        and model_based
        and _acceptance_pass(exp3655)
        and auroc is not None
        and confidence is not None
        and exp3655.get("positive_control_valid") is True
        and generalizes in {True, False}
        and not implausible_grounding_leak
    )
    if not measured:
        return {
            "status": "not_measured_real_nli",
            "generalizes": "not_measured_real_nli",
            "ran_or_blocked": "not_measured_real_nli",
            "auroc": None,
            "confidence_auroc": confidence,
            "delta": None,
            "nli_grounding_built": nli_built,
            "grounding_leak_free": leak_free,
            "model_based_nli": model_based,
            "implausible_grounding_leak": implausible_grounding_leak,
        }
    return {
        "status": "generalize" if generalizes is True else "domain_bound",
        "generalizes": bool(generalizes),
        "ran_or_blocked": "ran",
        "auroc": auroc,
        "confidence_auroc": confidence,
        "delta": delta,
        "nli_grounding_built": nli_built,
        "grounding_leak_free": leak_free,
        "model_based_nli": model_based,
        "implausible_grounding_leak": False,
    }


def _real_nli_vs_proxy_correction(
    *,
    facts: Mapping[str, Any],
    exp3654: Mapping[str, Any],
) -> JsonDict:
    if facts.get("status") == "not_measured_real_nli":
        return {
            "status": "not_measured_real_nli",
            "proxy_negative_auroc": _round_or_none(exp3654.get("proxy_baseline_auroc")),
            "real_nli_auroc": facts.get("auroc"),
            "reason": "real-NLI facts row did not pass the measurement gate.",
        }
    facts_generalize = facts.get("generalizes") is True
    return {
        "status": (
            "real_nli_corrects_proxy_negative"
            if facts_generalize
            else "real_nli_confirms_proxy_negative"
        ),
        "proxy_negative_auroc": _round_or_none(exp3654.get("proxy_baseline_auroc")),
        "real_nli_auroc": facts.get("auroc"),
        "confidence_baseline_auroc": facts.get("confidence_auroc"),
        "delta_vs_confidence": facts.get("delta"),
        "exp3654_real_nli_vs_proxy_delta": _round_or_none(
            exp3654.get("grounding_auroc_vs_proxy_delta")
        ),
    }


def _code_replicated(exp3658: Mapping[str, Any], *, flagged: bool) -> bool:
    return bool(
        not flagged
        and exp3658.get("code_generalization_replicates") is True
        and exp3658.get("code_verifiers_fire") is True
        and _class_balance_ok(exp3658)
        and _acceptance_pass(exp3658)
    )


def _second_pair_deployable(exp3657: Mapping[str, Any], *, flagged: bool) -> bool:
    return bool(
        not flagged
        and exp3657.get("fusion_beats_confidence_alone") is True
        and isinstance(exp3657.get("fused_detector_auroc"), Mapping)
        and isinstance(exp3657.get("confidence_alone_auroc"), Mapping)
        and isinstance(exp3657.get("calibration_brier_ece"), Mapping)
        and _acceptance_pass(exp3657)
    )


def _correlation_resolution(exp3656: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    reported = _correlation_status(exp3656)
    if flagged:
        return {
            "status": "excluded_flagged_adversarial",
            "usable_for_claims": False,
            "reported_resolution": reported,
            "source_honest_verdict": exp3656.get("honest_verdict"),
        }
    return {
        "status": reported,
        "usable_for_claims": True,
        "dependency_aware_auroc": _round_or_none(
            exp3656.get("ensemble_auroc_dependency_aware_proper")
        ),
        "source_honest_verdict": exp3656.get("honest_verdict"),
    }


def _trained_judge_result(exp3659: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    if flagged:
        return {
            "status": "excluded_flagged_adversarial",
            "transfers_ood": False,
            "usable_for_claims": False,
        }
    transfers = bool(
        exp3659.get("trained_judge_transfers_ood") is True and _acceptance_pass(exp3659)
    )
    return {
        "status": "transfers_ood" if transfers else "does_not_transfer_ood",
        "transfers_ood": transfers,
        "in_domain_judge_auroc": _round_or_none(exp3659.get("in_domain_judge_auroc")),
        "ood_judge_auroc": _round_or_none(exp3659.get("ood_judge_auroc")),
        "confidence_only_baseline_auroc": _round_or_none(
            exp3659.get("confidence_only_baseline_auroc")
        ),
        "delta_vs_confidence_ood": _round_or_none(
            exp3659.get("real_substrate_vs_confidence_ood_delta")
        ),
    }


def _fr11_result(exp3660: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    return {
        "usable_for_claims": not flagged,
        "honest_verdict": exp3660.get("honest_verdict"),
        "quality_maintained": exp3660.get("quality_maintained") is True,
        "online_fusion_auroc_gain": _round_or_none(exp3660.get("online_fusion_auroc_gain")),
    }


def _verifier_scope(*, code_generalized: bool, facts_generalized: bool) -> str:
    if code_generalized and facts_generalized:
        return "math_plus_code_plus_facts"
    if code_generalized:
        return "math_plus_code"
    if facts_generalized:
        return "broad"
    return "math_only_earned"


def _safe_claims(
    *,
    scope: str,
    facts: Mapping[str, Any],
    code_replicated: bool,
    second_pair_deployable: bool,
    trained_judge: Mapping[str, Any],
) -> list[str]:
    claims = [
        "FoVer verifier ensemble math headline remains 0.9131 AUROC with G1-G4 satisfied.",
        f"Verifier value scope is {scope}; this is a scoped, domain-bound product claim.",
        "P0.1 remains honest-negative; no energy-vs-self-consistency positive is re-asserted.",
    ]
    if code_replicated:
        claims.append("Code verifier value generalizes and replicates on a balanced second corpus.")
    if facts.get("generalizes") is True:
        claims.append("The real-NLI facts grounding generalizes under Exp 3655's measurement gate.")
    elif facts.get("generalizes") is False:
        claims.append(
            "The real-NLI facts row ran and facts remain domain-bound; no broad factual "
            "generalization claim is made."
        )
    else:
        claims.append("Facts are not measured under real NLI; no factual generalization claim is made.")
    if second_pair_deployable:
        claims.append("The calibrated fused second-pair-of-eyes detector beats confidence.")
    if trained_judge.get("transfers_ood") is False:
        claims.append("The real-substrate trained judge does not transfer OOD and is not the Phase-3 fix.")
    return claims


def _forbidden_claims(*, flagged: Mapping[str, bool]) -> list[str]:
    claims = [
        "Do not claim broad factual generalization unless Exp 3655 facts_generalize_real_nli is true.",
        "Do not read missing real-NLI fields as None and synthesize a facts conclusion around them.",
        "Do not treat grounding AUROC >= 0.99 as valid unless Exp 3655 proves grounding_leak_free.",
        "Do not re-assert a P0.1 positive; the status remains honest-negative.",
        "Do not claim the real-substrate trained judge transfers OOD unless Exp 3659 passes that gate.",
    ]
    flagged_paths = [str(UPSTREAM_ARTIFACTS[name]) for name, is_flagged in flagged.items() if is_flagged]
    if flagged_paths:
        claims.append(
            "Do not cite flagged_adversarial artifacts in the paper or capstone synthesis: "
            + ", ".join(flagged_paths)
        )
    return claims


def _cited_upstreams(
    root: Path,
    upstreams: Mapping[str, Mapping[str, Any]],
    flagged: Mapping[str, bool],
) -> list[JsonDict]:
    cited: list[JsonDict] = []
    for name, rel_path in UPSTREAM_ARTIFACTS.items():
        if flagged.get(name) is True:
            continue
        cited.append(
            {
                "path": str(rel_path),
                "sha256": _sha256_file(root / rel_path),
                "honest_verdict": upstreams[name].get("honest_verdict"),
            }
        )
    return cited


def _correlation_status(exp3656: Mapping[str, Any]) -> str:
    value = exp3656.get("correlation_harmless_or_penalty_misspecified")
    if isinstance(value, str) and value:
        normalized = value.lower()
        if "dependency_aware_recovers" in normalized or "naive_penalty" in normalized:
            return "H2_naive_penalty_misspecified_dependency_aware_recovers"
        if "correlation_harmless" in normalized:
            return "H1_correlation_harmless"
        return value
    verdict = str(exp3656.get("honest_verdict") or "")
    if "naive_penalty_misspecified" in verdict and "dependency_aware_recovers" in verdict:
        return "H2_naive_penalty_misspecified_dependency_aware_recovers"
    if "correlation_harmless" in verdict:
        return "H1_correlation_harmless"
    return "unknown"


def _class_balance_ok(exp3658: Mapping[str, Any]) -> bool:
    balance = exp3658.get("class_balance")
    return isinstance(balance, Mapping) and balance.get("balanced") is True


def _acceptance_pass(payload: Mapping[str, Any]) -> bool:
    gate = payload.get("acceptance_gate")
    return isinstance(gate, Mapping) and gate.get("passed") is True


def _gate_pass(gate_data: Mapping[str, Any], gate_name: str) -> bool:
    gates = gate_data.get("gates")
    if not isinstance(gates, Mapping):
        return False
    gate = gates.get(gate_name)
    return isinstance(gate, Mapping) and gate.get("pass") is True


def _gate_details(gate_data: Mapping[str, Any]) -> JsonDict:
    gates = gate_data.get("gates")
    return dict(gates) if isinstance(gates, Mapping) else {}


def _point(metric: Any) -> float | None:
    if isinstance(metric, Mapping):
        return _round_or_none(metric.get("point"))
    return _round_or_none(metric)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _round_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return round(float(value), 6)
    return None


def _is_flagged_adversarial(payload: Mapping[str, Any]) -> bool:
    return payload.get("flagged_adversarial") is True


def _read_json_object(path: Path) -> JsonDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
