"""Exp 3677 v336 capstone and G-gate synthesis.

Spec: REQ-PUBLISH-3677, SCENARIO-PUBLISH-3677.
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
OUTPUT_REL_PATH = Path("results/experiment_3677_capstone_and_g_gate_v336.json")
RANDOM_SEED = 3677
FROZEN_FOVER_HEADLINE_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts "
    "(principle: reads the gate script + artifacts; no live inference)."
)

UPSTREAM_ARTIFACTS: Mapping[str, Path] = {
    "exp3667": Path("results/experiment_3667_dependency_aware_weighting_clean.json"),
    "exp3668": Path("results/experiment_3668_dependency_aware_weighting_heldout.json"),
    "exp3669": Path("results/experiment_3669_build_real_factual_corpus.json"),
    "exp3670": Path("results/experiment_3670_facts_row_real_benchmark.json"),
    "exp3671": Path("results/experiment_3671_ship_second_pair_of_eyes_detector.json"),
    "exp3672": Path("results/experiment_3672_ensemble_selection_where_sc_weak.json"),
    "exp3673": Path("results/experiment_3673_fr11_continuous_self_learning_v10.json"),
}

DEPENDENCY_STATUSES = {
    "clean_and_heldout_validated",
    "clean_but_overfit",
    "no_significant_gain",
    "flagged_still",
}
FACTS_VERDICTS = {
    "generalizes_real",
    "auroc_parity_with_catch_value",
    "domain_bound_real_earned",
    "not_measured",
}
SC_WEAK_RESULTS = {
    "ensemble_adds_value",
    "no_value_with_headroom",
    "no_headroom",
    "not_measured",
}
FR11_RESULTS = {
    "held_no_collapse_quality_maintained",
    "collapse_or_quality_regression",
    "not_measured",
}
BASE_SCOPES = {"math_plus_code", "math_plus_code_plus_facts", "math_only_earned"}
ALLOWED_SCOPES = {
    f"{base}_sc_weak_{sc_result}"
    for base in BASE_SCOPES
    for sc_result in SC_WEAK_RESULTS
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "dependency_aware_headline_candidate_status",
    "facts_real_benchmark_verdict",
    "facts_real_vs_synthetic",
    "second_pair_of_eyes_shipped",
    "sc_weak_selection_direction_result",
    "fr11_v10_result",
    "verifier_value_scope",
    "g1",
    "g2",
    "g3",
    "g4",
    "paper_ready",
    "unmet_gates",
    "p01_status",
    "trained_judge_ood_retired",
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
    "dependency_aware_headline_candidate_status": (
        "clean_and_heldout_validated / clean_but_overfit / no_significant_gain / "
        "flagged_still -- the status of the .336 headline-advancement lead "
        "(exp3667/3668)."
    ),
    "facts_real_benchmark_verdict": (
        "generalizes_real / auroc_parity_with_catch_value / "
        "domain_bound_real_earned / not_measured -- the corrected core-mission "
        "answer (exp3670)."
    ),
    "facts_real_vs_synthetic": (
        "States how the REAL-corpus facts result corrects or confirms the .335 "
        "synthetic negative."
    ),
    "second_pair_of_eyes_shipped": (
        "Whether the calibrated fused detector is wired to a real surface with a "
        "passing E2E test (exp3671) -- Phase-1 product advancement."
    ),
    "sc_weak_selection_direction_result": (
        "ensemble_adds_value / no_value_with_headroom / no_headroom / "
        "not_measured -- the NEW-direction result (exp3672)."
    ),
    "fr11_v10_result": "Whether online dependency-aware weighting held without collapse (exp3673).",
    "verifier_value_scope": (
        "math_plus_code / math_plus_code_plus_facts / math_only_earned + the "
        "SC-weak selection finding -- the scoped product claim after the fair "
        "re-measurements."
    ),
    "g1": "Headline measured (FoVer 0.9131, 5-seed, CI, adversarial-clean).",
    "g2": "Independently reproduced (CI runner 26725185125).",
    "g3": "Prose narrowing-clean.",
    "g4": "Numbers trace to primary artifacts.",
    "paper_ready": "G1 and G2 and G3 and G4 -- must remain true; the milestone does not regress the gate.",
    "unmet_gates": "Report which gates are unmet, not a count (publication_blocker_count is retired).",
    "p01_status": "P0.1 stays honest-negative; do not re-assert a positive.",
    "trained_judge_ood_retired": "Records the trained-judge-OOD hypothesis as retired (exp3659 same-verdict).",
    "paper_v6_safe_claims": "Narrowing-clean claims.",
    "paper_v6_forbidden_claims": (
        "Overclaims to avoid (including: do NOT cite the dependency-aware win as "
        "the headline until re-frozen + re-reproduced)."
    ),
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
    """Build the Exp 3677 terminal artifact from upstream result files."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    gate = dict(gate_data) if gate_data is not None else load_publication_gate(root_path)
    summaries = (
        [dict(record) for record in summary_records]
        if summary_records is not None
        else run_summarize_artifacts(root_path)
    )
    upstreams = {
        name: _read_optional_json_object(root_path / rel_path)
        for name, rel_path in UPSTREAM_ARTIFACTS.items()
    }
    stamped_flagged = {
        name: _is_flagged_adversarial(payload) for name, payload in upstreams.items()
    }

    dependency = _dependency_candidate(
        upstreams["exp3667"],
        upstreams["exp3668"],
        exp3667_flagged=stamped_flagged["exp3667"],
        exp3668_flagged=stamped_flagged["exp3668"],
    )
    facts = _facts_verdict(upstreams["exp3670"], flagged=stamped_flagged["exp3670"])
    second_pair_shipped = _second_pair_shipped(
        upstreams["exp3671"], flagged=stamped_flagged["exp3671"]
    )
    sc_weak = _sc_weak_result(upstreams["exp3672"], flagged=stamped_flagged["exp3672"])
    fr11 = _fr11_result(upstreams["exp3673"], flagged=stamped_flagged["exp3673"])
    scope = _verifier_scope(
        facts_verdict=facts["verdict"],
        second_pair_shipped=second_pair_shipped,
        sc_weak_result=sc_weak,
    )

    g1 = _gate_pass(gate, "G1")
    g2 = _gate_pass(gate, "G2")
    g3 = _gate_pass(gate, "G3")
    g4 = _gate_pass(gate, "G4")
    paper_ready = bool(gate.get("paper_ready") is True and g1 and g2 and g3 and g4)
    finished = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, finished - start), 6)

    artifact: JsonDict = {
        "honest_verdict": (
            "complete: capstone_v336_dependency_aware_"
            f"{dependency['status']}_facts_real_{facts['verdict']}_"
            f"detector_shipped_{str(second_pair_shipped).lower()}_"
            f"paper_ready_{str(paper_ready).lower()}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "dependency_aware_headline_candidate_status": dependency["status"],
        "facts_real_benchmark_verdict": facts["verdict"],
        "facts_real_vs_synthetic": _facts_real_vs_synthetic(facts),
        "second_pair_of_eyes_shipped": second_pair_shipped,
        "sc_weak_selection_direction_result": sc_weak,
        "fr11_v10_result": fr11,
        "verifier_value_scope": scope,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "paper_ready": paper_ready,
        "unmet_gates": list(gate.get("unmet_gates") or []),
        "p01_status": "honest-negative",
        "trained_judge_ood_retired": True,
        "paper_v6_safe_claims": _safe_claims(
            dependency_status=dependency["status"],
            facts_verdict=facts["verdict"],
            second_pair_shipped=second_pair_shipped,
            sc_weak_result=sc_weak,
            fr11_result=fr11,
        ),
        "paper_v6_forbidden_claims": _forbidden_claims(
            stamped_flagged=stamped_flagged,
            exp3667_clean=dependency["exp3667_adversarial_verify_clean"],
            facts=facts,
        ),
        "cited_upstream_artifacts": _cited_upstreams(root_path, upstreams, stamped_flagged),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_fover_headline_auroc": FROZEN_FOVER_HEADLINE_AUROC,
        "dependency_aware_candidate": dependency,
        "facts_real_benchmark": facts,
        "publication_gate": _gate_details(gate),
        "summarized_upstream_artifacts": summaries,
        "flagged_upstream_artifacts_excluded": [
            str(UPSTREAM_ARTIFACTS[name])
            for name, is_flagged in stamped_flagged.items()
            if is_flagged
        ],
        "source_artifacts": [
            str(path)
            for name, path in UPSTREAM_ARTIFACTS.items()
            if upstreams.get(name)
        ],
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    artifact["reproducibility_checksum"] = _payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    gate_data: Mapping[str, Any] | None = None,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3677 artifact."""

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
    """Validate the required Exp 3677 schema and publication-gate invariants."""

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
        raise ValueError("inference_substrate is not the Exp 3677 aggregation substrate")
    if artifact.get("dependency_aware_headline_candidate_status") not in DEPENDENCY_STATUSES:
        raise ValueError("dependency_aware_headline_candidate_status is unsupported")
    if artifact.get("facts_real_benchmark_verdict") not in FACTS_VERDICTS:
        raise ValueError("facts_real_benchmark_verdict is unsupported")
    if artifact.get("sc_weak_selection_direction_result") not in SC_WEAK_RESULTS:
        raise ValueError("sc_weak_selection_direction_result is unsupported")
    if artifact.get("fr11_v10_result") not in FR11_RESULTS:
        raise ValueError("fr11_v10_result is unsupported")
    if artifact.get("verifier_value_scope") not in ALLOWED_SCOPES:
        raise ValueError("verifier_value_scope is outside the allowed scoped claim set")
    if artifact.get("second_pair_of_eyes_shipped") not in {True, False}:
        raise ValueError("second_pair_of_eyes_shipped must be a bare boolean")
    if artifact.get("paper_ready") is not True:
        raise ValueError("paper_ready must remain true for this capstone")
    for gate in ("g1", "g2", "g3", "g4"):
        if artifact.get(gate) is not True:
            raise ValueError(f"{gate} must be true")
    if not isinstance(artifact.get("unmet_gates"), list):
        raise ValueError("unmet_gates must be a list")
    if artifact.get("p01_status") != "honest-negative":
        raise ValueError("p01_status must remain honest-negative")
    if artifact.get("trained_judge_ood_retired") is not True:
        raise ValueError("trained_judge_ood_retired must remain true")
    if not isinstance(artifact.get("paper_v6_safe_claims"), list):
        raise ValueError("paper_v6_safe_claims must be a list")
    if not isinstance(artifact.get("paper_v6_forbidden_claims"), list):
        raise ValueError("paper_v6_forbidden_claims must be a list")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or float(duration) < 0.0:
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
    """Run summarize_artifact.py for Exp 3667 through Exp 3673."""

    records: list[JsonDict] = []
    for exp_id in range(3667, 3674):
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


def _dependency_candidate(
    exp3667: Mapping[str, Any],
    exp3668: Mapping[str, Any],
    *,
    exp3667_flagged: bool,
    exp3668_flagged: bool,
) -> JsonDict:
    exp3667_clean = exp3667.get("adversarial_verify_clean") is True
    exp3667_significant = bool(
        not exp3667_flagged
        and exp3667_clean
        and _acceptance_pass(exp3667)
        and exp3667.get("dependency_aware_beats_carnot") is True
        and _significant_positive(
            exp3667.get("dependency_aware_vs_carnot_delta_ci"),
            exp3667.get("delong_p_dependency_vs_carnot"),
        )
    )
    if exp3667_flagged or not exp3667_clean:
        status = "flagged_still"
    elif not exp3667_significant:
        status = "no_significant_gain"
    elif exp3668_flagged:
        status = "flagged_still"
    else:
        heldout_validated = bool(
            exp3668
            and _acceptance_pass(exp3668)
            and exp3668.get("dependency_aware_generalizes_heldout") is True
            and _significant_positive(
                exp3668.get("heldout_delta_ci"),
                exp3668.get("heldout_delong_p"),
            )
        )
        status = "clean_and_heldout_validated" if heldout_validated else "clean_but_overfit"

    if not exp3667_significant and not exp3668:
        heldout_status = "not_measured"
        missing_reason = "exp3668 skipped_or_missing_after_exp3667_no_significant_gain"
    elif exp3668_flagged:
        heldout_status = "excluded_flagged_adversarial"
        missing_reason = None
    elif not exp3668:
        heldout_status = "not_measured"
        missing_reason = "exp3668 skipped_or_missing"
    elif status == "clean_and_heldout_validated":
        heldout_status = "validated"
        missing_reason = None
    else:
        heldout_status = "overfit_or_not_validated"
        missing_reason = None

    return {
        "status": status,
        "exp3667_adversarial_verify_clean": exp3667_clean,
        "exp3667_flagged_adversarial": exp3667_flagged,
        "exp3667_dependency_aware_beats_carnot": exp3667.get("dependency_aware_beats_carnot") is True,
        "exp3667_significant": exp3667_significant,
        "exp3667_auroc_dependency_aware_proper": _point(
            exp3667.get("auroc_dependency_aware_proper")
        ),
        "exp3667_auroc_carnot_current": _point(exp3667.get("auroc_carnot_current")),
        "exp3667_delong_p_dependency_vs_carnot": _point(
            exp3667.get("delong_p_dependency_vs_carnot")
        ),
        "heldout_status": heldout_status,
        "heldout_missing_reason": missing_reason,
        "heldout_auroc_dependency_aware": _point(
            exp3668.get("heldout_auroc_dependency_aware")
        ),
        "heldout_auroc_carnot": _point(exp3668.get("heldout_auroc_carnot")),
        "heldout_delong_p": _point(exp3668.get("heldout_delong_p")),
        "claim_boundary": (
            "headline-advancement candidate pending re-freeze + re-reproduction "
            "in a future milestone"
        ),
    }


def _facts_verdict(exp3670: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    auroc = _point(exp3670.get("grounding_auroc_real_corpus"))
    confidence = _point(exp3670.get("confidence_baseline_auroc"))
    leak_free = exp3670.get("grounding_leak_free") is True
    leak_risk = bool(auroc is not None and auroc >= 0.99 and not leak_free)
    measured = bool(
        not flagged
        and exp3670
        and _acceptance_pass(exp3670)
        and exp3670.get("positive_control_valid") is True
        and leak_free
        and auroc is not None
        and confidence is not None
        and not leak_risk
    )
    outcome = str(exp3670.get("honest_outcome") or "")
    if not measured:
        verdict = "not_measured"
    elif outcome == "generalizes_real":
        verdict = "generalizes_real"
    elif outcome == "catch_value_at_parity" or exp3670.get("catch_value_at_parity") is True:
        verdict = "auroc_parity_with_catch_value"
    elif outcome == "domain_bound_real":
        verdict = "domain_bound_real_earned"
    else:
        verdict = "not_measured"
    return {
        "verdict": verdict,
        "honest_outcome": exp3670.get("honest_outcome"),
        "grounding_auroc_real_corpus": auroc,
        "confidence_baseline_auroc": confidence,
        "grounding_minus_confidence_delta": _point(
            exp3670.get("grounding_minus_confidence_delta")
        ),
        "grounding_leak_free": leak_free,
        "grounding_leak_risk": leak_risk,
        "positive_control_valid": exp3670.get("positive_control_valid") is True,
        "real_vs_synthetic_grounding_delta": exp3670.get("real_vs_synthetic_grounding_delta"),
    }


def _second_pair_shipped(exp3671: Mapping[str, Any], *, flagged: bool) -> bool:
    return bool(
        not flagged
        and exp3671.get("detector_shipped") is True
        and exp3671.get("e2e_test_passed") is True
        and _acceptance_pass(exp3671)
    )


def _sc_weak_result(exp3672: Mapping[str, Any], *, flagged: bool) -> str:
    if flagged or not exp3672:
        return "not_measured"
    if exp3672.get("positive_control_valid") is not True:
        return "no_headroom"
    if exp3672.get("ensemble_adds_selection_value_sc_weak") is True:
        return "ensemble_adds_value"
    if _acceptance_pass(exp3672):
        return "no_value_with_headroom"
    return "not_measured"


def _fr11_result(exp3673: Mapping[str, Any], *, flagged: bool) -> str:
    if flagged or not exp3673:
        return "not_measured"
    if (
        exp3673.get("collapse_detected_deploy_arm") is False
        and exp3673.get("quality_maintained") is True
        and exp3673.get("pass_rate_vs_true_accuracy_distinct_assert") is True
        and _acceptance_pass(exp3673)
    ):
        return "held_no_collapse_quality_maintained"
    return "collapse_or_quality_regression"


def _verifier_scope(
    *,
    facts_verdict: str,
    second_pair_shipped: bool,
    sc_weak_result: str,
) -> str:
    facts_add_value = facts_verdict in {"generalizes_real", "auroc_parity_with_catch_value"}
    if second_pair_shipped and facts_add_value:
        base = "math_plus_code_plus_facts"
    elif second_pair_shipped:
        base = "math_plus_code"
    elif facts_add_value:
        base = "math_plus_code_plus_facts"
    else:
        base = "math_only_earned"
    return f"{base}_sc_weak_{sc_weak_result}"


def _facts_real_vs_synthetic(facts: Mapping[str, Any]) -> str:
    verdict = facts.get("verdict")
    if verdict == "generalizes_real":
        return "REAL-corpus facts correct the .335 synthetic negative: factual verifier value generalizes on the real benchmark."
    if verdict == "auroc_parity_with_catch_value":
        return "REAL-corpus facts partially correct the .335 synthetic negative: AUROC is at parity, but conditional catch value is measured."
    if verdict == "domain_bound_real_earned":
        return "REAL-corpus facts confirm the .335 synthetic negative as earned: factual verifier value remains domain-bound on the real benchmark."
    return "REAL-corpus facts are not measured; the .335 synthetic negative is not overwritten."


def _safe_claims(
    *,
    dependency_status: str,
    facts_verdict: str,
    second_pair_shipped: bool,
    sc_weak_result: str,
    fr11_result: str,
) -> list[str]:
    claims = [
        "FoVer headline remains frozen at 0.9131 AUROC with G1-G4 satisfied.",
        "P0.1 remains honest-negative; no energy-vs-self-consistency positive is re-asserted.",
        "The trained-judge-OOD hypothesis is retired and is not re-asserted.",
    ]
    if dependency_status == "clean_and_heldout_validated":
        claims.append(
            "Dependency-aware weighting is a headline-advancement candidate pending re-freeze and re-reproduction in a future milestone."
        )
    elif dependency_status == "clean_but_overfit":
        claims.append("Dependency-aware weighting is clean but not held-out validated.")
    elif dependency_status == "no_significant_gain":
        claims.append("Dependency-aware weighting does not provide a significant headline advancement.")
    else:
        claims.append("Dependency-aware weighting remains flagged or unclean and is not claimable.")
    if facts_verdict == "domain_bound_real_earned":
        claims.append("The real facts benchmark records an earned domain-bound negative.")
    elif facts_verdict == "generalizes_real":
        claims.append("The real facts benchmark supports factual verifier generalization.")
    elif facts_verdict == "auroc_parity_with_catch_value":
        claims.append("The real facts benchmark supports scoped catch value at AUROC parity.")
    else:
        claims.append("Facts-real value is not measured; no factual generalization claim is made.")
    if second_pair_shipped:
        claims.append("The calibrated second-pair-of-eyes detector is shipped with a passing E2E surface.")
    claims.append(f"SC-weak ensemble-selection result: {sc_weak_result}.")
    claims.append(f"FR-11 v10 result: {fr11_result}.")
    return claims


def _forbidden_claims(
    *,
    stamped_flagged: Mapping[str, bool],
    exp3667_clean: bool,
    facts: Mapping[str, Any],
) -> list[str]:
    claims = [
        "Do not cite the dependency-aware win as the headline until it is re-frozen and independently re-reproduced.",
        "Do not overwrite the frozen FoVer 0.9131 headline with the dependency-aware candidate.",
        "Do not re-assert a P0.1 positive; the status remains honest-negative.",
        "Do not re-assert the trained-judge-OOD hypothesis; it is retired.",
        "Do not read missing gated-task fields as None and synthesize a result around them.",
        "Do not treat grounding AUROC >= 0.99 as valid unless leak-free evidence is explicit.",
    ]
    if facts.get("verdict") != "generalizes_real":
        claims.append("Do not claim broad factual generalization from the real facts benchmark.")
    if not exp3667_clean:
        claims.append("Do not cite Exp 3667's number unless adversarial_verify_clean is true.")
    flagged_paths = [
        str(UPSTREAM_ARTIFACTS[name])
        for name, is_flagged in stamped_flagged.items()
        if is_flagged
    ]
    if flagged_paths:
        claims.append(
            "Do not cite flagged_adversarial artifacts in paper-v6 claims: "
            + ", ".join(flagged_paths)
        )
    return claims


def _cited_upstreams(
    root: Path,
    upstreams: Mapping[str, Mapping[str, Any]],
    stamped_flagged: Mapping[str, bool],
) -> list[JsonDict]:
    cited: list[JsonDict] = []
    for name, rel_path in UPSTREAM_ARTIFACTS.items():
        payload = upstreams.get(name) or {}
        if not payload or stamped_flagged.get(name) is True:
            continue
        if name == "exp3667" and payload.get("adversarial_verify_clean") is not True:
            continue
        cited.append(
            {
                "path": str(rel_path),
                "sha256": _sha256_file(root / rel_path),
                "honest_verdict": payload.get("honest_verdict"),
            }
        )
    return cited


def _acceptance_pass(payload: Mapping[str, Any]) -> bool:
    gate = payload.get("acceptance_gate")
    if not isinstance(gate, Mapping):
        return False
    if gate.get("passed") is True:
        return True
    return gate.get("required_fields_present") is True


def _significant_positive(delta_ci: Any, p_value: Any) -> bool:
    if not isinstance(delta_ci, Mapping):
        return False
    point = _point(delta_ci)
    ci95 = delta_ci.get("ci95")
    lower = ci95[0] if isinstance(ci95, list | tuple) and ci95 else None
    return bool(
        point is not None
        and point > 0.0
        and isinstance(lower, int | float)
        and not isinstance(lower, bool)
        and float(lower) > 0.0
        and isinstance(p_value, int | float)
        and not isinstance(p_value, bool)
        and float(p_value) < 0.05
    )


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


def _round_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return round(float(value), 6)
    return None


def _is_flagged_adversarial(payload: Mapping[str, Any]) -> bool:
    return payload.get("flagged_adversarial") is True


def _read_optional_json_object(path: Path) -> JsonDict:
    if not path.exists():
        return {}
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
