"""Exp 5152 DiffusionGemma gate reexamination.

This module does not run DiffusionGemma. It audits the reasoning behind the
gate because the operational risk is a premise mix-up: a MuSR text-reasoning
null can be relevant to the broad oracle-distinct verifier thesis, but it is
not the same evidence as an ARC candidate-reranking result. The artifact keeps
that distinction explicit so a later scaling decision depends on
domain-relevant evidence instead of a convenient but mismatched null.

Spec refs: REQ-VERIFY-5152, SCENARIO-VERIFY-5152,
SCENARIO-VERIFY-5152-SUCCESS, SCENARIO-VERIFY-5152-MISSING-5151.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


D1_REL = Path("results/experiment_phase_d_musr_trained_verifier.json")
EXP4245_REL = Path("results/experiment_4245_arc_set_encoder_beats_vote.json")
EXP5151_REL = Path("results/experiment_5151_arc_oracle_distinct_hardening_v472.json")
OUTPUT_REL = Path("results/experiment_5152_diffusiongemma_gate_reexamination_v472.json")
KNOWN_ISSUES_REL = Path("ops/known-issues.md")
CORRIGENDUM_MARKER = "CORRIGENDUM 2026-07-02 (exp5152)"
SCHEMA = "carnot.diffusiongemma_gate_reexamination_5152.v1"
SPEC_REFS = [
    "REQ-VERIFY-5152",
    "SCENARIO-VERIFY-5152",
    "SCENARIO-VERIFY-5152-SUCCESS",
    "SCENARIO-VERIFY-5152-MISSING-5151",
]
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
ALLOWED_RECOMMENDATIONS = {"keep_gated", "ungate_pending_exp5151", "ungate_now"}

FIELD_PRINCIPLES = {
    "d1_claim_vs_exp4245_claim_same_hypothesis": (
        "The precise question this task exists to answer -- conflating two different "
        "domains under one gate is exactly the error class this project has been burned by before."
    ),
    "recommendation": (
        "A clear, actionable recommendation, not just an analysis -- this feeds directly "
        "into whether DiffusionGemma scaling gets queued next milestone."
    ),
    "honest_verdict": "Must start with complete:/complete_/success:/success_.",
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "honest_verdict",
    "d1_claim_vs_exp4245_claim_same_hypothesis",
    "recommendation",
    "domain_conflation_found",
    "d1_claim",
    "exp4245_claim",
    "exp5151_status",
    "diffusiongemma_artifacts",
    "known_issues_corrigendum",
    "field_principles",
    "spec_refs",
    "reproducibility_checksum",
)


def _terminal_prefixed(value: str) -> bool:
    return any(value.startswith(prefix) for prefix in TERMINAL_PREFIXES)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _optional_json(root: Path, rel_path: Path) -> dict[str, Any] | None:
    path = root / rel_path
    return _read_json_object(path) if path.exists() else None


def _claim_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _d1_claim(d1: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(D1_REL),
        "claim_tested": "MuSR reasoning-text embedding-verifier-vs-SC",
        "domain": "MuSR reasoning questions over generated text traces",
        "verifier": "all-MiniLM embeddings plus LogisticRegression over question+reasoning",
        "baseline": "matched self-consistency over the same candidate set",
        "n_questions": d1.get("n_questions"),
        "trained_verifier_accuracy": _claim_float(d1, "trained_verifier_accuracy"),
        "sc_accuracy_matched": _claim_float(d1, "sc_accuracy_matched"),
        "delta_vs_sc": _claim_float(d1, "delta_vs_sc"),
        "delta_ci95": d1.get("delta_ci95"),
        "moat_realized": d1.get("moat_realized") is True,
        "verifier_is_oracle": d1.get("verifier_is_oracle") is True,
        "honest_verdict": str(d1.get("honest_verdict", "")),
    }


def _exp4245_claim(exp4245: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(EXP4245_REL),
        "claim_tested": "ARC-1 Set-Encoder-vs-vote candidate reranking",
        "domain": "ARC candidate pools generated before the reranker chooses a top candidate",
        "verifier": "DeepSets-style Set-Encoder using candidate and cross-candidate features",
        "baseline": "vote@1/self-consistency-style candidate majority",
        "held_out_task_n": exp4245.get("held_out_task_n"),
        "oracle_distinct_beats_vote": exp4245.get("oracle_distinct_beats_vote") is True,
        "set_encoder_minus_vote_delta": _claim_float(exp4245, "set_encoder_minus_vote_delta"),
        "set_encoder_minus_vote_ci95": exp4245.get("set_encoder_minus_vote_ci95"),
        "oracle_at_k": _claim_float(exp4245, "oracle_at_k"),
        "verifier_is_oracle": exp4245.get("verifier_is_oracle") is True,
        "honest_verdict": str(exp4245.get("honest_verdict", "")),
    }


def _exp5151_status(exp5151: dict[str, Any] | None) -> dict[str, Any]:
    if exp5151 is None:
        return {
            "available": False,
            "path": str(EXP5151_REL),
            "reason": "exp5151_absent",
            "supports_ungating": False,
            "hardened_arc_domain_win": False,
            "honest_verdict": None,
            "headline_outcome": None,
        }
    verdict = str(exp5151.get("honest_verdict", ""))
    success = verdict.startswith("success_") or verdict.startswith("success:")
    return {
        "available": True,
        "path": str(EXP5151_REL),
        "reason": "hardened_arc_win" if success else "not_fully_hardened_or_null",
        "supports_ungating": bool(success),
        "hardened_arc_domain_win": bool(success),
        "honest_verdict": verdict,
        "headline_outcome": exp5151.get("headline_outcome"),
        "acceptance_gate": exp5151.get("acceptance_gate"),
        "multiseed_delta_ci95": exp5151.get("multiseed_delta_ci95"),
        "leak_audit_passed": exp5151.get("leak_audit_passed"),
        "cross_game_blocked_reason": exp5151.get("cross_game_blocked_reason"),
        "cross_game_replication_delta": exp5151.get("cross_game_replication_delta"),
        "verifier_is_oracle": exp5151.get("verifier_is_oracle"),
    }


def _recommendation_from_5151(status: dict[str, Any]) -> tuple[str, str]:
    if status["supports_ungating"] is True:
        return (
            "ungate_now",
            "Exp 5151 reports a hardened ARC-domain win; the MuSR null remains a different-domain "
            "result and does not override ARC-domain hardening succeeded evidence.",
        )
    if status["available"] is True:
        return (
            "keep_gated",
            "Exp 5151 is present but not fully hardened, so DiffusionGemma stays gated "
            "for missing decision-grade ARC-domain evidence rather than the MuSR D1 null.",
        )
    return (
        "keep_gated",
        "Exp 5151 is absent and Exp 4245 is still single-seed/as-yet-unhardened, so "
        "DiffusionGemma stays gated for missing decision-grade ARC-domain evidence.",
    )


def _scan_diffusiongemma_artifacts(root: Path) -> dict[str, Any]:
    result_root = root / "results"
    python_root = root / "python" / "carnot"
    json_results = []
    for path in sorted(result_root.glob("*diffusiongemma*.json")):
        if path == root / OUTPUT_REL:
            continue
        payload = _read_json_object(path)
        json_results.append(
            {
                "path": str(path.relative_to(root)),
                "experiment": payload.get("experiment"),
                "status": payload.get("status"),
                "honest_verdict": payload.get("honest_verdict"),
            }
        )
    runtime_files = [
        str(path.relative_to(root))
        for suffix in ("*.log", "*.pid")
        for path in sorted(result_root.glob(f"*diffusiongemma*{suffix[1:]}"))
    ]
    python_modules = [
        str(path.relative_to(root)) for path in sorted(python_root.glob("*diffusiongemma*.py"))
    ]
    return {
        "search_roots": ["results", "python/carnot"],
        "json_results": json_results,
        "json_result_count": len(json_results),
        "runtime_logs_or_pids": runtime_files,
        "python_modules": python_modules,
        "python_module_count": len(python_modules),
        "cached_ready_summary": (
            "DiffusionGemma artifacts are observed as cached JSON/runtime/module records; "
            "Exp 5152 does not execute or scale DiffusionGemma."
        ),
    }


def _corrigendum_text(recommendation: str, status: dict[str, Any]) -> str:
    if recommendation == "ungate_now":
        action = (
            "Recommendation: UNGATE NOW, because the ARC-domain hardening succeeded; the MuSR "
            "D1 null remains relevant only to MuSR reasoning-text verifier selection."
        )
    elif status["available"] is True:
        action = (
            "Recommendation: KEEP GATED, because Exp 5151 is present but not fully hardened; "
            "the gate is still closed for missing decision-grade ARC-domain evidence."
        )
    else:
        action = (
            "Recommendation: KEEP GATED, because Exp 5151 is absent and Exp 4245 remains "
            "single-seed/as-yet-unhardened ARC evidence."
        )
    return (
        f"> **{CORRIGENDUM_MARKER}:** The prior 'DiffusionGemma stays gated' action is "
        "reexamined as a domain-rationale issue. The MuSR D1 null conflated domains if "
        "used as the reason to close the ARC DiffusionGemma gate: D1 tested MuSR "
        "reasoning-text embedding-verifier-vs-SC, while Exp 4245/5151 concern ARC "
        "candidate-pool Set-Encoder-vs-vote reranking. "
        f"{action}"
    )


def _write_known_issues_corrigendum(root: Path, text: str) -> dict[str, str]:
    path = root / KNOWN_ISSUES_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_text(encoding="utf-8") if path.exists() else "# Carnot -- Known Issues\n"
    if CORRIGENDUM_MARKER not in original:
        anchor = "\n\n**Origin:**"
        index = original.find(anchor)
        split_at = index if index >= 0 else len(original.rstrip())
        updated = original[:split_at].rstrip() + "\n\n" + text + "\n" + original[split_at:]
        path.write_text(updated, encoding="utf-8")
    return {"path": str(KNOWN_ISSUES_REL), "marker": CORRIGENDUM_MARKER, "text": text}


def _checksum(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_artifact(root: Path) -> dict[str, Any]:
    d1 = _read_json_object(root / D1_REL)
    exp4245 = _read_json_object(root / EXP4245_REL)
    status_5151 = _exp5151_status(_optional_json(root, EXP5151_REL))
    recommendation, reason = _recommendation_from_5151(status_5151)
    honest_verdict = (
        "success_diffusiongemma_gate_reexamined_arc_hardened_win_ungate_now"
        if recommendation == "ungate_now"
        else "complete_diffusiongemma_gate_reexamined_keep_gated_corrected_arc_evidence"
    )
    corrigendum_text = _corrigendum_text(recommendation, status_5151)
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_5152_diffusiongemma_gate_reexamination_v472",
        "honest_verdict": honest_verdict,
        "d1_claim_vs_exp4245_claim_same_hypothesis": {
            "value": True,
            "principle": FIELD_PRINCIPLES["d1_claim_vs_exp4245_claim_same_hypothesis"],
            "same_underlying_hypothesis": (
                "Both ask whether an oracle-distinct learned verifier can add selection value "
                "over a non-oracle baseline."
            ),
            "not_same_claim": (
                "D1 is MuSR text reasoning over question+reasoning embeddings vs SC; Exp 4245 "
                "is ARC candidate-pool grid reranking vs vote. The broad thesis overlaps, but "
                "the domain-specific gate evidence is not interchangeable."
            ),
        },
        "recommendation": {
            "value": recommendation,
            "principle": FIELD_PRINCIPLES["recommendation"],
            "reason": reason,
        },
        "domain_conflation_found": True,
        "d1_claim": _d1_claim(d1),
        "exp4245_claim": _exp4245_claim(exp4245),
        "exp5151_status": status_5151,
        "diffusiongemma_artifacts": _scan_diffusiongemma_artifacts(root),
        "known_issues_corrigendum": {
            "path": str(KNOWN_ISSUES_REL),
            "marker": CORRIGENDUM_MARKER,
            "text": corrigendum_text,
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not _terminal_prefixed(verdict):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if artifact["domain_conflation_found"] is not True:
        raise ValueError("domain_conflation_found must be bare bool true")
    for field in ("d1_claim_vs_exp4245_claim_same_hypothesis", "recommendation"):
        item = artifact[field]
        if not isinstance(item, dict) or item.get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} must carry the required principle")
    if artifact["recommendation"].get("value") not in ALLOWED_RECOMMENDATIONS:
        raise ValueError("recommendation value is not allowed")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or not checksum.startswith("sha256:") or len(checksum) != 71:
        raise ValueError("reproducibility_checksum must be sha256-prefixed")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles drifted")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs drifted")


def run(repo_root: Path | str = Path(".")) -> dict[str, Any]:
    root = Path(repo_root)
    artifact = _build_artifact(root)
    validate_artifact(artifact)
    _write_known_issues_corrigendum(root, artifact["known_issues_corrigendum"]["text"])
    output_path = root / OUTPUT_REL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    run(Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
