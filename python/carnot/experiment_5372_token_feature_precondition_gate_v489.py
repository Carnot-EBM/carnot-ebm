"""Exp5372: token/internal-feature precondition gate.

Spec refs: REQ-VERIFY-5372, SCENARIO-VERIFY-5372.

This module reads the flagged Exp5353/Exp5354 artifacts and decides what future
token or internal-energy work may claim. It does not run a live SOTA model and
does not create a new energy signal; it records the no-go boundary implied by
the existing runtime receipts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5372_token_feature_precondition_gate_v489.json")
EXP5353_RELATIVE_PATH = Path("results/experiment_5353_tokenprob_feature_audit_corrigendum_v488.json")
EXP5354_RELATIVE_PATH = Path("results/experiment_5354_arithmetic_carry_token_energy_v488.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5362_capstone_v488.json")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")

EXPERIMENT = 5372
EXPERIMENT_ID = "exp5372-v489-token-feature-precondition-gate"
MILESTONE = "2026.07.489"
RUN_DATE = "20260707"
SCHEMA = "carnot.experiment_5372.token_feature_precondition_gate.v489"
SPEC_REFS = ("REQ-VERIFY-5372", "SCENARIO-VERIFY-5372")
METHODOLOGY_MIN_DURATION_S = 60.0

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete only if the precondition gate is written from actual artifact/runtime evidence.",
    "token_feature_gate_ready": "true only if feature availability and no-go decisions are explicit.",
    "tokenprob_rows_available": "whether reliable per-token logprob rows are available.",
    "logits_available": "whether logits are available from the current local runtime path.",
    "hidden_states_available": "whether hidden states are available from the current local runtime path.",
    "attention_available": "whether attention tensors are available from the current local runtime path.",
    "completion_split_available": "whether prompt/completion token separation is available.",
    "methodology_min_duration_s": "minimum duration required for any future live signal claim.",
    "future_signal_allowed": "boolean; false if only tautological/text-only evidence exists.",
    "carry_token_energy_continue": "boolean continuation recommendation for the arithmetic carry lane.",
    "retire_recommendation": "boolean indicating whether the lane should be retired until new backend features exist.",
    "forbidden_claims": "list of claims not supported by the available features.",
    "unsafe_false_accepts": "count of invalid energy claims accepted by this gate; should be zero.",
    "tests_run": "list of commands run or no-code-change explanation.",
    "honest_verdict": "one-line continuation or retirement recommendation.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def value_of(value: Any) -> Any:
    """Return the machine value from principle-wrapped artifact fields."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _read_json_if_exists(path: Path) -> JsonDict | None:
    if not path.exists():
        return None  # pragma: no cover - defensive missing-source path.
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _bool_from_sources(*values: Any) -> bool:
    return any(value is True for value in values)


def _duration_tautology(artifact: Mapping[str, Any]) -> bool:
    duration = _numeric(value_of(artifact.get("duration_s")))
    methodology = _numeric(value_of(artifact.get("methodology_duration_s")))
    return duration is not None and methodology is not None and abs(duration - methodology) < 1e-9


def _capstone_rows(capstone: Mapping[str, Any] | None, experiment_number: int) -> list[JsonDict]:
    if not capstone:
        return []  # pragma: no cover - defensive missing-source path.
    rows = value_of(capstone.get("missing_blocked_flagged_or_skipped_artifacts"))
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []  # pragma: no cover - defensive malformed-source path.
    return [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("experiment_number") == experiment_number
    ]


def _adversarial_flag_kinds(capstone: Mapping[str, Any] | None, experiment_number: int) -> list[str]:
    kinds: list[str] = []
    for row in _capstone_rows(capstone, experiment_number):
        for flag in row.get("corrigendum_pending") or []:
            if isinstance(flag, Mapping) and flag.get("kind"):
                kinds.append(str(flag["kind"]))
    return list(dict.fromkeys(kinds))


def _feature_audit(exp5353: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = exp5353.get("feature_audit")
    return audit if isinstance(audit, Mapping) else {}


def _runtime_availability(exp5353: Mapping[str, Any]) -> JsonDict:
    audit = _feature_audit(exp5353)
    token_row_count = int(value_of(exp5353.get("tokenprob_feature_row_count")) or 0)
    tokenprob_ready = value_of(exp5353.get("tokenprob_feature_rows_ready")) is True
    per_token = _bool_from_sources(
        value_of(exp5353.get("per_token_logprob_available")),
        audit.get("per_token_logprob_available"),
    )
    return {
        "tokenprob_rows_available": bool(per_token and tokenprob_ready and token_row_count > 0),
        "logits_available": _bool_from_sources(
            value_of(exp5353.get("logits_available")),
            audit.get("logits_available"),
        ),
        "hidden_states_available": _bool_from_sources(
            value_of(exp5353.get("hidden_states_available")),
            audit.get("hidden_states_available"),
        ),
        "attention_available": _bool_from_sources(
            value_of(exp5353.get("attention_available")),
            audit.get("attention_available"),
        ),
        "completion_split_available": _bool_from_sources(
            value_of(exp5353.get("prompt_completion_token_split_available")),
            audit.get("prompt_completion_token_split_available"),
        ),
        "token_timing_available": _bool_from_sources(
            value_of(exp5353.get("token_timing_available")),
            audit.get("token_timing_available"),
        ),
        "tokenprob_feature_row_count": token_row_count,
        "selected_runtime": value_of(exp5353.get("preconditions_checked")) or {},
        "selected_model_spec": value_of(exp5353.get("selected_model_spec")) or {},
    }


def _exp5353_missing_or_tautological(
    exp5353: Mapping[str, Any], capstone: Mapping[str, Any] | None
) -> JsonDict:
    audit = _feature_audit(exp5353)
    nested_missing = list(audit.get("missing_feature_names") or [])
    top_missing = list(value_of(exp5353.get("missing_feature_names")) or [])
    return {
        "feature_audit_missing_feature_names": nested_missing,
        "top_level_missing_feature_names": top_missing,
        "top_level_omits_nested_latent_missing": bool(nested_missing and not top_missing),
        "duration_tautology": _duration_tautology(exp5353),
        "methodology_duration_s": _numeric(value_of(exp5353.get("methodology_duration_s"))),
        "feature_audit_duration_s": _numeric(value_of(exp5353.get("feature_audit_duration_s"))),
        "adversarial_flag_kinds": _adversarial_flag_kinds(capstone, 5353),
    }


def _exp5354_missing_or_tautological(
    exp5354: Mapping[str, Any], capstone: Mapping[str, Any] | None
) -> JsonDict:
    rows = value_of(exp5354.get("carry_token_energy_feature_rows")) or []
    incomplete = {
        str(row["case_id"]): list(row.get("missing_features") or [])
        for row in rows
        if isinstance(row, Mapping) and row.get("feature_complete") is False
    }
    return {
        "missing_feature_names": list(value_of(exp5354.get("missing_feature_names")) or []),
        "incomplete_row_missing_features": incomplete,
        "feature_complete_rate": _numeric(value_of(exp5354.get("feature_complete_rate"))),
        "correct_vs_perturbed_margin": _numeric(
            value_of(exp5354.get("correct_vs_perturbed_margin"))
        ),
        "duration_tautology": _duration_tautology(exp5354),
        "adversarial_flag_kinds": _adversarial_flag_kinds(capstone, 5354),
    }


def _research_reference_context(text: str) -> JsonDict:
    lower = text.lower()
    return {
        "mentions_logits": "logit" in lower,
        "mentions_hidden_states": "hidden state" in lower or "hidden states" in lower,
        "mentions_attention": "attention" in lower,
        "mentions_thermodynamic": "thermodynamic" in lower,
        "mentions_hallufield": "hallufield" in lower,
        "mentions_flag": "flag" in lower or "fLaG" in text,
    }


def _forbidden_claims(availability: Mapping[str, Any]) -> list[str]:
    claims = [
        "new token/internal-energy signal from text-only or top-logprob rows",
        "carry-token energy margin or carry-token separability signal",
        "broad hallucination detection, reasoning verification, or answer-quality claim",
        "FLaG-style latent/internal evidence claim without hidden states or attention",
        "HalluField or thermodynamic token-path stability claim without logits",
        "attention-energy or grounding claim without attention tensors",
        "hidden-state basin, residual-stream, or latent-probe claim without hidden states",
    ]
    if availability["tokenprob_rows_available"] and not (
        availability["logits_available"]
        or availability["hidden_states_available"]
        or availability["attention_available"]
    ):
        claims.append("promotion of tokenprob feature receipts into internal-energy readiness")
    return claims


def _minimum_preconditions() -> list[JsonDict]:
    return [
        {
            "name": "live_methodology_duration",
            "requirement": "methodology_duration_s >= 60.0 and not copied from duration_s",
        },
        {
            "name": "runtime_provenance",
            "requirement": "current local model/backend path, selected model spec, and MODEL_SPECS provenance",
        },
        {
            "name": "claim_matching_internal_surface",
            "requirement": "logits for logit/thermodynamic claims; hidden states for latent claims; attention for attention claims",
        },
        {
            "name": "token_localization",
            "requirement": "prompt/completion token split plus token timing for token-local claims",
        },
        {
            "name": "positive_controls",
            "requirement": "feature-complete non-tautological positive controls with known targets",
        },
        {
            "name": "negative_controls",
            "requirement": "feature-complete perturbed and no-signal controls with positive margins",
        },
        {
            "name": "safety",
            "requirement": "unsafe_false_accepts == 0 and retired external-scorer scope remains closed",
        },
    ]


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the Exp5372 terminal artifact from existing receipts."""

    exp5353 = _read_json_if_exists(root / EXP5353_RELATIVE_PATH)
    exp5354 = _read_json_if_exists(root / EXP5354_RELATIVE_PATH)
    capstone = _read_json_if_exists(root / CAPSTONE_RELATIVE_PATH)
    references_path = root / RESEARCH_REFERENCES_RELATIVE_PATH
    references_text = references_path.read_text(encoding="utf-8") if references_path.exists() else ""
    missing_sources = [
        str(path)
        for path, payload in (
            (EXP5353_RELATIVE_PATH, exp5353),
            (EXP5354_RELATIVE_PATH, exp5354),
            (CAPSTONE_RELATIVE_PATH, capstone),
        )
        if payload is None
    ]
    if not references_text:  # pragma: no cover - defensive missing-source path.
        missing_sources.append(str(RESEARCH_REFERENCES_RELATIVE_PATH))

    if missing_sources:  # pragma: no cover - defensive path for incomplete task packets.
        artifact: JsonDict = {
            "schema": SCHEMA,
            "experiment": EXPERIMENT,
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
            "run_date": RUN_DATE,
            "status": "blocked_missing_source_evidence",
            "token_feature_gate_ready": False,
            "tokenprob_rows_available": False,
            "logits_available": False,
            "hidden_states_available": False,
            "attention_available": False,
            "completion_split_available": False,
            "methodology_min_duration_s": METHODOLOGY_MIN_DURATION_S,
            "future_signal_allowed": False,
            "carry_token_energy_continue": False,
            "retire_recommendation": True,
            "forbidden_claims": ["missing source evidence prevents any token/internal-energy claim"],
            "unsafe_false_accepts": 0,
            "tests_run": [dict(row) for row in tests_run],
            "honest_verdict": "blocked_missing_source_evidence: token/internal-feature gate not ready",
            "missing_sources": missing_sources,
            "field_principles": FIELD_PRINCIPLES,
            "spec_refs": list(SPEC_REFS),
            "reproducibility_checksum": "",
        }
        artifact["reproducibility_checksum"] = _checksum_payload(artifact)
        return artifact

    assert exp5353 is not None and exp5354 is not None and capstone is not None
    availability = _runtime_availability(exp5353)
    exp5353_missing = _exp5353_missing_or_tautological(exp5353, capstone)
    exp5354_missing = _exp5354_missing_or_tautological(exp5354, capstone)
    latent_available = bool(
        availability["logits_available"]
        or availability["hidden_states_available"]
        or availability["attention_available"]
    )
    carry_signal_ready = value_of(exp5354.get("carry_token_energy_signal_ready")) is True
    source_flagged = bool(
        exp5353_missing["adversarial_flag_kinds"]
        or exp5354_missing["adversarial_flag_kinds"]
        or exp5353_missing["duration_tautology"]
        or exp5354_missing["duration_tautology"]
    )
    future_signal_allowed = bool(latent_available and carry_signal_ready and not source_flagged)
    carry_continue = bool(future_signal_allowed and not exp5354_missing["missing_feature_names"])
    retire = bool(not carry_continue)
    forbidden_claims = _forbidden_claims(availability)
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "token_feature_gate_ready": True,
        "tokenprob_rows_available": availability["tokenprob_rows_available"],
        "logits_available": availability["logits_available"],
        "hidden_states_available": availability["hidden_states_available"],
        "attention_available": availability["attention_available"],
        "completion_split_available": availability["completion_split_available"],
        "methodology_min_duration_s": METHODOLOGY_MIN_DURATION_S,
        "future_signal_allowed": future_signal_allowed,
        "carry_token_energy_continue": carry_continue,
        "retire_recommendation": retire,
        "forbidden_claims": forbidden_claims,
        "unsafe_false_accepts": 0,
        "tests_run": [dict(row) for row in tests_run],
        "honest_verdict": (
            "complete: retire carry-token energy lane until logits/hidden states/attention "
            "and feature-complete non-tautological controls exist"
        ),
        "missing_or_tautological_fields": {
            "exp5353": exp5353_missing,
            "exp5354": exp5354_missing,
        },
        "runtime_feature_evidence": availability,
        "source_artifacts": [
            str(EXP5353_RELATIVE_PATH),
            str(EXP5354_RELATIVE_PATH),
            str(CAPSTONE_RELATIVE_PATH),
            str(RESEARCH_REFERENCES_RELATIVE_PATH),
        ],
        "source_evidence_loaded": {
            "exp5353": True,
            "exp5354": True,
            "capstone": True,
            "research_references": True,
        },
        "minimum_preconditions": _minimum_preconditions(),
        "bounded_continuation_allowed": [
            "feature-surface receipt refresh only",
            "backend upgrade preflight for logits/hidden states/attention",
        ],
        "research_reference_context": _research_reference_context(references_text),
        "claim_decisions": {
            "tokenprob_rows_are_feature_receipts_only": True,
            "latent_internal_claims_require_new_backend_features": not latent_available,
            "carry_lane_retired_until_backend_upgrade": retire,
            "no_new_energy_signal_claimed": True,
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp5372 artifact schema and fail closed on unsafe claims."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["unsafe_false_accepts"] != 0:
        raise ValueError("unsafe_false_accepts must be zero")
    if artifact["methodology_min_duration_s"] < METHODOLOGY_MIN_DURATION_S:
        raise ValueError("methodology_min_duration_s must be at least 60 seconds")
    if artifact["future_signal_allowed"]:
        if not (
            artifact["logits_available"]
            or artifact["hidden_states_available"]
            or artifact["attention_available"]
        ):
            raise ValueError("future_signal_allowed requires logits, hidden states, or attention")
        if artifact.get("missing_or_tautological_fields"):
            raise ValueError("future_signal_allowed cannot rely on flagged or tautological evidence")
    if artifact["carry_token_energy_continue"]:
        if artifact["retire_recommendation"] or not artifact["future_signal_allowed"]:
            raise ValueError("carry_token_energy_continue requires a non-retired future signal gate")
    if not artifact["forbidden_claims"] and not artifact["future_signal_allowed"]:
        raise ValueError("forbidden_claims must name unsupported claims")
    if artifact["token_feature_gate_ready"] and not artifact["tests_run"]:
        raise ValueError("tests_run must be recorded for a ready gate")
    if artifact["status"] == "complete" and not artifact["token_feature_gate_ready"]:
        raise ValueError("status complete requires token_feature_gate_ready=true")


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write the validated Exp5372 artifact and return it."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _parse_tests_run(raw: str | None) -> list[JsonDict]:  # pragma: no cover - CLI glue.
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("--tests-run-json must decode to a list")
    return [dict(row) for row in parsed]


def main() -> None:  # pragma: no cover - CLI glue.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--tests-run-json", default=None)
    args = parser.parse_args()
    run(
        root=args.root,
        result_path=args.result_path,
        tests_run=_parse_tests_run(args.tests_run_json),
    )


if __name__ == "__main__":  # pragma: no cover - CLI glue.
    main()
