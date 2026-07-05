"""Exp 5248: final GAP-4 receipt salvage or retirement decision.

Spec refs: REQ-REPORT-5248,
SCENARIO-REPORT-5248-SALVAGED-CLEAN-NULL,
SCENARIO-REPORT-5248-BLOCKED-OR-RETIRED.

This module is deliberately a receipt classifier over checked-in artifacts. It
does not rescore rows, regenerate the GAP-4 pool, search for replacements, or
invoke a model. The important distinction is between evidence that was missing
from the original GAP-4 claim and QA metadata that was appended after the source
artifacts were checksummed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5224_gap4_canonical_pool_builder_v478 as exp5224
from carnot import experiment_5225_gap4_clean_scale_validation_gated_v478 as exp5225
from carnot import experiment_5235_adversarial_qa_null_tautology_calibration_v479 as exp5235
from carnot import experiment_5236_gap4_clean_status_after_qa_calibration_v479 as exp5236
from carnot import experiment_5247_slot_artifact_normalizer_v480 as exp5247


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5248_gap4_receipt_salvage_or_retire_v480.json")
EXPERIMENT = "experiment_5248_gap4_receipt_salvage_or_retire_v480"
EXPERIMENT_ID = "exp5248-gap4-receipt-salvage-or-retire-v480"
MILESTONE = "2026.07.480"
RUN_DATE = "2026-07-05"
SCHEMA = "carnot.experiment_5248.gap4_receipt_salvage_or_retire.v480"
INFERENCE_SUBSTRATE = "cached_fixture_replay_no_llm"
EXP5247_VERSION = exp5247.EXPERIMENT
EXPECTED_EXP5247_CHECKSUM = "sha256:07f5a76a71370c90e39eebf8561357bb6d32dca5dc723f2a9dce837f89e229c9"

EXP5247_RELATIVE_PATH = exp5247.RESULT_RELATIVE_PATH
EXP5224_RELATIVE_PATH = Path(exp5224.RESULT_RELATIVE_PATH)
EXP5225_RELATIVE_PATH = Path(exp5225.RESULT_RELATIVE_PATH)
EXP5235_RELATIVE_PATH = exp5235.RESULT_RELATIVE_PATH
EXP5236_RELATIVE_PATH = exp5236.RESULT_RELATIVE_PATH
SOURCE_ARTIFACT_RELATIVE_PATHS = (
    EXP5224_RELATIVE_PATH,
    EXP5225_RELATIVE_PATH,
    EXP5235_RELATIVE_PATH,
    EXP5236_RELATIVE_PATH,
)
SPEC_REFS = (
    "REQ-REPORT-5248",
    "SCENARIO-REPORT-5248-SALVAGED-CLEAN-NULL",
    "SCENARIO-REPORT-5248-BLOCKED-OR-RETIRED",
)
FINAL_DECISIONS = {
    "salvaged_clean_null",
    "blocked_missing_receipts",
    "retire_current_gap4_pool",
}
TERMINAL_PREFIXES = ("complete:", "blocked_")
QA_ANNOTATION_FIELDS = ("flagged_adversarial", "corrigendum_pending")

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Must start with complete: or blocked_ and state the final GAP-4 receipt decision.",
    "inference_substrate": (
        "Must be cached_fixture_replay_no_llm because Exp5248 replays checked-in "
        "artifacts without LLM generation."
    ),
    "gap4_final_decision": (
        "Exactly one of salvaged_clean_null, blocked_missing_receipts, or "
        "retire_current_gap4_pool."
    ),
    "normalized_artifacts": (
        "Lists each source artifact, normalizer receipts, checksum classification, "
        "and claim-criticality."
    ),
    "unsafe_missing_receipts": (
        "Lists claim-critical missing evidence still unsafe after safe normalization."
    ),
    "wins": "Integer copied from frozen Exp5225 validation wins; never rescored by Exp5248.",
    "losses": "Integer copied from frozen Exp5225 validation losses; never rescored by Exp5248.",
    "ties": "Integer copied from frozen Exp5225 validation ties; never rescored by Exp5248.",
    "pool_retired": "True only when the final decision retires the current GAP-4 pool.",
    "no_new_generation": (
        "True because Exp5248 does not regenerate the pool, run a broad search, or "
        "invoke LLM generation."
    ),
}
EXTRA_FIELD_PRINCIPLES = {
    "normalizer_version": "Records the Exp5247 normalizer implementation identity used for this replay.",
    "normalizer_checksum": "Records the checked Exp5247 result checksum that gated this replay.",
}
REQUIRED_SCHEMA_FIELDS = {
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "duration_s",
    "field_principles",
    "normalizer_version",
    "normalizer_checksum",
    "tests_run",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES,
}


@dataclass(frozen=True)
class DecisionResult:
    """Decision data before it is wrapped into the final JSON artifact."""

    final_decision: str
    normalized_artifacts: list[JsonDict]
    unsafe_missing_receipts: list[JsonDict]
    wins: int
    losses: int
    ties: int
    pool_retired: bool
    normalizer_version: str
    normalizer_checksum: str


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Return a checksum over every emitted field except the checksum itself."""

    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return dict(payload)


def load_exp5247(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the Exp 5247 normalizer receipt that gates this decision."""

    return _read_json(Path(root) / EXP5247_RELATIVE_PATH)


def load_source_payloads(root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:
    """Load the four frozen GAP-4 artifacts without mutating them."""

    root_path = Path(root)
    return {
        str(relative): _read_json(root_path / relative)
        for relative in SOURCE_ARTIFACT_RELATIVE_PATHS
    }


def _is_wrapper(value: Any) -> bool:
    return isinstance(value, Mapping) and "value" in value and "principle" in value


def _wrap(field: str, value: Any) -> JsonDict:
    principles = FIELD_PRINCIPLES | EXTRA_FIELD_PRINCIPLES
    return {"value": value, "principle": principles[field]}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _source_checksum(relative: str, payload: Mapping[str, Any]) -> str:
    if relative == str(EXP5224_RELATIVE_PATH):
        return exp5224.payload_checksum(payload)
    if relative == str(EXP5225_RELATIVE_PATH):
        return exp5225.payload_checksum(payload)
    if relative == str(EXP5235_RELATIVE_PATH):
        return exp5235.reproducibility_checksum(payload)
    if relative == str(EXP5236_RELATIVE_PATH):
        return exp5236.reproducibility_checksum(payload)
    raise ValueError(f"unknown GAP-4 source artifact: {relative}")


def _pre_qa_payload(payload: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in payload.items() if key not in QA_ANNOTATION_FIELDS}


def _checksum_receipt(relative: str, payload: Mapping[str, Any]) -> JsonDict:
    stored = payload.get("reproducibility_checksum")
    full_checksum = _source_checksum(relative, payload)
    pre_qa_checksum = _source_checksum(relative, _pre_qa_payload(payload))
    if full_checksum == stored:
        classification = "safe-normalized"
        reason = "stored checksum already matches the full checked-in payload"
        pre_qa_matches = True
    elif pre_qa_checksum == stored:
        classification = "safe-normalized"
        reason = (
            "stored checksum matches after removing only post-hoc QA annotations: "
            "flagged_adversarial and corrigendum_pending"
        )
        pre_qa_matches = True
    else:
        classification = "unsafe-missing"
        reason = "stored checksum does not match full or pre-QA payload"
        pre_qa_matches = False
    return {
        "stored_checksum": stored,
        "full_payload_checksum": full_checksum,
        "pre_qa_checksum": pre_qa_checksum,
        "pre_qa_checksum_matches_stored": pre_qa_matches,
        "classification": classification,
        "claim_critical": True,
        "reason": reason,
    }


def _field_classification(
    *,
    relative: str,
    rejection: Mapping[str, Any],
) -> JsonDict:
    kind = str(rejection.get("kind") or "")
    field = str(rejection.get("field") or "unknown")
    if kind in {"source_flagged_adversarial", "source_corrigendum_pending"}:
        return {
            "field": field,
            "kind": kind,
            "classification": "irrelevant-to-claim",
            "claim_critical": False,
            "reason": "post-hoc adversarial-QA annotation is preserved but not missing claim evidence",
        }
    if relative == str(EXP5235_RELATIVE_PATH) and kind in {
        "duration_too_short",
        "missing_methodology_receipt",
    }:
        return {
            "field": field,
            "kind": kind,
            "classification": "irrelevant-to-claim",
            "claim_critical": False,
            "reason": "Exp5235 uses compute-bound fixture flags as QA calibration sentinels",
        }
    return {
        "field": field,
        "kind": kind,
        "classification": "unsafe-missing",
        "claim_critical": True,
        "reason": str(rejection.get("detail") or "normalizer rejected claim-critical receipt"),
    }


def _normalizer_artifact_row(relative: str, payload: Mapping[str, Any]) -> JsonDict:
    result = exp5247.normalize_artifact(
        payload,
        required_principle_fields=("honest_verdict", "inference_substrate"),
    )
    checksum = _checksum_receipt(relative, payload)
    field_classifications = [
        {
            "field": "reproducibility_checksum",
            "kind": "checksum_receipt",
            "classification": checksum["classification"],
            "claim_critical": checksum["claim_critical"],
            "reason": checksum["reason"],
        },
        *[
            _field_classification(relative=relative, rejection=rejection)
            for rejection in result.unsafe_rejections
        ],
    ]
    if relative == str(EXP5236_RELATIVE_PATH) and "reproducibility_checksum" in str(
        payload.get("remaining_blocker") or ""
    ):
        field_classifications.append(
            {
                "field": "remaining_blocker",
                "kind": "prior_schema_checksum_blocker",
                "classification": "safe-normalized",
                "claim_critical": True,
                "reason": "Exp5236 blocker named only upstream checksum drift resolved by pre-QA checksum replay",
            }
        )
    return {
        "path": relative,
        "experiment": payload.get("experiment"),
        "experiment_id": payload.get("experiment_id"),
        "normalizer_ready_for_gates": result.ready_for_gated_consumers,
        "normalizer_safe_repairs": [dict(row) for row in result.safe_repairs],
        "normalizer_unsafe_rejections": [dict(row) for row in result.unsafe_rejections],
        "checksum_receipt": checksum,
        "field_classifications": field_classifications,
    }


def _claim_critical_missing(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    missing: list[JsonDict] = []
    for row in rows:
        artifact = str(row.get("path"))
        for item in row.get("field_classifications", []):
            if not isinstance(item, Mapping):
                continue
            if item.get("classification") == "unsafe-missing" and item.get("claim_critical") is True:
                missing.append(
                    {
                        "artifact": artifact,
                        "field": str(item.get("field")),
                        "reason": str(item.get("kind") or item.get("reason")),
                    }
                )
    return missing


def _counts_and_missing(payloads: Mapping[str, Mapping[str, Any]]) -> tuple[int, int, int, list[JsonDict]]:
    validation = payloads[str(EXP5225_RELATIVE_PATH)]
    status = payloads[str(EXP5236_RELATIVE_PATH)]
    missing: list[JsonDict] = []
    values: dict[str, int] = {}
    for field in ("wins", "losses", "ties"):
        value = validation.get(field)
        if not _is_int(value):
            missing.append(
                {
                    "artifact": str(EXP5225_RELATIVE_PATH),
                    "field": field,
                    "reason": "frozen_count_missing_or_not_int",
                }
            )
            values[field] = 0
            continue
        values[field] = int(value)
        status_value = status.get(field)
        if _is_int(status_value) and int(status_value) != int(value):
            missing.append(
                {
                    "artifact": str(EXP5236_RELATIVE_PATH),
                    "field": field,
                    "reason": "frozen_count_contradiction",
                }
            )
    return values["wins"], values["losses"], values["ties"], missing


def _crosses_min_six(wins: int, losses: int) -> bool:
    return wins >= 6 and losses == 0


def _dedupe_missing(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out: list[JsonDict] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row.get("artifact")), str(row.get("field")), str(row.get("reason")))
        if key in seen:
            continue
        seen.add(key)
        out.append({"artifact": key[0], "field": key[1], "reason": key[2]})
    return out


def _normalizer_precondition_missing(normalizer_artifact: Mapping[str, Any]) -> JsonDict | None:
    if normalizer_artifact.get("artifact_normalizer_ready") is True:
        return None
    return {
        "artifact": str(EXP5247_RELATIVE_PATH),
        "field": "artifact_normalizer_ready",
        "reason": "exp5247_not_ready",
    }


def build_decision_from_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    normalizer_artifact: Mapping[str, Any],
    retire_on_blocked: bool = False,
    forced_missing_receipts: Sequence[Mapping[str, Any]] = (),
) -> DecisionResult:
    """Classify GAP-4 receipts and decide whether the frozen null is usable."""

    normalizer_version = str(normalizer_artifact.get("experiment") or EXP5247_VERSION)
    normalizer_checksum = str(normalizer_artifact.get("reproducibility_checksum") or "")
    wins, losses, ties, count_missing = _counts_and_missing(payloads)
    precondition_missing = _normalizer_precondition_missing(normalizer_artifact)
    if precondition_missing is not None:
        missing = _dedupe_missing([precondition_missing])
        return DecisionResult(
            final_decision="blocked_missing_receipts",
            normalized_artifacts=[],
            unsafe_missing_receipts=missing,
            wins=wins,
            losses=losses,
            ties=ties,
            pool_retired=False,
            normalizer_version=normalizer_version,
            normalizer_checksum=normalizer_checksum,
        )

    normalized_artifacts = [
        _normalizer_artifact_row(str(relative), payloads[str(relative)])
        for relative in SOURCE_ARTIFACT_RELATIVE_PATHS
    ]
    missing = [
        *_claim_critical_missing(normalized_artifacts),
        *count_missing,
        *[dict(row) for row in forced_missing_receipts],
    ]
    if _crosses_min_six(wins, losses):
        missing.append(
            {
                "artifact": str(EXP5225_RELATIVE_PATH),
                "field": "wins/losses",
                "reason": "frozen_counts_cross_min6_not_clean_null",
            }
        )
    unsafe_missing = _dedupe_missing(missing)
    if unsafe_missing and retire_on_blocked:
        final_decision = "retire_current_gap4_pool"
    elif unsafe_missing:
        final_decision = "blocked_missing_receipts"
    else:
        final_decision = "salvaged_clean_null"
    return DecisionResult(
        final_decision=final_decision,
        normalized_artifacts=normalized_artifacts,
        unsafe_missing_receipts=unsafe_missing,
        wins=wins,
        losses=losses,
        ties=ties,
        pool_retired=final_decision == "retire_current_gap4_pool",
        normalizer_version=normalizer_version,
        normalizer_checksum=normalizer_checksum,
    )


def build_decision(root: Path | str = REPO_ROOT) -> DecisionResult:
    """Build the read-only Exp 5248 decision from repository artifacts."""

    return build_decision_from_payloads(
        load_source_payloads(root),
        normalizer_artifact=load_exp5247(root),
    )


def _honest_verdict(decision: DecisionResult) -> str:
    if decision.final_decision == "salvaged_clean_null":
        return (
            "complete: GAP-4 final decision salvaged_clean_null; frozen validation "
            f"preserves wins={decision.wins}, losses={decision.losses}, ties={decision.ties}, "
            "and all claim-critical receipts are present after safe normalization."
        )
    if decision.final_decision == "retire_current_gap4_pool":
        return (
            "blocked_retire_current_gap4_pool: GAP-4 final decision "
            "retire_current_gap4_pool because claim-critical receipts remain missing."
        )
    return (
        "blocked_missing_receipts: GAP-4 final decision blocked_missing_receipts "
        "because claim-critical receipts remain missing after safe normalization."
    )


def build_artifact(
    *,
    decision: DecisionResult,
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float = 0.0,
) -> JsonDict:
    """Wrap the decision into the required principle-annotated result schema."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "field_principles": FIELD_PRINCIPLES | EXTRA_FIELD_PRINCIPLES,
        "normalizer_version": _wrap("normalizer_version", decision.normalizer_version),
        "normalizer_checksum": _wrap("normalizer_checksum", decision.normalizer_checksum),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(decision)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "gap4_final_decision": _wrap("gap4_final_decision", decision.final_decision),
        "normalized_artifacts": _wrap("normalized_artifacts", decision.normalized_artifacts),
        "unsafe_missing_receipts": _wrap(
            "unsafe_missing_receipts", decision.unsafe_missing_receipts
        ),
        "wins": _wrap("wins", decision.wins),
        "losses": _wrap("losses", decision.losses),
        "ties": _wrap("ties", decision.ties),
        "pool_retired": _wrap("pool_retired", decision.pool_retired),
        "no_new_generation": _wrap("no_new_generation", True),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _required_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    if not _is_wrapper(value):
        raise ValueError(f"{field} must be principle-wrapped")
    principles = FIELD_PRINCIPLES | EXTRA_FIELD_PRINCIPLES
    if value.get("principle") != principles[field]:
        raise ValueError(f"{field} principle mismatch")
    return value.get("value")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal Exp 5248 artifact before it is trusted."""

    missing = REQUIRED_SCHEMA_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES | EXTRA_FIELD_PRINCIPLES:
        raise ValueError("field_principles")
    verdict = _required_value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict terminal prefix")
    if _required_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    decision = _required_value(artifact, "gap4_final_decision")
    if decision not in FINAL_DECISIONS:
        raise ValueError("gap4_final_decision")
    normalized = _required_value(artifact, "normalized_artifacts")
    if not isinstance(normalized, list):
        raise ValueError("normalized_artifacts")
    unsafe = _required_value(artifact, "unsafe_missing_receipts")
    if not isinstance(unsafe, list):
        raise ValueError("unsafe_missing_receipts")
    if decision != "salvaged_clean_null" and not unsafe:
        raise ValueError("unsafe_missing_receipts")
    for field in ("wins", "losses", "ties"):
        value = _required_value(artifact, field)
        if not _is_int(value):
            raise ValueError(f"{field} must be an integer")
    pool_retired = _required_value(artifact, "pool_retired")
    if pool_retired is not (decision == "retire_current_gap4_pool"):
        raise ValueError("pool_retired")
    if _required_value(artifact, "no_new_generation") is not True:
        raise ValueError("no_new_generation")
    if _required_value(artifact, "normalizer_version") != EXP5247_VERSION:
        raise ValueError("normalizer_version")
    normalizer_checksum = _required_value(artifact, "normalizer_checksum")
    if not isinstance(normalizer_checksum, str) or not normalizer_checksum.startswith("sha256:"):
        raise ValueError("normalizer_checksum")
    tests = artifact.get("tests_run")
    if not isinstance(tests, list):
        raise ValueError("tests_run")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def write_artifact(
    *,
    output_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float = 0.0,
) -> JsonDict:
    """Write the Exp 5248 JSON artifact and return its parsed content."""

    artifact = build_artifact(
        decision=build_decision(root),
        tests_run=tests_run,
        duration_s=duration_s,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--test-run", action="append", default=[])
    args = parser.parse_args(argv)
    started = time.monotonic()
    tests_run = [
        {"command": item.split("=", 1)[0], "outcome": item.split("=", 1)[1]}
        if "=" in item
        else {"command": item, "outcome": "RECORDED"}
        for item in args.test_run
    ]
    artifact = write_artifact(
        output_path=args.output,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    print(json.dumps({"result_path": str(args.output), "checksum": artifact["reproducibility_checksum"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
