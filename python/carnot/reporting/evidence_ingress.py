"""Classify artifact evidence before an aggregation reads its measurements.

The helper keeps rejected payloads out of the accepted list. This boundary
prevents a consumer from reading a metric first and checking quarantine later.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any


JsonMapping = Mapping[str, Any]
Verifier = Callable[[Path], Mapping[str, Any]]


def _sha256(path: Path) -> str:
    """Hash the bytes that the consumer is about to classify."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_date(value: object) -> str | None:
    """Return one compact calendar date, or None when the claim is invalid."""

    if not isinstance(value, str):
        return None
    for shape in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, shape).strftime("%Y%m%d")
        except ValueError:
            pass
    return None


def _critical_verifier_flag(report: Mapping[str, Any]) -> bool:
    """Treat only critical verifier findings as adversarial quarantine signals."""

    if report.get("flagged_adversarial") is True:
        return True
    flags = report.get("flags")
    return isinstance(flags, list) and any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in flags
    )


def _clean_verifier_result(report: Mapping[str, Any]) -> bool:
    """Require evidence that the verifier loaded the artifact before calling it clean."""

    if report.get("flagged_adversarial") is False:
        return True
    if report.get("loaded") is not True:
        return False
    flags = report.get("flags")
    if isinstance(flags, list):
        return not _critical_verifier_flag(report)
    return report.get("flag_count") == 0


def _expectation(value: object) -> tuple[Path, str | None, dict[str, Any]]:
    """Normalize one caller-owned expectation without changing the caller's object."""

    if isinstance(value, (str, os.PathLike)):
        return Path(value), None, {}
    if not isinstance(value, Mapping) or "path" not in value:
        raise TypeError("each evidence expectation must be a path or mapping with path")
    path = Path(value["path"])
    expected_hash = value.get("expected_sha256")
    if expected_hash is not None and not isinstance(expected_hash, str):
        raise TypeError("expected_sha256 must be a string when supplied")
    metadata = {
        str(key): item for key, item in value.items() if key not in {"path", "expected_sha256"}
    }
    return path, expected_hash, metadata


def _duplicate_row(
    path: Path,
    expected_hash: str | None,
    metadata: Mapping[str, Any],
    occurrences: int,
) -> dict[str, Any]:
    """Record one duplicate path once so a consumer cannot count either copy."""

    observed_hash = None
    run_date = None
    normalized = None
    honest_verdict = None
    verdict_class = None
    try:
        observed_hash = _sha256(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            run_date = payload.get("run_date")
            normalized = _normalized_date(run_date)
            honest_verdict = payload.get("honest_verdict")
            verdict_class = payload.get("verdict_class")
    except (OSError, ValueError):
        pass
    return {
        "path": str(path),
        "sha256": observed_hash,
        "expected_sha256": expected_hash,
        "run_date": run_date,
        "normalized_run_date": normalized,
        "flag_status": "unknown",
        "flag_sources": [],
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "ingestion_eligible": False,
        "reason_codes": ["duplicate_expected_artifact"],
        "expected_occurrences": occurrences,
        **metadata,
    }


def ingest_evidence(
    expected_artifacts: Sequence[object],
    *,
    verifier: Verifier | None = None,
    require_run_date: bool = True,
) -> dict[str, Any]:
    """Return only inputs whose identity, date, and quarantine state are usable.

    A consumer gets payload bytes only through ``accepted_upstreams``. Rejected
    rows retain verdict text for audit, but they do not retain the payload that
    contains measurements. This shape makes late or accidental aggregation hard.
    """

    expectations = [_expectation(value) for value in expected_artifacts]
    canonical_paths = [str(path.resolve(strict=False)) for path, _, _ in expectations]
    path_counts = Counter(canonical_paths)
    handled_duplicates: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for (path, expected_hash, metadata), canonical in zip(
        expectations, canonical_paths, strict=True
    ):
        occurrences = path_counts[canonical]
        if occurrences > 1:
            if canonical not in handled_duplicates:
                rejected.append(_duplicate_row(path, expected_hash, metadata, occurrences))
                handled_duplicates.add(canonical)
            continue

        base: dict[str, Any] = {
            "path": str(path),
            "sha256": None,
            "expected_sha256": expected_hash,
            "run_date": None,
            "normalized_run_date": None,
            "flag_status": "unknown",
            "flag_sources": [],
            "honest_verdict": None,
            "verdict_class": None,
            "ingestion_eligible": False,
            "reason_codes": [],
            **metadata,
        }
        if not path.exists():
            base["reason_codes"] = ["artifact_not_found"]
            rejected.append(base)
            continue
        try:
            base["sha256"] = _sha256(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            base["reason_codes"] = ["artifact_unreadable"]
            base["read_error"] = f"{type(exc).__name__}: {exc}"
            rejected.append(base)
            continue
        if not isinstance(payload, Mapping):
            base["reason_codes"] = ["artifact_not_object"]
            rejected.append(base)
            continue

        base["run_date"] = payload.get("run_date")
        base["normalized_run_date"] = _normalized_date(base["run_date"])
        base["honest_verdict"] = payload.get("honest_verdict")
        base["verdict_class"] = payload.get("verdict_class")
        artifact_flag = payload.get("flagged_adversarial")
        flag_sources: list[str] = []
        if artifact_flag is True:
            flag_sources.append("artifact")

        verifier_report: Mapping[str, Any] | None = None
        verifier_error: str | None = None
        if verifier is not None:
            try:
                candidate = verifier(path)
                if isinstance(candidate, Mapping):
                    verifier_report = candidate
            except Exception as exc:  # noqa: BLE001 - verifier failure is evidence state.
                verifier_error = f"{type(exc).__name__}: {exc}"
        if verifier_report is not None and _critical_verifier_flag(verifier_report):
            flag_sources.append("verifier")

        if flag_sources:
            base["flag_status"] = "flagged"
        elif artifact_flag is False or (
            verifier_report is not None and _clean_verifier_result(verifier_report)
        ):
            base["flag_status"] = "clean"
        if verifier_error is not None:
            base["verifier_error"] = verifier_error
        base["flag_sources"] = flag_sources

        reasons: list[str] = []
        if "artifact" in flag_sources:
            reasons.append("artifact_flagged_adversarial")
        if "verifier" in flag_sources:
            reasons.append("verifier_flagged_adversarial")
        if base["flag_status"] == "unknown":
            reasons.append("flag_status_unknown")
        if require_run_date:
            if base["run_date"] is None:
                reasons.append("run_date_missing")
            elif base["normalized_run_date"] is None:
                reasons.append("run_date_malformed")
        if expected_hash is not None and expected_hash != base["sha256"]:
            reasons.append("hash_drift")

        base["reason_codes"] = reasons
        if reasons:
            rejected.append(base)
        else:
            base["ingestion_eligible"] = True
            base["payload"] = dict(payload)
            accepted.append(base)

    excluded = [
        row
        for row in rejected
        if any(reason.endswith("flagged_adversarial") for reason in row["reason_codes"])
    ]
    return {
        "classification_complete": True,
        "accepted_upstreams": accepted,
        "rejected_upstream_rows": rejected,
        "excluded_flagged_upstreams": excluded,
        "missing_run_date_rows": [
            row for row in rejected if "run_date_missing" in row["reason_codes"]
        ],
        "malformed_run_date_rows": [
            row for row in rejected if "run_date_malformed" in row["reason_codes"]
        ],
    }


__all__ = ["ingest_evidence"]
