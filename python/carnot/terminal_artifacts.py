"""Shared terminal-state classification for experiment artifacts.

The classifier is deliberately small and data-only. It trusts the artifact file
it is given, not orchestration logs. That makes bootstrap, missing, malformed,
and contradictory artifacts stay visible instead of being laundered by an
external completion receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


TERMINAL_CLASSES = frozenset(
    {"complete", "ready", "positive", "null", "blocked", "skipped", "retired", "flagged"}
)
NONTERMINAL_CLASSES = frozenset(
    {
        "missing",
        "malformed",
        "running",
        "running_bootstrap",
        "bootstrap_only",
        "partial",
        "contradictory",
        "unknown",
    }
)

ACCEPTED_TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "complete",
    "completed",
    "complete_ready",
    "ready",
    "complete_positive",
    "positive",
    "complete_null",
    "null",
    "blocked",
    "skipped",
    "gated",
    "retired",
    "flagged",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REJECTED_NONTERMINAL_PREFIXES = (
    "in_progress",
    "running",
    "running_bootstrap",
    "bootstrap",
    "bootstrap_only",
    "complete_partial",
    "partial",
    "unknown",
)

_RUNNING_CLASSES = frozenset({"running", "running_bootstrap", "bootstrap_only"})
_COMPLETE_ALIASES = frozenset({"complete", "completed", "success", "passed", "shipped"})
_TERMINAL_RECEIPT_STATUSES = frozenset(
    {"OK", "COMPLETE", "COMPLETED", "SUCCESS", "PASSED", "SHIPPED", "FLAGGED", "GATE_BLOCK"}
)


@dataclass(frozen=True)
class NormalizedMarker:
    """Normalized view of one status-like field."""

    raw: str | None
    prefix: str | None
    classification: str


@dataclass(frozen=True)
class TerminalClassification:
    """A fail-closed classification of one artifact path or payload."""

    classification: str
    terminal: bool
    reason: str
    path: str | None = None
    present: bool = True
    loadable: bool = True
    sha256: str | None = None
    status_raw: str | None = None
    status_prefix: str | None = None
    status_class: str = "unknown"
    honest_verdict_raw: str | None = None
    honest_verdict_prefix: str | None = None
    verdict_class: str = "unknown"
    receipt_override_attempted: bool = False
    receipt_overrode: bool = False
    conductor_receipt_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "terminal": self.terminal,
            "reason": self.reason,
            "path": self.path,
            "present": self.present,
            "loadable": self.loadable,
            "sha256": self.sha256,
            "status_raw": self.status_raw,
            "status_prefix": self.status_prefix,
            "status_class": self.status_class,
            "honest_verdict_raw": self.honest_verdict_raw,
            "honest_verdict_prefix": self.honest_verdict_prefix,
            "verdict_class": self.verdict_class,
            "receipt_override_attempted": self.receipt_override_attempted,
            "receipt_overrode": self.receipt_overrode,
            "conductor_receipt_status": self.conductor_receipt_status,
        }


@dataclass(frozen=True)
class GateFieldEligibility:
    """Whether one exact artifact field may feed a downstream gate."""

    field: str
    eligible: bool
    reason: str
    classification: TerminalClassification
    field_present: bool
    field_is_bare: bool
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "eligible": self.eligible,
            "reason": self.reason,
            "classification": self.classification.to_dict(),
            "field_present": self.field_present,
            "field_is_bare": self.field_is_bare,
            "value": self.value,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def path_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _prefix_text(value: Any) -> tuple[str | None, str | None, str]:
    if value is None:
        return None, None, ""
    raw = str(value).strip()
    if not raw:
        return raw, None, ""
    normalized = raw.lower().replace("-", "_")
    normalized = re.sub(r"\s+", " ", normalized)
    prefix = re.split(r"[:\s]", normalized, maxsplit=1)[0].strip("_")
    return raw, prefix or None, normalized


def normalize_marker(value: Any) -> NormalizedMarker:
    """Normalize one status or verdict field into a classifier class."""

    raw, prefix, normalized = _prefix_text(value)
    if not prefix:
        return NormalizedMarker(raw=raw, prefix=prefix, classification="unknown")
    if prefix.startswith("running_bootstrap"):
        return NormalizedMarker(raw=raw, prefix=prefix, classification="running_bootstrap")
    if "bootstrap_only" in normalized or "bootstrap only" in normalized:
        return NormalizedMarker(raw=raw, prefix=prefix, classification="bootstrap_only")
    if prefix.startswith("running") or prefix.startswith("in_progress"):
        return NormalizedMarker(raw=raw, prefix=prefix, classification="running")
    if prefix.startswith("complete_partial") or prefix.startswith("partial"):
        return NormalizedMarker(raw=raw, prefix=prefix, classification="partial")
    if prefix.startswith("complete_ready") or prefix == "ready":
        return NormalizedMarker(raw=raw, prefix=prefix, classification="ready")
    if prefix.startswith("complete_positive") or prefix == "positive":
        return NormalizedMarker(raw=raw, prefix=prefix, classification="positive")
    if prefix.startswith("complete_null") or prefix == "null":
        return NormalizedMarker(raw=raw, prefix=prefix, classification="null")
    if prefix.startswith("blocked"):
        return NormalizedMarker(raw=raw, prefix=prefix, classification="blocked")
    if prefix.startswith(("skipped", "gated", "gate_block")):
        return NormalizedMarker(raw=raw, prefix=prefix, classification="skipped")
    if prefix.startswith("retired"):
        return NormalizedMarker(raw=raw, prefix=prefix, classification="retired")
    if prefix.startswith("flagged"):
        return NormalizedMarker(raw=raw, prefix=prefix, classification="flagged")
    if prefix in _COMPLETE_ALIASES or prefix.startswith(
        ("complete_", "success_", "passed_", "shipped_")
    ):
        return NormalizedMarker(raw=raw, prefix=prefix, classification="complete")
    return NormalizedMarker(raw=raw, prefix=prefix, classification="unknown")


def _nonempty(value: Any) -> bool:
    if value in (None, False, "", [], {}, ()):
        return False
    return True


def _principle_wrapped(value: Any) -> bool:
    return isinstance(value, Mapping) and "value" in value and "principle" in value


def _artifact_flagged(payload: Mapping[str, Any]) -> bool:
    return payload.get("flagged_adversarial") is True or _nonempty(
        payload.get("corrigendum_pending")
    )


def _artifact_gated(payload: Mapping[str, Any], status: NormalizedMarker, verdict: NormalizedMarker) -> bool:
    gates = payload.get("gates_evaluated")
    if payload.get("blocked_at_layer") == "conductor_pre_gate":
        return True
    if isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)) and bool(gates):
        return True
    haystack = " ".join(
        part
        for part in (
            str(status.raw or ""),
            str(verdict.raw or ""),
            str(payload.get("gate_check_summary") or ""),
        )
        if part
    ).lower()
    return "gate_check" in haystack or "gate failed" in haystack


def _receipt_status(conductor_receipt: Mapping[str, Any] | None) -> str | None:
    if not isinstance(conductor_receipt, Mapping):
        return None
    raw = conductor_receipt.get("status")
    if raw is None:
        return None
    return str(raw).strip() or None


def _receipt_claims_terminal(conductor_receipt: Mapping[str, Any] | None) -> bool:
    status = _receipt_status(conductor_receipt)
    return bool(status and status.upper() in _TERMINAL_RECEIPT_STATUSES)


def _running_class(left: str, right: str) -> str | None:
    classes = {left, right}
    if "running_bootstrap" in classes:
        return "running_bootstrap"
    if "bootstrap_only" in classes:
        return "bootstrap_only"
    if "running" in classes:
        return "running"
    return None


def _combine_classes(
    payload: Mapping[str, Any], status: NormalizedMarker, verdict: NormalizedMarker
) -> tuple[str, str]:
    status_class = status.classification
    verdict_class = verdict.classification
    running = _running_class(status_class, verdict_class)
    if running is not None:
        return running, f"nonterminal {running} marker present"
    if _artifact_flagged(payload):
        return "flagged", "artifact carries flagged_adversarial or corrigendum_pending"
    if "partial" in {status_class, verdict_class}:
        return "partial", "partial marker present"
    if "unknown" in {status_class, verdict_class}:
        return "unknown", "status or honest_verdict is absent or unknown"
    if status_class == verdict_class:
        chosen = status_class
    elif status_class == "complete":
        chosen = verdict_class
    elif verdict_class == "complete":
        chosen = status_class
    else:
        return "contradictory", f"status={status_class} conflicts with verdict={verdict_class}"
    if chosen == "blocked" and _artifact_gated(payload, status, verdict):
        return "skipped", "blocked artifact records a gate skip"
    return chosen, f"status={status_class} verdict={verdict_class}"


def classify_artifact_payload(
    payload: Mapping[str, Any] | Any,
    *,
    path: str | Path | None = None,
    sha256: str | None = None,
    conductor_receipt: Mapping[str, Any] | None = None,
) -> TerminalClassification:
    """Classify an already-loaded artifact payload."""

    path_text = Path(path).as_posix() if path is not None else None
    receipt_status = _receipt_status(conductor_receipt)
    if not isinstance(payload, Mapping):
        attempted = _receipt_claims_terminal(conductor_receipt)
        return TerminalClassification(
            classification="malformed",
            terminal=False,
            reason="JSON payload is not an object",
            path=path_text,
            loadable=False,
            sha256=sha256,
            receipt_override_attempted=attempted,
            conductor_receipt_status=receipt_status,
        )

    status = normalize_marker(payload.get("status"))
    verdict = normalize_marker(payload.get("honest_verdict"))
    classification, reason = _combine_classes(payload, status, verdict)
    terminal = classification in TERMINAL_CLASSES
    attempted = (not terminal) and _receipt_claims_terminal(conductor_receipt)
    return TerminalClassification(
        classification=classification,
        terminal=terminal,
        reason=reason,
        path=path_text,
        loadable=True,
        sha256=sha256,
        status_raw=status.raw,
        status_prefix=status.prefix,
        status_class=status.classification,
        honest_verdict_raw=verdict.raw,
        honest_verdict_prefix=verdict.prefix,
        verdict_class=verdict.classification,
        receipt_override_attempted=attempted,
        receipt_overrode=False,
        conductor_receipt_status=receipt_status,
    )


def classify_artifact_path(
    path: str | Path,
    *,
    conductor_receipt: Mapping[str, Any] | None = None,
) -> TerminalClassification:
    """Classify a JSON artifact path, making all load failures nonterminal."""

    artifact_path = Path(path)
    digest = path_sha256(artifact_path)
    receipt_status = _receipt_status(conductor_receipt)
    attempted = _receipt_claims_terminal(conductor_receipt)
    if not artifact_path.exists():
        return TerminalClassification(
            classification="missing",
            terminal=False,
            reason="artifact path is missing",
            path=artifact_path.as_posix(),
            present=False,
            loadable=False,
            sha256=None,
            receipt_override_attempted=attempted,
            receipt_overrode=False,
            conductor_receipt_status=receipt_status,
        )
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return TerminalClassification(
            classification="malformed",
            terminal=False,
            reason=f"artifact JSON could not be loaded: {exc}",
            path=artifact_path.as_posix(),
            present=True,
            loadable=False,
            sha256=digest,
            receipt_override_attempted=attempted,
            receipt_overrode=False,
            conductor_receipt_status=receipt_status,
        )
    got = classify_artifact_payload(
        payload,
        path=artifact_path,
        sha256=digest,
        conductor_receipt=conductor_receipt,
    )
    return TerminalClassification(
        **{**got.to_dict(), "present": True, "sha256": digest}
    )


def gate_field_eligibility(
    payload: Mapping[str, Any] | Any,
    field: str,
    *,
    path: str | Path | None = None,
    sha256: str | None = None,
    conductor_receipt: Mapping[str, Any] | None = None,
) -> GateFieldEligibility:
    """Allow a gate to read only a terminal artifact's exact bare field."""

    classification = classify_artifact_payload(
        payload,
        path=path,
        sha256=sha256,
        conductor_receipt=conductor_receipt,
    )
    if not isinstance(payload, Mapping):
        return GateFieldEligibility(
            field=field,
            eligible=False,
            reason="artifact payload is not an object",
            classification=classification,
            field_present=False,
            field_is_bare=False,
        )

    field_present = field in payload
    value = payload.get(field) if field_present else None
    field_is_bare = field_present and not _principle_wrapped(value)
    if not classification.terminal:
        return GateFieldEligibility(
            field=field,
            eligible=False,
            reason=f"nonterminal artifact classification={classification.classification}",
            classification=classification,
            field_present=field_present,
            field_is_bare=field_is_bare,
            value=value,
        )
    if not field_present:
        return GateFieldEligibility(
            field=field,
            eligible=False,
            reason=f"exact bare field {field!r} is absent",
            classification=classification,
            field_present=False,
            field_is_bare=False,
        )
    if not field_is_bare:
        return GateFieldEligibility(
            field=field,
            eligible=False,
            reason=f"exact field {field!r} exists but is not bare",
            classification=classification,
            field_present=True,
            field_is_bare=False,
            value=value,
        )
    return GateFieldEligibility(
        field=field,
        eligible=True,
        reason=f"terminal artifact exposes exact bare field {field!r}",
        classification=classification,
        field_present=True,
        field_is_bare=True,
        value=value,
    )


def gate_field_eligibility_for_path(
    path: str | Path,
    field: str,
    *,
    conductor_receipt: Mapping[str, Any] | None = None,
) -> GateFieldEligibility:
    """Classify the exact path before exposing any gate field."""

    artifact_path = Path(path)
    classification = classify_artifact_path(artifact_path, conductor_receipt=conductor_receipt)
    payload: Any = {}
    if classification.present and classification.loadable:
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    if not isinstance(payload, Mapping):
        payload = {}

    field_present = field in payload
    value = payload.get(field) if field_present else None
    field_is_bare = field_present and not _principle_wrapped(value)
    if not classification.terminal:
        return GateFieldEligibility(
            field=field,
            eligible=False,
            reason=f"nonterminal artifact classification={classification.classification}",
            classification=classification,
            field_present=field_present,
            field_is_bare=field_is_bare,
            value=value,
        )
    return gate_field_eligibility(
        payload,
        field,
        path=artifact_path,
        sha256=classification.sha256,
        conductor_receipt=conductor_receipt,
    )


def status_verdict_cross_product(
    statuses: Sequence[str],
    verdicts: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status in statuses:
        for verdict in verdicts:
            got = classify_artifact_payload({"status": status, "honest_verdict": verdict})
            rows.append(
                {
                    "status": status,
                    "honest_verdict": verdict,
                    "classification": got.classification,
                    "terminal": got.terminal,
                    "reason": got.reason,
                }
            )
    return rows
