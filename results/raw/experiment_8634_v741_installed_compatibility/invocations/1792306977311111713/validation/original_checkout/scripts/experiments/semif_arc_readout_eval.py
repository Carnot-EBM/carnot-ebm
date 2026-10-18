"""Pure B2 induction-timing evaluation.

This module reads telemetry rows. It never calls a model, game, or policy.
Spec: REQ-ARC-WMTE-7530.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
GATE_DECISIONS = frozenset(
    {
        "continue_explore",
        "induce_now",
        "reinduce_now",
        "delegate_to_current_gate",
    }
)
MIN_GATE_OPPORTUNITIES = 1_000
MIN_INDUCTION_ATTEMPTS = 100


def load_jsonl(paths: Sequence[Path]) -> list[JsonDict]:
    """Load mapping rows from explicit JSONL paths."""

    rows: list[JsonDict] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def gate_opportunities(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return only admission decisions from the induction seam."""

    return [
        deepcopy(dict(row))
        for row in rows
        if row.get("record_type") == "decision"
        and row.get("seam") == "induction_timing"
        and row.get("gate_decision") in GATE_DECISIONS
    ]


def induction_attempts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return complete or explicitly censored fired-attempt rows."""

    return [
        deepcopy(dict(row))
        for row in rows
        if row.get("record_type") == "induction_attempt" and row.get("attempt_id")
    ]


def oracle_positive_control(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Measure perfect-information token-saving headroom.

    The oracle preserves every later-progress attempt and every accepted planned
    attempt. It suppresses only the remainder. This is deliberately unavailable
    online because it sees final verifier and progress labels.
    """

    kept: list[str] = []
    suppressed: list[str] = []
    saved_tokens = 0
    progress_attempts = 0
    progress_attempts_suppressed = 0
    for row in attempts:
        attempt_id = str(row.get("attempt_id"))
        progress = row.get("progress_within_window") is True
        accepted_plan = row.get("planned") is True and row.get("verifier_result") == "accept"
        useful = progress or accepted_plan
        if progress:
            progress_attempts += 1
        if useful:
            kept.append(attempt_id)
            continue
        suppressed.append(attempt_id)
        tokens = row.get("completion_tokens")
        if isinstance(tokens, int) and tokens > 0:
            saved_tokens += tokens
        if progress:
            progress_attempts_suppressed += 1
    headroom = bool(suppressed and saved_tokens > 0 and progress_attempts_suppressed == 0)
    return {
        "analysis_only": True,
        "oracle_inputs": ["planned", "verifier_result", "progress_within_window"],
        "attempt_count": len(attempts),
        "kept_attempt_count": len(kept),
        "suppressed_attempt_count": len(suppressed),
        "completion_tokens_saved": saved_tokens,
        "progress_attempt_count": progress_attempts,
        "progress_attempts_suppressed": progress_attempts_suppressed,
        "progress_recall": 1.0 if progress_attempts_suppressed == 0 else 0.0,
        "headroom_exists": headroom,
        "kept_attempt_ids": kept,
        "suppressed_attempt_ids": suppressed,
    }


def canonical_hash(value: Mapping[str, Any]) -> str:
    """Hash canonical JSON after blanking the checksum field."""

    copied = deepcopy(dict(value))
    copied["reproducibility_checksum"] = ""
    payload = json.dumps(copied, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_measurement(
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any],
) -> JsonDict:
    """Build the B2 measurement section from immutable telemetry rows."""

    opportunities = gate_opportunities(rows)
    attempts = induction_attempts(rows)
    floor_met = (
        len(opportunities) >= MIN_GATE_OPPORTUNITIES and len(attempts) >= MIN_INDUCTION_ATTEMPTS
    )
    oracle = oracle_positive_control(attempts)
    if not floor_met:
        verdict = "complete_feasibility_only_sample_floor_not_met"
    elif oracle["headroom_exists"]:
        verdict = "complete_b2_oracle_headroom_present_measurement_only"
    else:
        verdict = "complete_b2_kill_no_oracle_headroom"
    artifact = {
        **deepcopy(dict(metadata)),
        "gate_opportunity_count": len(opportunities),
        "induction_attempt_count": len(attempts),
        "sample_floor": {
            "minimum_gate_opportunities": MIN_GATE_OPPORTUNITIES,
            "minimum_induction_attempts": MIN_INDUCTION_ATTEMPTS,
            "met": floor_met,
        },
        "publication_mode": "numeric_measurement" if floor_met else "feasibility_only",
        "positive_control": oracle,
        "positive_control_headroom_exists": oracle["headroom_exists"],
        "numeric_gate_quality_claim": False,
        "gate_ready_to_ship": False,
        "per_attempt_rows": attempts,
        "honest_verdict": verdict,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = canonical_hash(artifact)
    return artifact
