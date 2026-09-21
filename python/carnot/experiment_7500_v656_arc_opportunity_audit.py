"""Audit ARC intervention opportunity without changing the live policy.

This host-only reducer reads the two frozen cost panels. It counts only an arm
whose logged eligibility is explicitly true and whose timed selection finishes.
Unknown eligibility therefore remains missing evidence instead of becoming
permission to relabel model generation as removable work.

Spec refs: REQ-ARC-WMTE-7500 and SCENARIO-ARC-WMTE-7500-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7478_v655_arc_interval_protocol as interval_protocol
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.656"
EXPERIMENT_ID = "exp7500-arc-opportunity-audit"
SCHEMA = "carnot.exp7500.v656.arc_opportunity_audit.v1"

MODEL_SPECS: list[str] = []
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"

PANEL_A_PATH = Path("results/experiment_7485_v655_arc_cost_panel_a.json")
PANEL_B_PATH = Path("results/experiment_7499_v656_arc_panel_b.json")
RESULT_PATH = Path("results/experiment_7500_v656_arc_opportunity_audit.json")
RAW_DIR = Path("results/raw/experiment_7500_v656_arc_opportunity_audit")
CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7500_v656_arc_opportunity_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7500_v656_arc_opportunity_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7500_v656_arc_opportunity_audit.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")

PANEL_A_GAMES = ("sk48", "tr87", "s5i5", "lp85", "lf52", "cn04")
PANEL_B_GAMES = ("tu93", "g50t", "tn36", "vc33", "re86", "dc22")
EPISODE_SEEDS = (65_501, 65_502, 65_503)
IDENTITY_NAMES = ("observer", "policy", "model", "protocol")

ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
REQUIRED_TERMINAL_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7478_v655_arc_interval_protocol.py"),
    Path("python/carnot/agentic/arc_decision_telemetry.py"),
    Path("ops/arc_solve_registry.yaml"),
    Path("docs/research-notes/semif-ebm-arc-experiment-plan-2026-09-20.md"),
    SPEC_PATH,
    PANEL_A_PATH,
    PANEL_B_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def utc_now() -> str:
    """Return a measured UTC boundary for the current host aggregation."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush a truthful boundary so a long reduction never looks abandoned."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7500] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so one changed row breaks independent replay."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum that stores the result."""

    copied = deepcopy(dict(value))
    copied["reproducibility_checksum"] = ""
    return canonical_hash(copied)


def load_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed for absent or malformed bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    op: str,
    upstream: str,
    field: str,
    principle: str,
) -> JsonDict:
    """Record an exact comparison and why a failure must stay visible."""

    if op == "==":
        passed = observed == expected
    elif op == ">=":
        passed = isinstance(observed, (int, float)) and observed >= expected
    elif op == ">":
        passed = isinstance(observed, (int, float)) and observed > expected
    elif op == "in":
        passed = observed in expected
    else:
        raise ValueError(f"unsupported_gate_operator:{op}")
    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": bool(passed),
        "upstream": upstream,
        "path": upstream,
        "field": field,
        "artifact_field": field,
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failure and keep required checks separate from benefit nulls."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    required = [row for row in failures if row.get("category") in {"validity", "readiness"}]
    first = required[0] if required else failures[0] if failures else None
    return {
        "all_passed": not failures,
        "required_validity_and_readiness_passed": not required,
        "failed_count": len(failures),
        "required_failed_count": len(required),
        "failed_checks": failures,
        "first_failed_check": first.get("check") if first else None,
        "upstream": first.get("upstream") if first else None,
        "exact_field_path": first.get("field") if first else None,
        "expected_value": first.get("expected") if first else None,
        "observed_value": first.get("observed") if first else None,
    }


def _tick(row: Mapping[str, Any], field: str) -> int | None:
    """Return one real integer clock tick without accepting booleans."""

    value = row.get(field)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    fallback = row.get("event_monotonic_ns")
    return fallback if isinstance(fallback, int) and not isinstance(fallback, bool) else None


def reduce_supervisor_episode(episode_id: str, events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count selected arms and only complete, explicitly eligible service.

    Candidate presence is not eligibility. A selected arm contributes service
    only when its own eligibility is ``True`` and compatible start/end evidence
    supplies a reproducible runtime observation.
    """

    episode_events = [
        dict(row)
        for row in events
        if str(row.get("episode_id")) == episode_id
        and row.get("seam") == "supervisor_arm_selection"
    ]
    by_decision: dict[str, list[JsonDict]] = defaultdict(list)
    for row in episode_events:
        by_decision[str(row.get("decision_id") or "")].append(row)

    exposure: Counter[str] = Counter()
    true_count = 0
    false_count = 0
    unknown_count = 0
    abstentions = 0
    incomplete = 0
    mismatched = 0
    eligible_intervals: dict[tuple[str, str, str, str], list[tuple[int, int]]] = defaultdict(list)
    intervention_ledger: list[JsonDict] = []

    for decision_id, rows in sorted(by_decision.items()):
        selection = next((row for row in rows if row.get("event") == "selection"), None)
        if selection is None:
            continue
        selected = [str(value) for value in selection.get("selected_candidate_ids") or []]
        chosen = selected[0] if selected else "abstain"
        exposure[chosen] += 1
        abstained = not selected or chosen == "no_redirect"
        abstentions += int(abstained)

        eligibility: bool | None = None
        for row in rows:
            if row.get("event") != "eligible_candidate_set":
                continue
            for candidate in row.get("candidates") or []:
                if not isinstance(candidate, Mapping):
                    continue
                if str(candidate.get("stable_candidate_id")) == chosen:
                    value = candidate.get("eligibility")
                    eligibility = value if isinstance(value, bool) else None
        true_count += int(eligibility is True)
        false_count += int(eligibility is False)
        unknown_count += int(eligibility is None)

        starts = [row for row in rows if row.get("event") == "stage_start"]
        ends = [row for row in rows if row.get("event") in {"stage_end", "stage_terminal"}]
        compatible: list[tuple[int, int, tuple[str, str, str, str]]] = []
        for start in starts:
            for end in ends:
                identities = (
                    str(start.get("run_id") or "unavailable"),
                    str(start.get("process_id") or "unavailable"),
                    episode_id,
                    str(start.get("clock_identity") or "unavailable"),
                )
                end_identities = (
                    str(end.get("run_id") or "unavailable"),
                    str(end.get("process_id") or "unavailable"),
                    episode_id,
                    str(end.get("clock_identity") or "unavailable"),
                )
                if identities != end_identities:
                    continue
                left = _tick(start, "interval_start_monotonic_ns")
                right = _tick(end, "interval_end_monotonic_ns")
                if left is not None and right is not None and right >= left:
                    compatible.append((left, right, identities))
        if starts and ends and not compatible:
            mismatched += 1
        if starts and not compatible:
            incomplete += 1
        if eligibility is not True or abstained or not compatible:
            continue
        left, right, identities = min(
            compatible, key=lambda item: (item[1] - item[0], item[0], item[1])
        )
        eligible_intervals[identities].append((left, right))
        intervention_ledger.append(
            {
                "episode_id": episode_id,
                "decision_id": decision_id,
                "selected_arm": chosen,
                "eligibility": True,
                "runtime_observation_reproduced": True,
                "start_ns": left,
                "end_ns": right,
                "duration_ns": right - left,
                "clock_identity": identities[3],
                "process_id": identities[1],
            }
        )

    eligible_ns = sum(
        interval_protocol.union_duration_ns(intervals) for intervals in eligible_intervals.values()
    )
    return {
        "episode_id": episode_id,
        "selection_exposure_count": sum(exposure.values()),
        "selected_arms": dict(sorted(exposure.items())),
        "explicit_true_selected_count": true_count,
        "explicit_false_selected_count": false_count,
        "unknown_eligibility_selected_count": unknown_count,
        "abstention_count": abstentions,
        "triggered_opportunity_count": len(intervention_ledger),
        "eligible_service_lower_ns": eligible_ns,
        "eligible_service_upper_ns": eligible_ns,
        "incomplete_selected_interval_count": incomplete,
        "mismatched_selected_clock_count": mismatched,
        "intervention_ledger": intervention_ledger,
        "efficacy_estimate": None,
    }


def reduce_episode(
    source_row: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    episode_start_ns: int,
    episode_end_ns: int,
) -> JsonDict:
    """Recompute one compact episode from qualified raw interval events."""

    episode_id = str(source_row.get("episode_id"))
    cost = interval_protocol.reduce_episode_intervals(
        events,
        episode_id=episode_id,
        episode_start_ns=episode_start_ns,
        episode_end_ns=episode_end_ns,
    )
    supervisor = reduce_supervisor_episode(episode_id, events)
    observed_ns = int(cost["observed_episode_ns"])
    eligible_upper = min(observed_ns, int(supervisor["eligible_service_upper_ns"]))
    eligible_lower = min(eligible_upper, int(supervisor["eligible_service_lower_ns"]))
    return {
        "episode_id": episode_id,
        "game": source_row.get("game"),
        "seed": source_row.get("seed"),
        "disposition": source_row.get("disposition"),
        "valid": source_row.get("disposition") == "complete" and cost["bounds_valid"] is True,
        "censored": source_row.get("disposition") != "complete",
        "progress_censored": source_row.get("actions_to_progress_censored") is True,
        "missing": False,
        "observed_episode_ns": observed_ns,
        "replaceable_lower_ns": int(cost["replaceable_lower_ns"]),
        "replaceable_upper_ns": int(cost["replaceable_upper_ns"]),
        "eligible_service_lower_ns": eligible_lower,
        "eligible_service_upper_ns": eligible_upper,
        "eligible_service_fraction_lower": eligible_lower / observed_ns if observed_ns else 0.0,
        "eligible_service_fraction_upper": eligible_upper / observed_ns if observed_ns else 0.0,
        "complete_interval_count": int(cost["complete_interval_count"]),
        "incomplete_interval_count": int(cost["incomplete_interval_count"]),
        "duplicate_event_count": int(cost["duplicate_event_count"]),
        "mismatched_clock_count": int(cost["mismatched_clock_count"]),
        "missing_run_identity_group_count": int(cost["missing_run_identity_group_count"]),
        "missing_process_identity_group_count": int(cost["missing_process_identity_group_count"]),
        "bounds_valid": bool(cost["bounds_valid"]),
        "stage_union_ns": int(cost["stage_union_ns"]),
        "unattributed_ns": int(cost["unattributed_ns"]),
        "supervisor": supervisor,
        "offline_reproduced": source_row.get("offline_reproduced") is True,
        "solve_provenance": source_row.get("solve_provenance") or "unavailable",
    }


def _source_flags(artifact: Mapping[str, Any], score_field: str) -> JsonDict:
    """Preserve the producer's original disposition without reclassifying it."""

    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "verdict_class": artifact.get("verdict_class"),
        "flagged_adversarial": artifact.get("flagged_adversarial"),
        score_field: artifact.get(score_field),
    }


def _source_record(
    path: Path,
    *,
    root: Path,
    role: str,
    flags: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Hash one source when present and make external absence explicit."""

    try:
        label = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(path.resolve())
    row: JsonDict = {"path": label, "role": role, "exists": path.is_file()}
    if path.is_file():
        row.update({"bytes": path.stat().st_size, "sha256": current_work_receipt.sha256_file(path)})
    else:
        row.update({"bytes": None, "sha256": None})
    if flags is not None:
        row["original_flags"] = deepcopy(dict(flags))
    return row


def _hash_for_suffix(artifact: Mapping[str, Any], suffix: str) -> str | None:
    """Find a producer-recorded source hash by stable repository suffix."""

    for name, row in (artifact.get("source_artifact_hashes") or {}).items():
        if str(name).endswith(suffix) and isinstance(row, Mapping):
            value = row.get("sha256")
            return str(value) if isinstance(value, str) else None
    return None


def _panel_identities(artifact: Mapping[str, Any]) -> JsonDict:
    """Extract the four identities that authorize cross-panel pooling."""

    specs = [row for row in artifact.get("model_specs") or [] if isinstance(row, Mapping)]
    model_hash = (
        specs[0].get("sha256") if len(specs) == 1 else canonical_hash(specs) if specs else None
    )
    return {
        "observer": _hash_for_suffix(artifact, "python/carnot/agentic/arc_decision_telemetry.py"),
        "policy": _hash_for_suffix(artifact, "python/carnot/agentic/arc_competition_agent.py"),
        "model": str(model_hash) if model_hash is not None else None,
        "protocol": _hash_for_suffix(
            artifact, "python/carnot/experiment_7478_v655_arc_interval_protocol.py"
        ),
    }


def _panel_validity(
    artifact: Mapping[str, Any], *, panel: str, path_exists: bool
) -> tuple[str, list[str]]:
    """Classify one upstream before any episode is allowed into the audit."""

    if not path_exists:
        return "blocked_missing", ["source_artifact_missing"]
    if not artifact:
        return "blocked_invalid", ["source_artifact_malformed"]
    expected_id = "exp7485-v655-arc-cost-panel-a" if panel == "A" else "exp7499-arc-panel-b"
    score_field = f"arc_panel_{panel.lower()}_complete_score"
    games = PANEL_A_GAMES if panel == "A" else PANEL_B_GAMES
    rows = [row for row in artifact.get("rows") or [] if isinstance(row, Mapping)]
    errors: list[str] = []
    if artifact.get("experiment_id") != expected_id:
        errors.append("experiment_id_mismatch")
    if artifact.get(score_field) != 1:
        errors.append(f"{score_field}_not_one")
    if artifact.get("flagged_adversarial") is not False:
        errors.append("flagged_adversarial")
    if artifact.get("verdict_class") not in {"null", "positive", "circular_positive"}:
        errors.append("upstream_verdict_not_usable")
    if len(rows) != 18:
        errors.append("episode_count_not_18")
    observed_pairs = {(row.get("game"), row.get("seed")) for row in rows}
    expected_pairs = {(game, seed) for game in games for seed in EPISODE_SEEDS}
    if observed_pairs != expected_pairs:
        errors.append("frozen_schedule_mismatch")
    return ("blocked_invalid", errors) if errors else ("valid", [])


def _episode_bounds(row: Mapping[str, Any]) -> tuple[int, int] | None:
    """Read producer boundaries without inventing a synthetic clock span."""

    cost = row.get("exclusive_cost") if isinstance(row.get("exclusive_cost"), Mapping) else {}
    start = row.get("episode_start_ns", cost.get("episode_start_ns"))
    end = row.get("episode_end_ns", cost.get("episode_end_ns"))
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and end >= start
    ):
        return start, end
    return None


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read a complete shard and reject a malformed or non-object row."""

    rows: list[JsonDict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed_jsonl:{path}:{number}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"non_object_jsonl:{path}:{number}")
        rows.append(dict(value))
    return rows


def _reduce_panel(
    root: Path,
    *,
    panel: str,
    artifact_path: Path,
    artifact: Mapping[str, Any],
) -> JsonDict:
    """Recompute all available episodes and bind every raw shard byte."""

    state, errors = _panel_validity(artifact, panel=panel, path_exists=artifact_path.is_file())
    result: JsonDict = {
        "panel": panel,
        "state": state,
        "errors": list(errors),
        "rows": [],
        "identities": _panel_identities(artifact)
        if artifact
        else {name: None for name in IDENTITY_NAMES},
        "source_flags": _source_flags(artifact, f"arc_panel_{panel.lower()}_complete_score")
        if artifact
        else {},
    }
    if state != "valid":
        return result

    shard_by_episode = {
        str(row.get("episode_id")): dict(row)
        for row in artifact.get("seam_event_shards") or []
        if isinstance(row, Mapping) and row.get("episode_id") not in {None, "all"}
    }
    source_rows = [dict(row) for row in artifact.get("rows") or [] if isinstance(row, Mapping)]
    began = time.monotonic()
    reduced_rows: list[JsonDict] = []
    for index, source in enumerate(source_rows):
        episode_id = str(source.get("episode_id"))
        shard = shard_by_episode.get(episode_id)
        if shard is None:
            result["errors"].append(f"shard_missing:{episode_id}")
            continue
        raw_path = Path(str(shard.get("path") or ""))
        raw_path = raw_path if raw_path.is_absolute() else root / raw_path
        if not raw_path.is_file():
            result["errors"].append(f"shard_missing:{episode_id}:{raw_path}")
            continue
        observed_hash = current_work_receipt.sha256_file(raw_path)
        if observed_hash != shard.get("sha256"):
            result["errors"].append(f"shard_hash_mismatch:{episode_id}")
            continue
        bounds = _episode_bounds(source)
        if bounds is None:
            result["errors"].append(f"episode_bounds_missing:{episode_id}")
            continue
        try:
            events = _read_jsonl(raw_path)
        except ValueError as exc:
            result["errors"].append(str(exc))
            continue
        row = reduce_episode(
            source,
            events,
            episode_start_ns=bounds[0],
            episode_end_ns=bounds[1],
        )
        row.update(
            {
                "panel": panel,
                "shard": {
                    "path": str(shard.get("path")),
                    "sha256": observed_hash,
                    "bytes": raw_path.stat().st_size,
                    "row_count": len(events),
                },
            }
        )
        reduced_rows.append(row)
        print(
            f"[exp7500] phase=reduce_panel_{panel.lower()} event=unit_complete "
            f"elapsed_s={time.monotonic() - began:.3f} completed_units={index + 1}/{len(source_rows)}",
            flush=True,
        )
    result["rows"] = reduced_rows
    if result["errors"] or len(reduced_rows) != 18 or not all(row["valid"] for row in reduced_rows):
        result["state"] = "blocked_invalid"
    return result


def evaluate_pooling(
    rows: Sequence[Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    *,
    panel_states: Mapping[str, str],
) -> JsonDict:
    """Require matching versions plus 30 episodes across 10 games."""

    valid = [row for row in rows if row.get("valid") is True]
    games = sorted({str(row.get("game")) for row in valid if row.get("game")})
    a_identity = identities.get("A") or {}
    b_identity = identities.get("B") or {}
    mismatches = [name for name in IDENTITY_NAMES if a_identity.get(name) != b_identity.get(name)]
    missing_identities = [
        name for name in IDENTITY_NAMES if not a_identity.get(name) or not b_identity.get(name)
    ]
    identities_compatible = not mismatches and not missing_identities
    panels_complete = panel_states.get("A") == panel_states.get("B") == "valid"
    episode_support = len(valid) >= 30
    game_support = len(games) >= 10
    allowed = panels_complete and identities_compatible and episode_support and game_support
    return {
        "pooling_allowed": allowed,
        "panel_states": dict(panel_states),
        "identities_compatible": identities_compatible,
        "mismatched_identities": mismatches,
        "missing_identities": missing_identities,
        "valid_episode_count": len(valid),
        "valid_game_count": len(games),
        "valid_games": games,
        "episode_support_floor": 30,
        "game_support_floor": 10,
        "episode_support_passed": episode_support,
        "game_support_passed": game_support,
        "planned_episode_count": 36,
        "planned_game_count": 12,
        "stratification": "pooled_cross_game" if allowed else "separate_panels",
        "principle": "Version drift, an absent panel, or low cross-game support cannot become one homogeneous claim.",
    }


def _per_game(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Aggregate compact seed evidence without losing its panel identity."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("panel")), str(row.get("game")))].append(row)
    output: list[JsonDict] = []
    for (panel, game), members in sorted(groups.items()):
        output.append(
            {
                "panel": panel,
                "game": game,
                "version_identity": None,
                "episode_count": len(members),
                "valid_episode_count": sum(row.get("valid") is True for row in members),
                "censored_episode_count": sum(row.get("censored") is True for row in members),
                "progress_censored_episode_count": sum(
                    row.get("progress_censored") is True for row in members
                ),
                "observed_episode_ns": sum(
                    int(row.get("observed_episode_ns") or 0) for row in members
                ),
                "eligible_service_lower_ns": sum(
                    int(row.get("eligible_service_lower_ns") or 0) for row in members
                ),
                "eligible_service_upper_ns": sum(
                    int(row.get("eligible_service_upper_ns") or 0) for row in members
                ),
                "selection_exposure_count": sum(
                    int((row.get("supervisor") or {}).get("selection_exposure_count") or 0)
                    for row in members
                ),
                "triggered_opportunity_count": sum(
                    int((row.get("supervisor") or {}).get("triggered_opportunity_count") or 0)
                    for row in members
                ),
                "abstention_count": sum(
                    int((row.get("supervisor") or {}).get("abstention_count") or 0)
                    for row in members
                ),
                "solve_provenance": sorted({str(row.get("solve_provenance")) for row in members}),
                "seed_rows": [
                    {
                        "episode_id": row.get("episode_id"),
                        "seed": row.get("seed"),
                        "valid": row.get("valid"),
                        "censored": row.get("censored"),
                        "eligible_service_lower_ns": row.get("eligible_service_lower_ns"),
                        "eligible_service_upper_ns": row.get("eligible_service_upper_ns"),
                        "shard": deepcopy(row.get("shard")),
                    }
                    for row in members
                ],
            }
        )
    return output


def _supervisor_disposition(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Combine actual selections without treating observation as efficacy."""

    selected: Counter[str] = Counter()
    interventions: list[JsonDict] = []
    for row in rows:
        supervisor = row.get("supervisor") or {}
        selected.update(supervisor.get("selected_arms") or {})
        for entry in supervisor.get("intervention_ledger") or []:
            interventions.append({"panel": row.get("panel"), **deepcopy(dict(entry))})
    return {
        "selected_arms": dict(sorted(selected.items())),
        "selection_exposure_count": sum(selected.values()),
        "abstention_count": sum(
            int((row.get("supervisor") or {}).get("abstention_count") or 0) for row in rows
        ),
        "explicit_true_selected_count": sum(
            int((row.get("supervisor") or {}).get("explicit_true_selected_count") or 0)
            for row in rows
        ),
        "explicit_false_selected_count": sum(
            int((row.get("supervisor") or {}).get("explicit_false_selected_count") or 0)
            for row in rows
        ),
        "unknown_eligibility_selected_count": sum(
            int((row.get("supervisor") or {}).get("unknown_eligibility_selected_count") or 0)
            for row in rows
        ),
        "triggered_opportunity_count": len(interventions),
        "intervention_ledger": interventions,
        "efficacy_estimate": None,
        "observational_benefit_inference_allowed": False,
        "principle": "Exposure and abstention stay visible, but observational selection cannot establish arm benefit.",
    }


def _service_accounting(rows: Sequence[Mapping[str, Any]]) -> tuple[JsonDict, JsonDict]:
    """Report eligible fractions and the perfect-removal Amdahl ceiling."""

    observed = sum(int(row.get("observed_episode_ns") or 0) for row in rows)
    lower = sum(int(row.get("eligible_service_lower_ns") or 0) for row in rows)
    upper = sum(int(row.get("eligible_service_upper_ns") or 0) for row in rows)
    lower_fraction = lower / observed if observed else 0.0
    upper_fraction = upper / observed if observed else 0.0
    upper_fraction = min(1.0, max(0.0, upper_fraction))
    speedup = None if upper_fraction >= 1.0 else 1.0 / (1.0 - upper_fraction)
    fractions = {
        "observed_episode_ns": observed,
        "eligible_service_lower_ns": lower,
        "eligible_service_upper_ns": upper,
        "eligible_service_fraction_lower": lower_fraction,
        "eligible_service_fraction_upper": upper_fraction,
    }
    amdahl = {
        "eligible_fraction_upper": upper_fraction,
        "speedup_upper": speedup,
        "assumption": "perfect_removal_of_authenticated_eligible_service_at_zero_overhead",
        "omitted_stages": [
            "text_generation",
            "world_model_construction",
            "verifier_work",
            "dispatch",
            "idle",
            "unattributed_time",
            "observer_overhead",
            "replacement_readout_overhead",
        ],
        "hardware_acceleration_claim": False,
        "principle": "A conditional ceiling names every omitted stage and cannot pose as an expected speedup.",
    }
    return fractions, amdahl


def reduce_upstreams(root: Path = REPO_ROOT) -> JsonDict:
    """Read both exact roadmap deliverables and preserve an unavailable B."""

    a_path = root / PANEL_A_PATH
    b_path = root / PANEL_B_PATH
    panel_a = load_object(a_path)
    panel_b = load_object(b_path)
    a_result = _reduce_panel(root, panel="A", artifact_path=a_path, artifact=panel_a)
    b_result = _reduce_panel(root, panel="B", artifact_path=b_path, artifact=panel_b)
    rows = [*a_result["rows"], *b_result["rows"]]
    identities = {"A": a_result["identities"], "B": b_result["identities"]}
    states = {"A": str(a_result["state"]), "B": str(b_result["state"])}
    pooling = evaluate_pooling(rows, identities, panel_states=states)
    disposition = _supervisor_disposition(rows)
    fractions, amdahl = _service_accounting(rows)
    complete = sum(row.get("valid") is True for row in rows)
    censored = sum(row.get("censored") is True for row in rows)
    failed = sum(row.get("disposition") in {"failed", "complete_error"} for row in rows)
    missing_units = sum(18 for result in (a_result, b_result) if result["state"] != "valid")
    source_hashes = {
        PANEL_A_PATH.as_posix(): _source_record(
            a_path,
            root=root,
            role="panel_a_upstream",
            flags=_source_flags(panel_a, "arc_panel_a_complete_score") if panel_a else {},
        ),
        PANEL_B_PATH.as_posix(): _source_record(
            b_path,
            root=root,
            role="panel_b_exact_roadmap_deliverable",
            flags=_source_flags(panel_b, "arc_panel_b_complete_score") if panel_b else {},
        ),
    }
    return {
        "rows": rows,
        "panel_results": {"A": a_result, "B": b_result},
        "panel_identities": identities,
        "pooling": pooling,
        "per_game_results": _per_game(rows),
        "supervisor_disposition": disposition,
        "intervention_ledger": disposition["intervention_ledger"],
        "efficacy_estimate": None,
        "eligible_service_fractions": fractions,
        "amdahl_upper_bound": amdahl,
        "sample_size_budget": {
            "planned_independent_units": 36,
            "attempted_independent_units": len(rows),
            "complete_independent_units": complete,
            "failed_independent_units": failed,
            "excluded_independent_units": 0,
            "censored_independent_units": censored,
            "unstarted_independent_units": missing_units,
            "independent_game_clusters": len({row.get("game") for row in rows}),
        },
        "source_artifact_hashes": source_hashes,
        "errors": [*a_result["errors"], *b_result["errors"]],
    }


FIELD_PRINCIPLES = {
    "schema": "Versioned schema, exact experiment_id, milestone and terminal status prevent reader drift.",
    "experiment_id": "The roadmap identifier prevents evidence from being attached to a different experiment.",
    "milestone": "The milestone binds this reduction to the V656 task ordering.",
    "status": "A terminal status distinguishes finished accounting from an abandoned checkpoint.",
    "run_date": "Use 20260921 and retain measured clock and process identity.",
    "started_at_utc": "A measured UTC start exposes stale or copied execution receipts.",
    "ended_at_utc": "A measured UTC end exposes unfinished aggregation.",
    "preconditions_checked": "Exact paths, observed values, ownership and input validity prevent guessed prerequisites.",
    "MODEL_SPECS": "An empty current model list prevents historical Qwen evidence from becoming a current call.",
    "model_specs": "The lowercase empty mirror prevents schema readers from inventing a current model.",
    "model_invoked": "False distinguishes zero current loads and generations from cited historical work.",
    "invocation_counts": "Balanced zero current calls prevent historical invocation inflation.",
    "inference_substrate": "The exact aggregation substrate prevents cached evidence from posing as native inference.",
    "inference_substrate_class": "The aggregation class prevents a synthetic model duration floor.",
    "execution_venue": "Host execution stays distinct from archived CUDA or board evidence.",
    "duration_s": "Measured work without padding prevents fabricated runtime credibility.",
    "duration_breakdown_s": "Separate aggregation and validation time prevents hidden synthetic inference time.",
    "phase_spans": "Flushed real boundaries expose unfinished operations and stalls.",
    "clock_identity": "Monotonic process timing prevents unrelated clocks from being unioned.",
    "process_id": "The owned host process prevents another run's timing from entering this receipt.",
    "random_seed": "Frozen role, audit, ordering and interval seeds explain this deterministic null reducer.",
    "reproducibility_checksum": "Code, source bytes, raw shards and validation scope remain bound together.",
    "source_artifact_hashes": "Original upstream bytes, verdicts and flags cannot be laundered.",
    "rows": "Per-episode intervals, exposure, failures and censoring permit independent reduction.",
    "panel_results": "A missing or invalid panel stays one blocked branch instead of fake zero episodes.",
    "panel_identities": "Observer, policy, model and protocol versions decide whether pooling is legal.",
    "pooling": "Cross-game support cannot open under version drift, missing panels or low support.",
    "sample_size_budget": "Planned, attempted, complete, failed, excluded, censored and unstarted units stay separate.",
    "acceptance_gate_results": "Every validity, support and benefit comparison names the failure it prevents.",
    "gate_check_summary": "Blocked claims name the exact upstream path, expected value and observation.",
    "honest_verdict": "A complete terminal finding preserves actual blocked conductor evidence.",
    "verdict_class": "The closed enum prevents external absence from becoming retryable owned work.",
    "verifier_is_oracle": "An oracle-defined analytic benefit could not become a positive claim.",
    "flagged_adversarial": "Actual reader flags remain visible and cannot be cleared to open a gate.",
    "validation_receipts": "Exact commands, scopes, exits and log hashes make checks reviewable.",
    "validation_manifest": "A frozen affected-file list prevents validation from expanding after results are known.",
    "field_principles": "Every emitted field states its purpose and prevented failure mode.",
    "arc_opportunity_audit_complete_score": "Auditable row reduction includes absent branches without requiring a benefit.",
    "opportunity_present_score": "Only real eligible work with compatible support can set this score.",
    "per_game_results": "Panel, version, intervals, exposure and censoring remain recheckable by game.",
    "solve_provenance": "Repeated upstream levels retain provenance and receive no new solve credit.",
    "supervisor_disposition": "Empty exposure stays empty and cannot justify a policy default.",
    "intervention_ledger": "Only authenticated triggered opportunities appear; zero triggers produce an empty ledger.",
    "efficacy_estimate": "Observational selection cannot become a causal arm-benefit estimate.",
    "eligible_service_fractions": "Only explicitly eligible completed service enters the removable fraction.",
    "amdahl_upper_bound": "A perfect-removal ceiling names omitted stages and is not an expected speedup.",
    "claim_dispositions": "Reporting does not authorize selectors, E4/E5, submissions, registry changes or hardware claims.",
}


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one successful, non-timeout receipt for every command."""

    counts = Counter(str(row.get("name")) for row in receipts)
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        counts[name] == 1
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def _input_hashes(root: Path, reduction: Mapping[str, Any]) -> JsonDict:
    """Hash listed current sources while retaining the two panel records."""

    hashes = deepcopy(dict(reduction["source_artifact_hashes"]))
    if root.resolve() != REPO_ROOT.resolve():
        return hashes
    for relative in INPUT_PATHS:
        if relative in {PANEL_A_PATH, PANEL_B_PATH}:
            continue
        path = root / relative
        hashes[relative.as_posix()] = _source_record(
            path, root=root, role="current_input" if path.is_file() else "declared_output"
        )
    return hashes


def _preconditions(root: Path, reduction: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate source dispositions and the exact task requirement."""

    panel_results = reduction["panel_results"]
    checks = [
        gate(
            "panel_a_valid",
            "validity",
            "valid",
            panel_results["A"]["state"],
            op="==",
            upstream=PANEL_A_PATH.as_posix(),
            field="panel_results.A.state",
            principle="The audit cannot cite Panel A if its original bytes or flags are invalid.",
        ),
        gate(
            "panel_b_explicit_disposition",
            "validity",
            ["valid", "blocked_missing", "blocked_invalid"],
            panel_results["B"]["state"],
            op="in",
            upstream=PANEL_B_PATH.as_posix(),
            field="panel_results.B.state",
            principle="External absence must stay explicit instead of becoming eighteen zero episodes.",
        ),
    ]
    spec = root / SPEC_PATH
    if spec.is_file():
        checks.append(
            gate(
                "driving_requirement_present",
                "validity",
                True,
                "REQ-ARC-WMTE-7500" in spec.read_text(encoding="utf-8"),
                op="==",
                upstream=SPEC_PATH.as_posix(),
                field="REQ-ARC-WMTE-7500",
                principle="Implementation without its requirement would bypass spec-first review.",
            )
        )
    return checks


def _acceptance_gates(
    reduction: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    require_terminal: bool,
) -> list[JsonDict]:
    """Keep validity, readiness and observational benefit independent."""

    affected = _receipts_pass(receipts, AFFECTED_CHECK_NAMES)
    terminal = not require_terminal or _receipts_pass(receipts, REQUIRED_TERMINAL_NAMES)
    rows = reduction["rows"]
    pooling = reduction["pooling"]
    eligible_ns = reduction["eligible_service_fractions"]["eligible_service_upper_ns"]
    audit_complete = (
        reduction["panel_results"]["A"]["state"] == "valid"
        and reduction["panel_results"]["B"]["state"]
        in {"valid", "blocked_missing", "blocked_invalid"}
        and all(row.get("bounds_valid") is True for row in rows)
        and affected
        and terminal
    )
    return [
        gate(
            "affected_validation",
            "validity",
            True,
            affected,
            op="==",
            upstream="validation_receipts",
            field="required_scoped_checks",
            principle="A favorable metric cannot excuse invalid affected code or imports.",
        ),
        gate(
            "terminal_independent_readers",
            "validity",
            True,
            terminal,
            op="==",
            upstream="validation_receipts",
            field="terminal_readers",
            principle="A self-reported reduction cannot replace cold replay and unchanged readers.",
        ),
        gate(
            "exclusive_bounds_valid",
            "validity",
            True,
            all(row.get("bounds_valid") is True for row in rows),
            op="==",
            upstream="rows",
            field="rows[].bounds_valid",
            principle="Duplicate, nested or incompatible intervals cannot inflate eligible cost.",
        ),
        gate(
            "opportunity_audit_complete",
            "readiness",
            True,
            audit_complete,
            op="==",
            upstream="rows|panel_results|validation_receipts",
            field="arc_opportunity_audit_complete_score",
            principle="A valid scientific null must not block independent accounting completeness.",
        ),
        gate(
            "pooled_episode_support",
            "scientific_benefit",
            30,
            pooling["valid_episode_count"],
            op=">=",
            upstream="pooling",
            field="pooling.valid_episode_count",
            principle="A favorable seed or low-support panel cannot replace cross-game evidence.",
        ),
        gate(
            "pooled_game_support",
            "scientific_benefit",
            10,
            pooling["valid_game_count"],
            op=">=",
            upstream="pooling",
            field="pooling.valid_game_count",
            principle="Repeated episodes cannot substitute for independent games.",
        ),
        gate(
            "identity_compatible_pooling",
            "scientific_benefit",
            True,
            pooling["identities_compatible"],
            op="==",
            upstream="panel_identities",
            field="pooling.identities_compatible",
            principle="Version drift cannot become one pooled cost claim.",
        ),
        gate(
            "eligible_service_present",
            "scientific_benefit",
            0,
            eligible_ns,
            op=">",
            upstream="rows[].supervisor",
            field="eligible_service_fractions.eligible_service_upper_ns",
            principle="Hypothesized removable generation cannot replace actual eligible supervisor work.",
        ),
        gate(
            "effect_size_threshold",
            "scientific_benefit",
            True,
            False,
            op="==",
            upstream="observation_only_no_intervention",
            field="efficacy_estimate",
            principle="Observational arm selection cannot establish a causal effect.",
        ),
        gate(
            "retention_threshold",
            "scientific_benefit",
            True,
            False,
            op="==",
            upstream="observation_only_no_intervention",
            field="retention",
            principle="A cost opportunity cannot excuse lost ARC progress.",
        ),
        gate(
            "multiplicity_threshold",
            "scientific_benefit",
            "not_applicable_no_intervention_family",
            "not_applicable_no_intervention_family",
            op="==",
            upstream="observation_scope",
            field="multiplicity",
            principle="One selected observation cannot bypass a declared hypothesis-family correction.",
        ),
    ]


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    run_date: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    require_terminal: bool,
    started_at_utc: str | None = None,
    ended_at_utc: str | None = None,
) -> JsonDict:
    """Build one compact terminal record from the two exact deliverables."""

    reduction = reduce_upstreams(root)
    per_game = deepcopy(reduction["per_game_results"])
    for row in per_game:
        row["version_identity"] = deepcopy(reduction["panel_identities"].get(str(row.get("panel"))))
    preconditions = _preconditions(root, reduction)
    gates = _acceptance_gates(reduction, validation_receipts, require_terminal=require_terminal)
    summary = _gate_summary(gates)
    readiness = next(row for row in gates if row["check"] == "opportunity_audit_complete")
    audit_complete = int(readiness["passed"] is True)
    pooling = reduction["pooling"]
    eligible_present = reduction["eligible_service_fractions"]["eligible_service_upper_ns"] > 0
    opportunity_present = int(
        eligible_present and pooling["pooling_allowed"] is True and audit_complete == 1
    )
    required_passed = summary["required_validity_and_readiness_passed"]
    panel_b_state = reduction["panel_results"]["B"]["state"]
    if not required_passed:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_arc_opportunity_audit_validation"
    elif panel_b_state != "valid":
        verdict_class = "blocked"
        honest_verdict = f"complete_blocked_panel_b_{panel_b_state}_opportunity_audit"
    elif opportunity_present:
        verdict_class = "null"
        honest_verdict = "complete_null_observational_opportunity_present_no_efficacy"
    else:
        verdict_class = "null"
        honest_verdict = "complete_null_no_supported_arc_intervention_opportunity"

    receipts = [deepcopy(dict(row)) for row in validation_receipts]
    validation_s = sum(float(row.get("duration_s") or 0.0) for row in receipts)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest_verdict,
        "run_date": run_date,
        "started_at_utc": started_at_utc or utc_now(),
        "ended_at_utc": ended_at_utc or utc_now(),
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": float(duration_s),
        "duration_breakdown_s": {
            "aggregation_and_replay": max(0.0, float(duration_s) - validation_s),
            "validation": validation_s,
            "inference": 0.0,
            "optimization": 0.0,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "clock_identity": "time.monotonic",
        "process_id": os.getpid(),
        "random_seed": {
            "role_seed": 75_000,
            "optimizer_seed": None,
            "audit_seed": 75_000,
            "order_seed": 65_500,
            "interval_seed": None,
            "deterministic_null_seed_note": "No sampling, fitting or intervention occurs.",
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": _input_hashes(root, reduction),
        "rows": deepcopy(reduction["rows"]),
        "panel_results": deepcopy(reduction["panel_results"]),
        "panel_identities": deepcopy(reduction["panel_identities"]),
        "pooling": deepcopy(pooling),
        "sample_size_budget": deepcopy(reduction["sample_size_budget"]),
        "acceptance_gate_results": gates,
        "gate_check_summary": summary,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "validation_manifest": {
            "experiment_id": MANIFEST.experiment_id,
            "test_paths": list(MANIFEST.test_paths),
            "changed_modules": list(MANIFEST.changed_modules),
            "static_paths": list(MANIFEST.static_paths),
            "frozen_before_checks": True,
        },
        "field_principles": {},
        "arc_opportunity_audit_complete_score": audit_complete,
        "opportunity_present_score": opportunity_present,
        "per_game_results": per_game,
        "solve_provenance": "upstream_preserved_no_new_solve",
        "supervisor_disposition": deepcopy(reduction["supervisor_disposition"]),
        "intervention_ledger": deepcopy(reduction["intervention_ledger"]),
        "efficacy_estimate": None,
        "eligible_service_fractions": deepcopy(reduction["eligible_service_fractions"]),
        "amdahl_upper_bound": deepcopy(reduction["amdahl_upper_bound"]),
        "claim_dispositions": {
            "selector_default": False,
            "e4_gate_opened": False,
            "e5_gate_opened": False,
            "submission_authorized": False,
            "registry_increment": 0,
            "hardware_acceleration_claim": False,
            "game_specific_rules_generated": False,
        },
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key,
            f"Retaining {key} prevents an omitted audit field from changing the finding silently.",
        )
        for key in artifact
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_receipts() -> list[JsonDict]:
    """Supply typed passing receipts only for deterministic unit fixtures."""

    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "required": True,
            "scope": "deterministic_fixture",
        }
        for name in (*AFFECTED_CHECK_NAMES, *REQUIRED_TERMINAL_NAMES)
    ]


def build_artifact_for_test(root: Path) -> JsonDict:
    """Build a terminal-shaped fixture through the production reducers."""

    return build_artifact(
        root,
        run_date=RUN_DATE,
        duration_s=1.0,
        phase_spans=[{"phase": "fixture", "duration_s": 1.0, "completed_units": 18}],
        validation_receipts=_fixture_receipts(),
        require_terminal=True,
        started_at_utc="2026-09-21T00:00:00Z",
        ended_at_utc="2026-09-21T00:00:01Z",
    )


def _reduction_view(reduction: Mapping[str, Any]) -> JsonDict:
    """Select the raw-derived fields that an independent replay must match."""

    per_game = deepcopy(list(reduction["per_game_results"]))
    for row in per_game:
        row["version_identity"] = deepcopy(reduction["panel_identities"].get(str(row.get("panel"))))
    return {
        "rows": deepcopy(reduction["rows"]),
        "panel_results": deepcopy(reduction["panel_results"]),
        "panel_identities": deepcopy(reduction["panel_identities"]),
        "pooling": deepcopy(reduction["pooling"]),
        "sample_size_budget": deepcopy(reduction["sample_size_budget"]),
        "per_game_results": per_game,
        "supervisor_disposition": deepcopy(reduction["supervisor_disposition"]),
        "intervention_ledger": deepcopy(reduction["intervention_ledger"]),
        "efficacy_estimate": reduction["efficacy_estimate"],
        "eligible_service_fractions": deepcopy(reduction["eligible_service_fractions"]),
        "amdahl_upper_bound": deepcopy(reduction["amdahl_upper_bound"]),
    }


def independent_reduce(artifact: Mapping[str, Any], root: Path = REPO_ROOT) -> JsonDict:
    """Re-read both producers and all shard bytes in a fresh reduction."""

    reduction = reduce_upstreams(root)
    recomputed = _reduction_view(reduction)
    declared = {key: artifact.get(key) for key in recomputed}
    return {
        **recomputed,
        "matches_declared": canonical_hash(declared) == canonical_hash(recomputed),
        "errors": list(reduction["errors"]),
    }


REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "honest_verdict",
    "verdict_class",
    "verifier_is_oracle",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "arc_opportunity_audit_complete_score",
    "opportunity_present_score",
    "per_game_results",
    "solve_provenance",
    "supervisor_disposition",
)


def validate_artifact(
    value: Mapping[str, Any] | Path,
    *,
    root: Path = REPO_ROOT,
    require_terminal: bool,
) -> list[str]:
    """Cold-check identity, replay, zero calls, principles and receipts."""

    artifact = load_object(value) if isinstance(value, Path) else dict(value)
    errors = [
        f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    ]
    for field, expected in (
        ("schema", SCHEMA),
        ("experiment_id", EXPERIMENT_ID),
        ("milestone", MILESTONE),
        ("run_date", RUN_DATE),
        ("MODEL_SPECS", []),
        ("model_specs", []),
        ("model_invoked", False),
        ("inference_substrate", INFERENCE_SUBSTRATE),
        ("inference_substrate_class", INFERENCE_SUBSTRATE_CLASS),
        ("execution_venue", EXECUTION_VENUE),
    ):
        if artifact.get(field) != expected:
            errors.append(f"identity_mismatch:{field}")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_not_zero")
    replay = independent_reduce(artifact, root)
    if replay["matches_declared"] is not True:
        errors.append("independent_reduction_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("invalid_verdict_class")
    if artifact.get("verifier_is_oracle") is True and artifact.get("verdict_class") == "positive":
        errors.append("oracle_positive_forbidden")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal_complete")
    principles = artifact.get("field_principles") or {}
    if any(not principles.get(key) for key in artifact):
        errors.append("field_principle_missing")
    if any(not row.get("principle") for row in artifact.get("acceptance_gate_results") or []):
        errors.append("gate_principle_missing")
    if any(row.get("passed") is not True for row in artifact.get("preconditions_checked") or []):
        errors.append("precondition_failed")
    receipts = artifact.get("validation_receipts") or []
    if not _receipts_pass(receipts, AFFECTED_CHECK_NAMES):
        errors.append("affected_validation_missing_or_failed")
    if require_terminal and not _receipts_pass(receipts, REQUIRED_TERMINAL_NAMES):
        errors.append("terminal_validation_missing_or_failed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if len(json.dumps(artifact, sort_keys=True, default=str).encode()) >= 20 * 1024 * 1024:
        errors.append("artifact_exceeds_20_mib")
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze only affected reporting checks; no numbered runtime E2E applies."""

    private_root.mkdir(parents=True, exist_ok=True)
    return validation_contract.build_command_plan(root, MANIFEST, private_root / "affected")


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command expansion, broad pytest targets and runtime E2E drift."""

    errors = validation_contract.validate_command_plan(root, MANIFEST, commands)
    counts = Counter(row.name for row in commands)
    for name in AFFECTED_CHECK_NAMES:
        if counts[name] != 1:
            errors.append(f"command_count:{name}:{counts[name]}")
    for row in commands:
        if row.name.startswith("e2e_"):
            errors.append(f"runtime_e2e_forbidden:{row.name}")
        if any(argument.rstrip("/") in {"tests", "tests/python"} for argument in row.argv):
            errors.append(f"broad_test_target:{row.name}")
    return list(dict.fromkeys(errors))


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build cold replay and the two unchanged strict artifact readers."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", wrapper, "--replay", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", wrapper, "--replay", str(candidate), "--reduce-only"),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
    ]


def _phase_span(
    name: str, began: float, run_started: float, completed_units: int
) -> JsonDict:  # pragma: no cover - measured orchestration receipt.
    """Record one real monotonic phase interval from the process start."""

    ended = time.monotonic()
    return {
        "phase": name,
        "started_offset_s": began - run_started,
        "ended_offset_s": ended - run_started,
        "duration_s": ended - began,
        "completed_units": completed_units,
    }


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - orchestration.
    """Run scoped checks, replay the exact candidate, then publish atomically."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    receipts: list[JsonDict] = []

    phase = time.monotonic()
    progress(started, "manifest", "before_freeze")
    private = Path(tempfile.mkdtemp(prefix="exp7500-validation-", dir="/tmp"))
    plan = build_validation_plan(root, private / "scoped")
    plan_errors = validate_validation_plan(root, plan)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    spans.append(_phase_span("validation_manifest_frozen", phase, started, len(plan)))
    progress(started, "manifest", "after_freeze", commands=len(plan))

    phase = time.monotonic()
    progress(started, "validation", "before_affected_subprocesses", commands=len(plan))
    receipts.extend(
        validation_contract.run_categorized_commands(
            root,
            [validation_contract.PlannedCommand(row, "required_validation", True) for row in plan],
            log_dir=root / RAW_DIR / "validation/affected",
            heartbeat_s=60.0,
        )
    )
    spans.append(_phase_span("affected_validation", phase, started, len(receipts)))
    progress(started, "validation", "after_affected_subprocesses", completed=len(receipts))
    if not _receipts_pass(receipts, AFFECTED_CHECK_NAMES):
        raise RuntimeError("affected_validation_failed")

    phase = time.monotonic()
    progress(started, "aggregation", "before_upstream_and_shard_reduction")
    candidate = build_artifact(
        root,
        run_date=run_date,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        validation_receipts=receipts,
        require_terminal=False,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
    )
    candidate_errors = validate_artifact(candidate, root=root, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")
    current_work_receipt.atomic_json(root / CANDIDATE_PATH, candidate)
    spans.append(
        _phase_span("aggregation_and_independent_reduction", phase, started, len(candidate["rows"]))
    )
    progress(
        started,
        "aggregation",
        "after_upstream_and_shard_reduction",
        completed_units=len(candidate["rows"]),
    )

    phase = time.monotonic()
    terminal_specs = terminal_command_specs(root, root / CANDIDATE_PATH)
    progress(started, "terminal_validation", "before_subprocesses", commands=len(terminal_specs))
    terminal = validation_contract.run_categorized_commands(
        root,
        [
            validation_contract.PlannedCommand(row, "terminal_reader", True)
            for row in terminal_specs
        ],
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    receipts.extend(terminal)
    spans.append(_phase_span("terminal_validation", phase, started, len(terminal)))
    progress(started, "terminal_validation", "after_subprocesses", completed=len(terminal))
    if not _receipts_pass(receipts, REQUIRED_TERMINAL_NAMES):
        raise RuntimeError("terminal_validation_failed")

    artifact = build_artifact(
        root,
        run_date=run_date,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        validation_receipts=receipts,
        require_terminal=True,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
    )
    errors = validate_artifact(artifact, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic_terminal_write", path=RESULT_PATH)
    current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "publish",
        "after_atomic_terminal_write",
        path=RESULT_PATH,
        complete=artifact["arc_opportunity_audit_complete_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public run date or one cold-replay candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--reduce-only", action="store_true")
    args = parser.parse_args(argv)
    if args.date is None and args.replay is None:
        parser.error("--date or --replay is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run host aggregation or independently replay an exact candidate."""

    args = parse_args(argv)
    if args.replay is not None:
        artifact = load_object(args.replay)
        replay = independent_reduce(artifact, REPO_ROOT)
        errors = (
            []
            if args.reduce_only
            else validate_artifact(artifact, root=REPO_ROOT, require_terminal=False)
        )
        print(
            json.dumps(
                {
                    "matches_declared": replay["matches_declared"],
                    "row_count": len(replay["rows"]),
                    "reduction_errors": replay["errors"],
                    "validation_errors": errors,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return int(bool(errors) or replay["matches_declared"] is not True)
    run_experiment(REPO_ROOT, str(args.date))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
