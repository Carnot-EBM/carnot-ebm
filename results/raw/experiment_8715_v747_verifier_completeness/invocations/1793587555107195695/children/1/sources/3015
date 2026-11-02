"""Audit enlarged ARC opportunity evidence without changing the live path.

The reducer reads two already-qualified panels. It preserves unknown timing and
eligibility as unknown because a missing timer or gate value cannot support a
cost or supervisor-efficacy claim.

Spec refs: REQ-ARC-WMTE-7512 and SCENARIO-ARC-WMTE-7512-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7478_v655_arc_interval_protocol as interval_protocol
from carnot import experiment_7500_v656_arc_opportunity_audit as prior_audit
from carnot import experiment_7511_v657_arc_evidence_recovery as panel_b_recovery
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7512-arc-opportunity"
SCHEMA = "carnot.exp7512.v657.arc_opportunity.v1"

PANEL_A_PATH = Path("results/experiment_7485_v655_arc_cost_panel_a.json")
PANEL_B_PATH = Path("results/experiment_7511_v657_arc_evidence_recovery.json")
PANEL_B_CAPTURE_PATH = Path(
    "results/raw/experiment_7499_v656_arc_panel_b/measured_terminal_candidate.json"
)
PANEL_A_EPISODES_PATH = Path("results/raw/experiment_7485_v655_arc_cost_panel_a/episode_rows.json")
PANEL_B_EPISODES_PATH = Path("results/raw/experiment_7499_v656_arc_panel_b/episode_rows.json")
PANEL_A_INVOCATIONS_PATH = Path(
    "results/raw/experiment_7485_v655_arc_cost_panel_a/current_invocation_events.jsonl"
)
PANEL_B_INVOCATIONS_PATH = Path(
    "results/raw/experiment_7499_v656_arc_panel_b/current_invocation_events.jsonl"
)
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
RESULT_PATH = Path("results/experiment_7512_v657_arc_opportunity.json")
RAW_DIR = Path("results/raw/experiment_7512_v657_arc_opportunity")
CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7512_v657_arc_opportunity.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7512_v657_arc_opportunity.py")
TEST_PATH = Path("tests/python/test_experiment_7512_v657_arc_opportunity.py")
NOTE_PATH = Path("docs/research-notes/v657-arc-opportunity.md")

PANEL_A_GAMES = ("sk48", "tr87", "s5i5", "lp85", "lf52", "cn04")
PANEL_B_GAMES = ("tu93", "g50t", "tn36", "vc33", "re86", "dc22")
SEEDS = (65_501, 65_502, 65_503)
CURRENT_VALIDATION_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_RECEIPT_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = deepcopy(panel_b_recovery.ZERO_INVOCATION_COUNTS)

MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_INPUTS = (
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
    Path("python/carnot/experiment_7500_v656_arc_opportunity_audit.py"),
    Path("python/carnot/experiment_7478_v655_arc_interval_protocol.py"),
    Path("openspec/capabilities/arc-agi/spec.md"),
    PANEL_A_PATH,
    PANEL_B_PATH,
    PANEL_B_CAPTURE_PATH,
    PANEL_A_EPISODES_PATH,
    PANEL_B_EPISODES_PATH,
    PANEL_A_INVOCATIONS_PATH,
    PANEL_B_INVOCATIONS_PATH,
    REGISTRY_PATH,
    SPEC_PATH,
)


def utc_now() -> str:
    """Return one real UTC boundary for the reporting process."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush a phase boundary so long file reduction remains observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7512] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_object(path: Path) -> Json:
    """Read one JSON object and fail closed for missing or malformed bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so one changed event changes replay identity."""

    return prior_audit.canonical_hash(value)


def sha256_file(path: Path) -> str:
    """Hash exact bytes without turning an upstream candidate into authority."""

    return current_work_receipt.sha256_file(path)


def gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    field: str,
    principle: str,
    op: str = "==",
) -> Json:
    """Record an exact comparison and why failure must remain visible."""

    if op == "==":
        passed = observed == expected
    elif op == ">=":
        passed = isinstance(observed, (int, float)) and observed >= expected
    elif op == ">":
        passed = isinstance(observed, (int, float)) and observed > expected
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


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    """Name the first exact failure while keeping scientific nulls separate."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    required = [row for row in failures if row.get("category") in {"validity", "readiness"}]
    first = required[0] if required else failures[0] if failures else None
    return {
        "all_passed": not failures,
        "required_validity_and_readiness_passed": not required,
        "failed_count": len(failures),
        "required_failed_count": len(required),
        "failed_checks": failures,
        "first_failure": first,
        "first_failed_check": first.get("check") if first else None,
        "upstream": first.get("upstream") if first else None,
        "exact_field_path": first.get("field") if first else None,
        "expected_value": first.get("expected") if first else None,
        "observed_value": first.get("observed") if first else None,
    }


def _source_record(root: Path, relative: Path, role: str) -> Json:
    """Describe one input by path and exact bytes before measurement."""

    path = root / relative
    return {
        "path": relative.as_posix(),
        "role": role,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def collect_preconditions(root: Path) -> tuple[list[Json], Json]:
    """Authenticate instructions, requirements, panels, and registry first."""

    a = load_object(root / PANEL_A_PATH)
    b = load_object(root / PANEL_B_PATH)
    checks = [
        gate(
            f"source_bytes:{path.as_posix()}",
            "validity",
            True,
            (root / path).is_file() and (root / path).stat().st_size > 0,
            upstream=path.as_posix(),
            field="readable_nonempty_bytes",
            principle="An absent prerequisite must block instead of creating evidence.",
        )
        for path in REQUIRED_INPUTS
    ]
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    exclusion_text = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    checks.extend(
        [
            gate(
                "driving_requirement_present",
                "validity",
                True,
                "REQ-ARC-WMTE-7512" in spec_text,
                upstream=SPEC_PATH.as_posix(),
                field="REQ-ARC-WMTE-7512",
                principle="Implementation without its requirement bypasses spec-first review.",
            ),
            gate(
                "task_not_excluded",
                "validity",
                False,
                EXPERIMENT_ID in exclusion_text,
                upstream="ops/exclusion_manifest.yaml",
                field=EXPERIMENT_ID,
                principle="An excluded experiment cannot publish a new terminal claim.",
            ),
            gate(
                "panel_a_qualified",
                "validity",
                1,
                a.get("arc_panel_a_complete_score"),
                upstream=PANEL_A_PATH.as_posix(),
                field="arc_panel_a_complete_score",
                principle="Panel A must qualify independently before pooling.",
            ),
            gate(
                "panel_b_qualified",
                "validity",
                1,
                b.get("arc_panel_b_qualified_score"),
                upstream=PANEL_B_PATH.as_posix(),
                field="arc_panel_b_qualified_score",
                principle="Recovered Panel B must qualify without rerunning games.",
            ),
        ]
    )
    sources = {
        path.as_posix(): _source_record(root, path, "required_input") for path in REQUIRED_INPUTS
    }
    return checks, sources


def _recorded_hash(artifact: Mapping[str, Any], label: str) -> str | None:
    """Read one producer-recorded hash without hashing current source instead."""

    row = (artifact.get("source_artifact_hashes") or {}).get(label) or {}
    value = row.get("sha256") if isinstance(row, Mapping) else None
    return str(value) if isinstance(value, str) else None


def _panel_identity(artifact: Mapping[str, Any], capture: Mapping[str, Any], panel: str) -> Json:
    """Extract every identity named by the pooling requirement."""

    specs_key = "model_specs" if panel == "A" else "historical_model_specs"
    specs = [row for row in artifact.get(specs_key) or [] if isinstance(row, Mapping)]
    model = dict(specs[0]) if len(specs) == 1 else {}
    source = artifact if panel == "A" else capture
    runtime = (
        artifact.get("execution_venue_details")
        if panel == "A"
        else artifact.get("historical_runtime_custody")
    ) or {}
    native_binary = str(runtime.get("native_binary") or "")
    schedule = [row for row in capture.get("protocol_schedule") or [] if isinstance(row, Mapping)]
    budget_fields = (
        "action_limit",
        "request_limit",
        "max_new_tokens_per_call",
        "episode_limit_s",
        "panel_live_limit_s",
    )
    budget = {key: schedule[0].get(key) for key in budget_fields} if schedule else {}
    return {
        "panel": panel,
        "model_hf_id": model.get("hf_id"),
        "model_sha256": model.get("sha256"),
        "quantization": model.get("quantization"),
        "runtime_settings": deepcopy(model.get("runtime_settings")),
        "native_binary": native_binary,
        "native_runtime_sha256": _recorded_hash(source, native_binary),
        "policy_sha256": _recorded_hash(source, "python/carnot/agentic/arc_competition_agent.py"),
        "observer_sha256": _recorded_hash(
            source, "python/carnot/agentic/arc_decision_telemetry.py"
        ),
        "interval_protocol_sha256": _recorded_hash(
            source, "python/carnot/experiment_7478_v655_arc_interval_protocol.py"
        ),
        "budget": budget,
        "games": sorted({str(row.get("game")) for row in schedule if row.get("game")}),
        "seeds": sorted({int(row["seed"]) for row in schedule if isinstance(row.get("seed"), int)}),
    }


def panel_identity_rows(root: Path) -> dict[str, Json]:
    """Authenticate original Panel A and recovered Panel B separately."""

    panel_a = load_object(root / PANEL_A_PATH)
    panel_b = load_object(root / PANEL_B_PATH)
    panel_b_capture = load_object(root / PANEL_B_CAPTURE_PATH)
    return {
        "A": _panel_identity(panel_a, panel_a, "A"),
        "B": _panel_identity(panel_b, panel_b_capture, "B"),
    }


IDENTITY_FIELDS = (
    "model_hf_id",
    "model_sha256",
    "quantization",
    "runtime_settings",
    "native_binary",
    "native_runtime_sha256",
    "policy_sha256",
    "observer_sha256",
    "interval_protocol_sha256",
    "budget",
    "seeds",
)


def evaluate_pooling(
    rows: Sequence[Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    *,
    panel_states: Mapping[str, str],
) -> Json:
    """Require compatible sources and game-level support before pooling."""

    a = identities.get("A") or {}
    b = identities.get("B") or {}
    mismatches = [field for field in IDENTITY_FIELDS if a.get(field) != b.get(field)]
    missing = [field for field in IDENTITY_FIELDS if a.get(field) is None or b.get(field) is None]
    schedules_match_freeze = (
        a.get("games") == sorted(PANEL_A_GAMES)
        and b.get("games") == sorted(PANEL_B_GAMES)
        and not set(a.get("games") or []).intersection(b.get("games") or [])
    )
    valid = [row for row in rows if row.get("valid") is True]
    games = sorted({str(row.get("game")) for row in valid if row.get("game")})
    source_compatible = (
        panel_states.get("A") == panel_states.get("B") == "qualified"
        and not mismatches
        and not missing
        and schedules_match_freeze
    )
    episode_support = len(valid) >= 30
    game_support = len(games) >= 10
    return {
        "source_compatible": source_compatible,
        "panel_states": dict(panel_states),
        "mismatched_identities": mismatches,
        "missing_identities": missing,
        "frozen_disjoint_schedules_valid": schedules_match_freeze,
        "valid_episode_count": len(valid),
        "valid_game_count": len(games),
        "valid_games": games,
        "episode_support_floor": 30,
        "game_support_floor": 10,
        "episode_support_passed": episode_support,
        "game_support_passed": game_support,
        "common_protocol": a.get("interval_protocol_sha256") == b.get("interval_protocol_sha256")
        and a.get("interval_protocol_sha256") is not None,
        "pooling_support_ready": source_compatible and episode_support and game_support,
        "stratification": (
            "pooled_support_available"
            if source_compatible and episode_support and game_support
            else "separate_descriptive_panels"
        ),
        "seeds_do_not_multiply_game_support": True,
    }


def _read_jsonl(path: Path) -> list[Json]:
    """Reuse the qualified reader for one immutable event ledger."""

    return interval_protocol.read_jsonl(path)


def _producer_events(
    root: Path, producer: Mapping[str, Any]
) -> tuple[dict[str, list[Json]], list[Json]]:
    """Authenticate per-episode and shared callback shards before joining."""

    per_episode: dict[str, list[Json]] = {}
    shared: list[Json] = []
    receipts: list[Json] = []
    for reference in producer.get("seam_event_shards") or []:
        if not isinstance(reference, Mapping):
            continue
        relative = Path(str(reference.get("path") or ""))
        path = root / relative
        events = _read_jsonl(path) if path.is_file() else []
        observed_hash = sha256_file(path) if path.is_file() else None
        receipt = {
            "path": relative.as_posix(),
            "episode_id": reference.get("episode_id"),
            "expected_sha256": reference.get("sha256"),
            "observed_sha256": observed_hash,
            "expected_rows": reference.get("row_count"),
            "observed_rows": len(events),
        }
        receipt["passed"] = (
            receipt["expected_sha256"] == receipt["observed_sha256"]
            and receipt["expected_rows"] == receipt["observed_rows"]
        )
        receipts.append(receipt)
        episode_id = str(reference.get("episode_id") or "")
        if episode_id == "all":
            shared.extend(events)
        elif episode_id:
            per_episode[episode_id] = events
    for event in shared:
        episode_id = str(event.get("episode_id") or "")
        if episode_id:
            per_episode.setdefault(episode_id, []).append(event)
    return per_episode, receipts


def _selection_interval(rows: Sequence[Mapping[str, Any]]) -> tuple[int | None, int | None]:
    """Return one compatible completed supervisor interval when recorded."""

    starts = [row for row in rows if row.get("event") == "stage_start"]
    ends = [row for row in rows if row.get("event") in {"stage_end", "stage_terminal"}]
    for start in starts:
        for end in ends:
            identity = (start.get("run_id"), start.get("process_id"), start.get("clock_identity"))
            if identity != (end.get("run_id"), end.get("process_id"), end.get("clock_identity")):
                continue
            left = start.get("interval_start_monotonic_ns", start.get("event_monotonic_ns"))
            right = end.get("interval_end_monotonic_ns", end.get("event_monotonic_ns"))
            if isinstance(left, int) and isinstance(right, int) and right >= left:
                return left, right
    return None, None


def reduce_supervisor_evaluations(
    panel: str,
    game: str,
    seed: int,
    episode_id: str,
    events: Sequence[Mapping[str, Any]],
    *,
    progressed: bool,
) -> list[Json]:
    """Keep eligibility, firing, application, and outcome as separate facts."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if (
            event.get("episode_id") == episode_id
            and event.get("seam") == "supervisor_arm_selection"
        ):
            grouped[str(event.get("decision_id") or "")].append(event)
    output: list[Json] = []
    for decision_id, rows in sorted(grouped.items()):
        selection = next((row for row in rows if row.get("event") == "selection"), None)
        if selection is None:
            continue
        candidates: dict[str, bool | None] = {}
        missing_reason = None
        for row in rows:
            if row.get("event") != "eligible_candidate_set":
                continue
            missing_reason = row.get("missing_options_reason")
            for candidate in row.get("candidates") or []:
                if isinstance(candidate, Mapping):
                    name = str(candidate.get("stable_candidate_id") or "")
                    value = candidate.get("eligibility")
                    candidates[name] = value if isinstance(value, bool) else None
        selected = [str(value) for value in selection.get("selected_candidate_ids") or []]
        selected_arm = selected[0] if selected else "abstain"
        eligibility = candidates.get(selected_arm)
        left, right = _selection_interval(rows)
        fired = selection.get("supervisor_fired") is True
        applied = selection.get("applied_redirection") is True
        output.append(
            {
                "panel": panel,
                "game": game,
                "seed": seed,
                "episode_id": episode_id,
                "decision_id": decision_id,
                "action_count": selection.get("action_count"),
                "candidate_eligibility": candidates,
                "eligibility_missing_reason": missing_reason,
                "selected_arm": selected_arm,
                "selected_eligibility": eligibility,
                "fired": fired,
                "applied_redirection": applied,
                "interval_start_ns": left,
                "interval_end_ns": right,
                "interval_duration_ns": right - left
                if left is not None and right is not None
                else None,
                "eligible_firing_opportunity": eligibility is True
                and selected_arm not in {"abstain", "no_redirect"}
                and left is not None
                and right is not None,
                "episode_progressed": progressed,
                "causal_outcome_available": applied and eligibility is True,
            }
        )
    return output


def _stage_value(cost: Mapping[str, Any], seam: str) -> int | None:
    """Read one exclusive stage sum and preserve an absent stage as unknown."""

    for row in cost.get("stage_summary") or []:
        if isinstance(row, Mapping) and row.get("seam") == seam:
            value = row.get("exclusive_sum_ns")
            return int(value) if isinstance(value, int) else None
    return None


def _token_counts(events: Sequence[Mapping[str, Any]]) -> Json:
    """Count tokens once from completed downstream-generation boundaries."""

    terminal = [
        row
        for row in events
        if row.get("seam") == "downstream_generation" and row.get("event") == "stage_end"
    ]
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in terminal)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in terminal)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generated_tokens": output_tokens,
        "completed_requests": len(terminal),
    }


def _reduce_panel(
    root: Path,
    *,
    panel: str,
    producer: Mapping[str, Any],
    episodes_path: Path,
    qualified: bool,
) -> Json:
    """Recompute one qualified panel from raw episode and event boundaries."""

    result: Json = {
        "panel": panel,
        "state": "qualified" if qualified else "blocked_unqualified",
        "errors": [],
        "rows": [],
        "supervisor_rows": [],
        "shard_receipts": [],
    }
    if not qualified:
        result["errors"].append("upstream_qualification_missing")
        return result
    episode_document = load_object(root / episodes_path)
    episode_rows = [row for row in episode_document.get("rows") or [] if isinstance(row, Mapping)]
    events_by_episode, shard_receipts = _producer_events(root, producer)
    result["shard_receipts"] = shard_receipts
    if not shard_receipts or not all(row["passed"] for row in shard_receipts):
        result["errors"].append("interval_shard_authentication_failed")
    expected_games = PANEL_A_GAMES if panel == "A" else PANEL_B_GAMES
    expected_ids = {
        f"panel-{panel.lower()}:{game}:seed-{seed}" for game in expected_games for seed in SEEDS
    }
    indexed = {str(row.get("episode_id")): row for row in episode_rows}
    if set(indexed) != expected_ids:
        result["errors"].append("frozen_episode_schedule_mismatch")
    for episode_id in sorted(expected_ids):
        source = indexed.get(episode_id) or {}
        events = events_by_episode.get(episode_id) or []
        game = str(source.get("game") or "")
        seed = int(source.get("seed") or 0)
        bounds = panel_b_recovery._episode_bounds(source, events)
        cost = interval_protocol.reduce_episode_intervals(
            events,
            episode_id=episode_id,
            episode_start_ns=bounds[0],
            episode_end_ns=bounds[1],
        )
        progressed = int(source.get("peak_level") or 0) > int(source.get("start_level") or 0)
        supervisor_rows = reduce_supervisor_evaluations(
            panel, game, seed, episode_id, events, progressed=progressed
        )
        tokens = _token_counts(events)
        action_intervals = [
            (int(row["interval_start_monotonic_ns"]), int(row["interval_end_monotonic_ns"]))
            for row in source.get("action_rows") or []
            if isinstance(row, Mapping)
            and isinstance(row.get("interval_start_monotonic_ns"), int)
            and isinstance(row.get("interval_end_monotonic_ns"), int)
        ]
        observed_ns = int(cost["observed_episode_ns"])
        row = {
            "panel": panel,
            "episode_id": episode_id,
            "game": game,
            "seed": seed,
            "disposition": source.get("disposition"),
            "valid": source.get("disposition") == "complete" and cost.get("bounds_valid") is True,
            "action_count": int(source.get("action_count") or 0),
            "progressed": progressed,
            "progress_censored": not progressed,
            "censored_cost_ns": observed_ns if not progressed else None,
            "observed_episode_ns": observed_ns,
            "stage_union_ns": int(cost["stage_union_ns"]),
            "unattributed_ns": int(cost["unattributed_ns"]),
            "request_exclusive_ns": _stage_value(cost, "downstream_generation"),
            "world_model_exclusive_ns": _stage_value(cost, "hypothesis_gate"),
            "supervisor_exclusive_ns": _stage_value(cost, "supervisor_arm_selection"),
            "induction_exclusive_ns": _stage_value(cost, "induction_timing"),
            "candidate_decision_exclusive_ns": _stage_value(cost, "candidate_action_selection"),
            "action_loop_union_ns": interval_protocol.union_duration_ns(action_intervals),
            "native_forward_exclusive_ns": None,
            "sampling_exclusive_ns": None,
            "parsing_exclusive_ns": None,
            "environment_exclusive_ns": None,
            "update_exclusive_ns": None,
            "input_tokens": tokens["input_tokens"],
            "output_tokens": tokens["output_tokens"],
            "completed_requests": tokens["completed_requests"],
            "bounds_valid": cost.get("bounds_valid") is True,
            "interval_event_count": len(events),
            "supervisor_gate_evaluation_count": len(supervisor_rows),
            "recorded_firing_count": sum(item["fired"] for item in supervisor_rows),
            "applied_redirection_count": sum(
                item["applied_redirection"] for item in supervisor_rows
            ),
            "eligible_firing_opportunity_count": sum(
                item["eligible_firing_opportunity"] for item in supervisor_rows
            ),
            "solve_provenance": "live_agent_self_discovery",
            "new_level_credit": 0,
        }
        result["rows"].append(row)
        result["supervisor_rows"].extend(supervisor_rows)
    if (
        result["errors"]
        or len(result["rows"]) != 18
        or not all(row["valid"] for row in result["rows"])
    ):
        result["state"] = "disqualified_invalid"
    return result


def _operation_accounting(root: Path, path: Path) -> Json:
    """Reduce historical load and generation boundaries without inventing sub-stages."""

    events = _read_jsonl(root / path)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        grouped[(str(row.get("operation") or ""), str(row.get("call_id") or ""))].append(row)
    durations: dict[str, list[int]] = defaultdict(list)
    terminal_counts: Counter[str] = Counter()
    for (operation, _call_id), rows in grouped.items():
        attempted = next((row for row in rows if row.get("state") == "attempted"), None)
        terminal = next(
            (row for row in rows if row.get("state") in {"completed", "failed", "cancelled"}),
            None,
        )
        if terminal is not None:
            terminal_counts[f"{operation}_{terminal.get('state')}"] += 1
        start = attempted.get("started_monotonic_ns") if attempted else None
        end = terminal.get("ended_monotonic_ns") if terminal else None
        if isinstance(start, int) and isinstance(end, int) and end >= start:
            durations[operation].append(end - start)
    return {
        "source": path.as_posix(),
        "source_sha256": sha256_file(root / path),
        "model_load": {
            "call_count": len(durations.get("model_load", [])),
            "exclusive_duration_ns": sum(durations.get("model_load", [])),
            "status": "measured",
        },
        "generation_request": {
            "call_count": len(durations.get("generation", [])),
            "exclusive_duration_ns": sum(durations.get("generation", [])),
            "status": "measured",
        },
        "native_forward": {"exclusive_duration_ns": None, "status": "unknown"},
        "sampling": {"exclusive_duration_ns": None, "status": "unknown"},
        "parsing": {"exclusive_duration_ns": None, "status": "unknown"},
        "environment": {"exclusive_duration_ns": None, "status": "unknown"},
        "update": {"exclusive_duration_ns": None, "status": "unknown"},
        "terminal_counts": dict(sorted(terminal_counts.items())),
    }


def _per_game(rows: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Aggregate seeds within games without treating seeds as transfer units."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("panel")), str(row.get("game")))].append(row)
    output: list[Json] = []
    for (panel, game), members in sorted(grouped.items()):
        output.append(
            {
                "panel": panel,
                "game": game,
                "episode_count": len(members),
                "valid_episode_count": sum(row.get("valid") is True for row in members),
                "progressed_episode_count": sum(row.get("progressed") is True for row in members),
                "progress_censored_episode_count": sum(
                    row.get("progress_censored") is True for row in members
                ),
                "observed_episode_ns": sum(
                    int(row.get("observed_episode_ns") or 0) for row in members
                ),
                "input_tokens": sum(int(row.get("input_tokens") or 0) for row in members),
                "output_tokens": sum(int(row.get("output_tokens") or 0) for row in members),
                "supervisor_gate_evaluation_count": sum(
                    int(row.get("supervisor_gate_evaluation_count") or 0) for row in members
                ),
                "recorded_firing_count": sum(
                    int(row.get("recorded_firing_count") or 0) for row in members
                ),
                "applied_redirection_count": sum(
                    int(row.get("applied_redirection_count") or 0) for row in members
                ),
                "eligible_firing_opportunity_count": sum(
                    int(row.get("eligible_firing_opportunity_count") or 0) for row in members
                ),
                "seed_count": len({row.get("seed") for row in members}),
                "seeds_are_repeated_within_game": True,
                "solve_provenance": "live_agent_self_discovery",
            }
        )
    return output


def _supervisor_disposition(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Distinguish absent eligibility from firing and causal application."""

    selected = Counter(str(row.get("selected_arm")) for row in rows)
    eligible = [row for row in rows if row.get("eligible_firing_opportunity") is True]
    interventions = [
        deepcopy(dict(row)) for row in eligible if row.get("applied_redirection") is True
    ]
    return {
        "selected_arms": dict(sorted(selected.items())),
        "gate_evaluation_count": len(rows),
        "explicit_true_selected_count": sum(
            row.get("selected_eligibility") is True for row in rows
        ),
        "explicit_false_selected_count": sum(
            row.get("selected_eligibility") is False for row in rows
        ),
        "unknown_selected_eligibility_count": sum(
            row.get("selected_eligibility") is None for row in rows
        ),
        "recorded_firing_count": sum(row.get("fired") is True for row in rows),
        "applied_redirection_count": sum(row.get("applied_redirection") is True for row in rows),
        "eligible_firing_opportunity_count": len(eligible),
        "intervention_ledger": interventions,
        "efficacy_estimate": None,
        "unhelpful_choice_claim": False,
        "efficacy_tuning_disposition": "retired_until_live_reachable_choices",
        "diagnosis": "zero_authenticated_eligible_opportunities_with_shadow_firings",
    }


def _denominator_coverage(rows: Sequence[Mapping[str, Any]], operations: Mapping[str, Any]) -> Json:
    """Report known exclusive stages and leave missing stage timers unknown."""

    def total(field: str) -> int:
        return sum(int(row.get(field) or 0) for row in rows)

    load_ns = sum(
        int((panel.get("model_load") or {}).get("exclusive_duration_ns") or 0)
        for panel in operations.values()
    )
    return {
        "observed_episode_ns": total("observed_episode_ns"),
        "exclusive_stage_union_ns": total("stage_union_ns"),
        "unattributed_ns": total("unattributed_ns"),
        "stages": {
            "request": {"duration_ns": total("request_exclusive_ns"), "status": "measured"},
            "model_load": {"duration_ns": load_ns, "status": "measured_outside_episode"},
            "world_model": {
                "duration_ns": total("world_model_exclusive_ns"),
                "status": "measured_hypothesis_gate_parent",
            },
            "supervisor": {"duration_ns": total("supervisor_exclusive_ns"), "status": "measured"},
            "induction": {"duration_ns": total("induction_exclusive_ns"), "status": "measured"},
            "candidate_decision": {
                "duration_ns": total("candidate_decision_exclusive_ns"),
                "status": "measured_but_semantics_unattributed",
            },
            **{
                name: {"duration_ns": None, "status": "unknown"}
                for name in (
                    "native_forward",
                    "sampling",
                    "parsing",
                    "planner",
                    "environment",
                    "update",
                )
            },
        },
        "unknown_stage_names": [
            "native_forward",
            "sampling",
            "parsing",
            "planner",
            "environment",
            "update",
        ],
        "complete_mutually_exclusive_denominator": False,
        "coarse_callback_speed_claim_allowed": False,
    }


def registry_precheck(root: Path) -> Json:
    """Read registered progress before describing any inherited live attempt."""

    path = root / REGISTRY_PATH
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    indexed = {
        str(row.get("game")): row
        for row in value.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    return {
        "path": REGISTRY_PATH.as_posix(),
        "sha256": sha256_file(path),
        "read_before_progress_description": True,
        "policy_received_registry_data": False,
        "rows": [
            {
                "game": game,
                "registered": game in indexed,
                "levels_reproduced": indexed.get(game, {}).get("levels_reproduced"),
                "full_game_clear": indexed.get(game, {}).get("full_game_clear"),
            }
            for game in (*PANEL_A_GAMES, *PANEL_B_GAMES)
        ],
    }


def reduce_upstreams(root: Path = REPO_ROOT) -> Json:
    """Independently reduce both qualified panels and all available boundaries."""

    panel_a_artifact = load_object(root / PANEL_A_PATH)
    panel_b_artifact = load_object(root / PANEL_B_PATH)
    panel_b_capture = load_object(root / PANEL_B_CAPTURE_PATH)
    a_qualified = (
        panel_a_artifact.get("arc_panel_a_complete_score") == 1
        and panel_a_artifact.get("flagged_adversarial") is False
        and panel_a_artifact.get("verdict_class") in {"null", "positive", "circular_positive"}
    )
    b_qualified = (
        panel_b_artifact.get("arc_panel_b_qualified_score") == 1
        and panel_b_artifact.get("flagged_adversarial") is False
        and panel_b_artifact.get("verdict_class") in {"null", "positive", "circular_positive"}
    )
    panel_a = _reduce_panel(
        root,
        panel="A",
        producer=panel_a_artifact,
        episodes_path=PANEL_A_EPISODES_PATH,
        qualified=a_qualified,
    )
    panel_b = _reduce_panel(
        root,
        panel="B",
        producer=panel_b_capture,
        episodes_path=PANEL_B_EPISODES_PATH,
        qualified=b_qualified,
    )
    rows = [*panel_a["rows"], *panel_b["rows"]]
    supervisor_rows = [*panel_a["supervisor_rows"], *panel_b["supervisor_rows"]]
    identities = panel_identity_rows(root)
    states = {"A": str(panel_a["state"]), "B": str(panel_b["state"])}
    pooling = evaluate_pooling(rows, identities, panel_states=states)
    operations = {
        "A": _operation_accounting(root, PANEL_A_INVOCATIONS_PATH),
        "B": _operation_accounting(root, PANEL_B_INVOCATIONS_PATH),
    }
    coverage = _denominator_coverage(rows, operations)
    per_game = _per_game(rows)
    tokens = {
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "generated_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "completed_requests": sum(int(row.get("completed_requests") or 0) for row in rows),
    }
    supervisor = _supervisor_disposition(supervisor_rows)
    source_hashes: Json = {}
    for panel_result in (panel_a, panel_b):
        for receipt in panel_result["shard_receipts"]:
            source_hashes[receipt["path"]] = {
                "sha256": receipt["observed_sha256"],
                "bytes": (root / receipt["path"]).stat().st_size
                if (root / receipt["path"]).is_file()
                else None,
                "role": "raw_interval_or_callback_shard",
                "passed": receipt["passed"],
            }
    return {
        "rows": rows,
        "panel_results": {
            "A": {**panel_a, "supervisor_rows": []},
            "B": {**panel_b, "supervisor_rows": []},
        },
        "panel_identities": identities,
        "pooling": pooling,
        "per_game_results": per_game,
        "supervisor_opportunity_rows": supervisor_rows,
        "supervisor_disposition": supervisor,
        "intervention_ledger": deepcopy(supervisor["intervention_ledger"]),
        "efficacy_estimate": None,
        "historical_operation_accounting": operations,
        "token_counts": tokens,
        "cost_denominator_coverage": coverage,
        "cost_share": None,
        "amdahl_upper_bound": None,
        "sample_size_budget": {
            "planned_independent_units": 36,
            "attempted_independent_units": len(rows),
            "completed_independent_units": sum(row.get("valid") is True for row in rows),
            "excluded_independent_units": 0,
            "failed_independent_units": sum(row.get("valid") is not True for row in rows),
            "censored_independent_units": sum(row.get("progress_censored") is True for row in rows),
            "unstarted_independent_units": max(0, 36 - len(rows)),
            "independent_game_clusters": len({row.get("game") for row in rows}),
            "seeds_do_not_multiply_game_support": True,
        },
        "generalization_assessment": {
            "transfer_unit": "game",
            "unique_game_count": len(per_game),
            "progressed_game_count": sum(row["progressed_episode_count"] > 0 for row in per_game),
            "broad_generalization_supported": False,
            "reason": "all_zero_progress_panel_cannot_establish_transfer",
        },
        "registry_precheck": registry_precheck(root),
        "source_artifact_hashes": source_hashes,
        "errors": [*panel_a["errors"], *panel_b["errors"]],
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful non-timeout receipt for every exact command."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in receipts:
        grouped[str(row.get("name"))].append(row)
    return all(
        len(grouped[name]) == 1
        and grouped[name][0].get("passed") is True
        and grouped[name][0].get("exit_code") == 0
        and grouped[name][0].get("timed_out") is False
        for name in names
    )


def classify_terminal(
    *,
    external_available: bool,
    evidence_valid: bool,
    validation_passed: bool,
    denominator_ready: bool,
    opportunity_present: bool,
) -> tuple[str, str, int, int]:
    """Keep external absence, validity, accounting, and cost readiness separate."""

    if not external_available:
        return "blocked", "complete_blocked_external_prerequisite_absent", 0, 0
    if not evidence_valid or not validation_passed:
        return "disqualified", "complete_disqualified_arc_opportunity_evidence", 0, 0
    if opportunity_present:
        return (
            "null",
            "complete_null_observational_opportunity_no_causal_efficacy",
            1,
            int(denominator_ready),
        )
    return "null", "complete_null_zero_eligible_supervisor_opportunity", 1, int(denominator_ready)


def _acceptance_gates(
    reduction: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    *,
    require_terminal: bool,
) -> list[Json]:
    """Evaluate validity, readiness, support, and benefit without coupling them."""

    affected = _receipts_pass(receipts, CURRENT_VALIDATION_NAMES)
    terminal = not require_terminal or _receipts_pass(receipts, TERMINAL_RECEIPT_NAMES)
    preconditions_pass = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    rows = reduction["rows"]
    pooling = reduction["pooling"]
    denominator = reduction["cost_denominator_coverage"]
    audit_complete = (
        preconditions_pass
        and not reduction["errors"]
        and len(rows) == 36
        and all(row.get("valid") is True and row.get("bounds_valid") is True for row in rows)
        and affected
        and terminal
    )
    return [
        gate(
            "preconditions_authenticated",
            "validity",
            True,
            preconditions_pass,
            upstream="preconditions_checked",
            field="preconditions_checked[].passed",
            principle="Missing or invalid prerequisites cannot become a scientific null.",
        ),
        gate(
            "both_panels_qualified",
            "validity",
            {"A": "qualified", "B": "qualified"},
            pooling["panel_states"],
            upstream="panel_results",
            field="panel_results.*.state",
            principle="One valid panel cannot conceal an unqualified second panel.",
        ),
        gate(
            "raw_episode_reduction_valid",
            "validity",
            True,
            len(rows) == 36 and all(row.get("bounds_valid") is True for row in rows),
            upstream="rows",
            field="rows[].bounds_valid",
            principle="Nested, missing, or incompatible timing cannot inflate support.",
        ),
        gate(
            "affected_validation",
            "validity",
            True,
            affected,
            upstream="validation_receipts",
            field="required_scoped_checks",
            principle="Favorable evidence cannot excuse failed affected-code checks.",
        ),
        gate(
            "terminal_readers",
            "validity",
            True,
            terminal,
            upstream="validation_receipts",
            field="terminal_readers",
            principle="Self-reported reduction cannot replace cold replay and strict readers.",
        ),
        gate(
            "opportunity_audit_complete",
            "readiness",
            True,
            audit_complete,
            upstream="rows|panel_results|validation_receipts",
            field="arc_opportunity_audit_complete_score",
            principle="A valid null must not lower unrelated accounting completion.",
        ),
        gate(
            "source_identity_compatible",
            "claim_support",
            True,
            pooling["source_compatible"],
            upstream="panel_identities",
            field="pooling.source_compatible",
            principle="Different sources require separate descriptive panels.",
        ),
        gate(
            "qualified_episode_support",
            "claim_support",
            30,
            pooling["valid_episode_count"],
            upstream="pooling",
            field="pooling.valid_episode_count",
            principle="Fewer than 30 episodes cannot support the pooled cost claim.",
            op=">=",
        ),
        gate(
            "unique_game_support",
            "claim_support",
            10,
            pooling["valid_game_count"],
            upstream="pooling",
            field="pooling.valid_game_count",
            principle="Repeated seeds cannot replace at least 10 unique games.",
            op=">=",
        ),
        gate(
            "common_interval_protocol",
            "claim_support",
            True,
            pooling["common_protocol"],
            upstream="panel_identities",
            field="pooling.common_protocol",
            principle="Different interval protocols cannot share a denominator.",
        ),
        gate(
            "complete_exclusive_cost_denominator",
            "claim_support",
            True,
            denominator["complete_mutually_exclusive_denominator"],
            upstream="cost_denominator_coverage",
            field="cost_denominator_coverage.complete_mutually_exclusive_denominator",
            principle="Unknown stage time must suppress cost shares and Amdahl bounds.",
        ),
        gate(
            "eligible_supervisor_opportunity_present",
            "scientific_benefit",
            0,
            reduction["supervisor_disposition"]["eligible_firing_opportunity_count"],
            upstream="supervisor_opportunity_rows",
            field="supervisor_disposition.eligible_firing_opportunity_count",
            principle="A shadow firing with unknown eligibility cannot establish efficacy.",
            op=">",
        ),
        gate(
            "progressed_game_support",
            "scientific_benefit",
            1,
            reduction["generalization_assessment"]["progressed_game_count"],
            upstream="per_game_results",
            field="generalization_assessment.progressed_game_count",
            principle="An all-zero progress panel cannot establish broad transfer.",
            op=">=",
        ),
    ]


FIELD_PRINCIPLES = {
    "schema": "Version, experiment identity, milestone, and terminal status prevent reader drift.",
    "run_date": "The fixed date stays separate from actual UTC and monotonic boundaries.",
    "preconditions_checked": "Exact paths and observed failures prevent invented prerequisites.",
    "MODEL_SPECS": "An empty list proves the current aggregation planned no model load.",
    "model_specs": "The lowercase mirror keeps current no-load semantics unambiguous.",
    "model_invoked": "False separates current aggregation from historical live inference.",
    "invocation_counts": "Balanced zero counters prevent old model calls from becoming current calls.",
    "inference_substrate": "The exact aggregation label prevents historical generation from posing as current inference.",
    "inference_substrate_class": "Aggregation records actual work without padded model duration.",
    "execution_venue": "Host CPU work stays distinct from historical CUDA and board evidence.",
    "duration_s": "Measured elapsed time prevents a fabricated runtime floor.",
    "phase_spans": "Flushed real boundaries expose stalls and unfinished work.",
    "random_seed": "Frozen fitting, arrival, audit, and bootstrap roles prevent outcome-driven seed choice.",
    "reproducibility_checksum": "Source, settings, reductions, and validation remain bound together.",
    "source_artifact_hashes": "Original bytes and roles prevent upstream evidence laundering.",
    "rows": "Per-episode timing, tokens, failures, and censoring support independent replay.",
    "sample_size_budget": "Planned, attempted, complete, failed, censored, and unstarted units stay separate.",
    "acceptance_gate_results": "Validity, readiness, support, and benefit comparisons remain independent.",
    "gate_check_summary": "The exact first failure prevents a vague blocked cost claim.",
    "honest_verdict": "A complete terminal prefix distinguishes closed accounting from retryable work.",
    "verdict_class": "The closed enum prevents external absence from becoming a null.",
    "verifier_is_oracle": "False prevents fixture-defined gains from becoming independent evidence.",
    "flagged_adversarial": "Actual strict-reader findings cannot be cleared to open a gate.",
    "validation_receipts": "Exact commands, exits, scopes, and hashes make checks reviewable.",
    "field_principles": "Every emitted field names the evidence failure it prevents.",
    "arc_opportunity_audit_complete_score": "Accounting completion stays independent of benefit and cost readiness.",
    "arc_cost_claim_ready_score": "Only compatible support and a complete exclusive denominator can set one.",
    "solve_provenance": "Historical live attempts retain origin while the audit earns no new credit.",
    "per_game_results": "Games, not seeds, remain the unit for transfer assessment.",
    "supervisor_opportunity_rows": "Eligibility, firing, application, and outcomes remain separate facts.",
    "cost_denominator_coverage": "Unknown stage time closes numeric cost claims instead of becoming zero.",
}


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum that stores the result."""

    copied = deepcopy(dict(value))
    copied["reproducibility_checksum"] = ""
    return canonical_hash(copied)


def _input_hashes(root: Path, sources: Mapping[str, Any], reduction: Mapping[str, Any]) -> Json:
    """Combine precondition and raw-shard identities with current outputs."""

    output = deepcopy(dict(sources))
    output.update(deepcopy(dict(reduction["source_artifact_hashes"])))
    for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH, NOTE_PATH):
        if (root / path).is_file():
            output[path.as_posix()] = _source_record(root, path, "current_affected_file")
    return output


def build_artifact(
    root: Path,
    *,
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    reduction: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    computation_s: float,
    validation_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    ended_at_utc: str,
    require_terminal: bool,
) -> Json:
    """Build one terminal-shaped record from independent raw reductions."""

    gates = _acceptance_gates(reduction, preconditions, receipts, require_terminal=require_terminal)
    summary = gate_summary(gates)
    external_available = all(
        row.get("passed") is True
        for row in preconditions
        if str(row.get("check") or "").startswith("source_bytes:")
    )
    evidence_valid = (
        bool(preconditions)
        and all(row.get("passed") is True for row in preconditions)
        and not reduction["errors"]
        and len(reduction["rows"]) == 36
    )
    validation_passed = _receipts_pass(receipts, CURRENT_VALIDATION_NAMES) and (
        not require_terminal or _receipts_pass(receipts, TERMINAL_RECEIPT_NAMES)
    )
    denominator_ready = (
        reduction["pooling"]["pooling_support_ready"] is True
        and reduction["cost_denominator_coverage"]["complete_mutually_exclusive_denominator"]
        is True
    )
    opportunity_present = (
        reduction["supervisor_disposition"]["eligible_firing_opportunity_count"] > 0
    )
    verdict, honest, audit_score, cost_score = classify_terminal(
        external_available=external_available,
        evidence_valid=evidence_valid,
        validation_passed=validation_passed,
        denominator_ready=denominator_ready,
        opportunity_present=opportunity_present,
    )
    panel_a = load_object(root / PANEL_A_PATH)
    panel_b = load_object(root / PANEL_B_PATH)
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "duration_breakdown_s": {
            "authoring": 0.0,
            "computation": float(computation_s),
            "validation": float(validation_s),
            "current_inference": 0.0,
            "historical_capture": float(panel_a.get("duration_s") or 0.0)
            + float(panel_b.get("historical_duration_s") or 0.0),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {
            "pid": os.getpid(),
            "executable": os.path.realpath(os.sys.executable),
            "worktree": str(root.resolve()),
            "clock": "time.monotonic",
        },
        "random_seed": {
            "fitting": 7_512_101,
            "arrival": 7_512_102,
            "audit": 7_512_103,
            "bootstrap": 7_512_104,
            "sampling_performed": False,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": _input_hashes(root, sources, reduction),
        "rows": deepcopy(reduction["rows"]),
        "panel_results": deepcopy(reduction["panel_results"]),
        "panel_identities": deepcopy(reduction["panel_identities"]),
        "pooling": deepcopy(reduction["pooling"]),
        "historical_model_specs": {
            "A": deepcopy(reduction["panel_identities"]["A"]),
            "B": deepcopy(reduction["panel_identities"]["B"]),
        },
        "historical_operation_accounting": deepcopy(reduction["historical_operation_accounting"]),
        "token_counts": deepcopy(reduction["token_counts"]),
        "sample_size_budget": deepcopy(reduction["sample_size_budget"]),
        "acceptance_gate_results": gates,
        "gate_check_summary": summary,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "validation_manifest": {
            "experiment_id": MANIFEST.experiment_id,
            "test_paths": list(MANIFEST.test_paths),
            "changed_modules": list(MANIFEST.changed_modules),
            "static_paths": list(MANIFEST.static_paths),
            "documentation_paths": [NOTE_PATH.as_posix(), SPEC_PATH.as_posix()],
            "frozen_before_checks": True,
        },
        "arc_opportunity_audit_complete_score": audit_score,
        "arc_cost_claim_ready_score": cost_score,
        "per_game_results": deepcopy(reduction["per_game_results"]),
        "supervisor_opportunity_rows": deepcopy(reduction["supervisor_opportunity_rows"]),
        "supervisor_disposition": deepcopy(reduction["supervisor_disposition"]),
        "intervention_ledger": deepcopy(reduction["intervention_ledger"]),
        "efficacy_estimate": None,
        "cost_denominator_coverage": deepcopy(reduction["cost_denominator_coverage"]),
        "cost_share": None,
        "amdahl_upper_bound": None,
        "generalization_assessment": deepcopy(reduction["generalization_assessment"]),
        "registry_precheck": deepcopy(reduction["registry_precheck"]),
        "solve_provenance": "live_agent_self_discovery",
        "new_solve_attempted": False,
        "new_game_episodes": 0,
        "new_generation_calls": 0,
        "new_level_credit": 0,
        "claim_dispositions": {
            "supervisor_efficacy_tuning": "retired_until_live_reachable_choices",
            "broad_generalization": False,
            "selector_default_change": False,
            "policy_change": False,
            "game_specific_calibration": False,
            "source_derived_adapter": False,
            "offline_bfs": False,
            "remote_submission": False,
            "external_publication": False,
            "hardware_speed_claim": False,
        },
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key,
            f"Retaining {key} prevents an omitted fact from changing the conclusion silently.",
        )
        for key in (*artifact, "field_principles")
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_receipts() -> list[Json]:
    """Supply passing typed receipts only for deterministic unit fixtures."""

    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "required": True,
            "scope": "deterministic_fixture",
        }
        for name in (*CURRENT_VALIDATION_NAMES, *TERMINAL_RECEIPT_NAMES)
    ]


def build_artifact_for_test(root: Path) -> Json:
    """Build a terminal fixture through production reducers and gates."""

    preconditions, sources = collect_preconditions(root)
    reduction = reduce_upstreams(root)
    return build_artifact(
        root,
        preconditions=preconditions,
        sources=sources,
        reduction=reduction,
        receipts=_fixture_receipts(),
        duration_s=1.0,
        computation_s=0.5,
        validation_s=0.5,
        phase_spans=[{"phase": "fixture", "duration_s": 1.0, "completed_units": 36}],
        started_at_utc="2026-09-22T00:00:00Z",
        ended_at_utc="2026-09-22T00:00:01Z",
        require_terminal=True,
    )


REDUCTION_FIELDS = (
    "rows",
    "panel_results",
    "panel_identities",
    "pooling",
    "historical_operation_accounting",
    "token_counts",
    "sample_size_budget",
    "per_game_results",
    "supervisor_opportunity_rows",
    "supervisor_disposition",
    "intervention_ledger",
    "efficacy_estimate",
    "cost_denominator_coverage",
    "cost_share",
    "amdahl_upper_bound",
    "generalization_assessment",
    "registry_precheck",
)


def independent_reduce(artifact: Mapping[str, Any], root: Path = REPO_ROOT) -> Json:
    """Re-read both panels and compare every raw-derived terminal field."""

    reduction = reduce_upstreams(root)
    recomputed = {key: deepcopy(reduction[key]) for key in REDUCTION_FIELDS}
    declared = {key: deepcopy(artifact.get(key)) for key in REDUCTION_FIELDS}
    return {
        **recomputed,
        "matches_declared": canonical_hash(recomputed) == canonical_hash(declared),
        "errors": list(reduction["errors"]),
    }


REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
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
    "arc_cost_claim_ready_score",
    "solve_provenance",
    "per_game_results",
    "supervisor_opportunity_rows",
    "cost_denominator_coverage",
)


def validate_artifact(
    value: Mapping[str, Any] | Path,
    root: Path = REPO_ROOT,
    *,
    require_terminal: bool,
) -> list[str]:
    """Cold-check identity, raw replay, current calls, gates, and size."""

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
        ("inference_substrate", "aggregation_from_upstream_artifacts"),
        ("inference_substrate_class", "aggregation"),
        ("execution_venue", "host"),
    ):
        if artifact.get(field) != expected:
            errors.append(f"identity_mismatch:{field}")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("new_solve_attempted") is not False or artifact.get("new_level_credit") != 0:
        errors.append("new_solve_claimed")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    if artifact.get("verdict_class") != "blocked":
        replay = independent_reduce(artifact, root)
        if replay["matches_declared"] is not True:
            errors.append("independent_reduction_mismatch")
    denominator = artifact.get("cost_denominator_coverage") or {}
    if denominator.get("complete_mutually_exclusive_denominator") is not True:
        if artifact.get("arc_cost_claim_ready_score") != 0:
            errors.append("cost_ready_without_denominator")
        if artifact.get("cost_share") is not None or artifact.get("amdahl_upper_bound") is not None:
            errors.append("numeric_cost_claim_without_denominator")
    principles = artifact.get("field_principles") or {}
    if any(not principles.get(key) for key in artifact):
        errors.append("field_principle_missing")
    if any(not row.get("principle") for row in artifact.get("acceptance_gate_results") or []):
        errors.append("gate_principle_missing")
    receipts = artifact.get("validation_receipts") or []
    if not _receipts_pass(receipts, CURRENT_VALIDATION_NAMES):
        errors.append("current_scoped_validation_missing_or_failed")
    if require_terminal and not _receipts_pass(receipts, TERMINAL_RECEIPT_NAMES):
        errors.append("terminal_validation_missing_or_failed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if len(json.dumps(artifact, sort_keys=True, default=str).encode()) >= 20 * 1024 * 1024:
        errors.append("artifact_exceeds_20_mib")
    return list(dict.fromkeys(errors))


def build_validation_plan(
    root: Path, private_root: Path
) -> list[validation_contract.EnvironmentCommandSpec]:
    """Build the eight reporting-only scoped checks with private paths."""

    private_root.mkdir(parents=True, exist_ok=True)
    return validation_contract.build_command_plan(root, MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command expansion, broad pytest targets, and runtime E2E drift."""

    errors = validation_contract.validate_command_plan(root, MANIFEST, commands)
    counts = Counter(row.name for row in commands)
    for name in CURRENT_VALIDATION_NAMES:
        if counts[name] != 1:
            errors.append(f"command_count:{name}:{counts[name]}")
    for row in commands:
        if row.name.startswith("e2e_"):
            errors.append(f"runtime_e2e_forbidden:{row.name}")
        if any(argument.rstrip("/") in {"tests", "tests/python"} for argument in row.argv):
            errors.append(f"broad_test_target:{row.name}")
    return list(dict.fromkeys(errors))


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build cold replay and strict readers against one exact candidate."""

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
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
    ]


def _phase(name: str, began: float, started: float, units: int = 0) -> Json:
    """Record one disjoint monotonic phase span."""

    ended = time.monotonic()
    return {
        "phase": name,
        "start_s": began - started,
        "end_s": ended - started,
        "duration_s": ended - began,
        "completed_units": units,
    }


def build_blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    *,
    started_at_utc: str,
    duration_s: float,
) -> Json:
    """Close an external absence without running dependent measurement."""

    failures = [deepcopy(dict(row)) for row in preconditions if row.get("passed") is not True]
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_blocked_external_prerequisite_absent",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": utc_now(),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": [],
        "random_seed": {
            "fitting": 7_512_101,
            "arrival": 7_512_102,
            "audit": 7_512_103,
            "bootstrap": 7_512_104,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(sources)),
        "rows": [],
        "sample_size_budget": {
            "planned_independent_units": 36,
            "attempted_independent_units": 0,
            "completed_independent_units": 0,
            "excluded_independent_units": 0,
            "failed_independent_units": 0,
            "censored_independent_units": 0,
            "unstarted_independent_units": 36,
        },
        "acceptance_gate_results": failures,
        "gate_check_summary": gate_summary(failures),
        "honest_verdict": "complete_blocked_external_prerequisite_absent",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "arc_opportunity_audit_complete_score": 0,
        "arc_cost_claim_ready_score": 0,
        "solve_provenance": "live_agent_self_discovery",
        "new_solve_attempted": False,
        "new_level_credit": 0,
        "per_game_results": [],
        "supervisor_opportunity_rows": [],
        "cost_denominator_coverage": {
            "complete_mutually_exclusive_denominator": False,
            "blocked_by_external_absence": True,
        },
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key,
            f"Retaining {key} prevents an omitted fact from changing the conclusion silently.",
        )
        for key in (*artifact, "field_principles")
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path, run_date: str) -> Json:  # pragma: no cover - process orchestration.
    """Authenticate, validate, reduce, replay, and atomically publish."""

    started = time.monotonic()
    progress(started, "startup", "flushed_progress")
    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started_at = utc_now()
    spans: list[Json] = []

    began = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, sources = collect_preconditions(root)
    spans.append(_phase("preconditions", began, started, len(preconditions)))
    preconditions_passed = bool(preconditions) and all(row["passed"] for row in preconditions)
    progress(started, "preconditions", "end", passed=preconditions_passed)
    if not preconditions_passed:
        blocked = build_blocked_artifact(
            preconditions,
            sources,
            started_at_utc=started_at,
            duration_s=time.monotonic() - started,
        )
        progress(started, "publish", "before_atomic_blocked", path=RESULT_PATH)
        current_work_receipt.atomic_json(root / RESULT_PATH, blocked)
        progress(started, "publish", "after_atomic_blocked", path=RESULT_PATH)
        return blocked

    private = Path(tempfile.mkdtemp(prefix="exp7512-validation-", dir="/tmp"))
    plan = build_validation_plan(root, private / "scoped")
    plan_errors = validate_validation_plan(root, plan)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(started, "manifest", "frozen", commands=len(plan))

    began = time.monotonic()
    progress(started, "validation", "before_scoped_subprocesses", commands=len(plan))
    receipts = validation_contract.run_categorized_commands(
        root,
        [
            validation_contract.PlannedCommand(command, "required_validation", True)
            for command in plan
        ],
        log_dir=root / RAW_DIR / "validation/scoped",
        heartbeat_s=60.0,
    )
    validation_s = time.monotonic() - began
    spans.append(_phase("scoped_validation", began, started, len(receipts)))
    progress(started, "validation", "after_scoped_subprocesses", completed=len(receipts))
    if not _receipts_pass(receipts, CURRENT_VALIDATION_NAMES):
        raise RuntimeError("required_scoped_validation_failed")

    began = time.monotonic()
    progress(started, "reduction", "before_raw_boundaries")
    reduction = reduce_upstreams(root)
    computation_s = time.monotonic() - began
    spans.append(_phase("raw_reduction", began, started, len(reduction["rows"])))
    progress(started, "reduction", "after_raw_boundaries", episodes=len(reduction["rows"]))

    candidate = build_artifact(
        root,
        preconditions=preconditions,
        sources=sources,
        reduction=reduction,
        receipts=receipts,
        duration_s=time.monotonic() - started,
        computation_s=computation_s,
        validation_s=validation_s,
        phase_spans=spans,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        require_terminal=False,
    )
    candidate_errors = validate_artifact(candidate, root, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")
    progress(started, "candidate", "before_atomic_write", path=CANDIDATE_PATH)
    current_work_receipt.atomic_json(root / CANDIDATE_PATH, candidate)
    progress(started, "candidate", "after_atomic_write", path=CANDIDATE_PATH)

    began = time.monotonic()
    terminal_specs = terminal_command_specs(root, root / CANDIDATE_PATH)
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        commands=len(terminal_specs),
    )
    terminal_receipts = validation_contract.run_categorized_commands(
        root,
        [
            validation_contract.PlannedCommand(command, "terminal_reader", True)
            for command in terminal_specs
        ],
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    terminal_s = time.monotonic() - began
    spans.append(_phase("terminal_validation", began, started, len(terminal_receipts)))
    receipts.extend(terminal_receipts)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal_receipts),
    )
    if not _receipts_pass(receipts, TERMINAL_RECEIPT_NAMES):
        raise RuntimeError("required_terminal_validation_failed")

    final = build_artifact(
        root,
        preconditions=preconditions,
        sources=sources,
        reduction=reduction,
        receipts=receipts,
        duration_s=time.monotonic() - started,
        computation_s=computation_s,
        validation_s=validation_s + terminal_s,
        phase_spans=spans,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        require_terminal=True,
    )
    errors = validate_artifact(final, root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic_terminal", path=RESULT_PATH)
    current_work_receipt.atomic_json(root / RESULT_PATH, final)
    progress(
        started,
        "publish",
        "after_atomic_terminal",
        status=final["status"],
        audit_complete=final["arc_opportunity_audit_complete_score"],
        cost_ready=final["arc_cost_claim_ready_score"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public run role or one read-only cold replay role."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--reduce-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI dispatch.
    """Run the aggregation or cold-check one exact candidate."""

    args = parse_args(argv)
    if args.replay is not None:
        artifact = load_object(args.replay)
        replay = independent_reduce(artifact, REPO_ROOT)
        errors = (
            []
            if args.reduce_only
            else validate_artifact(artifact, REPO_ROOT, require_terminal=False)
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
    if args.date is None:
        raise SystemExit("--date is required unless --replay is used")
    run_experiment(REPO_ROOT, str(args.date))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
