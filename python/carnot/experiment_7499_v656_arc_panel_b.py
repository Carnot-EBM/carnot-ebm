"""Complete the frozen V656 ARC Panel B observation.

The live work stays in the qualified Experiment 7471 driver that Experiment
7485 used. This module supplies Panel B through that driver's schedule input,
then adds the Panel B reduction and comparability contract.

Spec refs: REQ-ARC-WMTE-7499 and SCENARIO-ARC-WMTE-7499-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from carnot import experiment_7471_v654_arc_seam_observation as live_driver
from carnot import experiment_7478_v655_arc_interval_protocol as interval_protocol
from carnot import experiment_7485_v655_arc_cost_panel_a as panel_a
from carnot.agentic.arc_inference_boundary import InvocationBoundaryLedger
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.656"
EXPERIMENT_ID = "exp7499-arc-panel-b"
TASK_ID = "experiment_7499_v656_arc_panel_b"
SCHEMA = "carnot.exp7499.v656.arc_panel_b.v1"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
EXECUTION_VENUE = "host"
PANEL_GAMES = ("tu93", "g50t", "tn36", "vc33", "re86", "dc22")
EPISODE_SEEDS = (65_501, 65_502, 65_503)
ACTION_LIMIT = 180
REQUEST_LIMIT = 2
MAX_NEW_TOKENS = 256
EPISODE_LIMIT_S = 240.0
PANEL_LIVE_LIMIT_S = 3600.0

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
PROTOCOL_PATH = Path("results/experiment_7478_v655_arc_interval_protocol.json")
PANEL_A_PATH = Path("results/experiment_7485_v655_arc_cost_panel_a.json")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
RESULT_PATH = Path("results/experiment_7499_v656_arc_panel_b.json")
RAW_DIR = Path("results/raw/experiment_7499_v656_arc_panel_b")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "current_invocation_events.jsonl"
RUNTIME_EVENT_PATH = RAW_DIR / "runtime_events.jsonl"
ACTION_PATH = RAW_DIR / "live_action_rows.jsonl"
CALLBACK_SEAM_PATH = RAW_DIR / "callback_seam_events.jsonl"
NORMALIZED_DIR = RAW_DIR / "normalized_intervals"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7499_v656_arc_panel_b.json")
CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7499_v656_arc_panel_b.py")
PANEL_A_MODULE_PATH = Path("python/carnot/experiment_7485_v655_arc_cost_panel_a.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7499_v656_arc_panel_b.py")
TEST_PATH = Path("tests/python/test_experiment_7499_v656_arc_panel_b.py")

REQUIRED_TERMINAL_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
CAPABILITY_E2E_NAMES = ("e2e_009", "e2e_010", "e2e_011", "private_arc_smoke")
TERMINAL_DISPOSITIONS = panel_a.TERMINAL_DISPOSITIONS

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
    "arc_panel_b_complete_score",
    "solve_provenance",
    "per_game_results",
    "panel_comparability",
    "supervisor_opportunity_rows",
    "arc_support_score",
)

FIELD_PRINCIPLES = {
    "schema": "Versioned schema, exact experiment_id, milestone and terminal status prevent reader drift.",
    "run_date": "Use 20260921; retain measured UTC and monotonic clock and process identity.",
    "preconditions_checked": "Record exact resource paths, observed values, ownership and input validity.",
    "MODEL_SPECS": "Current LLM work names unsloth/Qwen3.8-27B-GGUF with resolved file, hash, quantization and runtime.",
    "model_specs": "The lowercase mirror prevents readers from losing the authenticated model identity.",
    "model_invoked": "Attempted live loads and generations remain distinct from historical or scripted events.",
    "invocation_counts": "Attempted, completed, failed, cancelled and in-flight calls must reconcile.",
    "inference_substrate": "The named native runtime prevents cached evidence from posing as current execution.",
    "inference_substrate_class": "The declared bounded-generation class prevents duration-floor ambiguity and padding.",
    "execution_venue": "Host CPU and owned CUDA identity remain distinct from archived board evidence.",
    "duration_s": "Measured inference, episode and validation time prevents a synthetic duration floor.",
    "phase_spans": "Flushed boundaries and checkpoints expose unfinished work and stalls.",
    "random_seed": "Frozen role, audit, order and interval seeds prevent outcome-driven resampling.",
    "reproducibility_checksum": "Code, model, prompts, roles, shards and validation scope remain hash-bound.",
    "source_artifact_hashes": "Original upstream bytes, verdicts and flags cannot be laundered.",
    "rows": "Per-unit failures, censoring and missingness permit independent headline reduction.",
    "sample_size_budget": "Planned, attempted, complete, failed, excluded, censored and unstarted units stay separate.",
    "acceptance_gate_results": "Every exact validity, support and benefit comparison states the failure it prevents.",
    "gate_check_summary": "Blocked verdicts name the exact upstream field, expected value and observation.",
    "honest_verdict": "A complete terminal finding cannot hide an actual blocked conductor record.",
    "verdict_class": "The closed enum prevents external absence from becoming partial owned work.",
    "verifier_is_oracle": "An evaluation oracle forbids a positive verifier-value verdict.",
    "flagged_adversarial": "Actual reader flags remain visible and cannot be cleared to open a gate.",
    "validation_receipts": "Commands, scope, exits and log hashes make required checks reviewable.",
    "field_principles": "Every emitted field explains the failure mode that it prevents.",
    "arc_panel_b_complete_score": "Complete valid scheduled episode dispositions are not a progress or speed gate.",
    "solve_provenance": "Every level claim uses live_agent_self_discovery; other paths receive no credit.",
    "per_game_results": "All eighteen game-seed units expose cost, progress, censoring and missingness.",
    "panel_comparability": "Actual code, model, protocol and build identity decides whether panels can pool.",
    "supervisor_opportunity_rows": "Selected arms and eligible opportunities remain visible even when empty.",
    "arc_support_score": "Completed episode and game support stays distinct from a null progress result.",
}

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
    ROADMAP_PATH,
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7478_v655_arc_interval_protocol.py"),
    PANEL_A_MODULE_PATH,
    Path("python/carnot/experiment_7471_v654_arc_seam_observation.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_decision_telemetry.py"),
    REGISTRY_PATH,
    Path("docs/research-notes/semif-ebm-arc-experiment-plan-2026-09-20.md"),
    SPEC_PATH,
    PROTOCOL_PATH,
    PANEL_A_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def utc_now() -> str:
    """Return one UTC timestamp for a durable experiment boundary."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush one phase or long-operation boundary to the conductor."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7499] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed for missing or malformed bytes."""

    return panel_a.load_object(path)


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    field: str,
    principle: str,
    op: str = "==",
) -> JsonDict:
    """Attach one failure-prevention principle to an exact comparison."""

    return panel_a._gate(
        check,
        category,
        expected,
        observed,
        upstream=upstream,
        field=field,
        principle=principle,
        op=op,
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep required failures separate from observed scientific shortfalls."""

    return panel_a._gate_summary(gates)


def _source_record(path: Path, role: str, flags: Mapping[str, Any] | None = None) -> JsonDict:
    """Bind input bytes while preserving upstream flags separately."""

    return panel_a._source_record(path, role, flags)


def build_panel_schedule() -> list[JsonDict]:
    """Select Panel B from the frozen protocol through its explicit panel label."""

    manifest = interval_protocol.build_arc_schedule_manifest()
    rows = [deepcopy(row) for row in manifest["rows"] if row.get("panel") == "B"]
    return [
        {
            **row,
            "execution_order": index,
            "adapter_disabled": True,
            "stored_engines_disabled": True,
            "banked_trajectories_disabled": True,
            "cross_game_state_disabled": True,
            "game_source_disabled": True,
            "injected_action_recipes_disabled": True,
            "outer_loop_re_disabled": True,
            "offline_ground_truth_bfs_disabled": True,
        }
        for index, row in enumerate(rows)
    ]


def _registry_rows(root: Path) -> JsonDict:
    """Read prior public credit for every game without exposing it to the policy."""

    value = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
    indexed = {
        str(row.get("game")): row
        for row in value.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    return {
        "rows": [
            {
                "game": game,
                "registered": game in indexed,
                "levels_reproduced": indexed.get(game, {}).get("levels_reproduced"),
                "full_game_clear": indexed.get(game, {}).get("full_game_clear"),
            }
            for game in PANEL_GAMES
        ],
        "policy_received_registry_data": False,
        "new_credit_allowed": False,
    }


def collect_preconditions(
    root: Path, *, force_live: str | None = None
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict, JsonDict]:
    """Authenticate protocol, Panel A, external absence, games and registry."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _gate(
                f"source_bytes:{relative.as_posix()}",
                "validity",
                True,
                available,
                upstream=relative.as_posix(),
                field="bytes",
                principle="Missing source bytes would make current evidence non-replayable.",
            )
        )
        if available:
            hashes[relative.as_posix()] = _source_record(path, "current_input")

    protocol = load_object(root / PROTOCOL_PATH)
    panel_a_artifact = load_object(root / PANEL_A_PATH)
    protocol_flags = {
        key: protocol.get(key)
        for key in (
            "status",
            "honest_verdict",
            "verdict_class",
            "flagged_adversarial",
            "arc_interval_protocol_ready_score",
        )
    }
    hashes[PROTOCOL_PATH.as_posix()] = _source_record(
        root / PROTOCOL_PATH, "structured_prerequisite", protocol_flags
    )
    panel_a_flags = {
        key: panel_a_artifact.get(key)
        for key in (
            "status",
            "honest_verdict",
            "verdict_class",
            "flagged_adversarial",
            "arc_panel_a_complete_score",
        )
    }
    hashes[PANEL_A_PATH.as_posix()] = _source_record(
        root / PANEL_A_PATH, "immutable_panel_a", panel_a_flags
    )
    for field, expected in (
        ("arc_interval_protocol_ready_score", 1),
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
    ):
        checks.append(
            _gate(
                f"exp7478.{field}",
                "validity",
                expected,
                protocol.get(field),
                upstream=PROTOCOL_PATH.as_posix(),
                field=field,
                principle="Panel B requires the exact qualified and unflagged interval protocol.",
            )
        )
    schedule_manifest = protocol.get("arc_schedule_manifest") or {}
    checks.extend(
        [
            _gate(
                "exp7478.panel_b_roster",
                "validity",
                list(PANEL_GAMES),
                schedule_manifest.get("panel_b_games"),
                upstream=PROTOCOL_PATH.as_posix(),
                field="arc_schedule_manifest.panel_b_games",
                principle="A changed roster would turn held-out measurement into outcome selection.",
            ),
            _gate(
                "exp7478.observer_parity",
                "validity",
                True,
                (protocol.get("observer_parity") or {}).get("passed"),
                upstream=PROTOCOL_PATH.as_posix(),
                field="observer_parity.passed",
                principle="A measurement instrument must not change the policy that it measures.",
            ),
            _gate(
                "exp7478.raw_timing_fields",
                "validity",
                True,
                bool(protocol.get("interval_rows"))
                and all(
                    {
                        "episode_start_ns",
                        "episode_end_ns",
                        "observed_episode_ns",
                        "replaceable_lower_ns",
                        "replaceable_upper_ns",
                    }.issubset(row)
                    for row in protocol.get("interval_rows") or []
                )
                and (protocol.get("timing_correction") or {}).get(
                    "future_protocol_field_sufficiency"
                )
                is True,
                upstream=PROTOCOL_PATH.as_posix(),
                field="interval_rows|timing_correction.future_protocol_field_sufficiency",
                principle="Declared readiness cannot replace actual raw clock boundaries and bounds.",
            ),
            _gate(
                "exp7485.panel_a_complete",
                "validity",
                1,
                panel_a_artifact.get("arc_panel_a_complete_score"),
                upstream=PANEL_A_PATH.as_posix(),
                field="arc_panel_a_complete_score",
                principle="Panel comparison must preserve the actual completed Panel A stratum.",
            ),
        ]
    )
    old_producer_paths = sorted((root / "results").glob("experiment_7486*.json"))
    checks.append(
        _gate(
            "exp7486_external_absence",
            "validity",
            "absent",
            "absent" if not old_producer_paths else [path.name for path in old_producer_paths],
            upstream="openspec/change-proposals/research-roadmap-vNEXT.md",
            field="Exp7486 producer artifact",
            principle="External absence cannot be converted into a scientific null or completed row.",
        )
    )
    checks.append(
        _gate(
            "force_live",
            "validity",
            "1",
            os.environ.get("CARNOT_FORCE_LIVE") if force_live is None else force_live,
            upstream="environment",
            field="CARNOT_FORCE_LIVE",
            principle="A live panel cannot silently substitute scripted or simulated events.",
        )
    )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _gate(
            "driving_requirement",
            "validity",
            True,
            "REQ-ARC-WMTE-7499" in spec,
            upstream=SPEC_PATH.as_posix(),
            field="REQ-ARC-WMTE-7499",
            principle="Implementation without its requirement would bypass spec-first review.",
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.append(
        _gate(
            "task_not_excluded",
            "validity",
            False,
            "7499" in exclusion or EXPERIMENT_ID in exclusion,
            upstream="ops/exclusion_manifest.yaml",
            field=EXPERIMENT_ID,
            principle="A quarantined scope must not publish fresh scientific evidence.",
        )
    )
    available = set(live_driver.accessible_games(root))
    checks.append(
        _gate(
            "exact_panel_games_available",
            "validity",
            [],
            [game for game in PANEL_GAMES if game not in available],
            upstream="environment_files",
            field="panel_b_games",
            principle="Unavailable games must remain explicit and cannot be outcome-replaced.",
        )
    )
    registry = _registry_rows(root)
    checks.append(
        _gate(
            "registry_levels_reproduced_typed",
            "validity",
            True,
            all(
                isinstance(row.get("levels_reproduced"), int)
                and not isinstance(row.get("levels_reproduced"), bool)
                for row in registry["rows"]
            ),
            upstream=REGISTRY_PATH.as_posix(),
            field="games[].levels_reproduced",
            principle="Missing prior credit could let a banked public level pose as a new solve.",
        )
    )
    return checks, hashes, protocol, panel_a_artifact, registry


def _supervisor_rows(
    schedule: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Reduce every supervisor opportunity, including explicit zero rows."""

    output: list[JsonDict] = []
    for sealed in schedule:
        episode_id = str(sealed["episode_id"])
        rows = [
            row
            for row in events
            if str(row.get("episode_id")) == episode_id
            and row.get("seam") == "supervisor_arm_selection"
        ]
        opportunity_ids = {
            str(row.get("decision_id"))
            for row in rows
            if row.get("event") in {"stage_start", "eligible_candidate_set", "selection"}
        }
        eligible = sorted(
            {
                str(arm)
                for row in rows
                if row.get("event") == "eligible_candidate_set"
                for arm in row.get("candidate_ids") or []
            }
        )
        selected = sorted(
            {
                str(arm)
                for row in rows
                if row.get("event") == "selection"
                for arm in row.get("selected_candidate_ids") or []
            }
        )
        abstentions = sum(
            row.get("event") == "selection"
            and (
                not row.get("selected_candidate_ids")
                or row.get("selected_candidate_ids") == ["no_redirect"]
            )
            for row in rows
        )
        output.append(
            {
                "episode_id": episode_id,
                "game": sealed.get("game"),
                "seed": sealed.get("seed"),
                "opportunity_count": len(opportunity_ids),
                "eligible_arms": eligible,
                "selected_arms": selected,
                "abstention_count": abstentions,
                "supervisor_firing_count": sum(
                    row.get("event") == "selection" and row.get("supervisor_fired") is True
                    for row in rows
                ),
                "applied_redirection_count": sum(
                    row.get("event") == "selection" and row.get("applied_redirection") is True
                    for row in rows
                ),
                "no_selector_change": True,
            }
        )
    return output


def reduce_panel(
    schedule: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    seam_events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reuse the qualified interval reducer and add Panel B opportunity rows."""

    original_games = panel_a.PANEL_GAMES
    panel_a.PANEL_GAMES = PANEL_GAMES
    try:
        reduced = panel_a.reduce_panel(schedule, episode_rows, seam_events)
    finally:
        panel_a.PANEL_GAMES = original_games
    reduced["supervisor_opportunity_rows"] = _supervisor_rows(schedule, seam_events)
    return reduced


def _compact_reduction(reduced: Mapping[str, Any]) -> JsonDict:
    """Remove repeated raw action and stage bodies while retaining their hashes."""

    compacted = panel_a.compact_reduction(reduced)
    compacted["supervisor_opportunity_rows"] = deepcopy(
        list(reduced.get("supervisor_opportunity_rows") or [])
    )
    return compacted


def _model_identity(model_specs: Sequence[Mapping[str, Any]]) -> str:
    """Hash only model and runtime fields that decide panel compatibility."""

    selected = [
        {
            key: row.get(key)
            for key in (
                "hf_id",
                "model_filename",
                "sha256",
                "quantization",
                "revision",
                "runtime_settings",
                "flags",
            )
        }
        for row in model_specs
    ]
    return panel_a.canonical_hash(selected)


def _source_hash_by_role(
    artifact: Mapping[str, Any], *, suffix: str = "", role: str = ""
) -> str | None:
    """Find one recorded source identity without trusting its absolute path."""

    for key, row in (artifact.get("source_artifact_hashes") or {}).items():
        if not isinstance(row, Mapping):
            continue
        if suffix and not str(key).endswith(suffix):
            continue
        if role and row.get("role") != role:
            continue
        value = row.get("sha256")
        return str(value) if isinstance(value, str) else None
    return None


def _panel_a_identities(panel_a_artifact: Mapping[str, Any]) -> JsonDict:
    """Recover identities that Panel A recorded at measurement time."""

    return {
        "panel_runner_code": _source_hash_by_role(
            panel_a_artifact, suffix=PANEL_A_MODULE_PATH.as_posix()
        ),
        "model": _model_identity(panel_a_artifact.get("model_specs") or []),
        "interval_protocol": _source_hash_by_role(
            panel_a_artifact, suffix=PROTOCOL_PATH.as_posix()
        ),
        "native_build": _source_hash_by_role(panel_a_artifact, role="native_runtime_binary"),
    }


def current_comparability_inputs(
    root: Path,
    model_specs: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Resolve current code, model, protocol and native runtime identities."""

    native = Path(str(runtime_receipt.get("native_binary") or ""))
    return {
        "panel_runner_code": current_work_receipt.sha256_file(root / PANEL_A_MODULE_PATH),
        "model": _model_identity(model_specs),
        "interval_protocol": current_work_receipt.sha256_file(root / PROTOCOL_PATH),
        "native_build": current_work_receipt.sha256_file(native) if native.is_file() else None,
    }


def compare_panel_a(
    panel_a_artifact: Mapping[str, Any], current_identities: Mapping[str, Any]
) -> JsonDict:
    """Allow pooling only when all four preregistered identities match."""

    previous = _panel_a_identities(panel_a_artifact)
    current = {key: current_identities.get(key) for key in previous}
    mismatches = [key for key in previous if previous[key] != current[key]]
    comparable = not mismatches and all(previous.values())
    return {
        "comparable": comparable,
        "pooling_allowed": comparable,
        "mismatched_identities": mismatches,
        "panel_a_identities": previous,
        "panel_b_identities": current,
        "stratification": "homogeneous_a_plus_b" if comparable else "panel_version_strata",
        "pooled_episode_count": 36 if comparable else None,
        "panel_a_rerun_authorized": False,
        "principle": "Actual build, model, code and protocol identity prevents a fabricated homogeneous 36-episode panel.",
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful, non-timeout receipt for every named command."""

    return panel_a._receipts_pass(receipts, names)


def _support_score(reduced: Mapping[str, Any]) -> JsonDict:
    """Count completed episode and game support without using progress."""

    rows = reduced.get("rows") or []
    complete_episodes = sum(row.get("disposition") == "complete" for row in rows)
    complete_games = sum(
        all(row.get("disposition") == "complete" for row in rows if row.get("game") == game)
        and sum(row.get("game") == game for row in rows) == len(EPISODE_SEEDS)
        for game in PANEL_GAMES
    )
    return {"complete_episodes": complete_episodes, "complete_games": complete_games}


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    registry_precheck: Mapping[str, Any],
    panel_a_artifact: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    seam_shards: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    require_terminal: bool,
) -> JsonDict:
    """Build one terminal record from current rows and immutable sidecars."""

    seam_events = panel_a._events_from_shards(seam_shards)
    reduced = _compact_reduction(reduce_panel(schedule, episode_rows, seam_events))
    invocation = panel_a._invocation_reduction(
        boundary_events, child_terminal=bool(runtime_receipt.get("child_terminal", True))
    )
    counts = invocation["invocation_counts"]
    balanced = all(
        counts[f"{prefix}_attempted"]
        == sum(
            counts[f"{prefix}_{state}"]
            for state in ("completed", "failed", "cancelled", "in_flight")
        )
        for prefix in ("model_loads", "forward_calls", "generation_calls")
    )
    load_count_valid = counts["model_loads_attempted"] == 1
    generation_count_valid = counts["generation_calls_attempted"] <= 36
    offload = bool((runtime_receipt.get("observed_cuda_offload") or {}).get("passed"))
    affected_ok = (
        _receipts_pass(validation_receipts, validation_scope.REQUIRED_CHECK_NAMES)
        if require_terminal
        else True
    )
    e2e_ok = _receipts_pass(validation_receipts, CAPABILITY_E2E_NAMES) if require_terminal else True
    terminal_ok = (
        _receipts_pass(validation_receipts, REQUIRED_TERMINAL_NAMES) if require_terminal else True
    )
    support = _support_score(reduced)
    all_terminal = (
        reduced["all_dispositions_present"]
        and reduced["sample_size_budget"]["unstarted_independent_units"] == 0
        and reduced["sample_size_budget"]["attempted_independent_units"] == 18
    )
    current_identities = current_comparability_inputs(REPO_ROOT, model_specs, runtime_receipt)
    comparability = compare_panel_a(panel_a_artifact, current_identities)
    validity = [
        _gate(
            "preconditions_checked",
            "validity",
            True,
            bool(preconditions) and all(row.get("passed") is True for row in preconditions),
            upstream="preconditions_checked",
            field="passed",
            principle="A favorable metric cannot excuse invalid evidence.",
        ),
        _gate(
            "exact_panel_schedule",
            "validity",
            True,
            panel_a.canonical_hash(schedule) == panel_a.canonical_hash(build_panel_schedule()),
            upstream="frozen_schedule",
            field="rows",
            principle="Outcome-driven substitution would invalidate held-out measurement.",
        ),
        _gate(
            "eighteen_terminal_dispositions",
            "validity",
            True,
            all_terminal,
            upstream="raw_episode_rows",
            field="disposition",
            principle="Missing work cannot be hidden as a completed scheduled episode.",
        ),
        _gate(
            "exclusive_interval_accounting",
            "validity",
            True,
            reduced["all_interval_bounds_valid"],
            upstream="exclusive_cost_rows",
            field="bounds_valid",
            principle="Duplicate or nested work cannot increase an interval union.",
        ),
        _gate(
            "invocation_accounting_balanced",
            "validity",
            True,
            balanced,
            upstream="current_invocation_events",
            field="invocation_counts",
            principle="Each attempted call must retain exactly one terminal or in-flight state.",
        ),
        _gate(
            "one_owned_model_load",
            "validity",
            True,
            load_count_valid,
            upstream="current_invocation_events",
            field="model_loads_attempted",
            principle="Repeated or absent loads would violate the one-owned-model design.",
        ),
        _gate(
            "bounded_generations",
            "validity",
            True,
            generation_count_valid,
            upstream="current_invocation_events",
            field="generation_calls_attempted",
            principle="An over-budget model can manufacture support through extra retries.",
        ),
        _gate(
            "actual_cuda_offload_receipt",
            "validity",
            True,
            offload,
            upstream="execution_venue_details.observed_cuda_offload",
            field="passed",
            principle="Requested layers do not prove actual owned CUDA placement.",
        ),
        _gate(
            "panel_comparability_evaluated",
            "validity",
            True,
            isinstance(comparability.get("pooling_allowed"), bool),
            upstream="panel_comparability",
            field="pooling_allowed",
            principle="A code or build mismatch must stratify panels instead of fabricating pooling.",
        ),
        _gate(
            "affected_validation",
            "validity",
            True,
            affected_ok,
            upstream="validation_receipts",
            field="affected_check_names",
            principle="A favorable metric cannot excuse failed scoped validation.",
        ),
        _gate(
            "capability_e2e",
            "validity",
            True,
            e2e_ok,
            upstream="validation_receipts",
            field="E2E-009|E2E-010|E2E-011|private_arc_smoke",
            principle="Unit fixtures cannot replace actual ARC policy and transport checks.",
        ),
        _gate(
            "terminal_readers",
            "validity",
            True,
            terminal_ok,
            upstream="validation_receipts",
            field="terminal_command_names",
            principle="Independent replay and strict readers must accept the exact candidate.",
        ),
    ]
    required_valid = all(row["passed"] for row in validity)
    readiness = [
        _gate(
            "panel_b_instrument_complete",
            "readiness",
            True,
            all_terminal
            and reduced["all_interval_bounds_valid"]
            and balanced
            and load_count_valid
            and generation_count_valid
            and offload
            and required_valid,
            upstream="rows|exclusive_cost_rows|current_invocation_events",
            field="arc_panel_b_complete_score",
            principle="A valid scientific null must not block independent completion measurement.",
        )
    ]
    opportunities = reduced["supervisor_opportunity_rows"]
    zero_opportunity_games = {
        row["game"] for row in opportunities if row.get("opportunity_count") == 0
    }
    progressed = [row for row in reduced["rows"] if row.get("actions_to_progress") is not None]
    retention = bool(progressed) and all(
        row.get("offline_reproduced") is True for row in progressed
    )
    benefit = [
        _gate(
            "panel_b_episode_support_floor",
            "scientific_benefit",
            18,
            support["complete_episodes"],
            upstream="rows",
            field="arc_support_score.complete_episodes",
            principle="A favorable seed or low-support result cannot replace held-out value.",
            op=">=",
        ),
        _gate(
            "panel_b_game_support_floor",
            "scientific_benefit",
            6,
            support["complete_games"],
            upstream="per_game_results",
            field="arc_support_score.complete_games",
            principle="Repeated seeds cannot replace independent game support.",
            op=">=",
        ),
        _gate(
            "zero_opportunity_effect",
            "scientific_benefit",
            18,
            sum(row.get("opportunity_count") == 0 for row in opportunities),
            upstream="supervisor_opportunity_rows",
            field="opportunity_count",
            principle="A favorable subset cannot establish that the zero-opportunity finding extends.",
            op=">=",
        ),
        _gate(
            "progress_retention",
            "scientific_benefit",
            True,
            retention,
            upstream="rows.trace_reproduction",
            field="offline_reproduced",
            principle="A timing finding cannot excuse lost or unreproduced progress.",
        ),
        _gate(
            "six_game_multiplicity",
            "scientific_benefit",
            6,
            len(zero_opportunity_games),
            upstream="supervisor_opportunity_rows",
            field="game",
            principle="One favorable game cannot replace a cross-game observation.",
            op=">=",
        ),
    ]
    gates = [*validity, *readiness, *benefit]
    required_ok = required_valid and all(row["passed"] for row in readiness)
    complete_score = int(required_ok)
    status = (
        "complete_null_live_arc_panel_b"
        if required_ok
        else "complete_disqualified_live_arc_panel_b"
    )
    reproduced_rows = [row for row in reduced["rows"] if row.get("offline_reproduced") is True]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "clock_identity": {"wall": "datetime.now(UTC)", "interval": "time.monotonic_ns"},
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": deepcopy(list(model_specs)) if load_count_valid else [MODEL_ID],
        "model_specs": deepcopy(list(model_specs)) if load_count_valid else [MODEL_ID],
        "model_invoked": bool(invocation["model_invoked"]),
        "invocation_counts": deepcopy(counts),
        "current_invocation_rows": deepcopy(invocation.get("call_rows") or []),
        "inference_substrate": invocation["inference_substrate"],
        "inference_substrate_class": invocation["inference_substrate_class"],
        "planned_inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "execution_venue_details": deepcopy(dict(runtime_receipt)),
        "duration_s": round(float(duration_s), 6),
        "duration_breakdown_s": {
            str(row.get("phase")): round(float(row.get("duration_s") or 0.0), 6)
            for row in phase_spans
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "selection": 7_478,
            "episodes": list(EPISODE_SEEDS),
            "ordering": 7_478,
            "interval": 7_499_091,
            "audit": 7_499_093,
            "deterministic_null": "No random imputation is used for empty opportunities.",
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "registry_precheck": deepcopy(dict(registry_precheck)),
        "protocol_schedule": deepcopy(list(schedule)),
        "rows": reduced["rows"],
        "sample_size_budget": reduced["sample_size_budget"],
        "per_game_results": reduced["per_game_results"],
        "exclusive_cost_rows": reduced["exclusive_cost_rows"],
        "supervisor_opportunity_rows": opportunities,
        "seam_event_shards": deepcopy(list(seam_shards)),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": status,
        "verdict_class": "null" if required_ok else "disqualified",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "validation_manifest": {
            "experiment_id": MANIFEST.experiment_id,
            "test_paths": list(MANIFEST.test_paths),
            "changed_modules": list(MANIFEST.changed_modules),
            "static_paths": list(MANIFEST.static_paths),
        },
        "validation_receipts": deepcopy(list(validation_receipts)),
        "arc_panel_b_complete_score": complete_score,
        "arc_support_score": support,
        "scientific_benefit_score": int(all(row["passed"] for row in benefit)),
        "panel_comparability": comparability,
        "pooled_estimate": (
            {
                "episode_count": 36,
                "strata": ["panel_a", "panel_b"],
                "interpretation": "pooling_identity_only_no_hidden_game_claim",
            }
            if comparability["pooling_allowed"]
            else None
        ),
        "solve_provenance": "live_agent_self_discovery",
        "offline_reproduced": bool(reproduced_rows),
        "reproduced_levels": sum(int(row.get("reproduced_levels") or 0) for row in reproduced_rows),
        "new_level_credit": 0,
        "generalization_scope": "public_adapter_withheld_proxy",
        "hidden_game_efficacy_claim": False,
        "selector_change": False,
        "panel_a_rerun": False,
        "exp7486_external_absence": True,
        "remote_submission": False,
        "scored_path_changed": False,
        "research_conductor_changed": False,
        "small_ebm_training": {"performed": False},
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key,
            "This typed field prevents an omitted fact from silently changing the conclusion.",
        )
        for key in artifact
    }
    artifact["field_principles"]["field_principles"] = FIELD_PRINCIPLES["field_principles"]
    artifact["reproducibility_checksum"] = panel_a.artifact_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute every Panel B row and opportunity from frozen sidecars."""

    reduced = _compact_reduction(
        reduce_panel(
            artifact.get("protocol_schedule") or [],
            artifact.get("rows") or [],
            panel_a._events_from_shards(artifact.get("seam_event_shards") or []),
        )
    )
    keys = (
        "rows",
        "sample_size_budget",
        "per_game_results",
        "exclusive_cost_rows",
        "supervisor_opportunity_rows",
    )
    declared = {key: artifact.get(key) for key in keys}
    recomputed = {key: reduced[key] for key in keys}
    return {
        **recomputed,
        "matches_declared": panel_a.canonical_hash(declared) == panel_a.canonical_hash(recomputed),
    }


def validate_artifact(value: Mapping[str, Any] | Path, *, require_terminal: bool) -> list[str]:
    """Cold-check identity, reductions, calls, principles and checksum."""

    artifact = load_object(value) if isinstance(value, Path) else dict(value)
    errors = [
        f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    ]
    for field, expected in (
        ("schema", SCHEMA),
        ("experiment_id", EXPERIMENT_ID),
        ("milestone", MILESTONE),
        ("run_date", RUN_DATE),
        ("generalization_scope", "public_adapter_withheld_proxy"),
        ("solve_provenance", "live_agent_self_discovery"),
    ):
        if artifact.get(field) != expected:
            errors.append(f"identity_mismatch:{field}")
    if independent_reduce(artifact)["matches_declared"] is not True:
        errors.append("independent_reduction_mismatch")
    counts = artifact.get("invocation_counts") or {}
    for prefix in ("model_loads", "forward_calls", "generation_calls"):
        attempted = int(counts.get(f"{prefix}_attempted") or 0)
        terminal = sum(
            int(counts.get(f"{prefix}_{state}") or 0)
            for state in ("completed", "failed", "cancelled", "in_flight")
        )
        if attempted != terminal:
            errors.append(f"unbalanced_invocations:{prefix}")
    if counts.get("model_loads_attempted") != 1:
        errors.append("model_load_count_invalid")
    if int(counts.get("generation_calls_attempted") or 0) > 36:
        errors.append("generation_budget_exceeded")
    specs = artifact.get("model_specs") or []
    if not any(
        row == MODEL_ID or isinstance(row, Mapping) and row.get("hf_id") == MODEL_ID
        for row in specs
    ):
        errors.append("required_model_missing")
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
    if artifact.get("new_level_credit") != 0:
        errors.append("registered_public_credit_nonzero")
    principles = artifact.get("field_principles") or {}
    if any(not principles.get(key) for key in artifact):
        errors.append("field_principle_missing")
    if any(not row.get("principle") for row in artifact.get("acceptance_gate_results") or []):
        errors.append("gate_principle_missing")
    if require_terminal and not _receipts_pass(
        artifact.get("validation_receipts") or [],
        (*validation_scope.REQUIRED_CHECK_NAMES, *CAPABILITY_E2E_NAMES, *REQUIRED_TERMINAL_NAMES),
    ):
        errors.append("required_validation_missing_or_failed")
    if len(json.dumps(artifact, sort_keys=True, default=str).encode()) >= 20 * 1024 * 1024:
        errors.append("artifact_exceeds_20_mib")
    if artifact.get("reproducibility_checksum") != panel_a.artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _fixture_receipts() -> list[JsonDict]:
    """Return typed passing receipts for the reducer-only fixture."""

    return [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (
            *validation_scope.REQUIRED_CHECK_NAMES,
            *CAPABILITY_E2E_NAMES,
            *REQUIRED_TERMINAL_NAMES,
        )
    ]


def build_artifact_for_test() -> JsonDict:
    """Build a terminal-shaped fixture through production reducers."""

    schedule = build_panel_schedule()
    episodes = []
    for sealed in schedule:
        episodes.append(
            {
                **deepcopy(sealed),
                "disposition": "complete",
                "action_count": 1,
                "start_level": 0,
                "peak_level": 0,
                "terminal_level": 0,
                "action_rows": [
                    {
                        "action_index": 1,
                        "level": 0,
                        "later_progress": False,
                        "interval_start_monotonic_ns": 1_000,
                        "interval_end_monotonic_ns": 2_000,
                    }
                ],
                "request_budget_receipt": {
                    "attempted": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "in_flight": 0,
                },
                "trace_reproduction": {"attempted": False, "passed": False},
                "solve_provenance": "no_level_reached",
                "elapsed_s": 0.000001,
                "new_level_credit": 0,
                "error": None,
            }
        )
    events = panel_a.normalize_interval_events(
        [
            {
                "episode_id": schedule[0]["episode_id"],
                "decision_id": "fixture-decision",
                "seam": "induction_timing",
                "event": "stage_start",
                "event_monotonic_ns": 1_100,
                "interval_start_monotonic_ns": 1_100,
            },
            {
                "episode_id": schedule[0]["episode_id"],
                "decision_id": "fixture-decision",
                "seam": "induction_timing",
                "event": "stage_end",
                "event_monotonic_ns": 1_200,
                "interval_end_monotonic_ns": 1_200,
            },
        ],
        owner_pid=1,
    )
    shard = {
        "episode_id": schedule[0]["episode_id"],
        "path": "inline_fixture",
        "sha256": panel_a.canonical_hash(events),
        "row_count": len(events),
        "inline_rows": events,
    }
    panel_a_artifact = load_object(REPO_ROOT / PANEL_A_PATH)
    model_specs = panel_a_artifact.get("model_specs") or [
        {"hf_id": MODEL_ID, "quantization": "Q4_K_M"}
    ]
    runtime = deepcopy(panel_a_artifact.get("execution_venue_details") or {})
    runtime["child_terminal"] = True
    runtime["observed_cuda_offload"] = {"passed": True, "owned_server_vram_mb": 1}
    return build_terminal_artifact(
        started_at_utc="2026-09-21T00:00:00Z",
        ended_at_utc="2026-09-21T00:01:00Z",
        duration_s=60.0,
        phase_spans=[{"phase": "fixture", "duration_s": 60.0, "completed_units": 18}],
        preconditions=[{"check": "fixture", "passed": True}],
        source_hashes={},
        registry_precheck={"rows": [], "new_credit_allowed": False},
        panel_a_artifact=panel_a_artifact,
        schedule=schedule,
        episode_rows=episodes,
        seam_shards=[shard],
        boundary_events=panel_a._fixture_boundary_events(),
        runtime_receipt=runtime,
        model_specs=model_specs,
        validation_receipts=_fixture_receipts(),
        require_terminal=False,
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze affected checks plus unchanged ARC capability checks."""

    private_root.mkdir(parents=True, exist_ok=True)
    affected = validation_contract.build_command_plan(root, MANIFEST, private_root / "affected")
    shared = interval_protocol.build_validation_plan(root, private_root / "shared")
    return [*affected, *(row for row in shared if row.name in CAPABILITY_E2E_NAMES)]


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command drift, duplicates and broad test targets."""

    affected = [row for row in commands if row.name in validation_scope.REQUIRED_CHECK_NAMES]
    errors = validation_contract.validate_command_plan(root, MANIFEST, affected)
    counts = Counter(row.name for row in commands)
    for name in (*validation_scope.REQUIRED_CHECK_NAMES, *CAPABILITY_E2E_NAMES):
        if counts[name] != 1:
            errors.append(f"command_count:{name}:{counts[name]}")
    for row in commands:
        if any(argument.rstrip("/") in {"tests", "tests/python"} for argument in row.argv):
            errors.append(f"broad_test_target:{row.name}")
    return errors


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build cold replay, independent reduction and strict reader commands."""

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


def _configure_live_driver() -> None:  # pragma: no cover - owned live child configuration.
    """Give the shared qualified driver an explicit Panel B configuration."""

    values = {
        "TASK_ID": TASK_ID,
        "RAW_DIR": RAW_DIR,
        "SCHEDULE_PATH": SCHEDULE_PATH,
        "SESSION_PATH": SESSION_PATH,
        "BOUNDARY_PATH": BOUNDARY_PATH,
        "RUNTIME_EVENT_PATH": RUNTIME_EVENT_PATH,
        "ACTION_PATH": ACTION_PATH,
        "CALLBACK_SEAM_PATH": CALLBACK_SEAM_PATH,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "RUN_DATE": RUN_DATE,
        "EPISODE_SEEDS": EPISODE_SEEDS,
        "ACTION_LIMIT": ACTION_LIMIT,
        "REQUEST_LIMIT": REQUEST_LIMIT,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "EPISODE_LIMIT_S": EPISODE_LIMIT_S,
        "AGGREGATE_LIVE_LIMIT_S": PANEL_LIVE_LIMIT_S,
    }
    for name, value in values.items():
        setattr(live_driver, name, value)


def _normalize_live_shards(
    root: Path, episodes: Sequence[Mapping[str, Any]], owner_pid: int
) -> list[JsonDict]:  # pragma: no cover - live evidence persistence.
    """Persist qualified intervals while keeping original shards immutable."""

    shards: list[JsonDict] = []
    source_shards = live_driver._seam_shards(root, episodes)
    callback = live_driver._write_callback_seam_shard(
        root, load_object(root / SESSION_PATH).get("runtime_receipt", {}).get("gpu_uuid")
    )
    if callback is not None:
        source_shards.append(callback)
    for index, source in enumerate(source_shards):
        raw = Path(str(source["path"]))
        raw = raw if raw.is_absolute() else root / raw
        rows = panel_a.normalize_interval_events(live_driver.read_jsonl(raw), owner_pid=owner_pid)
        for row in rows:
            row["run_id"] = EXPERIMENT_ID
        output = (
            root
            / NORMALIZED_DIR
            / f"{index:02d}_{str(source.get('episode_id')).replace(':', '__')}.jsonl"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        for row in rows:
            live_driver._append_jsonl(output, row)
        shards.append(
            {
                "episode_id": source.get("episode_id"),
                "path": output.relative_to(root).as_posix(),
                "sha256": current_work_receipt.sha256_file(output),
                "bytes": output.stat().st_size,
                "row_count": len(rows),
                "source_raw_sha256": source.get("sha256"),
            }
        )
    return shards


def _phase(
    spans: list[JsonDict], name: str, began: float, started: float, units: int
) -> None:  # pragma: no cover - measured orchestration boundary.
    """Append one real monotonic phase receipt."""

    ended = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_s": round(began - started, 6),
            "end_s": round(ended - started, 6),
            "duration_s": round(ended - began, 6),
            "completed_units": units,
            "ended_at_utc": utc_now(),
        }
    )


def _blocked_artifact(
    *,
    started_at: str,
    started: float,
    spans: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    registry: Mapping[str, Any],
    panel_a_artifact: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:  # pragma: no cover - fail-closed live branch.
    """Build an honest pre-model block without invented current calls."""

    zero = deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)
    artifact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked_panel_b_precondition",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "preconditions_checked": deepcopy(list(checks)),
        "MODEL_SPECS": [MODEL_ID],
        "model_specs": [MODEL_ID],
        "model_invoked": False,
        "invocation_counts": zero,
        "inference_substrate": "pre_model_validation_only",
        "inference_substrate_class": "no_model_load",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": round(time.monotonic() - started, 6),
        "phase_spans": deepcopy(list(spans)),
        "random_seed": {"episodes": list(EPISODE_SEEDS)},
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "rows": [panel_a._unstarted_row(row) for row in schedule],
        "sample_size_budget": {
            "planned_independent_units": 18,
            "attempted_independent_units": 0,
            "complete_independent_units": 0,
            "failed_independent_units": 0,
            "censored_independent_units": 0,
            "excluded_independent_units": 0,
            "unstarted_independent_units": 18,
            "independent_game_clusters": 6,
        },
        "acceptance_gate_results": deepcopy(list(checks)),
        "gate_check_summary": _gate_summary(checks),
        "honest_verdict": "blocked_panel_b_precondition",
        "verdict_class": "blocked",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "validation_receipts": deepcopy(list(receipts)),
        "arc_panel_b_complete_score": 0,
        "solve_provenance": "live_agent_self_discovery",
        "per_game_results": [],
        "panel_comparability": compare_panel_a(panel_a_artifact, {}),
        "supervisor_opportunity_rows": _supervisor_rows(schedule, []),
        "arc_support_score": {"complete_episodes": 0, "complete_games": 0},
        "registry_precheck": deepcopy(dict(registry)),
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key,
            "This typed field prevents an omitted fact from silently changing the conclusion.",
        )
        for key in artifact
    }
    artifact["field_principles"]["field_principles"] = FIELD_PRINCIPLES["field_principles"]
    artifact["reproducibility_checksum"] = panel_a.artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - orchestration.
    """Validate, run one owned live child, reduce, replay and publish."""

    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    receipts: list[JsonDict] = []
    progress(started, "startup", "begin", run_date=run_date)
    phase = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes, _protocol, panel_a_artifact, registry = collect_preconditions(root)
    checks.insert(
        0,
        _gate(
            "run_date",
            "validity",
            RUN_DATE,
            run_date,
            upstream="command_line",
            field="--date",
            principle="A wrong run date would detach evidence from its roadmap task.",
        ),
    )
    schedule = build_panel_schedule()
    current_work_receipt.atomic_json(
        root / SCHEDULE_PATH,
        {"rows": schedule, "registry_precheck": registry, "panel": "B"},
    )
    _phase(spans, "preconditions", phase, started, len(checks))
    progress(started, "preconditions", "after", passed=all(row["passed"] for row in checks))
    if not all(row["passed"] for row in checks):
        artifact = _blocked_artifact(
            started_at=started_at,
            started=started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            registry=registry,
            panel_a_artifact=panel_a_artifact,
            schedule=schedule,
            receipts=receipts,
        )
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        return artifact

    private = Path(tempfile.mkdtemp(prefix="exp7499-validation-", dir="/tmp"))
    phase = time.monotonic()
    progress(started, "validation", "before_affected_and_e2e_subprocesses")
    plan = build_validation_plan(root, private / "scoped")
    errors = validate_validation_plan(root, plan)
    if errors:
        raise RuntimeError(f"validation_plan_invalid:{errors}")
    receipts.extend(
        validation_contract.run_categorized_commands(
            root,
            [validation_contract.PlannedCommand(row, "required_validation", True) for row in plan],
            log_dir=root / RAW_DIR / "validation/affected_and_e2e",
            heartbeat_s=60.0,
        )
    )
    _phase(spans, "affected_and_capability_validation", phase, started, len(receipts))
    progress(
        started, "validation", "after_affected_and_e2e_subprocesses", completed_units=len(receipts)
    )
    if not _receipts_pass(
        receipts, (*validation_scope.REQUIRED_CHECK_NAMES, *CAPABILITY_E2E_NAMES)
    ):
        artifact = _blocked_artifact(
            started_at=started_at,
            started=started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            registry=registry,
            panel_a_artifact=panel_a_artifact,
            schedule=schedule,
            receipts=receipts,
        )
        artifact["status"] = "complete_disqualified_affected_validation"
        artifact["honest_verdict"] = artifact["status"]
        artifact["verdict_class"] = "disqualified"
        artifact["reproducibility_checksum"] = panel_a.artifact_checksum(artifact)
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        return artifact

    _configure_live_driver()
    phase = time.monotonic()
    progress(started, "runtime_preconditions", "before_cached_model_cuda_lease_checks")
    runtime_checks, runtime_hashes, resources = live_driver._runtime_preconditions(root, started)
    checks.extend(runtime_checks)
    hashes.update(runtime_hashes)
    _phase(spans, "runtime_preconditions", phase, started, len(runtime_checks))
    progress(
        started,
        "runtime_preconditions",
        "after_cached_model_cuda_lease_checks",
        passed=all(row.get("passed") is True for row in checks),
    )
    if not all(row.get("passed") is True for row in checks):
        artifact = _blocked_artifact(
            started_at=started_at,
            started=started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            registry=registry,
            panel_a_artifact=panel_a_artifact,
            schedule=schedule,
            receipts=receipts,
        )
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        return artifact

    for relative in (
        BOUNDARY_PATH,
        RUNTIME_EVENT_PATH,
        ACTION_PATH,
        CALLBACK_SEAM_PATH,
        SESSION_PATH,
        CHECKPOINT_PATH,
        CANDIDATE_PATH,
        RAW_DIR / "episode_rows.json",
    ):
        (root / relative).unlink(missing_ok=True)
    phase = time.monotonic()
    progress(started, "live", "before_model_load_generation_benchmark", planned_units=18)
    session = live_driver.run_child_with_lease(
        resources=resources, schedule_path=root / SCHEDULE_PATH, started=started
    )
    episodes = [dict(row) for row in session.get("episodes") or [] if isinstance(row, Mapping)]
    progress(
        started,
        "live",
        "after_model_load_generation_benchmark",
        completed_units=len(episodes),
    )
    _phase(spans, "live_model_and_episodes", phase, started, len(episodes))
    boundary_events = InvocationBoundaryLedger(root / BOUNDARY_PATH).read_events()
    seam_shards = _normalize_live_shards(
        root, episodes, int(session.get("child_pid") or os.getpid())
    )
    for relative, role in (
        (SCHEDULE_PATH, "frozen_protocol"),
        (BOUNDARY_PATH, "current_invocation_ledger"),
        (RUNTIME_EVENT_PATH, "request_event_shard"),
        (ACTION_PATH, "action_outcome_shard"),
    ):
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = _source_record(path, role)
    for shard in seam_shards:
        path = root / str(shard["path"])
        hashes[str(shard["path"])] = _source_record(path, "qualified_interval_shard")

    phase = time.monotonic()
    candidate = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=hashes,
        registry_precheck=registry,
        panel_a_artifact=panel_a_artifact,
        schedule=schedule,
        episode_rows=episodes,
        seam_shards=seam_shards,
        boundary_events=boundary_events,
        runtime_receipt=session.get("runtime_receipt") or {},
        model_specs=[resources["model_spec"]],
        validation_receipts=receipts,
        require_terminal=False,
    )
    candidate_errors = validate_artifact(candidate, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")
    current_work_receipt.atomic_json(root / CANDIDATE_PATH, candidate)
    _phase(spans, "independent_reduction", phase, started, len(candidate["rows"]))

    phase = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses", candidate=CANDIDATE_PATH)
    terminal = validation_scope.run_commands(
        root,
        terminal_command_specs(root, root / CANDIDATE_PATH),
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    receipts.extend(terminal)
    _phase(spans, "terminal_validation", phase, started, len(terminal))
    progress(started, "terminal_validation", "after_subprocesses", completed_units=len(terminal))

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=hashes,
        registry_precheck=registry,
        panel_a_artifact=panel_a_artifact,
        schedule=schedule,
        episode_rows=episodes,
        seam_shards=seam_shards,
        boundary_events=boundary_events,
        runtime_receipt=session.get("runtime_receipt") or {},
        model_specs=[resources["model_spec"]],
        validation_receipts=receipts,
        require_terminal=True,
    )
    errors = validate_artifact(artifact, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic_terminal_write", path=RESULT_PATH)
    current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "publish",
        "after_atomic_terminal_write",
        path=RESULT_PATH,
        complete=artifact["arc_panel_b_complete_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse experiment, live-child and cold-replay roles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--role", choices=("experiment", "live-session"), default="experiment")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--reduce-only", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--model-hash")
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--schedule-path", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--session-path", type=Path, default=SESSION_PATH)
    args = parser.parse_args(argv)
    if args.replay is None and args.date is None:
        parser.error("--date or --replay is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI roles.
    """Run the host experiment, owned live child, or fresh replay."""

    args = parse_args(argv)
    if args.replay is not None:
        artifact = load_object(args.replay)
        reduced = independent_reduce(artifact)
        errors = [] if args.reduce_only else validate_artifact(artifact, require_terminal=False)
        print(
            json.dumps(
                {
                    "matches_declared": reduced.get("matches_declared"),
                    "row_count": len(reduced.get("rows") or []),
                    "validation_errors": errors,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return int(bool(errors) or reduced.get("matches_declared") is not True)
    if args.role == "live-session":
        _configure_live_driver()
        return live_driver.run_live_session(args)
    artifact = run_experiment(REPO_ROOT, str(args.date))
    return 0 if artifact.get("verdict_class") in {"null", "blocked"} else 1


__all__ = [
    "CAPABILITY_E2E_NAMES",
    "EXPERIMENT_ID",
    "INFERENCE_SUBSTRATE_CLASS",
    "MODEL_SPECS",
    "PANEL_A_PATH",
    "PANEL_GAMES",
    "REQUIRED_ARTIFACT_FIELDS",
    "REQUIRED_TERMINAL_NAMES",
    "SPEC_PATH",
    "build_artifact_for_test",
    "build_panel_schedule",
    "build_validation_plan",
    "collect_preconditions",
    "compare_panel_a",
    "current_comparability_inputs",
    "independent_reduce",
    "main",
    "parse_args",
    "reduce_panel",
    "terminal_command_specs",
    "validate_artifact",
    "validate_validation_plan",
    "validation_scope",
]
