"""Observe E3 decision seams during a bounded public ARC pilot.

The module keeps all observation wiring inside this development experiment. It
wraps existing E3 methods on one policy instance, records what those call sites
expose, and returns every original value unchanged.

Spec refs: REQ-ARC-WMTE-7471 and SCENARIO-ARC-WMTE-7471-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
from types import MethodType
from typing import Any

import yaml

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7431_v651_arc_live_sentinel as live_support
from carnot.agentic.arc_inference_boundary import BOUNDARY_LEDGER_ENV, InvocationBoundaryLedger
from carnot.agentic.arc_request_budget import attach_request_budget
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.654"
PHASE = 4
EXPERIMENT_ID = "exp7471-v654-arc-seam-observation"
TASK_ID = "experiment_7471_v654_arc_seam_observation"
SCHEMA = "carnot.exp7471.v654.arc_seam_observation.v1"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
MODEL_SPECS = [MODEL_ID]
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
EXECUTION_VENUE = "host"
ACTION_LIMIT = 180
EPISODE_LIMIT_S = 240.0
AGGREGATE_LIVE_LIMIT_S = 2400.0
REQUEST_LIMIT = 2
MAX_NEW_TOKENS = 256
EPISODE_SEEDS = (7_471_001, 7_471_002)
PANEL_SEED = 7_471
SUPERVISOR_THRESHOLD = 120
SUPERVISOR_ARM_ORDER = ("no_redirect", *ARM_ORDER[:3])
SEAM_NAMES = (
    "candidate_action_selection",
    "hypothesis_gate",
    "supervisor_arm_selection",
    "induction_timing",
)
TERMINAL_DISPOSITIONS = {
    "complete",
    "complete_error",
    "failed",
    "censored_timeout",
    "censored_aggregate_limit",
    "censored_no_first_action",
    "unstarted",
}
WITHHELD_INPUTS = (
    "per_game_adapter",
    "stored_engine",
    "banked_trajectory",
    "banked_solution",
    "cross_game_state",
    "game_source",
    "offline_ground_truth_bfs",
    "per_game_model",
)

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
E2E_PATH = Path("ops/e2e-test-plan.md")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
UPSTREAM_PATH = Path("results/experiment_7464_v654_semif_e6_decision_cost_profile.json")
RESULT_PATH = Path("results/experiment_7471_v654_arc_seam_observation.json")
RAW_DIR = Path("results/raw/experiment_7471_v654_arc_seam_observation")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "current_invocation_events.jsonl"
RUNTIME_EVENT_PATH = RAW_DIR / "runtime_events.jsonl"
ACTION_PATH = RAW_DIR / "live_action_rows.jsonl"
CALLBACK_SEAM_PATH = RAW_DIR / "callback_seam_events.jsonl"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7471_v654_arc_seam_observation.json")
TERMINAL_CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7471_v654_arc_seam_observation.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7471_v654_arc_seam_observation.py")
TEST_PATH = Path("tests/python/test_experiment_7471_v654_arc_seam_observation.py")

REQUIRED_TERMINAL = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
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
    "arc_observation_complete_score",
    "per_game_results",
    "solve_provenance",
    "seam_event_shards",
    "public_development_generalization_proxy",
)

VALIDATION_MANIFEST = validation_contract.AffectedManifest(
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
    EXCLUSION_PATH,
    E2E_PATH,
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7457_v653_arc_exposure.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_request_budget.py"),
    Path("python/carnot/agentic/arc_trajectory_supervisor.py"),
    REGISTRY_PATH,
    UPSTREAM_PATH,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


FIELD_PRINCIPLES = {
    "schema": "Use a versioned schema with exact experiment, milestone, and terminal status.",
    "run_date": "Use 20260921 and measured UTC and monotonic boundaries with clock identity.",
    "preconditions_checked": "Record exact paths, ownership, device identity, and observed prerequisites before work.",
    "MODEL_SPECS": "Name the current Qwen model only when current model work was attempted.",
    "model_specs": "Mirror model specifications for lowercase-field readers.",
    "model_invoked": "Distinguish current attempted calls from archived and scripted events.",
    "invocation_counts": "Balance attempted, completed, failed, cancelled, and in-flight current work.",
    "inference_substrate": "Name the actual current generation, load-only, or simulator substrate.",
    "inference_substrate_class": "Use the declared compute class only when that work occurred.",
    "execution_venue": "Name the host and retain actual CPU, CUDA, lease, and offload identities separately.",
    "duration_s": "Measure current work without padding and retain phase-specific durations.",
    "phase_spans": "Bind progress events, monotonic timings, and completed-unit checkpoints.",
    "random_seed": "Freeze selection, episode, ordering, audit, and bootstrap seeds.",
    "reproducibility_checksum": "Bind code, protocol, data roles, models, raw shards, and validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes, flags, and evidence classes.",
    "rows": "Keep every game and seed, including failed, censored, and unstarted units.",
    "sample_size_budget": "Separate planned, attempted, complete, failed, censored, and unstarted units.",
    "acceptance_gate_results": "Keep validity and benefit checks as typed operands with principles.",
    "gate_check_summary": "Name every failed check and its exact upstream field without hiding nulls.",
    "honest_verdict": "Use complete terminal findings; blocked findings retain the exact failed check.",
    "verdict_class": "Use the closed terminal enum and reserve partial for retryable owned work.",
    "verifier_is_oracle": "The public environment is the level oracle, so positive is forbidden.",
    "flagged_adversarial": "Retain structural reader findings and never clear them to open a gate.",
    "validation_receipts": "Capture exact affected commands, exits, hashes, and required status.",
    "field_principles": "Explain why each ordinary field and acceptance gate exists.",
    "arc_observation_complete_score": "One means all eight dispositions and actual invocation receipts exist.",
    "per_game_results": "Treat four games as clusters and two seeds as repeated runs.",
    "solve_provenance": "Credit only reproduced progress from the agent's own runtime trace.",
    "seam_event_shards": "Preserve real candidate sets and intervals, including unavailable fields.",
    "public_development_generalization_proxy": "Public adapter-withheld games do not establish hidden efficacy.",
}


def utc_now() -> str:
    """Return one aware UTC boundary for durable experiment evidence."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase and long-operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7471] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash canonical JSON bytes so any changed receipt moves its identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to an empty mapping."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read complete JSON object rows and ignore no malformed evidence silently."""

    rows: list[JsonDict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"event": "malformed_jsonl", "raw_sha256": canonical_hash(line)})
            continue
        rows.append(dict(value) if isinstance(value, Mapping) else {"event": "non_object"})
    return rows


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Append and fsync one boundary before dependent work continues."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), sort_keys=True, default=str) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum itself."""

    copied = deepcopy(dict(value))
    copied["reproducibility_checksum"] = ""
    return canonical_hash(copied)


def _compare(operator: str, expected: Any, observed: Any) -> bool:
    if operator == "==":
        return observed == expected
    if operator == ">=":
        return observed is not None and observed >= expected
    if operator == "in":
        return observed in expected
    raise ValueError(f"unsupported operator: {operator}")


def gate_row(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    artifact_field: str,
    principle: str,
    operator: str = "==",
) -> JsonDict:
    """Keep one acceptance comparison as values plus its audit meaning."""

    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": operator,
        "passed": _compare(operator, expected, observed),
        "upstream": upstream,
        "path": upstream,
        "field": artifact_field,
        "artifact_field": artifact_field,
        "principle": principle,
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failure and expose the first exact comparison."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "failed_checks": failures,
        "first_failed_check": first.get("check") if first else None,
        "upstream": first.get("upstream") if first else None,
        "exact_field_path": first.get("artifact_field") if first else None,
        "observed_value": first.get("observed") if first else None,
    }


def _source_record(path: Path, role: str, flags: Mapping[str, Any] | None = None) -> JsonDict:
    row: JsonDict = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "role": role,
    }
    if flags is not None:
        row["original_flags"] = deepcopy(dict(flags))
    return row


def collect_preconditions(
    root: Path, *, force_live: str | None = None
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate required bytes, E6 flags, the manifest, and registry."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                "validity",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                upstream=relative.as_posix(),
                artifact_field="bytes",
                principle="Dependent work starts only after exact input bytes exist.",
            )
        )
        if available:
            hashes[relative.as_posix()] = _source_record(path, "current_input")

    upstream = load_object(root / UPSTREAM_PATH)
    flags = {
        "status": upstream.get("status"),
        "verdict_class": upstream.get("verdict_class"),
        "flagged_adversarial": upstream.get("flagged_adversarial"),
        "decision_profile_complete_score": upstream.get("decision_profile_complete_score"),
    }
    if (root / UPSTREAM_PATH).is_file():
        hashes[UPSTREAM_PATH.as_posix()] = _source_record(
            root / UPSTREAM_PATH, "structured_prerequisite", flags
        )
    for field, expected in (
        ("decision_profile_complete_score", 1),
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
    ):
        checks.append(
            gate_row(
                f"exp7464.{field}",
                "validity",
                expected,
                upstream.get(field),
                upstream=UPSTREAM_PATH.as_posix(),
                artifact_field=field,
                principle="E6 keeps its original disposition while supplying the seam schema.",
            )
        )
    upstream_seams = [
        row.get("seam")
        for row in (upstream.get("seam_observation_spec") or {}).get("seams", [])[:4]
        if isinstance(row, Mapping)
    ]
    checks.append(
        gate_row(
            "exp7464.four_decision_seams",
            "validity",
            list(SEAM_NAMES),
            upstream_seams,
            upstream=UPSTREAM_PATH.as_posix(),
            artifact_field="seam_observation_spec.seams[0:4]",
            principle="The live observer implements the exact E6 missing-event contract.",
        )
    )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        gate_row(
            "driving_requirement",
            "validity",
            True,
            "REQ-ARC-WMTE-7471" in spec_text,
            upstream=SPEC_PATH.as_posix(),
            artifact_field="REQ-ARC-WMTE-7471",
            principle="The capability requirement must exist before implementation runs.",
        )
    )
    checks.append(
        gate_row(
            "force_live",
            "validity",
            "1",
            os.environ.get("CARNOT_FORCE_LIVE") if force_live is None else force_live,
            upstream="environment",
            artifact_field="CARNOT_FORCE_LIVE",
            principle="Current model work cannot fall back to simulation.",
        )
    )
    try:
        exclusion = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8")) or {}
        registry_value = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
        parse_ok = isinstance(exclusion, Mapping) and isinstance(registry_value, Mapping)
    except (OSError, yaml.YAMLError):  # pragma: no cover - authenticated repo inputs exist.
        exclusion, registry_value, parse_ok = {}, {}, False
    checks.append(
        gate_row(
            "manifest_and_registry_parse",
            "validity",
            True,
            parse_ok,
            upstream=f"{EXCLUSION_PATH.as_posix()}|{REGISTRY_PATH.as_posix()}",
            artifact_field="yaml_objects",
            principle="Quarantine and duplicate-credit checks require structured inputs.",
        )
    )
    excluded_text = json.dumps(exclusion, sort_keys=True)
    checks.append(
        gate_row(
            "task_not_excluded",
            "validity",
            False,
            "7471" in excluded_text or EXPERIMENT_ID in excluded_text,
            upstream=EXCLUSION_PATH.as_posix(),
            artifact_field="experiment_id:7471",
            principle="An excluded task cannot start model work.",
        )
    )
    registry = dict(registry_value) if isinstance(registry_value, Mapping) else {}
    return checks, hashes, registry


def accessible_games(root: Path) -> list[str]:  # pragma: no cover - environment catalog boundary.
    """List public environment identifiers without importing their source."""

    directory = root / "environment_files"
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


def freeze_game_panel(games: Iterable[str]) -> JsonDict:
    """Select four non-E6 games by a stable label-only hash."""

    candidates = sorted({str(game) for game in games if str(game) not in {"bp35", "cn04"}})
    rows = [
        {
            "game": game,
            "stable_hash": canonical_hash({"panel_seed": PANEL_SEED, "game": game}),
        }
        for game in candidates
    ]
    rows.sort(key=lambda row: (row["stable_hash"], row["game"]))
    selected = rows[:4]
    return {
        "passed": len(selected) == 4,
        "games": [row["game"] for row in selected],
        "game_rows": selected,
        "accessible_game_count": len(candidates),
        "selection_basis": "ascending_sha256_of_panel_seed_and_public_game_id",
        "selection_used_current_outcomes": False,
        "game_source_read": False,
        "registry_used_for_policy": False,
    }


def registry_precheck(registry: Mapping[str, Any], games: Sequence[str]) -> JsonDict:
    """Record prior credit before runtime without sending it to the policy."""

    indexed = {
        str(row.get("game")): row
        for row in registry.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    rows = [
        {
            "game": game,
            "registered": game in indexed,
            "levels_reproduced": indexed.get(game, {}).get("levels_reproduced"),
            "full_game_clear": indexed.get(game, {}).get("full_game_clear"),
        }
        for game in games
    ]
    return {
        "passed": len(rows) == 4,
        "rows": rows,
        "new_credit_allowed": False,
        "policy_received_registry_data": False,
    }


def build_schedule(games: Sequence[str]) -> list[JsonDict]:
    """Seal four game clusters with two repeated seeds each."""

    rows: list[JsonDict] = []
    for game in tuple(games)[:4]:
        for seed in EPISODE_SEEDS:
            rows.append(
                {
                    "episode_id": f"{game}:seed-{seed}",
                    "game": str(game),
                    "seed": seed,
                    "execution_order": len(rows),
                    "action_limit": ACTION_LIMIT,
                    "episode_limit_s": EPISODE_LIMIT_S,
                    "request_limit": REQUEST_LIMIT,
                    "max_new_tokens_per_call": MAX_NEW_TOKENS,
                    "adapter_disabled": True,
                    "stored_engines_disabled": True,
                    "banked_trajectories_disabled": True,
                    "cross_game_state_disabled": True,
                    "hidden_game_source_disabled": True,
                    "offline_ground_truth_bfs_disabled": True,
                    "withheld_inputs": list(WITHHELD_INPUTS),
                }
            )
    return rows


class E3SeamObserver:
    """Wrap one E3-shaped policy and persist call-site evidence unchanged."""

    def __init__(
        self,
        episode_id: str,
        event_path: Path,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.episode_id = str(episode_id)
        self.event_path = Path(event_path)
        self.clock_ns = clock_ns
        self.errors: list[str] = []
        self._serial = 0
        self._action_count = 0
        self._hypothesis_candidates: list[JsonDict] = []
        self._hypothesis_active = False

    def _decision_id(self, seam: str) -> str:
        self._serial += 1
        return f"{self.episode_id}:{seam}:{self._serial}"

    @staticmethod
    def _state(policy: Any, latest: Any = None) -> JsonDict:
        level: int | None = None
        if latest is not None:
            try:
                level = int(getattr(latest, "levels_completed"))
            except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive only.
                level = None
        return {
            "phase": getattr(policy, "phase", None),
            "level": level,
            "goal_level": getattr(policy, "_current_goal_level", None),
        }

    def _emit(self, seam: str, event: str, decision_id: str, **fields: Any) -> None:
        entered = self.clock_ns()
        row = {
            "schema": "carnot.arc.e3_decision_seam_event.v1",
            "episode_id": self.episode_id,
            "decision_id": decision_id,
            "parent_decision_id": fields.pop("parent_decision_id", None),
            "seam": seam,
            "event": event,
            "event_monotonic_ns": entered,
            "clock_identity": "time.monotonic_ns",
            "action_count": self._action_count,
            "input_tokens": fields.pop("input_tokens", 0),
            "output_tokens": fields.pop("output_tokens", 0),
            "cpu_work_ns": fields.pop("cpu_work_ns", 0),
            "gpu_work_ns": fields.pop("gpu_work_ns", 0),
            "gpu_device_id": fields.pop("gpu_device_id", None),
            **fields,
        }
        _append_jsonl(self.event_path, row)

    @staticmethod
    def _move(move: Any) -> JsonDict:
        if isinstance(move, tuple):
            return {"action": move[0], "payload": deepcopy(move[1] if len(move) > 1 else None)}
        return {"action": str(getattr(move, "name", move)), "payload": None}

    @staticmethod
    def _legal_candidates(latest: Any) -> tuple[list[JsonDict], str | None]:
        if latest is None:
            return [], "legal_action_payloads_unavailable_at_call_site"
        try:
            from carnot.agentic.arc_agi3_live_adapter import _available_action_ids

            ids = tuple(_available_action_ids(latest))
        except Exception:  # pragma: no cover - SDK helper is qualified separately.
            ids = ()
        candidates = [
            {
                "stable_candidate_id": f"action:{int(action_id)}:payload:unavailable",
                "action": int(action_id),
                "payload": None,
                "rank": rank,
            }
            for rank, action_id in enumerate(ids)
        ]
        reason = None if candidates else "legal_action_payloads_unavailable_at_call_site"
        return candidates, reason

    @staticmethod
    def _supervisor_rows(supervisor: Any) -> list[JsonDict]:
        try:
            receipt = supervisor.receipt()
        except Exception:
            return []
        return [
            dict(row)
            for key in ("redirects", "would_have_redirects")
            for row in receipt.get(key, [])
            if isinstance(row, Mapping)
        ]

    @staticmethod
    def _selected_hypothesis(policy: Any) -> tuple[str | None, bool | None]:
        selection = getattr(policy, "world_model_trust_selection", None)
        selected = getattr(selection, "selected_name", None)
        score = getattr(selection, "selected_score", None)
        passed = getattr(score, "binary_gate_pass", None)
        attempts = getattr(policy, "induction_attempts", []) or []
        if selected is None and attempts and isinstance(attempts[-1], Mapping):
            selected = attempts[-1].get("selected_candidate_name")
        return (str(selected) if selected is not None else None, passed)

    def install(self, policy: Any) -> Any:
        """Install wrappers once and return the same policy object."""

        if getattr(policy, "_exp7471_seam_observer", None) is not None:
            raise ValueError("policy already has an Exp7471 observer")
        policy._exp7471_seam_observer = self
        self._install_action(policy)
        self._install_induction_timing(policy)
        self._install_supervisor(policy)
        self._install_hypothesis(policy)
        return policy

    def _install_action(self, policy: Any) -> None:
        original = policy.next_move

        def observed(_policy: Any, frames: Any, latest: Any) -> Any:
            decision_id = self._decision_id("candidate_action_selection")
            started = self.clock_ns()
            state_before = self._state(policy, latest)
            self._emit(
                "candidate_action_selection",
                "stage_start",
                decision_id,
                interval_start_monotonic_ns=started,
                state_before=state_before,
                progress_before=state_before.get("level"),
            )
            candidates, missing = self._legal_candidates(latest)
            self._emit(
                "candidate_action_selection",
                "candidate_set",
                decision_id,
                candidate_ids=[row["stable_candidate_id"] for row in candidates],
                candidates=candidates,
                missing_options_reason=missing,
            )
            move = original(frames, latest)
            self._emit(
                "candidate_action_selection",
                "selection",
                decision_id,
                selected=self._move(move),
            )
            ended = self.clock_ns()
            self._action_count += 1
            self._emit(
                "candidate_action_selection",
                "stage_end",
                decision_id,
                interval_start_monotonic_ns=started,
                interval_end_monotonic_ns=ended,
                observer_cpu_ns=max(0, self.clock_ns() - ended),
                cpu_work_ns=max(0, ended - started),
                state_after=self._state(policy, latest),
                progress_after=self._state(policy, latest).get("level"),
            )
            return move

        policy.next_move = MethodType(observed, policy)

    def _install_induction_timing(self, policy: Any) -> None:
        original = policy._should_enter_induction

        def observed(_policy: Any, *, stalled: bool, won: bool) -> Any:
            decision_id = self._decision_id("induction_timing")
            started = self.clock_ns()
            self._emit(
                "induction_timing",
                "stage_start",
                decision_id,
                interval_start_monotonic_ns=started,
                trigger_state={
                    "stalled": stalled,
                    "won": won,
                    "induced": getattr(policy, "induced", None),
                },
            )
            candidates = [
                {
                    "stable_candidate_id": candidate,
                    "trigger_reason": "evaluated_by_E3AgentPolicy._should_enter_induction",
                    "defer_until": None,
                }
                for candidate in ("induce_now", "defer_induction", "skip_induction")
            ]
            self._emit(
                "induction_timing",
                "candidate_set",
                decision_id,
                candidate_ids=[row["stable_candidate_id"] for row in candidates],
                candidates=candidates,
                missing_options_reason=None,
            )
            result = original(stalled=stalled, won=won)
            enter, reason = result
            selected = "induce_now" if enter else "skip_induction" if won else "defer_induction"
            self._emit(
                "induction_timing",
                "selection",
                decision_id,
                selected_candidate_ids=[selected],
                reason=reason,
            )
            ended = self.clock_ns()
            self._emit(
                "induction_timing",
                "stage_end",
                decision_id,
                interval_start_monotonic_ns=started,
                interval_end_monotonic_ns=ended,
                cpu_work_ns=max(0, ended - started),
                observer_cpu_ns=0,
            )
            return result

        policy._should_enter_induction = MethodType(observed, policy)

    def _install_supervisor(self, policy: Any) -> None:
        original = policy._maybe_supervise_trajectory

        def observed(_policy: Any, latest: Any) -> Any:
            decision_id = self._decision_id("supervisor_arm_selection")
            started = self.clock_ns()
            supervisor = getattr(policy, "_trajectory_supervisor", None)
            before = self._supervisor_rows(supervisor)
            self._emit(
                "supervisor_arm_selection",
                "stage_start",
                decision_id,
                interval_start_monotonic_ns=started,
                state_before=self._state(policy, latest),
            )
            candidates = [
                {
                    "stable_candidate_id": arm,
                    "eligibility": None,
                    "rank": rank,
                }
                for rank, arm in enumerate(SUPERVISOR_ARM_ORDER)
            ]
            self._emit(
                "supervisor_arm_selection",
                "eligible_candidate_set",
                decision_id,
                candidate_ids=list(SUPERVISOR_ARM_ORDER),
                candidates=candidates,
                missing_options_reason="call_site_exposes_only_selected_redirect_not_all_arm_eligibility",
            )
            result = original(latest)
            after = self._supervisor_rows(supervisor)
            selected = str(after[-1].get("arm")) if len(after) > len(before) else "no_redirect"
            applied = bool(
                selected != "no_redirect"
                and getattr(policy, "_trajectory_supervisor_applies", False)
            )
            self._emit(
                "supervisor_arm_selection",
                "selection",
                decision_id,
                selected_candidate_ids=[selected],
                supervisor_fired=selected != "no_redirect",
                applied_redirection=applied,
            )
            ended = self.clock_ns()
            self._emit(
                "supervisor_arm_selection",
                "stage_end",
                decision_id,
                interval_start_monotonic_ns=started,
                interval_end_monotonic_ns=ended,
                cpu_work_ns=max(0, ended - started),
                observer_cpu_ns=0,
                state_after=self._state(policy, latest),
            )
            return result

        policy._maybe_supervise_trajectory = MethodType(observed, policy)

    def _install_hypothesis(self, policy: Any) -> None:
        original_candidates = policy._world_model_candidates
        original_induction = policy._induce_and_plan_timed

        def candidates_observed(_policy: Any, engine: Any, is_done: Any) -> Any:
            result = original_candidates(engine, is_done)
            if self._hypothesis_active:
                self._hypothesis_candidates = [
                    {
                        "stable_candidate_id": str(getattr(row, "name", f"candidate_{index}")),
                        "hypothesis_hash": canonical_hash(str(getattr(row, "name", index))),
                        "gate_decision": None,
                    }
                    for index, row in enumerate(result)
                ]
            return result

        def induction_observed(_policy: Any) -> Any:
            decision_id = self._decision_id("hypothesis_gate")
            started = self.clock_ns()
            self._hypothesis_candidates = []
            self._hypothesis_active = True
            self._emit(
                "hypothesis_gate",
                "stage_start",
                decision_id,
                interval_start_monotonic_ns=started,
                state_before=self._state(policy),
            )
            try:
                result = original_induction()
            finally:
                self._hypothesis_active = False
            missing = None if self._hypothesis_candidates else "no_candidate_pool_reached"
            self._emit(
                "hypothesis_gate",
                "candidate_set",
                decision_id,
                candidate_ids=[row["stable_candidate_id"] for row in self._hypothesis_candidates],
                candidates=deepcopy(self._hypothesis_candidates),
                missing_options_reason=missing,
            )
            selected, passed = self._selected_hypothesis(policy)
            self._emit(
                "hypothesis_gate",
                "gate_result",
                decision_id,
                selected_candidate_ids=[selected] if selected else [],
                gate_decision="accept"
                if passed is True
                else "reject"
                if passed is False
                else "escalate",
            )
            ended = self.clock_ns()
            self._emit(
                "hypothesis_gate",
                "stage_end",
                decision_id,
                interval_start_monotonic_ns=started,
                interval_end_monotonic_ns=ended,
                cpu_work_ns=max(0, ended - started),
                observer_cpu_ns=0,
                state_after=self._state(policy),
            )
            return result

        policy._world_model_candidates = MethodType(candidates_observed, policy)
        policy._induce_and_plan_timed = MethodType(induction_observed, policy)


def _unstarted_row(schedule: Mapping[str, Any]) -> JsonDict:
    """Represent one sealed unit that aggregate stopping prevented."""

    return {
        **deepcopy(dict(schedule)),
        "disposition": "unstarted",
        "action_count": 0,
        "start_level": None,
        "peak_level": None,
        "terminal_level": None,
        "action_rows": [],
        "request_budget_receipt": None,
        "actions_to_progress": None,
        "solve_provenance": "unstarted",
        "trace_reproduction": {"attempted": False, "passed": False},
        "new_level_credit": 0,
        "elapsed_s": 0.0,
        "error": None,
    }


def _actions_to_progress(row: Mapping[str, Any]) -> int | None:
    start = row.get("start_level")
    if not isinstance(start, int):
        return None
    for action in row.get("action_rows", []):
        if isinstance(action, Mapping) and int(action.get("level") or 0) > start:
            return int(action.get("action_index") or 0)
    return None


def _wilson(successes: int, total: int) -> JsonDict:
    if total <= 0:
        return {"estimate": None, "lower_95": None, "upper_95": None, "clusters": 0}
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return {
        "estimate": p,
        "lower_95": max(0.0, center - radius),
        "upper_95": min(1.0, center + radius),
        "clusters": total,
    }


def _cost_bounds(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    starts: dict[str, int] = {}
    completed: set[str] = set()
    lower = 0
    overhead = 0
    latest = 0
    for event in events:
        if event.get("seam") not in SEAM_NAMES:
            continue
        decision_id = str(event.get("decision_id") or "")
        tick = int(event.get("event_monotonic_ns") or 0)
        latest = max(latest, tick)
        if event.get("event") == "stage_start" and decision_id:
            starts[decision_id] = int(event.get("interval_start_monotonic_ns") or tick)
        if event.get("event") == "stage_end" and decision_id in starts:
            end = int(event.get("interval_end_monotonic_ns") or tick)
            lower += max(0, end - starts[decision_id])
            overhead += max(0, int(event.get("observer_cpu_ns") or 0))
            completed.add(decision_id)
    unfinished = [key for key in starts if key not in completed]
    upper = lower + sum(max(0, latest - starts[key]) for key in unfinished)
    return {
        "observed_lower": lower,
        "observed_upper": upper,
        "observation_overhead": overhead,
        "unfinished_decisions": len(unfinished),
    }


def reduce_panel(
    schedule: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    seam_events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reduce sealed episodes and raw seams without outcome reconstruction."""

    observed = {
        str(row.get("episode_id")): dict(row)
        for row in episode_rows
        if isinstance(row, Mapping) and row.get("episode_id")
    }
    rows: list[JsonDict] = []
    for sealed in schedule:
        episode_id = str(sealed["episode_id"])
        row = (
            {**deepcopy(dict(sealed)), **deepcopy(observed[episode_id])}
            if episode_id in observed
            else _unstarted_row(sealed)
        )
        row["actions_to_progress"] = _actions_to_progress(row)
        progressed = row["actions_to_progress"] is not None
        reproduced = bool((row.get("trace_reproduction") or {}).get("passed"))
        row["solve_provenance"] = (
            "live_agent_self_discovery"
            if progressed and reproduced
            else "unreproduced_live_progress"
            if progressed
            else row.get("solve_provenance") or "no_level_reached"
        )
        row["new_level_credit"] = 0
        rows.append(row)

    dispositions = Counter(str(row.get("disposition")) for row in rows)
    censored = sum(count for key, count in dispositions.items() if key.startswith("censored_"))
    failed = dispositions["failed"] + dispositions["complete_error"]
    unstarted = dispositions["unstarted"]
    complete = dispositions["complete"]
    per_game: list[JsonDict] = []
    for game in dict.fromkeys(str(row.get("game")) for row in rows):
        game_rows = [row for row in rows if str(row.get("game")) == game]
        per_game.append(
            {
                "game": game,
                "seed_rows": [
                    {
                        "seed": row.get("seed"),
                        "episode_id": row.get("episode_id"),
                        "disposition": row.get("disposition"),
                        "action_count": row.get("action_count"),
                        "actions_to_progress": row.get("actions_to_progress"),
                        "progressed": row.get("actions_to_progress") is not None,
                    }
                    for row in game_rows
                ],
                "completed_seeds": sum(row.get("disposition") == "complete" for row in game_rows),
                "any_reproduced_progress": any(
                    row.get("solve_provenance") == "live_agent_self_discovery" for row in game_rows
                ),
                "right_censored": any(row.get("disposition") != "complete" for row in game_rows),
            }
        )
    cluster_successes = sum(row["any_reproduced_progress"] for row in per_game)
    actions = [
        int(row["actions_to_progress"])
        for row in rows
        if isinstance(row.get("actions_to_progress"), int)
    ]
    seam_counts = Counter(str(row.get("seam")) for row in seam_events if row.get("seam"))
    all_dispositions = len(rows) == len(schedule) == 8 and all(
        row.get("disposition") in TERMINAL_DISPOSITIONS for row in rows
    )
    return {
        "rows": rows,
        "per_game_results": per_game,
        "sample_size_budget": {
            "planned_independent_units": len(schedule),
            "attempted_independent_units": len(rows) - unstarted,
            "complete_independent_units": complete,
            "failed_independent_units": failed,
            "censored_independent_units": censored,
            "unstarted_independent_units": unstarted,
            "independent_game_clusters": len(per_game),
        },
        "all_dispositions_present": all_dispositions,
        "clustered_progress_uncertainty": _wilson(cluster_successes, len(per_game)),
        "actions_to_progress": {
            "observed": actions,
            "median": sorted(actions)[len(actions) // 2] if actions else None,
            "right_censored_episode_count": sum(
                value is None for value in (row.get("actions_to_progress") for row in rows)
            ),
        },
        "replaceable_cost_bounds_ns": _cost_bounds(seam_events),
        "seam_event_counts": {name: seam_counts[name] for name in SEAM_NAMES},
    }


def _events_from_shards(
    shards: Sequence[Mapping[str, Any]], root: Path = REPO_ROOT
) -> list[JsonDict]:
    events: list[JsonDict] = []
    for shard in shards:
        inline = shard.get("inline_rows")
        if isinstance(inline, list):
            events.extend(dict(row) for row in inline if isinstance(row, Mapping))
            continue
        raw_path = shard.get("path")
        if isinstance(raw_path, str):
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            events.extend(read_jsonl(path))
    return events


def _field_principles(keys: Sequence[str]) -> JsonDict:
    return {
        key: FIELD_PRINCIPLES.get(
            key, "Retain this typed field so independent readers can audit the experiment."
        )
        for key in keys
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def _invocation_reduction(
    boundary_events: Sequence[Mapping[str, Any]], *, child_terminal: bool
) -> JsonDict:
    current = live_support.reduce_current_invocations(
        boundary_events, child_terminal=child_terminal
    )
    counts = dict(current["invocation_counts"])
    generated = counts["generation_calls_attempted"] > 0
    loaded = counts["model_loads_attempted"] > 0
    substrate_class = (
        "model_bounded_generation"
        if generated
        else "model_load_no_generation"
        if loaded
        else "cpu_exact_solver_or_simulator"
    )
    substrate = (
        "owned_native_cuda_llama_cpp_qwen3.8_27b_gguf"
        if generated
        else "owned_native_cuda_model_load_only"
        if loaded
        else "no_model_cpu_exact_solver_or_simulator"
    )
    return {
        **current,
        "inference_substrate_class": substrate_class,
        "inference_substrate": substrate,
    }


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    selection: Mapping[str, Any],
    registry_receipt: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    seam_shards: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    require_terminal: bool,
) -> JsonDict:
    """Build one terminal record only from current raw inputs."""

    seam_events = _events_from_shards(seam_shards)
    panel = reduce_panel(schedule, episode_rows, seam_events)
    invocation = _invocation_reduction(
        boundary_events, child_terminal=bool(runtime_receipt.get("child_terminal", True))
    )
    counts = invocation["invocation_counts"]
    attempted_model = counts["model_loads_attempted"] + counts["generation_calls_attempted"] > 0
    offload_observed = not attempted_model or bool(
        (runtime_receipt.get("observed_cuda_offload") or {}).get("passed")
    )
    invocation_balanced = all(
        counts[f"{prefix}_attempted"]
        == sum(
            counts[f"{prefix}_{state}"]
            for state in ("completed", "failed", "cancelled", "in_flight")
        )
        for prefix in ("model_loads", "generation_calls")
    )
    affected_ok = (
        _receipts_pass(validation_receipts, validation_scope.REQUIRED_CHECK_NAMES)
        if require_terminal
        else True
    )
    terminal_ok = (
        _receipts_pass(validation_receipts, REQUIRED_TERMINAL) if require_terminal else True
    )
    validity = [
        gate_row(
            "preconditions_checked",
            "validity",
            True,
            all(row.get("passed") is True for row in preconditions),
            upstream="preconditions_checked",
            artifact_field="passed",
            principle="Every exact prerequisite must pass before dependent work.",
        ),
        gate_row(
            "eight_episode_dispositions",
            "validity",
            True,
            panel["all_dispositions_present"],
            upstream="raw_episode_rows",
            artifact_field="disposition",
            principle="Every sealed episode needs a terminal or unstarted disposition.",
        ),
        gate_row(
            "invocation_accounting_balanced",
            "validity",
            True,
            invocation_balanced,
            upstream="current_invocation_events",
            artifact_field="invocation_counts",
            principle="Attempted current calls must have exactly one disposition.",
        ),
        gate_row(
            "actual_cuda_offload_receipt",
            "validity",
            True,
            offload_observed,
            upstream="execution_venue_details.observed_cuda_offload",
            artifact_field="passed",
            principle="Requested GPU layers do not prove that the owned model occupied CUDA memory.",
        ),
        gate_row(
            "affected_validation",
            "validity",
            True,
            affected_ok,
            upstream="validation_receipts",
            artifact_field="required_affected_commands",
            principle="All scoped checks must pass once on the frozen manifest.",
        ),
        gate_row(
            "terminal_readers",
            "validity",
            True,
            terminal_ok,
            upstream="validation_receipts",
            artifact_field="required_terminal_commands",
            principle="Cold replay and unchanged strict readers must pass.",
        ),
    ]
    benefit = [
        gate_row(
            "four_game_clusters",
            "benefit",
            4,
            panel["sample_size_budget"]["independent_game_clusters"],
            upstream="frozen_schedule",
            artifact_field="independent_game_clusters",
            principle="Uncertainty treats games, not repeated seeds, as independent.",
            operator=">=",
        ),
        gate_row(
            "all_four_seams_reachable",
            "benefit",
            True,
            all(panel["seam_event_counts"][name] > 0 for name in SEAM_NAMES),
            upstream="seam_event_shards",
            artifact_field="seam_event_counts",
            principle="A zero opportunity is a measured null rather than a fabricated event.",
        ),
    ]
    gates = [*validity, *benefit]
    valid = all(row["passed"] for row in validity)
    complete_score = int(
        panel["all_dispositions_present"] and invocation_balanced and offload_observed
    )
    verdict_class = "null" if valid else "disqualified"
    status = (
        "complete_null_live_arc_seam_observation"
        if valid
        else "complete_disqualified_arc_seam_observation"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "clock_identity": {"wall": "datetime.now(UTC)", "interval": "time.monotonic_ns"},
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": deepcopy(list(model_specs)) if attempted_model else [],
        "model_specs": deepcopy(list(model_specs)) if attempted_model else [],
        "model_invoked": bool(invocation["model_invoked"]),
        "invocation_counts": deepcopy(counts),
        "inference_substrate": invocation["inference_substrate"],
        "inference_substrate_class": invocation["inference_substrate_class"],
        "planned_inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "execution_venue_details": deepcopy(dict(runtime_receipt)),
        "requested_offload": {"n_gpu_layers": 999, "device": "one_owned_cuda_gpu"},
        "observed_offload": {
            "device_uuid": runtime_receipt.get("gpu_uuid"),
            "device_name": runtime_receipt.get("gpu_name"),
            "receipt": deepcopy(runtime_receipt.get("observed_cuda_offload")),
            "resident_observed": bool(
                (runtime_receipt.get("observed_cuda_offload") or {}).get("passed")
            )
            if attempted_model
            else False,
        },
        "duration_s": round(float(duration_s), 6),
        "duration_breakdown_s": {
            str(row.get("phase")): round(float(row.get("duration_s") or 0.0), 6)
            for row in phase_spans
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "selection": PANEL_SEED,
            "episodes": list(EPISODE_SEEDS),
            "ordering": PANEL_SEED,
            "audit": 7_471_091,
            "bootstrap": 7_471_095,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "selection_receipt": deepcopy(dict(selection)),
        "registry_precheck": deepcopy(dict(registry_receipt)),
        "protocol_schedule": deepcopy(list(schedule)),
        "rows": panel["rows"],
        "sample_size_budget": panel["sample_size_budget"],
        "per_game_results": panel["per_game_results"],
        "clustered_uncertainty": panel["clustered_progress_uncertainty"],
        "actions_to_progress": panel["actions_to_progress"],
        "replaceable_cost_bounds_ns": panel["replaceable_cost_bounds_ns"],
        "seam_event_counts": panel["seam_event_counts"],
        "seam_event_shards": deepcopy(list(seam_shards)),
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary(gates),
        "honest_verdict": status,
        "verdict_class": verdict_class,
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        "validation_receipts": deepcopy(list(validation_receipts)),
        "arc_observation_complete_score": complete_score,
        "solve_provenance": "live_agent_self_discovery_only_after_fresh_trace_reproduction",
        "new_level_credit": 0,
        "promotion_score": 0,
        "public_development_generalization_proxy": True,
        "hidden_game_efficacy_claim": False,
        "selector_added": False,
        "supervisor_arm_added": False,
        "e7_e12_enabled": False,
        "research_conductor_changed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute all row, cluster, and seam claims from frozen raw inputs."""

    reduced = reduce_panel(
        artifact.get("protocol_schedule", []),
        artifact.get("rows", []),
        _events_from_shards(artifact.get("seam_event_shards", [])),
    )
    declared = {
        "rows": artifact.get("rows"),
        "sample_size_budget": artifact.get("sample_size_budget"),
        "per_game_results": artifact.get("per_game_results"),
        "clustered_uncertainty": artifact.get("clustered_uncertainty"),
        "actions_to_progress": artifact.get("actions_to_progress"),
        "replaceable_cost_bounds_ns": artifact.get("replaceable_cost_bounds_ns"),
        "seam_event_counts": artifact.get("seam_event_counts"),
    }
    recomputed = {
        "rows": reduced["rows"],
        "sample_size_budget": reduced["sample_size_budget"],
        "per_game_results": reduced["per_game_results"],
        "clustered_uncertainty": reduced["clustered_progress_uncertainty"],
        "actions_to_progress": reduced["actions_to_progress"],
        "replaceable_cost_bounds_ns": reduced["replaceable_cost_bounds_ns"],
        "seam_event_counts": reduced["seam_event_counts"],
    }
    return {
        **recomputed,
        "matches_declared": canonical_hash(declared) == canonical_hash(recomputed),
    }


def validate_artifact(value: Mapping[str, Any] | Path, *, require_terminal: bool) -> list[str]:
    """Cold-check identity, independent reduction, counters, and checksum."""

    artifact = load_object(value) if isinstance(value, Path) else dict(value)
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    for field, expected in (
        ("schema", SCHEMA),
        ("experiment_id", EXPERIMENT_ID),
        ("milestone", MILESTONE),
        ("run_date", RUN_DATE),
    ):
        if artifact.get(field) != expected:
            errors.append(f"identity_mismatch:{field}")
    if independent_reduce(artifact)["matches_declared"] is not True:
        errors.append("independent_reduction_mismatch")
    counts = artifact.get("invocation_counts") or {}
    for prefix in ("model_loads", "generation_calls"):
        attempted = int(counts.get(f"{prefix}_attempted") or 0)
        terminal = sum(
            int(counts.get(f"{prefix}_{state}") or 0)
            for state in ("completed", "failed", "cancelled", "in_flight")
        )
        if attempted != terminal:
            errors.append(f"unbalanced_invocations:{prefix}")
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
    if require_terminal and not _receipts_pass(
        artifact.get("validation_receipts", []),
        (*validation_scope.REQUIRED_CHECK_NAMES, *REQUIRED_TERMINAL),
    ):
        errors.append("required_validation_missing_or_failed")
    expected_checksum = artifact_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected_checksum:
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _fixture_events(episode_id: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    tick = 100
    for index, seam in enumerate(SEAM_NAMES):
        decision = f"{episode_id}:{seam}:{index}"
        rows.extend(
            [
                {
                    "episode_id": episode_id,
                    "decision_id": decision,
                    "seam": seam,
                    "event": "stage_start",
                    "event_monotonic_ns": tick,
                    "interval_start_monotonic_ns": tick,
                },
                {
                    "episode_id": episode_id,
                    "decision_id": decision,
                    "seam": seam,
                    "event": "stage_end",
                    "event_monotonic_ns": tick + 10,
                    "interval_start_monotonic_ns": tick,
                    "interval_end_monotonic_ns": tick + 10,
                    "observer_cpu_ns": 1,
                },
            ]
        )
        tick += 20
    return rows


def build_artifact_for_test() -> JsonDict:
    """Build one deterministic terminal-shaped fixture through production reducers."""

    selection = freeze_game_panel(("g1", "g2", "g3", "g4", "g5"))
    schedule = build_schedule(selection["games"])
    episodes = [
        {
            **deepcopy(row),
            "disposition": "complete",
            "action_count": 2,
            "start_level": 0,
            "peak_level": 0,
            "terminal_level": 0,
            "action_rows": [
                {"action_index": 1, "level": 0, "later_progress": False},
                {"action_index": 2, "level": 0, "later_progress": False},
            ],
            "request_budget_receipt": {
                "attempted": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "in_flight": 0,
                "callback_rows": [],
            },
            "solve_provenance": "no_level_reached",
            "trace_reproduction": {"attempted": False, "passed": False},
            "new_level_credit": 0,
            "elapsed_s": 1.0,
            "error": None,
        }
        for row in schedule
    ]
    events = _fixture_events(schedule[0]["episode_id"])
    shard = {
        "episode_id": schedule[0]["episode_id"],
        "path": "inline_fixture",
        "sha256": canonical_hash(events),
        "row_count": len(events),
        "inline_rows": events,
    }
    zero_events: list[JsonDict] = []
    return build_terminal_artifact(
        started_at_utc="2026-09-21T00:00:00Z",
        ended_at_utc="2026-09-21T00:00:01Z",
        duration_s=1.0,
        phase_spans=[{"phase": "fixture", "duration_s": 1.0, "completed_units": 8}],
        preconditions=[{"check": "fixture", "passed": True}],
        source_hashes={},
        selection=selection,
        registry_receipt={"passed": True, "new_credit_allowed": False},
        schedule=schedule,
        episode_rows=episodes,
        seam_shards=[shard],
        boundary_events=zero_events,
        runtime_receipt={"child_terminal": True},
        model_specs=[],
        validation_receipts=[],
        require_terminal=False,
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed Exp7303 commands through the Exp7358 scope helper."""

    return validation_contract.build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject any expansion beyond the frozen affected-file manifest."""

    return validation_contract.validate_command_plan(root, VALIDATION_MANIFEST, commands)


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build cold replay, reduction, and unchanged strict-reader commands."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", wrapper, "--replay", str(candidate)),
            "exact candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", wrapper, "--replay", str(candidate), "--reduce-only"),
            "exact candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact candidate",
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
            "exact candidate",
            300.0,
        ),
    ]


class EpisodeTimeout(Exception):
    """Stop one policy episode at its fixed wall-clock boundary."""


def _reproduce_trace(  # pragma: no cover - real public environment replay.
    game: str, trace: Sequence[Mapping[str, Any]], target_level: int
) -> JsonDict:
    """Replay only the agent's own actions on a fresh public environment."""

    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit

    arcade = kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    peak = 0
    error: str | None = None
    try:
        for row in trace:
            name = str(row.get("action"))
            data = row.get("data")
            latest = (
                env.reset() if name == "RESET" else env.step(getattr(GameAction, name), data=data)
            )
            peak = max(peak, live_support._level(latest))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
    return {
        "attempted": True,
        "trace_length": len(trace),
        "target_level": target_level,
        "reproduced_peak_level": peak,
        "passed": error is None and peak >= target_level,
        "error": error,
    }


def _disable_cross_game_loaders() -> tuple[Any, ...]:  # pragma: no cover - live harness guard.
    """Replace stored cross-game readers only inside the owned child process."""

    from carnot.agentic import arc_competition_agent as agent

    originals = (
        agent._recommend_live_approach,
        agent.load_cross_game_value_head,
        agent._load_submitted_candidate_router,
        agent._load_submitted_goal_energy_bias,
    )
    agent._recommend_live_approach = lambda _game: {
        "strategy": {"uses_goal_distance_heuristic": False},
        "source": "exp7471_runtime_observations_only",
    }
    agent.load_cross_game_value_head = lambda: None
    agent._load_submitted_candidate_router = lambda game_id: None
    agent._load_submitted_goal_energy_bias = lambda: None
    return originals


def _restore_cross_game_loaders(originals: tuple[Any, ...]) -> None:  # pragma: no cover
    """Restore child-local module functions after one episode."""

    from carnot.agentic import arc_competition_agent as agent

    (
        agent._recommend_live_approach,
        agent.load_cross_game_value_head,
        agent._load_submitted_candidate_router,
        agent._load_submitted_goal_energy_bias,
    ) = originals


def _run_policy_episode(  # pragma: no cover - real public environment and model integration.
    schedule: Mapping[str, Any], proposer: Any, capture: Any, event_path: Path, action_path: Path
) -> JsonDict:
    """Run one actual E3 episode and persist action and seam boundaries."""

    from arcengine import GameAction
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import make_carnot_agent

    episode_id = str(schedule["episode_id"])
    game = str(schedule["game"])
    seed = int(schedule["seed"])
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    os.environ["CARNOT_ARC_RANDOM_SEED"] = str(seed)
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(seed)
    os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW"] = str(SUPERVISOR_THRESHOLD)
    os.environ.pop("CARNOT_ARC_TRAJECTORY_SUPERVISOR", None)
    os.environ.pop("CARNOT_ARC_SUPERVISOR_TOOL_ARM", None)

    episode_dir = event_path.parent / "episodes" / episode_id.replace(":", "__")
    episode_dir.mkdir(parents=True, exist_ok=True)
    seam_path = episode_dir / "seam_events.jsonl"
    old_e3_dir = e3.E3_DIR
    e3.E3_DIR = episode_dir / "fresh_e3"
    capture.begin_episode(episode_id)
    budget = live_support.DurableEpisodeRequestBudget(
        episode_id,
        limit=REQUEST_LIMIT,
        deadline_s=EPISODE_LIMIT_S,
        event_path=event_path,
    )
    attach_request_budget(proposer, budget)

    class LocalAgentBase:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

    originals = _disable_cross_game_loaders()
    try:
        agent_type = make_carnot_agent(LocalAgentBase, cascade=True, proposer=proposer)
        agent = agent_type(game_id=game)
    finally:
        _restore_cross_game_loaders(originals)
    policy = agent._policy
    observer = E3SeamObserver(episode_id, seam_path)
    observer.install(policy)
    arcade = kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    frames: list[Any] = []
    latest: Any = None
    action_rows: list[JsonDict] = []
    trace: list[JsonDict] = []
    entered = time.monotonic()
    start_level: int | None = None
    peak_level = 0
    terminal_level = 0
    disposition = "complete"
    error: str | None = None

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise EpisodeTimeout(f"episode exceeded {EPISODE_LIMIT_S}s")

    previous_alarm = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, EPISODE_LIMIT_S)
    try:
        for action_index in range(ACTION_LIMIT):
            if agent.is_done(frames, latest):
                break
            action_started = time.monotonic_ns()
            _append_jsonl(
                action_path,
                {
                    "event": "action_start",
                    "episode_id": episode_id,
                    "action_index": action_index + 1,
                    "monotonic_ns": action_started,
                },
            )
            action = agent.choose_action(frames, latest)
            name = str(getattr(action, "name", action))
            data_value = getattr(action, "action_data", None)
            data = data_value.model_dump() if hasattr(data_value, "model_dump") else None
            if isinstance(data, Mapping):
                data = {key: value for key, value in data.items() if key != "game_id"}
            latest = (
                env.reset()
                if name == "RESET"
                else env.step(action if isinstance(action, GameAction) else action, data=data)
            )
            level = live_support._level(latest)
            start_level = level if start_level is None else start_level
            peak_level = max(peak_level, level)
            terminal_level = level
            trace.append({"action": name, "data": deepcopy(data)})
            row = {
                "episode_id": episode_id,
                "action_index": action_index + 1,
                "action": name,
                "data": deepcopy(data),
                "interval_start_monotonic_ns": action_started,
                "interval_end_monotonic_ns": time.monotonic_ns(),
                "level": level,
                "state_sha256": canonical_hash(str(latest)),
                "later_progress": False,
            }
            action_rows.append(row)
            _append_jsonl(action_path, {"event": "action_end", **row})
            frames.append(latest)
    except EpisodeTimeout as exc:
        disposition = "censored_timeout"
        error = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        disposition = "complete_error" if action_rows else "censored_no_first_action"
        error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        budget.cancel("episode_terminal")
        e3.E3_DIR = old_e3_dir

    for index, row in enumerate(action_rows):
        row["later_progress"] = any(
            int(later.get("level") or 0) > int(row.get("level") or 0)
            for later in action_rows[index + 1 :]
        )
    reached = start_level is not None and peak_level > start_level
    reproduction = (
        _reproduce_trace(game, trace, peak_level)
        if reached
        else {"attempted": False, "passed": False, "reason": "no_level_reached"}
    )
    solve_provenance = (
        "live_agent_self_discovery"
        if reached and reproduction.get("passed") is True
        else "unreproduced_live_progress"
        if reached
        else "no_level_reached"
    )
    return {
        **deepcopy(dict(schedule)),
        "disposition": disposition,
        "policy_entry": {
            "factory": "make_carnot_agent",
            "policy_class": type(policy).__name__,
            "choose_action_path": True,
            "is_done_path": True,
            "observation_local_to_harness": True,
            "withheld_inputs": list(WITHHELD_INPUTS),
        },
        "action_count": len(action_rows),
        "start_level": start_level,
        "peak_level": peak_level if start_level is not None else None,
        "terminal_level": terminal_level if start_level is not None else None,
        "action_rows": action_rows,
        "request_budget_receipt": budget.receipt(),
        "server_request_rows": live_support._transport_rows(
            live_support._read_jsonl(event_path), episode_id
        ),
        "elapsed_s": round(time.monotonic() - entered, 6),
        "solve_provenance": solve_provenance,
        "trace_reproduction": reproduction,
        "new_level_credit": 0,
        "seam_event_path": str(seam_path),
        "observer_errors": list(observer.errors),
        "error": error,
    }


def session_environment(
    base: Mapping[str, str], *, gpu_index: int, port: int, raw_dir: Path
) -> dict[str, str]:  # pragma: no cover - native child environment.
    """Build the adapter-withheld, runtime-observation-only child environment."""

    env = live_support.session_environment(base, gpu_index=gpu_index, port=port, raw_dir=raw_dir)
    for key in (
        "CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED",
        "CARNOT_ARC_PLAYBOOK_RETRIEVAL",
        "CARNOT_ARC_RUN_LOCAL_ADAPTATION",
        "CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY",
        "CARNOT_ARC_SUPPLY_WIN_TRANSITION",
        "CARNOT_ARC_SGE_CANDIDATE_ROUTER",
        "CARNOT_ARC_SUPERVISOR_TOOL_ARM",
    ):
        env.pop(key, None)
    env.update(
        {
            "CARNOT_FORCE_LIVE": "1",
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(MAX_NEW_TOKENS),
            "CARNOT_ARC_INDUCE_TIMEOUT": str(int(EPISODE_LIMIT_S)),
            "CARNOT_ARC_MAX_REFINEMENT_ROUNDS": str(REQUEST_LIMIT),
            "CARNOT_ARC_INDUCE_TOOL_TURNS": str(REQUEST_LIMIT),
            "CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW": str(SUPERVISOR_THRESHOLD),
            BOUNDARY_LEDGER_ENV: str(REPO_ROOT / BOUNDARY_PATH),
        }
    )
    return env


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover - live child.
    """Load one owned Qwen server and run all eight frozen episodes."""

    started = time.monotonic()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    event_path = raw_dir / RUNTIME_EVENT_PATH.name
    action_path = raw_dir / ACTION_PATH.name
    schedule = load_object(Path(args.schedule_path)).get("rows") or []
    capture = live_support.DurableRequestCapture(raw_dir, event_path)
    proposer: Any = None
    rows: list[JsonDict] = []
    session: JsonDict = {
        "child_pid": os.getpid(),
        "model_loaded": False,
        "model_invoked": False,
        "episodes": rows,
        "runtime_receipt": {},
        "error": None,
    }
    try:
        capture.install()
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

        progress(started, "model_load", "before", model_path=args.model_path)
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.8-27B",
            model_path=live_support._absolute_model_path(args.model_path),
            port=int(args.port),
            mtp=False,
            kv_quant="q8_0",
            use_chat_template=True,
            n_gpu_layers=999,
            n_ctx=49_152,
            max_tokens=MAX_NEW_TOKENS,
            timeout=int(EPISODE_LIMIT_S),
            tries=1,
        )
        proposer.model_repository = MODEL_ID
        proposer.model_revision = str(args.model_revision)
        proposer.requested_model_filename = MODEL_FILENAME
        proposer.requested_model_path = live_support._absolute_model_path(args.model_path)
        if not proposer._ensure_server():
            raise RuntimeError("owned native CUDA llama-server failed to start")
        session["model_loaded"] = True
        progress(started, "model_load", "after", server_pid=getattr(proposer._proc, "pid", None))
        session["runtime_receipt"] = {
            "child_pid": os.getpid(),
            "server_pid": getattr(proposer._proc, "pid", None),
            "native_binary": proposer.last_launch_argv[0] if proposer.last_launch_argv else None,
            "server_command": list(proposer.last_launch_argv),
            "requested_n_gpu_layers": 999,
            "observed_cuda_offload": bool("-ngl" in proposer.last_launch_argv),
            "n_ctx": 49_152,
            "kv_quantization": "q8_0",
            "embedded_tokenizer": True,
            "use_chat_template": True,
            "mtp": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "request_limit_per_episode": REQUEST_LIMIT,
        }
        current_work_receipt.atomic_json(
            Path(args.checkpoint_path),
            {
                "stage": "model_loaded",
                "model_loaded": True,
                "completed_units": 0,
                "server_pid": getattr(proposer._proc, "pid", None),
            },
        )
        for index, sealed in enumerate(schedule):
            progress(
                started,
                "episode",
                "before_benchmark",
                episode_id=sealed["episode_id"],
                completed_units=index,
            )
            row = _run_policy_episode(sealed, proposer, capture, event_path, action_path)
            rows.append(row)
            current_work_receipt.atomic_json(raw_dir / "episode_rows.json", {"rows": rows})
            current_work_receipt.atomic_json(
                Path(args.checkpoint_path),
                {
                    "stage": "episodes",
                    "model_loaded": True,
                    "completed_units": len(rows),
                    "total_units": len(schedule),
                },
            )
            progress(
                started,
                "episode",
                "after_benchmark",
                episode_id=sealed["episode_id"],
                completed_units=len(rows),
                disposition=row["disposition"],
            )
        session["model_invoked"] = bool(
            InvocationBoundaryLedger(REPO_ROOT / BOUNDARY_PATH).read_events()
        )
    except BaseException as exc:
        session["error"] = f"{type(exc).__name__}: {exc}"[:500]
        progress(started, "live_child", "error", error=session["error"])
    finally:
        capture.restore()
        progress(started, "model_unload", "before")
        if proposer is not None:
            proposer.stop()
        progress(started, "model_unload", "after")
        session["duration_s"] = time.monotonic() - started
        current_work_receipt.atomic_json(Path(args.session_path), session)
        current_work_receipt.atomic_json(
            Path(args.checkpoint_path),
            {
                "stage": "child_terminal",
                "model_loaded": session["model_loaded"],
                "completed_units": len(rows),
                "terminal_child": True,
            },
        )
    return 0


def _owned_server_vram_mb(server_pid: int | None, gpu_uuid: str) -> int | None:  # pragma: no cover
    """Read actual CUDA memory charged to the owned llama-server process."""

    if not isinstance(server_pid, int):
        return None
    command = (
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    )
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0] == str(server_pid) and parts[2] == gpu_uuid:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def _gpu_used_mb(gpu_uuid: str) -> int | None:  # pragma: no cover
    """Read total device memory so unload evidence does not assume zero."""

    command = (
        "nvidia-smi",
        "--query-gpu=uuid,memory.used",
        "--format=csv,noheader,nounits",
    )
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == gpu_uuid:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def run_child_with_lease(
    *, resources: Mapping[str, Any], schedule_path: Path, started: float
) -> JsonDict:  # pragma: no cover - owned process boundary.
    """Acquire one GPU lease and supervise only this experiment's child."""

    from carnot.gpu_lease_phase_journal import GpuLease

    gpu = dict(resources["gpu"])
    lease = GpuLease.acquire(
        runtime_dir=REPO_ROOT / RAW_DIR / "gpu_lease",
        task_id=TASK_ID,
        device_uuid=str(gpu.get("uuid")),
        expected_model=str(resources["model_path"]),
        vram_before_mb=int(gpu.get("total_memory_mb") or 0) - int(gpu.get("free_memory_mb") or 0),
        ttl_s=90,
    )
    lease.transition("admitted")
    lease.transition("loading")
    port = live_support._free_port()
    model_spec = dict(resources.get("model_spec") or {})
    revision = str(model_spec.get("revision") or model_spec.get("model_revision") or "unknown")
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / WRAPPER_PATH),
        "--role",
        "live-session",
        "--date",
        RUN_DATE,
        "--model-path",
        str(resources["model_path"]),
        "--model-hash",
        str(resources["model_hash"]),
        "--model-revision",
        revision,
        "--gpu-index",
        str(gpu["index"]),
        "--port",
        str(port),
        "--schedule-path",
        str(schedule_path),
        "--raw-dir",
        str(REPO_ROOT / RAW_DIR),
        "--checkpoint-path",
        str(REPO_ROOT / CHECKPOINT_PATH),
        "--session-path",
        str(REPO_ROOT / SESSION_PATH),
    ]
    env = session_environment(
        os.environ, gpu_index=int(gpu["index"]), port=port, raw_dir=REPO_ROOT / RAW_DIR
    )
    env["CARNOT_ARC_GGUF_PATH"] = str(resources["model_path"])
    env["CARNOT_LLAMA_SERVER"] = str(resources["server"])
    progress(started, "live_subprocess", "before", command=" ".join(command))
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, start_new_session=True)
    child_started = time.monotonic()
    next_heartbeat = child_started
    timed_out = False
    resident = False
    resident_vram_mb: int | None = None
    while process.poll() is None:
        now = time.monotonic()
        checkpoint = load_object(REPO_ROOT / CHECKPOINT_PATH)
        if checkpoint.get("model_loaded") is True and not resident:
            resident_vram_mb = _owned_server_vram_mb(
                checkpoint.get("server_pid"), str(gpu.get("uuid"))
            )
            if resident_vram_mb is not None:
                lease.transition("resident", vram_mb=resident_vram_mb)
                lease.transition("inferencing")
                resident = True
        if now - child_started >= AGGREGATE_LIVE_LIMIT_S:
            timed_out = True
            break
        if now >= next_heartbeat:
            lease.heartbeat()
            progress(
                started,
                "live_subprocess",
                "heartbeat",
                completed_units=int(checkpoint.get("completed_units") or 0),
                model_loaded=bool(checkpoint.get("model_loaded")),
                pending_operation=checkpoint.get("stage", "child_startup"),
            )
            next_heartbeat = now + 45.0
        time.sleep(0.5)
    signals_sent: list[str] = []
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        signals_sent.append("SIGTERM:owned_process_group")
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            signals_sent.append("SIGKILL:owned_process_group")
            process.wait(timeout=10)
    progress(
        started, "live_subprocess", "after", returncode=process.returncode, timed_out=timed_out
    )
    session = load_object(REPO_ROOT / SESSION_PATH)
    if not session:
        session = {
            "child_pid": process.pid,
            "model_loaded": False,
            "model_invoked": bool(
                InvocationBoundaryLedger(REPO_ROOT / BOUNDARY_PATH).read_events()
            ),
            "episodes": list(
                load_object(REPO_ROOT / RAW_DIR / "episode_rows.json").get("rows") or []
            ),
            "error": "live_child_did_not_write_session",
        }
    if session.get("model_loaded") and not resident:
        resident_vram_mb = _owned_server_vram_mb(
            (session.get("runtime_receipt") or {}).get("server_pid"), str(gpu.get("uuid"))
        )
        if resident_vram_mb is not None:
            lease.transition("resident", vram_mb=resident_vram_mb)
            lease.transition("inferencing")
            resident = True
    if session.get("model_loaded") and resident:
        lease.transition("unloading")
        after_vram = _gpu_used_mb(str(gpu.get("uuid")))
        lease.transition(
            "validating",
            vram_mb=int(after_vram if after_vram is not None else 0),
            exit_code=int(process.returncode or 0),
            unload_observed=after_vram is not None,
        )
        lease.transition("terminal_complete")
    else:
        lease.transition("terminal_blocked")
    release = lease.release()
    runtime = dict(session.get("runtime_receipt") or {})
    runtime.update(
        {
            "gpu_uuid": gpu.get("uuid"),
            "gpu_index": gpu.get("index"),
            "gpu_name": gpu.get("name"),
            "lease_owner": lease.owner_receipt(),
            "lease_release": release,
            "fresh_lease": True,
            "signals_sent": signals_sent,
            "timed_out": timed_out,
            "child_returncode": process.returncode,
            "child_terminal": True,
            "observed_cuda_offload": {
                "server_pid": (session.get("runtime_receipt") or {}).get("server_pid"),
                "gpu_uuid": gpu.get("uuid"),
                "owned_server_vram_mb": resident_vram_mb,
                "measurement": "nvidia-smi_compute_process_used_gpu_memory",
                "passed": isinstance(resident_vram_mb, int) and resident_vram_mb > 0,
            },
        }
    )
    session["runtime_receipt"] = runtime
    session["timed_out"] = timed_out
    current_work_receipt.atomic_json(REPO_ROOT / SESSION_PATH, session)
    return session


def _runtime_preconditions(  # pragma: no cover - native CUDA and model-cache integration.
    root: Path, started: float
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Resolve the current cached model, tokenizer, native runner, and CUDA device."""

    checks, hashes, resources = live_support._runtime_preconditions(root, started)
    return [dict(row) for row in checks], dict(hashes), dict(resources)


def _phase(
    spans: list[JsonDict],
    name: str,
    phase_start: float,
    run_start: float,
    completed_units: int,
    checkpoint: str | None = None,
) -> None:
    now = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_s": round(phase_start - run_start, 6),
            "end_s": round(now - run_start, 6),
            "duration_s": round(now - phase_start, 6),
            "completed_units": completed_units,
            "checkpoint": checkpoint,
            "ended_at_utc": utc_now(),
        }
    )


def _seam_shards(root: Path, episodes: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Hash-bind every episode seam file without copying its large rows."""

    shards: list[JsonDict] = []
    for episode in episodes:
        raw = episode.get("seam_event_path")
        if not isinstance(raw, str):
            continue
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = str(path)
        shards.append(
            {
                "episode_id": episode.get("episode_id"),
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "row_count": len(read_jsonl(path)),
            }
        )
    return shards


def _write_callback_seam_shard(
    root: Path, gpu_device_id: Any
) -> JsonDict | None:  # pragma: no cover
    """Normalize already-durable callback boundaries and retain real token usage."""

    events = read_jsonl(root / RUNTIME_EVENT_PATH)
    if not events:
        return None
    permits: dict[tuple[str, int], str] = {}
    for event in events:
        if event.get("event") == "permit_acquired":
            permits[(str(event.get("episode_id")), int(event.get("reservation_index") or 0))] = str(
                event.get("request_id")
            )
    output = root / CALLBACK_SEAM_PATH
    output.unlink(missing_ok=True)
    row_count = 0
    for event in events:
        if event.get("event") not in {"server_request", "server_response", "server_error"}:
            continue
        episode_id = str(event.get("episode_id"))
        call_index = int(event.get("call_index") or 0)
        request_id = permits.get((episode_id, call_index))
        response: JsonDict = {}
        response_path = event.get("response_path")
        if isinstance(response_path, str):
            response = load_object(Path(response_path))
        usage = dict(response.get("usage") or {})
        started_s = float(event.get("request_started_monotonic") or 0.0)
        elapsed_s = float(event.get("elapsed_s") or 0.0)
        stage = "stage_start" if event.get("event") == "server_request" else "stage_end"
        row = {
            "schema": "carnot.arc.e3_decision_seam_event.v1",
            "episode_id": episode_id,
            "decision_id": f"{episode_id}:downstream_generation:{call_index}",
            "parent_decision_id": None,
            "request_id": request_id,
            "seam": "downstream_generation",
            "event": stage,
            "event_monotonic_ns": int(
                (started_s + (elapsed_s if stage == "stage_end" else 0.0)) * 1e9
            ),
            "clock_identity": "time.monotonic_ns",
            "interval_start_monotonic_ns": int(started_s * 1e9),
            "interval_end_monotonic_ns": (
                int((started_s + elapsed_s) * 1e9) if stage == "stage_end" else None
            ),
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "cpu_work_ns": 0,
            "gpu_work_ns": int(elapsed_s * 1e9),
            "gpu_device_id": gpu_device_id,
            "disposition": (
                "completed"
                if event.get("event") == "server_response"
                else "failed"
                if event.get("event") == "server_error"
                else "in_flight"
            ),
            "missing_options_reason": "not_a_candidate_set_seam",
        }
        _append_jsonl(output, row)
        row_count += 1
    if not output.is_file():
        return None
    return {
        "episode_id": "all",
        "seam": "downstream_generation",
        "path": CALLBACK_SEAM_PATH.as_posix(),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "row_count": row_count,
    }


def _blocked_artifact(
    *,
    started_at: str,
    duration_s: float,
    spans: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    registry_receipt: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]] = (),
    session: Mapping[str, Any] | None = None,
) -> JsonDict:  # pragma: no cover - external absence path.
    """Publish a schema-complete blocked result without invented model work."""

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=duration_s,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        registry_receipt=registry_receipt,
        schedule=schedule,
        episode_rows=list((session or {}).get("episodes") or []),
        seam_shards=_seam_shards(REPO_ROOT, list((session or {}).get("episodes") or [])),
        boundary_events=boundary_events,
        runtime_receipt=(session or {}).get("runtime_receipt") or {"child_terminal": True},
        model_specs=[],
        validation_receipts=receipts,
        require_terminal=False,
    )
    invocation = _invocation_reduction(boundary_events, child_terminal=True)
    artifact["status"] = (
        "model_load_no_generation"
        if invocation["inference_substrate_class"] == "model_load_no_generation"
        else "blocked_owned_live_failure"
        if invocation["model_invoked"]
        else "blocked_no_run"
    )
    artifact["honest_verdict"] = artifact["status"]
    artifact["verdict_class"] = "blocked"
    artifact["arc_observation_complete_score"] = 0
    artifact["gate_check_summary"] = gate_summary(checks)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - orchestration.
    """Run prechecks, scoped validation, live episodes, readers, and publish."""

    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    receipts: list[JsonDict] = []
    progress(started, "startup", "begin", run_date=run_date)

    phase_start = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes, registry = collect_preconditions(root)
    checks.insert(
        0,
        gate_row(
            "run_date",
            "validity",
            RUN_DATE,
            run_date,
            upstream="command_line",
            artifact_field="--date",
            principle="The run date is fixed by the protocol.",
        ),
    )
    selection = freeze_game_panel(accessible_games(root))
    schedule = build_schedule(selection["games"])
    registry_receipt = registry_precheck(registry, selection["games"])
    checks.extend(
        [
            gate_row(
                "four_game_stable_panel",
                "validity",
                True,
                selection["passed"] and len(schedule) == 8,
                upstream="environment_files",
                artifact_field="selection_receipt.games",
                principle="Four public games must be sealed before any current outcome.",
            ),
            gate_row(
                "registry_precheck",
                "validity",
                True,
                registry_receipt["passed"],
                upstream=REGISTRY_PATH.as_posix(),
                artifact_field="selected_games",
                principle="Prior credit is recorded before runtime and never sent to policy.",
            ),
        ]
    )
    current_work_receipt.atomic_json(
        root / SCHEDULE_PATH,
        {
            "selection_receipt": selection,
            "registry_precheck": registry_receipt,
            "rows": schedule,
        },
    )
    _phase(spans, "preconditions", phase_start, started, len(checks), SCHEDULE_PATH.as_posix())
    progress(started, "preconditions", "after", passed=all(row["passed"] for row in checks))
    if not all(row["passed"] for row in checks):
        artifact = _blocked_artifact(
            started_at=started_at,
            duration_s=time.monotonic() - started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            schedule=schedule,
            selection=selection,
            registry_receipt=registry_receipt,
            receipts=receipts,
        )
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        return artifact

    private = Path(tempfile.mkdtemp(prefix="exp7471-validation-", dir="/tmp"))
    phase_start = time.monotonic()
    progress(started, "validation", "before_affected_subprocesses")
    plan = build_validation_plan(root, private / "scoped")
    plan_errors = validate_validation_plan(root, plan)
    if plan_errors:
        raise RuntimeError(f"validation command plan drift: {plan_errors}")
    receipts.extend(
        validation_contract.run_categorized_commands(
            root,
            [validation_contract.PlannedCommand(row, "required_validation", True) for row in plan],
            log_dir=root / RAW_DIR / "validation/affected",
            heartbeat_s=60.0,
        )
    )
    _phase(spans, "affected_validation", phase_start, started, len(receipts))
    progress(started, "validation", "after_affected_subprocesses", completed_units=len(receipts))
    if not _receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES):
        artifact = _blocked_artifact(
            started_at=started_at,
            duration_s=time.monotonic() - started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            schedule=schedule,
            selection=selection,
            registry_receipt=registry_receipt,
            receipts=receipts,
        )
        artifact["status"] = "complete_disqualified_affected_validation"
        artifact["honest_verdict"] = artifact["status"]
        artifact["verdict_class"] = "disqualified"
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        return artifact

    phase_start = time.monotonic()
    progress(started, "runtime_preconditions", "before_model_cuda_lease_checks")
    runtime_checks, runtime_hashes, resources = _runtime_preconditions(root, started)
    checks.extend(runtime_checks)
    hashes.update(runtime_hashes)
    _phase(spans, "runtime_preconditions", phase_start, started, len(runtime_checks))
    progress(
        started,
        "runtime_preconditions",
        "after_model_cuda_lease_checks",
        passed=all(row.get("passed") is True for row in checks),
    )
    if not all(row.get("passed") is True for row in checks):
        artifact = _blocked_artifact(
            started_at=started_at,
            duration_s=time.monotonic() - started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            schedule=schedule,
            selection=selection,
            registry_receipt=registry_receipt,
            receipts=receipts,
        )
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        return artifact

    for path in (
        root / BOUNDARY_PATH,
        root / RUNTIME_EVENT_PATH,
        root / ACTION_PATH,
        root / CALLBACK_SEAM_PATH,
        root / SESSION_PATH,
        root / CHECKPOINT_PATH,
        root / TERMINAL_CANDIDATE_PATH,
        root / RAW_DIR / "episode_rows.json",
    ):
        path.unlink(missing_ok=True)
    phase_start = time.monotonic()
    progress(started, "live", "before_model_load_generation_benchmark", planned_units=8)
    session = run_child_with_lease(
        resources=resources, schedule_path=root / SCHEDULE_PATH, started=started
    )
    episodes = [dict(row) for row in session.get("episodes", []) if isinstance(row, Mapping)]
    progress(
        started,
        "live",
        "after_model_load_generation_benchmark",
        completed_units=len(episodes),
    )
    _phase(
        spans,
        "live_model_and_episodes",
        phase_start,
        started,
        len(episodes),
        SESSION_PATH.as_posix(),
    )
    boundary_events = InvocationBoundaryLedger(root / BOUNDARY_PATH).read_events()
    seam_shards = _seam_shards(root, episodes)
    callback_shard = _write_callback_seam_shard(
        root, (session.get("runtime_receipt") or {}).get("gpu_uuid")
    )
    if callback_shard is not None:
        seam_shards.append(callback_shard)
    source_hashes = deepcopy(hashes)
    for relative, role in (
        (SCHEDULE_PATH, "frozen_protocol"),
        (MODULE_PATH, "producer_code"),
        (WRAPPER_PATH, "declared_entrypoint"),
        (TEST_PATH, "new_behavior_tests"),
        (RUNTIME_EVENT_PATH, "callback_boundary_shard"),
        (CALLBACK_SEAM_PATH, "normalized_callback_seam_shard"),
        (ACTION_PATH, "action_boundary_shard"),
    ):
        path = root / relative
        if path.is_file():
            source_hashes[relative.as_posix()] = _source_record(path, role)

    phase_start = time.monotonic()
    candidate = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=source_hashes,
        selection=selection,
        registry_receipt=registry_receipt,
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
        raise RuntimeError(f"measured candidate invalid: {candidate_errors}")
    current_work_receipt.atomic_json(root / TERMINAL_CANDIDATE_PATH, candidate)
    _phase(
        spans,
        "independent_reduction",
        phase_start,
        started,
        len(candidate["rows"]),
        TERMINAL_CANDIDATE_PATH.as_posix(),
    )

    phase_start = time.monotonic()
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        candidate=TERMINAL_CANDIDATE_PATH,
    )
    terminal = validation_scope.run_commands(
        root,
        terminal_command_specs(root, root / TERMINAL_CANDIDATE_PATH),
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    receipts.extend(terminal)
    progress(started, "terminal_validation", "after_subprocesses", completed_units=len(terminal))
    _phase(spans, "terminal_validation", phase_start, started, len(terminal))

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=source_hashes,
        selection=selection,
        registry_receipt=registry_receipt,
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
        raise RuntimeError(f"terminal artifact invalid: {errors}")
    progress(started, "publish", "before_atomic_terminal_write", path=RESULT_PATH)
    current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "publish",
        "after_atomic_terminal_write",
        path=RESULT_PATH,
        observation=artifact["arc_observation_complete_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public experiment, private child, and cold replay roles."""

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
        parser.error("--date is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the host experiment, owned child, or cold replay."""

    args = parse_args(argv)
    if args.replay is not None:
        artifact = load_object(args.replay)
        reduced = independent_reduce(artifact)
        errors = [] if args.reduce_only else validate_artifact(artifact, require_terminal=False)
        print(
            json.dumps({"reduced": reduced, "validation_errors": errors}, sort_keys=True),
            flush=True,
        )
        return int(bool(errors) or reduced.get("matches_declared") is not True)
    if args.role == "live-session":
        return run_live_session(args)
    artifact = run_experiment(REPO_ROOT, str(args.date))
    return 0 if artifact.get("verdict_class") in {"null", "blocked"} else 1


__all__ = [
    "ACTION_LIMIT",
    "AGGREGATE_LIVE_LIMIT_S",
    "EPISODE_LIMIT_S",
    "E3SeamObserver",
    "INFERENCE_SUBSTRATE_CLASS",
    "MAX_NEW_TOKENS",
    "MODEL_SPECS",
    "REQUEST_LIMIT",
    "REQUIRED_ARTIFACT_FIELDS",
    "SPEC_PATH",
    "SUPERVISOR_ARM_ORDER",
    "UPSTREAM_PATH",
    "build_artifact_for_test",
    "build_schedule",
    "collect_preconditions",
    "freeze_game_panel",
    "independent_reduce",
    "main",
    "read_jsonl",
    "reduce_panel",
    "validate_artifact",
]
