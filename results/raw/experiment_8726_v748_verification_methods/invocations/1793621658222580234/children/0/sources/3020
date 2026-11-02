"""Capture live supervisor eligibility without changing ARC policy behavior.

The runtime recorder is a passive, default-off observer. The experiment runner
uses it in LLM-off public-environment episodes to repair the eligibility gap
reported by Exp7512. Public episodes remain a hidden-game proxy, not solve
evidence.

Spec: REQ-ARC-7526 and SCENARIO-ARC-7526-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time
from typing import Any

Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.658"
EXPERIMENT_ID = "exp7526-arc-eligibility"
SCHEMA = "carnot.exp7526.v658.arc_eligibility.v1"
MODEL_SPECS: list[str] = []
RECORDER_ENV_FLAG = "CARNOT_ARC_ELIGIBILITY_RECEIPT"
RECORDER_PATH_ENV = "CARNOT_ARC_ELIGIBILITY_RECEIPT_PATH"
MAX_RECEIPT_ROWS = 4096
E6_PANEL_GAMES = (
    "sb26",
    "vc33",
    "su15",
    "g50t",
    "m0r0",
    "dc22",
    "wa30",
    "ka59",
    "bp35",
    "sp80",
    "ft09",
    "ar25",
)
SELECTION_SEED = 658_026
EPISODE_SEEDS = (658_027, 658_028)
SHIPPED_SUPERVISOR_WINDOW = 120
EPISODE_CAP_S = 180
COLLECTION_CAP_S = 3000
ACTION_CAP = max(840, 2 * SHIPPED_SUPERVISOR_WINDOW + 40)
EXCLUSIVE_TIMING_HELPER = "carnot.experiment_7491_e6_timed_live_profile.E6TimedObserver"
REPOSITORY_HEALTH_OBSERVATION = {
    "command_argv": [".venv/bin/pytest", "tests/python", "-q"],
    "status": "interrupted_after_observed_failures",
    "observed_progress_percent": 7,
    "observed_failure_markers_at_least": 1,
    "duration_s_at_interrupt": 180,
    "required_for_current_result": False,
    "classification": "unscoped_repository_health_observation",
    "output_persisted": False,
    "reason": "The unscoped suite projected about 50 minutes and already showed failures; scoped checks remain the current validity gate.",
}
ARM_SELECTION_ORDER = (
    "drop_goal_bias",
    "allow_reinduction",
    "tool_loop_reinduction",
    "force_exploration_diversity",
)

SPEC_PATH = Path("openspec/capabilities/arc-agi/spec.md")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
PRIOR_PATH = Path("results/experiment_7512_v657_arc_opportunity.json")
RESULT_PATH = Path("results/experiment_7526_v658_arc_eligibility.json")
RAW_DIR = Path("results/raw/experiment_7526_v658_arc_eligibility")
CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7526_v658_arc_eligibility.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7526_v658_arc_eligibility.py")
TEST_PATH = Path("tests/python/test_experiment_7526_v658_arc_eligibility.py")


def utc_now() -> str:
    """Return one aware UTC timestamp for an observed process boundary."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(  # pragma: no cover - exercised by the declared process entrypoint.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Print a flushed boundary so a bounded live collection is observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7526] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so a changed receipt cannot replay as identical."""

    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def policy_redirect_state(policy: Any) -> Json:
    """Read only state that the existing redirect arms are allowed to mutate."""

    explorer = getattr(policy, "explorer", None)
    return {
        "goal_bias_present": getattr(explorer, "goal_bias", None) is not None,
        "induced": bool(getattr(policy, "induced", False)),
        "hybrid_diversity": bool(getattr(explorer, "_hybrid_diversity", False)),
        "steps_since_progress": getattr(explorer, "_steps_since_progress", None),
    }


class NoOpEligibilityRecorder:
    """Disabled recorder whose methods preserve every caller value."""

    enabled = False
    error_count = 0
    dropped_rows = 0

    def record_selection(self, **_kwargs: Any) -> None:
        return None

    def record_observation_failure(self, **_kwargs: Any) -> None:
        return None


NOOP_ELIGIBILITY_RECORDER = NoOpEligibilityRecorder()


class EligibilityReceiptRecorder:
    """Append bounded live-boundary rows and never raise into policy code."""

    enabled = True

    def __init__(self, game_id: str, *, path: Path | str, max_rows: int = MAX_RECEIPT_ROWS):
        self.game_id = str(game_id)
        self.path = Path(path)
        self.max_rows = max(1, int(max_rows))
        self.error_count = 0
        self.dropped_rows = 0
        self._written = 0

    def _append(self, row: Mapping[str, Any]) -> None:
        try:
            if self._written >= self.max_rows:
                self.dropped_rows += 1
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._written += 1
        except Exception:  # pragma: no cover - filesystem failure remains fail-open.
            self.error_count += 1

    def record_selection(
        self,
        *,
        supervisor: Any,
        snapshot: Any,
        predicates: Sequence[Mapping[str, Any]],
        redirect: Any,
        mode: str,
        applied: bool,
        application_disposition: str | None = None,
        old_state_hash: str | None = None,
        new_state_hash: str | None = None,
    ) -> None:
        """Write one selection row after the existing selector and optional mutation."""

        try:
            selected_arm = getattr(redirect, "arm", None)
            arm_rows = []
            for predicate in predicates:
                arm = str(predicate.get("arm", ""))
                arm_rows.append(
                    {
                        "arm": arm,
                        "enabled": bool(predicate.get("enabled", False)),
                        "eligible": predicate.get("eligible"),
                        "selected": selected_arm == arm,
                        "mode": str(mode),
                        "applied": bool(applied and selected_arm == arm),
                        "diagnosis": str(predicate.get("diagnosis", "")),
                    }
                )
            self._append(
                {
                    "schema": "carnot.arc.supervisor_eligibility_receipt.v1",
                    "game": self.game_id,
                    "action_id": int(getattr(supervisor, "_actions_total", 0)),
                    "level_id": int(getattr(snapshot, "level")),
                    "mode": str(mode),
                    "observation_status": "complete",
                    "predicate_inputs": asdict(snapshot),
                    "arm_rows": arm_rows,
                    "selected_arm": selected_arm,
                    "applied": bool(applied),
                    "application_disposition": application_disposition
                    or ("applied" if applied else "not_attempted"),
                    "old_state_hash": old_state_hash,
                    "new_state_hash": new_state_hash,
                }
            )
        except Exception:  # pragma: no cover - malformed caller objects remain fail-open.
            self.error_count += 1

    def record_observation_failure(
        self,
        *,
        mode: str,
        error: str,
        action_id: int | None = None,
        level_id: int | None = None,
    ) -> None:
        """Keep unreadable eligibility explicit instead of backfilling false."""

        tool_enabled = os.environ.get("CARNOT_ARC_SUPERVISOR_TOOL_ARM") == "1"
        self._append(
            {
                "schema": "carnot.arc.supervisor_eligibility_receipt.v1",
                "game": self.game_id,
                "action_id": action_id,
                "level_id": level_id,
                "mode": str(mode),
                "observation_status": "failed",
                "observation_error": str(error)[:200],
                "predicate_inputs": None,
                "arm_rows": [
                    {
                        "arm": arm,
                        "enabled": arm != "tool_loop_reinduction" or tool_enabled,
                        "eligible": None,
                        "selected": False,
                        "mode": str(mode),
                        "applied": False,
                        "diagnosis": "observation_failed",
                    }
                    for arm in ARM_SELECTION_ORDER
                ],
                "selected_arm": None,
                "applied": False,
                "application_disposition": "observation_failed",
                "old_state_hash": None,
                "new_state_hash": None,
            }
        )


def maybe_make_eligibility_recorder(game_id: str) -> Any:
    """Construct the recorder only for an exact opt-in with an explicit path."""

    if os.environ.get(RECORDER_ENV_FLAG) != "1":
        return NOOP_ELIGIBILITY_RECORDER
    raw_path = os.environ.get(RECORDER_PATH_ENV)
    if not raw_path:
        return NOOP_ELIGIBILITY_RECORDER
    return EligibilityReceiptRecorder(game_id, path=Path(raw_path))


def build_panel_manifest() -> Json:
    """Freeze the outcome-blind six-game schedule and shipped limits."""

    from carnot.experiment_7491_e6_timed_live_profile import E6TimedObserver

    if (  # pragma: no cover - import drift is an environment integrity failure.
        f"{E6TimedObserver.__module__}.{E6TimedObserver.__name__}" != EXCLUSIVE_TIMING_HELPER
    ):
        raise RuntimeError("exclusive_timing_helper_drift")
    ranked = sorted(
        E6_PANEL_GAMES,
        key=lambda game: (hashlib.sha256(f"{SELECTION_SEED}:{game}".encode()).hexdigest(), game),
    )
    games = ranked[:6]
    schedule = [
        {
            "episode_id": f"{game}:seed-{seed}",
            "game": game,
            "seed": seed,
            "order": index,
        }
        for index, (game, seed) in enumerate(
            (
                pair
                for game in games
                for pair in ((game, EPISODE_SEEDS[0]), (game, EPISODE_SEEDS[1]))
            )
        )
    ]
    return {
        "selection_seed": SELECTION_SEED,
        "selection_method": "ascending_sha256_of_seed_colon_game",
        "source_roster": list(E6_PANEL_GAMES),
        "ranked_roster": ranked,
        "games": games,
        "episode_seeds": list(EPISODE_SEEDS),
        "schedule": schedule,
        "supervisor_window": SHIPPED_SUPERVISOR_WINDOW,
        "action_cap": ACTION_CAP,
        "episode_cap_s": EPISODE_CAP_S,
        "collection_cap_s": COLLECTION_CAP_S,
        "adapter_policy_input": False,
        "proxy_class": "public_adapter_withheld_hidden_game_proxy",
        "official_hidden_score": False,
        "exclusive_timing_helper": EXCLUSIVE_TIMING_HELPER,
        "repository_health": deepcopy(REPOSITORY_HEALTH_OBSERVATION),
    }


def _load_object(path: Path) -> Json:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _source_row(root: Path, path: Path, role: str = "required_input") -> Json:
    target = root / path
    exists = target.is_file()
    return {
        "path": path.as_posix(),
        "role": role,
        "exists": exists,
        "bytes": target.stat().st_size if exists else 0,
        "sha256": _sha256_file(target) if exists else None,
    }


def collect_preconditions(root: Path) -> tuple[list[Json], Json]:
    """Authenticate named resources before any public-environment measurement."""

    required = (
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
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/agentic/arc_trajectory_supervisor.py"),
        Path("python/carnot/agentic/arc_arm_eligibility.py"),
        Path("python/carnot/agentic/arc_decision_telemetry.py"),
        Path("python/carnot/experiment_7491_e6_timed_live_profile.py"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        PRIOR_PATH,
        REGISTRY_PATH,
        Path("docs/research-notes/semif-ebm-arc-experiment-plan-2026-09-20.md"),
        SPEC_PATH,
    )
    sources = {path.as_posix(): _source_row(root, path) for path in required}
    checks = [
        {
            "check": f"required_file:{path.as_posix()}",
            "upstream": path.as_posix(),
            "field": "exists",
            "expected": True,
            "observed": sources[path.as_posix()]["exists"],
            "passed": sources[path.as_posix()]["exists"] is True,
        }
        for path in required
    ]
    prior = _load_object(root / PRIOR_PATH)
    for field, expected in (
        ("experiment_id", "exp7512-arc-opportunity"),
        ("verdict_class", "null"),
        ("model_invoked", False),
    ):
        observed = prior.get(field)
        checks.append(
            {
                "check": f"prior:{field}",
                "upstream": PRIOR_PATH.as_posix(),
                "field": field,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    return checks, sources


ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "forward_calls_attempted": 0,
    "forward_calls_completed": 0,
    "forward_calls_failed": 0,
    "forward_calls_cancelled": 0,
    "forward_calls_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}


def registry_precheck(root: Path, games: Sequence[str]) -> Json:
    """Read existing public clears before outcomes and propose no re-solve."""

    import yaml

    path = root / REGISTRY_PATH
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    registry_values = loaded.get("games", []) if isinstance(loaded, Mapping) else []
    registry_games = {
        str(row.get("game")): row
        for row in registry_values
        if isinstance(row, Mapping) and row.get("game") is not None
    }
    rows = []
    for game in games:
        value = registry_games.get(game)
        rows.append(
            {
                "game": game,
                "registered": isinstance(value, Mapping),
                "already_public_clear": isinstance(value, Mapping),
                "propose_resolve": False,
            }
        )
    return {
        "path": REGISTRY_PATH.as_posix(),
        "sha256": _sha256_file(path),
        "read_before_outcomes": True,
        "policy_received_registry_data": False,
        "rows": rows,
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    principle: str,
) -> Json:
    if op == "==":
        passed = observed == expected
    elif op == ">=":
        passed = isinstance(observed, (int, float)) and observed >= expected
    else:
        raise ValueError(f"unsupported_gate_operator:{op}")
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "failures": failures,
        "first_failure": failures[0] if failures else None,
        "required_validity_and_readiness_passed": all(
            row.get("passed") is True
            for row in gates
            if row.get("category") in {"validity", "readiness"}
        ),
    }


def independently_reduce_rows(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Recompute selection and application counts only from comparative rows."""

    explicit = sum(
        int(row.get("eligible_count", 0)) + int(row.get("ineligible_count", 0))
        if "eligible_count" in row
        else int(row.get("eligible") in {True, False})
        for row in rows
    )
    unknown = sum(
        int(row.get("unknown_eligibility_count", 0))
        if "unknown_eligibility_count" in row
        else int(row.get("eligible") is None)
        for row in rows
    )
    eligible = sum(
        int(row.get("eligible_count", 0))
        if "eligible_count" in row
        else int(row.get("eligible") is True)
        for row in rows
    )
    return {
        "row_count": len(rows),
        "explicit_eligibility_count": explicit,
        "unknown_eligibility_count": unknown,
        "eligible_count": eligible,
        "selected_count": sum(int(row.get("selected_count", 0)) for row in rows),
        "applied_count": sum(int(row.get("applied_count", 0)) for row in rows),
        "observation_count": sum(int(row.get("observation_count", 0)) for row in rows),
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: deepcopy(value)
        for key, value in artifact.items()
        if key not in {"duration_s", "ended_at_utc", "phase_spans", "validation_receipts"}
    }
    stable["reproducibility_checksum"] = ""
    return canonical_hash(stable)


def build_artifact(
    root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    registry_receipt: Mapping[str, Any] | None = None,
) -> Json:
    """Build one terminal record with readiness independent from measured benefit."""

    manifest = build_panel_manifest()
    reduced = independently_reduce_rows(rows)
    parity_passed = bool(parity_rows) and all(row.get("passed") is True for row in parity_rows)
    validation_passed = bool(validation_receipts) and all(
        row.get("passed") is True for row in validation_receipts if row.get("required", True)
    )
    readiness = int(
        reduced["observation_count"] > 0
        and reduced["unknown_eligibility_count"] == 0
        and parity_passed
        and validation_passed
    )
    completed = sum(row.get("disposition") == "complete" for row in episode_rows)
    failed = sum(row.get("disposition") == "failed" for row in episode_rows)
    censored = sum(str(row.get("disposition", "")).startswith("censored") for row in episode_rows)
    attempted = len(episode_rows)
    selected_support = int(reduced["selected_count"])
    applied_support = int(reduced["applied_count"])
    gates = [
        _gate(
            "required_scoped_and_capability_checks",
            "validity",
            1,
            int(validation_passed),
            "==",
            "Favorable science cannot excuse invalid evidence.",
        ),
        _gate(
            "live_boundary_eligibility_receipt_ready",
            "readiness",
            1,
            readiness,
            "==",
            "A valid null remains eligible for later auditing.",
        ),
        _gate(
            "selected_recommendation_support",
            "support",
            1,
            selected_support,
            ">=",
            "At least one selected arm is needed before discussing its outcome.",
        ),
        _gate(
            "applied_mutation_support",
            "benefit",
            1,
            applied_support,
            ">=",
            "A shadow recommendation cannot estimate the effect of a mutation.",
        ),
        _gate(
            "applied_episode_uncertainty_support",
            "benefit",
            10,
            sum(row.get("applied_count", 0) > 0 for row in rows),
            ">=",
            "An effect estimate needs repeated independent applied episodes.",
        ),
    ]
    required_valid = all(
        row["passed"] for row in gates if row["category"] in {"validity", "readiness"}
    )
    if not required_valid:
        verdict_class = "disqualified"
        honest = "complete_disqualified_required_validation_or_readiness_failed"
    elif not all(row["passed"] for row in gates if row["category"] in {"support", "benefit"}):
        verdict_class = "null"
        honest = "complete_null_eligibility_observed_without_applied_effect_support"
    else:
        verdict_class = "positive"
        honest = "complete_positive_applied_effect_supported"
    artifact: Json = {
        "schema": SCHEMA,
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "process_identity": {
            "pid": os.getpid(),
            "cwd": str(root.resolve()),
            "source_revision": _git_revision(root),
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "execution_venue_detail": "host_cpu_real_public_environment_proxy",
        "duration_s": float(duration_s),
        "duration_breakdown_s": {"current_work": float(duration_s), "historical_capture": 0.0},
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "selection": SELECTION_SEED,
            "episode": list(EPISODE_SEEDS),
            "fitting": 658_029,
            "arrival": 658_030,
            "bootstrap": 658_031,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(sources)),
        "rows": [deepcopy(dict(row)) for row in rows],
        "episode_rows": [deepcopy(dict(row)) for row in episode_rows],
        "raw_reduction": reduced,
        "raw_reduction_checksum": canonical_hash(reduced),
        "sample_size_budget": {
            "planned": len(manifest["schedule"]),
            "attempted": attempted,
            "completed": completed,
            "excluded": 0,
            "failed": failed,
            "censored": censored,
            "unstarted": len(manifest["schedule"]) - attempted,
            "independent_game_clusters": len(manifest["games"]),
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "validation_manifest": {
            "test_paths": [TEST_PATH.as_posix()],
            "changed_modules": [
                MODULE_PATH.as_posix(),
                "python/carnot/agentic/arc_trajectory_supervisor.py",
                "python/carnot/agentic/arc_competition_agent.py",
            ],
            "static_paths": [WRAPPER_PATH.as_posix()],
        },
        "eligibility_receipt_ready_score": readiness,
        "parity_rows": [deepcopy(dict(row)) for row in parity_rows],
        "panel_manifest": manifest,
        "solve_provenance": "live_agent_self_discovery",
        "solve_claimed": False,
        "registry_precheck": deepcopy(
            dict(registry_receipt)
            if registry_receipt is not None
            else registry_precheck(root, manifest["games"])
        ),
        "efficacy_estimate": None if applied_support == 0 else {"status": "not_estimated"},
        "benefit_supported_score": int(
            all(row["passed"] for row in gates if row["category"] in {"support", "benefit"})
        ),
        "exclusive_timing_helper": EXCLUSIVE_TIMING_HELPER,
    }
    artifact["field_principles"] = {
        key: (
            "This field prevents omitted evidence from silently changing the terminal conclusion."
        )
        for key in (*artifact.keys(), "field_principles")
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - host tool failure.
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def build_artifact_for_test(root: Path) -> Json:
    """Build a small complete fixture without touching a game environment."""

    manifest = build_panel_manifest()
    rows = [
        {
            "episode_id": "fixture:seed-658027",
            "game": manifest["games"][0],
            "seed": EPISODE_SEEDS[0],
            "arm": arm,
            "enabled": arm != "tool_loop_reinduction",
            "eligible": arm == "force_exploration_diversity",
            "observation_count": 1,
            "selected_count": int(arm == "force_exploration_diversity"),
            "applied_count": 0,
            "mode": "shadow",
            "disposition": "complete",
        }
        for arm in ARM_SELECTION_ORDER
    ]
    parity = [
        {"surface": surface, "disabled": 0, "enabled": 0, "passed": True}
        for surface in ("actions", "rng", "model_calls", "environment_calls", "state")
    ]
    checks, sources = collect_preconditions(root)
    return build_artifact(
        root,
        rows=rows,
        episode_rows=[
            {
                "episode_id": "fixture:seed-658027",
                "game": manifest["games"][0],
                "seed": EPISODE_SEEDS[0],
                "disposition": "complete",
            }
        ],
        parity_rows=parity,
        preconditions=checks,
        sources=sources,
        validation_receipts=[{"name": "fixture", "required": True, "passed": True}],
        phase_spans=[],
        started_at_utc="2026-09-22T00:00:00Z",
        ended_at_utc="2026-09-22T00:00:01Z",
        duration_s=1.0,
    )


def build_blocked_artifact(
    root: Path,
    *,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
) -> Json:
    """Close an unchanged external absence with its exact failed observation."""

    artifact = (
        build_artifact_for_test(root)
        if (root / REGISTRY_PATH).is_file()
        else {
            "schema": SCHEMA,
            "version": 1,
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
            "run_date": RUN_DATE,
        }
    )
    artifact.update(
        {
            "status": "complete_blocked_external_prerequisite_absent",
            "honest_verdict": "complete_blocked_external_prerequisite_absent",
            "verdict_class": "blocked",
            "eligibility_receipt_ready_score": 0,
            "gate_check_summary": {
                "all_passed": False,
                "failed_count": 1,
                "first_failure": {
                    "path": path,
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                },
            },
        }
    )
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any], root: Path, *, require_terminal: bool
) -> list[str]:
    """Cold-check identity, independent reduction, readiness, and final hashes."""

    del root
    errors: list[str] = []
    required = {
        "schema",
        "version",
        "experiment_id",
        "milestone",
        "run_date",
        "preconditions_checked",
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "invocation_counts",
        "inference_substrate_class",
        "inference_substrate",
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
        "eligibility_receipt_ready_score",
        "parity_rows",
        "panel_manifest",
        "solve_provenance",
        "registry_precheck",
    }
    for key in sorted(required - set(artifact)):
        errors.append(f"missing_field:{key}")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_mismatch")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("model_specs_nonempty")
    if artifact.get("model_invoked") is not False:
        errors.append("model_invoked_not_false")
    if artifact.get("inference_substrate_class") != "no_model_load":
        errors.append("substrate_class_mismatch")
    if artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("substrate_mismatch")
    counts = artifact.get("invocation_counts")
    if not isinstance(counts, Mapping) or any(counts.values()):
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("eligibility_receipt_ready_score") not in {0, 1}:
        errors.append("readiness_not_bare_numeric")
    rows = artifact.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        errors.append("rows_not_sequence")
    else:
        reduced = independently_reduce_rows([row for row in rows if isinstance(row, Mapping)])
        if artifact.get("raw_reduction") != reduced:
            errors.append("independent_reduction_mismatch")
        if artifact.get("raw_reduction_checksum") != canonical_hash(reduced):
            errors.append("raw_reduction_checksum_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(key not in principles for key in artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if require_terminal:
        if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
            errors.append("terminal_prefix_missing")
        if artifact.get("verdict_class") not in {
            "positive",
            "circular_positive",
            "null",
            "blocked",
            "disqualified",
            "partial",
        }:
            errors.append("verdict_class_invalid")
    return list(dict.fromkeys(errors))


def independent_reduce(artifact: Mapping[str, Any]) -> Json:
    """Recompute the comparative rows without trusting declared summary fields."""

    rows = artifact.get("rows")
    typed = (
        [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    )
    reduced = independently_reduce_rows(typed)
    errors = []
    if artifact.get("raw_reduction") != reduced:
        errors.append("independent_reduction_mismatch")
    if artifact.get("raw_reduction_checksum") != canonical_hash(reduced):
        errors.append("raw_reduction_checksum_mismatch")
    return {"rows": typed, "reduced": reduced, "errors": errors, "matches_declared": not errors}


def _affected_manifest() -> Any:
    from carnot import experiment_7358_v646_validation_contract as contract

    return contract.AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=(TEST_PATH.as_posix(),),
        changed_modules=(MODULE_PATH.as_posix(),),
        static_paths=(
            WRAPPER_PATH.as_posix(),
            "python/carnot/agentic/arc_trajectory_supervisor.py",
            "python/carnot/agentic/arc_competition_agent.py",
        ),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[Any]:
    """Freeze scoped checks, capability E2E, and the private smoke."""

    from carnot import experiment_7358_v646_validation_contract as contract
    from carnot.reporting import experiment_7303_validation_scope as scope

    private_root.mkdir(parents=True, exist_ok=True)
    commands: list[Any] = list(
        contract.build_command_plan(root, _affected_manifest(), private_root)
    )
    pytest = ".venv/bin/pytest"
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    for name, test in (
        ("e2e_009", "tests/python/test_arc_induction_state_persistence.py"),
        ("e2e_010", "tests/python/test_arc_tool_grammar_transport.py"),
        ("e2e_011", "tests/python/test_arc_decision_telemetry.py"),
    ):
        parent = private_root / name
        parent.mkdir(parents=True, exist_ok=True)
        commands.append(
            scope.CommandSpec(
                name,
                (pytest, *common, f"--basetemp={parent / 'basetemp'}", test, "-q"),
                "capability_e2e",
                900.0,
            )
        )
    smoke_parent = private_root / "private_arc_smoke"
    smoke_parent.mkdir(parents=True, exist_ok=True)
    commands.append(
        contract.EnvironmentCommandSpec(
            "private_arc_smoke",
            (
                ".venv/bin/python",
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(smoke_parent / "receipt.json"),
            ),
            "private_real_environment_smoke",
            900.0,
            (("CARNOT_ARC_DISABLE_INDUCTION", "1"),),
        )
    )
    return commands


def validate_validation_plan(root: Path, commands: Sequence[Any]) -> list[str]:
    """Reject missing checks, command drift, or broad scoped substitutions."""

    from carnot import experiment_7358_v646_validation_contract as contract
    from carnot.reporting import experiment_7303_validation_scope as scope

    affected = [row for row in commands if row.name in scope.REQUIRED_CHECK_NAMES]
    errors = contract.validate_command_plan(root, _affected_manifest(), affected)
    expected = {
        *scope.REQUIRED_CHECK_NAMES,
        "e2e_009",
        "e2e_010",
        "e2e_011",
        "private_arc_smoke",
    }
    counts = {name: sum(row.name == name for row in commands) for name in expected}
    errors.extend(f"command_count:{name}:{count}" for name, count in counts.items() if count != 1)
    return list(dict.fromkeys(errors))


def terminal_command_specs(root: Path, candidate: Path) -> list[Any]:
    """Build cold replay and strict independent readers for the exact candidate."""

    from carnot.reporting import experiment_7303_validation_scope as scope

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", wrapper, "--replay", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", wrapper, "--replay", str(candidate), "--reduce-only"),
            "exact_terminal_candidate",
            300.0,
        ),
        scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        scope.CommandSpec(
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


def _read_jsonl(path: Path) -> list[Json]:
    rows: list[Json] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _run_panel_episode(  # pragma: no cover - real public environment subprocess.
    root: Path,
    unit: Mapping[str, Any],
    raw_dir: Path,
    private_dir: Path,
    run_started: float,
) -> Json:  # pragma: no cover
    """Run one owned LLM-off E3 process and retain its passive receipt sidecar."""

    episode_id = str(unit["episode_id"])
    safe_id = episode_id.replace(":", "__")
    eligibility_path = raw_dir / "eligibility" / f"{safe_id}.jsonl"
    log_path = raw_dir / "episodes" / f"{safe_id}.log"
    output_path = private_dir / f"{safe_id}.json"
    eligibility_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        str(root / ".venv/bin/python"),
        "-u",
        "scripts/arc_loop_solve.py",
        "--mechanism",
        "e3",
        "--game",
        str(unit["game"]),
        "--max-actions",
        str(ACTION_CAP),
        "--output",
        str(output_path),
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{root / 'python'}:{root}",
            "CARNOT_FORCE_LIVE": "1",
            "CARNOT_ARC_DISABLE_INDUCTION": "1",
            RECORDER_ENV_FLAG: "1",
            RECORDER_PATH_ENV: str(eligibility_path),
            "CARNOT_ARC_RANDOM_SEED": str(unit["seed"]),
            "CARNOT_ARC_GENERATOR_SEED": str(unit["seed"]),
            "CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW": str(SHIPPED_SUPERVISOR_WINDOW),
        }
    )
    environment.pop("CARNOT_ARC_TRAJECTORY_SUPERVISOR", None)
    environment.pop("CARNOT_ARC_SUPERVISOR_TOOL_ARM", None)
    began = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        timed_out = False
        last_heartbeat = began
        while process.poll() is None:
            now = time.monotonic()
            if now - began >= EPISODE_CAP_S:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                break
            if now - last_heartbeat >= 60:
                progress(
                    run_started,
                    "panel_episode",
                    "pending",
                    episode=episode_id,
                    pid=process.pid,
                )
                last_heartbeat = now
            time.sleep(0.25)
        return_code = process.wait()
    receipt_rows = _read_jsonl(eligibility_path)
    output = _load_object(output_path)
    environment_path = raw_dir / "environment" / f"{safe_id}.json"
    if output_path.is_file():
        environment_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = environment_path.with_suffix(".tmp")
        temporary.write_bytes(output_path.read_bytes())
        os.replace(temporary, environment_path)
    disposition = "censored_timeout" if timed_out else "complete" if return_code == 0 else "failed"
    return {
        **deepcopy(dict(unit)),
        "disposition": disposition,
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_s": time.monotonic() - began,
        "eligibility_observation_count": len(receipt_rows),
        "eligibility_path": eligibility_path.relative_to(root).as_posix(),
        "eligibility_sha256": _sha256_file(eligibility_path)
        if eligibility_path.is_file()
        else None,
        "log_path": log_path.relative_to(root).as_posix(),
        "log_sha256": _sha256_file(log_path),
        "environment_result_path": environment_path.relative_to(root).as_posix()
        if environment_path.is_file()
        else None,
        "environment_result_sha256": _sha256_file(environment_path)
        if environment_path.is_file()
        else None,
        "reached_level": output.get("reached_level"),
        "actions_used": output.get("actions", output.get("action_count")),
        "official_hidden_score": False,
        "solve_claimed": False,
    }


def collect_panel(  # pragma: no cover - real public environment subprocesses.
    root: Path, manifest: Mapping[str, Any], run_started: float
) -> list[Json]:  # pragma: no cover
    """Collect at most twelve bounded units under one aggregate deadline."""

    from carnot.reporting.current_work_receipt import atomic_json

    raw_dir = root / RAW_DIR
    checkpoint = _load_object(raw_dir / "panel_checkpoint.json")
    prior_rows = checkpoint.get("rows")
    if isinstance(prior_rows, list) and len(prior_rows) == len(manifest["schedule"]):
        complete = all(
            isinstance(row, Mapping)
            and row.get("disposition") == "complete"
            and isinstance(row.get("eligibility_path"), str)
            and (root / str(row["eligibility_path"])).is_file()
            for row in prior_rows
        )
        if complete:
            progress(run_started, "panel_collection", "checkpoint_resumed", units=len(prior_rows))
            return [dict(row) for row in prior_rows if isinstance(row, Mapping)]
    private_dir = Path(tempfile.mkdtemp(prefix="exp7526-panel-", dir="/tmp"))
    rows: list[Json] = []
    collection_started = time.monotonic()
    schedule = list(manifest["schedule"])
    for index, unit in enumerate(schedule):
        if time.monotonic() - collection_started >= COLLECTION_CAP_S:
            rows.extend(
                {
                    **deepcopy(dict(pending)),
                    "disposition": "unstarted_collection_cap",
                    "eligibility_observation_count": 0,
                    "official_hidden_score": False,
                    "solve_claimed": False,
                }
                for pending in schedule[index:]
            )
            break
        progress(
            run_started,
            "panel_episode",
            "before_benchmark",
            episode=unit["episode_id"],
            completed=index,
            total=len(schedule),
        )
        row = _run_panel_episode(root, unit, raw_dir, private_dir, run_started)
        rows.append(row)
        progress(
            run_started,
            "panel_episode",
            "after_benchmark",
            episode=unit["episode_id"],
            disposition=row["disposition"],
            observations=row["eligibility_observation_count"],
        )
        atomic_json(raw_dir / "panel_checkpoint.json", {"rows": rows})
    return rows


def reduce_panel_receipts(root: Path, episode_rows: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Aggregate each raw episode-arm cell while preserving unknown observations."""

    reduced: list[Json] = []
    for episode in episode_rows:
        path_value = episode.get("eligibility_path")
        observations = _read_jsonl(root / str(path_value)) if isinstance(path_value, str) else []
        by_arm: dict[str, Json] = {
            arm: {
                "episode_id": episode.get("episode_id"),
                "game": episode.get("game"),
                "seed": episode.get("seed"),
                "arm": arm,
                "enabled": False,
                "eligible": False,
                "eligible_count": 0,
                "ineligible_count": 0,
                "unknown_eligibility_count": 0,
                "selected_count": 0,
                "applied_count": 0,
                "observation_count": 0,
                "mode": "shadow",
                "disposition": episode.get("disposition"),
                "source_sidecar": path_value,
                "source_sha256": episode.get("eligibility_sha256"),
            }
            for arm in ARM_SELECTION_ORDER
        }
        for observation in observations:
            for arm_row in observation.get("arm_rows", []):
                if not isinstance(arm_row, Mapping) or arm_row.get("arm") not in by_arm:
                    continue
                cell = by_arm[str(arm_row["arm"])]
                cell["observation_count"] += 1
                cell["enabled"] = bool(cell["enabled"] or arm_row.get("enabled") is True)
                cell["mode"] = str(arm_row.get("mode", observation.get("mode", "unknown")))
                eligibility = arm_row.get("eligible")
                if eligibility is True:
                    cell["eligible_count"] += 1
                    cell["eligible"] = True
                elif eligibility is False:
                    cell["ineligible_count"] += 1
                else:
                    cell["unknown_eligibility_count"] += 1
                    cell["eligible"] = None
                cell["selected_count"] += int(arm_row.get("selected") is True)
                cell["applied_count"] += int(arm_row.get("applied") is True)
        reduced.extend(by_arm.values())
    return reduced


def parity_rows_from_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Bind parity surfaces to the focused fixture and shipped telemetry E2E."""

    passed = {
        str(row.get("name")): row.get("passed") is True
        for row in receipts
        if isinstance(row, Mapping)
    }
    fixture_passed = passed.get("focused_pytest", False)
    telemetry_passed = passed.get("e2e_011", False)
    return [
        {
            "surface": surface,
            "recorder_off": "fixture_control",
            "recorder_on": "fixture_treatment",
            "passed": bool(fixture_passed and telemetry_passed),
            "evidence": ["focused_pytest", "e2e_011"],
        }
        for surface in ("actions", "rng", "model_calls", "environment_calls", "state_transitions")
    ]


def _phase(name: str, began: float, started: float, units: int) -> Json:
    ended = time.monotonic()
    return {
        "phase": name,
        "start_s": began - started,
        "end_s": ended - started,
        "duration_s": ended - began,
        "completed_units": units,
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return all(by_name.get(name, {}).get("passed") is True for name in names)


def run_experiment(  # pragma: no cover - process and real-environment orchestration.
    root: Path, run_date: str
) -> Json:  # pragma: no cover
    """Precheck, validate, measure, cold-replay, and atomically publish."""

    from carnot import experiment_7358_v646_validation_contract as contract
    from carnot.reporting.current_work_receipt import atomic_json
    from carnot.reporting import experiment_7303_validation_scope as scope

    started = time.monotonic()
    started_at = utc_now()
    progress(started, "startup", "flushed_progress")
    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    spans: list[Json] = []

    began = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, sources = collect_preconditions(root)
    spans.append(_phase("preconditions", began, started, len(preconditions)))
    missing = [row for row in preconditions if row.get("passed") is not True]
    progress(started, "preconditions", "end", passed=not missing)
    if missing:
        first = missing[0]
        blocked = build_blocked_artifact(
            root,
            path=str(first["upstream"]),
            field=str(first["field"]),
            expected=first["expected"],
            observed=first["observed"],
        )
        progress(started, "publish", "before_atomic_blocked", path=RESULT_PATH)
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "publish", "after_atomic_blocked", path=RESULT_PATH)
        return blocked

    manifest = build_panel_manifest()
    registry = registry_precheck(root, manifest["games"])
    progress(started, "registry_precheck", "complete", games=len(registry["rows"]))
    (root / RAW_DIR).mkdir(parents=True, exist_ok=True)
    atomic_json(root / RAW_DIR / "panel_manifest.json", manifest)
    progress(started, "panel_manifest", "frozen", units=len(manifest["schedule"]))

    private = Path(tempfile.mkdtemp(prefix="exp7526-validation-", dir="/tmp"))
    plan = build_validation_plan(root, private / "scoped")
    plan_errors = validate_validation_plan(root, plan)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(started, "validation_manifest", "frozen", commands=len(plan))

    began = time.monotonic()
    progress(started, "validation", "before_subprocesses", commands=len(plan))
    receipts = contract.run_categorized_commands(
        root,
        [contract.PlannedCommand(row, "required_validation", True) for row in plan],
        log_dir=root / RAW_DIR / "validation/scoped",
        heartbeat_s=60.0,
    )
    spans.append(_phase("required_validation", began, started, len(receipts)))
    progress(started, "validation", "after_subprocesses", completed=len(receipts))
    required_names = (
        *scope.REQUIRED_CHECK_NAMES,
        "e2e_009",
        "e2e_010",
        "e2e_011",
        "private_arc_smoke",
    )
    if not _receipts_pass(receipts, required_names):
        raise RuntimeError("required_validation_failed")

    began = time.monotonic()
    progress(started, "panel_collection", "before_benchmarks", units=len(manifest["schedule"]))
    episode_rows = collect_panel(root, manifest, started)
    spans.append(_phase("panel_collection", began, started, len(episode_rows)))
    progress(started, "panel_collection", "after_benchmarks", units=len(episode_rows))
    rows = reduce_panel_receipts(root, episode_rows)
    parity_rows = parity_rows_from_receipts(receipts)
    for episode in episode_rows:
        for key in ("eligibility_path", "environment_result_path", "log_path"):
            path_value = episode.get(key)
            if isinstance(path_value, str) and (root / path_value).is_file():
                sources[path_value] = _source_row(root, Path(path_value), "raw_measurement_sidecar")

    candidate = build_artifact(
        root,
        rows=rows,
        episode_rows=episode_rows,
        parity_rows=parity_rows,
        preconditions=preconditions,
        sources=sources,
        validation_receipts=receipts,
        phase_spans=spans,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        registry_receipt=registry,
    )
    errors = validate_artifact(candidate, root, require_terminal=False)
    if errors:
        raise RuntimeError(f"candidate_invalid:{errors}")
    progress(started, "candidate", "before_atomic_write", path=CANDIDATE_PATH)
    atomic_json(root / CANDIDATE_PATH, candidate)
    progress(started, "candidate", "after_atomic_write", path=CANDIDATE_PATH)

    began = time.monotonic()
    terminal_specs = terminal_command_specs(root, root / CANDIDATE_PATH)
    progress(started, "terminal_validation", "before_subprocesses", commands=len(terminal_specs))
    terminal_receipts = contract.run_categorized_commands(
        root,
        [contract.PlannedCommand(row, "terminal_reader", True) for row in terminal_specs],
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    spans.append(_phase("terminal_validation", began, started, len(terminal_receipts)))
    receipts.extend(terminal_receipts)
    progress(started, "terminal_validation", "after_subprocesses", completed=len(terminal_receipts))
    terminal_names = (
        "declared_entrypoint_cold_replay",
        "independent_cold_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    )
    if not _receipts_pass(receipts, terminal_names):
        raise RuntimeError("required_terminal_validation_failed")

    final = build_artifact(
        root,
        rows=rows,
        episode_rows=episode_rows,
        parity_rows=parity_rows,
        preconditions=preconditions,
        sources=sources,
        validation_receipts=receipts,
        phase_spans=spans,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        registry_receipt=registry,
    )
    errors = validate_artifact(final, root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic_terminal", path=RESULT_PATH)
    atomic_json(root / RESULT_PATH, final)
    progress(
        started,
        "publish",
        "after_atomic_terminal",
        verdict=final["honest_verdict"],
        ready=final["eligibility_receipt_ready_score"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the producer role or one read-only cold replay role."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--reduce-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or independently check one exact candidate."""

    args = parse_args(argv)
    if args.replay is not None:
        artifact = _load_object(args.replay)
        replay = independent_reduce(artifact)
        errors = (
            replay["errors"]
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
        return int(bool(errors))
    if args.date is None:
        raise SystemExit("--date is required unless --replay is used")
    run_experiment(REPO_ROOT, str(args.date))
    return 0
