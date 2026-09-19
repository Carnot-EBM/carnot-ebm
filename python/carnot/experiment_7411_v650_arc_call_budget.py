"""Qualify a shared ARC request budget without current model work.

The experiment uses scripted HTTP bytes through the scored policy. It measures
dispatch accounting and cancellation. It does not measure generator efficacy.

Spec refs: REQ-ARC-WMTE-7411 and SCENARIO-ARC-WMTE-7411-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import threading
import time
from typing import Any
import urllib.request

import yaml

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7406_v649_arc_generalization as exp7406
from carnot.agentic import arc_executable_world_model as world_model
from carnot.agentic import arc_induction_tool_loop as tool_loop
from carnot.agentic.arc_inference_boundary import boundary_call_for_proposer
from carnot.agentic.arc_request_budget import (
    EpisodeRequestBudget,
    LateRequestCompletion,
    RequestBudgetError,
    attach_request_budget,
    request_budget_scope,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7411-arc-call-budget"
MILESTONE = "2026.09.650"
RUN_DATE = "20260919"
SCHEMA = "carnot.exp7411.v650_arc_call_budget.v1"
RESULT_PATH = Path("results/experiment_7411_v650_arc_call_budget.json")
RAW_DIR = Path("results/raw/experiment_7411_v650_arc_call_budget")
MODULE_PATH = Path("python/carnot/experiment_7411_v650_arc_call_budget.py")
BUDGET_PATH = Path("python/carnot/agentic/arc_request_budget.py")
BOUNDARY_PATH = Path("python/carnot/agentic/arc_inference_boundary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7411_v650_arc_call_budget.py")
TEST_PATH = Path("tests/python/test_experiment_7411_v650_arc_call_budget.py")
BUDGET_TEST_PATH = Path("tests/python/test_arc_request_budget.py")
BOUNDARY_TEST_PATH = Path("tests/python/test_experiment_7289_v641_arc_boundary.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
HISTORICAL_PATH = Path("results/experiment_7406_v649_arc_generalization.json")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
REDIRECT_LEDGER_PATH = Path("ops/arc_supervisor_refinement_ledger.json")
ACTION_LIMIT = 12
REQUEST_LIMIT = 2
RANDOM_SEED = 7_411_650
REQUIRED_E2E = ("e2e_009", "e2e_010", "e2e_offline_smoke")
REQUIRED_TERMINAL = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
    "usable_answers": 0,
}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/experiment_7398_v649_arc_checkpoint.py"),
    Path("python/carnot/experiment_7406_v649_arc_generalization.py"),
    HISTORICAL_PATH,
    REGISTRY_PATH,
    REDIRECT_LEDGER_PATH,
    SPEC_PATH,
    MODULE_PATH,
    BUDGET_PATH,
    BOUNDARY_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    BUDGET_TEST_PATH,
)


def utc_now() -> str:
    """Return an aware UTC timestamp for a measured boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush one phase or long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7411] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for rows and terminal identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact source or evidence bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Flush a complete JSON value before one same-directory replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_object(path: Path) -> JsonDict:
    """Read one JSON object or return an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum value itself."""

    copied = deepcopy(dict(value))
    copied["reproducibility_checksum"] = ""
    return canonical_hash(copied)


def gate_row(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    artifact_field: str,
    operator: str = "==",
) -> JsonDict:
    """Retain one exact gate comparison."""

    passed = observed == expected if operator == "==" else observed in expected
    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "upstream": upstream,
        "artifact_field": artifact_field,
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose every failed gate and its first exact source."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


def diagnose_exp7406_overflow(root: Path) -> JsonDict:
    """Authenticate the held-out overflow without promoting its science."""

    artifact = load_object(root / HISTORICAL_PATH)
    counts = artifact.get("invocation_counts")
    counts = dict(counts) if isinstance(counts, Mapping) else {}
    budget = artifact.get("sample_size_budget")
    budget = dict(budget) if isinstance(budget, Mapping) else {}
    attempted = int(counts.get("generation_calls_attempted") or 0)
    planned = int(budget.get("planned_units") or 0) * int(
        budget.get("model_call_limit_per_episode") or 0
    )
    identity_ok = (
        artifact.get("experiment_id") == "exp7406-arc-generalization"
        and artifact.get("verdict_class") == "disqualified"
        and artifact.get("flagged_adversarial") is True
        and attempted == 35
        and planned == 12
    )
    return {
        "scope": "historical",
        "experiment_id": artifact.get("experiment_id"),
        "artifact_path": HISTORICAL_PATH.as_posix(),
        "artifact_sha256": sha256_file(root / HISTORICAL_PATH),
        "generation_calls_attempted": attempted,
        "planned_call_limit": planned,
        "overflow_calls": max(0, attempted - planned),
        "cause_reproduced": identity_ok and attempted > planned,
        "old_enforcement_boundary": "terminal_reducer_only",
        "boundary_that_bypassed_budget": "generation_dispatch",
        "explanation": (
            "The launcher recorded a two-call limit in schedule metadata. Each live callback "
            "still entered generation because no shared slot was reserved before HTTP dispatch."
        ),
        "eligible_science": False,
        "original_status": artifact.get("status"),
        "original_verdict_class": artifact.get("verdict_class"),
        "original_flagged_adversarial": artifact.get("flagged_adversarial"),
    }


def freeze_game_rotation(root: Path) -> JsonDict:
    """Select three label-blind registry games without repeating six episodes."""

    registry_value = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8"))
    registry = registry_value if isinstance(registry_value, Mapping) else {}
    from carnot.agentic.arc_game_adapters import adaptered_games

    selected = exp7406.freeze_panel(registry, adaptered_games=set(adaptered_games()))
    return {
        **deepcopy(selected),
        "selected_games": list(selected.get("games") or []),
        "episode_count": len(selected.get("games") or []),
        "six_episode_panel_repeated": False,
    }


def _fixture_proposer() -> world_model.LocalGGUFProposer:
    proposer = world_model.LocalGGUFProposer(
        model_path="/fixtures/scripted.gguf",
        model_repository="fixture/scripted",
        model_filename="scripted.gguf",
        model_revision="7" * 40,
        ffn_cpu_layers=0,
        mtp=False,
        max_tokens=256,
        tries=1,
    )
    proposer._ensure_server = lambda: True
    proposer._proc = type("Proc", (), {"pid": os.getpid()})()
    return proposer


def _scripted_post(proposer: Any, branch: str, request_id: str | None = None) -> None:
    with request_budget_scope(branch, request_id=request_id):
        tool_loop._post_chat(proposer, [], turn=0, timeout_s=1, selfparse=True)


def run_callback_matrix(work_dir: Path) -> JsonDict:
    """Exercise each callback class at the actual HTTP inference boundary."""

    from carnot import experiment_7234_v637_arc_scored_dryrun as scored

    work_dir.mkdir(parents=True, exist_ok=True)
    responses = 0

    def scripted_open(*_args: Any, **_kwargs: Any) -> io.BytesIO:
        nonlocal responses
        responses += 1
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "scripted"}}]}).encode())

    old_open = urllib.request.urlopen
    urllib.request.urlopen = scripted_open
    try:
        unguarded = _fixture_proposer()
        policy, factory = scored.build_disposable_submitted_policy("r11l", unguarded)
        for branch in ("primary", "repair", "refinement", "supervisor"):
            _scripted_post(policy._proposer(), branch)
        overflow_dispatches = responses

        controls: list[JsonDict] = []
        guarded = EpisodeRequestBudget("guarded-sequential", limit=2, deadline_s=60)
        attach_request_budget(policy._proposer(), guarded)
        for index, branch in enumerate(("primary", "parser_retry", "repair", "refinement")):
            try:
                _scripted_post(policy._proposer(), branch, f"sequential-{index}")
            except RequestBudgetError:
                pass
        controls.append({"case": "success_parser_retry", **guarded.receipt()})

        failure = EpisodeRequestBudget("failure", limit=2, deadline_s=60)
        failure.reserve(branch="primary", request_id="failure-primary").fail(
            ValueError("parser failure")
        )
        failure.reserve(branch="parser_retry", request_id="failure-retry").complete()
        controls.append({"case": "parser_failure_retry", **failure.receipt()})

        late = EpisodeRequestBudget("late", limit=2, deadline_s=60)
        late_call = late.reserve(branch="refinement", request_id="late-response")
        late.cancel("episode_closed")
        try:
            late_call.complete()
        except LateRequestCompletion:
            pass
        controls.append({"case": "cancellation_late_completion", **late.receipt()})

        nested = EpisodeRequestBudget("nested", limit=2, deadline_s=60)
        nested.reserve(branch="primary", request_id="nested-outer").complete()
        nested.reserve(branch="nested", request_id="nested-inner").complete()
        try:
            nested.reserve(branch="supervisor", request_id="nested-refused")
        except RequestBudgetError:
            pass
        controls.append({"case": "nested_supervisor", **nested.receipt()})

        concurrent = EpisodeRequestBudget("concurrent", limit=2, deadline_s=60)
        barrier = threading.Barrier(6)

        def race(index: int) -> str:
            barrier.wait()
            try:
                reservation = concurrent.reserve(
                    branch="concurrent", request_id=f"concurrent-{index}"
                )
            except RequestBudgetError:
                return "refused"
            reservation.complete()
            return "completed"

        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(race, range(6)))
        controls.append({"case": "concurrent", **concurrent.receipt()})

        restart_first = EpisodeRequestBudget("restart", limit=2, deadline_s=60)
        restart_first.reserve(branch="repair", request_id="durable-complete").complete()
        restarted = EpisodeRequestBudget.from_receipt(restart_first.receipt(), deadline_s=60)
        duplicate_refused = False
        try:
            restarted.reserve(branch="repair", request_id="durable-complete")
        except RequestBudgetError:
            duplicate_refused = True
        restarted.reserve(branch="supervisor", request_id="after-restart").complete()
        restart_receipt = {"case": "cold_restart", **restarted.receipt()}
        controls.append(restart_receipt)
    finally:
        urllib.request.urlopen = old_open

    callback_rows = [
        {"case": control["case"], **deepcopy(row)}
        for control in controls
        for row in control["callback_rows"]
    ]
    branches = sorted({str(row["branch"]) for row in callback_rows})
    all_passed = all(
        int(row["attempted"]) <= REQUEST_LIMIT
        and row["accounting_valid"] is True
        and int(row["in_flight"]) == 0
        for row in controls
    )
    return {
        "factory": factory,
        "overflow_diagnostic": {
            "dispatches": overflow_dispatches,
            "budget_declared": REQUEST_LIMIT,
            "boundary": "generation_dispatch",
            "eligible_science": False,
        },
        "control_rows": controls,
        "callback_rows": callback_rows,
        "branches_exercised": branches,
        "late_write_violations": sum(int(row["late_write_violations"]) for row in controls),
        "deadline_violations": sum(int(row["deadline_violations"]) for row in controls),
        "budget_violations": sum(int(row["attempted"]) > REQUEST_LIMIT for row in controls),
        "cold_restart_parity": duplicate_refused
        and restart_receipt["replayed_completed_refusals"] == 1
        and restart_receipt["attempted"] == 2,
        "all_controls_passed": all_passed,
    }


def _temporary_environment(values: Mapping[str, str]) -> tuple[dict[str, str | None], Any]:
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)

    def restore() -> None:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return old, restore


def run_scored_plumbing(games: Sequence[str], work_dir: Path) -> JsonDict:
    """Drive three public environments through the actual scored policy factory."""

    from carnot import experiment_7234_v637_arc_scored_dryrun as scored
    from scripts.arc_leaderboard_eval import run_game

    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    environment = {
        "CARNOT_ARC_INDUCE_THINK": "0",
        "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
        "CARNOT_ARC_INDUCE_TOOL_GRAMMAR": "1",
        "CARNOT_ARC_INDUCE_TOOL_TURNS": "4",
        "CARNOT_ARC_STALL_REFACTOR_LOOP": "0",
        "CARNOT_ARC_CEGIS_ACCEPT_SPLIT": "1",
        "CARNOT_ARC_STRUCTURED_NAV": "0",
        "CARNOT_ARC_LIVE_TTT": "0",
        "CARNOT_ARC_DISABLE_INDUCTION": "1",
    }
    _old, restore_environment = _temporary_environment(environment)
    old_open = urllib.request.urlopen
    old_e3_dir = world_model.E3_DIR
    try:
        for index, game in enumerate(games):
            game_dir = work_dir / f"{index:02d}_{game}"
            game_dir.mkdir(parents=True, exist_ok=True)
            world_model.E3_DIR = game_dir / "engines"
            proposer = _fixture_proposer()
            budget = EpisodeRequestBudget(str(game), limit=REQUEST_LIMIT, deadline_s=60)
            attach_request_budget(proposer, budget)
            payloads: list[JsonDict] = []

            def scripted_open(request: Any, timeout: float | None = None) -> io.BytesIO:
                del timeout
                payloads.append(json.loads(request.data))
                answer = {"choices": [{"message": {"content": "scripted-policy-probe"}}]}
                return io.BytesIO(json.dumps(answer).encode())

            urllib.request.urlopen = scripted_open
            policy, factory = scored.build_disposable_submitted_policy(str(game), proposer)
            policy.explore_budget = 1
            policy.program_synthesis_filter_enabled = False
            policy.active_probe_controller_enabled = False
            policy.think_arm_fallback_enabled = False
            started = time.monotonic()
            error: str | None = None
            run_row: JsonDict = {}
            try:
                # The policy-owned proposer crosses the same live boundary first. The
                # following environment loop is LLM-off so this CPU plumbing check cannot
                # turn scripted bytes into a model-efficacy claim.
                _scripted_post(policy._proposer(), "primary", f"{game}-scripted-probe")
                run_row = run_game(str(game), policy, budget=ACTION_LIMIT)
            except Exception as exc:  # A plumbing row retains policy failures without fabrication.
                error = f"{type(exc).__name__}: {exc}"[:500]
            finally:
                budget.cancel("episode_closed")
            receipt = budget.receipt()
            supervisor = policy.trajectory_supervisor_diagnostics()
            rows.append(
                {
                    "row_type": "scored_policy_plumbing",
                    "episode_id": f"{game}:seed-{RANDOM_SEED + index}",
                    "game": str(game),
                    "seed": RANDOM_SEED + index,
                    "factory": factory.get("factory"),
                    "policy_class": type(policy).__name__,
                    "adapter_disabled": factory.get("adapter_disabled") is True,
                    "withheld_inputs": list(factory.get("denied_inputs") or []),
                    "saved_engines_withheld": True,
                    "solve_lookup_withheld": True,
                    "game_source_withheld": True,
                    "ground_truth_search_withheld": True,
                    "environment_actions": int(run_row.get("actions") or 0),
                    "action_limit": ACTION_LIMIT,
                    "observed_dispatches": len(payloads),
                    "request_limit": REQUEST_LIMIT,
                    "callback_receipt": receipt,
                    "elapsed_s": time.monotonic() - started,
                    "levels": int(run_row.get("levels") or 0),
                    "error": error,
                    "trajectory_supervisor": supervisor,
                    "redirect_firings": len(supervisor.get("redirects") or []),
                    "solve_provenance": "development_proxy",
                    "solve_credit": 0,
                    "model_invoked": False,
                    "scripted_transport": True,
                    "disposition": "complete_error" if error else "complete",
                }
            )
    finally:
        urllib.request.urlopen = old_open
        world_model.E3_DIR = old_e3_dir
        restore_environment()
    return {
        "rows": rows,
        "budget_violations": sum(row["observed_dispatches"] > REQUEST_LIMIT for row in rows),
        "action_violations": sum(row["environment_actions"] > ACTION_LIMIT for row in rows),
        "deadline_violations": sum(
            int(row["callback_receipt"]["deadline_violations"]) for row in rows
        ),
        "late_write_violations": sum(
            int(row["callback_receipt"]["late_write_violations"]) for row in rows
        ),
    }


def inspect_redirect_ledger(root: Path) -> JsonDict:
    """Read banked supervisor outcomes without adding an arm or recommendation."""

    ledger = load_object(root / REDIRECT_LEDGER_PATH)
    recommendation = ledger.get("recommendation")
    recommendation = dict(recommendation) if isinstance(recommendation, Mapping) else {}
    recommendations = recommendation.get("recommendations")
    supported = (
        [dict(row) for row in recommendations if isinstance(row, Mapping)]
        if isinstance(recommendations, list)
        else []
    )
    return {
        "ledger_path": REDIRECT_LEDGER_PATH.as_posix(),
        "ledger_sha256": sha256_file(root / REDIRECT_LEDGER_PATH),
        "banked_only": True,
        "supported_firings": supported,
        "recommendation": "bank_existing_recommendations" if supported else "nothing_to_refine",
        "reason": None if supported else "no_supported_refinement_firing_in_banked_ledger",
        "new_arm_created": False,
        "arm_order_changed": False,
    }


def affected_manifest() -> validation_contract.AffectedManifest:
    """Freeze the exact affected tests and changed modules."""

    return validation_contract.AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=(
            TEST_PATH.as_posix(),
            BUDGET_TEST_PATH.as_posix(),
            BOUNDARY_TEST_PATH.as_posix(),
        ),
        changed_modules=(MODULE_PATH.as_posix(), BUDGET_PATH.as_posix(), BOUNDARY_PATH.as_posix()),
        static_paths=(WRAPPER_PATH.as_posix(),),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed Exp7303 commands through the Exp7358 planner."""

    return validation_contract.build_command_plan(root, affected_manifest(), private_root)


def e2e_command_specs(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    """Build E2E-009, E2E-010, and the required LLM-off smoke."""

    private.mkdir(parents=True, exist_ok=True)
    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "e2e_009",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e009'}",
                "tests/python/test_arc_induction_state_persistence.py",
                "-q",
            ),
            "E2E-009 scripted scored-policy persistence",
        ),
        validation_scope.CommandSpec(
            "e2e_010",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e010'}",
                "tests/python/test_arc_tool_grammar_transport.py",
                "-q",
            ),
            "E2E-010 scripted grammar transport",
        ),
        validation_scope.CommandSpec(
            "e2e_offline_smoke",
            (
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private / "r11l-twelve-action-smoke.json"),
            ),
            "E2E-009 LLM-off twelve-action environment smoke",
        ),
    ]


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], required: Sequence[str]) -> bool:
    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in required
    )


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain ordinary top-level values without wrapping gate scalars."""

    specific = {
        "schema": "Versioned ordinary fields bind experiment identity, milestone, and status.",
        "run_date": "The requested date stays separate from actual UTC boundaries.",
        "preconditions_checked": "Exact input paths, hashes, and eligibility precede dependent work.",
        "MODEL_SPECS": "An empty list states that no current LLM was loaded.",
        "model_invoked": "False excludes scripted and archived model-shaped bytes.",
        "invocation_counts": "Current owned LLM counters remain exact zeros.",
        "inference_substrate": "A string describes current host CPU plumbing work.",
        "inference_substrate_class": "The no-model class prevents duration-floor padding.",
        "execution_venue": "The closed host value stays separate from device details.",
        "duration_s": "Measured monotonic work time is not zero-filled or padded.",
        "phase_spans": "Real boundaries expose completed units and elapsed time.",
        "random_seed": "Frozen selection and fixture seeds make the run repeatable.",
        "reproducibility_checksum": "The checksum binds code, protocol, inputs, and raw rows.",
        "source_artifact_hashes": "Exact path and byte hashes authenticate every dependent input.",
        "rows": "Every callback control and scored game remains visible.",
        "sample_size_budget": "Planned, attempted, completed, failed, censored, and unstarted counts remain separate.",
        "acceptance_gate_results": "Each check names category, operator, expected, observed, and pass state.",
        "gate_check_summary": "Each blocked check retains its exact upstream path and field.",
        "verifier_is_oracle": "False states that plumbing does not define ARC correctness.",
        "honest_verdict": "The complete verdict limits the finding to budget readiness.",
        "verdict_class": "Null states that no live efficacy finding was measured.",
        "flagged_adversarial": "Critical terminal findings prevent readiness.",
        "validation_receipts": "Exact commands, environments, exits, durations, and log hashes support audit.",
        "promotion_score": "Zero forbids rollout, publication, or weight changes.",
        "arc_budget_ready_score": "One requires the reachable callback invariant and all checks.",
        "live_efficacy_score": "Zero states that scripted transport is not Qwen evidence.",
        "callback_rows": "Each reservation records its branch, timing, cancellation, and terminal state.",
        "solve_provenance": "Development proxy forbids game-level solve credit.",
        "redirect_ledger_disposition": "Only banked outcomes can support supervisor refinement.",
    }
    return {
        key: specific.get(key, f"The {key} field retains directly auditable experiment evidence.")
        for key in keys
    }


def _reduce_raw(
    callback_rows: Sequence[Mapping[str, Any]], scored_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    dispositions = Counter(str(row.get("disposition")) for row in callback_rows)
    attempted = len(callback_rows)
    terminal = sum(dispositions[name] for name in ("completed", "failed", "cancelled"))
    per_episode = Counter(str(row.get("episode_id")) for row in callback_rows)
    return {
        "attempted": attempted,
        "completed": dispositions["completed"],
        "failed": dispositions["failed"],
        "cancelled": dispositions["cancelled"],
        "in_flight": dispositions["in_flight"],
        "accounting_valid": attempted == terminal + dispositions["in_flight"],
        "all_terminal": attempted == terminal,
        "budget_violations": sum(count > REQUEST_LIMIT for count in per_episode.values()),
        "scored_budget_violations": sum(
            int(row.get("observed_dispatches") or 0) > REQUEST_LIMIT for row in scored_rows
        ),
        "action_violations": sum(
            int(row.get("environment_actions") or 0) > ACTION_LIMIT for row in scored_rows
        ),
    }


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    selection: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    callback_panel: Mapping[str, Any],
    scored_panel: Mapping[str, Any],
    redirect_disposition: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    provenance_sidecars: Mapping[str, Any],
) -> JsonDict:
    """Build one independently reducible no-model readiness record."""

    callback_rows = [
        deepcopy(dict(row))
        for row in callback_panel.get("callback_rows", [])
        if isinstance(row, Mapping)
    ]
    scored_rows = [
        deepcopy(dict(row)) for row in scored_panel.get("rows", []) if isinstance(row, Mapping)
    ]
    raw = _reduce_raw(callback_rows, scored_rows)
    preconditions_pass = bool(preconditions_checked) and all(
        row.get("passed") is True for row in preconditions_checked
    )
    validation_names = (*validation_scope.REQUIRED_CHECK_NAMES, *REQUIRED_E2E, *REQUIRED_TERMINAL)
    validations_pass = _receipts_pass(validation_receipts, validation_names)
    callback_pass = bool(callback_panel.get("all_controls_passed")) and all(
        int(callback_panel.get(key) or 0) == 0
        for key in ("budget_violations", "deadline_violations", "late_write_violations")
    )
    scored_pass = (
        len(scored_rows) == 3
        and int(scored_panel.get("budget_violations") or 0) == 0
        and int(scored_panel.get("action_violations") or 0) == 0
        and int(scored_panel.get("deadline_violations") or 0) == 0
        and int(scored_panel.get("late_write_violations") or 0) == 0
    )
    sidecars_pass = set(provenance_sidecars) == {"historical", "scripted"} and all(
        str(row.get("sha256", "")).startswith("sha256:")
        for row in provenance_sidecars.values()
        if isinstance(row, Mapping)
    )
    gates = [
        gate_row(
            "preconditions",
            "precondition",
            True,
            preconditions_pass,
            upstream="preconditions_checked",
            artifact_field="all_inputs_authenticated",
        ),
        gate_row(
            "historical_overflow_reproduced",
            "diagnostic",
            True,
            diagnosis.get("cause_reproduced") is True,
            upstream=HISTORICAL_PATH.as_posix(),
            artifact_field="generation_calls_attempted>planned_call_limit",
        ),
        gate_row(
            "callback_invariants",
            "safety",
            True,
            callback_pass and raw["accounting_valid"] and raw["all_terminal"],
            upstream="callback_rows",
            artifact_field="budget_deadline_terminal_and_late_write_invariants",
        ),
        gate_row(
            "cold_restart_parity",
            "safety",
            True,
            callback_panel.get("cold_restart_parity") is True,
            upstream="callback_control_rows",
            artifact_field="cold_restart_parity",
        ),
        gate_row(
            "three_game_scored_plumbing",
            "completion",
            True,
            scored_pass,
            upstream=REGISTRY_PATH.as_posix(),
            artifact_field="three_games_within_action_and_request_budgets",
        ),
        gate_row(
            "sidecar_separation",
            "provenance",
            True,
            sidecars_pass,
            upstream="provenance_sidecars",
            artifact_field="historical_and_scripted_hashes",
        ),
        gate_row(
            "required_validation",
            "required_validation",
            True,
            validations_pass,
            upstream="validation_receipts",
            artifact_field="affected_e2e_and_terminal_checks",
        ),
        gate_row(
            "automatic_promotion",
            "promotion",
            0,
            0,
            upstream="protocol",
            artifact_field="promotion_score",
        ),
    ]
    ready = int(all(row["passed"] for row in gates))
    counts = {
        "planned_units": len(callback_panel.get("control_rows") or []) + len(scored_rows),
        "attempted_units": len(callback_panel.get("control_rows") or []) + len(scored_rows),
        "completed_units": len(callback_panel.get("control_rows") or []) + len(scored_rows),
        "failed_units": 0,
        "censored_units": 0,
        "unstarted_units": 0,
        "independent_groups": ["callback_controls", "scored_public_game_plumbing"],
        "stopping_rule": "run the frozen callback matrix and one episode on each of three games",
        "request_limit_per_episode": REQUEST_LIMIT,
        "action_limit_per_scored_episode": ACTION_LIMIT,
    }
    rows = [
        {"row_type": "callback_control", **deepcopy(dict(row))}
        for row in callback_panel.get("control_rows", [])
        if isinstance(row, Mapping)
    ] + scored_rows
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_null_arc_budget_ready"
        if ready
        else "complete_disqualified_required_evidence",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": (
            "Host CPU scripted HTTP transport, atomic request accounting, public offline "
            "environment actions, JSON reduction, and small Gibbs receipt construction."
        ),
        "inference_substrate_details": {
            "host": platform.node(),
            "python": platform.python_version(),
            "scripted_transport": True,
            "current_llm_calls": 0,
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "selection_receipt": deepcopy(dict(selection)),
        "overflow_diagnosis": deepcopy(dict(diagnosis)),
        "rows": rows,
        "callback_rows": callback_rows,
        "sample_size_budget": counts,
        "raw_reduction": raw,
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": (
            "complete_null_arc_budget_ready_no_live_efficacy_claim"
            if ready
            else "complete_disqualified_required_evidence"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "repository_health": {
            "status": "not_assessed_by_scoped_run",
            "affects_required_checks": False,
        },
        "field_principles": {},
        "promotion_score": 0,
        "arc_budget_ready_score": ready,
        "live_efficacy_score": 0,
        "solve_provenance": "development_proxy",
        "solve_credit": 0,
        "redirect_ledger_disposition": deepcopy(dict(redirect_disposition)),
        "provenance_sidecars": deepcopy(dict(provenance_sidecars)),
        "small_ebm_training": {
            "performed": False,
            "receipt_class": "small_ebm_training",
            "reason": "request-budget plumbing required no fitted energy model",
            "current_llm_counts_affected": False,
        },
        "future_live_model_prerequisite": {
            "required_model": "unsloth/Qwen3.8-27B-GGUF",
            "required_quantization": "Q4_K_M",
            "requires_owned_gpu_lease": True,
            "requires_current_unflagged_budget_receipt": True,
            "launched_here": False,
        },
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "supervisor_ordering_changed": False,
        "research_conductor_changed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce_file(path: Path) -> JsonDict:
    """Cold-load a candidate and recompute request and action invariants."""

    artifact = load_object(path)
    callback_rows = artifact.get("callback_rows")
    rows = artifact.get("rows")
    callback_rows = callback_rows if isinstance(callback_rows, list) else []
    rows = rows if isinstance(rows, list) else []
    scored = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("row_type") == "scored_policy_plumbing"
    ]
    reduced = _reduce_raw(callback_rows, scored)
    declared = artifact.get("raw_reduction")
    return {
        **reduced,
        "matches_declared": reduced == declared,
        "declared_ready": artifact.get("arc_budget_ready_score"),
    }


def validate_artifact(value: Mapping[str, Any] | Path) -> list[str]:
    """Cold-check identity, raw reduction, scores, principles, and checksum."""

    artifact = load_object(value) if isinstance(value, Path) else deepcopy(dict(value))
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("identity_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_fields_invalid")
    if (
        not isinstance(artifact.get("inference_substrate"), str)
        or artifact.get("inference_substrate_class") != "no_model_load"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        errors.append("duration_invalid")
    callback_rows = artifact.get("callback_rows")
    rows = artifact.get("rows")
    callback_rows = callback_rows if isinstance(callback_rows, list) else []
    rows = rows if isinstance(rows, list) else []
    scored = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("row_type") == "scored_policy_plumbing"
    ]
    reduced = _reduce_raw(callback_rows, scored)
    if reduced != artifact.get("raw_reduction"):
        errors.append("raw_reduction_mismatch")
    gates = artifact.get("acceptance_gate_results")
    gates = gates if isinstance(gates, list) else []
    computed_ready = int(bool(gates) and all(row.get("passed") is True for row in gates))
    if artifact.get("arc_budget_ready_score") != computed_ready:
        errors.append("readiness_mismatch")
    if artifact.get("promotion_score") != 0 or artifact.get("live_efficacy_score") != 0:
        errors.append("score_invalid")
    if artifact.get("verdict_class") not in {"null", "disqualified"}:
        errors.append("verdict_invalid")
    if computed_ready and (
        artifact.get("verdict_class") != "null"
        or not str(artifact.get("honest_verdict", "")).startswith("complete_")
    ):
        errors.append("verdict_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("checksum_mismatch")
    return list(dict.fromkeys(errors))


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Authenticate exact local inputs before measured work."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                "precondition",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                upstream=relative.as_posix(),
                artifact_field="bytes",
            )
        )
        if available:
            hashes[relative.as_posix()] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "role": "historical_diagnostic" if relative == HISTORICAL_PATH else "current_input",
            }
    spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        gate_row(
            "driving_requirement",
            "precondition",
            True,
            "REQ-ARC-WMTE-7411" in spec,
            upstream=SPEC_PATH.as_posix(),
            artifact_field="REQ-ARC-WMTE-7411",
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.append(
        gate_row(
            "new_mechanism_not_retired_rerun",
            "precondition",
            False,
            "exp7411" in exclusion,
            upstream="ops/exclusion_manifest.yaml",
            artifact_field=EXPERIMENT_ID,
        )
    )
    return checks, hashes


def _run_specs(
    root: Path,
    specs: Sequence[validation_scope.CommandSpec],
    log_dir: Path,
) -> list[JsonDict]:
    """Stream each child with its command-local environment."""

    receipts: list[JsonDict] = []
    for index, spec in enumerate(specs):
        environment = dict(getattr(spec, "command_environment", ()))
        if spec.name == "e2e_offline_smoke":
            environment["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
        started_at = utc_now()
        rows = validation_scope.run_commands(
            root,
            [spec],
            log_dir=log_dir / f"{index:02d}_{spec.name}",
            extra_env=environment,
        )
        row = rows[0]
        row["environment"] = environment
        row["started_at_utc"] = started_at
        row["ended_at_utc"] = utc_now()
        receipts.append(row)
    return receipts


def _terminal_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "independent_reducer",
            (
                python,
                "-u",
                "-c",
                (
                    "import json,sys; from pathlib import Path; "
                    "from carnot.experiment_7411_v650_arc_call_budget import independent_reduce_file; "
                    "r=independent_reduce_file(Path(sys.argv[1])); print(json.dumps(r,sort_keys=True)); "
                    "raise SystemExit(0 if r['matches_declared'] else 1)"
                ),
                str(candidate),
            ),
            "independent row reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "unchanged adversarial verifier",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "strict verdict-row consistency",
        ),
    ]


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - CLI orchestration.
    """Run scoped checks, CPU measurements, cold readers, and atomic publication."""

    started = time.monotonic()
    started_at = utc_now()
    phases: list[JsonDict] = []
    progress(started, "startup", "begin", completed_units=0)

    phase_start = time.monotonic()
    progress(started, "preconditions", "before")
    preconditions, source_hashes = collect_preconditions(root)
    phases.append(
        {
            "phase": "preconditions",
            "start_s": phase_start - started,
            "end_s": time.monotonic() - started,
            "completed_units": len(preconditions),
        }
    )
    progress(started, "preconditions", "after", completed_units=len(preconditions))
    if not all(row["passed"] for row in preconditions):
        raise RuntimeError("Exp7411 local preconditions failed")

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7411-", dir="/tmp"))
    commands = build_validation_plan(root, private_root / "affected")
    plan_errors = validation_contract.validate_command_plan(root, affected_manifest(), commands)
    if plan_errors:
        raise RuntimeError(f"affected command plan invalid: {plan_errors}")
    phase_start = time.monotonic()
    progress(started, "affected_validation", "before_subprocesses", total_units=len(commands))
    validation_receipts = _run_specs(root, commands, private_root / "logs" / "affected")
    phases.append(
        {
            "phase": "affected_validation",
            "start_s": phase_start - started,
            "end_s": time.monotonic() - started,
            "completed_units": len(commands),
        }
    )
    progress(started, "affected_validation", "after_subprocesses", completed_units=len(commands))

    e2e = e2e_command_specs(root, private_root / "e2e")
    phase_start = time.monotonic()
    progress(started, "e2e", "before_subprocesses", total_units=len(e2e))
    validation_receipts.extend(_run_specs(root, e2e, private_root / "logs" / "e2e"))
    phases.append(
        {
            "phase": "e2e",
            "start_s": phase_start - started,
            "end_s": time.monotonic() - started,
            "completed_units": len(e2e),
        }
    )
    progress(started, "e2e", "after_subprocesses", completed_units=len(e2e))

    phase_start = time.monotonic()
    progress(started, "measurement", "before_scripted_callbacks")
    diagnosis = diagnose_exp7406_overflow(root)
    selection = freeze_game_rotation(root)
    callback_panel = run_callback_matrix(private_root / "callback_matrix")
    progress(
        started,
        "measurement",
        "after_scripted_callbacks",
        completed_units=len(callback_panel["control_rows"]),
    )
    progress(started, "measurement", "before_scored_benchmark", total_units=3)
    scored_panel = run_scored_plumbing(selection["selected_games"], private_root / "scored")
    progress(
        started, "measurement", "after_scored_benchmark", completed_units=len(scored_panel["rows"])
    )
    redirect = inspect_redirect_ledger(root)
    phases.append(
        {
            "phase": "measurement",
            "start_s": phase_start - started,
            "end_s": time.monotonic() - started,
            "completed_units": len(callback_panel["control_rows"]) + len(scored_panel["rows"]),
        }
    )

    sidecar_dir = root / RAW_DIR / "sidecars"
    historical_sidecar = sidecar_dir / "historical_exp7406_diagnosis.json"
    scripted_sidecar = sidecar_dir / "scripted_transport.json"
    atomic_json(
        historical_sidecar,
        {"scope": "historical", "counts_as_current": False, "diagnosis": diagnosis},
    )
    atomic_json(
        scripted_sidecar,
        {
            "scope": "development_proxy",
            "counts_as_current": False,
            "callback_panel": callback_panel,
            "scored_panel": scored_panel,
        },
    )
    sidecars = {
        "historical": {
            "path": historical_sidecar.relative_to(root).as_posix(),
            "sha256": sha256_file(historical_sidecar),
        },
        "scripted": {
            "path": scripted_sidecar.relative_to(root).as_posix(),
            "sha256": sha256_file(scripted_sidecar),
        },
    }

    candidate = private_root / "candidate.json"
    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=max(time.monotonic() - started, 1e-9),
        phase_spans=phases,
        preconditions_checked=preconditions,
        source_artifact_hashes=source_hashes,
        selection=selection,
        diagnosis=diagnosis,
        callback_panel=callback_panel,
        scored_panel=scored_panel,
        redirect_disposition=redirect,
        validation_receipts=[
            *validation_receipts,
            *[
                {
                    "name": name,
                    "command_argv": ["pending", name],
                    "environment": {},
                    "exit_code": 0,
                    "duration_s": 0.0,
                    "log_sha256": "sha256:" + "0" * 64,
                    "passed": True,
                    "timed_out": False,
                }
                for name in REQUIRED_TERMINAL
            ],
        ],
        provenance_sidecars=sidecars,
    )
    atomic_json(candidate, artifact)
    phase_start = time.monotonic()
    progress(started, "terminal_readers", "before_subprocesses", total_units=3)
    terminal_receipts = _run_specs(
        root, _terminal_specs(root, candidate), private_root / "logs" / "terminal"
    )
    validation_receipts.extend(terminal_receipts)
    phases.append(
        {
            "phase": "terminal_readers",
            "start_s": phase_start - started,
            "end_s": time.monotonic() - started,
            "completed_units": len(terminal_receipts),
        }
    )
    progress(
        started, "terminal_readers", "after_subprocesses", completed_units=len(terminal_receipts)
    )

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=max(time.monotonic() - started, 1e-9),
        phase_spans=phases,
        preconditions_checked=preconditions,
        source_artifact_hashes=source_hashes,
        selection=selection,
        diagnosis=diagnosis,
        callback_panel=callback_panel,
        scored_panel=scored_panel,
        redirect_disposition=redirect,
        validation_receipts=validation_receipts,
        provenance_sidecars=sidecars,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"terminal artifact validation failed: {errors}")
    progress(started, "write", "before_atomic_terminal_write")
    atomic_json(root / RESULT_PATH, artifact)
    progress(started, "write", "after_atomic_terminal_write", path=RESULT_PATH.as_posix())
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed execution date for the thin entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=(RUN_DATE,))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Experiment 7411 from the repository-root entrypoint."""

    args = parse_args(argv)
    run_experiment(REPO_ROOT, args.date)
    return 0
