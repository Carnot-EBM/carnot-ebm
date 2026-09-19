"""Qualify the ARC receipt boundary without current model work.

Scripted HTTP bytes and old model-shaped counters are useful audit evidence.
They are not calls made by this run. This producer keeps those bytes in typed
sidecars and builds current provenance only from the shipped owned-event schema.

Spec refs: REQ-ARC-WMTE-7424 and SCENARIO-ARC-WMTE-7424-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
import urllib.request

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7411_v650_arc_call_budget as exp7411
from carnot.agentic import arc_induction_tool_loop as tool_loop
from carnot.agentic.arc_request_budget import (
    EpisodeRequestBudget,
    LateRequestCompletion,
    RequestBudgetError,
    attach_request_budget,
)
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7424-arc-receipt-boundary"
MILESTONE = "2026.09.651"
RUN_DATE = "20260919"
SCHEMA = "carnot.exp7424.v651.arc_receipt_boundary.v1"
RESULT_PATH = Path("results/experiment_7424_v651_arc_receipt_boundary.json")
RAW_DIR = Path("results/raw/experiment_7424_v651_arc_receipt_boundary")
MODULE_PATH = Path("python/carnot/experiment_7424_v651_arc_receipt_boundary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7424_v651_arc_receipt_boundary.py")
TEST_PATH = Path("tests/python/test_experiment_7424_v651_arc_receipt_boundary.py")
CURRENT_RECEIPT_TEST_PATH = Path("tests/python/test_current_work_receipt.py")
BUDGET_TEST_PATH = Path("tests/python/test_arc_request_budget.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXP7411_PATH = Path("results/experiment_7411_v650_arc_call_budget.json")
RANDOM_SEED = 7_424_651
REQUEST_LIMIT = 2
ORIGINAL_REPLAY_NAME = "exp7411_original_adversarial_replay"
CONTRADICTION_CONTROL_NAME = "current_invocation_contradiction_control"
REQUIRED_E2E = ("e2e_009", "e2e_010", "e2e_offline_smoke")
REQUIRED_TERMINAL = (
    "fresh_process_replay",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
TYPED_INVOCATION_FIELDS = frozenset(current_work_receipt.ZERO_INVOCATION_COUNTS)
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
    Path("python/carnot/experiment_7411_v650_arc_call_budget.py"),
    Path("python/carnot/agentic/arc_request_budget.py"),
    Path("python/carnot/agentic/arc_inference_boundary.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    EXP7411_PATH,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    CURRENT_RECEIPT_TEST_PATH,
    BUDGET_TEST_PATH,
)


def utc_now() -> str:
    """Return an aware UTC timestamp for a measured boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush one truthful phase or long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7424] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for an independently reproducible identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without interpreting their evidence scope."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON object after its bytes reach storage."""

    current_work_receipt.atomic_json(path, value)


def load_object(path: Path) -> JsonDict:
    """Load one JSON object, or return an empty object for unavailable bytes."""

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
    principle: str,
    operator: str = "==",
) -> JsonDict:
    """Record one exact comparison and why that comparison exists."""

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
        "principle": principle,
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every failed upstream comparison instead of hiding later failures."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "failed_checks": failures,
        "first_failure": failures[0] if failures else None,
    }


def find_typed_invocation_paths(value: Any) -> list[JsonDict]:
    """Find nonzero model-counter fields anywhere in a JSON-shaped value."""

    found: list[JsonDict] = []

    def visit(node: Any, parts: list[str]) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                path = [*parts, str(key)]
                if (
                    key in TYPED_INVOCATION_FIELDS
                    and isinstance(child, (int, float))
                    and not isinstance(child, bool)
                    and child != 0
                ):
                    found.append(
                        {
                            "path": "$." + ".".join(path),
                            "field": str(key),
                            "value": child,
                        }
                    )
                visit(child, path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, [*parts, str(index)])

    visit(value, [])
    return sorted(found, key=lambda row: str(row["path"]))


def inspect_original_disqualification(root: Path) -> JsonDict:
    """Read the frozen Exp7411 failure without changing or promoting it."""

    artifact = load_object(root / EXP7411_PATH)
    findings = [
        str(row.get("kind"))
        for row in artifact.get("corrigendum_pending", [])
        if isinstance(row, Mapping)
    ]
    receipts = [
        deepcopy(dict(row))
        for row in artifact.get("validation_receipts", [])
        if isinstance(row, Mapping) and row.get("name") == "adversarial_verify"
    ]
    flags = {
        "status": artifact.get("status"),
        "verdict_class": artifact.get("verdict_class"),
        "flagged_adversarial": artifact.get("flagged_adversarial"),
    }
    typed = find_typed_invocation_paths(artifact)
    preserved = (
        flags
        == {
            "status": "complete_disqualified_required_evidence",
            "verdict_class": "disqualified",
            "flagged_adversarial": True,
        }
        and findings[:2] == ["INFERENCE_PROVENANCE_CONTRADICTION", "SUBSTRATE_CLASS_MISMATCH"]
        and typed
        == [
            {
                "path": "$.overflow_diagnosis.generation_calls_attempted",
                "field": "generation_calls_attempted",
                "value": 35,
            }
        ]
        and len(receipts) == 1
        and receipts[0].get("exit_code") == 1
    )
    return {
        "artifact_path": EXP7411_PATH.as_posix(),
        "artifact_sha256": sha256_file(root / EXP7411_PATH)
        if (root / EXP7411_PATH).is_file()
        else None,
        "original_flags": flags,
        "original_findings": findings,
        "typed_counter_paths": typed,
        "original_validation_receipt": receipts[0] if len(receipts) == 1 else None,
        "original_disqualification_preserved": preserved,
    }


def contradiction_fixture() -> JsonDict:
    """Return a private mutation that must fail the current-provenance guard."""

    counts = deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)
    counts["generation_calls_attempted"] = 1
    return {
        "schema": "carnot.exp7424.current_invocation_contradiction.v1",
        "experiment_id": "exp7424-current-invocation-contradiction-control",
        "status": "complete_disqualified_control",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": counts,
        "inference_substrate": "host CPU no-model contradiction control",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": 1.0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:" + "0" * 64,
        "honest_verdict": "complete_disqualified_control",
        "verdict_class": "disqualified",
        "flagged_adversarial": True,
        "arc_receipt_boundary_ready_score": 0,
        "promotion_score": 0,
    }


def _scripted_open(
    events: list[JsonDict],
    *,
    branch: str,
    request_id: str,
    content: str = "scripted",
    error: BaseException | None = None,
    entered: threading.Event | None = None,
    release: threading.Event | None = None,
) -> Any:
    """Build one HTTP fixture that records complete request and response bytes."""

    def open_request(request: Any, timeout: float | None = None) -> io.BytesIO:
        payload = json.loads(request.data)
        if entered is not None:
            entered.set()
        if release is not None:
            release.wait(timeout=5.0)
        if error is not None:
            events.append(
                {
                    "event_type": "scripted_http_exception",
                    "branch": branch,
                    "request_id": request_id,
                    "request": payload,
                    "timeout_s": timeout,
                    "exception": f"{type(error).__name__}: {error}",
                    "counts_as_current_model_event": False,
                }
            )
            raise error
        response = {"choices": [{"message": {"content": content}}]}
        events.append(
            {
                "event_type": "scripted_http_exchange",
                "branch": branch,
                "request_id": request_id,
                "request": payload,
                "response": response,
                "timeout_s": timeout,
                "counts_as_current_model_event": False,
            }
        )
        return io.BytesIO(json.dumps(response).encode())

    return open_request


def _request_rows(control_id: str, receipt: Mapping[str, Any]) -> list[JsonDict]:
    """Expose permit and terminal accounting without copying scripted payloads."""

    limit = int(receipt["limit"])
    return [
        {
            "control_id": control_id,
            "episode_id": row["episode_id"],
            "request_id": row["request_id"],
            "branch": row["branch"],
            "permit_granted": True,
            "attempted": True,
            "dispatched": control_id != "interrupted_child_release",
            "cancelled": row["disposition"] == "cancelled",
            "remaining_capacity": limit - int(row["reservation_index"]) - 1,
            "terminal_result": row["disposition"],
            "recovered_after_restart": row.get("recovered_after_restart") is True,
        }
        for row in receipt["callback_rows"]
    ]


def run_callback_panel(work_dir: Path) -> JsonDict:
    """Drive the real E3 proposer boundary with scripted transport outcomes."""

    from carnot import experiment_7234_v637_arc_scored_dryrun as scored

    work_dir.mkdir(parents=True, exist_ok=True)
    proposer = exp7411._fixture_proposer()
    policy, factory = scored.build_disposable_submitted_policy("r11l", proposer)
    proposer = policy._proposer()
    events: list[JsonDict] = []
    controls: list[tuple[str, JsonDict]] = []
    rejections: list[JsonDict] = []
    old_open = urllib.request.urlopen
    child: subprocess.Popen[str] | None = None
    try:
        limit_budget = EpisodeRequestBudget("limit", limit=REQUEST_LIMIT, deadline_s=60)
        attach_request_budget(proposer, limit_budget)
        for branch in ("primary", "refinement"):
            request_id = f"limit-{branch}"
            urllib.request.urlopen = _scripted_open(
                events, branch=branch, request_id=request_id, content=f"{branch}-response"
            )
            exp7411._scripted_post(proposer, branch, request_id)
        before_refusal = len(events)
        try:
            exp7411._scripted_post(proposer, "repair", "limit-repair-refused")
        except RequestBudgetError:
            rejections.append(
                {
                    "control_id": "limit",
                    "episode_id": "limit",
                    "request_id": "limit-repair-refused",
                    "branch": "repair",
                    "permit_granted": False,
                    "attempted": False,
                    "dispatched": False,
                    "cancelled": False,
                    "remaining_capacity": 0,
                    "terminal_result": "rejected_before_dispatch",
                    "recovered_after_restart": False,
                }
            )
        third_rejected = len(events) == before_refusal
        controls.append(("limit", limit_budget.receipt()))

        exception_budget = EpisodeRequestBudget("exception", limit=2, deadline_s=60)
        attach_request_budget(proposer, exception_budget)
        urllib.request.urlopen = _scripted_open(
            events,
            branch="repair",
            request_id="exception-repair",
            error=RuntimeError("scripted transport exception"),
        )
        try:
            exp7411._scripted_post(proposer, "repair", "exception-repair")
        except RuntimeError:
            pass
        controls.append(("exception", exception_budget.receipt()))

        cancellation_budget = EpisodeRequestBudget("cancellation", limit=2, deadline_s=60)
        attach_request_budget(proposer, cancellation_budget)
        entered = threading.Event()
        release = threading.Event()
        urllib.request.urlopen = _scripted_open(
            events,
            branch="refinement",
            request_id="cancellation-refinement",
            entered=entered,
            release=release,
        )
        worker_errors: list[str] = []

        def cancellation_worker() -> None:
            try:
                exp7411._scripted_post(proposer, "refinement", "cancellation-refinement")
            except BaseException as exc:
                worker_errors.append(type(exc).__name__)

        worker = threading.Thread(target=cancellation_worker, daemon=True)
        worker.start()
        entered.wait(timeout=5.0)
        cancellation_budget.cancel("episode_cancelled")
        release.set()
        worker.join(timeout=5.0)
        controls.append(("cancellation", cancellation_budget.receipt()))

        first = EpisodeRequestBudget("restart", limit=2, deadline_s=60)
        attach_request_budget(proposer, first)
        urllib.request.urlopen = _scripted_open(
            events, branch="primary", request_id="restart-primary"
        )
        exp7411._scripted_post(proposer, "primary", "restart-primary")
        restarted = EpisodeRequestBudget.from_receipt(first.receipt(), deadline_s=60)
        attach_request_budget(proposer, restarted)
        replay_refused = False
        try:
            exp7411._scripted_post(proposer, "primary", "restart-primary")
        except RequestBudgetError:
            replay_refused = True
        urllib.request.urlopen = _scripted_open(
            events, branch="repair", request_id="restart-repair"
        )
        exp7411._scripted_post(proposer, "repair", "restart-repair")
        controls.append(("restart", restarted.receipt()))

        child_budget = EpisodeRequestBudget("child", limit=2, deadline_s=60)
        reservation = child_budget.reserve(branch="refinement", request_id="interrupted-child")
        child = subprocess.Popen(  # noqa: S603 - fixed interpreter and literal fixture code.
            [
                sys.executable,
                "-u",
                "-c",
                "import sys; print('ready', flush=True); sys.stdin.read()",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert child.stdout is not None
        child_ready = child.stdout.readline().strip() == "ready"
        child_budget.cancel("owner_interrupted")
        assert child.stdin is not None
        child.stdin.close()
        child_returncode = child.wait(timeout=5.0)
        try:
            reservation.complete()
        except LateRequestCompletion:
            pass
        controls.append(("interrupted_child_release", child_budget.receipt()))
        interrupted_child_released = child_ready and child_returncode == 0
    finally:
        urllib.request.urlopen = old_open
        if child is not None and child.poll() is None:  # pragma: no cover - OS failure cleanup.
            child.terminate()  # pragma: no cover
            child.wait(timeout=5.0)  # pragma: no cover

    rows = [row for control_id, receipt in controls for row in _request_rows(control_id, receipt)]
    rows.extend(rejections)
    receipts_ok = all(
        receipt["attempted"] <= REQUEST_LIMIT
        and receipt["accounting_valid"] is True
        and receipt["in_flight"] == 0
        for _control_id, receipt in controls
    )
    terminal_results = {str(row["terminal_result"]) for row in rows}
    restart_receipt = dict(next(receipt for name, receipt in controls if name == "restart"))
    restart_ownership = (
        replay_refused
        and restart_receipt["replayed_completed_refusals"] == 1
        and all(row["recovered_after_restart"] for row in restart_receipt["callback_rows"][:1])
    )
    cancellation_passed = worker_errors == ["LateRequestCompletion"] and not worker.is_alive()
    return {
        "factory": factory["factory"],
        "policy_class": factory["policy_class"],
        "request_budget_rows": rows,
        "scripted_events": events,
        "third_rejected_before_dispatch": third_rejected,
        "restart_ownership_passed": restart_ownership,
        "interrupted_child_released": interrupted_child_released,
        "cancellation_worker_released": cancellation_passed,
        "terminal_dispositions_observed": sorted(terminal_results),
        "all_controls_passed": receipts_ok
        and third_rejected
        and restart_ownership
        and interrupted_child_released
        and cancellation_passed
        and {"completed", "failed", "cancelled", "rejected_before_dispatch"} <= terminal_results,
    }


def reduce_request_budget_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Independently reduce permits, attempts, dispatches, and terminal results."""

    attempted_rows = [row for row in rows if row.get("attempted") is True]
    dispositions = Counter(str(row.get("terminal_result")) for row in attempted_rows)
    per_episode = Counter(str(row.get("episode_id")) for row in attempted_rows)
    terminal = sum(dispositions[name] for name in ("completed", "failed", "cancelled"))
    return {
        "permit_granted": sum(row.get("permit_granted") is True for row in rows),
        "attempted": len(attempted_rows),
        "dispatched": sum(row.get("dispatched") is True for row in rows),
        "completed": dispositions["completed"],
        "failed": dispositions["failed"],
        "cancelled": dispositions["cancelled"],
        "in_flight": dispositions["in_flight"],
        "rejected_before_dispatch": sum(
            row.get("terminal_result") == "rejected_before_dispatch" for row in rows
        ),
        "accounting_valid": len(attempted_rows) == terminal + dispositions["in_flight"],
        "all_attempts_terminal": len(attempted_rows) == terminal,
        "budget_violations": sum(count > REQUEST_LIMIT for count in per_episode.values()),
    }


def write_evidence_sidecars(
    *,
    root: Path,
    sidecar_dir: Path,
    historical_payload: Mapping[str, Any],
    scripted_payload: Mapping[str, Any],
) -> tuple[dict[str, JsonDict], list[JsonDict]]:
    """Write typed immutable sidecars and return opaque current-receipt references."""

    payloads = (
        (
            "historical_receipt_fixture",
            "historical_model_receipts",
            historical_payload,
            "frozen_historical_model_shaped_evidence",
        ),
        (
            "scripted_request_fixture",
            "simulated_transport_events",
            scripted_payload,
            "complete_scripted_http_and_contradiction_evidence",
        ),
    )
    manifest: dict[str, JsonDict] = {}
    references: list[JsonDict] = []
    for name, scope, payload, evidence_type in payloads:
        suffix = canonical_hash(payload).removeprefix("sha256:")[:16]
        reference = current_work_receipt.write_immutable_sidecar(
            sidecar_dir / f"{name}-{suffix}.json",
            scope=scope,
            payload=payload,
            root=root,
        )
        references.append(reference)
        manifest[name] = {
            **deepcopy(reference),
            "evidence_type": evidence_type,
            "counts_as_current_model_event": False,
            "immutable": True,
        }
    return manifest, references


def affected_manifest() -> validation_contract.AffectedManifest:
    """Freeze the exact implementation and shared-module test scope."""

    return validation_contract.AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=(
            TEST_PATH.as_posix(),
            CURRENT_RECEIPT_TEST_PATH.as_posix(),
            BUDGET_TEST_PATH.as_posix(),
        ),
        changed_modules=(MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed Exp7358 plan with private coverage and basetemp parents."""

    return validation_contract.build_command_plan(root, affected_manifest(), private_root)


def e2e_command_specs(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Reuse the shipped E2E-009, E2E-010, and LLM-off smoke plan."""

    return exp7411.e2e_command_specs(root, private_root)


def run_specs(
    root: Path,
    specs: Sequence[validation_scope.CommandSpec],
    log_dir: Path,
) -> list[JsonDict]:
    """Stream each bounded child and retain its command-local environment."""

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


def classify_expected_failure(
    receipt: Mapping[str, Any], *, required_tokens: Sequence[str]
) -> JsonDict:
    """Treat an exact nonzero guard exit as a passing negative control."""

    row = deepcopy(dict(receipt))
    output = str(row.get("output_tail") or "")
    control_passed = (
        row.get("exit_code") == 1
        and row.get("timed_out") is False
        and all(token in output for token in required_tokens)
    )
    row["expected_exit_code"] = 1
    row["passed"] = control_passed
    row["control_passed"] = control_passed
    return row


def validation_receipts_pass(
    receipts: Sequence[Mapping[str, Any]], required: Sequence[str]
) -> bool:
    """Require one receipt per check and preserve expected nonzero control exits."""

    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        and next(row for row in receipts if row.get("name") == name).get("exit_code")
        == next(row for row in receipts if row.get("name") == name).get("expected_exit_code", 0)
        for name in required
    )


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build fresh-process reduction and unchanged terminal reader commands."""

    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "fresh_process_replay",
            (
                python,
                "-u",
                "-c",
                (
                    "import json,sys; from pathlib import Path; "
                    "from carnot.experiment_7424_v651_arc_receipt_boundary import "
                    "independent_reduce_file,validate_artifact; p=Path(sys.argv[1]); "
                    "r=independent_reduce_file(p); e=validate_artifact(p); "
                    "print(json.dumps({'reduction':r,'errors':e},sort_keys=True)); "
                    "raise SystemExit(0 if r['matches_declared'] and not e else 1)"
                ),
                str(candidate),
            ),
            "fresh-process cold artifact replay",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "unchanged adversarial verifier",
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
            "strict verdict-row consistency",
        ),
    ]


def _scripted_manifest_valid(
    manifest: Mapping[str, Any], references: Sequence[Mapping[str, Any]]
) -> bool:
    """Match each typed manifest entry to one opaque current-receipt reference."""

    if set(manifest) != {"historical_receipt_fixture", "scripted_request_fixture"}:
        return False
    ref_keys = {(row.get("path"), row.get("sha256"), row.get("scope")) for row in references}
    return (
        all(
            row.get("counts_as_current_model_event") is False
            and row.get("immutable") is True
            and (row.get("path"), row.get("sha256"), row.get("scope")) in ref_keys
            for row in manifest.values()
            if isinstance(row, Mapping)
        )
        and len(ref_keys) == 2
    )


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain each ordinary field without wrapping its machine-readable value."""

    specific = {
        "schema": "A versioned top-level schema binds experiment identity, milestone, and terminal status.",
        "run_date": "The requested date is separate from measured UTC start and end times.",
        "preconditions_checked": "Exact resource, path, identity, and observed values precede dependent work.",
        "MODEL_SPECS": "An empty list states that this run used no current LLM.",
        "model_invoked": "False excludes archived and scripted transport bytes from current model use.",
        "invocation_counts": "Only owned current events contribute to load and generation counters.",
        "inference_substrate": "A plain string describes the current CPU qualification work.",
        "inference_substrate_class": "The no-model class cannot be inferred from padding or historical calls.",
        "execution_venue": "The closed host value is separate from CPU and platform details.",
        "duration_s": "Monotonic current work is measured, with validation and cold-start costs separated.",
        "phase_spans": "Real phase times retain completed units and resumable evidence references.",
        "random_seed": "The fixed seed identifies deterministic fixture and reduction choices.",
        "reproducibility_checksum": "The checksum binds code, protocol, inputs, raw rows, and validation scope.",
        "source_artifact_hashes": "Exact input hashes retain the original Exp7411 determination flags.",
        "rows": "Every request permit, attempt, refusal, cancellation, and terminal result remains visible.",
        "sample_size_budget": "Planned, attempted, complete, failed, censored, and unstarted units stay separate.",
        "acceptance_gate_results": "Each check states category, operator, expected, observed, pass, and principle.",
        "gate_check_summary": "Each failed gate retains its upstream path, check, field, expected, and observed value.",
        "verifier_is_oracle": "True states that verifier behavior defines this qualification, not ARC correctness.",
        "honest_verdict": "A complete verdict limits the finding to receipt-boundary readiness.",
        "verdict_class": "Null states that no live-model benefit or ARC efficacy was measured.",
        "flagged_adversarial": "A critical current candidate finding prevents readiness.",
        "validation_receipts": "Exact scoped arguments, environments, exits, durations, and hashed logs remain auditable.",
        "promotion_score": "Zero forbids rollout, publication, or generator-weight updates.",
        "arc_receipt_boundary_ready_score": "One requires current isolation, real callback limits, and all affected checks.",
        "scripted_evidence_manifest": "Typed hash-bound sidecars keep fixtures outside current model events.",
        "request_budget_rows": "Each callback shows its permit, attempt, cancellation, capacity, and terminal result.",
        "live_efficacy_score": "Zero marks this CPU qualification as plumbing, not a live ARC claim.",
    }
    return {
        key: specific.get(key, f"The {key} field retains directly auditable experiment evidence.")
        for key in keys
    }


def _duration_breakdown(phase_spans: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Separate validation, model, measurement, and cold-process wall time."""

    totals: Counter[str] = Counter()
    for row in phase_spans:
        phase = str(row.get("phase") or "other")
        elapsed = max(0.0, float(row.get("end_s") or 0) - float(row.get("start_s") or 0))
        if "validation" in phase or phase == "e2e":
            totals["validation_s"] += elapsed
        elif "replay" in phase or "terminal" in phase or "control" in phase:
            totals["cold_start_s"] += elapsed
        elif "measurement" in phase:
            totals["measurement_s"] += elapsed
        else:
            totals["other_s"] += elapsed
    return {
        "validation_s": totals["validation_s"],
        "model_s": 0.0,
        "cold_start_s": totals["cold_start_s"],
        "measurement_s": totals["measurement_s"],
        "other_s": totals["other_s"],
    }


def build_terminal_artifact(
    *,
    root: Path,
    started_at_utc: str,
    ended_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    callback_panel: Mapping[str, Any],
    scripted_evidence_manifest: Mapping[str, Any],
    sidecar_references: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    original_summary: Mapping[str, Any],
) -> JsonDict:
    """Build one independently reducible current-work receipt and terminal record."""

    spans = [deepcopy(dict(row)) for row in phase_spans]
    references = [deepcopy(dict(row)) for row in sidecar_references]
    current = current_work_receipt.build_current_work_receipt(
        run_id=EXPERIMENT_ID,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_runtime_receipt_validation_no_llm",
        inference_substrate_details={
            "host": platform.node(),
            "python": platform.python_version(),
            "scripted_transport": True,
            "current_llm_calls": 0,
            "claim_scope": "CPU plumbing only; no ARC solve or live-model claim",
        },
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        sidecar_references=references,
        phase_spans=spans,
        small_ebm_training={
            "performed": False,
            "receipt_class": "small_ebm_training",
            "reason": "receipt isolation and request-budget controls require no fitted energy head",
            "current_llm_counts_affected": False,
        },
    )
    request_rows = [
        deepcopy(dict(row))
        for row in callback_panel.get("request_budget_rows", [])
        if isinstance(row, Mapping)
    ]
    reduction = reduce_request_budget_rows(request_rows)
    receipt_errors = current_work_receipt.validate_current_work_receipt(current, root=root)
    preconditions_pass = bool(preconditions_checked) and all(
        row.get("passed") is True for row in preconditions_checked
    )
    original_pass = (
        original_summary.get("status") == "complete_disqualified_required_evidence"
        and original_summary.get("verdict_class") == "disqualified"
        and original_summary.get("flagged_adversarial") is True
        and original_summary.get("finding_kinds")
        == ["INFERENCE_PROVENANCE_CONTRADICTION", "SUBSTRATE_CLASS_MISMATCH"]
        and original_summary.get("typed_counter_locations")
        == ["$.overflow_diagnosis.generation_calls_attempted"]
    )
    callback_pass = (
        callback_panel.get("all_controls_passed") is True
        and reduction["accounting_valid"] is True
        and reduction["all_attempts_terminal"] is True
        and reduction["budget_violations"] == 0
    )
    manifest_pass = _scripted_manifest_valid(scripted_evidence_manifest, references)
    required_names = (
        *validation_scope.REQUIRED_CHECK_NAMES,
        *REQUIRED_E2E,
        ORIGINAL_REPLAY_NAME,
        CONTRADICTION_CONTROL_NAME,
        *REQUIRED_TERMINAL,
    )
    validation_pass = validation_receipts_pass(validation_receipts, required_names)
    gates = [
        gate_row(
            "preconditions",
            "precondition",
            True,
            preconditions_pass,
            upstream="preconditions_checked",
            artifact_field="all_branch_local_inputs_authenticated",
            principle="Unavailable branch-local evidence never promotes readiness.",
        ),
        gate_row(
            "original_disqualification",
            "historical_preservation",
            True,
            original_pass,
            upstream=EXP7411_PATH.as_posix(),
            artifact_field="original_status_findings_and_typed_counter_location",
            principle="A producer repair must not erase the original disqualification.",
        ),
        gate_row(
            "current_work_receipt",
            "provenance",
            [],
            receipt_errors,
            upstream="carnot.reporting.current_work_receipt",
            artifact_field="owned_current_event_ledger",
            principle="Only owned current events can change current invocation counts.",
        ),
        gate_row(
            "scripted_sidecar_boundary",
            "provenance",
            True,
            manifest_pass,
            upstream="scripted_evidence_manifest",
            artifact_field="typed_hash_bound_non_current_sidecars",
            principle="Scripted and historical model-shaped bytes stay opaque to current provenance.",
        ),
        gate_row(
            "callback_lifecycle",
            "safety",
            True,
            callback_pass,
            upstream="request_budget_rows",
            artifact_field="permits_attempts_capacity_and_terminal_results",
            principle="A permit is consumed before dispatch and every attempt becomes terminal.",
        ),
        gate_row(
            "restart_ownership",
            "safety",
            True,
            callback_panel.get("restart_ownership_passed") is True,
            upstream="request_budget_rows",
            artifact_field="recovered_request_identity",
            principle="A restart cannot replay a completed request as new work.",
        ),
        gate_row(
            "interrupted_child_release",
            "safety",
            True,
            callback_panel.get("interrupted_child_released") is True,
            upstream="request_budget_rows",
            artifact_field="owned_fixture_child_release",
            principle="Cancellation releases only the fixture child owned by this control.",
        ),
        gate_row(
            "required_validation",
            "required_validation",
            True,
            validation_pass,
            upstream="validation_receipts",
            artifact_field="affected_e2e_negative_and_terminal_checks",
            principle="Readiness requires every scoped check and unchanged terminal reader.",
        ),
        gate_row(
            "automatic_promotion",
            "promotion",
            0,
            0,
            upstream="protocol",
            artifact_field="promotion_score",
            principle="CPU qualification cannot update weights, roll out, or publish externally.",
        ),
    ]
    ready = int(all(row["passed"] for row in gates))
    sample_size = {
        "planned_units": 5,
        "attempted_units": 5,
        "completed_units": 5,
        "failed_units": 0,
        "censored_units": 0,
        "unstarted_units": 0,
        "independent_groups": [
            "dispatch_limit",
            "transport_exception",
            "late_cancellation",
            "cold_restart",
            "owned_child_release",
        ],
        "stopping_rule": "Run each frozen CPU control once and stop after terminal accounting.",
        "request_limit_per_episode": REQUEST_LIMIT,
    }
    manifest = affected_manifest()
    manifest_value = {**asdict(manifest), "frozen_before_checks": True}
    manifest_value["manifest_sha256"] = canonical_hash(manifest_value)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_null_arc_receipt_boundary_ready"
        if ready
        else "complete_disqualified_required_evidence",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        **current,
        "duration_breakdown_s": _duration_breakdown(spans),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "affected_file_validation_manifest": manifest_value,
        "rows": [{"row_type": "request_budget_callback", **deepcopy(row)} for row in request_rows],
        "request_budget_rows": request_rows,
        "sample_size_budget": sample_size,
        "raw_reduction": reduction,
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": (
            "complete_null_arc_receipt_boundary_ready_no_live_efficacy_claim"
            if ready
            else "complete_disqualified_required_evidence"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "field_principles": {},
        "promotion_score": 0,
        "arc_receipt_boundary_ready_score": ready,
        "scripted_evidence_manifest": deepcopy(dict(scripted_evidence_manifest)),
        "live_efficacy_score": 0,
        "solve_provenance": "no_game_solve_cpu_transport_fixture",
        "solve_credit": 0,
        "original_disqualification_preserved": deepcopy(dict(original_summary)),
        "small_ebm_training": current["small_ebm_training"],
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "research_conductor_changed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute request accounting and current typed-counter isolation."""

    rows = value.get("request_budget_rows")
    rows = rows if isinstance(rows, list) else []
    reduction = reduce_request_budget_rows([row for row in rows if isinstance(row, Mapping)])
    typed = find_typed_invocation_paths(value)
    return {
        **reduction,
        "nonzero_typed_invocation_paths": typed,
        "current_invocation_isolated": not typed,
    }


def independent_reduce_file(path: Path) -> JsonDict:
    """Cold-load a candidate and compare its raw independent reduction."""

    artifact = load_object(path)
    reduced = independent_reduce(artifact)
    declared = artifact.get("raw_reduction")
    comparable = {key: reduced[key] for key in declared} if isinstance(declared, Mapping) else {}
    return {
        **reduced,
        "matches_declared": comparable == declared,
        "declared_ready": artifact.get("arc_receipt_boundary_ready_score"),
    }


def validate_artifact(value: Mapping[str, Any] | Path, *, root: Path | None = None) -> list[str]:
    """Cold-check current provenance, raw rows, gates, sidecars, and checksum."""

    artifact = load_object(value) if isinstance(value, Path) else deepcopy(dict(value))
    base = (root or REPO_ROOT).resolve()
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("identity_invalid")
    receipt_errors = current_work_receipt.validate_current_work_receipt(artifact, root=base)
    if artifact.get("MODEL_SPECS") != []:
        receipt_errors.append("MODEL_SPECS_must_be_empty")
    if receipt_errors:
        errors.append("current_receipt_invalid")
    if (
        artifact.get("inference_substrate_class") != "no_model_load"
        or artifact.get("execution_venue") != "host"
        or not isinstance(artifact.get("inference_substrate"), str)
    ):
        errors.append("substrate_invalid")
    reduced = independent_reduce(artifact)
    declared = artifact.get("raw_reduction")
    comparable = {key: reduced[key] for key in declared} if isinstance(declared, Mapping) else {}
    if comparable != declared:
        errors.append("raw_reduction_mismatch")
    if reduced["nonzero_typed_invocation_paths"]:
        errors.append("typed_invocation_leak")
    manifest = artifact.get("scripted_evidence_manifest")
    references = artifact.get("receipt_sidecars")
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(references, list)
        or not _scripted_manifest_valid(
            manifest, [row for row in references if isinstance(row, Mapping)]
        )
    ):
        errors.append("scripted_manifest_invalid")
    gates = artifact.get("acceptance_gate_results")
    gates = gates if isinstance(gates, list) else []
    ready = int(bool(gates) and all(row.get("passed") is True for row in gates))
    if artifact.get("arc_receipt_boundary_ready_score") != ready:
        errors.append("readiness_mismatch")
    if artifact.get("promotion_score") != 0 or artifact.get("live_efficacy_score") != 0:
        errors.append("score_invalid")
    if artifact.get("verdict_class") not in {"null", "disqualified"}:
        errors.append("verdict_invalid")
    if ready and (
        artifact.get("verdict_class") != "null"
        or not str(artifact.get("honest_verdict") or "").startswith("complete_")
        or artifact.get("flagged_adversarial") is not False
    ):
        errors.append("verdict_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("checksum_mismatch")
    return list(dict.fromkeys(errors))


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, JsonDict], JsonDict]:
    """Authenticate exact local inputs and the unchanged Exp7411 disposition."""

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
                principle="Dependent work starts only after exact source bytes are available.",
            )
        )
        if available:
            hashes[relative.as_posix()] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "role": "historical_disqualified_input"
                if relative == EXP7411_PATH
                else "current_input",
            }
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        gate_row(
            "driving_requirement",
            "precondition",
            True,
            "REQ-ARC-WMTE-7424" in spec_text,
            upstream=SPEC_PATH.as_posix(),
            artifact_field="REQ-ARC-WMTE-7424",
            principle="The implementation follows a named requirement and scenarios.",
        )
    )
    original = inspect_original_disqualification(root)
    checks.append(
        gate_row(
            "original_disqualification_available",
            "precondition",
            True,
            original["original_disqualification_preserved"],
            upstream=EXP7411_PATH.as_posix(),
            artifact_field="status_verdict_findings_and_nested_counter",
            principle="The correction begins from the exact old failure, not a rewritten copy.",
        )
    )
    if EXP7411_PATH.as_posix() in hashes:
        hashes[EXP7411_PATH.as_posix()]["original_flags"] = deepcopy(original["original_flags"])
    exclusion = root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion.read_text(encoding="utf-8") if exclusion.is_file() else ""
    checks.append(
        gate_row(
            "new_boundary_not_retired_rerun",
            "precondition",
            False,
            "exp7424-arc-receipt-boundary" in exclusion_text,
            upstream="ops/exclusion_manifest.yaml",
            artifact_field=EXPERIMENT_ID,
            principle="A retired unchanged mechanism cannot be promoted as new work.",
        )
    )
    return checks, hashes, original


def _control_command(root: Path, name: str, candidate: Path) -> validation_scope.CommandSpec:
    """Build one unchanged adversarial-verifier negative-control command."""

    return validation_scope.CommandSpec(
        name,
        (str(root / ".venv/bin/python"), "-u", "scripts/adversarial_verify.py", str(candidate)),
        "unchanged adversarial verifier negative control",
    )


def _original_summary(original: Mapping[str, Any]) -> JsonDict:
    """Keep old flags and locations while model-shaped values stay in a sidecar."""

    return {
        "artifact_path": original.get("artifact_path"),
        "artifact_sha256": original.get("artifact_sha256"),
        "status": (original.get("original_flags") or {}).get("status"),
        "verdict_class": (original.get("original_flags") or {}).get("verdict_class"),
        "flagged_adversarial": (original.get("original_flags") or {}).get("flagged_adversarial"),
        "finding_kinds": list(original.get("original_findings") or [])[:2],
        "typed_counter_locations": [
            str(row.get("path"))
            for row in original.get("typed_counter_paths", [])
            if isinstance(row, Mapping)
        ],
    }


def _phase(
    phases: list[JsonDict], phase: str, phase_started: float, run_started: float, units: int
) -> None:
    """Append one measured span with a resumable evidence label."""

    phases.append(
        {
            "phase": phase,
            "start_s": phase_started - run_started,
            "end_s": time.monotonic() - run_started,
            "completed_units": units,
            "checkpoint_reference": f"{RAW_DIR.as_posix()}/{phase}",
        }
    )


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - CLI orchestration.
    """Run controls, scoped checks, exact readers, and atomic publication."""

    run_started = time.monotonic()
    monotonic_started_ns = time.monotonic_ns()
    started_at = utc_now()
    phases: list[JsonDict] = []
    validation_receipts: list[JsonDict] = []
    progress(run_started, "startup", "begin", completed_units=0)

    phase_started = time.monotonic()
    progress(run_started, "preconditions", "before", completed_units=0)
    preconditions, source_hashes, original = collect_preconditions(root)
    _phase(phases, "preconditions", phase_started, run_started, len(preconditions))
    progress(
        run_started,
        "preconditions",
        "after",
        completed_units=len(preconditions),
    )
    if not all(row["passed"] for row in preconditions):
        raise RuntimeError("Exp7424 branch-local preconditions failed")

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7424-", dir="/tmp"))
    manifest = affected_manifest()
    commands = build_validation_plan(root, private_root / "affected")
    plan_errors = validation_contract.validate_command_plan(root, manifest, commands)
    if plan_errors:
        raise RuntimeError(f"affected command plan invalid: {plan_errors}")

    phase_started = time.monotonic()
    progress(run_started, "affected_validation", "before_subprocesses", total_units=len(commands))
    validation_receipts.extend(run_specs(root, commands, private_root / "logs" / "affected"))
    _phase(phases, "affected_validation", phase_started, run_started, len(commands))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(commands),
    )

    e2e_specs = e2e_command_specs(root, private_root / "e2e")
    phase_started = time.monotonic()
    progress(run_started, "e2e", "before_subprocesses", total_units=len(e2e_specs))
    validation_receipts.extend(run_specs(root, e2e_specs, private_root / "logs" / "e2e"))
    _phase(phases, "e2e", phase_started, run_started, len(e2e_specs))
    progress(run_started, "e2e", "after_subprocesses", completed_units=len(e2e_specs))

    historical_copy = private_root / "controls" / "experiment_7411.json"
    historical_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / EXP7411_PATH, historical_copy)
    phase_started = time.monotonic()
    progress(run_started, "original_replay", "before_subprocess", completed_units=0)
    replay = run_specs(
        root,
        [_control_command(root, ORIGINAL_REPLAY_NAME, historical_copy)],
        private_root / "logs" / "original_replay",
    )[0]
    replay = classify_expected_failure(
        replay,
        required_tokens=("INFERENCE_PROVENANCE_CONTRADICTION", "SUBSTRATE_CLASS_MISMATCH"),
    )
    validation_receipts.append(replay)
    _phase(phases, "original_replay", phase_started, run_started, 1)
    progress(run_started, "original_replay", "after_subprocess", completed_units=1)

    phase_started = time.monotonic()
    progress(run_started, "measurement", "before_scripted_callbacks", completed_units=0)
    callback_panel = run_callback_panel(private_root / "callback_panel")
    _phase(
        phases,
        "measurement",
        phase_started,
        run_started,
        len(callback_panel["request_budget_rows"]),
    )
    progress(
        run_started,
        "measurement",
        "after_scripted_callbacks",
        completed_units=len(callback_panel["request_budget_rows"]),
    )

    contradiction = contradiction_fixture()
    contradiction_path = private_root / "controls" / "current_contradiction.json"
    atomic_json(contradiction_path, contradiction)
    phase_started = time.monotonic()
    progress(run_started, "contradiction_control", "before_subprocess", completed_units=0)
    contradiction_receipt = run_specs(
        root,
        [_control_command(root, CONTRADICTION_CONTROL_NAME, contradiction_path)],
        private_root / "logs" / "contradiction_control",
    )[0]
    contradiction_receipt = classify_expected_failure(
        contradiction_receipt,
        required_tokens=("INFERENCE_PROVENANCE_CONTRADICTION", "SUBSTRATE_CLASS_MISMATCH"),
    )
    validation_receipts.append(contradiction_receipt)
    _phase(phases, "contradiction_control", phase_started, run_started, 1)
    progress(run_started, "contradiction_control", "after_subprocess", completed_units=1)

    historical_payload = {
        "evidence_scope": "historical_model_receipts",
        "counts_as_current_model_event": False,
        "original_flags": original["original_flags"],
        "original_findings": original["original_findings"],
        "typed_counter_paths": original["typed_counter_paths"],
        "original_validation_receipt": original["original_validation_receipt"],
        "private_replay_receipt": replay,
    }
    scripted_payload = {
        "evidence_scope": "simulated_transport_events",
        "counts_as_current_model_event": False,
        "scripted_events": callback_panel["scripted_events"],
        "contradiction_control": contradiction,
        "contradiction_control_receipt": contradiction_receipt,
    }
    sidecar_manifest, sidecar_references = write_evidence_sidecars(
        root=root,
        sidecar_dir=root / RAW_DIR / "sidecars",
        historical_payload=historical_payload,
        scripted_payload=scripted_payload,
    )

    provisional_receipts = [
        *validation_receipts,
        *[
            {
                "name": name,
                "command_argv": ["pending", name],
                "environment": {},
                "exit_code": 0,
                "expected_exit_code": 0,
                "duration_s": 0.0,
                "log_sha256": "sha256:" + "0" * 64,
                "passed": True,
                "timed_out": False,
                "provisional_candidate_only": True,
            }
            for name in REQUIRED_TERMINAL
        ],
    ]
    candidate = private_root / "candidate.json"
    artifact = build_terminal_artifact(
        root=root,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        started_monotonic_ns=monotonic_started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=phases,
        preconditions_checked=preconditions,
        source_artifact_hashes=source_hashes,
        callback_panel=callback_panel,
        scripted_evidence_manifest=sidecar_manifest,
        sidecar_references=sidecar_references,
        validation_receipts=provisional_receipts,
        original_summary=_original_summary(original),
    )
    atomic_json(candidate, artifact)

    phase_started = time.monotonic()
    progress(run_started, "terminal_readers", "before_subprocesses", total_units=3)
    terminal_receipts = run_specs(
        root,
        terminal_command_specs(root, candidate),
        private_root / "logs" / "terminal_preflight",
    )
    validation_receipts.extend(terminal_receipts)
    _phase(phases, "terminal_readers", phase_started, run_started, len(terminal_receipts))
    progress(
        run_started,
        "terminal_readers",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
    )

    artifact = build_terminal_artifact(
        root=root,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        started_monotonic_ns=monotonic_started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=phases,
        preconditions_checked=preconditions,
        source_artifact_hashes=source_hashes,
        callback_panel=callback_panel,
        scripted_evidence_manifest=sidecar_manifest,
        sidecar_references=sidecar_references,
        validation_receipts=validation_receipts,
        original_summary=_original_summary(original),
    )
    errors = validate_artifact(artifact, root=root)
    if errors:
        raise RuntimeError(f"terminal artifact validation failed: {errors}")
    atomic_json(candidate, artifact)

    exact_log_dir = root / RAW_DIR / "exact_candidate_validation"
    progress(run_started, "exact_candidate", "before_subprocesses", total_units=3)
    exact_receipts = run_specs(root, terminal_command_specs(root, candidate), exact_log_dir)
    progress(
        run_started,
        "exact_candidate",
        "after_subprocesses",
        completed_units=len(exact_receipts),
    )
    if not validation_receipts_pass(exact_receipts, REQUIRED_TERMINAL):
        raise RuntimeError("exact terminal candidate readers failed")

    progress(run_started, "write", "before_atomic_terminal_write", completed_units=0)
    atomic_json(root / RESULT_PATH, artifact)
    progress(
        run_started,
        "write",
        "after_atomic_terminal_write",
        completed_units=1,
        path=RESULT_PATH.as_posix(),
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed execution date for the thin entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=(RUN_DATE,))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Experiment 7424 from the repository-root entrypoint."""

    args = parse_args(argv)
    run_experiment(REPO_ROOT, args.date)
    return 0
