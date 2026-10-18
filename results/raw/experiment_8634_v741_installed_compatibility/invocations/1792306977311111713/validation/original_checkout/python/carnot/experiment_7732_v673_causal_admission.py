"""Bounded fixture admission and terminal custody for REQ-REPORT-7732."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.reporting.acquisition_qualification import (
    EPSILON,
    fit_static_closure,
    fixture_groups,
    static_probability,
)
from carnot.reporting.constraint_bank_protocol import Bank, grammar
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7732_v673_causal_admission")
OUTPUT = Path("results/experiment_7732_v673_causal_admission.json")
MODULE = "python/carnot/experiment_7732_v673_causal_admission.py"
TEST = "tests/python/test_experiment_7732_v673_causal_admission.py"
CLI = "scripts/experiments/experiment_7732_v673_causal_admission.py"
REUSED = [
    "python/carnot/reporting/acquisition_qualification.py",
    "python/carnot/reporting/constraint_bank_protocol.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
]
REUSED_TESTS = [
    "tests/python/test_experiment_7719_v672_acquisition_qualification.py",
    "tests/python/test_experiment_7705_v671_constraint_bank_protocol.py",
    "tests/python/test_experiment_7303_v642_validation_scope.py",
]
INPUTS = {
    "results/experiment_7705_v671_constraint_bank_protocol.json": (
        "carnot.exp7705.v671.constraint_bank.v1",
        7705,
    ),
    "results/experiment_7719_v672_acquisition_qualification.json": (
        "carnot.exp7719.v672.acquisition_qualification.v1",
        7719,
    ),
    "results/experiment_7728_v673_set_energy_protocol.json": (None, "exp7728-set-energy-protocol"),
}
PRINCIPLE = "Measured evidence bounds the claim and downstream use."


def progress(started: float, phase: str, event: str, units: int = 0) -> None:
    """Print a flushed phase or work boundary with elapsed time."""
    print(
        f"[exp7732] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def check(name: str, upstream: str, path: str, field: str, expected: Any, observed: Any) -> dict:
    """Preserve literal operands for a precondition or required check."""
    return {
        "check": name,
        "upstream_id": upstream,
        "artifact_path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight(root: Path) -> tuple[list[dict], dict]:
    """Check external bytes, schemas, CPU and space before owned measurement."""
    root = root.resolve()
    checks = [
        check("repo_root", "host", str(root), "is_dir", True, root.is_dir()),
        check("cpu", "host", "/proc/self", "cpu_available", True, (os.cpu_count() or 0) > 0),
        check(
            "disk",
            "host",
            str(root),
            "free_at_least_10m",
            True,
            os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= 10_000_000,
        ),
    ]
    hashes: dict[str, Any] = {
        "eligible_producers": {},
        "flagged_historical_inputs": {},
        "pre_gate_receipts": {},
        "absent_sources": [],
    }
    for relative, (schema, experiment_id) in INPUTS.items():
        path = root / relative
        exists = path.is_file()
        checks.append(check("input_exists", str(experiment_id), relative, "exists", True, exists))
        if not exists:
            hashes["absent_sources"].append(relative)
            continue
        digest = sha256_file(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {}
        if schema is not None:
            checks.append(
                check(
                    "input_schema",
                    str(experiment_id),
                    relative,
                    "schema",
                    schema,
                    value.get("schema"),
                )
            )
        checks.append(
            check(
                "input_id",
                str(experiment_id),
                relative,
                "experiment_id",
                experiment_id,
                value.get("experiment_id"),
            )
        )
        category = (
            "flagged_historical_inputs"
            if value.get("verdict_class") == "disqualified"
            else "eligible_producers"
        )
        hashes[category][relative] = digest
    return checks, hashes


def admission_decision(frozen: list[dict], labels: list[int]) -> dict:
    """Apply the preregistered eight-family loss and false-accept guards."""
    if len(frozen) != 8 or len(labels) != 8 or len({row["unit_id"] for row in frozen}) != 8:
        raise ValueError("eight_independent_admission_cases_required")
    if any(label not in (0, 1) or isinstance(label, bool) for label in labels):
        raise ValueError("binary_admission_labels_required")
    before = sum((row["base"] - label) ** 2 for row, label in zip(frozen, labels)) / 8
    after = sum((row["candidate"] - label) ** 2 for row, label in zip(frozen, labels)) / 8
    false_before = sum(row["base"] >= 0.5 and label == 0 for row, label in zip(frozen, labels))
    false_after = sum(row["candidate"] >= 0.5 and label == 0 for row, label in zip(frozen, labels))
    return {
        "accepted": before - after >= 0.01 and false_after <= false_before,
        "mean_brier_reduction": before - after,
        "base_false_accepts": false_before,
        "candidate_false_accepts": false_after,
        "independent_families": 8,
    }


class BoundedBank(Bank):
    """Use the durable production bank with this task's twelve-handle cap."""

    def predict(
        self,
        event_id: str,
        tick: int,
        features: dict[str, float],
        base_probability: float,
        exact_status: str,
        partition: str,
        source_id: str,
    ) -> dict[str, Any]:
        pending = sum(key not in self.state["feedback"] for key in self.state["predictions"])
        if pending >= 12:
            before = self._before()
            self.budget["overflow_rejections"] += 1
            self._save("overflow", {"event_id": event_id, "tick": tick}, before)
            raise ValueError("pending_overflow")
        return super().predict(
            event_id, tick, features, base_probability, exact_status, partition, source_id
        )


def run_fixture(raw: Path) -> dict:
    """Measure three matched banks and every durable lifecycle outcome."""
    raw.mkdir(parents=True, exist_ok=True)
    groups = fixture_groups()
    closure = fit_static_closure(groups["development"])
    arms = ("growth", "frozen", "complete_static")
    rows: list[dict] = []
    events: list[dict] = []
    state_paths: dict[str, str] = {}
    decisions: list[dict] = []
    high_water = 0
    restart_parity = True
    exactly_once = True
    for arm in arms:
        path = raw / f"bank_{arm}.json"
        path.unlink(missing_ok=True)
        scheduler = "priority" if arm == "growth" else "read_only" if arm == "frozen" else "static"
        bank = BoundedBank(path, grammar(), scheduler, 0.1, 2)
        tick = 0
        arm_events: list[dict] = []

        def record(kind: str, case_id: str, at: int) -> None:
            arm_events.append(
                {
                    "arm": arm,
                    "kind": kind,
                    "unit_id": case_id,
                    "tick": at,
                    "wall_time_s": time.time(),
                    "state_hash": bank.state_hash,
                }
            )

        def predict_batch(cases: list[dict], partition: str) -> None:
            nonlocal tick, high_water
            for case in cases:
                base = (
                    static_probability(case, closure)
                    if arm == "complete_static"
                    else case["base_probability"]
                )
                prediction = bank.predict(
                    case["unit_id"],
                    tick,
                    case["features"],
                    base,
                    case["exact_status"],
                    partition,
                    case["unit_id"],
                )
                source = bytes.fromhex(case["source_bytes_hex"])
                rows.append(
                    {
                        "unit_id": case["unit_id"],
                        "arm": arm,
                        "role": case["role"],
                        "probability": prediction["probability"],
                        "base_probability": base,
                        "label": None,
                        "brier": None,
                        "exact_status": case["exact_status"],
                        "advisory_status": case["exact_status"],
                        "source_sha256": hashlib.sha256(source).hexdigest(),
                        "input_hash": canonical_hash(case),
                        "denominators": {"independent_family": 1, "paired_arm": 1},
                        "censored": False,
                        "exclusions": [],
                        "prediction_tick": tick,
                        "feedback_tick": None,
                    }
                )
                record("prediction", case["unit_id"], tick)
                tick += 1
                high_water = max(
                    high_water,
                    sum(key not in bank.state["feedback"] for key in bank.state["predictions"]),
                )

        def release_batch(cases: list[dict]) -> None:
            nonlocal tick
            release_tick = tick + 8
            for case in cases:
                bank.release(case["unit_id"], case["label"], release_tick)
                row = next(
                    row
                    for row in reversed(rows)
                    if row["arm"] == arm and row["unit_id"] == case["unit_id"]
                )
                row["label"] = case["label"]
                row["brier"] = (row["probability"] - case["label"]) ** 2
                row["feedback_tick"] = release_tick
                record("feedback_arrival", case["unit_id"], release_tick)
                record("durable_ack", case["unit_id"], release_tick)
            tick = release_tick + 1

        for block in range(4):
            development = groups["development"][8 * block : 8 * (block + 1)]
            predict_batch(development, "update")
            release_batch(development)
            proposal = bank.propose()
            if proposal is not None:
                record("proposal", proposal["counterexample_id"], proposal["freeze_tick"])
            admission = groups["admission"][8 * block : 8 * (block + 1)]
            frozen: list[dict] = []
            for case in admission:
                predict_batch([case], "admission")
                base = rows[-1]["probability"]
                active = bool(
                    proposal and all(case["features"][name] > 0 for name in proposal["pair"])
                )
                candidate = min(
                    1 - EPSILON,
                    max(EPSILON, base + (proposal["weight"] if active and proposal else 0)),
                )
                frozen.append(
                    {
                        "unit_id": case["unit_id"],
                        "base": base,
                        "candidate": candidate,
                        "freeze_tick": proposal["freeze_tick"] if proposal else None,
                    }
                )
            release_batch(admission)
            if proposal is not None:
                decision = admission_decision(frozen, [case["label"] for case in admission])
                prior = bank.state_hash
                bank.simulate_crash_before_commit()
                before_ack = BoundedBank(path, grammar(), scheduler, 0.1, 2)
                restart_parity &= before_ack.state_hash == prior
                record("one_use_admission", admission[0]["unit_id"], tick)
                bank.admit(decision["accepted"], admission[0]["unit_id"])
                record(
                    "commit" if decision["accepted"] else "rollback", admission[0]["unit_id"], tick
                )
                record("durable_ack", admission[0]["unit_id"], tick)
                after_ack = BoundedBank(path, grammar(), scheduler, 0.1, 2)
                restart_parity &= after_ack.state_hash == bank.state_hash
                exactly_once &= after_ack.budget["admission_credits_spent"] == len(decisions) + 1
                decisions.append(
                    {
                        "arm": arm,
                        "proposal": proposal,
                        "frozen": frozen,
                        "labels": [case["label"] for case in admission],
                        **decision,
                    }
                )
            reopened = BoundedBank(path, grammar(), scheduler, 0.1, 2)
            restart_parity &= reopened.state_hash == bank.state_hash
            atomic_json(
                raw / "checkpoint.json",
                {
                    "arm": arm,
                    "block": block,
                    "completed_rows": len(rows),
                    "state_hash": bank.state_hash,
                },
            )
            print(
                f"[exp7732] measurement checkpoint arm={arm} block={block} completed_units={len(rows)}",
                flush=True,
            )
        for start in range(0, 32, 8):
            batch = groups["retention"][start : start + 8]
            predict_batch(batch, "retention")
            release_batch(batch)
        reopened = BoundedBank(path, grammar(), scheduler, 0.1, 2)
        restart_parity &= reopened.state_hash == bank.state_hash
        restart_parity &= reopened.replay_ledger()["state_hash"] == bank.state_hash
        state_paths[arm] = str(path)
        events.extend(arm_events)
        print(
            f"[exp7732] measurement arm_complete arm={arm} completed_units={len(rows)}", flush=True
        )
    diagnostic_path = raw / "bank_diagnostics.json"
    diagnostic_path.unlink(missing_ok=True)
    diagnostic = BoundedBank(diagnostic_path, grammar(), "priority", 0.1, 2)
    case = groups["development"][0]
    for index in range(12):
        diagnostic.predict(
            f"diagnostic-{index:02d}",
            index,
            case["features"],
            0.1,
            "unknown",
            "update",
            f"diagnostic-{index:02d}",
        )
    overflow = False
    try:
        diagnostic.predict(
            "diagnostic-overflow",
            12,
            case["features"],
            0.1,
            "unknown",
            "update",
            "diagnostic-overflow",
        )
    except ValueError as error:
        overflow = str(error) == "pending_overflow"
    diagnostic.release("diagnostic-00", 1, 20)
    duplicate = False
    try:
        diagnostic.release("diagnostic-00", 1, 21)
    except ValueError as error:
        duplicate = str(error) == "duplicate_feedback"
    reopened_diagnostic = BoundedBank(diagnostic_path, grammar(), "priority", 0.1, 2)
    restart_parity &= reopened_diagnostic.state_hash == diagnostic.state_hash
    growth = BoundedBank(Path(state_paths["growth"]), grammar(), "priority", 0.1, 2)
    later = next(
        row
        for row in rows
        if row["arm"] == "growth" and row["unit_id"] == groups["retention"][0]["unit_id"]
    )
    lifecycle = {
        "beneficial_commit": any(
            row["accepted"] and row["proposal"]["pair"] == grammar()["pairs"][0]
            for row in decisions
        )
        and later["probability"] > later["base_probability"],
        "harmful_rejection": any(not row["accepted"] for row in decisions)
        and len(growth.state["templates"]) == 1,
        "duplicate_feedback": duplicate,
        "pending_overflow": overflow and diagnostic.budget["overflow_rejections"] == 1,
        "interrupted_write_replay": restart_parity,
    }
    static_complete = {
        "dictionary": closure["features"],
        "weights": closure["coefficients"],
        "fit_role": "development_only",
        "equality_check": closure == fit_static_closure(groups["development"]),
        "nonempty": closure["different_from_empty_bank"],
    }
    atomic_json(raw / "rows.json", rows)
    atomic_json(raw / "event_rows.json", events)
    return {
        "rows": rows,
        "event_rows": events,
        "arms": list(arms),
        "decisions": decisions,
        "lifecycle": lifecycle,
        "restart_exact_parity": restart_parity,
        "exactly_once": exactly_once,
        "pending_high_water": high_water,
        "proposal_count": growth.budget["proposal_credits_spent"],
        "static_closure_complete": static_complete,
        "bank_state_paths": state_paths,
        "raw_rows_sha256": sha256_file(raw / "rows.json"),
        "raw_rows_path": str(raw / "rows.json"),
        "raw_events_sha256": sha256_file(raw / "event_rows.json"),
        "fixture_checksum": canonical_hash(groups),
        "budget_accounting": {"growth": dict(growth.budget), "diagnostic": dict(diagnostic.budget)},
    }


def cold_reduce(path: Path) -> dict:
    """Recompute family losses and ledger parity from exact saved raw bytes."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact.get("rows", [])
    arms = artifact.get("arms", [])
    groups = fixture_groups()
    originals = {case["unit_id"]: case for group in groups.values() for case in group}
    keys = {(row["arm"], row["unit_id"]) for row in rows}
    complete = len(arms) == 3 and len(rows) == 96 * 3 and len(keys) == len(rows)
    source_valid = True
    arithmetic = True
    bank_valid = True
    for row in rows:
        case = originals.get(row.get("unit_id"))
        if case is None:
            source_valid = False
            continue
        source_valid &= (
            row.get("input_hash") == canonical_hash(case)
            and row.get("source_sha256")
            == hashlib.sha256(bytes.fromhex(case["source_bytes_hex"])).hexdigest()
        )
        arithmetic &= (
            row.get("label") == case["label"]
            and row.get("role") == case["role"]
            and row.get("censored") is False
            and row.get("exclusions") == []
            and row.get("exact_status") == row.get("advisory_status")
            and abs(row.get("brier", -1) - (row.get("probability", -2) - case["label"]) ** 2)
            < 1e-12
        )
    for arm, state_path in artifact.get("bank_state_paths", {}).items():
        scheduler = "priority" if arm == "growth" else "read_only" if arm == "frozen" else "static"
        try:
            bank = BoundedBank(Path(state_path), grammar(), scheduler, 0.1, 2)
            bank_valid &= bank.replay_ledger()["state_hash"] == bank.state_hash
            for row in (item for item in rows if item["arm"] == arm):
                prediction = bank.state["predictions"].get(row["unit_id"])
                feedback = bank.state["feedback"].get(row["unit_id"])
                bank_valid &= bool(
                    prediction
                    and feedback
                    and prediction["probability"] == row["probability"]
                    and prediction["tick"] == row["prediction_tick"]
                    and feedback["release_tick"] == row["feedback_tick"]
                    and feedback["label"] == row["label"]
                )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            bank_valid = False
    bank_valid &= len(artifact.get("bank_state_paths", {})) == 3
    closure = fit_static_closure(groups["development"])
    static_valid = (
        artifact.get("static_closure_complete", {}).get("weights") == closure["coefficients"]
    )
    decisions_valid = (
        all(
            admission_decision(row["frozen"], row["labels"])["accepted"] == row["accepted"]
            for row in artifact.get("decisions", [])
        )
        and len(artifact.get("decisions", [])) == 2
    )
    raw_path = artifact.get("raw_rows_path")
    raw_valid = bool(
        raw_path
        and Path(raw_path).is_file()
        and sha256_file(Path(raw_path)) == artifact.get("raw_rows_sha256")
        and json.loads(Path(raw_path).read_text()) == rows
    )
    valid = all(
        (complete, source_valid, arithmetic, bank_valid, static_valid, decisions_valid, raw_valid)
    )
    return {
        "valid": valid,
        "complete": complete,
        "source_valid": source_valid,
        "arithmetic": arithmetic,
        "bank_valid": bank_valid,
        "static_valid": static_valid,
        "decisions_valid": decisions_valid,
        "raw_valid": raw_valid,
        "rows": len(rows),
        "independent_families": len({row["unit_id"] for row in rows}),
    }


def phase_span(
    started: float, phase: str, begin: float, units: int, checkpoint: Path | None = None
) -> dict:
    """Record one nonoverlapping monotonic phase and its real checkpoint."""
    end = time.monotonic() - started
    return {
        "phase": phase,
        "start_s": begin,
        "end_s": end,
        "duration_s": end - begin,
        "run_date": "20260927",
        "heartbeat_times": [time.time()],
        "completed_units": units,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint)
        if checkpoint and checkpoint.is_file()
        else None,
    }


def build_artifact(
    root: Path,
    date: str,
    started: float,
    checks: list[dict],
    hashes: dict,
    measured: dict,
    spans: list[dict],
    scope: dict,
) -> dict:
    """Bind fixture mechanics to scoped claims without inferring empirical benefit."""
    rows = measured.get("rows", [])
    lifecycle = measured.get("lifecycle", {})
    complete = len(rows) == 288 and len({row["unit_id"] for row in rows}) == 96
    mechanics = (
        complete
        and all(lifecycle.values())
        and measured.get("restart_exact_parity") is True
        and measured.get("exactly_once") is True
        and measured.get("pending_high_water", 99) <= 12
        and measured.get("proposal_count", 99) <= 8
        and measured.get("static_closure_complete", {}).get("equality_check") is True
    )
    blocked = any(not row["passed"] for row in checks)
    verdict = (
        "complete_blocked_external_input"
        if blocked
        else "complete_circular_positive_causal_admission"
        if mechanics
        else "complete_null_causal_admission_unqualified"
    )
    verdict_class = "blocked" if blocked else "circular_positive" if mechanics else "null"
    gates = {
        "validity": bool(not blocked and mechanics),
        "readiness": None,
        "brier_score": None,
        "decision_cost": None,
        "coverage": complete if rows else None,
        "retention": measured.get("restart_exact_parity") if rows else None,
        "efficiency": None,
    }
    historical = (
        root
        / "results/raw/experiment_7719_v672_acquisition_qualification/validation/full/00_full_python_suite.log"
    )
    debt = (
        [
            {
                "experiment_id": 7719,
                "scope": "repository_health",
                "log_path": str(historical.relative_to(root)),
                "log_sha256": sha256_file(historical),
                "original_exit_code": 2,
                "resolved": False,
            }
        ]
        if historical.is_file()
        else []
    )
    artifact = {
        "schema": "carnot.exp7732.v673.causal_admission.v1",
        "experiment_id": 7732,
        "milestone": "2026.09.673",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": [row for row in checks if not row["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "arms": measured.get("arms", []),
        "event_rows": measured.get("event_rows", []),
        "decisions": measured.get("decisions", []),
        "lifecycle": lifecycle,
        "exactly_once": measured.get("exactly_once"),
        "pending_high_water": measured.get("pending_high_water"),
        "proposal_count": measured.get("proposal_count"),
        "sample_size_budget": {
            "intended": {"development": 32, "admission": 32, "retained": 32},
            "observed": len({row["unit_id"] for row in rows}),
            "eligible": len({row["unit_id"] for row in rows}),
            "excluded": 0,
            "censored": 0,
            "effective_independent_families": len({row["unit_id"] for row in rows}),
            "arms_do_not_increase_n": True,
        },
        "claim_scope": {
            "reused_RAGTruth": "development_only",
            "exact_fixtures": "fixture_only",
            "ARC_public_games": "adapter_withheld_public_unmeasured",
            "fresh_generalization_eligible": False,
        },
        "inference_substrate": "deterministic_cpu_pair_feature_bank",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {
            "model_loads": 0,
            "forwards": 0,
            "generations": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "failures": 0,
            "cancellations": 0,
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "pid": os.getpid(),
            "effective_coding_backend": os.getenv("CODEX_MODEL", "gpt-6 session"),
        },
        "phase_spans": spans,
        "duration_s": time.monotonic() - started,
        "random_seed": {
            "fixture": 7719,
            "fixture_purpose": "reused deterministic family construction",
            "admission": 7732,
            "admission_purpose": "fixed block order; no random draw",
        },
        "reproducibility_checksum": canonical_hash(
            {
                "input_hashes": hashes,
                "fixture": measured.get("fixture_checksum"),
                "configuration": {"admission_n": 8, "brier_delta": 0.01, "max_pending": 12},
                "reducer_code": sha256_file(ROOT / MODULE),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks
        + [
            {
                "resource": "coding_backend",
                "effective": os.getenv("CODEX_MODEL", "gpt-6 session"),
                "model_execution": "none",
            }
        ],
        "validation_receipts": {
            "frozen_affected_scope": scope,
            "affected": [],
            "task_e2e": [],
            "terminal_readers": [],
            "repository_health": {
                "status": "degraded_open" if debt else "healthy",
                "historical_failures": debt,
                "affects_required_checks": False,
            },
        },
        "verifier_is_oracle": True,
        "acquisition_protocol_ready_score": 0,
        "static_closure_complete": measured.get("static_closure_complete", {}),
        "bank_state_paths": measured.get("bank_state_paths", {}),
        "restart_exact_parity": measured.get("restart_exact_parity"),
        "budget_accounting": measured.get("budget_accounting", {}),
        "raw_rows_path": measured.get("raw_rows_path"),
        "raw_rows_sha256": measured.get("raw_rows_sha256"),
        "raw_events_sha256": measured.get("raw_events_sha256"),
        "fixture_checksum": measured.get("fixture_checksum"),
        "same_verdict_retirements": [],
    }
    artifact["field_principles"] = {
        **{key: PRINCIPLE for key in artifact},
        "field_principles": PRINCIPLE,
        **{f"acceptance_gate_{key}": PRINCIPLE for key in gates},
    }
    return artifact


def run_experiment(root: Path, date: str, output: Path, *, validate: bool = True) -> dict:
    """Freeze scope, measure, validate exact candidate and atomically publish."""
    root = root.resolve()
    output = output.resolve()
    started = time.monotonic()
    spans: list[dict] = []
    progress(started, "preflight", "start")
    begin = time.monotonic() - started
    checks, hashes = preflight(root)
    spans.append(phase_span(started, "preflight", begin, len(checks)))
    progress(started, "preflight", "complete", len(checks))
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    affected_files = [
        MODULE,
        CLI,
        TEST,
        *REUSED,
        *REUSED_TESTS,
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/continuous-learning/spec.md",
    ]
    scope = {
        "tests": [TEST, *REUSED_TESTS],
        "changed_modules": [MODULE],
        "coverage_tests": [TEST],
        "reused_modules": REUSED,
        "static_paths": [CLI],
        "specs": ["REQ-REPORT-7732", "REQ-CL-7732-CAUSAL-ADMISSION"],
        "frozen_at_epoch_s": time.time(),
        "input_hashes": hashes,
        "file_sha256": {
            name: sha256_file(ROOT / name) for name in affected_files if (ROOT / name).is_file()
        },
    }
    atomic_json(raw / "frozen_affected_scope.json", scope)
    measured: dict = {}
    if all(row["passed"] for row in checks):
        progress(started, "measurement", "start")
        begin = time.monotonic() - started
        measured = run_fixture(raw)
        atomic_json(raw / "measurement.json", measured)
        spans.append(
            phase_span(
                started, "measurement", begin, len(measured["rows"]), raw / "checkpoint.json"
            )
        )
        progress(started, "measurement", "complete", len(measured["rows"]))
    else:
        atomic_json(
            raw / "checkpoint.json",
            {"completed_units": 0, "blocked_checks": [row for row in checks if not row["passed"]]},
        )
    artifact = build_artifact(root, date, started, checks, hashes, measured, spans, scope)
    if not validate:
        atomic_json(output, artifact)
        return artifact
    with tempfile.TemporaryDirectory(prefix="exp7732-validation-") as private_name:
        private = Path(private_name)
        (private / "basetemp").mkdir()
        commands = build_scoped_commands(
            root,
            scope["tests"],
            [MODULE],
            static_paths=[CLI],
            basetemp=private / "basetemp",
            coverage_file=private / ".coverage",
        )
        commands = [
            CommandSpec(
                item.name,
                tuple(arg for arg in item.argv if arg not in REUSED_TESTS),
                item.scope,
                item.timeout_s,
            )
            if item.name == "changed_module_coverage"
            else item
            for item in commands
        ]
        progress(started, "affected_validation", "before_subprocess")
        begin = time.monotonic() - started
        affected = run_commands(
            root,
            commands,
            log_dir=raw / "validation/affected",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["affected"] = affected
        spans.append(phase_span(started, "affected_validation", begin, len(affected)))
        progress(started, "affected_validation", "after_subprocess", len(affected))
        python = str(root / ".venv/bin/python")
        full = [
            CommandSpec(
                "full_python_suite",
                (
                    str(root / ".venv/bin/pytest"),
                    "tests/python",
                    "-q",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={private / 'full'}",
                ),
                "repository_health",
                timeout_s=1800,
            )
        ]
        progress(started, "repository_health", "before_subprocess")
        begin = time.monotonic() - started
        previous = json.loads(output.read_text()) if output.is_file() else {}
        prior_suite = (
            previous.get("validation_receipts", {})
            .get("repository_health", {})
            .get("current_full_suite", [])
        )
        prior_log = root / prior_suite[0]["log_path"] if len(prior_suite) == 1 else None
        reusable = bool(
            previous.get("experiment_id") == 7732
            and previous.get("run_date") == date
            and prior_log
            and prior_log.is_file()
            and sha256_file(prior_log) == prior_suite[0].get("log_sha256")
        )
        if reusable:
            suite = prior_suite
            progress(started, "repository_health", "reused_hash_verified_log", len(suite))
        else:
            suite = run_commands(
                root,
                full,
                log_dir=raw / "validation/full",
                extra_env={"JAX_PLATFORMS": "cpu"},
                heartbeat_s=30,
            )
        artifact["validation_receipts"]["repository_health"]["current_full_suite"] = suite
        if not all(item["passed"] for item in suite):
            artifact["validation_receipts"]["repository_health"]["status"] = "degraded_open"
        spans.append(phase_span(started, "repository_health", begin, len(suite)))
        progress(started, "repository_health", "after_subprocess", len(suite))
        e2e_commands = [
            CommandSpec(
                "task_e2e",
                (
                    str(root / ".venv/bin/pytest"),
                    TEST,
                    "-q",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={private / 'e2e'}",
                ),
                "task_e2e",
            )
        ]
        progress(started, "task_e2e", "before_subprocess")
        begin = time.monotonic() - started
        e2e = run_commands(
            root,
            e2e_commands,
            log_dir=raw / "validation/e2e",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["task_e2e"] = e2e
        spans.append(phase_span(started, "task_e2e", begin, len(e2e)))
        progress(started, "task_e2e", "after_subprocess", len(e2e))
        required_passed = bool(
            affected and e2e and all(item["passed"] for item in [*affected, *e2e])
        )
        artifact["acquisition_protocol_ready_score"] = int(
            required_passed and artifact["acceptance_gate_results"]["validity"]
        )
        artifact["phase_spans"] = spans
        artifact["duration_s"] = time.monotonic() - started
        candidate = raw / "terminal_candidate.json"
        atomic_json(candidate, artifact)
        readers = [
            CommandSpec(
                "cold_reduce",
                (
                    python,
                    "-u",
                    "-m",
                    "carnot.experiment_7732_v673_causal_admission",
                    "--cold-reduce",
                    str(candidate),
                ),
                "exact_candidate",
            ),
            CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "exact_candidate",
            ),
            CommandSpec(
                "verdict_row_consistency",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "exact_candidate",
            ),
        ]
        progress(started, "terminal_readers", "before_subprocess")
        begin = time.monotonic() - started
        terminal = run_commands(
            root,
            readers,
            log_dir=raw / "validation/terminal",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["terminal_readers"] = terminal
        artifact["validation_receipts"]["exact_terminal_candidate_sha256"] = sha256_file(candidate)
        spans.append(phase_span(started, "terminal_readers", begin, len(terminal)))
        progress(started, "terminal_readers", "after_subprocess", len(terminal))
    failures = [row for row in [*affected, *e2e, *terminal] if not row["passed"]]
    if failures:
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["verdict_class"] = "disqualified"
        artifact["acquisition_protocol_ready_score"] = 0
        artifact["flagged_adversarial"] = any(
            row["name"] == "adversarial_verify" for row in failures
        )
        artifact["gate_check_summary"].extend(
            check(
                "required_validation", "current", row["log_path"], "exit_code", 0, row["exit_code"]
            )
            for row in failures
        )
        artifact["acceptance_gate_results"]["validity"] = False
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    progress(started, "publish", "before_atomic_write")
    atomic_json(output, artifact)
    progress(started, "publish", "complete")
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run the CPU experiment or verify an exact saved candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        result = cold_reduce(args.cold_reduce)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["valid"] else 1
    run_experiment(ROOT, args.date, ROOT / args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI process boundary.
    raise SystemExit(main())
