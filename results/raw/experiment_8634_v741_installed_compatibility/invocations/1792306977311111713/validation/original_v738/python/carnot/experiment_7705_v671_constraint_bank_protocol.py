"""Run the bounded CPU constraint-bank protocol (REQ-REPORT-7705)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.reporting.constraint_bank_protocol import Bank, advisory_status, grammar, replay_ledger
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.reporting.typed_decision_energy import action, decision_cost


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7705_v671_constraint_bank_protocol")
OUTPUT = Path("results/experiment_7705_v671_constraint_bank_protocol.json")
HEAD = Path("results/experiment_7703_v671_typed_decision_energy.json")
COHORT = Path("results/raw/experiment_7701_v671_sealed_cohort/protocol.json")
FROZEN = Path("results/raw/experiment_7703_v671_typed_decision_energy/online_protocol.json")
MODULE = "python/carnot/reporting/constraint_bank_protocol.py"
ORCHESTRATION = "python/carnot/experiment_7705_v671_constraint_bank_protocol.py"
WRAPPER = "scripts/experiments/experiment_7705_v671_constraint_bank_protocol.py"
TEST = "tests/python/test_experiment_7705_v671_constraint_bank_protocol.py"
MODEL_SPECS: list[str] = []
ARMS = (
    "priority",
    *(f"fixed_{period}" for period in (2, 4, 6, 8, 10, 12)),
    "read_only",
    "weight_only",
    "static",
)
SCOPE = {
    "tests": [TEST],
    "changed_modules": [MODULE, ORCHESTRATION],
    "static_paths": [WRAPPER],
    "specs": ["REQ-REPORT-7705", "REQ-CL-7705-BOUNDED-BANK"],
    "e2e": ["task_restart_once", "task_cold_ledger"],
}
PRINCIPLES = {
    "honest_verdict": "A terminal disposition prevents retries of unchanged external blocks.",
    "verdict_class": "A closed enum carries claim eligibility into downstream readers.",
    "flagged_adversarial": "Disqualified evidence cannot open gates.",
    "gate_check_summary": "Exact upstream operands distinguish missing evidence from false gates.",
    "acceptance_gate_results": "Measured operands limit claim scope.",
    "rows": "Original unit observations allow recomputation.",
    "sample_size_budget": "Seeds and views never enlarge independent n.",
    "inference_substrate": "Actual CPU work must match the declared path.",
    "inference_substrate_class": "Duration floors describe actual model work.",
    "MODEL_SPECS": "Only invoked models belong in current model specs.",
    "model_invoked": "Invocation counts separate current and historical model use.",
    "execution_venue": "Host and PID identify current work.",
    "phase_spans": "Monotonic spans and heartbeats bound current work.",
    "random_seed": "Declared deterministic inputs permit replay.",
    "reproducibility_checksum": "Input and reducer bytes bind independent replay.",
    "source_artifact_hashes": "Producer, pre-gate and missing custody stay separate.",
    "preconditions_checked": "Input and resource checks precede labels.",
    "validation_receipts": "Real exits and log hashes bind validation.",
    "verifier_is_oracle": "Fixture truth cannot establish oracle-distinct benefit.",
    "constraint_bank_ready_score": "Only valid durable mechanics may open an administrative gate.",
    "continuous_self_learning_task": "Counters and bank additions are separate learning tiers.",
    "bank_schema_path": "A saved schema lets readers inspect limits.",
    "lifecycle_rows": "Events and state hashes expose causal order.",
    "hardware_path": "CPU measurements cannot claim unmeasured hardware speedup.",
    "budget_accounting": "Hard credits include rejected acquisitions.",
}
GATE_PRINCIPLES = {
    "validity": "Invalid evidence must not propagate.",
    "readiness": "Durable bounded mechanics gate future evaluation, not benefit.",
    "coverage": "Quality thresholds prevent effects from being inferred from plumbing.",
    "freshness": "Prior exposure does not become fresh evidence.",
    "probability": "An empty bank or fixture truth cannot prove predictive gain.",
    "utility": "A policy benefit needs untouched outcomes and a matched control.",
    "retention": "Retention bounds prevent apparent gains from forgetting.",
    "efficiency": "Resource gains need a measured comparable workload.",
}


def progress(started: float, phase: str, event: str, units: int = 0) -> None:
    """Flush each boundary so an active CPU task never looks idle."""
    print(
        f"[exp7705] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def _check(
    name: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
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


def preflight(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hash current producers and check their literal gates before labels."""
    checks = [
        _check(
            "cpu_available",
            "host",
            "/proc/self",
            "cpu_count_positive",
            True,
            (os.cpu_count() or 0) > 0,
        ),
        _check(
            "disk_available",
            "host",
            str(root),
            "free_bytes_at_least_10m",
            True,
            os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= 10_000_000,
        ),
    ]
    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_inputs": []}
    inputs = [
        (
            Path("results/experiment_7700_v671_record_span_protocol.json"),
            "exp7700-record-span-protocol",
        ),
        (Path("results/experiment_7701_v671_sealed_cohort.json"), "exp7701-sealed-cohort"),
        (HEAD, "exp7703-typed-decision-energy"),
        (COHORT, "exp7701-sealed-cohort"),
        (FROZEN, "exp7703-typed-decision-energy"),
    ]
    for relative, upstream in inputs:
        path = root / relative
        checks.append(
            _check("input_exists", upstream, str(relative), "exists", True, path.is_file())
        )
        if path.is_file():
            hashes["producers"][str(relative)] = sha256_file(path)
        else:
            hashes["missing_inputs"].append(str(relative))
    if hashes["missing_inputs"]:
        return checks, hashes
    for relative, upstream, field in (
        (inputs[0][0], inputs[0][1], "record_protocol_ready_score"),
        (inputs[1][0], inputs[1][1], "cohort_ready_score"),
        (HEAD, inputs[2][1], "decision_energy_ready_score"),
    ):
        value = json.loads((root / relative).read_text())
        checks.extend(
            (
                _check("upstream_gate", upstream, str(relative), field, 1, value.get(field)),
                _check(
                    "upstream_flag",
                    upstream,
                    str(relative),
                    "flagged_adversarial",
                    False,
                    value.get("flagged_adversarial"),
                ),
            )
        )
    head = json.loads((root / HEAD).read_text())
    checks.append(
        _check(
            "frozen_hash",
            "exp7703-typed-decision-energy",
            str(FROZEN),
            "sha256",
            head.get("frozen_output_hashes", {}).get(str(FROZEN)),
            hashes["producers"][str(FROZEN)],
        )
    )
    protocol = json.loads((root / COHORT).read_text())
    for role in ("online_update", "online_admission", "retention"):
        info = protocol["roles"][role]
        for name, expected, bucket in (
            (info["model_inputs"], info["model_inputs_sha256"], "producers"),
            (
                protocol["evaluator_stores"][role]["path"],
                protocol["evaluator_stores"][role]["sha256"],
                "pre_gate_receipts",
            ),
        ):
            relative = COHORT.parent / name
            path = root / relative
            observed = sha256_file(path) if path.is_file() else None
            checks.append(
                _check(
                    "input_hash",
                    "exp7701-sealed-cohort",
                    str(relative),
                    "sha256",
                    expected,
                    observed,
                )
            )
            if observed is None:
                hashes["missing_inputs"].append(str(relative))
            else:
                hashes[bucket][str(relative)] = observed
    frozen = json.loads((root / FROZEN).read_text())
    checks.append(
        _check(
            "frozen_grammar",
            "exp7703-typed-decision-energy",
            str(FROZEN),
            "primitives_and_pairs",
            grammar(),
            {"primitives": frozen.get("primitives"), "pairs": frozen.get("pairs")},
        )
    )
    checks.append(
        _check(
            "frozen_budget",
            "exp7703-typed-decision-energy",
            str(FROZEN),
            "max_proposals_per_arm",
            6,
            frozen.get("max_proposals_per_arm"),
        )
    )
    checks.append(
        _check(
            "frozen_steps",
            "exp7703-typed-decision-energy",
            str(FROZEN),
            "max_gradient_steps_per_proposal",
            50,
            frozen.get("max_gradient_steps_per_proposal"),
        )
    )
    return checks, hashes


def fixture_cases() -> dict[str, list[dict[str, Any]]]:
    """Make independent deterministic cases to test mechanics, not science."""
    names = grammar()["primitives"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for partition, count in (("update", 64), ("admission", 32), ("retention", 32)):
        cases = []
        for index in range(count):
            label = int(index % 4 == 0)
            cases.append(
                {
                    "unit_id": f"fixture-{partition}-{index:03d}",
                    "source_id": f"源/{partition}/{index:03d}",
                    "partition": partition,
                    "features": {
                        name: float((index + offset) % 3 > 0) for offset, name in enumerate(names)
                    },
                    "base_probability": 0.1 if label else 0.2,
                    "verifier_status": ("contradicted", "unknown", "supported")[index % 3],
                    "label": label,
                    "oracle_role": "deterministic_fixture_truth",
                }
            )
        groups[partition] = cases
    return groups


def _row(
    case: dict[str, Any], arm: str, probability: float, tick: int | None, release_tick: int | None
) -> dict[str, Any]:
    """Keep the original case, measured forecast, and exact authority."""
    label = case["label"]
    return {
        "unit_id": case["unit_id"],
        "arm": arm,
        "partition": case["partition"],
        "prediction_tick": tick,
        "label_release_tick": release_tick,
        "raw_metrics": {
            "probability": probability,
            "label": label,
            "brier": (probability - label) ** 2,
            "base_probability": case["base_probability"],
            "base_brier": (case["base_probability"] - label) ** 2,
            "advisory_status": advisory_status(
                case["verifier_status"], probability != case["base_probability"]
            ),
            "exact_status": case["verifier_status"],
        },
        "counts": {"independent_unit": 1, "paired_arm": 1},
        "exclusions": [],
        "censored": False,
        "provenance": {
            "source_id": case["source_id"],
            "oracle_role": case["oracle_role"],
            "grammar": str(FROZEN),
        },
    }


def _admit_decision(bank: Bank, event_id: str) -> bool:
    """Compare a frozen candidate on a released, unused admission label."""
    proposal = bank.state["proposal"]
    row = bank.state["predictions"][event_id]
    label = bank.state["feedback"][event_id]["label"]
    if proposal is None or row["tick"] <= proposal["freeze_tick"]:
        return False
    active = all(row["features"][name] > 0 for name in proposal["pair"])
    candidate = min(1 - 1e-6, max(1e-6, row["probability"] + proposal["weight"] * active))
    return (candidate - label) ** 2 < (row["probability"] - label) ** 2


def measure_fixture(raw: Path, frozen_grammar: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Replay identical fixture cases through each bounded scheduler and control."""
    raw.mkdir(parents=True, exist_ok=True)
    cases = fixture_cases()
    rows: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    budgets: dict[str, dict[str, int]] = {}
    states: dict[str, str] = {}
    parity = True
    started = time.monotonic()
    for arm in ARMS:
        progress(started, "measurement", f"arm_start:{arm}", len(rows))
        scheduler = "fixed" if arm.startswith("fixed_") else arm
        period = int(arm.split("_")[1]) if scheduler == "fixed" else 2
        state_path = raw / f"bank_{arm}.json"
        if state_path.exists():
            state_path.unlink()
        bank = Bank(state_path, frozen_grammar, scheduler, threshold, period)
        tick = 0
        next_admission = 0
        pending_order: list[str] = []
        source_cases = {case["unit_id"]: case for group in cases.values() for case in group}

        def drain(now: int) -> None:
            nonlocal next_admission
            while pending_order and bank.state["predictions"][pending_order[0]]["tick"] + 8 <= now:
                event_id = pending_order.pop(0)
                case = source_cases[event_id]
                bank.release(event_id, case["label"], now)
                if case["partition"] == "admission" and bank.state["proposal"] is not None:
                    if (
                        bank.state["predictions"][event_id]["tick"]
                        > bank.state["proposal"]["freeze_tick"]
                    ):
                        bank.admit(_admit_decision(bank, event_id), event_id)
                elif case["partition"] == "update" and bank.state["proposal"] is None:
                    bank.propose()

        for index, case in enumerate(cases["update"]):
            drain(tick)
            forecast = bank.predict(
                case["unit_id"],
                tick,
                case["features"],
                case["base_probability"],
                case["verifier_status"],
                "update",
                case["source_id"],
            )
            pending_order.append(case["unit_id"])
            rows.append(_row(case, arm, forecast["probability"], tick, None))
            tick += 1
            if index % 2 and next_admission < 32:
                drain(tick)
                admission = cases["admission"][next_admission]
                forecast = bank.predict(
                    admission["unit_id"],
                    tick,
                    admission["features"],
                    admission["base_probability"],
                    admission["verifier_status"],
                    "admission",
                    admission["source_id"],
                )
                pending_order.append(admission["unit_id"])
                rows.append(_row(admission, arm, forecast["probability"], tick, None))
                next_admission += 1
                tick += 1
            if index == 32:
                reloaded = Bank(state_path, frozen_grammar, scheduler, threshold, period)
                parity &= reloaded.state_hash == bank.state_hash and bool(pending_order)
            if (index + 1) % 16 == 0:
                atomic_json(
                    raw / "measurement_checkpoint.json",
                    {
                        "arm": arm,
                        "completed_updates": index + 1,
                        "completed_rows": len(rows),
                        "state_hash": bank.state_hash,
                    },
                )
                progress(started, "measurement", f"checkpoint:{arm}", index + 1)
        while pending_order:
            tick = max(tick, bank.state["predictions"][pending_order[0]]["tick"] + 8)
            drain(tick)
        if bank.state["proposal"] is not None:
            bank.reject("no_future_released_admission")
        for case in cases["retention"]:
            adjustment = sum(
                item["weight"]
                for item in bank.state["templates"]
                if all(case["features"][key] > 0 for key in item["pair"])
            )
            probability = min(1 - 1e-6, max(1e-6, case["base_probability"] + adjustment))
            rows.append(_row(case, arm, probability, None, None))
        for row in rows:
            if row["arm"] == arm and row["partition"] != "retention":
                row["label_release_tick"] = bank.state["feedback"][row["unit_id"]]["release_tick"]
        budgets[arm] = dict(bank.budget)
        states[arm] = str(state_path)
        parity &= bank.replay_ledger()["state_hash"] == bank.state_hash
        lifecycle.extend(
            {
                "arm": arm,
                "sequence": event["sequence"],
                "kind": event["kind"],
                "detail": event["detail"],
                "previous": event["previous"],
                "event_hash": event["event_hash"],
                "state_hash": event["state_hash"],
            }
            for event in bank.state["ledger"]
        )
        progress(started, "measurement", f"arm_complete:{arm}", len(rows))
    return {
        "rows": rows,
        "lifecycle_rows": lifecycle,
        "budget_accounting": budgets,
        "bank_state_paths": states,
        "restart_exact_parity": parity,
        "ledger_reduction_valid": all(
            replay_ledger(json.loads(Path(path).read_text())["ledger"])["state_hash"]
            == json.loads(Path(path).read_text())["ledger"][-1]["state_hash"]
            for path in states.values()
        ),
    }


def cold_reduce(path: Path) -> dict[str, Any]:
    """Recompute raw losses and replay every bank in a fresh process."""
    artifact = json.loads(path.read_text())
    rows = artifact["rows"]
    units = {row["unit_id"] for row in rows}
    arms = {row["arm"] for row in rows}
    keys = [(row["arm"], row["unit_id"]) for row in rows]
    arithmetic = all(
        abs(
            row["raw_metrics"]["brier"]
            - (row["raw_metrics"]["probability"] - row["raw_metrics"]["label"]) ** 2
        )
        < 1e-12
        and row["raw_metrics"]["advisory_status"] == row["raw_metrics"]["exact_status"]
        and (
            row["prediction_tick"] is None
            or row["label_release_tick"] >= row["prediction_tick"] + 8
        )
        for row in rows
    )
    lifecycle = artifact["lifecycle_rows"]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for event in lifecycle:
        by_arm.setdefault(event["arm"], []).append(event)
    chain = all(
        all(
            event["sequence"] == index
            and event["previous"] == (events[index - 1]["event_hash"] if index else "genesis")
            for index, event in enumerate(events)
        )
        for events in by_arm.values()
    )
    paths = artifact.get("bank_state_paths", {})
    bank_replay = all(
        replay_ledger(json.loads(Path(state_path).read_text())["ledger"])["state_hash"]
        == events[-1]["state_hash"]
        for arm, events in by_arm.items()
        if (state_path := paths.get(arm))
    )
    valid = (
        arithmetic
        and chain
        and bank_replay
        and len(keys) == len(set(keys))
        and len(units) == 128
        and arms == set(ARMS)
    )
    return {
        "independent_units": len(units),
        "arm_count": len(arms),
        "row_count": len(rows),
        "lifecycle_valid": valid,
        "arithmetic_valid": arithmetic,
        "ledger_chain_valid": chain,
        "bank_replay_valid": bank_replay,
        "mean_brier_by_arm": {
            arm: sum(row["raw_metrics"]["brier"] for row in rows if row["arm"] == arm) / 128
            for arm in sorted(arms)
        },
        "retention_brier_by_arm": {
            arm: sum(
                row["raw_metrics"]["brier"]
                for row in rows
                if row["arm"] == arm and row["partition"] == "retention"
            )
            / 32
            for arm in sorted(arms)
        },
    }


def _gate(name: str, passed: bool | None, operands: dict[str, Any]) -> dict[str, Any]:
    return {
        "gate": name,
        "passed": passed,
        "operands": operands,
        "principle": GATE_PRINCIPLES[name],
    }


def build_artifact(
    date: str,
    started: float,
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    measured: dict[str, Any],
    scope: dict[str, Any],
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep fixture mechanics and administrative readiness separate."""
    blocked = any(not check["passed"] for check in checks)
    rows = measured.get("rows", [])
    budgets = measured.get("budget_accounting", {})
    bounded = bool(budgets) and all(
        item["proposal_credits_spent"] <= 6
        and item["admission_credits_spent"] <= 6
        and item["gradient_steps_spent"] <= 300
        for item in budgets.values()
    )
    complete = len({row["unit_id"] for row in rows}) == 128 and len(rows) == 128 * len(ARMS)
    durable = (
        measured.get("restart_exact_parity") is True
        and measured.get("ledger_reduction_valid") is True
    )
    acquired = any(row["kind"] == "commit" for row in measured.get("lifecycle_rows", []))
    mean_brier = (
        {
            arm: sum(row["raw_metrics"]["brier"] for row in rows if row["arm"] == arm) / 128
            for arm in ARMS
        }
        if complete
        else {}
    )
    mean_base_brier = (
        {
            arm: sum(row["raw_metrics"]["base_brier"] for row in rows if row["arm"] == arm) / 128
            for arm in ARMS
        }
        if complete
        else {}
    )
    retention_brier = (
        {
            arm: sum(
                row["raw_metrics"]["brier"]
                for row in rows
                if row["arm"] == arm and row["partition"] == "retention"
            )
            / 32
            for arm in ARMS
        }
        if complete
        else {}
    )
    thresholds = measured.get("policy_thresholds")
    charged_cost = (
        {
            arm: sum(
                decision_cost(
                    action(row["raw_metrics"]["probability"], tuple(thresholds)),
                    row["raw_metrics"]["label"],
                )
                for row in rows
                if row["arm"] == arm
            )
            / 128
            for arm in ARMS
        }
        if complete and thresholds is not None
        else {}
    )
    gates = [
        _gate(
            "validity",
            not blocked and complete and bounded and durable,
            {
                "input_checks_passed": sum(check["passed"] for check in checks),
                "input_checks_total": len(checks),
                "rows": len(rows),
                "bounded": bounded,
                "durable": durable,
            },
        ),
        _gate(
            "readiness",
            None,
            {
                "administrative_readiness": None,
                "fixture_oracle": True,
                "qualified_empirical_acquisition": False,
            },
        ),
        _gate(
            "coverage",
            complete,
            {
                "observed_groups": len({row["unit_id"] for row in rows}),
                "intended_groups": 128,
                "arms": len(ARMS),
            },
        ),
        _gate("freshness", None, {"fixture_groups": 128, "empirical_fresh_groups": 0}),
        _gate(
            "probability",
            None,
            {
                "fixture_oracle": True,
                "acquired": acquired,
                "mean_brier_by_arm": mean_brier,
                "mean_base_brier_by_arm": mean_base_brier,
                "empirical_effect": None,
            },
        ),
        _gate(
            "utility",
            None,
            {
                "fixture_oracle": True,
                "untouched_admission": 32,
                "untouched_retention": 32,
                "frozen_policy_thresholds": thresholds,
                "mean_charged_decision_cost_by_arm": charged_cost,
            },
        ),
        _gate(
            "retention",
            None,
            {
                "retention_groups": 32,
                "mean_retention_brier_by_arm": retention_brier,
                "empirical_forgetting": None,
            },
        ),
        _gate(
            "efficiency",
            None,
            {
                "cpu_fixture": True,
                "elapsed_s": time.monotonic() - started,
                "rows_per_elapsed_s": len(rows) / max(time.monotonic() - started, 1e-9),
                "native_speedup_measured": False,
            },
        ),
    ]
    verdict_class = (
        "blocked"
        if blocked
        else "circular_positive"
        if complete and bounded and durable
        else "null"
    )
    verdict = (
        "complete_blocked_current_upstream"
        if blocked
        else "complete_circular_positive_fixture_mechanics"
        if acquired
        else "complete_circular_positive_fixture_mechanics_no_acquisition"
        if complete and bounded and durable
        else "complete_null_fixture_incomplete"
    )
    return {
        "schema": "carnot.exp7705.v671.constraint_bank.v1",
        "experiment_id": 7705,
        "milestone": "2026.09.671",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": [check for check in checks if not check["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "lifecycle_rows": measured.get("lifecycle_rows", []),
        "budget_accounting": budgets,
        "bank_state_paths": measured.get("bank_state_paths", {}),
        "bank_schema_path": str(RAW / "bank_schema.json"),
        "constraint_bank_ready_score": 0,
        "continuous_self_learning_task": True,
        "sample_size_budget": {
            "intended_groups": {"update": 64, "admission": 32, "retention": 32},
            "observed_groups": len({row["unit_id"] for row in rows}),
            "eligible_groups": len({row["unit_id"] for row in rows}),
            "excluded_groups": 0,
            "censored_groups": 0,
            "effective_blocks": len({row["unit_id"] for row in rows}),
            "prior_exposure": "deterministic_circular_fixture",
            "inference_limits": "No independent empirical acquisition claim; arms and seeds do not enlarge n.",
        },
        "inference_substrate": "cpu_deterministic_fixture_constraint_bank",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [{"declaration": "no_model_invoked", "model_id": None}],
        "model_invoked": False,
        "invocation_counts": {
            "model_loads": 0,
            "forwards": 0,
            "generations": 0,
            "tokens": 0,
            "failures": 0,
            "cancellations": 0,
        },
        "execution_venue": "host",
        "execution_venue_details": {"host": platform.node(), "pid": os.getpid(), "gpu_uuid": None},
        "effective_agent_backend": "codex"
        if os.getenv("CODEX_FORCE_EXPERIMENTS") == "1"
        else "codex_current_session",
        "execution_recovery_receipt": {
            "force_codex": os.getenv("CODEX_FORCE_EXPERIMENTS"),
            "session_id": os.getenv("CODEX_SESSION_ID"),
            "successful_current_invocation": bool(os.getenv("CODEX_SESSION_ID")),
        },
        "phase_spans": spans,
        "duration_s": time.monotonic() - started,
        "random_seed": {
            "fixture": 7705,
            "purpose": "fixed case IDs and deterministic labels; no stochastic training",
        },
        "reproducibility_checksum": canonical_hash(
            {
                "inputs": hashes,
                "grammar": grammar(),
                "module_sha256": sha256_file(ROOT / MODULE),
                "reducer_sha256": sha256_file(ROOT / ORCHESTRATION),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": scope,
            "affected": [],
            "full_suite": [],
            "terminal_readers": [],
        },
        "verifier_is_oracle": True,
        "field_principles": {
            **PRINCIPLES,
            **{f"acceptance_gate_{name}": principle for name, principle in GATE_PRINCIPLES.items()},
        },
        "hardware_path": "CPU counters and bitsets now; bounded vectorized batches and native kernels may remove Python overhead; no speedup measured.",
        "restart_exact_parity": measured.get("restart_exact_parity", False),
        "ledger_reduction_valid": measured.get("ledger_reduction_valid", False),
        "prior_failures": [
            {
                "experiment_id": "exp7691-constraint-bank-protocol",
                "custody": "not_emitted_upstream_gate_skip",
                "scientific_verdict": None,
            }
        ],
        "same_verdict_retirements": [
            {
                "mechanism": "frozen source-specific typed-certificate decision head",
                "status": "already_retired_by_exp7704; bank proposal is a distinct mechanism",
            }
        ],
    }


def run_experiment(root: Path, date: str, output: Path, *, validate: bool = True) -> dict[str, Any]:
    """Freeze scope, run fixture mechanics, validate exact bytes and publish."""
    root = root.resolve()
    output = output.resolve()
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    progress(started, "preflight", "start")
    begin = time.monotonic() - started
    checks, hashes = preflight(root)
    spans.append(
        {
            "phase": "preflight",
            "start_s": begin,
            "end_s": time.monotonic() - started,
            "duration_s": time.monotonic() - started - begin,
            "completed_units": len(checks),
            "heartbeat_timestamps": [time.time()],
        }
    )
    progress(started, "preflight", "complete", len(checks))
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    scope = dict(SCOPE)
    scope["frozen_at_utc_epoch_s"] = time.time()
    atomic_json(raw / "frozen_affected_scope.json", scope)
    atomic_json(
        raw / "bank_schema.json",
        {
            "schema": "carnot.exp7705.bank.v1",
            "grammar": grammar(),
            "max_pending": 16,
            "max_templates": 36,
            "max_proposals_per_arm": 6,
            "max_gradient_steps_per_proposal": 50,
            "delay_ticks": 8,
            "arms": list(ARMS),
            "threshold_source": str(FROZEN),
        },
    )
    measured: dict[str, Any] = {}
    if all(check["passed"] for check in checks):
        frozen = json.loads((root / FROZEN).read_text())
        progress(started, "measurement", "start")
        begin = time.monotonic() - started
        measured = measure_fixture(raw, grammar(), float(frozen["scheduler"]["threshold"]))
        measured["policy_thresholds"] = frozen["thresholds"]
        spans.append(
            {
                "phase": "measurement",
                "start_s": begin,
                "end_s": time.monotonic() - started,
                "duration_s": time.monotonic() - started - begin,
                "completed_units": 128 * len(ARMS),
                "checkpoint": str(RAW / "measurement_checkpoint.json"),
                "heartbeat_timestamps": [time.time()],
            }
        )
        progress(started, "measurement", "complete", len(measured["rows"]))
    else:
        atomic_json(
            raw / "measurement_checkpoint.json",
            {
                "completed_units": 0,
                "blocked_checks": [check for check in checks if not check["passed"]],
            },
        )
    artifact = build_artifact(date, started, checks, hashes, measured, scope, spans)
    if not validate:
        atomic_json(output, artifact)
        return artifact
    with tempfile.TemporaryDirectory(prefix="exp7705-validation-") as private_name:
        private = Path(private_name)
        (private / "basetemp").mkdir()
        commands = build_scoped_commands(
            root,
            [TEST],
            [MODULE, ORCHESTRATION],
            static_paths=[WRAPPER],
            basetemp=private / "basetemp",
            coverage_file=private / ".coverage",
        )
        commands = [
            CommandSpec(spec.name, spec.argv, spec.scope, timeout_s=1800)
            if spec.name == "changed_module_coverage"
            else spec
            for spec in commands
        ]
        progress(started, "affected_validation", "before_subprocess")
        begin = time.monotonic() - started
        receipts = run_commands(
            root,
            commands,
            log_dir=raw / "validation/affected",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        spans.append(
            {
                "phase": "affected_validation",
                "start_s": begin,
                "end_s": time.monotonic() - started,
                "duration_s": time.monotonic() - started - begin,
                "completed_units": len(receipts),
                "heartbeat_timestamps": [time.time()],
            }
        )
        artifact["validation_receipts"]["affected"] = receipts
        progress(started, "affected_validation", "after_subprocess", len(receipts))
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
        progress(started, "full_python_suite", "before_subprocess")
        begin = time.monotonic() - started
        full_receipts = run_commands(
            root,
            full,
            log_dir=raw / "validation/full",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        spans.append(
            {
                "phase": "full_python_suite",
                "start_s": begin,
                "end_s": time.monotonic() - started,
                "duration_s": time.monotonic() - started - begin,
                "completed_units": len(full_receipts),
                "heartbeat_timestamps": [time.time()],
            }
        )
        artifact["validation_receipts"]["full_suite"] = full_receipts
        progress(started, "full_python_suite", "after_subprocess", len(full_receipts))
        candidate = raw / "terminal_candidate.json"
        artifact["phase_spans"] = spans
        artifact["duration_s"] = time.monotonic() - started
        atomic_json(candidate, artifact)
        readers = [
            CommandSpec(
                "cold_reduce",
                (
                    python,
                    "-u",
                    "-m",
                    "carnot.experiment_7705_v671_constraint_bank_protocol",
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
        spans.append(
            {
                "phase": "terminal_readers",
                "start_s": begin,
                "end_s": time.monotonic() - started,
                "duration_s": time.monotonic() - started - begin,
                "completed_units": len(terminal),
                "heartbeat_timestamps": [time.time()],
            }
        )
        artifact["validation_receipts"]["terminal_readers"] = terminal
        artifact["validation_receipts"]["exact_terminal_candidate_sha256"] = sha256_file(candidate)
        progress(started, "terminal_readers", "after_subprocess", len(terminal))
    failed = [
        receipt
        for receipt in (
            *artifact["validation_receipts"]["affected"],
            *artifact["validation_receipts"]["terminal_readers"],
        )
        if not receipt["passed"]
    ]
    if failed:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["constraint_bank_ready_score"] = 0
        artifact["flagged_adversarial"] = any(
            receipt["name"] == "adversarial_verify" for receipt in failed
        )
        artifact["gate_check_summary"].extend(
            _check(
                "required_validation",
                "current",
                receipt["log_path"],
                "exit_code",
                0,
                receipt["exit_code"],
            )
            for receipt in failed
        )
        for gate in artifact["acceptance_gate_results"]:
            if gate["gate"] in {"validity", "readiness"}:
                gate["passed"] = False
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    progress(started, "publish", "before_atomic_write")
    atomic_json(output, artifact)
    progress(started, "publish", "complete")
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Accept the declared entrypoint and independent cold-reduction mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        result = cold_reduce(args.cold_reduce)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["lifecycle_valid"] else 1
    run_experiment(ROOT.resolve(), args.date, ROOT / args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI process boundary
    raise SystemExit(main())
