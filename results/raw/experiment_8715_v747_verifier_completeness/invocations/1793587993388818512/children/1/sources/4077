"""Deterministic causal admission mechanics for REQ-CL-7719-CAUSAL-ADMISSION."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from carnot.reporting.constraint_bank_protocol import Bank, grammar
from carnot.reporting.current_work_receipt import canonical_hash


ARMS = ("priority", "fixed_2", "read_only", "weight_only", "static")
EPSILON = 1e-6


def feature_vector(kind: str) -> dict[str, float]:
    """Expose only the frozen thresholded advisory names to a predictor."""
    names = grammar()["primitives"]
    active = names[:2] if kind == "beneficial" else names[2:4] if kind == "harmful" else []
    return {name: float(name in active) for name in names}


def fixture_groups() -> dict[str, list[dict[str, Any]]]:
    """Create 96 independent source families with labels kept out of features."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for role in ("development", "admission", "retention"):
        cases = []
        for index in range(32):
            if role == "development":
                kind = "beneficial" if index == 0 else "harmful" if index == 16 else "neutral"
                label = int(index in (0, 16))
            else:
                kind = (
                    "beneficial"
                    if index < 5 or 10 <= index < 15
                    else "harmful"
                    if index < 10
                    else "neutral"
                )
                label = int(index < 5 or 10 <= index < 15)
            probability = (
                1 - EPSILON
                if 10 <= index < 15 and role != "development"
                else 0.1
                if kind == "beneficial"
                else 0.2
            )
            source = f"Original fixture source {role} {index:02d}: {kind} advisory evidence.\n"
            cases.append(
                {
                    "unit_id": f"v672-{role}-{index:02d}",
                    "role": role,
                    "kind": kind,
                    "features": feature_vector(kind),
                    "base_probability": probability,
                    "label": label,
                    "exact_status": ("unknown", "contradicted", "supported")[index % 3],
                    "source_bytes_hex": source.encode("utf-8").hex(),
                }
            )
        groups[role] = cases
    return groups


def _closure_features(features: dict[str, float]) -> dict[str, float]:
    values = {name: features[name] for name in grammar()["primitives"]}
    values.update({f"{a}&{b}": features[a] * features[b] for a, b in grammar()["pairs"]})
    return values


def fit_static_closure(development: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit every primitive and pair on development labels, never admissions."""
    names = list(_closure_features(development[0]["features"]))
    coefficients = {name: 0.0 for name in names}
    for _ in range(50):
        for case in development:
            active = _closure_features(case["features"])
            forecast = min(
                1 - EPSILON,
                max(
                    EPSILON,
                    case["base_probability"]
                    + sum(coefficients[name] * value for name, value in active.items()),
                ),
            )
            error = forecast - case["label"]
            for name, value in active.items():
                coefficients[name] = max(
                    -0.4, min(0.4, coefficients[name] - 0.01 * error * value / 32)
                )
    changed = any(abs(value) > 1e-12 for value in coefficients.values())
    return {
        "features": names,
        "coefficients": coefficients,
        "fit_family_count": len(development),
        "fit_steps": 50,
        "nonzero_coefficient_count": sum(abs(value) > 1e-12 for value in coefficients.values()),
        "different_from_empty_bank": changed,
        "fit_role": "development_only",
    }


def static_probability(case: dict[str, Any], receipt: dict[str, Any]) -> float:
    """Use the fitted 36-feature control without opening held-out labels."""
    delta = sum(
        receipt["coefficients"][name] * value
        for name, value in _closure_features(case["features"]).items()
    )
    return min(1 - EPSILON, max(EPSILON, case["base_probability"] + delta))


def admission_decision(bank: Bank, frozen: list[dict[str, Any]]) -> bool:
    """Compare a single frozen candidate on exactly five later released labels."""
    if len(frozen) != 5 or len({row["unit_id"] for row in frozen}) != 5:
        raise ValueError("five_frozen_forecasts_required")
    proposal = bank.state["proposal"]
    if proposal is None:
        raise ValueError("proposal_missing")
    losses = [0.0, 0.0]
    for row in frozen:
        prediction = bank.state["predictions"].get(row["unit_id"])
        feedback = bank.state["feedback"].get(row["unit_id"])
        if prediction is None or feedback is None or feedback["status"] != "released":
            raise ValueError("admission_feedback_missing")
        if prediction["partition"] != "admission" or prediction["tick"] <= proposal["freeze_tick"]:
            raise ValueError("admission_not_postfreeze")
        if row["base_probability"] != prediction["probability"]:
            raise ValueError("frozen_forecast_mismatch")
        label = feedback["label"]
        losses[0] += (row["base_probability"] - label) ** 2
        losses[1] += (row["candidate_probability"] - label) ** 2
    return losses[1] + 1e-12 < losses[0]


def feedback_diagnostics(raw: Path) -> dict[str, bool]:
    """Exercise malformed delayed feedback and an actual child hard exit."""
    path = raw / "bank_diagnostics.json"
    if path.exists():
        path.unlink()
    bank = Bank(path, grammar(), "priority", 0.1, 2)
    for index in range(16):
        bank.predict(
            f"diagnostic-{index}",
            index,
            feature_vector("beneficial"),
            0.1,
            "unknown",
            "update",
            f"diagnostic-{index}",
        )
    flags = {}
    for name, callback, expected in (
        (
            "overflow",
            lambda: bank.predict(
                "overflow", 16, feature_vector("beneficial"), 0.1, "unknown", "update", "overflow"
            ),
            "pending_overflow",
        ),
        ("out_of_order", lambda: bank.release("diagnostic-1", 1, 16), "feedback_order"),
    ):
        flags[name] = False
        try:
            callback()
        except ValueError as error:
            flags[name] = str(error) == expected
    bank.release("diagnostic-0", 1, 16)
    flags["duplicate"] = False
    try:
        bank.release("diagnostic-0", 1, 17)
    except ValueError as error:
        flags["duplicate"] = str(error) == "duplicate_feedback"
    proposal = bank.propose()
    before = bank.state_hash
    child = subprocess.run(
        (
            sys.executable,
            "-c",
            "import os,sys; from pathlib import Path; from carnot.reporting.constraint_bank_protocol import Bank,grammar; Bank(Path(sys.argv[1]),grammar(),'priority',.1,2); os._exit(0)",
            str(path),
        ),
        check=False,
        timeout=30,
    )
    reopened = Bank(path, grammar(), "priority", 0.1, 2)
    flags["hard_exit_replay"] = (
        child.returncode == 0
        and reopened.state_hash == before
        and reopened.state["proposal"] == proposal
    )
    reopened.reject("diagnostic_end")
    reopened.mark_missing("diagnostic-1", 17)
    flags["missing"] = False
    try:
        reopened.propose()
    except ValueError as error:
        flags["missing"] = str(error) == "missing_feedback"
    flags["rejected_credit"] = (
        reopened.budget["proposal_credits_spent"] == 1
        and reopened.budget["admission_credits_spent"] == 1
    )
    return flags


def run_fixture(raw: Path, progress: Any = None) -> dict[str, Any]:
    """Run matched advisory arms, save checkpoints and reopen each ledger."""
    raw.mkdir(parents=True, exist_ok=True)
    groups = fixture_groups()
    closure = fit_static_closure(groups["development"])
    rows: list[dict[str, Any]] = []
    reachability: list[dict[str, Any]] = []
    budgets: dict[str, dict[str, int]] = {}
    state_paths: dict[str, str] = {}
    parity = True

    for arm in ARMS:
        if progress:
            progress("measurement", f"arm_start:{arm}", len(rows))
        scheduler = "fixed" if arm == "fixed_2" else arm
        state_path = raw / f"bank_{arm}.json"
        if state_path.exists():
            state_path.unlink()
        bank = Bank(state_path, grammar(), scheduler, 0.1, 2)
        tick = 0

        def predict(case: dict[str, Any], partition: str) -> dict[str, Any]:
            nonlocal tick
            model_base = (
                static_probability(case, closure) if arm == "static" else case["base_probability"]
            )
            saved = bank.predict(
                case["unit_id"],
                tick,
                case["features"],
                model_base,
                case["exact_status"],
                partition,
                case["unit_id"],
            )
            row = {
                "unit_id": case["unit_id"],
                "arm": arm,
                "role": case["role"],
                "raw_metrics": {
                    "probability": saved["probability"],
                    "original_base_probability": case["base_probability"],
                    "model_base_probability": model_base,
                    "label": None,
                    "brier": None,
                    "exact_status": case["exact_status"],
                    "advisory_status": case["exact_status"],
                },
                "denominators": {"independent_family": 1, "paired_arm": 1},
                "exclusions": [],
                "censored": False,
                "provenance": {
                    "source_bytes_hex": case["source_bytes_hex"],
                    "oracle_role": "deterministic_fixture_truth",
                    "prediction_tick": tick,
                    "label_release_tick": None,
                },
            }
            rows.append(row)
            tick += 1
            return row

        def release(cases: list[dict[str, Any]]) -> None:
            nonlocal tick
            release_tick = tick + 8
            for case in cases:
                bank.release(case["unit_id"], case["label"], release_tick)
                row = next(
                    item
                    for item in reversed(rows)
                    if item["arm"] == arm and item["unit_id"] == case["unit_id"]
                )
                row["raw_metrics"]["label"] = case["label"]
                row["raw_metrics"]["brier"] = (
                    row["raw_metrics"]["probability"] - case["label"]
                ) ** 2
                row["provenance"]["label_release_tick"] = release_tick
            tick = release_tick + 1

        def batch(cases: list[dict[str, Any]]) -> None:
            proposal = bank.state["proposal"]
            frozen = []
            for case in cases:
                row = predict(case, "admission")
                base = row["raw_metrics"]["probability"]
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
                        "base_probability": base,
                        "candidate_probability": candidate,
                        "freeze_tick": proposal["freeze_tick"] if proposal else None,
                    }
                )
            release(cases)
            if proposal is not None:
                accepted = admission_decision(bank, frozen)
                bank.admit(accepted, frozen[0]["unit_id"])
                pair = proposal["pair"]
                later = next(
                    case
                    for case in groups["retention"]
                    if all(case["features"][name] > 0 for name in pair)
                )
                reachability.append(
                    {
                        "arm": arm,
                        "counterexample_id": proposal["counterexample_id"],
                        "proposal_freeze_tick": proposal["freeze_tick"],
                        "proposal_weight": proposal["weight"],
                        "frozen_forecasts": frozen,
                        "label_release_tick": bank.state["feedback"][frozen[0]["unit_id"]][
                            "release_tick"
                        ],
                        "admission_labels": [case["label"] for case in cases],
                        "decision": "commit" if accepted else "rollback",
                        "later_unit_id": later["unit_id"],
                        "later_forecast_delta": None,
                    }
                )

        for block in range(2):
            development = groups["development"][block * 16 : (block + 1) * 16]
            for case in development:
                predict(case, "update")
            release(development)
            bank.propose()
            batch(groups["admission"][block * 5 : (block + 1) * 5])
            reopened = Bank(state_path, grammar(), scheduler, 0.1, 2)
            parity &= reopened.state_hash == bank.state_hash
            if progress:
                progress("measurement", f"checkpoint:{arm}:{block}", len(rows))
            (raw / "measurement_checkpoint.json").write_text(
                json.dumps(
                    {
                        "arm": arm,
                        "block": block,
                        "completed_rows": len(rows),
                        "state_hash": bank.state_hash,
                    }
                )
            )
        for start in (10, 15, 20, 25, 30):
            batch(groups["admission"][start : min(start + 5, 32)])
        for start in range(0, 32, 8):
            group = groups["retention"][start : start + 8]
            for case in group:
                predict(case, "retention")
            release(group)
        for claim in (item for item in reachability if item["arm"] == arm):
            observed = next(
                item
                for item in rows
                if item["arm"] == arm and item["unit_id"] == claim["later_unit_id"]
            )
            claim["later_forecast_delta"] = (
                observed["raw_metrics"]["probability"]
                - observed["raw_metrics"]["model_base_probability"]
            )
        reopened = Bank(state_path, grammar(), scheduler, 0.1, 2)
        parity &= reopened.state_hash == bank.state_hash
        parity &= reopened.replay_ledger()["state_hash"] == bank.state_hash
        budgets[arm] = dict(bank.budget)
        state_paths[arm] = str(state_path)
        if progress:
            progress("measurement", f"arm_complete:{arm}", len(rows))

    historical = (
        Path(__file__).resolve().parents[3]
        / "results/experiment_7705_v671_constraint_bank_protocol.json"
    )
    v671 = json.loads(historical.read_text()) if historical.is_file() else {}
    rollback_count = sum(row["kind"] == "rollback" for row in v671.get("lifecycle_rows", []))
    reproduced = (
        v671.get("constraint_bank_ready_score") == 0
        and rollback_count > 0
        and not any(row["kind"] == "commit" for row in v671.get("lifecycle_rows", []))
    )
    (raw / "v671_rollback_reproduction.json").write_text(
        json.dumps(
            {
                "rollback_count": rollback_count,
                "ready_score": v671.get("constraint_bank_ready_score"),
                "reproduced": reproduced,
            }
        )
    )
    bounded = all(
        item["proposal_credits_spent"] <= 6
        and item["admission_credits_spent"] <= 6
        and item["gradient_steps_spent"] <= 300
        and item["overflow_rejections"] == 0
        for item in budgets.values()
    )
    diagnostics = feedback_diagnostics(raw)
    return {
        "rows": rows,
        "arms": list(ARMS),
        "commit_reachability_rows": reachability,
        "static_closure_receipt": closure,
        "budget_accounting": budgets,
        "bank_state_paths": state_paths,
        "restart_exact_parity": parity,
        "budget_valid": bounded,
        "v671_rollback_reproduced": reproduced,
        "fixture_checksum": canonical_hash(groups),
        "feedback_diagnostics": diagnostics,
    }
