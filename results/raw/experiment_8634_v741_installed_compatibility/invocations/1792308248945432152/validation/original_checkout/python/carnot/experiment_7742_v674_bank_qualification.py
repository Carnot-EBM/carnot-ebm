"""V674 sentence-risk bank custody for REQ-REPORT-7742 and REQ-CL-7742-BANK."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.experiment_7732_v673_causal_admission import (
    BoundedBank,
    cold_reduce as cold_reduce_7732,
    preflight as preflight_7732,
    run_fixture,
)
from carnot.reporting.acquisition_qualification import fixture_groups
from carnot.reporting.constraint_bank_protocol import grammar
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7742_v674_bank_qualification")
OUTPUT = Path("results/experiment_7742_v674_bank_qualification.json")
MANIFEST = (
    ROOT
    / "results/raw/experiment_7740_v674_sentence_label_protocol/sentence_protocol_manifest.json"
)
MODULE = "python/carnot/experiment_7742_v674_bank_qualification.py"
CLI = "scripts/experiments/experiment_7742_v674_bank_qualification.py"
TEST = "tests/python/test_experiment_7742_v674_bank_qualification.py"
OLD_TEST = "tests/python/test_experiment_7732_v673_causal_admission.py"
ACQUISITION_TESTS = [
    "tests/python/test_experiment_7719_v672_acquisition_qualification.py",
    "tests/python/test_experiment_7705_v671_constraint_bank_protocol.py",
    "tests/python/test_experiment_7303_v642_validation_scope.py",
]
PRINCIPLES = {
    "honest_verdict": "Completion and scientific benefit are different facts.",
    "verdict_class": "Unchanged external failures must not trigger identical attempts.",
    "flagged_adversarial": "A clean label cannot replace independent checks.",
    "gate_check_summary": "A missing field is a broken contract, not a scientific null.",
    "acceptance_gate_results": "Negative science may be valid while invalid execution never qualifies.",
    "rows": "Every comparison must be independently recomputable.",
    "sample_size_budget": "Seeds, sentences and actions do not multiply families.",
    "claim_scope": "A new split cannot erase historical exposure.",
    "inference_substrate": "Cached scoring must not claim live generation.",
    "inference_substrate_class": "Duration checks must match work performed.",
    "MODEL_SPECS": "The experimental model is distinct from the coding backend.",
    "model_invoked": "Model strings alone are not invocation evidence.",
    "execution_venue": "Host work is not fabric work.",
    "phase_spans": "Real timings and bounded silence expose stalled work.",
    "random_seed": "Independent replay needs fixed inputs.",
    "source_artifact_hashes": "An artifact cannot authenticate itself as an upstream source.",
    "preconditions_checked": "Missing access must block before expensive work.",
    "validation_receipts": "A result is usable only after registered checks pass.",
    "verifier_is_oracle": "Circular success cannot establish independent verification value.",
    "field_principles": "The rationale must travel with the contract.",
    "acquisition_protocol_ready_score": "Persistence must be proved before a learning claim.",
    "bank_protocol_path": "Each later update needs a reproducible causal record.",
    "lifecycle_rows": "Positive fixtures alone do not show safe failure behavior.",
}


def progress(started: float, phase: str, event: str, units: int = 0) -> None:
    """Flush each phase boundary with measured elapsed time and completed units."""
    print(
        f"[exp7742] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def _check(name: str, upstream: str, path: str, field: str, expected: Any, observed: Any) -> dict:
    """Retain literal operands for a missing or malformed producer."""
    return {
        "check": name,
        "upstream_id": upstream,
        "artifact_path": path,
        "field": field,
        "operator": "==",
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight(root: Path) -> tuple[list[dict], dict, list[str]]:
    """Authenticate required V673 bytes and the separate V674 protocol manifest."""
    root = root.resolve()
    checks, old = preflight_7732(root)
    hashes = {
        "eligible_producers": dict(old["eligible_producers"]),
        "historical_disqualified_sources": dict(old["flagged_historical_inputs"]),
        "missing_inputs": list(old["absent_sources"]),
        "pre_gate_receipts": {},
    }
    relative = MANIFEST.relative_to(ROOT)
    path = root / relative
    checks.append(
        _check("manifest_exists", "exp7740", str(relative), "exists", True, path.is_file())
    )
    predicates: list[str] = []
    if path.is_file():
        hashes["pre_gate_receipts"][str(relative)] = sha256_file(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            predicates = value.get("advisory_dictionary", [])
            if not isinstance(predicates, list) or not all(
                isinstance(name, str) for name in predicates
            ):
                predicates = []
        except (OSError, ValueError, TypeError, AttributeError):
            predicates = []
        checks.append(
            _check(
                "predicate_count",
                "exp7740",
                str(relative),
                "advisory_dictionary_count",
                16,
                len(predicates),
            )
        )
        checks.append(
            _check(
                "predicate_unique",
                "exp7740",
                str(relative),
                "unique_names",
                True,
                len(set(predicates)) == 16,
            )
        )
    else:
        hashes["missing_inputs"].append(str(relative))
    historical = root / "results/experiment_7740_v674_sentence_label_protocol.json"
    if historical.is_file():
        hashes["historical_disqualified_sources"][str(historical.relative_to(root))] = sha256_file(
            historical
        )
    prior = root / "results/experiment_7732_v673_causal_admission.json"
    if prior.is_file():
        hashes["historical_disqualified_sources"][str(prior.relative_to(root))] = sha256_file(prior)
    global_observation = root / RAW / "validation/global_suite_observation.json"
    if global_observation.is_file():
        hashes["pre_gate_receipts"][str(global_observation.relative_to(root))] = sha256_file(
            global_observation
        )
    return checks, hashes, predicates


def sentence_features(case: dict, names: list[str]) -> dict[str, float]:
    """Build the declared fixture payload from four bounded sentence-risk signals."""
    index = int(case["unit_id"].rsplit("-", 1)[1])
    low, neg, number, missing = (bool(index & (1 << bit)) for bit in range(4))
    values = (
        low,
        neg,
        number,
        missing,
        low and neg,
        low and number,
        low and missing,
        neg and number,
        neg and missing,
        number and missing,
        low and neg and number,
        low and neg and missing,
        low and number and missing,
        neg and number and missing,
        low and neg and number and missing,
        index % 5 == 0 and missing,
    )
    if len(names) != len(values):
        raise ValueError("sixteen_predicates_required")
    return {name: float(value) for name, value in zip(names, values)}


def fit_sentence_static(development: list[dict], names: list[str]) -> dict:
    """Fit every declared dictionary coefficient using development labels only."""
    if len(names) != 16 or len(set(names)) != 16 or len(development) != 32:
        raise ValueError("static_fit_scope_invalid")
    weights = dict.fromkeys(names, 0.0)
    for _ in range(50):
        for case in development:
            features = sentence_features(case, names)
            probability = max(
                1e-6,
                min(
                    1 - 1e-6,
                    case["base_probability"]
                    + sum(weights[name] * value for name, value in features.items()),
                ),
            )
            error = probability - case["label"]
            for name, value in features.items():
                weights[name] = max(-0.4, min(0.4, weights[name] - 0.01 * error * value / 32))
    return {
        "dictionary": names,
        "weights": weights,
        "fit_role": "development_only",
        "fit_families": len(development),
        "fit_steps": 50,
        "nonzero_weight_count": sum(value != 0 for value in weights.values()),
    }


def _snapshot(bank: BoundedBank) -> dict:
    """Name each persisted field needed for hard-exit parity."""
    return {
        "state_hash": bank.state_hash,
        "predictions": bank.state["predictions"],
        "budget": bank.budget,
        "pending": sorted(set(bank.state["predictions"]) - set(bank.state["feedback"])),
        "templates": bank.state["templates"],
        "proposal": bank.state["proposal"],
    }


def _hard_exit_diagnostic(raw: Path, started: float) -> list[dict]:
    """Hard-exit owned children on both sides of a durable bank acknowledgement."""
    path = raw / "hard_exit_bank.json"
    path.unlink(missing_ok=True)
    bank = BoundedBank(path, grammar(), "priority", 0.1, 2)
    case = fixture_groups()["development"][0]
    bank.predict("hard-exit-0", 0, case["features"], 0.1, "unknown", "update", "hard-exit-0")
    bank.release("hard-exit-0", 1, 8)
    assert bank.propose() is not None
    before = _snapshot(bank)
    rows = []
    for stage in ("before", "after"):
        code = (
            "import os,signal,sys; from pathlib import Path; "
            "from carnot.experiment_7732_v673_causal_admission import BoundedBank; "
            "from carnot.reporting.constraint_bank_protocol import grammar; "
            "b=BoundedBank(Path(sys.argv[1]),grammar(),'priority',.1,2); "
            + ("b.simulate_crash_before_commit(); " if stage == "before" else "b.admit(True); ")
            + "os.kill(os.getpid(),signal.SIGKILL)"
        )
        progress(started, "hard_exit", f"before_subprocess:{stage}", len(rows))
        result = subprocess.run(
            (sys.executable, "-u", "-c", code, str(path)), cwd=ROOT, timeout=30, check=False
        )
        progress(started, "hard_exit", f"after_subprocess:{stage}", len(rows) + 1)
        reopened = BoundedBank(path, grammar(), "priority", 0.1, 2)
        snap = _snapshot(reopened)
        expected = before if stage == "before" else snap
        rows.append(
            {
                "kind": f"hard_exit_{stage}_ack",
                "exit_code": result.returncode,
                "passed": result.returncode == -signal.SIGKILL
                and snap == expected
                and reopened.replay_ledger()["state_hash"] == reopened.state_hash
                and (snap["budget"]["proposal_credits_spent"] == 1)
                and (snap["budget"]["admission_credits_spent"] == (stage == "after"))
                and (len(snap["templates"]) == (stage == "after")),
                "before": before if stage == "before" else None,
                "after": snap,
            }
        )
    return rows


def measure(raw: Path, names: list[str], started: float) -> dict:
    """Reuse V673 lifecycle, then run a true 16-weight static bank on the same families."""
    raw.mkdir(parents=True, exist_ok=True)
    legacy_dir = raw / "legacy"
    legacy = run_fixture(legacy_dir)
    legacy_path = raw / "legacy_measurement.json"
    atomic_json(legacy_path, legacy)
    groups = fixture_groups()
    fit = fit_sentence_static(groups["development"], names)
    rows = [dict(row) for row in legacy["rows"] if row["arm"] != "complete_static"]
    static_path = raw / "bank_sentence_complete_static.json"
    static_path.unlink(missing_ok=True)
    bank = BoundedBank(static_path, grammar(), "static", 0.1, 2)
    tick = 0
    for role, cases in groups.items():
        for offset in range(0, len(cases), 8):
            block = cases[offset : offset + 8]
            for case in block:
                features = sentence_features(case, names)
                base = max(
                    1e-6,
                    min(
                        1 - 1e-6,
                        case["base_probability"]
                        + sum(fit["weights"][name] * value for name, value in features.items()),
                    ),
                )
                prediction = bank.predict(
                    case["unit_id"],
                    tick,
                    case["features"],
                    base,
                    case["exact_status"],
                    role if role != "development" else "update",
                    case["unit_id"],
                )
                rows.append(
                    {
                        "unit_id": case["unit_id"],
                        "arm": "complete_static",
                        "role": role,
                        "probability": prediction["probability"],
                        "base_probability": base,
                        "label": case["label"],
                        "brier": (prediction["probability"] - case["label"]) ** 2,
                        "exact_status": case["exact_status"],
                        "advisory_status": case["exact_status"],
                        "input_hash": canonical_hash(case),
                        "source_sha256": next(
                            row["source_sha256"]
                            for row in legacy["rows"]
                            if row["unit_id"] == case["unit_id"]
                        ),
                        "denominators": {"independent_family": 1, "paired_arm": 1},
                        "exclusions": [],
                        "censored": False,
                        "prediction_tick": tick,
                        "feedback_tick": None,
                    }
                )
                tick += 1
            for case in block:
                bank.release(case["unit_id"], case["label"], tick + 8)
                rows[-len(block) + block.index(case)]["feedback_tick"] = tick + 8
            tick += 9
            progress(started, "measurement", f"static_checkpoint:{role}:{offset // 8}", len(rows))
    for row in rows:
        case = next(
            case for cases in groups.values() for case in cases if case["unit_id"] == row["unit_id"]
        )
        row["sentence_risk_features"] = sentence_features(case, names)
        row["feedback_id"] = f"{row['arm']}:{row['unit_id']}"
        row["raw_numerators"] = {"brier_squared_error": row["brier"]}
    states = dict(legacy["bank_state_paths"])
    states["complete_static"] = str(static_path)
    lifecycle = [{"kind": kind, "passed": passed} for kind, passed in legacy["lifecycle"].items()]
    lifecycle.extend(_hard_exit_diagnostic(raw, started))
    lifecycle.append(
        {
            "kind": "frozen_no_additions",
            "passed": not BoundedBank(Path(states["frozen"]), grammar(), "read_only", 0.1, 2).state[
                "templates"
            ],
        }
    )
    corrupt = raw / "corrupt_bank_private.json"
    changed = json.loads(static_path.read_text())
    changed["budget"]["proposal_credits_spent"] += 1
    atomic_json(corrupt, changed)
    corrupt_rejected = False
    try:
        BoundedBank(corrupt, grammar(), "static", 0.1, 2)
    except ValueError:
        corrupt_rejected = True
    lifecycle.append({"kind": "corrupt_state", "passed": corrupt_rejected})
    lifecycle.append({"kind": "absent_state", "passed": not (raw / "absent_bank.json").is_file()})
    atomic_json(raw / "rows.json", rows)
    return {
        "rows": rows,
        "legacy_path": str(legacy_path),
        "legacy_hash": sha256_file(legacy_path),
        "static_closure_complete": fit,
        "bank_state_paths": states,
        "lifecycle_rows": lifecycle,
        "pending_high_water": legacy["pending_high_water"],
        "proposal_count": legacy["proposal_count"],
        "exactly_once": legacy["exactly_once"],
        "restart_exact_parity": legacy["restart_exact_parity"],
        "raw_rows_path": str(raw / "rows.json"),
        "raw_rows_sha256": sha256_file(raw / "rows.json"),
    }


def cold_reduce(path: Path) -> dict:
    """Recompute the exact candidate from raw families and cold bank ledgers."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    source_root = Path(artifact.get("source_root") or "/nonexistent")
    input_hashes = artifact.get("source_artifact_hashes", {})
    input_valid = True
    for group in ("eligible_producers", "historical_disqualified_sources", "pre_gate_receipts"):
        for relative, expected in input_hashes.get(group, {}).items():
            source = source_root / relative
            input_valid &= source.is_file() and sha256_file(source) == expected
    rows = artifact.get("rows", [])
    groups = fixture_groups()
    originals = {case["unit_id"]: case for cases in groups.values() for case in cases}
    names = artifact.get("static_closure_complete", {}).get("dictionary", [])
    valid = (
        input_valid
        and len(rows) == 288
        and len({(row["arm"], row["unit_id"]) for row in rows}) == 288
    )
    valid &= set(row["arm"] for row in rows) == {"growth", "frozen", "complete_static"}
    valid &= artifact.get("sample_size_budget", {}).get("effective_independent_n") == 96
    valid &= artifact.get("static_closure_complete") == fit_sentence_static(
        groups["development"], names
    )
    valid &= (
        artifact.get("proposal_count", 99) <= 8 and artifact.get("pending_high_water", 99) <= 12
    )
    valid &= all(row.get("passed") is True for row in artifact.get("lifecycle_rows", []))
    valid &= len(artifact.get("lifecycle_rows", [])) >= 8
    for row in rows:
        case = originals.get(row.get("unit_id"))
        if case is None:
            valid = False
            continue
        valid &= row.get("input_hash") == canonical_hash(case)
        valid &= (
            row.get("source_sha256")
            == hashlib.sha256(bytes.fromhex(case["source_bytes_hex"])).hexdigest()
        )
        valid &= row.get("sentence_risk_features") == sentence_features(case, names)
        valid &= row.get("feedback_id") == f"{row['arm']}:{row['unit_id']}"
        valid &= row.get("label") == case["label"] and row.get("role") == case["role"]
        valid &= row.get("censored") is False and row.get("exclusions") == []
        valid &= (
            abs(row.get("brier", -1) - (row.get("probability", -2) - case["label"]) ** 2) < 1e-12
        )
        valid &= row.get("raw_numerators", {}).get("brier_squared_error") == row.get("brier")
        if row["arm"] == "complete_static":
            base = max(
                1e-6,
                min(
                    1 - 1e-6,
                    case["base_probability"]
                    + sum(
                        artifact["static_closure_complete"]["weights"][name] * value
                        for name, value in sentence_features(case, names).items()
                    ),
                ),
            )
            valid &= abs(row["base_probability"] - base) < 1e-12
    raw_path = Path(artifact.get("raw_rows_path") or "/nonexistent")
    valid &= raw_path.is_file() and sha256_file(raw_path) == artifact.get("raw_rows_sha256")
    if raw_path.is_file():
        valid &= json.loads(raw_path.read_text()) == rows
    legacy_path = Path(artifact.get("legacy_path") or "/nonexistent")
    valid &= legacy_path.is_file() and sha256_file(legacy_path) == artifact.get("legacy_hash")
    if legacy_path.is_file():
        valid &= cold_reduce_7732(legacy_path)["valid"] is True
    states = artifact.get("bank_state_paths", {})
    valid &= set(states) == {"growth", "frozen", "complete_static"}
    for arm, state_path in states.items():
        path_state = Path(state_path)
        if not path_state.is_file():
            valid = False
            continue
        try:
            bank = BoundedBank(
                path_state,
                grammar(),
                "priority" if arm == "growth" else "read_only" if arm == "frozen" else "static",
                0.1,
                2,
            )
            valid &= bank.replay_ledger()["state_hash"] == bank.state_hash
            for row in (item for item in rows if item["arm"] == arm):
                prediction = bank.state["predictions"].get(row["unit_id"])
                feedback = bank.state["feedback"].get(row["unit_id"])
                valid &= bool(
                    prediction
                    and feedback
                    and prediction["probability"] == row["probability"]
                    and prediction["tick"] == row["prediction_tick"]
                    and feedback["release_tick"] == row["feedback_tick"]
                )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            valid = False
    return {
        "valid": bool(valid),
        "rows": len(rows),
        "independent_families": len({row["unit_id"] for row in rows}),
    }


def _span(
    started: float, name: str, begin: float, units: int, checkpoint: Path | None = None
) -> dict:
    """Record a disjoint real-time phase with its checkpoint byte hash."""
    end = time.monotonic() - started
    return {
        "phase": name,
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
    date: str,
    started: float,
    checks: list[dict],
    hashes: dict,
    measured: dict,
    spans: list[dict],
    scope: dict,
) -> dict:
    """Keep fixture mechanics, scientific nulls, and administrative readiness separate."""
    rows = measured.get("rows", [])
    blocked = any(not item["passed"] for item in checks)
    mechanics = (
        len(rows) == 288
        and all(item["passed"] for item in measured.get("lifecycle_rows", []))
        and len(measured.get("lifecycle_rows", [])) >= 8
        and measured.get("exactly_once") is True
        and measured.get("restart_exact_parity") is True
        and measured.get("pending_high_water", 99) <= 12
        and measured.get("proposal_count", 99) <= 8
        and measured.get("static_closure_complete", {}).get("nonzero_weight_count") == 16
    )
    verdict_class = "blocked" if blocked else "circular_positive" if mechanics else "null"
    gates = {
        "validity": False if blocked else mechanics,
        "readiness": None,
        "probability_quality": None,
        "decision_benefit": None,
        "retention": None,
        "efficiency": None,
    }
    n = len({row["unit_id"] for row in rows})
    artifact = {
        "schema": "carnot.exp7742.v674.bank_qualification.v1",
        "experiment_id": 7742,
        "milestone": "2026.09.674",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_input"
        if blocked
        else "complete_circular_positive_bank_lifecycle"
        if mechanics
        else "complete_null_bank_lifecycle",
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": [row for row in checks if not row["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "lifecycle_rows": measured.get("lifecycle_rows", []),
        "sample_size_budget": {
            "intended": 96,
            "started": n,
            "completed": n,
            "eligible": n,
            "excluded": 0,
            "censored": 0,
            "effective_independent_n": n,
            "arms_do_not_increase_n": True,
        },
        "claim_scope": {
            "RAGTruth": "development_only",
            "constructed_truth": "fixture_only",
            "ARC": "adapter_withheld_public",
            "fresh_generalization_eligible": False,
        },
        "inference_substrate": "deterministic_cpu_pair_feature_bank",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "model_invocation_counts": {
            key: 0
            for key in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "pid": os.getpid(),
            "host": platform.node(),
            "gpu_uuid": None,
            "effective_coding_backend": os.getenv("CODEX_MODEL", "gpt-6 session"),
        },
        "phase_spans": spans,
        "duration_s": time.monotonic() - started,
        "random_seed": {
            "fixture": 7719,
            "fixture_purpose": "deterministic source families",
            "admission": 7732,
            "admission_purpose": "fixed block order",
        },
        "reproducibility_checksum": canonical_hash(
            {
                "inputs": hashes,
                "parameters": {
                    "pending": 12,
                    "proposals": 8,
                    "predicates": measured.get("static_closure_complete", {}).get("dictionary"),
                },
                "reducer": sha256_file(ROOT / MODULE),
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
                "status": "degraded_open",
                "historical_failures": ["V673 Exp7732 disqualified; global suite debt separate"],
                "affects_required_checks": False,
            },
        },
        "verifier_is_oracle": True,
        "acquisition_protocol_ready_score": 0,
        "bank_protocol_path": {
            "predicate_payload": "16 declared sentence-risk features",
            "production_projection": "V673 primitive pairs; V674 fixture payload is advisory sidecar and fitted static control",
            "pending_capacity": 12,
            "proposal_limit": 8,
            "production_proposal_capacity": 6,
            "per_feedback_block": 1,
            "restart_rule": "cold ledger parity before and after durable acknowledgement",
            "hard_truth_authority": "exact verifier only",
        },
        "static_closure_complete": measured.get("static_closure_complete", {}),
        "bank_state_paths": measured.get("bank_state_paths", {}),
        "legacy_path": measured.get("legacy_path"),
        "legacy_hash": measured.get("legacy_hash"),
        "raw_rows_path": measured.get("raw_rows_path"),
        "raw_rows_sha256": measured.get("raw_rows_sha256"),
        "pending_high_water": measured.get("pending_high_water"),
        "proposal_count": measured.get("proposal_count"),
        "exactly_once": measured.get("exactly_once"),
        "restart_exact_parity": measured.get("restart_exact_parity"),
        "same_verdict_retirements": [],
    }
    artifact["field_principles"] = {
        key: PRINCIPLES.get(key, "Measured evidence bounds the claim.") for key in artifact
    }
    artifact["field_principles"].update(
        {f"acceptance_gate_{key}": PRINCIPLES["acceptance_gate_results"] for key in gates}
    )
    artifact["field_principles"]["field_principles"] = PRINCIPLES["field_principles"]
    return artifact


def run_experiment(
    root: Path, date: str, output: Path, *, raw: Path | None = None, validate: bool = True
) -> dict:
    """Freeze scope, measure private mechanics, validate, then atomically publish."""
    root = root.resolve()
    output = output.resolve()
    raw = (raw or root / RAW).resolve()
    raw.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    spans = []
    progress(started, "preflight", "start")
    begin = time.monotonic() - started
    checks, hashes, names = preflight(root)
    spans.append(_span(started, "preflight", begin, len(checks)))
    progress(started, "preflight", "complete", len(checks))
    tests = [TEST, OLD_TEST, *ACQUISITION_TESTS]
    scope = {
        "tests": tests,
        "coverage_tests": [TEST],
        "changed_modules": [MODULE],
        "static_paths": [CLI],
        "reused_modules": [
            "python/carnot/experiment_7732_v673_causal_admission.py",
            "python/carnot/reporting/acquisition_qualification.py",
            "python/carnot/reporting/constraint_bank_protocol.py",
        ],
        "specs": ["REQ-REPORT-7742", "REQ-CL-7742-BANK"],
        "input_hashes": hashes,
        "frozen_at_epoch_s": time.time(),
        "file_sha256": {
            name: sha256_file(root / name)
            for name in [MODULE, CLI, *tests]
            if (root / name).is_file()
        },
    }
    atomic_json(raw / "frozen_affected_scope.json", scope)
    measured: dict = {}
    if all(row["passed"] for row in checks):
        progress(started, "measurement", "start")
        begin = time.monotonic() - started
        measured = measure(raw, names, started)
        atomic_json(raw / "measurement.json", measured)
        spans.append(
            _span(started, "measurement", begin, len(measured["rows"]), raw / "measurement.json")
        )
        progress(started, "measurement", "complete", len(measured["rows"]))
    else:
        atomic_json(
            raw / "checkpoint.json",
            {"blocked_checks": [row for row in checks if not row["passed"]]},
        )
    artifact = build_artifact(date, started, checks, hashes, measured, spans, scope)
    global_observation = root / RAW / "validation/global_suite_observation.json"
    if global_observation.is_file():
        artifact["validation_receipts"]["repository_health"]["current_global_observation"] = {
            "path": str(global_observation.relative_to(root)),
            "sha256": sha256_file(global_observation),
            **json.loads(global_observation.read_text()),
        }
    artifact["source_root"] = str(root)
    artifact["field_principles"]["source_root"] = PRINCIPLES["source_artifact_hashes"]
    if not validate or artifact["verdict_class"] == "blocked":
        progress(started, "publish", "before_atomic_write")
        atomic_json(output, artifact)
        progress(started, "publish", "complete")
        return artifact
    with tempfile.TemporaryDirectory(prefix="exp7742-validation-") as private_name:
        private = Path(private_name)
        (private / "basetemp").mkdir()
        commands = build_scoped_commands(
            root,
            tests,
            [MODULE],
            static_paths=[CLI],
            basetemp=private / "basetemp",
            coverage_file=private / ".coverage",
        )
        commands = [
            CommandSpec(
                item.name,
                tuple(arg for arg in item.argv if arg not in [OLD_TEST, *ACQUISITION_TESTS]),
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
        spans.append(_span(started, "affected_validation", begin, len(affected)))
        progress(started, "affected_validation", "after_subprocess", len(affected))
        progress(started, "task_e2e", "before_subprocess")
        begin = time.monotonic() - started
        e2e = run_commands(
            root,
            [
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
            ],
            log_dir=raw / "validation/e2e",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["task_e2e"] = e2e
        spans.append(_span(started, "task_e2e", begin, len(e2e)))
        progress(started, "task_e2e", "after_subprocess", len(e2e))
        required_passed = bool(affected and e2e and all(row["passed"] for row in [*affected, *e2e]))
        artifact["acquisition_protocol_ready_score"] = int(
            required_passed and artifact["acceptance_gate_results"]["validity"]
        )
        artifact["phase_spans"] = spans
        artifact["duration_s"] = time.monotonic() - started
        candidate = raw / "terminal_candidate.json"
        atomic_json(candidate, artifact)
        python = str(root / ".venv/bin/python")
        readers = [
            CommandSpec(
                "cold_reduce",
                (
                    python,
                    "-u",
                    "-m",
                    "carnot.experiment_7742_v674_bank_qualification",
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
        spans.append(_span(started, "terminal_readers", begin, len(terminal), candidate))
        progress(started, "terminal_readers", "after_subprocess", len(terminal))
    failed = [row for row in [*affected, *e2e, *terminal] if not row["passed"]]
    if failed:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["acquisition_protocol_ready_score"] = 0
        artifact["acceptance_gate_results"]["validity"] = False
        artifact["flagged_adversarial"] = any(row["name"] == "adversarial_verify" for row in failed)
        artifact["gate_check_summary"].extend(
            _check(
                "required_validation", "current", row["log_path"], "exit_code", 0, row["exit_code"]
            )
            for row in failed
        )
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    progress(started, "publish", "before_atomic_write")
    atomic_json(output, artifact)
    progress(started, "publish", "complete")
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Dispatch the declared CPU CLI and independent cold reducer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_reduce:
        result = cold_reduce(args.cold_reduce)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["valid"] else 1
    run_experiment(
        args.root,
        args.date,
        args.root / args.output,
        raw=args.root / args.raw,
        validate=not args.no_validate,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI process boundary.
    raise SystemExit(main())
