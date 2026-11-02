"""Bounded numerical and durable-bank qualification for REQ-VERIFY-7797.

All labels in this module are explicit synthetic fixture labels. Natural source
families and their target labels are never loaded by this experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from carnot import experiment_7760_v675_online_runner as online
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import evidence_views, training_qualification as qualification
from carnot.verify import training_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7797_v678_training_runtime"
OUTPUT = ROOT / "results/experiment_7797_v678_training_runtime.json"
SCOPE = RAW / "frozen_scope.json"
SEEDS = (67801, 67802, 67803)
ARMS = tuple(evidence_views.ARMS)
MODEL_SPECS: list[dict[str, Any]] = []
PRINCIPLES = {
    "experiment_id": "Each result has one owner.",
    "milestone": "Bind the current research cycle.",
    "run_date": "Bind the measured run date.",
    "honest_verdict": "External incompleteness cannot be fixed by retrying owned work.",
    "verdict_class": "Claim strength travels with the record.",
    "flagged_adversarial": "Invalid evidence cannot open a gate.",
    "gate_check_summary": "Missing evidence differs from a scientific null.",
    "rows": "Recompute every comparison from its units.",
    "acceptance_gate_results": "A working fixture proves no scientific gain.",
    "duration_s": "Duration reflects actual work.",
    "phase_spans": "Phase duration reflects actual work.",
    "random_seed": "Replay requires identical seeds.",
    "reproducibility_checksum": "Replay requires identical inputs.",
    "sample_size_budget": "Views and seeds are not new families.",
    "source_artifact_hashes": "Old files cannot replace missing current producers.",
    "preconditions_checked": "Cheap failures precede compute.",
    "validation_receipts": "Every required check must pass.",
    "verifier_is_oracle": "Fixture success is circular evidence.",
    "claim_scope": "No natural gain follows from fixture truth.",
    "inference_substrate": "Floors follow invoked work.",
    "inference_substrate_class": "No model is loaded.",
    "MODEL_SPECS": "Citing a model is not invoking it.",
    "model_specs": "Citing a model is not invoking it.",
    "model_invocation_counts": "Count actual calls and loaded files.",
    "training_runtime_ready_score": "Natural fitting needs working optimization.",
    "online_runtime_ready_score": "Restart correctness precedes retention claims.",
    "training_protocol_path": "Validation must be replayable.",
    "online_protocol_path": "Validation must be replayable.",
    "coverage_shard_rows": "Preserve shard exits and logs.",
}


def progress(start: float, phase: str, event: str, units: int) -> None:
    """Flush a measured phase or completed unit boundary."""
    print(
        f"[exp7797] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


def fixture_records() -> list[dict[str, Any]]:
    """Return six fixed source families with both positive and negative controls."""
    pairs = (
        ("Alpha", "12", "13", "fit"),
        ("Beta", "30", "31", "tune"),
        ("Gamma", "44", "45", "evaluation"),
    )
    rows = []
    for name, correct, wrong, role in pairs:
        source = f"{name} is {correct}."
        for label, answer in ((0, source), (1, f"{name} is {wrong}.")):
            rows.append(
                {
                    "id": f"{name.lower()}-{label}",
                    "family": f"{name.lower()}-{label}",
                    "source": source,
                    "answer": answer,
                    "label": label,
                    "known": [1 - label],
                    "role": role,
                }
            )
    return rows


def validate_roles(records: list[dict[str, Any]]) -> None:
    """Reject family overlap and any label-bearing feature payload."""
    seen: dict[str, str] = {}
    for row in records:
        family, role = row["family"], row["role"]
        if family in seen or role not in {"fit", "tune", "evaluation"}:
            raise ValueError("role_poisoning")
        seen[family] = role
        if any(key.startswith("view") or key == "features" for key in row):
            raise ValueError("feature_label_poisoning")
    if {row["role"] for row in records} != {"fit", "tune", "evaluation"}:
        raise ValueError("role_poisoning")


def reject_nonfinite(values: np.ndarray) -> None:
    """Fail before a nonfinite feature can reach the optimizer."""
    if not np.isfinite(values).all():
        raise ValueError("nonfinite_features")


def training_protocol(names: list[str]) -> dict[str, Any]:
    """Freeze the natural recipe and its two-epoch fixture miniature."""
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError("sixteen_predicates_required")
    return {
        "schema": "exp7797-training-protocol-v1",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "miniature_epochs": 2,
        "natural_recipe": {
            "epochs": 16,
            "learning_rate": 0.01,
            "optimizer": "full_batch_gradient_descent",
        },
        "parameter_max": 4096,
        "complete_static_predicates": names,
        "temperature_grid": runtime.temperature_grid(),
        "dual_constraints": {
            "symmetric_kl_half_tolerance": 0.01,
            "second_view_ce_tolerance": 0.70,
            "step": 0.01,
            "clip": [0, 10],
        },
        "roles": {
            "fit": ["alpha-0", "alpha-1"],
            "tune": ["beta-0", "beta-1"],
            "evaluation": ["gamma-0", "gamma-1"],
        },
        "fixture_only": True,
    }


def fit_one(
    arm: str, seed: int, records: list[dict[str, Any]], names: list[str], folder: Path
) -> dict[str, Any]:
    """Train one fixture head with the exact registered optimizer and cold-score it."""
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("unregistered_arm_or_seed")
    started = time.monotonic()
    validate_roles(records)
    folder.mkdir(parents=True, exist_ok=True)
    subsets = {
        role: [row for row in records if row["role"] == role]
        for role in ("fit", "tune", "evaluation")
    }
    batches = {}
    for role, subset in subsets.items():
        progress(started, "features", f"before_generation_{role}", len(batches))
        batch, excluded = qualification.make_batch(subset, arm, names)
        progress(started, "features", f"after_generation_{role}", len(batches) + 1)
        if excluded:
            raise ValueError("unexpected_fixture_exclusion")
        for view in ("a", "b"):
            reject_nonfinite(np.asarray(batch[view]["x"]))
        batches[role] = batch
    config = evidence_views.ARMS[arm]
    head_arm = "response_set" if arm == "response_set" else config["head"]
    initial = runtime.init_params(head_arm, seed)
    static = arm == "complete_static_constrained_set"
    if static:
        initial = {**initial, "w": jnp.pad(initial["w"], ((0, 16), (0, 0)))}
    progress(started, "fit", "before_benchmark", 0)
    fit = runtime.fit(
        head_arm,
        batches["fit"],
        batches["tune"],
        seed,
        0.01,
        2,
        config["mode"],
        initial=initial,
    )
    progress(started, "fit", "after_benchmark", 2)
    fit["temperature"] = runtime.calibrate(
        fit["params"], batches["tune"], head_arm, config["paired"]
    )
    fit["paired"] = config["paired"]
    fit["static_predicates"] = names if static else []
    path = folder / f"{arm}_{seed}.json"
    progress(started, "head", "before_model_save", 0)
    runtime.save(path, fit)
    progress(started, "head", "after_model_save", 1)
    progress(started, "head", "before_model_load", 1)
    loaded = runtime.load(path)
    progress(started, "head", "after_model_load", 2)
    adapter = qualification.NaturalHeadAdapter(loaded)
    warm = qualification.NaturalHeadAdapter(fit)
    rows = []
    reload_equal = True
    for role in ("fit", "tune", "evaluation"):
        for index, record in enumerate(subsets[role]):
            decision = adapter.decide(batches[role], index)
            reload_equal &= decision == warm.decide(batches[role], index)
            rows.append(
                {
                    "family": record["family"],
                    "role": role,
                    "seed": seed,
                    "arm": arm,
                    "label": record["label"],
                    "raw_path": str(path),
                    "excluded": False,
                    "censored": False,
                    **decision,
                }
            )
    support = np.asarray(
        runtime.sentence_support(fit["params"], batches["evaluation"]["a"], head_arm)
    )
    normalization_error = float(np.maximum(0, -support).max() + np.maximum(0, support - 1).max())
    dual_effect = None
    if config["mode"] == "constrained":
        zero = runtime.loss(fit["params"], batches["fit"], head_arm, config["mode"], (0.0, 0.0))
        active = runtime.loss(fit["params"], batches["fit"], head_arm, config["mode"], (1.0, 1.0))
        dual_effect = abs(float(active - zero)) > 1e-9
    count = runtime.parameter_count(fit["params"], 2 if config["mode"] == "constrained" else 0)
    return {
        "arm": arm,
        "seed": seed,
        "mode": config["mode"],
        "head_arm": head_arm,
        "initial_hash": fit["initial_hash"],
        "final_hash": fit["final_hash"],
        "gradient_norm": fit["curve"][-1]["gradient_norm"],
        "gradient_error": runtime.gradient_error(initial, batches["fit"], head_arm),
        "initial_loss": fit["curve"][0]["loss"],
        "final_loss": fit["curve"][-1]["loss"],
        "dual_variables": fit["duals"],
        "dual_effect_nonzero": dual_effect,
        "temperature": fit["temperature"],
        "normalization_error": normalization_error,
        "parameter_count": count,
        "static_predicates": fit["static_predicates"],
        "head_path": str(path),
        "head_hash": sha256_file(path),
        "reload_decision_equal": reload_equal,
        "rows": rows,
    }


def online_protocol(names: list[str]) -> dict[str, Any]:
    """Freeze the two causal commit schedules and bank resources."""
    return {
        "schema": "exp7797-online-protocol-v1",
        "bank_module": online.BANK_MODULE,
        "runner_module": online.MODULE,
        "predicates": names,
        "blocks": 8,
        "queue_capacity": 12,
        "label_delay_blocks": 1,
        "commit_modes": ["next_query", "next_block"],
        "query_immutable_bank_and_model": True,
    }


def exercise_online(folder: Path, names: list[str]) -> dict[str, Any]:
    """Measure bank versions, feedback delay and a cold pending-queue restart."""
    folder = folder / f"run_{time.time_ns()}"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    query_rows: list[dict[str, Any]] = []
    summaries = {}
    restart_parity = True
    duplicate_rejected = False
    for mode in ("next_query", "next_block"):
        state_path, bank_path = folder / mode / "state.json", folder / mode / "bank.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        runner = online.OnlineRunner(state_path, bank_path, names, "adaptive")
        for block in range(8):
            if mode == "next_query" and block:
                runner.release_block(block - 1, block)
            bank_before = canonical_hash(runner.bank.state["templates"])
            model_before = canonical_hash(runner.state["weights"])
            runner.predict_block(block)
            query_rows.append(
                {
                    "mode": mode,
                    "query": block,
                    "bank_before": bank_before,
                    "bank_after": canonical_hash(runner.bank.state["templates"]),
                    "model_before": model_before,
                    "model_after": canonical_hash(runner.state["weights"]),
                    "pending_count": len(runner.state["queue"]),
                    "state_path": str(state_path),
                    "bank_path": str(bank_path),
                }
            )
            if block == 3:
                persisted = json.loads(state_path.read_text())
                runner = online.OnlineRunner(state_path, bank_path, names, "adaptive")
                restart_parity &= all(
                    runner.state[key] == persisted[key]
                    for key in ("queue", "rng_counter", "credits", "weights", "active")
                )
            if mode == "next_block":
                runner.release_block(block, block + 1)
            progress(started, "online_query", mode, len(query_rows))
        if mode == "next_query":
            runner.release_block(7, 8)
        try:
            runner.release_block(7, 8)
        except ValueError as error:
            duplicate_rejected |= str(error) == "credit_reused"
        summaries[mode] = runner.finish()
    overflow_runner = online.OnlineRunner(
        folder / "overflow_state.json", folder / "overflow_bank.json", names, "adaptive"
    )
    overflow_runner.predict_block(0)
    overflow_rejected = False
    try:
        overflow_runner.predict_block(1)
    except ValueError as error:
        overflow_rejected = str(error) == "pending_overflow"
    progress(started, "online_controls", "before_bank_proposal", len(query_rows))
    bank = online.Bank(folder / "rollback_bank.json", online.grammar(), "priority", 0.1, 2)
    features = {name: float(index < 2) for index, name in enumerate(online.grammar()["primitives"])}
    bank.predict("rollback_case", 0, features, 0.4, "unknown", "update", "rollback_case")
    bank.release("rollback_case", 1, 10)
    proposal = bank.propose()
    candidate_compiled = proposal is not None and proposal["pair"] in online.grammar()["pairs"]
    bank.admit(False)
    false_admission_rolled_back = not bank.state["templates"] and any(
        row["kind"] == "rollback" for row in bank.state["ledger"]
    )
    progress(started, "online_controls", "after_bank_proposal", len(query_rows) + 1)
    progress(started, "hard_exit", "before_subprocess", len(query_rows) + 1)
    hard_exit = online.hard_exit_commit_check(folder / "hard_exit", names)
    progress(started, "hard_exit", "after_subprocess", len(query_rows) + 2)
    valid = all(
        (
            restart_parity,
            duplicate_rejected,
            overflow_rejected,
            false_admission_rolled_back,
            candidate_compiled,
            hard_exit,
        )
    )
    valid &= all(
        row["bank_before"] == row["bank_after"] and row["model_before"] == row["model_after"]
        for row in query_rows
    )
    valid &= all(
        summary["credits"] == 8
        and summary["update_calls"] == 96
        and summary["rng_counter"] == 160
        and not summary["queue_hash"] == ""
        for summary in summaries.values()
    )
    return {
        "query_rows": query_rows,
        "summaries": summaries,
        "restart_parity": restart_parity,
        "duplicate_rejected": duplicate_rejected,
        "overflow_rejected": overflow_rejected,
        "false_admission_rolled_back": false_admission_rolled_back,
        "candidate_compiled": candidate_compiled,
        "hard_exit": hard_exit,
        "valid": valid,
    }


def failed_operand(
    upstream: str, path: str, field: str, expected: Any, observed: Any, digest: str | None = None
) -> dict[str, Any]:
    """Keep the literal failed value beside its path and byte hash."""
    return {
        "upstream_id": upstream,
        "artifact_path": path,
        "artifact_hash": digest,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
    }


def reduce_validation(
    scope: dict[str, Any], receipts: list[dict[str, Any]], required: set[str]
) -> dict[str, Any]:
    """Close both readiness gates on any missing, stale or undeclared check."""
    failures = []
    observed_names = [row["name"] for row in receipts]
    for name in sorted(required):
        matches = [row for row in receipts if row["name"] == name]
        if len(matches) != 1:
            failures.append(
                failed_operand("exp7797", "frozen_scope.json", name + ".count", 1, len(matches))
            )
    for row in receipts:
        name = row["name"]
        if name not in required:
            failures.append(
                failed_operand("exp7797", row.get("log_path", ""), "undeclared_child", False, name)
            )
        path = Path(row.get("log_path", ""))
        digest = sha256_file(path) if path.is_file() else None
        if digest != row.get("log_sha256"):
            failures.append(
                failed_operand(
                    "exp7797",
                    str(path),
                    name + ".log_sha256",
                    row.get("log_sha256"),
                    digest,
                    digest,
                )
            )
        if row.get("exit_code") != 0 or row.get("timed_out", False):
            failures.append(
                failed_operand(
                    "exp7797", str(path), name + ".exit_code", 0, row.get("exit_code"), digest
                )
            )
    focused = next((row for row in receipts if row["name"] == "affected_pytest"), {})
    argv = focused.get("command_argv", [])
    for test in (*scope["direct_tests"], *scope["transitive_tests"]):
        if test not in argv:
            failures.append(
                failed_operand(
                    "exp7797", "frozen_scope.json", "affected_pytest.consumer", test, None
                )
            )
    coverage = next((row for row in receipts if row["name"] == "coverage_report"), {})
    if "coverage_report" in required and coverage.get("coverage_percent", 0) != 100:
        failures.append(
            failed_operand(
                "exp7797",
                coverage.get("log_path", ""),
                "coverage_percent",
                100,
                coverage.get("coverage_percent"),
            )
        )
    ready = int(not failures and set(observed_names) == required)
    return {
        "gate_check_summary": failures,
        "verdict_class": "circular_positive" if ready else "disqualified",
        "training_runtime_ready_score": ready,
        "online_runtime_ready_score": ready,
    }


def cold_reduce(candidate: Path) -> dict[str, Any]:
    """Reopen independent raw rows and reject a changed candidate table."""
    value = json.loads(candidate.read_text())
    rows_path = Path(value["raw_paths"]["rows"])
    rows = json.loads(rows_path.read_text())
    if rows != value["rows"]:
        raise ValueError("raw_rows_invalid")
    return {"valid": True, "row_count": len(rows), "raw_sha256": sha256_file(rows_path)}


def preflight(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Check exact producer fields, code bytes, scope and CPU before compute."""
    checks, _, names = online.preflight(root)
    source_rows = []
    for relative, upstream in (
        (online.SOURCE, "7742"),
        (online.MANIFEST, "7740"),
        ("results/experiment_7784_v677_training_runtime.json", "7784"),
    ):
        path = root / relative
        value = json.loads(path.read_text()) if path.is_file() else {}
        digest = sha256_file(path) if path.is_file() else None
        source_rows.append(
            {
                "upstream_id": upstream,
                "artifact_path": relative,
                "artifact_hash": digest,
                "run_date": value.get("run_date"),
                "imported_fields": {
                    key: value.get(key)
                    for key in ("honest_verdict", "verdict_class", "flagged_adversarial")
                },
                "eligibility": upstream != "7784" and path.is_file(),
            }
        )
        if upstream == "7784":
            checks.append(
                {
                    "upstream_id": upstream,
                    "artifact_path": relative,
                    "artifact_sha256": digest,
                    "field": "exists",
                    "operator": "==",
                    "expected": True,
                    "observed": path.is_file(),
                    "passed": path.is_file(),
                }
            )
    scope = root / "results/raw/experiment_7797_v678_training_runtime/frozen_scope.json"
    checks.append(
        {
            "upstream_id": "exp7797",
            "artifact_path": str(scope.relative_to(root)),
            "artifact_sha256": sha256_file(scope) if scope.is_file() else None,
            "field": "exists",
            "operator": "==",
            "expected": True,
            "observed": scope.is_file(),
            "passed": scope.is_file(),
        }
    )
    return checks, source_rows, names


def blocked_record(date: str, checks: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    """Preserve every planned fixture cell when an external input is absent."""
    failed = [
        failed_operand(
            row["upstream_id"],
            row["artifact_path"],
            row["field"],
            row["expected"],
            row["observed"],
            row.get("artifact_sha256"),
        )
        for row in checks
        if not row["passed"]
    ]
    rows = [
        {
            "family": family,
            "arm": arm,
            "seed": seed,
            "status": "unstarted_external_precondition",
            "excluded": True,
            "censored": False,
            "raw_path": None,
        }
        for family in (row["family"] for row in fixture_records())
        for arm in ARMS
        for seed in SEEDS
    ]
    return {
        "experiment_id": 7797,
        "milestone": "2026.09.678",
        "run_date": date,
        "honest_verdict": "complete_blocked_external_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "rows": rows,
        "acceptance_gate_results": {
            "validity": False,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": duration,
        "phase_spans": [
            {"phase": "preflight", "duration_s": duration, "completed_units": len(checks)}
        ],
        "random_seed": list(SEEDS),
        "reproducibility_checksum": canonical_hash(checks),
        "sample_size_budget": {
            "intended": 6,
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "excluded": 6,
            "censored": 0,
            "effective_independent_n": 0,
        },
        "source_artifact_hashes": [],
        "preconditions_checked": {"checks": checks},
        "validation_receipts": {},
        "verifier_is_oracle": True,
        "claim_scope": "unstarted_fixture_external_block",
        "field_principles": PRINCIPLES,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"calls": 0, "tokens": 0, "loaded_files": []},
        "training_runtime_ready_score": 0,
        "online_runtime_ready_score": 0,
        "training_protocol_path": None,
        "online_protocol_path": None,
        "coverage_shard_rows": [],
    }


def measure(date: str, raw: Path = RAW) -> dict[str, Any]:
    """Run only current fixture mechanics and assemble a zero-readiness candidate."""
    started = time.monotonic()
    progress(started, "preflight", "before", 0)
    checks, sources, names = preflight(ROOT)
    progress(started, "preflight", "after", len(checks))
    failed = [row for row in checks if not row["passed"]]
    if failed:
        blocked = blocked_record(date, checks, time.monotonic() - started)
        blocked["source_artifact_hashes"] = sources
        return blocked
    raw.mkdir(parents=True, exist_ok=True)
    records = fixture_records()
    validate_roles(records)
    train_protocol_path = raw / "training_protocol.json"
    online_protocol_path = raw / "online_protocol.json"
    fixtures_path = raw / "fixture_records.json"
    atomic_json(train_protocol_path, training_protocol(names))
    atomic_json(online_protocol_path, online_protocol(names))
    atomic_json(fixtures_path, records)
    spans = [
        {
            "phase": "preflight",
            "duration_s": time.monotonic() - started,
            "completed_units": len(checks),
        }
    ]
    phase = time.monotonic()
    training_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            progress(started, "numerical", f"before_benchmark_{arm}_{seed}", len(training_rows))
            result = fit_one(arm, seed, records, names, raw / "heads")
            rows.extend(result.pop("rows"))
            training_rows.append(result)
            progress(started, "numerical", f"after_benchmark_{arm}_{seed}", len(training_rows))
    spans.append(
        {
            "phase": "numerical",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(training_rows),
        }
    )
    phase = time.monotonic()
    progress(started, "online", "before_benchmark", 0)
    online_result = exercise_online(raw / "online", names)
    progress(started, "online", "after_benchmark", len(online_result["query_rows"]))
    spans.append(
        {
            "phase": "online",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(online_result["query_rows"]),
        }
    )
    rows_path, training_path = raw / "rows.json", raw / "training_rows.json"
    atomic_json(rows_path, rows)
    atomic_json(training_path, training_rows)
    numerical_valid = all(
        row["initial_hash"] != row["final_hash"]
        and row["gradient_norm"] > 0
        and row["gradient_error"] < 1e-4
        and row["normalization_error"] < 1e-8
        and row["reload_decision_equal"]
        and row["parameter_count"] <= 4096
        and np.isfinite(row["final_loss"])
        and (row["dual_effect_nonzero"] if row["mode"] == "constrained" else True)
        for row in training_rows
    )
    failures = []
    if not numerical_valid:
        failures.append(
            failed_operand(
                "exp7797",
                str(training_path),
                "numerical_valid",
                True,
                False,
                sha256_file(training_path),
            )
        )
    if not online_result["valid"]:
        failures.append(failed_operand("exp7797", str(raw / "online"), "online_valid", True, False))
    inputs = {
        "code": sha256_file(Path(__file__)),
        "runtime": sha256_file(ROOT / "python/carnot/verify/training_runtime.py"),
        "fixtures": sha256_file(fixtures_path),
        "roles": training_protocol(names)["roles"],
        "training_protocol": sha256_file(train_protocol_path),
        "online_protocol": sha256_file(online_protocol_path),
        "scope": sha256_file(SCOPE),
        "seeds": list(SEEDS),
        "sources": {row["artifact_path"]: row["artifact_hash"] for row in sources},
    }
    valid = not failures
    return {
        "experiment_id": 7797,
        "milestone": "2026.09.678",
        "run_date": date,
        "honest_verdict": "complete_disqualified_required_validation",
        "verdict_class": "disqualified",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "acceptance_gate_results": {
            "validity": valid,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - started,
        "phase_spans": spans,
        "random_seed": list(SEEDS),
        "reproducibility_checksum": canonical_hash(inputs),
        "reproducibility_inputs": inputs,
        "sample_size_budget": {
            "intended": 6,
            "eligible": 6,
            "started": 6,
            "completed": 6,
            "excluded": 0,
            "censored": 0,
            "effective_independent_n": 6,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {"checks": checks},
        "validation_receipts": {"frozen_affected_scope": json.loads(SCOPE.read_text())},
        "verifier_is_oracle": True,
        "claim_scope": "six_public_fixture_families_only; no natural fitting or held-out benefit",
        "field_principles": PRINCIPLES,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {"calls": 0, "tokens": 0, "loaded_files": []},
        "training_runtime_ready_score": 0,
        "online_runtime_ready_score": 0,
        "training_protocol_path": str(train_protocol_path),
        "online_protocol_path": str(online_protocol_path),
        "coverage_shard_rows": [],
        "fixture_training_rows": training_rows,
        "online_fixture": online_result,
        "raw_paths": {
            "rows": str(rows_path),
            "training": str(training_path),
            "fixtures": str(fixtures_path),
        },
    }
