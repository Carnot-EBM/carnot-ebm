"""Qualify fixture optimization and durable queries for REQ-VERIFY-7811."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot import experiment_7760_v675_online_runner as online
from carnot import experiment_7797_v678_training_runtime as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import evidence_views, training_qualification as qualification
from carnot.verify import training_runtime as runtime
from carnot.verify import source_alignment as alignment

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7811_v679_training_runtime"
OUTPUT = ROOT / "results/experiment_7811_v679_training_runtime.json"
SCOPE = RAW / "frozen_validation_scope.json"
SEEDS = (67815, 67816, 67817)
ARMS = tuple(evidence_views.ARMS)
HISTORICAL_7797_SHA256 = "sha256:d59ab0b88dabe36938f54854c24b134e97c45d493ce062b3e7b0ae5db6bb8012"
MODEL_SPECS: list[dict[str, Any]] = []
PRINCIPLES = {
    **prior.PRINCIPLES,
    "validation_command_manifest_path": "A command scope must be reviewable before dispatch.",
    "validation_command_manifest_sha256": "The frozen command bytes bind the attempt.",
    "observed_child_commands": "Dispatch must equal the manifest exactly.",
    "repository_health": "A broad diagnostic retains its real exit.",
    "historical_receipt_byte_audit": "Old receipt bytes and later writes remain distinguishable.",
    "reproducibility_inputs": "The complete code and input closure binds replay.",
}


def progress(start: float, phase: str, event: str, units: int) -> None:
    """Show elapsed time and completed work at each boundary."""
    print(
        f"[exp7811] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed_units={units}",
        flush=True,
    )


fixture_records = prior.fixture_records
reject_nonfinite = prior.reject_nonfinite


def validate_roles(records: list[dict[str, Any]]) -> None:
    """Keep each fixture family in its registered fit, tune or evaluation role."""
    prior.validate_roles(records)
    registered = {row["family"]: row["role"] for row in fixture_records()}
    if {row["family"]: row["role"] for row in records} != registered:
        raise ValueError("role_poisoning")


def _init_params(arm: str, seed: int) -> dict[str, Any]:
    """Seed the same bounded shapes without changing the shared seed registry."""
    if seed not in SEEDS or arm not in runtime.ARMS:
        raise ValueError("unregistered arm or seed")
    rng = np.random.default_rng(seed)
    if arm == "mlp_local":
        return {
            "w1": jnp.asarray(rng.normal(0, 0.02, (alignment.FEATURE_DIM, 16))),
            "b1": jnp.zeros(16),
            "w2": jnp.asarray(rng.normal(0, 0.02, (16, 2))),
            "b2": jnp.zeros(2),
        }
    return {"w": jnp.asarray(rng.normal(0, 0.02, (alignment.FEATURE_DIM, 2))), "b": jnp.zeros(2)}


def _param_hash(params: dict[str, Any]) -> str:
    """Bind every parameter, including hidden layers, to the receipt."""
    return canonical_hash(
        {key: np.asarray(value).tolist() for key, value in sorted(params.items())}
    )


def protocols(names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze both recipes without importing a historical result as evidence."""
    training = prior.training_protocol(names)
    training["schema"] = "exp7811-training-protocol-v1"
    training["seeds"] = list(SEEDS)
    online_protocol = prior.online_protocol(names)
    online_protocol["schema"] = "exp7811-online-protocol-v1"
    return training, online_protocol


def failed_operand(
    upstream: str, path: str, field: str, expected: Any, observed: Any, digest: str | None = None
) -> dict[str, Any]:
    """Keep the literal operand that failed its declared gate."""
    return {
        "upstream_id": upstream,
        "artifact_path": path,
        "artifact_hash": digest,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
    }


def preflight(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Check source custody, schema, code, scope and host before optimization."""
    checks, _, names = online.preflight(root)
    sources = []
    for relative, upstream, expected_hash in (
        (online.SOURCE, "7742", None),
        (online.MANIFEST, "7740", None),
        (
            "results/experiment_7784_v677_training_runtime.json",
            "7784",
            "sha256:974ec40d734bf678cafeb7370605176ccc7c812075a19b18b9f2fd12523793e6",
        ),
        ("results/experiment_7797_v678_training_runtime.json", "7797", HISTORICAL_7797_SHA256),
    ):
        path = root / relative
        value = json.loads(path.read_text()) if path.is_file() else {}
        digest = sha256_file(path) if path.is_file() else None
        sources.append(
            {
                "upstream_id": upstream,
                "artifact_path": relative,
                "artifact_hash": digest,
                "run_date": value.get("run_date"),
                "imported_fields": {
                    key: value.get(key)
                    for key in ("experiment_id", "verdict_class", "flagged_adversarial")
                },
                "eligibility": upstream == "7742" and path.is_file(),
                "source_kind": (
                    "science_producer"
                    if upstream == "7742"
                    else "conductor_pre_gate"
                    if upstream == "7740"
                    else "historical_context"
                ),
            }
        )
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
        if expected_hash and digest is not None:
            checks.append(
                {
                    "upstream_id": upstream,
                    "artifact_path": relative,
                    "artifact_sha256": digest,
                    "field": "sha256",
                    "operator": "==",
                    "expected": expected_hash,
                    "observed": digest,
                    "passed": digest == expected_hash,
                }
            )
            checks.append(
                {
                    "upstream_id": upstream,
                    "artifact_path": relative,
                    "artifact_sha256": digest,
                    "field": "verdict_class",
                    "operator": "==",
                    "expected": "disqualified",
                    "observed": value.get("verdict_class"),
                    "passed": value.get("verdict_class") == "disqualified",
                }
            )
    scope = root / "results/raw/experiment_7811_v679_training_runtime/frozen_validation_scope.json"
    checks.append(
        {
            "upstream_id": "exp7811",
            "artifact_path": str(scope),
            "artifact_sha256": sha256_file(scope) if scope.is_file() else None,
            "field": "exists",
            "operator": "==",
            "expected": True,
            "observed": scope.is_file(),
            "passed": scope.is_file(),
        }
    )
    return checks, sources, names


def seal_log(private_log: Path, durable_root: Path, name: str) -> dict[str, str]:
    """Copy a closed child log once into an attempt-specific byte-addressed path."""
    digest = sha256_file(private_log)
    target = durable_root / name / f"{digest.removeprefix('sha256:')}.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    with private_log.open("rb") as source, target.open("xb") as destination:
        shutil.copyfileobj(source, destination)
        destination.flush()
    if sha256_file(target) != digest:
        raise ValueError("log_sha256_mismatch_after_copy")
    return {"log_path": str(target), "log_sha256": digest}


def read_sealed_log(receipt: dict[str, Any]) -> bytes:
    """Reject any later byte change before trusting the child result."""
    path = Path(receipt["log_path"])
    if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
        raise ValueError("log_sha256_mismatch")
    return path.read_bytes()


def historical_receipt_byte_audit(root: Path) -> list[dict[str, Any]]:
    """Show old claimed hashes beside current bytes and identify the reused writer."""
    artifact = json.loads((root / "results/experiment_7797_v678_training_runtime.json").read_text())
    rows = []
    for receipt in artifact["validation_receipts"]["commands"]:
        path = root / receipt["log_path"]
        current = path.read_bytes() if path.is_file() else None
        observed = sha256_file(path) if current is not None else None
        if observed == receipt["log_sha256"]:
            continue
        rows.append(
            {
                "name": receipt["name"],
                "artifact_path": receipt["log_path"],
                "original_receipt_sha256": receipt["log_sha256"],
                "original_bytes_available": False,
                "current_sha256": observed,
                "current_bytes_hex": current.hex() if current is not None else None,
                "writer_code_path": "experiment_7797_v678_training_runtime._one_command -> experiment_7303_validation_scope.run_commands",
                "writer_path_reason": "run_commands writes 00_<name>.log on each one-command call",
            }
        )
    return rows


def _fit_head(
    arm: str,
    train: dict[str, Any],
    tune: dict[str, Any],
    seed: int,
    mode: str,
    initial: dict[str, Any],
) -> dict[str, Any]:
    """Use the registered loss and dual rule with this task's frozen seeds."""
    if seed not in SEEDS or runtime.parameter_count(initial, int(mode == "constrained") * 2) > 4096:
        raise ValueError("unregistered training budget")
    params = initial
    initial_hash = _param_hash(params)
    duals = (0.0, 0.0)
    curve = []
    gradient_fn = jax.value_and_grad(lambda p, d: runtime.loss(p, train, arm, mode, d))
    for epoch in range(2):
        value, gradient = gradient_fn(params, duals)
        norm = float(jnp.sqrt(sum(jnp.sum(g * g) for g in gradient.values())))
        if not np.isfinite(float(value)) or not np.isfinite(norm):
            raise ValueError("nonfinite training loss or gradient")
        params = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, gradient)
        if mode == "constrained":
            observed = runtime.constraints(params, train, arm)
            duals = runtime.dual_step(duals, (float(observed[0]), float(observed[1])))
        curve.append(
            {"epoch": epoch + 1, "loss": float(value), "gradient_norm": norm, "dual": list(duals)}
        )
    return {
        "arm": arm,
        "mode": mode,
        "seed": seed,
        "learning_rate": 0.01,
        "params": params,
        "curve": curve,
        "initial_hash": initial_hash,
        "final_hash": _param_hash(params),
        "duals": list(duals),
        "tune_nll": runtime.response_nll(params, tune, arm, mode != "canonical"),
    }


def fit_one(
    arm: str, seed: int, records: list[dict[str, Any]], names: list[str], folder: Path
) -> dict[str, Any]:
    """Train, save, reload and score one independent fixture arm and seed."""
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
    initial = _init_params(head_arm, seed)
    static = arm == "complete_static_constrained_set"
    if static:
        initial = {**initial, "w": jnp.pad(initial["w"], ((0, 16), (0, 0)))}
    progress(started, "fit", "before_benchmark", 0)
    fit = _fit_head(head_arm, batches["fit"], batches["tune"], seed, config["mode"], initial)
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


def _base_record(
    date: str, checks: list[dict[str, Any]], sources: list[dict[str, Any]], duration: float
) -> dict[str, Any]:
    """Keep a complete terminal schema even when a producer is unavailable."""
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
    blocked = bool(failed)
    rows = (
        [
            {
                "family": item["family"],
                "arm": arm,
                "seed": seed,
                "status": "unstarted_external_precondition",
                "excluded": True,
                "censored": False,
                "raw_path": None,
            }
            for item in fixture_records()
            for arm in ARMS
            for seed in SEEDS
        ]
        if blocked
        else []
    )
    return {
        "experiment_id": 7811,
        "milestone": "2026.09.679",
        "run_date": date,
        "honest_verdict": "complete_blocked_external_precondition"
        if blocked
        else "complete_disqualified_required_validation",
        "verdict_class": "blocked" if blocked else "disqualified",
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
            "eligible": 0 if blocked else 6,
            "started": 0,
            "completed": 0,
            "excluded": 6 if blocked else 0,
            "censored": 0,
            "effective_independent_n": 0,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {"checks": checks},
        "validation_receipts": {},
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
        "training_protocol_path": None,
        "online_protocol_path": None,
        "coverage_shard_rows": [],
        "validation_command_manifest_path": None,
        "validation_command_manifest_sha256": None,
        "observed_child_commands": [],
        "repository_health": None,
    }


def measure(date: str, raw: Path) -> dict[str, Any]:
    """Measure new fixture heads and bank state without opening natural labels."""
    started = time.monotonic()
    progress(started, "preflight", "before", 0)
    checks, sources, names = preflight(ROOT)
    progress(started, "preflight", "after", len(checks))
    record = _base_record(date, checks, sources, time.monotonic() - started)
    if record["verdict_class"] == "blocked":
        return record
    raw.mkdir(parents=True, exist_ok=True)
    records = fixture_records()
    validate_roles(records)
    training, online_protocol = protocols(names)
    training_path, online_path = raw / "training_protocol.json", raw / "online_protocol.json"
    fixture_path = raw / "fixture_records.json"
    atomic_json(training_path, training)
    atomic_json(online_path, online_protocol)
    atomic_json(fixture_path, records)
    record["training_protocol_path"] = str(training_path)
    record["online_protocol_path"] = str(online_path)
    training_rows, rows = [], []
    phase = time.monotonic()
    for arm in ARMS:
        for seed in SEEDS:
            progress(started, "numerical", f"before_benchmark_{arm}_{seed}", len(training_rows))
            result = fit_one(arm, seed, records, names, raw / "heads")
            rows.extend(result.pop("rows"))
            training_rows.append(result)
            progress(started, "numerical", f"after_benchmark_{arm}_{seed}", len(training_rows))
    record["phase_spans"].append(
        {
            "phase": "numerical",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(training_rows),
        }
    )
    phase = time.monotonic()
    progress(started, "online", "before_benchmark", 0)
    online_result = prior.exercise_online(raw / "online", names)
    progress(started, "online", "after_benchmark", len(online_result["query_rows"]))
    record["phase_spans"].append(
        {
            "phase": "online",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(online_result["query_rows"]),
        }
    )
    rows_path, training_rows_path = raw / "rows.json", raw / "training_rows.json"
    atomic_json(rows_path, rows)
    atomic_json(training_rows_path, training_rows)
    valid = (
        all(
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
        and online_result["valid"]
    )
    if not valid:
        record["gate_check_summary"].append(
            failed_operand(
                "exp7811",
                str(training_rows_path),
                "fixture_valid",
                True,
                False,
                sha256_file(training_rows_path),
            )
        )
    record["rows"] = rows
    record["acceptance_gate_results"]["validity"] = valid
    record["sample_size_budget"].update(
        {"started": 6, "completed": 6, "effective_independent_n": 6}
    )
    record["fixture_training_rows"] = training_rows
    record["online_fixture"] = online_result
    record["raw_paths"] = {
        "rows": str(rows_path),
        "training": str(training_rows_path),
        "fixtures": str(fixture_path),
    }
    record["reproducibility_inputs"] = {
        "code": sha256_file(Path(__file__)),
        "fixtures": sha256_file(fixture_path),
        "training_protocol": sha256_file(training_path),
        "online_protocol": sha256_file(online_path),
        "roles": training["roles"],
        "seeds": list(SEEDS),
        "sources": {item["artifact_path"]: item["artifact_hash"] for item in sources},
    }
    record["reproducibility_checksum"] = canonical_hash(record["reproducibility_inputs"])
    record["duration_s"] = time.monotonic() - started
    return record


def reduce_validation(
    scope: dict[str, Any], manifest: dict[str, Any], receipts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Close both gates if command identity, coverage or sealed bytes drift."""
    failures = []
    expected = manifest["commands"]
    actual_names = [row.get("name") for row in receipts]
    for command in expected:
        name = command["name"]
        matching = [row for row in receipts if row.get("name") == name]
        if len(matching) != 1:
            failures.append(
                failed_operand(
                    "exp7811", "validation_command_manifest.json", name + ".count", 1, len(matching)
                )
            )
            continue
        row = matching[0]
        for key in ("argv", "classification"):
            if row.get(key) != command[key]:
                failures.append(
                    failed_operand(
                        "exp7811",
                        "validation_command_manifest.json",
                        name + "." + key,
                        command[key],
                        row.get(key),
                    )
                )
        try:
            read_sealed_log(row)
        except (KeyError, ValueError, OSError):
            failures.append(
                failed_operand(
                    "exp7811",
                    row.get("log_path", ""),
                    name + ".log_sha256",
                    row.get("log_sha256"),
                    sha256_file(Path(row["log_path"]))
                    if Path(row.get("log_path", "")).is_file()
                    else None,
                )
            )
        if command["classification"] == "required" and (
            row.get("exit_code") != 0 or row.get("timed_out")
        ):
            failures.append(
                failed_operand(
                    "exp7811",
                    row.get("log_path", ""),
                    name + ".exit_code",
                    0,
                    row.get("exit_code"),
                    row.get("log_sha256"),
                )
            )
    for row in receipts:
        if row.get("name") not in {command["name"] for command in expected}:
            failures.append(
                failed_operand(
                    "exp7811", row.get("log_path", ""), "undeclared_child", False, row.get("name")
                )
            )
    affected = next((row for row in receipts if row.get("name") == "affected_pytest"), {})
    for test in (*scope["direct_tests"], *scope["transitive_tests"]):
        if test not in affected.get("argv", []):
            failures.append(
                failed_operand(
                    "exp7811",
                    "validation_command_manifest.json",
                    "affected_pytest.consumer",
                    test,
                    None,
                )
            )
    coverage = next((row for row in receipts if row.get("name") == "coverage_report"), {})
    if coverage.get("coverage_percent") != 100:
        failures.append(
            failed_operand(
                "exp7811",
                coverage.get("log_path", ""),
                "coverage_percent",
                100,
                coverage.get("coverage_percent"),
            )
        )
    ready = int(not failures and actual_names == [row["name"] for row in expected])
    if not ready and not failures:
        failures.append(
            failed_operand(
                "exp7811",
                "validation_command_manifest.json",
                "command_order",
                [row["name"] for row in expected],
                actual_names,
            )
        )
    return {
        "gate_check_summary": failures,
        "verdict_class": "circular_positive" if ready else "disqualified",
        "training_runtime_ready_score": ready,
        "online_runtime_ready_score": ready,
    }


def cold_reduce(candidate: Path) -> dict[str, Any]:
    """Reopen raw rows and prior sealed receipts in a fresh process."""
    value = json.loads(candidate.read_text())
    rows_path = Path(value["raw_paths"]["rows"])
    rows = json.loads(rows_path.read_text())
    if rows != value["rows"]:
        raise ValueError("raw_rows_invalid")
    for receipt in value.get("validation_receipts", {}).get("commands", []):
        read_sealed_log(receipt)
    return {"valid": True, "row_count": len(rows), "raw_sha256": sha256_file(rows_path)}
