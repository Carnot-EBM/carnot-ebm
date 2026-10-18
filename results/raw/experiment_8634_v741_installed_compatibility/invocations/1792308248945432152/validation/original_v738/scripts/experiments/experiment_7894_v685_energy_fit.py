"""Fit matched natural energies on qualified exposed families (REQ-VERIFY-7894-V685)."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file  # noqa: E402
from carnot.verify import energy_fit_7894 as custody  # noqa: E402
from carnot.verify import evidence_views, natural_training, training_runtime  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "results/experiment_7892_v685_source_boundary.json"
OUTPUT = ROOT / "results/experiment_7894_v685_energy_fit.json"
RAW = ROOT / "results/raw/experiment_7894_v685_energy_fit"
ARMS = tuple(evidence_views.ARMS)
SEEDS = natural_training.SEEDS
ROLE_BUDGET = {
    "fit": 256,
    "tune": 64,
    "policy_design": 32,
    "calibration_replay": 32,
    "online_update": 96,
    "online_admission": 64,
    "evaluation": 64,
    "retention": 32,
}
OWNED = [
    "python/carnot/verify/energy_fit_7894.py",
    "scripts/experiments/experiment_7894_v685_energy_fit.py",
]
TESTS = [
    "tests/python/test_energy_fit_7894.py",
    "tests/python/test_natural_runtime_7867.py",
    "tests/python/test_natural_runtime_7853.py",
]
START = time.monotonic()


def progress(phase: str, event: str, units: int = 0) -> None:
    """A live phase counter lets an operator distinguish compute from a stall."""
    print(
        f"[exp7894] phase={phase} event={event} elapsed_s={time.monotonic() - START:.3f} "
        f"completed={units}",
        flush=True,
    )


def heartbeat(phase: str, stop: threading.Event, units: int) -> None:
    """JAX may spend minutes compiling; keep its owning process observable."""
    while not stop.wait(45):
        progress(phase, "heartbeat", units)


def base(
    upstream: Path, sources: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep a complete terminal schema even when external evidence blocks work."""
    history = json.loads(upstream.read_text()) if upstream.is_file() else {}
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.is_file() else {}
    retained = [
        *history.get("historical_required_failures", []),
        *previous.get("historical_required_failures", []),
        *(
            {
                "experiment_id": 7894,
                "name": item["name"],
                "exit_code": item["actual_exit"],
                "log_path": item["log_path"],
                "log_sha256": item["log_sha256"],
            }
            for item in previous.get("validation_receipts", [])
            if not item["passed"]
        ),
    ]
    retained = list({(item["name"], item["log_sha256"]): item for item in retained}.values())
    return {
        "experiment_id": 7894,
        "task_id": "exp7894-energy-fit",
        "milestone": "2026.09.685",
        "run_date": "20260929",
        "honest_verdict": "complete_blocked_source_eligibility",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": [],
        "sample_size_budget": {
            "intended": 640,
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "excluded": 0,
            "independent": 0,
        },
        "acceptance_gate_results": {
            "validity": False,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": list(SEEDS),
        "reproducibility_checksum": None,
        "source_artifact_hashes": sources,
        "preconditions_checked": {"failed": failures},
        "resolved_imports": {},
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "historical_required_failures": retained,
        "repository_health": {
            "affects_required_checks": False,
            "historical": history.get("repository_health", {}),
        },
        "verifier_is_oracle": False,
        "claim_scope": "exposed_development",
        "inference_substrate": "deterministic_cpu",
        "inference_substrate_class": "blocked_no_run" if failures else "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (small fitted heads only)",
        "model_invocation_counts": {
            "model_loads_attempted": 0,
            "model_loads_completed": 0,
            "generation_calls_attempted": 0,
            "generation_calls_completed": 0,
        },
        "trained_head_specs": [],
        "energy_fit_ready_score": 0,
        "checkpoint_manifest_path": None,
        "prediction_rows_path": None,
        "trained_arm_specs": [],
        "observed_label_masks": {},
        "shortcut_control_rows": [],
        "parameter_counts": {},
        "source_cluster_counts": {},
        "field_principles": {
            "identity": "Current evidence belongs to this producer.",
            "gate": "A missing prerequisite is a terminal external block with exact operands.",
            "rows": "Primitive family and arm records permit cold reduction.",
            "readiness": "A complete null can be ready without proving benefit.",
            "custody": "Source and evaluator bytes remain hash bound and role separated.",
            "duration": "Monotonic intervals describe actual work only.",
            "models": "Small CPU heads are separate from pretrained model invocations.",
        },
    }


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(raw: Path) -> Path:
    """Write exact affected commands and deadlines before opening result labels."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    tests = [str(ROOT / item) for item in TESTS]
    commands = [
        {
            "name": "affected_unit",
            "argv": [pytest, "-n", "0", "-o", "addopts=", "--no-cov", "-q", *tests],
            "expected_exit": 0,
            "deadline_s": 600,
        },
        {
            "name": "e2e_015",
            "argv": [
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                str(ROOT / "tests/python/test_source_boundary_7852.py"),
            ],
            "expected_exit": 0,
            "deadline_s": 180,
        },
        {
            "name": "e2e_016_fixture",
            "argv": [
                py,
                str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"),
                "--date",
                "20260929",
                "--fixture-e2e",
                str(raw / "e2e_016_fixture.json"),
            ],
            "expected_exit": 0,
            "deadline_s": 180,
        },
        {
            "name": "e2e_016_replay",
            "argv": [
                py,
                str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"),
                "--date",
                "20260929",
                "--cold-replay",
                str(raw / "e2e_016_fixture.json"),
            ],
            "expected_exit": 0,
            "deadline_s": 180,
        },
        {
            "name": "ruff_check",
            "argv": [
                str(ROOT / ".venv/bin/ruff"),
                "check",
                *OWNED,
                "tests/python/test_energy_fit_7894.py",
            ],
            "expected_exit": 0,
            "deadline_s": 120,
        },
        {
            "name": "ruff_format",
            "argv": [
                str(ROOT / ".venv/bin/ruff"),
                "format",
                "--check",
                *OWNED,
                "tests/python/test_energy_fit_7894.py",
            ],
            "expected_exit": 0,
            "deadline_s": 120,
        },
        {
            "name": "mypy",
            "argv": [
                str(ROOT / ".venv/bin/mypy"),
                "--strict",
                "python/carnot/verify/energy_fit_7894.py",
            ],
            "expected_exit": 0,
            "deadline_s": 180,
        },
        {
            "name": "spec_coverage",
            "argv": [py, "scripts/check_spec_coverage.py", *tests],
            "expected_exit": 0,
            "deadline_s": 180,
        },
        {
            "name": "cli_expected_failure",
            "argv": [
                py,
                str(ROOT / OWNED[1]),
                "--date",
                "20260929",
                "--upstream",
                str(raw / "missing-upstream.json"),
                "--output",
                str(raw / "negative.json"),
                "--assert-ready",
            ],
            "expected_exit": 2,
            "expected_reason": "source evidence blocked",
            "deadline_s": 90,
        },
        {
            "name": "cli_cold_replay",
            "argv": [
                py,
                str(ROOT / OWNED[1]),
                "--date",
                "20260929",
                "--cold-replay",
                str(raw / "candidate.json"),
            ],
            "expected_exit": 0,
            "deadline_s": 180,
        },
    ]
    path = raw / "validation_command_manifest.json"
    atomic_json(
        path,
        {
            "schema": "carnot.exp7894.validation.v1",
            "commands": commands,
            "changed_modules": OWNED,
            "regression_tests": TESTS,
            "coverage_includes": [str((ROOT / item).resolve()) for item in OWNED],
            "unrelated_e2e_inapplicable": ["E2E-017", "E2E-018"],
        },
    )
    return path


def qualify(
    upstream: Path, records: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Carry all 640 families, including explicit view-budget exclusions."""
    artifact = json.loads(upstream.read_text())
    by_id = {row["id"]: row for row in records}
    counts = Counter(row["role"] for row in records)
    if counts != ROLE_BUDGET:
        raise custody.InputBlocked(
            [custody.operand(upstream, "role_counts", "==", ROLE_BUDGET, dict(counts))]
        )
    feature_rows, refs = custody._shard_rows(upstream, artifact, "feature_shards")
    sources.extend(refs)
    if len(feature_rows) != len(records) or {r["family_id"] for r in feature_rows} != set(by_id):
        raise custody.InputBlocked(
            [custody.operand(upstream, "feature_families", "==", 640, len(feature_rows))]
        )
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for feature in feature_rows:
        record = by_id[feature["family_id"]]
        if feature["feature_dim"] != 132:
            raise custody.InputBlocked(
                [custody.operand(upstream, "feature_dim", "==", 132, feature["feature_dim"])]
            )
        reason = feature["abstention"]
        if reason:
            excluded.append(
                {
                    "family_id": record["id"],
                    "role": record["role"],
                    "reason": reason,
                    "source_cluster_id": record["source_cluster_id"],
                }
            )
        else:
            eligible.append(record)
    cohort = Path(artifact["cohort_manifest_path"])
    if (
        not cohort.is_file()
        or canonical_hash(json.loads(cohort.read_text())).split(":")[1] not in cohort.name
    ):
        raise custody.InputBlocked(
            [
                custody.operand(
                    upstream,
                    "cohort_manifest_path",
                    "hash_named_file",
                    True,
                    str(cohort) if cohort.is_file() else None,
                )
            ]
        )
    sources.append({"role": "cohort_manifest", "path": str(cohort), "sha256": _sha(cohort)})
    return eligible, excluded


def fit_heads(eligible: list[dict[str, Any]], raw: Path) -> tuple[list[dict[str, Any]], Path]:
    """Run the registered library and save every trace before later-role scoring."""
    fit_rows = [row for row in eligible if row["role"] == "fit"]
    tune_rows = [row for row in eligible if row["role"] == "tune"]
    if not fit_rows or not tune_rows:
        raise ValueError("empty eligible fit or tune")
    checkpoints = raw / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    previous = raw / "checkpoint_manifest.json"
    bound = {name: _sha(ROOT / name) for name in OWNED}
    if previous.is_file():
        saved = json.loads(previous.read_text())
        # A scoring-only revision may reuse fitted heads while preserving the
        # original script hash in the checkpoint record.
        compatible = (
            saved["upstream_sha256"] == _sha(UPSTREAM)
            and saved["code_hashes"][OWNED[0]] == bound[OWNED[0]]
        )
        manifest = saved["checkpoints"] if compatible else []
        manifest = [
            item
            for item in manifest
            if Path(item["path"]).is_file() and _sha(Path(item["path"])) == item["sha256"]
        ]
    else:
        manifest = []
    deadline = time.monotonic() + 3000
    for arm in ARMS:
        for seed in SEEDS:
            if any(item["arm"] == arm and item["seed"] == seed for item in manifest):
                progress("fit", f"resume arm={arm} seed={seed}", len(manifest))
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("numerical budget 3000 seconds")
            count = len(manifest)
            progress("fit", f"before arm={arm} seed={seed}", count)
            stop = threading.Event()
            thread = threading.Thread(target=heartbeat, args=("fit", stop, count), daemon=True)
            thread.start()
            try:
                head = natural_training.fit(fit_rows, tune_rows, arm, seed, 0.01, 16)
            finally:
                stop.set()
                thread.join(timeout=1)
            if head["parameter_count"] > 4096 or len(head["curve"]) != 16:
                raise ValueError("head budget mismatch")
            path = checkpoints / f"{arm}_{seed}.json"
            training_runtime.save(path, head)
            manifest.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "path": str(path),
                    "sha256": _sha(path),
                    "parameter_count": head["parameter_count"],
                    "temperature": head["temperature"],
                    "initial_hash": head["initial_hash"],
                    "final_hash": head["final_hash"],
                    "tune_nll": head["tune_nll"],
                    "gradient_error": head["gradient_error"],
                    "epochs": head["curve"],
                }
            )
            progress("fit", f"after arm={arm} seed={seed}", len(manifest))
            atomic_json(
                raw / "checkpoint_manifest.json",
                {"checkpoints": manifest, "code_hashes": bound, "upstream_sha256": _sha(UPSTREAM)},
            )
    return manifest, raw / "checkpoint_manifest.json"


def _loss(probability: float, label: int) -> float:
    p = min(1 - 1e-6, max(1e-6, probability))
    return float(-label * np.log(p) - (1 - label) * np.log1p(-p))


def predict_batch(
    head: dict[str, Any],
    records: list[dict[str, Any]],
    batch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute the library's shared risk once before reading each family result."""
    if batch is None:
        batch, excluded = natural_training.prepare(records, head["view_arm"])
        if excluded:
            raise ValueError("ineligible prediction row")
    raw = np.asarray(
        training_runtime.deployed_risk(head["params"], batch, head["arm"], head["paired"])
    )
    result = []
    for value in raw:
        risk = float(value)
        probability = training_runtime.temperature_risk(risk, float(head["temperature"]))
        result.append(
            {
                "raw_risk": risk,
                "probability_unsupported": probability,
                "action": training_runtime.action(probability),
            }
        )
    return result


def score(
    eligible: list[dict[str, Any]], manifest: list[dict[str, Any]], raw: Path
) -> tuple[Path, list[dict[str, Any]]]:
    """Open later-role labels only after all model bytes have been sealed."""
    rows_path = raw / "prediction_rows.jsonl"
    controls: list[dict[str, Any]] = []
    role_members = {role: [row for row in eligible if row["role"] == role] for role in ROLE_BUDGET}
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    with rows_path.open("w") as stream:
        for index, spec in enumerate(manifest):
            head = training_runtime.load(Path(spec["path"]))
            for role in ROLE_BUDGET:
                members = role_members[role]
                if not members:
                    continue
                progress(
                    "score", f"before arm={spec['arm']} seed={spec['seed']} role={role}", index
                )
                stop = threading.Event()
                thread = threading.Thread(
                    target=heartbeat, args=("score", stop, index), daemon=True
                )
                thread.start()
                try:
                    view_class = (
                        head["view_arm"]
                        if head["view_arm"]
                        in ("source_erased_constrained_set", "complete_static_constrained_set")
                        else "local_set"
                    )
                    key = (role, view_class)
                    if key not in prepared:
                        batch, omitted = natural_training.prepare(members, view_class)
                        if omitted:
                            raise ValueError("ineligible prediction row")
                        prepared[key] = batch
                    predictions = predict_batch(head, members, prepared[key])
                finally:
                    stop.set()
                    thread.join(timeout=1)
                for record, prediction in zip(members, predictions, strict=True):
                    probability = prediction["probability_unsupported"]
                    row = {
                        "family_id": record["id"],
                        "role": role,
                        "arm": spec["arm"],
                        "seed": spec["seed"],
                        "source_cluster_id": record["source_cluster_id"],
                        "label": record["label"],
                        "known_mask": record["known"],
                        "raw_risk": prediction["raw_risk"],
                        "probability": probability,
                        "action": prediction["action"],
                        "loss": _loss(probability, record["label"]),
                        "source_length": len(record["source"]),
                        "answer_length": len(record["answer"]),
                        "status": "completed",
                        "checkpoint_sha256": spec["sha256"],
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                progress(
                    "score", f"after arm={spec['arm']} seed={spec['seed']} role={role}", index + 1
                )
    return rows_path, controls


def controls(
    eligible: list[dict[str, Any]], manifest: list[dict[str, Any]], raw: Path
) -> list[dict[str, Any]]:
    """Fit simple shortcuts on fit and tune, then score every eligible role."""
    fit = [r for r in eligible if r["role"] == "fit"]
    tune = [r for r in eligible if r["role"] == "tune"]
    mean = float(np.mean([r["label"] for r in fit]))

    def x(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [[1.0, np.log1p(len(r["source"])) / 10, np.log1p(len(r["answer"])) / 10] for r in rows]
        )

    weights = np.zeros(3)
    xf = x(fit)
    yf = np.asarray([r["label"] for r in fit])
    for _ in range(100):
        prediction = 1 / (1 + np.exp(-xf @ weights))
        weights -= 0.01 * xf.T @ (prediction - yf) / len(fit)
    tune_prob = 1 / (1 + np.exp(-x(tune) @ weights))
    tune_y = [r["label"] for r in tune]
    temperature = min(
        training_runtime.temperature_grid(),
        key=lambda t: np.mean(
            [
                _loss(training_runtime.temperature_risk(float(p), t), y)
                for p, y in zip(tune_prob, tune_y, strict=True)
            ]
        ),
    )
    rows = []
    for record, linear in zip(eligible, 1 / (1 + np.exp(-x(eligible) @ weights)), strict=True):
        for name, probability in (
            ("prevalence", mean),
            (
                "source_answer_length_logistic",
                training_runtime.temperature_risk(float(linear), temperature),
            ),
        ):
            rows.append(
                {
                    "family_id": record["id"],
                    "role": record["role"],
                    "control": name,
                    "label": record["label"],
                    "probability": probability,
                    "loss": _loss(probability, record["label"]),
                    "source_cluster_id": record["source_cluster_id"],
                    "status": "completed",
                }
            )
    target = next(
        item for item in manifest if item["arm"] == "constrained_set" and item["seed"] == SEEDS[0]
    )
    head = training_runtime.load(Path(target["path"]))
    for role in ROLE_BUDGET:
        members = [r for r in eligible if r["role"] == role]
        if len(members) < 2:
            continue
        ordered, donors = length_matched_donors(members)
        permuted = [
            {**r, "source": donor["source"]} for r, donor in zip(ordered, donors, strict=True)
        ]
        progress("controls", f"before permutation role={role}", len(rows))
        predictions = predict_batch(head, permuted)
        for record, donor, prediction in zip(ordered, donors, predictions, strict=True):
            probability = prediction["probability_unsupported"]
            rows.append(
                {
                    "family_id": record["id"],
                    "role": role,
                    "control": "length_matched_source_permutation",
                    "donor_family_id": donor["id"],
                    "source_length_delta": abs(len(record["source"]) - len(donor["source"])),
                    "label": record["label"],
                    "probability": probability,
                    "loss": _loss(probability, record["label"]),
                    "source_cluster_id": record["source_cluster_id"],
                    "status": "completed",
                }
            )
        progress("controls", f"after permutation role={role}", len(rows))
    atomic_json(
        raw / "shortcut_control_rows.json",
        {"rows": rows, "fit_only_weights": weights.tolist(), "tune_temperature": temperature},
    )
    return rows


def length_matched_donors(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair nearby source lengths without a shortest-to-longest wraparound."""
    if len(records) < 2:
        raise ValueError("source permutation requires two families")
    ordered = sorted(records, key=lambda row: (len(row["source"]), row["id"]))
    donors = ordered.copy()
    pair_limit = len(ordered) - (3 if len(ordered) % 2 else 0)
    for index in range(0, pair_limit, 2):
        donors[index], donors[index + 1] = ordered[index + 1], ordered[index]
    if len(ordered) % 2:
        donors[-3:] = [ordered[-2], ordered[-1], ordered[-3]]
    return ordered, donors


def run_checks(manifest_path: Path, raw: Path, *, late: bool = False) -> list[dict[str, Any]]:
    """Keep each child's exact argv, exit, deadline and durable log bytes."""
    frozen = json.loads(manifest_path.read_text())["commands"]
    receipts = []
    for spec in frozen:
        if (spec["name"] == "cli_cold_replay") != late:
            continue
        name = spec["name"]
        progress("validation", f"before {name}", len(receipts))
        began = time.monotonic()
        log_dir = raw / "validation_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        temporary = log_dir / f".{name}.running"
        with temporary.open("wb") as stream:
            child = subprocess.Popen(
                spec["argv"],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": "python:."},
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            while True:
                try:
                    exit_code = child.wait(timeout=45)
                    break
                except subprocess.TimeoutExpired:
                    progress("validation", f"heartbeat {name}", len(receipts))
                    if time.monotonic() - began > spec["deadline_s"]:
                        child.terminate()
                        try:
                            exit_code = child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            exit_code = child.wait()
                        break
        log_hash = _sha(temporary)
        durable = temporary.with_name(f"{name}_{log_hash.split(':')[1]}.log")
        temporary.replace(durable)
        reason = spec.get("expected_reason")
        content = durable.read_text(errors="replace")
        passed = exit_code == spec["expected_exit"] and (reason is None or reason in content)
        receipts.append(
            {
                "name": name,
                "argv": spec["argv"],
                "expected_exit": spec["expected_exit"],
                "actual_exit": exit_code,
                "expected_reason": reason,
                "passed": passed,
                "deadline_s": spec["deadline_s"],
                "duration_s": time.monotonic() - began,
                "log_path": str(durable),
                "log_sha256": log_hash,
                "classification": "required_affected",
            }
        )
        progress("validation", f"after {name} exit={exit_code} passed={passed}", len(receipts))
    return receipts


def replay(path: Path) -> None:
    """Cold-reduce primitive rows and reject changed checkpoint or loss bytes."""
    artifact = json.loads(path.read_text())
    rows_path = Path(artifact["prediction_rows_path"])
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    if rows != artifact["rows"] or sha256_file(rows_path) != artifact["prediction_rows_sha256"]:
        raise ValueError("prediction rows drift")
    for item in json.loads(Path(artifact["checkpoint_manifest_path"]).read_text())["checkpoints"]:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("checkpoint hash drift")
    for row in rows:
        if abs(_loss(row["probability"], row["label"]) - row["loss"]) > 1e-9:
            raise ValueError("primitive loss drift")
    progress("replay", "passed", len(rows))


def candidate(
    eligible: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    checkpoint_path: Path,
    prediction_path: Path,
    control_rows: list[dict[str, Any]],
    manifest_path: Path,
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce evidence without promoting development agreement into benefit."""
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    actual = len(rows)
    expected = len(eligible) * len(checkpoints)
    if actual != expected:
        raise ValueError(f"prediction count mismatch {actual} != {expected}")
    by_role = {
        role: {
            "intended": budget,
            "eligible": sum(r["role"] == role for r in eligible),
            "excluded": sum(r["role"] == role for r in excluded),
        }
        for role, budget in ROLE_BUDGET.items()
    }
    masks = {
        role: dict(
            Counter(label for row in eligible if row["role"] == role for label in row["known"])
        )
        for role in ROLE_BUDGET
    }
    clusters = {
        role: len({r["source_cluster_id"] for r in eligible if r["role"] == role})
        for role in ROLE_BUDGET
    }
    failures = [item for item in receipts if not item["passed"]]
    result = base(UPSTREAM, sources, [])
    result.update(
        {
            "honest_verdict": "complete_disqualified_required_checks"
            if failures
            else "complete_null_energy_fit",
            "verdict_class": "disqualified" if failures else "null",
            "rows": rows,
            "sample_size_budget": {
                "intended": 640 * 27,
                "eligible": len(eligible) * 27,
                "started": actual,
                "completed": actual,
                "failed": 0,
                "censored": 0,
                "excluded": len(excluded) * 27,
                "independent": len(eligible),
                "by_role": by_role,
                "exclusion_rows": excluded,
            },
            "acceptance_gate_results": {
                "validity": not failures,
                "readiness": 0 if failures else 1,
                "probability_quality": None,
                "decision_benefit": None,
                "retention": None,
                "efficiency": None,
            },
            "duration_s": time.monotonic() - START,
            "phase_spans": [
                {"phase": "source_and_fit_validation_score", "elapsed_s": time.monotonic() - START}
            ],
            "reproducibility_checksum": canonical_hash(
                {
                    "inputs": sources,
                    "code": {name: _sha(ROOT / name) for name in OWNED},
                    "seeds": SEEDS,
                    "arms": ARMS,
                    "epochs": 16,
                    "learning_rate": 0.01,
                }
            ),
            "preconditions_checked": {
                "failed": [],
                "role_counts": ROLE_BUDGET,
                "eligible_role_counts": {
                    role: values["eligible"] for role, values in by_role.items()
                },
                "gate_fields": {
                    "source_boundary_ready_score": 1,
                    "flagged_adversarial": False,
                    "verdict_class": "circular_positive",
                },
            },
            "resolved_imports": {
                "carnot.verify.natural_training": str(Path(natural_training.__file__).resolve()),
                "carnot.verify.training_runtime": str(Path(training_runtime.__file__).resolve()),
                "carnot.verify.energy_fit_7894": str(Path(custody.__file__).resolve()),
            },
            "validation_receipts": receipts,
            "validation_command_manifest_path": str(manifest_path),
            "observed_child_commands": [item["argv"] for item in receipts],
            "inference_substrate_class": "no_model_load",
            "trained_head_specs": [
                {
                    "arm": item["arm"],
                    "seed": item["seed"],
                    "parameter_count": item["parameter_count"],
                    "temperature": item["temperature"],
                }
                for item in checkpoints
            ],
            "energy_fit_ready_score": 0 if failures else 1,
            "checkpoint_manifest_path": str(checkpoint_path),
            "prediction_rows_path": str(prediction_path),
            "prediction_rows_sha256": sha256_file(prediction_path),
            "trained_arm_specs": checkpoints,
            "observed_label_masks": masks,
            "shortcut_control_rows": control_rows,
            "parameter_counts": {
                arm: sorted({item["parameter_count"] for item in checkpoints if item["arm"] == arm})
                for arm in ARMS
            },
            "source_cluster_counts": clusters,
        }
    )
    return result


def terminal_validate(path: Path, raw: Path) -> tuple[bool, list[dict[str, Any]]]:
    """Inspect the exact candidate bytes with both independent repository readers."""
    reports = []
    for name, argv in (
        (
            "adversarial",
            [sys.executable, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)],
        ),
        (
            "row_consistency",
            [
                sys.executable,
                str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(path),
            ],
        ),
    ):
        progress("terminal", f"before {name}", len(reports))
        result = subprocess.run(
            argv, cwd=ROOT, text=True, capture_output=True, timeout=180, check=False
        )
        report = {
            "name": name,
            "argv": argv,
            "candidate_sha256": sha256_file(path),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        location = raw / f"{name}_{sha256_file(path).split(':')[1]}.json"
        atomic_json(location, report)
        report["sidecar_path"] = str(location)
        reports.append(report)
        progress("terminal", f"after {name} exit={result.returncode}", len(reports))
    adversarial = json.loads(reports[0]["stdout"])
    flagged = adversarial["flagged_count"] > 0 or reports[0]["exit_code"] != 0
    return flagged or reports[1]["exit_code"] != 0, reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--assert-ready", action="store_true")
    args = parser.parse_args()
    progress("start", "flushed", 0)
    if args.date != "20260929":
        parser.error("V685 run date must be 20260929")
    if args.cold_replay:
        progress("replay", "before", 0)
        replay(args.cold_replay)
        return 0
    try:
        progress("preconditions", "before input hashes and gate", 0)
        records, sources = custody.load_records(args.upstream)
        eligible, excluded = qualify(args.upstream, records, sources)
        progress("preconditions", "after input hashes and gate", len(records))
    except custody.InputBlocked as exc:
        progress("preconditions", "source evidence blocked", 0)
        blocked = base(args.upstream, [], exc.operands)
        blocked["duration_s"] = time.monotonic() - START
        atomic_json(args.output, blocked)
        return 2 if args.assert_ready else 0
    raw = args.raw_root
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = freeze(raw)
    progress("freeze", "validation manifest and budgets", 0)
    try:
        checkpoints, checkpoint_path = fit_heads(eligible, raw)
        prediction_path, _ = score(eligible, checkpoints, raw)
        control_rows = controls(eligible, checkpoints, raw)
    except (TimeoutError, KeyboardInterrupt) as exc:
        partial = base(args.upstream, sources, [])
        partial.update(
            {
                "honest_verdict": "partial_numerical_work",
                "verdict_class": "partial",
                "duration_s": time.monotonic() - START,
                "checkpoint_manifest_path": str(raw / "checkpoint_manifest.json"),
                "validation_command_manifest_path": str(manifest_path),
                "unfinished_reason": str(exc),
            }
        )
        atomic_json(args.output, partial)
        return 1
    receipts = run_checks(manifest_path, raw)
    value = candidate(
        eligible,
        excluded,
        sources,
        checkpoints,
        checkpoint_path,
        prediction_path,
        control_rows,
        manifest_path,
        receipts,
    )
    staging = raw / "candidate.json"
    atomic_json(staging, value)
    progress("candidate", "before cold replay", len(checkpoints))
    late_receipts = run_checks(manifest_path, raw, late=True)
    receipts.extend(late_receipts)
    value = candidate(
        eligible,
        excluded,
        sources,
        checkpoints,
        checkpoint_path,
        prediction_path,
        control_rows,
        manifest_path,
        receipts,
    )
    atomic_json(staging, value)
    failed, reports = terminal_validate(staging, raw)
    if failed:
        value["honest_verdict"] = "complete_disqualified_terminal_validation"
        value["verdict_class"] = "disqualified"
        value["energy_fit_ready_score"] = 0
        value["acceptance_gate_results"]["validity"] = False
        value["acceptance_gate_results"]["readiness"] = 0
        value["flagged_adversarial"] = any(
            item["name"] == "adversarial" and json.loads(item["stdout"])["flagged_count"] > 0
            for item in reports
        )
        value["duration_s"] = time.monotonic() - START
        atomic_json(staging, value)
        failed, reports = terminal_validate(staging, raw)
    else:
        value["duration_s"] = time.monotonic() - START
        atomic_json(staging, value)
        failed, reports = terminal_validate(staging, raw)
    if failed:
        raise RuntimeError("terminal candidate remains flagged; see sealed reports")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(staging.read_bytes())
    temporary.replace(args.output)
    progress("publish", f"stable sha256={sha256_file(args.output)}", len(checkpoints))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
