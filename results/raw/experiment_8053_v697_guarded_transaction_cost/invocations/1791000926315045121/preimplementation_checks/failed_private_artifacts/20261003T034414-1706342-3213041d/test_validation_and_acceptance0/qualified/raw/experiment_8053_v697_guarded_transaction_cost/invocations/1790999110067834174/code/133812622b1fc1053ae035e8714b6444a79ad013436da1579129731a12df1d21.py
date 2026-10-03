"""REQ-REPORT-8040: measure repaired native work through durable next state.

This producer reuses the recorded binary and sealed released feedback. A valid
numerical measurement cannot establish learning benefit or a complete service.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_8027_v695_native_update_cost as old
from carnot import experiment_8032_v696_sealed_methods as sealed
from carnot import experiment_8039_v696_learning_benefit_audit as audit
from carnot.experiment_artifacts import artifact_output_root
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import windowed_online_8038 as window

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8040_v696_native_transaction_cost"
TASK = "exp8040-native-transaction-cost"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_native_transaction_8040.py"
OWNED = [f"python/carnot/{NAME}.py", SCRIPT]
CONDITIONS = ("recent64", "cumulative")
CONFIG: Json = dict(
    seed=8040,
    repetitions=30,
    warmups=5,
    draws=10000,
    tolerance=1e-10,
    budget_s=600,
    nfr01_target=10,
    statistic="ratio of summed paired transaction times",
    interval="paired percentile bootstrap; timing repeats are not natural n",
    policy="Exp8038 unchanged selector; four released gradients per block",
)
COMPONENTS = (
    "initialization_ns",
    "replay_selection_ns",
    "feature_gather_ns",
    "design_ns",
    "binding_ns",
    "serialization_ns",
    "persistence_fsync_ns",
    "reload_ns",
)
BEGAN = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed real counts make bounded work visible without invented waiting."""
    print(
        f"[exp8040] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} "
        f"completed={completed} pending={pending}",
        flush=True,
    )


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Reuse authenticated trajectory custody without reading retention labels."""
    data, failed = audit.load_inputs(root, raw)
    if failed:
        return data, failed
    path = root / "results" / (audit.NAME + ".json")
    try:
        value = json.loads(path.read_text())
        binding = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())[
            "publication"
        ]
        report = json.loads(Path(binding["sidecar_path"]).read_text())
        for field, expected, observed in (
            (
                "learning_audit_ready_score",
                1,
                value.get("learning_audit_ready_score", "MISSING_CONTRACT_FIELD"),
            ),
            (
                "flagged_adversarial",
                False,
                value.get("flagged_adversarial", "MISSING_CONTRACT_FIELD"),
            ),
            ("primary_sha256", sha256_file(path), binding["primary_sha256"]),
            ("sidecar.primary_sha256", sha256_file(path), report["primary_sha256"]),
            ("report.passed", True, report["report"]["passed"]),
        ):
            sealed.require(path, field, expected, observed)
            data["gate_checks"].append(
                dict(
                    upstream_id=value["task_id"],
                    path=str(path),
                    sha256=sha256_file(path),
                    artifact_field=field,
                    expected=expected,
                    observed=observed,
                    passed=True,
                    check_name=field,
                )
            )
        data["references"].extend(
            sealed.copy_bound(reference(p), raw)
            for p in (
                path,
                Path(binding["sidecar_path"]),
                Path(value["terminal_validation_sidecar_path"]),
            )
        )
        path = root / "results" / (old.NAME + ".json")
        previous = json.loads(path.read_text())
        library = {k: previous["loaded_extension_receipt"][k] for k in ("path", "sha256")}
        checked(library)
        repair = root / "results/raw" / old.NAME / "fixgate-validation"
        fixed = json.loads((repair / "repair_audit.json").read_text())
        sealed.require(
            repair / "repair_audit.json", "repair.process_exit", 0, fixed["repair"]["process_exit"]
        )
        for p in (
            path,
            repair / "repair_audit.json",
            repair / "original-assertions-pass-shutdown-crash.log",
            repair / "repaired-subset.log",
        ):
            data["references"].append(sealed.copy_bound(reference(p), raw, "historical_failure"))
        trajectory = Path(data["trajectory"])
        inputs = json.loads((trajectory / "inputs.json").read_text())
        data["references"].append(sealed.copy_bound(reference(trajectory / "inputs.json"), raw))
        data["references"].append(sealed.copy_bound(reference(trajectory / "ledger.sqlite"), raw))
        progress("qualified_small_head_load_before")
        inputs.update(
            workloads=workloads(trajectory, inputs),
            library_reference=library,
            references=data["references"],
            gate_checks=data["gate_checks"],
        )
        capture = root / "results/experiment_8033_v696_scoring_isolation.json"
        if capture.exists():
            current = json.loads(capture.read_text())
            inputs["references"].append(
                sealed.copy_bound(reference(capture), raw, "unqualified_acquisition")
            )
            inputs["likelihood_acquisition_cost"] = dict(
                status="unqualified",
                scope="separate current likelihood acquisition; no accepted timing imported",
                gate_operand=dict(
                    upstream_id=current["task_id"],
                    path=str(capture),
                    sha256=sha256_file(capture),
                    artifact_field="scoring_isolation_ready_score",
                    expected=1,
                    observed=current.get("scoring_isolation_ready_score", "MISSING_CONTRACT_FIELD"),
                    passed=False,
                    check_name="qualified_current_acquisition",
                ),
                duration_s=None,
            )
        progress("qualified_small_head_load_after", len(inputs["workloads"]))
        return inputs, []
    except sealed.Contract as error:
        return data, [error.gate]
    except (KeyError, ValueError, OSError) as error:
        return data, [
            sealed.Contract(
                path,
                str(error.args[0]) if isinstance(error, KeyError) else "input_contract",
                "complete byte-bound operands",
                "MISSING_CONTRACT_FIELD" if isinstance(error, KeyError) else str(error),
            ).gate
        ]


def workloads(trajectory: Path, data: Json) -> list[Json]:
    """Retain original block starts and releases so selection stays causal."""
    db = sqlite3.connect(f"file:{trajectory / 'ledger.sqlite'}?mode=ro", uri=True)
    events = [
        (k, json.loads(p))
        for k, p in db.execute(
            "SELECT kind,payload FROM events WHERE kind IN ('release','commit','issue') ORDER BY seq"
        )
    ]
    db.close()
    issues = {(r["arm"], r["seed"], r["slot"]): r for k, r in events if k == "issue"}
    heads, released, result = {}, {}, []
    for kind, r in events:
        if r["arm"] not in CONDITIONS:
            continue
        key = (r["arm"], r["seed"])
        if key not in heads:
            heads[key], released[key] = data["head"], []
        if kind == "release" and r["eligibility"]:
            released[key].append(r)
        if kind == "commit":
            checkpoint = json.loads(checked(r["checkpoint"]).read_text())
            source = (
                data["sources"][r["slot"] + 1] if r["slot"] + 1 < len(data["sources"]) else None
            )
            source = source if source and source["public_eligible"] else None
            result.append(
                dict(
                    condition=r["arm"],
                    seed=r["seed"],
                    block=r["block"],
                    head=heads[key],
                    released=list(released[key]),
                    next_source=source,
                    expected_coefficients=old.coefficients(checkpoint).tolist(),
                    expected_selected_ids=[g["family_id"] for g in r["gradients"]],
                    expected_probability=issues[(r["arm"], r["seed"], r["slot"] + 1)]["probability"]
                    if source
                    else None,
                )
            )
            heads[key] = checkpoint
    return result


def load_library(ref: Json, raw: Path) -> tuple[Any, Json]:
    """Atomic copies preserve mapped ELF bytes and forbid a Python substitute."""
    source = checked(ref)
    target = raw / "extension" / source.name
    old.copy_extension(source, target)
    progress("native_library_load_before")
    module = old.build.load_native_extension(target)
    sealed.require(
        target, "RustNumericalUpdate8027", True, hasattr(module, "RustNumericalUpdate8027")
    )
    sealed.require(
        target, "loaded_path", str(target.resolve()), str(Path(module.__file__).resolve())
    )
    sealed.require(target, "sha256", ref["sha256"], sha256_file(target))
    progress("native_library_load_after")
    return module, dict(
        reference(target),
        original_reference=ref,
        actual_loaded_path=str(Path(module.__file__).resolve()),
        interpreter=sys.executable,
        interpreter_sha256=sha256_file(Path(sys.executable)),
        api="RustNumericalUpdate8027.design/update_batch/state_json/effective",
        atomic_replacement=True,
        library_generation=False,
        actual_loaded=True,
        loaded_inode=target.stat().st_ino,
    )


def regression(ref: Json, raw: Path, scratch: Path) -> Json:
    """Require the shipped regression to survive Python's ELF finalizers."""
    tests = "tests/python/test_native_update_8027.py"
    command = CommandSpec(
        "mapped_inode_regression",
        (
            sys.executable,
            "-u",
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=" + str(scratch / "regression-pytest"),
            tests + "::test_req_pybind_8027_repeated_load_exits_normally",
            tests + "::test_req_pybind_8027_atomic_copy_preserves_previous_binary",
            "-q",
        ),
        "owned",
        60,
    )
    return run_commands(
        ROOT,
        [command],
        log_dir=raw / "regression_logs",
        heartbeat_s=10,
        extra_env=dict(
            CARNOT_8027_EXTENSION=str(checked(ref)), CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(scratch)
        ),
    )[0]


def transaction(data: Json, w: Json, native: Any, arm: str, file: Path) -> Json:
    """One elapsed span includes feature work through synchronous restart.

    Cached likelihood scalars enter as inputs. Their acquisition and the external
    feedback provider are absent, so this span is a numerical branch cost only.
    """
    progress("transaction_before")
    components: Json = {}
    began = start = time.perf_counter_ns()
    state = (
        copy.deepcopy(w["head"])
        if arm == "python"
        else native.RustNumericalUpdate8027(json.dumps(w["head"]))
    )
    components["initialization_ns"] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    pool, chosen = window.select(w["released"], w["condition"], w["seed"], w["block"])
    components["replay_selection_ns"] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    selected = [data["sources"][r["origin_slot"]] for r in chosen]
    public = [
        old.features.extract({k: r[k] for k in ("family_id", "source_bytes", "answer_bytes")})[
            "values"
        ]
        for r in selected
    ]
    if any(v != r["features"] for v, r in zip(public, selected, strict=True)):
        raise ValueError("public_feature_drift")
    fresh = np.asarray([[r["q"], *v] for r, v in zip(selected, public, strict=True)])
    labels = [r["y"] for r in chosen]
    components["feature_gather_ns"] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    matrix = (
        old.python_design(w["head"], fresh) if arm == "python" else state.design(fresh.tolist())
    )
    components["design_ns"] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    predictions, kernel_ns, arithmetic_ns = (
        old.python_batch(state, matrix, labels)
        if arm == "python"
        else state.update_batch(matrix, labels)
    )
    components["binding_ns"] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    head = state if arm == "python" else json.loads(state.state_json())
    envelope = dict(
        head=head,
        replay_pool=pool,
        selected_ids=[r["family_id"] for r in chosen],
        condition=w["condition"],
        seed=w["seed"],
        block=w["block"],
    )
    encoded = json.dumps(envelope, sort_keys=True)
    pool_bytes = len(json.dumps(pool, sort_keys=True).encode())
    components["serialization_ns"] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    file.parent.mkdir(parents=True, exist_ok=True)
    staged = file.with_suffix(".pending")
    with staged.open("w") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, file)
    for parent in (file.parent, file.parent.parent):
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    components["persistence_fsync_ns"] = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    reopened = json.loads(file.read_text())
    restored = (
        reopened["head"]
        if arm == "python"
        else native.RustNumericalUpdate8027(json.dumps(reopened["head"]))
    )
    effective = old.coefficients(restored).tolist() if arm == "python" else restored.effective()
    source = w["next_source"]
    probability = None
    if source:
        x = old.causal.design(w["head"], source)
        probability = (
            old.causal.probability(restored, x)
            if arm == "python"
            else restored.update_batch([x.tolist()], [None])[0][0]
        )
    components["reload_ns"] = time.perf_counter_ns() - start
    elapsed = time.perf_counter_ns() - began
    error = float(np.max(np.abs(np.asarray(effective) - w["expected_coefficients"])))
    probability_error = abs(probability - w["expected_probability"]) if source else 0.0
    progress("transaction_after", 1)
    return dict(
        components,
        arm=arm,
        condition=w["condition"],
        seed=w["seed"],
        block=w["block"],
        transaction_ns=elapsed,
        kernel_ns=kernel_ns,
        arithmetic_ns=arithmetic_ns,
        ffi_overhead_ns=max(0, components["binding_ns"] - kernel_ns),
        update_overhead_ns=elapsed - arithmetic_ns,
        replay_memory_bytes=pool_bytes,
        replay_memory_scope="measured UTF-8 serialized replay pool; allocator/RSS is not isolated",
        serialized_bytes=len(encoded.encode()),
        checkpoint=reference(file),
        effective=effective,
        predictions=predictions,
        selected_ids=envelope["selected_ids"],
        next_probability=probability,
        next_probability_error=probability_error,
        next_action_equal=old.action(probability) == old.action(w["expected_probability"]),
        coefficient_error=error,
        input_hash=canonical_hash(w),
        numerator=elapsed,
        denominator=len(chosen),
        eligibility=True,
        failure_reason=None,
        censor_reason=None,
        process_identity=dict(
            pid=os.getpid(),
            executable=sys.executable,
            host=platform.node(),
            affinity=sorted(os.sched_getaffinity(0)),
        ),
    )


def parity(data: Json, native: Any, raw: Path) -> Json:
    """Replay every original commit, including calibrated restart predictions."""
    rows, checkpoints = [], []
    progress("parity_before", 0, len(data["workloads"]))
    for i, w in enumerate(data["workloads"]):
        h = copy.deepcopy(w["head"])
        n = native.RustNumericalUpdate8027(json.dumps(h))
        chosen = window.select(w["released"], w["condition"], w["seed"], w["block"])[1]
        selected = [data["sources"][r["origin_slot"]] for r in chosen]
        fresh = np.asarray([[r["q"], *r["features"]] for r in selected])
        x = old.python_design(h, fresh)
        nx = n.design(fresh.tolist())
        p = old.python_batch(h, x, [r["y"] for r in chosen])[0]
        q = n.update_batch(nx, [r["y"] for r in chosen])[0]
        path = raw / "parity_checkpoints" / f"{i}.json"
        atomic_json(path, json.loads(n.state_json()))
        restored = native.RustNumericalUpdate8027(path.read_text())
        source = w["next_source"]
        pp = qq = None
        if source:
            v = old.causal.design(h, source)
            pp = old.causal.probability(h, v)
            qq = restored.update_batch([v.tolist()], [None])[0][0]
        row = dict(
            condition=w["condition"],
            seed=w["seed"],
            block=w["block"],
            probability_error=max(abs(a - b) for a, b in zip(p, q, strict=True)),
            basis_error=float(np.max(np.abs(x - np.asarray(nx)))),
            coefficient_error=float(np.max(np.abs(old.coefficients(h) - restored.effective()))),
            original_state_error=float(
                np.max(np.abs(old.coefficients(h) - w["expected_coefficients"]))
            ),
            decay_error=abs(h["decay_scale"] - json.loads(n.state_json())["decay_scale"]),
            next_probability_error=max(abs(pp - qq), abs(pp - w["expected_probability"]))
            if source
            else 0.0,
            action_disagreements=sum(
                old.action(a) != old.action(b) for a, b in zip(p, q, strict=True)
            )
            + int(old.action(pp) != old.action(qq)),
            selected_ids_equal=[r["family_id"] for r in chosen]
            == w.get("expected_selected_ids", [r["family_id"] for r in chosen]),
            checkpoint=reference(path),
            numerator=len(chosen),
            denominator=len(chosen),
            eligibility=True,
            exclusion_reason=None,
            failure_reason=None,
            censor_reason=None,
        )
        row["passed"] = (
            all(
                row[k] <= CONFIG["tolerance"]
                for k in (
                    "probability_error",
                    "basis_error",
                    "coefficient_error",
                    "original_state_error",
                    "decay_error",
                    "next_probability_error",
                )
            )
            and row["action_disagreements"] == 0
            and row["selected_ids_equal"]
        )
        rows.append(row)
        checkpoints.append(reference(path))
        if (i + 1) % 40 == 0:
            progress("parity_commits", i + 1, len(data["workloads"]) - i - 1)
    atomic_json(raw / "parity_rows.json", dict(rows=rows))
    progress("parity_after", len(rows))
    return dict(
        numerical_parity_rows=rows,
        parity_passed=bool(rows) and all(r["passed"] for r in rows),
        checkpoint_references=checkpoints,
    )


def controls(head: Json, native: Any) -> Json:
    """Reuse isolated stress without adding synthetic cases to natural evidence."""
    with tempfile.TemporaryDirectory(prefix="carnot-8040-controls-") as temporary:
        result = old.parity(dict(head=head, updates=[], sources=[]), native, Path(temporary))
    return dict(
        passed=result["parity_passed"],
        rows=result["parity_rows"],
        scope="synthetic basis and calibrated sparse-update controls; independent_n=0",
    )


def measure(data: Json, native: Any, raw: Path) -> Json:
    """Random order and excluded warmups compare equivalent complete transactions."""
    rng = np.random.default_rng(CONFIG["seed"])
    rows = []
    deadline = time.monotonic() + CONFIG["budget_s"]
    total = len(CONDITIONS) * (CONFIG["warmups"] + CONFIG["repetitions"]) * 2
    progress("benchmark_before", 0, total)
    for condition in CONDITIONS:
        candidates = [w for w in data["workloads"] if w["condition"] == condition]
        for repetition in range(-CONFIG["warmups"], CONFIG["repetitions"]):
            if time.monotonic() > deadline:
                raise TimeoutError("benchmark_budget")
            w = candidates[(repetition + CONFIG["warmups"]) % len(candidates)]
            order = rng.permutation(["python", "native"]).tolist()
            pair = []
            for arm in order:
                row = transaction(
                    data,
                    w,
                    native,
                    arm,
                    raw / "transactions" / f"{condition}-{repetition}-{arm}.json",
                )
                row.update(
                    repetition=repetition,
                    pair_order=order,
                    excluded=repetition < 0,
                    exclusion_reason="excluded_warmup" if repetition < 0 else None,
                )
                rows.append(row)
                pair.append(row)
            sealed.require(raw, "paired_input_hash", pair[0]["input_hash"], pair[1]["input_hash"])
            sealed.require(
                raw,
                "paired_coefficients",
                True,
                bool(
                    np.max(np.abs(np.asarray(pair[0]["effective"]) - pair[1]["effective"]))
                    <= CONFIG["tolerance"]
                ),
            )
            sealed.require(
                raw,
                "transaction_parity",
                True,
                all(
                    r["coefficient_error"] <= CONFIG["tolerance"]
                    and r["next_probability_error"] <= CONFIG["tolerance"]
                    and r["next_action_equal"]
                    for r in pair
                ),
            )
            progress("benchmark_pair_after", len(rows), total - len(rows))
    atomic_json(raw / "transaction_rows.json", dict(rows=rows))
    progress("benchmark_after", len(rows))
    return summarize(rows, CONFIG)


def summarize(rows: list[Json], config: Json) -> Json:
    """Paired resampling retains order-matched input costs in each draw."""
    distributions, full, arithmetic = [], [], []
    for condition in CONDITIONS:
        pairs = [
            [
                r
                for r in rows
                if r["condition"] == condition and r["arm"] == arm and not r["excluded"]
            ]
            for arm in ("python", "native")
        ]
        pairs = [sorted(part, key=lambda r: r["repetition"]) for part in pairs]
        for arm, part in zip(("python", "native"), pairs, strict=True):
            distributions.append(
                dict(
                    condition=condition,
                    arm=arm,
                    count=len(part),
                    **{
                        k: old.percentiles([r[k] for r in part])
                        for k in (
                            *COMPONENTS,
                            "transaction_ns",
                            "arithmetic_ns",
                            "replay_memory_bytes",
                            "update_overhead_ns",
                        )
                    },
                )
            )
        for metric, target in [("transaction_ns", full), ("arithmetic_ns", arithmetic)]:
            a, b = (np.asarray([r[metric] for r in part], dtype=float) for part in pairs)
            draws = np.random.default_rng(config["seed"]).integers(
                0, len(a), (config["draws"], len(a))
            )
            bootstrap = np.sum(a[draws], axis=1) / np.sum(b[draws], axis=1)
            target.append(
                dict(
                    condition=condition,
                    speedup=float(a.sum() / b.sum()),
                    lower_95=float(np.quantile(bootstrap, 0.025)),
                    upper_95=float(np.quantile(bootstrap, 0.975)),
                    paired_repetitions=len(a),
                    independent_streams=1,
                    uncertainty_scope="conditional timing uncertainty for this frozen finite workload",
                )
            )
    return dict(
        transaction_rows=rows,
        timing_components=distributions,
        complete_workload_speedup=full,
        arithmetic_speedup=arithmetic,
        nfr01_met=all(r["speedup"] >= 10 and r["lower_95"] >= 10 for r in full),
    )


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Reuse bounded validation and measure only this producer and thin CLI."""
    prior = old.eligible
    commands = prior.validation_plan(scratch)
    config = scratch / "coverage.ini"
    config.write_text(config.read_text().replace(prior.NAME, NAME))
    adapted = []
    for c in commands:
        argv = tuple(a.replace(prior.NAME, NAME).replace(prior.TEST, TEST) for a in c.argv)
        if c.name == "unit_consumers_e2e015_019":
            argv = tuple(
                a
                for a in argv
                if not a.startswith("tests/")
                or a in {TEST, "tests/python/test_primary_publication_7928.py"}
            )
        adapted.append(replace(c, argv=argv, timeout_s=min(c.timeout_s, 120)))
    return adapted


def validate(raw: Path, scratch: Path) -> tuple[list[Json], Json, list[Json]]:
    """Separate owned checks from the single bounded whole-repository diagnostic."""
    progress("validation_before")
    receipts = run_commands(
        ROOT,
        validation_plan(scratch),
        log_dir=raw / "validation_logs",
        heartbeat_s=10,
        extra_env=dict(CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(scratch)),
    )
    file = scratch / "coverage.json"
    counts = (
        {k: v["summary"] for k, v in json.loads(file.read_text())["files"].items()}
        if file.exists()
        else {}
    )
    atomic_json(raw / "coverage.json", dict(files=counts))
    progress("validation_after", len(receipts))
    return (
        [r for r in receipts if r["scope"] == "owned"],
        counts,
        [r for r in receipts if r["scope"] == "repository_health"],
    )


def replay(path: Path) -> Json:
    """Fresh readers bind raw bytes, reduce timings and recompute equations."""
    v = json.loads(path.read_text())
    if v["native_transaction_ready_score"] and (
        v["verdict_class"] in {"blocked", "disqualified"}
        or not all(v["acceptance_gate_results"].values())
    ):
        raise ValueError("unsafe_readiness")
    for ref in v["raw_shard_hashes"] + v["code_config_hashes"] + v["checkpoint_references"]:
        checked(ref)
    if v["acceptance_gate_results"]["measurement"]:
        raw = Path(v["raw_directory"])
        receipt = v["loaded_library_receipt"]
        checked(receipt)
        native = old.build.load_native_extension(Path(receipt["path"]))
        data = json.loads((raw / "inputs.json").read_text())
        rows = json.loads((raw / "transaction_rows.json").read_text())["rows"]
        for k, observed in summarize(rows, v["config"]).items():
            sealed.require(path, "reduction." + k, v[k], observed)
        parity_rows = json.loads((raw / "parity_rows.json").read_text())["rows"]
        sealed.require(path, "parity_rows", v["numerical_parity_rows"], parity_rows)
        with tempfile.TemporaryDirectory(prefix="carnot-8040-cold-") as temporary:
            cold = parity(data, native, Path(temporary))
            stripped = [{k: x for k, x in r.items() if k != "checkpoint"} for r in parity_rows]
            sealed.require(
                path,
                "cold_parity",
                stripped,
                [
                    {k: x for k, x in r.items() if k != "checkpoint"}
                    for r in cold["numerical_parity_rows"]
                ],
            )
        workload_index = {canonical_hash(w): w for w in data["workloads"]}
        for row in rows:
            checkpoint = json.loads(checked(row["checkpoint"]).read_text())
            w = workload_index[row["input_hash"]]
            pool, chosen = window.select(w["released"], w["condition"], w["seed"], w["block"])
            sealed.require(path, "durable_replay_pool", pool, checkpoint["replay_pool"])
            sealed.require(
                path, "causal_selected_ids", [r["family_id"] for r in chosen], row["selected_ids"]
            )
            sealed.require(
                path,
                "original_transaction_state",
                True,
                bool(
                    np.max(np.abs(np.asarray(w["expected_coefficients"]) - row["effective"]))
                    <= v["config"]["tolerance"]
                ),
            )
            restored = native.RustNumericalUpdate8027(json.dumps(checkpoint["head"]))
            sealed.require(
                path,
                "checkpoint_restart",
                True,
                bool(
                    np.max(np.abs(np.asarray(restored.effective()) - row["effective"]))
                    <= v["config"]["tolerance"]
                ),
            )
            sealed.require(
                path, "durable_selected_ids", row["selected_ids"], checkpoint["selected_ids"]
            )
    return dict(passed=True)


def terminal_commands(path: Path) -> Json:
    """Check both candidate and published bytes with existing terminal readers."""
    raw = Path(json.loads(path.read_text())["raw_directory"])
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / SCRIPT), "--cold-replay", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
            "terminal",
            60,
        ),
    ]
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=raw
        / "terminal_logs"
        / ("candidate" if path.name == "terminal_candidate.json" else "published"),
        heartbeat_s=10,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def terminal(path: Path) -> Json:
    """Use a shared adapter so private tests still exercise publication locking."""
    return terminal_commands(path)


def base(failures: list[Json]) -> Json:
    """Complete numerical negatives qualify readiness without service promotion."""
    return dict(
        experiment_id=8040,
        task_id=TASK,
        milestone="2026.10.696",
        schema="carnot.v696.native_transaction_cost.v1",
        run_date="20261002",
        claim_scope="This invocation measures opt-in durable numerical updates on sealed exposed-development feedback; no service or learning benefit claim.",
        honest_verdict="complete_blocked_native_transaction_cost"
        if failures
        else "complete_null_native_transaction_cost",
        verdict_class="blocked" if failures else "null",
        gate_check_summary=failures,
        native_transaction_ready_score=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=False,
        genuine_headroom=dict(learning_benefit=False),
        positive_control_results={},
        mapped_inode_regression={},
        loaded_library_receipt={},
        numerical_parity_rows=[],
        transaction_rows=[],
        timing_components=[],
        complete_workload_speedup=[],
        missing_service_components=[
            "current scalar likelihood-feature acquisition",
            "external feedback-provider cost",
        ],
        likelihood_acquisition_cost=dict(
            status="missing_qualified_current_stream_join",
            scope="separate acquisition only; no historical timing imported",
        ),
        cost_scope="Input cached likelihood scalar and public bytes plus released feedback, through durable head and replay pool reload; complete numerical branch only.",
        acceptance_gate_results=dict(
            regression=False, parity=False, measurement=False, owned_checks=False
        ),
        checkpoint_references=[],
        raw_shard_hashes=[],
        code_config_hashes=[],
        cited_upstream_artifacts=[],
        rows=[],
        sample_size_budget={},
        intended_count=0,
        eligible_count=0,
        completed_count=0,
        excluded_count=0,
        failed_count=0,
        censored_count=0,
        independent_count=0,
        config=copy.deepcopy(CONFIG),
        random_seed=CONFIG["seed"],
        reproducibility_checksum=None,
        inference_substrate="verifier_scoring",
        inference_substrate_class="no_model_load",
        substrate_declaration=dict(
            custody="aggregation_from_upstream_artifacts",
            numerical="verifier_scoring",
            no_model_load=True,
        ),
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        duration_s=0,
        phase_spans=[],
        validation_receipts=[],
        coverage_statement_counts={},
        terminal_validation_sidecar_path=None,
        flagged_adversarial=False,
        field_principles={},
        default_enabled=False,
        continuous_self_learning_task=False,
        execution_venue="host",
        nfr01_met=False,
    )


def main(argv: list[str] | None = None) -> int:
    """Freeze, validate, measure and publish one stable task identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        print(json.dumps(replay(args.cold_replay)), flush=True)
        return 0
    started = time.monotonic()
    output = (args.output or artifact_output_root(root=args.root) / (NAME + ".json")).absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    progress("freeze_before")
    with tempfile.TemporaryDirectory(prefix="carnot-8040-") as temporary:
        scratch = Path(temporary)
        data, failures = (
            (json.loads(args.fixture_input.read_text()), [])
            if args.fixture_input
            else load_inputs(args.root, raw)
        )
        value = base(failures)
        value.update(
            raw_directory=str(raw),
            cited_upstream_artifacts=data.get("references", []),
            gate_check_summary=data.get("gate_checks", []) + failures,
            likelihood_acquisition_cost=data.get(
                "likelihood_acquisition_cost", value["likelihood_acquisition_cost"]
            ),
        )
        dependencies = [
            *OWNED,
            TEST,
            old.OWNED[0],
            audit.OWNED[0],
            "python/carnot/verify/windowed_online_8038.py",
            "python/carnot/verify/causal_online_8025.py",
            "python/carnot/verify/evidence_features_7980.py",
            "python/carnot/experiment_8008_v694_conditioned_energy_fit.py",
            "crates/carnot-core/src/numerical_update_8027.rs",
            "crates/carnot-python/src/numerical_update_8027.rs",
        ]
        value["code_config_hashes"] = [
            sealed.copy_bound(reference(ROOT / p), raw, "code") for p in dependencies
        ]
        methods = raw / "methods.json"
        atomic_json(
            methods,
            dict(
                config=CONFIG,
                identity=TASK,
                data_access="original public bytes and already released feedback; no retention labels",
                numerical_acceptance="probability and state error <=1e-10; zero action disagreement",
                artifact_guard_enabled=True,
                model_loads=0,
                fixture=bool(args.fixture_input),
                production_defaults_changed=False,
                binary_generation=False,
            ),
        )
        value["code_config_hashes"].append(reference(methods))
        progress("freeze_after")
        if not failures:
            phase = time.monotonic()
            reg = regression(data["library_reference"], raw, scratch)
            value["mapped_inode_regression"] = reg
            value["acceptance_gate_results"]["regression"] = reg["passed"]
            if not reg["passed"]:
                value["gate_check_summary"].append(
                    sealed.Contract(
                        Path(reg.get("log_path", str(raw))),
                        "mapped_inode_regression.exit_code",
                        0,
                        reg.get("exit_code", "MISSING_CONTRACT_FIELD"),
                    ).gate
                )
                value.update(
                    verdict_class="disqualified",
                    honest_verdict="complete_disqualified_mapped_inode_regression",
                )
            else:
                native, receipt = load_library(data["library_reference"], raw)
                value["loaded_library_receipt"] = receipt
                os.environ["CARNOT_8027_EXTENSION"] = receipt["path"]
                receipts, counts, health = (
                    ([], {}, []) if args.validation_worker else validate(raw, scratch)
                )
                good = (
                    bool(receipts)
                    and all(r["passed"] for r in receipts)
                    and set(counts) == set(OWNED)
                    and all(
                        r["missing_lines"] == 0 and r["num_statements"] > 0 for r in counts.values()
                    )
                )
                value.update(
                    validation_receipts=receipts,
                    coverage_statement_counts=counts,
                    repository_health=health,
                )
                value["acceptance_gate_results"]["owned_checks"] = good
                atomic_json(raw / "inputs.json", data)
                progress("measurement_before")
                try:
                    value.update(parity(data, native, raw))
                    value["positive_control_results"] = controls(data["head"], native)
                    value.update(measure(data, native, raw))
                    value["acceptance_gate_results"].update(
                        parity=value["parity_passed"]
                        and value["positive_control_results"]["passed"],
                        measurement=True,
                    )
                except (ValueError, TimeoutError, sealed.Contract) as error:
                    value["gate_check_summary"].append(
                        error.gate
                        if isinstance(error, sealed.Contract)
                        else sealed.Contract(
                            raw, "measurement_contract", "complete valid measurement", str(error)
                        ).gate
                    )
                value["native_transaction_ready_score"] = int(
                    all(value["acceptance_gate_results"].values())
                )
                if not value["native_transaction_ready_score"]:
                    value.update(
                        verdict_class="disqualified",
                        honest_verdict="complete_disqualified_native_transaction_cost",
                    )
                value["rows"] = value["transaction_rows"]
                value["checkpoint_references"] += [
                    r["checkpoint"] for r in value["transaction_rows"]
                ]
                measured = [r for r in value["transaction_rows"] if not r["excluded"]]
                intended = len(CONDITIONS) * CONFIG["repetitions"] * 2
                value.update(
                    intended_count=intended,
                    eligible_count=intended,
                    completed_count=len(measured),
                    excluded_count=len(value["transaction_rows"]) - len(measured),
                    independent_count=len(
                        {r["family_id"] for r in data["sources"] if r["public_eligible"]}
                    ),
                    sample_size_budget=dict(
                        intended=intended,
                        eligible=intended,
                        completed=len(measured),
                        excluded=len(value["transaction_rows"]) - len(measured),
                        failed=0,
                        censored=0,
                        independent_streams=1,
                        seeds_are_independent=False,
                        warmup_pairs_per_condition=CONFIG["warmups"],
                        measured_pairs_per_condition=CONFIG["repetitions"],
                    ),
                    trained_head_specs=[
                        dict(
                            pretrained=False,
                            parameters=110,
                            operation="replay original calibrated small-head updates; no benefit credit",
                        )
                    ],
                )
                progress("measurement_after", len(measured))
            value["phase_spans"].append(
                dict(
                    phase="regression_validation_and_numerical_work",
                    duration_s=time.monotonic() - phase,
                )
            )
        value["raw_shard_hashes"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
        value["reproducibility_checksum"] = canonical_hash(
            dict(
                config=CONFIG,
                code=value["code_config_hashes"],
                inputs=value["cited_upstream_artifacts"],
            )
        )
        value["duration_s"] = time.monotonic() - started
        value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
        value["field_principles"] = {
            k: "Bind "
            + k
            + " to actual work, durable evidence and its numerical-only claim boundary."
            for k in value
        }
        value["field_principles"].update(
            native_transaction_ready_score="Complete valid parity, measurement and owned checks qualify numerical readiness independently of speed or benefit.",
            complete_workload_speedup="Only complete equivalent durable transactions and their paired interval can close NFR-01.",
            missing_service_components="Missing external acquisition and feedback costs prevent a complete current service claim.",
            mapped_inode_regression="Normal process exit after assertions proves live mapped bytes survive finalizers.",
            model_invocation_counts="Imported receipts never create current pretrained calls.",
        )
        progress("publication_before")
        publication = publish_primary(output, value, terminal)
        report = terminal(output)
        reader = reader_receipt(
            TASK,
            output.parent,
            field="native_transaction_ready_score",
            expected=value["native_transaction_ready_score"],
        )
        atomic_json(
            raw / "terminal_validation.json",
            dict(publication=publication, published=report, reader=reader, process_exit_required=0),
        )
        progress("publication_after", 1)
        return 0 if report["passed"] and reader["passed"] else 1
