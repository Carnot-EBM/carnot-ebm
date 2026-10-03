"""REQ-REPORT-8053: publish guarded CPU transaction costs with exact evidence limits."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from typing import Any

from carnot import experiment_8040_v696_native_transaction_cost as prior
from carnot.experiment_artifacts import artifact_output_root
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import guarded_transaction_8053 as m

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8053_v697_guarded_transaction_cost"
TASK = "exp8053-guarded-transaction-cost"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_guarded_transaction_8053.py"
OWNED = [f"python/carnot/{NAME}.py", "python/carnot/verify/guarded_transaction_8053.py", SCRIPT]


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Authenticate terminal producers and retain original SQLite rows before replay."""
    data: Json = dict(references=[], gate_checks=[])
    failures = []
    values: Json = {}
    for identity, name, ready in [
        (
            8051,
            "experiment_8051_v697_feedback_constrained_learning",
            "learning_trajectory_ready_score",
        ),
        (8040, prior.NAME, "native_transaction_ready_score"),
    ]:
        path = root / "results" / (name + ".json")
        try:
            value = json.loads(path.read_text())
            sidecar = Path(value["terminal_validation_sidecar_path"])
            binding = json.loads(sidecar.read_text())["publication"]
            report_path = Path(binding["sidecar_path"])
            report = json.loads(report_path.read_text())
            for field, expected, observed in [
                ("experiment_id", identity, value.get("experiment_id")),
                (ready, 1, value.get(ready, "MISSING_CONTRACT_FIELD")),
                (
                    "flagged_adversarial",
                    False,
                    value.get("flagged_adversarial", "MISSING_CONTRACT_FIELD"),
                ),
                ("primary_sha256", sha256_file(path), binding["primary_sha256"]),
                ("sidecar.primary_sha256", sha256_file(path), report["primary_sha256"]),
                ("primary_path", str(path), binding["primary_path"]),
                ("report.passed", True, report["report"]["passed"]),
            ]:
                gate = dict(
                    upstream_id=value["task_id"],
                    path=str(path),
                    sha256=sha256_file(path),
                    check_name=field,
                    artifact_field=field,
                    expected=expected,
                    observed=observed,
                    passed=expected == observed,
                )
                data["gate_checks"].append(gate)
                if not gate["passed"]:
                    failures.append(gate)
            data["references"] += [
                prior.sealed.copy_bound(reference(p), raw) for p in (path, sidecar, report_path)
            ]
            values[identity] = value
        except (OSError, ValueError, KeyError) as error:
            failures.append(
                prior.sealed.Contract(
                    path, "input_contract", "present authenticated upstream", str(error)
                ).gate
            )
    for name in [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "scripts/experiment_template.py",
        ".venv/bin/python",
        ".venv/bin/pytest",
        ".venv/bin/coverage",
        ".venv/bin/ruff",
        ".venv/bin/mypy",
    ]:
        p = ROOT / name
        gate = dict(
            upstream_id="local_resource",
            path=str(p),
            sha256=sha256_file(p) if p.exists() else None,
            check_name="resource_exists",
            artifact_field="resource_exists",
            expected=True,
            observed=p.is_file(),
            passed=p.is_file(),
        )
        data["gate_checks"].append(gate)
        if not gate["passed"]:
            failures.append(gate)
    if not failures:
        try:
            library = values[8040]["loaded_library_receipt"]
            checked(library)
            data["library_reference"] = {k: library[k] for k in ("path", "sha256")}
            trajectory = Path(values[8051]["trajectory_directory"])
            for p in [
                trajectory / "inputs.json",
                trajectory / "methods.json",
                *sorted(trajectory.glob("seed-*/ledger.sqlite")),
            ]:
                data["references"].append(prior.sealed.copy_bound(reference(p), raw))
            data.update(json.loads((trajectory / "inputs.json").read_text()))
            m.progress("qualified_small_head_load_before")
            data["cases"] = workloads(trajectory, data)
            m.progress("qualified_small_head_load_after", len(data["cases"]))
        except (OSError, ValueError, KeyError) as error:
            failures.append(
                prior.sealed.Contract(
                    root, "trajectory_contract", "complete primitive custody", str(error)
                ).gate
            )
    return data, failures


def workloads(trajectory: Path, data: Json) -> list[Json]:
    """Carry every candidate and guard expansion with its complete released buffer."""
    cases = []
    for file in sorted(trajectory.glob("seed-*/ledger.sqlite")):
        db = sqlite3.connect(f"file:{file}?mode=ro", uri=True)
        pools: Json = {}
        gradients: Json = {}
        states: Json = {}
        for kind, identity, text in db.execute(
            "SELECT kind,identity,payload FROM events ORDER BY seq"
        ):
            row = json.loads(text)
            arm = row["arm"]
            if arm == "frozen_no_write":
                continue
            pools.setdefault(arm, [])
            gradients.setdefault(arm, None)
            states.setdefault(arm, m.prior.old.coefficients(data["head"]).tolist())
            if kind == "release" and row["eligible"]:
                pools[arm].append(row)
            if kind == "gradient":
                gradients[arm] = row
            if kind == "acceptance":
                if (
                    max(
                        abs(a - b)
                        for a, b in zip(states[arm], row["before_coefficients"], strict=True)
                    )
                    > 1e-10
                ):
                    raise ValueError("original_state_chain")
                gradient = gradients[arm] if row["reason"].startswith("gradient/") else None
                w = dict(
                    identity=identity,
                    arm=arm,
                    seed=row["seed"],
                    slot=row["slot"],
                    natural=True,
                    **{
                        "class": "reset"
                        if row["reset"]
                        else "rejected"
                        if row["rejected"]
                        else "accepted"
                    },
                    before=row["before_coefficients"],
                    delta=[
                        a - b
                        for a, b in zip(
                            row["proposed_coefficients"], row["before_coefficients"], strict=True
                        )
                    ],
                    updates=[gradient["origin_slot"]] if gradient else [],
                    labels=[gradient["y"]] if gradient else [],
                    guards=[
                        [r["origin_slot"], r["y"]]
                        for r in pools[arm]
                        if r["feedback_role"] == "guard"
                    ],
                    released=pools[arm].copy(),
                    reason=row["reason"],
                    expected={
                        k: row[k]
                        for k in (
                            "alpha",
                            "parameters",
                            "diagnostics",
                            "reset",
                            "rejected",
                            "status",
                        )
                    },
                )
                cases.append(w)
                states[arm] = row["parameters"]
        db.close()
    return cases


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Adapt shipped checks, retaining one bounded whole-repository observation."""
    commands = prior.validation_plan(scratch)
    (scratch / "coverage.ini").write_text(
        "[run]\nparallel=True\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    result = []
    for c in commands:
        argv = tuple(a.replace(prior.NAME, NAME).replace(prior.TEST, TEST) for a in c.argv)
        if c.name in ("ruff_check", "ruff_format", "strict_mypy"):
            argv += ("python/carnot/verify/guarded_transaction_8053.py",)
        result.append(
            replace(
                c,
                argv=argv,
                timeout_s=300 if c.name == "unit_consumers_e2e015_019" else min(c.timeout_s, 120),
            )
        )
    return result


def validate(raw: Path, scratch: Path) -> tuple[list[Json], Json, list[Json]]:
    """Measure only added statements; unrelated health never changes owned gates."""
    receipts = run_commands(
        ROOT,
        validation_plan(scratch),
        log_dir=raw / "validation_logs",
        heartbeat_s=10,
        extra_env=dict(
            CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(scratch),
            CARNOT_8019_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    file = scratch / "coverage.json"
    counts = (
        {k: v["summary"] for k, v in json.loads(file.read_text())["files"].items()}
        if file.exists()
        else {}
    )
    atomic_json(raw / "coverage.json", dict(files=counts))
    return (
        [r for r in receipts if r["scope"] == "owned"],
        counts,
        [r for r in receipts if r["scope"] == "repository_health"],
    )


def write_data(raw: Path, data: Json) -> Json:
    """Keep case shards small while binding complete original operands."""
    refs = []
    for i in range(0, len(data["cases"]), 100):
        p = raw / "cases" / f"{i:04d}.json"
        atomic_json(p, dict(cases=data["cases"][i : i + 100]))
        refs.append(reference(p))
    saved = {k: v for k, v in data.items() if k != "cases"}
    saved["case_references"] = refs
    atomic_json(raw / "inputs.json", saved)
    return saved


def replay(path: Path) -> Json:
    """Fresh reduction checks raw timings and independently repeats original parity."""
    v = json.loads(path.read_text())
    if v["native_transaction_ready_score"] and (
        v["verdict_class"] in ("blocked", "disqualified")
        or not all(v["acceptance_gate_results"].values())
    ):
        raise ValueError("unsafe_readiness")
    for ref in v["raw_shard_hashes"] + v["code_config_hashes"] + v["checkpoint_references"]:
        checked(ref)
    if v["acceptance_gate_results"]["measurement"]:
        raw = Path(v["raw_directory"])
        data = json.loads((raw / "inputs.json").read_text())
        data["cases"] = [
            w
            for ref in data["case_references"]
            for w in json.loads(checked(ref).read_text())["cases"]
        ]
        native = prior.old.build.load_native_extension(checked(v["loaded_library_receipt"]))
        rows = json.loads((raw / "transaction_rows.json").read_text())["rows"]
        for k, observed in m.summarize(rows, v["config"]).items():
            if v[k] != observed:
                raise ValueError("reduction_drift." + k)
        with tempfile.TemporaryDirectory(prefix="carnot-8053-cold-") as temp:
            cold = m.parity(data, native, Path(temp))
        if cold["parity_rows"] != v["parity_rows"] or not cold["parity_passed"]:
            raise ValueError("cold_parity_drift")
        index = {canonical_hash(w): w for w in data["cases"] + m.fixtures(data["head"])}
        for row in rows:
            w = index[row["input_hash"]]
            p = json.loads(checked(row["checkpoint"]).read_bytes())
            expected = m.calculate(data, w, native, "python", reextract=False)["result"]
            if (
                not m.comparison(expected, p["acceptance"])["passed"]
                or p["replay_pool"] != w["released"]
            ):
                raise ValueError("checkpoint_drift")
            if row["transaction_ns"] != sum(row["components"].values()):
                raise ValueError("span_drift")
    return dict(passed=True)


def terminal(path: Path) -> Json:
    """Check candidate and published bytes with fresh processes and strict consumers."""
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
        log_dir=raw / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=10,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def base(failures: list[Json]) -> Json:
    """Numerical readiness is independent of a scientific or service speed win."""
    v = prior.base(failures)
    v.update(
        experiment_id=8053,
        task_id=TASK,
        milestone="2026.10.697",
        schema="carnot.v697.guarded_transaction_cost.v1",
        run_date="20261003",
        honest_verdict="complete_blocked_guarded_transaction_inputs"
        if failures
        else "complete_null_guarded_transaction_cost",
        config=copy.deepcopy(m.CONFIG),
        random_seed=m.CONFIG["seed"],
        parity_rows=[],
        transaction_class_rows=[],
        rejected_work_cost=[],
        complete_numerical_transaction_speedup=[],
        loaded_extension_hash=None,
        preconditions_checked=[],
        measured_service_scope="Cached public features and released feedback through all guard decisions, synchronous journal/checkpoint and restart. No LLM or external feedback-provider request is measured.",
        claim_scope="This invocation measures exposed-development guarded numerical transactions only. No full verification-service, generalized learning, deployment safety or hardware speed claim.",
        binding_support="Existing RustNumericalUpdate8027 design/update_batch/effective/state_json; policy arithmetic and serialization remain Python and are charged.",
        timing_component_principles=dict(
            gradient_arithmetic_ns="The shipped per-step sparse coefficient timer.",
            gradient_construction_ns="Batch arithmetic time minus coefficient update time; includes calibrated probability and gradient construction.",
            ffi_ns="Outer gradient and guard call time minus Rust-reported arithmetic; includes Python conversion, state construction and call bookkeeping.",
            design_and_initialization_ns="Complete design and state construction; includes design FFI, which has no separate shipped inner timer.",
            guard_scans_ns="All guard policy and probability work excluding measured guard binding overhead.",
            storage_fsync_ns="Synchronous SQLite journal and checkpoint write, file fsync and directory fsync.",
            reload_ns="Reopen checkpoint and journal and reconstruct path state.",
            feature_gather_ns="Actual extraction from public source/answer bytes on every measured transaction.",
            orchestration_ns="Unassigned exclusive elapsed time within the transaction.",
        ),
        ops_docs_updated=False,
    )
    return v


def main(argv: list[str] | None = None) -> int:
    """Freeze evidence, run owned acceptance and atomically publish one terminal result."""
    m.progress("start_preconditions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.cold_replay:
        try:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        except (OSError, KeyError, ValueError) as error:
            print(json.dumps(dict(passed=False, error=str(error))), flush=True)
            return 1
    output = (args.output or artifact_output_root(root=args.root) / (NAME + ".json")).absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="carnot-8053-") as temp:
        scratch = Path(temp)
        data, failures = (
            (json.loads(args.fixture_input.read_text()), [])
            if args.fixture_input
            else load_inputs(args.root, raw)
        )
        v = base(failures)
        v.update(
            raw_directory=str(raw),
            cited_upstream_artifacts=data.get("references", []),
            gate_check_summary=data.get("gate_checks", []) + failures,
            preconditions_checked=data.get("gate_checks", []),
        )
        dependencies = [
            *OWNED,
            TEST,
            "openspec/capabilities/research-reporting/spec.md",
            "python/carnot/verify/feedback_constrained_8051.py",
            "python/carnot/verify/causal_online_8025.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/experiment_8027_v695_native_update_cost.py",
            "python/carnot/experiment_8040_v696_native_transaction_cost.py",
            "crates/carnot-python/src/lib.rs",
            "crates/carnot-python/src/numerical_update_8027.rs",
            "crates/carnot-core/src/numerical_update_8027.rs",
            "scripts/experiment_template.py",
        ]
        v["code_config_hashes"] = [
            prior.sealed.copy_bound(reference(ROOT / p), raw, "code") for p in dependencies
        ]
        commands = validation_plan(scratch)
        atomic_json(
            raw / "methods.json",
            dict(
                config=m.CONFIG,
                identity=TASK,
                commands=[asdict(c) for c in commands],
                fixture=bool(args.fixture_input),
                artifact_guard_enabled=True,
                model_loads=0,
                guard=m.learning.protocol.METHODS["guard"],
                optimizer=prior.old.causal.CONFIG,
                acquisition="Optional named Qwen phase receipts absent; no cost imported",
                measured_boundary=v["measured_service_scope"],
            ),
        )
        v["code_config_hashes"].append(reference(raw / "methods.json"))
        first = Path("/tmp/carnot-8053-first")
        if first.is_dir():
            shutil.copytree(first, raw / "preimplementation_checks")
        m.progress("freeze_after")
        if not failures:
            v["sample_size_budget"] = dict(
                natural_transactions=len(data["cases"]),
                intended_pairs_per_class=m.CONFIG["repetitions"],
                excluded_warmup_pairs=m.CONFIG["warmups"],
                numerical_budget_s=m.CONFIG["budget_s"],
                timing_repeats_independent=False,
                seeds_independent=False,
            )
            reg = prior.regression(data["library_reference"], raw, scratch)
            v["mapped_inode_regression"] = reg
            v["acceptance_gate_results"]["regression"] = reg["passed"]
            if reg["passed"]:
                native, receipt = prior.load_library(data["library_reference"], raw)
                v.update(loaded_library_receipt=receipt, loaded_extension_hash=receipt["sha256"])
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
                v.update(
                    validation_receipts=receipts,
                    coverage_statement_counts=counts,
                    repository_health=health,
                )
                v["acceptance_gate_results"]["owned_checks"] = good
                write_data(raw, data)
                m.progress("measurement_before")
                try:
                    v.update(m.parity(data, native, raw))
                    v["positive_control_results"] = dict(
                        scaling=m.scaling(data["head"], native),
                        boundaries=prior.controls(data["head"], native),
                    )
                    v.update(m.measure(data, native, raw))
                    v["acceptance_gate_results"].update(
                        parity=v["parity_passed"]
                        and all(r["passed"] for r in v["positive_control_results"].values()),
                        measurement=True,
                    )
                except (ValueError, TimeoutError) as error:
                    v["gate_check_summary"].append(
                        prior.sealed.Contract(
                            raw, "measurement_contract", "complete valid measurement", str(error)
                        ).gate
                    )
                measured = [r for r in v["rows"] if not r["excluded"]]
                v.update(
                    intended_count=len(
                        {(r["condition"], r["transaction_class"], r["natural"]) for r in v["rows"]}
                    )
                    * m.CONFIG["repetitions"]
                    * 2,
                    eligible_count=len(measured),
                    completed_count=len(measured),
                    excluded_count=len(v["rows"]) - len(measured),
                    independent_count=len(
                        {r["source_cluster_id"] for r in data["sources"] if r["public_eligible"]}
                    )
                    if not args.fixture_input
                    else 0,
                    checkpoint_references=[
                        ref for r in v["rows"] for ref in (r["checkpoint"], r["journal"])
                    ],
                    trained_head_specs=[
                        dict(
                            pretrained=False,
                            parameters=110,
                            operation="replay of frozen calibrated head; no benefit credit",
                        )
                    ],
                )
                m.progress("measurement_after", len(measured))
            if not args.fixture_input and not all(v["acceptance_gate_results"].values()):
                v.update(
                    honest_verdict="complete_disqualified_guarded_transaction_cost",
                    verdict_class="disqualified",
                )
            v["native_transaction_ready_score"] = int(
                not args.fixture_input and all(v["acceptance_gate_results"].values())
            )
        v["raw_shard_hashes"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
        v["reproducibility_checksum"] = canonical_hash(
            dict(
                config=m.CONFIG, code=v["code_config_hashes"], inputs=v["cited_upstream_artifacts"]
            )
        )
        v["duration_s"] = time.monotonic() - started
        v["phase_spans"] = [
            dict(phase="owned_validation_and_guarded_cpu_work", duration_s=v["duration_s"])
        ]
        v["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
        v["field_principles"] = {
            k: "Bind "
            + k
            + " to this invocation, exact bytes and the numerical-only evidence boundary."
            for k in v
        }
        v["field_principles"].update(
            native_transaction_ready_score="Qualified owned checks and measured parity determine readiness; speed wins are not required.",
            rejected_work_cost="Rejected gradients and all guard scans consume charged complete transaction time.",
            complete_numerical_transaction_speedup="Summed paired durable costs define the ratio; kernel time cannot substitute for useful work.",
            independent_count="Unique original source clusters; duplicate calls and seeds add no independent support.",
        )
        m.progress("publication_before")
        publication = publish_primary(output, v, terminal)
        report = terminal(output)
        reader = reader_receipt(
            TASK,
            output.parent,
            field="native_transaction_ready_score",
            expected=v["native_transaction_ready_score"],
        )
        atomic_json(
            raw / "terminal_validation.json",
            dict(publication=publication, published=report, reader=reader, process_exit_required=0),
        )
        m.progress("publication_after", 1)
        return 0 if report["passed"] and reader["passed"] else 1
