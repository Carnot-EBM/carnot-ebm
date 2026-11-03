"""REQ-REPORT-8225: bind causal trajectory readiness to normal-exit evidence.

This producer does not measure H2 or open retention outcomes. A fully executed
trajectory can end without any admitted change; readiness describes the causal
work and recovery, rather than a scientific improvement on exposed sources.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import delayed_utility_learning_8225 as k
from carnot.verify import utility_kernel_qualification_8221 as qualified
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME = "experiment_8225_v711_delayed_utility_learning"
TASK = "exp8225-delayed-utility-learning"
MODULE = "python/carnot/verify/delayed_utility_execution_8225.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_delayed_utility_learning_8225.py"
OWNED = ["python/carnot/verify/delayed_utility_learning_8225.py", MODULE, CLI, qualified.KERNEL]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
run_check = qualified.run_check
reference = qualified.reference
BASE_MANIFEST = qualified.BASE_MANIFEST


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real counts distinguish cached CPU work from an unobserved model invocation."""
    print(f"[exp8225] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path, raw: Path, work: Json) -> Json:
    """Current terminal receipts and frozen primitive hashes authorize this branch only."""
    gate, bind = k.frozen.gate, k.frozen.bind
    bind(work, root / qualified.PROTOCOL, qualified.PIN, raw)
    bind(work, root / k.calibration.PROTOCOL, k.calibration.PROTOCOL_HASH, raw)
    values = []
    for name, field in [
        (qualified.NAME, "causal_kernel_ready_score"),
        (k.durable.NAME, "learning_trajectory_ready_score"),
    ]:
        path = root / "results" / (name + ".json")
        gate(work, path, "exists", True, True if path.is_file() else None)
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict):
            gate(work, path, "input_schema", "object", type(value).__name__)
        for key, expected in [
            ("experiment_id", int(name.split("_")[1])),
            (field, 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            gate(work, path, key, expected, value.get(key))
        terminal = Path(value["terminal_validation_sidecar_path"])
        terminal = root / terminal.relative_to(ROOT)
        bind(work, terminal, sha256_file(terminal), raw)
        side = json.loads(terminal.read_bytes())
        receipt = root / Path(side["publication"]["sidecar_path"]).relative_to(ROOT)
        gate(
            work,
            path,
            "terminal.report.passed",
            True,
            read_bound_sidecar(path, receipt)["report"]["passed"],
        )
        for source in [path, receipt]:
            bind(work, source, sha256_file(source), raw)
        values.append(value)
    gate(
        work,
        root / qualified.PROTOCOL,
        "protocol_sha256",
        qualified.PIN,
        values[0].get("protocol_sha256"),
    )
    manifests = values[1]["input_manifests"]
    for role in ["stream", "retention"]:
        for ref in [
            manifests[role + "_feature_manifest"],
            manifests["evaluator_label_manifests"][role],
        ]:
            bind(work, Path(ref["path"]), ref["sha256"], raw)
    bind(
        work,
        ROOT / "ops/exclusion_manifest.yaml",
        sha256_file(ROOT / "ops/exclusion_manifest.yaml"),
        raw,
    )
    return dict(manifests=manifests)


def evidence(states: list[Json], rows: list[Json], retained: list[Json], raw: Path) -> Json:
    """Issue rows, outcomes and future action changes remain primitive audit operands."""
    learning, scored, updates, releases, changes, timing, predictions = [], [], [], [], [], [], []
    for states_by_arm in states:
        for arm, state in states_by_arm.items():
            seed = state["seed"]
            directory = raw / f"seed-{seed}" / arm
            if not directory.exists():
                directory = raw / arm
            delivered = k.journal(directory / "released.jsonl")
            labels = {r["label_slot"]: r["outcome"] for r in delivered}
            releases.extend(dict(r, arm=arm, seed=seed) for r in delivered)
            timing.extend(
                dict(r, arm=arm, seed=seed) for r in k.journal(directory / "timing.jsonl")
            )
            installed = [v for v in state["events"] if v["kind"] == "admit_once" and v["step"]]
            updates.extend(
                dict(v, arm=arm, seed=seed)
                for v in state["events"]
                if v["kind"] in ["commit_candidate", "admit_once", "defer_candidate"]
            )
            for row, issued in zip(rows, state["issued"], strict=True):
                learning.append(
                    dict(
                        issued,
                        arm=arm,
                        seed=seed,
                        unit_id=row["unit_id"],
                        source_cluster_id=row["source_cluster_id"],
                        baseline_action=row["baseline_action"],
                        missing=row["p"] is None,
                    )
                )
                if (
                    installed
                    and issued["slot"] > installed[0]["slot"]
                    and issued["action"] != k.kernel.rule.action(row["p"], row["baseline_action"])
                ):
                    changes.append(
                        dict(
                            slot=issued["slot"],
                            arm=arm,
                            seed=seed,
                            action=issued["action"],
                            frozen_action=k.kernel.rule.action(row["p"], row["baseline_action"]),
                            install_slot=installed[0]["slot"],
                        )
                    )
                if row["slot"] < 65:
                    continue
                target = labels.get(row["slot"], {})
                y = target.get("y")
                status = (
                    "censored"
                    if row["slot"] > 236
                    else "excluded"
                    if y is None or issued["p"] is None
                    else "completed"
                )
                for metric in ["cost", "brier", "false_accept"]:
                    numerator = (
                        None
                        if status != "completed"
                        else k.kernel.rule.base.loss(issued["action"], y)
                        if metric == "cost"
                        else (issued["p"] - y) ** 2
                        if metric == "brier"
                        else int(issued["action"] == "accept" and y == 1)
                    )
                    scored.append(
                        dict(
                            unit_id=row["unit_id"],
                            source_cluster_id=row["source_cluster_id"],
                            slot=row["slot"],
                            arm=arm,
                            seed=seed,
                            condition="later_stream",
                            metric=metric,
                            numerator=numerator,
                            denominator=1,
                            status=status,
                            exclusion_reason=None
                            if status == "completed"
                            else "feedback_unresolved_tail"
                            if status == "censored"
                            else "missing_operand",
                            y=y,
                            p=issued["p"],
                            action=issued["action"],
                        )
                    )
            for row in retained:
                before = time.process_time_ns()
                p = k.predict(state["model"], row)
                cpu = time.process_time_ns() - before
                predictions.append(
                    dict(
                        slot=row["slot"],
                        unit_id=row["unit_id"],
                        source_cluster_id=row["source_cluster_id"],
                        seed=seed,
                        arm=arm,
                        p=p,
                        action=k.kernel.rule.action(p, row["baseline_action"]),
                        missing=p is None,
                        state_sha256=canonical_hash(state),
                        labels_opened=False,
                    )
                )
                timing.append(
                    dict(kind="retention_lookup", slot=row["slot"], seed=seed, arm=arm, cpu_ns=cpu)
                )
    return dict(
        learning_rows=learning,
        rows=scored,
        update_log=updates,
        release_log=releases,
        later_decision_changes=changes,
        update_timing_rows=timing,
        retention_predictions=predictions,
        memory_bytes=sum(len(json.dumps(s, sort_keys=True).encode()) for s in states),
    )


def restart_specs(raw: Path) -> list[Json]:
    """Fix the five real child commands before any natural fit or admission occurs."""
    directory = raw / "child_coverage"
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "coverage.ini"
    config.write_text(
        "[run]\npatch = _exit\nparallel = true\ndata_file = "
        + str(directory / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
        + "[report]\nexclude_lines =\n"
    )
    base = [
        "/usr/bin/env",
        "-u",
        "PYTHONPATH",
        "-u",
        "COVERAGE_PROCESS_START",
        "-u",
        "COVERAGE_FILE",
        str(ROOT / ".venv/bin/python"),
        "-u",
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(config),
        str(ROOT / CLI),
        "--seed-input",
        str(raw / "restart-input.json"),
    ]
    return [
        dict(
            name=name,
            argv=base + ["--seed-output", str(raw / folder)] + args,
            deadline_s=120,
            expected_exit=code,
            classification="required",
        )
        for name, folder, args, code in [
            ("uninterrupted", "uninterrupted", [], 0),
            ("crash90", "restart90", ["--crash-slot", "90"], 73),
            ("resume90", "restart90", ["--resume-state", str(raw / "restart90/crash.json")], 0),
            ("crash170", "restart170", ["--crash-slot", "170"], 73),
            ("resume170", "restart170", ["--resume-state", str(raw / "restart170/crash.json")], 0),
        ]
    ]


def measure(root: Path, raw: Path, *, fixture: bool = False, **kwargs: Any) -> Json:
    """Authenticate inputs, execute finite trajectories, and keep retention labels sealed."""
    start, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        owned_failure="",
        input_ready=0,
        states=[],
        rows=[],
        learning_rows=[],
        release_log=[],
        update_log=[],
        later_decision_changes=[],
        update_timing_rows=[],
        memory_bytes=0,
        retention_predictions=[],
        child_exit_rows=[],
        restart_state_hashes=[],
        input_manifests={},
        trajectory_path=None,
        final_states_path=None,
        retention_predictions_path=None,
        timing_rows_path=None,
        fixture_mode=fixture,
    )
    work["code_config_hashes"] = []
    for name in [
        *OWNED,
        TEST,
        qualified.KERNEL,
        k.calibration.MODULE,
        qualified.PROTOCOL,
        k.calibration.PROTOCOL,
    ]:
        path = ROOT / name
        snapshot = k.frozen.bind(dict(checks=[], refs=[]), path, sha256_file(path), raw)
        work["code_config_hashes"].append(dict(reference(path), snapshot_path=str(snapshot)))
    progress("before_preconditions")
    try:
        with TemporaryDirectory(prefix="carnot-8225-probe-") as temporary:
            probe = Path(temporary) / "probe"
            probe.write_bytes(b"private writable scratch")
            k.frozen.gate(
                work,
                probe,
                "private_scratch_writable",
                True,
                probe.read_bytes() == b"private writable scratch",
            )
        for name in ["python", "pytest", "coverage", "ruff", "mypy"]:
            path = ROOT / ".venv/bin" / name
            k.frozen.gate(work, path, "runtime_" + name, True, path.is_file())
        k.frozen.gate(
            work, raw, "storage_available", True, shutil.disk_usage(raw).free > 32 * 1024 * 1024
        )
        inputs = authenticate(root, raw, work)
        work["input_manifests"] = inputs["manifests"]
        if fixture:
            rows, path, retained = k.fixture(raw / "private_fixture")
        else:
            rows = json.loads(
                Path(inputs["manifests"]["stream_feature_manifest"]["path"]).read_bytes()
            )["rows"]
            retained = json.loads(
                Path(inputs["manifests"]["retention_feature_manifest"]["path"]).read_bytes()
            )["rows"]
            k.calibration.engine.historical.public_rows(rows, 256)
            k.calibration.engine.historical.public_rows(retained, 64)
            rows, retained = k.prepare(rows), k.prepare(retained)
            path = Path(inputs["manifests"]["evaluator_label_manifests"]["stream"]["path"])
        atomic_json(
            raw / "restart-input.json", dict(rows=rows, label_path=str(path), retained=retained)
        )
        specs = restart_specs(raw)
        atomic_json(raw / "restart_commands.json", dict(commands=specs))
        work["input_ready"] = 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="input_schema",
                    path=str(root),
                    upstream=str(root),
                    hash=None,
                    artifact_field="input_schema",
                    op="==",
                    expected="valid",
                    observed=str(error),
                    passed=False,
                )
            )
    progress("after_preconditions", len(work["checks"]))
    if work["input_ready"]:
        try:
            for seed in [101] if fixture else range(101, 121):
                if (time.monotonic_ns() - start) / 1e9 > k.BUDGET_S:
                    raise TimeoutError("cpu_science_budget")
                progress("before_natural_benchmark", seed - 101, 121 - seed)
                work["states"].append(k.run(rows, path, seed, raw / f"seed-{seed}"))
                progress("after_natural_benchmark", seed - 100, 120 - seed)
            work.update(evidence(work["states"], rows, retained, raw))
            for name, value in [
                ("trajectory", dict(rows=rows, learning_rows=work["learning_rows"])),
                ("final_states", work["states"]),
                ("retention_predictions", work["retention_predictions"]),
                ("timing_rows", work["update_timing_rows"]),
            ]:
                output = raw / (name + ".json")
                atomic_json(output, value)
                work[name + "_path"] = str(output)
            progress("final_states_and_retention_predictions_sealed", len(work["states"]))
            for spec in specs:
                progress("before_subprocess_" + spec["name"])
                work["child_exit_rows"].append(
                    run_check(ROOT, spec, raw, raw / "restart_logs", heartbeat_s=20)
                )
                progress("after_subprocess_" + spec["name"])
            baseline = json.loads((raw / "uninterrupted/final.json").read_bytes())
            for slot in [90, 170]:
                resumed = json.loads((raw / f"restart{slot}/final.json").read_bytes())
                saved = json.loads((raw / f"restart{slot}/crash.json").read_bytes())
                passed = (
                    baseline == resumed == work["states"][0]
                    and saved["global_plus_group"]["cursor"] == slot
                    and bool(saved["global_plus_group"]["pending"])
                )
                for arm in k.ARMS:
                    for journal in ["issued", "released"]:
                        passed = passed and k.journal(
                            raw / "uninterrupted" / arm / (journal + ".jsonl")
                        ) == k.journal(raw / f"restart{slot}" / arm / (journal + ".jsonl"))
                work["restart_state_hashes"].append(
                    dict(
                        crash_slot=slot,
                        passed=passed,
                        baseline_sha256=canonical_hash(baseline),
                        resumed_sha256=canonical_hash(resumed),
                        saved_sha256=canonical_hash(saved),
                        pending_count=len(saved["global_plus_group"]["pending"]),
                    )
                )
            parent = os.environ.get("COVERAGE_RCFILE")
            if parent:
                for shard in (raw / "child_coverage").glob(".coverage.*"):
                    shutil.copyfile(
                        shard, Path(parent).parent / (shard.name + "-8225-" + str(time.time_ns()))
                    )
        except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
            work["owned_failure"] = str(error)
    work["raw_shard_hashes"] = [
        reference(p)
        for p in sorted(raw.rglob("*"))
        if p.is_file()
        and p.name not in ["measurement.json", "measurement.stdout", "measurement.stderr"]
    ]
    work.update(
        duration_s=(time.monotonic_ns() - start) / 1e9,
        clock=dict(
            started_monotonic_ns=start, ended_monotonic_ns=time.monotonic_ns(), started_wall_ns=wall
        ),
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["states"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Only complete execution and exact recovery grant readiness; H2 stays unmeasured."""
    failures = [r for r in work["checks"] if not r["passed"]]
    owned = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    complete = (
        len(work["states"]) == (1 if fixture else 20)
        and len(work["learning_rows"]) == len(work["states"]) * 256 * 5
        and len(work["retention_predictions"]) == len(work["states"]) * 64 * 5
        and len(work["restart_state_hashes"]) == 2
        and all(r["passed"] for r in work["restart_state_hashes"] + work["child_exit_rows"])
    )
    owned = bool(owned and (failures or complete))
    verdict = "disqualified" if not owned else "blocked" if failures else "null"
    units = [
        r
        for r in work["rows"]
        if r["seed"] == 101 and r["arm"] == "frozen" and r["metric"] == "brier"
    ]
    value = dict(
        work,
        experiment_id=8225,
        task_id=TASK,
        milestone="2026.10.711",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (
            Path(failures[0]["path"]).stem
            if verdict == "blocked"
            else "owned_validation"
            if verdict == "disqualified"
            else "utility_trajectory_benefit_reserved_for_8226"
        ),
        verdict_class=verdict,
        utility_trajectory_ready_score=int(owned and not failures and complete),
        required_checks_passed=owned,
        gate_check_summary=work["checks"],
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if work["input_ready"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=dict(
            kind="global_scale_intercept_and_ordered_probability_patches",
            arms=k.ARMS[1:],
            maximum_patches_per_opportunity=4,
            maximum_total_patch_operations=8,
            optimizer=k.calibration.PROTOCOL_HASH,
        ),
        intended_count=192,
        completed_count=sum(r["status"] == "completed" for r in units),
        failed_count=0,
        censored_count=sum(r["status"] == "censored" for r in units),
        excluded_count=sum(r["status"] == "excluded" for r in units) if units else 192,
        independent_count=0
        if fixture
        else len({r["source_cluster_id"] for r in units if r["status"] == "completed"}),
        verifier_is_oracle=True,
        exposure_scope="private_oracle_mechanics"
        if fixture
        else "historically_exposed_development_stream",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        flagged_adversarial=False,
        acceptance_gates=dict(
            input_authentication=not failures,
            owned_validation=owned,
            causal_execution=complete,
            benefit="H2 and retention verdicts reserved for Exp8226",
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["checks"],
        random_seed=101,
        source_artifact_hashes=work["refs"],
        phase_spans=[
            dict(phase="authenticate_and_execute", duration_s=work["duration_s"], **work["clock"])
        ],
        cited_upstream_artifacts=[
            dict(
                path=r.get("upstream_path", r["path"]),
                sha256=r["sha256"],
                fields_imported=[
                    "causal readiness and terminal binding"
                    if "8221" in r.get("upstream_path", "")
                    else "frozen protocol, original features, opaque stream-label fragments or baseline permission"
                ],
            )
            for r in work["refs"]
        ],
        retention_labels_opened=False,
        H2=dict(qualified.PROTOCOL_VALUE["H2"], measured_here=False),
        claim_scope="Causal persistent corrections and exact recovery on exposed development only; no learning benefit established.",
        methodology_note="No model loaded or called. The original stream, delayed roles and frozen dictionary govern all five arms. Missing rows remain original slots; future admission and retention outcomes never fit corrections. CPU costs are measured, not accelerator performance claims.",
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        key: "Bind actual causal execution, original missing masks and normal-exit receipts; readiness supplies no scientific benefit."
        for key in value
    }
    value["field_principles"].update(
        utility_trajectory_ready_score="Complete causal issue/release and exact crash recovery regardless of outcome.",
        later_decision_changes="Only changed actions strictly after admission count; probability movement alone supplies no benefit.",
        update_timing_rows="Measured process CPU nanoseconds, with original update IDs and missing masks.",
        retention_predictions_path="Sealed final-state predictions; retention labels remain unopened for Exp8226.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reexecute authenticated primitives so a recomputed headline hash cannot hide edits."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if canonical_hash(value) != checksum:
            return False
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        for ref in [
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if (
                sha256_file(Path(ref["path"])) != ref["sha256"]
                or "snapshot_path" in ref
                and sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]
                or "upstream_path" in ref
                and sha256_file(Path(ref["upstream_path"])) != ref["sha256"]
            ):
                return False
        work = json.loads((raw / "measurement.json").read_bytes())
        for receipt in value["validation_receipts"] + value["child_exit_rows"]:
            for prefix in ["stdout", "stderr"]:
                if (
                    prefix + "_path" in receipt
                    and sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]
                ):
                    return False
        if work["input_ready"] and not work["owned_failure"]:
            inputs = json.loads((raw / "restart-input.json").read_bytes())
            with TemporaryDirectory(prefix="carnot-8225-replay-") as temporary:
                rebuilt = [
                    k.run(
                        inputs["rows"],
                        Path(inputs["label_path"]),
                        s["frozen"]["seed"],
                        Path(temporary) / f"seed-{s['frozen']['seed']}",
                    )
                    for s in work["states"]
                ]
                actual = evidence(rebuilt, inputs["rows"], inputs["retained"], Path(temporary))
            if rebuilt != work["states"]:
                return False
            for key in [
                "learning_rows",
                "rows",
                "update_log",
                "release_log",
                "later_decision_changes",
                "retention_predictions",
                "memory_bytes",
            ]:
                if actual[key] != work[key]:
                    return False
            if (
                json.loads(Path(work["final_states_path"]).read_bytes()) != rebuilt
                or json.loads(Path(work["retention_predictions_path"]).read_bytes())
                != actual["retention_predictions"]
                or json.loads(Path(work["trajectory_path"]).read_bytes())
                != dict(rows=inputs["rows"], learning_rows=actual["learning_rows"])
                or json.loads(Path(work["timing_rows_path"]).read_bytes())
                != work["update_timing_rows"]
            ):
                return False
        return build(work, raw, value["validation_receipts"], fixture=work["fixture_mode"]) == dict(
            value, reproducibility_checksum=checksum
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze scoped typing, coverage and private E2E commands before measurement."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("[run]", "[run]\npatch = _exit")
        + "[report]\nexclude_lines =\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    for spec in specs["commands"]:
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["argv"].extend(
                [
                    "tests/python/test_utility_kernel_8221.py",
                    "--basetemp=" + str(private / "owned_pytest"),
                ]
            )
            spec["deadline_s"] = 600
        if spec["name"] == "consumer_and_E2E015_019":
            begin = spec["argv"].index("tests/python/test_development_methods_8098.py")
            spec["argv"][begin:] = [
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
                "tests/python/test_hard_exit_learning_qualification_8206.py",
                "--basetemp=" + str(private / "consumer_pytest"),
            ]
            spec["deadline_s"] = 900
        if spec["name"] == "strict_mypy":
            spec["argv"] = [
                a.replace("--follow-imports=silent", "--follow-imports=skip") for a in spec["argv"]
            ]
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    specs["repository_health"]["deadline_s"] = 180
    specs["repository_health"]["argv"][:0] = [
        "/usr/bin/env",
        "COVERAGE_FILE=" + str(private / "repository_health.coverage"),
    ]
    return specs


def main(argv: list[str] | None = None) -> int:
    """The direct CLI shares the same causal runner in private and real children."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--seed-input" in args:
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed-input", type=Path, required=True)
        parser.add_argument("--seed-output", type=Path, required=True)
        parser.add_argument("--resume-state", type=Path)
        parser.add_argument("--crash-slot", type=int, default=0)
        parsed = parser.parse_args(args)
        inputs = json.loads(parsed.seed_input.read_bytes())
        state = json.loads(parsed.resume_state.read_bytes()) if parsed.resume_state else None
        k.run(
            inputs["rows"],
            Path(inputs["label_path"]),
            101,
            parsed.seed_output,
            state=state,
            crash_slot=parsed.crash_slot,
        )
        return 0
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(args))
