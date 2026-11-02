"""REQ-REPORT-8306: publish primitive-bound constructed numeric evidence.

Readiness certifies local mechanics and recovery. Timing and reused cached model
provenance cannot establish natural benefit or independent generalization.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify.hard_exit_learning_qualification_8206 import run_check
from carnot.verify import local_update_isolation_8306 as k

Json = dict[str, Any]
ROOT, NAME, CLI = k.ROOT, k.NAME, k.CLI
TASK = "exp8306-local-update-isolation"
MODULE = "python/carnot/reporting/local_update_isolation_8306.py"
RUNNER = "python/carnot/reporting/local_update_execution_8306.py"
TEST = "tests/python/test_local_update_isolation_8306.py"
OWNED = ["python/carnot/verify/local_update_isolation_8306.py", MODULE, RUNNER, CLI]
MODEL_SPECS: list[str] = []
progress = k.progress
PINS = {
    "experiment_8291_v716_dependency_scoped_admission": "4239c5f9c476219731a957f02474556aa47e699d0a67b77a8443d4d6f7757843",
    "experiment_8304_v717_contract_methods": "53c4286349b038c34b994dd339e871bc8baca8477c7fb819aac1ab18ebe7a292",
    "experiment_8305_v717_cached_sentence_custody": "84564c8702db3a3665c84e9db9f62d6a2bffe00729cc19a19bb4c3a6c2ea13ba",
}


def reference(path: Path) -> Json:
    """Bind one source or receipt to its actual immutable bytes."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def authenticate(root: Path, raw: Path) -> Json:
    """Check external operands before numeric work; absent evidence stays absent."""
    plan: Json = dict(checks=[], refs=[], historical=[])

    def require(path: Path, field: str, expected: Any, observed: Any) -> None:
        plan["checks"].append(
            dict(
                upstream=path.stem,
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                artifact_field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=expected == observed,
            )
        )
        if expected != observed:
            raise ValueError(field)

    def bind(path: Path) -> Json:
        dest = raw / "inputs" / (sha256_file(path)[7:] + "-" + path.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        plan["refs"].append(reference(dest))
        return dict(json.loads(dest.read_text()))

    try:
        raw.mkdir(parents=True, exist_ok=True)
        raw.chmod(0o700)
        require(
            raw,
            "private_scratch",
            True,
            os.access(raw, os.W_OK) and raw.stat().st_mode & 0o077 == 0,
        )
        require(raw, "available_disk_at_least_1GiB", True, shutil.disk_usage(raw).free >= 1024**3)
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
            require(
                ROOT / ".venv/bin" / tool,
                "tool_executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        for name, pin in PINS.items():
            path = root / "results" / (name + ".json")
            require(
                path, "input_sha256", "sha256:" + pin, sha256_file(path) if path.is_file() else None
            )
            value = bind(path)
            require(path, "required_checks_passed", True, value.get("required_checks_passed"))
            require(path, "flagged_adversarial", False, value.get("flagged_adversarial"))
            terminal_path = Path(value["terminal_validation_sidecar_path"])
            require(terminal_path, "exists", True, terminal_path.is_file())
            terminal = bind(terminal_path)
            side = Path(terminal["publication"]["sidecar_path"])
            require(side, "exists", True, side.is_file())
            attestation = read_bound_sidecar(path, side)
            require(side, "report.passed", True, attestation["report"]["passed"])
            require(
                terminal_path,
                "publication.primary_sha256",
                "sha256:" + pin,
                terminal["publication"]["primary_sha256"],
            )
            checks = attestation["report"].get("checks", [])
            require(
                side,
                "adversarial_check_passed",
                True,
                any("adversarial" in r["name"] and r["passed"] for r in checks),
            )
            bind(side)
            if "8305" in name:
                plan["historical"] = [
                    dict(
                        primary=reference(path),
                        scope="historical_only_zero_current_calls",
                        imported_fields=["historical_model_provenance"],
                        original_capture_experiments=[8153, 8155, 8182, 8184],
                    )
                ]
            progress("authenticated_input", len(plan["refs"]), 0)
    except (OSError, ValueError, KeyError) as error:
        if not plan["checks"] or plan["checks"][-1]["passed"]:
            plan["checks"].append(
                dict(
                    upstream="external_input",
                    path=str(root),
                    hash=None,
                    artifact_field="structure",
                    op="==",
                    expected="readable_authenticated_operand",
                    observed=str(error),
                    passed=False,
                )
            )
    return plan


def reduce_states(trajectory: Json, full: Json, indexed: Json, unsafe: Json) -> Json:
    """Reduce primitive states without treating operation repetitions as sources."""
    errors = [
        abs(a - b)
        for f, i in zip(full["releases"], indexed["releases"], strict=True)
        for a, b in zip(f["coefficients"], i["coefficients"], strict=True)
    ]
    errors += [
        abs(k.logit(f["coefficients"], x, "full") - k.logit(i["coefficients"], x))
        for f, i in zip(full["releases"], indexed["releases"], strict=True)
        for x in trajectory["cache_x"]
    ]
    return dict(
        source_id=trajectory["id"],
        condition=[trajectory["pattern"], trajectory["fault"]],
        numerator=int(k.semantic(full) == k.semantic(indexed)),
        denominator=1,
        eligibility=True,
        failure=False,
        censored=False,
        dense_sparse_error_max=max(errors),
        stale_cache_count=indexed["stale_cache_count"],
        unsafe_stale_cache_count=unsafe["stale_cache_count"],
        double_updates=len(indexed["applied"]) - len(set(indexed["applied"])),
        typed_actions_identical=[r["action"] for r in full["issues"]]
        == [r["action"] for r in indexed["issues"]],
    )


def measure(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Run correctness before paid paired timing, with real owned child deaths."""
    began = time.monotonic()
    progress("before_authentication")
    work = authenticate(root, raw)
    progress("after_authentication")
    protocol = k.manifest()
    atomic_json(raw / "protocol.json", protocol)
    work.update(
        rows=[],
        states=[],
        costs=[],
        crashes=[],
        audit=k.numeric_audit(),
        negative=k.negative_control(),
        fixture=fixture,
    )
    if all(c["passed"] for c in work["checks"]):
        with TemporaryDirectory(prefix="carnot-8306-measure-") as directory:
            private = Path(directory)
            trajectories = protocol["trajectories"][:1] if fixture else protocol["trajectories"]
            for n, trajectory in enumerate(trajectories):
                states = {
                    arm: k.execute(trajectory, arm, private / (trajectory["id"] + arm))
                    for arm in k.ARMS
                }
                work["states"].append(dict(trajectory_id=trajectory["id"], arms=states))
                work["rows"].append(
                    reduce_states(
                        trajectory, states["full"], states["indexed"], states["truncated"]
                    )
                )
                progress("correctness", n + 1, len(trajectories) - n - 1)
            progress("before_recovery_benchmark")
            checkpoint = private / "recovered"
            manifest_path = raw / "protocol.json"
            prefix = [str(ROOT / ".venv/bin/python")]
            if os.environ.get("COVERAGE_RCFILE"):
                prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
            base = [*prefix, "-u" if len(prefix) == 1 else str(ROOT / CLI)]
            if len(prefix) == 1:
                base.append(str(ROOT / CLI))
            base += [
                "--worker",
                str(manifest_path),
                "--trajectory",
                "-1",
                "--cohort-count",
                str(len(trajectories)),
                "--arm",
                "indexed",
                "--checkpoint",
                str(checkpoint),
            ]
            for event in [24, 48, -1]:
                spec = dict(
                    name="recovery_" + str(event),
                    argv=[*base, "--crash", str(event)],
                    deadline_s=120,
                    expected_exit=-9 if event != -1 else 0,
                    classification="required",
                )
                receipt = run_check(ROOT, spec, private, raw / "child_logs", heartbeat_s=20)
                receipt["event"] = event
                receipt["pending_counts"] = [
                    len(k.load(t, checkpoint / (t["id"] + ".json"))["pending"])
                    for t in trajectories
                ]
                work["crashes"].append(receipt)
            work["recovered_states"] = [
                k.load(t, checkpoint / (t["id"] + ".json")) for t in trajectories
            ]
            work["crash_parity"] = all(
                k.semantic(state) == k.semantic(stored["arms"]["indexed"])
                for state, stored in zip(work["recovered_states"], work["states"], strict=True)
            )
            progress("after_recovery_benchmark")
            progress("before_paired_benchmark")
            for repetition in range(-1, 5):
                order = ["full", "indexed"] if repetition % 2 == 0 else ["indexed", "full"]
                for arm in order:
                    arm_start = time.monotonic_ns()
                    for n, trajectory in enumerate(trajectories):
                        if time.monotonic() - began > 600:
                            raise TimeoutError("measurement_deadline")
                        path = private / f"pair-{repetition}-{arm}-{n}"
                        started = time.monotonic_ns()
                        state = k.execute(trajectory, arm, path)
                        ended = time.monotonic_ns()
                        if repetition >= 0:
                            work["costs"].append(
                                dict(
                                    repetition=repetition,
                                    arm=arm,
                                    source_id=trajectory["id"],
                                    started_monotonic_ns=started,
                                    ended_monotonic_ns=ended,
                                    complete_run_ns=ended - started,
                                    transactions=state["costs"],
                                    coefficient_visits=sum(
                                        r["coefficient_visits"] for r in state["releases"]
                                    ),
                                    invalidation_touches=sum(
                                        r["invalidation_touches"] for r in state["releases"]
                                    ),
                                    global_invalidations=sum(
                                        len(r["invalidated"])
                                        for r in state["releases"]
                                        if r["global_change"]
                                    ),
                                )
                            )
                        progress(f"pair_{repetition}_{arm}", n + 1, len(trajectories) - n - 1)
                    progress(f"paired_arm_complete_{repetition}_{arm}", len(trajectories), 0)
                    work.setdefault("pair_spans", []).append(
                        dict(
                            repetition=repetition,
                            arm=arm,
                            duration_ns=time.monotonic_ns() - arm_start,
                            warmup=repetition == -1,
                        )
                    )
            progress("after_paired_benchmark")
    work.update(
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_correctness_recovery_cost",
                duration_s=time.monotonic() - began,
                start_s=0,
            )
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "python/carnot/experiment_7425_v651_spline_prototype.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/methods_stream_execution_8111.py",
                "python/carnot/reporting/v709_execution.py",
                "python/carnot/verify/hard_exit_learning_qualification_8206.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "ops/exclusion_manifest.yaml",
            ]
        ],
        protocol_reference=reference(raw / "protocol.json"),
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness depends on correctness and owned receipts, independent of speed."""
    rows = work["rows"]
    failed = [c for c in work["checks"] if not c["passed"]]
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    correct = bool(rows) and all(
        r["numerator"] == 1
        and r["dense_sparse_error_max"] <= 1e-10
        and r["stale_cache_count"] == r["double_updates"] == 0
        and r["typed_actions_identical"]
        for r in rows
    )
    recovery = bool(work.get("crash_parity")) and all(r["passed"] for r in work["crashes"])
    controls = (
        work["negative"]["detected"]
        and work["negative"]["action_changed"]
        and work["audit"]["finite_difference_error_max"] < 1e-8
    )
    ready = int(owned and not failed and correct and recovery and controls)
    klass = (
        "disqualified"
        if not owned or (not failed and not ready)
        else "blocked"
        if failed
        else "circular_positive"
    )
    reason = (
        "owned_checks"
        if klass == "disqualified"
        else failed[0]["artifact_field"]
        if failed
        else "local_update_isolation"
    )
    intended = 1 if work["fixture"] else 48
    per_event = [
        dict(
            source_id=s["trajectory_id"],
            arm=arm,
            issues=state["issues"],
            releases=state["releases"],
        )
        for s in work["states"]
        for arm, state in s["arms"].items()
    ]
    value: Json = dict(
        experiment_id=8306,
        task_id=TASK,
        milestone="2026.10.717",
        run_date="20261008",
        honest_verdict="complete_" + klass + "_" + reason,
        verdict_class=klass,
        local_kernel_ready_score=ready,
        required_checks_passed=owned,
        flagged_adversarial=False,
        verifier_is_oracle=True,
        exposure_scope="constructed_mechanics_on_exposed_cached_development_contract",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            model_loads_attempted=0,
            model_loads_completed=0,
            generation_calls_attempted=0,
            generation_calls_completed=0,
            generate=0,
        ),
        historical_model_provenance=work["historical"],
        rows=rows,
        intended_count=intended,
        completed_count=len(rows),
        failed_count=sum(r["failure"] for r in rows),
        censored_count=intended - len(rows),
        excluded_count=0,
        independent_count=len(rows),
        sample_size_budget=dict(
            trajectories=48,
            seeds=3,
            patterns=4,
            fault_patterns=4,
            events_per_trajectory=64,
            repetitions=5,
            unit="constructed_trajectory_no_natural_sources",
        ),
        gate_check_summary=work["checks"],
        preconditions_checked=work["checks"],
        acceptance_gates=dict(
            dense_sparse_max=1e-10,
            stale_cache_count=0,
            double_updates=0,
            exact_crash_parity=True,
            negative_control=True,
            readiness_independent_of_speed=True,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7178306,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[reference(raw / "measurement.json"), work["protocol_reference"]],
        measurement_reference=reference(raw / "measurement.json"),
        protocol_reference=work["protocol_reference"],
        owned_coverage_reference=work.get("owned_coverage_reference"),
        invocation_argv=work.get("invocation_argv", []),
        dense_sparse_error_max=max((r["dense_sparse_error_max"] for r in rows), default=None),
        stale_cache_count=sum(r["stale_cache_count"] for r in rows) if rows else None,
        crash_parity=work.get("crash_parity"),
        negative_control_detected=work["negative"]["detected"],
        numeric_audit=work["audit"],
        negative_control=work["negative"],
        per_event_rows=per_event,
        coefficient_touch_rows=[
            dict(
                source_id=r["source_id"],
                arm=r["arm"],
                repetition=r["repetition"],
                visits=r["coefficient_visits"],
            )
            for r in work["costs"]
        ],
        invalidation_rows=[
            dict(
                source_id=r["source_id"],
                arm=r["arm"],
                repetition=r["repetition"],
                touches=r["invalidation_touches"],
                global_invalidations=r["global_invalidations"],
            )
            for r in work["costs"]
        ],
        complete_transaction_cost_rows=work["costs"],
        actual_crash_receipts=work["crashes"],
        cited_upstream_artifacts=[
            dict(
                name=n,
                sha256="sha256:" + pin,
                fields_imported=[
                    "required_checks_passed",
                    "terminal_validation_sidecar_path",
                    "methodology_only",
                ],
            )
            for n, pin in PINS.items()
        ],
        methodology_note="Fixed 34 parameter sigmoid head; exact local writes and global stress invalidations on constructed oracle trajectories. No natural benefit or speed gate.",
        fixture=work["fixture"],
    )
    value["field_principles"] = {
        field: "Bind actual invocation, byte custody and complete intended constructed evidence; no natural generalization."
        for field in value
    }
    for fields, purpose in [
        (
            [
                "local_kernel_ready_score",
                "dense_sparse_error_max",
                "stale_cache_count",
                "crash_parity",
                "negative_control_detected",
            ],
            "Correctness, cache integrity and durable recovery precede efficiency; owned failures zero readiness.",
        ),
        (
            [
                "per_event_rows",
                "coefficient_touch_rows",
                "invalidation_rows",
                "complete_transaction_cost_rows",
                "actual_crash_receipts",
            ],
            "Preserve global-intercept costs and full issue/release/checkpoint work. Repetitions are not independent samples; both deaths preserve all intended trajectory states.",
        ),
        (
            [
                "verifier_is_oracle",
                "exposure_scope",
                "independent_generalization_score",
                "generalized_learning_benefit_score",
            ],
            "Oracle fixtures are circular positives; exposed cached data establishes no independent benefit.",
        ),
        (
            [
                "inference_substrate",
                "inference_substrate_class",
                "MODEL_SPECS",
                "model_invocation_counts",
                "historical_model_provenance",
            ],
            "Zero current model calls; imported Qwen captures remain historical.",
        ),
        (
            [
                "gate_check_summary",
                "preconditions_checked",
                "validation_receipts",
                "terminal_validation_sidecar_path",
            ],
            "Exact operands and byte-bound child exits establish execution authority; absence differs from measured zero.",
        ),
    ]:
        value["field_principles"].update(dict.fromkeys(fields, purpose))
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold replay recomputes causal rows, rejecting aggregate and rehashed drift."""
    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        value["reproducibility_checksum"] = checksum
        refs = (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        for ref in refs:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        protocol = json.loads(Path(work["protocol_reference"]["path"]).read_text())
        if (
            protocol != k.manifest()
            or work["audit"] != k.numeric_audit()
            or work["negative"] != k.negative_control()
        ):
            return False
        if work["states"]:
            for name, pin in PINS.items():
                if not any(
                    Path(r["path"]).name == pin + "-" + name + ".json"
                    and r["sha256"] == "sha256:" + pin
                    for r in work["refs"]
                ):
                    return False
        for receipt in [*value["validation_receipts"], *work["crashes"]]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        if work["states"] and not all(
            k.semantic(r) == k.semantic(s["arms"]["indexed"])
            for r, s in zip(work["recovered_states"], work["states"], strict=True)
        ):
            return False
        for receipt in [*value["validation_receipts"], *work["crashes"]]:
            for prefix in ["stdout", "stderr"]:
                if (
                    receipt.get(prefix + "_path")
                    and sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]
                ):
                    return False
        expected_rows = []
        for n, stored in enumerate(work["states"]):
            trajectory = protocol["trajectories"][n]
            states = {}
            for arm in k.ARMS:
                state = k.initial(trajectory, arm)
                for slot in range(72):
                    k.issue(state, slot)
                    k.release(trajectory, state, slot, arm)
                if k.semantic(state) != k.semantic(stored["arms"][arm]):
                    return False
                states[arm] = state
            expected_rows.append(
                reduce_states(trajectory, states["full"], states["indexed"], states["truncated"])
            )
        if work["rows"] != expected_rows:
            return False
        for row in work["costs"]:
            if row["complete_run_ns"] != row["ended_monotonic_ns"] - row["started_monotonic_ns"]:
                return False
            for event in row["transactions"]:
                if (
                    event["total_ns"] != event["ended_monotonic_ns"] - event["started_monotonic_ns"]
                    or event["total_ns"] <= 0
                ):
                    return False
        raw = Path(value["measurement_reference"]["path"]).parent
        return bool(
            build(work, raw, value["validation_receipts"], fixture=value["fixture"]) == value
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Use one bounded execution adapter for worker, private and terminal routes."""
    from carnot.reporting.local_update_execution_8306 import main as execute

    return execute(argv)
