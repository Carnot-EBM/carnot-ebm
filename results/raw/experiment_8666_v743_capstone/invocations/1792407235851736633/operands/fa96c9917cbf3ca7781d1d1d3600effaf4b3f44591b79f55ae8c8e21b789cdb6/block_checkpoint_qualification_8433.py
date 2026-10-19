"""REQ-REPORT-8433: synthetic process recovery qualifies a bounded state schema.

Every process death and resume retains primitive operands. This is plumbing
evidence only; no public human labels or current generator outputs are used.
"""

from __future__ import annotations

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

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import expit

from carnot.reporting import v726_contract_methods as methods
from carnot.reporting import v717_contract_runner as shipped
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.human_label_custody_8389 import gate
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.verify import block_feedback_state_8433 as state
from carnot.verify import sparse_energy_7996 as optimizer

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = (
    "experiment_8433_v726_block_checkpoint_qualification",
    "exp8433-block-checkpoint-qualification",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_block_feedback_state_8433.py"
OWNED = [
    "python/carnot/verify/block_feedback_state_8433.py",
    "python/carnot/reporting/block_checkpoint_qualification_8433.py",
    "python/carnot/reporting/block_checkpoint_runner_8433.py",
    CLI,
]
METHODS_PATH = "results/experiment_8429_v726_contract_methods.json"
METHODS_PIN = "sha256:dfad99775207ce8073b35c7d5efbc0cbefa70ad1ddadadafa584157add3a3095"
MODEL_SPECS: list[Json] = []
REQUIRED_CHECKS = {
    "owned_tests",
    "private_E2E018_consumers",
    "private_E2E020",
    "private_E2E021",
    "coverage_combine",
    "coverage_report",
    "coverage_json",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
}
reference, read, checksum = methods.reference, methods.read, methods.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Each phase and completed child remains visible during bounded CPU work."""
    print(f"[exp8433] phase={phase} completed={completed} pending={pending}", flush=True)


def independent_update() -> Json:
    """Evaluate each SciPy basis separately instead of using the optimizer's basis."""
    h = state.head()
    x = np.asarray(state.questions(0)[0]["answers"][0]["x"])
    knots = [0.0] * 4 + [i / 9 for i in range(1, 9)] + [1.0] * 4
    basis = np.asarray([BSpline(knots, np.eye(12)[k], 3)(v) for v in x for k in range(12)] + [1.0])
    theta = np.asarray(h["parameters"]) * h["decay_scale"]
    z = np.log(x[0] / (1 - x[0])) + basis @ theta
    expected = theta - 0.01 * (
        basis * (expit(z / h["temperature"]) - 1) / h["temperature"] + 0.002 * theta
    )
    changed, _ = optimizer.update(h, x, 1, 0.01)
    error = float(np.max(np.abs(optimizer.parameters(changed) - expected)))
    delta = float(np.max(np.abs(expected - theta)))
    return dict(
        passed=error < 1e-12 and delta > 1e-6, max_error=error, nonzero_delta=delta, tolerance=1e-12
    )


def fixtures() -> list[Json]:
    """Repeated crashes remain dependent controls of the same synthetic questions."""
    return [state.config(c, p) for c in (8, 16) for p in ("fifo", "uniform")]


def filesystem_type(path: Path) -> str:
    """Mounted filesystem evidence distinguishes private disk from memory storage."""
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ")
        mount = Path(left.split()[4].replace("\\040", " "))
        if path.resolve().is_relative_to(mount):
            mounts.append((len(str(mount)), right.split()[0]))
    return max(mounts)[1] if mounts else "unknown"


def measure(raw: Path, private: Path, root: Path = ROOT) -> Json:
    """Authenticate only actual CPU operands before launching real recovery children."""
    started = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions")
    authority = methods.authority(root, raw / "authority")
    failures = list(authority["gate_check_summary"])
    refs = []
    try:
        upstream = read(dict(path=str(root / METHODS_PATH), sha256=METHODS_PIN))
        if (
            upstream["methods_ready_score"] != 1
            or upstream["methods_protocol_sha256"] != methods.PROTOCOL_PIN
        ):
            raise ValueError("methods_not_ready")
        methods.protocol()
        refs = [reference(root / METHODS_PATH), reference(root / methods.PROTOCOL)]
    except (OSError, ValueError, KeyError) as error:
        failures.append(
            gate(
                str(root / METHODS_PATH),
                "methods_ready_score",
                1,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream=methods.TASK,
            )
        )
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    filesystem = filesystem_type(private)
    preconditions: Json = dict(
        private_disk_backed_scratch=filesystem not in {"tmpfs", "ramfs", "devtmpfs", "unknown"},
        private_filesystem_type=filesystem,
        private_path=str(private),
        free_disk_bytes=shutil.disk_usage(private).free,
        available_memory_bytes=memory,
        minimum_memory_bytes=128 * 1024**2,
        task_cap_s=4800,
        validation_reserve_s=900,
        child_poll_s=30,
        tools={
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
    )
    if (
        not preconditions["private_disk_backed_scratch"]
        or memory < preconditions["minimum_memory_bytes"]
        or preconditions["free_disk_bytes"] < 104857600
        or not all(preconditions["tools"].values())
    ):
        failures.append(gate(str(private), "private_resources", True, preconditions, upstream=TASK))
    rows: list[Json] = []
    crashes: list[Json] = []
    receipts: list[Json] = []
    primitives: list[Json] = []
    for cfg in fixtures():
        family = f"c{cfg['capacity']}-{cfg['policy']}"
        config_path = raw / family / "config.json"
        atomic_json(config_path, cfg)
        primitives.append(reference(config_path))
        baseline_hash = None
        cases = [
            "none",
            *[f"{t}:1:{b}" for t in ("seal", "release") for b in state.BARRIERS],
            "after_block2",
        ]
        for case in cases:
            progress("recovery", len(rows), 40 - len(rows))
            if (time.monotonic_ns() - started) / 1e9 > 3900 and not failures:
                failures.append(
                    gate(
                        str(raw),
                        "measurement_deadline_s",
                        3900,
                        (time.monotonic_ns() - started) / 1e9,
                        operator="<=",
                        upstream=TASK,
                    )
                )
            directory = raw / family / case.replace(":", "-")
            row = dict(
                unit_id=family + ":" + case,
                family="synthetic-100-questions",
                arm=cfg["policy"],
                capacity=cfg["capacity"],
                seed=17,
                barrier=case,
                status="censored" if failures else "completed",
                passed=None,
                absolute_metric=0 if failures else None,
                raw_numerator=0 if failures else None,
                raw_denominator=1,
                censored=bool(failures),
                excluded=False,
                missing_reason="required_cpu_operand_unavailable" if failures else None,
            )
            if not failures:
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / CLI),
                    "--worker",
                    str(directory),
                    "--configuration",
                    str(config_path),
                ]
                if case != "none":
                    crash = child(
                        family + "-crash-" + case.replace(":", "-"),
                        command + ["--crash", case],
                        raw / "processes",
                        expected=73,
                        deadline=60,
                    )
                    receipts.append(crash)
                    try:
                        snapshot = state.Store(directory, cfg).read()
                    except (OSError, ValueError, KeyError):
                        snapshot = {}
                    expected_version = (
                        6
                        if case == "after_block2"
                        else (2 if case.startswith("seal") else 3)
                        + int(case.split(":")[-1] in ("atomic_replace", "acknowledgment"))
                    )
                    snapshot_path = directory / "crash_state.json"
                    atomic_json(snapshot_path, dict(state=snapshot))
                    primitives.append(reference(snapshot_path))
                    crashes.append(
                        dict(
                            unit_id=row["unit_id"],
                            actual_exit=crash["exit_code"],
                            boundary=case,
                            committed_phase=snapshot.get("phase"),
                            committed_version=snapshot.get("version"),
                            expected_version=expected_version,
                            state=reference(snapshot_path),
                            passed=crash["passed"]
                            and snapshot.get("version") == expected_version
                            and state.replay_state(snapshot, cfg),
                        )
                    )
                resumed = child(
                    family + "-resume-" + case.replace(":", "-"),
                    command,
                    raw / "processes",
                    deadline=60,
                )
                receipts.append(resumed)
                final_path = directory / "final.json"
                result = json.loads(final_path.read_bytes()) if final_path.exists() else {}
                digest = canonical_hash(result)
                if case == "none":
                    baseline_hash = digest
                passed = (
                    resumed["passed"]
                    and digest == baseline_hash
                    and state.replay_state(result, cfg)
                )
                if case != "none":
                    passed = passed and crashes[-1]["passed"]
                row.update(
                    passed=passed,
                    absolute_metric=int(passed),
                    raw_numerator=int(passed),
                    state_hash=digest,
                    final_state=reference(final_path) if final_path.exists() else None,
                    configuration=reference(config_path),
                )
                if final_path.exists():
                    primitives.append(reference(final_path))
                if not passed:
                    row.update(status="failed", missing_reason="owned_recovery_failure")
            rows.append(row)
    work = dict(
        root=str(root),
        rows=rows,
        crash_rows=crashes,
        recovery_receipts=receipts,
        failures=failures,
        authority=authority,
        preconditions=preconditions,
        source_refs=refs,
        code_refs=[reference(ROOT / p) for p in OWNED],
        primitive_files=primitives,
        nonzero_update_controls=independent_update(),
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    progress("recovery_complete", len(rows), 0)
    work["rng_resume_checks"] = [
        dict(
            unit_id=r["unit_id"],
            passed=r["passed"],
            complete_state_and_future_decisions_equal=r["passed"],
        )
        for r in rows
        if r["barrier"] != "none"
    ]
    work["label_access_rows"] = [
        dict(
            unit_id=r["unit_id"],
            future_label_reads=0 if r["passed"] else None,
            all_siblings_sealed_before_release=r["passed"],
        )
        for r in rows
    ]
    work["frozen_controls"] = []
    for shuffled in (False, True):
        cfg = state.config(8, "uniform", frozen=True, shuffled=shuffled)
        result = state.drive(private / f"frozen-{shuffled}", cfg)
        target = raw / f"frozen-{shuffled}.json"
        atomic_json(target, dict(configuration=cfg, state=result))
        primitives.append(reference(target))
        work["frozen_controls"].append(
            dict(
                passed=result["head"] == state.head() and result["parameter_updates"] == 0,
                primitive=reference(target),
            )
        )
    if not failures and all(r["passed"] for r in rows):
        cfg = state.config(8, "uniform", shuffled=True)
        config_path = raw / "shuffled_configuration.json"
        atomic_json(config_path, cfg)
        primitives.append(reference(config_path))
        states = []
        for label, subdirectory, crash, expected in [
            ("shuffled_baseline", "baseline", "", 0),
            ("shuffled_crash", "resume", "release:1:temporary_write", 73),
            ("shuffled_resume", "resume", "", 0),
        ]:
            directory = raw / "shuffled" / subdirectory
            command = [
                sys.executable,
                "-u",
                str(ROOT / CLI),
                "--worker",
                str(directory),
                "--configuration",
                str(config_path),
                "--crash",
                crash,
            ]
            receipts.append(
                child(label, command, raw / "processes", expected=expected, deadline=60)
            )
            if expected == 0:
                final_path = directory / "final.json"
                states.append(read(reference(final_path)))
                primitives.append(reference(final_path))
        work["rng_resume_checks"].append(
            dict(
                unit_id="shuffled_feedback_rng",
                passed=all(r["passed"] for r in receipts[-3:]) and states[0] == states[1],
                feedback_rng_changed=states[0]["feedback_rng"]
                != state.initial(cfg)["feedback_rng"],
                complete_state_and_future_decisions_equal=states[0] == states[1],
            )
        )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """A plumbing qualification never becomes an independent learning claim."""
    rows = work["rows"]
    qualified = (
        bool(receipts)
        and REQUIRED_CHECKS <= {r["name"] for r in receipts}
        and all(r["passed"] for r in receipts)
        and not work.get("adversarial_findings")
        and not any(f.get("check") == "measurement_deadline_s" for f in work["failures"])
    )
    recovered = (
        all(r["passed"] is True for r in rows)
        and work["nonzero_update_controls"]["passed"]
        and all(r["passed"] for r in work["frozen_controls"])
        and all(r["passed"] for r in work["rng_resume_checks"])
    )
    ready = qualified and recovered and not work["failures"]
    verdict = (
        "disqualified"
        if not qualified or (not recovered and not work["failures"])
        else "circular_positive"
        if ready
        else "blocked"
    )
    finals = {r["unit_id"]: read(r["final_state"]) for r in rows if r.get("final_state")}
    value = dict(
        experiment_id=8433,
        task_id=TASK,
        milestone="2026.10.726",
        run_date="20261012",
        honest_verdict=f"complete_{verdict}_v726_block_checkpoint_qualification",
        verdict_class=verdict,
        gate_check_summary=[
            *work["failures"],
            *(
                [
                    gate(
                        str(raw / "execution_manifest.json"),
                        "required_validation_manifest",
                        sorted(REQUIRED_CHECKS),
                        sorted({r["name"] for r in receipts}),
                        upstream=TASK,
                    )
                ]
                if not REQUIRED_CHECKS <= {r["name"] for r in receipts}
                else []
            ),
            *[
                gate(str(r["stdout_path"]), "required_owned_check", True, False, upstream=TASK)
                for r in receipts
                if not r["passed"]
            ],
        ],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=[],
        reader_inference_substrate="aggregation_from_upstream_artifacts",
        rows=rows,
        intended_count=40,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["passed"] is False for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            synthetic_questions=100,
            answers=300,
            recovery_cases=40,
            independent_generalization_questions=0,
            repeats_are_dependent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="synthetic plumbing fixtures; no human-label stream or current Qwen inference",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=not qualified,
        acceptance_gates=dict(
            owned_validation=qualified,
            actual_recovery=recovered,
            required_cpu_inputs=not work["failures"],
            synthetic_only=True,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(name=r["name"], duration_s=r["duration_s"])
            for r in [*work["recovery_receipts"], *receipts]
        ],
        random_seed=17,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["primitive_files"] + work.get("coverage_shards", []),
        cited_upstream_artifacts=[
            dict(
                r,
                producer=methods.TASK,
                imported_fields=["methods_ready_score", "methods_protocol_sha256"]
                if r["path"].endswith(METHODS_PATH)
                else ["feedback"],
            )
            for r in work["source_refs"]
        ],
        block_store_ready_score=int(ready),
        checkpoint_schema=state.SCHEMA,
        checkpoint_schema_hash=canonical_hash(state.SCHEMA),
        crash_rows=work["crash_rows"],
        resume_state_hashes=[
            dict(unit_id=r["unit_id"], state_hash=r.get("state_hash")) for r in rows
        ],
        pending_high_water={
            str(c): max(
                (
                    finals[r["unit_id"]]["pending_high_water"]
                    for r in rows
                    if r["capacity"] == c and r["unit_id"] in finals
                ),
                default=None,
            )
            for c in (8, 16)
        },
        applied_feedback_ids={
            identity: final["applied_feedback_ids"] for identity, final in finals.items()
        },
        label_access_rows=work["label_access_rows"],
        rng_resume_checks=work["rng_resume_checks"],
        nonzero_update_controls=work["nonzero_update_controls"],
        frozen_controls=work["frozen_controls"],
        primitive_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        methodology_note="Independent basis arithmetic and real crash recovery qualify synthetic two-transaction blocks only.",
    )
    value["field_principles"] = {
        k: "Bound to this invocation's synthetic process evidence; no independent factuality or learning benefit."
        for k in value
    }
    value["field_principles"].update(
        block_store_ready_score="Requires actual recovery, nonzero/frozen controls, fixed CPU operands and all owned validation.",
        pending_high_water="At most C selected question clusters and 3*C answer feature records; audit metadata is separate.",
        acknowledgment_ids="Stable transaction identities prevent duplicate application after lost delivery.",
        checkpoint_schema="Separately qualified nine-input/109-coefficient schema; older store assertions remain unchanged.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Recompute primitive semantics and validate real child records before reduction."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        cases = [
            "none",
            *[f"{t}:1:{b}" for t in ("seal", "release") for b in state.BARRIERS],
            "after_block2",
        ]
        roster = [
            f"c{cfg['capacity']}-{cfg['policy']}:{case}" for cfg in fixtures() for case in cases
        ]
        if [r["unit_id"] for r in work["rows"]] != roster:
            return False
        with TemporaryDirectory(prefix="8433-cold-authority-", dir="/var/tmp") as directory:
            actual = methods.authority(Path(work["root"]), Path(directory))
        if actual["activated"] != work["authority"]["activated"]:
            return False
        if work["authority"]["activated"] and not work["failures"]:
            upstream = read(dict(path=str(Path(work["root"]) / METHODS_PATH), sha256=METHODS_PIN))
            if upstream["methods_ready_score"] != 1:
                return False
        for ref in [
            *work["source_refs"],
            *work["code_refs"],
            *work["primitive_files"],
            *work.get("coverage_shards", []),
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if work["nonzero_update_controls"] != independent_update():
            return False
        for row in work["rows"]:
            if row.get("final_state"):
                cfg, result = read(row["configuration"]), read(row["final_state"])
                if (
                    cfg not in fixtures()
                    or not state.replay_state(result, cfg)
                    or row["state_hash"] != canonical_hash(result)
                    or row["passed"] is not True
                ):
                    return False
        if not work["failures"] and all(r["passed"] for r in work["rows"]):
            shuffled = [
                read(ref)
                for ref in work["primitive_files"]
                if "/shuffled/" in ref["path"] and ref["path"].endswith("final.json")
            ]
            if (
                len(shuffled) != 2
                or shuffled[0] != shuffled[1]
                or not state.replay_state(shuffled[0], state.config(8, "uniform", shuffled=True))
            ):
                return False
        for crash in work["crash_rows"]:
            row = next(r for r in work["rows"] if r["unit_id"] == crash["unit_id"])
            result = read(crash["state"])["state"]
            valid = (
                state.replay_state(result, read(row["configuration"]))
                and crash["actual_exit"] == 73
                and crash["expected_version"] == result.get("version")
            )
            if (
                valid != crash["passed"]
                or crash["committed_version"] != result.get("version")
                or crash["committed_phase"] != result.get("phase")
            ):
                return False
        for control in work["frozen_controls"]:
            primitive = read(control["primitive"])
            if (
                not state.replay_state(primitive["state"], primitive["configuration"])
                or primitive["state"]["head"] != state.head()
            ):
                return False
        for receipt in [*work["recovery_receipts"], *value["validation_receipts"]]:
            saved = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            if any(
                receipt[k] != saved[k]
                for k in (
                    "name",
                    "argv",
                    "exit_code",
                    "expected_exit",
                    "timed_out",
                    "duration_s",
                    "stdout_sha256",
                    "stderr_sha256",
                )
            ):
                return False
            if (
                sha256_file(Path(receipt["stdout_path"])) != receipt["stdout_sha256"]
                or sha256_file(Path(receipt["stderr_path"])) != receipt["stderr_sha256"]
            ):
                return False
            expected_pass = (
                receipt["exit_code"] == receipt["expected_exit"] and not receipt["timed_out"]
            )
            if receipt["name"] == "adversarial":
                report = json.loads(Path(receipt["stdout_path"]).read_bytes())
                flags = [flag for row in report["reports"] for flag in row.get("flags", [])]
                expected_pass = expected_pass and not flags
                if work.get("adversarial_findings", []) != flags:
                    return False
            if receipt["passed"] != expected_pass:
                return False
        expected = build(
            work,
            value["validation_receipts"],
            Path(value["primitive_reference"]["path"]).parent,
            Path(value["publication_output"]),
        )
        return bool(value == expected)
    except (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError):
        return False


def manifest(private: Path) -> list[Json]:
    """Freeze scoped checks and genuine exit/subprocess coverage before qualification."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(shipped, "m", sys.modules[__name__]):
        plan = shipped.manifest(private)
    config_path = private / "coverage.ini"
    config_path.write_text(
        config_path.read_text().replace("patch=subprocess", "patch=subprocess,_exit")
    )
    next(r for r in plan if r["name"] == "owned_tests")["deadline"] = 900
    next(r for r in plan if r["name"] == "coverage_combine")["argv"].append("--keep")
    for index, (label, test) in enumerate(
        [
            ("020", "test_hard_exit_learning_qualification_8206.py"),
            ("021", "test_restricted_decision_audit_8210.py"),
        ]
    ):
        plan.insert(
            2 + index,
            dict(
                name="private_E2E" + label,
                argv=[
                    str(ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "tests/python/" + test,
                    "--basetemp=" + str(private / ("e2e" + label)),
                ],
                expected=0,
                deadline=300,
                scope="owned",
            ),
        )
    return list(plan)
