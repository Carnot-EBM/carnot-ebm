"""REQ-REPORT-8490: keep an unqualified objective adapter outside real-label learning.

This reader records the stopped study. A producer's reported readiness cannot
replace its own primitive replay or authorize use of reserved targets.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import objective_checkpoint_8489 as upstream
from carnot.reporting import v727_historical_inputs as history
from carnot.reporting import v730_contract_methods as methods
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8490_v730_brier_online_learning", "exp8490-brier-online-learning"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_brier_online_learning_8490.py"
OWNED = [
    "python/carnot/reporting/brier_online_learning_8490.py",
    "python/carnot/reporting/brier_online_runner_8490.py",
    CLI,
]
ARMS, SEEDS = ("frozen", "log_loss", "brier", "brier_shuffled_released"), (17, 29, 43, 71, 101)
MODEL_SPECS: list[Json] = []
UPSTREAM_TASK, UPSTREAM_PIN = (
    upstream.TASK,
    "sha256:f306c910fd8fe95b12ca2505655347c73846c63d7a46700f433f7b67d9384bad",
)
PLAN_PATH = ROOT / "results/experiment_8434_v726_capacity_online_learning.json"
PLAN_PIN = "sha256:" + history.PINS[8434]
read, reference = methods.read, methods.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so a supervisor can distinguish waiting from abandoned work."""
    print(f"[exp8490] phase={phase} completed={completed} pending={pending}", flush=True)


def checksum(value: Json) -> str:
    """The content seal excludes itself so fresh readers can recompute it."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def gate(
    path: Path, digest: Any, field: str, expected: Any, observed: Any, producer: str = UPSTREAM_TASK
) -> Json:
    """Record the exact failed operand instead of treating missing evidence as zero."""
    return dict(
        check=field,
        upstream=producer,
        path=str(path),
        sha256=digest,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def qualification_failures(path: Path, digest: str, value: Json) -> list[Json]:
    """Mechanics readiness also requires clean flags and successful required checks."""
    return [
        gate(path, digest, field, expected, value.get(field))
        for field, expected in [
            ("objective_kernel_ready_score", 1),
            ("flagged_adversarial", False),
            ("required_checks_passed", True),
        ]
        if value.get(field) != expected
    ]


def roster() -> tuple[list[Json], list[Json]]:
    """Only identifiers enter the stopped study; no cached target becomes a new measurement."""
    plan = read(dict(path=str(PLAN_PATH), sha256=PLAN_PIN))
    answers = [
        r for r in plan["rows"] if r["seed"] == 17 and r["arm"] == "frozen" and r["capacity"] == 8
    ]
    tune = [
        r
        for r in plan["retention_rows"]
        if r["seed"] == 17 and r["arm"] == "frozen" and r["capacity"] == 8
    ]
    missing = dict(
        status="censored",
        censored=True,
        excluded=False,
        missing_reason="qualified_exp8489_update_adapter_unavailable",
        probability=None,
        brier=None,
        log_loss=None,
        decision_loss=None,
        absolute_metric=None,
        raw_numerator=None,
        raw_denominator=1,
    )
    rows = [
        dict(
            missing,
            unit_id=f"s{seed}:{arm}:{r['answer_id']}",
            answer_id=r["answer_id"],
            question_hash=r["question_hash"],
            family=r["question_hash"],
            block=r["block"],
            seed=seed,
            arm=arm,
            capacity=8,
        )
        for seed in SEEDS
        for arm in ARMS
        for r in answers
    ]
    retention = [
        dict(
            missing,
            unit_id=f"s{seed}:{arm}:w{r['boundary']}:{r['unit_id']}",
            answer_id=r["unit_id"],
            question_hash=r["question_hash"],
            boundary=r["boundary"],
            seed=seed,
            arm=arm,
            read_only=True,
        )
        for seed in SEEDS
        for arm in ARMS
        for r in tune
    ]
    return rows, retention


def inspect(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate current task authority separately from sealed historical producer authority."""
    failures, refs, receipts = [], [dict(path=str(PLAN_PATH), sha256=PLAN_PIN)], []
    try:
        authority = methods.authority(root, raw / "authority")
        failures.extend(authority["gate_check_summary"])
        task = next(t for t in authority["tasks"] if t["id"] == TASK)
        if not authority["activated"] or task["deliverable"] != f"results/{NAME}.json":
            raise ValueError("activated_task_identity")
        methods.protocol()
    except (OSError, ValueError, KeyError, StopIteration) as error:
        authority = dict(activated=False)
        failures.append(
            gate(
                root / methods.DESIGN,
                None,
                "activated_task_authority",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
                TASK,
            )
        )
    resources = dict(
        methods.base.legacy.resources(private),
        free_disk_bytes=shutil.disk_usage(private).free,
        tools={
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
        task_cap_s=4800,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        child_poll_s=30,
        model_loads=0,
        generations=0,
        optional_hardware_required=False,
    )
    if (
        not resources["private_disk_backed_scratch"]
        or resources["available_memory_bytes"] < resources["minimum_memory_bytes"]
        or resources["free_disk_bytes"] < 104857600
        or not all(resources["tools"].values())
    ):
        failures.append(gate(private, None, "private_disk_memory_tools", True, resources, TASK))
    path = root / "results" / (upstream.NAME + ".json")
    try:
        value = read(dict(path=str(path), sha256=UPSTREAM_PIN))
        refs.append(reference(path))
        failures.extend(qualification_failures(path, UPSTREAM_PIN, value))
        terminal = read(reference(Path(value["terminal_validation_sidecar_path"])))
        side = Path(terminal["publication"]["sidecar_path"])
        refs.append(reference(side))
        try:
            bound = read_bound_sidecar(path, side)
            if bound["report"]["passed"] is not True:
                failures.append(
                    gate(
                        side,
                        sha256_file(side),
                        "bound_publication_validation",
                        True,
                        bound["report"]["passed"],
                    )
                )
        except (OSError, ValueError, KeyError) as error:
            failures.append(
                gate(side, sha256_file(side), "bound_publication_validation", True, str(error))
            )
        receipt = child(
            "upstream_cold_replay",
            [sys.executable, "-u", str(ROOT / upstream.CLI), "--cold-replay", str(path)],
            raw / "external",
            deadline=180,
            scope="external",
        )
        receipts.append(receipt)
        if not receipt["passed"]:
            failures.append(gate(path, UPSTREAM_PIN, "cold_replay", True, False))
        refs.append(value["update_adapter_reference"])
        if (
            sha256_file(Path(value["update_adapter_reference"]["path"]))
            != value["update_adapter_reference"]["sha256"]
        ):
            raise ValueError("update_adapter_hash")
    except (OSError, ValueError, KeyError) as error:
        failures.append(
            gate(
                path,
                UPSTREAM_PIN,
                "qualified_update_adapter",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
            )
        )
    try:
        sealed = history.custody(root, raw / "original_authority")
        refs.extend(dict(path=r["snapshot_path"], sha256=r["sha256"]) for r in sealed)
        request, response = raw / "historical_request.json", raw / "historical_response.json"
        atomic_json(
            request,
            dict(
                number=8434,
                primary=dict(
                    path=str(root / "results" / PLAN_PATH.name),
                    snapshot_path=str(root / "results" / PLAN_PATH.name),
                    sha256=PLAN_PIN,
                ),
                authority=sealed,
            ),
        )
        receipt = child(
            "historical_v726_replay",
            [
                sys.executable,
                "-u",
                str(ROOT / methods.CLI),
                "--history-request",
                str(request),
                "--history-output",
                str(response),
            ],
            raw / "external",
            deadline=180,
            scope="external",
        )
        receipts.append(receipt)
        qualified = json.loads(response.read_bytes())["qualified"] if receipt["passed"] else False
        if not qualified:
            failures.append(
                gate(
                    root / "results" / PLAN_PATH.name,
                    PLAN_PIN,
                    "original_authority_replay",
                    True,
                    qualified,
                    "exp8434-capacity-online-learning",
                )
            )
    except (OSError, ValueError, KeyError) as error:
        failures.append(
            gate(
                root / history.DESIGN,
                history.DESIGN_PIN,
                "historical_authority",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
                "exp8434-capacity-online-learning",
            )
        )
    return dict(
        authority=authority,
        preconditions=resources,
        failures=failures,
        source_refs=refs,
        external_receipts=receipts,
    )


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Record failed admission before any learner can see real or retention targets."""
    progress("preconditions")
    started = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    checked = inspect(root, raw, private)
    if not checked["failures"]:
        raise ValueError("sealed_blocking_condition_changed_rerun_full_causal_study")
    rows, retention = roster()
    atomic_json(raw / "intended_roster.json", dict(rows=rows, retention_rows=retention))
    work = dict(
        checked,
        root=str(root),
        raw=str(raw),
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        label_updates=0,
        roster_reference=reference(raw / "intended_roster.json"),
        code_refs=[reference(ROOT / p) for p in OWNED],
        adversarial_findings=[],
    )
    atomic_json(raw / "measurement.json", work)
    progress("blocked_before_labels", 0, 6000)
    return work


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """External qualification blocks science; unsuccessful owned validation disqualifies it."""
    roster_value = read(work["roster_reference"])
    passed = (
        bool(receipts) and all(r["passed"] for r in receipts) and not work["adversarial_findings"]
    )
    verdict = "blocked" if passed else "disqualified"
    failures = [
        *work["failures"],
        *[
            gate(
                Path(r["stdout_path"]),
                r["stdout_sha256"],
                "validation_checks." + r["name"],
                True,
                r["passed"],
                TASK,
            )
            for r in receipts
            if not r["passed"]
        ],
    ]
    if not receipts:
        failures.append(
            gate(Path(work["raw"]), None, "owned_validation_receipts", True, None, TASK)
        )
    value = dict(
        experiment_id=8490,
        task_id=TASK,
        milestone="2026.10.730",
        run_date="20261013",
        honest_verdict=f"complete_{verdict}_v730_brier_online_learning",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=read(dict(path=str(PLAN_PATH), sha256=PLAN_PIN))[
            "historical_model_provenance"
        ],
        **roster_value,
        intended_count=6000,
        completed_count=0,
        failed_count=0,
        censored_count=6000,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            test_questions=100,
            tune_questions=50,
            siblings=3,
            seeds=list(SEEDS),
            arms=list(ARMS),
            capacity=8,
            blocks=5,
            block_questions=20,
            intended_answer_rows=6000,
            intended_retention_rows=15000,
            repeats_are_dependent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="V726 exposed historical roster; no objective learning executed",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        acceptance_gates=dict(
            current_authority=work["authority"]["activated"],
            required_upstream=False,
            owned_validation=passed,
            objective_stream=False,
            favorable_contrast=None,
        ),
        validation_receipts=receipts,
        external_validation_receipts=work["external_receipts"],
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work["adversarial_findings"],
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(name=r["name"], duration_s=r["duration_s"])
            for r in [*work["external_receipts"], *receipts]
        ],
        random_seed=list(SEEDS),
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[work["roster_reference"], *work.get("coverage_refs", [])],
        cited_upstream_artifacts=[
            dict(
                r,
                imported_fields=(
                    [
                        "rows[].answer_id",
                        "rows[].question_hash",
                        "rows[].block",
                        "retention_rows[].unit_id",
                        "retention_rows[].question_hash",
                        "retention_rows[].boundary",
                        "historical_model_provenance",
                    ]
                    if Path(r["path"]).name == PLAN_PATH.name
                    else [
                        "objective_kernel_ready_score",
                        "required_checks_passed",
                        "flagged_adversarial",
                        "terminal_validation_sidecar_path",
                        "update_adapter_reference",
                    ]
                    if Path(r["path"]).name == upstream.NAME + ".json"
                    else ["primary_sha256", "report.passed"]
                    if Path(r["path"]).suffix == ".json"
                    else ["sealed original authority or adapter file bytes"]
                ),
            )
            for r in work["source_refs"]
        ],
        objective_stream_ready_score=0,
        stream_rows_reference=work["roster_reference"],
        coefficient_change_count=0,
        admission_identity_checks=[],
        gradient_checks=[],
        restart_checks=[],
        update_rule_hash=upstream.adapter.RULE_HASH,
        pending_high_water=None,
        admission_count=0,
        lost_feedback_count=None,
        update_cost_s=None,
        label_updates=work["label_updates"],
        primitive_reference=reference(Path(work["raw"]) / "measurement.json"),
        publication_output=str(output),
        future_dependencies=[dict(experiment_id=8491, status="future_dependency_not_observation")],
        historical_limits=dict(
            V717_H1=-0.00390625,
            V717_H2=0,
            V717_closed=True,
            V726="exposed historical results preserved",
            V728_journal_equivalence="qualified",
            V728_complete_service_speedup=1.0447833367,
            V728_complete_service_CI95=[0.9942346768, 1.1095895448],
        ),
        methodology_note="No scientific null or benefit is measured. The upstream objective adapter fails qualification before labels. The planned matched four-arm study remains unmeasured. Objective mismatch is a hypothesis. No optional hardware gate applies.",
    )
    value["field_principles"] = {
        k: "Invocation-bound stopped study; missing evidence cannot establish scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        rows="All6000 intended slots retain historical identities and explicit missing metrics.",
        retention_rows="Five read-only windows are intended, but unmeasured; no reserved target reaches learning.",
        coefficient_change_count="Zero means no parameter updates were executed, not evidence against learning.",
        gate_check_summary="Separate absent inputs from failed upstream qualification and owned validation.",
        update_rule_hash="Identifies the planned Brier rule; its presence does not qualify semantics.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Pinned roster, primitive references and saved command receipts defeat summary rehashing."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        for ref in [*work["source_refs"], *work["code_refs"], *work.get("coverage_refs", [])]:
            read(ref) if Path(ref["path"]).suffix == ".json" else sha256_file(Path(ref["path"]))
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        rows, retention = roster()
        if (
            read(work["roster_reference"]) != dict(rows=rows, retention_rows=retention)
            or work["label_updates"] != 0
        ):
            return False
        for receipt in [*work["external_receipts"], *value["validation_receipts"]]:
            saved = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            if saved != receipt:
                return False
            if any(
                sha256_file(Path(receipt[k + "_path"])) != receipt[k + "_sha256"]
                for k in ("stdout", "stderr")
            ):
                return False
            if receipt["passed"] != (
                receipt["actual_exit"] == receipt["expected_exit"]
                and not receipt["timed_out"]
                and receipt["normal_exit"]
            ):
                return False
        upstream_refs = [
            r for r in work["source_refs"] if Path(r["path"]).name == upstream.NAME + ".json"
        ]
        if upstream_refs:
            ref = upstream_refs[0]
            if ref["sha256"] != UPSTREAM_PIN:
                return False
            expected = qualification_failures(Path(ref["path"]), UPSTREAM_PIN, read(ref))
            if any(f not in work["failures"] for f in expected):
                return False
            cold = next(r for r in work["external_receipts"] if r["name"] == "upstream_cold_replay")
            if (
                not cold["passed"]
                and gate(Path(ref["path"]), UPSTREAM_PIN, "cold_replay", True, False)
                not in work["failures"]
            ):
                return False
        return bool(
            value == build(work, value["validation_receipts"], Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
