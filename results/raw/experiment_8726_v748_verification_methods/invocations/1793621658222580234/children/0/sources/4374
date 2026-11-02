"""REQ-REPORT-8434: qualify causal histories separately from retrospective benefit.

Independent upstream seals bind every intended answer to the frozen protocol.
Scientific reductions preserve missing slots and never multiply question counts.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from typing import Any, cast

import numpy as np

from carnot.reporting import v726_contract_methods as methods
from carnot.reporting import block_checkpoint_qualification_8433 as qualification
from carnot.reporting import v726_criterion_decision_audit as upstream
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v709_execution import child
from carnot.verify import block_feedback_state_8433 as state
from carnot.verify import capacity_online_learning_8434 as e

Json = dict[str, Any]
ROOT, NAME, TASK, CLI = e.ROOT, e.NAME, e.TASK, e.CLI
TEST = "tests/python/test_capacity_online_learning_8434.py"
OWNED = [
    "python/carnot/verify/capacity_online_learning_8434.py",
    "python/carnot/reporting/capacity_learning_report_8434.py",
    "python/carnot/reporting/capacity_learning_runner_8434.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
STORE_PATH = "results/experiment_8433_v726_block_checkpoint_qualification.json"
STORE_PIN = "sha256:5cbdfe3e9b9498c1430fa125e53b635c72b0979bed8522d4fa3183b7991a704c"
REQUIRED_CHECKS = qualification.REQUIRED_CHECKS
reference, read, checksum, progress = e.reference, e.read, methods.checksum, e.progress


def gate(
    path: Path,
    field: str,
    expected: Any,
    observed: Any,
    *,
    producer: str = TASK,
    digest: Any = None,
) -> Json:
    """Name the failed operand so absent evidence remains distinct from an observed zero."""
    return dict(upstream.gate(path, field, expected, observed, upstream=producer, digest=digest))


def operands(root: Path) -> Json:
    """The decision audit supplies feature custody, never a scientific gate for learning."""
    head_receipt, panel, frozen, features = upstream.operands(root)
    store_ref = dict(path=str(root / STORE_PATH), sha256=STORE_PIN)
    store = read(store_ref)
    if (
        store["block_store_ready_score"] != 1
        or not store["required_checks_passed"]
        or store["flagged_adversarial"]
        or store["checkpoint_schema"] != state.SCHEMA
    ):
        raise ValueError("unqualified_checkpoint")
    terminal = json.loads(Path(store["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(root / STORE_PATH, Path(terminal["publication"]["sidecar_path"]))
    if not bound["report"]["passed"]:
        raise ValueError("unqualified_checkpoint_publication")
    protocol = methods.protocol()
    by_question: Json = {}
    for row in features["rows"]:
        if row["role"] == "test":
            by_question.setdefault(row["question_hash"], []).append(
                dict(id=row["unit_id"], x=row["x"])
            )
    blocks = [
        [dict(id=q, answers=by_question[q]) for q in block]
        for block in protocol["feedback"]["blocks"]
    ]
    if any(a["x"] is None for block in blocks for q in block for a in q["answers"]):
        raise ValueError("required_test_feature_missing")
    sources = [
        dict(path=str(root / upstream.UPSTREAM), sha256=upstream.UPSTREAM_PIN),
        dict(path=str(root / upstream.fit.UPSTREAM), sha256=upstream.fit.UPSTREAM_PIN),
        store_ref,
        reference(ROOT / methods.PROTOCOL),
        dict(path=head_receipt["frozen_heads_path"], sha256=head_receipt["frozen_heads_sha256"]),
        dict(path=panel["feature_panel_path"], sha256=panel["feature_panel_sha256"]),
        panel["evaluator_custody"],
        panel["trainer_view"],
    ]
    return dict(heads=frozen["heads"]["spline9"], blocks=blocks, panel=panel, source_refs=sources)


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Check actual CPU operands and resources before any bounded learning child runs."""
    started = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions", 0, 40)
    authority = methods.authority(root, raw / "authority")
    failures = [
        gate(
            Path(f.get("path", f.get("artifact_path", str(root)))),
            f.get("check", "authority"),
            f.get("expected", True),
            f.get("observed"),
            producer=methods.TASK,
            digest=f.get("sha256"),
        )
        for f in authority["gate_check_summary"]
    ]
    resources = dict(
        methods.legacy.resources(private),
        free_disk_bytes=shutil.disk_usage(private).free,
        tools={
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
        task_cap_s=4800,
        measurement_cap_s=2400,
        seed_capacity_cap_s=240,
        validation_reserve_s=900,
        child_poll_s=30,
        model_loads=0,
        generations=0,
    )
    if (
        not resources["private_disk_backed_scratch"]
        or resources["available_memory_bytes"] < resources["minimum_memory_bytes"]
        or resources["free_disk_bytes"] < 104857600
        or not all(resources["tools"].values())
    ):
        failures.append(gate(private, "private_disk_memory_tools_deadline", True, resources))
    source: Json = {}
    for relative, digest, field, producer in [
        (
            upstream.UPSTREAM,
            upstream.UPSTREAM_PIN,
            "head_ready_score",
            "exp8431-criterion-energy-fit",
        ),
        (
            upstream.fit.UPSTREAM,
            upstream.fit.UPSTREAM_PIN,
            "feature_panel_ready_score",
            "exp8430-criterion-feature-panel",
        ),
        (
            STORE_PATH,
            STORE_PIN,
            "block_store_ready_score",
            "exp8433-block-checkpoint-qualification",
        ),
    ]:
        path = root / relative
        try:
            receipt = read(dict(path=str(path), sha256=digest))
            if receipt.get(field) != 1:
                failures.append(
                    gate(path, field, 1, receipt.get(field), producer=producer, digest=digest)
                )
        except (OSError, ValueError) as error:
            failures.append(
                gate(
                    path,
                    "required_input_hash",
                    digest,
                    None if isinstance(error, FileNotFoundError) else str(error),
                    producer=producer,
                    digest=digest,
                )
            )
    try:
        source = operands(root)
    except (OSError, ValueError, KeyError) as error:
        failures.append(
            gate(
                root / upstream.UPSTREAM,
                "qualified_head_panel_checkpoint",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
                producer="exp8431-criterion-energy-fit/exp8433-block-checkpoint-qualification",
                digest=upstream.UPSTREAM_PIN,
            )
        )
    cells, receipts, restarts = [], [], []
    for seed in e.SEEDS:
        for capacity in (8, 16):
            family_start = time.monotonic()
            for arm in e.ARMS:
                identity = f"s{seed}-c{capacity}-{arm}"
                cell: Json = dict(
                    identity=identity,
                    seed=seed,
                    capacity=capacity,
                    arm=arm,
                    status="censored",
                    missing_reason="required_input_unavailable",
                    steps=[],
                )
                cells.append(cell)
                remaining = min(
                    240 - (time.monotonic() - family_start),
                    2400 - (time.monotonic_ns() - started) / 1e9,
                )
                if failures or remaining <= 0:
                    cell["missing_reason"] = (
                        "measurement_deadline" if remaining <= 0 else cell["missing_reason"]
                    )
                    continue
                cfg = e.configuration(
                    next(h for h in source["heads"] if h["seed"] == seed),
                    source["blocks"],
                    seed,
                    capacity,
                    arm,
                    source["panel"]["evaluator_custody"],
                    source["panel"]["trainer_view"],
                )
                config_path, directory = (
                    raw / "configurations" / (identity + ".json"),
                    raw / "cells" / identity,
                )
                atomic_json(config_path, cfg)
                argv = [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--worker",
                    str(directory),
                    "--configuration",
                    str(config_path),
                ]
                receipt = child(identity, argv, raw / "processes", deadline=remaining)
                receipts.append(receipt)
                cell.update(
                    configuration=reference(config_path),
                    steps=[reference(p) for p in sorted((directory / "steps").glob("*.json"))],
                    status="completed" if receipt["passed"] else "failed",
                    missing_reason=None if receipt["passed"] else "owned_child_failed",
                )
                if (directory / "final.json").is_file():
                    cell["final"] = reference(directory / "final.json")
                retention_path = directory / "retention.json"
                receipts.append(
                    child(
                        identity + "-retention",
                        [
                            str(ROOT / ".venv/bin/python"),
                            "-u",
                            str(ROOT / CLI),
                            "--retention",
                            str(directory),
                            "--configuration",
                            str(config_path),
                            "--operand-output",
                            str(retention_path),
                        ],
                        raw / "processes",
                        deadline=max(0.01, min(60, remaining - receipt["duration_s"])),
                    )
                )
                if retention_path.is_file():
                    cell["retention"] = reference(retention_path)
                if seed == 17 and capacity == 8:
                    resumed_dir = raw / "restarts" / arm
                    restart_argv = [*argv[:4], str(resumed_dir), *argv[5:]]
                    for suffix, args, expected in [("exit", ["--crash"], 73), ("resume", [], 0)]:
                        remaining = min(
                            240 - (time.monotonic() - family_start),
                            2400 - (time.monotonic_ns() - started) / 1e9,
                        )
                        receipts.append(
                            child(
                                identity + "-" + suffix,
                                [*restart_argv, *args],
                                raw / "processes",
                                expected=expected,
                                deadline=max(0.01, remaining),
                            )
                        )
                    restart_steps = [
                        reference(p) for p in sorted((resumed_dir / "steps").glob("*.json"))
                    ]
                    equal = (
                        (resumed_dir / "final.json").is_file()
                        and "final" in cell
                        and read(reference(resumed_dir / "final.json")) == read(cell["final"])
                    )
                    restarts.append(
                        dict(
                            arm=arm,
                            seed=17,
                            capacity=8,
                            actual_exit=receipts[-2]["actual_exit"],
                            passed=bool(equal and all(r["passed"] for r in receipts[-2:])),
                            steps=restart_steps,
                            final=reference(resumed_dir / "final.json") if equal else None,
                        )
                    )
                progress("cell_complete", len(cells), 40 - len(cells))
    work = dict(
        root=str(root),
        cells=cells,
        restart_rows=restarts,
        failures=failures,
        authority=authority,
        preconditions=resources,
        source_refs=source.get("source_refs", []),
        code_refs=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                "python/carnot/verify/block_feedback_state_8433.py",
                "python/carnot/verify/sparse_energy_7996.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "scripts/experiment_template.py",
            ]
        ],
        process_receipts=receipts,
        independent_update=e.independent_update(),
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        adversarial_findings=[],
    )
    work["primitive_files"] = [
        reference(path)
        for path in sorted(raw.rglob("*.json"))
        if any(part in {"cells", "restarts", "configurations", "processes"} for part in path.parts)
    ]
    work["process_receipts"] += [
        json.loads(path.read_bytes())
        for path in sorted(raw.rglob("*.receipt.json"))
        if "evaluator_processes" in path.parts
    ]
    atomic_json(raw / "measurement.json", work)
    return work


def reductions(work: Json) -> Json:
    """Keep every intended slot while averaging siblings only within real question units."""
    original = operands(ROOT)
    blocks = original["blocks"]
    targets = {r["unit_id"]: r["y"] for r in read(original["panel"]["evaluator_custody"])["rows"]}
    rows, questions, admissions, releases, trajectory = [], [], [], [], []
    retention_rows: list[Json] = []
    changes, lost, high_water = 0, 0, {"8": 0, "16": 0}
    for cell in work["cells"]:
        history = {read(ref)["block"]: read(ref) for ref in cell["steps"]}
        cfg = read(cell["configuration"]) if cell.get("configuration") else None
        for block, records in enumerate(blocks):
            step = history.get(block)
            probabilities = {r["id"]: r for r in step["predictions"]} if step else {}
            updates = (
                {r["id"]: (index, r) for index, r in enumerate(step["updates"])} if step else {}
            )
            chosen = set(step["sealed_state"]["selected_cluster_ids"]) if step else set()
            if step:
                high_water[str(cell["capacity"])] = max(
                    high_water[str(cell["capacity"])], step["sealed_state"]["pending_high_water"]
                )
                trajectory.append(
                    dict(
                        identity=cell["identity"],
                        block=block,
                        before=canonical_hash(step["before_state"]),
                        sealed=step["sealed_checkpoint_hash"],
                        committed=step["checkpoint_hash"],
                        primitive=cell["steps"][block],
                    )
                )
            for position, q in enumerate(records):
                admission_probability = (
                    float(position < cell["capacity"])
                    if cell["arm"] in e.ARMS[:2]
                    else cell["capacity"] / 20
                )
                admissions.append(
                    dict(
                        identity=cell["identity"],
                        block=block,
                        question_hash=q["id"],
                        admitted=q["id"] in chosen if step else None,
                        inclusion_probability=admission_probability,
                        admission_target_reads=0 if step else None,
                    )
                )
                siblings = []
                for a in q["answers"]:
                    prediction = probabilities.get(a["id"])
                    p: Any = prediction["probability"] if prediction else None
                    y: Any = targets.get(a["id"])
                    valid = p is not None and y is not None
                    update_slot = updates.get(a["id"])
                    update = update_slot[1] if update_slot else None
                    update_ref = (
                        dict(cell["steps"][block], json_pointer=f"/updates/{update_slot[0]}")
                        if update_slot
                        else None
                    )
                    if update:
                        changes += int(any(delta != 0 for delta in update["coefficient_delta"]))
                        releases.append(
                            dict(
                                identity=cell["identity"],
                                block=block,
                                id=update["id"],
                                target=update["target"],
                                original_target=update["original_target"],
                                feedback_id=update["feedback_id"],
                                log_loss=update["log_loss"],
                                parameter_write=update["parameter_write"],
                                before_head_hash=update["before_head_hash"],
                                after_head_hash=update["after_head_hash"],
                                primitive=update_ref,
                            )
                        )
                    action = (
                        "escalate"
                        if p is None or 0.2 < p < 0.8
                        else "accept"
                        if p <= 0.2
                        else "reject"
                    )
                    loss = (
                        None
                        if not valid
                        else 0.2
                        if action == "escalate"
                        else float(y == (action == "accept"))
                    )
                    row = dict(
                        unit_id=f"{cell['identity']}:{a['id']}",
                        answer_id=a["id"],
                        question_hash=q["id"],
                        family=q["id"],
                        arm=cell["arm"],
                        seed=cell["seed"],
                        capacity=cell["capacity"],
                        block=block,
                        status="completed"
                        if valid
                        else "failed"
                        if cell["status"] == "failed"
                        else "censored",
                        censored=not valid and cell["status"] != "failed",
                        excluded=False,
                        missing_reason=None
                        if valid
                        else cell["missing_reason"] or "missing_prediction_or_target",
                        probability=p,
                        evaluator_target=y,
                        admitted=q["id"] in chosen if step else None,
                        released_feedback=step["released_feedback"].get(a["id"]) if step else None,
                        inclusion_probability=admission_probability,
                        brier=(p - y) ** 2 if valid else None,
                        log_loss=float(-y * np.log(p) - (1 - y) * np.log1p(-p)) if valid else None,
                        action=action,
                        decision_loss=loss,
                        later_loss_eligible=block in (1, 2, 3, 4),
                        issued_wall_ns=step["issued_wall_ns"] if step else None,
                        released_wall_ns=cast(Json, step)["released_wall_ns"] if update else None,
                        issued_version=cast(Json, prediction)["issued_version"]
                        if prediction
                        else None,
                        issued_head_hash=prediction["head_hash"] if prediction else None,
                        checkpoint_hash=step["checkpoint_hash"] if step else None,
                        update_evidence=update_ref,
                        absolute_metric=int(valid),
                        raw_numerator=int(valid),
                        raw_denominator=1,
                    )
                    rows.append(row)
                    siblings.append(row)
                questions.append(
                    dict(
                        identity=cell["identity"],
                        question_hash=q["id"],
                        block=block,
                        arm=cell["arm"],
                        seed=cell["seed"],
                        capacity=cell["capacity"],
                        completed_answers=sum(r["status"] == "completed" for r in siblings),
                        brier=float(np.mean([r["brier"] for r in siblings]))
                        if all(r["brier"] is not None for r in siblings)
                        else None,
                    )
                )
        if cell.get("retention"):
            retention_rows.extend(
                dict(
                    r,
                    identity=cell["identity"],
                    seed=cell["seed"],
                    capacity=cell["capacity"],
                    arm=cell["arm"],
                )
                for r in read(cell["retention"])["rows"]
            )
        if cfg and cell.get("final"):
            lost += len(read(cell["final"])["pending"])
    block_losses, seed_losses = [], []
    for cell in work["cells"]:
        subset = [
            r
            for r in rows
            if r["seed"] == cell["seed"]
            and r["capacity"] == cell["capacity"]
            and r["arm"] == cell["arm"]
        ]
        for block in range(5):
            observed = [r for r in subset if r["block"] == block and r["brier"] is not None]
            block_losses.append(
                dict(
                    identity=cell["identity"],
                    block=block,
                    later_loss_eligible=block > 0,
                    completed=len(observed),
                    intended=60,
                    **{
                        k: float(np.mean([r[k] for r in observed])) if observed else None
                        for k in ("brier", "log_loss", "decision_loss")
                    },
                )
            )
        observed = [r for r in subset if r["block"] > 0 and r["brier"] is not None]
        seed_losses.append(
            dict(
                identity=cell["identity"],
                seed=cell["seed"],
                capacity=cell["capacity"],
                arm=cell["arm"],
                intended=240,
                completed=len(observed),
                **{
                    k: float(np.mean([r[k] for r in observed])) if observed else None
                    for k in ("brier", "log_loss", "decision_loss")
                },
            )
        )
    return dict(
        rows=rows,
        question_units=questions,
        admission_rows=admissions,
        release_rows=releases,
        retention_rows=retention_rows,
        trajectory_manifest=trajectory,
        coefficient_change_count=changes,
        pending_high_water=high_water,
        lost_feedback_count=lost,
        block_losses=block_losses,
        seed_losses=seed_losses,
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """A causal stream can qualify with any loss contrast; scientific benefit needs its own audit."""
    reduced = reductions(work)
    qualified = (
        REQUIRED_CHECKS <= {r["name"] for r in receipts}
        and all(r["passed"] for r in receipts)
        and not work.get("adversarial_findings")
        and all(r["passed"] for r in work["process_receipts"])
    )
    causal = (
        all(c["status"] == "completed" for c in work["cells"])
        and len(reduced["rows"]) == 12000
        and all(r["status"] == "completed" for r in reduced["rows"])
        and len(reduced["retention_rows"]) == 30000
        and len(work["restart_rows"]) == 4
        and all(r["passed"] for r in work["restart_rows"])
        and reduced["coefficient_change_count"] > 0
        and work["independent_update"]["passed"]
    )
    ready = qualified and causal and not work["failures"]
    owned_failure = any(not r["passed"] for r in [*receipts, *work["process_receipts"]])
    verdict = (
        "disqualified"
        if owned_failure
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if ready
        else "disqualified"
    )
    value = dict(
        experiment_id=8434,
        task_id=TASK,
        milestone="2026.10.726",
        run_date="20261012",
        honest_verdict=f"complete_{verdict}_v726_capacity_online_learning",
        verdict_class=verdict,
        gate_check_summary=[
            *work["failures"],
            *[
                gate(
                    Path(r["stdout_path"]),
                    "required_owned_check",
                    True,
                    False,
                    digest=r["stdout_sha256"],
                )
                for r in [*receipts, *work["process_receipts"]]
                if not r["passed"]
            ],
        ],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        reader_inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=[
            dict(
                models=["GPT-5-mini", "gemma-2-9b", "llama3-8b", "mistral-7b"],
                scope="cached released judgments and answers",
            )
        ],
        **reduced,
        intended_count=12000,
        completed_count=sum(r["status"] == "completed" for r in reduced["rows"]),
        failed_count=sum(r["status"] == "failed" for r in reduced["rows"]),
        censored_count=sum(r["status"] == "censored" for r in reduced["rows"]),
        excluded_count=0,
        independent_count=100 if causal else 0,
        sample_size_budget=dict(
            test_questions=100,
            tune_questions=50,
            intended_question_units=4000,
            intended_answer_rows=12000,
            siblings=3,
            seeds=list(e.SEEDS),
            capacities=[8, 16],
            repeats_are_dependent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="retrospective released public labels; old overlap unknown; no current Qwen inference",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=not qualified,
        acceptance_gates=dict(
            qualified_causal_measurements=causal,
            owned_validation=qualified,
            required_cpu_inputs=not work["failures"],
            favorable_contrast_required=False,
            independent_generalization=False,
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
            for r in [*work["process_receipts"], *receipts]
        ],
        random_seed=list(e.SEEDS),
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=[
            *work["code_refs"],
            *[c["configuration"] for c in work["cells"] if c.get("configuration")],
        ],
        raw_shard_hashes=work["primitive_files"] + work.get("coverage_shards", []),
        cited_upstream_artifacts=[
            dict(ref, producer=producer, imported_fields=fields)
            for ref, (producer, fields) in zip(
                work["source_refs"],
                [
                    (
                        "exp8431-criterion-energy-fit",
                        [
                            "head_ready_score",
                            "required_checks_passed",
                            "frozen_heads_path",
                            "frozen_heads_sha256",
                        ],
                    ),
                    (
                        "exp8430-criterion-feature-panel",
                        [
                            "feature_panel_path",
                            "feature_panel_sha256",
                            "evaluator_custody",
                            "trainer_view",
                        ],
                    ),
                    (
                        "exp8433-block-checkpoint-qualification",
                        ["block_store_ready_score", "checkpoint_schema", "required_checks_passed"],
                    ),
                    ("exp8429-contract-methods", ["feedback", "fit", "decision"]),
                    ("exp8431-criterion-energy-fit", ["heads.spline9"]),
                    (
                        "exp8430-criterion-feature-panel",
                        ["rows[].x", "rows[].question_hash", "rows[].unit_id"],
                    ),
                    ("exp8430-criterion-feature-panel", ["rows[].y", "rows[].unit_id"]),
                    ("exp8430-criterion-feature-panel", ["rows[role=tune].x", "rows[role=tune].y"]),
                ],
            )
        ],
        learning_stream_ready_score=int(ready),
        stream_rows_path=str(raw / "stream_rows.json"),
        stream_rows_sha256=sha256_file(raw / "stream_rows.json")
        if (raw / "stream_rows.json").is_file()
        else None,
        restart_rows=work["restart_rows"],
        inclusion_probabilities=dict(FIFO="1 for first C; 0 otherwise", uniform="C/20"),
        primitive_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        methodology_note="Fixed rate 0.01 and L2 0.001; two durable transactions per block; no importance weighting. Block-admission ablation differs from the cited delayed-OCO scheduler. Benefit is unqualified until Exp8435.",
    )
    value["field_principles"] = {
        k: "Bound to this invocation's causal retrospective evidence; no independent factuality or current-model skill."
        for k in value
    }
    value["field_principles"].update(
        learning_stream_ready_score="Requires qualified causal measurements; no favorable contrast requirement.",
        coefficient_change_count="Real committed parameter updates following issued predictions; frozen writes zero.",
        retention_rows="Read-only tune evaluation; no retention target reaches the updater.",
        restart_rows="Actual process death and exact recovery, separate from learning benefit.",
        rows="Retain intended slots; sibling answers and seeds do not multiply independent questions.",
        release_rows="Admitted original and applied targets plus hash-bound pointers to full loss, gradient and coefficient-delta operands.",
        acceptance_gates="Causal completeness, executed validation and actual CPU operands govern readiness; favorable contrasts and independent generalization are separate.",
        trajectory_manifest="Both durable transactions and every parameter version bind to the full primitive block state.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def receipts_valid(receipts: list[Json], findings: list[Json]) -> bool:
    """Derive qualification from durable command and stream receipts, never a claimed flag."""
    for receipt in receipts:
        saved = json.loads(Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes())
        if any(receipt.get(k) != v for k, v in saved.items() if k != "passed"):
            return False
        if any(
            sha256_file(Path(receipt[k + "_path"])) != receipt[k + "_sha256"]
            for k in ("stdout", "stderr")
        ):
            return False
        passed = (
            saved["actual_exit"] == saved["exit_code"] == saved["expected_exit"]
            and not saved["timed_out"]
            and saved["normal_exit"]
        )
        if receipt["name"] == "adversarial":
            report = json.loads(Path(receipt["stdout_path"]).read_bytes())
            flags = [f for row in report["reports"] for f in row.get("flags", [])]
            if flags != findings:
                return False
            passed = passed and not flags
        if receipt["passed"] != passed:
            return False
    return True


def replay(path: Path) -> bool:
    """Cold replay reconstructs every history with independently hash-pinned upstream inputs."""
    from tempfile import TemporaryDirectory

    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        progress("cold_replay_start", 0, 40)
        for ref in [
            *work["source_refs"],
            *work["code_refs"],
            *work["primitive_files"],
            *work.get("coverage_shards", []),
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        with TemporaryDirectory(prefix="exp8434-cold-", dir="/var/tmp") as directory:
            cold = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                snap = work["authority"]["authority_snapshots"][role]
                if "snapshot_path" in snap:
                    original = Path(snap["snapshot_path"])
                    if sha256_file(original) != snap["sha256"]:
                        return False
                    target = cold / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(original.read_bytes())
            if (
                methods.authority(cold, cold / "authority")["activated"]
                != work["authority"]["activated"]
            ):
                return False
        source = operands(Path(work["root"])) if not work["failures"] else None
        roster = [(seed, c, arm) for seed in e.SEEDS for c in (8, 16) for arm in e.ARMS]
        if [(c["seed"], c["capacity"], c["arm"]) for c in work["cells"]] != roster:
            return False
        for index, cell in enumerate(work["cells"]):
            if cell.get("configuration"):
                cfg = read(cell["configuration"])
                if source is None:
                    return False
                expected_cfg = e.configuration(
                    next(h for h in source["heads"] if h["seed"] == cell["seed"]),
                    source["blocks"],
                    cell["seed"],
                    cell["capacity"],
                    cell["arm"],
                    source["panel"]["evaluator_custody"],
                    source["panel"]["trainer_view"],
                )
                if cfg != expected_cfg:
                    return False
                previous = state.initial(cfg)
                stream = []
                for ref in cell["steps"]:
                    step = read(ref)
                    if step["before_state"] != previous:
                        return False
                    previous = e.check_step(cfg, step)
                    stream.append(step)
                if cell.get("final") and read(cell["final"]) != previous:
                    return False
                if cell.get("retention") and read(cell["retention"]) != e.retention(cfg, stream):
                    return False
                matching = [
                    r
                    for r in work["restart_rows"]
                    if r["arm"] == cell["arm"] and cell["seed"] == 17 and cell["capacity"] == 8
                ]
                for restart in matching:
                    restarted = [read(ref) for ref in restart["steps"]]
                    equal = (
                        [e.semantic_step(s) for s in stream]
                        == [e.semantic_step(s) for s in restarted]
                        and restart["final"] is not None
                        and read(restart["final"]) == previous
                        and restart["actual_exit"] == 73
                    )
                    if restart["passed"] != equal:
                        return False
            progress("cold_replay_cells", index + 1, 39 - index)
        if work["independent_update"] != e.independent_update():
            return False
        if not receipts_valid(
            [*work["process_receipts"], *value["validation_receipts"]],
            work.get("adversarial_findings", []),
        ):
            return False
        raw = Path(value["primitive_reference"]["path"]).parent
        if read(reference(raw / "stream_rows.json")) != dict(rows=reductions(work)["rows"]):
            return False
        return bool(
            value
            == build(work, value["validation_receipts"], raw, Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError):
        return False
