"""REQ-REPORT-8435: bind primitive reductions separately from administrative readiness.

A measured null survives consumer rejection. All required operands remain
explicit so unavailable history cannot masquerade as a zero learning effect.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting import capacity_learning_report_8434 as upstream
from carnot.reporting import v726_contract_methods as methods
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v709_execution import child
from carnot.verify import block_feedback_state_8433 as state
from carnot.verify import capacity_benefit_audit_8435 as e
from carnot.verify import sparse_energy_7996 as sparse

Json = dict[str, Any]
ROOT, NAME, TASK, CLI, TEST, OWNED = e.ROOT, e.NAME, e.TASK, e.CLI, e.TEST, e.OWNED
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8434_v726_capacity_online_learning.json"
PIN = "sha256:e3bced9f078ce48912394c9047c2d94d44c7cefdc5df1519530b25924caba51c"
reference, read, checksum, progress = e.reference, e.read, e.checksum, e.progress
REQUIRED = {
    "owned_tests",
    "private_E2E018_consumers",
    "private_E2E021",
    "coverage_combine",
    "coverage_report",
    "coverage_json",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
}


def gate(
    path: Path, field: str, expected: Any, observed: Any, producer: str = TASK, digest: Any = None
) -> Json:
    """Name the missing or failed operand rather than replacing it with a metric zero."""
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


def reconstruct(work: Json, source: Json) -> Json:
    """Use pinned initial operands to reject even self-consistently rehashed state changes."""
    roster = [(s, c, a) for s in e.SEEDS for c in (8, 16) for a in e.ARMS]
    if [(c["seed"], c["capacity"], c["arm"]) for c in work["cells"]] != roster:
        raise ValueError("intended_roster")
    dense_checks = 0
    for index, cell in enumerate(work["cells"]):
        if cell.get("configuration"):
            cfg = read(cell["configuration"])
            expected = upstream.e.configuration(
                next(h for h in source["heads"] if h["seed"] == cell["seed"]),
                source["blocks"],
                cell["seed"],
                cell["capacity"],
                cell["arm"],
                source["panel"]["evaluator_custody"],
                source["panel"]["trainer_view"],
            )
            if cfg != expected:
                raise ValueError("frozen_configuration")
            previous, steps = state.initial(cfg), []
            for ref in cell["steps"]:
                step = read(ref)
                if step["before_state"] != previous:
                    raise ValueError("history_order")
                previous = upstream.e.check_step(cfg, step)
                head = deepcopy(step["before_state"]["head"])
                for pending, update in zip(
                    step["sealed_state"]["pending"], step["updates"], strict=True
                ):
                    delta = (
                        np.zeros(109)
                        if cfg["frozen"]
                        else -0.01
                        * sparse.dense_gradient(head, np.asarray(pending["x"]), update["target"])
                    )
                    if not np.allclose(update["coefficient_delta"], delta, atol=1e-12, rtol=1e-12):
                        raise ValueError("dense_write_control")
                    head = (
                        head
                        if cfg["frozen"]
                        else sparse.update(head, np.asarray(pending["x"]), update["target"], 0.01)[
                            0
                        ]
                    )
                    dense_checks += 1
                steps.append(step)
            if cell.get("final") and read(cell["final"]) != previous:
                raise ValueError("final_state")
            if cell.get("retention") and read(cell["retention"]) != upstream.e.retention(
                cfg, steps
            ):
                raise ValueError("retention_primitive")
        progress("reconstructed_cells", index + 1, 39 - index)
    # The shipped adapter reduces primitive fields only, with pinned features and labels.
    with patch.object(upstream, "operands", return_value=source):
        reduced = upstream.reductions(work)
    identities = {cell["identity"]: cell for cell in work["cells"]}
    reduced["admission_rows"] = [
        dict(row, **{k: identities[row["identity"]][k] for k in ("capacity", "seed", "arm")})
        for row in reduced["admission_rows"]
    ]
    reduced["dense_update_controls"] = dense_checks
    return dict(reduced)


def numeric(root: Path) -> Json:
    """The primary authorizes identity and readiness, never imported scientific scores."""
    primary = read(dict(path=str(root / UPSTREAM), sha256=PIN))
    if (
        primary["learning_stream_ready_score"] != 1
        or not primary["required_checks_passed"]
        or primary["flagged_adversarial"]
    ):
        raise ValueError("unqualified_upstream")
    terminal = json.loads(Path(primary["terminal_validation_sidecar_path"]).read_bytes())
    if not read_bound_sidecar(root / UPSTREAM, Path(terminal["publication"]["sidecar_path"]))[
        "report"
    ]["passed"]:
        raise ValueError("upstream_publication")
    work = read(primary["primitive_reference"])
    for ref in [*work["source_refs"], *work["code_refs"], *work["primitive_files"]]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("pinned_primitive_hash")
    source = upstream.operands(root)
    reduced = reconstruct(work, source)
    return dict(
        data=reduced,
        source_refs=[
            dict(path=str(root / UPSTREAM), sha256=PIN),
            primary["primitive_reference"],
            *work["source_refs"],
            *work["primitive_files"],
        ],
        analysis=e.analyze(reduced),
    )


def empty_data() -> Json:
    """Absent required evidence retains all registered answer slots with no imputed score."""
    blocks = methods.protocol()["feedback"]["blocks"]
    rows = [
        dict(
            unit_id=f"s{s}-c{c}-{a}:{q}:{j}",
            question_hash=q,
            family=q,
            seed=s,
            capacity=c,
            arm=a,
            block=b,
            probability=None,
            brier=None,
            decision_loss=None,
            status="censored",
            censored=True,
            excluded=False,
            missing_reason="required_primitive_unavailable",
            absolute_metric=0,
            raw_numerator=0,
            raw_denominator=1,
        )
        for s in e.SEEDS
        for c in (8, 16)
        for a in e.ARMS
        for b, block in enumerate(blocks)
        for q in block
        for j in range(3)
    ]
    return dict(rows=rows, retention_rows=[], admission_rows=[], release_rows=[])


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate only CPU dependencies and reserve validation time before reconstruction."""
    started = time.monotonic_ns()
    progress("preconditions")
    authority = methods.authority(root, raw / "authority")
    failures = [
        gate(
            Path(f.get("artifact_path", str(root))),
            f["check"],
            f.get("expected"),
            f.get("observed"),
            methods.TASK,
            f.get("sha256"),
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
        validation_reserve_s=900,
        child_poll_s=30,
        optional_CUDA_board_V722="not required for independent CPU audit",
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
    try:
        p = read(dict(path=str(root / UPSTREAM), sha256=PIN))
        if p.get("learning_stream_ready_score") != 1:
            failures.append(
                gate(
                    root / UPSTREAM,
                    "learning_stream_ready_score",
                    1,
                    p.get("learning_stream_ready_score"),
                    upstream.TASK,
                    PIN,
                )
            )
        else:
            primitive = p["primitive_reference"]
            try:
                saved = read(primitive)
                for ref in [*saved["source_refs"], *saved["code_refs"], *saved["primitive_files"]]:
                    path = Path(ref["path"])
                    observed = sha256_file(path) if path.is_file() else None
                    if observed != ref["sha256"]:
                        failures.append(
                            gate(
                                path,
                                "required_upstream_primitive_hash",
                                ref["sha256"],
                                observed,
                                upstream.TASK,
                                ref["sha256"],
                            )
                        )
            except (OSError, ValueError) as error:
                failures.append(
                    gate(
                        Path(primitive["path"]),
                        "required_upstream_primitive_hash",
                        primitive["sha256"],
                        None if isinstance(error, FileNotFoundError) else str(error),
                        upstream.TASK,
                        primitive["sha256"],
                    )
                )
    except (OSError, ValueError) as error:
        failures.append(
            gate(
                root / UPSTREAM,
                "required_input_hash",
                PIN,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream.TASK,
                PIN,
            )
        )
    result: Json = dict(data=empty_data(), source_refs=[])
    receipts = []
    if not failures:
        operand = raw / "numeric_reconstruction.json"
        receipts.append(
            child(
                "independent_numeric_reconstruction",
                [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--numeric",
                    str(root),
                    "--operand-output",
                    str(operand),
                ],
                raw / "processes",
                deadline=2400,
            )
        )
        if receipts[-1]["passed"]:
            result = read(reference(operand))
        else:
            failures.append(gate(operand, "owned_numeric_reconstruction", True, False))
    return dict(
        root=str(root),
        authority=authority,
        failures=failures,
        preconditions=resources,
        result=result,
        process_receipts=receipts,
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        adversarial_findings=[],
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Keep scientific effect and consumer qualification separate, including honest zero deltas."""
    result = work["result"]
    data = result["data"]
    analysis = result.get("analysis") or e.analyze(data)
    qualified = (
        REQUIRED <= {r["name"] for r in receipts}
        and all(r["passed"] for r in [*receipts, *work["process_receipts"]])
        and not work["adversarial_findings"]
    )
    owned_failure = any(not r["passed"] for r in [*receipts, *work["process_receipts"]]) or bool(
        work["adversarial_findings"]
    )
    ready = qualified and not work["failures"] and analysis["support_adequate"]
    verdict = (
        "disqualified"
        if owned_failure
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if ready and analysis["exploratory_learning_benefit_score"]
        else "null"
        if ready
        else "disqualified"
    )
    value = dict(
        experiment_id=8435,
        task_id=TASK,
        milestone="2026.10.726",
        run_date="20261012",
        honest_verdict=f"complete_{verdict}_v726_capacity_benefit_audit",
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
        inference_substrate="aggregation_from_upstream_artifacts",
        numeric_inference_substrate="verifier_ensemble_against_cached_candidates",
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
        **data,
        **analysis,
        intended_count=12000,
        completed_count=sum(r["status"] == "completed" for r in data["rows"]),
        failed_count=sum(r["status"] == "failed" for r in data["rows"]),
        censored_count=sum(r["status"] == "censored" for r in data["rows"]),
        excluded_count=0,
        sample_size_budget=dict(
            test_questions=100,
            later_questions=80,
            tune_questions=50,
            intended_question_units=4000,
            intended_answer_rows=12000,
            seeds=list(e.SEEDS),
            siblings=3,
            capacities=[8, 16],
            repeats_are_dependent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="Retrospective public human labels; old overlap unknown; no independent factuality, current Qwen skill or hidden-game generalization.",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=not qualified,
        acceptance_gates=dict(
            required_CPU_operands=not work["failures"],
            owned_validation=qualified,
            support=analysis["support_adequate"],
            primary_lower95_positive=analysis["conditional_ci95"] is not None
            and analysis["conditional_ci95"][0] > 0,
            gain_lower95_positive=analysis["gain_vs_frozen"]["conditional_ci95"] is not None
            and analysis["gain_vs_frozen"]["conditional_ci95"][0] > 0,
            no_decision_loss_increase=analysis["decision_loss_delta"] is not None
            and analysis["decision_loss_delta"] <= 0,
            simultaneous_retention=analysis["retention_simultaneous_upper95"] is not None
            and analysis["retention_simultaneous_upper95"] <= 0.01,
            controls_qualified=analysis["affected_claim_qualified"],
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work["adversarial_findings"],
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(name=r["name"], duration_s=r["duration_s"])
            for r in [*work["process_receipts"], *receipts]
        ],
        source_artifact_hashes=result["source_refs"],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "scripts/experiment_template.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(raw / "measurement.json"), *work.get("coverage_shards", [])],
        cited_upstream_artifacts=[
            dict(
                ref,
                producer=upstream.TASK
                if "8434" in ref["path"]
                else "exp8431-criterion-energy-fit"
                if "8431" in ref["path"]
                else "exp8430-criterion-feature-panel"
                if "8430" in ref["path"]
                else "exp8433-block-checkpoint-qualification"
                if "8433" in ref["path"]
                else methods.TASK,
                imported_fields=[
                    "identity/readiness/primitive custody; no summary scores",
                    "source features and evaluator labels; frozen coefficients; issued predictions; sealed admissions; released updates; retention operands",
                ],
            )
            for ref in result["source_refs"]
        ],
        learning_audit_ready_score=int(ready),
        primary_capacity=8,
        primitive_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        methodology_note="Reconstruct issued block primitives and dense writes; average siblings and seeds within questions. No importance weighting. Independent factuality and population inference remain unqualified.",
    )
    value["field_principles"] = {
        k: "Bind this invocation to replayable retrospective operands; missing values remain distinct from zero."
        for k in value
    }
    value["field_principles"].update(
        acceptance_gates="Fixed scientific thresholds and independent administrative qualification are both required.",
        decision_loss_delta="Uniform minus frozen issued decision loss; exact zero remains an honest metric even if a consumer rejects it.",
        retention_simultaneous_upper95="Fixed maximum of five window increases with identical question resamples in each bootstrap draw.",
        conditional_ci95="Describes dependent exposed histories only; never a population generalization confidence interval.",
        dense_update_controls="Count of recorded feedback writes checked against independent dense gradients, including frozen identity.",
        question_units="Every registered question/arm/seed/capacity slot; dependent repetitions do not increase independent_count.",
        admission_rows="Selected question IDs and known design inclusion probabilities, committed before feedback release.",
        release_rows="Original and applied released labels with hash-bound update operands; shuffled labels stay within their admitted block.",
        trajectory_manifest="Pinned before, sealed and committed checkpoint states for each block.",
        coefficient_change_count="Recorded nonzero coefficient writes; parameter motion alone is not learning benefit.",
        pending_high_water="Maximum learner pending answer records; bounded by three times capacity.",
        lost_feedback_count="Pending records remaining in final committed states; zero is a measured accounting result.",
        block_losses="Each arm, seed, capacity and block's recomputed issued losses, including block0.",
        seed_losses="Each seed/arm/capacity's later-block losses; seeds share one exposed dataset.",
        retention_rows="Read-only tune outcomes reconstructed from each committed boundary; no tune targets enter updates.",
        block_seed_capacity_controls="All200 registered block cells retain observed scores and intended counts.",
        secondary_capacity16="Prespecified secondary contrast using the same question grouping and frozen resampling seed.",
        missing_data_bounds="All intended block slots bound Brier and decision losses in[0,1]; unknown records are not imputed.",
        primary_missing_bounds="Intended80 later questions bound unobserved paired contrasts in[-1,1].",
        inclusion_frequencies="Five-seed empirical admission frequency beside the design probability; seeds are not new datasets.",
        control_degeneracy="Exact issued-prediction equality detects ineffective shuffled or frozen comparisons.",
        support_adequate="Requires all12000 answer rows,80 paired later questions and50 complete retention questions.",
        affected_claim_qualified="Scientific benefit requires complete support and nondegenerate matched controls.",
        scientific_outcome="Unmeasured, measured null or exploratory positive is separate from administrative verdict_class.",
        gain_vs_frozen="Frozen minus uniform Brier with paired whole-question conditional uncertainty.",
        exploratory_learning_benefit_score="One only when all frozen benefit, retention, decision and control conditions hold.",
        next_evidence_condition="A null retires this exact unchanged attempt; a positive earns only a future independent stream.",
    )
    value["acceptance_gate_principles"] = dict(
        required_CPU_operands="Only authenticated task authority and required CPU evidence gate this task.",
        owned_validation="Every frozen scoped check must pass with durable receipts; all consumer findings remain binding.",
        support="Preserve the full intended denominator and all five retention windows.",
        primary_lower95_positive="Capacity8 FIFO minus uniform conditional lower95 must exceed zero.",
        gain_lower95_positive="Frozen minus uniform conditional lower95 must exceed zero.",
        no_decision_loss_increase="Uniform minus frozen issued decision loss must be<=0, including honest zero.",
        simultaneous_retention="Maximum-window one-sided95 Brier-increase bound must be<=0.01.",
        controls_qualified="Matched controls must be nondegenerate and supported; five seeds share one dataset.",
    )
    value["field_principles"]["acceptance_gate_principles"] = (
        "Explain every frozen scientific and administrative acceptance gate."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """A new process reconstructs operands rather than accepting rehashed aggregate claims."""
    from tempfile import TemporaryDirectory

    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        for ref in [*value["code_config_hashes"], *work.get("coverage_shards", [])]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        with TemporaryDirectory(prefix="exp8435-authority-", dir="/var/tmp") as directory:
            cold = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                saved = work["authority"]["authority_snapshots"][role]
                if "snapshot_path" in saved:
                    original = Path(saved["snapshot_path"])
                    if sha256_file(original) != saved["sha256"]:
                        return False
                    target = cold / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(original.read_bytes())
            if (
                methods.authority(cold, cold / "checked")["activated"]
                != work["authority"]["activated"]
            ):
                return False
        if work["result"]["source_refs"]:
            if numeric(Path(work["root"])) != work["result"]:
                return False
        elif work["result"] != dict(data=empty_data(), source_refs=[]) or not work["failures"]:
            return False
        if not upstream.receipts_valid(
            [*work["process_receipts"], *value["validation_receipts"]], work["adversarial_findings"]
        ):
            return False
        raw = Path(value["primitive_reference"]["path"]).parent
        return bool(
            value
            == build(work, value["validation_receipts"], raw, Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False
