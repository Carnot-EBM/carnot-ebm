"""REQ-REPORT-8458: freeze information controls without fitting or reading test targets.

The released panel fixes question roles and historical exposure. Small adapters
retain original full-head inputs and move whole source bundles. A later task
fits the missing scalar spline; this task measures control readiness only.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np

from carnot.reporting import v726_feature_panel as base
from carnot.reporting import v727_historical_inputs as history_reader
from carnot.reporting import v728_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify import v726_criterion_energy_fit as fitted
from carnot.verify.runtime_evidence_execution_8382 import resources

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8458_v728_source_control_panel", "exp8458-source-control-panel"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v728_source_control_panel_8458.py"
OWNED = [
    "python/carnot/reporting/v728_source_control_panel.py",
    "python/carnot/reporting/v728_source_control_runner.py",
    CLI,
]
ARMS = (
    "raw_p2",
    "scalar_logistic",
    "scalar_cubic_spline",
    "full_spline9",
    "source_permuted_spline9",
)
MODEL_SPECS: list[Json] = []
read, reference, checksum, sha256_file = (
    methods.read,
    methods.reference,
    methods.checksum,
    methods.sha256_file,
)
measurement_clock = time.monotonic_ns


def failure(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Bind missing operands and failed checks to this task instead of the imported producer."""
    return dict(base.failure(path, field, expected, observed), upstream_id=TASK)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Visible actual counts distinguish bounded computation from a stalled process."""
    print(f"[exp8458] phase={phase} completed={completed} pending={pending}", flush=True)


def predictor_read(ref: Json, evaluator: Json, log: list[Json]) -> Json:
    """Deny evaluator custody before opening bytes, even when a caller holds its reference."""
    denied = Path(ref["path"]).resolve() == Path(evaluator["path"]).resolve()
    log.append(
        dict(reference=ref, purpose="evaluator" if denied else "prediction_visible", denied=denied)
    )
    if denied:
        raise PermissionError("evaluator_label_read_denied")
    return dict(read(ref))


def scalar_basis(q: list[float], scaler: Json) -> list[list[float]]:
    """Only q's original fit bounds determine the cubic basis; source edits cannot enter it."""
    low, high = scaler["minimum"][0], scaler["maximum"][0]
    scaled = np.clip((np.asarray(q) - low) / (high - low if high > low else 1), 0, 1)
    matrix = np.column_stack(
        (sparse.BSpline.design_matrix(scaled, sparse.KNOTS, 3).toarray(), np.ones(len(q)))
    )
    return list(matrix.tolist())


def freeze(upstream: Json, heads: Json, protocol: Json, log: list[Json]) -> Json:
    """The declared operators preserve original vectors, roles and evaluator target custody."""
    roles = protocol["partition"]["roles"]
    role_by_question = {q: role for role, questions in roles.items() for q in questions}
    original = upstream["rows"]
    support = upstream["class_support"]
    if (
        len(original) != 900
        or len({r["unit_id"] for r in original}) != 900
        or Counter(r["question_hash"] for r in original)
        != Counter({q: 3 for q in role_by_question})
        or any(r["role"] != role_by_question[r["question_hash"]] or "y" in r for r in original)
        or any(
            support[role]["usable_questions"] < floor
            or (role != "fit" and min(support[role]["class_questions"].values()) < 8)
            for role, floor in zip(("fit", "tune", "test"), (128, 32, 80), strict=True)
        )
        or len(heads["scaler"]["minimum"]) != 9
        or len(heads["scaler"]["maximum"]) != 9
    ):
        raise ValueError("row_role_support_or_head_contract")
    mappings = fitted.donor_mappings(upstream["stratum_roster"])
    usable = [r for r in original if r["x"] is not None]
    permuted = fitted.permute_rows(usable, [m for group in mappings.values() for m in group])
    source_x = {r["unit_id"]: r["x"] for r in permuted}
    donors = {
        m["recipient_unit_id"]: m["donor_unit_id"] for group in mappings.values() for m in group
    }
    joins = {r["unit_id"]: r for r in upstream["source_join_rows"]}
    strata = {r["unit_id"]: r for r in upstream["stratum_roster"]}
    rows = []
    for index, original_row in enumerate(original):
        if index % 100 == 0:
            progress("freeze_controls", index * 5, (900 - index) * 5)
        uid, qid = original_row["unit_id"], original_row["question_hash"]
        for arm in ARMS:
            x = original_row["x"]
            vector = (
                None
                if x is None
                else source_x[uid]
                if arm == ARMS[-1]
                else x
                if arm == ARMS[-2]
                else [x[0]]
            )
            row = dict(
                original_row,
                unit_id=f"{uid}:{arm}",
                answer_id=uid,
                arm=arm,
                x=vector,
                predictor_names=upstream["feature_schema"]["names"]
                if arm in ARMS[-2:]
                else ["p2_hallucination_probability"],
                predictor_vector_hash=canonical_hash(vector),
                **{
                    k: v
                    for k, v in joins[uid].items()
                    if k not in {"unit_id", "role", "question_hash"}
                },
                reporting_stratum=strata[uid],
                donor_answer_id=donors[uid] if arm == ARMS[-1] else uid,
                target_custody=dict(
                    answer_id=uid,
                    question_hash=qid,
                    role=original_row["role"],
                    values_visible_to_predictor=False,
                ),
            )
            rows.append(row)
    result = {
        k: deepcopy(upstream[k])
        for k in [
            "class_support",
            "source_join_rows",
            "stratum_roster",
            "partition_hashes",
            "question_exclusions",
            "exposure_ledger",
        ]
    }
    result.update(
        rows=rows,
        support_passed=True,
        row_complete=len(rows) == 4500,
        question_roles=roles,
        donor_mappings=mappings,
        predictor_access_log=deepcopy(log),
        control_parameter_counts=dict(
            zip(ARMS, [0, 2, methods.scalar_dimension(), 109, 109], strict=True)
        ),
        operators=dict(
            arms=list(ARMS),
            permutation_seeds=protocol["source_controls"]["permutation_seeds"],
            knots=sparse.KNOTS,
            degree=3,
            scaler=heads["scaler"],
            frozen_before_fit=True,
            generator_is_predictor=False,
            test_target_reads=0,
            target_primary="A1",
            target_sensitivity="A2",
        ),
        scalar_basis_probe=scalar_basis([0.1, 0.5, 0.9], heads["scaler"]),
    )
    progress("controls_frozen", 4500, 0)
    return result


def historical(root: Path, raw: Path) -> Json:
    """Original sealed authority qualifies each historical producer in a bounded process."""
    authority = history_reader.custody(root, raw / "authority")
    families: list[Json] = []
    refs: list[Json] = []
    receipts: list[Json] = []
    failures: list[Json] = []
    for number, name in [(8430, base.NAME), (8431, fitted.NAME)]:
        ref = dict(
            path=str(root / "results" / (name + ".json")),
            sha256="sha256:" + history_reader.PINS[number],
        )
        read(ref)
        request, output = raw / f"request-{number}.json", raw / f"summary-{number}.json"
        atomic_json(request, dict(number=number, primary=ref, authority=authority))
        receipt = child(
            f"original_producer_{number}",
            [
                sys.executable,
                "-u",
                str(ROOT / methods.CLI),
                "--history-request",
                str(request),
                "--history-output",
                str(output),
            ],
            raw / "logs",
            deadline=120,
            scope="upstream",
        )
        family: Json = (
            json.loads(output.read_bytes())
            if receipt["passed"]
            else dict(
                experiment_id=number,
                qualified=False,
                gate_check_summary=[
                    failure(Path(ref["path"]), "original_producer_child", True, False)
                ],
            )
        )
        families.append(family)
        refs.append(ref)
        receipts.append(receipt)
        failures.extend(family["gate_check_summary"])
    return dict(
        families=families,
        references=refs,
        authority=authority,
        receipts=receipts,
        failures=failures,
    )


def measure(root: Path, raw: Path) -> Json:
    """Required CPU operands admit controls; unrelated history and optional devices do not gate them."""
    start = measurement_clock()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions", 0, 4500)
    pre = dict(
        resources(raw),
        free_disk_bytes=shutil.disk_usage(raw).free,
        tool_paths={
            name: shutil.which(str(ROOT / ".venv/bin" / name))
            for name in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
        measurement_cap_s=3600,
        validation_reserve_s=900,
        task_cap_s=4800,
        model_loads=0,
        generations=0,
        optional_cuda_and_boards_gate_cpu=False,
    )
    authority = methods.authority(root, raw / "authority")
    failures = deepcopy(authority["gate_check_summary"])
    if (
        not pre["private_disk_backed_scratch"]
        or pre["available_memory_bytes"] < pre["minimum_memory_bytes"]
        or pre["free_disk_bytes"] < 104857600
        or not all(pre["tool_paths"].values())
    ):
        failures.append(failure(raw, "private_disk_memory_tools", True, pre))
    protocol = methods.protocol()
    roles = protocol["partition"]["roles"]
    blank_rows = [
        dict(
            unit_id=f"{q}:unbound-sibling-{slot}:{arm}",
            answer_id=None,
            family=q,
            question_hash=q,
            role=role,
            arm=arm,
            seed=None,
            x=None,
            status="censored",
            censored=True,
            excluded=False,
            absolute_metric=0,
            raw_numerator=0,
            raw_denominator=1,
            missing_reason="required_original_operands_unavailable",
        )
        for role, questions in roles.items()
        for q in questions
        for slot in range(3)
        for arm in ARMS
    ]
    panel = dict(
        rows=blank_rows,
        support_passed=False,
        row_complete=False,
        question_roles=roles,
        predictor_access_log=[],
        control_parameter_counts=dict(zip(ARMS, [0, 2, 13, 109, 109], strict=True)),
        class_support={},
        source_join_rows=[],
        stratum_roster=[],
        partition_hashes={},
        question_exclusions=[],
        exposure_ledger=[],
        donor_mappings={},
        operators={},
    )
    history: Json = dict(families=[], references=[], authority=[], receipts=[], failures=[])
    upstream: Json = {}
    log: list[Json] = []
    source_refs = [dict(path=str(ROOT / methods.PROTOCOL), sha256=methods.PROTOCOL_PIN)]
    owned_failure = False
    try:
        history = historical(root, raw / "history")
        failures.extend(history["failures"])
        source_refs.extend([*history["references"], *history["authority"]])
        upstream = dict(read(history["references"][0]))
        imported_head = predictor_read(history["references"][1], upstream["evaluator_custody"], log)
        feature_ref = dict(
            path=upstream["feature_panel_path"], sha256=upstream["feature_panel_sha256"]
        )
        features = predictor_read(feature_ref, upstream["evaluator_custody"], log)
        head_ref = dict(
            path=imported_head["frozen_heads_path"], sha256=imported_head["frozen_heads_sha256"]
        )
        heads = predictor_read(head_ref, upstream["evaluator_custody"], log)
        source_refs.extend([feature_ref, head_ref, upstream["trainer_view"]])
        if features["rows"] != upstream["rows"] or not all(
            f["qualified"] for f in history["families"]
        ):
            raise ValueError("original_panel_or_heads_not_qualified")
    except (OSError, ValueError, KeyError) as error:
        failures.append(
            failure(
                root / "results" / (base.NAME + ".json"),
                "required_original_operands",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
            )
        )
    else:
        if not failures:
            try:
                panel = freeze(upstream, heads, protocol, log)
            except (ValueError, KeyError, TypeError) as error:
                owned_failure = True
                failures.append(failure(raw, "required_owned_controls", True, str(error)))
    end = measurement_clock()
    if (end - start) / 1e9 > 3600:
        owned_failure = True
        failures.append(failure(raw, "measurement_cap_s", 3600, (end - start) / 1e9))
    primitive = dict(
        feature_panel=dict(rows=panel["rows"], operators=panel["operators"]),
        trainer_view=dict(reference=upstream.get("trainer_view"), label_reads=0),
        evaluator_custody=dict(reference=upstream.get("evaluator_custody"), label_reads=0),
    )
    for name, value in primitive.items():
        atomic_json(raw / (name + ".json"), value)
        (raw / (name + ".json")).chmod(0o600)
    work = dict(
        authority=authority,
        methods_ready=authority["activated"],
        panel=panel,
        failures=failures,
        owned_failure=owned_failure,
        history=history,
        upstream_refs=history["references"],
        source_refs=source_refs,
        code_refs=[
            reference(ROOT / name)
            for name in [
                *OWNED,
                "python/carnot/reporting/v726_feature_panel.py",
                "python/carnot/verify/v726_criterion_energy_fit.py",
                "python/carnot/verify/sparse_energy_7996.py",
                "python/carnot/reporting/v727_historical_inputs.py",
            ]
        ],
        primitive_files=[reference(raw / (name + ".json")) for name in primitive],
        preconditions=pre,
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[dict(name="authenticate_and_freeze", duration_s=(end - start) / 1e9)],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """The shipped schema separates qualification from benefits; this adapter binds all control slots."""
    value = dict(base.build(work, receipts, raw, output))
    qualified = value["required_checks_passed"] and not work["owned_failure"]
    ready = (
        qualified
        and not work["failures"]
        and work["panel"]["row_complete"]
        and work["panel"]["support_passed"]
        and work["authority"]["activated"]
        and work["methods_ready"]
        and [f["experiment_id"] for f in work["history"]["families"]] == [8430, 8431]
        and all(f["qualified"] for f in work["history"]["families"])
    )
    verdict = "disqualified" if not qualified else "circular_positive" if ready else "blocked"
    value.update(
        experiment_id=8458,
        task_id=TASK,
        milestone="2026.10.728",
        run_date="20261012",
        honest_verdict=f"complete_{verdict}_v728_source_control_panel",
        verdict_class=verdict,
        source_controls_ready_score=int(ready),
        feature_panel_ready_score=int(ready),
        intended_count=4500,
        independent_count=300 if ready else 0,
        required_checks_passed=qualified,
        flagged_adversarial=not qualified,
        sample_size_budget=dict(
            intended_answers=900,
            intended_control_rows=4500,
            question_clusters=300,
            independent_generalization_questions=0,
            siblings_and_arms_are_dependent=True,
        ),
        feature_manifest_reference=reference(raw / "feature_panel.json"),
        methods_protocol_reference=reference(ROOT / methods.PROTOCOL),
        question_roles=work["panel"]["question_roles"],
        predictor_access_log=work["panel"]["predictor_access_log"],
        control_parameter_counts=work["panel"]["control_parameter_counts"],
        operator_manifest=work["panel"]["operators"],
        donor_mappings=work["panel"]["donor_mappings"],
        historical_qualification=work["history"]["families"],
        historical_validation_receipts=work["history"]["receipts"],
        historical_authority_limits=dict(
            V727="six blocked primaries; eight skipped; methods_ready_score=0; no source/objective/journal science",
            V726="qualified retrospective features/heads/checkpoints; null admission audit; kernel0.093%; persistence96.44%",
            V717_H1=-0.00390625,
            V717_H2=0,
            old_exposure="public labels exposed; older overlap unknown",
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation") for n in [8459, 8460]
        ],
        fitting_performed=False,
        current_llm_invocations=0,
        methodology_note="Frozen retrospective q-only and nine-input controls with original sealed V726 authority, whole-question source permutations, no fitting and no evaluator target reads. Readiness is not scientific benefit.",
    )
    value["acceptance_gates"].update(
        owned_validation=qualified, source_controls=bool(ready), scientific_generalization=False
    )
    if qualified and not ready and not value["gate_check_summary"]:
        value["gate_check_summary"].append(
            failure(
                raw,
                "authority_support_and_original_producers",
                True,
                dict(
                    authority=work["authority"]["activated"],
                    support=work["panel"]["support_passed"],
                    families=work["history"]["families"],
                ),
            )
        )
    for ref in value["cited_upstream_artifacts"]:
        ref["imported_fields"] = (
            [
                "rows",
                "class_support",
                "source_join_rows",
                "stratum_roster",
                "partition_hashes",
                "trainer_view",
                "evaluator_custody",
                "feature_panel_ready_score",
            ]
            if ref["sha256"] == "sha256:" + history_reader.PINS[8430]
            else ["frozen_heads_path", "frozen_heads_sha256", "head_ready_score"]
            if ref["sha256"] == "sha256:" + history_reader.PINS[8431]
            else ["partition", "source_controls", "scalar_fit"]
            if ref["sha256"] == methods.PROTOCOL_PIN
            else ["sealed_original_operand_bytes"]
        )
    value["field_principles"].update(
        {
            k: "Bound to this invocation and independently sealed original operands; control qualification cannot undo public exposure."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        source_controls_ready_score="All4500 control slots and inherited question/class support qualify after required validation; no scientific gain is measured.",
        control_parameter_counts="Raw0, logistic2, scalar spline13, full109, permuted109; only full/permuted match parameter count.",
        predictor_access_log="Prediction readers open sealed public vectors and numerical heads; evaluator custody is denied before opening bytes.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Independent producer seals and reconstruction reject rehashed control or authority changes."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("operand_drift")
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    raise ValueError("receipt_drift")
        raw = Path(value["primitive_reference"]["path"]).parent
        if value != build(
            work, value["validation_receipts"], raw, Path(value["publication_output"])
        ):
            raise ValueError("aggregate_reduction_drift")
        with TemporaryDirectory(prefix="exp8458-cold-", dir="/var/tmp") as directory:
            root = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                ref = work["authority"]["authority_snapshots"][role]
                if ref.get("snapshot_path"):
                    source = Path(ref["snapshot_path"])
                    if sha256_file(source) != ref["sha256"]:
                        raise ValueError("authority_snapshot_drift")
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            actual = methods.authority(root, root / "reduction")
            for key in ["activated", "tasks", "canonical_tasks_sha256"]:
                if actual[key] != work["authority"][key]:
                    raise ValueError("full_authority_drift")
            if work["upstream_refs"]:
                for ref, number in zip(work["upstream_refs"], [8430, 8431], strict=True):
                    if ref["sha256"] != "sha256:" + history_reader.PINS[number]:
                        raise ValueError("original_producer_pin")
                families = [
                    history_reader.qualify(
                        number, ref, work["history"]["authority"], root / str(number)
                    )
                    for number, ref in zip([8430, 8431], work["upstream_refs"], strict=True)
                ]
                if families != work["history"]["families"]:
                    raise ValueError("historical_qualification_drift")
            if work["panel"]["row_complete"]:
                upstream, imported_head = (read(ref) for ref in work["upstream_refs"])
                heads = read(
                    dict(
                        path=imported_head["frozen_heads_path"],
                        sha256=imported_head["frozen_heads_sha256"],
                    )
                )
                expected = freeze(
                    upstream, heads, methods.protocol(), work["panel"]["predictor_access_log"]
                )
                if expected != work["panel"]:
                    raise ValueError("primitive_control_drift")
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
