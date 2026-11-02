"""REQ-REPORT-8432: seal frozen predictions before this evaluator opens targets.

Independent producer hashes bind replay to released operands. The evaluator
records prior exposure explicitly; a new seal cannot make public labels private.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from carnot.reporting import v726_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import v726_criterion_energy_fit as fit
from carnot.verify import v726_decision_metrics as n

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8432_v726_criterion_decision_audit", "exp8432-criterion-decision-audit"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v726_criterion_decision_8432.py"
OWNED = [
    "python/carnot/verify/v726_decision_metrics.py",
    "python/carnot/reporting/v726_criterion_decision_audit.py",
    "python/carnot/reporting/v726_criterion_decision_runner.py",
    CLI,
]
UPSTREAM = "results/experiment_8431_v726_criterion_energy_fit.json"
UPSTREAM_PIN = "sha256:38db38433d5e2803a7d4b008c88fba5a1a91138abceb42e52f3dadf5b2f10b7f"
MODEL_SPECS: list[Json] = []
reference, read, checksum = methods.reference, methods.read, methods.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real phase counts keep numeric reductions and child waits visible."""
    print(f"[exp8432] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(
    path: Path,
    field: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str = TASK,
    digest: Any = None,
) -> Json:
    """Missing stays null so a gate never substitutes an observed zero."""
    return dict(
        check=field,
        upstream=upstream,
        path=str(path),
        sha256=digest,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def operands(root: Path) -> tuple[Json, Json, Json, Json]:
    """Open only predictor operands; evaluator targets are kept in a separate file."""
    upstream = dict(read(dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN)))
    panel = dict(read(dict(path=str(root / fit.UPSTREAM), sha256=fit.UPSTREAM_PIN)))
    for value, field in [(upstream, "head_ready_score"), (panel, "feature_panel_ready_score")]:
        if (
            value.get(field) != 1
            or not value.get("required_checks_passed")
            or value.get("flagged_adversarial")
        ):
            raise ValueError("upstream_not_qualified")
        primary = root / (UPSTREAM if field == "head_ready_score" else fit.UPSTREAM)
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        bound = read_bound_sidecar(primary, Path(terminal["publication"]["sidecar_path"]))
        if not bound["report"]["passed"]:
            raise ValueError("upstream_publication_failed")
    frozen = dict(
        read(dict(path=upstream["frozen_heads_path"], sha256=upstream["frozen_heads_sha256"]))
    )
    features = dict(
        read(dict(path=panel["feature_panel_path"], sha256=panel["feature_panel_sha256"]))
    )
    return upstream, panel, frozen, features


def predict_rows(frozen: Json, features: list[Json], strata: list[Json]) -> list[Json]:
    """Only public numeric vectors enter the frozen predictor, never human targets."""
    metadata = {r["unit_id"]: r for r in strata}
    test = [r for r in features if r["role"] == "test"]
    valid = [r for r in test if r["x"] is not None]
    original = {r["unit_id"]: r for r in valid}
    donors = {
        d["recipient_unit_id"]: d["donor_unit_id"]
        for d in frozen.get("donor_mappings", {}).get("test", [])
    }
    rows = []
    for arm in n.ARMS:
        for seed in n.SEEDS:
            head = next(
                (h for h in frozen.get("heads", {}).get(arm, []) if h["seed"] == seed), None
            )
            eligible, vectors = [], []
            for r in valid:
                donor = (
                    original.get(donors.get(r["unit_id"], ""))
                    if arm == "source_permuted_spline9"
                    else r
                )
                if donor:
                    eligible.append(r["unit_id"])
                    vectors.append([r["x"][0], *donor["x"][1:]])
            probabilities = (
                dict(
                    zip(
                        eligible,
                        fit.predict(head, np.asarray(vectors, dtype=float)).tolist(),
                        strict=True,
                    )
                )
                if head and vectors
                else {}
            )
            for r in test:
                p = probabilities.get(r["unit_id"])
                rows.append(
                    dict(
                        unit_id=r["unit_id"],
                        question_hash=r["question_hash"],
                        dataset=metadata[r["unit_id"]]["dataset"],
                        generator=metadata[r["unit_id"]]["generator"],
                        arm=arm,
                        seed=seed,
                        probability=p,
                        missing_reason=None
                        if p is not None
                        else "missing_head_feature_or_permutation_donor",
                    )
                )
            progress(f"frozen_prediction_{arm}_{seed}", len(rows), len(test) * 25 - len(rows))
    return rows


def roster() -> tuple[list[Json], list[Json]]:
    """Pinned public IDs preserve intended slots even when required fitting is absent."""
    panel = dict(read(dict(path=str(ROOT / fit.UPSTREAM), sha256=fit.UPSTREAM_PIN)))
    features = read(dict(path=panel["feature_panel_path"], sha256=panel["feature_panel_sha256"]))
    return [dict(r, x=None) for r in features["rows"] if r["role"] == "test"], panel[
        "stratum_roster"
    ]


def measure(root: Path, raw: Path) -> Json:
    """Authenticate only actual CPU operands, then seal predictions before targets."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions", 0, 7500)
    resources = dict(
        methods.legacy.resources(raw),
        free_disk_bytes=shutil.disk_usage(raw).free,
        tool_paths={
            name: shutil.which(str(ROOT / ".venv/bin" / name))
            for name in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
        task_cap_s=4800,
        measurement_cap_s=3900,
        validation_reserve_s=900,
        model_loads=0,
        generations=0,
    )
    authority = methods.authority(root, raw / "authority")
    failures = list(authority["gate_check_summary"])
    if (
        not resources["private_disk_backed_scratch"]
        or resources["available_memory_bytes"] < resources["minimum_memory_bytes"]
        or resources["free_disk_bytes"] < 104857600
        or not all(resources["tool_paths"].values())
    ):
        failures.append(gate(raw, "private_disk_memory_tools_deadline", True, resources))
    upstream: Json = {}
    panel: Json = {}
    frozen: Json = {}
    features: Json = {}
    sources: list[Json] = []
    try:
        upstream, panel, frozen, features = operands(root)
        methods.protocol()
        sources = [
            dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN),
            dict(path=str(root / fit.UPSTREAM), sha256=fit.UPSTREAM_PIN),
            dict(path=upstream["frozen_heads_path"], sha256=upstream["frozen_heads_sha256"]),
            dict(path=panel["feature_panel_path"], sha256=panel["feature_panel_sha256"]),
            dict(path=str(ROOT / methods.PROTOCOL), sha256=methods.PROTOCOL_PIN),
        ]
    except (OSError, ValueError, KeyError) as error:
        failures.append(
            gate(
                root / UPSTREAM,
                "head_ready_score_and_authenticated_operands",
                1,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream="exp8431-criterion-energy-fit",
                digest=UPSTREAM_PIN,
            )
        )
    prediction_rows, targets, result = [], [], None
    clock: Json = dict(predictions_sealed_ns=None, targets_opened_ns=None)
    owned_failure = False
    if failures:
        intended, strata = roster()
        prediction_rows = predict_rows({}, intended, strata)
    else:
        progress("frozen_prediction_begin", 0, 7500)
        try:
            prediction_rows = predict_rows(frozen, features["rows"], panel["stratum_roster"])
            if (
                len(prediction_rows) != 7500
                or len({r["question_hash"] for r in prediction_rows}) != 100
            ):
                raise ValueError("intended_roster_drift")
        except (ValueError, KeyError, FloatingPointError) as error:
            owned_failure = True
            failures.append(gate(raw, "frozen_numeric_prediction", True, str(error)))
            intended, strata = roster()
            prediction_rows = predict_rows({}, intended, strata)
    atomic_json(raw / "sealed_predictions.json", dict(rows=prediction_rows))
    clock["predictions_sealed_ns"] = time.time_ns()
    prediction_reference = reference(raw / "sealed_predictions.json")
    progress("predictions_sealed", 7500, 0)
    if not failures:
        progress("evaluator_targets_open_begin", 0, 300)
        try:
            clock["targets_opened_ns"] = time.time_ns()
            targets = read(panel["evaluator_custody"])["rows"]
            sources.append(panel["evaluator_custody"])
        except (OSError, ValueError, KeyError) as error:
            failures.append(
                gate(
                    Path(panel["evaluator_custody"]["path"]),
                    "A1_evaluator_custody",
                    "authenticated_targets",
                    None if isinstance(error, FileNotFoundError) else str(error),
                    upstream="exp8430-criterion-feature-panel",
                    digest=panel["evaluator_custody"]["sha256"],
                )
            )
        progress("evaluator_targets_open_end", len(targets), 300 - len(targets))
    atomic_json(raw / "evaluator_targets.json", dict(rows=targets))
    if not failures:
        progress("decision_reduction_begin", 0, 7500)
        try:
            result = n.evaluate(prediction_rows, targets)
        except (ValueError, KeyError, FloatingPointError) as error:
            owned_failure = True
            failures.append(gate(raw, "decision_reduction", True, str(error)))
        progress("decision_reduction_end", 7500 if result else 0, 0 if result else 7500)
    work = dict(
        authority=authority,
        failures=failures,
        owned_failure=owned_failure,
        result=result,
        preconditions=resources,
        source_refs=sources,
        upstream_reference=dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN),
        code_refs=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                "python/carnot/verify/v726_criterion_energy_fit.py",
                "python/carnot/verify/sparse_energy_7996.py",
                "python/carnot/reporting/primary_publication.py",
            ]
        ],
        primitive_files=[prediction_reference, reference(raw / "evaluator_targets.json")],
        prediction_reference=prediction_reference,
        clock=clock,
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=[
            dict(name="authenticate_seal_evaluate", duration_s=(time.monotonic_ns() - start) / 1e9)
        ],
    )
    for path in raw.glob("*.json"):
        path.chmod(0o600)
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """A measured null remains science even if publication qualification fails."""
    qualified = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and not work.get("adversarial_findings")
        and not work["owned_failure"]
    )
    measured = bool(work["result"])
    ready = qualified and measured and not work["failures"] and work["authority"]["activated"]
    result = work["result"] or {}
    benefit = ready and bool(result.get("exploratory_benefit_score"))
    scientific = (
        "circular_positive"
        if result.get("exploratory_benefit_score")
        else "null"
        if measured
        else "unmeasured"
    )
    verdict = (
        "disqualified"
        if not qualified
        else "circular_positive"
        if benefit
        else "null"
        if ready
        else "blocked"
    )
    rows = result.get("rows") or [
        dict(
            r,
            family=r["question_hash"],
            status="censored",
            censored=True,
            excluded=False,
            absolute_metric=0,
            raw_numerator=0,
            raw_denominator=1,
            metric=None,
            missing_reason=r["missing_reason"] or "owned_measurement_failed",
        )
        for r in read(work["prediction_reference"])["rows"]
    ]
    checks = [
        *work["failures"],
        *[
            gate(
                Path(r.get("stdout_path", raw)),
                "required_owned_check:" + r["name"],
                True,
                False,
                digest=r.get("stdout_sha256"),
            )
            for r in receipts
            if not r["passed"]
        ],
    ]
    if work.get("adversarial_findings"):
        checks.append(gate(raw, "publication_findings_resolved", [], work["adversarial_findings"]))
    value = dict(
        experiment_id=8432,
        task_id=TASK,
        milestone="2026.10.726",
        run_date="20261011",
        honest_verdict=f"complete_{verdict}_v726_criterion_decision_audit",
        verdict_class=verdict,
        gate_check_summary=checks,
        inference_substrate="aggregation_from_upstream_artifacts",
        numeric_learning_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(methods.legacy.ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=[
            dict(
                producer="exp8430-criterion-feature-panel",
                models=["GPT-5-mini", "gemma-2-9b", "llama3-8b", "mistral-7b"],
                scope="released cached judgments and answers; no current Qwen inference",
            )
        ],
        rows=rows,
        intended_count=7500,
        completed_count=sum(not r["censored"] for r in rows),
        failed_count=0,
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=result.get("bootstrap", {}).get("independent_questions", 0),
        sample_size_budget=dict(
            intended_answers=300,
            intended_questions=100,
            intended_arm_seed_slots=7500,
            intended_arms=5,
            seeds=list(n.SEEDS),
            siblings_per_question=3,
            independent_generalization_questions=0,
            seeds_are_not_independent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="retrospective public human labels; old overlap unknown; setup inspected one evaluator record before the audit invocation; frozen predictor receives public vectors only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(qualified),
        flagged_adversarial=not qualified,
        acceptance_gates=dict(
            owned_validation=bool(qualified),
            decision_audit_measurement=measured,
            decision_audit_publication=bool(ready),
            exploratory_lower_bound_positive=bool(
                result.get("brier_gain_ci95") and result["brier_gain_ci95"][0] > 0
            ),
            decision_loss_nonincrease=bool(
                result.get("decision_loss_delta") is not None and result["decision_loss_delta"] <= 0
            ),
            independent_generalization=False,
        ),
        validation_receipts=receipts,
        adversarial_findings=work.get("adversarial_findings", []),
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=n.RANDOM_SEED,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=[
            *work["code_refs"],
            dict(config=n.CONFIG, sha256=methods.legacy.canonical_hash(n.CONFIG)),
        ],
        raw_shard_hashes=[
            reference(raw / "measurement.json"),
            *work["primitive_files"],
            *work.get("coverage_shards", []),
        ],
        cited_upstream_artifacts=[
            dict(
                ref,
                imported_fields=[
                    "frozen_heads",
                    "feature_panel",
                    "stratum_roster",
                    "evaluator_custody",
                    "methods_protocol",
                ],
            )
            for ref in work["source_refs"]
        ],
        primitive_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        decision_audit_ready_score=int(ready),
        scientific_verdict_class=scientific,
        scientific_measurement_complete=measured,
        exploratory_benefit_score=int(benefit),
        sealed_predictions_sha256=work["prediction_reference"]["sha256"],
        prediction_reference=work["prediction_reference"],
        seal_clock=work["clock"],
        arm_roster=list(n.ARMS),
        methodology_note="Frozen exposed-label evaluation; fixed decisions; A1 primary and observed A2 sensitivity. Paired losses average seeds and sibling answers within question. Conditional 10000 whole-question bootstrap draws. No threshold search, current model call or independent factuality claim.",
    )
    for key in [
        "brier_gain_vs_logistic",
        "brier_gain_ci95",
        "decision_loss_delta",
        "A1_metrics",
        "A2_sensitivity",
        "per_seed_metrics",
        "bootstrap",
        "sensitivity_bootstrap",
        "per_stratum_metrics",
        "confidence_calibration",
        "risk_coverage",
        "missing_worst_case_bounds",
        "negative_control_degeneracy",
    ]:
        value[key] = result.get(key)
    value["field_principles"] = {
        key: "Bound to this invocation and pinned retrospective operands; qualification and benefit remain separate."
        for key in value
    }
    value["field_principles"].update(
        independent_count="Complete paired whole questions only; seeds and siblings never multiply independence.",
        decision_audit_ready_score="One only after valid measurement and publication qualification, including valid nulls.",
        scientific_verdict_class="Measured null survives administrative disqualification; unmeasured remains distinct.",
        acceptance_gates="Fixed CI and loss conditions; required checks and non-inert control also govern benefit.",
        missing_worst_case_bounds="Unknown probabilities contribute [0,1] loss bounds in each observed target denominator.",
        seal_clock="This evaluator opens targets after its durable prediction seal; public prior exposure stays explicit.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Independent frozen heads defeat repaired local hashes and summary tampering."""
    from tempfile import TemporaryDirectory

    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        for ref in [
            *work["source_refs"],
            *work["code_refs"],
            *work["primitive_files"],
            *work.get("coverage_shards", []),
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("operand_drift")
        with TemporaryDirectory(prefix="exp8432-cold-", dir="/var/tmp") as directory:
            cold = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                snap = work["authority"]["authority_snapshots"][role]
                if "snapshot_path" in snap:
                    source = Path(snap["snapshot_path"])
                    if sha256_file(source) != snap["sha256"]:
                        raise ValueError("authority_snapshot_drift")
                    target = cold / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            if (
                methods.authority(cold, cold / "cold")["activated"]
                != work["authority"]["activated"]
            ):
                raise ValueError("authority_drift")
        raw = Path(value["primitive_reference"]["path"]).parent
        if work["result"]:
            if work["upstream_reference"]["sha256"] != UPSTREAM_PIN:
                raise ValueError("independent_upstream_pin")
            _, panel, frozen, features = operands(
                Path(work["upstream_reference"]["path"]).parents[1]
            )
            predictions = predict_rows(frozen, features["rows"], panel["stratum_roster"])
            if dict(rows=predictions) != read(work["prediction_reference"]):
                raise ValueError("sealed_prediction_drift")
            targets = read(panel["evaluator_custody"])["rows"]
            if (
                dict(rows=targets) != read(reference(raw / "evaluator_targets.json"))
                or n.evaluate(predictions, targets) != work["result"]
            ):
                raise ValueError("target_or_reduction_drift")
            if not work["clock"]["predictions_sealed_ns"] < work["clock"]["targets_opened_ns"]:
                raise ValueError("seal_order")
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    raise ValueError("receipt_drift")
            saved = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            if any(receipt.get(key) != actual for key, actual in saved.items() if key != "passed"):
                raise ValueError("receipt_metadata_drift")
            expected_pass = bool(
                saved["passed"]
                and saved["actual_exit"] == saved["exit_code"] == saved["expected_exit"]
                and not saved["timed_out"]
                and saved["normal_exit"]
            )
            if saved["name"] == "adversarial":
                report = json.loads(Path(saved["stdout_path"]).read_bytes())
                expected_pass = expected_pass and not any(
                    row.get("flags") for row in report["reports"]
                )
            if receipt.get("passed") is not expected_pass:
                raise ValueError("receipt_status_drift")
        return bool(
            value
            == build(work, value["validation_receipts"], raw, Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
