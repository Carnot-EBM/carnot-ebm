"""REQ-REPORT-8972: retain exposed graph diagnostics without claiming fresh benefit."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, SEED = (
    "experiment_8972_v767_encoder_head_diagnosis",
    "exp8972-encoder-head-diagnosis",
    767072,
)
EXPERIMENT_ID = 8972
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/autoresearch/encoder_head_8972.py",
    "python/carnot/reporting/encoder_head_diagnosis_8972.py",
    "python/carnot/reporting/encoder_head_runner_8972.py",
    CLI,
]
TESTS = ["tests/python/test_encoder_head_diagnosis_8972.py"]
INPUTS = [
    (
        8958,
        "results/experiment_8958_v766_confidence_diagnosis.json",
        "ff279de4b7b8f2a8eae95d978288da6f588bf6c0d5635e83bf062a72b605f6b0",
        "diagnosis_ready_score",
    ),
    (
        8959,
        "results/experiment_8959_v766_relational_methods.json",
        "eaa01cdf08424373c0ea12b703a09099bf489cd015593111a5142c1c2f25c966",
        "relational_methods_ready_score",
    ),
    (
        8963,
        "results/experiment_8963_v766_corpus_admission.json",
        "586a8f5bff118cd9416f0f03cf9606bc35db0eb4068d946bbf17105329547d24",
        "corpus_ready_score",
    ),
    (
        8964,
        "results/experiment_8964_v766_relational_decisions.json",
        "f70d9becc3486f2dec4f8c0c9400cb6ebb49c1bd1fcd0ae67e36b56f74872408",
        "detector_rows_ready_score",
    ),
    (
        8965,
        "results/experiment_8965_v766_relational_controls.json",
        "188c29a7ee88ae6ca363a7e9137d4fe11c2308c1b3b9f037ce21d51259d1ad7d",
        "relational_controls_ready_score",
    ),
]
METHODS = f"results/raw/{NAME}/methods/method_sources.json"
METHOD_PIN = "sha256:65f81ab20e9ba9219d0566bd79fe9ef97acc33ce724691832828393dffcbf2d0"
ARMS = [
    f"{encoder}_encoder_{head}_head"
    for encoder in ("logistic", "energy")
    for head in ("linear", "gibbs")
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured phase counts so CPU work is visible to the parent."""
    print(f"[exp8972] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """A missing file has no digest; it must remain distinct from changed bytes."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def checksum(value: Json) -> str:
    """Hash every published field except the checksum itself."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def failure(
    path: Path, upstream: str, field: str, expected: Any, observed: Any, check: str
) -> Json:
    """Keep the actual file digest beside an exact field comparison."""
    return dict(
        reference(path),
        upstream=upstream,
        check=check,
        artifact_field=field,
        operator="==",
        expected=expected,
        observed=observed,
    )


def retain(ref: Json, raw: Path) -> Json:
    """Snapshot only authenticated original bytes; a coordinated rehash cannot change a pin."""
    path = Path(ref["path"])
    if reference(path)["sha256"] != ref["sha256"]:
        raise ValueError(f"operand_bytes:{path}")
    target = raw / "inputs" / ref["sha256"].split(":")[-1] / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != path.read_bytes():
        raise ValueError("snapshot_bytes")
    target.write_bytes(path.read_bytes())
    return dict(reference(target), source_path=str(path))


def resources(raw: Path) -> list[Json]:
    """CPU dependencies, free memory and durable scratch are the only resource gates."""
    rows = []
    for module in ("jax", "numpy", "sklearn", "coverage", "pytest", "ruff", "mypy"):
        if importlib.util.find_spec(module) is None:
            rows.append(
                failure(raw, "environment", module, "installed", "missing_dependency", "dependency")
            )
    try:
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=raw) as stream:
            stream.write(b"8972")
            stream.flush()
            os.fsync(stream.fileno())
        for field, observed, minimum in (
            ("disk_free_bytes", shutil.disk_usage(raw).free, 100_000_000),
            (
                "memory_available_bytes",
                os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
                500_000_000,
            ),
        ):
            if observed < minimum:
                rows.append(
                    dict(
                        failure(raw, "environment", field, minimum, observed, "resource"),
                        operator=">=",
                    )
                )
    except OSError as error:
        rows.append(failure(raw, "environment", "writable_disk", True, str(error), "resource"))
    return rows


def authenticate(root: Path, raw: Path) -> Json:
    """Read pinned historical producers; retired scheduler tasks are never runtime requirements."""
    result: Json = dict(
        failures=[], references=[], producers={}, states={}, views={}, method_source_rows=[]
    )
    for identity, name, pin, gate in INPUTS:
        path, task = root / name, f"exp{identity}"
        observed = reference(path)["sha256"]
        pin = pin if pin.startswith("sha256:") else "sha256:" + pin
        if observed != pin:
            result["failures"].append(
                failure(
                    path,
                    task,
                    "sha256",
                    pin,
                    "missing_file" if observed is None else "changed_bytes",
                    "producer_bytes",
                )
            )
            continue
        try:
            value = json.loads(path.read_text())
            for field, expected in ((gate, 1), ("required_checks_passed", True)):
                actual = value.get(field)
                if type(actual) is not type(expected) or actual != expected:
                    result["failures"].append(
                        failure(
                            path,
                            task,
                            field,
                            expected,
                            actual,
                            "missing_field" if field not in value else "observed_value",
                        )
                    )
                    break
            else:
                for ref in (
                    value["source_artifact_hashes"]
                    + value["code_config_hashes"]
                    + value["raw_shard_hashes"]
                ):
                    if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                        raise ValueError(f"retained_bytes:{ref['path']}")
                saved = retain(reference(path), raw)
                result["references"].append(
                    dict(
                        saved,
                        experiment_id=identity,
                        fields_imported=[
                            gate,
                            "required_checks_passed",
                            "frozen_development_evidence",
                        ],
                    )
                )
                result["producers"][str(identity)] = value
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            result["failures"].append(
                failure(
                    path,
                    task,
                    "producer_json",
                    "original complete object",
                    str(error),
                    "producer_structure",
                )
            )
    if result["failures"]:
        return result
    try:

        def read(ref: Json) -> Json:
            saved = retain(ref, raw)
            result["references"].append(saved)
            return dict(json.loads(Path(saved["path"]).read_text()))

        method, corpus = result["producers"]["8959"], result["producers"]["8963"]
        result["states"] = read(method["trained_state_reference"])["states"]
        for role in ("fit", "guard", "calibration", "test", "online"):
            view = read(corpus["inference_views"][role])
            if role not in ("fit", "guard"):
                labels = read(corpus["broker_manifest"][role]["label_reference"])["rows"]
                if [r["unit_id"] for r in labels] != view["unit_ids"]:
                    raise ValueError("label_roster")
                view["labels"] = [r["label"] for r in labels]
            result["views"][role] = view
        result["original_service"] = read(result["producers"]["8964"]["calibrated_state_reference"])
        result["method_source_rows"] = read(dict(path=str(root / METHODS), sha256=METHOD_PIN))[
            "rows"
        ]
        for row in result["method_source_rows"]:
            result["references"].append(retain(row, raw))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["failures"].append(
            failure(
                root,
                "historical_operands",
                "frozen_views",
                "authenticated complete operands",
                str(error),
                "operand_structure",
            )
        )
    return result


def seal(path: Path, value: Json, replay: bool) -> None:
    """Replay compares immutable primitive bytes rather than overwriting changed evidence."""
    if replay:
        if json.loads(path.read_text()) != value:
            raise ValueError(f"primitive_drift:{path.name}")
    else:
        atomic_json(path, value)


def empty_result() -> Json:
    """Retain all intended graph/arm slots when an external precondition blocks work."""
    rows = [
        dict(
            unit_id=f"unavailable-{i:04d}",
            cluster_id=f"unavailable-{i:04d}",
            arm=arm,
            metric="brier",
            value=None,
            numerator=None,
            denominator=1,
            status="censored",
            censor_reason="external_precondition",
            condition="exposed_development",
        )
        for i in range(1344)
        for arm in ARMS
    ]
    return dict(
        rows=rows,
        diagnostic_prediction_rows=[],
        diagnostic_metric_rows=[],
        intended_count=1344,
        completed_count=0,
        censored_count=1344,
        failed_count=0,
        excluded_count=0,
        independent_count=0,
        numerical_training_calls=0,
        original_metric_rows=[],
        original_bound_rows=[],
        encoder_hashes=[],
        feature_hashes=[],
        head_state_references=[],
        gradient_rows=[],
        candidate_reference=None,
    )


def evaluate(evidence: Json, raw: Path, replay: bool = False) -> Json:
    """Fit only original fit labels, then expose guard and historical metrics descriptively."""
    from carnot.autoresearch import encoder_head_8972 as n
    from carnot.autoresearch import relational_graph_encoder as g
    from carnot.reporting import relational_decisions_8964 as old
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    views, states = evidence["views"], evidence["states"]
    roster = [u for view in views.values() for u in view["unit_ids"]]
    if (
        len(roster) != 1344
        or len(set(roster)) != 1344
        or len(views["fit"]["rows"]) != 256
        or len(views["guard"]["rows"]) != 64
        or views["fit"]["unit_ids"] != evidence["producers"]["8959"]["development_fit_ids"]
        or views["guard"]["unit_ids"] != evidence["producers"]["8959"]["development_guard_ids"]
    ):
        raise ValueError("development_roles")
    for view in views.values():
        n.b.labels(view["labels"], len(view["rows"]))
        g.pack(view["rows"])
    for encoder in ("logistic", "energy"):
        if [s["seed"] for s in states[f"relational_{encoder}"]] != [71, 72, 73]:
            raise ValueError("seed_roster")
    candidate = dict(
        selection="protocol_fixed_logistic_encoder_all_three_seeds_before_diagnostic_scores",
        encoder_states=[
            dict(seed=s["seed"], encoder=s["params"]["encoder"], scaling=s["scaling"])
            for s in states["relational_logistic"]
        ],
        exposure_scope="exposed_development_only",
    )
    seal(raw / "candidate.json", candidate, replay)
    progress("before_original_metrics_and_five_bounds", 0, 128)
    scored = old.score(evidence["original_service"], views["test"]["rows"])
    original = old.reduce(
        scored["probabilities"], views["test"]["labels"], views["test"]["unit_ids"]
    )
    producer = evidence["producers"]["8964"]
    if (
        original["descriptive_rows"] != producer["descriptive_rows"]
        or original["paired_intervals"] != producer["paired_intervals"]
    ):
        raise ValueError("original_metric_reproduction")
    progress("after_original_metrics_and_five_bounds", 128, 0)
    rows: list[Json] = []
    gradients: list[Json] = []
    encoders: list[Json] = []
    feature_hashes: list[Json] = []
    head_refs: list[Json] = []
    summaries: list[Json] = []
    probabilities: Json = {a: {role: [] for role in views} for a in ARMS}
    for encoder in ("logistic", "energy"):
        for state in states[f"relational_{encoder}"]:
            before = canonical_hash(state["params"]["encoder"])
            encoders.append(dict(encoder=encoder, seed=state["seed"], sha256=before))
            for row in n.original_gradients(state, views["fit"]["rows"], views["fit"]["labels"]):
                gradients.append(
                    dict(
                        row,
                        encoder=encoder,
                        head=encoder,
                        seed=state["seed"],
                        condition="original_joint_training",
                    )
                )
            for role, view in views.items():
                x = n.features(state, view["rows"])
                path = raw / "features" / f"{encoder}-{state['seed']}-{role}.json"
                seal(path, dict(unit_ids=view["unit_ids"], features=x.tolist()), replay)
                feature_hashes.append(
                    dict(
                        reference(path),
                        encoder=encoder,
                        seed=state["seed"],
                        role=role,
                        feature_count=27,
                    )
                )
            for head, energy in (("linear", False), ("gibbs", True)):
                arm = f"{encoder}_encoder_{head}_head"
                progress(
                    f"before_training_{arm}_seed{state['seed']}",
                    len(head_refs),
                    12 - len(head_refs),
                )
                trained, grad_rows = n.fit_head(
                    state, views["fit"]["rows"], views["fit"]["labels"], energy
                )
                if canonical_hash(trained["params"]["encoder"]) != before:
                    raise ValueError("frozen_encoder_changed")
                gradients.extend(
                    dict(
                        r,
                        encoder=encoder,
                        head=head,
                        seed=state["seed"],
                        condition="frozen_head_training",
                    )
                    for r in grad_rows
                )
                path = raw / "heads" / f"{arm}-{state['seed']}.json"
                seal(path, trained, replay)
                head_refs.append(
                    dict(reference(path), arm=arm, seed=state["seed"], encoder_sha256=before)
                )
                for role, view in views.items():
                    p = g.predict(trained, view["rows"])
                    probabilities[arm][role].append(p.tolist())
                progress(f"after_training_{arm}", len(head_refs), 12 - len(head_refs))
    for arm in ARMS:
        for role, view in views.items():
            y = np.asarray(view["labels"])
            p = np.mean(probabilities[arm][role], axis=0)
            clipped = np.clip(p, 1e-15, 1 - 1e-15)
            loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
            for i, unit in enumerate(view["unit_ids"]):
                value = float((p[i] - y[i]) ** 2)
                rows.append(
                    dict(
                        unit_id=unit,
                        cluster_id=unit,
                        arm=arm,
                        role=role,
                        condition="exposed_development",
                        metric="brier",
                        value=value,
                        numerator=value,
                        denominator=1,
                        status="completed",
                        censor_reason=None,
                        p_error=float(p[i]),
                        error=int(y[i]),
                        log_loss=float(loss[i]),
                        seed_probabilities=[s[i] for s in probabilities[arm][role]],
                    )
                )
            for metric, value in dict(
                brier=float(np.mean((p - y) ** 2)),
                log_loss=float(loss.mean()),
                auroc=float(roc_auc_score(y, p)),
                auprc=float(average_precision_score(y, p)),
                prediction_variance=float(np.var(p)),
            ).items():
                summaries.append(
                    dict(
                        arm=arm,
                        role=role,
                        metric=metric,
                        value=value,
                        scope="exposed_development_only",
                    )
                )
    seal(raw / "predictions.json", dict(rows=rows), replay)
    result = dict(
        rows=rows,
        diagnostic_prediction_rows=rows,
        diagnostic_metric_rows=summaries,
        original_metric_rows=original["descriptive_rows"],
        original_bound_rows=[
            dict(metric=k, **v) for k, v in original["paired_intervals"]["contrasts"].items()
        ],
        original_paired_intervals=original["paired_intervals"],
        encoder_hashes=encoders,
        feature_hashes=feature_hashes,
        head_state_references=head_refs,
        gradient_rows=gradients,
        candidate_reference=reference(raw / "candidate.json"),
        intended_count=1344,
        completed_count=1344,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=1344,
        numerical_training_calls=12,
        numerical_full_batch_updates=2400,
        gradient_measurement_scope="frozen derivatives are potential only; original joint rows measure original0/1/200 states",
        role_counts={role: len(v["unit_ids"]) for role, v in views.items()},
        controls=dict(
            frozen_encoders_unchanged=True,
            original_metrics_reproduced=True,
            fit_guard_roles_preserved=True,
            inference_only_schema=True,
            candidate_fixed_in_advance=True,
        ),
    )
    seal(raw / "reduction.json", result, replay)
    return result


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Readiness qualifies a complete diagnostic even when every energy hypothesis is null."""
    result, evidence = work["result"], work["evidence"]
    owned = False
    if validation["receipts"]:
        from carnot.reporting.matched_corpus_a_8949 import qualified

        owned = qualified(validation, OWNED) and not validation.get("terminal_rejected", False)
    ready = int(
        owned
        and not fixture
        and not evidence["failures"]
        and result["completed_count"] == 1344
        and bool(result.get("controls"))
        and all(result.get("controls", {}).values())
    )
    failed_owned = (
        not fixture and not owned and (bool(validation["receipts"]) or not evidence["failures"])
    )
    cls = "disqualified" if failed_owned else "blocked" if evidence["failures"] else "null"
    value = dict(
        result,
        experiment_id=8972,
        task_id=TASK,
        milestone="2026.11.767",
        run_date="20261110",
        random_seed=SEED,
        honest_verdict=f"complete_{cls}_exposed_encoder_head_diagnosis",
        verdict_class=cls,
        status="blocked" if cls == "blocked" else "error" if cls == "disqualified" else "success",
        diagnosis_ready_score=ready,
        independent_generalization_score=0,
        verifier_is_oracle=False,
        exposure_scope="all_V765_V766_graph_labels_exposed_development_only",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        execution_venue="host",
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        gate_check_summary=evidence["failures"],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        source_artifact_hashes=evidence["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(k),
                task_id=v["task_id"],
                honest_verdict=v["honest_verdict"],
                sha256=next(
                    r["sha256"] for r in evidence["references"] if r.get("experiment_id") == int(k)
                ),
                fields_imported=["frozen_development_evidence", "readiness"],
                provenance_scope="historical_only",
            )
            for k, v in evidence["producers"].items()
        ],
        method_source_rows=evidence["method_source_rows"],
        required_checks_passed=owned,
        flagged_adversarial=bool(validation.get("terminal_rejected", False)),
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        global_test_debt=work["global_test_debt"],
        fixture_only=fixture,
        fresh_independent_count=0,
        future_dependencies=[
            dict(
                task_id="exp8973-shared-feature-methods",
                status="future_dependency_not_required_input",
            )
        ],
        diagnosis_limits=dict(
            leakage="Strict input schema and original roles constrain leakage; all labels remain exposed.",
            optimization="Same frozen feature bytes isolate head training conditional on each historical encoder; no causal root cause is identified.",
            information="Encoder differences may reflect historical optimization or information. Exposed correlations do not separate these causes.",
            gradients="Zero Gibbs output causes the first encoder derivative to vanish. Step1 and200 measurements are required before any persistent-defect claim.",
            generalization="Cheap BFS truth and exposed graphs cannot close the no-cheap-oracle north star.",
            candidate="All three logistic encoders were fixed by protocol before diagnostic scores, with no gain-based selection.",
        ),
        methodology_note="Four frozen encoder/head combinations, three seeds, original256 fit and64 guard labels. Fresh heads use200 full batches at0.01 and original initialization. All1024 V766 answers remain exposed development evidence. Seed averaging creates no extra independent graphs. Original five failed bounds retain their interpretation. No current generator activity or energy gain is assumed.",
    )
    value["acceptance_gates"] = {
        name: dict(passed=passed, principle=principle)
        for name, passed, principle in (
            (
                "authenticated_inputs",
                not evidence["failures"],
                "Only pinned original bytes and exact declared readiness permit work.",
            ),
            (
                "complete_diagnosis",
                result["completed_count"] == 1344,
                "Retain all graph/arm rows and original fit/guard roles.",
            ),
            (
                "owned_checks",
                owned,
                "Measured coverage and private controls must qualify independently of science.",
            ),
            (
                "scientific_benefit",
                False,
                "Exposed development diagnosis cannot establish fresh benefit.",
            ),
        )
    }
    value["field_principles"] = {
        k: "Bind task identity, original bytes, actual work, exposed graph rows and separate diagnostic readiness to a terminal result."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        diagnosis_ready_score="Complete reproducible diagnosis and private controls qualify readiness even for a null hypothesis.",
        numerical_training_calls="Twelve current small-head fits; gradient probes and historical generator work are separate.",
        verifier_is_oracle="Learned scoring sees inference-only inputs; evaluation truth still comes from cheap BFS.",
        independent_count="Distinct original graph units across exposed development roles; zero new confirmation graphs.",
        gate_check_summary="Name upstream, exact field, operator, expected and observed including missing files, fields, zero and changed bytes.",
        gradient_rows="Separate potential frozen derivatives from actual zero frozen updates and original joint-training derivatives.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return deepcopy(value)
