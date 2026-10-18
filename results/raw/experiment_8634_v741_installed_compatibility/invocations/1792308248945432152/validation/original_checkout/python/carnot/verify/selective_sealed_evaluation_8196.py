"""REQ-VERIFY-8196: seal cached decisions before evaluator label access.

Original exposed sources qualify execution only. Missing evidence remains in
every arm, so successful sealing does not imply support or scientific benefit.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, redirect_stdout
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import selective_energy_fit_8195 as upstream

Json = dict[str, Any]
fit, n = upstream.fit, upstream.n
ROOT = fit.ROOT
NAME = "experiment_8196_v708_selective_sealed_evaluation"
TASK = "exp8196-selective-sealed-evaluation"
MODULE = "python/carnot/verify/selective_sealed_evaluation_8196.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_selective_sealed_evaluation_8196.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8195_v708_selective_energy_fit.json"
CAPTURE = "results/experiment_8184_v707_reserved_sentence_capture.json"
PINS = {
    UPSTREAM: "sha256:f3c0098b0d8f59847a59539170f089d53c53849d7cb608e1f626eebc1e68de2d",
    CAPTURE: "sha256:e70bc2604c31780fb61a7a13aafb11e97ad422383523a88d8ec21e416e7fbf82",
}
CONFIG = dict(arms=n.ARMS, intended=128, minimum_pairs=96, evaluator_labels=False)
execution, reference, BASE_MAIN = fit.execution, fit.reference, fit.main


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report actual work so waiting children cannot look like idle execution."""
    print(f"[exp8196] phase={phase} completed={completed} pending={pending}", flush=True)


def reject_labels(value: Any) -> None:
    """Reject target fields even when nested inside otherwise public evidence."""
    if isinstance(value, dict):
        if {"y", "human_target", "evaluator_label", "entailment_label"} & value.keys():
            raise ValueError("evaluator_label")
        for item in value.values():
            reject_labels(item)
    elif isinstance(value, list):
        for item in value:
            reject_labels(item)


def reduce(data: Json) -> Json:
    """Rebuild all slots in original order using an independent scalar equation."""
    reject_labels(data)
    frozen, features, roles = data["frozen"], data["features"], data["roles"]
    if canonical_hash(frozen) != data["frozen_content_sha256"]:
        raise ValueError("modified_head")
    if roles != frozen["role_manifest"] or frozen["arms"] != n.ARMS:
        raise ValueError("frozen_roles_or_arms")
    roster = {(r["unit_id"], r["source_cluster_id"]) for r in roles["reserved"]}
    observed = {(r["unit_id"], r["source_cluster_id"]) for r in features}
    if len(features) != 128 or len(roster) != 128 or observed != roster:
        raise ValueError("original_source_join")
    if sorted(r["slot"] for r in features) != list(range(1, 129)):
        raise ValueError("original_slot_join")
    comparator = {r["unit_id"]: r for r in data["comparator"]}
    if (
        len(comparator) != 128
        or len(data["comparator"]) != 128
        or set(comparator) != {r["unit_id"] for r in features}
    ):
        raise ValueError("unmatched_comparator")
    rows, predictions, parity = [], [], []
    for source in sorted(features, key=lambda r: r["slot"]):
        old = comparator[source["unit_id"]]
        if old["source_cluster_id"] != source["source_cluster_id"]:
            raise ValueError("unmatched_comparator_source")
        complete = source["status"] == "completed" and source["x"] is not None
        if complete and (len(source["x"]) != 16 or not all(map(math.isfinite, source["x"]))):
            raise ValueError("feature_dimensions")
        x = source["x"] if complete else None
        public = dict(
            unit_id=source["unit_id"],
            source_cluster_id=source["source_cluster_id"],
            x=x,
            historical_x=x[:12] if x is not None else None,
            status=source["status"],
            exclusion_reason=source["exclusion_reason"],
        )
        scalar = n.predict([public], frozen["heads"], frozen["controls"], scalar=True)
        vector = n.predict([public], frozen["heads"], frozen["controls"])
        for row, energy in zip(scalar, vector, strict=True):
            if row["p"] is not None and abs(row["p"] - energy["p"]) > 1e-10:
                raise ValueError("energy_probability_parity")
            if row["arm"] == "frozen_v707_radial" and complete:
                if old["p"] is None or abs(row["p"] - old["p"]) > 1e-10:
                    raise ValueError("authentic_comparator_probability")
                row.update(p=old["p"], action=old["action"])
            head = frozen["heads"][
                0 if row["arm"] in ("local_set", "local_point", "equivalent_logistic_set") else 1
            ]
            labels = n.prediction_set(row["p"], head["quantiles"])
            row.update(
                metric="sealed_probability",
                numerator=row["p"],
                denominator=int(row["p"] is not None),
                status=source["status"],
                exclusion_reason=source["exclusion_reason"],
                source_sha256=source["source_sha256"],
                answer_sha256=source["answer_sha256"],
                feature_sha256=source["feature_sha256"],
                slot=source["slot"],
                class_probabilities=[1 - row["p"], row["p"]]
                if row["p"] is not None
                else [None, None],
                class_thresholds=head["quantiles"],
                label_set_membership=[y in labels for y in (0, 1)],
                decision_thresholds=frozen["point_thresholds"],
                original_comparator_probability=old["p"],
                original_comparator_action=old["action"],
            )
            predictions.append(dict(row, prediction_sha256=canonical_hash(row)))
        local, logistic = scalar[0], scalar[5]
        if any(local[k] != logistic[k] for k in ("p", "action", "prediction_set")):
            raise ValueError("logistic_decision_parity")
        parity.append(dict(unit_id=source["unit_id"], passed=True))
        rows.append(
            dict(
                unit_id=source["unit_id"],
                source_cluster_id=source["source_cluster_id"],
                slot=source["slot"],
                arm="all_seven_frozen_arms",
                condition="all_original_reserved_slots",
                metric="complete_paired_source",
                numerator=int(complete),
                denominator=1,
                status=source["status"] if not complete else "completed",
                exclusion_reason=source["exclusion_reason"],
            )
        )
    counts = Counter(r["status"] for r in rows)
    return dict(
        rows=rows,
        prediction_rows=predictions,
        intended_count=128,
        independent_count=128,
        eligible_count=counts["completed"],
        completed_count=counts["completed"],
        excluded_count=counts["excluded"],
        failed_count=counts["failed"],
        censored_count=counts["censored"],
        missing_slot_rows=[r for r in rows if not r["numerator"]],
        paired_source_ids=[r["source_cluster_id"] for r in rows if r["numerator"]],
        equivalent_logistic_parity=dict(passed=True, rows=parity),
    )


def fixture() -> Json:
    """Small zero-weight heads qualify custody without training or model calls."""
    rows, reserved, control = upstream.methods.fixture()
    roles = n.freeze_roles(rows, reserved)
    heads = []
    for dimensions in (16, 12):
        g = n.base.geometry(n.np.zeros((2, dimensions)), ["fit-a", "fit-b"])
        heads.append(
            dict(
                arm="local_evidence_radial16" if dimensions == 16 else "radial16",
                dimensions=dimensions,
                geometry=g,
                weights=[0.0] * (len(g["centers"]) + 1),
                temperature=1.0,
                quantiles=[n.quantile([0.6] * 20)] * 2,
            )
        )
    frozen = dict(
        heads=heads, controls=control, role_manifest=roles, arms=n.ARMS, point_thresholds=[0.1, 0.5]
    )
    features, comparator = [], []
    for i, r in enumerate(reserved):
        features.append(
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                slot=i + 1,
                x=r["x"],
                status="completed",
                exclusion_reason=None,
                source_sha256=canonical_hash(r["unit_id"]),
                answer_sha256=canonical_hash("answer"),
                feature_sha256=canonical_hash(r),
            )
        )
        comparator.append(
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                p=0.5,
                action="reject",
            )
        )
    return dict(
        features=features,
        comparator=comparator,
        frozen=frozen,
        roles=roles,
        frozen_content_sha256=canonical_hash(frozen),
    )


def measure(root: Path, raw: Path, *, fixture_mode: bool = False, mutation: str = "") -> Json:
    """Check external custody before reading feature-only reserved views."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        raw_shard_hashes=[],
        precondition_receipts=[],
        trained_head_specs=[],
        historical_model_provenance={},
        cited_upstream_artifacts=[],
        owned_failure="",
        seal_receipt={},
    )
    progress("before_preconditions")
    try:
        fit.gate(
            work,
            Path(sys.executable),
            "python_runtime_supported",
            True,
            sys.version_info >= (3, 11),
        )
        probe = raw / ".storage_probe"
        probe.write_bytes(b"private writable custody")
        fit.gate(
            work,
            raw,
            "private_writable_storage",
            True,
            probe.read_bytes() == b"private writable custody",
        )
        probe.unlink()
        retired_path = ROOT / "ops/exclusion_manifest.yaml"
        retired = yaml.safe_load(retired_path.read_text())
        ids = {
            r.get("experiment_id")
            for k in ("retired", "retired_experiments")
            for r in retired.get(k, [])
        }
        fit.gate(work, retired_path, "upstream_not_retired", True, not bool({8184, 8195} & ids))
        if fixture_mode:
            data = fixture()
            atomic_json(raw / "frozen_heads.json", data["frozen"])
        else:
            values = {}
            for name, pin in PINS.items():
                progress("before_subprocess_summarize_" + Path(name).stem)
                spec = dict(
                    name="summarize_" + Path(name).stem,
                    argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        "scripts/summarize_artifact.py",
                        str(root / name),
                    ],
                    deadline_s=60,
                    expected_exit=0,
                )
                with (
                    (raw / (spec["name"] + "_supervision.log")).open("x") as stream,
                    redirect_stdout(stream),
                ):
                    receipt = execution.run_check(
                        ROOT, spec, raw, raw / "precondition_logs", heartbeat_s=20
                    )
                work["precondition_receipts"].append(receipt)
                progress("after_subprocess_summarize_" + Path(name).stem, 1, 0)
                fit.gate(
                    work, root / name, "upstream_summary_normal_exit", 0, receipt["actual_exit"]
                )
                value = fit.bind(work, dict(path=str(root / name), sha256=pin), raw)
                for field, expected in (
                    ("required_checks_passed", True),
                    ("flagged_adversarial", False),
                    (
                        "selective_fit_ready_score"
                        if name == UPSTREAM
                        else "evaluation_capture_ready_score",
                        1,
                    ),
                ):
                    fit.gate(work, root / name, field, expected, value.get(field))
                report = read_bound_sidecar(root / name, fit.historical.publication_sidecar(value))
                fit.gate(
                    work, root / name, "upstream_terminal_passed", True, report["report"]["passed"]
                )
                values[name] = value
                work["cited_upstream_artifacts"].append(
                    dict(
                        experiment_id=value["experiment_id"],
                        sha256=pin,
                        fields_imported=[
                            "frozen_heads",
                            "original_feature_call_prediction_manifests",
                        ],
                    )
                )
            trained, capture = values[UPSTREAM], values[CAPTURE]
            frozen = fit.bind(
                work,
                dict(path=trained["frozen_heads_path"], sha256=trained["frozen_heads_sha256"]),
                raw,
            )
            (raw / "frozen_heads.json").write_bytes(Path(trained["frozen_heads_path"]).read_bytes())
            roles = fit.bind(
                work,
                next(
                    r
                    for r in trained["raw_shard_hashes"]
                    if Path(r["path"]).stem == "role_manifest"
                ),
                raw,
            )
            shards = {Path(r["path"]).stem: r for r in capture["raw_shard_hashes"]}
            features = fit.bind(work, shards["complete_features"], raw)["feature_rows"]
            calls = fit.bind(work, shards["primitive_calls"], raw)["rows"]
            prior = fit.bind(work, shards["sealed_predictions"], raw)["rows"]
            fit.gate(work, root / CAPTURE, "original_call_manifest", capture["call_ledger"], calls)
            for ref in trained["code_config_hashes"]:
                if Path(ref["path"]).name in ("selective_rule_8194.py", "evidence_energy_8154.py"):
                    fit.gate(
                        work,
                        Path(ref["path"]),
                        "prediction_runtime_sha256",
                        ref["sha256"],
                        sha256_file(Path(ref["path"])),
                    )
            protocol = fit.bind(
                work,
                dict(
                    path=str(ROOT / upstream.methods.PROTOCOL), sha256=upstream.methods.PROTOCOL_PIN
                ),
                raw,
            )
            fit.gate(
                work,
                root / UPSTREAM,
                "frozen_protocol_sha256",
                upstream.methods.PROTOCOL_PIN,
                frozen["protocol_sha256"],
            )
            fit.gate(
                work,
                root / UPSTREAM,
                "original_reserved_roles",
                {
                    k: sorted(v, key=lambda r: r["source_cluster_id"])
                    for k, v in protocol["role_manifest"].items()
                },
                {k: sorted(v, key=lambda r: r["source_cluster_id"]) for k, v in roles.items()},
            )
            work["trained_head_specs"] = trained["trained_head_specs"]
            work["trained_head_specs"].append(
                dict(
                    arm="frozen_v707_radial",
                    dimensions=12,
                    parameter_count=len(frozen["controls"]["weights"]) + 2,
                    trained_in_current_run=False,
                    imported_from_experiment=8154,
                    head_sha256=canonical_hash(frozen["controls"]),
                )
            )
            work["historical_model_provenance"] = dict(
                experiment_id=8184,
                model_invocation_counts=capture["model_invocation_counts"],
                call_ledger=calls,
                charged_current_model_work=0,
                acquisition_clock_scope="original_8184_calls",
            )
            baseline_ref = next(
                r
                for r in capture["source_artifact_hashes"]
                if Path(r["path"]).name.endswith("-primitive_calls.json")
            )
            fit.gate(
                work,
                Path(baseline_ref["path"]),
                "original_baseline_call_manifest_sha256",
                baseline_ref["sha256"],
                sha256_file(Path(baseline_ref["path"])),
            )
            # Keep the immutable clock manifest by reference. Its legacy schema
            # has target fields, so the label-blind worker must not parse it.
            work["historical_model_provenance"]["original_baseline_call_manifest"] = baseline_ref
            reject_labels([features, calls, prior])
            source_ids = {(r["unit_id"], r["source_cluster_id"]) for r in features}
            for call in calls:
                fit.gate(
                    work,
                    root / CAPTURE,
                    "original_call_source_and_clock",
                    True,
                    (call["unit_id"], call["source_cluster_id"]) in source_ids
                    and call["ended_monotonic_ns"] >= call["started_monotonic_ns"],
                )
            data = dict(
                features=features,
                comparator=[r for r in prior if r["arm"] == "radial16"],
                frozen=frozen,
                roles=roles,
                frozen_content_sha256=canonical_hash(frozen),
            )
        if mutation:
            fit.gate(work, root / UPSTREAM, "selective_fit_ready_score", 1, 0)
        progress("after_preconditions", 128, 0)
        progress("before_benchmark_apply_frozen_heads", 0, 128)
        try:
            reduced = reduce(data)
        except ValueError as error:
            if str(error).endswith("_parity"):
                work["owned_failure"] = str(error)
            raise
        progress("after_benchmark_apply_frozen_heads", 128, 0)
        work["evidence"] = data
        for name, content in (
            ("primitive_evidence", data),
            ("role_manifest", data["roles"]),
            ("sealed_predictions", dict(rows=reduced["prediction_rows"], labels_opened=False)),
            ("independent_reduction", reduced),
        ):
            atomic_json(raw / (name + ".json"), content)
        work["seal_receipt"] = dict(
            labels_opened=False,
            prediction_count=896,
            predictions=reference(raw / "sealed_predictions.json"),
            roles=reference(raw / "role_manifest.json"),
            heads=reference(raw / "frozen_heads.json"),
            order="heads_roles_predictions_before_evaluator",
        )
        work["raw_shard_hashes"] = [
            reference(raw / (name + ".json"))
            for name in (
                "primitive_evidence",
                "role_manifest",
                "sealed_predictions",
                "independent_reduction",
                "frozen_heads",
            )
        ]
        progress("predictions_roles_and_heads_sealed", 896, 0)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        if not work["owned_failure"] and all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="input_custody",
                    upstream=str(root / UPSTREAM),
                    path=str(root / UPSTREAM),
                    hash=None,
                    artifact_field="input_custody",
                    op="==",
                    expected="authenticated_label_free_reserved_evidence",
                    observed=str(error),
                    passed=False,
                )
            )
    work.update(
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_apply_and_seal", start_s=0, duration_s=time.monotonic() - began
            )
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                upstream.methods.NUMERIC,
                fit.historical.NUMERIC,
                upstream.methods.PROTOCOL,
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", int(bool(work["evidence"])), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Sealing readiness needs validation; support and later utility stay separate."""
    reduced = (
        reduce(work["evidence"])
        if work["evidence"]
        else dict(
            rows=[],
            prediction_rows=[],
            intended_count=128,
            independent_count=0,
            eligible_count=0,
            completed_count=0,
            excluded_count=128,
            failed_count=0,
            censored_count=0,
            missing_slot_rows=[],
            paired_source_ids=[],
            equivalent_logistic_parity=dict(passed=False, rows=[]),
        )
    )
    checked = (
        bool(receipts)
        and all(r.get("passed") is True for r in receipts)
        and not work["owned_failure"]
    )
    failures = [c for c in work["checks"] if not c["passed"]]
    ready = int(checked and not failures and len(reduced["prediction_rows"]) == 896)
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
    )
    suffix = (
        "owned_validation"
        if not checked
        else failures[0]["check"]
        if failures
        else "selective_predictions_sealed"
    )
    seal = work["seal_receipt"]
    support = reduced["completed_count"] >= 96
    value: Json = dict(
        experiment_id=8196,
        task_id=TASK,
        milestone="2026.10.708",
        run_date=RUN_DATE,
        honest_verdict=f"complete_{verdict}_{suffix}",
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        fixture_protocol_only=fixture,
        claim_scope="Sealed selective predictions on original exposed development slots; label audit and class support await Exp8197",
        exposure_scope="registered_extension_on_previously_evaluated_development_sources",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=receipts,
        precondition_receipts=work["precondition_receipts"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        trained_head_specs=work["trained_head_specs"],
        model_invocation_counts=dict(model_loads=0, generations=0, live_model_calls=0),
        call_ledger=[],
        historical_model_provenance=work["historical_model_provenance"],
        cited_upstream_artifacts=work["cited_upstream_artifacts"],
        sample_size_budget=CONFIG,
        duration_s=work["duration_s"],
        random_seed=n.SEED,
        reproducibility_checksum=canonical_hash(
            dict(config=CONFIG, refs=work["refs"], reduced=reduced)
        ),
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        acceptance_gates=CONFIG,
        field_principles=dict(
            readiness="Normal owned exits certify sealing separately from scientific support.",
            inference="Historical clocks are retained; no current model work is charged.",
            support="All128 slots remain; low support gives a terminal null in the later audit.",
            custody="Head bytes, source roles and prediction rows seal before evaluator labels.",
        ),
        sealed_evaluation_ready_score=ready,
        predictions_path=seal.get("predictions", {}).get("path"),
        predictions_sha256=seal.get("predictions", {}).get("sha256"),
        seal_receipt=seal,
        role_manifest_sha256=seal.get("roles", {}).get("sha256"),
        frozen_heads_sha256=seal.get("heads", {}).get("sha256"),
        complete_pair_support_sufficient=support,
        support_audit=dict(
            pair_minimum=96,
            complete_pairs=reduced["completed_count"],
            class_minimum=12,
            class_support=None,
            insufficient_support_verdict="complete_null_insufficient_support",
            evaluator_labels_opened=False,
            later_audit_required=True,
        ),
        evaluator_targets_opened=False,
        label_access_ledger=[],
        measurement_reference=reference(raw / "measurement.json"),
        repository_health=work.get("repository_health", {}),
        methodology_note="Scalar frozen-head application on cached label-free features; authentic prior radial rows; all incomplete sources escalate in all seven arms. Exact logistic decisions provide arithmetic parity only.",
        **reduced,
    )
    return value


def replay(path: Path) -> bool:
    """Rehash immutable bytes and rebuild headlines rather than trusting hashes alone."""
    try:
        value = json.loads(path.read_text())
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
            *(
                [value["historical_model_provenance"]["original_baseline_call_manifest"]]
                if "original_baseline_call_manifest" in value["historical_model_provenance"]
                else []
            ),
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in [*value["validation_receipts"], *value["precondition_receipts"]]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        if work["evidence"] and not value["fixture_protocol_only"]:
            copies = {Path(r["path"]).name.split("-", 1)[1]: r for r in work["refs"]}
            originals = {
                name: json.loads(Path(copies[Path(name).name]["path"]).read_text()) for name in PINS
            }
            if any(copies[Path(name).name]["sha256"] != pin for name, pin in PINS.items()):
                return False
            capture, trained = originals[CAPTURE], originals[UPSTREAM]
            refs = [
                r
                for r in capture["raw_shard_hashes"]
                if Path(r["path"]).stem
                in ("complete_features", "primitive_calls", "sealed_predictions")
            ]
            refs += [
                r
                for r in trained["raw_shard_hashes"]
                if Path(r["path"]).stem in ("frozen_heads", "role_manifest")
            ]
            saved = {Path(r["path"]).stem: copies[Path(r["path"]).name] for r in refs}
            if any(
                saved[Path(r["path"]).stem]["sha256"] != r["sha256"]
                for r in refs
                if Path(r["path"]).stem
                in (
                    "complete_features",
                    "primitive_calls",
                    "sealed_predictions",
                    "frozen_heads",
                    "role_manifest",
                )
            ):
                return False
            load = lambda name: json.loads(Path(saved[name]["path"]).read_text())
            evidence = work["evidence"]
            if (
                evidence["features"] != load("complete_features")["feature_rows"]
                or evidence["frozen"] != load("frozen_heads")
                or evidence["roles"] != load("role_manifest")
                or evidence["comparator"]
                != [r for r in load("sealed_predictions")["rows"] if r["arm"] == "radial16"]
                or work["historical_model_provenance"]["call_ledger"]
                != load("primitive_calls")["rows"]
            ):
                return False
        if work["evidence"]:
            actual = reduce(work["evidence"])
            for name, expected in (
                ("primitive_evidence", work["evidence"]),
                ("independent_reduction", actual),
                ("frozen_heads", work["evidence"]["frozen"]),
                ("role_manifest", work["evidence"]["roles"]),
                ("sealed_predictions", dict(rows=actual["prediction_rows"], labels_opened=False)),
            ):
                ref = next(r for r in work["raw_shard_hashes"] if Path(r["path"]).stem == name)
                if json.loads(Path(ref["path"]).read_text()) != expected:
                    return False
        return bool(
            value
            == build(
                work,
                Path(value["terminal_validation_sidecar_path"]).parent,
                value["validation_receipts"],
                fixture=value["fixture_protocol_only"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Static tools receive paths; pytest alone receives test selection arguments."""
    with patch.object(upstream, "OWNED", OWNED), patch.object(upstream, "TEST", TEST):
        with (
            patch.object(execution, "e", sys.modules[__name__]),
            patch.object(execution, "OWNED", OWNED),
        ):
            specs = execution.manifest(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["deadline_s"] = 300
    specs["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    specs["repository_health"]["deadline_s"] = 300
    return specs


def main(argv: list[str] | None = None) -> int:
    """Reuse qualified supervision and publication so caller paths stay private."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(execution, "run_check", n.audit.producer.supervise))
        for name in (
            "ROOT",
            "NAME",
            "TASK",
            "MODULE",
            "CLI",
            "TEST",
            "OWNED",
            "RUN_DATE",
            "MODEL_SPECS",
            "progress",
            "manifest",
            "measure",
            "build",
            "replay",
        ):
            stack.enter_context(patch.object(fit, name, globals()[name]))
        return int(BASE_MAIN(argv))
