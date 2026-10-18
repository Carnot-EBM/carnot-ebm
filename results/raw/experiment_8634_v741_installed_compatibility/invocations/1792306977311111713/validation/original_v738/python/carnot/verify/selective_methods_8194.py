"""REQ-REPORT-8194: test selective decisions on authenticated exposed evidence.

The current run trains only two small heads. Imported Qwen calls belong to
historical extraction; execution readiness is separate from scientific utility.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import sentence_decision_audit_8185 as audit
from carnot.verify import sentence_energy_fit_8183 as fit
from carnot.verify import selective_rule_8194 as n
import yaml

Json = dict[str, Any]
ROOT = fit.ROOT
NAME = "experiment_8194_v708_selective_methods"
TASK = "exp8194-selective-methods"
MODULE = "python/carnot/verify/selective_methods_8194.py"
NUMERIC = "python/carnot/verify/selective_rule_8194.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_selective_methods_8194.py"
OWNED = [MODULE, NUMERIC, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
PROTOCOL = "openspec/change-proposals/v708-selective-decision-protocol.json"
PROTOCOL_PIN = "sha256:fb5e909618eefd791bd188aac387f368dd3b4d9d759242d11cfafc3bea0a1e6d"
AUDIT = "results/experiment_8185_v707_sentence_decision_audit.json"
AUDIT_PIN = "sha256:85e67f7b39e7d89e621326a0bc8572468c33b43e7caee781628c69ff67cf1083"
CONFIG = dict(audit.CONFIG, seed=n.SEED, alpha=0.05, H1_alpha=0.025, H2_alpha=0.025)
execution = audit.execution
reference = audit.reference
progress = n.progress
ORIGINAL_AUDIT_REDUCE = audit.reduce


def record_minimum(work: Json, path: Path, name: str, expected: int, observed: int) -> None:
    """Keep the numeric failed operand visible instead of recording only a boolean."""
    work["checks"].append(
        dict(
            check=name,
            upstream=str(path),
            path=str(path),
            hash=sha256_file(path),
            artifact_field=name,
            op=">=",
            expected=expected,
            observed=observed,
            passed=observed >= expected,
        )
    )
    if observed < expected:
        raise ValueError(name)


def fixture() -> tuple[list[Json], list[Json], Json]:
    """Scripted oracle labels certify plumbing and never become public evidence."""
    rng = np.random.default_rng(n.SEED)
    rows = []
    for role, count in [("fit", 128), ("tune", 64)]:
        for i in range(count):
            x = rng.normal(size=16).tolist()
            rows.append(
                dict(
                    unit_id=f"{role}{i}",
                    source_cluster_id=f"{role}{i}",
                    source_id=f"{role}{i}",
                    role=role,
                    x=x,
                    y=i % 2,
                    status="completed",
                    exclusion_reason=None,
                )
            )
    reserved = [
        dict(
            unit_id=f"reserved{i}",
            source_cluster_id=f"reserved{i}",
            x=rng.normal(size=16).tolist(),
            historical_x=rng.normal(size=12).tolist(),
            status="completed",
            exclusion_reason=None,
        )
        for i in range(128)
    ]
    g = fit.energy.BASE_GEOMETRY(
        np.asarray([r["x"][:12] for r in rows[:128]]), [r["source_cluster_id"] for r in rows[:128]]
    )
    return (
        rows,
        reserved,
        dict(arm="radial16", geometry=g, weights=[0.0] * 17, calibration=[0.0, 1.0]),
    )


def reduce(data: Json) -> Json:
    """Recompute primitive probabilities and costs without trusting producer totals."""
    if not 0 < data["clock"]["predictions_sealed_ns"] < data["clock"]["labels_opened_ns"]:
        raise ValueError("label_clock")
    targets = {r["unit_id"]: r for r in data["targets"]}
    if len(targets) != 128 or set(targets) != {r["unit_id"] for r in data["features"]}:
        raise ValueError("target_join")
    if data.get("original_audit_evidence"):
        ORIGINAL_AUDIT_REDUCE(data["original_audit_evidence"])
        if data["targets"] != data["original_audit_evidence"]["targets"]:
            raise ValueError("original_labels")
    scalar = n.predict(data["features"], data["heads"], data["frozen"], scalar=True)
    if len(scalar) != len(data["predictions"]):
        raise ValueError("prediction_count")
    scored, parity = [], []
    for expected, actual in zip(scalar, data["predictions"], strict=True):
        p, ap = expected["p"], actual["p"]
        if (
            any(expected[k] != actual[k] for k in expected if k != "p")
            or (p is None) != (ap is None)
            or (p is not None and abs(p - ap) > 1e-10)
        ):
            raise ValueError("independent_prediction_drift")
        target = targets[expected["unit_id"]]
        if target["source_cluster_id"] != expected["source_cluster_id"] or target["y"] not in (
            0,
            1,
        ):
            raise ValueError("original_target_identity")
        scored.append(audit.score(dict(expected, p=ap), target["y"]))
        if expected["arm"] == "equivalent_logistic_set" and p is not None:
            parity.append(
                dict(
                    unit_id=expected["unit_id"],
                    maximum_absolute_error=abs(p - ap),
                    passed=abs(p - ap) <= 1e-10,
                )
            )
    return dict(
        rows=scored,
        **n.statistics(scored),
        equivalent_logistic_parity=dict(
            passed=bool(parity) and all(r["passed"] for r in parity), rows=parity
        ),
    )


def measure(root: Path, raw: Path, *, fixture: bool = False, mutation: str = "") -> Json:
    """Authenticate operands and freeze IDs before labels can affect evaluation."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        raw_shard_hashes=[],
        trained_head_specs=[],
        historical_model_provenance={},
        cited_upstream_artifacts=[],
        role_manifest={},
        class_support_by_role={},
        owned_failure="",
    )
    gate = lambda path, field, expected, observed: fit.gate(work, path, field, expected, observed)
    read = lambda ref: fit.bind(work, ref, raw)
    progress("before_preconditions")
    try:
        gate(Path(sys.executable), "python_runtime_supported", True, sys.version_info >= (3, 11))
        probe = raw / ".storage_probe"
        probe.write_bytes(b"private writable custody")
        gate(
            raw, "private_writable_storage", True, probe.read_bytes() == b"private writable custody"
        )
        probe.unlink()
        protocol = read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN))
        work["protocol"] = protocol
        retired_path = ROOT / "ops/exclusion_manifest.yaml"
        retired = yaml.safe_load(retired_path.read_text())
        retired_ids = {
            r.get("experiment_id")
            for k in ("retired", "retired_experiments")
            for r in retired.get(k, [])
        }
        gate(
            retired_path,
            "selective_upstream_not_retired",
            True,
            not bool({8182, 8184, 8185} & retired_ids),
        )
        if fixture:
            rows, reserved, frozen = globals()["fixture"]()
            roles = n.freeze_roles(rows, reserved)
            plan: Json = dict(refs=[], controls=[], upstream=[], historical_model_provenance={})
        else:
            plan = fit.inputs(root, raw / "fit_inputs")
            work["checks"].extend(plan["checks"])
            work["refs"].extend(plan["refs"])
            if any(not c["passed"] for c in plan["checks"]):
                raise ValueError("fit_upstream")
            rows = plan["rows"]
            for r in rows:
                r["source_id"] = protocol["fit_source_id_map"][r["unit_id"]]
            reserved = protocol["role_manifest"]["reserved"]
            roles = n.freeze_roles(rows, reserved)
            normalize = lambda rs: sorted(rs, key=lambda r: r["source_cluster_id"])
            gate(
                ROOT / PROTOCOL,
                "frozen_role_ids",
                {k: normalize(v) for k, v in protocol["role_manifest"].items()},
                {k: normalize(v) for k, v in roles.items()},
            )
            for ref in protocol["source_artifact_hashes"]:
                gate(
                    Path(ref["path"]),
                    "versioned_operand_sha256",
                    ref["sha256"],
                    sha256_file(Path(ref["path"])),
                )
            frozen = next(h for h in plan["controls"] if h["arm"] == "radial16")
        work["role_manifest"] = roles
        atomic_json(raw / "role_manifest.json", roles)
        clock = dict(roles_sealed_ns=time.time_ns())
        progress("roles_saved_before_reserved_labels", 320, 0)
        supports = n.class_support(rows, roles)
        work["class_support_by_role"] = supports
        for role, minimum, per_class in [
            ("head_fit", 72, 0),
            ("temperature_fit", 24, 8),
            ("calibration", 48, 20),
        ]:
            record_minimum(
                work, ROOT / PROTOCOL, role + "_support", minimum, supports[role]["completed"]
            )
            for label in (0, 1):
                record_minimum(
                    work,
                    ROOT / PROTOCOL,
                    role + "_class_" + str(label),
                    per_class,
                    supports[role]["classes"][str(label)],
                )
        if mutation:
            gate(root / fit.UPSTREAM, "fit_trainable_score", 1, 0)
        progress("after_preconditions", 3, 0)
        try:
            trained = n.train(rows, roles, raw)
        except (ValueError, TimeoutError, TypeError) as error:
            work["owned_failure"] = str(error)
            raise
        work["trained_head_specs"] = trained["heads"]
        if fixture:
            original: Json = {}
        else:
            progress("before_reserved_custody")
            path = root / AUDIT
            value = read(dict(path=str(path), sha256=AUDIT_PIN))
            for field, expected in [
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
                ("decision_audit_ready_score", 1),
            ]:
                gate(path, field, expected, value[field])
            gate(
                path,
                "upstream_terminal_passed",
                True,
                read_bound_sidecar(path, fit.historical.publication_sidecar(value))["report"][
                    "passed"
                ],
            )
            old_work = read(value["measurement_reference"])
            original = old_work["evidence"]
            captured = original["capture"]
            features = audit.producer.reduce(
                captured["slots"], captured["calls"], captured["plan"]["baseline"]
            )["feature_rows"]
            by_id = {r["unit_id"]: r for r in features}
            historical = {r["unit_id"]: r for r in captured["plan"]["baseline"]}
            reserved = [
                dict(
                    unit_id=s["unit_id"],
                    source_cluster_id=s["source_cluster_id"],
                    x=by_id[s["unit_id"]]["x"] if s["unit_id"] in by_id else None,
                    historical_x=historical[s["unit_id"]]["x"]
                    if historical[s["unit_id"]]["status"] == "completed"
                    else None,
                    status="completed" if s["unit_id"] in by_id else "excluded",
                    exclusion_reason=None if s["unit_id"] in by_id else "missing_evidence",
                )
                for s in roles["reserved"]
            ]
            work["historical_model_provenance"] = plan["historical_model_provenance"]
            work["cited_upstream_artifacts"] = [
                *plan["upstream"],
                dict(
                    experiment_id=8185,
                    fields_imported=[
                        "measurement_reference.evidence",
                        "original human targets",
                        "frozen radial16",
                    ],
                    sha256=AUDIT_PIN,
                ),
            ]
            progress("after_reserved_custody", 128, 0)
        predictions = n.predict(reserved, trained["heads"], frozen)
        atomic_json(raw / "sealed_predictions.json", dict(rows=predictions))
        clock["predictions_sealed_ns"] = time.time_ns()
        clock["labels_opened_ns"] = time.time_ns()
        targets = (
            original["targets"]
            if not fixture
            else [
                dict(unit_id=r["unit_id"], source_cluster_id=r["source_cluster_id"], y=i % 2)
                for i, r in enumerate(reserved)
            ]
        )
        data = dict(
            features=reserved,
            heads=trained["heads"],
            frozen=frozen,
            targets=targets,
            predictions=predictions,
            clock=clock,
            role_manifest=roles,
            original_audit_evidence=original,
        )
        reduced = reduce(data)
        supports["reserved"] = dict(
            intended=128,
            completed=reduced["eligible_count"],
            classes=reduced["class_support_reserved"],
        )
        record_minimum(
            work, ROOT / PROTOCOL, "reserved_complete_pairs", 96, reduced["eligible_count"]
        )
        for label in (0, 1):
            record_minimum(
                work,
                ROOT / PROTOCOL,
                "reserved_class_" + str(label),
                12,
                supports["reserved"]["classes"][str(label)],
            )
        work["evidence"] = data
        for name, value in [
            ("decision_evidence", data),
            ("independent_reduction", reduced),
            ("matched_fit_features", dict(rows=rows)),
        ]:
            atomic_json(raw / (name + ".json"), value)
        work["raw_shard_hashes"] = [
            reference(raw / (p + ".json"))
            for p in [
                "role_manifest",
                "frozen_heads",
                "sealed_predictions",
                "decision_evidence",
                "independent_reduction",
                "matched_fit_features",
            ]
        ]
        if not fixture:
            work["raw_shard_hashes"] += [
                dict(path=r["path"], sha256=r["sha256"])
                for r in protocol["literature_source_receipts"]
                if r["status"] == "readable"
            ]
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        if not work["owned_failure"] and (not work["checks"] or work["checks"][-1]["passed"]):
            try:
                gate(
                    root / fit.UPSTREAM,
                    "input_custody",
                    "authenticated_original_evidence",
                    str(error),
                )
            except ValueError:
                pass
    work.update(
        duration_s=time.monotonic() - began,
        code_config_hashes=[
            reference(ROOT / p)
            for p in [*OWNED, TEST, PROTOCOL, fit.NUMERIC, fit.historical.NUMERIC]
        ],
        phase_spans=[
            dict(
                phase="authenticate_roles_fit_temperature_calibrate_evaluate",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", int(bool(work["evidence"])), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Use the qualified reporting schema while keeping readiness and H1 separate."""
    if work["owned_failure"] and not any(r.get("name") == "owned_head_fitting" for r in receipts):
        receipts = [
            *receipts,
            dict(name="owned_head_fitting", passed=False, actual=work["owned_failure"]),
        ]
    with (
        patch.object(audit, "reduce", reduce),
        patch.object(audit, "statistics", n.statistics),
        patch.object(audit, "TASK", TASK),
        patch.object(audit, "CONFIG", CONFIG),
    ):
        value = audit.build(work, raw, receipts, fixture=fixture)
    value.update(
        experiment_id=8194,
        milestone="2026.10.708",
        task_id=TASK,
        honest_verdict=value["honest_verdict"].replace("sentence_decision", "selective_decision"),
        selective_protocol_ready_score=value.pop("decision_audit_ready_score"),
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PROTOCOL_PIN,
        role_manifest=work["role_manifest"],
        class_support_by_role=work["class_support_by_role"],
        theory_assumptions=work.get("protocol", {}).get("theory_assumptions", {}),
        literature_mapping=work.get("protocol", {}).get("literature_mapping", []),
        finite_sample_fixture_rows=[
            dict(score_count=k, **n.quantile([0.2] * k)) for k in [0, 1, 18, 19, 20, 25, 28]
        ],
        claim_scope="exposed development prediction-set decisions on existing extraction; no fresh or prospective guarantee",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        source_artifact_hashes=work["refs"],
        trained_head_specs=work["trained_head_specs"],
        H2=dict(alpha=0.025, measured_here=False, scope="separate learning test"),
        acceptance_gates=work.get("protocol", {}).get("H1", CONFIG),
        sample_size_budget=dict(
            original_fit_slots=128,
            head_fit_slots=96,
            temperature_slots=32,
            calibration_slots=64,
            reserved_slots=128,
            head_fit_minimum=72,
            temperature_minimum=24,
            temperature_per_class=8,
            calibration_minimum=48,
            calibration_per_class=20,
            reserved_minimum=96,
            reserved_per_class=12,
        ),
        calibration_rows=[],
        methodology_note="Fit-only V707 Gaussian centers and ridge grid; separate bounded scalar temperatures; class-conditional inclusive finite-sample quantiles at alpha=.05. Original typed costs and Bayes thresholds; H1 paired source bootstrap on all128 slots. Imported Qwen provenance is historical; zero current model calls.",
    )
    value["field_principles"].update(
        roles="Identity-only frozen roles precede evaluator targets; calibration cannot fit a head.",
        coverage="Class-conditional label-set coverage differs from error among accepts.",
        theory="Exchangeability is required by conformal theory and is not established here.",
        protocol="Execution readiness does not require positive H1 and never establishes independent benefit.",
    )
    if value["verdict_class"] in ("blocked", "disqualified"):
        value["selective_protocol_ready_score"] = 0
    value["reproducibility_checksum"] = canonical_hash(
        dict(protocol=PROTOCOL_PIN, roles=work["role_manifest"], rows=value["rows"], config=CONFIG)
    )
    return value


def replay(path: Path) -> bool:
    """Rehash bound bytes, independently reduce rows and reject rehashed aggregates."""
    try:
        value = json.loads(path.read_text())
        if value["protocol_sha256"] != PROTOCOL_PIN:
            return False
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        if work["evidence"]:
            for name, expected in [
                ("decision_evidence", work["evidence"]),
                ("independent_reduction", reduce(work["evidence"])),
                ("sealed_predictions", dict(rows=work["evidence"]["predictions"])),
            ]:
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
    """Freeze explicit file paths, private coverage and normal-exit requirements."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = audit.BASE_MANIFEST(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["deadline_s"] = 180
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
    """Reuse bounded supervision, immutable log custody and atomic publication."""
    with ExitStack() as stack:
        for name in [
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
        ]:
            stack.enter_context(patch.object(audit.base, name, globals()[name]))
        return int(audit.base.main(argv))
