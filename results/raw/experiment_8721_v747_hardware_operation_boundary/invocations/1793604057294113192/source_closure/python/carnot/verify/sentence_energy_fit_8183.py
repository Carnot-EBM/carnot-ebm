"""REQ-REPORT-8183: seal exposed CPU heads for an honest later reserved test.

The gate, immutable inputs, primitive reductions and normal validation exits
bind readiness. No fit/tune result claims independent scientific benefit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import evidence_energy_fit_8154 as historical
from carnot.verify import fit_sentence_capture_8182 as capture
from carnot.verify import sentence_energy_8183 as energy

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8183_v707_sentence_energy_fit"
TASK = "exp8183-sentence-energy-fit"
MODULE = "python/carnot/verify/sentence_energy_fit_8183.py"
NUMERIC = "python/carnot/verify/sentence_energy_8183.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_sentence_energy_fit_8183.py"
OWNED = [MODULE, NUMERIC, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8182_v707_fit_sentence_capture.json"
METHODS = "results/experiment_8179_v707_sentence_transport_methods.json"
CONTROL = "results/experiment_8154_v705_evidence_energy_fit.json"
PINS = {
    UPSTREAM: "sha256:6f0f6f3dd5c1b067ab50040a3ce9bc3358a48dee2e71180cf91b3c335bdbcf19",
    METHODS: "sha256:9e64df8a53c2542a7af5c1ed3e1c139bfd463b6023933efb8d40996a73f65ea1",
    CONTROL: "sha256:5ab0405b7eb9d06fb461bbde2038f88e2c44c32a29ab02b2f650983521200350",
}
CONFIG = energy.CONFIG
reference = historical.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase counters before and after measurement or child work."""
    print(f"[exp8183] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(plan: Json, path: Path, field: str, expected: Any, observed: Any) -> None:
    """Record the exact failed external operand before stopping input access."""
    plan["checks"].append(
        dict(
            check=field,
            upstream=str(path),
            path=str(path.absolute()),
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


def bind(plan: Json, ref: Json, raw: Path) -> Json:
    """Copy authenticated JSON bytes into invocation custody for cold replay."""
    path = Path(ref["path"])
    gate(plan, path, "input_sha256", ref["sha256"], sha256_file(path) if path.is_file() else None)
    target = raw / "inputs" / (ref["sha256"][7:] + "-" + path.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    plan["refs"].append(reference(target))
    return dict(json.loads(target.read_text()))


def inputs(root: Path, raw: Path) -> Json:
    """Authenticate gates and reconstruct only fit/tune source-matched features.

    Protocol and head references come from sealed versioned bytes. The reserved
    source manifest and its targets are never opened by this experiment.
    """
    plan: Json = dict(
        checks=[],
        refs=[],
        rows=[],
        controls=[],
        protocol={},
        historical_model_provenance={},
        upstream=[],
    )
    progress("before_input_authentication")
    try:
        for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
            gate(
                plan,
                ROOT / ".venv/bin" / tool,
                "runtime_tool_executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        retired = yaml.safe_load((ROOT / "ops/exclusion_manifest.yaml").read_text())
        ids = {
            r.get("experiment_id")
            for key in ("retired", "retired_experiments")
            for r in retired.get(key, [])
        }
        gate(
            plan,
            ROOT / "ops/exclusion_manifest.yaml",
            "upstream_not_retired",
            True,
            not bool({8154, 8179, 8182} & ids),
        )
        values = {}
        for name, pin in PINS.items():
            path = root / name
            value = bind(plan, dict(path=str(path), sha256=pin), raw)
            for field, expected in (
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ):
                gate(plan, path, field, expected, value.get(field))
            gate(
                plan,
                path,
                "upstream_terminal_passed",
                True,
                read_bound_sidecar(path, historical.publication_sidecar(value))["report"]["passed"],
            )
            values[name] = value
            plan["upstream"].append(
                dict(
                    experiment_id=value["experiment_id"],
                    sha256=pin,
                    fields_imported=[
                        "raw_shard_hashes",
                        "fit_tune_features",
                        "frozen_heads",
                        "protocol",
                        "original_tune_costs",
                    ],
                )
            )
            if name == UPSTREAM:
                gate(plan, path, "fit_trainable_score", 1, value.get("fit_trainable_score"))
        source, methods, control = (values[p] for p in (UPSTREAM, METHODS, CONTROL))
        numerical_ref = next(
            r for r in control["code_config_hashes"] if r["path"].endswith(historical.NUMERIC)
        )
        gate(
            plan,
            Path(numerical_ref["path"]),
            "qualified_fit_code_sha256",
            numerical_ref["sha256"],
            sha256_file(Path(numerical_ref["path"])),
        )
        plan["protocol"] = bind(
            plan, dict(path=methods["protocol_path"], sha256=methods["protocol_sha256"]), raw
        )
        gate(
            plan,
            Path(methods["protocol_path"]),
            "feature_schema",
            methods["feature_schema"],
            plan["protocol"]["features"],
        )
        gate(
            plan,
            Path(methods["protocol_path"]),
            "immutable_cost_matrix",
            historical.energy.CONFIG["costs"],
            plan["protocol"]["decision_costs"],
        )
        calls = bind(plan, source["raw_shard_hashes"][0], raw)["rows"]
        slots = bind(plan, source["raw_shard_hashes"][1], raw)["rows"]
        original = bind(plan, control["raw_shard_hashes"][0], raw)["rows"]
        frozen = bind(plan, control["frozen_head_manifest"], raw)
        plan["controls"] = [
            next(h for h in frozen["heads"] if h["arm"] == a) for a in energy.CONTROL_ARMS
        ]
        reduced = capture.reduce(slots, calls, original)
        gate(
            plan,
            root / UPSTREAM,
            "original_role_counts",
            dict(fit=128, tune=64),
            dict(Counter(r["role"] for r in reduced["rows"])),
        )
        gate(
            plan,
            root / UPSTREAM,
            "feature_primitive_parity",
            source["feature_rows"],
            reduced["feature_rows"],
        )
        features = {r["unit_id"]: r for r in reduced["feature_rows"]}
        originals = {r["unit_id"]: r for r in original}
        for s in reduced["rows"]:
            old = originals[s["unit_id"]]
            row = features.get(s["unit_id"])
            plan["rows"].append(
                dict(
                    unit_id=s["unit_id"],
                    source_cluster_id=s["source_cluster_id"],
                    role=s["role"],
                    slot=s["slot"],
                    x=row["x"] if row else None,
                    y=old["y"],
                    status="completed" if row else "excluded",
                    exclusion_reason=None
                    if row
                    else s["exclusion_reason"] or "missing_paired_features",
                    historical_paired_control=old,
                )
            )
        energy.validate_rows(plan["rows"])
        plan["historical_model_provenance"] = dict(
            MODEL_SPECS=source["MODEL_SPECS"],
            imported_invocation_counts=source["model_invocation_counts"],
            sha256=PINS[UPSTREAM],
            scope="historical_only_zero_current_calls",
        )
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as error:
        if all(c["passed"] for c in plan["checks"]):
            plan["checks"].append(
                dict(
                    check="input_structure",
                    upstream=UPSTREAM,
                    path=str(root / UPSTREAM),
                    hash=None,
                    artifact_field="input_structure",
                    op="==",
                    expected="bound_fit_tune_primitives",
                    observed=str(error),
                    passed=False,
                )
            )
    progress("after_input_authentication", len(plan["rows"]), 192 - len(plan["rows"]))
    return plan


def fixture() -> tuple[list[Json], list[Json], Json]:
    """Build private scripted signals that certify execution, not scientific truth."""
    rng = np.random.default_rng(CONFIG["seed"])
    rows = []
    for role, count in (("fit", 32), ("tune", 16)):
        for i in range(count):
            x = rng.normal(size=16).tolist()
            x[12:] = rng.uniform(size=4).tolist()
            row = dict(
                unit_id=f"{role}{i}",
                source_cluster_id=f"{role}{i}",
                role=role,
                x=x,
                y=i % 2,
                status="completed",
                exclusion_reason=None,
            )
            row["historical_paired_control"] = dict(row, x=x[:12])
            rows.append(row)
    g = historical.energy.geometry(
        np.asarray([r["x"][:12] for r in rows[:32]]), [r["unit_id"] for r in rows[:32]]
    )
    controls = [
        dict(
            arm=a,
            geometry=g,
            weights=[0.0]
            * historical.energy.design(a, np.asarray([rows[0]["x"][:12]]), g).shape[1],
            calibration=[0.0, 1.0],
            ridge=0.01,
            converged=True,
        )
        for a in energy.CONTROL_ARMS
    ]
    protocol = dict(
        features=historical.CONFIG["features"]
        + [
            "local_unsupported_mean",
            "local_unsupported_max",
            "local_contradicted_fraction",
            "local_baseless_fraction",
        ],
        decision_costs=historical.energy.CONFIG["costs"],
        H1=dict(
            original_tune_costs=dict(scalar_span=0.3515625, linear12=0.4140625, radial16=0.3359375)
        ),
    )
    return rows, controls, protocol


def measure(root: Path, raw: Path, *, fixture_mode: bool = False, mutation: str = "") -> Json:
    """Measure only owned CPU fitting time and preserve every missing source slot."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    if fixture_mode:
        rows, controls, protocol = fixture()
        plan = dict(
            checks=[],
            refs=[],
            rows=rows,
            controls=controls,
            protocol=protocol,
            upstream=[],
            historical_model_provenance={},
        )
    else:
        plan = inputs(root, raw)
    if mutation:
        try:
            gate(plan, root / UPSTREAM, "fit_trainable_score", 1, 0)
        except ValueError:
            pass
    atomic_json(raw / "matched_features.json", dict(rows=plan["rows"]))
    code_snapshots = []
    for p in OWNED:
        source = ROOT / p
        target = raw / "prediction_code" / (sha256_file(source)[7:] + "-" + source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        code_snapshots.append(reference(target))
    atomic_json(
        raw / "frozen_configuration.json",
        dict(config=CONFIG, protocol=plan["protocol"], prediction_code_hashes=code_snapshots),
    )
    fitted: Json = dict(heads=[], failures=[], fit_fold_rows=[], training_budget=CONFIG)
    owned_failure = ""
    if all(c["passed"] for c in plan["checks"]):
        progress("before_benchmark_fit")
        try:
            fitted = energy.train(plan["rows"], plan["controls"], plan["protocol"], raw)
        except (ValueError, TimeoutError) as error:
            owned_failure = str(error)
        progress("after_benchmark_fit", len(fitted["heads"]), 6 - len(fitted["heads"]))
    fitted["prediction_code_hashes"] = code_snapshots
    fitted["configuration_reference"] = reference(raw / "frozen_configuration.json")
    atomic_json(raw / "frozen_heads.json", fitted)
    result = energy.evaluate(plan["rows"], [h for h in fitted["heads"] if "calibration" in h])
    atomic_json(raw / "primitive_decisions.json", result)
    work = dict(
        plan=plan,
        fitted=fitted,
        result=result,
        owned_failure=owned_failure,
        raw_shard_hashes=[
            reference(raw / p)
            for p in (
                "matched_features.json",
                "frozen_heads.json",
                "primitive_decisions.json",
                "frozen_configuration.json",
            )
        ]
        + code_snapshots,
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_fit_calibrate_seal",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                historical.NUMERIC,
                historical.MODULE,
                capture.MODULE,
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/methods_stream_execution_8111.py",
            ]
        ],
        config_sha256=canonical_hash(CONFIG),
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(plan["rows"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Reuse the qualified reporting contract, then bind this experiment's scope."""
    with (
        patch.object(historical, "energy", energy),
        patch.object(historical, "TASK", TASK),
        patch.object(historical, "RUN_DATE", RUN_DATE),
        patch.object(historical, "CONFIG", CONFIG),
    ):
        value = historical.build(work, raw, receipts, fixture=fixture)
    plan, fitted = work["plan"], work["fitted"]
    value.update(
        experiment_id=8183,
        milestone="2026.10.707",
        honest_verdict=value["honest_verdict"].replace(
            "evidence_energy_fit", "sentence_energy_fit"
        ),
        comparator_id=fitted.get("comparator_id"),
        cited_upstream_artifacts=plan["upstream"],
        calibration_receipts=[
            dict(
                arm=h["arm"],
                receipt=h["calibration_receipt"],
                policy=h["policy"],
                source_ids=h["tune_source_ids"],
            )
            for h in fitted["heads"]
            if h["arm"] in energy.NEW_ARMS and "calibration" in h
        ],
        calibration_rows=[r for r in value["tuning_rows"] if r["arm"] in energy.NEW_ARMS],
        fitting_rows=[r for r in plan["rows"] if r["role"] == "fit"],
        equivalent_logistic_parity=value["energy_logistic_parity_rows"],
        role_access_ledger=[
            dict(
                role=role,
                fields=["features", "original_human_target"],
                source_count=sum(r["role"] == role for r in plan["rows"]),
                purpose="fit_geometry_weights" if role == "fit" else "calibration_policy",
            )
            for role in ("fit", "tune")
        ],
        frozen_thresholds=[
            dict(arm=h["arm"], policy=h["policy"]) for h in fitted["heads"] if "policy" in h
        ],
        reserved_outcomes_opened=False,
        claim_scope="sealed calibrated source energy decisions; exposed fit/tune only; reserved H1 untested",
    )
    value["imported_control_head_specs"] = [
        h for h in value["trained_head_specs"] if h["arm"] in energy.CONTROL_ARMS
    ]
    value["trained_head_specs"] = [
        h for h in value["trained_head_specs"] if h["arm"] in energy.NEW_ARMS
    ]
    value["methodology_note"] = (
        "Qualified V705 fit-only Gaussian geometry and ridge grid; sixteen registered source signals; tune-only affine calibration and minimum typed-cost thresholds. Original V705 heads retain their twelve-input information. Logistic parity adds no correctness evidence."
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Recompute predictions and headlines while rejecting changed custody bytes."""
    try:
        value = json.loads(path.read_text())
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for r in value["validation_receipts"]:
            if r.get("log_path") and sha256_file(Path(r["log_path"])) != r["log_sha256"]:
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        frozen = json.loads(Path(value["frozen_head_manifest"]["path"]).read_text())
        if frozen != work["fitted"] or work["result"] != energy.evaluate(
            work["plan"]["rows"], [h for h in frozen["heads"] if "calibration" in h]
        ):
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
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze real file paths and owned checks before any fitting starts."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = execution.manifest(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["deadline_s"] = 600
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
        "tests/python/test_evidence_energy_8154.py",
    ]
    return specs


def main(argv: list[str] | None = None) -> int:
    """Run a dated CPU fit or private fixture with bounded child supervision."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE], default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--mutation", choices=["", "block"], default="")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("replay_passed" if passed else "replay_rejected")
        return 0 if passed else 1
    fixture_mode = args.fixture_output is not None
    if args.mutation and not fixture_mode:
        parser.error("mutations require private fixtures")
    output = (args.fixture_output or args.output).absolute()
    if fixture_mode and output.resolve().is_relative_to((ROOT / "results").resolve()):
        parser.error("private fixture output required")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="carnot8183-validation-") as directory:
        private = Path(directory)
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        specs = manifest(private, candidate)
        atomic_json(raw / "validation_commands.json", specs)
        work = measure(args.root, raw, fixture_mode=fixture_mode, mutation=args.mutation)
        receipts = [dict(name="private_fixture_normal_exit", passed=True)] if fixture_mode else []
        if not fixture_mode:
            for spec in specs["commands"]:
                progress(
                    "before_subprocess_" + spec["name"],
                    len(receipts),
                    len(specs["commands"]) - len(receipts),
                )
                receipts.append(
                    execution.run_check(ROOT, spec, private, raw / "logs", heartbeat_s=20)
                )
                progress(
                    "after_subprocess_" + spec["name"],
                    len(receipts),
                    len(specs["commands"]) - len(receipts),
                )
            progress("before_subprocess_repository_full_suite")
            work["repository_health"] = execution.run_check(
                ROOT, specs["repository_health"], private, raw / "repository_health", heartbeat_s=20
            )
            progress("after_subprocess_repository_full_suite", 1, 0)
            coverage = private / "coverage.json"
            if coverage.is_file():
                saved = raw / "changed_code_coverage.json"
                saved.write_bytes(coverage.read_bytes())
                work["raw_shard_hashes"].append(reference(saved))
            atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = build(work, raw, receipts, fixture=fixture_mode)
        if output.exists():
            (raw / "preserved_historical_primary.json").write_bytes(output.read_bytes())
        with patch.object(execution, "e", sys.modules[__name__]):
            execution.publish(value, output, private, raw, specs["terminal_commands"], fixture_mode)
    return 0
