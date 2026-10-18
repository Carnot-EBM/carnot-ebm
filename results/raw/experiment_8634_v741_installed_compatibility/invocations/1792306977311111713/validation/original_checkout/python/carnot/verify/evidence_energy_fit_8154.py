"""REQ-REPORT-8154: authenticate, fit and publish source decision heads.

The current run loads no language model. Only fit and tune targets are opened;
reserved source capture belongs to the next experiment after weights seal.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import evidence_energy_8154 as energy
from carnot.verify import evidence_features_7980 as lexical
from carnot.verify import fit_evidence_capture_8153 as capture

Json = dict[str, Any]
ROOT = capture.ROOT
NAME = "experiment_8154_v705_evidence_energy_fit"
TASK = "exp8154-evidence-energy-fit"
MODULE = "python/carnot/verify/evidence_energy_fit_8154.py"
NUMERIC = "python/carnot/verify/evidence_energy_8154.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_evidence_energy_fit_8154.py"
NUMERIC_TEST = "tests/python/test_evidence_energy_8154.py"
OWNED = [MODULE, NUMERIC, CLI]
RUN_DATE = "20261005"
MODEL_SPECS: list[str] = []
UPSTREAM = "results/experiment_8153_v705_fit_evidence_capture.json"
PIN = "sha256:c4d1be0937d6fe6d5e538e009d8ed82242973d8616569c8369cc0ea7172103ec"
METHOD = "openspec/change-proposals/research-roadmap-v704-preserved-20261005.md"
METHOD_PIN = "sha256:92dca5a4ce979d47a47da2e9622da8e1cdd306a8a45c051f2737c3ee1e984f28"
progress = energy.progress
CONFIG = dict(
    energy.CONFIG,
    seed=70554,
    features=[
        "holistic_logit",
        *lexical.FEATURES,
        "span_logit",
        "quote_validity",
        "quote_source_byte_ratio",
    ],
)


def reference(path: Path) -> Json:
    """Exact bytes bind an input or immutable head, even after a cold restart."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def publication_sidecar(value: Json) -> Path:
    """Use the producer's bound attestation rather than trusting a readiness flag."""
    return Path(
        json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())["publication"][
            "sidecar_path"
        ]
    )


def inputs(root: Path, raw: Path) -> Json:
    """Gate transport and trainability separately before joining matched features."""
    plan: Json = dict(checks=[], refs=[], rows=[], historical_model_provenance={})
    path = root / UPSTREAM

    def require(p: Path, field: str, expected: Any, observed: Any) -> None:
        """Preserve the precise failed operand so an external block is terminal."""
        plan["checks"].append(
            dict(
                check=field,
                upstream=UPSTREAM,
                path=str(p.absolute()),
                hash=sha256_file(p) if p.is_file() else None,
                artifact_field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=expected == observed,
            )
        )
        if expected != observed:
            raise ValueError(field)

    def bind(ref: Json) -> Json:
        """Copy authenticated bytes so replay never relies on a mutable upstream."""
        p = Path(ref["path"])
        require(p, "input_sha256", ref["sha256"], sha256_file(p) if p.is_file() else None)
        target = raw / "inputs" / (ref["sha256"].split(":")[-1] + "-" + p.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
        plan["refs"].append(reference(target))
        return dict(json.loads(target.read_text()))

    progress("before_input_authentication")
    try:
        require(Path(sys.executable), "python_runtime_supported", True, sys.version_info >= (3, 11))
        for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
            require(
                ROOT / ".venv/bin" / tool,
                "runtime_tool_executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        require(path, "upstream_exists", True, path.is_file())
        require(path, "upstream_sha256", PIN, sha256_file(path))
        value = bind(reference(path))
        for field, expected in (
            ("fit_capture_ready_score", 1),
            ("fit_trainable_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ):
            require(path, field, expected, value.get(field))
        require(
            path,
            "terminal_passed",
            True,
            read_bound_sidecar(path, publication_sidecar(value))["report"]["passed"],
        )
        require(ROOT / METHOD, "preserved_v704_sha256", METHOD_PIN, sha256_file(ROOT / METHOD))
        plan["refs"].append(reference(ROOT / METHOD))
        refs = {Path(r["path"]).name: r for r in value["raw_shard_hashes"]}
        calls = bind(refs["primitive_calls.json"])["rows"]
        labels = bind(refs["fit_tune_targets.json"])
        reduced = capture.reduce(calls)["rows"]
        require(path, "primitive_row_reduction", value["rows"], reduced)
        public: Json = {}
        for role in ("fit", "tune"):
            ref = next(
                r
                for r in value["source_artifact_hashes"]
                if r["path"].endswith("/public/" + role + ".json")
            )
            for row in bind(ref)["request_rows"]:
                public[row["family_id"]] = row
        spans = {
            r["unit_id"]: r
            for r in calls
            if r["arm"] == "source_span" and r["condition"] == "original"
        }
        for row in reduced:
            r = dict(row, x=None, y=labels.get(row["unit_id"]))
            if row["status"] == "completed":
                span = spans[row["unit_id"]]
                p = public[row["unit_id"]]
                require(path, "matched_source_bytes", p["source_bytes"], span["source_bytes"])
                require(path, "matched_answer_bytes", p["answer_bytes"], span["answer_bytes"])
                features = lexical.extract(p)
                if features["values"] is None:
                    r.update(status="excluded", exclusion_reason=features["abstention"])
                else:
                    h, s = np.clip(
                        [row["holistic_probability"], row["span_probability"]], 1e-6, 1 - 1e-6
                    )
                    r["x"] = [
                        float(np.log(h / (1 - h))),
                        *features["values"],
                        float(np.log(s / (1 - s))),
                        span["parsed"]["valid_quote"],
                        span["parsed"]["quote_source_byte_ratio"],
                    ]
            plan["rows"].append(r)
        support = {
            role: Counter(
                r["y"] for r in plan["rows"] if r["role"] == role and r["status"] == "completed"
            )
            for role in ("fit", "tune")
        }
        plan["usable_class_support"] = {
            role: {str(y): counts[y] for y in (0, 1)} for role, counts in support.items()
        }
        require(
            path,
            "feature_trainable_score",
            1,
            int(all(counts[y] >= 12 for counts in support.values() for y in (0, 1))),
        )
        plan["historical_model_provenance"] = dict(
            path=str(path),
            sha256=PIN,
            MODEL_SPECS=value["MODEL_SPECS"],
            imported_invocation_counts=value["model_invocation_counts"],
            scope="historical_only_zero_current_calls",
        )
    except (ValueError, OSError, KeyError, StopIteration, TypeError) as error:
        if not plan["checks"] or plan["checks"][-1]["passed"]:
            plan["checks"].append(
                dict(
                    check="input_structure",
                    upstream=UPSTREAM,
                    path=str(path),
                    hash=None,
                    artifact_field="input_structure",
                    op="==",
                    expected="valid_bound_inputs",
                    observed=str(error),
                    passed=False,
                )
            )
    progress("after_input_authentication", len(plan["rows"]), 192 - len(plan["rows"]))
    return plan


def fixture_rows() -> list[Json]:
    """Scripted source signals qualify execution only and earn no science credit."""
    rng = np.random.default_rng(CONFIG["seed"])
    return [
        dict(
            unit_id=f"{role}{i}",
            source_cluster_id=f"{role}{i}",
            role=role,
            x=rng.normal(size=12).tolist(),
            y=i % 2,
            status="completed",
            exclusion_reason=None,
        )
        for role, count in (("fit", 32), ("tune", 16))
        for i in range(count)
    ]


def measure(root: Path, raw: Path, *, fixture: bool = False, mutation: str = "") -> Json:
    """Measure real fitting time and preserve every excluded original source slot."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    plan = (
        dict(checks=[], refs=[], rows=fixture_rows(), historical_model_provenance={})
        if fixture
        else inputs(root, raw)
    )
    if mutation == "block":
        plan["checks"] = [
            dict(
                check="fit_trainable_score",
                upstream=UPSTREAM,
                path=str(root / UPSTREAM),
                hash=None,
                artifact_field="fit_trainable_score",
                op="==",
                expected=1,
                observed=0,
                passed=False,
            )
        ]
    input_path = raw / "matched_features.json"
    atomic_json(input_path, dict(rows=plan["rows"]))
    plan["refs"].append(reference(input_path))
    fitted: Json = dict(heads=[], failures=[], fit_fold_rows=[], training_budget=CONFIG)
    owned_failure = ""
    if all(r["passed"] for r in plan["checks"]):
        try:
            fitted = energy.train(plan["rows"], raw)
        except (ValueError, TimeoutError) as error:
            owned_failure = str(error)
    if not (raw / "frozen_heads.json").exists():
        atomic_json(raw / "frozen_heads.json", fitted)
    result = energy.evaluate(plan["rows"], [h for h in fitted["heads"] if "calibration" in h])
    primitive = raw / "primitive_decisions.json"
    atomic_json(primitive, result)
    work = dict(
        plan=plan,
        fitted=fitted,
        result=result,
        owned_failure=owned_failure,
        raw_shard_hashes=[
            reference(input_path),
            reference(raw / "frozen_heads.json"),
            reference(primitive),
        ],
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
                NUMERIC_TEST,
                "python/carnot/verify/evidence_features_7980.py",
                "python/carnot/verify/fit_evidence_capture_8153.py",
                "python/carnot/verify/evidence_protocol_8124.py",
                "python/carnot/verify/source_alignment.py",
                "python/carnot/experiment_8021_v695_typed_decision_test.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/methods_stream_execution_8111.py",
            ]
        ],
        config_sha256=canonical_hash(CONFIG),
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(plan["rows"]), 0)
    return work


def independent_reduce(rows: list[Json]) -> list[Json]:
    """Recompute scalar headlines with separate arithmetic over primitive units."""
    for row in rows:
        if row["denominator"]:
            y, decision = row["y"], row["action"]
            cost = (
                0.5 if decision == "escalate" else float(5 * y if decision == "accept" else 1 - y)
            )
            if cost != row["numerator"]:
                raise ValueError("primitive_cost_drift")
    summaries = []
    for role in ("fit", "tune"):
        for arm in energy.ARMS:
            selected = [
                r for r in rows if r["role"] == role and r["arm"] == arm and r["denominator"]
            ]
            summary = dict(role=role, arm=arm, n=len(selected))
            for field, key in (
                ("typed_cost", "numerator"),
                ("brier", "brier"),
                ("log_loss", "log_loss"),
                ("coverage", "coverage"),
                ("false_accept", "false_accept"),
            ):
                values = [r[key] for r in selected if r[key] is not None]
                summary[field] = math.fsum(values) / len(values) if values else None
            summaries.append(summary)
    return summaries


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness certifies trained head custody, never independent correctness."""
    plan, fitted = work["plan"], work["fitted"]
    checks = plan["checks"]
    failures = [r for r in checks if not r["passed"]]
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and not work["owned_failure"]
        and not fitted["failures"]
    )
    klass = (
        "disqualified"
        if not owned
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
    )
    measured = energy.evaluate(plan["rows"], [h for h in fitted["heads"] if "calibration" in h])
    independent = independent_reduce(measured["rows"])
    reduced_correctly = all(
        a[k] == b[k] if a[k] is None or isinstance(a[k], (str, int)) else abs(a[k] - b[k]) <= 1e-12
        for a, b in zip(independent, measured["development_metrics"], strict=True)
        for k in a
    )
    ready = int(
        owned
        and reduced_correctly
        and not failures
        and len(fitted["heads"]) == 6
        and bool(measured["energy_logistic_parity_rows"])
        and all(r["passed"] for r in measured["energy_logistic_parity_rows"])
    )
    statuses = Counter(r["status"] if r["y"] in (0, 1) else "excluded" for r in plan["rows"])
    value = dict(
        **measured,
        experiment_id=8154,
        task_id=TASK,
        milestone="2026.10.705",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + klass
        + "_"
        + (
            failures[0]["check"]
            if klass == "blocked"
            else "owned_validation"
            if klass == "disqualified"
            else "evidence_energy_fit"
        ),
        verdict_class=klass,
        energy_fit_ready_score=ready,
        required_checks_passed=bool(owned),
        flagged_adversarial=False,
        verifier_is_oracle=fixture,
        fixture_protocol_only=fixture,
        claim_scope="frozen calibrated source decision heads; fit/tune development only; reserved H1 untested",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        preconditions_checked=checks,
        gate_check_summary=checks,
        MODEL_SPECS=[],
        trained_head_specs=[
            dict(
                arm=h["arm"],
                parameter_count=len(h["weights"]),
                calibration_parameter_count=2,
                ridge=h["ridge"],
                manifest=reference(raw / "frozen_heads.json"),
            )
            for h in fitted["heads"]
        ],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        historical_model_provenance=plan["historical_model_provenance"],
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if fitted["heads"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_mode="fixture" if fixture else "cached_candidates_cpu",
        intended_count=len(plan["rows"]) if fixture else 192,
        eligible_count=statuses["completed"],
        independent_count=statuses["completed"],
        completed_count=statuses["completed"],
        excluded_count=statuses["excluded"],
        censored_count=statuses["censored"],
        failed_count=statuses["failed"],
        sample_size_budget=dict(
            intended=192,
            fit=128,
            tune=64,
            independent_unit="source_cluster",
            repeats_create_no_new_sources=True,
        ),
        random_seed=CONFIG["seed"],
        source_artifact_hashes=plan["refs"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        config_sha256=work["config_sha256"],
        phase_spans=work["phase_spans"],
        duration_s=work["duration_s"],
        frozen_head_manifest=reference(raw / "frozen_heads.json"),
        fit_fold_rows=fitted["fit_fold_rows"],
        fit_failures=fitted["failures"],
        training_budget=fitted["training_budget"],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        measurement_reference=reference(raw / "measurement.json"),
        repository_health=work.get("repository_health"),
        acceptance_gates=dict(
            upstream="capture_ready=1 and fit_trainable=1",
            owned="normal scoped validation and 100% changed statements",
            parity="probability error <=1e-10 and identical typed actions",
            readiness="six converged calibrated heads; no fit failures",
        ),
        methodology_note="Immutable V704 geometry; four source folds; fit-only ridge selection; tune-only affine calibration. No reserved targets. Energy/logistic identity adds no correctness evidence.",
        independent_headline_reduction=independent,
        independent_reduction_passed=reduced_correctly,
        field_principles=dict(
            verdict="External blocks terminate; owned failures disqualify.",
            independence="Only original source clusters count; exposed development earns no independent generalization.",
            targets="Original human hallucination labels only; quote validity is an input.",
            inference="Historical provenance is separate from zero current LLM calls.",
            custody="Frozen geometry, weights, calibrators and cost rules bind to exact bytes.",
            duration="Measured time is never padded.",
        ),
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild headlines from sealed primitive decisions and reject changed bytes."""
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
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        frozen = json.loads(Path(value["frozen_head_manifest"]["path"]).read_text())
        if frozen != work["fitted"] or work["result"] != energy.evaluate(
            work["plan"]["rows"], [h for h in frozen["heads"] if "calibration" in h]
        ):
            return False
        expected = build(
            work,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
            fixture=value["fixture_protocol_only"],
        )
        return bool(expected == value)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze owned argv before measurement and keep repository health separate."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = execution.manifest(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["argv"].append(NUMERIC_TEST)
    specs["commands"][0]["deadline_s"] = 600
    specs["commands"][1]["name"] = "qualified_components_and_E2E015_016"
    specs["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7868_v683_intervention_protocol.py",
        "tests/python/test_evidence_protocol_8124.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    specs["commands"][-1]["argv"].append(NUMERIC_TEST)
    return specs


def main(argv: list[str] | None = None) -> int:
    """Run the dated CPU fit or private fixtures with bounded child supervision."""
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
    fixture = args.fixture_output is not None
    if args.mutation and not fixture:
        parser.error("mutations require private fixture output")
    output = (args.fixture_output or args.output).absolute()
    if fixture and output.is_relative_to(ROOT / "results"):
        parser.error("private fixture output required")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="carnot8154-validation-") as directory:
        private = Path(directory)
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        specs = manifest(private, candidate)
        atomic_json(raw / "validation_commands.json", specs)
        work = measure(args.root, raw, fixture=fixture, mutation=args.mutation)
        receipts = [dict(name="private_fixture_normal_exit", passed=True)] if fixture else []
        if not fixture:
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
            coverage_path = private / "coverage.json"
            if coverage_path.is_file():
                saved_coverage = raw / "changed_code_coverage.json"
                saved_coverage.write_bytes(coverage_path.read_bytes())
                work["raw_shard_hashes"].append(reference(saved_coverage))
            atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = build(work, raw, receipts, fixture=fixture)
        if output.exists():
            (raw / "preserved_historical_primary.json").write_bytes(output.read_bytes())
        with patch.object(execution, "e", sys.modules[__name__]):
            execution.publish(value, output, private, raw, specs["terminal_commands"], fixture)
    return 0
