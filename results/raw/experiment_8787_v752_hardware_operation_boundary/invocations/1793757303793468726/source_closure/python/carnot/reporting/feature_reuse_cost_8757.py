"""REQ-REPORT-8757: equal installed work needs complete paired process costs.

Qualification proves parity. This separate producer measures runtime variation
without treating exposed items or requests as independent scientific cases.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np

from carnot.reporting import installed_batch_8702 as prior
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK = "experiment_8757_v750_feature_reuse_cost", "exp8757-feature-reuse-cost"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_feature_reuse_cost_8757.py"]
OWNED = [
    f"python/carnot/reporting/feature_reuse_cost{s}_8757.py" for s in ["", "_measure", "_runner"]
] + [CLI]
PRODUCER = "results/experiment_8756_v750_feature_reuse_qualification.json"
PIN = "sha256:444580deef2cc1733553659c21b418dd22b94873a2d182fd246bc62b1e185d24"
PROTOCOL = dict(
    path=str(ROOT / "openspec/change-proposals/v750-feature-reuse-8756-protocol.json"),
    sha256="sha256:ca0b73760a21e8a90da8db6ad78f40f197172dc026458da8d43a8e9adc9c594a",
)
SEED = 7508757
reference, load, atomic_json, canonical_hash = (
    prior.reference,
    prior.load,
    prior.atomic_json,
    prior.canonical_hash,
)


def method() -> Json:
    """Read fixed thresholds from the qualification's premeasurement registration."""
    return dict(load(PROTOCOL)["original_cost_protocol"])


def inputs(root: Path) -> Json:
    """Bind the new qualified producer and every retained installed operand."""
    ref = dict(path=str(root / PRODUCER), sha256=PIN)
    checked = prior.authenticate(
        ref,
        dict(
            experiment_id=8756,
            task_id="exp8756-feature-reuse-qualification",
            feature_reuse_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
        ),
    )
    result: Json = dict(
        passed=False, references=checked["references"], gate_check_summary=checked["failures"]
    )
    if not checked["passed"]:
        return result
    try:
        value = load(ref)
        wrapper = reference(Path(value["terminal_validation_sidecar_path"]))
        side = Path(load(wrapper)["publication"]["sidecar_path"])
        assert read_bound_sidecar(root / PRODUCER, side)["report"]["passed"] is True
        work = load(value["primitive_reference"])
        assert load(work["seal_reference"]) == {
            k: v for k, v in work.items() if k != "seal_reference"
        }
        frozen = value["frozen_8744_workload"]
        refs = [
            wrapper,
            reference(side),
            value["primitive_reference"],
            work["seal_reference"],
            PROTOCOL,
        ]
        refs += [frozen[k] for k in ["wheel", "extension", "head", "native_build_reference"]]
        refs += frozen["source_references"]
        for operand in refs:
            if reference(Path(operand["path"])) != operand:
                raise ValueError("operand_hash:" + operand["path"])
        assert canonical_hash(frozen["requests"]) == method()["request_panel_sha256"]
        baseline, reuse = (frozen["configuration_templates"][m] for m in ["baseline", "reuse"])
        assert dict(baseline, reuse_request_features=True) == reuse
        assert baseline["head"] == load(frozen["head"])
        assert baseline["item_budget"] == 1024 and baseline["max_batch_size"] == 16
        assert baseline["reuse_request_features"] is False
        result.update(
            passed=True,
            workload=dict(frozen, configuration_template=baseline),
            head=baseline["head"],
            requests=frozen["requests"],
            repository_health_reference=value["repository_health_reference"],
        )
        result["references"] += refs
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        result["gate_check_summary"].append(
            dict(
                upstream="exp8756-feature-reuse-qualification",
                **ref,
                check="sealed_qualified_workload",
                field="frozen_8744_workload",
                operator="==",
                expected="authenticated original workload",
                observed=str(error),
                observed_state="rejected_producer",
            )
        )
    return result


def roster() -> list[Json]:
    """Reverse each four-arm order on alternate repeats and randomize complete blocks."""
    repeats = list(range(10))
    random.Random(SEED).shuffle(repeats)
    arms = [
        ("python", "baseline"),
        ("native", "baseline"),
        ("native", "reuse"),
        ("python", "reuse"),
    ]
    return [
        dict(
            unit_id=f"R{repeat}-{arm}-{mode}",
            repeat=repeat,
            arm=arm,
            mode=mode,
            batch=16,
            panels=16,
            item_evaluations=1024,
            status="censored",
            passed=False,
        )
        for repeat in repeats
        for arm, mode in (arms if repeat % 2 == 0 else arms[::-1])
    ]


def aggregate(rows: list[Json]) -> Json:
    """Bootstrap paired process blocks; requests inside a process add no repeats."""
    complete = len(rows) == 40 and all(r["status"] == "completed" and r["passed"] for r in rows)
    intervals, p95 = [], {}
    if complete:
        lookup = {(r["arm"], r["mode"], r["repeat"]): r for r in rows}
        assert len(lookup) == 40 and set(lookup) == {
            (r["arm"], r["mode"], r["repeat"]) for r in roster()
        }
        indices = np.random.default_rng(SEED).integers(0, 10, size=(10000, 10))
        for label, slow, fast in [
            ("native_reuse", ("native", "baseline"), ("native", "reuse")),
            ("python_reuse", ("python", "baseline"), ("python", "reuse")),
            ("optimized_native_python", ("python", "reuse"), ("native", "reuse")),
        ]:
            a, b = [
                np.asarray([lookup[(*arm, i)]["duration_s"] for i in range(10)])
                for arm in [slow, fast]
            ]
            assert np.isfinite(a).all() and np.isfinite(b).all() and min(*a, *b) > 0
            draws = a[indices].sum(axis=1) / b[indices].sum(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            intervals.append(
                dict(
                    contrast=label,
                    estimate=float(a.sum() / b.sum()),
                    lower95=float(low),
                    upper95=float(high),
                    paired_repeats=10,
                    bootstrap_draws=10000,
                    bootstrap_seed=SEED,
                )
            )
        for mode in ["baseline", "reuse"]:
            p95[mode] = float(
                np.quantile(
                    [
                        v
                        for r in rows
                        if r["arm"] == "native" and r["mode"] == mode
                        for v in r["steady_item_completion_ms"]
                    ],
                    0.95,
                )
            )
    benefit = bool(
        complete
        and intervals[0]["lower95"] > 1.05
        and p95["reuse"] <= 50
        and p95["reuse"] <= 1.05 * p95["baseline"]
    )
    return dict(
        complete=complete,
        paired_process_intervals=intervals,
        native_steady_p95_ms=p95,
        feature_reuse_benefit_score=int(benefit),
        same_batch_10x_score=int(complete and intervals[2]["lower95"] >= 10),
    )


def build(work: Json, output: Path) -> Json:
    """Execution gates control readiness while measured scientific nulls remain valid."""
    measured, validation = work["measurement"], work["validation"]
    summary = aggregate(measured["process_rows"])
    owned = (
        validation["coverage"]["complete"]
        and validation["custody"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    failures = work["inputs"]["gate_check_summary"] + measured.get("gate_check_summary", [])
    ready = bool(owned and not failures and not work["owned_error"] and summary["complete"])
    verdict = (
        "disqualified"
        if work["owned_error"] or (not work["private_control"] and not owned)
        else "blocked"
        if failures
        else "positive"
        if ready and summary["feature_reuse_benefit_score"]
        else "null"
    )
    rows = measured["process_rows"]
    value: Json = dict(
        experiment_id=8757,
        task_id=TASK,
        milestone="2026.11.750",
        run_date="20261103",
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_feature_reuse_cost",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0),
        numerical_training_calls=0,
        rows=rows,
        intended_count=40,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        independent_process_repeat_count=10 if summary["complete"] else 0,
        sample_size_budget=dict(
            process_cells=40,
            paired_process_repeats=10,
            item_evaluations=40960,
            original_exposed_items=64,
            independent_semantic_questions=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="original64 exposed requests; process timing only",
        independent_generalization_score=0,
        required_checks_passed=bool(owned and not work["owned_error"]),
        flagged_adversarial=bool(
            work.get("adversarial_findings")
            or work["owned_error"]
            or (not owned and not work["private_control"])
        ),
        acceptance_gates=[dict(name="complete_installed_cost_execution", passed=ready)],
        installed_cost_ready_score=int(ready),
        feature_reuse_benefit_score=int(ready and summary["feature_reuse_benefit_score"]),
        same_batch_10x_score=int(ready and summary["same_batch_10x_score"]),
        scientific_promotion_gates=[
            dict(name=n, passed=bool(ready and summary[n]))
            for n in ["feature_reuse_benefit_score", "same_batch_10x_score"]
        ],
        process_cost_rows=rows,
        request_latency_rows=measured["request_rows"],
        paired_process_intervals=summary["paired_process_intervals"],
        native_steady_p95_ms=summary["native_steady_p95_ms"],
        admission=measured.get("admission"),
        probe_rows=measured.get("probe_rows", []),
        validation_receipts=validation["receipts"] + measured["receipts"],
        owned_coverage=validation["coverage"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=SEED,
        source_artifact_hashes=work["inputs"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"] + validation["raw_shard_hashes"],
        cited_upstream_artifacts=work["inputs"]["references"],
        repository_health_reference=dict(
            historical=work["inputs"].get("repository_health_reference", {}),
            current_diagnostic=work["preconditions"].get("repository_health_reference"),
            proves_complete_full_suite_pass=False,
        ),
        registered_method=method(),
        scoped_validation_scope=dict(
            test_files=TESTS, e2e=["E2E-003", "E2E-021"], repository_health_is_dependency=False
        ),
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        owned_error=work["owned_error"],
        adversarial_findings=work.get("adversarial_findings", []),
        inference_boundary="parent launch through clean shutdown; environment installation separate; feedback durability excluded",
        methodology="Exact qualified installed wheel and extension. Four arms, B16, ten paired complete-process repeats;16 panels of64 exposed items. No profiler or extraction counter. Canonical scientific request bytes identical across arms; authenticated configuration hashes and current deadlines are envelope fields. Native reuse/original and optimized native/Python are separate contrasts.10000 whole-paired-process draws. Steady p95 is separate from complete-process throughput. No historical child fraction extrapolation, learning, natural generalization or durability claim.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind one actual producer to unchanged scientific operands, complete process evidence and independent execution gates.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
