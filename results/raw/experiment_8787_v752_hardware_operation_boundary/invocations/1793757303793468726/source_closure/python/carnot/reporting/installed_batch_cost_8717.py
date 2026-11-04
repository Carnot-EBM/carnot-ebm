"""REQ-REPORT-8717: compare equal installed work with full process clocks.

The corpus and numerical head were already exposed. Repeating them measures
runtime variation; it cannot create new evidence about semantic generalization.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting import installed_batch_8702 as previous
from carnot.reporting import v747_contract_methods as contract
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = previous.ROOT
NAME, TASK = "experiment_8717_v747_installed_batch_cost", "exp8717-installed-batch-cost"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_installed_batch_cost_8717.py"]
OWNED = [
    f"python/carnot/reporting/installed_batch_cost{s}_8717.py" for s in ["", "_measure", "_runner"]
] + [CLI]
PRODUCER = "results/experiment_8716_v747_installed_import_isolation.json"
PIN = "sha256:7152eb60177c873d646ec02305b63819269e7c4410471f7ea5a00c28e7c77c13"
reference, load, atomic_json, canonical_hash = (
    previous.reference,
    previous.load,
    previous.atomic_json,
    previous.canonical_hash,
)


def method() -> Json:
    """Read the original registration so thresholds cannot drift with observed costs."""
    name, digest, _ = previous.PINS[2]
    return dict(load(dict(path=str(ROOT / name), sha256=digest))["installed_batching_protocol"])


def inputs(root: Path) -> Json:
    """Require original operands, current contract and the independently pinned producer."""
    with patch.object(previous, "authority", contract.authority):
        result = previous.inputs(root)
    checked = previous.authenticate(
        dict(path=str(root / PRODUCER), sha256=PIN),
        dict(
            experiment_id=8716,
            task_id="exp8716-installed-import-isolation",
            required_checks_passed=True,
            flagged_adversarial=False,
            batch_service_ready_score=1,
        ),
    )
    result["references"] += checked["references"]
    result["gate_check_summary"] += checked["failures"]
    if checked["passed"] and result["passed"]:
        try:
            value = checked["value"]
            wrapper = reference(Path(value["terminal_validation_sidecar_path"]))
            side = Path(load(wrapper)["publication"]["sidecar_path"])
            if read_bound_sidecar(root / PRODUCER, side)["report"]["passed"] is not True:
                raise ValueError("terminal_rejected")
            primitive = value["primitive_reference"]
            work = load(primitive)
            seal = work["seal_reference"]
            if load(seal) != {k: v for k, v in work.items() if k != "seal_reference"}:
                raise ValueError("producer_seal")
            protocol = value["complete_cost_protocol"]
            frozen = protocol["frozen_installed_workload"]
            refs = [wrapper, reference(side), primitive, seal] + [
                frozen[k] for k in ["wheel", "extension", "config"]
            ]
            for ref in refs:
                if reference(Path(ref["path"])) != ref:
                    raise ValueError("changed_bytes:" + ref["path"])
            config = load(frozen["config"])
            if (
                protocol["method"] != result["method"]
                or frozen["method"] != result["method"]
                or frozen["requests"] != result["requests"]
                or config["head"] != result["head"]
                or config["extension"]["sha256"] != frozen["extension"]["sha256"]
                or frozen["configuration_template"]
                != dict(head=result["head"], max_batch_size=16, item_budget=1024)
            ):
                raise ValueError("frozen_protocol_drift")
            result.update(workload=frozen)
            result["references"] += refs
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["gate_check_summary"].append(
                contract.gate(
                    root / PRODUCER, "authenticated_frozen_workload", True, str(error), TASK
                )
            )
    result["passed"] = not result["gate_check_summary"]
    return result


def roster(protocol: Json) -> list[Json]:
    """Randomize paired blocks once and retain the registered alternating arm order."""
    blocks = [
        (size, repeat)
        for size in protocol["batch_sizes"]
        for repeat in range(protocol["paired_repeats"])
    ]
    random.Random(protocol["bootstrap_seed"]).shuffle(blocks)
    return [
        dict(
            unit_id=f"B{size}-R{repeat}-{arm}",
            batch=size,
            repeat=repeat,
            arm=arm,
            panels=protocol["panels_per_session"],
            item_evaluations=1024,
            status="censored",
        )
        for size, repeat in blocks
        for arm in (
            protocol["arithmetic_arms"] if repeat % 2 == 0 else protocol["arithmetic_arms"][::-1]
        )
    ]


def aggregate(rows: list[Json]) -> Json:
    """Resample whole paired processes so repeated requests do not shrink uncertainty."""
    complete = len(rows) == 60 and all(r["status"] == "completed" and r["passed"] for r in rows)
    intervals: list[Json] = []
    if complete:
        lookup = {(r["arm"], r["batch"], r["repeat"]): r for r in rows}
        indices = np.random.default_rng(7448675).integers(0, 10, size=(10000, 10))
        for label, slow, fast in [("batching", ("native", 1), ("native", 16))] + [
            (f"native_python_B{b}", ("python", b), ("native", b)) for b in [1, 4, 16]
        ]:
            a = np.asarray([lookup[(*slow, i)]["duration_s"] for i in range(10)])
            b = np.asarray([lookup[(*fast, i)]["duration_s"] for i in range(10)])
            boot = a[indices].sum(axis=1) / b[indices].sum(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            intervals.append(
                dict(
                    contrast=label,
                    estimate=float(a.sum() / b.sum()),
                    lower95=float(low),
                    upper95=float(high),
                    paired_repeats=10,
                    bootstrap_draws=10000,
                    bootstrap_seed=7448675,
                )
            )
    latencies = [
        x
        for r in rows
        if r["arm"] == "native" and r["batch"] == 16
        for x in r.get("steady_item_completion_ms", [r.get("steady_p95_ms", 0)])
    ]
    p95 = float(np.quantile(latencies, 0.95)) if latencies else 0.0
    return dict(
        complete=complete,
        paired_intervals=intervals,
        batching_benefit_score=int(complete and intervals[0]["lower95"] > 1.05 and p95 <= 50),
        same_batch_10x_score=int(complete and intervals[-1]["lower95"] >= 10),
        native_B16_steady_p95_ms=p95,
    )


def build(work: Json, output: Path) -> Json:
    """A complete scientific null remains usable cost evidence when owned checks pass."""
    measured, validation = work["measurement"], work["validation"]
    rows = deepcopy(measured["process_rows"])
    summary = aggregate(rows)
    owned = work["private_control"] or (
        validation["coverage"]["complete"]
        and validation["custody"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    blocked = bool(work["inputs"]["gate_check_summary"])
    ready = bool(
        owned
        and not blocked
        and not work["owned_error"]
        and summary["complete"]
        and not work["private_control"]
    )
    verdict = (
        "disqualified"
        if not owned or work["owned_error"]
        else "blocked"
        if blocked
        else "positive"
        if ready and (summary["batching_benefit_score"] or summary["same_batch_10x_score"])
        else "null"
    )
    scores = dict(
        batch_cost_ready_score=int(ready),
        batching_benefit_score=int(ready and summary["batching_benefit_score"]),
        same_batch_10x_score=int(ready and summary["same_batch_10x_score"]),
    )
    value = dict(
        experiment_id=8717,
        task_id=TASK,
        milestone="2026.11.747",
        run_date="20261102",
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_v747_installed_batch_cost",
        verdict_class=verdict,
        gate_check_summary=work["inputs"]["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0),
        current_language_model_calls=0,
        target_model=work["inputs"].get("target_model"),
        historical_model_provenance=work["inputs"].get("historical_model_provenance", {}),
        rows=rows,
        intended_count=60,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            process_cells=60,
            paired_process_repeats=10,
            item_evaluations=61440,
            completed_item_evaluations=len(measured["parity_rows"]),
            original_exposed_items=64,
            independent_semantic_questions=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="fixed exposed original64 requests; timings only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(owned),
        flagged_adversarial=bool(not owned or work["owned_error"]),
        acceptance_gates=[
            dict(name="qualified_complete_cost", passed=ready),
        ],
        benefit_gates=[
            dict(name="batching_benefit", passed=bool(scores["batching_benefit_score"])),
            dict(name="NFR-01_same_batch_B16", passed=bool(scores["same_batch_10x_score"])),
        ],
        validation_receipts=validation["receipts"] + measured["receipts"],
        owned_coverage=validation["coverage"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=7448675,
        source_artifact_hashes=work["inputs"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"] + validation["raw_shard_hashes"],
        cited_upstream_artifacts=work["inputs"]["references"],
        branch_gate_values=scores,
        **scores,
        process_rows=rows,
        request_rows=measured["request_rows"],
        parity_rows=measured["parity_rows"],
        complete_cost_rows=measured["complete_cost_rows"],
        paired_intervals=summary["paired_intervals"],
        native_B16_steady_p95_ms=summary["native_B16_steady_p95_ms"],
        task_contract_rows=work["inputs"]["contract"]["rows"],
        canonical_tasks_sha256=contract.PIN,
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        owned_error=work["owned_error"],
        private_control=work["private_control"],
        inference_boundary="launch through clean shutdown; setup included; offline environment installation reported separately",
        excluded_costs=["feedback updates", "feedback durability"],
        cost_component_limit="Installed wheel exposes scoring spans. Residual round-trip time includes transport, duplicate feature-digest extraction, protocol validation and wire serialization; those overlapping costs are not separately attributed.",
        physical_device_execution_score=0,
        methodology="Exact Exp8716 wheel and extension, immutable original numerical head and complete text inputs. Sixty fresh Python/native processes; randomized paired blocks, registered AB/BA order;16 panels of64 requests per cell. Recompute features, compare every typed output, retain full launch/setup/request/shutdown clocks.10000 paired process bootstrap draws seed7448675. Native B16/B1 batching and equal-batch native/Python NFR-01 are separate estimands. Fixed exposed inputs establish no learning or independent generalization.",
    )
    value["field_principles"] = {
        k: "Bind exact task, sealed operands and actual complete process work; readiness and benefit are separate."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
