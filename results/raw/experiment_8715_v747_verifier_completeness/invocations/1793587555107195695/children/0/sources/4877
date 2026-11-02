"""REQ-REPORT-8466: sealed engineering costs cannot establish generalization.

The prior admission and publication workflows remain unchanged. Narrow adapters
bind them to this task while original sealed producer bytes retain their meaning.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import journal_equivalence_8465 as prior
from carnot.reporting.current_work_receipt import atomic_json, ZERO_INVOCATION_COUNTS
from carnot.reporting.v709_execution import child
from carnot.verify import journal_service_cost_8466 as k
from carnot.verify import journal_equivalence_8465 as stores

Json = dict[str, Any]
ROOT = prior.ROOT
NAME, TASK = "experiment_8466_v728_journal_service_cost", "exp8466-journal-service-cost"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_journal_service_cost_8466.py"
OWNED = [
    "python/carnot/verify/journal_service_cost_8466.py",
    "python/carnot/reporting/journal_service_cost_8466.py",
    "python/carnot/reporting/journal_service_cost_runner_8466.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8465_v728_journal_equivalence.json"
UPSTREAM_PIN = "sha256:6becdfa9334f6f6ab5b1695eb025ce0cb09ff09f838f0e6c415edb97c580a086"
reference, read, checksum, REQUIRED, progress = (
    prior.reference,
    prior.read,
    prior.checksum,
    prior.REQUIRED,
    k.progress,
)
ORIGINAL_DEPENDENCIES = prior.dependencies


def dependencies(root: Path, raw: Path, private: Path) -> Json:
    """Require the exact qualified journal producer and independently sealed service tape."""
    value = read(dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN))
    if value.get("journal_ready_score") != 1 or not value["required_checks_passed"]:
        raise ValueError("journal_ready_score")
    deps = ORIGINAL_DEPENDENCIES(root, raw, private)
    for ref in value["code_config_hashes"] + value["source_artifact_hashes"]:
        if prior.upstream.shared.sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("qualified_store_bytes")
    deps["refs"] += [
        reference(root / UPSTREAM),
        value["work_reference"],
        *value["code_config_hashes"],
    ]
    return dict(deps)


def worker(frozen_path: Path, directory: Path, arm: str, barrier: str = "none") -> int:
    """The real child saves primitive clocks only after all durable work and restart finish."""
    frozen = json.loads(frozen_path.read_bytes())
    row = k.run_repeat(frozen, directory, arm)
    row["frozen_reference"] = reference(frozen_path)
    row["store_references"] = [
        reference(p) for p in sorted((directory / "store").iterdir()) if p.is_file()
    ]
    atomic_json(directory / "primitive.json", row)
    return 0


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Reuse authenticated CPU admission before timing24 restored arms within1800 seconds."""
    with (
        patch.multiple(prior, TASK=TASK, OWNED=OWNED, TEST=TEST, SLOTS=[]),
        patch.object(prior, "dependencies", dependencies),
    ):
        work = prior.measure(root, raw, private)
    for gate in work["gate_check_summary"]:
        if gate["check"] == "sealed_8464_service_tape":
            gate.update(
                check="sealed_8465_store_and_8464_tape",
                field="journal_ready_score_and_sealed_operands",
                upstream="exp8465-journal-equivalence",
                path=str(root / UPSTREAM),
                sha256=prior.upstream.shared.sha256_file(root / UPSTREAM)
                if (root / UPSTREAM).is_file()
                else None,
                observed=gate["observed"] if (root / UPSTREAM).is_file() else None,
            )
    work["preconditions_checked"]["measurement_cap_s"] = 1800
    work.update(run_date="20261013", warmups=[], pairs=[], cells=[])
    start = time.monotonic_ns()
    for cell_index, cell in enumerate(k.CELLS):
        for repeat in range(6):
            arms = ("snapshot", "journal") if repeat % 2 == 0 else ("journal", "snapshot")
            for arm in arms:
                unit = f"cell{cell_index}-repeat{repeat}-{arm}"
                row: Json = dict(
                    unit_id=unit,
                    family="journal8466_engineering_fixture",
                    cell=cell,
                    repeat=repeat,
                    arm=arm,
                    status="censored",
                    absolute_metric=None,
                    missing_reason="required_operands_missing",
                )
                if not work["gate_check_summary"]:
                    remaining = 1800 - (time.monotonic_ns() - start) / 1e9
                    if remaining <= 0:
                        row["missing_reason"] = "measurement_deadline"
                    else:
                        progress(
                            "benchmark_before_" + unit, len(work["rows"]), 24 - len(work["rows"])
                        )
                        try:
                            directory = raw / unit
                            frozen = stores.freeze(work["bundle"], cell, repeat)
                            frozen_path = directory / "frozen.json"
                            atomic_json(frozen_path, frozen)
                            receipt = child(
                                unit,
                                [
                                    sys.executable,
                                    "-u",
                                    str(ROOT / CLI),
                                    "--worker",
                                    str(frozen_path),
                                    "--store",
                                    str(directory),
                                    "--arm",
                                    arm,
                                ],
                                directory / "logs",
                                deadline=remaining,
                            )
                            if not receipt["passed"]:
                                raise ValueError("measurement_child_failed")
                            ref = reference(directory / "primitive.json")
                            primitive = read(ref)
                            metrics = k.reduce(frozen, primitive)
                            row.update(
                                status="completed",
                                absolute_metric=metrics["wall_s"],
                                missing_reason=None,
                                metrics=metrics,
                                primitive_reference=ref,
                                receipt=receipt,
                            )
                            work["primitives"].append(ref)
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            work["owned_failures"][unit] = str(error)
                            row.update(status="failed", missing_reason=str(error))
                        progress(
                            "benchmark_after_" + unit, len(work["rows"]) + 1, 23 - len(work["rows"])
                        )
                work["rows"].append(row)
    work["code_config_hashes"] = [
        reference(ROOT / p) for p in [*OWNED, TEST, *prior.upstream.shared.GUARDS]
    ]
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"].append(
        dict(phase="complete_service_pairs", start_ns=start, end_ns=work["ended_monotonic_ns"])
    )
    atomic_json(raw / "measurement.json", work)
    return dict(work)


def coverage_ready(report: Json) -> bool:
    """Require actual coverage report bytes for precisely this task's new statements."""
    with patch.object(prior, "OWNED", OWNED):
        return bool(prior.coverage_ready(report))


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Qualification requires complete equal work; an observed slowdown remains a ready null."""
    rows, cells, pairs = work["rows"], [], []
    complete = len(rows) == 24 and all(r["status"] == "completed" for r in rows)
    invalid_primitives = False
    for cell in k.CELLS:
        blocks = []
        for repeat in range(6):
            arms = [row for row in rows if row["cell"] == cell and row["repeat"] == repeat]
            if len(arms) == 2 and all(row["status"] == "completed" for row in arms):
                try:
                    indexed = {row["arm"]: read(row["primitive_reference"]) for row in arms}
                    frozen = read(indexed["snapshot"]["frozen_reference"])
                    paired = dict(
                        k.pair(frozen, indexed["snapshot"], indexed["journal"]),
                        cell=cell,
                        repeat=repeat,
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    invalid_primitives = True
                    continue
                if repeat:
                    blocks.append(paired)
                    pairs.append(paired)
        stats = (
            k.interval(blocks)
            if len(blocks) == 5 and complete
            else dict(pooled_speedup=None, speedup_ci95=None)
        )
        p95 = (
            sum(p["journal"]["p95_s"] for p in blocks) / sum(p["snapshot"]["p95_s"] for p in blocks)
            if len(blocks) == 5 and complete
            else None
        )
        cells.append(
            dict(cell=cell, completed_pairs=len(blocks), intended_pairs=5, p95_ratio=p95, **stats)
        )
    complete = complete and not invalid_primitives
    if not complete:
        for result in cells:
            result.update(pooled_speedup=None, speedup_ci95=None, p95_ratio=None)
    passed = bool(receipts) and all(c["passed"] for c in receipts)
    required = REQUIRED <= {c.get("name") for c in receipts if c["passed"]}
    covered = coverage_ready(work.get("owned_coverage", {}))
    flags = work.get("adversarial_findings", [])
    qualified = (
        complete and passed and required and covered and not flags and not work["owned_failures"]
    )
    primary = cells[0]
    benefit = bool(qualified and primary["speedup_ci95"][0] > 1 and primary["p95_ratio"] <= 1.05)
    tenfold = bool(qualified and primary["speedup_ci95"][0] >= 10)
    verdict = (
        "disqualified"
        if not passed or flags or work["owned_failures"]
        else "blocked"
        if work["gate_check_summary"]
        else "circular_positive"
        if benefit
        else "null"
        if qualified
        else "disqualified"
    )
    value: Json = dict(
        experiment_id=8466,
        task_id=TASK,
        milestone="2026.10.728",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_journal_service_cost",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        reader_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=24,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            cells=2,
            warmup_pairs_per_cell=1,
            measured_pairs_per_cell=5,
            bootstrap_draws=10000,
            independent_science_questions=0,
        ),
        verifier_is_oracle=True,
        exposure_scope="retrospective_cached_same_host_engineering_fixture",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed and required,
        flagged_adversarial=bool(flags) or not passed,
        acceptance_gates=dict(
            complete_service=complete,
            exact_accounting=complete,
            authentic_operands=not work["gate_check_summary"],
            owned_coverage=covered,
            required_validation=passed and required,
            no_unresolved_findings=not flags,
            primary_speed_benefit=benefit,
            tenfold_host_service=tenfold,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=flags,
        preconditions_checked=work["preconditions_checked"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=k.SEED,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["primitives"] + work.get("coverage_shards", []),
        cited_upstream_artifacts=[
            dict(
                ref,
                imported_fields="original producer authority, sealed five-input tape, head, labels, qualified store code and acceptance gates",
            )
            for ref in work["refs"]
        ],
        service_cost_ready_score=int(qualified and not work["gate_check_summary"]),
        journal_ready_score=int(qualified and not work["gate_check_summary"]),
        journal_speed_benefit_score=int(benefit),
        tenfold_service_score=int(tenfold),
        paired_repeat_rows=pairs,
        cell_results=cells,
        speedup_ci95=primary["speedup_ci95"],
        pooled_speedup=primary["pooled_speedup"],
        p95_ratio=primary["p95_ratio"],
        restart_cost=[
            dict(
                unit_id=r["unit_id"],
                restart_replay_s=r.get("metrics", {}).get("restart_replay_s"),
                missing_reason=r["missing_reason"],
            )
            for r in rows
        ],
        accounting_equality=True if complete else None,
        owned_coverage=work.get("owned_coverage", {}),
        work_reference=reference(Path(work["raw"]) / "measurement.json"),
        prior_attempts=work.get("prior_attempts", []),
        closed_exposed_utility=dict(H1=-0.00390625, H2=0, reopened=False),
        preserved_history=dict(
            V727_methods_ready_score=0,
            V727_science_executed=False,
            V726_admission_learning="null",
            V726_kernel_percent=0.093,
            V726_persistence_percent=96.44,
        ),
        NFR_01_status="open_requires_matched_Rust_versus_Python",
        shared_model_acquisition_performed=False,
        service_boundary="cached-feature local verifier service; cold numerical reconstruction, durable startup, all requests and tail checkpoint; no current Qwen service",
        bytes_scope="actual Linux process wchar minus flushed progress bytes; includes overwritten metadata; kernel_write_bytes separate",
        memory_scope="fresh arm process peak RSS through complete service, before restart; retained files grow monotonically so final store size equals peak disk",
        p95_scope="journal/snapshot ratio of mean repeat p95; batch members each receive their complete transaction latency",
        durability_scope="qualified actual fsync and process-exit recovery; physical power loss is not certified",
        methodology_note="Five whole paired repeats per cell are conditional same-host engineering evidence only. Each arm clears cached prefixes, charges cold reconstruction and restores initial state. All checkpoint and feedback-tail work is inside complete clocks. No favorable partial ratio is reported.",
    )
    value["field_principles"] = {
        key: "Bind complete durable work to immutable operands; engineering repeats do not establish generalization."
        for key in value
    }
    value["field_principles"].update(
        journal_ready_score="Compatibility scalar for the existing private workflow; equals service_cost_ready_score and makes no additional qualification claim.",
        service_cost_ready_score="Complete accounting and all validation gates are required, regardless of speed.",
        speedup_ci95="Resample paired repeat indices10000 times, using pooled complete walls.",
        restart_cost="Fresh-cache replay is separate from complete run startup and never hidden.",
        tenfold_service_score="A host journal gain cannot close matched Rust/Python NFR-01.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh reduction checks independent operands, retained disk bytes and exact aggregates."""
    try:
        value = json.loads(path.read_bytes())
        if value["experiment_id"] != 8466 or value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        if len(work["rows"]) != 24:
            return False
        if (
            work["authority"]["activated"]
            and prior.upstream.authority.tasks_digest(work["authority"]["tasks"])
            != prior.upstream.authority.TASKS_PIN
        ):
            return False
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8466-replay-") as folder:
            private = Path(folder)
            if work["bundle"]:
                deps = dependencies(Path(work["root"]), private, private)
                if stores.encode(deps["bundle"]) != stores.encode(work["bundle"]) or stores.encode(
                    deps["workload"]
                ) != stores.encode(work["workload"]):
                    return False
            for ref in [
                *value["source_artifact_hashes"],
                *value["code_config_hashes"],
                *value["raw_shard_hashes"],
            ]:
                if prior.upstream.shared.sha256_file(Path(ref["path"])) != ref["sha256"]:
                    return False
            roster = [
                (cell, repeat, arm)
                for cell in k.CELLS
                for repeat in range(6)
                for arm in (("snapshot", "journal") if repeat % 2 == 0 else ("journal", "snapshot"))
            ]
            for index, (row, slot) in enumerate(zip(work["rows"], roster, strict=True)):
                progress("cold_primitive", index, 24 - index)
                if (row["cell"], row["repeat"], row["arm"]) != slot:
                    return False
                if row["status"] != "completed":
                    continue
                primitive = read(row["primitive_reference"])
                frozen = read(primitive["frozen_reference"])
                if stores.encode(frozen) != stores.encode(
                    stores.freeze(work["bundle"], row["cell"], row["repeat"])
                ):
                    return False
                if stores.encode(row["metrics"]) != stores.encode(k.reduce(frozen, primitive)):
                    return False
                for ref in primitive["store_references"]:
                    if prior.upstream.shared.sha256_file(Path(ref["path"])) != ref["sha256"]:
                        return False
                copy = private / str(index)
                shutil.copytree(Path(primitive["directory"]) / "store", copy)
                store = (stores.Snapshot if row["arm"] == "snapshot" else stores.Journal)(
                    copy, frozen
                )
                if stores.encode(store.read()) != stores.encode(primitive["final"]):
                    return False
        return bool(
            stores.encode(value) == stores.encode(build(work, value["validation_receipts"], path))
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
