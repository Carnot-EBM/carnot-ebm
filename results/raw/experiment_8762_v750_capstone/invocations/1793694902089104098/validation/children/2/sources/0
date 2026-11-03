"""REQ-REPORT-8762: close V750 scopes using the retained reconciliation driver."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from carnot.reporting import v749_capstone as previous_capstone
from carnot.reporting import v750_archive_bound as binding
from carnot.reporting import v746_capstone_inputs as previous
from carnot.reporting import v747_capstone_inputs as inputs
from carnot.reporting import v747_capstone_measure as measurement
from carnot.reporting import v747_capstone_report as reporting
from carnot.reporting import v746_capstone as driver
from carnot.reporting.v748_collection_contract import namespace
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v750_capstone_reduce import reduce

Json = dict[str, Any]
e, m = namespace(inputs, "8762_inputs"), namespace(measurement, "8762_measure")
report, runtime = namespace(reporting, "8762_report"), namespace(driver, "8762_driver")
e.NAME, e.TASK, e.MILESTONE = "experiment_8762_v750_capstone", "exp8762-capstone", "2026.11.750"
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_v750_capstone_8762.py"]
e.OWNED = [
    "python/carnot/reporting/v750_capstone.py",
    "python/carnot/reporting/v750_capstone_reduce.py",
    e.CLI,
]
e.DESIGN, e.contract_source = binding.DESIGN, binding
PINS = {
    8749: "d007049f2a219af6248c2d6120b47a87e13fd7bc3cf2c24a26f09ca33c301c08",
    8750: "193d96fe2e805ac2d9ccfdb9228fe31f5780735287c20e19a4fa5c9c190c698d",
    8751: "f730f1da1325143b48aba884352c6346553109bfd4806bfd214b8d22c51dfe9e",
    8754: "0b38625f0f02cab300b597e53f1cb46f984f225cd37ae985e67bf0371fbe9007",
    8755: "55677e18cee63db095dc16c3c3154f8a72099ba06348f029cfc175c64f083490",
    8756: "444580deef2cc1733553659c21b418dd22b94873a2d182fd246bc62b1e185d24",
    8757: "668218f9dffdaafa2a886789af61b26b9ab2f9743f56cf05d2092e610a144258",
    8758: "8e0c904ab7ff9859d002293e05fc2aa3957ca0c0b8379019afe125d448431ac3",
    8759: "fb07a0aecf1631fe10f9b92eb915ff6c70de814aa6066b7f8d7af9cdfe754938",
    8760: "bae0676342b3f95be81e168a7dd56448d436bbb62e96aaf29390b6ba3e3c5903",
    8761: "f0f3e7dd142b277b467ed16d37181501f414687c5a1402bd487960473f80f76c",
}
ARCHIVE = "openspec/change-proposals/research-roadmap-v750-preserved-20261103.md"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose phase boundaries without padding measured execution time."""
    print(f"[exp8762] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path) -> Json:
    """Bind complete objects even when the upstream archive producer failed checks."""
    result = binding.authority(root)
    table, tasks = parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)
    return dict(result, tasks=tasks, visible_table=table, canonical_tasks_sha256=binding.PIN)


def controls() -> list[Json]:
    """Run full-object controls only in private scratch, preserving active queue bytes."""
    tasks = authority(e.ROOT)["tasks"]
    rows = []
    for i, task in enumerate(tasks):
        for field in task:
            changed = deepcopy(tasks)
            changed[i][field] = "forged"
            rows.append(
                dict(
                    task_id=task["id"],
                    field=field,
                    passed=binding.tasks_digest(changed) != binding.PIN,
                )
            )
        progress("full_object_controls", i + 1, len(tasks) - i - 1)
    with TemporaryDirectory(prefix="capstone8762-controls-", dir="/var/tmp"):
        methods = namespace(binding.prior, "8762_controls")
        methods.ROOT, methods.PIN = e.ROOT, binding.PIN
        methods.MILESTONE, methods.VERSION, methods.FIRST_ID = e.MILESTONE, "750", 8749
        methods.authority = binding.authority
        rows += methods.mutations()
    return rows


def methods() -> Json:
    """Bind original methods, unchanged V749 quarantines and the preserved V750 plan."""
    value = previous_capstone.methods()
    value["references"] += [e.reference(e.ROOT / ARCHIVE)]
    value["gate_check_summary"] = [
        r for r in value["gate_check_summary"] if r["check"] != "exclusion_manifest"
    ]
    manifest = e.ROOT / "ops/exclusion_manifest.yaml"
    retired = binding.yaml.safe_load(manifest.read_bytes())
    entries = retired["retired"] + retired.get("retired_experiments", [])
    value["gate_check_summary"].append(
        dict(
            upstream=e.TASK,
            **e.reference(manifest),
            check="not_retired",
            field="retired_experiment_ids",
            operator="not_contains",
            expected=8762,
            observed=[str(r.get("experiment_id")) for r in entries if isinstance(r, dict)],
            passed=not any(
                str(r.get("experiment_id")) in {"8762", e.TASK}
                for r in entries
                if isinstance(r, dict)
            ),
        )
    )
    return dict(value)


def slot(task: Json, raw: Path, lines: list[str], root: Path = e.ROOT) -> Json:
    """Authenticate exact producer bytes and keep rejected producers terminal."""
    row = previous.resolve(task, raw, lines, root)
    number = row["experiment_id"]
    if row["producer_executed"] and row["declared_reference"]["sha256"] != "sha256:" + PINS[number]:
        row.update(
            producer_qualified=False, scientific_result_measured=False, verdict_class="disqualified"
        )
        row["authentication_errors"].append("independent_v750_primary_pin")
    row["repository_health_receipts"] = []
    if row["producer_executed"] and row["terminal_authenticated"]:
        value = e.load(row["primary_reference"])
        health = value.get("repository_health_reference", {}).get("receipt")
        failed = [r for r in value.get("validation_receipts", []) if not r.get("passed")]
        if (
            failed
            and failed == [health]
            and not row["authentication_errors"]
            and value.get("required_checks_passed") is True
            and value.get("flagged_adversarial") is False
        ):
            row.update(
                producer_qualified=True,
                absolute_metric=1,
                scientific_result_measured=value["verdict_class"]
                in {"positive", "circular_positive", "null"},
            )
            row["repository_health_receipts"] = failed
    if row["producer_qualified"]:
        value = e.load(row["primary_reference"])
        work = e.load(value["primitive_reference"])
        refs = [work[k] for k in ["measurement_reference", "analysis_reference"] if k in work]
        refs += work.get("raw_references", [])
        for ref in refs:
            e.verify(ref)
        row["evidence_references"] += refs
    if number == 8754:
        row["scientific_result_measured"] = False
    return dict(row)


e.contract, e.slot, e.progress = authority, slot, progress
e.authority = SimpleNamespace(authority=authority)
e.previous = SimpleNamespace(original_inputs=methods, prior_rows=previous.prior_rows)
m.e, report.e, report.reduce = e, e, reduce
BASE_BUILD = report.build


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Close administrative work while retaining all scientific promotion boundaries."""
    value: Json = BASE_BUILD(work, receipts, output)
    value.update(
        experiment_id=8762,
        honest_verdict=f"complete_{value['verdict_class']}_v750_capstone",
        random_seed=7508762,
    )
    value["rows"][-1].update(experiment_id=8762, honest_verdict=value["honest_verdict"])
    value["methodology_note"] = (
        "Read-only reconciliation of fourteen complete V750 tasks. Reconstruct admitted verifier and installed process costs from primitive rows. Missing learning, retention and ordinary durable-window comparisons remain terminal blocked. Preserve V749 quarantines, historical batching1.1382x and same-batch10x null. Historical FoVer publication readiness certifies no V750 claim. No model, numerical training or physical device execution."
    )
    value["repository_health_reference"] = work["repository_health_reference"]
    value["scoped_validation_scope"] = dict(
        test_files=e.TESTS,
        consumers=runtime.E2E,
        repository_health_is_dependency=False,
        e2e=["E2E-018", "E2E-021"],
    )
    value["full_task_mutation_rows"] = work["contract_mutation_rows"]
    value["gate_check_summary"] += value["scientific_reconstruction"]["blocked_comparison_gates"]
    value["registered_prior_failure_count"] = len(work["prior_rows"])
    for row in value["retirement_candidate_rows"]:
        row["retire_if_same_verdict"] = row["registration"]["retire_if_same_verdict"]
        row["changed_method_reopen_conditions"] = [
            row["registration"]["addressed_by"],
            "Require changed authenticated operands and complete scoped validation; unchanged external absence is terminal.",
        ]
    for row in value["next_decision_rows"]:
        row["changed_mechanism_conditions"] = [
            row["scope"],
            "New authenticated input or measured correction of the failed execution mechanism; preserve the original scientific estimand.",
        ]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Bind actual producer bytes and clocks; scheduling is not science; null, missing and rejected operands differ.",
    )
    value["reproducibility_checksum"] = e.digest(value)
    return value


def measure(raw: Path, date: str, private: bool) -> Json:
    """Seal nodes and complete-task controls before reducing any scientific operands."""
    mutations = controls()
    collection = runtime.child(
        "freeze_test_nodes",
        [
            runtime.sys.executable,
            "-m",
            "pytest",
            *runtime.COMMON,
            "--collect-only",
            *e.TESTS,
            *runtime.E2E,
        ],
        raw / "premeasurement",
        deadline=90,
    )
    nodes = [
        line
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    e.atomic_json(raw / "frozen_nodes.json", dict(nodes=nodes, collection=collection))
    work: Json = m.measure(raw, date, private)
    work["code_refs"].append(e.reference(raw / "frozen_nodes.json"))
    work["repository_health_reference"] = dict(
        reference=e.reference(
            e.ROOT / "results/raw" / e.NAME / "startup/full_python_suite.receipt.json"
        ),
        full_suite_pass=False,
        acceptance_dependency=False,
    )
    work["contract_mutation_rows"] = mutations
    e.atomic_json(raw / "measurement.json", work)
    e.atomic_json(
        raw.parents[1] / "seals" / (raw.name + ".json"),
        dict(measurement=e.reference(raw / "measurement.json"), code_refs=work["code_refs"]),
    )
    return work


runtime.e, runtime.measure, runtime.report = e, measure, SimpleNamespace(build=build)
main = runtime.main
