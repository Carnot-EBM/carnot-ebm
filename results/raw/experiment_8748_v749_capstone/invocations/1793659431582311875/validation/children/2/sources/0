"""REQ-REPORT-8748: reconcile sealed V749 evidence without reopening blocked science."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting import v746_capstone as driver
from carnot.reporting import v746_capstone_inputs as previous
from carnot.reporting import v747_capstone_inputs as original_inputs
from carnot.reporting import v747_capstone_measure as original_measure
from carnot.reporting import v747_capstone_report as original_report
from carnot.reporting import v748_collection_contract as namespaces
from carnot.reporting import v749_contract_methods as binding
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v749_capstone_reduce import reduce, HISTORY

Json = dict[str, Any]
e = namespaces.namespace(original_inputs, "capstone8748_inputs")
m = namespaces.namespace(original_measure, "capstone8748_measure")
report = namespaces.namespace(original_report, "capstone8748_report")
runtime = namespaces.namespace(driver, "capstone8748_runner")
e.NAME, e.TASK, e.MILESTONE = "experiment_8748_v749_capstone", "exp8748-capstone", binding.MILESTONE
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_v749_capstone_8748.py"]
e.OWNED = [
    "python/carnot/reporting/v749_capstone.py",
    "python/carnot/reporting/v749_capstone_reduce.py",
    e.CLI,
]
e.DESIGN, e.contract_source = binding.DESIGN, binding
PINS = {
    8737: "e02195338993c16ffba57e73c5d2c49c5e5952de15c19099b8563227890018c9",
    8738: "cb2d2b4a2e2b8425907af8a90f40d8bf2706ee250ef58c47768ce906faef29a6",
    8743: "3d14cab0a2e546fff5f3adece9f62751809b19ef176b87dce67d0c2d07bb40ae",
    8745: "8852ad29e6bff49d22deb10793acca2348d79ddf42b3deafd5504c83d670df6b",
    8746: "97a365f58aecb2053246acc020375ac7e523e0b06d683c5b78d163ae4af1f155",
    8747: "7bf48a3412da025fcf2c3336911ec109c9e9674c24ba6d31b39b1d1f28e91f51",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries because a supervisor must distinguish evidence reads from a stall."""
    print(f"[exp8748] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path) -> Json:
    """Bind complete tasks independently of any producer's execution readiness."""
    result = binding.authority(root)
    table, tasks = parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)
    return dict(result, tasks=tasks, visible_table=table, canonical_tasks_sha256=binding.PIN)


def methods() -> Json:
    """Keep the old protocol and archive as operands, never substitute current outcomes."""
    result = previous.original_inputs()
    result["references"] += [
        HISTORY,
        e.reference(e.ROOT / "docs/research-notes/v749-capstone.md"),
        e.reference(
            e.ROOT / "openspec/change-proposals/research-roadmap-v748-preserved-20261102.md"
        ),
    ]
    manifest = e.ROOT / "ops/exclusion_manifest.yaml"
    value = binding.yaml.safe_load(manifest.read_bytes())
    retired = value["retired"] + value.get("retired_experiments", [])
    excluded = any(
        str(r.get("experiment_id")) in {"8748", e.TASK} for r in retired if isinstance(r, dict)
    )
    result["references"].append(e.reference(manifest))
    result["gate_check_summary"].append(
        dict(
            upstream=e.TASK,
            **e.reference(manifest),
            check="exclusion_manifest",
            field="retired_experiment_ids",
            operator="not_contains",
            expected=8748,
            observed=8748 if excluded else [],
            passed=not excluded,
        )
    )
    return dict(result)


def slot(task: Json, raw: Path, lines: list[str], root: Path = e.ROOT) -> Json:
    """Authenticate exact primary and executable bytes; skipped studies have no science."""
    row = previous.resolve(task, raw, lines, root)
    failures = []
    if row["producer_executed"]:
        declared = row["declared_reference"]
        exact = (
            row["primary_reference"]["path"] == declared["path"]
            and declared["sha256"] == "sha256:" + PINS[row["experiment_id"]]
        )
        if not exact:
            failures.append(
                dict(
                    upstream=task["id"],
                    **declared,
                    check="exact_declared_primary",
                    field="sha256",
                    operator="==",
                    expected="sha256:" + PINS[row["experiment_id"]],
                    observed=declared["sha256"],
                    passed=False,
                )
            )
        value = e.load(row["primary_reference"]) if exact else {}
        refs = value.get("code_config_hashes", [])
        if isinstance(refs, dict):
            refs = [dict(path=p, sha256=h) for p, h in refs.items()]
        for ref in refs:
            if Path(ref["path"]).suffix not in {".py", ".so"}:
                continue
            observed = e.reference(Path(ref["path"]))
            row["evidence_references"].append(observed)
            if observed["sha256"] != ref["sha256"]:
                failures.append(
                    dict(
                        upstream=task["id"],
                        **observed,
                        check="unchanged_executable_identity",
                        field="sha256",
                        operator="==",
                        expected=ref["sha256"],
                        observed=observed["sha256"],
                        passed=False,
                    )
                )
        if failures:
            row.update(
                producer_qualified=False,
                scientific_result_measured=False,
                verdict_class="disqualified",
            )
            row["authentication_errors"] += [f["check"] for f in failures]
    row["source_identity_failures"] = failures
    return dict(row)


e.contract, e.slot, e.progress = authority, slot, progress
e.authority = SimpleNamespace(authority=authority)
e.previous = SimpleNamespace(original_inputs=methods, prior_rows=previous.prior_rows)
m.e, report.e, report.reduce = e, e, reduce
BASE_BUILD, BASE_VALIDATE = report.build, runtime.validate
REOPEN = [
    "Qualify the sealed full task contract and original methods with normally exiting validation.",
    "Identify the original deadline failure's measured cause and prove the unchanged controls within their bound.",
    "Qualify all original 64 nodes, 27 runner nodes and 83 statement anchors under unchanged scientific methods.",
    "Admit the qualified learning union, then measure predictions before delayed feedback and future Brier at complete cost.",
    "Admit future proper-loss rows and demonstrate simultaneous retention at the frozen complete-cost boundary.",
    "Admit the frozen five-call verifier protocol before scoring untouched new-family constructed worlds.",
    "Qualify feature ownership, exact outputs and rejection boundaries with complete current validation.",
    "Admit feature reuse first, then measure matched B16 full parent-lifetime process costs.",
    "Provide an authenticated trajectory with dense renormalization in the middle window and original per-feedback durability.",
    "Admit genuinely new reachable cross-game supervisor receipts beyond the sealed frontier.",
    "Run operation-specific KV260, PolarFire or GateMate parity with complete host, transfer and fabric costs.",
    "Reconcile only changed authenticated inputs; unchanged external absence requires no retry.",
]


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Administrative closure cannot turn an absent comparison into research success."""
    value: Json = BASE_BUILD(work, receipts, output)
    history = e.load(HISTORY)
    value.update(
        experiment_id=8748,
        honest_verdict=f"complete_{value['verdict_class']}_v749_capstone",
        random_seed=7498748,
        intended_count=12,
        completed_count=12,
    )
    value["sample_size_budget"]["intended_scheduling_slots"] = 12
    value["rows"][-1].update(experiment_id=8748, honest_verdict=value["honest_verdict"])
    value["methodology_note"] = (
        "Read-only reconciliation of twelve complete V749 tasks. Authenticate producer bytes separately from conductor skips. Missing learning, verifier transfer, complete feature-reuse costs and durable-window comparisons remain terminal blocked. Preserve historical constructed oracle-dependent verification and same-batch 10x null. Stable FoVer publication gates grant no new learning, ARC generalization or physical acceleration. No model operations."
    )
    value["historical_v748_reference"] = HISTORY
    value["historical_v748_scientific_reconstruction"] = history["scientific_reconstruction"]
    value["gate_check_summary"] += [
        f for r in value["rows"][:-1] for f in r["source_identity_failures"]
    ]
    value["gate_check_summary"] += value["scientific_reconstruction"]["blocked_comparison_gates"]
    value["gate_check_summary"] += [
        dict(
            upstream=e.TASK,
            path=r.get("stdout_path", str(Path(work["raw"]) / "validation.json")),
            sha256=r.get("stdout_sha256"),
            check=r.get("name", "owned_validation"),
            field="exit_code" if "exit_code" in r else "passed",
            operator="==",
            expected=r.get("expected_exit", True),
            observed=r.get("exit_code", r["passed"]),
            passed=False,
        )
        for r in receipts
        if not r["passed"]
    ]
    value["registered_prior_failure_count"] = len(work["prior_rows"])
    for decision, condition in zip(value["next_decision_rows"], REOPEN, strict=True):
        decision["changed_mechanism_conditions"] = [condition]
    for row in value["retirement_candidate_rows"]:
        row["retire_if_same_verdict"] = row["registration"]["retire_if_same_verdict"]
        row["changed_method_reopen_conditions"] = [
            REOPEN[
                next(
                    i for i, t in enumerate(work["contract"]["tasks"]) if t["id"] == row["task_id"]
                )
            ]
        ]
    value["full_task_mutation_rows"] = work["contract_mutation_rows"]
    value["required_checks_passed"] = bool(receipts) and all(r["passed"] for r in receipts)
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Bind current claims to exact producer bytes and clocks; missing science differs from zero and closure differs from benefit.",
    )
    value["reproducibility_checksum"] = e.digest(value)
    return value


def measure(raw: Path, date: str, private: bool) -> Json:
    """Seal full-object controls in the same independent measurement used by cold replay."""
    mutations = binding.mutations()
    work: Json = m.measure(raw, date, private)
    work["contract_mutation_rows"] = mutations
    e.atomic_json(raw / "measurement.json", work)
    e.atomic_json(
        raw.parents[1] / "seals" / (raw.name + ".json"),
        dict(measurement=e.reference(raw / "measurement.json"), code_refs=work["code_refs"]),
    )
    return work


def validate(raw: Path) -> list[Json]:
    """Retain the full suite's real failure alongside scoped coverage and E2E evidence."""
    receipts: list[Json] = BASE_VALIDATE(raw)
    full = e.ROOT / "results/raw" / e.NAME / "startup/full_python_suite.receipt.json"
    receipts.append(e.load(e.reference(full)))
    return receipts


runtime.e, runtime.measure, runtime.report, runtime.validate = (
    e,
    measure,
    SimpleNamespace(build=build),
    validate,
)
main = runtime.main
