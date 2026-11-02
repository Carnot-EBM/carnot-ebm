"""REQ-REPORT-8736: reuse custody while keeping historical controllers unchanged."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting import v746_capstone as driver
from carnot.reporting import v746_capstone_inputs as previous
from carnot.reporting import v747_capstone_inputs as original_inputs
from carnot.reporting import v747_capstone_measure as original_measure
from carnot.reporting import v747_capstone_report as original_report
from carnot.reporting import v748_collection_contract as binding
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v748_capstone_reduce import reduce

Json = dict[str, Any]
e = binding.namespace(original_inputs, "capstone_inputs")
m = binding.namespace(original_measure, "capstone_measure")
report = binding.namespace(original_report, "capstone_report")
runtime = binding.namespace(driver, "capstone_runner")
e.NAME, e.TASK, e.MILESTONE = "experiment_8736_v748_capstone", "exp8736-capstone", "2026.11.748"
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_v748_capstone_8736.py"]
e.OWNED = [
    "python/carnot/reporting/v748_capstone.py",
    "python/carnot/reporting/v748_capstone_reduce.py",
    e.CLI,
]
e.DESIGN, e.contract_source = binding.m.DESIGN, binding.m
e.PINS = dict(original_inputs.PINS)
PINS = {
    8723: "d1641e717c05a4d31d0b267cb5be76899773e71d233f1c73852471d4bc53553f",
    8724: "53a6baafaa3e23f12758f9f58433c69c5df8dc14b73a525a1a620f29ecaddebb",
    8726: "bdcd68bbbfe5d16568adbfebd09b1415602fd25d7acc43c8949bff95e08ba680",
    8729: "2ba30690e2ef226818658114ecf02d9cff002aa7a6ba6e9cdd242e868fd66a9e",
    8731: "e2c6271989cee2e644644311f2068fd0c8f66bf050d2a9b48a3828325f1f5e89",
    8732: "063218e46ed7d2ad36a3bf3e4282ffde79a9888b7cdbacfcaf3178132157e6e3",
    8733: "955b3ccac700d28094d896f458ca9c8aea44b73f25917f76ad83b2529fe99b44",
    8734: "38b02bfcd3421c9f5cc420862fdd035d33fa78a8c671092afb81e3622ed1d331",
    8735: "d60f2112d6ba55c48675c39a5c38039d42f3d1877b4ae1ad79fd60be7bae7651",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual boundaries so a supervisor can distinguish evidence reads from a stall."""
    print(f"[exp8736] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path) -> Json:
    """The pinned full contract grants eligibility independently of all scientific gates."""
    result = binding.m.authority(root)
    table, tasks = parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)
    return dict(result, tasks=tasks, visible_table=table, canonical_tasks_sha256=binding.m.PIN)


def methods() -> Json:
    """Historical qualifiers remain operands; a current table cannot replace original methods."""
    result = previous.original_inputs()
    result["references"] += [
        e.reference(e.ROOT / name)
        for name in [
            "results/experiment_8722_v747_capstone.json",
            "docs/research-notes/v748-capstone.md",
            "_bmad/architecture.md",
        ]
    ]
    return dict(result)


def slot(task: Json, raw: Path, lines: list[str], root: Path = e.ROOT) -> Json:
    """Authenticate exact producers; changed executable identities cannot certify current work."""
    row = previous.resolve(task, raw, lines, root)
    failures = []
    if row["producer_executed"]:
        declared, number = row["declared_reference"], row["experiment_id"]
        exact = (
            row["primary_reference"]["path"] == declared["path"]
            and declared["sha256"] == "sha256:" + PINS[number]
        )
        if not exact:
            failures.append(
                dict(
                    check="exact_declared_versioned_primary",
                    **declared,
                    upstream=task["id"],
                    field="sha256",
                    operator="==",
                    expected="sha256:" + PINS[number],
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
                        check="unchanged_executable_identity",
                        **observed,
                        upstream=task["id"],
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
BASE_BUILD = report.build

REOPEN = [
    "Restore exact source/test identities and qualify the invocation-owned collection inventory.",
    "Change the deadline mechanism and prove a normally exiting deadline smoke within its bound.",
    "Qualify the unchanged64 original nodes,27 runner nodes and83 statement anchors with real deadline controls.",
    "Freeze a changed natural evidence representation and cost comparator before scoring.",
    "Admit a qualified learning union, then measure delayed future proper loss on untouched units.",
    "Require measured future proper-loss improvement and simultaneous retention at equal complete cost.",
    "Use independent natural evidence and match predicate work or wall time as well as calls.",
    "Provide a causally changed authenticated CUDA runtime and cached candidate probabilities at the frozen budget.",
    "Change the state boundary while preserving sequential arithmetic, acknowledgements and restart parity.",
    "Provide a measured faster durability mechanism with both original barriers and a complete roster within budget.",
    "Remove duplicate feature extraction, preserve exact same-arm outputs and remeasure matched B16 complete costs.",
    "Admit genuinely new live cross-game supervisor receipts beyond the frozen frontier.",
    "Run an operation-specific board kernel with parity and complete host, transfer and fabric costs.",
    "Reconcile only changed authenticated evidence; unchanged external absence needs no retry.",
]


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Administrative completion cannot promote a blocked learning or physical claim."""
    value: Json = BASE_BUILD(work, receipts, output)
    verdict = value["verdict_class"]
    value.update(
        experiment_id=8736,
        honest_verdict=f"complete_{verdict}_v748_capstone",
        random_seed=7488736,
        preserved_v747_qualifiers=binding.methods()["v747_qualifiers"],
        semantic_decision_rows=value["scientific_reconstruction"].get("verifier", {}),
        resident_cost_decision_rows=value["scientific_reconstruction"]["resident_cost"],
        installed_attribution_decision_rows=value["scientific_reconstruction"]["installed"],
        next_measurement_priorities=[
            "Close PRD delayed-learning gap: qualify the original execution union, then measure future Brier and retention.",
            "Test duplicate-feature removal against exact output parity and same-batch B16 complete costs; attribution is not speedup.",
            "Stop unchanged CUDA, exposed-oracle generalization, resident durable throughput and physical acceleration claims.",
        ],
        methodology_note="Read-only reconciliation of fourteen complete V748 tasks, original scientific methods and sealed producer rows. Equal-call verifier utility remains constructed/oracle-dependent. Resident parity grants no speed; the durability probe leaves complete cost absent. Installed attribution grants no optimization benefit. Missing learning, probabilities, CUDA repair and physical execution remain terminal blocked. Historical verdicts and stable publication meanings remain intact.",
    )
    value["rows"][-1].update(experiment_id=8736, honest_verdict=value["honest_verdict"])
    for row in value["rows"][:-1]:
        value["gate_check_summary"] += row["source_identity_failures"]
    for decision, condition in zip(value["next_decision_rows"], REOPEN, strict=True):
        decision["changed_mechanism_conditions"] = [condition]
    for row in value["retirement_candidate_rows"]:
        row["changed_method_reopen_conditions"] = [
            next(
                d["changed_mechanism_conditions"][0]
                for d in value["next_decision_rows"]
                if d["task_id"] == row["task_id"]
            )
        ]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Exact sealed evidence and current clocks; absent science differs from zero; execution readiness grants no scientific benefit.",
    )
    value["reproducibility_checksum"] = e.digest(value)
    return value


runtime.e, runtime.measure, runtime.report = e, m.measure, SimpleNamespace(build=build)
main = runtime.main
