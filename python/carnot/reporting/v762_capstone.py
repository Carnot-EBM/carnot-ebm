"""REQ-REPORT-8916: complete custody preserves null and blocked scientific outcomes."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from carnot.reporting import v758_capstone as previous
from carnot.reporting import v758_capstone_runner as previous_runner
from carnot.reporting import v759_capstone as historical
from carnot.reporting import v762_branch_contract as original
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.v748_collection_contract import namespace
from carnot.reporting.v762_capstone_reduction import reduce as reduce
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT, MILESTONE = original.ROOT, original.MILESTONE
NAME, TASK = "experiment_8916_v762_capstone", "exp8916-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v762_capstone.py",
    "python/carnot/reporting/v762_capstone_reduction.py",
    CLI,
]
TEST = "tests/python/test_v762_capstone_8916.py"
EXECUTION = (
    "results/raw/experiment_8904_v762_branch_contract/invocations/1794078350765445874/authority"
)
CAPSTONE = "results/experiment_8903_v761_capstone.json"
SPEC = "openspec/capabilities/research-reporting/spec.md"
base = namespace(previous, "capstone8916")
reference, retain, checksum = base.reference, base.retain, base.checksum
accounted = historical.accounted


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries and real counts without renewing the fixed supervisor deadline."""
    print(f"[exp8916] phase={phase} completed={completed} pending={pending}", flush=True)


def admit(path: Path, task_id: str) -> Json:
    """Authenticate original fields and support both measured producer coverage schemas."""
    value: Json = {}
    refs: list[Json] = []
    try:
        value = json.loads(path.read_bytes())
        refs = base.shared.references(value)
        validate_primary(value, path)
        if (
            value["experiment_id"] == 8904
            and reference(path)["sha256"]
            != "sha256:6da560933050f3f9a35c0230dc0cc7151c6b4edcf19f272db7202f41c8dace18"
        ):
            raise ValueError("original_execution_producer_drift")
        coverage, receipts = value["owned_statement_coverage"], value["validation_receipts"]
        measured = coverage.get("complete") is True and coverage.get("percent") == 100
        measured |= bool(
            coverage.get("total_statements", 0) > 0
            and coverage.get("covered_statements") == coverage.get("total_statements")
            and coverage.get("percent_covered") == 100
            and coverage.get("missing_lines") == []
            and coverage.get("excluded_lines") == []
        )
        if (
            value["task_id"] != task_id
            or value["required_checks_passed"] is not True
            or value.get("flagged_adversarial")
            or value["verdict_class"] == "disqualified"
            or value.get("verifier_is_oracle") is True
            and value["verdict_class"] == "positive"
            or not measured
            or not receipts
            or any(not r["passed"] for r in receipts if r.get("scope") != "repository_health")
        ):
            raise ValueError("producer_identity_or_owned_validation")
        originals = json.loads(
            (ROOT / "results/experiment_8915_v762_hardware_operation_budget.json").read_bytes()
        )["raw_shard_hashes"]
        specs = {r["sha256"]: r for r in originals if r.get("original_path", "").endswith(SPEC)}
        refs = []
        for ref in base.shared.references(value):
            resolved = dict(ref)
            if (
                Path(ref["path"]) == ROOT / SPEC
                and reference(Path(ref["path"]))["sha256"] != ref["sha256"]
            ):
                resolved = dict(specs.get(ref["sha256"], ref), original_path=ref["path"])
            if reference(Path(resolved["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("producer_operand_drift:" + ref["path"])
            refs.append(resolved)
        if not refs:
            raise ValueError("producer_custody_missing")
        return dict(admitted=True, errors=[], references=refs, value=value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(admitted=False, errors=[str(error)], references=refs, value=value)


def contract(root: Path) -> Json:
    """Use original execution bytes so subsequent queue edits cannot change the roster."""
    return dict(original.authority(root / EXECUTION))


def summary(value: Json) -> Json:
    """Keep upstream compute distinct from this capstone's administrative work."""
    result: Json = historical.summary(value)
    result.update(
        {
            k: value[k]
            for k in [
                "model_invocation_counts",
                "numerical_training_calls",
                "independent_count",
                "intended_count",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "inference_substrate",
                "inference_substrate_class",
                "MODEL_SPECS",
            ]
            if k in value
        }
    )
    return result


base.authority = SimpleNamespace(
    ROOT=ROOT,
    MILESTONE=MILESTONE,
    PIN=original.PIN,
    PINS={
        f"{EXECUTION}/{original.DESIGN}": "b9663afee4f50f303219e263359656e852e9d0c6259bea33fd74c5e34848a918",
        f"{EXECUTION}/{original.ACTIVE}": "924cf84e75570a869a5e2fb743dadd6359f2155f9ef5387666f065e5e8bd8912",
    },
    DESIGN=f"{EXECUTION}/{original.DESIGN}",
    ACTIVE=f"{EXECUTION}/{original.ACTIVE}",
    STAGED=f"{EXECUTION}/{original.STAGED}",
    CAPSTONE=CAPSTONE,
    authority=contract,
)
base.OWNED, base.TEST, base.NAME, base.TASK, base.CLI, base.MILESTONE = (
    OWNED,
    TEST,
    NAME,
    TASK,
    CLI,
    MILESTONE,
)
base.reduce, base.progress = reduce, progress
base.admit = base.shared.admit = admit
base.shared.producer_summary = summary


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Seal every original producer and scheduler receipt before interpreting science."""
    if (raw / "work.json").exists():
        raise FileExistsError("immutable_invocation_already_exists")
    template = ExperimentTemplate(8916, "V762 capstone", f"results/{NAME}.json", seed=7628916)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(base.freeze, root, raw, run_date)
        while not future.done():
            try:
                future.result(timeout=30)
            except TimeoutError:
                progress("operand_custody_wait", 0, 1)
        work: Json = future.result()
    work["historical"] = {
        k: work["historical"][k]
        for k in [
            "rows",
            "honest_verdict",
            "verdict_class",
            "constructed_measurement_count",
            "independent_constructed_unit_count",
            "natural_generalization_count",
            "historical_publication_gates",
        ]
    }
    work["preconditions"].update(
        random_seed=template.random_seed,
        producer_invocation=dict(executable=sys.executable, argv=sys.argv, cwd=os.getcwd()),
        hard_cap_s=4800,
    )
    log = next(
        r for r in work["snapshots"] if r.get("source_path", "").endswith("ops/conductor-log.md")
    )
    for slot, task in zip(work["slots"], work["contract"]["tasks"][:-1], strict=True):
        slot["conductor_reference"] = log
        observed = accounted(slot, task)
        slot.update(
            accounted=observed,
            metric=int(observed),
            status="completed" if observed else "censored",
            missing_reason=None if observed else "missing_file",
        )
    work["code"] += [
        reference(ROOT / p)
        for p in [
            "python/carnot/reporting/v758_capstone.py",
            "python/carnot/reporting/v758_capstone_runner.py",
            "python/carnot/reporting/v762_branch_contract.py",
        ]
    ]
    atomic_json(raw / "work.json", work)
    return work


DECISIONS = [
    (
        "contract_ready_score",
        "Keep exact authority and independent CPU roots",
        "Authenticated causal CUDA environment change plus actual model offload",
    ),
    (
        "candidate_ready_score",
        "Retain pilot qualification separately from future benefit",
        "Changed representation or candidate promotion mechanism with untouched retention",
    ),
    (
        "finite_budget_ready_score",
        "Retain safe refusal proof under its finite-family assumptions",
        "Changed allocation and fresh disjoint labels before another savings claim",
    ),
    (
        "retained_evidence_ready_score",
        "Retain original observer custody",
        "Qualified future-world observer overhead and complete clock closure",
    ),
    (
        "natural_canary_ready_score",
        "Retain paired annotation audit without independent natural truth",
        "Independent source-semantic annotations and broader source coverage",
    ),
    (
        "causal_rows_ready_score",
        "Retain sealed causal rows without learning benefit",
        "Changed informative candidate mechanism with future and retention evidence",
    ),
    (
        "shared_union_benefit_score",
        "Retain certificate null with shared and standalone costs",
        "All-policy allocation must lower paid label union costs with terminal decision equality",
    ),
    (
        "guarded_benefit_score",
        "Retire unchanged guarded-learning benefit retry",
        "Changed mechanism must pass all nine frozen future and retention bounds",
    ),
    (
        "cost_attribution_ready_score",
        "Retain whole-request clocks; removable fractions remain unqualified",
        "Lower measured observer overhead and complete disjoint parent windows",
    ),
    (
        "natural_canary_ready_score",
        "Keep Qwen natural canary terminal blocked",
        "Authenticated changed runtime plus actual Qwen GGUF invocation and paired case outputs",
    ),
    (
        "adaptation_cost_ready_score",
        "Quarantine disqualified durable-service measurements",
        "Repair owned strict-row control before qualifying inclusive durable costs",
    ),
    (
        "hardware_map_ready_score",
        "Retain unknown hardware fractions and physical limits",
        "Supported equivalent device kernel plus measured host, transfer and durability fractions",
    ),
]


def build(work: Json, validation: Json) -> Json:
    """Administrative readiness cannot promote a scientific null or rejected timing."""
    base.report.c = SimpleNamespace(
        TASK=TASK,
        ROOT=ROOT,
        MILESTONE=MILESTONE,
        authority=SimpleNamespace(PIN=original.PIN, HISTORY=CAPSTONE),
        reference=reference,
        retirement=base.retirement,
        shared=base.shared,
        checksum=checksum,
    )
    value: Json = base.report.build(work, validation)
    owned = value["required_checks_passed"]
    ready = bool(
        owned
        and work["contract"]["ready"]
        and len(work["slots"]) == 12
        and all(s["accounted"] for s in work["slots"])
    )
    blocked = not work["contract"]["ready"] or any(
        not s["admitted"] or s["verdict_class"] in {"blocked", "disqualified"}
        for s in work["slots"]
    )
    verdict = "disqualified" if not owned else "blocked" if blocked else "null"
    honest = f"complete_{verdict}_v762_capstone"
    value["rows"][-1].update(
        experiment_id=8916,
        verdict_class=verdict,
        primary_honest_verdict=honest,
        metric=int(ready),
        publication_evidence=dict(
            method="hash-bound atomic publisher requires cold replay before public bytes",
            validation=reference(Path(work["raw"]) / "validation.json"),
        ),
    )
    decisions, denominators = [], []
    for slot, branch, (field, decision, mechanism) in zip(
        work["slots"], value["branch_conclusions"], DECISIONS, strict=True
    ):
        producer, reduction = slot["producer"], slot["reduction"]
        gate = dict(
            check="continuation_gate",
            upstream=slot["task_id"],
            artifact_field=field,
            expected=1,
            observed=producer.get(field),
            passed=slot["admitted"] and type(producer.get(field)) is int and producer[field] == 1,
            **slot["primary_reference"],
        )
        branch.update(
            primary_reference=slot["primary_reference"],
            scheduler_receipts=slot["scheduler_receipts"],
            conductor_reference=slot["conductor_reference"],
            upstream_compute=slot["upstream_compute"],
            producer_counts={
                k: producer.get(k)
                for k in [
                    "intended_count",
                    "completed_count",
                    "failed_count",
                    "censored_count",
                    "excluded_count",
                    "independent_count",
                ]
            },
            inference_substrate_class=producer.get("inference_substrate_class", "blocked_no_run"),
            verifier_is_oracle=producer.get("verifier_is_oracle", False),
            continuation_gate=gate,
            next_decision=decision,
            changed_mechanism_needed=mechanism,
            exact_scope_retirement="repair_owned_validation"
            if slot["verdict_class"] == "disqualified"
            else "block_external_once"
            if not slot["admitted"]
            else "retire_unchanged_retry"
            if not gate["passed"]
            else "retain_qualified_scope",
        )
        decisions.append(
            {
                k: branch[k]
                for k in [
                    "task_id",
                    "next_decision",
                    "changed_mechanism_needed",
                    "continuation_gate",
                    "exact_scope_retirement",
                ]
            }
        )
        denominators.append(
            dict(
                task_id=slot["task_id"],
                streams=len(reduction.get("stream_ids", [])),
                calibration_replicates=len(reduction.get("calibration_replicate_ids", [])),
                corpus_cases=reduction.get("corpus_case_count", 0),
                cost_cells=reduction.get("cpu_cost_window_count", 0)
                + reduction.get("cpu_request_count", 0),
                producer_independent_count=producer.get("independent_count"),
                admitted=slot["admitted"],
                lineage=slot["primary_reference"],
                scope="branch-local only; streams, repeats, cases and cost cells never form one independent n",
            )
        )
    decisions.append(
        dict(
            task_id=TASK,
            next_decision="Keep scientific conjunctions separate; no activation or publication",
            changed_mechanism_needed="Admit changed evidence for each blocked or failed conjunction",
            continuation_gate=dict(
                artifact_field="reconciliation_ready_score",
                observed=int(ready),
                expected=1,
                passed=ready,
            ),
            exact_scope_retirement="administrative_self_slot_only",
        )
    )
    for recommendation in value["retirement_recommendations"]:
        recommendation["broader_hypothesis_retired"] = False
    gates = value["historical_publication_gates"]
    value.update(
        experiment_id=8916,
        honest_verdict=honest,
        verdict_class=verdict,
        random_seed=7628916,
        execution_venue="host",
        reconciliation_ready_score=int(ready),
        contract_ready_score=int(work["contract"]["ready"]),
        outcome_rows=value["rows"],
        continuation_rows=decisions,
        independent_denominator_rows=denominators,
        cited_upstream_artifacts=[s["primary_reference"] for s in work["slots"]],
        publication_gate_rows=[
            dict(
                gate=k,
                historical_scope="historical FoVer paper only",
                historical_result=gates["gates"][k],
                current_scope="V762 constructed streams, finite certificates, assisted annotation and CPU costs",
                current_independent_publication_evidence=False,
            )
            for k in ["G1", "G2", "G3", "G4"]
        ],
        paper_ready=gates["paper_ready"],
        unmet_gates=gates["unmet_gates"],
        exposure_scope="thirteen V762 administrative slots; constructed guarded streams, finite certificates, assisted corpus annotations and CPU costs",
    )
    value["gate_check_summary"] += work["contract"]["failures"]
    value["acceptance_gates"]["complete_accounting"]["passed"] = ready
    value["methodology_note"] = (
        "No new model, numerical training, device probe or natural-language inference. Independent branch reductions preserve the guarded future/retention conjunction, all-policy paid labels and terminal equality, annotation dependence and complete CPU costs. Oracle truth and assisted annotations do not establish independent generalization. Disqualified durable costs cannot qualify hardware fractions. Historical FoVer G1-G4 retain historical scope. Immutable external absence stays terminal blocked."
    )
    value["raw_shard_hashes"] = list(
        {(r["path"], r["sha256"]): r for r in value["raw_shard_hashes"]}.values()
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind original bytes for "
        + k
        + "; complete administrative accounting never grants scientific benefit, independent n or deployment authority."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


base.build = build
legacy_replay = base.replay


def replay(path: Path) -> Json:
    """Rebuild scientific reductions and require authentic scheduler order after rehashing."""
    result: Json = legacy_replay(path)
    if not result["passed"]:
        return result
    value = json.loads(path.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    passed = all(
        s["task_id"] == t["id"] and s["accounted"] == accounted(s, t)
        for s, t in zip(work["slots"], work["contract"]["tasks"][:-1], strict=True)
    )
    return dict(passed=passed, reason="replay_passed" if passed else "scheduler_custody_drift")


base.replay = replay
runner = namespace(previous_runner, "capstone8916_runner")
runner.c = runner.runner.c = runner.runner.checks.c = sys.modules[__name__]
checks = namespace(runner.runner.checks, "capstone8916_bounded_checks")


def run_check(
    name: str, argv: list[str], raw: Path, deadline: float = 120, scope: str = "owned"
) -> Json:
    """Bound custody-heavy children while forwarding their original streams and elapsed time."""
    return dict(
        checks.run_check(
            name,
            argv,
            raw,
            1200
            if name == "owned_tests"
            else 600
            if name in {"private_e2e018_021", "cold_replay"}
            else deadline,
            scope,
        )
    )


def evidence_path(path: str | Path) -> Path:
    """Keep unrelated repository health scoped to this invocation's retained log."""
    return (
        ROOT / "results/raw" / NAME / "development/repository_health.json"
        if str(path) == "/tmp/exp8839-global/receipt.json"
        else Path(path)
    )


runner.runner.checks.run_check = run_check
runner.runner.Path = evidence_path
main = runner.main
