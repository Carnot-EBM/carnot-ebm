"""REQ-REPORT-9067-V773: fourteen terminal outcomes preserve unresolved science."""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v772_capstone as old
from carnot.reporting import v771_capstone as lifecycle
from carnot.reporting import v770_capstone as oldest
from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import v773_branch_contract as authority
from carnot.reporting import mdh_human_release_9056 as human
from carnot.reporting import v773_branch_evidence as history
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from carnot.reporting.v766_branch_evidence import arc_support
from carnot.reporting.invocation_provenance_9000 import environment
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.v768_capstone import edge as edge

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_9067_v773_capstone", "exp9067-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v773_capstone.py", CLI]
TEST = "tests/python/test_v773_capstone_9067.py"
DIRECT = [*old.DIRECT, authority.TEST, human.TEST]
REUSED = list(
    dict.fromkeys(
        [
            *old.REUSED,
            *old.OWNED,
            *authority.OWNED,
            *human.OWNED,
            "python/carnot/reporting/v773_branch_evidence.py",
        ]
    )
)
CHECKS, checksum = old.CHECKS, old.checksum
MODEL_SPECS: list[Json] = []
PINS = dict(
    zip(
        [9054, 9055, 9056, 9057, 9058, 9060, 9066],
        [
            "sha256:9d2757fa9ba48ed23457d667cb92bc88187b1466fbdb948caab4d84e8155fe74",
            "sha256:80aa684f1cc271897823983b66bc2d673073223413233438ad135a81edfbc867",
            "sha256:5a0f0e0fd0831b61fd30f03a311c50327f46c6676f4a77fe33b431eacc86224a",
            "sha256:ec2fb0d9952cf7e48db583e539879164bd15b81c4904d43a05f637be703b13c8",
            "sha256:88b49f41bf789b5e28e37e80d22c983a6cd36993c0f71f146241fffbe9414048",
            "sha256:d920be68911ea7ce262f81361e9c3bdde0ea0767ca54debc165140686111915f",
            "sha256:55ad80aaf449877fffc5c3ac38dca78b44561b6368104592dbf63b4d61c45c50",
        ],
        strict=True,
    )
)
AUTHORITY_PINS = [
    "sha256:96163ef0782d8c6125561eec52e89d4d8982a8829bc0dd41704a9ee5ecac57c4",
    "sha256:a18957b6f7490cc3b3a2506ff98d4e656d606e99f19ef637458372c49ff507b8",
]
LOG_PIN = "sha256:cbb68084a914d3c1659964b71ead616da6f83b872fa8a9ec6018318eca3b6b88"
ALTERNATES = {
    n: f"experiment_{n}_{s}.json"
    for n, s in [
        (9057, "human_probe_methods"),
        (9060, "sealed_corpus_admission"),
        (9066, "arc_transfer_evidence"),
    ]
}
_plan, _scope, _receipts, _validate, _terminal, _main, _build, _slots = (
    old.validation_plan,
    old.scope_manifest,
    old.receipts_pass,
    old.validate,
    old.terminal,
    old.main,
    old.build,
    old.slots,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase counts while the qualified runner supervises its children."""
    print(f"[exp9067] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse the existing lifecycle without editing its consumers or validators."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        CLI=CLI,
        OWNED=OWNED,
        TEST=TEST,
        DIRECT=DIRECT,
        REUSED=REUSED,
        authority=authority,
        PINS=PINS,
        AUTHORITY_PINS=AUTHORITY_PINS,
        LOG_PIN=LOG_PIN,
        progress=progress,
        validation_plan=validation_plan,
        scope_manifest=scope_manifest,
        receipts_pass=receipts_pass,
        collect=collect,
        reduce=reduce,
        build=build,
        replay=replay,
        validate=validate,
        terminal=terminal,
        verify_measured=verify_measured,
        measured=measured,
    ):
        yield


def validation_plan(private: Path) -> list[CommandSpec]:
    """Preserve direct-consumer assertions and private E2E-018/020/021 commands."""
    with binding():
        return _plan(private)


def scope_manifest(private: Path) -> Json:
    """Freeze exact owned code, consumer assertions and commands before accounting."""
    with binding():
        return _scope(private)


def receipts_pass(work: Json) -> bool:
    """Only current normal exits with the frozen command and log hashes qualify."""
    with binding():
        return _receipts(work)


def slots(work: Json) -> list[Json]:
    """Keep scheduler skips, producer verdicts and present-day code drift distinct."""
    with binding():
        return _slots(work)


def contract(saved: Json) -> Json:
    """Full objects and an independent digest prevent authority substitution after rehashing."""
    text = Path(saved["design_reference"]["path"]).read_text()
    queue = yaml.safe_load(Path(saved["queue_reference"]["path"]).read_bytes())
    tasks = parse_design(text, milestone=MILESTONE)[1]
    compared = compare_contract(text, queue, queue, milestone=MILESTONE, first_id=9054, count=14)
    ready = (
        compared["passed"]
        and tasks == queue["tasks"] == saved["tasks"]
        and canonical_hash(tasks) == authority.PIN
        and [saved[k]["sha256"] for k in ["design_reference", "queue_reference"]] == AUTHORITY_PINS
        and all(
            reference(Path(saved[k]["path"]))["sha256"] == saved[k]["sha256"]
            for k in ["design_reference", "queue_reference"]
        )
    )
    for row in compared["rows"]:
        row["checks"]["full_object"] = tasks == queue["tasks"] == saved["tasks"]
        row["matched"] = row["matched"] and ready
        row.update(raw_numerator=sum(row["checks"].values()), raw_denominator=len(row["checks"]))
    return dict(saved, ready=bool(ready), rows=compared["rows"])


def collect(raw: Path, run_date: str, sources: Path = ROOT / "results") -> Json:
    """Retain accepted and rejected input bytes before opening any reduction operands."""
    progress("preconditions_before", 0, 14)
    raw.mkdir(parents=True, exist_ok=True)
    started = time.monotonic_ns()
    private = Path(tempfile.mkdtemp(prefix="carnot-9067-"))
    manifest = scope_manifest(private)
    atomic_json(raw / "validation_scope_manifest.json", manifest)
    frozen = time.monotonic_ns()
    preconditions = environment(raw)
    require(run_date == "20261112", "run_date")
    branch_ref = retain(ROOT / "results/experiment_9054_v773_branch_contract.json", raw, PINS[9054])
    branch = load(branch_ref)
    branch_primitive = branch["raw_shard_hashes"][0]
    branch_primitive_ref = retain(Path(branch_primitive["path"]), raw, branch_primitive["sha256"])
    refs = [branch_ref, branch_primitive_ref]
    authorities = [
        retain(Path(r["path"]), raw, r["sha256"]) for r in branch["authority_manifest"][:2]
    ]
    refs.extend(authorities)
    bound = contract(
        dict(
            tasks=branch["full_task_objects"],
            design_reference=authorities[0],
            queue_reference=authorities[1],
        )
    )
    require(bound["ready"], "full_object_contract")
    # The active queue encodes prompts; the retained original is the decoded authority.
    require(authority.authority(ROOT)["ready"], "current_full_object_contract")
    log = retain(ROOT / "ops/conductor-log.md", raw, LOG_PIN)
    refs.append(log)
    lines = (
        Path(log["path"]).read_text().split(f"Milestone {MILESTONE} activated", 1)[-1].splitlines()
    )
    values: Json = {}
    checks = list(preconditions["checks"])
    primitives: Json = {}
    for index, task in enumerate(branch["full_task_objects"][:-1]):
        n = int(task["id"][3:7])
        primary = sources / Path(task["deliverable"]).name
        selected = primary if primary.is_file() else sources / ALTERNATES.get(n, primary.name)
        if selected.is_file():
            ref = retain(selected, raw, reference(selected)["sha256"])
            original = load(ref)
            passed = ref["sha256"] == PINS.get(n)
            checks.append(
                dict(
                    check="input_bytes",
                    upstream=task["id"],
                    **ref,
                    artifact_field="sha256",
                    op="==",
                    expected=PINS.get(n),
                    observed=ref["sha256"],
                    passed=passed,
                )
            )
            values[task["id"]] = dict(
                original=original,
                value=original if passed else dict(original, flagged_adversarial=True),
                reference=ref,
                primary_status="present" if primary.is_file() else "missing_file",
            )
            refs.append(ref)
            for r in original.get("raw_shard_hashes", []):
                refs.append(retain(Path(r["path"]), raw, r["sha256"]))
            if original.get("raw_shard_hashes"):
                primitives[task["id"]] = refs[-len(original["raw_shard_hashes"])]
            for receipt in original.get("validation_receipts", []):
                if not receipt.get("passed", True) and receipt.get("log_sha256"):
                    refs.append(retain(ROOT / receipt["log_path"], raw, receipt["log_sha256"]))
        progress("producer_authenticated", index + 1, 13 - index)
    historical: Json = {}
    for task in branch["full_task_objects"]:
        for prior in task["prior_failures"]:
            identity_key = prior["experiment_id"]
            paths = list((ROOT / "results").glob(f"experiment_{identity_key[3:7]}_*.json"))
            if identity_key not in historical and len(paths) == 1:
                ref = retain(paths[0], raw, reference(paths[0])["sha256"])
                historical[identity_key] = ref
                refs.append(ref)
    health = history.health(ROOT)
    require(all(r["passed"] for r in health["checks"]), "historical_health_custody")
    for r in health["references"]:
        refs.append(retain(ROOT / r["path"], raw, r["sha256"]))
    for name in ["research-roadmap.yaml", "research-complete.yaml", "ops/exclusion_manifest.yaml"]:
        refs.append(
            retain(
                ROOT / name,
                raw,
                AUTHORITY_PINS[1]
                if name == "research-roadmap.yaml"
                else reference(ROOT / name)["sha256"],
            )
        )
    code = [retain(Path(r["path"]), raw, r["sha256"]) for r in manifest["code_test_hashes"]]
    config = manifest["coverage_config_reference"]
    code.append(dict(retain(Path(config["path"]), raw, config["sha256"]), executed_source=False))
    with binding(), old.binding(), lifecycle.binding(), oldest.binding():
        receipt = runner.run_check(
            CommandSpec(
                "publication_gate",
                (str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"),
                "historical_FoVer",
                60,
            ),
            raw / "publication",
            manifest,
        )
    require(receipt["passed"] and receipt["normal_exit"], "publication_gate_execution")
    gate = json.loads(Path(receipt["log_path"]).read_bytes())
    refs.append(reference(Path(receipt["log_path"])))
    for name in [
        "ops/publication_gate_state.json",
        "docs/technical-report.md",
        "docs/arxiv-paper/main.tex",
        "results/" + gate["gates"]["G1"]["source"],
    ]:
        refs.append(retain(ROOT / name, raw, reference(ROOT / name)["sha256"]))
    work = dict(
        raw=str(raw.absolute()),
        private=str(private),
        run_date=run_date,
        started_ns=started,
        frozen_ns=frozen,
        ended_ns=time.monotonic_ns(),
        validation_scope_manifest=manifest,
        preconditions=preconditions,
        tasks=branch["full_task_objects"],
        contract=bound,
        branch_reference=branch_ref,
        branch_primitive_reference=branch_primitive_ref,
        references=refs,
        code=code,
        values=values,
        scheduler_lines=lines,
        history=historical,
        repository_health=health,
        primitive_references=primitives,
        publication_gate=gate,
        publication_receipts=[receipt],
        checks=checks,
        validation=dict(receipts=[], coverage=dict(complete=False)),
    )
    work["slots"] = slots(work)
    atomic_json(raw / "work.json", work)
    progress("preconditions_after", 14, 0)
    return work


def reduce(work: Json) -> Json:
    """Recount available units; unavailable paired predictions remain unmeasured."""
    branch = load(work["branch_reference"])
    for task_id, ref in work["primitive_references"].items():
        require(
            ref["sha256"] == work["values"][task_id]["original"]["raw_shard_hashes"][0]["sha256"],
            "primitive_binding",
        )
    result: Json = {}
    for name, policy in [
        ("seven_bound_rows", "seven_bound_contract"),
        ("ten_bound_rows", "ten_bound_contract"),
    ]:
        require(not work["slots"][9]["qualified"], "registered_raw_reducer_required")
        result[policy] = branch["frozen_science"][policy]
        result[name] = [
            dict(
                bound_id=f"{name}:{i + 1}",
                **bound,
                value=None,
                numerator=None,
                denominator=0,
                observed="missing_file",
                status="unmeasured",
                passed=False,
                upstream=work["tasks"][9]["id"],
                path=work["tasks"][9]["deliverable"],
            )
            for i, bound in enumerate(result[policy]["bounds"])
        ]
    require(not work["slots"][10]["qualified"], "registered_human_reducer_required")
    result["four_bound_rows"] = [
        dict(
            bound_id=f"human:{i + 1}",
            metric=metric,
            control=control,
            criterion=criterion,
            value=None,
            numerator=None,
            denominator=0,
            observed="missing_file",
            status="unmeasured",
            passed=False,
            upstream=work["tasks"][10]["id"],
            path=work["tasks"][10]["deliverable"],
            bootstrap_draws=10000,
            tail_probability=0.0125,
        )
        for i, (metric, control, criterion) in enumerate(
            [
                ("brier_improvement", "prevalence", "lower>0"),
                ("brier_improvement", "converged_linear", "lower>0"),
                ("accepted_error_delta_50", "converged_linear", "upper<=0"),
                ("auroc_improvement", "converged_linear", "lower>0"),
            ]
        )
    ]
    result["fresh_source_summary"] = dict(status="missing_file", completed_count=0)
    entry = work["values"].get(work["tasks"][1]["id"])
    if entry and work["slots"][1]["qualified"]:
        primitive = load(work["primitive_references"][work["tasks"][1]["id"]])
        roster, ledger = primitive["result"]["graph_manifest"], primitive["result"]["proposal_rows"]
        counts = Counter(r["status"] for r in ledger)
        require(
            len({r["unit_id"] for r in roster})
            == len(roster)
            == counts["accepted"]
            == entry["original"]["completed_count"],
            "roster_primitive_counts",
        )
        require(len(ledger) == entry["original"]["proposal_count"], "proposal_primitive_counts")
        result["fresh_source_summary"] = dict(
            status=entry["original"]["honest_verdict"],
            intended_count=1024,
            completed_count=len(roster),
            censored_count=1024 - len(roster),
            proposal_count=len(ledger),
            proposal_status_counts=dict(counts),
            fresh_source_ready_score=entry["original"]["fresh_source_ready_score"],
            primitive_reference=work["primitive_references"][work["tasks"][1]["id"]],
            fixed_proposal_cap=100000,
            exhausted=len(ledger) >= 100000,
        )
    result["human_authority_summary"] = dict(status="missing_file", human_source_ready_score=0)
    entry = work["values"].get(work["tasks"][2]["id"])
    if entry and entry["reference"]["sha256"] == PINS[9056]:
        primitive = load(work["primitive_references"][work["tasks"][2]["id"]])
        reduction = primitive["reduction"]
        rows = reduction["rows"]
        groups = human.groups(reduction["records"])
        require(
            groups == reduction["group_rows"] and rows == entry["original"]["rows"],
            "human_primitive_rows",
        )
        require(
            len({r["unit_id"] for r in rows}) == len(rows) == entry["original"]["completed_count"],
            "human_primitive_counts",
        )
        result["human_authority_summary"] = dict(
            status=entry["original"]["honest_verdict"],
            completed_count=len(rows),
            candidate_group_count=len(groups),
            authority_supported_response_count=sum(r["numerator"] for r in rows),
            human_source_ready_score=0,
            source_family_closure=reduction["source_family_closure"],
            original_criteria=entry["original"]["criterion_mapping"],
            failed_gate_operands=[g for g in reduction["gate_check_summary"] if not g["passed"]],
            protected_outcome_values_projected=False,
        )
    preserved = branch["preserved_v772_outcomes"]
    result["historical_fresh_source_summary"] = preserved["fresh_source_summary"]
    result["natural_feasibility_summary"] = preserved["natural_feasibility_summary"]
    result["cost_quality_summary"] = dict(
        status="missing_file",
        measured_process_units=0,
        cost_ratio_upper95=None,
        demand_measured=False,
        hardware_deployment_ready=False,
        path=work["tasks"][11]["deliverable"],
    )
    arc_ref = work["branch_primitive_reference"]
    require(arc_ref["sha256"] == branch["raw_shard_hashes"][0]["sha256"], "arc_primitive_binding")
    arc = load(arc_ref)["arc"]
    support = arc_support(arc["new_event_rows"])
    require(
        support == arc["support"] == branch["arc_frontier"]["support"]
        and canonical_hash(arc["new_event_rows"]) == arc["delta_sha256"],
        "arc_primitive_support",
    )
    result["arc_summary"] = dict(
        support,
        source_event_count=len(arc["new_event_rows"]),
        source_reference=arc_ref,
        frontier_delta_rows=branch["frontier_delta_rows"],
        qualified_association=False,
        new_gameplay_executed=False,
    )
    result["archive_lag"] = branch["archive_lag"]
    return result


def build(work: Json) -> Json:
    """Complete owned accounting can qualify while the external science stays blocked."""
    with binding():
        value = _build(work)
    value.update(
        experiment_id=9067,
        intended_count=14,
        random_seed=7739067,
        claim_scope="Fourteen administrative outcomes; seven graph, ten learning and four human bounds unmeasured. No new oracle-distinct, cost or deployment benefit.",
        paper_readiness_scope="V773 combined science; historical FoVer gates remain separate.",
        unmet_gates=[
            "V773_qualified_graph_and_learning_predictions",
            "V773_original_human_authority",
            "V773_complete_service_cost",
        ],
        methodology_note="Authenticate fourteen complete task objects and exact original or scheduler bytes. Independently recount the fresh prospective roster and original human authority records. Preserve prior 414/1024 construction, failed Mu-SHROOM authority, archive lag and source-code drift separately. Seven graph, ten learning and four human criteria remain unmeasured without paired predictions and authorized targets. No model, generation, fit, benchmark or game is run; upstream model receipts are references only. Frozen scoped normal exits, zero-exclusion statement coverage, foreign CLI replay and unchanged scanners qualify accounting only.",
    )
    value["slot_rows"][-1]["experiment_id"] = 9067
    value["continuation_evidence"] += [
        dict(
            scope="MDH_400_group_human_authority",
            status="closed_unchanged",
            next_evidence="Changed original release bytes with independent individual judgments, original factual criterion and authenticated unexposed 400-group authority; another download is insufficient.",
        ),
        dict(
            scope="fixed_100000_proposal_continuation",
            status="closed_if_exhausted",
            next_evidence="An exhausted continuation requires a new preregistered mechanism, never another budget increase. Current roster completed before the cap.",
        ),
    ]
    value["cited_upstream_artifacts"] += value["source_artifact_hashes"]
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles"],
        "Original authenticated operands bind independent recounts. Complete accounting grants no scientific benefit or inherited model-call credit.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild every field from pinned operands so rehashing cannot create a win."""
    try:
        value = json.loads(path.read_bytes())
        require(checksum(value) == value["reproducibility_checksum"], "checksum_drift")
        work = load(value["work_reference"])
        for ref in work["references"] + work["code"]:
            require(reference(Path(ref["path"]))["sha256"] == ref["sha256"], "operand_drift")
        for ref in work["code"]:
            if ref.get("executed_source", True):
                require(
                    reference(Path(ref["original_path"]))["sha256"] == ref["sha256"],
                    "executed_code_drift",
                )
        require(work["branch_reference"]["sha256"] == PINS[9054], "branch_pin")
        require(work["contract"] == contract(work["contract"]), "authority_drift")
        log = next(
            r
            for r in work["references"]
            if r.get("original_path") == str(ROOT / "ops/conductor-log.md")
        )
        require(
            log["sha256"] == LOG_PIN
            and work["scheduler_lines"]
            == Path(log["path"])
            .read_text()
            .split(f"Milestone {MILESTONE} activated", 1)[-1]
            .splitlines(),
            "scheduler_drift",
        )
        require(
            work["validation_scope_manifest"] == scope_manifest(Path(work["private"])),
            "scope_drift",
        )
        for task in work["tasks"][:-1]:
            entry = work["values"].get(task["id"])
            if entry:
                original = load(entry["reference"])
                require(
                    entry["reference"]["sha256"] == PINS.get(int(task["id"][3:7]))
                    and entry["original"] == entry["value"] == original,
                    "producer_drift",
                )
        require(work["slots"] == slots(work), "slot_drift")
        require(value == build(work), "reduction_drift")
        return dict(passed=True, replay_passed=True, slots=14)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        return dict(passed=False, replay_passed=False, reason=str(error))


def validate(work: Json) -> Json:
    """Run the frozen scope once and reopen the actual SQLite statement measurements."""
    with binding():
        return _validate(work)


def terminal(candidate: Path, raw: Path, manifest: Json) -> Json:
    """Unchanged scanners and the real foreign-directory replay qualify publication bytes."""
    with binding():
        return _terminal(candidate, raw, manifest)


def main(argv: list[str] | None = None) -> int:
    """Use the qualified lifecycle, deadlines and atomic publication unchanged."""
    with binding():
        return _main(argv)
