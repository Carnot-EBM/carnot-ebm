"""REQ-REPORT-9040-V771: complete accounting preserves unavailable science."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v770_capstone as old
from carnot.reporting import v771_branch_contract as authority
from carnot.reporting import v771_branch_evidence as history
from carnot.reporting import v770_branch_contract as historical
from carnot.reporting import v769_branch_contract as historical_reader
from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import graph_corpus_b_9031 as graph_b
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.invocation_provenance_9000 import environment
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import compare_contract as compare_roadmap
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v768_capstone import edge

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_9040_v771_capstone", "exp9040-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v771_capstone.py", CLI]
TEST = "tests/python/test_v771_capstone_9040.py"
DIRECT = old.DIRECT
REUSED = list(
    dict.fromkeys(
        [
            *old.REUSED,
            *old.OWNED,
            *authority.OWNED,
            "python/carnot/reporting/graph_corpus_b_9031.py",
            "python/carnot/reporting/graph_corpus_a_8961.py",
            "python/carnot/reporting/graph_source_8931.py",
        ]
    )
)
CHECKS = old.CHECKS
MODEL_SPECS: list[Json] = []
PINS = {
    9027: "sha256:3c233de7321f62d1286e8e94bd045da059bc1d65a33f503b637dbd904d084ff7",
    9028: "sha256:a2cb53fd67a42e3bb9614849c42f27fc1d694a65bdc5d508ac5943a77e774067",
    9029: "sha256:a9ca396d7bc115c221a948bb835a3da3e6221feba16dbdf65bf0779fc19a2120",
    9030: "sha256:9755b2dd55459cc48361deb85cbc2bae16e6fc80f231373d56d7033486033944",
    9031: "sha256:be48f8994445432da78d1daa0b0058501fa227e2d3e49c11392ad62369488367",
    9032: "sha256:59ab0aa09fd6bac94a0c85789dc7c6e228b3dab72180f244e420ca002f608e53",
    9033: "sha256:e8a1bd9080155673562425c1ab97bbf1bd0ab52fa273f25c8fd97f96b2614dc0",
    9039: "sha256:2e2fee2deb989b496d0145e0d0b5a5579702d8d53b680b3b36fb7afddda85326",
}
ALTERNATES = {
    n: f"experiment_{n}_{s}.json"
    for n, s in [
        (9032, "corpus_admission"),
        (9033, "human_label_methods"),
        (9039, "arc_transfer_evidence"),
    ]
}
AUTHORITY_PINS = [
    "sha256:645327851ea953d7cfd0961838dcc36eb7e10dd8dbabd32a81c52adadb76d3fd",
    "sha256:2f18f190413146ad123228723a2e9d0780981a5c69a471473b5d06b757041d42",
]
LOG_PIN = "sha256:ed14a6a417e8d4e254bf59a910aa4da44c7edf0197b76e86fe46362900bfbae6"
checksum = old.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual flushed counts distinguish completed work from a waiting child."""
    print(f"[exp9040] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse frozen mechanics without changing historical producers or validators."""
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
        history=history,
        PINS=PINS,
        AUTHORITY_PINS=AUTHORITY_PINS,
        progress=progress,
        validation_plan=validation_plan,
        reduce=reduce,
        verify_measured=verify_measured,
        measured=measured,
    ):
        yield


_plan, _scope, _reduce = old.validation_plan, old.scope_manifest, old.reduce


def validation_plan(private: Path) -> list[CommandSpec]:
    """Run every original consumer with preserved private historical authority."""
    with binding():
        plan = _plan(private)
    return [
        CommandSpec(
            c.name,
            (
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(private / "coverage.ini"),
                str(ROOT / CLI),
                "--direct-consumers",
                *c.argv[c.argv.index("--noconftest") :],
            )
            if c.name == "direct_consumers"
            else c.argv,
            c.scope,
            600 if c.name == "owned_tests" else c.timeout_s,
        )
        for c in plan
    ]


def scope_manifest(private: Path) -> Json:
    """Freeze exact commands, all assertions and zero-exclusion ownership before reduction."""
    with binding():
        manifest = _scope(private)
    manifest["commands"] = [
        dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s)
        for c in validation_plan(private)
    ]
    return manifest


def receipts_pass(work: Json) -> bool:
    """A passing scalar cannot replace actual current command and log evidence."""
    with binding():
        return old.receipts_pass(work)


def contract(saved: Json) -> Json:
    """Independent pins and full-object equality bind the retained V771 authorities."""
    with binding(), patch.object(old, "compare_contract", compare_contract):
        return old.contract(saved)


def compare_contract(text: str, staged: Json, active: Json, **kwargs: Any) -> Json:
    """Keep the original comparisons while selecting the current fourteen task identities."""
    return compare_roadmap(text, staged, active, milestone=MILESTONE, first_id=9027)


def slots(work: Json) -> list[Json]:
    """Scheduler events retain absence and never invent scientific verdicts."""
    with binding():
        return old.slots(work)


def collect(raw: Path, run_date: str, sources: Path = ROOT / "results") -> Json:
    """Retain authenticated inputs and rejected evidence before any scientific reduction."""
    progress("preconditions_before", 0, 14)
    raw.mkdir(parents=True, exist_ok=True)
    started = time.monotonic_ns()
    private = Path(tempfile.mkdtemp(prefix="carnot-9040-"))
    manifest = scope_manifest(private)
    atomic_json(raw / "validation_scope_manifest.json", manifest)
    frozen = time.monotonic_ns()
    preconditions = environment(raw)
    require(run_date == "20261112", "run_date")
    branch_ref = retain(ROOT / "results/experiment_9027_v771_branch_contract.json", raw, PINS[9027])
    branch = load(branch_ref)
    refs = [branch_ref]
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
    log = retain(ROOT / "ops/conductor-log.md", raw, LOG_PIN)
    refs.append(log)
    lines = (
        Path(log["path"]).read_text().split(f"Milestone {MILESTONE} activated", 1)[-1].splitlines()
    )
    values: Json = {}
    checks = list(preconditions["checks"])
    for index, task in enumerate(branch["full_task_objects"][:-1]):
        n = 9027 + index
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
        progress("producer_authenticated", index + 1, 13 - index)
    historical: Json = {}
    for task in branch["full_task_objects"]:
        for prior in task["prior_failures"]:
            identity = prior["experiment_id"]
            paths = list((ROOT / "results").glob(f"experiment_{identity[3:7]}_*.json"))
            if identity not in historical and len(paths) == 1:
                ref = retain(
                    paths[0],
                    raw,
                    history.PINS.get(str(paths[0].relative_to(ROOT)))
                    or reference(paths[0])["sha256"],
                )
                historical[identity] = ref
                refs.append(ref)
    health = history.health(ROOT)
    require(all(r["passed"] for r in health["checks"]), "historical_health_custody")
    for r in health["references"]:
        refs.append(retain(ROOT / r["path"], raw, r["sha256"]))
    health["current_observations"] = []
    observation = (
        ROOT / "results/raw" / NAME / "repository_health/full_python_suite_once.receipt.json"
    )
    if observation.is_file():
        ref = retain(observation, raw, reference(observation)["sha256"])
        health["current_observations"].append(ref)
        refs.append(ref)
        receipt = load(ref)
        for kind in ["stdout", "stderr"]:
            refs.append(retain(ROOT / receipt[kind + "_path"], raw, receipt[kind + "_sha256"]))
    for name in ["research-complete.yaml", "ops/exclusion_manifest.yaml"]:
        path = ROOT / name
        refs.append(retain(path, raw, reference(path)["sha256"]))
    primitive = None
    primitives: Json = {}
    for identity, entry in values.items():
        value = entry["original"]
        for r in value.get("raw_shard_hashes", [])[:1]:
            saved = retain(Path(r["path"]), raw, r["sha256"])
            refs.append(saved)
            primitives[identity] = saved
            if identity == "exp9029-source-identity-resolution":
                primitive = saved
        for receipt in value.get("validation_receipts", []):
            if not receipt.get("passed", True) and receipt.get("log_sha256"):
                refs.append(retain(ROOT / receipt["log_path"], raw, receipt["log_sha256"]))
    code = [retain(Path(r["path"]), raw, r["sha256"]) for r in manifest["code_test_hashes"]]
    config = manifest["coverage_config_reference"]
    code.append(dict(retain(Path(config["path"]), raw, config["sha256"]), executed_source=False))
    with binding(), old.binding():
        receipt = runner.run_check(
            CommandSpec(
                "publication_gate",
                (sys.executable, "-u", "scripts/publication_gate.py", "--json"),
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
        path = ROOT / name
        refs.append(retain(path, raw, reference(path)["sha256"]))
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
        references=refs,
        code=code,
        values=values,
        scheduler_lines=lines,
        history=historical,
        repository_health=health,
        human_primitive=primitive,
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
    """Independently count metadata groups while unavailable measurements stay unmeasured."""
    result = _reduce(work)
    human = work["values"].get("exp9029-source-identity-resolution", {}).get("original", {})
    if human:
        primitive = load(work["human_primitive"])["reduction"]
        require(
            all(
                primitive[k] == human[k]
                for k in [
                    "rows",
                    "source_group_rows",
                    "unresolved_source_groups",
                    "source_identity_ready_score",
                ]
            ),
            "human_primitive_rows",
        )
        groups = primitive["source_group_rows"]
        require(len({g["group_id"] for g in groups}) == len(groups), "human_source_group_integrity")
        result["human_source_audit"] = dict(
            status=human["honest_verdict"],
            source_group_count=len(groups),
            eligible_source_group_count=sum(g["exclusion_reason"] is None for g in groups),
            original_annotation_count=0,
            source_identity_ready_score=primitive["source_identity_ready_score"],
            unresolved_source_group_count=len(primitive["unresolved_source_groups"]),
            primitive_reference=work["human_primitive"],
            limit="Metadata custody alone cannot establish original human labels, unseen source groups or absence of pretrained-model contamination.",
        )
    result["source_custody_rows"] = [
        dict(
            experiment_id=s["experiment_id"],
            qualified=s["qualified"],
            numerical_ready=work["values"]
            .get(s["task_id"], {})
            .get("original", {})
            .get("shard_ready_score"),
            honest_verdict=s["honest_verdict"],
            reference=s["reference"],
        )
        for s in work["slots"]
        if s["experiment_id"] in [9028, 9030, 9031]
    ]
    for row in result["source_custody_rows"]:
        if row["experiment_id"] == 9028:
            continue
        identity = work["tasks"][row["experiment_id"] - 9027]["id"]
        if identity not in work["primitive_references"]:
            row.update(
                status="missing_file", intended_count=512, completed_count=0, censored_count=512
            )
            continue
        primitive = load(work["primitive_references"][identity])
        rows = (
            primitive["evidence"]["rows"]
            if row["experiment_id"] == 9030
            else graph_b.reduce(primitive["units"], primitive["runtime"]["rows"])["rows"]
        )
        original = work["values"][identity]["original"]["rows"]
        require(
            len(rows) == len({r["unit_id"] for r in rows}) == 512
            and all(
                all(r[k] == observed[k] for k in r)
                for r, observed in zip(rows, original, strict=True)
            ),
            "source_primitive_rows",
        )
        row.update(
            intended_count=len(rows),
            completed_count=sum(r["status"] == "completed" for r in rows),
            censored_count=sum(r["status"] == "censored" for r in rows),
            failed_count=sum(r["status"] == "failed" for r in rows),
            primitive_reference=work["primitive_references"][identity],
        )
    return result


def build(work: Json) -> Json:
    """Complete current accounting can qualify while external science remains blocked."""
    with binding():
        value = old.build(work)
    value.update(
        experiment_id=9040,
        random_seed=7719040,
        v771_science_summary=value.pop("v770_science_summary"),
    )
    value["slot_rows"][-1].update(experiment_id=9040, honest_verdict=value["honest_verdict"])
    for row in value["retirement_scope_rows"]:
        if row["task_id"] == TASK:
            row.update(
                actual_current_verdict=value["honest_verdict"],
                unchanged_verdict=row["actual_prior_verdict"] is not None
                and row["actual_prior_verdict"] == value["honest_verdict"],
            )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles"],
        "Retained primitive bytes bind each claim; administrative custody, original failed health and historical publication gates grant no new scientific benefit.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from authenticated operands so a rehashed aggregate cannot invent readiness."""
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
        require(work["branch_reference"]["sha256"] == PINS[9027], "branch_pin")
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
        for n, task in enumerate(work["tasks"][:-1], 9027):
            entry = work["values"].get(task["id"], {})
            if entry:
                original = load(entry["reference"])
                expected = (
                    original
                    if entry["reference"]["sha256"] == PINS.get(n)
                    else dict(original, flagged_adversarial=True)
                )
                require(
                    entry["original"] == original and entry["value"] == expected, "producer_drift"
                )
        require(work["slots"] == slots(work), "slot_drift")
        require(
            all(work["repository_health"][k] == v for k, v in history.health(ROOT).items()),
            "historical_health_drift",
        )
        require(value == build(work), "reduction_drift")
        return dict(passed=True, replay_passed=True, slots=14)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        return dict(passed=False, replay_passed=False, reason=str(error))


def validate(work: Json) -> Json:
    """Run the frozen scoped checks once and reopen their actual statement measurements."""
    with binding():
        return old.validate(work)


def terminal(candidate: Path, raw: Path, manifest: Json) -> Json:
    """Unchanged scanners and a foreign-directory CLI check the exact candidate bytes."""
    with binding():
        return old.terminal(candidate, raw, manifest)


def direct_consumers(argv: list[str]) -> int:
    """Supply preserved V770 authority privately while every original assertion still runs."""
    import pytest

    class HistoricalInputs:
        @pytest.fixture(autouse=True)
        def historical_authority(self, request: Any) -> Iterator[None]:
            """Only V770 tests receive V770 authority; other consumers keep their own inputs."""
            if request.module.__name__ == "test_v770_branch_contract_9013":
                paths: Json = dict(
                    DESIGN=history.ARCHIVE,
                    ACTIVE=history.QUEUE,
                    STAGED="research-roadmap-v770-private-next.yaml",
                )
                with (
                    patch.multiple(historical, **paths),
                    patch.multiple(historical_reader, **paths),
                ):
                    yield
            else:
                yield

    return int(pytest.main(argv, plugins=[HistoricalInputs()]))


def main(argv: list[str] | None = None) -> int:
    """Expose a primary only after owned checks pass; retain every rejected candidate."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261112"], default="20261112")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--sources", type=Path, default=ROOT / "results")
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--check-receipts", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--direct-consumers", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.from_work and args.output.absolute().is_relative_to(ROOT):
        parser.error("private replay publication requires output outside the checkout")
    try:
        if args.direct_consumers is not None:
            return direct_consumers(args.direct_consumers)
        if args.cold_replay or args.check_receipts:
            report = (
                replay(args.cold_replay)
                if args.cold_replay
                else dict(passed=receipts_pass(json.loads(args.check_receipts.read_bytes())))
            )
            print(json.dumps(report), flush=True)
            return 0 if report["passed"] else 1
        raw = args.output.absolute().parent / "raw" / NAME / "invocations" / str(time.time_ns())
        if args.from_work:
            work = json.loads(args.from_work.read_bytes())
            work["raw"] = str(raw)
        else:
            work = collect(raw, args.date, args.sources)
            progress("reductions_before", 0, 14)
            reduce(work)
            progress("reductions_after", 14, 0)
            require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 3300, "measurement_deadline")
            work["validation"] = validate(work)
        work["ended_ns"] = time.monotonic_ns()
        work["phase_spans"] = [
            dict(phase="measurement", duration_s=(work["frozen_ns"] - work["started_ns"]) / 1e9),
            dict(
                phase="reconciliation_and_validation",
                duration_s=(work["ended_ns"] - work["frozen_ns"]) / 1e9,
            ),
        ]
        atomic_json(raw / "work.json", work)
        candidate = raw / "candidate.json"
        value = build(work)
        atomic_json(candidate, value)
        require(value["required_checks_passed"], "owned_validation_missing_or_failed")
        report = terminal(candidate, raw / "preflight_checks", work["validation_scope_manifest"])
        work["validation"].update(
            terminal_failed=not report["passed"], terminal_receipts=report["receipts"]
        )
        atomic_json(raw / "work.json", work)
        value = build(work)
        atomic_json(candidate, value)
        require(report["passed"], "owned_terminal_checks")
        require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 4800, "hard_deadline")
        published = publish_primary(
            args.output,
            value,
            lambda p: terminal(p, raw / "publication_checks", work["validation_scope_manifest"]),
        )
        progress("complete", 14, 0)
        print(
            json.dumps(
                dict(
                    published,
                    honest_verdict=value["honest_verdict"],
                    reconciliation_ready_score=value["reconciliation_ready_score"],
                )
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, reason=str(error))), flush=True)
        return 1
