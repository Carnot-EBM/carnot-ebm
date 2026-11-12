"""REQ-REPORT-9068-V774: exact dispatch custody grants no scientific benefit."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v773_branch_contract as previous
from carnot.reporting import v773_branch_evidence as evidence
from carnot.reporting import v770_branch_evidence as arc_history
from carnot.reporting import scoped_provenance_9014 as validation
from carnot.reporting.invocation_provenance_9000 import environment
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from scripts import research_conductor as conductor
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = previous.ROOT, previous.DESIGN, previous.ACTIVE, previous.STAGED
NAME, TASK, MILESTONE = (
    "experiment_9068_v774_dispatch_contract",
    "exp9068-dispatch-contract",
    "2026.11.774",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v774_dispatch_contract_9068.py"
OWNED = ["python/carnot/reporting/v774_dispatch_contract.py", CLI]
DIRECT = [*previous.DIRECT, "tests/python/test_conductor_deliverable_status.py"]
REUSED = list(
    dict.fromkeys(
        [
            *previous.REUSED,
            *previous.OWNED,
            *validation.REUSED,
            previous.TEST,
            "tests/python/test_v773_capstone_9067.py",
            "tests/python/test_prospective_roster_9055.py",
            "scripts/research_conductor.py",
            "scripts/experiment_template.py",
            "scripts/conductor_exclusion_manifest.json",
            "ops/exclusion_manifest.yaml",
        ]
    )
)
READINESS = ["contract_ready_score", "frozen_methods_ready_score", "arc_new_outcomes_ready_score"]
LOG = "ops/conductor-log.md"
ARCHIVE = "openspec/change-proposals/research-roadmap-v773-preserved-20261112.md"
QUEUE = "openspec/change-proposals/research-roadmap-v773-queue-preserved-20261112.yaml"
BRANCH = "results/experiment_9054_v773_branch_contract.json"
ROSTER = "results/experiment_9055_v773_prospective_roster.json"
CAPSTONE = "results/experiment_9067_v773_capstone.json"
HEALTH = ROOT / "results/raw" / NAME / "development/global_python_suite.receipt.json"
DESIGN_PIN = "sha256:9737d705002f952c03fc9f0a57e3404e8ad8efc5a7119af10d891d967fad8b10"
PIN = "sha256:497b564087d45d7776ba6bd463b41847608bf98a7354cff6617849ebbfe19174"
PINS = dict(
    evidence.PINS,
    **{
        DESIGN: DESIGN_PIN,
        ARCHIVE: "sha256:96163ef0782d8c6125561eec52e89d4d8982a8829bc0dd41704a9ee5ecac57c4",
        QUEUE: "sha256:a18957b6f7490cc3b3a2506ff98d4e656d606e99f19ef637458372c49ff507b8",
        BRANCH: "sha256:9d2757fa9ba48ed23457d667cb92bc88187b1466fbdb948caab4d84e8155fe74",
        ROSTER: "sha256:80aa684f1cc271897823983b66bc2d673073223413233438ad135a81edfbc867",
        CAPSTONE: "sha256:eb76ada3c0777398b8ac9c6910a59678400ab3b7deb7f8eb92048412c14bcf75",
    },
)
reference, checksum, check = previous.reference, previous.checksum, evidence.check
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured boundaries while existing child supervision supplies heartbeats."""
    print(f"[exp9068] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse qualification without changing any historical producer or assertion."""
    with patch.multiple(
        validation,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        OWNED=OWNED,
        TEST=TEST,
        DIRECT=DIRECT,
        REUSED=REUSED,
        progress=progress,
    ):
        yield


def scope_manifest(private: Path) -> Json:
    """Freeze exact commands and original test bytes before looking at measurements."""
    with binding():
        return validation.scope_manifest(private)


def authority(root: Path) -> Json:
    """Independent prose and full-object pins bind routing and prompts without inventing a table."""
    tasks, rows, keys, checks = [], [], [], []
    mode = "unavailable"
    try:
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        mode = "activated" if active["milestone"] == MILESTONE else "staged"
        path = root / (ACTIVE if mode == "activated" else STAGED)
        queue = Roadmap.model_validate(yaml.safe_load(path.read_bytes()))
        tasks = yaml.safe_load(path.read_bytes())["tasks"]
        checks += [
            check(
                root / DESIGN,
                "sha256",
                DESIGN_PIN,
                reference(root / DESIGN)["sha256"] or "missing_file",
                "design_bytes",
            ),
            check(path, "full_task_digest", PIN, canonical_hash(tasks), "full_queue"),
            check(path, "milestone", MILESTONE, queue.milestone, "lifecycle"),
        ]
        prior = {
            p[2].strip()
            for line in (root / LOG).read_text().splitlines()
            if len(p := line.split("|")) >= 4 and p[3].strip() in {"OK", "FLAGGED"}
        }
        titles = [t["title"][:50].strip() for t in tasks]
        for i, task in enumerate(tasks):
            key = titles[i]
            keys.append(
                dict(
                    unit_id=task["id"],
                    title=task["title"],
                    dispatch_key=key,
                    queue_occurrences=titles.count(key),
                    prior_completion=key in prior,
                    passed=titles.count(key) == 1 and key not in prior,
                )
            )
        checks.append(
            check(
                path,
                "unique_uncompleted_dispatch_keys",
                True,
                len(tasks) == 14 and all(r["passed"] for r in keys),
                "dispatch_keys",
            )
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
        checks.append(
            check(
                root / ACTIVE,
                "full_task_objects",
                "14 exact ordered objects",
                "missing_file"
                if isinstance(error, FileNotFoundError)
                else "missing_field"
                if isinstance(error, KeyError)
                else str(error),
                "authority",
            )
        )
    ready = all(r["passed"] for r in checks)
    for i in range(14):
        task = tasks[i] if i < len(tasks) else {}
        identity = task.get("id", f"exp{9068 + i}-missing")
        matched = ready and identity.startswith(f"exp{9068 + i}-")
        rows.append(
            dict(
                unit_id=identity,
                group_id=identity,
                arm="full_task_contract",
                condition="exact_authority",
                status="completed" if matched else "censored",
                metric="ordered_full_object_match",
                value=int(matched),
                numerator=int(matched),
                denominator=1,
                exclusion_reason=None,
                task=task,
                task_sha256=canonical_hash(task),
                matched=matched,
            )
        )
    return dict(
        ready=ready and all(r["matched"] for r in rows),
        tasks=tasks,
        rows=rows,
        title_key_rows=keys,
        checks=checks,
        comparison_mode=mode,
    )


def scheduler_replay(root: Path, tasks: list[Json], directory: Path) -> list[Json]:
    """Keep the selector unchanged; only private queue, log and primary bytes differ."""
    historical = yaml.safe_load((root / QUEUE).read_bytes())["tasks"]
    pairs = [
        ([t for t in historical if t["id"].startswith(("exp9058-", "exp9059-"))], "old"),
        ([t for t in tasks if t["id"] in {"exp9069-source-custody", "exp9071-collect-b"}], "new"),
    ]
    rows = []
    for pair, label in pairs:
        a, b = pair
        private = directory / label
        roadmap = private / ACTIVE
        artifact = private / a["deliverable"]
        log = f"| date | {a['title'][:50].strip()} | OK | private A only |\n"
        files = {
            artifact: json.dumps(
                dict(
                    experiment_id=int(a["id"][3:7]),
                    honest_verdict="complete_null_fixture",
                    verdict_class="null",
                )
            ),
            roadmap: yaml.safe_dump(dict(milestone=MILESTONE, tasks=pair)),
            private / "log.md": log,
            private / "scripts/conductor_exclusion_manifest.json": (
                root / "scripts/conductor_exclusion_manifest.json"
            ).read_text(),
        }
        for path, content in files.items():
            if path.exists():
                if path.read_text() != content:
                    raise ValueError("fixture_drift")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
        with patch.multiple(
            conductor,
            PROJECT_ROOT=private,
            ROADMAP_FILE=roadmap,
            RESEARCH_TASKS=[],
            _tasks_loaded=False,
            _roadmap_mtime=0,
            _EXCLUSION_MANIFEST=None,
            _EXCLUSION_MANIFEST_MTIME=0,
        ):
            reason = conductor._artifact_unfinished_reason(b)
            selected = conductor.pick_next_task(log)
        selected_id = selected["id"] if selected else None
        success = selected_id == b["id"]
        rows.append(
            dict(
                condition=label,
                expected_id=b["id"],
                selected_id=selected_id,
                B_primary_absent=not (private / b["deliverable"]).exists(),
                missing_primary_reason=reason,
                expected_failure_observed=label == "old" and not success,
                passed=success if label == "new" else not success,
                fixture_only=True,
                references=[reference(roadmap), reference(artifact), reference(private / "log.md")],
            )
        )
    return rows


def reduce(work: Json) -> Json:
    """Authenticate retained methods without loading models or opening protected TEST truth."""
    root = Path(work["retained"])
    contract = authority(root)
    science = evidence.science(root)
    history = evidence.history(root)
    roster = json.loads((root / ROSTER).read_bytes())
    capstone = json.loads((root / CAPSTONE).read_bytes())
    roster_checks = [
        check(root / ROSTER, field, expected, roster.get(field, "missing_field"), "roster")
        for field, expected in [
            ("fresh_source_ready_score", 1),
            ("completed_count", 1024),
            ("intended_count", 1024),
            ("proposal_count", 47987),
        ]
    ]
    with patch.multiple(
        arc_history, BRANCH=BRANCH, PINS=dict(arc_history.PINS, **{BRANCH: PINS[BRANCH]})
    ):
        arc = arc_history.frontier(root)
    arc["baseline_experiment_id"] = 9054
    arc["ledger_reference"] = reference(root / evidence.LEDGER)
    if not arc["ready"]:
        arc["checks"].append(
            check(
                root / evidence.LEDGER,
                "new_uncensored_episode_count",
                ">=30; >=3 families; both arms in every held family",
                arc["support"]["new_uncensored_episode_count"],
                "optional_arc_support",
            )
        )
    archived = yaml.safe_load((root / "research-complete.yaml").read_bytes())["milestones"]
    return dict(
        contract=contract,
        science=science,
        history=history,
        roster_checks=roster_checks,
        arc=arc,
        scheduler_replay_rows=scheduler_replay(
            root, contract["tasks"], Path(work["scheduler_dir"])
        ),
        archive_lag=dict(
            observed_archive_end=archived[-1]["id"],
            planning_archive_end="2026.11.772",
            completed_scheduling_frontier="2026.11.773",
            archive_lagged=not any(m["id"] == "2026.11.773" for m in archived),
            archive_reference=reference(root / "research-complete.yaml"),
            archive_modified=False,
        ),
        preserved_v773=dict(
            design_reference=reference(root / ARCHIVE),
            queue_reference=reference(root / QUEUE),
            capstone_reference=reference(root / CAPSTONE),
            honest_verdict=capstone["honest_verdict"],
            missing_B_is_proof_of_no_attempt=False,
            fresh_source_summary=capstone["fresh_source_summary"],
            scientific_result_from_scheduler=False,
        ),
    )


def collect(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Stop before reduction if any exact input, dependency or private storage check fails."""
    started = time.monotonic_ns()
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    private.mkdir(parents=True, mode=0o700, exist_ok=True)
    progress("preconditions_before")
    manifest = scope_manifest(private)
    env = environment(private)
    checks = [
        *env["checks"],
        check(raw, "run_date", "20261112", run_date, "date"),
        check(raw, "private_storage_mode", 0o700, raw.stat().st_mode & 0o777, "private_storage"),
    ]
    refs = evidence.operands(root)
    for name in [ROSTER, BRANCH]:
        if reference(root / name)["sha256"] == PINS[name]:
            value = json.loads((root / name).read_bytes())
            refs += value["raw_shard_hashes"]
    pins = dict(PINS)
    pins.update({str(Path(r["path"]).relative_to(ROOT)): r["sha256"] for r in refs})
    names = list(
        dict.fromkeys(
            [
                *pins,
                ACTIVE,
                LOG,
                evidence.LEDGER,
                "research-complete.yaml",
                *manifest_path_names(manifest),
            ]
        )
    )
    snapshots = []
    for i, name in enumerate(names):
        source, target = root / name, raw / "authority" / name
        # Code belongs to this checkout even when a private fixture supplies its inputs.
        if name in manifest_path_names(manifest):
            source = ROOT / name
        ref = reference(source)
        checks.append(
            check(
                source,
                "sha256",
                pins.get(name, ref["sha256"] or "required_file"),
                ref["sha256"] or "missing_file",
                "source_bytes",
            )
        )
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            snapshots.append(dict(reference(target), source=ref))
            checks.append(
                check(
                    target,
                    "snapshot_sha256",
                    ref["sha256"],
                    reference(target)["sha256"],
                    "snapshot_bytes",
                )
            )
        if i % 100 == 0:
            progress("authenticate_and_freeze", i + 1, len(names) - i - 1)
    staged = root / STAGED
    if staged.is_file():
        target = raw / "authority" / STAGED
        target.write_bytes(staged.read_bytes())
        snapshots.append(dict(reference(target), source=reference(staged)))
    selected = authority(raw / "authority")
    checks += selected["checks"]
    work: Json = dict(
        raw=str(raw),
        retained=str(raw / "authority"),
        private=str(private),
        scheduler_dir=str(raw / "scheduler"),
        start_ns=started,
        frozen_ns=time.monotonic_ns(),
        run_date=run_date,
        snapshots=snapshots,
        validation_scope_manifest=manifest,
        preconditions_checked=dict(env, checks=checks),
        reduction={},
        validation=dict(receipts=[], coverage=dict(complete=False, percent=0)),
    )
    progress("preconditions_after", len(checks), sum(not r["passed"] for r in checks))
    if all(r["passed"] for r in checks):
        progress("reduction_before")
        work["reduction"] = reduce(work)
        work["repository_health"] = evidence.health(raw / "authority")
        current = (
            json.loads(HEALTH.read_bytes())
            if HEALTH.is_file()
            else dict(passed=False, reason="not_recorded")
        )
        health_refs = [reference(HEALTH)] if HEALTH.is_file() else []
        health_refs += [
            dict(path=current[k], sha256=current[k.replace("_path", "_sha256")])
            for k in ["stdout_path", "stderr_path"]
            if k in current
        ]
        work["repository_health"]["current_suite"] = dict(
            result=current, references=health_refs, affects_owned_readiness=False
        )
        progress("reduction_after", 14, 0)
    work["end_ns"] = time.monotonic_ns()
    atomic_json(raw / "work.json", work)
    return work


def manifest_path_names(manifest: Json) -> list[str]:
    """Retain source and original test bytes named by the frozen qualification scope."""
    return [str(Path(r["path"]).relative_to(ROOT)) for r in manifest["code_test_hashes"]]


def build(work: Json) -> Json:
    """Keep three gates independent; current owned failure disqualifies every readiness claim."""
    r, valid = work["reduction"], work["validation"]
    preconditions = work["preconditions_checked"]["checks"]
    blocked = not all(c["passed"] for c in preconditions)
    with binding():
        checked = validation.receipts_pass(work) and verify_measured(valid["coverage"], OWNED)
    contract = bool(
        checked
        and r
        and r["contract"]["ready"]
        and all(row["passed"] for row in r["scheduler_replay_rows"])
    )
    methods = bool(
        checked
        and r
        and r["science"]["frozen_graph_authority_ready_score"]
        and r["science"]["online_protocol_ready_score"]
        and r["history"]["methods_provenance_ready"]
        and all(c["passed"] for c in r["roster_checks"])
    )
    cls = "blocked" if blocked else "null" if checked and contract else "disqualified"
    rows = (
        r["contract"]["rows"]
        if r
        else [
            dict(
                unit_id=f"exp{9068 + i}",
                group_id=f"exp{9068 + i}",
                arm="full_task_contract",
                condition="preconditions_blocked",
                status="censored",
                metric="ordered_full_object_match",
                value=None,
                numerator=0,
                denominator=0,
                exclusion_reason=None,
            )
            for i in range(14)
        ]
    )
    end = max(
        [work["end_ns"], *[c.get("ended_monotonic_ns", work["end_ns"]) for c in valid["receipts"]]]
    )
    failed = next((c["check"] for c in preconditions if not c["passed"]), "dispatch_contract")
    value: Json = dict(
        experiment_id=9068,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{cls}_{failed if blocked else 'v774_dispatch_contract'}",
        verdict_class=cls,
        verifier_is_oracle=False,
        claim_scope="Administrative dispatch custody and historical fitted-method qualification only; no new scientific benefit.",
        exposure_scope="Metadata and previously qualified development states; protected TEST outcomes stay sealed.",
        independent_generalization_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(model_loads=0, generation_calls=0, llm=0, numerical_head=0),
        numerical_training_calls=0,
        required_checks_passed=bool(checked),
        flagged_adversarial=not checked and not blocked,
        gate_check_summary=preconditions
        + [
            check(
                Path(work["raw"]) / "work.json",
                "required_checks_passed",
                True,
                bool(checked),
                "current_qualification",
            )
        ]
        + (
            r["science"]["checks"]
            + r["history"]["checks"]
            + r["roster_checks"]
            + r["arc"]["checks"]
            if r
            else []
        ),
        rows=rows,
        full_contract_rows=rows,
        full_task_objects=r["contract"]["tasks"] if r else [],
        intended_count=14,
        completed_count=sum(row["status"] == "completed" for row in rows),
        failed_count=0,
        censored_count=sum(row["status"] == "censored" for row in rows),
        excluded_count=0,
        independent_count=0,
        contract_ready_score=int(contract),
        frozen_methods_ready_score=int(methods),
        arc_new_outcomes_ready_score=int(bool(checked and r and r["arc"]["ready"])),
        title_key_rows=r["contract"]["title_key_rows"] if r else [],
        scheduler_replay_rows=r.get("scheduler_replay_rows", []),
        authority_manifest=work["snapshots"],
        authority_comparison_mode=r["contract"]["comparison_mode"] if r else "unavailable",
        design_sha256=DESIGN_PIN,
        full_task_digest=PIN,
        design_contract_form="independently pinned prose bytes and complete queue; no embedded machine table",
        archive_lag=r.get("archive_lag", {}),
        frontier_delta_rows=r["arc"]["new_event_rows"] if r else [],
        arc_frontier=r.get("arc", {}),
        frozen_science=r.get("science", {}),
        preserved_v773=r.get("preserved_v773", {}),
        preconditions_checked=work["preconditions_checked"],
        validation_scope_manifest=work["validation_scope_manifest"],
        validation_receipts=valid["receipts"],
        owned_statement_coverage=valid["coverage"],
        repository_health=work.get(
            "repository_health", dict(passed=False, reason="preconditions_blocked")
        ),
        repository_health_passed=False,
        duration_s=(end - work["start_ns"]) / 1e9,
        random_seed=7749068,
        phase_spans=[
            dict(
                name="authentication_and_reduction",
                start_ns=work["start_ns"],
                end_ns=work["end_ns"],
                duration_s=(work["end_ns"] - work["start_ns"]) / 1e9,
            ),
            dict(
                name="validation",
                start_ns=work["end_ns"],
                end_ns=end,
                duration_s=(end - work["end_ns"]) / 1e9,
            ),
        ],
        raw_shard_hashes=[
            reference(Path(work["raw"]) / "work.json"),
            *work["snapshots"],
            *work.get("repository_health", {}).get("current_suite", {}).get("references", []),
        ],
        code_config_hashes=work["validation_scope_manifest"]["code_test_hashes"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(Path(n).name.split("_")[1]),
                sha256=p,
                fields_imported=["frozen_methods_and_original_custody_only"],
            )
            for n, p in PINS.items()
            if n.startswith("results/")
        ],
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Pinned original bytes and current measured checks certify custody only; optional science, historical failure and exposed identities remain separate.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def validate(work: Json) -> Json:
    """Run the frozen checks once, then retain actual SQLite statement measurements."""
    receipts = []
    private, raw = Path(work["private"]), Path(work["raw"])
    with binding(), patch.dict("os.environ", COVERAGE_RCFILE=str(private / "coverage.ini")):
        for command in work["validation_scope_manifest"]["commands"]:
            spec = CommandSpec(
                command["name"], tuple(command["argv"]), "current_scoped", command["timeout_s"]
            )
            receipts.append(
                validation.run_check(spec, raw / "validation", work["validation_scope_manifest"])
            )
    coverage = measured([private / ".coverage"], OWNED, raw / "coverage")
    coverage.pop("rows", None)
    return dict(receipts=receipts, coverage=coverage)


def replay(path: Path) -> Json:
    """Rebuild summaries from frozen authority; rehashing changed aggregates is insufficient."""
    try:
        value = json.loads(path.read_bytes())
        for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        if work["reduction"] and work["reduction"] != reduce(work):
            raise ValueError("reduction_drift")
        if work["validation_scope_manifest"] != scope_manifest(Path(work["private"])):
            raise ValueError("scope_drift")
        if value != build(work) or value["reproducibility_checksum"] != checksum(value):
            raise ValueError("aggregate_drift")
        return dict(passed=True, replay_passed=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(passed=False, replay_passed=False, reason=str(error))


def terminal(candidate: Path, raw: Path, manifest: Json) -> Json:
    """Unchanged scanners and an actual foreign-directory child qualify publication bytes."""
    with binding():
        return validation.terminal(candidate, raw, manifest)


def main(argv: list[str] | None = None) -> int:
    """Keep private qualification outputs isolated and publish only checked primary bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261112")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = replay(args.cold_replay)
        print(json.dumps(result), flush=True)
        return int(not result["passed"])
    if args.date != "20261112":
        parser.error("this contract requires --date 20261112")
    output = args.output.absolute()
    if args.from_work and output.is_relative_to(ROOT):
        parser.error("private work publication requires output outside the checkout")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    if args.from_work:
        work = json.loads(args.from_work.read_bytes())
        work["raw"] = str(raw)
    else:
        private = Path(tempfile.mkdtemp(prefix="carnot9068-", dir="/var/tmp"))
        work = collect(args.root, raw, private, args.date)
        if work["reduction"]:
            if (time.monotonic_ns() - work["start_ns"]) / 1e9 >= 3300:
                raise TimeoutError("measurement_deadline")
            progress("validation_before")
            work["validation"] = validate(work)
            progress("validation_after")
    atomic_json(raw / "work.json", work)
    value = build(work)
    atomic_json(raw / "candidate.json", value)
    if not args.from_work and work["reduction"] and not value["required_checks_passed"]:
        progress("publication_disqualified", 0, 1)
        return 1
    progress("publication_before")
    receipt = publish_primary(
        output, value, lambda p: terminal(p, raw / "terminal", work["validation_scope_manifest"])
    )
    atomic_json(raw / "publication.json", receipt)
    progress("publication_after", 1, 0)
    print(json.dumps(dict(receipt, honest_verdict=value["honest_verdict"])), flush=True)
    return 0
