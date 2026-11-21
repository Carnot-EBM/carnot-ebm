"""REQ-REPORT-9124-V778: retain authentic methods without borrowing scientific credit.

Only immutable historical snapshots feed the reduction. The established contract
lifecycle supplies command supervision, measured coverage and atomic publication.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import csv
import io
import argparse
import json
import tempfile
import time
from pathlib import Path
import re
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import scoped_provenance_9014 as validation
from carnot.reporting import v776_contract as previous
from carnot.reporting import v777_contract as old
from carnot.reporting import v776_capstone_evidence as history
from carnot.reporting import v775_contract as collection
from carnot.reporting import v775_capstone_evidence as inherited_history
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import parse_design, TABLE_FIELDS
from carnot.reporting.v749_contract_methods import encode
from scripts.roadmap_schema import Roadmap
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED, LOG = (
    previous.ROOT,
    previous.DESIGN,
    previous.ACTIVE,
    previous.STAGED,
    previous.LOG,
)
NAME, TASK, MILESTONE = "experiment_9124_v778_contract", "exp9124-contract", "2026.11.778"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v778_contract_9124.py"
OWNED = ["python/carnot/reporting/v778_contract.py", CLI]
DIRECT = previous.DIRECT
REUSED = list(
    dict.fromkeys(
        [
            *previous.REUSED,
            *previous.OWNED,
            *old.OWNED,
            "python/carnot/reporting/v777_capstone_evidence.py",
            "python/carnot/reporting/v776_capstone_evidence.py",
            "python/carnot/reporting/human_authority_9097.py",
            "python/carnot/reporting/online_methods_9099.py",
        ]
    )
)
CAPSTONE = "results/experiment_9123_v777_capstone.json"
ARCHIVE = "openspec/change-proposals/research-roadmap-v777-preserved-20261116.md"
QUEUE = "openspec/change-proposals/research-roadmap-v777-preserved-20261116-queue.yaml"
PINS = {
    **old.PINS,
    DESIGN: "sha256:22c845b107d82b95169f902a400ff989cf47398eabda86fdad6b025264967cd7",
    ARCHIVE: "sha256:6e6635dadd64b3f59916ce35cd7a879323d480faf6e94ff1c6b76eeccea2c61b",
    QUEUE: "sha256:078ce21250bf76f43aec0f42b701c88a3371c45a4f2ad77d622df6edc2bfb7dc",
    CAPSTONE: "sha256:befa9ca0c3339aefc42c48919baced2f181f099a2ef61b799d6d029e4fb86580",
    "results/experiment_9110_v777_contract.json": "sha256:63d99c27406fe67e603977f89bae2f80f24d504a327672dfa9dd913864f084f6",
    "results/experiment_9111_v777_forward_evidence.json": "sha256:d62ba29c76582a20981d8643d9c61d00b984d1c7530476bfd8702df780087db7",
}
AUTHORITIES = [
    DESIGN,
    ACTIVE,
    LOG,
    "research-complete.yaml",
    ARCHIVE,
    QUEUE,
    "ops/exclusion_manifest.yaml",
]
MODEL_SPECS: list[Json] = []
reference, checksum = previous.reference, previous.checksum
_collect, _build, _replay, _main, _validate = (
    collection.collect,
    previous.build,
    previous.replay,
    previous.main,
    collection.validate,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries and inherited child heartbeats expose every long operation."""
    print(f"[exp9124] phase={phase} completed={completed} pending={pending}", flush=True)


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Name the actual operand producer so blocked inputs identify their owner."""
    row = previous.check(path, field, expected, observed, name)
    producer = re.match(r"experiment_(\d+)_", path.name)
    row["upstream"] = "exp" + producer[1] if producer else TASK
    return row


def authority(root: Path) -> Json:
    """Compare complete objects so a matching title cannot hide a changed prompt or gate."""
    tasks, rows, keys, checks = [], [], [], []
    mode = "unavailable"
    try:
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        mode = "activated" if active["milestone"] == MILESTONE else "staged"
        path = root / (ACTIVE if mode == "activated" else STAGED)
        queue = yaml.safe_load(path.read_bytes())
        Roadmap.model_validate(queue)
        tasks = queue["tasks"]
        table, machine = parse_design((root / DESIGN).read_text(), milestone=MILESTONE)
        checks = [
            check(
                root / DESIGN,
                "sha256",
                PINS[DESIGN],
                reference(root / DESIGN)["sha256"],
                "design_bytes",
            ),
            check(path, "milestone", MILESTONE, queue["milestone"], "lifecycle"),
            check(
                path, "task_count", [14] * 3, [len(tasks), len(table), len(machine)], "task_count"
            ),
        ]
        completed = {
            p[2].strip()
            for line in (root / LOG).read_text().splitlines()
            if len(p := line.split("|")) >= 4 and p[3].strip() in {"OK", "FLAGGED"}
        }
        archived = (root / "research-complete.yaml").read_text()
        titles = [t["title"][:50].strip() for t in tasks]
        earlier: dict[str, set[str]] = {}
        for i, task in enumerate(tasks):
            identity, key = task["id"], titles[i]
            predicates = dict(
                full_object=i < len(machine)
                and task in (machine[i], dict(machine[i], prompt=encode(machine[i]["prompt"]))),
                visible_table=i < len(table)
                and all(table[i][k] == task[k] for k in TABLE_FIELDS)
                and table[i]["order"] == i + 1,
                sequence=identity.startswith(f"exp{9124 + i}-"),
                gates=all(
                    set(g) == {"upstream", "artifact_field", "op", "value"}
                    and g["upstream"] in earlier
                    and g["artifact_field"] in earlier[g["upstream"]]
                    for g in task.get("gated_on", [])
                ),
                unique_key=titles.count(key) == 1,
                uncompleted_key=key not in completed,
                unused_id=re.search(rf"\bexp{9124 + i}[-_]", archived) is None
                and all(
                    identity == TASK and p.name == NAME + ".json"
                    for p in (root / "results").glob(f"experiment_{9124 + i}_*.json")
                ),
            )
            matched = all(predicates.values())
            rows.append(
                dict(
                    unit_id=identity,
                    original_qid=None,
                    sibling=None,
                    group_id=identity,
                    family=identity,
                    arm="full_task_contract",
                    condition="exact_authority",
                    seed=None,
                    metric="ordered_full_object_match",
                    value=int(matched),
                    absolute_metric=int(matched),
                    numerator=int(matched),
                    denominator=1,
                    status="completed" if matched else "failed",
                    exclusion_reason=None,
                    censor_reason=None,
                    task=task,
                    checks=predicates,
                    matched=matched,
                )
            )
            keys.append(
                dict(
                    unit_id=identity,
                    dispatch_key=key,
                    passed=predicates["unique_key"] and predicates["uncompleted_key"],
                )
            )
            earlier[identity] = set(
                re.findall(
                    r"\b[A-Za-z][A-Za-z0-9_]*\b",
                    task["prompt"].split("REQUIRED ARTIFACT FIELDS:", 1)[-1],
                )
            )
        checks.append(
            check(
                path,
                "all_complete_objects",
                True,
                len(rows) == 14 and all(r["matched"] for r in rows),
                "full_contract",
            )
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        yaml.YAMLError,
    ) as error:
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
    return dict(
        ready=bool(checks) and all(r["passed"] for r in checks),
        tasks=tasks,
        rows=rows,
        title_key_rows=keys,
        checks=checks,
        comparison_mode=mode,
    )


@contextmanager
def validation_binding() -> Iterator[None]:
    """Bind actual commands and coverage to this reader's exact owned statements."""
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


@contextmanager
def binding() -> Iterator[None]:
    """Reuse the checked lifecycle without editing the conductor or old producers."""
    with (
        validation_binding(),
        patch.multiple(
            previous,
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            AUTHORITIES=AUTHORITIES,
            PINS=PINS,
            CAPSTONE=CAPSTONE,
            validation_binding=validation_binding,
            scope_manifest=scope_manifest,
            input_pins=input_pins,
            reduce=reduce,
            build=build,
            replay=replay,
            progress=progress,
            collect=collect,
            validate=validate,
            terminal=terminal,
            authority=authority,
        ),
        patch.multiple(
            collection,
            NAME=NAME,
            OWNED=OWNED,
            TEST=TEST,
            AUTHORITIES=AUTHORITIES,
            binding=validation_binding,
            scope_manifest=scope_manifest,
            authority=authority,
            input_pins=input_pins,
            reduce=reduce,
            progress=progress,
        ),
    ):
        yield


def scope_manifest(private: Path) -> Json:
    """Freeze commands, code, tests and execution limits before reading any outcomes."""
    with validation_binding():
        manifest = validation.scope_manifest(private)
    manifest.update(
        activation_commands=[],
        global_suite_command=[str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        claim_scope="V778 administrative contract; no model or scientific measurement.",
    )
    return manifest


def input_pins(root: Path) -> dict[str, str]:
    """Walk only byte-authenticated retained work, never current producer replacements."""
    pins = dict(old.input_pins(root))
    pins.update(PINS)
    pending = [CAPSTONE]
    visited = set()
    while pending:
        name = pending.pop()
        path = root / name
        if name in visited or reference(path)["sha256"] != pins[name]:
            continue
        visited.add(name)
        value = json.loads(path.read_bytes())
        refs = (
            [value["work_reference"]]
            if name == CAPSTONE
            else [
                *(value["snapshots"].values() if isinstance(value.get("snapshots"), dict) else []),
                *value.get("operands", {}).values(),
            ]
        )
        for ref in refs:
            relative = str(Path(ref["path"]).relative_to(ROOT))
            if relative in pins and pins[relative] != ref["sha256"]:
                raise ValueError("historical_pin_conflict")
            pins[relative] = ref["sha256"]
            if Path(relative).name == "work.json":
                pending.append(relative)
    return pins


def reduce(work: Json) -> Json:
    """Preserve original source roles and failed science before any new study runs."""
    root, raw = Path(work["retained"]), Path(work["evidence_raw"])
    capstone = json.loads((root / CAPSTONE).read_bytes())
    source = json.loads((root / "results/experiment_9110_v777_contract.json").read_bytes())
    gates = [
        check(root / name, field, expected, value.get(field, "missing_field"), "upstream_gate")
        for name, value, fields in [
            (
                "results/experiment_9110_v777_contract.json",
                source,
                {
                    "contract_ready_score": 1,
                    "source_ready_score": 1,
                    "online_methods_ready_score": 1,
                    "required_checks_passed": True,
                },
            ),
            (CAPSTONE, capstone, {"reconciliation_ready_score": 1, "required_checks_passed": True}),
        ]
        for field, expected in fields.items()
    ]
    work["preconditions_checked"]["checks"].extend(
        r for r in gates if r not in work["preconditions_checked"]["checks"]
    )
    if not all(r["passed"] for r in gates):
        return {}
    with patch.multiple(old, DESIGN=ARCHIVE, authority=authority, check=check):
        reduction = old.reduce(work)
    if not reduction:
        return {}
    section = (
        (root / DESIGN)
        .read_text()
        .split("## Frozen scientific protocols\n", 1)[1]
        .split("## Execution and acceptance discipline", 1)[0]
    )
    protocol = dict(
        name="separate_v778_logit_temperature_study",
        protocol=section,
        measured=False,
        inherited_embedding_bounds_unchanged=True,
        inherited_graph_bounds_unchanged=True,
        design_reference=reference(root / DESIGN),
    )
    target = raw / "frozen_protocols/v778_logit_temperature.json"
    if target.exists() and json.loads(target.read_bytes()) != protocol:
        raise ValueError("immutable_protocol_drift")
    atomic_json(target, protocol)
    saved = old.historical(root)
    methods = saved["values"]["exp9099-online-methods"]["value"]
    online = json.loads(Path(reduction["online_protocol_reference"]["path"]).read_bytes())
    roster = dict(
        ordering=online["protocol"]["ordering"],
        donors=online["protocol"]["donors"],
        roles=online["protocol"]["roles"],
        original_reference=source["online_protocol_reference"],
    )
    roster.update(
        {
            key: methods[key]
            for key in [
                "initial_group_roster",
                "stream_roster",
                "guard_roster",
                "retention_roster",
                "sibling_roster",
                "donor_map",
            ]
        }
    )
    roster_path = raw / "frozen_protocols/online_roster.json"
    if roster_path.exists() and json.loads(roster_path.read_bytes()) != roster:
        raise ValueError("immutable_roster_drift")
    atomic_json(roster_path, roster)
    manifest = json.loads(Path(reduction["source_manifest_reference"]["path"]).read_bytes())
    master = manifest["release_files"]["data/human_judge_release_master.csv"]
    original = saved["operands"][master["path"]]
    path = root / Path(original["path"]).relative_to(ROOT)
    view_rows = [
        dict(
            unit_id=row["qid"] + "__" + row["model"],
            qid=row["qid"],
            sibling=row["model"],
            question=row["question"],
            answer_text=row["answer_text"],
        )
        for row in csv.DictReader(io.StringIO(path.read_text()))
    ]
    view = dict(
        rows=view_rows,
        source_reference=reference(path),
        token_view=dict(
            prefix="Question:\n{question}\nAnswer:\n",
            bos=False,
            chat_template=False,
            question_cap=256,
            answer_cap=768,
            tokenizer="future embedded tokenizer; no tokenizer loaded by this contract",
        ),
    )
    view_path = raw / "frozen_protocols/inference_view.json"
    if view_path.exists() and json.loads(view_path.read_bytes()) != view:
        raise ValueError("immutable_inference_view_drift")
    atomic_json(view_path, view)
    outcomes = capstone["task_outcome_rows"]
    outcomes[-1] = dict(
        outcomes[-1], primary_status="present", reference=reference(root / CAPSTONE)
    )
    reduction.update(
        previous_outcome_rows=outcomes,
        historical_model_provenance=capstone["historical_model_provenance"],
        online_roster_reference=reference(roster_path),
        new_study_protocol_reference=reference(target),
        inference_view_reference=reference(view_path),
        inherited_receipt_failure=json.loads(
            (root / "results/experiment_9111_v777_forward_evidence.json").read_bytes()
        ),
        hardware_limits=capstone["hardware_limits"],
        closed_scope_rows=capstone["closed_scope_rows"],
    )
    reduction["protocol_references"].append(reference(target))
    reduction["archive_lag"].update(
        planning_archive_end="2026.11.776",
        completed_scheduling_frontier="2026.11.777",
        archive_lagged=reduction["archive_lag"]["observed_archive_end"] != "2026.11.777",
    )
    return reduction


def date_check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """Only the current date changes; historical byte and scientific gates stay fixed."""
    return check(path, field, "20261116" if name == "date" else expected, observed, name)


def collect(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Authenticate and freeze every input before the historical reducer can consume it."""
    with binding():
        with patch.object(collection, "check", date_check):
            work = dict(_collect(root, raw, private, run_date))
    return work


def validate(work: Json) -> Json:
    """Keep normal exits, direct consumers and real zero-exclusion statement coverage."""
    with binding():
        return dict(_validate(work))


def build(work: Json) -> Json:
    """Custody and mechanics readiness depend on owned checks, never measured benefit."""
    with binding():
        value = _build(work)
    reduction = work["reduction"]
    value.update(
        experiment_id=9124,
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict=value["honest_verdict"].replace("v776_contract", "v778_contract"),
        requested_inference_substrate_class="no_model_load",
        generated_tokens=0,
        random_seed=7789124,
        graph_study_status="deferred_unchanged_seven_and_ten_bounds",
        source_ready_score=int(
            bool(value["required_checks_passed"] and reduction.get("source_closure_ready"))
        ),
        online_methods_ready_score=int(
            bool(value["required_checks_passed"] and reduction.get("online_closure_ready"))
        ),
    )
    for field in [
        "source_manifest_reference",
        "evaluator_label_reference",
        "shard_a_roster_reference",
        "shard_b_roster_reference",
        "static_protocol_reference",
        "online_protocol_reference",
        "energy_protocol_reference",
        "online_roster_reference",
        "new_study_protocol_reference",
        "inference_view_reference",
        "inherited_receipt_failure",
        "hardware_limits",
    ]:
        value[field] = reduction.get(field, {})
    value["field_principles"].update(
        {
            "requested_inference_substrate_class": "The requested no-model operation remains distinct from cited historical inference.",
            "generated_tokens": "Zero current generated tokens cannot be filled by archived model output.",
            "source_ready_score": "Only complete original source custody and current owned checks authorize reuse of exposed labels.",
            "online_methods_ready_score": "Original learner protocol, rosters and qualified recovery must remain authenticated independently of benefit.",
            "source_manifest_reference": "Original source closure preserves human annotation authority rather than recreating labels.",
            "evaluator_label_reference": "Evaluator-only labels cannot enter the model's text or feature views.",
            "shard_a_roster_reference": "The first lexical qid half retains every original sibling and role.",
            "shard_b_roster_reference": "The second disjoint qid half cannot disappear after shard A completes.",
            "static_protocol_reference": "Six fixed arms and TRAIN-only transforms prevent outcome-driven method selection.",
            "online_protocol_reference": "Delayed feedback, original rosters and six bounds prevent future-label leakage.",
            "energy_protocol_reference": "A separate premeasurement logit protocol prevents changing inherited scientific thresholds.",
            "online_roster_reference": "Original initialization, streams and donors cannot change after outcomes.",
            "new_study_protocol_reference": "Seven-feature inputs and the sixth temperature arm form a separate study, without changing inherited bounds.",
            "inference_view_reference": "Only question and answer text reach future tokenization; evaluator labels and source roles stay separate.",
            "inherited_receipt_failure": "The original failed receipts and uncovered statements stay disqualified even when this contract qualifies.",
            "hardware_limits": "Authenticated historical hardware scope grants no new operation, speed or energy claim.",
        }
    )
    for i, row in enumerate(value["rows"]):
        row.setdefault("sibling", None)
        if not reduction:
            row.update(unit_id=f"exp{9124 + i}", group_id=f"exp{9124 + i}", family=f"exp{9124 + i}")
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild aggregates from pinned rows so a newly hashed invention still fails."""
    with binding():
        return _replay(path)


def terminal(candidate: Path, raw: Path, manifest: Json) -> Json:
    """Unchanged adversarial and strict-row checks plus foreign replay guard publication."""
    with validation_binding():
        return validation.terminal(candidate, raw, manifest)


def main(argv: list[str] | None = None) -> int:
    """Publish checked bytes once; private replays cannot rewrite a live primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261116")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = replay(args.cold_replay)
        print(json.dumps(result), flush=True)
        return int(not result["passed"])
    if args.date != "20261116":
        parser.error("this contract requires --date 20261116")
    output = args.output.absolute()
    if args.from_work and output.is_relative_to(ROOT):
        parser.error("private work publication requires output outside the checkout")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    if args.from_work:
        work = json.loads(args.from_work.read_bytes())
        work["raw"] = str(raw)
    else:
        private = Path(tempfile.mkdtemp(prefix="carnot9124-", dir="/var/tmp"))
        work = collect(args.root, raw, private, args.date)
        if (time.monotonic_ns() - work["start_ns"]) / 1e9 >= 3300:
            raise TimeoutError("measurement_deadline")
        progress("validation_before")
        work["validation"] = validate(work)
        progress("validation_after")
        if not build(work)["required_checks_passed"]:
            atomic_json(raw / "work.json", work)
            progress("publication_disqualified", 0, 1)
            return 1
        receipt = validation.run_check(
            CommandSpec(
                "global_python_suite",
                tuple(work["validation_scope_manifest"]["global_suite_command"]),
                "broader_history",
                900,
            ),
            raw / "global_suite",
            work["validation_scope_manifest"],
        )
        work["repository_health"] = dict(
            passed=receipt["passed"],
            receipt=receipt,
            affects_owned_readiness=False,
            historical_failures_preserved=True,
        )
        work["end_ns"] = time.monotonic_ns()
        if (work["end_ns"] - work["start_ns"]) / 1e9 >= 4800:
            raise TimeoutError("hard_deadline")
    atomic_json(raw / "work.json", work)
    value = build(work)
    progress("publication_before")
    receipt = publish_primary(
        output,
        value,
        lambda path: terminal(path, raw / "terminal", work["validation_scope_manifest"]),
    )
    atomic_json(raw / "publication.json", receipt)
    progress("publication_after", 1, 0)
    print(json.dumps(dict(receipt, honest_verdict=value["honest_verdict"])), flush=True)
    return 0
