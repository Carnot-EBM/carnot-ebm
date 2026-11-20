"""REQ-REPORT-9110-V777: retain authentic methods without borrowing scientific credit.

Only immutable historical snapshots feed the reduction. The established contract
lifecycle supplies command supervision, measured coverage and atomic publication.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import re
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import scoped_provenance_9014 as validation
from carnot.reporting import v776_contract as previous
from carnot.reporting import v776_capstone_evidence as history
from carnot.reporting import v775_contract as collection
from carnot.reporting import v775_capstone_evidence as inherited_history
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import parse_design, TABLE_FIELDS
from carnot.reporting.v749_contract_methods import encode
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED, LOG = (
    previous.ROOT,
    previous.DESIGN,
    previous.ACTIVE,
    previous.STAGED,
    previous.LOG,
)
NAME, TASK, MILESTONE = "experiment_9110_v777_contract", "exp9110-contract", "2026.11.777"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v777_contract_9110.py"
OWNED = ["python/carnot/reporting/v777_contract.py", CLI]
DIRECT = previous.DIRECT
REUSED = list(
    dict.fromkeys(
        [
            *previous.REUSED,
            *previous.OWNED,
            "python/carnot/reporting/v776_capstone_evidence.py",
            "python/carnot/reporting/human_authority_9097.py",
            "python/carnot/reporting/online_methods_9099.py",
        ]
    )
)
CAPSTONE = "results/experiment_9109_v776_capstone.json"
ARCHIVE = "openspec/change-proposals/research-roadmap-v776-preserved-20261115.md"
QUEUE = "openspec/change-proposals/research-roadmap-v776-queue-preserved-20261115.yaml"
PINS = {
    DESIGN: "sha256:6e6635dadd64b3f59916ce35cd7a879323d480faf6e94ff1c6b76eeccea2c61b",
    ARCHIVE: "sha256:9636b61f464b350f38e2317f2e197ddc79f22b4ed4c0f34d13099c36c5e26700",
    QUEUE: "sha256:05c2129cbd2cba5a828860c221e1e459d337579167d97241d9626cb5c3fa0517",
    CAPSTONE: "sha256:095eaccad916fbc7d12b7a84009951a423c0c411c8f35b82a7993f3f33e9b6ed",
    **{p: pin for p, pin in history.PINS.items() if p.startswith("results/")},
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
reference, checksum, check = previous.reference, previous.checksum, previous.check
_collect, _build, _replay, _main, _validate = (
    collection.collect,
    previous.build,
    previous.replay,
    previous.main,
    collection.validate,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries and inherited child heartbeats expose every long operation."""
    print(f"[exp9110] phase={phase} completed={completed} pending={pending}", flush=True)


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
                sequence=identity.startswith(f"exp{9110 + i}-"),
                gates=all(
                    set(g) == {"upstream", "artifact_field", "op", "value"}
                    and g["upstream"] in earlier
                    and g["artifact_field"] in earlier[g["upstream"]]
                    for g in task.get("gated_on", [])
                ),
                unique_key=titles.count(key) == 1,
                uncompleted_key=key not in completed,
                unused_id=re.search(rf"\bexp{9110 + i}[-_]", archived) is None
                and all(
                    identity == TASK and p.name == NAME + ".json"
                    for p in (root / "results").glob(f"experiment_{9110 + i}_*.json")
                ),
            )
            matched = all(predicates.values())
            rows.append(
                dict(
                    unit_id=identity,
                    original_qid=None,
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
        claim_scope="V777 administrative contract; no model or scientific measurement.",
    )
    return manifest


def input_pins(root: Path) -> dict[str, str]:
    """Walk only byte-authenticated retained work, never current producer replacements."""
    pins = dict(PINS)
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


def historical(root: Path) -> Json:
    """Resolve nested primitive readers through retained copies of original snapshots."""
    capstone = json.loads((root / CAPSTONE).read_bytes())
    path = root / Path(capstone["work_reference"]["path"]).relative_to(ROOT)
    return json.loads(path.read_bytes())  # type: ignore[no-any-return]


def reduce(work: Json) -> Json:
    """Recompute retained units and seal unchanged methods before any future measurement."""
    root, raw = Path(work["retained"]), Path(work["evidence_raw"])
    saved = historical(root)

    def primitive(envelope: Json, ref: Json) -> Json:
        """Nested historical work must use authenticated snapshots, including old code."""
        bound = envelope["operands"][ref["path"]]
        if bound["sha256"] != ref["sha256"]:
            raise ValueError("primitive_identity")
        return json.loads((root / Path(bound["path"]).relative_to(ROOT)).read_bytes())  # type: ignore[no-any-return]

    source = saved["values"]["exp9097-human-authority"]["value"]
    methods = saved["values"]["exp9099-online-methods"]["value"]
    gates = [
        check(root / name, field, expected, value.get(field, "missing_field"), "upstream_gate")
        for name, value, field, expected in [
            ("results/experiment_9097_v776_human_authority.json", source, "source_ready_score", 1),
            (
                "results/experiment_9097_v776_human_authority.json",
                source,
                "required_checks_passed",
                True,
            ),
            (
                "results/experiment_9099_v776_online_methods.json",
                methods,
                "online_methods_ready_score",
                1,
            ),
            (
                "results/experiment_9099_v776_online_methods.json",
                methods,
                "required_checks_passed",
                True,
            ),
            ("results/experiment_9099_v776_online_methods.json", methods, "exact_resume", True),
        ]
    ]
    for producer in [source, methods]:
        refs_to_check = [
            *producer["source_references"],
            *producer["code_config_hashes"],
            *producer["raw_shard_hashes"],
            *[
                ref
                for key, ref in producer.items()
                if key.endswith("_reference") and isinstance(ref, dict)
            ],
        ]
        for ref in refs_to_check:
            if not ref.get("path") or not ref.get("sha256"):
                continue
            bound = saved["operands"].get(ref["path"], {})
            row = check(
                root,
                "snapshot_sha256",
                ref["sha256"],
                bound.get("sha256", "missing_field"),
                "original_source_closure",
            )
            row.update(upstream=producer["task_id"], original_path=ref["path"])
            gates.append(row)
    gates.append(
        check(
            root,
            "qualified_recovery",
            True,
            len(methods["recovery_receipts"]) == len(methods["hard_exit_receipts"]) == 3
            and all(r["passed"] and r["exit_code"] == 0 for r in methods["recovery_receipts"])
            and all(r["passed"] and r["exit_code"] == 73 for r in methods["hard_exit_receipts"]),
            "original_recovery",
        )
    )
    work["preconditions_checked"]["checks"].extend(
        r for r in gates if r not in work["preconditions_checked"]["checks"]
    )
    if not all(r["passed"] for r in gates):
        return {}
    text = (root / DESIGN).read_text()
    section = text.split("## Frozen inherited scientific protocols\n", 1)[1].split(
        "## Execution discipline", 1
    )[0]
    protocols = {
        "static": primitive(saved, source["static_protocol_reference"]),
        "online": primitive(saved, methods["online_protocol_reference"]),
        "signal": section.split("### Static signal controls\n", 1)[1].split(
            "### New raw-logit protocol", 1
        )[0],
        "energy": section.split("### New raw-logit protocol\n", 1)[1].split(
            "### Complete costs", 1
        )[0],
        "cost": section.split("### Complete costs and hardware\n", 1)[1],
    }
    refs = {}
    for name, protocol in protocols.items():
        target = raw / "frozen_protocols" / (name + ".json")
        value = dict(
            name=name, protocol=protocol, design_reference=reference(root / DESIGN), measured=False
        )
        if target.exists() and json.loads(target.read_bytes()) != value:
            raise ValueError("immutable_protocol_drift")
        atomic_json(target, value)
        refs[name] = reference(target)
    with (
        patch.object(history, "primitive", primitive),
        patch.object(inherited_history, "primitive", primitive),
    ):
        science = history.reduce(saved)
    outcomes = history.outcomes(saved)
    capstone = json.loads((root / CAPSTONE).read_bytes())
    outcomes.append(
        dict(
            capstone["task_outcome_rows"][-1],
            primary_status="present",
            reference=reference(root / CAPSTONE),
            selected_path=CAPSTONE,
        )
    )
    references = {
        key: saved["operands"][source[key]["path"]]
        for key in [
            "source_manifest_reference",
            "evaluator_label_reference",
            "shard_a_roster_reference",
            "shard_b_roster_reference",
        ]
    }
    archive = yaml.load((root / "research-complete.yaml").read_bytes(), Loader=yaml.CSafeLoader)[
        "milestones"
    ][-1]["id"]
    return dict(
        contract=authority(root),
        previous_outcome_rows=outcomes,
        historical_model_provenance=capstone["historical_model_provenance"],
        protocol_references=list(refs.values()),
        static_protocol_reference=refs["static"],
        online_protocol_reference=refs["online"],
        energy_protocol_reference=refs["energy"],
        **references,
        preserved_results=science,
        deferred_graph_bounds={k: science[k] for k in ["seven_bound_rows", "ten_bound_rows"]},
        closed_scope_rows=science["continuation_decisions"],
        archive_lag=dict(
            planning_archive_end="2026.11.775",
            observed_archive_end=archive,
            completed_scheduling_frontier="2026.11.776",
            archive_lagged=archive != "2026.11.776",
            archive_modified=False,
            scientific_result_from_scheduler=False,
        ),
        source_closure_ready=True,
        online_closure_ready=bool(methods["online_methods_ready_score"]),
    )


def collect(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Authenticate and freeze every input before the historical reducer can consume it."""
    with binding():
        return dict(_collect(root, raw, private, run_date))


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
        experiment_id=9110,
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict=value["honest_verdict"].replace("v776_contract", "v777_contract"),
        requested_inference_substrate_class="no_model_load",
        generated_tokens=0,
        random_seed=7779110,
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
        }
    )
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
    """Publish only checked bytes; inherited limits reserve time for actual validation."""
    with binding():
        return _main(argv)
