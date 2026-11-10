"""REQ-REPORT-8971: retain original outcomes without promoting scheduler completion to science."""

import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import v766_branch_evidence as arc
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v755_branch_contract import reference

Json = dict[str, Any]
ARCHIVE = "openspec/change-proposals/research-roadmap-v766-preserved-20261110.md"
QUEUE = "openspec/change-proposals/research-roadmap-v766-queue-preserved-20261110.yaml"
RECORDS = ["research-complete.yaml", "ops/conductor-log.md"]
PINS = {
    "openspec/change-proposals/research-roadmap-v766-preserved-20261110.md": "sha256:bd25e5f98dd161d3cdb34e70b4715d56f085a35b205afe36ceb3485601594a16",
    "openspec/change-proposals/research-roadmap-v766-queue-preserved-20261110.yaml": "sha256:5d35d511200eeeb684fd33d5f8ef8a9578b5c051cabbf8ab64929d2e8500bb6e",
    "results/experiment_8957_v766_branch_contract.json": "sha256:ce8805f086e616552e2168ea0f1495d02c177afc58e0728aae9ede733f957077",
    "results/experiment_8958_v766_confidence_diagnosis.json": "sha256:ff279de4b7b8f2a8eae95d978288da6f588bf6c0d5635e83bf062a72b605f6b0",
    "results/experiment_8959_v766_relational_methods.json": "sha256:eaa01cdf08424373c0ea12b703a09099bf489cd015593111a5142c1c2f25c966",
    "results/experiment_8960_v766_online_calibration_methods.json": "sha256:1af36d37457cd5b7cda9be1b6cc80bc89c96f29dcf6427c4fbb2946728b3b09b",
    "results/experiment_8961_v766_graph_corpus_a.json": "sha256:0ba3322b01052ad632c494186451b5ac29d375547147d55c0470a962bcb1f616",
    "results/experiment_8962_v766_graph_corpus_b.json": "sha256:e941ec67fc5dbcb3949efd64f704a8a23c6feeb9140aa797b8d774905941960c",
    "results/experiment_8963_v766_corpus_admission.json": "sha256:586a8f5bff118cd9416f0f03cf9606bc35db0eb4068d946bbf17105329547d24",
    "results/experiment_8964_v766_relational_decisions.json": "sha256:f70d9becc3486f2dec4f8c0c9400cb6ebb49c1bd1fcd0ae67e36b56f74872408",
    "results/experiment_8965_v766_relational_controls.json": "sha256:188c29a7ee88ae6ca363a7e9137d4fe11c2308c1b3b9f037ce21d51259d1ad7d",
    "results/experiment_8966_v766_online_response_streams.json": "sha256:906a2143c36d902d19615ddaef00f5fc4b3b1c0c883e77f54e91c0821ac92cbd",
    "results/experiment_8967_v766_online_benefit_audit.json": "sha256:dbb48d10ce064fb08fe6d0efcd6671111863641c7591dcfc4bf6df68c5dc3f21",
    "results/experiment_8968_v766_complete_service_cost.json": "sha256:7fc603abadcfe0e42c9a4dbf9ddf3d5f9d34088bcd0ec06d19e0eb2905d39df0",
    "results/experiment_8969_arc_transfer_evidence.json": "sha256:633f2a4a9bfc7d81781ef14fa2d918efe1372918c034d0417b3ba6c72830aeb2",
    "research-complete.yaml": "sha256:74afb5583176f1dc9261fbbc54e0857590c175db01fa096ee42abed8f8ee861e",
    "ops/conductor-log.md": "sha256:eca7d4890fde3815a55d81b23ff5acb58ab0ec62e7f25abd36492251c54b7f69",
}


def diagnostic(path: Path, check: str, field: str, expected: Any, observed: Any) -> Json:
    """Name the failed operand so absence cannot silently become an observed zero."""
    return dict(
        **reference(path),
        upstream="v766_original_evidence",
        check=check,
        artifact_field=field,
        operator="==",
        expected=expected,
        observed=observed,
    )


def _history(root: Path) -> Json:
    """Read each original producer and scheduler outcome; a missing capstone earns no result."""
    refs, rows, failures = [], [], []
    values: Json = {}
    for name, pin in PINS.items():
        path = root / name
        actual = sha256_file(path) if path.is_file() else "missing_file"
        if actual != pin:
            failures.append(diagnostic(path, "original_bytes", "sha256", pin, actual))
            continue
        value = json.loads(path.read_bytes()) if path.suffix == ".json" else {}
        identity = value.get("experiment_id", value.get("experiment"))
        refs.append(
            dict(
                reference(path),
                experiment_id=identity,
                fields_imported=list(value) if value else ["original_authority_bytes"],
            )
        )
        values[str(identity)] = value
    if failures:
        return dict(
            authenticated=False,
            rows=rows,
            references=refs,
            failures=failures,
            quarantine={},
            original_child_qualified=False,
        )
    tasks = parse_design((root / ARCHIVE).read_text(), milestone="2026.11.766")[1]
    queue = yaml.load((root / QUEUE).read_bytes(), Loader=yaml.CSafeLoader)
    archive = yaml.load((root / RECORDS[0]).read_bytes(), Loader=yaml.CSafeLoader)
    completed = next(m for m in archive["milestones"] if m["id"] == "2026.11.766")
    log = (root / RECORDS[1]).read_text()
    refs += [
        dict(
            reference(root / n),
            experiment_id=None,
            fields_imported=["V766_terminal_scheduler_outcomes"],
        )
        for n in RECORDS
    ]
    for task, scheduled in zip(tasks, completed["tasks"], strict=True):
        identity = int(task["id"].split("-")[0][3:])
        path = root / task["deliverable"]
        value = values.get(str(identity), {})
        kind = "primary" if path.is_file() else "structured_skip" if value else "missing_primary"
        if kind == "primary":
            validate_primary(value, path)
        receipt = next(
            (root / n for n in PINS if n.startswith(f"results/experiment_{identity}_")), None
        )
        log_rows = [line for line in log.splitlines() if scheduled["title"][:45] in line]
        if scheduled["id"] != task["id"] or not log_rows:
            failures.append(
                diagnostic(
                    root / RECORDS[1],
                    "scheduler_identity",
                    task["id"],
                    "matching task and conductor lines",
                    log_rows,
                )
            )
        rows.append(
            dict(
                task_id=task["id"],
                experiment_id=identity,
                kind=kind,
                primary_reference=reference(path),
                conductor_reference=reference(
                    receipt if kind == "structured_skip" and receipt else root / RECORDS[1]
                ),
                conductor_rows=log_rows,
                scheduler_outcome=scheduled["result"],
                verdict=value.get("verdict_class", value.get("honest_verdict")),
                scientific_results_admitted=False,
            )
        )
    return dict(
        authenticated=not failures,
        rows=rows,
        references=refs,
        failures=failures,
        quarantine=values["8957"]["history"]["quarantine"],
        original_child_qualified=True,
        preserved_queue_milestone=queue["milestone"],
    )


def history(root: Path) -> Json:
    """Malformed required history remains an explicit terminal diagnostic, never invented science."""
    try:
        return _history(root)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        path = (
            Path(error.filename)
            if isinstance(error, OSError) and error.filename
            else root / RECORDS[0]
        )
        return dict(
            authenticated=False,
            rows=[],
            references=[],
            quarantine={},
            original_child_qualified=False,
            failures=[
                diagnostic(
                    path,
                    "history_schema",
                    str(error.args[0]) if isinstance(error, KeyError) else "terminal_history",
                    "authenticated original outcomes",
                    "missing_field" if isinstance(error, KeyError) else str(error),
                )
            ],
        )
