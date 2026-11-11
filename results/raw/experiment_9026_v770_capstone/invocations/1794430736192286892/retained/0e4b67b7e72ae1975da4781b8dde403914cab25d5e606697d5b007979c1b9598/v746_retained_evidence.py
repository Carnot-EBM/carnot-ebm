"""REQ-VERIFY-8695: keep completed child evidence before scratch can change.

A child receipt owns copies of its SQLite, tests, sources and exit. Combining
copies never changes that receipt, so later consumers cannot erase execution.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET

from coverage import Coverage, CoverageData

from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
HISTORY = ROOT / "results/experiment_8681_v745_contract_methods.json"
HISTORY_PIN = "sha256:fdaa9313e6466b2bc0c3e9e72516d4d17286de1a5148a090e41cd6bfe5b3e284"
REJECTION_SOURCE = ROOT / (
    "results/raw/experiment_8667_v744_contract_methods/invocations/1793500792730659524/"
    "source_before_report_fix/python/carnot/reporting/v744_contract_report.py"
)
REJECTION_PIN = "sha256:c3a4b3756f768c9a402f61ae1c5af915c68c92e09faf2bd8f88626a35b3deeac"


def reference(path: Path) -> Json:
    """Bind exact bytes rather than a filename that could later hold different evidence."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def load(ref: Json) -> Json:
    """Refuse replacement bytes before a reader interprets their reported values."""
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("changed_bytes:" + ref["path"])
    return dict(json.loads(Path(ref["path"]).read_bytes()))


def retain(
    command: Json, nodes: list[str], scratch: Path, raw: Path, sources: list[Json], role: str
) -> Json:
    """Copy a finished child's evidence immediately, including unsuccessful exits."""
    raw.mkdir(parents=True, exist_ok=True)
    refs, source_refs, errors = [], [], []
    for i, source in enumerate(sorted(scratch.glob(".coverage*"))):
        target = raw / f"child-{i}.sqlite"
        shutil.copyfile(source, target)
        refs.append(reference(target))
        if target.read_bytes()[:16] != b"SQLite format 3\x00":
            errors.append("not_sqlite")
    for i, ref in enumerate(sources):
        source = Path(ref["path"])
        if not source.is_file():
            errors.append("missing_source")
            continue
        if sha256_file(source) != ref["sha256"]:
            errors.append("source_drift")
        target = raw / "sources" / str(i)
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        source_refs.append(
            dict(reference(target), original_path=str(source), original_sha256=ref["sha256"])
        )
    junit = scratch / "junit.xml"
    junit_ref = None
    if junit.is_file():
        shutil.copyfile(junit, raw / "junit.xml")
        junit_ref = reference(raw / "junit.xml")
    receipt = deepcopy(command)
    for stream in ["stdout", "stderr"]:
        source = Path(receipt[stream + "_path"])
        target = raw / stream
        shutil.copyfile(source, target)
        receipt[stream + "_path"] = str(target)
    value = dict(
        command_receipt=receipt,
        node_ids=nodes,
        role=role,
        sqlite_references=refs,
        junit_reference=junit_ref,
        source_hashes=source_refs,
        errors=errors,
    )
    atomic_json(raw / "receipt.json", value)
    ref = reference(raw / "receipt.json")
    atomic_json(raw / "receipt.seal.json", ref)
    return ref


def aggregate(rows: list[Json], expected: list[str]) -> Json:
    """Join immutable completed children; neither scratch contents nor consumers grant credit."""
    nodes: list[str] = []
    errors: list[str] = []
    for ref in rows:
        try:
            value = load(ref)
            if json.loads(Path(ref["path"]).with_suffix(".seal.json").read_bytes()) != ref:
                raise ValueError("independent_child_seal")
            command = value["command_receipt"]
            if not (
                command["passed"]
                and command["normal_exit"]
                and not command["timed_out"]
                and command["actual_exit"] == command["expected_exit"] == 0
            ):
                errors.append("failed_child")
            if value["errors"] or not value["sqlite_references"] or not value["source_hashes"]:
                errors.append("incomplete_child")
            references = value["sqlite_references"] + value["source_hashes"]
            references += [value["junit_reference"]]
            references += [
                dict(path=command[s + "_path"], sha256=command[s + "_sha256"])
                for s in ["stdout", "stderr"]
            ]
            if any(sha256_file(Path(r["path"])) != r["sha256"] for r in references):
                errors.append("changed_child_bytes")
            junit = ET.parse(value["junit_reference"]["path"])
            cases = junit.findall(".//testcase")
            names = [n.rsplit("::", 1)[-1] for n in value["node_ids"]]
            if sorted(c.get("name") for c in cases) != sorted(names) or any(len(c) for c in cases):
                errors.append("junit_node_or_outcome_mismatch")
            nodes += value["node_ids"]
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
            errors.append(str(error))
    if not rows or len(nodes) != len(set(nodes)) or sorted(nodes) != sorted(expected):
        errors.append("exact_disjoint_node_union")
    return dict(passed=not errors, errors=errors, node_ids=nodes, retained_child_rows=rows)


def combine(rows: list[Json], root: Path) -> Json:
    """Combine only fresh copies in a role-specific root; preserve original SQLite bytes."""
    root.mkdir(parents=True, exist_ok=True)
    references = [r for row in rows for r in load(row)["sqlite_references"]]
    if not references:
        return dict(references=[], combined_reference=None)
    for i, ref in enumerate(references):
        source = Path(ref["path"])
        if sha256_file(source) != ref["sha256"]:
            raise ValueError("changed_sqlite")
        shutil.copyfile(source, root / f".coverage.{i}")
    coverage = Coverage(data_file=str(root / ".coverage"), config_file=False)
    coverage.combine(keep=True)
    coverage.save()
    return dict(references=references, combined_reference=reference(root / ".coverage"))


def historical_failure() -> Json:
    """Reproduce Exp8681's recorded failure without granting its flagged producer readiness."""
    artifact = load(dict(path=str(HISTORY), sha256=HISTORY_PIN))
    validation = load(artifact["validation_reference"])
    rows = validation["receipts"]
    authority = next(r for r in rows if r["name"] == "original_contract_node_authority")
    final = next(r for r in rows if r["name"] == "original_coverage_retention")
    raw = Path(artifact["validation_reference"]["path"]).parent
    custody = [reference(p) for p in sorted((raw / "checks").glob("*.custody.json"))]
    sqlite = [ref for r in custody for ref in load(r)["original_references"]]
    if not all(sha256_file(Path(r["path"])) == r["sha256"] for r in sqlite):
        raise ValueError("changed_historical_child_sqlite")
    terminal_refs = [
        dict(path=str(ROOT / "results" / name), sha256=pin)
        for name, pin in [
            (
                "experiment_8691_v745_arc_censoring_decision.json",
                "sha256:89a9805fe678365ca347958c497f90229a6daa6bd051294ae294998f2d8d9366",
            ),
            (
                "experiment_8692_v745_spline_precision_feasibility.json",
                "sha256:bb960180068d3f6960df74f7f0dfeb440ed64ac4951071674e820a0d912eaa33",
            ),
            (
                "experiment_8693_v745_hardware_operation_boundary.json",
                "sha256:1edc62c1f83d68185ba712fa181275ab711ba9b7e51a1ac684e8e6d4b1004006",
            ),
            (
                "experiment_8694_v745_capstone.json",
                "sha256:f9551a44d8d29f01cbb344d96647808253fd524262e582f7a4782e8e97e30947",
            ),
        ]
    ]
    terminal_rows = [
        {
            k: load(ref)[k]
            for k in [
                "experiment_id",
                "honest_verdict",
                "required_checks_passed",
                "flagged_adversarial",
            ]
        }
        for ref in terminal_refs
    ]
    return dict(
        original_node_count=len(authority["original_node_ids"]),
        original_node_ids=authority["original_node_ids"],
        original_test_source_hashes=authority.get("original_test_source_hashes", []),
        retained_child_sqlite_count=len(sqlite),
        final_retained_reference_count=len(final["original_references"]),
        original_rejection_line_executed=authority["rejection_line_executed"],
        flagged_adversarial=artifact["flagged_adversarial"],
        required_checks_passed=artifact["required_checks_passed"],
        preserved_v745_terminal_rows=terminal_rows,
        references=[
            reference(HISTORY),
            artifact["validation_reference"],
            *custody,
            *sqlite,
            *terminal_refs,
        ],
    )


def reject_original(private: Path) -> None:
    """Execute the frozen original rejection statement, rather than today's line number."""
    if sha256_file(REJECTION_SOURCE) != REJECTION_PIN:
        raise ValueError("original_rejection_source_drift")
    if REJECTION_SOURCE.read_text().splitlines()[112].strip() != "return False":
        raise ValueError("original_rejection_statement_drift")
    spec = importlib.util.spec_from_file_location("v746_original_rejection", REJECTION_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = private / "original-rejection.json"
    atomic_json(
        candidate, dict(raw_shard_hashes=[dict(path=str(private / "absent"), sha256="missing")])
    )
    assert module.replay(candidate) is False


def rejection_executed(rows: list[Json]) -> bool:
    """Only producer SQLite naming the exact frozen original source can prove line113 ran."""
    lines: set[int] = set()
    for row in rows:
        value = load(row)
        if value["role"] not in {"producer", "producer_control"}:
            continue
        for ref in value["sqlite_references"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("changed_producer_sqlite")
            data = CoverageData(basename=ref["path"])
            data.read()
            lines.update(data.lines(str(REJECTION_SOURCE)) or [])
    return sha256_file(REJECTION_SOURCE) == REJECTION_PIN and 113 in lines
