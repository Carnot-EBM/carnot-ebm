"""REQ-VERIFY-8716: reuse installed batching and retain bounded process evidence."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any
from unittest.mock import patch

from coverage import Coverage
import yaml

from carnot.reporting import installed_batch_measure_8702 as measurement
from carnot.reporting import installed_batch_runner_8702 as previous
from carnot.reporting import installed_import_isolation_8716 as e
from carnot.reporting import installed_batch_8702 as data
from carnot.reporting import v746_contract_runner as shared
from carnot.reporting import v746_retained_evidence as retained
from carnot.reporting import v747_contract_methods as contract
from carnot.reporting import v744_contract_methods as old_methods
from carnot.reporting import v746_contract_authority as historical_contract
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
BASE_MEASURE, BASE_VALIDATE = measurement.measure, previous.validate
BASE_CHILD, BASE_SOURCES = child, shared.source_hashes


def reproduce(raw: Path) -> Json:
    """Run original test bytes with the real unchanged watchdog, retaining its failure."""
    raw.mkdir(parents=True)
    original = subprocess.check_output(["git", "show", "HEAD:" + e.LEGACY], cwd=e.ROOT, timeout=10)
    source = raw / "original_8633.py"
    source.write_bytes(original)
    if e.reference(source)["sha256"] != e.LEGACY_PIN:
        raise ValueError("original_legacy_source_changed")
    old = next(
        n
        for n in ast.parse(original).body
        if isinstance(n, ast.FunctionDef) and n.name == "test_legacy_contract_snapshot"
    )
    fixed = next(
        n
        for n in ast.parse((e.ROOT / e.LEGACY).read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_legacy_contract_assertions"
    )
    exact = [ast.dump(n) for n in old.body] == [ast.dump(n) for n in fixed.body]
    receipt = child(
        "original_guard_failure",
        [
            "bwrap",
            "--bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--proc",
            "/proc",
            "--ro-bind",
            str(source),
            str(e.ROOT / e.LEGACY),
            "--",
            sys.executable,
            "-m",
            "pytest",
            *previous.OPTIONS,
            e.LEGACY,
        ],
        raw,
        deadline=120,
        expected=1,
    )
    text = Path(receipt["stdout_path"]).read_text()
    deltas = re.findall(r"Memory leak: \+(\d+)MB", text)
    return dict(
        receipt=receipt,
        original_source=e.reference(source),
        assertions_identical=exact,
        observed_delta_mb=int(deltas[0]) if deltas else None,
        passed=receipt["passed"] and "5 passed, 1 error" in text and bool(deltas) and exact,
    )


def measure(raw: Path, scratch: Path, inputs: Json, *, deadline_ns: int = 2**63) -> Json:
    """Retain guarded reproduction, actual installed parity and export child lifecycle."""
    reproduction = reproduce(raw / "guard-reproduction")
    receipts = raw / "legacy-children"
    with patch.dict(os.environ, CARNOT_8716_LEGACY_RECEIPTS=str(receipts)):
        result = BASE_MEASURE(raw / "installed", scratch, inputs, deadline_ns=deadline_ns)
    rows = [json.loads(p.read_bytes()) for p in sorted(receipts.glob("*.json"))]
    traces = [e.load(ref) for ref in result["traces"]]
    identity_matches = all(
        trace["ready"]["loaded_extension"] == result["installed_extension"] for trace in traces[6:]
    ) and all(trace["ready"]["loaded_extension"] is None for trace in traces[:6])
    identities = [
        dict(arm=arm, ready=trace["ready"])
        for arm, trace in zip(
            ["python-single", "python-batch", "native-single", "native-batch"],
            [traces[i] for i in [0, 1, 6, 7]],
            strict=True,
        )
    ]
    result.update(
        guard_reproduction=reproduction,
        memory_lifecycle_rows=rows,
        installed_identity_rows=identities,
        worker_exit_rows=[
            dict(
                actual_exit=t["actual_exit"],
                argv=t["argv"],
                start_ns=t["start_ns"],
                end_ns=t["end_ns"],
                reaped=not Path(f"/proc/{t['ready']['pid']}").exists(),
            )
            for t in traces
        ],
        frozen_installed_workload=dict(
            wheel=result["installed_wheel_reference"],
            config=result["config_reference"],
            extension=result["loaded_extension_reference"],
            requests=inputs["requests"],
            method=inputs["method"],
            configuration_template=dict(head=inputs["head"], max_batch_size=16, item_budget=1024),
            entry="carnot.inference.resident_protocol",
        ),
    )
    result["isolation_references"] = [e.reference(p) for p in sorted(receipts.glob("*.json"))]
    result["isolation_references"] += [reproduction["original_source"]]
    result["isolation_references"] += [
        dict(
            path=reproduction["receipt"][s + "_path"], sha256=reproduction["receipt"][s + "_sha256"]
        )
        for s in ["stdout", "stderr"]
    ]
    result["passed"] &= (
        reproduction["passed"]
        and bool(rows)
        and all(r["passed"] for r in rows)
        and identity_matches
        and all(r["reaped"] for r in result["worker_exit_rows"])
    )
    return result


def replay(path: Path, authority: Json | None = None) -> bool:
    """Require independently supplied work identity and all additional process receipts."""
    try:
        value = json.loads(path.read_bytes())
        if authority is None or value["primitive_reference"] != authority:
            return False
        work = e.load(authority)
        seal = e.load(work["seal_reference"])
        if seal != {k: v for k, v in work.items() if k != "seal_reference"}:
            return False
        missing = {
            (r["path"], r.get("sha256"))
            for r in work["inputs"]["gate_check_summary"]
            if r.get("observed_state", r.get("operand_state")) == "missing_file"
        }
        measured = work["measurement"]
        refs = work["sources"] + work["inputs"]["references"]
        refs += measured.get("traces", []) + measured.get("isolation_references", [])
        refs += [e.load(ref)["stderr_reference"] for ref in measured.get("traces", [])]
        refs += [
            measured[k]
            for k in [
                "installed_wheel_reference",
                "loaded_extension_reference",
                "config_reference",
                "installed_identity_reference",
            ]
            if measured.get(k)
        ]
        for ref in refs:
            operand = Path(ref["path"])
            if not operand.is_file() and (ref["path"], ref.get("sha256")) in missing:
                continue
            if not operand.is_file() or e.reference(operand)["sha256"] != ref["sha256"]:
                return False
        children = work["validation"]["children"]
        if children:
            nodes = [n for ref in children for n in e.load(ref)["node_ids"]]
            expected = next(
                (
                    r["passed"]
                    for r in work["validation"]["checks"]
                    if r["name"] == "exact_disjoint_children"
                ),
                True,
            )
            if retained.aggregate(children, nodes)["passed"] != expected:
                return False
        return bool(e.build(work, Path(work["output"])) == value)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(output: Path, *, root: Path = e.ROOT, private: bool = False) -> int:
    """Reuse bounded execution and unchanged publication with this branch's frozen sources."""

    def sources() -> list[Json]:
        """Include every changed test and independently pinned scheduling operand."""
        return BASE_SOURCES() + [
            e.reference(e.ROOT / p)
            for p in e.OWNED
            + e.TESTS
            + [e.LEGACY, "openspec/change-proposals/research-roadmap-v746-preserved-20261102.md"]
        ]

    with (
        patch.multiple(previous, e=e, validate=validate),
        patch.object(measurement, "measure", measure),
        patch.object(shared, "source_hashes", sources),
    ):
        return int(previous.run(output, root=root, private=private))


def main(argv: list[str] | None = None) -> int:
    """Reject private checkout aliases before creating evidence; bound the whole invocation."""
    progress("8716_start_no_model_load_MODEL_SPECS=[]_language_model_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261102")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", nargs=2)
    args = parser.parse_args(argv)
    if args.date != "20261102":
        return 2
    if args.private_output and (
        args.private_output.absolute().is_relative_to(e.ROOT)
        or args.private_output.resolve().is_relative_to(e.ROOT)
    ):
        return 2
    if args.cold_replay:
        authority = dict(zip(["path", "sha256"], args.authority)) if args.authority else None
        return int(not replay(args.cold_replay, authority))
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            root=args.root,
            private=bool(args.private_output),
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def historical_schedule(raw: Path) -> list[str]:
    """Give unchanged V746 producer tests their own authenticated historical schedule."""
    design = e.ROOT / "openspec/change-proposals/research-roadmap-v746-preserved-20261102.md"
    text = design.read_text()
    envelope = json.loads(
        text.split("<!-- V746_TASK_CONTRACT_START -->")[1].split("```json")[1].split("```")[0]
    )
    if old_methods.tasks_digest(envelope["tasks"]) != historical_contract.PIN:
        raise ValueError("historical_schedule_changed")
    original_design = raw / "v746-design.md"
    original_design.write_text(
        re.sub(r'^  "canonical_tasks_sha256": "[^\n]*\n', "", text, flags=re.M)
    )
    snapshot = raw / "v746-schedule.yaml"
    snapshot.write_text(
        yaml.safe_dump(dict(milestone=envelope["milestone"], tasks=envelope["tasks"]))
    )
    manifest = raw / "v746-binding.json"
    e.atomic_json(
        manifest, dict(DESIGN=str(original_design), ACTIVE=str(snapshot), STAGED=str(snapshot))
    )
    return ["env", "CARNOT_8716_HISTORICAL_AUTHORITY=" + str(manifest)]


def pytest_configure(config: Any) -> None:
    """Bind only historical file locations, leaving original validators and assertions intact."""
    paths = json.loads(Path(os.environ["CARNOT_8716_HISTORICAL_AUTHORITY"]).read_bytes())
    config._isolation8716 = patch.multiple(historical_contract, **paths)
    config._isolation8716.__enter__()


def pytest_unconfigure(config: Any) -> None:
    """Restore original locations before the bounded pytest child exits."""
    config._isolation8716.__exit__(None, None, None)


def validate(raw: Path, scratch: Path, sources: list[Json], original: list[str]) -> Json:
    """Keep original, producer and owned execution disjoint while reusing durable custody."""
    schedule = historical_schedule(raw)

    def execute(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        """Only unchanged V746 batch tests see their old scheduling authority."""
        if name.startswith("tests-") and any(
            any(a.startswith(s + "::") for a in argv) for s in data.TESTS
        ):
            position = argv.index("pytest") + 1
            argv = (
                argv[:position]
                + ["-p", "carnot.reporting.installed_import_runner_8716"]
                + argv[position:]
            )
            argv = schedule + argv
        return dict(BASE_CHILD(name, argv, logs, **kwargs))

    with patch.multiple(previous, e=e, child=execute):
        result = BASE_VALIDATE(raw, scratch, sources, original)
    combined = result["coverage"]["combined"]["combined_reference"]
    cov = Coverage(data_file=combined["path"], config_file=False)
    cov.load()
    files = []
    for name in data.OWNED:
        _, statements, _, missing, _ = cov.analysis2(str(e.ROOT / name))
        files.append(dict(path=name, statements=len(statements), missing=missing))
    result["producer_coverage"] = dict(
        complete=all(r["statements"] and not r["missing"] for r in files), files=files
    )
    result["checks"].append(
        dict(
            name="retained_batch_producer_coverage", passed=result["producer_coverage"]["complete"]
        )
    )
    result["original_node_rows"] = [
        dict(unit_id=n, passed=result["original_node_union"]["passed"]) for n in original
    ]
    return result
