"""REQ-REPORT-9028-HISTORY: qualify custody with preserved historical authority."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import replace
import difflib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
from xml.etree import ElementTree

from coverage import Coverage

from carnot.reporting import invocation_provenance_9000 as original
from carnot.reporting import scoped_provenance_9014 as old
from carnot.reporting import v767_branch_contract as authority
from carnot.reporting import v767_capstone as capstone
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured

Json = dict[str, Any]
ROOT = original.ROOT
NAME = "experiment_9028_v771_historical_provenance_qualification"
TASK = "exp9028-historical-provenance-qualification"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/historical_provenance_9028.py", CLI]
CHANGED = "python/carnot/reporting/v767_capstone.py"
TEST = "tests/python/test_historical_provenance_9028.py"
DIRECT = [*old.DIRECT, old.TEST]
CHECKS = old.CHECKS
RECEIPT_CONTROLS = (*old.RECEIPT_CONTROLS, "log", "command", "duplicate", "code")
AUTHORITY_CONTROLS = ("valid", "missing_file", "missing", "order", "prompt", "rehash", "successor")
ARCHIVE = "openspec/change-proposals/research-roadmap-v767-preserved-20261110.md"
QUEUE = "openspec/change-proposals/research-roadmap-v767-queue-preserved-20261110.yaml"
HISTORICAL_PRIMARY = "results/experiment_9014_v770_scoped_provenance_qualification.json"
BASELINE = ROOT / "results/raw" / NAME / "baseline/reproduction.json"
PINS = {
    ARCHIVE: "sha256:f6b4cd77f27fc64054a2dcb345eb300a4e694838ec541f00b868fee3331a203d",
    QUEUE: "sha256:03d7d8ce109e2372cbaf55d799553fcf099874c6673bc1b8cc176d20351bfea7",
    HISTORICAL_PRIMARY: "sha256:ac6037edc4bcb9fad3d7d62610153962090da35bd1b0cc597c68e7b3bd41dcdb",
    str(
        BASELINE.relative_to(ROOT)
    ): "sha256:fcfd3644f3fd42436ae9aa5bb9d7b7ee580f7aa9535b4a92f4c5b508de0da265",
}
reference, retain, checksum, run_check = old.reference, old.retain, old.checksum, old.run_check


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report completed work so a supervising parent can detect stalled children."""
    print(f"[exp9028] phase={phase} completed={completed} pending={pending}", flush=True)


def historical_authority(root: Path, source: Path = ROOT) -> Path:
    """Give the original reader preserved bytes without changing its independent digest."""
    for name, target in [(ARCHIVE, authority.DESIGN), (QUEUE, authority.ACTIVE)]:
        if reference(source / name)["sha256"] != PINS[name]:
            raise ValueError("historical_authority_bytes:" + name)
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    return root


def foreign_argv(*args: str) -> tuple[str, ...]:
    """Use the same foreign-directory transport as the retained qualification."""
    return tuple(
        str(ROOT / CLI) if a == str(ROOT / old.CLI) else a for a in old.foreign_argv(*args)
    )


def validation_plan(private: Path) -> list[CommandSpec]:
    """Reuse the original check roster, replacing only owned paths and adding its tests."""
    plan = old.validation_plan(private)
    substitutions = dict(zip([*old.OWNED, old.TEST], [*OWNED, TEST]))
    result = []
    for spec in plan:
        argv = [substitutions.get(a, a) for a in spec.argv]
        if spec.name == "direct_consumers":
            argv.append(old.TEST)
        if spec.name.startswith("ruff"):
            argv.extend([CHANGED, capstone.TEST])
        if spec.name == "spec_coverage":
            argv.append(old.TEST)
        if spec.name == "coverage_report":
            argv.append("--include=" + ",".join(str(ROOT / p) for p in OWNED))
        result.append(replace(spec, argv=tuple(argv)))
    rc = private / "coverage.ini"
    rc.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in [*OWNED, CHANGED])
        + "[report]\nexclude_lines=\n"
    )
    return result


def scope_manifest(private: Path) -> Json:
    """Freeze original node identities against the failed primary before current checks."""
    manifest = old.scope_manifest(private)
    prior = json.loads((ROOT / HISTORICAL_PRIMARY).read_bytes())
    original_nodes = [
        r
        for r in prior["original_test_node_hashes"]
        if r["node_id"].startswith(capstone.TEST + "::")
    ]
    current_nodes = [
        r
        for r in manifest["original_test_node_hashes"]
        if r["node_id"].startswith(capstone.TEST + "::")
    ]
    files = list(
        dict.fromkeys(
            [
                *OWNED,
                TEST,
                *DIRECT,
                *old.REUSED,
                CHANGED,
                "python/carnot/reporting/v767_branch_contract.py",
                old.CLI,
                *(p.split("::")[0] for _, p in old.E2E),
            ]
        )
    )
    manifest.update(
        owned_paths=OWNED,
        changed_paths=[CHANGED],
        direct_consumer_tests=DIRECT,
        original_v767_test_node_hashes=original_nodes,
        original_v767_test_nodes_preserved=current_nodes == original_nodes,
        code_test_hashes=[reference(ROOT / p) for p in files],
        commands=[
            dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s)
            for c in validation_plan(private)
        ],
        receipt_controls=["valid", *RECEIPT_CONTROLS],
        historical_authority_hashes=[reference(ROOT / p) for p in (ARCHIVE, QUEUE)],
        independent_task_digest=authority.PIN,
    )
    return manifest


def authenticate(root: Path, raw: Path) -> tuple[Json, Json]:
    """Preserve failed historical receipts and their evidence rather than repairing them."""
    evidence, health = old.authenticate(root, raw)
    for name, pin in PINS.items():
        path = root / name
        evidence["checks"].append(
            original.gate(
                path,
                "sha256",
                pin,
                reference(path)["sha256"] or "missing_file",
                "historical_authority_bytes",
            )
        )
        if path.is_file():
            evidence["references"].append(retain(path, raw))
    if reference(root / HISTORICAL_PRIMARY)["sha256"] == PINS[HISTORICAL_PRIMARY]:
        prior = json.loads((root / HISTORICAL_PRIMARY).read_bytes())
        operands = (
            prior["source_artifact_hashes"]
            + prior["raw_shard_hashes"]
            + prior["code_config_hashes"]
        )
        for index, ref in enumerate(operands):
            path = root / Path(ref["path"]).relative_to(ROOT)
            evidence["checks"].append(
                original.gate(
                    path,
                    "sha256",
                    ref["sha256"],
                    reference(path)["sha256"] or "missing_file",
                    "historical_9014_evidence",
                )
            )
            if path.is_file():
                evidence["references"].append(retain(path, raw))
            if index % 100 == 0:
                progress("historical_9014_custody", index + 1, len(operands) - index - 1)
        for field, expected in [
            ("verdict_class", "disqualified"),
            ("provenance_ready_score", 0),
            ("validation_scope_ready_score", 0),
            ("required_checks_passed", False),
        ]:
            evidence["checks"].append(
                original.gate(
                    root / HISTORICAL_PRIMARY,
                    field,
                    expected,
                    prior.get(field, "missing_field"),
                    "historical_9014_verdict",
                )
            )
        health["rows"].append(
            dict(
                experiment_id=9014,
                original_verdict=prior["honest_verdict"],
                original_verdict_class=prior["verdict_class"],
                original_receipts=prior["validation_receipts"],
                rejected_valid_control=prior["current_receipt_mutation_rows"][0],
            )
        )
    baseline = root / BASELINE.relative_to(ROOT)
    evidence["checks"].append(
        original.gate(baseline, "exists", True, baseline.is_file(), "original_failure_reproduction")
    )
    if reference(baseline)["sha256"] == PINS[str(BASELINE.relative_to(ROOT))]:
        evidence["references"].append(retain(baseline, raw))
        baseline_refs = json.loads(baseline.read_bytes())["references"]
        for ref in baseline_refs:
            path = root / Path(ref["path"]).relative_to(ROOT)
            evidence["checks"].append(
                original.gate(
                    path,
                    "sha256",
                    ref["sha256"],
                    reference(path)["sha256"] or "missing_file",
                    "original_failure_evidence",
                )
            )
            if path.is_file():
                evidence["references"].append(retain(path, raw))
        xml = root / BASELINE.parent.relative_to(ROOT) / "original.xml"
        count = (
            len(ElementTree.parse(xml).findall(".//testcase/failure"))
            if xml.is_file()
            else "missing_file"
        )
        evidence["checks"].append(
            original.gate(xml, "failure_count", 6, count, "original_six_failures")
        )
    evidence["failures"] = [r for r in evidence["checks"] if not r["passed"]]
    return evidence, health


def changed_statements(data: Path) -> Json:
    """Measure only changed shared-reader statements against its retained old source."""
    prior = json.loads((ROOT / HISTORICAL_PRIMARY).read_bytes())
    before = next(r for r in prior["code_config_hashes"] if r.get("source_path") == CHANGED)
    tracer = Coverage(data_file=str(data), config_file=False)
    tracer.set_option("report:exclude_lines", [])
    tracer.load()
    _, statements, excluded, missing, _ = tracer.analysis2(str(ROOT / CHANGED))
    changed = {
        i + 1
        for tag, _, _, start, end in difflib.SequenceMatcher(
            a=Path(before["path"]).read_text().splitlines(),
            b=(ROOT / CHANGED).read_text().splitlines(),
        ).get_opcodes()
        if tag in {"insert", "replace"}
        for i in range(start, end)
    }
    selected = set(statements) & changed
    return dict(
        path=CHANGED,
        statements=len(selected),
        statement_lines=sorted(selected),
        missing=sorted(selected & set(missing)),
        excluded=sorted(selected & set(excluded)),
        baseline_reference=before,
    )


def measure(private: Path, raw: Path) -> Json:
    """Keep real SQLite custody and separate shared-reader changes from legacy coverage."""
    report = measured([private / ".coverage"], OWNED, raw)
    report.pop("rows", None)
    changed = changed_statements(Path(report["data_reference"]["path"]))
    report["changed_statements"] = changed
    report["complete"] = bool(
        report["complete"]
        and changed["statements"]
        and not changed["missing"]
        and not changed["excluded"]
    )
    return report


def verify_coverage(report: Json) -> bool:
    """Reopen measured bytes; headline percentages cannot authorize current readiness."""
    try:
        if not verify_measured(report, OWNED):
            return False
        changed = changed_statements(Path(report["data_reference"]["path"]))
        return bool(
            report["complete"]
            and changed == report["changed_statements"]
            and changed["statements"]
            and not changed["missing"]
            and not changed["excluded"]
        )
    except (OSError, ValueError, KeyError):
        return False


def control_passed(row: Json) -> bool:
    """Use the existing primitive-exit check for each additional refused receipt mutation."""
    renamed = dict(row, name="failed") if row.get("name") in RECEIPT_CONTROLS else row
    return old.control_passed(renamed)


def build(work: Json) -> Json:
    """Reuse the retained schema while reducing only this frozen current scope."""
    value = old.build(work)
    controls, mutations = work["controls"], work["current_receipt_mutation_rows"]
    scoped = old.receipts_pass(work) and verify_coverage(work["validation"]["coverage"])
    control_ok = (
        len(controls) == len(original.CONTROLS)
        and {r["name"] for r in controls} == set(original.CONTROLS)
        and all(control_passed(r) for r in controls)
    )
    receipt_ok = (
        len(mutations) == 1 + len(RECEIPT_CONTROLS)
        and {r["name"] for r in mutations} == {"valid", *RECEIPT_CONTROLS}
        and all(control_passed(r) for r in mutations)
    )
    authority_ok = (
        work["validation_scope_manifest"]["original_v767_test_nodes_preserved"]
        and len(work["historical_authority_rows"]) == len(AUTHORITY_CONTROLS)
        and {r["name"] for r in work["historical_authority_rows"]}
        == {"authority_" + k for k in AUTHORITY_CONTROLS}
        and all(
            old.control_passed(
                dict(r, name="valid" if r["name"] == "authority_valid" else "failed")
            )
            for r in work["historical_authority_rows"]
        )
    )
    terminal_ok = work.get("terminal_validation", {}).get("passed") is True
    failures = work["evidence"]["failures"]
    checks_passed = bool(scoped and control_ok and receipt_ok and authority_ok and terminal_ok)
    ready = int(checks_passed and not failures)
    cls = "blocked" if failures else "positive" if ready else "disqualified"
    rows = []
    for arm, names, collection in [
        ("serializer", original.CONTROLS, controls),
        ("receipt_policy", ("valid", *RECEIPT_CONTROLS), mutations),
        (
            "historical_authority",
            tuple("authority_" + k for k in AUTHORITY_CONTROLS),
            work["historical_authority_rows"],
        ),
    ]:
        by_name = {r["name"]: r for r in collection}
        for name in names:
            row = by_name.get(name)
            checked = (
                dict(row, name="valid" if name == "authority_valid" else "failed")
                if row and arm == "historical_authority"
                else row
            )
            passed = bool(checked and control_passed(checked))
            rows.append(
                dict(
                    unit_id=name,
                    group_id=name,
                    arm=arm,
                    condition=name,
                    metric="expected_control_outcome",
                    value=int(passed),
                    numerator=int(passed),
                    denominator=1,
                    status="censored" if row is None else "completed" if passed else "failed",
                    exclusion_reason=None
                    if passed
                    else "not_run_external_block"
                    if row is None and failures
                    else "unfinished_current_control"
                    if row is None
                    else "control_failed",
                )
            )
    value.update(
        experiment_id=9028,
        task_id=TASK,
        title="Historical authority provenance qualification",
        milestone="2026.11.771",
        run_date="20261112",
        random_seed=7719028,
        honest_verdict="complete_blocked_" + failures[0]["check"]
        if failures
        else "complete_" + cls + "_historical_provenance",
        verdict_class=cls,
        status="success" if ready else "blocked" if failures else "error",
        required_checks_passed=checks_passed,
        provenance_ready_score=ready,
        validation_scope_ready_score=ready,
        claim_scope="Current scoped provenance custody with immutable historical test inputs; no repository health or verifier benefit claim.",
        current_receipt_mutation_rows=mutations,
        historical_authority_rows=work["historical_authority_rows"],
        rows=rows,
        intended_count=len(rows),
        independent_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        historical_reproduction_reference=work["baseline_reference"],
    )
    value["acceptance_gates"] = {
        k: dict(
            passed=v,
            principle="Actual current normal exits qualify custody only; historical failed verdicts remain failed.",
        )
        for k, v in dict(
            authenticated_inputs=not failures,
            scoped_validation=scoped,
            serializer_controls=control_ok,
            current_receipt_controls=receipt_ok,
            historical_authority=authority_ok,
            cold_replay_and_scanners=terminal_ok,
        ).items()
    }
    value["cited_upstream_artifacts"].append(
        dict(
            experiment_id=9014,
            sha256=PINS[HISTORICAL_PRIMARY],
            fields_imported=[
                "verdict_class",
                "validation_receipts",
                "current_receipt_mutation_rows",
                "code_config_hashes",
                "original_test_node_hashes",
            ],
        )
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Retained historical bytes remain separate from actual current checks. Fixture controls grant no live science credit.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def authority_controls(private: Path, raw: Path, manifest: Json) -> list[Json]:
    """Run original child paths with exact history and private refused substitutions."""
    source = historical_authority(private / "valid")
    authority.mutations(source, private / "mutations")
    missing = historical_authority(private / "missing_file")
    (missing / authority.DESIGN).unlink()
    successor = private / "successor"
    for name, target in [(ARCHIVE, authority.DESIGN), (QUEUE, authority.ACTIVE)]:
        path = successor / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            (ROOT / name.replace("v767", "v770").replace("20261110", "20261111")).read_bytes()
        )
    rows = []
    for kind in AUTHORITY_CONTROLS:
        root = (
            source
            if kind == "valid"
            else missing
            if kind == "missing_file"
            else successor
            if kind == "successor"
            else private / "mutations" / kind
        )
        output = private / kind / "output" / (capstone.NAME + ".json")
        spec = CommandSpec(
            "authority_" + kind,
            (
                *old.foreign_argv()[:-1],
                str(ROOT / capstone.CLI),
                "--authority-root",
                str(root),
                "--output",
                str(output),
            ),
            "private_historical_authority",
            120,
        )
        receipt = run_check(spec, raw, manifest)
        expected = int(kind != "valid")
        rows.append(
            dict(
                name="authority_" + kind,
                expected_exit=expected,
                observed_exit=receipt["exit_code"],
                passed=receipt["normal_exit"] and receipt["exit_code"] == expected,
                receipt=receipt,
                fixture_only=True,
                authority_references=[
                    retain(p, raw)
                    for p in sorted(root.rglob("*"))
                    if p.is_file() and "output" not in p.parts
                ],
            )
        )
        progress("authority_controls", len(rows), 7 - len(rows))
    return rows


def receipt_controls(work: Json, private: Path) -> list[Json]:
    """Check actual current receipts, then reject each mutation in a separate copy."""
    rows = []
    for kind in ("valid", *RECEIPT_CONTROLS):
        changed = deepcopy(work)
        old.mutate_receipt(changed, kind)
        path = private / (kind + ".json")
        atomic_json(path, changed)
        receipt = run_check(
            CommandSpec(
                "receipt_" + kind,
                foreign_argv("--check-receipts", str(path)),
                "private_receipt_control",
                60,
            ),
            Path(work["raw"]) / "receipt_controls",
            work["validation_scope_manifest"],
        )
        expected = int(kind != "valid")
        rows.append(
            dict(
                name=kind,
                expected_exit=expected,
                observed_exit=receipt["exit_code"],
                passed=receipt["normal_exit"] and receipt["exit_code"] == expected,
                receipt=receipt,
                input_reference=retain(path, Path(work["raw"])),
                fixture_only=True,
            )
        )
        progress("receipt_controls", len(rows), 1 + len(RECEIPT_CONTROLS) - len(rows))
    return rows


def replay(path: Path) -> Json:
    """Independently reduce saved evidence and reject even rehashed aggregate drift."""
    try:
        value = json.loads(path.read_bytes())
        if checksum(value) != value["reproducibility_checksum"]:
            raise ValueError("checksum_drift")
        for ref in (
            value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
            + value["code_config_hashes"]
        ):
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        passed = value == build(work)
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False)


def terminal(candidate: Path, raw: Path, manifest: Json) -> Json:
    """Keep the original validators unchanged and require actual fresh-process replay."""
    specs = [
        CommandSpec("cold_replay", foreign_argv("--cold-replay", str(candidate)), "terminal", 60)
    ]
    for name, script, flag in [
        ("adversarial", "adversarial_verify.py", "--json"),
        ("strict_rows", "verdict_row_consistency_lint.py", "--strict"),
    ]:
        specs.append(
            CommandSpec(
                name,
                (str(ROOT / ".venv/bin/python"), "scripts/" + script, flag, str(candidate)),
                "terminal",
                60,
            )
        )
    receipts = [run_check(spec, raw, manifest) for spec in specs]
    return dict(passed=all(r["passed"] and r["normal_exit"] for r in receipts), receipts=receipts)


def execute(raw: Path, input_root: Path) -> Json:
    """Freeze scope before reducing history, then require the full current check roster."""
    started = time.monotonic()
    private = Path(tempfile.mkdtemp(prefix="carnot-9028-"))
    manifest = scope_manifest(private)
    atomic_json(raw / "validation_scope_manifest.json", manifest)
    frozen_ns = time.monotonic_ns()
    progress("preconditions_before")
    environment = original.environment(raw)
    dependencies = {
        name: importlib.util.find_spec(name) is not None for name in ("yaml", "numpy", "sklearn")
    }
    environment["dependencies"].update(dependencies)
    environment["checks"].append(
        original.gate(
            raw, "consumer_dependencies", True, all(dependencies.values()), "consumer_dependencies"
        )
    )
    evidence, health = authenticate(input_root, raw)
    evidence["checks"].extend(environment["checks"])
    evidence["failures"].extend(r for r in environment["checks"] if not r["passed"])
    code = [
        dict(retain(Path(r["path"]), raw), source_path=str(Path(r["path"]).relative_to(ROOT)))
        for r in manifest["code_test_hashes"]
    ]
    progress("preconditions_after", len(evidence["checks"]), 0)
    controls, authorities = [], []
    if not evidence["failures"]:
        authorities = authority_controls(
            private / "authority", raw / "authority_controls", manifest
        )
        for index, kind in enumerate(original.CONTROLS):
            output = private / kind / (original.NAME + ".json")
            spec = CommandSpec(
                kind,
                original.foreign_argv(
                    "--date", "20261111", "--fixture-e2e", str(output), "--control", kind
                ),
                "private_serializer_control",
                60,
            )
            receipt = run_check(spec, raw / "serializer_controls", manifest)
            expected = int(kind in {"missing_ledger", "contradictory_current"})
            controls.append(
                dict(
                    name=kind,
                    expected_exit=expected,
                    observed_exit=receipt["exit_code"],
                    passed=receipt["normal_exit"] and receipt["exit_code"] == expected,
                    receipt=receipt,
                    fixture_only=True,
                )
            )
            progress("serializer_controls", index + 1, len(original.CONTROLS) - index - 1)
    measured_at = time.monotonic()
    if measured_at - started >= 3300:
        raise TimeoutError("measurement_deadline")
    progress("validation_before", 0, len(CHECKS))
    previous = os.environ.get("COVERAGE_RCFILE")
    os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
    try:
        receipts = []
        for index, spec in enumerate(validation_plan(private)):
            receipts.append(run_check(spec, raw / "validation", manifest))
            progress("validation", index + 1, len(CHECKS) - index - 1)
    finally:
        if previous is None:
            os.environ.pop("COVERAGE_RCFILE", None)
        else:
            os.environ["COVERAGE_RCFILE"] = previous
    coverage = (
        measure(private, raw / "coverage")
        if (private / ".coverage").is_file()
        else dict(complete=False, percent=0, files=[])
    )
    work: Json = dict(
        raw=str(raw),
        private=str(private),
        frozen_ns=frozen_ns,
        validation_scope_manifest=manifest,
        evidence=evidence,
        environment=environment,
        repository_health=health,
        code_references=code,
        raw_references=[],
        controls=controls,
        historical_authority_rows=authorities,
        baseline_reference=reference(BASELINE),
        current_receipt_mutation_rows=[],
        validation=dict(receipts=receipts, coverage=coverage),
        duration_s=time.monotonic() - started,
        phase_spans=[
            dict(phase="authentication_and_controls", duration_s=measured_at - started),
            dict(phase="validation", duration_s=time.monotonic() - measured_at),
        ],
    )
    work["current_receipt_mutation_rows"] = receipt_controls(work, private / "receipt_inputs")
    work["raw_references"] = [
        retain(p, raw)
        for name in [*original.CONTROLS, "receipt_inputs"]
        for p in sorted((private / name).rglob("*"))
        if p.is_file()
    ]
    work["raw_references"].append(retain(private / "coverage.ini", raw))
    work["raw_references"].extend(
        reference(p) for p in sorted(raw.rglob("*")) if p.is_file() and "objects" not in p.parts
    )
    work["duration_s"] = time.monotonic() - started
    atomic_json(raw / "work.json", work)
    progress("validation_after", len(receipts), 0)
    return work


def main(argv: list[str] | None = None) -> int:
    """Publish through the existing checked writer; private controls cannot rewrite history."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--check-receipts", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.check_receipts or args.cold_replay:
        report = (
            replay(args.cold_replay)
            if args.cold_replay
            else dict(passed=old.receipts_pass(json.loads(args.check_receipts.read_bytes())))
        )
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261112":
        parser.error("this consumer requires --date 20261112")
    output = (args.output or ROOT / "results" / (NAME + ".json")).absolute()
    if args.from_work and output.is_relative_to(ROOT):
        parser.error("private replay publication requires output outside the checkout")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    progress("qualification_before")
    if args.from_work:
        work = json.loads(args.from_work.read_bytes())
        work["raw"] = str(raw)
        atomic_json(raw / "work.json", work)
    else:
        work = execute(raw, args.input_root)
    candidate = raw / "preflight.json"
    atomic_json(candidate, build(work))
    work["terminal_validation"] = terminal(
        candidate, raw / "preflight_checks", work["validation_scope_manifest"]
    )
    work["terminal_rejected"] = not work["terminal_validation"]["passed"]
    atomic_json(raw / "work.json", work)
    value = build(work)
    published = publish_primary(
        output,
        value,
        lambda p: terminal(p, raw / "publication_checks", work["validation_scope_manifest"]),
    )
    progress("publication_complete", 1, 0)
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                provenance_ready_score=value["provenance_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
