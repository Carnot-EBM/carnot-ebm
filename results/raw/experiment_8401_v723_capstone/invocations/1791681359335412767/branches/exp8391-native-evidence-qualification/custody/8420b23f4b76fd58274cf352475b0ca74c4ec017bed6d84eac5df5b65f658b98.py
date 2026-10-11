"""REQ-REPORT-8391: qualify native evidence without replacing historical authority.

The adapter keeps the shipped arithmetic, publication and finding consumers.
Real invocation coverage has its own custody and cannot inherit a prior pass.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import sysconfig
from typing import Any, Iterator
from unittest.mock import patch

import coverage
import yaml

from carnot.reporting import native_direct_8379 as base
from carnot.reporting import native_direct_runner_8379 as runner
from carnot.reporting import v723_contract_methods as authority
from carnot.reporting import v717_contract_runner as commands
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
ROOT = base.ROOT
NAME = "experiment_8391_v723_native_evidence_qualification"
TASK = "exp8391-native-evidence-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_native_evidence_qualification_8391.py"
OWNED = ["python/carnot/reporting/native_evidence_8391.py", CLI]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:ed8cbcbb7b470a13f9ba1b99720341edf629234c38895f927706ea552a290414"
ORIGINAL = "results/experiment_8379_v722_native_direct_parity.json"
PRESERVED = [ORIGINAL, *base.k.SOURCES, "scripts/research_conductor.py", "research-roadmap.yaml"]
PRESERVED += [
    "openspec/change-proposals/v717-local-learning-protocol.json",
    "openspec/change-proposals/v721-deployment-protocol.json",
    base.authority.PROTOCOL,
]
reference = base.reference
BASE_BUILD, BASE_REPLAY = base.build, base.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush invocation identity and real counts so the supervisor sees progress."""
    print(f"[exp8391] phase={phase} completed={completed} pending={pending}", flush=True)


def bind(root: Path, raw: Path) -> tuple[Json | None, list[Json], list[Json]]:
    """The complete current contract grants authority only to this exact task."""
    current = authority.authority(root, raw / "authority")
    gates = list(current["gate_check_summary"])
    tasks = [t for t in current["tasks"] if t["id"] == TASK]
    if len(tasks) != 1 or canonical_hash(tasks[0]) != TASK_PIN:
        gates.append(
            authority.failure(
                root / authority.ACTIVE,
                "exact_task_hash",
                TASK_PIN,
                canonical_hash(tasks[0]) if tasks else None,
            )
        )
    refs = []
    pinned = [
        (authority.PROTOCOL, authority.PROTOCOL_PIN),
        (authority.METHODS, authority.METHODS_PIN),
        (base.authority.PROTOCOL, base.authority.PROTOCOL_PIN),
        (base.authority.METHODS, base.authority.METHODS_PIN),
        (authority.DESIGN, None),
        (authority.ACTIVE, None),
    ]
    protocol = None
    for name, pin in pinned:
        source = root / name
        observed = reference(source)["sha256"] if source.is_file() else None
        if observed is None or (pin and observed != pin):
            gates.append(authority.failure(source, "input_hash", pin or "present", observed))
            continue
        refs.append(reference(source))
        if name == base.authority.PROTOCOL:
            protocol = json.loads(source.read_bytes())
    if protocol is not None:
        operands = [
            protocol["checkpoint"],
            protocol["scientific_protocol"],
            protocol["preserved_v721_protocol"],
            *[protocol["panels"][key] for key in ("vectors", "metadata", "natural_predictor")],
        ]
        for operand in operands:
            source = Path(operand["path"])
            observed = reference(source)["sha256"] if source.is_file() else None
            if observed != operand["sha256"]:
                gates.append(authority.failure(source, "operand_hash", operand["sha256"], observed))
            else:
                refs.append(reference(source))
    for ref in refs:
        source = Path(ref["path"])
        saved = raw / "inputs" / (ref["sha256"][7:] + source.suffix)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(source.read_bytes())
        ref["snapshot_path"] = str(saved)
    atomic_json(raw / "current_authority.json", current)
    return (None if gates else protocol), refs, gates


@contextmanager
def bindings() -> Iterator[None]:
    """Keep all changes local to this process so old consumers stay immutable."""
    with patch.multiple(
        base,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        bind=bind,
        build=build,
        progress=progress,
    ):
        yield


def reproduce(private: Path, raw: Path) -> Json:
    """A merged file without shards must still produce the real combine error."""
    old = json.loads((ROOT / ORIGINAL).read_bytes())
    receipt = next(r for r in old["validation_receipts"] if r["name"] == "coverage_combine")
    directory = Path(receipt["stdout_path"]).parent.parent
    topology = [reference(p) for p in sorted(directory.glob(".coverage*")) if p.is_file()]
    saved = private / "empty-combine"
    saved.mkdir()
    (saved / ".coverage").write_bytes((directory / ".coverage").read_bytes())
    actual = child(
        "reproduce_empty_combine",
        [
            sys.executable,
            "-m",
            "coverage",
            "combine",
            "--keep",
            "--append",
            "--data-file=" + str(saved / ".coverage"),
            str(saved),
        ],
        raw,
        expected=1,
        deadline=30,
        scope="historical",
    )
    return dict(
        passed=actual["passed"] and "No data to combine" in Path(actual["stdout_path"]).read_text(),
        original_receipt=receipt,
        topology=topology,
        reproduction=actual,
    )


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Record native execution and frozen historical replay as separate observations."""
    with bindings():
        work = dict(base.measure(root, raw, private))
    work["root"] = str(root)
    work["current_authority"] = json.loads((raw / "current_authority.json").read_bytes())
    work["preconditions_checked"]["task_sha256"] = next(
        (canonical_hash(t) for t in work["current_authority"]["tasks"] if t["id"] == TASK), None
    )
    work["historical_consumers"] = authority.historical(raw / "historical")
    work["coverage_failure_reproduction"] = reproduce(private, raw / "coverage_failure")
    toolchain = child("rustc_version", ["rustc", "-vV"], raw / "toolchain", deadline=30)
    work["extension"].update(
        abi=sysconfig.get_config_var("SOABI"), python_version=sys.version, toolchain=toolchain
    )
    work["repository_health"] = [
        r
        for r in json.loads((ROOT / ORIGINAL).read_bytes())["validation_receipts"]
        if r.get("scope") == "global"
    ]
    work["code_config_hashes"].extend(reference(ROOT / p) for p in PRESERVED[1:])
    for source in (ROOT / ORIGINAL, Path(work["historical_consumers"]["primary_path"])):
        saved = raw / "history_inputs" / source.name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(source.read_bytes())
        work["source_artifact_hashes"].append(dict(reference(source), snapshot_path=str(saved)))
    atomic_json(raw / "measurement.json", work)
    return work


def retain_shards(private: Path) -> int:
    """Save real pre-combine shards before report commands can delete them."""
    progress("coverage_custody_before")
    shards = sorted(p for p in private.glob(".coverage.*") if p.is_file())
    saved = private / "coverage_shards"
    saved.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in shards:
        target = saved / source.name
        target.write_bytes(source.read_bytes())
        manifest.append(dict(reference(target), source_path=str(source)))
    atomic_json(
        saved / "manifest.json", dict(files=manifest, coverage_version=coverage.__version__)
    )
    progress("coverage_custody_after", len(shards), 0)
    return int(len(shards) < 2)


def coverage_evidence(private: Path) -> Json:
    """Check the actual combined database as well as its human-readable report."""
    directory = private / "coverage_shards"
    shards = sorted(p for p in directory.glob(".coverage.*") if p.is_file())
    manifest = [reference(p) for p in shards]
    result = dict(
        passed=False,
        mode="parallel_real_test_and_cli_children",
        manifest=manifest,
        report_reference=None,
        database_reference=None,
        owned_statements=None,
        missing_statements=None,
    )
    report, data = private / "coverage.json", private / ".coverage"
    if not report.is_file() or not data.is_file() or len(shards) < 2:
        return result
    value = json.loads(report.read_bytes())
    cov = coverage.Coverage(config_file=False, data_file=str(data))
    cov.load()
    statements, missing = 0, 0
    for name in OWNED:
        path = ROOT / name
        _, lines, _, absent, _ = cov.analysis2(str(path))
        summary = next(
            (
                v["summary"]
                for k, v in value["files"].items()
                if (ROOT / k).resolve() == path.resolve()
            ),
            {},
        )
        if summary.get("num_statements") != len(lines) or summary.get("missing_lines") != len(
            absent
        ):
            return result
        statements += len(lines)
        missing += len(absent)
    result.update(
        passed=statements > 0 and missing == 0,
        report_reference=reference(report),
        database_reference=reference(data),
        owned_statements=statements,
        missing_statements=missing,
    )
    return result


def build(work: Json, receipts: list[Json]) -> Json:
    """Numerical equality needs real coverage and immutable consumers for readiness."""
    value = dict(BASE_BUILD(work, receipts))
    evidence = coverage_evidence(Path(work["preconditions_checked"]["scratch"]))
    for ref in evidence["manifest"]:
        saved = (
            Path(work["primitive_reference"]["path"]).parent
            / "coverage_shards"
            / Path(ref["path"]).name
        )
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(Path(ref["path"]).read_bytes())
        ref["snapshot_path"] = str(saved)
    current = work["current_authority"]["activated"]
    historical = work["historical_consumers"]
    consumers = historical["ready"] and historical["missing_design_stays_blocked"]
    checks = value["required_checks_passed"] and work["coverage_failure_reproduction"]["passed"]
    instrumentation = evidence["passed"]
    ready = bool(
        value["native_parity_ready_score"] and current and consumers and checks and instrumentation
    )
    verdict = (
        "blocked"
        if not work["input_ready"] and checks
        else "circular_positive"
        if ready
        else "disqualified"
    )
    value.update(
        experiment_id=8391,
        task_id=TASK,
        milestone="2026.10.723",
        verdict_class=verdict,
        honest_verdict="complete_" + verdict + "_native_evidence_qualification",
        native_parity_ready_score=int(ready),
        required_checks_passed=bool(checks and (instrumentation or not work["input_ready"])),
        flagged_adversarial=not bool(checks and (instrumentation or not work["input_ready"])),
        coverage_file_manifest=evidence["manifest"],
        coverage_combination_mode=evidence["mode"],
        coverage_receipts=[r for r in receipts if "coverage" in r.get("name", "")],
        owned_coverage=evidence,
        current_authority=work["current_authority"],
        historical_consumers=historical,
        coverage_failure_reproduction=work["coverage_failure_reproduction"],
        repository_health=work["repository_health"],
        extension_abi=work["extension"]["abi"],
        extension_toolchain=work["extension"]["toolchain"],
        methodology_note="Finite oracle-checked native parity only. Frozen V717/V721/V722 policies; H1=-0.00390625 and H2=0 stay closed exposed-development findings. Dyadic-logit policy is separate. No speed or generalization claim.",
    )
    value["acceptance_gates"].update(
        real_owned_coverage=instrumentation,
        current_authority=current,
        immutable_historical_consumers=consumers,
        coverage_failure_reproduced=work["coverage_failure_reproduction"]["passed"],
    )
    value["field_principles"].update(
        {
            k: "Bind actual invocation coverage, compiled arithmetic and immutable authority; no scientific benefit follows."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute current authority from sealed files before the unchanged native replay."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(Path(value["measurement_reference"]["path"]).read_bytes())
        private = Path(work["preconditions_checked"]["scratch"])
        sealed = [dict(r, exists=True) for r in work["source_artifact_hashes"]]
        with frozen_inputs(sealed, private / "replay_inputs"):
            checked = authority.authority(Path(work["root"]), private / "replay_authority")
        if any(
            checked[k] != work["current_authority"][k]
            for k in ("activated", "tasks", "canonical_tasks_sha256")
        ):
            return False
        with bindings():
            return bool(BASE_REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def plan(private: Path) -> list[Json]:
    """Coverage.py owns real child shards; pytest-cov cannot pre-combine them."""
    with patch.object(commands, "m", sys.modules[__name__]):
        frozen = commands.manifest(private)
    os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
    os.environ["COVERAGE_FILE"] = str(private / ".coverage")
    for command in frozen:
        command["deadline_s"] = 600 if command["name"] == "owned_tests" else command["deadline"]
        if command["name"] == "coverage_combine":
            command["argv"].insert(2, "--keep")
    frozen.insert(
        1,
        dict(
            name="private_E2E003_native_and_source_isolation",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_native_direct_parity_8379.py",
                "tests/python/test_v723_contract_methods_8388.py",
                "-k",
                "req_verify_8379 or historical_source_alias_rejected",
                "--basetemp=" + str(private / "native-consumers"),
            ],
            deadline_s=600,
            scope="owned",
        ),
    )
    index = next(i for i, c in enumerate(frozen) if c["name"] == "coverage_combine")
    frozen.insert(
        index,
        dict(
            name="coverage_custody",
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--coverage-custody",
                str(private),
            ],
            deadline_s=60,
            scope="owned",
        ),
    )
    return [dict(c) for c in frozen]


def main(argv: list[str] | None = None) -> int:
    """Reuse bounded publication and cold children with a 4800-second cap."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--coverage-custody":
        return retain_shards(Path(args[1])) if len(args) == 2 else 1
    with (
        bindings(),
        patch.multiple(
            runner,
            e=sys.modules[__name__],
            plan=plan,
            SCRATCH=Path.home() / ".cache/carnot-exp8391-private",
        ),
    ):
        return int(runner.main(args))
