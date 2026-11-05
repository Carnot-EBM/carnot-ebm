"""REQ-REPORT-8840: retain missing authority without manufacturing a contract."""

from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v755_branch_contract as base
from carnot.reporting import v756_branch_contract as legacy
from carnot.reporting.v756_branch_qualification import verify_measured
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from scripts.experiment_template import normalize_artifact_for_template_write
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8840_v757_branch_contract",
    "exp8840-branch-contract",
    "2026.11.757",
)
DESIGN, ACTIVE, STAGED = legacy.DESIGN, legacy.ACTIVE, legacy.STAGED
ARCHIVE = "openspec/change-proposals/research-roadmap-v756-preserved-20261105.md"
PRIOR = "results/experiment_8827_v756_branch_contract.json"
CAPSTONE = "results/experiment_8839_v756_capstone.json"
PINS = {
    PRIOR: "578246c56225e6760c3fc1da952288a092ef5f4c9b50759cf481cf3c2f7a1a65",
    CAPSTONE: "468df336de8a09882212b9c3b8f1931fbff121cd4efc6b8e3bbecb03894235e9",
    ARCHIVE: "90e26450de62dbfb1f19cbc2bf78ac6bc645c3c8bdd7028dc11082c17fd814e6",
}
PIN = "99319d2a1b9ac1c77cba55e27430fdd6f5cbe035dcee16154a7a92f537bdf12c"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v757_branch_contract.py",
    "python/carnot/reporting/v757_branch_execution.py",
    CLI,
]
TEST = "tests/python/test_v757_branch_contract_8840.py"
reference, atomic_json, checksum = legacy.reference, legacy.atomic_json, legacy.checksum
sha256_file, canonical_hash, progress = legacy.sha256_file, legacy.canonical_hash, legacy.progress


def authenticate() -> Json:
    """Check historical bytes before allowing their receipts to act as frozen operands."""
    for path, pin in PINS.items():
        if sha256_file(ROOT / path) != "sha256:" + pin:
            raise ValueError("historical_authority_drift:" + path)
    prior: Json = json.loads((ROOT / PRIOR).read_bytes())
    for ref in (
        prior["code_config_hashes"] + prior["raw_shard_hashes"] + prior["source_artifact_hashes"]
    ):
        if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("historical_operand_drift:" + ref["path"])
    if not verify_measured(prior["original_owned_statement_coverage"], base.OWNED):
        raise ValueError("unqualified_frozen_v755")
    return prior


def missing_rows(prior: Json) -> list[Json]:
    """Read the original coverage receipt so the target list cannot drift with new tests."""
    report = json.loads(
        Path(prior["owned_statement_coverage"]["report_reference"]["path"]).read_bytes()
    )
    path = legacy.OWNED[2]
    row = next(v for k, v in report["files"].items() if k.endswith(path))
    return [dict(path=path, line=n, source=reference(ROOT / path)) for n in row["missing_lines"]]


def authority(root: Path) -> Json:
    """Require a complete design and exact full objects; active YAML cannot repair absence."""
    try:
        text = (root / DESIGN).read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        staged = (
            yaml.safe_load((root / STAGED).read_bytes())
            if (root / STAGED).is_file()
            else deepcopy(active)
        )
        if active["milestone"] == legacy.MILESTONE:
            active = deepcopy(staged)
        for roadmap in [staged, active]:
            Roadmap.model_validate(roadmap)
            for actual, expected in zip(roadmap["tasks"], tasks):
                if actual == dict(expected, prompt=base.encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
        checked = compare_contract(
            text, staged, active, milestone=MILESTONE, first_id=8840, count=13
        )
        ready = (
            checked["passed"]
            and tasks == staged["tasks"] == active["tasks"]
            and canonical_hash(tasks) == "sha256:" + PIN
            and len({t["deliverable"] for t in tasks}) == 13
            and "<!-- V757_TASK_CONTRACT_END -->" in text
        )
        rows = checked["rows"]
        for row in rows:
            row.update(
                cluster_id="administrative_contract",
                metric=int(row["matched"]),
                denominator=1,
                missing_reason=None if row["matched"] else "authority_mismatch",
                excluded=False,
                effective_independent_groups=0,
            )
        return dict(ready=ready, rows=rows, tasks=tasks, table=table, state="complete_contract")
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return dict(
            ready=False,
            rows=[],
            tasks=[],
            table=[],
            state="incomplete_contract",
            failure=dict(
                check="complete_design_authority",
                upstream=TASK,
                artifact_field="full_task_objects",
                expected="13 complete task objects and closing marker",
                observed="missing_field" if (root / DESIGN).is_file() else "missing_file",
                **reference(root / DESIGN),
            ),
        )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze supplied evidence and measured resources before original execution starts."""
    progress("v757_preconditions_before")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    prior = authenticate()
    contract = authority(root)
    branches = legacy.optional_admission(root, raw)
    if not contract["ready"]:
        branches["failures"].append(contract["failure"])
    methods = [
        dict(method=method, tests=tests, retrieved=True, result_inherited=False)
        for method, tests in [
            ("VIGIL", ["sealed delayed feedback", "untouched retention"]),
            (
                "Certified by Abstention",
                ["population sampling with replacement", "certificate availability"],
            ),
            ("Anechoic/Torx", ["operation boundaries", "host and communication cost"]),
        ]
    ]
    methods.append(
        dict(
            method="citation discovery",
            retrieved=False,
            missing_reason="Semantic Scholar citation endpoints unavailable in supplied scan",
        )
    )
    sources = [ROOT / p for p in PINS] + [
        root / p for p in [DESIGN, ACTIVE, STAGED] if (root / p).is_file()
    ]
    sources += [
        ROOT / p
        for p in [
            "research-references.md",
            "ops/arc_solve_registry.yaml",
            "ops/exclusion_manifest.yaml",
        ]
    ]
    sources += [
        Path(r["path"]) for r in branches["runtime"]["references"] + branches["arc"]["references"]
    ]
    snapshots = []
    for source in sources:
        target = raw / "inputs" / (sha256_file(source)[7:] + "-" + source.name)
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(source.read_bytes())
        snapshots.append(dict(source=reference(source), **reference(target)))
    for name in [DESIGN, ACTIVE, STAGED]:
        if (root / name).is_file():
            target = raw / "authority" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    work = dict(
        contract=contract,
        branches=branches,
        methods=methods,
        manifests=[],
        snapshots=snapshots,
        history=dict(
            original_child_qualified=False, references=[reference(ROOT / p) for p in PINS]
        ),
        original_missing=missing_rows(prior),
        run_date=run_date,
        owner_pid=os.getpid(),
        start_ns=start,
        end_ns=time.monotonic_ns(),
        raw=str(raw.absolute()),
        preconditions=dict(
            repository_virtualenv=Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
            installed_dependencies={
                p: importlib.metadata.version(p)
                for p in ["coverage", "pytest", "pydantic", "PyYAML"]
            },
            private_output_writable=True,
            free_bytes=shutil.disk_usage(raw).free,
            required_free_bytes=134217728,
            time_budget_s=1800,
            no_model_loads=True,
            no_device_probes=True,
            activation_authorized=False,
            staged_source_present=(root / STAGED).is_file(),
        ),
    )
    atomic_json(raw / "work.json", work)
    progress("v757_preconditions_after", len(snapshots))
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Reuse the receipt reducer while making failed checks and external absence explicit."""
    qualified = validation["original"]["complete"]
    prepared = dict(work, history=dict(work["history"], original_child_qualified=qualified))
    value = base.build(prepared, validation)
    checked = value["required_checks_passed"] and qualified
    resources = work["preconditions"]
    checked = (
        checked
        and resources["repository_virtualenv"]
        and resources["free_bytes"] >= resources["required_free_bytes"]
    )
    ready = checked and work["contract"]["ready"]
    verdict = "disqualified" if not checked else "null" if ready else "blocked"
    value.pop("preserved_v754_reference")
    value.update(
        experiment_id=8840,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7578840,
        honest_verdict=f"complete_{verdict}_"
        + ("incomplete_v757_design_authority" if verdict == "blocked" else "v757_branch_contract"),
        verdict_class=verdict,
        contract_ready_score=int(ready),
        required_checks_passed=checked,
        canonical_tasks_sha256=PIN,
        execution_venue="host",
        preconditions_checked=resources,
        runtime_repair_ready_score=int(work["branches"]["runtime"]["ready"]),
        arc_new_outcomes_available_score=int(work["branches"]["arc"]["ready"]),
        original_missing_statement_rows=validation["original"].get(
            "rows", work["original_missing"]
        ),
        original_owned_statement_coverage=validation["original"],
        source_method_map=work["methods"],
        preserved_v756_reference=work["history"],
        external_admission_rows=value["branch_admission_rows"],
        code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
        phase_spans=work.get("phase_spans", value["phase_spans"]),
    )
    value["censored_count"] = 13 - value["completed_count"]
    value["raw_shard_hashes"] += validation.get("references", [])
    path = Path(work["raw"]) / "validation.json"
    if path.is_file():
        value["raw_shard_hashes"].append(reference(path))
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Frozen operands and real statement traces establish administrative custody only; optional admission never gates independent CPU roots. Missing design bytes cannot be inferred from a queue.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from sealed inputs and SQLite traces rather than trusting a reported score."""
    try:
        value = json.loads(path.read_bytes())
        if value["required_checks_passed"] and not (
            verify_measured(value["owned_statement_coverage"], OWNED)
            and verify_measured(value["original_owned_statement_coverage"], legacy.OWNED)
        ):
            raise ValueError("unqualified_statements")
        for ref in (
            value["raw_shard_hashes"]
            + value["code_config_hashes"]
            + value["source_artifact_hashes"]
        ):
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
        ref = value["raw_shard_hashes"][0]
        work = json.loads(Path(ref["path"]).read_bytes())
        frozen = authority(Path(work["raw"]) / "authority")
        # Failure paths name their original location; the frozen copy must have identical bytes.
        if "failure" in frozen:
            frozen["failure"].update(reference(Path(work["contract"]["failure"]["path"])))
        if work["contract"] != frozen:
            raise ValueError("authority_drift")
        validation = (
            json.loads((Path(work["raw"]) / "validation.json").read_bytes())
            if (Path(work["raw"]) / "validation.json").is_file()
            else dict(
                receipts=value["validation_receipts"],
                coverage=value["owned_statement_coverage"],
                original=value["original_owned_statement_coverage"],
                references=[],
            )
        )
        expected = build(dict(work, raw_reference=ref), validation)
        passed = value == expected and checksum(value) == value["reproducibility_checksum"]
        return dict(passed=passed, reason="replay_passed" if passed else "semantic_drift")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
