"""REQ-REPORT-8388: current custody cannot create missing historical authority."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v722_contract_methods as old
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v721_capstone_evidence import frozen_inputs
from carnot.reporting.v685_authority_lifecycle import assess_authorities, tasks_digest
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = old.ROOT, old.DESIGN, old.ACTIVE, old.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8388_v723_contract_methods",
    "exp8388-contract-methods",
    "2026.10.723",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v723_contract_methods_8388.py"
PROTOCOL = "openspec/change-proposals/v723-service-and-label-protocol.json"
METHODS = "openspec/change-proposals/v723-methods-manifest.json"
PROTOCOL_PIN = "sha256:0ed6f87ede5046c2113257c9bff4dacfee88415fddb8d7be18506668f6301e90"
METHODS_PIN = "sha256:63a373b38b409b1124b1a317d100d0264230a5f7c3e3abb8c8ea7c8a29e08a15"
OWNED = [
    "python/carnot/reporting/v723_contract_methods.py",
    "python/carnot/reporting/v723_contract_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def failure(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name the current producer so a missing operand cannot inherit another task's authority."""
    return dict(old.failure(path, field, expected, observed), upstream_id=TASK)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual boundaries so the operator can distinguish progress from a stall."""
    print(f"[exp8388] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path, raw: Path) -> Json:
    """Full objects bind prompts as well as titles; active bytes alone grant activation."""
    staged = root / STAGED
    try:
        tasks = parse_design((root / DESIGN).read_text(), milestone=MILESTONE)[1]
        result = assess_authorities(
            root / DESIGN, staged, root / ACTIVE, raw, milestone=MILESTONE, first_id=8388, count=14
        )
        if tasks_digest(tasks) != result["canonical_tasks_sha256"]:
            raise ValueError("complete_task_digest")
        for row, task in zip(result["contract_rows"], tasks, strict=True):
            if not task.get("prior_failures"):
                row["checks"]["prior"] = True
            row["matched"] = all(row["checks"].values())
            row.update(
                absolute_metric=int(row["matched"]),
                raw_numerator=sum(row["checks"].values()),
                excluded=not row["matched"],
                full_task_sha256=canonical_hash(task),
            )
        result["gate_check_summary"] = [
            g
            for g in result["gate_check_summary"]
            if g["artifact_field"] != "contract_rows.matched"
            or not all(r["matched"] for r in result["contract_rows"])
        ]
        result.update(activated=not result["gate_check_summary"], tasks=tasks)
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        result = dict(
            activated=False,
            tasks=[],
            canonical_tasks_sha256=None,
            contract_rows=[
                dict(
                    unit_id=f"exp{8388 + i}",
                    family=f"exp{8388 + i}",
                    arm="active_contract",
                    seed=None,
                    order=i + 1,
                    status="unstarted",
                    matched=False,
                    checks=dict(independent_design_contract=False),
                    absolute_metric=0,
                    raw_numerator=0,
                    raw_denominator=1,
                    censored=True,
                    excluded=True,
                    effective_independent_groups=0,
                    missing_reason="independent_design_contract_absent",
                )
                for i in range(14)
            ],
            gate_check_summary=[
                failure(
                    root / DESIGN,
                    "complete_current_design",
                    True,
                    str(error) if (root / DESIGN).exists() else None,
                )
            ],
        )
    result["staging_disposition"] = "existing" if staged.exists() else "consumed_by_activation"
    return dict(result)


def historical(raw: Path) -> Json:
    """Replay captured source aliases privately while retaining the original blocked verdict."""
    primary = ROOT / "results/experiment_8374_v722_contract_methods.json"
    value = json.loads(primary.read_bytes())
    refs = value["source_artifact_hashes"]
    for ref in refs:
        require_reference(ref)
        if ref.get("source_path", ref["path"]) != ref["path"]:
            raise ValueError("historical_source_alias")
    with frozen_inputs(refs, raw / "private_history"):
        ready = old.replay(primary)
    return dict(
        ready=ready,
        primary_path=str(primary),
        primary_sha256=sha256_file(primary),
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        current_authority_imported=False,
        missing_design_stays_blocked=True,
    )


def measure(root: Path, raw: Path) -> Json:
    """Reuse original numeric custody, then qualify history outside current authority."""
    progress("current_authority_and_inputs_before")
    with patch.multiple(
        old,
        authority=authority,
        PROTOCOL=PROTOCOL,
        METHODS=METHODS,
        PROTOCOL_PIN=PROTOCOL_PIN,
        METHODS_PIN=METHODS_PIN,
    ):
        work = old.measure(root, raw)
    progress("current_authority_and_inputs_after", 14, 0)
    work["historical_replay"] = historical(raw)
    mismatch = ROOT / "results/raw/experiment_8388_v723_contract_methods/first_replay_mismatch.json"
    work["first_replay_mismatch"] = json.loads(mismatch.read_bytes())
    for name in [
        "results/experiment_8374_v722_contract_methods.json",
        "results/experiment_8387_v722_capstone.json",
        "results/raw/experiment_8388_v723_contract_methods/first_replay_mismatch.json",
        "results/raw/experiment_8388_v723_contract_methods/historical_before/historical_before.stdout",
    ]:
        work["refs"].append(snapshot(ROOT / name, raw / "history", str(len(work["refs"]))))
    path = root / "openspec/change-proposals/research-roadmap-v722-preserved-20261010.md"
    work["failures"].append(
        dict(
            failure(path, "historical_independent_design_contract_available", True, None),
            upstream_id=old.TASK,
        )
    )
    work["input_failures"] = list(work["failures"])
    work["historical_dispositions"]["v722_contract"] = work["historical_replay"]
    capstone = json.loads((ROOT / "results/experiment_8387_v722_capstone.json").read_bytes())
    work["historical_dispositions"]["v722_capstone"] = dict(
        verdict_class=capstone["verdict_class"],
        honest_verdict=capstone["honest_verdict"],
        repository_health=capstone.get("repository_health", []),
    )
    history_root = ROOT / "results/raw/experiment_8388_v723_contract_methods"
    work["historical_reproduction_receipts"] = [
        json.loads(p.read_bytes()) for p in sorted(history_root.glob("historical*/*.receipt.json"))
    ]
    for receipt in work["historical_reproduction_receipts"]:
        for stream in ["stdout", "stderr"]:
            path = ROOT / Path(receipt[stream + "_path"])
            work["refs"].append(snapshot(path, raw / "history", str(len(work["refs"]))))
    return dict(work)


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Historical design absence blocks its scope without clearing qualified current inputs."""
    value = old.build(work, receipts, raw, output)
    value.update(
        experiment_id=8388,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261010",
        honest_verdict="complete_" + value["verdict_class"] + "_v723_contract_methods",
        historical_replay_ready_score=int(
            value["required_checks_passed"] and work["historical_replay"]["ready"]
        ),
        first_replay_mismatch=work["first_replay_mismatch"],
        protocol_path=str(Path(work["root"]) / PROTOCOL),
        protocol_sha256=PROTOCOL_PIN if work["deployment"] else None,
        random_seed=7238388,
        historical_reproduction_receipts=work["historical_reproduction_receipts"],
    )
    value["acceptance_gates"]["historical_replay"] = bool(value["historical_replay_ready_score"])
    value["field_principles"].update(
        {
            k: "Bind current task authority and numerical custody separately from honest historical replay and missing design."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return dict(value)


def replay(path: Path) -> bool:
    """Recompute primitive meaning so altered summaries cannot pass by rehashing themselves."""
    try:
        value = json.loads(path.read_bytes())
        for ref in [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
        ]:
            require_reference(ref)
            if ref.get("source_path", ref["path"]) != ref["path"]:
                return False
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        for field in ["execution_manifest_reference", "owned_coverage_reference"]:
            if work.get(field):
                require_reference(work[field])
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                require_reference(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        with TemporaryDirectory(prefix="exp8388-cold-", dir="/var/tmp") as directory:
            with frozen_inputs(value["source_artifact_hashes"], Path(directory) / "writes"):
                actual = measure(Path(work["root"]), Path(directory))
            for key in [
                "contract",
                "inputs",
                "deployment",
                "methods",
                "support",
                "historical_dispositions",
                "kernel",
                "trajectory",
                "protocol",
                "protocol_sha256",
                "historical_model_provenance",
                "input_failures",
                "historical_replay",
                "first_replay_mismatch",
                "historical_reproduction_receipts",
            ]:
                left, right = actual[key], work[key]
                omitted = (
                    "refs"
                    if key == "inputs"
                    else "authority_snapshots"
                    if key == "contract"
                    else None
                )
                if omitted:
                    left, right = (
                        {k: v for k, v in obj.items() if k != omitted} for obj in [left, right]
                    )
                if left != right:
                    return False
        return bool(
            value["source_artifact_hashes"] == work["refs"]
            and build(
                work,
                value["validation_receipts"],
                Path(value["work_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
