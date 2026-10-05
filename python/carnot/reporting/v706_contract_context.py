"""REQ-REPORT-8164: preserve method scope without granting new scientific credit.

A prior null, expired schedule and failed validator need different next actions.
The ledger therefore keeps the original declarations and the actual source bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from carnot.reporting.v698_fixture_consumer_contract import failure

Json = dict[str, Any]
METHODS = [
    ("RT4CHART and evidence alignment", [8166, 8167, 8168, 8169, 8170]),
    ("GASP source-removal diagnostic", [8167]),
    ("Delayed feedback and finite capacity", [8165, 8171, 8172]),
    ("KAC and KAN forgetting", [8171, 8172]),
    ("EBT and ARM-EBM with logistic equivalence", [8168, 8170]),
    ("Extropic execution versus improvement", [8164, 8177]),
    ("FPGA-Ising decomposition and Z1T", [8176]),
    ("Neural constraint satisfaction and visual decoding deferred", []),
]


def hash_refs(value: Json) -> list[Json]:
    """Normalize producer schemas; a null hash preserves an explicitly absent input."""
    refs = []
    for name in ["raw_shard_hashes", "source_artifact_hashes"]:
        group = value.get(name, [])
        group = (
            [dict(path=p, sha256=h) for p, h in group.items()] if isinstance(group, dict) else group
        )
        refs.extend(r for r in group if r.get("sha256") and r.get("exists", True))
    return refs


def bind_logs(binder: Any, value: Json) -> None:
    """A changed diagnostic blocks its operand but must not erase later evidence."""
    try:
        binder.logs(value)
    except ValueError:
        row = binder.failures[-1]
        path = Path(row["path"])
        if path.is_file():
            binder.bind(path)


def validate_producers(tasks: list[Json], path: Path, binder: Any) -> None:
    """Gate names must appear in the producing task's own declared artifact fields."""
    producers = {t["id"]: t for t in tasks}
    for task in tasks:
        for gate in task.get("gated_on", []):
            prompt = producers[gate["upstream"]]["prompt"]
            fields = prompt.split("REQUIRED ARTIFACT FIELDS:")[-1].split("Run command:")[0]
            name = gate["artifact_field"]
            try:
                binder.require(path, "producer_field_declared", True, name in fields)
            except ValueError:
                binder.failures[-1]["observed"] = name


def scope_ledger(root: Path, tasks: list[Json], history: Json, binder: Any) -> list[Json]:
    """Derive paths from immutable authorities, including older schedule failures."""
    sources = {r["task_id"]: r for r in history["historical_dispositions"]}
    rows = []
    for task in tasks:
        for prior in task.get("prior_failures", []):
            identity = prior["experiment_id"]
            source = sources.get(identity)
            if source is None:
                admin = root / "results/experiment_8136_v704_contract_custody.json"
                try:
                    old = binder.read(admin)["authority_snapshots"]["active"]
                    active = binder.bind(Path(old["snapshot_path"]), old["sha256"])
                    old_tasks = yaml.safe_load(Path(active["snapshot_path"]).read_bytes())["tasks"]
                    producer = next(t for t in old_tasks if t["id"] == identity)
                    path = root / producer["deliverable"]
                    value = binder.read(path)
                    bind_logs(binder, value)
                    source = dict(value, path=str(path), sha256=binder.bind(path)["sha256"])
                except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
                    binder.failures.append(
                        failure(admin, "prior_authority_readable", identity, str(error))
                    )
                    source = dict(path=None, honest_verdict=None, verdict_class=None)
            failed = source.get(
                "failed_receipts",
                [r for r in source.get("validation_receipts", []) if not r["passed"]],
            )
            cause = (
                "conductor_gate_skip"
                if source.get("disposition") == "gate_skipped"
                else "owned_validation_failure"
                if source.get("verdict_class") == "disqualified"
                else "expired_learning_schedule"
                if identity.startswith("exp8143-") or identity.startswith("exp8144-")
                else "external_gate_block"
                if source.get("verdict_class") == "blocked"
                else "missing_primary"
                if source.get("honest_verdict") is None
                else "terminal_null"
                if source.get("verdict_class") == "null"
                else "completed_measurement"
            )
            rows.append(
                dict(
                    consumer=task["id"],
                    prior_failure=prior,
                    path=source.get("path"),
                    sha256=source.get("sha256"),
                    honest_verdict=source.get("honest_verdict"),
                    verdict_class=source.get("verdict_class"),
                    cause_class=cause,
                    failed_receipts=failed,
                    scope_change=prior["addressed_by"],
                    retry_unchanged=False,
                )
            )
    return rows


def literature(root: Path, binder: Any, contract: Json) -> Json:
    """Keep the planning entry verbatim; literature findings remain imported context."""
    path = root / "research-references.md"
    try:
        ref = binder.bind(path)
        text = Path(ref["snapshot_path"]).read_text()
        entry = text.split("## 2026-10-05 — V706 planning scan:", 1)[1].split("\n## ", 1)[0]
        design = Path(contract["authority_snapshots"]["design"]["snapshot_path"]).read_text()
        return dict(
            source=ref,
            entry=entry,
            method_to_tasks=[dict(method=m, tasks=t) for m, t in METHODS],
            frozen_scientific_protocol=design.split("## Phase 2", 1)[-1].split(
                "## Structured dependency graph", 1
            )[0],
            hypothesis_family=dict(
                hypotheses=["H1", "H2"],
                family_alpha=0.05,
                per_hypothesis_alpha=0.025,
                independent_unit="source_cluster",
                generalization_credit=0,
            ),
            board_obligations=["KV260", "PolarFire", "GateMate"],
            capstone_always_runs=not contract["tasks"][-1].get("gated_on"),
        )
    except (OSError, ValueError, KeyError, IndexError) as error:
        binder.failures.append(failure(path, "V706_reference_entry_readable", True, str(error)))
        return {}
