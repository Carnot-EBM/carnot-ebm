"""REQ-REPORT-8217: preserve exact authority and every producer disposition.

Typed readers qualify current evidence without changing old primary bytes.
An unavailable sibling cannot erase another branch's authenticated evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import v709_qualification as q
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v708_capstone_science import operand

Json = dict[str, Any]
ROOT = q.ROOT
INPUT = "results/experiment_8205_v709_contract_consumer_qualification.json"
TASK = "exp8217-capstone"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed real counts keep long reductions visible without invented activity."""
    print(f"[exp8217] phase={phase} completed={completed} pending={pending}", flush=True)


def terminal(binder: Any, path: Path, value: Json) -> bool:
    """Reuse the qualified reader and the unchanged hash-specific publisher schema."""
    if value.get("terminal_validation_sidecar_path"):
        result = q.terminal(path, value, binder.raw / "terminal")
        for ref in result["references"]:
            binder.bind(Path(ref["path"]), ref["sha256"])
        return bool(result["passed"])
    side = path.parent / "raw" / path.stem / "validators" / (sha256_file(path)[7:] + ".json")
    report = read_bound_sidecar(path, side)
    binder.bind(side)
    binder.logs(report)
    return bool(report["report"]["passed"])


def load(root: Path, raw: Path) -> Json:
    """Use frozen activation first; separately record a failed original qualification."""
    binder = q.base.Binder(raw / "custody", task=TASK)
    path = root / INPUT
    original: list[Json] = []
    receipt: Json = {}
    try:
        receipt = binder.read(path)
        for field in (
            "required_checks_passed",
            "consumer_reader_ready_score",
            "contract_ready_score",
        ):
            binder.require(
                path, field, True if field == "required_checks_passed" else 1, receipt.get(field)
            )
        binder.require(path, "terminal.report.passed", True, terminal(binder, path, receipt))
        snapshots = receipt["authority_snapshots"]
        frozen = []
        for role in ("design", "active"):
            ref = snapshots[role]
            binder.bind(Path(ref["snapshot_path"]), ref["sha256"])
            frozen.append(Path(ref["snapshot_path"]))
        authority = q.assess(frozen[0], raw / "absent-staged", frozen[1], raw / "authority")
        binder.require(
            path,
            "canonical_tasks_sha256",
            authority["canonical_tasks_sha256"],
            receipt["canonical_tasks_sha256"],
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        original = [
            operand(
                str(path),
                "qualified_frozen_activation",
                True,
                str(error),
                digest=sha256_file(path) if path.is_file() else None,
            )
        ]
        authority = q.assess(
            root / q.DESIGN,
            root / "research-roadmap-next.yaml",
            root / "research-roadmap.yaml",
            raw / "fallback_authority",
        )
    tasks = authority["tasks"]
    if not tasks:
        # A missing authority still leaves thirteen explicit unmeasured slots.
        _, tasks = q.parse_design((ROOT / q.DESIGN).read_text(), milestone=q.MILESTONE)
    failures = [
        dict(
            g,
            artifact_field=g.get("artifact_field", g.get("check")),
            path=g.get("path", g.get("artifact_path")),
            hash=g.get("hash", g.get("artifact_hash")),
            op=g.get("op", "=="),
        )
        for g in authority["gate_check_summary"]
    ]
    refs_before = len(binder.refs)
    manifest = root / "ops/exclusion_manifest.yaml"
    log_path = root / "ops/conductor-log.md"
    retired = yaml.safe_load(manifest.read_bytes()) if manifest.is_file() else {}
    log = log_path.read_text() if log_path.is_file() else ""
    for observed in (manifest, log_path):
        if observed.is_file():
            binder.bind(observed)
    primaries: Json = {}
    dispositions = []
    progress("task_inventory_before", 0, 12)
    for index, task in enumerate(tasks[:-1]):
        number, path = 8205 + index, root / task["deliverable"]
        gates: list[Json] = []
        skips = []
        value: Json = {}
        for candidate in sorted((root / "results").glob(f"experiment_{number}_*.json")):
            try:
                observed = binder.read(candidate)
                if observed.get("schema") == "blocked_gate_check_v1":
                    skips.append(
                        dict(path=str(candidate), sha256=sha256_file(candidate), artifact=observed)
                    )
                elif candidate == path:
                    value = observed
            except (OSError, ValueError, KeyError, TypeError) as error:
                gates.append(operand(str(candidate), "readable_primary", True, str(error)))
        scope_matches = [
            r
            for key in ("retired", "retired_experiments")
            for r in retired.get(key, [])
            if str(r.get("experiment_id")) in (str(number), task["id"])
            or task["id"] in r.get("experiment_ids", [])
        ]
        qualified = False
        receipt_rows: list[Json] = []
        if value:
            try:
                for field, expected in [
                    ("task_id", task["id"]),
                    ("required_checks_passed", True),
                    ("flagged_adversarial", False),
                ]:
                    binder.require(path, field, expected, value.get(field))
                binder.require(path, "terminal.report.passed", True, terminal(binder, path, value))
                for ref in q.hash_references(value.get("code_config_hashes", [])):
                    named = Path(ref["path"])
                    binder.bind(named if named.is_absolute() else ROOT / named, ref["sha256"])
                receipt_rows = q.receipts(value.get("validation_receipts", []))
                for child_receipt in receipt_rows:
                    if child_receipt.get("scope") == "repository_health":
                        continue
                    binder.require(
                        path,
                        "validation_receipt." + child_receipt["name"],
                        True,
                        child_receipt["passed"],
                    )
                    for stream in ("stdout", "stderr"):
                        if stream + "_path" in child_receipt:
                            binder.bind(
                                Path(child_receipt[stream + "_path"]),
                                child_receipt[stream + "_sha256"],
                            )
                qualified = (
                    value.get("verdict_class") not in ("partial", "disqualified")
                    and not scope_matches
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                gates.append(
                    operand(
                        str(path), "producer_validation", True, str(error), digest=sha256_file(path)
                    )
                )
        for gate in task["gated_on"]:
            parent = next(t for t in tasks if t["id"] == gate["upstream"])
            source = root / parent["deliverable"]
            check = operand(
                str(source),
                gate["artifact_field"],
                gate["value"],
                primaries.get(parent["id"], {}).get(gate["artifact_field"]),
                gate["op"],
                sha256_file(source) if source.is_file() else None,
            )
            check["upstream_id"] = parent["id"]
            gates.append(check)
        for skip in skips:
            v = skip["artifact"]
            gates.append(
                dict(
                    upstream_id=v["failed_upstream"],
                    path=v["failed_evidence_path"],
                    hash=v["failed_evidence_sha256"],
                    artifact_field=v["failed_field"],
                    op=v["failed_operator"],
                    expected=v["failed_expected"],
                    observed=v["failed_observed"],
                    passed=False,
                )
            )
        state = value.get("verdict_class", "blocked")
        if value and not qualified:
            state = "disqualified"
        eligible = (
            qualified
            and state not in ("blocked", "disqualified")
            and all(g["passed"] for g in gates)
        )
        if state == "blocked" and isinstance(value.get("gate_check_summary"), list):
            gates.extend(g for g in value["gate_check_summary"] if not g.get("passed", False))
        dispositions.append(
            dict(
                experiment_id=number,
                task_id=task["id"],
                unit_id=task["id"],
                source_cluster_id=task["id"],
                arm="task_disposition",
                condition="terminal_accounting",
                metric="eligible_branch",
                numerator=int(eligible),
                denominator=1,
                completed=True,
                status="completed",
                failed=state == "disqualified",
                excluded=not eligible,
                censored=False,
                eligible=eligible,
                qualified=qualified,
                primary_present=bool(value),
                path=str(path),
                sha256=sha256_file(path) if path.is_file() else None,
                primary_honest_verdict=value.get("honest_verdict"),
                honest_verdict=value.get("honest_verdict", "complete_blocked_missing_primary"),
                verdict_class=state,
                disposition="disqualified"
                if state == "disqualified"
                else "producer"
                if value
                else "pre_gate_skip"
                if skips
                else "missing",
                conductor_skips=skips,
                conductor_log_rows=[
                    line
                    for line in log.splitlines()
                    if task["id"] in line or task["title"][:48] in line
                ],
                retired_scope_matches=scope_matches,
                repository_health_failures=[
                    r
                    for r in receipt_rows
                    if r.get("scope") == "repository_health" and not r["passed"]
                ],
                gate_check_summary=gates,
            )
        )
        if not value and not skips:
            gates.append(operand(str(path), "primary_exists", True, False))
        failures.extend(g for g in gates if not g["passed"])
        primaries[task["id"]] = value
        progress("task_inventory", index + 1, 11 - index)
    priors = []
    for task in tasks:
        for prior in task["prior_failures"]:
            number = prior["experiment_id"].split("-")[0].removeprefix("exp")
            candidates = sorted((root / "results").glob(f"experiment_{number}_*.json"))
            evidence = []
            for candidate in candidates:
                old = binder.read(candidate)
                if (
                    old.get("task_id") == prior["experiment_id"]
                    or old.get("schema") == "blocked_gate_check_v1"
                ):
                    evidence.append(
                        dict(
                            path=str(candidate),
                            sha256=sha256_file(candidate),
                            honest_verdict=old.get("honest_verdict"),
                            pre_gate=old.get("schema") == "blocked_gate_check_v1",
                            configuration_scope={
                                k: old[k]
                                for k in (
                                    "configuration_scope_sha256",
                                    "numerical_protocol_sha256",
                                    "protocol_sha256",
                                    "config",
                                    "claim_scope",
                                )
                                if k in old
                            },
                            artifact=old if old.get("schema") == "blocked_gate_check_v1" else None,
                        )
                    )
            archive = [
                r["original_archive_row"]
                for r in receipt.get("task_dispositions", [])
                if r["task_id"] == prior["experiment_id"]
                and r.get("conductor_result") == "GATE_BLOCKED"
            ]
            priors.append(
                dict(
                    prior,
                    task_id=task["id"],
                    evidence=evidence,
                    prior_verdict_matches_artifact=any(
                        r["honest_verdict"] == prior["verdict"] for r in evidence
                    ),
                    conductor_pre_gate_records=archive,
                    prior_verdict_matches_conductor_pre_gate=bool(
                        archive and prior["verdict"] == "blocked_gate_check_failed"
                    ),
                )
            )
    return dict(
        root=str(root),
        authority=authority,
        tasks=tasks,
        primaries=primaries,
        dispositions=dispositions,
        failures=failures + binder.failures,
        references=binder.refs,
        prior_evidence=priors,
        original_contract_failures=original,
        preconditions=binder.observations,
        frozen_input_count=refs_before,
    )
