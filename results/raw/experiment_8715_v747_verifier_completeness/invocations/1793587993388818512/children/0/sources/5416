"""REQ-REPORT-8220: current custody preserves history without science credit.

Copied bytes bind later checks to this invocation, rather than to mutable files.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.reporting import v700_contract_custody as base
from carnot.reporting import v710_contract_replay as prior
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8220_v711_current_contract"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v711_current_contract_8220.py"
OWNED = [
    "python/carnot/reporting/v711_current_contract.py",
    "python/carnot/reporting/v711_current_runner.py",
    CLI,
]
DESIGN = prior.DESIGN
STAGED = "research-roadmap-next.yaml"
ACTIVE = "research-roadmap.yaml"
HISTORY = "openspec/change-proposals/research-roadmap-v710-preserved-20261006.md"
UPSTREAM = [
    "results/experiment_8218_v710_contract_replay_qualification.json",
    "results/experiment_8219_v710_utility_patch_methods.json",
]
MILESTONE = "2026.10.711"
TASK = "exp8220-current-contract"
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "python/carnot/reporting/v709_qualification.py",
    DESIGN,
    ACTIVE,
    HISTORY,
    *UPSTREAM,
    "research-complete.yaml",
]


def failure(
    path: Path, field: str, expected: Any, observed: Any, digest: str | None = None
) -> Json:
    """Exact failed operands distinguish missing evidence from measured zero."""
    return dict(prior.failure(path, field, expected, observed, digest), upstream_id=TASK)


def assess(design: Path, staged: Path, active: Path, raw: Path) -> Json:
    """The qualified reader compares independent real authorities without activation."""
    snapshots = {
        role: prior.snapshot(path, raw, role)
        for role, path in [("design", design), ("staged", staged), ("active", active)]
    }
    frozen = [
        Path(snapshots[k].get("snapshot_path", raw / ("absent-" + k)))
        for k in ["design", "staged", "active"]
    ]
    try:
        value = base.authority.assess_authorities(
            frozen[0],
            frozen[1],
            frozen[2],
            raw / "assessment",
            milestone=MILESTONE,
            first_id=8220,
            count=14,
        )
        _, tasks = prior.parse_design(frozen[0].read_text(), milestone=MILESTONE)
        value["tasks"] = tasks
        digest = base.authority.tasks_digest(tasks)
        if digest != value["canonical_tasks_sha256"]:
            value["gate_check_summary"].append(
                dict(
                    artifact_path=str(frozen[0]),
                    artifact_hash=snapshots["design"]["sha256"],
                    artifact_field="design_tasks_sha256",
                    expected=value["canonical_tasks_sha256"],
                    observed=digest,
                )
            )
        value["gate_check_summary"] = [
            dict(
                failure(
                    Path(
                        next(
                            (
                                snapshots[k]["path"]
                                for i, k in enumerate(["design", "staged", "active"])
                                if str(frozen[i]) == g["artifact_path"]
                            ),
                            str(design),
                        )
                    ),
                    g["artifact_field"],
                    g["expected"],
                    g["observed"],
                    g["artifact_hash"],
                ),
                field=g["artifact_field"],
            )
            for g in value["gate_check_summary"]
        ]
        value["activated"] = not value["gate_check_summary"]
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        value = dict(
            activated=False,
            planning_matched=False,
            contract_rows=[],
            tasks=[],
            canonical_tasks_sha256=None,
            gate_check_summary=[
                dict(
                    failure(
                        design,
                        "authority_readable",
                        True,
                        str(error),
                        snapshots["design"]["sha256"],
                    ),
                    field="authority_readable",
                )
            ],
        )
    return dict(value, authority_snapshots=snapshots)


def mutations(design: Path, active: Path, raw: Path) -> list[Json]:
    """Private mutations demonstrate that each executable operand can fail."""
    original = yaml.safe_load(active.read_bytes())
    rows = []
    for name in ("count", "order", "prompt", "title", "gate", "model", "digest"):
        changed, text = deepcopy(original), design.read_text()
        parent = raw / name
        parent.mkdir(parents=True, exist_ok=True)
        if name == "count":
            changed["tasks"].pop()
        elif name == "order":
            changed["tasks"].reverse()
        elif name in ("prompt", "title"):
            changed["tasks"][0][name] += " changed"
        elif name == "gate":
            next(t for t in changed["tasks"] if t["gated_on"])["gated_on"][0]["value"] = 999
        elif name == "model":
            changed["tasks"][0]["MODEL_SPECS"] = [{"hf_id": "changed"}]
        else:
            text = re.sub(
                r"(Canonical full-task SHA-256: `)[0-9a-f]{64}", r"\g<1>" + "0" * 64, text
            )
        d, a = parent / "design.md", parent / "active.yaml"
        d.write_text(text)
        a.write_text(yaml.safe_dump(changed))
        result = assess(d, parent / "absent", a, parent / "snapshots")
        rows.append(
            dict(
                control=name,
                rejected=not result["activated"],
                gate_check_summary=result["gate_check_summary"],
            )
        )
    return rows


def receipt_controls(raw: Path) -> list[Json]:
    """Use the unchanged byte-bound reader on valid, altered and unavailable cases."""
    primary = raw / (NAME + ".json")
    value = dict(
        experiment_id=8220,
        task_id=TASK,
        honest_verdict="complete_null_private_control",
        verdict_class="null",
    )
    validate_primary(value, primary)
    atomic_json(primary, value)
    side = raw / "raw" / NAME / "receipt.json"
    atomic_json(side, dict(primary_path=str(primary), primary_sha256=sha256_file(primary)))
    rows = [
        dict(
            control="valid",
            passed=read_bound_sidecar(primary, side)["primary_sha256"] == sha256_file(primary),
        )
    ]
    atomic_json(side, dict(primary_sha256="sha256:" + "0" * 64))
    for name, path in [("tampered", side), ("externally_blocked", side.with_name("absent"))]:
        try:
            read_bound_sidecar(primary, path)
            rejected = False
        except (ValueError, OSError):
            rejected = True
        rows.append(dict(control=name, passed=rejected, operand=str(path)))
    return rows


def historical(refs: list[Json]) -> tuple[list[Json], list[Json]]:
    """Keep imported dispositions and document excerpts separate from executed tasks."""
    by_name = {Path(r["path"]).name: r for r in refs if r["exists"]}
    outcomes = []
    for name in UPSTREAM:
        ref = by_name.get(Path(name).name)
        if ref:
            value = json.loads(Path(ref["snapshot_path"]).read_bytes())
            keys = [
                "experiment_id",
                "task_id",
                "honest_verdict",
                "verdict_class",
                "gate_check_summary",
                "utility_protocol_ready_score",
                "H1",
                "H2",
                "intended_count",
                "completed_count",
                "failed_count",
                "excluded_count",
            ]
            outcomes.append(
                dict(
                    {k: value[k] for k in keys if k in value},
                    original_primary=ref,
                    execution_status="scheduled_outcome",
                )
            )
    entries = []
    ref = by_name.get(Path(HISTORY).name)
    if ref:
        text = Path(ref["snapshot_path"]).read_text()
        for identity in range(8220, 8232):
            match = re.search(rf"\*\*Exp{identity}\*\*.*?(?=\n\n)", text, re.S)
            if match:
                entries.append(
                    dict(
                        experiment_id=identity,
                        milestone="2026.10.710",
                        execution_status="unexecuted_design_entry",
                        observed_outcome=None,
                        design_excerpt=match.group(),
                        source_reference=ref,
                    )
                )
    return outcomes, entries


def measure(root: Path, raw: Path) -> Json:
    """Authenticate inputs before any validation; unavailable originals stay historical."""
    refs, failures, checked = [], [], []
    for index, name in enumerate(INPUTS):
        path = root / name
        ref = prior.snapshot(path, raw / "inputs", f"input{index}")
        refs.append(ref)
        row = dict(
            failure(path, "exists", True, True if ref["exists"] else None, ref["sha256"]),
            passed=ref["exists"],
        )
        checked.append(row)
        if not ref["exists"]:
            failures.append(row)
        elif name in UPSTREAM:
            try:
                value = json.loads(Path(ref["snapshot_path"]).read_bytes())
                expected_id = 8218 + UPSTREAM.index(name)
                good = (
                    isinstance(value, dict)
                    and value.get("experiment_id") == expected_id
                    and isinstance(value.get("honest_verdict"), str)
                    and value.get("verdict_class")
                    in {"disqualified", "null", "blocked", "positive", "circular_positive"}
                    and (
                        expected_id == 8218
                        or all(
                            isinstance(value.get(k), dict)
                            and value[k].get("measured_here") is False
                            for k in ["H1", "H2"]
                        )
                    )
                )
            except (ValueError, TypeError):
                good = False
            ref["schema_valid"] = good
            if not good:
                failures.append(failure(path, "historical_schema", True, False, ref["sha256"]))
        print(
            f"[exp8220] phase=input completed={index + 1} pending={len(INPUTS) - index - 1}",
            flush=True,
        )
    code = [
        prior.snapshot(ROOT / p, raw / "current_code", f"code{i}")
        for i, p in enumerate([*OWNED, TEST, *prior.REUSED])
    ]
    contract = assess(root / DESIGN, root / STAGED, root / ACTIVE, raw / "authority")
    outcomes, entries = historical([r for r in refs if r.get("schema_valid", True)])
    if any(Path(r["path"]).name == Path(HISTORY).name and r["exists"] for r in refs):
        if len(entries) != 12:
            failures.append(failure(root / HISTORY, "unexecuted_design_count", 12, len(entries)))
    controls = []
    if contract["activated"]:
        controls = mutations(root / DESIGN, root / ACTIVE, raw / "mutation_controls")
    readers = receipt_controls(raw / "receipt_controls")
    return dict(
        contract=contract,
        failures=failures,
        historical_dispositions=outcomes,
        unexecuted_v710_design_entries=entries,
        replay_controls=controls,
        receipt_controls=readers,
        immutable_code_snapshots=code,
        h1_custody={},
        preconditions_checked=checked,
        source_artifact_hashes=refs,
    )


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Qualified failure precedence prevents readiness from becoming benefit credit."""
    value = prior.reduce(work, receipts)
    if value["verdict_class"] == "blocked":
        operand = value["gate_check_summary"][0]
        suffix = Path(operand["path"]).name + "_" + str(operand["artifact_field"])
        value["honest_verdict"] = "complete_blocked_" + re.sub("[^a-z0-9_]", "_", suffix.lower())
    elif value["verdict_class"] == "circular_positive":
        value["honest_verdict"] = "complete_circular_positive_current_contract"
    for key in [
        "historical_replay_ready_score",
        "contract_ready_score",
        "h1_custody_qualification",
        "immutable_code_snapshots",
    ]:
        value.pop(key)
    rows = []
    source = work["contract"]["contract_rows"]
    for index in range(14):
        primitive = source[index] if index < len(source) else {}
        checks = primitive.get("checks", {})
        missing = not bool(primitive) or primitive.get("status") == "unstarted"
        rows.append(
            dict(
                unit_id=f"exp{8220 + index}",
                source_id="V711_authority",
                source_cluster_id="V711_authority",
                arm="current_contract",
                condition="complete_task_agreement",
                seed=None,
                status="censored" if missing else "completed",
                completed=not missing,
                missing_status=missing,
                checks=checks,
                metric="contract_agreement",
                absolute_metric=None if missing else int(all(checks.values())),
                numerator=None if missing else sum(checks.values()),
                denominator=len(checks),
                raw_numerator=None if missing else sum(checks.values()),
                raw_denominator=len(checks),
                failed=not missing and not all(checks.values()),
                censored=missing,
                excluded=False,
                effective_independent_groups=0,
            )
        )
    controls_ok = all(r["rejected"] for r in work["replay_controls"]) and all(
        r["passed"] for r in work["receipt_controls"]
    )
    if not controls_ok:
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_controls",
            required_checks_passed=False,
        )
        value["gate_check_summary"].append(failure(ROOT / CLI, "current_controls", True, False))
    value.update(
        experiment_id=8220,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261007",
        schema="carnot.v711.current-contract.v1",
        title="Current contract custody",
        rows=rows,
        intended_count=14,
        completed_count=sum(r["completed"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        current_contract_ready_score=int(
            value["required_checks_passed"]
            and controls_ok
            and not work["failures"]
            and work["contract"]["activated"]
        ),
        current_code_snapshots=work["immutable_code_snapshots"],
        receipt_controls=work["receipt_controls"],
        unexecuted_v710_design_entries=work["unexecuted_v710_design_entries"],
        cited_upstream_artifacts=work["source_artifact_hashes"],
        methodology_note="Fourteen actual current task authorities are compared from saved bytes. "
        "Two historical outcomes and twelve unexecuted design entries remain distinct. "
        "Current authority readiness cannot recover unavailable original code or establish "
        "independent generalization, utility improvement or learning benefit.",
    )
    value["acceptance_gates"]["current_controls"] = dict(passed=controls_ok)
    return dict(value)
