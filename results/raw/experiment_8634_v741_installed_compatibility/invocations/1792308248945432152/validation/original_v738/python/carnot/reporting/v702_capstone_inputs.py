"""REQ-REPORT-8122: read actual operands, preserving every excluded branch.

The activation snapshot supplies the complete executable task list even when
the prose design is malformed. That permits complete accounting, while the
failed design operand remains a block rather than fabricated authority.
"""

import importlib
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from carnot.reporting import v699_capstone as previous
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import tasks_digest

Json = dict[str, Any]
ROOT = previous.ROOT
INPUT = "results/experiment_8110_v702_contract_custody.json"
NAMED = [
    *previous.NAMED,
    "openspec/capabilities/verification/spec.md",
    "ops/verifier_gaps.md",
    "research-program.md",
    "_bmad/prd.md",
    "tests/python/test_primary_publication_7928.py",
    "scripts/experiments/experiment_8109_v701_capstone.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
]
reference = previous.reference
failure = previous.failure


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real completed counts distinguish custody work from a stalled process."""
    print(f"[exp8122] phase={phase} completed={completed} pending={pending}", flush=True)


def authorities(root: Path) -> tuple[list[Json], list[Json], list[Json]]:
    """Check executable bytes without granting a malformed design authority."""
    source = root / INPUT
    value = json.loads(source.read_bytes())
    refs, issues = [reference(source)], []
    snapshots = value["authority_snapshots"]
    for role in ("active", "design", "saved_staged"):
        snap = snapshots[role]
        path = checked(dict(path=snap["snapshot_path"], sha256=snap["sha256"]))
        refs.append(reference(path))
    active = yaml.safe_load(Path(snapshots["active"]["snapshot_path"]).read_bytes())
    tasks = active["tasks"]
    if (
        active["milestone"] != "2026.10.702"
        or tasks_digest(tasks) != value["canonical_tasks_sha256"]
        or [t["id"].split("-")[0] for t in tasks] != [f"exp{n}" for n in range(8110, 8123)]
    ):
        raise ValueError("activation_contract_drift")
    staged = yaml.safe_load(Path(snapshots["saved_staged"]["snapshot_path"]).read_bytes())
    if staged["tasks"] != tasks:
        raise ValueError("staged_contract_drift")
    design = Path(snapshots["design"]["snapshot_path"])
    try:
        table, proposed = parse_design(design.read_text(), milestone="2026.10.702")
        expected_table = [
            dict(order=n + 1, **{k: t[k] for k in ("id", "title", "phase", "deliverable")})
            for n, t in enumerate(tasks)
        ]
        if proposed != tasks or table != expected_table:
            raise ValueError("design_contract_drift")
    except (IndexError, ValueError) as error:
        issues.append(
            failure(design, tasks[0]["id"], "design_exact_task_contract", True, str(error))
        )
    return tasks, refs, issues


def primitive_audit(value: Json, number: int) -> Json:
    """Reopen sealed rows and run equations without training, inference or benchmarks."""
    mapping = {
        8111: (
            "carnot.verify.methods_stream_custody_8111",
            "reduce_rows",
            "primitive_rows.json",
            "rows",
        ),
        8116: (
            "carnot.verify.independent_online_memory_8116",
            "reductions",
            "primitive_rows.json",
            "rows",
        ),
        8118: (
            "carnot.verify.fresh_acquisition_cost_8118",
            "reduce",
            "primitive_rows.json",
            "rows",
        ),
        8119: (
            "carnot.experiment_8119_v702_batched_service_cost",
            "reduce_rows",
            "primitive_rows.json",
            None,
        ),
        8121: ("carnot.reporting.hardware_batch_8121", "reduce", "replay_inputs.json", None),
    }
    if number not in mapping:
        return dict(available=False, reason="No decision primitives in this branch.")
    module, function, name, key = mapping[number]
    ref = next(r for r in value["raw_shard_hashes"] if Path(r["path"]).name == name)
    operand = json.loads(checked(ref).read_bytes())
    result = getattr(importlib.import_module(module), function)(operand[key] if key else operand)
    if number == 8116 and result != value["reductions"]:
        raise ValueError("memory_reduction_drift")
    if number == 8119 and result != value["reduction"]:
        raise ValueError("service_reduction_drift")
    return dict(
        available=True,
        primitive_reference=ref,
        result=result,
        science_qualified=False,
        scope="Descriptive reduction; qualification is separate.",
    )


def load(root: Path, raw: Path) -> Json:
    """Keep failed operands and custody bytes even when their science is unusable."""
    try:
        tasks, refs, failures = authorities(root)
    except (ValueError, KeyError, OSError, TypeError) as error:
        fallback = ROOT / INPUT
        tasks = json.loads(fallback.read_bytes())["task_contract"]
        refs = [reference(root / INPUT), reference(fallback)]
        failures = [
            failure(
                root / INPUT,
                "exp8110-contract-custody",
                "activation_contract",
                "authenticated activation bytes",
                str(error),
            )
        ]
    preconditions = []
    for item in [
        *(root / p for p in NAMED),
        *(ROOT / ".venv/bin" / p for p in ("python", "pytest", "coverage", "ruff", "mypy")),
    ]:
        gate = failure(item, "preconditions", "resource_exists", True, item.is_file())
        gate["passed"] = item.is_file()
        preconditions.append(gate)
        refs.append(reference(item))
        if not gate["passed"]:
            failures.append(gate)
    rows, primaries, audits = [], {}, {}
    for i, task in enumerate(tasks[:-1]):
        progress("before_task_read", i, 12 - i)
        selected, observed_refs, issues = previous.collect(root, [task, tasks[-1]])
        row = selected[0]
        source = root / task["deliverable"]
        value = previous.read(source)
        passing = {
            canonical_hash(
                {k: v for k, v in g.items() if k not in ("passed", "field", "artifact_field")}
            )
            for g in value.get("gate_check_summary", [])
            if g.get("passed") is True
        }
        issues = [
            g
            for g in issues
            if canonical_hash(
                {k: v for k, v in g.items() if k not in ("passed", "field", "artifact_field")}
            )
            not in passing
        ]
        for gate in task["gated_on"]:
            upstream = primaries.get(gate["upstream"], {})
            actual = upstream.get(gate["artifact_field"])
            if actual != gate["value"]:
                operand = root / next(
                    t["deliverable"] for t in tasks if t["id"] == gate["upstream"]
                )
                issues.append(
                    failure(
                        operand, gate["upstream"], gate["artifact_field"], gate["value"], actual
                    )
                )
        progress("before_primitive_reduction", i, 12 - i)
        try:
            audits[task["id"]] = (
                primitive_audit(value, 8110 + i)
                if source.is_file()
                else dict(available=False, reason="Missing primary")
            )
        except (ValueError, KeyError, OSError, TypeError, StopIteration) as error:
            audits[task["id"]] = dict(available=False, reason=str(error))
            issues.append(
                failure(source, task["id"], "primitive_reduction", "valid equations", str(error))
            )
        eligible = (
            source.is_file()
            and row["verdict_class"] not in ("blocked", "disqualified")
            and not issues
        )
        row.update(
            eligible=eligible,
            excluded=not eligible,
            numerator=int(eligible),
            exclusion_reason=None if eligible else "external_prerequisite_or_invalid_measurement",
            gate_check_summary=issues,
        )
        audits[task["id"]]["science_qualified"] = eligible
        # The large failed learner contains diagnostic state, not an eligible audit.
        primaries[task["id"]] = {
            k: v
            for k, v in value.items()
            if k.endswith("_ready_score")
            or k in ("decision_rows", "retention_rows", "honest_verdict", "verdict_class")
        }
        rows.append(row)
        refs.extend(observed_refs)
        failures.extend(issues)
        progress("after_primitive_reduction", i + 1, 11 - i)
    priors = {}
    for task in tasks:
        for prior in task["prior_failures"]:
            eid = prior["experiment_id"].split("-")[0][3:]
            candidates = sorted((root / "results").glob(f"experiment_{eid}_*.json"))
            path = (
                candidates[0]
                if candidates
                else root / "results" / f"missing_{prior['experiment_id']}.json"
            )
            priors[prior["experiment_id"]] = reference(path)
            priors[prior["experiment_id"]]["honest_verdict"] = previous.read(path).get(
                "honest_verdict"
            )
            refs.append(priors[prior["experiment_id"]])
    preserved = []
    for i, ref in enumerate(refs):
        path = Path(ref["path"])
        actual = reference(path)["sha256"]
        saved = dict(ref, sha256=actual, declared_sha256=ref.get("sha256"))
        if actual:
            target = raw / "custody" / (actual.split(":")[-1] + path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(path, target)
            saved.update(source_path=str(path), path=str(target))
        preserved.append(saved)
        if i % 25 == 0:
            progress("input_custody", i + 1, len(refs) - i - 1)
    return dict(
        tasks=tasks,
        dispositions=rows,
        primaries=primaries,
        references=preserved,
        failures=failures,
        preconditions=preconditions,
        independent_reductions=audits,
        prior_evidence=priors,
        exclusion_manifest_reference=reference(root / "ops/exclusion_manifest.yaml"),
    )
