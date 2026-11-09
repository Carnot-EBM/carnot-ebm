"""REQ-REPORT-8930: exact task custody is administrative readiness, never detector benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v755_branch_contract as base
from carnot.reporting import v763_branch_contract as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v749_contract_methods import encode as encode
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.audit_roadmap_gates import _required_artifact_fields_block
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = prior.ROOT, prior.DESIGN, prior.ACTIVE, prior.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8930_v764_branch_contract",
    "exp8930-branch-contract",
    "2026.11.764",
)
PIN = "sha256:ac24ff28a5c49c06dbf0ce332c5836c0eae1ce6b79df28a000d01175577bac4c"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v764_branch_contract.py", CLI]
TEST = "tests/python/test_v764_branch_contract_8930.py"
HISTORY = "results/experiment_8929_v763_capstone.json"
LITERATURE = "research-references.md"
PINS = {
    HISTORY: "sha256:0d1df5182cf572b14e9b8b61ce74748b66a0ca0afd03a670f1bfe9bd96f90826",
    "openspec/change-proposals/research-roadmap-v763-preserved-20261108.md": "sha256:7be492e5706b10e6c7b848ce94a23642e02e4b3be718a5af4c22f594c42fede5",
    "openspec/change-proposals/research-roadmap-v763-queue-preserved-20261108.yaml": "sha256:a48ddf4b2d7830f19813110d54e39bbf6ebb4f0bd267dd81981ea99467845ac3",
    "results/experiment_8923_label_demand_evaluation.json": "sha256:7a86cdde9ed000881c9ee292f3223e2c529fb8177f8ffbfcd013ca581574b99b",
    "results/experiment_8927_inline_request_cost.json": "sha256:353ea1de2edc7f3a85e6ceebdc7598434d0ae4cfea9703c00307b6cfddac1399",
}
reference, checksum, seal = prior.reference, prior.checksum, prior.seal
METHOD_ROWS = [
    dict(
        method="ImplicitHop",
        source="https://arxiv.org/abs/2610.41169",
        tasks=[8931, 8932, 8934, 8935, 8936, 8938],
        limits="Depth controls and seeded final-answer truth; no copied prefix labels or localization.",
    ),
    dict(
        method="Type-aware hallucination analysis",
        source="https://arxiv.org/abs/2611.02258",
        tasks=[8932, 8937, 8939],
        limits="Separate semantic and format errors with length controls; no financial-domain score transfer.",
    ),
    dict(
        method="Transfer-Selective Replay",
        source="https://arxiv.org/abs/2607.15587",
        tasks=[8933, 8940, 8941],
        limits="Released signature-compatible feedback, equal gradients and labels; local variant, not full TSR.",
    ),
    dict(
        method="Distributional EBMs",
        source="https://arxiv.org/abs/2605.18871",
        tasks=[8937, 8938, 8939],
        limits="Oracle-separated learned scoring and calibration controls; no new adapter ensemble.",
    ),
    dict(
        method="FPGA–ASIC co-design",
        source="https://arxiv.org/abs/2602.15985",
        tasks=[8942],
        limits="Charge complete host, memory and persistence work; unknown fractions stay unknown.",
    ),
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so supervision can keep the original wall-clock budget."""
    print(f"[exp8930] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, check: str, field: str, expected: Any, observed: str) -> Json:
    """Keep absent bytes separate from rejected producers and measured zeros."""
    return dict(
        check=check,
        upstream=TASK,
        artifact_field=field,
        expected=expected,
        observed=observed,
        **reference(path),
    )


def authority(root: Path) -> Json:
    """Bind complete independent objects; accept activation after staging is consumed."""
    tasks, table, rows, failures = [], [], [], []
    mode = "unavailable"
    path = root / DESIGN
    try:
        text = path.read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.load((root / ACTIVE).read_bytes(), Loader=yaml.CSafeLoader)
        staged = (
            yaml.load((root / STAGED).read_bytes(), Loader=yaml.CSafeLoader)
            if (root / STAGED).is_file()
            else None
        )
        selected = active if active["milestone"] == MILESTONE else staged
        if selected is None or selected["milestone"] != MILESTONE:
            raise ValueError("no_current_authority")
        mode = "activated" if active["milestone"] == MILESTONE else "staged"
        queues = [deepcopy(staged or selected), deepcopy(selected)]
        for queue in queues:
            Roadmap.model_validate(queue)
            for actual, expected in zip(queue["tasks"], tasks):
                if actual == dict(expected, prompt=encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
        checked = compare_contract(text, *queues, milestone=MILESTONE, first_id=8930, count=14)
        rows = checked["rows"]
        for i, row in enumerate(rows):
            row["checks"]["full_object"] = (
                i < min(len(tasks), *(len(q["tasks"]) for q in queues))
                and tasks[i] == queues[0]["tasks"][i] == queues[1]["tasks"][i]
            )
            row["matched"] = all(row["checks"].values())
        if not (
            checked["passed"]
            and all(r["matched"] for r in rows)
            and canonical_hash(tasks) == PIN
            and "<!-- V764_TASK_CONTRACT_END -->" in text
        ):
            raise ValueError("independent_authority_mismatch")
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, yaml.YAMLError):
        failures.append(
            failure(
                path,
                "complete_authority",
                "full_task_objects",
                "14 exact ordered full objects, table, digest and queue",
                "missing_file" if not path.is_file() else "rejected_producer",
            )
        )
    for i in range(14):
        if i >= len(rows):
            rows.append(
                dict(
                    unit_id=f"exp{8930 + i}-missing",
                    order=i + 1,
                    checks=dict(independent_authority=False),
                    matched=False,
                )
            )
        row = rows[i]
        matched = row["matched"] and not failures
        row.update(
            matched=matched,
            value=int(matched),
            numerator=int(matched),
            absolute_metric=int(matched),
            denominator=1,
            cluster_id="administrative_contract",
            effective_independent_groups=0,
            censored=not matched,
            excluded=False,
            seed=None,
            status="completed" if matched else "censored",
            missing_reason=None if matched else "authority_unavailable_or_rejected",
        )
    return dict(
        ready=not failures,
        rows=rows,
        tasks=tasks,
        table=table,
        failures=failures,
        comparison_mode=mode,
    )


def mutations(source: Path, raw: Path) -> list[Json]:
    """Each control changes one valid private queue, preserving causal rejection."""
    text = (source / DESIGN).read_text()
    queue = yaml.load(
        (
            source / (STAGED if authority(source)["comparison_mode"] == "staged" else ACTIVE)
        ).read_bytes(),
        Loader=yaml.CSafeLoader,
    )
    rows = []
    for mode in ["missing", "order", "prompt", "gate", "path"]:
        root, changed = raw / mode, deepcopy(queue)
        if mode == "missing":
            changed["tasks"].pop()
        if mode == "order":
            changed["tasks"].reverse()
        if mode == "prompt":
            changed["tasks"][0]["prompt"] += "changed"
        if mode == "gate":
            next(t for t in changed["tasks"] if t["gated_on"])["gated_on"][0]["artifact_field"] = (
                "forged"
            )
        if mode == "path":
            changed["tasks"][0]["deliverable"] = "results/forged.json"
        path = root / DESIGN
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        (root / ACTIVE).write_text(yaml.safe_dump(changed))
        rows.append(
            dict(
                mutation=mode,
                passed=not authority(root)["ready"],
                references=[reference(path), reference(root / ACTIVE)],
            )
        )
        progress("control_" + mode, len(rows), 5 - len(rows))
    return rows


def history(raw: Path) -> Json:
    """Preserve pinned producer bytes and actual skips without re-executing their science."""
    refs, failures, outcomes = [], [], []
    for name, pin in PINS.items():
        path = ROOT / name
        if reference(path)["sha256"] != pin:
            failures.append(
                failure(
                    path,
                    "original_v763",
                    "sha256",
                    pin,
                    "missing_file" if not path.is_file() else "rejected_producer",
                )
            )
            continue
        refs.append(seal(path, raw))
        if name == HISTORY:
            try:
                outcomes = json.loads(path.read_bytes())["outcome_rows"]
            except (ValueError, KeyError, TypeError):
                failures.append(
                    failure(
                        path,
                        "original_v763_schema",
                        "outcome_rows",
                        "original producer object",
                        "rejected_producer",
                    )
                )
    for index, row in enumerate(outcomes):
        ref = row.get("primary_reference")
        if ref:
            path = Path(ref["path"])
            if reference(path)["sha256"] != ref["sha256"]:
                failures.append(
                    failure(
                        path,
                        "original_retained_primary",
                        "sha256",
                        ref["sha256"],
                        "missing_file" if not path.is_file() else "rejected_producer",
                    )
                )
            else:
                refs.append(seal(path, raw))
        progress("historical_outcomes", index + 1, len(outcomes) - index - 1)
    return dict(
        authenticated=not failures,
        references=refs,
        failures=failures,
        outcome_rows=outcomes,
        scientific_results_admitted=False,
        compute_scope="Opaque original producer references; no current compute credit.",
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze each invocation before validation without model or device setup."""
    progress("preconditions_before", 0, 14)
    if (raw / "work.json").exists():
        raise FileExistsError("immutable_invocation_already_exists")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    historical = history(raw)
    snapshots, failures = list(historical["references"]), []
    for name in [DESIGN, ACTIVE, STAGED, LITERATURE]:
        path = root / name
        if path.is_file():
            snapshots.append(dict(seal(path, raw), current_execution_authority=name != LITERATURE))
            target = raw / "authority" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        elif name != STAGED:
            failures.append(
                failure(path, "authority_input", name, "original bytes", "missing_file")
            )
    for name in [
        *OWNED,
        TEST,
        "python/carnot/reporting/roadmap_contract.py",
        "python/carnot/reporting/v755_branch_contract.py",
        "python/carnot/reporting/v755_branch_runner.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/v759_branch_contract.py",
        "python/carnot/reporting/v756_branch_qualification.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "scripts/experiment_template.py",
        "ops/exclusion_manifest.yaml",
    ]:
        snapshots.append(dict(seal(ROOT / name, raw), executed_source=True))
    contract = authority(raw / "authority")
    controls = mutations(raw / "authority", raw / "controls") if contract["ready"] else []
    work = dict(
        contract=contract,
        history=historical,
        methods=deepcopy(METHOD_ROWS),
        snapshots=snapshots,
        input_failures=failures,
        authority_mutation_rows=controls,
        run_date=run_date,
        owner_pid=os.getpid(),
        start_ns=start,
        end_ns=time.monotonic_ns(),
        raw=str(raw.absolute()),
        preconditions=dict(
            producer_argv=sys.orig_argv,
            no_model_loads=True,
            no_device_probes=True,
            activation_authorized=False,
            generator_weights_frozen=True,
            arc_suspended=True,
            run_date_matches=run_date == "20261108",
            repository_virtualenv=Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
            free_bytes=shutil.disk_usage(raw).free,
            required_free_bytes=536870912,
            time_budget_s=4200,
            hard_cap_s=4800,
        ),
        raw_operands=[reference(p) for p in sorted((raw / "controls").rglob("*")) if p.is_file()],
    )
    atomic_json(raw / "work.json", work)
    progress(
        "preconditions_after",
        sum(r["matched"] for r in contract["rows"]),
        14 - sum(r["matched"] for r in contract["rows"]),
    )
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Reuse the current receipt reducer; only exact custody earns readiness."""
    end = max(
        [
            work["end_ns"],
            *[r.get("ended_monotonic_ns", work["end_ns"]) for r in validation["receipts"]],
        ]
    )
    prepared = dict(
        work,
        end_ns=end,
        manifests=[],
        history=dict(work["history"], original_child_qualified=work["history"]["authenticated"]),
        branches=dict(runtime=dict(ready=False), arc=dict(ready=False), failures=[]),
    )
    value = base.build(prepared, validation)
    tasks, resources, controls = (
        work["contract"]["tasks"],
        work["preconditions"],
        work["authority_mutation_rows"],
    )
    producers = {t["id"]: t for t in tasks}
    gates = [
        dict(
            edge,
            required_field_declared=edge["artifact_field"]
            in _required_artifact_fields_block(producers[edge["upstream"]]["prompt"]),
        )
        for edge in base.edges(tasks)
    ]
    independent = all(g["upstream"] != TASK for t in tasks for g in t["gated_on"])
    checked = bool(
        value["required_checks_passed"]
        and resources["repository_virtualenv"]
        and resources["run_date_matches"]
        and resources["free_bytes"] >= resources["required_free_bytes"]
        and value["duration_s"] < 4200
        and all(r["passed"] for r in controls)
    )
    ready = bool(
        checked
        and work["contract"]["ready"]
        and work["history"]["authenticated"]
        and not work["input_failures"]
        and len(controls) == 5
        and independent
        and all(r["required_field_declared"] for r in gates)
    )
    verdict = "disqualified" if not checked else "null" if ready else "blocked"
    for field in [
        "preserved_v754_reference",
        "runtime_repair_ready_score",
        "arc_new_outcomes_available_score",
        "branch_admission_rows",
    ]:
        value.pop(field)
    value.update(
        experiment_id=8930,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7648930,
        honest_verdict=f"complete_{verdict}_v764_branch_contract",
        verdict_class=verdict,
        contract_ready_score=int(ready),
        required_checks_passed=checked,
        rows=[],
        contract_rows=work["contract"]["rows"],
        intended_count=14,
        censored_count=14 - value["completed_count"],
        full_task_digest=canonical_hash(tasks),
        canonical_tasks_sha256=PIN,
        execution_snapshot=[s for s in work["snapshots"] if s.get("current_execution_authority")],
        preserved_v763_reference=work["history"],
        authority_comparison_mode=work["contract"]["comparison_mode"],
        source_method_rows=work["methods"],
        prior_scope_rows=[dict(task_id=t["id"], **p) for t in tasks for p in t["prior_failures"]],
        authority_mutation_rows=controls,
        gate_edge_rows=gates,
        execution_venue="host",
        preconditions_checked=resources,
        gate_check_summary=work["contract"]["failures"]
        + work["history"]["failures"]
        + work["input_failures"],
        acceptance_gates=dict(
            exact_contract=work["contract"]["ready"],
            owned_validation=checked,
            preserved_history=work["history"]["authenticated"],
            independent_cpu_roots=independent,
            exact_gate_fields=all(r["required_field_declared"] for r in gates),
            mutation_controls=len(controls) == 5 and all(r["passed"] for r in controls),
        ),
        code_config_hashes=[s for s in work["snapshots"] if s.get("executed_source")],
        methodology_note="Administrative custody only. The live pilot qualifies an error source, not detector benefit. CPU science admits its own inputs. Historical outcomes and actual scheduler skips retain their original verdicts. No activation, device probes, model work or external publication.",
    )
    if verdict == "blocked":
        value["inference_substrate_class"] = "blocked_no_run"
        value["current_work_receipt"]["inference_substrate_class"] = "blocked_no_run"
    value["raw_shard_hashes"] += work["raw_operands"]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Exact immutable custody and owned checks establish administrative readiness only. Contract, prior and literature rows are receipts outside comparative rows. Current compute is zero; upstream evidence remains opaque and proves no new detector value or independent natural-language generalization.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild reported claims from sealed operands, including rehashed authority edits."""
    try:
        value = json.loads(path.read_bytes())
        if value["required_checks_passed"] and not verify_measured(
            value["owned_statement_coverage"], OWNED
        ):
            raise ValueError("unqualified_statements")
        for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
        ref = value["raw_shard_hashes"][0]
        work = json.loads(Path(ref["path"]).read_bytes())
        if (
            work["contract"] != authority(Path(work["raw"]) / "authority")
            or work["methods"] != METHOD_ROWS
        ):
            raise ValueError("authority_drift")
        for saved in work["history"]["references"]:
            if saved["source"]["path"] == str(ROOT / HISTORY):
                if (
                    work["history"]["outcome_rows"]
                    != json.loads(Path(saved["path"]).read_bytes())["outcome_rows"]
                ):
                    raise ValueError("history_drift")
        validation_path = Path(work["raw"]) / "validation.json"
        validation = (
            json.loads(validation_path.read_bytes())
            if validation_path.is_file()
            else dict(
                receipts=value["validation_receipts"],
                coverage=value["owned_statement_coverage"],
                global_test_debt=value["global_test_debt"],
            )
        )
        expected = build(dict(work, raw_reference=ref), validation)
        if validation_path.is_file():
            expected["raw_shard_hashes"].append(reference(validation_path))
        expected["reproducibility_checksum"] = checksum(expected)
        passed = value == expected and checksum(value) == value["reproducibility_checksum"]
        return dict(passed=passed, reason="replay_passed" if passed else "semantic_drift")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
