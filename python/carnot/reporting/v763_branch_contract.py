"""REQ-REPORT-8917: bind administrative authority without claiming scientific benefit."""

from copy import deepcopy
import atexit
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v755_branch_contract as base
from carnot.reporting import v762_branch_contract as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v749_contract_methods import encode as encode
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.audit_roadmap_gates import _required_artifact_fields_block
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = prior.ROOT, prior.DESIGN, prior.ACTIVE, prior.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8917_v763_branch_contract",
    "exp8917-branch-contract",
    "2026.11.763",
)
PIN = "0c93ebaa0d8873bbe95ae266e47c76f0128433bc6c3dd3a4d10f9e72bbb96ab7"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v763_branch_contract.py", CLI]
TEST = "tests/python/test_v763_branch_contract_8917.py"
HISTORY = "results/experiment_8916_v762_capstone.json"
DIRECTIVE = "docs/research-notes/bfs-labeled-error-source-plan-2026-11-07.md"
LITERATURE = "research-references.md"
PINS = {
    HISTORY: "sha256:41508277b0c79c4bbcd0493f888b29a5117577916e5edfcddb8f6cbf166a8066",
    "openspec/change-proposals/research-roadmap-v762-preserved-20261108.md": "sha256:b9663afee4f50f303219e263359656e852e9d0c6259bea33fd74c5e34848a918",
    "openspec/change-proposals/research-roadmap-v762-queue-preserved-20261108.yaml": "sha256:924cf84e75570a869a5e2fb743dadd6359f2155f9ef5387666f065e5e8bd8912",
    DIRECTIVE: "sha256:b5e946e76a79130a28c4bac31f5db35dfa09245b04edb8bd72287f55ca9b6b3d",
}
reference, checksum, seal = prior.reference, prior.checksum, prior.seal
METHOD_ROWS = [
    dict(
        method="ImplicitHop",
        tasks=[8918],
        source="https://arxiv.org/abs/2610.41169",
        limits="Exactly 50 graphs, k=2, reasoning off, 50-60 nodes, mean degree 3, two-letter labels; BFS final answers; >=10 errors; GPU 0 only. No full collection or localization.",
    ),
    dict(
        method="SPIN",
        tasks=[8919, 8922, 8925],
        source="https://arxiv.org/abs/2610.27835",
        limits="Alpha=0.25, beta=0.5; existing small head, objective and promotion guard; 8 development and 24 future streams; nine simultaneous bounds, alpha=0.01. Frozen generator.",
    ),
    dict(
        method="active_testing_label_accounting",
        tasks=[8920, 8923],
        source="https://arxiv.org/abs/2607.08347",
        limits="Exact accepted-label identity, all 7 policies, 4 groups, 240 selection labels per group, 960 certification candidates, 40 fresh draw streams. Not PPAT's estimator.",
    ),
    dict(
        method="hardware_orchestration",
        tasks=[8921, 8927, 8928],
        source="https://arxiv.org/abs/2602.15985",
        limits="Preallocated timestamps, deferred receipts, <=5% simultaneous observer overhead, 24 worlds and 36 cells; orchestration and transfer charged; unknown fractions stay unknown.",
    ),
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the supervisor can distinguish work from a stalled run."""
    print(f"[exp8917] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, check: str, field: str, expected: Any, observed: str) -> Json:
    """Retain missing bytes separately from a rejected producer or a measured zero."""
    return dict(
        check=check,
        upstream=TASK,
        artifact_field=field,
        expected=expected,
        observed=observed,
        **reference(path),
    )


def authority(root: Path) -> Json:
    """Compare the visible table, complete JSON objects and current queue independently."""
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
        if active["milestone"] == MILESTONE:
            mode, selected = "activated", active
        elif staged is not None and staged["milestone"] == MILESTONE:
            mode, selected = "staged", staged
        else:
            raise ValueError("no_current_authority")
        queues = [deepcopy(staged or selected), deepcopy(selected)]
        for queue in queues:
            Roadmap.model_validate(queue)
            for actual, expected in zip(queue["tasks"], tasks):
                if actual == dict(expected, prompt=encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
        checked = compare_contract(text, *queues, milestone=MILESTONE, first_id=8917, count=13)
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
            and canonical_hash(tasks) == "sha256:" + PIN
            and "<!-- V763_TASK_CONTRACT_END -->" in text
        ):
            raise ValueError("independent_authority_mismatch")
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, yaml.YAMLError):
        failures.append(
            failure(
                path,
                "complete_authority",
                "full_task_objects",
                "13 exact ordered full objects, table, digest, closing marker and current queue",
                "missing_file" if not path.is_file() else "rejected_producer",
            )
        )
    for i in range(13):
        if i >= len(rows):
            rows.append(
                dict(
                    unit_id=f"exp{8917 + i}-missing",
                    order=i + 1,
                    checks=dict(independent_authority=False),
                    matched=False,
                )
            )
        row = rows[i]
        matched = row["matched"] and not failures
        row.update(
            matched=matched,
            metric=int(matched),
            absolute_metric=int(matched),
            denominator=1,
            raw_numerator=sum(row["checks"].values()),
            raw_denominator=len(row["checks"]),
            cluster_id="administrative_contract",
            arm="full_object_custody",
            seed=None,
            status="completed" if matched else "censored",
            censored=not matched,
            missing_reason=None if matched else "independent_authority_unavailable_or_rejected",
            excluded=False,
            effective_independent_groups=0,
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
    """Each rejection starts with a valid private authority, so controls remain causal."""
    text = (source / DESIGN).read_text()
    queue = yaml.load(
        (
            source / (STAGED if authority(source)["comparison_mode"] == "staged" else ACTIVE)
        ).read_bytes(),
        Loader=yaml.CSafeLoader,
    )
    rows = []
    for mode in ["order", "prompt", "gate", "model", "path", "prior", "table", "closing_marker"]:
        root, changed, body = raw / mode, deepcopy(queue), text
        if mode == "order":
            changed["tasks"].reverse()
        if mode in {"prompt", "model"}:
            changed["tasks"][0][mode] = "forged"
        if mode == "gate":
            next(t for t in changed["tasks"] if t["gated_on"])["gated_on"][0]["artifact_field"] = (
                "forged"
            )
        if mode == "path":
            changed["tasks"][0]["deliverable"] = "results/forged.json"
        if mode == "prior":
            changed["tasks"][0]["prior_failures"][0]["verdict"] = "forged"
        if mode == "table":
            body = text.replace(
                "| 1 | `exp8917-branch-contract`", "| 2 | `exp8917-branch-contract`"
            )
        if mode == "closing_marker":
            body = text.replace("<!-- V763_TASK_CONTRACT_END -->", "")
        path = root / DESIGN
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        (root / ACTIVE).write_text(yaml.safe_dump(changed))
        rows.append(
            dict(
                mutation=mode,
                passed=not authority(root)["ready"],
                references=[reference(path), reference(root / ACTIVE)],
            )
        )
        progress("authority_control_" + mode, len(rows), 8 - len(rows))
    return rows


def history(raw: Path) -> Json:
    """Preserve original outcomes and retained bytes; never replay a stale queue as work."""
    refs, failures, outcomes, operands, absences = [], [], [], {}, []
    for name, pin in PINS.items():
        path = ROOT / name
        if reference(path)["sha256"] != pin:
            failures.append(
                failure(
                    path,
                    "original_v762",
                    "sha256",
                    pin,
                    "missing_file" if not path.is_file() else "rejected_producer",
                )
            )
            continue
        refs.append(seal(path, raw))
        if name != HISTORY:
            continue
        try:
            value = json.loads(path.read_bytes())
            outcomes = value["outcome_rows"]
            retained = value["raw_shard_hashes"]
        except (ValueError, KeyError, TypeError):
            failures.append(
                failure(
                    path,
                    "original_v762_schema",
                    "outcome_rows",
                    "original producer object",
                    "rejected_producer",
                )
            )
            continue
        catalog = {r["sha256"]: r for r in retained if r.get("sha256")}
        imported = retained + value["code_config_hashes"]
        for row in outcomes:
            imported += [r["reference"] for r in row.get("scheduler_receipts", [])]
            if row.get("primary_reference") and row["disposition"] != "conductor_pre_gate":
                imported.append(row["primary_reference"])
        for ref in imported:
            bound = catalog.get(ref["sha256"], ref)
            operands[(bound["path"], bound["sha256"])] = bound
    for i, ref in enumerate(operands.values()):
        path = Path(ref["path"])
        if ref["sha256"] is None and not path.is_file():
            absences.append(dict(ref, status="historical_asserted_absence"))
        elif reference(path)["sha256"] != ref["sha256"]:
            failures.append(
                failure(
                    path,
                    "original_retained_operand",
                    "sha256",
                    ref["sha256"],
                    "missing_file" if not path.is_file() else "rejected_producer",
                )
            )
        else:
            refs.append(seal(path, raw))
        if i % 100 == 0:
            progress("historical_operands", i + 1, len(operands) - i - 1)
    return dict(
        authenticated=not failures,
        references=refs,
        failures=failures,
        outcome_rows=outcomes,
        asserted_absence_rows=absences,
        scientific_results_admitted=False,
        historical_replay_required=False,
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze this invocation before validation, with no model or device access."""
    progress("preconditions_before", 0, 13)
    if (raw / "work.json").exists():
        raise FileExistsError("immutable_invocation_already_exists")
    start = time.monotonic_ns()
    template = ExperimentTemplate(
        8917, "V763 contract", "results/private.json", repo_root=raw, seed=7638917
    )
    atexit.unregister(template.teardown)  # The default exit cleanup probes CUDA.
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    historical = history(raw)
    snapshots, failures = list(historical["references"]), []
    for name in [DESIGN, ACTIVE, STAGED, DIRECTIVE, LITERATURE]:
        path = root / name
        if path.is_file():
            snapshots.append(
                dict(seal(path, raw), current_execution_authority=name in {DESIGN, ACTIVE, STAGED})
            )
            target = raw / "authority" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        elif name != STAGED:
            failures.append(
                failure(path, "authority_input", name, "original source bytes", "missing_file")
            )
    for name in [
        *OWNED,
        TEST,
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "python/carnot/reporting/roadmap_contract.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/v755_branch_contract.py",
        "python/carnot/reporting/v755_branch_runner.py",
        "scripts/experiment_template.py",
        "ops/exclusion_manifest.yaml",
    ]:
        snapshots.append(seal(ROOT / name, raw))
    contract = authority(raw / "authority")
    controls = mutations(raw / "authority", raw / "controls") if contract["ready"] else []
    directive = reference(raw / "authority" / DIRECTIVE)
    if directive["sha256"] != PINS.get(
        DIRECTIVE, "sha256:b5e946e76a79130a28c4bac31f5db35dfa09245b04edb8bd72287f55ca9b6b3d"
    ):
        failures.append(
            failure(
                root / DIRECTIVE,
                "operator_directive",
                "sha256",
                "November 7 approved conductor pilot",
                "missing_file" if directive["sha256"] is None else "rejected_producer",
            )
        )
    resources = dict(
        producer_argv=sys.orig_argv,
        template_experiment_id=template.exp_id,
        repository_virtualenv=Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
        private_output_writable=True,
        private_mode=oct(raw.stat().st_mode & 0o777),
        free_bytes=shutil.disk_usage(raw).free,
        required_free_bytes=536870912,
        time_budget_s=4200,
        hard_cap_s=4800,
        no_model_loads=True,
        no_device_probes=True,
        activation_authorized=False,
        run_date_matches=run_date == "20261108",
    )
    work = dict(
        contract=contract,
        history=historical,
        methods=deepcopy(METHOD_ROWS),
        snapshots=snapshots,
        input_failures=failures,
        operator_directive_reference=directive,
        authority_mutation_rows=controls,
        run_date=run_date,
        owner_pid=os.getpid(),
        start_ns=start,
        end_ns=time.monotonic_ns(),
        raw=str(raw.absolute()),
        preconditions=resources,
        raw_operands=[reference(p) for p in sorted((raw / "controls").rglob("*")) if p.is_file()],
    )
    atomic_json(raw / "work.json", work)
    progress(
        "preconditions_after",
        sum(r["matched"] for r in contract["rows"]),
        13 - sum(r["matched"] for r in contract["rows"]),
    )
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Only validated immutable custody earns readiness; administrative success is null."""
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
        and len(controls) == 8
        and all(r["required_field_declared"] for r in gates)
        and independent
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
        experiment_id=8917,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7638917,
        honest_verdict=f"complete_{verdict}_v763_branch_contract",
        verdict_class=verdict,
        contract_ready_score=int(ready),
        contract_rows=work["contract"]["rows"],
        full_task_digest=canonical_hash(tasks),
        canonical_tasks_sha256=PIN,
        execution_snapshot=[s for s in work["snapshots"] if s.get("current_execution_authority")],
        literature_method_rows=work["methods"],
        operator_directive_reference=work["operator_directive_reference"],
        preserved_v762_reference=work["history"],
        authority_comparison_mode=work["contract"]["comparison_mode"],
        authority_mutation_rows=controls,
        gate_edge_rows=gates,
        execution_venue="host",
        required_checks_passed=checked,
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
            operator_directive=not work["input_failures"],
            mutation_controls=len(controls) == 8 and all(r["passed"] for r in controls),
        ),
        code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
        methodology_note="Administrative ingestion only. The approved live preflight belongs to Exp8918 alone and supersedes passive receipt inspection only there. CPU branches check their own custody. V762 original nulls, scheduler skip and cost disqualification remain unchanged; no activation, GPU probe, science or publication.",
    )
    value["raw_shard_hashes"] += work["raw_operands"]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Immutable authority and owned validation establish administrative readiness only. Original producer bytes retain their verdicts; external absence blocks, failed owned controls disqualify. Pilot authority does not authorize work here or gate independent CPU branches.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild claims from sealed operands, rejecting rehashed semantic tampering."""
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
        if work["contract"] != authority(Path(work["raw"]) / "authority"):
            raise ValueError("authority_drift")
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
