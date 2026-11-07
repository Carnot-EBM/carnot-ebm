"""REQ-REPORT-8853: bind complete authority without reviving failed history."""

from copy import deepcopy
from datetime import UTC, datetime
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
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.v756_branch_qualification import verify_measured
from carnot.reporting.v725_contract_methods import runtime as read_runtime
from carnot.verify.sentence_transport_methods_8179 import TOKENIZER_PATH, TOKENIZER_PIN
from scripts.audit_roadmap_gates import _required_artifact_fields_block
from scripts.experiment_template import normalize_artifact_for_template_write
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8853_v758_branch_contract",
    "exp8853-branch-contract",
    "2026.11.758",
)
DESIGN, ACTIVE, STAGED = base.DESIGN, base.ACTIVE, base.STAGED
ARCHIVE = "openspec/change-proposals/research-roadmap-v757-preserved-20261105.md"
PRIOR = "results/experiment_8840_v757_branch_contract.json"
CAPSTONE = "results/experiment_8852_v757_capstone.json"
PINS = {
    PRIOR: "f58e5248daf988f09ae888a953d8586abe261512a9b3d3456bdd70e4471f0e72",
    CAPSTONE: "766fc97c3d2398858ccd3178f8b52e4222c47833cef8b8cc11798feab6d99962",
    ARCHIVE: "282398e6c06caf439de5591b1e245312fe4ddc42f6c0812e48c110a1e21f21ed",
}
PIN = "6eda8e81b765b13fec4210605ab6d2367f530b19c656b5e81b5ae4512bfeb7f5"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v758_branch_contract.py", CLI]
TEST = "tests/python/test_v758_branch_contract_8853.py"
MODEL_REVISION, MODEL_PIN = TOKENIZER_PATH.parent.name, TOKENIZER_PIN
RUNTIME_VALIDATORS = [
    "python/carnot/reporting/v725_contract_methods.py",
    "python/carnot/verify/cuda_reader_8410.py",
]
REUSED = [
    "python/carnot/reporting/v755_branch_contract.py",
    "python/carnot/reporting/v755_branch_runner.py",
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "scripts/experiment_template.py",
    "scripts/roadmap_schema.py",
    "scripts/audit_roadmap_gates.py",
    "scripts/exclusion_manifest_lint.py",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
    *RUNTIME_VALIDATORS,
]
reference, checksum = base.reference, base.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the supervisor can distinguish work from a stalled run."""
    print(f"[exp8853] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, check: str, field: str, expected: Any, observed: Any) -> Json:
    """Keep absent inputs distinct from rejected bytes and measured zero."""
    return dict(
        check=check,
        upstream=TASK,
        artifact_field=field,
        expected=expected,
        observed=observed,
        **reference(path),
    )


def authority(root: Path) -> Json:
    """Use full objects and an independent digest; a queue cannot repair missing JSON."""
    try:
        text = (root / DESIGN).read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        staged = (
            yaml.safe_load((root / STAGED).read_bytes())
            if (root / STAGED).is_file()
            else deepcopy(active)
        )
        if active["milestone"] == "2026.11.757" and (root / STAGED).is_file():
            active = deepcopy(staged)
        for queue in [staged, active]:
            Roadmap.model_validate(queue)
        # Only activation may encode braces for the conductor's format operation.
        if active["milestone"] == MILESTONE:
            for actual, expected in zip(active["tasks"], tasks):
                if actual == dict(expected, prompt=encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
            if not (root / STAGED).is_file():
                staged = deepcopy(active)
        checked = compare_contract(
            text, staged, active, milestone=MILESTONE, first_id=8853, count=13
        )
        for i, row in enumerate(checked["rows"]):
            matched = (
                row["matched"]
                and i < len(tasks)
                and tasks[i] == staged["tasks"][i] == active["tasks"][i]
            )
            row["checks"]["full_object"] = matched
            row["raw_numerator"] = sum(row["checks"].values())
            row["raw_denominator"] = len(row["checks"])
            row.update(
                matched=matched,
                absolute_metric=int(matched),
                metric=int(matched),
                denominator=1,
                cluster_id="administrative_contract",
                status="completed" if matched else "censored",
                missing_reason=None if matched else "authority_mismatch",
                excluded=False,
                effective_independent_groups=0,
            )
        ready = (
            checked["passed"]
            and all(r["matched"] for r in checked["rows"])
            and canonical_hash(tasks) == "sha256:" + PIN
            and len({t["deliverable"] for t in tasks}) == 13
            and "<!-- V758_TASK_CONTRACT_END -->" in text
        )
        return dict(ready=ready, rows=checked["rows"], tasks=tasks, table=table)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return dict(ready=False, rows=[], tasks=[], table=[])


def mutations(source: Path, raw: Path) -> list[Json]:
    """Private corruptions demonstrate rejection without editing the live authority."""
    text = (source / DESIGN).read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    queue = yaml.safe_load((source / ACTIVE).read_bytes())
    rows = []
    for mode in [
        "missing_last_task",
        "truncation",
        "prompt",
        "gate",
        "model",
        "closing_marker",
        "table",
    ]:
        root = raw / mode
        design = root / DESIGN
        design.parent.mkdir(parents=True, exist_ok=True)
        changed, body = deepcopy(tasks), text
        if mode == "missing_last_task":
            changed.pop()
        if mode == "truncation":
            body = text[: text.index('"id": "exp8865-')]
        if mode in {"prompt", "model"}:
            changed[0][mode] = "forged"
        if mode == "gate":
            changed[3]["gated_on"][0]["artifact_field"] = "wrong_gate"
        if mode == "closing_marker":
            body = text.replace("<!-- V758_TASK_CONTRACT_END -->", "")
        if mode == "table":
            body = text.replace("| 1 | exp8853-branch-contract", "| 2 | exp8853-branch-contract")
        design.write_text(body)
        (root / ACTIVE).write_text(yaml.safe_dump(dict(queue, tasks=changed)))
        rows.append(dict(mutation=mode, passed=not authority(root)["ready"]))
        progress("authority_control_" + mode, len(rows), 7 - len(rows))
    return rows


def seal(source: Path, raw: Path) -> Json:
    """Store content-addressed copies so future rollover cannot rewrite this invocation."""
    target = raw / "inputs" / (sha256_file(source)[7:] + "-" + source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError("immutable_operand_conflict")
    if not target.exists():
        shutil.copyfile(source, target)
    return dict(source=reference(source), **reference(target))


def history(raw: Path) -> Json:
    """Authenticate historical authority without turning failed replay into a science gate."""
    refs, failures = [], []
    for name, pin in PINS.items():
        path = ROOT / name
        ref = reference(path)
        if ref["sha256"] != "sha256:" + pin:
            failures.append(
                failure(
                    path,
                    "preserved_history",
                    "sha256",
                    pin,
                    "missing_file" if ref["sha256"] is None else "rejected_producer",
                )
            )
        else:
            refs.append(seal(path, raw))
            if name == PRIOR:
                producer = json.loads(path.read_bytes())
                operands = {r["path"]: r for r in producer["raw_shard_hashes"]}
                for operand in operands.values():
                    source = Path(operand["path"])
                    if reference(source)["sha256"] != operand["sha256"]:
                        failures.append(
                            failure(
                                source,
                                "preserved_raw_operand",
                                "sha256",
                                operand["sha256"],
                                "missing_file" if not source.is_file() else "rejected_producer",
                            )
                        )
                    else:
                        refs.append(seal(source, raw))
    return dict(
        authenticated=not failures,
        references=refs,
        failures=failures,
        verdict_class="disqualified",
        scientific_results_admitted=False,
        original_child_qualified=False,
        historical_replay_required=False,
    )


def runtime(root: Path, raw: Path) -> Json:
    """Admit only supplied authenticated execution; metadata alone cannot prove repair."""
    path = root / "ops/cuda-runtime-repair-receipt.json"
    checks: Json = {}
    refs = []
    if path.is_file():
        refs.append(seal(path, raw))
        try:
            value = json.loads(path.read_bytes())
            admitted = read_runtime(root)
            checks["original_causal_validator"] = admitted["ready"]
            changes = admitted.get("changes", [])
            checks["dated_causal_change"] = (
                value["run_date"] == "20261105"
                and datetime.fromtimestamp(value["executed_at_ns"] / 1e9, UTC).strftime("%Y%m%d")
                == value["run_date"]
                and bool(value["description"])
                and any(
                    r["field"] in {"libraries", "environment"} and r["changed"] for r in changes
                )
            )
            model = value["mandated_model_provenance"]
            checks["exact_model"] = (
                model["hf_id"] == "unsloth/Qwen3.8-27B-GGUF"
                and model["revision"] == MODEL_REVISION
                and model["revision"] in model["path"]
                and model["sha256"] == MODEL_PIN
                and reference(Path(model["path"]))["sha256"] == model["sha256"]
            )
            validators = value["validator_references"]
            checks["current_validator_custody"] = {r["source_path"] for r in validators} == set(
                RUNTIME_VALIDATORS
            ) and all(
                reference(ROOT / r["source_path"])["sha256"] == r["sha256"] for r in validators
            )
            execution = value["transcript_reference"]
            checks["authenticated_execution"] = (
                reference(Path(execution["path"]))["sha256"] == execution["sha256"]
            )
            if checks["authenticated_execution"]:
                evidence = json.loads(Path(execution["path"]).read_bytes())
                checks["actual_context_copy_offload"] = (
                    admitted["ready"]
                    and evidence["model_provenance"] == model
                    and evidence["offloaded_layers"] > 0
                )
                refs.append(seal(Path(execution["path"]), raw))
            for key in ["current_reference", "previous_reference"]:
                operand = value[key]
                if reference(Path(operand["path"]))["sha256"] != operand["sha256"]:
                    raise ValueError("runtime_operand_drift")
                refs.append(seal(Path(operand["path"]), raw))
        except (OSError, ValueError, KeyError, TypeError) as error:
            checks["complete_supplied_evidence"] = False
            checks["diagnostic"] = str(error)
    ready = bool(checks) and all(checks.values())
    failures = (
        []
        if ready
        else [
            failure(
                path,
                "authenticated_changed_runtime",
                "runtime_repair_ready_score",
                "dated causal library/environment change, current validators, exact GGUF and actual context/copy/offload",
                "missing_file" if not path.is_file() else "rejected_producer",
            )
        ]
    )
    if path.is_file() and not ready:
        failures += [
            failure(
                path,
                key,
                key,
                True,
                "missing_field" if key == "complete_supplied_evidence" else "observed_zero",
            )
            for key, passed in checks.items()
            if passed is False
        ]
    return dict(
        ready=ready,
        checks=checks,
        references=refs,
        gate_check_summary=failures,
        model_loads=0,
        device_probes=0,
    )


def methods(root: Path) -> list[Json]:
    """Read the supplied map, retaining retrieval limits instead of importing paper claims."""
    text = (root / DESIGN).read_text()
    section = text.split("## Research basis", 1)[1].split("## V758 architecture", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [v.strip() for v in line.strip("|").split("|")]
        if len(cells) == 3 and cells[0].startswith("["):
            rows.append(
                dict(
                    source=cells[0],
                    experimental_use=cells[1],
                    limit=cells[2],
                    paper_results_inherited=False,
                )
            )
    return rows


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Check resources and freeze operands before validation or publication can begin."""
    progress("preconditions_before")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    probe = raw / "write_probe"
    probe.write_text(TASK)
    writable = probe.read_text() == TASK
    probe.unlink()
    resources = dict(
        repository_virtualenv=Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
        installed_dependencies={
            p: importlib.metadata.version(p) for p in ["coverage", "pytest", "pydantic", "PyYAML"]
        },
        private_output_writable=writable,
        free_bytes=shutil.disk_usage(raw).free,
        required_free_bytes=134217728,
        time_budget_s=4200,
        no_model_loads=True,
        no_device_probes=True,
        activation_authorized=False,
    )
    contract = authority(root)
    historical = history(raw)
    branches = runtime(root, raw)
    snapshots = historical["references"] + branches["references"]
    for name in [DESIGN, ACTIVE, STAGED]:
        if (root / name).is_file():
            snapshots.append(seal(root / name, raw))
            target = raw / "authority" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    for name in ["research-references.md", "ops/exclusion_manifest.yaml"]:
        snapshots.append(seal(ROOT / name, raw))
    controls = mutations(root, raw / "controls") if contract["ready"] else []
    raw_operands = [reference(p) for p in sorted((raw / "controls").rglob("*")) if p.is_file()]
    work = dict(
        contract=contract,
        history=historical,
        branches=branches,
        methods=methods(root) if contract["ready"] else [],
        snapshots=snapshots,
        authority_mutation_rows=controls,
        run_date=run_date,
        owner_pid=os.getpid(),
        start_ns=start,
        end_ns=time.monotonic_ns(),
        raw=str(raw.absolute()),
        preconditions=resources,
        raw_operands=raw_operands,
    )
    atomic_json(raw / "work.json", work)
    progress("preconditions_after", len(contract["rows"]), 13 - len(contract["rows"]))
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Separate current checks, full authority and optional runtime admission."""
    prepared = dict(
        work,
        history=dict(work["history"], original_child_qualified=True),
        manifests=[],
        branches=dict(runtime=work["branches"], arc=dict(ready=False), failures=[]),
    )
    prepared["end_ns"] = max(
        [
            work["end_ns"],
            *[r.get("ended_monotonic_ns", work["end_ns"]) for r in validation["receipts"]],
        ]
    )
    value = base.build(prepared, validation)
    value["phase_spans"] = [
        dict(phase="operand_custody", start_s=0, end_s=(work["end_ns"] - work["start_ns"]) / 1e9),
        dict(
            phase="owned_validation",
            start_s=(work["end_ns"] - work["start_ns"]) / 1e9,
            end_s=value["duration_s"],
        ),
    ]
    resources = work["preconditions"]
    controls = work["authority_mutation_rows"]
    checked = (
        value["required_checks_passed"]
        and resources["private_output_writable"]
        and resources["repository_virtualenv"]
        and resources["free_bytes"] >= resources["required_free_bytes"]
        and all(r["passed"] for r in controls)
    )
    ready = bool(
        checked
        and work["contract"]["ready"]
        and work["history"]["authenticated"]
        and len(controls) == 7
        and len(work["methods"]) == 5
    )
    verdict = "disqualified" if not checked else "null" if ready else "blocked"
    failures = work["history"]["failures"] + work["branches"]["gate_check_summary"]
    if not work["contract"]["ready"]:
        failures.append(
            failure(
                Path(work["raw"]) / "authority" / DESIGN,
                "complete_authority",
                "full_task_objects",
                "13 complete matching task objects and closing marker",
                "missing_field",
            )
        )
    value.pop("preserved_v754_reference")
    value.pop("arc_new_outcomes_available_score")
    value.update(
        experiment_id=8853,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7588853,
        honest_verdict=f"complete_{verdict}_v758_branch_contract",
        verdict_class=verdict,
        contract_ready_score=int(ready),
        required_checks_passed=bool(checked),
        runtime_repair_ready_score=int(ready and work["branches"]["ready"]),
        canonical_tasks_sha256=PIN,
        execution_venue="host",
        preconditions_checked=resources,
        gate_check_summary=failures,
        authority_mutation_rows=controls,
        source_method_map=work["methods"],
        preserved_v757_reference=work["history"],
        external_admission_rows=[
            dict(
                branch="runtime",
                downstream="exp8863-qwen-source-canary",
                optional=True,
                ready=bool(ready and work["branches"]["ready"]),
                checks=work["branches"]["checks"],
                independent_cpu_roots_gated=False,
            )
        ],
        acceptance_gates=dict(
            exact_contract=work["contract"]["ready"],
            owned_validation=bool(checked),
            mutation_controls=len(controls) == 7 and all(r["passed"] for r in controls),
            preserved_history=work["history"]["authenticated"],
            method_map=len(work["methods"]) == 5,
        ),
        code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST] + REUSED],
        methodology_note="Administrative full-object equality and supplied method ingestion; no science, model execution, generator training, ARC tasks or activation. Historical disqualification remains preserved.",
    )
    value["source_method_rows"] = work["methods"]
    tasks = work["contract"]["tasks"]
    producers = {t["id"]: t for t in tasks}
    value["gate_producer_rows"] = [
        dict(
            downstream=t["id"],
            **g,
            required_field_declared=g["artifact_field"]
            in _required_artifact_fields_block(producers[g["upstream"]]["prompt"]),
        )
        for t in tasks
        for g in t["gated_on"]
    ]
    value["arc_task_count"] = sum("arc" in t["title"].lower().split() for t in tasks)
    value["generator_training_task_count"] = sum(
        "generator training" in t["title"].lower() for t in tasks
    )
    value["raw_shard_hashes"] += work["raw_operands"]
    value["branch_admission_rows"] = value["external_admission_rows"]
    value["raw_shard_hashes"] += (
        [reference(Path(work["raw"]) / "validation.json")]
        if (Path(work["raw"]) / "validation.json").is_file()
        else []
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Complete authority and authenticated current checks establish administrative readiness only. Optional external absence never gates independent CPU roots; historical results are not new measurements.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Reconstruct sealed authority and reduction instead of trusting reported scores."""
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
        # The existing runner appends the same sealed validation operand before publication.
        if validation_path.is_file():
            expected["raw_shard_hashes"].append(reference(validation_path))
            expected["reproducibility_checksum"] = checksum(expected)
        passed = value == expected and checksum(value) == value["reproducibility_checksum"]
        return dict(passed=passed, reason="replay_passed" if passed else "semantic_drift")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
