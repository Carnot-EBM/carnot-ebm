"""REQ-REPORT-8814: certify custody without treating it as scientific benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
)
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.v749_contract_methods import encode as encode

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, MILESTONE = (
    "experiment_8814_v755_branch_contract",
    "exp8814-branch-contract",
    "2026.11.755",
)
DESIGN, ACTIVE, STAGED = (
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "research-roadmap.yaml",
    "research-roadmap-next.yaml",
)
PIN = "12f51392c099f3ae4f18a879428d7512b6ac0e5fa231cea0321033859d9463f7"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v755_branch_contract.py",
    "python/carnot/reporting/v755_branch_runner.py",
    CLI,
]
TEST = "tests/python/test_v755_branch_contract_8814.py"
HISTORY = "results/experiment_8813_v754_capstone.json"
ARCHIVE = "openspec/change-proposals/research-roadmap-v754-preserved-20261104.md"
HISTORY_PINS = {
    HISTORY: "1305c14e79e41a83d601e53eae0ba8ed02152f92d071cd90885e08fa52b48beb",
    ARCHIVE: "021962592d30e304c97c7d45b99ffef9997837e4a21b395893c3d76316a56f17",
    "results/experiment_8801_v754_repaired_memory_qualification.json": "c55ed8c40986921597e6f9a1eb760f0844a32585d40ca3ef95b8b7316aa3c1e6",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """A flushed boundary lets the parent distinguish work from a silent process."""
    print(f"[exp8814] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Record absence explicitly so it cannot be confused with an observed zero."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def authority(root: Path) -> Json:
    """Compare complete objects; the independent pin prevents coordinated rehashing."""
    try:
        text = (root / DESIGN).read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.load((root / ACTIVE).read_bytes(), Loader=yaml.CSafeLoader)
        staged = (
            yaml.load((root / STAGED).read_bytes(), Loader=yaml.CSafeLoader)
            if (root / STAGED).is_file()
            else deepcopy(active)
        )
        state = "active_v755"
        if active["milestone"] == "2026.11.754" and (root / STAGED).is_file():
            active, state = deepcopy(staged), "staged_v755"
        for roadmap in [active, staged]:
            for actual, expected in zip(roadmap["tasks"], tasks):
                if actual == dict(expected, prompt=encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
        checked = compare_contract(
            text, staged, active, milestone=MILESTONE, first_id=8814, count=13
        )
        for i, row in enumerate(checked["rows"]):
            matched = (
                row["matched"]
                and i < min(len(tasks), len(active["tasks"]), len(staged["tasks"]))
                and tasks[i] == active["tasks"][i] == staged["tasks"][i]
            )
            row.update(
                matched=matched,
                metric=int(matched),
                denominator=1,
                cluster_id="administrative_contract",
                arm="custody",
                status="completed" if matched else "censored",
                missing_reason=None if matched else "authority_mismatch",
                effective_independent_groups=0,
            )
        ready = (
            checked["passed"]
            and all(r["matched"] for r in checked["rows"])
            and canonical_hash(tasks) == "sha256:" + PIN
            and f"Canonical task SHA-256: `{PIN}`" in text
            and "<!-- V755_TASK_CONTRACT_END -->" in text
        )
        return dict(
            ready=bool(ready), rows=checked["rows"], tasks=tasks, table=table, active_state=state
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return dict(ready=False, rows=[], tasks=[], table=[], active_state="rejected")


def edges(tasks: list[Json]) -> list[Json]:
    """Retain each exact field and ordinal, including multiple gates on one task."""
    positions = {t["id"]: i + 1 for i, t in enumerate(tasks)}
    return [
        dict(
            downstream=t["id"],
            downstream_order=i + 1,
            gate_order=j + 1,
            upstream_order=positions[g["upstream"]],
            **g,
        )
        for i, t in enumerate(tasks)
        for j, g in enumerate(t["gated_on"])
    ]


def history(root: Path) -> Json:
    """Use actual V754 primaries and skips, never a completion marker or mutable design."""
    for name, pin in HISTORY_PINS.items():
        if sha256_file(root / name) != "sha256:" + pin:
            raise ValueError("historical_authority_drift:" + name)
    capstone = json.loads((root / HISTORY).read_bytes())
    tasks = parse_design((root / ARCHIVE).read_text(), milestone="2026.11.754")[1]
    if tasks != capstone["full_task_objects"]:
        raise ValueError("historical_task_drift")
    rows, refs = [], [reference(root / p) for p in HISTORY_PINS]
    for slot in capstone["rows"][:12]:
        key = "primary_reference" if slot["producer_executed"] else "conductor_reference"
        ref = slot[key]
        path = Path(ref["path"])
        if sha256_file(path) != ref["sha256"]:
            raise ValueError("historical_receipt_drift")
        value = json.loads(path.read_bytes())
        if slot["producer_executed"]:
            validate_primary(value, path)
        elif (
            value.get("blocked_at_layer") != "conductor_pre_gate"
            or value.get("honest_verdict") != "blocked_gate_check_failed"
        ):
            raise ValueError("unstructured_skip")
        rows.append(
            dict(
                task_id=slot["task_id"],
                kind="primary" if slot["producer_executed"] else "structured_skip",
                reference=ref,
                verdict=value.get("verdict_class", value["honest_verdict"]),
            )
        )
        refs.append(ref)
    rows.append(
        dict(
            task_id="exp8813-capstone",
            kind="primary",
            reference=reference(root / HISTORY),
            verdict=capstone["verdict_class"],
        )
    )
    qualification = json.loads((root / list(HISTORY_PINS)[2]).read_bytes())
    qualified = (
        qualification["acceptance_gates"]["repaired_original_validation"]
        and qualification["original_owned_statement_coverage"]["complete"]
        and qualification["repaired_owned_statement_coverage"]["complete"]
    )
    return dict(
        rows=rows,
        references=refs,
        primary_count=sum(r["kind"] == "primary" for r in rows),
        structured_skip_count=sum(r["kind"] == "structured_skip" for r in rows),
        original_child_qualified=bool(qualified),
        design_reference=reference(root / ARCHIVE),
    )


def methods(root: Path) -> list[Json]:
    """The notes bind only methods actually read, with no transfer of paper results."""
    return json.loads((root / "docs/research-notes/v755-methods.json").read_bytes())["rows"]


def optional_admission(root: Path, raw: Path) -> Json:
    """The shipped validators read receipts once without running a device or game."""
    from carnot.reporting.v752_branch_methods import runtime_input
    from carnot.reporting.v751_branch_evidence import freeze as arc_freeze

    progress("optional_runtime_before")
    runtime = runtime_input(root, raw / "runtime")
    progress("optional_runtime_after", 1)
    progress("optional_arc_before")
    arc = arc_freeze(raw / "arc")
    progress("optional_arc_after", 1)
    failures = runtime["gate_check_summary"] + arc["gate_check_summary"]
    for row in failures:
        row["artifact_field"] = row.get("artifact_field", row.get("field", row["check"]))
        if not Path(row["path"]).is_file():
            row["observed"] = "missing_file"
    return dict(runtime=runtime, arc=arc, failures=failures)


def freeze(root: Path, raw: Path, run_date: str, *, optional: bool = True) -> Json:
    """Copy branch operands before validation, keeping independent producer admission local."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    progress("contract_before")
    contract, prior, method_rows = authority(root), history(ROOT), methods(ROOT)
    progress("contract_after", len(contract["rows"]))
    for name in [DESIGN, ACTIVE, STAGED]:
        if (root / name).is_file():
            target = raw / "authority" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    branch = (
        optional_admission(root, raw)
        if optional
        else dict(
            runtime=dict(ready=False, references=[]),
            arc=dict(ready=False, references=[]),
            failures=[],
        )
    )
    refs = [reference(root / p) for p in [DESIGN, ACTIVE, STAGED] if (root / p).is_file()] + prior[
        "references"
    ]
    refs += [
        reference(ROOT / p)
        for p in [
            "docs/research-notes/v755-methods.md",
            "docs/research-notes/v755-methods.json",
            "results/raw/" + NAME + "/literature/retrievals.json",
        ]
    ]
    refs += (
        [r["source_reference"] for r in method_rows]
        + branch["runtime"]["references"]
        + branch["arc"]["references"]
    )
    snapshots = []
    for i, ref in enumerate(refs):
        source = Path(ref["path"])
        target = raw / "inputs" / f"{i:03d}-{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        snapshots.append(dict(source=ref, **reference(target)))
    manifests = []
    for task in contract["tasks"]:
        path = raw / "branches" / (task["id"] + ".json")
        atomic_json(
            path,
            dict(
                task=task,
                canonical_tasks_sha256=PIN,
                gates=task["gated_on"],
                upstream_task_ids=[g["upstream"] for g in task["gated_on"]],
                administrative_root_required=False,
            ),
        )
        manifests.append(reference(path))
    work = dict(
        contract=contract,
        history=prior,
        methods=method_rows,
        branches=branch,
        manifests=manifests,
        snapshots=snapshots,
        run_date=run_date,
        owner_pid=os.getpid(),
        start_ns=start,
        end_ns=time.monotonic_ns(),
        raw=str(raw.absolute()),
    )
    atomic_json(raw / "work.json", work)
    return dict(work, raw_reference=reference(raw / "work.json"))


def checksum(value: Json) -> str:
    """Exclude only the checksum itself so every reported claim is bound."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def build(work: Json, validation: Json) -> Json:
    """A missing optional receipt cannot turn valid administrative custody into failed science."""
    contract = work["contract"]
    checked = (
        all(r["passed"] for r in validation["receipts"]) and validation["coverage"]["complete"]
    )
    ready = checked and contract["ready"] and work["history"]["original_child_qualified"]
    verdict = "disqualified" if not checked else "null" if ready else "blocked"
    rows = contract["rows"]
    provenance = build_current_work_receipt(
        run_id=Path(work["raw"]).name,
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(administrative_only=True),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["start_ns"],
        ended_monotonic_ns=work["end_ns"],
    )
    value = dict(
        experiment_id=8814,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v755_branch_contract",
        verdict_class=verdict,
        MODEL_SPECS=[],
        model_invocation_counts=provenance["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=False,
        inference_substrate=provenance["inference_substrate"],
        inference_substrate_class=provenance["inference_substrate_class"],
        duration_s=provenance["duration_s"],
        random_seed=7558814,
        phase_spans=[dict(phase="artifact_custody", start_s=0, end_s=provenance["duration_s"])],
        preconditions_checked=dict(
            no_model_loads=True,
            no_device_probes=True,
            no_game_runs=True,
            activation_authorized=False,
            original_child_qualified=work["history"]["original_child_qualified"],
        ),
        rows=rows,
        intended_count=13,
        completed_count=sum(r["matched"] for r in rows),
        failed_count=0,
        censored_count=13 - sum(r["matched"] for r in rows),
        excluded_count=0,
        independent_count=0,
        gate_check_summary=work["branches"]["failures"],
        acceptance_gates=dict(
            exact_contract=contract["ready"],
            owned_validation=checked,
            original_child=work["history"]["original_child_qualified"],
        ),
        required_checks_passed=checked,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        exposure_scope="Administrative custody and method ingestion only; no new scientific benefit, natural-text or hidden-game generalization.",
        independent_generalization_score=0,
        contract_ready_score=int(ready),
        runtime_repair_ready_score=int(ready and work["branches"]["runtime"]["ready"]),
        arc_new_outcomes_available_score=int(ready and work["branches"]["arc"]["ready"]),
        full_task_objects=contract["tasks"],
        canonical_tasks_sha256=PIN,
        visible_task_table=contract["table"],
        gate_edge_rows=edges(contract["tasks"]),
        source_method_rows=work["methods"],
        branch_admission_rows=[
            dict(branch=k, ready=work["branches"][k]["ready"], optional=True)
            for k in ["runtime", "arc"]
        ],
        preserved_v754_reference=work["history"],
        source_artifact_hashes=[r["source"] for r in work["snapshots"]],
        code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
        raw_shard_hashes=[work["raw_reference"]] + work["snapshots"] + work["manifests"],
        cited_upstream_artifacts=work["history"]["references"],
        flagged_adversarial=False,
        current_work_receipt=provenance,
        branch_input_manifests=work["manifests"],
        global_test_debt=validation.get("global_test_debt", {}),
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Administrative custody is separate from scientific evidence; immutable bytes and exact operands establish readiness, never benefit.",
    )
    value["raw_shard_hashes"] += [
        dict(path=r[s + "_path"], sha256=r[s + "_sha256"])
        for r in validation["receipts"]
        for s in ["stdout", "stderr"]
        if s + "_path" in r
    ]
    value["raw_shard_hashes"] += [
        validation["coverage"][k]
        for k in ["report_reference", "data_reference"]
        if k in validation["coverage"]
    ]
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute from frozen raw evidence and reject both byte and semantic drift."""
    try:
        value = json.loads(path.read_bytes())
        raw = value["raw_shard_hashes"]
        if any(
            reference(Path(r["path"]))["sha256"] != r["sha256"]
            for r in raw + value["code_config_hashes"] + value["source_artifact_hashes"]
        ):
            raise ValueError("operand_drift")
        work = json.loads(Path(raw[0]["path"]).read_bytes())
        if work["contract"] != authority(Path(work["raw"]) / "authority"):
            raise ValueError("frozen_authority_drift")
        for task, ref in zip(work["contract"]["tasks"], work["manifests"], strict=True):
            manifest = json.loads(Path(ref["path"]).read_bytes())
            if manifest != dict(
                task=task,
                canonical_tasks_sha256=PIN,
                gates=task["gated_on"],
                upstream_task_ids=[g["upstream"] for g in task["gated_on"]],
                administrative_root_required=False,
            ):
                raise ValueError("branch_manifest_drift")
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
        expected = build(
            dict(work, raw_reference=raw[0]),
            validation,
        )
        if validation_path.is_file():
            expected["raw_shard_hashes"].append(reference(validation_path))
        expected["reproducibility_checksum"] = checksum(expected)
        passed = (
            value == expected
            and canonical_hash(value["full_task_objects"]) == "sha256:" + PIN
            and value["reproducibility_checksum"] == checksum(value)
        )
        return dict(passed=passed, reason="replay_passed" if passed else "semantic_drift")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
