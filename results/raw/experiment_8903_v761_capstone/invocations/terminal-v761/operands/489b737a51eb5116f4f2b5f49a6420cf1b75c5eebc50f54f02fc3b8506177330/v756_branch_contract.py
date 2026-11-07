"""REQ-REPORT-8827: bind administration without imposing a science prerequisite."""

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v755_branch_contract as old
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import compare_contract, parse_design

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8827_v756_branch_contract",
    "exp8827-branch-contract",
    "2026.11.756",
)
DESIGN, ACTIVE, STAGED = old.DESIGN, old.ACTIVE, old.STAGED
PIN = "9702757c3b1e6747d2341fb22ef435c3171cf1bfb2413aa1f947fe0eecc5ce9b"
ARCHIVE = "openspec/change-proposals/research-roadmap-v755-preserved-20261104.md"
HISTORY = "results/experiment_8826_v755_capstone.json"
HISTORY_PINS = {
    ARCHIVE: "90a2b04b863a04a452035e88efb9c5634297f8777ed8469f8852a3649895a3d8",
    HISTORY: "9e023a8a121da2fc9622c166eeb94342b528250d8ab39aa8cacca190f05c369a",
}
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/v756_branch_{s}.py" for s in ["contract", "qualification", "runner"]
] + [CLI]
TEST = "tests/python/test_v756_branch_contract_8827.py"
reference, checksum, edges = old.reference, old.checksum, old.edges
RUNTIME_VALIDATORS = [
    "python/carnot/reporting/v725_contract_methods.py",
    "python/carnot/reporting/v752_branch_methods.py",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the parent can distinguish work from a stalled child."""
    print(f"[exp8827] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path) -> Json:
    """Compare complete objects against a fixed pin, retaining staging explicitly."""
    try:
        text = (root / DESIGN).read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.load((root / ACTIVE).read_bytes(), Loader=yaml.CSafeLoader)
        staged = (
            yaml.load((root / STAGED).read_bytes(), Loader=yaml.CSafeLoader)
            if (root / STAGED).is_file()
            else deepcopy(active)
        )
        state = "active_v756"
        if active["milestone"] == old.MILESTONE and (root / STAGED).is_file():
            active, state = deepcopy(staged), "private_staged_v756"
        checked = compare_contract(
            text, staged, active, milestone=MILESTONE, first_id=8827, count=13
        )
        for i, row in enumerate(checked["rows"]):
            matched = row["matched"] and tasks[i] == active["tasks"][i] == staged["tasks"][i]
            row.update(
                matched=matched,
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
            and "<!-- V756_TASK_CONTRACT_END -->" in text
        )
        return dict(ready=bool(ready), rows=checked["rows"], tasks=tasks, table=table, state=state)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return dict(ready=False, rows=[], tasks=[], table=[], state="rejected")


def history(root: Path) -> Json:
    """Retain actual primary and skip verdicts, including the capstone itself."""
    for name, pin in HISTORY_PINS.items():
        if sha256_file(root / name) != "sha256:" + pin:
            raise ValueError("historical_authority_drift:" + name)
    cap = json.loads((root / HISTORY).read_bytes())
    if (
        parse_design((root / ARCHIVE).read_text(), milestone=old.MILESTONE)[1]
        != cap["full_task_objects"]
    ):
        raise ValueError("historical_task_drift")
    rows, refs = [], [reference(root / p) for p in HISTORY_PINS]
    for slot in cap["rows"]:
        primary = slot.get("primary_reference")
        primary = primary if primary and primary.get("sha256") else None
        sources = (
            [primary] if primary else [r["reference"] for r in slot.get("scheduler_receipts", [])]
        )
        if slot["experiment_id"] == 8826:
            sources = [reference(root / HISTORY)]
        for ref in sources:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("historical_receipt_drift")
        if not sources:
            sources = [reference(root / HISTORY)]
        rows.append(
            dict(
                task_id=slot["task_id"],
                kind="primary" if primary or slot["experiment_id"] == 8826 else "structured_skip",
                verdict=slot["verdict_class"],
                honest_verdict=slot["primary_honest_verdict"],
                references=sources,
                conductor_lines=slot.get("conductor_lines", []),
                scheduler_receipts=slot.get("scheduler_receipts", []),
            )
        )
        refs += sources
    return dict(
        rows=rows,
        references=refs,
        primary_count=sum(r["kind"] == "primary" for r in rows),
        structured_skip_count=sum(r["kind"] == "structured_skip" for r in rows),
        original_child_qualified=False,
    )


def methods(root: Path) -> list[Json]:
    """Bind versioned primary methods and local assumptions without importing results."""
    data = [
        (
            "2610.09541v1",
            "ed041c42a0a0d8fba700ab618f3f1a26fa38666a26d165c865cc89c5a6678475",
            "Certified by Abstention",
            "Use one independent world per calibration unit; report certificate firing, served risk and the exact zero-error floor separately.",
            "Exchangeability and frozen thresholds are required; abstention is not zero observed conditional risk. No natural-text guarantee transfers.",
        ),
        (
            "2609.37493v1",
            "59254dfc62077d4debe5e0d823e08cf28c8768d916741537ae24e3323d6c08cb",
            "PriceCheck",
            "Fit completeness, leak and complete check costs on development worlds; freeze a small ordered menu before independent risk calibration.",
            "Correlated deterministic checks do not inherit independent vote composition; charge all checks and stopping overhead. No model result transfers.",
        ),
    ]
    rows = []
    for version, pin, title, adaptation, assumptions in data:
        path = root / "results/raw" / NAME / "literature" / (version + ".html")
        if sha256_file(path) != "sha256:" + pin:
            raise ValueError("primary_method_drift")
        rows.append(
            dict(
                title=title,
                version=version,
                url="https://arxiv.org/html/" + version,
                source_reference=reference(path),
                sections_read="Full method, certification, dependence and cost appendices",
                local_adaptation=adaptation,
                assumptions=assumptions,
                ingested=True,
                scientific_result_inherited=False,
            )
        )
    return rows


def arc_support(rows: list[Json]) -> Json:
    """Require shared games and every arm cell; seeds cannot create new games."""
    cells = Counter((r["game"], r["arm"]) for r in rows)
    arms = {arm for _, arm in cells}
    shared = sorted(g for g in {g for g, _ in cells} if all(cells[g, a] >= 5 for a in arms))
    return dict(
        ready=len(rows) >= 30 and len(shared) >= 3 and len(arms) >= 2,
        episode_count=len(rows),
        shared_games=shared,
        cells=[dict(game=g, arm=a, firings=n) for (g, a), n in sorted(cells.items())],
    )


def runtime_custody(root: Path, runtime: Json) -> Json:
    """Require exact external model pins plus the current causal validator's transcript."""
    path = root / "ops/cuda-runtime-repair-receipt.json"
    authority = runtime["authority"]
    transcript = authority.get("transcript", {})
    provenance = transcript.get("mandated_model_provenance", {})
    mandate, document = {}, {}
    try:
        document = json.loads(path.read_bytes())
        ref = document["mandated_model_contract_reference"]
        if sha256_file(Path(ref["path"])) == ref["sha256"]:
            mandate = json.loads(Path(ref["path"]).read_bytes())
    except (OSError, ValueError, KeyError, TypeError):
        pass
    model = Path(provenance.get("path", "/nonexistent"))
    build = transcript.get("cuda_build_reference", {})
    checks = dict(
        current_causal_validator=authority.get("ready", False)
        and any(
            r.get("changed") and r.get("field") in {"libraries", "environment"}
            for r in authority.get("changes", [])
        ),
        exact_mandated_identity=bool(mandate)
        and mandate.get("hf_id") == provenance.get("hf_id") == "unsloth/Qwen3.8-27B-GGUF",
        exact_model_revision=bool(mandate.get("revision"))
        and mandate.get("revision") == provenance.get("revision"),
        exact_gguf_checksum=model.is_file()
        and sha256_file(model) == provenance.get("sha256") == mandate.get("sha256"),
        exact_cuda_build=bool(mandate.get("cuda_build_sha256"))
        and Path(build.get("path", "/nonexistent")).is_file()
        and sha256_file(Path(build["path"]))
        == build.get("sha256")
        == mandate.get("cuda_build_sha256")
        == provenance.get("cuda_build_sha256"),
        actual_device_copy_offload=bool(transcript.get("actual_device"))
        and transcript.get("byte_copy_parity") is True
        and isinstance(transcript.get("offloaded_layers"), int)
        and transcript["offloaded_layers"] > 0,
        current_validator_custody=document.get("validator_code_hashes")
        == {p: sha256_file(ROOT / p) for p in RUNTIME_VALIDATORS},
    )
    failures = [
        dict(
            check=k,
            upstream="exp8410",
            artifact_field=k,
            expected=True,
            observed="missing_file"
            if not path.is_file()
            else "missing_field"
            if not mandate
            else "rejected_producer",
            **reference(path),
        )
        for k, passed in checks.items()
        if not passed
    ]
    refs = [reference(path)] if path.is_file() else []
    if mandate:
        refs += [document["mandated_model_contract_reference"]]
    return dict(
        ready=all(checks.values()), checks=checks, gate_check_summary=failures, references=refs
    )


def optional_admission(root: Path, raw: Path) -> Json:
    """Existing validators authenticate receipts without opening a device or game."""
    value = old.optional_admission(root, raw)
    custody = runtime_custody(root, value["runtime"])
    value["runtime"].update(ready=custody["ready"], current_custody=custody)
    value["runtime"]["references"] += custody["references"]
    value["failures"] += custody["gate_check_summary"]
    arc = value["arc"]
    support = arc_support(arc.get("new_outcome_refs", []))
    arc.update(ready=arc["ready"] and support["ready"], support=support)
    for row in value["failures"]:
        if row.get("observed") == 0:
            row["operand_state"] = "observed_zero"
    return value


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Check resources before measurement and retain exact readable operands privately."""
    progress("preconditions_before")
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "inputs").mkdir(exist_ok=True)
    probe = raw / "writable_probe"
    probe.write_bytes(b"private output")
    resources = dict(
        python=sys.executable,
        repository_virtualenv=Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
        private_output_writable=probe.read_bytes() == b"private output",
        free_bytes=shutil.disk_usage(raw).free,
        required_free_bytes=134217728,
        cpu_numerical_dependencies="not_used",
        no_device_probes=True,
        no_model_loads=True,
        activation_authorized=False,
    )
    start = time.monotonic_ns()
    contract = authority(root)
    sources = [root / DESIGN, root / ACTIVE] + [ROOT / p for p in HISTORY_PINS]
    sources += [ROOT / "research-references.md", ROOT / "ops/exclusion_manifest.yaml"]
    if (root / STAGED).is_file():
        sources.append(root / STAGED)
    failures, snapshots = [], []
    for source in sources:
        ref = reference(source)
        if ref["sha256"] is None:
            failures.append(
                dict(
                    check="required_operand",
                    upstream=TASK,
                    artifact_field="readable_sha256",
                    expected=True,
                    observed="missing_file",
                    **ref,
                )
            )
            continue
        target = raw / "inputs" / (ref["sha256"][7:] + "-" + source.name)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("immutable_operand_drift")
        target.write_bytes(source.read_bytes())
        snapshots.append(dict(source=ref, **reference(target)))
    prior = history(ROOT)
    method_rows = methods(ROOT)
    for ref in prior["references"] + [r["source_reference"] for r in method_rows]:
        target = raw / "inputs" / (ref["sha256"][7:] + "-" + Path(ref["path"]).name)
        target.write_bytes(Path(ref["path"]).read_bytes())
        snapshots.append(dict(source=ref, **reference(target)))
    stage = raw / "authority"
    (stage / DESIGN).parent.mkdir(parents=True, exist_ok=True)
    for name in [DESIGN, ACTIVE]:
        if (root / name).is_file():
            (stage / name).write_bytes((root / name).read_bytes())
    if (root / STAGED).is_file():
        (stage / STAGED).write_bytes((root / STAGED).read_bytes())
    elif contract["ready"]:
        (stage / STAGED).write_text(
            yaml.safe_dump(dict(milestone=MILESTONE, tasks=contract["tasks"]))
        )
    resources.update(
        staged_source_present=(root / STAGED).is_file(),
        staged_snapshot_explicit=True,
        authority_state=contract["state"],
        operands=snapshots,
    )
    progress("preconditions_after", len(snapshots))
    branches = optional_admission(root, raw)
    branches["failures"] += failures
    if not contract["ready"]:
        branches["failures"].append(
            dict(
                check="full_task_authority",
                upstream=TASK,
                artifact_field="full_task_objects",
                expected="13 exact pinned task objects",
                observed="missing_file" if failures else "rejected_producer",
                **reference(root / DESIGN),
            )
        )
    for ref in branches["runtime"]["references"] + branches["arc"]["references"]:
        source = Path(ref["path"])
        target = raw / "inputs" / (ref["sha256"][7:] + "-" + source.name)
        target.write_bytes(source.read_bytes())
        snapshots.append(dict(source=ref, **reference(target)))
    manifests = []
    for task in contract["tasks"]:
        path = raw / "branches" / (task["id"] + ".json")
        atomic_json(
            path, dict(task=task, gates=task["gated_on"], administrative_root_required=False)
        )
        manifests.append(reference(path))
    work = dict(
        contract=contract,
        history=prior,
        methods=method_rows,
        branches=branches,
        manifests=manifests,
        snapshots=snapshots,
        run_date=run_date,
        owner_pid=os.getpid(),
        start_ns=start,
        end_ns=time.monotonic_ns(),
        raw=str(raw.absolute()),
        preconditions=resources,
        operand_failures=failures,
    )
    atomic_json(raw / "work.json", work)
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Reuse current-work provenance while keeping the optional edges independent."""
    qualified = validation["original"]["complete"]
    prepared = dict(work, history=dict(work["history"], original_child_qualified=qualified))
    value = old.build(prepared, validation)
    value.pop("preserved_v754_reference")
    checked = value["required_checks_passed"] and qualified
    resources = work["preconditions"]
    ready = (
        checked
        and work["contract"]["ready"]
        and not work["operand_failures"]
        and resources["repository_virtualenv"]
        and resources["private_output_writable"]
        and resources["free_bytes"] >= resources["required_free_bytes"]
    )
    verdict = (
        "disqualified"
        if not checked or (not ready and not work["operand_failures"])
        else "null"
        if ready
        else "blocked"
    )
    value.update(
        experiment_id=8827,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7568827,
        honest_verdict=f"complete_{verdict}_v756_branch_contract",
        verdict_class=verdict,
        contract_ready_score=int(ready),
        required_checks_passed=checked,
        canonical_tasks_sha256=PIN,
        runtime_repair_ready_score=int(work["branches"]["runtime"]["ready"]),
        arc_new_outcomes_available_score=int(work["branches"]["arc"]["ready"]),
        original_coverage_rows=validation["original"]["rows"],
        original_owned_statement_coverage=validation["original"],
        preserved_v755_reference=work["history"],
        preconditions_checked=resources,
        code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
        acceptance_gates=dict(
            exact_contract=work["contract"]["ready"],
            original_child=qualified,
            owned_validation=checked,
            scientific_benefit=False,
        ),
        branch_admission_rows=[
            dict(
                branch=k,
                ready=work["branches"][k]["ready"],
                optional=k in {"runtime", "arc"},
                administrative_root_required=False,
            )
            for k in ["runtime", "arc"]
        ]
        + [
            dict(
                branch=k,
                ready=None,
                optional=False,
                admission="own_operands",
                administrative_root_required=False,
            )
            for k in ["calibrated_learning_diagnostic", "hardware_cost"]
        ],
    )
    value["raw_shard_hashes"] += validation.get("references", [])
    value["acceptance_gate_principles"] = dict(
        exact_contract="Exactly thirteen full objects and visible rows match the independently pinned canonical task hash.",
        original_child="All 279 original executable statements and fourteen recorded missing statements must be measured without exclusions; the unchanged original validator must pass.",
        owned_validation="Every new owned statement is measured and all scoped checks pass; repository health debt is separate.",
        scientific_benefit="Administrative custody and reading methods establish no scientific improvement.",
    )
    value["phase_spans"] = work.get("phase_spans", value["phase_spans"])
    validation_path = Path(work["raw"]) / "validation.json"
    if validation_path.is_file():
        value["raw_shard_hashes"].append(reference(validation_path))
    value["source_artifact_hashes"] = [
        dict(path=r["path"], sha256=r["sha256"], source=r["source"]) for r in work["snapshots"]
    ]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Exact custody and measured original validation establish administrative readiness only. Optional runtime and ARC edges never gate independent CPU science.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from immutable work and measured validation, rejecting rehashed claims."""
    try:
        from carnot.reporting.v756_branch_qualification import verify_measured

        value = json.loads(path.read_bytes())
        if value["required_checks_passed"] and not (
            verify_measured(value["owned_statement_coverage"], OWNED)
            and verify_measured(value["original_owned_statement_coverage"], old.OWNED)
        ):
            raise ValueError("unqualified_measured_statements")
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
        if work["contract"] != frozen:
            raise ValueError("authority_drift")
        for task, manifest in zip(work["contract"]["tasks"], work["manifests"], strict=True):
            if json.loads(Path(manifest["path"]).read_bytes()) != dict(
                task=task, gates=task["gated_on"], administrative_root_required=False
            ):
                raise ValueError("branch_manifest_drift")
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
        passed = value == expected and value["reproducibility_checksum"] == checksum(value)
        return dict(passed=passed, reason="replay_passed" if passed else "semantic_drift")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
