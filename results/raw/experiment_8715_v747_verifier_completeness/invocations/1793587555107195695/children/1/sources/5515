"""REQ-REPORT-8525: exact producer seals keep failed studies and absent evidence visible."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml
import jiter

from carnot.reporting import v731_capstone_evidence as old
from carnot.reporting import v732_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK, MILESTONE = "experiment_8525_v732_capstone", "exp8525-capstone", "2026.10.732"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v732_capstone_8525.py"
OWNED = [f"python/carnot/reporting/v732_capstone{s}.py" for s in ["_evidence", "_report", ""]] + [
    CLI
]
MODEL_SPECS: list[Json] = []
PINS = dict(
    zip(
        [8512, 8513, 8514, 8516, 8517, 8518, 8519, 8520, 8521, 8522, 8523, 8524],
        [
            "d8aad6fcd8662fe8abc98acaa95add49df44b3b5359700ab09bbaafee3114eec",
            "2a843e589453122f245c3eec366dc6407381d134a34787c1c176e832c1ed98e4",
            "d80d23a883dc0ec2f54af8e4621023cefba11060e427c3573a0c24fdb2099762",
            "2aa6420f606e6eea82afa9ee8d1bd62e149efd2b608df3ecae1a1f68a82b69df",
            "cf39595a1d7ee1b5dff2212a0b51014eb46b079499cbe5c194d43d9de6def1fd",
            "113e1efcf501dfcd7a7777cfd0a8a52db768e14916044c603e5d5a4906ca172a",
            "6e4233edc4865de8ada5a6775429ed04ac379306e5a25f7b4dcfad002f46a904",
            "ae2d903338bd15168fa05ebc5cf8d529888c316e11bfebb36269b5fd852f691f",
            "9604b4efc1c65b7b329f6ffa3a9f32a5f878f03573389548c01c5415872f0a6f",
            "4a82097fa0faf69b8d5d3b8da81e5fd9b2acb3bc92ff722f56095cacd9e7356d",
            "26d4935b96524fa0ce12af38f8bfae4fb823e848f85bb0afbc68ce97881e73fc",
            "84d2393d711e1b8040de2e7265af40efcea18515b1a91e6e37a0794169fa7ee1",
        ],
        strict=True,
    )
)
PREGATE_PATHS = {
    n: f"results/experiment_{n}_{s}.json"
    for n, s in [
        (8514, "identity_brier_fit"),
        (8518, "objective_benefit_audit"),
        (8521, "bounded_qwen_evidence"),
    ]
}
READERS = dict(
    zip(
        [8512, 8513, 8516, 8517, 8519, 8520, 8522, 8523, 8524],
        [
            "reporting.v732_contract_methods",
            "reporting.v732_strongest_baseline_audit",
            "reporting.objective_publication_8516",
            "reporting.brier_online_learning_8517",
            "reporting.append_import_qualification_8519",
            "reporting.append_service_cost_8520",
            "reporting.arc_effect_qualification_8522",
            "reporting.arc_effect_panel_8523",
            "reporting.board_cost_boundary_8524",
        ],
        strict=True,
    )
)
checksum, reference, freeze, load, slot, memory = (
    old.checksum,
    old.reference,
    old.freeze,
    old.load,
    old.slot,
    old.memory,
)
SELECT = old.SELECT | {
    "redirect_effect_rows",
    "shadow_rows",
    "action_divergence_rows",
    "arm_refinement_authorized",
    "complete_cost_shares",
    "append_cost_boundary",
    "primary_p95_ratio",
    "primary_speedup_ci95",
    "speedup_ci95",
    "gain_vs_frozen",
    "conditional_ci95",
    "per_window_retention",
    "retention_simultaneous_upper95",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real counts show bounded work without extending the task's duration."""
    print(f"[exp8525] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(ref: Json, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Missing evidence differs from a measured zero and names its exact operand."""
    return dict(old.gate(ref, field, expected, observed, upstream))


def native_json(path: Path, digest: str) -> Json:
    """Parse authenticated bytes directly, avoiding a second large Unicode buffer."""
    if sha256_file(path) != digest:
        raise ValueError("operand_hash")
    return dict(jiter.from_json(path.read_bytes()))


def outcome(task: Json, primary: Json, raw: Path, closure: Any = None) -> Json:
    """Shipped primitive replay qualifies evidence independently of a producer's benefit."""
    number = int(task["id"][3:7])
    if number < 8512:
        result = dict(old.outcome(task, primary, raw, closure))
        if number == 8488:
            value = load(primary)
            reader = importlib.import_module("carnot.reporting.v730_source_information_audit")
            work = load(value["primitive_reference"])
            metrics = reader.n.evaluate(
                load(work["prediction_reference"])["rows"], load(work["primitive_files"][1])["rows"]
            )
            result["selected"]["historical_three_source_contrasts"] = [
                dict(
                    control=arm,
                    brier_gain=metrics["A1_metrics"][arm]["brier"]
                    - metrics["A1_metrics"]["full_spline9"]["brier"],
                    typed_cost_delta=metrics["A1_metrics"]["full_spline9"]["decision_loss"]
                    - metrics["A1_metrics"][arm]["decision_loss"],
                    raw_contrast_descriptive_only=arm == "raw_p2",
                    exposure="exposed_history",
                )
                for arm in ["raw_p2", "scalar_cubic_spline", "source_permuted_spline9"]
            ]
        return result
    value = load(primary)
    if primary["sha256"] != "sha256:" + PINS[number]:
        raise ValueError("primary_pin")
    if value.get("blocked_at_layer") == "conductor_pre_gate":
        with patch.dict(old.old.base.PINS, PINS):
            result = dict(old.old.base.outcome(task, primary, raw, closure))
    else:
        old.old.base.validate_primary(value, Path(primary["path"]))
        if value["task_id"] != task["id"]:
            raise ValueError("task_identity")
        terminal = load(reference(Path(value["terminal_validation_sidecar_path"])))
        publication = terminal.get("publication", terminal)
        side = load(reference(Path(publication["sidecar_path"])))
        sealed = (
            publication["primary_sha256"] == primary["sha256"] == side["primary_sha256"]
            and side["report"]["passed"] is True
        )
        reader = importlib.import_module("carnot." + READERS[number])
        from contextlib import nullcontext

        with patch.object(reader, "checked", native_json) if number == 8523 else nullcontext():
            replayed = bool(
                reader.replay(value if number in {8522, 8523} else Path(primary["snapshot_path"]))
            )
        qualified = bool(
            sealed
            and replayed
            and value["required_checks_passed"]
            and not value["flagged_adversarial"]
            and value["verdict_class"] != "disqualified"
        )
        result = dict(
            producer_executed=True,
            eligible=qualified,
            selected={k: v for k, v in value.items() if k in SELECT or k.endswith("_ready_score")},
            checks=[]
            if qualified
            else [
                gate(
                    primary,
                    "producer_terminal_replay",
                    True,
                    dict(sealed=sealed, replayed=replayed),
                    task["id"],
                )
            ],
            disposition="qualified_producer" if qualified else "disqualified_producer",
            honest_verdict=value["honest_verdict"],
            verdict_class=value["verdict_class"],
            source_gates=value.get("gate_check_summary", []),
            historical_model_provenance=value.get("historical_model_provenance", []),
            upstream_adversarial_findings=value.get("adversarial_findings", []),
            primitive_reference=value.get("primitive_reference", value.get("work_reference")),
            branch_replay=dict(
                status="replayed" if qualified else "disqualified_upstream_evidence",
                passed=replayed,
            ),
        )
        if number == 8517:
            from carnot.verify import capacity_benefit_audit_8435 as numeric

            reduced = reader.reductions(load(value["primitive_reference"]))
            with patch.object(numeric, "ARMS", reader.ARMS):
                result["selected"]["unqualified_later_reduction"] = dict(
                    qualification=False,
                    scope="descriptive exposed-history; Exp8518 remains unmeasured",
                    later=numeric.conditional_bootstrap(
                        numeric.question_contrasts(reduced["rows"], 8)
                    ),
                    retention=numeric.retention_bound(reduced["retention_rows"]),
                    seed_losses=reduced["seed_losses"],
                    initial_head="original_V726",
                )
    result.update(
        original_run_date=value.get("run_date"),
        upstream_flagged_adversarial=value.get("flagged_adversarial"),
        original_required_checks_passed=value.get("required_checks_passed"),
    )
    return result


def worker_process(request: Path, output: Path) -> int:
    """Imports precede the replay baseline; total peak and growth still have hard limits."""
    with patch.multiple(
        old.old, READERS=old.old.READERS | old.READERS | READERS, outcome=outcome, progress=progress
    ):
        return int(old.old.worker_process(request, output))


def invoke(task: Json, primary: Json, raw: Path) -> tuple[Json, Json]:
    """Each branch starts a fresh supervised worker and retains both hashed log streams."""
    with patch.multiple(old.old.workers, CLI=CLI, progress=progress):
        result, receipt = old.old.workers.invoke(task, primary, raw)
    source = load(primary)
    result.update(
        original_run_date=source.get("run_date"),
        upstream_flagged_adversarial=source.get("flagged_adversarial"),
        original_required_checks_passed=source.get("required_checks_passed"),
    )
    atomic_json(raw / "summary.json", result)
    return dict(result), dict(receipt)


def historical_refs() -> list[Json]:
    """Original sealed records preserve history even when the current roadmap is complete."""
    protocol = methods.protocol()
    return [
        protocol["v731_capstone_reference"],
        protocol["v730_fact_references"][-1],
        protocol["preserved_v731_design_reference"],
    ]


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate before reading each independent CPU branch; optional devices cannot gate it."""
    start, before = time.monotonic_ns(), memory()
    progress("preconditions", 0, 14)
    authority = methods.authority(root, raw / "authority")
    checks = list(authority["gate_check_summary"])
    resources = dict(
        old.old.history.previous.resources(private),
        task_cap_s=4800,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        optional_cuda_required=False,
        optional_boards_required=False,
    )
    if not (
        resources["disk_backed"]
        and resources["memory_available_bytes"] >= 1500 * 1024**2
        and resources["free_disk_bytes"] >= 1024**3
        and resources["tools_available"]
    ):
        checks.append(gate(dict(path=str(private)), "required_resources", True, resources))
    tasks = authority["tasks"] or methods.protocol()["task_map_v732"]
    if (
        methods.tasks_digest(tasks) != methods.TASKS_PIN
        or tasks[-1].get("requires")
        or tasks[-1].get("gated_on")
    ):
        raise ValueError("exact_unconditional_fourteen_tasks")
    protocol = methods.protocol()
    log = freeze(root / "ops/conductor-log.md", raw / "scheduling")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    activation = next(
        (i for i, line in enumerate(lines) if "Milestone 2026.10.732 activated" in line), len(lines)
    )
    lines = lines[activation:]
    refs: list[Json] = [log]
    historical_tasks = []
    for family, inputs in dict(
        protocol["historical_authorities"], V730=protocol["v730_authority_references"]
    ).items():
        sealed = [freeze(Path(r["path"]), raw / "authorities" / family) for r in inputs]
        refs.extend(sealed)
        if any(
            r["sha256"] != original["sha256"] for r, original in zip(sealed, inputs, strict=True)
        ):
            checks.append(
                gate(
                    sealed[1],
                    "original_authority_pin",
                    inputs[1]["sha256"],
                    sealed[1]["sha256"],
                    family,
                )
            )
            continue
        historical_tasks += [
            t
            for t in yaml.safe_load(Path(sealed[1]["snapshot_path"]).read_bytes())["tasks"]
            if int(t["id"][3:7]) in [8435, 8437, 8488, 8489, 8494]
            and (root / t["deliverable"]).is_file()
        ]
    slots: list[Json] = []
    historical: list[Json] = []
    receipts: list[Json] = []
    roster = [*tasks[:-1], *historical_tasks]
    for index, task in enumerate(roster):
        progress("branch_before_" + task["id"], index, len(roster) - index)
        if (time.monotonic_ns() - start) / 1e9 >= 3600:
            raise TimeoutError("measurement_deadline")
        number = int(task["id"][3:7])
        primary = freeze(root / task["deliverable"], raw / "primaries" / str(number))
        refs.append(primary)
        if not primary["exists"] and number in PREGATE_PATHS:
            receipt_path = root / PREGATE_PATHS[number]
            if receipt_path.is_file():
                primary = freeze(receipt_path, raw / "pregates" / str(number))
                refs.append(primary)
        summary = None
        if primary["exists"]:
            summary, receipt = invoke(task, primary, raw / "branches" / task["id"])
            receipts.append(receipt)
            refs.append(freeze(raw / "branches" / task["id"] / "summary.json", raw / "summaries"))
        cascade = next(
            (
                line
                for line in lines
                if "GATE_BLOCK" in line and line.split("|")[2].strip() == task["title"][:50].strip()
            ),
            None,
        )
        row = slot(task, primary, cascade, summary)
        (slots if number >= 8512 else historical).append(row)
        checks.extend(row["checks"])
        progress("branch_after_" + task["id"], index + 1, len(roster) - index - 1)
    refs.extend(
        freeze(ROOT / name, raw / "code")
        for name in [
            *OWNED,
            TEST,
            "python/carnot/reporting/primary_publication.py",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "ops/exclusion_manifest.yaml",
            methods.PROTOCOL,
            "docs/research-notes/v732-capstone.md",
        ]
    )
    originals = historical_refs()
    predecessors = [freeze(Path(r["path"]), raw / "predecessors") for r in originals[:2]]
    refs.append(freeze(Path(originals[2]["source_path"]), raw / "preserved_design"))
    after = memory()
    work = dict(
        root=str(root),
        raw=str(raw),
        tasks=tasks,
        authority=authority,
        contract_digest=methods.TASKS_PIN,
        slots=slots,
        historical_slots=historical,
        refs=refs,
        predecessors=predecessors,
        branch_receipts=receipts,
        checks=checks,
        preconditions=resources,
        invocation_argv=list(__import__("sys").argv),
        memory=dict(
            before=before,
            after=after,
            peak_bound_mb=1500,
            growth_bound_mb=500,
            passed=after["peak_rss_mb"] <= 1500
            and after["current_rss_mb"] - before["current_rss_mb"] <= 500,
        ),
        phase_spans=[
            dict(
                name="sealed_branch_reduction",
                started_monotonic_ns=start,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        ],
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        adversarial_findings=[],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def replay(path: Path) -> bool:
    """Reconstruct authority and every primitive branch before comparing the complete report."""
    from carnot.reporting.v732_capstone_report import build

    try:
        value = json.loads(path.read_bytes())
        work = load(value["primitive_reference"])
        if (
            value["reproducibility_checksum"] != checksum(value)
            or value["task_count"] != 14
            or methods.tasks_digest(work["tasks"]) != methods.TASKS_PIN
            or work["contract_digest"] != methods.TASKS_PIN
        ):
            return False
        if [r["sha256"] for r in work["predecessors"]] != [
            r["sha256"] for r in historical_refs()[:2]
        ]:
            return False
        for ref in [*work["refs"], *work["predecessors"], *work.get("coverage_shards", [])]:
            if (
                ref.get("exists", True)
                and sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]
            ):
                return False
        with TemporaryDirectory(prefix="exp8525-authority-", dir="/var/tmp") as directory:
            cold = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                ref = work["authority"]["authority_snapshots"][role]
                if ref["exists"]:
                    if sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]:
                        return False
                    target = cold / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ref["snapshot_path"], target)
            checked = methods.authority(cold, cold / "checked")
            if (
                checked["activated"] != work["authority"]["activated"]
                or checked["tasks"] != work["authority"]["tasks"]
            ):
                return False
        for row in [*work["slots"], *work["historical_slots"]]:
            if (
                row["experiment_id"] >= 8512
                and row["task"] != work["tasks"][row["experiment_id"] - 8512]
            ):
                return False
            if row["primary_reference"]["exists"]:
                with TemporaryDirectory(prefix="exp8525-cold-", dir="/var/tmp") as directory:
                    actual, receipt = invoke(row["task"], row["primary_reference"], Path(directory))
                if not receipt["passed"] or any(
                    actual[k] != row[k]
                    for k in actual
                    if k not in {"memory", "reader_initialization"}
                ):
                    return False
            elif row != slot(row["task"], row["primary_reference"], row["cascade_log_line"]):
                return False
        for receipt in [*work["branch_receipts"], *value["validation_receipts"]]:
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    return False
        return bool(
            value
            == build(
                work,
                value["validation_receipts"],
                Path(work["raw"]),
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
