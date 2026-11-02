"""REQ-REPORT-8359: bounded readers keep sealed history outside the parent.

Each child reduces one producer. Imported inference remains historical, and
cached row diagnostics cannot repair a failed producer's execution qualification.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v719_capstone_evidence as old
from carnot.reporting import v720_frozen_input_contract as contract
from carnot.reporting.v720_terminal_replay import gate
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v709_execution import execute
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT, DESIGN, STAGED, ACTIVE, PROTOCOL = (
    old.ROOT,
    old.DESIGN,
    old.STAGED,
    old.ACTIVE,
    old.PROTOCOL,
)
NAME, TASK, MILESTONE = "experiment_8359_v720_capstone", "exp8359-capstone", "2026.10.720"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v720_capstone_8359.py"
OWNED = [
    "python/carnot/reporting/v720_capstone_evidence.py",
    "python/carnot/reporting/v720_capstone_reduction.py",
    "python/carnot/reporting/v720_capstone.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
ADAPTER_PINS = {
    "v719_capstone_evidence": "6446dbff27e2153d16dbfc5f3bd187ba8ceaa3946db10af2be5be0352152d0f7",
    "v719_capstone_reduction": "d56fb050dc7d8cc731f3debb8f17fa8f6184818cbe599ca1c37af48725344b2d",
    "v720_replay_closure": "5141070191f0edf948dc0b1ff762fe5311fbdad9cdce85151f9eac7344d2148d",
    "v720_terminal_replay": "6b5978c7e452c46cdd97e8aa6b15e710d91f6bbe2c30db7579d49dc6bdd9e294",
    "v720_replay_execution": "7cfd86ace96ec34441c5328a8f7674eb37e00447984e714d6be558d47629c9c1",
}
PROTOCOL_PIN, failure, read, memory = old.PROTOCOL_PIN, old.failure, old.read, old.memory
_operand, _compact, _worker, _authority, _build, _replay = (
    old.operand,
    old.compact,
    old.worker,
    old.authority,
    old.build,
    old.replay,
)
SELECT = [
    "frozen_heads_ready_score",
    "frozen_predictions_ready_score",
    "trajectory_ready_score",
    "capacity_ready_score",
    "maximum_pending_count",
    "feedback_lost",
    "feedback_retained",
    "constructed_utility_rows",
    "table_ready_score",
    "table_fidelity_ready_score",
    "table_rows",
    "refresh_rows",
    "resolution_rows",
    "table_timing_rows",
    "timing_rows",
    "operation_rows",
    "cpu_cost_scope",
    "board_obligations",
    "generalization_frontier",
    "support_frontier",
    "arm_support_rows",
    "support_status",
    "arc_outcome_support_score",
    "current_frontier",
    "table_bytes",
    "board_rows",
    "claim_scope",
    "branch_replay_ready_score",
    "historical_dispositions",
    "parent_child_memory_rows",
    "current_device_execution_count",
    "current_jtag_retry_count",
    "current_model_calls",
    "update_budget_control",
    "reachable_action_rows",
    "learning_scope",
    "full_h1_measured",
    "science_disposition",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real flushed boundaries make unfinished work visible to the conductor."""
    print(f"[exp8359] phase={phase} completed={completed} pending={pending}", flush=True)


def compact_provenance(value: Any) -> Any:
    """Keep imported call counts while sealed source bytes retain large device logs."""
    if isinstance(value, dict):
        return {k: compact_provenance(v) for k, v in value.items() if k != "model_receipt"}
    if isinstance(value, list):
        return [compact_provenance(v) for v in value]
    return value


def operand(task: Json, item: Json) -> Json:
    """Reuse the bounded reader; recompute science from authenticated primitive rows."""
    before = memory()
    result = dict(_operand(task, item))
    if result["row"]["producer_executed"]:
        source = read(item["reference"])
        result["selected"].update({k: source[k] for k in SELECT if k in source})
        provenance = source.get("historical_model_provenance", [])
        result["selected"]["historical_model_provenance"] = compact_provenance(provenance)
        for field in ["timing_rows", "table_timing_rows"]:
            if field in source:
                result["selected"][field] = [
                    dict(
                        {
                            k: v
                            for k, v in row.items()
                            if k
                            not in {
                                "probabilities",
                                "actions",
                                "coefficients",
                                "head",
                                "table_hex",
                                "affected_entries",
                                "grid_points",
                            }
                        },
                        payload_reference=item["reference"],
                        payload_field=field,
                        payload_row=index,
                        payload_sha256=canonical_hash(row),
                    )
                    for index, row in enumerate(source[field])
                ]
        identity = result["row"]["experiment_id"]
        if identity in [8350, 8351]:
            from carnot.verify import static_benefit_audit_8350 as h1
            from carnot.verify import learning_retention_audit_8351 as h2

            primitive = source["measurement_reference"]
            require_reference(primitive)
            work = json.loads(Path(primitive.get("snapshot_path", primitive["path"])).read_bytes())
            for ref in work["raw_refs"] + work["refs"]:
                require_reference(ref)
            progress("before_benchmark_independent_science_reduction")
            reduced = (
                h1.reduce(
                    work["predictions"], work["targets"], work["comparator"], work["optimizer"]
                )
                if identity == 8350
                else h2.reduce(work["state"], work["targets"], work["checks"])
            )
            result["selected"]["independent_reduction"] = reduced
            result["selected"]["science_controls"] = work.get("optimizer", work.get("checks"))
            controls = result["selected"]["science_controls"]
            if "later_source_attribution" in controls:
                controls["later_source_attribution"] = dict(
                    reference=primitive,
                    field="checks.later_source_attribution",
                    sha256=canonical_hash(controls["later_source_attribution"]),
                )
            result["selected"]["sealed_before_labels"] = work["label_access_log"]
            result["primitive_references"].extend(work["raw_refs"])
            progress("after_benchmark_independent_science_reduction", 1, 0)
    after = memory()
    result["memory"] = dict(
        before=before,
        after=after,
        growth_mb=after["peak_rss_mb"] - before["peak_rss_mb"],
        passed=after["peak_rss_mb"] - before["peak_rss_mb"] <= 500,
    )
    return result


def bindings() -> ExitStack:
    """Scope legacy adapters to current identity without mutating old artifacts."""
    stack = ExitStack()
    for key, value in dict(
        CLI=CLI, OWNED=OWNED, TEST=TEST, contract=contract, operand=operand, progress=progress
    ).items():
        stack.enter_context(patch.object(old, key, value))
    return stack


def adapter_checks() -> list[Json]:
    """Producer-captured code pins authorize adapters, independently of their outcomes."""
    return [
        gate(
            str(ROOT / f"python/carnot/reporting/{name}.py"),
            "adapter_source_sha256",
            "sha256:" + pin,
            sha256_file(ROOT / f"python/carnot/reporting/{name}.py"),
        )
        for name, pin in ADAPTER_PINS.items()
    ]


def worker(request: Path, output: Path) -> int:
    """An isolated source reader measures its own RSS under the unchanged guard."""
    with bindings():
        return int(_worker(request, output))


def compact(task: Json, item: Json, raw: Path, name: str) -> tuple[Json, Json | None]:
    """Both measurement and cold replay use the same bounded child entrypoint."""
    with bindings():
        result, receipt = _compact(task, item, raw, name)
        return dict(result), receipt


def authority(work: Json) -> Json:
    """Compare visible order, all task fields and digest using saved authority bytes."""
    with bindings():
        return dict(_authority(work))


def measure(root: Path, raw: Path) -> Json:
    """Freeze declared inputs before current replay; future tasks never become fixtures."""
    progress("measurement_before", 0, 14)
    start, wall, before = time.monotonic_ns(), time.time_ns(), memory()
    from carnot.reporting import v720_replay_execution as validation

    with (
        TemporaryDirectory(prefix="exp8359-resources-") as directory,
        patch.object(validation, "e", __import__(__name__, fromlist=["gate"])),
    ):
        checks = validation.preflight(Path(directory)) + adapter_checks()
    refs = [
        snapshot(root / name, raw / "custody", str(i))
        for i, name in enumerate([DESIGN, STAGED, ACTIVE, PROTOCOL])
    ]
    tasks = parse_design(Path(refs[0]["snapshot_path"]).read_text(), milestone=MILESTONE)[1]
    if [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8346, 8360)]:
        raise ValueError("exact_fourteen_task_contract")
    inputs, audits, plan = [], [], []
    for index, task in enumerate(tasks[:-1]):
        progress("input_before", index, 13 - index)
        declared = root / task["deliverable"]
        path = old.reader.prior.resolve(task, 8346 + index, root)
        if path != declared:
            refs.append(snapshot(declared, raw / "custody", str(len(refs))))
        item = old.reader.bind(path, raw, refs)
        summary, receipt = compact(task, item, raw / "operands", f"input_{8346 + index}")
        item.update(summary=summary, summary_sha256=canonical_hash(summary))
        inputs.append(item)
        if receipt is not None:
            audits.append(receipt)
        for gate in summary["conductor_gates"]:
            refs.append(
                dict(
                    snapshot(Path(gate["artifact_path"]), raw / "custody", str(len(refs))),
                    expected_sha256=gate["artifact_sha256"],
                )
            )
        for ref in summary["primitive_references"]:
            refs.append(
                dict(
                    snapshot(Path(ref["path"]), raw / "custody", str(len(refs))),
                    expected_sha256=ref["sha256"],
                )
            )
        script = ROOT / "scripts/experiments" / (Path(task["deliverable"]).stem + ".py")
        if summary["row"]["producer_executed"] and script.is_file():
            plan.append(
                dict(
                    name=f"branch_{8346 + index}_replay",
                    argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(script),
                        "--cold-replay",
                        item["reference"]["snapshot_path"],
                    ],
                    expected=0,
                    deadline=180,
                    scope="owned",
                )
            )
        progress("input_after", index + 1, 12 - index)
    history = {}
    for prior in {p["experiment_id"] for t in tasks for p in t["prior_failures"]}:
        identity = prior.split("-")[0][3:]
        paths = sorted((root / "results").glob(f"experiment_{identity}_*.json"))
        if paths:
            history[prior] = old.reader.bind(paths[0], raw, refs)
    fixtures = [
        snapshot(root / name, raw / "custody", "historical-" + str(i))
        for i, name in enumerate(
            [
                "results/experiment_8317_v717_capstone.json",
                "results/experiment_8331_v718_capstone.json",
                "results/experiment_8345_v719_capstone.json",
                "tests/python/test_v718_capstone_8331.py",
                "python/carnot/testing/pytest_memory_watchdog.py",
                "ops/exclusion_manifest.yaml",
            ]
        )
    ]
    refs.extend(fixtures)
    health_path = (
        root / "results/raw" / NAME / "development_attempts/quota/repository_suite_attempt.json"
    )
    health: Json | None = None
    if health_path.is_file():
        ref = snapshot(health_path, raw / "custody", "repository-health")
        refs.append(ref)
        health = read(ref)
        for stream in ["stdout", "stderr"]:
            saved = snapshot(Path(health[stream + "_path"]), raw / "custody", "health-" + stream)
            refs.append(dict(saved, expected_sha256=health[stream + "_sha256"]))
    publication = dict(
        name="publication_gate",
        argv=[str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"],
        expected=0,
        deadline=90,
        scope="publication",
    )
    atomic_json(raw / "branch_manifest.json", dict(commands=plan, publication=publication))
    audits.extend(execute(plan, raw / "branches"))
    gate = execute([publication], raw / "publication")[0]
    after = memory()
    work = dict(
        root=str(root),
        tasks=tasks,
        inputs=inputs,
        history=history,
        references=refs,
        failures=[c for c in checks if not c["passed"]],
        preconditions=checks,
        repository_health_attempt=health,
        audits=audits,
        publication=gate,
        historical_fixture_hashes=fixtures,
        started_monotonic_ns=start,
        started_wall_ns=wall,
        ended_monotonic_ns=time.monotonic_ns(),
        memory_measurements=dict(
            parent_before=before,
            parent_after=after,
            parent_growth_mb=after["peak_rss_mb"] - before["peak_rss_mb"],
            workers=[i["summary"]["memory"] for i in inputs],
            retained_payload_bytes=len(json.dumps(inputs).encode()),
        ),
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_after", 13, 1)
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Current reduction adds scientific diagnostics without changing source verdicts."""
    from carnot.reporting.v720_capstone_reduction import reduce as reduction

    return dict(reduction(work, receipts))


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reuse byte-bound artifact normalization with only current owned code coverage."""
    if "code_refs" not in work:
        reused = [
            "v719_capstone_evidence",
            "v719_capstone_reduction",
            "v718_capstone_evidence",
            "v718_capstone_reduction",
            "v717_capstone_evidence",
            "v720_replay_execution",
            "v720_terminal_replay",
            "v720_replay_closure",
            "v720_frozen_input_contract",
            "primary_publication",
            "roadmap_contract",
            "v709_execution",
            "v718_replay_history",
        ]
        paths = [
            *OWNED,
            TEST,
            *[f"python/carnot/reporting/{name}.py" for name in reused],
            "python/carnot/verify/static_benefit_audit_8350.py",
            "python/carnot/verify/learning_retention_audit_8351.py",
            "python/carnot/testing/pytest_memory_watchdog.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/publication_gate.py",
        ]
        work["code_refs"] = [snapshot(ROOT / p, raw / "code", str(i)) for i, p in enumerate(paths)]
    with bindings(), patch.object(old, "reduce", reduce):
        value = dict(_build(work, receipts, raw, output))
    value.update(
        experiment_id=8359,
        task_id=TASK,
        milestone=MILESTONE,
        schema="carnot.v720.capstone.v1",
        random_seed=7208359,
    )
    value["adversarial_findings"] = work.get("adversarial_findings", [])
    value["repository_health_attempt"] = work.get("repository_health_attempt")
    value["field_principles"]["repository_health_attempt"] = (
        "Keep the single outside-experiment repository-suite failure separate from frozen owned checks."
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Fresh reconstruction rechecks source summaries rather than trusting their hashes."""
    with bindings(), patch.object(old, "build", build), patch.object(old, "compact", compact):
        return bool(_replay(path))
