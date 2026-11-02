"""Reconcile V677 evidence without promoting missing science (REQ-REPORT-7794)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.experiment_7781_v677_contract_methods import (
    DESIGN,
    compare_contract,
    resolve_authority,
)
from carnot.experiment_7787_v677_qwen_event_confidence import cold_reduce as qwen_cold_reduce
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("results/experiment_7794_v677_capstone.json")
CLI = Path("scripts/experiments/experiment_7794_v677_capstone.py")
MODULE = Path("python/carnot/experiment_7794_v677_capstone.py")
TEST = Path("tests/python/test_experiment_7794_v677_capstone.py")
QWEN_CANDIDATE = Path("results/raw/experiment_7787_v677_qwen_event_confidence/candidate.json")
ELIGIBLE = {"null", "positive", "circular_positive"}
PRINCIPLES = {
    "experiment_id": "Each record needs a unique owner.",
    "milestone": "The output belongs to the current milestone.",
    "run_date": "The requested date identifies this run.",
    "honest_verdict": "Unchanged external inputs must not consume retries.",
    "verdict_class": "Claim strength travels with the result.",
    "flagged_adversarial": "Invalid evidence cannot open a gate.",
    "gate_check_summary": "A missing producer differs from a scientific null.",
    "rows": "A headline must be reproducible from individual units.",
    "acceptance_gate_results": "Working fixtures do not establish benefit.",
    "duration_s": "Real elapsed work determines substrate authenticity.",
    "phase_spans": "Each phase records measured monotonic work.",
    "random_seed": "The reducer uses deterministic inputs.",
    "reproducibility_checksum": "Another process must recover the same inputs.",
    "sample_size_budget": "Views and seeds do not create independent families.",
    "source_artifact_hashes": "Old files cannot replace current producers.",
    "preconditions_checked": "Cheap failures precede expensive work.",
    "validation_receipts": "Registered checks must pass before readiness.",
    "verifier_is_oracle": "Fixture truth is not independent accuracy.",
    "claim_scope": "Natural data stay exposed development evidence.",
    "inference_substrate": "Duration follows invoked work.",
    "MODEL_SPECS": "Only current invocations count as model work.",
    "task_dispositions": "No planned task may vanish.",
    "G1": "Publication uses the stable gate.",
    "G2": "Independent reproduction is a separate gate.",
    "G3": "Prose narrowing is a separate gate.",
    "G4": "Headline traceability is a separate gate.",
    "paper_ready": "Publication needs every fixed gate.",
    "unmet_gates": "Report unmet gates without redefining them.",
    "continuation_decisions": "A qualified null retires its unchanged scope.",
    "external_blockers": "An unchanged external block is terminal.",
}


def failure(
    number: int,
    path: str,
    digest: str | None,
    field: str,
    expected: Any,
    observed: Any,
    operator: str = "==",
) -> dict[str, Any]:
    """Name the exact source operand that failed."""
    return dict(
        upstream_id=f"Exp{number}",
        artifact_path=path,
        artifact_hash=digest,
        field=field,
        operator=operator,
        expected=expected,
        observed=observed,
    )


def authority(root: Path) -> dict[str, Any]:
    """Compare the independent table, JSON block and matching YAML bytes."""
    path, roadmap, candidates = resolve_authority(root)
    text = (root / DESIGN).read_text()
    comparison = compare_contract(text, roadmap)
    return dict(
        tasks=roadmap["tasks"],
        roadmap=roadmap,
        comparison=comparison,
        candidates=candidates,
        design_path=str(DESIGN),
        roadmap_path=path.relative_to(root).as_posix(),
        hashes={
            str(DESIGN): sha256_file(root / DESIGN),
            path.relative_to(root).as_posix(): sha256_file(path),
        },
    )


def account(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep declared science, pre-gate receipts and planned self output separate."""
    if len(tasks) != 14 or [t["id"].split("-")[0] for t in tasks] != [
        f"exp{n}" for n in range(7781, 7795)
    ]:
        raise ValueError("V677 fourteen-task order required")
    rows, sources, failures = [], [], []
    values: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(tasks):
        number = 7781 + index
        label = task["deliverable"]
        producer = root / label
        exists = number != 7794 and producer.is_file()
        value = json.loads(producer.read_text()) if exists else {}
        digest = sha256_file(producer) if exists else None
        queue = (
            root
            / f"results/experiment_{number}_{task['id'].split('-', 1)[1].replace('-', '_')}.json"
        )
        queue_ok = not exists and number != 7794 and queue != producer and queue.is_file()
        queue_label = queue.relative_to(root).as_posix() if queue_ok else None
        queue_hash = sha256_file(queue) if queue_ok else None
        receipt = json.loads(queue.read_text()) if queue_ok else {}
        state = (
            "planned_output"
            if number == 7794
            else "producer"
            if exists
            else "pre_gate_receipt"
            if queue_ok
            else "absent"
        )
        eligible = bool(
            exists
            and value.get("verdict_class") in ELIGIBLE
            and value.get("flagged_adversarial") is False
            and str(value.get("honest_verdict", "")).startswith("complete_")
        )
        row = dict(
            unit_id=task["id"],
            task_id=task["id"],
            experiment_id=number,
            order=index + 1,
            phase=task["phase"],
            arm="task_accounting",
            producer_path=label,
            producer_hash=digest,
            availability=state,
            pre_gate_receipt_path=queue_label,
            pre_gate_receipt_hash=queue_hash,
            pre_gate_schema=receipt.get("schema"),
            producer_eligible=eligible,
            verdict_class=value.get("verdict_class"),
            honest_verdict=value.get("honest_verdict"),
            flagged_adversarial=value.get("flagged_adversarial"),
            raw_metrics={k: v for k, v in value.items() if k.endswith("_score")},
            censored=state in {"absent", "pre_gate_receipt"},
            exclusions=[] if eligible else [state],
            effective_independent_N=None,
        )
        rows.append(row)
        values[task["id"]] = value
        if number == 7794:
            continue
        sources.append(
            dict(
                upstream_id=f"Exp{number}",
                path=label,
                sha256=digest,
                date=value.get("run_date"),
                imported_fields=["verdict_class", "honest_verdict", "gate_check_summary"]
                if exists
                else [],
                eligible=eligible,
                pre_gate_path=queue_label,
                pre_gate_sha256=queue_hash,
            )
        )
        if not exists:
            failures.append(failure(number, label, None, "producer_exists", True, False))
        elif not eligible:
            failures.append(
                failure(
                    number,
                    label,
                    digest,
                    "producer_eligible",
                    True,
                    {
                        "verdict_class": value.get("verdict_class"),
                        "flagged_adversarial": value.get("flagged_adversarial"),
                    },
                )
            )
        for check in value.get("gate_check_summary", []):
            failures.append(
                failure(
                    number,
                    label,
                    digest,
                    str(check.get("field", "upstream_gate")),
                    check.get("expected", True),
                    check.get("observed", False),
                    str(check.get("operator", "==")),
                )
            )
    for task in tasks:
        for gate in task.get("gated_on", []):
            upstream = gate["upstream"]
            number = int(upstream.split("-", 1)[0].removeprefix("exp"))
            source = rows[number - 7781]
            observed = values[upstream].get(gate["artifact_field"])
            passed = observed == gate["value"] if gate["op"] == "==" else observed in gate["value"]
            if not passed:
                failures.append(
                    failure(
                        number,
                        source["producer_path"],
                        source["producer_hash"],
                        gate["artifact_field"],
                        gate["value"],
                        observed,
                        gate["op"],
                    )
                )
    return rows, sources, failures


def qwen_evidence(root: Path) -> dict[str, Any]:
    """Reopen every saved Qwen request and response before accepting its null."""
    path = root / QWEN_CANDIDATE
    if not path.is_file():
        raise FileNotFoundError(path)
    return qwen_cold_reduce(path)


def historical_boundaries(root: Path) -> list[dict[str, Any]]:
    """Preserve previous terminal determinations from their result bytes."""
    labels = (
        "results/experiment_7738_v673_capstone.json",
        "results/experiment_7752_v674_capstone.json",
        "results/experiment_7766_v675_capstone.json",
        "results/experiment_7780_v676_capstone.json",
    )
    rows = []
    for label in labels:
        path = root / label
        value = json.loads(path.read_text()) if path.is_file() else {}
        rows.append(
            dict(
                path=label,
                sha256=sha256_file(path) if path.is_file() else None,
                honest_verdict=value.get("honest_verdict"),
                verdict_class=value.get("verdict_class"),
            )
        )
    return rows


def build_artifact(
    root: Path,
    publication: dict[str, Any],
    receipts: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Compute one bounded terminal result from authenticated current inputs."""
    root = root.resolve()
    contract = authority(root)
    rows, sources, failures = account(root, contract["tasks"])
    if not contract["comparison"]["passed"]:
        failures.append(
            failure(
                7794,
                str(DESIGN),
                contract["hashes"][str(DESIGN)],
                "table_json_yaml_match",
                True,
                contract["comparison"]["errors"],
            )
        )
    qwen_source = rows[7787 - 7781]
    qwen = qwen_evidence(root) if qwen_source["producer_eligible"] else None
    qwen_path = root / QWEN_CANDIDATE
    if qwen is not None:
        producer = json.loads((root / qwen_source["producer_path"]).read_text())
        for field in ("parse_coverage_by_arm", "semantic_comparison_rows", "paired_improvements"):
            if qwen[field] != producer.get(field):
                failures.append(
                    failure(
                        7787,
                        qwen_source["producer_path"],
                        qwen_source["producer_hash"],
                        field,
                        producer.get(field),
                        qwen[field],
                    )
                )
        if qwen["benefit_passed"] != (producer.get("verdict_class") == "positive"):
            failures.append(
                failure(
                    7787,
                    qwen_source["producer_path"],
                    qwen_source["producer_hash"],
                    "benefit_passed",
                    False,
                    qwen["benefit_passed"],
                )
            )
    owned = list(receipts or [])
    for receipt in owned:
        if receipt.get("passed") is not True:
            failures.append(
                failure(
                    7794,
                    str(receipt.get("log_path")),
                    receipt.get("log_sha256"),
                    f"validation.{receipt.get('name')}.exit_code",
                    0,
                    receipt.get("exit_code"),
                )
            )
    checks_passed = bool(owned) and all(r.get("passed") is True for r in owned)
    valid = contract["comparison"]["passed"] and checks_passed
    verdict = "disqualified" if not valid else "blocked" if failures else "null"
    honest = (
        "complete_disqualified_v677_capstone_validation"
        if not valid
        else "complete_blocked_required_v677_evidence"
        if failures
        else "complete_null_v677_scientific_accounting"
    )
    rows[-1].update(
        verdict_class=verdict,
        honest_verdict=honest,
        raw_metrics={"capstone_complete_score": int(valid and not failures)},
    )
    decisions = {
        "normalized_decisions": {"action": "repair_Exp7785_then_measure_Exp7786", "evidence": None},
        "source_dependence": {"action": "produce_Exp7786_raw_rows", "evidence": None},
        "retained_learning": {
            "action": "repair_Exp7784_validation_then_run_Exp7788",
            "evidence": None,
        },
        "qwen_event_confidence": {
            "action": "retire_unchanged_exposed_scope"
            if qwen and not qwen["benefit_passed"]
            else "await_qualified_rows",
            "evidence": qwen,
        },
        "arc_organic": {"action": "repair_Exp7790_validation_then_run_Exp7791", "evidence": None},
        "service_efficiency": {"action": "produce_Exp7792_after_qualified_heads", "evidence": None},
        "hardware": {
            "action": "retain_board_prerequisites;_measure_device_service_before_advantage",
            "evidence": None,
        },
    }
    all_sources = sources + [
        dict(
            upstream_id="V677 authority",
            path=label,
            sha256=digest,
            date="20260928",
            imported_fields=["contract_bytes"],
            eligible=True,
        )
        for label, digest in contract["hashes"].items()
    ]
    all_sources.append(
        dict(
            upstream_id="Exp7787 raw",
            path=str(QWEN_CANDIDATE),
            sha256=sha256_file(qwen_path) if qwen_path.is_file() else None,
            date="20260928",
            imported_fields=["rows", "panel", "protocol"] if qwen else [],
            eligible=qwen is not None,
        )
    )
    history = historical_boundaries(root)
    all_sources.extend(
        dict(
            upstream_id="historical capstone",
            path=h["path"],
            sha256=h["sha256"],
            date=None,
            imported_fields=["honest_verdict", "verdict_class"],
            eligible=False,
        )
        for h in history
    )
    dispositions = [
        dict(
            experiment_id=row["experiment_id"],
            task_id=row["task_id"],
            availability=row["availability"],
            producer_path=row["producer_path"],
            producer_hash=row["producer_hash"],
            pre_gate_receipt_path=row["pre_gate_receipt_path"],
            pre_gate_receipt_hash=row["pre_gate_receipt_hash"],
            verdict_class=row["verdict_class"],
            honest_verdict=row["honest_verdict"],
            producer_eligible=row["producer_eligible"],
            minimum_changed_prerequisite="none"
            if row["producer_eligible"]
            else "complete_owned_capstone"
            if row["experiment_id"] == 7794
            else "produce_declared_science"
            if row["availability"] != "producer"
            else "repair_producer_validation",
        )
        for row in rows
    ]
    pub = {
        name: publication.get(name)
        for name in ("G1", "G2", "G3", "G4", "paper_ready", "unmet_gates")
    }
    artifact: dict[str, Any] = dict(
        schema="carnot.exp7794.v677.capstone.v1",
        experiment_id=7794,
        milestone="2026.09.677",
        run_date="20260928",
        honest_verdict=honest,
        verdict_class=verdict,
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=rows,
        task_dispositions=dispositions,
        acceptance_gate_results=dict(
            validity=valid,
            readiness=int(valid and not failures),
            probability_quality=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        duration_s=duration_s,
        phase_spans=spans or [],
        sample_size_budget=dict(
            intended=14,
            eligible=sum(r["producer_eligible"] for r in rows),
            started=sum(r["availability"] == "producer" for r in rows),
            completed=sum(r["verdict_class"] is not None for r in rows),
            excluded=sum(
                r["availability"] == "producer" and not r["producer_eligible"] for r in rows
            ),
            censored=sum(r["censored"] for r in rows),
            independent_N=None,
        ),
        source_artifact_hashes=all_sources,
        historical_boundaries=history,
        preconditions_checked=dict(
            root=str(root),
            authority_candidates=contract["candidates"],
            authority_match=contract["comparison"]["passed"],
            output_parent_exists=(root / OUTPUT.parent).is_dir(),
            backend="host_cpu_aggregation",
            resources="no model or board invoked",
        ),
        validation_receipts=dict(commands=owned, required_checks_passed=checks_passed),
        verifier_is_oracle=False,
        claim_scope=dict(
            natural_data="exposed_development_only",
            fixture="circular_positive_only",
            ARC="adapter_withheld_public_unmeasured",
        ),
        field_principles=PRINCIPLES,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="aggregation",
        planned_inference_substrate_class="aggregation",
        actual_inference_substrate_class="aggregation",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(
            loads=0,
            generations=0,
            forwards=0,
            input_tokens=0,
            output_tokens=0,
            failures=0,
            cancellations=0,
            loaded_files=[],
        ),
        random_seed=dict(seeds=[], purpose="deterministic aggregation"),
        continuation_decisions=decisions,
        external_blockers=[f for f in failures if f["upstream_id"] != "Exp7794"],
        next_smallest_falsifiable_experiment="Repair Exp7784 coverage receipt, then one qualified held-out decision family",
        publication_gate_results=publication,
        publication_source_hash=publication.get("source_hash"),
        capstone_complete_score=int(valid and not failures),
        **pub,
    )
    artifact["reproducibility_checksum"] = canonical_hash(
        dict(
            code=sha256_file(root / MODULE),
            cli=sha256_file(root / CLI) if (root / CLI).is_file() else None,
            sources=[(s["path"], s["sha256"]) for s in all_sources],
            roles="V677 capstone",
            parameters={"eligible": sorted(ELIGIBLE)},
            seeds=[],
        )
    )
    return artifact


def cold_replay(value: dict[str, Any], root: Path) -> list[str]:
    """Reopen source bytes and independently compare mutable conclusions."""
    errors = []
    for source in value.get("source_artifact_hashes", []):
        path = root / source["path"]
        observed = sha256_file(path) if path.is_file() else None
        if observed != source["sha256"]:
            errors.append("source_artifact_hashes")
    expected = build_artifact(
        root,
        value.get("publication_gate_results", {}),
        value.get("validation_receipts", {}).get("commands", []),
        value.get("phase_spans", []),
        value.get("duration_s", 0.0),
    )
    for name in (
        "rows",
        "task_dispositions",
        "gate_check_summary",
        "honest_verdict",
        "verdict_class",
        "continuation_decisions",
        "reproducibility_checksum",
    ):
        if value.get(name) != expected[name]:
            errors.append(name)
    return sorted(set(errors))
