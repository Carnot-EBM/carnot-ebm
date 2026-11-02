"""REQ-REPORT-8415: sealed accounting separates qualification from scientific benefit."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from types import ModuleType
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import historical_consumer_roots_8402 as contract
from carnot.reporting import v721_capstone_evidence as base
from carnot.reporting import v723_capstone_evidence as previous
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import validate_primary

Json = dict[str, Any]
ROOT, DESIGN, STAGED, ACTIVE = contract.ROOT, contract.DESIGN, contract.STAGED, contract.ACTIVE
PROTOCOL, PROTOCOL_PIN = contract.PROTOCOL, contract.PROTOCOL_PIN
NAME, TASK, MILESTONE = "experiment_8415_v724_capstone", "exp8415-capstone", "2026.10.724"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v724_capstone_8415.py"
OWNED = [
    "python/carnot/reporting/v724_capstone_evidence.py",
    "python/carnot/reporting/v724_capstone.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
load, gate, memory = base.load, base.gate, previous.memory
PINS = {
    8402: "94d4baf2872bcf2bc58a7dd7a96f431c33ccad339c272927f1fc9673306b944d",
    8403: "13afe888c9f7f516d63adff8eab9eea25e011745db6530d41cd836925661235b",
    8404: "186b11a57be0c51b32fccb4a5b5064cd9a63220f5f4866a9c3d2740f2f102c67",
    8405: "b94b031f975ec6415e466fdf4c88df73180ac6b3e3afb7016fa2f47c75515270",
    8406: "b2aabf5bddc46f7b876a60f7d4484aa874a71331a70907775e90ee7e2fb7b83a",
    8407: "033cdbbe98be7790889354e50646f49c24a9e270cbb5c3c83b8d708c540c293f",
    8408: "d3951fcc7032dd586e1502a2c3ff1537e1479e44e7aafb39054c25438db7b80e",
    8409: "0929b927f547d028f05ad466aba9575a797d25a146a671b453db9de5a6eef2e8",
    8410: "c829f16dc732890872edd0224a3909e4cb3301adfffda11de19b06da5e283d4e",
    8411: "68195475cd2592ef87dc352cb84ecb5f0e075f5dc5e7e49c1f03bab4fc64d137",
    8412: "9b0bd990a672736b0b798a17d4f90a7c216e779b95265f7651e55cf05a9e37eb",
    8413: "ab4bd6cc49f95cc4c207cb35c0a721b8b2569e3c22d4e3c596dcbfadd887bf50",
    8414: "e572d8a1b6c31eaaf632338a6e55228c1696f652a44927afb4a151ba569d495a",
    8361: "33cdb0ba681c49eed70d57d5c49fd2c3fd76f7f2a7dd2a5253bbc3248e177ab4",
    8381: "fc3fee4db7a3dd9b238522c5164df6058ae699065ea4c3974b19f0f4f2663722",
    8401: "ba5b47cf9281d1614f8dec412c5fbdedf099bf40b8a5de9ef1e62558d038d4fd",
}
READERS = {
    8402: "reporting.historical_consumer_roots_8402",
    8403: "reporting.released_question_identity_8403",
    8404: "reporting.direct_state_qualification_8404",
    8406: "reporting.continuous_direct_learning_8406",
    8407: "reporting.python_transaction_cost_8407",
    8409: "reporting.retrospective_label_criterion_8409",
    8410: "verify.cuda_reader_8410",
    8412: "reporting.arc_panel_qualification_8412",
    8414: "reporting.gatemate_obligation_8414",
}
NEXT = [
    "Supply an independently preserved full V722 design with its original task digest; retain all original failed receipts.",
    "Resolve every prior question-exposure obligation and release80 disjoint human-labeled clusters, at least eight per class, with independent extraction truth.",
    "Qualify exact recovery under a preregistered deployment fault model beyond the current process-death controls; preserve pending predictions and acknowledgments.",
    "Resolve the exact max_coefficient_error informational finding through an existing typed primitive consumer with a deliberate-error control; preserve finite parity scope.",
    "Measure changed independent delayed feedback with all96 slots, three arms and retention windows0/32/64/96 under unchanged thresholds.",
    "Measure new complete request costs with production arrival and feedback conditions, including all copies, persistence and tail drain.",
    "After native_parity_ready_score=1, measure matched complete native/Python cycles and every request operation.",
    "Replicate criterion and extraction effects on80 independently unexposed question clusters with eight per class and sealed human criteria.",
    "Supply a hash-bound causal driver/runtime repair receipt and pass one bounded CUDA context/copy probe.",
    "After runtime reader, actual runtime change and context qualify, measure the bounded canary as feasibility only.",
    "Supply authenticated adapter-withheld cross-game outcomes with three overlapping games and five firings per shared cell before any arm refinement.",
    "Resolve original_source_drift_and_tamper_controls and coverage_report, then supply complete native/transfer clocks and an actually supported operation mapping.",
    "Recover both exact missing source hashes and dated physical change, IDCODE0x20000001, n16 flash and device sample/hash smoke before probing.",
    "Continue only changed qualified evidence for independent criterion truth, causal adaptation/recovery and complete native costs; preserve closed H1/H2 nulls.",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts keep bounded work visible without inventing completed observations."""
    print(f"[exp8415] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(path: Path, raw: Path, expected: str | None = None) -> Json:
    """A supplied producer digest authenticates copied bytes, while absence remains explicit."""
    ref = dict(base.freeze(path, raw, expected))
    if ref["exists"] and expected is not None and ref["sha256"] != expected:
        raise ValueError("source_sha256")
    return ref


def references(value: Any) -> list[Json]:
    """Snapshot-only authority records retain their declared original read aliases."""
    refs: list[Json] = []
    if isinstance(value, dict):
        if isinstance(value.get("sha256"), str) and isinstance(
            value.get("path", value.get("snapshot_path")), str
        ):
            refs.append(
                dict(
                    value,
                    path=value.get("path", value.get("source_path", value.get("snapshot_path"))),
                )
            )
        for item in value.values():
            refs.extend(references(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(references(item))
    return refs


def seal(value: Json, raw: Path) -> list[Json]:
    """Copy one declared operand closure, retaining aliases without recursively replaying history."""
    operands = references(
        {
            k: v
            for k, v in value.items()
            if k
            not in ["cited_upstream_artifacts", "exact_missing_hashes", "missing_source_hashes"]
        }
    )
    if isinstance(value.get("code_config_hashes"), dict):
        operands.extend(
            dict(path=str(ROOT / path), sha256=digest)
            for path, digest in value["code_config_hashes"].items()
        )
    for key in ["work_reference", "measurement_reference", "primitive_reference"]:
        if value.get(key):
            state = load(value[key])
            operands.extend(references(state))
    refs = []
    unique = {(r.get("snapshot_path", r["path"]), r["sha256"]): r for r in operands}
    for index, ref in enumerate(unique.values()):
        if index % 50 == 0:
            progress("seal_operands", index, len(unique) - index)
        source = Path(str(ref.get("snapshot_path", ref["path"])))
        saved = freeze(source, raw, ref["sha256"])
        saved.update(path=str(source))
        alias = ref.get("source_path", ref.get("original_path", ref["path"]))
        if alias != str(source) or ref.get("source_path"):
            saved["source_path"] = alias
        refs.append(saved)
    return refs


def outcome(task: Json, primary: Json, raw: Path, sealed: list[Json] | None = None) -> Json:
    """Authenticate original terminal identity before importing fields or replaying its primitives."""
    value = load(primary)
    number = int(task["id"][3:7])
    if primary["sha256"] != "sha256:" + PINS[number]:
        raise ValueError("primary_pin")
    pre = value.get("blocked_at_layer") == "conductor_pre_gate"
    refs = list(sealed or [primary])
    publication_changed = False
    if not pre:
        validate_primary(value, Path(primary["path"]))
        if value["task_id"] != task["id"]:
            raise ValueError("task_identity")
        if sealed is None:
            terminal = freeze(Path(value["terminal_validation_sidecar_path"]), raw / "terminal")
            publication = load(terminal)
            publication = publication.get("publication", publication)
            side = freeze(Path(publication["sidecar_path"]), raw / "terminal")
            refs.extend([terminal, side, *seal(value, raw / "custody")])
        else:
            publication = load(refs[1])
            publication = publication.get("publication", publication)
        side_value = load(refs[2])
        if (
            publication["primary_sha256"] != primary["sha256"]
            or side_value["primary_sha256"] != primary["sha256"]
            or side_value["report"]["passed"] is not True
            or publication["primary_path"] != primary["path"]
        ):
            original_path = Path(publication["sidecar_path"]).parents[1] / "terminal_candidate.json"
            original_ref = freeze(original_path, raw / "original", publication["primary_sha256"])
            original = load(original_ref)
            if (
                original_ref["sha256"] != publication["primary_sha256"]
                or side_value["primary_sha256"] != publication["primary_sha256"]
                or side_value["report"]["passed"] is not True
                or {k: v for k, v in original.items() if k != "flagged_adversarial"}
                != {k: v for k, v in value.items() if k != "flagged_adversarial"}
                or value["flagged_adversarial"] is not True
            ):
                raise ValueError("terminal_binding")
            refs.append(original_ref)
            publication_changed = True
    checks = []
    if publication_changed:
        checks.append(
            gate(
                primary["path"],
                "flagged_adversarial.changed_after_publication",
                False,
                True,
                primary["sha256"],
            )
        )
    if pre:
        if value["experiment"] != number:
            raise ValueError("pre_gate_identity")
        observed = value["gates_evaluated"]
        if [(g["upstream"], g["artifact_field"], g["op"], g["expected"]) for g in observed] != [
            (g["upstream"], g["artifact_field"], g["op"], g["value"]) for g in task["gated_on"]
        ]:
            raise ValueError("pre_gate_contract")
        for g in observed:
            ref = next((r for r in refs if r["path"] == g["artifact_path"]), None)
            if ref is None:
                ref = freeze(Path(g["artifact_path"]), raw / "pregates", g["artifact_sha256"])
                refs.append(ref)
            if (
                ref["sha256"] != g["artifact_sha256"]
                or load(ref).get(g["artifact_field"]) != g["actual"]
            ):
                raise ValueError("pre_gate_operand")
            checks.append(
                dict(
                    gate(
                        g["artifact_path"],
                        g["artifact_field"],
                        g["expected"],
                        g["actual"],
                        g["artifact_sha256"],
                    ),
                    upstream=g["upstream"],
                )
            )
    else:
        checks.extend(
            g
            for g in value.get("gate_check_summary", [])
            if isinstance(g, dict) and not g.get("passed")
        )
        checks.extend(
            gate(
                r["stdout_path"],
                r["name"],
                r["expected_exit"],
                r["actual_exit"],
                r["stdout_sha256"],
            )
            for r in value["validation_receipts"]
            if not r["passed"] and r.get("scope") != "global"
        )
    missing = [r for r in refs if not r["exists"] and r.get("expected_sha256")]
    checks.extend(gate(r["path"], "source.exists", r["expected_sha256"], None) for r in missing)
    qualified = (
        not pre
        and value.get("required_checks_passed") is True
        and value.get("flagged_adversarial") is False
    )
    branch = dict(
        status="pre_gate_nonexecution" if pre else "original_disposition_preserved", passed=False
    )
    if qualified and not missing:
        module = importlib.import_module("carnot." + READERS[number])
        source = Path(str(module.__file__))
        source_ref = next(r for r in refs if r.get("source_path", r["path"]) == str(source))
        pin = source_ref["sha256"]
        if sha256_file(source) != pin:
            # An old receipt keeps its old consumer after a later source repair.
            # The imported helpers still use their explicitly sealed operands.
            frozen = ModuleType(module.__name__ + "__frozen_8415")
            frozen.__file__ = str(source)
            code = contract.checked(Path(source_ref["snapshot_path"]), pin).read_text()
            exec(compile(code, str(source), "exec"), frozen.__dict__)
            module = frozen
        with TemporaryDirectory(prefix="exp8415-reader-", dir="/var/tmp") as directory:
            scratch = Path(directory)
            original_family = contract.authenticate_family
            family_refs = (
                seal(
                    load(next(r for r in refs if r["sha256"] == "sha256:" + PINS[8402])),
                    scratch / "family",
                )
                if number in [8404, 8410]
                else []
            )

            def family(primary_path: Path, name: str) -> Json:
                """Each nested consumer receives its own producer's authority aliases."""
                (scratch / "family_writes").mkdir(parents=True, exist_ok=True, mode=0o700)
                with base.frozen_inputs(family_refs, scratch / "family_writes"):
                    return dict(original_family(primary_path, name))

            with base.frozen_inputs(refs, scratch):
                with patch.object(contract, "authenticate_family", family):
                    passed = module.replay(
                        value if number == 8412 else Path(primary["snapshot_path"])
                    )
        branch = dict(
            status="cold_replay_passed" if passed else "disqualified_cold_replay",
            passed=bool(passed),
        )
        if not passed:
            checks.append(
                gate(primary["path"], "branch.primitive_replay", True, False, primary["sha256"])
            )
    selected = (
        {}
        if pre
        else {
            k: v
            for k, v in value.items()
            if k.endswith("_score")
            or k
            in [
                "intended_count",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "independent_count",
                "sample_size_budget",
                "exposure_scope",
                "verifier_is_oracle",
                "required_checks_passed",
                "flagged_adversarial",
                "adversarial_findings",
                "coverage_combination_mode",
                "coverage_file_manifest",
                "model_invocation_counts",
            ]
        }
    )
    if not pre and isinstance(value.get("rows"), list):
        selected["primitive_rows_reference"] = dict(
            primary, json_pointer="/rows", row_count=len(value["rows"])
        )
    if not pre and "historical_model_provenance" in value:
        selected["historical_model_provenance"] = [
            dict(
                primary_reference=primary,
                json_pointer="/historical_model_provenance",
                exposure_scope="cached_upstream_only; zero_current_LLM_calls",
            )
        ]
    return dict(
        producer_executed=not pre,
        eligible=qualified and branch["passed"] and value["verdict_class"] != "blocked",
        honest_verdict="complete_disqualified_upstream_flagged_after_publication"
        if publication_changed
        else value["honest_verdict"],
        verdict_class="disqualified"
        if publication_changed
        else "blocked"
        if pre
        else value["verdict_class"],
        original_honest_verdict=value["honest_verdict"],
        original_verdict_class="blocked" if pre else value["verdict_class"],
        selected=selected,
        checks=checks,
        closure=refs,
        branch_replay=branch,
        disposition="pre_gate_receipt" if pre else "executed_producer",
    )


def worker_process(request: Path, output: Path) -> int:
    """Reuse fresh exec memory sampling so parent watermarks cannot distort worker growth."""
    with patch.multiple(base, outcome=outcome, memory=memory, progress=progress):
        return int(base.worker_process(request, output))


def invoke(
    task: Json, primary: Json, raw: Path, closure: list[Json] | None = None
) -> tuple[Json, Json]:
    """Shipped bounded children retain exact argv, clocks, logs and failure exits."""
    with patch.multiple(base, CLI=CLI, progress=progress):
        summary, receipt = base.invoke(task, primary, raw, closure)
    return dict(summary), dict(receipt)


def slot(task: Json, primary: Json, cascade: str | None, summary: Json | None = None) -> Json:
    """Keep scheduling evidence distinct from authentic executed measurements and absent inputs."""
    state = summary or (
        outcome(task, primary, Path(primary["snapshot_path"]).parent)
        if primary["exists"]
        else dict(
            producer_executed=False,
            eligible=False,
            honest_verdict=None,
            verdict_class="blocked",
            selected={},
            checks=[gate(primary["path"], "producer.exists", True, None)],
            closure=[primary],
            branch_replay=dict(status="absent_input", passed=False),
            disposition="cascade_skip" if cascade else "absent_primary",
        )
    )
    number = int(task["id"][3:7])
    return dict(
        state,
        task=task,
        experiment_id=number,
        task_id=task["id"],
        unit_id=task["id"],
        primary_reference=primary,
        metrics=state["selected"] or None,
        missing_reason=None if state["eligible"] else state["branch_replay"]["status"],
        continuation_condition=NEXT[number - 8402],
        cascade_log_line=cascade,
    )


def authority(work: Json) -> list[Json]:
    """Recompute complete task objects from sealed design and YAML, never the current roadmap."""
    refs = work["authority_refs"]
    text = contract.checked(Path(refs[0]["snapshot_path"]), refs[0]["sha256"]).read_text()
    tasks = contract.parse_design(text, milestone=MILESTONE)[1]
    active = yaml.safe_load(
        contract.checked(Path(refs[2]["snapshot_path"]), refs[2]["sha256"]).read_bytes()
    )
    if (
        contract.tasks_digest(tasks) != contract.TASKS_PIN
        or tasks != active["tasks"]
        or active["milestone"] != MILESTONE
        or tasks[-1]["id"] != TASK
        or work["tasks"] != tasks
        or work["canonical_tasks_sha256"] != canonical_hash(tasks)
    ):
        raise ValueError("full_task_authority")
    if (
        refs[1]["exists"]
        and yaml.safe_load(Path(refs[1]["snapshot_path"]).read_bytes())["tasks"] != tasks
    ):
        raise ValueError("staged_task_authority")
    return list(tasks)


def measure(root: Path, raw: Path) -> Json:
    """Freeze only declared current closures and selected historical operands before reduction."""
    progress("preconditions_before", 0, 14)
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    start, before = time.monotonic_ns(), memory()
    auth = contract.authority(root, raw / "authority")
    if not auth["activated"] or len(auth["tasks"]) != 14:
        raise ValueError("complete_current_authority")
    resources = contract.measure(root, raw / "preconditions")
    refs = [
        dict(freeze(root / name, raw / "authority_custody"), source_path=str(root / name))
        for name in [DESIGN, STAGED, ACTIVE]
    ]
    protocol = freeze(root / PROTOCOL, raw / "custody", PROTOCOL_PIN)
    if protocol["sha256"] != PROTOCOL_PIN:
        raise ValueError("protocol_pin")
    conductor = freeze(root / "ops/conductor-log.md", raw / "custody")
    lines = Path(conductor["snapshot_path"]).read_text().splitlines() if conductor["exists"] else []
    inputs, slots, receipts = [], [], []
    for index, task in enumerate(auth["tasks"][:-1]):
        progress("producer_before", index, 13 - index)
        candidates = list((root / "results").glob(f"experiment_{8402 + index}_*.json"))
        if len(candidates) > 1:
            raise ValueError("ambiguous_primary")
        path = candidates[0] if candidates else root / task["deliverable"]
        primary = freeze(path, raw / "custody", "sha256:" + PINS[8402 + index])
        summary = None
        if primary["exists"]:
            summary, receipt = invoke(task, primary, raw / "branches" / task["id"])
            receipts.append(receipt)
        cascade = next(
            (
                s
                for s in lines
                if "Pre-emptive skip: upstream retired" in s
                and any(g["upstream"] in s for g in task.get("gated_on", []))
            ),
            None,
        )
        row = slot(task, primary, cascade, summary)
        slots.append(row)
        inputs.append(dict(task=task, primary=primary, summary=row))
        progress("producer_after", index + 1, 12 - index)
    history = []
    for number in [8361, 8381, 8401]:
        paths = list((root / "results").glob(f"experiment_{number}_*.json"))
        if not paths:
            continue
        if sha256_file(paths[0]) != "sha256:" + PINS[number]:
            raise ValueError("historical_primary_pin")
        item = base.historical(paths[0], raw / "history")
        if not item["authenticated"]:
            raise ValueError("historical_terminal_binding")
        value = load(item["reference"])
        if number == 8361:
            item["refs"].extend(seal(value, raw / "utility"))
            state = load(value["measurement_reference"])
            for audit in state["audits"].values():
                item["refs"].extend(seal(load(audit["measurement_reference"]), raw / "utility"))
            item["selected"] = base.utility(value, item["refs"])
        history.append(item)
    exclusion = dict(
        freeze(root / "ops/exclusion_manifest.yaml", raw / "custody"),
        source_path=str(root / "ops/exclusion_manifest.yaml"),
    )
    after = memory()
    checks = [
        gate(str(root / DESIGN), "authority.activated", True, auth["activated"]),
        gate(str(root / PROTOCOL), "protocol.sha256", PROTOCOL_PIN, protocol["sha256"]),
    ]
    checks.extend(resources["failures"])
    work = dict(
        root=str(root),
        tasks=auth["tasks"],
        canonical_tasks_sha256=canonical_hash(auth["tasks"]),
        authority_refs=refs,
        protocol_reference=protocol,
        slots=slots,
        inputs=inputs,
        history=history,
        extra_refs=[conductor, exclusion, *resources["refs"]],
        checks=checks,
        branch_receipts=receipts,
        preconditions=resources["preconditions"],
        fixtures=resources["fixtures"],
        original_failures=resources["historical_failure_reproduction"],
        global_health=resources.get("global_health", []),
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        memory_measurements=dict(
            parent_before=before,
            parent_after=after,
            parent_growth_mb=max(0, after["current_rss_mb"] - before["current_rss_mb"]),
            peak_bound_mb=1500,
            growth_bound_mb=500,
            workers=[s.get("memory", {}) for s in slots],
        ),
    )
    if work["global_health"]:
        work["extra_refs"].append(
            freeze(
                ROOT / "results/experiment_8388_v723_contract_methods.json",
                raw / "health",
                contract.PRIMARY_PINS[8388],
            )
        )
        for receipt in work["global_health"]:
            for stream in ["stdout", "stderr"]:
                work["extra_refs"].append(
                    freeze(
                        Path(receipt[stream + "_path"]), raw / "health", receipt[stream + "_sha256"]
                    )
                )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_after", 13, 1)
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Readiness qualifies this reduction; original blocked and failed science stays unchanged."""
    authority(work)
    owned = bool(receipts) and all(
        r["passed"] for r in receipts if r.get("scope") not in ["global", "publication"]
    )
    owned &= all(r["passed"] for r in work["branch_receipts"])
    owned &= (
        work["memory_measurements"]["parent_growth_mb"] <= 500
        and work["memory_measurements"]["parent_after"]["current_rss_mb"] <= 1500
    )
    rows = [
        {k: v for k, v in s.items() if k not in ["task", "closure", "selected", "checks", "memory"]}
        for s in work["slots"]
    ]
    kind = (
        "disqualified"
        if not owned
        else "blocked"
        if any(not g["passed"] for g in work["checks"]) or any(not r["eligible"] for r in rows)
        else "null"
    )
    rows.append(
        dict(
            experiment_id=8415,
            task_id=TASK,
            unit_id=TASK,
            disposition="current_capstone",
            producer_executed=True,
            eligible=owned,
            honest_verdict="complete_" + kind + "_v724_capstone",
            verdict_class=kind,
            primary_reference=None,
            metrics=None,
            missing_reason=None,
            continuation_condition=NEXT[-1],
        )
    )
    utility: Json = next((h["selected"] for h in work["history"] if h["experiment_id"] == 8361), {})
    failures = [
        *work["checks"],
        *[g for s in work["slots"] for g in s["checks"] if not g.get("passed")],
    ]
    failures.extend(
        gate(r["stdout_path"], r["name"], True, False, r["stdout_sha256"])
        for r in receipts
        if not r["passed"] and r.get("scope") not in ["global", "publication"]
    )
    gates = [
        dict(
            check=g.get("check", g.get("artifact_field", g.get("field"))),
            upstream=g.get("upstream", g.get("upstream_id", TASK)),
            path=g.get("path", g.get("artifact_path")),
            sha256=g.get("sha256", g.get("artifact_hash", g.get("hash"))),
            artifact_field=g.get("artifact_field", g.get("field")),
            operator=g.get("operator", g.get("op", "==")),
            expected=g.get("expected", g.get("expected_value")),
            observed=g.get("observed", g.get("observed_value", g.get("actual"))),
            passed=g["passed"],
        )
        for g in failures
    ]
    return dict(
        rows=rows,
        task_dispositions=rows,
        honest_verdict=rows[-1]["honest_verdict"],
        verdict_class=kind,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        capstone_execution_ready_score=int(owned),
        science_ready_score=0,
        gate_check_summary=gates,
        actual_executed_task_count=sum(r["producer_executed"] for r in rows),
        pre_gate_count=sum(r["disposition"] == "pre_gate_receipt" for r in rows),
        cascade_skip_count=sum(r["disposition"] == "cascade_skip" for r in rows),
        missing_output_count=sum(not r["primary_reference"]["exists"] for r in rows[:-1]),
        H1=utility.get("H1", dict(qualified=False, intended_count=128)),
        H2=utility.get("H2", dict(qualified=False, intended_count=88)),
        qualified_scope_decision=utility.get("qualified_scope_decision", {}),
        retirements=[],
        next_evidence_conditions=NEXT,
        three_prd_gaps=[
            dict(gap=name, closed=False, next_evidence_condition=NEXT[index])
            for name, index in [
                ("independent_criterion_and_extraction_truth", 7),
                ("actual_causal_adaptation_and_deployment_recovery", 4),
                ("full_native_cost", 6),
            ]
        ],
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reuse the terminal schema while binding every V724 field to sealed source operands."""
    with patch.multiple(base, reduce=reduce, TASK=TASK, MILESTONE=MILESTONE, NEXT=NEXT):
        value = base.build(work, receipts, raw, output)
    rows = value["rows"]
    excluded = sum(r["disposition"] in ["pre_gate_receipt", "cascade_skip"] for r in rows)
    failed = sum(r["verdict_class"] == "disqualified" for r in rows)
    completed = sum(r["eligible"] and r["verdict_class"] != "disqualified" for r in rows)
    old: Json = next(
        (load(h["reference"]) for h in work["history"] if h["experiment_id"] == 8401), {}
    )
    original: Json = next((r for r in work["original_failures"] if r["experiment_id"] == 8401), {})
    value.update(
        experiment_id=8415,
        run_date="20261011",
        random_seed=7248415,
        completed_count=completed,
        failed_count=failed,
        excluded_count=excluded,
        censored_count=14 - completed - failed - excluded,
        historical_consumer_receipts=work.get("historical_receipts", []),
        historical_v723=dict(
            honest_verdict=old.get("honest_verdict"),
            verdict_class=old.get("verdict_class"),
            actual_executed_task_count=old.get("actual_executed_task_count"),
            pre_gate_count=old.get("pre_gate_count"),
            cascade_skip_count=old.get("cascade_skip_count"),
            failed_count=old.get("failed_count"),
            failed_consumer_node_ids=original.get("failed_node_ids", []),
            original_failure_log=original.get("log"),
            original_assertions_preserved=True,
        ),
        repository_health=dict(
            status="authenticated_prior_scope_only" if work["global_health"] else "unknown",
            receipts=work["global_health"],
            source_scope="Exp8388 historical global-health receipts imported by authenticated Exp8402; no current global pass",
            current_suite_launched=False,
        ),
        dyadic_logit_certificate=dict(
            reference=next(
                (h["reference"] for h in work["history"] if h["experiment_id"] == 8381), None
            ),
            separate_numerical_policy=True,
            deployed=False,
            semantic_benefit=False,
        ),
        closed_exposed_utility=dict(H1=-0.00390625, H2=0, reopened=False),
        scope_reduction_compliance=load(work["protocol_reference"])["scope_reduction_compliance"],
        preconditions_checked=[*work["checks"], work["preconditions"]],
        phase_spans=[
            dict(phase=r["name"], duration_s=r["duration_s"])
            for r in [*work["branch_receipts"], *work.get("historical_receipts", []), *receipts]
        ],
        invocation_argv=work.get("invocation_argv", []),
        historical_fixture_hashes=work["fixtures"],
        retirement_assessment=dict(
            append_performed=False,
            unchanged_scope_independently_proven=False,
            hypothesis_retired=False,
            reason="No independently qualified exact prior verdict with substantive scope equality; blocked nonexecution cannot retire an unmeasured hypothesis.",
        ),
        methodology_note="Read-only aggregation authenticates fourteen full-task slots and actual terminal bytes. Fresh bounded children replay cached primitives with immutable source aliases. Original blocked and disqualified evidence remains visible. Exposed H1/H2 nulls stay closed; independent criterion truth, causal adaptation/recovery and full native costs remain open. Retrospective labels, two-game ARC and CUDA feasibility grant no generalization or headline credit. No current LLM calls, generator updates, activation or external publication.",
    )
    value["source_artifact_hashes"].extend(work["extra_refs"])
    value["raw_shard_hashes"].extend(work.get("coverage_shards", []))
    value["cited_upstream_artifacts"].extend(
        dict(s["primary_reference"], fields_imported=list(s["metrics"]))
        for s in work["slots"]
        if s["metrics"]
    )
    value["field_principles"].update(
        {
            k: "Bind "
            + k
            + " to authenticated primitive operands and actual invocation; qualification grants no scientific benefit."
            for k in value
        }
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return dict(value)


def replay(path: Path) -> bool:
    """Rehashed summaries cannot replace independently rebuilt rows, authority or child receipts."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != canonical_hash(
            {k: v for k, v in value.items() if k != "reproducibility_checksum"}
        ):
            return False
        work = load(value["work_reference"])
        authority(work)
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if ref.get("exists", True):
                contract.checked(Path(ref.get("snapshot_path", ref["path"])), ref["sha256"])
        for receipt in [*value["validation_receipts"], *work["branch_receipts"]]:
            for stream in ["stdout", "stderr"]:
                contract.checked(Path(receipt[stream + "_path"]), receipt[stream + "_sha256"])
            original = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            if any(
                original.get(k) != receipt.get(k)
                for k in ["argv", "actual_exit", "stdout_sha256", "stderr_sha256"]
            ):
                return False
        with TemporaryDirectory(prefix="exp8415-cold-", dir="/var/tmp") as directory:
            raw = Path(directory)
            actual = build(
                work, value["validation_receipts"], raw, Path(value["publication_output"])
            )
            for key in ["work_reference", "raw_shard_hashes"]:
                actual[key] = value[key]
            actual["reproducibility_checksum"] = canonical_hash(
                {k: v for k, v in actual.items() if k != "reproducibility_checksum"}
            )
            if actual != value:
                return False
            for h in work["history"]:
                primary, terminal, side = [load(r) for r in h["refs"][:3]]
                if (
                    primary["honest_verdict"] != h["honest_verdict"]
                    or not h["authenticated"]
                    or terminal["publication"]["primary_sha256"] != h["reference"]["sha256"]
                    or side["primary_sha256"] != h["reference"]["sha256"]
                ):
                    return False
                if h["experiment_id"] == 8361 and base.utility(primary, h["refs"]) != h["selected"]:
                    return False
            for index, row in enumerate(work["slots"]):
                if row["task"] != work["tasks"][index]:
                    return False
                summary = None
                if row["primary_reference"]["exists"]:
                    summary, receipt = invoke(
                        row["task"], row["primary_reference"], raw / str(index), row["closure"]
                    )
                    if not receipt["passed"]:
                        return False
                actual = slot(
                    row["task"], row["primary_reference"], row.get("cascade_log_line"), summary
                )
                if any(actual.get(k) != row.get(k) for k in row if k not in ["memory", "closure"]):
                    return False
            return True
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False
