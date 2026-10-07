"""REQ-REPORT-8242: authenticate inputs and publish replayable request accounting.

Readiness describes complete measured evidence. It does not claim answer quality,
generalization, production demand or a Rust acceleration comparison.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import recorder_execution_8213 as recorded
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.request_trace_inventory_8200 import copy_bytes, operand
from carnot.verify import concurrency_canary_8227 as e
from carnot.verify import independent_concurrent_service_8242 as measured

Json = dict[str, Any]
NAME = "experiment_8242_v712_independent_concurrent_service"
TASK = "exp8242-independent-concurrent-service"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_independent_concurrent_service_8242.py"
MODULES = [
    "python/carnot/verify/independent_concurrent_service_8242.py",
    "python/carnot/reporting/independent_concurrent_service_8242.py",
    "python/carnot/reporting/independent_concurrent_execution_8242.py",
]
UPSTREAM = "results/experiment_8236_v712_qualified_concurrency_canary.json"
PIN = "sha256:bb9fb9b43154811c880442b94d94a39e4866c23c987d57bc506a2db18eb735b5"
BINDINGS = "openspec/change-proposals/v712-concurrency-execution-bindings.json"
checksum = recorded.checksum


def inputs(root: Path, raw: Path) -> Json:
    """Exact qualified bytes grant authority; absent rows are never substituted."""
    data: Json = dict(ready=False, protocol={}, checks=[], refs=[], code=[])
    primary = root / UPSTREAM
    data["checks"].append(
        operand("qualified_canary_exists", primary, True, True if primary.is_file() else None)
    )
    path = primary
    try:
        upstream = json.loads(primary.read_text())
        data["refs"].append(copy_bytes(primary, raw))
        data["checks"].append(
            operand("qualified_canary_sha256", primary, PIN, e.sha256_file(primary))
        )
        for field, expected in [
            ("experiment_id", 8236),
            ("concurrent_canary_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            data["checks"].append(operand(field, primary, expected, upstream.get(field)))
        bound_path = (
            primary.parent
            / "raw"
            / primary.stem
            / "validators"
            / (e.sha256_file(primary).split(":")[1] + ".json")
        )
        bound = read_bound_sidecar(primary, bound_path)
        data["refs"].append(copy_bytes(bound_path, raw))
        terminal = json.loads(Path(upstream["terminal_validation_sidecar_path"]).read_text())
        data["checks"].append(
            operand(
                "qualified_terminal_binding",
                bound_path,
                True,
                terminal
                == dict(
                    receipts=bound["report"]["receipts"], candidate_sha256=e.sha256_file(primary)
                ),
            )
        )
        references = [
            r
            for r in upstream["raw_shard_hashes"]
            if r["path"] == upstream["terminal_validation_sidecar_path"]
        ]
        references += [
            dict(
                path=upstream["concurrent_protocol_path"],
                sha256=upstream["concurrent_protocol_sha256"],
            )
        ]
        references += upstream["code_config_hashes"]
        references += [
            dict(
                path=upstream["terminal_validation_sidecar_path"],
                sha256=e.sha256_file(Path(upstream["terminal_validation_sidecar_path"])),
            )
        ]
        for receipt in terminal["receipts"]:
            for stream in ["stdout", "stderr"]:
                references.append(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        for ref in references:
            path = Path(ref["path"])
            observed = e.sha256_file(path) if path.is_file() else None
            data["checks"].append(
                operand("qualified_operand_sha256", path, ref["sha256"], observed)
            )
            if observed == ref["sha256"]:
                data["refs"].append(copy_bytes(path, raw))
        path = Path(upstream["concurrent_protocol_path"])
        protocol = json.loads(path.read_text())
        if (
            protocol["schema"] != "carnot.v712.concurrency-execution-bindings.v1"
            or protocol["config"] != e.CONFIG
            or len(protocol["benchmark"]) != 96
            or len({r["source_cluster_id"] for r in protocol["benchmark"]}) != 24
            or len({r["request_id"] for r in protocol["benchmark"]}) != 96
            or protocol["launch_order"]
            != ["s1_serial", "s1_concurrent", "s2_concurrent", "s2_serial"]
        ):
            raise ValueError("protocol_obligations")
        for row in protocol["benchmark"]:
            e.recorder.validate_envelope(row["envelope"])
        data["protocol"] = protocol
        terminal = json.loads(Path(upstream["terminal_validation_sidecar_path"]).read_text())
        data["checks"].append(
            operand(
                "qualified_terminal_receipts",
                primary,
                True,
                bool(terminal["receipts"])
                and all(r["passed"] and r["normal_exit"] for r in terminal["receipts"]),
            )
        )
        base = recorded.inputs(root, raw / "service")
        data.update(
            {k: v for k, v in base.items() if k not in {"ready", "checks", "refs", "protocol"}}
        )
        data["checks"].extend(base["checks"])
        data["refs"].extend(base["refs"])
        data["ready"] = base["ready"] and all(c["passed"] for c in data["checks"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        data["checks"].append(
            operand(
                "qualified_operand_structure",
                path,
                "authenticated qualified protocol and terminal",
                str(error) if path.exists() else None,
            )
        )
    data["code"] = [copy_bytes(e.ROOT / p, raw / "owned_code") for p in [*MODULES, CLI, TEST]]
    return data


def validation_plan(private: Path) -> list[recorded.CommandSpec]:
    """Reuse bounded supervision while scoping coverage to newly owned statements."""
    with patch.multiple(recorded, MODULES=MODULES, TEST=TEST), patch.object(recorded.e, "CLI", CLI):
        plan = recorded.validation_plan(private)
    plan.append(
        recorded.CommandSpec(
            "e2e022",
            (
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_prospective_request_recorder_8213.py",
            ),
            "private_e2e",
            180,
        )
    )
    plan.append(
        recorded.CommandSpec(
            "coverage_json",
            (
                str(e.ROOT / ".venv/bin/coverage"),
                "json",
                "--data-file=" + str(private / ".coverage"),
                "-o",
                str(private / "coverage.json"),
            ),
            "owned",
            30,
        )
    )
    return plan


def validators(path: Path) -> list[recorded.CommandSpec]:
    """Fresh replay and unchanged auditors inspect a private terminal candidate."""
    with patch.object(recorded.e, "CLI", CLI):
        return recorded.validators(path)


def build(data: Json, work: Json, raw: Path, receipts: list[Json], duration: float) -> Json:
    """Keep execution qualification separate from speed and scientific benefit."""
    if not work.get("rows"):
        work = dict(
            work,
            rows=[
                dict(deepcopy(r), status="blocked", response_id=None, result={})
                for r in data.get("protocol", {}).get("benchmark", [])
            ],
        )
    value = measured.reduce(work)
    passed = bool(receipts) and all(r["passed"] and r["normal_exit"] for r in receipts)
    checks = data["checks"] + work.get("checks", [])
    blocked = not data["ready"] or any(not c["passed"] for c in checks)
    ready = passed and not blocked and value["evidence_complete"]
    cls = "blocked" if blocked else "null"
    verdict = (
        "complete_blocked_qualified_operand"
        if blocked
        else "complete_null_independent_request_costs"
    )
    if not passed:
        cls, verdict = "disqualified", "complete_disqualified_owned_validation"
    counts = deepcopy(ZERO_INVOCATION_COUNTS)
    loads = work.get("loads", [])
    calls = [r for r in value["rows"] if r.get("clocks", {}).get("start")]
    for prefix, events in [("model_loads", loads), ("generation_calls", calls)]:
        counts[prefix + "_attempted"] = len(events)
        counts[prefix + "_completed"] = sum(
            r.get(
                "completed",
                bool(r.get("result", {}).get("response", r.get("result", {})).get("usage")),
            )
            for r in events
        )
        counts[prefix + "_failed"] = len(events) - counts[prefix + "_completed"]
    counts["generation_calls"] = len(calls)
    value.update(
        experiment_id=8242,
        experiment=8242,
        task_id=TASK,
        milestone="2026.10.712",
        run_date="20261007",
        title="Independent serial and concurrent Qwen request costs",
        honest_verdict=verdict,
        verdict_class=cls,
        gate_check_summary=[c for c in checks if not c["passed"]],
        inference_substrate="live_llm_inference"
        if calls
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation" if calls else "no_model_load",
        MODEL_SPECS=e.MODEL_SPECS if loads else [],
        planned_model_specs=e.MODEL_SPECS,
        model_invocation_counts=counts,
        verifier_is_oracle=False,
        exposure_scope="public_development_designed_workload",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=False,
        concurrent_service_ready_score=int(ready),
        acceptance_gates=dict(
            authenticated=not blocked,
            owned_checks=passed,
            complete_measurement=ready,
            benefit=value["improvement_qualified"] and ready,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_receipt.json"),
        preconditions_checked=checks,
        duration_s=duration,
        random_seed=e.SEED,
        source_artifact_hashes=data.get("refs", []),
        code_config_hashes=data.get("code", []),
        raw_shard_hashes=[],
        cited_upstream_artifacts=[
            dict(
                path=r["path"],
                sha256=r["sha256"],
                fields_imported=[
                    "qualified request protocol, runtime identity, terminal and service configuration"
                ],
            )
            for r in data.get("refs", [])
        ],
        request_rows_path=str(raw / "request_rows.json"),
        model_load_receipts=loads,
        replay_inputs=dict(data_path=str(raw / "data.json"), work_path=str(raw / "work.json")),
        deployment_demand_observed=False,
        methodology="96 independent calls,24 development sources, two counterbalanced sweeps; Rust held fixed",
        claim_scope="descriptive request costs; parse validity does not measure answer quality",
        measurement_duration_s=work.get("measurement_duration_s", 0),
        coverage_statement_counts=data.get("coverage_statement_counts", {}),
    )
    value["field_principles"] = {
        k: "Bind actual work and evidence; missing measurements remain null; execution readiness supplies no scientific benefit."
        for k in value
    }
    return value


def replay(path: Path) -> bool:
    """Authenticate primitive custody and rebuild headline fields in a cold process."""
    try:
        value = json.loads(path.read_text())
        validate_primary(value, path)
        if value["reproducibility_checksum"] != checksum(value):
            return False
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if e.sha256_file(Path(ref.get("frozen_path", ref["path"]))) != ref["sha256"]:
                return False
        saved = value["replay_inputs"]
        data = json.loads(Path(saved["data_path"]).read_text())
        work = json.loads(Path(saved["work_path"]).read_text())
        for row in work.get("rows", []):
            if not row.get("journal_path"):
                continue
            journal = e.recorder.Journal(Path(row["journal_path"]))
            terminal = next(
                r
                for r in journal.events
                if r["event"] == "terminal" and r["request_id"] == row["request_id"]
            )
            if terminal["result"] != row["result"] or terminal["status"] != row.get(
                "acquisition_status", row["status"]
            ):
                return False
            primitive = json.loads(
                (Path(row["journal_path"]).parent / (row["request_id"] + ".json")).read_text()
            )
            clocks = row["clocks"]
            if (
                primitive["clocks"] != clocks
                or terminal["envelope"] != row["envelope"]
                or terminal["issued_monotonic_ns"] != clocks["issue"]
            ):
                return False
            if row["latency_s"] != (clocks["durability"] - clocks["issue"]) / 1e9:
                return False
        rebuilt = build(
            data,
            work,
            Path(saved["data_path"]).parent,
            value["validation_receipts"],
            value["duration_s"],
        )
        fields = list(measured.reduce(work)) + [
            "concurrent_service_ready_score",
            "verdict_class",
            "honest_verdict",
            "model_invocation_counts",
            "inference_substrate_class",
        ]
        return all(value[k] == rebuilt[k] for k in fields)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
