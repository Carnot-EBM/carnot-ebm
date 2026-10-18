"""REQ-REPORT-8236: bind current execution to the unchanged request protocol.

The old reducer remains in use so HTTP failures receive the same accounting.
Current readiness has no bearing on speed or independent reasoning benefit.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import concurrency_execution_8227 as old
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.request_trace_inventory_8200 import copy_bytes, operand
from carnot.verify import concurrency_canary_8227 as e

Json = dict[str, Any]
NAME = "experiment_8236_v712_qualified_concurrency_canary"
TASK = "exp8236-qualified-concurrency-canary"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_qualified_concurrency_8236.py"
MODULES = [
    *e.MODULES,
    "python/carnot/reporting/qualified_concurrency_8236.py",
    "python/carnot/reporting/qualified_concurrency_execution_8236.py",
]
BINDINGS = "openspec/change-proposals/v712-concurrency-execution-bindings.json"
PRIOR = "results/experiment_8227_v711_concurrency_canary.json"
EXECUTION_FILES = [
    "python/carnot/inference/sota_models.py",
    "python/carnot/inference/qwen_sufficiency_7920.py",
    "python/carnot/inference/llama_cpp_process.py",
    "python/carnot/inference/gguf_metadata.py",
    "python/carnot/gpu_lease_phase_journal.py",
    "python/carnot/verify/request_recorder_8213.py",
]


def bind(original: Json, code: list[Json]) -> Json:
    """Reject changed obligations before adding independent invocation identities."""
    canary, benchmark = original["canary"], original["benchmark"]
    sources = {r["source_cluster_id"] for r in canary}
    future = {r["source_cluster_id"] for r in benchmark}
    if (
        original["schema"] != "carnot.v711.concurrent-acquisition.v1"
        or original["config"] != e.CONFIG
        or len(canary) != 8
        or len(sources) != 4
        or len(benchmark) != 96
        or len(future) != 24
        or sources & future
    ):
        raise ValueError("original_protocol_obligations")
    for row in canary + benchmark:
        e.recorder.validate_envelope(row["envelope"])
    value = deepcopy(original)
    value.update(
        schema="carnot.v712.concurrency-execution-bindings.v1",
        original_protocol_sha256=e.key(original),
        execution_code=code,
        benchmark_executed_here=False,
        benefit_claim=False,
    )
    for row in value["canary"] + value["benchmark"]:
        row.update(
            request_id="v712-" + row["request_id"], response_id="intended-v712-" + row["request_id"]
        )
    return value


def inputs(root: Path, raw: Path) -> Json:
    """Authenticate historical primary and protocol without replacing missing rows."""
    data = old.qualified.inputs(root, raw / "lineage")
    data["code"] = [
        copy_bytes(e.ROOT / p, raw / "code")
        for p in [*MODULES, *EXECUTION_FILES, CLI, TEST, e.TEST]
    ]
    path = root / old.PROTOCOL
    try:
        prior_path = root / PRIOR
        prior = json.loads(prior_path.read_text())
        data["refs"].append(copy_bytes(prior_path, raw / "prior"))
        for field, expected in [("experiment_id", 8227), ("verdict_class", "disqualified")]:
            data["checks"].append(operand(field, prior_path, expected, prior.get(field)))
        current_path = path
        frozen = next(
            r
            for r in prior["source_artifact_hashes"]
            if r["sha256"] == prior["concurrent_protocol_sha256"]
        )
        path = Path(frozen["frozen_path"])
        data["current_protocol_observation"] = dict(
            path=str(current_path),
            expected=prior["concurrent_protocol_sha256"],
            observed=e.sha256_file(current_path) if current_path.is_file() else None,
            disposition="historical_frozen_protocol_is_authority",
        )
        data["checks"].append(
            operand(
                "original_protocol_sha256",
                path,
                prior["concurrent_protocol_sha256"],
                e.sha256_file(path),
            )
        )
        original = json.loads(path.read_text())
        data["refs"].append(copy_bytes(path, raw / "protocol"))
        data["protocol"] = bind(original, data["code"])
        data["checks"].append(
            operand("protocol_identity", path, data["identity"], original["identity"])
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        data["checks"].append(
            operand(
                "original_protocol",
                path,
                "authenticated Exp8227 protocol",
                str(error) if path.exists() else None,
            )
        )
    for name in old.NAMED + [old.UPSTREAM]:
        source = root / name
        data["checks"].append(
            operand("named_input_" + source.stem, source, True, True if source.is_file() else None)
        )
        if source.is_file():
            data["refs"].append(copy_bytes(source, raw / "named"))
    data["ready"] = all(c["passed"] for c in data["checks"]) and all(
        r["passed"] for r in data["precondition_receipts"]
    )
    return data


def validation_plan(private: Path) -> list[old.CommandSpec]:
    """Measure both old error statements and new real CLI children before GPU use."""
    with (
        patch.object(e, "MODULES", MODULES),
        patch.object(e, "TEST", TEST),
        patch.object(e, "CLI", CLI),
    ):
        plan = old.validation_plan(private)
    result = []
    for spec in plan:
        argv = list(spec.argv)
        if spec.name in {
            "focused_pytest",
            "changed_module_coverage",
            "ruff_check",
            "ruff_format",
            "scoped_spec_coverage",
        }:
            argv.append(e.TEST)
        result.append(old.CommandSpec(spec.name, tuple(argv), spec.scope, spec.timeout_s))
    result.append(
        old.CommandSpec(
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
    return result


def validators(path: Path) -> list[old.CommandSpec]:
    """Use the unchanged terminal auditors with this producer's actual replay CLI."""
    with patch.object(old.qualified.e, "CLI", CLI):
        return old.qualified.validators(path)


def build(
    data: Json, result: Json, raw: Path, receipts: list[Json], duration: float, fixture: bool
) -> Json:
    """Current calls choose the substrate; planned model identity earns no credit."""
    value = old.build(data, result, raw, receipts, duration, fixture)
    value.update(
        experiment_id=8236,
        experiment=8236,
        task_id=TASK,
        milestone="2026.10.712",
        schema="carnot.v712.qualified-concurrency-canary.v1",
        coverage_statement_counts=data.get("coverage_statement_counts", {}),
        request_rows_path=str(raw / "request_rows.json"),
        protocol_sha256=data.get("protocol_sha256"),
        planned_model_specs=e.MODEL_SPECS,
        current_protocol_observation=data.get("current_protocol_observation"),
    )
    counts = value["model_invocation_counts"]
    if not counts["model_loads_attempted"]:
        value["inference_substrate_class"] = "no_model_load"
    value["declared_live_substrate"] = (
        "live_llm_inference" if counts["generation_calls_attempted"] else None
    )
    value["declared_live_substrate_class"] = (
        "model_bounded_generation" if counts["generation_calls_attempted"] else None
    )
    for row in value["rows"]:
        row["response_id"] = next(
            (
                r["result"].get("id")
                for r in result.get("rows", [])
                if r["request_id"] == row["request_id"]
            ),
            None,
        )
    value["field_principles"].update(
        {
            k: "Bind current requests and measured owned coverage; planned bytes imply no model work."
            for k in value
            if k not in value["field_principles"]
        }
    )
    return value


def replay(path: Path) -> bool:
    """Reopen primitive custody and recompute the public claims in a fresh process."""
    try:
        value = json.loads(path.read_text())
        validate_primary(value, path)
        if value["reproducibility_checksum"] != old.checksum(value):
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
        result = json.loads(Path(saved["result_path"]).read_text())
        for row in result.get("rows", []):
            journal = e.recorder.Journal(Path(row["journal_path"]))
            terminal = next(
                r
                for r in journal.events
                if r["request_id"] == row["request_id"] and r["event"] == "terminal"
            )
            clocks = row["clocks"]
            if (
                terminal["status"] != row["status"]
                or terminal["result"] != row["result"]
                or not (
                    clocks["issue"]
                    == terminal["issued_monotonic_ns"]
                    <= clocks["queue"]
                    <= clocks["end"]
                    <= terminal["observed_monotonic_ns"]
                    <= clocks["durability"]
                )
            ):
                return False
        rebuilt = build(
            data,
            result,
            Path(saved["data_path"]).parent,
            value["validation_receipts"],
            value["duration_s"],
            saved["fixture"],
        )
        return all(value[k] == rebuilt[k] for k in rebuilt if k != "raw_shard_hashes")
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
