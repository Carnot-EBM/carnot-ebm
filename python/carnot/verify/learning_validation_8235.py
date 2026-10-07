"""REQ-REPORT-8235: qualify execution without changing or rerunning scientific H2.

The historical primary remains disqualified. Current tests must reach its two
missed failure statements and recover real exit73 children before another task
may request natural learning. This audit reads cached evidence and loads no LLM.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.verify import delayed_utility_execution_8225 as legacy
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME = "experiment_8235_v712_learning_validation"
TASK = "exp8235-learning-validation"
MODULE = "python/carnot/verify/learning_validation_8235.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_learning_validation_8235.py"
OWNED = [*legacy.OWNED, MODULE, CLI]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
BINDINGS = "openspec/change-proposals/v712-learning-execution-bindings.json"
UPSTREAM = "results/" + legacy.NAME + ".json"
ORIGINAL_SHA256 = "sha256:fa51388af179d1c28880a484da5c87ddea6e7d36d80690f3f05d5888da6d5de6"
reference = legacy.reference


def run_check(root: Path, spec: Json, private: Path, raw: Path, *, heartbeat_s: float = 20) -> Json:
    """Archive private failure controls after real exit while keeping fixtures outside results."""
    receipt = legacy.run_check(root, spec, private, raw, heartbeat_s=heartbeat_s)
    if spec["name"] == "owned_unit_and_private_CLI":
        receipt["control_evidence"] = []
        for index, source in enumerate(
            sorted((private / "control_reports").glob("*-8235-control.json"))
        ):
            target = raw / "controls" / f"{index}-{sha256_file(source).split(':')[1]}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            receipt["control_evidence"].append(reference(target))
    return receipt


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so the operator can distinguish bounded checks from a stalled child."""
    print(f"[exp8235] phase={phase} completed={completed} pending={pending}", flush=True)


def failure_line_map() -> dict[str, int]:
    """Find actual statements through the AST because historical line numbers can drift."""
    tree = ast.parse((ROOT / legacy.MODULE).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    handler = next(
        node for node in ast.walk(functions["measure"]) if isinstance(node, ast.ExceptHandler)
    )
    schema = next(node for node in ast.walk(handler) if isinstance(node, ast.Expr))
    receipt_loop = next(
        node
        for node in ast.walk(functions["replay"])
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "receipt"
    )
    mismatch = next(node for node in ast.walk(receipt_loop) if isinstance(node, ast.Return))
    return dict(
        authenticated_schema_failure=schema.lineno, replay_log_hash_mismatch=mismatch.lineno
    )


def measure(root: Path, raw: Path, **kwargs: Any) -> Json:
    """Authenticate the failed run and original bytes without opening new trajectory labels."""
    start, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(checks=[], refs=[], owned_failure="", original_failure_receipt={}, rows=[])
    gate, bind = legacy.k.frozen.gate, legacy.k.frozen.bind
    progress("before_preconditions")
    try:
        with TemporaryDirectory(prefix="carnot-8235-probe-") as directory:
            probe = Path(directory) / "probe"
            probe.write_bytes(b"private writable scratch")
            gate(
                work,
                probe,
                "private_scratch_writable",
                True,
                probe.read_bytes() == b"private writable scratch",
            )
        for name in ["python", "coverage", "pytest", "ruff", "mypy"]:
            gate(
                work,
                ROOT / ".venv/bin" / name,
                "runtime_" + name,
                True,
                (ROOT / ".venv/bin" / name).is_file(),
            )
        gate(work, raw, "storage_available", True, shutil.disk_usage(raw).free > 32 * 1024 * 1024)
        path = root / UPSTREAM
        gate(work, path, "exists", True, True if path.is_file() else None)
        value = json.loads(path.read_bytes())
        gate(work, path, "experiment_id", 8225, value.get("experiment_id"))
        gate(
            work,
            path,
            "honest_verdict",
            "complete_disqualified_owned_validation",
            value.get("honest_verdict"),
        )
        gate(
            work,
            path,
            "reproducibility_checksum",
            value["reproducibility_checksum"],
            canonical_hash(
                {key: item for key, item in value.items() if key != "reproducibility_checksum"}
            ),
        )
        bind(work, path, ORIGINAL_SHA256, raw)
        receipt = next(r for r in value["validation_receipts"] if r["name"] == "coverage_report")
        work["original_failure_receipt"] = receipt
        gate(work, path, "coverage_report.actual_exit", 2, receipt["actual_exit"])
        for prefix in ["stdout", "stderr"]:
            bind(work, Path(receipt[prefix + "_path"]), receipt[prefix + "_sha256"], raw)
        coverage_ref = next(
            r for r in value["validation_receipts"] if r["name"] == "coverage_json"
        )["coverage_reference"]
        bind(work, Path(coverage_ref["path"]), coverage_ref["sha256"], raw)
        original = json.loads(Path(coverage_ref["path"]).read_bytes())
        gate(
            work,
            path,
            "coverage.missing_lines",
            [367, 595],
            original["files"][legacy.MODULE]["missing_lines"],
        )
        for ref in (
            value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
            + value["code_config_hashes"]
        ):
            bind(work, Path(ref["path"]), ref["sha256"], raw)
        for key in ["trajectory_path", "final_states_path"]:
            operand = Path(value[key])
            gate(work, operand, key + ".schema", True, bool(json.loads(operand.read_bytes())))
        bind(work, ROOT / BINDINGS, sha256_file(ROOT / BINDINGS), raw)
        work["historical_primary_sha256"] = sha256_file(path)
        work["historical_coverage_statement_counts"] = original["totals"]
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        if all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="input_schema",
                    path=str(root),
                    upstream=str(root),
                    hash=None,
                    artifact_field="input_schema",
                    op="==",
                    expected="valid",
                    observed=str(error),
                    passed=False,
                )
            )
    work.update(
        duration_s=(time.monotonic_ns() - start) / 1e9,
        clock=dict(
            started_monotonic_ns=start, ended_monotonic_ns=time.monotonic_ns(), started_wall_ns=wall
        ),
    )
    work["code_config_hashes"] = [reference(ROOT / name) for name in [*OWNED, TEST, BINDINGS]]
    atomic_json(raw / "measurement.json", work)
    progress("after_preconditions", len(work["checks"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """A finished private qualification earns readiness while scientific benefit stays zero."""
    failed = [c for c in work["checks"] if not c["passed"]]
    owned = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    coverage = raw / "logs/changed_code_coverage.json"
    counts = json.loads(coverage.read_bytes())["files"] if coverage.is_file() else {}
    covered = fixture or all(
        name in counts
        and counts[name]["summary"]["num_statements"] > 0
        and not counts[name]["missing_lines"]
        for name in OWNED
    )
    controls = [
        json.loads(Path(ref["path"]).read_bytes())
        for receipt in receipts
        for ref in receipt.get("control_evidence", [])
    ]
    recovery = fixture or (
        {control["case"] for control in controls}
        == {"authenticated_schema_failure", "stdout_receipt_mismatch", "stderr_receipt_mismatch"}
        and all(control["passed"] for control in controls)
        and all(
            [r["actual_exit"] for r in control["child_receipts"]] == [0, 73, 0, 73, 0]
            and all(r["passed"] for r in control["recovery"])
            for control in controls
            if "child_receipts" in control
        )
    )
    owned = bool(owned and (failed or covered and recovery))
    verdict = "disqualified" if not owned else "blocked" if failed else "null"
    rows = [
        dict(
            unit_id=r.get("name", "private_validation"),
            condition="owned_execution_check",
            metric="check_pass",
            numerator=int(r["passed"]),
            denominator=1,
            status="completed" if r["passed"] else "failed",
        )
        for r in receipts
    ]
    value = dict(
        work,
        experiment_id=8235,
        task_id=TASK,
        milestone="2026.10.712",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (
            Path(failed[0]["path"]).stem
            if verdict == "blocked"
            else "owned_validation"
            if verdict == "disqualified"
            else "learning_execution_qualified_benefit_unmeasured"
        ),
        verdict_class=verdict,
        learning_execution_ready_score=int(owned and not failed),
        required_checks_passed=owned,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=dict(
            kind="unchanged_historical_scale_intercept_and_ordered_probability_patches",
            current_audit_fits=0,
        ),
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=True,
        exposure_scope="private_execution_controls_and_exposed_historical_bytes",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        flagged_adversarial=False,
        acceptance_gates=dict(
            input_authentication=not failed,
            owned_validation=owned,
            owned_statement_coverage=covered,
            failure_controls_and_exact_recovery=recovery,
            scientific_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["checks"],
        random_seed=101,
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=[
            reference(p)
            for p in sorted(raw.rglob("*"))
            if p.is_file()
            and p.name
            in {
                "changed_code_coverage.json",
                "measurement.json",
                "validation_commands.json",
                "validation_receipts.json",
            }
        ],
        phase_spans=[
            dict(phase="historical_authentication", duration_s=work["duration_s"], **work["clock"])
        ]
        + [
            dict(
                phase=r.get("name", "private_validation"),
                duration_s=r.get("duration_s", 0),
                started_monotonic_ns=r.get("started_monotonic_ns"),
                ended_monotonic_ns=r.get("ended_monotonic_ns"),
                started_wall_ns=r.get("started_wall_ns"),
            )
            for r in receipts
        ],
        duration_s=work["duration_s"]
        + sum(r.get("duration_s", 0) for r in receipts)
        + work.get("global_health", {}).get("duration_s", 0),
        cited_upstream_artifacts=[
            dict(
                path=r["upstream_path"],
                sha256=r["sha256"],
                fields_imported=[
                    "historical failed coverage and authenticated original primitives"
                ],
            )
            for r in work["refs"]
        ],
        covered_failure_paths=failure_line_map(),
        coverage_statement_counts={name: entry["summary"] for name, entry in counts.items()},
        execution_bindings_path=BINDINGS,
        scientific_protocol_sha256=legacy.qualified.PIN,
        fixture_mode=fixture,
        historical_primary_verdict="complete_disqualified_owned_validation",
        claim_scope="Private execution qualification only. No learning benefit or probability improvement established.",
        methodology_note="Authenticate historical failures and unchanged scientific methods; current unit, CLI and child coverage and recovery qualify execution. No model load or new natural trajectory is requested. Private oracle controls do not establish independent generalization.",
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        key: "Bind actual execution and missing evidence; private qualification supplies no scientific benefit."
        for key in value
    }
    value["field_principles"].update(
        learning_execution_ready_score="Only current complete owned coverage and exact recovery authorize natural execution.",
        original_failure_receipt="Retain the original exit2 without relabeling scientific H2.",
        coverage_statement_counts="Current unit, CLI and hard-exit child statements must all be measured.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Authenticate primitive bytes and reduce again so a rehashed headline cannot pass."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if (
                sha256_file(Path(ref["path"])) != ref["sha256"]
                or "upstream_path" in ref
                and sha256_file(Path(ref["upstream_path"])) != ref["sha256"]
            ):
                return False
        for receipt in value["validation_receipts"]:
            for ref in receipt.get("control_evidence", []):
                if sha256_file(Path(ref["path"])) != ref["sha256"]:
                    return False
            for prefix in ["stdout", "stderr"]:
                if (
                    prefix + "_path" in receipt
                    and sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]
                ):
                    return False
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        return build(
            work, raw, value["validation_receipts"], fixture=value["fixture_mode"]
        ) == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze kernel, restored wrapper, real recovery, failure controls and consumers together."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = legacy.BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("[run]", "[run]\npatch = _exit")
        + "[report]\nexclude_lines =\ninclude =\n"
        + "".join("    " + str(ROOT / name) + "\n" for name in OWNED)
    )
    for spec in specs["commands"]:
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["argv"].insert(1, "CARNOT_8235_CONTROL_DIR=" + str(private / "control_reports"))
            spec["argv"] += [
                legacy.TEST,
                "tests/python/test_utility_kernel_8221.py",
                "--basetemp=" + str(private / "owned_pytest"),
            ]
            spec["deadline_s"] = 900
        if spec["name"] == "consumer_and_E2E015_019":
            begin = spec["argv"].index("tests/python/test_development_methods_8098.py")
            spec["argv"][begin:] = [
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_hard_exit_learning_qualification_8206.py",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "consumer_pytest"),
            ]
            spec["deadline_s"] = 900
        if spec["name"] == "strict_mypy":
            spec["argv"] = [
                a.replace("--follow-imports=silent", "--follow-imports=skip") for a in spec["argv"]
            ]
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    specs["repository_health"]["deadline_s"] = 180
    specs["repository_health"]["argv"][:0] = [
        "/usr/bin/env",
        "COVERAGE_FILE=" + str(private / "repository_health.coverage"),
    ]
    return specs


def main(argv: list[str] | None = None) -> int:
    """Use the qualified supervisor so publication follows bounded normally exited validation."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(argv))
