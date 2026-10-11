"""REQ-REPORT-8347: historical fixture custody and current readiness stay separate.

Old policy bytes explain old measurements. They cannot authorize new work, so
the current manifest and current task authority are checked independently.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import local_qualification_8333 as prior
from carnot.reporting import local_update_isolation_8306 as base
from carnot.reporting import v718_contract_replay as authority
from carnot.reporting import v718_replay_history as history
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v710_contract_replay import require_reference
from carnot.verify import hard_exit_learning_qualification_8206 as consumer
from carnot.verify import local_qualification_8333 as numeric
from carnot.verify import local_update_isolation_8306 as kernel
from carnot.verify.methods_stream_custody_8111 import Custody

Json = dict[str, Any]
ROOT = consumer.ROOT
NAME = "experiment_8347_v720_local_consumer_qualification"
TASK, MILESTONE = "exp8347-local-consumer-qualification", "2026.10.720"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_local_consumer_qualification_8347.py"
MODULE = "python/carnot/reporting/local_consumer_qualification_8347.py"
RUNNER = "python/carnot/reporting/local_consumer_execution_8347.py"
OWNED = [MODULE, RUNNER, CLI]
MODEL_SPECS: list[Json] = []
HISTORICAL_POLICY_HASH = "sha256:004acbfa4adb4ea1ef03b2cc58f3679827c34e5b017318e2f1a2a62bb7951ba3"
HISTORICAL_PRIMARY_HASH = "sha256:3ff40ac3a82fc4a9f8e42f626c5d19db10bdfd5546d7aa0e998353a4bce87fd8"
HISTORICAL_CLOSURE_HASH = "sha256:a40ae9ed2d5f5c72d5a5e55fe761f463597faedfd2a26a9e945cf12cabe284ab"
BASE_CUSTODY = consumer.legacy.engine.methods.Custody


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed work so the supervisor sees actual phase boundaries."""
    print(f"[exp8347] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze_historical(private: Path) -> Path:
    """Copy the entire authenticated old input closure rather than fabricate receipts."""
    primary = ROOT / "results" / (consumer.NAME + ".json")
    if sha256_file(primary) != HISTORICAL_PRIMARY_HASH:
        raise ValueError("historical_primary_sha256")
    value = json.loads(primary.read_bytes())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    report = read_bound_sidecar(primary, Path(terminal["publication"]["sidecar_path"]))
    if report["report"]["passed"] is not True:
        raise ValueError("historical_terminal")
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    refs = []
    for index, ref in enumerate(value["source_artifact_hashes"]):
        source = Path(ref["snapshot_path"])
        if sha256_file(source) != ref["sha256"]:
            raise ValueError("historical_operand_sha256")
        saved = private / (str(index) + ".bin")
        saved.write_bytes(source.read_bytes())
        refs.append(dict(ref, snapshot_path=str(saved)))
    authority = dict(
        version="exp8206-v709-historical-operands-v1",
        milestone=value["milestone"],
        primary_path=str(primary),
        primary_sha256=HISTORICAL_PRIMARY_HASH,
        terminal_sidecar_path=value["terminal_validation_sidecar_path"],
        terminal_sidecar_sha256=sha256_file(Path(value["terminal_validation_sidecar_path"])),
        validator_path=terminal["publication"]["sidecar_path"],
        validator_sha256=sha256_file(Path(terminal["publication"]["sidecar_path"])),
        operands=refs,
        grants_current_authority=False,
    )
    path = private / "historical_fixture_manifest.json"
    atomic_json(path, authority)
    return path


class HistoricalCustody(Custody):
    """Resolve every historical input explicitly and retain unchanged gate checks."""

    def __init__(self, raw: Path, closure: Path):
        super().__init__(raw)
        manifest = json.loads(closure.read_bytes())
        self.require(
            closure,
            "historical_operand_closure",
            HISTORICAL_CLOSURE_HASH,
            canonical_hash(
                [dict(path=r["path"], sha256=r["sha256"]) for r in manifest["operands"]]
            ),
        )
        self.saved = {r["path"]: r for r in manifest["operands"]}

    def bind(self, path: Path, digest: str | None = None) -> Json:
        """Missing private bytes fail instead of falling back to mutable live inputs."""
        ref = self.saved.get(str(path))
        self.require(path, "historical_operand_present", True, ref is not None)
        assert ref is not None
        saved = Path(ref["snapshot_path"])
        self.require(path, "resource_exists", True, saved.is_file())
        self.require(path, "sha256", digest or ref["sha256"], sha256_file(saved))
        if ref not in self.refs:
            self.refs.append(ref)
        return dict(ref)


@contextmanager
def historical_operands(closure: Path) -> Iterator[None]:
    """Adapt only input resolution; the real consumer still measures and validates."""
    with patch.object(
        consumer.legacy.engine.methods,
        "Custody",
        lambda raw: HistoricalCustody(raw, closure),
    ):
        yield


def current_policy(root: Path) -> Json:
    """Current retirement is evaluated from live bytes, outside historical adapters."""
    path = root / "ops/exclusion_manifest.yaml"
    policy = yaml.safe_load(path.read_bytes())
    retired = [
        row
        for key in ["retired_experiments", "retired_extras"]
        for row in policy.get(key, [])
        if row.get("experiment_id") in [8347, TASK] or TASK in str(row.get("experiment_scope", ""))
    ]
    return dict(
        upstream="current_retirement_policy",
        path=str(path),
        hash=sha256_file(path),
        artifact_field="experiment8347_not_retired",
        op="==",
        expected=[],
        observed=retired,
        passed=not retired,
    )


def measure(root: Path, raw: Path) -> Json:
    """Recompute unchanged primitives, retaining the old failed consumer as evidence."""
    began = time.monotonic()
    progress("preconditions_before")
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    work: Json = dict(
        checks=[],
        refs=[],
        historical=[],
        rows=[],
        states=[],
        costs=[],
        crashes=[],
        recovered_states=[],
        fixture=False,
        crash_parity=False,
        numeric={},
        false_zero={},
        negative=kernel.negative_control(),
        audit=kernel.numeric_audit(),
        duration_s=0.0,
        phase_spans=[],
        first_failed_operand={},
        historical_fixture_manifest={},
    )
    atomic_json(raw / "protocol.json", kernel.manifest())
    work["protocol_reference"] = base.reference(raw / "protocol.json")
    try:
        for name in ["python", "pytest", "coverage", "ruff", "mypy"]:
            if not os.access(ROOT / ".venv/bin" / name, os.X_OK):
                raise OSError("missing executable: " + name)
        if shutil.disk_usage(raw).free < 1024**3:
            raise OSError("private storage below 1GiB")
        policy = current_policy(root)
        work["checks"].append(policy)
        if not policy["passed"]:
            raise ValueError("current_retirement")
        with patch.object(history, "design", lambda r, m: r / authority.DESIGN):
            contract = authority.authority(root, raw / "authority", MILESTONE)
        task = next(t for t in contract["tasks"] if t["id"] == TASK)
        if (
            not contract["activated"]
            or task["deliverable"] != f"results/{NAME}.json"
            or task["gated_on"] != []
            or task["MODEL_SPECS"] != []
        ):
            raise ValueError("current_task_authority")
        work["current_authority"] = contract
        for name in [
            authority.DESIGN,
            authority.ACTIVE,
            authority.PROTOCOL,
            "ops/exclusion_manifest.yaml",
        ]:
            work["refs"].append(base.reference(root / name))
        path = root / "results" / (prior.NAME + ".json")
        imported = json.loads(path.read_bytes())
        require_reference(
            dict(
                path=str(path),
                sha256="sha256:57c88cbefc2798a7a29fdd739f2d58189826bab166d9dc48b7141f0e124b50f0",
            )
        )
        errors: list[Json] = []
        authenticated = authority.authenticate(path, raw, work["refs"], errors)
        if not authenticated or errors:
            raise ValueError(str(errors))
        failed_consumer = next(
            r for r in imported["validation_receipts"] if r["name"] == "consumer_and_E2E020_021"
        )
        for stream in ["stdout", "stderr"]:
            ref = dict(
                path=failed_consumer[stream + "_path"], sha256=failed_consumer[stream + "_sha256"]
            )
            require_reference(ref)
            work["refs"].append(ref)
        require_reference(imported["measurement_reference"])
        require_reference(imported["protocol_reference"])
        reused = json.loads(Path(imported["measurement_reference"]["path"]).read_bytes())
        protocol = json.loads(Path(imported["protocol_reference"]["path"]).read_bytes())
        if protocol != kernel.manifest():
            raise ValueError("frozen_48_by_64_protocol")
        for receipt in reused["crashes"]:
            for stream in ["stdout", "stderr"]:
                require_reference(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        for key in [
            "rows",
            "states",
            "crashes",
            "recovered_states",
            "historical",
            "negative",
            "audit",
            "crash_parity",
        ]:
            work[key] = reused[key]
        work["refs"].extend(
            [
                base.reference(path),
                imported["measurement_reference"],
                imported["protocol_reference"],
            ]
        )
        work["historical_disposition"] = dict(
            verdict_class=imported["verdict_class"], readiness_promoted=False
        )
        work["historical_findings"] = imported["adversarial_findings"]
        progress("before_independent_numeric_benchmark")
        work["numeric"] = numeric.audit(work, protocol)
        work["false_zero"] = numeric.controls(work, protocol)
        work["crash_parity"] = all(
            numeric.durable(saved) == numeric.durable(stored["arms"]["indexed"])
            for saved, stored in zip(work["recovered_states"], work["states"], strict=True)
        )
        progress("after_independent_numeric_benchmark", 48, 0)
        closure = freeze_historical(raw / "historical_operands")
        work["historical_fixture_manifest"] = base.reference(closure)
        manifest = json.loads(closure.read_bytes())
        work["refs"].extend(
            [base.reference(Path(ref["snapshot_path"])) for ref in manifest["operands"]]
        )
        work["refs"].append(base.reference(closure))
        blocked = consumer.measure(ROOT, raw / "reproduced_operand", fixture=True)
        work["first_failed_operand"] = next(
            r for r in blocked["gate_check_summary"] if not r["passed"]
        )
        work["refs"].append(base.reference(raw / "reproduced_operand/measurement.json"))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        work["checks"].append(
            dict(
                upstream="exp8333_and_current_authority",
                path=str(getattr(error, "filename", None) or root),
                hash=None,
                artifact_field="authenticated_external_operands",
                op="==",
                expected=True,
                observed=False if isinstance(error, FileNotFoundError) else str(error),
                passed=False,
            )
        )
    work["code_config_hashes"] = [
        base.reference(ROOT / name)
        for name in [
            *OWNED,
            TEST,
            consumer.TEST,
            *prior.OWNED,
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/v709_execution.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/experiment_template.py",
        ]
    ]
    work.update(
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(phase="authenticate_and_recompute", start_s=0, duration_s=time.monotonic() - began)
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("preconditions_after", len(work["rows"]), 48 - len(work["rows"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Consumer and coverage receipts gate mechanics; they grant no learning benefit."""
    value = base.build(work, raw, receipts)
    coverage_ref = work.get("owned_coverage_reference")
    report = json.loads(Path(coverage_ref["path"]).read_bytes()) if coverage_ref else {}
    checks = dict(
        numerical=work["numeric"].get("passed", False),
        false_zero=work["false_zero"].get("false_zero_rejected", False),
        recovery=work["crash_parity"] and all(r["passed"] for r in work["crashes"]),
        consumer_assertions=all(
            any(r.get("name") == n and r["passed"] for r in receipts)
            for n in ["first_three_consumers", "consumer_and_E2E020_021"]
        ),
        owned_coverage=report.get("totals", {}).get("percent_covered") == 100
        and len(report.get("files", {})) == len(OWNED),
        owned_checks=bool(receipts) and all(r["passed"] for r in receipts),
    )
    external = any(not r["passed"] for r in work["checks"])
    ready = int(not external and all(checks.values()))
    kind = (
        "disqualified"
        if not checks["owned_checks"] or (not external and not ready)
        else "blocked"
        if external
        else "circular_positive"
    )
    value.update(
        experiment_id=8347,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261009",
        honest_verdict="complete_" + kind + "_local_consumer_qualification",
        verdict_class=kind,
        local_kernel_ready_score=ready,
        independent_count=0,
        random_seed=7208347,
        acceptance_gates=checks,
        independent_dense_reference=work["numeric"],
        false_zero_control=work["false_zero"],
        first_failed_operand=work["first_failed_operand"],
        historical_fixture_manifest=work["historical_fixture_manifest"],
        historical_disposition=work.get("historical_disposition", {}),
        adversarial_findings=work.get("historical_findings", []),
        finding_dispositions=work.get("finding_dispositions", []),
        consumer_qualification=[r for r in receipts if "consumer" in r.get("name", "")],
        cited_upstream_artifacts=[
            dict(r, fields_imported=["authenticated historical primitives; no outcome promotion"])
            for r in work["refs"]
        ],
        execution_manifest_reference=work.get("execution_manifest_reference"),
    )
    value["field_principles"].update(
        {
            key: "Bind current qualification to explicit bytes without promoting exposed historical science."
            for key in value
            if key not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        local_kernel_ready_score="One requires independent dense arithmetic, durable recovery, unchanged consumers, all checks and 100 percent newly owned statements.",
        first_failed_operand="Preserve the reproduced historical digest mismatch with upstream, path, field, operator, expected and observed operands.",
        historical_fixture_manifest="An exact versioned 133-operand historical closure supplies old bytes without granting current retirement authority.",
        consumer_qualification="Retain unchanged assertion command vectors, actual exits, deadlines, clocks and both stream hashes.",
        independent_count="Forty-eight constructed trajectories and repeated timings contribute zero independent natural sources.",
        adversarial_findings="Preserve original informational findings; resolution requires independent recomputation and deliberate-error rejection.",
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute causal states so rehashed summaries cannot create readiness."""
    progress("cold_replay_before")
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if canonical_hash(value) != checksum:
            return False
        value["reproducibility_checksum"] = checksum
        for ref in (
            value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
            + value["code_config_hashes"]
        ):
            require_reference(ref)
        raw = Path(value["measurement_reference"]["path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        protocol = json.loads(Path(work["protocol_reference"]["path"]).read_bytes())
        if protocol != kernel.manifest():
            return False
        if work["states"]:
            if work["numeric"] != numeric.audit(work, protocol) or work[
                "false_zero"
            ] != numeric.controls(work, protocol):
                return False
            for index, (stored, trajectory, saved) in enumerate(
                zip(work["states"], protocol["trajectories"], work["recovered_states"], strict=True)
            ):
                for arm in kernel.ARMS:
                    state = kernel.initial(trajectory, arm)
                    for slot in range(72):
                        kernel.issue(state, slot)
                        kernel.release(trajectory, state, slot, arm)
                    if numeric.durable(state) != numeric.durable(stored["arms"][arm]):
                        return False
                if numeric.durable(saved) != numeric.durable(stored["arms"]["indexed"]):
                    return False
                progress("causal_replay", index + 1, 48 - index - 1)
            closure = work["historical_fixture_manifest"]
            require_reference(closure)
            manifest = json.loads(Path(closure["path"]).read_bytes())
            for ref in manifest["operands"]:
                require_reference(dict(path=ref["snapshot_path"], sha256=ref["sha256"]))
            for receipt in work["crashes"] + value["validation_receipts"]:
                for stream in ["stdout", "stderr"]:
                    if receipt.get(stream + "_path"):
                        require_reference(
                            dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                        )
        passed = build(work, raw, value["validation_receipts"]) == value
        progress("cold_replay_after", int(passed), 0)
        return bool(passed)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
