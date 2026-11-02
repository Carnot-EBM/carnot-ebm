"""REQ-REPORT-8333: qualify current arithmetic without upgrading old failures."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.reporting import local_update_isolation_8306 as base
from carnot.reporting import v718_contract_replay as custody
from carnot.reporting import v718_replay_history as history
from carnot.reporting import v718_replay_runner as findings
from carnot.reporting import v719_contract_replay as current
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child
from carnot.reporting.v710_contract_replay import snapshot, require_reference
from carnot.verify import local_qualification_8333 as numeric
from carnot.verify import local_update_isolation_8306 as k

Json = dict[str, Any]
ROOT = k.ROOT
NAME = "experiment_8333_v719_local_evidence_qualification"
TASK, MILESTONE = "exp8333-local-evidence-qualification", "2026.10.719"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v719_local_qualification_8333.py"
OWNED = [
    "python/carnot/verify/local_qualification_8333.py",
    "python/carnot/reporting/local_qualification_8333.py",
    "python/carnot/reporting/local_qualification_execution_8333.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
reference = base.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so the supervisor can detect stalls during owned work."""
    print(f"[exp8333] phase={phase} completed={completed} pending={pending}", flush=True)


def consumer_controls(candidate: Path, proof: Json, raw: Path) -> Json:
    """Qualify the unchanged consumer against false proofs and unknown severities."""
    valid = findings.audit(candidate, raw / "valid", proof)
    report = valid["report"]
    unknown = deepcopy(report)
    unknown["reports"][0]["flags"][0]["severity"] = "unknown"
    false = history.consume(
        report, candidate, 1, dict(recomputed=False, deliberate_error_rejected=True)
    )
    rejected = history.consume(unknown, candidate, 1, proof)
    errors = history.consume(report, candidate, 2, proof)
    malformed = history.consume(dict(reports=None), candidate, 1, proof)
    return dict(
        passed=valid["passed"]
        and not any(r["passed"] for r in [false, rejected, errors, malformed]),
        valid=valid,
        false_zero=false,
        unknown_severity=rejected,
        process_error=errors,
        malformed=malformed,
    )


def measure(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Authenticate independent inputs before executing the frozen constructed plan."""
    began = time.monotonic()
    progress("preconditions_before")
    work = base.authenticate(root, raw)
    work.update(
        failures=[r for r in work["checks"] if not r["passed"]],
        original={},
        rows=[],
        states=[],
        costs=[],
        crashes=[],
        recovered_states=[],
        fixture=fixture,
        audit=k.numeric_audit(),
        negative=k.negative_control(),
        numeric={},
        false_zero={},
        original_finding={},
        consumer_controls={},
        crash_parity=False,
    )
    try:
        authority = current.authority(root, raw / "authority")
        own = next(t for t in authority["tasks"] if t["id"] == TASK)
        matched = (
            authority["activated"]
            and own["deliverable"] == f"results/{NAME}.json"
            and own["gated_on"] == []
            and own["MODEL_SPECS"] == []
        )
        gate = dict(
            custody.failure(root / current.ACTIVE, "current_task_authority", True, bool(matched)),
            passed=bool(matched),
        )
        work["checks"].append(gate)
        if not matched:
            work["failures"].append(gate)
        work["authority"] = authority
        for path in [current.DESIGN, current.ACTIVE, current.PROTOCOL]:
            work["refs"].append(snapshot(root / path, raw / "inputs", Path(path).stem))
        require_reference(dict(work["refs"][-1], sha256=custody.base.PIN))
        original = custody.authenticate(
            root / "results" / (k.NAME + ".json"), raw, work["refs"], work["failures"]
        )
        work["original"] = original
        if original:
            for ref in [original["measurement_reference"], original["protocol_reference"]]:
                saved = snapshot(Path(ref["path"]), raw / "original", Path(ref["path"]).stem)
                require_reference(dict(saved, sha256=ref["sha256"]))
                work["refs"].append(saved)
            old_work, old_protocol = [
                json.loads(Path(ref["snapshot_path"]).read_bytes()) for ref in work["refs"][-2:]
            ]
            work["historical_numeric"] = numeric.audit(old_work, old_protocol)
            work["historical_controls"] = numeric.controls(old_work, old_protocol)
            work["consumer_controls"] = consumer_controls(
                root / "results" / (k.NAME + ".json"),
                work["historical_controls"],
                raw / "finding_controls",
            )
            work["original_finding"] = work["consumer_controls"]["valid"]
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        work["failures"].append(
            custody.failure(
                root / current.DESIGN, "authenticated_external_operands", True, str(error)
            )
        )
    progress("preconditions_after")
    protocol = k.manifest()
    if fixture:
        protocol["trajectories"] = protocol["trajectories"][:1]
    atomic_json(raw / "protocol.json", protocol)
    if not work["failures"]:
        progress("before_correctness_benchmark")
        for index, trajectory in enumerate(protocol["trajectories"]):
            states = {
                arm: k.execute(trajectory, arm, raw / "baseline" / (trajectory["id"] + arm))
                for arm in k.ARMS
            }
            work["states"].append(dict(trajectory_id=trajectory["id"], arms=states))
            work["rows"].append(
                base.reduce_states(
                    trajectory, states["full"], states["indexed"], states["truncated"]
                )
            )
            progress("correctness", index + 1, len(protocol["trajectories"]) - index - 1)
        progress("after_correctness_benchmark")
        work["numeric"] = numeric.audit(work, protocol)
        work["false_zero"] = numeric.controls(work, protocol)
        checkpoint = raw / "recovered"
        prefix = [str(ROOT / ".venv/bin/python")]
        if os.environ.get("COVERAGE_RCFILE"):
            prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
        args = [
            *prefix,
            str(ROOT / CLI),
            "--worker",
            str(raw / "protocol.json"),
            "--checkpoint",
            str(checkpoint),
            "--cohort-count",
            str(len(protocol["trajectories"])),
        ]
        for boundary in [31, 63, -1]:
            receipt = child(
                f"recovery_{boundary}",
                [*args, "--crash", str(boundary)],
                raw / "child_logs",
                deadline=180,
                expected=-9 if boundary >= 0 else 0,
                heartbeat=20,
            )
            saved = [k.load(t, checkpoint / (t["id"] + ".json")) for t in protocol["trajectories"]]
            receipt["boundary"] = boundary
            receipt["pending_counts"] = [len(s["pending"]) for s in saved]
            receipt["issued_counts"] = [len(s["issues"]) for s in saved]
            work["crashes"].append(receipt)
        work["recovered_states"] = saved
        work["crash_parity"] = all(
            numeric.durable(s) == numeric.durable(t["arms"]["indexed"])
            for s, t in zip(saved, work["states"], strict=True)
        )
    work.update(
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authentication_arithmetic_recovery",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        derivatives=numeric.derivatives(),
        feedback_control=numeric.feedback_control(),
        protocol_reference=reference(raw / "protocol.json"),
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                *base.OWNED,
                "python/carnot/reporting/v718_replay_history.py",
                "python/carnot/reporting/v718_replay_runner.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/roadmap_contract.py",
                "python/carnot/reporting/v709_execution.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "scripts/experiment_template.py",
                "ops/exclusion_manifest.yaml",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress(
        "measurement_complete", len(work["rows"]), len(protocol["trajectories"]) - len(work["rows"])
    )
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Reuse the established schema while adding independent qualification gates."""
    value = base.build(work, raw, receipts)
    coverage_ref = work.get("owned_coverage_reference")
    covered = False
    if coverage_ref:
        report = json.loads(Path(coverage_ref["path"]).read_bytes())
        covered = report["totals"]["percent_covered"] == 100 and len(report["files"]) == len(OWNED)
    checks = dict(
        numerical=work["numeric"].get("passed", False),
        false_zero=work["false_zero"].get("false_zero_rejected", False),
        historical_arithmetic=work.get("historical_numeric", {}).get("passed", False),
        finding_consumer=work["consumer_controls"].get("passed", False),
        derivatives=work["derivatives"]["passed"],
        feedback=work["feedback_control"]["passed"],
        recovery=work["crash_parity"] and all(r["passed"] for r in work["crashes"]),
        owned_coverage=covered,
        owned_checks=bool(receipts) and all(r["passed"] for r in receipts),
    )
    ready = int(not work["failures"] and all(checks.values()))
    kind = (
        "disqualified"
        if any(not r["passed"] for r in receipts) or (not work["failures"] and not ready)
        else "blocked"
        if work["failures"]
        else "circular_positive"
    )
    reason = (
        work["failures"][0]["artifact_field"]
        if kind == "blocked"
        else "local_evidence_qualification"
    )
    value.update(
        experiment_id=8333,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261009",
        honest_verdict="complete_" + kind + "_" + reason,
        verdict_class=kind,
        local_kernel_ready_score=ready,
        flagged_adversarial=any(
            not row["resolved"] for row in work["original_finding"].get("dispositions", [])
        ),
        adversarial_findings=work["original_finding"].get("findings", []),
        finding_dispositions=work["original_finding"].get("dispositions", []),
        independent_dense_reference=work["numeric"],
        false_zero_control=work["false_zero"],
        numeric_audit=dict(
            work["audit"],
            independent_derivatives=work["derivatives"],
            out_of_order_feedback=work["feedback_control"],
        ),
        crash_replay_rows=work["crashes"],
        acceptance_gates=checks,
        gate_check_summary=work["checks"]
        + [r for r in work["failures"] if r not in work["checks"]],
        independent_count=0,
        random_seed=7198333,
        historical_disposition=dict(
            verdict_class=work["original"].get("verdict_class"), readiness_promoted=False
        ),
        consumer_qualification=work["consumer_controls"],
        historical_numeric_audit=work.get("historical_numeric", {}),
        cited_upstream_artifacts=[
            dict(
                ref,
                fields_imported=[
                    "immutable protocol, primitive states or historical terminal disposition"
                ],
            )
            for ref in work["refs"]
        ],
        execution_manifest_reference=work.get("execution_manifest_reference"),
        invocation_argv=work.get("invocation_argv", []),
    )
    value["field_principles"].update(
        {
            field: "Bind independent arithmetic, durable recovery and exact finding evidence; constructed success supplies no natural generalization."
            for field in value
            if field not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        flagged_adversarial="Unresolved findings quarantine current evidence; retain resolved original flags in adversarial_findings and finding_dispositions.",
        independent_count="Constructed trajectories and repeated arithmetic contain zero independent natural sources.",
        local_kernel_ready_score="One requires every numeric, recovery, consumer, owned coverage and validation gate; no scientific improvement is required.",
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold replay rebuilds causal state and rejects self-consistent primitive drift."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        value["reproducibility_checksum"] = checksum
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            require_reference(ref)
        raw = Path(value["measurement_reference"]["path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        protocol = json.loads(Path(work["protocol_reference"]["path"]).read_bytes())
        expected = k.manifest()
        if work["fixture"]:
            expected["trajectories"] = expected["trajectories"][:1]
        if (
            protocol != expected
            or work["derivatives"] != numeric.derivatives()
            or work["feedback_control"] != numeric.feedback_control()
        ):
            return False
        for stored, trajectory, recovered in zip(
            work["states"],
            protocol["trajectories"] if work["states"] else [],
            work["recovered_states"],
            strict=True,
        ):
            for arm in k.ARMS:
                state = k.initial(trajectory, arm)
                for slot in range(72):
                    k.issue(state, slot)
                    k.release(trajectory, state, slot, arm)
                if numeric.durable(state) != numeric.durable(stored["arms"][arm]):
                    return False
            if numeric.durable(recovered) != numeric.durable(stored["arms"]["indexed"]):
                return False
        if work["states"] and (
            work["numeric"] != numeric.audit(work, protocol)
            or work["false_zero"] != numeric.controls(work, protocol)
        ):
            return False
        for receipt in [
            *value["validation_receipts"],
            *work["crashes"],
            *([work["original_finding"]["receipt"]] if work["original_finding"] else []),
        ]:
            for stream in ["stdout", "stderr"]:
                if (
                    receipt.get(stream + "_path")
                    and sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]
                ):
                    return False
        if work["original_finding"]:
            finding = work["original_finding"]
            policy = history.consume(
                finding["report"],
                Path(finding["report"]["reports"][0]["artifact"]),
                finding["receipt"]["exit_code"],
                work["historical_controls"],
            )
            if any(policy[key] != finding[key] for key in ["passed", "findings", "dispositions"]):
                return False
        return bool(build(work, raw, value["validation_receipts"]) == value)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
