"""REQ-VERIFY-8382: separate qualified custody from a repaired CUDA substrate.

A missing repair receipt is a complete observation of absence. It cannot
justify repeating the failed historical probes or claim semantic benefit.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.v710_contract_replay import snapshot
from carnot.reporting import v722_contract_methods as authority
from carnot.verify import runtime_closure_8368 as qualified
from carnot.verify import typed_runtime_8368 as typed
from carnot.reporting.v709_execution import child
from carnot.gpu_lease_phase_journal import GpuLease, LeaseError

Json = dict[str, Any]
ROOT = qualified.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8382_v722_runtime_evidence_delta",
    "exp8382-runtime-evidence-delta",
    "2026.10.722",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_runtime_evidence_delta_8382.py"
OWNED = [
    "python/carnot/verify/runtime_evidence_delta_8382.py",
    "python/carnot/verify/runtime_evidence_execution_8382.py",
    CLI,
]
UPSTREAM = "results/experiment_8368_v721_typed_runtime_closure.json"
PIN = "sha256:8d5810e06735b445b4b1bb6602f7263dff6b1e56d456f7e9b76ba1f9449eb323"
BASELINE_PIN = "sha256:d8a163de59b6da2d8ad261a277b50ac796370e3ddce9c9b0a547002890d7275b"
PROTOCOL_PINS = {
    "openspec/change-proposals/v717-local-learning-protocol.json": "sha256:853709123024de763e96dd688e819f0430205ae6d97d6561a2b95cca23b81c6f",
    "openspec/change-proposals/v721-deployment-protocol.json": "sha256:d4441f7619a1d349a4958038af93c36f2c4df97b31fd32278c4ea6def7c7b4eb",
}
MODEL_SPECS: list[Json] = []
old, checksum = qualified.old, qualified.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so a blocked task never looks like silent computation."""
    print(f"[exp8382] phase={phase} completed={completed} pending={pending}", flush=True)


def frontier(value: Json) -> int:
    """Use the old measured finish time; a newer file timestamp proves no repair."""
    return int(max(r["started_wall_ns"] + r["duration_s"] * 1e9 for r in value["phase_spans"]))


def delta(work: Json) -> list[Json]:
    """Reuse qualified causal comparisons while keeping every missing operand."""
    operands = deepcopy(dict(previous=work["baseline"], current=work["current"]))
    for item in operands.values():
        if "libraries" in item:
            item["libraries"] = [
                r
                for r in item["libraries"]
                if Path(r["path"]).name.startswith(("libcuda", "libnvidia"))
            ]
    rows = qualified.changes(operands)
    return [
        dict(
            r,
            arm="new_environment_receipt",
            absolute_metric=int(r["changed"]),
            missing_reason=None if r["available"] else "new_authenticated_operand_absent",
        )
        for r in rows
    ]


def historical_aliases(root: Path, raw: Path, source: Json) -> list[Json]:
    """Resolve old citations through authenticated producer snapshots.

    Current roadmap and source paths may have advanced. The pinned primary's
    own retained copies remain the authority for those historical bytes.
    """
    original = typed.authenticate
    value = json.loads(original(source).read_bytes())
    work = json.loads(original(value["primitive_reference"]).read_bytes())
    aliases = {
        r["sha256"]: r
        for r in typed.references(value) + typed.references(work)
        if r.get("snapshot_path")
    }

    def bind(ref: Json) -> Path:
        """Use a retained copy only for the exact digest declared by the producer."""
        return original(aliases.get(ref.get("sha256"), ref))

    with patch.object(typed, "authenticate", bind):
        return typed.closure(root, raw, [UPSTREAM])


def probe(current: Json, raw: Path) -> Json:
    """Lease one device and run only the direct driver context/copy child once."""
    lease: Any = None
    result: Json = dict(rows=[], checks=[], lease_receipt={}, cleanup_receipt={})
    try:
        lease = GpuLease.acquire(
            runtime_dir="/tmp/carnot-gpu-leases",
            task_id=TASK,
            device_uuid=current["permitted_uuid"],
            expected_model="no_model_load:cuda_byte_copy",
            vram_before_mb=0,
            ttl_s=120,
        )
        result["lease_receipt"] = lease.owner_receipt()
        lease.transition("admitted")
        receipt = child(
            "direct_driver_context_copy",
            [
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=" + current["permitted_uuid"],
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--cuda-probe",
                "driver",
                "--uuid",
                current["permitted_uuid"],
                "--library",
                current["driver_library"],
            ],
            raw,
            deadline=60,
            heartbeat=20,
            scope="external_cuda",
        )
        text = Path(receipt["stdout_path"]).read_text().splitlines()
        primitive = json.loads(text[-1]) if receipt["passed"] and text else dict(error="child_exit")
        result["rows"].append(dict(receipt=receipt, primitive=primitive))
    except (OSError, ValueError, KeyError, LeaseError) as error:
        result["checks"].append(old.gate(raw, "exclusive_direct_context_copy", True, str(error)))
    finally:
        if lease:
            lease.transition("terminal_blocked")
            result["cleanup_receipt"] = lease.release()
    return result


def measure(
    root: Path,
    raw: Path,
    receipt: Path | None = None,
    receipt_hash: str | None = None,
    *,
    fixture: bool = False,
) -> Json:
    """Seal old aliases before reading and admit only explicitly pinned new evidence."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        baseline={},
        current={},
        historical={},
        refs=[],
        failures=[],
        diagnostic=dict(rows=[], checks=[]),
        observation_receipts=[],
        fixture=fixture,
        root=str(root),
        receipt_reference=None,
        observed_wall_ns=time.time_ns(),
    )
    progress("task_authority_before")
    work["refs"].extend(
        snapshot(root / p, raw / "authority_inputs", str(i))
        for i, p in enumerate([authority.ACTIVE, authority.DESIGN])
    )
    private_root = raw / "frozen_task_authority"
    for ref in work["refs"]:
        if ref.get("exists"):
            target = private_root / Path(ref["path"]).relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(typed.authenticate(ref).read_bytes())
    work["authority"] = authority.authority(private_root, raw / "authority")
    work["failures"].extend(work["authority"]["gate_check_summary"])
    progress("historical_aliases_before")
    try:
        source = snapshot(root / UPSTREAM, raw / "sources", "qualified_primary")
        work["refs"].append(source)
        if source["sha256"] != PIN:
            raise ValueError("immutable_exp8368_primary")
        work["refs"].extend(historical_aliases(root, raw, source))
        historical = json.loads(typed.authenticate(source).read_bytes())
        baseline = dict(path=historical["runtime_binding_path"], sha256=BASELINE_PIN)
        work["baseline_reference"] = snapshot(
            typed.authenticate(baseline), raw / "sources", "baseline"
        )
        work["refs"].append(work["baseline_reference"])
        work["historical"] = historical
        work["baseline"] = json.loads(typed.authenticate(work["baseline_reference"]).read_bytes())
        work["historical_reference"] = source
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["failures"].append(
            old.gate(root / UPSTREAM, "historical_typed_closure", True, str(error))
        )
    progress("historical_aliases_after", len(work["refs"]), 0)
    for name, pin in PROTOCOL_PINS.items():
        ref = snapshot(root / name, raw / "protocols", Path(name).stem)
        work["refs"].append(ref)
        if ref["sha256"] != pin:
            work["failures"].append(old.gate(root / name, "immutable_protocol", pin, ref["sha256"]))
    path = receipt or root / "ops/cuda-runtime-repair-receipt.json"
    try:
        if receipt is None:
            raise FileNotFoundError("no newly supplied hash-bound environment receipt")
        ref = dict(path=str(path), sha256=receipt_hash)
        document = json.loads(typed.authenticate(ref).read_bytes())
        if not (
            document["authority"] == "operator"
            and document["previous_primary_sha256"] == PIN
            and frontier(work["historical"])
            < document["executed_at_ns"]
            <= work["observed_wall_ns"]
            and document["action"] in {"driver_repair", "device_repair", "lease_rebind", "reboot"}
            and document["description"]
        ):
            raise ValueError("new_causal_receipt_authority")
        supplied = json.loads(typed.authenticate(document["current_reference"]).read_bytes())
        work["receipt_reference"] = snapshot(path, raw / "new_receipt", "receipt")
        work["refs"].append(work["receipt_reference"])
        work["refs"].append(
            snapshot(
                typed.authenticate(document["current_reference"]),
                raw / "new_receipt",
                "environment",
            )
        )
        work["new_environment_reference"] = work["refs"][-1]
        progress("current_inventory_before")
        observed, work["observation_receipts"] = old.inventory(work["baseline"], raw / "inventory")
        progress("current_inventory_after", 1, 0)
        if any(
            r["changed"] or not r["available"]
            for r in qualified.changes(dict(previous=supplied, current=observed))
        ):
            raise ValueError("supplied_environment_differs_from_current_inventory")
        work["current"] = observed
        for item in observed["libraries"]:
            if Path(item["path"]).name.startswith(("libcuda", "libnvidia")):
                work["refs"].append(
                    snapshot(typed.authenticate(item), raw / "current_libraries", "library")
                )
        if not fixture and not work["failures"] and any(r["changed"] for r in delta(work)):
            progress("direct_context_copy_before")
            work["diagnostic"] = probe(observed, raw / "direct_probe")
            progress("direct_context_copy_after", 1, 0)
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["failures"].append(
            old.gate(
                path,
                "new_authenticated_environment_receipt",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
            )
        )
    atomic_json(raw / "measurement.json", work)
    return work


def reduction(work: Json, passed: bool) -> Json:
    """Qualified reading, causal change and successful current copy have separate gates."""
    rows = delta(work)
    reader = bool(
        passed
        and work["historical"].get("runtime_reader_ready_score") == 1
        and work["baseline"]
        and not work["fixture"]
    )
    changed = bool(
        work["baseline"]
        and work["receipt_reference"]
        and not work["fixture"]
        and any(r["changed"] for r in rows)
    )
    context = bool(
        reader
        and changed
        and not work["failures"]
        and work["diagnostic"]["rows"]
        and all(
            r["receipt"]["passed"]
            and r["primitive"].get("context_copy_ready")
            and r["primitive"].get("byte_copy_parity")
            and r["primitive"].get("cleanup_passed")
            for r in work["diagnostic"]["rows"]
        )
        and not work["diagnostic"]["checks"]
    )
    gates = list(work["failures"]) + work["diagnostic"]["checks"]
    if not context:
        gates.append(
            old.gate(
                Path(work["root"]) / UPSTREAM,
                "authenticated_causal_delta" if not changed else "current_direct_context_copy",
                True,
                changed if not changed else context,
            )
        )
    if not passed:
        gates.append(old.gate(Path(work["root"]) / CLI, "required_checks_passed", True, False))
    for gate in gates:
        gate.update(check=gate.get("field", gate.get("artifact_field")), upstream_id=TASK)
    verdict = "disqualified" if not passed else "blocked" if not context else "null"
    return dict(
        honest_verdict="complete_"
        + verdict
        + "_"
        + (
            "owned_checks"
            if not passed
            else "cuda_environment_unchanged"
            if not changed
            else "cuda_context"
        ),
        verdict_class=verdict,
        gate_check_summary=gates,
        rows=rows,
        environment_delta=rows,
        runtime_reader_ready_score=int(reader),
        runtime_changed_score=int(changed),
        cuda_context_ready_score=int(context),
        intended_count=7,
        completed_count=sum(r["available"] for r in rows),
        failed_count=0,
        censored_count=sum(not r["available"] for r in rows),
        excluded_count=0,
        independent_count=0,
        acceptance_gates=dict(
            typed_reader=reader, authenticated_change=changed, current_context_copy=context
        ),
    )


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Bind terminal claims to primitive rows and keep cached errors out of current calls."""
    passed = bool(receipts) and all(r["passed"] for r in receipts if r.get("scope") != "global")
    value = reduction(work, passed)
    value.update(
        experiment_id=8382,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261010",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical"].get("historical_model_provenance", {}),
        verifier_is_oracle=False,
        exposure_scope="exposed development; fixtures are constructed mechanics",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        validation_receipts=receipts,
        sample_size_budget=dict(intended_operands=7, independent_scientific_samples=0),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        resource_preconditions=work.get("resources", {}),
        duration_s=work.get("duration_s", 0.001),
        phase_spans=work.get("phase_spans", []),
        random_seed=7228382,
        source_artifact_hashes=work["refs"],
        code_config_hashes=[
            snapshot(ROOT / p, raw / "code", str(i))
            for i, p in enumerate(
                OWNED
                + [
                    TEST,
                    "python/carnot/verify/typed_runtime_8368.py",
                    "python/carnot/reporting/primary_publication.py",
                    "scripts/adversarial_verify.py",
                    "scripts/verdict_row_consistency_lint.py",
                ]
            )
        ],
        primitive_reference=reference(raw / "measurement.json"),
        raw_shard_hashes=[reference(raw / "measurement.json")],
        context_copy_receipts=work["diagnostic"],
        device_identity=work["current"].get("devices"),
        historical_device_identity=work["baseline"].get("devices"),
        historical_closure_status=dict(
            authenticated=bool(work["baseline"]),
            producer_sha256=PIN,
            original_verdict=work["historical"].get("honest_verdict"),
            driver_errors=work["historical"]
            .get("historical_model_provenance", {})
            .get("failure_layer"),
        ),
        execution_authority=work["authority"],
        fixture_mode=work["fixture"],
        current_probe_count=len(work["diagnostic"]["rows"]),
        owned_statement_coverage=work.get("owned_statement_coverage", {}),
        adversarial_findings=work.get("finding_audits", []),
        cited_upstream_artifacts=[
            dict(r, fields_imported=["byte custody, historical runtime identity and disposition"])
            for r in work["refs"]
        ],
        methodology_note="Authenticate the qualified typed reader and immutable aliases, compare only explicitly supplied new environment receipts, then admit one direct driver context/copy probe. No model or semantic-benefit measurement.",
    )
    value["field_principles"] = {
        k: "Bind this invocation to sealed evidence; missing observations and historical calls cannot become current readiness."
        for k in value
    }
    value["field_principles"]["reproducibility_checksum"] = (
        "Bind every result field except this checksum."
    )
    value["field_principles"]["field_principles"] = "Explain why each result field is required."
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Recompute scores from sealed rows; rehashing a headline cannot repair evidence."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value) or [
            value[k] for k in ("experiment_id", "task_id", "milestone", "run_date")
        ] != [8382, TASK, MILESTONE, "20261010"]:
            return False
        for ref in typed.references(value):
            typed.authenticate(ref)
        work = json.loads(typed.authenticate(value["primitive_reference"]).read_bytes())
        for ref in typed.references(work):
            typed.authenticate(ref)
        for name, pin in PROTOCOL_PINS.items():
            ref = next(
                (r for r in work["refs"] if r["path"] == str(Path(work["root"]) / name)), None
            )
            if ref is None:
                return False
            if ref.get("exists") and ref["sha256"] != pin:
                return False
        if work["historical"]:
            source = work["historical_reference"]
            if (
                source["sha256"] != PIN
                or json.loads(typed.authenticate(source).read_bytes()) != work["historical"]
            ):
                return False
            baseline = work["baseline_reference"]
            if (
                baseline["sha256"] != BASELINE_PIN
                or json.loads(typed.authenticate(baseline).read_bytes()) != work["baseline"]
            ):
                return False
        if work["current"]:
            receipt = json.loads(typed.authenticate(work["receipt_reference"]).read_bytes())
            supplied = json.loads(
                typed.authenticate(work["new_environment_reference"]).read_bytes()
            )
            if not (
                receipt["authority"] == "operator"
                and receipt["previous_primary_sha256"] == PIN
                and frontier(work["historical"])
                < receipt["executed_at_ns"]
                <= work["observed_wall_ns"]
                and receipt["action"]
                in {"driver_repair", "device_repair", "lease_rebind", "reboot"}
                and receipt["description"]
                and receipt["current_reference"]["sha256"]
                == work["new_environment_reference"]["sha256"]
                and not any(
                    r["changed"] or not r["available"]
                    for r in qualified.changes(dict(previous=supplied, current=work["current"]))
                )
            ):
                return False
        for receipt in (
            value["validation_receipts"]
            + work["observation_receipts"]
            + [r["receipt"] for r in work["diagnostic"]["rows"]]
        ):
            recorded = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            if receipt != recorded or receipt["passed"] != (
                receipt["exit_code"] == receipt["expected_exit"]
                and receipt["normal_exit"]
                and not receipt["timed_out"]
            ):
                return False
        for row in work["diagnostic"]["rows"]:
            if (
                row["receipt"]["passed"]
                and json.loads(Path(row["receipt"]["stdout_path"]).read_text().splitlines()[-1])
                != row["primitive"]
            ):
                return False
        passed = bool(value["validation_receipts"]) and all(
            r["passed"] for r in value["validation_receipts"] if r.get("scope") != "global"
        )
        return (
            value["required_checks_passed"] == passed
            and value["flagged_adversarial"] == (not passed)
            and all(value[k] == v for k, v in reduction(work, passed).items())
            and not value["MODEL_SPECS"]
            and value["model_invocation_counts"] == ZERO_INVOCATION_COUNTS
            and value["independent_generalization_score"]
            == value["generalized_learning_benefit_score"]
            == 0
            and value["inference_substrate_class"] == "no_model_load"
            and value["no_model_load"] is True
            and value["context_copy_receipts"] == work["diagnostic"]
            and value["fixture_mode"] == work["fixture"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
