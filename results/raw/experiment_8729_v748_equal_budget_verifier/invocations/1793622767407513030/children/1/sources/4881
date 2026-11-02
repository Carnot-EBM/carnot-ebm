"""REQ-REPORT-8633: retain private evidence and publish only checked terminal bytes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import installed_lean_8633 as e
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
REQUIRED = {
    "owned_tests",
    "owned_coverage",
    "exact_node_union",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "e2e018",
    "e2e021",
    "primary_publication",
}


def artifact(authority: Json, receipts: list[Json]) -> Json:
    """Keep executable readiness distinct from any independent scientific benefit."""
    work = e.load(authority)
    evidence = work["evidence"]
    checks = {r["name"]: r["passed"] for r in receipts}
    clean = (
        bool(evidence)
        and all(r["clean"] and r["clean_shutdown"] for r in evidence["import_trace_rows"])
        and all(r["passed"] for r in evidence["export_contract_rows"])
        and all(r["passed"] for r in evidence.get("build_receipts", []))
    )
    required = REQUIRED <= checks.keys() and all(checks.values())
    ready = int(clean and required and not work["gate_check_summary"])
    counts = (
        e.reduce_rows(evidence["prototype_parity_rows"], work["bundle"])
        if evidence
        else dict(
            intended_count=128,
            completed_count=0,
            failed_count=0,
            censored_count=128,
            excluded_count=0,
            independent_count=0,
        )
    )
    verdict = (
        "disqualified"
        if any(not v for v in checks.values()) or work.get("owned_execution_failed", False)
        else "blocked"
        if work["gate_check_summary"]
        else "circular_positive"
        if ready
        else "disqualified"
    )
    value: Json = dict(
        experiment_id=8633,
        task_id=e.TASK,
        milestone="2026.10.741",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_installed_lean_mechanics",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"],
        inference_substrate_class="no_model_load",
        inference_substrate="CPU small-head spline numerical inference; offline wheel mechanics",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_language_model_calls=0,
        historical_model_provenance=work["bundle"].get("historical_model_provenance", {}),
        target_model=work["bundle"].get("target_model", {}),
        rows=evidence.get("prototype_parity_rows", []),
        **counts,
        sample_size_budget=dict(
            original_requests=64,
            arithmetic_arms=2,
            intended=128,
            independent_questions=0,
            reason="exposed compatibility requests",
        ),
        verifier_is_oracle=True,
        exposure_scope="original exposed64 research requests; mechanical compatibility only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        installed_prototype_ready_score=ready,
        branch_gate_values=dict(
            installed_prototype_ready_score=ready,
            independent_generalization_score=0,
            generalized_learning_benefit_score=0,
        ),
        required_checks_passed=bool(ready),
        flagged_adversarial=False,
        acceptance_gates=[dict(name=k, passed=v) for k, v in sorted(checks.items())]
        + [dict(name="installed_mechanics", passed=bool(clean))],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7418633,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=[
            e.reference(e.ROOT / p)
            for p in e.OWNED
            + [
                "python/carnot/inference/resident_energy.py",
                "python/carnot/inference/__init__.py",
                "python/carnot/verify/source_alignment.py",
                "scripts/experiments/resident_worker_8608.py",
                "python/carnot/_version.py",
                "pyproject.toml",
            ]
        ],
        raw_shard_hashes=work.get("raw_shard_hashes", []),
        cited_upstream_artifacts=work["source_artifact_hashes"],
        wheel_reference=evidence.get("wheel_reference"),
        installed_entry_command=evidence.get("installed_entry_command"),
        installed_distribution_record=evidence.get("installed_distribution_record"),
        dependency_versions=evidence.get("dependency_versions"),
        import_trace_rows=evidence.get("import_trace_rows", []),
        export_contract_rows=evidence.get("export_contract_rows", []),
        prototype_parity_rows=evidence.get("prototype_parity_rows", []),
        original_head_reference=work["bundle"].get("target_model", {}).get("checkpoint"),
        authority_reference=authority,
        changed_public_paths=[
            dict(
                path="carnot",
                change="core/models/samplers/training and optional JAX/Rust exports resolve lazily; names and __all__ order retained",
                exact_exports={
                    "carnot.core": ["AutoGradMixin", "EnergyFunction", "ModelConfig", "ModelState"],
                    "carnot.models": [
                        "BoltzmannConfig",
                        "BoltzmannModel",
                        "EBTConfig",
                        "EBTransformer",
                        "GibbsConfig",
                        "GibbsModel",
                        "IsingConfig",
                        "IsingModel",
                    ],
                    "carnot.samplers": ["HMCSampler", "LangevinSampler"],
                    "carnot.training": [
                        "ReplayBuffer",
                        "dsm_loss",
                        "dsm_loss_stochastic",
                        "nce_loss",
                        "nce_loss_stochastic",
                        "nce_loss_with_replay",
                        "optimization_training_loss",
                        "snl_loss",
                        "snl_loss_stochastic",
                    ],
                    "optional": ["jax", "jnp", "JAX_AVAILABLE", "RUST_AVAILABLE", "__all__"],
                },
            ),
            dict(
                path="carnot.inference.resident_protocol",
                change="package-relative resident_energy import; main(argv) and explicit module CLI",
                old_import="import resident_energy as numeric",
                package_import="from . import resident_energy as numeric",
            ),
        ],
        preserved_historical_nulls=work["bundle"].get("historical_cost_null"),
        preserved_quarantines=work["bundle"].get("preserved_quarantines"),
        methodology="Compare actual private and installed Python/native children on identical original64 frames. No fitting, generation, speed or confidence claim.",
    )
    value["field_principles"] = {
        key: "Bind this invocation and preserve mechanical evidence; no independent science credit."
        for key in value
    }
    value["field_principles"].update(
        installed_prototype_ready_score="One only when original64 Python/native comparisons, imports, exports and all owned checks pass. No surface qualification or speed credit.",
        independent_generalization_score="Zero: all requests are exposed compatibility operands; no independent questions were sampled.",
        generalized_learning_benefit_score="Zero: this invocation fits no model and measures no learning benefit.",
        completed_count="128 installed responses:64 per arithmetic arm, each checked against its own original-worker response.",
        raw_shard_hashes="Original wheel, framed session transcripts, import logs and installation RECORD are immutable replay operands.",
        changed_public_paths="Exact initialization exports and resident module imports for the later full compatibility qualification.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, authority: Json) -> None:
    """An external authority hash prevents a rehashed candidate from rewriting its inputs."""
    value = json.loads(path.read_bytes())
    work = e.load(authority)
    if (
        work.get("validation_receipts", value["validation_receipts"])
        != value["validation_receipts"]
    ):
        raise ValueError("validation_receipt_drift")
    if value != artifact(authority, value["validation_receipts"]):
        raise ValueError("immutable_reduction_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if e.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("replay_input_hash")


def terminal(path: Path, authority: Json, logs: Path) -> Json:
    """Retain the actual validator findings for exactly the candidate that readers see."""
    changed = json.loads(path.read_bytes())
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    bad = logs / "rehashed_tamper.json"
    atomic_json(bad, changed)
    receipts = [
        child(name, [sys.executable, "-u", *argv], logs, deadline=90)
        for name, argv in [
            (
                "cold",
                [
                    e.CLI,
                    "--cold-replay",
                    str(path),
                    "--authority",
                    authority["path"],
                    authority["sha256"],
                ],
            ),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
            ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
        ]
    ]
    for name, candidate in [("missing", logs / "missing.json"), ("rehashed_tamper", bad)]:
        receipts.append(
            child(
                name,
                [
                    sys.executable,
                    "-u",
                    e.CLI,
                    "--cold-replay",
                    str(candidate),
                    "--authority",
                    authority["path"],
                    authority["sha256"],
                ],
                logs,
                expected=1,
                deadline=90,
            )
        )
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        actual_findings=json.loads(Path(receipts[1]["stdout_path"]).read_text()),
    )


def run(output: Path, run_date: str, root: Path) -> int:
    """Measure within3600 seconds and keep the final900 seconds for validation."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    start = time.monotonic_ns()
    work: Json = dict(
        raw=str(raw),
        run_date=run_date,
        evidence={},
        bundle={},
        gate_check_summary=[],
        preconditions_checked={},
        source_artifact_hashes=[],
        phase_spans=[],
    )
    try:
        memory = Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]
        work["preconditions_checked"] = dict(
            memory_available_kib=int(memory),
            disk_free_bytes=shutil.disk_usage("/var/tmp").free,
            python=e.reference(Path(sys.executable)),
            cpu_branch=True,
            scratch="/var/tmp",
            optional_gpu_required=False,
            tools={
                name: (e.ROOT / ".venv/bin" / name).is_file()
                for name in ["python", "pytest", "coverage", "ruff", "mypy"]
            },
        )
        if (
            int(memory) < 262144
            or shutil.disk_usage("/var/tmp").free < 268435456
            or not all(work["preconditions_checked"]["tools"].values())
        ):
            raise ValueError("memory_disk_or_tools")
        work["bundle"] = e.inputs(root)
        work["source_artifact_hashes"] = work["bundle"]["refs"]
        work["evidence"] = e.measure(raw, work["bundle"])
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        work["owned_execution_failed"] = bool(work["bundle"])
        work["gate_check_summary"].append(
            dict(
                check="required_branch_inputs_or_execution",
                upstream="exp8622-lean-resident-cost",
                path=str(root / e.SOURCE),
                sha256=e.PIN,
                field="original_operands_and_installed_execution",
                operator="==",
                expected="available and valid",
                observed=str(error),
            )
        )
    end = time.monotonic_ns()
    work["duration_s"] = (end - start) / 1e9
    work["phase_spans"] = [
        dict(phase="preconditions_and_installed_execution", start_ns=start, end_ns=end)
    ]
    work["raw_shard_hashes"] = [e.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    authority_path = raw / "authority.json"
    from carnot.reporting.installed_validation_8633 import validate

    receipts = validate(raw)
    work["raw_shard_hashes"].extend(
        e.reference(p) for p in sorted((raw / "validation").glob(".coverage*")) if p.is_file()
    )
    work["validation_receipts"] = receipts
    validated = time.monotonic_ns()
    work["duration_s"] = (validated - start) / 1e9
    work["phase_spans"].append(dict(phase="validation", start_ns=end, end_ns=validated))
    atomic_json(authority_path, work)
    authority = e.reference(authority_path)
    value = artifact(authority, receipts)
    published = publish_primary(output, value, lambda p: terminal(p, authority, raw / "terminal"))
    atomic_json(
        raw / "terminal_validation.json",
        read_bound_sidecar(output, Path(published["sidecar_path"])),
    )
    reader = reader_receipt(
        e.TASK,
        output.parent,
        field="installed_prototype_ready_score",
        expected=value["installed_prototype_ready_score"],
    )
    atomic_json(raw / "reader_receipt.json", reader)
    progress(
        "8633_complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return int(value["verdict_class"] == "disqualified")


def main(argv: list[str] | None = None) -> int:
    """Keep private paths explicit so tests cannot replace the public primary."""
    progress("8633_start_no_model_load_MODEL_SPECS=[]_current_language_model_calls=0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261018")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", nargs=2)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            if not args.authority:
                raise ValueError("immutable_authority_required")
            replay(args.cold_replay, dict(path=args.authority[0], sha256=args.authority[1]))
            return 0
        if (
            datetime.strptime(args.date, "%Y%m%d").strftime("%Y%m%d") != args.date
            or (args.input_root and not args.private_output)
            or (args.private_output and args.private_output.resolve().is_relative_to(e.ROOT))
        ):
            raise ValueError("date_or_private_boundary")
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            args.date,
            args.input_root or e.ROOT,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[8633] error={error}", file=sys.stderr, flush=True)
        return 1
