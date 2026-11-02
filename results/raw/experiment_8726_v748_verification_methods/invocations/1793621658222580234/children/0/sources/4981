"""REQ-REPORT-8238: expose authenticated prediction custody without benefit credit.

The public child receives no evaluator operand. Current readiness records a
complete seal; historical development exposure cannot be removed by sealing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import decision_margin_methods_8234 as methods
from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v686_contract_validation import dependency_hashes
from carnot.verify import margin_prediction_seal_8238 as numeric
from carnot.verify import utility_fit_execution_8222 as qualified
from carnot.verify import utility_patch_methods_8219 as custody

Json = dict[str, Any]
ROOT = methods.ROOT
NAME = "experiment_8238_v712_margin_prediction_seal"
TASK = "exp8238-margin-prediction-seal"
MODULE = "python/carnot/reporting/margin_prediction_seal_8238.py"
RUNNER = MODULE
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_margin_prediction_seal_8238.py"
OWNED = ["python/carnot/verify/margin_prediction_seal_8238.py", MODULE, CLI]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
PROTOCOL_VALUE = methods.PROTOCOL_VALUE
run_check, reference = custody.run_check, custody.reference
UPSTREAM = "results/experiment_8237_v712_margin_energy_training.json"
PUBLIC = "results/raw/experiment_8209_v709_restricted_sealed_evaluation/invocations/1791313083770766474/primitive_evidence.json"
PINS = {
    UPSTREAM: "sha256:0e4b9317c60d1184315bfca2cce58f6ab4e07f2cf7425563a6cb27d3b6c93e86",
    "results/experiment_8234_v712_decision_margin_methods.json": "sha256:018d57444d13ec1402de739b5141bc7df57a6edc4004e097c9caa31968418f37",
    PUBLIC: "sha256:549dae11d23cc7d05bf3b7162d7f22328c129b7da87a19ace31d6dc27cfd9b78",
    methods.PROTOCOL: methods.PIN,
    str(
        Path(PROTOCOL_VALUE["native_reserved_measurement"]["path"]).relative_to(ROOT)
    ): PROTOCOL_VALUE["native_reserved_measurement"]["sha256"],
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so the supervisor can distinguish progress from a stall."""
    print(f"[exp8238] phase={phase} completed={completed} pending={pending}", flush=True)


def inputs(work: Json, root: Path, raw: Path) -> Json:
    """Read pinned operands, projecting native capture provenance into public rows."""
    copies = {name: custody.bind(work, root / name, digest, raw) for name, digest in PINS.items()}
    upstream = json.loads(copies[UPSTREAM].read_bytes())
    for field, expected in [
        ("schema", "carnot.v712.margin-energy-training.v1"),
        ("margin_fit_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
        ("labels_opened", False),
        ("protocol_sha256", methods.PIN),
    ]:
        custody.gate(work, root / UPSTREAM, field, expected, upstream.get(field))
    custody.gate(
        work,
        root / UPSTREAM,
        "verdict_class_usable",
        True,
        upstream.get("verdict_class") in ["positive", "null", "circular_positive"],
    )
    head_path = root / Path(upstream["trained_heads_path"]).relative_to(ROOT)
    bundle_copy = custody.bind(work, head_path, upstream["trained_heads_sha256"], raw)
    bundle = json.loads(bundle_copy.read_bytes())
    for field, expected in [
        ("schema", "carnot.v712.margin-heads.v1"),
        ("labels_opened", False),
        ("protocol_sha256", methods.PIN),
        ("scoring_api", "carnot.verify.margin_energy_training_8237.score"),
    ]:
        custody.gate(work, head_path, field, expected, bundle.get(field))
    public = json.loads(copies[PUBLIC].read_bytes())
    custody.gate(
        work, root / PUBLIC, "roles", PROTOCOL_VALUE["role_manifest"], public["frozen"]["roles"]
    )
    custody.gate(
        work,
        head_path,
        "global_base_head",
        next(h for h in public["frozen"]["heads"] if h["arm"] == "energy"),
        bundle["global_head"]["base_head"],
    )
    scorer = "python/carnot/verify/margin_energy_training_8237.py"
    pinned_scorer = next(
        r for r in upstream["code_config_hashes"] if r["path"] == str(ROOT / scorer)
    )
    custody.bind(work, ROOT / scorer, pinned_scorer["sha256"], raw)
    capture_name = str(
        Path(PROTOCOL_VALUE["native_reserved_measurement"]["path"]).relative_to(ROOT)
    )
    captured = json.loads(copies[capture_name].read_bytes())["plan"]["baseline"]
    native = {r["unit_id"]: r for r in captured}
    custody.gate(work, root / capture_name, "native_row_count", 128, len(captured))
    custody.gate(work, root / capture_name, "native_unique_count", 128, len(native))
    data = dict(
        heads=bundle["heads"],
        head_hashes=[canonical_hash(h) for h in bundle["heads"]],
        global_head=bundle["global_head"],
        global_head_hash=canonical_hash(bundle["global_head"]),
        public=public,
        roles=PROTOCOL_VALUE["role_manifest"],
        roles_sha256=canonical_hash(PROTOCOL_VALUE["role_manifest"]),
        comparator=dict(arm=upstream["comparator_name"], sha256=upstream["comparator_sha256"]),
        native=[
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=native[r["unit_id"]]["source_cluster_id"],
                p0=native[r["unit_id"]]["holistic_probability"],
            )
            for r in public["roster"]
        ],
        method_sha256=methods.PIN,
    )
    numeric.original.reject_labels(data)
    work["heads_copy"] = str(bundle_copy)
    work["heads_sha256"] = upstream["trained_heads_sha256"]
    return data


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """A bounded child scores only the authenticated public payload and saves evidence."""
    start, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        root=str(root),
        checks=[],
        refs=[],
        evidence={},
        diagnostics={},
        owned_failure="",
        precondition_receipts=[],
        worker_receipts=[],
        heads_copy=None,
        heads_sha256=None,
    )
    progress("before_preconditions", 0, 128)
    try:
        with TemporaryDirectory(prefix="carnot8238-probe-") as directory:
            probe = Path(directory) / "writable"
            probe.write_bytes(b"private scratch")
            custody.gate(
                work,
                probe,
                "private_scratch_writable",
                True,
                probe.read_bytes() == b"private scratch"
                and probe.parent.stat().st_mode & 0o777 == 0o700,
            )
        if mutation:
            custody.gate(work, root / UPSTREAM, "source_custody", "authenticated", None)
        work["evidence"] = inputs(work, root, raw)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        if all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="authenticated_input_schema",
                    path=str(root),
                    upstream=str(root),
                    hash=None,
                    artifact_field="authenticated_input_schema",
                    op="==",
                    expected="valid",
                    observed=str(error),
                    passed=False,
                )
            )
    progress("after_preconditions", len(work["checks"]), 0)
    atomic_json(raw / "public_worker_input.json", work["evidence"])
    if work["evidence"]:
        command = dict(
            name="public_prediction_worker",
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--public-input",
                str(raw / "public_worker_input.json"),
                "--prediction-output",
                str(raw / "public_worker_result.json"),
            ],
            deadline_s=60,
            expected_exit=0,
        )
        progress("before_benchmark_subprocess", 0, 128)
        with TemporaryDirectory(prefix="carnot8238-child-") as directory:
            receipt = run_check(ROOT, command, Path(directory), raw / "worker_logs", heartbeat_s=20)
        work["worker_receipts"].append(receipt)
        progress("after_benchmark_subprocess", 128, 0)
        if receipt["passed"]:
            work["diagnostics"] = json.loads((raw / "public_worker_result.json").read_bytes())
        else:
            work["owned_failure"] = "public_prediction_worker_exit:" + str(receipt["actual_exit"])
    work["code_config_hashes"] = []
    for name, digest in dependency_hashes(ROOT, paths=[*OWNED, TEST]).items():
        copy = custody.bind(dict(checks=[], refs=[]), ROOT / name, digest, raw)
        copy.chmod(0o444)
        work["code_config_hashes"].append(dict(reference(ROOT / name), snapshot_path=str(copy)))
    manifest_value = dict(
        input_path=str(raw / "public_worker_input.json"),
        input_sha256=sha256_file(raw / "public_worker_input.json"),
        public_fields=list(work["evidence"]),
        evaluator_label_paths=[],
        labels_opened=False,
        access_scope="public features, native probabilities, frozen heads, roles and original permissions only",
        worker_receipts=work["worker_receipts"],
    )
    seal = dict(
        rows=work["diagnostics"].get("prediction_rows", []),
        labels_opened=False,
        complete_heads_path=work["heads_copy"],
        complete_heads_sha256=work["heads_sha256"],
        input_sha256=manifest_value["input_sha256"],
        permission_mask=work["diagnostics"].get("permission_mask", []),
    )
    for name, value in [
        ("primitive_evidence", work["evidence"]),
        ("sealed_predictions", seal),
        ("public_worker_input_manifest", manifest_value),
    ]:
        atomic_json(raw / (name + ".json"), value)
    for ref in work["refs"]:
        Path(ref["path"]).chmod(0o444)
    paths = [
        raw / (n + ".json")
        for n in [
            "primitive_evidence",
            "sealed_predictions",
            "public_worker_input",
            "public_worker_input_manifest",
        ]
    ]
    if (raw / "public_worker_result.json").exists():
        paths.append(raw / "public_worker_result.json")
    for path in paths:
        path.chmod(0o444)
    work["raw_shard_hashes"] = [reference(p) for p in paths]
    work["public_worker_input_manifest"] = manifest_value
    end = time.monotonic_ns()
    work.update(
        duration_s=(end - start) / 1e9,
        clock=dict(
            started_monotonic_ns=start,
            ended_monotonic_ns=end,
            started_wall_ns=wall,
            ended_wall_ns=time.time_ns(),
        ),
    )
    atomic_json(raw / "measurement.json", work)
    progress("predictions_sealed", len(seal["rows"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Only a complete checked seal receives readiness; benefit remains unmeasured."""
    from scripts.experiment_template import normalize_artifact_for_template_write

    failures = [c for c in work["checks"] if not c["passed"]]
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    result = work["diagnostics"]
    ready = int(
        checked
        and not failures
        and len(result.get("rows", [])) == 128
        and len(result.get("prediction_rows", [])) == 1152
    )
    verdict = "disqualified" if not checked else "blocked" if failures else "null"
    suffix = (
        (Path(failures[0]["path"]).stem + "_" + failures[0]["artifact_field"])
        if failures
        else "margin_prediction_seal"
    )
    rows = result.get(
        "rows",
        [
            dict(
                r,
                slot=i + 1,
                arm="all_frozen_margin_arms",
                condition="original_reserved_slot",
                status="excluded",
                exclusion_reason="external_operand_unavailable"
                if failures
                else "owned_prediction_failure",
                metric="complete_paired_source",
                numerator=0,
                denominator=1,
            )
            for i, r in enumerate(PROTOCOL_VALUE["role_manifest"]["reserved"])
        ],
    )
    value: Json = dict(
        experiment_id=8238,
        task_id=TASK,
        milestone="2026.10.712",
        run_date=RUN_DATE,
        schema="carnot.v712.margin-prediction-seal.v1",
        honest_verdict="complete_" + verdict + "_" + suffix.lower(),
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        owned_failure=work["owned_failure"],
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if result
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            model_loads=0, generate_calls=0, forward_calls=0, model_count=0
        ),
        rows=rows,
        intended_count=128,
        completed_count=result.get("completed_count", 0),
        failed_count=result.get("failed_count", 0),
        censored_count=result.get("censored_count", 0),
        excluded_count=result.get("excluded_count", 128),
        independent_count=result.get("independent_count", 0),
        verifier_is_oracle=False,
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        scientific_benefit_measured=False,
        H1="unmeasured",
        H2="unmeasured",
        required_checks_passed=bool(checked),
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_checks=bool(checked),
            authenticated_inputs=not failures,
            all_slots_and_predictions_sealed=bool(ready),
            acceptance_subset=bool(result),
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["checks"],
        precondition_receipts=work["precondition_receipts"],
        duration_s=work["duration_s"],
        random_seed=7128238,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        phase_spans=[
            dict(
                phase="authenticate_public_worker_and_seal",
                **work["clock"],
                duration_s=work["duration_s"],
            )
        ],
        measurement_clocks=work["clock"],
        measurement_reference=reference(raw / "measurement.json"),
        margin_predictions_ready_score=ready,
        labels_opened=False,
        predictions_path=str(raw / "sealed_predictions.json"),
        predictions_sha256=sha256_file(raw / "sealed_predictions.json"),
        prediction_count=len(result.get("prediction_rows", [])),
        public_worker_input_manifest=work["public_worker_input_manifest"],
        original_reserved_mask=work["evidence"].get("public", {}).get("roster", rows),
        frozen_head_hashes=dict(
            heads=work["evidence"].get("head_hashes", []),
            global_head=work["evidence"].get("global_head_hash"),
            complete_head_bytes=work["heads_sha256"],
        ),
        complete_heads_path=work["heads_copy"],
        complete_heads_sha256=work["heads_sha256"],
        trained_head_specs=[
            dict(arm=a, coefficients=17, current_fit=False, generator=False) for a in numeric.ARMS
        ],
        frozen_comparator=work["evidence"].get("comparator"),
        protocol_path=str(ROOT / methods.PROTOCOL),
        protocol_sha256=methods.PIN,
        role_manifest=PROTOCOL_VALUE["role_manifest"],
        roles_sha256=canonical_hash(PROTOCOL_VALUE["role_manifest"]),
        cited_upstream_artifacts=[
            dict(
                path=r["upstream_path"],
                sha256=r["sha256"],
                fields_imported=[
                    "frozen readiness, roles, public features, native probabilities, comparator or complete head bytes"
                ],
            )
            for r in work["refs"]
        ],
        repository_health=work.get("global_health", {}),
        claim_scope="Prediction custody on the same exposed development sources; no H1 or independent generalization measurement.",
        methodology_note="No model loads or current LLM calls. A separate public-input-only child applies the frozen six-head API and historical energy_global comparator to every original reserved slot. Native public margins and feasible expected losses preserve V707 permissions. Missing evidence and ties escalate. Complete head bytes and predictions are sealed before a separate audit can access labels. Historical exposure persists; readiness grants no scientific benefit.",
    )
    coverage_path = raw / "logs" / "changed_code_coverage.json"
    value["coverage"] = json.loads(coverage_path.read_bytes()) if coverage_path.exists() else {}
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind actual public execution to authenticated bytes; readiness grants no benefit credit."
        for k in value
    }
    value["field_principles"].update(
        margin_predictions_ready_score="One requires all 128 accounted slots, all 1152 immutable decisions and passing owned checks.",
        public_worker_input_manifest="The scoring child receives one authenticated public payload and no evaluator-label path.",
        frozen_head_hashes="Complete bundle bytes and individual heads bind methods, roles and comparator before audit access.",
        exposure_scope="A current seal cannot undo prior source exposure.",
        independent_generalization_score="Reused development sources establish no independent generalization.",
        generalized_learning_benefit_score="Neither H1 nor a learning benefit is measured in this task.",
        coverage="Coverage measures only newly owned statements including real CLI children; no exclusions.",
        reproducibility_checksum="Canonical content binds the terminal result to its primitive, code and receipt hashes.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Fresh reduction uses immutable source copies and rejects rehashed alterations."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if canonical_hash(value) != checksum:
            return False
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"] or (
                "snapshot_path" in ref and sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]
            ):
                return False
        for receipt in [
            *value["validation_receipts"],
            *value["public_worker_input_manifest"]["worker_receipts"],
        ]:
            for stream in ["stdout", "stderr"]:
                if (
                    stream + "_path" in receipt
                    and sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]
                ):
                    return False
        raw = Path(value["measurement_reference"]["path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        if work["evidence"] != json.loads((raw / "primitive_evidence.json").read_bytes()) or work[
            "evidence"
        ] != json.loads((raw / "public_worker_input.json").read_bytes()):
            return False
        if work["evidence"]:
            with TemporaryDirectory(prefix="carnot8238-replay-") as directory:
                mirror, copied = Path(directory) / "root", Path(directory) / "copied"
                for ref in work["refs"]:
                    name = Path(ref["upstream_path"]).relative_to(Path(work["root"]))
                    destination = mirror / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.symlink_to(Path(ref["path"]))
                fresh: Json = dict(checks=[], refs=[])
                if inputs(fresh, mirror, copied) != work["evidence"]:
                    return False
            if (
                not work["owned_failure"]
                and numeric.reduce(work["evidence"]) != work["diagnostics"]
            ):
                return False
        seal = json.loads((raw / "sealed_predictions.json").read_bytes())
        if seal != dict(
            rows=work["diagnostics"].get("prediction_rows", []),
            labels_opened=False,
            complete_heads_path=work["heads_copy"],
            complete_heads_sha256=work["heads_sha256"],
            input_sha256=work["public_worker_input_manifest"]["input_sha256"],
            permission_mask=work["diagnostics"].get("permission_mask", []),
        ):
            return False
        if work["public_worker_input_manifest"] != json.loads(
            (raw / "public_worker_input_manifest.json").read_bytes()
        ):
            return False
        return build(work, raw, value["validation_receipts"]) == dict(
            value, reproducibility_checksum=checksum
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Reuse bounded private coverage and E2E commands with this task's exact paths."""
    with (
        patch.object(qualified, "OWNED", OWNED),
        patch.object(qualified, "TEST", TEST),
        patch.object(qualified, "CLI", CLI),
        patch.object(execution, "e", sys.modules[__name__]),
    ):
        specs: Json = qualified.manifest(private, candidate)
    specs["commands"].append(
        dict(
            name="private_E2E021",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "e2e021"),
            ],
            deadline_s=240,
            expected_exit=0,
        )
    )
    specs["commands"].append(
        dict(
            name="frozen_head_consumer",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_margin_energy_training_8237.py",
                "--basetemp=" + str(private / "head_consumer"),
            ],
            deadline_s=240,
            expected_exit=0,
        )
    )
    return specs


def main(argv: list[str] | None = None) -> int:
    """The direct public child opens one payload; the existing supervisor publishes."""
    arguments = sys.argv[1:] if argv is None else argv
    if "--public-input" in arguments:
        progress("public_worker_start", 0, 128)
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--public-input", type=Path, required=True)
        parser.add_argument("--prediction-output", type=Path, required=True)
        args = parser.parse_args(arguments)
        try:
            data = json.loads(args.public_input.read_bytes())
            progress("before_benchmark_public_score", 0, 128)
            atomic_json(args.prediction_output, numeric.reduce(data))
            progress("after_benchmark_public_score", 128, 0)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(json.dumps(dict(passed=False, error=str(error))), flush=True)
            return 1
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(arguments))
