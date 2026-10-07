"""REQ-REPORT-8223: publish predictions after checks, without benefit credit.

Only target-free historical primitives and fitted parameters enter scoring.
The separate audit must authenticate this seal before accessing reserved labels.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v686_contract_validation import dependency_hashes
from carnot.verify import utility_fit_execution_8222 as upstream
from carnot.verify import utility_patch_methods_8219 as frozen
from carnot.verify import utility_seal_8223 as numeric

Json = dict[str, Any]
ROOT = frozen.ROOT
NAME = "experiment_8223_v711_utility_seal"
TASK = "exp8223-utility-seal"
MODULE = "python/carnot/verify/utility_seal_execution_8223.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_utility_seal_8223.py"
OWNED = ["python/carnot/verify/utility_seal_8223.py", MODULE, CLI]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
PROTOCOL_VALUE = frozen.PROTOCOL_VALUE
run_check, reference = frozen.run_check, frozen.reference
UPSTREAM = "results/experiment_8222_v711_utility_fit.json"
PUBLIC = "results/raw/experiment_8209_v709_restricted_sealed_evaluation/invocations/1791313083770766474/primitive_evidence.json"
PINS = {
    UPSTREAM: "sha256:e205fe51b27fe8975704e9ff30869625a7a72a21aadf3d6eec435087f06101b8",
    PUBLIC: "sha256:549dae11d23cc7d05bf3b7162d7f22328c129b7da87a19ace31d6dc27cfd9b78",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the supervisor can distinguish scoring from a stall."""
    print(f"[exp8223] phase={phase} completed={completed} pending={pending}", flush=True)


def inputs(work: Json, root: Path, raw: Path) -> Json:
    """Pin exact upstream bytes and read no label-bearing measurement or audit."""
    frozen.bind(work, root / frozen.PROTOCOL, frozen.PIN, raw)
    copies = {name: frozen.bind(work, root / name, pin, raw) for name, pin in PINS.items()}
    value = upstream.primary(work, root, raw, Path(UPSTREAM).stem, "utility_fit_ready_score")
    parameters_path = root / Path(value["fitted_heads_path"]).relative_to(ROOT)
    parameters = json.loads(
        frozen.bind(work, parameters_path, value["fitted_heads_sha256"], raw).read_bytes()
    )
    public = json.loads(copies[PUBLIC].read_bytes())
    seal = PROTOCOL_VALUE["sealed_predictions"]
    old = json.loads(
        frozen.bind(
            work, root / Path(seal["path"]).relative_to(ROOT), seal["sha256"], raw
        ).read_bytes()
    )
    frozen.gate(work, Path(seal["path"]), "labels_opened", False, old.get("labels_opened"))
    frozen.gate(
        work,
        root / PUBLIC,
        "reserved_roles",
        PROTOCOL_VALUE["role_manifest"]["reserved"],
        public["frozen"]["roles"]["reserved"],
    )
    frozen.gate(
        work,
        parameters_path,
        "comparator_model",
        value["frozen_comparator"]["model"],
        parameters["models"][value["frozen_comparator"]["arm"] + ":none"],
    )
    numeric.original.reject_labels([parameters, public, old["rows"]])
    return dict(
        parameters=parameters,
        parameters_sha256=canonical_hash(parameters),
        public=public,
        comparator=value["frozen_comparator"],
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Authenticate scratch and operands before beginning owned scoring work."""
    start, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        diagnostics={},
        owned_failure="",
        protocol=frozen.PROTOCOL_VALUE,
        precondition_receipts=[],
    )
    progress("before_preconditions")
    try:
        with TemporaryDirectory(prefix="carnot-8223-probe-") as directory:
            probe = Path(directory) / "probe"
            probe.write_bytes(b"private writable scratch")
            frozen.gate(
                work,
                probe,
                "private_scratch_writable",
                True,
                probe.read_bytes() == b"private writable scratch",
            )
        progress("before_subprocess_python_environment")
        receipt = run_check(
            ROOT,
            dict(
                name="python_environment",
                argv=[sys.executable, "--version"],
                deadline_s=30,
                expected_exit=0,
            ),
            raw,
            raw / "preconditions",
        )
        progress("after_subprocess_python_environment")
        work["precondition_receipts"].append(receipt)
        frozen.gate(
            work, Path(sys.executable), "python_environment_exit", 0, receipt["actual_exit"]
        )
        if mutation:
            frozen.gate(work, root / UPSTREAM, "source_custody", "authenticated", None)
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
    progress("after_preconditions", len(work["checks"]))
    if all(c["passed"] for c in work["checks"]):
        try:
            progress("before_benchmark_predictions", 0, 128)
            work["diagnostics"] = numeric.reduce(work["evidence"])
            progress("after_benchmark_predictions", 128, 0)
        except (ValueError, KeyError, TypeError) as error:
            work["owned_failure"] = str(error)
    work["code_config_hashes"] = []
    dependencies = [
        *OWNED,
        TEST,
        frozen.PROTOCOL,
        upstream.MODULE,
        "python/carnot/verify/utility_kernel_8221.py",
        "python/carnot/verify/restricted_sealed_rule_8209.py",
        "python/carnot/verify/restricted_action_rule_8207.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/methods_stream_execution_8111.py",
    ]
    for name, digest in dependency_hashes(ROOT, paths=dependencies).items():
        snapshot = frozen.bind(dict(checks=[], refs=[]), ROOT / name, digest, raw)
        work["code_config_hashes"].append(dict(reference(ROOT / name), snapshot_path=str(snapshot)))
    seal = dict(
        rows=work["diagnostics"].get("prediction_rows", []),
        labels_opened=False,
        code_config_hashes=work["code_config_hashes"],
        parameters_sha256=work["evidence"].get("parameters_sha256"),
        permission_mask_sha256=work["diagnostics"].get("permission_mask_sha256"),
    )
    for name, value in [
        ("primitive_evidence", work["evidence"]),
        ("sealed_predictions", seal),
        ("permission_mask", dict(rows=work["diagnostics"].get("permission_mask", []))),
    ]:
        atomic_json(raw / (name + ".json"), value)
        (raw / (name + ".json")).chmod(0o444)
    work["raw_shard_hashes"] = [
        reference(raw / (name + ".json"))
        for name in ["primitive_evidence", "sealed_predictions", "permission_mask"]
    ]
    end = time.monotonic_ns()
    work.update(
        duration_s=(end - start) / 1e9,
        clock=dict(started_monotonic_ns=start, ended_monotonic_ns=end, started_wall_ns=wall),
    )
    atomic_json(raw / "measurement.json", work)
    progress("predictions_sealed", len(seal["rows"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness certifies a complete seal, while utility benefit remains unmeasured."""
    value = frozen.build(work, raw, receipts, fixture=False)
    result = work["diagnostics"]
    ready = value.pop("utility_protocol_ready_score")
    value.update(
        experiment_id=8223,
        task_id=TASK,
        milestone="2026.10.711",
        honest_verdict=value["honest_verdict"].replace("utility_patch_methods", "utility_seal"),
        inference_substrate="verifier_ensemble_against_cached_candidates",
        utility_predictions_ready_score=ready,
        rows=result.get("rows", value["rows"]),
        sealed_predictions_path=str(raw / "sealed_predictions.json"),
        prediction_sha256=sha256_file(raw / "sealed_predictions.json"),
        permission_mask_sha256=result.get("permission_mask_sha256"),
        permission_mask_path=str(raw / "permission_mask.json"),
        labels_opened=False,
        energy_parity_max_error=result.get("energy_parity_max_error"),
        frozen_comparator=work["evidence"].get("comparator"),
        original_reserved_mask=work["evidence"].get("public", {}).get("roster", []),
        parameters_sha256=work["evidence"].get("parameters_sha256"),
        prediction_count=len(result.get("prediction_rows", [])),
        fixture_protocol_only=fixture,
        random_seed=7108223,
        trained_head_specs=[
            dict(arm=a, seed=s, current_fit=False, kind="frozen_probability_patch")
            for a, s in numeric.fitted.SPECS
        ],
        claim_scope="Prediction custody on historically exposed development sources; no measured utility or generalization benefit.",
        methodology_note="No generator load or invocation. Original public features and Exp8222 frozen parameters produce all corrected predictions. Preserve every original slot and missing mask. Expected costs and original V707 accept permissions determine actions. Direct probability and normalized energy agree. Reserved labels stay closed for a separate audit. Reused sources do not establish independent generalization.",
    )
    for key in [
        "intended_count",
        "completed_count",
        "failed_count",
        "excluded_count",
        "censored_count",
        "independent_count",
    ]:
        if key in result:
            value[key] = result[key]
    value["cited_upstream_artifacts"] = [
        dict(
            path=r["upstream_path"],
            sha256=r["sha256"],
            fields_imported=[
                "authenticated public primitives, sealed parameters, comparator or terminal readiness"
            ],
        )
        for r in work["refs"]
    ]
    value["field_principles"] = {
        k: "Bind actual target-free execution to authenticated evidence; readiness grants no benefit credit."
        for k in value
    }
    value["field_principles"].update(
        utility_predictions_ready_score="One requires every original slot accounted for, a target-free seal and passing owned checks.",
        prediction_sha256="Bind all arm probabilities and decisions to sealed bytes before evaluator access.",
        permission_mask_sha256="Preserve original V707 accept permissions including unavailable sources.",
        labels_opened="False records this invocation only; historical exposure remains.",
        energy_parity_max_error="Normalized energies encode the same probabilities within 1e-10.",
        trained_head_specs="Sealed numerical parameters are not current training or LLM calls.",
    )
    value.pop("reproducibility_checksum", None)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Authenticate primitives and recompute decisions instead of trusting new aggregate hashes."""
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
        for receipt in value["validation_receipts"]:
            for label in ["stdout", "stderr"]:
                if (
                    label + "_path" in receipt
                    and sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]
                ):
                    return False
        raw = Path(value["measurement_reference"]["path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        if work["evidence"] != json.loads((raw / "primitive_evidence.json").read_bytes()):
            return False
        if work["evidence"]:
            refs = {r["upstream_path"]: r for r in work["refs"]}
            if any(
                ref["sha256"] != PINS[str(Path(ref["upstream_path"]).relative_to(ROOT))]
                for ref in work["refs"]
                if ref["upstream_path"] in {str(ROOT / n) for n in PINS}
            ):
                return False
            source = json.loads(Path(refs[str(ROOT / UPSTREAM)]["path"]).read_bytes())
            parameters_ref = refs[source["fitted_heads_path"]]
            public_ref = refs[str(ROOT / PUBLIC)]
            if (
                parameters_ref["sha256"] != source["fitted_heads_sha256"]
                or work["evidence"]["parameters"]
                != json.loads(Path(parameters_ref["path"]).read_bytes())
                or work["evidence"]["public"] != json.loads(Path(public_ref["path"]).read_bytes())
                or work["evidence"]["comparator"] != source["frozen_comparator"]
            ):
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
            code_config_hashes=work["code_config_hashes"],
            parameters_sha256=work["evidence"].get("parameters_sha256"),
            permission_mask_sha256=work["diagnostics"].get("permission_mask_sha256"),
        ):
            return False
        return build(
            work, raw, value["validation_receipts"], fixture=value["fixture_protocol_only"]
        ) == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Reuse qualified E2E and supervision commands with this task's coverage scope."""
    with (
        patch.object(upstream, "OWNED", OWNED),
        patch.object(upstream, "TEST", TEST),
        patch.object(upstream, "CLI", CLI),
        patch.object(execution, "e", sys.modules[__name__]),
    ):
        return upstream.manifest(private, candidate)


def main(argv: list[str] | None = None) -> int:
    """The existing supervisor provides bounded children and atomic terminal publication."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(argv))
