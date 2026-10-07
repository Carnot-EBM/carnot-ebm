"""REQ-REPORT-8222: bind numerical corrections to actual checks and sealed inputs.

Execution readiness grants permission for downstream evaluation. It does not
measure useful decisions or independent generalization.
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
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v686_contract_validation import dependency_hashes
from carnot.verify import utility_fit_8222 as numeric
from carnot.verify import utility_patch_methods_8219 as frozen

Json = dict[str, Any]
ROOT = frozen.ROOT
NAME = "experiment_8222_v711_utility_fit"
TASK = "exp8222-utility-fit"
MODULE = "python/carnot/verify/utility_fit_execution_8222.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_utility_fit_8222.py"
OWNED = ["python/carnot/verify/utility_fit_8222.py", MODULE, CLI]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
PROTOCOL_VALUE = frozen.PROTOCOL_VALUE
run_check = frozen.run_check
reference = frozen.reference
BASE_MANIFEST = execution.manifest


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so the parent can distinguish fitting from a stalled child."""
    print(f"[exp8222] phase={phase} completed={completed} pending={pending}", flush=True)


def primary(work: Json, root: Path, raw: Path, name: str, field: str) -> Json:
    """A hash-bound terminal report authenticates current upstream readiness."""
    path = root / "results" / (name + ".json")
    frozen.gate(work, path, "exists", True, True if path.is_file() else None)
    copied = frozen.bind(work, path, sha256_file(path), raw)
    value: Json = json.loads(copied.read_bytes())
    for key, expected in [
        (field, 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        frozen.gate(work, path, key, expected, value.get(key))
    terminal_path = root / Path(value["terminal_validation_sidecar_path"]).relative_to(ROOT)
    copied = frozen.bind(work, terminal_path, sha256_file(terminal_path), raw)
    terminal = json.loads(copied.read_bytes())
    sidecar = root / Path(terminal["publication"]["sidecar_path"]).relative_to(ROOT)
    frozen.bind(work, sidecar, sha256_file(sidecar), raw)
    frozen.gate(
        work,
        path,
        "terminal_publication_passed",
        True,
        read_bound_sidecar(path, sidecar)["report"]["passed"],
    )
    return value


def fitted_heads(work: Json) -> Json:
    """Bind downstream lookup trees to the same authenticated heads and role manifests."""
    return dict(
        base_heads=work["evidence"].get("heads", []),
        baseline=work["evidence"].get("baseline"),
        models=work["diagnostics"].get("selected_models", {}),
        fitted_models=work["models"],
        role_hashes=numeric.role_hashes(work["evidence"]) if work["evidence"] else {},
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Authenticate named operands before reading permitted labels or fitting deltas."""
    start, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        models={},
        fit_costs=[],
        diagnostics={},
        owned_failure="",
        protocol=PROTOCOL_VALUE,
        precondition_receipts=[],
    )
    progress("before_preconditions")
    try:
        with TemporaryDirectory(prefix="carnot-8222-probe-") as directory:
            probe = Path(directory) / "probe"
            probe.write_bytes(b"private writable scratch")
            frozen.gate(
                work,
                probe,
                "private_scratch_writable",
                True,
                probe.read_bytes() == b"private writable scratch",
            )
        frozen.gate(
            work, Path(sys.executable), "python_supported", True, sys.version_info >= (3, 11)
        )
        frozen.bind(work, root / frozen.PROTOCOL, frozen.PIN, raw)
        current = primary(
            work, root, raw, "experiment_8221_v711_utility_kernel", "static_kernel_ready_score"
        )
        protocol_primary = primary(
            work,
            root,
            raw,
            "experiment_8219_v710_utility_patch_methods",
            "utility_protocol_ready_score",
        )
        frozen.gate(
            work,
            root / frozen.PROTOCOL,
            "protocol_sha256",
            frozen.PIN,
            protocol_primary.get("protocol_sha256"),
        )
        for ref in current["kernel_code_snapshots"]:
            source = root / Path(ref["path"]).relative_to(ROOT)
            frozen.bind(work, source, ref["sha256"], raw)
        copies = {}
        for operand in [
            PROTOCOL_VALUE["fit_measurement"],
            PROTOCOL_VALUE["sealed_predictions"],
            next(
                r
                for r in PROTOCOL_VALUE["source_artifact_hashes"]
                if Path(r["path"]).name == "experiment_8208_v709_restricted_energy_fit.json"
            ),
        ]:
            source = root / Path(operand["path"]).relative_to(ROOT)
            copies[operand["path"]] = frozen.bind(work, source, operand["sha256"], raw)
        v709 = primary(
            work, root, raw, "experiment_8208_v709_restricted_energy_fit", "action_fit_ready_score"
        )
        evidence = json.loads(copies[PROTOCOL_VALUE["fit_measurement"]["path"]].read_bytes())[
            "evidence"
        ]
        sealed = json.loads(copies[PROTOCOL_VALUE["sealed_predictions"]["path"]].read_bytes())
        frozen.gate(
            work,
            copies[PROTOCOL_VALUE["sealed_predictions"]["path"]],
            "labels_opened",
            False,
            sealed.get("labels_opened"),
        )
        frozen.gate(work, raw, "role_manifest", PROTOCOL_VALUE["role_manifest"], evidence["roles"])
        heads_path = root / Path(v709["frozen_heads_path"]).relative_to(ROOT)
        heads_copy = frozen.bind(work, heads_path, v709["frozen_heads_sha256"], raw)
        frozen.gate(
            work,
            heads_path,
            "base_heads",
            json.loads(heads_copy.read_bytes())["heads"],
            evidence["heads"],
        )
        mask = [
            {
                k: r[k]
                for k in ["unit_id", "source_cluster_id", "slot", "status", "exclusion_reason"]
            }
            | dict(y=None, original_status=r["status"])
            for r in sealed["rows"]
            if r["arm"] == "energy"
        ]
        frozen.gate(
            work,
            raw,
            "reserved_identity_set",
            sorted(r["unit_id"] for r in evidence["roles"]["reserved"]),
            sorted(r["unit_id"] for r in mask),
        )
        if mutation:
            frozen.gate(work, root, "source_custody", "authenticated", None)
        work["evidence"] = {k: evidence[k] for k in ["rows", "roles", "heads", "baseline"]} | dict(
            reserved_mask=mask
        )
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
            progress("before_benchmark_fitting")
            work["models"], work["fit_costs"] = numeric.fit(work["evidence"], raw / "arms")
            progress("after_benchmark_fitting", 72)
            progress("before_benchmark_selection")
            work["diagnostics"] = numeric.reduce(work["evidence"], work["models"])
            progress("after_benchmark_selection", 72)
        except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
            work["owned_failure"] = str(error)
    work["code_config_hashes"] = []
    for name, digest in dependency_hashes(ROOT, paths=[*OWNED, TEST]).items():
        snapshot = frozen.bind(dict(checks=[], refs=[]), ROOT / name, digest, raw)
        work["code_config_hashes"].append(dict(reference(ROOT / name), snapshot_path=str(snapshot)))
    atomic_json(raw / "fixture_primitives.json", work["evidence"])
    atomic_json(raw / "fitted_heads.json", fitted_heads(work))
    work["raw_shard_hashes"] = [
        reference(p)
        for p in [
            raw / "fixture_primitives.json",
            raw / "fitted_heads.json",
            *sorted((raw / "arms").glob("*.json")),
        ]
    ]
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    work["clock"] = dict(
        started_monotonic_ns=start, ended_monotonic_ns=time.monotonic_ns(), started_wall_ns=wall
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["models"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """A complete zero-depth fit is ready; H1 remains for the independent audit."""
    value = frozen.build(work, raw, receipts, fixture=False)
    result = work["diagnostics"]
    rows = result.get("rows", [])
    ready = value.pop("utility_protocol_ready_score")
    value.update(
        experiment_id=8222,
        task_id=TASK,
        milestone="2026.10.711",
        honest_verdict=value["honest_verdict"].replace("utility_patch_methods", "utility_fit"),
        inference_substrate="verifier_ensemble_against_cached_candidates",
        utility_fit_ready_score=ready,
        rows=rows or value["rows"],
        fitted_heads_path=str(raw / "fitted_heads.json"),
        fitted_heads_sha256=sha256_file(raw / "fitted_heads.json"),
        frozen_comparator=result.get("frozen_comparator"),
        selected_depths=result.get("selected_depths", {}),
        fit_tune_role_hashes=numeric.role_hashes(work["evidence"]) if work["evidence"] else {},
        utility_residuals=result.get("utility_residuals", {}),
        fit_costs=work["fit_costs"],
        depth_selection=result.get("depth_grid", {}),
        trained_head_specs=[
            dict(
                arm=a,
                seed=s,
                kind="finite_probability_patch",
                base_head_refitted=False,
                current_fit=True,
                fit_role="head_fit",
                depth=result.get("selected_depths", {}).get(a + ":" + (str(s) if s else "none")),
            )
            for a, s in numeric.SPECS
        ],
        primary_comparator_selected=result.get("frozen_comparator"),
        mandatory_group_controls=result.get("mandatory_group_controls", []),
        random_seed=101,
        fixture_protocol_only=fixture,
        claim_scope="Matched fitting and calibration on exposed development data; H1 benefit remains unmeasured.",
        methodology_note="No generator loads or calls. Authenticated V709 base heads and temperatures remain fixed. Only head_fit labels fit finite probability deltas; calibration selects depth and the exact eligible comparator. Reserved and retention labels stay closed. Random seeds add no independent sources. Fit readiness measures reproducibility, not useful decisions or independent generalization.",
    )
    if rows:
        value.update(
            intended_count=len(rows),
            completed_count=sum(r["status"] == "completed" for r in rows),
            excluded_count=sum(r["status"] == "excluded" for r in rows),
            independent_count=len({r["source_cluster_id"] for r in rows}),
        )
    value["cited_upstream_artifacts"] = [
        dict(
            path=r["upstream_path"],
            sha256=r["sha256"],
            fields_imported=[
                "authenticated named input bytes; role rows, fixed heads, protocol or current kernel readiness"
            ],
        )
        for r in work["refs"]
    ]
    value["field_principles"] = {
        k: "Bind fitting to actual evidence; readiness supplies no benefit or generalization credit."
        for k in value
    }
    value["field_principles"].update(
        utility_fit_ready_score="Complete fitting and calibration, including zero-depth choices, require passing owned checks.",
        trained_head_specs="Current corrections are separate from frozen base heads and from absent LLM invocations.",
        frozen_comparator="Only the frozen eligible list and original calibration rows select the comparator.",
        selected_depths="Calibration chooses depth before reserved targets are opened.",
        fit_tune_role_hashes="Exact original manifests prevent moving missing sources or labels between roles.",
        utility_residuals="All-slot finite witness residuals describe calibration; H1 is unmeasured.",
        fit_costs="Actual monotonic arm clocks measure numerical work; checkpoints retain previous completed costs.",
    )
    value.pop("reproducibility_checksum", None)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Fresh arithmetic refits primitive rows and rejects rehashed invented aggregates."""
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
        if work["evidence"] != json.loads((raw / "fixture_primitives.json").read_bytes()):
            return False
        if fitted_heads(work) != json.loads((raw / "fitted_heads.json").read_bytes()):
            return False
        if work["models"]:
            source = next(
                r
                for r in work["refs"]
                if r["upstream_path"] == PROTOCOL_VALUE["fit_measurement"]["path"]
            )
            upstream = json.loads(Path(source["path"]).read_bytes())["evidence"]
            if (
                source["sha256"] != PROTOCOL_VALUE["fit_measurement"]["sha256"]
                or work["protocol"] != PROTOCOL_VALUE
                or any(
                    work["evidence"][k] != upstream[k]
                    for k in ["rows", "roles", "heads", "baseline"]
                )
            ):
                return False
            with TemporaryDirectory(prefix="carnot-8222-replay-") as directory:
                models, _ = numeric.fit(work["evidence"], Path(directory))
            if (
                models != work["models"]
                or numeric.reduce(work["evidence"], models) != work["diagnostics"]
            ):
                return False
        return build(
            work, raw, value["validation_receipts"], fixture=value["fixture_protocol_only"]
        ) == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError, StopIteration, TimeoutError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze exact checks while limiting measured coverage to newly owned code."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "[report]\nexclude_lines =\n")
    for spec in specs["commands"]:
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["argv"].append("--basetemp=" + str(private / "owned_pytest"))
            spec["deadline_s"] = 300
        if spec["name"] == "consumer_and_E2E015_019":
            begin = spec["argv"].index("tests/python/test_development_methods_8098.py")
            spec["argv"][begin:] = [
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
                "--basetemp=" + str(private / "consumer_pytest"),
            ]
            spec["deadline_s"] = 240
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
    """The qualified supervisor owns normal child exit, deadlines and atomic publication."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(argv))
