"""REQ-REPORT-8224: authenticate a current seal before exposing evaluator targets.

Replay uses immutable producer snapshots. Old code and unavailable historical
primaries supply provenance, never an archive-wide prerequisite for this audit.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting import v709_execution as supervisor
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import utility_audit_8224 as numeric
from carnot.verify import utility_fit_execution_8222 as upstream
from carnot.verify import utility_patch_methods_8219 as frozen

Json = dict[str, Any]
ROOT = frozen.ROOT
NAME = "experiment_8224_v711_utility_audit"
TASK = "exp8224-utility-audit"
MODULE = "python/carnot/verify/utility_audit_execution_8224.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_utility_audit_8224.py"
OWNED = ["python/carnot/verify/utility_audit_8224.py", MODULE, CLI]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8223_v711_utility_seal.json"
LABELS = "results/raw/experiment_8185_v707_sentence_decision_audit/invocations/1791261147226658146/measurement.json"
PINS = {
    UPSTREAM: "sha256:d04b88a905c6c38efc36894bd4699d3cc0327a7b81ae2c36cc0c77e41e2fac67",
    LABELS: "sha256:08d8a8efe35c0c44e49ee14e761264f20b70b1b665e1d706b20bb94047d81e44",
}
reference = frozen.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries and real counts so cached reduction never looks stalled."""
    print(f"[exp8224] phase={phase} completed={completed} pending={pending}", flush=True)


def run_check(
    root: Path, spec: Json, private: Path, durable: Path, *, heartbeat_s: float = 20
) -> Json:
    """The qualified supervisor bounds process groups and retains separate full streams."""
    cwd = private if spec["name"] == "cold_replay" else root
    with patch.object(supervisor, "ROOT", cwd):
        receipt = supervisor.child(
            spec["name"],
            spec["argv"],
            durable,
            deadline=spec["deadline_s"],
            expected=spec["expected_exit"],
            heartbeat=heartbeat_s,
            scope=spec.get("classification", "required"),
        )
    return dict(receipt, cwd=str(cwd))


def inputs(work: Json, root: Path, raw: Path) -> Json:
    """Authenticate saved current predictions fully before the first evaluator read."""

    def read(path: Path, digest: str) -> Json:
        return dict(json.loads(frozen.bind(work, path, digest, raw).read_bytes()))

    read(root / frozen.PROTOCOL, frozen.PIN)
    value = read(root / UPSTREAM, PINS[UPSTREAM])
    for field, expected in [
        ("utility_predictions_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
        ("labels_opened", False),
    ]:
        frozen.gate(work, root / UPSTREAM, field, expected, value.get(field))
    terminal = read(
        Path(value["terminal_validation_sidecar_path"]),
        sha256_file(Path(value["terminal_validation_sidecar_path"])),
    )
    sidecar = Path(terminal["publication"]["sidecar_path"])
    read(sidecar, sha256_file(sidecar))
    frozen.gate(
        work,
        root / UPSTREAM,
        "terminal_publication_passed",
        True,
        read_bound_sidecar(root / UPSTREAM, sidecar)["report"]["passed"],
    )
    for ref in value["source_artifact_hashes"]:
        read(Path(ref["path"]), ref["sha256"])
    for ref in value["code_config_hashes"]:
        frozen.bind(work, Path(ref["snapshot_path"]), ref["sha256"], raw)
    saved = read(
        Path(value["measurement_reference"]["path"]), value["measurement_reference"]["sha256"]
    )
    predictions = read(Path(value["sealed_predictions_path"]), value["prediction_sha256"])
    frozen.gate(
        work,
        Path(value["sealed_predictions_path"]),
        "labels_opened",
        False,
        predictions["labels_opened"],
    )
    sealed = saved["evidence"]
    progress("before_benchmark_authenticate_seal", 0, 128)
    frozen.gate(
        work,
        raw,
        "sealed_prediction_reduction",
        True,
        numeric.seal.reduce(sealed)["prediction_rows"] == predictions["rows"],
    )
    progress("after_benchmark_authenticate_seal", 128)
    frozen.gate(work, raw, "comparator", value["frozen_comparator"], sealed["comparator"])
    clock = dict(predictions_sealed_ns=time.monotonic_ns())
    progress("predictions_authenticated_before_label_access", len(predictions["rows"]))
    clock["labels_opened_ns"] = time.monotonic_ns()
    labels = read(root / LABELS, PINS[LABELS])["evidence"]
    return dict(
        sealed=sealed,
        predictions=predictions["rows"],
        comparator=value["frozen_comparator"],
        slots=labels["capture"]["slots"],
        original_response_records=labels["original_response_records"],
        clock=clock,
        protocol=numeric.PROTOCOL,
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """External operand failures stay blocked; owned arithmetic failures disqualify."""
    start, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        diagnostics={},
        owned_failure="",
        protocol=numeric.PROTOCOL,
        precondition_receipts=[],
    )
    progress("before_preconditions")
    try:
        with TemporaryDirectory(prefix="carnot-8224-probe-") as directory:
            probe = Path(directory) / "probe"
            probe.write_bytes(b"private writable scratch")
            frozen.gate(
                work,
                probe,
                "private_scratch_writable",
                True,
                probe.read_bytes() == b"private writable scratch",
            )
        receipt = run_check(
            ROOT,
            dict(
                name="python_environment",
                argv=[
                    sys.executable,
                    "-c",
                    "import pytest,coverage,numpy,scipy; print('cached audit resources available')",
                ],
                deadline_s=30,
                expected_exit=0,
            ),
            raw,
            raw / "preconditions",
        )
        work["precondition_receipts"].append(receipt)
        frozen.gate(
            work, Path(sys.executable), "python_environment_exit", 0, receipt["actual_exit"]
        )
        if mutation:
            frozen.gate(work, root / UPSTREAM, "source_custody", "authenticated", None)
        work["evidence"] = inputs(work, root, raw)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="authenticated_input_schema",
                    path=str(root / UPSTREAM),
                    upstream=str(root / UPSTREAM),
                    hash=PINS[UPSTREAM],
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
            progress("before_benchmark_independent_costs", 0, 128)
            work["diagnostics"] = numeric.reduce(work["evidence"])
            progress("after_benchmark_independent_costs", 128)
        except (ValueError, KeyError, TypeError) as error:
            work["owned_failure"] = str(error)
    work["code_config_hashes"] = []
    for path in [
        ROOT / p
        for p in [
            *OWNED,
            TEST,
            frozen.PROTOCOL,
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/verify/utility_seal_8223.py",
            "python/carnot/verify/restricted_decision_rule_8210.py",
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "ops/e2e-test-plan.md",
            "ops/exclusion_manifest.yaml",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "scripts/experiment_template.py",
        ]
    ]:
        snapshot = frozen.bind(dict(checks=[], refs=[]), path, sha256_file(path), raw)
        work["code_config_hashes"].append(dict(reference(path), snapshot_path=str(snapshot)))
    for name, content in [
        ("primitive_evidence", work["evidence"]),
        ("independent_reduction", work["diagnostics"]),
    ]:
        atomic_json(raw / (name + ".json"), content)
    work["raw_shard_hashes"] = [
        reference(raw / (n + ".json")) for n in ["primitive_evidence", "independent_reduction"]
    ]
    end = time.monotonic_ns()
    work.update(
        duration_s=(end - start) / 1e9,
        clock=dict(started_monotonic_ns=start, ended_monotonic_ns=end, started_wall_ns=wall),
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """A ready null has valid execution; readiness cannot grant scientific credit."""
    value = frozen.build(work, raw, receipts, fixture=False)
    value.pop("descriptive_utility_residuals")
    ready = value.pop("utility_protocol_ready_score")
    result = work["diagnostics"] or numeric.statistics([], dict(arm="energy_global"))
    signal = ready * result["h1_development_signal_score"]
    verdict = value["verdict_class"] if not ready else "positive" if signal else "null"
    failures = [c for c in work["checks"] if not c["passed"]]
    operand = (
        (Path(failures[0]["path"]).stem.lower() + "_" + failures[0]["artifact_field"])
        if failures
        else "utility_audit"
    )
    value.update(result)
    value.update(
        experiment_id=8224,
        task_id=TASK,
        milestone="2026.10.711",
        run_date=RUN_DATE,
        honest_verdict="complete_" + verdict + "_" + operand,
        verdict_class=verdict,
        utility_audit_ready_score=ready,
        random_seed=numeric.H1["seed"],
        h1_development_signal_score=signal,
        frozen_comparator=work["evidence"].get("comparator"),
        fixture_protocol_only=fixture,
        verifier_is_oracle=False,
        exposure_scope="exposed_development_within_run_prediction_label_boundary",
        trained_head_specs=[
            dict(arm=a, seed=s, current_fit=False, kind="frozen_probability_patch")
            for a, s in numeric.seal.fitted.SPECS
        ],
        methodology_note="No model loads or current LLM calls. Independently join original human annotations to authenticated corrected predictions. Every original slot carries cost, including unavailable escalation. Source-cluster H1 uses the unchanged frozen comparator and equally group-patched simple controls. Historical Qwen is provenance only. Reused development data cannot establish independent generalization.",
        claim_scope="Corrected decision utility on exposed development sources only.",
        acceptance_gates=dict(
            external_inputs=not failures,
            owned_validation=bool(value["required_checks_passed"]),
            scientific_H1=bool(result["H1"]["passed"]),
        ),
    )
    if work["diagnostics"]:
        for key in [
            "intended_count",
            "completed_count",
            "failed_count",
            "excluded_count",
            "censored_count",
            "independent_count",
        ]:
            value[key] = result[key]
    imported_fields = {
        str(ROOT / UPSTREAM): [
            "utility_predictions_ready_score",
            "required_checks_passed",
            "flagged_adversarial",
            "labels_opened",
            "measurement_reference",
            "sealed_predictions_path",
            "prediction_sha256",
            "frozen_comparator",
            "terminal_validation_sidecar_path",
            "source_artifact_hashes",
            "code_config_hashes",
        ],
        str(ROOT / LABELS): ["evidence.capture.slots", "evidence.original_response_records"],
    }
    value["cited_upstream_artifacts"] = [
        dict(
            path=r["upstream_path"],
            sha256=r["sha256"],
            fields_imported=imported_fields.get(
                r["upstream_path"], ["authenticated_byte_snapshot"]
            ),
        )
        for r in work["refs"]
    ]
    value["changed_code_coverage_reference"] = [
        reference(p)
        for p in [
            raw.parents[1] / "owned_coverage.json",
            raw.parents[1] / "frozen_coverage_config.ini",
        ]
        if p.is_file()
    ]
    value["field_principles"] = {
        k: "Bind actual invocation, original source units, exact inputs, denominators and measured validation; readiness grants no benefit."
        for k in value
    }
    value["field_principles"].update(
        utility_audit_ready_score="One means all owned checks pass and every source is accounted for; null benefit is allowed.",
        h1_development_signal_score="One requires readiness and every unchanged registered H1 and control safeguard.",
        H1="Only energy_group versus the tune-frozen primary comparator can establish the development signal.",
        per_source_deltas="One paired original source with both costs and missing status; repeated arms or seeds add no sources.",
        calibration_and_cost_comparisons="All-slot cost and complete-case Brier preserve explicit numerators and distinct denominators.",
        shared_calibration_effect="Each family's corrected versus original cost movement is descriptive and cannot replace H1.",
        energy_specific_advantage="Equally group-patched additive and logistic costs constrain energy-specific interpretation.",
        bootstrap_diagnostics="Original source clusters are sampled together10000 times with the protocol seed and one-sided alpha0.025.",
        comparator_sha256="Canonical bytes bind the selected arm, depth and model before evaluator access.",
    )
    value.pop("reproducibility_checksum", None)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild costs from pinned copies; rehashing an edited aggregate cannot authorize it."""
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
            *value["changed_code_coverage_reference"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"] or (
                "snapshot_path" in ref and sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]
            ):
                return False
        for receipt in [*value["validation_receipts"], *value["precondition_receipts"]]:
            for label in ["stdout", "stderr"]:
                if (
                    label + "_path" in receipt
                    and sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]
                ):
                    return False
        raw = Path(value["measurement_reference"]["path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        if work["evidence"] != json.loads((raw / "primitive_evidence.json").read_bytes()) or work[
            "diagnostics"
        ] != json.loads((raw / "independent_reduction.json").read_bytes()):
            return False
        if work["evidence"]:
            refs = {r["upstream_path"]: r for r in work["refs"]}

            def load(name: str, digest: str) -> Json:
                ref = refs[name]
                if ref["sha256"] != digest:
                    raise ValueError("pinned_input_changed")
                return dict(json.loads(Path(ref["path"]).read_bytes()))

            source = load(str(ROOT / UPSTREAM), PINS[UPSTREAM])
            labels = load(str(ROOT / LABELS), PINS[LABELS])["evidence"]
            saved = load(
                source["measurement_reference"]["path"], source["measurement_reference"]["sha256"]
            )
            predictions = load(source["sealed_predictions_path"], source["prediction_sha256"])
            data = work["evidence"]
            if (
                data["sealed"] != saved["evidence"]
                or data["predictions"] != predictions["rows"]
                or data["comparator"] != source["frozen_comparator"]
                or data["slots"] != labels["capture"]["slots"]
                or data["original_response_records"] != labels["original_response_records"]
            ):
                return False
            if not work["owned_failure"] and numeric.reduce(data) != work["diagnostics"]:
                return False
        return build(
            work, raw, value["validation_receipts"], fixture=value["fixture_protocol_only"]
        ) == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze this audit CLI and scope, restoring all upstream bindings afterward."""
    with (
        patch.object(upstream, "OWNED", OWNED),
        patch.object(upstream, "TEST", TEST),
        patch.object(upstream, "CLI", CLI),
        patch.object(execution, "e", sys.modules[__name__]),
    ):
        specs = upstream.manifest(private, candidate)
    (candidate.parent / "frozen_coverage_config.ini").write_bytes(
        (private / "coverage.ini").read_bytes()
    )
    for spec in specs["commands"]:
        if spec["name"] == "consumer_and_E2E015_019":
            spec["argv"].insert(-1, "tests/python/test_restricted_decision_audit_8210.py")
            spec["argv"].insert(-1, "tests/python/test_utility_seal_8223.py")
        if spec["name"] == "coverage_json":
            spec["argv"][-1] = str(candidate.parent / "owned_coverage.json")
    unit = next(s for s in specs["commands"] if s["name"] == "owned_unit_and_private_CLI")
    custody = deepcopy(unit)
    unit["argv"].extend(["-k", "original_labels or every_registered or actual_cli"])
    custody["name"] = "owned_custody_failure_batch"
    custody["argv"][-1] = "--basetemp=" + str(private / "custody_pytest")
    custody["argv"].extend(["-k", "owned_and_external or replay_all or missing_labels"])
    specs["commands"].insert(1, custody)
    for spec in specs["terminal_commands"]:
        if spec["name"] == "cold_replay":
            spec["argv"][:0] = ["/usr/bin/env", "-u", "PYTHONPATH"]
    return specs


def main(argv: list[str] | None = None) -> int:
    """The qualified runner validates normally exited private work before atomic publication."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(argv))
